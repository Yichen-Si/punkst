#include "concurrent_tree.hpp"
#include "numerical_utils.hpp"
#include "error.hpp"
#include <cassert>

ConcurrentTree::ConcurrentTree(int L, double gamma, int max_n_nodes,
    int _thr_heavy, int _thr_prune, int _branching_factor, int _debug)
    : L(L), max_n_nodes(max_n_nodes), thr_heavy(_thr_heavy), thr_prune(_thr_prune), branching_factor(_branching_factor), debug(_debug), nodes(max_n_nodes) {
    std::vector<double> log_gamma(L);
    for (int l = 0; l < L; l++)
        log_gamma[l] = std::log(gamma);
    init();
}
ConcurrentTree::ConcurrentTree(int L, std::vector<double> log_gamma,
    int max_n_nodes, int _thr_heavy, int _thr_prune, int _branching_factor, int _debug) : L(L), log_gamma(log_gamma), max_n_nodes(max_n_nodes), thr_heavy(_thr_heavy), thr_prune(_thr_prune), branching_factor(_branching_factor), debug(_debug), nodes(max_n_nodes) {
    assert(log_gamma.size() == L);
    init();
}

void ConcurrentTree::init() {
    max_id = 1;
    num_heavy.resize(L);
    std::fill(num_heavy.begin(), num_heavy.end(), 0);
    node_ids.resize(L);
    num_nodes.resize(L);
    std::fill(num_nodes.begin(), num_nodes.end(), 0);
    num_nodes[0] = 1;
    auto &root = nodes[0];
    root.parent_id = -1;
}

bool ConcurrentTree::IsLeaf(int node_id) {
    return nodes[node_id].depth + 1 == L;
}

bool ConcurrentTree::Exist(int node_id) {
    return (node_id == 0) || (nodes[node_id].depth);
}

void ConcurrentTree::DecNumDocs(int node_id) {
    if (!IsLeaf(node_id))
        error("%s receives non-leaf", __func__);
    while (node_id != -1) {
        auto &node = nodes[node_id];
        --node.num_docs;
        node_id = node.parent_id;
    }
}

ConcurrentTree::IncResult ConcurrentTree::IncNumDocs(int node_id, int delta) {
    if (!IsLeaf(node_id)) {
        node_id = AddNodes(node_id);
    }
    IncResult result(node_id, L);
    int l = L - 1;
    while (node_id != -1) { // trace the path to root
        auto &node = nodes[node_id];
        result.pos[l] = node.pos;
        node.num_docs += delta;
        node_id = node.parent_id;
        l--;
    }
    return std::move(result);
}

ConcurrentTree::RetTree ConcurrentTree::GetTree() {
    // Copy nodes
    int current_max_id = max_id; // freeze
    RetTree ret;
    ret.nodes.resize(current_max_id);
    ret.num_nodes.resize(L);
    std::fill(ret.num_nodes.begin(), ret.num_nodes.end(), 0);
    for (int i = 0; i < current_max_id; i++) {
        auto &node = nodes[i];
        ret.nodes[i] = RetNode{node.parent_id, node.pos,
            node.depth, 0, 0};
        if (node.depth + 1 == L) // only collect the num of docs for leaves
            ret.nodes[i].num_docs =
                node.num_docs.load(std::memory_order_relaxed);
    }

    // Calculate the actual num_docs for each internal node
    for (int i = current_max_id - 1; i >= 0; i--) { // bottom-up
        auto &node = ret.nodes[i];
        if (node.depth)
            ret.nodes[node.parent_id].num_docs += node.num_docs;
    }

    // Calculate log path weight & num_nodes for all nodes
    std::vector<float> log_num_docs(current_max_id, -1e9);
    log_num_docs[0] = logf(ret.nodes[0].num_docs);
    for (int i = 0; i < current_max_id; i++) { // top-down
        auto &node = ret.nodes[i];
        if (!Exist(i) || (node.num_docs == 0 && branching_factor == -1))
            node.log_path_weight = -1e9; // A nonexistent node
        else if (node.depth) {
            auto &parent = ret.nodes[node.parent_id];
            if (branching_factor == -1) {
                float current_l = logf(node.num_docs);
                log_num_docs[i]  = current_l;
                if (parent.num_docs > 0) {
                    float parent_l = log_num_docs[node.parent_id];
                    node.log_path_weight = current_l - parent_l + parent.log_path_weight;
                } else { // should not happen
                    error("%s: inconsistent num_docs", __func__);
                }
            } else { // CAUTIOUS: assume Instantiate() was called
                node.log_path_weight = nodes[i].log_weight +
                                       parent.log_path_weight;
            }
        }
        if (Exist(i)) {
            ret.num_nodes[node.depth] =
                std::max(ret.num_nodes[node.depth], node.pos + 1);
        }
    }

    // Step 4: add new-child probability for internal nodes
    for (int i = 0; i < current_max_id; i++) {
        auto &node = ret.nodes[i];
        if (node.depth + 1 < L) {
            if (node.num_docs > 0)
                node.log_path_weight += log_gamma[node.depth] - log_num_docs[i];
        }
    }

    return std::move(ret);
}

int ConcurrentTree::AddNodes(int root_id) {
    std::lock_guard<std::mutex> guard(mutex);
    while (nodes[root_id].depth + 1 < L) {
        auto &node = nodes[root_id];
        root_id = AddChildren(root_id);
    }
    return root_id; // return the leaf node id
}

void ConcurrentTree::SetThreshold(int thr_heavy, int thr_prune) {
    this->thr_heavy = thr_heavy;
    this->thr_prune = thr_prune;
}

void ConcurrentTree::SetBranchingFactor(int branching_factor) {
    this->branching_factor = branching_factor;
}

bool ConcurrentTree::Consolidate(std::vector<std::vector<int>>& pos_map) {
    pos_map.resize(L); // within-level index new -> old
    if (nodes[0].num_docs <= thr_prune) {
        notice("%s: (almost) empty tree, skip", __func__);
        return false;
    }
    std::lock_guard<std::mutex> guard(mutex);
    // Remove zero nodes; count heavy vs light nodes
    std::vector<std::vector<int>> node_ids_new(L); // stores the index in nodes
    std::vector<std::vector<int>> node_ids_light(L);
    int n_pruned = 0, first_prune = -1;
    std::vector<int> node_id_map(max_id, -1);
    int max_id_new = 0;
    std::set<int> node_pruned;
    for (size_t i = 0; i < max_id; i++) {
        if (!Exist(i) || nodes[i].num_docs <= thr_prune || node_pruned.count(i)) {
            if (n_pruned == 0) {
                first_prune = i;
            }
            n_pruned++;
            node_pruned.insert(i);
            if (Exist(i) && nodes[i].num_docs > 0) {
                nodes[nodes[i].parent_id].num_docs += nodes[i].num_docs;
            }
            nodes[i].parent_id = nodes[i].pos = nodes[i].depth = 0;
            continue;
        }
        node_id_map[i] = max_id_new++;
        if (nodes[i].num_docs >= thr_heavy) {
            node_ids_new[nodes[i].depth].push_back(i); // keep it stable
        } else {
            node_ids_light[nodes[i].depth].push_back(i);
        }
    }
    if (debug) {
        notice("%s: pruned %d out of %d nodes", __func__, n_pruned, max_id);
        for (int l = 0; l < L; l++) {
            std::cout << "Level " << l << ": "
                      << node_ids_new[l].size() << " heavy, "
                      << node_ids_light[l].size() << " light" << std::endl;
        }
    }

    for (int l = 0; l < L; l++) {
        num_heavy[l] = node_ids_new[l].size();
        node_ids_new[l].insert(node_ids_new[l].end(),
            node_ids_light[l].begin(), node_ids_light[l].end());
        uint32_t n = node_ids_new[l].size();
        num_nodes[l] = n;
        pos_map[l].resize(n);
        for (uint32_t i = 0; i < n; i++) {
            pos_map[l][i] = nodes[node_ids_new[l][i]].pos;
            nodes[node_ids_new[l][i]].pos = i;
        }
    }
    bool reordered = node_ids != node_ids_new;

    if (n_pruned > 0) {
        std::vector<Node> nodes_copy(max_id_new - first_prune);
        for (int i = first_prune; i < max_id; i++) {
            int j = node_id_map[i];
            if (j == -1) {
                continue;
            }
            nodes_copy[j-first_prune] = nodes[i];
            nodes_copy[j-first_prune].parent_id = node_id_map[nodes[i].parent_id];
        }
        for (int i = first_prune; i < max_id_new; i++) {
            int j = i - first_prune;
            nodes[i] = nodes_copy[j];
        }
        max_id = max_id_new;
        for (auto& v : node_ids_new) {
            for (auto &id : v) {
                id = node_id_map[id];
            }
        }
    }
    node_ids = std::move(node_ids_new);
    return reordered;
}

void ConcurrentTree::Instantiate() {
    // Gather the children for each parent
    std::vector<std::vector<int>> children(max_n_nodes);
    for (int i = 1; i < max_id; i++)
        if (Exist(i))
            children[nodes[i].parent_id].push_back(i);

    for (int i = 0; i < max_id; i++) {
        if (!Exist(i) || IsLeaf(i)) {
            continue;
        }
        // Sort the children by decreasing num_docs
        auto &ch = children[i];
        std::sort(ch.begin(), ch.end(), [&](int a, int b) {
                return nodes[a].num_docs > nodes[b].num_docs;
        });

        // Add new (empty) nodes if needed
        int num_empty = 0;
        for (auto c: ch) {
            if (nodes[c].num_docs == 0)
                num_empty++;
        }
        for (int j = num_empty; j < branching_factor; j++) {
            auto child = AddChildren(i);
            ch.push_back(child);
        }

        // tail weight: m_>i = \sum_{j>i} m_j
        std::vector<int> m_gt_i(ch.size());
        for (int n = (int)ch.size() - 2; n >= 0; n--)
            m_gt_i[n] = m_gt_i[n+1] + nodes[ch[n+1]].num_docs;

        // Compute stick-breaking weight (conditional prob: parent->child)
        // nodes[c].log_weight: \log\pi_i = \log v_i+\sum_{j<i}\log(1-v_j)
        double log_stick_length = 0; // \sum_{j<i} \log(E[1 - v_j])
        for (size_t n = 0; n < ch.size(); n++) {
            if (n + 1 == ch.size()) {
                nodes[ch[n]].log_weight = log_stick_length;
                break;
            }
            // Vi ~ Beta(1 + m_i, gamma + m_{>i})
            double a = 1.0 + nodes[ch[n]].num_docs;
            double b = exp(log_gamma[nodes[i].depth]) + m_gt_i[n];

            nodes[ch[n]].log_weight = log_stick_length + log(a) - log(a + b);
            log_stick_length += log(b) - log(a + b);
        }
    }
}

int ConcurrentTree::AddChildren(int parent_id) {
    auto &child = nodes[max_id++];
    child.parent_id = parent_id;
    child.depth = nodes[child.parent_id].depth + 1;
    child.pos = num_nodes[child.depth]++;
    child.num_docs.store(0);
    return max_id - 1;
}

std::ostream& operator << (std::ostream &out, const ConcurrentTree::RetTree &tree) {
    for (auto &node: tree.nodes) {
        out << " parent: " << node.parent_id
            << " pos: " << node.pos
            << " num_docs: " << node.num_docs
            << " depth: " << node.depth
            << " weight: " << node.log_path_weight << std::endl;
    }
    for (size_t l = 0; l < tree.num_nodes.size(); l++) {
        out << " num nodes " << tree.num_nodes[l] << std::endl;
    }
    return out;
}

std::ostream& operator << (std::ostream &out, const ConcurrentTree::IncResult &tree) {
    out << "ID: " << tree.id << " pos ";
    for (auto k: tree.pos)
        out << ' ' << k;
    return out;
}
