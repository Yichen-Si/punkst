#include "concurrent_tree.hpp"
#include "numerical_utils.hpp"
#include "error.hpp"
#include <cassert>

ConcurrentTree::ConcurrentTree(int _L, std::vector<double> _log_gamma,
    int _max_n_nodes, int _thr_heavy, int _thr_prune, int _debug)
    : L(_L), log_gamma(_log_gamma), max_n_nodes(_max_n_nodes), thr_heavy(_thr_heavy), thr_prune(_thr_prune), debug_(_debug), nodes(_max_n_nodes) {
    assert(log_gamma.size() > 0);
    if(log_gamma.size() < L) {
        log_gamma.reserve(L);
        for (int l = log_gamma.size(); l < L; l++) {
            log_gamma.push_back(log_gamma.back());
        }
    }
    gamma = std::vector<double>(L, 0.0);
    for (int l = 0; l < L; l++) {
        if (log_gamma[l] > -10) {
            gamma[l] = std::exp(log_gamma[l]);
        }
    }
    init();
    debug("%s: initialized", __func__);
}

void ConcurrentTree::init() {
    max_id = 1;
    n_heavy.resize(L);
    std::fill(n_heavy.begin(), n_heavy.end(), 0);
    node_ids.resize(L);
    n_nodes.resize(L);
    std::fill(n_nodes.begin(), n_nodes.end(), 0);
    n_nodes[0] = 1;
    auto &root = nodes[0];
    root.parent_id = -1;
    root.pos = 0;
    root.depth = 0;
}

void ConcurrentTree::SetGamma(std::vector<double> gamma) {
    assert(gamma.size() == L);
    for (int l = 0; l < L; l++) {
        log_gamma[l] = std::log(gamma[l]);
    }
}

void ConcurrentTree::SetLogGamma(std::vector<double> log_gamma) {
    assert(log_gamma.size() == L);
    this->log_gamma = std::move(log_gamma);
}

void ConcurrentTree::SetThreshold(int thr_heavy, int thr_prune) {
    this->thr_heavy = thr_heavy;
    this->thr_prune = thr_prune;
}

bool ConcurrentTree::IsLeaf(int node_id) const {
    return node_id >= 0 && node_id < max_id && nodes[node_id].depth + 1 == L;
}

bool ConcurrentTree::Exist(int node_id) const {
    return (node_id >= 0 && node_id < max_id && nodes[node_id].depth >= 0);
}

void ConcurrentTree::DecNumDocs(int node_id) {
    if (!Exist(node_id)) {
        return;
    }
    while (node_id >= 0) {
        auto &node = nodes[node_id];
        --node.n_docs;
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
        node.n_docs += delta;
        node_id = node.parent_id;
        l--;
    }
    return result;
}

ConcurrentTree::RetTree ConcurrentTree::GetTree() const {
    // Copy nodes
    int current_max_id = max_id; // freeze
    RetTree ret;
    ret.nodes.resize(current_max_id);
    ret.n_nodes.resize(L);
    std::fill(ret.n_nodes.begin(), ret.n_nodes.end(), 0);
    for (int i = 0; i < current_max_id; i++) {
        auto &node = nodes[i];
        ret.nodes[i] = RetNode{node.parent_id, node.pos, node.depth, 0, node.n_children.load(), 0};
        if (node.depth + 1 == L) // only collect the num of docs for leaves
            ret.nodes[i].n_docs =
                node.n_docs.load(std::memory_order_relaxed);
    }
    // Calculate the actual n_docs for each internal node
    for (int i = current_max_id - 1; i >= 0; i--) { // bottom-up
        auto &node = ret.nodes[i];
        if (node.depth)
            ret.nodes[node.parent_id].n_docs += node.n_docs;
    }
    // Calculate log path weight for all nodes
    std::vector<float> log_num_docs(current_max_id, -1e9);
    log_num_docs[0] = std::log(ret.nodes[0].n_docs + gamma[0]);
    for (int i = 0; i < current_max_id; i++) { // top-down
        auto &node = ret.nodes[i];
        if (!Exist(i) || node.n_docs <= 0) {
            node.log_path_weight = -1e9; // A nonexistent node
        } else if (node.depth) {
            auto &parent = ret.nodes[node.parent_id];
            log_num_docs[i] = std::log(node.n_docs + gamma[node.depth]);
            if (parent.n_docs > 0) {
                node.log_path_weight = parent.log_path_weight +
                    std::log(node.n_docs) - log_num_docs[node.parent_id];
            } else { // should not happen
                error("%s: parent %d has no docs", __func__, node.parent_id);
            }
        } else {
            node.log_path_weight = 0;
        }
        if (Exist(i)) {
            ret.n_nodes[node.depth] =
                std::max(ret.n_nodes[node.depth], node.pos + 1);
        }
    }
    // Add new-child probability for internal nodes
    for (int i = 0; i < current_max_id; i++) {
        auto &node = ret.nodes[i];
        if (node.depth + 1 < L) {
            if (node.n_docs > 0) {
                node.log_path_weight += log_gamma[node.depth] - log_num_docs[i];
            }
        }
    }
    return std::move(ret);
}

int ConcurrentTree::AddNodes(int root_id) {
    // if (debug_) {
    //     std::cerr << "ConcurrentTree::AddNodes: adding new path from "
    //         << root_id  << " at depth " << nodes[root_id].depth << std::endl;
    // }
    std::lock_guard<std::mutex> guard(mutex);
    while (nodes[root_id].depth + 1 < L) {
        auto &node = nodes[root_id];
        root_id = AddChildren(root_id);
    }
    // if (debug_) {
    //     std::cerr << "\tadded leaf: " << root_id << std::endl;
    // }
    return root_id; // return the leaf node id
}

bool ConcurrentTree::Consolidate(std::vector<NodeRemap>& pos_map) {
    pos_map.clear();
    if (nodes[0].n_docs <= thr_prune) {
        std::cerr << "ConcurrentTree::Consolidate: (almost) empty tree, skip" << std::endl;
        return false;
    }
    std::lock_guard<std::mutex> guard(mutex);

    const int old_max = max_id;

    id_old_to_survived_anc.resize(old_max);
    std::fill(id_old_to_survived_anc.begin(), id_old_to_survived_anc.end(), 0);
    pos_old2new.resize(L);
    for (int l = 0; l < L; ++l) {
        pos_old2new[l].resize(n_nodes[l]);
        std::fill(pos_old2new[l].begin(), pos_old2new[l].end(), -1);
    }

    std::vector<std::vector<int>> children(old_max);
    for (int v = 1; v < old_max; ++v)
        if (Exist(v))
            children[nodes[v].parent_id].push_back(v);
    std::vector<char> keep(old_max, 0);
    keep[0] = 1;
    std::vector<int> stack{0};
    while (!stack.empty()) {
        int u = stack.back(); stack.pop_back();
        // sort children[u] by decreasing n_docs
        std::sort(children[u].begin(), children[u].end(), [&](int a, int b) {
            return nodes[a].n_docs > nodes[b].n_docs;
        });
        size_t i = 0;
        int n_docs_kept = 0;
        while (i < children[u].size()) {
            int v = children[u][i];
            // Leave at least one child so all leaves are on the same level
            if (i == 0 || nodes[v].n_docs > thr_prune) {
                keep[v] = 1;
                if (nodes[v].depth + 1 < L) {
                    stack.push_back(v);
                }
                n_docs_kept += nodes[v].n_docs;
            } else {
                break;
            }
            i++;
        }
        // Re-distribute the pruned weights to the survived children
        int n_docs_pruned = nodes[u].n_docs - n_docs_kept;
        if (i == children[u].size() || n_docs_pruned <= 0) {
            continue;
        }
        i -= 1;
        while (i >= 0) {
            int v = children[u][i];
            if (i == 0) {
                nodes[v].n_docs += n_docs_pruned;
                break;
            }
            int delta = nodes[v].n_docs * n_docs_pruned / n_docs_kept;
            n_docs_kept -= nodes[v].n_docs;
            nodes[v].n_docs += delta;
            n_docs_pruned -= delta;
            if (n_docs_pruned <= 0) {
                break;
            }
            i--;
        }
    }
if (debug_) {
    std::cout << "Consolidate: " << old_max << " nodes, keep " << std::count(keep.begin(), keep.end(), 1) << std::endl;
}
    for (int v = 1; v < old_max; ++v) {
        int u = v;
        if (!Exist(v) || nodes[v].n_docs == 0) {
            continue;
        }
        // TODO: this caused a (small) problem later when we delete a doc
        // re-assigned to an internal node, since we do not subtract from
        // its descendants causing the total n_docs recorded in tree to grow
        while(!keep[u]) {
            u = nodes[u].parent_id;
        }
        id_old_to_survived_anc[v] = u;
    }

    std::vector<Node>              new_nodes;
    std::vector<std::vector<int>>  ids_by_level(L);
    struct Item { int old_id, parent_new, depth; };
    std::vector<Item> Q{{0, -1, 0}}; // root seed
    while (!Q.empty()) {
        auto [cur, par, depth] = Q.back(); Q.pop_back();
        Node nn(par, -1, depth, nodes[cur].n_docs);
        NodeRemap remap(depth, -1,
            std::vector<int>{nodes[cur].depth},
            std::vector<int>{nodes[cur].pos},
            std::vector<int>{cur});
        // while (cur != 0) { // Contract unary chains
        //     std::vector<int> kept_child;
        //     for (int c : children[cur]) if (keep[c]) kept_child.push_back(c);
        //     if (kept_child.size() != 1) break; // 0 or >1 -> stop
        //     cur = kept_child[0]; // only child -> skip current node
        //     remap.l_old.push_back(nodes[cur].depth);
        //     remap.k_old.push_back(nodes[cur].pos);
        // }
        // add to new_nodes
        const int new_id = static_cast<int>(new_nodes.size());
        new_nodes.push_back(nn);
        pos_map.push_back(remap);
        ids_by_level[depth].push_back(new_id);
        for (int c : children[cur]) if (keep[c])
            Q.push_back({c, new_id, depth + 1});
    }
    // Re-order the nodes by heavy/light
    n_nodes.assign(L, 0);
    n_heavy.assign(L, 0);
    node_ids.assign(L, {});
    bool remapped = false;
    std::vector<int> id_old2new(old_max, -1);
    for (int l = 0; l < L; ++l) {
        const auto& ids = ids_by_level[l]; // index in new_nodes
        const std::size_t n = ids.size();
        if (!n) break;
        std::vector<int> ids_re; ids_re.reserve(n); // index in new_nodes
        for (std::size_t i = 0; i < n; ++i)
            if (new_nodes[ids[i]].n_docs >= thr_heavy) {
                ids_re.push_back(ids[i]);
            }
        const int heavy_cnt = static_cast<int>(ids_re.size());
        for (std::size_t i = 0; i < n; ++i)
            if (new_nodes[ids[i]].n_docs < thr_heavy) {
                ids_re.push_back(ids[i]);
            }
        // update pos field + build pos_map
        node_ids[l] = ids_re;
        for (std::size_t i = 0; i < n; ++i) {
            int id = ids_re[i];
            pos_map[id].k_new = static_cast<int>(i);
            new_nodes[id].pos = static_cast<int>(i);
            remapped |= (pos_map[id].k_old.size() > 1 || pos_map[id].k_old[0] != i);
            for (auto v : pos_map[id].k_old) {
                pos_old2new[l][v] = i;
            }
            for (auto v : pos_map[id].id_old) {
                id_old2new[v] = id;
            }
        }
        n_nodes[l]  = static_cast<int>(n);
        n_heavy[l]  = heavy_cnt;
    }
    // update n_children
    for (auto& v : new_nodes) {
        v.n_children.store(0, std::memory_order_relaxed);
    }
    for (const auto& v : new_nodes) {
        if (v.parent_id >= 0) {
            new_nodes[v.parent_id].n_children++;
        }
    }
    // map node ids
    for (int i = 0; i < old_max; ++i) {
        if (id_old_to_survived_anc[i] != -1) {
            id_old_to_survived_anc[i] = id_old2new[id_old_to_survived_anc[i]];
        }
    }

if (debug_) {
    std::cout << "Consolidate: n_heavy/n_nodes ";
    for (int l = 0; l < L; ++l) {
        std::cout << n_heavy[l] << "/" << n_nodes[l] << " ";
    }
    std::cout << std::endl;
}
    nodes = std::move(new_nodes);
    max_id = static_cast<int>(nodes.size());

    leaf_to_path.clear();
    for (int i = 0; i < max_id; ++i) {
        if (IsLeaf(i)) {
            int cur = i;
            std::vector<int> path;
            while (cur != -1) {
                path.push_back(nodes[cur].pos);
                cur = nodes[cur].parent_id;
            }
            std::reverse(path.begin(), path.end());
            leaf_to_path[nodes[i].pos] = std::move(path);
        }
    }

    return remapped;
}

void ConcurrentTree::Instantiate() {
    // Gather the children for each parent
    int current_max_id = max_id; // freeze
    std::vector<std::vector<int>> children(current_max_id);
    for (int i = 1; i < current_max_id; i++)
        if (Exist(i))
            children[nodes[i].parent_id].push_back(i);

    for (int i = 0; i < current_max_id; i++) {
        if (!Exist(i) || IsLeaf(i)) {
            continue;
        }
        // Sort the children by decreasing n_docs
        auto &ch = children[i];
        std::sort(ch.begin(), ch.end(), [&](int a, int b) {
                return nodes[a].n_docs > nodes[b].n_docs;
        });
        // tail weight: m_>i = \sum_{j>i} m_j
        std::vector<int> m_gt_i(ch.size(), 0);
        for (int n = (int)ch.size() - 2; n >= 0; n--)
            m_gt_i[n] = m_gt_i[n+1] + nodes[ch[n+1]].n_docs;

        // Compute stick-breaking weight (conditional prob: parent->child)
        // nodes[c].log_weight: \log\pi_i = \log v_i+\sum_{j<i}\log(1-v_j)
        double log_stick_length = 0; // \sum_{j<i} \log(E[1 - v_j])
        for (size_t n = 0; n < ch.size(); n++) {
            if (n + 1 == ch.size()) {
                nodes[ch[n]].log_weight = log_stick_length;
                break;
            }
            // Vi ~ Beta(1 + m_i, gamma + m_{>i})
            double a = 1.0 + nodes[ch[n]].n_docs;
            double b = exp(log_gamma[nodes[i].depth]) + m_gt_i[n];

            nodes[ch[n]].log_weight = log_stick_length + log(a) - log(a + b);
            log_stick_length += log(b) - log(a + b);
        }
    }
}

int ConcurrentTree::AddChildren(int parent_id) {
    while (nodes.size() <= max_id + 1) {
        nodes.push_back(Node());
    }
    auto &child = nodes[max_id++];
    child.parent_id = parent_id;
    child.depth = nodes[child.parent_id].depth + 1;
    child.pos = n_nodes[child.depth]++;
    child.n_docs.store(0);
    // if (debug_) {
    //     std::cout << "\tAddChildren " << max_id-1 << ": parent_id = " << parent_id
    //             << ", child depth = " << child.depth
    //             << ", pos = " << child.pos << std::endl;
    // }
    return max_id - 1;
}

std::ostream& operator << (std::ostream &out, const ConcurrentTree::IncResult &tree) {
    out << "ID: " << tree.id << " pos ";
    for (auto k: tree.pos)
        out << ' ' << k;
    return out;
}

void printTree(const ConcurrentTree::RetTree &tree, std::ostream &out, bool leaf_only, bool marginal) {
    const auto& nodes = tree.nodes;
    const auto& n_nodes = tree.n_nodes;

    std::vector<int> node_idx(nodes.size());
    std::vector<int> node_depth(nodes.size());
    for (size_t i = 0; i < nodes.size(); ++i) {
        node_idx[i] = static_cast<int>(i);
        node_depth[i] = nodes[i].depth;
    }
    std::sort(node_idx.begin(), node_idx.end(), [&](int a, int b) {
        return node_depth[a] < node_depth[b];
    });

    if (!marginal && !leaf_only) {
        // Full tree print
        for (int i : node_idx) {
            const auto& node = nodes[i];
            if (node.depth + 1 == static_cast<int>(n_nodes.size())) {
                continue;
            }
            if (node.depth > 0) {
                out << "(" << nodes[node.parent_id].depth << ", " << nodes[node.parent_id].pos << ", " << nodes[node.parent_id].n_children << ")";
            } else {
                out << "-1";
            }
            out << " -> (" << node.depth << ", " << node.pos
                << ") n_docs: " << node.n_docs
                << " o_dg: " << node.n_children
                << " logPr: " << node.log_path_weight
                << std::endl;
        }
    }
    if (!marginal) {
        // Print leaf node info only
        for (int i : node_idx) {
            const auto& node = nodes[i];
            if (node.depth + 1 == static_cast<int>(n_nodes.size())) {
                if (node.depth > 0) {
                    out << "(" << nodes[node.parent_id].depth << ", " << nodes[node.parent_id].pos << ", " << nodes[node.parent_id].n_children << ")";
                } else {
                    out << "-1";
                }
                out << " -> (" << node.depth << ", " << node.pos
                    << ") n_docs: " << node.n_docs
                    << " logPr: " << node.log_path_weight
                    << std::endl;
            }
        }
    }
    out << "# Per-level node counts and out-degrees\n";
    for (size_t l = 0; l < n_nodes.size(); ++l) {
        out << "Level " << l << ", " << n_nodes[l] << " nodes\n";
        if (l < n_nodes.size()-1) {
            out << "\t";
            for (const auto& node : nodes) {
                if (node.depth == static_cast<int>(l)) {
                    out << " (" << node.pos << ", " << node.n_children << ")";
                }
            }
            out << std::endl;
        }
    }
}

void WriteTreeAsDot(const ConcurrentTree::RetTree& tree, const std::string& filename, const std::vector<std::vector<std::string>>* extra_labels, size_t words_per_line, int max_depth) {
    int L = tree.n_nodes.size();
    std::vector<int> offsets(L + 1, 0);
    for (int l = 0; l < L; ++l) {
        offsets[l + 1] = offsets[l] + tree.n_nodes[l];
    }
    if (max_depth < 0) {
        max_depth = L - 1;
    }

    std::ofstream out(filename);
    out << "digraph G {\n";
    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        const auto& node = tree.nodes[i];
        if (node.depth > max_depth) {
            continue;
        }
        std::ostringstream label;
                label << node.depth << "_" << node.pos
                    << " n_doc=" << node.n_docs;
        if (node.depth < tree.n_nodes.size() - 1)
            label << " n_child=" << node.n_children;
        if (extra_labels) {
            uint32_t idx = tree.nodes[i].pos + offsets[tree.nodes[i].depth];
            const auto& extras = (*extra_labels)[idx];
            label << "\\n";
            for (size_t j = 0; j < extras.size(); ++j) {
                label << extras[j];
                if ((j + 1) % words_per_line == 0 && (j + 1) < extras.size()) {
                    label << "\\n";
                } else if (j + 1 < extras.size()) {
                    label << ",";
                }
            }
        }
        out << "  n" << i << " [label=\"" << label.str() << "\"];\n";
        if (node.parent_id >= 0) {
            out << "  n" << node.parent_id << " -> n" << i << ";\n";
        }
    }
    out << "}\n";
}

void WriteTreeAsTSV(const ConcurrentTree::RetTree& tree,
                    const std::string& outpref) {

    std::ofstream edge_out(outpref + ".edges.tsv");
    std::ofstream node_out(outpref + ".nodes.tsv");

    edge_out << "source\ttarget\n";
    node_out << "node_id\tdepth\tpos\tn_docs\tn_children\n";

    for (size_t i = 0; i < tree.nodes.size(); ++i) {
        const auto& node = tree.nodes[i];
        node_out << i << '\t' << node.depth << '\t' << node.pos << '\t'
                 << node.n_docs << '\t' << node.n_children << '\n';

        if (node.parent_id >= 0) {
            edge_out << node.parent_id << '\t' << i << '\n';
        }
    }
}
