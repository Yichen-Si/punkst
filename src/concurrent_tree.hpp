# pragma once

#include <array>
#include <atomic>
#include <iostream>
#include <fstream>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <cstring>
#include <cassert>
#include <climits>

class ConcurrentTree {
public:

    struct Node {
        int parent_id = -1;
        int pos = -1; // index within the level
        int depth = -1;
        std::atomic<int> n_docs {0};
        std::atomic<int> n_children {0};
        double log_weight = 0;

        Node(int parent_id = -1, int pos = -1, int depth = -1, int n_docs = 0)
            : parent_id(parent_id), pos(pos), depth(depth), n_docs(n_docs), n_children(0) {}
        Node(Node&&) noexcept = default;
        Node(const Node& other)
            : parent_id(other.parent_id), pos(other.pos), depth(other.depth),
              n_docs(other.n_docs.load(std::memory_order_relaxed)),
              n_children(other.n_children.load(std::memory_order_relaxed)),
              log_weight(other.log_weight) {}
        Node& operator=(Node&&) noexcept = default;
        Node& operator=(const Node& other) {
            parent_id = other.parent_id; pos = other.pos; depth = other.depth;
            n_docs.store(other.n_docs.load());
            n_children.store(other.n_children.load());
            log_weight = other.log_weight;
            return *this;
        }
    };

    struct RetNode {
        int parent_id, pos, depth, n_docs, n_children;
        double log_path_weight;
    };

    struct RetTree {
        std::vector<RetNode> nodes;
        std::vector<int> n_nodes;
    };

    struct IncResult {
        int id;
        std::vector<int32_t> pos;
        IncResult(int id, int L) : id(id), pos(L) {}
    };

    struct NodeRemap {
        int l_new, k_new;
        std::vector<int> l_old, k_old, id_old;
    };

    ConcurrentTree() {}
    ConcurrentTree(int _L, std::vector<double> _log_gamma, int _max_n_nodes = 1024, int _thr_heavy = INT_MAX, int _thr_prune = 0, int _debug = 0);

    void init();

    // Lock-free operations
    bool IsLeaf(int node_id) const;
    bool Exist(int node_id) const;
        // Decrease the doc count by 1 for all nodes on node_id->root
    void DecNumDocs(int node_id);
        // Increase the doc count by delta for all nodes on node_id->root
        // If node_id is not a leaf, do AddNodes first (lock)
        // The returned path is always from root to a leaf (could be new)
    IncResult IncNumDocs(int node_id, int delta = 1);
        // Realize the tree with current nCRP probabilities
    RetTree GetTree() const;

    // Lock operations
        // The only place where a new node can be added
        // Start from root_id, add a new path (up to the leaf level)
    int AddNodes(int root_id);

    // Global operations (not meant to be called concurrently)
    void SetGamma(std::vector<double> gamma);
    void SetLogGamma(std::vector<double> log_gamma);
    void SetThreshold(int thr_heavy, int thr_prune = 0);
    void SetMaxChildren(std::vector<int> max_outdg) {
        assert(max_outdg.size() == L);
        this->max_outdg = std::move(max_outdg);
        force_degree = true;
    }
        // The only place where a node can be removed (prune empty/rare nodes)
        //                   or a node's within-level pos modified
    bool Consolidate(std::vector<NodeRemap>& pos_map);
        // Compute nCRP prob at each node given the current obs (counts)
        //     log(child | parent) (\pi), stored in nodes[i].log_weight
        //     (the proper stick breaking prob)
    void Instantiate();

    const std::vector<std::vector<int>>& GetNodeIdsC() const {return node_ids;}
    const std::vector<int>& GetNumHeavyC() const {return n_heavy;}
    const std::vector<int>& GetNumNodesC() const {return n_nodes;}
    std::vector<std::vector<int>> GetNodeIds() const {return node_ids;}
    std::vector<int> GetNumHeavy() const {return n_heavy;}
    std::vector<int> GetNumNodes() const {return n_nodes;}
    int GetSize() const {return max_id;}

    std::vector<int> id_old_to_survived_anc;
    std::vector<std::vector<int>> pos_old2new;
    std::unordered_map<int, std::vector<int>> leaf_to_path;

private:

    // Add a new child to the parent_id node (without incrementing n_docs)
    int AddChildren(int parent_id);

    std::vector<Node> nodes; // all nodes, nodes[0] is root (topological order)
    int L; // (max) number of levels
    int max_id; // total num of nodes, i.e. max node id + 1
    int thr_heavy;
    int thr_prune;
    int max_n_nodes;
    std::mutex mutex;
    // For each level:
    std::vector<double> log_gamma, gamma;
    std::vector<int> n_nodes; // num of nodes
    std::vector<int> n_heavy; // num of nodes with >thr_heavy count
    std::vector<std::vector<int>> node_ids;
    int debug_;
    std::vector<int> max_outdg;
    bool force_degree;
};

std::ostream& operator << (std::ostream &out, const ConcurrentTree::IncResult &tree);

void printTree(const ConcurrentTree::RetTree &tree, std::ostream &out = std::cout, bool leaf_only = false, bool marginal = false);

void WriteTreeAsDot(const ConcurrentTree::RetTree& tree, const std::string& filename, const std::vector<std::vector<std::string>>* extra_labels = nullptr, size_t words_per_line = 3, int max_depth = -1);

void WriteTreeAsTSV(const ConcurrentTree::RetTree& tree, const std::string& outpref);
