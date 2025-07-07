# pragma once

#include <array>
#include <atomic>
#include <iostream>
#include <mutex>
#include <set>
#include <vector>
#include <algorithm>
#include <memory>
#include <cmath>
#include <cstring>

class ConcurrentTree {
public:

    struct Node {
        int parent_id = -1;
        int pos = -1; // index within the level
        int depth = -1;
        std::atomic<int> num_docs {0};
        double log_weight = 0;
        Node() = default;
        Node(Node&&) noexcept = default;
        Node& operator=(Node&&) noexcept = default;
        Node& operator=(const Node& other) {
            parent_id = other.parent_id;
            pos = other.pos;
            depth = other.depth;
            num_docs.store(other.num_docs.load());
            log_weight = other.log_weight;
            return *this;
        }
    };

    struct RetNode {
        int parent_id, pos, depth, num_docs;
        double log_path_weight;
    };

    struct RetTree {
        std::vector<RetNode> nodes;
        std::vector<int> num_nodes;
    };

    struct IncResult {
        int id;
        std::vector<uint32_t> pos;
        IncResult(int id, int L) : id(id), pos(L) {}
    };

    ConcurrentTree(int L, double gamma, int max_n_nodes = 1024, int _thr_heavy = INT_MAX, int _thr_prune = 0, int _branching_factor = -1, int _debug = 0);
    ConcurrentTree(int L, std::vector<double> log_gamma, int max_n_nodes = 1024, int _thr_heavy = INT_MAX, int _thr_prune = 0, int _branching_factor = -1, int _debug = 0);

    // Lock-free operations
    bool IsLeaf(int node_id);
    bool Exist(int node_id);
        // Decrease the doc count by 1 for all nodes on node_id->root
    void DecNumDocs(int node_id);
        // Increase the doc count by delta for all nodes on node_id->root
        // If node_id is not a leaf, do AddNodes first (lock)
        // The returned path is always from root to a leaf (could be new)
    IncResult IncNumDocs(int node_id, int delta = 1);
        // Realize the tree with current nCRP probabilities
    RetTree GetTree();

    // Lock operations
        // The only place where a new node can be added
        // Start from root_id, add a new path (up to the leaf level)
    int AddNodes(int root_id);

    // Global operations (not meant to be called concurrently)
    void SetGamma(std::vector<double> gamma);
    void SetLogGamma(std::vector<double> log_gamma);
    void SetThreshold(int thr_heavy, int thr_prune = 0);
    void SetBranchingFactor(int branching_factor);
        // The only place where a node can be removed (prune empty/rare nodes)
        //                   or a node's within-level pos modified
    bool Consolidate(std::vector<std::vector<int>>& pos_map);
        // Compute nCRP prob at each node given the current obs (counts)
        //     log(child | parent) (\pi), stored in nodes[i].log_weight
        //     (the proper stick breaking prob)
    void Instantiate();

    const std::vector<std::vector<int>>& GetNodeIdsC() const {return node_ids;}
    const std::vector<int>& GetNumHeavyC() const {return num_heavy;}
    const std::vector<int>& GetNumNodesC() const {return num_nodes;}
    std::vector<std::vector<int>> GetNodeIds() const {return node_ids;}
    std::vector<int> GetNumHeavy() const {return num_heavy;}
    std::vector<int> GetNumNodes() const {return num_nodes;}
    int GetSize() const {return max_id;}

private:

    void init();

    // Add a new child to the parent_id node (without incrementing num_docs)
    int AddChildren(int parent_id);

    std::vector<Node> nodes; // all nodes, nodes[0] is root (topological order)
    int L; // number of levels
    int max_id; // total num of nodes, i.e. max node id + 1
    int thr_heavy;
    int thr_prune;
    int branching_factor;
    int max_n_nodes;
    std::mutex mutex;
    // For each level:
    std::vector<double> log_gamma;
    std::vector<int> num_nodes; // num of nodes
    std::vector<int> num_heavy; // num of nodes with >thr_heavy count
    std::vector<std::vector<int>> node_ids;
    int debug;
};

std::ostream& operator << (std::ostream &out, const ConcurrentTree::RetTree &tree);

std::ostream& operator << (std::ostream &out, const ConcurrentTree::IncResult &tree);
