#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "Eigen/Core"
#include "Eigen/Sparse"

// Minimal Leiden community detection on a weighted, undirected graph using the
// RBConfiguration objective (modularity with a resolution parameter). This is a
// clean-room native implementation on plain CSR arrays; it does not depend on
// any external graph library. Algorithm notes live in leiden.cpp.
//
// The resolution parameter follows the same convention as libleidenalg /
// leidenalg: gamma == 1 corresponds to standard modularity.

using LeidenSparseMatrix = Eigen::SparseMatrix<double, Eigen::RowMajor>;

struct LeidenOptions {
    double  resolution     = 1.0;   // gamma; 1.0 == standard modularity
    int32_t max_iterations = -1;    // < 0: iterate to convergence; otherwise cap the number of passes
    int32_t seed           = 1;     // RNG seed for reproducible runs
};

struct LeidenResult {
    Eigen::VectorXi membership;         // community label per node in [0, n_communities)
    int32_t         n_communities = 0;
    double          quality       = 0.0; // RBConfiguration quality (== modularity at gamma = 1)
    int32_t         iterations    = 0;   // passes actually run
    bool            converged     = false;
};

// Cluster a weighted, undirected graph given as a symmetric CSR adjacency matrix.
// Diagonal entries are treated as self-loop weights. Weights must be finite and
// non-negative; the matrix is assumed symmetric.
LeidenResult leiden_cluster(const Eigen::Ref<const LeidenSparseMatrix>& adjacency,
                            const LeidenOptions& options);

// Cluster a weighted, undirected graph given as an edge list. Each undirected
// edge should be listed once; parallel edges are summed. Endpoints must lie in
// [0, n_nodes); self-edges (u == v) are treated as self-loops.
LeidenResult leiden_cluster(int32_t n_nodes,
                            const std::vector<std::pair<int32_t, int32_t>>& edges,
                            const std::vector<double>& weights,
                            const LeidenOptions& options);
