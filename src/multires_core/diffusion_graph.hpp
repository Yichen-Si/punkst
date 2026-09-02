#pragma once

#include <cstdint>
#include <utility>
#include <vector>

#include "clustering_core/cosine_clustering.hpp"

struct HellingerKnnGraphOptions {
    HellingerKnnGraphOptions();

    CosineKnnOptions knn;
    bool build_bridge_candidates = true;
    bool retain_diffusion_geometry = true;
    int32_t bridges_per_component_link = 3;
    int32_t bridge_shortlist_size = 64;
};

struct HellingerBridgeCandidate {
    int32_t first = -1;
    int32_t second = -1;
    int32_t first_component = -1;
    int32_t second_component = -1;
    double hellinger_distance_squared = 0.0;
};

// Diffusion-independent geometry produced by a Hellinger k-NN search. The
// canonical edge support excludes empirical bridges. Directed distances and
// coordinates may be omitted when publishing an artifact that will only be
// used for raw-affinity clustering.
struct HellingerKnnGraph {
    int32_t n_nodes = 0;
    int32_t n_neighbors = 0;
    std::vector<int32_t> directed_neighbor_indices;
    std::vector<double> directed_distance_squared;
    std::vector<std::pair<int32_t, int32_t>> edges;
    std::vector<double> raw_affinities;
    std::vector<int32_t> component_labels;
    std::vector<int32_t> component_sizes;
    std::vector<HellingerBridgeCandidate> bridge_candidates;
    RowMajorMatrixXd coordinates;
    CosineKnnDiagnostics knn;
};

HellingerKnnGraph build_hellinger_knn_graph(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const HellingerKnnGraphOptions& options = HellingerKnnGraphOptions());

struct DiffusionGraphOptions {
    DiffusionGraphOptions();

    CosineKnnOptions knn;
    // Zero resolves to k.
    int32_t embedding_bandwidth_neighbor_rank = 0;
    double embedding_bandwidth_minimum_ratio = 0.05;
    double embedding_bandwidth_maximum_ratio = 4.0;
    double embedding_alpha = 1.0;
    double embedding_beta = 0.0;
    double bridge_weight_quantile = 0.05;
    int32_t bridges_per_component_link = 3;
    int32_t bridge_shortlist_size = 64;
    int64_t quantile_sample_size = 1000000;
};

struct DiffusionBridge {
    int32_t first = -1;
    int32_t second = -1;
    int32_t first_component = -1;
    int32_t second_component = -1;
    double hellinger_distance_squared = 0.0;
    double embedding_geometric_weight = 0.0;
    double embedding_kernel_weight = 0.0;
    double embedding_weight_inflation = 1.0;
    bool embedding_geometric_weight_underflow = false;
    double component_pair_conductance = 0.0;
};

struct DiffusionGraphDiagnostics {
    int32_t initial_components = 0;
    int32_t final_components = 0;
    int32_t embedding_bandwidth_neighbor_rank = 0;
    int64_t embedding_bandwidth_sample_size = 0;
    int64_t quantile_sample_size = 0;
    int64_t base_edges = 0;
    int64_t bridge_edges = 0;
    double embedding_median_bandwidth_squared = 0.0;
    double embedding_minimum_bandwidth_squared = 0.0;
    double embedding_maximum_bandwidth_squared = 0.0;
    int64_t embedding_bandwidth_floor_count = 0;
    int64_t embedding_bandwidth_cap_count = 0;
    // Exact percentiles 1,...,99 of the post-clipping local bandwidth h_i
    // (not h_i^2) used by the self-tuning embedding kernel.
    std::vector<double> embedding_bandwidth_percentiles;
    double embedding_bridge_weight_floor = 0.0;
    double maximum_bridge_pair_conductance = 0.0;
    std::vector<int32_t> initial_component_sizes;
};

struct DiffusionOperatorView {
    double alpha = 1.0;
    double beta = 0.0;
    // Self-tuning kernel and alpha-normalized weights, aligned with graph.edges.
    std::vector<double> kernel_weights;
    std::vector<double> diffusion_weights;
    std::vector<double> kernel_degree;
    std::vector<double> diffusion_degree;
    std::vector<double> node_mass;
    std::vector<double> stationary_probability;
};

// Canonical undirected support shared by diffusion eigendecomposition and
// subsequent multiresolution construction. Edges are lexicographically sorted
// and all aligned vectors have edges.size() entries.
struct DiffusionGraph {
    int32_t n_nodes = 0;
    std::vector<std::pair<int32_t, int32_t>> edges;
    std::vector<uint8_t> edge_is_bridge;
    // Squared per-node self-tuning Hellinger bandwidth used for embedding.
    std::vector<double> embedding_bandwidth_squared;
    // Unnormalized Bhattacharyya affinity 1-d_H^2 on every canonical edge,
    // including empirical bridges. edge_is_bridge controls whether an edge is
    // eligible for matching or clustering; stored geometry is never masked.
    std::vector<double> raw_affinities;
    DiffusionOperatorView embedding;
    std::vector<DiffusionBridge> bridges;
    CosineKnnDiagnostics knn;
    DiffusionGraphDiagnostics diagnostics;
};

DiffusionGraph build_hellinger_diffusion_graph(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DiffusionGraphOptions& options = DiffusionGraphOptions());
