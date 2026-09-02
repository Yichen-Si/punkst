#pragma once

// Bridge-excluded raw Bhattacharyya graphs used for density-sensitive
// clustering and scene construction.

#include <cstdint>
#include <utility>
#include <vector>

#include "multires_core/diffusion_graph.hpp"
#include "multires_core/graph_coarsening.hpp"

struct RawClusteringGraph {
    int32_t n_nodes = 0;
    std::vector<std::pair<int32_t, int32_t>> edges;
    std::vector<double> weights;
    std::vector<int32_t> component_labels;
    int64_t excluded_bridge_edges = 0;
    double excluded_bridge_affinity = 0.0;
    int64_t self_loop_edges = 0;
    double total_affinity = 0.0;
};

RawClusteringGraph make_raw_clustering_graph(const DiffusionGraph& graph);

// Aggregated self-loops are retained so weighted degrees and the RB objective
// agree with the corresponding constrained fine-point partition.
RawClusteringGraph make_raw_clustering_graph(
    const CoarseningLevelGraph& graph);

int32_t weighted_c90(
    const std::vector<int32_t>& membership,
    const std::vector<int64_t>& fine_point_counts);

double fine_weighted_adjusted_rand_index(
    const std::vector<int32_t>& first,
    const std::vector<int32_t>& second,
    const std::vector<int64_t>& fine_point_counts);
