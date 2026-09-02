#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "multires_core/diffusion_graph.hpp"

struct CoarseningEdge {
    int32_t first = -1;
    int32_t second = -1;
    // Embedding contributions remain separated by bridge provenance.
    double embedding_nonbridge_weight = 0.0;
    double embedding_bridge_weight = 0.0;
    // Exact Bhattacharyya-affinity sum over every original edge and the subset
    // contributed by empirical bridges. Clustering uses their difference;
    // matching uses its mean over the retained non-bridge edges.
    double raw_affinity = 0.0;
    double raw_bridge_affinity = 0.0;
    int64_t raw_nonbridge_edge_count = 0;

    double embedding_weight() const {
        return embedding_nonbridge_weight + embedding_bridge_weight;
    }

    double nonbridge_raw_affinity() const {
        return raw_affinity - raw_bridge_affinity;
    }

    double mean_matching_raw_affinity() const {
        return raw_nonbridge_edge_count > 0
            ? nonbridge_raw_affinity()
                / static_cast<double>(raw_nonbridge_edge_count)
            : 0.0;
    }
};

// A coarse graph retains self-loops for exact aggregation. Embedding bridge
// and non-bridge contributions remain separate even when they collapse onto
// the same coarse edge. Raw affinity retains both contributions while its
// bridge subset remains identifiable and excluded from clustering by default.
struct CoarseningLevelGraph {
    int32_t n_nodes = 0;
    std::vector<CoarseningEdge> edges;
    std::vector<int32_t> fine_node_counts;
    std::vector<double> embedding_node_masses;
};

struct GraphCoarseningOptions {
    // Zero selects min(n, clamp(ceil(n / target_divisor), target_minimum,
    // target_maximum)). An explicit target activates coarsening below the
    // normal activation threshold, which is useful for small inputs and tests.
    int32_t target_nodes = 0;
    int32_t activation_threshold = 50000;
    int32_t target_minimum = 20000;
    int32_t target_maximum = 50000;
    int32_t target_divisor = 48;
    int32_t maximum_microcluster_size = 96;
    int32_t n_threads = 1;
};

struct CoarseningDiagnostics {
    std::string strategy_name;
    int32_t requested_target_nodes = 0;
    int32_t resolved_target_nodes = 0;
    int32_t matching_passes = 0;
    int64_t accepted_matches = 0;
    int64_t size_rejected_candidates = 0;
    bool activated = false;
    bool target_reached = false;
    bool stopped_without_matches = false;
    std::vector<int32_t> level_node_counts;
};

struct GalerkinOperator {
    int32_t n_nodes = 0;
    // S_c = diag(mass), and A_c = S_c^-1/2 C_c S_c^-1/2.
    std::vector<double> mass;
    std::vector<double> inverse_sqrt_mass;
    std::vector<double> diagonal;
    std::vector<std::pair<int32_t, int32_t>> off_diagonal_edges;
    std::vector<double> off_diagonal_values;
    double gershgorin_upper_bound = 0.0;
};

struct GraphCoarseningResult {
    // Fine node -> final microcluster, with contiguous canonical labels.
    std::vector<int32_t> membership;
    std::vector<int32_t> representatives;
    CoarseningLevelGraph graph;
    GalerkinOperator embedding_galerkin;
    CoarseningDiagnostics diagnostics;
};

class GraphMatchingStrategy {
public:
    virtual ~GraphMatchingStrategy() = default;
    virtual const char* name() const = 0;

    // Return an involutive partner vector. An unmatched node maps to itself.
    // At most max_pairs nontrivial pairs may be returned.
    virtual std::vector<int32_t> match(
        const CoarseningLevelGraph& graph,
        int32_t max_pairs,
        int32_t maximum_microcluster_size,
        int64_t& size_rejected_candidates) const = 0;
};

const GraphMatchingStrategy& heavy_edge_matching_strategy();

CoarseningLevelGraph make_fine_coarsening_graph(
    const DiffusionGraph& graph);

CoarseningLevelGraph aggregate_coarsening_graph(
    const CoarseningLevelGraph& graph,
    const std::vector<int32_t>& membership,
    int32_t n_coarse_nodes);

GalerkinOperator build_galerkin_operator(
    const CoarseningLevelGraph& graph);

Eigen::VectorXd apply_galerkin_operator(
    const GalerkinOperator& op,
    const Eigen::Ref<const Eigen::VectorXd>& values);

std::vector<int32_t> select_hellinger_representatives(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const std::vector<int32_t>& membership,
    int32_t n_microclusters,
    const std::vector<double>& node_masses,
    int32_t n_threads = 1);

GraphCoarseningResult coarsen_diffusion_graph(
    const DiffusionGraph& graph,
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const GraphCoarseningOptions& options = GraphCoarseningOptions());

GraphCoarseningResult coarsen_diffusion_graph(
    const DiffusionGraph& graph,
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const GraphMatchingStrategy& strategy,
    const GraphCoarseningOptions& options);
