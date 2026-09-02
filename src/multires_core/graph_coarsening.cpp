#include "multires_core/graph_coarsening.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

namespace {

struct MatchingCandidate {
    int32_t first = -1;
    int32_t second = -1;
    double score = 0.0;
};

class HeavyEdgeMatchingStrategy final : public GraphMatchingStrategy {
public:
    const char* name() const override { return "heavy-edge"; }

    std::vector<int32_t> match(
            const CoarseningLevelGraph& graph,
            int32_t max_pairs,
            int32_t maximum_microcluster_size,
            int64_t& size_rejected_candidates) const override {
        std::vector<int32_t> partner(static_cast<size_t>(graph.n_nodes));
        std::iota(partner.begin(), partner.end(), 0);
        if (max_pairs <= 0) return partner;

        // Matching is geometric: rank by mean raw Bhattacharyya affinity
        // over original non-bridge edges crossing the two current
        // aggregates. W^(alpha), local degree, and bridges play no role.
        std::vector<MatchingCandidate> candidates;
        candidates.reserve(graph.edges.size());
        for (const CoarseningEdge& edge : graph.edges) {
            if (edge.first == edge.second
                || edge.raw_nonbridge_edge_count == 0) {
                continue;
            }
            const double score = edge.mean_matching_raw_affinity();
            if (!(score > 0.0) || !std::isfinite(score)) continue;
            candidates.push_back({edge.first, edge.second, score});
        }
        tbb::parallel_sort(candidates.begin(), candidates.end(),
            [](const MatchingCandidate& first,
               const MatchingCandidate& second) {
                if (first.score != second.score) {
                    return first.score > second.score;
                }
                if (first.first != second.first) {
                    return first.first < second.first;
                }
                return first.second < second.second;
            });

        int32_t accepted = 0;
        for (const MatchingCandidate& candidate : candidates) {
            if (accepted == max_pairs) break;
            if (static_cast<int64_t>(graph.fine_node_counts[
                    static_cast<size_t>(candidate.first)])
                    + graph.fine_node_counts[static_cast<size_t>(
                        candidate.second)]
                > maximum_microcluster_size) {
                ++size_rejected_candidates;
                continue;
            }
            if (partner[static_cast<size_t>(candidate.first)]
                    != candidate.first
                || partner[static_cast<size_t>(candidate.second)]
                    != candidate.second) {
                continue;
            }
            partner[static_cast<size_t>(candidate.first)] = candidate.second;
            partner[static_cast<size_t>(candidate.second)] = candidate.first;
            ++accepted;
        }
        return partner;
    }
};

void validate_options(const GraphCoarseningOptions& options) {
    if (options.target_nodes < 0 || options.activation_threshold < 1
        || options.target_minimum < 1
        || options.target_maximum < options.target_minimum
        || options.target_divisor < 1
        || options.maximum_microcluster_size < 1
        || options.n_threads < 1) {
        throw std::invalid_argument("Invalid graph coarsening options");
    }
}

void validate_level_graph(const CoarseningLevelGraph& graph) {
    if (graph.n_nodes <= 0
        || graph.fine_node_counts.size()
            != static_cast<size_t>(graph.n_nodes)
        || graph.embedding_node_masses.size()
            != static_cast<size_t>(graph.n_nodes)) {
        throw std::invalid_argument("Invalid coarsening level graph");
    }
    std::pair<int32_t, int32_t> previous{-1, -1};
    for (int32_t node = 0; node < graph.n_nodes; ++node) {
        if (graph.fine_node_counts[static_cast<size_t>(node)] <= 0
            || !(graph.embedding_node_masses[
                static_cast<size_t>(node)] > 0.0)
            || !std::isfinite(graph.embedding_node_masses[
                static_cast<size_t>(node)])) {
            throw std::invalid_argument(
                "Coarsening nodes require positive sizes and masses");
        }
    }
    for (const CoarseningEdge& edge : graph.edges) {
        const std::pair<int32_t, int32_t> endpoints{
            edge.first, edge.second};
        if (edge.first < 0 || edge.second < edge.first
            || edge.second >= graph.n_nodes || endpoints <= previous
            || !std::isfinite(edge.embedding_nonbridge_weight)
            || !std::isfinite(edge.embedding_bridge_weight)
            || !std::isfinite(edge.raw_affinity)
            || !std::isfinite(edge.raw_bridge_affinity)
            || edge.embedding_nonbridge_weight < 0.0
            || edge.embedding_bridge_weight < 0.0
            || edge.raw_affinity < 0.0
            || edge.raw_bridge_affinity < 0.0
            || edge.raw_bridge_affinity > edge.raw_affinity
            || edge.raw_nonbridge_edge_count < 0
            || (edge.raw_nonbridge_edge_count == 0
                && edge.nonbridge_raw_affinity() != 0.0)
            || (!(edge.embedding_weight() > 0.0)
                && !(edge.raw_affinity > 0.0)
                && edge.raw_nonbridge_edge_count == 0)
            || !std::isfinite(edge.embedding_weight())) {
            throw std::invalid_argument(
                "Coarsening edges must be sorted, unique, and nonnegative");
        }
        previous = endpoints;
    }
}

int32_t resolved_target(int32_t n, const GraphCoarseningOptions& options) {
    if (options.target_nodes > 0) {
        return std::min(n, options.target_nodes);
    }
    const int64_t divided = (static_cast<int64_t>(n)
        + options.target_divisor - 1) / options.target_divisor;
    const int64_t bounded = std::clamp<int64_t>(divided,
        options.target_minimum, options.target_maximum);
    return static_cast<int32_t>(std::min<int64_t>(n, bounded));
}

void validate_partner(const std::vector<int32_t>& partner,
                      const CoarseningLevelGraph& graph,
                      int32_t max_pairs,
                      int32_t maximum_microcluster_size) {
    const int32_t n = graph.n_nodes;
    if (partner.size() != static_cast<size_t>(n)) {
        throw std::runtime_error("Graph matching strategy returned wrong size");
    }
    int32_t pairs = 0;
    for (int32_t node = 0; node < n; ++node) {
        const int32_t other = partner[static_cast<size_t>(node)];
        if (other < 0 || other >= n
            || partner[static_cast<size_t>(other)] != node) {
            throw std::runtime_error(
                "Graph matching strategy returned a non-involutive matching");
        }
        if (other != node && node < other) {
            if (static_cast<int64_t>(graph.fine_node_counts[
                    static_cast<size_t>(node)])
                    + graph.fine_node_counts[static_cast<size_t>(other)]
                > maximum_microcluster_size) {
                throw std::runtime_error(
                    "Graph matching strategy exceeded the microcluster cap");
            }
            ++pairs;
        }
    }
    if (pairs > max_pairs) {
        throw std::runtime_error(
            "Graph matching strategy exceeded the requested pair count");
    }
}

std::vector<int32_t> matching_membership(
        const std::vector<int32_t>& partner, int32_t& n_coarse) {
    const int32_t n = static_cast<int32_t>(partner.size());
    std::vector<int32_t> membership(static_cast<size_t>(n), -1);
    n_coarse = 0;
    for (int32_t node = 0; node < n; ++node) {
        if (membership[static_cast<size_t>(node)] >= 0) continue;
        const int32_t other = partner[static_cast<size_t>(node)];
        membership[static_cast<size_t>(node)] = n_coarse;
        membership[static_cast<size_t>(other)] = n_coarse;
        ++n_coarse;
    }
    return membership;
}

void hellinger_row_parameters(
        const Eigen::Ref<const RowMajorMatrixXd>& observations,
        int32_t row, double& scale, double& total) {
    if (!observations.row(row).allFinite()
        || (observations.row(row).array() < 0.0).any()) {
        throw std::invalid_argument(
            "Hellinger representatives require finite nonnegative rows");
    }
    scale = observations.row(row).maxCoeff();
    if (!(scale > 0.0)) {
        throw std::invalid_argument(
            "Hellinger representatives require positive row sums");
    }
    total = (observations.row(row) / scale).sum();
    if (!(total > 0.0) || !std::isfinite(total)) {
        throw std::invalid_argument(
            "Hellinger representatives require positive row sums");
    }
}

} // namespace

const GraphMatchingStrategy& heavy_edge_matching_strategy() {
    static const HeavyEdgeMatchingStrategy strategy;
    return strategy;
}

CoarseningLevelGraph make_fine_coarsening_graph(
        const DiffusionGraph& graph) {
    if (graph.n_nodes <= 0
        || graph.edges.size() != graph.embedding.diffusion_weights.size()
        || graph.edges.size() != graph.raw_affinities.size()
        || graph.edges.size() != graph.edge_is_bridge.size()
        || graph.embedding.node_mass.size()
            != static_cast<size_t>(graph.n_nodes)) {
        throw std::invalid_argument("Invalid diffusion graph for coarsening");
    }
    CoarseningLevelGraph out;
    out.n_nodes = graph.n_nodes;
    out.fine_node_counts.assign(static_cast<size_t>(graph.n_nodes), 1);
    out.embedding_node_masses = graph.embedding.node_mass;
    out.edges.reserve(graph.edges.size());
    std::pair<int32_t, int32_t> previous{-1, -1};
    for (size_t position = 0; position < graph.edges.size(); ++position) {
        const auto endpoints = graph.edges[position];
        const double embedding_weight =
            graph.embedding.diffusion_weights[position];
        const double raw_affinity = graph.raw_affinities[position];
        if (endpoints.first < 0 || endpoints.second <= endpoints.first
            || endpoints.second >= graph.n_nodes || endpoints <= previous
            || embedding_weight < 0.0
            || !std::isfinite(embedding_weight)
            || raw_affinity < 0.0
            || raw_affinity > 1.0
            || !std::isfinite(raw_affinity)
            || graph.edge_is_bridge[position] > 1) {
            throw std::invalid_argument(
                "Diffusion graph edges are invalid for coarsening");
        }
        previous = endpoints;
        CoarseningEdge edge;
        edge.first = endpoints.first;
        edge.second = endpoints.second;
        if (graph.edge_is_bridge[position]) {
            edge.embedding_bridge_weight = embedding_weight;
            edge.raw_affinity = raw_affinity;
            edge.raw_bridge_affinity = raw_affinity;
        } else {
            edge.embedding_nonbridge_weight = embedding_weight;
            edge.raw_affinity = raw_affinity;
            edge.raw_nonbridge_edge_count = 1;
        }
        if (!(edge.embedding_weight() > 0.0)
            && !(edge.raw_affinity > 0.0)
            && edge.raw_nonbridge_edge_count == 0) continue;
        out.edges.push_back(edge);
    }
    validate_level_graph(out);
    return out;
}

CoarseningLevelGraph aggregate_coarsening_graph(
        const CoarseningLevelGraph& graph,
        const std::vector<int32_t>& membership,
        int32_t n_coarse_nodes) {
    validate_level_graph(graph);
    if (membership.size() != static_cast<size_t>(graph.n_nodes)
        || n_coarse_nodes <= 0 || n_coarse_nodes > graph.n_nodes) {
        throw std::invalid_argument("Invalid coarsening membership");
    }
    CoarseningLevelGraph out;
    out.n_nodes = n_coarse_nodes;
    out.fine_node_counts.assign(static_cast<size_t>(n_coarse_nodes), 0);
    out.embedding_node_masses.assign(
        static_cast<size_t>(n_coarse_nodes), 0.0);
    std::vector<uint8_t> used(static_cast<size_t>(n_coarse_nodes), 0);
    for (int32_t node = 0; node < graph.n_nodes; ++node) {
        const int32_t coarse = membership[static_cast<size_t>(node)];
        if (coarse < 0 || coarse >= n_coarse_nodes) {
            throw std::invalid_argument("Coarsening label is out of range");
        }
        used[static_cast<size_t>(coarse)] = 1;
        const int64_t combined_size = static_cast<int64_t>(
            out.fine_node_counts[static_cast<size_t>(coarse)])
            + graph.fine_node_counts[static_cast<size_t>(node)];
        if (combined_size > std::numeric_limits<int32_t>::max()) {
            throw std::overflow_error("Coarse node size exceeds int32 range");
        }
        out.fine_node_counts[static_cast<size_t>(coarse)] =
            static_cast<int32_t>(combined_size);
        out.embedding_node_masses[static_cast<size_t>(coarse)] +=
            graph.embedding_node_masses[static_cast<size_t>(node)];
    }
    if (std::find(used.begin(), used.end(), uint8_t{0}) != used.end()) {
        throw std::invalid_argument("Coarsening labels must be contiguous");
    }

    std::vector<CoarseningEdge> mapped;
    mapped.reserve(graph.edges.size());
    for (const CoarseningEdge& edge : graph.edges) {
        int32_t first = membership[static_cast<size_t>(edge.first)];
        int32_t second = membership[static_cast<size_t>(edge.second)];
        if (second < first) std::swap(first, second);
        mapped.push_back({first, second,
            edge.embedding_nonbridge_weight,
            edge.embedding_bridge_weight,
            edge.raw_affinity, edge.raw_bridge_affinity,
            edge.raw_nonbridge_edge_count});
    }
    tbb::parallel_sort(mapped.begin(), mapped.end(),
        [](const CoarseningEdge& first, const CoarseningEdge& second) {
            return std::pair<int32_t, int32_t>{first.first, first.second}
                < std::pair<int32_t, int32_t>{second.first, second.second};
        });
    out.edges.reserve(mapped.size());
    for (size_t position = 0; position < mapped.size();) {
        CoarseningEdge combined = mapped[position];
        size_t next = position + 1;
        while (next < mapped.size()
            && mapped[next].first == combined.first
            && mapped[next].second == combined.second) {
            combined.embedding_nonbridge_weight +=
                mapped[next].embedding_nonbridge_weight;
            combined.embedding_bridge_weight +=
                mapped[next].embedding_bridge_weight;
            combined.raw_affinity += mapped[next].raw_affinity;
            combined.raw_bridge_affinity +=
                mapped[next].raw_bridge_affinity;
            if (mapped[next].raw_nonbridge_edge_count
                    > std::numeric_limits<int64_t>::max()
                        - combined.raw_nonbridge_edge_count) {
                throw std::overflow_error(
                    "Coarse raw-affinity edge count exceeds int64 range");
            }
            combined.raw_nonbridge_edge_count +=
                mapped[next].raw_nonbridge_edge_count;
            ++next;
        }
        if (!std::isfinite(combined.embedding_nonbridge_weight)
            || !std::isfinite(combined.embedding_bridge_weight)
            || !std::isfinite(combined.raw_affinity)
            || !std::isfinite(combined.raw_bridge_affinity)) {
            throw std::overflow_error("Coarse edge weight overflow");
        }
        out.edges.push_back(combined);
        position = next;
    }
    validate_level_graph(out);
    return out;
}

GalerkinOperator build_galerkin_operator(
        const CoarseningLevelGraph& graph) {
    validate_level_graph(graph);
    const std::vector<double>& node_masses = graph.embedding_node_masses;
    GalerkinOperator out;
    out.n_nodes = graph.n_nodes;
    out.mass = node_masses;
    out.inverse_sqrt_mass.resize(static_cast<size_t>(graph.n_nodes));
    out.diagonal.assign(static_cast<size_t>(graph.n_nodes), 0.0);
    std::vector<double> laplacian_degree(
        static_cast<size_t>(graph.n_nodes), 0.0);
    for (int32_t node = 0; node < graph.n_nodes; ++node) {
        out.inverse_sqrt_mass[static_cast<size_t>(node)] = 1.0 / std::sqrt(
            node_masses[static_cast<size_t>(node)]);
        if (!std::isfinite(
                out.inverse_sqrt_mass[static_cast<size_t>(node)])) {
            throw std::overflow_error(
                "Galerkin inverse mass scale is not finite");
        }
    }
    for (const CoarseningEdge& edge : graph.edges) {
        if (edge.first == edge.second) continue;
        const double weight = edge.embedding_weight();
        if (!(weight > 0.0)) continue;
        laplacian_degree[static_cast<size_t>(edge.first)] += weight;
        laplacian_degree[static_cast<size_t>(edge.second)] += weight;
        out.off_diagonal_edges.emplace_back(edge.first, edge.second);
        out.off_diagonal_values.push_back(-weight
            * out.inverse_sqrt_mass[static_cast<size_t>(edge.first)]
            * out.inverse_sqrt_mass[static_cast<size_t>(edge.second)]);
        if (!std::isfinite(out.off_diagonal_values.back())) {
            throw std::overflow_error("Galerkin edge is not finite");
        }
    }
    std::vector<double> radius(static_cast<size_t>(graph.n_nodes), 0.0);
    for (int32_t node = 0; node < graph.n_nodes; ++node) {
        out.diagonal[static_cast<size_t>(node)] = laplacian_degree[
            static_cast<size_t>(node)]
            / node_masses[static_cast<size_t>(node)];
        if (!std::isfinite(out.diagonal[static_cast<size_t>(node)])) {
            throw std::overflow_error("Galerkin diagonal is not finite");
        }
    }
    for (size_t edge = 0; edge < out.off_diagonal_edges.size(); ++edge) {
        const double magnitude = -out.off_diagonal_values[edge];
        radius[static_cast<size_t>(out.off_diagonal_edges[edge].first)] +=
            magnitude;
        radius[static_cast<size_t>(out.off_diagonal_edges[edge].second)] +=
            magnitude;
    }
    for (int32_t node = 0; node < graph.n_nodes; ++node) {
        out.gershgorin_upper_bound = std::max(
            out.gershgorin_upper_bound,
            out.diagonal[static_cast<size_t>(node)]
                + radius[static_cast<size_t>(node)]);
    }
    if (!std::isfinite(out.gershgorin_upper_bound)) {
        throw std::overflow_error("Galerkin Gershgorin bound is not finite");
    }
    return out;
}

Eigen::VectorXd apply_galerkin_operator(
        const GalerkinOperator& op,
        const Eigen::Ref<const Eigen::VectorXd>& values) {
    if (op.n_nodes <= 0 || values.size() != op.n_nodes || !values.allFinite()
        || op.mass.size() != static_cast<size_t>(op.n_nodes)
        || op.inverse_sqrt_mass.size() != static_cast<size_t>(op.n_nodes)
        || op.diagonal.size() != static_cast<size_t>(op.n_nodes)
        || op.off_diagonal_edges.size() != op.off_diagonal_values.size()) {
        throw std::invalid_argument("Invalid Galerkin operator application");
    }
    Eigen::VectorXd out(op.n_nodes);
    for (int32_t node = 0; node < op.n_nodes; ++node) {
        out(node) = op.diagonal[static_cast<size_t>(node)] * values(node);
    }
    for (size_t edge = 0; edge < op.off_diagonal_edges.size(); ++edge) {
        const auto endpoints = op.off_diagonal_edges[edge];
        const double weight = op.off_diagonal_values[edge];
        if (endpoints.first < 0 || endpoints.second <= endpoints.first
            || endpoints.second >= op.n_nodes || !std::isfinite(weight)) {
            throw std::invalid_argument("Invalid Galerkin off-diagonal edge");
        }
        out(endpoints.first) += weight * values(endpoints.second);
        out(endpoints.second) += weight * values(endpoints.first);
    }
    return out;
}

std::vector<int32_t> select_hellinger_representatives(
        const Eigen::Ref<const RowMajorMatrixXd>& observations,
        const std::vector<int32_t>& membership,
        int32_t n_microclusters,
        const std::vector<double>& node_masses,
        int32_t n_threads) {
    if (observations.rows() <= 0 || observations.cols() <= 0
        || observations.rows() > std::numeric_limits<int32_t>::max()
        || membership.size() != static_cast<size_t>(observations.rows())
        || node_masses.size() != static_cast<size_t>(observations.rows())
        || n_microclusters <= 0 || n_threads <= 0) {
        throw std::invalid_argument("Invalid representative-selection input");
    }
    std::vector<std::vector<int32_t>> members(
        static_cast<size_t>(n_microclusters));
    for (int32_t node = 0; node < observations.rows(); ++node) {
        const int32_t cluster = membership[static_cast<size_t>(node)];
        if (cluster < 0 || cluster >= n_microclusters
            || !(node_masses[static_cast<size_t>(node)] > 0.0)
            || !std::isfinite(node_masses[static_cast<size_t>(node)])) {
            throw std::invalid_argument(
                "Representative labels and masses must be valid");
        }
        members[static_cast<size_t>(cluster)].push_back(node);
    }
    for (const auto& cluster : members) {
        if (cluster.empty()) {
            throw std::invalid_argument(
                "Representative labels must be contiguous");
        }
    }

    tbb::global_control parallelism(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(n_threads));
    RowMajorMatrixXd centroids = RowMajorMatrixXd::Zero(
        n_microclusters, observations.cols());
    std::vector<double> cluster_mass(
        static_cast<size_t>(n_microclusters), 0.0);
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, n_microclusters, 8),
        [&](const tbb::blocked_range<int32_t>& range) {
            for (int32_t cluster = range.begin(); cluster < range.end();
                 ++cluster) {
                for (int32_t node : members[static_cast<size_t>(cluster)]) {
                    double scale = 0.0;
                    double total = 0.0;
                    hellinger_row_parameters(
                        observations, node, scale, total);
                    const double mass = node_masses[static_cast<size_t>(node)];
                    cluster_mass[static_cast<size_t>(cluster)] += mass;
                    for (Eigen::Index column = 0;
                         column < observations.cols(); ++column) {
                        centroids(cluster, column) += mass * std::sqrt(
                            observations(node, column) / scale / total);
                    }
                }
                if (!(cluster_mass[static_cast<size_t>(cluster)] > 0.0)
                    || !std::isfinite(
                        cluster_mass[static_cast<size_t>(cluster)])) {
                    throw std::overflow_error(
                        "Representative cluster mass is not finite");
                }
                centroids.row(cluster) /=
                    cluster_mass[static_cast<size_t>(cluster)];
                if (!centroids.row(cluster).allFinite()) {
                    throw std::overflow_error(
                        "Representative centroid is not finite");
                }
            }
        });

    std::vector<int32_t> representatives(
        static_cast<size_t>(n_microclusters), -1);
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, n_microclusters, 8),
        [&](const tbb::blocked_range<int32_t>& range) {
            for (int32_t cluster = range.begin(); cluster < range.end();
                 ++cluster) {
                double best_distance = std::numeric_limits<double>::infinity();
                int32_t best_node = -1;
                for (int32_t node : members[static_cast<size_t>(cluster)]) {
                    double scale = 0.0;
                    double total = 0.0;
                    hellinger_row_parameters(
                        observations, node, scale, total);
                    double distance = 0.0;
                    for (Eigen::Index column = 0;
                         column < observations.cols(); ++column) {
                        const double value = std::sqrt(
                            observations(node, column) / scale / total);
                        const double delta = value - centroids(cluster, column);
                        distance += delta * delta;
                    }
                    const double tie_tolerance = 64.0
                        * std::numeric_limits<double>::epsilon()
                        * std::max({1.0, std::abs(distance),
                            std::abs(best_distance)});
                    if (best_node < 0
                        || distance < best_distance - tie_tolerance
                        || (std::abs(distance - best_distance)
                                <= tie_tolerance
                            && node < best_node)) {
                        best_distance = distance;
                        best_node = node;
                    }
                }
                representatives[static_cast<size_t>(cluster)] = best_node;
            }
        });
    return representatives;
}

GraphCoarseningResult coarsen_diffusion_graph(
        const DiffusionGraph& graph,
        const Eigen::Ref<const RowMajorMatrixXd>& observations,
        const GraphCoarseningOptions& options) {
    return coarsen_diffusion_graph(
        graph, observations, heavy_edge_matching_strategy(), options);
}

GraphCoarseningResult coarsen_diffusion_graph(
        const DiffusionGraph& graph,
        const Eigen::Ref<const RowMajorMatrixXd>& observations,
        const GraphMatchingStrategy& strategy,
        const GraphCoarseningOptions& options) {
    validate_options(options);
    tbb::global_control parallelism(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(options.n_threads));
    CoarseningLevelGraph current = make_fine_coarsening_graph(graph);
    if (observations.rows() != current.n_nodes) {
        throw std::invalid_argument(
            "Coarsening observations and graph row counts differ");
    }
    GraphCoarseningResult out;
    const char* strategy_name = strategy.name();
    if (strategy_name == nullptr || strategy_name[0] == '\0') {
        throw std::invalid_argument("Graph matching strategy has no name");
    }
    out.diagnostics.strategy_name = strategy_name;
    out.diagnostics.requested_target_nodes = options.target_nodes;
    out.diagnostics.resolved_target_nodes = resolved_target(
        current.n_nodes, options);
    out.diagnostics.activated = options.target_nodes > 0
        ? out.diagnostics.resolved_target_nodes < current.n_nodes
        : current.n_nodes > options.activation_threshold;
    out.diagnostics.level_node_counts.push_back(current.n_nodes);
    out.membership.resize(static_cast<size_t>(current.n_nodes));
    std::iota(out.membership.begin(), out.membership.end(), 0);

    if (out.diagnostics.activated) {
        while (current.n_nodes > out.diagnostics.resolved_target_nodes) {
            const int32_t max_pairs = current.n_nodes
                - out.diagnostics.resolved_target_nodes;
            std::vector<int32_t> partner = strategy.match(current, max_pairs,
                options.maximum_microcluster_size,
                out.diagnostics.size_rejected_candidates);
            validate_partner(partner, current, max_pairs,
                options.maximum_microcluster_size);
            int32_t next_nodes = 0;
            const std::vector<int32_t> level_membership =
                matching_membership(partner, next_nodes);
            const int32_t matches = current.n_nodes - next_nodes;
            if (matches == 0) {
                out.diagnostics.stopped_without_matches = true;
                break;
            }
            for (int32_t& cluster : out.membership) {
                cluster = level_membership[static_cast<size_t>(cluster)];
            }
            current = aggregate_coarsening_graph(
                current, level_membership, next_nodes);
            ++out.diagnostics.matching_passes;
            out.diagnostics.accepted_matches += matches;
            out.diagnostics.level_node_counts.push_back(current.n_nodes);
        }
    }
    out.diagnostics.target_reached = current.n_nodes
        <= out.diagnostics.resolved_target_nodes;
    out.representatives = select_hellinger_representatives(observations,
        out.membership, current.n_nodes, graph.embedding.node_mass,
        options.n_threads);
    out.embedding_galerkin = build_galerkin_operator(current);
    out.graph = std::move(current);
    return out;
}
