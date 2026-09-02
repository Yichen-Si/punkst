#include "multires_core/raw_affinity_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

class DisjointSet {
public:
    explicit DisjointSet(int32_t size) : parent_(static_cast<size_t>(size)),
        rank_(static_cast<size_t>(size), 0) {
        std::iota(parent_.begin(), parent_.end(), 0);
    }

    int32_t find(int32_t node) {
        int32_t root = node;
        while (parent_[static_cast<size_t>(root)] != root) {
            root = parent_[static_cast<size_t>(root)];
        }
        while (parent_[static_cast<size_t>(node)] != node) {
            const int32_t next = parent_[static_cast<size_t>(node)];
            parent_[static_cast<size_t>(node)] = root;
            node = next;
        }
        return root;
    }

    void unite(int32_t first, int32_t second) {
        first = find(first);
        second = find(second);
        if (first == second) return;
        if (rank_[static_cast<size_t>(first)]
                < rank_[static_cast<size_t>(second)]) {
            std::swap(first, second);
        }
        parent_[static_cast<size_t>(second)] = first;
        if (rank_[static_cast<size_t>(first)]
                == rank_[static_cast<size_t>(second)]) {
            ++rank_[static_cast<size_t>(first)];
        }
    }

private:
    std::vector<int32_t> parent_;
    std::vector<uint8_t> rank_;
};

void finalize_graph(RawClusteringGraph& graph) {
    if (graph.n_nodes <= 0 || graph.edges.size() != graph.weights.size()) {
        throw std::invalid_argument("Invalid raw clustering graph");
    }
    DisjointSet components(graph.n_nodes);
    double total = 0.0;
    std::pair<int32_t, int32_t> previous{-1, -1};
    for (size_t index = 0; index < graph.edges.size(); ++index) {
        const auto edge = graph.edges[index];
        const double weight = graph.weights[index];
        if (edge.first < 0 || edge.second < edge.first
            || edge.second >= graph.n_nodes || edge < previous
            || !(weight > 0.0) || !std::isfinite(weight)
            || !std::isfinite(total + weight)) {
            throw std::invalid_argument("Raw clustering edges are invalid");
        }
        previous = edge;
        total += weight;
        if (edge.first == edge.second) {
            ++graph.self_loop_edges;
        } else {
            components.unite(edge.first, edge.second);
        }
    }
    if (!(total > 0.0)) {
        throw std::invalid_argument(
            "Raw clustering graph has no positive non-bridge affinity");
    }
    graph.total_affinity = total;
    graph.component_labels.resize(static_cast<size_t>(graph.n_nodes));
    std::unordered_map<int32_t, int32_t> labels;
    for (int32_t node = 0; node < graph.n_nodes; ++node) {
        const int32_t root = components.find(node);
        const auto inserted = labels.emplace(
            root, static_cast<int32_t>(labels.size()));
        graph.component_labels[static_cast<size_t>(node)] =
            inserted.first->second;
    }
}

} // namespace

RawClusteringGraph make_raw_clustering_graph(const DiffusionGraph& graph) {
    if (graph.n_nodes <= 0 || graph.edges.size() != graph.raw_affinities.size()
        || graph.edges.size() != graph.edge_is_bridge.size()) {
        throw std::invalid_argument(
            "Invalid diffusion graph for raw clustering");
    }
    RawClusteringGraph out;
    out.n_nodes = graph.n_nodes;
    out.edges.reserve(graph.edges.size());
    out.weights.reserve(graph.edges.size());
    for (size_t index = 0; index < graph.edges.size(); ++index) {
        const double affinity = graph.raw_affinities[index];
        if (!(affinity >= 0.0) || affinity > 1.0
            || !std::isfinite(affinity)) {
            throw std::invalid_argument(
                "Fine raw affinity must lie in [0, 1]");
        }
        if (graph.edge_is_bridge[index]) {
            ++out.excluded_bridge_edges;
            out.excluded_bridge_affinity += affinity;
        } else if (affinity > 0.0) {
            out.edges.push_back(graph.edges[index]);
            out.weights.push_back(affinity);
        }
    }
    finalize_graph(out);
    return out;
}

RawClusteringGraph make_raw_clustering_graph(
        const CoarseningLevelGraph& graph) {
    if (graph.n_nodes <= 0) {
        throw std::invalid_argument(
            "Invalid coarsening graph for raw clustering");
    }
    RawClusteringGraph out;
    out.n_nodes = graph.n_nodes;
    out.edges.reserve(graph.edges.size());
    out.weights.reserve(graph.edges.size());
    for (const CoarseningEdge& edge : graph.edges) {
        if (edge.first < 0 || edge.second < edge.first
            || edge.second >= graph.n_nodes
            || !(edge.raw_affinity >= 0.0)
            || !(edge.raw_bridge_affinity >= 0.0)
            || edge.raw_bridge_affinity > edge.raw_affinity
            || !std::isfinite(edge.raw_affinity)
            || !std::isfinite(edge.raw_bridge_affinity)) {
            throw std::invalid_argument("Invalid coarse raw-affinity edge");
        }
        if (edge.raw_bridge_affinity > 0.0) {
            ++out.excluded_bridge_edges;
            out.excluded_bridge_affinity += edge.raw_bridge_affinity;
        }
        double affinity = edge.nonbridge_raw_affinity();
        const double tolerance = 1e-12 * std::max(1.0, edge.raw_affinity);
        if (affinity < 0.0 && affinity >= -tolerance) affinity = 0.0;
        if (affinity < 0.0 || !std::isfinite(affinity)) {
            throw std::invalid_argument(
                "Coarse non-bridge raw affinity is invalid");
        }
        if (affinity > 0.0) {
            out.edges.emplace_back(edge.first, edge.second);
            out.weights.push_back(affinity);
        }
    }
    finalize_graph(out);
    return out;
}

int32_t weighted_c90(const std::vector<int32_t>& membership,
        const std::vector<int64_t>& fine_point_counts) {
    if (membership.empty() || membership.size() != fine_point_counts.size()) {
        throw std::invalid_argument("C90 membership and counts must align");
    }
    std::unordered_map<int32_t, int64_t> masses;
    int64_t total = 0;
    for (size_t node = 0; node < membership.size(); ++node) {
        if (membership[node] < 0 || fine_point_counts[node] <= 0
            || total > std::numeric_limits<int64_t>::max()
                - fine_point_counts[node]) {
            throw std::invalid_argument(
                "C90 labels and fine-point counts must be valid");
        }
        masses[membership[node]] += fine_point_counts[node];
        total += fine_point_counts[node];
    }
    std::vector<int64_t> ordered;
    ordered.reserve(masses.size());
    for (const auto& item : masses) ordered.push_back(item.second);
    std::sort(ordered.begin(), ordered.end(), std::greater<int64_t>());
    int64_t cumulative = 0;
    for (size_t index = 0; index < ordered.size(); ++index) {
        cumulative += ordered[index];
        if (static_cast<long double>(cumulative)
                >= 0.9L * static_cast<long double>(total)) {
            return static_cast<int32_t>(index + 1);
        }
    }
    return static_cast<int32_t>(ordered.size());
}

double fine_weighted_adjusted_rand_index(
        const std::vector<int32_t>& first,
        const std::vector<int32_t>& second,
        const std::vector<int64_t>& fine_point_counts) {
    if (first.empty() || first.size() != second.size()
        || first.size() != fine_point_counts.size()) {
        throw std::invalid_argument("Weighted ARI inputs must align");
    }
    std::unordered_map<int32_t, int64_t> first_masses;
    std::unordered_map<int32_t, int64_t> second_masses;
    std::map<std::pair<int32_t, int32_t>, int64_t> cells;
    int64_t total = 0;
    for (size_t node = 0; node < first.size(); ++node) {
        const int64_t count = fine_point_counts[node];
        if (first[node] < 0 || second[node] < 0 || count <= 0
            || total > std::numeric_limits<int64_t>::max() - count) {
            throw std::invalid_argument("Weighted ARI inputs are invalid");
        }
        first_masses[first[node]] += count;
        second_masses[second[node]] += count;
        cells[{first[node], second[node]}] += count;
        total += count;
    }
    const auto choose_two = [](long double value) {
        return value * (value - 1.0L) / 2.0L;
    };
    const long double total_pairs = choose_two(total);
    if (!(total_pairs > 0.0L)) return 1.0;
    long double cell_pairs = 0.0L;
    long double first_pairs = 0.0L;
    long double second_pairs = 0.0L;
    for (const auto& item : cells) cell_pairs += choose_two(item.second);
    for (const auto& item : first_masses) first_pairs += choose_two(item.second);
    for (const auto& item : second_masses) second_pairs += choose_two(item.second);
    const long double expected = first_pairs * second_pairs / total_pairs;
    const long double maximum = 0.5L * (first_pairs + second_pairs);
    const long double denominator = maximum - expected;
    if (std::abs(denominator) <= std::numeric_limits<long double>::epsilon()
            * std::max(1.0L, std::abs(maximum))) {
        return std::abs(cell_pairs - expected)
                <= std::numeric_limits<long double>::epsilon()
                    * std::max(1.0L, std::abs(expected))
            ? 1.0 : 0.0;
    }
    const long double result = (cell_pairs - expected) / denominator;
    return static_cast<double>(std::clamp(result, -1.0L, 1.0L));
}
