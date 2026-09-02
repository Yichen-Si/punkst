#include "multires_core/diffusion_graph.hpp"

#include "nanoflann.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace {

class DisjointSet {
public:
    explicit DisjointSet(int32_t size)
        : parent_(static_cast<size_t>(size)),
          size_(static_cast<size_t>(size), 1) {
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

    bool join(int32_t first, int32_t second) {
        first = find(first);
        second = find(second);
        if (first == second) return false;
        if (size_[static_cast<size_t>(first)]
                < size_[static_cast<size_t>(second)]) {
            std::swap(first, second);
        }
        parent_[static_cast<size_t>(second)] = first;
        size_[static_cast<size_t>(first)] +=
            size_[static_cast<size_t>(second)];
        return true;
    }

private:
    std::vector<int32_t> parent_;
    std::vector<int32_t> size_;
};

struct Components {
    std::vector<int32_t> labels;
    std::vector<std::vector<int32_t>> members;
};

Components connected_components(
    int32_t n, const std::vector<std::pair<int32_t, int32_t>>& edges) {
    DisjointSet sets(n);
    for (const auto& edge : edges) sets.join(edge.first, edge.second);
    std::vector<int32_t> root_label(static_cast<size_t>(n), -1);
    Components out;
    out.labels.resize(static_cast<size_t>(n));
    for (int32_t node = 0; node < n; ++node) {
        const int32_t root = sets.find(node);
        int32_t& label = root_label[static_cast<size_t>(root)];
        if (label < 0) {
            label = static_cast<int32_t>(out.members.size());
            out.members.emplace_back();
        }
        out.labels[static_cast<size_t>(node)] = label;
        out.members[static_cast<size_t>(label)].push_back(node);
    }
    return out;
}

double positive_median(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot estimate a positive graph bandwidth");
    }
    const size_t middle = values.size() / 2;
    std::nth_element(values.begin(), values.begin() + middle, values.end());
    if (values.size() % 2 == 1) return values[middle];
    const double upper = values[middle];
    const double lower = *std::max_element(values.begin(), values.begin() + middle);
    return 0.5 * (lower + upper);
}

double sampled_quantile(const std::vector<double>& values, double probability,
                        int64_t maximum_sample, int64_t& sample_size) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot estimate a bridge weight floor");
    }
    const size_t wanted = std::min(values.size(),
        static_cast<size_t>(maximum_sample));
    std::vector<double> sample;
    sample.reserve(wanted);
    if (wanted == values.size()) {
        sample = values;
    } else {
        for (size_t j = 0; j < wanted; ++j) {
            const size_t index = static_cast<size_t>(
                static_cast<long double>(j) * values.size() / wanted);
            sample.push_back(values[index]);
        }
    }
    sample_size = static_cast<int64_t>(sample.size());
    const size_t position = static_cast<size_t>(std::floor(
        probability * static_cast<double>(sample.size() - 1)));
    std::nth_element(sample.begin(), sample.begin() + position, sample.end());
    return sample[position];
}

std::vector<double> exact_percentiles_1_to_99(std::vector<double> values) {
    if (values.empty()) {
        throw std::invalid_argument("Cannot summarize empty graph bandwidths");
    }
    std::sort(values.begin(), values.end());
    std::vector<double> out(99);
    for (size_t percentile = 1; percentile <= 99; ++percentile) {
        const long double position = static_cast<long double>(percentile)
            * static_cast<long double>(values.size() - 1) / 100.0L;
        const size_t lower = static_cast<size_t>(std::floor(position));
        const size_t upper = static_cast<size_t>(std::ceil(position));
        const double fraction = static_cast<double>(
            position - static_cast<long double>(lower));
        out[percentile - 1] = values[lower]
            + fraction * (values[upper] - values[lower]);
    }
    return out;
}

double hellinger_distance_squared(
    const RowMajorMatrixXd& coordinates, int32_t first, int32_t second) {
    return std::clamp(1.0 - coordinates.row(first).dot(
        coordinates.row(second)), 0.0, 1.0);
}

struct ComponentLink {
    int32_t first = -1;
    int32_t second = -1;
    int32_t first_node = -1;
    int32_t second_node = -1;
    double distance_squared = std::numeric_limits<double>::infinity();
};

using BridgeIndex = nanoflann::KDTreeEigenMatrixAdaptor<
    RowMajorMatrixXd, -1, nanoflann::metric_L2, true>;

class CrossComponentResultSet {
public:
    CrossComponentResultSet(const std::vector<int32_t>& node_groups,
                            int32_t forbidden_group)
        : node_groups_(node_groups), forbidden_group_(forbidden_group) {}

    size_t size() const { return index_ >= 0 ? 1 : 0; }
    bool empty() const { return index_ < 0; }
    bool full() const { return index_ >= 0; }
    void sort() {}

    bool addPoint(double distance, Eigen::Index index) {
        if (node_groups_[static_cast<size_t>(index)] == forbidden_group_) {
            return true;
        }
        const int32_t candidate = static_cast<int32_t>(index);
        if (distance < distance_
            || (distance == distance_ && candidate < index_)) {
            distance_ = distance;
            index_ = candidate;
        }
        return true;
    }

    double worstDist() const { return distance_; }
    int32_t index() const { return index_; }
    double distance() const { return distance_; }

private:
    const std::vector<int32_t>& node_groups_;
    int32_t forbidden_group_ = -1;
    int32_t index_ = -1;
    double distance_ = std::numeric_limits<double>::infinity();
};

bool component_link_less(
        const ComponentLink& first, const ComponentLink& second) {
    if (first.distance_squared != second.distance_squared) {
        return first.distance_squared < second.distance_squared;
    }
    if (first.first_node != second.first_node) {
        return first.first_node < second.first_node;
    }
    return first.second_node < second.second_node;
}

std::vector<ComponentLink> component_mst(
        const RowMajorMatrixXd& coordinates, const Components& components) {
    const int32_t count = static_cast<int32_t>(components.members.size());
    std::vector<ComponentLink> links;
    if (count <= 1) return links;
    BridgeIndex index(static_cast<int32_t>(coordinates.cols()),
        std::cref(coordinates), 10);
    DisjointSet forest(count);
    int32_t groups = count;
    while (groups > 1) {
        std::vector<int32_t> component_groups(static_cast<size_t>(count));
        for (int32_t component = 0; component < count; ++component) {
            component_groups[static_cast<size_t>(component)] =
                forest.find(component);
        }
        std::vector<int32_t> node_groups(static_cast<size_t>(coordinates.rows()));
        for (int32_t node = 0; node < coordinates.rows(); ++node) {
            node_groups[static_cast<size_t>(node)] = component_groups[
                static_cast<size_t>(components.labels[static_cast<size_t>(node)])];
        }
        std::vector<ComponentLink> best(static_cast<size_t>(count));
        for (int32_t node = 0; node < coordinates.rows(); ++node) {
            const int32_t group = node_groups[static_cast<size_t>(node)];
            CrossComponentResultSet result(node_groups, group);
            index.index_->findNeighbors(result, coordinates.row(node).data(),
                nanoflann::SearchParameters());
            if (result.empty()) continue;
            int32_t first_node = node;
            int32_t second_node = result.index();
            if (second_node < first_node) std::swap(first_node, second_node);
            ComponentLink candidate;
            candidate.first_node = first_node;
            candidate.second_node = second_node;
            candidate.first = components.labels[static_cast<size_t>(first_node)];
            candidate.second = components.labels[static_cast<size_t>(second_node)];
            candidate.distance_squared = 0.5 * result.distance();
            ComponentLink& retained = best[static_cast<size_t>(group)];
            if (component_link_less(candidate, retained)) retained = candidate;
        }
        std::vector<ComponentLink> round;
        for (int32_t component = 0; component < count; ++component) {
            if (forest.find(component) != component) continue;
            const ComponentLink& candidate = best[static_cast<size_t>(component)];
            if (candidate.first >= 0) round.push_back(candidate);
        }
        std::sort(round.begin(), round.end(), component_link_less);
        int32_t joined = 0;
        for (const ComponentLink& candidate : round) {
            if (forest.join(candidate.first, candidate.second)) {
                links.push_back(candidate);
                --groups;
                ++joined;
            }
        }
        if (joined == 0) {
            throw std::runtime_error("Component MST could not find an outgoing edge");
        }
    }
    return links;
}

struct RankedNode {
    double distance = 0.0;
    int32_t node = -1;
};

struct RankedNodeBetter {
    bool operator()(const RankedNode& first, const RankedNode& second) const {
        return first.distance < second.distance
            || (first.distance == second.distance && first.node < second.node);
    }
};

std::vector<int32_t> nearest_to_centroid(
    const RowMajorMatrixXd& coordinates,
    const std::vector<int32_t>& members,
    const Eigen::Ref<const Eigen::RowVectorXd>& centroid,
    int32_t limit) {
    std::priority_queue<RankedNode, std::vector<RankedNode>, RankedNodeBetter>
        retained;
    for (int32_t node : members) {
        const RankedNode candidate{
            (coordinates.row(node) - centroid).squaredNorm(), node};
        if (static_cast<int32_t>(retained.size()) < limit) {
            retained.push(candidate);
        } else if (candidate.distance < retained.top().distance
                   || (candidate.distance == retained.top().distance
                       && candidate.node < retained.top().node)) {
            retained.pop();
            retained.push(candidate);
        }
    }
    std::vector<RankedNode> ordered;
    ordered.reserve(retained.size());
    while (!retained.empty()) {
        ordered.push_back(retained.top());
        retained.pop();
    }
    std::sort(ordered.begin(), ordered.end(),
        [](const RankedNode& first, const RankedNode& second) {
            if (first.distance != second.distance) {
                return first.distance < second.distance;
            }
            return first.node < second.node;
        });
    std::vector<int32_t> out;
    out.reserve(ordered.size());
    for (const RankedNode& item : ordered) out.push_back(item.node);
    return out;
}

struct BridgeCandidate {
    DiffusionBridge bridge;
};

std::vector<BridgeCandidate> select_bridges(
    const RowMajorMatrixXd& coordinates, const Components& components,
    const RowMajorMatrixXd& centroids, const ComponentLink& link,
    int32_t requested, int32_t shortlist_size) {
    const std::vector<int32_t> first = nearest_to_centroid(coordinates,
        components.members[static_cast<size_t>(link.first)],
        centroids.row(link.second), shortlist_size);
    const std::vector<int32_t> second = nearest_to_centroid(coordinates,
        components.members[static_cast<size_t>(link.second)],
        centroids.row(link.first), shortlist_size);
    std::vector<BridgeCandidate> candidates;
    candidates.reserve(first.size() * second.size());
    for (int32_t first_node : first) {
        for (int32_t second_node : second) {
            const double distance = hellinger_distance_squared(
                coordinates, first_node, second_node);
            DiffusionBridge bridge;
            bridge.first = std::min(first_node, second_node);
            bridge.second = std::max(first_node, second_node);
            bridge.first_component = first_node <= second_node
                ? link.first : link.second;
            bridge.second_component = first_node <= second_node
                ? link.second : link.first;
            bridge.hellinger_distance_squared = distance;
            candidates.push_back({bridge});
        }
    }
    if (std::find(first.begin(), first.end(), link.first_node) == first.end()
        || std::find(second.begin(), second.end(), link.second_node)
            == second.end()) {
        DiffusionBridge bridge;
        bridge.first = std::min(link.first_node, link.second_node);
        bridge.second = std::max(link.first_node, link.second_node);
        bridge.first_component = link.first_node <= link.second_node
            ? link.first : link.second;
        bridge.second_component = link.first_node <= link.second_node
            ? link.second : link.first;
        bridge.hellinger_distance_squared = link.distance_squared;
        candidates.push_back({bridge});
    }
    std::sort(candidates.begin(), candidates.end(),
        [](const BridgeCandidate& first, const BridgeCandidate& second) {
            if (first.bridge.hellinger_distance_squared
                    != second.bridge.hellinger_distance_squared) {
                return first.bridge.hellinger_distance_squared
                    < second.bridge.hellinger_distance_squared;
            }
            if (first.bridge.first != second.bridge.first) {
                return first.bridge.first < second.bridge.first;
            }
            return first.bridge.second < second.bridge.second;
        });
    std::vector<int32_t> used_nodes;
    used_nodes.reserve(static_cast<size_t>(2 * requested));
    std::vector<BridgeCandidate> selected;
    selected.reserve(static_cast<size_t>(requested));
    for (const BridgeCandidate& candidate : candidates) {
        const int32_t first_node = candidate.bridge.first;
        const int32_t second_node = candidate.bridge.second;
        if (std::find(used_nodes.begin(), used_nodes.end(), first_node)
                != used_nodes.end()
            || std::find(used_nodes.begin(), used_nodes.end(), second_node)
                != used_nodes.end()) continue;
        used_nodes.push_back(first_node);
        used_nodes.push_back(second_node);
        selected.push_back(candidate);
        if (static_cast<int32_t>(selected.size()) == requested) break;
    }
    if (selected.empty()) {
        throw std::runtime_error("Failed to select a component bridge");
    }
    return selected;
}

DiffusionOperatorView normalize_operator_view(
    int32_t n_nodes,
    const std::vector<std::pair<int32_t, int32_t>>& edges,
    std::vector<double> kernel_weights, double alpha, double beta) {
    if (kernel_weights.size() != edges.size()) {
        throw std::logic_error("Diffusion kernel is not aligned with its edges");
    }
    DiffusionOperatorView out;
    out.alpha = alpha;
    out.beta = beta;
    out.kernel_weights = std::move(kernel_weights);
    out.kernel_degree.assign(static_cast<size_t>(n_nodes), 0.0);
    for (size_t edge = 0; edge < edges.size(); ++edge) {
        const auto endpoints = edges[edge];
        const double weight = out.kernel_weights[edge];
        if (!(weight >= 0.0) || !std::isfinite(weight)) {
            throw std::runtime_error("Diffusion graph has an invalid kernel weight");
        }
        out.kernel_degree[static_cast<size_t>(endpoints.first)] += weight;
        out.kernel_degree[static_cast<size_t>(endpoints.second)] += weight;
    }
    for (double degree : out.kernel_degree) {
        if (!(degree > 0.0) || !std::isfinite(degree)) {
            throw std::runtime_error("Diffusion graph has a nonpositive kernel degree");
        }
    }

    out.diffusion_weights.resize(edges.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, edges.size(), 4096),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t edge = range.begin(); edge < range.end(); ++edge) {
                const auto endpoints = edges[edge];
                const double denominator = std::pow(out.kernel_degree[
                    static_cast<size_t>(endpoints.first)], alpha)
                    * std::pow(out.kernel_degree[
                        static_cast<size_t>(endpoints.second)], alpha);
                out.diffusion_weights[edge] =
                    out.kernel_weights[edge] / denominator;
            }
        });
    out.diffusion_degree.assign(static_cast<size_t>(n_nodes), 0.0);
    for (size_t edge = 0; edge < edges.size(); ++edge) {
        const auto endpoints = edges[edge];
        const double weight = out.diffusion_weights[edge];
        out.diffusion_degree[static_cast<size_t>(endpoints.first)] += weight;
        out.diffusion_degree[static_cast<size_t>(endpoints.second)] += weight;
    }
    out.node_mass.resize(static_cast<size_t>(n_nodes));
    out.stationary_probability.resize(static_cast<size_t>(n_nodes));
    double total_mass = 0.0;
    for (int32_t node = 0; node < n_nodes; ++node) {
        const double mass = std::pow(out.diffusion_degree[
            static_cast<size_t>(node)], beta + 1.0);
        if (!(mass > 0.0) || !std::isfinite(mass)) {
            throw std::runtime_error("Diffusion graph has an invalid node mass");
        }
        out.node_mass[static_cast<size_t>(node)] = mass;
        total_mass += mass;
    }
    if (!(total_mass > 0.0) || !std::isfinite(total_mass)) {
        throw std::runtime_error("Diffusion graph has an invalid total mass");
    }
    for (int32_t node = 0; node < n_nodes; ++node) {
        out.stationary_probability[static_cast<size_t>(node)] =
            out.node_mass[static_cast<size_t>(node)] / total_mass;
    }
    return out;
}

void validate_options(const DiffusionGraphOptions& options) {
    if (options.embedding_bandwidth_neighbor_rank < 0
        || !std::isfinite(options.embedding_bandwidth_minimum_ratio)
        || !(options.embedding_bandwidth_minimum_ratio > 0.0)
        || !std::isfinite(options.embedding_bandwidth_maximum_ratio)
        || options.embedding_bandwidth_maximum_ratio
            < options.embedding_bandwidth_minimum_ratio
        || !std::isfinite(options.embedding_alpha)
        || options.embedding_alpha < 0.0 || options.embedding_alpha > 1.0
        || !std::isfinite(options.embedding_beta)
        || !std::isfinite(options.bridge_weight_quantile)
        || options.bridge_weight_quantile < 0.0
        || options.bridge_weight_quantile > 1.0
        || options.bridges_per_component_link <= 0
        || options.bridge_shortlist_size <= 0
        || options.quantile_sample_size <= 0) {
        throw std::invalid_argument("Invalid diffusion graph options");
    }
}

} // namespace

HellingerKnnGraphOptions::HellingerKnnGraphOptions() {
    knn.n_neighbors = 30;
}

HellingerKnnGraph build_hellinger_knn_graph(
        const Eigen::Ref<const RowMajorMatrixXd>& observations,
        const HellingerKnnGraphOptions& options) {
    if (options.bridges_per_component_link <= 0
            || options.bridge_shortlist_size <= 0) {
        throw std::invalid_argument("Invalid Hellinger k-NN graph options");
    }
    CosineDirectedKnnResult directed = simplex_directed_knn(
        observations, SimplexMetric::Hellinger, options.knn);
    tbb::global_control parallelism(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(options.knn.n_threads));
    HellingerKnnGraph out;
    out.n_nodes = directed.graph.n_nodes;
    out.n_neighbors = directed.graph.n_neighbors;
    out.knn = std::move(directed.diagnostics);
    if (options.retain_diffusion_geometry) {
        out.directed_neighbor_indices.reserve(
            directed.graph.neighbors.size());
        out.directed_distance_squared.reserve(
            directed.graph.neighbors.size());
        for (const DirectedKnnNeighbor& neighbor : directed.graph.neighbors) {
            out.directed_neighbor_indices.push_back(neighbor.index);
            out.directed_distance_squared.push_back(std::clamp(
                1.0 - neighbor.similarity, 0.0, 1.0));
        }
    }

    KnnGraph support = union_max_knn_graph(directed.graph);
    out.edges = std::move(support.edges);
    RowMajorMatrixXd coordinates = simplex_metric_coordinates(
        observations, SimplexMetric::Hellinger);
    out.raw_affinities.resize(out.edges.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, out.edges.size(), 4096),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t edge = range.begin(); edge < range.end(); ++edge) {
                const auto endpoints = out.edges[edge];
                out.raw_affinities[edge] = 1.0 - hellinger_distance_squared(
                    coordinates, endpoints.first, endpoints.second);
            }
        });

    Components components = connected_components(out.n_nodes, out.edges);
    out.component_labels = components.labels;
    out.component_sizes.reserve(components.members.size());
    for (const auto& members : components.members) {
        out.component_sizes.push_back(static_cast<int32_t>(members.size()));
    }
    if (options.build_bridge_candidates && components.members.size() > 1) {
        RowMajorMatrixXd centroids = RowMajorMatrixXd::Zero(
            static_cast<Eigen::Index>(components.members.size()),
            coordinates.cols());
        for (size_t component = 0; component < components.members.size();
                ++component) {
            for (int32_t node : components.members[component]) {
                centroids.row(static_cast<Eigen::Index>(component)) +=
                    coordinates.row(node);
            }
            centroids.row(static_cast<Eigen::Index>(component)) /=
                static_cast<double>(components.members[component].size());
        }
        for (const ComponentLink& link : component_mst(
                coordinates, components)) {
            const std::vector<BridgeCandidate> selected = select_bridges(
                coordinates, components, centroids, link,
                options.bridges_per_component_link,
                options.bridge_shortlist_size);
            for (const BridgeCandidate& candidate : selected) {
                const DiffusionBridge& bridge = candidate.bridge;
                out.bridge_candidates.push_back({bridge.first, bridge.second,
                    bridge.first_component, bridge.second_component,
                    bridge.hellinger_distance_squared});
            }
        }
        std::sort(out.bridge_candidates.begin(),
            out.bridge_candidates.end(),
            [](const HellingerBridgeCandidate& first,
                    const HellingerBridgeCandidate& second) {
                return std::pair<int32_t, int32_t>{first.first, first.second}
                    < std::pair<int32_t, int32_t>{second.first, second.second};
            });
    }
    if (options.retain_diffusion_geometry) {
        out.coordinates = std::move(coordinates);
    }
    return out;
}

DiffusionGraphOptions::DiffusionGraphOptions() {
    knn.n_neighbors = 30;
}

DiffusionGraph build_hellinger_diffusion_graph(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DiffusionGraphOptions& options) {
    validate_options(options);
    CosineDirectedKnnResult directed = simplex_directed_knn(
        observations, SimplexMetric::Hellinger, options.knn);
    tbb::global_control parallelism(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(options.knn.n_threads));
    DiffusionGraph out;
    out.n_nodes = directed.graph.n_nodes;
    out.knn = std::move(directed.diagnostics);
    const int32_t neighbors = directed.graph.n_neighbors;
    const int32_t embedding_rank = options.embedding_bandwidth_neighbor_rank > 0
        ? options.embedding_bandwidth_neighbor_rank : neighbors;
    if (embedding_rank > neighbors) {
        throw std::invalid_argument(
            "Diffusion bandwidth neighbor rank exceeds graph k");
    }
    std::vector<double> embedding_bandwidth_values;
    embedding_bandwidth_values.reserve(static_cast<size_t>(out.n_nodes));
    out.embedding_bandwidth_squared.resize(static_cast<size_t>(out.n_nodes));
    for (int32_t node = 0; node < out.n_nodes; ++node) {
        const size_t offset = static_cast<size_t>(node) * neighbors;
        const double embedding_distance = std::clamp(
            1.0 - directed.graph.neighbors[
                offset + static_cast<size_t>(embedding_rank - 1)].similarity,
            0.0, 1.0);
        out.embedding_bandwidth_squared[static_cast<size_t>(node)] =
            embedding_distance;
        if (embedding_distance > 0.0) {
            embedding_bandwidth_values.push_back(embedding_distance);
        }
    }
    if (embedding_bandwidth_values.empty()) {
        std::vector<double> all_positive;
        for (const DirectedKnnNeighbor& neighbor : directed.graph.neighbors) {
            const double distance = std::clamp(
                1.0 - neighbor.similarity, 0.0, 1.0);
            if (distance > 0.0) all_positive.push_back(distance);
        }
        embedding_bandwidth_values = std::move(all_positive);
    }
    out.diagnostics.embedding_bandwidth_neighbor_rank = embedding_rank;
    out.diagnostics.embedding_bandwidth_sample_size = static_cast<int64_t>(
        embedding_bandwidth_values.size());
    out.diagnostics.embedding_median_bandwidth_squared = positive_median(
        std::move(embedding_bandwidth_values));
    if (!(out.diagnostics.embedding_median_bandwidth_squared > 0.0)
        || !std::isfinite(
            out.diagnostics.embedding_median_bandwidth_squared)) {
        throw std::invalid_argument("Diffusion graph bandwidth is not positive");
    }
    out.diagnostics.embedding_minimum_bandwidth_squared =
        out.diagnostics.embedding_median_bandwidth_squared
        * std::pow(options.embedding_bandwidth_minimum_ratio, 2.0);
    out.diagnostics.embedding_maximum_bandwidth_squared =
        out.diagnostics.embedding_median_bandwidth_squared
        * std::pow(options.embedding_bandwidth_maximum_ratio, 2.0);
    for (double& value : out.embedding_bandwidth_squared) {
        if (value < out.diagnostics.embedding_minimum_bandwidth_squared) {
            value = out.diagnostics.embedding_minimum_bandwidth_squared;
            ++out.diagnostics.embedding_bandwidth_floor_count;
        } else if (value
            > out.diagnostics.embedding_maximum_bandwidth_squared) {
            value = out.diagnostics.embedding_maximum_bandwidth_squared;
            ++out.diagnostics.embedding_bandwidth_cap_count;
        }
    }
    std::vector<double> local_bandwidths(
        out.embedding_bandwidth_squared.size());
    std::transform(out.embedding_bandwidth_squared.begin(),
        out.embedding_bandwidth_squared.end(), local_bandwidths.begin(),
        [](double value) { return std::sqrt(value); });
    out.diagnostics.embedding_bandwidth_percentiles =
        exact_percentiles_1_to_99(std::move(local_bandwidths));

    KnnGraph support = union_max_knn_graph(directed.graph);
    directed.graph.neighbors.clear();
    directed.graph.neighbors.shrink_to_fit();
    const RowMajorMatrixXd coordinates = simplex_metric_coordinates(
        observations, SimplexMetric::Hellinger);
    std::vector<double> embedding_base_weights(support.edges.size());
    std::vector<double> raw_affinities(support.edges.size());
    tbb::parallel_for(tbb::blocked_range<size_t>(0, support.edges.size(), 4096),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t edge = range.begin(); edge < range.end(); ++edge) {
                const auto endpoints = support.edges[edge];
                const double distance = hellinger_distance_squared(
                    coordinates, endpoints.first, endpoints.second);
                const double embedding_denominator = std::sqrt(
                    out.embedding_bandwidth_squared[
                        static_cast<size_t>(endpoints.first)]
                    * out.embedding_bandwidth_squared[
                        static_cast<size_t>(endpoints.second)]);
                embedding_base_weights[edge] = std::exp(
                    -distance / embedding_denominator);
                raw_affinities[edge] = 1.0 - distance;
            }
        });
    std::vector<double> positive_weights;
    positive_weights.reserve(embedding_base_weights.size());
    for (double weight : embedding_base_weights) {
        if (weight > 0.0 && std::isfinite(weight)) {
            positive_weights.push_back(weight);
        }
    }
    out.diagnostics.embedding_bridge_weight_floor = sampled_quantile(
        positive_weights, options.bridge_weight_quantile,
        options.quantile_sample_size, out.diagnostics.quantile_sample_size);

    Components components = connected_components(out.n_nodes, support.edges);
    out.diagnostics.initial_components = static_cast<int32_t>(
        components.members.size());
    out.diagnostics.initial_component_sizes.reserve(components.members.size());
    for (const auto& members : components.members) {
        out.diagnostics.initial_component_sizes.push_back(
            static_cast<int32_t>(members.size()));
    }
    std::vector<BridgeCandidate> bridge_candidates;
    if (components.members.size() > 1) {
        std::vector<double> component_volume(components.members.size(), 0.0);
        for (size_t edge = 0; edge < support.edges.size(); ++edge) {
            const int32_t component = components.labels[
                static_cast<size_t>(support.edges[edge].first)];
            component_volume[static_cast<size_t>(component)] +=
                2.0 * embedding_base_weights[edge];
        }
        RowMajorMatrixXd centroids = RowMajorMatrixXd::Zero(
            static_cast<Eigen::Index>(components.members.size()),
            coordinates.cols());
        for (size_t component = 0; component < components.members.size();
             ++component) {
            for (int32_t node : components.members[component]) {
                centroids.row(static_cast<Eigen::Index>(component)) +=
                    coordinates.row(node);
            }
            centroids.row(static_cast<Eigen::Index>(component)) /=
                static_cast<double>(components.members[component].size());
        }
        for (const ComponentLink& link : component_mst(
                 coordinates, components)) {
            std::vector<BridgeCandidate> selected = select_bridges(
                coordinates, components, centroids, link,
                options.bridges_per_component_link,
                options.bridge_shortlist_size);
            double bridge_volume = 0.0;
            for (BridgeCandidate& candidate : selected) {
                DiffusionBridge& item = candidate.bridge;
                const double denominator = std::sqrt(
                    out.embedding_bandwidth_squared[
                        static_cast<size_t>(item.first)]
                    * out.embedding_bandwidth_squared[
                        static_cast<size_t>(item.second)]);
                item.embedding_geometric_weight = std::exp(
                    -item.hellinger_distance_squared / denominator);
                item.embedding_kernel_weight = std::max(
                    item.embedding_geometric_weight,
                    out.diagnostics.embedding_bridge_weight_floor);
                item.embedding_geometric_weight_underflow =
                    item.embedding_geometric_weight == 0.0;
                item.embedding_weight_inflation =
                    item.embedding_geometric_weight > 0.0
                    ? item.embedding_kernel_weight
                        / item.embedding_geometric_weight
                    : std::numeric_limits<double>::max();
                bridge_volume += item.embedding_kernel_weight;
            }
            const double local_volume = std::min(
                component_volume[static_cast<size_t>(link.first)],
                component_volume[static_cast<size_t>(link.second)]);
            const double conductance = local_volume > 0.0
                ? bridge_volume / local_volume
                : std::numeric_limits<double>::infinity();
            for (BridgeCandidate& candidate : selected) {
                candidate.bridge.component_pair_conductance = conductance;
            }
            out.diagnostics.maximum_bridge_pair_conductance = std::max(
                out.diagnostics.maximum_bridge_pair_conductance, conductance);
            bridge_candidates.insert(bridge_candidates.end(),
                selected.begin(), selected.end());
        }
    }
    std::sort(bridge_candidates.begin(), bridge_candidates.end(),
        [](const BridgeCandidate& first, const BridgeCandidate& second) {
            return std::pair<int32_t, int32_t>{first.bridge.first,
                first.bridge.second}
                < std::pair<int32_t, int32_t>{second.bridge.first,
                    second.bridge.second};
        });

    out.edges.reserve(support.edges.size() + bridge_candidates.size());
    std::vector<double> embedding_kernel_weights;
    embedding_kernel_weights.reserve(
        embedding_base_weights.size() + bridge_candidates.size());
    out.raw_affinities.reserve(
        raw_affinities.size() + bridge_candidates.size());
    out.edge_is_bridge.reserve(
        embedding_base_weights.size() + bridge_candidates.size());
    size_t base = 0;
    size_t bridge = 0;
    while (base < support.edges.size() || bridge < bridge_candidates.size()) {
        const bool take_bridge = base == support.edges.size()
            || (bridge < bridge_candidates.size()
                && std::pair<int32_t, int32_t>{
                    bridge_candidates[bridge].bridge.first,
                    bridge_candidates[bridge].bridge.second}
                    < support.edges[base]);
        if (take_bridge) {
            const DiffusionBridge& item = bridge_candidates[bridge].bridge;
            out.edges.emplace_back(item.first, item.second);
            embedding_kernel_weights.push_back(item.embedding_kernel_weight);
            out.raw_affinities.push_back(
                1.0 - item.hellinger_distance_squared);
            out.edge_is_bridge.push_back(1);
            out.bridges.push_back(item);
            ++bridge;
        } else {
            if (bridge < bridge_candidates.size()
                && support.edges[base] == std::pair<int32_t, int32_t>{
                    bridge_candidates[bridge].bridge.first,
                    bridge_candidates[bridge].bridge.second}) {
                throw std::logic_error("Component bridge duplicates a k-NN edge");
            }
            out.edges.push_back(support.edges[base]);
            embedding_kernel_weights.push_back(embedding_base_weights[base]);
            out.raw_affinities.push_back(raw_affinities[base]);
            out.edge_is_bridge.push_back(0);
            ++base;
        }
    }
    out.diagnostics.base_edges = static_cast<int64_t>(support.edges.size());
    out.diagnostics.bridge_edges = static_cast<int64_t>(out.bridges.size());
    out.diagnostics.final_components = static_cast<int32_t>(
        connected_components(out.n_nodes, out.edges).members.size());
    if (out.diagnostics.final_components != 1) {
        throw std::runtime_error("Diffusion graph bridge repair did not connect the graph");
    }

    out.embedding = normalize_operator_view(
        out.n_nodes, out.edges, std::move(embedding_kernel_weights),
        options.embedding_alpha, options.embedding_beta);
    return out;
}
