#include "multires_core/metric_graph.hpp"

#include "multires_core/raw_affinity_graph.hpp"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace punkst::multires {

namespace {

using Clock = std::chrono::steady_clock;

double elapsed(const Clock::time_point& begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

DiffusionGraph coarsening_adapter(const HellingerKnnGraph& graph) {
    DiffusionGraph adapted;
    adapted.n_nodes = graph.n_nodes;
    adapted.edges = graph.edges;
    adapted.edge_is_bridge.assign(graph.edges.size(), 0);
    adapted.raw_affinities = graph.raw_affinities;
    // The legacy coarsening container requires an embedding channel, although
    // matching and the public result use only raw affinity. A unit channel and
    // uniform node masses make the adapter behavior diffusion-independent.
    adapted.embedding.diffusion_weights.assign(graph.edges.size(), 1.0);
    adapted.embedding.node_mass.assign(static_cast<size_t>(graph.n_nodes), 1.0);
    return adapted;
}

void write_identifier_table(const fs::path& path,
        const std::vector<std::string>& identifiers) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write identifiers: " + path.string());
    output << "row\tid\n";
    for (size_t row = 0; row < identifiers.size(); ++row) {
        output << row << '\t' << identifiers[row] << '\n';
    }
    if (!output) throw std::runtime_error("Failed writing identifiers: " + path.string());
}

void write_factor_table(const fs::path& path, const ThetaTable& theta) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write factors: " + path.string());
    output << std::setprecision(17)
        << "input_factor_index\tsource_column\tfactor\trelative_weight"
           "\tretained\toutput_factor_index\n";
    std::vector<int32_t> output_index(
        theta.factor_filter.input_factor_names.size(), -1);
    for (size_t index = 0; index < theta.factor_filter.retained_indices.size();
            ++index) {
        output_index[static_cast<size_t>(
            theta.factor_filter.retained_indices[index])] =
            static_cast<int32_t>(index);
    }
    for (size_t factor = 0;
            factor < theta.factor_filter.input_factor_names.size(); ++factor) {
        output << factor << '\t'
            << theta.factor_filter.input_factor_columns[factor] << '\t'
            << theta.factor_filter.input_factor_names[factor] << '\t'
            << theta.factor_filter.relative_weights[factor] << '\t'
            << static_cast<int32_t>(output_index[factor] >= 0) << '\t';
        if (output_index[factor] >= 0) output << output_index[factor];
        else output << '.';
        output << '\n';
    }
    if (!output) throw std::runtime_error("Failed writing factors: " + path.string());
}

void write_coarsening_tables(const fs::path& root,
        const std::vector<std::string>& identifiers,
        const GraphCoarseningResult& coarse) {
    const fs::path membership_path = root / "coarsening/membership.tsv";
    fs::create_directories(membership_path.parent_path());
    std::ofstream membership_output(membership_path);
    if (!membership_output) {
        throw std::runtime_error(
            "Cannot write coarsening membership: "
            + membership_path.string());
    }
    membership_output << "id\tmicrocluster\n";
    for (size_t row = 0; row < identifiers.size(); ++row) {
        membership_output << identifiers[row] << '\t'
            << coarse.membership[row] << '\n';
    }
    if (!membership_output) {
        throw std::runtime_error(
            "Failed writing coarsening membership: "
            + membership_path.string());
    }

    const fs::path representative_path =
        root / "coarsening/representatives.tsv";
    std::ofstream representative_output(representative_path);
    if (!representative_output) {
        throw std::runtime_error(
            "Cannot write coarsening representatives: "
            + representative_path.string());
    }
    representative_output << "microcluster\trepresentative_id\n";
    for (size_t cluster = 0; cluster < coarse.representatives.size();
            ++cluster) {
        const int32_t row = coarse.representatives[cluster];
        representative_output << cluster << '\t'
            << identifiers[static_cast<size_t>(row)] << '\n';
    }
    if (!representative_output) {
        throw std::runtime_error(
            "Failed writing coarsening representatives: "
            + representative_path.string());
    }
}

json knn_diagnostics_json(const CosineKnnDiagnostics& diagnostics) {
    json trials = json::array();
    for (const CosineKnnAuditTrial& trial : diagnostics.audit_trials) {
        trials.push_back({{"parameter", trial.parameter},
            {"mean_recall", trial.mean_recall},
            {"recall_lcb", trial.recall_lcb}});
    }
    return {
        {"requested_backend", cosine_knn_backend_name(
            diagnostics.requested_backend)},
        {"resolved_backend", cosine_knn_backend_name(
            diagnostics.resolved_backend)},
        {"resolved_flat_kernel", cosine_flat_kernel_name(
            diagnostics.resolved_flat_kernel)},
        {"sample_size", diagnostics.sample_size},
        {"requested_ann_parameter", diagnostics.requested_ann_parameter},
        {"resolved_ann_parameter", diagnostics.resolved_ann_parameter},
        {"resolved_ann_candidates", diagnostics.resolved_ann_candidates},
        {"audit_mean_recall", diagnostics.audit_mean_recall},
        {"audit_recall_lcb", diagnostics.audit_recall_lcb},
        {"audit_passed", diagnostics.audit_passed},
        {"forced", diagnostics.forced},
        {"audit_trials", trials},
        {"timings", {
            {"normalization", diagnostics.timings.normalization_seconds},
            {"index_build", diagnostics.timings.index_build_seconds},
            {"query", diagnostics.timings.query_seconds},
            {"topk", diagnostics.timings.topk_seconds},
            {"graph_reduction", diagnostics.timings.graph_reduction_seconds},
            {"audit", diagnostics.timings.audit_seconds}
        }}
    };
}

RawClusteringGraph fine_raw_graph(const HellingerKnnGraph& graph) {
    DiffusionGraph adapted;
    adapted.n_nodes = graph.n_nodes;
    adapted.edges = graph.edges;
    adapted.raw_affinities = graph.raw_affinities;
    adapted.edge_is_bridge.assign(graph.edges.size(), 0);
    return make_raw_clustering_graph(adapted);
}

json write_raw_graph(const fs::path& root, const std::string& prefix,
        const RawClusteringGraph& graph,
        const std::vector<int32_t>& fine_counts,
        std::vector<json>& arrays) {
    std::vector<int32_t> rows;
    std::vector<int32_t> columns;
    rows.reserve(graph.edges.size());
    columns.reserve(graph.edges.size());
    for (const auto& edge : graph.edges) {
        rows.push_back(edge.first);
        columns.push_back(edge.second);
    }
    const json row_spec = write_array(root, prefix + "/edge_rows.i32", rows);
    const json column_spec = write_array(
        root, prefix + "/edge_columns.i32", columns);
    const json weight_spec = write_array(
        root, prefix + "/edge_weights.f64", graph.weights);
    const json component_spec = write_array(
        root, prefix + "/component_labels.i32", graph.component_labels);
    const json count_spec = write_array(
        root, prefix + "/fine_point_counts.i32", fine_counts);
    arrays.insert(arrays.end(), {row_spec, column_spec, weight_spec,
        component_spec, count_spec});
    return {{"nodes", graph.n_nodes}, {"edges", graph.edges.size()},
        {"edge_rows", row_spec}, {"edge_columns", column_spec},
        {"edge_weights", weight_spec}, {"component_labels", component_spec},
        {"fine_point_counts", count_spec},
        {"self_loop_edges", graph.self_loop_edges},
        {"total_affinity", graph.total_affinity}};
}

json write_canonical_graph(const fs::path& root,
        const HellingerKnnGraph& graph, std::vector<json>& arrays) {
    std::vector<int32_t> rows;
    std::vector<int32_t> columns;
    rows.reserve(graph.edges.size());
    columns.reserve(graph.edges.size());
    for (const auto& edge : graph.edges) {
        rows.push_back(edge.first);
        columns.push_back(edge.second);
    }
    const json row_spec = write_array(root, "graph/edge_rows.i32", rows);
    const json column_spec = write_array(
        root, "graph/edge_columns.i32", columns);
    const json affinity_spec = write_array(
        root, "graph/raw_affinities.f64", graph.raw_affinities);
    arrays.insert(arrays.end(), {row_spec, column_spec, affinity_spec});
    return {{"nodes", graph.n_nodes}, {"neighbors", graph.n_neighbors},
        {"edges", graph.edges.size()}, {"edge_rows", row_spec},
        {"edge_columns", column_spec}, {"raw_affinities", affinity_spec},
        {"knn", knn_diagnostics_json(graph.knn)}};
}

json write_diffusion_sidecar(const fs::path& root,
        const HellingerKnnGraph& graph, std::vector<json>& arrays) {
    const json neighbor_spec = write_array(root,
        "diffusion_input/directed_neighbor_indices.i32",
        graph.directed_neighbor_indices);
    const json distance_spec = write_array(root,
        "diffusion_input/directed_distance_squared.f64",
        graph.directed_distance_squared);
    const json coordinate_spec = write_array(root,
        "diffusion_input/hellinger_coordinates.f64", graph.coordinates);
    const json support_component_spec = write_array(root,
        "diffusion_input/support_component_labels.i32",
        graph.component_labels);
    std::vector<int32_t> bridge_rows;
    std::vector<int32_t> bridge_columns;
    std::vector<int32_t> bridge_first_components;
    std::vector<int32_t> bridge_second_components;
    std::vector<double> bridge_distances;
    std::vector<double> bridge_raw_affinities;
    for (const HellingerBridgeCandidate& bridge : graph.bridge_candidates) {
        bridge_rows.push_back(bridge.first);
        bridge_columns.push_back(bridge.second);
        bridge_first_components.push_back(bridge.first_component);
        bridge_second_components.push_back(bridge.second_component);
        bridge_distances.push_back(bridge.hellinger_distance_squared);
        bridge_raw_affinities.push_back(
            1.0 - bridge.hellinger_distance_squared);
    }
    const json bridge_row_spec = write_array(root,
        "diffusion_input/bridge_rows.i32", bridge_rows);
    const json bridge_column_spec = write_array(root,
        "diffusion_input/bridge_columns.i32", bridge_columns);
    const json bridge_first_spec = write_array(root,
        "diffusion_input/bridge_first_components.i32",
        bridge_first_components);
    const json bridge_second_spec = write_array(root,
        "diffusion_input/bridge_second_components.i32",
        bridge_second_components);
    const json bridge_distance_spec = write_array(root,
        "diffusion_input/bridge_distance_squared.f64", bridge_distances);
    const json bridge_affinity_spec = write_array(root,
        "diffusion_input/bridge_raw_affinities.f64",
        bridge_raw_affinities);
    arrays.insert(arrays.end(), {neighbor_spec, distance_spec, coordinate_spec,
        support_component_spec, bridge_row_spec, bridge_column_spec,
        bridge_first_spec, bridge_second_spec, bridge_distance_spec,
        bridge_affinity_spec});
    return {
        {"directed_shape", {graph.n_nodes, graph.n_neighbors}},
        {"directed_neighbor_indices", neighbor_spec},
        {"directed_distance_squared", distance_spec},
        {"hellinger_coordinates", coordinate_spec},
        {"support_component_labels", support_component_spec},
        {"bridge_candidates", {
            {"count", graph.bridge_candidates.size()},
            {"rows", bridge_row_spec}, {"columns", bridge_column_spec},
            {"first_components", bridge_first_spec},
            {"second_components", bridge_second_spec},
            {"distance_squared", bridge_distance_spec},
            {"raw_affinities", bridge_affinity_spec}
        }}
    };
}

} // namespace

MetricGraphResult build_metric_graph(const fs::path& theta_path,
        const MetricGraphOptions& options) {
    const auto total_begin = Clock::now();
    MetricGraphResult out;
    auto begin = Clock::now();
    out.theta = read_theta_table(theta_path, options.theta);
    out.timings.input_seconds = elapsed(begin);
    begin = Clock::now();
    HellingerKnnGraphOptions graph_options = options.graph;
    graph_options.build_bridge_candidates = options.diffusion_sidecar;
    graph_options.retain_diffusion_geometry = options.diffusion_sidecar;
    out.graph = build_hellinger_knn_graph(out.theta.values, graph_options);
    out.timings.graph_seconds = elapsed(begin);
    if (options.coarsening_enabled) {
        begin = Clock::now();
        const DiffusionGraph adapted = coarsening_adapter(out.graph);
        GraphCoarseningResult coarsening = coarsen_diffusion_graph(
            adapted, out.theta.values, options.coarsening);
        // An inactive coarsening request is semantically the fine graph.  Do
        // not publish an identity membership or let downstream code mistake
        // it for a distinct coarse population.
        if (coarsening.diagnostics.activated) {
            out.coarsening = std::move(coarsening);
        }
        out.timings.coarsening_seconds = elapsed(begin);
    }
    out.timings.total_seconds = elapsed(total_begin);
    return out;
}

void write_metric_graph_artifact(const fs::path& output,
        const fs::path& theta_path, const json& resolved_request,
        const MetricGraphOptions& options, const MetricGraphResult& result) {
    const std::string source_sha256 = sha256_file(theta_path);
    publish_directory_atomic(output, [&](const fs::path& root) {
        std::vector<json> arrays;
        write_identifier_table(root / "identifiers.tsv", result.theta.identifiers);
        write_factor_table(root / "factors.tsv", result.theta);
        const std::string identifiers_sha256 = sha256_file(
            root / "identifiers.tsv");
        const std::string factors_sha256 = sha256_file(root / "factors.tsv");
        const json graph = write_canonical_graph(root, result.graph, arrays);
        const RawClusteringGraph fine_raw = fine_raw_graph(result.graph);
        const std::vector<int32_t> unit_counts(
            static_cast<size_t>(result.graph.n_nodes), 1);
        const json raw_graph = write_raw_graph(
            root, "raw_graph_fine", fine_raw, unit_counts, arrays);
        const json diffusion_input = options.diffusion_sidecar
            ? write_diffusion_sidecar(root, result.graph, arrays) : json(nullptr);

        json coarsening = nullptr;
        if (result.coarsening.has_value()) {
            const GraphCoarseningResult& coarse = *result.coarsening;
            write_coarsening_tables(root, result.theta.identifiers, coarse);
            const json membership = write_array(root,
                "coarsening/fine_to_microcluster.i32", coarse.membership);
            const json representatives = write_array(root,
                "coarsening/representative_rows.i32", coarse.representatives);
            arrays.insert(arrays.end(), {membership, representatives});
            const RawClusteringGraph coarse_raw = make_raw_clustering_graph(
                coarse.graph);
            const json coarse_raw_json = write_raw_graph(root,
                "raw_graph_coarse", coarse_raw,
                coarse.graph.fine_node_counts, arrays);
            coarsening = {
                {"membership", membership},
                {"representative_rows", representatives},
                {"membership_table", "coarsening/membership.tsv"},
                {"representatives_table",
                    "coarsening/representatives.tsv"},
                {"microcluster_sizes",
                    coarse_raw_json.at("fine_point_counts")},
                {"raw_graph", coarse_raw_json},
                {"representative_definition",
                    "unweighted_squared_hellinger_medoid"},
                {"diagnostics", {
                    {"strategy", coarse.diagnostics.strategy_name},
                    {"requested_target_nodes",
                        coarse.diagnostics.requested_target_nodes},
                    {"resolved_target_nodes",
                        coarse.diagnostics.resolved_target_nodes},
                    {"matching_passes", coarse.diagnostics.matching_passes},
                    {"accepted_matches", coarse.diagnostics.accepted_matches},
                    {"size_rejected_candidates",
                        coarse.diagnostics.size_rejected_candidates},
                    {"activated", coarse.diagnostics.activated},
                    {"target_reached", coarse.diagnostics.target_reached},
                    {"stopped_without_matches",
                        coarse.diagnostics.stopped_without_matches},
                    {"level_node_counts",
                        coarse.diagnostics.level_node_counts}
                }}
            };
        }
        json manifest = {
            {"artifact_type", "punkst.knn_graph"},
            {"schema_version", 1},
            {"resolved_request", resolved_request},
            {"source", {
                {"theta_path", fs::absolute(theta_path).string()},
                {"theta_sha256", source_sha256}
            }},
            {"population", {
                {"points", result.theta.values.rows()},
                {"input_factors",
                    result.theta.factor_filter.input_factor_names.size()},
                {"retained_factors", result.theta.values.cols()},
                {"factor_weight_threshold",
                    result.theta.factor_filter.threshold},
                {"identifiers_table", "identifiers.tsv"},
                {"identifiers_sha256", identifiers_sha256},
                {"factors_table", "factors.tsv"},
                {"factors_sha256", factors_sha256}
            }},
            {"graph", graph},
            {"raw_graph", raw_graph},
            {"diffusion_input", diffusion_input},
            {"coarsening", coarsening},
            {"timings", {
                {"input", result.timings.input_seconds},
                {"graph", result.timings.graph_seconds},
                {"coarsening", result.timings.coarsening_seconds},
                {"total", result.timings.total_seconds}
            }}
        };
        manifest["fingerprint"] = artifact_fingerprint(manifest, root, arrays);
        write_json(root / "manifest.json", manifest, 2);
    });
}

} // namespace punkst::multires
