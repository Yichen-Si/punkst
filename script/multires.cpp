#include "multires_core/artifacts.hpp"
#include "multires_core/metric_graph.hpp"
#include "punkst.h"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <limits>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using punkst::multires::json;
namespace fs = std::filesystem;

int32_t optional_column(const json& input, const char* name) {
    const auto found = input.find(name);
    if (found == input.end() || found->is_null()) return -1;
    return found->get<int32_t>();
}

struct ParsedMetricGraphRequest {
    fs::path theta_path;
    punkst::multires::MetricGraphOptions options;
    json resolved;
};

void parse_knn_options(const json& graph, CosineKnnOptions& options) {
    options.n_neighbors = graph.value("neighbors", options.n_neighbors);
    options.backend = parse_cosine_knn_backend(
        graph.value("knn_backend", std::string("auto")));
    options.knn_search_epsilon = graph.value(
        "knn_epsilon", options.knn_search_epsilon);
    options.flat_kernel = parse_cosine_flat_kernel(
        graph.value("flat_kernel", std::string("auto")));
    options.hnsw_m = graph.value("hnsw_m", options.hnsw_m);
    options.hnsw_ef_construction = graph.value(
        "hnsw_ef_construction", options.hnsw_ef_construction);
    options.hnsw_ef_search = graph.value(
        "hnsw_ef_search", options.hnsw_ef_search);
    options.hnsw_max_ef_search = graph.value(
        "hnsw_max_ef_search", options.hnsw_max_ef_search);
    options.hnsw_candidates = graph.value(
        "hnsw_candidates", options.hnsw_candidates);
    options.hnsw_audit_queries = graph.value(
        "hnsw_audit_queries", options.hnsw_audit_queries);
    options.hnsw_recall = graph.value("hnsw_recall", options.hnsw_recall);
    options.hnsw_force = graph.value("hnsw_force", options.hnsw_force);
    options.nndescent_iterations = graph.value(
        "nndescent_iterations", options.nndescent_iterations);
    options.nndescent_graph_size = graph.value(
        "nndescent_graph_size", options.nndescent_graph_size);
    options.nndescent_sample_candidates = graph.value(
        "nndescent_sample_candidates", options.nndescent_sample_candidates);
    options.nndescent_audit_queries = graph.value(
        "nndescent_audit_queries", options.nndescent_audit_queries);
    options.nndescent_recall = graph.value(
        "nndescent_recall", options.nndescent_recall);
    options.ann_seed = graph.value("ann_seed", options.ann_seed);
}

json resolved_knn_options(const HellingerKnnGraphOptions& graph) {
    const CosineKnnOptions& options = graph.knn;
    return {
        {"neighbors", options.n_neighbors},
        {"knn_backend", cosine_knn_backend_name(options.backend)},
        {"knn_epsilon", options.knn_search_epsilon},
        {"flat_kernel", cosine_flat_kernel_name(options.flat_kernel)},
        {"hnsw_m", options.hnsw_m},
        {"hnsw_ef_construction", options.hnsw_ef_construction},
        {"hnsw_ef_search", options.hnsw_ef_search},
        {"hnsw_max_ef_search", options.hnsw_max_ef_search},
        {"hnsw_candidates", options.hnsw_candidates},
        {"hnsw_audit_queries", options.hnsw_audit_queries},
        {"hnsw_recall", options.hnsw_recall},
        {"hnsw_force", options.hnsw_force},
        {"nndescent_iterations", options.nndescent_iterations},
        {"nndescent_graph_size", options.nndescent_graph_size},
        {"nndescent_sample_candidates", options.nndescent_sample_candidates},
        {"nndescent_audit_queries", options.nndescent_audit_queries},
        {"nndescent_recall", options.nndescent_recall},
        {"ann_seed", options.ann_seed},
        {"bridges_per_component_link", graph.bridges_per_component_link},
        {"bridge_shortlist_size", graph.bridge_shortlist_size}
    };
}

ParsedMetricGraphRequest parse_metric_graph_request(
        const fs::path& request_path, const json& request) {
    punkst::multires::reject_unknown_keys(request, {"artifact_type", "schema_version", "input",
        "graph", "coarsening", "diffusion_sidecar", "runtime"}, "request");
    if (request.value("artifact_type", "") != "punkst.knn_graph.request"
            || request.value("schema_version", 0) != 1) {
        throw std::invalid_argument(
            "Expected punkst.knn_graph.request schema version 1");
    }

    ParsedMetricGraphRequest out;
    out.resolved = request;
    const json& input = punkst::multires::require_object_field(request, "input");
    punkst::multires::reject_unknown_keys(input, {"theta_path", "identifier_column",
        "factor_column_start", "factor_column_end",
        "factor_weight_threshold"}, "input");
    if (!input.contains("theta_path")) {
        throw std::invalid_argument("input.theta_path is required");
    }
    out.theta_path = input.at("theta_path").get<std::string>();
    if (out.theta_path.is_relative()) {
        out.theta_path = request_path.parent_path() / out.theta_path;
    }
    out.theta_path = fs::absolute(out.theta_path).lexically_normal();
    out.options.theta.identifier_column = input.value("identifier_column", 0);
    out.options.theta.factor_column_start = optional_column(
        input, "factor_column_start");
    out.options.theta.factor_column_end = optional_column(
        input, "factor_column_end");
    out.options.theta.factor_weight_threshold = input.value(
        "factor_weight_threshold", 1e-5);

    int32_t threads = 1;
    if (request.contains("runtime")) {
        const json& runtime = punkst::multires::require_object_field(request, "runtime");
        punkst::multires::reject_unknown_keys(runtime, {"threads"}, "runtime");
        threads = runtime.value("threads", 1);
    }
    if (threads <= 0) {
        throw std::invalid_argument("runtime.threads must be positive");
    }

    if (request.contains("graph")) {
        const json& graph = punkst::multires::require_object_field(request, "graph");
        punkst::multires::reject_unknown_keys(graph, {"neighbors", "knn_backend", "knn_epsilon",
            "flat_kernel", "hnsw_m", "hnsw_ef_construction",
            "hnsw_ef_search", "hnsw_max_ef_search", "hnsw_candidates",
            "hnsw_audit_queries", "hnsw_recall", "hnsw_force",
            "nndescent_iterations", "nndescent_graph_size",
            "nndescent_sample_candidates", "nndescent_audit_queries",
            "nndescent_recall", "ann_seed", "bridges_per_component_link",
            "bridge_shortlist_size"}, "graph");
        parse_knn_options(graph, out.options.graph.knn);
        out.options.graph.bridges_per_component_link = graph.value(
            "bridges_per_component_link",
            out.options.graph.bridges_per_component_link);
        out.options.graph.bridge_shortlist_size = graph.value(
            "bridge_shortlist_size", out.options.graph.bridge_shortlist_size);
    }
    out.options.graph.knn.n_threads = threads;

    if (request.contains("coarsening")) {
        const json& coarsening = punkst::multires::require_object_field(request, "coarsening");
        punkst::multires::reject_unknown_keys(coarsening, {"enabled", "target_nodes",
            "activation_threshold", "target_minimum", "target_maximum",
            "target_divisor", "maximum_microcluster_size"}, "coarsening");
        out.options.coarsening_enabled = coarsening.value("enabled", false);
        auto& options = out.options.coarsening;
        options.target_nodes = coarsening.value(
            "target_nodes", options.target_nodes);
        options.activation_threshold = coarsening.value(
            "activation_threshold", options.activation_threshold);
        options.target_minimum = coarsening.value(
            "target_minimum", options.target_minimum);
        options.target_maximum = coarsening.value(
            "target_maximum", options.target_maximum);
        options.target_divisor = coarsening.value(
            "target_divisor", options.target_divisor);
        options.maximum_microcluster_size = coarsening.value(
            "maximum_microcluster_size", options.maximum_microcluster_size);
    }
    out.options.coarsening.n_threads = threads;
    out.options.diffusion_sidecar = request.value("diffusion_sidecar", false);

    out.resolved["input"] = {
        {"theta_path", out.theta_path.string()},
        {"identifier_column", out.options.theta.identifier_column},
        {"factor_column_start", out.options.theta.factor_column_start < 0
            ? json(nullptr) : json(out.options.theta.factor_column_start)},
        {"factor_column_end", out.options.theta.factor_column_end < 0
            ? json(nullptr) : json(out.options.theta.factor_column_end)},
        {"factor_weight_threshold",
            out.options.theta.factor_weight_threshold}
    };
    out.resolved["runtime"] = {{"threads", threads}};
    out.resolved["graph"] = resolved_knn_options(out.options.graph);
    const auto& coarsening = out.options.coarsening;
    out.resolved["coarsening"] = {
        {"enabled", out.options.coarsening_enabled},
        {"target_nodes", coarsening.target_nodes},
        {"activation_threshold", coarsening.activation_threshold},
        {"target_minimum", coarsening.target_minimum},
        {"target_maximum", coarsening.target_maximum},
        {"target_divisor", coarsening.target_divisor},
        {"maximum_microcluster_size", coarsening.maximum_microcluster_size}
    };
    out.resolved["diffusion_sidecar"] = out.options.diffusion_sidecar;
    return out;
}

} // namespace

int32_t cmdKnnGraph(int32_t argc, char** argv) {
    std::string request_path;
    std::string output_path;
    bool diffusion_sidecar = false;
    double factor_weight_threshold = std::numeric_limits<double>::quiet_NaN();
    ParamList parameters;
    parameters.add_option("request", "punkst.knn_graph.request JSON",
            request_path, true)
        .add_option("out-dir", "New output directory for the graph artifact",
            output_path, true)
        .add_option("diffusion-sidecar",
            "Retain geometry needed to construct diffusion operators",
            diffusion_sidecar)
        .add_option("factor-weight-threshold",
            "Override the minimum relative theta column sum",
            factor_weight_threshold);
    try {
        parameters.readArgs(argc, argv);
        const fs::path request_file = fs::absolute(request_path);
        const json request = punkst::multires::read_json(request_file);
        ParsedMetricGraphRequest parsed = parse_metric_graph_request(
            request_file, request);
        if (diffusion_sidecar) {
            parsed.options.diffusion_sidecar = true;
            parsed.resolved["diffusion_sidecar"] = true;
        }
        if (!std::isnan(factor_weight_threshold)) {
            if (factor_weight_threshold < 0.0) {
                throw std::invalid_argument(
                    "--factor-weight-threshold must be nonnegative");
            }
            parsed.options.theta.factor_weight_threshold =
                factor_weight_threshold;
            parsed.resolved["input"]["factor_weight_threshold"] =
                factor_weight_threshold;
        }
        const punkst::multires::MetricGraphResult result =
            punkst::multires::build_metric_graph(
                parsed.theta_path, parsed.options);
        const fs::path output = fs::absolute(output_path);
        punkst::multires::write_metric_graph_artifact(output,
            parsed.theta_path, parsed.resolved, parsed.options, result);
        const json manifest = punkst::multires::read_json(
            output / "manifest.json");
        std::cout << json({
            {"artifact", (output / "manifest.json").string()},
            {"fingerprint", manifest.at("fingerprint")},
            {"points", manifest.at("population").at("points")},
            {"retained_factors",
                manifest.at("population").at("retained_factors")},
            {"coarsening_enabled", !manifest.at("coarsening").is_null()},
            {"diffusion_sidecar",
                !manifest.at("diffusion_input").is_null()}
        }).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "knn-graph: " << error.what() << '\n';
        return 1;
    }
}
