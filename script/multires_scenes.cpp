#include "multires_core/artifacts.hpp"
#include "multires_core/scene_artifact.hpp"
#include "punkst.h"

#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using punkst::multires::json;
namespace fs = std::filesystem;

SceneCoreMode parse_core_mode(const std::string& value) {
    if (value == "inherit") return SceneCoreMode::Inherit;
    if (value == "classifier-plugin") return SceneCoreMode::ClassifierPlugin;
    if (value == "classifier-lrvb") return SceneCoreMode::ClassifierLrvb;
    throw std::invalid_argument(
        "core_mode must be inherit, classifier-plugin, or classifier-lrvb");
}

struct ParsedRequest {
    punkst::multires::SceneArtifactOptions options;
    json resolved;
};

ParsedRequest parse_request(const fs::path& request_path,
        const json& request) {
    punkst::multires::reject_unknown_keys(request, {"artifact_type", "schema_version", "source",
        "core_mode", "scenes", "classifier", "runtime"}, "request");
    if (request.value("artifact_type", "")
            != "punkst.multires.scenes_request"
            || request.value("schema_version", 0) != 1) {
        throw std::invalid_argument(
            "Expected punkst.multires.scenes_request schema version 1");
    }
    ParsedRequest out;
    const json& source = punkst::multires::require_object_field(request, "source");
    punkst::multires::reject_unknown_keys(source, {"graph_manifest", "selection_manifest"},
        "source");
    if (!source.contains("graph_manifest")
            || !source.contains("selection_manifest")) {
        throw std::invalid_argument(
            "source.graph_manifest and source.selection_manifest are required");
    }
    out.options.graph_manifest = punkst::multires::resolve_request_path(request_path,
        source.at("graph_manifest").get<std::string>());
    out.options.selection_manifest = punkst::multires::resolve_request_path(request_path,
        source.at("selection_manifest").get<std::string>());
    out.options.core_mode = parse_core_mode(
        request.value("core_mode", std::string("inherit")));

    auto& scenes = out.options.scenes;
    if (request.contains("scenes")) {
        const json& value = punkst::multires::require_object_field(request, "scenes");
        punkst::multires::reject_unknown_keys(value, {"minimum_core_members", "halo_minimum_score",
            "halo_relative_to_core", "maximum_halo_scenes",
            "portal_minimum_child_fraction"}, "scenes");
        scenes.minimum_scene_core_members = value.value(
            "minimum_core_members", scenes.minimum_scene_core_members);
        scenes.halo_minimum_score = value.value(
            "halo_minimum_score", scenes.halo_minimum_score);
        scenes.halo_relative_to_core = value.value(
            "halo_relative_to_core", scenes.halo_relative_to_core);
        scenes.maximum_halo_scenes = value.value(
            "maximum_halo_scenes", scenes.maximum_halo_scenes);
        scenes.portal_minimum_child_fraction = value.value(
            "portal_minimum_child_fraction",
            scenes.portal_minimum_child_fraction);
    }

    auto& classifier = out.options.classifier;
    if (request.contains("classifier")) {
        const json& value = punkst::multires::require_object_field(request, "classifier");
        punkst::multires::reject_unknown_keys(value, {"minimum_crossfit_ari",
            "minimum_scene_recall", "minimum_representatives_per_scene",
            "seed", "folds", "maximum_iterations", "lbfgs_history",
            "quadratic_rank", "gradient_tolerance"}, "classifier");
        classifier.minimum_crossfit_ari = value.value(
            "minimum_crossfit_ari", classifier.minimum_crossfit_ari);
        classifier.minimum_scene_recall = value.value(
            "minimum_scene_recall", classifier.minimum_scene_recall);
        classifier.minimum_representatives_per_scene = value.value(
            "minimum_representatives_per_scene",
            classifier.minimum_representatives_per_scene);
        classifier.seed = value.value("seed", classifier.seed);
        classifier.fit.folds = value.value("folds", classifier.fit.folds);
        classifier.fit.max_iterations = value.value(
            "maximum_iterations", classifier.fit.max_iterations);
        classifier.fit.lbfgs_history = value.value(
            "lbfgs_history", classifier.fit.lbfgs_history);
        classifier.fit.quadratic_rank = value.value(
            "quadratic_rank", classifier.fit.quadratic_rank);
        classifier.fit.gradient_tolerance = value.value(
            "gradient_tolerance", classifier.fit.gradient_tolerance);
    }
    int32_t threads = 1;
    if (request.contains("runtime")) {
        const json& runtime = punkst::multires::require_object_field(request, "runtime");
        punkst::multires::reject_unknown_keys(runtime, {"threads"}, "runtime");
        threads = runtime.value("threads", 1);
    }
    if (threads <= 0) {
        throw std::invalid_argument("runtime.threads must be positive");
    }
    classifier.fit.threads = threads;

    out.resolved = {
        {"artifact_type", "punkst.multires.scenes_request"},
        {"schema_version", 1},
        {"source", {
            {"graph_manifest", out.options.graph_manifest.string()},
            {"selection_manifest", out.options.selection_manifest.string()}
        }},
        {"core_mode", scene_core_mode_name(out.options.core_mode)},
        {"scenes", {
            {"minimum_core_members", scenes.minimum_scene_core_members},
            {"halo_minimum_score", scenes.halo_minimum_score},
            {"halo_relative_to_core", scenes.halo_relative_to_core},
            {"maximum_halo_scenes", scenes.maximum_halo_scenes},
            {"portal_minimum_child_fraction",
                scenes.portal_minimum_child_fraction}
        }},
        {"classifier", {
            {"minimum_crossfit_ari", classifier.minimum_crossfit_ari},
            {"minimum_scene_recall", classifier.minimum_scene_recall},
            {"minimum_representatives_per_scene",
                classifier.minimum_representatives_per_scene},
            {"seed", classifier.seed},
            {"folds", classifier.fit.folds},
            {"maximum_iterations", classifier.fit.max_iterations},
            {"lbfgs_history", classifier.fit.lbfgs_history},
            {"quadratic_rank", classifier.fit.quadratic_rank},
            {"gradient_tolerance", classifier.fit.gradient_tolerance}
        }},
        {"runtime", {{"threads", threads}}}
    };
    return out;
}

} // namespace

int32_t cmdMultiresScenes(int32_t argc, char** argv) {
    std::string request_path;
    std::string output_path;
    ParamList parameters;
    parameters.add_option("request", "punkst.multires.scenes_request JSON",
            request_path, true)
        .add_option("out-dir", "New output directory for the scene artifact",
            output_path, true);
    try {
        parameters.readArgs(argc, argv);
        const fs::path request_file = fs::absolute(request_path);
        const json request = punkst::multires::read_json(request_file);
        ParsedRequest parsed = parse_request(request_file, request);
        punkst::multires::SceneArtifactResult result =
            punkst::multires::run_multires_scenes(parsed.options);
        const fs::path output = fs::absolute(output_path);
        punkst::multires::write_scene_artifact(output, parsed.resolved,
            parsed.options, result);
        const json manifest = punkst::multires::read_json(
            output / "manifest.json");
        std::cout << json({
            {"artifact", (output / "manifest.json").string()},
            {"levels", result.levels.size()},
            {"scene_nodes", result.dag.nodes.size()},
            {"scene_edges", result.dag.edges.size()},
            {"fingerprint", manifest.at("fingerprint")}
        }).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multires-scenes: " << error.what() << '\n';
        return 1;
    }
}
