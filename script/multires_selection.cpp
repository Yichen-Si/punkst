#include "multires_core/artifacts.hpp"
#include "multires_core/selection_artifact.hpp"
#include "punkst.h"

#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>

namespace {

using punkst::multires::json;
namespace fs = std::filesystem;

struct ParsedRequest {
    punkst::multires::SelectionArtifactOptions options;
    json resolved;
};

ParsedRequest parse_request(const fs::path& request_path,
        const json& request) {
    punkst::multires::reject_unknown_keys(request, {"artifact_type", "schema_version", "source",
        "scan_population", "selection", "refinement", "runtime"},
        "request");
    if (request.value("artifact_type", "")
            != "punkst.multires.selection_request"
            || request.value("schema_version", 0) != 1) {
        throw std::invalid_argument(
            "Expected punkst.multires.selection_request schema version 1");
    }
    ParsedRequest out;
    const json& source = punkst::multires::require_object_field(request, "source");
    punkst::multires::reject_unknown_keys(source, {"graph_manifest", "diffusion_manifest"},
        "source");
    if (!source.contains("graph_manifest")) {
        throw std::invalid_argument("source.graph_manifest is required");
    }
    out.options.graph_manifest = punkst::multires::resolve_request_path(
        request_path, source.at("graph_manifest").get<std::string>());
    if (source.contains("diffusion_manifest")
            && !source.at("diffusion_manifest").is_null()) {
        out.options.diffusion_manifest = punkst::multires::resolve_request_path(request_path,
            source.at("diffusion_manifest").get<std::string>());
    }
    out.options.scan_population = punkst::multires::parse_scan_population(
        request.value("scan_population", std::string("auto")));

    auto& selection = out.options.selection;
    if (request.contains("selection")) {
        const json& value = punkst::multires::require_object_field(request, "selection");
        punkst::multires::reject_unknown_keys(value, {"level1_c90_minimum", "level1_c90_maximum",
            "level1_scene_count_minimum", "level1_scene_count_maximum",
            "minimum_scene_core_members",
            "min_level", "max_level",
            "next_level_c90_multiplier", "fallback_c90_max_multiplier",
            "scout_max_iterations", "maximum_scout_steps",
            "maximum_midpoints", "restarts", "stop_c90",
            "maximum_scan_communities",
            "maximum_scan_steps", "initial_resolution",
            "scout_resolution_factor", "scan_resolution_factor",
            "seed_stability_threshold", "persistence_threshold",
            "minimum_resolution", "maximum_resolution", "seed"},
            "selection");
        selection.level1_c90_minimum = value.value(
            "level1_c90_minimum", selection.level1_c90_minimum);
        selection.level1_c90_maximum = value.value(
            "level1_c90_maximum", selection.level1_c90_maximum);
        selection.level1_scene_count_minimum = value.value(
            "level1_scene_count_minimum",
            selection.level1_scene_count_minimum);
        selection.level1_scene_count_maximum = value.value(
            "level1_scene_count_maximum",
            selection.level1_scene_count_maximum);
        selection.minimum_scene_core_members = value.value(
            "minimum_scene_core_members",
            selection.minimum_scene_core_members);
        selection.minimum_levels = value.value(
            "min_level", selection.minimum_levels);
        selection.maximum_levels = value.value(
            "max_level", selection.maximum_levels);
        selection.next_level_c90_multiplier = value.value(
            "next_level_c90_multiplier",
            selection.next_level_c90_multiplier);
        selection.fallback_c90_max_multiplier = value.value(
            "fallback_c90_max_multiplier",
            selection.fallback_c90_max_multiplier);
        selection.scout_max_iterations = value.value(
            "scout_max_iterations", selection.scout_max_iterations);
        selection.maximum_scout_steps = value.value(
            "maximum_scout_steps", selection.maximum_scout_steps);
        selection.maximum_midpoints = value.value(
            "maximum_midpoints", selection.maximum_midpoints);
        selection.final_restarts = value.value(
            "restarts", selection.final_restarts);
        selection.stop_c90 = value.value("stop_c90", selection.stop_c90);
        selection.maximum_scan_communities = value.value(
            "maximum_scan_communities",
            selection.maximum_scan_communities);
        selection.maximum_scan_steps = value.value(
            "maximum_scan_steps", selection.maximum_scan_steps);
        selection.initial_resolution = value.value(
            "initial_resolution", selection.initial_resolution);
        selection.scout_resolution_factor = value.value(
            "scout_resolution_factor", selection.scout_resolution_factor);
        selection.scan_resolution_factor = value.value(
            "scan_resolution_factor", selection.scan_resolution_factor);
        selection.seed_stability_threshold = value.value(
            "seed_stability_threshold",
            selection.seed_stability_threshold);
        selection.persistence_threshold = value.value(
            "persistence_threshold", selection.persistence_threshold);
        selection.minimum_resolution = value.value(
            "minimum_resolution", selection.minimum_resolution);
        selection.maximum_resolution = value.value(
            "maximum_resolution", selection.maximum_resolution);
        selection.seed = value.value("seed", selection.seed);
    }
    if (request.contains("refinement")) {
        const json& refinement = punkst::multires::require_object_field(request, "refinement");
        punkst::multires::reject_unknown_keys(refinement, {"full_data_leiden", "seed"},
            "refinement");
        out.options.refine_on_full_graph = refinement.value(
            "full_data_leiden", false);
        out.options.refinement_seed = refinement.value("seed", 1);
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
    selection.n_threads = threads;

    out.resolved = {
        {"artifact_type", "punkst.multires.selection_request"},
        {"schema_version", 1},
        {"source", {
            {"graph_manifest", out.options.graph_manifest.string()},
            {"diffusion_manifest", out.options.diffusion_manifest.has_value()
                ? json(out.options.diffusion_manifest->string()) : json(nullptr)}
        }},
        {"scan_population", punkst::multires::scan_population_name(
            out.options.scan_population)},
        {"selection", {
            {"level1_c90_minimum", selection.level1_c90_minimum},
            {"level1_c90_maximum", selection.level1_c90_maximum},
            {"level1_scene_count_minimum",
                selection.level1_scene_count_minimum},
            {"level1_scene_count_maximum",
                selection.level1_scene_count_maximum},
            {"minimum_scene_core_members",
                selection.minimum_scene_core_members},
            {"min_level", selection.minimum_levels},
            {"max_level", selection.maximum_levels},
            {"next_level_c90_multiplier",
                selection.next_level_c90_multiplier},
            {"fallback_c90_max_multiplier",
                selection.fallback_c90_max_multiplier},
            {"scout_max_iterations", selection.scout_max_iterations},
            {"maximum_scout_steps", selection.maximum_scout_steps},
            {"maximum_midpoints", selection.maximum_midpoints},
            {"restarts", selection.final_restarts},
            {"stop_c90", selection.stop_c90},
            {"maximum_scan_communities",
                selection.maximum_scan_communities},
            {"maximum_scan_steps", selection.maximum_scan_steps},
            {"initial_resolution", selection.initial_resolution},
            {"scout_resolution_factor",
                selection.scout_resolution_factor},
            {"scan_resolution_factor", selection.scan_resolution_factor},
            {"seed_stability_threshold",
                selection.seed_stability_threshold},
            {"persistence_threshold", selection.persistence_threshold},
            {"minimum_resolution", selection.minimum_resolution},
            {"maximum_resolution", selection.maximum_resolution},
            {"seed", selection.seed}
        }},
        {"refinement", {
            {"full_data_leiden", out.options.refine_on_full_graph},
            {"seed", out.options.refinement_seed}
        }},
        {"runtime", {{"threads", threads}}}
    };
    return out;
}

} // namespace

int32_t cmdMultiresSelection(int32_t argc, char** argv) {
    std::string request_path;
    std::string output_path;
    ParamList parameters;
    parameters.add_option("request",
            "punkst.multires.selection_request JSON", request_path, true)
        .add_option("out-dir",
            "New output directory for the selection artifact",
            output_path, true);
    try {
        parameters.readArgs(argc, argv);
        const fs::path request_file = fs::absolute(request_path);
        const json request = punkst::multires::read_json(request_file);
        ParsedRequest parsed = parse_request(request_file, request);
        punkst::multires::SelectionArtifactResult result =
            punkst::multires::run_multires_selection(parsed.options);
        const fs::path output = fs::absolute(output_path);
        punkst::multires::write_selection_artifact(output, parsed.resolved,
            parsed.options, result);
        std::cout << json({
            {"artifact", (output / "manifest.json").string()},
            {"scan_population", punkst::multires::scan_population_name(
                result.resolved_population)},
            {"evaluated_resolutions", result.selection.evaluations.size()},
            {"stable_plateaus", result.selection.plateaus.size()},
            {"selected_levels", result.selection.levels.size()}
        }).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multires-selection: " << error.what() << '\n';
        return 1;
    }
}
