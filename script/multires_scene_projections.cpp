#include "multires_core/artifacts.hpp"
#include "multires_core/scene_projection_artifact.hpp"
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
    punkst::multires::SceneProjectionArtifactOptions options;
    json resolved;
};

ParsedRequest parse_request(const fs::path& request_path,
        const json& request) {
    punkst::multires::reject_unknown_keys(request, {"artifact_type", "schema_version", "source",
        "embedding", "fallback", "runtime"}, "request");
    if (request.value("artifact_type", "")
            != "punkst.multires.scene_projection_request"
            || request.value("schema_version", 0) != 1) {
        throw std::invalid_argument(
            "Expected punkst.multires.scene_projection_request schema version 1");
    }
    ParsedRequest out;
    const json& source = punkst::multires::require_object_field(request, "source");
    punkst::multires::reject_unknown_keys(source, {"graph_manifest", "scenes_manifest"}, "source");
    if (!source.contains("graph_manifest")
            || !source.contains("scenes_manifest")) {
        throw std::invalid_argument(
            "source.graph_manifest and source.scenes_manifest are required");
    }
    out.options.graph_manifest = punkst::multires::resolve_request_path(request_path,
        source.at("graph_manifest").get<std::string>());
    out.options.scenes_manifest = punkst::multires::resolve_request_path(request_path,
        source.at("scenes_manifest").get<std::string>());

    auto& projection = out.options.projection;
    if (request.contains("embedding")) {
        const json& value = punkst::multires::require_object_field(request, "embedding");
        punkst::multires::reject_unknown_keys(value, {"maximum_dimensions", "covariance_floor",
            "minimum_cover_mass", "minimum_factor_mass", "minimum_factors"},
            "embedding");
        projection.maximum_dimensions = value.value(
            "maximum_dimensions", projection.maximum_dimensions);
        projection.covariance_floor = value.value(
            "covariance_floor", projection.covariance_floor);
        projection.minimum_cover_mass = value.value(
            "minimum_cover_mass", projection.minimum_cover_mass);
        projection.minimum_factor_mass = value.value(
            "minimum_factor_mass", projection.minimum_factor_mass);
        projection.minimum_factors = value.value(
            "minimum_factors", projection.minimum_factors);
    }
    if (request.contains("fallback")) {
        const json& value = punkst::multires::require_object_field(request, "fallback");
        punkst::multires::reject_unknown_keys(value, {"neighbors", "minimum_clue_members",
            "resolution", "seed"}, "fallback");
        out.options.local_neighbors = value.value(
            "neighbors", out.options.local_neighbors);
        out.options.minimum_clue_members = value.value(
            "minimum_clue_members", out.options.minimum_clue_members);
        out.options.fallback_resolution = value.value(
            "resolution", out.options.fallback_resolution);
        out.options.seed = value.value("seed", out.options.seed);
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
    projection.n_threads = threads;

    out.resolved = {
        {"artifact_type", "punkst.multires.scene_projection_request"},
        {"schema_version", 1},
        {"source", {
            {"graph_manifest", out.options.graph_manifest.string()},
            {"scenes_manifest", out.options.scenes_manifest.string()}
        }},
        {"embedding", {
            {"maximum_dimensions", projection.maximum_dimensions},
            {"covariance_floor", projection.covariance_floor},
            {"minimum_cover_mass", projection.minimum_cover_mass},
            {"minimum_factor_mass", projection.minimum_factor_mass},
            {"minimum_factors", projection.minimum_factors}
        }},
        {"fallback", {
            {"neighbors", out.options.local_neighbors},
            {"minimum_clue_members", out.options.minimum_clue_members},
            {"resolution", out.options.fallback_resolution},
            {"seed", out.options.seed}
        }},
        {"runtime", {{"threads", threads}}}
    };
    return out;
}

} // namespace

int32_t cmdMultiresSceneProjections(int32_t argc, char** argv) {
    std::string request_path;
    std::string output_path;
    ParamList parameters;
    parameters.add_option("request",
            "punkst.multires.scene_projection_request JSON", request_path, true)
        .add_option("out-dir",
            "New output directory for per-scene projection artifacts",
            output_path, true);
    try {
        parameters.readArgs(argc, argv);
        const fs::path request_file = fs::absolute(request_path);
        const json request = punkst::multires::read_json(request_file);
        const ParsedRequest parsed = parse_request(request_file, request);
        const fs::path output = fs::absolute(output_path);
        const auto summary = punkst::multires::write_scene_projection_artifact(
            output, parsed.resolved, parsed.options);
        std::cout << json({
            {"artifact", (output / "manifest.json").string()},
            {"scenes", summary.scenes},
            {"supervised_views", summary.supervised_views},
            {"quartimax_pca_views", summary.quartimax_pca_views},
            {"fallback_leiden_runs", summary.fallback_leiden_runs},
            {"fingerprint", summary.fingerprint}
        }).dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "multires-scene-projections: " << error.what() << '\n';
        return 1;
    }
}
