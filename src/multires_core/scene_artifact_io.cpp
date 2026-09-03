#include "multires_core/scene_artifact_io.hpp"

#include "multires_core/graph_artifact_io.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace punkst::multires {
namespace {

SceneCoreMode parse_core_mode(const std::string& value) {
    if (value == "inherit") return SceneCoreMode::Inherit;
    if (value == "classifier-plugin") return SceneCoreMode::ClassifierPlugin;
    if (value == "classifier-lrvb") return SceneCoreMode::ClassifierLrvb;
    throw std::runtime_error("Scene artifact core mode is invalid");
}

const json& require_object(const json& parent, const char* name) {
    const auto found = parent.find(name);
    if (found == parent.end() || !found->is_object()) {
        throw std::runtime_error(
            std::string("Scene artifact ") + name + " is missing");
    }
    return *found;
}

} // namespace

LoadedSceneArtifact load_scene_artifact(const fs::path& input) {
    LoadedSceneArtifact output;
    output.manifest_path = resolve_artifact_manifest(input);
    output.root = output.manifest_path.parent_path();
    output.manifest = load_verified_artifact(
        output.manifest_path, "punkst.multires.scenes");
    output.fingerprint = output.manifest.at("fingerprint").get<std::string>();
    const json& source = require_object(output.manifest, "source");
    output.graph_fingerprint = source.at("graph_fingerprint").get<std::string>();
    output.selection_fingerprint = source.at(
        "selection_fingerprint").get<std::string>();
    output.fine_points = output.manifest.at("fine_points").get<int64_t>();
    if (output.fine_points <= 0
            || output.fine_points > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("Scene artifact point count is invalid");
    }
    const json& levels = output.manifest.at("levels");
    if (!levels.is_array() || levels.empty()) {
        throw std::runtime_error("Scene artifact has no levels");
    }
    output.levels.reserve(levels.size());
    for (size_t index = 0; index < levels.size(); ++index) {
        const json& encoded = levels[index];
        FineSceneLevel level;
        level.metadata.level = encoded.at("level").get<int32_t>();
        level.metadata.resolution = encoded.at("resolution").get<double>();
        level.metadata.c90 = encoded.at("selection_c90").get<int32_t>();
        level.metadata.plateau_index = encoded.at(
            "plateau_index").get<int32_t>();
        level.metadata.plateau_fallback = encoded.at(
            "plateau_fallback").get<bool>();
        level.n_partition_clusters = encoded.at(
            "partition_clusters").get<int32_t>();
        level.n_scenes = encoded.at("scenes").get<int32_t>();
        level.requested_core_mode = parse_core_mode(
            encoded.at("requested_core_mode").get<std::string>());
        level.applied_core_mode = parse_core_mode(
            encoded.at("applied_core_mode").get<std::string>());
        level.classifier_fallback = encoded.at(
            "classifier_fallback").get<bool>();
        level.excluded_fine_points = encoded.at(
            "excluded_fine_points").get<int64_t>();
        if (level.metadata.level != static_cast<int32_t>(index + 1)
                || level.n_partition_clusters <= 0 || level.n_scenes <= 0
                || !std::isfinite(level.metadata.resolution)
                || level.metadata.resolution <= 0.0) {
            throw std::runtime_error("Scene artifact level metadata is invalid");
        }
        const json& internal = require_object(encoded, "internal");
        level.fine_partition_membership = read_int32_array(output.root,
            internal.at("fine_partition_membership"));
        level.fine_core_membership = read_int32_array(output.root,
            internal.at("fine_core_membership"));
        const std::vector<int32_t> points = read_int32_array(output.root,
            internal.at("membership_points"));
        const std::vector<int32_t> scenes = read_int32_array(output.root,
            internal.at("membership_scenes"));
        const std::vector<double> scores = read_float64_array(output.root,
            internal.at("membership_scores"));
        const std::vector<int32_t> ranks = read_int32_array(output.root,
            internal.at("membership_ranks"));
        const std::vector<uint8_t> cores = read_uint8_array(output.root,
            internal.at("membership_core"));
        if (level.fine_partition_membership.size()
                    != static_cast<size_t>(output.fine_points)
                || level.fine_core_membership.size()
                    != static_cast<size_t>(output.fine_points)
                || points.size() != scenes.size() || points.size() != scores.size()
                || points.size() != ranks.size() || points.size() != cores.size()) {
            throw std::runtime_error("Scene artifact arrays do not align");
        }
        level.memberships.reserve(points.size());
        for (size_t row = 0; row < points.size(); ++row) {
            if (points[row] < 0 || points[row] >= output.fine_points
                    || scenes[row] < 0 || scenes[row] >= level.n_scenes
                    || !(scores[row] >= 0.0) || !std::isfinite(scores[row])
                    || ranks[row] < 0 || cores[row] > 1) {
                throw std::runtime_error(
                    "Scene artifact membership row is invalid");
            }
            level.memberships.push_back({points[row], scenes[row], scores[row],
                ranks[row], cores[row] != 0});
        }
        output.levels.push_back(std::move(level));
    }
    return output;
}

int32_t scene_dag_node_id(const LoadedSceneArtifact& artifact,
        int32_t level, int32_t scene) {
    if (level <= 0 || level > static_cast<int32_t>(artifact.levels.size())
            || scene < 0
            || scene >= artifact.levels[static_cast<size_t>(level - 1)].n_scenes) {
        throw std::invalid_argument("Scene level/index is outside the DAG");
    }
    int64_t node = 1 + scene;
    for (int32_t previous = 1; previous < level; ++previous) {
        node += artifact.levels[static_cast<size_t>(previous - 1)].n_scenes;
    }
    if (node > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error("Scene DAG node id exceeds int32");
    }
    return static_cast<int32_t>(node);
}

} // namespace punkst::multires
