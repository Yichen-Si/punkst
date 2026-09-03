#pragma once

#include "multires_core/artifacts.hpp"
#include "multires_core/scene_construction.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace punkst::multires {

struct LoadedSceneArtifact {
    fs::path manifest_path;
    fs::path root;
    json manifest;
    std::string fingerprint;
    std::string graph_fingerprint;
    std::string selection_fingerprint;
    int64_t fine_points = 0;
    std::vector<FineSceneLevel> levels;
};

// Loads the committed scene format backed by verified binary arrays.
LoadedSceneArtifact load_scene_artifact(const fs::path& input);

int32_t scene_dag_node_id(
    const LoadedSceneArtifact& artifact, int32_t level, int32_t scene);

} // namespace punkst::multires
