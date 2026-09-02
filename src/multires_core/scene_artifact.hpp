#pragma once

#include "multires_core/artifacts.hpp"
#include "multires_core/scene_construction.hpp"

#include <filesystem>
#include <string>
#include <vector>

namespace punkst::multires {

struct SceneArtifactOptions {
    fs::path graph_manifest;
    fs::path selection_manifest;
    SceneCoreMode core_mode = SceneCoreMode::Inherit;
    SceneConstructionOptions scenes;
    SceneClassifierOptions classifier;
};

struct SceneArtifactResult {
    std::string graph_fingerprint;
    std::string selection_fingerprint;
    std::vector<std::string> identifiers;
    std::vector<FineSceneLevel> levels;
    SceneDag dag;
};

SceneArtifactResult run_multires_scenes(
    const SceneArtifactOptions& options);

void write_scene_artifact(
    const fs::path& output,
    const json& resolved_request,
    const SceneArtifactOptions& options,
    const SceneArtifactResult& result);

} // namespace punkst::multires
