#pragma once

#include "multires_core/artifacts.hpp"
#include "multires_core/scene_projection.hpp"

#include <filesystem>

namespace punkst::multires {

struct SceneProjectionArtifactOptions {
    fs::path graph_manifest;
    fs::path scenes_manifest;
    SceneProjectionOptions projection;
    int32_t local_neighbors = 30;
    int32_t minimum_clue_members = 20;
    double fallback_resolution = 1.0;
    uint64_t seed = 260821;
};

struct SceneProjectionArtifactSummary {
    int32_t scenes = 0;
    int32_t supervised_views = 0;
    int32_t quartimax_pca_views = 0;
    int32_t fallback_leiden_runs = 0;
    std::string fingerprint;
};

SceneProjectionArtifactSummary write_scene_projection_artifact(
    const fs::path& output,
    const json& resolved_request,
    const SceneProjectionArtifactOptions& options);

} // namespace punkst::multires
