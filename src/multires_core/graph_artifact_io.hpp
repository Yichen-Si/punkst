#pragma once

#include "multires_core/artifacts.hpp"
#include "multires_core/raw_affinity_graph.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace punkst::multires {

fs::path resolve_artifact_manifest(const fs::path& input);

// Verify the schema, embedded binary arrays, and declared fingerprint. The
// returned JSON still contains its fingerprint field.
json load_verified_artifact(
    const fs::path& input, const std::string& expected_type);

struct RawGraphArtifactData {
    RawClusteringGraph graph;
    std::vector<int64_t> fine_point_counts;
};

struct KnnGraphArtifactData {
    fs::path manifest_path;
    fs::path root;
    json manifest;
    std::string fingerprint;
    std::vector<std::string> identifiers;
    RawGraphArtifactData fine;
    std::optional<RawGraphArtifactData> coarse;
    std::vector<int32_t> fine_to_microcluster;
    std::vector<int32_t> representative_rows;
};

KnnGraphArtifactData load_knn_graph_artifact(const fs::path& input);

} // namespace punkst::multires
