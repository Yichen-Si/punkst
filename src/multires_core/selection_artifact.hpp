#pragma once

#include "multires_core/artifacts.hpp"
#include "multires_core/resolution_selection.hpp"

#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace punkst::multires {

enum class ScanPopulation : uint8_t {
    Auto,
    Points,
    Microclusters,
};

const char* scan_population_name(ScanPopulation population);
ScanPopulation parse_scan_population(const std::string& value);

struct SelectionArtifactOptions {
    fs::path graph_manifest;
    std::optional<fs::path> diffusion_manifest;
    ScanPopulation scan_population = ScanPopulation::Auto;
    bool refine_on_full_graph = false;
    int32_t refinement_seed = 1;
    ResolutionSelectionOptions selection;
};

struct SelectionArtifactResult {
    ScanPopulation resolved_population = ScanPopulation::Points;
    ResolutionSelectionResult selection;
    std::vector<std::string> identifiers;
    std::vector<std::vector<int32_t>> scan_memberships;
    std::vector<std::vector<int32_t>> full_memberships;
    std::string graph_fingerprint;
    std::optional<std::string> diffusion_fingerprint;
    std::optional<std::string> diffusion_selection_identity_fingerprint;
};

SelectionArtifactResult run_multires_selection(
    const SelectionArtifactOptions& options);

void write_selection_artifact(
    const fs::path& output,
    const json& resolved_request,
    const SelectionArtifactOptions& options,
    const SelectionArtifactResult& result);

} // namespace punkst::multires
