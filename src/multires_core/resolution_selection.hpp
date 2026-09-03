#pragma once

// Stable Leiden resolution selection on a raw Bhattacharyya-affinity graph.

#include "multires_core/raw_affinity_graph.hpp"

#include <cstdint>
#include <limits>
#include <vector>

struct ResolutionSelectionOptions {
    int32_t level1_c90_minimum = 3;
    int32_t level1_c90_maximum = 10;
    // Zero/zero preserves the C90-based Level-1 policy. When enabled, these
    // are hard bounds on clusters that survive the scene core-size filter.
    int32_t level1_scene_count_minimum = 0;
    int32_t level1_scene_count_maximum = 0;
    int64_t minimum_scene_core_members = 1;
    int32_t minimum_levels = 1;
    int32_t maximum_levels = 2;
    double next_level_c90_multiplier = 2.0;
    double fallback_c90_max_multiplier = 6.0;
    int32_t scout_max_iterations = 3;
    int32_t maximum_scout_steps = 16;
    int32_t maximum_midpoints = 4;
    int32_t final_restarts = 5;
    int32_t stop_c90 = 500;
    int32_t maximum_scan_communities = 300;
    int32_t maximum_scan_steps = 64;
    double initial_resolution = 1.0;
    double scout_resolution_factor = 2.0;
    double scan_resolution_factor = 1.4142135623730951;
    double seed_stability_threshold = 0.9;
    double persistence_threshold = 0.9;
    double minimum_resolution = 1e-8;
    double maximum_resolution = 1e8;
    int32_t seed = 1;
    int32_t n_threads = 1;
};

struct ResolutionScoutEvaluation {
    double resolution = 1.0;
    int32_t c90 = 0;
    int32_t n_communities = 0;
    int32_t retained_scenes = 0;
    double quality = 0.0;
    int32_t iterations = 0;
    bool converged = false;
};

struct ResolutionEvaluation {
    double resolution = 1.0;
    int32_t c90 = 0;
    int32_t n_communities = 0;
    double mean_n_communities = 0.0;
    int32_t retained_scenes = 0;
    double mean_pairwise_ari = 0.0;
    double minimum_pairwise_ari = 0.0;
    double persistence_from_previous =
        std::numeric_limits<double>::quiet_NaN();
    int32_t medoid_restart = 0;
    std::vector<int32_t> restart_seeds;
    std::vector<int32_t> restart_n_communities;
    std::vector<double> pairwise_ari;
    std::vector<double> restart_quality;
    std::vector<int32_t> restart_iterations;
    std::vector<uint8_t> restart_converged;
};

struct ResolutionPlateau {
    int32_t first_evaluation = -1;
    int32_t last_evaluation = -1;
    int32_t representative_evaluation = -1;
    double first_resolution = 0.0;
    double last_resolution = 0.0;
    double representative_resolution = 0.0;
    int32_t c90 = 0;
    int32_t n_communities = 0;
    int32_t retained_scenes = 0;
    double minimum_seed_stability = 0.0;
    double minimum_adjacent_persistence = 0.0;
    std::vector<int32_t> membership;
};

struct SelectedResolutionLevel {
    int32_t level = 0;
    int32_t evaluation = -1;
    int32_t plateau = -1;
    double resolution = 0.0;
    int32_t c90 = 0;
    int32_t n_communities = 0;
    int32_t retained_scenes = 0;
    double mean_pairwise_ari = 0.0;
    double minimum_pairwise_ari = 0.0;
    bool stable_plateau = false;
    bool fallback = false;
    bool fallback_ceiling_relaxed = false;
    std::vector<int32_t> membership;
};

// Count communities whose represented fine-point mass reaches the scene
// construction threshold.
int32_t count_retained_scenes(const std::vector<int32_t>& membership,
    const std::vector<int64_t>& fine_point_counts,
    int64_t minimum_scene_core_members);

struct ResolutionSelectionResult {
    double anchor_resolution = 1.0;
    double lower_context_resolution = 0.0;
    bool has_lower_context = false;
    double selection_seconds = 0.0;
    std::vector<ResolutionScoutEvaluation> scout_evaluations;
    std::vector<ResolutionEvaluation> evaluations;
    std::vector<ResolutionPlateau> plateaus;
    std::vector<SelectedResolutionLevel> levels;
};

ResolutionSelectionResult select_stable_resolutions(
    const RawClusteringGraph& graph,
    const std::vector<int64_t>& fine_point_counts,
    const ResolutionSelectionOptions& options = ResolutionSelectionOptions());

// One converged full-data Leiden run initialized by a lifted coarse
// membership. Graph strengths are supplied as node masses, preserving the
// standard RBConfiguration objective.
std::vector<int32_t> refine_partition_on_full_graph(
    const RawClusteringGraph& graph,
    const std::vector<int32_t>& initial_membership,
    double resolution,
    int32_t seed);
