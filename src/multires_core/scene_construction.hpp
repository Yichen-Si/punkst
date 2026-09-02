#pragma once

// Fine-point core assignment, halos, and cross-level scene DAG construction.

#include "clustering_core/partition_classifier.hpp"
#include "multires_core/raw_affinity_graph.hpp"
#include "numerical_utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

enum class SceneCoreMode : uint8_t {
    Inherit,
    ClassifierPlugin,
    ClassifierLrvb,
};

const char* scene_core_mode_name(SceneCoreMode mode);

struct SceneConstructionOptions {
    int64_t minimum_scene_core_members = 200;
    double halo_minimum_score = 0.15;
    double halo_relative_to_core = 0.25;
    int32_t maximum_halo_scenes = 2;
    double portal_minimum_child_fraction = 0.10;
};

struct SceneClassifierOptions {
    double minimum_crossfit_ari = 0.9;
    double minimum_scene_recall = 0.8;
    int32_t minimum_representatives_per_scene = 3;
    uint64_t seed = 1;
    punkst::partition_classifier::FitOptions fit;
};

struct SceneClassifierResult {
    SceneCoreMode mode = SceneCoreMode::ClassifierPlugin;
    bool attempted = false;
    bool passed = false;
    std::string fallback_reason;
    double crossfit_ari = 0.0;
    std::vector<double> scene_recall;
    std::vector<uint8_t> eligible_scene;
    // Retained after the audit passes so an upstream LDA or gamma-Poisson
    // posterior can propagate this exact audited classifier through LRVB.
    bool plugin_model_available = false;
    punkst::partition_classifier::Model plugin_model;
    std::vector<int32_t> model_class_to_scene;
    // Fine-point rows by canonical scene columns. Ineligible scene columns are
    // zero. Candidate restriction and normalization are component-local.
    RowMajorMatrixXd fine_probabilities;
};

// Fit on one representative per microcluster with equal training weights.
// The outer crossfit audit is weighted by represented fine-point counts.
SceneClassifierResult fit_plugin_scene_classifier(
    const Eigen::Ref<const RowMajorMatrixXd>& fine_compositions,
    const std::vector<int32_t>& representative_rows,
    const std::vector<int32_t>& microcluster_membership,
    const std::vector<int64_t>& microcluster_fine_counts,
    const std::vector<int32_t>& fine_component_labels,
    const SceneClassifierOptions& options = SceneClassifierOptions());

// Replace an audited plug-in prediction matrix with probabilities propagated
// from upstream posterior state. The audit and eligible-scene mask are kept;
// invalid or unavailable LRVB input should not call this function.
SceneClassifierResult use_lrvb_scene_probabilities(
    const SceneClassifierResult& audited_plugin,
    const Eigen::Ref<const RowMajorMatrixXd>& fine_probabilities);

struct SceneHaloMembership {
    int32_t fine_node = -1;
    int32_t scene = -1;
    double score = 0.0;
    int32_t rank = 0;
    bool core = false;
};

struct SceneLevelMetadata {
    int32_t level = 1;
    double resolution = 0.0;
    int32_t c90 = 0;
    int32_t plateau_index = -1;
    bool plateau_fallback = false;
};

struct FineSceneLevel {
    SceneLevelMetadata metadata;
    int32_t n_partition_clusters = 0;
    int32_t n_scenes = 0;
    SceneCoreMode requested_core_mode = SceneCoreMode::Inherit;
    SceneCoreMode applied_core_mode = SceneCoreMode::Inherit;
    bool classifier_fallback = false;
    SceneClassifierResult classifier;
    // Canonical selected-partition labels before the minimum-size filter.
    std::vector<int32_t> fine_partition_membership;
    // Created-scene labels, or -1 when the point's partition cluster is too
    // small to become a scene.
    std::vector<int32_t> fine_core_membership;
    // Created-scene index -> original selected-partition cluster label.
    std::vector<int32_t> scene_source_clusters;
    int64_t excluded_fine_points = 0;
    std::vector<int64_t> scene_fine_counts;
    std::vector<int32_t> scene_component_labels;
    std::vector<uint8_t> tail_scene;
    // Contains one core row and at most maximum_halo_scenes alternatives per
    // fine point. Halos never change fine_core_membership.
    std::vector<SceneHaloMembership> memberships;
};

FineSceneLevel construct_fine_scene_level(
    const SceneLevelMetadata& metadata,
    const std::vector<int32_t>& microcluster_membership,
    const std::vector<int32_t>& fine_to_microcluster,
    const std::vector<int32_t>& fine_component_labels,
    const RawClusteringGraph& fine_raw_graph,
    SceneCoreMode requested_mode = SceneCoreMode::Inherit,
    const SceneClassifierResult* classifier = nullptr,
    const SceneConstructionOptions& options = SceneConstructionOptions());

struct SceneDagNode {
    int32_t id = -1;
    int32_t level = 0;
    int32_t scene = 0;
    int32_t source_cluster = -1;
    int64_t fine_count = 0;
    int32_t component = -1;
    bool tail = false;
    double resolution = 0.0;
    int32_t c90 = 0;
    int32_t plateau_index = -1;
    bool plateau_fallback = false;
    SceneCoreMode core_mode = SceneCoreMode::Inherit;
    bool classifier_fallback = false;
    int32_t major_parent = -1;
    int32_t parent_count = 0;
    int32_t child_count = 0;
    bool merge = false;
    bool split = false;
};

struct SceneDagEdge {
    int32_t parent = -1;
    int32_t child = -1;
    int64_t overlap = 0;
    double child_fraction = 0.0;
    bool major = false;
    bool portal = false;
};

struct SceneDag {
    int64_t fine_nodes = 0;
    std::vector<SceneDagNode> nodes;
    std::vector<SceneDagEdge> edges;
};

// Build adjacent-level overlap edges. Level 0 is one root. Every child gets
// one maximum-overlap major parent; other parents covering the configured
// fraction of the child become portals. Halo memberships are not consulted.
SceneDag build_scene_dag(
    const std::vector<FineSceneLevel>& levels,
    const SceneConstructionOptions& options = SceneConstructionOptions());
