#pragma once

// Reusable composition-space projection models for per-scene views.

#include "linear_embedding.hpp"
#include "numerical_utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace punkst::multires {

struct SceneProjectionOptions {
    int32_t maximum_dimensions = 6;
    int32_t n_threads = 1;
    double covariance_floor = 1e-5;
    double minimum_cover_mass = 0.997;
    double minimum_factor_mass = 1e-6;
    int32_t minimum_factors = 3;
};

struct SceneComposition {
    int32_t input_factors = 0;
    std::vector<int32_t> retained_factors;
    std::vector<std::string> retained_factor_names;
    linear_embedding::ProjectionData linear;
    Eigen::MatrixXd helmert;
    double retained_core_mass_proportion = 0.0;
    bool minimum_factors_restored = false;
};

struct SceneProjectionView {
    RowMajorMatrixXd coordinates;
    // Rows use the original input-factor order; locally omitted factors are 0.
    Eigen::MatrixXd topic_contrasts;
    Eigen::VectorXd axis_scores;
    int32_t fit_rows = 0;
    int32_t represented_groups = 0;
    double retained_subspace_variance_fraction = 0.0;
    double quartimax_objective = 0.0;
};

void validate_scene_projection_options(
    const SceneProjectionOptions& options);

// values contains every output member of one scene. core_rows are local row
// indices and are the only rows allowed to influence factor selection or fits.
SceneComposition prepare_scene_composition(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    const std::vector<std::string>& factor_names,
    const std::vector<int32_t>& core_rows,
    const SceneProjectionOptions& options = SceneProjectionOptions());

SceneProjectionView fit_scene_mean_separation(
    const SceneComposition& composition,
    const std::vector<int32_t>& core_rows,
    const Eigen::Ref<const Eigen::VectorXi>& core_assignments,
    int32_t groups,
    const SceneProjectionOptions& options = SceneProjectionOptions());

SceneProjectionView fit_scene_quartimax_pca(
    const SceneComposition& composition,
    const std::vector<int32_t>& core_rows,
    const SceneProjectionOptions& options = SceneProjectionOptions());

} // namespace punkst::multires
