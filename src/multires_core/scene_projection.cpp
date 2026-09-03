#include "multires_core/scene_projection.hpp"

#include "clustering_core/projection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>

namespace punkst::multires {
namespace {

std::vector<int32_t> validate_core_rows(
        const std::vector<int32_t>& core_rows, Eigen::Index rows) {
    if (rows <= 0 || core_rows.empty()) {
        throw std::invalid_argument("Scene projection requires core rows");
    }
    std::vector<int32_t> sorted = core_rows;
    std::sort(sorted.begin(), sorted.end());
    if (sorted.front() < 0 || sorted.back() >= rows
            || std::adjacent_find(sorted.begin(), sorted.end())
                != sorted.end()) {
        throw std::invalid_argument(
            "Scene projection core rows must be unique and in range");
    }
    return sorted;
}

RowMajorMatrixXd select_rows(
        const Eigen::Ref<const RowMajorMatrixXd>& values,
        const std::vector<int32_t>& rows) {
    RowMajorMatrixXd output(rows.size(), values.cols());
    for (size_t index = 0; index < rows.size(); ++index) {
        output.row(static_cast<Eigen::Index>(index)) =
            values.row(rows[index]);
    }
    return output;
}

Eigen::MatrixXd expand_contrasts(
        int32_t input_factors, const std::vector<int32_t>& retained,
        const Eigen::Ref<const Eigen::MatrixXd>& contrasts) {
    if (input_factors <= 0
            || contrasts.rows() != static_cast<Eigen::Index>(retained.size())) {
        throw std::invalid_argument("Invalid scene projection contrasts");
    }
    Eigen::MatrixXd output = Eigen::MatrixXd::Zero(
        input_factors, contrasts.cols());
    for (size_t index = 0; index < retained.size(); ++index) {
        output.row(retained[index]) =
            contrasts.row(static_cast<Eigen::Index>(index));
    }
    return output;
}

template<class Matrix>
Matrix reordered_columns(const Matrix& input,
        const std::vector<int32_t>& order) {
    Matrix output(input.rows(), static_cast<Eigen::Index>(order.size()));
    for (size_t target = 0; target < order.size(); ++target) {
        output.col(static_cast<Eigen::Index>(target)) =
            input.col(order[target]);
    }
    return output;
}

} // namespace

void validate_scene_projection_options(
        const SceneProjectionOptions& options) {
    if (options.maximum_dimensions <= 0 || options.n_threads <= 0
            || !(options.covariance_floor > 0.0)
            || !std::isfinite(options.covariance_floor)
            || !(options.minimum_cover_mass > 0.0)
            || options.minimum_cover_mass > 1.0
            || !std::isfinite(options.minimum_cover_mass)
            || options.minimum_factor_mass < 0.0
            || !std::isfinite(options.minimum_factor_mass)
            || options.minimum_factors < 2) {
        throw std::invalid_argument("Invalid scene projection options");
    }
}

SceneComposition prepare_scene_composition(
        const Eigen::Ref<const RowMajorMatrixXd>& values,
        const std::vector<std::string>& factor_names,
        const std::vector<int32_t>& core_rows,
        const SceneProjectionOptions& options) {
    validate_scene_projection_options(options);
    validate_core_rows(core_rows, values.rows());
    if (values.cols() != static_cast<Eigen::Index>(factor_names.size())
            || values.cols() < options.minimum_factors
            || !values.allFinite() || (values.array() < 0.0).any()) {
        throw std::invalid_argument("Invalid scene composition values");
    }
    const auto selection = linear_embedding::select_projection_factors(
        values, core_rows, options.minimum_cover_mass,
        options.minimum_factor_mass, options.minimum_factors, true);

    RowMajorMatrixXd retained(values.rows(),
        static_cast<Eigen::Index>(selection.retained_indices.size()));
    SceneComposition output;
    output.input_factors = static_cast<int32_t>(values.cols());
    output.retained_factors = selection.retained_indices;
    output.retained_factor_names.reserve(selection.retained_indices.size());
    for (size_t target = 0;
            target < selection.retained_indices.size(); ++target) {
        const int32_t source = selection.retained_indices[target];
        retained.col(static_cast<Eigen::Index>(target)) = values.col(source);
        output.retained_factor_names.push_back(
            factor_names[static_cast<size_t>(source)]);
    }
    output.helmert = normalized_helmert(
        static_cast<int32_t>(selection.retained_indices.size()));
    output.linear = linear_embedding::prepare_projection(
        retained, linear_embedding::ProjectionSpace::Linear,
        output.helmert, 1e-12);
    output.retained_core_mass_proportion =
        selection.retained_mass_proportion;
    output.minimum_factors_restored = selection.minimum_restored;
    return output;
}

SceneProjectionView fit_scene_mean_separation(
        const SceneComposition& composition,
        const std::vector<int32_t>& core_rows,
        const Eigen::Ref<const Eigen::VectorXi>& core_assignments,
        int32_t groups, const SceneProjectionOptions& options) {
    validate_scene_projection_options(options);
    validate_core_rows(core_rows, composition.linear.coordinates.rows());
    if (groups < 2
            || core_assignments.size()
                != static_cast<Eigen::Index>(core_rows.size())
            || composition.helmert.cols()
                != static_cast<Eigen::Index>(
                    composition.retained_factors.size())) {
        throw std::invalid_argument("Invalid scene mean-separation input");
    }
    const RowMajorMatrixXd core = select_rows(
        composition.linear.coordinates, core_rows);
    const auto means = projection::summarize_hard_partition_means(
        core, core_assignments, groups);
    const auto sample = projection::summarize_visualization_sample(
        core, options.n_threads);
    projection::VisualizationOptions projection_options;
    projection_options.whitening = projection::VisualizationWhitening::Mixture;
    projection_options.dimensions = std::min(
        options.maximum_dimensions, groups - 1);
    projection_options.n_threads = options.n_threads;
    projection_options.covariance_floor = options.covariance_floor;
    auto fitted = projection::make_mean_visualization(
        means, composition.helmert, projection_options, sample);
    if (fitted.mean.projection.cols() <= 0) {
        throw std::runtime_error(
            "Scene partition has no positive mean-separation axis");
    }
    projection::quartimax_rotate(fitted.mean);

    SceneProjectionView output;
    output.coordinates = composition.linear.coordinates
        * fitted.mean.projection;
    output.topic_contrasts = expand_contrasts(
        composition.input_factors, composition.retained_factors,
        fitted.mean.topic_contrasts);
    output.axis_scores = fitted.mean.axis_scores;
    output.fit_rows = static_cast<int32_t>(core_rows.size());
    output.represented_groups = groups;
    output.quartimax_objective = projection::quartimax_objective(
        fitted.mean.topic_contrasts);
    return output;
}

SceneProjectionView fit_scene_quartimax_pca(
        const SceneComposition& composition,
        const std::vector<int32_t>& core_rows,
        const SceneProjectionOptions& options) {
    validate_scene_projection_options(options);
    validate_core_rows(core_rows, composition.linear.coordinates.rows());
    const RowMajorMatrixXd core = select_rows(
        composition.linear.coordinates, core_rows);
    const Eigen::RowVectorXd center = core.colwise().mean();
    RowMajorMatrixXd centered_core = core;
    centered_core.rowwise() -= center;
    Eigen::MatrixXd covariance = centered_core.transpose() * centered_core;
    covariance /= static_cast<double>(core.rows());
    covariance = 0.5 * (covariance + covariance.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(covariance);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Scene PCA eigendecomposition failed");
    }
    const double largest = std::max(
        0.0, solver.eigenvalues().maxCoeff());
    const double tolerance = 256.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, largest);
    const int32_t positive = static_cast<int32_t>(
        (solver.eigenvalues().array() > tolerance).count());
    const int32_t dimensions = std::min(
        options.maximum_dimensions, positive);
    if (dimensions <= 0) {
        throw std::runtime_error("Scene composition has no positive PCA rank");
    }
    Eigen::MatrixXd basis(covariance.rows(), dimensions);
    double retained_variance = 0.0;
    for (int32_t axis = 0; axis < dimensions; ++axis) {
        const Eigen::Index source = covariance.rows() - 1 - axis;
        basis.col(axis) = solver.eigenvectors().col(source);
        retained_variance += solver.eigenvalues()(source);
    }
    Eigen::MatrixXd contrasts = composition.helmert.transpose() * basis;
    projection::quartimax_rotate(basis, contrasts);

    RowMajorMatrixXd centered_all = composition.linear.coordinates;
    centered_all.rowwise() -= center;
    RowMajorMatrixXd coordinates = centered_all * basis;
    const RowMajorMatrixXd core_coordinates = centered_core * basis;
    Eigen::VectorXd variances(dimensions);
    const double denominator = static_cast<double>(
        std::max<Eigen::Index>(1, core.rows() - 1));
    for (int32_t axis = 0; axis < dimensions; ++axis) {
        variances(axis) = core_coordinates.col(axis).squaredNorm()
            / denominator;
    }
    std::vector<int32_t> order(static_cast<size_t>(dimensions));
    std::iota(order.begin(), order.end(), int32_t{0});
    std::stable_sort(order.begin(), order.end(),
        [&](int32_t left, int32_t right) {
            return variances(left) > variances(right);
        });
    basis = reordered_columns(basis, order);
    contrasts = reordered_columns(contrasts, order);
    coordinates = reordered_columns(coordinates, order);
    Eigen::VectorXd ordered_variances(dimensions);
    for (int32_t axis = 0; axis < dimensions; ++axis) {
        ordered_variances(axis) = variances(order[static_cast<size_t>(axis)]);
        Eigen::Index pivot = 0;
        contrasts.col(axis).cwiseAbs().maxCoeff(&pivot);
        if (contrasts(pivot, axis) < 0.0) {
            basis.col(axis) *= -1.0;
            contrasts.col(axis) *= -1.0;
            coordinates.col(axis) *= -1.0;
        }
    }
    const double total_variance = solver.eigenvalues()
        .cwiseMax(0.0).sum();

    SceneProjectionView output;
    output.coordinates = std::move(coordinates);
    output.topic_contrasts = expand_contrasts(
        composition.input_factors, composition.retained_factors, contrasts);
    output.axis_scores = std::move(ordered_variances);
    output.fit_rows = static_cast<int32_t>(core_rows.size());
    output.retained_subspace_variance_fraction = total_variance > 0.0
        ? retained_variance / total_variance : 0.0;
    output.quartimax_objective = projection::quartimax_objective(contrasts);
    return output;
}

} // namespace punkst::multires
