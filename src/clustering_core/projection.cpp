#include "clustering_core/projection.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <vector>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace punkst::projection {
namespace {

int32_t checked_int32(Eigen::Index value, const char* label) {
    if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error(std::string(label) + " exceeds int32");
    }
    return static_cast<int32_t>(value);
}

Eigen::MatrixXd symmetrize(const Eigen::Ref<const Eigen::MatrixXd>& value) {
    return 0.5 * (value + value.transpose());
}

Eigen::MatrixXd sample_whitening_covariance(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Eigen::Ref<const Eigen::VectorXd>& center, int32_t n_threads) {
    const int32_t documents = checked_int32(
        coordinates.rows(), "visualization document count");
    const int32_t dimension = checked_int32(
        coordinates.cols(), "visualization coordinate dimension");
    const int32_t workers = std::max(1, n_threads);
    const int32_t blocks = static_cast<int32_t>(std::min<int64_t>(
        documents, static_cast<int64_t>(workers) * 4));
    std::vector<Eigen::MatrixXd> partials(
        blocks, Eigen::MatrixXd::Zero(dimension, dimension));
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(workers));
    tbb::parallel_for(int32_t{0}, blocks, [&](int32_t block) {
        const int32_t first = static_cast<int32_t>(
            static_cast<int64_t>(documents) * block / blocks);
        const int32_t last = static_cast<int32_t>(
            static_cast<int64_t>(documents) * (block + 1) / blocks);
        constexpr int32_t kRowsPerChunk = 4096;
        for (int32_t row = first; row < last; row += kRowsPerChunk) {
            const int32_t count = std::min(kRowsPerChunk, last - row);
            Eigen::MatrixXd centered = coordinates.middleRows(row, count);
            centered.rowwise() -= center.transpose();
            partials[block].noalias() += centered.transpose() * centered;
        }
    });
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(
        dimension, dimension);
    for (const Eigen::MatrixXd& partial : partials) covariance += partial;
    covariance /= static_cast<double>(documents);
    return symmetrize(covariance);
}

VisualizationProjection solve_projection_axes(VisualizationView view,
    const Eigen::Ref<const Eigen::MatrixXd>& kernel,
    const Eigen::Ref<const Eigen::MatrixXd>& whitening_covariance,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const RowMajorMatrixXd>& component_means,
    int32_t maximum_dimensions, bool positive_only) {
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        symmetrize(kernel), whitening_covariance);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Visualization generalized eigendecomposition failed");
    }
    const int32_t dimension = static_cast<int32_t>(kernel.rows());
    const double rank_tolerance = 256.0
        * std::numeric_limits<double>::epsilon()
        * std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
    int32_t output_dimensions = std::min(dimension, maximum_dimensions);
    if (positive_only) {
        output_dimensions = std::min<int32_t>(output_dimensions,
            static_cast<int32_t>((solver.eigenvalues().array()
                > rank_tolerance).count()));
    }
    VisualizationProjection out;
    out.view = view;
    out.eigenvalues.resize(output_dimensions);
    out.projection.resize(dimension, output_dimensions);
    for (int32_t axis = 0; axis < output_dimensions; ++axis) {
        const int32_t source = dimension - 1 - axis;
        double eigenvalue = solver.eigenvalues()(source);
        const double tolerance = 64.0 * std::numeric_limits<double>::epsilon()
            * std::max(1.0, solver.eigenvalues().cwiseAbs().maxCoeff());
        if (eigenvalue < -tolerance) {
            throw std::runtime_error(
                "Visualization kernel has a negative eigenvalue");
        }
        out.eigenvalues(axis) = std::max(0.0, eigenvalue);
        out.projection.col(axis) = solver.eigenvectors().col(source);
    }
    out.topic_contrasts = helmert.transpose() * out.projection;
    for (int32_t axis = 0; axis < output_dimensions; ++axis) {
        Eigen::Index pivot = 0;
        out.topic_contrasts.col(axis).cwiseAbs().maxCoeff(&pivot);
        if (out.topic_contrasts(pivot, axis) < 0.0) {
            out.projection.col(axis) *= -1.0;
            out.topic_contrasts.col(axis) *= -1.0;
        }
    }
    out.component_means = component_means * out.projection;
    return out;
}

VisualizationProjection solve_projection(VisualizationView view,
    const Eigen::Ref<const Eigen::MatrixXd>& kernel,
    const Eigen::Ref<const Eigen::MatrixXd>& whitening_covariance,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationMoments& moments, int32_t output_dimensions) {
    VisualizationProjection out = solve_projection_axes(view, kernel,
        whitening_covariance, helmert, moments.means, output_dimensions,
        false);
    out.component_covariances.reserve(moments.weights.size());
    for (int32_t component = 0; component < moments.weights.size();
            ++component) {
        Eigen::MatrixXd covariance = out.projection.transpose()
            * moments.covariances[static_cast<size_t>(component)]
            * out.projection;
        out.component_covariances.push_back(symmetrize(covariance));
    }
    return out;
}

void validate_visualization_means(const VisualizationMeans& means,
        int32_t dimension) {
    const int32_t components = checked_int32(
        means.weights.size(), "visualization component count");
    if (components <= 0 || means.means.rows() != components
        || means.means.cols() != dimension
        || !means.weights.allFinite() || !means.means.allFinite()
        || (means.weights.array() < 0.0).any()
        || std::abs(means.weights.sum() - 1.0) > 1e-8
        || !(means.weights.array() > 0.0).any()) {
        throw std::invalid_argument("Invalid visualization means");
    }
}

void validate_visualization_moments(const VisualizationMoments& moments,
        int32_t dimension) {
    const int32_t components = checked_int32(
        moments.weights.size(), "visualization component count");
    if (components <= 0 || moments.means.rows() != components
        || moments.means.cols() != dimension
        || moments.covariances.size() != static_cast<size_t>(components)
        || !moments.weights.allFinite() || !moments.means.allFinite()
        || (moments.weights.array() < 0.0).any()
        || std::abs(moments.weights.sum() - 1.0) > 1e-8
        || !(moments.weights.array() > 0.0).any()) {
        throw std::invalid_argument("Invalid visualization moments");
    }
    for (const Eigen::MatrixXd& covariance : moments.covariances) {
        if (covariance.rows() != dimension || covariance.cols() != dimension
            || !covariance.allFinite()
            || (covariance - covariance.transpose()).cwiseAbs().maxCoeff()
                > 1e-8) {
            throw std::invalid_argument(
                "Invalid visualization component covariance");
        }
    }
}

} // namespace

const char* visualization_whitening_name(VisualizationWhitening value) {
    switch (value) {
        case VisualizationWhitening::Sample: return "sample";
        case VisualizationWhitening::Mixture: return "mixture";
    }
    return "unknown";
}

const char* visualization_view_name(VisualizationView value) {
    switch (value) {
        case VisualizationView::Mean: return "mean";
        case VisualizationView::Full: return "full";
    }
    return "unknown";
}

VisualizationWhitening parse_visualization_whitening(
        const std::string& value) {
    if (value == "sample") return VisualizationWhitening::Sample;
    if (value == "mixture") return VisualizationWhitening::Mixture;
    throw std::invalid_argument(
        "Visualization whitening must be sample or mixture");
}

double quartimax_objective(
        const Eigen::Ref<const Eigen::MatrixXd>& loadings) {
    return loadings.array().square().square().sum();
}

void quartimax_rotate(Eigen::MatrixXd& basis, Eigen::MatrixXd& loadings) {
    constexpr int32_t maximum_sweeps = 100;
    constexpr double relative_tolerance = 1e-12;
    constexpr double update_tolerance = 32.0
        * std::numeric_limits<double>::epsilon();
    if (basis.cols() != loadings.cols()) {
        throw std::invalid_argument("Quartimax basis/loading dimensions differ");
    }
    if (basis.cols() <= 1) return;

    double objective_value = quartimax_objective(loadings);
    for (int32_t sweep = 0; sweep < maximum_sweeps; ++sweep) {
        for (Eigen::Index left = 0; left < basis.cols(); ++left) {
            for (Eigen::Index right = left + 1;
                    right < basis.cols(); ++right) {
                const Eigen::ArrayXd x = loadings.col(left).array();
                const Eigen::ArrayXd y = loadings.col(right).array();
                const Eigen::ArrayXd difference = 0.5
                    * (x.square() - y.square());
                const Eigen::ArrayXd product = x * y;
                const double cosine_coefficient =
                    (difference.square() - product.square()).sum();
                const double sine_coefficient =
                    2.0 * (difference * product).sum();
                const double gain = std::hypot(
                    cosine_coefficient, sine_coefficient)
                    - cosine_coefficient;
                const double pair_scale = x.square().square().sum()
                    + y.square().square().sum();
                if (gain <= update_tolerance * std::max(1.0, pair_scale)) {
                    continue;
                }
                const double angle = 0.25 * std::atan2(
                    sine_coefficient, cosine_coefficient);
                const double cosine = std::cos(angle);
                const double sine = std::sin(angle);
                const Eigen::VectorXd basis_left = basis.col(left);
                const Eigen::VectorXd loading_left = loadings.col(left);
                basis.col(left) = cosine * basis_left
                    + sine * basis.col(right);
                basis.col(right) = -sine * basis_left
                    + cosine * basis.col(right);
                loadings.col(left) = cosine * loading_left
                    + sine * loadings.col(right);
                loadings.col(right) = -sine * loading_left
                    + cosine * loadings.col(right);
            }
        }
        const double next_objective = quartimax_objective(loadings);
        const double improvement = next_objective - objective_value;
        objective_value = next_objective;
        if (improvement <= relative_tolerance
                * std::max(1.0, std::abs(objective_value))) {
            break;
        }
    }
}

void quartimax_rotate(VisualizationProjection& projection) {
    const Eigen::Index dimensions = projection.projection.cols();
    if (dimensions != projection.eigenvalues.size()
            || projection.topic_contrasts.cols() != dimensions
            || projection.component_means.cols() != dimensions) {
        throw std::invalid_argument(
            "Invalid visualization projection for quartimax rotation");
    }
    Eigen::MatrixXd rotation = Eigen::MatrixXd::Identity(
        dimensions, dimensions);
    Eigen::MatrixXd contrasts = projection.topic_contrasts;
    quartimax_rotate(rotation, contrasts);

    Eigen::VectorXd scores = (rotation.array().square().colwise()
        * projection.eigenvalues.array()).colwise().sum().matrix();
    std::vector<Eigen::Index> order(static_cast<size_t>(dimensions));
    for (Eigen::Index axis = 0; axis < dimensions; ++axis) {
        order[static_cast<size_t>(axis)] = axis;
    }
    std::stable_sort(order.begin(), order.end(),
        [&](Eigen::Index left, Eigen::Index right) {
            if (scores(left) != scores(right)) {
                return scores(left) > scores(right);
            }
            const double left_concentration = contrasts.col(left).array()
                .square().square().sum();
            const double right_concentration = contrasts.col(right).array()
                .square().square().sum();
            if (left_concentration != right_concentration) {
                return left_concentration > right_concentration;
            }
            return left < right;
        });
    Eigen::MatrixXd ordered_rotation(dimensions, dimensions);
    Eigen::MatrixXd ordered_contrasts(contrasts.rows(), dimensions);
    Eigen::VectorXd ordered_scores(dimensions);
    for (Eigen::Index axis = 0; axis < dimensions; ++axis) {
        const Eigen::Index source = order[static_cast<size_t>(axis)];
        ordered_rotation.col(axis) = rotation.col(source);
        ordered_contrasts.col(axis) = contrasts.col(source);
        ordered_scores(axis) = scores(source);
    }
    rotation = std::move(ordered_rotation);
    contrasts = std::move(ordered_contrasts);
    projection.axis_scores = std::move(ordered_scores);
    projection.projection *= rotation;
    projection.topic_contrasts = std::move(contrasts);
    projection.component_means *= rotation;
    for (Eigen::MatrixXd& covariance : projection.component_covariances) {
        if (covariance.rows() != dimensions
                || covariance.cols() != dimensions) {
            throw std::invalid_argument(
                "Invalid component covariance for quartimax rotation");
        }
        covariance = (rotation.transpose() * covariance * rotation).eval();
    }
    for (Eigen::Index axis = 0; axis < dimensions; ++axis) {
        Eigen::Index pivot = 0;
        projection.topic_contrasts.col(axis).cwiseAbs().maxCoeff(&pivot);
        if (projection.topic_contrasts(pivot, axis) < 0.0) {
            projection.projection.col(axis) *= -1.0;
            projection.topic_contrasts.col(axis) *= -1.0;
            projection.component_means.col(axis) *= -1.0;
            for (Eigen::MatrixXd& covariance :
                    projection.component_covariances) {
                covariance.row(axis) *= -1.0;
                covariance.col(axis) *= -1.0;
            }
        }
    }
}

static VisualizationResult make_visualization_impl(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const VisualizationMoments& moments,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options,
    const VisualizationSampleMoments* sample_moments) {
    const int32_t documents = checked_int32(
        coordinates.rows(), "visualization document count");
    const int32_t dimension = checked_int32(
        coordinates.cols(), "visualization coordinate dimension");
    const int32_t topics = checked_int32(
        helmert.cols(), "visualization topic count");
    if (documents <= 0 || dimension <= 0 || topics != dimension + 1
        || helmert.rows() != dimension
        || !coordinates.allFinite() || !helmert.allFinite()
        || !is_normalized_helmert(helmert) || options.dimensions <= 0
        || options.n_threads <= 0 || !(options.covariance_floor > 0.0)
        || !std::isfinite(options.covariance_floor)) {
        throw std::invalid_argument("Invalid visualization input");
    }
    validate_visualization_moments(moments, dimension);
    const int32_t output_dimensions = std::min(
        dimension, options.dimensions);
    Eigen::VectorXd mixture_mean =
        moments.means.transpose() * moments.weights;
    Eigen::MatrixXd average_covariance = Eigen::MatrixXd::Zero(
        dimension, dimension);
    Eigen::MatrixXd mean_kernel = Eigen::MatrixXd::Zero(
        dimension, dimension);
    for (int32_t component = 0; component < moments.weights.size();
            ++component) {
        if (!(moments.weights(component) > 0.0)) continue;
        const Eigen::VectorXd difference =
            moments.means.row(component).transpose() - mixture_mean;
        average_covariance.noalias() += moments.weights(component)
            * moments.covariances[static_cast<size_t>(component)];
        mean_kernel.noalias() += moments.weights(component)
            * difference * difference.transpose();
    }
    average_covariance = symmetrize(average_covariance);
    mean_kernel = symmetrize(mean_kernel);

    Eigen::MatrixXd whitening_covariance;
    switch (options.whitening) {
        case VisualizationWhitening::Sample:
            if (sample_moments) {
                if (sample_moments->mean.size() != dimension
                    || sample_moments->covariance.rows() != dimension
                    || sample_moments->covariance.cols() != dimension
                    || !sample_moments->mean.allFinite()
                    || !sample_moments->covariance.allFinite()) {
                    throw std::invalid_argument(
                        "Invalid visualization sample moments");
                }
                const Eigen::VectorXd mean_difference =
                    sample_moments->mean - mixture_mean;
                whitening_covariance = sample_moments->covariance
                    + mean_difference * mean_difference.transpose();
            } else {
                whitening_covariance = sample_whitening_covariance(
                    coordinates, mixture_mean, options.n_threads);
            }
            break;
        case VisualizationWhitening::Mixture:
            whitening_covariance = average_covariance + mean_kernel;
            break;
        default:
            throw std::invalid_argument(
                "Invalid visualization whitening mode");
    }
    whitening_covariance = floor_covariance(
        symmetrize(whitening_covariance), options.covariance_floor);
    Eigen::LLT<Eigen::MatrixXd> whitening_solver(whitening_covariance);
    if (whitening_solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "Visualization whitening covariance is not positive definite");
    }

    VisualizationResult out;
    out.whitening = options.whitening;
    out.whitening_covariance = whitening_covariance;
    out.mean = solve_projection(VisualizationView::Mean, mean_kernel,
        whitening_covariance, helmert, moments, output_dimensions);
    out.full.view = VisualizationView::Full;
    if (options.include_full) {
        Eigen::MatrixXd covariance_kernel = Eigen::MatrixXd::Zero(
            dimension, dimension);
        for (int32_t component = 0; component < moments.weights.size();
                ++component) {
            if (!(moments.weights(component) > 0.0)) continue;
            const Eigen::MatrixXd difference =
                moments.covariances[static_cast<size_t>(component)]
                - average_covariance;
            covariance_kernel.noalias() += moments.weights(component)
                * difference * whitening_solver.solve(difference);
        }
        covariance_kernel = symmetrize(covariance_kernel);
        Eigen::MatrixXd full_kernel = mean_kernel
            * whitening_solver.solve(mean_kernel) + covariance_kernel;
        full_kernel = symmetrize(full_kernel);
        out.full = solve_projection(VisualizationView::Full, full_kernel,
            whitening_covariance, helmert, moments, output_dimensions);
    }
    return out;
}

VisualizationResult make_visualization(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const VisualizationMoments& moments,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options) {
    return make_visualization_impl(
        coordinates, moments, helmert, options, nullptr);
}

VisualizationResult make_visualization(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const VisualizationMoments& moments,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options,
    const VisualizationSampleMoments& sample_moments) {
    return make_visualization_impl(
        coordinates, moments, helmert, options, &sample_moments);
}

VisualizationResult make_mean_visualization(
    const VisualizationMeans& means,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options,
    const VisualizationSampleMoments& sample_moments) {
    const int32_t dimension = checked_int32(
        means.means.cols(), "visualization coordinate dimension");
    const int32_t topics = checked_int32(
        helmert.cols(), "visualization topic count");
    if (dimension <= 0 || topics != dimension + 1
        || helmert.rows() != dimension || !helmert.allFinite()
        || !is_normalized_helmert(helmert) || options.dimensions <= 0
        || !(options.covariance_floor > 0.0)
        || !std::isfinite(options.covariance_floor)
        || sample_moments.mean.size() != dimension
        || sample_moments.covariance.rows() != dimension
        || sample_moments.covariance.cols() != dimension
        || !sample_moments.mean.allFinite()
        || !sample_moments.covariance.allFinite()) {
        throw std::invalid_argument("Invalid mean visualization input");
    }
    validate_visualization_means(means, dimension);
    const Eigen::VectorXd mixture_mean =
        means.means.transpose() * means.weights;
    Eigen::MatrixXd mean_kernel = Eigen::MatrixXd::Zero(
        dimension, dimension);
    for (int32_t component = 0; component < means.weights.size();
            ++component) {
        if (!(means.weights(component) > 0.0)) continue;
        const Eigen::VectorXd difference =
            means.means.row(component).transpose() - mixture_mean;
        mean_kernel.noalias() += means.weights(component)
            * difference * difference.transpose();
    }
    const Eigen::VectorXd center_difference =
        sample_moments.mean - mixture_mean;
    Eigen::MatrixXd whitening_covariance = sample_moments.covariance
        + center_difference * center_difference.transpose();
    whitening_covariance = floor_covariance(
        symmetrize(whitening_covariance), options.covariance_floor);

    VisualizationResult out;
    out.whitening = options.whitening;
    out.whitening_covariance = whitening_covariance;
    out.mean = solve_projection_axes(VisualizationView::Mean,
        symmetrize(mean_kernel), whitening_covariance, helmert, means.means,
        options.dimensions, true);
    out.full.view = VisualizationView::Full;
    return out;
}

VisualizationMeans summarize_hard_partition_means(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components) {
    const int32_t documents = checked_int32(
        coordinates.rows(), "hard-partition document count");
    const int32_t dimension = checked_int32(
        coordinates.cols(), "hard-partition dimension");
    if (documents <= 0 || dimension <= 0 || components <= 0
        || assignments.size() != documents || !coordinates.allFinite()) {
        throw std::invalid_argument(
            "Invalid hard-partition visualization input");
    }
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(components);
    VisualizationMeans out;
    out.means = RowMajorMatrixXd::Zero(components, dimension);
    for (int32_t document = 0; document < documents; ++document) {
        const int32_t component = assignments(document);
        if (component < 0 || component >= components) {
            throw std::invalid_argument(
                "Hard-partition label is out of range");
        }
        ++counts(component);
        out.means.row(component) += coordinates.row(document);
    }
    out.weights.resize(components);
    for (int32_t component = 0; component < components; ++component) {
        if (counts(component) <= 0) {
            throw std::invalid_argument(
                "Hard partition contains an empty component");
        }
        out.means.row(component) /= counts(component);
        out.weights(component) = static_cast<double>(counts(component))
            / static_cast<double>(documents);
    }
    return out;
}

VisualizationMoments summarize_hard_partition(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components, int32_t n_threads) {
    const int32_t documents = checked_int32(
        coordinates.rows(), "hard-partition document count");
    const int32_t dimension = checked_int32(
        coordinates.cols(), "hard-partition dimension");
    if (documents <= 0 || dimension <= 0 || components <= 0
        || n_threads <= 0
        || assignments.size() != documents || !coordinates.allFinite()) {
        throw std::invalid_argument(
            "Invalid hard-partition visualization input");
    }
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(components);
    std::vector<std::vector<int32_t>> members(
        static_cast<size_t>(components));
    VisualizationMoments out;
    out.means = RowMajorMatrixXd::Zero(components, dimension);
    for (int32_t document = 0; document < documents; ++document) {
        const int32_t component = assignments(document);
        if (component < 0 || component >= components) {
            throw std::invalid_argument(
                "Hard-partition label is out of range");
        }
        ++counts(component);
        members[static_cast<size_t>(component)].push_back(document);
        out.means.row(component) += coordinates.row(document);
    }
    out.weights.resize(components);
    for (int32_t component = 0; component < components; ++component) {
        if (counts(component) <= 0) {
            throw std::invalid_argument(
                "Hard partition contains an empty component");
        }
        out.means.row(component) /= counts(component);
        out.weights(component) = static_cast<double>(counts(component))
            / static_cast<double>(documents);
    }
    out.covariances.assign(static_cast<size_t>(components),
        Eigen::MatrixXd::Zero(dimension, dimension));
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(n_threads));
    tbb::parallel_for(int32_t{0}, components, [&](int32_t component) {
        Eigen::MatrixXd& covariance =
            out.covariances[static_cast<size_t>(component)];
        for (const int32_t document :
                members[static_cast<size_t>(component)]) {
            const Eigen::VectorXd residual =
                coordinates.row(document).transpose()
                - out.means.row(component).transpose();
            covariance.noalias() += residual * residual.transpose();
        }
    });
    for (int32_t component = 0; component < components; ++component) {
        Eigen::MatrixXd& covariance =
            out.covariances[static_cast<size_t>(component)];
        covariance /= static_cast<double>(counts(component));
        covariance = symmetrize(covariance);
    }
    return out;
}

VisualizationSampleMoments summarize_visualization_sample(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    int32_t n_threads) {
    const int32_t documents = checked_int32(
        coordinates.rows(), "visualization sample document count");
    const int32_t dimension = checked_int32(
        coordinates.cols(), "visualization sample dimension");
    if (documents <= 0 || dimension <= 0 || n_threads <= 0
        || !coordinates.allFinite()) {
        throw std::invalid_argument("Invalid visualization sample input");
    }
    VisualizationSampleMoments out;
    out.mean = coordinates.colwise().mean();
    out.covariance = sample_whitening_covariance(
        coordinates, out.mean, n_threads);
    return out;
}

void write_visualization_axes(const std::string& path,
    const std::vector<std::string>& topics,
    const VisualizationResult& visualization) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Cannot write visualization axes: " + path);
    }
    out << "#whitening\tview\taxis\tseparation_score\tbasis\tindex\tname"
        "\tcoefficient\tcontrast_scale\tside\tnormalized_weight\n"
        << std::scientific << std::setprecision(10);
    std::vector<const VisualizationProjection*> views{&visualization.mean};
    if (visualization.full.projection.cols() > 0) {
        views.push_back(&visualization.full);
    }
    for (const VisualizationProjection* view : views) {
        if (view->projection.cols() != view->axis_scores.size()
            || view->topic_contrasts.rows()
                != static_cast<Eigen::Index>(topics.size())
            || view->topic_contrasts.cols() != view->projection.cols()) {
            throw std::invalid_argument("Invalid visualization axes");
        }
        for (Eigen::Index axis = 0; axis < view->projection.cols(); ++axis) {
            double scale = 0.0;
            for (Eigen::Index topic = 0;
                    topic < view->topic_contrasts.rows(); ++topic) {
                if (view->topic_contrasts(topic, axis) > 0.0) {
                    scale += view->topic_contrasts(topic, axis);
                }
            }
            if (!(scale > 0.0) || !std::isfinite(scale)) {
                throw std::runtime_error(
                    "Visualization contrast has no positive mass");
            }
            for (Eigen::Index coordinate = 0;
                    coordinate < view->projection.rows(); ++coordinate) {
                out << visualization_whitening_name(visualization.whitening)
                    << "\t" << visualization_view_name(view->view)
                    << "\t" << axis + 1 << "\t"
                    << view->axis_scores(axis)
                    << "\tilr\t" << coordinate << "\tilr_" << coordinate
                    << "\t" << view->projection(coordinate, axis)
                    << "\tNA\tNA\tNA\n";
            }
            for (Eigen::Index topic = 0;
                    topic < view->topic_contrasts.rows(); ++topic) {
                const double coefficient =
                    view->topic_contrasts(topic, axis);
                const char* side = coefficient > 0.0 ? "positive"
                    : coefficient < 0.0 ? "negative" : "zero";
                out << visualization_whitening_name(visualization.whitening)
                    << "\t" << visualization_view_name(view->view)
                    << "\t" << axis + 1 << "\t"
                    << view->axis_scores(axis)
                    << "\ttopic\t" << topic << "\t"
                    << topics[static_cast<size_t>(topic)] << "\t"
                    << coefficient
                    << "\t" << scale << "\t" << side
                    << "\t" << std::abs(coefficient) / scale << "\n";
            }
        }
    }
}

} // namespace punkst::projection
