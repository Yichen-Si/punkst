#include "clustering/uac.hpp"
#include "clustering/uac_common_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace uac {
namespace {

Eigen::MatrixXd symmetrize(const Eigen::Ref<const Eigen::MatrixXd>& value) {
    return 0.5 * (value + value.transpose());
}

Eigen::MatrixXd sample_whitening_covariance(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Eigen::Ref<const Eigen::VectorXd>& center, int32_t n_threads) {
    const int32_t documents = detail::checked_int32(
        coordinates.rows(), "visualization document count");
    const int32_t dimension = detail::checked_int32(
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

VisualizationProjection solve_projection(VisualizationView view,
    const Eigen::Ref<const Eigen::MatrixXd>& kernel,
    const Eigen::Ref<const Eigen::MatrixXd>& whitening_covariance,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Model& model, int32_t output_dimensions) {
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        symmetrize(kernel), whitening_covariance);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "UAC visualization generalized eigendecomposition failed");
    }
    const int32_t dimension = static_cast<int32_t>(kernel.rows());
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
                "UAC visualization kernel has a negative eigenvalue");
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
    out.component_means = model.means * out.projection;
    out.component_covariances.reserve(model.weights.size());
    for (int32_t component = 0; component < model.weights.size();
            ++component) {
        Eigen::MatrixXd covariance = out.projection.transpose()
            * detail::model_covariance_dense(model, component)
            * out.projection;
        out.component_covariances.push_back(symmetrize(covariance));
    }
    return out;
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
        "UAC visualization whitening must be sample or mixture");
}

VisualizationResult make_visualization(const Dataset& data,
    const Model& model, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options) {
    detail::validate_model(model);
    const int32_t documents = detail::checked_int32(
        data.coordinates.rows(), "visualization document count");
    const int32_t dimension = detail::checked_int32(
        data.coordinates.cols(), "visualization coordinate dimension");
    const int32_t topics = detail::checked_int32(
        helmert.cols(), "visualization topic count");
    if (documents <= 0 || dimension <= 0 || topics != dimension + 1
        || helmert.rows() != dimension || model.means.cols() != dimension
        || !data.coordinates.allFinite() || !helmert.allFinite()
        || !is_normalized_helmert(helmert) || options.dimensions <= 0
        || options.n_threads <= 0 || !(options.covariance_floor > 0.0)
        || !std::isfinite(options.covariance_floor)) {
        throw std::invalid_argument("Invalid UAC visualization input");
    }
    const int32_t output_dimensions = std::min(
        dimension, options.dimensions);
    Eigen::VectorXd mixture_mean = model.means.transpose() * model.weights;
    Eigen::MatrixXd average_covariance = Eigen::MatrixXd::Zero(
        dimension, dimension);
    Eigen::MatrixXd mean_kernel = Eigen::MatrixXd::Zero(
        dimension, dimension);
    for (int32_t component = 0; component < model.weights.size();
            ++component) {
        if (!(model.weights(component) > 0.0)) continue;
        const Eigen::VectorXd difference =
            model.means.row(component).transpose() - mixture_mean;
        average_covariance.noalias() += model.weights(component)
            * detail::model_covariance_dense(model, component);
        mean_kernel.noalias() += model.weights(component)
            * difference * difference.transpose();
    }
    average_covariance = symmetrize(average_covariance);
    mean_kernel = symmetrize(mean_kernel);

    Eigen::MatrixXd whitening_covariance;
    switch (options.whitening) {
        case VisualizationWhitening::Sample:
            whitening_covariance = sample_whitening_covariance(
                data.coordinates, mixture_mean, options.n_threads);
            break;
        case VisualizationWhitening::Mixture:
            whitening_covariance = average_covariance + mean_kernel;
            break;
        default:
            throw std::invalid_argument(
                "Invalid UAC visualization whitening mode");
    }
    whitening_covariance = floor_covariance(
        symmetrize(whitening_covariance), options.covariance_floor);
    Eigen::LLT<Eigen::MatrixXd> whitening_solver(whitening_covariance);
    if (whitening_solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "UAC visualization whitening covariance is not positive definite");
    }

    Eigen::MatrixXd covariance_kernel = Eigen::MatrixXd::Zero(
        dimension, dimension);
    for (int32_t component = 0; component < model.weights.size();
            ++component) {
        if (!(model.weights(component) > 0.0)) continue;
        const Eigen::MatrixXd difference =
            detail::model_covariance_dense(model, component)
            - average_covariance;
        covariance_kernel.noalias() += model.weights(component)
            * difference * whitening_solver.solve(difference);
    }
    covariance_kernel = symmetrize(covariance_kernel);
    Eigen::MatrixXd full_kernel = mean_kernel
        * whitening_solver.solve(mean_kernel) + covariance_kernel;
    full_kernel = symmetrize(full_kernel);

    VisualizationResult out;
    out.whitening = options.whitening;
    out.whitening_covariance = whitening_covariance;
    out.mean = solve_projection(VisualizationView::Mean, mean_kernel,
        whitening_covariance, helmert, model, output_dimensions);
    out.full = solve_projection(VisualizationView::Full, full_kernel,
        whitening_covariance, helmert, model, output_dimensions);
    return out;
}

} // namespace uac
