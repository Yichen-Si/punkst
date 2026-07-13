#include "gamma_pois_cluster_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

namespace gamma_pois_cluster_detail {

RowMajorMatrixXd leading_orientation(
    const Eigen::Ref<const Eigen::MatrixXd>& covariance, int32_t rank) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        0.5 * (covariance + covariance.transpose()));
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Shared orientation eigendecomposition failed");
    }
    RowMajorMatrixXd out(covariance.rows(), rank);
    for (int32_t j = 0; j < rank; ++j) {
        Eigen::VectorXd vector = solver.eigenvectors().col(covariance.rows() - 1 - j);
        Eigen::Index pivot = 0;
        vector.cwiseAbs().maxCoeff(&pivot);
        if (vector(pivot) < 0.0) vector = -vector;
        out.col(j) = vector;
    }
    return out;
}

std::vector<int32_t> maximum_correlation_assignment(
    const Eigen::Ref<const RowMajorMatrixXd>& previous,
    const Eigen::Ref<const RowMajorMatrixXd>& candidate) {
    const int32_t n = static_cast<int32_t>(previous.cols());
    std::vector<std::vector<double>> cost(n + 1, std::vector<double>(n + 1));
    double maximum = 0.0;
    for (int32_t i = 1; i <= n; ++i) {
        for (int32_t j = 1; j <= n; ++j) {
            maximum = std::max(maximum,
                std::abs(previous.col(i - 1).dot(candidate.col(j - 1))));
        }
    }
    for (int32_t i = 1; i <= n; ++i) {
        for (int32_t j = 1; j <= n; ++j) {
            cost[i][j] = maximum
                - std::abs(previous.col(i - 1).dot(candidate.col(j - 1)));
        }
    }
    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int32_t> p(n + 1), way(n + 1);
    for (int32_t i = 1; i <= n; ++i) {
        p[0] = i;
        int32_t j0 = 0;
        std::vector<double> minv(n + 1, std::numeric_limits<double>::infinity());
        std::vector<uint8_t> used(n + 1);
        do {
            used[j0] = 1;
            const int32_t i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int32_t j1 = 0;
            for (int32_t j = 1; j <= n; ++j) {
                if (used[j]) continue;
                const double current = cost[i0][j] - u[i0] - v[j];
                if (current < minv[j]) {
                    minv[j] = current;
                    way[j] = j0;
                }
                if (minv[j] < delta) {
                    delta = minv[j];
                    j1 = j;
                }
            }
            for (int32_t j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int32_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<int32_t> assignment(n);
    for (int32_t j = 1; j <= n; ++j) assignment[p[j] - 1] = j - 1;
    return assignment;
}

double align_and_blend_orientation(RowMajorMatrixXd& orientation,
    RowMajorMatrixXd candidate, double step) {
    const std::vector<int32_t> assignment =
        maximum_correlation_assignment(orientation, candidate);
    RowMajorMatrixXd aligned(candidate.rows(), candidate.cols());
    for (Eigen::Index j = 0; j < orientation.cols(); ++j) {
        aligned.col(j) = candidate.col(assignment[j]);
        if (orientation.col(j).dot(aligned.col(j)) < 0.0) aligned.col(j) *= -1.0;
    }
    Eigen::MatrixXd blended = (1.0 - step) * Eigen::MatrixXd(orientation)
        + step * Eigen::MatrixXd(aligned);
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(blended);
    RowMajorMatrixXd orthogonal = qr.householderQ()
        * Eigen::MatrixXd::Identity(blended.rows(), blended.cols());
    for (Eigen::Index j = 0; j < orientation.cols(); ++j) {
        if (orientation.col(j).dot(orthogonal.col(j)) < 0.0) {
            orthogonal.col(j) *= -1.0;
        }
    }
    const double change = (orientation * orientation.transpose()
        - orthogonal * orthogonal.transpose()).norm();
    orientation = std::move(orthogonal);
    return change;
}


Eigen::MatrixXd make_orientation_sketch(int32_t dim, int32_t rank, int32_t seed) {
    const int32_t size = std::min(dim, 2 * rank + 5);
    Eigen::MatrixXd sketch(dim, size);
    std::mt19937 random(static_cast<uint32_t>(seed) ^ 0x9e3779b9u);
    std::bernoulli_distribution sign(0.5);
    const double scale = 1.0 / std::sqrt(static_cast<double>(size));
    for (Eigen::Index j = 0; j < sketch.size(); ++j) {
        sketch.data()[j] = sign(random) ? scale : -scale;
    }
    return sketch;
}

RowMajorMatrixXd orientation_from_signed_sketch(
    const Eigen::Ref<const Eigen::MatrixXd>& sketch,
    const Eigen::Ref<const Eigen::MatrixXd>& action,
    const Eigen::Ref<const RowMajorMatrixXd>& fallback, int32_t rank) {
    Eigen::MatrixXd core = 0.5 * (
        sketch.transpose() * action + action.transpose() * sketch);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> core_solver(core);
    if (core_solver.info() != Eigen::Success) {
        throw std::runtime_error("Orientation sketch core eigendecomposition failed");
    }
    const double threshold = std::max(1e-12,
        core_solver.eigenvalues().cwiseAbs().maxCoeff() * 1e-10);
    std::vector<int32_t> kept;
    for (Eigen::Index j = 0; j < core.rows(); ++j) {
        if (std::abs(core_solver.eigenvalues()(j)) > threshold) kept.push_back(j);
    }
    if (kept.empty()) return fallback;
    Eigen::MatrixXd vectors(core.rows(), kept.size());
    Eigen::VectorXd inverse_root(kept.size());
    Eigen::VectorXd signs(kept.size());
    for (size_t j = 0; j < kept.size(); ++j) {
        vectors.col(j) = core_solver.eigenvectors().col(kept[j]);
        inverse_root(j) = 1.0
            / std::sqrt(std::abs(core_solver.eigenvalues()(kept[j])));
        signs(j) = core_solver.eigenvalues()(kept[j]) > 0.0 ? 1.0 : -1.0;
    }
    Eigen::MatrixXd signed_factor = action * vectors * inverse_root.asDiagonal();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(signed_factor);
    Eigen::MatrixXd q = qr.householderQ()
        * Eigen::MatrixXd::Identity(action.rows(), signed_factor.cols());
    Eigen::MatrixXd r = q.transpose() * signed_factor;
    Eigen::MatrixXd small = r * signs.asDiagonal() * r.transpose();
    small = 0.5 * (small + small.transpose());
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> small_solver(small);
    if (small_solver.info() != Eigen::Success) {
        throw std::runtime_error("Orientation sketch signed eigendecomposition failed");
    }
    RowMajorMatrixXd candidate(action.rows(), rank);
    int32_t filled = 0;
    for (Eigen::Index j = small.rows() - 1; j >= 0 && filled < rank; --j) {
        if (small_solver.eigenvalues()(j) <= threshold) continue;
        candidate.col(filled++) = q * small_solver.eigenvectors().col(j);
    }
    for (Eigen::Index j = 0; filled < rank && j < fallback.cols(); ++j) {
        Eigen::VectorXd vector = fallback.col(j);
        if (filled > 0) {
            vector -= candidate.leftCols(filled)
                * (candidate.leftCols(filled).transpose() * vector);
        }
        if (vector.norm() > 1e-8) candidate.col(filled++) = vector.normalized();
    }
    if (filled < rank) return fallback;
    return candidate;
}

Eigen::MatrixXd pooled_residual_sketch(const ClusterStatistics& stats,
    const GammaPoissonClusterModel& model,
    const Eigen::Ref<const Eigen::MatrixXd>& sketch) {
    Eigen::MatrixXd action = stats.pooled_second_sketch;
    Eigen::VectorXd weighted_diagonal = Eigen::VectorXd::Zero(model.means.cols());
    for (Eigen::Index c = 0; c < model.means.rows(); ++c) {
        action.noalias() -= stats.membership(c)
            * model.means.row(c).transpose()
            * (model.means.row(c) * sketch);
        weighted_diagonal.array() += stats.membership(c)
            * model.variances.row(c).transpose().array();
    }
    action.array() -= sketch.array().colwise() * weighted_diagonal.array();
    return action;
}

Eigen::MatrixXd pooled_residual_covariance(
    const ClusterStatistics& stats, const GammaPoissonClusterModel& model) {
    if (!stats.dense) {
        throw std::invalid_argument(
            "Dense residual covariance requires dense sufficient statistics");
    }
    const Eigen::Index dim = model.means.cols();
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(dim, dim);
    for (Eigen::Index c = 0; c < model.means.rows(); ++c) {
        covariance += stats.second_dense[c];
        covariance.noalias() -= stats.membership(c)
            * model.means.row(c).transpose()
            * model.means.row(c);
        covariance.diagonal().array() -= stats.membership(c)
            * model.variances.row(c).transpose().array();
    }
    return 0.5 * (covariance + covariance.transpose());
}

void transport_orientation(GammaPoissonClusterModel& model,
    const Eigen::Ref<const RowMajorMatrixXd>& previous,
    const GammaPoissonClusterFitOptions& options) {
    const Eigen::MatrixXd overlap = model.orientation.transpose() * previous;
    for (Eigen::Index c = 0; c < model.means.rows(); ++c) {
        const Eigen::VectorXd old_low =
            model.low_rank_variances.row(c).transpose();
        const Eigen::VectorXd old_total_diagonal =
            model.variances.row(c).transpose()
            + previous.array().square().matrix() * old_low;
        const Eigen::VectorXd new_low = (overlap.array().square().matrix()
            * old_low).cwiseMax(options.low_rank_variance_floor);
        model.low_rank_variances.row(c) = new_low.transpose();
        model.variances.row(c) = (old_total_diagonal
            - model.orientation.array().square().matrix() * new_low)
            .cwiseMax(options.variance_floor).transpose();
    }
}


} // namespace gamma_pois_cluster_detail

double gamma_poisson_cluster_log_volume(
    const GammaPoissonClusterModel& model, int32_t component) {
    gamma_pois_cluster_detail::validate_model(model, static_cast<int32_t>(model.variances.cols()));
    if (component < 0 || component >= model.variances.rows()) {
        throw std::invalid_argument("Invalid component for cluster volume");
    }
    const RowMajorMatrixXd factor = gamma_pois_cluster_detail::covariance_factor(
        model.orientation, model.low_rank_variances.row(component));
    LowRankDiagonalSolver solver(model.variances.row(component).transpose(), factor);
    return 0.5 * solver.log_determinant();
}

GammaPoissonClusterSeparation gamma_poisson_cluster_separation(
    const GammaPoissonClusterModel& model, int32_t first, int32_t second) {
    gamma_pois_cluster_detail::validate_model(model, static_cast<int32_t>(model.variances.cols()));
    if (first < 0 || second < 0 || first >= model.variances.rows()
        || second >= model.variances.rows() || first == second) {
        throw std::invalid_argument("Invalid components for cluster separation");
    }
    const Eigen::VectorXd average_diagonal = 0.5 * (
        model.variances.row(first) + model.variances.row(second)).transpose();
    const Eigen::RowVectorXd average_low_rank = 0.5 * (
        model.low_rank_variances.row(first)
        + model.low_rank_variances.row(second));
    const RowMajorMatrixXd average_factor = gamma_pois_cluster_detail::covariance_factor(
        model.orientation, average_low_rank);
    LowRankDiagonalSolver average_solver(average_diagonal, average_factor);
    const Eigen::VectorXd difference =
        (model.means.row(first) - model.means.row(second)).transpose();
    GammaPoissonClusterSeparation out;
    const double squared = average_solver.quadratic(difference);
    out.standardized_distance = std::sqrt(std::max(0.0, squared));
    const double first_log_determinant = 2.0
        * gamma_poisson_cluster_log_volume(model, first);
    const double second_log_determinant = 2.0
        * gamma_poisson_cluster_log_volume(model, second);
    out.bhattacharyya_distance = 0.125 * squared + 0.5 * (
        average_solver.log_determinant()
        - 0.5 * (first_log_determinant + second_log_determinant));
    out.bhattacharyya_distance = std::max(0.0, out.bhattacharyya_distance);
    return out;
}
