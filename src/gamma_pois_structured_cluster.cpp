#include "gamma_pois_cluster.hpp"
#include "dense_kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace {

constexpr double LOG_2PI = 1.83787706640934548356;

Eigen::VectorXd expected_log_dirichlet(
    const Eigen::Ref<const Eigen::VectorXd>& parameters) {
    Eigen::VectorXd out(parameters.size());
    const double psi_sum = psi(parameters.sum());
    for (Eigen::Index c = 0; c < parameters.size(); ++c) {
        out(c) = psi(parameters(c)) - psi_sum;
    }
    return out;
}

double dirichlet_elbo(const Eigen::Ref<const Eigen::VectorXd>& parameters,
    double alpha) {
    const double sum = parameters.sum();
    const double psi_sum = psi(sum);
    double out = std::lgamma(alpha * parameters.size())
        - parameters.size() * std::lgamma(alpha) - std::lgamma(sum);
    for (Eigen::Index c = 0; c < parameters.size(); ++c) {
        const double expected_log = psi(parameters(c)) - psi_sum;
        out += std::lgamma(parameters(c))
            + (alpha - parameters(c)) * expected_log;
    }
    return out;
}

RowMajorMatrixXd covariance_factor(
    const Eigen::Ref<const RowMajorMatrixXd>& orientation,
    const Eigen::Ref<const Eigen::RowVectorXd>& variances) {
    RowMajorMatrixXd out(orientation.rows(), orientation.cols());
    if (orientation.cols() > 0) {
        out = orientation.array().rowwise() * variances.array().sqrt();
    }
    return out;
}

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

struct PreparedComponent {
    Eigen::VectorXd diagonal;
    RowMajorMatrixXd factor;
    Eigen::VectorXd covariance_diagonal;
    Eigen::MatrixXd covariance;
};

Eigen::VectorXd apply_component_vector(const PreparedComponent& component,
    const Eigen::Ref<const Eigen::VectorXd>& value) {
    Eigen::VectorXd out = component.diagonal.array() * value.array();
    if (component.factor.cols() > 0) {
        out.noalias() += component.factor
            * (component.factor.transpose() * value);
    }
    return out;
}

Eigen::MatrixXd apply_component_matrix(const PreparedComponent& component,
    const Eigen::Ref<const Eigen::MatrixXd>& value) {
    Eigen::MatrixXd out =
        (value.array().colwise() * component.diagonal.array()).matrix();
    if (component.factor.cols() > 0) {
        out.noalias() += component.factor
            * (component.factor.transpose() * value);
    }
    return out;
}

Eigen::VectorXd conditional_covariance_diagonal(
    const PreparedComponent& component,
    const LowRankDiagonalSolver& solver) {
    const Eigen::VectorXd& inverse = solver.inverse_diagonal();
    Eigen::VectorXd sandwich = component.diagonal.array().square()
        * inverse.array();
    if (component.factor.cols() > 0) {
        const Eigen::VectorXd factor_diagonal =
            component.factor.array().square().rowwise().sum();
        sandwich.array() += 2.0 * component.diagonal.array()
            * inverse.array() * factor_diagonal.array();
        const Eigen::MatrixXd middle = component.factor.transpose()
            * inverse.asDiagonal() * component.factor;
        for (Eigen::Index j = 0; j < sandwich.size(); ++j) {
            sandwich(j) += component.factor.row(j)
                * middle * component.factor.row(j).transpose();
        }
    }
    const Eigen::MatrixXd& inverse_factor = solver.inverse_diagonal_factor();
    if (inverse_factor.cols() > 0) {
        const Eigen::MatrixXd projected = apply_component_matrix(component, inverse_factor);
        const Eigen::MatrixXd corrected = solver.solve_core(projected.transpose());
        sandwich.array() -= (projected.array()
            * corrected.transpose().array()).rowwise().sum();
    }
    return (component.covariance_diagonal - sandwich).cwiseMax(0.0);
}

struct ConditionalContractions {
    Eigen::VectorXd mean;
    Eigen::VectorXd covariance_diagonal;
    Eigen::MatrixXd covariance_projected;
    Eigen::MatrixXd covariance_sketch;
};

ConditionalContractions conditional_contractions(
    const Eigen::Ref<const Eigen::VectorXd>& observation,
    const Eigen::Ref<const Eigen::VectorXd>& center,
    const PreparedComponent& component, const LowRankDiagonalSolver& solver,
    const Eigen::Ref<const RowMajorMatrixXd>& orientation,
    const Eigen::MatrixXd* sketch) {
    ConditionalContractions out;
    const Eigen::VectorXd residual = observation - center;
    out.mean = center + apply_component_vector(
        component, solver.solve_vector(residual));
    out.covariance_diagonal = conditional_covariance_diagonal(component, solver);
    if (orientation.cols() > 0) {
        const Eigen::MatrixXd sigma_projection = apply_component_matrix(
            component, Eigen::MatrixXd(orientation));
        out.covariance_projected = orientation.transpose() * sigma_projection
            - sigma_projection.transpose() * solver.solve_matrix(sigma_projection);
        out.covariance_projected = 0.5
            * (out.covariance_projected + out.covariance_projected.transpose());
    } else {
        out.covariance_projected.resize(0, 0);
    }
    if (sketch) {
        const Eigen::MatrixXd sigma_sketch = apply_component_matrix(component, *sketch);
        out.covariance_sketch = sigma_sketch - apply_component_matrix(
            component, solver.solve_matrix(sigma_sketch));
    }
    return out;
}

struct CompactStatistics {
    Eigen::VectorXd membership;
    RowMajorMatrixXd first;
    RowMajorMatrixXd second_diagonal;
    std::vector<Eigen::MatrixXd> second_projected;
    Eigen::MatrixXd pooled_second_sketch;
    std::vector<Eigen::MatrixXd> second_dense;
    bool dense = false;
    double elbo_local = 0.0;
    double predictive_log_likelihood = 0.0;

    CompactStatistics(int32_t components, int32_t dim, int32_t rank,
        int32_t sketch_size, bool dense_ = false)
        : membership(Eigen::VectorXd::Zero(components)),
          first(RowMajorMatrixXd::Zero(components, dim)),
          second_diagonal(RowMajorMatrixXd::Zero(
              components, dense_ ? 0 : dim)),
          second_projected(dense_ ? 0 : components,
              Eigen::MatrixXd::Zero(rank, rank)),
          pooled_second_sketch(Eigen::MatrixXd::Zero(
              dense_ ? 0 : dim, sketch_size)),
          second_dense(dense_ ? components : 0,
              Eigen::MatrixXd::Zero(dim, dim)), dense(dense_) {}
};

struct StructuredState {
    GammaPoissonClusterModel model;
};

CompactStatistics compact_e_step(const GammaPoissonClusterCoordinates& coordinates,
    const StructuredState& state, int32_t n_threads,
    RowMajorMatrixXd& responsibilities, const Eigen::MatrixXd* sketch,
    bool collect_moments, bool dense_accumulation) {
    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
    const int32_t components = static_cast<int32_t>(state.model.means.rows());
    const int32_t document_rank = coordinates.uncertainty_rank;
    const int32_t rank = static_cast<int32_t>(state.model.orientation.cols());
    const int32_t sketch_size = sketch ? static_cast<int32_t>(sketch->cols()) : 0;
    const int32_t shards = std::max(1, std::min(n_threads, n));
    const Eigen::VectorXd expected_log =
        expected_log_dirichlet(state.model.dirichlet_parameters);
    const Eigen::VectorXd weights = gamma_poisson_cluster_weights(state.model);
    std::vector<PreparedComponent> prepared(components);
    for (int32_t c = 0; c < components; ++c) {
        prepared[c].diagonal = state.model.variances.row(c).transpose();
        prepared[c].factor = covariance_factor(state.model.orientation,
            state.model.low_rank_variances.row(c));
        prepared[c].covariance_diagonal = prepared[c].diagonal;
        if (rank > 0) {
            prepared[c].covariance_diagonal.array() +=
                prepared[c].factor.array().square().rowwise().sum();
        }
        if (dense_accumulation && collect_moments) {
            prepared[c].covariance = prepared[c].diagonal.asDiagonal();
            if (rank > 0) {
                prepared[c].covariance.noalias() += prepared[c].factor
                    * prepared[c].factor.transpose();
            }
        }
    }
    responsibilities.resize(n, components);
    const int32_t statistics_dim = collect_moments ? dim : 0;
    const int32_t statistics_rank = collect_moments ? rank : 0;
    const int32_t statistics_sketch_size = collect_moments ? sketch_size : 0;
    const bool dense_statistics = collect_moments && dense_accumulation;
    std::vector<CompactStatistics> partial;
    partial.reserve(shards);
    for (int32_t shard = 0; shard < shards; ++shard) {
        partial.emplace_back(components, statistics_dim, statistics_rank,
            statistics_sketch_size, dense_statistics);
    }
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, shards),
        [&](const tbb::blocked_range<int32_t>& range) {
            for (int32_t shard = range.begin(); shard < range.end(); ++shard) {
                CompactStatistics& stats = partial[shard];
                const int32_t begin = n * shard / shards;
                const int32_t end = n * (shard + 1) / shards;
                std::vector<double> gaussian(components);
                std::vector<LowRankDiagonalSolver> solvers(components);
                RowMajorMatrixXd total_factor(dim, document_rank + rank);
                for (int32_t d = begin; d < end; ++d) {
                    const Eigen::VectorXd observation =
                        coordinates.mean.row(d).transpose();
                    const auto document_factor = coordinates.factor(d);
                    double maximum = -std::numeric_limits<double>::infinity();
                    double predictive_maximum = maximum;
                    for (int32_t c = 0; c < components; ++c) {
                        if (document_rank > 0) {
                            total_factor.leftCols(document_rank) = document_factor;
                        }
                        if (rank > 0) {
                            total_factor.rightCols(rank) = prepared[c].factor;
                        }
                        const Eigen::VectorXd total_diagonal =
                            coordinates.uncertainty_diagonal.row(d).transpose()
                            + prepared[c].diagonal;
                        solvers[c].compute(total_diagonal, total_factor);
                        const Eigen::VectorXd residual = observation
                            - state.model.means.row(c).transpose();
                        gaussian[c] = -0.5 * (dim * LOG_2PI
                            + solvers[c].log_determinant()
                            + solvers[c].quadratic(residual));
                        responsibilities(d, c) = expected_log(c) + gaussian[c];
                        maximum = std::max(maximum, responsibilities(d, c));
                        predictive_maximum = std::max(predictive_maximum,
                            std::log(std::max(weights(c), 1e-300)) + gaussian[c]);
                    }
                    double sum = 0.0;
                    double predictive_sum = 0.0;
                    for (int32_t c = 0; c < components; ++c) {
                        sum += std::exp(responsibilities(d, c) - maximum);
                        predictive_sum += std::exp(
                            std::log(std::max(weights(c), 1e-300))
                            + gaussian[c] - predictive_maximum);
                    }
                    const double normalizer = maximum + std::log(sum);
                    for (int32_t c = 0; c < components; ++c) {
                        const double responsibility = std::exp(
                            responsibilities(d, c) - normalizer);
                        responsibilities(d, c) = responsibility;
                        if (responsibility > 0.0) {
                            stats.elbo_local += responsibility * (
                                expected_log(c) + gaussian[c]
                                - std::log(responsibility));
                        }
                        if (!collect_moments) continue;
                        if (dense_accumulation) {
                            const Eigen::VectorXd residual = observation
                                - state.model.means.row(c).transpose();
                            const Eigen::VectorXd conditional_mean =
                                state.model.means.row(c).transpose()
                                + apply_component_vector(
                                    prepared[c], solvers[c].solve_vector(residual));
                            Eigen::MatrixXd conditional_covariance =
                                prepared[c].covariance
                                - apply_component_matrix(prepared[c],
                                    solvers[c].solve_matrix(
                                        prepared[c].covariance));
                            conditional_covariance = 0.5
                                * (conditional_covariance
                                    + conditional_covariance.transpose());
                            stats.membership(c) += responsibility;
                            stats.first.row(c) += responsibility
                                * conditional_mean.transpose();
                            stats.second_dense[c].noalias() += responsibility * (
                                conditional_covariance
                                + conditional_mean * conditional_mean.transpose());
                            continue;
                        }
                        ConditionalContractions conditional = conditional_contractions(
                            observation, state.model.means.row(c).transpose(), prepared[c],
                            solvers[c], state.model.orientation, sketch);
                        stats.membership(c) += responsibility;
                        stats.first.row(c) += responsibility * conditional.mean.transpose();
                        stats.second_diagonal.row(c) += responsibility * (
                            conditional.covariance_diagonal.array()
                            + conditional.mean.array().square()).matrix().transpose();
                        if (rank > 0) {
                            const Eigen::VectorXd projected_mean =
                                state.model.orientation.transpose() * conditional.mean;
                            stats.second_projected[c].noalias() += responsibility * (
                                conditional.covariance_projected
                                + projected_mean * projected_mean.transpose());
                        }
                        if (sketch) {
                            stats.pooled_second_sketch.noalias() += responsibility * (
                                conditional.covariance_sketch
                                + conditional.mean
                                    * (conditional.mean.transpose() * *sketch));
                        }
                    }
                    stats.predictive_log_likelihood += predictive_maximum
                        + std::log(predictive_sum);
                }
            }
        });
    CompactStatistics out(components, statistics_dim, statistics_rank,
        statistics_sketch_size, dense_statistics);
    for (int32_t shard = 0; shard < shards; ++shard) {
        out.membership += partial[shard].membership;
        out.first += partial[shard].first;
        if (dense_statistics) {
            for (int32_t c = 0; c < components; ++c) {
                out.second_dense[c] += partial[shard].second_dense[c];
            }
        } else {
            out.second_diagonal += partial[shard].second_diagonal;
            for (int32_t c = 0; c < components; ++c) {
                out.second_projected[c] += partial[shard].second_projected[c];
            }
            out.pooled_second_sketch += partial[shard].pooled_second_sketch;
        }
        out.elbo_local += partial[shard].elbo_local;
        out.predictive_log_likelihood += partial[shard].predictive_log_likelihood;
    }
    return out;
}

void compact_m_step(const CompactStatistics& stats, StructuredState& state,
    double alpha, const GammaPoissonClusterFitOptions& options,
    GammaPoissonClusterFitDiagnostics& diagnostics) {
    const RowMajorMatrixXd old_means = state.model.means;
    const RowMajorMatrixXd old_diagonal = state.model.variances;
    const RowMajorMatrixXd old_low_rank = state.model.low_rank_variances;
    const Eigen::VectorXd old_weights = gamma_poisson_cluster_weights(state.model);
    const Eigen::MatrixXd orientation_squared =
        state.model.orientation.array().square();
    for (Eigen::Index c = 0; c < state.model.means.rows(); ++c) {
        if (stats.membership(c) <= 1e-12) continue;
        state.model.means.row(c) = stats.first.row(c) / stats.membership(c);
        Eigen::VectorXd target_diagonal;
        if (stats.dense) {
            target_diagonal = (stats.second_dense[c].diagonal()
                / stats.membership(c)
                - state.model.means.row(c).transpose().array().square().matrix())
                .cwiseMax(0.0);
        } else {
            target_diagonal = (
                stats.second_diagonal.row(c) / stats.membership(c)
                - state.model.means.row(c).array().square().matrix()).transpose()
                .cwiseMax(0.0);
        }
        if (state.model.orientation.cols() == 0) {
            state.model.variances.row(c) = target_diagonal.array()
                .max(options.variance_floor).transpose();
            continue;
        }
        const Eigen::VectorXd projected_mean = state.model.orientation.transpose()
            * state.model.means.row(c).transpose();
        Eigen::MatrixXd target_projected;
        if (stats.dense) {
            target_projected = state.model.orientation.transpose()
                * stats.second_dense[c] * state.model.orientation
                / stats.membership(c)
                - projected_mean * projected_mean.transpose();
        } else {
            target_projected = stats.second_projected[c] / stats.membership(c)
                - projected_mean * projected_mean.transpose();
        }
        Eigen::VectorXd diagonal = state.model.variances.row(c).transpose();
        Eigen::VectorXd low_rank =
            state.model.low_rank_variances.row(c).transpose();
        for (int32_t pass = 0; pass < 4; ++pass) {
            low_rank = (target_projected.diagonal()
                - orientation_squared.transpose() * diagonal)
                .cwiseMax(options.low_rank_variance_floor);
            diagonal = (target_diagonal - orientation_squared * low_rank)
                .cwiseMax(options.variance_floor);
        }
        state.model.variances.row(c) = diagonal.transpose();
        state.model.low_rank_variances.row(c) = low_rank.transpose();
    }
    state.model.dirichlet_parameters = stats.membership.array() + alpha;
    diagnostics.max_standardized_center_change = 0.0;
    diagnostics.max_log_variance_change = 0.0;
    for (Eigen::Index c = 0; c < state.model.means.rows(); ++c) {
        for (Eigen::Index j = 0; j < state.model.means.cols(); ++j) {
            diagnostics.max_standardized_center_change = std::max(
                diagnostics.max_standardized_center_change,
                std::abs(state.model.means(c, j) - old_means(c, j))
                    / std::sqrt(old_diagonal(c, j)));
            diagnostics.max_log_variance_change = std::max(
                diagnostics.max_log_variance_change,
                std::abs(std::log(state.model.variances(c, j))
                    - std::log(old_diagonal(c, j))));
        }
    }
    if (state.model.low_rank_variances.size() > 0
        && old_low_rank.size() > 0) {
        diagnostics.max_log_variance_change = std::max(
            diagnostics.max_log_variance_change,
            (state.model.low_rank_variances.array().log()
                - old_low_rank.array().log()).abs().maxCoeff());
    }
    diagnostics.max_weight_change = (gamma_poisson_cluster_weights(state.model)
        - old_weights).cwiseAbs().maxCoeff();
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

Eigen::MatrixXd pooled_residual_sketch(const CompactStatistics& stats,
    const StructuredState& state,
    const Eigen::Ref<const Eigen::MatrixXd>& sketch) {
    Eigen::MatrixXd action = stats.pooled_second_sketch;
    Eigen::VectorXd weighted_diagonal = Eigen::VectorXd::Zero(state.model.means.cols());
    for (Eigen::Index c = 0; c < state.model.means.rows(); ++c) {
        action.noalias() -= stats.membership(c)
            * state.model.means.row(c).transpose()
            * (state.model.means.row(c) * sketch);
        weighted_diagonal.array() += stats.membership(c)
            * state.model.variances.row(c).transpose().array();
    }
    action.array() -= sketch.array().colwise() * weighted_diagonal.array();
    return action;
}

Eigen::MatrixXd pooled_residual_covariance(
    const CompactStatistics& stats, const StructuredState& state) {
    if (!stats.dense) {
        throw std::invalid_argument(
            "Dense residual covariance requires dense sufficient statistics");
    }
    const Eigen::Index dim = state.model.means.cols();
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(dim, dim);
    for (Eigen::Index c = 0; c < state.model.means.rows(); ++c) {
        covariance += stats.second_dense[c];
        covariance.noalias() -= stats.membership(c)
            * state.model.means.row(c).transpose()
            * state.model.means.row(c);
        covariance.diagonal().array() -= stats.membership(c)
            * state.model.variances.row(c).transpose().array();
    }
    return 0.5 * (covariance + covariance.transpose());
}

void transport_orientation(StructuredState& state,
    const Eigen::Ref<const RowMajorMatrixXd>& previous,
    const GammaPoissonClusterFitOptions& options) {
    const Eigen::MatrixXd overlap = state.model.orientation.transpose() * previous;
    for (Eigen::Index c = 0; c < state.model.means.rows(); ++c) {
        const Eigen::VectorXd old_low =
            state.model.low_rank_variances.row(c).transpose();
        const Eigen::VectorXd old_total_diagonal =
            state.model.variances.row(c).transpose()
            + previous.array().square().matrix() * old_low;
        const Eigen::VectorXd new_low = (overlap.array().square().matrix()
            * old_low).cwiseMax(options.low_rank_variance_floor);
        state.model.low_rank_variances.row(c) = new_low.transpose();
        state.model.variances.row(c) = (old_total_diagonal
            - state.model.orientation.array().square().matrix() * new_low)
            .cwiseMax(options.variance_floor).transpose();
    }
}

void update_diagnostics(const CompactStatistics& stats, double alpha,
    GammaPoissonClusterFitResult& out) {
    out.diagnostics.elbo = stats.elbo_local
        + dirichlet_elbo(out.model.dirichlet_parameters, alpha);
    out.diagnostics.log_likelihood = stats.predictive_log_likelihood;
    out.diagnostics.elbo_trace.push_back(out.diagnostics.elbo);
}

void validate_model(const GammaPoissonClusterModel& model, int32_t dim) {
    const Eigen::Index components = model.dirichlet_parameters.size();
    const Eigen::Index rank = model.orientation.cols();
    if (components <= 0 || model.means.rows() != components
        || model.means.cols() != dim || model.variances.rows() != components
        || model.variances.cols() != dim || model.orientation.rows() != dim
        || model.low_rank_variances.rows() != components
        || model.low_rank_variances.cols() != rank
        || rank < 0 || rank > dim
        || !model.dirichlet_parameters.allFinite() || !model.means.allFinite()
        || !model.variances.allFinite() || !model.orientation.allFinite()
        || !model.low_rank_variances.allFinite()
        || (model.dirichlet_parameters.array() <= 0.0).any()
        || (model.variances.array() <= 0.0).any()
        || (rank > 0 && (model.low_rank_variances.array() <= 0.0).any())
        || (rank > 0 && (model.orientation.transpose() * model.orientation
            - Eigen::MatrixXd::Identity(rank, rank)).norm() > 1e-8)) {
        throw std::invalid_argument("Invalid Gamma-Poisson cluster model");
    }
}

} // namespace

Eigen::VectorXd gamma_poisson_cluster_weights(
    const GammaPoissonClusterModel& model) {
    if (model.dirichlet_parameters.size() == 0
        || !model.dirichlet_parameters.allFinite()
        || (model.dirichlet_parameters.array() <= 0.0).any()) {
        throw std::invalid_argument("Invalid Gamma-Poisson Dirichlet parameters");
    }
    return model.dirichlet_parameters / model.dirichlet_parameters.sum();
}

GammaPoissonConditionalMoments gamma_poisson_conditional_moments(
    const Eigen::Ref<const Eigen::VectorXd>& observation,
    const Eigen::Ref<const Eigen::VectorXd>& cluster_mean,
    const LowRankDiagonalCovariance& document_covariance,
    const LowRankDiagonalCovariance& cluster_covariance) {
    if (observation.size() == 0 || cluster_mean.size() != observation.size()
        || document_covariance.diagonal.size() != observation.size()
        || cluster_covariance.diagonal.size() != observation.size()) {
        throw std::invalid_argument("Conditional Gaussian moment dimensions do not match");
    }
    RowMajorMatrixXd total_factor(observation.size(),
        document_covariance.factor.cols() + cluster_covariance.factor.cols());
    if (document_covariance.factor.cols() > 0) {
        total_factor.leftCols(document_covariance.factor.cols()) =
            document_covariance.factor;
    }
    if (cluster_covariance.factor.cols() > 0) {
        total_factor.rightCols(cluster_covariance.factor.cols()) =
            cluster_covariance.factor;
    }
    LowRankDiagonalSolver solver(document_covariance.diagonal
        + cluster_covariance.diagonal, total_factor);
    PreparedComponent component;
    component.diagonal = cluster_covariance.diagonal;
    component.factor = cluster_covariance.factor;
    component.covariance_diagonal = component.diagonal;
    if (component.factor.cols() > 0) {
        component.covariance_diagonal.array() +=
            component.factor.array().square().rowwise().sum();
    }
    const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(
        observation.size(), observation.size());
    const RowMajorMatrixXd orientation = identity;
    const ConditionalContractions conditional = conditional_contractions(
        observation, cluster_mean, component, solver, orientation, &identity);
    const Eigen::VectorXd residual = observation - cluster_mean;
    GammaPoissonConditionalMoments out;
    out.log_likelihood = -0.5 * (observation.size() * LOG_2PI
        + solver.log_determinant() + solver.quadratic(residual));
    out.mean = conditional.mean;
    out.covariance = conditional.covariance_sketch;
    if ((out.covariance - conditional.covariance_projected).norm()
        > 1e-8 * (1.0 + out.covariance.norm())) {
        throw std::runtime_error(
            "Inconsistent compact conditional covariance contractions");
    }
    out.covariance.diagonal() = conditional.covariance_diagonal;
    out.covariance = 0.5 * (out.covariance + out.covariance.transpose());
    return out;
}

GammaPoissonClusterFitResult fit_gamma_poisson_cluster_mixture(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options) {
    const auto finite = [](double value) { return std::isfinite(value); };
    if (coordinates.mean.rows() == 0 || coordinates.mean.cols() == 0
        || coordinates.uncertainty_diagonal.rows() != coordinates.mean.rows()
        || coordinates.uncertainty_diagonal.cols() != coordinates.mean.cols()
        || coordinates.uncertainty_rank < 0
        || coordinates.uncertainty_factor.rows() != coordinates.mean.rows()
        || coordinates.uncertainty_factor.cols()
            != coordinates.mean.cols() * coordinates.uncertainty_rank
        || options.n_components <= 0 || options.max_iterations <= 0
        || options.kmeans_max_iterations <= 0 || options.convergence_patience <= 0
        || options.n_threads <= 0 || options.cluster_covariance_rank < 0
        || options.cluster_covariance_rank > coordinates.mean.cols()
        || static_cast<int32_t>(options.covariance_accumulation)
            < static_cast<int32_t>(GammaPoissonClusterFitOptions::
                CovarianceAccumulation::Auto)
        || static_cast<int32_t>(options.covariance_accumulation)
            > static_cast<int32_t>(GammaPoissonClusterFitOptions::
                CovarianceAccumulation::Compact)
        || options.diagonal_warmup_iterations < 0
        || options.orientation_update_interval <= 0
        || options.orientation_max_updates < 0 || options.orientation_patience <= 0
        || !finite(options.variance_floor) || options.variance_floor <= 0.0
        || !finite(options.low_rank_variance_floor)
        || options.low_rank_variance_floor <= 0.0
        || !finite(options.orientation_step) || options.orientation_step <= 0.0
        || options.orientation_step > 1.0
        || !finite(options.orientation_tolerance) || options.orientation_tolerance < 0.0
        || !finite(options.tolerance) || options.tolerance < 0.0
        || !finite(options.responsibility_p90_tolerance)
        || options.responsibility_p90_tolerance < 0.0
        || options.responsibility_p90_tolerance > 2.0
        || !finite(options.top_assignment_change_tolerance)
        || options.top_assignment_change_tolerance < 0.0
        || options.top_assignment_change_tolerance > 1.0
        || !finite(options.dirichlet_concentration)
        || options.dirichlet_concentration <= 0.0) {
        throw std::invalid_argument("Invalid structured Gamma-Poisson mixture input or options");
    }
    if (!coordinates.mean.allFinite()
        || !coordinates.uncertainty_diagonal.allFinite()
        || !coordinates.uncertainty_factor.allFinite()
        || (coordinates.uncertainty_diagonal.array() <= 0.0).any()) {
        throw std::invalid_argument("Structured mixture inputs must be finite with positive uncertainty");
    }
    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
    const int32_t components = std::min(options.n_components, n);
    const int32_t rank = options.cluster_covariance_rank;
    const bool dense_accumulation = rank > 0 && (
        options.covariance_accumulation
            == GammaPoissonClusterFitOptions::CovarianceAccumulation::Dense
        || (options.covariance_accumulation
                == GammaPoissonClusterFitOptions::CovarianceAccumulation::Auto
            && dim + 1 <= 48));
    const double alpha = options.dirichlet_concentration / components;
    tbb::global_control parallelism(tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(options.n_threads));

    DenseKMeansOptions kmeans_options;
    kmeans_options.n_clusters = components;
    kmeans_options.max_iterations = options.kmeans_max_iterations;
    kmeans_options.seed = options.seed;
    DenseKMeansResult kmeans = dense_kmeans(coordinates.mean, kmeans_options);
    StructuredState state;
    state.model.means = kmeans.centers;
    state.model.variances = RowMajorMatrixXd::Zero(components, dim);
    Eigen::MatrixXd initial_shared = Eigen::MatrixXd::Zero(dim, dim);
    for (int32_t d = 0; d < n; ++d) {
        const int32_t c = kmeans.assignments(d);
        const Eigen::VectorXd residual = coordinates.mean.row(d).transpose()
            - state.model.means.row(c).transpose();
        state.model.variances.row(c).array() += residual.array().square().transpose();
        initial_shared.noalias() += residual * residual.transpose();
    }
    for (int32_t c = 0; c < components; ++c) {
        state.model.variances.row(c) = (state.model.variances.row(c)
            / static_cast<double>(kmeans.counts(c))).array()
            .max(options.variance_floor);
    }
    state.model.orientation.resize(dim, 0);
    state.model.low_rank_variances.resize(components, 0);
    state.model.dirichlet_parameters = kmeans.counts.cast<double>().array() + alpha;
    GammaPoissonClusterFitResult out;
    out.diagnostics.covariance_accumulation = rank == 0
        ? "diagonal" : (dense_accumulation ? "dense" : "compact");
    out.diagnostics.kmeans_iterations = kmeans.iterations;
    out.diagnostics.kmeans_inertia = kmeans.inertia;
    out.diagnostics.kmeans_converged = kmeans.converged;
    const Eigen::MatrixXd sketch = rank > 0 && !dense_accumulation
        ? make_orientation_sketch(dim, rank, options.seed) : Eigen::MatrixXd(dim, 0);

    CompactStatistics stats(components, dim, 0, 0, dense_accumulation);
    const int32_t warmup = rank > 0 ? options.diagonal_warmup_iterations : 0;
    for (int32_t iteration = 0; iteration < warmup; ++iteration) {
        const Eigen::MatrixXd* use_sketch = !dense_accumulation
                && iteration + 1 == warmup
            ? &sketch : nullptr;
        stats = compact_e_step(coordinates, state, options.n_threads,
            out.responsibilities, use_sketch, true, dense_accumulation);
        out.model = state.model;
        update_diagnostics(stats, alpha, out);
        compact_m_step(stats, state, alpha, options, out.diagnostics);
        ++out.diagnostics.warmup_iterations;
    }

    if (rank > 0) {
        if (warmup > 0) {
            const RowMajorMatrixXd fallback = leading_orientation(
                initial_shared, rank);
            state.model.orientation = dense_accumulation
                ? leading_orientation(
                    pooled_residual_covariance(stats, state), rank)
                : orientation_from_signed_sketch(sketch,
                    pooled_residual_sketch(stats, state, sketch),
                    fallback, rank);
        } else {
            state.model.orientation = leading_orientation(initial_shared, rank);
        }
        state.model.low_rank_variances = RowMajorMatrixXd::Constant(
            components, rank, options.low_rank_variance_floor);
        int32_t stable_updates = 0;
        for (int32_t update = 0; update < options.orientation_max_updates; ++update) {
            for (int32_t iteration = 0;
                 iteration < options.orientation_update_interval; ++iteration) {
                const Eigen::MatrixXd* use_sketch =
                    !dense_accumulation
                        && iteration + 1 == options.orientation_update_interval
                    ? &sketch : nullptr;
                stats = compact_e_step(coordinates, state, options.n_threads,
                    out.responsibilities, use_sketch, true,
                    dense_accumulation);
                out.model = state.model;
                update_diagnostics(stats, alpha, out);
                compact_m_step(stats, state, alpha, options, out.diagnostics);
                ++out.diagnostics.structured_iterations;
            }
            const RowMajorMatrixXd previous = state.model.orientation;
            RowMajorMatrixXd candidate = dense_accumulation
                ? leading_orientation(
                    pooled_residual_covariance(stats, state), rank)
                : orientation_from_signed_sketch(sketch,
                    pooled_residual_sketch(stats, state, sketch), previous, rank);
            out.diagnostics.orientation_change = align_and_blend_orientation(
                state.model.orientation, std::move(candidate), options.orientation_step);
            transport_orientation(state, previous, options);
            ++out.diagnostics.orientation_updates;
            stable_updates = out.diagnostics.orientation_change
                    <= options.orientation_tolerance
                ? stable_updates + 1 : 0;
            if (stable_updates >= options.orientation_patience) {
                out.diagnostics.orientation_converged = true;
                break;
            }
        }
    } else {
        out.diagnostics.orientation_converged = true;
        out.diagnostics.orientation_change = 0.0;
    }

    RowMajorMatrixXd previous_responsibilities;
    double previous_elbo = -std::numeric_limits<double>::infinity();
    int32_t stable_iterations = 0;
    for (int32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        stats = compact_e_step(coordinates, state, options.n_threads,
            out.responsibilities, nullptr, true, dense_accumulation);
        out.model = state.model;
        update_diagnostics(stats, alpha, out);
        out.effective_membership = stats.membership;
        ++out.diagnostics.structured_iterations;
        if (previous_responsibilities.rows() == n) {
            const GammaPoissonResponsibilityChange change =
                gamma_poisson_responsibility_change(
                    previous_responsibilities, out.responsibilities);
            out.diagnostics.mean_responsibility_l1_change = change.mean_l1;
            out.diagnostics.p90_responsibility_l1_change = change.p90_l1;
            out.diagnostics.top_assignment_change_fraction =
                change.top_assignment_fraction;
            out.diagnostics.relative_elbo_change = std::abs(
                out.diagnostics.elbo - previous_elbo)
                / (1.0 + std::abs(previous_elbo));
            const bool stable = out.diagnostics.relative_elbo_change
                    <= options.tolerance
                && out.diagnostics.p90_responsibility_l1_change
                    <= options.responsibility_p90_tolerance
                && out.diagnostics.top_assignment_change_fraction
                    <= options.top_assignment_change_tolerance;
            stable_iterations = stable ? stable_iterations + 1 : 0;
            if (stable_iterations >= options.convergence_patience) {
                out.diagnostics.converged = true;
                break;
            }
        }
        previous_responsibilities = out.responsibilities;
        previous_elbo = out.diagnostics.elbo;
        compact_m_step(stats, state, alpha, options, out.diagnostics);
    }
    if (!out.diagnostics.converged) {
        stats = compact_e_step(coordinates, state, options.n_threads,
            out.responsibilities, nullptr, true, dense_accumulation);
        out.model = state.model;
        update_diagnostics(stats, alpha, out);
        out.effective_membership = stats.membership;
    }
    out.model = state.model;
    out.diagnostics.iterations = out.diagnostics.warmup_iterations
        + out.diagnostics.structured_iterations;
    return out;
}

GammaPoissonClusterScore score_gamma_poisson_cluster_mixture(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, int32_t n_threads) {
    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
    const int32_t components = static_cast<int32_t>(model.dirichlet_parameters.size());
    if (n <= 0 || dim <= 0 || n_threads <= 0
        || coordinates.uncertainty_diagonal.rows() != n
        || coordinates.uncertainty_diagonal.cols() != dim
        || coordinates.uncertainty_rank < 0
        || coordinates.uncertainty_factor.rows() != n
        || coordinates.uncertainty_factor.cols()
            != dim * coordinates.uncertainty_rank
        || !coordinates.mean.allFinite()
        || !coordinates.uncertainty_diagonal.allFinite()
        || !coordinates.uncertainty_factor.allFinite()
        || (coordinates.uncertainty_diagonal.array() <= 0.0).any()) {
        throw std::invalid_argument("Invalid Gamma-Poisson cluster scoring input");
    }
    validate_model(model, dim);
    tbb::global_control parallelism(tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(n_threads));
    StructuredState state{model};
    RowMajorMatrixXd ignored;
    CompactStatistics stats = compact_e_step(
        coordinates, state, n_threads, ignored, nullptr, false, false);
    GammaPoissonClusterScore out;
    out.responsibilities = std::move(ignored);
    out.predictive_log_likelihood = stats.predictive_log_likelihood;
    return out;
}

double gamma_poisson_cluster_log_volume(
    const GammaPoissonClusterModel& model, int32_t component) {
    validate_model(model, static_cast<int32_t>(model.variances.cols()));
    if (component < 0 || component >= model.variances.rows()) {
        throw std::invalid_argument("Invalid component for cluster volume");
    }
    const RowMajorMatrixXd factor = covariance_factor(
        model.orientation, model.low_rank_variances.row(component));
    LowRankDiagonalSolver solver(model.variances.row(component).transpose(), factor);
    return 0.5 * solver.log_determinant();
}

GammaPoissonClusterSeparation gamma_poisson_cluster_separation(
    const GammaPoissonClusterModel& model, int32_t first, int32_t second) {
    validate_model(model, static_cast<int32_t>(model.variances.cols()));
    if (first < 0 || second < 0 || first >= model.variances.rows()
        || second >= model.variances.rows() || first == second) {
        throw std::invalid_argument("Invalid components for cluster separation");
    }
    const Eigen::VectorXd average_diagonal = 0.5 * (
        model.variances.row(first) + model.variances.row(second)).transpose();
    const Eigen::RowVectorXd average_low_rank = 0.5 * (
        model.low_rank_variances.row(first)
        + model.low_rank_variances.row(second));
    const RowMajorMatrixXd average_factor = covariance_factor(
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
