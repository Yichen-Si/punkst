#include "gamma_pois_cluster_internal.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

GammaPoissonClusterSufficientStatistics::GammaPoissonClusterSufficientStatistics(
    int32_t components, int32_t dim, int32_t rank, int32_t sketch_size,
    bool dense_)
    : membership(Eigen::VectorXd::Zero(std::max(components, int32_t{0}))),
      first(RowMajorMatrixXd::Zero(std::max(components, int32_t{0}),
          std::max(dim, int32_t{0}))),
      second_diagonal(RowMajorMatrixXd::Zero(
          std::max(components, int32_t{0}),
          dense_ ? 0 : std::max(dim, int32_t{0}))),
      second_projected(dense_ ? 0 : std::max(components, int32_t{0}),
          Eigen::MatrixXd::Zero(std::max(rank, int32_t{0}),
              std::max(rank, int32_t{0}))),
      pooled_second_sketch(Eigen::MatrixXd::Zero(
          dense_ ? 0 : std::max(dim, int32_t{0}),
          std::max(sketch_size, int32_t{0}))),
      second_dense(dense_ ? std::max(components, int32_t{0}) : 0,
          Eigen::MatrixXd::Zero(std::max(dim, int32_t{0}),
              std::max(dim, int32_t{0}))), dense(dense_) {
    if (components < 0 || dim < 0 || rank < 0 || sketch_size < 0) {
        throw std::invalid_argument(
            "Invalid Gamma-Poisson sufficient-statistics dimensions");
    }
}

void GammaPoissonClusterSufficientStatistics::add_scaled(
    const GammaPoissonClusterSufficientStatistics& other, double factor) {
    bool compatible = dense == other.dense
        && membership.size() == other.membership.size()
        && first.rows() == other.first.rows() && first.cols() == other.first.cols()
        && second_diagonal.rows() == other.second_diagonal.rows()
        && second_diagonal.cols() == other.second_diagonal.cols()
        && second_projected.size() == other.second_projected.size()
        && pooled_second_sketch.rows() == other.pooled_second_sketch.rows()
        && pooled_second_sketch.cols() == other.pooled_second_sketch.cols()
        && second_dense.size() == other.second_dense.size();
    for (size_t c = 0; compatible && c < second_projected.size(); ++c) {
        compatible = second_projected[c].rows() == other.second_projected[c].rows()
            && second_projected[c].cols() == other.second_projected[c].cols();
    }
    for (size_t c = 0; compatible && c < second_dense.size(); ++c) {
        compatible = second_dense[c].rows() == other.second_dense[c].rows()
            && second_dense[c].cols() == other.second_dense[c].cols();
    }
    if (!compatible || !std::isfinite(factor) || factor < 0.0) {
        throw std::invalid_argument(
            "Incompatible Gamma-Poisson sufficient-statistics accumulation");
    }
    membership.noalias() += factor * other.membership;
    first.noalias() += factor * other.first;
    second_diagonal.noalias() += factor * other.second_diagonal;
    for (size_t c = 0; c < second_projected.size(); ++c) {
        second_projected[c].noalias() += factor * other.second_projected[c];
    }
    pooled_second_sketch.noalias() += factor * other.pooled_second_sketch;
    for (size_t c = 0; c < second_dense.size(); ++c) {
        second_dense[c].noalias() += factor * other.second_dense[c];
    }
    elbo_local += factor * other.elbo_local;
    predictive_log_likelihood += factor * other.predictive_log_likelihood;
}

void GammaPoissonClusterSufficientStatistics::scale(double factor) {
    if (!std::isfinite(factor) || factor < 0.0) {
        throw std::invalid_argument(
            "Invalid Gamma-Poisson sufficient-statistics scale");
    }
    membership *= factor;
    first *= factor;
    second_diagonal *= factor;
    for (Eigen::MatrixXd& value : second_projected) value *= factor;
    pooled_second_sketch *= factor;
    for (Eigen::MatrixXd& value : second_dense) value *= factor;
    elbo_local *= factor;
    predictive_log_likelihood *= factor;
}

void GammaPoissonClusterSufficientStatistics::interpolate(
    const GammaPoissonClusterSufficientStatistics& target, double step) {
    if (!std::isfinite(step) || step < 0.0 || step > 1.0) {
        throw std::invalid_argument(
            "Invalid Gamma-Poisson sufficient-statistics interpolation step");
    }
    if (this == &target) return;
    scale(1.0 - step);
    add_scaled(target, step);
}

namespace gamma_pois_cluster_detail {

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

RowMajorMatrixXd covariance_factor(
    const Eigen::Ref<const RowMajorMatrixXd>& orientation,
    const Eigen::Ref<const Eigen::RowVectorXd>& variances) {
    RowMajorMatrixXd out(orientation.rows(), orientation.cols());
    if (orientation.cols() > 0) {
        out = orientation.array().rowwise() * variances.array().sqrt();
    }
    return out;
}

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

PreparedEStepModel prepare_e_step_model(const GammaPoissonClusterModel& model,
    bool prepare_dense_covariance) {
    const int32_t components = static_cast<int32_t>(model.means.rows());
    const int32_t rank = static_cast<int32_t>(model.orientation.cols());
    PreparedEStepModel out;
    out.expected_log = expected_log_dirichlet(
        model.dirichlet_parameters);
    out.weights = gamma_poisson_cluster_weights(model);
    out.components.resize(components);
    out.marginal_variances = model.variances;
    for (int32_t c = 0; c < components; ++c) {
        PreparedComponent& prepared = out.components[c];
        prepared.diagonal = model.variances.row(c).transpose();
        prepared.factor = covariance_factor(model.orientation,
            model.low_rank_variances.row(c));
        prepared.covariance_diagonal = prepared.diagonal;
        if (rank > 0) {
            prepared.covariance_diagonal.array() +=
                prepared.factor.array().square().rowwise().sum();
        }
        out.marginal_variances.row(c) =
            prepared.covariance_diagonal.transpose();
        if (prepare_dense_covariance) {
            prepared.covariance = prepared.diagonal.asDiagonal();
            if (rank > 0) {
                prepared.covariance.noalias() += prepared.factor
                    * prepared.factor.transpose();
            }
        }
    }
    return out;
}

void retain_top_component(double value, int32_t component,
    std::array<double, 3>& top_values, std::array<int32_t, 3>& top_ids) {
    int32_t position = 3;
    for (int32_t top = 0; top < 3; ++top) {
        if (value > top_values[top]
            || (value == top_values[top]
                && (top_ids[top] < 0 || component < top_ids[top]))) {
            position = top;
            break;
        }
    }
    if (position == 3) return;
    for (int32_t top = 2; top > position; --top) {
        top_values[top] = top_values[top - 1];
        top_ids[top] = top_ids[top - 1];
    }
    top_values[position] = value;
    top_ids[position] = component;
}

static ClusterStatistics run_e_step_prepared_impl(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, const PreparedEStepModel& prepared_model,
    const int32_t* document_indices,
    int32_t n_documents, int32_t n_threads,
    RowMajorMatrixXd* responsibilities,
    const Eigen::MatrixXd* sketch, bool collect_moments,
    bool dense_accumulation, const int32_t* candidates = nullptr,
    int32_t candidate_stride = 0, int32_t* document_top_components = nullptr) {
    const int32_t n = document_indices
        ? n_documents
        : static_cast<int32_t>(coordinates.mean.rows());
    const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
    const int32_t components = static_cast<int32_t>(model.means.rows());
    const int32_t document_rank = coordinates.uncertainty_rank;
    const int32_t rank = static_cast<int32_t>(model.orientation.cols());
    const int32_t sketch_size = sketch ? static_cast<int32_t>(sketch->cols()) : 0;
    if (candidates && candidate_stride <= 0) {
        throw std::invalid_argument("Invalid Gamma-Poisson candidate E-step view");
    }
    const int32_t shards = std::max(1, std::min(n_threads, n));
    if (prepared_model.components.size() != static_cast<size_t>(components)
        || prepared_model.expected_log.size() != components
        || prepared_model.weights.size() != components) {
        throw std::invalid_argument("Invalid prepared Gamma-Poisson E-step model");
    }
    const int32_t maximum_local_components = candidates
        ? candidate_stride : components;
    if (responsibilities) {
        responsibilities->resize(n, maximum_local_components);
    }
    const int32_t statistics_dim = collect_moments ? dim : 0;
    const int32_t statistics_rank = collect_moments ? rank : 0;
    const int32_t statistics_sketch_size = collect_moments ? sketch_size : 0;
    const bool dense_statistics = collect_moments && dense_accumulation;
    std::vector<ClusterStatistics> partial;
    partial.reserve(shards);
    for (int32_t shard = 0; shard < shards; ++shard) {
        partial.emplace_back(components, statistics_dim, statistics_rank,
            statistics_sketch_size, dense_statistics);
    }
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, shards),
        [&](const tbb::blocked_range<int32_t>& range) {
            for (int32_t shard = range.begin(); shard < range.end(); ++shard) {
                ClusterStatistics& stats = partial[shard];
                const int32_t begin = n * shard / shards;
                const int32_t end = n * (shard + 1) / shards;
                std::vector<double> gaussian(maximum_local_components);
                std::vector<double> log_joint(maximum_local_components);
                std::vector<LowRankDiagonalSolver> solvers(
                    maximum_local_components);
                RowMajorMatrixXd total_factor(dim, document_rank + rank);
                for (int32_t position = begin; position < end; ++position) {
                    const int32_t d = document_indices
                        ? document_indices[position] : position;
                    int32_t local_components = components;
                    if (candidates) {
                        local_components = 0;
                        while (local_components < candidate_stride
                            && candidates[position * candidate_stride
                                + local_components] >= 0) {
                            ++local_components;
                        }
                        if (local_components == 0) {
                            throw std::runtime_error(
                                "Gamma-Poisson candidate set is empty");
                        }
                    }
                    const auto component_at = [&](int32_t local) {
                        return candidates
                            ? candidates[position * candidate_stride + local]
                            : local;
                    };
                    const Eigen::VectorXd observation =
                        coordinates.mean.row(d).transpose();
                    const auto document_factor = coordinates.factor(d);
                    double maximum = -std::numeric_limits<double>::infinity();
                    double predictive_maximum = maximum;
                    for (int32_t local = 0; local < local_components; ++local) {
                        const int32_t c = component_at(local);
                        if (c < 0 || c >= components) {
                            throw std::runtime_error(
                                "Gamma-Poisson candidate component is out of range");
                        }
                        const PreparedComponent& prepared =
                            prepared_model.components[c];
                        if (document_rank > 0) {
                            total_factor.leftCols(document_rank) = document_factor;
                        }
                        if (rank > 0) {
                            total_factor.rightCols(rank) = prepared.factor;
                        }
                        const Eigen::VectorXd total_diagonal =
                            coordinates.uncertainty_diagonal.row(d).transpose()
                            + prepared.diagonal;
                        solvers[local].compute(total_diagonal, total_factor);
                        const Eigen::VectorXd residual = observation
                            - model.means.row(c).transpose();
                        gaussian[local] = -0.5 * (dim * LOG_2PI
                            + solvers[local].log_determinant()
                            + solvers[local].quadratic(residual));
                        log_joint[local] = prepared_model.expected_log(c)
                            + gaussian[local];
                        maximum = std::max(maximum, log_joint[local]);
                        predictive_maximum = std::max(predictive_maximum,
                            std::log(std::max(
                                prepared_model.weights(c), 1e-300))
                                + gaussian[local]);
                    }
                    double sum = 0.0;
                    double predictive_sum = 0.0;
                    for (int32_t local = 0; local < local_components; ++local) {
                        const int32_t c = component_at(local);
                        sum += std::exp(log_joint[local] - maximum);
                        predictive_sum += std::exp(
                            std::log(std::max(
                                prepared_model.weights(c), 1e-300))
                            + gaussian[local] - predictive_maximum);
                    }
                    const double normalizer = maximum + std::log(sum);
                    std::array<double, 3> top_values = {
                        -std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity(),
                        -std::numeric_limits<double>::infinity()};
                    std::array<int32_t, 3> top_ids = {-1, -1, -1};
                    for (int32_t local = 0; local < local_components; ++local) {
                        const int32_t c = component_at(local);
                        const double responsibility = std::exp(
                            log_joint[local] - normalizer);
                        if (responsibilities) {
                            (*responsibilities)(position, local) = responsibility;
                        }
                        if (document_top_components) {
                            retain_top_component(responsibility, c,
                                top_values, top_ids);
                        }
                        if (responsibility > 0.0) {
                            stats.elbo_local += responsibility * (
                                prepared_model.expected_log(c) + gaussian[local]
                                - std::log(responsibility));
                        }
                        if (!collect_moments) continue;
                        if (dense_accumulation) {
                            const Eigen::VectorXd residual = observation
                                - model.means.row(c).transpose();
                            const Eigen::VectorXd conditional_mean =
                                model.means.row(c).transpose()
                                + apply_component_vector(
                                    prepared_model.components[c],
                                    solvers[local].solve_vector(residual));
                            Eigen::MatrixXd conditional_covariance =
                                prepared_model.components[c].covariance
                                - apply_component_matrix(
                                    prepared_model.components[c],
                                    solvers[local].solve_matrix(
                                        prepared_model.components[c].covariance));
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
                            observation, model.means.row(c).transpose(),
                            prepared_model.components[c], solvers[local],
                            model.orientation, sketch);
                        stats.membership(c) += responsibility;
                        stats.first.row(c) += responsibility * conditional.mean.transpose();
                        stats.second_diagonal.row(c) += responsibility * (
                            conditional.covariance_diagonal.array()
                            + conditional.mean.array().square()).matrix().transpose();
                        if (rank > 0) {
                            const Eigen::VectorXd projected_mean =
                                model.orientation.transpose() * conditional.mean;
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
                    if (document_top_components) {
                        for (int32_t top = 0; top < 3; ++top) {
                            document_top_components[
                                static_cast<size_t>(d) * 3 + top] = top_ids[top];
                        }
                    }
                    stats.predictive_log_likelihood += predictive_maximum
                        + std::log(predictive_sum);
                }
            }
        });
    ClusterStatistics out(components, statistics_dim, statistics_rank,
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



ClusterStatistics run_e_step_prepared(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, const PreparedEStepModel& prepared,
    const EStepRequest& request) {
    return run_e_step_prepared_impl(coordinates, model, prepared,
        request.document_indices, request.n_documents, request.n_threads,
        request.responsibilities, request.sketch, request.collect_moments,
        request.dense_accumulation, request.candidates,
        request.candidate_stride, request.document_top_components);
}

ClusterStatistics run_e_step(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, const EStepRequest& request) {
    const PreparedEStepModel prepared = prepare_e_step_model(
        model, request.dense_accumulation && request.collect_moments);
    return run_e_step_prepared(coordinates, model, prepared, request);
}


} // namespace gamma_pois_cluster_detail

GammaPoissonClusterBatchExpectation gamma_poisson_cluster_e_step(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model,
    const Eigen::Ref<const Eigen::VectorXi>& document_indices,
    int32_t n_threads, bool dense_accumulation,
    bool collect_responsibilities) {
    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
    if (n <= 0 || dim <= 0 || n_threads <= 0 || document_indices.size() <= 0
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
        throw std::invalid_argument(
            "Invalid Gamma-Poisson cluster E-step input");
    }
    for (Eigen::Index i = 0; i < document_indices.size(); ++i) {
        if (document_indices(i) < 0 || document_indices(i) >= n) {
            throw std::invalid_argument(
                "Gamma-Poisson cluster E-step document index is out of range");
        }
    }
    gamma_pois_cluster_detail::validate_model(model, dim);
    tbb::global_control parallelism(tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(n_threads));
    GammaPoissonClusterBatchExpectation out;
    RowMajorMatrixXd* responsibilities = collect_responsibilities
        ? &out.responsibilities : nullptr;
    gamma_pois_cluster_detail::EStepRequest request;
    request.document_indices = document_indices.data();
    request.n_documents = static_cast<int32_t>(document_indices.size());
    request.n_threads = n_threads;
    request.responsibilities = responsibilities;
    request.collect_moments = true;
    request.dense_accumulation = dense_accumulation;
    out.statistics = gamma_pois_cluster_detail::run_e_step(
        coordinates, model, request);
    return out;
}

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
    gamma_pois_cluster_detail::PreparedComponent component;
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
    const gamma_pois_cluster_detail::ConditionalContractions conditional =
        gamma_pois_cluster_detail::conditional_contractions(
        observation, cluster_mean, component, solver, orientation, &identity);
    const Eigen::VectorXd residual = observation - cluster_mean;
    GammaPoissonConditionalMoments out;
    out.log_likelihood = -0.5 * (observation.size() * gamma_pois_cluster_detail::LOG_2PI
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


GammaPoissonClusterScore score_gamma_poisson_cluster_mixture(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, int32_t n_threads) {
    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
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
    gamma_pois_cluster_detail::validate_model(model, dim);
    tbb::global_control parallelism(tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(n_threads));
        RowMajorMatrixXd ignored;
    gamma_pois_cluster_detail::EStepRequest request;
    request.n_threads = n_threads;
    request.responsibilities = &ignored;
    request.collect_moments = false;
    const GammaPoissonClusterSufficientStatistics stats =
        gamma_pois_cluster_detail::run_e_step(coordinates, model, request);
    GammaPoissonClusterScore out;
    out.responsibilities = std::move(ignored);
    out.predictive_log_likelihood = stats.predictive_log_likelihood;
    return out;
}
