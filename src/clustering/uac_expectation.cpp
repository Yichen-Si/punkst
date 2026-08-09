#include "clustering/uac_expectation_internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include <tbb/parallel_for.h>

namespace uac::detail {


int32_t expectation_shards(int32_t documents, int32_t components,
    int32_t dimension, int32_t factor_rank) {
    constexpr uint64_t memory_budget = 64ull * 1024ull * 1024ull;
    constexpr int32_t maximum_shards = 32;
    uint64_t bytes = sizeof(double) * static_cast<uint64_t>(components)
        * (1 + dimension);
    if (factor_rank < 0) {
        bytes += sizeof(double) * static_cast<uint64_t>(components)
            * dimension * dimension;
    } else {
        bytes += sizeof(double) * static_cast<uint64_t>(components)
            * (dimension + factor_rank + factor_rank * factor_rank
                + dimension * factor_rank);
    }
    const int32_t memory_shards = static_cast<int32_t>(std::max<uint64_t>(
        1, memory_budget / std::max<uint64_t>(1, bytes)));
    return std::max(1, std::min({documents, maximum_shards, memory_shards}));
}

uint64_t expectation_block_bytes(int32_t components, int32_t dimension,
    int32_t factor_rank) {
    uint64_t values = static_cast<uint64_t>(components) * (1 + dimension);
    if (factor_rank < 0) {
        values += static_cast<uint64_t>(components) * dimension * dimension;
    } else {
        values += static_cast<uint64_t>(components)
            * (dimension + factor_rank + factor_rank * factor_rank
                + dimension * factor_rank);
    }
    return sizeof(double) * values;
}

void reduce_expectation_blocks(Expectation& out,
    const std::vector<ExpectationBlock>& blocks) {
    for (const auto& block : blocks) {
        if (block.membership.size() > 0) {
            out.membership += block.membership;
            out.first += block.first;
        }
        out.log_likelihood += block.log_likelihood;
        out.log_likelihood_upper += block.log_likelihood_upper;
        out.responsibility_entropy_sum +=
            block.responsibility_entropy_sum;
        out.component_bound_seconds += block.component_bound_seconds;
        out.evaluated_component_documents +=
            block.evaluated_component_documents;
        out.possible_component_documents +=
            block.possible_component_documents;
        out.full_component_documents += block.full_component_documents;
        out.component_bound_violations +=
            block.component_bound_violations;
        out.omitted_component_mass_sum +=
            block.omitted_component_mass_sum;
        out.maximum_omitted_component_mass = std::max(
            out.maximum_omitted_component_mass,
            block.maximum_omitted_component_mass);
        if (block.membership.size() == 0) {
            continue;
        } else if (out.sum_y2.size() > 0) {
            out.sum_y2 += block.sum_y2;
            out.sum_f += block.sum_f;
            for (size_t c = 0; c < out.sum_ff.size(); ++c) {
                out.sum_ff[c] += block.sum_ff[c];
                out.sum_yf[c] += block.sum_yf[c];
            }
        } else {
            for (size_t c = 0; c < out.second.size(); ++c) {
                out.second[c] += block.second[c];
            }
        }
    }
}

Expectation empty_expectation(int32_t documents, int32_t components,
    int32_t dimension, int32_t factor_rank,
    bool accumulate_moments) {
    Expectation out;
    out.documents = documents;
    if (!accumulate_moments) return out;
    out.membership = Eigen::VectorXd::Zero(components);
    out.first = RowMajorMatrixXd::Zero(components, dimension);
    out.second.resize(components);
    if (factor_rank < 0) {
        for (auto& value : out.second) {
            value = Eigen::MatrixXd::Zero(dimension, dimension);
        }
    } else {
        out.sum_y2 = RowMajorMatrixXd::Zero(components, dimension);
        out.sum_f = RowMajorMatrixXd::Zero(components, factor_rank);
        out.sum_ff.assign(components,
            Eigen::MatrixXd::Zero(factor_rank, factor_rank));
        out.sum_yf.assign(components,
            Eigen::MatrixXd::Zero(dimension, factor_rank));
    }
    return out;
}

void accumulate_expectation(Expectation& target, const Expectation& source) {
    target.membership += source.membership;
    target.first += source.first;
    target.log_likelihood += source.log_likelihood;
    target.log_likelihood_upper += source.log_likelihood_upper;
    target.responsibility_entropy_sum +=
        source.responsibility_entropy_sum;
    target.component_bound_seconds += source.component_bound_seconds;
    target.evaluated_component_documents +=
        source.evaluated_component_documents;
    target.possible_component_documents +=
        source.possible_component_documents;
    target.full_component_documents += source.full_component_documents;
    target.component_bound_violations +=
        source.component_bound_violations;
    target.omitted_component_mass_sum +=
        source.omitted_component_mass_sum;
    target.maximum_omitted_component_mass = std::max(
        target.maximum_omitted_component_mass,
        source.maximum_omitted_component_mass);
    if (target.sum_y2.size() > 0) {
        target.sum_y2 += source.sum_y2;
        target.sum_f += source.sum_f;
        for (size_t c = 0; c < target.sum_ff.size(); ++c) {
            target.sum_ff[c] += source.sum_ff[c];
            target.sum_yf[c] += source.sum_yf[c];
        }
    } else {
        for (size_t c = 0; c < target.second.size(); ++c) {
            target.second[c] += source.second[c];
        }
    }
}

Expectation map_expectation(const Dataset& data, const Model& model,
    const ExpectationRequest& request,
    const ComponentScreeningOptions& screening) {
    validate_component_screening(screening);
    const bool screen =
        screening.mode != ComponentScreeningMode::Off;
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    const int32_t factor_rank = model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(model.factor_covariances.front().factor.cols())
        : -1;
    Expectation out = empty_expectation(documents, components, dimension,
        factor_rank, request.accumulate_moments);
    if (request.store_responsibilities) {
        out.responsibilities.resize(documents, components);
        out.per_document_evaluated_components.resize(documents);
        out.per_document_omitted_component_mass.resize(documents);
    }
    std::vector<LowRankDiagonalSolver> factor_solvers;
    std::vector<Eigen::MatrixXd> factor_beta, factor_conditional;
    if (factor_rank >= 0) {
        factor_solvers.reserve(components);
        for (int32_t c = 0; c < components; ++c) {
            const auto& covariance = model.factor_covariances[c];
            factor_solvers.emplace_back(
                covariance.diagonal, covariance.factor);
            if (request.accumulate_moments) {
                factor_beta.push_back(factor_solvers.back().solve_matrix(
                    covariance.factor).transpose());
                factor_conditional.push_back(
                    Eigen::MatrixXd::Identity(factor_rank, factor_rank)
                    - factor_beta.back() * covariance.factor);
            }
        }
    }
    const std::vector<DenseGaussianSolver> dense_solvers =
        dense_model_solvers(model);
    const std::vector<double> eigenvalue_upper = screen
        ? model_eigenvalue_upper_bounds(model) : std::vector<double>{};
    Eigen::VectorXd gaussian_constant = Eigen::VectorXd::Constant(
        components, -std::numeric_limits<double>::infinity());
    for (int32_t c = 0; c < components; ++c) {
        if (!(model.weights(c) > 0.0)) continue;
        const double logdet = factor_rank < 0
            ? dense_solvers[c].log_determinant
            : factor_solvers[c].log_determinant();
        gaussian_constant(c) = std::log(model.weights(c))
            - 0.5 * (dimension * kLog2Pi + logdet);
    }
    const int32_t possible_components = active_component_count(model);
    const int32_t requested_blocks = expectation_shards(
        documents, components, dimension, factor_rank);
    const int32_t block_size =
        (documents + requested_blocks - 1) / requested_blocks;
    const int32_t n_blocks = (documents + block_size - 1) / block_size;
    std::vector<ExpectationBlock> blocks;
    blocks.reserve(n_blocks);
    for (int32_t block = 0; block < n_blocks; ++block) {
        blocks.emplace_back(components, dimension, factor_rank,
            request.accumulate_moments);
    }
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        ExpectationBlock& block = blocks[block_index];
        Eigen::VectorXd responsibility(components);
        Eigen::VectorXd upper(components);
        Eigen::VectorXd value(dimension);
        Eigen::VectorXd dense_standardized;
        Eigen::VectorXd factor_residual;
        ComponentScreeningWorkspace screening_workspace;
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        for (int32_t d = begin; d < end; ++d) {
            value = data.coordinates.row(d).transpose();
            auto exact_score = [&](int32_t c) {
                    if (factor_rank < 0) {
                        return std::log(model.weights(c))
                            + dense_solvers[c].log_density(
                                value, dense_standardized);
                    }
                    factor_residual = value
                        - model.means.row(c).transpose();
                    return std::log(model.weights(c)) - 0.5
                        * (dimension * kLog2Pi
                            + factor_solvers[c].log_determinant()
                            + factor_solvers[c].quadratic(factor_residual));
                };
            ScreenedComponents selected;
            if (screen) {
                const auto bound_start = std::chrono::steady_clock::now();
                upper.setConstant(-std::numeric_limits<double>::infinity());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    upper(c) = gaussian_constant(c) - 0.5
                        * (value - model.means.row(c).transpose()).squaredNorm()
                        / eigenvalue_upper[c];
                }
                block.component_bound_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now()
                        - bound_start).count();
                selected = screen_component_scores(
                    upper, screening, true, exact_score,
                    &screening_workspace);
            } else {
                selected.score = Eigen::VectorXd::Constant(
                    components, -std::numeric_limits<double>::infinity());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    selected.score(c) = exact_score(c);
                    selected.evaluated.push_back(c);
                }
                selected.log_mass = logsumexp(selected.score);
                selected.full = true;
            }
            const double normalizer = selected.log_mass;
            const double upper_normalizer = logaddexp(
                selected.log_mass, selected.log_upper_mass);
            block.log_likelihood += normalizer;
            block.log_likelihood_upper += upper_normalizer;
            block.evaluated_component_documents += selected.evaluated.size();
            block.possible_component_documents += possible_components;
            block.full_component_documents += selected.full ? 1 : 0;
            block.component_bound_violations +=
                selected.bound_violation ? 1 : 0;
            block.omitted_component_mass_sum +=
                selected.omitted_mass_bound;
            block.maximum_omitted_component_mass = std::max(
                block.maximum_omitted_component_mass,
                selected.omitted_mass_bound);
            responsibility = (selected.score.array() - normalizer).exp();
            if (request.store_responsibilities) {
                out.responsibilities.row(d) = responsibility.transpose();
                out.per_document_evaluated_components[d] =
                    static_cast<int32_t>(selected.evaluated.size());
                out.per_document_omitted_component_mass[d] =
                    selected.omitted_mass_bound;
            }
            if (!request.accumulate_moments) continue;
            for (const int32_t c : selected.evaluated) {
                const double weight = responsibility(c);
                if (!(weight > 0.0)) continue;
                block.membership(c) += weight;
                block.first.row(c) += weight * value.transpose();
                if (factor_rank < 0) {
                    block.second[c].noalias() += weight
                        * value * value.transpose();
                } else {
                    const Eigen::VectorXd factor = factor_beta[c]
                        * (value - model.means.row(c).transpose());
                    block.sum_y2.row(c).array() += weight
                        * value.array().square().transpose();
                    block.sum_f.row(c) += weight * factor.transpose();
                    block.sum_ff[c].noalias() += weight
                        * (factor_conditional[c]
                            + factor * factor.transpose());
                    block.sum_yf[c].noalias() += weight
                        * value * factor.transpose();
                }
            }
        }
    });
    reduce_expectation_blocks(out, blocks);
    return out;
}

bool resolve_map_component_screening(const Dataset& data,
    const Model& model, const ComponentScreeningOptions& requested,
    uint64_t seed) {
    if (requested.mode != ComponentScreeningMode::Auto) {
        return requested.mode == ComponentScreeningMode::On;
    }
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    const int32_t active = active_component_count(model);
    if (documents == 0 || active <= requested.minimum_components) {
        return false;
    }
    const int32_t factor_rank = model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(model.factor_covariances.front().factor.cols())
        : -1;
    const std::vector<DenseGaussianSolver> dense_solvers =
        dense_model_solvers(model);
    std::vector<LowRankDiagonalSolver> factor_solvers;
    if (factor_rank >= 0) {
        factor_solvers.reserve(components);
        for (int32_t c = 0; c < components; ++c) {
            const auto& covariance = model.factor_covariances[c];
            factor_solvers.emplace_back(
                covariance.diagonal, covariance.factor);
        }
    }
    const std::vector<double> eigenvalue_upper =
        model_eigenvalue_upper_bounds(model);
    Eigen::VectorXd constant = Eigen::VectorXd::Constant(
        components, -std::numeric_limits<double>::infinity());
    for (int32_t c = 0; c < components; ++c) {
        if (!(model.weights(c) > 0.0)) continue;
        const double logdet = factor_rank < 0
            ? dense_solvers[c].log_determinant
            : factor_solvers[c].log_determinant();
        constant(c) = std::log(model.weights(c))
            - 0.5 * (dimension * kLog2Pi + logdet);
    }
    std::vector<std::vector<int32_t>> groups(components);
    std::vector<double> entropy(documents, 0.0);
    for (int32_t d = 0; d < documents; ++d) {
        Eigen::VectorXd upper = Eigen::VectorXd::Constant(
            components, -std::numeric_limits<double>::infinity());
        const Eigen::VectorXd value =
            data.coordinates.row(d).transpose();
        int32_t top = -1;
        for (int32_t c = 0; c < components; ++c) {
            if (!(model.weights(c) > 0.0)) continue;
            upper(c) = constant(c) - 0.5
                * (value - model.means.row(c).transpose()).squaredNorm()
                    / eigenvalue_upper[c];
            if (top < 0 || upper(c) > upper(top)) top = c;
        }
        const double normalizer = logsumexp(upper);
        for (int32_t c = 0; c < components; ++c) {
            if (!std::isfinite(upper(c))) continue;
            const double probability = std::exp(upper(c) - normalizer);
            if (probability > 0.0) {
                entropy[d] -= probability * std::log(probability);
            }
        }
        groups[top].push_back(d);
    }
    const int32_t requested_budget = requested.audit_documents > 0
        ? requested.audit_documents
        : std::min(256, std::max(16, 2 * active));
    const int32_t budget = std::min(documents, requested_budget);
    std::vector<int32_t> audit;
    std::vector<uint8_t> chosen(documents, 0);
    auto add = [&](int32_t d) {
        if (static_cast<int32_t>(audit.size()) < budget && !chosen[d]) {
            chosen[d] = 1;
            audit.push_back(d);
        }
    };
    std::vector<int32_t> represented;
    for (int32_t c = 0; c < components; ++c) {
        if (!groups[c].empty()) represented.push_back(c);
    }
    std::stable_sort(represented.begin(), represented.end(),
        [&](int32_t left, int32_t right) {
            return hash_string(seed, std::to_string(left))
                < hash_string(seed, std::to_string(right));
        });
    if (static_cast<int32_t>(represented.size()) > budget) {
        represented.resize(budget);
    }
    for (const int32_t c : represented) {
        add(*std::max_element(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return entropy[left] < entropy[right];
            }));
    }
    for (const int32_t c : represented) {
        add(*std::min_element(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return document_effective_total(data, left)
                    < document_effective_total(data, right);
            }));
    }
    std::vector<int32_t> remaining(documents);
    std::iota(remaining.begin(), remaining.end(), int32_t{0});
    std::stable_sort(remaining.begin(), remaining.end(),
        [&](int32_t left, int32_t right) {
            return hash_string(seed, data.identifiers[left])
                < hash_string(seed, data.identifiers[right]);
        });
    for (const int32_t d : remaining) add(d);

    ComponentScreeningOptions enabled = requested;
    enabled.mode = ComponentScreeningMode::On;
    enabled.maximum_components = 0;
    int64_t evaluated = 0;
    int32_t violations = 0;
    Eigen::VectorXd upper(components);
    ComponentScreeningWorkspace screening_workspace;
    for (const int32_t d : audit) {
        const Eigen::VectorXd value =
            data.coordinates.row(d).transpose();
        upper.setConstant(-std::numeric_limits<double>::infinity());
        for (int32_t c = 0; c < components; ++c) {
            if (model.weights(c) > 0.0) {
                upper(c) = constant(c) - 0.5
                    * (value - model.means.row(c).transpose()).squaredNorm()
                        / eigenvalue_upper[c];
            }
        }
        auto exact = [&](int32_t c) {
            if (factor_rank < 0) {
                return std::log(model.weights(c))
                    + dense_solvers[c].log_density(value);
            }
            const Eigen::VectorXd residual =
                value - model.means.row(c).transpose();
            return constant(c) - 0.5
                * factor_solvers[c].quadratic(residual);
        };
        const ScreenedComponents selected = screen_component_scores(
            upper, enabled, true, exact, &screening_workspace);
        evaluated += selected.evaluated.size();
        violations += selected.bound_violation ? 1 : 0;
    }
    const double mean_evaluated = static_cast<double>(evaluated)
        / std::max<size_t>(1, audit.size());
    const double exact_cost = factor_rank < 0
        ? static_cast<double>(dimension) * dimension
        : static_cast<double>(dimension) * factor_rank
            + factor_rank * factor_rank + dimension;
    const double ratio = (active * static_cast<double>(dimension)
            + mean_evaluated * exact_cost)
        / (active * exact_cost);
    return violations == 0
        && ratio <= 1.0 - requested.minimum_work_reduction;
}

template<class ParticleCollection>
Expectation particle_expectation_impl(const ParticleCollection& particles,
    const Model& model, const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {},
    int32_t forced_blocks = 0,
    ExpectationBlock* external_block = nullptr) {
    const bool accumulate_moments = request.accumulate_moments;
    validate_component_screening(screening);
    const bool screen =
        screening.mode != ComponentScreeningMode::Off;
    const int32_t documents = particles.documents;
    const int32_t maximum_samples = [&]() {
        if constexpr (std::is_same_v<ParticleCollection, ParticleSet>) {
            return particles.samples;
        } else {
            return particles.maximum_samples;
        }
    }();
    const int32_t dimension = particles.dimension;
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t factor_rank = model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(model.factor_covariances.front().factor.cols())
        : -1;
    Expectation out;
    if (accumulate_moments) {
        out = empty_expectation(
            documents, components, dimension, factor_rank);
    } else {
        out.documents = documents;
    }
    if (request.store_responsibilities) {
        out.responsibilities.resize(documents, components);
    }
    if (request.store_responsibilities || request.collect_diagnostics) {
        out.per_document_evaluated_components.resize(documents);
        out.per_document_omitted_component_mass.resize(documents);
    }
    if (request.collect_diagnostics) {
        out.particle_diagnostics.resize(documents);
    }
    std::vector<LowRankDiagonalSolver> factor_solvers;
    std::vector<Eigen::MatrixXd> factor_beta, factor_conditional;
    if (factor_rank >= 0) {
        factor_solvers.reserve(components);
        for (int32_t c = 0; c < components; ++c) {
            const auto& covariance = model.factor_covariances[c];
            factor_solvers.emplace_back(
                covariance.diagonal, covariance.factor);
            if (accumulate_moments) {
                factor_beta.push_back(factor_solvers.back().solve_matrix(
                    covariance.factor).transpose());
                factor_conditional.push_back(
                    Eigen::MatrixXd::Identity(factor_rank, factor_rank)
                    - factor_beta.back() * covariance.factor);
            }
        }
    }
    const std::vector<DenseGaussianSolver> dense_solvers =
        dense_model_solvers(model);
    const std::vector<double> eigenvalue_upper = screen
        ? model_eigenvalue_upper_bounds(model) : std::vector<double>{};
    Eigen::VectorXd gaussian_constant = Eigen::VectorXd::Constant(
        components, -std::numeric_limits<double>::infinity());
    for (int32_t c = 0; c < components; ++c) {
        if (!(model.weights(c) > 0.0)) continue;
        const double logdet = factor_rank < 0
            ? dense_solvers[c].log_determinant
            : factor_solvers[c].log_determinant();
        gaussian_constant(c) = std::log(model.weights(c))
            - 0.5 * (dimension * kLog2Pi + logdet);
    }
    const int32_t possible_components = active_component_count(model);
    const int32_t requested_blocks = external_block
        ? 1 : forced_blocks > 0
        ? std::min(documents, forced_blocks)
        : expectation_shards(
            documents, components, dimension, factor_rank);
    const int32_t block_size =
        (documents + requested_blocks - 1) / requested_blocks;
    const int32_t n_blocks = (documents + block_size - 1) / block_size;
    uint64_t block_workspace_values =
        static_cast<uint64_t>(components) * maximum_samples
        + 2 * static_cast<uint64_t>(components);
    if (screen) {
        block_workspace_values +=
            static_cast<uint64_t>(maximum_samples) * dimension
            + maximum_samples + components;
    }
    out.peak_workspace_bytes = static_cast<uint64_t>(n_blocks)
        * (sizeof(double) * block_workspace_values
            + (accumulate_moments && !external_block
                ? expectation_block_bytes(
                    components, dimension, factor_rank)
                : 0));
    std::vector<ExpectationBlock> blocks;
    if (!external_block) {
        blocks.reserve(n_blocks);
        for (int32_t block = 0; block < n_blocks; ++block) {
            blocks.emplace_back(
                components, dimension, factor_rank, accumulate_moments);
        }
    }
    std::atomic<int64_t> gaussian_nanoseconds{0};
    std::atomic<int64_t> moment_nanoseconds{0};
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        ExpectationBlock& block = external_block
            ? *external_block : blocks[block_index];
        Eigen::MatrixXd log_tilt(components, maximum_samples);
        Eigen::VectorXd evidence(components);
        Eigen::VectorXd responsibility(components);
        Eigen::VectorXd base;
        Eigen::MatrixXd dense_standardized;
        Eigen::VectorXd dense_log_density;
        RowMajorMatrixXd factor_residual;
        RowMajorMatrixXd bound_residual;
        Eigen::VectorXd bound_term;
        Eigen::VectorXd upper;
        if (screen) {
            bound_residual.resize(maximum_samples, dimension);
            bound_term.resize(maximum_samples);
            upper.resize(components);
        }
        ComponentScreeningWorkspace screening_workspace;
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        int64_t local_gaussian_nanoseconds = 0;
        int64_t local_moment_nanoseconds = 0;
        double local_bound_seconds = 0.0;
        for (int32_t d = begin; d < end; ++d) {
            const int32_t samples = particles.samples_for_document(d);
            const auto gaussian_start = std::chrono::steady_clock::now();
            const auto values = particles.values_for_document(d);
            base = particles.log_likelihood_for_document(d)
                - particles.log_proposal_for_document(d)
                - Eigen::VectorXd::Constant(samples, std::log(samples));
            evidence.setConstant(
                -std::numeric_limits<double>::infinity());
            auto exact_score = [&](int32_t c) {
                if (factor_rank < 0) {
                    dense_solvers[c].log_density_rows(values,
                        dense_standardized, dense_log_density);
                    log_tilt.row(c).head(samples) =
                        (base + dense_log_density).transpose();
                } else {
                    factor_residual.resize(samples, dimension);
                    factor_residual = values.rowwise() - model.means.row(c);
                    log_tilt.row(c).head(samples) = (base.array()
                        - 0.5 * (dimension * kLog2Pi
                            + factor_solvers[c].log_determinant()
                            + factor_solvers[c].quadratic_rows(
                                factor_residual).array())).matrix().transpose();
                }
                evidence(c) = logsumexp(
                    log_tilt.row(c).head(samples).transpose());
                return std::log(model.weights(c)) + evidence(c);
            };
            ScreenedComponents selected;
            if (screen) {
                const auto bound_start = std::chrono::steady_clock::now();
                upper.setConstant(-std::numeric_limits<double>::infinity());
                // Round the squared-distance contribution downward so the
                // vectorized floating-point calculation remains conservative.
                const double squared_distance_roundoff = std::max(0.5,
                    1.0 - 16.0 * (dimension + 1)
                        * std::numeric_limits<double>::epsilon());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    auto residual = bound_residual.topRows(samples);
                    residual = values.rowwise() - model.means.row(c);
                    bound_term.head(samples).array() = base.array()
                        + gaussian_constant(c)
                        - 0.5 * squared_distance_roundoff
                            * residual.rowwise().squaredNorm().array()
                            / eigenvalue_upper[c];
                    upper(c) = logsumexp(bound_term.head(samples));
                }
                local_bound_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now()
                    - bound_start).count();
                selected = screen_component_scores(
                    upper, screening, true, exact_score,
                    &screening_workspace);
            } else {
                selected.score = Eigen::VectorXd::Constant(
                    components, -std::numeric_limits<double>::infinity());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    selected.score(c) = exact_score(c);
                    selected.evaluated.push_back(c);
                }
                selected.log_mass = logsumexp(selected.score);
                selected.full = true;
            }
            local_gaussian_nanoseconds +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - gaussian_start).count();
            const double normalizer = selected.log_mass;
            block.log_likelihood += normalizer;
            block.log_likelihood_upper += logaddexp(
                selected.log_mass, selected.log_upper_mass);
            block.evaluated_component_documents += selected.evaluated.size();
            block.possible_component_documents += possible_components;
            block.full_component_documents += selected.full ? 1 : 0;
            block.component_bound_violations +=
                selected.bound_violation ? 1 : 0;
            block.omitted_component_mass_sum +=
                selected.omitted_mass_bound;
            block.maximum_omitted_component_mass = std::max(
                block.maximum_omitted_component_mass,
                selected.omitted_mass_bound);
            if (accumulate_moments || request.store_responsibilities
                || request.collect_diagnostics) {
                responsibility =
                    (selected.score.array() - normalizer).exp();
            }
            if (request.store_responsibilities) {
                out.responsibilities.row(d) = responsibility.transpose();
            }
            if (request.store_responsibilities
                || request.collect_diagnostics) {
                out.per_document_evaluated_components[d] =
                    static_cast<int32_t>(selected.evaluated.size());
                out.per_document_omitted_component_mass[d] =
                    selected.omitted_mass_bound;
            }
            if (request.collect_diagnostics) {
                Eigen::VectorXd log_weight(samples);
                Eigen::VectorXd component_weight = Eigen::VectorXd::Constant(
                    components, -std::numeric_limits<double>::infinity());
                for (int32_t s = 0; s < samples; ++s) {
                    for (const int32_t c : selected.evaluated) {
                        component_weight(c) =
                            std::log(model.weights(c)) + log_tilt(c, s);
                    }
                    log_weight(s) = logsumexp(component_weight);
                }
                const double weight_normalizer = logsumexp(log_weight);
                const Eigen::VectorXd probability =
                    (log_weight.array() - weight_normalizer).exp();
                auto& diagnostic = out.particle_diagnostics[d];
                diagnostic.relative_ess = 1.0
                    / (samples * probability.squaredNorm());
                diagnostic.maximum_weight = probability.maxCoeff();
                const auto document_likelihood =
                    particles.log_likelihood_for_document(d);
                const auto document_proposal =
                    particles.log_proposal_for_document(d);
                diagnostic.log_likelihood_range =
                    document_likelihood.maxCoeff()
                    - document_likelihood.minCoeff();
                diagnostic.log_proposal_range = document_proposal.maxCoeff()
                    - document_proposal.minCoeff();
                const Eigen::VectorXd log_target = log_weight
                    + document_proposal
                    + Eigen::VectorXd::Constant(samples, std::log(samples));
                diagnostic.hpd80_log_density_threshold =
                    weighted_hpd_threshold(log_target, probability, 0.8);
                diagnostic.hpd95_log_density_threshold =
                    weighted_hpd_threshold(log_target, probability, 0.95);
            }
            if (accumulate_moments) {
                const auto moment_start = std::chrono::steady_clock::now();
                for (const int32_t c : selected.evaluated) {
                    const double component_responsibility = responsibility(c);
                    if (!(component_responsibility > 0.0)) continue;
                    block.membership(c) += component_responsibility;
                    const Eigen::VectorXd tau =
                        (log_tilt.row(c).head(samples).transpose().array()
                            - evidence(c)).exp();
                    if (factor_rank < 0) {
                        block.first.row(c).noalias() +=
                            component_responsibility
                            * (values.transpose() * tau).transpose();
                        RowMajorMatrixXd weighted = values;
                        weighted.array().colwise() *= tau.array().sqrt();
                        block.second[c].noalias() += component_responsibility
                            * weighted.transpose() * weighted;
                    } else {
                        const Eigen::VectorXd weight =
                            component_responsibility * tau;
                        const RowMajorMatrixXd residual =
                            values.rowwise() - model.means.row(c);
                        const RowMajorMatrixXd factors =
                            residual * factor_beta[c].transpose();
                        block.first.row(c).noalias() +=
                            weight.transpose() * values;
                        block.sum_y2.row(c).array() +=
                            (weight.transpose()
                                * values.array().square().matrix()).array();
                        block.sum_f.row(c).noalias() +=
                            weight.transpose() * factors;
                        RowMajorMatrixXd weighted_factors = factors;
                        weighted_factors.array().colwise() *=
                            weight.array().sqrt();
                        block.sum_ff[c].noalias() += weight.sum()
                            * factor_conditional[c]
                            + weighted_factors.transpose() * weighted_factors;
                        RowMajorMatrixXd weighted_values = values;
                        weighted_values.array().colwise() *= weight.array();
                        block.sum_yf[c].noalias() +=
                            weighted_values.transpose() * factors;
                    }
                }
                local_moment_nanoseconds +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()
                        - moment_start).count();
            }
        }
        gaussian_nanoseconds.fetch_add(
            local_gaussian_nanoseconds, std::memory_order_relaxed);
        moment_nanoseconds.fetch_add(
            local_moment_nanoseconds, std::memory_order_relaxed);
        block.component_bound_seconds += local_bound_seconds;
    });
    if (!external_block) {
        reduce_expectation_blocks(out, blocks);
    }
    out.gaussian_seconds = 1e-9 * gaussian_nanoseconds.load();
    out.moment_seconds = 1e-9 * moment_nanoseconds.load();
    return out;
}

Expectation particle_expectation(const ParticleSet& particles,
    const Model& model, const ExpectationRequest& request,
    const ComponentScreeningOptions& screening) {
    return particle_expectation_impl(
        particles, model, request, screening);
}

Expectation particle_expectation(const RaggedParticleSet& particles,
    const Model& model, const ExpectationRequest& request,
    const ComponentScreeningOptions& screening) {
    return particle_expectation_impl(
        particles, model, request, screening);
}

template<class ParticleCollection>
bool resolve_particle_component_screening(
    const ParticleCollection& particles, const Model& model,
    const ComponentScreeningOptions& requested,
    const std::vector<int32_t>& audit_documents) {
    if (requested.mode != ComponentScreeningMode::Auto) {
        return requested.mode == ComponentScreeningMode::On;
    }
    ComponentScreeningOptions enabled = requested;
    enabled.mode = ComponentScreeningMode::On;
    enabled.maximum_components = 0;
    RaggedParticleSet audit;
    audit.documents = static_cast<int32_t>(audit_documents.size());
    audit.dimension = particles.dimension;
    audit.offsets.assign(audit.documents + 1, 0);
    audit.proposal_candidates.resize(audit.documents);
    for (int32_t local = 0; local < audit.documents; ++local) {
        const int32_t document = audit_documents[local];
        if (document < 0 || document >= particles.documents) {
            throw std::runtime_error(
                "UAC particle screening audit document is out of range");
        }
        const int32_t samples = particles.samples_for_document(document);
        audit.offsets[local + 1] = audit.offsets[local] + samples;
        audit.maximum_samples = std::max(audit.maximum_samples, samples);
        const auto values = particles.values_for_document(document);
        audit.values.insert(audit.values.end(), values.data(),
            values.data() + static_cast<int64_t>(samples)
                * particles.dimension);
        const auto likelihood =
            particles.log_likelihood_for_document(document);
        audit.log_likelihood.insert(audit.log_likelihood.end(),
            likelihood.data(), likelihood.data() + samples);
        const auto proposal =
            particles.log_proposal_for_document(document);
        audit.log_proposal.insert(audit.log_proposal.end(),
            proposal.data(), proposal.data() + samples);
        const auto origins =
            particles.proposal_origins_for_document(document);
        audit.proposal_origins.insert(audit.proposal_origins.end(),
            origins.data(), origins.data() + samples);
        audit.proposal_candidates[local] =
            particles.proposal_candidates[document];
    }
    const Expectation probe = particle_expectation_impl(
        audit, model, ExpectationRequest{false, false, false}, enabled);
    if (probe.component_bound_violations > 0
        || probe.possible_component_documents == 0) {
        return false;
    }
    const double exact_fraction =
        static_cast<double>(probe.evaluated_component_documents)
        / probe.possible_component_documents;
    const double dimension = std::max(1, audit.dimension);
    const double bound_fraction = 1.0 / dimension;
    return exact_fraction <= 0.5
        && bound_fraction + exact_fraction
            <= 1.0 - requested.minimum_work_reduction;
}

Expectation particle_expectation_into(const ParticleSet& particles,
    const Model& model, const ExpectationRequest& request,
    const ComponentScreeningOptions& screening, ExpectationBlock& block) {
    return particle_expectation_impl(
        particles, model, request, screening, 1, &block);
}

Expectation particle_expectation_into(const RaggedParticleSet& particles,
    const Model& model, const ExpectationRequest& request,
    const ComponentScreeningOptions& screening, ExpectationBlock& block) {
    return particle_expectation_impl(
        particles, model, request, screening, 1, &block);
}

bool resolve_particle_component_screening(
    const ParticleSet& particles, const Model& model,
    const ComponentScreeningOptions& requested,
    const std::vector<int32_t>& audit_documents) {
    return resolve_particle_component_screening<ParticleSet>(
        particles, model, requested, audit_documents);
}

bool resolve_particle_component_screening(
    const RaggedParticleSet& particles, const Model& model,
    const ComponentScreeningOptions& requested,
    const std::vector<int32_t>& audit_documents) {
    return resolve_particle_component_screening<RaggedParticleSet>(
        particles, model, requested, audit_documents);
}

} // namespace uac::detail
