#include "clustering/uac.hpp"

#include "clustering_core/cosine_clustering.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace uac {
namespace {

constexpr double kLog2Pi = 1.83787706640934548356;
constexpr double kCovarianceShrinkageStrength = 20.0;

double logsumexp(const Eigen::Ref<const Eigen::VectorXd>& values) {
    const double maximum = values.maxCoeff();
    if (!std::isfinite(maximum)) return maximum;
    return maximum + std::log((values.array() - maximum).exp().sum());
}

double weighted_hpd_threshold(
    const Eigen::Ref<const Eigen::VectorXd>& log_density,
    const Eigen::Ref<const Eigen::VectorXd>& probability, double level) {
    std::vector<int32_t> order(log_density.size());
    std::iota(order.begin(), order.end(), int32_t{0});
    std::stable_sort(order.begin(), order.end(), [&](int32_t left,
            int32_t right) { return log_density(left) > log_density(right); });
    double cumulative = 0.0;
    for (const int32_t index : order) {
        cumulative += probability(index);
        if (cumulative >= level) return log_density(index);
    }
    return log_density(order.back());
}

uint64_t fnv_append(uint64_t value, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        value ^= bytes[i];
        value *= 1099511628211ull;
    }
    return value;
}

uint64_t hash_string(uint64_t value, const std::string& text) {
    value = fnv_append(value, text.data(), text.size());
    const unsigned char separator = 0xff;
    return fnv_append(value, &separator, 1);
}

Eigen::MatrixXd floor_covariance(const Eigen::Ref<const Eigen::MatrixXd>& input,
    double floor) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        0.5 * (input + input.transpose()));
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("UAC covariance eigendecomposition failed");
    }
    return solver.eigenvectors()
        * solver.eigenvalues().cwiseMax(floor).asDiagonal()
        * solver.eigenvectors().transpose();
}

std::pair<double, double> gershgorin_bounds(
    const Eigen::Ref<const Eigen::MatrixXd>& input) {
    double lower = std::numeric_limits<double>::infinity();
    double upper = -std::numeric_limits<double>::infinity();
    for (Eigen::Index i = 0; i < input.rows(); ++i) {
        const double radius = input.row(i).cwiseAbs().sum()
            - std::abs(input(i, i));
        lower = std::min(lower, input(i, i) - radius);
        upper = std::max(upper, input(i, i) + radius);
    }
    return {lower, upper};
}

double log_gaussian(const Eigen::Ref<const Eigen::VectorXd>& value,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance) {
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("UAC covariance is not positive definite");
    }
    const Eigen::VectorXd residual = value - mean;
    const Eigen::MatrixXd lower = llt.matrixL();
    const double logdet = 2.0 * lower.diagonal().array().log().sum();
    return -0.5 * (value.size() * kLog2Pi + logdet
        + residual.dot(llt.solve(residual)));
}

struct DenseGaussianSolver {
    Eigen::VectorXd mean;
    Eigen::MatrixXd lower;
    double log_determinant = 0.0;

    DenseGaussianSolver() = default;

    DenseGaussianSolver(const Eigen::Ref<const Eigen::VectorXd>& input_mean,
        const Eigen::Ref<const Eigen::MatrixXd>& covariance)
        : mean(input_mean) {
        Eigen::LLT<Eigen::MatrixXd> llt(covariance);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("UAC covariance is not positive definite");
        }
        lower = llt.matrixL();
        log_determinant = 2.0 * lower.diagonal().array().log().sum();
    }

    double log_density(
        const Eigen::Ref<const Eigen::VectorXd>& value) const {
        Eigen::VectorXd standardized = value - mean;
        lower.triangularView<Eigen::Lower>().solveInPlace(standardized);
        return -0.5 * (value.size() * kLog2Pi + log_determinant
            + standardized.squaredNorm());
    }

    Eigen::VectorXd log_density_rows(
        const Eigen::Ref<const RowMajorMatrixXd>& values) const {
        Eigen::MatrixXd standardized =
            (values.rowwise() - mean.transpose()).transpose();
        lower.triangularView<Eigen::Lower>().solveInPlace(standardized);
        return (-0.5 * (mean.size() * kLog2Pi + log_determinant
            + standardized.colwise().squaredNorm().array())).matrix();
    }
};

std::vector<DenseGaussianSolver> dense_model_solvers(const Model& model) {
    std::vector<DenseGaussianSolver> out(model.weights.size());
    if (model.covariance_kind != CovarianceKind::Dense) return out;
    for (Eigen::Index c = 0; c < model.weights.size(); ++c) {
        if (model.weights(c) > 0.0) {
            out[c] = DenseGaussianSolver(
                model.means.row(c).transpose(), model.covariances[c]);
        }
    }
    return out;
}

Eigen::MatrixXd model_covariance_dense(const Model& model,
    int32_t component) {
    return model.covariance_kind == CovarianceKind::Dense
        ? model.covariances[component]
        : model.factor_covariances[component].dense();
}

LowRankDiagonalCovariance factorize_covariance(
    const Eigen::Ref<const Eigen::MatrixXd>& covariance, int32_t rank,
    double floor) {
    const int32_t dimension = static_cast<int32_t>(covariance.rows());
    if (covariance.cols() != dimension || rank < 0 || rank > dimension
        || !(floor > 0.0)) {
        throw std::invalid_argument("Invalid factor covariance conversion");
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
        0.5 * (covariance + covariance.transpose()));
    if (eigen.info() != Eigen::Success) {
        throw std::runtime_error("UAC factor covariance eigendecomposition failed");
    }
    LowRankDiagonalCovariance out;
    out.factor = RowMajorMatrixXd::Zero(dimension, rank);
    for (int32_t j = 0; j < rank; ++j) {
        const int32_t index = dimension - 1 - j;
        const double value = std::max(0.0, eigen.eigenvalues()(index) - floor);
        out.factor.col(j) = std::sqrt(value) * eigen.eigenvectors().col(index);
    }
    out.diagonal = (covariance.diagonal()
        - (out.factor.array().square().rowwise().sum()).matrix())
        .cwiseMax(floor);
    return out;
}

double covariance_prior(const Model& model, double strength) {
    double out = 0.0;
    if (model.covariance_kind == CovarianceKind::FactorAnalytic) {
        for (const auto& covariance : model.factor_covariances) {
            LowRankDiagonalSolver solver(
                covariance.diagonal, covariance.factor);
            const Eigen::MatrixXd core_inverse = solver.solve_core(
                Eigen::MatrixXd::Identity(covariance.factor.cols(),
                    covariance.factor.cols()));
            Eigen::VectorXd inverse_diagonal = solver.inverse_diagonal();
            if (covariance.factor.cols() > 0) {
                const Eigen::MatrixXd& scaled =
                    solver.inverse_diagonal_factor();
                inverse_diagonal.array() -= (scaled * core_inverse).cwiseProduct(
                    scaled).rowwise().sum().array();
            }
            double trace = inverse_diagonal.dot(
                model.factor_shrinkage_target.diagonal);
            if (model.factor_shrinkage_target.factor.cols() > 0) {
                trace += (model.factor_shrinkage_target.factor.transpose()
                    * solver.solve_matrix(
                        model.factor_shrinkage_target.factor)).trace();
            }
            out -= 0.5 * strength * (solver.log_determinant() + trace);
        }
        return out;
    }
    for (const auto& covariance : model.covariances) {
        Eigen::LLT<Eigen::MatrixXd> llt(covariance);
        if (llt.info() != Eigen::Success) {
            return -std::numeric_limits<double>::infinity();
        }
        const Eigen::MatrixXd lower = llt.matrixL();
        const double logdet = 2.0 * lower.diagonal().array().log().sum();
        out -= 0.5 * strength * (logdet
            + (llt.solve(model.shrinkage_target)).trace());
    }
    return out;
}

int32_t active_component_count(const Model& model) {
    return static_cast<int32_t>((model.weights.array() > 0.0).count());
}

double membership_epsilon(int32_t documents) {
    return std::max(1e-12, 64.0
        * std::numeric_limits<double>::epsilon() * documents);
}

int32_t map_start_seed(int32_t seed, int32_t start) {
    constexpr uint64_t modulus = 2147483647ull;
    const uint64_t value = static_cast<uint32_t>(seed)
        + 104729ull * static_cast<uint32_t>(start);
    return static_cast<int32_t>(value % modulus);
}

Eigen::VectorXd composition_from_coordinate(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    Eigen::VectorXd logits = helmert.transpose() * coordinate;
    logits.array() -= logits.maxCoeff();
    Eigen::VectorXd values = logits.array().exp();
    values /= values.sum();
    return values;
}

Eigen::VectorXd count_log_likelihood_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    if (coordinates.cols() != helmert.rows()
        || basis.probabilities.cols() != helmert.cols()
        || document.ids.size() != document.cnts.size()) {
        throw std::invalid_argument("Invalid UAC batched likelihood input");
    }
    const Eigen::Index samples = coordinates.rows();
    Eigen::MatrixXd logits = coordinates * helmert;
    for (Eigen::Index s = 0; s < samples; ++s) {
        logits.row(s).array() -= logits.row(s).maxCoeff();
        logits.row(s) = logits.row(s).array().exp();
        logits.row(s) /= logits.row(s).sum();
    }
    if (document.ids.empty()) return Eigen::VectorXd::Zero(samples);
    RowMajorMatrixXd observed(document.ids.size(), basis.probabilities.cols());
    for (size_t j = 0; j < document.ids.size(); ++j) {
        const uint32_t feature = document.ids[j];
        if (feature >= static_cast<uint32_t>(basis.probabilities.rows())) {
            throw std::runtime_error("UAC document feature index is out of range");
        }
        observed.row(j) = basis.probabilities.row(feature);
    }
    const Eigen::MatrixXd probability = observed * logits.transpose();
    Eigen::VectorXd out = Eigen::VectorXd::Zero(samples);
    for (size_t j = 0; j < document.ids.size(); ++j) {
        for (Eigen::Index s = 0; s < samples; ++s) {
            const double value = probability(j, s);
            if (!(value > 0.0) || !std::isfinite(value)) {
                out(s) = -std::numeric_limits<double>::infinity();
            } else if (std::isfinite(out(s))) {
                out(s) += document.cnts[j] * std::log(value);
            }
        }
    }
    return out;
}

FisherApproximation fisher_approximation_impl(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal) {
    if (coordinate.size() != helmert.rows()
        || basis.probabilities.cols() != helmert.cols()
        || (proposal != ProposalKind::ExactFisher
            && proposal != ProposalKind::SparseEmpiricalFisher)) {
        throw std::invalid_argument("Invalid UAC Fisher input");
    }
    const Eigen::VectorXd composition = composition_from_coordinate(
        coordinate, helmert);
    Eigen::MatrixXd simplex_covariance = composition.asDiagonal();
    simplex_covariance.noalias() -= composition * composition.transpose();
    const Eigen::MatrixXd simplex_derivative =
        simplex_covariance * helmert.transpose();
    FisherApproximation out;
    out.gradient = Eigen::VectorXd::Zero(coordinate.size());
    out.information = Eigen::MatrixXd::Zero(
        coordinate.size(), coordinate.size());
    if (proposal == ProposalKind::SparseEmpiricalFisher) {
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const uint32_t feature = document.ids[j];
            if (feature >= static_cast<uint32_t>(basis.probabilities.rows())) {
                throw std::runtime_error(
                    "UAC document feature index is out of range");
            }
            const double probability =
                basis.probabilities.row(feature).dot(composition);
            if (!(probability > 0.0) || !std::isfinite(probability)) {
                throw std::runtime_error(
                    "UAC observed feature has zero Fisher probability");
            }
            const Eigen::VectorXd derivative = (
                basis.probabilities.row(feature) * simplex_derivative)
                .transpose();
            const Eigen::VectorXd score = derivative / probability;
            out.gradient.noalias() += document.cnts[j] * score;
            out.information.noalias() += document.cnts[j]
                * score * score.transpose();
        }
    } else {
        const Eigen::VectorXd probability = (basis.probabilities * composition)
            .array().max(1e-300).matrix();
        const RowMajorMatrixXd probability_derivative =
            basis.probabilities * simplex_derivative;
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const uint32_t feature = document.ids[j];
            if (feature >= static_cast<uint32_t>(probability.size())) {
                throw std::runtime_error(
                    "UAC document feature index is out of range");
            }
            out.gradient.noalias() += document.cnts[j]
                / probability(feature)
                * probability_derivative.row(feature).transpose();
        }
        const double total = std::accumulate(document.cnts.begin(),
            document.cnts.end(), 0.0);
        const Eigen::VectorXd curvature =
            Eigen::VectorXd::Constant(probability.size(), total).array()
                / probability.array();
        out.information.noalias() = probability_derivative.transpose()
            * curvature.asDiagonal() * probability_derivative;
    }
    out.information = 0.5 * (
        out.information + out.information.transpose());
    return out;
}

struct PilotCache {
    std::vector<Eigen::MatrixXd> inverse_covariances;
    Eigen::VectorXd log_determinants;

    explicit PilotCache(const Pilot& pilot)
        : log_determinants(pilot.weights.size()) {
        inverse_covariances.reserve(pilot.weights.size());
        for (Eigen::Index c = 0; c < pilot.weights.size(); ++c) {
            Eigen::LLT<Eigen::MatrixXd> llt(pilot.covariances[c]);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error(
                    "UAC Fisher pilot covariance is not positive definite");
            }
            inverse_covariances.push_back(llt.solve(
                Eigen::MatrixXd::Identity(
                    pilot.covariances[c].rows(),
                    pilot.covariances[c].cols())));
            const Eigen::MatrixXd lower = llt.matrixL();
            log_determinants(c) =
                2.0 * lower.diagonal().array().log().sum();
        }
    }
};

struct DocumentProposal {
    Eigen::VectorXd weights;
    std::vector<Eigen::VectorXd> means;
    std::vector<Eigen::MatrixXd> lower;
    Eigen::VectorXd log_determinants;
};

DocumentProposal fisher_proposal(
    const Eigen::Ref<const Eigen::VectorXd>& center,
    const FisherApproximation& fisher, const Pilot& pilot,
    const PilotCache& cache, double broadening) {
    if (!(broadening > 0.0) || !std::isfinite(broadening)) {
        throw std::invalid_argument("Invalid UAC Fisher broadening");
    }
    const int32_t dimension = static_cast<int32_t>(center.size());
    DocumentProposal out;
    out.weights.resize(pilot.weights.size());
    out.means.reserve(pilot.weights.size());
    out.lower.reserve(pilot.weights.size());
    out.log_determinants.resize(pilot.weights.size());
    Eigen::VectorXd log_weight(pilot.weights.size());
    for (Eigen::Index c = 0; c < pilot.weights.size(); ++c) {
        if (!(pilot.weights(c) > 0.0)) {
            out.means.push_back(center);
            out.lower.push_back(Eigen::MatrixXd::Identity(
                dimension, dimension));
            out.log_determinants(c) = 0.0;
            log_weight(c) = -std::numeric_limits<double>::infinity();
            continue;
        }
        const Eigen::MatrixXd& inverse_covariance =
            cache.inverse_covariances[c];
        const double pilot_logdet = cache.log_determinants(c);
        const Eigen::VectorXd pilot_mean = pilot.means.row(c).transpose();
        const Eigen::VectorXd residual = center - pilot_mean;
        const Eigen::VectorXd b = fisher.gradient
            - inverse_covariance * residual;
        const Eigen::MatrixXd raw_precision = 0.5
            * (fisher.information + inverse_covariance
                + fisher.information.transpose()
                + inverse_covariance.transpose());
        const auto precision_bounds = gershgorin_bounds(raw_precision);
        const bool precision_floor_inactive =
            precision_bounds.first >= 1e-8;
        const Eigen::MatrixXd precision = precision_floor_inactive
            ? raw_precision : floor_covariance(raw_precision, 1e-8);
        Eigen::LLT<Eigen::MatrixXd> precision_llt(precision);
        if (precision_llt.info() != Eigen::Success) {
            throw std::runtime_error(
                "UAC Fisher precision is not positive definite");
        }
        const Eigen::VectorXd step = precision_llt.solve(b);
        const Eigen::MatrixXd raw_local_covariance = precision_llt.solve(
            Eigen::MatrixXd::Identity(dimension, dimension));
        const bool covariance_floor_inactive = precision_floor_inactive
            && precision_bounds.second <= 1e8;
        const Eigen::MatrixXd local_covariance = covariance_floor_inactive
            ? 0.5 * (raw_local_covariance + raw_local_covariance.transpose())
            : floor_covariance(raw_local_covariance, 1e-8);
        Eigen::LLT<Eigen::MatrixXd> local_llt(
            broadening * local_covariance);
        if (local_llt.info() != Eigen::Success) {
            throw std::runtime_error(
                "UAC proposal covariance is not positive definite");
        }
        const Eigen::MatrixXd precision_lower = precision_llt.matrixL();
        const double precision_logdet = 2.0
            * precision_lower.diagonal().array().log().sum();
        out.means.push_back(center + step);
        out.lower.push_back(local_llt.matrixL());
        out.log_determinants(c) = 2.0
            * out.lower.back().diagonal().array().log().sum();
        log_weight(c) = std::log(pilot.weights(c))
            - 0.5 * (dimension * kLog2Pi + pilot_logdet
                + residual.dot(inverse_covariance * residual))
            + 0.5 * b.dot(step) + 0.5 * dimension * kLog2Pi
            - 0.5 * precision_logdet;
    }
    out.weights = (log_weight.array() - logsumexp(log_weight)).exp();
    return out;
}

Eigen::VectorXd proposal_log_density_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    const DocumentProposal& proposal) {
    Eigen::MatrixXd terms(proposal.weights.size(), values.rows());
    for (Eigen::Index j = 0; j < proposal.weights.size(); ++j) {
        if (!(proposal.weights(j) > 0.0)) {
            terms.row(j).setConstant(
                -std::numeric_limits<double>::infinity());
            continue;
        }
        Eigen::MatrixXd standardized =
            (values.rowwise() - proposal.means[j].transpose()).transpose();
        proposal.lower[j].triangularView<Eigen::Lower>()
            .solveInPlace(standardized);
        terms.row(j) = (std::log(proposal.weights(j)) - 0.5
            * (values.cols() * kLog2Pi + proposal.log_determinants(j)
                + standardized.colwise().squaredNorm().array())).matrix();
    }
    Eigen::VectorXd out(values.rows());
    for (Eigen::Index s = 0; s < values.rows(); ++s) {
        out(s) = logsumexp(terms.col(s));
    }
    return out;
}

struct Expectation {
    int32_t documents = 0;
    RowMajorMatrixXd responsibilities;
    Eigen::VectorXd membership;
    RowMajorMatrixXd first;
    std::vector<Eigen::MatrixXd> second;
    RowMajorMatrixXd sum_y2;
    RowMajorMatrixXd sum_f;
    std::vector<Eigen::MatrixXd> sum_ff;
    std::vector<Eigen::MatrixXd> sum_yf;
    std::vector<ParticleDiagnostic> particle_diagnostics;
    double log_likelihood = 0.0;
    double gaussian_seconds = 0.0;
    double moment_seconds = 0.0;
    uint64_t accumulator_bytes = 0;
};

struct ExpectationBlock {
    Eigen::VectorXd membership;
    RowMajorMatrixXd first;
    std::vector<Eigen::MatrixXd> second;
    RowMajorMatrixXd sum_y2;
    RowMajorMatrixXd sum_f;
    std::vector<Eigen::MatrixXd> sum_ff;
    std::vector<Eigen::MatrixXd> sum_yf;
    double log_likelihood = 0.0;

    ExpectationBlock(int32_t components, int32_t dimension,
        int32_t factor_rank = -1)
        : membership(Eigen::VectorXd::Zero(components)),
          first(RowMajorMatrixXd::Zero(components, dimension)),
          second(components, factor_rank < 0
              ? Eigen::MatrixXd::Zero(dimension, dimension)
              : Eigen::MatrixXd()) {
        if (factor_rank >= 0) {
            sum_y2 = RowMajorMatrixXd::Zero(components, dimension);
            sum_f = RowMajorMatrixXd::Zero(components, factor_rank);
            sum_ff.assign(components,
                Eigen::MatrixXd::Zero(factor_rank, factor_rank));
            sum_yf.assign(components,
                Eigen::MatrixXd::Zero(dimension, factor_rank));
        }
    }
};

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
        out.membership += block.membership;
        out.first += block.first;
        out.log_likelihood += block.log_likelihood;
        if (out.sum_y2.size() > 0) {
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
    int32_t dimension, int32_t factor_rank = -1) {
    Expectation out;
    out.documents = documents;
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
    bool store_responsibilities = false) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    const int32_t factor_rank = model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(model.factor_covariances.front().factor.cols())
        : -1;
    Expectation out = empty_expectation(
        documents, components, dimension, factor_rank);
    if (store_responsibilities) {
        out.responsibilities.resize(documents, components);
    }
    std::vector<LowRankDiagonalSolver> factor_solvers;
    std::vector<Eigen::MatrixXd> factor_beta, factor_conditional;
    if (factor_rank >= 0) {
        factor_solvers.reserve(components);
        for (int32_t c = 0; c < components; ++c) {
            const auto& covariance = model.factor_covariances[c];
            factor_solvers.emplace_back(
                covariance.diagonal, covariance.factor);
            factor_beta.push_back(factor_solvers.back().solve_matrix(
                covariance.factor).transpose());
            factor_conditional.push_back(
                Eigen::MatrixXd::Identity(factor_rank, factor_rank)
                - factor_beta.back() * covariance.factor);
        }
    }
    const std::vector<DenseGaussianSolver> dense_solvers =
        dense_model_solvers(model);
    const int32_t requested_blocks = expectation_shards(
        documents, components, dimension, factor_rank);
    const int32_t block_size =
        (documents + requested_blocks - 1) / requested_blocks;
    const int32_t n_blocks = (documents + block_size - 1) / block_size;
    std::vector<ExpectationBlock> blocks;
    blocks.reserve(n_blocks);
    for (int32_t block = 0; block < n_blocks; ++block) {
        blocks.emplace_back(components, dimension, factor_rank);
    }
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        ExpectationBlock& block = blocks[block_index];
        Eigen::VectorXd score(components);
        Eigen::VectorXd responsibility(components);
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        for (int32_t d = begin; d < end; ++d) {
            const Eigen::VectorXd value = data.coordinates.row(d).transpose();
            for (int32_t c = 0; c < components; ++c) {
                if (!(model.weights(c) > 0.0)) {
                    score(c) = -std::numeric_limits<double>::infinity();
                } else if (factor_rank < 0) {
                    score(c) = std::log(model.weights(c))
                        + dense_solvers[c].log_density(value);
                } else {
                    const Eigen::VectorXd residual = value
                        - model.means.row(c).transpose();
                    score(c) = std::log(model.weights(c)) - 0.5
                        * (dimension * kLog2Pi
                            + factor_solvers[c].log_determinant()
                            + factor_solvers[c].quadratic(residual));
                }
            }
            const double normalizer = logsumexp(score);
            block.log_likelihood += normalizer;
            responsibility = (score.array() - normalizer).exp();
            if (store_responsibilities) {
                out.responsibilities.row(d) = responsibility.transpose();
            }
            for (int32_t c = 0; c < components; ++c) {
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

template<class ParticleCollection>
Expectation particle_expectation_impl(const ParticleCollection& particles,
    const Model& model, bool store_responsibilities = false,
    bool collect_diagnostics = false) {
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
    Expectation out = empty_expectation(
        documents, components, dimension, factor_rank);
    if (store_responsibilities) {
        out.responsibilities.resize(documents, components);
    }
    if (collect_diagnostics) {
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
            factor_beta.push_back(factor_solvers.back().solve_matrix(
                covariance.factor).transpose());
            factor_conditional.push_back(
                Eigen::MatrixXd::Identity(factor_rank, factor_rank)
                - factor_beta.back() * covariance.factor);
        }
    }
    const std::vector<DenseGaussianSolver> dense_solvers =
        dense_model_solvers(model);
    const int32_t requested_blocks = expectation_shards(
        documents, components, dimension, factor_rank);
    const int32_t block_size =
        (documents + requested_blocks - 1) / requested_blocks;
    const int32_t n_blocks = (documents + block_size - 1) / block_size;
    out.accumulator_bytes = static_cast<uint64_t>(n_blocks)
        * expectation_block_bytes(components, dimension, factor_rank);
    std::vector<ExpectationBlock> blocks;
    blocks.reserve(n_blocks);
    for (int32_t block = 0; block < n_blocks; ++block) {
        blocks.emplace_back(components, dimension, factor_rank);
    }
    std::atomic<int64_t> gaussian_nanoseconds{0};
    std::atomic<int64_t> moment_nanoseconds{0};
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        ExpectationBlock& block = blocks[block_index];
        Eigen::MatrixXd log_tilt(components, maximum_samples);
        Eigen::VectorXd evidence(components), score(components);
        Eigen::VectorXd responsibility(components);
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        int64_t local_gaussian_nanoseconds = 0;
        int64_t local_moment_nanoseconds = 0;
        for (int32_t d = begin; d < end; ++d) {
            const int32_t samples = particles.samples_for_document(d);
            const auto gaussian_start = std::chrono::steady_clock::now();
            const auto values = particles.values_for_document(d);
            const Eigen::VectorXd base =
                particles.log_likelihood_for_document(d)
                - particles.log_proposal_for_document(d)
                - Eigen::VectorXd::Constant(samples, std::log(samples));
            for (int32_t c = 0; c < components; ++c) {
                if (!(model.weights(c) > 0.0)) {
                    evidence(c) = -std::numeric_limits<double>::infinity();
                    score(c) = -std::numeric_limits<double>::infinity();
                    continue;
                }
                if (factor_rank < 0) {
                    log_tilt.row(c).head(samples) = (base
                        + dense_solvers[c].log_density_rows(values)).transpose();
                } else {
                    for (int32_t s = 0; s < samples; ++s) {
                        const Eigen::VectorXd value = values.row(s).transpose();
                        const Eigen::VectorXd residual = value
                            - model.means.row(c).transpose();
                        const double gaussian = -0.5 * (dimension * kLog2Pi
                            + factor_solvers[c].log_determinant()
                            + factor_solvers[c].quadratic(residual));
                        log_tilt(c, s) = base(s) + gaussian;
                    }
                }
                evidence(c) = logsumexp(
                    log_tilt.row(c).head(samples).transpose());
                score(c) = std::log(model.weights(c)) + evidence(c);
            }
            local_gaussian_nanoseconds +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - gaussian_start).count();
            const double normalizer = logsumexp(score);
            block.log_likelihood += normalizer;
            responsibility = (score.array() - normalizer).exp();
            if (store_responsibilities) {
                out.responsibilities.row(d) = responsibility.transpose();
            }
            if (collect_diagnostics) {
                Eigen::VectorXd log_weight(samples);
                Eigen::VectorXd component_weight(components);
                for (int32_t s = 0; s < samples; ++s) {
                    for (int32_t c = 0; c < components; ++c) {
                        component_weight(c) = model.weights(c) > 0.0
                            ? std::log(model.weights(c)) + log_tilt(c, s)
                            : -std::numeric_limits<double>::infinity();
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
            const auto moment_start = std::chrono::steady_clock::now();
            for (int32_t c = 0; c < components; ++c) {
                const double component_responsibility = responsibility(c);
                if (!(component_responsibility > 0.0)) continue;
                block.membership(c) += component_responsibility;
                const Eigen::VectorXd tau =
                    (log_tilt.row(c).head(samples).transpose().array()
                        - evidence(c)).exp();
                if (factor_rank < 0) {
                    block.first.row(c).noalias() += component_responsibility
                        * (values.transpose() * tau).transpose();
                    RowMajorMatrixXd weighted = values;
                    weighted.array().colwise() *= tau.array().sqrt();
                    block.second[c].noalias() += component_responsibility
                        * weighted.transpose() * weighted;
                } else {
                    Eigen::VectorXd first = Eigen::VectorXd::Zero(dimension);
                    for (int32_t s = 0; s < samples; ++s) {
                        const Eigen::VectorXd value = values.row(s).transpose();
                        const double weight = component_responsibility * tau(s);
                        first.noalias() += tau(s) * value;
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
                    block.first.row(c) += component_responsibility
                        * first.transpose();
                }
            }
            local_moment_nanoseconds +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - moment_start).count();
        }
        gaussian_nanoseconds.fetch_add(
            local_gaussian_nanoseconds, std::memory_order_relaxed);
        moment_nanoseconds.fetch_add(
            local_moment_nanoseconds, std::memory_order_relaxed);
    });
    reduce_expectation_blocks(out, blocks);
    out.gaussian_seconds = 1e-9 * gaussian_nanoseconds.load();
    out.moment_seconds = 1e-9 * moment_nanoseconds.load();
    return out;
}

Expectation particle_expectation(const ParticleSet& particles,
    const Model& model, bool store_responsibilities = false,
    bool collect_diagnostics = false) {
    return particle_expectation_impl(
        particles, model, store_responsibilities, collect_diagnostics);
}

Expectation particle_expectation(const RaggedParticleSet& particles,
    const Model& model, bool store_responsibilities = false,
    bool collect_diagnostics = false) {
    return particle_expectation_impl(
        particles, model, store_responsibilities, collect_diagnostics);
}

Model initialize_model_from_partition(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments, int32_t components,
    double shrinkage, double covariance_floor, double relative_floor) {
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (assignments.size() != data.coordinates.rows() || components <= 0
        || !(shrinkage >= 0.0) || !(relative_floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC initial partition");
    }
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(components);
    Model model;
    model.means = RowMajorMatrixXd::Zero(components, dimension);
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        const int32_t component = assignments(d);
        if (component < 0 || component >= components) {
            throw std::invalid_argument(
                "UAC initial partition label is out of range");
        }
        ++counts(component);
        model.means.row(component) += data.coordinates.row(d);
    }
    for (int32_t c = 0; c < components; ++c) {
        if (counts(c) <= 0) {
            throw std::runtime_error(
                "UAC initial partition produced an empty component");
        }
        model.means.row(c) /= counts(c);
    }
    std::vector<Eigen::MatrixXd> scatter(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        const int32_t component = assignments(d);
        const Eigen::VectorXd residual = data.coordinates.row(d).transpose()
            - model.means.row(component).transpose();
        scatter[component].noalias() += residual * residual.transpose();
    }
    model.shrinkage_target = Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& value : scatter) model.shrinkage_target += value;
    model.shrinkage_target /= data.coordinates.rows();
    const double target_floor = std::max(1e-12,
        relative_floor * model.shrinkage_target.trace() / dimension);
    model.shrinkage_target = floor_covariance(
        model.shrinkage_target, target_floor);
    model.weights = counts.cast<double>() / data.coordinates.rows();
    model.covariances.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        model.covariances.push_back(floor_covariance(
            (scatter[c] + shrinkage * model.shrinkage_target)
                / (counts(c) + shrinkage),
            covariance_floor));
    }
    return model;
}

Pilot pilot_from_map(const Dataset& data, const Model& model,
    const Expectation& expectation,
    double relative_floor) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (expectation.documents != data.coordinates.rows()
        || expectation.membership.size() != components
        || expectation.first.rows() != components
        || expectation.second.size() != static_cast<size_t>(components)
        || !(relative_floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC winning MAP responsibilities");
    }
    Pilot out;
    out.weights = model.weights;
    out.means = model.means;
    out.raw_covariances.assign(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    const Eigen::VectorXd& membership = expectation.membership;
    for (int32_t c = 0; c < components; ++c) {
        const Eigen::VectorXd mean = model.means.row(c).transpose();
        out.raw_covariances[c] = expectation.second[c]
            - expectation.first.row(c).transpose() * mean.transpose()
            - mean * expectation.first.row(c)
            + membership(c) * mean * mean.transpose();
        out.raw_covariances[c] = 0.5 * (out.raw_covariances[c]
            + out.raw_covariances[c].transpose());
    }
    out.pooled_covariance = Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& scatter : out.raw_covariances) {
        out.pooled_covariance += scatter;
    }
    const double total_membership = membership.sum();
    if (!(total_membership > 0.0)) {
        throw std::runtime_error("UAC winning MAP has no active membership");
    }
    out.pooled_covariance /= total_membership;
    const double target_floor = std::max(1e-12,
        relative_floor * out.pooled_covariance.trace() / dimension);
    out.pooled_covariance = floor_covariance(
        out.pooled_covariance, target_floor);
    out.covariances = model.covariances;
    const double epsilon = membership_epsilon(data.coordinates.rows());
    for (int32_t c = 0; c < components; ++c) {
        if (membership(c) > epsilon && model.weights(c) > 0.0) {
            out.raw_covariances[c] /= membership(c);
            out.raw_covariances[c] = 0.5 * (out.raw_covariances[c]
                + out.raw_covariances[c].transpose());
        } else {
            out.raw_covariances[c] = out.pooled_covariance;
            out.covariances[c] = out.pooled_covariance;
        }
    }
    return out;
}

struct ModelUpdate {
    bool valid = false;
    int32_t active_components = 0;
};

LowRankDiagonalCovariance shrink_factor_covariance(
    const LowRankDiagonalCovariance& raw,
    const LowRankDiagonalCovariance& target, double membership,
    double shrinkage, int32_t rank, double floor) {
    const double alpha = membership / (membership + shrinkage);
    LowRankDiagonalCovariance out;
    out.diagonal = (alpha * raw.diagonal
        + (1.0 - alpha) * target.diagonal).cwiseMax(floor);
    const int32_t columns = static_cast<int32_t>(raw.factor.cols()
        + target.factor.cols());
    if (rank == 0 || columns == 0) {
        out.factor = RowMajorMatrixXd(raw.diagonal.size(), 0);
        if (columns > 0) {
            out.diagonal.array() += alpha
                * raw.factor.array().square().rowwise().sum();
            out.diagonal.array() += (1.0 - alpha)
                * target.factor.array().square().rowwise().sum();
        }
        return out;
    }
    Eigen::MatrixXd combined(raw.diagonal.size(), columns);
    if (raw.factor.cols() > 0) {
        combined.leftCols(raw.factor.cols()) = std::sqrt(alpha) * raw.factor;
    }
    if (target.factor.cols() > 0) {
        combined.rightCols(target.factor.cols()) = std::sqrt(1.0 - alpha)
            * target.factor;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(combined,
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    const int32_t retained = std::min<int32_t>(rank,
        static_cast<int32_t>(svd.singularValues().size()));
    out.factor = svd.matrixU().leftCols(retained)
        * svd.singularValues().head(retained).asDiagonal();
    if (retained < svd.singularValues().size()) {
        const Eigen::MatrixXd discarded = svd.matrixU().middleCols(
            retained, svd.matrixU().cols() - retained)
            * svd.singularValues().segment(retained,
                svd.singularValues().size() - retained).asDiagonal();
        out.diagonal.array() += discarded.array().square().rowwise().sum();
    }
    out.diagonal = out.diagonal.cwiseMax(floor);
    return out;
}

ModelUpdate update_model(Model& model, const Expectation& expectation,
    double shrinkage, double covariance_floor, bool adaptive_target = false) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t documents = expectation.documents;
    const double epsilon = membership_epsilon(documents);
    if (!expectation.membership.allFinite()
        || (expectation.membership.array() < 0.0).any()) {
        return {};
    }
    double active_mass = 0.0;
    for (int32_t c = 0; c < components; ++c) {
        if (expectation.membership(c) > epsilon) {
            active_mass += expectation.membership(c);
        }
    }
    if (!(active_mass > 0.0) || !std::isfinite(active_mass)) return {};

    Model next = model;
    next.weights.setZero();
    ModelUpdate result;
    std::vector<Eigen::MatrixXd> raw_dense(components);
    std::vector<LowRankDiagonalCovariance> raw_factor(components);
    for (int32_t c = 0; c < components; ++c) {
        const double membership = expectation.membership(c);
        if (!(membership > epsilon)) continue;
        if (!expectation.first.row(c).allFinite()
            || (model.covariance_kind == CovarianceKind::Dense
                && !expectation.second[c].allFinite())) {
            return {};
        }
        next.weights(c) = membership / active_mass;
        if (model.covariance_kind == CovarianceKind::FactorAnalytic) {
            const int32_t rank = static_cast<int32_t>(
                model.factor_covariances[c].factor.cols());
            if (expectation.sum_y2.cols() != model.means.cols()
                || expectation.sum_f.cols() != rank) {
                return {};
            }
            Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(
                rank + 1, rank + 1);
            gram(0, 0) = membership;
            if (rank > 0) {
                gram.block(0, 1, 1, rank) = expectation.sum_f.row(c);
                gram.block(1, 0, rank, 1) =
                    expectation.sum_f.row(c).transpose();
                gram.bottomRightCorner(rank, rank) =
                    expectation.sum_ff[c];
            }
            Eigen::LLT<Eigen::MatrixXd> regression_llt(gram);
            if (regression_llt.info() != Eigen::Success) return {};
            Eigen::MatrixXd cross(model.means.cols(), rank + 1);
            cross.col(0) = expectation.first.row(c).transpose();
            if (rank > 0) cross.rightCols(rank) = expectation.sum_yf[c];
            const Eigen::MatrixXd coefficient = regression_llt.solve(
                cross.transpose()).transpose();
            const Eigen::VectorXd mean = coefficient.col(0);
            const Eigen::MatrixXd loading = coefficient.rightCols(rank);
            const Eigen::MatrixXd centered_yf = expectation.sum_yf[c]
                - mean * expectation.sum_f.row(c);
            Eigen::VectorXd centered_y2 =
                expectation.sum_y2.row(c).transpose();
            centered_y2.array() -= 2.0 * mean.array()
                * expectation.first.row(c).transpose().array();
            centered_y2.array() += membership * mean.array().square();
            Eigen::VectorXd residual = centered_y2;
            residual.array() -= 2.0
                * (loading.cwiseProduct(centered_yf)).rowwise().sum().array();
            residual.array() += (loading * expectation.sum_ff[c])
                .cwiseProduct(loading).rowwise().sum().array();
            raw_factor[c].diagonal =
                (residual / membership).cwiseMax(covariance_floor);
            raw_factor[c].factor = loading;
            next.means.row(c) = mean.transpose();
            ++result.active_components;
            continue;
        }
        next.means.row(c) = expectation.first.row(c) / membership;
        const Eigen::VectorXd mean = next.means.row(c).transpose();
        Eigen::MatrixXd scatter = expectation.second[c]
            - membership * mean * mean.transpose();
        scatter = 0.5 * (scatter + scatter.transpose());
        if (!scatter.allFinite()) return {};
        raw_dense[c] = scatter / membership;
        ++result.active_components;
    }
    if (result.active_components == 0 || !next.weights.allFinite()
        || !next.means.allFinite()) {
        return {};
    }
    try {
        if (adaptive_target && shrinkage > 0.0) {
            Eigen::MatrixXd pooled = Eigen::MatrixXd::Zero(
                model.means.cols(), model.means.cols());
            for (int32_t c = 0; c < components; ++c) {
                const double membership = expectation.membership(c);
                if (!(membership > epsilon)) continue;
                pooled += membership * (model.covariance_kind
                        == CovarianceKind::Dense
                    ? raw_dense[c] : raw_factor[c].dense());
            }
            next.shrinkage_target = floor_covariance(
                pooled / active_mass, covariance_floor);
            if (model.covariance_kind == CovarianceKind::FactorAnalytic) {
                const int32_t rank = static_cast<int32_t>(
                    model.factor_covariances.front().factor.cols());
                next.factor_shrinkage_target = factorize_covariance(
                    next.shrinkage_target, rank, covariance_floor);
            }
        }
        for (int32_t c = 0; c < components; ++c) {
            const double membership = expectation.membership(c);
            if (!(membership > epsilon)) {
                if (model.covariance_kind == CovarianceKind::Dense) {
                    next.covariances[c] = next.shrinkage_target;
                } else {
                    next.factor_covariances[c] =
                        next.factor_shrinkage_target;
                }
                continue;
            }
            if (model.covariance_kind == CovarianceKind::Dense) {
                next.covariances[c] = floor_covariance(
                    (membership * raw_dense[c]
                        + shrinkage * next.shrinkage_target)
                        / (membership + shrinkage),
                    covariance_floor);
            } else {
                const int32_t rank = static_cast<int32_t>(
                    model.factor_covariances[c].factor.cols());
                next.factor_covariances[c] = shrink_factor_covariance(
                    raw_factor[c], next.factor_shrinkage_target, membership,
                    shrinkage, rank, covariance_floor);
            }
        }
    } catch (const std::runtime_error&) {
        return {};
    }
    result.valid = true;
    model = std::move(next);
    return result;
}

struct Candidate {
    Model model;
    RestartTrace trace;
    double objective = -std::numeric_limits<double>::infinity();
};

double mean_max_responsibility_change(
    const Eigen::Ref<const RowMajorMatrixXd>& current,
    const Eigen::Ref<const RowMajorMatrixXd>& previous) {
    if (current.rows() != previous.rows() || current.cols() != previous.cols()
        || current.rows() == 0) {
        throw std::invalid_argument(
            "Incompatible UAC responsibility convergence matrices");
    }
    double total = 0.0;
    for (Eigen::Index d = 0; d < current.rows(); ++d) {
        total += (current.row(d) - previous.row(d)).cwiseAbs().maxCoeff();
    }
    return total / current.rows();
}

void record_iteration_diagnostic(RestartTrace& trace,
    const FitOptions& options, int32_t iteration,
    double relative_objective_change, double responsibility_change) {
    trace.relative_objective_change.push_back(relative_objective_change);
    trace.mean_max_responsibility_change.push_back(responsibility_change);
    if (options.iteration_callback) {
        options.iteration_callback({trace.particle, trace.start, iteration,
            relative_objective_change, responsibility_change});
    }
}

Candidate fit_map_candidate(const Dataset& data, Model initial,
    const FitOptions& options, const RestartTrace& metadata) {
    Candidate out;
    out.trace = metadata;
    out.trace.particle = false;
    out.model = std::move(initial);
    const double shrinkage = options.adaptive_covariance_shrinkage
        ? kCovarianceShrinkageStrength : 0.0;
    RowMajorMatrixXd previous_responsibilities;
    double converged_log_likelihood =
        -std::numeric_limits<double>::infinity();
    bool reuse_converged_expectation = false;
    for (int32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        Expectation expectation = map_expectation(data, out.model, true);
        const double objective = expectation.log_likelihood
            + covariance_prior(out.model, shrinkage);
        out.trace.objective.push_back(objective);
        out.trace.active_components.push_back(
            active_component_count(out.model));
        double relative_change = std::numeric_limits<double>::quiet_NaN();
        double responsibility_change =
            std::numeric_limits<double>::quiet_NaN();
        if (previous_responsibilities.size() > 0) {
            responsibility_change = mean_max_responsibility_change(
                expectation.responsibilities, previous_responsibilities);
        }
        if (out.trace.objective.size() >= 2) {
            const double previous = out.trace.objective[
                out.trace.objective.size() - 2];
            relative_change = std::abs(objective - previous)
                / std::max(1.0, std::abs(previous));
        }
        record_iteration_diagnostic(out.trace, options, iteration,
            relative_change, responsibility_change);
        if (iteration > 0
            && (relative_change < options.objective_change_tolerance
                || responsibility_change
                    < options.responsibility_change_tolerance)) {
                out.trace.converged = true;
                converged_log_likelihood = expectation.log_likelihood;
                reuse_converged_expectation = true;
                break;
        }
        previous_responsibilities = expectation.responsibilities;
        const ModelUpdate update = update_model(out.model, expectation,
            shrinkage, options.covariance_floor);
        if (!update.valid) {
            out.trace.collapsed = true;
            return out;
        }
    }
    out.objective = reuse_converged_expectation
        ? converged_log_likelihood
        : map_expectation(data, out.model).log_likelihood;
    out.trace.selection_objective = out.objective;
    out.trace.objective.push_back(out.objective
        + covariance_prior(out.model, shrinkage));
    out.trace.relative_objective_change.push_back(
        std::numeric_limits<double>::quiet_NaN());
    out.trace.mean_max_responsibility_change.push_back(
        std::numeric_limits<double>::quiet_NaN());
    out.trace.active_components.push_back(active_component_count(out.model));
    return out;
}

Candidate fit_particle_candidate(
    const std::function<Expectation(const Model&)>& expectation_function,
    Model initial, const FitOptions& options, const RestartTrace& map_trace) {
    Candidate out;
    out.trace = map_trace;
    out.trace.objective.clear();
    out.trace.relative_objective_change.clear();
    out.trace.mean_max_responsibility_change.clear();
    out.trace.active_components.clear();
    out.trace.converged = false;
    out.trace.collapsed = false;
    out.trace.particle = true;
    out.model = std::move(initial);
    const double shrinkage = options.adaptive_covariance_shrinkage
        ? kCovarianceShrinkageStrength : 0.0;
    RowMajorMatrixXd previous_responsibilities;
    double converged_log_likelihood =
        -std::numeric_limits<double>::infinity();
    bool reuse_converged_expectation = false;
    int32_t iteration_offset = 0;
    if (options.adaptive_covariance_shrinkage) {
        Expectation bootstrap = expectation_function(out.model);
        out.trace.objective.push_back(bootstrap.log_likelihood);
        out.trace.active_components.push_back(
            active_component_count(out.model));
        record_iteration_diagnostic(out.trace, options, 0,
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
        previous_responsibilities = bootstrap.responsibilities;
        const ModelUpdate update = update_model(out.model, bootstrap, 0.0,
            options.covariance_floor);
        if (!update.valid) {
            out.trace.collapsed = true;
            return out;
        }
        iteration_offset = 1;
    }
    for (int32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        Expectation expectation = expectation_function(out.model);
        const double objective = expectation.log_likelihood;
        out.trace.objective.push_back(objective);
        out.trace.active_components.push_back(
            active_component_count(out.model));
        double relative_change = std::numeric_limits<double>::quiet_NaN();
        double responsibility_change =
            std::numeric_limits<double>::quiet_NaN();
        if (out.trace.objective.size() >= 2) {
            const double previous = out.trace.objective[
                out.trace.objective.size() - 2];
            relative_change = std::abs(objective - previous)
                / std::max(1.0, std::abs(previous));
        }
        if (previous_responsibilities.size() > 0) {
            responsibility_change = mean_max_responsibility_change(
                expectation.responsibilities, previous_responsibilities);
        }
        record_iteration_diagnostic(out.trace, options,
            iteration + iteration_offset, relative_change,
            responsibility_change);
        if (std::isfinite(relative_change)
            && (relative_change < options.objective_change_tolerance
                || responsibility_change
                    < options.responsibility_change_tolerance)) {
            out.trace.converged = true;
            converged_log_likelihood = expectation.log_likelihood;
            reuse_converged_expectation = true;
            break;
        }
        previous_responsibilities = expectation.responsibilities;
        const ModelUpdate update = update_model(out.model, expectation,
            shrinkage, options.covariance_floor,
            options.adaptive_covariance_shrinkage);
        if (!update.valid) {
            out.trace.collapsed = true;
            return out;
        }
    }
    out.objective = reuse_converged_expectation
        ? converged_log_likelihood
        : expectation_function(out.model).log_likelihood;
    out.trace.objective.push_back(out.objective);
    out.trace.relative_objective_change.push_back(
        std::numeric_limits<double>::quiet_NaN());
    out.trace.mean_max_responsibility_change.push_back(
        std::numeric_limits<double>::quiet_NaN());
    out.trace.active_components.push_back(active_component_count(out.model));
    return out;
}

template<class ParticleCollection>
ScoreResult score_particles_impl(
    const ParticleCollection& particles, const Model& model) {
    ScoreResult out;
    Expectation expectation = particle_expectation(
        particles, model, true, true);
    out.responsibilities = std::move(expectation.responsibilities);
    out.particle_diagnostics = std::move(expectation.particle_diagnostics);
    out.gaussian_seconds = expectation.gaussian_seconds;
    out.moment_seconds = expectation.moment_seconds;
    out.expectation_accumulator_bytes = expectation.accumulator_bytes;
    out.sampling_seconds = particles.sampling_seconds;
    out.likelihood_seconds = particles.likelihood_seconds;
    out.fisher_work_seconds = particles.fisher_work_seconds;
    out.proposal_component_work_seconds =
        particles.proposal_component_work_seconds;
    out.proposal_draw_density_work_seconds =
        particles.proposal_draw_density_work_seconds;
    out.proposal_workspace_bytes = particles.proposal_workspace_bytes;
    int64_t total_samples = 0;
    out.per_document_particles.resize(particles.documents);
    for (int32_t d = 0; d < particles.documents; ++d) {
        const int32_t samples = particles.samples_for_document(d);
        out.per_document_particles[d] = samples;
        total_samples += samples;
    }
    out.particle_samples = total_samples;
    out.particle_bytes = sizeof(double) * static_cast<uint64_t>(total_samples)
        * (particles.dimension + 2);
    out.particle_block_size = particles.documents;
    out.particle_generation_passes = 1;
    out.particle_replay = false;
    return out;
}

ScoreResult score_particles(const ParticleSet& particles, const Model& model) {
    return score_particles_impl(particles, model);
}

ScoreResult score_particles(
    const RaggedParticleSet& particles, const Model& model) {
    ScoreResult out = score_particles_impl(particles, model);
    out.calibration_seconds = particles.calibration_seconds;
    out.calibration_samples = particles.calibration_samples;
    out.reused_calibration_samples = particles.reused_calibration_samples;
    out.adaptive_particle_diagnostics = particles.adaptive_diagnostics;
    return out;
}

std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> out;
    std::string token;
    std::istringstream input(line);
    while (input >> token) out.push_back(token);
    return out;
}

double entropy(const Eigen::Ref<const Eigen::RowVectorXd>& probability) {
    double value = 0.0;
    for (Eigen::Index i = 0; i < probability.size(); ++i) {
        if (probability(i) > 0.0) value -= probability(i) * std::log(probability(i));
    }
    return value;
}

} // namespace

const char* handoff_name(HandoffMode value) {
    return value == HandoffMode::Map ? "map" : "particle";
}

const char* proposal_name(ProposalKind value) {
    return value == ProposalKind::ExactFisher
        ? "exact_fisher" : "sparse_empirical_fisher";
}

const char* initializer_name(MapInitializer value) {
    return value == MapInitializer::KMeans ? "kmeans++" : "leiden";
}

const char* adaptive_particle_rule_name(AdaptiveParticleRule value) {
    switch (value) {
        case AdaptiveParticleRule::Legacy: return "legacy";
        case AdaptiveParticleRule::ResponsibilityOnly:
            return "responsibility_only";
        case AdaptiveParticleRule::MomentOnly:
            return "moment_only";
        case AdaptiveParticleRule::ResponsibilityMoment:
            return "responsibility_moment";
    }
    throw std::invalid_argument("Unknown adaptive particle rule");
}

const char* adaptive_particle_binding_name(AdaptiveParticleBinding value) {
    switch (value) {
        case AdaptiveParticleBinding::Minimum: return "minimum";
        case AdaptiveParticleBinding::LegacyEss: return "legacy_ess";
        case AdaptiveParticleBinding::LegacyMaximumWeight:
            return "legacy_maximum_weight";
        case AdaptiveParticleBinding::LegacyContrast:
            return "legacy_contrast";
        case AdaptiveParticleBinding::Responsibility:
            return "responsibility";
        case AdaptiveParticleBinding::MomentEss: return "moment_ess";
    }
    throw std::invalid_argument("Unknown adaptive particle binding");
}

HandoffMode parse_handoff(const std::string& value) {
    if (value == "map") return HandoffMode::Map;
    if (value == "particle") return HandoffMode::Particle;
    throw std::invalid_argument("UAC handoff must be map or particle");
}

ProposalKind parse_proposal(const std::string& value) {
    if (value == "exact_fisher") return ProposalKind::ExactFisher;
    if (value == "sparse_empirical_fisher") {
        return ProposalKind::SparseEmpiricalFisher;
    }
    throw std::invalid_argument(
        "UAC proposal must be exact_fisher or sparse_empirical_fisher");
}

MapInitializer parse_initializer(const std::string& value) {
    if (value == "kmeans++") return MapInitializer::KMeans;
    if (value == "leiden") return MapInitializer::Leiden;
    throw std::invalid_argument("UAC initializer must be kmeans++ or leiden");
}

AdaptiveParticleRule parse_adaptive_particle_rule(const std::string& value) {
    if (value == "legacy") return AdaptiveParticleRule::Legacy;
    if (value == "responsibility_only") {
        return AdaptiveParticleRule::ResponsibilityOnly;
    }
    if (value == "moment_only") {
        return AdaptiveParticleRule::MomentOnly;
    }
    if (value == "responsibility_moment") {
        return AdaptiveParticleRule::ResponsibilityMoment;
    }
    throw std::invalid_argument("Unknown adaptive particle rule: " + value);
}

namespace detail {

double increased_leiden_resolution(double resolution, int32_t raw_communities,
    int32_t requested_communities) {
    if (!(resolution > 0.0) || !std::isfinite(resolution)
        || raw_communities <= 0 || requested_communities <= 0
        || raw_communities >= requested_communities) {
        throw std::invalid_argument("Invalid adaptive Leiden resolution input");
    }
    const double ratio = static_cast<double>(requested_communities)
        / raw_communities;
    const double multiplier = std::min(2.0, std::max(1.25, ratio));
    const double next = resolution * multiplier;
    if (!(next > resolution) || !std::isfinite(next)) {
        throw std::runtime_error("Adaptive Leiden resolution became nonfinite");
    }
    return next;
}

double midpoint_leiden_resolution(double lower, double upper) {
    if (!(lower > 0.0) || !(upper > lower) || !std::isfinite(lower)
        || !std::isfinite(upper)) {
        throw std::invalid_argument("Invalid Leiden resolution bracket");
    }
    const double midpoint = lower + 0.5 * (upper - lower);
    if (!(midpoint > lower && midpoint < upper)
        || !std::isfinite(midpoint)) {
        throw std::runtime_error("Leiden resolution midpoint is invalid");
    }
    return midpoint;
}

} // namespace detail

FisherApproximation fisher_approximation(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal) {
    return fisher_approximation_impl(
        coordinate, document, basis, helmert, proposal);
}

Eigen::Map<const Eigen::VectorXd> ParticleSet::value(int32_t document,
    int32_t sample) const {
    return Eigen::Map<const Eigen::VectorXd>(
        values.data() + (static_cast<int64_t>(document) * samples + sample)
            * dimension,
        dimension);
}

double ParticleSet::log_q(int32_t document, int32_t sample) const {
    return log_proposal(document, sample);
}

int32_t ParticleSet::samples_for_document(int32_t) const {
    return samples;
}

Eigen::Map<const RowMajorMatrixXd> ParticleSet::values_for_document(
    int32_t document) const {
    return Eigen::Map<const RowMajorMatrixXd>(
        values.data() + static_cast<int64_t>(document) * samples * dimension,
        samples, dimension);
}

Eigen::Map<const Eigen::VectorXd> ParticleSet::log_likelihood_for_document(
    int32_t document) const {
    return Eigen::Map<const Eigen::VectorXd>(
        log_likelihood.data() + static_cast<int64_t>(document) * samples,
        samples);
}

Eigen::Map<const Eigen::VectorXd> ParticleSet::log_proposal_for_document(
    int32_t document) const {
    return Eigen::Map<const Eigen::VectorXd>(
        log_proposal.data() + static_cast<int64_t>(document) * samples,
        samples);
}

int32_t RaggedParticleSet::samples_for_document(int32_t document) const {
    return static_cast<int32_t>(offsets.at(document + 1)
        - offsets.at(document));
}

Eigen::Map<const RowMajorMatrixXd> RaggedParticleSet::values_for_document(
    int32_t document) const {
    const int64_t offset = offsets.at(document);
    return Eigen::Map<const RowMajorMatrixXd>(
        values.data() + offset * dimension,
        samples_for_document(document), dimension);
}

Eigen::Map<const Eigen::VectorXd>
RaggedParticleSet::log_likelihood_for_document(int32_t document) const {
    const int64_t offset = offsets.at(document);
    return Eigen::Map<const Eigen::VectorXd>(log_likelihood.data() + offset,
        samples_for_document(document));
}

Eigen::Map<const Eigen::VectorXd>
RaggedParticleSet::log_proposal_for_document(int32_t document) const {
    const int64_t offset = offsets.at(document);
    return Eigen::Map<const Eigen::VectorXd>(log_proposal.data() + offset,
        samples_for_document(document));
}

Eigen::MatrixXd normalized_helmert(int32_t topics) {
    if (topics < 2) throw std::invalid_argument("UAC requires at least two topics");
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(topics - 1, topics);
    for (int32_t row = 0; row < topics - 1; ++row) {
        const double denominator = std::sqrt((row + 1.0) * (row + 2.0));
        out.block(row, 0, 1, row + 1).setConstant(1.0 / denominator);
        out(row, row + 1) = -(row + 1.0) / denominator;
    }
    return out;
}

RowMajorMatrixXd ilr_transform(const Eigen::Ref<const RowMajorMatrixXd>& values,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, double floor) {
    if (values.cols() != helmert.cols() || floor <= 0.0) {
        throw std::invalid_argument("Invalid UAC ILR transform dimensions or floor");
    }
    RowMajorMatrixXd out(values.rows(), helmert.rows());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        Eigen::VectorXd normalized = values.row(row).transpose().array().max(floor);
        normalized /= normalized.sum();
        out.row(row) = (helmert * normalized.array().log().matrix()).transpose();
    }
    return out;
}

RowMajorMatrixXd ilr_inverse(const Eigen::Ref<const RowMajorMatrixXd>& values,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    if (values.cols() != helmert.rows()) {
        throw std::invalid_argument("Invalid UAC inverse ILR dimensions");
    }
    RowMajorMatrixXd out(values.rows(), helmert.cols());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        out.row(row) = composition_from_coordinate(
            values.row(row).transpose(), helmert).transpose();
    }
    return out;
}

void normalize_basis(Basis& basis) {
    if (basis.probabilities.rows() == 0 || basis.probabilities.cols() < 2
        || basis.features.size() != static_cast<size_t>(basis.probabilities.rows())
        || basis.topics.size() != static_cast<size_t>(basis.probabilities.cols())
        || !basis.probabilities.allFinite()
        || (basis.probabilities.array() < 0.0).any()) {
        throw std::invalid_argument("Invalid UAC topic basis");
    }
    for (Eigen::Index topic = 0; topic < basis.probabilities.cols(); ++topic) {
        const double total = basis.probabilities.col(topic).sum();
        if (!(total > 0.0)) throw std::invalid_argument("UAC basis has an empty topic");
        basis.probabilities.col(topic) /= total;
    }
    basis.checksum = basis_checksum(basis);
}

void normalize_centers(RowMajorMatrixXd& centers, double floor) {
    if (centers.rows() == 0 || centers.cols() < 2 || !centers.allFinite()
        || (centers.array() < 0.0).any() || !(floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC point centers");
    }
    for (Eigen::Index row = 0; row < centers.rows(); ++row) {
        centers.row(row) = centers.row(row).array().max(floor);
        centers.row(row) /= centers.row(row).sum();
    }
}

uint64_t basis_checksum(const Basis& basis) {
    uint64_t value = 14695981039346656037ull;
    for (const auto& name : basis.features) value = hash_string(value, name);
    for (const auto& name : basis.topics) value = hash_string(value, name);
    for (Eigen::Index row = 0; row < basis.probabilities.rows(); ++row) {
        for (Eigen::Index column = 0; column < basis.probabilities.cols(); ++column) {
            const double number = basis.probabilities(row, column);
            value = fnv_append(value, &number, sizeof(number));
        }
    }
    return value;
}

ParticleSet make_particle_range(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    const PilotCache& pilot_cache,
    ProposalKind proposal_kind, int32_t samples, uint64_t seed,
    double fisher_broadening, int32_t n_threads, int32_t first_document,
    int32_t documents) {
    if (samples <= 0 || data.counts.size() != data.identifiers.size()
        || data.coordinates.rows() != static_cast<Eigen::Index>(data.counts.size())
        || basis.probabilities.cols() != helmert.cols()
        || first_document < 0 || documents <= 0
        || static_cast<int64_t>(first_document) + documents
            > data.coordinates.rows()
        || !(fisher_broadening > 0.0)
        || !std::isfinite(fisher_broadening)) {
        throw std::invalid_argument("Invalid UAC particle input");
    }
    ParticleSet out;
    out.first_document = first_document;
    out.documents = documents;
    out.samples = samples;
    out.dimension = static_cast<int32_t>(helmert.rows());
    const size_t particle_rows = checked_mul(
        static_cast<size_t>(out.documents), static_cast<size_t>(samples),
        "UAC particle rows");
    checked_mul(particle_rows, static_cast<size_t>(out.dimension),
        "UAC particle values");
    if (particle_rows > static_cast<size_t>(
        std::numeric_limits<Eigen::Index>::max())) {
        throw std::overflow_error(
            "UAC particle row count exceeds Eigen index range");
    }
    out.values.resize(static_cast<Eigen::Index>(particle_rows), out.dimension);
    out.log_likelihood.resize(out.documents, samples);
    out.log_proposal.resize(out.documents, samples);
    const uint64_t proposal_workspace = sizeof(double)
        * static_cast<uint64_t>(pilot.weights.size())
        * (2ull * out.dimension * out.dimension + out.dimension + 3);
    out.proposal_workspace_bytes = proposal_workspace;
    std::atomic<int64_t> fisher_nanoseconds{0};
    std::atomic<int64_t> proposal_nanoseconds{0};
    std::atomic<int64_t> draw_nanoseconds{0};
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    const auto sampling_start = std::chrono::steady_clock::now();
    tbb::parallel_for(int32_t{0}, out.documents, [&](int32_t local_document) {
        const int32_t document = first_document + local_document;
        const Eigen::VectorXd center =
            data.coordinates.row(document).transpose();
        const auto fisher_start = std::chrono::steady_clock::now();
        const FisherApproximation fisher = fisher_approximation_impl(center,
            data.counts[document], basis, helmert, proposal_kind);
        const auto proposal_start = std::chrono::steady_clock::now();
        fisher_nanoseconds.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                proposal_start - fisher_start).count(),
            std::memory_order_relaxed);
        const DocumentProposal proposal = fisher_proposal(center, fisher,
            pilot, pilot_cache, fisher_broadening);
        const auto draw_start = std::chrono::steady_clock::now();
        proposal_nanoseconds.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                draw_start - proposal_start).count(),
            std::memory_order_relaxed);
        const uint64_t document_seed = hash_string(
            seed ^ 0x9e3779b97f4a7c15ull, data.identifiers[document]);
        std::mt19937_64 engine(document_seed);
        std::discrete_distribution<int32_t> choose(proposal.weights.data(),
            proposal.weights.data() + proposal.weights.size());
        std::normal_distribution<double> normal(0.0, 1.0);
        for (int32_t sample = 0; sample < samples; ++sample) {
            const int32_t component = choose(engine);
            Eigen::VectorXd draw(out.dimension);
            for (int32_t dim = 0; dim < out.dimension; ++dim) {
                draw(dim) = normal(engine);
            }
            const Eigen::VectorXd value = proposal.means[component]
                + proposal.lower[component] * draw;
            out.values.row(
                static_cast<Eigen::Index>(local_document) * samples + sample)
                = value.transpose();
        }
        const auto values = out.values.middleRows(
            static_cast<Eigen::Index>(local_document) * samples, samples);
        out.log_proposal.row(local_document) =
            proposal_log_density_rows(values, proposal).transpose();
        draw_nanoseconds.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - draw_start).count(),
            std::memory_order_relaxed);
    });
    out.sampling_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sampling_start).count();
    out.fisher_work_seconds = 1e-9 * fisher_nanoseconds.load();
    out.proposal_component_work_seconds =
        1e-9 * proposal_nanoseconds.load();
    out.proposal_draw_density_work_seconds =
        1e-9 * draw_nanoseconds.load();
    const auto likelihood_start = std::chrono::steady_clock::now();
    tbb::parallel_for(int32_t{0}, out.documents, [&](int32_t local_document) {
        const int32_t document = first_document + local_document;
        const auto values = out.values.middleRows(
            static_cast<Eigen::Index>(local_document) * samples, samples);
        out.log_likelihood.row(local_document) = count_log_likelihood_rows(
            values, data.counts[document], basis, helmert).transpose();
    });
    out.likelihood_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - likelihood_start).count();
    return out;
}

template<class Matrix>
void draw_proposal_values(const DocumentProposal& proposal,
    uint64_t document_seed, Matrix& values) {
    std::mt19937_64 engine(document_seed);
    std::discrete_distribution<int32_t> choose(proposal.weights.data(),
        proposal.weights.data() + proposal.weights.size());
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd draw(values.cols());
    for (Eigen::Index sample = 0; sample < values.rows(); ++sample) {
        const int32_t component = choose(engine);
        for (Eigen::Index dim = 0; dim < values.cols(); ++dim) {
            draw(dim) = normal(engine);
        }
        values.row(sample) = (proposal.means[component]
            + proposal.lower[component] * draw).transpose();
    }
}

struct AdaptiveCountResult {
    int32_t particles = 0;
    AdaptiveParticleDiagnostic diagnostic;
};

AdaptiveCountResult adaptive_particle_count(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    const Eigen::Ref<const Eigen::VectorXd>& log_likelihood,
    const Eigen::Ref<const Eigen::VectorXd>& log_proposal,
    const Model& model,
    const std::vector<DenseGaussianSolver>& solvers,
    const AdaptiveParticleOptions& options) {
    const int32_t samples = static_cast<int32_t>(values.rows());
    const int32_t components = static_cast<int32_t>(model.weights.size());
    if (samples < 2 || log_likelihood.size() != samples
        || log_proposal.size() != samples
        || static_cast<int32_t>(solvers.size()) != components) {
        throw std::invalid_argument("Invalid adaptive particle calibration");
    }
    const Eigen::VectorXd base = log_likelihood - log_proposal
        - Eigen::VectorXd::Constant(samples, std::log(samples));
    Eigen::MatrixXd log_tilt(components, samples);
    Eigen::VectorXd evidence(components), score(components);
    for (int32_t c = 0; c < components; ++c) {
        if (!(model.weights(c) > 0.0)) {
            evidence(c) = -std::numeric_limits<double>::infinity();
            score(c) = -std::numeric_limits<double>::infinity();
            log_tilt.row(c).setConstant(
                -std::numeric_limits<double>::infinity());
            continue;
        }
        log_tilt.row(c) = (base
            + solvers[c].log_density_rows(values)).transpose();
        evidence(c) = logsumexp(log_tilt.row(c).transpose());
        score(c) = std::log(model.weights(c)) + evidence(c);
    }
    const double normalizer = logsumexp(score);
    const Eigen::VectorXd responsibility =
        (score.array() - normalizer).exp();
    std::vector<int32_t> order;
    order.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        if (model.weights(c) > 0.0) order.push_back(c);
    }
    std::sort(order.begin(), order.end(), [&](int32_t left, int32_t right) {
        return responsibility(left) > responsibility(right);
    });
    if (order.empty()) {
        throw std::runtime_error("Adaptive calibration has no active component");
    }
    std::vector<int32_t> plausible;
    double cumulative = 0.0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        const int32_t c = order[rank];
        if (cumulative < options.plausible_mass
            || responsibility(c) >= options.plausible_responsibility) {
            plausible.push_back(c);
        }
        cumulative += responsibility(c);
    }
    AdaptiveCountResult out;
    out.diagnostic.preliminary_maximum_responsibility =
        responsibility.maxCoeff();
    for (int32_t c = 0; c < components; ++c) {
        if (responsibility(c) > 0.0) {
            out.diagnostic.preliminary_entropy -= responsibility(c)
                * std::log(responsibility(c));
        }
    }
    out.diagnostic.plausible_components =
        static_cast<int32_t>(plausible.size());

    Eigen::MatrixXd tau = Eigen::MatrixXd::Zero(components, samples);
    for (const int32_t c : order) {
        tau.row(c) = (log_tilt.row(c).array() - evidence(c)).exp();
    }
    for (const int32_t c : plausible) {
        out.diagnostic.plausible_maximum_weight = std::max(
            out.diagnostic.plausible_maximum_weight,
            tau.row(c).maxCoeff());
    }

    auto subset_responsibility = [&](int32_t first, int32_t count)
            -> Eigen::VectorXd {
        Eigen::VectorXd subset_score = Eigen::VectorXd::Constant(components,
            -std::numeric_limits<double>::infinity());
        for (const int32_t c : order) {
            Eigen::VectorXd values_for_component =
                log_tilt.row(c).segment(first, count).transpose();
            subset_score(c) = std::log(model.weights(c))
                + logsumexp(values_for_component);
        }
        const double subset_normalizer = logsumexp(subset_score);
        return (subset_score.array() - subset_normalizer).exp().matrix().eval();
    };
    const int32_t first_half = samples / 2;
    const Eigen::VectorXd left = subset_responsibility(0, first_half);
    const Eigen::VectorXd right = subset_responsibility(
        first_half, samples - first_half);
    out.diagnostic.half_sample_maximum_responsibility_difference =
        (left - right).cwiseAbs().maxCoeff();
    Eigen::Index left_top = 0, right_top = 0;
    left.maxCoeff(&left_top);
    right.maxCoeff(&right_top);
    out.diagnostic.half_sample_top_disagreement = left_top != right_top;

    double required = options.minimum_particles;
    AdaptiveParticleBinding binding = AdaptiveParticleBinding::Minimum;
    auto update_required = [&](double candidate,
            AdaptiveParticleBinding candidate_binding) {
        if (candidate > required) {
            required = candidate;
            binding = candidate_binding;
        }
    };
    if (options.rule == AdaptiveParticleRule::Legacy) {
        std::vector<int32_t> material;
        cumulative = 0.0;
        for (size_t rank = 0; rank < order.size(); ++rank) {
            const int32_t c = order[rank];
            if (cumulative < options.material_mass
                || responsibility(c) >= options.material_responsibility
                || rank < std::min<size_t>(2, order.size())) {
                material.push_back(c);
            }
            cumulative += responsibility(c);
        }
        for (const int32_t c : material) {
            const double relative_ess = 1.0
                / (samples * tau.row(c).squaredNorm());
            update_required(options.component_ess_target
                    / std::max(1e-12, relative_ess),
                AdaptiveParticleBinding::LegacyEss);
            update_required(samples * tau.row(c).maxCoeff()
                    / options.maximum_weight_target,
                AdaptiveParticleBinding::LegacyMaximumWeight);
        }
        if (order.size() >= 2) {
            const double contrast_se = std::sqrt(samples / (samples - 1.0)
                * (tau.row(order[0]) - tau.row(order[1])).squaredNorm());
            update_required(samples
                    * std::pow(contrast_se / options.contrast_se_target, 2.0),
                AdaptiveParticleBinding::LegacyContrast);
        }
    } else {
        Eigen::RowVectorXd mixture_tau = Eigen::RowVectorXd::Zero(samples);
        for (const int32_t c : order) {
            mixture_tau += responsibility(c) * tau.row(c);
        }
        for (const int32_t c : order) {
            const double responsibility_se = responsibility(c)
                * std::sqrt(samples / (samples - 1.0)
                    * (tau.row(c) - mixture_tau).squaredNorm());
            out.diagnostic.maximum_responsibility_se = std::max(
                out.diagnostic.maximum_responsibility_se,
                responsibility_se);
        }
        out.diagnostic.projected_responsibility_particles = samples
            * std::pow(out.diagnostic.maximum_responsibility_se
                / options.responsibility_se_target, 2.0);
        if (options.rule != AdaptiveParticleRule::MomentOnly) {
            update_required(out.diagnostic.projected_responsibility_particles,
                AdaptiveParticleBinding::Responsibility);
        }
        if (options.rule != AdaptiveParticleRule::ResponsibilityOnly) {
            for (const int32_t c : plausible) {
                const double relative_ess = 1.0
                    / (samples * tau.row(c).squaredNorm());
                out.diagnostic.projected_moment_particles = std::max(
                    out.diagnostic.projected_moment_particles,
                    options.moment_ess_target
                        / std::max(1e-12, relative_ess));
            }
            update_required(out.diagnostic.projected_moment_particles,
                AdaptiveParticleBinding::MomentEss);
        }
    }
    int32_t selected = options.minimum_particles;
    while (selected < options.maximum_particles && selected < required) {
        selected = std::min(options.maximum_particles, selected * 2);
    }
    out.particles = selected;
    out.diagnostic.selected_particles = selected;
    out.diagnostic.binding = binding;
    return out;
}

RaggedParticleSet make_adaptive_particles(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal_kind, uint64_t seed, double fisher_broadening,
    int32_t n_threads, const Model& calibration_model,
    const AdaptiveParticleOptions& options) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t dimension = static_cast<int32_t>(helmert.rows());
    if (!options.enabled || documents <= 0
        || options.calibration_particles < 2
        || options.minimum_particles <= 0
        || options.minimum_particles < options.calibration_particles
        || options.maximum_particles < options.minimum_particles
        || !(options.material_mass > 0.0 && options.material_mass <= 1.0)
        || !(options.material_responsibility >= 0.0
            && options.material_responsibility <= 1.0)
        || !(options.component_ess_target > 0.0)
        || !(options.contrast_se_target > 0.0)
        || !(options.maximum_weight_target > 0.0
            && options.maximum_weight_target <= 1.0)
        || !(options.responsibility_se_target > 0.0)
        || !(options.plausible_mass > 0.0 && options.plausible_mass <= 1.0)
        || !(options.plausible_responsibility >= 0.0
            && options.plausible_responsibility <= 1.0)
        || !(options.moment_ess_target > 0.0)) {
        throw std::invalid_argument("Invalid adaptive particle options");
    }
    std::vector<DenseGaussianSolver> calibration_solvers;
    calibration_solvers.reserve(calibration_model.weights.size());
    for (Eigen::Index c = 0; c < calibration_model.weights.size(); ++c) {
        if (calibration_model.weights(c) > 0.0) {
            calibration_solvers.emplace_back(
                calibration_model.means.row(c).transpose(),
                model_covariance_dense(calibration_model, c));
        } else {
            calibration_solvers.emplace_back();
        }
    }
    RaggedParticleSet out;
    out.documents = documents;
    out.dimension = dimension;
    out.maximum_samples = options.maximum_particles;
    out.offsets.assign(static_cast<size_t>(documents) + 1, 0);
    out.adaptive_diagnostics.resize(documents);
    out.calibration_samples = static_cast<int64_t>(documents)
        * options.calibration_particles;
    out.reused_calibration_samples = out.calibration_samples;
    const size_t minimum_total = checked_mul(static_cast<size_t>(documents),
        static_cast<size_t>(options.minimum_particles),
        "UAC adaptive minimum particle count");
    out.values.reserve(checked_mul(minimum_total,
        static_cast<size_t>(dimension), "UAC adaptive particle values"));
    out.log_likelihood.reserve(minimum_total);
    out.log_proposal.reserve(minimum_total);
    out.proposal_workspace_bytes = sizeof(double)
        * static_cast<uint64_t>(pilot.weights.size())
        * (2ull * dimension * dimension + dimension + 3);
    std::atomic<int64_t> fisher_nanoseconds{0};
    std::atomic<int64_t> proposal_nanoseconds{0};
    std::atomic<int64_t> draw_nanoseconds{0};
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    constexpr int32_t kGenerationChunk = 128;
    for (int32_t begin = 0; begin < documents; begin += kGenerationChunk) {
        const int32_t end = std::min(documents, begin + kGenerationChunk);
        const int32_t size = end - begin;
        std::vector<DocumentProposal> proposals(size);
        std::vector<int32_t> counts(size, options.minimum_particles);
        std::vector<RowMajorMatrixXd> calibration_values(size);
        std::vector<Eigen::VectorXd> calibration_log_q(size);
        std::vector<Eigen::VectorXd> calibration_log_likelihood(size);
        const auto calibration_start = std::chrono::steady_clock::now();
        tbb::parallel_for(int32_t{0}, size, [&](int32_t local) {
            const int32_t document = begin + local;
            const Eigen::VectorXd center =
                data.coordinates.row(document).transpose();
            const auto fisher_start = std::chrono::steady_clock::now();
            const FisherApproximation fisher = fisher_approximation_impl(
                center, data.counts[document], basis, helmert, proposal_kind);
            const auto proposal_start = std::chrono::steady_clock::now();
            fisher_nanoseconds.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    proposal_start - fisher_start).count(),
                std::memory_order_relaxed);
            proposals[local] = fisher_proposal(center, fisher, pilot,
                pilot_cache, fisher_broadening);
            proposal_nanoseconds.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - proposal_start).count(),
                std::memory_order_relaxed);
            RowMajorMatrixXd& calibration = calibration_values[local];
            calibration.resize(options.calibration_particles, dimension);
            const uint64_t calibration_seed = hash_string(
                seed ^ 0x6a09e667f3bcc909ull,
                data.identifiers[document]);
            draw_proposal_values(
                proposals[local], calibration_seed, calibration);
            calibration_log_q[local] = proposal_log_density_rows(
                calibration, proposals[local]);
            calibration_log_likelihood[local] = count_log_likelihood_rows(
                calibration, data.counts[document], basis, helmert);
            const AdaptiveCountResult allocation = adaptive_particle_count(
                calibration, calibration_log_likelihood[local],
                calibration_log_q[local], calibration_model,
                calibration_solvers, options);
            counts[local] = allocation.particles;
            out.adaptive_diagnostics[document] = allocation.diagnostic;
        });
        out.calibration_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - calibration_start).count();
        for (int32_t local = 0; local < size; ++local) {
            out.offsets[begin + local + 1] = out.offsets[begin + local]
                + counts[local];
        }
        const int64_t total_samples = out.offsets[end];
        out.values.resize(checked_mul(static_cast<size_t>(total_samples),
            static_cast<size_t>(dimension), "UAC adaptive particle values"));
        out.log_likelihood.resize(total_samples);
        out.log_proposal.resize(total_samples);
        const auto sampling_start = std::chrono::steady_clock::now();
        tbb::parallel_for(int32_t{0}, size, [&](int32_t local) {
            const int32_t document = begin + local;
            const int64_t offset = out.offsets[document];
            const int32_t samples = counts[local];
            Eigen::Map<RowMajorMatrixXd> values(
                out.values.data() + offset * dimension, samples, dimension);
            const int32_t calibration_samples =
                options.calibration_particles;
            values.topRows(calibration_samples) = calibration_values[local];
            Eigen::Map<Eigen::VectorXd> stored_log_q(
                out.log_proposal.data() + offset, samples);
            stored_log_q.head(calibration_samples) =
                calibration_log_q[local];
            Eigen::Map<Eigen::VectorXd> stored_log_likelihood(
                out.log_likelihood.data() + offset, samples);
            stored_log_likelihood.head(calibration_samples) =
                calibration_log_likelihood[local];
            const auto draw_start = std::chrono::steady_clock::now();
            const uint64_t document_seed = hash_string(
                seed ^ 0x9e3779b97f4a7c15ull,
                data.identifiers[document]);
            auto additional = values.bottomRows(
                samples - calibration_samples);
            draw_proposal_values(proposals[local], document_seed, additional);
            stored_log_q.tail(samples - calibration_samples) =
                proposal_log_density_rows(additional, proposals[local]);
            draw_nanoseconds.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - draw_start).count(),
                std::memory_order_relaxed);
        });
        out.sampling_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - sampling_start).count();
        const auto likelihood_start = std::chrono::steady_clock::now();
        tbb::parallel_for(int32_t{0}, size, [&](int32_t local) {
            const int32_t document = begin + local;
            const int64_t offset = out.offsets[document];
            const int32_t samples = counts[local];
            const int32_t additional_samples = samples
                - options.calibration_particles;
            if (additional_samples == 0) return;
            const Eigen::Map<const RowMajorMatrixXd> values(
                out.values.data() + (offset + options.calibration_particles)
                    * dimension,
                additional_samples, dimension);
            Eigen::Map<Eigen::VectorXd>(out.log_likelihood.data() + offset
                    + options.calibration_particles, additional_samples) =
                count_log_likelihood_rows(
                    values, data.counts[document], basis, helmert);
        });
        out.likelihood_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - likelihood_start).count();
    }
    out.fisher_work_seconds = 1e-9 * fisher_nanoseconds.load();
    out.proposal_component_work_seconds =
        1e-9 * proposal_nanoseconds.load();
    out.proposal_draw_density_work_seconds =
        1e-9 * draw_nanoseconds.load();
    return out;
}

ParticleSet make_particles(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    ProposalKind proposal_kind, int32_t samples, uint64_t seed,
    double fisher_broadening, int32_t n_threads) {
    const PilotCache pilot_cache(pilot);
    return make_particle_range(data, basis, helmert, pilot, pilot_cache,
        proposal_kind,
        samples, seed, fisher_broadening, n_threads, 0,
        static_cast<int32_t>(data.coordinates.rows()));
}

uint64_t particle_set_bytes(const ParticleSet& particles) {
    return sizeof(double) * static_cast<uint64_t>(particles.documents)
        * particles.samples * (particles.dimension + 2);
}

struct ParticleReplayMetrics {
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    double fisher_work_seconds = 0.0;
    double proposal_component_work_seconds = 0.0;
    double proposal_draw_density_work_seconds = 0.0;
    uint64_t peak_bytes = 0;
    uint64_t proposal_workspace_bytes = 0;
    int32_t passes = 0;

    void add(const ParticleSet& particles) {
        sampling_seconds += particles.sampling_seconds;
        likelihood_seconds += particles.likelihood_seconds;
        fisher_work_seconds += particles.fisher_work_seconds;
        proposal_component_work_seconds +=
            particles.proposal_component_work_seconds;
        proposal_draw_density_work_seconds +=
            particles.proposal_draw_density_work_seconds;
        peak_bytes = std::max(peak_bytes, particle_set_bytes(particles));
        proposal_workspace_bytes = std::max(
            proposal_workspace_bytes, particles.proposal_workspace_bytes);
    }
};

Expectation replay_particle_expectation(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal, int32_t samples, uint64_t seed,
    double broadening, int32_t n_threads, int32_t block_size,
    const Model& model, ParticleReplayMetrics& metrics,
    bool store_responsibilities = false) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    const int32_t factor_rank = model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(model.factor_covariances.front().factor.cols())
        : -1;
    Expectation out = empty_expectation(
        documents, components, dimension, factor_rank);
    if (store_responsibilities) {
        out.responsibilities.resize(documents, components);
    }
    ++metrics.passes;
    for (int32_t first = 0; first < documents; first += block_size) {
        const int32_t count = std::min(block_size, documents - first);
        ParticleSet block = make_particle_range(data, basis, helmert, pilot,
            pilot_cache, proposal, samples, seed, broadening, n_threads,
            first, count);
        metrics.add(block);
        Expectation local = particle_expectation(
            block, model, store_responsibilities);
        if (store_responsibilities) {
            out.responsibilities.middleRows(first, count) =
                local.responsibilities;
        }
        accumulate_expectation(out, local);
    }
    return out;
}

ScoreResult score_particle_replay(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    const PilotCache& pilot_cache, ProposalKind proposal, int32_t samples,
    uint64_t seed, double broadening,
    int32_t n_threads, int32_t block_size, const Model& model,
    ParticleReplayMetrics& metrics) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    ScoreResult out;
    out.responsibilities.resize(documents, model.weights.size());
    out.particle_diagnostics.resize(documents);
    ++metrics.passes;
    for (int32_t first = 0; first < documents; first += block_size) {
        const int32_t count = std::min(block_size, documents - first);
        ParticleSet block = make_particle_range(data, basis, helmert, pilot,
            pilot_cache, proposal, samples, seed, broadening, n_threads,
            first, count);
        metrics.add(block);
        ScoreResult local = score_particles(block, model);
        out.responsibilities.middleRows(first, count) =
            local.responsibilities;
        for (int32_t d = 0; d < count; ++d) {
            out.particle_diagnostics[first + d] =
                local.particle_diagnostics[d];
        }
        out.gaussian_seconds += local.gaussian_seconds;
        out.moment_seconds += local.moment_seconds;
        out.expectation_accumulator_bytes = std::max(
            out.expectation_accumulator_bytes,
            local.expectation_accumulator_bytes);
    }
    out.sampling_seconds = metrics.sampling_seconds;
    out.likelihood_seconds = metrics.likelihood_seconds;
    out.fisher_work_seconds = metrics.fisher_work_seconds;
    out.proposal_component_work_seconds =
        metrics.proposal_component_work_seconds;
    out.proposal_draw_density_work_seconds =
        metrics.proposal_draw_density_work_seconds;
    out.particle_generation_seconds = out.sampling_seconds
        + out.likelihood_seconds;
    out.particle_bytes = metrics.peak_bytes;
    out.proposal_workspace_bytes = metrics.proposal_workspace_bytes;
    out.particle_block_size = block_size;
    out.particle_generation_passes = metrics.passes;
    out.particle_replay = true;
    return out;
}

FitResult fit(const Dataset& data, const Basis* basis,
    const FitOptions& options) {
    const int64_t total_starts = static_cast<int64_t>(options.kmeans_starts)
        + options.leiden_starts;
    if (options.n_components <= 0 || options.kmeans_starts < 0
        || options.leiden_starts < 0 || total_starts <= 0
        || options.max_iterations <= 0 || options.n_particles <= 0
        || options.particle_block_size < 0
        || (options.adaptive_particles.enabled
            && options.particle_block_size != 0)
        || options.cluster_covariance_rank < -1
        || options.kmeans_max_iterations <= 0
        || data.centers.rows() < options.n_components
        || data.coordinates.rows() != data.centers.rows()
        || !(options.objective_change_tolerance > 0.0)
        || !(options.responsibility_change_tolerance > 0.0)
        || !(options.target_relative_floor > 0.0)
        || !(options.covariance_floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC fit options or dataset");
    }
    if (options.leiden_starts > 0
        && (options.leiden_neighbors <= 0
            || options.leiden_neighbors >= data.centers.rows()
            || options.leiden_max_iterations == 0
            || !(options.leiden_knn_epsilon >= 0.0)
            || !std::isfinite(options.leiden_knn_epsilon)
            || !(options.leiden_resolution > 0.0)
            || !std::isfinite(options.leiden_resolution))) {
        throw std::invalid_argument("Invalid UAC Leiden start options");
    }
    if (options.handoff == HandoffMode::Particle
        && (basis == nullptr || data.counts.size() != data.identifiers.size())) {
        throw std::invalid_argument("Particle UAC requires basis and aligned counts");
    }
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, options.n_threads));
    FitResult result;
    std::vector<Candidate> maps;
    maps.reserve(static_cast<size_t>(total_starts));
    auto append_map = [&](const Eigen::VectorXi& assignments,
                          const RestartTrace& metadata) {
        try {
            Model initial = initialize_model_from_partition(data, assignments,
                options.n_components, options.adaptive_covariance_shrinkage
                    ? kCovarianceShrinkageStrength : 0.0,
                options.covariance_floor, options.target_relative_floor);
            maps.push_back(fit_map_candidate(
                data, std::move(initial), options, metadata));
        } catch (const std::exception&) {
            Candidate failed;
            failed.trace = metadata;
            failed.trace.collapsed = true;
            maps.push_back(std::move(failed));
        }
        result.traces.push_back(maps.back().trace);
    };

    int32_t global_start = 0;
    for (int32_t start = 0; start < options.kmeans_starts;
            ++start, ++global_start) {
        RestartTrace metadata;
        metadata.start = global_start;
        metadata.initializer = MapInitializer::KMeans;
        metadata.seed = map_start_seed(options.seed, global_start);
        metadata.raw_communities = options.n_components;
        DenseKMeansOptions kmeans;
        kmeans.n_clusters = options.n_components;
        kmeans.max_iterations = options.kmeans_max_iterations;
        kmeans.seed = metadata.seed;
        const DenseKMeansResult clustering = cosine_dense_kmeans(
            data.centers, kmeans);
        append_map(clustering.assignments, metadata);
    }

    if (options.leiden_starts > 0) {
        CosineKnnOptions knn_options;
        knn_options.n_neighbors = options.leiden_neighbors;
        knn_options.knn_search_epsilon = options.leiden_knn_epsilon;
        knn_options.backend = options.leiden_knn_backend;
        knn_options.n_threads = options.n_threads;
        const CosineKnnResult knn = cosine_knn(data.centers, knn_options);
        double resolution = options.leiden_resolution;
        double last_under_resolution = 0.0;
        bool adapting = true;
        for (int32_t start = 0; start < options.leiden_starts;
                ++start, ++global_start) {
            RestartTrace metadata;
            metadata.start = global_start;
            metadata.initializer = MapInitializer::Leiden;
            metadata.seed = map_start_seed(options.seed, global_start);
            metadata.leiden_resolution = resolution;
            LeidenOptions leiden_options;
            leiden_options.resolution = resolution;
            leiden_options.max_iterations = options.leiden_max_iterations;
            leiden_options.seed = metadata.seed;
            const LeidenResult leiden = leiden_cluster(knn.graph.n_nodes,
                knn.graph.edges, knn.graph.weights, leiden_options);
            metadata.raw_communities = leiden.n_communities;
            metadata.reconciliation_count = std::abs(
                leiden.n_communities - options.n_components);
            DenseKMeansOptions reconcile_options;
            reconcile_options.n_clusters = options.n_components;
            reconcile_options.max_iterations = options.kmeans_max_iterations;
            reconcile_options.seed = metadata.seed;
            const Eigen::VectorXi assignments = reconcile_cosine_communities(
                leiden.membership, leiden.n_communities,
                options.n_components, data.centers, reconcile_options);
            append_map(assignments, metadata);

            if (!adapting) continue;
            if (leiden.n_communities < options.n_components) {
                last_under_resolution = resolution;
                resolution = detail::increased_leiden_resolution(resolution,
                    leiden.n_communities, options.n_components);
            } else if (leiden.n_communities == options.n_components) {
                adapting = false;
            } else {
                if (last_under_resolution > 0.0) {
                    resolution = detail::midpoint_leiden_resolution(
                        last_under_resolution, resolution);
                }
                adapting = false;
            }
        }
    }

    Candidate* selected = nullptr;
    for (auto& candidate : maps) {
        if (candidate.trace.collapsed || !std::isfinite(candidate.objective)) {
            continue;
        }
        if (selected == nullptr || candidate.objective > selected->objective
            || (candidate.objective == selected->objective
                && candidate.trace.start < selected->trace.start)) {
            selected = &candidate;
        }
    }
    if (selected == nullptr) {
        throw std::runtime_error("Every UAC MAP start failed numerically");
    }
    const Expectation selected_expectation = map_expectation(
        data, selected->model, false);
    result.pilot = pilot_from_map(data, selected->model,
        selected_expectation, options.target_relative_floor);
    selected->model.shrinkage_target = result.pilot.pooled_covariance;
    if (options.cluster_covariance_rank >= 0) {
        const int32_t dimension = static_cast<int32_t>(
            selected->model.means.cols());
        const int32_t rank = options.cluster_covariance_rank;
        if (rank > dimension) {
            throw std::invalid_argument(
                "UAC factor rank exceeds the ILR dimension");
        }
        selected->model.covariance_kind = CovarianceKind::FactorAnalytic;
        selected->model.factor_shrinkage_target = factorize_covariance(
            selected->model.shrinkage_target, rank,
            options.covariance_floor);
        selected->model.factor_covariances.clear();
        selected->model.factor_covariances.reserve(options.n_components);
        for (const auto& covariance : selected->model.covariances) {
            selected->model.factor_covariances.push_back(
                factorize_covariance(covariance, rank,
                    options.covariance_floor));
        }
        const double shrinkage = options.adaptive_covariance_shrinkage
            ? kCovarianceShrinkageStrength : 0.0;
        const double before = map_expectation(data, selected->model, false)
            .log_likelihood + covariance_prior(selected->model, shrinkage);
        Model refined = selected->model;
        const Expectation refinement = map_expectation(data, refined, false);
        const ModelUpdate refinement_update = update_model(refined, refinement,
            shrinkage, options.covariance_floor);
        if (refinement_update.valid) {
            const double after = map_expectation(data, refined, false)
                .log_likelihood + covariance_prior(refined, shrinkage);
            if (std::isfinite(after) && after >= before) {
                selected->model = std::move(refined);
            }
        }
    }
    result.selected_start = selected->trace.start;
    result.selected_initializer = selected->trace.initializer;
    result.selected_leiden_resolution = selected->trace.leiden_resolution;

    if (options.handoff == HandoffMode::Map) {
        result.model = selected->model;
        result.score = score_map(data, result.model, options.n_threads);
        result.converged = selected->trace.converged;
        return result;
    }
    const Eigen::MatrixXd helmert = normalized_helmert(data.centers.cols());
    const PilotCache pilot_cache(result.pilot);
    const uint64_t particle_seed = static_cast<uint64_t>(options.seed) ^ 0xF604;
    Candidate particle;
    try {
        if (options.adaptive_particles.enabled) {
            const RaggedParticleSet particles = make_adaptive_particles(
                data, *basis, helmert, result.pilot, pilot_cache,
                options.proposal, particle_seed, options.fisher_broadening,
                options.n_threads, selected->model,
                options.adaptive_particles);
            auto expectation_function = [&](const Model& model) {
                return particle_expectation(particles, model, true);
            };
            particle = fit_particle_candidate(expectation_function,
                selected->model, options, selected->trace);
            if (!particle.trace.collapsed) {
                const auto score_start = std::chrono::steady_clock::now();
                result.score = score_particles(particles, particle.model);
                result.score.adaptive_particle_options =
                    options.adaptive_particles;
                result.score.particle_generation_seconds =
                    particles.calibration_seconds
                    + particles.sampling_seconds + particles.likelihood_seconds;
                result.score.scoring_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - score_start).count();
            }
        } else if (options.particle_block_size == 0) {
            const ParticleSet particles = make_particles(data, *basis, helmert,
                result.pilot, options.proposal, options.n_particles,
                particle_seed, options.fisher_broadening, options.n_threads);
            auto expectation_function = [&](const Model& model) {
                return particle_expectation(particles, model, true);
            };
            particle = fit_particle_candidate(expectation_function,
                selected->model, options, selected->trace);
            if (!particle.trace.collapsed) {
                const auto score_start = std::chrono::steady_clock::now();
                result.score = score_particles(particles, particle.model);
                result.score.particle_generation_seconds =
                    particles.sampling_seconds + particles.likelihood_seconds;
                result.score.scoring_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - score_start).count();
            }
        } else {
            ParticleReplayMetrics metrics;
            auto expectation_function = [&](const Model& model) {
                return replay_particle_expectation(data, *basis, helmert,
                    result.pilot, pilot_cache, options.proposal,
                    options.n_particles,
                    particle_seed, options.fisher_broadening,
                    options.n_threads, options.particle_block_size, model,
                    metrics, true);
            };
            particle = fit_particle_candidate(expectation_function,
                selected->model, options, selected->trace);
            if (!particle.trace.collapsed) {
                const auto score_start = std::chrono::steady_clock::now();
                result.score = score_particle_replay(data, *basis, helmert,
                    result.pilot, pilot_cache, options.proposal,
                    options.n_particles,
                    particle_seed, options.fisher_broadening,
                    options.n_threads, options.particle_block_size,
                    particle.model, metrics);
                result.score.scoring_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - score_start).count();
            }
        }
    } catch (const std::exception& exception) {
        throw std::runtime_error(
            "Selected UAC MAP start failed during particle EM: "
            + std::string(exception.what()));
    }
    result.traces.push_back(particle.trace);
    if (particle.trace.collapsed) {
        throw std::runtime_error(
            "Selected UAC MAP start collapsed during particle EM");
    }
    result.model = particle.model;
    result.converged = particle.trace.converged;
    return result;
}

ScoreResult score_map(const Dataset& data, const Model& model,
    int32_t n_threads) {
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    ScoreResult out;
    out.responsibilities = map_expectation(data, model, true).responsibilities;
    return out;
}

ScoreResult score_particle(const Dataset& data, const Basis& basis,
    const State& state, ProposalKind proposal, int32_t particles,
    const AdaptiveParticleOptions& adaptive_particles,
    int32_t n_threads, int32_t particle_block_size) {
    if (particle_block_size < 0) {
        throw std::invalid_argument("UAC particle block size cannot be negative");
    }
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    if (adaptive_particles.enabled && particle_block_size != 0) {
        throw std::invalid_argument(
            "Adaptive particles cannot be combined with particle block replay");
    }
    if (adaptive_particles.enabled) {
        AdaptiveParticleOptions configured = adaptive_particles;
        configured.maximum_particles = particles;
        const PilotCache pilot_cache(state.pilot);
        const auto particle_start = std::chrono::steady_clock::now();
        const RaggedParticleSet set = make_adaptive_particles(data, basis,
            state.helmert, state.pilot, pilot_cache, proposal,
            static_cast<uint64_t>(state.seed) ^ 0xF604,
            state.fisher_broadening, n_threads, state.model, configured);
        const double particle_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - particle_start).count();
        const auto score_start = std::chrono::steady_clock::now();
        ScoreResult out = score_particles(set, state.model);
        out.adaptive_particle_options = configured;
        out.particle_generation_seconds = particle_seconds;
        out.scoring_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - score_start).count();
        return out;
    }
    if (particle_block_size > 0) {
        ParticleReplayMetrics metrics;
        const PilotCache pilot_cache(state.pilot);
        const auto score_start = std::chrono::steady_clock::now();
        ScoreResult out = score_particle_replay(data, basis, state.helmert,
            state.pilot, pilot_cache, proposal, particles,
            static_cast<uint64_t>(state.seed) ^ 0xF604,
            state.fisher_broadening, n_threads, particle_block_size,
            state.model, metrics);
        out.scoring_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - score_start).count();
        return out;
    }
    const auto particle_start = std::chrono::steady_clock::now();
    const ParticleSet set = make_particles(data, basis, state.helmert,
        state.pilot, proposal, particles,
        static_cast<uint64_t>(state.seed) ^ 0xF604,
        state.fisher_broadening, n_threads);
    const double particle_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - particle_start).count();
    const auto score_start = std::chrono::steady_clock::now();
    ScoreResult out = score_particles(set, state.model);
    out.particle_generation_seconds = particle_seconds;
    out.scoring_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - score_start).count();
    return out;
}

State make_state(const FitResult& fit_result, const Basis* basis,
    const FitOptions& options, const Eigen::VectorXd& feature_weights,
    bool weighted_counts) {
    State state;
    state.handoff = options.handoff;
    state.proposal = options.proposal;
    state.n_particles = options.n_particles;
    state.seed = options.seed;
    state.cluster_covariance_rank = fit_result.model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(
            fit_result.model.factor_covariances.front().factor.cols())
        : -1;
    state.kmeans_starts = options.kmeans_starts;
    state.leiden_starts = options.leiden_starts;
    state.kmeans_max_iterations = options.kmeans_max_iterations;
    state.leiden_neighbors = options.leiden_neighbors;
    state.leiden_knn_backend = options.leiden_knn_backend;
    state.leiden_max_iterations = options.leiden_max_iterations;
    state.selected_start = fit_result.selected_start;
    state.selected_initializer = fit_result.selected_initializer;
    state.selected_leiden_resolution =
        fit_result.selected_leiden_resolution;
    state.converged = fit_result.converged;
    state.center_floor = options.center_floor;
    state.target_relative_floor = options.target_relative_floor;
    state.leiden_knn_epsilon = options.leiden_knn_epsilon;
    state.leiden_resolution = options.leiden_resolution;
    state.covariance_floor = options.covariance_floor;
    state.objective_change_tolerance = options.objective_change_tolerance;
    state.responsibility_change_tolerance =
        options.responsibility_change_tolerance;
    state.adaptive_covariance_shrinkage =
        options.adaptive_covariance_shrinkage;
    state.fisher_broadening = options.fisher_broadening;
    state.fit_adaptive_particles = options.adaptive_particles;
    state.weighted_counts = weighted_counts;
    state.feature_weights = feature_weights;
    state.pilot = fit_result.pilot;
    state.model = fit_result.model;
    if (basis) {
        state.topics = basis->topics;
        state.basis_checksum = basis->checksum;
    } else {
        state.topics.resize(fit_result.model.means.cols() + 1);
        for (size_t i = 0; i < state.topics.size(); ++i) {
            state.topics[i] = std::to_string(i);
        }
    }
    state.helmert = normalized_helmert(state.topics.size());
    return state;
}

void write_state(const std::string& path, const State& state) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC state: " + path);
    const int32_t components = static_cast<int32_t>(state.model.weights.size());
    const int32_t dimension = static_cast<int32_t>(state.model.means.cols());
    out << "##punkst_uac_state_v4\n"
        << "##initialization_contract\tmixed_map_starts\n"
        << "##handoff\t" << handoff_name(state.handoff) << "\n"
        << "##proposal\t" << proposal_name(state.proposal) << "\n"
        << "##particles\t" << state.n_particles << "\n"
        << "##seed\t" << state.seed << "\n"
        << "##cluster_covariance_rank\t"
        << state.cluster_covariance_rank << "\n"
        << "##kmeans_starts\t" << state.kmeans_starts << "\n"
        << "##leiden_starts\t" << state.leiden_starts << "\n"
        << "##kmeans_max_iterations\t"
        << state.kmeans_max_iterations << "\n"
        << "##leiden_neighbors\t" << state.leiden_neighbors << "\n"
        << "##leiden_knn_backend\t"
        << cosine_knn_backend_name(state.leiden_knn_backend) << "\n"
        << "##leiden_max_iterations\t"
        << state.leiden_max_iterations << "\n"
        << "##selected_start\t" << state.selected_start << "\n"
        << "##selected_initializer\t"
        << initializer_name(state.selected_initializer) << "\n"
        << "##converged\t" << static_cast<int32_t>(state.converged) << "\n"
        << "##components\t" << components << "\n"
        << "##dimension\t" << dimension << "\n"
        << "##basis_checksum\t" << state.basis_checksum << "\n"
        << "##weighted_counts\t" << static_cast<int32_t>(state.weighted_counts) << "\n"
        << "##count_likelihood\t"
        << (state.weighted_counts ? "weighted_multinomial_kernel" : "multinomial")
        << "\n"
        << std::setprecision(17)
        << "##center_floor\t" << state.center_floor << "\n"
        << "##target_relative_floor\t"
        << state.target_relative_floor << "\n"
        << "##leiden_knn_epsilon\t" << state.leiden_knn_epsilon << "\n"
        << "##leiden_resolution\t" << state.leiden_resolution << "\n"
        << "##selected_leiden_resolution\t"
        << state.selected_leiden_resolution << "\n"
        << "##covariance_floor\t" << state.covariance_floor << "\n"
        << "##objective_change_tolerance\t"
        << state.objective_change_tolerance << "\n"
        << "##responsibility_change_tolerance\t"
        << state.responsibility_change_tolerance << "\n"
        << "##covariance_shrinkage\t"
        << (state.adaptive_covariance_shrinkage
            ? "adaptive_particle" : "none") << "\n"
        << "##fisher_broadening\t" << state.fisher_broadening << "\n"
        << "##particle_adapt_mode\t"
        << (state.fit_adaptive_particles.enabled
            ? adaptive_particle_rule_name(state.fit_adaptive_particles.rule)
            : "fixed") << "\n"
        << "##particle_adapt_resp\t"
        << state.fit_adaptive_particles.responsibility_se_target << "\n"
        << "##particle_adapt_moment\t"
        << state.fit_adaptive_particles.moment_ess_target << "\n"
        << "##particle_adapt_calibration\t"
        << state.fit_adaptive_particles.calibration_particles << "\n"
        << "##particle_adapt_min\t"
        << state.fit_adaptive_particles.minimum_particles << "\n"
        << "##particle_adapt_plausible_mass\t"
        << state.fit_adaptive_particles.plausible_mass << "\n"
        << "##particle_adapt_plausible_resp\t"
        << state.fit_adaptive_particles.plausible_responsibility << "\n";
    out << "TOPICS";
    for (const auto& topic : state.topics) out << "\t" << topic;
    out << "\nFEATURE_WEIGHTS";
    for (Eigen::Index i = 0; i < state.feature_weights.size(); ++i) {
        out << "\t" << state.feature_weights(i);
    }
    out << "\nMODEL_WEIGHTS";
    for (Eigen::Index c = 0; c < state.model.weights.size(); ++c) out << "\t" << state.model.weights(c);
    out << "\nPILOT_WEIGHTS";
    for (Eigen::Index c = 0; c < state.pilot.weights.size(); ++c) out << "\t" << state.pilot.weights(c);
    out << "\n";
    for (Eigen::Index r = 0; r < state.helmert.rows(); ++r) {
        out << "HELMERT\t" << r;
        for (Eigen::Index j = 0; j < state.helmert.cols(); ++j) out << "\t" << state.helmert(r, j);
        out << "\n";
    }
    for (int32_t c = 0; c < components; ++c) {
        out << "MODEL_MEAN\t" << c;
        for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.model.means(c, j);
        out << "\nPILOT_MEAN\t" << c;
        for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.pilot.means(c, j);
        out << "\n";
        for (int32_t r = 0; r < dimension; ++r) {
            if (state.cluster_covariance_rank < 0) {
                out << "MODEL_COV\t" << c << "\t" << r;
                for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.model.covariances[c](r, j);
                out << "\n";
            } else {
                out << "MODEL_FACTOR\t" << c << "\t" << r;
                for (int32_t j = 0; j < state.cluster_covariance_rank; ++j) {
                    out << "\t" << state.model.factor_covariances[c].factor(r, j);
                }
                out << "\n";
            }
            out << "PILOT_RAW_COV\t" << c << "\t" << r;
            for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.pilot.raw_covariances[c](r, j);
            out << "\nPILOT_COV\t" << c << "\t" << r;
            for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.pilot.covariances[c](r, j);
            out << "\n";
        }
    }
    for (int32_t r = 0; r < dimension; ++r) {
        if (state.cluster_covariance_rank < 0) {
            out << "SHRINKAGE_TARGET\t" << r;
            for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.model.shrinkage_target(r, j);
        } else {
            out << "FA_DIAGONALS\t" << r;
            for (int32_t c = 0; c < components; ++c) {
                out << "\t" << state.model.factor_covariances[c].diagonal(r);
            }
            out << "\nFA_TARGET\t" << r << "\t"
                << state.model.factor_shrinkage_target.diagonal(r);
            for (int32_t j = 0; j < state.cluster_covariance_rank; ++j) {
                out << "\t" << state.model.factor_shrinkage_target.factor(r, j);
            }
        }
        out << "\nPILOT_POOLED\t" << r;
        for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.pilot.pooled_covariance(r, j);
        out << "\n";
    }
}

State read_state(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot read UAC state: " + path);
    State state;
    std::string line;
    int32_t components = -1, dimension = -1;
    bool saw_version = false, saw_proposal = false;
    bool saw_fisher_broadening = false, saw_initialization_contract = false;
    bool saw_kmeans_starts = false, saw_leiden_starts = false;
    bool saw_selected_start = false, saw_selected_initializer = false;
    bool saw_target_relative_floor = false;
    bool saw_cluster_covariance_rank = false;
    bool saw_objective_change_tolerance = false;
    bool saw_responsibility_change_tolerance = false;
    bool saw_covariance_shrinkage = false;
    bool saw_particle_adapt_mode = false, saw_particle_adapt_resp = false;
    bool saw_particle_adapt_moment = false;
    bool saw_particle_adapt_calibration = false;
    bool saw_particle_adapt_min = false;
    bool saw_particle_adapt_plausible_mass = false;
    bool saw_particle_adapt_plausible_resp = false;
    std::vector<std::vector<std::string>> records;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> token = fields(line);
        if (token.empty()) continue;
        if (token[0] == "##punkst_uac_state_v4") { saw_version = true; continue; }
        if (token[0].rfind("##", 0) == 0) {
            if (token.size() != 2) throw std::runtime_error("Malformed UAC state metadata");
            const std::string key = token[0].substr(2);
            if (key == "initialization_contract") {
                saw_initialization_contract = token[1] == "mixed_map_starts";
            }
            else if (key == "handoff") state.handoff = parse_handoff(token[1]);
            else if (key == "proposal") {
                state.proposal = parse_proposal(token[1]);
                saw_proposal = true;
            }
            else if (key == "particles") state.n_particles = std::stoi(token[1]);
            else if (key == "seed") state.seed = std::stoi(token[1]);
            else if (key == "cluster_covariance_rank") {
                state.cluster_covariance_rank = std::stoi(token[1]);
                saw_cluster_covariance_rank = true;
            }
            else if (key == "kmeans_starts") {
                state.kmeans_starts = std::stoi(token[1]);
                saw_kmeans_starts = true;
            }
            else if (key == "leiden_starts") {
                state.leiden_starts = std::stoi(token[1]);
                saw_leiden_starts = true;
            }
            else if (key == "kmeans_max_iterations") {
                state.kmeans_max_iterations = std::stoi(token[1]);
            }
            else if (key == "leiden_neighbors") {
                state.leiden_neighbors = std::stoi(token[1]);
            }
            else if (key == "leiden_knn_backend") {
                state.leiden_knn_backend = parse_cosine_knn_backend(token[1]);
            }
            else if (key == "leiden_max_iterations") {
                state.leiden_max_iterations = std::stoi(token[1]);
            }
            else if (key == "selected_start") {
                state.selected_start = std::stoi(token[1]);
                saw_selected_start = true;
            }
            else if (key == "selected_initializer") {
                state.selected_initializer = parse_initializer(token[1]);
                saw_selected_initializer = true;
            }
            else if (key == "converged") state.converged = std::stoi(token[1]) != 0;
            else if (key == "components") components = std::stoi(token[1]);
            else if (key == "dimension") dimension = std::stoi(token[1]);
            else if (key == "basis_checksum") state.basis_checksum = std::stoull(token[1]);
            else if (key == "weighted_counts") state.weighted_counts = std::stoi(token[1]) != 0;
            else if (key == "center_floor") state.center_floor = std::stod(token[1]);
            else if (key == "target_relative_floor") {
                state.target_relative_floor = std::stod(token[1]);
                saw_target_relative_floor = true;
            }
            else if (key == "leiden_knn_epsilon") {
                state.leiden_knn_epsilon = std::stod(token[1]);
            }
            else if (key == "leiden_resolution") {
                state.leiden_resolution = std::stod(token[1]);
            }
            else if (key == "selected_leiden_resolution") {
                state.selected_leiden_resolution = std::stod(token[1]);
            }
            else if (key == "covariance_floor") state.covariance_floor = std::stod(token[1]);
            else if (key == "objective_change_tolerance") {
                state.objective_change_tolerance = std::stod(token[1]);
                saw_objective_change_tolerance = true;
            }
            else if (key == "responsibility_change_tolerance") {
                state.responsibility_change_tolerance = std::stod(token[1]);
                saw_responsibility_change_tolerance = true;
            }
            else if (key == "covariance_shrinkage") {
                if (token[1] == "adaptive_particle") {
                    state.adaptive_covariance_shrinkage = true;
                } else if (token[1] == "none") {
                    state.adaptive_covariance_shrinkage = false;
                } else {
                    throw std::runtime_error(
                        "Unsupported UAC covariance shrinkage mode");
                }
                saw_covariance_shrinkage = true;
            }
            else if (key == "fisher_broadening") {
                state.fisher_broadening = std::stod(token[1]);
                saw_fisher_broadening = true;
            }
            else if (key == "particle_adapt_mode") {
                state.fit_adaptive_particles.enabled = token[1] != "fixed";
                if (state.fit_adaptive_particles.enabled) {
                    state.fit_adaptive_particles.rule =
                        parse_adaptive_particle_rule(token[1]);
                }
                saw_particle_adapt_mode = true;
            }
            else if (key == "particle_adapt_resp") {
                state.fit_adaptive_particles.responsibility_se_target =
                    std::stod(token[1]);
                saw_particle_adapt_resp = true;
            }
            else if (key == "particle_adapt_moment") {
                state.fit_adaptive_particles.moment_ess_target =
                    std::stod(token[1]);
                saw_particle_adapt_moment = true;
            }
            else if (key == "particle_adapt_calibration") {
                state.fit_adaptive_particles.calibration_particles =
                    std::stoi(token[1]);
                saw_particle_adapt_calibration = true;
            }
            else if (key == "particle_adapt_min") {
                state.fit_adaptive_particles.minimum_particles =
                    std::stoi(token[1]);
                saw_particle_adapt_min = true;
            }
            else if (key == "particle_adapt_plausible_mass") {
                state.fit_adaptive_particles.plausible_mass =
                    std::stod(token[1]);
                saw_particle_adapt_plausible_mass = true;
            }
            else if (key == "particle_adapt_plausible_resp") {
                state.fit_adaptive_particles.plausible_responsibility =
                    std::stod(token[1]);
                saw_particle_adapt_plausible_resp = true;
            }
            continue;
        }
        records.push_back(std::move(token));
    }
    if (!saw_version || !saw_proposal || !saw_fisher_broadening
        || !saw_initialization_contract || !saw_kmeans_starts
        || !saw_leiden_starts || !saw_selected_start
        || !saw_selected_initializer || !saw_target_relative_floor
        || !saw_cluster_covariance_rank
        || !saw_objective_change_tolerance
        || !saw_responsibility_change_tolerance
        || !saw_covariance_shrinkage
        || !saw_particle_adapt_mode || !saw_particle_adapt_resp
        || !saw_particle_adapt_moment || !saw_particle_adapt_calibration
        || !saw_particle_adapt_min || !saw_particle_adapt_plausible_mass
        || !saw_particle_adapt_plausible_resp
        || components <= 0 || dimension <= 0) {
        throw std::runtime_error("Invalid, stale, or unsupported UAC state");
    }
    state.helmert = Eigen::MatrixXd::Zero(dimension, dimension + 1);
    state.model.weights.resize(components);
    state.model.covariance_kind = state.cluster_covariance_rank < 0
        ? CovarianceKind::Dense : CovarianceKind::FactorAnalytic;
    state.model.means = RowMajorMatrixXd::Zero(components, dimension);
    state.model.covariances.assign(components, Eigen::MatrixXd::Zero(dimension, dimension));
    state.model.shrinkage_target = Eigen::MatrixXd::Zero(dimension, dimension);
    if (state.cluster_covariance_rank >= 0) {
        if (state.cluster_covariance_rank > dimension) {
            throw std::runtime_error("Invalid UAC state factor rank");
        }
        state.model.factor_covariances.resize(components);
        for (auto& covariance : state.model.factor_covariances) {
            covariance.diagonal = Eigen::VectorXd::Zero(dimension);
            covariance.factor = RowMajorMatrixXd::Zero(
                dimension, state.cluster_covariance_rank);
        }
        state.model.factor_shrinkage_target.diagonal =
            Eigen::VectorXd::Zero(dimension);
        state.model.factor_shrinkage_target.factor = RowMajorMatrixXd::Zero(
            dimension, state.cluster_covariance_rank);
    }
    state.pilot.weights.resize(components);
    state.pilot.means = RowMajorMatrixXd::Zero(components, dimension);
    state.pilot.raw_covariances.assign(components, Eigen::MatrixXd::Zero(dimension, dimension));
    state.pilot.covariances.assign(components, Eigen::MatrixXd::Zero(dimension, dimension));
    state.pilot.pooled_covariance = Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& token : records) {
        auto values = [&](size_t offset, Eigen::Ref<Eigen::VectorXd> target) {
            if (token.size() != offset + static_cast<size_t>(target.size())) throw std::runtime_error("Malformed UAC state row");
            for (Eigen::Index j = 0; j < target.size(); ++j) target(j) = std::stod(token[offset + j]);
        };
        if (token[0] == "TOPICS") state.topics.assign(token.begin() + 1, token.end());
        else if (token[0] == "FEATURE_WEIGHTS") {
            state.feature_weights.resize(token.size() - 1);
            for (size_t j = 1; j < token.size(); ++j) state.feature_weights(j - 1) = std::stod(token[j]);
        } else if (token[0] == "MODEL_WEIGHTS") values(1, state.model.weights);
        else if (token[0] == "PILOT_WEIGHTS") values(1, state.pilot.weights);
        else if (token[0] == "HELMERT") {
            const int32_t row = std::stoi(token[1]);
            Eigen::VectorXd target(dimension + 1);
            values(2, target);
            state.helmert.row(row) = target.transpose();
        } else if (token[0] == "MODEL_MEAN" || token[0] == "PILOT_MEAN") {
            const int32_t c = std::stoi(token[1]);
            Eigen::VectorXd target(dimension);
            values(2, target);
            if (token[0] == "MODEL_MEAN") state.model.means.row(c) = target.transpose();
            else state.pilot.means.row(c) = target.transpose();
        } else if (token[0] == "MODEL_FACTOR") {
            const int32_t c = std::stoi(token[1]);
            const int32_t row = std::stoi(token[2]);
            Eigen::VectorXd target(state.cluster_covariance_rank);
            values(3, target);
            state.model.factor_covariances[c].factor.row(row) =
                target.transpose();
        } else if (token[0] == "FA_DIAGONALS") {
            const int32_t row = std::stoi(token[1]);
            Eigen::VectorXd target(components);
            values(2, target);
            for (int32_t c = 0; c < components; ++c) {
                state.model.factor_covariances[c].diagonal(row) = target(c);
            }
        } else if (token[0] == "FA_TARGET") {
            const int32_t row = std::stoi(token[1]);
            Eigen::VectorXd target(state.cluster_covariance_rank + 1);
            values(2, target);
            state.model.factor_shrinkage_target.diagonal(row) = target(0);
            if (state.cluster_covariance_rank > 0) {
                state.model.factor_shrinkage_target.factor.row(row) =
                    target.tail(state.cluster_covariance_rank).transpose();
            }
        } else if (token[0] == "MODEL_COV" || token[0] == "PILOT_RAW_COV" || token[0] == "PILOT_COV") {
            const int32_t c = std::stoi(token[1]);
            const int32_t row = std::stoi(token[2]);
            Eigen::VectorXd target(dimension);
            values(3, target);
            if (token[0] == "MODEL_COV") state.model.covariances[c].row(row) = target.transpose();
            else if (token[0] == "PILOT_RAW_COV") state.pilot.raw_covariances[c].row(row) = target.transpose();
            else state.pilot.covariances[c].row(row) = target.transpose();
        } else if (token[0] == "SHRINKAGE_TARGET" || token[0] == "PILOT_POOLED") {
            const int32_t row = std::stoi(token[1]);
            Eigen::VectorXd target(dimension);
            values(2, target);
            if (token[0] == "SHRINKAGE_TARGET") state.model.shrinkage_target.row(row) = target.transpose();
            else state.pilot.pooled_covariance.row(row) = target.transpose();
        }
    }
    const int64_t total_starts = static_cast<int64_t>(state.kmeans_starts)
        + state.leiden_starts;
    const bool selected_kind_matches = state.selected_initializer
            == MapInitializer::KMeans
        ? state.selected_start < state.kmeans_starts
        : state.selected_start >= state.kmeans_starts;
    bool covariance_valid = true;
    if (state.cluster_covariance_rank >= 0) {
        covariance_valid = state.model.factor_shrinkage_target.diagonal.allFinite()
            && state.model.factor_shrinkage_target.factor.allFinite()
            && (state.model.factor_shrinkage_target.diagonal.array() > 0.0).all();
        for (const auto& covariance : state.model.factor_covariances) {
            covariance_valid = covariance_valid
                && covariance.diagonal.allFinite()
                && covariance.factor.allFinite()
                && (covariance.diagonal.array() > 0.0).all();
        }
    } else {
        covariance_valid = state.model.shrinkage_target.allFinite();
        for (const auto& covariance : state.model.covariances) {
            covariance_valid = covariance_valid && covariance.allFinite()
                && Eigen::LLT<Eigen::MatrixXd>(covariance).info()
                    == Eigen::Success;
        }
    }
    if (state.topics.size() != static_cast<size_t>(dimension + 1)
        || !state.model.weights.allFinite() || !state.model.means.allFinite()
        || (state.model.weights.array() < 0.0).any()
        || !(state.model.weights.sum() > 0.0)
        || std::abs(state.model.weights.sum() - 1.0) > 1e-8
        || state.kmeans_starts < 0 || state.leiden_starts < 0
        || total_starts <= 0 || state.selected_start < 0
        || state.selected_start >= total_starts || !selected_kind_matches
        || !(state.target_relative_floor > 0.0)
        || !(state.objective_change_tolerance > 0.0)
        || !(state.responsibility_change_tolerance > 0.0)
        || !(state.fisher_broadening > 0.0)
        || (state.fit_adaptive_particles.enabled
            && (state.fit_adaptive_particles.calibration_particles < 2
                || state.fit_adaptive_particles.minimum_particles
                    < state.fit_adaptive_particles.calibration_particles
                || state.n_particles
                    < state.fit_adaptive_particles.minimum_particles
                || !(state.fit_adaptive_particles.responsibility_se_target
                    > 0.0)
                || !(state.fit_adaptive_particles.moment_ess_target > 0.0)
                || !(state.fit_adaptive_particles.plausible_mass > 0.0
                    && state.fit_adaptive_particles.plausible_mass <= 1.0)
                || !(state.fit_adaptive_particles.plausible_responsibility
                        >= 0.0
                    && state.fit_adaptive_particles.plausible_responsibility
                        <= 1.0)))
        || !std::isfinite(state.fisher_broadening) || !covariance_valid) {
        throw std::runtime_error("Incomplete UAC state");
    }
    return state;
}

void write_model(const std::string& path, const State& state,
    const Eigen::VectorXd* effective_membership) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC model: " + path);
    out << "#cluster\tactive\tweight\teffective_membership"
        "\tmean_variance\tlog_volume";
    for (const auto& topic : state.topics) out << "\t" << topic;
    out << "\n" << std::scientific << std::setprecision(10);
    RowMajorMatrixXd compositions = ilr_inverse(state.model.means, state.helmert);
    for (Eigen::Index c = 0; c < state.model.weights.size(); ++c) {
        const Eigen::MatrixXd covariance = model_covariance_dense(
            state.model, c);
        Eigen::LLT<Eigen::MatrixXd> llt(covariance);
        const Eigen::MatrixXd lower = llt.matrixL();
        const double log_volume = lower.diagonal().array().log().sum();
        out << c << "\t" << static_cast<int32_t>(
            state.model.weights(c) > 0.0) << "\t"
            << state.model.weights(c) << "\t"
            << (effective_membership ? (*effective_membership)(c) : -1.0)
            << "\t" << covariance.trace() / state.model.means.cols()
            << "\t" << log_volume;
        for (Eigen::Index k = 0; k < compositions.cols(); ++k) out << "\t" << compositions(c, k);
        out << "\n";
    }
}

void write_results(const std::string& path, const Dataset& data,
    const ScoreResult& score) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC results: " + path);
    out << "#id\ttop_cluster\ttop_probability\tsecond_cluster\tsecond_probability\tentropy";
    for (Eigen::Index c = 0; c < score.responsibilities.cols(); ++c) out << "\tcluster_" << c;
    out << "\n" << std::scientific << std::setprecision(10);
    for (Eigen::Index d = 0; d < score.responsibilities.rows(); ++d) {
        std::vector<Eigen::Index> order(score.responsibilities.cols());
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + std::min<size_t>(2, order.size()), order.end(),
            [&](Eigen::Index a, Eigen::Index b) { return score.responsibilities(d, a) > score.responsibilities(d, b); });
        const Eigen::Index first = order[0];
        const Eigen::Index second = order.size() > 1
            && score.responsibilities(d, order[1]) > 0.0
            ? order[1] : order[0];
        out << data.identifiers[d] << "\t" << first << "\t" << score.responsibilities(d, first)
            << "\t" << second << "\t" << score.responsibilities(d, second)
            << "\t" << entropy(score.responsibilities.row(d));
        for (Eigen::Index c = 0; c < score.responsibilities.cols(); ++c) out << "\t" << score.responsibilities(d, c);
        out << "\n";
    }
}

void write_diagnostics(const std::string& path, const Dataset& data,
    const ScoreResult& score) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC diagnostics: " + path);
    out << "##particle_generation_seconds\t"
        << score.particle_generation_seconds << "\n"
        << "##scoring_seconds\t" << score.scoring_seconds << "\n"
        << "##particle_sampling_seconds\t" << score.sampling_seconds << "\n"
        << "##particle_fisher_work_seconds\t"
        << score.fisher_work_seconds << "\n"
        << "##particle_proposal_component_work_seconds\t"
        << score.proposal_component_work_seconds << "\n"
        << "##particle_proposal_draw_density_work_seconds\t"
        << score.proposal_draw_density_work_seconds << "\n"
        << "##particle_likelihood_seconds\t" << score.likelihood_seconds << "\n"
        << "##particle_calibration_seconds\t"
        << score.calibration_seconds << "\n"
        << "##particle_samples\t" << score.particle_samples << "\n"
        << "##particle_calibration_samples\t"
        << score.calibration_samples << "\n"
        << "##particle_reused_calibration_samples\t"
        << score.reused_calibration_samples << "\n"
        << "##particle_adapt_mode\t"
        << (score.adaptive_particle_options.enabled
            ? adaptive_particle_rule_name(score.adaptive_particle_options.rule)
            : "fixed") << "\n"
        << "##particle_adapt_resp\t"
        << score.adaptive_particle_options.responsibility_se_target << "\n"
        << "##particle_adapt_moment\t"
        << score.adaptive_particle_options.moment_ess_target << "\n"
        << "##particle_adapt_calibration\t"
        << score.adaptive_particle_options.calibration_particles << "\n"
        << "##particle_adapt_min\t"
        << score.adaptive_particle_options.minimum_particles << "\n"
        << "##particle_adapt_plausible_mass\t"
        << score.adaptive_particle_options.plausible_mass << "\n"
        << "##particle_adapt_plausible_resp\t"
        << score.adaptive_particle_options.plausible_responsibility << "\n"
        << "##particle_estep_gaussian_seconds\t"
        << score.gaussian_seconds << "\n"
        << "##particle_estep_moment_seconds\t"
        << score.moment_seconds << "\n"
        << "##particle_bytes\t" << score.particle_bytes << "\n"
        << "##proposal_workspace_bytes\t"
        << score.proposal_workspace_bytes << "\n"
        << "##expectation_accumulator_bytes\t"
        << score.expectation_accumulator_bytes << "\n"
        << "##particle_replay\t" << static_cast<int32_t>(
            score.particle_replay) << "\n"
        << "##particle_block_size\t" << score.particle_block_size << "\n"
        << "##particle_generation_passes\t"
        << score.particle_generation_passes << "\n"
        << "#id\traw_total\teffective_total\tparticles\trelative_ess\tmaximum_weight\tlog_likelihood_range\tlog_proposal_range\thpd80_log_density_threshold\thpd95_log_density_threshold"
        << "\tadapt_preliminary_max_resp\tadapt_preliminary_entropy"
        << "\tadapt_plausible_components\tadapt_max_resp_se"
        << "\tadapt_projected_resp_particles\tadapt_projected_moment_particles"
        << "\tadapt_plausible_max_weight\tadapt_half_max_resp_difference"
        << "\tadapt_half_top_disagreement\tadapt_binding\n"
        << std::scientific << std::setprecision(10);
    for (size_t d = 0; d < data.identifiers.size(); ++d) {
        const bool particle = d < score.particle_diagnostics.size();
        out << data.identifiers[d] << "\t"
            << (data.raw_totals.size() ? data.raw_totals(d) : 0.0) << "\t"
            << (data.effective_totals.size() ? data.effective_totals(d) : 0.0)
            << "\t" << (d < score.per_document_particles.size()
                ? score.per_document_particles[d] : 0) << "\t";
        if (particle) {
            const auto& value = score.particle_diagnostics[d];
            out << value.relative_ess << "\t" << value.maximum_weight << "\t"
                << value.log_likelihood_range << "\t"
                << value.log_proposal_range << "\t"
                << value.hpd80_log_density_threshold << "\t"
                << value.hpd95_log_density_threshold;
        } else {
            out << "NA\tNA\tNA\tNA\tNA\tNA";
        }
        out << "\t";
        if (d < score.adaptive_particle_diagnostics.size()) {
            const auto& value = score.adaptive_particle_diagnostics[d];
            out << value.preliminary_maximum_responsibility << "\t"
                << value.preliminary_entropy << "\t"
                << value.plausible_components << "\t"
                << value.maximum_responsibility_se << "\t"
                << value.projected_responsibility_particles << "\t"
                << value.projected_moment_particles << "\t"
                << value.plausible_maximum_weight << "\t"
                << value.half_sample_maximum_responsibility_difference
                << "\t" << static_cast<int32_t>(
                    value.half_sample_top_disagreement) << "\t"
                << adaptive_particle_binding_name(value.binding);
        } else {
            out << "NA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA";
        }
        out << "\n";
    }
}

void write_trace(const std::string& path,
    const std::vector<RestartTrace>& traces) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC trace: " + path);
    out << "#handoff\tstart\tinitializer\tseed\traw_communities"
        "\treconciliation_count\tleiden_resolution\tselection_objective"
        "\titeration\tobjective\trelative_objective_change"
        "\tmean_max_responsibility_change"
        "\tactive_components\tconverged\tcollapsed\n"
        << std::scientific << std::setprecision(12);
    for (const auto& trace : traces) {
        auto write_metadata = [&]() {
            out << (trace.particle ? "particle" : "map") << "\t"
                << trace.start << "\t" << initializer_name(trace.initializer)
                << "\t" << trace.seed << "\t" << trace.raw_communities
                << "\t" << trace.reconciliation_count << "\t";
            if (trace.initializer == MapInitializer::Leiden) {
                out << trace.leiden_resolution;
            } else {
                out << "NA";
            }
            out << "\t";
            if (std::isfinite(trace.selection_objective)) {
                out << trace.selection_objective;
            } else {
                out << "NA";
            }
        };
        if (trace.objective.empty()) {
            write_metadata();
            out << "\t-1\tNA\tNA\tNA\t-1\t"
                << static_cast<int32_t>(trace.converged) << "\t"
                << static_cast<int32_t>(trace.collapsed) << "\n";
            continue;
        }
        for (size_t iteration = 0; iteration < trace.objective.size(); ++iteration) {
            write_metadata();
            out << "\t" << iteration << "\t" << trace.objective[iteration]
                << "\t";
            if (iteration < trace.relative_objective_change.size()
                && std::isfinite(
                    trace.relative_objective_change[iteration])) {
                out << trace.relative_objective_change[iteration];
            } else {
                out << "NA";
            }
            out << "\t";
            if (iteration < trace.mean_max_responsibility_change.size()
                && std::isfinite(
                    trace.mean_max_responsibility_change[iteration])) {
                out << trace.mean_max_responsibility_change[iteration];
            } else {
                out << "NA";
            }
            out << "\t" << (iteration < trace.active_components.size()
                    ? trace.active_components[iteration] : -1)
                << "\t" << static_cast<int32_t>(trace.converged)
                << "\t" << static_cast<int32_t>(trace.collapsed) << "\n";
        }
    }
}

void write_separation(const std::string& path, const Model& model) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC separation: " + path);
    out << "#cluster_a\tcluster_b\tstandardized_separation\tbhattacharyya_distance\n"
        << std::scientific << std::setprecision(10);
    for (Eigen::Index a = 0; a < model.weights.size(); ++a) {
        if (!(model.weights(a) > 0.0)) continue;
        for (Eigen::Index b = a + 1; b < model.weights.size(); ++b) {
            if (!(model.weights(b) > 0.0)) continue;
            const Eigen::VectorXd difference = model.means.row(a).transpose() - model.means.row(b).transpose();
            const Eigen::MatrixXd covariance = 0.5
                * (model_covariance_dense(model, a)
                    + model_covariance_dense(model, b));
            Eigen::LLT<Eigen::MatrixXd> llt(covariance);
            const double standardized = std::sqrt(std::max(0.0, difference.dot(llt.solve(difference))));
            const double logdet_mean = 2.0 * Eigen::MatrixXd(llt.matrixL()).diagonal().array().log().sum();
            Eigen::LLT<Eigen::MatrixXd> llt_a(
                model_covariance_dense(model, a)), llt_b(
                model_covariance_dense(model, b));
            const double logdet_a = 2.0 * Eigen::MatrixXd(llt_a.matrixL()).diagonal().array().log().sum();
            const double logdet_b = 2.0 * Eigen::MatrixXd(llt_b.matrixL()).diagonal().array().log().sum();
            const double bhattacharyya = 0.125 * standardized * standardized
                + 0.5 * (logdet_mean - 0.5 * (logdet_a + logdet_b));
            out << a << "\t" << b << "\t" << standardized << "\t" << bhattacharyya << "\n";
        }
    }
}

void write_representatives(const std::string& path, const Dataset& data,
    const ScoreResult& score, int32_t n_representatives) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC representatives: " + path);
    out << "#cluster\trank\tid\tprobability\ttop_probability\tentropy\n"
        << std::scientific << std::setprecision(10);
    std::vector<Eigen::Index> order(score.responsibilities.rows());
    for (Eigen::Index c = 0; c < score.responsibilities.cols(); ++c) {
        if (!(score.responsibilities.col(c).sum() > 0.0)) continue;
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + std::min<int32_t>(n_representatives, order.size()), order.end(),
            [&](Eigen::Index a, Eigen::Index b) { return score.responsibilities(a, c) > score.responsibilities(b, c); });
        const int32_t take = std::min<int32_t>(n_representatives, order.size());
        for (int32_t rank = 0; rank < take; ++rank) {
            const Eigen::Index d = order[rank];
            out << c << "\t" << rank + 1 << "\t" << data.identifiers[d]
                << "\t" << score.responsibilities(d, c)
                << "\t" << score.responsibilities.row(d).maxCoeff()
                << "\t" << entropy(score.responsibilities.row(d)) << "\n";
        }
    }
}

} // namespace uac
