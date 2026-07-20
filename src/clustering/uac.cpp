#include "clustering/uac.hpp"

#include "clustering_core/cosine_clustering.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace uac {
namespace {

constexpr double kLog2Pi = 1.83787706640934548356;

double logsumexp(const Eigen::Ref<const Eigen::VectorXd>& values) {
    const double maximum = values.maxCoeff();
    if (!std::isfinite(maximum)) return maximum;
    return maximum + std::log((values.array() - maximum).exp().sum());
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

double covariance_prior(const Model& model, double strength) {
    double out = 0.0;
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

Eigen::VectorXd composition_from_coordinate(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    Eigen::VectorXd logits = helmert.transpose() * coordinate;
    logits.array() -= logits.maxCoeff();
    Eigen::VectorXd values = logits.array().exp();
    values /= values.sum();
    return values;
}

double count_log_likelihood(const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    const Eigen::VectorXd composition = composition_from_coordinate(
        coordinate, helmert);
    const Eigen::VectorXd probability = basis.probabilities * composition;
    double out = 0.0;
    for (size_t j = 0; j < document.ids.size(); ++j) {
        const uint32_t feature = document.ids[j];
        if (feature >= static_cast<uint32_t>(probability.size())) {
            throw std::runtime_error("UAC document feature index is out of range");
        }
        const double p = probability(feature);
        if (!(p > 0.0) || !std::isfinite(p)) {
            return -std::numeric_limits<double>::infinity();
        }
        out += document.cnts[j] * std::log(p);
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

struct DocumentProposal {
    Eigen::VectorXd weights;
    std::vector<Eigen::VectorXd> means;
    std::vector<Eigen::MatrixXd> covariances;
};

DocumentProposal fisher_proposal(
    const Eigen::Ref<const Eigen::VectorXd>& center, const Document& document,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, ProposalKind proposal_kind, double broadening) {
    if (!(broadening > 0.0) || !std::isfinite(broadening)) {
        throw std::invalid_argument("Invalid UAC Fisher broadening");
    }
    const int32_t dimension = static_cast<int32_t>(center.size());
    const FisherApproximation fisher = fisher_approximation_impl(center,
        document, basis, helmert, proposal_kind);
    DocumentProposal out;
    out.weights.resize(pilot.weights.size());
    out.means.reserve(pilot.weights.size());
    out.covariances.reserve(pilot.weights.size());
    Eigen::VectorXd log_weight(pilot.weights.size());
    for (Eigen::Index c = 0; c < pilot.weights.size(); ++c) {
        Eigen::LLT<Eigen::MatrixXd> pilot_llt(pilot.covariances[c]);
        if (pilot_llt.info() != Eigen::Success) {
            throw std::runtime_error(
                "UAC Fisher pilot covariance is not positive definite");
        }
        const Eigen::MatrixXd inverse_covariance = pilot_llt.solve(
            Eigen::MatrixXd::Identity(dimension, dimension));
        const Eigen::MatrixXd pilot_lower = pilot_llt.matrixL();
        const double pilot_logdet = 2.0
            * pilot_lower.diagonal().array().log().sum();
        const Eigen::VectorXd pilot_mean = pilot.means.row(c).transpose();
        const Eigen::VectorXd residual = center - pilot_mean;
        const Eigen::VectorXd b = fisher.gradient
            - inverse_covariance * residual;
        const Eigen::MatrixXd precision = floor_covariance(
            fisher.information + inverse_covariance, 1e-8);
        Eigen::LLT<Eigen::MatrixXd> precision_llt(precision);
        if (precision_llt.info() != Eigen::Success) {
            throw std::runtime_error(
                "UAC Fisher precision is not positive definite");
        }
        const Eigen::VectorXd step = precision_llt.solve(b);
        const Eigen::MatrixXd local_covariance = floor_covariance(
            precision_llt.solve(
                Eigen::MatrixXd::Identity(dimension, dimension)),
            1e-8);
        const Eigen::MatrixXd precision_lower = precision_llt.matrixL();
        const double precision_logdet = 2.0
            * precision_lower.diagonal().array().log().sum();
        out.means.push_back(center + step);
        out.covariances.push_back(broadening * local_covariance);
        log_weight(c) = std::log(pilot.weights(c))
            - 0.5 * (dimension * kLog2Pi + pilot_logdet
                + residual.dot(inverse_covariance * residual))
            + 0.5 * b.dot(step) + 0.5 * dimension * kLog2Pi
            - 0.5 * precision_logdet;
    }
    out.weights = (log_weight.array() - logsumexp(log_weight)).exp();
    return out;
}

double proposal_log_density(const Eigen::Ref<const Eigen::VectorXd>& value,
    const DocumentProposal& proposal) {
    Eigen::VectorXd terms(proposal.weights.size());
    for (Eigen::Index j = 0; j < proposal.weights.size(); ++j) {
        terms(j) = std::log(proposal.weights(j)) + log_gaussian(value,
            proposal.means[j], proposal.covariances[j]);
    }
    return logsumexp(terms);
}

struct Expectation {
    RowMajorMatrixXd responsibilities;
    Eigen::VectorXd membership;
    RowMajorMatrixXd first;
    std::vector<Eigen::MatrixXd> second;
    Eigen::VectorXd per_document;
    double log_likelihood = 0.0;
};

struct ExpectationBlock {
    Eigen::VectorXd membership;
    RowMajorMatrixXd first;
    std::vector<Eigen::MatrixXd> second;
    double log_likelihood = 0.0;

    ExpectationBlock(int32_t components, int32_t dimension)
        : membership(Eigen::VectorXd::Zero(components)),
          first(RowMajorMatrixXd::Zero(components, dimension)),
          second(components, Eigen::MatrixXd::Zero(dimension, dimension)) {}
};

void reduce_expectation_blocks(Expectation& out,
    const std::vector<ExpectationBlock>& blocks) {
    for (const auto& block : blocks) {
        out.membership += block.membership;
        out.first += block.first;
        out.log_likelihood += block.log_likelihood;
        for (size_t c = 0; c < out.second.size(); ++c) {
            out.second[c] += block.second[c];
        }
    }
}

Expectation map_expectation(const Dataset& data, const Model& model) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    Expectation out;
    out.responsibilities.resize(documents, components);
    out.membership = Eigen::VectorXd::Zero(components);
    out.first = RowMajorMatrixXd::Zero(components, dimension);
    out.second.assign(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    out.per_document.resize(documents);
    constexpr int32_t block_size = 128;
    const int32_t n_blocks = (documents + block_size - 1) / block_size;
    std::vector<ExpectationBlock> blocks;
    blocks.reserve(n_blocks);
    for (int32_t block = 0; block < n_blocks; ++block) {
        blocks.emplace_back(components, dimension);
    }
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        ExpectationBlock& block = blocks[block_index];
        Eigen::VectorXd score(components);
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        for (int32_t d = begin; d < end; ++d) {
            const Eigen::VectorXd value = data.coordinates.row(d).transpose();
            for (int32_t c = 0; c < components; ++c) {
                score(c) = std::log(model.weights(c)) + log_gaussian(value,
                    model.means.row(c).transpose(), model.covariances[c]);
            }
            const double normalizer = logsumexp(score);
            out.per_document(d) = normalizer;
            block.log_likelihood += normalizer;
            out.responsibilities.row(d) =
                (score.array() - normalizer).exp().transpose();
            for (int32_t c = 0; c < components; ++c) {
                const double weight = out.responsibilities(d, c);
                block.membership(c) += weight;
                block.first.row(c) += weight * value.transpose();
                block.second[c].noalias() += weight * value * value.transpose();
            }
        }
    });
    reduce_expectation_blocks(out, blocks);
    return out;
}

Expectation particle_expectation(const ParticleSet& particles,
    const Model& model) {
    const int32_t documents = particles.documents;
    const int32_t samples = particles.samples;
    const int32_t dimension = particles.dimension;
    const int32_t components = static_cast<int32_t>(model.weights.size());
    Expectation out;
    out.responsibilities.resize(documents, components);
    out.membership = Eigen::VectorXd::Zero(components);
    out.first = RowMajorMatrixXd::Zero(components, dimension);
    out.second.assign(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    out.per_document.resize(documents);
    constexpr int32_t block_size = 64;
    const int32_t n_blocks = (documents + block_size - 1) / block_size;
    std::vector<ExpectationBlock> blocks;
    blocks.reserve(n_blocks);
    for (int32_t block = 0; block < n_blocks; ++block) {
        blocks.emplace_back(components, dimension);
    }
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        ExpectationBlock& block = blocks[block_index];
        Eigen::MatrixXd log_tilt(components, samples);
        Eigen::VectorXd evidence(components), score(components);
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        for (int32_t d = begin; d < end; ++d) {
            for (int32_t c = 0; c < components; ++c) {
                for (int32_t s = 0; s < samples; ++s) {
                    log_tilt(c, s) = particles.log_likelihood(d, s)
                        - particles.log_q(d, s) - std::log(samples)
                        + log_gaussian(particles.value(d, s),
                            model.means.row(c).transpose(), model.covariances[c]);
                }
                evidence(c) = logsumexp(log_tilt.row(c).transpose());
                score(c) = std::log(model.weights(c)) + evidence(c);
            }
            const double normalizer = logsumexp(score);
            out.per_document(d) = normalizer;
            block.log_likelihood += normalizer;
            out.responsibilities.row(d) =
                (score.array() - normalizer).exp().transpose();
            for (int32_t c = 0; c < components; ++c) {
                const double responsibility = out.responsibilities(d, c);
                block.membership(c) += responsibility;
                const Eigen::VectorXd tau =
                    (log_tilt.row(c).transpose().array() - evidence(c)).exp();
                Eigen::VectorXd first = Eigen::VectorXd::Zero(dimension);
                Eigen::MatrixXd second = Eigen::MatrixXd::Zero(dimension, dimension);
                for (int32_t s = 0; s < samples; ++s) {
                    const Eigen::VectorXd value = particles.value(d, s);
                    first.noalias() += tau(s) * value;
                    second.noalias() += tau(s) * value * value.transpose();
                }
                block.first.row(c) += responsibility * first.transpose();
                block.second[c].noalias() += responsibility * second;
            }
        }
    });
    reduce_expectation_blocks(out, blocks);
    return out;
}

Model initialize_model(const Dataset& data, const Pilot& pilot,
    int32_t components, int32_t seed, int32_t kmeans_iterations,
    double covariance_floor) {
    DenseKMeansOptions options;
    options.n_clusters = components;
    options.max_iterations = kmeans_iterations;
    options.seed = seed;
    const DenseKMeansResult clustering = cosine_dense_kmeans(data.centers, options);
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    Model model;
    model.weights = clustering.counts.cast<double>();
    model.weights /= model.weights.sum();
    model.means = RowMajorMatrixXd::Zero(components, dimension);
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        model.means.row(clustering.assignments(d)) += data.coordinates.row(d);
    }
    for (int32_t c = 0; c < components; ++c) {
        model.means.row(c) /= clustering.counts(c);
    }
    Eigen::VectorXd global_mean = data.coordinates.colwise().mean();
    Eigen::MatrixXd global_covariance = Eigen::MatrixXd::Zero(
        dimension, dimension);
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        const Eigen::VectorXd residual = data.coordinates.row(d).transpose()
            - global_mean;
        global_covariance.noalias() += residual * residual.transpose();
    }
    global_covariance /= data.coordinates.rows();
    global_covariance = floor_covariance(global_covariance, covariance_floor);
    model.covariances.assign(components, global_covariance);
    model.shrinkage_target = pilot.pooled_covariance;
    return model;
}

void update_model(Model& model, const Expectation& expectation,
    double shrinkage, double covariance_floor) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t documents = static_cast<int32_t>(expectation.responsibilities.rows());
    model.weights = expectation.membership / documents;
    for (int32_t c = 0; c < components; ++c) {
        model.means.row(c) = expectation.first.row(c)
            / expectation.membership(c);
        const Eigen::VectorXd mean = model.means.row(c).transpose();
        Eigen::MatrixXd scatter = expectation.second[c]
            - expectation.membership(c) * mean * mean.transpose();
        scatter = 0.5 * (scatter + scatter.transpose());
        model.covariances[c] = floor_covariance(
            (scatter + shrinkage * model.shrinkage_target)
                / (expectation.membership(c) + shrinkage),
            covariance_floor);
    }
}

struct Candidate {
    Model model;
    RowMajorMatrixXd responsibilities;
    RestartTrace trace;
    double objective = -std::numeric_limits<double>::infinity();
};

Candidate fit_map_candidate(const Dataset& data, const Pilot& pilot,
    const FitOptions& options, int32_t restart) {
    Candidate out;
    out.trace.restart = restart;
    out.trace.particle = false;
    out.model = initialize_model(data, pilot, options.n_components,
        options.seed + 104729 * restart, options.kmeans_max_iterations,
        options.covariance_floor);
    for (int32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        Expectation expectation = map_expectation(data, out.model);
        const double objective = expectation.log_likelihood
            + covariance_prior(out.model, options.covariance_shrinkage);
        out.trace.objective.push_back(objective);
        const double collapse_threshold = data.coordinates.cols() + 1.0;
        bool repaired = false;
        for (int32_t c = 0; c < options.n_components; ++c) {
            if (expectation.membership(c) > collapse_threshold) continue;
            Eigen::Index worst = 0;
            expectation.per_document.minCoeff(&worst);
            out.model.means.row(c) = data.coordinates.row(worst);
            out.model.covariances[c] = out.model.shrinkage_target;
            out.model.weights(c) = 1.0 / data.coordinates.rows();
            out.model.weights /= out.model.weights.sum();
            repaired = true;
        }
        if (repaired) continue;
        update_model(out.model, expectation, options.covariance_shrinkage,
            options.covariance_floor);
        if (out.trace.objective.size() >= 2) {
            const double previous = out.trace.objective[
                out.trace.objective.size() - 2];
            if (std::abs(objective - previous)
                / std::max(1.0, std::abs(previous)) < options.tolerance) {
                out.trace.converged = true;
                break;
            }
        }
    }
    Expectation final = map_expectation(data, out.model);
    out.objective = final.log_likelihood
        + covariance_prior(out.model, options.covariance_shrinkage);
    out.trace.objective.push_back(out.objective);
    out.responsibilities = std::move(final.responsibilities);
    return out;
}

Candidate fit_particle_candidate(const ParticleSet& particles,
    Model initial, const FitOptions& options, int32_t restart) {
    Candidate out;
    out.trace.restart = restart;
    out.trace.particle = true;
    out.model = std::move(initial);
    for (int32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        Expectation expectation = particle_expectation(particles, out.model);
        const double objective = expectation.log_likelihood
            + covariance_prior(out.model, options.covariance_shrinkage);
        out.trace.objective.push_back(objective);
        if ((expectation.membership.array()
            <= particles.dimension + 1.0).any()) {
            out.trace.collapsed = true;
            return out;
        }
        update_model(out.model, expectation, options.covariance_shrinkage,
            options.covariance_floor);
        if (out.trace.objective.size() >= 2) {
            const double previous = out.trace.objective[
                out.trace.objective.size() - 2];
            if (std::abs(objective - previous)
                / std::max(1.0, std::abs(previous)) < options.tolerance) {
                out.trace.converged = true;
                break;
            }
        }
    }
    Expectation final = particle_expectation(particles, out.model);
    out.objective = final.log_likelihood
        + covariance_prior(out.model, options.covariance_shrinkage);
    out.trace.objective.push_back(out.objective);
    out.responsibilities = std::move(final.responsibilities);
    return out;
}

ScoreResult score_particles(const ParticleSet& particles, const Model& model) {
    ScoreResult out;
    Expectation expectation = particle_expectation(particles, model);
    out.responsibilities = std::move(expectation.responsibilities);
    out.particle_diagnostics.resize(particles.documents);
    const int32_t components = static_cast<int32_t>(model.weights.size());
    Eigen::VectorXd log_weight(particles.samples);
    Eigen::VectorXd component(components);
    for (int32_t d = 0; d < particles.documents; ++d) {
        double ll_min = std::numeric_limits<double>::infinity();
        double ll_max = -std::numeric_limits<double>::infinity();
        double lq_min = std::numeric_limits<double>::infinity();
        double lq_max = -std::numeric_limits<double>::infinity();
        for (int32_t s = 0; s < particles.samples; ++s) {
            for (int32_t c = 0; c < components; ++c) {
                component(c) = std::log(model.weights(c))
                    + log_gaussian(particles.value(d, s),
                        model.means.row(c).transpose(), model.covariances[c]);
            }
            log_weight(s) = particles.log_likelihood(d, s)
                - particles.log_q(d, s) + logsumexp(component);
            ll_min = std::min(ll_min, particles.log_likelihood(d, s));
            ll_max = std::max(ll_max, particles.log_likelihood(d, s));
            lq_min = std::min(lq_min, particles.log_q(d, s));
            lq_max = std::max(lq_max, particles.log_q(d, s));
        }
        const double normalizer = logsumexp(log_weight);
        const Eigen::VectorXd probability =
            (log_weight.array() - normalizer).exp();
        auto& diagnostic = out.particle_diagnostics[d];
        diagnostic.relative_ess = 1.0
            / (particles.samples * probability.squaredNorm());
        diagnostic.maximum_weight = probability.maxCoeff();
        diagnostic.log_likelihood_range = ll_max - ll_min;
        diagnostic.log_proposal_range = lq_max - lq_min;
    }
    out.sampling_seconds = particles.sampling_seconds;
    out.likelihood_seconds = particles.likelihood_seconds;
    out.particle_bytes = sizeof(double)
        * static_cast<uint64_t>(particles.documents) * particles.samples
        * (particles.dimension + 2);
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

Pilot make_pilot(const Dataset& data, int32_t n_components, int32_t seed,
    int32_t max_iterations, double rho, double relative_floor) {
    if (data.centers.rows() <= n_components || n_components <= 0
        || data.coordinates.rows() != data.centers.rows()
        || !(rho >= 0.0 && rho <= 1.0) || !(relative_floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC pilot input");
    }
    DenseKMeansOptions options;
    options.n_clusters = n_components;
    options.max_iterations = max_iterations;
    options.seed = seed;
    const DenseKMeansResult clustering = cosine_dense_kmeans(data.centers, options);
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    Pilot out;
    out.weights = clustering.counts.cast<double>();
    out.means = RowMajorMatrixXd::Zero(n_components, dimension);
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        out.means.row(clustering.assignments(d)) += data.coordinates.row(d);
    }
    for (int32_t c = 0; c < n_components; ++c) {
        if (clustering.counts(c) <= dimension) {
            throw std::runtime_error("UAC pilot component is too small for covariance estimation");
        }
        out.means.row(c) /= clustering.counts(c);
    }
    out.raw_covariances.assign(n_components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        const int32_t c = clustering.assignments(d);
        const Eigen::VectorXd residual = data.coordinates.row(d).transpose()
            - out.means.row(c).transpose();
        out.raw_covariances[c].noalias() += residual * residual.transpose();
    }
    out.pooled_covariance = Eigen::MatrixXd::Zero(dimension, dimension);
    for (int32_t c = 0; c < n_components; ++c) {
        out.raw_covariances[c] /= clustering.counts(c);
        out.pooled_covariance.noalias() += clustering.counts(c)
            * out.raw_covariances[c];
    }
    out.pooled_covariance /= data.coordinates.rows();
    const double floor = relative_floor * out.pooled_covariance.trace()
        / dimension;
    out.covariances.reserve(n_components);
    for (int32_t c = 0; c < n_components; ++c) {
        out.covariances.push_back(floor_covariance(
            (1.0 - rho) * out.raw_covariances[c]
                + rho * out.pooled_covariance,
            std::max(floor, 1e-12)));
    }
    out.pooled_covariance = floor_covariance(out.pooled_covariance,
        std::max(floor, 1e-12));
    out.weights /= out.weights.sum();
    return out;
}

ParticleSet make_particles(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    ProposalKind proposal_kind, int32_t samples, uint64_t seed,
    double fisher_broadening, int32_t n_threads) {
    if (samples <= 0 || data.counts.size() != data.identifiers.size()
        || data.coordinates.rows() != static_cast<Eigen::Index>(data.counts.size())
        || basis.probabilities.cols() != helmert.cols()
        || !(fisher_broadening > 0.0)
        || !std::isfinite(fisher_broadening)) {
        throw std::invalid_argument("Invalid UAC particle input");
    }
    ParticleSet out;
    out.documents = static_cast<int32_t>(data.counts.size());
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
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    const auto sampling_start = std::chrono::steady_clock::now();
    tbb::parallel_for(int32_t{0}, out.documents, [&](int32_t document) {
        const Eigen::VectorXd center =
            data.coordinates.row(document).transpose();
        const DocumentProposal proposal = fisher_proposal(center,
            data.counts[document], basis, helmert, pilot, proposal_kind,
            fisher_broadening);
        const uint64_t document_seed = hash_string(
            seed ^ 0x9e3779b97f4a7c15ull, data.identifiers[document]);
        std::mt19937_64 engine(document_seed);
        std::discrete_distribution<int32_t> choose(proposal.weights.data(),
            proposal.weights.data() + proposal.weights.size());
        std::normal_distribution<double> normal(0.0, 1.0);
        std::vector<Eigen::MatrixXd> cholesky;
        cholesky.reserve(proposal.covariances.size());
        for (const auto& covariance : proposal.covariances) {
            Eigen::LLT<Eigen::MatrixXd> llt(covariance);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error(
                    "UAC proposal covariance is not positive definite");
            }
            cholesky.push_back(llt.matrixL());
        }
        for (int32_t sample = 0; sample < samples; ++sample) {
            const int32_t component = choose(engine);
            Eigen::VectorXd draw(out.dimension);
            for (int32_t dim = 0; dim < out.dimension; ++dim) {
                draw(dim) = normal(engine);
            }
            const Eigen::VectorXd value = proposal.means[component]
                + cholesky[component] * draw;
            out.values.row(
                static_cast<Eigen::Index>(document) * samples + sample)
                = value.transpose();
            out.log_proposal(document, sample) =
                proposal_log_density(value, proposal);
        }
    });
    out.sampling_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sampling_start).count();
    const auto likelihood_start = std::chrono::steady_clock::now();
    tbb::parallel_for(int32_t{0}, out.documents, [&](int32_t document) {
        for (int32_t sample = 0; sample < samples; ++sample) {
            out.log_likelihood(document, sample) = count_log_likelihood(
                out.value(document, sample), data.counts[document],
                basis, helmert);
        }
    });
    out.likelihood_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - likelihood_start).count();
    return out;
}

FitResult fit(const Dataset& data, const Basis* basis,
    const FitOptions& options) {
    if (options.n_components <= 0 || options.restarts <= 0
        || options.max_iterations <= 0 || options.n_particles <= 0
        || data.centers.rows() < options.n_components
        || data.coordinates.rows() != data.centers.rows()) {
        throw std::invalid_argument("Invalid UAC fit options or dataset");
    }
    if (options.handoff == HandoffMode::Particle
        && (basis == nullptr || data.counts.size() != data.identifiers.size())) {
        throw std::invalid_argument("Particle UAC requires basis and aligned counts");
    }
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, options.n_threads));
    FitResult result;
    result.pilot = make_pilot(data, options.n_components, options.seed ^ 0xE503,
        options.kmeans_max_iterations, options.pilot_rho,
        options.pilot_relative_floor);
    std::vector<Candidate> maps;
    maps.reserve(options.restarts);
    for (int32_t restart = 0; restart < options.restarts; ++restart) {
        maps.push_back(fit_map_candidate(data, result.pilot, options, restart));
        result.traces.push_back(maps.back().trace);
    }
    if (options.handoff == HandoffMode::Map) {
        auto selected = std::max_element(maps.begin(), maps.end(),
            [](const Candidate& first, const Candidate& second) {
                return first.objective < second.objective;
            });
        result.model = selected->model;
        result.score.responsibilities = selected->responsibilities;
        result.selected_restart = selected->trace.restart;
        result.converged = selected->trace.converged;
        return result;
    }
    const Eigen::MatrixXd helmert = normalized_helmert(data.centers.cols());
    const auto particle_start = std::chrono::steady_clock::now();
    const ParticleSet particles = make_particles(data, *basis, helmert,
        result.pilot, options.proposal, options.n_particles,
        static_cast<uint64_t>(options.seed) ^ 0xF604,
        options.fisher_broadening, options.n_threads);
    const double particle_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - particle_start).count();
    std::vector<Candidate> candidates;
    candidates.reserve(options.restarts);
    for (int32_t restart = 0; restart < options.restarts; ++restart) {
        Candidate candidate = fit_particle_candidate(particles,
            maps[restart].model, options, restart);
        result.traces.push_back(candidate.trace);
        if (!candidate.trace.collapsed) candidates.push_back(std::move(candidate));
    }
    if (candidates.empty()) {
        throw std::runtime_error("Every UAC particle restart collapsed");
    }
    auto selected = std::max_element(candidates.begin(), candidates.end(),
        [](const Candidate& first, const Candidate& second) {
            return first.objective < second.objective;
        });
    result.model = selected->model;
    const auto score_start = std::chrono::steady_clock::now();
    result.score = score_particles(particles, result.model);
    result.score.particle_generation_seconds = particle_seconds;
    result.score.scoring_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - score_start).count();
    result.selected_restart = selected->trace.restart;
    result.converged = selected->trace.converged;
    return result;
}

ScoreResult score_map(const Dataset& data, const Model& model,
    int32_t n_threads) {
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    ScoreResult out;
    out.responsibilities = map_expectation(data, model).responsibilities;
    return out;
}

ScoreResult score_particle(const Dataset& data, const Basis& basis,
    const State& state, ProposalKind proposal, int32_t particles,
    int32_t n_threads) {
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
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
    state.selected_restart = fit_result.selected_restart;
    state.converged = fit_result.converged;
    state.center_floor = options.center_floor;
    state.pilot_rho = options.pilot_rho;
    state.pilot_relative_floor = options.pilot_relative_floor;
    state.covariance_floor = options.covariance_floor;
    state.covariance_shrinkage = options.covariance_shrinkage;
    state.fisher_broadening = options.fisher_broadening;
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
    out << "##punkst_uac_state_v1\n"
        << "##handoff\t" << handoff_name(state.handoff) << "\n"
        << "##proposal\t" << proposal_name(state.proposal) << "\n"
        << "##particles\t" << state.n_particles << "\n"
        << "##seed\t" << state.seed << "\n"
        << "##selected_restart\t" << state.selected_restart << "\n"
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
        << "##pilot_rho\t" << state.pilot_rho << "\n"
        << "##pilot_relative_floor\t" << state.pilot_relative_floor << "\n"
        << "##covariance_floor\t" << state.covariance_floor << "\n"
        << "##covariance_shrinkage\t" << state.covariance_shrinkage << "\n"
        << "##fisher_broadening\t" << state.fisher_broadening << "\n";
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
            out << "MODEL_COV\t" << c << "\t" << r;
            for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.model.covariances[c](r, j);
            out << "\nPILOT_RAW_COV\t" << c << "\t" << r;
            for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.pilot.raw_covariances[c](r, j);
            out << "\nPILOT_COV\t" << c << "\t" << r;
            for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.pilot.covariances[c](r, j);
            out << "\n";
        }
    }
    for (int32_t r = 0; r < dimension; ++r) {
        out << "SHRINKAGE_TARGET\t" << r;
        for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.model.shrinkage_target(r, j);
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
    bool saw_fisher_broadening = false;
    std::vector<std::vector<std::string>> records;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> token = fields(line);
        if (token.empty()) continue;
        if (token[0] == "##punkst_uac_state_v1") { saw_version = true; continue; }
        if (token[0].rfind("##", 0) == 0) {
            if (token.size() != 2) throw std::runtime_error("Malformed UAC state metadata");
            const std::string key = token[0].substr(2);
            if (key == "handoff") state.handoff = parse_handoff(token[1]);
            else if (key == "proposal") {
                state.proposal = parse_proposal(token[1]);
                saw_proposal = true;
            }
            else if (key == "particles") state.n_particles = std::stoi(token[1]);
            else if (key == "seed") state.seed = std::stoi(token[1]);
            else if (key == "selected_restart") state.selected_restart = std::stoi(token[1]);
            else if (key == "converged") state.converged = std::stoi(token[1]) != 0;
            else if (key == "components") components = std::stoi(token[1]);
            else if (key == "dimension") dimension = std::stoi(token[1]);
            else if (key == "basis_checksum") state.basis_checksum = std::stoull(token[1]);
            else if (key == "weighted_counts") state.weighted_counts = std::stoi(token[1]) != 0;
            else if (key == "center_floor") state.center_floor = std::stod(token[1]);
            else if (key == "pilot_rho") state.pilot_rho = std::stod(token[1]);
            else if (key == "pilot_relative_floor") state.pilot_relative_floor = std::stod(token[1]);
            else if (key == "covariance_floor") state.covariance_floor = std::stod(token[1]);
            else if (key == "covariance_shrinkage") state.covariance_shrinkage = std::stod(token[1]);
            else if (key == "fisher_broadening") {
                state.fisher_broadening = std::stod(token[1]);
                saw_fisher_broadening = true;
            }
            continue;
        }
        records.push_back(std::move(token));
    }
    if (!saw_version || !saw_proposal || !saw_fisher_broadening
        || components <= 0 || dimension <= 0) {
        throw std::runtime_error("Invalid, stale, or unsupported UAC state");
    }
    state.helmert = Eigen::MatrixXd::Zero(dimension, dimension + 1);
    state.model.weights.resize(components);
    state.model.means = RowMajorMatrixXd::Zero(components, dimension);
    state.model.covariances.assign(components, Eigen::MatrixXd::Zero(dimension, dimension));
    state.model.shrinkage_target = Eigen::MatrixXd::Zero(dimension, dimension);
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
    if (state.topics.size() != static_cast<size_t>(dimension + 1)
        || !state.model.weights.allFinite() || !state.model.means.allFinite()
        || !(state.fisher_broadening > 0.0)
        || !std::isfinite(state.fisher_broadening)) {
        throw std::runtime_error("Incomplete UAC state");
    }
    return state;
}

void write_model(const std::string& path, const State& state,
    const Eigen::VectorXd* effective_membership) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC model: " + path);
    out << "#cluster\tweight\teffective_membership\tmean_variance\tlog_volume";
    for (const auto& topic : state.topics) out << "\t" << topic;
    out << "\n" << std::scientific << std::setprecision(10);
    RowMajorMatrixXd compositions = ilr_inverse(state.model.means, state.helmert);
    for (Eigen::Index c = 0; c < state.model.weights.size(); ++c) {
        Eigen::LLT<Eigen::MatrixXd> llt(state.model.covariances[c]);
        const Eigen::MatrixXd lower = llt.matrixL();
        const double log_volume = lower.diagonal().array().log().sum();
        out << c << "\t" << state.model.weights(c) << "\t"
            << (effective_membership ? (*effective_membership)(c) : -1.0)
            << "\t" << state.model.covariances[c].trace() / state.model.means.cols()
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
        const Eigen::Index second = order.size() > 1 ? order[1] : order[0];
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
        << "##particle_likelihood_seconds\t" << score.likelihood_seconds << "\n"
        << "##particle_bytes\t" << score.particle_bytes << "\n"
        << "#id\traw_total\teffective_total\trelative_ess\tmaximum_weight\tlog_likelihood_range\tlog_proposal_range\n"
        << std::scientific << std::setprecision(10);
    for (size_t d = 0; d < data.identifiers.size(); ++d) {
        const bool particle = d < score.particle_diagnostics.size();
        out << data.identifiers[d] << "\t"
            << (data.raw_totals.size() ? data.raw_totals(d) : 0.0) << "\t"
            << (data.effective_totals.size() ? data.effective_totals(d) : 0.0) << "\t";
        if (particle) {
            const auto& value = score.particle_diagnostics[d];
            out << value.relative_ess << "\t" << value.maximum_weight << "\t"
                << value.log_likelihood_range << "\t"
                << value.log_proposal_range;
        } else {
            out << "NA\tNA\tNA\tNA";
        }
        out << "\n";
    }
}

void write_trace(const std::string& path,
    const std::vector<RestartTrace>& traces) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC trace: " + path);
    out << "#handoff\trestart\titeration\tobjective\tconverged\tcollapsed\n"
        << std::scientific << std::setprecision(12);
    for (const auto& trace : traces) {
        for (size_t iteration = 0; iteration < trace.objective.size(); ++iteration) {
            out << (trace.particle ? "particle" : "map") << "\t" << trace.restart
                << "\t" << iteration << "\t" << trace.objective[iteration]
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
        for (Eigen::Index b = a + 1; b < model.weights.size(); ++b) {
            const Eigen::VectorXd difference = model.means.row(a).transpose() - model.means.row(b).transpose();
            const Eigen::MatrixXd covariance = 0.5 * (model.covariances[a] + model.covariances[b]);
            Eigen::LLT<Eigen::MatrixXd> llt(covariance);
            const double standardized = std::sqrt(std::max(0.0, difference.dot(llt.solve(difference))));
            const double logdet_mean = 2.0 * Eigen::MatrixXd(llt.matrixL()).diagonal().array().log().sum();
            Eigen::LLT<Eigen::MatrixXd> llt_a(model.covariances[a]), llt_b(model.covariances[b]);
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
