#include "clustering/uac_stochastic_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>

namespace uac::detail {

double model_covariance_frobenius_norm(
    const Model& model, int32_t component) {
    if (model.covariance_kind == CovarianceKind::Dense) {
        return model.covariances.at(component).norm();
    }
    const Eigen::VectorXd diagonal = factor_diagonal(model, component);
    const Eigen::MatrixXd& factor =
        model.factor_covariances.at(component).factor;
    double squared = diagonal.squaredNorm();
    if (factor.cols() > 0) {
        squared += (factor.transpose() * factor).squaredNorm();
        squared += 2.0 * (diagonal.array()
            * factor.rowwise().squaredNorm().array()).sum();
    }
    return std::sqrt(std::max(0.0, squared));
}

double model_covariance_frobenius_difference(
    const Model& before, const Model& after, int32_t component) {
    if (before.covariance_kind != after.covariance_kind) {
        throw std::invalid_argument(
            "Cannot compare different UAC covariance representations");
    }
    if (before.covariance_kind == CovarianceKind::Dense) {
        return (before.covariances.at(component)
            - after.covariances.at(component)).norm();
    }
    const Eigen::VectorXd diagonal_before =
        factor_diagonal(before, component);
    const Eigen::VectorXd diagonal_after =
        factor_diagonal(after, component);
    const Eigen::VectorXd diagonal_delta =
        diagonal_after - diagonal_before;
    const Eigen::MatrixXd& left =
        before.factor_covariances.at(component).factor;
    const Eigen::MatrixXd& right =
        after.factor_covariances.at(component).factor;
    double squared = diagonal_delta.squaredNorm();
    if (left.cols() > 0) {
        squared += (left.transpose() * left).squaredNorm();
        squared -= 2.0 * (diagonal_delta.array()
            * left.rowwise().squaredNorm().array()).sum();
    }
    if (right.cols() > 0) {
        squared += (right.transpose() * right).squaredNorm();
        squared += 2.0 * (diagonal_delta.array()
            * right.rowwise().squaredNorm().array()).sum();
    }
    if (left.cols() > 0 && right.cols() > 0) {
        squared -= 2.0 * (left.transpose() * right).squaredNorm();
    }
    return std::sqrt(std::max(0.0, squared));
}

double model_parameter_change(const Model& before, const Model& after) {
    double change = (before.weights - after.weights).cwiseAbs().sum();
    for (Eigen::Index c = 0; c < before.means.rows(); ++c) {
        const double mean_scale = std::max(1.0,
            before.means.row(c).norm());
        change = std::max(change,
            (before.means.row(c) - after.means.row(c)).norm() / mean_scale);
        change = std::max(change,
            model_covariance_frobenius_difference(
                before, after, static_cast<int32_t>(c))
            / std::max(1.0, model_covariance_frobenius_norm(
                before, static_cast<int32_t>(c))));
    }
    return change;
}

namespace {

uint64_t mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

double deterministic_uniform(uint64_t seed, int32_t document) {
    const uint64_t bits = mix64(seed ^ static_cast<uint64_t>(document));
    return static_cast<double>(bits >> 11) * 0x1.0p-53;
}

template<class ParticleCollection>
void append_document(RaggedParticleSet& destination,
    const ParticleCollection& source, int32_t document) {
    const int32_t samples = source.samples_for_document(document);
    const auto values = source.values_for_document(document);
    const auto likelihood = source.log_likelihood_for_document(document);
    const auto proposal = source.log_proposal_for_document(document);
    const auto origins = source.proposal_origins_for_document(document);
    destination.values.insert(destination.values.end(), values.data(),
        values.data() + static_cast<int64_t>(samples) * source.dimension);
    destination.log_likelihood.insert(destination.log_likelihood.end(),
        likelihood.data(), likelihood.data() + samples);
    destination.log_proposal.insert(destination.log_proposal.end(),
        proposal.data(), proposal.data() + samples);
    destination.proposal_origins.insert(destination.proposal_origins.end(),
        origins.data(), origins.data() + samples);
    if (document < static_cast<int32_t>(source.proposal_candidates.size())) {
        destination.proposal_candidates.push_back(
            source.proposal_candidates[document]);
    } else {
        destination.proposal_candidates.push_back(0);
    }
    if constexpr (std::is_same_v<ParticleCollection, RaggedParticleSet>) {
        if (document < static_cast<int32_t>(
                source.adaptive_diagnostics.size())) {
            destination.adaptive_diagnostics.push_back(
                source.adaptive_diagnostics[document]);
        }
    }
    destination.maximum_samples = std::max(
        destination.maximum_samples, samples);
    destination.offsets.push_back(destination.offsets.back() + samples);
    ++destination.documents;
}

void reserve_ragged_particles(RaggedParticleSet& particles,
    int32_t documents, int64_t samples, bool adaptive) {
    particles.offsets.reserve(static_cast<size_t>(documents) + 1);
    particles.values.reserve(static_cast<size_t>(samples)
        * particles.dimension);
    particles.log_likelihood.reserve(samples);
    particles.log_proposal.reserve(samples);
    particles.proposal_origins.reserve(samples);
    particles.proposal_candidates.reserve(documents);
    if (adaptive) particles.adaptive_diagnostics.reserve(documents);
}

uint64_t selected_particle_bytes(
    const std::vector<int32_t>& document_samples,
    const std::vector<uint8_t>& selected, int32_t dimension,
    bool adaptive) {
    uint64_t documents = 0;
    uint64_t samples = 0;
    for (size_t d = 0; d < selected.size(); ++d) {
        if (!selected[d]) continue;
        ++documents;
        samples += static_cast<uint64_t>(document_samples.at(d));
    }
    return sizeof(double) * samples * (dimension + 2)
        + sizeof(int32_t) * (samples + documents)
        + sizeof(int64_t) * (documents + 1)
        + (adaptive ? sizeof(AdaptiveParticleDiagnostic) * documents : 0);
}

int32_t selected_maximum_samples(
    const std::vector<int32_t>& document_samples,
    const std::vector<uint8_t>& selected) {
    int32_t out = 0;
    for (size_t d = 0; d < selected.size(); ++d) {
        if (selected[d]) out = std::max(out, document_samples.at(d));
    }
    return out;
}

void release_warmup_moments(Expectation& warmup) {
    warmup.responsibilities = {};
    warmup.membership_weight_squared = {};
    warmup.first = {};
    warmup.second.clear();
    warmup.second.shrink_to_fit();
    warmup.sum_y2 = {};
    warmup.sum_f = {};
    warmup.sum_ff.clear();
    warmup.sum_ff.shrink_to_fit();
    warmup.sum_yf.clear();
    warmup.sum_yf.shrink_to_fit();
    warmup.particle_diagnostics.clear();
    warmup.particle_diagnostics.shrink_to_fit();
    warmup.per_document_evaluated_components.clear();
    warmup.per_document_evaluated_components.shrink_to_fit();
    warmup.per_document_omitted_component_mass.clear();
    warmup.per_document_omitted_component_mass.shrink_to_fit();
}

void update_subsample_peak(FitScheduleDiagnostics& diagnostics,
    uint64_t bytes, const char* phase) {
    if (bytes > diagnostics.subsample_peak_bytes) {
        diagnostics.subsample_peak_bytes = bytes;
        diagnostics.subsample_peak_phase = phase;
    }
}

RaggedParticleSet selected_particles(const ParticleCache& cache,
    const std::vector<uint8_t>& selected,
    const std::vector<uint16_t>& strata,
    const std::vector<int32_t>& document_samples,
    const Eigen::VectorXd& inclusion_probability,
    Eigen::VectorXd* weights) {
    RaggedParticleSet out;
    out.dimension = cache.dimension_count();
    const int32_t selected_documents = static_cast<int32_t>(std::count(
        selected.begin(), selected.end(), uint8_t{1}));
    const int64_t selected_samples = std::inner_product(
        document_samples.begin(), document_samples.end(), selected.begin(),
        int64_t{0}, std::plus<int64_t>{},
        [](int32_t samples, uint8_t keep) {
            return keep ? static_cast<int64_t>(samples) : int64_t{0};
        });
    reserve_ragged_particles(out, selected_documents, selected_samples,
        cache.adaptive_particles());
    out.offsets.push_back(0);
    if (weights != nullptr) weights->resize(selected_documents);
    Eigen::Index next_weight = 0;
    for (size_t shard_index = 0; shard_index < cache.shard_count();
            ++shard_index) {
        const CachedParticleShard shard = read_particle_cache(
            cache.shard_path(shard_index));
        std::visit([&](const auto& particles) {
            for (int32_t local = 0; local < particles.documents; ++local) {
                const int32_t global = cache.shard_first_document(shard_index)
                    + local;
                if (!selected[global]) continue;
                append_document(out, particles, local);
                if (weights != nullptr) {
                    (*weights)(next_weight++) =
                        1.0 / inclusion_probability(strata.at(global));
                }
            }
        }, shard);
    }
    return out;
}

struct DiskSubsampleStore {
    std::vector<std::filesystem::path> paths;
    std::vector<int32_t> documents_per_shard;
    Eigen::VectorXd weights;
    int32_t documents = 0;
    uint64_t bytes = 0;
    uint64_t peak_shard_bytes = 0;
};

void clear_disk_subsample(DiskSubsampleStore& store) {
    for (const auto& path : store.paths) {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
    store = {};
}

DiskSubsampleStore write_disk_subsample(const ParticleCache& cache,
    const std::vector<uint8_t>& selected,
    const std::vector<uint16_t>& strata,
    const Eigen::VectorXd& probability,
    FitScheduleDiagnostics& diagnostics) {
    const auto start = std::chrono::steady_clock::now();
    DiskSubsampleStore out;
    const int32_t selected_documents = static_cast<int32_t>(std::count(
        selected.begin(), selected.end(), uint8_t{1}));
    out.weights.resize(selected_documents);
    Eigen::Index next_weight = 0;
    int32_t output_shard = 0;
    for (size_t shard_index = 0; shard_index < cache.shard_count();
            ++shard_index) {
        const auto& source_path = cache.shard_path(shard_index);
        diagnostics.subsample_read_bytes += std::filesystem::file_size(source_path);
        const CachedParticleShard shard = read_particle_cache(source_path);
        RaggedParticleSet compact;
        compact.dimension = cache.dimension_count();
        compact.first_document = out.documents;
        compact.offsets.push_back(0);
        std::visit([&](const auto& particles) {
            int32_t compact_documents = 0;
            int64_t compact_samples = 0;
            for (int32_t local = 0; local < particles.documents; ++local) {
                const int32_t global = cache.shard_first_document(shard_index)
                    + local;
                if (!selected[global]) continue;
                ++compact_documents;
                compact_samples += particles.samples_for_document(local);
            }
            reserve_ragged_particles(compact, compact_documents,
                compact_samples,
                std::is_same_v<std::decay_t<decltype(particles)>,
                    RaggedParticleSet>);
            for (int32_t local = 0; local < particles.documents; ++local) {
                const int32_t global = cache.shard_first_document(shard_index)
                    + local;
                if (!selected[global]) continue;
                append_document(compact, particles, local);
                out.weights(next_weight++) =
                    1.0 / probability(strata[global]);
            }
        }, shard);
        if (compact.documents == 0) continue;
        if (compact.adaptive_diagnostics.size()
                != static_cast<size_t>(compact.documents)) {
            compact.adaptive_diagnostics.resize(compact.documents);
        }
        std::ostringstream name;
        name << "subsample-" << std::setw(6) << std::setfill('0')
            << output_shard++ << ".bin";
        const std::filesystem::path path =
            cache.work_directory() / name.str();
        write_particle_cache(path, compact);
        const uint64_t bytes = std::filesystem::file_size(path);
        out.paths.push_back(path);
        out.documents_per_shard.push_back(compact.documents);
        out.documents += compact.documents;
        out.bytes += bytes;
        out.peak_shard_bytes = std::max(
            out.peak_shard_bytes, particle_set_bytes(compact));
        diagnostics.subsample_write_bytes += bytes;
    }
    diagnostics.subsample_io_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return out;
}

Expectation disk_subsample_expectation(const DiskSubsampleStore& store,
    const Model& model, const ComponentScreeningOptions& screening,
    FitScheduleDiagnostics& diagnostics) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(model.means.cols());
    const int32_t factor_rank = model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(
            model.factor_covariances.front().factor.cols()) : -1;
    Expectation out = empty_expectation(
        store.documents, components, dimension, factor_rank);
    out.membership_weight_squared = Eigen::VectorXd::Zero(components);
    Eigen::Index weight_offset = 0;
    const auto start = std::chrono::steady_clock::now();
    for (size_t index = 0; index < store.paths.size(); ++index) {
        diagnostics.subsample_read_bytes +=
            std::filesystem::file_size(store.paths[index]);
        const CachedParticleShard shard = read_particle_cache(
            store.paths[index]);
        const int32_t documents = store.documents_per_shard[index];
        const Eigen::VectorXd weights = store.weights.segment(
            weight_offset, documents);
        weight_offset += documents;
        Expectation local;
        std::visit([&](const auto& particles) {
            local = particle_expectation(particles, model,
                ExpectationRequest{false, false, true}, screening, &weights);
        }, shard);
        accumulate_expectation(out, local);
        out.gaussian_seconds += local.gaussian_seconds;
        out.moment_seconds += local.moment_seconds;
        out.peak_workspace_bytes = std::max(
            out.peak_workspace_bytes, local.peak_workspace_bytes);
        out.peak_particle_bytes = std::max(
            out.peak_particle_bytes, particle_set_bytes(
                std::get<RaggedParticleSet>(shard)));
    }
    diagnostics.subsample_io_seconds += std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start).count();
    return out;
}

Eigen::VectorXd kish_effective_membership(
    const Expectation& expectation) {
    Eigen::VectorXd out = Eigen::VectorXd::Zero(
        expectation.membership.size());
    if (expectation.membership_weight_squared.size()
            != expectation.membership.size()) {
        return expectation.membership;
    }
    for (Eigen::Index c = 0; c < out.size(); ++c) {
        if (expectation.membership_weight_squared(c) > 0.0) {
            out(c) = expectation.membership(c)
                * expectation.membership(c)
                / expectation.membership_weight_squared(c);
        }
    }
    return out;
}

void scale_expectation(Expectation& value, double factor,
    int32_t documents) {
    value.documents = documents;
    value.membership *= factor;
    value.first *= factor;
    for (auto& second : value.second) second *= factor;
    value.sum_y2 *= factor;
    value.sum_f *= factor;
    for (auto& sum : value.sum_ff) sum *= factor;
    for (auto& sum : value.sum_yf) sum *= factor;
    value.log_likelihood *= factor;
    value.log_likelihood_upper *= factor;
    value.responsibility_entropy_sum *= factor;
}

template<class Value>
void blend_value(Value& target, const Value& source, double step) {
    target = (1.0 - step) * target + step * source;
}

void blend_expectation(Expectation& target, const Expectation& source,
    double step) {
    if (!target.second.empty()) {
        for (size_t c = 0; c < target.second.size(); ++c) {
            const double left_membership = target.membership(c);
            const double right_membership = source.membership(c);
            const double left_mass = (1.0 - step) * left_membership;
            const double right_mass = step * right_membership;
            const double combined_mass = left_mass + right_mass;
            if (!(combined_mass > 0.0)) continue;
            Eigen::VectorXd left_mean =
                Eigen::VectorXd::Zero(target.first.cols());
            Eigen::VectorXd right_mean =
                Eigen::VectorXd::Zero(source.first.cols());
            if (left_membership > 0.0) {
                left_mean = target.first.row(c).transpose() / left_membership;
            }
            if (right_membership > 0.0) {
                right_mean = source.first.row(c).transpose()
                    / right_membership;
            }
            Eigen::MatrixXd scatter = Eigen::MatrixXd::Zero(
                target.first.cols(), target.first.cols());
            if (left_membership > 0.0) {
                scatter.noalias() += (1.0 - step)
                    * (target.second[c] - left_membership
                        * left_mean * left_mean.transpose());
            }
            if (right_membership > 0.0) {
                scatter.noalias() += step
                    * (source.second[c] - right_membership
                        * right_mean * right_mean.transpose());
            }
            if (left_mass > 0.0 && right_mass > 0.0) {
                const Eigen::VectorXd delta = left_mean - right_mean;
                scatter.noalias() += (left_mass * right_mass / combined_mass)
                    * delta * delta.transpose();
            }
            const Eigen::VectorXd combined_mean =
                (left_mass * left_mean + right_mass * right_mean)
                / combined_mass;
            target.membership(c) = combined_mass;
            target.first.row(c) =
                (combined_mass * combined_mean).transpose();
            target.second[c] = 0.5 * (scatter + scatter.transpose())
                + combined_mass * combined_mean * combined_mean.transpose();
        }
    } else {
        blend_value(target.membership, source.membership, step);
        blend_value(target.first, source.first, step);
    }
    blend_value(target.sum_y2, source.sum_y2, step);
    blend_value(target.sum_f, source.sum_f, step);
    for (size_t c = 0; c < target.sum_ff.size(); ++c) {
        blend_value(target.sum_ff[c], source.sum_ff[c], step);
        blend_value(target.sum_yf[c], source.sum_yf[c], step);
    }
    target.log_likelihood = (1.0 - step) * target.log_likelihood
        + step * source.log_likelihood;
    target.log_likelihood_upper =
        (1.0 - step) * target.log_likelihood_upper
        + step * source.log_likelihood_upper;
}

Candidate initialize_candidate(Model initial, const FitOptions& options,
    const RestartTrace& initialization_trace) {
    Candidate out;
    out.model = std::move(initial);
    out.trace = initialization_trace;
    out.trace.points.clear();
    out.trace.model_trace.clear();
    out.trace.estep_work = {};
    out.trace.converged = false;
    out.trace.collapsed = false;
    out.trace.succeeded = false;
    out.trace.handoff = HandoffMode::Particle;
    out.trace.phase = options.particle_fit_schedule
            == ParticleFitSchedule::Subsample
        ? TracePhase::SubsampleEm : TracePhase::OnlineEm;
    out.trace.fixed_em_iteration_schedule = true;
    out.trace.completed_updates = 0;
    return out;
}

bool apply_update(Candidate& candidate, const Expectation& expectation,
    const FitOptions& options, double shrinkage, bool update_weights,
    const Eigen::VectorXd* effective_membership = nullptr,
    bool allow_extinction = false) {
    if (options.capture_model_trace) {
        ModelTraceEntry entry;
        entry.completed_updates = candidate.trace.completed_updates;
        entry.event = TraceEvent::Evaluation;
        entry.update_shrinkage_strength = shrinkage;
        entry.model = candidate.model;
        candidate.trace.model_trace.push_back(std::move(entry));
    }
    const ModelUpdate update = update_model(candidate.model, expectation,
        shrinkage, options.covariance_floor,
        options.adaptive_covariance_shrinkage, update_weights,
        allow_extinction,
        effective_membership);
    if (!update.valid) {
        candidate.trace.collapsed = true;
        return false;
    }
    ++candidate.trace.completed_updates;
    return true;
}

void record_evaluation(Candidate& candidate, const FitOptions& options,
    const Expectation& expectation) {
    accumulate_estep_work(candidate.trace, expectation);
    record_trace_point(candidate.trace, options, TraceEvent::Evaluation,
        candidate.trace.completed_updates, expectation.log_likelihood,
        active_component_count(candidate.model),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());
}

struct SubsampleAllocation {
    Eigen::VectorXd probability;
    Eigen::VectorXd predicted_effective_size;
    bool converged = false;
    int32_t iterations = 0;
};

double material_subsample_transfer(const Expectation& warmup,
    int32_t stratum, int32_t component) {
    constexpr double minimum_relative_transfer = 1e-3;
    const double maximum = warmup.subsample_transfer.col(component).maxCoeff();
    const double value = warmup.subsample_transfer(stratum, component);
    return maximum > 0.0 && value >= minimum_relative_transfer * maximum
        ? value : 0.0;
}

SubsampleAllocation allocate_subsample(const Expectation& warmup,
    const FitOptions& options) {
    const int32_t components = static_cast<int32_t>(
        warmup.membership.size());
    SubsampleAllocation out;
    out.probability = Eigen::VectorXd::Constant(
        components, options.fit_subsample_base_fraction);
    out.predicted_effective_size = Eigen::VectorXd::Zero(components);
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(
        components, std::max(1e-12, options.fit_subsample_base_fraction));
    for (int32_t h = 0; h < components; ++h) {
        if (warmup.subsample_stratum_documents(h) == 0
            || warmup.subsample_stratum_documents(h)
                <= options.fit_subsample_target) {
            lower(h) = 1.0;
        }
    }
    Eigen::VectorXd limit(components);
    for (int32_t c = 0; c < components; ++c) {
        const double target = std::min(
            warmup.membership(c) / options.fit_subsample_safety_factor,
            options.fit_subsample_safety_factor * options.fit_subsample_target);
        limit(c) = target > 0.0
            ? warmup.membership(c) * warmup.membership(c) / target
            : std::numeric_limits<double>::infinity();
    }
    auto predicted = [&]() {
        for (int32_t c = 0; c < components; ++c) {
            double denominator = 0.0;
            for (int32_t h = 0; h < components; ++h) {
                denominator += material_subsample_transfer(warmup, h, c)
                    / out.probability(h);
            }
            out.predicted_effective_size(c) = denominator > 0.0
                ? warmup.membership(c) * warmup.membership(c) / denominator
                : 0.0;
        }
    };
    auto evaluate_dual = [&](const Eigen::VectorXd& lambda,
                             Eigen::VectorXd& probability,
                             Eigen::VectorXd& gradient) {
        double value = 0.0;
        for (int32_t h = 0; h < components; ++h) {
            double coupled = 0.0;
            for (int32_t c = 0; c < components; ++c) {
                coupled += lambda(c)
                    * material_subsample_transfer(warmup, h, c);
            }
            const double cost = std::max(
                1.0, warmup.subsample_stratum_bytes(h));
            probability(h) = std::min(1.0, std::max(lower(h),
                std::sqrt(std::max(0.0, coupled) / cost)));
            value += cost * probability(h)
                + coupled / probability(h);
        }
        gradient.setZero();
        for (int32_t c = 0; c < components; ++c) {
            if (!std::isfinite(limit(c))) continue;
            double denominator = 0.0;
            for (int32_t h = 0; h < components; ++h) {
                denominator += material_subsample_transfer(warmup, h, c)
                    / probability(h);
            }
            gradient(c) = denominator - limit(c);
            value -= lambda(c) * limit(c);
        }
        return value;
    };
    Eigen::VectorXd lambda = Eigen::VectorXd::Zero(components);
    Eigen::VectorXd gradient(components), trial_probability(components);
    double dual = evaluate_dual(lambda, out.probability, gradient);
    double step = 1.0;
    for (int32_t iteration = 0; iteration < 500; ++iteration) {
        out.iterations = iteration + 1;
        double maximum_violation = 0.0;
        for (int32_t c = 0; c < components; ++c) {
            if (std::isfinite(limit(c))) {
                maximum_violation = std::max(maximum_violation,
                    gradient(c) / std::max(1.0, limit(c)));
            }
        }
        if (maximum_violation <= 1e-6) {
            out.converged = true;
            break;
        }
        Eigen::VectorXd direction(components);
        for (int32_t c = 0; c < components; ++c) {
            direction(c) = std::isfinite(limit(c))
                ? gradient(c) / std::max(1.0, limit(c)) : 0.0;
        }
        bool accepted = false;
        for (int32_t backtrack = 0; backtrack < 40; ++backtrack) {
            const Eigen::VectorXd trial_lambda =
                (lambda + step * direction).cwiseMax(0.0);
            Eigen::VectorXd trial_gradient(components);
            const double trial_dual = evaluate_dual(
                trial_lambda, trial_probability, trial_gradient);
            const double ascent = gradient.dot(trial_lambda - lambda);
            if (std::isfinite(trial_dual)
                && trial_dual >= dual + 1e-4 * ascent) {
                lambda = trial_lambda;
                out.probability = trial_probability;
                gradient = trial_gradient;
                dual = trial_dual;
                step = std::min(1e6, step * 1.5);
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) break;
    }
    predicted();
    if (!out.converged) {
        // A monotone feasibility repair preserves every statistical target
        // without turning a stalled dual solve into an unconditional census.
        // Increasing any p_h can only decrease every Kish denominator, so
        // constraints repaired earlier remain feasible.
        out.probability = lower;
        for (int32_t c = 0; c < components; ++c) {
            if (!std::isfinite(limit(c))) continue;
            auto denominator = [&](double lambda_value) {
                double value = 0.0;
                for (int32_t h = 0; h < components; ++h) {
                    const double cost = std::max(
                        1.0, warmup.subsample_stratum_bytes(h));
                    const double transfer = material_subsample_transfer(
                        warmup, h, c);
                    const double proposal = std::sqrt(std::max(0.0,
                        lambda_value * transfer
                            / cost));
                    const double probability = std::min(1.0,
                        std::max(out.probability(h), proposal));
                    value += transfer / probability;
                }
                return value;
            };
            if (denominator(0.0) <= limit(c) * (1.0 + 1e-8)) continue;
            double low = 0.0;
            double high = 1.0;
            while (denominator(high) > limit(c) * (1.0 + 1e-10)
                    && high < 1e12) {
                high *= 2.0;
            }
            for (int32_t step_index = 0; step_index < 80; ++step_index) {
                const double middle = 0.5 * (low + high);
                if (denominator(middle) > limit(c)) {
                    low = middle;
                } else {
                    high = middle;
                }
            }
            for (int32_t h = 0; h < components; ++h) {
                const double cost = std::max(
                    1.0, warmup.subsample_stratum_bytes(h));
                const double proposal = std::sqrt(std::max(0.0,
                    high * material_subsample_transfer(warmup, h, c) / cost));
                out.probability(h) = std::min(1.0,
                    std::max(out.probability(h), proposal));
            }
        }
        predicted();
        out.converged = true;
    }
    return out;
}

void recompute_predicted_effective_size(const Expectation& warmup,
    SubsampleAllocation& allocation) {
    const int32_t components = static_cast<int32_t>(warmup.membership.size());
    allocation.predicted_effective_size.resize(components);
    for (int32_t c = 0; c < components; ++c) {
        double denominator = 0.0;
        for (int32_t h = 0; h < components; ++h) {
            denominator += material_subsample_transfer(warmup, h, c)
                / allocation.probability(h);
        }
        allocation.predicted_effective_size(c) = denominator > 0.0
            ? warmup.membership(c) * warmup.membership(c) / denominator
            : 0.0;
    }
}

bool increase_topup_probabilities(const Expectation& warmup,
    const Eigen::VectorXd& target, const Eigen::VectorXd& realized,
    const Model& model, SubsampleAllocation& allocation,
    FitScheduleDiagnostics& diagnostics) {
    bool deficient = false;
    const int32_t components = static_cast<int32_t>(target.size());
    for (int32_t c = 0; c < components; ++c) {
        if (model.weights(c) <= 0.0 || realized(c) >= target(c)) continue;
        deficient = true;
        ++diagnostics.subsample_component_topups[c];
        const double ratio = target(c) / std::max(1.0, realized(c));
        const double multiplier = std::max(2.0, ratio * ratio);
        const double maximum = warmup.subsample_transfer.col(c).maxCoeff();
        for (int32_t h = 0; h < components; ++h) {
            const double transfer = material_subsample_transfer(warmup, h, c);
            if (!(maximum > 0.0) || !(transfer > 0.0)) continue;
            allocation.probability(h) = std::min(1.0,
                allocation.probability(h) * std::pow(multiplier,
                    transfer / maximum));
        }
    }
    if (deficient) recompute_predicted_effective_size(warmup, allocation);
    return deficient;
}

void refresh_allocation_diagnostics(FitScheduleDiagnostics& diagnostics,
    const SubsampleAllocation& allocation) {
    diagnostics.subsample_stratum_probability.assign(
        allocation.probability.data(),
        allocation.probability.data() + allocation.probability.size());
    diagnostics.subsample_component_predicted_effective_size.assign(
        allocation.predicted_effective_size.data(),
        allocation.predicted_effective_size.data()
            + allocation.predicted_effective_size.size());
}

void populate_subsample_diagnostics(FitScheduleDiagnostics& diagnostics,
    const Expectation& warmup, const SubsampleAllocation& allocation,
    const FitOptions& options) {
    const int32_t components = static_cast<int32_t>(
        warmup.membership.size());
    diagnostics.subsample_allocator_converged = allocation.converged;
    diagnostics.subsample_allocator_iterations = allocation.iterations;
    diagnostics.subsample_stratum_documents.resize(components);
    diagnostics.subsample_stratum_purity.resize(components);
    diagnostics.subsample_stratum_probability.resize(components);
    diagnostics.subsample_stratum_bytes.resize(components);
    diagnostics.subsample_component_target.resize(components);
    diagnostics.subsample_component_predicted_effective_size.resize(components);
    diagnostics.subsample_component_topups.assign(components, 0);
    for (int32_t h = 0; h < components; ++h) {
        diagnostics.subsample_stratum_documents[h] =
            warmup.subsample_stratum_documents(h);
        diagnostics.subsample_stratum_purity[h] =
            warmup.subsample_stratum_documents(h) > 0
            ? warmup.subsample_stratum_purity(h)
                / warmup.subsample_stratum_documents(h) : 0.0;
        diagnostics.subsample_stratum_probability[h] =
            allocation.probability(h);
        diagnostics.subsample_stratum_bytes[h] = static_cast<uint64_t>(
            std::ceil(warmup.subsample_stratum_bytes(h)));
        diagnostics.subsample_component_target[h] = std::min(
            warmup.membership(h) / options.fit_subsample_safety_factor,
            static_cast<double>(options.fit_subsample_target));
        diagnostics.subsample_component_predicted_effective_size[h] =
            allocation.predicted_effective_size(h);
    }
}

std::vector<uint8_t> select_subsample_documents(
    const std::vector<uint16_t>& strata,
    const Eigen::VectorXd& probability, uint64_t seed) {
    std::vector<uint8_t> selected(strata.size(), 0);
    for (size_t d = 0; d < strata.size(); ++d) {
        selected[d] = deterministic_uniform(
            seed, static_cast<int32_t>(d)) < probability(strata[d]);
    }
    return selected;
}

Eigen::VectorXd selected_weights(const std::vector<uint16_t>& strata,
    const std::vector<uint8_t>& selected,
    const Eigen::VectorXd& probability) {
    Eigen::VectorXd out(std::count(
        selected.begin(), selected.end(), uint8_t{1}));
    Eigen::Index next = 0;
    for (size_t d = 0; d < selected.size(); ++d) {
        if (selected[d]) out(next++) = 1.0 / probability(strata[d]);
    }
    return out;
}

void update_realized_subsample_diagnostics(
    FitScheduleDiagnostics& diagnostics, const Expectation& expectation,
    const Eigen::VectorXd& target) {
    const Eigen::VectorXd effective = kish_effective_membership(expectation);
    diagnostics.subsample_component_realized_effective_size.assign(
        effective.data(), effective.data() + effective.size());
    diagnostics.subsample_minimum_effective_size = effective.size() > 0
        ? effective.minCoeff() : 0.0;
    diagnostics.subsample_minimum_target_ratio = 1.0;
    for (Eigen::Index c = 0; c < effective.size(); ++c) {
        if (target(c) > 0.0) {
            diagnostics.subsample_minimum_target_ratio = std::min(
                diagnostics.subsample_minimum_target_ratio,
                effective(c) / target(c));
        }
    }
}

template<class ExpectationFunction, class TopupFunction>
bool run_subsample_updates(Candidate& candidate,
    FitScheduleDiagnostics& diagnostics, const FitOptions& options,
    int32_t total_documents, int32_t& subsample_documents,
    const Eigen::VectorXd& target,
    ExpectationFunction expectation_function,
    TopupFunction topup_function) {
    const int32_t maximum_updates = options.fit_subsample_max_updates;
    int64_t document_cap = 0;
    if (options.fit_document_budget > 0.0) {
        document_cap = static_cast<int64_t>(std::floor(
            options.fit_document_budget * total_documents));
        if (static_cast<int64_t>(options.fit_subsample_min_updates)
                * subsample_documents > document_cap) {
            throw std::invalid_argument(
                "UAC subsample document budget cannot permit the minimum updates");
        }
    }
    const double shrinkage = options.adaptive_covariance_shrinkage
        ? options.covariance_shrinkage_strength : 0.0;
    diagnostics.subsample_convergence_reason = "maximum_updates";
    for (int32_t iteration = 0; iteration < maximum_updates; ++iteration) {
        if (document_cap > 0
            && diagnostics.approximate_documents + subsample_documents
                > document_cap) {
            diagnostics.subsample_convergence_reason = "document_budget";
            break;
        }
        Expectation expectation = expectation_function(candidate.model);
        ++diagnostics.subsample_evaluations;
        diagnostics.approximate_documents += subsample_documents;
        Eigen::VectorXd effective = kish_effective_membership(expectation);
        while (topup_function(expectation, candidate.model)) {
            if (document_cap > 0
                && diagnostics.approximate_documents + subsample_documents
                    > document_cap) {
                if (iteration < options.fit_subsample_min_updates) {
                    throw std::invalid_argument(
                        "UAC subsample top-up makes the document budget insufficient");
                }
                diagnostics.subsample_convergence_reason = "document_budget";
                return true;
            }
            expectation = expectation_function(candidate.model);
            ++diagnostics.subsample_evaluations;
            diagnostics.approximate_documents += subsample_documents;
            effective = kish_effective_membership(expectation);
        }
        update_realized_subsample_diagnostics(
            diagnostics, expectation, target);
        record_evaluation(candidate, options, expectation);
        const Model before = candidate.model;
        if (!apply_update(candidate, expectation, options, shrinkage,
                false, &effective, false)) {
            diagnostics.subsample_convergence_reason = "collapsed";
            return false;
        }
        diagnostics.subsample_parameter_change =
            model_parameter_change(before, candidate.model);
        ++diagnostics.approximate_updates;
        if (iteration + 1 >= options.fit_subsample_min_updates
            && diagnostics.subsample_parameter_change
                <= options.fit_subsample_change_tolerance) {
            diagnostics.subsample_convergence_reason = "parameter_change";
            candidate.trace.converged = true;
            break;
        }
    }
    return true;
}

template<class ExactExpectationFunction>
bool run_exact_tail(Candidate& candidate,
    FitScheduleDiagnostics& diagnostics, const FitOptions& options,
    ExactExpectationFunction exact_expectation) {
    diagnostics.tail_mode = options.fit_tail;
    const double shrinkage = options.adaptive_covariance_shrinkage
        ? options.covariance_shrinkage_strength : 0.0;
    auto assess = [&](const Expectation& exact) {
        Model counterfactual = candidate.model;
        const int32_t active_before = active_component_count(counterfactual);
        const ModelUpdate update = update_model(counterfactual, exact,
            shrinkage, options.covariance_floor,
            options.adaptive_covariance_shrinkage, true, true);
        diagnostics.audit_active_set_unchanged = update.valid
            && active_component_count(counterfactual) == active_before;
        diagnostics.audit_parameter_change = update.valid
            ? model_parameter_change(candidate.model, counterfactual)
            : std::numeric_limits<double>::infinity();
        diagnostics.audit_converged = update.valid
            && diagnostics.audit_active_set_unchanged
            && diagnostics.audit_parameter_change
                <= options.fit_subsample_change_tolerance;
    };
    auto audit = [&]() {
        Expectation exact = exact_expectation(candidate.model);
        ++diagnostics.full_data_evaluations;
        record_evaluation(candidate, options, exact);
        assess(exact);
        return exact;
    };
    if (options.fit_tail == FitTailMode::Adaptive) {
        Expectation exact = audit();
        if (!diagnostics.audit_converged
            && options.fit_full_tail_updates > 0) {
            if (!apply_update(candidate, exact, options, shrinkage,
                    true, nullptr, true)) {
                return false;
            }
            ++diagnostics.full_tail_updates;
            static_cast<void>(audit());
        }
    } else if (options.fit_tail == FitTailMode::Fixed) {
        for (int32_t update = 0;
                update < options.fit_full_tail_updates; ++update) {
            Expectation exact = exact_expectation(candidate.model);
            ++diagnostics.full_data_evaluations;
            record_evaluation(candidate, options, exact);
            if (!apply_update(candidate, exact, options, shrinkage,
                    true, nullptr, true)) {
                return false;
            }
            ++diagnostics.full_tail_updates;
        }
    }
    return true;
}

} // namespace

ApproximateParticleFit fit_cached_particle_approximate(
    const ParticleCache& cache, Model initial, const FitOptions& options,
    const ComponentScreeningOptions& screening,
    const RestartTrace& initialization_trace) {
    if (options.particle_fit_schedule == ParticleFitSchedule::Exact) {
        throw std::invalid_argument(
            "Exact UAC fitting does not use the approximate fit engine");
    }
    ApproximateParticleFit out;
    out.diagnostics.schedule = options.particle_fit_schedule;
    out.candidate = initialize_candidate(
        std::move(initial), options, initialization_trace);
    Candidate& candidate = out.candidate;
    const int32_t documents = cache.document_count();
    const double shrinkage = options.adaptive_covariance_shrinkage
        ? options.covariance_shrinkage_strength : 0.0;
    const auto fit_start = std::chrono::steady_clock::now();

    const bool subsample = options.particle_fit_schedule
        == ParticleFitSchedule::Subsample;
    Expectation warmup = cached_particle_expectation(cache, candidate.model,
        screening, ExpectationRequest{false, false, true, subsample},
        options.n_threads);
    ++out.diagnostics.full_data_evaluations;
    record_evaluation(candidate, options, warmup);
    if (!apply_update(candidate, warmup, options, 0.0, true,
            nullptr, true)) return out;
    out.diagnostics.full_warmup_updates = 1;
    const auto approximate_start = std::chrono::steady_clock::now();

    if (subsample) {
        const int32_t components = static_cast<int32_t>(
            candidate.model.weights.size());
        SubsampleAllocation allocation = allocate_subsample(warmup, options);
        out.diagnostics.subsample_full_cache_scans = 1;
        out.diagnostics.subsample_read_bytes = cache.storage_bytes();
        populate_subsample_diagnostics(
            out.diagnostics, warmup, allocation, options);
        out.diagnostics.subsample_memory_budget =
            options.fit_subsample_memory_budget;
        double expected_particle_bytes = 0.0;
        for (int32_t h = 0; h < components; ++h) {
            expected_particle_bytes += allocation.probability(h)
                * warmup.subsample_stratum_bytes(h);
        }
        const int32_t factor_rank = candidate.model.covariance_kind
                == CovarianceKind::FactorAnalytic
            ? static_cast<int32_t>(candidate.model
                .factor_covariances.front().factor.cols()) : -1;
        const bool screen = screening.mode != ComponentScreeningMode::Off;
        const uint64_t fixed_bytes =
            (sizeof(uint16_t) + sizeof(uint8_t) + sizeof(int32_t))
                * static_cast<uint64_t>(documents)
            + sizeof(double) * static_cast<uint64_t>(components)
                * (components + 16);
        const uint64_t predicted_payload = static_cast<uint64_t>(std::ceil(
            options.fit_subsample_safety_factor * expected_particle_bytes));
        const int32_t predicted_documents = std::max<int32_t>(1,
            static_cast<int32_t>(std::ceil(allocation.probability.dot(
                warmup.subsample_stratum_documents.cast<double>()))));
        const uint64_t predicted_workspace = particle_expectation_peak_bytes(
            predicted_documents, components, cache.dimension_count(),
            factor_rank, options.n_particles, screen);
        out.diagnostics.subsample_predicted_bytes = std::max(
            fixed_bytes + cache.peak_shard_bytes() + predicted_payload
                + sizeof(double) * static_cast<uint64_t>(predicted_documents),
            fixed_bytes + predicted_payload
                + sizeof(double) * static_cast<uint64_t>(predicted_documents)
                + predicted_workspace);
        SubsampleStorage storage = options.fit_subsample_storage;
        if (storage == SubsampleStorage::Auto) {
            storage = out.diagnostics.subsample_predicted_bytes
                    <= options.fit_subsample_memory_budget
                ? SubsampleStorage::Resident : SubsampleStorage::Disk;
        }
        out.diagnostics.subsample_storage = storage;
        if (storage == SubsampleStorage::Resident
            && out.diagnostics.subsample_predicted_bytes
                > options.fit_subsample_memory_budget) {
            throw std::runtime_error(
                "UAC resident subsample exceeds --fit-subsample-memory-budget");
        }

        Eigen::VectorXd target(components);
        for (int32_t c = 0; c < components; ++c) {
            target(c) = std::min<double>(
                options.fit_subsample_target,
                warmup.membership(c) / options.fit_subsample_safety_factor);
        }
        release_warmup_moments(warmup);
        RaggedParticleSet particles;
        DiskSubsampleStore disk_store;
        Eigen::VectorXd weights;
        std::vector<uint8_t> selected = select_subsample_documents(
            warmup.subsample_strata, allocation.probability, options.seed);
        int32_t subsample_documents = 0;
        auto rebuild_subsample = [&]() {
            const int32_t selected_documents = static_cast<int32_t>(std::count(
                selected.begin(), selected.end(), uint8_t{1}));
            if (selected_documents == 0) {
                throw std::runtime_error(
                    "UAC stratified subsample selected no documents");
            }
            const uint64_t payload = selected_particle_bytes(
                warmup.subsample_document_samples, selected,
                cache.dimension_count(), cache.adaptive_particles());
            const uint64_t weight_bytes = sizeof(double)
                * static_cast<uint64_t>(selected_documents);
            const int32_t maximum_samples = selected_maximum_samples(
                warmup.subsample_document_samples, selected);
            const uint64_t workspace = particle_expectation_peak_bytes(
                selected_documents, components, cache.dimension_count(),
                factor_rank, maximum_samples, screen);
            const uint64_t resident_peak = std::max(
                fixed_bytes + cache.peak_shard_bytes() + payload + weight_bytes,
                fixed_bytes + payload + weight_bytes + workspace);
            if (storage == SubsampleStorage::Resident
                && resident_peak > options.fit_subsample_memory_budget) {
                if (options.fit_subsample_storage == SubsampleStorage::Resident) {
                    throw std::runtime_error(
                        "UAC realized resident subsample exceeds memory budget");
                }
                storage = SubsampleStorage::Disk;
                out.diagnostics.subsample_storage = storage;
                ++out.diagnostics.subsample_storage_promotions;
                particles = {};
            }
            if (storage == SubsampleStorage::Resident) {
                particles = selected_particles(cache, selected,
                    warmup.subsample_strata,
                    warmup.subsample_document_samples,
                    allocation.probability, &weights);
                out.diagnostics.subsample_read_bytes += cache.storage_bytes();
                update_subsample_peak(out.diagnostics, resident_peak,
                    "resident_subsample_estep");
            } else {
                clear_disk_subsample(disk_store);
                disk_store = write_disk_subsample(cache, selected,
                    warmup.subsample_strata, allocation.probability,
                    out.diagnostics);
                weights = disk_store.weights;
                const uint64_t disk_peak = std::max(
                    fixed_bytes + cache.peak_shard_bytes()
                        + disk_store.peak_shard_bytes + weight_bytes,
                    fixed_bytes + disk_store.peak_shard_bytes
                        + weight_bytes + workspace);
                if (disk_peak > options.fit_subsample_memory_budget) {
                    throw std::runtime_error(
                        "UAC disk subsample working set exceeds memory budget");
                }
                update_subsample_peak(out.diagnostics, disk_peak,
                    "disk_subsample_estep");
            }
            ++out.diagnostics.subsample_full_cache_scans;
            subsample_documents = storage == SubsampleStorage::Resident
                ? particles.documents : disk_store.documents;
            out.diagnostics.subsample_documents = subsample_documents;
            out.diagnostics.subsample_selected_particle_bytes = payload;
            out.diagnostics.subsample_weighted_documents = weights.sum();
            out.diagnostics.subsample_maximum_weight = weights.maxCoeff();
            out.diagnostics.subsample_disk_bytes = disk_store.bytes;
            refresh_allocation_diagnostics(out.diagnostics, allocation);
        };
        rebuild_subsample();
        auto subsample_expectation = [&](const Model& model) {
            if (storage == SubsampleStorage::Resident) {
                return particle_expectation(particles, model,
                    ExpectationRequest{false, false, true}, screening,
                    &weights);
            }
            return disk_subsample_expectation(
                disk_store, model, screening, out.diagnostics);
        };
        auto topup_subsample = [&](const Expectation& expectation,
                               const Model& model) {
            if (out.diagnostics.subsample_topup_rounds
                    >= options.fit_subsample_topup_rounds) return false;
            const Eigen::VectorXd achievable = target.cwiseMin(
                expectation.membership);
            if (!increase_topup_probabilities(warmup, achievable,
                    kish_effective_membership(expectation),
                    model, allocation, out.diagnostics)) return false;
            ++out.diagnostics.subsample_topup_rounds;
            selected = select_subsample_documents(
                warmup.subsample_strata, allocation.probability, options.seed);
            rebuild_subsample();
            return true;
        };
        if (!run_subsample_updates(candidate, out.diagnostics, options,
                documents, subsample_documents,
                target, subsample_expectation, topup_subsample)) {
            return out;
        }
        out.diagnostics.approximate_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now()
                - approximate_start).count();
        ComponentScreeningOptions exact_screening = screening;
        exact_screening.mode = ComponentScreeningMode::Off;
        exact_screening.maximum_components = 0;
        auto exact_expectation = [&](const Model& model) {
            ++out.diagnostics.subsample_full_cache_scans;
            out.diagnostics.subsample_read_bytes += cache.storage_bytes();
            return cached_particle_expectation(cache, model,
                exact_screening, ExpectationRequest{false, false, true},
                options.n_threads);
        };
        if (options.fit_tail == FitTailMode::Adaptive) {
            out.diagnostics.tail_mode = FitTailMode::Adaptive;
            const double tail_shrinkage =
                options.adaptive_covariance_shrinkage
                ? options.covariance_shrinkage_strength : 0.0;
            auto score_audit = [&]() {
                const auto score_start = std::chrono::steady_clock::now();
                Expectation exact;
                ScoreResult score = score_particle_cache(cache,
                    candidate.model, exact_screening,
                    false,
                    options.n_threads, &exact, true);
                score.scoring_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now()
                    - score_start).count();
                ++out.diagnostics.subsample_full_cache_scans;
                out.diagnostics.subsample_read_bytes += cache.storage_bytes();
                ++out.diagnostics.full_data_evaluations;
                Model counterfactual = candidate.model;
                const int32_t active_before =
                    active_component_count(counterfactual);
                const ModelUpdate update = update_model(counterfactual,
                    exact, tail_shrinkage, options.covariance_floor,
                    options.adaptive_covariance_shrinkage, true, true);
                out.diagnostics.audit_active_set_unchanged = update.valid
                    && active_component_count(counterfactual)
                        == active_before;
                out.diagnostics.audit_parameter_change = update.valid
                    ? model_parameter_change(
                        candidate.model, counterfactual)
                    : std::numeric_limits<double>::infinity();
                out.diagnostics.audit_converged = update.valid
                    && out.diagnostics.audit_active_set_unchanged
                    && out.diagnostics.audit_parameter_change
                        <= options.fit_subsample_change_tolerance;
                return std::make_pair(
                    std::move(score), std::move(exact));
            };
            auto terminal = score_audit();
            if (!out.diagnostics.audit_converged
                && options.fit_full_tail_updates > 0) {
                record_evaluation(candidate, options, terminal.second);
                if (!apply_update(candidate, terminal.second, options,
                        tail_shrinkage, true, nullptr, true)) return out;
                ++out.diagnostics.full_tail_updates;
                terminal = score_audit();
            }
            out.terminal_score = std::move(terminal.first);
            out.terminal_expectation = std::move(terminal.second);
        } else if (!run_exact_tail(candidate, out.diagnostics, options,
                exact_expectation)) {
            return out;
        }
    } else {
        Expectation running = warmup;
        std::vector<size_t> order(cache.shard_count());
        std::iota(order.begin(), order.end(), size_t{0});
        std::mt19937_64 random(options.seed ^ 0x6f6e6c696e65ull);
        const int64_t target_documents = static_cast<int64_t>(std::ceil(
            options.fit_document_budget * documents));
        int64_t processed = 0;
        int32_t update_index = 0;
        RaggedParticleSet batch;
        auto reset_batch = [&]() {
            batch = RaggedParticleSet{};
            batch.dimension = cache.dimension_count();
            batch.offsets.push_back(0);
        };
        reset_batch();
        auto flush_batch = [&]() {
            if (batch.documents == 0) return true;
            Expectation estimate = particle_expectation(batch,
                candidate.model, ExpectationRequest{false, false, true},
                screening);
            accumulate_estep_work(candidate.trace, estimate);
            processed += batch.documents;
            out.diagnostics.approximate_documents += batch.documents;
            scale_expectation(estimate,
                static_cast<double>(documents) / batch.documents,
                documents);
            record_trace_point(candidate.trace, options,
                TraceEvent::Evaluation, candidate.trace.completed_updates,
                estimate.log_likelihood,
                active_component_count(candidate.model),
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN());
            const double tau0 = std::pow(options.fit_step_initial,
                -1.0 / options.fit_step_kappa);
            const double step = std::pow(update_index + tau0,
                -options.fit_step_kappa);
            blend_expectation(running, estimate, step);
            if (!apply_update(candidate, running, options, shrinkage, true)) {
                return false;
            }
            ++out.diagnostics.approximate_updates;
            ++update_index;
            reset_batch();
            return true;
        };
        while (processed < target_documents) {
            std::shuffle(order.begin(), order.end(), random);
            for (const size_t shard_index : order) {
                const CachedParticleShard shard = read_particle_cache(
                    cache.shard_path(shard_index));
                bool failed = false;
                std::visit([&](const auto& particles) {
                    for (int32_t local = 0; local < particles.documents;
                            ++local) {
                        append_document(batch, particles, local);
                        const int64_t remaining =
                            target_documents - processed;
                        if (batch.documents >= options.fit_batch_documents
                            || batch.documents >= remaining) {
                            if (!flush_batch()) {
                                failed = true;
                                break;
                            }
                            if (processed >= target_documents) break;
                        }
                    }
                }, shard);
                if (failed) return out;
                if (processed >= target_documents) break;
            }
            if (batch.documents > 0 && processed < target_documents
                && !flush_batch()) return out;
        }
    }

    if (!subsample) {
        out.diagnostics.approximate_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now()
                - approximate_start).count();
    }
    for (int32_t update = 0; !subsample
            && update < options.fit_full_tail_updates; ++update) {
        Expectation exact = cached_particle_expectation(cache,
            candidate.model, screening,
            ExpectationRequest{false, false, true}, options.n_threads);
        ++out.diagnostics.full_data_evaluations;
        record_evaluation(candidate, options, exact);
        if (!apply_update(candidate, exact, options, shrinkage, true,
                nullptr, true)) {
            return out;
        }
        ++out.diagnostics.full_tail_updates;
    }
    out.diagnostics.document_pass_equivalents =
        out.diagnostics.full_data_evaluations
        + static_cast<double>(out.diagnostics.approximate_documents)
            / documents;
    out.diagnostics.fitting_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fit_start).count();
    candidate.terminal_pending = true;
    return out;
}

namespace {

template<class ParticleCollection>
ApproximateParticleFit fit_particle_subsample_impl(
    const ParticleCollection& particles, Model initial,
    const FitOptions& options,
    const ComponentScreeningOptions& screening,
    const RestartTrace& initialization_trace) {
    if (options.particle_fit_schedule != ParticleFitSchedule::Subsample) {
        throw std::invalid_argument(
            "Resident UAC approximate fitting supports subsample mode only");
    }
    ApproximateParticleFit out;
    out.diagnostics.schedule = ParticleFitSchedule::Subsample;
    out.diagnostics.subsample_storage = SubsampleStorage::Resident;
    out.diagnostics.subsample_memory_budget = options.fit_subsample_memory_budget;
    out.candidate = initialize_candidate(
        std::move(initial), options, initialization_trace);
    Candidate& candidate = out.candidate;
    const int32_t documents = particles.documents;
    const int32_t components = static_cast<int32_t>(
        candidate.model.weights.size());
    const auto fit_start = std::chrono::steady_clock::now();

    Expectation warmup = particle_expectation(particles, candidate.model,
        ExpectationRequest{false, false, true, true}, screening);
    ++out.diagnostics.full_data_evaluations;
    record_evaluation(candidate, options, warmup);
    if (!apply_update(candidate, warmup, options, 0.0, true,
            nullptr, true)) return out;
    out.diagnostics.full_warmup_updates = 1;
    const auto approximate_start = std::chrono::steady_clock::now();

    SubsampleAllocation allocation = allocate_subsample(warmup, options);
    populate_subsample_diagnostics(out.diagnostics, warmup, allocation, options);
    const int32_t factor_rank = candidate.model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(
            candidate.model.factor_covariances.front().factor.cols()) : -1;
    const bool screen = screening.mode != ComponentScreeningMode::Off;
    const int32_t maximum_samples = *std::max_element(
        warmup.subsample_document_samples.begin(),
        warmup.subsample_document_samples.end());
    const int32_t predicted_documents = std::max<int32_t>(1,
        static_cast<int32_t>(std::ceil(allocation.probability.dot(
            warmup.subsample_stratum_documents.cast<double>()))));
    const uint64_t fixed_bytes =
        (sizeof(uint16_t) + sizeof(uint8_t) + sizeof(int32_t))
            * static_cast<uint64_t>(documents)
        + sizeof(double) * static_cast<uint64_t>(components)
            * (components + 16);
    const uint64_t predicted_workspace = particle_expectation_peak_bytes(
        predicted_documents, components, particles.dimension, factor_rank,
        maximum_samples, screen);
    out.diagnostics.subsample_predicted_bytes =
        fixed_bytes + static_cast<uint64_t>(predicted_documents)
            * (sizeof(int32_t) + sizeof(double)) + predicted_workspace;
    if (out.diagnostics.subsample_predicted_bytes
            > options.fit_subsample_memory_budget) {
        throw std::runtime_error(
            "UAC batch indexed subsample exceeds --fit-subsample-memory-budget");
    }

    std::vector<uint8_t> selected;
    std::vector<int32_t> selected_indices;
    Eigen::VectorXd weights;
    Eigen::VectorXd target(components);
    for (int32_t c = 0; c < components; ++c) {
        target(c) = std::min<double>(
            options.fit_subsample_target,
            warmup.membership(c) / options.fit_subsample_safety_factor);
    }
    selected = select_subsample_documents(
        warmup.subsample_strata, allocation.probability, options.seed);
    auto rebuild_index = [&]() {
        selected_indices.clear();
        selected_indices.reserve(std::count(
            selected.begin(), selected.end(), uint8_t{1}));
        for (int32_t d = 0; d < documents; ++d) {
            if (selected[d]) selected_indices.push_back(d);
        }
        if (selected_indices.empty()) {
            throw std::runtime_error(
                "UAC stratified subsample selected no documents");
        }
        weights = selected_weights(
            warmup.subsample_strata, selected, allocation.probability);
        const int32_t count = static_cast<int32_t>(selected_indices.size());
        const uint64_t actual_workspace = particle_expectation_peak_bytes(
            count, components, particles.dimension, factor_rank,
            selected_maximum_samples(
                warmup.subsample_document_samples, selected), screen);
        const uint64_t actual_bytes = fixed_bytes
            + static_cast<uint64_t>(count)
                * (sizeof(int32_t) + sizeof(double))
            + actual_workspace;
        if (actual_bytes > options.fit_subsample_memory_budget) {
            throw std::runtime_error(
                "UAC batch indexed subsample exceeds memory budget");
        }
        out.diagnostics.subsample_documents = count;
        out.diagnostics.subsample_weighted_documents = weights.sum();
        out.diagnostics.subsample_maximum_weight = weights.maxCoeff();
        update_subsample_peak(out.diagnostics, actual_bytes,
            "indexed_subsample_estep");
        refresh_allocation_diagnostics(out.diagnostics, allocation);
    };
    rebuild_index();
    release_warmup_moments(warmup);

    IndexedParticleView<ParticleCollection> subsample_view(
        particles, selected_indices);
    auto subsample_expectation = [&](const Model& model) {
        return particle_expectation(subsample_view, model,
            ExpectationRequest{false, false, true}, screening, &weights);
    };
    int32_t subsample_documents = subsample_view.documents;
    auto topup_subsample = [&](const Expectation& expectation,
                               const Model& model) {
        if (out.diagnostics.subsample_topup_rounds
                >= options.fit_subsample_topup_rounds) return false;
        const Eigen::VectorXd achievable = target.cwiseMin(
            expectation.membership);
        if (!increase_topup_probabilities(warmup, achievable,
                kish_effective_membership(expectation),
                model, allocation, out.diagnostics)) return false;
        ++out.diagnostics.subsample_topup_rounds;
        selected = select_subsample_documents(
            warmup.subsample_strata, allocation.probability, options.seed);
        rebuild_index();
        subsample_view = IndexedParticleView<ParticleCollection>(
            particles, selected_indices);
        subsample_documents = subsample_view.documents;
        return true;
    };
    if (!run_subsample_updates(candidate, out.diagnostics, options,
            documents, subsample_documents, target,
            subsample_expectation, topup_subsample)) return out;
    out.diagnostics.approximate_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - approximate_start).count();
    ComponentScreeningOptions exact_screening = screening;
    exact_screening.mode = ComponentScreeningMode::Off;
    exact_screening.maximum_components = 0;
    auto exact_expectation = [&](const Model& model) {
        return particle_expectation(particles, model,
            ExpectationRequest{false, false, true}, exact_screening);
    };
    if (!run_exact_tail(candidate, out.diagnostics, options,
            exact_expectation)) return out;
    out.diagnostics.document_pass_equivalents =
        out.diagnostics.full_data_evaluations
        + static_cast<double>(out.diagnostics.approximate_documents) / documents;
    out.diagnostics.fitting_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - fit_start).count();
    candidate.terminal_pending = true;
    return out;
}

} // namespace

ApproximateParticleFit fit_particle_subsample(
    const ParticleSet& particles, Model initial, const FitOptions& options,
    const ComponentScreeningOptions& screening,
    const RestartTrace& initialization_trace) {
    return fit_particle_subsample_impl(particles, std::move(initial), options,
        screening, initialization_trace);
}

ApproximateParticleFit fit_particle_subsample(
    const RaggedParticleSet& particles, Model initial,
    const FitOptions& options,
    const ComponentScreeningOptions& screening,
    const RestartTrace& initialization_trace) {
    return fit_particle_subsample_impl(particles, std::move(initial), options,
        screening, initialization_trace);
}

} // namespace uac::detail
