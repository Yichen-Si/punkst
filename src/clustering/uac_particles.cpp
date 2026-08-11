#include "clustering/uac_particles_internal.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <type_traits>

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>

namespace uac {

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

Eigen::Map<const Eigen::VectorXi>
ParticleSet::proposal_origins_for_document(int32_t document) const {
    return Eigen::Map<const Eigen::VectorXi>(
        proposal_origins.data()
            + static_cast<int64_t>(document) * samples,
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

Eigen::Map<const Eigen::VectorXi>
RaggedParticleSet::proposal_origins_for_document(int32_t document) const {
    const int64_t offset = offsets.at(document);
    return Eigen::Map<const Eigen::VectorXi>(
        proposal_origins.data() + offset, samples_for_document(document));
}

} // namespace uac

namespace uac::detail {


ParticleSet make_particle_range(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    const PilotCache& pilot_cache,
    ProposalKind proposal_kind, int32_t samples, uint64_t seed,
    double fisher_broadening, int32_t fisher_refinement_iterations,
    int32_t n_threads,
    const ProposalScreeningPlan* screening_plan, int32_t first_document,
    int32_t documents, int32_t global_first_document) {
    const int32_t global_first = global_first_document >= 0
        ? global_first_document : first_document;
    if (samples <= 0 || data.counts.size() != data.identifiers.size()
        || data.coordinates.rows() != static_cast<Eigen::Index>(data.counts.size())
        || basis.probabilities.cols() != helmert.cols()
        || first_document < 0 || documents <= 0
        || static_cast<int64_t>(first_document) + documents
            > data.coordinates.rows()
        || !(fisher_broadening > 0.0)
        || !std::isfinite(fisher_broadening)
        || fisher_refinement_iterations <= 0) {
        throw std::invalid_argument("Invalid UAC particle input");
    }
    ParticleSet out;
    out.first_document = global_first;
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
    out.proposal_origins.resize(particle_rows);
    out.proposal_candidates.resize(out.documents);
    const uint64_t proposal_workspace = sizeof(double)
        * static_cast<uint64_t>(pilot.weights.size())
        * (static_cast<uint64_t>(out.dimension) * out.dimension
            + out.dimension + 3);
    out.proposal_workspace_bytes = proposal_workspace;
    std::atomic<int64_t> fisher_nanoseconds{0};
    std::atomic<int64_t> proposal_nanoseconds{0};
    std::atomic<int64_t> draw_nanoseconds{0};
    std::atomic<int64_t> fallback_nanoseconds{0};
    std::atomic<int64_t> fallbacks{0};
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    const auto sampling_start = std::chrono::steady_clock::now();
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, out.documents),
        [&](const tbb::blocked_range<int32_t>& range) {
            FisherWorkspace fisher_workspace;
            for (int32_t local_document = range.begin();
                    local_document < range.end(); ++local_document) {
                const int32_t document = first_document + local_document;
                const int32_t global_document = global_first + local_document;
                const Eigen::VectorXd center =
                    data.coordinates.row(document).transpose();
                const auto fisher_start = std::chrono::steady_clock::now();
                const FisherApproximation fisher = fisher_approximation_impl(
                    center, data.counts[document], basis, helmert,
                    proposal_kind, true, &fisher_workspace);
                const auto proposal_start = std::chrono::steady_clock::now();
                fisher_nanoseconds.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        proposal_start - fisher_start).count(),
                    std::memory_order_relaxed);
                const std::vector<int32_t>* candidates =
                    screening_plan && screening_plan->enabled
                    ? &screening_plan->candidates[global_document] : nullptr;
                const DocumentProposal proposal = fisher_proposal(center,
                    fisher, data.counts[document], basis, helmert,
                    proposal_kind, pilot, pilot_cache, fisher_broadening,
                    fisher_refinement_iterations, candidates);
                out.proposal_candidates[local_document] =
                    static_cast<int32_t>(proposal.weights.size());
                fallback_nanoseconds.fetch_add(static_cast<int64_t>(
                    proposal.precision_fallback_seconds * 1e9),
                    std::memory_order_relaxed);
                fallbacks.fetch_add(proposal.precision_fallbacks,
                    std::memory_order_relaxed);
                const auto draw_start = std::chrono::steady_clock::now();
                proposal_nanoseconds.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        draw_start - proposal_start).count(),
                    std::memory_order_relaxed);
                const uint64_t document_seed = hash_string(
                    seed ^ 0x9e3779b97f4a7c15ull,
                    data.identifiers[document]);
                auto values = out.values.middleRows(
                    static_cast<Eigen::Index>(local_document) * samples,
                    samples);
                draw_proposal_values(proposal, document_seed, values,
                    out.proposal_origins.data()
                        + static_cast<size_t>(local_document) * samples);
                out.log_proposal.row(local_document) =
                    proposal_log_density_rows(values, proposal).transpose();
                draw_nanoseconds.fetch_add(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - draw_start).count(),
                    std::memory_order_relaxed);
            }
        });
    out.sampling_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sampling_start).count();
    out.fisher_work_seconds = 1e-9 * fisher_nanoseconds.load();
    out.proposal_component_work_seconds =
        1e-9 * proposal_nanoseconds.load();
    out.proposal_draw_density_work_seconds =
        1e-9 * draw_nanoseconds.load();
    out.proposal_precision_fallback_seconds =
        1e-9 * fallback_nanoseconds.load();
    out.proposal_precision_fallbacks = fallbacks.load();
    out.proposal_components_constructed = std::accumulate(
        out.proposal_candidates.begin(), out.proposal_candidates.end(),
        int64_t{0});
    const int32_t active_components = static_cast<int32_t>(
        (pilot.weights.array() > 0.0).count());
    out.proposal_components_possible = static_cast<int64_t>(out.documents)
        * active_components;
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
    const AdaptiveParticleOptions& options,
    int32_t maximum_particles) {
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
    Eigen::MatrixXd standardized;
    Eigen::VectorXd log_density;
    for (int32_t c = 0; c < components; ++c) {
        if (!(model.weights(c) > 0.0)) {
            evidence(c) = -std::numeric_limits<double>::infinity();
            score(c) = -std::numeric_limits<double>::infinity();
            log_tilt.row(c).setConstant(
                -std::numeric_limits<double>::infinity());
            continue;
        }
        solvers[c].log_density_rows(values, standardized, log_density);
        log_tilt.row(c) = (base + log_density).transpose();
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
    double required = options.minimum_particles;
    AdaptiveParticleBinding binding = AdaptiveParticleBinding::Minimum;
    auto update_required = [&](double candidate,
            AdaptiveParticleBinding candidate_binding) {
        if (candidate > required) {
            required = candidate;
            binding = candidate_binding;
        }
    };
    if (options.responsibility_se_target.has_value()) {
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
                / *options.responsibility_se_target, 2.0);
        update_required(out.diagnostic.projected_responsibility_particles,
            AdaptiveParticleBinding::Responsibility);
    }
    if (options.moment_ess_target.has_value()) {
        for (const int32_t c : plausible) {
            const double relative_ess = 1.0
                / (samples * tau.row(c).squaredNorm());
            out.diagnostic.projected_moment_particles = std::max(
                out.diagnostic.projected_moment_particles,
                *options.moment_ess_target
                    / std::max(1e-12, relative_ess));
        }
        update_required(out.diagnostic.projected_moment_particles,
            AdaptiveParticleBinding::MomentEss);
    }
    int32_t selected = options.minimum_particles;
    while (selected < maximum_particles && selected < required) {
        selected = std::min(maximum_particles, selected * 2);
    }
    out.particles = selected;
    out.diagnostic.selected_particles = selected;
    out.diagnostic.binding = binding;
    return out;
}

RaggedParticleSet make_adaptive_particle_range(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal_kind, uint64_t seed, double fisher_broadening,
    int32_t fisher_refinement_iterations,
    int32_t n_threads, const Model& calibration_model,
    const AdaptiveParticleOptions& options,
    int32_t maximum_particles,
    const ProposalScreeningPlan* screening_plan,
    int32_t first_document, int32_t documents,
    int32_t global_first_document) {
    const int32_t global_first = global_first_document >= 0
        ? global_first_document : first_document;
    const int32_t total_documents =
        static_cast<int32_t>(data.coordinates.rows());
    const int32_t dimension = static_cast<int32_t>(helmert.rows());
    if (!options.enabled() || documents <= 0 || first_document < 0
        || first_document > total_documents
        || documents > total_documents - first_document
        || options.calibration_particles < 2
        || options.minimum_particles <= 0
        || options.minimum_particles < options.calibration_particles
        || maximum_particles < options.minimum_particles
        || !(fisher_broadening > 0.0)
        || !std::isfinite(fisher_broadening)
        || fisher_refinement_iterations <= 0
        || (options.responsibility_se_target.has_value()
            && !(*options.responsibility_se_target > 0.0))
        || !(options.plausible_mass > 0.0 && options.plausible_mass <= 1.0)
        || !(options.plausible_responsibility >= 0.0
            && options.plausible_responsibility <= 1.0)
        || (options.moment_ess_target.has_value()
            && !(*options.moment_ess_target > 0.0))) {
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
    out.first_document = global_first;
    out.documents = documents;
    out.dimension = dimension;
    out.maximum_samples = maximum_particles;
    out.offsets.assign(static_cast<size_t>(documents) + 1, 0);
    out.adaptive_diagnostics.resize(documents);
    out.proposal_candidates.resize(documents);
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
        * (static_cast<uint64_t>(dimension) * dimension + dimension + 3);
    std::atomic<int64_t> fisher_nanoseconds{0};
    std::atomic<int64_t> proposal_nanoseconds{0};
    std::atomic<int64_t> draw_nanoseconds{0};
    std::atomic<int64_t> fallback_nanoseconds{0};
    std::atomic<int64_t> fallbacks{0};
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    constexpr int32_t kGenerationChunk = 128;
    for (int32_t begin = 0; begin < documents; begin += kGenerationChunk) {
        const int32_t end = std::min(documents, begin + kGenerationChunk);
        const int32_t size = end - begin;
        std::vector<DocumentProposal> proposals(size);
        std::vector<int32_t> counts(size, options.minimum_particles);
        std::vector<RowMajorMatrixXd> calibration_values(size);
        std::vector<std::vector<int32_t>> calibration_origins(size);
        std::vector<Eigen::VectorXd> calibration_log_q(size);
        std::vector<Eigen::VectorXd> calibration_log_likelihood(size);
        const auto calibration_start = std::chrono::steady_clock::now();
        tbb::parallel_for(tbb::blocked_range<int32_t>(0, size),
            [&](const tbb::blocked_range<int32_t>& range) {
                FisherWorkspace fisher_workspace;
                for (int32_t local = range.begin(); local < range.end();
                        ++local) {
                    const int32_t local_document = begin + local;
                    const int32_t document = first_document + local_document;
                    const int32_t global_document =
                        global_first + local_document;
                    const Eigen::VectorXd center =
                        data.coordinates.row(document).transpose();
                    const auto fisher_start =
                        std::chrono::steady_clock::now();
                    const FisherApproximation fisher =
                        fisher_approximation_impl(center,
                            data.counts[document], basis, helmert,
                            proposal_kind, true, &fisher_workspace);
                    const auto proposal_start =
                        std::chrono::steady_clock::now();
                    fisher_nanoseconds.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            proposal_start - fisher_start).count(),
                        std::memory_order_relaxed);
                    const std::vector<int32_t>* candidates =
                        screening_plan && screening_plan->enabled
                        ? &screening_plan->candidates[global_document]
                        : nullptr;
                    proposals[local] = fisher_proposal(center, fisher,
                        data.counts[document], basis, helmert, proposal_kind,
                        pilot, pilot_cache, fisher_broadening,
                        fisher_refinement_iterations, candidates);
                    out.proposal_candidates[local_document] =
                        static_cast<int32_t>(
                            proposals[local].weights.size());
                    fallback_nanoseconds.fetch_add(static_cast<int64_t>(
                        proposals[local].precision_fallback_seconds * 1e9),
                        std::memory_order_relaxed);
                    fallbacks.fetch_add(
                        proposals[local].precision_fallbacks,
                        std::memory_order_relaxed);
                    proposal_nanoseconds.fetch_add(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now()
                                - proposal_start).count(),
                        std::memory_order_relaxed);
                    RowMajorMatrixXd& calibration =
                        calibration_values[local];
                    calibration.resize(
                        options.calibration_particles, dimension);
                    calibration_origins[local].resize(
                        options.calibration_particles);
                    const uint64_t calibration_seed = hash_string(
                        seed ^ 0x6a09e667f3bcc909ull,
                        data.identifiers[document]);
                    draw_proposal_values(proposals[local], calibration_seed,
                        calibration, calibration_origins[local].data());
                    calibration_log_q[local] = proposal_log_density_rows(
                        calibration, proposals[local]);
                    calibration_log_likelihood[local] =
                        count_log_likelihood_rows(calibration,
                            data.counts[document], basis, helmert);
                    const AdaptiveCountResult allocation =
                        adaptive_particle_count(calibration,
                            calibration_log_likelihood[local],
                            calibration_log_q[local], calibration_model,
                            calibration_solvers, options,
                            maximum_particles);
                    counts[local] = allocation.particles;
                    out.adaptive_diagnostics[local_document] =
                        allocation.diagnostic;
                }
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
        out.proposal_origins.resize(total_samples);
        const auto sampling_start = std::chrono::steady_clock::now();
        tbb::parallel_for(int32_t{0}, size, [&](int32_t local) {
            const int32_t local_document = begin + local;
            const int32_t document = first_document + local_document;
            const int64_t offset = out.offsets[local_document];
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
            std::copy(calibration_origins[local].begin(),
                calibration_origins[local].end(),
                out.proposal_origins.begin() + offset);
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
            draw_proposal_values(proposals[local], document_seed, additional,
                out.proposal_origins.data() + offset + calibration_samples);
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
            const int32_t local_document = begin + local;
            const int32_t document = first_document + local_document;
            const int64_t offset = out.offsets[local_document];
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
    out.proposal_precision_fallback_seconds =
        1e-9 * fallback_nanoseconds.load();
    out.proposal_precision_fallbacks = fallbacks.load();
    out.proposal_components_constructed = std::accumulate(
        out.proposal_candidates.begin(), out.proposal_candidates.end(),
        int64_t{0});
    out.proposal_components_possible = static_cast<int64_t>(documents)
        * (pilot.weights.array() > 0.0).count();
    return out;
}

RaggedParticleSet make_adaptive_particles(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal_kind, uint64_t seed, double fisher_broadening,
    int32_t fisher_refinement_iterations, int32_t n_threads,
    const Model& calibration_model,
    const AdaptiveParticleOptions& options,
    int32_t maximum_particles,
    const ProposalScreeningPlan* screening_plan) {
    return make_adaptive_particle_range(data, basis, helmert, pilot,
        pilot_cache, proposal_kind, seed, fisher_broadening,
        fisher_refinement_iterations, n_threads,
        calibration_model, options, maximum_particles, screening_plan, 0,
        static_cast<int32_t>(data.coordinates.rows()));
}

} // namespace uac::detail

namespace uac {

ParticleSet make_particles(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    ProposalKind proposal_kind, int32_t samples, uint64_t seed,
    double fisher_broadening, int32_t n_threads,
    int32_t fisher_refinement_iterations) {
    const detail::PilotCache pilot_cache(pilot);
    return detail::make_particle_range(data, basis, helmert, pilot, pilot_cache,
        proposal_kind,
        samples, seed, fisher_broadening, fisher_refinement_iterations,
        n_threads, nullptr, 0,
        static_cast<int32_t>(data.coordinates.rows()));
}

} // namespace uac
