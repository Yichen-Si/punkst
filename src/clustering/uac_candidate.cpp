#include "clustering/uac_initialization_internal.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <type_traits>

namespace uac::detail {




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

void record_trace_point(RestartTrace& trace, const FitOptions& options,
    TraceEvent event, int32_t completed_updates, double objective,
    int32_t active_components,
    double relative_objective_change, double responsibility_change,
    double variance_change,
    double mean_responsibility_entropy) {
    RestartTrace::Point point;
    point.event = event;
    point.completed_updates = completed_updates;
    point.objective = objective;
    point.active_components = active_components;
    point.relative_objective_change = relative_objective_change;
    point.mean_max_responsibility_change = responsibility_change;
    point.median_absolute_relative_variance_change = variance_change;
    point.mean_responsibility_entropy = mean_responsibility_entropy;
    trace.points.push_back(std::move(point));
    if (options.iteration_callback) {
        options.iteration_callback({trace.phase, event, trace.start,
            completed_updates, relative_objective_change,
            responsibility_change, variance_change,
            mean_responsibility_entropy});
    }
}

void accumulate_estep_work(
    RestartTrace& trace, const Expectation& expectation) {
    trace.estep_work.gaussian_seconds += expectation.gaussian_seconds;
    trace.estep_work.component_bound_seconds +=
        expectation.component_bound_seconds;
    trace.estep_work.moment_seconds += expectation.moment_seconds;
    trace.estep_work.document_evaluations += expectation.documents;
    trace.estep_work.evaluated_component_documents +=
        expectation.evaluated_component_documents;
    trace.estep_work.possible_component_documents +=
        expectation.possible_component_documents;
    trace.estep_work.full_component_documents +=
        expectation.full_component_documents;
    trace.estep_work.component_bound_violations +=
        expectation.component_bound_violations;
}

Candidate fit_map_candidate(const Dataset& data, Model initial,
    const FitOptions& options, const RestartTrace& metadata) {
    Candidate out;
    out.trace = metadata;
    out.trace.handoff = HandoffMode::Map;
    out.trace.phase = TracePhase::PointMapEm;
    out.model = std::move(initial);
    ComponentScreeningOptions map_screening = options.component_screening;
    if (map_screening.mode == ComponentScreeningMode::Auto) {
        const bool enabled = resolve_map_component_screening(
            data, out.model, map_screening,
            static_cast<uint64_t>(metadata.seed));
        apply_auto_component_screening_resolution(
            map_screening, enabled);
    }
    const double shrinkage = options.adaptive_covariance_shrinkage
        ? options.covariance_shrinkage_strength : 0.0;
    RowMajorMatrixXd previous_responsibilities;
    double converged_log_likelihood =
        -std::numeric_limits<double>::infinity();
    double previous_objective_lower =
        -std::numeric_limits<double>::infinity();
    double previous_objective_upper =
        -std::numeric_limits<double>::infinity();
    double previous_omitted_mass = 0.0;
    bool reuse_converged_expectation = false;
    for (int32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        Expectation expectation = map_expectation(data, out.model,
            ExpectationRequest{true, false, true}, map_screening);
        accumulate_estep_work(out.trace, expectation);
        const double objective = expectation.log_likelihood
            + covariance_prior(out.model, shrinkage);
        const double objective_upper = expectation.log_likelihood_upper
            + covariance_prior(out.model, shrinkage);
        double relative_change = std::numeric_limits<double>::quiet_NaN();
        double responsibility_change =
            std::numeric_limits<double>::quiet_NaN();
        if (previous_responsibilities.size() > 0) {
            responsibility_change = mean_max_responsibility_change(
                expectation.responsibilities, previous_responsibilities)
                + expectation.maximum_omitted_component_mass
                + previous_omitted_mass;
        }
        if (std::isfinite(previous_objective_lower)) {
            relative_change = std::max(
                std::abs(objective_upper - previous_objective_lower),
                std::abs(objective - previous_objective_upper))
                / std::max({1.0, std::abs(previous_objective_lower),
                    std::abs(previous_objective_upper)});
        }
        record_trace_point(out.trace, options, TraceEvent::Evaluation,
            out.trace.completed_updates, objective,
            active_component_count(out.model), relative_change,
            responsibility_change);
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
        previous_objective_lower = objective;
        previous_objective_upper = objective_upper;
        previous_omitted_mass =
            expectation.maximum_omitted_component_mass;
        const ModelUpdate update = update_model(out.model, expectation,
            shrinkage, options.covariance_floor);
        if (!update.valid) {
            out.trace.collapsed = true;
            return out;
        }
        ++out.trace.completed_updates;
    }
    if (reuse_converged_expectation) {
        out.objective = converged_log_likelihood;
    } else {
        const Expectation final_expectation = map_expectation(data, out.model,
            ExpectationRequest{false, false, false}, map_screening);
        accumulate_estep_work(out.trace, final_expectation);
        out.objective = final_expectation.log_likelihood;
    }
    out.trace.selection_objective = out.objective;
    record_trace_point(out.trace, options, TraceEvent::Terminal,
        out.trace.completed_updates,
        out.objective + covariance_prior(out.model, shrinkage),
        active_component_count(out.model),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());
    out.trace.succeeded = true;
    return out;
}



Candidate fit_particle_candidate(
    const std::function<Expectation(const Model&)>& expectation_function,
    Model initial, const FitOptions& options,
    const RestartTrace& initialization_trace) {
    Candidate out;
    out.trace = initialization_trace;
    out.trace.points.clear();
    out.trace.model_trace.clear();
    out.trace.estep_work = {};
    out.trace.converged = false;
    out.trace.collapsed = false;
    out.trace.handoff = HandoffMode::Particle;
    out.trace.phase = TracePhase::ParticleEm;
    out.trace.fixed_em_iteration_schedule =
        options.particle_em_fixed_iterations > 0;
    out.trace.completed_updates = 0;
    out.model = std::move(initial);
    const double shrinkage = options.adaptive_covariance_shrinkage
        ? options.covariance_shrinkage_strength : 0.0;
    RowMajorMatrixXd previous_responsibilities;
    double previous_objective_lower =
        -std::numeric_limits<double>::infinity();
    double previous_objective_upper =
        -std::numeric_limits<double>::infinity();
    double previous_omitted_mass = 0.0;
    std::optional<Model> previous_variance_model;
    bool adaptive_update_completed = false;
    int32_t model_iteration = 0;
    auto record_model = [&](bool final, double update_shrinkage_strength) {
        if (!options.capture_model_trace) return;
        ModelTraceEntry entry;
        entry.completed_updates = model_iteration;
        entry.event = final ? TraceEvent::Terminal : TraceEvent::Evaluation;
        entry.update_shrinkage_strength = update_shrinkage_strength;
        entry.model = out.model;
        out.trace.model_trace.push_back(std::move(entry));
    };
    if (options.adaptive_covariance_shrinkage) {
        record_model(false, 0.0);
        Expectation bootstrap = expectation_function(out.model);
        accumulate_estep_work(out.trace, bootstrap);
        record_trace_point(out.trace, options, TraceEvent::Evaluation,
            out.trace.completed_updates, bootstrap.log_likelihood,
            active_component_count(out.model),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
        previous_responsibilities = bootstrap.responsibilities;
        previous_objective_lower = bootstrap.log_likelihood;
        previous_objective_upper = bootstrap.log_likelihood_upper;
        previous_omitted_mass =
            bootstrap.maximum_omitted_component_mass;
        previous_variance_model = out.model;
        const ModelUpdate update = update_model(out.model, bootstrap, 0.0,
            options.covariance_floor);
        if (!update.valid) {
            out.trace.collapsed = true;
            record_model(true,
                std::numeric_limits<double>::quiet_NaN());
            return out;
        }
        ++model_iteration;
        ++out.trace.completed_updates;
    }
    const int32_t update_budget = options.particle_em_fixed_iterations > 0
        ? options.particle_em_fixed_iterations
        : options.max_iterations;
    const int32_t regular_iterations = std::max(
        0, update_budget - out.trace.completed_updates);
    for (int32_t iteration = 0; iteration < regular_iterations; ++iteration) {
        record_model(false, shrinkage);
        Expectation expectation = expectation_function(out.model);
        accumulate_estep_work(out.trace, expectation);
        const double objective = expectation.log_likelihood;
        const double objective_upper = expectation.log_likelihood_upper;
        double relative_change = std::numeric_limits<double>::quiet_NaN();
        double responsibility_change =
            std::numeric_limits<double>::quiet_NaN();
        double variance_change =
            std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(previous_objective_lower)) {
            relative_change = std::max(
                std::abs(objective_upper - previous_objective_lower),
                std::abs(objective - previous_objective_upper))
                / std::max({1.0, std::abs(previous_objective_lower),
                    std::abs(previous_objective_upper)});
        }
        if (expectation.has_responsibility_change) {
            responsibility_change =
                expectation.mean_max_responsibility_change
                + expectation.maximum_omitted_component_mass
                + previous_omitted_mass;
        } else if (previous_responsibilities.size() > 0) {
            responsibility_change = mean_max_responsibility_change(
                expectation.responsibilities, previous_responsibilities)
                + expectation.maximum_omitted_component_mass
                + previous_omitted_mass;
        }
        if (previous_variance_model.has_value()) {
            variance_change = median_absolute_relative_variance_change(
                out.model, *previous_variance_model,
                options.covariance_floor);
        }
        record_trace_point(out.trace, options, TraceEvent::Evaluation,
            out.trace.completed_updates, objective,
            active_component_count(out.model), relative_change,
            responsibility_change, variance_change);
        const bool convergence_eligible =
            !options.adaptive_covariance_shrinkage
            || adaptive_update_completed;
        const bool variance_converged =
            options.particle_variance_change_tolerance == 0.0
            || (std::isfinite(variance_change)
                && variance_change
                    < options.particle_variance_change_tolerance);
        if (!out.trace.fixed_em_iteration_schedule
            && convergence_eligible && std::isfinite(relative_change)
            && variance_converged
            && (relative_change < options.objective_change_tolerance
                || responsibility_change
                    < options.responsibility_change_tolerance)) {
            out.trace.converged = true;
            break;
        }
        previous_responsibilities = expectation.responsibilities;
        previous_objective_lower = objective;
        previous_objective_upper = objective_upper;
        previous_omitted_mass =
            expectation.maximum_omitted_component_mass;
        previous_variance_model = out.model;
        const ModelUpdate update = update_model(out.model, expectation,
            shrinkage, options.covariance_floor,
            options.adaptive_covariance_shrinkage);
        if (!update.valid) {
            out.trace.collapsed = true;
            record_model(true,
                std::numeric_limits<double>::quiet_NaN());
            return out;
        }
        ++model_iteration;
        ++out.trace.completed_updates;
        if (options.adaptive_covariance_shrinkage) {
            adaptive_update_completed = true;
        }
    }
    out.terminal_pending = true;
    return out;
}

void finalize_particle_candidate(Candidate& candidate,
    const Expectation& terminal, const FitOptions& options) {
    if (!candidate.terminal_pending) return;
    accumulate_estep_work(candidate.trace, terminal);
    candidate.objective = terminal.log_likelihood;
    record_trace_point(candidate.trace, options, TraceEvent::Terminal,
        candidate.trace.completed_updates, candidate.objective,
        active_component_count(candidate.model),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());
    if (options.capture_model_trace) {
        ModelTraceEntry entry;
        entry.completed_updates = candidate.trace.completed_updates;
        entry.event = TraceEvent::Terminal;
        entry.update_shrinkage_strength =
            std::numeric_limits<double>::quiet_NaN();
        entry.model = candidate.model;
        candidate.trace.model_trace.push_back(std::move(entry));
    }
    candidate.trace.succeeded = true;
    candidate.terminal_pending = false;
}

template<class ParticleCollection>
ScoreResult score_particles_impl(
    const ParticleCollection& particles, const Model& model,
    const ComponentScreeningOptions& screening,
    Expectation* terminal_expectation) {
    ScoreResult out;
    Expectation expectation = particle_expectation(particles, model,
        ExpectationRequest{true, true, false}, screening);
    out.responsibilities = std::move(expectation.responsibilities);
    out.particle_diagnostics = std::move(expectation.particle_diagnostics);
    out.gaussian_seconds = expectation.gaussian_seconds;
    out.moment_seconds = expectation.moment_seconds;
    out.estimated_peak_expectation_workspace_bytes =
        expectation.peak_workspace_bytes;
    out.sampling_seconds = particles.sampling_seconds;
    out.likelihood_seconds = particles.likelihood_seconds;
    out.fisher_work_seconds = particles.fisher_work_seconds;
    out.proposal_component_work_seconds =
        particles.proposal_component_work_seconds;
    out.proposal_draw_density_work_seconds =
        particles.proposal_draw_density_work_seconds;
    out.proposal_precision_fallback_seconds =
        particles.proposal_precision_fallback_seconds;
    out.proposal_precision_fallbacks =
        particles.proposal_precision_fallbacks;
    out.estimated_peak_proposal_workspace_bytes =
        particles.proposal_workspace_bytes;
    out.component_screening_options = screening;
    out.particle_component_screening =
        screening.mode != ComponentScreeningMode::Off;
    out.terminal_component_screening =
        screening.mode != ComponentScreeningMode::Off;
    out.component_bound_seconds = expectation.component_bound_seconds;
    out.evaluated_component_documents =
        expectation.evaluated_component_documents;
    out.possible_component_documents =
        expectation.possible_component_documents;
    out.full_component_documents = expectation.full_component_documents;
    out.component_bound_violations =
        expectation.component_bound_violations;
    out.maximum_omitted_component_mass =
        expectation.maximum_omitted_component_mass;
    out.mean_omitted_component_mass = expectation.documents > 0
        ? expectation.omitted_component_mass_sum / expectation.documents
        : 0.0;
    out.per_document_evaluated_components =
        std::move(expectation.per_document_evaluated_components);
    out.per_document_omitted_component_mass =
        std::move(expectation.per_document_omitted_component_mass);
    out.proposal_components_constructed =
        particles.proposal_components_constructed;
    out.proposal_components_possible =
        particles.proposal_components_possible;
    out.per_document_proposal_components = particles.proposal_candidates;
    int64_t total_samples = 0;
    out.per_document_particles.resize(particles.documents);
    for (int32_t d = 0; d < particles.documents; ++d) {
        const int32_t samples = particles.samples_for_document(d);
        out.per_document_particles[d] = samples;
        total_samples += samples;
    }
    out.particle_samples = total_samples;
    out.resident_particle_bytes = sizeof(double)
            * static_cast<uint64_t>(total_samples)
            * (particles.dimension + 2)
        + sizeof(int32_t) * (
            static_cast<uint64_t>(total_samples)
            + static_cast<uint64_t>(particles.documents));
    if constexpr (std::is_same_v<ParticleCollection, RaggedParticleSet>) {
        out.resident_particle_bytes += sizeof(int64_t)
            * static_cast<uint64_t>(particles.offsets.size());
    }
    out.particle_generation_passes = 1;
    if (terminal_expectation) {
        *terminal_expectation = std::move(expectation);
    }
    return out;
}

ScoreResult score_particles(const ParticleSet& particles, const Model& model,
    const ComponentScreeningOptions& screening,
    Expectation* terminal_expectation) {
    return score_particles_impl(
        particles, model, screening, terminal_expectation);
}

ScoreResult score_particles(
    const RaggedParticleSet& particles, const Model& model,
    const ComponentScreeningOptions& screening,
    Expectation* terminal_expectation) {
    ScoreResult out = score_particles_impl(
        particles, model, screening, terminal_expectation);
    out.calibration_seconds = particles.calibration_seconds;
    out.calibration_samples = particles.calibration_samples;
    out.reused_calibration_samples = particles.reused_calibration_samples;
    out.adaptive_particle_diagnostics = particles.adaptive_diagnostics;
    return out;
}

} // namespace uac::detail
