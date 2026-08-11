#pragma once

#include "clustering/uac_cache_internal.hpp"
#include "clustering/uac_initialization_internal.hpp"

namespace uac::detail {

double model_covariance_frobenius_norm(
    const Model& model, int32_t component);
double model_covariance_frobenius_difference(
    const Model& before, const Model& after, int32_t component);
double model_parameter_change(const Model& before, const Model& after);

struct ApproximateParticleFit {
    Candidate candidate;
    FitScheduleDiagnostics diagnostics;
    std::optional<ScoreResult> terminal_score;
    std::optional<Expectation> terminal_expectation;
};

ApproximateParticleFit fit_cached_particle_approximate(
    const ParticleCache& cache, Model initial, const FitOptions& options,
    const ComponentScreeningOptions& screening,
    const RestartTrace& initialization_trace);
ApproximateParticleFit fit_particle_subsample(
    const ParticleSet& particles, Model initial, const FitOptions& options,
    const ComponentScreeningOptions& screening,
    const RestartTrace& initialization_trace);
ApproximateParticleFit fit_particle_subsample(
    const RaggedParticleSet& particles, Model initial,
    const FitOptions& options,
    const ComponentScreeningOptions& screening,
    const RestartTrace& initialization_trace);

} // namespace uac::detail
