#include "clustering/uac_stream.hpp"

#include <fstream>
#include <stdexcept>

namespace uac {
namespace {

template<class Function>
void read_score_rows(const ScoreResult& score, Function&& function) {
    if (score.responsibilities.size() > 0) {
        for (Eigen::Index d = 0;
                d < score.responsibilities.rows(); ++d) {
            function(static_cast<int64_t>(d),
                score.responsibilities.row(d).transpose());
        }
        return;
    }
    std::ifstream in(score.responsibility_sidecar, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "Cannot read streaming UAC responsibility sidecar");
    }
    Eigen::VectorXd row(score.scored_components);
    for (int64_t d = 0; d < score.scored_documents; ++d) {
        in.read(reinterpret_cast<char*>(row.data()),
            sizeof(double) * row.size());
        if (!in) {
            throw std::runtime_error(
                "Truncated streaming UAC responsibility sidecar");
        }
        function(d, row);
    }
}

void emit_score(const Dataset& data, const ScoreResult& score,
    int32_t components, StreamingScoreSink* sink) {
    if (!sink) return;
    sink->begin(static_cast<int64_t>(data.identifiers.size()), components);
    read_score_rows(score,
        [&](int64_t d, const Eigen::VectorXd& responsibility) {
        StreamingScoreRow row;
        row.document = d;
        row.identifier = data.identifiers[d];
        row.raw_total = data.raw_totals.size() ? data.raw_totals(d) : 0.0;
        row.effective_total = data.effective_totals.size()
            ? data.effective_totals(d) : 0.0;
        row.responsibilities = responsibility;
        if (d < static_cast<int64_t>(
                score.particle_diagnostics.size())) {
            row.particle_diagnostic =
                score.particle_diagnostics[d];
        }
        if (d < static_cast<int64_t>(
                score.adaptive_particle_diagnostics.size())) {
            row.adaptive_particle_diagnostic =
                score.adaptive_particle_diagnostics[d];
        }
        if (d < static_cast<int64_t>(
                score.per_document_particles.size())) {
            row.particles = score.per_document_particles[d];
        }
        if (d < static_cast<int64_t>(
                score.per_document_proposal_components.size())) {
            row.proposal_components =
                score.per_document_proposal_components[d];
        }
        if (d < static_cast<int64_t>(
                score.per_document_evaluated_components.size())) {
            row.evaluated_components =
                score.per_document_evaluated_components[d];
        }
        if (d < static_cast<int64_t>(
                score.per_document_omitted_component_mass.size())) {
            row.omitted_component_mass =
                score.per_document_omitted_component_mass[d];
        }
        sink->write(row);
    });
    sink->end();
}

} // namespace

void DocumentBlock::clear() {
    first_document = 0;
    identifiers.clear();
    counts.clear();
    raw_totals.resize(0);
    effective_totals.resize(0);
}

int32_t DocumentBlock::size() const {
    return static_cast<int32_t>(identifiers.size());
}

StreamingFitResult fit_streaming(const Dataset& data, const Basis& basis,
    const FitOptions& options, StreamingScoreSink* sink) {
    FitOptions configured = options;
    configured.handoff = HandoffMode::Particle;
    configured.particle_engine = ParticleEngine::Stream;
    configured.streaming.count_storage = StreamingCountStorage::Source;
    FitResult fitted = fit(data, &basis, configured);

    StreamingFitResult out;
    out.model = fitted.model;
    out.pilot = fitted.pilot;
    out.traces = fitted.traces;
    out.converged = fitted.converged;
    out.selected_start = fitted.selected_start;
    out.selected_start_method = fitted.selected_start_method;
    out.selected_leiden_resolution = fitted.selected_leiden_resolution;
    out.score.documents = static_cast<int64_t>(data.identifiers.size());
    out.score.effective_membership =
        fitted.score.effective_membership;
    emit_score(data, fitted.score,
        static_cast<int32_t>(fitted.model.weights.size()), sink);
    out.score.diagnostics = std::move(fitted.score);
    return out;
}

StreamingScoreSummary score_particle_streaming(const Dataset& data,
    const Basis& basis, const State& state,
    const ParticleScoreOptions& options, StreamingScoreSink* sink) {
    ParticleScoreOptions configured = options;
    configured.particle_engine = ParticleEngine::Stream;
    configured.streaming.count_storage = StreamingCountStorage::Source;
    ScoreResult score = score_particle(
        data, basis, state, configured);
    StreamingScoreSummary out;
    out.documents = static_cast<int64_t>(data.identifiers.size());
    out.effective_membership = score.effective_membership;
    emit_score(data, score,
        static_cast<int32_t>(state.model.weights.size()), sink);
    out.diagnostics = std::move(score);
    return out;
}

State make_state(const StreamingFitResult& fit_result,
    const FitOptions& options, const StateMetadata& metadata) {
    FitResult ordinary;
    ordinary.model = fit_result.model;
    ordinary.pilot = fit_result.pilot;
    ordinary.traces = fit_result.traces;
    ordinary.converged = fit_result.converged;
    ordinary.selected_start = fit_result.selected_start;
    ordinary.selected_start_method = fit_result.selected_start_method;
    ordinary.selected_leiden_resolution =
        fit_result.selected_leiden_resolution;
    return make_state(ordinary, options, metadata);
}

} // namespace uac
