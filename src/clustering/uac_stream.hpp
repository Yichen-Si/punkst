#pragma once

#include "clustering/uac.hpp"

#include <memory>

namespace uac {

struct DocumentBlock {
    int64_t first_document = 0;
    std::vector<std::string> identifiers;
    std::vector<Document> counts;
    Eigen::VectorXd raw_totals;
    Eigen::VectorXd effective_totals;

    void clear();
    int32_t size() const;
};

class DocumentBlockSource {
public:
    virtual ~DocumentBlockSource() = default;
    virtual void reset() = 0;
    virtual bool next(DocumentBlock& block, int32_t maximum_documents) = 0;
};

struct StreamingScoreRow {
    int64_t document = 0;
    std::string identifier;
    double raw_total = 0.0;
    double effective_total = 0.0;
    Eigen::VectorXd responsibilities;
    ParticleDiagnostic particle_diagnostic;
    AdaptiveParticleDiagnostic adaptive_particle_diagnostic;
    int32_t particles = 0;
    int32_t proposal_components = 0;
    int32_t evaluated_components = 0;
    double omitted_component_mass = 0.0;
};

class StreamingScoreSink {
public:
    virtual ~StreamingScoreSink() = default;
    virtual void begin(int64_t documents, int32_t components) = 0;
    virtual void write(const StreamingScoreRow& row) = 0;
    virtual void end() = 0;
};

struct StreamingScoreSummary {
    Eigen::VectorXd effective_membership;
    ScoreResult diagnostics;
    int64_t documents = 0;
};

struct StreamingFitResult {
    Model model;
    Pilot pilot;
    std::vector<RestartTrace> traces;
    StreamingScoreSummary score;
    bool converged = false;
    int32_t selected_start = -1;
    StartMethod selected_start_method = StartMethod::KMeans;
    double selected_leiden_resolution = 0.0;
};

// Dataset-backed entry points are useful to embedders that already have
// preprocessed documents. The command-line path may provide a resettable
// DocumentBlockSource instead.
StreamingFitResult fit_streaming(const Dataset& data, const Basis& basis,
    const FitOptions& options, StreamingScoreSink* sink = nullptr);
StreamingScoreSummary score_particle_streaming(const Dataset& data,
    const Basis& basis, const State& state,
    const ParticleScoreOptions& options, StreamingScoreSink* sink = nullptr);

State make_state(const StreamingFitResult& fit, const FitOptions& options,
    const StateMetadata& metadata);

} // namespace uac
