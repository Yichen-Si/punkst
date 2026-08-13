#pragma once

#include "clustering/uac.hpp"
#include "document_spool.hpp"

#include <filesystem>
#include <memory>

namespace uac {

using punkst::DocumentBlock;
using punkst::DocumentBatchSink;
using punkst::DocumentBlockSource;
using punkst::IndexedDocumentSource;
using punkst::DocumentSpoolMode;
using punkst::BinaryDocumentSpoolWriter;
using punkst::open_binary_document_spool;

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
    std::vector<InitializationPartition> initialization_partitions;
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

// Indexed-count entry points keep only identifiers, centers, coordinates,
// and totals in Dataset. Count-dependent work reads bounded ranges from source.
FitResult fit_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source, const FitOptions& options);
FitResult fit_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const FitOptions& options);
ScoreResult score_particle_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source, const State& state,
    const ParticleScoreOptions& options);

State make_state(const StreamingFitResult& fit, const FitOptions& options,
    const StateMetadata& metadata);

} // namespace uac
