#pragma once

#include "clustering/uac.hpp"

#include <filesystem>
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

class IndexedDocumentSource : public DocumentBlockSource {
public:
    virtual int64_t documents() const = 0;
    virtual int64_t features() const = 0;
    virtual uint64_t storage_bytes() const = 0;
    virtual uint64_t content_checksum() const = 0;
    virtual uint64_t peak_block_bytes() const = 0;
    virtual void read_range(
        int64_t first_document, int32_t documents,
        DocumentBlock& block) const = 0;
};

class BinaryDocumentSpoolWriter {
public:
    BinaryDocumentSpoolWriter(
        const std::filesystem::path& path, int32_t features);
    ~BinaryDocumentSpoolWriter();
    BinaryDocumentSpoolWriter(BinaryDocumentSpoolWriter&&) noexcept;
    BinaryDocumentSpoolWriter& operator=(
        BinaryDocumentSpoolWriter&&) noexcept;
    BinaryDocumentSpoolWriter(const BinaryDocumentSpoolWriter&) = delete;
    BinaryDocumentSpoolWriter& operator=(
        const BinaryDocumentSpoolWriter&) = delete;

    void append(const std::string& identifier, const Document& document,
        double raw_total, double effective_total);
    std::unique_ptr<IndexedDocumentSource> finish(
        bool remove_on_destruction = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<IndexedDocumentSource> open_binary_document_spool(
    const std::filesystem::path& path,
    bool remove_on_destruction = false);

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

// Indexed-count entry points keep only identifiers, centers, coordinates,
// and totals in Dataset. Count-dependent work reads bounded ranges from source.
FitResult fit_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source, const FitOptions& options);
ScoreResult score_particle_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source, const State& state,
    const ParticleScoreOptions& options);

State make_state(const StreamingFitResult& fit, const FitOptions& options,
    const StateMetadata& metadata);

} // namespace uac
