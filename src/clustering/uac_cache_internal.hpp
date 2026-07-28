#pragma once

#include "clustering/uac_expectation_internal.hpp"
#include "utils_sys.hpp"

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <variant>
#include <vector>

namespace uac::detail {

class ScoreTemporaryStorage {
public:
    explicit ScoreTemporaryStorage(const std::filesystem::path& parent)
        : directory(parent) {}

    ScopedTempDir directory;
};

uint64_t particle_set_bytes(const ParticleSet& particles);
uint64_t particle_set_bytes(const RaggedParticleSet& particles);

struct ParticleCacheMetrics {
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    double fisher_work_seconds = 0.0;
    double proposal_component_work_seconds = 0.0;
    double proposal_draw_density_work_seconds = 0.0;
    double proposal_precision_fallback_seconds = 0.0;
    double calibration_seconds = 0.0;
    int64_t proposal_precision_fallbacks = 0;
    uint64_t peak_bytes = 0;
    uint64_t proposal_workspace_bytes = 0;
    int64_t proposal_components_constructed = 0;
    int64_t proposal_components_possible = 0;
    int64_t calibration_samples = 0;
    int64_t reused_calibration_samples = 0;
    int32_t generation_passes = 0;

    void add(const ParticleSet& particles);
    void add(const RaggedParticleSet& particles);
};

class CacheEntryLock;

class ParticleCache {
public:
    bool auto_screening_enabled() const {
        return auto_component_screening_enabled_;
    }
    const std::filesystem::path& work_directory() const {
        return temporary_storage_->directory.path;
    }

private:
    std::filesystem::path directory;
    std::vector<std::filesystem::path> shards;
    std::vector<int32_t> first_documents;
    std::vector<int32_t> document_counts;
    std::vector<int32_t> arithmetic_shards;
    ParticleCacheMetrics metrics;
    int32_t documents = 0;
    int32_t dimension = 0;
    bool adaptive = false;
    bool reused = false;
    int32_t rebuilds = 0;
    bool auto_component_screening_enabled_ = false;
    uint64_t bytes = 0;
    StreamingParticleStorage storage = StreamingParticleStorage::Positions;
    std::shared_ptr<ScoreTemporaryStorage> temporary_storage_;
    std::shared_ptr<CacheEntryLock> entry_lock;

    friend ParticleCache open_or_build_particle_cache(
        const Dataset&, const Basis&,
        const Eigen::Ref<const Eigen::MatrixXd>&, const Pilot&,
        const PilotCache&, ProposalKind, int32_t, uint64_t, double,
        int32_t, const Model&, const AdaptiveParticleOptions&,
        const ProposalScreeningPlan*, const ComponentScreeningOptions&,
        const StreamingOptions&, const IndexedDocumentSource*);
    friend Expectation cached_particle_expectation(
        const ParticleCache&, const Model&,
        const ComponentScreeningOptions&, const ExpectationRequest&,
        int32_t, struct CachedDocumentMetadata*,
        class CachedResponsibilityState*, const std::filesystem::path*,
        Eigen::VectorXd*);
    friend ScoreResult score_particle_cache(
        const ParticleCache&, const Model&,
        const ComponentScreeningOptions&, bool, int32_t, Expectation*);
};

using CachedParticleShard = std::variant<ParticleSet, RaggedParticleSet>;
class ParticleCacheCorruption : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};
CachedParticleShard read_particle_cache(
    const std::filesystem::path& path);

ParticleCache open_or_build_particle_cache(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal, int32_t maximum_samples, uint64_t seed,
    double broadening, int32_t n_threads, const Model& initial_model,
    const AdaptiveParticleOptions& adaptive,
    const ProposalScreeningPlan* proposal_screening,
    const ComponentScreeningOptions& screening,
    const StreamingOptions& options,
    const IndexedDocumentSource* count_source = nullptr);

class CachedResponsibilityState {
public:
    explicit CachedResponsibilityState(
        const std::filesystem::path& directory);
    ~CachedResponsibilityState();

    bool has_previous() const;
    const std::filesystem::path& previous_path() const;
    const std::filesystem::path& current_path() const;
    void commit();

private:
    std::filesystem::path previous_;
    std::filesystem::path current_;
    bool has_previous_ = false;
};

struct CachedDocumentMetadata;
Expectation cached_particle_expectation(const ParticleCache& cache,
    const Model& model, const ComponentScreeningOptions& screening,
    const ExpectationRequest& request, int32_t n_threads,
    CachedDocumentMetadata* metadata = nullptr,
    CachedResponsibilityState* responsibility_state = nullptr,
    const std::filesystem::path* responsibility_spool = nullptr,
    Eigen::VectorXd* effective_membership = nullptr);
ScoreResult score_particle_cache(const ParticleCache& cache,
    const Model& model, const ComponentScreeningOptions& screening,
    bool materialize_responsibilities, int32_t n_threads,
    Expectation* terminal_expectation = nullptr);

} // namespace uac::detail
