#pragma once

#include "dataunits.hpp"
#include "clustering_core/cosine_clustering.hpp"
#include "clustering/low_rank_covariance.hpp"
#include "numerical_utils.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace uac {

namespace detail {
class ScoreTemporaryStorage;
}

enum class HandoffMode {
    Map,
    Particle,
};

enum class ProposalKind {
    ExactFisher,
    SparseEmpiricalFisher,
};

enum class StartMethod {
    KMeans,
    Leiden,
};

enum class TracePhase {
    CorrectedMomScore,
    PointMapEm,
    ParticleEm,
};

enum class TraceEvent {
    CandidateScore,
    Evaluation,
    Terminal,
    Failure,
};

enum class CovarianceKind {
    Dense,
    FactorAnalytic,
};

enum class AdaptiveParticleBinding {
    Minimum,
    Responsibility,
    MomentEss,
};

enum class ComponentScreeningMode {
    Off,
    On,
    Auto,
};

enum class ParticleEngine {
    Batch,
    Stream,
};

enum class StreamingCountStorage {
    Source,
    Memory,
};

enum class StreamingParticleStorage {
    Auto,
    Factors,
    Positions,
};

const char* handoff_name(HandoffMode value);
const char* proposal_name(ProposalKind value);
const char* start_method_name(StartMethod value);
const char* trace_phase_name(TracePhase value);
const char* trace_event_name(TraceEvent value);
const char* adaptive_particle_binding_name(AdaptiveParticleBinding value);
const char* component_screening_mode_name(ComponentScreeningMode value);
const char* particle_engine_name(ParticleEngine value);
const char* streaming_count_storage_name(StreamingCountStorage value);
const char* streaming_particle_storage_name(StreamingParticleStorage value);
HandoffMode parse_handoff(const std::string& value);
ProposalKind parse_proposal(const std::string& value);
StartMethod parse_start_method(const std::string& value);
ComponentScreeningMode parse_component_screening_mode(
    const std::string& value);
ParticleEngine parse_particle_engine(const std::string& value);
StreamingCountStorage parse_streaming_count_storage(
    const std::string& value);
StreamingParticleStorage parse_streaming_particle_storage(
    const std::string& value);

struct Basis {
    RowMajorMatrixXd probabilities; // feature x topic
    std::vector<std::string> features;
    std::vector<std::string> topics;
    uint64_t checksum = 0;
};

struct Dataset {
    std::vector<std::string> identifiers;
    RowMajorMatrixXd centers; // document x topic
    RowMajorMatrixXd coordinates; // document x (topic - 1)
    std::vector<Document> counts; // optional in MAP mode
    Eigen::VectorXd raw_totals;
    Eigen::VectorXd effective_totals;
};

struct Pilot {
    Eigen::VectorXd weights;
    RowMajorMatrixXd means;
    std::vector<Eigen::MatrixXd> covariances;
    Eigen::MatrixXd pooled_covariance;
};

struct AdaptiveParticleDiagnostic {
    double preliminary_maximum_responsibility = 0.0;
    double preliminary_entropy = 0.0;
    int32_t plausible_components = 0;
    double maximum_responsibility_se = 0.0;
    double projected_responsibility_particles = 0.0;
    double projected_moment_particles = 0.0;
    int32_t selected_particles = 0;
    AdaptiveParticleBinding binding = AdaptiveParticleBinding::Minimum;
};

struct AdaptiveParticleOptions {
    int32_t calibration_particles = 32;
    int32_t minimum_particles = 32;
    std::optional<double> responsibility_se_target;
    double plausible_mass = 0.95;
    double plausible_responsibility = 0.05;
    std::optional<double> moment_ess_target;

    bool enabled() const {
        return responsibility_se_target.has_value()
            || moment_ess_target.has_value();
    }
};

struct ComponentScreeningOptions {
    ComponentScreeningMode mode = ComponentScreeningMode::Off;
    double tail_mass = 1e-4;
    double proposal_proxy_tail_mass = 1e-4;
    int32_t minimum_components = 2;
    int32_t maximum_components = 0;
    int32_t audit_documents = 0;
    double minimum_work_reduction = 0.20;
};

struct StreamingOptions {
    std::string cache_directory;
    int32_t block_documents = 64;
    StreamingCountStorage count_storage = StreamingCountStorage::Source;
    StreamingParticleStorage particle_storage =
        StreamingParticleStorage::Positions;
    bool rebuild_cache = false;
};

struct Model {
    CovarianceKind covariance_kind = CovarianceKind::Dense;
    Eigen::VectorXd weights;
    RowMajorMatrixXd means;
    std::vector<Eigen::MatrixXd> covariances;
    Eigen::MatrixXd shrinkage_target;
    std::vector<LowRankDiagonalCovariance> factor_covariances;
    LowRankDiagonalCovariance factor_shrinkage_target;
};

struct IterationDiagnostic {
    TracePhase phase = TracePhase::CorrectedMomScore;
    TraceEvent event = TraceEvent::Evaluation;
    int32_t start = 0;
    int32_t completed_updates = 0;
    double relative_objective_change =
        std::numeric_limits<double>::quiet_NaN();
    double mean_max_responsibility_change =
        std::numeric_limits<double>::quiet_NaN();
    double median_absolute_relative_variance_change =
        std::numeric_limits<double>::quiet_NaN();
    double mean_responsibility_entropy =
        std::numeric_limits<double>::quiet_NaN();
};

struct ModelTraceEntry {
    int32_t completed_updates = 0;
    TraceEvent event = TraceEvent::Evaluation;
    double update_shrinkage_strength =
        std::numeric_limits<double>::quiet_NaN();
    Model model;
};

struct FitOptions {
    HandoffMode handoff = HandoffMode::Particle;
    ProposalKind proposal = ProposalKind::ExactFisher;
    ParticleEngine particle_engine = ParticleEngine::Batch;
    StreamingOptions streaming;
    int32_t n_components = 3;
    int32_t n_particles = 256;
    int32_t particle_em_fixed_iterations = 0;
    int32_t cluster_covariance_rank = -1;
    int32_t kmeans_starts = 5;
    int32_t leiden_starts = 0;
    int32_t max_iterations = 300;
    int32_t kmeans_max_iterations = 100;
    int32_t leiden_neighbors = 15;
    CosineKnnBackend leiden_knn_backend = CosineKnnBackend::Auto;
    int32_t leiden_max_iterations = -1;
    int32_t n_threads = 1;
    int32_t seed = 1;
    double objective_change_tolerance = 1e-5;
    double responsibility_change_tolerance = 1e-3;
    double particle_variance_change_tolerance = 0.0;
    double initialization_ridge_precision = 0.0;
    double target_relative_floor = 1e-4;
    double leiden_knn_epsilon = 0.0;
    double leiden_resolution = 1.0;
    double covariance_floor = 1e-5;
    bool adaptive_covariance_shrinkage = true;
    double covariance_shrinkage_strength = 20.0;
    double fisher_broadening = 1.5;
    AdaptiveParticleOptions adaptive_particles;
    ComponentScreeningOptions component_screening;
    std::optional<Model> particle_initial_model;
    bool exact_final_score = false;
    bool capture_model_trace = false;
    std::function<void(const IterationDiagnostic&)> iteration_callback;
};

struct EstepWorkDiagnostics {
    double gaussian_seconds = 0.0;
    double component_bound_seconds = 0.0;
    double moment_seconds = 0.0;
    int64_t document_evaluations = 0;
    int64_t evaluated_component_documents = 0;
    int64_t possible_component_documents = 0;
    int64_t full_component_documents = 0;
    int64_t component_bound_violations = 0;
};

struct RestartTrace {
    HandoffMode handoff = HandoffMode::Particle;
    TracePhase phase = TracePhase::CorrectedMomScore;
    int32_t start = 0;
    StartMethod start_method = StartMethod::KMeans;
    int32_t seed = 0;
    int32_t raw_communities = 0;
    int32_t reconciliation_count = 0;
    double leiden_resolution = 0.0;
    double selection_objective = -std::numeric_limits<double>::infinity();
    bool selected = false;
    bool succeeded = false;
    bool converged = false;
    bool collapsed = false;
    bool fixed_em_iteration_schedule = false;
    int32_t completed_updates = 0;
    struct Point {
        TraceEvent event = TraceEvent::Evaluation;
        int32_t completed_updates = 0;
        double objective = -std::numeric_limits<double>::infinity();
        double relative_objective_change =
            std::numeric_limits<double>::quiet_NaN();
        double mean_max_responsibility_change =
            std::numeric_limits<double>::quiet_NaN();
        double median_absolute_relative_variance_change =
            std::numeric_limits<double>::quiet_NaN();
        double mean_responsibility_entropy =
            std::numeric_limits<double>::quiet_NaN();
        int32_t active_components = 0;
    };
    std::vector<Point> points;
    std::vector<ModelTraceEntry> model_trace;
    EstepWorkDiagnostics estep_work;
};

struct ParticleDiagnostic {
    double relative_ess = 1.0;
    double maximum_weight = 1.0;
    double log_likelihood_range = 0.0;
    double log_proposal_range = 0.0;
    double hpd80_log_density_threshold =
        -std::numeric_limits<double>::infinity();
    double hpd95_log_density_threshold =
        -std::numeric_limits<double>::infinity();
};

struct ScoreResult {
    RowMajorMatrixXd responsibilities;
    std::string responsibility_sidecar;
    Eigen::VectorXd effective_membership;
    int64_t scored_documents = 0;
    int32_t scored_components = 0;
    std::vector<ParticleDiagnostic> particle_diagnostics;
    double particle_generation_seconds = 0.0;
    double initialization_seconds = 0.0;
    double scoring_seconds = 0.0;
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    double fisher_work_seconds = 0.0;
    double proposal_component_work_seconds = 0.0;
    double proposal_draw_density_work_seconds = 0.0;
    double proposal_precision_fallback_seconds = 0.0;
    int64_t proposal_precision_fallbacks = 0;
    double proposal_screening_seconds = 0.0;
    int64_t proposal_components_constructed = 0;
    int64_t proposal_components_possible = 0;
    int32_t proposal_audit_documents = 0;
    int32_t proposal_audit_represented_components = 0;
    int32_t proposal_audit_covered_components = 0;
    int32_t proposal_audit_violations = 0;
    double proposal_audit_maximum_omitted_mass = 0.0;
    double gaussian_seconds = 0.0;
    double moment_seconds = 0.0;
    double calibration_seconds = 0.0;
    uint64_t resident_particle_bytes = 0;
    uint64_t estimated_peak_proposal_workspace_bytes = 0;
    uint64_t estimated_peak_expectation_workspace_bytes = 0;
    int32_t particle_generation_passes = 0;
    int64_t particle_samples = 0;
    int64_t calibration_samples = 0;
    int64_t reused_calibration_samples = 0;
    AdaptiveParticleOptions adaptive_particle_options;
    ComponentScreeningOptions component_screening_options;
    bool map_component_screening = false;
    bool proposal_component_screening = false;
    bool particle_component_screening = false;
    bool terminal_component_screening = false;
    bool exact_final_score = false;
    double component_bound_seconds = 0.0;
    int64_t evaluated_component_documents = 0;
    int64_t possible_component_documents = 0;
    int32_t full_component_documents = 0;
    int32_t component_bound_violations = 0;
    double maximum_omitted_component_mass = 0.0;
    double mean_omitted_component_mass = 0.0;
    std::vector<int32_t> per_document_evaluated_components;
    std::vector<double> per_document_omitted_component_mass;
    std::vector<int32_t> per_document_proposal_components;
    std::vector<int32_t> per_document_particles;
    std::vector<AdaptiveParticleDiagnostic> adaptive_particle_diagnostics;
    bool streaming = false;
    bool streaming_cache_reused = false;
    uint64_t streaming_cache_bytes = 0;
    uint64_t streaming_peak_particle_bytes = 0;
    uint64_t streaming_count_spool_bytes = 0;
    uint64_t streaming_peak_count_block_bytes = 0;
    int32_t streaming_external_count_parses = 0;
    int32_t streaming_parallel_workers = 0;
    int32_t streaming_cache_shards = 0;
    int32_t streaming_cache_rebuilds = 0;
    StreamingCountStorage streaming_count_storage =
        StreamingCountStorage::Memory;
    StreamingParticleStorage streaming_particle_storage =
        StreamingParticleStorage::Positions;
    // Keeps streamed responsibility artifacts alive for as long as any copy
    // of this result may still be consumed by an output writer.
    std::shared_ptr<detail::ScoreTemporaryStorage> temporary_storage;
};

struct ParticleScoreOptions {
    ProposalKind proposal = ProposalKind::ExactFisher;
    int32_t maximum_particles = 256;
    AdaptiveParticleOptions adaptive_particles;
    int32_t n_threads = 1;
    ParticleEngine particle_engine = ParticleEngine::Batch;
    StreamingOptions streaming;
    ComponentScreeningOptions component_screening;
    bool exact_final_score = false;
};

struct FitResult {
    Model model;
    Pilot pilot;
    ScoreResult score;
    std::vector<RestartTrace> traces;
    bool converged = false;
    int32_t selected_start = -1;
    StartMethod selected_start_method = StartMethod::KMeans;
    double selected_leiden_resolution = 0.0;
    int64_t initialization_measurement_covariance_evaluations = 0;
};

struct State {
    HandoffMode handoff = HandoffMode::Particle;
    ProposalKind proposal = ProposalKind::ExactFisher;
    int32_t n_particles = 256;
    int32_t seed = 1;
    int32_t cluster_covariance_rank = -1;
    int32_t kmeans_starts = 5;
    int32_t leiden_starts = 0;
    int32_t kmeans_max_iterations = 100;
    int32_t leiden_neighbors = 15;
    CosineKnnBackend leiden_knn_backend = CosineKnnBackend::Auto;
    int32_t leiden_max_iterations = -1;
    int32_t selected_start = -1;
    StartMethod selected_start_method = StartMethod::KMeans;
    double selected_leiden_resolution = 0.0;
    bool converged = false;
    double center_floor = 1e-12;
    double target_relative_floor = 1e-4;
    double leiden_knn_epsilon = 0.0;
    double leiden_resolution = 1.0;
    double covariance_floor = 1e-5;
    double objective_change_tolerance = 1e-5;
    double responsibility_change_tolerance = 1e-3;
    double particle_variance_change_tolerance = 0.0;
    double initialization_ridge_precision = 0.0;
    bool adaptive_covariance_shrinkage = true;
    double covariance_shrinkage_strength = 20.0;
    double fisher_broadening = 1.5;
    AdaptiveParticleOptions fit_adaptive_particles;
    ComponentScreeningOptions component_screening;
    bool fit_map_component_screening = false;
    bool fit_proposal_component_screening = false;
    bool fit_particle_component_screening = false;
    // Canonical full basis in persisted states; runtime basis in scoring copies.
    uint64_t basis_checksum = 0;
    bool weighted_counts = false;
    std::vector<std::string> topics;
    Eigen::MatrixXd helmert;
    // Full model order in persisted states; runtime panel order in scoring copies.
    Eigen::VectorXd feature_weights;
    Pilot pilot;
    Model model;
};

struct StateMetadata {
    std::vector<std::string> topics;
    Eigen::MatrixXd helmert;
    double center_floor = 1e-12;
    uint64_t basis_checksum = 0; // canonical full basis
    Eigen::VectorXd feature_weights; // canonical full model order
    bool weighted_counts = false;
};

void normalize_basis(Basis& basis);
void normalize_centers(RowMajorMatrixXd& centers, double floor = 1e-12);
uint64_t basis_checksum(const Basis& basis);
double median_absolute_relative_variance_change(const Model& current,
    const Model& previous, double covariance_floor);

FitResult fit(Dataset& data, const Basis* basis,
    const FitOptions& options);
FitResult fit(const Dataset& data, const Basis* basis,
    const FitOptions& options);
FitResult fit(Dataset& data, const Basis* basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const FitOptions& options);
FitResult fit(const Dataset& data, const Basis* basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const FitOptions& options);
ScoreResult score_map(const Dataset& data, const Model& model,
    int32_t n_threads = 1,
    const ComponentScreeningOptions& component_screening = {});
ScoreResult score_particle(Dataset& data, const Basis& basis,
    const State& state, const ParticleScoreOptions& options);
ScoreResult score_particle(const Dataset& data, const Basis& basis,
    const State& state, const ParticleScoreOptions& options);

State make_state(const FitResult& fit, const FitOptions& options,
    const StateMetadata& metadata);
void write_state(const std::string& path, const State& state);
State read_state(const std::string& path);

void write_model(const std::string& path, const State& state,
    const Eigen::VectorXd* effective_membership = nullptr);
void write_results(const std::string& path, const Dataset& data,
    const ScoreResult& score, int32_t top_c = -1);
void write_diagnostics(const std::string& path, const Dataset& data,
    const ScoreResult& score);
void write_trace(const std::string& path,
    const std::vector<RestartTrace>& traces);
void write_model_trace(const std::string& path,
    const std::vector<RestartTrace>& traces);
void write_separation(const std::string& path, const Model& model);
void write_representatives(const std::string& path, const Dataset& data,
    const ScoreResult& score, int32_t n_representatives = 10);

} // namespace uac
