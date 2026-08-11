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
    SubsampleEm,
    OnlineEm,
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

enum class FactorDiagonalMode {
    Component,
    Shared,
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

enum class ParticleFitSchedule {
    Exact,
    Subsample,
    Online,
};

enum class SubsampleStorage {
    Auto,
    Resident,
    Disk,
};

enum class InitializationMeasurementMode {
    Legacy,
    Full,
    HorvitzThompson,
};

enum class FitTailMode {
    Adaptive,
    Fixed,
    Off,
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

enum class VisualizationWhitening {
    Sample,
    Mixture,
};

enum class VisualizationView {
    Mean,
    Full,
};

const char* handoff_name(HandoffMode value);
const char* proposal_name(ProposalKind value);
const char* start_method_name(StartMethod value);
const char* trace_phase_name(TracePhase value);
const char* trace_event_name(TraceEvent value);
const char* adaptive_particle_binding_name(AdaptiveParticleBinding value);
const char* component_screening_mode_name(ComponentScreeningMode value);
const char* particle_engine_name(ParticleEngine value);
const char* particle_fit_schedule_name(ParticleFitSchedule value);
const char* subsample_storage_name(SubsampleStorage value);
const char* initialization_measurement_mode_name(
    InitializationMeasurementMode value);
const char* fit_tail_mode_name(FitTailMode value);
const char* streaming_count_storage_name(StreamingCountStorage value);
const char* streaming_particle_storage_name(StreamingParticleStorage value);
const char* visualization_whitening_name(VisualizationWhitening value);
const char* visualization_view_name(VisualizationView value);
const char* factor_diagonal_mode_name(FactorDiagonalMode value);
HandoffMode parse_handoff(const std::string& value);
ProposalKind parse_proposal(const std::string& value);
StartMethod parse_start_method(const std::string& value);
ComponentScreeningMode parse_component_screening_mode(
    const std::string& value);
ParticleEngine parse_particle_engine(const std::string& value);
ParticleFitSchedule parse_particle_fit_schedule(const std::string& value);
SubsampleStorage parse_subsample_storage(const std::string& value);
InitializationMeasurementMode parse_initialization_measurement_mode(
    const std::string& value);
FitTailMode parse_fit_tail_mode(const std::string& value);
StreamingCountStorage parse_streaming_count_storage(
    const std::string& value);
StreamingParticleStorage parse_streaming_particle_storage(
    const std::string& value);
VisualizationWhitening parse_visualization_whitening(
    const std::string& value);
FactorDiagonalMode parse_factor_diagonal_mode(const std::string& value);

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
    FactorDiagonalMode factor_diagonal_mode =
        FactorDiagonalMode::Component;
    Eigen::VectorXd weights;
    RowMajorMatrixXd means;
    std::vector<Eigen::MatrixXd> covariances;
    Eigen::MatrixXd shrinkage_target;
    std::vector<LowRankDiagonalCovariance> factor_covariances;
    Eigen::VectorXd shared_factor_diagonal;
    LowRankDiagonalCovariance factor_shrinkage_target;
};

struct VisualizationOptions {
    VisualizationWhitening whitening = VisualizationWhitening::Mixture;
    int32_t dimensions = 2;
    int32_t n_threads = 1;
    double covariance_floor = 1e-5;
};

struct VisualizationProjection {
    VisualizationView view = VisualizationView::Mean;
    Eigen::VectorXd eigenvalues;
    Eigen::MatrixXd projection; // ILR coordinate x visualization axis
    Eigen::MatrixXd topic_contrasts; // topic x visualization axis
    RowMajorMatrixXd component_means; // component x visualization axis
    std::vector<Eigen::MatrixXd> component_covariances;
};

struct VisualizationResult {
    VisualizationWhitening whitening = VisualizationWhitening::Mixture;
    Eigen::MatrixXd whitening_covariance;
    VisualizationProjection mean;
    VisualizationProjection full;
};

struct VisualizationMoments {
    Eigen::VectorXd weights;
    RowMajorMatrixXd means;
    std::vector<Eigen::MatrixXd> covariances;
};

struct VisualizationSampleMoments {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
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
    ParticleFitSchedule particle_fit_schedule = ParticleFitSchedule::Exact;
    StreamingOptions streaming;
    int32_t n_components = 3;
    int32_t n_particles = 256;
    int32_t particle_em_fixed_iterations = 0;
    int32_t cluster_covariance_rank = -1;
    FactorDiagonalMode factor_diagonal_mode =
        FactorDiagonalMode::Component;
    int32_t kmeans_starts = 5;
    int32_t leiden_starts = 0;
    int32_t max_iterations = 300;
    int32_t kmeans_max_iterations = 100;
    int32_t leiden_neighbors = 15;
    SimplexMetric initialization_metric = SimplexMetric::Cosine;
    CosineKnnBackend leiden_knn_backend = CosineKnnBackend::Auto;
    int32_t leiden_hnsw_m = 16;
    int32_t leiden_hnsw_ef_construction = 100;
    int32_t leiden_hnsw_ef_search = 0;
    int32_t leiden_hnsw_max_ef_search = 512;
    int32_t leiden_hnsw_candidates = 0;
    int32_t leiden_hnsw_audit_queries = 256;
    double leiden_hnsw_recall = 0.98;
    bool leiden_hnsw_force = false;
    int32_t leiden_nndescent_iterations = 0;
    int32_t leiden_nndescent_graph_size = 0;
    int32_t leiden_nndescent_sample_candidates = 10;
    int32_t leiden_nndescent_audit_queries = 256;
    double leiden_nndescent_recall = 0.98;
    int32_t leiden_max_iterations = -1;
    int32_t n_threads = 1;
    int32_t seed = 1;
    double objective_change_tolerance = 1e-5;
    double responsibility_change_tolerance = 1e-3;
    double particle_variance_change_tolerance = 0.0;
    double initialization_ridge_precision = 0.0;
    InitializationMeasurementMode initialization_measurement_mode =
        InitializationMeasurementMode::HorvitzThompson;
    int32_t initialization_measurement_target = 1024;
    int32_t initialization_candidate_score_target = -1;
    int32_t initialization_sampling_seed = -1;
    bool initialization_only = false;
    double target_relative_floor = 1e-4;
    double leiden_knn_epsilon = 0.0;
    double leiden_resolution = 1.0;
    double covariance_floor = 1e-5;
    bool adaptive_covariance_shrinkage = true;
    double covariance_shrinkage_strength = 20.0;
    double fisher_broadening = 1.5;
    int32_t fisher_refinement_iterations = 1;
    double fit_document_budget = 0.0;
    int32_t fit_full_tail_updates = 1;
    int32_t fit_subsample_target = 1024;
    double fit_subsample_base_fraction = 0.01;
    SubsampleStorage fit_subsample_storage = SubsampleStorage::Auto;
    uint64_t fit_subsample_memory_budget = 1ull << 30;
    double fit_subsample_safety_factor = 1.1;
    int32_t fit_subsample_min_updates = 2;
    int32_t fit_subsample_max_updates = 20;
    double fit_subsample_change_tolerance = 1e-3;
    int32_t fit_subsample_topup_rounds = 2;
    FitTailMode fit_tail = FitTailMode::Adaptive;
    int32_t fit_batch_documents = 2048;
    double fit_step_kappa = 0.7;
    double fit_step_initial = 0.1;
    AdaptiveParticleOptions adaptive_particles;
    ComponentScreeningOptions component_screening;
    std::optional<Model> particle_initial_model;
    bool exact_final_score = false;
    bool capture_model_trace = false;
    std::function<void(const IterationDiagnostic&)> iteration_callback;
};

struct InitializationDiagnostics {
    InitializationMeasurementMode measurement_mode =
        InitializationMeasurementMode::Legacy;
    int64_t total_documents = 0;
    int64_t measurement_documents = 0;
    int64_t candidate_score_documents = 0;
    int64_t measurement_covariance_evaluations = 0;
    int32_t sampling_seed = 0;
    int32_t measurement_target = 0;
    int32_t candidate_score_target = 0;
    uint64_t cached_measurement_bytes = 0;
    double maximum_measurement_weight = 1.0;
    double maximum_candidate_score_weight = 1.0;
    double minimum_measurement_effective_size = 0.0;
    double minimum_candidate_score_effective_size = 0.0;
    int32_t covariance_floor_activations = 0;
    double partition_seconds = 0.0;
    double measurement_seconds = 0.0;
    double candidate_score_seconds = 0.0;
    double total_seconds = 0.0;
};

struct FitScheduleDiagnostics {
    ParticleFitSchedule schedule = ParticleFitSchedule::Exact;
    int32_t full_warmup_updates = 0;
    int32_t approximate_updates = 0;
    int32_t full_tail_updates = 0;
    int32_t full_data_evaluations = 0;
    int32_t subsample_evaluations = 0;
    int64_t approximate_documents = 0;
    int32_t subsample_documents = 0;
    int32_t subsample_topup_rounds = 0;
    double subsample_minimum_effective_size = 0.0;
    double subsample_minimum_target_ratio = 0.0;
    double subsample_weighted_documents = 0.0;
    double subsample_maximum_weight = 0.0;
    SubsampleStorage subsample_storage = SubsampleStorage::Auto;
    uint64_t subsample_memory_budget = 0;
    uint64_t subsample_predicted_bytes = 0;
    uint64_t subsample_peak_bytes = 0;
    uint64_t subsample_selected_particle_bytes = 0;
    uint64_t subsample_disk_bytes = 0;
    int32_t subsample_storage_promotions = 0;
    std::string subsample_peak_phase;
    int32_t subsample_full_cache_scans = 0;
    uint64_t subsample_read_bytes = 0;
    uint64_t subsample_write_bytes = 0;
    double subsample_io_seconds = 0.0;
    bool subsample_allocator_converged = false;
    int32_t subsample_allocator_iterations = 0;
    double subsample_parameter_change =
        std::numeric_limits<double>::quiet_NaN();
    std::string subsample_convergence_reason;
    FitTailMode tail_mode = FitTailMode::Off;
    bool audit_converged = false;
    double audit_parameter_change =
        std::numeric_limits<double>::quiet_NaN();
    bool audit_active_set_unchanged = false;
    std::vector<int32_t> subsample_stratum_documents;
    std::vector<double> subsample_stratum_purity;
    std::vector<double> subsample_stratum_probability;
    std::vector<uint64_t> subsample_stratum_bytes;
    std::vector<double> subsample_component_target;
    std::vector<double> subsample_component_predicted_effective_size;
    std::vector<double> subsample_component_realized_effective_size;
    std::vector<int32_t> subsample_component_topups;
    double document_pass_equivalents = 0.0;
    double fitting_seconds = 0.0;
    double approximate_seconds = 0.0;
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

struct InitializationPartition {
    int32_t start = 0;
    StartMethod start_method = StartMethod::KMeans;
    Eigen::VectorXi assignments;
    Eigen::VectorXi raw_assignments;
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
    FitScheduleDiagnostics fit_schedule;
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
    std::vector<InitializationPartition> initialization_partitions;
    bool converged = false;
    int32_t selected_start = -1;
    StartMethod selected_start_method = StartMethod::KMeans;
    double selected_leiden_resolution = 0.0;
    int64_t initialization_measurement_covariance_evaluations = 0;
    InitializationDiagnostics initialization;
    FitScheduleDiagnostics fit_schedule;
    bool has_leiden_knn_diagnostics = false;
    CosineKnnDiagnostics leiden_knn_diagnostics;
};

struct State {
    HandoffMode handoff = HandoffMode::Particle;
    ProposalKind proposal = ProposalKind::ExactFisher;
    int32_t n_particles = 256;
    int32_t seed = 1;
    int32_t cluster_covariance_rank = -1;
    FactorDiagonalMode factor_diagonal_mode =
        FactorDiagonalMode::Component;
    int32_t kmeans_starts = 5;
    int32_t leiden_starts = 0;
    int32_t kmeans_max_iterations = 100;
    int32_t leiden_neighbors = 15;
    SimplexMetric initialization_metric = SimplexMetric::Cosine;
    CosineKnnBackend leiden_knn_backend = CosineKnnBackend::Auto;
    int32_t leiden_hnsw_m = 16;
    int32_t leiden_hnsw_ef_construction = 100;
    int32_t leiden_hnsw_ef_search = 0;
    int32_t leiden_hnsw_max_ef_search = 512;
    int32_t leiden_hnsw_candidates = 0;
    int32_t leiden_hnsw_audit_queries = 256;
    double leiden_hnsw_recall = 0.98;
    bool leiden_hnsw_force = false;
    int32_t leiden_nndescent_iterations = 0;
    int32_t leiden_nndescent_graph_size = 0;
    int32_t leiden_nndescent_sample_candidates = 10;
    int32_t leiden_nndescent_audit_queries = 256;
    double leiden_nndescent_recall = 0.98;
    int32_t leiden_resolved_ann_parameter = 0;
    int32_t leiden_resolved_ann_candidates = 0;
    double leiden_ann_audit_mean_recall = 0.0;
    double leiden_ann_audit_recall_lcb = 0.0;
    bool leiden_ann_audit_passed = false;
    bool leiden_ann_forced = false;
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
    int32_t fisher_refinement_iterations = 1;
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

VisualizationResult make_visualization(const Dataset& data,
    const Model& model, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options = {});
VisualizationResult make_visualization(const Dataset& data,
    const VisualizationMoments& moments,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options = {});
VisualizationResult make_visualization(const Dataset& data,
    const VisualizationMoments& moments,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options,
    const VisualizationSampleMoments& sample_moments);
VisualizationMoments summarize_hard_partition(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components, int32_t n_threads = 1);
VisualizationSampleMoments summarize_visualization_sample(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    int32_t n_threads = 1);

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
void write_initialization_diagnostics(const std::string& path,
    const InitializationDiagnostics& diagnostics);
void write_initialization_results(const std::string& path,
    const Dataset& data,
    const std::vector<InitializationPartition>& partitions);
void write_subsample_diagnostics(const std::string& path,
    const FitScheduleDiagnostics& diagnostics);
void write_trace(const std::string& path,
    const std::vector<RestartTrace>& traces);
void write_model_trace(const std::string& path,
    const std::vector<RestartTrace>& traces);
void write_separation(const std::string& path, const Model& model);
void write_representatives(const std::string& path, const Dataset& data,
    const ScoreResult& score, int32_t n_representatives = 10);
void write_visualization_axes(const std::string& path,
    const State& state, const VisualizationResult& visualization);
void write_visualization_axes(const std::string& path,
    const std::vector<std::string>& topics,
    const VisualizationResult& visualization);
void write_visualization_model(const std::string& path,
    const State& state, const VisualizationResult& visualization);
void write_visualization_results(const std::string& path,
    const Dataset& data, const VisualizationResult& visualization);

} // namespace uac
