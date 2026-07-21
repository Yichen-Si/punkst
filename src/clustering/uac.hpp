#pragma once

#include "dataunits.hpp"
#include "clustering_core/cosine_clustering.hpp"
#include "clustering/low_rank_covariance.hpp"
#include "numerical_utils.hpp"

#include <cstdint>
#include <functional>
#include <limits>
#include <string>
#include <vector>

namespace uac {

enum class HandoffMode {
    Map,
    Particle,
};

enum class ProposalKind {
    ExactFisher,
    SparseEmpiricalFisher,
};

enum class MapInitializer {
    KMeans,
    Leiden,
};

enum class CovarianceKind {
    Dense,
    FactorAnalytic,
};

enum class AdaptiveParticleRule {
    Legacy,
    ResponsibilityOnly,
    MomentOnly,
    ResponsibilityMoment,
};

enum class AdaptiveParticleBinding {
    Minimum,
    LegacyEss,
    LegacyMaximumWeight,
    LegacyContrast,
    Responsibility,
    MomentEss,
};

const char* handoff_name(HandoffMode value);
const char* proposal_name(ProposalKind value);
const char* initializer_name(MapInitializer value);
const char* adaptive_particle_rule_name(AdaptiveParticleRule value);
const char* adaptive_particle_binding_name(AdaptiveParticleBinding value);
HandoffMode parse_handoff(const std::string& value);
ProposalKind parse_proposal(const std::string& value);
MapInitializer parse_initializer(const std::string& value);
AdaptiveParticleRule parse_adaptive_particle_rule(const std::string& value);

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
    std::vector<Eigen::MatrixXd> raw_covariances;
    std::vector<Eigen::MatrixXd> covariances;
    Eigen::MatrixXd pooled_covariance;
};

struct ParticleSet {
    int32_t first_document = 0;
    int32_t documents = 0;
    int32_t samples = 0;
    int32_t dimension = 0;
    RowMajorMatrixXd values; // (document * sample) x dimension
    RowMajorMatrixXd log_likelihood; // document x sample
    RowMajorMatrixXd log_proposal; // document x sample
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    double fisher_work_seconds = 0.0;
    double proposal_component_work_seconds = 0.0;
    double proposal_draw_density_work_seconds = 0.0;
    uint64_t proposal_workspace_bytes = 0;

    Eigen::Map<const Eigen::VectorXd> value(int32_t document,
        int32_t sample) const;
    double log_q(int32_t document, int32_t sample) const;
    int32_t samples_for_document(int32_t document) const;
    Eigen::Map<const RowMajorMatrixXd> values_for_document(
        int32_t document) const;
    Eigen::Map<const Eigen::VectorXd> log_likelihood_for_document(
        int32_t document) const;
    Eigen::Map<const Eigen::VectorXd> log_proposal_for_document(
        int32_t document) const;
};

struct AdaptiveParticleDiagnostic {
    double preliminary_maximum_responsibility = 0.0;
    double preliminary_entropy = 0.0;
    int32_t plausible_components = 0;
    double maximum_responsibility_se = 0.0;
    double projected_responsibility_particles = 0.0;
    double projected_moment_particles = 0.0;
    double plausible_maximum_weight = 0.0;
    double half_sample_maximum_responsibility_difference = 0.0;
    bool half_sample_top_disagreement = false;
    int32_t selected_particles = 0;
    AdaptiveParticleBinding binding = AdaptiveParticleBinding::Minimum;
};

struct RaggedParticleSet {
    int32_t first_document = 0;
    int32_t documents = 0;
    int32_t dimension = 0;
    int32_t maximum_samples = 0;
    std::vector<int64_t> offsets;
    std::vector<double> values;
    std::vector<double> log_likelihood;
    std::vector<double> log_proposal;
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    double calibration_seconds = 0.0;
    double fisher_work_seconds = 0.0;
    double proposal_component_work_seconds = 0.0;
    double proposal_draw_density_work_seconds = 0.0;
    uint64_t proposal_workspace_bytes = 0;
    int64_t calibration_samples = 0;
    int64_t reused_calibration_samples = 0;
    std::vector<AdaptiveParticleDiagnostic> adaptive_diagnostics;

    int32_t samples_for_document(int32_t document) const;
    Eigen::Map<const RowMajorMatrixXd> values_for_document(
        int32_t document) const;
    Eigen::Map<const Eigen::VectorXd> log_likelihood_for_document(
        int32_t document) const;
    Eigen::Map<const Eigen::VectorXd> log_proposal_for_document(
        int32_t document) const;
};

struct AdaptiveParticleOptions {
    bool enabled = false;
    AdaptiveParticleRule rule = AdaptiveParticleRule::Legacy;
    int32_t calibration_particles = 32;
    int32_t minimum_particles = 32;
    int32_t maximum_particles = 256;
    double material_mass = 0.99;
    double material_responsibility = 0.01;
    double component_ess_target = 32.0;
    double contrast_se_target = 0.2;
    double maximum_weight_target = 0.1;
    double responsibility_se_target = 0.05;
    double plausible_mass = 0.95;
    double plausible_responsibility = 0.05;
    double moment_ess_target = 16.0;
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
    bool particle = false;
    int32_t start = 0;
    int32_t iteration = 0;
    double relative_objective_change =
        std::numeric_limits<double>::quiet_NaN();
    double mean_max_responsibility_change =
        std::numeric_limits<double>::quiet_NaN();
};

struct FitOptions {
    HandoffMode handoff = HandoffMode::Particle;
    ProposalKind proposal = ProposalKind::ExactFisher;
    int32_t n_components = 3;
    int32_t n_particles = 256;
    int32_t particle_block_size = 0;
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
    double center_floor = 1e-12;
    double target_relative_floor = 1e-4;
    double leiden_knn_epsilon = 0.0;
    double leiden_resolution = 1.0;
    double covariance_floor = 1e-5;
    bool adaptive_covariance_shrinkage = true;
    double fisher_broadening = 1.5;
    AdaptiveParticleOptions adaptive_particles;
    std::function<void(const IterationDiagnostic&)> iteration_callback;
};

struct RestartTrace {
    int32_t start = 0;
    MapInitializer initializer = MapInitializer::KMeans;
    int32_t seed = 0;
    int32_t raw_communities = 0;
    int32_t reconciliation_count = 0;
    double leiden_resolution = 0.0;
    double selection_objective = -std::numeric_limits<double>::infinity();
    bool particle = false;
    bool converged = false;
    bool collapsed = false;
    std::vector<double> objective;
    std::vector<double> relative_objective_change;
    std::vector<double> mean_max_responsibility_change;
    std::vector<int32_t> active_components;
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
    std::vector<ParticleDiagnostic> particle_diagnostics;
    double particle_generation_seconds = 0.0;
    double scoring_seconds = 0.0;
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    double fisher_work_seconds = 0.0;
    double proposal_component_work_seconds = 0.0;
    double proposal_draw_density_work_seconds = 0.0;
    double gaussian_seconds = 0.0;
    double moment_seconds = 0.0;
    double calibration_seconds = 0.0;
    uint64_t particle_bytes = 0;
    uint64_t proposal_workspace_bytes = 0;
    uint64_t expectation_accumulator_bytes = 0;
    int32_t particle_block_size = 0;
    int32_t particle_generation_passes = 0;
    int64_t particle_samples = 0;
    int64_t calibration_samples = 0;
    int64_t reused_calibration_samples = 0;
    AdaptiveParticleOptions adaptive_particle_options;
    std::vector<int32_t> per_document_particles;
    std::vector<AdaptiveParticleDiagnostic> adaptive_particle_diagnostics;
    bool particle_replay = false;
};

struct FitResult {
    Model model;
    Pilot pilot;
    ScoreResult score;
    std::vector<RestartTrace> traces;
    bool converged = false;
    int32_t selected_start = -1;
    MapInitializer selected_initializer = MapInitializer::KMeans;
    double selected_leiden_resolution = 0.0;
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
    MapInitializer selected_initializer = MapInitializer::KMeans;
    double selected_leiden_resolution = 0.0;
    bool converged = false;
    double center_floor = 1e-12;
    double target_relative_floor = 1e-4;
    double leiden_knn_epsilon = 0.0;
    double leiden_resolution = 1.0;
    double covariance_floor = 1e-5;
    double objective_change_tolerance = 1e-5;
    double responsibility_change_tolerance = 1e-3;
    bool adaptive_covariance_shrinkage = true;
    double fisher_broadening = 1.5;
    AdaptiveParticleOptions fit_adaptive_particles;
    uint64_t basis_checksum = 0;
    bool weighted_counts = false;
    std::vector<std::string> topics;
    Eigen::MatrixXd helmert;
    Eigen::VectorXd feature_weights;
    Pilot pilot;
    Model model;
};

Eigen::MatrixXd normalized_helmert(int32_t topics);
RowMajorMatrixXd ilr_transform(const Eigen::Ref<const RowMajorMatrixXd>& values,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, double floor = 1e-12);
RowMajorMatrixXd ilr_inverse(const Eigen::Ref<const RowMajorMatrixXd>& values,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert);
void normalize_basis(Basis& basis);
void normalize_centers(RowMajorMatrixXd& centers, double floor = 1e-12);
uint64_t basis_checksum(const Basis& basis);

struct FisherApproximation {
    Eigen::VectorXd gradient;
    Eigen::MatrixXd information;
};

FisherApproximation fisher_approximation(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal);

ParticleSet make_particles(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    ProposalKind proposal, int32_t samples, uint64_t seed,
    double fisher_broadening = 1.5, int32_t n_threads = 1);

FitResult fit(const Dataset& data, const Basis* basis,
    const FitOptions& options);
ScoreResult score_map(const Dataset& data, const Model& model,
    int32_t n_threads = 1);
ScoreResult score_particle(const Dataset& data, const Basis& basis,
    const State& state, ProposalKind proposal, int32_t particles,
    const AdaptiveParticleOptions& adaptive_particles = {},
    int32_t n_threads = 1, int32_t particle_block_size = 0);
inline ScoreResult score_particle(const Dataset& data, const Basis& basis,
    const State& state, ProposalKind proposal, int32_t particles,
    int32_t n_threads, int32_t particle_block_size = 0) {
    return score_particle(data, basis, state, proposal, particles,
        AdaptiveParticleOptions{}, n_threads, particle_block_size);
}

State make_state(const FitResult& fit, const Basis* basis,
    const FitOptions& options, const Eigen::VectorXd& feature_weights,
    bool weighted_counts);
void write_state(const std::string& path, const State& state);
State read_state(const std::string& path);

void write_model(const std::string& path, const State& state,
    const Eigen::VectorXd* effective_membership = nullptr);
void write_results(const std::string& path, const Dataset& data,
    const ScoreResult& score);
void write_diagnostics(const std::string& path, const Dataset& data,
    const ScoreResult& score);
void write_trace(const std::string& path,
    const std::vector<RestartTrace>& traces);
void write_separation(const std::string& path, const Model& model);
void write_representatives(const std::string& path, const Dataset& data,
    const ScoreResult& score, int32_t n_representatives = 10);

namespace detail {

double increased_leiden_resolution(double resolution, int32_t raw_communities,
    int32_t requested_communities);
double midpoint_leiden_resolution(double lower, double upper);

} // namespace detail

} // namespace uac
