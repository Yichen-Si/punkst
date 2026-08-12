#pragma once

#include "clustering/uac_expectation_internal.hpp"

#include <functional>
#include <limits>
#include <vector>

namespace uac::detail {

struct HardPartitionMoments {
    Eigen::VectorXi counts;
    RowMajorMatrixXd means;
    std::vector<Eigen::MatrixXd> scatter;
    Eigen::MatrixXd pooled_scatter;
};

struct ModelUpdate {
    bool valid = false;
    int32_t active_components = 0;
};

struct InitializationMeasurements {
    std::vector<std::vector<Eigen::MatrixXd>> sums;
    int64_t measurement_documents = 0;
    std::vector<int32_t> score_documents;
    Eigen::VectorXd score_probabilities;
    RowMajorMatrixXd packed_score_covariances;
    int64_t covariance_evaluations = 0;
    uint64_t cache_bytes = 0;
    double maximum_measurement_weight = 1.0;
    double maximum_score_weight = 1.0;
    double minimum_measurement_effective_size = 0.0;
    double minimum_score_effective_size = 0.0;
    double seconds = 0.0;
};

struct Candidate {
    Model model;
    RestartTrace trace;
    double objective = -std::numeric_limits<double>::infinity();
    bool terminal_pending = false;
};

HardPartitionMoments hard_partition_moments(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components);
Eigen::MatrixXd shared_measurement_precision(
    const std::vector<HardPartitionMoments>& moments,
    double scalar_precision, double relative_floor);
std::vector<std::vector<Eigen::MatrixXd>>
measurement_sums_by_partition(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const std::vector<Eigen::VectorXi>& assignments,
    int32_t components, ProposalKind proposal,
    const IndexedDocumentSource* count_source = nullptr);
InitializationMeasurements collect_initialization_measurements(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const std::vector<Eigen::VectorXi>& assignments,
    const std::vector<HardPartitionMoments>& moments,
    int32_t components, ProposalKind proposal,
    InitializationMeasurementMode mode, int32_t measurement_target,
    int32_t score_target, int32_t seed,
    const IndexedDocumentSource* count_source = nullptr);
Model initialize_model_from_partition(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components, double shrinkage, double covariance_floor,
    double relative_floor);
Model initialize_model_from_corrected_moments(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    const HardPartitionMoments& moments,
    const std::vector<Eigen::MatrixXd>& measurement_sum,
    double shrinkage, double covariance_floor,
    int32_t* floor_activations = nullptr);
Pilot pilot_from_map(const Dataset& data, const Model& model,
    const Expectation& expectation, double relative_floor);
Pilot pilot_from_model(const Model& model);
ModelUpdate update_model(Model& model, const Expectation& expectation,
    double shrinkage, double covariance_floor,
    bool adaptive_target = false, bool update_weights = true,
    bool allow_extinction = true,
    const Eigen::VectorXd* shrinkage_membership = nullptr);

Candidate fit_map_candidate(const Dataset& data, Model initial,
    const FitOptions& options, const RestartTrace& metadata);
double mean_top_probability(const Expectation& expectation);
void record_trace_point(RestartTrace& trace, const FitOptions& options,
    TraceEvent event, int32_t completed_updates, double objective,
    int32_t active_components, double relative_objective_change,
    double responsibility_change,
    double variance_change =
        std::numeric_limits<double>::quiet_NaN(),
    double mean_top_probability =
        std::numeric_limits<double>::quiet_NaN());
void accumulate_estep_work(
    RestartTrace& trace, const Expectation& expectation);
void score_corrected_moment_candidates(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const FitOptions& options, std::vector<Candidate>& candidates,
    const IndexedDocumentSource* count_source = nullptr);
void score_corrected_moment_candidates(
    const Dataset& data, const FitOptions& options,
    const InitializationMeasurements& measurements,
    std::vector<Candidate>& candidates);
Candidate fit_particle_candidate(
    const std::function<Expectation(const Model&)>& expectation_function,
    Model initial, const FitOptions& options,
    const RestartTrace& initialization_trace);
void finalize_particle_candidate(Candidate& candidate,
    const Expectation& terminal, const FitOptions& options);
ScoreResult score_particles(const ParticleSet& particles,
    const Model& model,
    const ComponentScreeningOptions& screening = {},
    Expectation* terminal_expectation = nullptr);
ScoreResult score_particles(const RaggedParticleSet& particles,
    const Model& model,
    const ComponentScreeningOptions& screening = {},
    Expectation* terminal_expectation = nullptr);

} // namespace uac::detail
