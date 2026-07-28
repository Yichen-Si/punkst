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
Model initialize_model_from_partition(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components, double shrinkage, double covariance_floor,
    double relative_floor);
Model initialize_model_from_corrected_moments(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    const HardPartitionMoments& moments,
    const std::vector<Eigen::MatrixXd>& measurement_sum,
    double shrinkage, double covariance_floor);
Pilot pilot_from_map(const Dataset& data, const Model& model,
    const Expectation& expectation, double relative_floor);
Pilot pilot_from_model(const Model& model);
ModelUpdate update_model(Model& model, const Expectation& expectation,
    double shrinkage, double covariance_floor,
    bool adaptive_target = false);

Candidate fit_map_candidate(const Dataset& data, Model initial,
    const FitOptions& options, const RestartTrace& metadata);
void record_trace_point(RestartTrace& trace, const FitOptions& options,
    TraceEvent event, int32_t completed_updates, double objective,
    int32_t active_components, double relative_objective_change,
    double responsibility_change,
    double variance_change =
        std::numeric_limits<double>::quiet_NaN(),
    double mean_responsibility_entropy =
        std::numeric_limits<double>::quiet_NaN());
void score_corrected_moment_candidates(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const FitOptions& options, std::vector<Candidate>& candidates,
    const IndexedDocumentSource* count_source = nullptr);
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
