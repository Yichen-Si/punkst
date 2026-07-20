#pragma once

#include "dataunits.hpp"
#include "numerical_utils.hpp"

#include <cstdint>
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

const char* handoff_name(HandoffMode value);
const char* proposal_name(ProposalKind value);
HandoffMode parse_handoff(const std::string& value);
ProposalKind parse_proposal(const std::string& value);

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
    int32_t documents = 0;
    int32_t samples = 0;
    int32_t dimension = 0;
    RowMajorMatrixXd values; // (document * sample) x dimension
    RowMajorMatrixXd log_likelihood; // document x sample
    RowMajorMatrixXd log_proposal; // document x sample
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;

    Eigen::Map<const Eigen::VectorXd> value(int32_t document,
        int32_t sample) const;
    double log_q(int32_t document, int32_t sample) const;
};

struct Model {
    Eigen::VectorXd weights;
    RowMajorMatrixXd means;
    std::vector<Eigen::MatrixXd> covariances;
    Eigen::MatrixXd shrinkage_target;
};

struct FitOptions {
    HandoffMode handoff = HandoffMode::Particle;
    ProposalKind proposal = ProposalKind::ExactFisher;
    int32_t n_components = 3;
    int32_t n_particles = 256;
    int32_t restarts = 5;
    int32_t max_iterations = 300;
    int32_t kmeans_max_iterations = 100;
    int32_t n_threads = 1;
    int32_t seed = 1;
    double tolerance = 1e-6;
    double center_floor = 1e-12;
    double pilot_rho = 0.1;
    double pilot_relative_floor = 1e-4;
    double covariance_floor = 1e-5;
    double covariance_shrinkage = 20.0;
    double fisher_broadening = 1.5;
};

struct RestartTrace {
    int32_t restart = 0;
    bool particle = false;
    bool converged = false;
    bool collapsed = false;
    std::vector<double> objective;
};

struct ParticleDiagnostic {
    double relative_ess = 1.0;
    double maximum_weight = 1.0;
    double log_likelihood_range = 0.0;
    double log_proposal_range = 0.0;
};

struct ScoreResult {
    RowMajorMatrixXd responsibilities;
    std::vector<ParticleDiagnostic> particle_diagnostics;
    double particle_generation_seconds = 0.0;
    double scoring_seconds = 0.0;
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    uint64_t particle_bytes = 0;
};

struct FitResult {
    Model model;
    Pilot pilot;
    ScoreResult score;
    std::vector<RestartTrace> traces;
    bool converged = false;
    int32_t selected_restart = -1;
};

struct State {
    HandoffMode handoff = HandoffMode::Particle;
    ProposalKind proposal = ProposalKind::ExactFisher;
    int32_t n_particles = 256;
    int32_t seed = 1;
    int32_t selected_restart = -1;
    bool converged = false;
    double center_floor = 1e-12;
    double pilot_rho = 0.1;
    double pilot_relative_floor = 1e-4;
    double covariance_floor = 1e-5;
    double covariance_shrinkage = 20.0;
    double fisher_broadening = 1.5;
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

Pilot make_pilot(const Dataset& data, int32_t n_components, int32_t seed,
    int32_t max_iterations = 100, double rho = 0.1,
    double relative_floor = 1e-4);
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
    int32_t n_threads = 1);

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

} // namespace uac
