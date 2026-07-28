#pragma once

#include "clustering/uac_proposal_internal.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace uac {

struct ParticleSet {
    int32_t first_document = 0;
    int32_t documents = 0;
    int32_t samples = 0;
    int32_t dimension = 0;
    RowMajorMatrixXd values;
    RowMajorMatrixXd log_likelihood;
    RowMajorMatrixXd log_proposal;
    std::vector<int32_t> proposal_origins;
    std::vector<int32_t> proposal_candidates;
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    double fisher_work_seconds = 0.0;
    double proposal_component_work_seconds = 0.0;
    double proposal_draw_density_work_seconds = 0.0;
    double proposal_precision_fallback_seconds = 0.0;
    int64_t proposal_precision_fallbacks = 0;
    int64_t proposal_components_constructed = 0;
    int64_t proposal_components_possible = 0;
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
    Eigen::Map<const Eigen::VectorXi> proposal_origins_for_document(
        int32_t document) const;
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
    std::vector<int32_t> proposal_origins;
    std::vector<int32_t> proposal_candidates;
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    double calibration_seconds = 0.0;
    double fisher_work_seconds = 0.0;
    double proposal_component_work_seconds = 0.0;
    double proposal_draw_density_work_seconds = 0.0;
    double proposal_precision_fallback_seconds = 0.0;
    int64_t proposal_precision_fallbacks = 0;
    double proposal_screening_seconds = 0.0;
    int64_t proposal_components_constructed = 0;
    int64_t proposal_components_possible = 0;
    int32_t proposal_audit_documents = 0;
    int32_t proposal_audit_violations = 0;
    double proposal_audit_maximum_omitted_mass = 0.0;
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
    Eigen::Map<const Eigen::VectorXi> proposal_origins_for_document(
        int32_t document) const;
};

namespace detail {

ParticleSet make_particle_range(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    const PilotCache& pilot_cache, ProposalKind proposal_kind,
    int32_t samples, uint64_t seed, double fisher_broadening,
    int32_t n_threads, const ProposalScreeningPlan* screening_plan,
    int32_t first_document, int32_t documents,
    int32_t global_first_document = -1);


template<class Matrix>
void draw_proposal_values(const DocumentProposal& proposal,
    uint64_t document_seed, Matrix& values, int32_t* origins = nullptr) {
    std::mt19937_64 engine(document_seed);
    std::discrete_distribution<int32_t> choose(proposal.weights.data(),
        proposal.weights.data() + proposal.weights.size());
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd draw(values.cols());
    for (Eigen::Index sample = 0; sample < values.rows(); ++sample) {
        const int32_t component = choose(engine);
        if (origins) origins[sample] = proposal.component_ids[component];
        for (Eigen::Index dim = 0; dim < values.cols(); ++dim) {
            draw(dim) = normal(engine);
        }
        proposal.precision_lower[component].transpose()
            .triangularView<Eigen::Upper>().solveInPlace(draw);
        values.row(sample) = (proposal.means[component]
            + std::sqrt(proposal.broadening) * draw).transpose();
    }
}

RaggedParticleSet make_adaptive_particle_range(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal_kind, uint64_t seed, double fisher_broadening,
    int32_t n_threads, const Model& calibration_model,
    const AdaptiveParticleOptions& options, int32_t maximum_particles,
    const ProposalScreeningPlan* screening_plan,
    int32_t first_document, int32_t documents,
    int32_t global_first_document = -1);
RaggedParticleSet make_adaptive_particles(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal_kind, uint64_t seed, double fisher_broadening,
    int32_t n_threads, const Model& calibration_model,
    const AdaptiveParticleOptions& options, int32_t maximum_particles,
    const ProposalScreeningPlan* screening_plan);

} // namespace detail

ParticleSet make_particles(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    ProposalKind proposal, int32_t samples, uint64_t seed,
    double fisher_broadening = 1.5, int32_t n_threads = 1);

} // namespace uac
