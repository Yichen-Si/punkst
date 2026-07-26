#pragma once

#include "clustering/uac.hpp"

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

} // namespace uac
