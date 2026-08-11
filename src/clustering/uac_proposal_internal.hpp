#pragma once

#include "clustering/uac_common_internal.hpp"

#include <cstdint>
#include <vector>

namespace uac {

struct FisherApproximation {
    Eigen::VectorXd gradient;
    Eigen::MatrixXd information;
};

namespace detail {

struct FisherWorkspace {
    Eigen::VectorXd composition;
    Eigen::MatrixXd simplex_covariance;
    Eigen::MatrixXd simplex_derivative;
    Eigen::VectorXd probability;
    Eigen::VectorXd scale;
    RowMajorMatrixXd probability_derivative;
    Eigen::VectorXd derivative;
    Eigen::VectorXd score;
};

struct PilotCache {
    std::vector<Eigen::MatrixXd> inverse_covariances;
    Eigen::VectorXd log_determinants;

    explicit PilotCache(const Pilot& pilot);
};

struct DocumentProposal {
    Eigen::VectorXd weights;
    std::vector<int32_t> component_ids;
    std::vector<Eigen::VectorXd> means;
    std::vector<Eigen::MatrixXd> precision_lower;
    Eigen::VectorXd log_precision_determinants;
    double broadening = 1.0;
    double precision_fallback_seconds = 0.0;
    int64_t precision_fallbacks = 0;
};

struct ProposalScreeningPlan {
    bool enabled = false;
    double planning_seconds = 0.0;
    double predicted_work_ratio = 1.0;
    int32_t active_components = 0;
    std::vector<std::vector<int32_t>> candidates;
    std::vector<int32_t> audit_documents;
    int32_t audit_represented_components = 0;
    int32_t audit_covered_components = 0;
    int32_t audit_violations = 0;
    double maximum_audit_omitted_mass = 0.0;
};

FisherApproximation fisher_approximation_impl(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal, bool compute_gradient = true,
    FisherWorkspace* supplied_workspace = nullptr);
DocumentProposal fisher_proposal(
    const Eigen::Ref<const Eigen::VectorXd>& center,
    const FisherApproximation& fisher, const Document& document,
    const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal_kind, const Pilot& pilot,
    const PilotCache& cache, double broadening,
    int32_t refinement_iterations,
    const std::vector<int32_t>* candidate_components = nullptr);
Eigen::VectorXd proposal_log_density_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    const DocumentProposal& proposal);
void add_screening_metrics(ScoreResult& score,
    const ComponentScreeningOptions& requested,
    const ProposalScreeningPlan& proposal,
    const ComponentScreeningOptions& particle);
double document_effective_total(
    const Dataset& data, int32_t document);
ProposalScreeningPlan make_proposal_screening_plan(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& cache,
    ProposalKind proposal_kind, double broadening,
    int32_t refinement_iterations, uint64_t seed,
    const ComponentScreeningOptions& options,
    const IndexedDocumentSource* count_source = nullptr);

} // namespace detail

FisherApproximation fisher_approximation(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal);

} // namespace uac
