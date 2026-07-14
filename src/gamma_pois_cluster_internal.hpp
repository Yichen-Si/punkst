#pragma once

#include "gamma_pois_cluster.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace gamma_pois_cluster_detail {

class DormancyTracker {
public:
    explicit DormancyTracker(int32_t components);
    void observe_exact(const Eigen::Ref<const Eigen::VectorXd>& membership,
        double min_cluster_size, int32_t patience, bool update_status);
    Eigen::VectorXd preserve_exact_weights(
        const Eigen::Ref<const Eigen::VectorXd>& membership) const;

    const std::vector<uint8_t>& dormant() const { return dormant_; }
    const Eigen::VectorXd& last_exact_membership() const {
        return last_exact_membership_;
    }

private:
    std::vector<uint8_t> dormant_;
    std::vector<int32_t> low_mass_refreshes_;
    Eigen::VectorXd last_exact_membership_;
};

using ClusterStatistics = GammaPoissonClusterSufficientStatistics;

// Relative change statistics used for convergence, for the periodic notice.
// relative_label names the first quantity ("dELBO" for batch EM,
// "dPredLL" for SVI); the other two are shared across optimizers.
struct ClusterConvergenceStats {
    double relative_change = 0.0;
    double p90_responsibility_change = 0.0;
    double top_assignment_change = 0.0;
    const char* relative_label = "dRel";
};

// Emit a NOTICE with the top-10 cluster sizes as fractions of the total unit
// count (expected membership mass per cluster / n_units), for periodic
// progress. When convergence != nullptr, also append the relative change
// statistics used to determine convergence.
void log_top_cluster_size_fractions(
    const Eigen::Ref<const Eigen::VectorXd>& membership, int32_t n_units,
    const std::string& stage,
    const ClusterConvergenceStats* convergence = nullptr);

struct PreparedComponent {
    Eigen::VectorXd diagonal;
    RowMajorMatrixXd factor;
    Eigen::VectorXd covariance_diagonal;
    Eigen::MatrixXd covariance;
};

struct PreparedEStepModel {
    Eigen::VectorXd expected_log;
    Eigen::VectorXd weights;
    std::vector<PreparedComponent> components;
    RowMajorMatrixXd marginal_variances;
};

struct EStepRequest {
    const int32_t* document_indices = nullptr;
    int32_t n_documents = 0;
    int32_t n_threads = 1;
    RowMajorMatrixXd* responsibilities = nullptr;
    const Eigen::MatrixXd* sketch = nullptr;
    bool collect_moments = true;
    bool dense_accumulation = false;
    const int32_t* candidates = nullptr;
    int32_t candidate_stride = 0;
    int32_t* document_top_components = nullptr;
};

struct InitializedClusterFit {
    GammaPoissonClusterModel model;
    Eigen::MatrixXd initial_shared;
    int32_t kmeans_iterations = 0;
    double kmeans_inertia = 0.0;
    bool kmeans_converged = false;
};

RowMajorMatrixXd covariance_factor(
    const Eigen::Ref<const RowMajorMatrixXd>& orientation,
    const Eigen::Ref<const Eigen::RowVectorXd>& variances);
RowMajorMatrixXd leading_orientation(
    const Eigen::Ref<const Eigen::MatrixXd>& covariance, int32_t rank);
double align_and_blend_orientation(RowMajorMatrixXd& orientation,
    RowMajorMatrixXd candidate, double step);

PreparedEStepModel prepare_e_step_model(const GammaPoissonClusterModel& model,
    bool prepare_dense_covariance);

std::vector<int32_t> select_candidate_components_linear(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model,
    const PreparedEStepModel& prepared_model, const int32_t* documents,
    int32_t n_documents, int32_t candidate_count, int32_t candidate_dimensions,
    const std::vector<int32_t>& previous_top,
    const std::vector<uint8_t>& dormant);
std::vector<int32_t> select_candidate_components_kdtree(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model,
    const PreparedEStepModel& prepared_model, const int32_t* documents,
    int32_t n_documents, int32_t candidate_count, int32_t candidate_dimensions,
    int32_t pool_multiplier, const std::vector<int32_t>& previous_top,
    const std::vector<uint8_t>& dormant);

ClusterStatistics run_e_step_prepared(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, const PreparedEStepModel& prepared,
    const EStepRequest& request);
ClusterStatistics run_e_step(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, const EStepRequest& request);

void run_m_step(const ClusterStatistics& stats, GammaPoissonClusterModel& model,
    double alpha, const GammaPoissonClusterFitOptions& options,
    GammaPoissonClusterFitDiagnostics& diagnostics,
    const std::vector<uint8_t>* dormant = nullptr,
    const Eigen::VectorXd* weight_membership = nullptr);

Eigen::MatrixXd make_orientation_sketch(
    int32_t dim, int32_t rank, int32_t seed);
RowMajorMatrixXd orientation_from_signed_sketch(
    const Eigen::Ref<const Eigen::MatrixXd>& sketch,
    const Eigen::Ref<const Eigen::MatrixXd>& action,
    const Eigen::Ref<const RowMajorMatrixXd>& fallback, int32_t rank);
Eigen::MatrixXd pooled_residual_sketch(const ClusterStatistics& stats,
    const GammaPoissonClusterModel& model,
    const Eigen::Ref<const Eigen::MatrixXd>& sketch);
Eigen::MatrixXd pooled_residual_covariance(const ClusterStatistics& stats,
    const GammaPoissonClusterModel& model);
void transport_orientation(GammaPoissonClusterModel& model,
    const Eigen::Ref<const RowMajorMatrixXd>& previous,
    const GammaPoissonClusterFitOptions& options);

void update_diagnostics(const ClusterStatistics& stats, double alpha,
    GammaPoissonClusterFitResult& out);
void validate_common_fit_input(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options);
bool select_dense_accumulation(const GammaPoissonClusterFitOptions& options,
    int32_t dim, int32_t rank);
InitializedClusterFit initialize_cluster_fit(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options, bool sampled);
void initialize_fit_diagnostics(const InitializedClusterFit& initialized,
    const char* optimizer, bool dense_accumulation, int32_t rank,
    GammaPoissonClusterFitResult& out);
void run_exact_em(const GammaPoissonClusterCoordinates& coordinates,
    GammaPoissonClusterModel& model, const GammaPoissonClusterFitOptions& options,
    double alpha, bool dense_accumulation, int32_t max_iterations,
    bool refinement, GammaPoissonClusterFitResult& out);
void validate_model(const GammaPoissonClusterModel& model, int32_t dim);

GammaPoissonClusterFitResult fit_batch(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options);
GammaPoissonClusterFitResult fit_svi(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options);

} // namespace gamma_pois_cluster_detail
