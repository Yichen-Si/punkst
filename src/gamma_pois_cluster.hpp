#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "numerical_utils.hpp"
#include "gamma_pois_posterior_io.hpp"
#include "low_rank_covariance.hpp"

struct GammaPoissonLogMoments {
    Eigen::VectorXd mean;
    Eigen::VectorXd variance;
};

struct GammaPoissonTopicCovariance {
    Eigen::VectorXd diagonal;
    RowMajorMatrixXd factor;

    Eigen::MatrixXd dense() const;
    Eigen::MatrixXd transformed(const Eigen::Ref<const Eigen::MatrixXd>& basis) const;
};

struct GammaPoissonLogRatioBasis {
    Eigen::MatrixXd helmert;
    Eigen::MatrixXd rotation;
    Eigen::MatrixXd basis;
    Eigen::VectorXd mean_helmert;
    Eigen::VectorXd signal_eigenvalues;
    Eigen::MatrixXd observed_covariance;
    Eigen::MatrixXd mean_uncertainty;
    Eigen::MatrixXd signal_covariance;
};

GammaPoissonLogMoments gamma_poisson_log_moments(
    const Eigen::Ref<const Eigen::VectorXd>& shape,
    const Eigen::Ref<const Eigen::VectorXd>& rate,
    const Eigen::Ref<const Eigen::VectorXd>& topic_capacity);

Eigen::MatrixXd normalized_helmert_basis(int32_t n_topics);

class GammaPoissonBasisAccumulator {
public:
    explicit GammaPoissonBasisAccumulator(int32_t n_topics);

    void add(const Eigen::Ref<const Eigen::VectorXd>& log_mean,
        const GammaPoissonTopicCovariance& covariance);
    int64_t size() const { return count_; }
    GammaPoissonLogRatioBasis finish() const;

private:
    int32_t n_topics_;
    int32_t dim_;
    int64_t count_ = 0;
    Eigen::MatrixXd helmert_;
    Eigen::VectorXd sum_u_;
    Eigen::MatrixXd sum_uu_;
    Eigen::MatrixXd sum_uncertainty_;
};

struct GammaPoissonClusterDataset {
    GammaPoissonPosteriorHeader posterior_header;
    std::vector<std::string> identifiers;
    RowMajorMatrixXd log_mean;
    std::vector<GammaPoissonTopicCovariance> topic_covariance;
    enum class UncertaintyModel {
        MeanField,
        MeanFieldPlusDispersionDiagonal,
        MeanFieldPlusDispersionLowRank
    } uncertainty_model = UncertaintyModel::MeanField;
};

struct GammaPoissonClusterCoordinates {
    GammaPoissonLogRatioBasis log_ratio_basis;
    RowMajorMatrixXd mean;
    RowMajorMatrixXd uncertainty_diagonal;
    RowMajorMatrixXd uncertainty_factor;
    int32_t uncertainty_rank = 0;

    Eigen::Map<const RowMajorMatrixXd> factor(Eigen::Index document) const;
};

GammaPoissonClusterDataset load_gamma_poisson_cluster_dataset(
    const std::string& posterior_path, const std::string& dispersion_path,
    uint64_t expected_state_checksum,
    const Eigen::Ref<const Eigen::VectorXd>& topic_capacity,
    const std::vector<std::string>& expected_topic_names);

GammaPoissonClusterCoordinates make_gamma_poisson_cluster_coordinates(
    const GammaPoissonClusterDataset& dataset);

GammaPoissonClusterCoordinates make_gamma_poisson_cluster_coordinates(
    const GammaPoissonClusterDataset& dataset,
    const Eigen::Ref<const Eigen::MatrixXd>& basis);

struct GammaPoissonClusterFitOptions {
    enum class CovarianceAccumulation {
        Auto,
        Dense,
        Compact
    } covariance_accumulation = CovarianceAccumulation::Auto;
    int32_t n_components = 10;
    int32_t max_iterations = 100;
    int32_t kmeans_max_iterations = 20;
    int32_t convergence_patience = 5;
    int32_t n_threads = 1;
    int32_t cluster_covariance_rank = 0;
    int32_t diagonal_warmup_iterations = 50;
    int32_t orientation_update_interval = 10;
    int32_t orientation_max_updates = 10;
    int32_t orientation_patience = 2;
    int32_t seed = 1;
    double dirichlet_concentration = 1.0;
    double variance_floor = 1e-4;
    double tolerance = 1e-6;
    double responsibility_p90_tolerance = 1e-3;
    double top_assignment_change_tolerance = 1e-3;
    double low_rank_variance_floor = 1e-6;
    double orientation_tolerance = 1e-3;
    double orientation_step = 0.5;
};

struct GammaPoissonResponsibilityChange {
    double mean_l1 = 0.0;
    double p90_l1 = 0.0;
    double top_assignment_fraction = 0.0;
};

GammaPoissonResponsibilityChange gamma_poisson_responsibility_change(
    const Eigen::Ref<const RowMajorMatrixXd>& previous,
    const Eigen::Ref<const RowMajorMatrixXd>& current);

struct GammaPoissonConditionalMoments {
    double log_likelihood = 0.0;
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
};

GammaPoissonConditionalMoments gamma_poisson_conditional_moments(
    const Eigen::Ref<const Eigen::VectorXd>& observation,
    const Eigen::Ref<const Eigen::VectorXd>& cluster_mean,
    const LowRankDiagonalCovariance& document_covariance,
    const LowRankDiagonalCovariance& cluster_covariance);

struct GammaPoissonClusterModel {
    Eigen::VectorXd dirichlet_parameters;
    RowMajorMatrixXd means;
    RowMajorMatrixXd variances;
    RowMajorMatrixXd orientation;
    RowMajorMatrixXd low_rank_variances;
};

struct GammaPoissonClusterFitDiagnostics {
    std::string covariance_accumulation = "diagonal";
    double elbo = 0.0;
    double log_likelihood = 0.0;
    double relative_elbo_change = std::numeric_limits<double>::infinity();
    double mean_responsibility_l1_change = std::numeric_limits<double>::infinity();
    double p90_responsibility_l1_change = std::numeric_limits<double>::infinity();
    double top_assignment_change_fraction = 1.0;
    double max_standardized_center_change = std::numeric_limits<double>::infinity();
    double max_weight_change = std::numeric_limits<double>::infinity();
    double max_log_variance_change = std::numeric_limits<double>::infinity();
    double kmeans_inertia = 0.0;
    double orientation_change = std::numeric_limits<double>::infinity();
    std::vector<double> elbo_trace;
    int32_t iterations = 0;
    int32_t kmeans_iterations = 0;
    int32_t warmup_iterations = 0;
    int32_t structured_iterations = 0;
    int32_t orientation_updates = 0;
    bool kmeans_converged = false;
    bool orientation_converged = false;
    bool converged = false;
};

struct GammaPoissonClusterFitResult {
    GammaPoissonClusterModel model;
    RowMajorMatrixXd responsibilities;
    Eigen::VectorXd effective_membership;
    GammaPoissonClusterFitDiagnostics diagnostics;
};

struct GammaPoissonClusterState {
    uint64_t topic_state_checksum = 0;
    std::vector<std::string> topic_names;
    Eigen::MatrixXd basis;
    GammaPoissonClusterModel model;
    GammaPoissonClusterFitDiagnostics diagnostics;
    std::vector<uint8_t> active;
    GammaPoissonClusterDataset::UncertaintyModel document_uncertainty_model =
        GammaPoissonClusterDataset::UncertaintyModel::MeanField;
    int32_t document_uncertainty_rank = 0;
    double min_cluster_size = 5.0;
    std::string input_row_order = "input";
};

struct GammaPoissonClusterScore {
    RowMajorMatrixXd responsibilities;
    double predictive_log_likelihood = 0.0;
};

struct GammaPoissonClusterSeparation {
    double standardized_distance = 0.0;
    double bhattacharyya_distance = 0.0;
};

void write_gamma_poisson_cluster_state(const std::string& path,
    const GammaPoissonClusterState& state);

GammaPoissonClusterState read_gamma_poisson_cluster_state(
    const std::string& path, uint64_t expected_topic_state_checksum = 0);

GammaPoissonClusterScore score_gamma_poisson_cluster_mixture(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, int32_t n_threads);

Eigen::VectorXd gamma_poisson_cluster_weights(
    const GammaPoissonClusterModel& model);

double gamma_poisson_cluster_log_volume(
    const GammaPoissonClusterModel& model, int32_t component);

GammaPoissonClusterSeparation gamma_poisson_cluster_separation(
    const GammaPoissonClusterModel& model, int32_t first, int32_t second);

std::vector<uint8_t> gamma_poisson_active_components(
    const Eigen::Ref<const Eigen::VectorXd>& expected_size,
    double min_cluster_size);

Eigen::VectorXd gamma_poisson_effective_membership_size(
    const Eigen::Ref<const RowMajorMatrixXd>& responsibilities);

GammaPoissonClusterFitResult fit_gamma_poisson_cluster_mixture(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options);
