#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "numerical_utils.hpp"
#include "clustering_core/cosine_clustering.hpp"
#include "gamma_pois_posterior_io.hpp"
#include "clustering/low_rank_covariance.hpp"

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
    // Posterior mean topic proportions, normalized to sum to one per document.
    RowMajorMatrixXd topic_mean;
    RowMajorMatrixXd posterior_shape;
    RowMajorMatrixXd posterior_rate;
    Eigen::VectorXd topic_capacity;
    std::vector<uint64_t> posterior_rows;
    std::vector<GammaPoissonTopicCovariance> topic_covariance;
    // Optional transient-dispersion covariance in log-topic coordinates.
    std::vector<GammaPoissonTopicCovariance> dispersion_covariance;
    enum class UncertaintyModel {
        MeanField,
        MeanFieldPlusDispersionDiagonal,
        MeanFieldPlusDispersionLowRank
    } uncertainty_model = UncertaintyModel::MeanField;
};

struct GammaPoissonPowerIlrOptions {
    double lambda = 0.5;
    int32_t samples = 16;
    int32_t covariance_rank = 8;
    int32_t n_threads = 1;
    uint64_t seed = 1;
};

struct GammaPoissonCoordinateDiagnostics {
    int64_t fallback_rows = 0;
    double mean_covariance_trace = 0.0;
    double observed_coordinate_trace = 0.0;
};

struct GammaPoissonClusterCoordinates {
    GammaPoissonLogRatioBasis log_ratio_basis;
    RowMajorMatrixXd mean;
    // Plain topic proportions used by cosine initializers. Empty for callers
    // that construct Gaussian coordinates directly.
    RowMajorMatrixXd initialization_mean;
    RowMajorMatrixXd uncertainty_diagonal;
    RowMajorMatrixXd uncertainty_factor;
    int32_t uncertainty_rank = 0;
    GammaPoissonCoordinateDiagnostics diagnostics;

    Eigen::Map<const RowMajorMatrixXd> factor(Eigen::Index document) const;
};

GammaPoissonClusterDataset load_gamma_poisson_cluster_dataset(
    const std::string& posterior_path, const std::string& dispersion_path,
    uint64_t expected_state_checksum,
    const Eigen::Ref<const Eigen::VectorXd>& topic_capacity,
    const std::vector<std::string>& expected_topic_names,
    const std::string& coordinate_model = "");

GammaPoissonClusterCoordinates make_gamma_poisson_cluster_coordinates(
    const GammaPoissonClusterDataset& dataset);

GammaPoissonClusterCoordinates make_gamma_poisson_cluster_coordinates(
    const GammaPoissonClusterDataset& dataset,
    const Eigen::Ref<const Eigen::MatrixXd>& basis);

GammaPoissonClusterCoordinates make_gamma_poisson_theta_l2_coordinates(
    const GammaPoissonClusterDataset& dataset);

GammaPoissonClusterCoordinates make_gamma_poisson_power_ilr_coordinates(
    const GammaPoissonClusterDataset& dataset,
    const GammaPoissonPowerIlrOptions& options);

Eigen::VectorXd gamma_poisson_power_ilr_transform(
    const Eigen::Ref<const Eigen::VectorXd>& topic_intensity,
    double lambda);

Eigen::VectorXd gamma_poisson_power_ilr_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& basis,
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    double lambda);

struct GammaPoissonClusterFitOptions {
    enum class Initializer {
        KMeans,
        Leiden
    } initializer = Initializer::KMeans;
    enum class Optimizer {
        Batch,
        Svi
    } optimizer = Optimizer::Svi;
    enum class CovarianceAccumulation {
        Auto,
        Dense,
        Compact
    } covariance_accumulation = CovarianceAccumulation::Auto;
    enum class CandidateSearch {
        Auto,
        Linear,
        KdTree
    } candidate_search = CandidateSearch::Auto;
    int32_t n_components = 10;
    int32_t max_iterations = 50;
    int32_t kmeans_max_iterations = 20;
    int32_t leiden_neighbors = 15;
    int32_t leiden_max_iterations = -1;
    CosineKnnBackend leiden_knn_backend = CosineKnnBackend::Auto;
    double leiden_knn_epsilon = 0.0;
    int32_t convergence_patience = 3;
    int32_t n_threads = 1;
    int32_t cluster_covariance_rank = 0;
    int32_t diagonal_warmup_iterations = 5;
    int32_t orientation_update_interval = 1;
    int32_t orientation_max_updates = 5;
    int32_t orientation_patience = 2;
    int32_t minibatch_size = 1024;
    int32_t n_epochs = 30;
    int32_t svi_eval_size = 4096;
    int32_t refine_max_iterations = 20;
    int32_t candidate_components = 0;
    int32_t candidate_dimensions = 0;
    int32_t candidate_refresh_epochs = 5;
    int32_t prune_patience = 0;
    int32_t seed = 1;
    int32_t verbose = 0;
    double dirichlet_concentration = 1.0;
    double leiden_resolution = 1.0;
    double variance_floor = 1e-4;
    double tolerance = 1e-5;
    double responsibility_p90_tolerance = 0.01;
    double top_assignment_change_tolerance = 1e-3;
    double low_rank_variance_floor = 1e-6;
    double orientation_tolerance = 1e-3;
    double orientation_step = 0.5;
    double svi_kappa = 0.7;
    double svi_tau0 = 10.0;
    double min_cluster_size = 5.0;
};

struct GammaPoissonResponsibilityChange {
    double mean_linf = 0.0;
    double p90_linf = 0.0;
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

struct GammaPoissonClusterSufficientStatistics {
    Eigen::VectorXd membership;
    RowMajorMatrixXd first;
    RowMajorMatrixXd second_diagonal;
    std::vector<Eigen::MatrixXd> second_projected;
    Eigen::MatrixXd pooled_second_sketch;
    std::vector<Eigen::MatrixXd> second_dense;
    bool dense = false;
    double elbo_local = 0.0;
    double predictive_log_likelihood = 0.0;

    GammaPoissonClusterSufficientStatistics() = default;
    GammaPoissonClusterSufficientStatistics(int32_t components, int32_t dim,
        int32_t rank, int32_t sketch_size, bool dense_ = false);

    void add_scaled(const GammaPoissonClusterSufficientStatistics& other,
        double scale);
    void scale(double factor);
    void interpolate(
        const GammaPoissonClusterSufficientStatistics& target, double step);
};

struct GammaPoissonClusterBatchExpectation {
    GammaPoissonClusterSufficientStatistics statistics;
    RowMajorMatrixXd responsibilities;
};

GammaPoissonClusterBatchExpectation gamma_poisson_cluster_e_step(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model,
    const Eigen::Ref<const Eigen::VectorXi>& document_indices,
    int32_t n_threads, bool dense_accumulation,
    bool collect_responsibilities = true);

struct GammaPoissonClusterFitDiagnostics {
    std::string optimizer = "batch";
    std::string initializer = "kmeans++";
    std::string covariance_accumulation = "diagonal";
    std::string candidate_search = "none";
    std::string knn_backend_requested = "auto";
    std::string knn_backend_resolved = "none";
    double elbo = 0.0;
    double log_likelihood = 0.0;
    double relative_elbo_change = std::numeric_limits<double>::infinity();
    double relative_predictive_log_likelihood_change =
        std::numeric_limits<double>::infinity();
    double mean_responsibility_linf_change = std::numeric_limits<double>::infinity();
    double p90_responsibility_linf_change = std::numeric_limits<double>::infinity();
    double top_assignment_change_fraction = 1.0;
    double max_standardized_center_change = std::numeric_limits<double>::infinity();
    double max_weight_change = std::numeric_limits<double>::infinity();
    double max_log_variance_change = std::numeric_limits<double>::infinity();
    double kmeans_inertia = 0.0;
    double initializer_quality = 0.0;
    double knn_search_epsilon = 0.0;
    double orientation_change = std::numeric_limits<double>::infinity();
    std::vector<double> elbo_trace;
    std::vector<double> predictive_log_likelihood_trace;
    std::vector<double> mean_responsibility_entropy_trace;
    int32_t iterations = 0;
    int32_t kmeans_iterations = 0;
    int32_t initializer_communities = 0;
    int32_t knn_neighbors = 0;
    int32_t warmup_iterations = 0;
    int32_t structured_iterations = 0;
    int32_t orientation_updates = 0;
    int32_t epochs = 0;
    int32_t svi_updates = 0;
    int32_t refinement_iterations = 0;
    int32_t candidate_components = 0;
    int32_t candidate_dimensions = 0;
    int32_t full_refreshes = 0;
    bool kmeans_converged = false;
    bool orientation_converged = false;
    bool structured_covariance_fallback = false;
    bool svi_converged = false;
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
    std::string coordinate_model = "log-ratio";
    double power_ilr_lambda = 0.5;
    int32_t posterior_coordinate_samples = 16;
    int32_t coordinate_covariance_rank = 0;
    uint64_t coordinate_seed = 1;
    std::string coordinate_sampler = "none";
    std::string coordinate_rotation = "none";
    int64_t coordinate_fallback_rows = 0;
    double mean_document_uncertainty_trace = 0.0;
    double observed_coordinate_trace = 0.0;
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
