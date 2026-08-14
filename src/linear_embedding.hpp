#pragma once

#include "clustering_core/projection.hpp"
#include "numerical_utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace punkst::linear_embedding {

enum class ProjectionSpace {
    Linear,
    Ilr,
};

struct ProjectionData {
    RowMajorMatrixXd centers;
    RowMajorMatrixXd coordinates;
};

struct TopicCenterTable {
    std::vector<std::string> identifiers;
    std::vector<std::string> topics;
    RowMajorMatrixXd values;
};

struct Options {
    std::vector<ProjectionSpace> projection_spaces{ProjectionSpace::Linear};
    projection::VisualizationWhitening whitening =
        projection::VisualizationWhitening::Mixture;
    int32_t dimensions = 4;
    int32_t threads = 1;
    double center_floor = 1e-12;
    double covariance_floor = 1e-5;
    bool include_full = false;
    bool qda_projection = true;
    int32_t qda_train_max_rows = 12000;
    int32_t qda_validation_max_rows = 4000;
    double qda_validation_fraction = 0.20;
    int32_t qda_epochs = 250;
    double qda_learning_rate = 0.03;
    double qda_covariance_shrinkage = 0.10;
    double qda_ridge = 1e-5;
    int32_t qda_restarts = 2;
    int32_t qda_evaluate_every = 5;
    int32_t qda_patience = 12;
    int32_t qda_seed = 1;
    double qda_sparsity_strength = 0.0;
    bool qda_sparsity_cv = false;
    int32_t qda_sparsity_cv_folds = 5;
    std::vector<double> qda_sparsity_grid;

    void validate();
};

const char* projection_space_name(ProjectionSpace space);

ProjectionData prepare_projection(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    ProjectionSpace space,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    double center_floor);

Eigen::MatrixXd aggregate_cluster_factors(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components);

Eigen::MatrixXd aggregate_cluster_factors(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    const std::vector<int32_t>& value_rows,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components);

void write_cluster_factors(
    const std::string& path,
    const std::vector<std::string>& factor_names,
    const Eigen::Ref<const Eigen::MatrixXd>& sums);

void run_partition(
    const TopicCenterTable& theta,
    const std::vector<int32_t>& matched_theta_rows,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components,
    const std::string& partition_label,
    const std::string& output_prefix,
    const Options& options);

} // namespace punkst::linear_embedding
