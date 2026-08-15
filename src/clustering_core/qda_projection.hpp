#pragma once

#include "numerical_utils.hpp"

#include <cstdint>

namespace punkst::projection {

enum class DiscriminantModel {
    Qda,
    Lda,
};

struct DiscriminantProjectionOptions {
    DiscriminantModel model = DiscriminantModel::Qda;
    int32_t dimensions = 2;
    int32_t epochs = 250;
    int32_t restarts = 2;
    int32_t evaluate_every = 5;
    int32_t patience_checks = 12;
    int32_t seed = 1;
    int32_t n_threads = 1;
    double learning_rate = 0.03;
    double covariance_shrinkage = 0.10;
    double ridge = 1e-5;
    double improvement_tolerance = 1e-5;
    double sparsity_strength = 0.0;
};

struct DiscriminantProjectionResult {
    Eigen::MatrixXd projection;
    double training_loss = 0.0;
    double validation_loss = 0.0;
    double quartimax_score = 0.0;
    double training_objective = 0.0;
    double validation_objective = 0.0;
    int32_t restart = -1;
    int32_t epoch = -1;
};

DiscriminantProjectionResult fit_discriminant_projection(
    const Eigen::Ref<const RowMajorMatrixXd>& training,
    const Eigen::Ref<const Eigen::VectorXi>& training_labels,
    const Eigen::Ref<const RowMajorMatrixXd>& validation,
    const Eigen::Ref<const Eigen::VectorXi>& validation_labels,
    int32_t components,
    const DiscriminantProjectionOptions& options = {});

double discriminant_projection_log_loss(
    const Eigen::Ref<const Eigen::MatrixXd>& projection,
    const Eigen::Ref<const RowMajorMatrixXd>& training,
    const Eigen::Ref<const Eigen::VectorXi>& training_labels,
    const Eigen::Ref<const RowMajorMatrixXd>& evaluation,
    const Eigen::Ref<const Eigen::VectorXi>& evaluation_labels,
    int32_t components,
    const DiscriminantProjectionOptions& options = {});

using QdaProjectionOptions = DiscriminantProjectionOptions;
using QdaProjectionResult = DiscriminantProjectionResult;

QdaProjectionResult fit_qda_projection(
    const Eigen::Ref<const RowMajorMatrixXd>& training,
    const Eigen::Ref<const Eigen::VectorXi>& training_labels,
    const Eigen::Ref<const RowMajorMatrixXd>& validation,
    const Eigen::Ref<const Eigen::VectorXi>& validation_labels,
    int32_t components,
    const QdaProjectionOptions& options = {});

double qda_projection_log_loss(
    const Eigen::Ref<const Eigen::MatrixXd>& projection,
    const Eigen::Ref<const RowMajorMatrixXd>& training,
    const Eigen::Ref<const Eigen::VectorXi>& training_labels,
    const Eigen::Ref<const RowMajorMatrixXd>& evaluation,
    const Eigen::Ref<const Eigen::VectorXi>& evaluation_labels,
    int32_t components,
    const QdaProjectionOptions& options = {});

namespace testing {

void run_discriminant_projection_gradient_tests();

} // namespace testing

} // namespace punkst::projection
