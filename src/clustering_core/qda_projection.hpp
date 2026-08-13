#pragma once

#include "numerical_utils.hpp"

#include <cstdint>

namespace punkst::projection {

struct QdaProjectionOptions {
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
};

struct QdaProjectionResult {
    Eigen::MatrixXd projection;
    double training_loss = 0.0;
    double validation_loss = 0.0;
    int32_t restart = -1;
    int32_t epoch = -1;
};

QdaProjectionResult fit_qda_projection(
    const Eigen::Ref<const RowMajorMatrixXd>& training,
    const Eigen::Ref<const Eigen::VectorXi>& training_labels,
    const Eigen::Ref<const RowMajorMatrixXd>& validation,
    const Eigen::Ref<const Eigen::VectorXi>& validation_labels,
    int32_t components,
    const QdaProjectionOptions& options = {});

} // namespace punkst::projection
