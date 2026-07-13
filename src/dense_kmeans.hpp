#pragma once

#include <cstdint>

#include "numerical_utils.hpp"

struct DenseKMeansOptions {
    int32_t n_clusters = 2;
    int32_t max_iterations = 20;
    int32_t seed = 1;
};

struct DenseKMeansResult {
    RowMajorMatrixXd centers;
    Eigen::VectorXi assignments;
    Eigen::VectorXi counts;
    double inertia = 0.0;
    int32_t iterations = 0;
    bool converged = false;
};

DenseKMeansResult dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& options);

DenseKMeansResult sampled_dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& options, int32_t max_samples);
