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

// Deterministic k-means++ followed by Lloyd updates. Empty clusters are
// repaired by moving the farthest observation from a non-singleton donor.
DenseKMeansResult dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& options);

// Fit on a deterministic reservoir sample, then perform one assignment,
// empty-cluster repair, and centroid update on all rows. When sampling is
// actually used, converged is false because the full data did not run Lloyd
// iterations to convergence; iterations reports the sample fit iterations.
DenseKMeansResult sampled_dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& options, int32_t max_samples);
