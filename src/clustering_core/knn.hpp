#pragma once

// Reusable inner-product k-NN graph utilities.

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "numerical_utils.hpp"

enum class KnnFlatKernel {
    Auto,
    Eigen,
    Cblas
};

// Canonical union-symmetrized k-nearest-neighbor graph. Each edge is listed
// once with first < second and edges are lexicographically sorted. Weights are
// raw similarity scores and may be negative or zero.
struct KnnGraph {
    int32_t n_nodes = 0;
    std::vector<std::pair<int32_t, int32_t>> edges;
    std::vector<double> weights;
};

struct InnerProductKnnOptions {
    int32_t n_neighbors = 15;
    KnnFlatKernel flat_kernel = KnnFlatKernel::Auto;
    int32_t n_threads = 1;
};

struct InnerProductKnnTimings {
    double query_seconds = 0.0;
    double topk_seconds = 0.0;
    double graph_reduction_seconds = 0.0;
};

struct InnerProductKnnDiagnostics {
    KnnFlatKernel resolved_flat_kernel = KnnFlatKernel::Eigen;
    InnerProductKnnTimings timings;
};

struct InnerProductKnnResult {
    KnnGraph graph;
    InnerProductKnnDiagnostics diagnostics;
};

const char* knn_flat_kernel_name(KnnFlatKernel kernel);
KnnFlatKernel parse_knn_flat_kernel(const std::string& value);
bool knn_cblas_available();

// Exact all-points maximum-inner-product k-NN. Rows are not normalized or
// otherwise transformed. Self matches are excluded, score ties are resolved
// by row index, and reciprocal neighborhoods are unioned with maximum score.
InnerProductKnnResult inner_product_knn(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const InnerProductKnnOptions& options);
