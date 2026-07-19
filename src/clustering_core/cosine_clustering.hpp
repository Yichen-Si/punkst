#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "clustering_core/dense_kmeans.hpp"
#include "clustering_core/knn.hpp"
#include "clustering_core/leiden.hpp"

enum class CosineKnnBackend {
    Auto,
    KdTree,
    Flat
};

using CosineFlatKernel = KnnFlatKernel;

struct CosineKnnOptions {
    int32_t n_neighbors = 15;
    // nanoflann epsilon; 0 is exact and positive values enable approximate k-NN.
    double knn_search_epsilon = 0.0;
    CosineKnnBackend backend = CosineKnnBackend::Auto;
    CosineFlatKernel flat_kernel = CosineFlatKernel::Auto;
    int32_t n_threads = 1;
};

struct CosineLeidenOptions : CosineKnnOptions {
    LeidenOptions leiden;
};

struct CosineKnnTimings {
    double normalization_seconds = 0.0;
    double index_build_seconds = 0.0;
    double query_seconds = 0.0;
    double topk_seconds = 0.0;
    double graph_reduction_seconds = 0.0;
};

struct CosineKnnDiagnostics {
    CosineKnnBackend requested_backend = CosineKnnBackend::Auto;
    CosineKnnBackend resolved_backend = CosineKnnBackend::KdTree;
    CosineFlatKernel resolved_flat_kernel = CosineFlatKernel::Eigen;
    CosineKnnTimings timings;
};

struct CosineLeidenResult {
    LeidenResult clustering;
    int64_t n_edges = 0;
    CosineKnnDiagnostics knn;
};

// Canonical union-symmetrized cosine k-nearest-neighbor graph. Each edge is
// listed once with first < second; edges are lexicographically sorted and
// weights are positive cosine similarities.
using CosineKnnGraph = KnnGraph;

struct CosineKnnResult {
    CosineKnnGraph graph;
    CosineKnnDiagnostics diagnostics;
};

const char* cosine_knn_backend_name(CosineKnnBackend backend);
const char* cosine_flat_kernel_name(CosineFlatKernel kernel);
CosineKnnBackend parse_cosine_knn_backend(const std::string& value);
CosineFlatKernel parse_cosine_flat_kernel(const std::string& value);
bool cosine_knn_cblas_available();

RowMajorMatrixXd l2_normalize_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& observations);

// Euclidean k-means on L2-normalized rows (not spherical center updates).
DenseKMeansResult cosine_dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& options);

CosineKnnGraph cosine_knn_graph(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    int32_t n_neighbors, double search_epsilon = 0.0);

CosineKnnResult cosine_knn(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const CosineKnnOptions& options);

CosineLeidenResult cosine_leiden_cluster(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const CosineLeidenOptions& options);

Eigen::VectorXi reconcile_cosine_communities(
    const Eigen::VectorXi& membership, int32_t n_communities,
    int32_t requested_communities,
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& kmeans_options);
