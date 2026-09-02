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
    Flat,
    Hnsw,
    NnDescent
};

enum class SimplexMetric {
    Cosine,
    Hellinger
};

using CosineFlatKernel = KnnFlatKernel;

struct CosineKnnOptions {
    int32_t n_neighbors = 15;
    // nanoflann epsilon; 0 is exact and positive values enable approximate k-NN.
    double knn_search_epsilon = 0.0;
    CosineKnnBackend backend = CosineKnnBackend::Auto;
    CosineFlatKernel flat_kernel = CosineFlatKernel::Auto;
    int32_t n_threads = 1;
    int32_t hnsw_m = 16;
    int32_t hnsw_ef_construction = 100;
    // Zero selects the data-size-aware automatic value and recall tuning.
    int32_t hnsw_ef_search = 0;
    int32_t hnsw_max_ef_search = 512;
    // Zero selects max(64, 4 * n_neighbors).
    int32_t hnsw_candidates = 0;
    int32_t hnsw_audit_queries = 256;
    double hnsw_recall = 0.90;
    bool hnsw_force = false;
    // Zero selects max(10, round(log2(n_rows))).
    int32_t nndescent_iterations = 0;
    // Zero selects max(64, 4 * n_neighbors).
    int32_t nndescent_graph_size = 0;
    int32_t nndescent_sample_candidates = 10;
    int32_t nndescent_audit_queries = 256;
    double nndescent_recall = 0.90;
    int32_t ann_seed = 1;
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
    double audit_seconds = 0.0;
};

struct CosineKnnAuditTrial {
    int32_t parameter = 0;
    double mean_recall = 0.0;
    double recall_lcb = 0.0;
};

struct CosineKnnDiagnostics {
    CosineKnnBackend requested_backend = CosineKnnBackend::Auto;
    CosineKnnBackend resolved_backend = CosineKnnBackend::KdTree;
    CosineFlatKernel resolved_flat_kernel = CosineFlatKernel::Eigen;
    int64_t sample_size = 0;
    int32_t requested_ann_parameter = 0;
    int32_t resolved_ann_parameter = 0;
    int32_t resolved_ann_candidates = 0;
    double audit_mean_recall = 0.0;
    double audit_recall_lcb = 0.0;
    bool audit_passed = false;
    bool forced = false;
    std::vector<CosineKnnAuditTrial> audit_trials;
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

struct CosineDirectedKnnResult {
    DirectedKnnGraph graph;
    CosineKnnDiagnostics diagnostics;
};

const char* cosine_knn_backend_name(CosineKnnBackend backend);
const char* cosine_flat_kernel_name(CosineFlatKernel kernel);
CosineKnnBackend parse_cosine_knn_backend(const std::string& value);
CosineFlatKernel parse_cosine_flat_kernel(const std::string& value);
bool cosine_knn_cblas_available();
bool cosine_knn_faiss_available();

const char* simplex_metric_name(SimplexMetric metric);
SimplexMetric parse_simplex_metric(const std::string& value);

// Embed nonnegative rows on the unit sphere. Cosine uses L2-normalized rows;
// Hellinger L1-normalizes rows and takes component-wise square roots, making
// inner products equal to Bhattacharyya affinities.
RowMajorMatrixXd simplex_metric_coordinates(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric);

RowMajorMatrixXd l2_normalize_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& observations);

// Euclidean k-means on L2-normalized rows (not spherical center updates).
DenseKMeansResult cosine_dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& options);

DenseKMeansResult simplex_dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric, const DenseKMeansOptions& options);

CosineKnnGraph cosine_knn_graph(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    int32_t n_neighbors, double search_epsilon = 0.0);

CosineKnnResult cosine_knn(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const CosineKnnOptions& options);

CosineDirectedKnnResult cosine_directed_knn(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const CosineKnnOptions& options);

CosineDirectedKnnResult simplex_directed_knn(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric, const CosineKnnOptions& options);

CosineKnnResult simplex_knn(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric, const CosineKnnOptions& options);

CosineLeidenResult cosine_leiden_cluster(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const CosineLeidenOptions& options);

CosineLeidenResult simplex_leiden_cluster(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric, const CosineLeidenOptions& options);

Eigen::VectorXi reconcile_cosine_communities(
    const Eigen::VectorXi& membership, int32_t n_communities,
    int32_t requested_communities,
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& kmeans_options);

Eigen::VectorXi reconcile_simplex_communities(
    const Eigen::VectorXi& membership, int32_t n_communities,
    int32_t requested_communities,
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric, const DenseKMeansOptions& kmeans_options);
