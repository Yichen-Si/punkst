#pragma once

#include "clustering_core/knn.hpp"

#include <cstdint>
#include <vector>

using FaissRowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic,
    Eigen::Dynamic, Eigen::RowMajor>;

struct FaissAnnAuditTrial {
    int32_t parameter = 0;
    double mean_recall = 0.0;
    double recall_lcb = 0.0;
};

struct FaissAnnCandidateResult {
    std::vector<int32_t> candidates;
    int32_t candidates_per_row = 0;
    int32_t requested_parameter = 0;
    int32_t resolved_parameter = 0;
    int32_t resolved_candidate_count = 0;
    bool audit_passed = false;
    bool forced = false;
    double audit_mean_recall = 0.0;
    double audit_recall_lcb = 0.0;
    double audit_seconds = 0.0;
    double index_build_seconds = 0.0;
    double query_seconds = 0.0;
    std::vector<FaissAnnAuditTrial> audit_trials;
};

struct FaissHnswOptions {
    int32_t m = 16;
    int32_t ef_construction = 100;
    int32_t ef_search = 0;
    int32_t max_ef_search = 512;
    int32_t candidates = 0;
    int32_t audit_queries = 256;
    double recall = 0.90;
    bool force = false;
    int32_t n_threads = 1;
};

struct FaissNnDescentOptions {
    int32_t iterations = 0;
    int32_t graph_size = 0;
    int32_t sample_candidates = 10;
    int32_t audit_queries = 256;
    double recall = 0.90;
    int32_t seed = 1;
    int32_t n_threads = 1;
};

int32_t automatic_nndescent_iterations(int64_t sample_size);
int32_t automatic_hnsw_ef_search(int64_t sample_size);

FaissAnnCandidateResult faiss_hnsw_candidates(
    const Eigen::Ref<const RowMajorMatrixXd>& normalized,
    int32_t neighbors, const FaissHnswOptions& options);

FaissAnnCandidateResult faiss_hnsw_candidates(
    const Eigen::Ref<const FaissRowMajorMatrixXf>& normalized,
    int32_t neighbors, const FaissHnswOptions& options);

FaissAnnCandidateResult faiss_nndescent_candidates(
    const Eigen::Ref<const RowMajorMatrixXd>& normalized,
    int32_t neighbors, const FaissNnDescentOptions& options);

FaissAnnCandidateResult faiss_nndescent_candidates(
    const Eigen::Ref<const FaissRowMajorMatrixXf>& normalized,
    int32_t neighbors, const FaissNnDescentOptions& options);
