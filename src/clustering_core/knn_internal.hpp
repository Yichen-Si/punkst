#pragma once

#include <cstdint>
#include <vector>

#include "clustering_core/knn.hpp"

namespace knn_detail {

struct DirectedNeighbor {
    int32_t index = -1;
    double similarity = 0.0;
};

// Shared implementation used by raw inner-product and normalized-cosine flat
// search. clamp_unit_scores preserves cosine's [-1,1] roundoff contract.
std::vector<DirectedNeighbor> flat_inner_product_neighbors(
    const RowMajorMatrixXd& observations, int32_t neighbors,
    KnnFlatKernel kernel, int32_t n_threads, bool clamp_unit_scores,
    InnerProductKnnTimings& timings);

KnnGraph union_max_knn_graph(
    const std::vector<DirectedNeighbor>& directed,
    int32_t n, int32_t neighbors);

} // namespace knn_detail
