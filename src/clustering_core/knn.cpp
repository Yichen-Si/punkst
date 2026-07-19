#include "clustering_core/knn.hpp"

#include "clustering_core/knn_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/parallel_sort.h>

#if PUNKST_HAVE_CBLAS
#include <cblas.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point& begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

struct Candidate {
    int32_t index = -1;
    double similarity = -std::numeric_limits<double>::infinity();
};

bool candidate_better(const Candidate& first, const Candidate& second) {
    return first.similarity > second.similarity
        || (first.similarity == second.similarity
            && first.index < second.index);
}

struct CandidateBetter {
    bool operator()(const Candidate& first, const Candidate& second) const {
        return candidate_better(first, second);
    }
};

using CandidateHeap = std::priority_queue<
    Candidate, std::vector<Candidate>, CandidateBetter>;

void retain_candidate(CandidateHeap& heap, int32_t capacity,
                      const Candidate& candidate) {
    if (static_cast<int32_t>(heap.size()) < capacity) {
        heap.push(candidate);
    } else if (candidate_better(candidate, heap.top())) {
        heap.pop();
        heap.push(candidate);
    }
}

struct WeightedEdge {
    int32_t first = 0;
    int32_t second = 0;
    double weight = 0.0;
};

bool weighted_edge_less(const WeightedEdge& first, const WeightedEdge& second) {
    if (first.first != second.first) return first.first < second.first;
    if (first.second != second.second) return first.second < second.second;
    return first.weight < second.weight;
}

KnnFlatKernel resolve_flat_kernel(KnnFlatKernel requested) {
    if (requested == KnnFlatKernel::Auto) return KnnFlatKernel::Eigen;
    if (requested == KnnFlatKernel::Cblas) {
#if !PUNKST_HAVE_CBLAS
        throw std::invalid_argument(
            "The CBLAS inner-product k-NN kernel is not available in this build");
#endif
    }
    return requested;
}

void validate_inner_product_options(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const InnerProductKnnOptions& options) {
    if (observations.rows() < 2 || observations.cols() <= 0
        || !observations.allFinite() || options.n_neighbors <= 0
        || options.n_threads <= 0
        || static_cast<int32_t>(options.flat_kernel)
            < static_cast<int32_t>(KnnFlatKernel::Auto)
        || static_cast<int32_t>(options.flat_kernel)
            > static_cast<int32_t>(KnnFlatKernel::Cblas)) {
        throw std::invalid_argument(
            "Inner-product k-NN requires at least two finite rows, at least "
            "one column, and positive neighbors and threads");
    }
    if (observations.rows() > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(
            "Inner-product k-NN row count exceeds int32 range");
    }
}

#if PUNKST_HAVE_OPENBLAS
class ScopedOpenBlasThreads {
public:
    explicit ScopedOpenBlasThreads(int32_t threads)
        : previous_(openblas_get_num_threads()) {
        openblas_set_num_threads(threads);
    }
    ~ScopedOpenBlasThreads() { openblas_set_num_threads(previous_); }
private:
    int previous_ = 1;
};
#endif

} // namespace

namespace knn_detail {

std::vector<DirectedNeighbor> flat_inner_product_neighbors(
    const RowMajorMatrixXd& observations, int32_t neighbors,
    KnnFlatKernel kernel, int32_t n_threads, bool clamp_unit_scores,
    InnerProductKnnTimings& timings) {
    constexpr int32_t query_tile = 512;
    constexpr int32_t database_tile = 4096;
    const int32_t n = static_cast<int32_t>(observations.rows());
    const int32_t dimensions = static_cast<int32_t>(observations.cols());
    std::vector<DirectedNeighbor> directed(
        static_cast<size_t>(n) * static_cast<size_t>(neighbors));

#if PUNKST_HAVE_OPENBLAS
    std::unique_ptr<ScopedOpenBlasThreads> blas_threads;
    if (kernel == KnnFlatKernel::Cblas) {
        blas_threads = std::make_unique<ScopedOpenBlasThreads>(n_threads);
    }
#endif

    for (int32_t query_begin = 0; query_begin < n;
         query_begin += query_tile) {
        const int32_t query_count = std::min(query_tile, n - query_begin);
        std::vector<CandidateHeap> heaps(static_cast<size_t>(query_count));
        for (int32_t database_begin = 0; database_begin < n;
             database_begin += database_tile) {
            const int32_t database_count = std::min(
                database_tile, n - database_begin);
            RowMajorMatrixXd similarities(query_count, database_count);
            const auto query_clock = Clock::now();
            if (kernel == KnnFlatKernel::Eigen) {
                tbb::parallel_for(tbb::blocked_range<int32_t>(
                    0, query_count, 32),
                    [&](const tbb::blocked_range<int32_t>& range) {
                    similarities.middleRows(range.begin(), range.size()).noalias() =
                        observations.middleRows(
                            query_begin + range.begin(), range.size())
                        * observations.middleRows(
                            database_begin, database_count).transpose();
                });
            } else {
#if PUNKST_HAVE_CBLAS
                cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                    query_count, database_count, dimensions, 1.0,
                    observations.data()
                        + static_cast<size_t>(query_begin) * dimensions,
                    dimensions,
                    observations.data()
                        + static_cast<size_t>(database_begin) * dimensions,
                    dimensions, 0.0, similarities.data(), database_count);
#else
                throw std::logic_error("CBLAS kernel selected without CBLAS");
#endif
            }
            timings.query_seconds += elapsed_seconds(query_clock);

            const auto topk_clock = Clock::now();
            tbb::parallel_for(tbb::blocked_range<int32_t>(
                0, query_count, 16),
                [&](const tbb::blocked_range<int32_t>& range) {
                for (int32_t local = range.begin(); local < range.end(); ++local) {
                    CandidateHeap& heap = heaps[static_cast<size_t>(local)];
                    const int32_t query = query_begin + local;
                    for (int32_t column = 0; column < database_count; ++column) {
                        const int32_t other = database_begin + column;
                        if (other == query) continue;
                        double similarity = similarities(local, column);
                        if (!std::isfinite(similarity)) {
                            throw std::overflow_error(
                                "Inner-product k-NN produced a nonfinite score");
                        }
                        if (clamp_unit_scores) {
                            similarity = std::clamp(similarity, -1.0, 1.0);
                        }
                        retain_candidate(heap, neighbors, {other, similarity});
                    }
                }
            });
            timings.topk_seconds += elapsed_seconds(topk_clock);
        }
        tbb::parallel_for(tbb::blocked_range<int32_t>(0, query_count, 32),
            [&](const tbb::blocked_range<int32_t>& range) {
            for (int32_t local = range.begin(); local < range.end(); ++local) {
                CandidateHeap& heap = heaps[static_cast<size_t>(local)];
                if (static_cast<int32_t>(heap.size()) != neighbors) {
                    throw std::runtime_error(
                        "Inner-product k-NN returned too few neighbors");
                }
                std::vector<Candidate> ordered;
                ordered.reserve(static_cast<size_t>(neighbors));
                while (!heap.empty()) {
                    ordered.push_back(heap.top());
                    heap.pop();
                }
                std::sort(ordered.begin(), ordered.end(), candidate_better);
                const int32_t query = query_begin + local;
                for (int32_t position = 0; position < neighbors; ++position) {
                    const Candidate& candidate =
                        ordered[static_cast<size_t>(position)];
                    directed[static_cast<size_t>(query) * neighbors + position] =
                        {candidate.index, candidate.similarity};
                }
            }
        });
    }
    return directed;
}

KnnGraph union_max_knn_graph(
    const std::vector<DirectedNeighbor>& directed,
    int32_t n, int32_t neighbors) {
    if (n < 2 || neighbors <= 0 || neighbors >= n
        || directed.size() != static_cast<size_t>(n)
            * static_cast<size_t>(neighbors)) {
        throw std::invalid_argument("Invalid directed k-NN graph");
    }
    std::vector<WeightedEdge> graph;
    graph.reserve(directed.size());
    for (int32_t row = 0; row < n; ++row) {
        for (int32_t position = 0; position < neighbors; ++position) {
            const DirectedNeighbor& neighbor = directed[
                static_cast<size_t>(row) * neighbors + position];
            if (neighbor.index < 0 || neighbor.index >= n
                || neighbor.index == row
                || !std::isfinite(neighbor.similarity)) {
                throw std::runtime_error("Invalid directed k-NN neighbor");
            }
            graph.push_back({std::min(row, neighbor.index),
                std::max(row, neighbor.index), neighbor.similarity});
        }
    }
    tbb::parallel_sort(graph.begin(), graph.end(), weighted_edge_less);

    KnnGraph out;
    out.n_nodes = n;
    out.edges.reserve(graph.size());
    out.weights.reserve(graph.size());
    for (size_t position = 0; position < graph.size();) {
        const int32_t first = graph[position].first;
        const int32_t second = graph[position].second;
        double weight = graph[position].weight;
        size_t next = position + 1;
        while (next < graph.size() && graph[next].first == first
               && graph[next].second == second) {
            weight = std::max(weight, graph[next].weight);
            ++next;
        }
        out.edges.emplace_back(first, second);
        out.weights.push_back(weight);
        position = next;
    }
    return out;
}

} // namespace knn_detail

const char* knn_flat_kernel_name(KnnFlatKernel kernel) {
    switch (kernel) {
        case KnnFlatKernel::Auto: return "auto";
        case KnnFlatKernel::Eigen: return "eigen";
        case KnnFlatKernel::Cblas: return "cblas";
    }
    return "unknown";
}

KnnFlatKernel parse_knn_flat_kernel(const std::string& value) {
    if (value == "auto") return KnnFlatKernel::Auto;
    if (value == "eigen") return KnnFlatKernel::Eigen;
    if (value == "cblas") return KnnFlatKernel::Cblas;
    throw std::invalid_argument(
        "k-NN flat kernel must be auto, eigen, or cblas");
}

bool knn_cblas_available() {
#if PUNKST_HAVE_CBLAS
    return true;
#else
    return false;
#endif
}

InnerProductKnnResult inner_product_knn(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const InnerProductKnnOptions& options) {
    validate_inner_product_options(observations, options);
    tbb::global_control parallelism(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(options.n_threads));

    InnerProductKnnResult out;
    out.diagnostics.resolved_flat_kernel = resolve_flat_kernel(
        options.flat_kernel);
    const int32_t n = static_cast<int32_t>(observations.rows());
    const int32_t neighbors = std::min(options.n_neighbors, n - 1);
    if (static_cast<size_t>(n) > std::numeric_limits<size_t>::max()
            / static_cast<size_t>(neighbors)) {
        throw std::invalid_argument(
            "Inner-product k-NN neighbor storage exceeds addressable memory");
    }
    std::vector<knn_detail::DirectedNeighbor> directed =
        knn_detail::flat_inner_product_neighbors(observations, neighbors,
            out.diagnostics.resolved_flat_kernel, options.n_threads, false,
            out.diagnostics.timings);
    const auto reduction_begin = Clock::now();
    out.graph = knn_detail::union_max_knn_graph(directed, n, neighbors);
    out.diagnostics.timings.graph_reduction_seconds =
        elapsed_seconds(reduction_begin);
    return out;
}
