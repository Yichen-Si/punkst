#include "clustering_core/faiss_ann.hpp"

#include <faiss/IndexHNSW.h>
#include <faiss/IndexNNDescent.h>
#include <faiss/impl/HNSW.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point& begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

class ScopedOpenMpThreads {
public:
    explicit ScopedOpenMpThreads(int32_t threads) {
#ifdef _OPENMP
        previous_ = omp_get_max_threads();
        omp_set_num_threads(threads);
#else
        (void)threads;
#endif
    }

    ~ScopedOpenMpThreads() {
#ifdef _OPENMP
        omp_set_num_threads(previous_);
#endif
    }

private:
    int previous_ = 1;
};

struct RankedNeighbor {
    double similarity = -std::numeric_limits<double>::infinity();
    int32_t index = -1;
};

bool ranked_better(const RankedNeighbor& first, const RankedNeighbor& second) {
    return first.similarity > second.similarity
        || (first.similarity == second.similarity
            && first.index < second.index);
}

struct BetterComparator {
    bool operator()(const RankedNeighbor& first,
                    const RankedNeighbor& second) const {
        return ranked_better(first, second);
    }
};

using NeighborHeap = std::priority_queue<RankedNeighbor,
    std::vector<RankedNeighbor>, BetterComparator>;

void retain_neighbor(NeighborHeap& heap, int32_t capacity,
        RankedNeighbor candidate) {
    if (capacity <= 0) return;
    if (static_cast<int32_t>(heap.size()) < capacity) {
        heap.push(candidate);
    } else if (ranked_better(candidate, heap.top())) {
        heap.pop();
        heap.push(candidate);
    }
}

std::vector<int32_t> audit_rows(int32_t rows, int32_t requested) {
    const int32_t count = std::min(rows, requested);
    std::vector<int32_t> selected(static_cast<size_t>(count));
    for (int32_t i = 0; i < count; ++i) {
        selected[static_cast<size_t>(i)] = static_cast<int32_t>(
            (static_cast<int64_t>(i) * rows) / count);
    }
    return selected;
}

std::vector<int32_t> exact_audit_neighbors(
        const Eigen::Ref<const RowMajorMatrixXd>& normalized,
        const std::vector<int32_t>& queries, int32_t neighbors) {
    const int32_t dimensions = static_cast<int32_t>(normalized.cols());
    RowMajorMatrixXd query_matrix(
        static_cast<Eigen::Index>(queries.size()), dimensions);
    for (size_t i = 0; i < queries.size(); ++i) {
        query_matrix.row(static_cast<Eigen::Index>(i)) =
            normalized.row(queries[i]);
    }
    std::vector<NeighborHeap> heaps(queries.size());
    constexpr int32_t block_size = 4096;
    for (int32_t begin = 0; begin < normalized.rows(); begin += block_size) {
        const int32_t count = std::min<int32_t>(
            block_size, static_cast<int32_t>(normalized.rows()) - begin);
        const Eigen::MatrixXd scores = query_matrix
            * normalized.middleRows(begin, count).transpose();
        for (size_t query = 0; query < queries.size(); ++query) {
            for (int32_t offset = 0; offset < count; ++offset) {
                const int32_t index = begin + offset;
                if (index == queries[query]) continue;
                retain_neighbor(heaps[query], neighbors,
                    {scores(static_cast<Eigen::Index>(query), offset), index});
            }
        }
    }
    std::vector<int32_t> result(
        queries.size() * static_cast<size_t>(neighbors), -1);
    for (size_t query = 0; query < queries.size(); ++query) {
        std::vector<RankedNeighbor> sorted;
        sorted.reserve(static_cast<size_t>(neighbors));
        while (!heaps[query].empty()) {
            sorted.push_back(heaps[query].top());
            heaps[query].pop();
        }
        std::sort(sorted.begin(), sorted.end(), ranked_better);
        for (int32_t j = 0; j < neighbors; ++j) {
            result[query * neighbors + static_cast<size_t>(j)] =
                sorted[static_cast<size_t>(j)].index;
        }
    }
    return result;
}

std::vector<float> float_coordinates(
        const Eigen::Ref<const RowMajorMatrixXd>& normalized) {
    std::vector<float> result(static_cast<size_t>(normalized.size()));
    for (Eigen::Index row = 0; row < normalized.rows(); ++row) {
        for (Eigen::Index column = 0; column < normalized.cols(); ++column) {
            result[static_cast<size_t>(row * normalized.cols() + column)] =
                static_cast<float>(normalized(row, column));
        }
    }
    return result;
}

std::vector<int32_t> reranked_query_neighbors(
        const Eigen::Ref<const RowMajorMatrixXd>& normalized,
        const std::vector<int32_t>& queries,
        const std::vector<faiss::idx_t>& labels, int32_t labels_per_query,
        int32_t neighbors) {
    std::vector<int32_t> result(
        queries.size() * static_cast<size_t>(neighbors), -1);
    for (size_t query = 0; query < queries.size(); ++query) {
        std::vector<int32_t> unique;
        unique.reserve(static_cast<size_t>(labels_per_query));
        for (int32_t j = 0; j < labels_per_query; ++j) {
            const faiss::idx_t raw = labels[query * labels_per_query + j];
            if (raw < 0 || raw >= normalized.rows()
                    || raw == queries[query]) continue;
            const int32_t index = static_cast<int32_t>(raw);
            if (std::find(unique.begin(), unique.end(), index) == unique.end()) {
                unique.push_back(index);
            }
        }
        std::vector<RankedNeighbor> ranked;
        ranked.reserve(unique.size());
        for (const int32_t index : unique) {
            ranked.push_back({normalized.row(queries[query]).dot(
                normalized.row(index)), index});
        }
        std::sort(ranked.begin(), ranked.end(), ranked_better);
        if (static_cast<int32_t>(ranked.size()) < neighbors) {
            throw std::runtime_error(
                "Faiss ANN audit returned too few distinct non-self neighbors");
        }
        for (int32_t j = 0; j < neighbors; ++j) {
            result[query * neighbors + static_cast<size_t>(j)] =
                ranked[static_cast<size_t>(j)].index;
        }
    }
    return result;
}

FaissAnnAuditTrial recall_trial(int32_t parameter,
        const std::vector<int32_t>& exact,
        const std::vector<int32_t>& approximate, int32_t neighbors) {
    int64_t matches = 0;
    const int64_t total = static_cast<int64_t>(exact.size());
    for (size_t begin = 0; begin < exact.size(); begin += neighbors) {
        for (int32_t j = 0; j < neighbors; ++j) {
            const int32_t value = approximate[begin + j];
            if (std::find(exact.begin() + static_cast<std::ptrdiff_t>(begin),
                    exact.begin() + static_cast<std::ptrdiff_t>(begin + neighbors),
                    value) != exact.begin()
                        + static_cast<std::ptrdiff_t>(begin + neighbors)) {
                ++matches;
            }
        }
    }
    const double p = total > 0
        ? static_cast<double>(matches) / static_cast<double>(total) : 0.0;
    constexpr double z = 1.6448536269514722;
    const double denominator = 1.0 + z * z / total;
    const double center = p + z * z / (2.0 * total);
    const double radius = z * std::sqrt(
        p * (1.0 - p) / total + z * z / (4.0 * total * total));
    return {parameter, p, std::max(0.0, (center - radius) / denominator)};
}

int32_t resolved_candidate_count(
        int32_t rows, int32_t neighbors, int32_t requested) {
    const int64_t automatic = std::max<int64_t>(64,
        static_cast<int64_t>(4) * neighbors);
    const int64_t value = requested == 0 ? automatic : requested;
    if (value < neighbors) {
        throw std::invalid_argument(
            "Faiss ANN candidate count must be at least the neighbor count");
    }
    return static_cast<int32_t>(std::min<int64_t>(rows - 1, value));
}

void validate_common(int32_t rows, int32_t neighbors, int32_t audit_queries,
        double recall, int32_t threads) {
    if (rows < 2 || neighbors <= 0 || neighbors >= rows
            || audit_queries <= 0 || !(recall > 0.0 && recall <= 1.0)
            || !std::isfinite(recall) || threads <= 0) {
        throw std::invalid_argument("Invalid Faiss ANN configuration");
    }
}

void set_final_audit(FaissAnnCandidateResult& result,
        const FaissAnnAuditTrial& trial, double target) {
    result.audit_mean_recall = trial.mean_recall;
    result.audit_recall_lcb = trial.recall_lcb;
    result.audit_passed = trial.recall_lcb >= target;
}

} // namespace

int32_t automatic_nndescent_iterations(int64_t sample_size) {
    if (sample_size <= 0) {
        throw std::invalid_argument("NN-descent sample size must be positive");
    }
    return std::max<int32_t>(10, static_cast<int32_t>(
        std::llround(std::log2(static_cast<double>(sample_size)))));
}

int32_t automatic_hnsw_ef_search(int64_t sample_size) {
    if (sample_size <= 0) {
        throw std::invalid_argument("HNSW sample size must be positive");
    }
    return static_cast<int32_t>(
        std::log2(static_cast<double>(sample_size)) * 6.0);
}

FaissAnnCandidateResult faiss_hnsw_candidates(
        const Eigen::Ref<const RowMajorMatrixXd>& normalized,
        int32_t neighbors, const FaissHnswOptions& options) {
    const int32_t rows = static_cast<int32_t>(normalized.rows());
    validate_common(rows, neighbors, options.audit_queries,
        options.recall, options.n_threads);
    if (options.m <= 0 || options.ef_construction <= 0
            || options.ef_search < 0 || options.max_ef_search <= 0
            || options.candidates < 0) {
        throw std::invalid_argument("Invalid Faiss HNSW configuration");
    }
    const int32_t candidate_count = resolved_candidate_count(
        rows, neighbors, options.candidates);
    if (options.max_ef_search < candidate_count) {
        throw std::invalid_argument(
            "HNSW maximum efSearch must be at least the candidate count");
    }
    if (options.ef_search > 0 && options.ef_search < candidate_count) {
        throw std::invalid_argument(
            "Explicit HNSW efSearch must be at least the candidate count");
    }

    ScopedOpenMpThreads omp_threads(options.n_threads);
    const Clock::time_point build_begin = Clock::now();
    const std::vector<float> coordinates = float_coordinates(normalized);
    faiss::hnsw_deterministic_build = true;
    faiss::IndexHNSWFlat index(
        normalized.cols(), options.m, faiss::METRIC_INNER_PRODUCT);
    index.hnsw.efConstruction = options.ef_construction;
    FaissAnnCandidateResult result;
    result.requested_parameter = options.ef_search;
    result.resolved_candidate_count = candidate_count;
    index.add(rows, coordinates.data());
    result.index_build_seconds = elapsed_seconds(build_begin);

    const Clock::time_point audit_begin = Clock::now();
    const std::vector<int32_t> queries = audit_rows(rows, options.audit_queries);
    const std::vector<int32_t> exact = exact_audit_neighbors(
        normalized, queries, neighbors);
    std::vector<float> audit_coordinates(
        queries.size() * static_cast<size_t>(normalized.cols()));
    for (size_t query = 0; query < queries.size(); ++query) {
        std::copy_n(coordinates.data()
                + static_cast<size_t>(queries[query]) * normalized.cols(),
            normalized.cols(), audit_coordinates.data()
                + query * static_cast<size_t>(normalized.cols()));
    }
    const int32_t searched = std::min(rows, candidate_count);
    std::map<int32_t, FaissAnnAuditTrial> trials;
    auto audit = [&](int32_t ef_search) -> const FaissAnnAuditTrial& {
        const auto existing = trials.find(ef_search);
        if (existing != trials.end()) return existing->second;
        index.hnsw.efSearch = ef_search;
        std::vector<float> distances(queries.size() * searched);
        std::vector<faiss::idx_t> labels(queries.size() * searched);
        index.search(static_cast<faiss::idx_t>(queries.size()),
            audit_coordinates.data(), searched, distances.data(), labels.data());
        const std::vector<int32_t> approximate = reranked_query_neighbors(
            normalized, queries, labels, searched, neighbors);
        return trials.emplace(ef_search,
            recall_trial(ef_search, exact, approximate, neighbors)).first->second;
    };

    int32_t resolved = options.ef_search;
    if (resolved > 0) {
        const FaissAnnAuditTrial& trial = audit(resolved);
        set_final_audit(result, trial, options.recall);
    } else {
        int32_t current = std::clamp(automatic_hnsw_ef_search(rows),
            candidate_count, options.max_ef_search);
        const FaissAnnAuditTrial* current_trial = &audit(current);
        int32_t failed = -1;
        int32_t passed = -1;
        if (current_trial->recall_lcb >= options.recall) {
            passed = current;
            while (current > candidate_count) {
                const int32_t next = std::max(candidate_count, current / 2);
                const FaissAnnAuditTrial& trial = audit(next);
                if (trial.recall_lcb >= options.recall) {
                    passed = next;
                    current = next;
                    if (next == candidate_count) break;
                } else {
                    failed = next;
                    break;
                }
            }
        } else {
            failed = current;
            while (current < options.max_ef_search) {
                const int32_t next = std::min(
                    options.max_ef_search, current * 2);
                const FaissAnnAuditTrial& trial = audit(next);
                current = next;
                if (trial.recall_lcb >= options.recall) {
                    passed = next;
                    break;
                }
                failed = next;
            }
        }
        if (passed >= 0 && failed >= 0) {
            int32_t low = failed;
            int32_t high = passed;
            if (low > high) std::swap(low, high);
            while (high - low > 8) {
                const int32_t middle = low + (high - low) / 2;
                const FaissAnnAuditTrial& trial = audit(middle);
                if (trial.recall_lcb >= options.recall) high = middle;
                else low = middle;
            }
            passed = high;
        }
        resolved = passed >= 0 ? passed : options.max_ef_search;
        const FaissAnnAuditTrial& final_trial = audit(resolved);
        set_final_audit(result, final_trial, options.recall);
    }
    result.resolved_parameter = resolved;
    result.forced = !result.audit_passed && options.force;
    for (const auto& item : trials) result.audit_trials.push_back(item.second);
    std::sort(result.audit_trials.begin(), result.audit_trials.end(),
        [](const auto& first, const auto& second) {
            return first.parameter < second.parameter;
        });
    result.audit_seconds = elapsed_seconds(audit_begin);
    if (!result.audit_passed && !options.force) {
        throw std::runtime_error(
            "HNSW sampled recall lower bound did not reach the requested target; increase --hnsw-max-ef-search or use --hnsw-force");
    }

    index.hnsw.efSearch = resolved;
    result.candidates_per_row = searched;
    const size_t full_candidate_size = static_cast<size_t>(rows) * searched;
    std::vector<float> distances(full_candidate_size);
    std::vector<faiss::idx_t> labels(full_candidate_size);
    const Clock::time_point query_begin = Clock::now();
    index.search(rows, coordinates.data(), searched,
        distances.data(), labels.data());
    result.query_seconds = elapsed_seconds(query_begin);
    std::vector<float>().swap(distances);
    result.candidates.resize(full_candidate_size);
    for (size_t i = 0; i < labels.size(); ++i) {
        result.candidates[i] = labels[i] >= 0
            && labels[i] <= std::numeric_limits<int32_t>::max()
            ? static_cast<int32_t>(labels[i]) : -1;
    }
    return result;
}

FaissAnnCandidateResult faiss_nndescent_candidates(
        const Eigen::Ref<const RowMajorMatrixXd>& normalized,
        int32_t neighbors, const FaissNnDescentOptions& options) {
    const int32_t rows = static_cast<int32_t>(normalized.rows());
    validate_common(rows, neighbors, options.audit_queries,
        options.recall, options.n_threads);
    if (options.iterations < 0 || options.graph_size < 0
            || options.sample_candidates <= 0) {
        throw std::invalid_argument("Invalid Faiss NN-descent configuration");
    }
    if (rows <= 100) {
        throw std::invalid_argument(
            "Faiss NN-descent requires more than 100 input units");
    }
    const int32_t graph_size = resolved_candidate_count(
        rows, neighbors, options.graph_size);
    const int32_t iterations = options.iterations > 0
        ? options.iterations : automatic_nndescent_iterations(rows);
    ScopedOpenMpThreads omp_threads(options.n_threads);
    const Clock::time_point build_begin = Clock::now();
    const std::vector<float> coordinates = float_coordinates(normalized);
    faiss::IndexNNDescentFlat index(
        normalized.cols(), graph_size, faiss::METRIC_INNER_PRODUCT);
    index.nndescent.random_seed = options.seed;
    index.nndescent.iter = iterations;
    index.nndescent.S = options.sample_candidates;
    index.nndescent.L = graph_size;
    FaissAnnCandidateResult result;
    result.requested_parameter = options.iterations;
    result.resolved_parameter = iterations;
    result.resolved_candidate_count = graph_size;
    index.add(rows, coordinates.data());
    result.index_build_seconds = elapsed_seconds(build_begin);
    const std::vector<faiss::IndexNNDescent::storage_idx_t>& graph =
        index.nndescent.final_graph;
    if (graph.size() != static_cast<size_t>(rows) * graph_size) {
        throw std::runtime_error("Faiss NN-descent returned an invalid graph");
    }
    result.candidates_per_row = graph_size;
    result.candidates.resize(graph.size());
    for (size_t i = 0; i < graph.size(); ++i) {
        result.candidates[i] = graph[i];
    }

    const Clock::time_point audit_begin = Clock::now();
    const std::vector<int32_t> queries = audit_rows(rows, options.audit_queries);
    const std::vector<int32_t> exact = exact_audit_neighbors(
        normalized, queries, neighbors);
    std::vector<faiss::idx_t> labels(
        queries.size() * static_cast<size_t>(graph_size));
    for (size_t query = 0; query < queries.size(); ++query) {
        const size_t source = static_cast<size_t>(queries[query]) * graph_size;
        for (int32_t j = 0; j < graph_size; ++j) {
            labels[query * graph_size + static_cast<size_t>(j)] =
                graph[source + static_cast<size_t>(j)];
        }
    }
    const std::vector<int32_t> approximate = reranked_query_neighbors(
        normalized, queries, labels, graph_size, neighbors);
    const FaissAnnAuditTrial trial = recall_trial(
        iterations, exact, approximate, neighbors);
    result.audit_trials.push_back(trial);
    set_final_audit(result, trial, options.recall);
    result.audit_seconds = elapsed_seconds(audit_begin);
    if (!result.audit_passed) {
        throw std::runtime_error(
            "NN-descent sampled recall lower bound did not reach the requested target; increase --nndescent-iterations");
    }
    return result;
}
