#include "clustering_core/cosine_clustering.hpp"

#include "clustering_core/knn_internal.hpp"
#if PUNKST_HAVE_FAISS_ANN
#include "clustering_core/faiss_ann.hpp"
#endif
#include "nanoflann.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace {

using CosineIndex = nanoflann::KDTreeEigenMatrixAdaptor<
    RowMajorMatrixXd, -1, nanoflann::metric_L2, true>;

struct Neighbor {
    Eigen::Index index = -1;
    double distance = std::numeric_limits<double>::infinity();
};

bool neighbor_less(const Neighbor& first, const Neighbor& second) {
    return first.distance < second.distance
        || (first.distance == second.distance && first.index < second.index);
}

// nanoflann's default result set does not replace an equal-distance boundary
// point. Returning the next representable distance from worstDist makes the
// tree visit exact ties, while addPoint keeps the lowest document indices.
class LexicographicKnnResultSet {
public:
    LexicographicKnnResultSet(
        size_t capacity, Eigen::Index* indices, double* distances)
        : capacity_(capacity), indices_(indices), distances_(distances) {}

    size_t size() const { return size_; }
    bool empty() const { return size_ == 0; }
    bool full() const { return size_ == capacity_; }
    void sort() {}

    bool addPoint(double distance, Eigen::Index index) {
        size_t position = size_;
        while (position > 0) {
            const Neighbor previous{
                indices_[position - 1], distances_[position - 1]};
            if (!neighbor_less(Neighbor{index, distance}, previous)) break;
            if (position < capacity_) {
                indices_[position] = previous.index;
                distances_[position] = previous.distance;
            }
            --position;
        }
        if (position < capacity_) {
            indices_[position] = index;
            distances_[position] = distance;
        }
        if (size_ < capacity_) ++size_;
        return true;
    }

    double worstDist() const {
        if (!full()) return std::numeric_limits<double>::infinity();
        return std::nextafter(
            distances_[capacity_ - 1], std::numeric_limits<double>::infinity());
    }

private:
    size_t capacity_ = 0;
    size_t size_ = 0;
    Eigen::Index* indices_ = nullptr;
    double* distances_ = nullptr;
};

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point& begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

using knn_detail::DirectedNeighbor;

CosineKnnBackend resolve_backend(
    const CosineKnnOptions& options, Eigen::Index dimensions) {
    if (options.backend != CosineKnnBackend::Auto) return options.backend;
    if (options.knn_search_epsilon > 0.0 || dimensions <= 16) {
        return CosineKnnBackend::KdTree;
    }
    return CosineKnnBackend::Flat;
}

std::vector<DirectedNeighbor> ann_candidate_neighbors(
        const RowMajorMatrixXd& normalized,
        const std::vector<int32_t>& candidates, int32_t candidates_per_row,
        int32_t neighbors, CosineKnnTimings& timings) {
    const int32_t rows = static_cast<int32_t>(normalized.rows());
    if (candidates_per_row < neighbors
            || candidates.size() != static_cast<size_t>(rows)
                * static_cast<size_t>(candidates_per_row)) {
        throw std::runtime_error("Invalid Faiss ANN candidate matrix");
    }
    const auto topk_begin = Clock::now();
    std::vector<DirectedNeighbor> directed(
        static_cast<size_t>(rows) * static_cast<size_t>(neighbors));
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, rows, 64),
        [&](const tbb::blocked_range<int32_t>& range) {
            std::vector<int32_t> unique;
            std::vector<Neighbor> ranked;
            unique.reserve(static_cast<size_t>(candidates_per_row));
            ranked.reserve(static_cast<size_t>(candidates_per_row));
            for (int32_t row = range.begin(); row < range.end(); ++row) {
                unique.clear();
                ranked.clear();
                const size_t begin = static_cast<size_t>(row)
                    * static_cast<size_t>(candidates_per_row);
                for (int32_t j = 0; j < candidates_per_row; ++j) {
                    const int32_t index = candidates[begin + j];
                    if (index < 0 || index >= rows || index == row
                            || std::find(unique.begin(), unique.end(), index)
                                != unique.end()) continue;
                    unique.push_back(index);
                    ranked.push_back({index,
                        -normalized.row(row).dot(normalized.row(index))});
                }
                std::sort(ranked.begin(), ranked.end(), neighbor_less);
                if (static_cast<int32_t>(ranked.size()) < neighbors) {
                    throw std::runtime_error(
                        "Faiss ANN returned too few distinct non-self candidates");
                }
                for (int32_t j = 0; j < neighbors; ++j) {
                    const Neighbor& item = ranked[static_cast<size_t>(j)];
                    directed[static_cast<size_t>(row) * neighbors + j] =
                        {static_cast<int32_t>(item.index), -item.distance};
                }
            }
        });
    timings.topk_seconds = elapsed_seconds(topk_begin);
    return directed;
}

CosineFlatKernel resolve_flat_kernel(CosineFlatKernel requested) {
    if (requested == CosineFlatKernel::Auto) {
        return CosineFlatKernel::Eigen;
    }
    if (requested == CosineFlatKernel::Cblas) {
        if (!knn_cblas_available()) {
            throw std::invalid_argument(
                "The CBLAS cosine k-NN kernel is not available in this build");
        }
    }
    return requested;
}

void validate_knn_options(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const CosineKnnOptions& options) {
    if (observations.rows() < 2 || options.n_neighbors <= 0
        || options.n_threads <= 0
        || !std::isfinite(options.knn_search_epsilon)
        || options.knn_search_epsilon < 0.0
        || options.knn_search_epsilon > std::numeric_limits<float>::max()
        || static_cast<int32_t>(options.backend)
            < static_cast<int32_t>(CosineKnnBackend::Auto)
        || static_cast<int32_t>(options.backend)
            > static_cast<int32_t>(CosineKnnBackend::NnDescent)
        || static_cast<int32_t>(options.flat_kernel)
            < static_cast<int32_t>(CosineFlatKernel::Auto)
        || static_cast<int32_t>(options.flat_kernel)
            > static_cast<int32_t>(CosineFlatKernel::Cblas)
        || options.hnsw_m <= 0 || options.hnsw_ef_construction <= 0
        || options.hnsw_ef_search < 0 || options.hnsw_max_ef_search <= 0
        || options.hnsw_candidates < 0 || options.hnsw_audit_queries <= 0
        || !(options.hnsw_recall > 0.0 && options.hnsw_recall <= 1.0)
        || !std::isfinite(options.hnsw_recall)
        || options.nndescent_iterations < 0
        || options.nndescent_graph_size < 0
        || options.nndescent_sample_candidates <= 0
        || options.nndescent_audit_queries <= 0
        || !(options.nndescent_recall > 0.0
            && options.nndescent_recall <= 1.0)
        || !std::isfinite(options.nndescent_recall)) {
        throw std::invalid_argument(
            "Simplex k-NN requires at least two rows, positive neighbors and "
            "threads, and a finite non-negative epsilon");
    }
    const CosineKnnBackend resolved = resolve_backend(
        options, observations.cols());
    if (options.knn_search_epsilon > 0.0
        && resolved != CosineKnnBackend::KdTree) {
        throw std::invalid_argument(
            "Simplex k-NN epsilon is supported only by the kd-tree backend");
    }
#if !PUNKST_HAVE_FAISS_ANN
    if (resolved == CosineKnnBackend::Hnsw
            || resolved == CosineKnnBackend::NnDescent) {
        throw std::invalid_argument(
            "The requested Faiss ANN backend is unavailable in this build; rebuild with PUNKST_ENABLE_FAISS_ANN=ON");
    }
#endif
    if (observations.rows() > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("Simplex k-NN row count exceeds int32 range");
    }
}

std::vector<DirectedNeighbor> kd_tree_neighbors(
    const RowMajorMatrixXd& normalized, int32_t neighbors, double epsilon,
    CosineKnnTimings& timings) {
    const int32_t n = static_cast<int32_t>(normalized.rows());
    const size_t queried = static_cast<size_t>(neighbors) + 1;
    const auto build_begin = Clock::now();
    CosineIndex index(static_cast<int32_t>(normalized.cols()),
        std::cref(normalized), 10);
    timings.index_build_seconds = elapsed_seconds(build_begin);

    std::vector<DirectedNeighbor> directed(
        static_cast<size_t>(n) * static_cast<size_t>(neighbors));
    const float search_epsilon = static_cast<float>(epsilon);
    const auto query_begin = Clock::now();
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, n, 64),
        [&](const tbb::blocked_range<int32_t>& range) {
            std::vector<Eigen::Index> indices(queried);
            std::vector<double> distances(queried);
            for (int32_t row = range.begin(); row < range.end(); ++row) {
                LexicographicKnnResultSet result(
                    queried, indices.data(), distances.data());
                index.index_->findNeighbors(result, normalized.row(row).data(),
                    nanoflann::SearchParameters(search_epsilon));
                int32_t retained = 0;
                for (size_t j = 0; j < result.size()
                     && retained < neighbors; ++j) {
                    const int32_t other = static_cast<int32_t>(indices[j]);
                    if (other == row) continue;
                    directed[static_cast<size_t>(row) * neighbors + retained] =
                        {other, std::clamp(1.0 - 0.5 * distances[j], -1.0, 1.0)};
                    ++retained;
                }
                if (retained != neighbors) {
                    throw std::runtime_error(
                        "Simplex k-NN query returned too few neighbors");
                }
            }
        });
    timings.query_seconds = elapsed_seconds(query_begin);
    return directed;
}

CosineKnnGraph filter_positive_similarity_graph(KnnGraph graph) {
    size_t write = 0;
    for (size_t read = 0; read < graph.weights.size(); ++read) {
        if (graph.weights[read] <= 0.0) continue;
        graph.edges[write] = graph.edges[read];
        graph.weights[write] = graph.weights[read];
        ++write;
    }
    graph.edges.resize(write);
    graph.weights.resize(write);
    if (graph.edges.empty()) {
        throw std::invalid_argument(
            "Simplex k-NN graph has no positive-similarity edges");
    }
    return graph;
}

struct CommunityStats {
    Eigen::RowVectorXd sum;
    double sum_squared_norm = 0.0;
    int32_t count = 0;
    int32_t parent = -1;
    uint64_t version = 0;
    bool active = true;
    std::vector<int32_t> members;
};

double community_sse(const CommunityStats& community) {
    if (community.count <= 0) return 0.0;
    return std::max(0.0, community.sum_squared_norm
        - community.sum.squaredNorm() / static_cast<double>(community.count));
}

double ward_cost(
    const CommunityStats& first, const CommunityStats& second) {
    const double first_count = static_cast<double>(first.count);
    const double second_count = static_cast<double>(second.count);
    return first_count * second_count / (first_count + second_count)
        * (first.sum / first_count - second.sum / second_count).squaredNorm();
}

int32_t find_root(std::vector<CommunityStats>& communities, int32_t group) {
    int32_t root = group;
    while (communities[root].parent != root) {
        root = communities[root].parent;
    }
    while (communities[group].parent != group) {
        const int32_t next = communities[group].parent;
        communities[group].parent = root;
        group = next;
    }
    return root;
}

struct WardCandidate {
    double cost = 0.0;
    int32_t owner = -1;
    int32_t neighbor = -1;
    int32_t first = -1;
    int32_t second = -1;
    uint64_t owner_version = 0;
    uint64_t neighbor_version = 0;
};

struct WardCandidateGreater {
    bool operator()(const WardCandidate& first, const WardCandidate& second) const {
        if (first.cost != second.cost) return first.cost > second.cost;
        if (first.first != second.first) return first.first > second.first;
        if (first.second != second.second) return first.second > second.second;
        return first.owner > second.owner;
    }
};

WardCandidate nearest_ward_candidate(
    const std::vector<CommunityStats>& communities, int32_t owner) {
    WardCandidate best;
    best.owner = owner;
    best.owner_version = communities[owner].version;
    best.cost = std::numeric_limits<double>::infinity();
    for (int32_t other = 0;
         other < static_cast<int32_t>(communities.size()); ++other) {
        if (other == owner || !communities[other].active) continue;
        const double cost = ward_cost(communities[owner], communities[other]);
        const int32_t first = std::min(owner, other);
        const int32_t second = std::max(owner, other);
        if (cost < best.cost
            || (cost == best.cost
                && std::pair<int32_t, int32_t>{first, second}
                    < std::pair<int32_t, int32_t>{best.first, best.second})) {
            best.cost = cost;
            best.neighbor = other;
            best.first = first;
            best.second = second;
            best.neighbor_version = communities[other].version;
        }
    }
    return best;
}

CommunityStats summarize_members(
    const std::vector<int32_t>& members,
    const Eigen::Ref<const RowMajorMatrixXd>& observations) {
    CommunityStats out;
    out.sum = Eigen::RowVectorXd::Zero(observations.cols());
    out.count = static_cast<int32_t>(members.size());
    out.parent = -1;
    out.members = members;
    for (int32_t document : members) {
        out.sum += observations.row(document);
        out.sum_squared_norm += observations.row(document).squaredNorm();
    }
    return out;
}

} // namespace

RowMajorMatrixXd l2_normalize_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& observations) {
    if (observations.rows() == 0 || observations.cols() == 0) {
        throw std::invalid_argument(
            "Cosine clustering requires a non-empty finite matrix");
    }
    RowMajorMatrixXd out = observations;
    for (Eigen::Index row = 0; row < out.rows(); ++row) {
        if (!out.row(row).allFinite()) {
            throw std::invalid_argument(
                "Cosine clustering requires a non-empty finite matrix");
        }
        const double scale = out.row(row).cwiseAbs().maxCoeff();
        if (scale <= 0.0) {
            throw std::invalid_argument(
                "Cosine clustering requires nonzero observation rows");
        }
        out.row(row) /= scale;
        const double norm = out.row(row).norm();
        if (!std::isfinite(norm) || norm <= 0.0) {
            throw std::invalid_argument(
                "Cosine clustering requires nonzero observation rows");
        }
        out.row(row) /= norm;
    }
    return out;
}

const char* simplex_metric_name(SimplexMetric metric) {
    switch (metric) {
        case SimplexMetric::Cosine: return "cosine";
        case SimplexMetric::Hellinger: return "hellinger";
    }
    return "unknown";
}

SimplexMetric parse_simplex_metric(const std::string& value) {
    if (value == "cosine") return SimplexMetric::Cosine;
    if (value == "hellinger") return SimplexMetric::Hellinger;
    throw std::invalid_argument(
        "Simplex clustering metric must be cosine or hellinger");
}

RowMajorMatrixXd simplex_metric_coordinates(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric) {
    if (metric == SimplexMetric::Cosine) {
        return l2_normalize_rows(observations);
    }
    if (metric != SimplexMetric::Hellinger
        || observations.rows() == 0 || observations.cols() == 0) {
        throw std::invalid_argument(
            "Hellinger clustering requires a non-empty nonnegative matrix");
    }
    RowMajorMatrixXd out = observations;
    for (Eigen::Index row = 0; row < out.rows(); ++row) {
        if (!out.row(row).allFinite()
            || (out.row(row).array() < 0.0).any()) {
            throw std::invalid_argument(
                "Hellinger clustering requires finite nonnegative rows");
        }
        const double scale = out.row(row).maxCoeff();
        if (!(scale > 0.0)) {
            throw std::invalid_argument(
                "Hellinger clustering requires positive row sums");
        }
        out.row(row) /= scale;
        const double total = out.row(row).sum();
        if (!std::isfinite(total) || !(total > 0.0)) {
            throw std::invalid_argument(
                "Hellinger clustering requires positive row sums");
        }
        out.row(row) = (out.row(row).array() / total).sqrt();
    }
    return out;
}

DenseKMeansResult cosine_dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& options) {
    return simplex_dense_kmeans(
        observations, SimplexMetric::Cosine, options);
}

DenseKMeansResult simplex_dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric, const DenseKMeansOptions& options) {
    const RowMajorMatrixXd coordinates = simplex_metric_coordinates(
        observations, metric);
    return dense_kmeans(coordinates, options);
}

const char* cosine_knn_backend_name(CosineKnnBackend backend) {
    switch (backend) {
        case CosineKnnBackend::Auto: return "auto";
        case CosineKnnBackend::KdTree: return "kdtree";
        case CosineKnnBackend::Flat: return "flat";
        case CosineKnnBackend::Hnsw: return "hnsw";
        case CosineKnnBackend::NnDescent: return "nndescent";
    }
    return "unknown";
}

const char* cosine_flat_kernel_name(CosineFlatKernel kernel) {
    return knn_flat_kernel_name(kernel);
}

CosineKnnBackend parse_cosine_knn_backend(const std::string& value) {
    if (value == "auto") return CosineKnnBackend::Auto;
    if (value == "kdtree") return CosineKnnBackend::KdTree;
    if (value == "flat") return CosineKnnBackend::Flat;
    if (value == "hnsw") return CosineKnnBackend::Hnsw;
    if (value == "nndescent") return CosineKnnBackend::NnDescent;
    throw std::invalid_argument(
        "Cosine k-NN backend must be auto, kdtree, flat, hnsw, or nndescent");
}

CosineFlatKernel parse_cosine_flat_kernel(const std::string& value) {
    return parse_knn_flat_kernel(value);
}

bool cosine_knn_cblas_available() {
    return knn_cblas_available();
}

bool cosine_knn_faiss_available() {
#if PUNKST_HAVE_FAISS_ANN
    return true;
#else
    return false;
#endif
}

CosineKnnResult simplex_knn(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric, const CosineKnnOptions& options) {
    validate_knn_options(observations, options);
    tbb::global_control parallelism(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(options.n_threads));
    CosineKnnResult out;
    out.diagnostics.requested_backend = options.backend;
    out.diagnostics.sample_size = observations.rows();
    out.diagnostics.resolved_backend = resolve_backend(
        options, observations.cols());
    if (out.diagnostics.resolved_backend == CosineKnnBackend::Flat) {
        out.diagnostics.resolved_flat_kernel = resolve_flat_kernel(
            options.flat_kernel);
    }

    const auto normalization_begin = Clock::now();
    const RowMajorMatrixXd normalized = simplex_metric_coordinates(
        observations, metric);
    out.diagnostics.timings.normalization_seconds =
        elapsed_seconds(normalization_begin);
    const int32_t n = static_cast<int32_t>(normalized.rows());
    const int32_t neighbors = std::min(options.n_neighbors, n - 1);
    if (static_cast<size_t>(n) > std::numeric_limits<size_t>::max()
            / static_cast<size_t>(neighbors)) {
        throw std::invalid_argument(
            "Simplex k-NN neighbor storage exceeds addressable memory");
    }
    std::vector<DirectedNeighbor> directed;
    switch (out.diagnostics.resolved_backend) {
        case CosineKnnBackend::Auto:
            throw std::logic_error("Unresolved automatic cosine k-NN backend");
        case CosineKnnBackend::KdTree:
            directed = kd_tree_neighbors(normalized, neighbors,
                options.knn_search_epsilon, out.diagnostics.timings);
            break;
        case CosineKnnBackend::Flat:
            {
                InnerProductKnnTimings flat_timings;
                directed = knn_detail::flat_inner_product_neighbors(
                    normalized, neighbors,
                    out.diagnostics.resolved_flat_kernel, options.n_threads,
                    true, flat_timings);
                out.diagnostics.timings.query_seconds =
                    flat_timings.query_seconds;
                out.diagnostics.timings.topk_seconds =
                    flat_timings.topk_seconds;
            }
            break;
        case CosineKnnBackend::Hnsw:
        case CosineKnnBackend::NnDescent:
            {
#if PUNKST_HAVE_FAISS_ANN
                FaissAnnCandidateResult ann;
                if (out.diagnostics.resolved_backend
                        == CosineKnnBackend::Hnsw) {
                    FaissHnswOptions faiss_options;
                    faiss_options.m = options.hnsw_m;
                    faiss_options.ef_construction =
                        options.hnsw_ef_construction;
                    faiss_options.ef_search = options.hnsw_ef_search;
                    faiss_options.max_ef_search =
                        options.hnsw_max_ef_search;
                    faiss_options.candidates = options.hnsw_candidates;
                    faiss_options.audit_queries = options.hnsw_audit_queries;
                    faiss_options.recall = options.hnsw_recall;
                    faiss_options.force = options.hnsw_force;
                    faiss_options.n_threads = options.n_threads;
                    ann = faiss_hnsw_candidates(
                        normalized, neighbors, faiss_options);
                } else {
                    FaissNnDescentOptions faiss_options;
                    faiss_options.iterations =
                        options.nndescent_iterations;
                    faiss_options.graph_size =
                        options.nndescent_graph_size;
                    faiss_options.sample_candidates =
                        options.nndescent_sample_candidates;
                    faiss_options.audit_queries =
                        options.nndescent_audit_queries;
                    faiss_options.recall = options.nndescent_recall;
                    faiss_options.seed = options.ann_seed;
                    faiss_options.n_threads = options.n_threads;
                    ann = faiss_nndescent_candidates(
                        normalized, neighbors, faiss_options);
                }
                out.diagnostics.requested_ann_parameter =
                    ann.requested_parameter;
                out.diagnostics.resolved_ann_parameter =
                    ann.resolved_parameter;
                out.diagnostics.resolved_ann_candidates =
                    ann.resolved_candidate_count;
                out.diagnostics.audit_mean_recall = ann.audit_mean_recall;
                out.diagnostics.audit_recall_lcb = ann.audit_recall_lcb;
                out.diagnostics.audit_passed = ann.audit_passed;
                out.diagnostics.forced = ann.forced;
                for (const FaissAnnAuditTrial& trial : ann.audit_trials) {
                    out.diagnostics.audit_trials.push_back({trial.parameter,
                        trial.mean_recall, trial.recall_lcb});
                }
                out.diagnostics.timings.index_build_seconds =
                    ann.index_build_seconds;
                out.diagnostics.timings.query_seconds = ann.query_seconds;
                out.diagnostics.timings.audit_seconds = ann.audit_seconds;
                directed = ann_candidate_neighbors(normalized,
                    ann.candidates, ann.candidates_per_row, neighbors,
                    out.diagnostics.timings);
#else
                throw std::logic_error("Faiss ANN backend was not compiled");
#endif
            }
            break;
    }
    const auto reduction_begin = Clock::now();
    out.graph = filter_positive_similarity_graph(
        knn_detail::union_max_knn_graph(directed, n, neighbors));
    out.diagnostics.timings.graph_reduction_seconds =
        elapsed_seconds(reduction_begin);
    return out;
}

CosineKnnResult cosine_knn(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const CosineKnnOptions& options) {
    return simplex_knn(observations, SimplexMetric::Cosine, options);
}

CosineKnnGraph cosine_knn_graph(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    int32_t n_neighbors, double search_epsilon) {
    CosineKnnOptions options;
    options.n_neighbors = n_neighbors;
    options.knn_search_epsilon = search_epsilon;
    return cosine_knn(observations, options).graph;
}

CosineLeidenResult cosine_leiden_cluster(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const CosineLeidenOptions& options) {
    CosineKnnResult knn = cosine_knn(observations, options);

    CosineLeidenResult out;
    out.n_edges = static_cast<int64_t>(knn.graph.edges.size());
    out.knn = knn.diagnostics;
    out.clustering = leiden_cluster(
        knn.graph.n_nodes, knn.graph.edges, knn.graph.weights, options.leiden);
    return out;
}

CosineLeidenResult simplex_leiden_cluster(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric, const CosineLeidenOptions& options) {
    CosineKnnResult knn = simplex_knn(observations, metric, options);

    CosineLeidenResult out;
    out.n_edges = static_cast<int64_t>(knn.graph.edges.size());
    out.knn = knn.diagnostics;
    out.clustering = leiden_cluster(
        knn.graph.n_nodes, knn.graph.edges, knn.graph.weights, options.leiden);
    return out;
}

Eigen::VectorXi reconcile_simplex_communities(
    const Eigen::VectorXi& membership, int32_t n_communities,
    int32_t requested_communities,
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    SimplexMetric metric, const DenseKMeansOptions& kmeans_options) {
    if (observations.rows() > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument("Community reconciliation exceeds int32 range");
    }
    const int32_t n = static_cast<int32_t>(observations.rows());
    if (membership.size() != n || n_communities <= 0
        || requested_communities <= 0 || requested_communities > n) {
        throw std::invalid_argument(
            "Invalid community-count reconciliation input");
    }
    Eigen::VectorXi labels = membership;
    std::vector<CommunityStats> communities(n_communities);
    for (int32_t community = 0; community < n_communities; ++community) {
        communities[community].parent = community;
    }
    for (int32_t document = 0; document < n; ++document) {
        const int32_t community = labels(document);
        if (community < 0 || community >= n_communities) {
            throw std::invalid_argument("Community label is out of range");
        }
        communities[community].members.push_back(document);
    }
    for (const CommunityStats& community : communities) {
        if (community.members.empty()) {
            throw std::invalid_argument("Community labels must be contiguous");
        }
    }
    if (n_communities == requested_communities) {
        (void)simplex_metric_coordinates(observations, metric);
        return labels;
    }

    const RowMajorMatrixXd normalized = simplex_metric_coordinates(
        observations, metric);
    for (int32_t community = 0; community < n_communities; ++community) {
        CommunityStats summary = summarize_members(
            communities[community].members, normalized);
        summary.parent = community;
        communities[community] = std::move(summary);
    }

    int32_t groups = n_communities;
    if (groups > requested_communities) {
        std::priority_queue<WardCandidate, std::vector<WardCandidate>,
            WardCandidateGreater> candidates;
        for (int32_t community = 0; community < groups; ++community) {
            candidates.push(nearest_ward_candidate(communities, community));
        }
        while (groups > requested_communities) {
            if (candidates.empty()) {
                throw std::runtime_error("Cannot merge communities further");
            }
            const WardCandidate candidate = candidates.top();
            candidates.pop();
            if (!communities[candidate.owner].active
                || communities[candidate.owner].version
                    != candidate.owner_version
                || candidate.neighbor < 0
                || !communities[candidate.neighbor].active
                || communities[candidate.neighbor].version
                    != candidate.neighbor_version) {
                if (communities[candidate.owner].active) {
                    candidates.push(nearest_ward_candidate(
                        communities, candidate.owner));
                }
                continue;
            }
            const WardCandidate current = nearest_ward_candidate(
                communities, candidate.owner);
            if (current.neighbor != candidate.neighbor
                || current.cost != candidate.cost) {
                candidates.push(current);
                continue;
            }

            const int32_t keep = candidate.first;
            const int32_t drop = candidate.second;
            communities[keep].sum += communities[drop].sum;
            communities[keep].sum_squared_norm +=
                communities[drop].sum_squared_norm;
            communities[keep].count += communities[drop].count;
            ++communities[keep].version;
            communities[drop].active = false;
            communities[drop].parent = keep;
            ++communities[drop].version;
            --groups;
            if (groups > 1) {
                candidates.push(nearest_ward_candidate(communities, keep));
            }
        }

        std::vector<int32_t> remap(n_communities, -1);
        int32_t next = 0;
        for (int32_t community = 0; community < n_communities; ++community) {
            if (communities[community].active) remap[community] = next++;
        }
        for (int32_t document = 0; document < n; ++document) {
            labels(document) = remap[find_root(
                communities, labels(document))];
        }
        return labels;
    }

    while (groups < requested_communities) {
        int32_t selected = -1;
        double largest_sse = -1.0;
        for (int32_t community = 0; community < groups; ++community) {
            if (communities[community].count < 2) continue;
            const double sse = community_sse(communities[community]);
            if (sse > largest_sse) {
                largest_sse = sse;
                selected = community;
            }
        }
        if (selected < 0) {
            throw std::runtime_error("Cannot split communities further");
        }

        const std::vector<int32_t> old_members =
            std::move(communities[selected].members);
        RowMajorMatrixXd subset(old_members.size(), normalized.cols());
        for (size_t j = 0; j < old_members.size(); ++j) {
            subset.row(static_cast<Eigen::Index>(j)) =
                normalized.row(old_members[j]);
        }
        DenseKMeansOptions split_options = kmeans_options;
        split_options.n_clusters = 2;
        const uint32_t mixed_seed = static_cast<uint32_t>(kmeans_options.seed)
            + static_cast<uint32_t>(groups);
        const int64_t signed_seed = mixed_seed
                <= static_cast<uint32_t>(std::numeric_limits<int32_t>::max())
            ? static_cast<int64_t>(mixed_seed)
            : static_cast<int64_t>(mixed_seed) - (int64_t{1} << 32);
        split_options.seed = static_cast<int32_t>(signed_seed);
        const DenseKMeansResult split = dense_kmeans(subset, split_options);

        std::vector<int32_t> first_members;
        std::vector<int32_t> second_members;
        first_members.reserve(old_members.size());
        second_members.reserve(old_members.size());
        for (size_t j = 0; j < old_members.size(); ++j) {
            const int32_t document = old_members[j];
            if (split.assignments(static_cast<Eigen::Index>(j)) == 0) {
                first_members.push_back(document);
                labels(document) = selected;
            } else {
                second_members.push_back(document);
                labels(document) = groups;
            }
        }
        CommunityStats first = summarize_members(first_members, normalized);
        first.parent = selected;
        communities[selected] = std::move(first);
        CommunityStats second = summarize_members(second_members, normalized);
        second.parent = groups;
        communities.push_back(std::move(second));
        ++groups;
    }
    return labels;
}

Eigen::VectorXi reconcile_cosine_communities(
    const Eigen::VectorXi& membership, int32_t n_communities,
    int32_t requested_communities,
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& kmeans_options) {
    return reconcile_simplex_communities(membership, n_communities,
        requested_communities, observations, SimplexMetric::Cosine,
        kmeans_options);
}
