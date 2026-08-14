#include "clustering_core/cosine_clustering.hpp"
#include "punkst.h"
#include "cli_common.hpp"
#include "linear_embedding_cli.hpp"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;
using punkst_cli::ProjectionSpace;

struct FactorTable {
    std::vector<std::string> identifiers;
    std::vector<std::string> factors;
    RowMajorMatrixXd values;
    bool topk = false;
};

struct LeidenRun {
    double resolution = 1.0;
    LeidenResult result;
    double seconds = 0.0;
};

double elapsed_seconds(const Clock::time_point& begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

std::string resolution_label(double resolution) {
    std::ostringstream output;
    output << std::setprecision(15) << std::defaultfloat << resolution;
    return output.str();
}

std::string cluster_column_name(const LeidenRun& run, size_t index) {
    return index == 0
        ? "cluster" : "cluster_r" + resolution_label(run.resolution);
}

int32_t parse_numbered_column(const std::string& name, char prefix) {
    if (name.size() < 2 || name.front() != prefix) return -1;
    int32_t index = 0;
    if (!str2int32(name.substr(1), index) || index <= 0) return -1;
    return index;
}

FactorTable read_factor_table(const std::string& path,
        int32_t identifier_column, int32_t factor_column_start,
        int32_t factor_column_end, bool allow_topk) {
    if (identifier_column < 0) {
        throw std::invalid_argument("--icol-id must be nonnegative");
    }
    if (factor_column_start < -1 || factor_column_end < -1) {
        throw std::invalid_argument(
            "factor column indices must be nonnegative");
    }
    const bool has_factor_column_start = factor_column_start >= 0;
    const bool has_factor_column_end = factor_column_end >= 0;
    if (has_factor_column_start != has_factor_column_end) {
        throw std::invalid_argument(
            "--icol-factor-start and --icol-factor-end must be supplied together");
    }
    const bool explicit_factor_columns = has_factor_column_start;
    if (explicit_factor_columns
            && factor_column_end < factor_column_start) {
        throw std::invalid_argument(
            "--icol-factor-end must not precede --icol-factor-start");
    }
    if (explicit_factor_columns
            && factor_column_end - factor_column_start + 1 < 2) {
        throw std::invalid_argument(
            "Leiden clustering requires at least two factor columns");
    }
    if (explicit_factor_columns && allow_topk) {
        throw std::invalid_argument(
            "--allow-topk cannot be combined with explicit factor columns");
    }
    TextLineReader reader(path);
    std::string line;
    while (reader.getline(line) && line.empty()) {}
    if (line.empty()) {
        throw std::runtime_error("Factor table is empty: " + path);
    }
    const std::vector<std::string> header =
        split_delimited(strip_leading_hash(line), '\t');
    if (identifier_column >= static_cast<int32_t>(header.size())) {
        throw std::invalid_argument(
            "--icol-id is outside the factor table");
    }
    if (explicit_factor_columns
            && factor_column_end >= static_cast<int32_t>(header.size())) {
        throw std::invalid_argument(
            "--icol-factor-end is outside the factor table");
    }
    if (explicit_factor_columns
            && identifier_column >= factor_column_start
            && identifier_column <= factor_column_end) {
        throw std::invalid_argument(
            "--icol-id must select a non-factor column");
    }

    std::unordered_set<std::string> header_names;
    std::vector<std::pair<int32_t, int32_t>> dense_columns;
    std::map<int32_t, std::pair<int32_t, int32_t>> topk_columns;
    for (int32_t column = 0; column < static_cast<int32_t>(header.size());
            ++column) {
        const std::string& name = header[static_cast<size_t>(column)];
        if (name.empty() || !header_names.insert(name).second) {
            throw std::runtime_error(
                "Factor table has an empty or duplicate header: " + name);
        }
        if (!explicit_factor_columns) {
            int32_t factor = -1;
            if (str2int32(name, factor) && factor >= 0) {
                dense_columns.emplace_back(factor, column);
            }
            const int32_t k_index = parse_numbered_column(name, 'K');
            if (k_index > 0) {
                if (topk_columns[k_index].first > 0) {
                    throw std::runtime_error(
                        "Duplicate K column index in factor table header");
                }
                topk_columns[k_index].first = column + 1;
            }
            const int32_t p_index = parse_numbered_column(name, 'P');
            if (p_index > 0) {
                if (topk_columns[p_index].second > 0) {
                    throw std::runtime_error(
                        "Duplicate P column index in factor table header");
                }
                topk_columns[p_index].second = column + 1;
            }
        }
    }
    if (explicit_factor_columns) {
        for (int32_t column = factor_column_start;
                column <= factor_column_end; ++column) {
            dense_columns.emplace_back(
                column - factor_column_start, column);
        }
    }

    const bool has_dense = !dense_columns.empty();
    const bool has_topk = !topk_columns.empty();
    if (has_dense && has_topk) {
        throw std::runtime_error(
            "Factor table cannot mix dense 0..K-1 columns with K/P columns");
    }
    if (!has_dense && !has_topk) {
        throw std::runtime_error(
            "Factor table has neither dense 0..K-1 nor K/P factor columns");
    }
    if (has_topk && !allow_topk) {
        throw std::runtime_error(
            "K/P top-k input is truncated; pass --allow-topk to reconstruct omitted factors as zero");
    }

    if (has_dense) {
        std::sort(dense_columns.begin(), dense_columns.end());
        for (size_t factor = 0; factor < dense_columns.size(); ++factor) {
            if (dense_columns[factor].first != static_cast<int32_t>(factor)) {
                throw std::runtime_error(
                    "Dense factor columns must be contiguous 0..K-1");
            }
        }
        if (dense_columns.size() < 2) {
            throw std::runtime_error(
                "Leiden clustering requires at least two factor columns");
        }
        for (const auto& factor : dense_columns) {
            if (factor.second == identifier_column) {
                throw std::invalid_argument(
                    "--icol-id must select a non-factor column");
            }
        }
    } else {
        int32_t expected = 1;
        for (const auto& item : topk_columns) {
            if (item.first != expected || item.second.first <= 0
                    || item.second.second <= 0) {
                throw std::runtime_error(
                    "K/P columns must be complete contiguous pairs K1/P1..Kk/Pk");
            }
            const int32_t k_column = item.second.first - 1;
            const int32_t p_column = item.second.second - 1;
            if (k_column == identifier_column || p_column == identifier_column) {
                throw std::invalid_argument(
                    "--icol-id must select a non-factor column");
            }
            ++expected;
        }
    }

    std::vector<double> dense_values;
    std::vector<std::vector<std::pair<int32_t, double>>> sparse_values;
    std::unordered_set<std::string> seen_identifiers;
    FactorTable table;
    table.topk = has_topk;
    uint64_t row_number = 1;
    int32_t maximum_factor = -1;
    while (reader.getline(line)) {
        ++row_number;
        if (line.empty() || is_comment_line(line)) continue;
        const std::vector<std::string> fields = split_delimited(line, '\t');
        if (fields.size() != header.size()) {
            throw std::runtime_error(
                "Factor table row has the wrong column count at line "
                + std::to_string(row_number));
        }
        const std::string& identifier =
            fields[static_cast<size_t>(identifier_column)];
        if (identifier.empty() || !seen_identifiers.insert(identifier).second) {
            throw std::runtime_error(
                "Factor table has an empty or duplicate identifier at line "
                + std::to_string(row_number));
        }
        table.identifiers.push_back(identifier);

        double squared_norm = 0.0;
        if (has_dense) {
            for (const auto& factor : dense_columns) {
                double value = 0.0;
                if (!str2double(fields[static_cast<size_t>(factor.second)], value)
                        || !std::isfinite(value) || value < 0.0) {
                    throw std::runtime_error(
                        "Invalid factor value at line "
                        + std::to_string(row_number));
                }
                dense_values.push_back(value);
                squared_norm += value * value;
            }
        } else {
            std::vector<std::pair<int32_t, double>> row;
            row.reserve(topk_columns.size());
            std::unordered_set<int32_t> seen_factors;
            for (const auto& item : topk_columns) {
                const int32_t k_column = item.second.first - 1;
                const int32_t p_column = item.second.second - 1;
                int32_t factor = -1;
                double probability = 0.0;
                if (!str2int32(fields[static_cast<size_t>(k_column)], factor)
                        || factor < 0
                        || !str2double(fields[static_cast<size_t>(p_column)],
                            probability)
                        || !std::isfinite(probability) || probability < 0.0) {
                    throw std::runtime_error(
                        "Invalid K/P factor pair at line "
                        + std::to_string(row_number));
                }
                if (!seen_factors.insert(factor).second) {
                    throw std::runtime_error(
                        "Duplicate K/P factor index at line "
                        + std::to_string(row_number));
                }
                row.emplace_back(factor, probability);
                maximum_factor = std::max(maximum_factor, factor);
                squared_norm += probability * probability;
            }
            sparse_values.push_back(std::move(row));
        }
        if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
            throw std::runtime_error(
                "Factor table contains a zero or invalid vector at line "
                + std::to_string(row_number));
        }
    }
    if (table.identifiers.size() < 2) {
        throw std::runtime_error(
            "Leiden clustering requires at least two input units");
    }

    const int32_t dimensions = has_dense
        ? static_cast<int32_t>(dense_columns.size()) : maximum_factor + 1;
    if (dimensions < 2) {
        throw std::runtime_error(
            "Leiden clustering requires at least two inferred factors");
    }
    table.factors.resize(static_cast<size_t>(dimensions));
    if (has_dense) {
        for (const auto& factor : dense_columns) {
            table.factors[static_cast<size_t>(factor.first)] =
                header[static_cast<size_t>(factor.second)];
        }
    } else {
        for (int32_t factor = 0; factor < dimensions; ++factor) {
            table.factors[static_cast<size_t>(factor)] =
                std::to_string(factor);
        }
    }
    table.values = RowMajorMatrixXd::Zero(
        static_cast<Eigen::Index>(table.identifiers.size()), dimensions);
    if (has_dense) {
        for (Eigen::Index row = 0; row < table.values.rows(); ++row) {
            for (int32_t factor = 0; factor < dimensions; ++factor) {
                table.values(row, factor) = dense_values[
                    static_cast<size_t>(row * dimensions + factor)];
            }
        }
    } else {
        for (size_t row = 0; row < sparse_values.size(); ++row) {
            for (const auto& factor : sparse_values[row]) {
                table.values(static_cast<Eigen::Index>(row), factor.first) =
                    factor.second;
            }
        }
    }
    return table;
}

void validate_resolutions(const std::vector<double>& resolutions) {
    std::set<double> seen;
    std::unordered_set<std::string> labels;
    for (const double resolution : resolutions) {
        if (!(resolution > 0.0) || !std::isfinite(resolution)) {
            throw std::invalid_argument(
                "--resolution values must be positive and finite");
        }
        if (!seen.insert(resolution).second) {
            throw std::invalid_argument(
                "--resolution values must not contain duplicates");
        }
        if (!labels.insert(resolution_label(resolution)).second) {
            throw std::invalid_argument(
                "--resolution values are too close to name output columns uniquely");
        }
    }
}

void write_clusters(const std::string& path, const FactorTable& table,
        const std::vector<LeidenRun>& runs) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open cluster output: " + path);
    }
    output << "#id";
    for (size_t run = 0; run < runs.size(); ++run) {
        output << '\t' << cluster_column_name(runs[run], run);
    }
    output << "\n";
    for (size_t row = 0; row < table.identifiers.size(); ++row) {
        output << table.identifiers[row];
        for (const LeidenRun& run : runs) {
            output << '\t' << run.result.membership(
                static_cast<Eigen::Index>(row));
        }
        output << '\n';
    }
}

void write_diagnostics(const std::string& path, const FactorTable& table,
        const CosineKnnResult& graph, int32_t requested_neighbors,
        double graph_seconds, int32_t max_iterations, int32_t seed,
        SimplexMetric metric, const CosineKnnOptions& knn_options,
        const std::vector<LeidenRun>& runs) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open Leiden diagnostics: " + path);
    }
    const bool ann_backend = graph.diagnostics.resolved_backend
            == CosineKnnBackend::Hnsw
        || graph.diagnostics.resolved_backend == CosineKnnBackend::NnDescent;
    output << "resolution\tcluster_column\tn_units\tn_factors\tmetric\tinput_format"
        "\ttopk_approximation\trequested_neighbors\tused_neighbors\tn_edges"
        "\tseed\tmax_iterations\tn_communities\tquality\titerations"
        "\tconverged\trequested_knn_backend\tresolved_knn_backend"
        "\tresolved_flat_kernel\tmetric_transform_seconds\tindex_build_seconds"
        "\tquery_seconds\ttopk_seconds\tgraph_reduction_seconds";
    if (ann_backend) {
        output << "\tann_sample_size\trequested_ann_parameter"
            "\tresolved_ann_parameter\tresolved_ann_candidates"
            "\taudit_mean_recall\taudit_recall_lcb\taudit_passed\tforced"
            "\taudit_trials\taudit_seconds\thnsw_m\thnsw_ef_construction"
            "\thnsw_max_ef_search\tnndescent_graph_size\tnndescent_s";
    }
    output << "\tgraph_seconds\tleiden_seconds\ttotal_clustering_seconds\n";
    output << std::setprecision(17);
    const int32_t used_neighbors = std::min<int32_t>(requested_neighbors,
        static_cast<int32_t>(table.values.rows()) - 1);
    const char* flat_kernel = graph.diagnostics.resolved_backend
            == CosineKnnBackend::Flat
        ? cosine_flat_kernel_name(graph.diagnostics.resolved_flat_kernel)
        : "NA";
    std::ostringstream audit_trials;
    for (size_t i = 0; i < graph.diagnostics.audit_trials.size(); ++i) {
        if (i > 0) audit_trials << ',';
        const CosineKnnAuditTrial& trial =
            graph.diagnostics.audit_trials[i];
        audit_trials << trial.parameter << ':' << trial.mean_recall
            << ':' << trial.recall_lcb;
    }
    const std::string audit_trial_text = graph.diagnostics.audit_trials.empty()
        ? "NA" : audit_trials.str();
    for (size_t index = 0; index < runs.size(); ++index) {
        const LeidenRun& run = runs[index];
        output << run.resolution << '\t'
            << cluster_column_name(run, index) << '\t'
            << table.values.rows() << '\t'
            << table.values.cols() << '\t'
            << simplex_metric_name(metric) << '\t'
            << (table.topk ? "topk" : "dense") << '\t'
            << (table.topk ? 1 : 0) << '\t' << requested_neighbors << '\t'
            << used_neighbors << '\t' << graph.graph.edges.size() << '\t'
            << seed << '\t' << max_iterations << '\t'
            << run.result.n_communities << '\t' << run.result.quality << '\t'
            << run.result.iterations << '\t' << (run.result.converged ? 1 : 0)
            << '\t' << cosine_knn_backend_name(
                graph.diagnostics.requested_backend)
            << '\t' << cosine_knn_backend_name(
                graph.diagnostics.resolved_backend)
            << '\t' << flat_kernel << '\t'
            << graph.diagnostics.timings.normalization_seconds << '\t'
            << graph.diagnostics.timings.index_build_seconds << '\t'
            << graph.diagnostics.timings.query_seconds << '\t'
            << graph.diagnostics.timings.topk_seconds << '\t'
            << graph.diagnostics.timings.graph_reduction_seconds;
        if (ann_backend) {
            output << '\t' << graph.diagnostics.sample_size << '\t'
                << graph.diagnostics.requested_ann_parameter << '\t'
                << graph.diagnostics.resolved_ann_parameter << '\t'
                << graph.diagnostics.resolved_ann_candidates << '\t'
                << graph.diagnostics.audit_mean_recall << '\t'
                << graph.diagnostics.audit_recall_lcb << '\t'
                << (graph.diagnostics.audit_passed ? 1 : 0) << '\t'
                << (graph.diagnostics.forced ? 1 : 0) << '\t'
                << audit_trial_text << '\t'
                << graph.diagnostics.timings.audit_seconds << '\t'
                << knn_options.hnsw_m << '\t'
                << knn_options.hnsw_ef_construction << '\t'
                << knn_options.hnsw_max_ef_search << '\t'
                << knn_options.nndescent_graph_size << '\t'
                << knn_options.nndescent_sample_candidates;
        }
        output << '\t' << graph_seconds << '\t' << run.seconds << '\t'
            << graph_seconds + run.seconds << '\n';
    }
}

} // namespace

int32_t cmdLeiden(int argc, char** argv) {
    std::string input_path, output_prefix, knn_backend = "auto";
    std::string metric_name = "cosine";
    std::vector<double> resolutions;
    int32_t identifier_column = 0;
    int32_t factor_column_start = -1, factor_column_end = -1;
    int32_t neighbors = 15, max_iterations = -1, seed = 1, threads = 1;
    double knn_epsilon = 0.0;
    int32_t hnsw_m = 16, hnsw_ef_construction = 100;
    int32_t hnsw_ef_search = 0, hnsw_max_ef_search = 512;
    int32_t hnsw_candidates = 0, hnsw_audit_queries = 256;
    int32_t nndescent_iterations = 0, nndescent_graph_size = 0;
    int32_t nndescent_s = 10, nndescent_audit_queries = 256;
    double hnsw_recall = 0.98, nndescent_recall = 0.98;
    punkst_cli::LinearEmbeddingCliOptions embedding_cli;
    bool allow_topk = false, hnsw_force = false;
    bool skip_projection = false;

    ParamList parameters;
    parameters
      .add_option("in-theta",
          "Dense Gamma-Poisson/LDA theta table", input_path, true)
      .add_option("out-prefix", "Output prefix", output_prefix, true)
      .add_option("icol-id",
          "0-based input column used as the unit identifier",
          identifier_column)
      .add_option("icol-factor-start",
          "0-based first factor-proportion column (inclusive)",
          factor_column_start)
      .add_option("icol-factor-end",
          "0-based last factor-proportion column (inclusive)",
          factor_column_end)
      .add_option("metric", "Simplex metric: cosine or hellinger",
          metric_name)
      .add_option("neighbors", "Metric k-NN neighbors", neighbors)
      .add_option("resolution",
          "One or more Leiden RBConfiguration resolutions", resolutions)
      .add_option("max-iter",
          "Maximum Leiden passes; negative runs to convergence",
          max_iterations)
      .add_option("seed", "Leiden random seed", seed)
      .add_option("threads", "Number of k-NN worker threads", threads)
      .add_option("knn-backend",
          "Metric k-NN backend: auto, kdtree, flat, hnsw, or nndescent", knn_backend)
      .add_option("knn-epsilon",
          "Nanoflann search epsilon; positive values require kdtree",
          knn_epsilon)
      .add_option("hnsw-m", "HNSW graph degree", hnsw_m)
      .add_option("hnsw-ef-construction",
          "HNSW construction effort", hnsw_ef_construction)
      .add_option("hnsw-ef-search",
          "HNSW search effort; 0 tunes automatically", hnsw_ef_search)
      .add_option("hnsw-max-ef-search",
          "Maximum automatically tuned HNSW search effort",
          hnsw_max_ef_search)
      .add_option("hnsw-candidates",
          "HNSW candidates per unit; 0 uses max(64,4*k)",
          hnsw_candidates)
      .add_option("hnsw-audit-queries",
          "Exact sampled queries for HNSW recall calibration",
          hnsw_audit_queries)
      .add_option("hnsw-recall",
          "Required HNSW sampled-recall lower bound", hnsw_recall)
      .add_option("hnsw-force",
          "Run HNSW despite a failed sampled-recall audit", hnsw_force)
      .add_option("nndescent-iterations",
          "NN-descent refinements; 0 uses max(10,round(log2(n)))",
          nndescent_iterations)
      .add_option("nndescent-graph-size",
          "NN-descent graph size; 0 uses max(64,4*k)",
          nndescent_graph_size)
      .add_option("nndescent-s",
          "NN-descent candidate-pool parameter", nndescent_s)
      .add_option("nndescent-audit-queries",
          "Exact sampled queries for NN-descent recall auditing",
          nndescent_audit_queries)
      .add_option("nndescent-recall",
          "Required NN-descent sampled-recall lower bound",
          nndescent_recall)
      .add_option("allow-topk",
          "Approximate K/P input by setting omitted factors to zero",
          allow_topk)
      .add_option("projection-space",
          "Projection coordinates: both, linear, or ilr",
          embedding_cli.projection_space)
      .add_option("projection-dim",
          "Maximum projection dimensions", embedding_cli.values.dimensions)
      .add_option("projection-center-floor",
          "Positive factor floor for the ILR projection",
          embedding_cli.values.center_floor)
      .add_option("projection-covariance-floor",
          "Positive eigenvalue floor for projection whitening",
          embedding_cli.values.covariance_floor)
      .add_option("projection-whitening",
          "Projection whitening covariance: sample or mixture",
          embedding_cli.whitening)
      .add_option("projection-full",
          "Also compute projections using covariance-shape differences",
          embedding_cli.include_full);
    embedding_cli.add_qda_options(parameters);
    parameters
      .add_option("skip-projection",
          "Disable all post-clustering projection work",
          skip_projection);

    try {
        parameters.readArgs(argc, argv);
        const bool projections_disabled = skip_projection;
        if (resolutions.empty()) resolutions.push_back(1.0);
        validate_resolutions(resolutions);
        if (neighbors <= 0) {
            throw std::invalid_argument("--neighbors must be positive");
        }
        if (max_iterations == 0) {
            throw std::invalid_argument("--max-iter must be nonzero");
        }
        if (threads <= 0) {
            throw std::invalid_argument("--threads must be positive");
        }
        if (!std::isfinite(knn_epsilon) || knn_epsilon < 0.0) {
            throw std::invalid_argument(
                "--knn-epsilon must be finite and nonnegative");
        }
        std::vector<ProjectionSpace> projection_spaces;
        if (!projections_disabled) {
            projection_spaces =
                punkst_cli::parse_projection_spaces(
                    embedding_cli.projection_space);
            if (embedding_cli.projection_space == "ilr") {
                throw std::invalid_argument(
                    "--projection-space must be linear or both for Leiden embeddings");
            }
            if (embedding_cli.values.dimensions <= 0) {
                throw std::invalid_argument(
                    "--projection-dim must be positive");
            }
            if (!(embedding_cli.values.covariance_floor > 0.0)
                || !std::isfinite(
                    embedding_cli.values.covariance_floor)) {
                throw std::invalid_argument(
                    "--projection-covariance-floor must be positive and finite");
            }
            if (std::find(projection_spaces.begin(), projection_spaces.end(),
                    ProjectionSpace::Ilr) != projection_spaces.end()
                && (!(embedding_cli.values.center_floor > 0.0)
                    || !std::isfinite(
                        embedding_cli.values.center_floor))) {
                throw std::invalid_argument(
                    "--projection-center-floor must be positive and finite");
            }
            embedding_cli.finalize_qda_options(parameters);
            embedding_cli.values.projection_spaces = projection_spaces;
            embedding_cli.values.whitening =
                punkst::projection::parse_visualization_whitening(
                    embedding_cli.whitening);
            embedding_cli.values.threads = threads;
            embedding_cli.values.include_full = embedding_cli.include_full;
            embedding_cli.values.validate();
        }
        parameters.print_options();

        const SimplexMetric metric = parse_simplex_metric(metric_name);
        FactorTable table = read_factor_table(
            input_path, identifier_column, factor_column_start,
            factor_column_end, allow_topk);
        if (table.topk) {
            warning("Clustering K/P top-k input after reconstructing omitted factors as zero; %s geometry is approximate",
                simplex_metric_name(metric));
        }

        CosineKnnOptions knn_options;
        knn_options.n_neighbors = neighbors;
        knn_options.knn_search_epsilon = knn_epsilon;
        knn_options.backend = parse_cosine_knn_backend(knn_backend);
        knn_options.n_threads = threads;
        knn_options.hnsw_m = hnsw_m;
        knn_options.hnsw_ef_construction = hnsw_ef_construction;
        knn_options.hnsw_ef_search = hnsw_ef_search;
        knn_options.hnsw_max_ef_search = hnsw_max_ef_search;
        knn_options.hnsw_candidates = hnsw_candidates;
        knn_options.hnsw_audit_queries = hnsw_audit_queries;
        knn_options.hnsw_recall = hnsw_recall;
        knn_options.hnsw_force = hnsw_force;
        knn_options.nndescent_iterations = nndescent_iterations;
        knn_options.nndescent_graph_size = nndescent_graph_size;
        knn_options.nndescent_sample_candidates = nndescent_s;
        knn_options.nndescent_audit_queries = nndescent_audit_queries;
        knn_options.nndescent_recall = nndescent_recall;
        knn_options.ann_seed = seed;
        const Clock::time_point graph_begin = Clock::now();
        const CosineKnnResult graph = simplex_knn(
            table.values, metric, knn_options);
        const double graph_seconds = elapsed_seconds(graph_begin);
        if (graph.diagnostics.forced) {
            warning("HNSW sampled recall LCB %.6g was below target %.6g; continuing because --hnsw-force was set",
                graph.diagnostics.audit_recall_lcb, hnsw_recall);
        }

        std::vector<LeidenRun> runs;
        runs.reserve(resolutions.size());
        for (const double resolution : resolutions) {
            LeidenOptions options;
            options.resolution = resolution;
            options.max_iterations = max_iterations;
            options.seed = seed;
            const Clock::time_point leiden_begin = Clock::now();
            LeidenRun run;
            run.resolution = resolution;
            run.result = leiden_cluster(graph.graph.n_nodes,
                graph.graph.edges, graph.graph.weights, options);
            run.seconds = elapsed_seconds(leiden_begin);
            notice("Leiden resolution %.8g found %d communities (quality %.8g, %d passes, %s)",
                resolution, run.result.n_communities, run.result.quality,
                run.result.iterations,
                run.result.converged ? "converged" : "iteration limit");
            runs.push_back(std::move(run));
        }

        const std::string clusters_path = output_prefix + ".clusters.tsv";
        const std::string diagnostics_path =
            output_prefix + ".diagnostics.tsv";
        write_clusters(clusters_path, table, runs);
        write_diagnostics(diagnostics_path, table, graph, neighbors,
            graph_seconds, max_iterations, seed, metric, knn_options, runs);
        for (const LeidenRun& run : runs) {
            const std::string partition_prefix = runs.size() == 1
                ? output_prefix
                : output_prefix + ".r" + resolution_label(run.resolution);
            const Eigen::MatrixXd cluster_factors =
                punkst::linear_embedding::aggregate_cluster_factors(
                    table.values, run.result.membership,
                    run.result.n_communities);
            punkst::linear_embedding::write_cluster_factors(
                partition_prefix + ".cluster_factors.tsv", table.factors,
                cluster_factors);
        }
        if (!projection_spaces.empty()) {
            punkst_cli::TopicCenterTable theta;
            theta.identifiers = table.identifiers;
            theta.topics = table.factors;
            theta.values = table.values;
            std::vector<int32_t> matched_rows(table.identifiers.size());
            std::iota(matched_rows.begin(), matched_rows.end(), 0);
            for (size_t run_index = 0; run_index < runs.size(); ++run_index) {
                const LeidenRun& run = runs[run_index];
                if (run.result.n_communities < 2) {
                    warning("Leiden resolution %.8g has one community; omitting its embeddings",
                        run.resolution);
                    continue;
                }
                const std::string label = "r"
                    + resolution_label(run.resolution);
                const std::string projection_prefix = runs.size() == 1
                    ? output_prefix + ".projection"
                    : output_prefix + ".projection." + label;
                try {
                    punkst::linear_embedding::run_partition(theta,
                        matched_rows, run.result.membership,
                        run.result.n_communities, label, projection_prefix,
                        embedding_cli.values);
                } catch (const std::exception& exception) {
                    warning("Leiden resolution %.8g embeddings omitted: %s",
                        run.resolution, exception.what());
                }
            }
        }
        notice("Reused one %zu-edge %s k-NN graph for %zu Leiden resolution(s) using the %s backend",
            graph.graph.edges.size(), simplex_metric_name(metric), runs.size(),
            cosine_knn_backend_name(graph.diagnostics.resolved_backend));
        if (graph.diagnostics.resolved_backend == CosineKnnBackend::Hnsw
                || graph.diagnostics.resolved_backend
                    == CosineKnnBackend::NnDescent) {
            notice("ANN resolved parameter %d with %d candidates (sampled recall %.6g, 95%% LCB %.6g)",
                graph.diagnostics.resolved_ann_parameter,
                graph.diagnostics.resolved_ann_candidates,
                graph.diagnostics.audit_mean_recall,
                graph.diagnostics.audit_recall_lcb);
        }
        notice("Leiden outputs written to %s and %s",
            clusters_path.c_str(), diagnostics_path.c_str());
    } catch (const std::exception& exception) {
        std::cerr << "Leiden clustering failed: " << exception.what() << "\n";
        return 1;
    }
    return 0;
}
