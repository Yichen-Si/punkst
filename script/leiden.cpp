#include "clustering_core/cosine_clustering.hpp"
#include "punkst.h"
#include "utils.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct FactorTable {
    std::vector<std::string> identifiers;
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
        int32_t identifier_column, bool allow_topk) {
    if (identifier_column < 0) {
        throw std::invalid_argument("--unit-icol-id must be nonnegative");
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
            "--unit-icol-id is outside the factor table");
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
                    "--unit-icol-id must select a non-factor column");
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
                    "--unit-icol-id must select a non-factor column");
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
        SimplexMetric metric, const std::vector<LeidenRun>& runs) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot open Leiden diagnostics: " + path);
    }
    output << "resolution\tcluster_column\tn_units\tn_factors\tmetric\tinput_format"
        "\ttopk_approximation\trequested_neighbors\tused_neighbors\tn_edges"
        "\tseed\tmax_iterations\tn_communities\tquality\titerations"
        "\tconverged\trequested_knn_backend\tresolved_knn_backend"
        "\tresolved_flat_kernel\tmetric_transform_seconds\tindex_build_seconds"
        "\tquery_seconds\ttopk_seconds\tgraph_reduction_seconds"
        "\tgraph_seconds\tleiden_seconds\ttotal_clustering_seconds\n";
    output << std::setprecision(17);
    const int32_t used_neighbors = std::min<int32_t>(requested_neighbors,
        static_cast<int32_t>(table.values.rows()) - 1);
    const char* flat_kernel = graph.diagnostics.resolved_backend
            == CosineKnnBackend::Flat
        ? cosine_flat_kernel_name(graph.diagnostics.resolved_flat_kernel)
        : "NA";
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
            << graph.diagnostics.timings.graph_reduction_seconds << '\t'
            << graph_seconds << '\t' << run.seconds << '\t'
            << graph_seconds + run.seconds << '\n';
    }
}

} // namespace

int32_t cmdLeiden(int argc, char** argv) {
    std::string input_path, output_prefix, knn_backend = "auto";
    std::string metric_name = "cosine";
    std::vector<double> resolutions;
    int32_t identifier_column = 0;
    int32_t neighbors = 15, max_iterations = -1, seed = 1, threads = 1;
    double knn_epsilon = 0.0;
    bool allow_topk = false;

    ParamList parameters;
    parameters
      .add_option("in-theta",
          "Dense Gamma-Poisson/LDA theta table", input_path, true)
      .add_option("out-prefix", "Output prefix", output_prefix, true)
      .add_option("unit-icol-id",
          "0-based input column used as the unit identifier",
          identifier_column)
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
          "Metric k-NN backend: auto, kdtree, or flat", knn_backend)
      .add_option("knn-epsilon",
          "Nanoflann search epsilon; positive values require kdtree",
          knn_epsilon)
      .add_option("allow-topk",
          "Approximate K/P input by setting omitted factors to zero",
          allow_topk);

    try {
        parameters.readArgs(argc, argv);
        if (resolutions.empty()) resolutions.push_back(1.0);
        parameters.print_options();
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

        const SimplexMetric metric = parse_simplex_metric(metric_name);
        FactorTable table = read_factor_table(
            input_path, identifier_column, allow_topk);
        if (table.topk) {
            warning("Clustering K/P top-k input after reconstructing omitted factors as zero; %s geometry is approximate",
                simplex_metric_name(metric));
        }

        CosineKnnOptions knn_options;
        knn_options.n_neighbors = neighbors;
        knn_options.knn_search_epsilon = knn_epsilon;
        knn_options.backend = parse_cosine_knn_backend(knn_backend);
        knn_options.n_threads = threads;
        const Clock::time_point graph_begin = Clock::now();
        const CosineKnnResult graph = simplex_knn(
            table.values, metric, knn_options);
        const double graph_seconds = elapsed_seconds(graph_begin);

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
            graph_seconds, max_iterations, seed, metric, runs);
        notice("Reused one %zu-edge %s k-NN graph for %zu Leiden resolution(s) using the %s backend",
            graph.graph.edges.size(), simplex_metric_name(metric), runs.size(),
            cosine_knn_backend_name(graph.diagnostics.resolved_backend));
        notice("Leiden outputs written to %s and %s",
            clusters_path.c_str(), diagnostics_path.c_str());
    } catch (const std::exception& exception) {
        std::cerr << "Leiden clustering failed: " << exception.what() << "\n";
        return 1;
    }
    return 0;
}
