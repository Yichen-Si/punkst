#include "punkst.h"
#include "cli_common.hpp"
#include "linear_embedding_cli.hpp"
#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

using punkst::linear_embedding::ProjectionSpace;

struct PartitionTable {
    std::vector<std::string> identifiers;
    std::vector<std::vector<std::string>> values;
};

void validate_partition_columns(int32_t identifier_column,
        const std::vector<int32_t>& partition_columns) {
    if (identifier_column < 0) {
        throw std::invalid_argument("--icol-id must be nonnegative");
    }
    if (partition_columns.empty()) {
        throw std::invalid_argument(
            "--icol-partition requires at least one column");
    }
    std::unordered_set<int32_t> seen;
    for (const int32_t column : partition_columns) {
        if (column < 0) {
            throw std::invalid_argument(
                "--icol-partition values must be nonnegative");
        }
        if (column == identifier_column) {
            throw std::invalid_argument(
                "--icol-id and --icol-partition must select different columns");
        }
        if (!seen.insert(column).second) {
            throw std::invalid_argument(
                "--icol-partition values must not contain duplicates");
        }
    }
}

PartitionTable read_partitions(const std::string& path,
    int32_t identifier_column,
    const std::vector<int32_t>& partition_columns) {
    validate_partition_columns(identifier_column, partition_columns);
    const int32_t maximum_column = std::max(identifier_column,
        *std::max_element(partition_columns.begin(), partition_columns.end()));
    TextLineReader reader(path);
    PartitionTable out;
    out.values.resize(partition_columns.size());
    std::unordered_set<std::string> identifiers;
    std::string line;
    uint64_t line_number = 0;
    while (reader.getline(line)) {
        ++line_number;
        if (line.empty() || is_comment_line(line)) continue;
        const std::vector<std::string> fields = split_delimited(line, '\t');
        if (maximum_column >= static_cast<int32_t>(fields.size())) {
            throw std::runtime_error(
                "Partition row has too few columns at line "
                + std::to_string(line_number));
        }
        const std::string& identifier = fields[identifier_column];
        if (identifier.empty() || !identifiers.insert(identifier).second) {
            throw std::runtime_error(
                "Empty or duplicate partition identifier at line "
                + std::to_string(line_number));
        }
        out.identifiers.push_back(identifier);
        for (size_t partition = 0;
                partition < partition_columns.size(); ++partition) {
            const std::string& value =
                fields[static_cast<size_t>(partition_columns[partition])];
            if (value.empty()) {
                throw std::runtime_error(
                    "Empty partition value at line "
                    + std::to_string(line_number));
            }
            out.values[partition].push_back(value);
        }
    }
    if (out.identifiers.empty()) {
        throw std::runtime_error("Partition table has no data rows: " + path);
    }
    return out;
}

bool safe_partition_label(const std::string& label) {
    if (label.empty() || label == "." || label == ".."
        || label.find('/') != std::string::npos
        || label.find('\\') != std::string::npos) {
        return false;
    }
    return std::none_of(label.begin(), label.end(), [](unsigned char value) {
        return std::iscntrl(value) != 0;
    });
}

std::vector<std::string> resolve_partition_labels(
    const std::vector<int32_t>& columns,
    const std::vector<std::string>& supplied) {
    if (!supplied.empty() && supplied.size() != columns.size()) {
        throw std::invalid_argument(
            "--partition-labels must provide one label per partition column");
    }
    std::vector<std::string> out;
    out.reserve(columns.size());
    for (size_t index = 0; index < columns.size(); ++index) {
        out.push_back(supplied.empty()
            ? std::to_string(columns[index]) : supplied[index]);
    }
    std::unordered_set<std::string> seen;
    for (const std::string& label : out) {
        if (!safe_partition_label(label) || !seen.insert(label).second) {
            throw std::invalid_argument(
                "Partition labels must be unique safe filename components");
        }
    }
    return out;
}

std::vector<int32_t> match_partition_rows(
    const punkst_cli::TopicCenterTable& theta,
    const PartitionTable& partitions, bool id_as_row_index) {
    std::vector<int32_t> matched(partitions.identifiers.size(), -1);
    std::unordered_set<int32_t> seen_rows;
    if (id_as_row_index) {
        for (size_t row = 0; row < partitions.identifiers.size(); ++row) {
            int64_t index = -1;
            if (!str2int64(partitions.identifiers[row], index)
                || index < 0 || index > std::numeric_limits<int32_t>::max()) {
                throw std::runtime_error(
                    "--id-as-row-index requires nonnegative integer identifiers");
            }
            const int32_t resolved = static_cast<int32_t>(index);
            if (!seen_rows.insert(resolved).second) {
                throw std::runtime_error(
                    "Duplicate interpreted theta row index in partition table");
            }
            if (resolved < static_cast<int32_t>(theta.identifiers.size())) {
                matched[row] = resolved;
            }
        }
        return matched;
    }

    std::unordered_map<std::string, int32_t> theta_rows;
    theta_rows.reserve(theta.identifiers.size());
    for (int32_t row = 0;
            row < static_cast<int32_t>(theta.identifiers.size()); ++row) {
        theta_rows.emplace(theta.identifiers[static_cast<size_t>(row)], row);
    }
    for (size_t row = 0; row < partitions.identifiers.size(); ++row) {
        const auto found = theta_rows.find(partitions.identifiers[row]);
        if (found != theta_rows.end()) {
            matched[row] = found->second;
        }
    }
    return matched;
}

void write_cluster_labels(const std::string& path,
        const std::vector<std::string>& labels) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error(
            "Cannot open cluster-label output: " + path);
    }
    output << "#cluster_index\tcluster_name\n";
    for (size_t index = 0; index < labels.size(); ++index) {
        output << index << '\t' << labels[index] << '\n';
    }
}

void filter_factors_by_relative_weight(
        punkst_cli::TopicCenterTable& theta, double threshold) {
    if (!(threshold > 0.0)) return;
    if (!std::isfinite(threshold)) {
        throw std::invalid_argument(
            "--factor-weight-threshold must be finite");
    }
    if (theta.values.rows() == 0 || theta.values.cols() == 0) {
        throw std::invalid_argument("Theta table has no factor values");
    }

    Eigen::VectorXd normalized_sums = Eigen::VectorXd::Zero(
        theta.values.cols());
    for (Eigen::Index row = 0; row < theta.values.rows(); ++row) {
        const double total = theta.values.row(row).sum();
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error(
                "Cannot L1-normalize theta row for factor-weight filtering");
        }
        normalized_sums += theta.values.row(row).transpose() / total;
    }
    const double minimum_weight = threshold
        * static_cast<double>(theta.values.rows());
    std::vector<Eigen::Index> retained;
    retained.reserve(static_cast<size_t>(theta.values.cols()));
    for (Eigen::Index factor = 0; factor < theta.values.cols(); ++factor) {
        if (normalized_sums(factor) > minimum_weight) {
            retained.push_back(factor);
        }
    }
    if (retained.size() < 3) {
        throw std::runtime_error(
            "Factor-weight filter retained "
            + std::to_string(retained.size()) + " of "
            + std::to_string(theta.values.cols())
            + " factors; at least three are required"
            + " (--factor-weight-threshold "
            + std::to_string(threshold) + ")");
    }
    if (retained.size() == static_cast<size_t>(theta.values.cols())) return;

    RowMajorMatrixXd filtered(theta.values.rows(), retained.size());
    std::vector<std::string> topics;
    topics.reserve(retained.size());
    for (size_t target = 0; target < retained.size(); ++target) {
        const Eigen::Index source = retained[target];
        filtered.col(static_cast<Eigen::Index>(target)) =
            theta.values.col(source);
        topics.push_back(theta.topics[static_cast<size_t>(source)]);
    }
    notice("Factor-weight filter retained %zu of %zu factors (threshold %.10g)",
        retained.size(), static_cast<size_t>(theta.values.cols()), threshold);
    theta.values = std::move(filtered);
    theta.topics = std::move(topics);
}

} // namespace

int32_t cmdLinearEmbed(int argc, char** argv) {
    punkst_cli::LinearEmbeddingCliOptions embedding_cli;
    std::string theta_path, partition_path, output_prefix;
    std::vector<int32_t> partition_columns;
    std::vector<std::string> partition_labels;
    int32_t theta_identifier_column = 0;
    int32_t partition_identifier_column = 0;
    int32_t factor_column_start = -1, factor_column_end = -1;
    int32_t dim = -1, visual_dim = -1;
    int32_t threads = embedding_cli.values.threads;
    double factor_weight_threshold = 1e-5;
    bool id_as_row_index = false;

    ParamList parameters;
    parameters
      .add_option("in-theta", "Dense Gamma-Poisson/LDA theta table",
          theta_path, true)
      .add_option("in-partition", "Hard-partition TSV",
          partition_path, true)
      .add_option("out-prefix", "Output prefix", output_prefix, true)
      .add_option("theta-icol-id",
          "0-based theta column used as unit identifier",
          theta_identifier_column)
      .add_option("icol-factor-start",
          "0-based first theta factor column (inclusive)",
          factor_column_start)
      .add_option("icol-factor-end",
          "0-based last theta factor column (inclusive)",
          factor_column_end)
      .add_option("factor-weight-threshold",
          "Keep theta factors with L1-normalized weight above this fraction of input units; zero or negative disables",
          factor_weight_threshold)
      .add_option("icol-id",
          "0-based partition column used as unit identifier",
          partition_identifier_column)
      .add_option("icol-partition",
          "One or more 0-based hard-partition columns",
          partition_columns)
      .add_option("partition-labels",
          "Optional output label for each partition column",
          partition_labels)
      .add_option("id-as-row-index",
          "Interpret partition IDs as 0-based theta row indices",
          id_as_row_index)
      .add_option("dim", "Maximum embedding dimensions", dim)
      .add_option("visual-dim", "Alias for --dim", visual_dim)
      .add_option("whitening", "Whitening covariance: sample or mixture",
          embedding_cli.whitening)
      .add_option("projection-space",
          "Projection coordinates: linear or both",
          embedding_cli.projection_space)
      .add_option("center-floor",
          "Positive factor floor for the ILR projection",
          embedding_cli.values.center_floor)
      .add_option("covariance-floor",
          "Positive eigenvalue floor for the whitening covariance",
          embedding_cli.values.covariance_floor)
      .add_option("visual-full",
          "Also compute the full visualization using covariance-shape differences",
          embedding_cli.include_full);
    embedding_cli.add_discriminant_options(parameters);
    parameters.add_option("threads",
          "Number of covariance and discriminant-projection worker threads",
          threads);

    try {
        parameters.readArgs(argc, argv);
        embedding_cli.finalize_discriminant_options(parameters);
        if (!embedding_cli.values.eigen_projection
                && !embedding_cli.values.qda_projection
                && !embedding_cli.values.lda_projection) {
            throw std::invalid_argument(
                "linear-embed requires at least one of eigen, QDA, or LDA projection");
        }
        if (partition_columns.empty()) partition_columns.push_back(1);
        const bool has_dim = parameters.was_provided("dim");
        const bool has_visual_dim = parameters.was_provided("visual-dim");
        if (has_dim && has_visual_dim && dim != visual_dim) {
            throw std::invalid_argument(
                "--dim and --visual-dim must agree when both are supplied");
        }
        const int32_t requested_dimensions = has_dim ? dim
            : has_visual_dim ? visual_dim : embedding_cli.values.dimensions;
        if (requested_dimensions <= 0) {
            throw std::invalid_argument(
                "--dim/--visual-dim must be positive");
        }
        if (threads <= 0) {
            throw std::invalid_argument("--threads must be positive");
        }
        if (!std::isfinite(factor_weight_threshold)) {
            throw std::invalid_argument(
                "--factor-weight-threshold must be finite");
        }
        if (embedding_cli.values.eigen_projection
            && (!(embedding_cli.values.covariance_floor > 0.0)
                || !std::isfinite(
                    embedding_cli.values.covariance_floor))) {
            throw std::invalid_argument(
                "--covariance-floor must be positive and finite");
        }
        std::vector<ProjectionSpace> projection_spaces{
            ProjectionSpace::Linear};
        if (embedding_cli.values.eigen_projection) {
            projection_spaces = punkst_cli::parse_projection_spaces(
                embedding_cli.projection_space);
            if (embedding_cli.projection_space == "ilr") {
                throw std::invalid_argument(
                    "--projection-space must be linear or both for linear-embed");
            }
        }
        if (embedding_cli.values.eigen_projection
            && std::find(projection_spaces.begin(), projection_spaces.end(),
                ProjectionSpace::Ilr) != projection_spaces.end()
            && (!(embedding_cli.values.center_floor > 0.0)
                || !std::isfinite(embedding_cli.values.center_floor))) {
            throw std::invalid_argument(
                "--center-floor must be positive and finite");
        }
        parameters.print_options();
        const punkst::projection::VisualizationWhitening whitening =
            embedding_cli.values.eigen_projection
            ? punkst::projection::parse_visualization_whitening(
                embedding_cli.whitening)
            : embedding_cli.values.whitening;
        const std::vector<std::string> labels = resolve_partition_labels(
            partition_columns, partition_labels);
        punkst_cli::TopicCenterTable theta = punkst_cli::read_topic_centers(
            theta_path, embedding_cli.values.center_floor,
            theta_identifier_column, nullptr,
            "--theta-icol-id", false, factor_column_start,
            factor_column_end);
        filter_factors_by_relative_weight(theta, factor_weight_threshold);
        const PartitionTable partitions = read_partitions(
            partition_path, partition_identifier_column, partition_columns);
        const std::vector<int32_t> matched_rows = match_partition_rows(
            theta, partitions, id_as_row_index);
        const int32_t matched_documents = static_cast<int32_t>(std::count_if(
            matched_rows.begin(), matched_rows.end(),
            [](int32_t row) { return row >= 0; }));
        if (matched_documents != static_cast<int32_t>(theta.identifiers.size())
            || matched_documents
                != static_cast<int32_t>(partitions.identifiers.size())) {
            warning("Linear embedding input mismatch: theta units %zu; partition units %zu; intersection %d",
                theta.identifiers.size(), partitions.identifiers.size(),
                matched_documents);
        }
        std::vector<int32_t> matched_partition_rows;
        std::vector<int32_t> matched_theta_rows;
        matched_partition_rows.reserve(static_cast<size_t>(matched_documents));
        matched_theta_rows.reserve(static_cast<size_t>(matched_documents));
        for (int32_t partition_row = 0;
                partition_row < static_cast<int32_t>(matched_rows.size());
                ++partition_row) {
            if (matched_rows[static_cast<size_t>(partition_row)] < 0) continue;
            matched_partition_rows.push_back(partition_row);
            matched_theta_rows.push_back(
                matched_rows[static_cast<size_t>(partition_row)]);
        }
        punkst::linear_embedding::Options pipeline = embedding_cli.values;
        pipeline.projection_spaces = projection_spaces;
        pipeline.whitening = whitening;
        pipeline.dimensions = requested_dimensions;
        pipeline.threads = threads;
        pipeline.center_floor = embedding_cli.values.center_floor;
        pipeline.covariance_floor = embedding_cli.values.covariance_floor;
        pipeline.include_full = embedding_cli.include_full;
        pipeline.validate();
        for (size_t partition = 0;
                partition < partition_columns.size(); ++partition) {
            std::unordered_map<std::string, int32_t> component_index;
            std::vector<std::string> component_labels;
            Eigen::VectorXi assignments(matched_documents);
            for (int32_t document = 0; document < matched_documents;
                    ++document) {
                const std::string& value = partitions.values[partition][
                    static_cast<size_t>(matched_partition_rows[
                        static_cast<size_t>(document)])];
                const auto inserted = component_index.emplace(
                    value, static_cast<int32_t>(component_index.size()));
                if (inserted.second) component_labels.push_back(value);
                assignments(document) = inserted.first->second;
            }
            const int32_t components =
                static_cast<int32_t>(component_index.size());
            if (components < 2) {
                throw std::runtime_error(
                    "Linear embedding requires at least two clusters");
            }
            const std::string prefix = partition_columns.size() == 1
                ? output_prefix : output_prefix + "." + labels[partition];
            write_cluster_labels(prefix + ".cluster_labels.tsv",
                component_labels);
            const Eigen::MatrixXd cluster_factors =
                punkst::linear_embedding::aggregate_cluster_factors(
                    theta.values, matched_theta_rows, assignments,
                    components);
            punkst::linear_embedding::write_cluster_factors(
                prefix + ".cluster_factor_abundance.tsv", theta.topics,
                cluster_factors);
            punkst::linear_embedding::run_partition(theta,
                matched_theta_rows, assignments, components,
                labels[partition], prefix, pipeline);
        }
    } catch (const std::exception& exception) {
        std::cerr << "Linear embedding failed: "
            << exception.what() << "\n";
        return 1;
    }
    return 0;
}
