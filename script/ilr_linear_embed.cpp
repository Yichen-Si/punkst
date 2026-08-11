#include "clustering/uac.hpp"
#include "punkst.h"
#include "uac_cli_common.hpp"
#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace {

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
    const uac_cli::TopicCenterTable& theta,
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

void truncate_projection(uac::VisualizationProjection& projection,
        int32_t dimensions) {
    projection.eigenvalues.conservativeResize(dimensions);
    projection.projection.conservativeResize(
        projection.projection.rows(), dimensions);
    projection.topic_contrasts.conservativeResize(
        projection.topic_contrasts.rows(), dimensions);
    projection.component_means.conservativeResize(
        projection.component_means.rows(), dimensions);
    for (Eigen::MatrixXd& covariance : projection.component_covariances) {
        covariance = covariance.topLeftCorner(dimensions, dimensions).eval();
    }
}

int32_t positive_rank(const Eigen::Ref<const Eigen::VectorXd>& eigenvalues) {
    if (eigenvalues.size() == 0) return 0;
    const double scale = std::max(1.0,
        eigenvalues.cwiseAbs().maxCoeff());
    const double tolerance = 256.0
        * std::numeric_limits<double>::epsilon() * scale;
    int32_t rank = 0;
    for (Eigen::Index axis = 0; axis < eigenvalues.size(); ++axis) {
        if (eigenvalues(axis) > tolerance) ++rank;
    }
    return rank;
}

void write_axis_weights(const std::string& path,
    const std::vector<std::string>& topics,
    const uac::VisualizationProjection& projection) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write ILR axis weights: " + path);
    }
    if (projection.topic_contrasts.rows()
            != static_cast<Eigen::Index>(topics.size())) {
        throw std::invalid_argument("Invalid ILR axis-weight dimensions");
    }
    out << "#Factor";
    for (Eigen::Index axis = 0;
            axis < projection.topic_contrasts.cols(); ++axis) {
        out << "\tw" << axis + 1 << "_p\tw" << axis + 1 << "_n";
    }
    out << "\n" << std::scientific << std::setprecision(10);
    Eigen::VectorXd scales(projection.topic_contrasts.cols());
    for (Eigen::Index axis = 0;
            axis < projection.topic_contrasts.cols(); ++axis) {
        scales(axis) = projection.topic_contrasts.col(axis)
            .cwiseMax(0.0).sum();
        if (!(scales(axis) > 0.0) || !std::isfinite(scales(axis))) {
            throw std::runtime_error(
                "ILR visualization contrast has no positive mass");
        }
    }
    for (Eigen::Index topic = 0;
            topic < projection.topic_contrasts.rows(); ++topic) {
        out << topics[static_cast<size_t>(topic)];
        for (Eigen::Index axis = 0;
                axis < projection.topic_contrasts.cols(); ++axis) {
            const double coefficient =
                projection.topic_contrasts(topic, axis);
            out << '\t' << std::max(0.0, coefficient) / scales(axis)
                << '\t' << std::max(0.0, -coefficient) / scales(axis);
        }
        out << '\n';
    }
}

void write_coordinates(const std::string& path,
    const uac::Dataset& data,
    const uac::VisualizationResult& visualization) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Cannot write ILR embedding coordinates: " + path);
    }
    out << "#id";
    for (Eigen::Index axis = 0;
            axis < visualization.mean.projection.cols(); ++axis) {
        out << "\tmean_" << axis + 1;
    }
    for (Eigen::Index axis = 0;
            axis < visualization.full.projection.cols(); ++axis) {
        out << "\tfull_" << axis + 1;
    }
    out << "\n" << std::scientific << std::setprecision(4);
    for (Eigen::Index document = 0;
            document < data.coordinates.rows(); ++document) {
        out << data.identifiers[static_cast<size_t>(document)];
        for (Eigen::Index axis = 0;
                axis < visualization.mean.projection.cols(); ++axis) {
            out << '\t' << data.coordinates.row(document).dot(
                visualization.mean.projection.col(axis));
        }
        for (Eigen::Index axis = 0;
                axis < visualization.full.projection.cols(); ++axis) {
            out << '\t' << data.coordinates.row(document).dot(
                visualization.full.projection.col(axis));
        }
        out << '\n';
    }
}

} // namespace

int32_t cmdIlrLinearEmbed(int argc, char** argv) {
    std::string theta_path, partition_path, output_prefix;
    std::string whitening_name = "mixture";
    std::vector<int32_t> partition_columns;
    std::vector<std::string> partition_labels;
    int32_t theta_identifier_column = 0;
    int32_t partition_identifier_column = 0;
    int32_t dim = -1, visual_dim = -1;
    int32_t threads = 1;
    double center_floor = 1e-12;
    double covariance_floor = 1e-5;
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
          whitening_name)
      .add_option("center-floor",
          "Positive floor applied to theta before row normalization",
          center_floor)
      .add_option("covariance-floor",
          "Positive eigenvalue floor for the whitening covariance",
          covariance_floor)
      .add_option("threads", "Number of covariance worker threads", threads);

    try {
        parameters.readArgs(argc, argv);
        if (partition_columns.empty()) partition_columns.push_back(1);
        const bool has_dim = parameters.was_provided("dim");
        const bool has_visual_dim = parameters.was_provided("visual-dim");
        if (has_dim && has_visual_dim && dim != visual_dim) {
            throw std::invalid_argument(
                "--dim and --visual-dim must agree when both are supplied");
        }
        const int32_t requested_dimensions = has_dim ? dim
            : has_visual_dim ? visual_dim : 2;
        if (requested_dimensions <= 0) {
            throw std::invalid_argument(
                "--dim/--visual-dim must be positive");
        }
        if (threads <= 0) {
            throw std::invalid_argument("--threads must be positive");
        }
        if (!(center_floor > 0.0) || !std::isfinite(center_floor)) {
            throw std::invalid_argument(
                "--center-floor must be positive and finite");
        }
        if (!(covariance_floor > 0.0)
            || !std::isfinite(covariance_floor)) {
            throw std::invalid_argument(
                "--covariance-floor must be positive and finite");
        }
        parameters.print_options();

        const uac::VisualizationWhitening whitening =
            uac::parse_visualization_whitening(whitening_name);
        const std::vector<std::string> labels = resolve_partition_labels(
            partition_columns, partition_labels);
        uac_cli::TopicCenterTable theta = uac_cli::read_topic_centers(
            theta_path, center_floor, theta_identifier_column, nullptr,
            "--theta-icol-id");
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
            warning("ILR linear embedding input mismatch: theta units %zu; partition units %zu; intersection %d",
                theta.identifiers.size(), partitions.identifiers.size(),
                matched_documents);
        }
        const int32_t topics = static_cast<int32_t>(theta.values.cols());
        if (matched_documents < topics) {
            throw std::runtime_error(
                "ILR linear embedding intersection is too small: requires at least "
                + std::to_string(topics) + " matched units");
        }

        const Eigen::MatrixXd helmert = normalized_helmert(topics);
        uac::Dataset all_data;
        all_data.identifiers = theta.identifiers;
        all_data.centers = theta.values;
        all_data.coordinates = ilr_transform(theta.values, helmert);
        std::optional<uac::VisualizationSampleMoments> sample_moments;
        if (whitening == uac::VisualizationWhitening::Sample) {
            sample_moments = uac::summarize_visualization_sample(
                all_data.coordinates, threads);
        }
        RowMajorMatrixXd matched_coordinates(
            matched_documents, all_data.coordinates.cols());
        std::vector<int32_t> matched_partition_rows;
        matched_partition_rows.reserve(static_cast<size_t>(matched_documents));
        int32_t matched_index = 0;
        for (int32_t partition_row = 0;
                partition_row < static_cast<int32_t>(matched_rows.size());
                ++partition_row) {
            if (matched_rows[static_cast<size_t>(partition_row)] < 0) continue;
            matched_coordinates.row(matched_index) = all_data.coordinates.row(
                matched_rows[static_cast<size_t>(partition_row)]);
            matched_partition_rows.push_back(partition_row);
            ++matched_index;
        }

        uac::VisualizationOptions options;
        options.whitening = whitening;
        options.dimensions = std::min(
            requested_dimensions, topics - 1);
        options.n_threads = threads;
        options.covariance_floor = covariance_floor;
        for (size_t partition = 0;
                partition < partition_columns.size(); ++partition) {
            std::unordered_map<std::string, int32_t> component_index;
            Eigen::VectorXi assignments(matched_documents);
            for (int32_t document = 0; document < matched_documents;
                    ++document) {
                const std::string& value = partitions.values[partition][
                    static_cast<size_t>(matched_partition_rows[
                        static_cast<size_t>(document)])];
                const auto inserted = component_index.emplace(
                    value, static_cast<int32_t>(component_index.size()));
                assignments(document) = inserted.first->second;
            }
            const int32_t components =
                static_cast<int32_t>(component_index.size());
            if (components < 2) {
                throw std::runtime_error(
                    "ILR linear embedding requires at least two clusters");
            }
            const uac::VisualizationMoments moments =
                uac::summarize_hard_partition(
                    matched_coordinates, assignments, components, threads);
            uac::VisualizationResult visualization = sample_moments
                ? uac::make_visualization(all_data, moments, helmert, options,
                    *sample_moments)
                : uac::make_visualization(
                    all_data, moments, helmert, options);
            const int32_t mean_dimensions = std::min({
                options.dimensions, components - 1,
                positive_rank(visualization.mean.eigenvalues)});
            const int32_t full_dimensions = std::min(
                options.dimensions,
                positive_rank(visualization.full.eigenvalues));
            if (mean_dimensions <= 0) {
                throw std::runtime_error(
                    "Hard partition has no positive mean-separation axis");
            }
            if (full_dimensions <= 0) {
                throw std::runtime_error(
                    "Hard partition has no positive full-separation axis");
            }
            truncate_projection(visualization.mean, mean_dimensions);
            truncate_projection(visualization.full, full_dimensions);
            if (mean_dimensions < requested_dimensions
                || full_dimensions < requested_dimensions) {
                notice("ILR linear embedding partition %s resolved dimensions: mean %d, full %d (requested %d)",
                    labels[partition].c_str(), mean_dimensions,
                    full_dimensions, requested_dimensions);
            }

            const std::string prefix = partition_columns.size() == 1
                ? output_prefix : output_prefix + "." + labels[partition];
            uac::write_visualization_axes(
                prefix + ".transform.tsv", theta.topics, visualization);
            write_axis_weights(prefix + ".mean.axes.tsv", theta.topics,
                visualization.mean);
            write_axis_weights(prefix + ".full.axes.tsv", theta.topics,
                visualization.full);
            write_coordinates(prefix + ".results.tsv", all_data,
                visualization);
            notice("ILR linear embedding outputs written to %s.{transform,mean.axes,full.axes,results}.tsv",
                prefix.c_str());
        }
    } catch (const std::exception& exception) {
        std::cerr << "ILR linear embedding failed: "
            << exception.what() << "\n";
        return 1;
    }
    return 0;
}
