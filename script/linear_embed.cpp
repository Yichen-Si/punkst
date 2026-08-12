#include "clustering/uac.hpp"
#include "punkst.h"
#include "cli_common.hpp"
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

using punkst_cli::ProjectionSpace;
using punkst_cli::projection_space_name;

struct PartitionTable {
    std::vector<std::string> identifiers;
    std::vector<std::vector<std::string>> values;
};

struct ProjectionOutput {
    ProjectionSpace space = ProjectionSpace::Linear;
    const uac::Dataset* data = nullptr;
    uac::VisualizationResult visualization;
};

struct ProjectionInput {
    ProjectionSpace space = ProjectionSpace::Linear;
    uac::Dataset data;
    RowMajorMatrixXd matched_coordinates;
    std::optional<uac::VisualizationSampleMoments> sample_moments;
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
        throw std::runtime_error("Cannot write embedding axis weights: " + path);
    }
    if (projection.topic_contrasts.rows()
            != static_cast<Eigen::Index>(topics.size())) {
        throw std::invalid_argument("Invalid embedding axis-weight dimensions");
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
                "Embedding contrast has no positive mass");
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
    const std::vector<std::string>& identifiers,
    const std::vector<ProjectionOutput>& projections) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Cannot write linear embedding coordinates: " + path);
    }
    out << "#id";
    for (const ProjectionOutput& projection : projections) {
        const char* space = projection_space_name(projection.space);
        for (Eigen::Index axis = 0;
                axis < projection.visualization.mean.projection.cols();
                ++axis) {
            out << '\t' << space << "_mean_" << axis + 1;
        }
        for (Eigen::Index axis = 0;
                axis < projection.visualization.full.projection.cols();
                ++axis) {
            out << '\t' << space << "_full_" << axis + 1;
        }
    }
    out << "\n" << std::scientific << std::setprecision(4);
    for (Eigen::Index document = 0;
            document < static_cast<Eigen::Index>(identifiers.size());
            ++document) {
        out << identifiers[static_cast<size_t>(document)];
        for (const ProjectionOutput& projection : projections) {
            if (projection.data == nullptr
                || projection.data->coordinates.rows()
                    != static_cast<Eigen::Index>(identifiers.size())) {
                throw std::invalid_argument(
                    "Invalid embedding coordinate dimensions");
            }
            for (Eigen::Index axis = 0;
                    axis < projection.visualization.mean.projection.cols();
                    ++axis) {
                out << '\t' << projection.data->coordinates.row(document).dot(
                    projection.visualization.mean.projection.col(axis));
            }
            for (Eigen::Index axis = 0;
                    axis < projection.visualization.full.projection.cols();
                    ++axis) {
                out << '\t' << projection.data->coordinates.row(document).dot(
                    projection.visualization.full.projection.col(axis));
            }
        }
        out << '\n';
    }
}

} // namespace

int32_t cmdLinearEmbed(int argc, char** argv) {
    std::string theta_path, partition_path, output_prefix;
    std::string whitening_name = "mixture";
    std::string projection_space_name_option = "both";
    std::vector<int32_t> partition_columns;
    std::vector<std::string> partition_labels;
    int32_t theta_identifier_column = 0;
    int32_t partition_identifier_column = 0;
    int32_t dim = -1, visual_dim = -1;
    int32_t threads = 1;
    double center_floor = 1e-12;
    double covariance_floor = 1e-5;
    bool id_as_row_index = false;
    bool visual_full = false;

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
      .add_option("projection-space",
          "Projection coordinates: both, linear, or ilr",
          projection_space_name_option)
      .add_option("center-floor",
          "Positive factor floor for the ILR projection",
          center_floor)
      .add_option("covariance-floor",
          "Positive eigenvalue floor for the whitening covariance",
          covariance_floor)
      .add_option("visual-full",
          "Also compute the full visualization using covariance-shape differences",
          visual_full)
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
        if (!(covariance_floor > 0.0)
            || !std::isfinite(covariance_floor)) {
            throw std::invalid_argument(
                "--covariance-floor must be positive and finite");
        }
        const std::vector<ProjectionSpace> projection_spaces =
            punkst_cli::parse_projection_spaces(
                projection_space_name_option);
        if (std::find(projection_spaces.begin(), projection_spaces.end(),
                ProjectionSpace::Ilr) != projection_spaces.end()
            && (!(center_floor > 0.0) || !std::isfinite(center_floor))) {
            throw std::invalid_argument(
                "--center-floor must be positive and finite");
        }
        parameters.print_options();
        const uac::VisualizationWhitening whitening =
            uac::parse_visualization_whitening(whitening_name);
        const std::vector<std::string> labels = resolve_partition_labels(
            partition_columns, partition_labels);
        punkst_cli::TopicCenterTable theta = punkst_cli::read_topic_centers(
            theta_path, center_floor, theta_identifier_column, nullptr,
            "--theta-icol-id", false);
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
        const int32_t topics = static_cast<int32_t>(theta.values.cols());
        if (matched_documents < topics) {
            throw std::runtime_error(
                "Linear embedding intersection is too small: requires at least "
                + std::to_string(topics) + " matched units");
        }

        const Eigen::MatrixXd helmert = normalized_helmert(topics);
        std::vector<int32_t> matched_partition_rows;
        matched_partition_rows.reserve(static_cast<size_t>(matched_documents));
        for (int32_t partition_row = 0;
                partition_row < static_cast<int32_t>(matched_rows.size());
                ++partition_row) {
            if (matched_rows[static_cast<size_t>(partition_row)] < 0) continue;
            matched_partition_rows.push_back(partition_row);
        }

        std::vector<ProjectionInput> inputs;
        inputs.reserve(projection_spaces.size());
        for (const ProjectionSpace space : projection_spaces) {
            ProjectionInput input;
            input.space = space;
            input.data.identifiers = theta.identifiers;
            punkst_cli::ProjectionData projection =
                punkst_cli::prepare_projection(theta.values,
                    space, helmert, center_floor);
            input.data.centers = std::move(projection.centers);
            input.data.coordinates = std::move(projection.coordinates);
            input.matched_coordinates.resize(
                matched_documents, input.data.coordinates.cols());
            for (int32_t matched = 0; matched < matched_documents; ++matched) {
                const int32_t partition_row =
                    matched_partition_rows[static_cast<size_t>(matched)];
                input.matched_coordinates.row(matched) =
                    input.data.coordinates.row(matched_rows[
                        static_cast<size_t>(partition_row)]);
            }
            if (whitening == uac::VisualizationWhitening::Sample) {
                input.sample_moments = uac::summarize_visualization_sample(
                    input.data.coordinates, threads);
            }
            inputs.push_back(std::move(input));
        }

        uac::VisualizationOptions options;
        options.whitening = whitening;
        options.dimensions = std::min(
            requested_dimensions, topics - 1);
        options.n_threads = threads;
        options.covariance_floor = covariance_floor;
        options.include_full = visual_full;
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
                    "Linear embedding requires at least two clusters");
            }
            const std::string prefix = partition_columns.size() == 1
                ? output_prefix : output_prefix + "." + labels[partition];
            std::vector<ProjectionOutput> projections;
            projections.reserve(inputs.size());
            for (ProjectionInput& input : inputs) {
                uac::VisualizationResult visualization;
                if (visual_full) {
                    const uac::VisualizationMoments moments =
                        uac::summarize_hard_partition(
                            input.matched_coordinates, assignments,
                            components, threads);
                    visualization = input.sample_moments
                        ? uac::make_visualization(input.data, moments, helmert,
                            options, *input.sample_moments)
                        : uac::make_visualization(
                            input.data, moments, helmert, options);
                } else {
                    const uac::VisualizationMeans means =
                        uac::summarize_hard_partition_means(
                            input.matched_coordinates, assignments,
                            components);
                    uac::VisualizationOptions mean_options = options;
                    mean_options.dimensions = std::min(
                        mean_options.dimensions, components - 1);
                    const uac::VisualizationSampleMoments whitening_moments =
                        input.sample_moments ? *input.sample_moments
                        : uac::summarize_visualization_sample(
                            input.matched_coordinates, threads);
                    visualization = uac::make_mean_visualization(
                        means, helmert, mean_options, whitening_moments);
                }
                const int32_t mean_dimensions = visual_full
                    ? std::min({options.dimensions, components - 1,
                        positive_rank(visualization.mean.eigenvalues)})
                    : static_cast<int32_t>(
                        visualization.mean.projection.cols());
                if (mean_dimensions <= 0) {
                    throw std::runtime_error(
                        "Hard partition has no positive mean-separation axis");
                }
                if (visual_full) {
                    truncate_projection(
                        visualization.mean, mean_dimensions);
                }
                int32_t full_dimensions = 0;
                if (visual_full) {
                    full_dimensions = std::min(options.dimensions,
                        positive_rank(visualization.full.eigenvalues));
                    if (full_dimensions <= 0) {
                        throw std::runtime_error(
                            "Hard partition has no positive full-separation axis");
                    }
                    truncate_projection(
                        visualization.full, full_dimensions);
                }
                if (mean_dimensions < requested_dimensions
                    || (visual_full
                        && full_dimensions < requested_dimensions)) {
                    if (visual_full) {
                        notice("Linear embedding partition %s %s-space resolved dimensions: mean %d, full %d (requested %d)",
                            labels[partition].c_str(),
                            projection_space_name(input.space),
                            mean_dimensions, full_dimensions,
                            requested_dimensions);
                    } else {
                        notice("Linear embedding partition %s %s-space resolved dimensions: mean %d (requested %d)",
                            labels[partition].c_str(),
                            projection_space_name(input.space),
                            mean_dimensions, requested_dimensions);
                    }
                }
                const std::string space_prefix = prefix + "."
                    + projection_space_name(input.space);
                uac::write_visualization_axes(
                    space_prefix + ".transform.tsv", theta.topics,
                    visualization);
                write_axis_weights(space_prefix + ".mean.axes.tsv",
                    theta.topics, visualization.mean);
                if (visual_full) {
                    write_axis_weights(space_prefix + ".full.axes.tsv",
                        theta.topics, visualization.full);
                }
                projections.push_back(
                    {input.space, &input.data, std::move(visualization)});
            }
            write_coordinates(
                prefix + ".results.tsv", theta.identifiers, projections);
            notice("Linear embedding wrote %zu projection space(s) under %s and coordinates to %s.results.tsv",
                projections.size(), prefix.c_str(), prefix.c_str());
        }
    } catch (const std::exception& exception) {
        std::cerr << "Linear embedding failed: "
            << exception.what() << "\n";
        return 1;
    }
    return 0;
}
