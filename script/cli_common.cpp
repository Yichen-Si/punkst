#include "cli_common.hpp"

#include "utils.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace punkst_cli {

namespace {

RowMajorMatrixXd normalize_factor_proportions(
        const Eigen::Ref<const RowMajorMatrixXd>& values) {
    RowMajorMatrixXd out = values;
    for (Eigen::Index row = 0; row < out.rows(); ++row) {
        const double scale = out.row(row).maxCoeff();
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::invalid_argument(
                "Projection requires positive finite factor rows");
        }
        out.row(row) /= scale;
        const double total = out.row(row).sum();
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::invalid_argument(
                "Projection requires positive finite factor rows");
        }
        out.row(row) /= total;
    }
    return out;
}

} // namespace

const char* projection_space_name(ProjectionSpace space) {
    return space == ProjectionSpace::Linear ? "linear" : "ilr";
}

std::vector<ProjectionSpace> parse_projection_spaces(
        const std::string& value) {
    if (value == "linear") return {ProjectionSpace::Linear};
    if (value == "ilr") return {ProjectionSpace::Ilr};
    if (value == "both") {
        return {ProjectionSpace::Linear, ProjectionSpace::Ilr};
    }
    throw std::invalid_argument(
        "--projection-space must be both, linear, or ilr");
}

ProjectionData prepare_projection(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    ProjectionSpace space,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    double center_floor) {
    ProjectionData out;
    if (space == ProjectionSpace::Linear) {
        out.centers = normalize_factor_proportions(values);
        out.coordinates = out.centers * helmert.transpose();
    } else {
        out.centers = values;
        normalize_compositions(out.centers, center_floor);
        out.coordinates = ilr_transform(out.centers, helmert);
    }
    return out;
}

TopicCenterTable read_topic_centers(const std::string& path, double floor,
    int32_t identifier_column,
    const std::vector<std::string>* expected_topics,
    const std::string& identifier_option, bool normalize,
    int32_t factor_column_start, int32_t factor_column_end) {
    if (identifier_column < 0) {
        throw std::invalid_argument(
            identifier_option + " must be nonnegative");
    }
    if (factor_column_start < -1 || factor_column_end < -1) {
        throw std::invalid_argument(
            "factor column indices must be nonnegative");
    }
    const bool explicit_factor_columns = factor_column_start >= 0;
    if (explicit_factor_columns != (factor_column_end >= 0)) {
        throw std::invalid_argument(
            "--icol-factor-start and --icol-factor-end must be supplied together");
    }
    if (explicit_factor_columns && expected_topics != nullptr) {
        throw std::invalid_argument(
            "Explicit factor columns cannot be combined with expected topics");
    }
    if (explicit_factor_columns && factor_column_end < factor_column_start) {
        throw std::invalid_argument(
            "--icol-factor-end must not precede --icol-factor-start");
    }
    if (explicit_factor_columns
            && factor_column_end - factor_column_start + 1 < 2) {
        throw std::invalid_argument(
            "Linear embedding requires at least two factor columns");
    }
    TextLineReader reader(path);
    std::string line;
    while (reader.getline(line) && line.empty()) {}
    if (line.empty()) {
        throw std::runtime_error(
            "Topic-center table is empty: " + path);
    }
    const std::vector<std::string> header =
        split_delimited(strip_leading_hash(line), '\t');
    if (identifier_column >= static_cast<int32_t>(header.size())) {
        throw std::runtime_error(
            identifier_option + " is outside the topic-center table");
    }
    if (explicit_factor_columns
            && factor_column_end >= static_cast<int32_t>(header.size())) {
        throw std::invalid_argument(
            "--icol-factor-end is outside the topic-center table");
    }
    if (explicit_factor_columns
            && identifier_column >= factor_column_start
            && identifier_column <= factor_column_end) {
        throw std::invalid_argument(
            identifier_option + " must select a non-factor column");
    }
    std::unordered_map<std::string, int32_t> header_index;
    for (int32_t i = 0; i < static_cast<int32_t>(header.size()); ++i) {
        if (header[i].empty()
            || !header_index.emplace(header[i], i).second) {
            throw std::runtime_error(
                "Empty or duplicate topic-center header: " + header[i]);
        }
        if (header[i] == "Background") {
            throw std::runtime_error(
                "Background-enabled LDA output is not a topic center");
        }
    }
    UnitFactorResultReadOptions factor_options;
    factor_options.xColName.clear();
    factor_options.yColName.clear();
    factor_options.topKColName.clear();
    factor_options.topPColName.clear();
    factor_options.requireFactorValues = false;
    std::vector<int32_t> topic_columns;
    TopicCenterTable table;
    if (explicit_factor_columns) {
        topic_columns.reserve(static_cast<size_t>(
            factor_column_end - factor_column_start + 1));
        table.topics.reserve(topic_columns.capacity());
        for (int32_t column = factor_column_start;
                column <= factor_column_end; ++column) {
            topic_columns.push_back(column);
            table.topics.push_back(header[static_cast<size_t>(column)]);
        }
    } else if (expected_topics) {
        table.topics = *expected_topics;
        topic_columns.reserve(expected_topics->size());
        for (const std::string& topic : *expected_topics) {
            const auto found = header_index.find(topic);
            if (found == header_index.end()) {
                throw std::runtime_error(
                    "Topic-center table is missing topic: " + topic);
            }
            topic_columns.push_back(found->second);
        }
    } else {
        const UnitFactorResultHeader factor_header =
            parse_unit_factor_result_header(header, factor_options);
        if (factor_header.hasTopPairs()) {
            throw std::runtime_error(
                "LDA K/P top-k output is not a dense topic center");
        }
        if (factor_header.factorCols.empty()) {
            throw std::runtime_error(
                "Topic columns must have trailing headers 0..K-1");
        }
        topic_columns.reserve(factor_header.factorCols.size());
        table.topics.reserve(factor_header.factorCols.size());
        const int32_t first_topic = static_cast<int32_t>(header.size()
            - factor_header.factorCols.size());
        for (size_t i = 0; i < factor_header.factorCols.size(); ++i) {
            const int32_t column = factor_header.factorCols[i].second;
            if (column != first_topic + static_cast<int32_t>(i)) {
                throw std::runtime_error(
                    "Topic columns must be the trailing 0..K-1 block");
            }
            topic_columns.push_back(column);
            table.topics.push_back(header[column]);
        }
    }
    if (topic_columns.size() < 2
        || std::find(topic_columns.begin(), topic_columns.end(),
            identifier_column) != topic_columns.end()) {
        throw std::runtime_error(
            identifier_option + " must select a non-topic column");
    }

    std::vector<double> values;
    std::unordered_set<std::string> seen;
    uint64_t input_row = 1;
    while (reader.getline(line)) {
        ++input_row;
        if (line.empty() || is_comment_line(line)) continue;
        const std::vector<std::string> fields =
            split_delimited(line, '\t');
        if (fields.size() != header.size()) {
            throw std::runtime_error(
                "Topic-center row has the wrong column count at line "
                + std::to_string(input_row));
        }
        const std::string& identifier = fields[identifier_column];
        if (identifier.empty() || !seen.insert(identifier).second) {
            throw std::runtime_error(
                "Empty or duplicate topic-center identifier: " + identifier);
        }
        table.identifiers.push_back(identifier);
        for (const int32_t column : topic_columns) {
            double value = 0.0;
            if (!str2double(fields[column], value) || value < 0.0
                || !std::isfinite(value)) {
                throw std::runtime_error(
                    "Invalid topic probability at line "
                    + std::to_string(input_row));
            }
            values.push_back(value);
        }
    }
    if (table.identifiers.empty()) {
        throw std::runtime_error(
            "Topic-center table has no data rows");
    }
    table.values.resize(table.identifiers.size(), topic_columns.size());
    for (Eigen::Index row = 0; row < table.values.rows(); ++row) {
        for (Eigen::Index column = 0; column < table.values.cols(); ++column) {
            table.values(row, column) = values[
                static_cast<size_t>(row * table.values.cols() + column)];
        }
    }
    if (normalize) normalize_compositions(table.values, floor);
    return table;
}

} // namespace punkst_cli
