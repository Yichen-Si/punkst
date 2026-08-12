#pragma once

#include "clustering/uac.hpp"

#include <string>
#include <vector>

namespace punkst_cli {

enum class ProjectionSpace {
    Linear,
    Ilr,
};

struct ProjectionData {
    RowMajorMatrixXd centers;
    RowMajorMatrixXd coordinates;
};

struct TopicCenterTable {
    std::vector<std::string> identifiers;
    std::vector<std::string> topics;
    RowMajorMatrixXd values;
};

TopicCenterTable read_topic_centers(const std::string& path, double floor,
    int32_t identifier_column,
    const std::vector<std::string>* expected_topics = nullptr,
    const std::string& identifier_option = "--unit-icol-id",
    bool normalize = true);

const char* projection_space_name(ProjectionSpace space);
std::vector<ProjectionSpace> parse_projection_spaces(
    const std::string& value);
ProjectionData prepare_projection(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    ProjectionSpace space,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    double center_floor);

} // namespace punkst_cli
