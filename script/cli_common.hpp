#pragma once

#include "dataunits.hpp"
#include "linear_embedding.hpp"
#include "numerical_utils.hpp"

#include <string>
#include <vector>

namespace punkst_cli {

using ProjectionSpace = punkst::linear_embedding::ProjectionSpace;
using TopicCenterTable = punkst::linear_embedding::TopicCenterTable;

TopicCenterTable read_topic_centers(const std::string& path, double floor,
    int32_t identifier_column,
    const std::vector<std::string>* expected_topics = nullptr,
    const std::string& identifier_option = "--unit-icol-id",
    bool normalize = true, int32_t factor_column_start = -1,
    int32_t factor_column_end = -1);

std::vector<ProjectionSpace> parse_projection_spaces(
    const std::string& value);

} // namespace punkst_cli
