#pragma once

#include "clustering/uac.hpp"

#include <string>
#include <vector>

namespace uac_cli {

struct TopicCenterTable {
    std::vector<std::string> identifiers;
    std::vector<std::string> topics;
    RowMajorMatrixXd values;
};

TopicCenterTable read_topic_centers(const std::string& path, double floor,
    int32_t identifier_column,
    const std::vector<std::string>* expected_topics = nullptr,
    const std::string& identifier_option = "--unit-icol-id");

} // namespace uac_cli
