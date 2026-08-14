#pragma once

#include "numerical_utils.hpp"

#include <string>
#include <vector>

struct LdaState {
    static constexpr int32_t SCHEMA_VERSION = 1;

    double alpha = -1.0;
    double eta = -1.0;
    std::vector<std::string> topics;
    std::vector<std::string> features;
    RowMajorMatrixXd components;
    bool feature_weights_active = false;
    std::vector<double> feature_weights;

    void validate() const;
    void write(const std::string& path) const;
    static LdaState read(const std::string& path);
};
