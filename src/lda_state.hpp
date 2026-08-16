#pragma once

#include "numerical_utils.hpp"

#include <string>
#include <vector>

struct LdaState {
    static constexpr int32_t SCHEMA_VERSION = 2;

    double alpha = -1.0;
    double eta = -1.0;
    std::vector<std::string> topics;
    std::vector<std::string> features;
    RowMajorMatrixXd components;
    bool feature_weights_active = false;
    std::vector<double> feature_weights;

    bool has_background = false;
    bool background_fixed = false;
    double background_prior_a = -1.0;
    double background_prior_b = -1.0;
    double background_count = 0.0;
    double foreground_count = 0.0;
    Eigen::VectorXd background_prior;
    Eigen::VectorXd background_components;

    void validate() const;
    void write(const std::string& path) const;
    static LdaState read(const std::string& path);
};
