#pragma once

#include "numerical_utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace punkst::partition_classifier {

constexpr int32_t MODEL_SCHEMA_VERSION = 1;

struct FitOptions {
    std::vector<double> ridge_grid{
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0};
    int32_t folds = 5;
    int32_t max_iterations = 300;
    int32_t lbfgs_history = 10;
    double gradient_tolerance = 1e-7;
};

struct Metrics {
    double log_loss = 0.0;
    double brier = 0.0;
    double accuracy = 0.0;
    double weight = 0.0;
};

struct CvResult {
    double ridge = 0.0;
    Metrics metrics;
    bool selected = false;
};

struct CalibrationBin {
    int32_t bin = 0;
    double lower = 0.0;
    double upper = 0.0;
    double weight = 0.0;
    double mean_confidence = 0.0;
    double accuracy = 0.0;
};

struct CalibrationResult {
    Metrics cross_fitted;
    std::vector<Metrics> classwise;
    std::vector<CalibrationBin> bins;
    double stored_temperature = 1.0;
};

struct FitResult;

class Model {
public:
    std::vector<std::string> topics;
    std::vector<std::string> classes;
    Eigen::VectorXd intercepts;
    RowMajorMatrixXd coefficients;
    double ridge = 0.0;
    double temperature = 1.0;
    uint64_t matched_rows = 0;
    uint64_t sampled_rows = 0;
    int32_t folds = 0;
    int32_t minimum_per_class = 0;
    uint64_t sampling_seed = 1;

    Eigen::VectorXd logits(
        const Eigen::Ref<const Eigen::VectorXd>& composition) const;
    Eigen::VectorXd probabilities(
        const Eigen::Ref<const Eigen::VectorXd>& composition) const;
    void validate(double tolerance = 1e-8) const;
    void write(const std::string& path) const;
    static Model read(const std::string& path);
};

struct FitResult {
    Model model;
    std::vector<CvResult> cv;
    CalibrationResult calibration;
    RowMajorMatrixXd oof_logits;
    RowMajorMatrixXd cross_fitted_probabilities;
};

FitResult fit(const Eigen::Ref<const RowMajorMatrixXd>& compositions,
    const Eigen::Ref<const Eigen::VectorXi>& labels,
    const Eigen::Ref<const Eigen::VectorXd>& weights,
    const std::vector<std::string>& topics,
    const std::vector<std::string>& classes,
    const FitOptions& options = {});

Metrics evaluate(const Eigen::Ref<const RowMajorMatrixXd>& probabilities,
    const Eigen::Ref<const Eigen::VectorXi>& labels,
    const Eigen::Ref<const Eigen::VectorXd>& weights);

double fit_temperature(const Eigen::Ref<const RowMajorMatrixXd>& logits,
    const Eigen::Ref<const Eigen::VectorXi>& labels,
    const Eigen::Ref<const Eigen::VectorXd>& weights);

RowMajorMatrixXd probabilities_from_logits(
    const Eigen::Ref<const RowMajorMatrixXd>& logits, double temperature);

namespace testing {

void run_classifier_gradient_test();

} // namespace testing

} // namespace punkst::partition_classifier
