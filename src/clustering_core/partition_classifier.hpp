#pragma once

#include "numerical_utils.hpp"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace punkst::partition_classifier {

constexpr int32_t MODEL_SCHEMA_VERSION = 1;
constexpr int32_t CROSSFIT_SCHEMA_VERSION = 1;

struct FitOptions {
    std::vector<double> ridge_grid{
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0};
    int32_t folds = 5;
    int32_t max_iterations = 300;
    int32_t lbfgs_history = 10;
    double gradient_tolerance = 1e-7;
    std::function<void(const std::string&)> progress_callback;
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

struct CrossfitFoldDiagnostic {
    int32_t fold = -1;
    int32_t training_rows = 0;
    int32_t heldout_rows = 0;
    int32_t inner_folds = 0;
    double ridge = 0.0;
    double temperature = 1.0;
    Metrics metrics;
};

struct CrossfitResult {
    std::vector<Model> fold_models;
    Eigen::VectorXi fold_by_row;
    RowMajorMatrixXd probabilities;
    std::vector<CrossfitFoldDiagnostic> diagnostics;
    Metrics overall;
};

class CrossfitBundle {
public:
    Model full_model;
    std::vector<Model> fold_models;
    std::unordered_map<std::string, int32_t> heldout_fold_by_identifier;

    void validate(double tolerance = 1e-8) const;
    void write(const std::string& path) const;
    static bool is_bundle(const std::string& path);
    static CrossfitBundle read(const std::string& path);
    const Model& model_for(const std::string& identifier,
        bool use_crossfit, int32_t* heldout_fold = nullptr) const;
};

FitResult fit(const Eigen::Ref<const RowMajorMatrixXd>& compositions,
    const Eigen::Ref<const Eigen::VectorXi>& labels,
    const Eigen::Ref<const Eigen::VectorXd>& weights,
    const std::vector<std::string>& identifiers,
    const std::vector<std::string>& topics,
    const std::vector<std::string>& classes,
    uint64_t seed,
    const FitOptions& options = {});

CrossfitResult fit_crossfit(
    const Eigen::Ref<const RowMajorMatrixXd>& compositions,
    const Eigen::Ref<const Eigen::VectorXi>& labels,
    const Eigen::Ref<const Eigen::VectorXd>& weights,
    const std::vector<std::string>& identifiers,
    const std::vector<std::string>& topics,
    const std::vector<std::string>& classes,
    uint64_t seed,
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
