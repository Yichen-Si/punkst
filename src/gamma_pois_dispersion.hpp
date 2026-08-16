#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "Eigen/Dense"
#include "dataunits.hpp"

struct GammaPoissonDocumentPosterior {
    Eigen::VectorXd shape;
    Eigen::VectorXd rate;
    double exposure = 0.0;
};

class GammaPoissonDispersionModel {
public:
    virtual ~GammaPoissonDispersionModel() = default;
    virtual int32_t get_n_topics() const = 0;
    virtual int32_t get_n_features() const = 0;
    virtual bool feature_weights_active() const = 0;
    virtual const std::vector<double>& get_feature_weight() const = 0;
    virtual void infer_document_posterior(const Document& doc,
        GammaPoissonDocumentPosterior& posterior) const = 0;
    virtual const Eigen::MatrixXd& get_expected_beta() const = 0;
};

enum class GammaPoissonDispersionEstimatorKind : int32_t {
    Factorial = 0,
    Residual = 1,
};

struct GammaPoissonDispersionOptions {
    GammaPoissonDispersionEstimatorKind estimator =
        GammaPoissonDispersionEstimatorKind::Factorial;
    double min_information = 8.0;
    double outlier_sd = 2.0;
    double loess_span = 0.3;
    double delta_min = 1e-8;
    double delta_max = 1e4;
    // Transform data use N_w / M_w to calibrate the marginal mean. Training
    // data, where that calibration is not meaningful, use a_w = 1.
    bool adjust_marginal_gain = false;
};

enum GammaPoissonDispersionStatus : int32_t {
    GAMMA_POIS_DISPERSION_INSUFFICIENT = -2,
    GAMMA_POIS_DISPERSION_CLAMPED_LOW = -1,
    GAMMA_POIS_DISPERSION_ESTIMATED = 0,
    GAMMA_POIS_DISPERSION_CLAMPED_HIGH = 1,
    GAMMA_POIS_DISPERSION_OUTLIER = 2,
    GAMMA_POIS_DISPERSION_OUTLIER_CLAMPED_HIGH = 3,
};

struct GammaPoissonDispersionDiagnostic {
    int64_t n_positive = 0;
    double information = 0.0;
    double marginal_gain = 1.0;
    double delta_raw = 0.0;
    double se_delta = 0.0;
    double delta_trend = 0.0;
    double delta_shrunk = 0.0;
    double tau = 0.0;
    int32_t status = GAMMA_POIS_DISPERSION_INSUFFICIENT;
    double max_influence = 0.0;
};

struct GammaPoissonDispersionResult {
    int32_t n_documents = 0;
    std::vector<double> tau;
    std::vector<GammaPoissonDispersionDiagnostic> diagnostics;
};

class GammaPoissonDispersionEstimator {
public:
    GammaPoissonDispersionEstimator(const GammaPoissonDispersionModel& model,
        const GammaPoissonDispersionOptions& options);

    void accumulate(DocumentView docs);
    GammaPoissonDispersionResult finish();

private:
    const GammaPoissonDispersionModel& model_;
    int32_t n_features_ = 0;
    int32_t n_topics_ = 0;
    GammaPoissonDispersionOptions options_;
    bool feature_weights_active_ = false;
    std::vector<double> feature_weight_;
    bool finished_ = false;
    int32_t n_documents_ = 0;

    Eigen::VectorXd sum_z_;
    Eigen::MatrixXd sum_zz_;
    Eigen::VectorXd sum_posterior_variance_;
    std::vector<int64_t> n_positive_;
    std::vector<double> sum_y_;
    std::vector<double> sum_y_squared_;
    std::vector<double> sum_y_mu_;
    std::vector<double> sum_positive_mu_cubed_;
    std::vector<double> sum_positive_mu_fourth_;
    std::vector<double> factorial_influence_sum_;
    std::vector<double> factorial_influence_max_;
};

void write_gamma_poisson_dispersion_diagnostics(const std::string& out_file,
    const std::vector<std::string>& feature_names,
    const GammaPoissonDispersionResult& result);

std::vector<double> read_gamma_poisson_dispersion(
    const std::string& input_file,
    const std::vector<std::string>& expected_feature_names);
