#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dataunits.hpp"

class GammaPoissonTopicModel;

struct GammaPoissonDispersionOptions {
    int32_t min_positive = 10;
    int32_t mu_bins = 32;
    double loess_span = 0.3;
    double delta_min = 1e-8;
    double delta_max = 1e4;
};

enum GammaPoissonDispersionStatus : int32_t {
    GAMMA_POIS_DISPERSION_LOW_POSITIVE = -2,
    GAMMA_POIS_DISPERSION_CLAMPED_LOW = -1,
    GAMMA_POIS_DISPERSION_ESTIMATED = 0,
    GAMMA_POIS_DISPERSION_CLAMPED_HIGH = 1,
};

struct GammaPoissonDispersionDiagnostic {
    double mean_mu = 0.0;
    int64_t n_positive = 0;
    double delta_raw = 0.0;
    double delta_trend = 0.0;
    double delta_shrunk = 0.0;
    double tau = 0.0;
    int32_t status = GAMMA_POIS_DISPERSION_LOW_POSITIVE;
};

struct GammaPoissonDispersionResult {
    int32_t n_documents = 0;
    std::vector<double> tau;
    std::vector<GammaPoissonDispersionDiagnostic> diagnostics;
};

class GammaPoissonDispersionEstimator {
public:
    GammaPoissonDispersionEstimator(int32_t n_features,
        const GammaPoissonDispersionOptions& options);

    void accumulate(const GammaPoissonTopicModel& model, DocumentView docs);
    GammaPoissonDispersionResult finish();

    static double positive_truncated_residual_expectation(double delta, double mu);

private:
    struct Observation {
        uint32_t feature = 0;
        double y = 0.0;
        double mu = 0.0;
    };

    double solve_feature_delta(int32_t feature) const;

    int32_t n_features_ = 0;
    GammaPoissonDispersionOptions options_;
    bool finished_ = false;
    int32_t n_documents_ = 0;
    std::vector<int64_t> n_positive_;
    std::vector<double> sum_mu_;
    std::vector<double> sum_g_;
    std::vector<double> sum_g_compensation_;
    std::vector<int64_t> bin_count_;
    std::vector<double> bin_log_mu_;
};

void write_gamma_poisson_dispersion_diagnostics(const std::string& out_file,
    const std::vector<std::string>& feature_names,
    const GammaPoissonDispersionResult& result);
