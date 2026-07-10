#include "gamma_pois_dispersion.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>

#include <tbb/parallel_for.h>

#include "error.hpp"
#include "gamma_pois_topic.hpp"
#include "numerical_utils.hpp"
#include "utils.h"

namespace {

constexpr double kLogMuMin = -20.0;
constexpr double kLogMuMax = 20.0;

void compensated_add(double value, double& sum, double& compensation) {
    const double adjusted = value - compensation;
    const double updated = sum + adjusted;
    compensation = (updated - sum) - adjusted;
    sum = updated;
}

} // namespace

GammaPoissonDispersionEstimator::GammaPoissonDispersionEstimator(int32_t n_features,
    const GammaPoissonDispersionOptions& options)
    : n_features_(n_features), options_(options) {
    if (n_features_ <= 0 || options_.min_positive < 1 || options_.mu_bins < 1
        || !std::isfinite(options_.loess_span) || options_.loess_span <= 0.0
        || options_.loess_span > 1.0 || !std::isfinite(options_.delta_min)
        || !std::isfinite(options_.delta_max) || options_.delta_min <= 0.0
        || options_.delta_max < options_.delta_min) {
        error("%s: invalid dispersion estimation options", __func__);
    }
    n_positive_.assign(n_features_, 0);
    sum_mu_.assign(n_features_, 0.0);
    sum_g_.assign(n_features_, 0.0);
    sum_g_compensation_.assign(n_features_, 0.0);
    const size_t n_bins = checked_mul(static_cast<size_t>(n_features_),
        static_cast<size_t>(options_.mu_bins), __func__);
    bin_count_.assign(n_bins, 0);
    bin_log_mu_.assign(n_bins, 0.0);
}

void GammaPoissonDispersionEstimator::accumulate(const GammaPoissonTopicModel& model,
    DocumentView docs) {
    if (finished_) error("%s: estimator has already been finalized", __func__);
    std::vector<std::vector<Observation>> observations(docs.size());
    tbb::parallel_for(size_t(0), docs.size(), [&](size_t d) {
        const Document& doc = docs[d];
        std::vector<double> means;
        model.expected_observed_counts(doc, means);
        auto& out = observations[d];
        out.reserve(doc.ids.size());
        for (size_t j = 0; j < doc.ids.size(); ++j) {
            out.push_back({doc.ids[j], doc.cnts[j], means[j]});
        }
    });

    const double bin_width = (kLogMuMax - kLogMuMin)
        / static_cast<double>(options_.mu_bins);
    for (size_t d = 0; d < observations.size(); ++d) {
        for (const Observation& obs : observations[d]) {
            if (obs.feature >= static_cast<uint32_t>(n_features_) || !std::isfinite(obs.y)
                || obs.y <= 0.0 || !std::isfinite(obs.mu) || obs.mu <= 0.0) {
                error("%s: invalid residual input for feature %u: y=%g, mu=%g",
                    __func__, obs.feature, obs.y, obs.mu);
            }
            const long double y = obs.y;
            const long double mu = obs.mu;
            const long double diff = y - mu;
            const long double g_long = (diff * diff - mu) / (mu * mu);
            const double g = static_cast<double>(g_long);
            if (!std::isfinite(g)) {
                error("%s: non-finite residual for feature %u: y=%g, mu=%g",
                    __func__, obs.feature, obs.y, obs.mu);
            }
            const int32_t w = static_cast<int32_t>(obs.feature);
            ++n_positive_[w];
            sum_mu_[w] += obs.mu;
            compensated_add(g, sum_g_[w], sum_g_compensation_[w]);
            const double log_mu = std::clamp(std::log(obs.mu), kLogMuMin, kLogMuMax);
            int32_t bin = static_cast<int32_t>((log_mu - kLogMuMin) / bin_width);
            bin = std::clamp(bin, 0, options_.mu_bins - 1);
            const size_t idx = static_cast<size_t>(w) * options_.mu_bins + bin;
            ++bin_count_[idx];
            bin_log_mu_[idx] += log_mu;
        }
    }
    n_documents_ += static_cast<int32_t>(docs.size());
}

double GammaPoissonDispersionEstimator::positive_truncated_residual_expectation(
    double delta, double mu) {
    if (!std::isfinite(delta) || delta <= 0.0 || !std::isfinite(mu) || mu <= 0.0) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    const double log_p0 = -std::log1p(delta * mu) / delta;
    const double p0 = std::exp(log_p0);
    const double positive_probability = -std::expm1(log_p0);
    const double g0 = 1.0 - 1.0 / mu;
    return (delta - g0 * p0) / positive_probability;
}

double GammaPoissonDispersionEstimator::solve_feature_delta(int32_t w) const {
    const int64_t n = n_positive_[w];
    const double observed = sum_g_[w] / static_cast<double>(n);
    if (!std::isfinite(observed)) {
        error("%s: non-finite observed moment for feature %d", __func__, w);
    }
    auto expected = [&](double delta) {
        double total = 0.0;
        for (int32_t b = 0; b < options_.mu_bins; ++b) {
            const size_t idx = static_cast<size_t>(w) * options_.mu_bins + b;
            const int64_t count = bin_count_[idx];
            if (count == 0) continue;
            const double mu = bin_log_mu_[idx];
            total += static_cast<double>(count)
                * positive_truncated_residual_expectation(delta, mu);
        }
        return total / static_cast<double>(n);
    };
    double lo = options_.delta_min;
    double hi = options_.delta_max;
    const double expected_lo = expected(lo);
    const double expected_hi = expected(hi);
    if (!std::isfinite(expected_lo) || !std::isfinite(expected_hi)) {
        error("%s: non-finite expected moment for feature %d", __func__, w);
    }
    if (expected_lo >= observed) return lo;
    if (expected_hi <= observed) return hi;
    for (int iter = 0; iter < 60; ++iter) {
        const double mid = std::sqrt(lo * hi);
        const double expected_mid = expected(mid);
        if (!std::isfinite(expected_mid)) {
            error("%s: non-finite expected moment for feature %d", __func__, w);
        }
        if (expected_mid < observed) lo = mid;
        else hi = mid;
    }
    return std::sqrt(lo * hi);
}

GammaPoissonDispersionResult GammaPoissonDispersionEstimator::finish() {
    if (finished_) error("%s: estimator has already been finalized", __func__);
    finished_ = true;
    for (size_t idx = 0; idx < bin_count_.size(); ++idx) {
        if (bin_count_[idx] > 0) {
            bin_log_mu_[idx] = std::exp(bin_log_mu_[idx]
                / static_cast<double>(bin_count_[idx]));
        }
    }

    std::vector<double> raw(n_features_, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> trend(n_features_, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> shrunk(n_features_, std::numeric_limits<double>::quiet_NaN());
    std::vector<std::string> status(n_features_, "low_positive");
    std::vector<double> trend_x, trend_y, trend_fit;
    std::vector<int32_t> trend_features;
    for (int32_t w = 0; w < n_features_; ++w) {
        const int64_t n = n_positive_[w];
        if (n < options_.min_positive) continue;
        raw[w] = solve_feature_delta(w);
        status[w] = raw[w] <= options_.delta_min * (1.0 + 1e-10)
            ? "clamped_low" : raw[w] >= options_.delta_max * (1.0 - 1e-10)
            ? "clamped_high" : "estimated";
        trend_x.push_back(std::log(std::max(sum_mu_[w] / n, 1e-12)));
        trend_y.push_back(std::log(raw[w]));
        trend_features.push_back(w);
    }
    if (trend_features.empty()) {
        error("%s: no features meet --dispersion-min-positive", __func__);
    }
    if (trend_features.size() >= 3) {
        loess_quadratic_tricube(trend_x, trend_y, trend_fit, options_.loess_span);
    } else {
        trend_fit = trend_y;
    }
    if (trend_fit.size() != trend_features.size()) {
        error("%s: dispersion LOESS returned an invalid result size", __func__);
    }
    for (size_t i = 0; i < trend_fit.size(); ++i) {
        if (!std::isfinite(trend_fit[i])) {
            error("%s: non-finite dispersion trend for feature %d",
                __func__, trend_features[i]);
        }
    }
    std::vector<int32_t> order(trend_features.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return trend_x[a] < trend_x[b];
    });
    auto trend_at = [&](double x) {
        if (order.size() == 1) return trend_fit[0];
        if (x <= trend_x[order.front()]) return trend_fit[order.front()];
        if (x >= trend_x[order.back()]) return trend_fit[order.back()];
        auto it = std::upper_bound(order.begin(), order.end(), x,
            [&](double value, int32_t idx) { return value < trend_x[idx]; });
        const int32_t right = *it;
        const int32_t left = *(it - 1);
        const double fraction = (x - trend_x[left])
            / std::max(trend_x[right] - trend_x[left], 1e-12);
        return (1.0 - fraction) * trend_fit[left] + fraction * trend_fit[right];
    };

    GammaPoissonDispersionResult result;
    result.n_documents = n_documents_;
    result.tau.resize(n_features_);
    result.diagnostics.resize(n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        const int64_t n = n_positive_[w];
        const double mean_mu = n > 0 ? sum_mu_[w] / n : 1e-12;
        if (!std::isfinite(mean_mu) || mean_mu <= 0.0) {
            error("%s: non-finite mean for feature %d", __func__, w);
        }
        const double log_trend = trend_at(std::log(std::max(mean_mu, 1e-12)));
        trend[w] = std::exp(log_trend);
        shrunk[w] = trend[w];
        if (std::isfinite(raw[w])) {
            const double strength = static_cast<double>(options_.min_positive);
            shrunk[w] = std::exp((strength * log_trend + n * std::log(raw[w]))
                / (strength + n));
        }
        shrunk[w] = std::clamp(shrunk[w], options_.delta_min, options_.delta_max);
        if (!std::isfinite(shrunk[w]) || shrunk[w] <= 0.0) {
            error("%s: non-finite shrunk dispersion for feature %d", __func__, w);
        }
        result.tau[w] = 1.0 / shrunk[w];
        result.diagnostics[w] = {n > 0 ? mean_mu : 0.0, n, raw[w], trend[w],
            shrunk[w], result.tau[w], status[w]};
    }
    return result;
}

void write_gamma_poisson_dispersion_diagnostics(const std::string& out_file,
    const std::vector<std::string>& feature_names,
    const GammaPoissonDispersionResult& result) {
    if (result.tau.size() != result.diagnostics.size()) {
        error("%s: inconsistent dispersion result", __func__);
    }
    std::ofstream out(out_file);
    if (!out) error("%s: Error opening output file: %s", __func__, out_file.c_str());
    out << "Feature\tmean_mu\tn_positive\tdelta_raw\tdelta_trend\tdelta_shrunk\ttau\tstatus\n";
    out << std::scientific << std::setprecision(6);
    for (size_t w = 0; w < result.diagnostics.size(); ++w) {
        const auto& d = result.diagnostics[w];
        out << (w < feature_names.size() ? feature_names[w] : std::to_string(w))
            << "\t" << d.mean_mu << "\t" << d.n_positive << "\t" << d.delta_raw
            << "\t" << d.delta_trend << "\t" << d.delta_shrunk << "\t" << d.tau
            << "\t" << d.status << "\n";
    }
}
