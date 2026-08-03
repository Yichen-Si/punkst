#include "gamma_pois_dispersion.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>

#include <Eigen/Eigenvalues>
#include <tbb/parallel_for.h>

#include "error.hpp"
#include "gamma_pois_topic.hpp"
#include "numerical_utils.hpp"

namespace {

constexpr double kTinyMean = 1e-300;

double local_polynomial_fit(const std::vector<double>& x,
    const std::vector<double>& y, const std::vector<double>& weights,
    const std::vector<int32_t>& order, int32_t sorted_index, int32_t window) {
    const int32_t n = static_cast<int32_t>(x.size());
    int32_t left = std::max<int32_t>(0, sorted_index - window / 2);
    int32_t right = std::min<int32_t>(n - 1, left + window - 1);
    left = std::max<int32_t>(0, right - window + 1);
    const double x0 = x[order[sorted_index]];
    const double radius = std::max(x0 - x[order[left]], x[order[right]] - x0);

    Eigen::Matrix3d normal = Eigen::Matrix3d::Zero();
    Eigen::Vector3d target = Eigen::Vector3d::Zero();
    double weight_sum = 0.0;
    double weighted_y = 0.0;
    for (int32_t r = left; r <= right; ++r) {
        const int32_t j = order[r];
        const double centered = x[j] - x0;
        const double scaled = radius > 0.0 ? centered / radius : 0.0;
        double kernel = 1.0;
        if (radius > 0.0) {
            const double u = std::min(1.0, std::abs(centered) / radius);
            kernel = std::pow(1.0 - u * u * u, 3.0);
        }
        const double weight = weights[j] * kernel;
        if (!(weight > 0.0) || !std::isfinite(weight)) continue;
        const Eigen::Vector3d row(1.0, scaled, scaled * scaled);
        normal.noalias() += weight * row * row.transpose();
        target.noalias() += weight * row * y[j];
        weight_sum += weight;
        weighted_y += weight * y[j];
    }
    if (!(weight_sum > 0.0)) return y[order[sorted_index]];
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> spectrum(normal);
    const bool quadratic_rank = spectrum.info() == Eigen::Success
        && spectrum.eigenvalues().maxCoeff() > 0.0
        && spectrum.eigenvalues().minCoeff()
            > 1e-12 * spectrum.eigenvalues().maxCoeff();
    Eigen::LDLT<Eigen::Matrix3d> solve(normal);
    if (quadratic_rank && solve.info() == Eigen::Success) {
        const Eigen::Vector3d coefficient = solve.solve(target);
        if (solve.info() == Eigen::Success && coefficient.allFinite()) {
            return coefficient(0);
        }
    }

    Eigen::Matrix2d linear = normal.topLeftCorner<2, 2>();
    Eigen::Vector2d linear_target = target.head<2>();
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix2d>
        linear_spectrum(linear);
    const bool linear_rank = linear_spectrum.info() == Eigen::Success
        && linear_spectrum.eigenvalues().maxCoeff() > 0.0
        && linear_spectrum.eigenvalues().minCoeff()
            > 1e-12 * linear_spectrum.eigenvalues().maxCoeff();
    Eigen::LDLT<Eigen::Matrix2d> linear_solve(linear);
    if (linear_rank && linear_solve.info() == Eigen::Success) {
        const Eigen::Vector2d coefficient = linear_solve.solve(linear_target);
        if (linear_solve.info() == Eigen::Success && coefficient.allFinite()) {
            return coefficient(0);
        }
    }
    return weighted_y / weight_sum;
}

void robust_weighted_loess(const std::vector<double>& x,
    const std::vector<double>& y, const std::vector<double>& base_weights,
    double span, std::vector<double>& fitted) {
    const int32_t n = static_cast<int32_t>(x.size());
    fitted.resize(n);
    if (n == 1) {
        fitted[0] = y[0];
        return;
    }
    if (n == 2) {
        fitted = y;
        return;
    }
    std::vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
        [&](int32_t a, int32_t b) { return x[a] < x[b]; });
    const int32_t window = std::max<int32_t>(3,
        static_cast<int32_t>(std::ceil(span * n)));
    std::vector<double> robustness(n, 1.0), weights(n), residual(n);
    for (int32_t iteration = 0; iteration <= 2; ++iteration) {
        for (int32_t j = 0; j < n; ++j) {
            weights[j] = base_weights[j] * robustness[j];
        }
        for (int32_t r = 0; r < n; ++r) {
            fitted[order[r]] = local_polynomial_fit(
                x, y, weights, order, r, std::min(window, n));
        }
        if (iteration == 2) break;
        for (int32_t j = 0; j < n; ++j) residual[j] = y[j] - fitted[j];
        const double center = median(residual);
        for (double& value : residual) value = std::abs(value - center);
        const double residual_mad = median(residual);
        if (!(residual_mad > 1e-15) || !std::isfinite(residual_mad)) break;
        const double cutoff = 6.0 * residual_mad;
        for (int32_t j = 0; j < n; ++j) {
            const double u = residual[j] / cutoff;
            robustness[j] = u < 1.0 ? std::pow(1.0 - u * u, 2.0) : 0.0;
        }
    }
}

double interpolate_trend(double x, const std::vector<double>& sample_x,
    const std::vector<double>& fitted, const std::vector<int32_t>& order) {
    if (order.size() == 1) return fitted[order.front()];
    if (x <= sample_x[order.front()]) return fitted[order.front()];
    if (x >= sample_x[order.back()]) return fitted[order.back()];
    const auto it = std::upper_bound(order.begin(), order.end(), x,
        [&](double value, int32_t index) { return value < sample_x[index]; });
    const int32_t right = *it;
    const int32_t left = *(it - 1);
    const double width = sample_x[right] - sample_x[left];
    if (!(width > 0.0)) return 0.5 * (fitted[left] + fitted[right]);
    const double fraction = (x - sample_x[left]) / width;
    return (1.0 - fraction) * fitted[left] + fraction * fitted[right];
}

void write_number_or_na(std::ostream& out, double value) {
    if (std::isfinite(value)) out << value;
    else out << "NA";
}

} // namespace

GammaPoissonDispersionEstimator::GammaPoissonDispersionEstimator(
    const GammaPoissonTopicModel& model,
    const GammaPoissonDispersionOptions& options)
    : model_(model), n_features_(model.get_n_features()),
      n_topics_(model.get_n_topics()), options_(options),
      feature_weights_active_(model.feature_weights_active()),
      feature_weight_(feature_weights_active_
          ? model.get_feature_weight()
          : std::vector<double>(static_cast<size_t>(n_features_), 1.0)),
      sum_z_(Eigen::VectorXd::Zero(n_topics_)),
      sum_zz_(Eigen::MatrixXd::Zero(n_topics_, n_topics_)),
      sum_posterior_variance_(Eigen::VectorXd::Zero(n_topics_)) {
    if (n_features_ <= 0 || n_topics_ <= 0
        || !std::isfinite(options_.min_information)
        || options_.min_information <= 0.0
        || !std::isfinite(options_.outlier_sd) || options_.outlier_sd < 0.0
        || !std::isfinite(options_.loess_span) || options_.loess_span <= 0.0
        || options_.loess_span > 1.0 || !std::isfinite(options_.delta_min)
        || !std::isfinite(options_.delta_max) || options_.delta_min <= 0.0
        || options_.delta_max < options_.delta_min) {
        error("%s: invalid dispersion estimation options", __func__);
    }
    if (feature_weight_.size() != static_cast<size_t>(n_features_)) {
        error("%s: invalid feature-weight metadata", __func__);
    }
    for (double weight : feature_weight_) {
        if (!std::isfinite(weight) || weight < 0.0) {
            error("%s: invalid feature weight", __func__);
        }
    }
    n_positive_.assign(n_features_, 0);
    sum_y_.assign(n_features_, 0.0);
    sum_y_squared_.assign(n_features_, 0.0);
    if (options_.estimator == GammaPoissonDispersionEstimatorKind::Residual) {
        sum_y_mu_.assign(n_features_, 0.0);
    }
    sum_positive_mu_cubed_.assign(n_features_, 0.0);
    sum_positive_mu_fourth_.assign(n_features_, 0.0);
    factorial_influence_sum_.assign(n_features_, 0.0);
    factorial_influence_max_.assign(n_features_, 0.0);
}

void GammaPoissonDispersionEstimator::accumulate(DocumentView docs) {
    if (finished_) error("%s: estimator has already been finalized", __func__);
    if (docs.size() > static_cast<size_t>(INT32_MAX - n_documents_)) {
        error("%s: too many documents", __func__);
    }
    for (size_t d = 0; d < docs.size(); ++d) {
        const Document& doc = docs[d];
        if (feature_weights_active_ && !doc.counts_weighted) {
            error("%s: weighted model received counts without the weighted marker",
                __func__);
        }
        if (doc.ids.size() != doc.cnts.size()) {
            error("%s: inconsistent sparse document", __func__);
        }
        for (size_t j = 0; j < doc.ids.size(); ++j) {
            if (doc.ids[j] >= static_cast<uint32_t>(n_features_)) {
                error("%s: feature index %u is out of range", __func__, doc.ids[j]);
            }
            if (!std::isfinite(doc.cnts[j])) {
                error("%s: invalid count for feature %u: %g",
                    __func__, doc.ids[j], doc.cnts[j]);
            }
        }
    }

    std::vector<GammaPoissonDocumentPosterior> posterior(docs.size());
    tbb::parallel_for(size_t(0), docs.size(), [&](size_t d) {
        model_.infer_document_posterior(docs[d], posterior[d]);
    });
    const Eigen::MatrixXd& beta = model_.get_expected_beta();
    RowMajorMatrixXd z_matrix(docs.size(), n_topics_);
    RowMajorMatrixXd variance_matrix(docs.size(), n_topics_);
    tbb::parallel_for(size_t(0), docs.size(), [&](size_t d) {
        const auto& post = posterior[d];
        if (post.shape.size() != n_topics_ || post.rate.size() != n_topics_
            || !std::isfinite(post.exposure) || post.exposure < 0.0) {
            error("%s: invalid document posterior", __func__);
        }
        if ((post.rate.array() <= 0.0).any()) {
            error("%s: non-positive document posterior rate", __func__);
        }
        const Eigen::VectorXd mean_theta = post.shape.array() / post.rate.array();
        const Eigen::VectorXd z = post.exposure * mean_theta;
        const Eigen::VectorXd posterior_variance = post.exposure * post.exposure
            * post.shape.array() / post.rate.array().square();
        if (!z.allFinite() || !posterior_variance.allFinite()) {
            error("%s: non-finite document posterior moments", __func__);
        }
        z_matrix.row(static_cast<int32_t>(d)) = z.transpose();
        variance_matrix.row(static_cast<int32_t>(d)) =
            posterior_variance.transpose();
    });
    sum_z_.noalias() += z_matrix.colwise().sum().transpose();
    sum_zz_.noalias() += z_matrix.transpose() * z_matrix;
    sum_posterior_variance_.noalias() +=
        variance_matrix.colwise().sum().transpose();

    for (size_t d = 0; d < docs.size(); ++d) {
        const Document& doc = docs[d];
        for (size_t j = 0; j < doc.ids.size(); ++j) {
            const int32_t w = static_cast<int32_t>(doc.ids[j]);
            const double feature_weight = feature_weight_[w];
            if (!(feature_weight > 0.0)) continue;
            const double y = feature_weights_active_
                ? doc.cnts[j] / feature_weight
                : doc.cnts[j];
            if (y <= 0.0) continue;
            const double mu = z_matrix.row(static_cast<int32_t>(d)).dot(
                beta.col(w)) / feature_weight;
            if (!std::isfinite(mu) || mu < 0.0) {
                error("%s: invalid fitted mean for feature %d", __func__, w);
            }
            ++n_positive_[w];
            sum_y_[w] += y;
            sum_y_squared_[w] += y * y;
            if (!sum_y_mu_.empty()) sum_y_mu_[w] += y * mu;
            const double mu2 = mu * mu;
            sum_positive_mu_cubed_[w] += mu2 * mu;
            sum_positive_mu_fourth_[w] += mu2 * mu2;
            const double influence = std::max(y * (y - 1.0), 0.0);
            factorial_influence_sum_[w] += influence;
            factorial_influence_max_[w] =
                std::max(factorial_influence_max_[w], influence);
        }
    }
    n_documents_ += static_cast<int32_t>(docs.size());
}

GammaPoissonDispersionResult GammaPoissonDispersionEstimator::finish() {
    if (finished_) error("%s: estimator has already been finalized", __func__);
    finished_ = true;
    const Eigen::MatrixXd& beta = model_.get_expected_beta();
    const Eigen::VectorXd effective_marginal_mean = beta.transpose() * sum_z_;
    const Eigen::MatrixXd second_moment_cross = sum_zz_ * beta;

    std::vector<double> q_mean(n_features_), q_corrected(n_features_);
    std::vector<double> gain(n_features_, 1.0), adjusted_q(n_features_);
    std::vector<double> raw(n_features_, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> se(n_features_, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> x(n_features_);
    std::vector<bool> eligible(n_features_, false);
    std::vector<int32_t> trend_features;
    std::vector<double> trend_x, trend_y, trend_weights, trend_fit;
    trend_features.reserve(n_features_);

    for (int32_t w = 0; w < n_features_; ++w) {
        const double feature_weight = feature_weight_[w];
        if (!(feature_weight > 0.0)) {
            q_mean[w] = q_corrected[w] = adjusted_q[w] =
                std::numeric_limits<double>::quiet_NaN();
            gain[w] = std::numeric_limits<double>::quiet_NaN();
            x[w] = std::log(kTinyMean);
            continue;
        }
        const double inverse_weight = 1.0 / feature_weight;
        const double inverse_weight_squared = inverse_weight * inverse_weight;
        q_mean[w] = beta.col(w).dot(second_moment_cross.col(w))
            * inverse_weight_squared;
        q_corrected[w] = q_mean[w]
            + (sum_posterior_variance_.array()
                * beta.col(w).array().square()).sum()
                * inverse_weight_squared;
        const double mean = effective_marginal_mean(w) * inverse_weight;
        if (options_.adjust_marginal_gain) {
            gain[w] = std::isfinite(mean) && mean > 0.0
                ? sum_y_[w] / mean
                : std::numeric_limits<double>::quiet_NaN();
        }
        const double a = gain[w];
        adjusted_q[w] = a * a * q_corrected[w];
        const double adjusted_mean = a * mean;
        x[w] = std::isfinite(adjusted_mean) && adjusted_mean > 0.0
            ? std::log(std::max(adjusted_mean
                / std::max(n_documents_, 1), kTinyMean))
            : std::log(kTinyMean);
        if (!std::isfinite(mean) || mean < 0.0 || !std::isfinite(q_mean[w])
            || q_mean[w] < 0.0 || !std::isfinite(q_corrected[w])
            || q_corrected[w] < 0.0 || !std::isfinite(a) || a < 0.0
            || !std::isfinite(adjusted_q[w])
            || adjusted_q[w] < options_.min_information) {
            continue;
        }
        const double a2 = a * a;
        if (options_.estimator == GammaPoissonDispersionEstimatorKind::Factorial) {
            raw[w] = (sum_y_squared_[w] - sum_y_[w]) / adjusted_q[w] - 1.0;
        } else {
            raw[w] = (sum_y_squared_[w] - 2.0 * a * sum_y_mu_[w]
                + 2.0 * a2 * q_mean[w] - a2 * q_corrected[w]
                - a * mean) / adjusted_q[w];
        }
        const double phi = std::max(raw[w], 0.0);
        const double t3 = a2 * a * sum_positive_mu_cubed_[w];
        const double t4 = a2 * a2 * sum_positive_mu_fourth_[w];
        double variance_numerator = 0.0;
        if (options_.estimator == GammaPoissonDispersionEstimatorKind::Factorial) {
            variance_numerator = 2.0 * (1.0 + phi) * adjusted_q[w]
                + 4.0 * (1.0 + phi) * (1.0 + 2.0 * phi) * t3
                + ((1.0 + phi) * (1.0 + 2.0 * phi) * (1.0 + 3.0 * phi)
                    - (1.0 + phi) * (1.0 + phi)) * t4;
        } else {
            variance_numerator = a * mean + (7.0 * phi + 2.0) * adjusted_q[w]
                + (12.0 * phi * phi + 4.0 * phi) * t3
                + (6.0 * phi * phi * phi + 2.0 * phi * phi) * t4;
        }
        se[w] = std::sqrt(std::max(variance_numerator, 0.0)) / adjusted_q[w];
        if (!std::isfinite(raw[w]) || !std::isfinite(se[w]) || !(se[w] > 0.0)) {
            raw[w] = std::numeric_limits<double>::quiet_NaN();
            se[w] = std::numeric_limits<double>::quiet_NaN();
            continue;
        }
        eligible[w] = true;
        trend_features.push_back(w);
        trend_x.push_back(x[w]);
        trend_y.push_back(raw[w]);
        trend_weights.push_back(1.0 / std::max(se[w] * se[w], 1e-24));
    }

    std::vector<double> trend(n_features_, options_.delta_min);
    std::vector<int32_t> trend_order;
    if (trend_features.empty()) {
        notice("No features have enough information for a raw dispersion estimate; using the lower-bound trend");
    } else {
        robust_weighted_loess(trend_x, trend_y, trend_weights,
            options_.loess_span, trend_fit);
        trend_order.resize(trend_features.size());
        std::iota(trend_order.begin(), trend_order.end(), 0);
        std::stable_sort(trend_order.begin(), trend_order.end(),
            [&](int32_t a, int32_t b) { return trend_x[a] < trend_x[b]; });
        for (int32_t w = 0; w < n_features_; ++w) {
            trend[w] = std::max(options_.delta_min,
                interpolate_trend(x[w], trend_x, trend_fit, trend_order));
        }
    }
    for (int32_t w = 0; w < n_features_; ++w) {
        if (!(feature_weight_[w] > 0.0)) trend[w] = options_.delta_min;
    }

    std::vector<double> deviations;
    double mean_se_squared = 0.0;
    for (int32_t w : trend_features) {
        deviations.push_back(raw[w] - trend[w]);
        mean_se_squared += se[w] * se[w];
    }
    const double minimum_prior_variance = std::max(
        options_.delta_min * options_.delta_min,
        std::numeric_limits<double>::min());
    double prior_variance = minimum_prior_variance;
    if (!deviations.empty()) {
        const double deviation_mad = mad(deviations);
        mean_se_squared /= deviations.size();
        prior_variance = std::max(minimum_prior_variance,
            deviation_mad * deviation_mad - mean_se_squared);
    }

    GammaPoissonDispersionResult result;
    result.n_documents = n_documents_;
    result.tau.resize(n_features_);
    result.diagnostics.resize(n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        double estimate = trend[w];
        int32_t status = GAMMA_POIS_DISPERSION_INSUFFICIENT;
        if (eligible[w]) {
            const double se2 = se[w] * se[w];
            estimate = (raw[w] / se2 + trend[w] / prior_variance)
                / (1.0 / se2 + 1.0 / prior_variance);
            const bool outlier = raw[w] > 0.0
                && raw[w] - trend[w] > options_.outlier_sd
                    * std::sqrt(se2 + prior_variance);
            if (outlier) estimate = raw[w];
            if (estimate >= options_.delta_max) {
                estimate = options_.delta_max;
                status = outlier ? GAMMA_POIS_DISPERSION_OUTLIER_CLAMPED_HIGH
                                 : GAMMA_POIS_DISPERSION_CLAMPED_HIGH;
            } else if (estimate <= options_.delta_min) {
                estimate = options_.delta_min;
                status = GAMMA_POIS_DISPERSION_CLAMPED_LOW;
            } else {
                status = outlier ? GAMMA_POIS_DISPERSION_OUTLIER
                                 : GAMMA_POIS_DISPERSION_ESTIMATED;
            }
        } else {
            estimate = std::clamp(estimate, options_.delta_min, options_.delta_max);
        }
        if (!std::isfinite(estimate) || !(estimate > 0.0)) {
            error("%s: non-finite shrunk dispersion for feature %d", __func__, w);
        }
        const double max_influence = factorial_influence_sum_[w] > 0.0
            ? factorial_influence_max_[w] / factorial_influence_sum_[w]
            : std::numeric_limits<double>::quiet_NaN();
        result.tau[w] = 1.0 / estimate;
        result.diagnostics[w] = {n_positive_[w], q_corrected[w], gain[w], raw[w],
            se[w], trend[w], estimate, result.tau[w], status, max_influence};
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
    out << "Feature\tn_positive\tQ_w\ta_w\tphi_raw\tse_phi\tphi_trend"
           "\tphi_shrunk\ttau\tstatus\tmax_influence\n";
    out << std::scientific << std::setprecision(4);
    for (size_t w = 0; w < result.diagnostics.size(); ++w) {
        const auto& d = result.diagnostics[w];
        out << (w < feature_names.size() ? feature_names[w] : std::to_string(w))
            << "\t" << d.n_positive << "\t";
        write_number_or_na(out, d.information);
        out << "\t";
        write_number_or_na(out, d.marginal_gain);
        out << "\t";
        write_number_or_na(out, d.delta_raw);
        out << "\t";
        write_number_or_na(out, d.se_delta);
        out << "\t";
        write_number_or_na(out, d.delta_trend);
        out << "\t";
        write_number_or_na(out, d.delta_shrunk);
        out << "\t";
        write_number_or_na(out, d.tau);
        out << "\t" << d.status << "\t";
        write_number_or_na(out, d.max_influence);
        out << "\n";
    }
}
