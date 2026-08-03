#include "gamma_pois_topic.hpp"
#include "lda.hpp"
#include "transform_helper.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

int32_t cmdGammaPoisTransform(int argc, char** argv);
int32_t cmdLDATransform(int argc, char** argv);
int32_t cmdUacFit(int argc, char** argv);
int32_t cmdUacTransform(int argc, char** argv);

namespace {

constexpr int32_t n_topics = 3;
constexpr int32_t n_features = 6;
constexpr int32_t seed = 31;
constexpr double inference_tolerance = 1e-10;

struct ExpectedDiagnostics {
    double total_variation = 0.0;
    double topic_information = 0.0;
    double cofeature_corroboration =
        std::numeric_limits<double>::quiet_NaN();
    double cofeature_conflict =
        std::numeric_limits<double>::quiet_NaN();
    double pull = 0.0;
    bool supported = false;
};

struct FeatureTable {
    std::vector<std::string> header;
    std::unordered_map<std::string, std::vector<std::string>> rows;
};

struct ExpectedVarianceDiagnostics {
    double factorial = 0.0;
    double adjusted_topic_second_moment =
        std::numeric_limits<double>::quiet_NaN();
    double depth_second_moment =
        std::numeric_limits<double>::quiet_NaN();
    double excess_variance_explained_by_structure =
        std::numeric_limits<double>::quiet_NaN();
    double total_variance_explained_by_structure =
        std::numeric_limits<double>::quiet_NaN();
    double uncertainty = std::numeric_limits<double>::quiet_NaN();
};

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(
            "Feature diagnostic test failed: " + message);
    }
}

void write_text(
        const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    require(static_cast<bool>(output),
        "cannot write " + path.string());
    output << text;
}

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> fields;
    std::istringstream input(line);
    std::string field;
    while (std::getline(input, field, '\t')) {
        fields.push_back(field);
    }
    return fields;
}

FeatureTable read_feature_table(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input),
        "cannot read " + path.string());
    FeatureTable table;
    std::string line;
    require(static_cast<bool>(std::getline(input, line)),
        "missing header in " + path.string());
    table.header = split_tabs(line);
    while (std::getline(input, line)) {
        std::vector<std::string> fields = split_tabs(line);
        require(!fields.empty(), "empty feature diagnostic row");
        table.rows.emplace(fields.front(), std::move(fields));
    }
    return table;
}

size_t column_index(const FeatureTable& table, const std::string& name) {
    const auto found = std::find(
        table.header.begin(), table.header.end(), name);
    require(found != table.header.end(),
        "missing feature diagnostic column " + name);
    return static_cast<size_t>(found - table.header.begin());
}

int32_t run_command(int32_t (*command)(int, char**),
        std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) {
        argv.push_back(argument.data());
    }
    return command(static_cast<int32_t>(argv.size()), argv.data());
}

ExpectedDiagnostics summarize_deletion(
        const VectorXd& full, const VectorXd& deleted) {
    ExpectedDiagnostics result;
    result.supported = true;
    result.total_variation =
        0.5 * (deleted - full).cwiseAbs().sum();
    return result;
}

void add_cofeature_references(const MatrixXd& topic_feature,
        const VectorXd& topic_reference, const Document& document,
        std::vector<ExpectedDiagnostics>& expected) {
    const feature_diagnostics::CofeatureModel model =
        feature_diagnostics::make_cofeature_model(
            topic_feature, topic_reference);
    VectorXd document_sum = VectorXd::Zero(topic_feature.rows());
    int32_t positive_features = 0;
    for (size_t j = 0; j < document.ids.size(); ++j) {
        if (document.cnts[j] <= 0.0) continue;
        const int32_t feature =
            static_cast<int32_t>(document.ids[j]);
        document_sum += model.featureTopic.col(feature);
        ++positive_features;
    }
    for (int32_t feature = 0; feature < topic_feature.cols(); ++feature) {
        expected[feature].topic_information =
            model.topicInformation(feature);
    }
    for (size_t j = 0; j < document.ids.size(); ++j) {
        if (document.cnts[j] <= 0.0) continue;
        const int32_t feature =
            static_cast<int32_t>(document.ids[j]);
        const double lift = feature_diagnostics::cofeature_log_lift(
            model.featureTopic, model.topicReference,
            document_sum, feature, positive_features);
        if (std::isfinite(lift)) {
            expected[feature].cofeature_corroboration =
                std::max(0.0, lift);
            expected[feature].cofeature_conflict =
                std::max(0.0, -lift);
        }
    }
}

void test_cofeature_metrics() {
    MatrixXd topic_feature(2, 3);
    topic_feature <<
        8.0, 1.0, 1.0,
        1.0, 8.0, 1.0;
    const VectorXd reference =
        (VectorXd(2) << 0.5, 0.5).finished();
    const feature_diagnostics::CofeatureModel model =
        feature_diagnostics::make_cofeature_model(
            topic_feature, reference);
    require(std::abs(model.featureTopic(0, 0) - 8.0 / 9.0)
                < 1e-15
            && std::abs(model.featureTopic(0, 2) - 0.5) < 1e-15,
        "feature-only topic signatures are incorrect");
    require(model.topicInformation(0) > 0.0
            && std::abs(model.topicInformation(2)) < 1e-15,
        "topic information does not distinguish a flat feature");

    const VectorXd same_context =
        2.0 * model.featureTopic.col(0);
    const VectorXd opposing_context =
        model.featureTopic.col(0) + model.featureTopic.col(1);
    require(feature_diagnostics::cofeature_log_lift(
                model.featureTopic, model.topicReference,
                same_context, 0, 2) > 0.0,
        "same-direction cofeature lift is not positive");
    require(feature_diagnostics::cofeature_log_lift(
                model.featureTopic, model.topicReference,
                opposing_context, 0, 2) < 0.0,
        "opposing cofeature lift is not negative");
    require(std::isnan(feature_diagnostics::cofeature_log_lift(
                model.featureTopic, model.topicReference,
                model.featureTopic.col(0), 0, 1)),
        "a feature without cofeatures has a finite lift");
}

std::vector<ExpectedDiagnostics> lda_references(
        const std::filesystem::path& model_path,
        const Document& document, bool use_training_prevalence = false) {
    LatentDirichletAllocation lda(
        model_path.string(), seed, 1, 0, InferenceType::SVB);
    lda.set_svb_parameters(200, inference_tolerance);
    std::vector<Document> documents{document};
    const RowMajorMatrixXd gamma =
        lda.transform_gamma(DocumentView(documents));
    VectorXd full = gamma.row(0).transpose();
    full /= full.sum();
    const MatrixXd beta_kernel =
        dirichlet_expectation_2d(lda.get_model());
    const RowMajorMatrixXd beta = rowNormalize(lda.get_model());
    VectorXd theta_kernel(n_topics);
    for (int32_t k = 0; k < n_topics; ++k) {
        theta_kernel(k) =
            std::exp(psi(gamma(0, k) + lda.get_doc_topic_prior()));
    }

    std::vector<ExpectedDiagnostics> expected(n_features);
    for (size_t j = 0; j < document.ids.size(); ++j) {
        const int32_t feature =
            static_cast<int32_t>(document.ids[j]);
        const double observed = document.cnts[j];
        VectorXd allocation =
            theta_kernel.array() * beta_kernel.col(feature).array();
        allocation /= allocation.sum();
        VectorXd deleted =
            gamma.row(0).transpose() - observed * allocation;
        deleted = deleted.cwiseMax(0.0);
        if (deleted.sum() > 0.0) {
            deleted /= deleted.sum();
        } else {
            deleted.setConstant(1.0 / n_topics);
        }
        expected[feature] =
            summarize_deletion(full, deleted);
        const double fitted_mean = document.ct_tot
            * full.dot(beta.col(feature));
        expected[feature].pull = std::abs(observed - fitted_mean)
            * 0.5 * (allocation - deleted).cwiseAbs().sum()
            / observed;
    }
    VectorXd topic_reference = full;
    if (use_training_prevalence) {
        topic_reference = lda.get_model().rowwise().sum();
    }
    add_cofeature_references(
        lda.get_model(), topic_reference, document, expected);
    return expected;
}

std::vector<ExpectedVarianceDiagnostics> lda_variance_references(
        const std::filesystem::path& model_path,
        const std::vector<Document>& inference_documents,
        const std::vector<Document>* raw_documents = nullptr) {
    LatentDirichletAllocation lda(
        model_path.string(), seed, 1, 0, InferenceType::SVB);
    lda.set_svb_parameters(200, inference_tolerance);
    const RowMajorMatrixXd gamma =
        lda.transform_gamma(DocumentView(inference_documents));
    RowMajorMatrixXd topics = gamma;
    for (int32_t d = 0; d < topics.rows(); ++d) {
        topics.row(d) /= topics.row(d).sum();
    }

    const RowMajorMatrixXd beta = rowNormalize(lda.get_model());
    VectorXd sum_n_theta = VectorXd::Zero(n_topics);
    MatrixXd sum_nn1_theta_theta = MatrixXd::Zero(n_topics, n_topics);
    VectorXd sum_theta = VectorXd::Zero(n_topics);
    MatrixXd sum_theta_theta = MatrixXd::Zero(n_topics, n_topics);
    VectorXd sum_theta_over_n = VectorXd::Zero(n_topics);
    MatrixXd sum_theta_theta_over_n = MatrixXd::Zero(n_topics, n_topics);
    VectorXd sum_uncertainty_theta = VectorXd::Zero(n_topics);
    MatrixXd sum_uncertainty_theta_theta =
        MatrixXd::Zero(n_topics, n_topics);
    VectorXd old_sum_uncertainty_theta = VectorXd::Zero(n_topics);
    MatrixXd old_sum_uncertainty_theta_theta =
        MatrixXd::Zero(n_topics, n_topics);
    double total_count = 0.0;
    double total_factorial_count = 0.0;
    int64_t positive_units = 0;
    std::vector<ExpectedVarianceDiagnostics> expected(n_features);
    std::vector<double> feature_total(n_features, 0.0);

    const std::vector<Document>& raw = raw_documents == nullptr
        ? inference_documents
        : *raw_documents;
    require(raw.size() == inference_documents.size(),
        "LDA raw reference documents do not align");
    for (size_t d = 0; d < raw.size(); ++d) {
        const Document& document = raw[d];
        double n = 0.0;
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const int32_t w = static_cast<int32_t>(document.ids[j]);
            const double observed = document.cnts[j];
            n += observed;
            feature_total[w] += observed;
            expected[w].factorial += observed * (observed - 1.0);
        }
        if (!(n > 0.0)) continue;
        const VectorXd theta = topics.row(static_cast<int32_t>(d)).transpose();
        const MatrixXd theta_second = theta * theta.transpose();
        const double nn1 = n * (n - 1.0);
        total_count += n;
        total_factorial_count += nn1;
        ++positive_units;
        sum_n_theta += n * theta;
        sum_nn1_theta_theta += nn1 * theta_second;
        sum_theta += theta;
        sum_theta_theta += theta_second;
        sum_theta_over_n += theta / n;
        sum_theta_theta_over_n += theta_second / n;
        const double concentration = gamma.row(static_cast<int32_t>(d)).sum()
            + static_cast<double>(n_topics) * lda.get_doc_topic_prior();
        const double uncertainty_weight = nn1 / (concentration + 1.0);
        const VectorXd posterior_mean =
            (gamma.row(static_cast<int32_t>(d)).array()
                + lda.get_doc_topic_prior()).matrix().transpose()
            / concentration;
        sum_uncertainty_theta += uncertainty_weight * posterior_mean;
        sum_uncertainty_theta_theta +=
            uncertainty_weight * posterior_mean * posterior_mean.transpose();
        old_sum_uncertainty_theta += uncertainty_weight * theta;
        old_sum_uncertainty_theta_theta +=
            uncertainty_weight * theta_second;
    }

    const VectorXd predicted = beta.transpose() * sum_n_theta;
    const MatrixXd factorial_cross = sum_nn1_theta_theta * beta;
    const MatrixXd unit_cross = sum_theta_theta * beta;
    const MatrixXd inverse_cross = sum_theta_theta_over_n * beta;
    const MatrixXd uncertainty_cross =
        sum_uncertainty_theta_theta * beta;
    const MatrixXd old_uncertainty_cross =
        old_sum_uncertainty_theta_theta * beta;
    bool distinguished_prior_free_uncertainty = false;
    for (int32_t w = 0; w < n_features; ++w) {
        if (!(predicted(w) > 0.0) || !(total_count > 0.0)
                || positive_units <= 0) {
            continue;
        }
        const VectorXd profile = beta.col(w);
        const double gain = feature_total[w] / predicted(w);
        const double gain_squared = gain * gain;
        const double qa = gain_squared
            * profile.dot(factorial_cross.col(w));
        const double q0 = feature_total[w] * feature_total[w]
            * total_factorial_count / (total_count * total_count);
        expected[w].adjusted_topic_second_moment = qa;
        expected[w].depth_second_moment = q0;
        const double total_extra = expected[w].factorial - q0;
        if (total_extra > 0.0) {
            expected[w].excess_variance_explained_by_structure =
                (qa - q0) / total_extra;
        }

        double structured_variance = gain_squared
                * profile.dot(unit_cross.col(w))
            - std::pow(gain * profile.dot(sum_theta), 2.0)
                / static_cast<double>(positive_units);
        double sampling_variance = gain * profile.dot(sum_theta_over_n)
            - gain_squared * profile.dot(inverse_cross.col(w));
        double dispersion_quadratic = gain_squared * profile.dot(
            unit_cross.col(w) - inverse_cross.col(w));
        if (structured_variance >= -1e-12) {
            structured_variance = std::max(0.0, structured_variance);
        }
        if (sampling_variance >= -1e-12) {
            sampling_variance = std::max(0.0, sampling_variance);
        }
        if (dispersion_quadratic >= -1e-12) {
            dispersion_quadratic = std::max(0.0, dispersion_quadratic);
        }
        if (qa > 0.0 && structured_variance >= 0.0
                && sampling_variance >= 0.0
                && dispersion_quadratic >= 0.0) {
            const double phi = std::max(expected[w].factorial / qa - 1.0, 0.0);
            const double denominator = structured_variance
                + sampling_variance + phi * dispersion_quadratic;
            if (denominator > 0.0) {
                expected[w].total_variance_explained_by_structure =
                    structured_variance / denominator;
            }
        }
        double uncertainty = gain_squared * (
            profile.array().square().matrix().dot(sum_uncertainty_theta)
            - profile.dot(uncertainty_cross.col(w)));
        if (uncertainty >= -1e-12) {
            expected[w].uncertainty = std::max(0.0, uncertainty);
        }
        const double old_uncertainty = gain_squared * (
            profile.array().square().matrix().dot(
                old_sum_uncertainty_theta)
            - profile.dot(old_uncertainty_cross.col(w)));
        distinguished_prior_free_uncertainty =
            distinguished_prior_free_uncertainty
            || std::abs(old_uncertainty - uncertainty)
                > 1e-8 * std::max(1.0, std::abs(uncertainty));
    }
    require(distinguished_prior_free_uncertainty,
        "LDA uncertainty fixture does not distinguish the posterior mean from the prior-free topic proportion");
    return expected;
}

class TestGammaPoissonModel : public GammaPoissonTopicModel {
public:
    using GammaPoissonTopicModel::GammaPoissonTopicModel;

    void set_beta_parameters(
            const MatrixXd& shape, const MatrixXd& rate) {
        beta_shape_ = shape;
        beta_rate_ = rate;
        refresh_cache();
    }
};

void test_batched_topic_moments() {
    RowMajorMatrixXd topics(4, 3);
    topics <<
        0.2, 0.3, 0.5,
        0.7, 0.1, 0.2,
        0.0, 0.4, 0.6,
        0.3, 0.6, 0.1;
    VectorXd weights(4);
    weights << 0.0, 2.0, 0.5, 3.0;
    VectorXd batched_sum = VectorXd::Zero(3);
    MatrixXd batched_second = MatrixXd::Zero(3, 3);
    RowMajorMatrixXd scratch;
    feature_diagnostics::accumulate_weighted_topic_sum(
        topics, weights, batched_sum);
    feature_diagnostics::accumulate_weighted_topic_second_moment(
        topics, weights, batched_second, scratch);
    VectorXd direct_sum = VectorXd::Zero(3);
    MatrixXd direct_second = MatrixXd::Zero(3, 3);
    for (int32_t d = 0; d < topics.rows(); ++d) {
        const VectorXd topic = topics.row(d).transpose();
        direct_sum += weights(d) * topic;
        direct_second.noalias() += weights(d) * topic * topic.transpose();
    }
    require((batched_sum - direct_sum).cwiseAbs().maxCoeff() < 1e-14
            && (batched_second - direct_second).cwiseAbs().maxCoeff() < 1e-14,
        "batched topic moments differ from direct accumulation");
}

void test_gamma_poisson_dispersion_all_cell_moments() {
    std::vector<double> feature_sums{1000.0, 1000.0};
    TestGammaPoissonModel model(
        1, 2, 23, 1, 0,
        0.3, 0.4, 0.3, 2.0, 3.0, -1.0,
        0.7, 10.0, 100, 20.0, false, -1.0, &feature_sums);
    MatrixXd shape(1, 2);
    shape << 10.0, 10.0;
    model.set_beta_parameters(shape, MatrixXd::Ones(1, 2));

    std::vector<Document> documents(4);
    documents[0].ids = {0, 1}; documents[0].cnts = {1.0, 9.0};
    documents[1].ids = {0, 1}; documents[1].cnts = {2.5, 38.0};
    documents[2].ids = {0, 1}; documents[2].cnts = {60.0, 40.0};
    documents[3].ids = {0, 1}; documents[3].cnts = {180.0, 20.0};
    Document explicit_zero;
    explicit_zero.ids = {0, 1};
    explicit_zero.cnts = {0.0, 0.0};
    documents.push_back(explicit_zero);

    const MatrixXd& beta = model.get_expected_beta();
    VectorXd sum_z = VectorXd::Zero(1);
    MatrixXd sum_zz = MatrixXd::Zero(1, 1);
    VectorXd sum_variance = VectorXd::Zero(1);
    std::vector<double> sum_y(2), sum_y2(2), sum_y_mu(2);
    for (const Document& document : documents) {
        GammaPoissonDocumentPosterior posterior;
        model.infer_document_posterior(document, posterior);
        const VectorXd z = posterior.exposure
            * (posterior.shape.array() / posterior.rate.array()).matrix();
        sum_z += z;
        sum_zz += z * z.transpose();
        sum_variance += posterior.exposure * posterior.exposure
            * (posterior.shape.array() / posterior.rate.array().square()).matrix();
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const int32_t w = document.ids[j];
            const double y = document.cnts[j];
            const double mu = z.dot(beta.col(w));
            sum_y[w] += y;
            sum_y2[w] += y * y;
            sum_y_mu[w] += y * mu;
        }
    }
    const VectorXd marginal_mean = beta.transpose() * sum_z;
    const MatrixXd cross = sum_zz * beta;
    std::vector<double> q_mean(2), q_corrected(2);
    for (int32_t w = 0; w < 2; ++w) {
        q_mean[w] = beta.col(w).dot(cross.col(w));
        q_corrected[w] = q_mean[w]
            + (sum_variance.array() * beta.col(w).array().square()).sum();
    }

    GammaPoissonDispersionOptions options;
    options.min_information = 1e-6;
    GammaPoissonDispersionEstimator factorial(model, options);
    factorial.accumulate(DocumentView(documents));
    const GammaPoissonDispersionResult factorial_result = factorial.finish();
    for (int32_t w = 0; w < 2; ++w) {
        const double expected = (sum_y2[w] - sum_y[w]) / q_corrected[w] - 1.0;
        require(std::abs(factorial_result.diagnostics[w].information
                    - q_corrected[w]) < 1e-10 * std::max(1.0, q_corrected[w]),
            "Gamma-Poisson posterior-corrected Q does not match dense moments");
        require(std::abs(factorial_result.diagnostics[w].delta_raw - expected)
                    < 1e-10 * std::max(1.0, std::abs(expected)),
            "Gamma-Poisson factorial raw dispersion is incorrect");
        require(factorial_result.diagnostics[w].marginal_gain == 1.0,
            "training-data dispersion unexpectedly adjusted marginal gain");
    }

    options.estimator = GammaPoissonDispersionEstimatorKind::Residual;
    GammaPoissonDispersionEstimator residual(model, options);
    residual.accumulate(DocumentView(documents));
    const GammaPoissonDispersionResult residual_result = residual.finish();
    for (int32_t w = 0; w < 2; ++w) {
        const double expected = (sum_y2[w] - 2.0 * sum_y_mu[w]
            + 2.0 * q_mean[w] - q_corrected[w] - marginal_mean(w))
            / q_corrected[w];
        require(std::abs(residual_result.diagnostics[w].delta_raw - expected)
                    < 1e-10 * std::max(1.0, std::abs(expected)),
            "Gamma-Poisson residual raw dispersion is incorrect");
    }

    options.estimator = GammaPoissonDispersionEstimatorKind::Factorial;
    options.adjust_marginal_gain = true;
    GammaPoissonDispersionEstimator transform(model, options);
    transform.accumulate(DocumentView(documents));
    const GammaPoissonDispersionResult transform_result = transform.finish();
    for (int32_t w = 0; w < 2; ++w) {
        const double gain = sum_y[w] / marginal_mean(w);
        require(std::abs(transform_result.diagnostics[w].marginal_gain - gain)
                    < 1e-12 * std::max(1.0, gain),
            "transform-data dispersion marginal gain is incorrect");
        const double expected = (sum_y2[w] - sum_y[w])
            / (gain * gain * q_corrected[w]) - 1.0;
        require(std::abs(transform_result.diagnostics[w].delta_raw - expected)
                    < 1e-10 * std::max(1.0, std::abs(expected)),
            "gain-adjusted factorial raw dispersion is incorrect");
    }
    const double expected_influence = 180.0 * 179.0
        / (2.5 * 1.5 + 60.0 * 59.0 + 180.0 * 179.0);
    require(std::abs(transform_result.diagnostics[0].max_influence
                - expected_influence) < 1e-12,
        "Gamma-Poisson maximum factorial influence is incorrect");

    const std::filesystem::path diagnostic_path =
        std::filesystem::temp_directory_path()
        / "punkst_gamma_pois_dispersion_test.tsv";
    write_gamma_poisson_dispersion_diagnostics(
        diagnostic_path.string(), {"a", "b"}, transform_result);
    std::ifstream diagnostic_input(diagnostic_path);
    std::string header;
    std::getline(diagnostic_input, header);
    require(header == "Feature\tn_positive\tQ_w\ta_w\tphi_raw\tse_phi"
            "\tphi_trend\tphi_shrunk\ttau\tstatus\tmax_influence",
        "Gamma-Poisson dispersion diagnostic schema is incorrect");
    std::filesystem::remove(diagnostic_path);
    require(run_command(cmdGammaPoisTransform, {
                "gamma-pois-transform", "--in-state", "unused",
                "--out-prefix", "unused", "--dispersion-mu-bins", "32"}) != 0,
        "removed --dispersion-mu-bins option was unexpectedly accepted");

    options.min_information = std::numeric_limits<double>::max();
    GammaPoissonDispersionEstimator insufficient(model, options);
    insufficient.accumulate(DocumentView(documents));
    const GammaPoissonDispersionResult insufficient_result = insufficient.finish();
    require(insufficient_result.diagnostics[0].status
                == GAMMA_POIS_DISPERSION_INSUFFICIENT
            && !std::isfinite(insufficient_result.diagnostics[0].delta_raw)
            && insufficient_result.diagnostics[0].delta_shrunk
                == options.delta_min,
        "insufficient dispersion information did not use the lower-bound trend");

    options.min_information = 1e-6;
    const std::vector<double> positive_weights{0.5, 2.0};
    model.set_training_calibration(
        feature_sums, positive_weights, true);
    std::vector<Document> weighted_documents = documents;
    for (Document& document : weighted_documents) {
        double raw_total = 0.0;
        for (size_t j = 0; j < document.ids.size(); ++j) {
            raw_total += document.cnts[j];
            document.cnts[j] *= positive_weights[document.ids[j]];
        }
        document.raw_ct_tot = raw_total;
        document.ct_tot = -1.0;
        document.counts_weighted = true;
    }
    sum_z.setZero();
    sum_zz.setZero();
    sum_variance.setZero();
    std::fill(sum_y.begin(), sum_y.end(), 0.0);
    std::fill(sum_y2.begin(), sum_y2.end(), 0.0);
    for (size_t d = 0; d < weighted_documents.size(); ++d) {
        GammaPoissonDocumentPosterior posterior;
        model.infer_document_posterior(weighted_documents[d], posterior);
        const VectorXd z = posterior.exposure
            * (posterior.shape.array() / posterior.rate.array()).matrix();
        sum_z += z;
        sum_zz += z * z.transpose();
        sum_variance += posterior.exposure * posterior.exposure
            * (posterior.shape.array()
                / posterior.rate.array().square()).matrix();
        for (size_t j = 0; j < documents[d].ids.size(); ++j) {
            const int32_t w = documents[d].ids[j];
            const double raw_y = documents[d].cnts[j];
            sum_y[w] += raw_y;
            sum_y2[w] += raw_y * raw_y;
        }
    }
    const VectorXd weighted_mean = beta.transpose() * sum_z;
    const MatrixXd weighted_cross = sum_zz * beta;
    GammaPoissonDispersionEstimator weighted_estimator(model, options);
    weighted_estimator.accumulate(DocumentView(weighted_documents));
    const GammaPoissonDispersionResult weighted_result =
        weighted_estimator.finish();
    for (int32_t w = 0; w < 2; ++w) {
        const double inverse_weight = 1.0 / positive_weights[w];
        const double raw_mean = weighted_mean(w) * inverse_weight;
        const double raw_q = (beta.col(w).dot(weighted_cross.col(w))
                + (sum_variance.array()
                    * beta.col(w).array().square()).sum())
            * inverse_weight * inverse_weight;
        const double gain = sum_y[w] / raw_mean;
        const double expected = (sum_y2[w] - sum_y[w])
            / (gain * gain * raw_q) - 1.0;
        require(std::abs(weighted_result.diagnostics[w].information - raw_q)
                    < 1e-10 * std::max(1.0, raw_q),
            "weighted Gamma-Poisson Q is not on the raw-count scale");
        require(std::abs(weighted_result.diagnostics[w].marginal_gain - gain)
                    < 1e-12 * std::max(1.0, gain),
            "weighted Gamma-Poisson gain is not on the raw-count scale");
        require(std::abs(weighted_result.diagnostics[w].delta_raw - expected)
                    < 1e-10 * std::max(1.0, std::abs(expected)),
            "weighted Gamma-Poisson dispersion is not on the raw-count scale");
    }

    const std::vector<double> zero_weights{0.0, 2.0};
    model.set_training_calibration(feature_sums, zero_weights, true);
    std::vector<Document> zero_weight_documents = documents;
    for (Document& document : zero_weight_documents) {
        double raw_total = 0.0;
        for (size_t j = 0; j < document.ids.size(); ++j) {
            raw_total += document.cnts[j];
            document.cnts[j] *= zero_weights[document.ids[j]];
        }
        document.raw_ct_tot = raw_total;
        document.ct_tot = -1.0;
        document.counts_weighted = true;
    }
    GammaPoissonDispersionEstimator zero_weight_estimator(model, options);
    zero_weight_estimator.accumulate(DocumentView(zero_weight_documents));
    const GammaPoissonDispersionResult zero_weight_result =
        zero_weight_estimator.finish();
    const auto& zero_diagnostic = zero_weight_result.diagnostics[0];
    require(zero_diagnostic.n_positive == 0
            && !std::isfinite(zero_diagnostic.information)
            && !std::isfinite(zero_diagnostic.marginal_gain)
            && !std::isfinite(zero_diagnostic.delta_raw)
            && !std::isfinite(zero_diagnostic.se_delta)
            && zero_diagnostic.delta_trend == options.delta_min
            && zero_diagnostic.delta_shrunk == options.delta_min
            && zero_diagnostic.tau == 1.0 / options.delta_min
            && zero_diagnostic.status == GAMMA_POIS_DISPERSION_INSUFFICIENT
            && !std::isfinite(zero_diagnostic.max_influence),
        "zero-weight Gamma-Poisson dispersion fallback is incorrect");

    std::vector<double> tied_sums(4, 1000.0);
    TestGammaPoissonModel tied_model(
        1, 4, 29, 1, 0,
        0.3, 0.4, 0.3, 2.0, 3.0, -1.0,
        0.7, 10.0, 100, 20.0, false, -1.0, &tied_sums);
    tied_model.set_beta_parameters(
        MatrixXd::Constant(1, 4, 10.0), MatrixXd::Ones(1, 4));
    std::vector<Document> tied_documents(4);
    tied_documents[0].ids = {0, 1, 2, 3};
    tied_documents[0].cnts = {2.0, 3.0, 4.0, 5.0};
    tied_documents[1].ids = {0, 1, 2, 3};
    tied_documents[1].cnts = {6.0, 3.0, 7.0, 2.0};
    tied_documents[2].ids = {0, 1, 2, 3};
    tied_documents[2].cnts = {4.0, 8.0, 2.0, 9.0};
    tied_documents[3].ids = {0, 1, 2, 3};
    tied_documents[3].cnts = {10.0, 5.0, 6.0, 3.0};
    GammaPoissonDispersionOptions tied_options;
    tied_options.min_information = 1e-6;
    GammaPoissonDispersionEstimator tied_estimator(
        tied_model, tied_options);
    tied_estimator.accumulate(DocumentView(tied_documents));
    const GammaPoissonDispersionResult tied_result = tied_estimator.finish();
    for (const auto& diagnostic : tied_result.diagnostics) {
        require(std::isfinite(diagnostic.delta_trend)
                && std::isfinite(diagnostic.delta_shrunk)
                && diagnostic.delta_trend >= tied_options.delta_min,
            "tied-mean LOESS fallback produced an invalid dispersion trend");
    }
}

std::vector<ExpectedDiagnostics> gamma_poisson_references(
        const std::filesystem::path& state_path,
        const Document& document, bool use_training_prevalence = false) {
    GammaPoissonTopicModel model(
        state_path.string(), seed, 1, 0);
    model.set_svb_parameters(200, inference_tolerance);
    std::vector<Document> documents{document};
    RowMajorMatrixXd topics;
    std::vector<GammaPoissonDocumentPosterior> posteriors;
    model.transform_with_posteriors(
        DocumentView(documents), topics, posteriors);

    const GammaPoissonDocumentPosterior& posterior =
        posteriors.front();
    const VectorXd full = topics.row(0).transpose();
    const MatrixXd& beta_kernel =
        model.get_beta_allocation_kernel();
    const MatrixXd& expected_beta = model.get_expected_beta();
    const VectorXd& capacity = model.get_topic_capacity();
    const VectorXd& dispersion = model.get_feature_dispersion();
    const VectorXd theta =
        posterior.shape.array()
        / posterior.rate.array().max(1e-12);
    VectorXd theta_kernel(n_topics);
    for (int32_t k = 0; k < n_topics; ++k) {
        theta_kernel(k) = std::exp(
            psi(posterior.shape(k)) - std::log(posterior.rate(k)));
    }

    std::vector<ExpectedDiagnostics> expected(n_features);
    for (size_t j = 0; j < document.ids.size(); ++j) {
        const int32_t feature =
            static_cast<int32_t>(document.ids[j]);
        const double observed = document.cnts[j];
        VectorXd allocation =
            theta_kernel.array() * beta_kernel.col(feature).array();
        allocation /= allocation.sum();
        const double fitted_mean = posterior.exposure
            * theta.dot(expected_beta.col(feature));
        const double epsilon = (dispersion(feature) + observed)
            / (dispersion(feature) + fitted_mean);
        VectorXd deleted(n_topics);
        for (int32_t k = 0; k < n_topics; ++k) {
            deleted(k) =
                (posterior.shape(k) - observed * allocation(k))
                / (posterior.rate(k)
                    - posterior.exposure * epsilon
                        * expected_beta(k, feature))
                * capacity(k);
        }
        require((deleted.array() > 0.0).all()
                && deleted.allFinite(),
            "invalid Gamma-Poisson reference deletion");
        deleted /= deleted.sum();
        expected[feature] =
            summarize_deletion(full, deleted);
        expected[feature].pull = std::abs(observed - fitted_mean)
            * 0.5 * (allocation - deleted).cwiseAbs().sum()
            / observed;
    }
    std::vector<double> abundance;
    model.get_topic_abundance(abundance);
    VectorXd topic_reference(n_topics);
    for (int32_t k = 0; k < n_topics; ++k) {
        topic_reference(k) = abundance[static_cast<size_t>(k)];
    }
    add_cofeature_references(
        expected_beta,
        use_training_prevalence ? topic_reference : full,
        document, expected);
    return expected;
}

std::vector<ExpectedVarianceDiagnostics>
gamma_poisson_variance_references(
        const std::filesystem::path& state_path,
        const std::vector<Document>& inference_documents,
        const std::vector<Document>* raw_documents = nullptr) {
    GammaPoissonTopicModel model(
        state_path.string(), seed, 1, 0);
    model.set_svb_parameters(200, inference_tolerance);
    RowMajorMatrixXd topics;
    std::vector<GammaPoissonDocumentPosterior> posteriors;
    model.transform_with_posteriors(
        DocumentView(inference_documents), topics, posteriors);

    const MatrixXd& beta = model.get_expected_beta();
    VectorXd sum_z = VectorXd::Zero(n_topics);
    MatrixXd sum_zz = MatrixXd::Zero(n_topics, n_topics);
    VectorXd sum_theta = VectorXd::Zero(n_topics);
    VectorXd sum_theta_over_exposure = VectorXd::Zero(n_topics);
    MatrixXd sum_theta_theta = MatrixXd::Zero(n_topics, n_topics);
    double exposure_total = 0.0;
    double exposure_squared_total = 0.0;
    int64_t positive_exposure_units = 0;
    std::vector<ExpectedVarianceDiagnostics> expected(n_features);
    std::vector<double> feature_total(n_features, 0.0);

    const std::vector<Document>& raw = raw_documents == nullptr
        ? inference_documents : *raw_documents;
    require(raw.size() == inference_documents.size(),
        "Gamma-Poisson raw reference documents do not align");
    for (size_t d = 0; d < inference_documents.size(); ++d) {
        const auto& posterior = posteriors[d];
        const VectorXd theta = posterior.shape.array()
            / posterior.rate.array().max(1e-12);
        const VectorXd z = posterior.exposure * theta;
        sum_z += z;
        sum_zz.noalias() += z * z.transpose();
        exposure_total += posterior.exposure;
        exposure_squared_total +=
            posterior.exposure * posterior.exposure;
        if (posterior.exposure > 0.0) {
            sum_theta += theta;
            sum_theta_over_exposure += theta / posterior.exposure;
            sum_theta_theta.noalias() += theta * theta.transpose();
            ++positive_exposure_units;
        }
        const Document& document = raw[d];
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const int32_t w = static_cast<int32_t>(document.ids[j]);
            const double observed = document.cnts[j];
            feature_total[w] += observed;
            expected[w].factorial += observed * (observed - 1.0);
        }
    }

    const VectorXd predicted = beta.transpose() * sum_z;
    const MatrixXd topic_cross = sum_zz * beta;
    const MatrixXd rate_cross = sum_theta_theta * beta;
    for (int32_t w = 0; w < n_features; ++w) {
        const double feature_weight = model.feature_weights_active()
            ? model.get_feature_weight()[static_cast<size_t>(w)] : 1.0;
        if (!(exposure_total > 0.0)) continue;
        const double q0 = feature_total[w] * feature_total[w]
            * exposure_squared_total
            / (exposure_total * exposure_total);
        expected[w].depth_second_moment = q0;
        if (!(feature_weight > 0.0)) continue;
        const double raw_predicted = predicted(w) / feature_weight;
        if (!(raw_predicted > 0.0)) continue;
        const double gain = feature_total[w] / raw_predicted;
        const double topic_q = beta.col(w).dot(topic_cross.col(w))
            / (feature_weight * feature_weight);
        const double qa = gain * gain * topic_q;
        expected[w].adjusted_topic_second_moment = qa;
        const double total_extra = expected[w].factorial - q0;
        if (total_extra > 0.0) {
            expected[w].excess_variance_explained_by_structure =
                (qa - q0) / total_extra;
        }
        if (!(qa > 0.0) || positive_exposure_units <= 0) continue;
        const double adjusted_rate_q = gain * gain
            * beta.col(w).dot(rate_cross.col(w))
            / (feature_weight * feature_weight);
        const double adjusted_rate_mean = gain
            * beta.col(w).dot(sum_theta) / feature_weight;
        double rate_variance = adjusted_rate_q
            - adjusted_rate_mean * adjusted_rate_mean
                / static_cast<double>(positive_exposure_units);
        if (rate_variance < 0.0
                && rate_variance >= -1e-12
                    * std::max(1.0, std::abs(adjusted_rate_q))) {
            rate_variance = 0.0;
        }
        const double poisson_variance = gain
            * beta.col(w).dot(sum_theta_over_exposure) / feature_weight;
        const double phi = std::max(expected[w].factorial / qa - 1.0, 0.0);
        const double denominator = rate_variance + poisson_variance
            + phi * adjusted_rate_q;
        if (rate_variance >= 0.0 && poisson_variance >= 0.0
                && denominator > 0.0) {
            expected[w].total_variance_explained_by_structure =
                rate_variance / denominator;
        }
    }
    return expected;
}

void validate_variance_diagnostics(const FeatureTable& table,
        const std::vector<ExpectedVarianceDiagnostics>& expected,
        const std::string& model_name, bool expect_uncertainty = false,
        bool require_negative_phi_floor = false) {
    std::vector<std::string> columns{
        "F_w", "Qa_w", "Q0_w", "EVES_w", "TVES_w"};
    if (expect_uncertainty) {
        columns.push_back("U_w");
    }
    std::vector<size_t> indices(columns.size());
    for (size_t i = 0; i < columns.size(); ++i) {
        indices[i] = column_index(table, columns[i]);
    }
    bool tested_negative_phi_floor = false;
    for (int32_t w = 0; w < n_features; ++w) {
        const auto& reference = expected[w];
        const auto& row = table.rows.at("feature_" + std::to_string(w));
        std::vector<double> values{
            reference.factorial,
            reference.adjusted_topic_second_moment,
            reference.depth_second_moment,
            reference.excess_variance_explained_by_structure,
            reference.total_variance_explained_by_structure,
        };
        if (expect_uncertainty) {
            values.push_back(reference.uncertainty);
        }
        for (size_t i = 0; i < values.size(); ++i) {
            if (std::isnan(values[i])) {
                require(row[indices[i]] == "NA",
                    "undefined variance diagnostic is not NA");
                continue;
            }
            const double observed = std::stod(row[indices[i]]);
            const double tolerance = i < 3 || i == 5 ? 1e-12 : 6e-5;
            require(std::abs(observed - values[i])
                        <= tolerance * std::max(1.0, std::abs(values[i])),
                model_name + " variance diagnostic differs from reference");
        }
        if (reference.adjusted_topic_second_moment > 0.0
                && reference.factorial
                    / reference.adjusted_topic_second_moment - 1.0 < 0.0
                && std::isfinite(
                    reference.total_variance_explained_by_structure)
                && reference.total_variance_explained_by_structure > 0.0) {
            tested_negative_phi_floor = true;
        }
        if (reference.adjusted_topic_second_moment > 0.0
                && reference.depth_second_moment > 0.0) {
            const double reconstructed_phi = std::stod(row[indices[0]])
                / std::stod(row[indices[1]]) - 1.0;
            const double reconstructed_phi0 = std::stod(row[indices[0]])
                / std::stod(row[indices[2]]) - 1.0;
            require(std::abs(reconstructed_phi
                        - (reference.factorial
                            / reference.adjusted_topic_second_moment - 1.0))
                        < 1e-12 * std::max(1.0, std::abs(reconstructed_phi)),
                "final-model phi cannot be reconstructed from output");
            require(std::abs(reconstructed_phi0
                        - (reference.factorial
                            / reference.depth_second_moment - 1.0))
                        < 1e-12 * std::max(1.0, std::abs(reconstructed_phi0)),
                "depth-null phi cannot be reconstructed from output");
        }
    }
    if (require_negative_phi_floor) {
        require(tested_negative_phi_floor,
            "variance fixture does not exercise the TVES negative-phi floor");
    }
}

void validate_tables(const FeatureTable& full,
        const FeatureTable& cheap,
        const std::vector<ExpectedDiagnostics>& expected,
        const std::string& model_name, bool variance_diagnostics = false,
        bool uncertainty_diagnostics = false) {
    std::vector<std::string> common_header{
        "Feature", "absDiff", "absDiffRate", "totCount", "nUnits",
        "log2Gain", "marginalDev", "conditionalDev", "factorDrift",
        "deletionTV", "topicInformation",
        "cofeatureCorroboration", "cofeatureConflict",
    };
    if (variance_diagnostics) {
        common_header.insert(common_header.end(),
            {"F_w", "Qa_w", "Q0_w", "EVES_w", "TVES_w"});
        if (uncertainty_diagnostics) {
            common_header.push_back("U_w");
        }
    }
    std::vector<std::string> full_header = common_header;
    full_header.push_back("adjAbsDiffRate");
    full_header.push_back("pull");
    require(full.header == full_header,
        model_name + " full diagnostics header is incorrect");
    require(cheap.header == common_header,
        model_name + " cheap diagnostics header is incorrect");
    require(full.rows.size() == n_features
            && cheap.rows.size() == n_features,
        model_name + " output has the wrong number of features");

    for (int32_t feature = 0; feature < n_features; ++feature) {
        const std::string name = "feature_" + std::to_string(feature);
        require(full.rows.count(name) == 1
                && cheap.rows.count(name) == 1,
            model_name + " output is missing " + name);
        const std::vector<std::string>& full_row = full.rows.at(name);
        const std::vector<std::string>& cheap_row = cheap.rows.at(name);
        require(full_row.size() == full_header.size()
                && cheap_row.size() == common_header.size(),
            model_name + " output row has the wrong width");
        require(std::equal(
                cheap_row.begin(), cheap_row.end(), full_row.begin()),
            model_name + " cheap diagnostics differ from full diagnostics");

        if (!expected[feature].supported) {
            require(full_row[9] == "NA"
                    && full_row[11] == "NA"
                    && full_row[12] == "NA",
                model_name
                + " zero-support contextual diagnostics are not NA");
            require(std::abs(std::stod(full_row[10])
                    - expected[feature].topic_information) < 6e-5,
                model_name
                + " zero-support topic information is incorrect");
            continue;
        }

        const std::array<double, 4> references{
            expected[feature].total_variation,
            expected[feature].topic_information,
            expected[feature].cofeature_corroboration,
            expected[feature].cofeature_conflict,
        };
        for (size_t offset = 0; offset < references.size(); ++offset) {
            if (std::isnan(references[offset])) {
                require(full_row[9 + offset] == "NA",
                    model_name
                    + " undefined cofeature diagnostic is not NA");
                continue;
            }
            const double observed =
                std::stod(full_row[9 + offset]);
            const double tolerance = 6e-5
                * std::max(1.0, std::abs(references[offset]));
            require(std::abs(observed - references[offset]) < tolerance,
                model_name
                + " feature diagnostic differs from its direct reference");
        }
    }
}

void validate_training_table(const FeatureTable& table,
        const std::vector<ExpectedDiagnostics>& expected,
        const std::string& model_name, bool variance_diagnostics = false,
        bool uncertainty_diagnostics = false) {
    std::vector<std::string> header{
        "Feature", "absDiff", "absDiffRate", "totCount", "nUnits",
        "log2Gain", "marginalDev", "conditionalDev", "factorDrift",
        "deletionTV", "topicInformation",
        "cofeatureCorroboration", "cofeatureConflict",
    };
    if (variance_diagnostics) {
        header.insert(header.end(),
            {"F_w", "Qa_w", "Q0_w", "EVES_w", "TVES_w"});
        if (uncertainty_diagnostics) {
            header.push_back("U_w");
        }
    }
    header.push_back("pull");
    require(table.header == header,
        model_name + " training diagnostic header is incorrect");
    require(table.rows.size() == n_features,
        model_name + " training output has the wrong number of features");
    for (int32_t feature = 0; feature < n_features; ++feature) {
        const std::string name = "feature_" + std::to_string(feature);
        require(table.rows.count(name) == 1,
            model_name + " training output is missing " + name);
        const std::vector<std::string>& row = table.rows.at(name);
        require(row.size() == header.size(),
            model_name + " training output row has the wrong width");
        if (!expected[feature].supported) {
            require(row[9] == "NA" && row[11] == "NA"
                    && row[12] == "NA" && row.back() == "NA",
                model_name + " unsupported training diagnostics are not NA");
            require(std::abs(std::stod(row[10])
                    - expected[feature].topic_information) < 6e-5,
                model_name + " training topic information is incorrect");
            continue;
        }
        const std::array<double, 4> references{
            expected[feature].total_variation,
            expected[feature].topic_information,
            expected[feature].cofeature_corroboration,
            expected[feature].cofeature_conflict,
        };
        for (size_t offset = 0; offset < references.size(); ++offset) {
            const double observed = std::stod(row[9 + offset]);
            const double tolerance = 6e-5
                * std::max(1.0, std::abs(references[offset]));
            require(std::abs(observed - references[offset]) < tolerance,
                model_name + " training diagnostic differs from reference");
        }
        const double pull = std::stod(row.back());
        require(std::abs(pull - expected[feature].pull)
                    < 6e-5 * std::max(1.0, std::abs(expected[feature].pull)),
            model_name + " training pull differs from reference");
    }
}

std::string metadata_json(int32_t units = 1) {
    std::ostringstream output;
    output
        << "{\"n_units\":" << units
        << ",\"n_modalities\":1,\"n_features\":"
        << n_features
        << ",\"offset_data\":1,\"header_info\":[\"document\"],"
        << "\"dictionary\":{";
    for (int32_t feature = 0; feature < n_features; ++feature) {
        if (feature > 0) {
            output << ",";
        }
        output << "\"feature_" << feature << "\":" << feature;
    }
    output << "}}";
    return output.str();
}

void remove_transform_outputs(const std::filesystem::path& prefix) {
    for (const std::string& suffix : {
            ".results.tsv", ".pseudobulk.tsv", ".unit_stats.tsv",
            ".feature_residuals.tsv"}) {
        std::filesystem::remove(prefix.string() + suffix);
    }
}

void validate_unit_averaged_deletion(const FeatureTable& table,
        const std::vector<std::vector<ExpectedDiagnostics>>& expected,
        const std::vector<Document>& documents,
        const std::string& model_name) {
    require(expected.size() == documents.size(),
        model_name + " deletion reference size mismatch");
    bool distinguishes_from_count_weighting = false;
    for (int32_t feature = 0; feature < n_features; ++feature) {
        double unit_sum = 0.0;
        double count_weighted_sum = 0.0;
        double total_count = 0.0;
        int64_t units = 0;
        for (size_t d = 0; d < documents.size(); ++d) {
            if (!expected[d][feature].supported) continue;
            double count = 0.0;
            for (size_t j = 0; j < documents[d].ids.size(); ++j) {
                if (documents[d].ids[j]
                        == static_cast<uint32_t>(feature)) {
                    count = documents[d].cnts[j];
                    break;
                }
            }
            require(count > 0.0,
                model_name + " deletion reference lacks a positive count");
            const double variation =
                expected[d][feature].total_variation;
            unit_sum += variation;
            count_weighted_sum += count * variation;
            total_count += count;
            ++units;
        }

        const std::string name = "feature_" + std::to_string(feature);
        require(table.rows.count(name) == 1,
            model_name + " multi-unit output is missing " + name);
        const std::vector<std::string>& row = table.rows.at(name);
        if (units == 0) {
            require(row[9] == "NA",
                model_name + " unsupported deletionTV is not NA");
            continue;
        }
        require(std::stoll(row[4]) == units,
            model_name + " deletionTV unit denominator is incorrect");
        const double unit_average = unit_sum / static_cast<double>(units);
        const double count_average = count_weighted_sum / total_count;
        const double tolerance =
            6e-5 * std::max(1.0, std::abs(unit_average));
        require(std::abs(std::stod(row[9]) - unit_average) < tolerance,
            model_name + " deletionTV is not averaged over expressing units");
        if (std::abs(unit_average - count_average) > 4.0 * tolerance) {
            distinguishes_from_count_weighting = true;
        }
    }
    require(distinguishes_from_count_weighting,
        model_name + " deletion test does not distinguish unit and count weighting");
}

void test_lda_output() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path()
        / "punkst_feature_diagnostics_lda";
    const std::filesystem::path model_path =
        base.string() + ".model.tsv";
    const std::filesystem::path input_path =
        base.string() + ".units.tsv";
    const std::filesystem::path metadata_path =
        base.string() + ".meta.json";
    const std::filesystem::path full_prefix =
        base.string() + ".full";
    const std::filesystem::path cheap_prefix =
        base.string() + ".cheap";
    const std::filesystem::path training_prefix =
        base.string() + ".training";
    const std::filesystem::path multi_prefix =
        base.string() + ".multi";
    const std::filesystem::path weighted_prefix =
        base.string() + ".weighted";
    const std::filesystem::path weighted_all_prefix =
        base.string() + ".weighted_all";
    const std::filesystem::path feature_path =
        base.string() + ".features.tsv";
    const std::filesystem::path temp_parent =
        base.string() + ".tmp";
    std::filesystem::remove_all(temp_parent);

    MatrixXd lambda(n_topics, n_features);
    lambda <<
        20.0, 1.0, 5.0, 2.0, 1.0, 1.0,
        1.0, 20.0, 5.0, 2.0, 1.0, 1.0,
        1.0, 1.0, 5.0, 2.0, 20.0, 1.0;
    std::ostringstream model_text;
    model_text << "Feature\tTopic0\tTopic1\tTopic2\n";
    for (int32_t feature = 0; feature < n_features; ++feature) {
        model_text << "feature_" << feature;
        for (int32_t topic = 0; topic < n_topics; ++topic) {
            model_text << "\t" << lambda(topic, feature);
        }
        model_text << "\n";
    }
    write_text(model_path, model_text.str());
    write_text(input_path,
        "doc_0\t5\t20\t0 8\t1 2\t2 3\t3 1\t4 6\n");
    write_text(metadata_path, metadata_json());

    Document document;
    document.ids = {0, 1, 2, 3, 4};
    document.cnts = {8.0, 2.0, 3.0, 1.0, 6.0};
    document.raw_ct_tot = document.ct_tot = 20.0;
    const std::vector<ExpectedDiagnostics> expected =
        lda_references(model_path, document);
    const std::vector<ExpectedDiagnostics> training_expected =
        lda_references(model_path, document, true);

    auto run = [&](const std::filesystem::path& prefix, bool cheap,
            bool training, bool weighted = false,
            bool all_features = false) {
        std::vector<std::string> arguments{
            "lda-transform",
            "--in-data", input_path.string(),
            "--in-meta", metadata_path.string(),
            "--in-model", model_path.string(),
            "--out-prefix", prefix.string(),
            "--min-count", "1",
            "--minibatch-size", "1",
            "--threads", "1",
            "--temp-dir", temp_parent.string(),
            "--seed", std::to_string(seed),
            "--max-iter", "200",
            "--mean-change-tol", "1e-10",
            "--residuals",
            "--unit-diagnostics-similarity",
        };
        if (cheap) {
            arguments.push_back("--feature-diagnostics-cheap");
            arguments.push_back("--pseudobulk-all-features");
        }
        if (weighted) {
            arguments.insert(arguments.end(), {
                "--features", feature_path.string(),
                "--icol-weight", "2",
            });
        }
        if (all_features && !cheap) {
            arguments.push_back("--pseudobulk-all-features");
        }
        if (training) {
            arguments.push_back("--use-training-prevalence");
        }
        require(run_command(cmdLDATransform, std::move(arguments)) == 0,
            "LDA transform failed");
    };
    run(full_prefix, false, false);
    run(cheap_prefix, true, false);
    run(training_prefix, true, true);
    require(std::filesystem::is_directory(temp_parent)
            && std::filesystem::is_empty(temp_parent),
        "LDA diagnostic temporary files were not cleaned up");
    validate_tables(
        read_feature_table(
            full_prefix.string() + ".feature_residuals.tsv"),
        read_feature_table(
            cheap_prefix.string() + ".feature_residuals.tsv"),
        expected, "LDA", true, true);
    validate_training_table(
        read_feature_table(
            training_prefix.string() + ".feature_residuals.tsv"),
        training_expected, "LDA", true, true);

    Document second_document;
    second_document.ids = {0, 1, 2, 3, 4};
    second_document.cnts = {2.0, 14.0, 3.0, 9.0, 2.0};
    second_document.raw_ct_tot = second_document.ct_tot = 30.0;
    write_text(input_path,
        "doc_0\t5\t20\t0 8\t1 2\t2 3\t3 1\t4 6\n"
        "doc_1\t5\t30\t0 2\t1 14\t2 3\t3 9\t4 2\n");
    write_text(metadata_path, metadata_json(2));
    run(multi_prefix, false, false);
    const FeatureTable multi_table = read_feature_table(
        multi_prefix.string() + ".feature_residuals.tsv");
    validate_unit_averaged_deletion(
        multi_table,
        {expected, lda_references(model_path, second_document)},
        {document, second_document}, "LDA");
    validate_variance_diagnostics(multi_table,
        lda_variance_references(
            model_path, {document, second_document}),
        "LDA", true);

    write_text(feature_path,
        "#feature\ttotal\tweight\n"
        "feature_0\t100\t0\n"
        "feature_1\t100\t0.5\n"
        "feature_2\t100\t2\n"
        "feature_3\t100\t1\n"
        "feature_4\t100\t1.5\n"
        "feature_5\t100\t1\n");
    const std::vector<Document> raw_documents{document, second_document};
    std::vector<Document> weighted_documents = raw_documents;
    const std::array<double, n_features> weights{
        0.0, 0.5, 2.0, 1.0, 1.5, 1.0};
    for (Document& weighted_document : weighted_documents) {
        for (size_t j = 0; j < weighted_document.ids.size(); ++j) {
            weighted_document.cnts[j] *=
                weights[weighted_document.ids[j]];
        }
        weighted_document.ct_tot = -1.0;
        weighted_document.counts_weighted = true;
    }
    run(weighted_prefix, false, false, true, false);
    run(weighted_all_prefix, false, false, true, true);
    const FeatureTable weighted_table = read_feature_table(
        weighted_prefix.string() + ".feature_residuals.tsv");
    const FeatureTable weighted_all_table = read_feature_table(
        weighted_all_prefix.string() + ".feature_residuals.tsv");
    validate_variance_diagnostics(weighted_table,
        lda_variance_references(model_path, weighted_documents,
            &raw_documents),
        "weighted LDA", true);
    for (const std::string& column : {
            "F_w", "Qa_w", "Q0_w", "EVES_w", "TVES_w", "U_w"}) {
        const size_t weighted_index = column_index(weighted_table, column);
        const size_t all_index = column_index(weighted_all_table, column);
        for (int32_t w = 0; w < n_features; ++w) {
            const std::string name = "feature_" + std::to_string(w);
            require(weighted_table.rows.at(name)[weighted_index]
                        == weighted_all_table.rows.at(name)[all_index],
                "LDA variance diagnostics changed with pseudobulk mode");
        }
    }

    remove_transform_outputs(full_prefix);
    remove_transform_outputs(cheap_prefix);
    remove_transform_outputs(training_prefix);
    remove_transform_outputs(multi_prefix);
    remove_transform_outputs(weighted_prefix);
    remove_transform_outputs(weighted_all_prefix);
    std::filesystem::remove(model_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(metadata_path);
    std::filesystem::remove(feature_path);
    std::filesystem::remove(temp_parent);
}

void test_gamma_poisson_output() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path()
        / "punkst_feature_diagnostics_gamma_pois";
    const std::filesystem::path state_path =
        base.string() + ".state.tsv";
    const std::filesystem::path input_path =
        base.string() + ".units.tsv";
    const std::filesystem::path metadata_path =
        base.string() + ".meta.json";
    const std::filesystem::path full_prefix =
        base.string() + ".full";
    const std::filesystem::path cheap_prefix =
        base.string() + ".cheap";
    const std::filesystem::path training_prefix =
        base.string() + ".training";
    const std::filesystem::path multi_prefix =
        base.string() + ".multi";
    const std::filesystem::path weighted_state_path =
        base.string() + ".weighted.state.tsv";
    const std::filesystem::path weighted_prefix =
        base.string() + ".weighted";
    const std::filesystem::path weighted_all_prefix =
        base.string() + ".weighted_all";
    const std::filesystem::path temp_parent =
        base.string() + ".tmp";
    std::filesystem::remove_all(temp_parent);

    std::vector<double> feature_sums(n_features, 100.0);
    TestGammaPoissonModel model(
        n_topics, n_features, 17, 1, 0,
        0.3, 0.4, 0.3, 2.0, 3.0, -1.0,
        0.7, 10.0, 100, 20.0, false, -1.0, &feature_sums);
    MatrixXd shape(n_topics, n_features);
    shape <<
        20.0, 1.0, 5.0, 2.0, 1.0, 1.0,
        1.0, 20.0, 5.0, 2.0, 1.0, 1.0,
        1.0, 1.0, 5.0, 2.0, 20.0, 1.0;
    model.set_beta_parameters(
        shape, MatrixXd::Constant(n_topics, n_features, 1.0));
    model.set_feature_dispersion(
        std::vector<double>(n_features, 4.0));
    std::vector<std::string> names(n_features);
    for (int32_t feature = 0; feature < n_features; ++feature) {
        names[feature] = "feature_" + std::to_string(feature);
    }
    model.write_state(state_path.string(), names);
    const std::vector<double> weights{0.0, 0.5, 2.0, 1.0, 1.5, 1.0};
    model.set_training_calibration(feature_sums, weights, true);
    model.write_state(weighted_state_path.string(), names);
    write_text(input_path,
        "doc_0\t5\t20\t0 8\t1 2\t2 3\t3 1\t4 6\n");
    write_text(metadata_path, metadata_json());

    Document document;
    document.ids = {0, 1, 2, 3, 4};
    document.cnts = {8.0, 2.0, 3.0, 1.0, 6.0};
    document.raw_ct_tot = document.ct_tot = 20.0;
    const std::vector<ExpectedDiagnostics> expected =
        gamma_poisson_references(state_path, document);
    const std::vector<ExpectedDiagnostics> training_expected =
        gamma_poisson_references(state_path, document, true);

    auto run = [&](const std::filesystem::path& prefix, bool cheap,
            bool training, bool weighted = false,
            bool all_features = false) {
        std::vector<std::string> arguments{
            "gamma-pois-transform",
            "--in-data", input_path.string(),
            "--in-meta", metadata_path.string(),
            "--in-state", weighted
                ? weighted_state_path.string() : state_path.string(),
            "--out-prefix", prefix.string(),
            "--min-count", "1",
            "--minibatch-size", "1",
            "--threads", "1",
            "--temp-dir", temp_parent.string(),
            "--seed", std::to_string(seed),
            "--max-iter", "200",
            "--mean-change-tol", "1e-10",
            "--use-stored-dispersion",
            "--residuals",
            "--unit-diagnostics-similarity",
        };
        if (cheap) {
            arguments.push_back("--feature-diagnostics-cheap");
            arguments.push_back("--pseudobulk-all-features");
        }
        if (training) {
            arguments.push_back("--use-training-prevalence");
        }
        if (all_features && !cheap) {
            arguments.push_back("--pseudobulk-all-features");
        }
        require(run_command(
                cmdGammaPoisTransform, std::move(arguments)) == 0,
            "Gamma-Poisson transform failed");
    };
    run(full_prefix, false, false);
    run(cheap_prefix, true, false);
    run(training_prefix, true, true);
    require(std::filesystem::is_directory(temp_parent)
            && std::filesystem::is_empty(temp_parent),
        "Gamma-Poisson diagnostic temporary files were not cleaned up");
    const FeatureTable full_table = read_feature_table(
        full_prefix.string() + ".feature_residuals.tsv");
    const FeatureTable cheap_table = read_feature_table(
        cheap_prefix.string() + ".feature_residuals.tsv");
    const FeatureTable training_table = read_feature_table(
        training_prefix.string() + ".feature_residuals.tsv");
    validate_tables(full_table, cheap_table,
        expected, "Gamma-Poisson", true);
    validate_training_table(training_table,
        training_expected, "Gamma-Poisson", true);
    for (const std::string& column : {
            "F_w", "Qa_w", "Q0_w", "EVES_w", "TVES_w"}) {
        const size_t full_index = column_index(full_table, column);
        const size_t training_index = column_index(training_table, column);
        for (int32_t w = 0; w < n_features; ++w) {
            const std::string name = "feature_" + std::to_string(w);
            require(full_table.rows.at(name)[full_index]
                        == training_table.rows.at(name)[training_index],
                "training prevalence changed final-model variance diagnostics");
        }
    }

    Document second_document;
    second_document.ids = {0, 1, 2, 3, 4};
    second_document.cnts = {1.0, 10.0, 2.0, 6.0, 1.0};
    second_document.raw_ct_tot = second_document.ct_tot = 20.0;
    write_text(input_path,
        "doc_0\t5\t20\t0 8\t1 2\t2 3\t3 1\t4 6\n"
        "doc_1\t5\t20\t0 1\t1 10\t2 2\t3 6\t4 1\n");
    write_text(metadata_path, metadata_json(2));
    run(multi_prefix, false, false);
    const FeatureTable multi_table = read_feature_table(
        multi_prefix.string() + ".feature_residuals.tsv");
    validate_unit_averaged_deletion(
        multi_table,
        {expected,
            gamma_poisson_references(state_path, second_document)},
        {document, second_document}, "Gamma-Poisson");
    validate_variance_diagnostics(multi_table,
        gamma_poisson_variance_references(
            state_path, {document, second_document}),
        "Gamma-Poisson", false, true);

    const std::vector<Document> raw_documents{document, second_document};
    std::vector<Document> weighted_documents = raw_documents;
    for (Document& weighted_document : weighted_documents) {
        for (size_t j = 0; j < weighted_document.ids.size(); ++j) {
            weighted_document.cnts[j] *=
                weights[weighted_document.ids[j]];
        }
        weighted_document.ct_tot = -1.0;
        weighted_document.counts_weighted = true;
    }
    run(weighted_prefix, false, false, true, false);
    run(weighted_all_prefix, false, false, true, true);
    const FeatureTable weighted_table = read_feature_table(
        weighted_prefix.string() + ".feature_residuals.tsv");
    const FeatureTable weighted_all_table = read_feature_table(
        weighted_all_prefix.string() + ".feature_residuals.tsv");
    validate_variance_diagnostics(weighted_table,
        gamma_poisson_variance_references(
            weighted_state_path, weighted_documents, &raw_documents),
        "weighted Gamma-Poisson");
    for (const std::string& column : {
            "F_w", "Qa_w", "Q0_w", "EVES_w", "TVES_w"}) {
        const size_t weighted_index = column_index(weighted_table, column);
        const size_t all_index = column_index(weighted_all_table, column);
        for (int32_t w = 0; w < n_features; ++w) {
            const std::string name = "feature_" + std::to_string(w);
            require(weighted_table.rows.at(name)[weighted_index]
                        == weighted_all_table.rows.at(name)[all_index],
                "Gamma-Poisson variance diagnostics changed with pseudobulk mode");
        }
    }

    remove_transform_outputs(full_prefix);
    remove_transform_outputs(cheap_prefix);
    remove_transform_outputs(training_prefix);
    remove_transform_outputs(multi_prefix);
    remove_transform_outputs(weighted_prefix);
    remove_transform_outputs(weighted_all_prefix);
    std::filesystem::remove(state_path);
    std::filesystem::remove(weighted_state_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(metadata_path);
    std::filesystem::remove(temp_parent);
}

void remove_uac_outputs(const std::filesystem::path& prefix) {
    for (const std::string& suffix : {
            ".state.tsv", ".model.tsv", ".results.tsv",
            ".diagnostics.tsv", ".trace.tsv", ".separation.tsv",
            ".representatives.tsv"}) {
        std::filesystem::remove(prefix.string() + suffix);
    }
}

void test_topic_to_uac_handoff() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path()
        / "punkst_topic_to_uac";
    const std::filesystem::path model_path =
        base.string() + ".topic.model.tsv";
    const std::filesystem::path center_path =
        base.string() + ".topic.results.tsv";
    const std::filesystem::path input_path =
        base.string() + ".units.tsv";
    const std::filesystem::path metadata_path =
        base.string() + ".meta.json";
    const std::filesystem::path fit_prefix =
        base.string() + ".fit";
    const std::filesystem::path transform_prefix =
        base.string() + ".transform";

    write_text(model_path,
        "Feature\tTopic0\tTopic1\n"
        "feature_0\t20\t1\n"
        "feature_1\t1\t20\n"
        "feature_2\t9\t9\n");
    write_text(metadata_path,
        "{\"n_units\":12,\"n_modalities\":1,\"n_features\":2,"
        "\"offset_data\":2,\"header_info\":[\"batch\",\"document\"],"
        "\"dictionary\":{\"feature_0\":0,\"feature_1\":1}}");
    std::ostringstream centers, counts;
    centers << "#batch\tdocument\tTopic1\tTopic0\n";
    for (int32_t document = 0; document < 12; ++document) {
        const bool first = document < 6;
        centers << "batch_" << (document % 2) << "\tdoc_" << document
            << "\t" << (first ? 0.05 : 0.95)
            << "\t" << (first ? 0.95 : 0.05) << "\n";
        counts << "batch_" << (document % 2) << "\tdoc_" << document
            << "\t2\t10\t0 " << (first ? 9 : 1)
            << "\t1 " << (first ? 1 : 9) << "\n";
    }
    write_text(center_path, centers.str());
    write_text(input_path, counts.str());

    std::vector<std::string> fit_arguments{
        "uac-fit",
        "--in-topic-center", center_path.string(),
        "--unit-icol-id", "1",
        "--in-model", model_path.string(),
        "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(),
        "--count-icol-id", "1",
        "--out-prefix", fit_prefix.string(),
        "--n-clusters", "2",
        "--particles", "16",
        "--particle-em-fixed-iterations", "1",
        "--kmeans-starts", "1",
        "--max-iter", "2",
        "--cluster-covariance-rank", "0",
        "--threads", "1",
        "--n-representatives", "1",
        "--seed", "43",
    };
    require(run_command(cmdUacFit, std::move(fit_arguments)) == 0,
        "direct topic-to-UAC fit failed");

    std::ifstream state_input(fit_prefix.string() + ".state.tsv");
    std::string state_header;
    require(static_cast<bool>(std::getline(state_input, state_header))
            && state_header == "##punkst_uac_state_v11",
        "direct topic-to-UAC fit did not write a v11 state");

    std::vector<std::string> transform_arguments{
        "uac-transform",
        "--in-state", fit_prefix.string() + ".state.tsv",
        "--in-topic-center", center_path.string(),
        "--unit-icol-id", "1",
        "--in-model", model_path.string(),
        "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(),
        "--count-icol-id", "1",
        "--out-prefix", transform_prefix.string(),
        "--particles", "16",
        "--threads", "1",
        "--n-representatives", "1",
    };
    require(run_command(cmdUacTransform,
            std::move(transform_arguments)) == 0,
        "direct topic-to-UAC transform failed");

    std::ifstream result_input(
        transform_prefix.string() + ".results.tsv");
    std::string line;
    int32_t rows = -1;
    while (std::getline(result_input, line)) ++rows;
    require(rows == 12,
        "direct topic-to-UAC transform wrote the wrong row count");

    remove_uac_outputs(fit_prefix);
    remove_uac_outputs(transform_prefix);
    std::filesystem::remove(model_path);
    std::filesystem::remove(center_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(metadata_path);
}

} // namespace

int32_t test(int32_t, char**) {
    try {
        test_cofeature_metrics();
        test_batched_topic_moments();
        test_gamma_poisson_dispersion_all_cell_moments();
        test_lda_output();
        test_gamma_poisson_output();
        test_topic_to_uac_handoff();
        std::cout << "Feature diagnostic and topic-to-UAC tests passed\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
        return 1;
    }
    return 0;
}
