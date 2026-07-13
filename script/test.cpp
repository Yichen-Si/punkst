#include "punkst.h"
#include "dataunits.hpp"
#include "eb_topic_activity.hpp"
#include "dense_kmeans.hpp"
#include "gamma_pois_dispersion.hpp"
#include "gamma_pois_cluster.hpp"
#include "gamma_pois_posterior_io.hpp"
#include "gamma_pois_topic.hpp"
#include "low_rank_covariance.hpp"
#include "numerical_utils.hpp"
#include "tiles2bins.hpp"
#include "vst.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check_close(const std::string& name, double got, double expected,
        double abs_tol, double rel_tol = 0.0) {
    const double tol = abs_tol + rel_tol * std::abs(expected);
    if (!std::isfinite(got) || std::abs(got - expected) > tol) {
        std::cerr << std::setprecision(17) << "FAIL " << name << ": got " << got
            << ", expected " << expected << ", tol " << tol << "\n";
        ++g_failures;
    } else {
        std::cerr << "PASS " << name << "\n";
    }
}

void check_true(const std::string& name, bool ok) {
    if (!ok) {
        std::cerr << "FAIL " << name << "\n";
        ++g_failures;
    } else {
        std::cerr << "PASS " << name << "\n";
    }
}

void test_gauss_legendre16() {
    const double log_poly = GaussLegendre16::log_integrate(
        [](double x) { return 30.0 * std::log(x); }, 0.0, 1.0);
    check_close("GaussLegendre16 integrates x^30 exactly",
        std::exp(log_poly), 1.0 / 31.0, 2e-15, 1e-13);

    const double log_exp = GaussLegendre16::log_integrate(
        [](double x) { return x; }, 0.0, 1.0);
    check_close("GaussLegendre16 integrates exp(x)",
        std::exp(log_exp), std::exp(1.0) - 1.0, 2e-15, 1e-14);

    const double log_scaled = GaussLegendre16::log_integrate(
        [](double) { return 700.0; }, 2.0, 5.0);
    check_close("GaussLegendre16 stays on log scale",
        log_scaled, 700.0 + std::log(3.0), 1e-12, 0.0);

    const double log_smooth = GaussLegendre16::log_integrate(
        [](double x) { return std::log1p(x * x); }, -2.0, 3.0, 8);
    const auto antiderivative = [](double x) {
        return x + (x * x * x) / 3.0;
    };
    check_close("GaussLegendre16 composite smooth integral",
        std::exp(log_smooth), antiderivative(3.0) - antiderivative(-2.0),
        1e-13, 1e-13);
}

void test_beta_helpers_and_evidence() {
    check_close("log_beta_fn matches closed form B(2,3)",
        log_beta_fn(2.0, 3.0), std::log(1.0 / 12.0), 1e-15, 0.0);

    RowMajorMatrixXd gamma_counts(1, 3);
    gamma_counts << 2.0, 1.5, 2.5;
    std::vector<ebact::Component> comps = {
        ebact::Component::Uniform(0.0, 1.0, true),
        ebact::Component::Beta(2.0, 4.0)
    };
    const auto logH = ebact::beta_marginal_log_evidence(gamma_counts, 0, 0.5, comps, 8);
    const double A = 3.0;
    const double B = 5.0;
    check_close("beta-marginal uniform [0,1] evidence",
        logH[0][0], log_beta_fn(A, B), 2e-13, 0.0);
    check_close("beta-marginal beta slab evidence",
        logH[0][1], log_beta_fn(A + 1.0, B + 3.0) - log_beta_fn(2.0, 4.0),
        2e-13, 0.0);
}

void test_multinomial_evidence() {
    Document doc;
    doc.ids = {0};
    doc.cnts = {2.0};
    RowMajorMatrixXd q(1, 2);
    q << 0.8, 0.2;
    RowVectorXd beta(2);
    beta << 0.2, 0.8;
    std::vector<ebact::Component> comps = {
        ebact::Component::Uniform(0.0, 1.0, true),
        ebact::Component::Beta(2.0, 2.0)
    };
    const auto logH = ebact::multinomial_log_evidence({doc}, q, beta, comps, 1e-12, 8);

    // L(a) = (1 - 0.75 a)^2 for this one-feature document.
    const double uniform_expected = 1.0 - 0.75 + (0.75 * 0.75) / 3.0;
    check_close("multinomial uniform evidence analytic",
        std::exp(logH[0][0]), uniform_expected, 2e-13, 1e-13);

    // Beta(2,2) density is 6a(1-a); integrate against the same quadratic.
    const double beta_expected = 6.0 * (
        0.5 - (1.0 + 1.5) / 3.0 + (1.5 + 0.5625) / 4.0 - 0.5625 / 5.0);
    check_close("multinomial beta-slab evidence analytic",
        std::exp(logH[0][1]), beta_expected, 2e-13, 1e-13);

    const auto compMeans = ebact::multinomial_component_activity_means({doc}, q, beta, comps, 1e-12, 8);
    const double uniform_num = 0.5 - 1.5 / 3.0 + 0.5625 / 4.0;
    check_close("multinomial uniform posterior activity mean analytic",
        compMeans[0][0], uniform_num / uniform_expected, 2e-13, 1e-13);
    const double beta_num = 6.0 * (
        1.0 / 3.0 - (1.0 + 1.5) / 4.0 + (1.5 + 0.5625) / 5.0 - 0.5625 / 6.0);
    check_close("multinomial beta posterior activity mean analytic",
        compMeans[0][1], beta_num / beta_expected, 2e-13, 1e-13);

    ebact::FitResult fit;
    fit.resp = {{0.25, 0.75}};
    const auto postMean = ebact::posterior_activity_means(fit, compMeans);
    check_close("posterior activity mean responsibility weighted",
        postMean[0], 0.25 * compMeans[0][0] + 0.75 * compMeans[0][1], 1e-15, 0.0);
    check_close("single-unit posterior activity mean responsibility weighted",
        ebact::multinomial_unit_posterior_activity_mean(doc, q.row(0), beta, comps, fit.resp[0], 1e-12, 8),
        postMean[0], 2e-13, 1e-13);
    check_close("single-unit posterior activity mean reuses logH",
        ebact::multinomial_unit_posterior_activity_mean_from_logH(
            doc, q.row(0), beta, comps, fit.resp[0], logH[0], 1e-12, 8),
        postMean[0], 2e-13, 1e-13);

    std::vector<Document> docs = {doc, doc, doc, doc};
    RowMajorMatrixXd q_many(4, 2);
    q_many << 0.8, 0.2,
        0.7, 0.3,
        0.6, 0.4,
        0.5, 0.5;
    const auto logH_serial = ebact::multinomial_log_evidence(docs, q_many, beta, comps, 1e-12, 8, 1);
    const auto logH_threaded = ebact::multinomial_log_evidence(docs, q_many, beta, comps, 1e-12, 8, 4);
    double max_logh_diff = 0.0;
    for (size_t i = 0; i < logH_serial.size(); ++i) {
        for (size_t g = 0; g < logH_serial[i].size(); ++g) {
            max_logh_diff = std::max(max_logh_diff, std::abs(logH_serial[i][g] - logH_threaded[i][g]));
        }
    }
    check_true("threaded multinomial evidence matches serial", max_logh_diff < 1e-14);

    ebact::FitResult fit_many;
    fit_many.resp = {
        {0.25, 0.75},
        {0.80, 0.20},
        {0.10, 0.90},
        {0.60, 0.40}
    };
    std::vector<int32_t> assignments = {1, 0, 1, 0};
    const auto activity_serial = ebact::multinomial_posterior_activity_means_from_logH(
        docs, q_many, beta, comps, fit_many, assignments, logH_serial, 1e-12, 8, 1);
    const auto activity_threaded = ebact::multinomial_posterior_activity_means_from_logH(
        docs, q_many, beta, comps, fit_many, assignments, logH_serial, 1e-12, 8, 4);
    double max_activity_diff = 0.0;
    for (size_t i = 0; i < activity_serial.size(); ++i) {
        max_activity_diff = std::max(max_activity_diff, std::abs(activity_serial[i] - activity_threaded[i]));
    }
    check_true("threaded posterior activity matches serial", max_activity_diff < 1e-14);
    check_close("posterior activity leaves null-assigned unit at zero",
        activity_serial[1], 0.0, 0.0, 0.0);
}

void test_fit_eta_em() {
    std::vector<std::vector<double>> logH;
    logH.reserve(1000);
    for (int i = 0; i < 600; ++i) {
        logH.push_back({std::log(1000.0), std::log(1.0)});
    }
    for (int i = 0; i < 400; ++i) {
        logH.push_back({std::log(1.0), std::log(1000.0)});
    }
    const auto fit = ebact::fit_eta_em(logH, 10000, 1e-13, 0.0);
    check_close("fit_eta_em eta[0] recovers separable mixture",
        fit.eta[0], 0.6, 8e-4, 0.0);
    check_close("fit_eta_em eta[1] recovers separable mixture",
        fit.eta[1], 0.4, 8e-4, 0.0);
    check_true("fit_eta_em converges before max_iter", fit.iterations < 10000);

    double max_resp_error = 0.0;
    for (int i = 0; i < 600; ++i) {
        max_resp_error = std::max(max_resp_error, std::abs(fit.resp[i][0] - 1.0));
    }
    for (int i = 600; i < 1000; ++i) {
        max_resp_error = std::max(max_resp_error, std::abs(fit.resp[i][1] - 1.0));
    }
    check_true("fit_eta_em responsibilities are near deterministic", max_resp_error < 0.003);

    double max_sum_error = 0.0;
    for (const auto& row : fit.resp) {
        max_sum_error = std::max(max_sum_error, std::abs(row[0] + row[1] - 1.0));
    }
    check_true("fit_eta_em responsibility rows sum to one", max_sum_error < 1e-14);
}

void test_unit_factor_result_header_ranges() {
    {
        const std::vector<std::string> header = {"x", "y", "0", "1", "2"};
        UnitFactorResultReadOptions opts;
        UnitFactorResultHeader parsed = parse_unit_factor_result_header(header, opts);
        check_true("unit-factor header detects numeric dense columns",
            parsed.factorCols.size() == 3 &&
            parsed.factorCols[0] == std::make_pair(0, 2) &&
            parsed.factorCols[1] == std::make_pair(1, 3) &&
            parsed.factorCols[2] == std::make_pair(2, 4));
    }
    {
        const std::vector<std::string> header = {"x", "y", "B_cell", "Fibroblast", "T_cell"};
        UnitFactorResultReadOptions opts;
        opts.factorColBegin = 2;
        opts.factorColEnd = 5;
        UnitFactorResultHeader parsed = parse_unit_factor_result_header(header, opts);
        check_true("unit-factor header maps explicit dense range to contiguous factors",
            parsed.factorCols.size() == 3 &&
            parsed.factorCols[0] == std::make_pair(0, 2) &&
            parsed.factorCols[1] == std::make_pair(1, 3) &&
            parsed.factorCols[2] == std::make_pair(2, 4) &&
            parsed.factorNames[0] == "B_cell" &&
            parsed.factorNames[2] == "T_cell");
    }
    {
        const std::vector<std::string> header = {"x", "y", "K1", "P1", "alpha", "beta"};
        UnitFactorResultReadOptions opts;
        opts.factorColBegin = 4;
        opts.factorColEnd = 6;
        UnitFactorResultHeader parsed = parse_unit_factor_result_header(header, opts);
        check_true("unit-factor header keeps K/P columns with explicit dense range",
            parsed.topPairCols.size() == 1 &&
            parsed.topPairCols[0] == std::make_pair(2, 3) &&
            parsed.factorCols.size() == 2 &&
            parsed.factorCols[0] == std::make_pair(0, 4) &&
            parsed.factorCols[1] == std::make_pair(1, 5));
    }
}

void test_vst_feature_eligibility() {
    std::vector<Document> docs(4);
    docs[0].ids = {0, 1, 2, 3, 4}; docs[0].cnts = {8, 1, 0, 5, 2};
    docs[1].ids = {0, 1, 2, 3, 4}; docs[1].cnts = {1, 2, 4, 5, 3};
    docs[2].ids = {0, 1, 2, 3, 4}; docs[2].cnts = {6, 3, 1, 5, 7};
    docs[3].ids = {0, 1, 2, 3, 4}; docs[3].cnts = {2, 4, 7, 5, 1};

    std::vector<uint8_t> eligible = {1, 0, 1, 0, 1};
    HVF_VST vst;
    const auto order = vst.SelectVST(docs, 5, nullptr, &eligible);
    check_true("VST eligibility only ranks eligible features",
        order.size() == 3 &&
        std::find(order.begin(), order.end(), 0u) != order.end() &&
        std::find(order.begin(), order.end(), 2u) != order.end() &&
        std::find(order.begin(), order.end(), 4u) != order.end() &&
        std::find(order.begin(), order.end(), 1u) == order.end() &&
        std::find(order.begin(), order.end(), 3u) == order.end());
    check_close("VST keeps full mean stats for ineligible features",
        vst.stats.mean_all[1], 2.5, 0.0, 0.0);
    check_close("VST leaves ineligible standardized variance at zero",
        vst.stats.var_standardized[3], 0.0, 0.0, 0.0);
}

void test_gamma_poisson_dispersion_helpers() {
    const double low_mean =
        GammaPoissonDispersionEstimator::positive_truncated_residual_expectation(1e-8, 0.01);
    check_true("Gamma-Poisson truncated residual preserves low-mean scale",
        std::isfinite(low_mean) && low_mean > 9000.0);
    const double d1 =
        GammaPoissonDispersionEstimator::positive_truncated_residual_expectation(0.1, 1.0);
    const double d2 =
        GammaPoissonDispersionEstimator::positive_truncated_residual_expectation(1.0, 1.0);
    check_true("Gamma-Poisson truncated residual increases with dispersion",
        std::isfinite(d1) && std::isfinite(d2) && d2 > d1);
}

void test_gamma_poisson_cluster_basis() {
    const double pi = std::acos(-1.0);
    check_close("trigamma(1)", trigamma(1.0), pi * pi / 6.0, 2e-12, 0.0);
    check_close("trigamma(1/2)", trigamma(0.5), pi * pi / 2.0, 2e-11, 0.0);

    Eigen::MatrixXd helmert = normalized_helmert_basis(4);
    check_true("Helmert annihilates common scale",
        (helmert * Eigen::VectorXd::Ones(4)).norm() < 1e-14);
    check_true("Helmert rows are orthonormal",
        (helmert * helmert.transpose() - Eigen::MatrixXd::Identity(3, 3)).norm() < 1e-14);

    Eigen::VectorXd shape(3), rate(3), capacity(3);
    shape << 1.0, 2.0, 3.0;
    rate << 2.0, 4.0, 5.0;
    capacity << 3.0, 7.0, 11.0;
    GammaPoissonLogMoments moments = gamma_poisson_log_moments(shape, rate, capacity);
    Eigen::VectorXd scaled_rate = rate.array() * Eigen::Array3d(2.0, 0.5, 4.0);
    Eigen::VectorXd scaled_capacity = capacity.array() * Eigen::Array3d(2.0, 0.5, 4.0);
    GammaPoissonLogMoments scaled = gamma_poisson_log_moments(
        shape, scaled_rate, scaled_capacity);
    check_true("token-unit log moments are factor-rescaling invariant",
        (moments.mean - scaled.mean).norm() < 1e-14);

    GammaPoissonTopicCovariance covariance;
    covariance.diagonal.resize(3);
    covariance.diagonal << 0.2, 0.3, 0.4;
    covariance.factor.resize(3, 1);
    covariance.factor << 0.1, -0.2, 0.3;
    Eigen::MatrixXd h3 = normalized_helmert_basis(3);
    check_true("structured covariance projection matches dense multiplication",
        (covariance.transformed(h3) - h3 * covariance.dense() * h3.transpose()).norm() < 1e-14);

    GammaPoissonBasisAccumulator accumulator(3);
    Eigen::VectorXd mean1(3), mean2(3), mean3(3), mean4(3);
    mean1 << 1.0, 0.0, -1.0;
    mean2 << 1.2, 0.1, -1.1;
    mean3 << -1.0, 0.0, 1.0;
    mean4 << -1.2, -0.1, 1.1;
    covariance.factor.resize(3, 0);
    covariance.diagonal.setConstant(0.01);
    accumulator.add(mean1, covariance);
    accumulator.add(mean2, covariance);
    accumulator.add(mean3, covariance);
    accumulator.add(mean4, covariance);
    GammaPoissonLogRatioBasis basis = accumulator.finish();
    check_true("adaptive basis rows are orthonormal",
        (basis.basis * basis.basis.transpose()
            - Eigen::MatrixXd::Identity(2, 2)).norm() < 1e-12);
    check_true("adaptive basis removes common scale",
        (basis.basis * Eigen::VectorXd::Ones(3)).norm() < 1e-12);
    check_true("signal eigenvalues are nonnegative and ordered",
        basis.signal_eigenvalues.minCoeff() >= 0.0
        && basis.signal_eigenvalues(0) >= basis.signal_eigenvalues(1));

    GammaPoissonClusterDataset dataset;
    dataset.log_mean.resize(4, 3);
    dataset.log_mean << mean1.transpose(), mean2.transpose(),
        mean3.transpose(), mean4.transpose();
    covariance.diagonal << 0.2, 0.3, 0.4;
    covariance.factor.resize(3, 1);
    covariance.factor << 0.1, -0.2, 0.3;
    dataset.topic_covariance.assign(4, covariance);
    GammaPoissonClusterCoordinates coordinates =
        make_gamma_poisson_cluster_coordinates(dataset);
    const Eigen::MatrixXd projected_factor =
        coordinates.log_ratio_basis.basis * covariance.factor;
    const Eigen::VectorXd projected_diagonal =
        coordinates.log_ratio_basis.basis.array().square().matrix()
        * covariance.diagonal;
    check_true("cluster handoff retains projected uncertainty factor",
        coordinates.uncertainty_rank == 1
        && (coordinates.factor(0) - projected_factor).norm() < 1e-14);
    check_true("cluster handoff retains transformed diagonal approximation",
        (coordinates.uncertainty_diagonal.row(0).transpose()
            - projected_diagonal).norm() < 1e-14);
}

void test_gamma_poisson_diagonal_mixture() {
    constexpr int32_t n_per_cluster = 300;
    constexpr int32_t n = 2 * n_per_cluster;
    RowMajorMatrixXd observations(n, 2);
    RowMajorMatrixXd uncertainty(n, 2);
    std::mt19937 random_engine(193);
    std::normal_distribution<double> standard_normal(0.0, 1.0);
    for (int32_t d = 0; d < n; ++d) {
        const double center = d < n_per_cluster ? -1.0 : 1.0;
        const double measurement_variance = d % 3 == 0 ? 0.8 : 0.05;
        for (int32_t j = 0; j < 2; ++j) {
            const double latent = center + std::sqrt(0.2) * standard_normal(random_engine);
            observations(d, j) = latent
                + std::sqrt(measurement_variance) * standard_normal(random_engine);
            uncertainty(d, j) = measurement_variance;
        }
    }
    GammaPoissonClusterCoordinates coordinates;
    coordinates.mean = observations;
    coordinates.uncertainty_diagonal = uncertainty;
    coordinates.uncertainty_rank = 0;
    coordinates.uncertainty_factor.resize(n, 0);
    GammaPoissonClusterFitOptions options;
    options.n_components = 2;
    options.max_iterations = 200;
    options.seed = 17;
    options.variance_floor = 1e-5;
    options.tolerance = 1e-8;
    GammaPoissonClusterFitResult fit = fit_gamma_poisson_cluster_mixture(
        coordinates, options);
    const GammaPoissonClusterModel& fitted_model = fit.model;
    const GammaPoissonClusterFitDiagnostics& diagnostics = fit.diagnostics;
    const Eigen::VectorXd weights = gamma_poisson_cluster_weights(fitted_model);
    const Eigen::Index low = fitted_model.means(0, 0) < fitted_model.means(1, 0)
        ? 0 : 1;
    const Eigen::Index high = 1 - low;
    check_close("diagonal heteroscedastic mixture low mean",
        fitted_model.means(low, 0), -1.0, 0.2, 0.0);
    check_close("diagonal heteroscedastic mixture high mean",
        fitted_model.means(high, 0), 1.0, 0.2, 0.0);
    check_close("diagonal heteroscedastic mixture weight",
        weights(low), 0.5, 0.08, 0.0);
    check_close("diagonal heteroscedastic mixture intrinsic variance",
        fitted_model.variances.mean(), 0.2, 0.12, 0.0);
    double certain_confidence = 0.0;
    double uncertain_confidence = 0.0;
    int32_t n_certain = 0;
    int32_t n_uncertain = 0;
    for (int32_t d = 0; d < n; ++d) {
        const double confidence = fit.responsibilities.row(d).maxCoeff();
        if (uncertainty(d, 0) > 0.5) {
            uncertain_confidence += confidence;
            ++n_uncertain;
        } else {
            certain_confidence += confidence;
            ++n_certain;
        }
    }
    check_true("high uncertainty softens mixture memberships",
        uncertain_confidence / n_uncertain < certain_confidence / n_certain);
    check_true("diagonal mixture responsibilities sum to one",
        (fit.responsibilities.rowwise().sum().array() - 1.0).abs().maxCoeff() < 1e-12);
    check_close("variational mixture Dirichlet mass",
        fitted_model.dirichlet_parameters.sum(), n + options.dirichlet_concentration,
        1e-10, 0.0);
    check_true("variational mixture diagnostics are finite",
        std::isfinite(diagnostics.elbo) && std::isfinite(diagnostics.log_likelihood)
        && std::isfinite(diagnostics.p90_responsibility_l1_change)
        && std::isfinite(diagnostics.top_assignment_change_fraction));
    check_true("variational mixture reaches composite convergence",
        diagnostics.converged
        && diagnostics.iterations >= options.convergence_patience);
    bool monotone_elbo = true;
    for (size_t i = 1; i < diagnostics.elbo_trace.size(); ++i) {
        monotone_elbo = monotone_elbo
            && diagnostics.elbo_trace[i] + 1e-8 >= diagnostics.elbo_trace[i - 1];
    }
    check_true("variational mixture ELBO is nondecreasing", monotone_elbo);

    RowMajorMatrixXd previous = RowMajorMatrixXd::Zero(10, 2);
    previous.col(0).setOnes();
    RowMajorMatrixXd current = previous;
    current.row(0) << 0.75, 0.25;
    current.row(1) << 0.0, 1.0;
    GammaPoissonResponsibilityChange change =
        gamma_poisson_responsibility_change(previous, current);
    check_close("responsibility change mean L1", change.mean_l1, 0.25, 1e-14, 0.0);
    check_close("responsibility change p90 L1", change.p90_l1, 0.5, 1e-14, 0.0);
    check_close("responsibility top-assignment fraction",
        change.top_assignment_fraction, 0.1, 1e-14, 0.0);

    GammaPoissonClusterFitOptions capped_options = options;
    capped_options.max_iterations = 1;
    GammaPoissonClusterFitResult capped = fit_gamma_poisson_cluster_mixture(
        coordinates, capped_options);
    check_true("variational mixture reports maximum-iteration termination",
        !capped.diagnostics.converged && capped.diagnostics.iterations == 1);
}

void test_dense_kmeans() {
    RowMajorMatrixXd observations(8, 2);
    observations <<
        -3.0, -3.0,
        -2.8, -3.1,
        -3.2, -2.9,
         3.0,  3.0,
         2.8,  3.1,
         3.2,  2.9,
         0.0,  0.1,
         0.0, -0.1;
    DenseKMeansOptions options;
    options.n_clusters = 3;
    options.max_iterations = 20;
    options.seed = 41;
    DenseKMeansResult first = dense_kmeans(observations, options);
    DenseKMeansResult second = dense_kmeans(observations, options);
    check_true("dense k-means is deterministic",
        (first.centers - second.centers).norm() == 0.0
        && (first.assignments - second.assignments).norm() == 0);
    check_true("dense k-means converges with nonempty clusters",
        first.converged && first.counts.minCoeff() > 0
        && first.iterations <= options.max_iterations);
    check_true("dense k-means separates compact groups",
        first.inertia < 0.3);

    RowMajorMatrixXd identical = RowMajorMatrixXd::Zero(5, 2);
    options.n_clusters = 3;
    options.max_iterations = 3;
    DenseKMeansResult reseeded = dense_kmeans(identical, options);
    check_true("dense k-means reseeds empty clusters",
        reseeded.counts.minCoeff() > 0 && reseeded.inertia == 0.0);
}

void test_low_rank_covariance() {
    LowRankDiagonalCovariance covariance;
    covariance.diagonal.resize(4);
    covariance.diagonal << 0.7, 1.2, 0.4, 2.0;
    covariance.factor.resize(4, 2);
    covariance.factor <<
        0.2, -0.1,
        0.4,  0.3,
       -0.2,  0.5,
        0.1,  0.2;
    const Eigen::MatrixXd dense = covariance.dense();
    LowRankDiagonalSolver solver(covariance.diagonal, covariance.factor);
    Eigen::VectorXd value(4);
    value << 0.5, -1.0, 0.3, 2.0;
    Eigen::VectorXd expected = dense.llt().solve(value);
    check_true("Woodbury vector solve matches dense LLT",
        (solver.solve_vector(value) - expected).norm() < 1e-12);
    Eigen::MatrixXd matrix = Eigen::MatrixXd::Random(4, 3);
    check_true("Woodbury matrix solve matches dense LLT",
        (solver.solve_matrix(matrix) - dense.llt().solve(matrix)).norm() < 1e-12);
    check_close("Woodbury quadratic matches dense LLT",
        solver.quadratic(value), value.dot(expected), 1e-12, 1e-12);
    check_close("Woodbury log determinant matches dense LLT",
        solver.log_determinant(), std::log(dense.determinant()), 1e-12, 1e-12);
    check_true("low-rank covariance apply matches dense",
        (covariance.apply_matrix(matrix) - dense * matrix).norm() < 1e-12);

    RowMajorMatrixXd no_factor(4, 0);
    LowRankDiagonalSolver diagonal_solver(covariance.diagonal, no_factor);
    check_true("rank-zero Woodbury solve is diagonal",
        (diagonal_solver.solve_vector(value)
            - covariance.diagonal.cwiseInverse().cwiseProduct(value)).norm() < 1e-14);
}

void test_gamma_poisson_conditional_moments() {
    LowRankDiagonalCovariance document;
    document.diagonal.resize(3);
    document.diagonal << 0.3, 0.5, 0.2;
    document.factor.resize(3, 1);
    document.factor << 0.4, -0.2, 0.1;
    LowRankDiagonalCovariance cluster;
    cluster.diagonal.resize(3);
    cluster.diagonal << 0.7, 0.4, 0.6;
    cluster.factor.resize(3, 1);
    cluster.factor << 0.5, 0.3, -0.1;
    Eigen::VectorXd observation(3), mean(3);
    observation << 1.2, -0.4, 0.8;
    mean << 0.2, 0.1, -0.3;
    GammaPoissonConditionalMoments moments = gamma_poisson_conditional_moments(
        observation, mean, document, cluster);
    const Eigen::MatrixXd sigma = cluster.dense();
    const Eigen::MatrixXd total = sigma + document.dense();
    const Eigen::VectorXd expected_mean = mean
        + sigma * total.llt().solve(observation - mean);
    const Eigen::MatrixXd expected_covariance = sigma
        - sigma * total.llt().solve(sigma);
    const double expected_log_likelihood = -0.5 * (
        3.0 * std::log(2.0 * std::acos(-1.0))
        + std::log(total.determinant())
        + (observation - mean).dot(total.llt().solve(observation - mean)));
    check_true("structured conditional mean matches dense Gaussian conditioning",
        (moments.mean - expected_mean).norm() < 1e-12);
    check_true("structured conditional covariance matches dense Gaussian conditioning",
        (moments.covariance - expected_covariance).norm() < 1e-12);
    check_close("structured marginal likelihood matches dense Gaussian",
        moments.log_likelihood, expected_log_likelihood, 1e-12, 1e-12);
}

void test_gamma_poisson_structured_mixture() {
    constexpr int32_t n = 240;
    constexpr int32_t dim = 3;
    GammaPoissonClusterCoordinates coordinates;
    coordinates.mean.resize(n, dim);
    coordinates.uncertainty_diagonal =
        RowMajorMatrixXd::Constant(n, dim, 0.08);
    coordinates.uncertainty_rank = 1;
    coordinates.uncertainty_factor.resize(n, dim);
    Eigen::Vector3d cluster_direction(1.0, 0.8, -0.3);
    cluster_direction.normalize();
    const Eigen::Matrix3d intrinsic = 0.12 * Eigen::Matrix3d::Identity()
        + 0.55 * cluster_direction * cluster_direction.transpose();
    std::mt19937 random_engine(931);
    std::normal_distribution<double> normal(0.0, 1.0);
    for (int32_t d = 0; d < n; ++d) {
        const double center = d < n / 2 ? -1.4 : 1.4;
        Eigen::Vector3d document_factor(0.25 + 0.1 * (d % 3), -0.2, 0.15);
        Eigen::Matrix3d total = intrinsic
            + document_factor * document_factor.transpose();
        total.diagonal() += coordinates.uncertainty_diagonal.row(d).transpose();
        Eigen::Vector3d noise;
        noise << normal(random_engine), normal(random_engine), normal(random_engine);
        coordinates.mean.row(d) = (
            Eigen::Vector3d::Constant(center) + total.llt().matrixL() * noise).transpose();
        Eigen::Map<RowMajorMatrixXd>(
            coordinates.uncertainty_factor.row(d).data(), dim, 1) = document_factor;
    }
    GammaPoissonClusterFitOptions options;
    options.n_components = 2;
    options.cluster_covariance_rank = 1;
    options.diagonal_warmup_iterations = 4;
    options.orientation_update_interval = 3;
    options.orientation_max_updates = 2;
    options.orientation_patience = 1;
    options.max_iterations = 50;
    options.convergence_patience = 3;
    options.n_threads = 2;
    options.seed = 37;
    options.tolerance = 1e-7;
    options.responsibility_p90_tolerance = 2e-3;
    GammaPoissonClusterFitResult first = fit_gamma_poisson_cluster_mixture(
        coordinates, options);
    GammaPoissonClusterFitResult second = fit_gamma_poisson_cluster_mixture(
        coordinates, options);
    GammaPoissonClusterFitOptions compact_options = options;
    compact_options.covariance_accumulation = GammaPoissonClusterFitOptions::
        CovarianceAccumulation::Compact;
    GammaPoissonClusterFitResult compact = fit_gamma_poisson_cluster_mixture(
        coordinates, compact_options);
    GammaPoissonClusterFitOptions dense_options = options;
    dense_options.covariance_accumulation = GammaPoissonClusterFitOptions::
        CovarianceAccumulation::Dense;
    GammaPoissonClusterFitResult explicit_dense =
        fit_gamma_poisson_cluster_mixture(coordinates, dense_options);
    options.n_threads = 1;
    GammaPoissonClusterFitResult serial = fit_gamma_poisson_cluster_mixture(
        coordinates, options);
    options.cluster_covariance_rank = 0;
    GammaPoissonClusterFitResult diagonal = fit_gamma_poisson_cluster_mixture(
        coordinates, options);
    const GammaPoissonClusterModel& model = first.model;
    const Eigen::Index low = model.means(0, 0) < model.means(1, 0) ? 0 : 1;
    const Eigen::Index high = 1 - low;
    check_close("structured mixture low mean", model.means(low, 0), -1.4, 0.3, 0.0);
    check_close("structured mixture high mean", model.means(high, 0), 1.4, 0.3, 0.0);
    check_true("structured mixture uses both covariance ranks",
        model.orientation.rows() == dim && model.orientation.cols() == 1
        && model.low_rank_variances.minCoeff() > 0.0);
    check_true("small topic models select dense covariance accumulation",
        first.diagnostics.covariance_accumulation == "dense"
        && explicit_dense.diagnostics.covariance_accumulation == "dense"
        && compact.diagnostics.covariance_accumulation == "compact");
    check_true("explicit dense accumulation matches automatic dense selection",
        (model.means - explicit_dense.model.means).norm() == 0.0
        && (model.variances - explicit_dense.model.variances).norm() == 0.0
        && (first.responsibilities
            - explicit_dense.responsibilities).norm() == 0.0);
    check_true("compact override produces finite structured fit",
        compact.model.means.allFinite()
        && compact.model.variances.allFinite()
        && compact.responsibilities.allFinite());
    check_true("structured orientation is orthonormal",
        (model.orientation.transpose() * model.orientation
            - Eigen::MatrixXd::Identity(1, 1)).norm() < 1e-12);
    check_true("fixed-shard parallel E-step is deterministic",
        (model.means - second.model.means).norm() == 0.0
        && (model.variances - second.model.variances).norm() == 0.0
        && (first.responsibilities - second.responsibilities).norm() == 0.0);
    check_true("parallel and serial structured fits agree numerically",
        (model.means - serial.model.means).norm() < 1e-10
        && (model.variances - serial.model.variances).norm() < 1e-10
        && (first.responsibilities - serial.responsibilities).norm() < 1e-9);
    check_true("structured responsibilities are normalized and finite",
        first.responsibilities.allFinite()
        && (first.responsibilities.rowwise().sum().array() - 1.0)
            .abs().maxCoeff() < 1e-12);
    check_true("structured covariance improves correlated-data likelihood",
        first.diagnostics.log_likelihood > diagonal.diagnostics.log_likelihood);

    GammaPoissonClusterCoordinates boundary_coordinates;
    boundary_coordinates.mean = RowMajorMatrixXd::Random(8, 47);
    boundary_coordinates.uncertainty_diagonal =
        RowMajorMatrixXd::Constant(8, 47, 0.1);
    boundary_coordinates.uncertainty_factor.resize(8, 0);
    boundary_coordinates.uncertainty_rank = 0;
    GammaPoissonClusterCoordinates threshold_coordinates;
    threshold_coordinates.mean = RowMajorMatrixXd::Random(8, 48);
    threshold_coordinates.uncertainty_diagonal =
        RowMajorMatrixXd::Constant(8, 48, 0.1);
    threshold_coordinates.uncertainty_factor.resize(8, 0);
    threshold_coordinates.uncertainty_rank = 0;
    GammaPoissonClusterFitOptions threshold_options;
    threshold_options.n_components = 2;
    threshold_options.cluster_covariance_rank = 1;
    threshold_options.diagonal_warmup_iterations = 0;
    threshold_options.orientation_max_updates = 0;
    threshold_options.max_iterations = 1;
    threshold_options.kmeans_max_iterations = 2;
    GammaPoissonClusterFitResult boundary_fit =
        fit_gamma_poisson_cluster_mixture(
            boundary_coordinates, threshold_options);
    GammaPoissonClusterFitResult threshold_fit =
        fit_gamma_poisson_cluster_mixture(
            threshold_coordinates, threshold_options);
    check_true("automatic covariance accumulation switches above 48 topics",
        boundary_fit.diagnostics.covariance_accumulation == "dense"
        && threshold_fit.diagnostics.covariance_accumulation == "compact");

    GammaPoissonClusterScore rescored = score_gamma_poisson_cluster_mixture(
        coordinates, model, 2);
    check_true("fixed-state scorer reproduces fitted responsibilities",
        (rescored.responsibilities - first.responsibilities).norm() < 1e-12);
    check_close("fixed-state scorer reproduces predictive likelihood",
        rescored.predictive_log_likelihood, first.diagnostics.log_likelihood,
        1e-10, 1e-12);

    Eigen::VectorXd expected_sizes(3);
    expected_sizes << 4.999, 5.0, 6.0;
    const std::vector<uint8_t> active = gamma_poisson_active_components(
        expected_sizes, 5.0);
    check_true("absolute cluster activation includes threshold boundary",
        active == std::vector<uint8_t>({0, 1, 1}));
    RowMajorMatrixXd membership_example(3, 2);
    membership_example << 1.0, 0.0,
                          0.5, 0.5,
                          0.0, 1.0;
    const Eigen::VectorXd effective_size =
        gamma_poisson_effective_membership_size(membership_example);
    check_true("effective membership size uses squared soft mass",
        (effective_size.array() - 1.8).abs().maxCoeff() < 1e-14);

    const int32_t first_component = 0;
    const int32_t second_component = 1;
    Eigen::MatrixXd covariance_first =
        model.variances.row(first_component).transpose().asDiagonal();
    covariance_first.noalias() += model.orientation
        * model.low_rank_variances.row(first_component).asDiagonal()
        * model.orientation.transpose();
    Eigen::MatrixXd covariance_second =
        model.variances.row(second_component).transpose().asDiagonal();
    covariance_second.noalias() += model.orientation
        * model.low_rank_variances.row(second_component).asDiagonal()
        * model.orientation.transpose();
    const Eigen::MatrixXd covariance_average =
        0.5 * (covariance_first + covariance_second);
    const Eigen::VectorXd center_difference =
        (model.means.row(first_component)
            - model.means.row(second_component)).transpose();
    const double squared_separation = center_difference.dot(
        covariance_average.llt().solve(center_difference));
    const double expected_bhattacharyya = 0.125 * squared_separation
        + 0.5 * (std::log(covariance_average.determinant())
            - 0.5 * (std::log(covariance_first.determinant())
                + std::log(covariance_second.determinant())));
    const GammaPoissonClusterSeparation separation =
        gamma_poisson_cluster_separation(
            model, first_component, second_component);
    check_close("cluster log volume matches dense determinant",
        gamma_poisson_cluster_log_volume(model, first_component),
        0.5 * std::log(covariance_first.determinant()), 1e-11, 1e-11);
    check_close("standardized cluster separation matches dense solve",
        separation.standardized_distance, std::sqrt(squared_separation),
        1e-11, 1e-11);
    check_close("Bhattacharyya distance matches dense calculation",
        separation.bhattacharyya_distance, expected_bhattacharyya,
        1e-11, 1e-11);

    GammaPoissonClusterState state;
    state.topic_state_checksum = 1234567;
    state.topic_names = {"0", "1", "2", "3"};
    state.basis = normalized_helmert_basis(4);
    state.model = model;
    state.diagnostics = first.diagnostics;
    state.document_uncertainty_model =
        GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionLowRank;
    state.document_uncertainty_rank = coordinates.uncertainty_rank;
    state.min_cluster_size = 5.0;
    state.active = gamma_poisson_active_components(
        first.effective_membership, state.min_cluster_size);
    const std::filesystem::path state_path =
        std::filesystem::temp_directory_path()
        / "punkst_gamma_pois_cluster_state_test.tsv";
    write_gamma_poisson_cluster_state(state_path.string(), state);
    GammaPoissonClusterState restored = read_gamma_poisson_cluster_state(
        state_path.string(), state.topic_state_checksum);
    check_true("cluster state round trip preserves fixed model",
        restored.topic_names == state.topic_names
        && restored.active == state.active
        && restored.min_cluster_size == state.min_cluster_size
        && (restored.basis - state.basis).norm() < 1e-14
        && restored.document_uncertainty_model == state.document_uncertainty_model
        && restored.document_uncertainty_rank == state.document_uncertainty_rank
        && (restored.model.means - model.means).norm() < 1e-14
        && (restored.model.variances - model.variances).norm() < 1e-14
        && (restored.model.orientation - model.orientation).norm() < 1e-14
        && (restored.model.low_rank_variances
            - model.low_rank_variances).norm() < 1e-14);
    bool cluster_checksum_rejected = false;
    try {
        read_gamma_poisson_cluster_state(
            state_path.string(), state.topic_state_checksum + 1);
    } catch (const std::exception&) {
        cluster_checksum_rejected = true;
    }
    check_true("cluster state rejects topic checksum mismatch",
        cluster_checksum_rejected);
    bool nonorthogonal_rejected = false;
    try {
        GammaPoissonClusterState invalid = state;
        invalid.basis(0, 0) += 0.1;
        write_gamma_poisson_cluster_state(
            (state_path.string() + ".invalid"), invalid);
    } catch (const std::exception&) {
        nonorthogonal_rejected = true;
    }
    check_true("cluster state rejects nonorthonormal basis",
        nonorthogonal_rejected);
    std::ifstream cluster_state_in(state_path);
    const std::string cluster_state_text{
        std::istreambuf_iterator<char>(cluster_state_in),
        std::istreambuf_iterator<char>()};
    const std::string marker = "##punkst_gamma_pois_cluster\n";
    const std::filesystem::path duplicate_metadata_path =
        state_path.string() + ".duplicate-metadata";
    {
        std::ofstream out(duplicate_metadata_path);
        out << marker << "##state_checksum\t1234567\n"
            << cluster_state_text.substr(marker.size());
    }
    bool duplicate_metadata_rejected = false;
    try {
        read_gamma_poisson_cluster_state(
            duplicate_metadata_path.string(), state.topic_state_checksum);
    } catch (const std::exception&) {
        duplicate_metadata_rejected = true;
    }
    check_true("cluster state rejects duplicate metadata",
        duplicate_metadata_rejected);

    std::string incomplete_state = cluster_state_text;
    const size_t record_begin = incomplete_state.find("dirichlet\t0\t");
    const size_t record_end = incomplete_state.find('\n', record_begin);
    if (record_begin != std::string::npos && record_end != std::string::npos) {
        incomplete_state.erase(record_begin, record_end - record_begin + 1);
    }
    const std::filesystem::path incomplete_state_path =
        state_path.string() + ".incomplete";
    {
        std::ofstream out(incomplete_state_path);
        out << incomplete_state;
    }
    bool incomplete_state_rejected = false;
    try {
        read_gamma_poisson_cluster_state(
            incomplete_state_path.string(), state.topic_state_checksum);
    } catch (const std::exception&) {
        incomplete_state_rejected = true;
    }
    check_true("cluster state rejects missing records",
        incomplete_state_rejected);
}

void test_gamma_poisson_posterior_and_sidecar() {
    GammaPoissonTopicModel model(3, 4, 19, 1, 0, 0.3, 0.3, 1.0,
        1.0, 1.0, 1.0, 0.7, 10.0, 10, 6.0);
    Document doc;
    doc.ids = {0, 2, 3};
    doc.cnts = {4.0, 2.0, 1.0};
    GammaPoissonDocumentPosterior posterior;
    model.infer_document_posterior(doc, posterior);
    RowVectorXd normalized = model.normalized_topic_mean(posterior);
    RowVectorXd reconstructed = (posterior.shape.array() / posterior.rate.array()
        * model.get_topic_capacity().array()).matrix().transpose();
    reconstructed /= reconstructed.sum();
    check_true("local posterior reconstructs normalized topic output",
        (normalized - reconstructed).norm() < 1e-14);
    check_true("local posterior has positive finite parameters",
        posterior.exposure > 0.0
        && posterior.shape.array().isFinite().all()
        && posterior.rate.array().isFinite().all()
        && (posterior.shape.array() > 0.0).all()
        && (posterior.rate.array() > 0.0).all());

    model.set_feature_dispersion({2.0, 3.0, 4.0, 5.0});
    model.infer_document_posterior(doc, posterior);
    GammaPoissonDispersionApproximation exact;
    model.dispersion_covariance_approximation(doc, posterior, 3, 41, exact);
    check_true("full-rank dispersion approximation preserves covariance diagonal",
        exact.residual_diagonal.maxCoeff() < 1e-10);
    GammaPoissonDispersionApproximation compressed1, compressed2;
    model.dispersion_covariance_approximation(doc, posterior, 1, 41, compressed1);
    model.dispersion_covariance_approximation(doc, posterior, 1, 41, compressed2);
    check_true("dispersion compression is deterministic",
        (compressed1.factor - compressed2.factor).norm() < 1e-14
        && (compressed1.residual_diagonal - compressed2.residual_diagonal).norm() < 1e-14);

    const std::filesystem::path dir = std::filesystem::temp_directory_path()
        / "punkst_gamma_pois_posterior_test";
    std::filesystem::create_directories(dir);
    const std::filesystem::path state = dir / "model.state.tsv";
    model.write_state(state.string(), {"A", "B", "C", "D"});
    const uint64_t checksum = gamma_poisson_state_checksum(state.string());
    GammaPoissonArtifactId artifact_id;
    artifact_id.words = {0x0123456789abcdefULL, 0xfedcba9876543210ULL};
    const std::filesystem::path posterior_file = dir / "posterior.tsv";
    {
        std::ofstream out(posterior_file);
        out << "##punkst_gamma_pois_posterior_v2\n";
        out << "##n_topics\t3\n";
        out << "##state_checksum\t" << checksum << "\n";
        out << "##row_order\trandomized\n";
        out << "##dispersion_sidecar_id\t"
            << gamma_poisson_artifact_id_string(artifact_id) << "\n";
        out << "#unit\tgp_row\tgp_exposure\tgp_shape_0\tgp_shape_1\tgp_shape_2"
            << "\tgp_rate_0\tgp_rate_1\tgp_rate_2\n";
        out << std::setprecision(17) << "unit-a\t0\t" << posterior.exposure;
        for (int32_t k = 0; k < 3; ++k) out << "\t" << posterior.shape(k);
        for (int32_t k = 0; k < 3; ++k) out << "\t" << posterior.rate(k);
        out << "\n";
        out << std::setprecision(17) << "unit-b\t1\t" << posterior.exposure;
        for (int32_t k = 0; k < 3; ++k) out << "\t" << posterior.shape(k);
        for (int32_t k = 0; k < 3; ++k) out << "\t" << posterior.rate(k);
        out << "\n";
    }
    GammaPoissonPosteriorReader posterior_reader(posterior_file.string(), checksum);
    GammaPoissonPosteriorRow posterior_row;
    check_true("posterior TSV header round trip",
        posterior_reader.header().n_topics == 3
        && posterior_reader.header().row_order == "randomized"
        && posterior_reader.header().dispersion_sidecar_id == artifact_id
        && posterior_reader.header().topic_names == std::vector<std::string>({"0", "1", "2"}));
    check_true("posterior TSV reads row", posterior_reader.read_next(posterior_row));
    check_true("posterior TSV preserves identifiers and parameters",
        posterior_row.row == 0 && posterior_row.identifiers == "unit-a"
        && (posterior_row.posterior.shape - posterior.shape).norm() < 1e-14
        && (posterior_row.posterior.rate - posterior.rate).norm() < 1e-14);
    check_true("posterior TSV reads sequential second row",
        posterior_reader.read_next(posterior_row) && posterior_row.row == 1);
    check_true("posterior TSV stops at end of input", !posterior_reader.read_next(posterior_row));
    GammaPoissonClusterDataset cluster_dataset = load_gamma_poisson_cluster_dataset(
        posterior_file.string(), "", checksum, model.get_topic_capacity(),
        model.get_topic_names());
    GammaPoissonClusterCoordinates cluster_coordinates =
        make_gamma_poisson_cluster_coordinates(cluster_dataset);
    check_true("in-memory clustering handoff dimensions",
        cluster_dataset.log_mean.rows() == 2
        && cluster_dataset.log_mean.cols() == 3
        && cluster_coordinates.mean.rows() == 2
        && cluster_coordinates.mean.cols() == 2
        && cluster_coordinates.uncertainty_diagonal.minCoeff() > 0.0);
    const std::filesystem::path sidecar = dir / "posterior-dispersion.bin";
    {
        GammaPoissonDispersionWriter writer(
            sidecar.string(), 3, 1, checksum, artifact_id);
        writer.append(compressed1);
        writer.append(compressed2);
        writer.close();
    }
    GammaPoissonDispersionReader reader(sidecar.string(), checksum);
    check_true("dispersion sidecar header round trip",
        reader.header().n_topics == 3 && reader.header().rank == 1
        && reader.header().record_count == 2
        && reader.header().state_checksum == checksum
        && reader.header().artifact_id == artifact_id);
    GammaPoissonDispersionApproximation restored;
    check_true("dispersion sidecar reads first record", reader.read_next(restored));
    check_true("dispersion sidecar float32 values round trip",
        (restored.factor - compressed1.factor).norm() < 1e-6
        && (restored.residual_diagonal - compressed1.residual_diagonal).norm() < 1e-6);
    check_true("dispersion sidecar reads second record", reader.read_next(restored));
    check_true("dispersion sidecar stops at declared record count", !reader.read_next(restored));
    GammaPoissonClusterDataset paired_dataset = load_gamma_poisson_cluster_dataset(
        posterior_file.string(), sidecar.string(), checksum,
        model.get_topic_capacity(), model.get_topic_names());
    check_true("paired posterior and sidecar IDs load together",
        paired_dataset.uncertainty_model
            == GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionLowRank
        && paired_dataset.topic_covariance.size() == 2
        && paired_dataset.topic_covariance.front().factor.cols() == 1);

    GammaPoissonArtifactId other_artifact_id;
    other_artifact_id.words = {artifact_id.words[0] + 1, artifact_id.words[1]};
    const std::filesystem::path mismatched_sidecar = dir / "posterior-mismatched.bin";
    {
        GammaPoissonDispersionWriter writer(
            mismatched_sidecar.string(), 3, 1, checksum, other_artifact_id);
        writer.append(compressed1);
        writer.append(compressed2);
        writer.close();
    }
    bool sidecar_id_mismatch_rejected = false;
    try {
        load_gamma_poisson_cluster_dataset(posterior_file.string(),
            mismatched_sidecar.string(), checksum, model.get_topic_capacity(),
            model.get_topic_names());
    } catch (const std::exception&) {
        sidecar_id_mismatch_rejected = true;
    }
    check_true("clustering handoff rejects mismatched posterior and sidecar IDs",
        sidecar_id_mismatch_rejected);
    bool mismatch_rejected = false;
    try {
        GammaPoissonDispersionReader bad(sidecar.string(), checksum + 1);
    } catch (const std::exception&) {
        mismatch_rejected = true;
    }
    check_true("dispersion sidecar rejects state checksum mismatch", mismatch_rejected);

    std::ifstream sidecar_input(sidecar, std::ios::binary);
    std::vector<char> sidecar_bytes((std::istreambuf_iterator<char>(sidecar_input)),
        std::istreambuf_iterator<char>());
    const std::filesystem::path truncated_sidecar = dir / "posterior-truncated.bin";
    {
        std::ofstream out(truncated_sidecar, std::ios::binary);
        out.write(sidecar_bytes.data(), sidecar_bytes.size() - 1);
    }
    bool truncation_rejected = false;
    try {
        GammaPoissonDispersionReader bad(truncated_sidecar.string(), checksum);
    } catch (const std::exception&) {
        truncation_rejected = true;
    }
    check_true("dispersion sidecar rejects truncated payload", truncation_rejected);
    const std::filesystem::path trailing_sidecar = dir / "posterior-trailing.bin";
    {
        std::ofstream out(trailing_sidecar, std::ios::binary);
        out.write(sidecar_bytes.data(), sidecar_bytes.size());
        out.put('\0');
    }
    bool trailing_rejected = false;
    try {
        GammaPoissonDispersionReader bad(trailing_sidecar.string(), checksum);
    } catch (const std::exception&) {
        trailing_rejected = true;
    }
    check_true("dispersion sidecar rejects trailing payload", trailing_rejected);

    const std::filesystem::path duplicate_columns = dir / "posterior-duplicate.tsv";
    {
        std::ofstream out(duplicate_columns);
        out << "##punkst_gamma_pois_posterior_v2\n"
            << "##n_topics\t1\n"
            << "##state_checksum\t" << checksum << "\n"
            << "##row_order\tinput\n"
            << "#gp_row\tgp_row\tgp_exposure\tgp_shape_0\tgp_rate_0\n";
    }
    bool duplicate_rejected = false;
    try {
        GammaPoissonPosteriorReader bad(duplicate_columns.string(), checksum);
    } catch (const std::exception&) {
        duplicate_rejected = true;
    }
    check_true("posterior TSV rejects duplicate columns", duplicate_rejected);

    const std::filesystem::path bad_row_file = dir / "posterior-bad-row.tsv";
    {
        std::ofstream out(bad_row_file);
        out << "##punkst_gamma_pois_posterior_v2\n"
            << "##n_topics\t1\n"
            << "##state_checksum\t" << checksum << "\n"
            << "##row_order\tinput\n"
            << "#gp_row\tgp_exposure\tgp_shape_0\tgp_rate_0\n"
            << "1\t1\t1\t1\n";
    }
    bool row_order_rejected = false;
    try {
        GammaPoissonPosteriorReader bad(bad_row_file.string(), checksum);
        GammaPoissonPosteriorRow bad_row;
        bad.read_next(bad_row);
    } catch (const std::exception&) {
        row_order_rejected = true;
    }
    check_true("posterior TSV rejects nonsequential rows", row_order_rejected);

    const std::filesystem::path obsolete_posterior = dir / "posterior-v1.tsv";
    {
        std::ofstream out(obsolete_posterior);
        out << "##punkst_gamma_pois_posterior_v1\n";
    }
    bool obsolete_posterior_rejected = false;
    try {
        GammaPoissonPosteriorReader bad(obsolete_posterior.string(), checksum);
    } catch (const std::exception&) {
        obsolete_posterior_rejected = true;
    }
    check_true("posterior reader rejects obsolete v1 format",
        obsolete_posterior_rejected);
}

void test_hexreader_feature_weights() {
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "punkst_hexreader_weight_test";
    std::filesystem::create_directories(dir);
    const std::filesystem::path metaFile = dir / "hex.json";
    const std::filesystem::path weightFile = dir / "weights.tsv";

    {
        std::ofstream meta(metaFile);
        meta << R"({
            "n_units": 1,
            "n_features": 3,
            "n_modalities": 1,
            "offset_data": 3,
            "header_info": ["random_key", "x", "y"],
            "dictionary": {"A": 0, "B": 1, "C": 2}
        })";
    }
    {
        std::ofstream weights(weightFile);
        weights << "#feature\tunused\tweight\n";
        weights << "\n";
        weights << "A\t100\t2\n";
        weights << "B\t100\t0.5\n";
        weights << "C\t100\t-1\n";
        weights << "missing\t100\tnan\n";
    }

    HexReader reader;
    reader.readMetadata(metaFile.string());
    reader.setWeights(weightFile.string(), 3.0, 2);

    const std::string line = "abcd\t0\t0\t3\t10\t0 2\t1 4\t2 1";
    Document doc;
    int32_t rawCount = reader.parseLine(doc, line);
    check_true("hexreader parse returns raw unit count", rawCount == 10);
    check_close("hexreader raw total preserved", doc.get_raw_sum(), 7.0, 0.0, 0.0);
    check_close("hexreader weighted total", doc.get_sum(), 9.0, 0.0, 0.0);
    check_true("hexreader marks parsed weighted counts", doc.counts_weighted);
    check_close("hexreader weighted feature A", doc.cnts[0], 4.0, 0.0, 0.0);
    check_close("hexreader weighted feature B", doc.cnts[1], 2.0, 0.0, 0.0);
    check_close("hexreader default feature weight", doc.cnts[2], 3.0, 0.0, 0.0);
    check_close("hexreader rawCountFor weighted A", reader.rawCountFor(0, doc.cnts[0], doc.counts_weighted), 2.0, 0.0, 0.0);
    check_close("hexreader rawCountFor weighted B", reader.rawCountFor(1, doc.cnts[1], doc.counts_weighted), 4.0, 0.0, 0.0);
    check_close("hexreader rawCountFor weighted default", reader.rawCountFor(2, doc.cnts[2], doc.counts_weighted), 1.0, 0.0, 0.0);
    check_close("hexreader rawCountFor raw unchanged", reader.rawCountFor(0, 2.0, false), 2.0, 0.0, 0.0);

    reader.applyWeights(doc);
    check_close("hexreader applyWeights is idempotent", doc.get_sum(), 9.0, 0.0, 0.0);

    const std::filesystem::path hexFile = dir / "hex.tsv";
    {
        std::ofstream out(hexFile);
        out << line << "\n";
    }
    std::vector<Document> docs;
    int32_t nkept = reader.readAll(docs, hexFile.string(), 8, false, INT_MAX, 0);
    check_true("hexreader min-count filtering uses raw counts before weights",
        nkept == 0 && docs.empty());

    std::vector<std::string> subset = {"B", "C"};
    reader.setFeatureIndexRemap(subset, false);
    Document remapped;
    reader.parseLine(remapped, line);
    check_true("hexreader remapped ids",
        remapped.ids.size() == 2 &&
        remapped.ids[0] == 0 &&
        remapped.ids[1] == 1);
    check_close("hexreader remapped B weight", remapped.cnts[0], 2.0, 0.0, 0.0);
    check_close("hexreader remapped C default weight", remapped.cnts[1], 3.0, 0.0, 0.0);

    const std::filesystem::path numericWeightFile = dir / "numeric_weights.tsv";
    {
        std::ofstream weights(numericWeightFile);
        weights << "#idx\tunused\tweight\n";
        weights << "0\tignored\t2\n";
        weights << "\n";
        weights << "1\tignored\t-5\n";
        weights << "2\tignored\t0\n";
    }
    lineParser parser;
    int32_t novlp = parser.readWeights(numericWeightFile.string(), 1.0, 3, 2);
    check_true("lineParser numeric weight overlap", novlp == 2);
    check_true("lineParser numeric weight values",
        parser.weights.size() == 3 &&
        parser.weights[0] == 2.0 &&
        parser.weights[1] == 1.0 &&
        parser.weights[2] == 0.0);

    const std::filesystem::path metaFile2 = dir / "hex_weighted_features_meta.json";
    {
        std::ofstream meta(metaFile2);
        meta << R"({
            "n_units": 1,
            "n_features": 4,
            "n_modalities": 1,
            "offset_data": 3,
            "header_info": ["random_key", "x", "y"],
            "dictionary": {"A": 0, "B": 1, "C": 2, "D": 3}
        })";
    }
    const std::filesystem::path weightedFeatureFile = dir / "weighted_features.tsv";
    {
        std::ofstream out(weightedFeatureFile);
        out << "#feature\ttotal_count\tidf\n";
        out << "A\t10\t2\n";
        out << "B\t4\t0.5\n";
        out << "C\t20\t-1\n";
    }
    std::string empty_regex;
    HexReader combined_reader;
    combined_reader.readMetadata(metaFile2.string());
    combined_reader.setFeatureFilterAndWeights(weightedFeatureFile.string(), 5,
        empty_regex, empty_regex, 2, -1.0, false);
    check_true("feature weights from feature file drop invalid and low-count rows",
        combined_reader.features.size() == 1 && combined_reader.features[0] == "A");
    Document combined_doc;
    combined_reader.parseLine(combined_doc, "abcd\t0\t0\t4\t10\t0 2\t1 4\t2 1\t3 3");
    check_true("feature weights from feature file remap valid feature only",
        combined_doc.ids.size() == 1 && combined_doc.ids[0] == 0);
    check_close("feature weights from feature file apply active weight",
        combined_doc.cnts[0], 4.0, 0.0, 0.0);
    const std::vector<double> positiveColumn = combined_reader.readPositiveFeatureColumn(
        weightedFeatureFile.string(), 2, "test value");
    check_true("aligned positive feature column",
        positiveColumn.size() == 1 && positiveColumn[0] == 2.0);

    const std::filesystem::path gzFeatureFile = dir / "positive_features.tsv.gz";
    {
        gzFile out = gzopen(gzFeatureFile.c_str(), "wb");
        const std::string contents = "#feature\tcount\ttau\nA\t10\t3.5\n";
        gzwrite(out, contents.data(), static_cast<unsigned int>(contents.size()));
        gzclose(out);
    }
    const std::vector<double> gzPositiveColumn = combined_reader.readPositiveFeatureColumn(
        gzFeatureFile.string(), 2, "test value");
    check_true("aligned positive feature column supports gzip",
        gzPositiveColumn.size() == 1 && gzPositiveColumn[0] == 3.5);

    HexReader default_reader;
    default_reader.readMetadata(metaFile2.string());
    default_reader.setFeatureFilterAndWeights(weightedFeatureFile.string(), 5,
        empty_regex, empty_regex, 2, 0.25, true);
    check_true("feature weights default keeps missing features but excludes invalid and filtered rows",
        default_reader.features.size() == 2 &&
        default_reader.features[0] == "A" &&
        default_reader.features[1] == "D");
    Document default_doc;
    default_reader.parseLine(default_doc, "abcd\t0\t0\t4\t10\t0 2\t1 4\t2 1\t3 3");
    check_true("feature weights default remapped ids",
        default_doc.ids.size() == 2 &&
        default_doc.ids[0] == 0 &&
        default_doc.ids[1] == 1);
    check_close("feature weights default valid row weight",
        default_doc.cnts[0], 4.0, 0.0, 0.0);
    check_close("feature weights default missing row weight",
        default_doc.cnts[1], 0.75, 0.0, 0.0);

    const std::filesystem::path modelFile = dir / "weighted_features_model.tsv";
    {
        std::ofstream model(modelFile);
        model << "Feature\t0\t1\n";
        model << "A\t1\t2\n";
        model << "B\t1\t2\n";
        model << "C\t1\t2\n";
        model << "D\t1\t2\n";
    }
    HexReader prior_drop_reader;
    prior_drop_reader.readMetadata(metaFile2.string());
    prior_drop_reader.setFeatureFilterAndWeights(weightedFeatureFile.string(), 5,
        empty_regex, empty_regex, 2, -1.0, false);
    LDA4Hex prior_drop(prior_drop_reader);
    prior_drop.preparePriorFeatureSpace(modelFile.string());
    std::vector<std::string> prior_drop_features = prior_drop.getFeatureNames();
    check_true("model prior drops missing features when default weight is negative",
        prior_drop_features.size() == 1 && prior_drop_features[0] == "A");

    HexReader prior_default_reader;
    prior_default_reader.readMetadata(metaFile2.string());
    prior_default_reader.setFeatureFilterAndWeights(weightedFeatureFile.string(), 5,
        empty_regex, empty_regex, 2, 0.0, true);
    LDA4Hex prior_default(prior_default_reader);
    prior_default.preparePriorFeatureSpace(modelFile.string());
    std::vector<std::string> prior_default_features = prior_default.getFeatureNames();
    check_true("model prior keeps missing features when default weight is non-negative",
        prior_default_features.size() == 2 &&
        prior_default_features[0] == "A" &&
        prior_default_features[1] == "D");
}

void test_feature_stats() {
    FeatureOccurrenceStats stats;
    std::map<uint32_t, uint32_t> unit0 = {{0, 1}, {1, 3}};
    std::map<uint32_t, uint32_t> unit1 = {{0, 2}, {2, 5}};
    std::map<uint32_t, uint32_t> unit2 = {{1, 1}};
    stats.observeUnit(unit0);
    stats.observeUnit(unit1);
    stats.observeUnit(unit2);
    stats.ensureFeatureCount(4);

    check_true("feature occurrence n units", stats.nUnits == 3);
    check_true("feature occurrence present counts",
        stats.nPresent[0] == 2 &&
        stats.nPresent[1] == 2 &&
        stats.nPresent[2] == 1 &&
        stats.nPresent[3] == 0);
    check_true("feature occurrence gt thresholds",
        stats.nGt1[0] == 1 &&
        stats.nGt2[1] == 1 &&
        stats.nGt2[2] == 1);
    check_true("feature occurrence gt mean is strict global mean",
        stats.countGreaterThanMean(0) == 1 &&
        stats.countGreaterThanMean(1) == 1 &&
        stats.countGreaterThanMean(2) == 1 &&
        stats.countGreaterThanMean(3) == 0);

    const std::filesystem::path outFile =
        std::filesystem::temp_directory_path() / "punkst_feature_stats_test.tsv";
    FeatureOccurrenceStats::writeTsv(outFile.string(), {"A", "B", "C", "D"}, 4, stats);
    std::ifstream in(outFile);
    std::string line;
    std::getline(in, line);
    check_true("feature occurrence header",
        line == "#feature\ttotal_count\tn_units_present\tn_units_count_gt1\tn_units_count_gt2\tn_units_count_gt_mean\tidf\tinfo_weight");
    std::getline(in, line);
    check_true("feature occurrence row A", line == "A\t3\t2\t1\t0\t1\t0.100000\t1.286517");
    std::getline(in, line);
    check_true("feature occurrence row B", line == "B\t4\t2\t1\t1\t1\t0.100000\t1.019540");
    std::getline(in, line);
    check_true("feature occurrence row C", line == "C\t5\t1\t1\t1\t1\t0.787589\t0.812458");
    std::getline(in, line);
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    double idf = 0.0, infoWeight = 0.0;
    check_true("feature occurrence absent idf parse",
        tokens.size() == 8 &&
        tokens[0] == "D" &&
        tokens[1] == "0" &&
        str2double(tokens[6], idf) &&
        str2double(tokens[7], infoWeight));
    check_close("feature occurrence absent idf", idf, 1.0, 1e-6, 0.0);
    check_close("feature occurrence absent info weight", infoWeight, 5.0, 1e-6, 0.0);
}

} // namespace

int32_t test(int32_t, char**) {
    try {
        test_gauss_legendre16();
        test_beta_helpers_and_evidence();
        test_multinomial_evidence();
        test_fit_eta_em();
        test_unit_factor_result_header_ranges();
        test_vst_feature_eligibility();
        test_gamma_poisson_dispersion_helpers();
        test_gamma_poisson_cluster_basis();
        test_dense_kmeans();
        test_low_rank_covariance();
        test_gamma_poisson_conditional_moments();
        test_gamma_poisson_diagonal_mixture();
        test_gamma_poisson_structured_mixture();
        test_gamma_poisson_posterior_and_sidecar();
        test_hexreader_feature_weights();
        test_feature_stats();
    } catch (const std::exception& ex) {
        std::cerr << "Unhandled exception in test command: " << ex.what() << "\n";
        return 1;
    }

    if (g_failures > 0) {
        std::cerr << g_failures << " test checks failed\n";
        return 1;
    }
    std::cerr << "All EB topic-activity tests passed\n";
    return 0;
}
