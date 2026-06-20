#include "punkst.h"
#include "dataunits.hpp"
#include "eb_topic_activity.hpp"
#include "numerical_utils.hpp"

#include <cmath>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check_close(const std::string& name, double got, double expected,
        double abs_tol, double rel_tol = 0.0) {
    const double tol = abs_tol + rel_tol * std::abs(expected);
    if (!std::isfinite(got) || std::abs(got - expected) > tol) {
        std::cerr << "FAIL " << name << ": got " << got
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

} // namespace

int32_t test(int32_t, char**) {
    try {
        test_gauss_legendre16();
        test_beta_helpers_and_evidence();
        test_multinomial_evidence();
        test_fit_eta_em();
        test_unit_factor_result_header_ranges();
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
