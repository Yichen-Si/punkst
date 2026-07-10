#include "punkst.h"
#include "dataunits.hpp"
#include "eb_topic_activity.hpp"
#include "gamma_pois_dispersion.hpp"
#include "numerical_utils.hpp"
#include "topic_svb.hpp"
#include "tiles2bins.hpp"
#include "vst.hpp"

#include <algorithm>
#include <cmath>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
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
