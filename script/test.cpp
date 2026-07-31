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
    bool supported = false;
};

struct FeatureTable {
    std::vector<std::string> header;
    std::unordered_map<std::string, std::vector<std::string>> rows;
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
        const Document& document) {
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
    }
    add_cofeature_references(
        lda.get_model(), lda.get_model().rowwise().sum(),
        document, expected);
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

std::vector<ExpectedDiagnostics> gamma_poisson_references(
        const std::filesystem::path& state_path,
        const Document& document) {
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
    }
    std::vector<double> abundance;
    model.get_topic_abundance(abundance);
    VectorXd topic_reference(n_topics);
    for (int32_t k = 0; k < n_topics; ++k) {
        topic_reference(k) = abundance[static_cast<size_t>(k)];
    }
    add_cofeature_references(
        expected_beta, topic_reference, document, expected);
    return expected;
}

void validate_tables(const FeatureTable& full,
        const FeatureTable& cheap,
        const std::vector<ExpectedDiagnostics>& expected,
        const std::string& model_name) {
    const std::vector<std::string> common_header{
        "Feature", "absDiff", "absDiffRate", "totCount", "nUnits",
        "log2Gain", "marginalDev", "conditionalDev", "factorDrift",
        "deletionTV", "topicInformation",
        "cofeatureCorroboration", "cofeatureConflict",
    };
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

std::string metadata_json() {
    std::ostringstream output;
    output
        << "{\"n_units\":1,\"n_modalities\":1,\"n_features\":"
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

    auto run = [&](const std::filesystem::path& prefix, bool cheap) {
        std::vector<std::string> arguments{
            "lda-transform",
            "--in-data", input_path.string(),
            "--in-meta", metadata_path.string(),
            "--in-model", model_path.string(),
            "--out-prefix", prefix.string(),
            "--min-count", "1",
            "--minibatch-size", "1",
            "--threads", "1",
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
        require(run_command(cmdLDATransform, std::move(arguments)) == 0,
            "LDA transform failed");
    };
    run(full_prefix, false);
    run(cheap_prefix, true);
    validate_tables(
        read_feature_table(
            full_prefix.string() + ".feature_residuals.tsv"),
        read_feature_table(
            cheap_prefix.string() + ".feature_residuals.tsv"),
        expected, "LDA");

    remove_transform_outputs(full_prefix);
    remove_transform_outputs(cheap_prefix);
    std::filesystem::remove(model_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(metadata_path);
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
    write_text(input_path,
        "doc_0\t5\t20\t0 8\t1 2\t2 3\t3 1\t4 6\n");
    write_text(metadata_path, metadata_json());

    Document document;
    document.ids = {0, 1, 2, 3, 4};
    document.cnts = {8.0, 2.0, 3.0, 1.0, 6.0};
    document.raw_ct_tot = document.ct_tot = 20.0;
    const std::vector<ExpectedDiagnostics> expected =
        gamma_poisson_references(state_path, document);

    auto run = [&](const std::filesystem::path& prefix, bool cheap) {
        std::vector<std::string> arguments{
            "gamma-pois-transform",
            "--in-data", input_path.string(),
            "--in-meta", metadata_path.string(),
            "--in-state", state_path.string(),
            "--out-prefix", prefix.string(),
            "--min-count", "1",
            "--minibatch-size", "1",
            "--threads", "1",
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
        require(run_command(
                cmdGammaPoisTransform, std::move(arguments)) == 0,
            "Gamma-Poisson transform failed");
    };
    run(full_prefix, false);
    run(cheap_prefix, true);
    validate_tables(
        read_feature_table(
            full_prefix.string() + ".feature_residuals.tsv"),
        read_feature_table(
            cheap_prefix.string() + ".feature_residuals.tsv"),
        expected, "Gamma-Poisson");

    remove_transform_outputs(full_prefix);
    remove_transform_outputs(cheap_prefix);
    std::filesystem::remove(state_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(metadata_path);
}

} // namespace

int32_t test(int32_t, char**) {
    try {
        test_cofeature_metrics();
        test_lda_output();
        test_gamma_poisson_output();
        std::cout << "Feature diagnostic tests passed\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
        return 1;
    }
    return 0;
}
