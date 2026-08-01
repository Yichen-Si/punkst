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

void validate_training_table(const FeatureTable& table,
        const std::vector<ExpectedDiagnostics>& expected,
        const std::string& model_name) {
    const std::vector<std::string> header{
        "Feature", "absDiff", "absDiffRate", "totCount", "nUnits",
        "log2Gain", "marginalDev", "conditionalDev", "factorDrift",
        "deletionTV", "topicInformation",
        "cofeatureCorroboration", "cofeatureConflict", "pull",
    };
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
                    && row[12] == "NA" && row[13] == "NA",
                model_name + " unsupported training diagnostics are not NA");
            require(std::abs(std::stod(row[10])
                    - expected[feature].topic_information) < 6e-5,
                model_name + " training topic information is incorrect");
            continue;
        }
        const std::array<double, 5> references{
            expected[feature].total_variation,
            expected[feature].topic_information,
            expected[feature].cofeature_corroboration,
            expected[feature].cofeature_conflict,
            expected[feature].pull,
        };
        for (size_t offset = 0; offset < references.size(); ++offset) {
            const double observed = std::stod(row[9 + offset]);
            const double tolerance = 6e-5
                * std::max(1.0, std::abs(references[offset]));
            require(std::abs(observed - references[offset]) < tolerance,
                model_name + " training diagnostic differs from reference");
        }
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
            bool training) {
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
        expected, "LDA");
    validate_training_table(
        read_feature_table(
            training_prefix.string() + ".feature_residuals.tsv"),
        training_expected, "LDA");

    Document second_document;
    second_document.ids = {0, 1, 2, 3, 4};
    second_document.cnts = {1.0, 10.0, 2.0, 6.0, 1.0};
    second_document.raw_ct_tot = second_document.ct_tot = 20.0;
    write_text(input_path,
        "doc_0\t5\t20\t0 8\t1 2\t2 3\t3 1\t4 6\n"
        "doc_1\t5\t20\t0 1\t1 10\t2 2\t3 6\t4 1\n");
    write_text(metadata_path, metadata_json(2));
    run(multi_prefix, false, false);
    validate_unit_averaged_deletion(
        read_feature_table(
            multi_prefix.string() + ".feature_residuals.tsv"),
        {expected, lda_references(model_path, second_document)},
        {document, second_document}, "LDA");

    remove_transform_outputs(full_prefix);
    remove_transform_outputs(cheap_prefix);
    remove_transform_outputs(training_prefix);
    remove_transform_outputs(multi_prefix);
    std::filesystem::remove(model_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(metadata_path);
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
            bool training) {
        std::vector<std::string> arguments{
            "gamma-pois-transform",
            "--in-data", input_path.string(),
            "--in-meta", metadata_path.string(),
            "--in-state", state_path.string(),
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
    validate_tables(
        read_feature_table(
            full_prefix.string() + ".feature_residuals.tsv"),
        read_feature_table(
            cheap_prefix.string() + ".feature_residuals.tsv"),
        expected, "Gamma-Poisson");
    validate_training_table(
        read_feature_table(
            training_prefix.string() + ".feature_residuals.tsv"),
        training_expected, "Gamma-Poisson");

    Document second_document;
    second_document.ids = {0, 1, 2, 3, 4};
    second_document.cnts = {1.0, 10.0, 2.0, 6.0, 1.0};
    second_document.raw_ct_tot = second_document.ct_tot = 20.0;
    write_text(input_path,
        "doc_0\t5\t20\t0 8\t1 2\t2 3\t3 1\t4 6\n"
        "doc_1\t5\t20\t0 1\t1 10\t2 2\t3 6\t4 1\n");
    write_text(metadata_path, metadata_json(2));
    run(multi_prefix, false, false);
    validate_unit_averaged_deletion(
        read_feature_table(
            multi_prefix.string() + ".feature_residuals.tsv"),
        {expected,
            gamma_poisson_references(state_path, second_document)},
        {document, second_document}, "Gamma-Poisson");

    remove_transform_outputs(full_prefix);
    remove_transform_outputs(cheap_prefix);
    remove_transform_outputs(training_prefix);
    remove_transform_outputs(multi_prefix);
    std::filesystem::remove(state_path);
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
