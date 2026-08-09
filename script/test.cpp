#include "clustering/uac.hpp"
#include "numerical_utils.hpp"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int32_t cmdUacFit(int argc, char** argv);
int32_t cmdUacTransform(int argc, char** argv);

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("UAC test failed: " + message);
    }
}

void write_text(
        const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    require(static_cast<bool>(output),
        "cannot write " + path.string());
    output << text;
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

std::pair<int32_t, std::string> run_command_capture_error(
    int32_t (*command)(int, char**), std::vector<std::string> arguments) {
    std::ostringstream captured;
    std::streambuf* previous = std::cerr.rdbuf(captured.rdbuf());
    int32_t status = 1;
    try {
        status = run_command(command, std::move(arguments));
    } catch (...) {
        std::cerr.rdbuf(previous);
        throw;
    }
    std::cerr.rdbuf(previous);
    return {status, captured.str()};
}

void remove_uac_outputs(const std::filesystem::path& prefix) {
    for (const std::string& suffix : {
            ".state.tsv", ".model.tsv", ".results.tsv",
            ".diagnostics.tsv", ".trace.tsv", ".separation.tsv",
            ".representatives.tsv", ".visual.axes.tsv",
            ".visual.model.tsv", ".visual.results.tsv"}) {
        std::filesystem::remove(prefix.string() + suffix);
    }
}

int32_t data_rows(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::string line;
    int32_t rows = -1;
    while (std::getline(input, line)) ++rows;
    return rows;
}

std::string first_data_field(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::string header, row;
    require(static_cast<bool>(std::getline(input, header))
            && static_cast<bool>(std::getline(input, row)),
        "missing data row in " + path.string());
    const size_t separator = row.find('\t');
    return row.substr(0, separator);
}

void test_simplex_metrics() {
    RowMajorMatrixXd compositions(5, 3);
    compositions <<
        0.1, 0.6, 0.3,
        0.0, 0.6, 0.4,
        0.5, 0.2, 0.3,
        0.1, 0.4, 0.5,
        0.2, 0.4, 0.4;
    const RowMajorMatrixXd hellinger = simplex_metric_coordinates(
        compositions, SimplexMetric::Hellinger);
    require((hellinger.rowwise().squaredNorm().array() - 1.0)
            .abs().maxCoeff() < 1e-12,
        "Hellinger coordinates are not unit normalized");
    require(std::abs(hellinger(0, 0) - std::sqrt(0.1)) < 1e-12,
        "Hellinger coordinates are not square-root proportions");

    RowMajorMatrixXd scaled = compositions;
    for (Eigen::Index row = 0; row < scaled.rows(); ++row) {
        scaled.row(row) *= static_cast<double>(row + 2);
    }
    require((simplex_metric_coordinates(
                scaled, SimplexMetric::Hellinger) - hellinger)
            .cwiseAbs().maxCoeff() < 1e-12,
        "Hellinger coordinates are not row-scale invariant");

    CosineKnnOptions knn;
    knn.n_neighbors = 1;
    knn.backend = CosineKnnBackend::Flat;
    knn.n_threads = 1;
    const CosineKnnResult cosine_graph = simplex_knn(
        compositions, SimplexMetric::Cosine, knn);
    const CosineKnnResult hellinger_graph = simplex_knn(
        compositions, SimplexMetric::Hellinger, knn);
    const std::pair<int32_t, int32_t> distinguishing_edge{0, 4};
    require(std::find(cosine_graph.graph.edges.begin(),
                cosine_graph.graph.edges.end(), distinguishing_edge)
            == cosine_graph.graph.edges.end()
        && std::find(hellinger_graph.graph.edges.begin(),
                hellinger_graph.graph.edges.end(), distinguishing_edge)
            != hellinger_graph.graph.edges.end(),
        "Hellinger and cosine did not produce their expected distinct graph");
    const auto edge = std::find(hellinger_graph.graph.edges.begin(),
        hellinger_graph.graph.edges.end(), distinguishing_edge);
    const size_t edge_index = static_cast<size_t>(
        edge - hellinger_graph.graph.edges.begin());
    const double expected_affinity = std::sqrt(0.1 * 0.2)
        + std::sqrt(0.6 * 0.4) + std::sqrt(0.3 * 0.4);
    require(std::abs(hellinger_graph.graph.weights[edge_index]
                - expected_affinity) < 1e-12,
        "Hellinger graph weight is not Bhattacharyya affinity");

    DenseKMeansOptions kmeans;
    kmeans.n_clusters = 2;
    kmeans.seed = 5;
    const DenseKMeansResult clustering = simplex_dense_kmeans(
        compositions, SimplexMetric::Hellinger, kmeans);
    require(clustering.counts.sum() == compositions.rows(),
        "Hellinger k-means did not assign every composition");

    Eigen::VectorXi membership(5);
    membership << 0, 0, 1, 1, 1;
    kmeans.n_clusters = 3;
    const Eigen::VectorXi reconciled = reconcile_simplex_communities(
        membership, 2, 3, compositions, SimplexMetric::Hellinger, kmeans);
    require(reconciled.minCoeff() == 0 && reconciled.maxCoeff() == 2,
        "Hellinger community reconciliation did not produce three groups");

    bool rejected_negative = false;
    RowMajorMatrixXd invalid = compositions;
    invalid(0, 0) = -0.1;
    try {
        (void)simplex_metric_coordinates(
            invalid, SimplexMetric::Hellinger);
    } catch (const std::invalid_argument&) {
        rejected_negative = true;
    }
    require(rejected_negative,
        "Hellinger metric accepted a negative composition");
}

void test_visualization_projection() {
    require(uac::VisualizationOptions{}.whitening
            == uac::VisualizationWhitening::Mixture,
        "mixture visualization whitening is not the API default");
    const Eigen::MatrixXd helmert = normalized_helmert(3);
    uac::Dataset data;
    data.coordinates.resize(8, 2);
    data.coordinates <<
        -1.6, -0.4,
        -1.1,  0.5,
        -0.7, -0.8,
        -0.2,  0.9,
         0.3, -0.6,
         0.8,  1.1,
         1.2, -0.2,
         1.7,  0.7;
    data.centers = ilr_inverse(data.coordinates, helmert);
    for (int32_t document = 0; document < data.coordinates.rows();
            ++document) {
        data.identifiers.push_back("visual_" + std::to_string(document));
    }

    uac::Model model;
    model.covariance_kind = uac::CovarianceKind::FactorAnalytic;
    model.weights.resize(2);
    model.weights << 0.4, 0.6;
    model.means.resize(2, 2);
    model.means << -0.9, 0.35, 0.75, -0.15;
    LowRankDiagonalCovariance first, second, target;
    first.diagonal.resize(2);
    first.diagonal << 0.5, 0.4;
    first.factor.resize(2, 1);
    first.factor << 0.3, 0.1;
    second.diagonal.resize(2);
    second.diagonal << 0.3, 0.6;
    second.factor.resize(2, 1);
    second.factor << -0.2, 0.25;
    target.diagonal = Eigen::VectorXd::Constant(2, 0.5);
    target.factor = RowMajorMatrixXd::Zero(2, 1);
    model.factor_covariances = {first, second};
    model.factor_shrinkage_target = target;

    const Eigen::VectorXd mixture_mean =
        model.means.transpose() * model.weights;
    Eigen::MatrixXd mean_kernel = Eigen::MatrixXd::Zero(2, 2);
    Eigen::MatrixXd average_covariance = Eigen::MatrixXd::Zero(2, 2);
    for (int32_t component = 0; component < 2; ++component) {
        const Eigen::VectorXd difference =
            model.means.row(component).transpose() - mixture_mean;
        mean_kernel += model.weights(component)
            * difference * difference.transpose();
        average_covariance += model.weights(component)
            * model.factor_covariances[component].dense();
    }

    uac::VisualizationOptions options;
    options.whitening = uac::VisualizationWhitening::Mixture;
    options.dimensions = 2;
    options.n_threads = 2;
    options.covariance_floor = 1e-8;
    const uac::VisualizationResult visualization =
        uac::make_visualization(data, model, helmert, options);
    const Eigen::MatrixXd expected_whitening =
        average_covariance + mean_kernel;
    require((visualization.whitening_covariance - expected_whitening)
            .cwiseAbs().maxCoeff() < 1e-12,
        "mixture visualization whitening covariance is incorrect");

    Eigen::MatrixXd covariance_kernel = Eigen::MatrixXd::Zero(2, 2);
    for (int32_t component = 0; component < 2; ++component) {
        const Eigen::MatrixXd difference =
            model.factor_covariances[component].dense()
            - average_covariance;
        covariance_kernel += model.weights(component) * difference
            * expected_whitening.inverse() * difference;
    }
    const Eigen::MatrixXd full_kernel = mean_kernel
        * expected_whitening.inverse() * mean_kernel + covariance_kernel;
    auto check_view = [&](const uac::VisualizationProjection& view,
            const Eigen::MatrixXd& kernel) {
        require((view.projection.transpose() * expected_whitening
                * view.projection - Eigen::MatrixXd::Identity(2, 2))
                .cwiseAbs().maxCoeff() < 1e-10,
            "visualization axes are not covariance-orthonormal");
        require((kernel * view.projection
                - expected_whitening * view.projection
                    * view.eigenvalues.asDiagonal())
                .cwiseAbs().maxCoeff() < 1e-9,
            "visualization generalized eigen residual is too large");
        require((view.topic_contrasts
                - helmert.transpose() * view.projection)
                .cwiseAbs().maxCoeff() < 1e-12,
            "visualization topic contrasts are incorrect");
        require(view.topic_contrasts.colwise().sum().cwiseAbs().maxCoeff()
                < 1e-12,
            "visualization topic contrasts do not sum to zero");
        for (Eigen::Index axis = 0;
                axis < view.topic_contrasts.cols(); ++axis) {
            Eigen::Index pivot = 0;
            view.topic_contrasts.col(axis).cwiseAbs().maxCoeff(&pivot);
            require(view.topic_contrasts(pivot, axis) >= 0.0,
                "visualization axis sign is not canonical");
        }
        require((view.component_means - model.means * view.projection)
                .cwiseAbs().maxCoeff() < 1e-12,
            "projected component means are incorrect");
        for (int32_t component = 0; component < 2; ++component) {
            const Eigen::MatrixXd expected = view.projection.transpose()
                * model.factor_covariances[component].dense()
                * view.projection;
            require((view.component_covariances[component] - expected)
                    .cwiseAbs().maxCoeff() < 1e-12,
                "projected component covariance is incorrect");
        }
    };
    check_view(visualization.mean, mean_kernel);
    check_view(visualization.full, full_kernel);

    uac::Model dense_model;
    dense_model.covariance_kind = uac::CovarianceKind::Dense;
    dense_model.weights = model.weights;
    dense_model.means = model.means;
    dense_model.covariances = {first.dense(), second.dense()};
    dense_model.shrinkage_target = target.dense();
    const uac::VisualizationResult dense_visualization =
        uac::make_visualization(data, dense_model, helmert, options);
    require((dense_visualization.mean.projection
            - visualization.mean.projection).cwiseAbs().maxCoeff() < 1e-12
        && (dense_visualization.full.projection
            - visualization.full.projection).cwiseAbs().maxCoeff() < 1e-12,
        "dense and factor visualization projections differ");

    uac::Model inactive_model = model;
    inactive_model.weights.resize(3);
    inactive_model.weights << 0.4, 0.6, 0.0;
    inactive_model.means.conservativeResize(3, 2);
    inactive_model.means.row(2) << 20.0, -20.0;
    inactive_model.factor_covariances.push_back(target);
    const uac::VisualizationResult inactive_visualization =
        uac::make_visualization(data, inactive_model, helmert, options);
    require((inactive_visualization.whitening_covariance
            - visualization.whitening_covariance).cwiseAbs().maxCoeff()
            < 1e-12
        && inactive_visualization.mean.component_means.rows() == 3,
        "inactive visualization component handling is incorrect");

    options.whitening = uac::VisualizationWhitening::Sample;
    const uac::VisualizationResult empirical_visualization =
        uac::make_visualization(data, model, helmert, options);
    Eigen::MatrixXd centered = data.coordinates;
    centered.rowwise() -= mixture_mean.transpose();
    const Eigen::MatrixXd empirical_expected =
        centered.transpose() * centered / data.coordinates.rows();
    require((empirical_visualization.whitening_covariance
            - empirical_expected).cwiseAbs().maxCoeff() < 1e-12,
        "sample visualization whitening covariance is incorrect");
    uac::Dataset singular = data;
    singular.coordinates.col(1) = 2.0 * singular.coordinates.col(0);
    const uac::VisualizationResult sample =
        uac::make_visualization(singular, model, helmert, options);
    require(positive_definite(sample.whitening_covariance),
        "singular sample whitening covariance was not regularized");
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

    const auto invalid_fit = run_command_capture_error(cmdUacFit, {
        "uac-fit",
        "--in-theta", center_path.string(),
        "--out-prefix", fit_prefix.string() + ".invalid",
        "--n-clusters", "2",
        "--visual-whitening", "invalid",
    });
    require(invalid_fit.first == 1
            && invalid_fit.second.find("visualization whitening")
                != std::string::npos,
        "UAC fit did not reject invalid visualization options upfront");

    std::vector<std::string> fit_arguments{
        "uac-fit",
        "--in-theta", center_path.string(),
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
        "--leiden-starts", "1",
        "--leiden-neighbors", "3",
        "--initialization-metric", "hellinger",
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
            && state_header == "##punkst_uac_state_v12",
        "direct topic-to-UAC fit did not write a v12 state");
    std::string state_line;
    bool saw_initialization_metric = false;
    while (std::getline(state_input, state_line)) {
        if (state_line == "##initialization_metric\thellinger") {
            saw_initialization_metric = true;
        }
    }
    require(saw_initialization_metric,
        "UAC state did not preserve the Hellinger initialization metric");
    const uac::State fitted_state = uac::read_state(
        fit_prefix.string() + ".state.tsv");
    require(fitted_state.initialization_metric == SimplexMetric::Hellinger,
        "UAC state did not round-trip the Hellinger initialization metric");

    std::ifstream state_text_input(fit_prefix.string() + ".state.tsv");
    std::ostringstream state_text_buffer;
    state_text_buffer << state_text_input.rdbuf();
    std::string legacy_state_text = state_text_buffer.str();
    legacy_state_text.replace(0, std::string("##punkst_uac_state_v12").size(),
        "##punkst_uac_state_v11");
    const std::string metric_record =
        "##initialization_metric\thellinger\n";
    const size_t metric_position = legacy_state_text.find(metric_record);
    require(metric_position != std::string::npos,
        "UAC state metric record is missing");
    legacy_state_text.erase(metric_position, metric_record.size());
    const std::filesystem::path legacy_state_path =
        fit_prefix.string() + ".legacy_v11.state.tsv";
    write_text(legacy_state_path, legacy_state_text);
    const uac::State legacy_state = uac::read_state(
        legacy_state_path.string());
    require(legacy_state.initialization_metric == SimplexMetric::Cosine,
        "v11 UAC state did not default its initialization metric to cosine");
    std::filesystem::remove(legacy_state_path);

    const auto invalid_transform = run_command_capture_error(
        cmdUacTransform, {
            "uac-transform",
            "--in-state", fit_prefix.string() + ".state.tsv",
            "--in-theta", center_path.string(),
            "--out-prefix", transform_prefix.string() + ".invalid",
            "--visual-dim", "0",
        });
    require(invalid_transform.first == 1
            && invalid_transform.second.find("--visual-dim must be positive")
                != std::string::npos,
        "UAC transform did not reject invalid visualization options upfront");

    std::vector<std::string> transform_arguments{
        "uac-transform",
        "--in-state", fit_prefix.string() + ".state.tsv",
        "--in-theta", center_path.string(),
        "--unit-icol-id", "1",
        "--in-model", model_path.string(),
        "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(),
        "--count-icol-id", "1",
        "--out-prefix", transform_prefix.string(),
        "--particles", "16",
        "--threads", "1",
        "--n-representatives", "1",
        "--visual-whitening", "sample",
    };
    require(run_command(cmdUacTransform,
            std::move(transform_arguments)) == 0,
        "direct topic-to-UAC transform failed");

    require(data_rows(transform_prefix.string() + ".results.tsv") == 12,
        "direct topic-to-UAC transform wrote the wrong row count");
    require(data_rows(fit_prefix.string() + ".visual.axes.tsv") == 6,
        "UAC fit wrote the wrong one-dimensional axis row count");
    require(data_rows(fit_prefix.string() + ".visual.model.tsv") == 4,
        "UAC fit wrote the wrong projected-model row count");
    require(data_rows(fit_prefix.string() + ".visual.results.tsv") == 12,
        "UAC fit wrote the wrong projected-document row count");
    require(data_rows(transform_prefix.string() + ".visual.results.tsv")
            == 12,
        "UAC transform wrote the wrong projected-document row count");
    require(first_data_field(
            fit_prefix.string() + ".visual.axes.tsv") == "mixture",
        "UAC fit did not use default mixture visualization whitening");
    require(first_data_field(
            transform_prefix.string() + ".visual.axes.tsv") == "sample",
        "UAC transform did not use requested sample visualization whitening");

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
        test_simplex_metrics();
        test_visualization_projection();
        test_topic_to_uac_handoff();
        std::cout << "UAC tests passed\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
        return 1;
    }
    return 0;
}
