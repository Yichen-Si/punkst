#include "clustering/uac.hpp"
#include "clustering/uac_stochastic_internal.hpp"
#include "gamma_pois_topic.hpp"
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
int32_t cmdLeiden(int argc, char** argv);
int32_t cmdLinearEmbed(int argc, char** argv);
int32_t cmdLDATransform(int argc, char** argv);

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
            ".initialization.tsv", ".initialization.results.tsv",
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

std::string metadata_value(const std::filesystem::path& path,
        const std::string& key) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    const std::string prefix = "##" + key + "\t";
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind(prefix, 0) == 0) return line.substr(prefix.size());
    }
    throw std::runtime_error("UAC test failed: missing metadata " + key);
}

std::string table_header(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("#id", 0) == 0) return line;
    }
    return {};
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
}

std::string read_table_data(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::ostringstream out;
    std::string line;
    while (std::getline(input, line)) {
        if (line.rfind("##", 0) != 0) out << line << '\n';
    }
    return out.str();
}

void require_scientific_fields(const std::filesystem::path& path,
        int32_t leading_fields) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::string line;
    require(static_cast<bool>(std::getline(input, line)),
        "missing header in " + path.string());
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::string field;
        int32_t column = 0;
        while (std::getline(row, field, '\t')) {
            if (column++ < leading_fields) continue;
            const size_t decimal = field.find('.');
            const size_t exponent = field.find_first_of("eE");
            require(decimal != std::string::npos
                    && exponent == decimal + 5,
                "field is not formatted as %.4e in " + path.string()
                    + ": " + field);
        }
    }
}

void require_uac_results_format(const std::filesystem::path& path,
        int32_t components) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::string line;
    require(static_cast<bool>(std::getline(input, line)),
        "missing UAC results header");
    std::ostringstream expected;
    expected << "#id\tC1\tP1\tC2\tP2\tentropy";
    for (int32_t c = 0; c < components; ++c) expected << '\t' << c;
    require(line == expected.str(),
        "unexpected UAC results header: " + line);
    int32_t rows = 0;
    while (std::getline(input, line)) {
        ++rows;
        std::istringstream row(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(row, field, '\t')) fields.push_back(field);
        require(fields.size() == static_cast<size_t>(6 + components),
            "unexpected UAC results field count");
        for (const int32_t column : {2, 4, 5}) {
            const size_t decimal = fields[column].find('.');
            const size_t exponent = fields[column].find_first_of("eE");
            require(decimal != std::string::npos
                    && exponent == decimal + 5,
                "UAC result field is not formatted as %.4e: "
                    + fields[column]);
        }
        for (int32_t column = 6; column < 6 + components; ++column) {
            const size_t decimal = fields[column].find('.');
            const size_t exponent = fields[column].find_first_of("eE");
            require(decimal != std::string::npos
                    && exponent == decimal + 5,
                "UAC component probability is not formatted as %.4e: "
                    + fields[column]);
        }
    }
    require(rows > 0, "UAC results table is empty");
}

void require_uac_trace_top_probability(
        const std::filesystem::path& path, const std::string& phase) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::string line;
    require(static_cast<bool>(std::getline(input, line)),
        "missing UAC trace header");
    std::vector<std::string> header;
    std::istringstream header_stream(line);
    std::string field;
    while (std::getline(header_stream, field, '\t')) {
        header.push_back(field);
    }
    const auto found = std::find(
        header.begin(), header.end(), "mean_top_probability");
    require(found != header.end(),
        "UAC trace omitted mean_top_probability");
    const size_t column = std::distance(header.begin(), found);
    bool saw_evaluation = false;
    while (std::getline(input, line)) {
        std::vector<std::string> fields;
        std::istringstream row(line);
        while (std::getline(row, field, '\t')) fields.push_back(field);
        if (fields.size() <= column || fields.size() < 3
            || fields[1] != phase || fields[2] != "evaluation") {
            continue;
        }
        saw_evaluation = true;
        require(fields[column] != "NA", phase +
            " trace has no mean top probability");
        const double value = std::stod(fields[column]);
        require(value > 0.0 && value <= 1.0,
            phase + " mean top probability is outside (0, 1]");
    }
    require(saw_evaluation,
        "UAC trace omitted " + phase + " evaluations");
}

void require_initialization_results_format(
        const std::filesystem::path& path, int32_t documents,
        int32_t kmeans_starts = 1, int32_t leiden_starts = 1) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::string line;
    std::ostringstream expected_header;
    expected_header << "#id";
    for (int32_t start = 1; start <= kmeans_starts; ++start) {
        expected_header << "\tkmeans";
        if (start > 1) expected_header << start;
    }
    for (int32_t start = 1; start <= leiden_starts; ++start) {
        expected_header << "\tleiden";
        if (start > 1) expected_header << start;
        expected_header << "_raw\tleiden";
        if (start > 1) expected_header << start;
    }
    require(static_cast<bool>(std::getline(input, line))
            && line == expected_header.str(),
        "unexpected UAC initialization results header: " + line);
    int32_t rows = 0;
    while (std::getline(input, line)) {
        std::istringstream row(line);
        std::vector<std::string> fields;
        std::string field;
        while (std::getline(row, field, '\t')) fields.push_back(field);
        require(fields.size()
                == static_cast<size_t>(1 + kmeans_starts
                    + 2 * leiden_starts),
            "unexpected UAC initialization results field count");
        require(fields[0] == "doc_" + std::to_string(rows),
            "UAC initialization results changed unit order");
        for (int32_t start = 0; start < kmeans_starts; ++start) {
            const size_t column = static_cast<size_t>(1 + start);
            const int32_t cluster = std::stoi(fields[column]);
            require(cluster >= 0 && cluster < 2,
                "invalid UAC initialization cluster assignment");
        }
        for (int32_t start = 0; start < leiden_starts; ++start) {
            const size_t raw_column = static_cast<size_t>(
                1 + kmeans_starts + 2 * start);
            require(std::stoi(fields[raw_column]) >= 0,
                "invalid raw UAC Leiden community assignment");
            const int32_t cluster = std::stoi(fields[raw_column + 1]);
            require(cluster >= 0 && cluster < 2,
                "invalid resolved UAC Leiden cluster assignment");
        }
        ++rows;
    }
    require(rows == documents,
        "unexpected UAC initialization results row count");
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
    require(!uac::VisualizationOptions{}.include_full,
        "full visualization is unexpectedly enabled by default");
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
    options.include_full = true;
    const uac::VisualizationResult visualization =
        uac::make_visualization(data, model, helmert, options);
    uac::VisualizationMoments model_moments;
    model_moments.weights = model.weights;
    model_moments.means = model.means;
    model_moments.covariances = {first.dense(), second.dense()};
    const uac::VisualizationResult moment_visualization =
        uac::make_visualization(data, model_moments, helmert, options);
    require((moment_visualization.mean.projection
                - visualization.mean.projection).cwiseAbs().maxCoeff()
                < 1e-12
            && (moment_visualization.full.projection
                - visualization.full.projection).cwiseAbs().maxCoeff()
                < 1e-12,
        "moment and model visualization projections differ");
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

    options.include_full = false;
    const uac::VisualizationResult mean_visualization =
        uac::make_visualization(data, model, helmert, options);
    require(mean_visualization.mean.projection.cols() == 2
            && mean_visualization.full.projection.cols() == 0,
        "mean-only visualization computed a full projection");
    options.include_full = true;

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

    Eigen::VectorXi assignments(8);
    assignments << 0, 0, 0, 0, 1, 1, 1, 1;
    const uac::VisualizationMoments hard_moments =
        uac::summarize_hard_partition(data.coordinates, assignments, 2);
    require((hard_moments.weights
                - Eigen::Vector2d::Constant(0.5)).cwiseAbs().maxCoeff()
                < 1e-12,
        "hard-partition visualization weights are incorrect");
    options.whitening = uac::VisualizationWhitening::Mixture;
    const uac::VisualizationResult hard_mixture =
        uac::make_visualization(data, hard_moments, helmert, options);
    options.whitening = uac::VisualizationWhitening::Sample;
    const uac::VisualizationResult hard_sample =
        uac::make_visualization(data, hard_moments, helmert, options);
    const uac::VisualizationSampleMoments sample_moments =
        uac::summarize_visualization_sample(data.coordinates, 2);
    const uac::VisualizationMeans hard_means =
        uac::summarize_hard_partition_means(
            data.coordinates, assignments, 2);
    options.whitening = uac::VisualizationWhitening::Mixture;
    const uac::VisualizationResult mean_only =
        uac::make_mean_visualization(
            hard_means, helmert, options, sample_moments);
    require(mean_only.mean.projection.cols() == 1
            && mean_only.full.projection.cols() == 0
            && (mean_only.mean.projection.col(0)
                - hard_mixture.mean.projection.col(0)).cwiseAbs().maxCoeff()
                < 1e-10
            && std::abs(mean_only.mean.topic_contrasts.col(0).sum()) < 1e-12,
        "mean-only hard-partition projection is incorrect");
    options.whitening = uac::VisualizationWhitening::Sample;
    const uac::VisualizationResult cached_hard_sample =
        uac::make_visualization(
            data, hard_moments, helmert, options, sample_moments);
    require((hard_mixture.whitening_covariance
                - hard_sample.whitening_covariance)
                .cwiseAbs().maxCoeff() < 1e-12,
        "complete hard partition has different sample and mixture whitening");
    require((cached_hard_sample.whitening_covariance
                - hard_sample.whitening_covariance)
                .cwiseAbs().maxCoeff() < 1e-12,
        "cached and direct sample whitening differ");
}

void test_factor_covariance_change_math() {
    uac::Model before;
    before.covariance_kind = uac::CovarianceKind::FactorAnalytic;
    before.factor_diagonal_mode = uac::FactorDiagonalMode::Component;
    before.weights = Eigen::Vector2d(0.45, 0.55);
    before.means.resize(2, 5);
    before.means << 0.1, -0.2, 0.3, 0.0, 0.4,
                    -0.1, 0.5, 0.2, -0.3, 0.1;
    before.factor_covariances.resize(2);
    before.factor_covariances[0].diagonal =
        (Eigen::VectorXd(5) << 0.4, 0.7, 0.5, 0.9, 0.6).finished();
    before.factor_covariances[0].factor.resize(5, 2);
    before.factor_covariances[0].factor <<
        0.2, -0.1, 0.3, 0.05, -0.2, 0.4, 0.1, 0.2, -0.3, 0.15;
    before.factor_covariances[1].diagonal =
        (Eigen::VectorXd(5) << 0.8, 0.3, 0.6, 0.5, 0.7).finished();
    before.factor_covariances[1].factor.resize(5, 2);
    before.factor_covariances[1].factor <<
        -0.1, 0.2, 0.15, -0.25, 0.35, 0.1, -0.2, 0.3, 0.05, -0.15;

    uac::Model after = before;
    after.weights << 0.48, 0.52;
    after.means.row(0).array() += 0.03;
    after.factor_covariances[0].diagonal.array() +=
        Eigen::ArrayXd::LinSpaced(5, -0.04, 0.06);
    after.factor_covariances[0].factor.resize(5, 3);
    after.factor_covariances[0].factor <<
        0.18, -0.08, 0.03, 0.28, 0.02, -0.04, -0.16, 0.36, 0.02,
        0.08, 0.24, -0.01, -0.27, 0.12, 0.05;

    double dense_change = (before.weights - after.weights).cwiseAbs().sum();
    for (int32_t c = 0; c < 2; ++c) {
        const auto dense = [](const LowRankDiagonalCovariance& covariance) {
            Eigen::MatrixXd out = covariance.diagonal.asDiagonal();
            out.noalias() += covariance.factor * covariance.factor.transpose();
            return out;
        };
        const Eigen::MatrixXd left = dense(before.factor_covariances[c]);
        const Eigen::MatrixXd right = dense(after.factor_covariances[c]);
        require(std::abs(uac::detail::model_covariance_frobenius_norm(
                before, c) - left.norm()) < 1e-12,
            "factor covariance Frobenius norm differs from dense reference");
        require(std::abs(uac::detail::model_covariance_frobenius_difference(
                before, after, c) - (right - left).norm()) < 1e-12,
            "factor covariance Frobenius difference differs from dense reference");
        dense_change = std::max(dense_change,
            (before.means.row(c) - after.means.row(c)).norm()
                / std::max(1.0, before.means.row(c).norm()));
        dense_change = std::max(dense_change,
            (right - left).norm() / std::max(1.0, left.norm()));
    }
    require(std::abs(uac::detail::model_parameter_change(before, after)
            - dense_change) < 1e-12,
        "factor parameter change differs from dense reference");
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

    const std::filesystem::path lda_prefix = base.string() + ".lda";
    require(run_command(cmdLDATransform, {
            "lda-transform",
            "--in-data", input_path.string(),
            "--in-meta", metadata_path.string(),
            "--in-model", model_path.string(),
            "--out-prefix", lda_prefix.string(),
            "--min-count", "1",
            "--threads", "1",
            "--seed", "43",
        }) == 0,
        "LDA transform for output-format regression test failed");
    require_scientific_fields(
        lda_prefix.string() + ".results.tsv", 2);

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

    const auto invalid_center_floor = run_command_capture_error(cmdUacFit, {
        "uac-fit",
        "--in-theta", center_path.string(),
        "--out-prefix", fit_prefix.string() + ".invalid_floor",
        "--n-clusters", "2",
        "--handoff", "map",
        "--center-floor", "0",
        "--threads", "1",
    });
    require(invalid_center_floor.first == 1
            && invalid_center_floor.second.find(
                "--center-floor must be positive and finite")
                != std::string::npos,
        "UAC fit did not reject an invalid center floor upfront");

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
        "--center-floor", "5e-5",
        "--particles", "16",
        "--fisher-refinement-iterations", "2",
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
    require_uac_results_format(
        fit_prefix.string() + ".results.tsv", 2);
    require_initialization_results_format(
        fit_prefix.string() + ".initialization.results.tsv", 12);
    require_scientific_fields(
        fit_prefix.string() + ".model.tsv", 2);
    require_scientific_fields(
        fit_prefix.string() + ".separation.tsv", 2);
    require_scientific_fields(
        fit_prefix.string() + ".visual.results.tsv", 1);
    require_uac_trace_top_probability(
        fit_prefix.string() + ".trace.tsv", "particle_em");
    const std::filesystem::path fit_diagnostics =
        fit_prefix.string() + ".diagnostics.tsv";
    require(table_header(fit_diagnostics).empty(),
        "UAC wrote per-unit diagnostics without --diagnosis-per-unit");
    require(metadata_value(fit_diagnostics, "initialization_seconds")
            .find('e') != std::string::npos,
        "UAC run diagnostics are not in scientific notation");
    const std::string default_initialization =
        read_text(fit_prefix.string() + ".initialization.tsv");
    require(default_initialization.find("measurement_mode\tht")
                != std::string::npos
            && default_initialization.find("measurement_target\t1024")
                != std::string::npos
            && default_initialization.find("candidate_score_target\t1024")
                != std::string::npos,
        "UAC did not use the default HT-1024 initializer");

    const std::filesystem::path legacy_init_prefix =
        base.string() + ".init_legacy";
    const std::filesystem::path full_init_prefix =
        base.string() + ".init_full";
    const std::filesystem::path ht_init_prefix =
        base.string() + ".init_ht";
    const std::filesystem::path ht_repeat_prefix =
        base.string() + ".init_ht_repeat";
    auto initialization_arguments = [&](const std::filesystem::path& prefix,
                                        const std::string& mode) {
        return std::vector<std::string>{
            "uac-fit", "--in-theta", center_path.string(),
            "--unit-icol-id", "1", "--in-model", model_path.string(),
            "--in-data", input_path.string(), "--in-meta",
            metadata_path.string(), "--count-icol-id", "1",
            "--out-prefix", prefix.string(), "--n-clusters", "2",
            "--initialization-only", "--init-measurement-mode", mode,
            "--init-measurement-target", "4",
            "--kmeans-starts", "2", "--leiden-starts", "2",
            "--leiden-neighbors", "3", "--initialization-metric", "hellinger",
            "--cluster-covariance-rank", "0", "--threads", "1",
            "--seed", "43"};
    };
    require(run_command(cmdUacFit,
            initialization_arguments(legacy_init_prefix, "legacy")) == 0,
        "legacy initialization-only fit failed");
    require_initialization_results_format(
        legacy_init_prefix.string() + ".initialization.results.tsv", 12,
        2, 2);
    require(data_rows(legacy_init_prefix.string() + ".visual.axes.tsv") > 0,
        "initialization-only fit omitted visualization axes");
    require(data_rows(legacy_init_prefix.string() + ".visual.model.tsv") == 2,
        "initialization-only fit omitted the projected initializer model");
    require(data_rows(legacy_init_prefix.string() + ".visual.results.tsv")
            == 12,
        "initialization-only fit omitted projected input centers");
    require(run_command(cmdUacFit,
            initialization_arguments(full_init_prefix, "full")) == 0,
        "cached full initialization-only fit failed");
    require(read_text(legacy_init_prefix.string() + ".model.tsv")
            == read_text(full_init_prefix.string() + ".model.tsv"),
        "cached full corrected-moment initializer differs from legacy");
    require(read_text(legacy_init_prefix.string()
                + ".initialization.results.tsv")
            == read_text(full_init_prefix.string()
                + ".initialization.results.tsv"),
        "initialization-only output changed across measurement modes");
    require(read_text(full_init_prefix.string() + ".initialization.tsv")
            .find("measurement_covariance_evaluations\t12")
                != std::string::npos,
        "cached full initializer did not evaluate each measurement once");
    require(read_text(full_init_prefix.string() + ".initialization.tsv")
            .find("candidate_score_target\t512") != std::string::npos,
        "full initializer did not resolve its candidate-score target to 512");
    require(run_command(cmdUacFit,
            initialization_arguments(ht_init_prefix, "ht")) == 0,
        "HT initialization-only fit failed");
    require(run_command(cmdUacFit,
            initialization_arguments(ht_repeat_prefix, "ht")) == 0,
        "repeated HT initialization-only fit failed");
    require(read_text(ht_init_prefix.string() + ".model.tsv")
            == read_text(ht_repeat_prefix.string() + ".model.tsv"),
        "HT corrected-moment initializer is not deterministic");
    require(read_text(ht_init_prefix.string() + ".initialization.tsv")
            .find("measurement_mode\tht") != std::string::npos
            && read_text(ht_init_prefix.string() + ".initialization.tsv")
                .find("candidate_score_target\t4") != std::string::npos,
        "HT initializer diagnostics omitted its mode-dependent target");

    const std::filesystem::path resident_prefix =
        base.string() + ".subsample_resident";
    const std::filesystem::path disk_prefix =
        base.string() + ".subsample_disk";
    const std::filesystem::path auto_prefix =
        base.string() + ".subsample_auto";
    const std::filesystem::path stream_cache =
        base.string() + ".particle_cache";
    auto subsample_arguments = [&](const std::filesystem::path& prefix,
                                   const std::string& storage) {
        return std::vector<std::string>{
            "uac-fit", "--in-theta", center_path.string(),
            "--unit-icol-id", "1", "--in-model", model_path.string(),
            "--in-data", input_path.string(), "--in-meta",
            metadata_path.string(), "--count-icol-id", "1",
            "--out-prefix", prefix.string(), "--n-clusters", "2",
            "--particles", "16", "--particle-engine", "stream",
            "--particle-fit-schedule", "subsample",
            "--fit-subsample-storage", storage,
            "--fit-subsample-memory-budget", "1M",
            "--fit-subsample-target", "2",
            "--fit-subsample-base-fraction", "0.2",
            "--fit-subsample-min-updates", "2",
            "--fit-subsample-max-updates", "2",
            "--fit-subsample-topup-rounds", "2",
            "--fit-document-budget", "4", "--fit-tail", "off",
            "--stream-cache", stream_cache.string(),
            "--stream-block-documents", "4",
            "--stream-particle-storage", "positions",
            "--kmeans-starts", "1", "--leiden-starts", "0",
            "--max-iter", "2", "--cluster-covariance-rank", "0",
            "--threads", "1", "--n-representatives", "1",
            "--seed", "43"};
    };
    require(run_command(cmdUacFit,
            subsample_arguments(resident_prefix, "resident")) == 0,
        "resident streamed subsample fit failed");
    require(run_command(cmdUacFit,
            subsample_arguments(disk_prefix, "disk")) == 0,
        "disk streamed subsample fit failed");
    std::vector<std::string> auto_arguments =
        subsample_arguments(auto_prefix, "auto");
    const auto budget_position = std::find(
        auto_arguments.begin(), auto_arguments.end(),
        "--fit-subsample-memory-budget");
    require(budget_position != auto_arguments.end(),
        "subsample test memory option is missing");
    *(budget_position + 1) = "5K";
    require(run_command(cmdUacFit, std::move(auto_arguments)) == 0,
        "automatic streamed subsample promotion failed");
    require(read_text(resident_prefix.string() + ".model.tsv")
            == read_text(disk_prefix.string() + ".model.tsv"),
        "resident and disk subsample models differ");
    require(read_table_data(resident_prefix.string() + ".results.tsv")
            == read_table_data(disk_prefix.string() + ".results.tsv"),
        "resident and disk subsample scores differ");
    require(read_text(resident_prefix.string() + ".model.tsv")
            == read_text(auto_prefix.string() + ".model.tsv"),
        "automatically promoted subsample model differs");
    require_uac_trace_top_probability(
        resident_prefix.string() + ".trace.tsv", "subsample_em");
    for (const auto& prefix : {resident_prefix, disk_prefix}) {
        const std::filesystem::path results =
            prefix.string() + ".diagnostics.tsv";
        const int32_t evaluations = std::stoi(metadata_value(
            results, "fit_subsample_evaluations"));
        const int64_t evaluated_documents = std::stoll(metadata_value(
            results, "fit_approximate_documents"));
        const int32_t updates = std::stoi(metadata_value(
            results, "fit_approximate_updates"));
        const int32_t topups = std::stoi(metadata_value(
            results, "fit_subsample_topup_rounds"));
        require(evaluations == updates + topups
                && evaluated_documents > 0 && evaluated_documents <= 48,
            "subsample diagnostics omitted an E-step from work accounting");
        require(std::stoull(metadata_value(results,
                    "fit_subsample_peak_bytes")) <= 1024 * 1024,
            "subsample working set exceeded its reported memory budget");
        require(!metadata_value(results,
                    "fit_subsample_peak_phase").empty(),
            "subsample peak phase was not reported");
    }
    const std::filesystem::path auto_diagnostics =
        auto_prefix.string() + ".diagnostics.tsv";
    require(metadata_value(auto_diagnostics, "fit_subsample_storage") == "disk"
            && std::stoi(metadata_value(auto_diagnostics,
                    "fit_subsample_storage_promotions")) == 1
            && std::stoull(metadata_value(auto_diagnostics,
                    "fit_subsample_peak_bytes")) <= 5 * 1024,
        "automatic storage did not promote at the realized memory boundary");
    std::vector<std::string> rejected_arguments =
        subsample_arguments(base.string() + ".subsample_rejected", "resident");
    const auto rejected_budget = std::find(rejected_arguments.begin(),
        rejected_arguments.end(), "--fit-subsample-memory-budget");
    *(rejected_budget + 1) = "5K";
    const auto rejected = run_command_capture_error(
        cmdUacFit, std::move(rejected_arguments));
    require(rejected.first == 1
            && rejected.second.find("memory budget") != std::string::npos,
        "forced resident subsample did not fail closed at its memory boundary");

    std::ifstream state_input(fit_prefix.string() + ".state.tsv");
    std::string state_header;
    require(static_cast<bool>(std::getline(state_input, state_header))
            && state_header == "##punkst_uac_state_v15",
        "direct topic-to-UAC fit did not write a v15 state");
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
    require(std::abs(fitted_state.center_floor - 5e-5) < 1e-15,
        "UAC state did not preserve the requested center floor");
    require(fitted_state.fisher_refinement_iterations == 2,
        "UAC state did not preserve Fisher refinement iterations");

    std::ifstream state_text_input(fit_prefix.string() + ".state.tsv");
    std::ostringstream state_text_buffer;
    state_text_buffer << state_text_input.rdbuf();
    std::string legacy_state_text = state_text_buffer.str();
    legacy_state_text.replace(0, std::string("##punkst_uac_state_v15").size(),
        "##punkst_uac_state_v11");
    const auto erase_metadata_prefix = [&](const std::string& prefix) {
        const size_t position = legacy_state_text.find(prefix);
        require(position != std::string::npos,
            "UAC v15 ANN metadata record is missing");
        const size_t end = legacy_state_text.find('\n', position);
        require(end != std::string::npos,
            "UAC v15 ANN metadata record is malformed");
        legacy_state_text.erase(position, end - position + 1);
    };
    for (const std::string& prefix : std::vector<std::string>{
            "##leiden_hnsw_m\t", "##leiden_hnsw_ef_construction\t",
            "##leiden_hnsw_ef_search\t", "##leiden_hnsw_max_ef_search\t",
            "##leiden_hnsw_candidates\t", "##leiden_hnsw_audit_queries\t",
            "##leiden_hnsw_recall\t", "##leiden_hnsw_force\t",
            "##leiden_nndescent_iterations\t",
            "##leiden_nndescent_graph_size\t", "##leiden_nndescent_s\t",
            "##leiden_nndescent_audit_queries\t",
            "##leiden_nndescent_recall\t",
            "##leiden_resolved_ann_parameter\t",
            "##leiden_resolved_ann_candidates\t",
            "##leiden_ann_audit_mean_recall\t",
            "##leiden_ann_audit_recall_lcb\t",
            "##leiden_ann_audit_passed\t", "##leiden_ann_forced\t"}) {
        erase_metadata_prefix(prefix);
    }
    const std::string metric_record =
        "##initialization_metric\thellinger\n";
    const size_t metric_position = legacy_state_text.find(metric_record);
    require(metric_position != std::string::npos,
        "UAC state metric record is missing");
    legacy_state_text.erase(metric_position, metric_record.size());
    const std::string diagonal_mode_record =
        "##factor_diagonal_mode\tcomponent\n";
    const size_t diagonal_mode_position =
        legacy_state_text.find(diagonal_mode_record);
    require(diagonal_mode_position != std::string::npos,
        "UAC state factor diagonal mode record is missing");
    legacy_state_text.erase(
        diagonal_mode_position, diagonal_mode_record.size());
    const std::string refinement_record =
        "##fisher_refinement_iterations\t2\n";
    const size_t refinement_position =
        legacy_state_text.find(refinement_record);
    require(refinement_position != std::string::npos,
        "UAC state Fisher refinement record is missing");
    legacy_state_text.erase(refinement_position, refinement_record.size());
    const std::filesystem::path legacy_state_path =
        fit_prefix.string() + ".legacy_v11.state.tsv";
    write_text(legacy_state_path, legacy_state_text);
    const uac::State legacy_state = uac::read_state(
        legacy_state_path.string());
    require(legacy_state.initialization_metric == SimplexMetric::Cosine,
        "v11 UAC state did not default its initialization metric to cosine");
    require(legacy_state.fisher_refinement_iterations == 1,
        "v11 UAC state did not default Fisher refinement to one step");
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
        "--diagnosis-per-unit",
        "--threads", "1",
        "--n-representatives", "1",
        "--visual-whitening", "sample",
        "--visual-full",
    };
    require(run_command(cmdUacTransform,
            std::move(transform_arguments)) == 0,
        "direct topic-to-UAC transform failed");

    require_uac_results_format(
        transform_prefix.string() + ".results.tsv", 2);
    require_scientific_fields(
        transform_prefix.string() + ".model.tsv", 2);
    require_scientific_fields(
        transform_prefix.string() + ".separation.tsv", 2);
    require_scientific_fields(
        transform_prefix.string() + ".visual.results.tsv", 1);

    const std::filesystem::path transform_diagnostics =
        transform_prefix.string() + ".diagnostics.tsv";
    require(table_header(transform_diagnostics)
            == "#id\traw_total\teffective_total\trelative_ess"
               "\tmaximum_weight\tlog_likelihood_range"
               "\tlog_proposal_range\thpd80_log_density_threshold"
               "\thpd95_log_density_threshold",
        "UAC per-unit diagnostics did not omit inapplicable columns");
    require(read_text(transform_diagnostics).find(
                "doc_0\t10.00\t10.00\t") != std::string::npos,
        "UAC count-like diagnostics are not written with two decimals");

    require(data_rows(transform_prefix.string() + ".results.tsv") == 12,
        "direct topic-to-UAC transform wrote the wrong row count");
    require(data_rows(fit_prefix.string() + ".visual.axes.tsv") == 3,
        "UAC fit wrote the wrong one-dimensional axis row count");
    require(data_rows(fit_prefix.string() + ".visual.model.tsv") == 2,
        "UAC fit wrote the wrong projected-model row count");
    require(data_rows(fit_prefix.string() + ".visual.results.tsv") == 12,
        "UAC fit wrote the wrong projected-document row count");
    require(data_rows(transform_prefix.string() + ".visual.results.tsv")
            == 12,
        "UAC transform wrote the wrong projected-document row count");
    require(read_text(fit_prefix.string() + ".visual.results.tsv").find(
                "#id\tmean_1\n") == 0,
        "UAC fit did not default to mean-only visualization");
    require(read_text(transform_prefix.string() + ".visual.results.tsv").find(
                "#id\tmean_1\tfull_1\n") == 0,
        "UAC transform did not include the requested full visualization");
    require(first_data_field(
            fit_prefix.string() + ".visual.axes.tsv") == "mixture",
        "UAC fit did not use default mixture visualization whitening");
    require(first_data_field(
            transform_prefix.string() + ".visual.axes.tsv") == "sample",
        "UAC transform did not use requested sample visualization whitening");

    const std::filesystem::path embed_theta =
        base.string() + ".embed.theta.tsv";
    const std::filesystem::path embed_partitions =
        base.string() + ".embed.partitions.tsv";
    const std::filesystem::path embed_prefix =
        base.string() + ".embed";
    std::ostringstream embed_theta_text, embed_partition_text;
    embed_theta_text << "#id\t0\t1\n";
    embed_partition_text << "#id\tprimary\tsecondary\n";
    for (int32_t document = 0; document < 12; ++document) {
        const bool first = document < 6;
        embed_theta_text << "doc_" << document << "\t"
            << (first ? 0.95 : 0.05) << "\t"
            << (first ? 0.05 : 0.95) << "\n";
        if (document < 10) {
            embed_partition_text << "doc_" << document << "\t"
                << (document < 5 ? "left" : "right") << "\t"
                << (document < 4 ? "early" : "late") << "\n";
        }
    }
    embed_partition_text << "extra_unit\tleft\tearly\n";
    write_text(embed_theta, embed_theta_text.str());
    write_text(embed_partitions, embed_partition_text.str());
    const auto embedded = run_command_capture_error(cmdLinearEmbed, {
        "linear-embed",
        "--in-theta", embed_theta.string(),
        "--in-partition", embed_partitions.string(),
        "--out-prefix", embed_prefix.string(),
        "--icol-partition", "1", "2",
        "--partition-labels", "primary", "secondary",
        "--dim", "2",
        "--whitening", "mixture",
        "--threads", "1",
    });
    require(embedded.first == 0
            && embedded.second.find(
                "theta units 12; partition units 11; intersection 10")
                != std::string::npos,
        "linear embedding did not report partial input matching");
    for (const std::string& label : {"primary", "secondary"}) {
        const std::string prefix = embed_prefix.string() + "." + label;
        for (const std::string& space : {"linear", "ilr"}) {
            require(data_rows(prefix + "." + space + ".transform.tsv") > 0,
                "linear embedding omitted a space transformation");
            require(data_rows(prefix + "." + space + ".mean.axes.tsv") == 2,
                "linear embedding wrote the wrong mean axis-weight shape");
            require(!std::filesystem::exists(
                        prefix + "." + space + ".full.axes.tsv"),
                "linear embedding wrote full axes without --visual-full");
        }
        require(data_rows(prefix + ".results.tsv") == 12,
            "linear embedding did not project every theta unit");
        require(read_text(prefix + ".results.tsv").find(
                "#id\tlinear_mean_1\tilr_mean_1\n") == 0,
            "linear embedding wrote the wrong coordinate header");
    }

    const std::filesystem::path row_partition =
        base.string() + ".embed.rows.tsv";
    const std::filesystem::path row_prefix =
        base.string() + ".embed_row";
    std::ostringstream row_partition_text;
    row_partition_text << "#row\tpartition\n";
    for (int32_t document = 0; document < 12; ++document) {
        row_partition_text << document << "\t"
            << (document < 6 ? 0 : 1) << "\n";
    }
    write_text(row_partition, row_partition_text.str());
    require(run_command(cmdLinearEmbed, {
        "linear-embed",
        "--in-theta", embed_theta.string(),
        "--in-partition", row_partition.string(),
        "--out-prefix", row_prefix.string(),
        "--id-as-row-index",
        "--visual-dim", "1",
        "--whitening", "sample",
        "--visual-full",
        "--threads", "1",
    }) == 0,
        "row-index linear embedding failed");
    require(data_rows(row_prefix.string() + ".results.tsv") == 12,
        "row-index linear embedding wrote the wrong coordinate row count");
    require(data_rows(row_prefix.string() + ".linear.full.axes.tsv") == 2
            && data_rows(row_prefix.string() + ".ilr.full.axes.tsv") == 2
            && read_text(row_prefix.string() + ".results.tsv").find(
                "#id\tlinear_mean_1\tlinear_full_1"
                "\tilr_mean_1\tilr_full_1\n") == 0,
        "row-index linear embedding omitted requested full visualization");

    remove_uac_outputs(fit_prefix);
    remove_uac_outputs(legacy_init_prefix);
    remove_uac_outputs(full_init_prefix);
    remove_uac_outputs(ht_init_prefix);
    remove_uac_outputs(ht_repeat_prefix);
    remove_uac_outputs(resident_prefix);
    remove_uac_outputs(disk_prefix);
    remove_uac_outputs(auto_prefix);
    std::filesystem::remove(resident_prefix.string() + ".subsample.tsv");
    std::filesystem::remove(disk_prefix.string() + ".subsample.tsv");
    std::filesystem::remove(auto_prefix.string() + ".subsample.tsv");
    std::filesystem::remove_all(stream_cache);
    remove_uac_outputs(transform_prefix);
    std::filesystem::remove(model_path);
    std::filesystem::remove(center_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(metadata_path);
    std::filesystem::remove(lda_prefix.string() + ".results.tsv");
    std::filesystem::remove(lda_prefix.string() + ".pseudobulk.tsv");
    std::filesystem::remove(embed_theta);
    std::filesystem::remove(embed_partitions);
    std::filesystem::remove(row_partition);
    for (const std::string& prefix : {
            embed_prefix.string() + ".primary",
            embed_prefix.string() + ".secondary",
            row_prefix.string()}) {
        std::filesystem::remove(prefix + ".results.tsv");
        for (const std::string& space : {"linear", "ilr"}) {
            for (const std::string& suffix : {
                    ".transform.tsv", ".mean.axes.tsv", ".full.axes.tsv"}) {
                std::filesystem::remove(prefix + "." + space + suffix);
            }
        }
    }
}

void test_gamma_poisson_model_initialization() {
    GammaPoissonTopicModel model(2, 3, 17, 1, 0,
        0.5, 0.3, -1.0, 1.0, 1.0, -1.0,
        0.7, 10.0, 100, 30.0);
    RowMajorMatrixXd profiles(2, 3);
    profiles << 9.0, 1.0, 0.0,
                1.0, 3.0, 6.0;
    model.initialize_topic_profiles(profiles, {"alpha", "beta"});
    const RowMajorMatrixXd initialized = model.copy_model();
    require((initialized.rowwise().sum().array() - 1.0).abs().maxCoeff()
            < 1e-12,
        "Gamma-Poisson model initialization did not normalize topics");
    require(initialized(0, 0) > 0.899999
            && initialized(0, 2) > 0.0
            && initialized(0, 2) < 1e-10,
        "Gamma-Poisson model initialization did not preserve the supplied profile");
    require(model.get_topic_names()
            == std::vector<std::string>({"alpha", "beta"}),
        "Gamma-Poisson model initialization did not preserve topic names");
    bool rejected = false;
    try {
        RowMajorMatrixXd wrong(1, 3);
        wrong.setOnes();
        model.initialize_topic_profiles(wrong);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected,
        "Gamma-Poisson model initialization accepted mismatched dimensions");
}

void test_leiden_projection_outputs() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path()
        / "punkst_leiden_projection";
    const std::filesystem::path theta_path =
        base.string() + ".theta.tsv";
    std::ostringstream theta;
    theta << "#id\t0\t1\t2\t3\n";
    for (int32_t group = 0; group < 3; ++group) {
        for (int32_t offset = 0; offset < 6; ++offset) {
            theta << "unit_" << group << "_" << offset;
            for (int32_t factor = 0; factor < 4; ++factor) {
                const double value = factor == group ? 1.0
                    : factor == (group + 1) % 4
                    ? 0.01 * (offset + 1) : 0.0;
                theta << '\t' << value;
            }
            theta << '\n';
        }
    }
    write_text(theta_path, theta.str());

    const std::filesystem::path default_prefix =
        base.string() + ".default";
    require(run_command(cmdLeiden, {
        "leiden", "--in-theta", theta_path.string(),
        "--out-prefix", default_prefix.string(), "--neighbors", "4",
        "--resolution", "0.5", "1", "--threads", "1", "--seed", "42",
    }) == 0, "Leiden default projection run failed");
    const std::string axes = read_text(
        default_prefix.string() + ".projection.axes.tsv");
    const std::string coordinates = read_text(
        default_prefix.string() + ".projection.results.tsv");
    require(axes.find("\nlinear\t") != std::string::npos
            && axes.find("\nilr\t") != std::string::npos
            && coordinates.find("\tlinear_r0.5_1") != std::string::npos
            && coordinates.find("\tilr_r0.5_1") != std::string::npos
            && coordinates.find("\tlinear_r1_1")
                != std::string::npos,
        "Leiden default projections omitted a space or resolution");

    const std::filesystem::path linear_prefix =
        base.string() + ".linear";
    require(run_command(cmdLeiden, {
        "leiden", "--in-theta", theta_path.string(),
        "--out-prefix", linear_prefix.string(), "--neighbors", "4",
        "--resolution", "1", "--projection-space", "linear",
        "--threads", "1", "--seed", "42",
    }) == 0, "Leiden linear-only projection run failed");
    const std::string linear_axes = read_text(
        linear_prefix.string() + ".projection.axes.tsv");
    require(linear_axes.find("\nlinear\t") != std::string::npos
            && linear_axes.find("\nilr\t") == std::string::npos,
        "Leiden linear-only projection included ILR axes");

    const std::filesystem::path disabled_prefix =
        base.string() + ".disabled";
    require(run_command(cmdLeiden, {
        "leiden", "--in-theta", theta_path.string(),
        "--out-prefix", disabled_prefix.string(), "--neighbors", "4",
        "--resolution", "1", "--no-projection", "--threads", "1",
        "--seed", "42",
    }) == 0, "Leiden projection opt-out run failed");
    require(!std::filesystem::exists(
                disabled_prefix.string() + ".projection.axes.tsv")
            && !std::filesystem::exists(
                disabled_prefix.string() + ".projection.results.tsv"),
        "Leiden projection opt-out wrote projection files");

    for (const std::filesystem::path& prefix : {
            default_prefix, linear_prefix, disabled_prefix}) {
        for (const std::string& suffix : {
                ".clusters.tsv", ".diagnostics.tsv",
                ".projection.axes.tsv", ".projection.results.tsv"}) {
            std::filesystem::remove(prefix.string() + suffix);
        }
    }
    std::filesystem::remove(theta_path);
}

} // namespace

int32_t test(int32_t, char**) {
    try {
        test_simplex_metrics();
        test_visualization_projection();
        test_factor_covariance_change_math();
        test_topic_to_uac_handoff();
        test_leiden_projection_outputs();
        test_gamma_poisson_model_initialization();
        std::cout << "UAC tests passed\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
        return 1;
    }
    return 0;
}
