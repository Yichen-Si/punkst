#include "clustering_core/cosine_clustering.hpp"
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

int32_t cmdLeiden(int argc, char** argv);
int32_t cmdLinearEmbed(int argc, char** argv);

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Test failed: " + message);
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

int32_t data_rows(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::string line;
    int32_t rows = -1;
    while (std::getline(input, line)) ++rows;
    return rows;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    require(static_cast<bool>(input), "cannot read " + path.string());
    std::ostringstream out;
    out << input.rdbuf();
    return out.str();
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

void test_linear_embedding_outputs() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path()
        / "punkst_linear_embed";
    const std::filesystem::path theta_path =
        base.string() + ".theta.tsv";
    const std::filesystem::path partition_path =
        base.string() + ".partitions.tsv";
    const std::filesystem::path output_prefix =
        base.string() + ".output";

    std::ostringstream theta, partitions;
    theta << "#id\t0\t1\n";
    partitions << "#row\tpartition\n";
    for (int32_t document = 0; document < 12; ++document) {
        const bool first = document < 6;
        theta << "doc_" << document << "\t"
            << (first ? 0.95 : 0.05) << "\t"
            << (first ? 0.05 : 0.95) << "\n";
        partitions << document << "\t" << (first ? 0 : 1) << "\n";
    }
    write_text(theta_path, theta.str());
    write_text(partition_path, partitions.str());

    require(run_command(cmdLinearEmbed, {
        "linear-embed",
        "--in-theta", theta_path.string(),
        "--in-partition", partition_path.string(),
        "--out-prefix", output_prefix.string(),
        "--id-as-row-index",
        "--visual-dim", "1",
        "--whitening", "sample",
        "--visual-full",
        "--threads", "1",
    }) == 0, "linear embedding failed");
    require(data_rows(output_prefix.string() + ".results.tsv") == 12,
        "linear embedding wrote the wrong coordinate row count");
    require(data_rows(output_prefix.string() + ".linear.full.axes.tsv") == 2
            && data_rows(output_prefix.string() + ".ilr.full.axes.tsv") == 2
            && read_text(output_prefix.string() + ".results.tsv").find(
                "#id\tlinear_mean_1\tlinear_full_1"
                "\tilr_mean_1\tilr_full_1\n") == 0,
        "linear embedding omitted requested projections");

    std::filesystem::remove(theta_path);
    std::filesystem::remove(partition_path);
    std::filesystem::remove(output_prefix.string() + ".results.tsv");
    for (const std::string& space : {"linear", "ilr"}) {
        for (const std::string& suffix : {
                ".transform.tsv", ".mean.axes.tsv", ".full.axes.tsv"}) {
            std::filesystem::remove(
                output_prefix.string() + "." + space + suffix);
        }
    }
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
        test_linear_embedding_outputs();
        test_leiden_projection_outputs();
        test_gamma_poisson_model_initialization();
        std::cout << "Tests passed\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
        return 1;
    }
    return 0;
}
