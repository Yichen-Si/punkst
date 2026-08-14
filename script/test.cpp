#include "clustering_core/cosine_clustering.hpp"
#include "clustering_core/projection.hpp"
#include "gamma_pois_topic.hpp"
#include "numerical_utils.hpp"
#include "partition_classifier.hpp"
#include "partition_classifier_lrvb.hpp"
#include "lda_state.hpp"

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
int32_t cmdPartitionClassifierFit(int argc, char** argv);
int32_t cmdPartitionClassifierPredict(int argc, char** argv);
int32_t cmdLDATransform(int argc, char** argv);
int32_t cmdGammaPoisTransform(int argc, char** argv);

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("Test failed: " + message);
    }
}

void test_quartimax_rotation() {
    punkst::projection::VisualizationProjection projection;
    projection.eigenvalues.resize(2);
    projection.eigenvalues << 3.0, 1.0;
    projection.projection = Eigen::MatrixXd::Identity(3, 2);
    projection.topic_contrasts.resize(4, 2);
    projection.topic_contrasts <<
        0.7, 0.7,
        0.7, -0.7,
        0.1, 0.1,
        0.1, -0.1;
    projection.component_means.resize(1, 2);
    projection.component_means << 0.5, -0.25;
    projection.component_covariances.push_back(
        Eigen::MatrixXd::Identity(2, 2));
    const Eigen::MatrixXd original_projector = projection.projection
        * projection.projection.transpose();
    const double original_objective =
        punkst::projection::quartimax_objective(
            projection.topic_contrasts);

    punkst::projection::quartimax_rotate(projection);

    require(projection.axis_scores.size() == 2
            && projection.axis_scores(0) >= projection.axis_scores(1),
        "quartimax rotation did not order separation scores");
    require(std::abs(projection.axis_scores.sum() - 4.0) < 1e-12,
        "quartimax rotation did not preserve total separation");
    require((projection.projection * projection.projection.transpose())
            .isApprox(original_projector, 1e-12),
        "quartimax rotation changed the embedding subspace");
    require(punkst::projection::quartimax_objective(
            projection.topic_contrasts) + 1e-12 >= original_objective,
        "quartimax rotation decreased sparsity objective");
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
        partitions << document << "\t"
            << (first ? "first" : "second") << "\n";
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
        "--projection-space", "both",
        "--skip-qda-projection",
        "--visual-full",
        "--threads", "1",
    }) == 0, "linear embedding failed");
    require(data_rows(output_prefix.string() + ".results.tsv") == 12,
        "linear embedding wrote the wrong coordinate row count");
    require(read_text(output_prefix.string() + ".cluster_labels.tsv")
            == "#cluster_index\tcluster_name\n0\tfirst\n1\tsecond\n",
        "linear embedding omitted the original cluster-label mapping");
    require(data_rows(output_prefix.string() + ".cluster_factors.tsv") == 2,
        "linear embedding omitted the cluster factor sums");
    require(data_rows(output_prefix.string() + ".linear.full.axes.tsv") == 2
            && data_rows(output_prefix.string() + ".ilr.full.axes.tsv") == 2
            && read_text(output_prefix.string() + ".results.tsv").find(
                "#id\tlinear_mean_1\tlinear_full_1"
                "\tilr_mean_1\tilr_full_1\n") == 0,
        "linear embedding omitted requested projections");

    std::filesystem::remove(theta_path);
    std::filesystem::remove(partition_path);
    std::filesystem::remove(output_prefix.string() + ".results.tsv");
    std::filesystem::remove(
        output_prefix.string() + ".cluster_labels.tsv");
    std::filesystem::remove(
        output_prefix.string() + ".cluster_factors.tsv");
    for (const std::string& space : {"linear", "ilr"}) {
        for (const std::string& suffix : {
                ".transform.tsv", ".mean.axes.tsv", ".full.axes.tsv"}) {
            std::filesystem::remove(
                output_prefix.string() + "." + space + suffix);
        }
    }
}

void test_partition_classifier_outputs() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path()
        / "punkst_partition_classifier";
    const std::filesystem::path theta_path = base.string() + ".theta.tsv";
    const std::filesystem::path partition_path =
        base.string() + ".partition.tsv";
    const std::filesystem::path output_prefix = base.string() + ".fit";
    const std::filesystem::path predict_prefix = base.string() + ".predict";

    std::ostringstream theta, partition;
    theta << "#id\t0\t1\t2\n";
    partition << "#id\tcluster\n";
    const std::vector<std::string> classes{
        "first class", "second-class", "third.class"};
    for (int32_t component = 0; component < 3; ++component) {
        for (int32_t offset = 0; offset < 12; ++offset) {
            const std::string id = "unit_" + std::to_string(component)
                + "_" + std::to_string(offset);
            theta << id;
            for (int32_t topic = 0; topic < 3; ++topic) {
                const double value = topic == component ? 0.94
                    : 0.03 + 0.001 * ((offset + topic) % 2);
                theta << '\t' << value;
            }
            theta << '\n';
            partition << id << '\t' << classes[component] << '\n';
        }
    }
    write_text(theta_path, theta.str());
    write_text(partition_path, partition.str());

    require(run_command(cmdPartitionClassifierFit, {
        "partition-classifier-fit",
        "--in-theta", theta_path.string(),
        "--in-partition", partition_path.string(),
        "--out-prefix", output_prefix.string(),
        "--folds", "3", "--ridge-grid", "1e-6", "1e-2",
        "--min-per-class", "2", "--train-max-rows", "24",
        "--max-iterations", "150",
    }) == 0, "partition classifier fit failed");
    const std::string compact = read_text(
        output_prefix.string() + ".results.tsv");
    require(compact.find("#id\tC1\tP1\tC2\tP2\tC3\tP3\n") == 0,
        "partition classifier compact columns have the wrong order");
    require(data_rows(output_prefix.string() + ".results.tsv") == 36
            && data_rows(output_prefix.string() + ".cv.tsv") == 2,
        "partition classifier wrote the wrong result or CV row count");
    const auto model = punkst::partition_classifier::Model::read(
        output_prefix.string() + ".classifier.tsv");
    require(model.classes == classes && model.topics
            == std::vector<std::string>({"0", "1", "2"}),
        "partition classifier did not preserve class/topic order");
    model.validate();
    const std::filesystem::path roundtrip_path =
        base.string() + ".roundtrip.classifier.tsv";
    model.write(roundtrip_path);
    const auto roundtrip = punkst::partition_classifier::Model::read(
        roundtrip_path);
    Eigen::VectorXd roundtrip_composition(3);
    roundtrip_composition << 0.2, 0.3, 0.5;
    require(model.intercepts.isApprox(roundtrip.intercepts, 1e-15)
            && model.coefficients.isApprox(roundtrip.coefficients, 1e-15)
            && model.probabilities(roundtrip_composition).isApprox(
                roundtrip.probabilities(roundtrip_composition), 1e-15),
        "partition classifier model did not survive a precise round trip");

    require(run_command(cmdPartitionClassifierPredict, {
        "partition-classifier-predict",
        "--in-theta", theta_path.string(),
        "--in-model", output_prefix.string() + ".classifier.tsv",
        "--out-prefix", predict_prefix.string(),
        "--dense-probabilities",
    }) == 0, "partition classifier prediction failed");
    const std::string dense = read_text(
        predict_prefix.string() + ".results.tsv");
    require(dense.find("#id\tP0\tP1\tP2\n") == 0
            && data_rows(predict_prefix.string() + ".results.tsv") == 36,
        "partition classifier dense output is malformed");
    {
        std::istringstream rows(dense);
        std::string line;
        std::getline(rows, line);
        while (std::getline(rows, line)) {
            const std::vector<std::string> fields = split_delimited(line, '\t');
            require(fields.size() == 4,
                "partition classifier dense row has the wrong width");
            const double total = std::stod(fields[1]) + std::stod(fields[2])
                + std::stod(fields[3]);
            require(std::abs(total - 1.0) < 1e-9,
                "partition classifier probabilities do not sum to one");
        }
    }

    std::filesystem::remove(theta_path);
    std::filesystem::remove(partition_path);
    for (const std::string& suffix : {
            ".classifier.tsv", ".results.tsv", ".cv.tsv",
            ".calibration.tsv"}) {
        std::filesystem::remove(output_prefix.string() + suffix);
    }
    std::filesystem::remove(predict_prefix.string() + ".results.tsv");
    std::filesystem::remove(roundtrip_path);
}

void test_partition_classifier_lda_lrvb() {
    punkst::partition_classifier::Model classifier;
    classifier.topics = {"0", "1", "2"};
    classifier.classes = {"A", "B", "C"};
    classifier.intercepts = Eigen::VectorXd::Zero(3);
    classifier.coefficients.resize(3, 3);
    classifier.coefficients <<
        2.0, -1.0, -1.0,
        -1.0, 2.0, -1.0,
        -1.0, -1.0, 2.0;
    classifier.temperature = 1.1;
    classifier.ridge = 1e-3;
    classifier.validate();

    RowMajorMatrixXd components(3, 3);
    components <<
        20.0, 2.0, 1.0,
        2.0, 20.0, 1.0,
        1.0, 1.0, 20.0;
    Document document;
    document.ids = {0, 1};
    document.cnts = {12.0, 10.0};
    Eigen::VectorXd assigned(3);
    assigned << 11.5, 9.5, 1.0;
    punkst::partition_classifier::PropagationOptions options;
    options.lrvb_all = true;
    const Eigen::MatrixXd allocation_kernel =
        dirichlet_expectation_2d(components);
    const auto prediction = punkst::partition_classifier::propagate_lda(
        classifier, assigned, document, allocation_kernel, 0.5, options);
    require(prediction.lrvb_attempted && prediction.lrvb_status == "ok"
            && prediction.method == "quadrature"
            && prediction.fixed_point_iterations > 0
            && prediction.fixed_point_residual
                <= options.fixed_point_tolerance
            && std::abs(prediction.probabilities.sum() - 1.0) < 1e-10
            && prediction.probabilities.minCoeff() >= 0.0,
        "LDA classifier propagation failed: " + prediction.lrvb_status);

    options.fixed_point_max_iterations = 1;
    options.fixed_point_tolerance = 1e-15;
    const auto nonconverged = punkst::partition_classifier::propagate_lda(
        classifier, assigned, document, allocation_kernel, 0.5, options);
    const Eigen::VectorXd initial = classifier.probabilities(
        assigned.array() + 0.5);
    require(nonconverged.lrvb_status == "local_nonconvergence"
            && nonconverged.fixed_point_iterations == 1
            && (nonconverged.probabilities - initial).cwiseAbs().maxCoeff()
                == 0.0,
        "LDA nonconvergence did not preserve the original plug-in prediction");
}

void test_partition_classifier_gamma_poisson_lrvb() {
    punkst::partition_classifier::Model classifier;
    classifier.topics = {"0", "1", "2"};
    classifier.classes = {"A", "B", "C"};
    classifier.intercepts = Eigen::VectorXd::Zero(3);
    classifier.coefficients.resize(3, 3);
    classifier.coefficients <<
        2.0, -1.0, -1.0,
        -1.0, 2.0, -1.0,
        -1.0, -1.0, 2.0;
    classifier.temperature = 1.0;
    classifier.ridge = 1e-3;

    GammaPoissonTopicModel model(3, 3, 29, 1, 0,
        0.5, 0.3, -1.0, 1.5, 1.0, -1.0,
        0.7, 10.0, 100, 30.0);
    RowMajorMatrixXd profiles(3, 3);
    profiles <<
        20.0, 2.0, 1.0,
        2.0, 20.0, 1.0,
        1.0, 1.0, 20.0;
    model.initialize_topic_profiles(profiles);
    model.prepare_inference_cache();
    Document document;
    document.ids = {0, 1};
    document.cnts = {12.0, 10.0};
    GammaPoissonDocumentPosterior posterior;
    model.infer_document_posterior(document, posterior);
    punkst::partition_classifier::PropagationOptions options;
    options.lrvb_all = true;
    const Eigen::VectorXd prior_rate = model.get_theta_prior_rate();
    const auto prediction =
        punkst::partition_classifier::propagate_gamma_poisson(
            classifier, posterior, document, model.get_topic_capacity(),
            model.get_beta_allocation_kernel(), model.get_expected_beta(),
            model.get_theta_prior_shape(), prior_rate, nullptr, options);
    require(prediction.lrvb_attempted && prediction.lrvb_status == "ok"
            && prediction.method == "quadrature"
            && prediction.fixed_point_iterations > 0
            && prediction.fixed_point_residual
                <= options.fixed_point_tolerance
            && std::abs(prediction.probabilities.sum() - 1.0) < 1e-10
            && prediction.probabilities.minCoeff() >= 0.0,
        "Gamma-Poisson classifier propagation failed: "
            + prediction.lrvb_status);

    punkst::partition_classifier::PropagationOptions nonconvergence_options;
    nonconvergence_options.lrvb_all = true;
    nonconvergence_options.fixed_point_max_iterations = 1;
    nonconvergence_options.fixed_point_tolerance = 1e-15;
    const auto nonconverged =
        punkst::partition_classifier::propagate_gamma_poisson(
            classifier, posterior, document, model.get_topic_capacity(),
            model.get_beta_allocation_kernel(), model.get_expected_beta(),
            model.get_theta_prior_shape(), prior_rate, nullptr,
            nonconvergence_options);
    const Eigen::VectorXd initial_abundance = model.get_topic_capacity().array()
        * posterior.shape.array() / posterior.rate.array();
    const Eigen::VectorXd initial_probability =
        classifier.probabilities(initial_abundance);
    require(nonconverged.lrvb_status == "local_nonconvergence"
            && (nonconverged.probabilities - initial_probability)
                .cwiseAbs().maxCoeff() == 0.0,
        "Gamma-Poisson nonconvergence did not preserve the original plug-in prediction");

    punkst::partition_classifier::PropagationOptions refined_options;
    refined_options.lrvb_all = true;
    refined_options.cg_tolerance = 1e9;
    const auto refined_plugin =
        punkst::partition_classifier::propagate_gamma_poisson(
            classifier, posterior, document, model.get_topic_capacity(),
            model.get_beta_allocation_kernel(), model.get_expected_beta(),
            model.get_theta_prior_shape(), prior_rate, nullptr,
            refined_options);
    punkst::partition_classifier::PropagationOptions curvature_failure_options =
        refined_options;
    curvature_failure_options.cg_tolerance = -1.0;
    const auto curvature_failure =
        punkst::partition_classifier::propagate_gamma_poisson(
            classifier, posterior, document, model.get_topic_capacity(),
            model.get_beta_allocation_kernel(), model.get_expected_beta(),
            model.get_theta_prior_shape(), prior_rate, nullptr,
            curvature_failure_options);
    require(refined_plugin.lrvb_status == "ok"
            && curvature_failure.lrvb_status == "curvature_solve_failed"
            && curvature_failure.probabilities.isApprox(
                refined_plugin.probabilities, 1e-12),
        "post-convergence failure did not preserve the refined plug-in prediction");

    model.set_feature_dispersion({10.0, 15.0, 20.0});
    model.infer_document_posterior(document, posterior);
    const auto dispersion_prediction =
        punkst::partition_classifier::propagate_gamma_poisson(
            classifier, posterior, document, model.get_topic_capacity(),
            model.get_beta_allocation_kernel(), model.get_expected_beta(),
            model.get_theta_prior_shape(), prior_rate,
            &model.get_feature_dispersion(), options);
    require(dispersion_prediction.lrvb_status == "ok"
            && std::abs(dispersion_prediction.probabilities.sum() - 1.0)
                < 1e-10,
        "Dispersed Gamma-Poisson classifier propagation failed: "
            + dispersion_prediction.lrvb_status);

    Document deep_document = document;
    deep_document.cnts = {12000.0, 10000.0};
    model.infer_document_posterior(deep_document, posterior);
    punkst::partition_classifier::PropagationOptions deep_options;
    deep_options.lrvb_all = true;
    const auto deep_prediction =
        punkst::partition_classifier::propagate_gamma_poisson(
            classifier, posterior, deep_document, model.get_topic_capacity(),
            model.get_beta_allocation_kernel(), model.get_expected_beta(),
            model.get_theta_prior_shape(), prior_rate,
            &model.get_feature_dispersion(), deep_options);
    punkst::partition_classifier::PropagationOptions reference_options =
        deep_options;
    reference_options.fixed_point_tolerance = 1e-8;
    reference_options.fixed_point_max_iterations = 20000;
    const auto reference_prediction =
        punkst::partition_classifier::propagate_gamma_poisson(
            classifier, posterior, deep_document, model.get_topic_capacity(),
            model.get_beta_allocation_kernel(), model.get_expected_beta(),
            model.get_theta_prior_shape(), prior_rate,
            &model.get_feature_dispersion(), reference_options);
    require(deep_prediction.lrvb_status == "ok"
            && reference_prediction.lrvb_status == "ok"
            && (deep_prediction.probabilities
                    - reference_prediction.probabilities)
                .cwiseAbs().maxCoeff() < 1e-4,
        "high-depth Gamma-Poisson propagation disagrees with a tighter reference");
}

void test_classifier_transform_wiring() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path()
        / "punkst_classifier_transform";
    const std::filesystem::path input_path = base.string() + ".units.tsv";
    const std::filesystem::path metadata_path = base.string() + ".meta.json";
    const std::filesystem::path classifier_path =
        base.string() + ".classifier.tsv";
    const std::filesystem::path lda_state_path = base.string() + ".lda.state.tsv";
    const std::filesystem::path lda_prefix = base.string() + ".lda";
    const std::filesystem::path lda_parallel_prefix =
        base.string() + ".lda.parallel";
    const std::filesystem::path lda_plain_prefix =
        base.string() + ".lda.plain";
    const std::filesystem::path gp_state_path = base.string() + ".gp.state.tsv";
    const std::filesystem::path gp_prefix = base.string() + ".gp";
    const std::filesystem::path gp_plain_prefix = base.string() + ".gp.plain";
    write_text(metadata_path,
        "{\"n_units\":2,\"n_modalities\":1,\"n_features\":3,"
        "\"offset_data\":1,\"header_info\":[\"document\"],"
        "\"dictionary\":{\"feature_0\":0,\"feature_1\":1,"
        "\"feature_2\":2}}");
    write_text(input_path,
        "doc_0\t3\t20\t0 12\t1 7\t2 1\n"
        "doc_1\t3\t20\t0 2\t1 9\t2 9\n");

    punkst::partition_classifier::Model classifier;
    classifier.topics = {"0", "1", "2"};
    classifier.classes = {"A", "B", "C"};
    classifier.intercepts = Eigen::VectorXd::Zero(3);
    classifier.coefficients.resize(3, 3);
    classifier.coefficients <<
        2.0, -1.0, -1.0,
        -1.0, 2.0, -1.0,
        -1.0, -1.0, 2.0;
    classifier.ridge = 1e-3;
    classifier.temperature = 1.0;
    classifier.write(classifier_path);

    LdaState lda_state;
    lda_state.alpha = 0.5;
    lda_state.eta = 0.5;
    lda_state.topics = classifier.topics;
    lda_state.features = {"feature_0", "feature_1", "feature_2"};
    lda_state.components.resize(3, 3);
    lda_state.components <<
        20.0, 2.0, 1.0,
        2.0, 20.0, 1.0,
        1.0, 1.0, 20.0;
    lda_state.write(lda_state_path);
    require(run_command(cmdLDATransform, {
        "lda-transform", "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(), "--in-state",
        lda_state_path.string(), "--out-prefix", lda_prefix.string(),
        "--classifier-model", classifier_path.string(),
        "--classifier-lrvb-all", "--min-count", "1",
        "--minibatch-size", "2", "--threads", "1", "--seed", "31",
    }) == 0, "state-backed LDA classifier transform failed");
    require(read_text(lda_prefix.string() + ".classifications.tsv").find(
            "#document\tprediction\tmaximum_probability\tentropy") == 0
            && data_rows(lda_prefix.string() + ".classifications.tsv") == 2,
        "LDA classification output metadata/header is malformed");
    require(read_text(lda_prefix.string() + ".classifications.tsv").find(
            "\tfixed_point_iterations\tfixed_point_residual"
            "\tcg_iterations\tcurvature_jitter\n") != std::string::npos
            && read_text(lda_prefix.string()
                + ".classification_diagnostics.tsv").find(
                    "\tlocal_nonconvergence\tcurvature_failures")
                != std::string::npos,
        "LDA classifier diagnostics are incomplete");
    require(run_command(cmdLDATransform, {
        "lda-transform", "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(), "--in-state",
        lda_state_path.string(), "--out-prefix", lda_parallel_prefix.string(),
        "--classifier-model", classifier_path.string(),
        "--classifier-lrvb-all", "--min-count", "1",
        "--minibatch-size", "2", "--threads", "4", "--seed", "31",
    }) == 0, "parallel LDA classifier transform failed");
    require(read_text(lda_prefix.string() + ".classifications.tsv")
            == read_text(lda_parallel_prefix.string() + ".classifications.tsv"),
        "LDA classifier transform depends on thread count or batch parallelism");
    require(run_command(cmdLDATransform, {
        "lda-transform", "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(), "--in-state",
        lda_state_path.string(), "--out-prefix", lda_plain_prefix.string(),
        "--min-count", "1", "--minibatch-size", "2",
        "--threads", "1", "--seed", "31",
    }) == 0, "classifier-free LDA transform failed");
    require(read_text(lda_prefix.string() + ".results.tsv")
            == read_text(lda_plain_prefix.string() + ".results.tsv"),
        "enabling the classifier changed LDA topic output");

    GammaPoissonTopicModel gp_model(3, 3, 37, 1, 0,
        0.5, 0.3, -1.0, 1.5, 1.0, -1.0,
        0.7, 10.0, 2, 30.0);
    gp_model.initialize_topic_profiles(lda_state.components);
    gp_model.write_state(gp_state_path,
        {"feature_0", "feature_1", "feature_2"});
    require(run_command(cmdGammaPoisTransform, {
        "gamma-pois-transform", "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(), "--in-state",
        gp_state_path.string(), "--out-prefix", gp_prefix.string(),
        "--classifier-model", classifier_path.string(),
        "--classifier-lrvb-all", "--use-stored-dispersion",
        "--min-count", "1", "--minibatch-size", "1",
        "--threads", "1", "--seed", "41",
    }) == 0, "Gamma-Poisson classifier transform failed");
    require(read_text(gp_prefix.string() + ".classifications.tsv").find(
            "#document\tprediction\tmaximum_probability\tentropy") == 0
            && data_rows(gp_prefix.string() + ".classifications.tsv") == 2,
        "Gamma-Poisson classification output metadata/header is malformed");
    require(run_command(cmdGammaPoisTransform, {
        "gamma-pois-transform", "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(), "--in-state",
        gp_state_path.string(), "--out-prefix", gp_plain_prefix.string(),
        "--use-stored-dispersion", "--min-count", "1",
        "--minibatch-size", "1", "--threads", "1", "--seed", "41",
    }) == 0, "classifier-free Gamma-Poisson transform failed");
    require(read_text(gp_prefix.string() + ".results.tsv")
            == read_text(gp_plain_prefix.string() + ".results.tsv"),
        "enabling the classifier changed Gamma-Poisson topic output");

    for (const std::filesystem::path& path : {
            input_path, metadata_path, classifier_path, lda_state_path,
            gp_state_path}) std::filesystem::remove(path);
    for (const std::filesystem::path& prefix : {
            lda_prefix, lda_parallel_prefix, lda_plain_prefix,
            gp_prefix, gp_plain_prefix}) {
        for (const std::string& suffix : {
                ".results.tsv", ".classifications.tsv", ".pseudobulk.tsv",
                ".classification_diagnostics.tsv"}) {
            std::filesystem::remove(prefix.string() + suffix);
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
        "--projection-space", "both",
    }) == 0, "Leiden default projection run failed");
    const std::string first_coordinates = read_text(
        default_prefix.string() + ".projection.r0.5.results.tsv");
    const std::string second_coordinates = read_text(
        default_prefix.string() + ".projection.r1.results.tsv");
    require(std::filesystem::exists(default_prefix.string()
                + ".projection.r0.5.linear.transform.tsv")
            && std::filesystem::exists(default_prefix.string()
                + ".projection.r0.5.ilr.transform.tsv")
            && std::filesystem::exists(default_prefix.string()
                + ".r0.5.cluster_factors.tsv")
            && std::filesystem::exists(default_prefix.string()
                + ".r1.cluster_factors.tsv")
            && first_coordinates.find("\tlinear_mean_1") != std::string::npos
            && first_coordinates.find("\tilr_mean_1") != std::string::npos
            && second_coordinates.find("\tlinear_mean_1")
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
    require(std::filesystem::exists(linear_prefix.string()
                + ".projection.linear.transform.tsv")
            && std::filesystem::exists(linear_prefix.string()
                + ".cluster_factors.tsv")
            && !std::filesystem::exists(linear_prefix.string()
                + ".projection.ilr.transform.tsv"),
        "Leiden linear-only projection included ILR axes");

    const std::filesystem::path disabled_prefix =
        base.string() + ".disabled";
    require(run_command(cmdLeiden, {
        "leiden", "--in-theta", theta_path.string(),
        "--out-prefix", disabled_prefix.string(), "--neighbors", "4",
        "--resolution", "1", "--skip-projection", "--threads", "1",
        "--seed", "42",
    }) == 0, "Leiden projection opt-out run failed");
    require(!std::filesystem::exists(
                disabled_prefix.string() + ".projection.linear.transform.tsv")
            && !std::filesystem::exists(
                disabled_prefix.string() + ".projection.results.tsv")
            && std::filesystem::exists(disabled_prefix.string()
                + ".cluster_factors.tsv"),
        "Leiden projection opt-out wrote projection files");

    for (const std::filesystem::path& prefix : {
            default_prefix, linear_prefix, disabled_prefix}) {
        for (const std::string& suffix : {
                ".clusters.tsv", ".diagnostics.tsv", ".projection.results.tsv",
                ".projection.linear.transform.tsv",
                ".projection.linear.mean.axes.tsv", ".cluster_factors.tsv"}) {
            std::filesystem::remove(prefix.string() + suffix);
        }
    }
    for (const std::string& label : {"r0.5", "r1"}) {
        for (const std::string& suffix : {".results.tsv",
                ".linear.transform.tsv", ".linear.mean.axes.tsv",
                ".ilr.transform.tsv", ".ilr.mean.axes.tsv"}) {
            std::filesystem::remove(default_prefix.string()
                + ".projection." + label + suffix);
        }
        std::filesystem::remove(default_prefix.string()
            + "." + label + ".cluster_factors.tsv");
    }
    std::filesystem::remove(theta_path);
}

} // namespace

int32_t test(int32_t, char**) {
    try {
        punkst::partition_classifier::testing::run_classifier_gradient_test();
        punkst::partition_classifier::testing::run_lrvb_numerical_tests();
        test_quartimax_rotation();
        test_simplex_metrics();
        test_linear_embedding_outputs();
        test_partition_classifier_outputs();
        test_partition_classifier_lda_lrvb();
        test_partition_classifier_gamma_poisson_lrvb();
        test_classifier_transform_wiring();
        test_leiden_projection_outputs();
        test_gamma_poisson_model_initialization();
        std::cout << "Tests passed\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
        return 1;
    }
    return 0;
}
