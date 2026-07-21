#include "clustering/uac.hpp"
#include "punkst.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("UAC test failed: " + message);
}

uac::Basis test_basis() {
    uac::Basis basis;
    basis.features = {"w0", "w1", "w2", "w3", "w4", "w5"};
    basis.topics = {"t0", "t1", "t2"};
    basis.probabilities.resize(6, 3);
    basis.probabilities <<
        8, 1, 1,
        6, 2, 1,
        1, 8, 1,
        1, 6, 2,
        1, 1, 8,
        2, 1, 6;
    uac::normalize_basis(basis);
    return basis;
}

uac::Dataset test_dataset() {
    constexpr int32_t documents = 54;
    constexpr int32_t topics = 3;
    uac::Dataset data;
    data.identifiers.resize(documents);
    data.centers.resize(documents, topics);
    data.counts.resize(documents);
    data.raw_totals.resize(documents);
    data.effective_totals.resize(documents);
    const std::array<Eigen::Vector3d, 3> centers{{
        Eigen::Vector3d(0.72, 0.20, 0.08),
        Eigen::Vector3d(0.13, 0.70, 0.17),
        Eigen::Vector3d(0.10, 0.22, 0.68),
    }};
    for (int32_t d = 0; d < documents; ++d) {
        const int32_t cluster = d % 3;
        data.identifiers[d] = "doc_" + std::to_string(d);
        Eigen::Vector3d value = centers[cluster];
        const double shift = 0.012 * ((d / 3) % 5 - 2);
        value((cluster + 1) % 3) += shift;
        value(cluster) -= shift;
        value = value.array().max(1e-4);
        value /= value.sum();
        data.centers.row(d) = value.transpose();
        Document document;
        const int32_t first = 2 * cluster;
        document.ids = {
            static_cast<uint32_t>(first),
            static_cast<uint32_t>(first + 1),
            static_cast<uint32_t>(2 * ((cluster + 1) % 3)),
        };
        document.cnts = {
            64.5 + (d % 4),
            29.25 + (d % 3),
            6.25,
        };
        const double total = std::accumulate(document.cnts.begin(),
            document.cnts.end(), 0.0);
        document.raw_ct_tot = total;
        document.ct_tot = total;
        data.raw_totals(d) = total;
        data.effective_totals(d) = total;
        data.counts[d] = std::move(document);
    }
    uac::normalize_centers(data.centers);
    const Eigen::MatrixXd helmert = uac::normalized_helmert(topics);
    data.coordinates = uac::ilr_transform(data.centers, helmert);
    return data;
}

struct RareComponentFixture {
    uac::Dataset data;
    uac::Basis basis;
};

RareComponentFixture rare_component_fixture() {
    constexpr int32_t documents = 28;
    constexpr int32_t topics = 12;
    RareComponentFixture out;
    out.basis.features.resize(topics);
    out.basis.topics.resize(topics);
    out.basis.probabilities = RowMajorMatrixXd::Constant(
        topics, topics, 0.1 / topics);
    for (int32_t topic = 0; topic < topics; ++topic) {
        out.basis.features[topic] = "feature_" + std::to_string(topic);
        out.basis.topics[topic] = "topic_" + std::to_string(topic);
        out.basis.probabilities(topic, topic) += 0.9;
    }
    uac::normalize_basis(out.basis);

    out.data.identifiers.resize(documents);
    out.data.centers.resize(documents, topics);
    out.data.counts.resize(documents);
    out.data.raw_totals = Eigen::VectorXd::Constant(documents, 200.0);
    out.data.effective_totals = out.data.raw_totals;
    RowMajorMatrixXd centers = RowMajorMatrixXd::Constant(3, topics,
        0.18 / (topics - 1));
    centers(0, 0) = 0.82;
    centers.row(1).setConstant(0.25 / (topics - 1));
    centers(1, 2) = 0.75;
    centers.row(2).setConstant(0.25 / (topics - 1));
    centers(2, 5) = 0.75;
    for (int32_t document = 0; document < documents; ++document) {
        const int32_t cluster = document < 2 ? 0 : (document < 15 ? 1 : 2);
        Eigen::VectorXd center = centers.row(cluster).transpose();
        const int32_t donor = (cluster * 2 + 1) % topics;
        const int32_t receiver = (donor + 1 + document) % topics;
        const double shift = 0.002 * ((document % 3) - 1);
        center(donor) -= shift;
        center(receiver) += shift;
        center = center.array().max(1e-8);
        center /= center.sum();
        out.data.centers.row(document) = center.transpose();
        out.data.identifiers[document] = "rare_doc_"
            + std::to_string(document);
        const Eigen::VectorXd probability =
            out.basis.probabilities * center;
        Document count;
        count.raw_ct_tot = count.ct_tot = 200.0;
        for (int32_t feature = 0; feature < topics; ++feature) {
            count.ids.push_back(feature);
            count.cnts.push_back(200.0 * probability(feature));
        }
        out.data.counts[document] = std::move(count);
    }
    out.data.coordinates = uac::ilr_transform(out.data.centers,
        uac::normalized_helmert(topics));
    return out;
}

std::string read_text(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read test output: "
        + path.string());
    return std::string(std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

struct ProfileSimulation {
    uac::Dataset data;
    uac::Basis basis;
    Eigen::VectorXi labels;
    RowMajorMatrixXd true_means;
    std::vector<Eigen::MatrixXd> true_covariances;
    RowMajorMatrixXd latent;
    Eigen::VectorXi document_lengths;
    Eigen::VectorXd low_topic_mass;
    Eigen::VectorXd map_ilr_error;
    Eigen::VectorXd oracle_fisher_trace;
    Eigen::VectorXd oracle_fisher_logdet;
    Eigen::VectorXd oracle_fisher_maximum_eigenvalue;
    Eigen::VectorXd oracle_fisher_condition;
    std::vector<bool> weak_topics;
    double mean_scale = 0.0;
    std::string length_profile = "fixed";
    std::string topic_identifiability = "homogeneous";
};

ProfileSimulation profile_simulation(int32_t documents, int32_t seed) {
    if (documents < 300) {
        throw std::invalid_argument("UAC profile requires at least 300 documents");
    }
    constexpr int32_t topics = 3, features = 60, components = 3;
    ProfileSimulation out;
    out.basis.features.resize(features);
    out.basis.topics = {"topic_0", "topic_1", "topic_2"};
    out.basis.probabilities.resize(features, topics);
    Eigen::VectorXd background(features);
    RowMajorMatrixXd anchor = RowMajorMatrixXd::Constant(features, topics,
        1e-4);
    for (int32_t feature = 0; feature < features; ++feature) {
        out.basis.features[feature] = "feature_" + std::to_string(feature);
        background(feature) = std::pow(feature + 1.0, -1.1);
        anchor(feature, feature % topics) += 1.0
            / (1.0 + feature / topics);
    }
    background /= background.sum();
    for (int32_t topic = 0; topic < topics; ++topic) {
        anchor.col(topic) /= anchor.col(topic).sum();
        out.basis.probabilities.col(topic) = 0.35 * background
            + 0.65 * anchor.col(topic);
    }
    uac::normalize_basis(out.basis);
    out.true_means.resize(components, topics - 1);
    out.true_means << 0.83259678, 0.0,
        -0.41629839, 0.72104997,
        -0.41629839, -0.72104997;
    out.true_covariances.resize(components);
    out.true_covariances[0].resize(2, 2);
    out.true_covariances[1].resize(2, 2);
    out.true_covariances[2].resize(2, 2);
    out.true_covariances[0] << 0.22556847, 0.10694488,
        0.10694488, 0.33443153;
    out.true_covariances[1] << 0.23019983, -0.00399334,
        -0.00399334, 0.30980017;
    out.true_covariances[2] << 0.26940591, -0.17738095,
        -0.17738095, 0.33059409;
    const Eigen::MatrixXd helmert = uac::normalized_helmert(topics);
    std::mt19937_64 engine(seed);
    std::uniform_int_distribution<int32_t> choose_cluster(0, components - 1);
    std::normal_distribution<double> normal(0.0, 1.0);
    out.labels.resize(documents);
    RowMajorMatrixXd latent(documents, topics - 1);
    RowMajorMatrixXd dense_counts = RowMajorMatrixXd::Zero(documents, features);
    out.data.identifiers.resize(documents);
    out.data.counts.resize(documents);
    out.data.raw_totals = Eigen::VectorXd::Constant(documents, 200.0);
    out.data.effective_totals = out.data.raw_totals;
    for (int32_t document = 0; document < documents; ++document) {
        const int32_t cluster = choose_cluster(engine);
        out.labels(document) = cluster;
        Eigen::Vector2d z(normal(engine), normal(engine));
        latent.row(document) = (out.true_means.row(cluster).transpose()
            + out.true_covariances[cluster].llt().matrixL() * z).transpose();
        const Eigen::VectorXd logits = helmert.transpose()
            * latent.row(document).transpose();
        Eigen::VectorXd composition = (logits.array()
            - logits.maxCoeff()).exp();
        composition /= composition.sum();
        const Eigen::VectorXd word_probability =
            out.basis.probabilities * composition;
        std::discrete_distribution<int32_t> choose_word(
            word_probability.data(), word_probability.data() + features);
        for (int32_t draw = 0; draw < 200; ++draw) {
            dense_counts(document, choose_word(engine)) += 1.0;
        }
        out.data.identifiers[document] = "doc_" + std::to_string(document);
        Document count;
        for (int32_t feature = 0; feature < features; ++feature) {
            if (dense_counts(document, feature) == 0.0) continue;
            count.ids.push_back(feature);
            count.cnts.push_back(dense_counts(document, feature));
        }
        count.raw_ct_tot = count.ct_tot = 200.0;
        out.data.counts[document] = std::move(count);
    }
    out.data.centers = RowMajorMatrixXd::Constant(documents, topics,
        1.0 / topics);
    for (int32_t iteration = 0; iteration < 100; ++iteration) {
        const RowMajorMatrixXd probability = (out.data.centers
            * out.basis.probabilities.transpose()).array().max(1e-300);
        RowMajorMatrixXd updated = out.data.centers.array()
            * ((dense_counts.array() / probability.array()).matrix()
                * out.basis.probabilities).array();
        updated.array() += 0.5;
        updated.array().colwise() /= updated.rowwise().sum().array();
        const double change = (updated - out.data.centers)
            .cwiseAbs().maxCoeff();
        out.data.centers = std::move(updated);
        if (change < 1e-10) break;
    }
    out.data.coordinates = uac::ilr_transform(out.data.centers, helmert);
    out.latent = std::move(latent);
    return out;
}

RowMajorMatrixXd profile_word_map(
    const Eigen::Ref<const RowMajorMatrixXd>& dense_counts,
    const uac::Basis& basis, const Eigen::Ref<const Eigen::VectorXd>& totals) {
    const int32_t documents = static_cast<int32_t>(dense_counts.rows());
    const int32_t topics = static_cast<int32_t>(basis.probabilities.cols());
    RowMajorMatrixXd centers = RowMajorMatrixXd::Constant(
        documents, topics, 1.0 / topics);
    for (int32_t iteration = 0; iteration < 200; ++iteration) {
        const RowMajorMatrixXd probability =
            (centers * basis.probabilities.transpose()).array().max(1e-300);
        RowMajorMatrixXd updated = centers.array()
            * ((dense_counts.array() / probability.array()).matrix()
                * basis.probabilities).array();
        updated.array() += 0.5;
        updated.array().colwise() /= updated.rowwise().sum().array();
        const double change = (updated - centers).cwiseAbs().maxCoeff();
        centers = std::move(updated);
        if (change < 1e-10) break;
    }
    return centers;
}

ProfileSimulation profile_simulation_large(int32_t documents, int32_t topics,
    int32_t features, int32_t components, int32_t seed,
    double point_center_error_target, const std::string& length_profile,
    const std::string& topic_identifiability, double supplied_mean_scale) {
    if (documents < 300 || topics < 3 || features < topics
        || components < 2 || components > topics) {
        throw std::invalid_argument(
            "Large UAC profile requires D>=300, K>=3, V>=K, and 2<=C<=K");
    }
    const int32_t dimension = topics - 1;
    if (length_profile != "fixed" && length_profile != "stratified") {
        throw std::invalid_argument(
            "Profile length mode must be fixed or stratified");
    }
    if (topic_identifiability != "homogeneous"
        && topic_identifiability != "mixed") {
        throw std::invalid_argument(
            "Profile topic identifiability must be homogeneous or mixed");
    }
    ProfileSimulation out;
    out.length_profile = length_profile;
    out.topic_identifiability = topic_identifiability;
    out.basis.features.resize(features);
    out.basis.topics.resize(topics);
    out.basis.probabilities.resize(features, topics);
    Eigen::VectorXd background(features);
    RowMajorMatrixXd anchor = RowMajorMatrixXd::Constant(
        features, topics, 1e-4);
    for (int32_t feature = 0; feature < features; ++feature) {
        out.basis.features[feature] = "feature_" + std::to_string(feature);
        background(feature) = std::pow(feature + 1.0, -1.1);
        anchor(feature, feature % topics) += 1.0
            / (1.0 + feature / topics);
    }
    for (int32_t topic = 0; topic < topics; ++topic) {
        out.basis.topics[topic] = "topic_" + std::to_string(topic);
    }
    background /= background.sum();
    std::vector<int32_t> topic_order(topics);
    std::iota(topic_order.begin(), topic_order.end(), int32_t{0});
    std::mt19937_64 topic_engine(
        static_cast<uint64_t>(seed) ^ 0x8f3f73b5cf1c9adeull);
    std::shuffle(topic_order.begin(), topic_order.end(), topic_engine);
    out.weak_topics.assign(topics, false);
    for (int32_t j = 0; j < topics / 2; ++j) {
        out.weak_topics[topic_order[j]] = true;
    }
    for (int32_t topic = 0; topic < topics; ++topic) {
        anchor.col(topic) /= anchor.col(topic).sum();
        const double strength = topic_identifiability == "mixed"
                && out.weak_topics[topic]
            ? 0.35 : 0.65;
        out.basis.probabilities.col(topic) = (1.0 - strength) * background
            + strength * anchor.col(topic);
    }
    uac::normalize_basis(out.basis);

    std::mt19937_64 engine(seed);
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::MatrixXd rotation_source(dimension, dimension);
    for (int32_t i = 0; i < dimension; ++i) {
        for (int32_t j = 0; j < dimension; ++j) {
            rotation_source(i, j) = normal(engine);
        }
    }
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(rotation_source);
    const Eigen::MatrixXd rotation = qr.householderQ()
        * Eigen::MatrixXd::Identity(dimension, dimension);
    RowMajorMatrixXd template_mean = RowMajorMatrixXd::Zero(
        components, dimension);
    template_mean.leftCols(components - 1) =
        uac::normalized_helmert(components).transpose();
    for (int32_t component = 0; component < components; ++component) {
        template_mean.row(component).normalize();
    }
    const RowMajorMatrixXd rotated_template =
        template_mean * rotation.transpose();
    const Eigen::MatrixXd helmert = uac::normalized_helmert(topics);
    out.true_covariances.resize(components);
    for (int32_t component = 0; component < components; ++component) {
        Eigen::VectorXd diagonal(dimension);
        for (int32_t j = 0; j < dimension; ++j) {
            diagonal(j) = 0.22 + 0.025 * ((j + component) % 5);
        }
        out.true_covariances[component] = rotation
            * diagonal.asDiagonal() * rotation.transpose();
    }

    uac::Basis calibration_basis = out.basis;
    for (int32_t topic = 0; topic < topics; ++topic) {
        calibration_basis.probabilities.col(topic) = 0.35 * background
            + 0.65 * anchor.col(topic);
    }
    uac::normalize_basis(calibration_basis);
    auto point_center_error = [&](double scale) {
        constexpr int32_t calibration_documents = 360;
        std::vector<Eigen::LLT<Eigen::MatrixXd>> covariance_factors;
        covariance_factors.reserve(components);
        std::vector<double> covariance_logdet(components);
        for (int32_t c = 0; c < components; ++c) {
            covariance_factors.emplace_back(out.true_covariances[c]);
            const Eigen::MatrixXd lower =
                covariance_factors.back().matrixL();
            covariance_logdet[c] =
                2.0 * lower.diagonal().array().log().sum();
        }
        RowMajorMatrixXd calibration_counts = RowMajorMatrixXd::Zero(
            calibration_documents, features);
        Eigen::VectorXi calibration_labels(calibration_documents);
        Eigen::VectorXd calibration_totals = Eigen::VectorXd::Constant(
            calibration_documents, 200.0);
        for (int32_t document = 0; document < calibration_documents;
                ++document) {
            const int32_t label = document % components;
            calibration_labels(document) = label;
            std::mt19937_64 calibration_engine(
                (static_cast<uint64_t>(seed) ^ 0x4a17b9d3ull)
                + static_cast<uint64_t>(document) * 0x9e3779b97f4a7c15ull);
            std::normal_distribution<double> calibration_normal(0.0, 1.0);
            Eigen::VectorXd z(dimension);
            for (int32_t j = 0; j < dimension; ++j) {
                z(j) = calibration_normal(calibration_engine);
            }
            const Eigen::VectorXd latent = scale
                * rotated_template.row(label).transpose()
                + covariance_factors[label].matrixL() * z;
            const Eigen::VectorXd logits = helmert.transpose() * latent;
            Eigen::VectorXd composition =
                (logits.array() - logits.maxCoeff()).exp();
            composition /= composition.sum();
            const Eigen::VectorXd word_probability =
                calibration_basis.probabilities * composition;
            std::discrete_distribution<int32_t> choose_word(
                word_probability.data(),
                word_probability.data() + features);
            for (int32_t draw = 0; draw < 200; ++draw) {
                ++calibration_counts(document,
                    choose_word(calibration_engine));
            }
        }
        const RowMajorMatrixXd calibration_centers = profile_word_map(
            calibration_counts, calibration_basis, calibration_totals);
        const RowMajorMatrixXd calibration_coordinates = uac::ilr_transform(
            calibration_centers, helmert);
        int32_t errors = 0;
        for (int32_t document = 0; document < calibration_documents;
                ++document) {
            const int32_t label = calibration_labels(document);
            const Eigen::VectorXd coordinate =
                calibration_coordinates.row(document).transpose();
            int32_t best = -1;
            double best_score = -std::numeric_limits<double>::infinity();
            for (int32_t c = 0; c < components; ++c) {
                const Eigen::VectorXd residual = coordinate
                    - scale * rotated_template.row(c).transpose();
                const Eigen::VectorXd solved =
                    covariance_factors[c].matrixL().solve(residual);
                const double score = -0.5
                    * (covariance_logdet[c] + solved.squaredNorm());
                if (score > best_score) {
                    best_score = score;
                    best = c;
                }
            }
            errors += best != label;
        }
        return static_cast<double>(errors) / calibration_documents;
    };
    if (supplied_mean_scale > 0.0) {
        out.mean_scale = supplied_mean_scale;
    } else {
        double lower_scale = 0.0, upper_scale = 8.0;
        for (int32_t iteration = 0; iteration < 14; ++iteration) {
            const double midpoint = 0.5 * (lower_scale + upper_scale);
            if (point_center_error(midpoint) > point_center_error_target) {
                lower_scale = midpoint;
            } else {
                upper_scale = midpoint;
            }
        }
        out.mean_scale = 0.5 * (lower_scale + upper_scale);
    }
    out.true_means = out.mean_scale * rotated_template;

    out.labels.resize(documents);
    std::vector<int32_t> balanced_labels(documents);
    for (int32_t document = 0; document < documents; ++document) {
        balanced_labels[document] = document % components;
    }
    std::shuffle(balanced_labels.begin(), balanced_labels.end(), engine);
    out.data.identifiers.resize(documents);
    out.data.centers.resize(documents, topics);
    out.data.counts.resize(documents);
    out.document_lengths.resize(documents);
    out.data.raw_totals.resize(documents);
    out.data.effective_totals.resize(documents);
    out.latent.resize(documents, dimension);
    std::vector<int32_t> within_cluster(components, 0);
    for (int32_t document = 0; document < documents; ++document) {
        const int32_t cluster = balanced_labels[document];
        const int32_t stratum = within_cluster[cluster]++ % 3;
        const int32_t length = length_profile == "fixed"
            ? 200 : std::array<int32_t, 3>{{50, 150, 400}}[stratum];
        out.document_lengths(document) = length;
        out.data.raw_totals(document) = length;
        out.data.effective_totals(document) = length;
    }
    std::mt19937_64 latent_engine(seed);
    std::normal_distribution<double> latent_normal(0.0, 1.0);
    for (int32_t document = 0; document < documents; ++document) {
        const int32_t cluster = balanced_labels[document];
        out.labels(document) = cluster;
        Eigen::VectorXd z(dimension);
        for (int32_t j = 0; j < dimension; ++j) {
            z(j) = latent_normal(latent_engine);
        }
        const Eigen::VectorXd latent = out.true_means.row(cluster).transpose()
            + out.true_covariances[cluster].llt().matrixL() * z;
        out.latent.row(document) = latent.transpose();
    }
    RowMajorMatrixXd dense_counts = RowMajorMatrixXd::Zero(
        documents, features);
    out.low_topic_mass.resize(documents);
    for (int32_t document = 0; document < documents; ++document) {
        const Eigen::VectorXd latent = out.latent.row(document).transpose();
        Eigen::VectorXd logits = helmert.transpose() * latent;
        Eigen::VectorXd composition =
            (logits.array() - logits.maxCoeff()).exp();
        composition /= composition.sum();
        double weak_mass = 0.0;
        for (int32_t topic = 0; topic < topics; ++topic) {
            if (out.weak_topics[topic]) weak_mass += composition(topic);
        }
        out.low_topic_mass(document) = weak_mass;
        const Eigen::VectorXd word_probability =
            out.basis.probabilities * composition;
        std::discrete_distribution<int32_t> choose_word(
            word_probability.data(), word_probability.data() + features);
        std::mt19937_64 word_engine(
            (static_cast<uint64_t>(seed) ^ 0xd1b54a32d192ed03ull)
            + static_cast<uint64_t>(document) * 0x9e3779b97f4a7c15ull);
        for (int32_t draw = 0; draw < out.document_lengths(document); ++draw) {
            ++dense_counts(document, choose_word(word_engine));
        }
        out.data.identifiers[document] = "doc_" + std::to_string(document);
        Document count;
        count.raw_ct_tot = count.ct_tot = out.document_lengths(document);
        for (int32_t feature = 0; feature < features; ++feature) {
            if (dense_counts(document, feature) == 0.0) continue;
            count.ids.push_back(feature);
            count.cnts.push_back(dense_counts(document, feature));
        }
        out.data.counts[document] = std::move(count);
    }
    out.data.centers = profile_word_map(
        dense_counts, out.basis, out.data.raw_totals);
    out.data.coordinates = uac::ilr_transform(out.data.centers, helmert);
    out.map_ilr_error.resize(documents);
    out.oracle_fisher_trace.resize(documents);
    out.oracle_fisher_logdet.resize(documents);
    out.oracle_fisher_maximum_eigenvalue.resize(documents);
    out.oracle_fisher_condition.resize(documents);
    for (int32_t document = 0; document < documents; ++document) {
        out.map_ilr_error(document) = (out.data.coordinates.row(document)
            - out.latent.row(document)).norm();
        const Eigen::VectorXd truth = out.latent.row(document).transpose();
        const uac::FisherApproximation fisher = uac::fisher_approximation(
            truth, out.data.counts[document], out.basis, helmert,
            uac::ProposalKind::ExactFisher);
        const Eigen::MatrixXd precision = fisher.information
            + out.true_covariances[out.labels(document)].inverse();
        const Eigen::MatrixXd covariance = precision.inverse();
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(covariance);
        out.oracle_fisher_trace(document) = covariance.trace();
        out.oracle_fisher_logdet(document) =
            eigen.eigenvalues().array().log().sum();
        out.oracle_fisher_maximum_eigenvalue(document) =
            eigen.eigenvalues().maxCoeff();
        out.oracle_fisher_condition(document) =
            eigen.eigenvalues().maxCoeff()
            / eigen.eigenvalues().minCoeff();
    }
    return out;
}

std::vector<int32_t> align_profile_components(
    const uac::Model& model, const RowMajorMatrixXd& truth) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    if (truth.rows() != components) {
        throw std::invalid_argument("Profile alignment component mismatch");
    }
    std::vector<double> u(components + 1), v(components + 1);
    std::vector<int32_t> p(components + 1), way(components + 1);
    for (int32_t i = 1; i <= components; ++i) {
        p[0] = i;
        int32_t j0 = 0;
        std::vector<double> minimum(components + 1,
            std::numeric_limits<double>::infinity());
        std::vector<bool> used(components + 1, false);
        do {
            used[j0] = true;
            const int32_t i0 = p[j0];
            double delta = std::numeric_limits<double>::infinity();
            int32_t j1 = 0;
            for (int32_t j = 1; j <= components; ++j) {
                if (used[j]) continue;
                const double cost = (model.means.row(i0 - 1)
                    - truth.row(j - 1)).squaredNorm();
                const double current = cost - u[i0] - v[j];
                if (current < minimum[j]) {
                    minimum[j] = current;
                    way[j] = j0;
                }
                if (minimum[j] < delta) {
                    delta = minimum[j];
                    j1 = j;
                }
            }
            for (int32_t j = 0; j <= components; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minimum[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            const int32_t j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 != 0);
    }
    std::vector<int32_t> out(components, -1);
    for (int32_t j = 1; j <= components; ++j) out[p[j] - 1] = j - 1;
    return out;
}

struct ProfilePosteriorMoments {
    Eigen::VectorXd responsibilities;
    std::vector<Eigen::VectorXd> means;
    std::vector<Eigen::MatrixXd> covariances;
    Eigen::VectorXd conditional_ess;
};

double profile_logsumexp(const Eigen::Ref<const Eigen::VectorXd>& values) {
    const double maximum = values.maxCoeff();
    return maximum
        + std::log((values.array() - maximum).exp().sum());
}

ProfilePosteriorMoments profile_posterior_moments(
    const uac::ParticleSet& particles, int32_t document, int32_t samples,
    const uac::Model& model) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = particles.dimension;
    const auto values = particles.values_for_document(document).topRows(samples);
    const auto log_likelihood =
        particles.log_likelihood_for_document(document).head(samples);
    const auto log_proposal =
        particles.log_proposal_for_document(document).head(samples);
    std::vector<Eigen::LLT<Eigen::MatrixXd>> factors;
    std::vector<double> logdet(components);
    factors.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        factors.emplace_back(model.covariances[c]);
        const Eigen::MatrixXd lower = factors.back().matrixL();
        logdet[c] = 2.0 * lower.diagonal().array().log().sum();
    }
    Eigen::MatrixXd probability(components, samples);
    Eigen::VectorXd score(components);
    for (int32_t c = 0; c < components; ++c) {
        Eigen::VectorXd log_weight(samples);
        for (int32_t s = 0; s < samples; ++s) {
            const Eigen::VectorXd residual = values.row(s).transpose()
                - model.means.row(c).transpose();
            const Eigen::VectorXd solved =
                factors[c].matrixL().solve(residual);
            log_weight(s) = log_likelihood(s) - log_proposal(s)
                - 0.5 * (dimension * std::log(2.0 * M_PI)
                    + logdet[c] + solved.squaredNorm());
        }
        const double evidence = profile_logsumexp(log_weight);
        probability.row(c) =
            (log_weight.array() - evidence).exp().transpose();
        score(c) = std::log(std::max(1e-300, model.weights(c)))
            + evidence - std::log(samples);
    }
    const double normalizer = profile_logsumexp(score);
    ProfilePosteriorMoments out;
    out.responsibilities = (score.array() - normalizer).exp();
    out.means.resize(components);
    out.covariances.resize(components);
    out.conditional_ess.resize(components);
    for (int32_t c = 0; c < components; ++c) {
        out.means[c] = values.transpose() * probability.row(c).transpose();
        Eigen::MatrixXd second = Eigen::MatrixXd::Zero(dimension, dimension);
        for (int32_t s = 0; s < samples; ++s) {
            second.noalias() += probability(c, s)
                * values.row(s).transpose() * values.row(s);
        }
        out.covariances[c] = second
            - out.means[c] * out.means[c].transpose();
        out.covariances[c] = 0.5
            * (out.covariances[c] + out.covariances[c].transpose());
        out.conditional_ess(c) =
            1.0 / probability.row(c).squaredNorm();
    }
    return out;
}

void write_profile_particle_audit(const std::string& output,
    const ProfileSimulation& simulation, const uac::FitResult& fit,
    int32_t requested_documents, int32_t reference_particles,
    int32_t seed, int32_t threads) {
    if (output.empty() || requested_documents <= 0
        || reference_particles < 256) return;
    const int32_t available = static_cast<int32_t>(
        simulation.data.identifiers.size());
    const int32_t documents = std::min(requested_documents, available);
    uac::Dataset audit;
    audit.centers.resize(documents, simulation.data.centers.cols());
    audit.coordinates.resize(documents, simulation.data.coordinates.cols());
    audit.raw_totals.resize(documents);
    audit.effective_totals.resize(documents);
    std::vector<int32_t> source(documents);
    for (int32_t d = 0; d < documents; ++d) {
        const int32_t original = static_cast<int32_t>(
            static_cast<int64_t>(d) * available / documents);
        source[d] = original;
        audit.identifiers.push_back(simulation.data.identifiers[original]);
        audit.counts.push_back(simulation.data.counts[original]);
        audit.centers.row(d) = simulation.data.centers.row(original);
        audit.coordinates.row(d) = simulation.data.coordinates.row(original);
        audit.raw_totals(d) = simulation.data.raw_totals(original);
        audit.effective_totals(d) = simulation.data.effective_totals(original);
    }
    const Eigen::MatrixXd helmert = uac::normalized_helmert(
        simulation.data.centers.cols());
    const uac::ParticleSet particles = uac::make_particles(audit,
        simulation.basis, helmert, fit.pilot,
        uac::ProposalKind::ExactFisher, reference_particles,
        static_cast<uint64_t>(seed) ^ 0x7f4a7c15ull, 1.5, threads);
    uac::Model truth;
    truth.weights = Eigen::VectorXd::Constant(
        simulation.true_means.rows(), 1.0 / simulation.true_means.rows());
    truth.means = simulation.true_means;
    truth.covariances = simulation.true_covariances;
    uac::Model pilot;
    pilot.weights = fit.pilot.weights;
    pilot.means = fit.pilot.means;
    pilot.covariances = fit.pilot.covariances;
    std::ofstream stream(output + ".audit.tsv");
    if (!stream) {
        throw std::runtime_error("Cannot write UAC particle audit: " + output);
    }
    stream << "document\tlabel\tlength\tlow_topic_mass\tmodel\tparticles"
        "\tmaximum_responsibility_error\tconditional_ess_mean"
        "\tstandardized_mean_error\tcovariance_relative_error"
        "\tcovariance_diagonal_signed_true_relative_error"
        "\tfirst_contribution_relative_error"
        "\tsecond_contribution_relative_error\n" << std::setprecision(12);
    const std::array<int32_t, 5> tiers{{32, 64, 128, 256,
        reference_particles}};
    for (const auto& named_model : {
            std::pair<const char*, const uac::Model*>{"truth", &truth},
            std::pair<const char*, const uac::Model*>{"pilot", &pilot}}) {
        for (int32_t d = 0; d < documents; ++d) {
            const ProfilePosteriorMoments reference =
                profile_posterior_moments(
                    particles, d, reference_particles, *named_model.second);
            for (const int32_t tier : tiers) {
                if (tier > reference_particles) continue;
                const ProfilePosteriorMoments estimate =
                    profile_posterior_moments(
                        particles, d, tier, *named_model.second);
                double ess = 0.0, mean_error = 0.0, covariance_error = 0.0;
                double diagonal_error = 0.0;
                double first_error = 0.0, second_error = 0.0;
                int32_t plausible = 0;
                for (Eigen::Index c = 0;
                        c < reference.responsibilities.size(); ++c) {
                    if (reference.responsibilities(c) < 0.05) continue;
                    ++plausible;
                    ess += estimate.conditional_ess(c);
                    Eigen::MatrixXd reference_covariance =
                        reference.covariances[c];
                    reference_covariance.diagonal().array() += 1e-8;
                    const Eigen::VectorXd delta =
                        estimate.means[c] - reference.means[c];
                    mean_error += std::sqrt(std::max(0.0,
                        delta.dot(reference_covariance.ldlt().solve(delta))
                            / delta.size()));
                    covariance_error += (estimate.covariances[c]
                        - reference.covariances[c]).norm()
                        / std::max(1e-12, reference.covariances[c].norm());
                    const Eigen::VectorXd true_diagonal =
                        simulation.true_covariances[simulation.labels(
                            source[d])].diagonal();
                    diagonal_error += ((estimate.covariances[c].diagonal()
                        - reference.covariances[c].diagonal()).array()
                        / true_diagonal.array()).mean();
                    const Eigen::VectorXd estimate_first =
                        estimate.responsibilities(c) * estimate.means[c];
                    const Eigen::VectorXd reference_first =
                        reference.responsibilities(c) * reference.means[c];
                    first_error += (estimate_first - reference_first).norm()
                        / std::max(1e-12, reference_first.norm() + 1.0);
                    const Eigen::MatrixXd estimate_second =
                        estimate.responsibilities(c)
                        * (estimate.covariances[c]
                            + estimate.means[c] * estimate.means[c].transpose());
                    const Eigen::MatrixXd reference_second =
                        reference.responsibilities(c)
                        * (reference.covariances[c]
                            + reference.means[c] * reference.means[c].transpose());
                    second_error += (estimate_second - reference_second).norm()
                        / std::max(1e-12, reference_second.norm() + 1.0);
                }
                const double divisor = std::max(1, plausible);
                const int32_t original = source[d];
                stream << original << "\t" << simulation.labels(original)
                    << "\t" << simulation.document_lengths(original)
                    << "\t" << simulation.low_topic_mass(original)
                    << "\t" << named_model.first << "\t" << tier
                    << "\t" << (estimate.responsibilities
                        - reference.responsibilities).cwiseAbs().maxCoeff()
                    << "\t" << ess / divisor
                    << "\t" << mean_error / divisor
                    << "\t" << covariance_error / divisor
                    << "\t" << diagonal_error / divisor
                    << "\t" << first_error / divisor
                    << "\t" << second_error / divisor << "\n";
            }
        }
    }
}

void run_uac_profile(int32_t documents, int32_t particles, int32_t seed,
    int32_t threads, int32_t topics, int32_t features, int32_t components,
    double point_center_error_target, const std::string& length_profile,
    const std::string& topic_identifiability, double simulation_mean_scale,
    int32_t audit_documents, int32_t audit_particles,
    uac::ProposalKind proposal,
    const uac::AdaptiveParticleOptions& adaptive,
    const std::string& output) {
    ProfileSimulation simulation = topics == 3 && features == 60
            && components == 3
        ? profile_simulation(documents, seed)
        : profile_simulation_large(
            documents, topics, features, components, seed,
            point_center_error_target, length_profile,
            topic_identifiability, simulation_mean_scale);
    uac::FitOptions options;
    options.n_components = components;
    options.n_particles = particles;
    options.kmeans_starts = 1;
    options.max_iterations = 60;
    options.kmeans_max_iterations = 100;
    options.seed = seed;
    options.n_threads = threads;
    options.proposal = proposal;
    options.adaptive_particles = adaptive;
    const auto begin = std::chrono::steady_clock::now();
    const uac::FitResult fit = uac::fit(simulation.data, &simulation.basis,
        options);
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    write_profile_particle_audit(output, simulation, fit,
        audit_documents, audit_particles, seed, threads);
    const auto alignment = align_profile_components(fit.model,
        simulation.true_means);
    std::vector<int32_t> fitted_for_truth(components, -1);
    for (int32_t fitted = 0; fitted < components; ++fitted) {
        fitted_for_truth[alignment[fitted]] = fitted;
    }
    std::vector<Eigen::LLT<Eigen::MatrixXd>> fitted_factors;
    std::vector<double> fitted_logdet(components);
    fitted_factors.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        fitted_factors.emplace_back(fit.model.covariances[c]);
        const Eigen::MatrixXd lower = fitted_factors.back().matrixL();
        fitted_logdet[c] = 2.0 * lower.diagonal().array().log().sum();
    }
    std::vector<Eigen::LLT<Eigen::MatrixXd>> oracle_factors;
    std::vector<double> oracle_logdet(components);
    oracle_factors.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        oracle_factors.emplace_back(simulation.true_covariances[c]);
        const Eigen::MatrixXd lower = oracle_factors.back().matrixL();
        oracle_logdet[c] = 2.0 * lower.diagonal().array().log().sum();
    }
    const Eigen::MatrixXd helmert = uac::normalized_helmert(topics);
    auto gaussian_scores = [&](const Eigen::VectorXd& value,
            const RowMajorMatrixXd& means,
            const std::vector<Eigen::LLT<Eigen::MatrixXd>>& factors,
            const std::vector<double>& logdet,
            const Eigen::VectorXd* weights) {
        Eigen::VectorXd score(components);
        for (int32_t c = 0; c < components; ++c) {
            const Eigen::VectorXd solved = factors[c].matrixL().solve(
                value - means.row(c).transpose());
            score(c) = -0.5 * (value.size() * std::log(2.0 * M_PI)
                + logdet[c] + solved.squaredNorm());
            if (weights != nullptr) score(c) += std::log((*weights)(c));
        }
        return score;
    };
    double log_loss = 0.0, true_probability = 0.0, entropy = 0.0;
    double oracle_entropy_mean = 0.0, entropy_cross = 0.0;
    double entropy_square = 0.0, oracle_entropy_square = 0.0;
    double hpd80 = 0.0, hpd95 = 0.0;
    int32_t point_center_errors = 0;
    std::vector<double> oracle_entropy_by_document(documents);
    for (int32_t document = 0; document < documents; ++document) {
        const double probability = fit.score.responsibilities(document,
            fitted_for_truth[simulation.labels(document)]);
        true_probability += probability;
        log_loss -= std::log(std::max(1e-300, probability));
        double document_entropy = 0.0;
        for (int32_t component = 0; component < components; ++component) {
            const double p = fit.score.responsibilities(document, component);
            if (p > 0.0) document_entropy -= p * std::log(p);
        }
        entropy += document_entropy;
        const Eigen::VectorXd oracle_score = gaussian_scores(
            simulation.data.coordinates.row(document).transpose(),
            simulation.true_means, oracle_factors, oracle_logdet, nullptr);
        point_center_errors += oracle_score.maxCoeff()
            != oracle_score(simulation.labels(document));
        const double oracle_maximum = oracle_score.maxCoeff();
        const double oracle_normalizer = oracle_maximum + std::log(
            (oracle_score.array() - oracle_maximum).exp().sum());
        const Eigen::VectorXd oracle_probability =
            (oracle_score.array() - oracle_normalizer).exp();
        double oracle_entropy = 0.0;
        for (int32_t c = 0; c < components; ++c) {
            if (oracle_probability(c) > 0.0) {
                oracle_entropy -= oracle_probability(c)
                    * std::log(oracle_probability(c));
            }
        }
        oracle_entropy_by_document[document] = oracle_entropy;
        oracle_entropy_mean += oracle_entropy;
        entropy_cross += document_entropy * oracle_entropy;
        entropy_square += document_entropy * document_entropy;
        oracle_entropy_square += oracle_entropy * oracle_entropy;

        const Eigen::VectorXd truth = simulation.latent.row(document).transpose();
        const RowMajorMatrixXd truth_row = truth.transpose();
        const Eigen::VectorXd composition =
            uac::ilr_inverse(truth_row, helmert).row(0).transpose();
        const Eigen::VectorXd word_probability =
            simulation.basis.probabilities * composition;
        double truth_log_likelihood = 0.0;
        const Document& count = simulation.data.counts[document];
        for (size_t j = 0; j < count.ids.size(); ++j) {
            truth_log_likelihood += count.cnts[j]
                * std::log(std::max(1e-300,
                    word_probability(count.ids[j])));
        }
        const Eigen::VectorXd truth_component = gaussian_scores(truth,
            fit.model.means, fitted_factors, fitted_logdet,
            &fit.model.weights);
        const double truth_maximum = truth_component.maxCoeff();
        const double truth_log_prior = truth_maximum + std::log(
            (truth_component.array() - truth_maximum).exp().sum());
        const double truth_log_target =
            truth_log_likelihood + truth_log_prior;
        hpd80 += truth_log_target >= fit.score.particle_diagnostics[document]
            .hpd80_log_density_threshold;
        hpd95 += truth_log_target >= fit.score.particle_diagnostics[document]
            .hpd95_log_density_threshold;
    }
    log_loss /= documents;
    true_probability /= documents;
    const double entropy_mean = entropy / documents;
    oracle_entropy_mean /= documents;
    const double entropy_covariance = entropy_cross / documents
        - entropy_mean * oracle_entropy_mean;
    const double entropy_variance = entropy_square / documents
        - entropy_mean * entropy_mean;
    const double oracle_entropy_variance = oracle_entropy_square / documents
        - oracle_entropy_mean * oracle_entropy_mean;
    const double entropy_correlation = entropy_covariance / std::sqrt(
        std::max(1e-300, entropy_variance * oracle_entropy_variance));
    double covariance_error = 0.0, diagonal_error = 0.0;
    double marginal_error = 0.0;
    for (int32_t fitted = 0; fitted < components; ++fitted) {
        const int32_t truth = alignment[fitted];
        covariance_error += (fit.model.covariances[fitted]
            - simulation.true_covariances[truth]).norm()
            / simulation.true_covariances[truth].norm();
        diagonal_error += (fit.model.covariances[fitted].diagonal()
            - simulation.true_covariances[truth].diagonal()).sum()
            / simulation.true_covariances[truth].diagonal().sum();
        for (int32_t dimension = 0;
                dimension < fit.model.covariances[fitted].rows(); ++dimension) {
            marginal_error += (fit.model.covariances[fitted](
                    dimension, dimension)
                - simulation.true_covariances[truth](dimension, dimension))
                / simulation.true_covariances[truth](dimension, dimension);
        }
    }
    covariance_error /= components;
    diagonal_error /= components;
    marginal_error /= components * (topics - 1);
    std::ostream* stream = &std::cout;
    std::ofstream file;
    if (!output.empty()) {
        file.open(output);
        if (!file) throw std::runtime_error("Cannot write UAC profile: " + output);
        stream = &file;
        std::ofstream responsibility(output + ".responsibilities.tsv");
        if (!responsibility) {
            throw std::runtime_error(
                "Cannot write UAC profile responsibilities: " + output);
        }
        responsibility << "document\tlabel";
        for (int32_t truth = 0; truth < components; ++truth) {
            responsibility << "\tp" << truth;
        }
        responsibility << "\n" << std::setprecision(12);
        for (int32_t document = 0; document < documents; ++document) {
            responsibility << document << "\t" << simulation.labels(document);
            for (int32_t truth = 0; truth < components; ++truth) {
                responsibility << "\t" << fit.score.responsibilities(
                    document, fitted_for_truth[truth]);
            }
            responsibility << "\n";
        }
        if (simulation.document_lengths.size() == documents) {
            std::ofstream diagnostics(output + ".simulation.tsv");
            if (!diagnostics) {
                throw std::runtime_error(
                    "Cannot write UAC profile simulation diagnostics: "
                    + output);
            }
            diagnostics << "document\tlabel\tlength\tnnz\tlow_topic_mass"
                "\tmap_ilr_error\toracle_fisher_trace"
                "\toracle_fisher_logdet\toracle_fisher_maximum_eigenvalue"
                "\toracle_fisher_condition\toracle_assignment_entropy\n"
                << std::setprecision(12);
            for (int32_t document = 0; document < documents; ++document) {
                diagnostics << document << "\t" << simulation.labels(document)
                    << "\t" << simulation.document_lengths(document)
                    << "\t" << simulation.data.counts[document].ids.size()
                    << "\t" << simulation.low_topic_mass(document)
                    << "\t" << simulation.map_ilr_error(document)
                    << "\t" << simulation.oracle_fisher_trace(document)
                    << "\t" << simulation.oracle_fisher_logdet(document)
                    << "\t" << simulation
                        .oracle_fisher_maximum_eigenvalue(document)
                    << "\t" << simulation.oracle_fisher_condition(document)
                    << "\t" << oracle_entropy_by_document[document] << "\n";
            }
        }
        if (!fit.score.adaptive_particle_diagnostics.empty()) {
            std::ofstream allocation(output + ".allocation.tsv");
            if (!allocation) {
                throw std::runtime_error(
                    "Cannot write UAC profile allocation diagnostics: "
                    + output);
            }
            allocation << "document\tlabel\toracle_entropy"
                "\tpreliminary_maximum_responsibility"
                "\tpreliminary_entropy\tplausible_components"
                "\tmaximum_responsibility_se"
                "\tprojected_responsibility_particles"
                "\tprojected_moment_particles"
                "\tplausible_maximum_weight"
                "\thalf_sample_maximum_responsibility_difference"
                "\thalf_sample_top_disagreement"
                "\tselected_particles\tbinding\n" << std::setprecision(12);
            for (int32_t document = 0; document < documents; ++document) {
                const auto& diagnostic =
                    fit.score.adaptive_particle_diagnostics[document];
                allocation << document << "\t" << simulation.labels(document)
                    << "\t" << oracle_entropy_by_document[document]
                    << "\t" << diagnostic.preliminary_maximum_responsibility
                    << "\t" << diagnostic.preliminary_entropy
                    << "\t" << diagnostic.plausible_components
                    << "\t" << diagnostic.maximum_responsibility_se
                    << "\t" << diagnostic.projected_responsibility_particles
                    << "\t" << diagnostic.projected_moment_particles
                    << "\t" << diagnostic.plausible_maximum_weight
                    << "\t" << diagnostic
                        .half_sample_maximum_responsibility_difference
                    << "\t" << diagnostic.half_sample_top_disagreement
                    << "\t" << diagnostic.selected_particles
                    << "\t" << uac::adaptive_particle_binding_name(
                        diagnostic.binding) << "\n";
            }
        }
    }
    std::map<int32_t, int32_t> particle_histogram;
    for (const int32_t value : fit.score.per_document_particles) {
        ++particle_histogram[value];
    }
    double mean_particles = fit.score.per_document_particles.empty() ? 0.0
        : static_cast<double>(fit.score.particle_samples) / documents;
    *stream << "documents\tfeatures\ttopics\tcomponents"
        "\tlength_profile\ttopic_identifiability\tmean_scale"
        "\tparticles\tadaptive"
        "\tadaptive_rule\tcalibration_particles"
        "\tess_target\tcontrast_se_target"
        "\tmaximum_weight_target\tproposal\twall_seconds"
        "\tparticle_generation_seconds"
        "\tsampling_seconds\tfisher_work_seconds"
        "\tproposal_component_work_seconds"
        "\tproposal_draw_density_work_seconds"
        "\tlikelihood_seconds\tscoring_seconds"
        "\tgaussian_work_seconds\tmoment_work_seconds"
        "\tparticle_em_objective_evaluations\tparticle_bytes"
        "\tproposal_workspace_bytes\texpectation_accumulator_bytes"
        "\tmean_true_label_probability\tlog_loss"
        "\tmean_assignment_entropy\toracle_assignment_entropy"
        "\tassignment_entropy_correlation\tpoint_center_oracle_error"
        "\thpd80\thpd95\tcovariance_relative_error"
        "\tcovariance_diagonal_signed_relative_error"
        "\tcovariance_marginal_signed_relative_error"
        "\tcalibration_seconds\tcalibration_samples\tretained_samples"
        "\tmean_particles\ttier32\ttier64\ttier128\ttier256\n"
        << std::setprecision(12) << documents << "\t" << features
        << "\t" << topics << "\t" << components
        << "\t" << simulation.length_profile
        << "\t" << simulation.topic_identifiability
        << "\t" << simulation.mean_scale << "\t" << particles
        << "\t" << static_cast<int32_t>(adaptive.enabled)
        << "\t" << uac::adaptive_particle_rule_name(adaptive.rule)
        << "\t" << adaptive.calibration_particles
        << "\t" << adaptive.component_ess_target
        << "\t" << adaptive.contrast_se_target
        << "\t" << adaptive.maximum_weight_target << "\t"
        << uac::proposal_name(proposal) << "\t" << wall_seconds
        << "\t" << fit.score.particle_generation_seconds << "\t"
        << fit.score.sampling_seconds << "\t"
        << fit.score.fisher_work_seconds << "\t"
        << fit.score.proposal_component_work_seconds << "\t"
        << fit.score.proposal_draw_density_work_seconds << "\t"
        << fit.score.likelihood_seconds << "\t" << fit.score.scoring_seconds
        << "\t" << fit.score.gaussian_seconds
        << "\t" << fit.score.moment_seconds
        << "\t" << fit.traces.back().objective.size() << "\t"
        << fit.score.particle_bytes << "\t"
        << fit.score.proposal_workspace_bytes << "\t"
        << fit.score.expectation_accumulator_bytes
        << "\t" << true_probability
        << "\t" << log_loss << "\t" << entropy_mean
        << "\t" << oracle_entropy_mean << "\t" << entropy_correlation
        << "\t" << static_cast<double>(point_center_errors) / documents
        << "\t" << hpd80 / documents << "\t" << hpd95 / documents
        << "\t" << covariance_error << "\t" << diagonal_error
        << "\t" << marginal_error
        << "\t" << fit.score.calibration_seconds
        << "\t" << fit.score.calibration_samples
        << "\t" << fit.score.particle_samples << "\t" << mean_particles
        << "\t" << particle_histogram[32]
        << "\t" << particle_histogram[64]
        << "\t" << particle_histogram[128]
        << "\t" << particle_histogram[256] << "\n";
}

void test_transforms() {
    const Eigen::MatrixXd helmert = uac::normalized_helmert(4);
    require((helmert * helmert.transpose()
        - Eigen::MatrixXd::Identity(3, 3)).norm() < 1e-12,
        "Helmert rows are not orthonormal");
    require((helmert * Eigen::VectorXd::Ones(4)).norm() < 1e-12,
        "Helmert contrasts do not annihilate the constant vector");
    RowMajorMatrixXd composition(2, 4);
    composition << 0.1, 0.2, 0.3, 0.4,
        0.65, 0.15, 0.12, 0.08;
    const RowMajorMatrixXd coordinate = uac::ilr_transform(composition, helmert);
    const RowMajorMatrixXd recovered = uac::ilr_inverse(coordinate, helmert);
    require((composition - recovered).cwiseAbs().maxCoeff() < 1e-12,
        "ILR round trip differs");
}

void test_profile_heterogeneous_simulation() {
    const ProfileSimulation simulation = profile_simulation_large(
        300, 12, 120, 3, 20260719, 0.03,
        "stratified", "mixed", 1.55);
    require(simulation.weak_topics.size() == 12
            && std::count(simulation.weak_topics.begin(),
                simulation.weak_topics.end(), true) == 6,
        "heterogeneous profile did not create six weak topics");
    require(std::abs(simulation.document_lengths.cast<double>().mean()
                - 200.0) < 2.0
            && (simulation.data.centers.rowwise().sum().array() - 1.0)
                .abs().maxCoeff() < 1e-12
            && simulation.data.centers.allFinite(),
        "heterogeneous profile lengths or word-derived centers are invalid");
    Eigen::VectorXd background(120);
    for (int32_t feature = 0; feature < 120; ++feature) {
        background(feature) = std::pow(feature + 1.0, -1.1);
    }
    background /= background.sum();
    double weak_distance = 0.0, moderate_distance = 0.0;
    for (int32_t topic = 0; topic < 12; ++topic) {
        const double distance = (simulation.basis.probabilities.col(topic)
            - background).norm();
        if (simulation.weak_topics[topic]) weak_distance += distance;
        else moderate_distance += distance;
    }
    require(weak_distance < moderate_distance,
        "weak profile topics are not closer to the shared background");
    std::map<int32_t, std::vector<double>> trace;
    for (int32_t document = 0; document < 300; ++document) {
        trace[simulation.document_lengths(document)].push_back(
            simulation.oracle_fisher_trace(document));
    }
    auto mean = [](const std::vector<double>& values) {
        return std::accumulate(values.begin(), values.end(), 0.0)
            / values.size();
    };
    require(mean(trace[50]) > mean(trace[150])
            && mean(trace[150]) > mean(trace[400]),
        "oracle Fisher uncertainty does not decrease with document length");
}

void test_fractional_likelihood_and_proposals() {
    const uac::Basis basis = test_basis();
    const uac::Dataset data = test_dataset();
    const Eigen::MatrixXd helmert = uac::normalized_helmert(3);
    uac::FitOptions pilot_options;
    pilot_options.handoff = uac::HandoffMode::Map;
    pilot_options.n_components = 3;
    pilot_options.kmeans_starts = 1;
    pilot_options.max_iterations = 80;
    pilot_options.seed = 19;
    const uac::Pilot pilot = uac::fit(data, nullptr, pilot_options).pilot;
    require(uac::parse_proposal("exact_fisher")
            == uac::ProposalKind::ExactFisher
        && uac::parse_proposal("sparse_empirical_fisher")
            == uac::ProposalKind::SparseEmpiricalFisher,
        "Fisher proposal names do not round trip");
    bool rejected_stale_proposal = false;
    try {
        static_cast<void>(uac::parse_proposal("laplace"));
    } catch (const std::invalid_argument&) {
        rejected_stale_proposal = true;
    }
    require(rejected_stale_proposal, "stale Laplace proposal was accepted");

    notice("UAC proposal test: exact Fisher draw");
    const uac::ParticleSet exact = uac::make_particles(data, basis,
        helmert, pilot, uac::ProposalKind::ExactFisher, 24, 991, 1.5, 2);
    const uac::ParticleSet exact_serial = uac::make_particles(data, basis,
        helmert, pilot, uac::ProposalKind::ExactFisher, 24, 991, 1.5, 1);
    require(exact.values.allFinite() && exact.log_proposal.allFinite()
        && exact.log_likelihood.allFinite(),
        "exact Fisher proposal produced nonfinite values");
    require((exact.values - exact_serial.values).cwiseAbs().maxCoeff() < 1e-14
        && (exact.log_proposal - exact_serial.log_proposal)
            .cwiseAbs().maxCoeff() < 1e-12
        && (exact.log_likelihood - exact_serial.log_likelihood)
            .cwiseAbs().maxCoeff() < 1e-12,
        "exact Fisher proposal depends on thread count");
    for (int32_t document = 0; document < exact.documents; ++document) {
        for (int32_t sample = 0; sample < exact.samples; ++sample) {
            const RowMajorMatrixXd coordinate =
                exact.value(document, sample).transpose();
            const Eigen::VectorXd composition =
                uac::ilr_inverse(coordinate, helmert).row(0).transpose();
            const Eigen::VectorXd probability =
                basis.probabilities * composition;
            double reference = 0.0;
            for (size_t j = 0; j < data.counts[document].ids.size(); ++j) {
                reference += data.counts[document].cnts[j] * std::log(
                    probability(data.counts[document].ids[j]));
            }
            require(std::abs(reference
                    - exact.log_likelihood(document, sample)) < 1e-10,
                "observed-feature batched likelihood differs from dense likelihood");
        }
    }
    require(exact.fisher_work_seconds > 0.0
            && exact.proposal_component_work_seconds > 0.0
            && exact.proposal_draw_density_work_seconds > 0.0
            && exact.proposal_workspace_bytes > 0,
        "proposal phase timing or workspace diagnostics are missing");

    notice("UAC proposal test: sparse empirical Fisher draw");
    const uac::ParticleSet sparse = uac::make_particles(data, basis,
        helmert, pilot, uac::ProposalKind::SparseEmpiricalFisher,
        24, 991, 1.5, 2);
    require(sparse.values.allFinite() && sparse.log_proposal.allFinite()
        && sparse.log_likelihood.allFinite(),
        "sparse empirical Fisher proposal produced nonfinite values");

    // Expected multinomial counts make empirical Fisher equal exact Fisher.
    const Eigen::VectorXd coordinate = data.coordinates.row(0).transpose();
    const Eigen::VectorXd composition = uac::ilr_inverse(
        coordinate.transpose(), helmert).row(0).transpose();
    const Eigen::VectorXd probability = basis.probabilities * composition;
    Document expected;
    expected.raw_ct_tot = expected.ct_tot = 200.0;
    for (int32_t feature = 0; feature < probability.size(); ++feature) {
        expected.ids.push_back(feature);
        expected.cnts.push_back(200.0 * probability(feature));
    }
    const uac::FisherApproximation expected_exact = uac::fisher_approximation(
        coordinate, expected, basis, helmert, uac::ProposalKind::ExactFisher);
    const uac::FisherApproximation expected_sparse = uac::fisher_approximation(
        coordinate, expected, basis, helmert,
        uac::ProposalKind::SparseEmpiricalFisher);
    require((expected_exact.gradient - expected_sparse.gradient).norm()
            < 1e-10
        && (expected_exact.information - expected_sparse.information).norm()
            < 1e-9,
        "exact and empirical Fisher disagree at expected counts");

    uac::Dataset half = data;
    half.raw_totals *= 0.5;
    half.effective_totals *= 0.5;
    for (auto& document : half.counts) {
        for (double& count : document.cnts) count *= 0.5;
    }
    for (const uac::ProposalKind proposal : {
            uac::ProposalKind::ExactFisher,
            uac::ProposalKind::SparseEmpiricalFisher}) {
        const uac::FisherApproximation full = uac::fisher_approximation(
            coordinate, data.counts[0], basis, helmert, proposal);
        const uac::FisherApproximation scaled = uac::fisher_approximation(
            coordinate, half.counts[0], basis, helmert, proposal);
        require((scaled.gradient - 0.5 * full.gradient).norm() < 1e-10
            && (scaled.information - 0.5 * full.information).norm() < 1e-9,
            "fractional counts do not scale Fisher curvature linearly");
    }
    const uac::ParticleSet half_exact = uac::make_particles(half, basis,
        helmert, pilot, uac::ProposalKind::ExactFisher, 8, 733, 1.5, 1);
    require(half_exact.values.allFinite()
        && half_exact.log_proposal.allFinite()
        && half_exact.log_likelihood.allFinite(),
        "fractional counts produced nonfinite exact-Fisher particles");
}

void test_rare_and_inactive_components() {
    RareComponentFixture fixture = rare_component_fixture();
    constexpr int32_t components = 3;
    constexpr int32_t topics = 12;
    uac::FitOptions options;
    options.n_components = components;
    options.n_particles = 32;
    options.kmeans_starts = 1;
    options.max_iterations = 60;
    options.seed = 101;
    options.n_threads = 2;
    const uac::FitResult fitted = uac::fit(fixture.data, &fixture.basis,
        options);
    const double smallest_pilot_membership = fitted.pilot.weights.minCoeff()
        * fixture.data.centers.rows();
    require(smallest_pilot_membership > 0.0
        && smallest_pilot_membership < topics,
        "winning MAP pilot did not retain a component smaller than the topic dimension");
    for (const auto& covariance : fitted.pilot.covariances) {
        require(Eigen::LLT<Eigen::MatrixXd>(covariance).info()
            == Eigen::Success,
            "winning MAP proposal covariance is not positive definite");
    }
    require((fitted.model.weights.array() > 0.0).all(),
        "rare positive-membership component was made inactive");
    require(fitted.model.weights.minCoeff() * fixture.data.centers.rows()
            < topics,
        "fit did not retain an effective component with N_c < K");
    for (const auto& trace : fitted.traces) {
        require(trace.active_components.size() == trace.objective.size()
            && std::all_of(trace.active_components.begin(),
                trace.active_components.end(), [](int32_t active) {
                    return active == components;
                }),
            "active-component trace lost the rare component");
    }

    uac::FitOptions overspecified_options = options;
    overspecified_options.n_components = 6;
    overspecified_options.n_particles = 16;
    overspecified_options.max_iterations = 40;
    const uac::FitResult overspecified = uac::fit(fixture.data,
        &fixture.basis, overspecified_options);
    require(overspecified.model.weights.size()
            == overspecified_options.n_components
        && (overspecified.model.weights.array() >= 0.0).all()
        && std::abs(overspecified.model.weights.sum() - 1.0) < 1e-12,
        "overspecified fit did not retain valid component slots");
    for (const auto& trace : overspecified.traces) {
        require(!trace.collapsed,
            "overspecified fit rejected a restart because of small components");
    }

    uac::Model inactive = fitted.model;
    inactive.weights(0) = 0.0;
    inactive.weights /= inactive.weights.sum();
    inactive.covariances[0] = inactive.shrinkage_target;
    const uac::ScoreResult score = uac::score_map(fixture.data, inactive, 2);
    require(score.responsibilities.col(0).isZero(0.0)
        && (score.responsibilities.rowwise().sum().array() - 1.0)
            .abs().maxCoeff() < 1e-12,
        "inactive component received responsibility");

    uac::State state = uac::make_state(fitted, &fixture.basis, options,
        Eigen::VectorXd::Ones(fixture.basis.probabilities.rows()), false);
    state.model = inactive;
    const uac::ScoreResult particle_score = uac::score_particle(fixture.data,
        fixture.basis, state, state.proposal, 16, 2);
    require(particle_score.responsibilities.col(0).isZero(0.0)
        && (particle_score.responsibilities.rowwise().sum().array() - 1.0)
            .abs().maxCoeff() < 1e-12,
        "inactive component received particle responsibility");
    const std::filesystem::path prefix =
        std::filesystem::temp_directory_path() / "punkst_uac_inactive_test";
    const std::filesystem::path state_path = prefix.string() + ".state.tsv";
    const std::filesystem::path model_path = prefix.string() + ".model.tsv";
    const std::filesystem::path separation_path =
        prefix.string() + ".separation.tsv";
    const std::filesystem::path representatives_path =
        prefix.string() + ".representatives.tsv";
    const std::filesystem::path trace_path = prefix.string() + ".trace.tsv";
    uac::write_state(state_path.string(), state);
    const uac::State restored = uac::read_state(state_path.string());
    require(restored.model.weights(0) == 0.0
        && std::abs(restored.model.weights.sum() - 1.0) < 1e-12,
        "state round trip lost an inactive component");
    uac::State all_inactive = state;
    all_inactive.model.weights.setZero();
    uac::write_state(state_path.string(), all_inactive);
    bool rejected_all_inactive = false;
    try {
        static_cast<void>(uac::read_state(state_path.string()));
    } catch (const std::runtime_error&) {
        rejected_all_inactive = true;
    }
    require(rejected_all_inactive,
        "state reader accepted a model with no active components");
    uac::write_state(state_path.string(), state);
    uac::write_model(model_path.string(), state);
    uac::write_separation(separation_path.string(), state.model);
    uac::write_representatives(representatives_path.string(), fixture.data,
        score, 2);
    uac::RestartTrace trace;
    trace.objective = {-10.0, -9.0};
    trace.active_components = {3, 2};
    uac::write_trace(trace_path.string(), {trace});
    const std::string model_text = read_text(model_path);
    const std::string separation_text = read_text(separation_path);
    const std::string representatives_text = read_text(representatives_path);
    const std::string trace_text = read_text(trace_path);
    require(model_text.find("#cluster\tactive\tweight") == 0
        && model_text.find("\n0\t0\t") != std::string::npos,
        "model summary did not mark the inactive slot");
    require(separation_text.find("\n0\t") == std::string::npos
        && separation_text.find("\t0\t") == std::string::npos,
        "separation summary included the inactive slot");
    require(representatives_text.find("\n0\t") == std::string::npos,
        "representative summary included the inactive slot");
    require(trace_text.find("\tactive_components\t") != std::string::npos
        && trace_text.find("\t2\t0\t0\n") != std::string::npos,
        "trace did not report active component counts");
    for (const auto& path : {state_path, model_path, separation_path,
            representatives_path, trace_path}) {
        std::filesystem::remove(path);
    }
}

void test_mixed_map_starts() {
    require(std::abs(uac::detail::increased_leiden_resolution(
                1.0, 1, 100) - 2.0) < 1e-12,
        "adaptive Leiden increase was not capped at two-fold");
    require(std::abs(uac::detail::increased_leiden_resolution(
                1.0, 9, 10) - 1.25) < 1e-12,
        "adaptive Leiden increase did not retain its minimum step");
    require(std::abs(uac::detail::midpoint_leiden_resolution(2.0, 4.0)
                - 3.0) < 1e-12,
        "adaptive Leiden overshoot did not use the bracket midpoint");

    const uac::Basis basis = test_basis();
    const uac::Dataset data = test_dataset();
    uac::FitOptions options;
    options.n_components = 3;
    options.n_particles = 16;
    options.kmeans_starts = 2;
    options.leiden_starts = 2;
    options.leiden_neighbors = 8;
    options.max_iterations = 50;
    options.seed = 313;
    options.n_threads = 2;
    const uac::FitResult fitted = uac::fit(data, &basis, options);
    uac::FitOptions ragged_options = options;
    ragged_options.iteration_callback = {};
    ragged_options.adaptive_particles.enabled = true;
    ragged_options.adaptive_particles.calibration_particles = 8;
    ragged_options.adaptive_particles.minimum_particles = options.n_particles;
    ragged_options.adaptive_particles.maximum_particles = options.n_particles;
    const uac::FitResult ragged = uac::fit(data, &basis, ragged_options);
    require(ragged.score.responsibilities.allFinite()
            && (ragged.score.responsibilities.rowwise().sum().array() - 1.0)
                .abs().maxCoeff() < 1e-10
            && ragged.score.particle_samples
                == static_cast<int64_t>(data.identifiers.size())
                    * options.n_particles
            && ragged.score.calibration_samples
                == static_cast<int64_t>(data.identifiers.size()) * 8
            && ragged.score.reused_calibration_samples
                == ragged.score.calibration_samples
            && ragged.score.adaptive_particle_diagnostics.size()
                == data.identifiers.size()
            && std::all_of(ragged.score.per_document_particles.begin(),
                ragged.score.per_document_particles.end(),
                [&](int32_t value) { return value == options.n_particles; }),
        "ragged particle path did not reuse its calibration prefix");
    for (const uac::AdaptiveParticleRule rule : {
            uac::AdaptiveParticleRule::ResponsibilityOnly,
            uac::AdaptiveParticleRule::MomentOnly,
            uac::AdaptiveParticleRule::ResponsibilityMoment}) {
        ragged_options.adaptive_particles.rule = rule;
        const uac::FitResult adaptive = uac::fit(
            data, &basis, ragged_options);
        require(adaptive.score.responsibilities.allFinite()
                && adaptive.score.adaptive_particle_diagnostics.size()
                    == data.identifiers.size()
                && adaptive.score.reused_calibration_samples
                    == static_cast<int64_t>(data.identifiers.size()) * 8
                && std::all_of(
                    adaptive.score.adaptive_particle_diagnostics.begin(),
                    adaptive.score.adaptive_particle_diagnostics.end(),
                    [&](const uac::AdaptiveParticleDiagnostic& diagnostic) {
                        return diagnostic.selected_particles
                                == options.n_particles
                            && diagnostic.plausible_components >= 1
                            && std::isfinite(
                                diagnostic.maximum_responsibility_se);
                    }),
            "responsibility-aware adaptive diagnostics are invalid");
    }
    int32_t map_traces = 0, particle_traces = 0;
    double best_map_objective = -std::numeric_limits<double>::infinity();
    double selected_map_objective = -std::numeric_limits<double>::infinity();
    std::vector<int32_t> seeds;
    for (const auto& trace : fitted.traces) {
        if (trace.particle) {
            ++particle_traces;
            require(trace.start == fitted.selected_start
                    && trace.initializer == fitted.selected_initializer,
                "particle EM did not inherit the selected MAP metadata");
        } else {
            ++map_traces;
            if (!trace.collapsed) {
                best_map_objective = std::max(
                    best_map_objective, trace.selection_objective);
                if (trace.start == fitted.selected_start) {
                    selected_map_objective = trace.selection_objective;
                }
            }
            seeds.push_back(trace.seed);
            if (trace.start < options.kmeans_starts) {
                require(trace.initializer == uac::MapInitializer::KMeans,
                    "k-means MAP start has the wrong initializer tag");
            } else {
                require(trace.initializer == uac::MapInitializer::Leiden
                        && trace.raw_communities > 0
                        && trace.reconciliation_count == std::abs(
                            trace.raw_communities - options.n_components),
                    "Leiden MAP metadata or reconciliation count is invalid");
            }
        }
    }
    std::sort(seeds.begin(), seeds.end());
    require(map_traces == options.kmeans_starts + options.leiden_starts
            && particle_traces == 1
            && std::adjacent_find(seeds.begin(), seeds.end()) == seeds.end()
            && selected_map_objective == best_map_objective,
        "mixed starts did not produce distinct MAP seeds and one particle fit");

    uac::FitOptions adaptive = options;
    adaptive.handoff = uac::HandoffMode::Map;
    adaptive.n_components = 6;
    adaptive.kmeans_starts = 0;
    adaptive.leiden_starts = 2;
    adaptive.leiden_resolution = 1e-8;
    const uac::FitResult adaptive_fit = uac::fit(data, nullptr, adaptive);
    require(adaptive_fit.traces.size() == 2
            && adaptive_fit.traces[0].raw_communities
                < adaptive.n_components,
        "low-resolution Leiden fixture was not under-resolved");
    const double expected = uac::detail::increased_leiden_resolution(
        adaptive_fit.traces[0].leiden_resolution,
        adaptive_fit.traces[0].raw_communities, adaptive.n_components);
    require(std::abs(adaptive_fit.traces[1].leiden_resolution - expected)
            < 1e-15,
        "Leiden fit did not apply the bounded adaptive schedule");
    const uac::State adaptive_state = uac::make_state(adaptive_fit, nullptr,
        adaptive, Eigen::VectorXd(), false);
    const std::filesystem::path adaptive_path =
        std::filesystem::temp_directory_path()
        / "punkst_uac_leiden_state.tsv";
    uac::write_state(adaptive_path.string(), adaptive_state);
    const uac::State restored_adaptive = uac::read_state(
        adaptive_path.string());
    std::filesystem::remove(adaptive_path);
    require(restored_adaptive.selected_initializer
            == uac::MapInitializer::Leiden
            && restored_adaptive.kmeans_starts == 0
            && restored_adaptive.leiden_starts == adaptive.leiden_starts
            && restored_adaptive.selected_leiden_resolution > 0.0,
        "Leiden-only initialization metadata did not round trip");

    uac::FitOptions map = options;
    map.handoff = uac::HandoffMode::Map;
    map.leiden_starts = 0;
    const uac::FitResult map_fit = uac::fit(data, nullptr, map);
    require((map_fit.pilot.weights - map_fit.model.weights).norm() < 1e-12
            && (map_fit.pilot.means - map_fit.model.means).norm() < 1e-12
            && (map_fit.pilot.pooled_covariance
                - map_fit.model.shrinkage_target).norm() < 1e-12,
        "winning MAP fit did not define the proposal and final target");

    uac::FitOptions defaults;
    defaults.handoff = uac::HandoffMode::Map;
    defaults.n_components = 3;
    defaults.max_iterations = 20;
    const uac::FitResult default_fit = uac::fit(data, nullptr, defaults);
    require(default_fit.traces.size() == 5,
        "default initialization did not run five k-means MAP starts");
}

void test_fit_score_and_state(const std::string& requested_output) {
    const uac::Basis basis = test_basis();
    const uac::Dataset data = test_dataset();
    uac::FitOptions options;
    options.handoff = uac::HandoffMode::Particle;
    options.proposal = uac::ProposalKind::ExactFisher;
    options.n_components = 3;
    options.n_particles = 48;
    options.kmeans_starts = 2;
    options.max_iterations = 80;
    options.seed = 71;
    options.n_threads = 2;
    int32_t iteration_notices = 0;
    int32_t finite_convergence_notices = 0;
    options.iteration_callback = [&](const uac::IterationDiagnostic& value) {
        ++iteration_notices;
        if (std::isfinite(value.relative_objective_change)
            && std::isfinite(value.mean_max_responsibility_change)) {
            ++finite_convergence_notices;
        }
    };
    const uac::FitResult fitted = uac::fit(data, &basis, options);
    for (const auto& trace : fitted.traces) {
        require(trace.relative_objective_change.size()
                == trace.objective.size()
            && trace.mean_max_responsibility_change.size()
                == trace.objective.size(),
            "convergence diagnostic trace lengths differ");
        for (size_t iteration = 1; iteration < trace.objective.size(); ++iteration) {
            require(trace.objective[iteration] + 1e-7
                >= trace.objective[iteration - 1],
                "penalized EM objective decreased");
        }
    }
    require(iteration_notices > 0 && finite_convergence_notices > 0,
        "iteration callback did not receive convergence changes");
    require(fitted.score.responsibilities.allFinite(),
        "particle responsibilities are nonfinite");
    require((fitted.score.responsibilities.rowwise().sum().array() - 1.0)
        .abs().maxCoeff() < 1e-10,
        "particle responsibilities are not normalized");
    require(fitted.score.particle_diagnostics.size()
            == data.identifiers.size()
        && fitted.score.particle_bytes == sizeof(double)
            * static_cast<uint64_t>(data.identifiers.size())
            * options.n_particles * (data.coordinates.cols() + 2),
        "particle diagnostics or memory accounting are incomplete");
    require(fitted.score.gaussian_seconds > 0.0
            && fitted.score.moment_seconds > 0.0
            && fitted.score.expectation_accumulator_bytes > 0
            && fitted.score.proposal_workspace_bytes > 0,
        "particle E-step timing or workspace diagnostics are missing");
    for (const auto& diagnostic : fitted.score.particle_diagnostics) {
        require(diagnostic.relative_ess > 0.0
            && diagnostic.relative_ess <= 1.0 + 1e-12
            && diagnostic.maximum_weight > 0.0
            && diagnostic.maximum_weight <= 1.0 + 1e-12,
            "particle ESS diagnostics are outside their valid range");
    }
    for (const auto& covariance : fitted.model.covariances) {
        require(Eigen::LLT<Eigen::MatrixXd>(covariance).info()
            == Eigen::Success, "fitted covariance is not positive definite");
    }
    const Eigen::VectorXd weights = Eigen::VectorXd::Ones(
        basis.probabilities.rows());
    const uac::State state = uac::make_state(fitted, &basis, options,
        weights, true);
    const uac::ScoreResult rescored = uac::score_particle(data, basis, state,
        state.proposal, state.n_particles, 1);
    require((rescored.responsibilities - fitted.score.responsibilities)
        .cwiseAbs().maxCoeff() < 1e-11,
        "fit and transform particle scores differ");
    for (int32_t block_size : {1, 5,
            static_cast<int32_t>(data.identifiers.size())}) {
        const uac::ScoreResult replayed = uac::score_particle(data, basis,
            state, state.proposal, state.n_particles, 2, block_size);
        require((replayed.responsibilities - rescored.responsibilities)
                .cwiseAbs().maxCoeff() < 1e-11
            && replayed.particle_replay
            && replayed.particle_block_size == block_size
            && replayed.particle_bytes <= fitted.score.particle_bytes,
            "blockwise particle replay changed transform inference");
    }

    uac::FitOptions replay_options = options;
    replay_options.particle_block_size = 5;
    const uac::FitResult replay_fit = uac::fit(data, &basis, replay_options);
    require((replay_fit.model.means - fitted.model.means)
            .cwiseAbs().maxCoeff() < 1e-9
        && (replay_fit.score.responsibilities
            - fitted.score.responsibilities).cwiseAbs().maxCoeff() < 1e-9
        && replay_fit.score.particle_replay
        && replay_fit.score.particle_generation_passes > 1,
        "blockwise particle fitting changed deterministic EM results");

    std::filesystem::path state_path = requested_output.empty()
        ? std::filesystem::temp_directory_path() / "punkst_uac_test_state.tsv"
        : std::filesystem::path(requested_output);
    uac::State state_to_write = state;
    state_to_write.fit_adaptive_particles.enabled = true;
    state_to_write.fit_adaptive_particles.rule =
        uac::AdaptiveParticleRule::ResponsibilityMoment;
    state_to_write.fit_adaptive_particles.calibration_particles = 8;
    state_to_write.fit_adaptive_particles.minimum_particles = 16;
    state_to_write.fit_adaptive_particles.maximum_particles =
        state_to_write.n_particles;
    state_to_write.fit_adaptive_particles.responsibility_se_target = 0.08;
    state_to_write.fit_adaptive_particles.moment_ess_target = 12.0;
    uac::write_state(state_path.string(), state_to_write);
    const uac::State restored = uac::read_state(state_path.string());
    require(restored.basis_checksum == state_to_write.basis_checksum
        && restored.weighted_counts
        && restored.selected_start == state.selected_start
        && restored.selected_initializer == state.selected_initializer
        && restored.kmeans_starts == state.kmeans_starts
        && restored.leiden_starts == state.leiden_starts
        && restored.leiden_knn_backend == state.leiden_knn_backend
        && restored.converged == state.converged
        && restored.adaptive_covariance_shrinkage
            == state_to_write.adaptive_covariance_shrinkage
        && restored.fit_adaptive_particles.enabled
        && restored.fit_adaptive_particles.rule
            == uac::AdaptiveParticleRule::ResponsibilityMoment
        && restored.fit_adaptive_particles.calibration_particles == 8
        && restored.fit_adaptive_particles.minimum_particles == 16
        && restored.fit_adaptive_particles.responsibility_se_target == 0.08
        && restored.fit_adaptive_particles.moment_ess_target == 12.0
        && restored.objective_change_tolerance
            == state_to_write.objective_change_tolerance
        && restored.responsibility_change_tolerance
            == state_to_write.responsibility_change_tolerance
        && restored.topics == state_to_write.topics
        && (restored.model.means - state_to_write.model.means).norm() < 1e-12
        && (restored.pilot.means - state_to_write.pilot.means).norm() < 1e-12
        && (restored.pilot.pooled_covariance
            - state_to_write.pilot.pooled_covariance).norm() < 1e-12,
        "UAC state round trip differs");
    const uac::ScoreResult restored_score = uac::score_particle(data, basis,
        restored, restored.proposal, restored.n_particles, 2);
    require((restored_score.responsibilities
            - fitted.score.responsibilities).cwiseAbs().maxCoeff() < 1e-11,
        "restored mixed-start state changed particle transform scores");
    uac::AdaptiveParticleOptions transform_adaptive;
    transform_adaptive.enabled = true;
    transform_adaptive.rule =
        uac::AdaptiveParticleRule::ResponsibilityOnly;
    transform_adaptive.calibration_particles = 8;
    transform_adaptive.minimum_particles = 8;
    transform_adaptive.maximum_particles = restored.n_particles;
    transform_adaptive.responsibility_se_target = 0.2;
    const uac::ScoreResult adaptive_transform = uac::score_particle(data,
        basis, restored, restored.proposal, restored.n_particles,
        transform_adaptive, 2);
    require(adaptive_transform.responsibilities.allFinite()
            && adaptive_transform.adaptive_particle_options.enabled
            && adaptive_transform.reused_calibration_samples
                == static_cast<int64_t>(data.identifiers.size()) * 8
            && adaptive_transform.adaptive_particle_diagnostics.size()
                == data.identifiers.size(),
        "adaptive particle transform is invalid");

    const uac::ScoreResult sparse_score = uac::score_particle(data, basis,
        state, uac::ProposalKind::SparseEmpiricalFisher,
        state.n_particles, 2);
    require(sparse_score.responsibilities.allFinite()
        && (sparse_score.responsibilities.rowwise().sum().array() - 1.0)
            .abs().maxCoeff() < 1e-10,
        "sparse empirical Fisher scoring is invalid");

    uac::FitOptions factor_options = options;
    factor_options.cluster_covariance_rank = 1;
    factor_options.n_particles = 32;
    factor_options.particle_block_size = 5;
    const uac::FitResult factor_fit = uac::fit(
        data, &basis, factor_options);
    require(factor_fit.model.covariance_kind
            == uac::CovarianceKind::FactorAnalytic
        && factor_fit.model.factor_covariances.size()
            == static_cast<size_t>(factor_options.n_components)
        && factor_fit.score.responsibilities.allFinite()
        && (factor_fit.score.responsibilities.rowwise().sum().array() - 1.0)
            .abs().maxCoeff() < 1e-10,
        "factor-analytic particle fit is invalid");
    for (const auto& covariance : factor_fit.model.factor_covariances) {
        require(covariance.factor.cols()
                == factor_options.cluster_covariance_rank
            && (covariance.diagonal.array() > 0.0).all()
            && Eigen::LLT<Eigen::MatrixXd>(covariance.dense()).info()
                == Eigen::Success,
            "factor-analytic covariance is not positive definite");
    }
    const uac::State factor_state = uac::make_state(factor_fit, &basis,
        factor_options, weights, false);
    const std::filesystem::path factor_path =
        std::filesystem::temp_directory_path() / "punkst_uac_factor_state.tsv";
    uac::write_state(factor_path.string(), factor_state);
    const uac::State restored_factor = uac::read_state(factor_path.string());
    const uac::ScoreResult restored_factor_score = uac::score_particle(data,
        basis, restored_factor, restored_factor.proposal,
        restored_factor.n_particles, 2, 5);
    require(restored_factor.cluster_covariance_rank
            == factor_options.cluster_covariance_rank
        && (restored_factor_score.responsibilities
            - factor_fit.score.responsibilities).cwiseAbs().maxCoeff() < 1e-10,
        "factor-analytic state round trip changed inference");
    std::filesystem::remove(factor_path);

    uac::FitOptions no_shrinkage_options = options;
    no_shrinkage_options.adaptive_covariance_shrinkage = false;
    no_shrinkage_options.kmeans_starts = 1;
    const uac::FitResult no_shrinkage_fit = uac::fit(
        data, &basis, no_shrinkage_options);
    const uac::State no_shrinkage_state = uac::make_state(
        no_shrinkage_fit, &basis, no_shrinkage_options, weights, false);
    require(!no_shrinkage_state.adaptive_covariance_shrinkage
            && no_shrinkage_fit.score.responsibilities.allFinite(),
        "no-covariance-shrinkage mode is invalid");

    uac::FitOptions diagonal_options = factor_options;
    diagonal_options.handoff = uac::HandoffMode::Map;
    diagonal_options.cluster_covariance_rank = 0;
    diagonal_options.particle_block_size = 0;
    const uac::FitResult diagonal_fit = uac::fit(
        data, nullptr, diagonal_options);
    require(std::all_of(diagonal_fit.model.factor_covariances.begin(),
            diagonal_fit.model.factor_covariances.end(),
            [](const LowRankDiagonalCovariance& covariance) {
                return covariance.factor.cols() == 0
                    && (covariance.diagonal.array() > 0.0).all();
            }),
        "diagonal factor-analytic mode is invalid");
    const std::filesystem::path legacy_path =
        std::filesystem::temp_directory_path() / "punkst_uac_v1_state.tsv";
    {
        std::ofstream legacy(legacy_path);
        legacy << "##punkst_uac_state_v1\n"
            << "##handoff\tparticle\n"
            << "##proposal\tmixture\n"
            << "##fisher_broadening\t1.5\n"
            << "##components\t3\n##dimension\t2\n";
    }
    bool rejected_v1 = false;
    try {
        static_cast<void>(uac::read_state(legacy_path.string()));
    } catch (const std::exception&) {
        rejected_v1 = true;
    }
    std::filesystem::remove(legacy_path);
    require(rejected_v1, "stale UAC state v1 was not rejected");
    const std::filesystem::path v3_path =
        std::filesystem::temp_directory_path() / "punkst_uac_v3_state.tsv";
    {
        std::ofstream stale(v3_path);
        stale << "##punkst_uac_state_v3\n";
    }
    bool rejected_v3 = false;
    try {
        static_cast<void>(uac::read_state(v3_path.string()));
    } catch (const std::exception&) {
        rejected_v3 = true;
    }
    std::filesystem::remove(v3_path);
    require(rejected_v3, "stale UAC state v3 was not rejected");
    if (requested_output.empty()) std::filesystem::remove(state_path);

    uac::FitOptions map_options = options;
    map_options.handoff = uac::HandoffMode::Map;
    const uac::FitResult map_fit = uac::fit(data, nullptr, map_options);
    const uac::ScoreResult map_score = uac::score_map(data, map_fit.model);
    require((map_score.responsibilities - map_fit.score.responsibilities)
        .cwiseAbs().maxCoeff() < 1e-12,
        "MAP fit and score differ");
}

} // namespace

int32_t test(int32_t argc, char** argv) {
    std::string suite = "fast";
    std::string output;
    int32_t documents = 6000, particles = 256, seed = 20260719;
    int32_t topics = 3, features = 60, components = 3;
    int32_t calibration_particles = 0, minimum_particles = 64;
    int32_t audit_documents = 0, audit_particles = 2048;
    int32_t threads = 4;
    double ess_target = 32.0, contrast_se_target = 0.2;
    double maximum_weight_target = 0.1;
    double point_center_error_target = 0.12;
    double responsibility_se_target = 0.05;
    double plausible_mass = 0.95, plausible_responsibility = 0.05;
    double moment_ess_target = 16.0;
    double simulation_mean_scale = -1.0;
    std::string proposal_name = "exact_fisher";
    std::string adaptive_rule_name = "legacy";
    std::string length_profile = "fixed";
    std::string topic_identifiability = "homogeneous";
    ParamList pl;
    pl.add_option("suite", "Test suite: fast, uac, or uac-profile", suite)
      .add_option("out", "Optional test output path", output)
      .add_option("documents", "Profile document count", documents)
      .add_option("particles", "Profile particles per document", particles)
      .add_option("seed", "Profile simulation and fit seed", seed)
      .add_option("threads", "Profile TBB worker count", threads)
      .add_option("topics", "Profile topic count", topics)
      .add_option("features", "Profile feature count", features)
      .add_option("components", "Profile cluster count", components)
      .add_option("point-center-error-target",
          "Profile point-center oracle error calibration target",
          point_center_error_target)
      .add_option("length-profile",
          "Profile document lengths: fixed or stratified", length_profile)
      .add_option("topic-identifiability",
          "Profile topic identifiability: homogeneous or mixed",
          topic_identifiability)
      .add_option("simulation-mean-scale",
          "Positive pre-calibrated cluster mean scale; negative recalibrates",
          simulation_mean_scale)
      .add_option("audit-documents",
          "Number of deterministic documents in the nested particle audit",
          audit_documents)
      .add_option("audit-particles",
          "Reference particle count for the nested audit", audit_particles)
      .add_option("calibration-particles",
          "Independent adaptive calibration particles; 0 disables adaptation",
          calibration_particles)
      .add_option("minimum-particles", "Adaptive minimum retained particles",
          minimum_particles)
      .add_option("ess-target", "Adaptive material-component ESS target",
          ess_target)
      .add_option("contrast-se-target",
          "Adaptive top-two log-evidence contrast SE target",
          contrast_se_target)
      .add_option("maximum-weight-target",
          "Adaptive maximum normalized component weight target",
          maximum_weight_target)
      .add_option("adaptive-rule",
          "Adaptive rule: legacy, responsibility_only, moment_only, or responsibility_moment",
          adaptive_rule_name)
      .add_option("responsibility-se-target",
          "Adaptive absolute responsibility Monte Carlo SE target",
          responsibility_se_target)
      .add_option("plausible-mass",
          "Adaptive moment-guard cumulative responsibility mass",
          plausible_mass)
      .add_option("plausible-responsibility",
          "Adaptive moment-guard individual responsibility threshold",
          plausible_responsibility)
      .add_option("moment-ess-target",
          "Adaptive plausible-component conditional ESS target",
          moment_ess_target)
      .add_option("proposal",
          "Profile proposal: exact_fisher or sparse_empirical_fisher",
          proposal_name);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        if (suite != "fast" && suite != "uac" && suite != "uac-profile") {
            throw std::invalid_argument(
                "--suite must be fast, uac, or uac-profile");
        }
        if (suite == "uac-profile") {
            uac::AdaptiveParticleOptions adaptive;
            adaptive.enabled = calibration_particles > 0;
            adaptive.rule = uac::parse_adaptive_particle_rule(
                adaptive_rule_name);
            adaptive.calibration_particles = calibration_particles > 0
                ? calibration_particles : 32;
            adaptive.minimum_particles = minimum_particles;
            adaptive.maximum_particles = particles;
            adaptive.component_ess_target = ess_target;
            adaptive.contrast_se_target = contrast_se_target;
            adaptive.maximum_weight_target = maximum_weight_target;
            adaptive.responsibility_se_target = responsibility_se_target;
            adaptive.plausible_mass = plausible_mass;
            adaptive.plausible_responsibility = plausible_responsibility;
            adaptive.moment_ess_target = moment_ess_target;
            run_uac_profile(documents, particles, seed, threads,
                topics, features, components, point_center_error_target,
                length_profile, topic_identifiability, simulation_mean_scale,
                audit_documents, audit_particles,
                uac::parse_proposal(proposal_name), adaptive, output);
            return 0;
        }
        notice("Running UAC transform tests");
        test_transforms();
        notice("Running heterogeneous UAC simulation tests");
        test_profile_heterogeneous_simulation();
        notice("Running UAC proposal tests");
        test_fractional_likelihood_and_proposals();
        notice("Running UAC rare/inactive component tests");
        test_rare_and_inactive_components();
        notice("Running UAC mixed MAP-start tests");
        test_mixed_map_starts();
        notice("Running UAC fit/state tests");
        test_fit_score_and_state(output);
        notice("Native UAC tests passed");
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
        return 1;
    }
    return 0;
}
