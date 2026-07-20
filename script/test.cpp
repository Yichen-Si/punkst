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

struct ProfileSimulation {
    uac::Dataset data;
    uac::Basis basis;
    Eigen::VectorXi labels;
    RowMajorMatrixXd true_means;
    std::vector<Eigen::MatrixXd> true_covariances;
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
    return out;
}

std::array<int32_t, 3> align_profile_components(
    const uac::Model& model, const RowMajorMatrixXd& truth) {
    std::array<int32_t, 3> permutation{{0, 1, 2}}, best = permutation;
    double best_error = std::numeric_limits<double>::infinity();
    do {
        double error = 0.0;
        for (int32_t fitted = 0; fitted < 3; ++fitted) {
            error += (model.means.row(fitted)
                - truth.row(permutation[fitted])).squaredNorm();
        }
        if (error < best_error) {
            best_error = error;
            best = permutation;
        }
    } while (std::next_permutation(permutation.begin(), permutation.end()));
    return best;
}

void run_uac_profile(int32_t documents, int32_t particles, int32_t seed,
    int32_t threads, uac::ProposalKind proposal, const std::string& output) {
    ProfileSimulation simulation = profile_simulation(documents, seed);
    uac::FitOptions options;
    options.n_components = 3;
    options.n_particles = particles;
    options.restarts = 1;
    options.max_iterations = 60;
    options.kmeans_max_iterations = 100;
    options.seed = seed;
    options.n_threads = threads;
    options.proposal = proposal;
    const auto begin = std::chrono::steady_clock::now();
    const uac::FitResult fit = uac::fit(simulation.data, &simulation.basis,
        options);
    const double wall_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - begin).count();
    const auto alignment = align_profile_components(fit.model,
        simulation.true_means);
    std::array<int32_t, 3> fitted_for_truth;
    for (int32_t fitted = 0; fitted < 3; ++fitted) {
        fitted_for_truth[alignment[fitted]] = fitted;
    }
    double log_loss = 0.0, true_probability = 0.0, entropy = 0.0;
    for (int32_t document = 0; document < documents; ++document) {
        const double probability = fit.score.responsibilities(document,
            fitted_for_truth[simulation.labels(document)]);
        true_probability += probability;
        log_loss -= std::log(std::max(1e-300, probability));
        for (int32_t component = 0; component < 3; ++component) {
            const double p = fit.score.responsibilities(document, component);
            if (p > 0.0) entropy -= p * std::log(p);
        }
    }
    log_loss /= documents;
    true_probability /= documents;
    entropy /= documents;
    double covariance_error = 0.0, diagonal_error = 0.0;
    for (int32_t fitted = 0; fitted < 3; ++fitted) {
        const int32_t truth = alignment[fitted];
        covariance_error += (fit.model.covariances[fitted]
            - simulation.true_covariances[truth]).norm()
            / simulation.true_covariances[truth].norm();
        diagonal_error += (fit.model.covariances[fitted].diagonal()
            - simulation.true_covariances[truth].diagonal()).sum()
            / simulation.true_covariances[truth].diagonal().sum();
    }
    covariance_error /= 3.0;
    diagonal_error /= 3.0;
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
        responsibility << "document\tlabel\tp0\tp1\tp2\n"
            << std::setprecision(12);
        for (int32_t document = 0; document < documents; ++document) {
            responsibility << document << "\t" << simulation.labels(document);
            for (int32_t truth = 0; truth < 3; ++truth) {
                responsibility << "\t" << fit.score.responsibilities(
                    document, fitted_for_truth[truth]);
            }
            responsibility << "\n";
        }
    }
    *stream << "documents\tparticles\tproposal\twall_seconds"
        "\tparticle_generation_seconds"
        "\tsampling_seconds\tlikelihood_seconds\tscoring_seconds"
        "\tparticle_em_objective_evaluations\tparticle_bytes"
        "\tmean_true_label_probability\tlog_loss"
        "\tmean_assignment_entropy\tcovariance_relative_error"
        "\tcovariance_diagonal_signed_relative_error\n"
        << std::setprecision(12) << documents << "\t" << particles << "\t"
        << uac::proposal_name(proposal) << "\t" << wall_seconds
        << "\t" << fit.score.particle_generation_seconds << "\t"
        << fit.score.sampling_seconds << "\t"
        << fit.score.likelihood_seconds << "\t" << fit.score.scoring_seconds
        << "\t" << fit.traces.back().objective.size() << "\t"
        << fit.score.particle_bytes << "\t" << true_probability
        << "\t" << log_loss << "\t" << entropy << "\t" << covariance_error
        << "\t" << diagonal_error << "\n";
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

void test_fractional_likelihood_and_proposals() {
    const uac::Basis basis = test_basis();
    const uac::Dataset data = test_dataset();
    const Eigen::MatrixXd helmert = uac::normalized_helmert(3);
    const uac::Pilot pilot = uac::make_pilot(data, 3, 19, 100, 0.1, 1e-4);
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

void test_fit_score_and_state(const std::string& requested_output) {
    const uac::Basis basis = test_basis();
    const uac::Dataset data = test_dataset();
    uac::FitOptions options;
    options.handoff = uac::HandoffMode::Particle;
    options.proposal = uac::ProposalKind::ExactFisher;
    options.n_components = 3;
    options.n_particles = 48;
    options.restarts = 2;
    options.max_iterations = 80;
    options.seed = 71;
    options.n_threads = 2;
    const uac::FitResult fitted = uac::fit(data, &basis, options);
    for (const auto& trace : fitted.traces) {
        for (size_t iteration = 1; iteration < trace.objective.size(); ++iteration) {
            require(trace.objective[iteration] + 1e-7
                >= trace.objective[iteration - 1],
                "penalized EM objective decreased");
        }
    }
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

    std::filesystem::path state_path = requested_output.empty()
        ? std::filesystem::temp_directory_path() / "punkst_uac_test_state.tsv"
        : std::filesystem::path(requested_output);
    uac::write_state(state_path.string(), state);
    const uac::State restored = uac::read_state(state_path.string());
    require(restored.basis_checksum == state.basis_checksum
        && restored.weighted_counts
        && restored.selected_restart == state.selected_restart
        && restored.converged == state.converged
        && restored.topics == state.topics
        && (restored.model.means - state.model.means).norm() < 1e-12,
        "UAC state round trip differs");

    const uac::ScoreResult sparse_score = uac::score_particle(data, basis,
        state, uac::ProposalKind::SparseEmpiricalFisher,
        state.n_particles, 2);
    require(sparse_score.responsibilities.allFinite()
        && (sparse_score.responsibilities.rowwise().sum().array() - 1.0)
            .abs().maxCoeff() < 1e-10,
        "sparse empirical Fisher scoring is invalid");
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
    int32_t threads = 4;
    std::string proposal_name = "exact_fisher";
    ParamList pl;
    pl.add_option("suite", "Test suite: fast, uac, or uac-profile", suite)
      .add_option("out", "Optional test output path", output)
      .add_option("documents", "Profile document count", documents)
      .add_option("particles", "Profile particles per document", particles)
      .add_option("seed", "Profile simulation and fit seed", seed)
      .add_option("threads", "Profile TBB worker count", threads)
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
            run_uac_profile(documents, particles, seed, threads,
                uac::parse_proposal(proposal_name), output);
            return 0;
        }
        notice("Running UAC transform tests");
        test_transforms();
        notice("Running UAC proposal tests");
        test_fractional_likelihood_and_proposals();
        notice("Running UAC fit/state tests");
        test_fit_score_and_state(output);
        notice("Native UAC tests passed");
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
        return 1;
    }
    return 0;
}
