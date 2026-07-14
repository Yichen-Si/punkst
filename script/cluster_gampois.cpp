#include "gamma_pois_cluster.hpp"
#include "gamma_pois_topic.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>

namespace {

Eigen::VectorXd cluster_topic_profile(
    const Eigen::Ref<const Eigen::MatrixXd>& basis,
    const Eigen::Ref<const Eigen::RowVectorXd>& cluster_mean) {
    Eigen::VectorXd logits = basis.transpose() * cluster_mean.transpose();
    logits.array() -= logits.maxCoeff();
    Eigen::VectorXd profile = logits.array().exp();
    profile /= profile.sum();
    return profile;
}

double responsibility_entropy(
    const Eigen::Ref<const Eigen::RowVectorXd>& responsibilities) {
    double out = 0.0;
    for (Eigen::Index c = 0; c < responsibilities.size(); ++c) {
        if (responsibilities(c) > 0.0) {
            out -= responsibilities(c) * std::log(responsibilities(c));
        }
    }
    return out;
}

void write_cluster_model(const std::string& path,
    const GammaPoissonClusterState& state,
    const GammaPoissonClusterFitResult& fit) {
    const auto& model = state.model;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open clustering model for writing: " + path);
    out << "#cluster\tactive\tweight\texpected_size\teffective_membership_size"
        << "\tmean_intrinsic_variance\tlog_intrinsic_volume";
    for (const auto& topic : state.topic_names) out << "\t" << topic;
    out << "\n" << std::scientific << std::setprecision(10);
    const Eigen::VectorXd effective_membership_size =
        gamma_poisson_effective_membership_size(fit.responsibilities);
    const Eigen::VectorXd weights = gamma_poisson_cluster_weights(model);
    for (Eigen::Index c = 0; c < weights.size(); ++c) {
        Eigen::VectorXd profile = cluster_topic_profile(state.basis, model.means.row(c));
        const double mean_variance = (model.variances.row(c).sum()
            + model.low_rank_variances.row(c).sum()) / model.variances.cols();
        out << c << "\t" << static_cast<int32_t>(state.active[c])
            << "\t" << weights(c)
            << "\t" << fit.effective_membership(c)
            << "\t" << effective_membership_size(c)
            << "\t" << mean_variance
            << "\t" << gamma_poisson_cluster_log_volume(model, c);
        for (Eigen::Index k = 0; k < profile.size(); ++k) out << "\t" << profile(k);
        out << "\n";
    }
}

void write_cluster_results(const std::string& path,
    const GammaPoissonClusterDataset& dataset,
    const Eigen::Ref<const RowMajorMatrixXd>& responsibilities) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open clustering results for writing: " + path);
    out << "#";
    for (size_t j = 0; j < dataset.posterior_header.identifier_columns.size(); ++j) {
        if (j > 0) out << "\t";
        out << dataset.posterior_header.identifier_columns[j];
    }
    if (!dataset.posterior_header.identifier_columns.empty()) out << "\t";
    out << "top_cluster\ttop_probability\tsecond_cluster\tsecond_probability\tentropy";
    for (Eigen::Index c = 0; c < responsibilities.cols(); ++c) out << "\tcluster_" << c;
    out << "\n" << std::scientific << std::setprecision(10);
    for (Eigen::Index d = 0; d < responsibilities.rows(); ++d) {
        if (!dataset.posterior_header.identifier_columns.empty()) {
            out << dataset.identifiers[d] << "\t";
        }
        Eigen::Index first = 0;
        Eigen::Index second = responsibilities.cols() > 1 ? 1 : 0;
        if (responsibilities(d, second) > responsibilities(d, first)) {
            std::swap(first, second);
        }
        for (Eigen::Index c = 2; c < responsibilities.cols(); ++c) {
            if (responsibilities(d, c) > responsibilities(d, first)) {
                second = first;
                first = c;
            } else if (responsibilities(d, c) > responsibilities(d, second)) {
                second = c;
            }
        }
        out << first << "\t" << responsibilities(d, first) << "\t"
            << second << "\t" << responsibilities(d, second) << "\t"
            << responsibility_entropy(responsibilities.row(d));
        for (Eigen::Index c = 0; c < responsibilities.cols(); ++c) {
            out << "\t" << responsibilities(d, c);
        }
        out << "\n";
    }
}

void write_cluster_separation(const std::string& path,
    const GammaPoissonClusterState& state) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open cluster separation output: " + path);
    out << "#cluster_a\tcluster_b\tstandardized_separation\tbhattacharyya_distance\n"
        << std::scientific << std::setprecision(10);
    for (int32_t first = 0; first < static_cast<int32_t>(state.active.size()); ++first) {
        if (!state.active[first]) continue;
        for (int32_t second = first + 1;
             second < static_cast<int32_t>(state.active.size()); ++second) {
            if (!state.active[second]) continue;
            const GammaPoissonClusterSeparation separation =
                gamma_poisson_cluster_separation(state.model, first, second);
            out << first << "\t" << second << "\t"
                << separation.standardized_distance << "\t"
                << separation.bhattacharyya_distance << "\n";
        }
    }
}

void write_cluster_representatives(const std::string& path,
    const GammaPoissonClusterDataset& dataset,
    const GammaPoissonClusterState& state,
    const GammaPoissonClusterFitResult& fit, int32_t n_representatives) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open representative-document output: " + path);
    out << "#cluster\trank";
    for (const auto& column : dataset.posterior_header.identifier_columns) {
        out << "\t" << column;
    }
    out << "\tprobability\ttop_probability\tentropy\n"
        << std::scientific << std::setprecision(10);
    std::vector<Eigen::Index> order(fit.responsibilities.rows());
    for (int32_t c = 0; c < static_cast<int32_t>(state.active.size()); ++c) {
        if (!state.active[c]) continue;
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](Eigen::Index first,
            Eigen::Index second) {
            return fit.responsibilities(first, c)
                > fit.responsibilities(second, c);
        });
        const int32_t take = std::min<int32_t>(n_representatives, order.size());
        for (int32_t rank = 0; rank < take; ++rank) {
            const Eigen::Index d = order[rank];
            out << c << "\t" << rank + 1;
            if (!dataset.posterior_header.identifier_columns.empty()) {
                out << "\t" << dataset.identifiers[d];
            }
            out << "\t" << fit.responsibilities(d, c)
                << "\t" << fit.responsibilities.row(d).maxCoeff()
                << "\t" << responsibility_entropy(
                    fit.responsibilities.row(d)) << "\n";
        }
    }
}

GammaPoissonClusterState make_cluster_state(
    const GammaPoissonClusterDataset& dataset,
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitResult& fit, double min_cluster_size) {
    GammaPoissonClusterState state;
    state.topic_state_checksum = dataset.posterior_header.state_checksum;
    state.topic_names = dataset.posterior_header.topic_names;
    state.basis = coordinates.log_ratio_basis.basis;
    state.model = fit.model;
    state.diagnostics = fit.diagnostics;
    state.document_uncertainty_model = dataset.uncertainty_model;
    state.document_uncertainty_rank = coordinates.uncertainty_rank;
    state.min_cluster_size = min_cluster_size;
    state.input_row_order = dataset.posterior_header.row_order;
    state.active = gamma_poisson_active_components(
        fit.effective_membership, min_cluster_size);
    return state;
}

void write_fit_outputs(const std::string& out_prefix,
    const GammaPoissonClusterDataset& dataset,
    const GammaPoissonClusterState& state,
    const GammaPoissonClusterFitResult& fit, int32_t n_representatives) {
    write_gamma_poisson_cluster_state(out_prefix + ".state.tsv", state);
    write_cluster_model(out_prefix + ".model.tsv", state, fit);
    write_cluster_results(out_prefix + ".results.tsv", dataset, fit.responsibilities);
    write_cluster_separation(out_prefix + ".separation.tsv", state);
    write_cluster_representatives(out_prefix + ".representatives.tsv",
        dataset, state, fit, n_representatives);
}

} // namespace

int32_t cmdGammaPoisClusterFit(int argc, char** argv) {
    std::string state_file, posterior_file, dispersion_file, out_prefix;
    std::string optimizer = "svi";
    std::string covariance_accumulation = "auto";
    int32_t n_clusters_max = 10;
    int32_t max_iterations = 50;
    int32_t kmeans_max_iterations = 20;
    int32_t convergence_patience = 3;
    int32_t cluster_covariance_rank = -1;
    int32_t diagonal_warmup_iterations = 5;
    int32_t orientation_update_interval = 1;
    int32_t orientation_max_updates = 5;
    int32_t orientation_patience = 2;
    int32_t minibatch_size = 1024;
    int32_t n_epochs = 30;
    int32_t svi_eval_size = 4096;
    int32_t refine_max_iterations = 20;
    int32_t candidate_components = 0;
    int32_t candidate_dimensions = 0;
    int32_t candidate_refresh_epochs = 5;
    int32_t prune_patience = 0;
    std::string candidate_search = "auto";
    int32_t n_representatives = 10;
    int32_t seed = 1;
    int32_t threads = 1;
    int32_t verbose = 0;
    double dirichlet_concentration = 1.0;
    double variance_floor = 1e-4;
    double tolerance = 1e-5;
    double responsibility_p90_tolerance = 0.01;
    double top_assignment_change_tolerance = 1e-3;
    double low_rank_variance_floor = 1e-6;
    double orientation_tolerance = 1e-3;
    double orientation_step = 0.5;
    double svi_kappa = 0.7;
    double svi_tau0 = 10.0;
    double min_cluster_size = 5.0;

    ParamList pl;
    pl.add_option("in-state", "Input Gamma-Poisson topic state", state_file, true)
      .add_option("in-posterior", "Input Gamma-Poisson local posterior", posterior_file, true)
      .add_option("in-posterior-dispersion", "Optional posterior dispersion sidecar", dispersion_file)
      .add_option("out-prefix", "Output prefix for clustering files", out_prefix, true)
      .add_option("optimizer", "Cluster optimizer: batch or svi", optimizer)
      .add_option("n-clusters-max", "Number of overfitted mixture components", n_clusters_max)
      .add_option("min-cluster-size", "Minimum absolute expected membership for an active cluster", min_cluster_size)
      .add_option("n-representatives", "Representative documents written per active cluster", n_representatives)
      .add_option("max-iter", "Maximum final fixed-covariance EM iterations", max_iterations)
      .add_option("kmeans-max-iter", "Maximum Lloyd initialization iterations", kmeans_max_iterations)
      .add_option("convergence-patience", "Consecutive stable iterations required for convergence", convergence_patience)
      .add_option("cluster-covariance-rank", "Shared-orientation cluster covariance rank; -1 selects min(5,K-1), zero selects diagonal", cluster_covariance_rank)
      .add_option("covariance-accumulation", "Conditional covariance accumulation: auto, dense, or compact", covariance_accumulation)
      .add_option("diagonal-warmup-iter", "Batch-EM diagonal warmup iterations before activating cluster covariance", diagonal_warmup_iterations)
      .add_option("orientation-update-interval", "EM iterations or SVI updates between shared-orientation updates", orientation_update_interval)
      .add_option("orientation-max-updates", "Maximum shared-orientation updates", orientation_max_updates)
      .add_option("orientation-patience", "Consecutive stable shared-orientation updates", orientation_patience)
      .add_option("minibatch-size", "Documents per in-memory SVI update", minibatch_size)
      .add_option("n-epochs", "Maximum shuffled SVI epochs", n_epochs)
      .add_option("svi-eval-size", "Fixed validation documents used for SVI convergence", svi_eval_size)
      .add_option("refine-max-iter", "Maximum deterministic EM iterations after SVI", refine_max_iterations)
      .add_option("candidate-components", "SVI candidate components per document; zero scores all", candidate_components)
      .add_option("candidate-dim", "Dimensions used by the SVI candidate proxy; zero uses all", candidate_dimensions)
      .add_option("candidate-refresh-epochs", "SVI epochs between exact full-component refreshes", candidate_refresh_epochs)
      .add_option("candidate-search", "Candidate selector: auto, linear, or kdtree", candidate_search)
      .add_option("prune-patience", "Full refreshes below minimum size before a component becomes dormant; zero disables", prune_patience)
      .add_option("svi-kappa", "Robbins-Monro SVI learning-rate exponent", svi_kappa)
      .add_option("svi-tau0", "Robbins-Monro SVI learning-rate offset", svi_tau0)
      .add_option("seed", "Initialization seed", seed)
      .add_option("threads", "Number of threads used by loading and the clustering E-step", threads)
      .add_option("verbose", "Progress reporting interval; zero disables", verbose)
      .add_option("dirichlet-concentration", "Symmetric total concentration for mixture weights", dirichlet_concentration)
      .add_option("variance-floor", "Minimum intrinsic variance per coordinate", variance_floor)
      .add_option("low-rank-variance-floor", "Minimum intrinsic low-rank variance", low_rank_variance_floor)
      .add_option("tol-orientation", "Shared-orientation convergence tolerance", orientation_tolerance)
      .add_option("orientation-step", "Shared-orientation damping step in (0,1]", orientation_step)
      .add_option("tol", "Relative ELBO convergence tolerance", tolerance)
      .add_option("tol-resp-p90", "90th percentile per-unit max-abs (L-inf) responsibility change convergence tolerance", responsibility_p90_tolerance)
      .add_option("tol-top-change", "Top-assignment change fraction convergence tolerance", top_assignment_change_tolerance);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        if (n_clusters_max <= 0 || !std::isfinite(min_cluster_size)
            || min_cluster_size < 0.0 || n_representatives <= 0
            || cluster_covariance_rank < -1) {
            throw std::invalid_argument("Invalid cluster count, activation threshold, or representative count");
        }
        const uint64_t checksum = gamma_poisson_state_checksum(state_file);
        GammaPoissonTopicModel topic_model(state_file, seed, threads, verbose);
        const auto& topic_names = topic_model.get_topic_names();
        GammaPoissonClusterDataset dataset = load_gamma_poisson_cluster_dataset(
            posterior_file, dispersion_file, checksum,
            topic_model.get_topic_capacity(), topic_names);
        GammaPoissonClusterCoordinates coordinates =
            make_gamma_poisson_cluster_coordinates(dataset);
        dataset.log_mean.resize(0, 0);
        dataset.topic_covariance.clear();
        dataset.topic_covariance.shrink_to_fit();
        const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
        if (cluster_covariance_rank < 0) {
            cluster_covariance_rank = std::min(5, dim);
        } else if (cluster_covariance_rank > dim) {
            throw std::invalid_argument("Cluster covariance rank exceeds K-1");
        }
        GammaPoissonClusterFitOptions options;
        if (optimizer == "batch") {
            options.optimizer = GammaPoissonClusterFitOptions::Optimizer::Batch;
        } else if (optimizer == "svi") {
            options.optimizer = GammaPoissonClusterFitOptions::Optimizer::Svi;
        } else {
            throw std::invalid_argument("--optimizer must be batch or svi");
        }
        if (covariance_accumulation == "auto") {
            options.covariance_accumulation = GammaPoissonClusterFitOptions::
                CovarianceAccumulation::Auto;
        } else if (covariance_accumulation == "dense") {
            options.covariance_accumulation = GammaPoissonClusterFitOptions::
                CovarianceAccumulation::Dense;
        } else if (covariance_accumulation == "compact") {
            options.covariance_accumulation = GammaPoissonClusterFitOptions::
                CovarianceAccumulation::Compact;
        } else {
            throw std::invalid_argument(
                "--covariance-accumulation must be auto, dense, or compact");
        }
        options.n_components = n_clusters_max;
        options.max_iterations = max_iterations;
        options.kmeans_max_iterations = kmeans_max_iterations;
        options.convergence_patience = convergence_patience;
        options.n_threads = threads;
        options.cluster_covariance_rank = cluster_covariance_rank;
        options.diagonal_warmup_iterations = diagonal_warmup_iterations;
        options.orientation_update_interval = orientation_update_interval;
        options.orientation_max_updates = orientation_max_updates;
        options.orientation_patience = orientation_patience;
        options.minibatch_size = minibatch_size;
        options.n_epochs = n_epochs;
        options.svi_eval_size = svi_eval_size;
        options.refine_max_iterations = refine_max_iterations;
        options.candidate_components = candidate_components;
        options.candidate_dimensions = candidate_dimensions;
        options.candidate_refresh_epochs = candidate_refresh_epochs;
        if (candidate_search == "auto") {
            options.candidate_search = GammaPoissonClusterFitOptions::
                CandidateSearch::Auto;
        } else if (candidate_search == "linear") {
            options.candidate_search = GammaPoissonClusterFitOptions::
                CandidateSearch::Linear;
        } else if (candidate_search == "kdtree") {
            options.candidate_search = GammaPoissonClusterFitOptions::
                CandidateSearch::KdTree;
        } else {
            throw std::invalid_argument(
                "--candidate-search must be auto, linear, or kdtree");
        }
        options.prune_patience = prune_patience;
        options.seed = seed;
        options.verbose = verbose;
        options.dirichlet_concentration = dirichlet_concentration;
        options.variance_floor = variance_floor;
        options.low_rank_variance_floor = low_rank_variance_floor;
        options.orientation_tolerance = orientation_tolerance;
        options.orientation_step = orientation_step;
        options.svi_kappa = svi_kappa;
        options.svi_tau0 = svi_tau0;
        options.min_cluster_size = min_cluster_size;
        options.tolerance = tolerance;
        options.responsibility_p90_tolerance = responsibility_p90_tolerance;
        options.top_assignment_change_tolerance = top_assignment_change_tolerance;
        GammaPoissonClusterFitResult fit = fit_gamma_poisson_cluster_mixture(
            coordinates, options);
        GammaPoissonClusterState state = make_cluster_state(
            dataset, coordinates, fit, min_cluster_size);
        write_fit_outputs(out_prefix, dataset, state, fit, n_representatives);
        notice("Loaded %zu posterior rows into memory and fitted %d components (%d active) with %s in %d iterations (%d epochs, %d SVI updates, %d refinement iterations; %s; %s covariance accumulation)",
            dataset.identifiers.size(), static_cast<int32_t>(state.model.dirichlet_parameters.size()),
            static_cast<int32_t>(std::count(state.active.begin(), state.active.end(), uint8_t{1})),
            state.diagnostics.optimizer.c_str(),
            state.diagnostics.iterations,
            state.diagnostics.epochs, state.diagnostics.svi_updates,
            state.diagnostics.refinement_iterations,
            state.diagnostics.converged ? "converged" : "maximum reached",
            state.diagnostics.covariance_accumulation.c_str());
        notice("Gamma-Poisson clustering outputs written to %s.{state,model,results,separation,representatives}.tsv",
            out_prefix.c_str());
    } catch (const std::exception& ex) {
        std::cerr << "Gamma-Poisson clustering failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}

int32_t cmdGammaPoisClusterTransform(int argc, char** argv) {
    std::string state_file, cluster_state_file, posterior_file, dispersion_file,
        out_prefix;
    int32_t seed = 1;
    int32_t threads = 1;
    int32_t verbose = 0;
    ParamList pl;
    pl.add_option("in-state", "Input Gamma-Poisson topic state", state_file, true)
      .add_option("in-cluster-state", "Input fitted Gamma-Poisson cluster state", cluster_state_file, true)
      .add_option("in-posterior", "Input Gamma-Poisson local posterior", posterior_file, true)
      .add_option("in-posterior-dispersion", "Optional posterior dispersion sidecar", dispersion_file)
      .add_option("out-prefix", "Output prefix for cluster assignments", out_prefix, true)
      .add_option("seed", "Topic-state loading seed", seed)
      .add_option("threads", "Number of threads used by loading and assignment", threads)
      .add_option("verbose", "Verbose level", verbose);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        const uint64_t checksum = gamma_poisson_state_checksum(state_file);
        GammaPoissonTopicModel topic_model(state_file, seed, threads, verbose);
        const auto& topic_names = topic_model.get_topic_names();
        GammaPoissonClusterState state = read_gamma_poisson_cluster_state(
            cluster_state_file, checksum);
        GammaPoissonClusterDataset dataset = load_gamma_poisson_cluster_dataset(
            posterior_file, dispersion_file, checksum,
            topic_model.get_topic_capacity(), topic_names);
        if (dataset.posterior_header.topic_names != state.topic_names) {
            throw std::runtime_error("Posterior topic names do not match cluster state");
        }
        GammaPoissonClusterCoordinates coordinates =
            make_gamma_poisson_cluster_coordinates(dataset, state.basis);
        if (dataset.uncertainty_model != state.document_uncertainty_model
            || coordinates.uncertainty_rank != state.document_uncertainty_rank) {
            throw std::runtime_error(
                "Posterior document uncertainty rank does not match cluster state");
        }
        dataset.log_mean.resize(0, 0);
        dataset.topic_covariance.clear();
        dataset.topic_covariance.shrink_to_fit();
        GammaPoissonClusterScore score = score_gamma_poisson_cluster_mixture(
            coordinates, state.model, threads);
        write_cluster_results(out_prefix + ".results.tsv", dataset,
            score.responsibilities);
        notice("Assigned %zu posterior rows using %d fixed Gamma-Poisson cluster components",
            dataset.identifiers.size(),
            static_cast<int32_t>(state.model.dirichlet_parameters.size()));
        notice("Gamma-Poisson cluster assignments written to %s.results.tsv",
            out_prefix.c_str());
    } catch (const std::exception& ex) {
        std::cerr << "Gamma-Poisson cluster transform failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
