#include "gamma_pois_cluster_internal.hpp"
#include "dense_kmeans.hpp"
#include "error.hpp"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>

#include <tbb/global_control.h>

namespace gamma_pois_cluster_detail {

void log_top_cluster_size_fractions(
    const Eigen::Ref<const Eigen::VectorXd>& membership, int32_t n_units,
    const std::string& stage, const ClusterConvergenceStats* convergence) {
    const int32_t components = static_cast<int32_t>(membership.size());
    if (n_units <= 0 || components == 0 || !membership.allFinite()) {
        return;
    }
    std::vector<int32_t> order(components);
    std::iota(order.begin(), order.end(), 0);
    const int32_t topk = std::min(10, components);
    std::partial_sort(order.begin(), order.begin() + topk, order.end(),
        [&](int32_t a, int32_t b) { return membership(a) > membership(b); });
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(4);
    for (int32_t i = 0; i < topk; ++i) {
        const int32_t c = order[i];
        oss << (i ? " " : "") << c << ":"
            << membership(c) / static_cast<double>(n_units);
    }
    if (convergence != nullptr) {
        oss << std::scientific << std::setprecision(3) << "\n    "
            << convergence->relative_label << "="
            << convergence->relative_change
            << " p90dResp=" << convergence->p90_responsibility_change
            << " dTop=" << convergence->top_assignment_change;
    }
    notice("Gamma-Poisson cluster sizes [%s] top %d fractions:\n    %s",
        stage.c_str(), topk, oss.str().c_str());
}

double dirichlet_elbo(const Eigen::Ref<const Eigen::VectorXd>& parameters,
    double alpha) {
    const double sum = parameters.sum();
    const double psi_sum = psi(sum);
    double out = std::lgamma(alpha * parameters.size())
        - parameters.size() * std::lgamma(alpha) - std::lgamma(sum);
    for (Eigen::Index c = 0; c < parameters.size(); ++c) {
        const double expected_log = psi(parameters(c)) - psi_sum;
        out += std::lgamma(parameters(c))
            + (alpha - parameters(c)) * expected_log;
    }
    return out;
}

void run_m_step(const ClusterStatistics& stats, GammaPoissonClusterModel& model,
    double alpha, const GammaPoissonClusterFitOptions& options,
    GammaPoissonClusterFitDiagnostics& diagnostics,
    const std::vector<uint8_t>* dormant,
    const Eigen::VectorXd* weight_membership) {
    if (dormant
        && dormant->size() != static_cast<size_t>(model.means.rows())) {
        throw std::invalid_argument("Invalid dormant Gamma-Poisson components");
    }
    if (weight_membership
        && weight_membership->size() != model.means.rows()) {
        throw std::invalid_argument(
            "Invalid Gamma-Poisson weight membership override");
    }
    const RowMajorMatrixXd old_means = model.means;
    const RowMajorMatrixXd old_diagonal = model.variances;
    const RowMajorMatrixXd old_low_rank = model.low_rank_variances;
    const Eigen::VectorXd old_weights = gamma_poisson_cluster_weights(model);
    const Eigen::MatrixXd orientation_squared =
        model.orientation.array().square();
    for (Eigen::Index c = 0; c < model.means.rows(); ++c) {
        if (dormant && (*dormant)[c]) continue;
        if (stats.membership(c) <= 1e-12) continue;
        model.means.row(c) = stats.first.row(c) / stats.membership(c);
        Eigen::VectorXd target_diagonal;
        if (stats.dense) {
            target_diagonal = (stats.second_dense[c].diagonal()
                / stats.membership(c)
                - model.means.row(c).transpose().array().square().matrix())
                .cwiseMax(0.0);
        } else {
            target_diagonal = (
                stats.second_diagonal.row(c) / stats.membership(c)
                - model.means.row(c).array().square().matrix()).transpose()
                .cwiseMax(0.0);
        }
        if (model.orientation.cols() == 0) {
            model.variances.row(c) = target_diagonal.array()
                .max(options.variance_floor).transpose();
            continue;
        }
        const Eigen::VectorXd projected_mean = model.orientation.transpose()
            * model.means.row(c).transpose();
        Eigen::MatrixXd target_projected;
        if (stats.dense) {
            target_projected = model.orientation.transpose()
                * stats.second_dense[c] * model.orientation
                / stats.membership(c)
                - projected_mean * projected_mean.transpose();
        } else {
            target_projected = stats.second_projected[c] / stats.membership(c)
                - projected_mean * projected_mean.transpose();
        }
        Eigen::VectorXd diagonal = model.variances.row(c).transpose();
        Eigen::VectorXd low_rank =
            model.low_rank_variances.row(c).transpose();
        for (int32_t pass = 0; pass < 4; ++pass) {
            low_rank = (target_projected.diagonal()
                - orientation_squared.transpose() * diagonal)
                .cwiseMax(options.low_rank_variance_floor);
            diagonal = (target_diagonal - orientation_squared * low_rank)
                .cwiseMax(options.variance_floor);
        }
        model.variances.row(c) = diagonal.transpose();
        model.low_rank_variances.row(c) = low_rank.transpose();
    }
    model.dirichlet_parameters = (weight_membership
        ? *weight_membership : stats.membership).array() + alpha;
    diagnostics.max_standardized_center_change = 0.0;
    diagnostics.max_log_variance_change = 0.0;
    for (Eigen::Index c = 0; c < model.means.rows(); ++c) {
        for (Eigen::Index j = 0; j < model.means.cols(); ++j) {
            diagnostics.max_standardized_center_change = std::max(
                diagnostics.max_standardized_center_change,
                std::abs(model.means(c, j) - old_means(c, j))
                    / std::sqrt(old_diagonal(c, j)));
            diagnostics.max_log_variance_change = std::max(
                diagnostics.max_log_variance_change,
                std::abs(std::log(model.variances(c, j))
                    - std::log(old_diagonal(c, j))));
        }
    }
    if (model.low_rank_variances.size() > 0
        && old_low_rank.size() > 0) {
        diagnostics.max_log_variance_change = std::max(
            diagnostics.max_log_variance_change,
            (model.low_rank_variances.array().log()
                - old_low_rank.array().log()).abs().maxCoeff());
    }
    diagnostics.max_weight_change = (gamma_poisson_cluster_weights(model)
        - old_weights).cwiseAbs().maxCoeff();
}



static ClusterStatistics run_full_e_step(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, int32_t n_threads,
    RowMajorMatrixXd& responsibilities, const Eigen::MatrixXd* sketch,
    bool collect_moments, bool dense_accumulation) {
    EStepRequest request;
    request.n_threads = n_threads;
    request.responsibilities = &responsibilities;
    request.sketch = sketch;
    request.collect_moments = collect_moments;
    request.dense_accumulation = dense_accumulation;
    return run_e_step(coordinates, model, request);
}


void update_diagnostics(const ClusterStatistics& stats, double alpha,
    GammaPoissonClusterFitResult& out) {
    out.diagnostics.elbo = stats.elbo_local
        + dirichlet_elbo(out.model.dirichlet_parameters, alpha);
    out.diagnostics.log_likelihood = stats.predictive_log_likelihood;
    out.diagnostics.elbo_trace.push_back(out.diagnostics.elbo);
}

void validate_common_fit_input(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options) {
    const auto finite = [](double value) { return std::isfinite(value); };
    if (coordinates.mean.rows() == 0 || coordinates.mean.cols() == 0
        || coordinates.uncertainty_diagonal.rows() != coordinates.mean.rows()
        || coordinates.uncertainty_diagonal.cols() != coordinates.mean.cols()
        || coordinates.uncertainty_rank < 0
        || coordinates.uncertainty_factor.rows() != coordinates.mean.rows()
        || coordinates.uncertainty_factor.cols()
            != coordinates.mean.cols() * coordinates.uncertainty_rank
        || options.n_components <= 0 || options.kmeans_max_iterations <= 0
        || options.convergence_patience <= 0 || options.n_threads <= 0
        || options.cluster_covariance_rank < 0
        || options.cluster_covariance_rank > coordinates.mean.cols()
        || options.orientation_update_interval <= 0
        || options.orientation_max_updates < 0 || options.orientation_patience <= 0
        || !finite(options.variance_floor) || options.variance_floor <= 0.0
        || !finite(options.low_rank_variance_floor)
        || options.low_rank_variance_floor <= 0.0
        || !finite(options.orientation_step) || options.orientation_step <= 0.0
        || options.orientation_step > 1.0
        || !finite(options.orientation_tolerance)
        || options.orientation_tolerance < 0.0
        || !finite(options.tolerance) || options.tolerance < 0.0
        || !finite(options.responsibility_p90_tolerance)
        || options.responsibility_p90_tolerance < 0.0
        || options.responsibility_p90_tolerance > 1.0
        || !finite(options.top_assignment_change_tolerance)
        || options.top_assignment_change_tolerance < 0.0
        || options.top_assignment_change_tolerance > 1.0
        || !finite(options.dirichlet_concentration)
        || options.dirichlet_concentration <= 0.0
        || static_cast<int32_t>(options.covariance_accumulation)
            < static_cast<int32_t>(GammaPoissonClusterFitOptions::
                CovarianceAccumulation::Auto)
        || static_cast<int32_t>(options.covariance_accumulation)
            > static_cast<int32_t>(GammaPoissonClusterFitOptions::
                CovarianceAccumulation::Compact)
        || static_cast<int32_t>(options.candidate_search)
            < static_cast<int32_t>(GammaPoissonClusterFitOptions::
                CandidateSearch::Auto)
        || static_cast<int32_t>(options.candidate_search)
            > static_cast<int32_t>(GammaPoissonClusterFitOptions::
                CandidateSearch::KdTree)) {
        throw std::invalid_argument(
            "Invalid structured Gamma-Poisson mixture input or options");
    }
    if (!coordinates.mean.allFinite()
        || !coordinates.uncertainty_diagonal.allFinite()
        || !coordinates.uncertainty_factor.allFinite()
        || (coordinates.uncertainty_diagonal.array() <= 0.0).any()) {
        throw std::invalid_argument(
            "Structured mixture inputs must be finite with positive uncertainty");
    }
}

bool select_dense_accumulation(const GammaPoissonClusterFitOptions& options,
    int32_t dim, int32_t rank) {
    return rank > 0 && (
        options.covariance_accumulation
            == GammaPoissonClusterFitOptions::CovarianceAccumulation::Dense
        || (options.covariance_accumulation
                == GammaPoissonClusterFitOptions::CovarianceAccumulation::Auto
            && dim + 1 <= 48));
}


InitializedClusterFit initialize_cluster_fit(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options, bool sampled) {
    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
    const int32_t components = std::min(options.n_components, n);
    const double alpha = options.dirichlet_concentration / components;
    DenseKMeansOptions kmeans_options;
    kmeans_options.n_clusters = components;
    kmeans_options.max_iterations = options.kmeans_max_iterations;
    kmeans_options.seed = options.seed;

    DenseKMeansResult kmeans;
    if (sampled) {
        const int64_t requested_sample = std::max<int64_t>(
            8192, int64_t{256} * components);
        const int32_t sample_size = static_cast<int32_t>(std::min<int64_t>(
            n, std::min<int64_t>(50000, requested_sample)));
        kmeans = sampled_dense_kmeans(
            coordinates.mean, kmeans_options, sample_size);
    } else {
        kmeans = dense_kmeans(coordinates.mean, kmeans_options);
    }

    InitializedClusterFit out;
    out.model.means = kmeans.centers;
    out.model.variances = RowMajorMatrixXd::Zero(components, dim);
    out.initial_shared = Eigen::MatrixXd::Zero(dim, dim);
    for (int32_t d = 0; d < n; ++d) {
        const int32_t c = kmeans.assignments(d);
        const Eigen::VectorXd residual = coordinates.mean.row(d).transpose()
            - out.model.means.row(c).transpose();
        out.model.variances.row(c).array() +=
            residual.array().square().transpose();
        out.initial_shared.noalias() += residual * residual.transpose();
    }
    for (int32_t c = 0; c < components; ++c) {
        if (kmeans.counts(c) <= 0) {
            throw std::runtime_error(
                "Gamma-Poisson initialization produced an empty cluster");
        }
        out.model.variances.row(c) = (
            out.model.variances.row(c)
                / static_cast<double>(kmeans.counts(c))).array()
            .max(options.variance_floor);
    }
    out.model.orientation.resize(dim, 0);
    out.model.low_rank_variances.resize(components, 0);
    out.model.dirichlet_parameters =
        kmeans.counts.cast<double>().array() + alpha;
    out.kmeans_iterations = kmeans.iterations;
    out.kmeans_inertia = kmeans.inertia;
    out.kmeans_converged = kmeans.converged;
    if (options.verbose > 0) {
        notice("Gamma-Poisson clusters initialized with k-means++ "
            "(%d components, %d iterations, %s)", components, kmeans.iterations,
            kmeans.converged ? "converged" : "not converged");
        const Eigen::VectorXd initial_sizes = kmeans.counts.cast<double>();
        log_top_cluster_size_fractions(initial_sizes, n, "kmeans++ init");
    }
    return out;
}

void initialize_fit_diagnostics(const InitializedClusterFit& initialized,
    const char* optimizer, bool dense_accumulation, int32_t rank,
    GammaPoissonClusterFitResult& out) {
    out.diagnostics.optimizer = optimizer;
    out.diagnostics.covariance_accumulation = rank == 0
        ? "diagonal" : (dense_accumulation ? "dense" : "compact");
    out.diagnostics.kmeans_iterations = initialized.kmeans_iterations;
    out.diagnostics.kmeans_inertia = initialized.kmeans_inertia;
    out.diagnostics.kmeans_converged = initialized.kmeans_converged;
}

void run_exact_em(const GammaPoissonClusterCoordinates& coordinates,
    GammaPoissonClusterModel& model, const GammaPoissonClusterFitOptions& options,
    double alpha, bool dense_accumulation, int32_t max_iterations,
    bool refinement, GammaPoissonClusterFitResult& out) {
    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    RowMajorMatrixXd previous_responsibilities;
    double previous_elbo = -std::numeric_limits<double>::infinity();
    double previous_log_likelihood = -std::numeric_limits<double>::infinity();
    int32_t stable_iterations = 0;
    ClusterStatistics stats(static_cast<int32_t>(model.means.rows()),
        static_cast<int32_t>(model.means.cols()),
        static_cast<int32_t>(model.orientation.cols()), 0,
        dense_accumulation);
    bool final_scored = false;
    for (int32_t iteration = 0; iteration < max_iterations; ++iteration) {
        stats = run_full_e_step(coordinates, model, options.n_threads,
            out.responsibilities, nullptr, true, dense_accumulation);
        final_scored = true;
        out.model = model;
        update_diagnostics(stats, alpha, out);
        out.effective_membership = stats.membership;
        if (refinement) {
            ++out.diagnostics.refinement_iterations;
        } else {
            ++out.diagnostics.structured_iterations;
        }
        const bool have_change = previous_responsibilities.rows() == n;
        bool converged_now = false;
        if (have_change) {
            const GammaPoissonResponsibilityChange change =
                gamma_poisson_responsibility_change(
                    previous_responsibilities, out.responsibilities);
            out.diagnostics.mean_responsibility_linf_change = change.mean_linf;
            out.diagnostics.p90_responsibility_linf_change = change.p90_linf;
            out.diagnostics.top_assignment_change_fraction =
                change.top_assignment_fraction;
            out.diagnostics.relative_elbo_change = std::abs(
                out.diagnostics.elbo - previous_elbo)
                / (1.0 + std::abs(previous_elbo));
            out.diagnostics.relative_predictive_log_likelihood_change = std::abs(
                out.diagnostics.log_likelihood - previous_log_likelihood)
                / (1.0 + std::abs(previous_log_likelihood));
            const bool stable = out.diagnostics.relative_elbo_change
                    <= options.tolerance
                && out.diagnostics.p90_responsibility_linf_change
                    <= options.responsibility_p90_tolerance
                && out.diagnostics.top_assignment_change_fraction
                    <= options.top_assignment_change_tolerance;
            stable_iterations = stable ? stable_iterations + 1 : 0;
            converged_now = stable_iterations >= options.convergence_patience;
        }
        if (options.verbose > 0
            && (iteration % options.verbose == 0 || converged_now)) {
            const ClusterConvergenceStats conv{
                out.diagnostics.relative_elbo_change,
                out.diagnostics.p90_responsibility_linf_change,
                out.diagnostics.top_assignment_change_fraction, "dELBO"};
            log_top_cluster_size_fractions(stats.membership, n,
                (refinement ? "refine iter " : "em iter ")
                + std::to_string(iteration),
                have_change ? &conv : nullptr);
        }
        if (converged_now) {
            out.diagnostics.converged = true;
            break;
        }
        previous_responsibilities = out.responsibilities;
        previous_elbo = out.diagnostics.elbo;
        previous_log_likelihood = out.diagnostics.log_likelihood;
        run_m_step(stats, model, alpha, options, out.diagnostics);
        final_scored = false;
    }
    if (!final_scored) {
        stats = run_full_e_step(coordinates, model, options.n_threads,
            out.responsibilities, nullptr, true, dense_accumulation);
        out.model = model;
        update_diagnostics(stats, alpha, out);
        out.effective_membership = stats.membership;
    }
    out.model = model;
}

void validate_model(const GammaPoissonClusterModel& model, int32_t dim) {
    const Eigen::Index components = model.dirichlet_parameters.size();
    const Eigen::Index rank = model.orientation.cols();
    if (components <= 0 || model.means.rows() != components
        || model.means.cols() != dim || model.variances.rows() != components
        || model.variances.cols() != dim || model.orientation.rows() != dim
        || model.low_rank_variances.rows() != components
        || model.low_rank_variances.cols() != rank
        || rank < 0 || rank > dim
        || !model.dirichlet_parameters.allFinite() || !model.means.allFinite()
        || !model.variances.allFinite() || !model.orientation.allFinite()
        || !model.low_rank_variances.allFinite()
        || (model.dirichlet_parameters.array() <= 0.0).any()
        || (model.variances.array() <= 0.0).any()
        || (rank > 0 && (model.low_rank_variances.array() <= 0.0).any())
        || (rank > 0 && (model.orientation.transpose() * model.orientation
            - Eigen::MatrixXd::Identity(rank, rank)).norm() > 1e-8)) {
        throw std::invalid_argument("Invalid Gamma-Poisson cluster model");
    }
}


GammaPoissonClusterFitResult fit_batch(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options) {
    validate_common_fit_input(coordinates, options);
    if (options.max_iterations <= 0
        || options.candidate_components != 0
        || options.candidate_dimensions != 0 || options.prune_patience != 0
        || options.diagonal_warmup_iterations < 0) {
        throw std::invalid_argument(
            "Invalid batch Gamma-Poisson mixture options");
    }
    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
    const int32_t components = std::min(options.n_components, n);
    const int32_t rank = options.cluster_covariance_rank;
    const bool dense_accumulation = select_dense_accumulation(
        options, dim, rank);
    const double alpha = options.dirichlet_concentration / components;
    tbb::global_control parallelism(tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(options.n_threads));

    InitializedClusterFit initialized = initialize_cluster_fit(
        coordinates, options, false);
    GammaPoissonClusterModel model = std::move(initialized.model);
    const Eigen::MatrixXd& initial_shared = initialized.initial_shared;
    GammaPoissonClusterFitResult out;
    initialize_fit_diagnostics(
        initialized, "batch", dense_accumulation, rank, out);
    const Eigen::MatrixXd sketch = rank > 0 && !dense_accumulation
        ? make_orientation_sketch(dim, rank, options.seed) : Eigen::MatrixXd(dim, 0);

    ClusterStatistics stats(components, dim, 0, 0, dense_accumulation);
    const int32_t warmup = rank > 0 ? options.diagonal_warmup_iterations : 0;
    for (int32_t iteration = 0; iteration < warmup; ++iteration) {
        const Eigen::MatrixXd* use_sketch = !dense_accumulation
                && iteration + 1 == warmup
            ? &sketch : nullptr;
        stats = run_full_e_step(coordinates, model, options.n_threads,
            out.responsibilities, use_sketch, true, dense_accumulation);
        out.model = model;
        update_diagnostics(stats, alpha, out);
        if (options.verbose > 0 && iteration % options.verbose == 0) {
            log_top_cluster_size_fractions(stats.membership, n,
                "warmup iter " + std::to_string(iteration));
        }
        run_m_step(stats, model, alpha, options, out.diagnostics);
        ++out.diagnostics.warmup_iterations;
    }

    if (rank > 0) {
        if (warmup > 0) {
            const RowMajorMatrixXd fallback = leading_orientation(
                initial_shared, rank);
            model.orientation = dense_accumulation
                ? leading_orientation(
                    pooled_residual_covariance(stats, model), rank)
                : orientation_from_signed_sketch(sketch,
                    pooled_residual_sketch(stats, model, sketch),
                    fallback, rank);
        } else {
            model.orientation = leading_orientation(initial_shared, rank);
        }
        model.low_rank_variances = RowMajorMatrixXd::Constant(
            components, rank, options.low_rank_variance_floor);
        int32_t stable_updates = 0;
        for (int32_t update = 0; update < options.orientation_max_updates; ++update) {
            for (int32_t iteration = 0;
                 iteration < options.orientation_update_interval; ++iteration) {
                const Eigen::MatrixXd* use_sketch =
                    !dense_accumulation
                        && iteration + 1 == options.orientation_update_interval
                    ? &sketch : nullptr;
                stats = run_full_e_step(coordinates, model, options.n_threads,
                    out.responsibilities, use_sketch, true,
                    dense_accumulation);
                out.model = model;
                update_diagnostics(stats, alpha, out);
                if (options.verbose > 0
                    && out.diagnostics.structured_iterations % options.verbose == 0) {
                    log_top_cluster_size_fractions(stats.membership, n,
                        "structured iter "
                        + std::to_string(out.diagnostics.structured_iterations));
                }
                run_m_step(stats, model, alpha, options, out.diagnostics);
                ++out.diagnostics.structured_iterations;
            }
            const RowMajorMatrixXd previous = model.orientation;
            RowMajorMatrixXd candidate = dense_accumulation
                ? leading_orientation(
                    pooled_residual_covariance(stats, model), rank)
                : orientation_from_signed_sketch(sketch,
                    pooled_residual_sketch(stats, model, sketch), previous, rank);
            out.diagnostics.orientation_change = align_and_blend_orientation(
                model.orientation, std::move(candidate), options.orientation_step);
            transport_orientation(model, previous, options);
            ++out.diagnostics.orientation_updates;
            if (options.verbose > 0) {
                notice("Gamma-Poisson orientation update %d: change = %.3e ", out.diagnostics.orientation_updates,
                    out.diagnostics.orientation_change);
            }
            stable_updates = out.diagnostics.orientation_change
                    <= options.orientation_tolerance
                ? stable_updates + 1 : 0;
            if (stable_updates >= options.orientation_patience) {
                out.diagnostics.orientation_converged = true;
                break;
            }
        }
    } else {
        out.diagnostics.orientation_converged = true;
        out.diagnostics.orientation_change = 0.0;
    }

    run_exact_em(coordinates, model, options, alpha, dense_accumulation,
        options.max_iterations, false, out);
    out.diagnostics.iterations = out.diagnostics.warmup_iterations
        + out.diagnostics.structured_iterations;
    return out;
}


} // namespace gamma_pois_cluster_detail

GammaPoissonClusterFitResult fit_gamma_poisson_cluster_mixture(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options) {
    switch (options.optimizer) {
        case GammaPoissonClusterFitOptions::Optimizer::Batch:
            return gamma_pois_cluster_detail::fit_batch(coordinates, options);
        case GammaPoissonClusterFitOptions::Optimizer::Svi:
            return gamma_pois_cluster_detail::fit_svi(coordinates, options);
    }
    throw std::invalid_argument("Invalid Gamma-Poisson cluster optimizer");
}
