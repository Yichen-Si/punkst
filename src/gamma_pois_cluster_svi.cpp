#include "gamma_pois_cluster_internal.hpp"
#include "error.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace gamma_pois_cluster_detail {

DormancyTracker::DormancyTracker(int32_t components) {
    if (components <= 0) {
        throw std::invalid_argument(
            "Dormancy tracker requires positive component count");
    }
    dormant_.assign(components, 0);
    low_mass_refreshes_.assign(components, 0);
    last_exact_membership_ = Eigen::VectorXd::Zero(components);
}

void DormancyTracker::observe_exact(
    const Eigen::Ref<const Eigen::VectorXd>& membership,
    double min_cluster_size, int32_t patience, bool update_status) {
    if (membership.size() != last_exact_membership_.size()
        || !membership.allFinite() || (membership.array() < 0.0).any()
        || !std::isfinite(min_cluster_size) || min_cluster_size < 0.0
        || patience < 0) {
        throw std::invalid_argument(
            "Invalid exact membership for dormancy tracker");
    }
    last_exact_membership_ = membership;
    if (!update_status || patience == 0) return;
    for (Eigen::Index c = 0; c < membership.size(); ++c) {
        if (membership(c) >= min_cluster_size) {
            low_mass_refreshes_[c] = 0;
            dormant_[c] = 0;
        } else {
            ++low_mass_refreshes_[c];
            if (low_mass_refreshes_[c] >= patience) dormant_[c] = 1;
        }
    }
    if (std::all_of(dormant_.begin(), dormant_.end(),
            [](uint8_t value) { return value != 0; })) {
        Eigen::Index largest = 0;
        membership.maxCoeff(&largest);
        dormant_[largest] = 0;
        low_mass_refreshes_[largest] = 0;
    }
}

Eigen::VectorXd DormancyTracker::preserve_exact_weights(
    const Eigen::Ref<const Eigen::VectorXd>& membership) const {
    if (membership.size() != last_exact_membership_.size()
        || !membership.allFinite() || (membership.array() < 0.0).any()) {
        throw std::invalid_argument(
            "Invalid candidate membership for dormancy tracker");
    }
    if (std::none_of(dormant_.begin(), dormant_.end(),
            [](uint8_t value) { return value != 0; })) {
        return membership;
    }
    Eigen::VectorXd out = membership;
    double reserved = 0.0;
    double active_mass = 0.0;
    for (Eigen::Index c = 0; c < membership.size(); ++c) {
        if (dormant_[c]) {
            reserved += last_exact_membership_(c);
        } else {
            active_mass += membership(c);
        }
    }
    const double available = std::max(
        0.0, last_exact_membership_.sum() - reserved);
    const double scale = active_mass > 0.0 ? available / active_mass : 0.0;
    for (Eigen::Index c = 0; c < membership.size(); ++c) {
        out(c) = dormant_[c] ? last_exact_membership_(c)
                             : scale * membership(c);
    }
    return out;
}


static ClusterStatistics run_svi_e_step(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, const int32_t* documents,
    int32_t n_documents, int32_t n_threads,
    RowMajorMatrixXd* responsibilities, const Eigen::MatrixXd* sketch,
    bool collect_moments, bool dense_accumulation) {
    EStepRequest request;
    request.document_indices = documents;
    request.n_documents = n_documents;
    request.n_threads = n_threads;
    request.responsibilities = responsibilities;
    request.sketch = sketch;
    request.collect_moments = collect_moments;
    request.dense_accumulation = dense_accumulation;
    return run_e_step(coordinates, model, request);
}


static std::vector<int32_t> deterministic_document_sample(
    int32_t n, int32_t sample_size, int32_t seed) {
    sample_size = std::min(n, sample_size);
    std::vector<int32_t> sample(sample_size);
    std::iota(sample.begin(), sample.end(), int32_t{0});
    std::mt19937 random(static_cast<uint32_t>(seed));
    for (int32_t d = sample_size; d < n; ++d) {
        std::uniform_int_distribution<int32_t> draw(0, d);
        const int32_t selected = draw(random);
        if (selected < sample_size) sample[selected] = d;
    }
    std::sort(sample.begin(), sample.end());
    return sample;
}

static ClusterStatistics exact_refresh_in_batches(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model, const GammaPoissonClusterFitOptions& options,
    const Eigen::MatrixXd* sketch, bool dense_accumulation,
    std::vector<int32_t>& previous_top) {
    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    const int32_t components = static_cast<int32_t>(model.means.rows());
    const int32_t dim = static_cast<int32_t>(model.means.cols());
    const int32_t rank = static_cast<int32_t>(model.orientation.cols());
    const int32_t sketch_size = sketch ? static_cast<int32_t>(sketch->cols()) : 0;
    ClusterStatistics total(components, dim, rank, sketch_size,
        dense_accumulation);
    const PreparedEStepModel prepared = prepare_e_step_model(
        model, dense_accumulation);
    std::vector<int32_t> documents(n);
    std::iota(documents.begin(), documents.end(), int32_t{0});
    for (int32_t begin = 0; begin < n; begin += options.minibatch_size) {
        const int32_t batch_size = std::min(options.minibatch_size, n - begin);
        EStepRequest request;
        request.document_indices = documents.data() + begin;
        request.n_documents = batch_size;
        request.n_threads = options.n_threads;
        request.sketch = sketch;
        request.collect_moments = true;
        request.dense_accumulation = dense_accumulation;
        request.document_top_components = previous_top.data();
        ClusterStatistics batch = run_e_step_prepared(
            coordinates, model, prepared, request);
        total.add_scaled(batch, 1.0);
    }
    return total;
}

GammaPoissonClusterFitResult fit_svi(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterFitOptions& options) {
    validate_common_fit_input(coordinates, options);
    const auto finite = [](double value) { return std::isfinite(value); };
    if (options.minibatch_size <= 0 || options.n_epochs <= 0
        || options.svi_eval_size <= 0 || options.refine_max_iterations < 0
        || options.candidate_components < 0
        || options.candidate_dimensions < 0
        || options.candidate_dimensions > coordinates.mean.cols()
        || options.candidate_refresh_epochs <= 0 || options.prune_patience < 0
        || !finite(options.min_cluster_size) || options.min_cluster_size < 0.0
        || !finite(options.svi_kappa) || options.svi_kappa <= 0.5
        || options.svi_kappa > 1.0 || !finite(options.svi_tau0)
        || options.svi_tau0 <= 0.0) {
        throw std::invalid_argument(
            "Invalid structured Gamma-Poisson SVI input or options");
    }

    const int32_t n = static_cast<int32_t>(coordinates.mean.rows());
    const int32_t dim = static_cast<int32_t>(coordinates.mean.cols());
    const int32_t components = std::min(options.n_components, n);
    const int32_t rank = options.cluster_covariance_rank;
    const bool candidate_enabled = options.candidate_components > 0
        && options.candidate_components < components;
    if (candidate_enabled && options.candidate_components < 3) {
        throw std::invalid_argument(
            "--candidate-components must be zero or at least three");
    }
    if (options.prune_patience > 0 && !candidate_enabled) {
        throw std::invalid_argument(
            "--prune-patience requires truncated candidate components");
    }
    const int32_t candidate_dimensions = options.candidate_dimensions == 0
        ? dim : options.candidate_dimensions;
    GammaPoissonClusterFitOptions::CandidateSearch candidate_search =
        options.candidate_search;
    if (!candidate_enabled) {
        if (candidate_search != GammaPoissonClusterFitOptions::
                CandidateSearch::Auto) {
            throw std::invalid_argument(
                "Explicit candidate search requires candidate truncation");
        }
    } else if (candidate_search == GammaPoissonClusterFitOptions::
            CandidateSearch::Auto) {
        const int32_t threshold = candidate_dimensions <= 16 ? 128
            : (candidate_dimensions <= 32 ? 128 : 64);
        candidate_search = components >= threshold
            ? GammaPoissonClusterFitOptions::CandidateSearch::KdTree
            : GammaPoissonClusterFitOptions::CandidateSearch::Linear;
    }
    const bool dense_accumulation = select_dense_accumulation(
        options, dim, rank);
    const double alpha = options.dirichlet_concentration / components;
    tbb::global_control parallelism(tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(options.n_threads));

    InitializedClusterFit initialized = initialize_cluster_fit(
        coordinates, options, true);
    GammaPoissonClusterModel model = std::move(initialized.model);
    const Eigen::MatrixXd& initial_shared = initialized.initial_shared;

    GammaPoissonClusterFitResult out;
    initialize_fit_diagnostics(
        initialized, "svi", dense_accumulation, rank, out);
    out.diagnostics.candidate_components = candidate_enabled
        ? options.candidate_components : 0;
    out.diagnostics.candidate_dimensions = candidate_enabled
        ? candidate_dimensions : 0;
    out.diagnostics.candidate_search = !candidate_enabled ? "none"
        : (candidate_search == GammaPoissonClusterFitOptions::
                CandidateSearch::KdTree ? "kdtree" : "linear");
    const Eigen::MatrixXd sketch = rank > 0 && !dense_accumulation
        ? make_orientation_sketch(dim, rank, options.seed)
        : Eigen::MatrixXd(dim, 0);
    const Eigen::MatrixXd* use_sketch = sketch.cols() > 0 ? &sketch : nullptr;
    std::vector<int32_t> previous_top(static_cast<size_t>(n) * 3, -1);
    gamma_pois_cluster_detail::DormancyTracker dormancy(components);

    ClusterStatistics persistent = run_svi_e_step(coordinates, model,
        nullptr, 0, options.n_threads, nullptr, use_sketch, true,
        dense_accumulation);
    out.model = model;
    update_diagnostics(persistent, alpha, out);
    run_m_step(persistent, model, alpha, options, out.diagnostics);
    ++out.diagnostics.warmup_iterations;

    if (rank > 0) {
        const RowMajorMatrixXd fallback = leading_orientation(initial_shared, rank);
        model.orientation = dense_accumulation
            ? leading_orientation(
                pooled_residual_covariance(persistent, model), rank)
            : orientation_from_signed_sketch(sketch,
                pooled_residual_sketch(persistent, model, sketch),
                fallback, rank);
        model.low_rank_variances = RowMajorMatrixXd::Constant(
            components, rank, options.low_rank_variance_floor);
        persistent = run_svi_e_step(coordinates, model, nullptr, 0,
            options.n_threads, nullptr, use_sketch, true,
            dense_accumulation);
        run_m_step(persistent, model, alpha, options, out.diagnostics);
        ++out.diagnostics.warmup_iterations;
    } else {
        out.diagnostics.orientation_converged = true;
        out.diagnostics.orientation_change = 0.0;
    }

    if (candidate_enabled) {
        persistent = exact_refresh_in_batches(coordinates, model, options,
            use_sketch, dense_accumulation, previous_top);
        dormancy.observe_exact(persistent.membership, options.min_cluster_size,
            options.prune_patience, false);
        run_m_step(persistent, model, alpha, options, out.diagnostics);
        ++out.diagnostics.warmup_iterations;
        if (options.verbose > 0) {
            log_top_cluster_size_fractions(persistent.membership, n,
                "initial refresh");
        }
    }

    std::vector<int32_t> permutation(n);
    std::iota(permutation.begin(), permutation.end(), int32_t{0});
    const std::vector<int32_t> validation = deterministic_document_sample(
        n, options.svi_eval_size,
        static_cast<int32_t>(static_cast<uint32_t>(options.seed) ^ 0x85ebca6bu));
    RowMajorMatrixXd previous_validation_responsibilities;
    double previous_validation_elbo = -std::numeric_limits<double>::infinity();
    double previous_validation_log_likelihood =
        -std::numeric_limits<double>::infinity();
    int32_t stable_epochs = 0;
    int32_t stable_orientation_updates = 0;
    bool orientation_frozen = rank == 0;

    for (int32_t epoch = 0; epoch < options.n_epochs; ++epoch) {
        std::mt19937 shuffle_engine(
            static_cast<uint32_t>(options.seed)
            + static_cast<uint32_t>(epoch) * 0x9e3779b9u);
        std::shuffle(permutation.begin(), permutation.end(), shuffle_engine);
        const bool refresh_epoch = candidate_enabled
            && (epoch + 1) % options.candidate_refresh_epochs == 0;
        if (refresh_epoch) {
            persistent = exact_refresh_in_batches(coordinates, model, options,
                use_sketch, dense_accumulation, previous_top);
            ++out.diagnostics.svi_updates;
            ++out.diagnostics.full_refreshes;
            dormancy.observe_exact(persistent.membership,
                options.min_cluster_size, options.prune_patience, true);
            run_m_step(persistent, model, alpha, options, out.diagnostics,
                &dormancy.dormant());
        } else {
            for (int32_t begin = 0; begin < n; begin += options.minibatch_size) {
                const int32_t batch_size = std::min(
                    options.minibatch_size, n - begin);
                std::vector<int32_t> candidates;
                const PreparedEStepModel prepared = prepare_e_step_model(
                    model, dense_accumulation);
                if (candidate_enabled) {
                    if (candidate_search == GammaPoissonClusterFitOptions::
                            CandidateSearch::KdTree) {
                        candidates = select_candidate_components_kdtree(
                            coordinates, model, prepared,
                            permutation.data() + begin, batch_size,
                            options.candidate_components, candidate_dimensions,
                            2, previous_top, dormancy.dormant());
                    } else {
                        candidates = select_candidate_components_linear(
                            coordinates, model, prepared,
                            permutation.data() + begin, batch_size,
                            options.candidate_components, candidate_dimensions,
                            previous_top, dormancy.dormant());
                    }
                }
                EStepRequest request;
                request.document_indices = permutation.data() + begin;
                request.n_documents = batch_size;
                request.n_threads = options.n_threads;
                request.sketch = use_sketch;
                request.collect_moments = true;
                request.dense_accumulation = dense_accumulation;
                request.candidates =
                    candidate_enabled ? candidates.data() : nullptr;
                request.candidate_stride =
                    candidate_enabled ? options.candidate_components : 0;
                request.document_top_components =
                    candidate_enabled ? previous_top.data() : nullptr;
                ClusterStatistics batch = run_e_step_prepared(
                    coordinates, model, prepared, request);
                batch.scale(static_cast<double>(n) / batch_size);
                ++out.diagnostics.svi_updates;
                const double rho = std::pow(options.svi_tau0
                    + out.diagnostics.svi_updates, -options.svi_kappa);
                persistent.interpolate(batch, rho);
                const Eigen::VectorXd weight_membership = candidate_enabled
                    ? dormancy.preserve_exact_weights(persistent.membership)
                    : Eigen::VectorXd();
                run_m_step(persistent, model, alpha, options,
                    out.diagnostics,
                    candidate_enabled ? &dormancy.dormant() : nullptr,
                    candidate_enabled ? &weight_membership : nullptr);

                const bool orientation_due = rank > 0 && !orientation_frozen
                    && out.diagnostics.orientation_updates
                        < options.orientation_max_updates
                    && out.diagnostics.svi_updates
                        % options.orientation_update_interval == 0;
                if (orientation_due) {
                    const RowMajorMatrixXd previous = model.orientation;
                    RowMajorMatrixXd candidate = dense_accumulation
                        ? leading_orientation(
                            pooled_residual_covariance(persistent, model), rank)
                        : orientation_from_signed_sketch(sketch,
                            pooled_residual_sketch(persistent, model, sketch),
                            previous, rank);
                    out.diagnostics.orientation_change =
                        align_and_blend_orientation(model.orientation,
                            std::move(candidate), options.orientation_step);
                    transport_orientation(model, previous, options);
                    ++out.diagnostics.orientation_updates;
                    if (options.verbose > 0) {
                        notice("Gamma-Poisson orientation update %d: change = %.3e",
                            out.diagnostics.orientation_updates,
                            out.diagnostics.orientation_change);
                    }
                    stable_orientation_updates =
                        out.diagnostics.orientation_change
                            <= options.orientation_tolerance
                        ? stable_orientation_updates + 1 : 0;
                    orientation_frozen = stable_orientation_updates
                        >= options.orientation_patience;
                    out.diagnostics.orientation_converged = orientation_frozen;
                    if (candidate_enabled) {
                        persistent = exact_refresh_in_batches(coordinates, model,
                            options, use_sketch, dense_accumulation,
                            previous_top);
                        dormancy.observe_exact(persistent.membership,
                            options.min_cluster_size,
                            options.prune_patience, false);
                    } else {
                        persistent = run_svi_e_step(coordinates, model,
                            nullptr, 0, options.n_threads, nullptr, use_sketch,
                            true, dense_accumulation);
                    }
                    run_m_step(persistent, model, alpha, options,
                        out.diagnostics,
                        candidate_enabled ? &dormancy.dormant() : nullptr);
                }
            }
        }

        GammaPoissonClusterModel validation_state{model};
        RowMajorMatrixXd validation_responsibilities;
        ClusterStatistics validation_stats = run_svi_e_step(
            coordinates, validation_state, validation.data(),
            static_cast<int32_t>(validation.size()), options.n_threads,
            &validation_responsibilities, nullptr, false, false);
        validation_stats.scale(
            static_cast<double>(n) / validation.size());
        out.model = model;
        update_diagnostics(validation_stats, alpha, out,
            &validation_responsibilities);
        ++out.diagnostics.epochs;
        const bool have_change = previous_validation_responsibilities.rows()
            == validation_responsibilities.rows();
        bool converged_now = false;
        if (have_change) {
            const GammaPoissonResponsibilityChange change =
                gamma_poisson_responsibility_change(
                    previous_validation_responsibilities,
                    validation_responsibilities);
            out.diagnostics.mean_responsibility_linf_change = change.mean_linf;
            out.diagnostics.p90_responsibility_linf_change = change.p90_linf;
            out.diagnostics.top_assignment_change_fraction =
                change.top_assignment_fraction;
            out.diagnostics.relative_elbo_change = std::abs(
                out.diagnostics.elbo - previous_validation_elbo)
                / (1.0 + std::abs(previous_validation_elbo));
            out.diagnostics.relative_predictive_log_likelihood_change = std::abs(
                out.diagnostics.log_likelihood
                    - previous_validation_log_likelihood)
                / (1.0 + std::abs(previous_validation_log_likelihood));
            const bool stable =
                out.diagnostics.relative_predictive_log_likelihood_change
                    <= options.tolerance
                && out.diagnostics.p90_responsibility_linf_change
                    <= options.responsibility_p90_tolerance
                && out.diagnostics.top_assignment_change_fraction
                    <= options.top_assignment_change_tolerance;
            stable_epochs = stable ? stable_epochs + 1 : 0;
            const int32_t minimum_epochs = candidate_enabled
                ? std::max(5, options.candidate_refresh_epochs) : 5;
            converged_now = epoch + 1 >= minimum_epochs
                && stable_epochs >= options.convergence_patience;
        }
        if (options.verbose > 0) {
            std::string label;
            if (refresh_epoch) {
                label = "full refresh "
                    + std::to_string(out.diagnostics.full_refreshes);
            } else if (!candidate_enabled
                && ((epoch + 1) % options.verbose == 0 || converged_now)) {
                label = "epoch " + std::to_string(epoch + 1);
            }
            if (!label.empty()) {
                const ClusterConvergenceStats conv{
                    out.diagnostics.relative_predictive_log_likelihood_change,
                    out.diagnostics.p90_responsibility_linf_change,
                    out.diagnostics.top_assignment_change_fraction, "dPredLL"};
                log_top_cluster_size_fractions(persistent.membership, n, label,
                    have_change ? &conv : nullptr);
            }
        }
        if (converged_now) {
            out.diagnostics.svi_converged = true;
            break;
        }
        previous_validation_responsibilities =
            std::move(validation_responsibilities);
        previous_validation_elbo = out.diagnostics.elbo;
        previous_validation_log_likelihood = out.diagnostics.log_likelihood;
    }

    out.diagnostics.converged = false;
    run_exact_em(coordinates, model, options, alpha, dense_accumulation,
        options.refine_max_iterations, true, out);
    out.diagnostics.structured_iterations = out.diagnostics.svi_updates
        + out.diagnostics.refinement_iterations;
    out.diagnostics.iterations = out.diagnostics.warmup_iterations
        + out.diagnostics.structured_iterations;
    return out;
}


} // namespace gamma_pois_cluster_detail
