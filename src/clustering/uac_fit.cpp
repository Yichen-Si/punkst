#include "clustering/uac_cache_internal.hpp"
#include "clustering/uac_initialization_internal.hpp"
#include "clustering/uac_stochastic_internal.hpp"

#include "clustering_core/cosine_clustering.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <tbb/global_control.h>

namespace uac::detail {


FitResult fit_impl(const Dataset& data, Dataset* mutable_data,
    const Basis* basis, const FitOptions& options,
    const IndexedDocumentSource* count_source,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    validate_dataset(data,
        options.handoff == HandoffMode::Particle && !count_source);
    validate_component_screening(options.component_screening);
    const int64_t total_starts = static_cast<int64_t>(options.kmeans_starts)
        + options.leiden_starts;
    const bool valid_initialization_metric =
        options.initialization_metric == SimplexMetric::Cosine
        || options.initialization_metric == SimplexMetric::Hellinger;
    if (options.factor_diagonal_mode == FactorDiagonalMode::Shared
        && options.cluster_covariance_rank < 0) {
        throw std::invalid_argument(
            "Shared cluster covariance diagonal requires --cluster-covariance-rank >= 0");
    }
    if (options.n_components <= 0 || options.kmeans_starts < 0
        || options.leiden_starts < 0 || total_starts <= 0
        || options.max_iterations <= 0 || options.n_particles <= 0
        || options.streaming.block_documents <= 0
        || options.particle_em_fixed_iterations < 0
        || options.cluster_covariance_rank < -1
        || options.kmeans_max_iterations <= 0
        || data.centers.rows() < options.n_components
        || data.coordinates.rows() != data.centers.rows()
        || !(options.objective_change_tolerance > 0.0)
        || !(options.responsibility_change_tolerance > 0.0)
        || !(options.particle_variance_change_tolerance >= 0.0)
        || !std::isfinite(options.particle_variance_change_tolerance)
        || !(options.initialization_ridge_precision >= 0.0)
        || !std::isfinite(options.initialization_ridge_precision)
        || options.initialization_measurement_target <= 0
        || options.initialization_candidate_score_target < -1
        || options.initialization_sampling_seed < -1
        || !(options.target_relative_floor > 0.0)
        || !(options.covariance_floor > 0.0)
        || !(options.covariance_shrinkage_strength >= 0.0)
        || !std::isfinite(options.covariance_shrinkage_strength)
        || !(options.fisher_broadening > 0.0)
        || !std::isfinite(options.fisher_broadening)
        || options.fisher_refinement_iterations <= 0
        || !(options.fit_document_budget >= 0.0)
        || !std::isfinite(options.fit_document_budget)
        || options.fit_full_tail_updates < 0
        || options.fit_subsample_target <= 0
        || !(options.fit_subsample_base_fraction >= 0.0)
        || !(options.fit_subsample_base_fraction < 1.0)
        || !std::isfinite(options.fit_subsample_base_fraction)
        || options.fit_subsample_memory_budget == 0
        || !(options.fit_subsample_safety_factor >= 1.0)
        || !std::isfinite(options.fit_subsample_safety_factor)
        || options.fit_subsample_min_updates <= 0
        || options.fit_subsample_max_updates < options.fit_subsample_min_updates
        || !(options.fit_subsample_change_tolerance > 0.0)
        || !std::isfinite(options.fit_subsample_change_tolerance)
        || options.fit_subsample_topup_rounds < 0
        || options.fit_batch_documents <= 0
        || !(options.fit_step_kappa > 0.5)
        || !(options.fit_step_kappa <= 1.0)
        || !std::isfinite(options.fit_step_kappa)
        || !(options.fit_step_initial > 0.0)
        || !(options.fit_step_initial <= 1.0)
        || !std::isfinite(options.fit_step_initial)
        || !valid_initialization_metric) {
        throw std::invalid_argument("Invalid UAC fit options or dataset");
    }
    if (options.leiden_starts > 0
        && (options.leiden_neighbors <= 0
            || options.leiden_neighbors >= data.centers.rows()
            || options.leiden_max_iterations == 0
            || !(options.leiden_knn_epsilon >= 0.0)
            || !std::isfinite(options.leiden_knn_epsilon)
            || options.leiden_hnsw_m <= 0
            || options.leiden_hnsw_ef_construction <= 0
            || options.leiden_hnsw_ef_search < 0
            || options.leiden_hnsw_max_ef_search <= 0
            || options.leiden_hnsw_candidates < 0
            || options.leiden_hnsw_audit_queries <= 0
            || !(options.leiden_hnsw_recall > 0.0
                && options.leiden_hnsw_recall <= 1.0)
            || !std::isfinite(options.leiden_hnsw_recall)
            || options.leiden_nndescent_iterations < 0
            || options.leiden_nndescent_graph_size < 0
            || options.leiden_nndescent_sample_candidates <= 0
            || options.leiden_nndescent_audit_queries <= 0
            || !(options.leiden_nndescent_recall > 0.0
                && options.leiden_nndescent_recall <= 1.0)
            || !std::isfinite(options.leiden_nndescent_recall)
            || !(options.leiden_resolution > 0.0)
            || !std::isfinite(options.leiden_resolution))) {
        throw std::invalid_argument("Invalid UAC Leiden start options");
    }
    if (options.handoff == HandoffMode::Particle
        && (basis == nullptr
            || (!count_source
                && data.counts.size() != data.identifiers.size())
            || (count_source
                && count_source->documents()
                    != static_cast<int64_t>(data.identifiers.size()))
            || (count_source && basis
                && count_source->features()
                    != basis->probabilities.rows()))) {
        throw std::invalid_argument("Particle UAC requires basis and aligned counts");
    }
    if (count_source
        && options.particle_engine != ParticleEngine::Stream) {
        throw std::invalid_argument(
            "Indexed UAC counts require the stream particle engine");
    }
    if (basis) {
        validate_basis(*basis,
            checked_int32(data.centers.cols(), "topic count"));
        if (!count_source) validate_count_features(data, *basis);
    }
    validate_adaptive_particles(
        options.adaptive_particles, options.n_particles);
    if (options.particle_initial_model.has_value()
        && options.handoff != HandoffMode::Particle) {
        throw std::invalid_argument(
            "Particle initial model requires particle handoff");
    }
    if (options.initialization_only
        && options.handoff != HandoffMode::Particle) {
        throw std::invalid_argument(
            "Initialization-only UAC requires particle handoff");
    }
    if (options.particle_em_fixed_iterations > 0
        && options.handoff != HandoffMode::Particle) {
        throw std::invalid_argument(
            "Fixed particle EM iterations require particle handoff");
    }
    if (options.particle_variance_change_tolerance > 0.0
        && options.handoff != HandoffMode::Particle) {
        throw std::invalid_argument(
            "Particle variance convergence requires particle handoff");
    }
    if (options.particle_em_fixed_iterations > 0
        && options.particle_variance_change_tolerance > 0.0) {
        throw std::invalid_argument(
            "Fixed particle EM iterations cannot use convergence stopping");
    }
    if (options.particle_engine == ParticleEngine::Stream
        && options.handoff != HandoffMode::Particle) {
        throw std::invalid_argument(
            "The UAC stream particle engine requires particle handoff");
    }
    if (options.particle_fit_schedule != ParticleFitSchedule::Exact
        && options.handoff != HandoffMode::Particle) {
        throw std::invalid_argument(
            "Approximate UAC particle fitting requires particle handoff");
    }
    if (options.particle_fit_schedule == ParticleFitSchedule::Online
        && options.particle_engine != ParticleEngine::Stream) {
        throw std::invalid_argument(
            "Online UAC particle fitting requires the stream engine");
    }
    if (options.particle_fit_schedule == ParticleFitSchedule::Subsample
        && options.particle_engine == ParticleEngine::Batch
        && options.fit_subsample_storage == SubsampleStorage::Disk) {
        throw std::invalid_argument(
            "Batch UAC subsample fitting does not support disk subsample storage");
    }
    if (options.particle_fit_schedule == ParticleFitSchedule::Online
        && !(options.fit_document_budget > 0.0)) {
        throw std::invalid_argument(
            "Online UAC particle fitting requires a positive document budget");
    }
    if (options.particle_fit_schedule != ParticleFitSchedule::Exact
        && (options.particle_em_fixed_iterations > 0
            || options.particle_variance_change_tolerance > 0.0)) {
        throw std::invalid_argument(
            "Approximate UAC particle fitting uses its document budget, not exact-EM stopping options");
    }
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, options.n_threads));
    const auto initialization_start =
        std::chrono::steady_clock::now();
    FitResult result;
    result.initialization.measurement_mode =
        options.initialization_measurement_mode;
    result.initialization.total_documents = data.coordinates.rows();
    result.initialization.sampling_seed =
        options.initialization_sampling_seed >= 0
        ? options.initialization_sampling_seed : options.seed;
    result.initialization.measurement_target =
        options.initialization_measurement_mode
                == InitializationMeasurementMode::HorvitzThompson
        ? options.initialization_measurement_target : 0;
    result.initialization.candidate_score_target =
        options.initialization_measurement_mode
                == InitializationMeasurementMode::Legacy
        ? 0
        : options.initialization_candidate_score_target >= 0
            ? options.initialization_candidate_score_target
            : options.initialization_measurement_mode
                    == InitializationMeasurementMode::Full
                ? 512 : options.initialization_measurement_target;
    struct StartPartition {
        Eigen::VectorXi raw_assignments;
        Eigen::VectorXi assignments;
        RestartTrace metadata;
    };
    std::vector<StartPartition> starts;
    starts.reserve(static_cast<size_t>(total_starts));
    auto append_start = [&](Eigen::VectorXi raw_assignments,
                            Eigen::VectorXi assignments,
                            RestartTrace metadata) {
        metadata.handoff = options.handoff;
        metadata.phase = options.handoff == HandoffMode::Particle
            ? TracePhase::CorrectedMomScore : TracePhase::PointMapEm;
        starts.push_back({
            std::move(raw_assignments), std::move(assignments),
            std::move(metadata)});
    };

    int32_t global_start = 0;
    for (int32_t start = 0; start < options.kmeans_starts;
            ++start, ++global_start) {
        RestartTrace metadata;
        metadata.start = global_start;
        metadata.start_method = StartMethod::KMeans;
        metadata.seed = map_start_seed(options.seed, global_start);
        metadata.raw_communities = options.n_components;
        DenseKMeansOptions kmeans;
        kmeans.n_clusters = options.n_components;
        kmeans.max_iterations = options.kmeans_max_iterations;
        kmeans.seed = metadata.seed;
        DenseKMeansResult clustering = simplex_dense_kmeans(
            data.centers, options.initialization_metric, kmeans);
        append_start({}, std::move(clustering.assignments), metadata);
    }

    if (options.leiden_starts > 0) {
        CosineKnnOptions knn_options;
        knn_options.n_neighbors = options.leiden_neighbors;
        knn_options.knn_search_epsilon = options.leiden_knn_epsilon;
        knn_options.backend = options.leiden_knn_backend;
        knn_options.n_threads = options.n_threads;
        knn_options.hnsw_m = options.leiden_hnsw_m;
        knn_options.hnsw_ef_construction =
            options.leiden_hnsw_ef_construction;
        knn_options.hnsw_ef_search = options.leiden_hnsw_ef_search;
        knn_options.hnsw_max_ef_search =
            options.leiden_hnsw_max_ef_search;
        knn_options.hnsw_candidates = options.leiden_hnsw_candidates;
        knn_options.hnsw_audit_queries =
            options.leiden_hnsw_audit_queries;
        knn_options.hnsw_recall = options.leiden_hnsw_recall;
        knn_options.hnsw_force = options.leiden_hnsw_force;
        knn_options.nndescent_iterations =
            options.leiden_nndescent_iterations;
        knn_options.nndescent_graph_size =
            options.leiden_nndescent_graph_size;
        knn_options.nndescent_sample_candidates =
            options.leiden_nndescent_sample_candidates;
        knn_options.nndescent_audit_queries =
            options.leiden_nndescent_audit_queries;
        knn_options.nndescent_recall = options.leiden_nndescent_recall;
        knn_options.ann_seed = options.seed;
        const CosineKnnResult knn = simplex_knn(
            data.centers, options.initialization_metric, knn_options);
        result.has_leiden_knn_diagnostics = true;
        result.leiden_knn_diagnostics = knn.diagnostics;
        double resolution = options.leiden_resolution;
        double last_under_resolution = 0.0;
        bool adapting = true;
        for (int32_t start = 0; start < options.leiden_starts;
                ++start, ++global_start) {
            RestartTrace metadata;
            metadata.start = global_start;
            metadata.start_method = StartMethod::Leiden;
            metadata.seed = map_start_seed(options.seed, global_start);
            metadata.leiden_resolution = resolution;
            LeidenOptions leiden_options;
            leiden_options.resolution = resolution;
            leiden_options.max_iterations = options.leiden_max_iterations;
            leiden_options.seed = metadata.seed;
            LeidenResult leiden = leiden_cluster(knn.graph.n_nodes,
                knn.graph.edges, knn.graph.weights, leiden_options);
            metadata.raw_communities = leiden.n_communities;
            metadata.reconciliation_count = std::abs(
                leiden.n_communities - options.n_components);
            DenseKMeansOptions reconcile_options;
            reconcile_options.n_clusters = options.n_components;
            reconcile_options.max_iterations = options.kmeans_max_iterations;
            reconcile_options.seed = metadata.seed;
            Eigen::VectorXi assignments = reconcile_simplex_communities(
                leiden.membership, leiden.n_communities,
                options.n_components, data.centers,
                options.initialization_metric, reconcile_options);
            append_start(std::move(leiden.membership),
                std::move(assignments), metadata);

            if (!adapting) continue;
            if (leiden.n_communities < options.n_components) {
                last_under_resolution = resolution;
                resolution = increased_leiden_resolution(resolution,
                    leiden.n_communities, options.n_components);
            } else if (leiden.n_communities == options.n_components) {
                adapting = false;
            } else {
                if (last_under_resolution > 0.0) {
                    resolution = midpoint_leiden_resolution(
                        last_under_resolution, resolution);
                }
                adapting = false;
            }
        }
    }

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(total_starts));
    Eigen::MatrixXd initialization_precision;
    if (helmert.rows() != data.coordinates.cols()
        || helmert.cols() != data.centers.cols()
        || !is_normalized_helmert(helmert)) {
        throw std::invalid_argument("Invalid UAC Helmert basis");
    }
    std::vector<HardPartitionMoments> partition_moments;
    std::vector<std::vector<Eigen::MatrixXd>> measurement_sums;
    InitializationMeasurements initialization_measurements;
    if (options.handoff == HandoffMode::Particle) {
        const auto partition_start = std::chrono::steady_clock::now();
        partition_moments.reserve(starts.size());
        for (const auto& start : starts) {
            partition_moments.push_back(hard_partition_moments(
                data, start.assignments, options.n_components));
        }
        initialization_precision = shared_measurement_precision(
            partition_moments, options.initialization_ridge_precision,
            options.target_relative_floor);
        result.initialization.partition_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - partition_start).count();
        std::vector<Eigen::VectorXi> assignments;
        assignments.reserve(starts.size());
        for (const auto& start : starts) {
            assignments.push_back(start.assignments);
        }
        if (options.initialization_measurement_mode
                == InitializationMeasurementMode::Legacy) {
            const auto measurement_start = std::chrono::steady_clock::now();
            measurement_sums = measurement_sums_by_partition(
                data, *basis, helmert, initialization_precision, assignments,
                options.n_components, options.proposal, count_source);
            result.initialization.measurement_seconds =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now()
                    - measurement_start).count();
            result.initialization.measurement_documents =
                data.coordinates.rows();
            result.initialization.candidate_score_documents =
                data.coordinates.rows();
            result.initialization.minimum_measurement_effective_size =
                partition_moments.front().counts.minCoeff();
            result.initialization.minimum_candidate_score_effective_size =
                result.initialization.minimum_measurement_effective_size;
        } else {
            initialization_measurements = collect_initialization_measurements(
                data, *basis, helmert, initialization_precision, assignments,
                partition_moments, options.n_components, options.proposal,
                options.initialization_measurement_mode,
                options.initialization_measurement_target,
                result.initialization.candidate_score_target,
                result.initialization.sampling_seed ^ 0x49ac2d1,
                count_source);
            measurement_sums = initialization_measurements.sums;
            result.initialization.measurement_seconds =
                initialization_measurements.seconds;
            result.initialization.measurement_documents =
                initialization_measurements.measurement_documents;
            result.initialization.candidate_score_documents =
                initialization_measurements.score_documents.size();
            result.initialization.measurement_covariance_evaluations =
                initialization_measurements.covariance_evaluations;
            result.initialization.cached_measurement_bytes =
                initialization_measurements.cache_bytes;
            result.initialization.maximum_measurement_weight =
                initialization_measurements.maximum_measurement_weight;
            result.initialization.maximum_candidate_score_weight =
                initialization_measurements.maximum_score_weight;
            result.initialization.minimum_measurement_effective_size =
                initialization_measurements.minimum_measurement_effective_size;
            result.initialization.minimum_candidate_score_effective_size =
                initialization_measurements.minimum_score_effective_size;
        }
    }
    for (size_t i = 0; i < starts.size(); ++i) {
        const auto& start = starts[i];
        try {
            const double shrinkage =
                options.adaptive_covariance_shrinkage
                ? options.covariance_shrinkage_strength : 0.0;
            if (options.handoff == HandoffMode::Map) {
                Model initial = initialize_model_from_partition(
                    data, start.assignments, options.n_components,
                    shrinkage, options.covariance_floor,
                    options.target_relative_floor);
                candidates.push_back(fit_map_candidate(
                    data, std::move(initial), options, start.metadata));
            } else {
                Candidate candidate;
                candidate.trace = start.metadata;
                candidate.model = initialize_model_from_corrected_moments(
                    data, start.assignments, partition_moments[i],
                    measurement_sums[i], shrinkage,
                    options.covariance_floor,
                    &result.initialization.covariance_floor_activations);
                candidates.push_back(std::move(candidate));
            }
        } catch (const std::exception&) {
            Candidate failed;
            failed.trace = start.metadata;
            failed.trace.collapsed = true;
            candidates.push_back(std::move(failed));
        }
    }
    result.initialization_partitions.reserve(starts.size());
    for (auto& start : starts) {
        result.initialization_partitions.push_back({
            start.metadata.start, start.metadata.start_method,
            std::move(start.assignments),
            std::move(start.raw_assignments)});
    }
    if (options.handoff == HandoffMode::Particle) {
        const auto candidate_score_start = std::chrono::steady_clock::now();
        if (options.initialization_measurement_mode
                == InitializationMeasurementMode::Legacy) {
            score_corrected_moment_candidates(data, *basis, helmert,
                initialization_precision, options, candidates, count_source);
            result.initialization_measurement_covariance_evaluations =
                2 * static_cast<int64_t>(data.coordinates.rows());
            result.initialization.measurement_covariance_evaluations =
                result.initialization_measurement_covariance_evaluations;
        } else {
            score_corrected_moment_candidates(
                data, options, initialization_measurements, candidates);
            result.initialization_measurement_covariance_evaluations =
                initialization_measurements.covariance_evaluations;
        }
        result.initialization.candidate_score_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now()
                - candidate_score_start).count();
    }

    Candidate* selected = nullptr;
    for (auto& candidate : candidates) {
        if (candidate.trace.collapsed || !std::isfinite(candidate.objective)) {
            continue;
        }
        if (selected == nullptr || candidate.objective > selected->objective
            || (candidate.objective == selected->objective
                && candidate.trace.start < selected->trace.start)) {
            selected = &candidate;
        }
    }
    if (selected == nullptr) {
        throw std::runtime_error(
            "Every UAC initialization start failed numerically");
    }
    selected->trace.selected = true;
    result.traces.reserve(candidates.size() + 1);
    for (const auto& candidate : candidates) {
        result.traces.push_back(candidate.trace);
    }
    ComponentScreeningOptions selected_map_screening =
        options.component_screening;
    if (options.handoff == HandoffMode::Particle) {
        selected_map_screening.mode = ComponentScreeningMode::Off;
    } else if (selected_map_screening.mode
            == ComponentScreeningMode::Auto) {
        const bool enabled = resolve_map_component_screening(
            data, selected->model, selected_map_screening,
            static_cast<uint64_t>(selected->trace.seed));
        apply_auto_component_screening_resolution(
            selected_map_screening, enabled);
    }
    if (options.handoff == HandoffMode::Map) {
        const Expectation selected_expectation = map_expectation(
            data, selected->model, ExpectationRequest{false, false, true},
            selected_map_screening);
        result.pilot = pilot_from_map(data, selected->model,
            selected_expectation, options.target_relative_floor);
    } else {
        result.pilot = pilot_from_model(selected->model);
    }
    selected->model.shrinkage_target = result.pilot.pooled_covariance;
    if (options.cluster_covariance_rank >= 0) {
        const int32_t dimension = static_cast<int32_t>(
            selected->model.means.cols());
        const int32_t rank = options.cluster_covariance_rank;
        if (rank > dimension) {
            throw std::invalid_argument(
                "UAC factor rank exceeds the ILR dimension");
        }
        convert_model_to_factor(selected->model, rank,
            options.factor_diagonal_mode, options.covariance_floor);
        if (options.handoff == HandoffMode::Map
            && options.factor_diagonal_mode
                == FactorDiagonalMode::Shared) {
            RestartTrace refinement_metadata = selected->trace;
            refinement_metadata.points.clear();
            refinement_metadata.model_trace.clear();
            refinement_metadata.completed_updates = 0;
            refinement_metadata.converged = false;
            refinement_metadata.collapsed = false;
            refinement_metadata.selected = true;
            Candidate refined = fit_map_candidate(data, selected->model,
                options, refinement_metadata);
            if (refined.trace.collapsed || !refined.trace.succeeded) {
                throw std::runtime_error(
                    "Selected UAC initializer collapsed during shared-factor MAP refinement");
            }
            selected->model = std::move(refined.model);
            selected->trace.converged = refined.trace.converged;
            result.traces.push_back(std::move(refined.trace));
            result.pilot = pilot_from_model(selected->model);
        } else if (options.handoff == HandoffMode::Map) {
            const double shrinkage =
                options.adaptive_covariance_shrinkage
                ? options.covariance_shrinkage_strength : 0.0;
            const double before = map_expectation(data, selected->model,
                ExpectationRequest{false, false, false},
                selected_map_screening)
                .log_likelihood
                + covariance_prior(selected->model, shrinkage);
            Model refined = selected->model;
            const Expectation refinement = map_expectation(data, refined,
                ExpectationRequest{false, false, true},
                selected_map_screening);
            const ModelUpdate refinement_update = update_model(
                refined, refinement, shrinkage,
                options.covariance_floor);
            if (refinement_update.valid) {
                const double after = map_expectation(data, refined,
                    ExpectationRequest{false, false, false},
                    selected_map_screening)
                    .log_likelihood
                    + covariance_prior(refined, shrinkage);
                if (std::isfinite(after) && after >= before) {
                    selected->model = std::move(refined);
                }
            }
        }
    }
    result.selected_start = selected->trace.start;
    result.selected_start_method = selected->trace.start_method;
    result.selected_leiden_resolution = selected->trace.leiden_resolution;
    const double initialization_seconds =
        std::chrono::duration<double>(
            std::chrono::steady_clock::now()
            - initialization_start).count();
    result.initialization.total_seconds = initialization_seconds;

    if (options.initialization_only) {
        result.model = selected->model;
        result.score.initialization_seconds = initialization_seconds;
        result.converged = false;
        return result;
    }

    if (options.handoff == HandoffMode::Map) {
        result.model = selected->model;
        ComponentScreeningOptions terminal_screening =
            options.component_screening;
        if (options.exact_final_score) {
            terminal_screening.mode = ComponentScreeningMode::Off;
            terminal_screening.maximum_components = 0;
        }
        result.score = score_map(data, result.model, options.n_threads,
            terminal_screening);
        result.score.component_screening_options =
            options.component_screening;
        result.score.map_component_screening =
            selected_map_screening.mode == ComponentScreeningMode::On;
        result.score.exact_final_score = options.exact_final_score;
        result.score.initialization_seconds = initialization_seconds;
        result.converged = selected->trace.converged;
        return result;
    }
    Model particle_initial = selected->model;
    if (options.particle_initial_model.has_value()) {
        validate_particle_initial_model(
            *options.particle_initial_model, selected->model);
        particle_initial = *options.particle_initial_model;
    }
    const PilotCache pilot_cache(result.pilot);
    const uint64_t particle_seed = static_cast<uint64_t>(options.seed) ^ 0xF604;
    const ProposalScreeningPlan proposal_screening =
        make_proposal_screening_plan(data, *basis, helmert, result.pilot,
            pilot_cache, options.proposal, options.fisher_broadening,
            options.fisher_refinement_iterations, particle_seed,
            options.component_screening, count_source);
    ComponentScreeningOptions particle_screening =
        options.component_screening;
    if (particle_screening.mode == ComponentScreeningMode::Auto) {
        apply_auto_component_screening_resolution(
            particle_screening, false);
    }
    Candidate particle;
    FitScheduleDiagnostics fit_schedule;
    std::optional<ScoreResult> approximate_terminal_score;
    std::optional<Expectation> approximate_terminal_expectation;
    fit_schedule.schedule = options.particle_fit_schedule;
    try {
        if (options.particle_engine == ParticleEngine::Stream) {
            for (int32_t attempt = 0; attempt < 2; ++attempt) {
                try {
                    StreamingOptions streaming = options.streaming;
                    if (options.particle_fit_schedule
                            == ParticleFitSchedule::Subsample) {
                        const int32_t components = static_cast<int32_t>(
                            particle_initial.weights.size());
                        const int32_t dimension = static_cast<int32_t>(
                            data.coordinates.cols());
                        const int32_t factor_rank =
                            particle_initial.covariance_kind
                                == CovarianceKind::FactorAnalytic
                            ? static_cast<int32_t>(particle_initial
                                .factor_covariances.front().factor.cols())
                            : -1;
                        const uint64_t base_bytes =
                            (sizeof(uint16_t) + sizeof(uint8_t)
                                + sizeof(int32_t))
                                * static_cast<uint64_t>(
                                    data.coordinates.rows())
                            + sizeof(double)
                                * static_cast<uint64_t>(components)
                                * (components + 16)
                            + expectation_block_bytes(components,
                                dimension, factor_rank)
                                * std::max(1, options.n_threads);
                        if (base_bytes >= options.fit_subsample_memory_budget) {
                            throw std::runtime_error(
                                "UAC subsample labels and E-step workspace exceed the memory budget");
                        }
                        const uint64_t per_document =
                            sizeof(double)
                                * static_cast<uint64_t>(options.n_particles)
                                * (dimension + 2)
                            + sizeof(int32_t)
                                * static_cast<uint64_t>(
                                    options.n_particles + 1)
                            + sizeof(int64_t);
                        const uint64_t available =
                            options.fit_subsample_memory_budget - base_bytes;
                        const uint64_t block_limit = available
                            / (std::max(1, options.n_threads)
                                * std::max<uint64_t>(1, per_document));
                        streaming.block_documents = std::min(
                            streaming.block_documents,
                            static_cast<int32_t>(std::max<uint64_t>(
                                1, std::min<uint64_t>(
                                    block_limit,
                                    std::numeric_limits<int32_t>::max()))));
                    }
                    if (attempt > 0) streaming.rebuild_cache = true;
                    ParticleCache cache = open_or_build_particle_cache(
                        data, *basis, helmert, result.pilot, pilot_cache,
                        options.proposal, options.n_particles, particle_seed,
                        options.fisher_broadening,
                        options.fisher_refinement_iterations,
                        options.n_threads,
                        particle_initial, options.adaptive_particles,
                        &proposal_screening, options.component_screening,
                        streaming, count_source);
                    if (mutable_data) {
                        mutable_data->centers = RowMajorMatrixXd{};
                    }
                    particle_screening = options.component_screening;
                    if (options.component_screening.mode
                            == ComponentScreeningMode::Auto) {
                        apply_auto_component_screening_resolution(
                            particle_screening,
                            cache.auto_screening_enabled());
                    }
                    if (options.particle_fit_schedule
                            == ParticleFitSchedule::Exact) {
                        CachedResponsibilityState responsibility_state(
                            cache.work_directory());
                        auto expectation_function = [&](const Model& model) {
                            return cached_particle_expectation(cache, model,
                                particle_screening,
                                ExpectationRequest{false, false, true},
                                options.n_threads,
                                nullptr, &responsibility_state);
                        };
                        const auto fit_start =
                            std::chrono::steady_clock::now();
                        particle = fit_particle_candidate(
                            expectation_function, particle_initial, options,
                            selected->trace);
                        fit_schedule.full_data_evaluations =
                            particle.trace.completed_updates;
                        fit_schedule.fitting_seconds =
                            std::chrono::duration<double>(
                                std::chrono::steady_clock::now()
                                - fit_start).count();
                    } else {
                        ApproximateParticleFit approximate =
                            fit_cached_particle_approximate(cache,
                                particle_initial, options,
                                particle_screening, selected->trace);
                        particle = std::move(approximate.candidate);
                        fit_schedule = approximate.diagnostics;
                        approximate_terminal_score =
                            std::move(approximate.terminal_score);
                        approximate_terminal_expectation =
                            std::move(approximate.terminal_expectation);
                    }
                    if (!particle.trace.collapsed) {
                        ComponentScreeningOptions terminal_screening =
                            particle_screening;
                        if (options.exact_final_score
                            || options.particle_fit_schedule
                                != ParticleFitSchedule::Exact) {
                            terminal_screening.mode =
                                ComponentScreeningMode::Off;
                            terminal_screening.maximum_components = 0;
                        }
                        const auto score_start =
                            std::chrono::steady_clock::now();
                        Expectation terminal_expectation;
                        const bool precomputed_terminal =
                            approximate_terminal_score.has_value()
                            && approximate_terminal_expectation.has_value();
                        if (precomputed_terminal) {
                            result.score = std::move(
                                *approximate_terminal_score);
                            terminal_expectation = std::move(
                                *approximate_terminal_expectation);
                        } else {
                            result.score = score_particle_cache(
                                cache, particle.model, terminal_screening,
                                options.streaming.count_storage
                                    == StreamingCountStorage::Memory
                                    && options.particle_fit_schedule
                                        != ParticleFitSchedule::Subsample,
                                options.n_threads, &terminal_expectation);
                            if (options.particle_fit_schedule
                                    == ParticleFitSchedule::Subsample) {
                                ++fit_schedule.subsample_full_cache_scans;
                                fit_schedule.subsample_read_bytes +=
                                    cache.storage_bytes();
                            }
                            ++fit_schedule.full_data_evaluations;
                        }
                        finalize_particle_candidate(
                            particle, terminal_expectation, options);
                        add_screening_metrics(result.score,
                            options.component_screening, proposal_screening,
                            particle_screening);
                        result.score.exact_final_score =
                            options.exact_final_score
                            || options.particle_fit_schedule
                                != ParticleFitSchedule::Exact;
                        result.score.map_component_screening =
                            selected_map_screening.mode
                            == ComponentScreeningMode::On;
                        result.score.adaptive_particle_options =
                            options.adaptive_particles;
                        result.score.streaming_count_storage =
                            options.streaming.count_storage;
                        if (count_source) {
                            result.score.streaming_count_spool_bytes =
                                count_source->storage_bytes();
                            result.score.streaming_peak_count_block_bytes =
                                count_source->peak_block_bytes();
                            result.score.streaming_external_count_parses = 1;
                        }
                        if (!precomputed_terminal) {
                            result.score.scoring_seconds =
                                std::chrono::duration<double>(
                                    std::chrono::steady_clock::now()
                                    - score_start).count();
                            fit_schedule.fitting_seconds +=
                                result.score.scoring_seconds;
                        }
                        fit_schedule.document_pass_equivalents =
                            fit_schedule.full_data_evaluations
                            + static_cast<double>(
                                fit_schedule.approximate_documents)
                                / data.coordinates.rows();
                        result.score.fit_schedule = fit_schedule;
                    }
                    break;
                } catch (const ParticleCacheCorruption&) {
                    if (attempt > 0) throw;
                    if (mutable_data && mutable_data->centers.rows() == 0) {
                        mutable_data->centers = ilr_inverse(
                            mutable_data->coordinates, helmert);
                    }
                }
            }
            if (options.streaming.count_storage
                    == StreamingCountStorage::Source
                && mutable_data) {
                mutable_data->counts.clear();
                mutable_data->counts.shrink_to_fit();
            }
        } else if (options.adaptive_particles.enabled()) {
            const RaggedParticleSet particles = make_adaptive_particles(
                data, *basis, helmert, result.pilot, pilot_cache,
                options.proposal, particle_seed, options.fisher_broadening,
                options.fisher_refinement_iterations, options.n_threads,
                particle_initial,
                options.adaptive_particles, options.n_particles,
                &proposal_screening);
            if (mutable_data) {
                mutable_data->centers = RowMajorMatrixXd{};
            }
            if (options.component_screening.mode
                    == ComponentScreeningMode::Auto) {
                const bool enabled = resolve_particle_component_screening(
                    particles, particle_initial,
                    options.component_screening,
                    proposal_screening.audit_documents);
                apply_auto_component_screening_resolution(
                    particle_screening, enabled);
            }
            if (options.particle_fit_schedule
                    == ParticleFitSchedule::Subsample) {
                ApproximateParticleFit approximate = fit_particle_subsample(
                    particles, particle_initial, options,
                    particle_screening, selected->trace);
                particle = std::move(approximate.candidate);
                fit_schedule = std::move(approximate.diagnostics);
            } else {
                auto expectation_function = [&](const Model& model) {
                    return particle_expectation(particles, model,
                        ExpectationRequest{true, false, true},
                        particle_screening);
                };
                particle = fit_particle_candidate(expectation_function,
                    particle_initial, options, selected->trace);
            }
            if (!particle.trace.collapsed) {
                ComponentScreeningOptions terminal_screening =
                    particle_screening;
                if (options.exact_final_score
                    || options.particle_fit_schedule
                        != ParticleFitSchedule::Exact) {
                    terminal_screening.mode =
                        ComponentScreeningMode::Off;
                    terminal_screening.maximum_components = 0;
                }
                const auto score_start = std::chrono::steady_clock::now();
                Expectation terminal_expectation;
                result.score = score_particles(
                    particles, particle.model, terminal_screening,
                    &terminal_expectation);
                if (options.particle_fit_schedule
                        == ParticleFitSchedule::Exact) {
                    fit_schedule.full_data_evaluations =
                        particle.trace.completed_updates + 1;
                } else {
                    ++fit_schedule.full_data_evaluations;
                }
                finalize_particle_candidate(
                    particle, terminal_expectation, options);
                add_screening_metrics(result.score,
                    options.component_screening, proposal_screening,
                    particle_screening);
                result.score.exact_final_score =
                    options.exact_final_score
                    || options.particle_fit_schedule
                        != ParticleFitSchedule::Exact;
                result.score.fit_schedule = fit_schedule;
                result.score.map_component_screening =
                    selected_map_screening.mode
                    == ComponentScreeningMode::On;
                result.score.adaptive_particle_options =
                    options.adaptive_particles;
                result.score.particle_generation_seconds =
                    particles.calibration_seconds
                    + particles.sampling_seconds + particles.likelihood_seconds;
                result.score.scoring_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - score_start).count();
                fit_schedule.fitting_seconds += result.score.scoring_seconds;
                fit_schedule.document_pass_equivalents =
                    fit_schedule.full_data_evaluations
                    + static_cast<double>(fit_schedule.approximate_documents)
                        / data.coordinates.rows();
                result.score.fit_schedule = fit_schedule;
            }
        } else {
            const ParticleSet particles = make_particle_range(data, *basis,
                helmert, result.pilot, pilot_cache, options.proposal,
                options.n_particles, particle_seed,
                options.fisher_broadening,
                options.fisher_refinement_iterations, options.n_threads,
                &proposal_screening, 0,
                static_cast<int32_t>(data.coordinates.rows()));
            if (mutable_data) {
                mutable_data->centers = RowMajorMatrixXd{};
            }
            if (options.component_screening.mode
                    == ComponentScreeningMode::Auto) {
                const bool enabled = resolve_particle_component_screening(
                    particles, particle_initial,
                    options.component_screening,
                    proposal_screening.audit_documents);
                apply_auto_component_screening_resolution(
                    particle_screening, enabled);
            }
            if (options.particle_fit_schedule
                    == ParticleFitSchedule::Subsample) {
                ApproximateParticleFit approximate = fit_particle_subsample(
                    particles, particle_initial, options,
                    particle_screening, selected->trace);
                particle = std::move(approximate.candidate);
                fit_schedule = std::move(approximate.diagnostics);
            } else {
                auto expectation_function = [&](const Model& model) {
                    return particle_expectation(particles, model,
                        ExpectationRequest{true, false, true},
                        particle_screening);
                };
                particle = fit_particle_candidate(expectation_function,
                    particle_initial, options, selected->trace);
            }
            if (!particle.trace.collapsed) {
                ComponentScreeningOptions terminal_screening =
                    particle_screening;
                if (options.exact_final_score
                    || options.particle_fit_schedule
                        != ParticleFitSchedule::Exact) {
                    terminal_screening.mode =
                        ComponentScreeningMode::Off;
                    terminal_screening.maximum_components = 0;
                }
                const auto score_start = std::chrono::steady_clock::now();
                Expectation terminal_expectation;
                result.score = score_particles(
                    particles, particle.model, terminal_screening,
                    &terminal_expectation);
                if (options.particle_fit_schedule
                        == ParticleFitSchedule::Exact) {
                    fit_schedule.full_data_evaluations =
                        particle.trace.completed_updates + 1;
                } else {
                    ++fit_schedule.full_data_evaluations;
                }
                finalize_particle_candidate(
                    particle, terminal_expectation, options);
                add_screening_metrics(result.score,
                    options.component_screening, proposal_screening,
                    particle_screening);
                result.score.exact_final_score =
                    options.exact_final_score
                    || options.particle_fit_schedule
                        != ParticleFitSchedule::Exact;
                result.score.fit_schedule = fit_schedule;
                result.score.map_component_screening =
                    selected_map_screening.mode
                    == ComponentScreeningMode::On;
                result.score.particle_generation_seconds =
                    particles.sampling_seconds + particles.likelihood_seconds;
                result.score.scoring_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - score_start).count();
                fit_schedule.fitting_seconds += result.score.scoring_seconds;
                fit_schedule.document_pass_equivalents =
                    fit_schedule.full_data_evaluations
                    + static_cast<double>(fit_schedule.approximate_documents)
                        / data.coordinates.rows();
                result.score.fit_schedule = fit_schedule;
            }
        }
    } catch (const std::exception& exception) {
        throw std::runtime_error(
            "Selected UAC initializer failed during particle EM: "
            + std::string(exception.what()));
    }
    result.traces.push_back(particle.trace);
    result.fit_schedule = fit_schedule;
    if (particle.trace.collapsed) {
        throw std::runtime_error(
            "Selected UAC initializer collapsed during particle EM");
    }
    result.model = particle.model;
    result.converged = particle.trace.converged;
    result.score.initialization_seconds = initialization_seconds;
    return result;
}
} // namespace uac::detail

namespace uac {

using namespace detail;

FitResult fit(Dataset& data, const Basis* basis,
    const FitOptions& options) {
    const Eigen::MatrixXd helmert = normalized_helmert(data.centers.cols());
    return fit_impl(data, &data, basis, options, nullptr, helmert);
}

FitResult fit(const Dataset& data, const Basis* basis,
    const FitOptions& options) {
    const Eigen::MatrixXd helmert = normalized_helmert(data.centers.cols());
    return fit_impl(data, nullptr, basis, options, nullptr, helmert);
}

FitResult fit(Dataset& data, const Basis* basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const FitOptions& options) {
    return fit_impl(data, &data, basis, options, nullptr, helmert);
}

FitResult fit(const Dataset& data, const Basis* basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const FitOptions& options) {
    return fit_impl(data, nullptr, basis, options, nullptr, helmert);
}

FitResult fit_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source, const FitOptions& options) {
    const Eigen::MatrixXd helmert = normalized_helmert(data.centers.cols());
    return fit_impl(data, nullptr, &basis, options, &source, helmert);
}

FitResult fit_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const FitOptions& options) {
    return fit_impl(data, nullptr, &basis, options, &source, helmert);
}

ScoreResult score_map(const Dataset& data, const Model& model,
    int32_t n_threads,
    const ComponentScreeningOptions& component_screening) {
    validate_dataset(data, false);
    validate_model(model);
    if (data.coordinates.cols() != model.means.cols()) {
        throw std::invalid_argument(
            "UAC score dataset and model dimensions differ");
    }
    validate_component_screening(component_screening);
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    ComponentScreeningOptions resolved = component_screening;
    if (resolved.mode == ComponentScreeningMode::Auto) {
        const bool enabled = resolve_map_component_screening(
            data, model, resolved, 0);
        apply_auto_component_screening_resolution(resolved, enabled);
    }
    ScoreResult out;
    Expectation expectation = map_expectation(data, model,
        ExpectationRequest{true, false, false}, resolved);
    out.responsibilities = std::move(expectation.responsibilities);
    out.component_screening_options = component_screening;
    out.map_component_screening =
        resolved.mode == ComponentScreeningMode::On;
    out.terminal_component_screening = out.map_component_screening;
    out.component_bound_seconds = expectation.component_bound_seconds;
    out.evaluated_component_documents =
        expectation.evaluated_component_documents;
    out.possible_component_documents =
        expectation.possible_component_documents;
    out.full_component_documents = expectation.full_component_documents;
    out.component_bound_violations =
        expectation.component_bound_violations;
    out.maximum_omitted_component_mass =
        expectation.maximum_omitted_component_mass;
    out.mean_omitted_component_mass = expectation.documents > 0
        ? expectation.omitted_component_mass_sum / expectation.documents
        : 0.0;
    out.per_document_evaluated_components =
        std::move(expectation.per_document_evaluated_components);
    out.per_document_omitted_component_mass =
        std::move(expectation.per_document_omitted_component_mass);
    return out;
}

ScoreResult score_particle_impl(const Dataset& data,
    Dataset* mutable_data, const Basis& basis, const State& state,
    const ParticleScoreOptions& options,
    const IndexedDocumentSource* count_source = nullptr) {
    const ProposalKind proposal = options.proposal;
    const int32_t particles = options.maximum_particles;
    const AdaptiveParticleOptions& adaptive_particles =
        options.adaptive_particles;
    const int32_t n_threads = options.n_threads;
    const ComponentScreeningOptions& component_screening =
        options.component_screening;
    ComponentScreeningOptions terminal_screening = component_screening;
    validate_dataset(data, !count_source);
    validate_basis(basis,
        checked_int32(data.centers.cols(), "topic count"));
    if (!count_source) validate_count_features(data, basis);
    validate_state(state);
    validate_component_screening(component_screening);
    validate_adaptive_particles(adaptive_particles, particles);
    if (particles <= 0 || state.basis_checksum != basis.checksum
        || (state.feature_weights.size() > 0
            && state.feature_weights.size()
                != basis.probabilities.rows())
        || (count_source
            && count_source->documents()
                != static_cast<int64_t>(data.identifiers.size()))
        || (count_source
            && count_source->features() != basis.probabilities.rows())
        || state.helmert.rows() != data.coordinates.cols()
        || state.helmert.cols() != data.centers.cols()
        || state.model.means.cols() != data.coordinates.cols()) {
        throw std::invalid_argument(
            "Invalid UAC particle score state or dimensions");
    }
    if (options.streaming.block_documents <= 0) {
        throw std::invalid_argument(
            "UAC streaming block document count must be positive");
    }
    if (count_source
        && options.particle_engine != ParticleEngine::Stream) {
        throw std::invalid_argument(
            "Indexed UAC counts require the stream particle engine");
    }
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    const PilotCache pilot_cache(state.pilot);
    const uint64_t particle_seed =
        static_cast<uint64_t>(state.seed) ^ 0xF604;
    const ProposalScreeningPlan proposal_screening =
        make_proposal_screening_plan(data, basis, state.helmert, state.pilot,
            pilot_cache, proposal, state.fisher_broadening,
            state.fisher_refinement_iterations, particle_seed,
            component_screening, count_source);
    ComponentScreeningOptions particle_screening = component_screening;
    if (particle_screening.mode == ComponentScreeningMode::Auto) {
        apply_auto_component_screening_resolution(
            particle_screening, false);
    }
    if (options.particle_engine == ParticleEngine::Stream) {
        ScoreResult out;
        for (int32_t attempt = 0; attempt < 2; ++attempt) {
            try {
                StreamingOptions streaming = options.streaming;
                if (attempt > 0) streaming.rebuild_cache = true;
                ParticleCache cache = open_or_build_particle_cache(
                    data, basis, state.helmert, state.pilot, pilot_cache,
                    proposal, particles, particle_seed,
                    state.fisher_broadening,
                    state.fisher_refinement_iterations,
                    n_threads, state.model, adaptive_particles,
                    &proposal_screening, component_screening, streaming,
                    count_source);
                particle_screening = component_screening;
                if (component_screening.mode
                        == ComponentScreeningMode::Auto) {
                    apply_auto_component_screening_resolution(
                        particle_screening,
                        cache.auto_screening_enabled());
                }
                terminal_screening = particle_screening;
                if (options.exact_final_score) {
                    terminal_screening.mode =
                        ComponentScreeningMode::Off;
                    terminal_screening.maximum_components = 0;
                }
                const auto score_start =
                    std::chrono::steady_clock::now();
                out = score_particle_cache(
                    cache, state.model, terminal_screening,
                    options.streaming.count_storage
                        == StreamingCountStorage::Memory,
                    n_threads);
                add_screening_metrics(out, component_screening,
                    proposal_screening, particle_screening);
                out.exact_final_score = options.exact_final_score;
                out.adaptive_particle_options = adaptive_particles;
                out.streaming_count_storage =
                    options.streaming.count_storage;
                if (count_source) {
                    out.streaming_count_spool_bytes =
                        count_source->storage_bytes();
                    out.streaming_peak_count_block_bytes =
                        count_source->peak_block_bytes();
                    out.streaming_external_count_parses = 1;
                }
                out.scoring_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now()
                    - score_start).count();
                break;
            } catch (const ParticleCacheCorruption&) {
                if (attempt > 0) throw;
            }
        }
        if (options.streaming.count_storage
                == StreamingCountStorage::Source
            && mutable_data) {
            mutable_data->counts.clear();
            mutable_data->counts.shrink_to_fit();
        }
        return out;
    }
    if (adaptive_particles.enabled()) {
        const auto particle_start = std::chrono::steady_clock::now();
        const RaggedParticleSet set = make_adaptive_particles(data, basis,
            state.helmert, state.pilot, pilot_cache, proposal,
            particle_seed,
            state.fisher_broadening, state.fisher_refinement_iterations,
            n_threads, state.model,
            adaptive_particles, particles, &proposal_screening);
        if (component_screening.mode == ComponentScreeningMode::Auto) {
            const bool enabled = resolve_particle_component_screening(
                set, state.model, component_screening,
                proposal_screening.audit_documents);
            apply_auto_component_screening_resolution(
                particle_screening, enabled);
        }
        terminal_screening = particle_screening;
        if (options.exact_final_score) {
            terminal_screening.mode = ComponentScreeningMode::Off;
            terminal_screening.maximum_components = 0;
        }
        const double particle_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - particle_start).count();
        const auto score_start = std::chrono::steady_clock::now();
        ScoreResult out = score_particles(
            set, state.model, terminal_screening);
        add_screening_metrics(out, component_screening,
            proposal_screening, particle_screening);
        out.exact_final_score = options.exact_final_score;
        out.adaptive_particle_options = adaptive_particles;
        out.particle_generation_seconds = particle_seconds;
        out.scoring_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - score_start).count();
        return out;
    }
    const auto particle_start = std::chrono::steady_clock::now();
    const ParticleSet set = make_particle_range(data, basis, state.helmert,
        state.pilot, pilot_cache, proposal, particles, particle_seed,
        state.fisher_broadening, state.fisher_refinement_iterations,
        n_threads, &proposal_screening, 0,
        static_cast<int32_t>(data.coordinates.rows()));
    if (component_screening.mode == ComponentScreeningMode::Auto) {
        const bool enabled = resolve_particle_component_screening(
            set, state.model, component_screening,
            proposal_screening.audit_documents);
        apply_auto_component_screening_resolution(
            particle_screening, enabled);
    }
    terminal_screening = particle_screening;
    if (options.exact_final_score) {
        terminal_screening.mode = ComponentScreeningMode::Off;
        terminal_screening.maximum_components = 0;
    }
    const double particle_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - particle_start).count();
    const auto score_start = std::chrono::steady_clock::now();
    ScoreResult out = score_particles(set, state.model, terminal_screening);
    add_screening_metrics(out, component_screening,
        proposal_screening, particle_screening);
    out.exact_final_score = options.exact_final_score;
    out.particle_generation_seconds = particle_seconds;
    out.scoring_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - score_start).count();
    return out;
}

ScoreResult score_particle(Dataset& data, const Basis& basis,
    const State& state, const ParticleScoreOptions& options) {
    prepare_particle_score_counts(data, state);
    return score_particle_impl(data, &data, basis, state, options);
}

ScoreResult score_particle(const Dataset& data, const Basis& basis,
    const State& state, const ParticleScoreOptions& options) {
    if (!has_nonidentity_feature_weights(state)) {
        return score_particle_impl(data, nullptr, basis, state, options);
    }
    Dataset prepared = data;
    prepare_particle_score_counts(prepared, state);
    return score_particle_impl(prepared, nullptr, basis, state, options);
}

ScoreResult score_particle_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source, const State& state,
    const ParticleScoreOptions& options) {
    ValidatingIndexedDocumentSource validated(
        source, has_nonidentity_feature_weights(state));
    return score_particle_impl(
        data, nullptr, basis, state, options, &validated);
}

} // namespace uac
