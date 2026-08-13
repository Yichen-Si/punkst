
#include "clustering/uac_common_internal.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <optional>
#include <queue>
#include <stdexcept>
#include <vector>

namespace uac {
namespace {



double entropy(const Eigen::Ref<const Eigen::RowVectorXd>& probability) {
    double value = 0.0;
    for (Eigen::Index i = 0; i < probability.size(); ++i) {
        if (probability(i) > 0.0) {
            value -= probability(i) * std::log(probability(i));
        }
    }
    return value;
}





int32_t score_components(const ScoreResult& score) {
    if (score.responsibilities.cols() > 0) {
        return static_cast<int32_t>(score.responsibilities.cols());
    }
    if (!score.responsibility_sidecar.empty()
        && score.scored_components > 0) {
        return score.scored_components;
    }
    throw std::runtime_error(
        "UAC score has neither resident nor streamed responsibilities");
}

template<class Function>
void for_each_responsibility_row(
    const ScoreResult& score, Function&& function) {
    const int32_t components = score_components(score);
    if (score.responsibilities.size() > 0) {
        for (Eigen::Index d = 0;
                d < score.responsibilities.rows(); ++d) {
            function(static_cast<int64_t>(d),
                Eigen::RowVectorXd(score.responsibilities.row(d)));
        }
        return;
    }
    std::ifstream in(score.responsibility_sidecar, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "Cannot read UAC responsibility sidecar: "
            + score.responsibility_sidecar);
    }
    Eigen::RowVectorXd row(components);
    for (int64_t d = 0; d < score.scored_documents; ++d) {
        in.read(reinterpret_cast<char*>(row.data()),
            sizeof(double) * components);
        if (!in) {
            throw std::runtime_error(
                "Truncated UAC responsibility sidecar");
        }
        function(d, row);
    }
    char extra = 0;
    if (in.read(&extra, 1)) {
        throw std::runtime_error(
            "Oversized UAC responsibility sidecar");
    }
}

} /* namespace */

void write_model(const std::string& path, const State& state,
    const Eigen::VectorXd* effective_membership) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC model: " + path);
    out << "#cluster\tactive\tweight\teffective_membership"
        "\tmean_variance\tlog_volume";
    for (const auto& topic : state.topics) out << "\t" << topic;
    out << "\n" << std::scientific << std::setprecision(4);
    RowMajorMatrixXd compositions = ilr_inverse(state.model.means, state.helmert);
    for (Eigen::Index c = 0; c < state.model.weights.size(); ++c) {
        const Eigen::MatrixXd covariance = detail::model_covariance_dense(
            state.model, c);
        Eigen::LLT<Eigen::MatrixXd> llt(covariance);
        const Eigen::MatrixXd lower = llt.matrixL();
        const double log_volume = lower.diagonal().array().log().sum();
        out << c << "\t" << static_cast<int32_t>(
            state.model.weights(c) > 0.0) << "\t"
            << state.model.weights(c) << "\t"
            << (effective_membership ? (*effective_membership)(c) : -1.0)
            << "\t" << covariance.trace() / state.model.means.cols()
            << "\t" << log_volume;
        for (Eigen::Index k = 0; k < compositions.cols(); ++k) out << "\t" << compositions(c, k);
        out << "\n";
    }
}

void write_results(const std::string& path, const Dataset& data,
    const ScoreResult& score, int32_t top_c) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC results: " + path);
    const int32_t components = score_components(score);
    if (top_c < -1) {
        throw std::invalid_argument(
            "UAC top-C output count cannot be negative");
    }
    const int32_t resolved_top_c = top_c < 0
        ? (score.terminal_component_screening ? 5 : 0)
        : top_c;
    if (resolved_top_c > 0) {
        out << "#id";
        for (int32_t rank = 1; rank <= resolved_top_c; ++rank) {
            out << "\tC" << rank << "\tP" << rank;
        }
        out << "\ttop_c_mass\tomitted_component_mass_bound\tentropy\n"
            << std::scientific << std::setprecision(4);
        for_each_responsibility_row(score,
            [&](int64_t d, const Eigen::RowVectorXd& probability) {
            if (d >= static_cast<int64_t>(data.identifiers.size())) {
                throw std::runtime_error(
                    "UAC responsibility sidecar exceeds dataset");
            }
            std::vector<Eigen::Index> order(components);
            std::iota(order.begin(), order.end(), 0);
            const int32_t available =
                std::min(resolved_top_c, components);
            std::partial_sort(order.begin(), order.begin() + available,
                order.end(), [&](Eigen::Index left, Eigen::Index right) {
                    return probability(left) == probability(right)
                        ? left < right
                        : probability(left) > probability(right);
                });
            out << data.identifiers[d];
            double retained_mass = 0.0;
            for (int32_t rank = 0; rank < resolved_top_c; ++rank) {
                if (rank < available
                    && probability(order[rank]) > 0.0) {
                    retained_mass += probability(order[rank]);
                    out << "\t" << order[rank]
                        << "\t" << probability(order[rank]);
                } else {
                    out << "\tNA\tNA";
                }
            }
            out << "\t" << retained_mass << "\t";
            if (d < static_cast<int64_t>(
                    score.per_document_omitted_component_mass.size())) {
                out << score.per_document_omitted_component_mass[d];
            } else if (!score.terminal_component_screening) {
                out << 0.0;
            } else {
                out << "NA";
            }
            out << "\t" << entropy(probability) << "\n";
        });
        return;
    }
    out << "#id\tC1\tP1\tC2\tP2\tentropy";
    for (int32_t c = 0; c < components; ++c) out << "\t" << c;
    out << "\n" << std::scientific << std::setprecision(4);
    for_each_responsibility_row(score,
        [&](int64_t d, const Eigen::RowVectorXd& probability) {
        if (d >= static_cast<int64_t>(data.identifiers.size())) {
            throw std::runtime_error(
                "UAC responsibility sidecar exceeds dataset");
        }
        std::vector<Eigen::Index> order(components);
        std::iota(order.begin(), order.end(), 0);
        std::partial_sort(order.begin(), order.begin() + std::min<size_t>(2, order.size()), order.end(),
            [&](Eigen::Index a, Eigen::Index b) {
                return probability(a) > probability(b);
            });
        const Eigen::Index first = order[0];
        const Eigen::Index second = order.size() > 1
            && probability(order[1]) > 0.0
            ? order[1] : order[0];
        out << data.identifiers[d] << "\t" << first << "\t"
            << probability(first)
            << "\t" << second << "\t" << probability(second)
            << "\t" << entropy(probability);
        for (int32_t c = 0; c < components; ++c) {
            out << "\t" << probability(c);
        }
        out << "\n";
    });
}

void write_diagnostics(const std::string& path, const Dataset& data,
    const ScoreResult& score, bool per_unit) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC diagnostics: " + path);
    out << std::scientific << std::setprecision(4)
        << "##initialization_seconds\t"
        << score.initialization_seconds << "\n"
        << "##particle_generation_seconds\t"
        << score.particle_generation_seconds << "\n"
        << "##scoring_seconds\t" << score.scoring_seconds << "\n"
        << "##particle_sampling_seconds\t" << score.sampling_seconds << "\n"
        << "##particle_fisher_work_seconds\t"
        << score.fisher_work_seconds << "\n"
        << "##particle_proposal_component_work_seconds\t"
        << score.proposal_component_work_seconds << "\n"
        << "##particle_proposal_draw_density_work_seconds\t"
        << score.proposal_draw_density_work_seconds << "\n"
        << "##particle_proposal_precision_fallback_seconds\t"
        << score.proposal_precision_fallback_seconds << "\n"
        << "##particle_proposal_precision_fallbacks\t"
        << score.proposal_precision_fallbacks << "\n"
        << "##particle_likelihood_seconds\t" << score.likelihood_seconds << "\n"
        << "##particle_calibration_seconds\t"
        << score.calibration_seconds << "\n"
        << "##particle_samples\t" << score.particle_samples << "\n"
        << "##particle_calibration_samples\t"
        << score.calibration_samples << "\n"
        << "##particle_reused_calibration_samples\t"
        << score.reused_calibration_samples << "\n"
        << "##particle_adapt_mode\t"
        << detail::adaptive_particle_mode_name(score.adaptive_particle_options)
        << "\n"
        << "##particle_adapt_resp\t"
        << detail::optional_target_or_zero(
            score.adaptive_particle_options.responsibility_se_target) << "\n"
        << "##particle_adapt_moment\t"
        << detail::optional_target_or_zero(
            score.adaptive_particle_options.moment_ess_target) << "\n"
        << "##particle_adapt_calibration\t"
        << score.adaptive_particle_options.calibration_particles << "\n"
        << "##particle_adapt_min\t"
        << score.adaptive_particle_options.minimum_particles << "\n"
        << "##particle_adapt_plausible_mass\t"
        << score.adaptive_particle_options.plausible_mass << "\n"
        << "##particle_adapt_plausible_resp\t"
        << score.adaptive_particle_options.plausible_responsibility << "\n"
        << "##particle_estep_gaussian_seconds\t"
        << score.gaussian_seconds << "\n"
        << "##particle_estep_moment_seconds\t"
        << score.moment_seconds << "\n"
        << "##resident_particle_bytes\t"
        << score.resident_particle_bytes << "\n"
        << "##estimated_peak_proposal_workspace_bytes\t"
        << score.estimated_peak_proposal_workspace_bytes << "\n"
        << "##estimated_peak_expectation_workspace_bytes\t"
        << score.estimated_peak_expectation_workspace_bytes << "\n"
        << "##particle_generation_passes\t"
        << score.particle_generation_passes << "\n"
        << "##particle_engine\t"
        << (score.streaming ? "stream" : "batch") << "\n"
        << "##particle_fit_schedule\t"
        << particle_fit_schedule_name(score.fit_schedule.schedule) << "\n"
        << "##fit_full_warmup_updates\t"
        << score.fit_schedule.full_warmup_updates << "\n"
        << "##fit_approximate_updates\t"
        << score.fit_schedule.approximate_updates << "\n"
        << "##fit_full_tail_updates\t"
        << score.fit_schedule.full_tail_updates << "\n"
        << "##fit_full_data_evaluations\t"
        << score.fit_schedule.full_data_evaluations << "\n"
        << "##fit_subsample_evaluations\t"
        << score.fit_schedule.subsample_evaluations << "\n"
        << "##fit_approximate_documents\t"
        << score.fit_schedule.approximate_documents << "\n"
        << "##fit_document_pass_equivalents\t"
        << score.fit_schedule.document_pass_equivalents << "\n"
        << "##fit_seconds\t"
        << score.fit_schedule.fitting_seconds << "\n"
        << "##fit_approximate_seconds\t"
        << score.fit_schedule.approximate_seconds << "\n"
        << "##fit_subsample_documents\t"
        << score.fit_schedule.subsample_documents << "\n"
        << "##fit_subsample_topup_rounds\t"
        << score.fit_schedule.subsample_topup_rounds << "\n"
        << "##fit_subsample_minimum_effective_size\t"
        << score.fit_schedule.subsample_minimum_effective_size << "\n"
        << "##fit_subsample_minimum_target_ratio\t"
        << score.fit_schedule.subsample_minimum_target_ratio << "\n"
        << "##fit_subsample_weighted_documents\t"
        << score.fit_schedule.subsample_weighted_documents << "\n"
        << "##fit_subsample_maximum_weight\t"
        << score.fit_schedule.subsample_maximum_weight << "\n"
        << "##fit_subsample_storage\t"
        << subsample_storage_name(score.fit_schedule.subsample_storage) << "\n"
        << "##fit_subsample_memory_budget\t"
        << score.fit_schedule.subsample_memory_budget << "\n"
        << "##fit_subsample_predicted_bytes\t"
        << score.fit_schedule.subsample_predicted_bytes << "\n"
        << "##fit_subsample_peak_bytes\t"
        << score.fit_schedule.subsample_peak_bytes << "\n"
        << "##fit_subsample_selected_particle_bytes\t"
        << score.fit_schedule.subsample_selected_particle_bytes << "\n"
        << "##fit_subsample_storage_promotions\t"
        << score.fit_schedule.subsample_storage_promotions << "\n"
        << "##fit_subsample_peak_phase\t"
        << score.fit_schedule.subsample_peak_phase << "\n"
        << "##fit_subsample_disk_bytes\t"
        << score.fit_schedule.subsample_disk_bytes << "\n"
        << "##fit_subsample_full_cache_scans\t"
        << score.fit_schedule.subsample_full_cache_scans << "\n"
        << "##fit_subsample_read_bytes\t"
        << score.fit_schedule.subsample_read_bytes << "\n"
        << "##fit_subsample_write_bytes\t"
        << score.fit_schedule.subsample_write_bytes << "\n"
        << "##fit_subsample_io_seconds\t"
        << score.fit_schedule.subsample_io_seconds << "\n"
        << "##fit_subsample_allocator_converged\t"
        << static_cast<int32_t>(
            score.fit_schedule.subsample_allocator_converged) << "\n"
        << "##fit_subsample_allocator_iterations\t"
        << score.fit_schedule.subsample_allocator_iterations << "\n"
        << "##fit_subsample_parameter_change\t"
        << score.fit_schedule.subsample_parameter_change << "\n"
        << "##fit_subsample_convergence_reason\t"
        << score.fit_schedule.subsample_convergence_reason << "\n"
        << "##fit_tail_mode\t"
        << fit_tail_mode_name(score.fit_schedule.tail_mode) << "\n"
        << "##fit_audit_converged\t"
        << static_cast<int32_t>(score.fit_schedule.audit_converged) << "\n"
        << "##fit_audit_parameter_change\t"
        << score.fit_schedule.audit_parameter_change << "\n"
        << "##fit_audit_active_set_unchanged\t"
        << static_cast<int32_t>(
            score.fit_schedule.audit_active_set_unchanged) << "\n"
        << "##stream_cache_reused\t"
        << static_cast<int32_t>(score.streaming_cache_reused) << "\n"
        << "##stream_cache_bytes\t"
        << score.streaming_cache_bytes << "\n"
        << "##stream_peak_particle_bytes\t"
        << score.streaming_peak_particle_bytes << "\n"
        << "##stream_count_spool_bytes\t"
        << score.streaming_count_spool_bytes << "\n"
        << "##stream_peak_count_block_bytes\t"
        << score.streaming_peak_count_block_bytes << "\n"
        << "##stream_external_count_parses\t"
        << score.streaming_external_count_parses << "\n"
        << "##stream_parallel_workers\t"
        << score.streaming_parallel_workers << "\n"
        << "##stream_cache_shards\t"
        << score.streaming_cache_shards << "\n"
        << "##stream_cache_rebuilds\t"
        << score.streaming_cache_rebuilds << "\n"
        << "##stream_count_storage\t"
        << streaming_count_storage_name(
            score.streaming_count_storage) << "\n"
        << "##stream_particle_storage\t"
        << streaming_particle_storage_name(
            score.streaming_particle_storage) << "\n"
        << "##component_screening_requested\t"
        << component_screening_mode_name(
            score.component_screening_options.mode) << "\n"
        << "##map_component_screening\t"
        << static_cast<int32_t>(score.map_component_screening) << "\n"
        << "##proposal_component_screening\t"
        << static_cast<int32_t>(
            score.proposal_component_screening) << "\n"
        << "##particle_component_screening\t"
        << static_cast<int32_t>(
            score.particle_component_screening) << "\n"
        << "##terminal_component_screening\t"
        << static_cast<int32_t>(
            score.terminal_component_screening) << "\n"
        << "##exact_final_score\t"
        << static_cast<int32_t>(score.exact_final_score) << "\n"
        << "##component_bound_seconds\t"
        << score.component_bound_seconds << "\n"
        << "##evaluated_component_documents\t"
        << score.evaluated_component_documents << "\n"
        << "##possible_component_documents\t"
        << score.possible_component_documents << "\n"
        << "##full_component_documents\t"
        << score.full_component_documents << "\n"
        << "##component_bound_violations\t"
        << score.component_bound_violations << "\n"
        << "##maximum_omitted_component_mass\t"
        << score.maximum_omitted_component_mass << "\n"
        << "##mean_omitted_component_mass\t"
        << score.mean_omitted_component_mass << "\n"
        << "##proposal_screening_seconds\t"
        << score.proposal_screening_seconds << "\n"
        << "##proposal_components_constructed\t"
        << score.proposal_components_constructed << "\n"
        << "##proposal_components_possible\t"
        << score.proposal_components_possible << "\n"
        << "##proposal_audit_documents\t"
        << score.proposal_audit_documents << "\n"
        << "##proposal_audit_represented_components\t"
        << score.proposal_audit_represented_components << "\n"
        << "##proposal_audit_covered_components\t"
        << score.proposal_audit_covered_components << "\n"
        << "##proposal_audit_violations\t"
        << score.proposal_audit_violations << "\n"
        << "##proposal_audit_maximum_omitted_mass\t"
        << score.proposal_audit_maximum_omitted_mass << "\n";
    if (!per_unit) return;

    const bool raw_total = data.raw_totals.size() > 0;
    const bool effective_total = data.effective_totals.size() > 0;
    const bool particle_diagnostics = !score.particle_diagnostics.empty();
    const bool adaptive = score.adaptive_particle_options.enabled()
        && !score.adaptive_particle_diagnostics.empty();
    const bool adaptive_responsibility = adaptive
        && score.adaptive_particle_options.responsibility_se_target.has_value();
    const bool adaptive_moment = adaptive
        && score.adaptive_particle_options.moment_ess_target.has_value();
    const bool proposal_screening = score.proposal_component_screening
        && !score.per_document_proposal_components.empty();
    const bool terminal_screening = score.terminal_component_screening;
    const bool evaluated_components = terminal_screening
        && !score.per_document_evaluated_components.empty();
    const bool omitted_component_mass = terminal_screening
        && !score.per_document_omitted_component_mass.empty();

    out << "#id";
    if (raw_total) out << "\traw_total";
    if (effective_total) out << "\teffective_total";
    if (adaptive && !score.per_document_particles.empty()) {
        out << "\tparticles";
    }
    if (particle_diagnostics) {
        out << "\trelative_ess\tmaximum_weight"
            << "\tlog_likelihood_range\tlog_proposal_range"
            << "\thpd80_log_density_threshold"
            << "\thpd95_log_density_threshold";
    }
    if (proposal_screening) out << "\tproposal_components";
    if (evaluated_components) out << "\tevaluated_components";
    if (omitted_component_mass) out << "\tomitted_component_mass_bound";
    if (adaptive) {
        out << "\tadapt_preliminary_max_resp"
            << "\tadapt_preliminary_entropy"
            << "\tadapt_plausible_components";
        if (adaptive_responsibility) {
            out << "\tadapt_max_resp_se"
                << "\tadapt_projected_resp_particles";
        }
        if (adaptive_moment) {
            out << "\tadapt_projected_moment_particles";
        }
        out << "\tadapt_binding";
    }
    out << "\n";

    auto write_scientific_or_na = [&](const auto& values, size_t document) {
        if (document < values.size()) {
            out << values[document];
        } else {
            out << "NA";
        }
    };
    auto write_count_or_na = [&](const Eigen::VectorXd& values,
                                 size_t document) {
        if (document < static_cast<size_t>(values.size())) {
            out << std::fixed << std::setprecision(2) << values(document)
                << std::scientific << std::setprecision(4);
        } else {
            out << "NA";
        }
    };
    for (size_t d = 0; d < data.identifiers.size(); ++d) {
        out << data.identifiers[d];
        if (raw_total) {
            out << "\t";
            write_count_or_na(data.raw_totals, d);
        }
        if (effective_total) {
            out << "\t";
            write_count_or_na(data.effective_totals, d);
        }
        if (adaptive && !score.per_document_particles.empty()) {
            out << "\t";
            write_scientific_or_na(score.per_document_particles, d);
        }
        if (particle_diagnostics) {
            out << "\t";
            if (d < score.particle_diagnostics.size()) {
                const auto& value = score.particle_diagnostics[d];
                out << value.relative_ess << "\t"
                    << value.maximum_weight << "\t"
                    << value.log_likelihood_range << "\t"
                    << value.log_proposal_range << "\t"
                    << value.hpd80_log_density_threshold << "\t"
                    << value.hpd95_log_density_threshold;
            } else {
                out << "NA\tNA\tNA\tNA\tNA\tNA";
            }
        }
        if (proposal_screening) {
            out << "\t";
            write_scientific_or_na(score.per_document_proposal_components, d);
        }
        if (evaluated_components) {
            out << "\t";
            write_scientific_or_na(score.per_document_evaluated_components, d);
        }
        if (omitted_component_mass) {
            out << "\t";
            write_scientific_or_na(
                score.per_document_omitted_component_mass, d);
        }
        if (adaptive) {
            out << "\t";
            if (d < score.adaptive_particle_diagnostics.size()) {
                const auto& value = score.adaptive_particle_diagnostics[d];
                out << value.preliminary_maximum_responsibility << "\t"
                    << value.preliminary_entropy << "\t"
                    << value.plausible_components;
                if (adaptive_responsibility) {
                    out << "\t" << value.maximum_responsibility_se << "\t"
                        << std::fixed << std::setprecision(2)
                        << value.projected_responsibility_particles
                        << std::scientific << std::setprecision(4);
                }
                if (adaptive_moment) {
                    out << "\t" << std::fixed << std::setprecision(2)
                        << value.projected_moment_particles
                        << std::scientific << std::setprecision(4);
                }
                out << "\t" << adaptive_particle_binding_name(value.binding);
            } else {
                out << "NA\tNA\tNA";
                if (adaptive_responsibility) out << "\tNA\tNA";
                if (adaptive_moment) out << "\tNA";
                out << "\tNA";
            }
        }
        out << "\n";
    }
}

void write_initialization_diagnostics(const std::string& path,
    const InitializationDiagnostics& diagnostics) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Cannot write UAC initialization diagnostics: " + path);
    }
    out << "metric\tvalue\n" << std::scientific << std::setprecision(4)
        << "measurement_mode\t"
        << initialization_measurement_mode_name(
            diagnostics.measurement_mode) << "\n"
        << "total_documents\t" << diagnostics.total_documents << "\n"
        << "measurement_documents\t"
        << diagnostics.measurement_documents << "\n"
        << "candidate_score_documents\t"
        << diagnostics.candidate_score_documents << "\n"
        << "measurement_covariance_evaluations\t"
        << diagnostics.measurement_covariance_evaluations << "\n"
        << "sampling_seed\t" << diagnostics.sampling_seed << "\n"
        << "measurement_target\t" << diagnostics.measurement_target << "\n"
        << "candidate_score_target\t"
        << diagnostics.candidate_score_target << "\n"
        << "cached_measurement_bytes\t"
        << diagnostics.cached_measurement_bytes << "\n"
        << "maximum_measurement_weight\t"
        << diagnostics.maximum_measurement_weight << "\n"
        << "maximum_candidate_score_weight\t"
        << diagnostics.maximum_candidate_score_weight << "\n"
        << "minimum_measurement_effective_size\t"
        << diagnostics.minimum_measurement_effective_size << "\n"
        << "minimum_candidate_score_effective_size\t"
        << diagnostics.minimum_candidate_score_effective_size << "\n"
        << "covariance_floor_activations\t"
        << diagnostics.covariance_floor_activations << "\n"
        << "partition_seconds\t" << diagnostics.partition_seconds << "\n"
        << "measurement_seconds\t" << diagnostics.measurement_seconds << "\n"
        << "candidate_score_seconds\t"
        << diagnostics.candidate_score_seconds << "\n"
        << "total_seconds\t" << diagnostics.total_seconds << "\n";
}

void write_initialization_results(const std::string& path,
    const Dataset& data,
    const std::vector<InitializationPartition>& partitions) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Cannot write UAC initialization results: " + path);
    }
    if (partitions.empty()) {
        throw std::invalid_argument(
            "UAC initialization results require at least one partition");
    }
    const Eigen::Index documents = static_cast<Eigen::Index>(
        data.identifiers.size());
    out << "#id";
    int32_t kmeans_index = 0;
    int32_t leiden_index = 0;
    for (const auto& partition : partitions) {
        if (partition.assignments.size() != documents) {
            throw std::invalid_argument(
                "UAC initialization partition size differs from dataset");
        }
        const char* method = nullptr;
        switch (partition.start_method) {
            case StartMethod::KMeans: method = "kmeans"; break;
            case StartMethod::Leiden: method = "leiden"; break;
        }
        if (!method) {
            throw std::invalid_argument(
                "Unknown UAC initialization partition method");
        }
        const int32_t method_index = partition.start_method
                == StartMethod::KMeans
            ? ++kmeans_index : ++leiden_index;
        const std::string column = std::string(method)
            + (method_index == 1 ? "" : std::to_string(method_index));
        if (partition.start_method == StartMethod::Leiden) {
            if (partition.raw_assignments.size() != documents) {
                throw std::invalid_argument(
                    "Raw UAC Leiden partition size differs from dataset");
            }
            out << "\t" << column << "_raw";
        } else if (partition.raw_assignments.size() != 0) {
            throw std::invalid_argument(
                "Raw UAC initialization partition is only valid for Leiden");
        }
        out << "\t" << column;
    }
    out << "\n";
    for (Eigen::Index d = 0; d < documents; ++d) {
        out << data.identifiers[d];
        for (const auto& partition : partitions) {
            if (partition.start_method == StartMethod::Leiden) {
                out << "\t" << partition.raw_assignments(d);
            }
            out << "\t" << partition.assignments(d);
        }
        out << "\n";
    }
}

void write_subsample_diagnostics(const std::string& path,
    const FitScheduleDiagnostics& diagnostics) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Cannot write UAC subsample diagnostics: " + path);
    }
    out << "record\tindex\tdocuments\tpurity\tinclusion_probability"
        "\tbytes\ttarget\tpredicted_kish\trealized_kish\tshortfall"
        "\ttopups\n";
    for (size_t h = 0;
            h < diagnostics.subsample_stratum_documents.size(); ++h) {
        const int32_t documents = diagnostics.subsample_stratum_documents[h];
        const double purity = h < diagnostics.subsample_stratum_purity.size()
            ? diagnostics.subsample_stratum_purity[h] : 0.0;
        const double probability =
            h < diagnostics.subsample_stratum_probability.size()
            ? diagnostics.subsample_stratum_probability[h] : 0.0;
        const uint64_t bytes = h < diagnostics.subsample_stratum_bytes.size()
            ? diagnostics.subsample_stratum_bytes[h] : 0;
        out << "stratum\t" << h << "\t" << documents << "\t"
            << purity << "\t" << probability << "\t" << bytes
            << "\tNA\tNA\tNA\tNA\tNA\n";
    }
    for (size_t c = 0;
            c < diagnostics.subsample_component_target.size(); ++c) {
        const double target = diagnostics.subsample_component_target[c];
        const double predicted =
            c < diagnostics.subsample_component_predicted_effective_size.size()
            ? diagnostics.subsample_component_predicted_effective_size[c] : 0.0;
        const double realized =
            c < diagnostics.subsample_component_realized_effective_size.size()
            ? diagnostics.subsample_component_realized_effective_size[c] : 0.0;
        const int32_t topups =
            c < diagnostics.subsample_component_topups.size()
            ? diagnostics.subsample_component_topups[c] : 0;
        out << "component\t" << c
            << "\tNA\tNA\tNA\tNA\t" << target << "\t"
            << predicted << "\t" << realized << "\t"
            << std::max(0.0, target - realized) << "\t"
            << topups << "\n";
    }
}

void write_trace(const std::string& path,
    const std::vector<RestartTrace>& traces) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC trace: " + path);
    out << "#handoff\tphase\tevent\tstart\tstart_method\tseed"
        "\traw_communities"
        "\treconciliation_count\tleiden_resolution\tselection_objective"
        "\tselected\tsucceeded\tcompleted_updates\tobjective"
        "\trelative_objective_change"
        "\tmean_max_responsibility_change"
        "\tmedian_absolute_relative_variance_change"
        "\tmean_top_probability"
        "\tactive_components\tconverged\tcollapsed"
        "\tfixed_em_iteration_schedule\n"
        << std::scientific << std::setprecision(12);
    for (const auto& trace : traces) {
        auto write_metadata = [&](TraceEvent event) {
            out << handoff_name(trace.handoff) << "\t"
                << trace_phase_name(trace.phase) << "\t"
                << trace_event_name(event) << "\t"
                << trace.start << "\t"
                << start_method_name(trace.start_method)
                << "\t" << trace.seed << "\t" << trace.raw_communities
                << "\t" << trace.reconciliation_count << "\t";
            if (trace.start_method == StartMethod::Leiden) {
                out << trace.leiden_resolution;
            } else {
                out << "NA";
            }
            out << "\t";
            if (std::isfinite(trace.selection_objective)) {
                out << trace.selection_objective;
            } else {
                out << "NA";
            }
            out << "\t" << static_cast<int32_t>(trace.selected)
                << "\t" << static_cast<int32_t>(trace.succeeded);
        };
        if (trace.points.empty()) {
            write_metadata(TraceEvent::Failure);
            out << "\t" << trace.completed_updates
                << "\tNA\tNA\tNA\tNA\tNA\t-1\t"
                << static_cast<int32_t>(trace.converged) << "\t"
                << static_cast<int32_t>(trace.collapsed) << "\t"
                << static_cast<int32_t>(
                    trace.fixed_em_iteration_schedule) << "\n";
            continue;
        }
        for (const auto& point : trace.points) {
            write_metadata(point.event);
            out << "\t" << point.completed_updates << "\t"
                << point.objective << "\t";
            if (std::isfinite(point.relative_objective_change)) {
                out << point.relative_objective_change;
            } else {
                out << "NA";
            }
            out << "\t";
            if (std::isfinite(point.mean_max_responsibility_change)) {
                out << point.mean_max_responsibility_change;
            } else {
                out << "NA";
            }
            out << "\t";
            if (std::isfinite(
                    point.median_absolute_relative_variance_change)) {
                out << point.median_absolute_relative_variance_change;
            } else {
                out << "NA";
            }
            out << "\t";
            if (std::isfinite(point.mean_top_probability)) {
                out << point.mean_top_probability;
            } else {
                out << "NA";
            }
            out << "\t" << point.active_components
                << "\t" << static_cast<int32_t>(trace.converged)
                << "\t" << static_cast<int32_t>(trace.collapsed)
                << "\t"
                << static_cast<int32_t>(
                    trace.fixed_em_iteration_schedule) << "\n";
        }
    }
}

void write_model_trace(const std::string& path,
    const std::vector<RestartTrace>& traces) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write UAC model trace: " + path);
    }
    out << "#handoff\tstart\tcompleted_updates\tevent"
        "\tupdate_shrinkage_strength\tcomponent\tparameter"
        "\trow\tcolumn\tvalue\n"
        << std::scientific << std::setprecision(12);
    for (const auto& trace : traces) {
        for (const auto& entry : trace.model_trace) {
            const Model& model = entry.model;
            auto write_prefix = [&]() {
                out << handoff_name(trace.handoff) << "\t"
                    << trace.start << "\t" << entry.completed_updates << "\t"
                    << trace_event_name(entry.event) << "\t";
                if (std::isfinite(entry.update_shrinkage_strength)) {
                    out << entry.update_shrinkage_strength;
                } else {
                    out << "NA";
                }
            };
            for (Eigen::Index c = 0; c < model.weights.size(); ++c) {
                write_prefix();
                out << "\t" << c << "\tweight\t-1\t-1\t"
                    << model.weights(c) << "\n";
                for (Eigen::Index j = 0; j < model.means.cols(); ++j) {
                    write_prefix();
                    out << "\t" << c << "\tmean\t" << j << "\t-1\t"
                        << model.means(c, j) << "\n";
                }
                const Eigen::MatrixXd covariance =
                    detail::model_covariance_dense(model, static_cast<int32_t>(c));
                for (Eigen::Index r = 0; r < covariance.rows(); ++r) {
                    for (Eigen::Index j = 0; j < covariance.cols(); ++j) {
                        write_prefix();
                        out << "\t" << c << "\tcovariance\t" << r
                            << "\t" << j << "\t" << covariance(r, j)
                            << "\n";
                    }
                }
            }
            if (model.covariance_kind == CovarianceKind::FactorAnalytic
                && model.factor_diagonal_mode
                    == FactorDiagonalMode::Shared) {
                for (Eigen::Index r = 0;
                        r < model.shared_factor_diagonal.size(); ++r) {
                    write_prefix();
                    out << "\t-1\tshared_diagonal\t" << r
                        << "\t-1\t" << model.shared_factor_diagonal(r)
                        << "\n";
                }
            }
            const Eigen::MatrixXd target =
                model.covariance_kind == CovarianceKind::Dense
                ? model.shrinkage_target
                : model.factor_shrinkage_target.dense();
            for (Eigen::Index r = 0; r < target.rows(); ++r) {
                for (Eigen::Index j = 0; j < target.cols(); ++j) {
                    write_prefix();
                    out << "\t-1\tshrinkage_target\t" << r << "\t"
                        << j << "\t" << target(r, j) << "\n";
                }
            }
        }
    }
}

void write_separation(const std::string& path, const Model& model) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC separation: " + path);
    out << "#cluster_a\tcluster_b\tstandardized_separation\tbhattacharyya_distance\n"
        << std::scientific << std::setprecision(4);
    for (Eigen::Index a = 0; a < model.weights.size(); ++a) {
        if (!(model.weights(a) > 0.0)) continue;
        for (Eigen::Index b = a + 1; b < model.weights.size(); ++b) {
            if (!(model.weights(b) > 0.0)) continue;
            const Eigen::VectorXd difference = model.means.row(a).transpose() - model.means.row(b).transpose();
            const Eigen::MatrixXd covariance = 0.5
                * (detail::model_covariance_dense(model, a)
                    + detail::model_covariance_dense(model, b));
            Eigen::LLT<Eigen::MatrixXd> llt(covariance);
            const double standardized = std::sqrt(std::max(0.0, difference.dot(llt.solve(difference))));
            const double logdet_mean = 2.0 * Eigen::MatrixXd(llt.matrixL()).diagonal().array().log().sum();
            Eigen::LLT<Eigen::MatrixXd> llt_a(
                detail::model_covariance_dense(model, a)), llt_b(
                detail::model_covariance_dense(model, b));
            const double logdet_a = 2.0 * Eigen::MatrixXd(llt_a.matrixL()).diagonal().array().log().sum();
            const double logdet_b = 2.0 * Eigen::MatrixXd(llt_b.matrixL()).diagonal().array().log().sum();
            const double bhattacharyya = 0.125 * standardized * standardized
                + 0.5 * (logdet_mean - 0.5 * (logdet_a + logdet_b));
            out << a << "\t" << b << "\t" << standardized << "\t" << bhattacharyya << "\n";
        }
    }
}

void write_representatives(const std::string& path, const Dataset& data,
    const ScoreResult& score, int32_t n_representatives) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC representatives: " + path);
    out << "#cluster\trank\tid\tprobability\ttop_probability\tentropy\n"
        << std::scientific << std::setprecision(10);
    struct Representative {
        double probability = 0.0;
        double top_probability = 0.0;
        double entropy = 0.0;
        int64_t document = 0;
    };
    auto less_desirable = [](const Representative& left,
            const Representative& right) {
        return left.probability > right.probability
            || (left.probability == right.probability
                && left.document < right.document);
    };
    const int32_t components = score_components(score);
    std::vector<std::priority_queue<Representative,
        std::vector<Representative>, decltype(less_desirable)>> heaps;
    Eigen::VectorXd membership = Eigen::VectorXd::Zero(components);
    heaps.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        heaps.emplace_back(less_desirable);
    }
    for_each_responsibility_row(score,
        [&](int64_t d, const Eigen::RowVectorXd& probability) {
        const double top = probability.maxCoeff();
        const double row_entropy = entropy(probability);
        for (int32_t c = 0; c < components; ++c) {
            membership(c) += probability(c);
            Representative value{
                probability(c), top, row_entropy, d};
            auto& heap = heaps[c];
            heap.push(value);
            if (static_cast<int32_t>(heap.size())
                    > n_representatives) {
                heap.pop();
            }
        }
    });
    for (int32_t c = 0; c < components; ++c) {
        if (!(membership(c) > 0.0)) continue;
        std::vector<Representative> selected;
        while (!heaps[c].empty()) {
            selected.push_back(heaps[c].top());
            heaps[c].pop();
        }
        std::sort(selected.begin(), selected.end(),
            [](const Representative& left,
                    const Representative& right) {
                return left.probability > right.probability
                    || (left.probability == right.probability
                        && left.document < right.document);
            });
        for (size_t rank = 0; rank < selected.size(); ++rank) {
            const auto& value = selected[rank];
            out << c << "\t" << rank + 1 << "\t"
                << data.identifiers[value.document]
                << "\t" << value.probability
                << "\t" << value.top_probability
                << "\t" << value.entropy << "\n";
        }
    }
}

void write_visualization_axes(const std::string& path,
    const State& state, const VisualizationResult& visualization) {
    punkst::projection::write_visualization_axes(
        path, state.topics, visualization);
}

void write_visualization_axes(const std::string& path,
    const std::vector<std::string>& topics,
    const VisualizationResult& visualization) {
    punkst::projection::write_visualization_axes(
        path, topics, visualization);
}

void write_visualization_model(const std::string& path,
    const State& state, const VisualizationResult& visualization) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Cannot write UAC visualization model: " + path);
    }
    const Eigen::Index dimensions = visualization.mean.projection.cols();
    const bool has_full = visualization.full.projection.cols() > 0;
    if (dimensions <= 0 || (has_full
            && visualization.full.projection.cols() != dimensions)) {
        throw std::invalid_argument(
            "Invalid UAC visualization model dimensions");
    }
    out << "#whitening\tview\tcluster\tactive\tweight";
    for (Eigen::Index axis = 0; axis < dimensions; ++axis) {
        out << "\tmean_" << axis + 1;
    }
    for (Eigen::Index row = 0; row < dimensions; ++row) {
        for (Eigen::Index column = row; column < dimensions; ++column) {
            out << "\tcov_" << row + 1 << "_" << column + 1;
        }
    }
    out << "\n" << std::scientific << std::setprecision(10);
    std::vector<const VisualizationProjection*> views{&visualization.mean};
    if (has_full) views.push_back(&visualization.full);
    for (const VisualizationProjection* view : views) {
        if (view->component_means.rows() != state.model.weights.size()
            || view->component_means.cols() != dimensions
            || view->component_covariances.size()
                != static_cast<size_t>(state.model.weights.size())) {
            throw std::invalid_argument(
                "Invalid UAC projected component model");
        }
        for (Eigen::Index component = 0;
                component < state.model.weights.size(); ++component) {
            const Eigen::MatrixXd& covariance =
                view->component_covariances[component];
            if (covariance.rows() != dimensions
                || covariance.cols() != dimensions) {
                throw std::invalid_argument(
                    "Invalid UAC projected component covariance");
            }
            out << visualization_whitening_name(visualization.whitening)
                << "\t" << visualization_view_name(view->view)
                << "\t" << component << "\t"
                << static_cast<int32_t>(state.model.weights(component) > 0.0)
                << "\t" << state.model.weights(component);
            for (Eigen::Index axis = 0; axis < dimensions; ++axis) {
                out << "\t" << view->component_means(component, axis);
            }
            for (Eigen::Index row = 0; row < dimensions; ++row) {
                for (Eigen::Index column = row;
                        column < dimensions; ++column) {
                    out << "\t" << covariance(row, column);
                }
            }
            out << "\n";
        }
    }
}

void write_visualization_results(const std::string& path,
    const Dataset& data, const VisualizationResult& visualization) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Cannot write UAC visualization results: " + path);
    }
    const Eigen::Index dimensions = visualization.mean.projection.cols();
    if (data.coordinates.rows()
            != static_cast<Eigen::Index>(data.identifiers.size())
        || data.coordinates.cols() != visualization.mean.projection.rows()) {
        throw std::invalid_argument(
            "Invalid UAC visualization result dimensions");
    }
    const bool has_full = visualization.full.projection.cols() > 0;
    if (has_full
        && (visualization.full.projection.rows() != data.coordinates.cols()
            || visualization.full.projection.cols() != dimensions)) {
        throw std::invalid_argument(
            "Invalid UAC full visualization result dimensions");
    }
    out << "#id";
    for (Eigen::Index axis = 0; axis < dimensions; ++axis) {
        out << "\tmean_" << axis + 1;
    }
    if (has_full) {
        for (Eigen::Index axis = 0; axis < dimensions; ++axis) {
            out << "\tfull_" << axis + 1;
        }
    }
    out << "\n" << std::scientific << std::setprecision(4);
    for (Eigen::Index document = 0;
            document < data.coordinates.rows(); ++document) {
        out << data.identifiers[document];
        for (Eigen::Index axis = 0; axis < dimensions; ++axis) {
            out << "\t" << data.coordinates.row(document).dot(
                visualization.mean.projection.col(axis));
        }
        if (has_full) {
            for (Eigen::Index axis = 0; axis < dimensions; ++axis) {
                out << "\t" << data.coordinates.row(document).dot(
                    visualization.full.projection.col(axis));
            }
        }
        out << "\n";
    }
}

} // namespace uac
