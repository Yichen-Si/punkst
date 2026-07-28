#include "clustering/uac_cache_internal.hpp"

#include <algorithm>
#include <fstream>
#include <functional>
#include <iomanip>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>

#include <tbb/parallel_for.h>

namespace uac::detail {


struct CachedDocumentMetadata {
    std::vector<ParticleDiagnostic> diagnostics;
    std::vector<int32_t> particles;
    std::vector<int32_t> proposal_components;
    std::vector<AdaptiveParticleDiagnostic> adaptive;
};

CachedResponsibilityState::CachedResponsibilityState(
    const std::filesystem::path& directory)
    : previous_(directory / "responsibilities.previous.bin"),
      current_(directory / "responsibilities.current.bin") {
    std::error_code error;
    std::filesystem::remove(previous_, error);
    std::filesystem::remove(current_, error);
}

CachedResponsibilityState::~CachedResponsibilityState() {
    std::error_code error;
    std::filesystem::remove(previous_, error);
    std::filesystem::remove(current_, error);
}

bool CachedResponsibilityState::has_previous() const {
    return has_previous_;
}

const std::filesystem::path&
CachedResponsibilityState::previous_path() const {
    return previous_;
}

const std::filesystem::path&
CachedResponsibilityState::current_path() const {
    return current_;
}

void CachedResponsibilityState::commit() {
    std::error_code error;
    std::filesystem::remove(previous_, error);
    error.clear();
    std::filesystem::rename(current_, previous_, error);
    if (error) {
        throw std::runtime_error(
            "Cannot commit streaming UAC responsibility sidecar: "
            + error.message());
    }
    has_previous_ = true;
}

std::filesystem::path responsibility_part_path(
    const std::filesystem::path& destination, int32_t index) {
    std::ostringstream suffix;
    suffix << destination.string() << ".part-"
        << std::setw(6) << std::setfill('0') << index;
    return suffix.str();
}

void remove_responsibility_parts(
    const std::vector<std::filesystem::path>& parts) {
    for (const auto& path : parts) {
        std::error_code error;
        std::filesystem::remove(path, error);
    }
}

void concatenate_responsibility_parts(
    const std::vector<std::filesystem::path>& parts,
    const std::filesystem::path& destination) {
    std::ofstream out(destination, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "Cannot create combined streaming UAC responsibility sidecar");
    }
    for (const auto& path : parts) {
        std::ifstream in(path, std::ios::binary);
        if (!in) {
            throw std::runtime_error(
                "Cannot read streaming UAC responsibility part");
        }
        out << in.rdbuf();
        if (!out) {
            throw std::runtime_error(
                "Failed combining streaming UAC responsibility parts");
        }
    }
    out.close();
    if (!out) {
        throw std::runtime_error(
            "Failed finalizing streaming UAC responsibility sidecar");
    }
}

Expectation cached_particle_expectation(const ParticleCache& cache,
    const Model& model, const ComponentScreeningOptions& screening,
    const ExpectationRequest& request, int32_t n_threads,
    CachedDocumentMetadata* metadata,
    CachedResponsibilityState* responsibility_state,
    const std::filesystem::path* responsibility_spool,
    Eigen::VectorXd* effective_membership) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t factor_rank =
        model.covariance_kind == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(
            model.factor_covariances.front().factor.cols())
        : -1;
    Expectation out = empty_expectation(cache.documents, components,
        cache.dimension, factor_rank, request.accumulate_moments);
    if (request.store_responsibilities) {
        out.responsibilities.resize(cache.documents, components);
        out.per_document_evaluated_components.resize(cache.documents);
        out.per_document_omitted_component_mass.resize(cache.documents);
    }
    if (request.collect_diagnostics) {
        out.particle_diagnostics.resize(cache.documents);
    }
    if (metadata) {
        metadata->particles.resize(cache.documents);
        metadata->proposal_components.resize(cache.documents);
        if (cache.adaptive) metadata->adaptive.resize(cache.documents);
    }
    if (responsibility_state && responsibility_spool) {
        throw std::invalid_argument(
            "Streaming UAC expectation has two responsibility destinations");
    }
    const bool compare_responsibilities =
        responsibility_state && responsibility_state->has_previous();
    const uint64_t responsibility_bytes =
        sizeof(double) * static_cast<uint64_t>(cache.documents)
        * components;
    if (compare_responsibilities
        && (!std::filesystem::exists(
                responsibility_state->previous_path())
            || std::filesystem::file_size(
                responsibility_state->previous_path())
                != responsibility_bytes)) {
        throw std::runtime_error(
            "Invalid previous streaming UAC responsibility sidecar");
    }

    struct ArithmeticShard {
        size_t begin = 0;
        size_t end = 0;
        int32_t first_document = 0;
        int32_t documents = 0;
    };
    std::vector<ArithmeticShard> arithmetic_shards;
    for (size_t begin = 0; begin < cache.shards.size();) {
        const int32_t arithmetic = cache.arithmetic_shards[begin];
        size_t end = begin + 1;
        while (end < cache.shards.size()
            && cache.arithmetic_shards[end] == arithmetic) {
            ++end;
        }
        ArithmeticShard group;
        group.begin = begin;
        group.end = end;
        group.first_document = cache.first_documents[begin];
        for (size_t shard = begin; shard < end; ++shard) {
            group.documents += cache.document_counts[shard];
        }
        arithmetic_shards.push_back(group);
        begin = end;
    }
    const int32_t arithmetic_count =
        static_cast<int32_t>(arithmetic_shards.size());
    if (arithmetic_count <= 0) {
        throw std::runtime_error("Streaming UAC cache has no arithmetic shards");
    }
    const int32_t parallel_workers =
        std::min(std::max(1, n_threads), arithmetic_count);
    out.parallel_workers = parallel_workers;

    std::vector<ExpectationBlock> blocks;
    blocks.reserve(arithmetic_count);
    for (int32_t arithmetic = 0;
            arithmetic < arithmetic_count; ++arithmetic) {
        blocks.emplace_back(components, cache.dimension, factor_rank,
            request.accumulate_moments);
    }
    std::vector<double> gaussian_seconds(arithmetic_count, 0.0);
    std::vector<double> moment_seconds(arithmetic_count, 0.0);
    std::vector<uint64_t> local_workspace_bytes(arithmetic_count, 0);
    std::vector<uint64_t> local_particle_bytes(arithmetic_count, 0);
    std::vector<double> responsibility_changes(
        compare_responsibilities ? cache.documents : 0, 0.0);

    const std::filesystem::path* part_destination = nullptr;
    if (responsibility_state) {
        part_destination = &responsibility_state->current_path();
    } else if (responsibility_spool) {
        part_destination = responsibility_spool;
    }
    std::vector<std::filesystem::path> responsibility_parts;
    if (part_destination) {
        responsibility_parts.reserve(arithmetic_count);
        for (int32_t arithmetic = 0;
                arithmetic < arithmetic_count; ++arithmetic) {
            responsibility_parts.push_back(
                responsibility_part_path(
                    *part_destination, arithmetic));
        }
        remove_responsibility_parts(responsibility_parts);
    }

    try {
        tbb::parallel_for(int32_t{0}, arithmetic_count,
            [&](int32_t arithmetic) {
            const ArithmeticShard& group = arithmetic_shards[arithmetic];
            std::ofstream part;
            if (part_destination) {
                part.open(responsibility_parts[arithmetic],
                    std::ios::binary | std::ios::trunc);
                if (!part) {
                    throw std::runtime_error(
                        "Cannot create streaming UAC responsibility part");
                }
            }
            std::ifstream previous;
            if (compare_responsibilities) {
                previous.open(responsibility_state->previous_path(),
                    std::ios::binary);
                if (!previous) {
                    throw std::runtime_error(
                        "Cannot read previous streaming UAC responsibilities");
                }
                previous.seekg(
                    static_cast<std::streamoff>(group.first_document)
                        * components * sizeof(double));
                if (!previous) {
                    throw std::runtime_error(
                        "Cannot seek previous streaming UAC responsibilities");
                }
            }
            int32_t expected_document = group.first_document;
            for (size_t shard_index = group.begin;
                    shard_index < group.end; ++shard_index) {
                const CachedParticleShard shard =
                    read_particle_cache(cache.shards[shard_index]);
                std::visit([&](const auto& particles) {
                    const int32_t first = particles.first_document;
                    if (first != expected_document) {
                        throw std::runtime_error(
                            "Noncontiguous streaming UAC cache shard");
                    }
                    expected_document += particles.documents;
                    local_particle_bytes[arithmetic] = std::max(
                        local_particle_bytes[arithmetic],
                        particle_set_bytes(particles));
                    ExpectationRequest local_request = request;
                    if (part_destination || effective_membership) {
                        local_request.store_responsibilities = true;
                    }
                    Expectation local = particle_expectation_into(
                        particles, model, local_request, screening,
                        blocks[arithmetic]);
                    if (part_destination) {
                        part.write(
                            reinterpret_cast<const char*>(
                                local.responsibilities.data()),
                            sizeof(double)
                                * local.responsibilities.size());
                        if (!part) {
                            throw std::runtime_error(
                                "Failed writing streaming UAC "
                                "responsibility part");
                        }
                    }
                    if (compare_responsibilities) {
                        Eigen::VectorXd previous_row(components);
                        for (int32_t d = 0;
                                d < particles.documents; ++d) {
                            previous.read(
                                reinterpret_cast<char*>(
                                    previous_row.data()),
                                sizeof(double) * components);
                            if (!previous) {
                                throw std::runtime_error(
                                    "Truncated streaming UAC "
                                    "responsibility sidecar");
                            }
                            responsibility_changes[first + d] =
                                (local.responsibilities.row(d).transpose()
                                    - previous_row)
                                .cwiseAbs().maxCoeff();
                        }
                    }
                    if (request.store_responsibilities) {
                        out.responsibilities.middleRows(
                            first, particles.documents) =
                            local.responsibilities;
                        std::copy(
                            local.per_document_evaluated_components.begin(),
                            local.per_document_evaluated_components.end(),
                            out.per_document_evaluated_components.begin()
                                + first);
                        std::copy(
                            local.per_document_omitted_component_mass.begin(),
                            local.per_document_omitted_component_mass.end(),
                            out.per_document_omitted_component_mass.begin()
                                + first);
                    }
                    if (request.collect_diagnostics) {
                        std::copy(local.particle_diagnostics.begin(),
                            local.particle_diagnostics.end(),
                            out.particle_diagnostics.begin() + first);
                    }
                    if (metadata) {
                        for (int32_t d = 0;
                                d < particles.documents; ++d) {
                            metadata->particles[first + d] =
                                particles.samples_for_document(d);
                            metadata->proposal_components[first + d] =
                                particles.proposal_candidates[d];
                        }
                        if constexpr (
                            std::is_same_v<
                                std::decay_t<decltype(particles)>,
                                RaggedParticleSet>) {
                            std::copy(
                                particles.adaptive_diagnostics.begin(),
                                particles.adaptive_diagnostics.end(),
                                metadata->adaptive.begin() + first);
                        }
                    }
                    gaussian_seconds[arithmetic] +=
                        local.gaussian_seconds;
                    moment_seconds[arithmetic] += local.moment_seconds;
                    local_workspace_bytes[arithmetic] = std::max(
                        local_workspace_bytes[arithmetic],
                        local.peak_workspace_bytes);
                }, shard);
            }
            if (expected_document
                    != group.first_document + group.documents) {
                throw std::runtime_error(
                    "Incomplete streaming UAC arithmetic shard");
            }
            if (part.is_open()) {
                part.close();
                if (!part) {
                    throw std::runtime_error(
                        "Failed finalizing streaming UAC "
                        "responsibility part");
                }
                const uint64_t expected_bytes =
                    sizeof(double)
                    * static_cast<uint64_t>(group.documents)
                    * components;
                if (std::filesystem::file_size(
                        responsibility_parts[arithmetic])
                        != expected_bytes) {
                    throw std::runtime_error(
                        "Invalid streaming UAC responsibility part size");
                }
            }
        });
    } catch (...) {
        remove_responsibility_parts(responsibility_parts);
        throw;
    }

    reduce_expectation_blocks(out, blocks);
    out.gaussian_seconds =
        std::accumulate(gaussian_seconds.begin(),
            gaussian_seconds.end(), 0.0);
    out.moment_seconds =
        std::accumulate(moment_seconds.begin(),
            moment_seconds.end(), 0.0);
    const uint64_t maximum_local_workspace =
        *std::max_element(local_workspace_bytes.begin(),
            local_workspace_bytes.end());
    out.peak_workspace_bytes =
        static_cast<uint64_t>(parallel_workers)
            * maximum_local_workspace
        + (request.accumulate_moments
            ? static_cast<uint64_t>(arithmetic_count)
                * expectation_block_bytes(
                    components, cache.dimension, factor_rank)
            : 0);
    std::sort(local_particle_bytes.begin(),
        local_particle_bytes.end(), std::greater<uint64_t>());
    out.peak_particle_bytes = std::accumulate(
        local_particle_bytes.begin(),
        local_particle_bytes.begin() + parallel_workers,
        uint64_t{0});

    if (compare_responsibilities) {
        double responsibility_change_sum = 0.0;
        for (double value : responsibility_changes) {
            responsibility_change_sum += value;
        }
        out.mean_max_responsibility_change =
            responsibility_change_sum / cache.documents;
        out.has_responsibility_change = true;
    }
    if (part_destination) {
        try {
            concatenate_responsibility_parts(
                responsibility_parts, *part_destination);
        } catch (...) {
            remove_responsibility_parts(responsibility_parts);
            throw;
        }
        remove_responsibility_parts(responsibility_parts);
    }
    if (responsibility_state) {
        responsibility_state->commit();
    }
    if (effective_membership) {
        effective_membership->setZero(components);
        if (request.store_responsibilities) {
            for (int32_t c = 0; c < components; ++c) {
                for (int32_t d = 0; d < cache.documents; ++d) {
                    (*effective_membership)(c) +=
                        out.responsibilities(d, c);
                }
            }
        } else {
            if (!responsibility_spool) {
                throw std::runtime_error(
                    "Streaming UAC effective membership has no rows");
            }
            std::ifstream in(
                *responsibility_spool, std::ios::binary);
            if (!in) {
                throw std::runtime_error(
                    "Cannot read final streaming UAC responsibilities");
            }
            Eigen::VectorXd row(components);
            for (int32_t d = 0; d < cache.documents; ++d) {
                in.read(reinterpret_cast<char*>(row.data()),
                    sizeof(double) * components);
                if (!in) {
                    throw std::runtime_error(
                        "Truncated final streaming UAC "
                        "responsibilities");
                }
                for (int32_t c = 0; c < components; ++c) {
                    (*effective_membership)(c) += row(c);
                }
            }
        }
    }
    return out;
}

ScoreResult score_particle_cache(const ParticleCache& cache,
    const Model& model, const ComponentScreeningOptions& screening,
    bool materialize_responsibilities, int32_t n_threads,
    Expectation* terminal_expectation) {
    CachedDocumentMetadata metadata;
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const std::filesystem::path sidecar =
        cache.temporary_storage_->directory.path
            / "score-responsibilities.bin";
    Eigen::VectorXd effective_membership =
        Eigen::VectorXd::Zero(components);
    Expectation expectation = cached_particle_expectation(cache, model,
        screening,
        ExpectationRequest{materialize_responsibilities, true, false},
        n_threads,
        &metadata, nullptr,
        materialize_responsibilities ? nullptr : &sidecar,
        &effective_membership);
    ScoreResult out;
    out.responsibilities = std::move(expectation.responsibilities);
    out.effective_membership = std::move(effective_membership);
    out.scored_documents = cache.documents;
    out.scored_components = components;
    if (!materialize_responsibilities) {
        out.responsibility_sidecar = sidecar.string();
        out.temporary_storage = cache.temporary_storage_;
    }
    out.particle_diagnostics =
        std::move(expectation.particle_diagnostics);
    out.per_document_evaluated_components =
        std::move(expectation.per_document_evaluated_components);
    out.per_document_omitted_component_mass =
        std::move(expectation.per_document_omitted_component_mass);
    out.per_document_particles = std::move(metadata.particles);
    out.per_document_proposal_components =
        std::move(metadata.proposal_components);
    out.adaptive_particle_diagnostics = std::move(metadata.adaptive);
    out.gaussian_seconds = expectation.gaussian_seconds;
    out.moment_seconds = expectation.moment_seconds;
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
    out.mean_omitted_component_mass = cache.documents > 0
        ? expectation.omitted_component_mass_sum / cache.documents : 0.0;
    out.estimated_peak_expectation_workspace_bytes =
        expectation.peak_workspace_bytes;
    out.sampling_seconds = cache.metrics.sampling_seconds;
    out.likelihood_seconds = cache.metrics.likelihood_seconds;
    out.fisher_work_seconds = cache.metrics.fisher_work_seconds;
    out.proposal_component_work_seconds =
        cache.metrics.proposal_component_work_seconds;
    out.proposal_draw_density_work_seconds =
        cache.metrics.proposal_draw_density_work_seconds;
    out.proposal_precision_fallback_seconds =
        cache.metrics.proposal_precision_fallback_seconds;
    out.proposal_precision_fallbacks =
        cache.metrics.proposal_precision_fallbacks;
    out.proposal_components_constructed =
        std::accumulate(out.per_document_proposal_components.begin(),
            out.per_document_proposal_components.end(), int64_t{0});
    out.proposal_components_possible =
        static_cast<int64_t>(cache.documents)
        * active_component_count(model);
    out.particle_samples = std::accumulate(
        out.per_document_particles.begin(),
        out.per_document_particles.end(), int64_t{0});
    out.resident_particle_bytes = 0;
    out.estimated_peak_proposal_workspace_bytes =
        cache.metrics.proposal_workspace_bytes;
    out.component_screening_options = screening;
    out.particle_component_screening =
        screening.mode != ComponentScreeningMode::Off;
    out.terminal_component_screening =
        screening.mode != ComponentScreeningMode::Off;
    out.particle_generation_seconds =
        cache.metrics.calibration_seconds
        + cache.metrics.sampling_seconds + cache.metrics.likelihood_seconds;
    out.calibration_seconds = cache.metrics.calibration_seconds;
    out.calibration_samples = cache.metrics.calibration_samples;
    out.reused_calibration_samples =
        cache.metrics.reused_calibration_samples;
    out.particle_generation_passes = cache.metrics.generation_passes;
    out.streaming = true;
    out.streaming_cache_reused = cache.reused;
    out.streaming_cache_bytes = cache.bytes;
    out.streaming_peak_particle_bytes = std::max(
        cache.metrics.peak_bytes, expectation.peak_particle_bytes);
    out.streaming_parallel_workers = expectation.parallel_workers;
    out.streaming_cache_shards =
        static_cast<int32_t>(cache.shards.size());
    out.streaming_cache_rebuilds = cache.rebuilds;
    out.streaming_particle_storage = cache.storage;
    if (terminal_expectation) {
        *terminal_expectation = std::move(expectation);
    }
    return out;
}

} // namespace uac::detail

