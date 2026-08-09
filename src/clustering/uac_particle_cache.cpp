#include "clustering/uac_cache_internal.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include <fcntl.h>
#include <sys/file.h>
#include <unistd.h>

#include <tbb/parallel_for.h>

namespace uac::detail {


uint64_t particle_set_bytes(const ParticleSet& particles) {
    const uint64_t samples = static_cast<uint64_t>(particles.documents)
        * particles.samples;
    return sizeof(double) * samples * (particles.dimension + 2)
        + sizeof(int32_t) * (
            samples + static_cast<uint64_t>(particles.documents));
}

uint64_t particle_set_bytes(const RaggedParticleSet& particles) {
    const uint64_t samples = particles.offsets.empty()
        ? 0 : static_cast<uint64_t>(particles.offsets.back());
    return sizeof(double) * samples * (particles.dimension + 2)
        + sizeof(int32_t) * (
            samples + static_cast<uint64_t>(particles.documents))
        + sizeof(int64_t)
            * static_cast<uint64_t>(particles.offsets.size());
}

void ParticleCacheMetrics::add(const ParticleSet& particles) {
    sampling_seconds += particles.sampling_seconds;
    likelihood_seconds += particles.likelihood_seconds;
    fisher_work_seconds += particles.fisher_work_seconds;
    proposal_component_work_seconds +=
        particles.proposal_component_work_seconds;
    proposal_draw_density_work_seconds +=
        particles.proposal_draw_density_work_seconds;
    proposal_precision_fallback_seconds +=
        particles.proposal_precision_fallback_seconds;
    proposal_precision_fallbacks += particles.proposal_precision_fallbacks;
    proposal_components_constructed +=
        particles.proposal_components_constructed;
    proposal_components_possible += particles.proposal_components_possible;
    peak_bytes = std::max(peak_bytes, particle_set_bytes(particles));
    proposal_workspace_bytes = std::max(
        proposal_workspace_bytes, particles.proposal_workspace_bytes);
}

void ParticleCacheMetrics::add(const RaggedParticleSet& particles) {
    sampling_seconds += particles.sampling_seconds;
    likelihood_seconds += particles.likelihood_seconds;
    fisher_work_seconds += particles.fisher_work_seconds;
    proposal_component_work_seconds +=
        particles.proposal_component_work_seconds;
    proposal_draw_density_work_seconds +=
        particles.proposal_draw_density_work_seconds;
    proposal_precision_fallback_seconds +=
        particles.proposal_precision_fallback_seconds;
    proposal_precision_fallbacks += particles.proposal_precision_fallbacks;
    calibration_seconds += particles.calibration_seconds;
    calibration_samples += particles.calibration_samples;
    reused_calibration_samples += particles.reused_calibration_samples;
    proposal_components_constructed +=
        particles.proposal_components_constructed;
    proposal_components_possible += particles.proposal_components_possible;
    peak_bytes = std::max(peak_bytes, particle_set_bytes(particles));
    proposal_workspace_bytes = std::max(
        proposal_workspace_bytes, particles.proposal_workspace_bytes);
}

constexpr uint64_t kParticleCacheMagic = 0x3148434143504341ull;
constexpr uint32_t kParticleCacheVersion = 2;

struct ParticleCacheHeader {
    uint64_t magic = kParticleCacheMagic;
    uint32_t version = kParticleCacheVersion;
    uint32_t ragged = 0;
    int32_t first_document = 0;
    int32_t documents = 0;
    int32_t dimension = 0;
    int32_t samples = 0;
    int64_t total_samples = 0;
    uint64_t payload_checksum = 0;
};

ParticleCacheHeader read_particle_cache_header(
    const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    ParticleCacheHeader header;
    if (!in
        || !in.read(reinterpret_cast<char*>(&header), sizeof(header))
        || header.magic != kParticleCacheMagic
        || header.version != kParticleCacheVersion
        || header.first_document < 0
        || header.documents <= 0
        || header.dimension <= 0
        || header.samples <= 0
        || header.total_samples <= 0
        || header.payload_checksum == 0
        || std::filesystem::file_size(path) <= sizeof(header)) {
        throw std::runtime_error(
            "Invalid UAC particle cache shard header: " + path.string());
    }
    return header;
}

uint64_t cache_file_hash(const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "Cannot hash UAC cache file: " + path.string());
    }
    uint64_t hash = 1469598103934665603ull;
    std::array<char, 64 * 1024> buffer;
    while (in) {
        in.read(buffer.data(), buffer.size());
        const std::streamsize count = in.gcount();
        if (count > 0) {
            hash = fnv_append(
                hash, buffer.data(), static_cast<size_t>(count));
        }
    }
    if (!in.eof()) {
        throw std::runtime_error(
            "Failed hashing UAC cache file: " + path.string());
    }
    return hash;
}

template<class Value>
uint64_t cache_hash_value(uint64_t hash, const Value& value) {
    return fnv_append(hash, &value, sizeof(value));
}

template<class Value>
void write_cache_values(std::ostream& out, const Value* values, size_t count,
    uint64_t& checksum) {
    if (count == 0) return;
    const size_t bytes = sizeof(Value) * count;
    out.write(reinterpret_cast<const char*>(values), bytes);
    checksum = fnv_append(checksum, values, bytes);
}

template<class Value>
void read_cache_values(std::istream& in, Value* values, size_t count,
    uint64_t& checksum) {
    if (count == 0) return;
    const size_t bytes = sizeof(Value) * count;
    in.read(reinterpret_cast<char*>(values), bytes);
    if (!in) throw std::runtime_error("Truncated UAC particle cache shard");
    checksum = fnv_append(checksum, values, bytes);
}

void rewrite_cache_header(const std::filesystem::path& path,
    const ParticleCacheHeader& header) {
    std::fstream out(path, std::ios::binary | std::ios::in | std::ios::out);
    if (!out) {
        throw std::runtime_error(
            "Cannot finalize UAC particle cache shard: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
}

void write_particle_cache(const std::filesystem::path& path,
    const ParticleSet& particles) {
    ParticleCacheHeader header;
    header.first_document = particles.first_document;
    header.documents = particles.documents;
    header.dimension = particles.dimension;
    header.samples = particles.samples;
    header.total_samples =
        static_cast<int64_t>(particles.documents) * particles.samples;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "Cannot create UAC particle cache shard: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    uint64_t checksum = 1469598103934665603ull;
    write_cache_values(out, particles.values.data(),
        static_cast<size_t>(particles.values.size()), checksum);
    write_cache_values(out, particles.log_likelihood.data(),
        static_cast<size_t>(particles.log_likelihood.size()), checksum);
    write_cache_values(out, particles.log_proposal.data(),
        static_cast<size_t>(particles.log_proposal.size()), checksum);
    write_cache_values(out, particles.proposal_origins.data(),
        particles.proposal_origins.size(), checksum);
    write_cache_values(out, particles.proposal_candidates.data(),
        particles.proposal_candidates.size(), checksum);
    if (!out) {
        throw std::runtime_error(
            "Failed writing UAC particle cache shard: " + path.string());
    }
    out.close();
    header.payload_checksum = checksum;
    rewrite_cache_header(path, header);
}

void write_particle_cache(const std::filesystem::path& path,
    const RaggedParticleSet& particles) {
    ParticleCacheHeader header;
    header.ragged = 1;
    header.first_document = particles.first_document;
    header.documents = particles.documents;
    header.dimension = particles.dimension;
    header.samples = particles.maximum_samples;
    header.total_samples =
        particles.offsets.empty() ? 0 : particles.offsets.back();
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "Cannot create UAC particle cache shard: " + path.string());
    }
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    uint64_t checksum = 1469598103934665603ull;
    write_cache_values(out, particles.offsets.data(),
        particles.offsets.size(), checksum);
    write_cache_values(out, particles.values.data(),
        particles.values.size(), checksum);
    write_cache_values(out, particles.log_likelihood.data(),
        particles.log_likelihood.size(), checksum);
    write_cache_values(out, particles.log_proposal.data(),
        particles.log_proposal.size(), checksum);
    write_cache_values(out, particles.proposal_origins.data(),
        particles.proposal_origins.size(), checksum);
    write_cache_values(out, particles.proposal_candidates.data(),
        particles.proposal_candidates.size(), checksum);
    write_cache_values(out, particles.adaptive_diagnostics.data(),
        particles.adaptive_diagnostics.size(), checksum);
    if (!out) {
        throw std::runtime_error(
            "Failed writing UAC particle cache shard: " + path.string());
    }
    out.close();
    header.payload_checksum = checksum;
    rewrite_cache_header(path, header);
}

DocumentProposal particle_cache_proposal(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal_kind, double broadening,
    const ProposalScreeningPlan* screening_plan, int32_t data_document,
    int32_t global_document, FisherWorkspace* fisher_workspace = nullptr) {
    const Eigen::VectorXd center =
        data.coordinates.row(data_document).transpose();
    const FisherApproximation fisher = fisher_approximation_impl(
        center, data.counts[data_document], basis, helmert, proposal_kind,
        true, fisher_workspace);
    const std::vector<int32_t>* candidates =
        screening_plan && screening_plan->enabled
        ? &screening_plan->candidates[global_document] : nullptr;
    return fisher_proposal(center, fisher, pilot, pilot_cache,
        broadening, candidates);
}

template<class ParticleCollection>
void write_factor_particle_cache(const std::filesystem::path& path,
    const ParticleCollection& particles, const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal_kind, uint64_t seed, double broadening,
    const AdaptiveParticleOptions& adaptive,
    const ProposalScreeningPlan* screening_plan,
    bool data_is_local_block = false) {
    constexpr bool ragged =
        std::is_same_v<ParticleCollection, RaggedParticleSet>;
    ParticleCacheHeader header;
    header.ragged = ragged ? 3 : 2;
    header.first_document = particles.first_document;
    header.documents = particles.documents;
    header.dimension = particles.dimension;
    header.samples = [&]() {
        if constexpr (ragged) return particles.maximum_samples;
        else return particles.samples;
    }();
    int64_t total_samples = 0;
    for (int32_t d = 0; d < particles.documents; ++d) {
        total_samples += particles.samples_for_document(d);
    }
    header.total_samples = total_samples;
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        throw std::runtime_error(
            "Cannot create UAC factor particle cache shard: "
            + path.string());
    }
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    uint64_t checksum = 1469598103934665603ull;
    FisherWorkspace fisher_workspace;
    for (int32_t local = 0; local < particles.documents; ++local) {
        const int32_t global_document = particles.first_document + local;
        const int32_t data_document = data_is_local_block
            ? local : global_document;
        const int32_t samples = particles.samples_for_document(local);
        const int32_t calibration_samples = ragged
            ? std::min(samples, adaptive.calibration_particles) : 0;
        const DocumentProposal proposal = particle_cache_proposal(
            data, basis, helmert, pilot, pilot_cache, proposal_kind,
            broadening, screening_plan, data_document, global_document,
            &fisher_workspace);
        const int32_t proposal_components =
            static_cast<int32_t>(proposal.weights.size());
        const uint64_t document_seed = hash_string(
            seed ^ 0x9e3779b97f4a7c15ull,
            data.identifiers[data_document]);
        const uint64_t calibration_seed = hash_string(
            seed ^ 0x6a09e667f3bcc909ull,
            data.identifiers[data_document]);
        write_cache_values(out, &samples, 1, checksum);
        write_cache_values(out, &calibration_samples, 1, checksum);
        write_cache_values(out, &proposal_components, 1, checksum);
        write_cache_values(out, &document_seed, 1, checksum);
        write_cache_values(out, &calibration_seed, 1, checksum);
        write_cache_values(out, &proposal.broadening, 1, checksum);
        write_cache_values(out, proposal.component_ids.data(),
            proposal.component_ids.size(), checksum);
        write_cache_values(out, proposal.weights.data(),
            proposal.weights.size(), checksum);
        for (int32_t component = 0;
                component < proposal_components; ++component) {
            write_cache_values(out, proposal.means[component].data(),
                proposal.means[component].size(), checksum);
            write_cache_values(out,
                proposal.precision_lower[component].data(),
                proposal.precision_lower[component].size(), checksum);
        }
        RowMajorMatrixXd regenerated(samples, particles.dimension);
        int32_t* regenerated_origins = nullptr;
        std::vector<int32_t> origins(samples);
        regenerated_origins = origins.data();
        if constexpr (ragged) {
            if (calibration_samples > 0) {
                auto calibration_values =
                    regenerated.topRows(calibration_samples);
                draw_proposal_values(proposal, calibration_seed,
                    calibration_values, regenerated_origins);
            }
            if (samples > calibration_samples) {
                auto additional =
                    regenerated.bottomRows(samples - calibration_samples);
                draw_proposal_values(proposal, document_seed, additional,
                    regenerated_origins + calibration_samples);
            }
        } else {
            draw_proposal_values(proposal, document_seed,
                regenerated, regenerated_origins);
        }
        const auto expected = particles.values_for_document(local);
        if (expected.rows() != regenerated.rows()
            || expected.cols() != regenerated.cols()
            || std::memcmp(expected.data(), regenerated.data(),
                sizeof(double) * regenerated.size()) != 0) {
            throw std::runtime_error(
                "UAC factor cache failed exact particle regeneration");
        }
        uint64_t position_checksum = fnv_append(
            1469598103934665603ull, regenerated.data(),
            sizeof(double) * regenerated.size());
        write_cache_values(out, &position_checksum, 1, checksum);
        const auto log_likelihood =
            particles.log_likelihood_for_document(local);
        const auto log_proposal =
            particles.log_proposal_for_document(local);
        write_cache_values(out, log_likelihood.data(),
            log_likelihood.size(), checksum);
        write_cache_values(out, log_proposal.data(),
            log_proposal.size(), checksum);
        if constexpr (ragged) {
            write_cache_values(out,
                &particles.adaptive_diagnostics[local], 1, checksum);
        }
    }
    if (!out) {
        throw std::runtime_error(
            "Failed writing UAC factor particle cache shard: "
            + path.string());
    }
    out.close();
    header.payload_checksum = checksum;
    rewrite_cache_header(path, header);
}

CachedParticleShard read_particle_cache_unchecked(
    const std::filesystem::path& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "Cannot open UAC particle cache shard: " + path.string());
    }
    ParticleCacheHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != kParticleCacheMagic
        || header.version != kParticleCacheVersion
        || header.documents <= 0 || header.dimension <= 0
        || header.samples <= 0 || header.total_samples <= 0) {
        throw std::runtime_error(
            "Invalid UAC particle cache shard header: " + path.string());
    }
    uint64_t checksum = 1469598103934665603ull;
    if (header.ragged == 0) {
        if (header.total_samples !=
                static_cast<int64_t>(header.documents) * header.samples) {
            throw std::runtime_error(
                "Invalid fixed UAC particle cache shape");
        }
        ParticleSet out;
        out.first_document = header.first_document;
        out.documents = header.documents;
        out.dimension = header.dimension;
        out.samples = header.samples;
        out.values.resize(header.total_samples, header.dimension);
        out.log_likelihood.resize(header.documents, header.samples);
        out.log_proposal.resize(header.documents, header.samples);
        out.proposal_origins.resize(header.total_samples);
        out.proposal_candidates.resize(header.documents);
        read_cache_values(in, out.values.data(), out.values.size(), checksum);
        read_cache_values(in, out.log_likelihood.data(),
            out.log_likelihood.size(), checksum);
        read_cache_values(in, out.log_proposal.data(),
            out.log_proposal.size(), checksum);
        read_cache_values(in, out.proposal_origins.data(),
            out.proposal_origins.size(), checksum);
        read_cache_values(in, out.proposal_candidates.data(),
            out.proposal_candidates.size(), checksum);
        if (checksum != header.payload_checksum || in.peek() != EOF) {
            throw std::runtime_error(
                "UAC particle cache shard checksum mismatch: "
                + path.string());
        }
        return out;
    }
    if (header.ragged == 2 || header.ragged == 3) {
        const bool ragged = header.ragged == 3;
        ParticleSet fixed;
        RaggedParticleSet adaptive;
        if (ragged) {
            adaptive.first_document = header.first_document;
            adaptive.documents = header.documents;
            adaptive.dimension = header.dimension;
            adaptive.maximum_samples = header.samples;
            adaptive.offsets.assign(header.documents + 1, 0);
            adaptive.proposal_candidates.resize(header.documents);
            adaptive.adaptive_diagnostics.resize(header.documents);
        } else {
            fixed.first_document = header.first_document;
            fixed.documents = header.documents;
            fixed.dimension = header.dimension;
            fixed.samples = header.samples;
            fixed.values.resize(header.total_samples, header.dimension);
            fixed.log_likelihood.resize(header.documents, header.samples);
            fixed.log_proposal.resize(header.documents, header.samples);
            fixed.proposal_origins.resize(header.total_samples);
            fixed.proposal_candidates.resize(header.documents);
        }
        int64_t sample_offset = 0;
        for (int32_t local = 0; local < header.documents; ++local) {
            int32_t samples = 0;
            int32_t calibration_samples = 0;
            int32_t proposal_components = 0;
            uint64_t document_seed = 0;
            uint64_t calibration_seed = 0;
            double broadening = 1.0;
            read_cache_values(in, &samples, 1, checksum);
            read_cache_values(in, &calibration_samples, 1, checksum);
            read_cache_values(in, &proposal_components, 1, checksum);
            read_cache_values(in, &document_seed, 1, checksum);
            read_cache_values(in, &calibration_seed, 1, checksum);
            read_cache_values(in, &broadening, 1, checksum);
            if (samples <= 0 || samples > header.samples
                || (!ragged && samples != header.samples)
                || calibration_samples < 0
                || calibration_samples > samples
                || proposal_components <= 0
                || !(broadening > 0.0)) {
                throw std::runtime_error(
                    "Invalid UAC factor cache document header");
            }
            DocumentProposal proposal;
            proposal.broadening = broadening;
            proposal.component_ids.resize(proposal_components);
            proposal.weights.resize(proposal_components);
            proposal.means.resize(proposal_components);
            proposal.precision_lower.resize(proposal_components);
            read_cache_values(in, proposal.component_ids.data(),
                proposal.component_ids.size(), checksum);
            read_cache_values(in, proposal.weights.data(),
                proposal.weights.size(), checksum);
            for (int32_t component = 0;
                    component < proposal_components; ++component) {
                proposal.means[component].resize(header.dimension);
                proposal.precision_lower[component].resize(
                    header.dimension, header.dimension);
                read_cache_values(in, proposal.means[component].data(),
                    proposal.means[component].size(), checksum);
                read_cache_values(in,
                    proposal.precision_lower[component].data(),
                    proposal.precision_lower[component].size(), checksum);
            }
            RowMajorMatrixXd values(samples, header.dimension);
            std::vector<int32_t> origins(samples);
            if (ragged && calibration_samples > 0) {
                auto calibration = values.topRows(calibration_samples);
                draw_proposal_values(proposal, calibration_seed,
                    calibration, origins.data());
            }
            if (ragged && samples > calibration_samples) {
                auto additional =
                    values.bottomRows(samples - calibration_samples);
                draw_proposal_values(proposal, document_seed, additional,
                    origins.data() + calibration_samples);
            } else if (!ragged) {
                draw_proposal_values(
                    proposal, document_seed, values, origins.data());
            }
            uint64_t expected_position_checksum = 0;
            read_cache_values(
                in, &expected_position_checksum, 1, checksum);
            const uint64_t actual_position_checksum = fnv_append(
                1469598103934665603ull, values.data(),
                sizeof(double) * values.size());
            if (actual_position_checksum != expected_position_checksum) {
                throw std::runtime_error(
                    "UAC factor cache particle regeneration mismatch");
            }
            Eigen::VectorXd log_likelihood(samples);
            Eigen::VectorXd log_proposal(samples);
            read_cache_values(in, log_likelihood.data(),
                log_likelihood.size(), checksum);
            read_cache_values(in, log_proposal.data(),
                log_proposal.size(), checksum);
            if (ragged) {
                adaptive.offsets[local] = sample_offset;
                adaptive.values.insert(adaptive.values.end(),
                    values.data(), values.data() + values.size());
                adaptive.log_likelihood.insert(
                    adaptive.log_likelihood.end(),
                    log_likelihood.data(),
                    log_likelihood.data() + samples);
                adaptive.log_proposal.insert(adaptive.log_proposal.end(),
                    log_proposal.data(), log_proposal.data() + samples);
                adaptive.proposal_origins.insert(
                    adaptive.proposal_origins.end(),
                    origins.begin(), origins.end());
                adaptive.proposal_candidates[local] = proposal_components;
                read_cache_values(in,
                    &adaptive.adaptive_diagnostics[local], 1, checksum);
            } else {
                fixed.values.middleRows(
                    static_cast<Eigen::Index>(sample_offset), samples) =
                    values;
                fixed.log_likelihood.row(local) =
                    log_likelihood.transpose();
                fixed.log_proposal.row(local) =
                    log_proposal.transpose();
                std::copy(origins.begin(), origins.end(),
                    fixed.proposal_origins.begin() + sample_offset);
                fixed.proposal_candidates[local] = proposal_components;
            }
            sample_offset += samples;
        }
        if (sample_offset != header.total_samples
            || checksum != header.payload_checksum || in.peek() != EOF) {
            throw std::runtime_error(
                "UAC factor cache shard checksum mismatch");
        }
        if (ragged) {
            adaptive.offsets[header.documents] = sample_offset;
            return adaptive;
        }
        return fixed;
    }
    if (header.ragged != 1) {
        throw std::runtime_error("Invalid UAC particle cache storage kind");
    }
    RaggedParticleSet out;
    out.first_document = header.first_document;
    out.documents = header.documents;
    out.dimension = header.dimension;
    out.maximum_samples = header.samples;
    out.offsets.resize(header.documents + 1);
    out.values.resize(header.total_samples * header.dimension);
    out.log_likelihood.resize(header.total_samples);
    out.log_proposal.resize(header.total_samples);
    out.proposal_origins.resize(header.total_samples);
    out.proposal_candidates.resize(header.documents);
    out.adaptive_diagnostics.resize(header.documents);
    read_cache_values(in, out.offsets.data(), out.offsets.size(), checksum);
    read_cache_values(in, out.values.data(), out.values.size(), checksum);
    read_cache_values(in, out.log_likelihood.data(),
        out.log_likelihood.size(), checksum);
    read_cache_values(in, out.log_proposal.data(),
        out.log_proposal.size(), checksum);
    read_cache_values(in, out.proposal_origins.data(),
        out.proposal_origins.size(), checksum);
    read_cache_values(in, out.proposal_candidates.data(),
        out.proposal_candidates.size(), checksum);
    read_cache_values(in, out.adaptive_diagnostics.data(),
        out.adaptive_diagnostics.size(), checksum);
    if (out.offsets.front() != 0
        || out.offsets.back() != header.total_samples
        || checksum != header.payload_checksum || in.peek() != EOF) {
        throw std::runtime_error(
            "UAC particle cache shard checksum mismatch: " + path.string());
    }
    return out;
}

CachedParticleShard read_particle_cache(
    const std::filesystem::path& path) {
    try {
        return read_particle_cache_unchecked(path);
    } catch (const std::exception& exception) {
        throw ParticleCacheCorruption(
            "Invalid UAC particle cache shard " + path.string()
            + ": " + exception.what());
    }
}

class CacheEntryLock {
public:
    enum class Mode { Shared, Exclusive };

    CacheEntryLock(const std::filesystem::path& path, Mode mode)
        : path_(path) {
        fd_ = ::open(path.c_str(), O_CREAT | O_RDWR, 0666);
        if (fd_ < 0) {
            throw std::runtime_error(
                "Cannot open UAC particle cache lock: " + path.string());
        }
        try {
            set_mode(mode);
        } catch (...) {
            ::close(fd_);
            fd_ = -1;
            throw;
        }
    }

    ~CacheEntryLock() {
        if (fd_ >= 0) {
            static_cast<void>(::flock(fd_, LOCK_UN));
            static_cast<void>(::close(fd_));
        }
    }

    CacheEntryLock(const CacheEntryLock&) = delete;
    CacheEntryLock& operator=(const CacheEntryLock&) = delete;

    void set_mode(Mode mode) {
        const int operation = mode == Mode::Shared ? LOCK_SH : LOCK_EX;
        while (::flock(fd_, operation) != 0) {
            if (errno == EINTR) continue;
            throw std::runtime_error(
                "Cannot lock UAC particle cache entry: "
                + path_.string());
        }
        mode_ = mode;
    }

private:
    std::filesystem::path path_;
    int fd_ = -1;
    Mode mode_ = Mode::Shared;
};

template<class Value>
void hash_cache_value(uint64_t& hash, const Value& value) {
    hash = fnv_append(hash, &value, sizeof(value));
}

uint64_t particle_cache_key(const Dataset& data, const Basis& basis,
    const Pilot& pilot, const Model& initial_model, ProposalKind proposal,
    int32_t maximum_samples, uint64_t seed, double broadening,
    const AdaptiveParticleOptions& adaptive,
    const ComponentScreeningOptions& screening,
    StreamingParticleStorage storage, int32_t block_documents,
    const IndexedDocumentSource* count_source) {
    uint64_t hash = 1469598103934665603ull;
    const char runtime[] = "uac-particle-cache-v2";
    hash = fnv_append(hash, runtime, sizeof(runtime));
    hash_cache_value(hash, basis.checksum);
    hash_cache_value(hash, proposal);
    hash_cache_value(hash, maximum_samples);
    hash_cache_value(hash, seed);
    hash_cache_value(hash, broadening);
    hash_cache_value(hash, storage);
    hash_cache_value(hash, block_documents);
    hash_cache_value(hash, adaptive.calibration_particles);
    hash_cache_value(hash, adaptive.minimum_particles);
    const bool has_responsibility_target =
        adaptive.responsibility_se_target.has_value();
    const bool has_moment_target =
        adaptive.moment_ess_target.has_value();
    hash_cache_value(hash, has_responsibility_target);
    hash_cache_value(hash,
        adaptive.responsibility_se_target.value_or(0.0));
    hash_cache_value(hash, adaptive.plausible_mass);
    hash_cache_value(hash, adaptive.plausible_responsibility);
    hash_cache_value(hash, has_moment_target);
    hash_cache_value(hash, adaptive.moment_ess_target.value_or(0.0));
    hash_cache_value(hash, screening.mode);
    hash_cache_value(hash, screening.tail_mass);
    hash_cache_value(hash, screening.proposal_proxy_tail_mass);
    hash_cache_value(hash, screening.minimum_components);
    hash_cache_value(hash, screening.maximum_components);
    hash_cache_value(hash, screening.audit_documents);
    hash_cache_value(hash, screening.minimum_work_reduction);
    for (const auto& identifier : data.identifiers) {
        hash = hash_string(hash, identifier);
    }
    hash = fnv_append(hash, data.coordinates.data(),
        sizeof(double) * data.coordinates.size());
    if (count_source) {
        hash_cache_value(hash, count_source->content_checksum());
    } else {
        for (const Document& document : data.counts) {
            hash = fnv_append(hash, document.ids.data(),
                sizeof(uint32_t) * document.ids.size());
            hash = fnv_append(hash, document.cnts.data(),
                sizeof(double) * document.cnts.size());
        }
    }
    hash = fnv_append(hash, pilot.weights.data(),
        sizeof(double) * pilot.weights.size());
    hash = fnv_append(hash, pilot.means.data(),
        sizeof(double) * pilot.means.size());
    for (const auto& covariance : pilot.covariances) {
        hash = fnv_append(hash, covariance.data(),
            sizeof(double) * covariance.size());
    }
    hash = fnv_append(hash, initial_model.weights.data(),
        sizeof(double) * initial_model.weights.size());
    hash = fnv_append(hash, initial_model.means.data(),
        sizeof(double) * initial_model.means.size());
    if (initial_model.covariance_kind == CovarianceKind::Dense) {
        for (const auto& covariance : initial_model.covariances) {
            hash = fnv_append(hash, covariance.data(),
                sizeof(double) * covariance.size());
        }
    } else {
        for (const auto& covariance : initial_model.factor_covariances) {
            hash = fnv_append(hash, covariance.diagonal.data(),
                sizeof(double) * covariance.diagonal.size());
            hash = fnv_append(hash, covariance.factor.data(),
                sizeof(double) * covariance.factor.size());
        }
    }
    return hash;
}

std::string cache_key_text(uint64_t key) {
    uint64_t second = key + 0x9e3779b97f4a7c15ull;
    second = (second ^ (second >> 30)) * 0xbf58476d1ce4e5b9ull;
    second = (second ^ (second >> 27)) * 0x94d049bb133111ebull;
    second ^= second >> 31;
    std::ostringstream out;
    out << std::hex << std::setfill('0')
        << std::setw(16) << key << std::setw(16) << second;
    return out.str();
}

ParticleCache open_or_build_particle_cache(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal, int32_t maximum_samples, uint64_t seed,
    double broadening, int32_t n_threads, const Model& initial_model,
    const AdaptiveParticleOptions& adaptive,
    const ProposalScreeningPlan* proposal_screening,
    const ComponentScreeningOptions& screening,
    const StreamingOptions& options,
    const IndexedDocumentSource* count_source) {
    if (options.block_documents <= 0) {
        throw std::invalid_argument(
            "UAC streaming block document count must be positive");
    }
    switch (options.count_storage) {
        case StreamingCountStorage::Source:
        case StreamingCountStorage::Memory:
            break;
        default:
            throw std::invalid_argument(
                "Invalid UAC streaming count storage");
    }
    switch (options.particle_storage) {
        case StreamingParticleStorage::Auto:
        case StreamingParticleStorage::Factors:
        case StreamingParticleStorage::Positions:
            break;
        default:
            throw std::invalid_argument(
                "Invalid UAC streaming particle storage");
    }
    ParticleCache cache;
    cache.documents = static_cast<int32_t>(data.coordinates.rows());
    cache.dimension = static_cast<int32_t>(data.coordinates.cols());
    cache.adaptive = adaptive.enabled();
    if (options.particle_storage == StreamingParticleStorage::Auto) {
        uint64_t factor_values = 0;
        const int32_t active =
            static_cast<int32_t>((pilot.weights.array() > 0.0).count());
        for (int32_t d = 0; d < cache.documents; ++d) {
            const int32_t proposals =
                proposal_screening && proposal_screening->enabled
                ? static_cast<int32_t>(
                    proposal_screening->candidates[d].size())
                : active;
            factor_values += static_cast<uint64_t>(proposals)
                * (1 + cache.dimension
                    + static_cast<uint64_t>(cache.dimension)
                        * cache.dimension);
        }
        const uint64_t position_values =
            static_cast<uint64_t>(cache.documents)
            * maximum_samples * cache.dimension;
        cache.storage = factor_values <= position_values
            ? StreamingParticleStorage::Factors
            : StreamingParticleStorage::Positions;
    } else {
        cache.storage = options.particle_storage;
    }
    const uint64_t key = particle_cache_key(data, basis, pilot,
        initial_model, proposal, maximum_samples, seed, broadening,
        adaptive, screening, options.particle_storage,
        options.block_documents, count_source);
    const std::filesystem::path root = options.cache_directory.empty()
        ? std::filesystem::path(".uac-cache")
        : std::filesystem::path(options.cache_directory);
    std::filesystem::create_directories(root);
    cache.directory = root / cache_key_text(key);
    cache.temporary_storage_ =
        std::make_shared<ScoreTemporaryStorage>(root);
    const int32_t factor_rank =
        initial_model.covariance_kind == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(
            initial_model.factor_covariances.front().factor.cols())
        : -1;
    const int32_t requested_shards = expectation_shards(cache.documents,
        static_cast<int32_t>(initial_model.weights.size()),
        cache.dimension, factor_rank);
    const int32_t natural_size =
        (cache.documents + requested_shards - 1) / requested_shards;
    // I/O shards may split an arithmetic shard. The E-step keeps one
    // accumulator alive across those files, preserving the batch grouping.
    for (int32_t arithmetic = 0;
            arithmetic < requested_shards; ++arithmetic) {
        const int32_t arithmetic_begin = arithmetic * natural_size;
        const int32_t arithmetic_end = std::min(
            cache.documents, arithmetic_begin + natural_size);
        for (int32_t first = arithmetic_begin;
                first < arithmetic_end;
                first += options.block_documents) {
            cache.first_documents.push_back(first);
            cache.document_counts.push_back(std::min(
                options.block_documents, arithmetic_end - first));
            cache.arithmetic_shards.push_back(arithmetic);
            std::ostringstream name;
            name << "particles-" << std::setw(6) << std::setfill('0')
                 << cache.shards.size() << ".bin";
            cache.shards.push_back(cache.directory / name.str());
        }
    }
    const int32_t n_shards =
        static_cast<int32_t>(cache.shards.size());
    const auto complete = cache.directory / "complete";
    const auto manifest = cache.directory / "manifest.tsv";
    auto validate_existing = [&]() {
        cache.metrics = {};
        cache.bytes = 0;
        cache.reused = false;
        cache.auto_component_screening_enabled_ = false;
        if (!std::filesystem::exists(complete)) {
            return false;
        }
        try {
            {
                std::ifstream marker(complete);
                std::string version;
                uint64_t expected_manifest_hash = 0;
                if (!(marker >> version >> std::hex
                        >> expected_manifest_hash)
                    || version != "uac-particle-cache-v2"
                    || expected_manifest_hash
                        != cache_file_hash(manifest)) {
                    throw std::runtime_error(
                        "Invalid UAC particle cache completion marker");
                }
            }
            {
                std::ifstream metadata(manifest);
                std::string label;
                int32_t enabled = 0;
                if (!(metadata >> label >> enabled)
                    || label != "auto_component_screening_enabled") {
                    throw std::runtime_error(
                        "Invalid UAC particle cache manifest");
                }
                cache.auto_component_screening_enabled_ = enabled != 0;
                while (metadata >> label) {
                    if (label == "storage") {
                        std::string value;
                        metadata >> value;
                        cache.storage =
                            parse_streaming_particle_storage(value);
                    } else if (label == "calibration_seconds"
                            || label == "sampling_seconds"
                            || label == "likelihood_seconds"
                            || label == "fisher_work_seconds"
                            || label == "proposal_component_work_seconds"
                            || label
                                == "proposal_draw_density_work_seconds"
                            || label
                                == "proposal_precision_fallback_seconds") {
                        double ignored = 0.0;
                        metadata >> ignored;
                    } else if (label == "proposal_precision_fallbacks") {
                        metadata >> cache.metrics.proposal_precision_fallbacks;
                    } else if (label == "proposal_workspace_bytes") {
                        metadata >> cache.metrics.proposal_workspace_bytes;
                    } else if (label == "calibration_samples") {
                        metadata >> cache.metrics.calibration_samples;
                    } else if (label == "reused_calibration_samples") {
                        metadata >> cache.metrics.reused_calibration_samples;
                    } else {
                        std::string ignored;
                        metadata >> ignored;
                    }
                    if (!metadata) {
                        throw std::runtime_error(
                            "Invalid UAC particle cache manifest value");
                    }
                }
            }
            const uint32_t expected_kind =
                cache.storage == StreamingParticleStorage::Factors
                ? (cache.adaptive ? 3u : 2u)
                : (cache.adaptive ? 1u : 0u);
            for (size_t index = 0; index < cache.shards.size(); ++index) {
                const ParticleCacheHeader header =
                    read_particle_cache_header(cache.shards[index]);
                if (header.ragged != expected_kind
                    || header.first_document
                        != cache.first_documents[index]
                    || header.documents
                        != cache.document_counts[index]
                    || header.dimension != cache.dimension) {
                    throw std::runtime_error(
                        "UAC particle cache shard metadata mismatch");
                }
                uint64_t resident_bytes =
                    sizeof(double)
                        * static_cast<uint64_t>(header.total_samples)
                        * (header.dimension + 2)
                    + sizeof(int32_t)
                        * (static_cast<uint64_t>(header.total_samples)
                            + static_cast<uint64_t>(header.documents));
                if (cache.adaptive) {
                    resident_bytes += sizeof(int64_t)
                        * static_cast<uint64_t>(header.documents + 1);
                }
                cache.metrics.peak_bytes = std::max(
                    cache.metrics.peak_bytes, resident_bytes);
                cache.bytes +=
                    std::filesystem::file_size(cache.shards[index]);
            }
            cache.reused = true;
            return true;
        } catch (const std::exception&) {
            return false;
        }
    };
    const std::filesystem::path lock_path =
        root / (cache_key_text(key) + ".lock");
    if (!options.rebuild_cache) {
        cache.entry_lock = std::make_shared<CacheEntryLock>(
            lock_path, CacheEntryLock::Mode::Shared);
        if (validate_existing()) return cache;
        cache.entry_lock.reset();
    }
    cache.entry_lock = std::make_shared<CacheEntryLock>(
        lock_path, CacheEntryLock::Mode::Exclusive);
    if (!options.rebuild_cache && validate_existing()) {
        cache.entry_lock->set_mode(CacheEntryLock::Mode::Shared);
        return cache;
    }
    const bool existing_entry = std::filesystem::exists(complete);
    if (existing_entry) {
        ++cache.rebuilds;
    }

    if (std::filesystem::exists(cache.directory)) {
        std::filesystem::remove_all(cache.directory);
    }
    ScopedTempDir staging(root);
    const std::filesystem::path temporary = staging.path;
    for (size_t shard = 0; shard < cache.shards.size(); ++shard) {
        cache.shards[shard] = temporary
            / cache.shards[shard].filename();
    }
    auto load_particle_block = [&](int32_t first, int32_t count) {
        Dataset block;
        DocumentBlock count_block = read_aligned_document_range(
            data, *count_source, first, count);
        block.identifiers = std::move(count_block.identifiers);
        block.counts = std::move(count_block.counts);
        block.raw_totals = std::move(count_block.raw_totals);
        block.effective_totals = std::move(count_block.effective_totals);
        block.centers = data.centers.middleRows(first, count);
        block.coordinates = data.coordinates.middleRows(first, count);
        return block;
    };
    ++cache.metrics.generation_passes;
    if (adaptive.enabled()) {
        for (int32_t shard = 0; shard < n_shards; ++shard) {
            const int32_t first = cache.first_documents[shard];
            const int32_t count = cache.document_counts[shard];
            const Dataset block = count_source
                ? load_particle_block(first, count) : Dataset{};
            const Dataset& particle_data = count_source ? block : data;
            const int32_t data_first = count_source ? 0 : first;
            RaggedParticleSet particles = make_adaptive_particle_range(
                particle_data, basis, helmert, pilot, pilot_cache, proposal, seed,
                broadening, n_threads, initial_model, adaptive,
                maximum_samples, proposal_screening, data_first, count,
                first);
            cache.metrics.add(particles);
            if (cache.storage == StreamingParticleStorage::Factors) {
                write_factor_particle_cache(cache.shards[shard], particles,
                    particle_data, basis, helmert, pilot, pilot_cache,
                    proposal, seed, broadening, adaptive,
                    proposal_screening, count_source != nullptr);
            } else {
                write_particle_cache(cache.shards[shard], particles);
            }
        }
    } else {
        for (int32_t shard = 0; shard < n_shards; ++shard) {
            const int32_t first = cache.first_documents[shard];
            const int32_t count = cache.document_counts[shard];
            const Dataset block = count_source
                ? load_particle_block(first, count) : Dataset{};
            const Dataset& particle_data = count_source ? block : data;
            const int32_t data_first = count_source ? 0 : first;
            ParticleSet particles = make_particle_range(particle_data, basis, helmert,
                pilot, pilot_cache, proposal, maximum_samples, seed,
                broadening, n_threads, proposal_screening, data_first, count,
                first);
            cache.metrics.add(particles);
            if (cache.storage == StreamingParticleStorage::Factors) {
                write_factor_particle_cache(cache.shards[shard], particles,
                    particle_data, basis, helmert, pilot, pilot_cache,
                    proposal, seed, broadening, adaptive,
                    proposal_screening, count_source != nullptr);
            } else {
                write_particle_cache(cache.shards[shard], particles);
            }
        }
    }
    if (screening.mode == ComponentScreeningMode::Auto) {
        RaggedParticleSet audit;
        audit.dimension = cache.dimension;
        audit.offsets.push_back(0);
        const std::vector<int32_t>& audit_documents =
            proposal_screening
            ? proposal_screening->audit_documents
            : std::vector<int32_t>{};
        for (const int32_t document : audit_documents) {
            if (document < 0 || document >= cache.documents) {
                throw std::runtime_error(
                    "UAC streaming audit document is out of range");
            }
            const Dataset block = count_source
                ? load_particle_block(document, 1) : Dataset{};
            const Dataset& particle_data = count_source ? block : data;
            const int32_t data_document = count_source ? 0 : document;
            if (adaptive.enabled()) {
                const RaggedParticleSet one =
                    make_adaptive_particle_range(particle_data, basis, helmert,
                        pilot, pilot_cache, proposal, seed, broadening,
                        n_threads, initial_model, adaptive,
                        maximum_samples, proposal_screening, data_document, 1,
                        document);
                const int32_t samples = one.samples_for_document(0);
                const auto values = one.values_for_document(0);
                audit.values.insert(audit.values.end(), values.data(),
                    values.data() + static_cast<int64_t>(samples)
                        * cache.dimension);
                const auto likelihood =
                    one.log_likelihood_for_document(0);
                audit.log_likelihood.insert(audit.log_likelihood.end(),
                    likelihood.data(), likelihood.data() + samples);
                const auto log_q = one.log_proposal_for_document(0);
                audit.log_proposal.insert(audit.log_proposal.end(),
                    log_q.data(), log_q.data() + samples);
                const auto origins =
                    one.proposal_origins_for_document(0);
                audit.proposal_origins.insert(
                    audit.proposal_origins.end(), origins.data(),
                    origins.data() + samples);
                audit.proposal_candidates.push_back(
                    one.proposal_candidates[0]);
                audit.offsets.push_back(
                    audit.offsets.back() + samples);
                audit.maximum_samples =
                    std::max(audit.maximum_samples, samples);
            } else {
                const ParticleSet one = make_particle_range(
                    particle_data, basis, helmert, pilot, pilot_cache, proposal,
                    maximum_samples, seed, broadening, n_threads,
                    proposal_screening, data_document, 1, document);
                const auto values = one.values_for_document(0);
                audit.values.insert(audit.values.end(), values.data(),
                    values.data() + static_cast<int64_t>(maximum_samples)
                        * cache.dimension);
                const auto likelihood =
                    one.log_likelihood_for_document(0);
                audit.log_likelihood.insert(audit.log_likelihood.end(),
                    likelihood.data(),
                    likelihood.data() + maximum_samples);
                const auto log_q = one.log_proposal_for_document(0);
                audit.log_proposal.insert(audit.log_proposal.end(),
                    log_q.data(), log_q.data() + maximum_samples);
                const auto origins =
                    one.proposal_origins_for_document(0);
                audit.proposal_origins.insert(
                    audit.proposal_origins.end(), origins.data(),
                    origins.data() + maximum_samples);
                audit.proposal_candidates.push_back(
                    one.proposal_candidates[0]);
                audit.offsets.push_back(
                    audit.offsets.back() + maximum_samples);
                audit.maximum_samples = maximum_samples;
            }
        }
        audit.documents =
            static_cast<int32_t>(audit_documents.size());
        std::vector<int32_t> local_audit(audit.documents);
        std::iota(local_audit.begin(), local_audit.end(), 0);
        cache.auto_component_screening_enabled_ =
            resolve_particle_component_screening(
                audit, initial_model, screening, local_audit);
    }
    {
        std::ofstream metadata(temporary / "manifest.tsv");
        metadata << "auto_component_screening_enabled\t"
            << static_cast<int32_t>(
                cache.auto_component_screening_enabled_) << "\n"
            << "storage\t"
            << streaming_particle_storage_name(cache.storage) << "\n"
            << "documents\t" << cache.documents << "\n"
            << "shards\t" << n_shards << "\n"
            << std::setprecision(17)
            << "calibration_seconds\t"
            << cache.metrics.calibration_seconds << "\n"
            << "sampling_seconds\t"
            << cache.metrics.sampling_seconds << "\n"
            << "likelihood_seconds\t"
            << cache.metrics.likelihood_seconds << "\n"
            << "fisher_work_seconds\t"
            << cache.metrics.fisher_work_seconds << "\n"
            << "proposal_component_work_seconds\t"
            << cache.metrics.proposal_component_work_seconds << "\n"
            << "proposal_draw_density_work_seconds\t"
            << cache.metrics.proposal_draw_density_work_seconds << "\n"
            << "proposal_precision_fallback_seconds\t"
            << cache.metrics.proposal_precision_fallback_seconds << "\n"
            << "proposal_precision_fallbacks\t"
            << cache.metrics.proposal_precision_fallbacks << "\n"
            << "proposal_workspace_bytes\t"
            << cache.metrics.proposal_workspace_bytes << "\n"
            << "calibration_samples\t"
            << cache.metrics.calibration_samples << "\n"
            << "reused_calibration_samples\t"
            << cache.metrics.reused_calibration_samples << "\n"
            << "rng\tstd_mt19937_64_discrete_normal_v1\n"
            << "eigen\t" << EIGEN_WORLD_VERSION << "."
            << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << "\n"
            << "fast_math\t"
#if defined(__FAST_MATH__)
            << 1
#else
            << 0
#endif
            << "\n"
            << "fma\t"
#if defined(__FMA__)
            << 1
#else
            << 0
#endif
            << "\n";
        if (!metadata) {
            throw std::runtime_error(
                "Failed writing UAC particle cache manifest");
        }
    }
    {
        std::ofstream marker(temporary / "complete");
        marker << "uac-particle-cache-v" << kParticleCacheVersion
            << "\t" << std::hex
            << cache_file_hash(temporary / "manifest.tsv") << "\n";
        if (!marker) {
            throw std::runtime_error(
                "Failed writing UAC particle cache completion marker");
        }
    }
    std::filesystem::rename(temporary, cache.directory);
    cache.bytes = 0;
    for (size_t shard = 0; shard < cache.shards.size(); ++shard) {
        cache.shards[shard] =
            cache.directory / cache.shards[shard].filename();
        cache.bytes += std::filesystem::file_size(cache.shards[shard]);
    }
    cache.entry_lock->set_mode(CacheEntryLock::Mode::Shared);
    return cache;
}

} // namespace uac::detail
