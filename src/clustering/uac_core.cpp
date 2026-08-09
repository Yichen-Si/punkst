
#include "clustering/uac_common_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace uac {

double median_absolute_relative_variance_change(const Model& current,
    const Model& previous, double covariance_floor) {
    if (!(covariance_floor > 0.0) || !std::isfinite(covariance_floor)
        || current.covariance_kind != previous.covariance_kind
        || current.weights.size() != previous.weights.size()
        || current.means.rows() != previous.means.rows()
        || current.means.cols() != previous.means.cols()
        || current.means.cols() <= 0) {
        throw std::invalid_argument(
            "Incompatible UAC models for variance convergence");
    }
    const size_t components = static_cast<size_t>(current.weights.size());
    if ((current.covariance_kind == CovarianceKind::Dense
            && (current.covariances.size() != components
                || previous.covariances.size() != components))
        || (current.covariance_kind == CovarianceKind::FactorAnalytic
            && (current.factor_covariances.size() != components
                || previous.factor_covariances.size() != components))) {
        throw std::invalid_argument(
            "Incomplete UAC covariance model for variance convergence");
    }
    std::vector<double> changes;
    changes.reserve(static_cast<size_t>(current.weights.size()));
    const double dimension = static_cast<double>(current.means.cols());
    for (Eigen::Index c = 0; c < current.weights.size(); ++c) {
        const bool current_active = current.weights(c) > 0.0;
        const bool previous_active = previous.weights(c) > 0.0;
        if (current_active != previous_active) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        if (!current_active) continue;
        const Eigen::MatrixXd current_covariance =
            detail::model_covariance_dense(current, static_cast<int32_t>(c));
        const Eigen::MatrixXd previous_covariance =
            detail::model_covariance_dense(previous, static_cast<int32_t>(c));
        if (current_covariance.rows() != current.means.cols()
            || current_covariance.cols() != current.means.cols()
            || previous_covariance.rows() != previous.means.cols()
            || previous_covariance.cols() != previous.means.cols()) {
            throw std::invalid_argument(
                "Invalid UAC covariance shape for variance convergence");
        }
        const double current_variance =
            current_covariance.trace() / dimension;
        const double previous_variance =
            previous_covariance.trace() / dimension;
        if (!std::isfinite(current_variance)
            || !std::isfinite(previous_variance)) {
            return std::numeric_limits<double>::quiet_NaN();
        }
        changes.push_back(std::abs(current_variance - previous_variance)
            / std::max(previous_variance, covariance_floor));
    }
    if (changes.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::sort(changes.begin(), changes.end());
    const size_t middle = changes.size() / 2;
    return changes.size() % 2 == 0
        ? 0.5 * (changes[middle - 1] + changes[middle])
        : changes[middle];
}

const char* handoff_name(HandoffMode value) {
    switch (value) {
        case HandoffMode::Map: return "map";
        case HandoffMode::Particle: return "particle";
    }
    throw std::invalid_argument("Unknown UAC handoff");
}

const char* proposal_name(ProposalKind value) {
    switch (value) {
        case ProposalKind::ExactFisher: return "exact_fisher";
        case ProposalKind::SparseEmpiricalFisher:
            return "sparse_empirical_fisher";
    }
    throw std::invalid_argument("Unknown UAC proposal");
}

const char* start_method_name(StartMethod value) {
    switch (value) {
        case StartMethod::KMeans: return "kmeans++";
        case StartMethod::Leiden: return "leiden";
    }
    throw std::invalid_argument("Unknown UAC start method");
}

const char* trace_phase_name(TracePhase value) {
    switch (value) {
        case TracePhase::CorrectedMomScore:
            return "corrected_mom_score";
        case TracePhase::PointMapEm: return "point_map_em";
        case TracePhase::ParticleEm: return "particle_em";
    }
    throw std::invalid_argument("Unknown UAC trace phase");
}

const char* trace_event_name(TraceEvent value) {
    switch (value) {
        case TraceEvent::CandidateScore: return "candidate_score";
        case TraceEvent::Evaluation: return "evaluation";
        case TraceEvent::Terminal: return "terminal";
        case TraceEvent::Failure: return "failure";
    }
    throw std::invalid_argument("Unknown UAC trace event");
}

const char* adaptive_particle_binding_name(AdaptiveParticleBinding value) {
    switch (value) {
        case AdaptiveParticleBinding::Minimum: return "minimum";
        case AdaptiveParticleBinding::Responsibility:
            return "responsibility";
        case AdaptiveParticleBinding::MomentEss: return "moment_ess";
    }
    throw std::invalid_argument("Unknown adaptive particle binding");
}

const char* component_screening_mode_name(ComponentScreeningMode value) {
    switch (value) {
        case ComponentScreeningMode::Off: return "off";
        case ComponentScreeningMode::On: return "on";
        case ComponentScreeningMode::Auto: return "auto";
    }
    throw std::invalid_argument("Unknown component screening mode");
}

const char* particle_engine_name(ParticleEngine value) {
    switch (value) {
        case ParticleEngine::Batch: return "batch";
        case ParticleEngine::Stream: return "stream";
    }
    throw std::invalid_argument("Unknown UAC particle engine");
}

const char* streaming_count_storage_name(StreamingCountStorage value) {
    switch (value) {
        case StreamingCountStorage::Source: return "source";
        case StreamingCountStorage::Memory: return "memory";
    }
    throw std::invalid_argument("Unknown UAC streaming count storage");
}

const char* streaming_particle_storage_name(
    StreamingParticleStorage value) {
    switch (value) {
        case StreamingParticleStorage::Auto: return "auto";
        case StreamingParticleStorage::Factors: return "factors";
        case StreamingParticleStorage::Positions: return "positions";
    }
    throw std::invalid_argument("Unknown UAC streaming particle storage");
}

HandoffMode parse_handoff(const std::string& value) {
    if (value == "map") return HandoffMode::Map;
    if (value == "particle") return HandoffMode::Particle;
    throw std::invalid_argument("UAC handoff must be map or particle");
}

ProposalKind parse_proposal(const std::string& value) {
    if (value == "exact_fisher") return ProposalKind::ExactFisher;
    if (value == "sparse_empirical_fisher") {
        return ProposalKind::SparseEmpiricalFisher;
    }
    throw std::invalid_argument(
        "UAC proposal must be exact_fisher or sparse_empirical_fisher");
}

StartMethod parse_start_method(const std::string& value) {
    if (value == "kmeans++") return StartMethod::KMeans;
    if (value == "leiden") return StartMethod::Leiden;
    throw std::invalid_argument("UAC start method must be kmeans++ or leiden");
}

ComponentScreeningMode parse_component_screening_mode(
    const std::string& value) {
    if (value == "off") return ComponentScreeningMode::Off;
    if (value == "on") return ComponentScreeningMode::On;
    if (value == "auto") return ComponentScreeningMode::Auto;
    throw std::invalid_argument(
        "Component screening mode must be off, on, or auto");
}

ParticleEngine parse_particle_engine(const std::string& value) {
    if (value == "batch") return ParticleEngine::Batch;
    if (value == "stream") return ParticleEngine::Stream;
    throw std::invalid_argument(
        "UAC particle engine must be batch or stream");
}

StreamingCountStorage parse_streaming_count_storage(
    const std::string& value) {
    if (value == "source") return StreamingCountStorage::Source;
    if (value == "memory") return StreamingCountStorage::Memory;
    throw std::invalid_argument(
        "UAC stream counts must be source or memory");
}

StreamingParticleStorage parse_streaming_particle_storage(
    const std::string& value) {
    if (value == "auto") return StreamingParticleStorage::Auto;
    if (value == "factors") return StreamingParticleStorage::Factors;
    if (value == "positions") return StreamingParticleStorage::Positions;
    throw std::invalid_argument(
        "UAC stream particle storage must be auto, factors, or positions");
}

namespace detail {

double increased_leiden_resolution(double resolution, int32_t raw_communities,
    int32_t requested_communities) {
    if (!(resolution > 0.0) || !std::isfinite(resolution)
        || raw_communities <= 0 || requested_communities <= 0
        || raw_communities >= requested_communities) {
        throw std::invalid_argument("Invalid adaptive Leiden resolution input");
    }
    const double ratio = static_cast<double>(requested_communities)
        / raw_communities;
    const double multiplier = std::min(2.0, std::max(1.25, ratio));
    const double next = resolution * multiplier;
    if (!(next > resolution) || !std::isfinite(next)) {
        throw std::runtime_error("Adaptive Leiden resolution became nonfinite");
    }
    return next;
}

double midpoint_leiden_resolution(double lower, double upper) {
    if (!(lower > 0.0) || !(upper > lower) || !std::isfinite(lower)
        || !std::isfinite(upper)) {
        throw std::invalid_argument("Invalid Leiden resolution bracket");
    }
    const double midpoint = lower + 0.5 * (upper - lower);
    if (!(midpoint > lower && midpoint < upper)
        || !std::isfinite(midpoint)) {
        throw std::runtime_error("Leiden resolution midpoint is invalid");
    }
    return midpoint;
}

void prepare_counts(std::vector<Document>& documents, int32_t feature_count,
    const Eigen::VectorXd* feature_weights, Eigen::VectorXd& raw_totals,
    Eigen::VectorXd& effective_totals) {
    const bool weighted = feature_weights != nullptr;
    if (feature_count <= 0
        || (weighted && (feature_weights->size() != feature_count
            || !feature_weights->allFinite()
            || (feature_weights->array() < 0.0).any()))) {
        throw std::invalid_argument("Invalid UAC feature weights");
    }
    raw_totals.resize(documents.size());
    effective_totals.resize(documents.size());
    for (size_t d = 0; d < documents.size(); ++d) {
        Document& document = documents[d];
        if (document.ids.size() != document.cnts.size()) {
            throw std::runtime_error("Invalid UAC sparse document");
        }
        double raw = 0.0, effective = 0.0;
        size_t retained = 0;
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const uint32_t feature = document.ids[j];
            const double count = document.cnts[j];
            if (feature >= static_cast<uint32_t>(feature_count)
                || !std::isfinite(count) || count < 0.0) {
                throw std::runtime_error("Invalid UAC count or feature index");
            }
            raw += count;
            const double value = weighted
                ? count * (*feature_weights)(feature) : count;
            if (!std::isfinite(value)) {
                throw std::runtime_error("Nonfinite UAC weighted count");
            }
            effective += value;
            if (weighted && value > 0.0) {
                document.ids[retained] = feature;
                document.cnts[retained] = value;
                ++retained;
            }
        }
        if (weighted) {
            document.ids.resize(retained);
            document.cnts.resize(retained);
        }
        if (!(effective > 0.0) || !std::isfinite(effective)) {
            throw std::runtime_error(
                "UAC document has zero/nonfinite effective total");
        }
        document.raw_ct_tot = raw;
        document.ct_tot = effective;
        document.counts_weighted = weighted;
        raw_totals(d) = raw;
        effective_totals(d) = effective;
    }
}

} // namespace detail


void normalize_basis(Basis& basis) {
    if (basis.probabilities.rows() == 0 || basis.probabilities.cols() < 2
        || basis.features.size() != static_cast<size_t>(basis.probabilities.rows())
        || basis.topics.size() != static_cast<size_t>(basis.probabilities.cols())
        || !basis.probabilities.allFinite()
        || (basis.probabilities.array() < 0.0).any()) {
        throw std::invalid_argument("Invalid UAC topic basis");
    }
    detail::normalizePositiveColumnsInPlace(basis.probabilities);
    basis.checksum = basis_checksum(basis);
}

void normalize_centers(RowMajorMatrixXd& centers, double floor) {
    if (centers.rows() == 0 || centers.cols() < 2 || !centers.allFinite()
        || (centers.array() < 0.0).any() || !(floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC point centers");
    }
    for (Eigen::Index row = 0; row < centers.rows(); ++row) {
        centers.row(row) = centers.row(row).array().max(floor);
        centers.row(row) /= centers.row(row).sum();
    }
}

uint64_t basis_checksum(const Basis& basis) {
    uint64_t value = 14695981039346656037ull;
    for (const auto& name : basis.features) value = detail::hash_string(value, name);
    for (const auto& name : basis.topics) value = detail::hash_string(value, name);
    for (Eigen::Index row = 0; row < basis.probabilities.rows(); ++row) {
        for (Eigen::Index column = 0; column < basis.probabilities.cols(); ++column) {
            const double number = basis.probabilities(row, column);
            value = detail::fnv_append(value, &number, sizeof(number));
        }
    }
    return value;
}

} // namespace uac
