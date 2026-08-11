#include "clustering/uac_common_internal.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace uac::detail {


void validate_component_screening(
    const ComponentScreeningOptions& options) {
    switch (options.mode) {
        case ComponentScreeningMode::Off:
        case ComponentScreeningMode::On:
        case ComponentScreeningMode::Auto:
            break;
        default:
            throw std::invalid_argument(
                "Invalid UAC component screening mode");
    }
    if (!(options.tail_mass > 0.0 && options.tail_mass < 1.0)
        || !(options.proposal_proxy_tail_mass > 0.0
            && options.proposal_proxy_tail_mass < 1.0)
        || options.minimum_components <= 0
        || options.maximum_components < 0
        || (options.maximum_components > 0
            && options.maximum_components < options.minimum_components)
        || options.audit_documents < 0
        || !(options.minimum_work_reduction >= 0.0
            && options.minimum_work_reduction < 1.0)) {
        throw std::invalid_argument("Invalid UAC component screening options");
    }
}

int32_t checked_int32(Eigen::Index value, const char* name) {
    if (value < 0
        || value > std::numeric_limits<int32_t>::max()) {
        throw std::invalid_argument(
            std::string("UAC ") + name + " exceeds int32 capacity");
    }
    return static_cast<int32_t>(value);
}

void validate_dataset(const Dataset& data, bool require_counts) {
    const int32_t documents =
        checked_int32(data.coordinates.rows(), "document count");
    const int32_t dimension =
        checked_int32(data.coordinates.cols(), "coordinate dimension");
    if (documents <= 0 || dimension <= 0
        || data.identifiers.size() != static_cast<size_t>(documents)
        || data.centers.rows() != documents
        || data.centers.cols() != dimension + 1
        || !data.centers.allFinite() || !data.coordinates.allFinite()
        || (data.centers.array() < 0.0).any()
        || (data.centers.rowwise().sum().array() - 1.0)
            .abs().maxCoeff() > 1e-8
        || (data.raw_totals.size() != 0
            && data.raw_totals.size() != documents)
        || (data.effective_totals.size() != 0
            && data.effective_totals.size() != documents)
        || !data.raw_totals.allFinite()
        || !data.effective_totals.allFinite()
        || (data.raw_totals.array() < 0.0).any()
        || (data.effective_totals.array() < 0.0).any()
        || (require_counts
            && data.counts.size() != static_cast<size_t>(documents))) {
        throw std::invalid_argument("Invalid UAC dataset");
    }
    std::unordered_set<std::string> identifiers;
    identifiers.reserve(data.identifiers.size());
    for (const std::string& identifier : data.identifiers) {
        if (identifier.empty() || !identifiers.insert(identifier).second) {
            throw std::invalid_argument(
                "UAC document identifiers must be nonempty and unique");
        }
    }
    if (data.counts.empty()) return;
    if (data.counts.size() != static_cast<size_t>(documents)) {
        throw std::invalid_argument("UAC counts are not document-aligned");
    }
    for (const auto& document : data.counts) {
        if (document.ids.size() != document.cnts.size()) {
            throw std::invalid_argument("Invalid UAC document counts");
        }
        for (double count : document.cnts) {
            if (!(count >= 0.0) || !std::isfinite(count)) {
                throw std::invalid_argument("Invalid UAC document count");
            }
        }
    }
}

void validate_basis(const Basis& basis, int32_t topics,
    bool require_checksum) {
    if (basis.probabilities.rows() <= 0
        || basis.probabilities.cols() != topics
        || basis.features.size()
            != static_cast<size_t>(basis.probabilities.rows())
        || basis.topics.size() != static_cast<size_t>(topics)
        || !basis.probabilities.allFinite()
        || (basis.probabilities.array() < 0.0).any()
        || (basis.probabilities.colwise().sum().array() - 1.0)
            .abs().maxCoeff() > 1e-8
        || (require_checksum && basis.checksum != basis_checksum(basis))) {
        throw std::invalid_argument("Invalid UAC basis");
    }
}

void validate_count_features(const Dataset& data, const Basis& basis) {
    const uint64_t features =
        static_cast<uint64_t>(basis.probabilities.rows());
    for (const auto& document : data.counts) {
        for (uint32_t feature : document.ids) {
            if (feature >= features) {
                throw std::invalid_argument(
                    "UAC count feature is absent from the basis");
            }
        }
    }
}

DocumentBlock read_aligned_document_range(const Dataset& data,
    const IndexedDocumentSource& source, int32_t first, int32_t count) {
    DocumentBlock block;
    source.read_range(first, count, block);
    if (first < 0 || count < 0
        || static_cast<int64_t>(first) + count
            > static_cast<int64_t>(data.identifiers.size())
        || block.first_document != first
        || block.size() != count
        || block.identifiers.size() != static_cast<size_t>(count)
        || block.counts.size() != static_cast<size_t>(count)
        || block.raw_totals.size() != count
        || block.effective_totals.size() != count
        || !block.raw_totals.allFinite()
        || !block.effective_totals.allFinite()
        || (block.raw_totals.array() < 0.0).any()
        || (block.effective_totals.array() < 0.0).any()) {
        throw std::runtime_error(
            "Invalid indexed UAC document block");
    }
    for (int32_t local = 0; local < count; ++local) {
        const int32_t document = first + local;
        if (block.identifiers[local] != data.identifiers[document]) {
            throw std::runtime_error(
                "Indexed UAC count identifiers are not aligned with centers");
        }
        const Document& counts = block.counts[local];
        if (counts.ids.size() != counts.cnts.size()) {
            throw std::runtime_error(
                "Invalid indexed UAC sparse count record");
        }
        for (size_t entry = 0; entry < counts.ids.size(); ++entry) {
            if (counts.ids[entry]
                    >= static_cast<uint64_t>(source.features())
                || !std::isfinite(counts.cnts[entry])
                || counts.cnts[entry] < 0.0) {
                throw std::runtime_error(
                    "Invalid indexed UAC feature count");
            }
        }
    }
    return block;
}

bool has_nonidentity_feature_weights(const State& state) {
    return state.feature_weights.size() > 0
        && !(state.feature_weights.array() == 1.0).all();
}

void prepare_particle_score_counts(Dataset& data, const State& state) {
    if (!has_nonidentity_feature_weights(state)) return;
    if (data.counts.empty()) {
        throw std::invalid_argument(
            "Nonidentity UAC feature weights require resident counts");
    }
    bool has_raw = false;
    bool has_weighted = false;
    for (const Document& document : data.counts) {
        has_weighted = has_weighted || document.counts_weighted;
        has_raw = has_raw || !document.counts_weighted;
    }
    if (has_raw && has_weighted) {
        throw std::invalid_argument(
            "UAC score counts mix raw and preweighted documents");
    }
    if (has_raw) {
        detail::prepare_counts(data.counts,
            static_cast<int32_t>(state.feature_weights.size()),
            &state.feature_weights, data.raw_totals,
            data.effective_totals);
        return;
    }
    if (data.effective_totals.size()
            != static_cast<Eigen::Index>(data.counts.size())) {
        throw std::invalid_argument(
            "Preweighted UAC counts require effective totals");
    }
    for (size_t document = 0; document < data.counts.size(); ++document) {
        const double total = std::accumulate(
            data.counts[document].cnts.begin(),
            data.counts[document].cnts.end(), 0.0);
        const double tolerance =
            1e-10 * std::max({1.0, std::abs(total),
                std::abs(data.effective_totals(document))});
        if (std::abs(total - data.effective_totals(document)) > tolerance) {
            throw std::invalid_argument(
                "Preweighted UAC effective totals do not match counts");
        }
    }
}

ValidatingIndexedDocumentSource::ValidatingIndexedDocumentSource(
    IndexedDocumentSource& source, bool require_weighted)
    : source_(source), require_weighted_(require_weighted) {}

int64_t ValidatingIndexedDocumentSource::documents() const {
    return source_.documents();
}

int64_t ValidatingIndexedDocumentSource::features() const {
    return source_.features();
}

uint64_t ValidatingIndexedDocumentSource::storage_bytes() const {
    return source_.storage_bytes();
}

uint64_t ValidatingIndexedDocumentSource::content_checksum() const {
    return source_.content_checksum();
}

uint64_t ValidatingIndexedDocumentSource::peak_block_bytes() const {
    return source_.peak_block_bytes();
}

void ValidatingIndexedDocumentSource::reset() {
    source_.reset();
}

bool ValidatingIndexedDocumentSource::next(
    DocumentBlock& block, int32_t maximum_documents) {
    const bool found = source_.next(block, maximum_documents);
    if (found) validate_weighting(block);
    return found;
}

void ValidatingIndexedDocumentSource::read_range(
    int64_t first_document, int32_t documents, DocumentBlock& block) const {
    source_.read_range(first_document, documents, block);
    validate_weighting(block);
}

void ValidatingIndexedDocumentSource::validate_weighting(
    const DocumentBlock& block) const {
    if (!require_weighted_) return;
    for (const Document& document : block.counts) {
        if (!document.counts_weighted) {
            throw std::invalid_argument(
                "Indexed UAC counts must be preweighted to match state");
        }
    }
}

void validate_model(const Model& model) {
    const int32_t components =
        checked_int32(model.weights.size(), "component count");
    const int32_t dimension =
        checked_int32(model.means.cols(), "model dimension");
    if (components <= 0 || dimension <= 0
        || model.means.rows() != components
        || !model.weights.allFinite() || !model.means.allFinite()
        || (model.weights.array() < 0.0).any()
        || std::abs(model.weights.sum() - 1.0) > 1e-8
        || !(model.weights.array() > 0.0).any()) {
        throw std::invalid_argument("Invalid UAC model weights or means");
    }
    if (model.covariance_kind == CovarianceKind::Dense) {
        if (model.factor_diagonal_mode != FactorDiagonalMode::Component
            || model.covariances.size() != static_cast<size_t>(components)
            || model.shrinkage_target.rows() != dimension
            || model.shrinkage_target.cols() != dimension
            || !positive_definite(model.shrinkage_target)) {
            throw std::invalid_argument(
                "Invalid UAC dense covariance structure");
        }
        for (const auto& covariance : model.covariances) {
            if (covariance.rows() != dimension
                || covariance.cols() != dimension
                || !positive_definite(covariance)) {
                throw std::invalid_argument(
                    "Invalid UAC dense covariance");
            }
        }
        return;
    }
    if (model.factor_covariances.size()
            != static_cast<size_t>(components)
        || model.factor_covariances.empty()) {
        throw std::invalid_argument(
            "Invalid UAC factor covariance structure");
    }
    const Eigen::Index rank =
        model.factor_covariances.front().factor.cols();
    auto valid_factor_matrix = [&](const LowRankDiagonalCovariance& covariance) {
        return covariance.factor.rows() == dimension
            && covariance.factor.cols() == rank
            && covariance.factor.allFinite();
    };
    auto valid_diagonal = [&](const Eigen::VectorXd& diagonal) {
        return diagonal.size() == dimension && diagonal.allFinite()
            && (diagonal.array() > 0.0).all();
    };
    if (!valid_factor_matrix(model.factor_shrinkage_target)
        || !valid_diagonal(model.factor_shrinkage_target.diagonal)) {
        throw std::invalid_argument(
            "Invalid UAC factor shrinkage target");
    }
    for (const auto& covariance : model.factor_covariances) {
        if (!valid_factor_matrix(covariance)
            || (model.factor_diagonal_mode == FactorDiagonalMode::Component
                && !valid_diagonal(covariance.diagonal))
            || (model.factor_diagonal_mode == FactorDiagonalMode::Shared
                && covariance.diagonal.size() != 0)) {
            throw std::invalid_argument("Invalid UAC factor covariance");
        }
    }
    if (model.factor_diagonal_mode == FactorDiagonalMode::Shared) {
        if (!valid_diagonal(model.shared_factor_diagonal)
            || (model.factor_shrinkage_target.diagonal
                    - model.shared_factor_diagonal).cwiseAbs().maxCoeff()
                > 1e-12
                    * std::max(1.0,
                        model.shared_factor_diagonal.cwiseAbs().maxCoeff())
            || (model.factor_shrinkage_target.factor.array() != 0.0).any()) {
            throw std::invalid_argument(
                "Invalid UAC shared factor diagonal structure");
        }
    } else if (model.shared_factor_diagonal.size() != 0) {
        throw std::invalid_argument(
            "Unexpected UAC shared factor diagonal");
    }
}

void validate_pilot(const Pilot& pilot, int32_t components,
    int32_t dimension) {
    if (pilot.weights.size() != components
        || pilot.means.rows() != components
        || pilot.means.cols() != dimension
        || pilot.covariances.size() != static_cast<size_t>(components)
        || !pilot.weights.allFinite() || !pilot.means.allFinite()
        || (pilot.weights.array() < 0.0).any()
        || std::abs(pilot.weights.sum() - 1.0) > 1e-8
        || !positive_definite(pilot.pooled_covariance)) {
        throw std::invalid_argument("Invalid UAC pilot");
    }
    for (const auto& covariance : pilot.covariances) {
        if (covariance.rows() != dimension
            || covariance.cols() != dimension
            || !positive_definite(covariance)) {
            throw std::invalid_argument("Invalid UAC pilot covariance");
        }
    }
}

void validate_adaptive_particles(const AdaptiveParticleOptions& options,
    int32_t maximum_particles) {
    if (options.calibration_particles < 2
        || options.minimum_particles < options.calibration_particles
        || (options.enabled()
            && maximum_particles < options.minimum_particles)
        || (options.responsibility_se_target.has_value()
            && (!(*options.responsibility_se_target > 0.0)
                || !std::isfinite(*options.responsibility_se_target)))
        || (options.moment_ess_target.has_value()
            && (!(*options.moment_ess_target > 0.0)
                || !std::isfinite(*options.moment_ess_target)))
        || !(options.plausible_mass > 0.0
            && options.plausible_mass <= 1.0)
        || !(options.plausible_responsibility >= 0.0
            && options.plausible_responsibility <= 1.0)) {
        throw std::invalid_argument(
            "Invalid UAC adaptive particle options");
    }
}

void validate_state(const State& state) {
    validate_model(state.model);
    const int32_t components =
        checked_int32(state.model.weights.size(), "component count");
    const int32_t dimension =
        checked_int32(state.model.means.cols(), "state dimension");
    validate_pilot(state.pilot, components, dimension);
    validate_component_screening(state.component_screening);
    validate_adaptive_particles(
        state.fit_adaptive_particles, state.n_particles);
    const int64_t total_starts =
        static_cast<int64_t>(state.kmeans_starts) + state.leiden_starts;
    const bool selected_kind_matches = state.selected_start_method
            == StartMethod::KMeans
        ? state.selected_start < state.kmeans_starts
        : state.selected_start >= state.kmeans_starts;
    const int32_t model_rank =
        state.model.covariance_kind == CovarianceKind::Dense
        ? -1 : checked_int32(
            state.model.factor_covariances.front().factor.cols(),
            "factor covariance rank");
    const bool valid_handoff = state.handoff == HandoffMode::Map
        || state.handoff == HandoffMode::Particle;
    const bool valid_proposal = state.proposal == ProposalKind::ExactFisher
        || state.proposal == ProposalKind::SparseEmpiricalFisher;
    const bool valid_start_method =
        state.selected_start_method == StartMethod::KMeans
        || state.selected_start_method == StartMethod::Leiden;
    const bool valid_knn_backend =
        state.leiden_knn_backend == CosineKnnBackend::Auto
        || state.leiden_knn_backend == CosineKnnBackend::KdTree
        || state.leiden_knn_backend == CosineKnnBackend::Flat
        || state.leiden_knn_backend == CosineKnnBackend::Hnsw
        || state.leiden_knn_backend == CosineKnnBackend::NnDescent;
    const bool valid_initialization_metric =
        state.initialization_metric == SimplexMetric::Cosine
        || state.initialization_metric == SimplexMetric::Hellinger;
    if (!valid_handoff || !valid_proposal || !valid_start_method
        || !valid_knn_backend || !valid_initialization_metric
        || state.factor_diagonal_mode != state.model.factor_diagonal_mode
        || state.n_particles <= 0 || state.kmeans_starts < 0
        || state.leiden_starts < 0 || total_starts <= 0
        || state.kmeans_max_iterations <= 0
        || (state.leiden_starts > 0
            && (state.leiden_neighbors <= 0
                || state.leiden_max_iterations == 0))
        || state.selected_start < 0 || state.selected_start >= total_starts
        || !selected_kind_matches
        || !std::isfinite(state.selected_leiden_resolution)
        || (state.selected_start_method == StartMethod::Leiden
            && !(state.selected_leiden_resolution > 0.0))
        || state.cluster_covariance_rank != model_rank
        || state.topics.size() != static_cast<size_t>(dimension + 1)
        || state.helmert.rows() != dimension
        || state.helmert.cols() != dimension + 1
        || !is_normalized_helmert(state.helmert)
        || !(state.center_floor > 0.0)
        || !(state.target_relative_floor > 0.0)
        || !(state.covariance_floor > 0.0)
        || !(state.objective_change_tolerance > 0.0)
        || !(state.responsibility_change_tolerance > 0.0)
        || !(state.particle_variance_change_tolerance >= 0.0)
        || !(state.initialization_ridge_precision >= 0.0)
        || !(state.leiden_knn_epsilon >= 0.0)
        || state.leiden_hnsw_m <= 0
        || state.leiden_hnsw_ef_construction <= 0
        || state.leiden_hnsw_ef_search < 0
        || state.leiden_hnsw_max_ef_search <= 0
        || state.leiden_hnsw_candidates < 0
        || state.leiden_hnsw_audit_queries <= 0
        || !(state.leiden_hnsw_recall > 0.0
            && state.leiden_hnsw_recall <= 1.0)
        || state.leiden_nndescent_iterations < 0
        || state.leiden_nndescent_graph_size < 0
        || state.leiden_nndescent_sample_candidates <= 0
        || state.leiden_nndescent_audit_queries <= 0
        || !(state.leiden_nndescent_recall > 0.0
            && state.leiden_nndescent_recall <= 1.0)
        || state.leiden_resolved_ann_parameter < 0
        || state.leiden_resolved_ann_candidates < 0
        || !(state.leiden_ann_audit_mean_recall >= 0.0
            && state.leiden_ann_audit_mean_recall <= 1.0)
        || !(state.leiden_ann_audit_recall_lcb >= 0.0
            && state.leiden_ann_audit_recall_lcb <= 1.0)
        || ((state.leiden_knn_backend == CosineKnnBackend::Hnsw
                || state.leiden_knn_backend == CosineKnnBackend::NnDescent)
            && state.leiden_starts > 0
            && (state.leiden_resolved_ann_parameter <= 0
                || state.leiden_resolved_ann_candidates <= 0
                || (!state.leiden_ann_audit_passed
                    && !state.leiden_ann_forced)))
        || !(state.leiden_resolution > 0.0)
        || !(state.covariance_shrinkage_strength >= 0.0)
        || !(state.fisher_broadening > 0.0)
        || state.fisher_refinement_iterations <= 0
        || !state.feature_weights.allFinite()
        || (state.feature_weights.array() < 0.0).any()
        || (state.feature_weights.size() > 0 && !state.weighted_counts)
        || (state.handoff == HandoffMode::Particle
            && state.basis_checksum == 0)) {
        throw std::invalid_argument("Invalid UAC state");
    }
}



void apply_auto_component_screening_resolution(
    ComponentScreeningOptions& options, bool enabled) {
    options.mode = enabled
        ? ComponentScreeningMode::On : ComponentScreeningMode::Off;
    // A hard maximum is an explicitly forced approximation. Automatic
    // screening continues to be governed only by its audited tail criteria.
    options.maximum_components = 0;
}



double weighted_hpd_threshold(
    const Eigen::Ref<const Eigen::VectorXd>& log_density,
    const Eigen::Ref<const Eigen::VectorXd>& probability, double level) {
    std::vector<int32_t> order(log_density.size());
    std::iota(order.begin(), order.end(), int32_t{0});
    std::stable_sort(order.begin(), order.end(), [&](int32_t left,
            int32_t right) { return log_density(left) > log_density(right); });
    double cumulative = 0.0;
    for (const int32_t index : order) {
        cumulative += probability(index);
        if (cumulative >= level) return log_density(index);
    }
    return log_density(order.back());
}

uint64_t fnv_append(uint64_t value, const void* data, size_t size) {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < size; ++i) {
        value ^= bytes[i];
        value *= 1099511628211ull;
    }
    return value;
}

uint64_t hash_string(uint64_t value, const std::string& text) {
    value = fnv_append(value, text.data(), text.size());
    const unsigned char separator = 0xff;
    return fnv_append(value, &separator, 1);
}

double log_gaussian(const Eigen::Ref<const Eigen::VectorXd>& value,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance) {
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("UAC covariance is not positive definite");
    }
    const Eigen::VectorXd residual = value - mean;
    const Eigen::MatrixXd lower = llt.matrixL();
    const double logdet = 2.0 * lower.diagonal().array().log().sum();
    return -0.5 * (value.size() * kLog2Pi + logdet
        + residual.dot(llt.solve(residual)));
}

DenseGaussianSolver::DenseGaussianSolver(
    const Eigen::Ref<const Eigen::VectorXd>& input_mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance)
    : mean(input_mean) {
    Eigen::LLT<Eigen::MatrixXd> llt(covariance);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error("UAC covariance is not positive definite");
    }
    lower = llt.matrixL();
    log_determinant = 2.0 * lower.diagonal().array().log().sum();
}

double DenseGaussianSolver::log_density(
    const Eigen::Ref<const Eigen::VectorXd>& value) const {
    Eigen::VectorXd standardized;
    return log_density(value, standardized);
}

double DenseGaussianSolver::log_density(
    const Eigen::Ref<const Eigen::VectorXd>& value,
    Eigen::VectorXd& standardized) const {
    standardized = value - mean;
    lower.triangularView<Eigen::Lower>().solveInPlace(standardized);
    return -0.5 * (value.size() * kLog2Pi + log_determinant
        + standardized.squaredNorm());
}

Eigen::VectorXd DenseGaussianSolver::log_density_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& values) const {
    Eigen::MatrixXd standardized;
    Eigen::VectorXd output;
    log_density_rows(values, standardized, output);
    return output;
}

void DenseGaussianSolver::log_density_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    Eigen::MatrixXd& standardized, Eigen::VectorXd& output) const {
    standardized.resize(mean.size(), values.rows());
    standardized = (values.rowwise() - mean.transpose()).transpose();
    lower.triangularView<Eigen::Lower>().solveInPlace(standardized);
    output.resize(values.rows());
    output = (-0.5 * (mean.size() * kLog2Pi + log_determinant
        + standardized.colwise().squaredNorm().array())).matrix();
}

std::vector<DenseGaussianSolver> dense_model_solvers(const Model& model) {
    std::vector<DenseGaussianSolver> out(model.weights.size());
    if (model.covariance_kind != CovarianceKind::Dense) return out;
    for (Eigen::Index c = 0; c < model.weights.size(); ++c) {
        if (model.weights(c) > 0.0) {
            out[c] = DenseGaussianSolver(
                model.means.row(c).transpose(), model.covariances[c]);
        }
    }
    return out;
}

Eigen::MatrixXd model_covariance_dense(const Model& model,
    int32_t component) {
    if (model.covariance_kind == CovarianceKind::Dense) {
        return model.covariances[component];
    }
    Eigen::MatrixXd out = factor_diagonal(model, component).asDiagonal();
    const auto& factor = model.factor_covariances[component].factor;
    if (factor.cols() > 0) out.noalias() += factor * factor.transpose();
    return out;
}

const Eigen::VectorXd& factor_diagonal(const Model& model,
    int32_t component) {
    return model.factor_diagonal_mode == FactorDiagonalMode::Shared
        ? model.shared_factor_diagonal
        : model.factor_covariances[component].diagonal;
}

void validate_particle_initial_model(const Model& model,
    const Model& reference) {
    validate_model(model);
    validate_model(reference);
    if (model.covariance_kind != reference.covariance_kind
        || model.weights.size() != reference.weights.size()
        || model.means.rows() != reference.means.rows()
        || model.means.cols() != reference.means.cols()) {
        throw std::invalid_argument(
            "Particle initial model does not match the fitted model shape");
    }
    if (model.covariance_kind == CovarianceKind::FactorAnalytic
        && (model.factor_diagonal_mode != reference.factor_diagonal_mode
            || model.factor_covariances.front().factor.cols()
                != reference.factor_covariances.front().factor.cols())) {
            throw std::invalid_argument(
                "Particle initial model factor rank or diagonal mode differs");
    }
}

std::vector<double> model_eigenvalue_upper_bounds(const Model& model) {
    std::vector<double> out(model.weights.size(), 1.0);
    for (Eigen::Index c = 0; c < model.weights.size(); ++c) {
        if (!(model.weights(c) > 0.0)) continue;
        double value = 0.0;
        if (model.covariance_kind == CovarianceKind::Dense) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
                model.covariances[c], Eigen::EigenvaluesOnly);
            if (eigen.info() != Eigen::Success) {
                throw std::runtime_error(
                    "UAC dense screening eigenvalue bound failed");
            }
            value = eigen.eigenvalues().maxCoeff();
        } else {
            const auto& covariance = model.factor_covariances[c];
            value = factor_diagonal(model, static_cast<int32_t>(c)).maxCoeff();
            if (covariance.factor.cols() > 0) {
                Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
                    covariance.factor.transpose() * covariance.factor,
                    Eigen::EigenvaluesOnly);
                if (eigen.info() != Eigen::Success) {
                    throw std::runtime_error(
                        "UAC factor screening eigenvalue bound failed");
                }
                value += eigen.eigenvalues().maxCoeff();
            }
        }
        if (!(value > 0.0) || !std::isfinite(value)) {
            throw std::runtime_error(
                "UAC component screening covariance bound is invalid");
        }
        out[c] = std::nextafter(value,
            std::numeric_limits<double>::infinity());
    }
    return out;
}

LowRankDiagonalCovariance factorize_covariance(
    const Eigen::Ref<const Eigen::MatrixXd>& covariance, int32_t rank,
    double floor) {
    const int32_t dimension = static_cast<int32_t>(covariance.rows());
    if (covariance.cols() != dimension || rank < 0 || rank > dimension
        || !(floor > 0.0)) {
        throw std::invalid_argument("Invalid factor covariance conversion");
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
        0.5 * (covariance + covariance.transpose()));
    if (eigen.info() != Eigen::Success) {
        throw std::runtime_error("UAC factor covariance eigendecomposition failed");
    }
    LowRankDiagonalCovariance out;
    out.factor = RowMajorMatrixXd::Zero(dimension, rank);
    for (int32_t j = 0; j < rank; ++j) {
        const int32_t index = dimension - 1 - j;
        const double value = std::max(0.0, eigen.eigenvalues()(index) - floor);
        out.factor.col(j) = std::sqrt(value) * eigen.eigenvectors().col(index);
    }
    out.diagonal = (covariance.diagonal()
        - (out.factor.array().square().rowwise().sum()).matrix())
        .cwiseMax(floor);
    return out;
}

void convert_model_to_factor(Model& model, int32_t rank,
    FactorDiagonalMode diagonal_mode, double floor) {
    const int32_t dimension = checked_int32(
        model.means.cols(), "factor covariance dimension");
    if (model.covariance_kind != CovarianceKind::Dense
        || model.covariances.size()
            != static_cast<size_t>(model.weights.size())
        || rank < 0 || rank > dimension || !(floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC factor model conversion");
    }
    std::vector<LowRankDiagonalCovariance> provisional;
    provisional.reserve(model.covariances.size());
    for (const auto& covariance : model.covariances) {
        provisional.push_back(factorize_covariance(covariance, rank, floor));
    }
    model.covariance_kind = CovarianceKind::FactorAnalytic;
    model.factor_diagonal_mode = diagonal_mode;
    model.factor_covariances = std::move(provisional);
    model.shared_factor_diagonal.resize(0);
    if (diagonal_mode == FactorDiagonalMode::Component) {
        model.factor_shrinkage_target = factorize_covariance(
            model.shrinkage_target, rank, floor);
        return;
    }

    model.shared_factor_diagonal = Eigen::VectorXd::Zero(dimension);
    double active_weight = 0.0;
    for (int32_t c = 0; c < model.weights.size(); ++c) {
        if (!(model.weights(c) > 0.0)) continue;
        active_weight += model.weights(c);
        model.shared_factor_diagonal.noalias() += model.weights(c)
            * model.factor_covariances[c].diagonal;
    }
    if (!(active_weight > 0.0)) {
        throw std::runtime_error(
            "UAC shared factor initialization has no active component");
    }
    model.shared_factor_diagonal =
        (model.shared_factor_diagonal / active_weight).cwiseMax(floor);
    const Eigen::MatrixXd shared =
        model.shared_factor_diagonal.asDiagonal();
    for (int32_t c = 0; c < model.weights.size(); ++c) {
        Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigen(
            0.5 * (model.covariances[c] - shared
                + (model.covariances[c] - shared).transpose()));
        if (eigen.info() != Eigen::Success) {
            throw std::runtime_error(
                "UAC shared factor initialization eigendecomposition failed");
        }
        auto& covariance = model.factor_covariances[c];
        covariance.factor = RowMajorMatrixXd::Zero(dimension, rank);
        for (int32_t j = 0; j < rank; ++j) {
            const int32_t index = dimension - 1 - j;
            const double value = std::max(0.0, eigen.eigenvalues()(index));
            covariance.factor.col(j) = std::sqrt(value)
                * eigen.eigenvectors().col(index);
        }
        covariance.diagonal.resize(0);
    }
    model.factor_shrinkage_target.diagonal =
        model.shared_factor_diagonal;
    model.factor_shrinkage_target.factor =
        RowMajorMatrixXd::Zero(dimension, rank);
}

double covariance_prior(const Model& model, double strength) {
    double out = 0.0;
    if (model.covariance_kind == CovarianceKind::FactorAnalytic) {
        if (model.factor_diagonal_mode == FactorDiagonalMode::Shared) {
            const Eigen::VectorXd inverse =
                model.shared_factor_diagonal.cwiseInverse();
            for (const auto& covariance : model.factor_covariances) {
                out -= 0.5 * strength
                    * (covariance.factor.array().square().rowwise().sum()
                        * inverse.array()).sum();
            }
            return out;
        }
        for (const auto& covariance : model.factor_covariances) {
            LowRankDiagonalSolver solver(
                covariance.diagonal, covariance.factor);
            const Eigen::MatrixXd core_inverse = solver.solve_core(
                Eigen::MatrixXd::Identity(covariance.factor.cols(),
                    covariance.factor.cols()));
            Eigen::VectorXd inverse_diagonal = solver.inverse_diagonal();
            if (covariance.factor.cols() > 0) {
                const Eigen::MatrixXd& scaled =
                    solver.inverse_diagonal_factor();
                inverse_diagonal.array() -= (scaled * core_inverse).cwiseProduct(
                    scaled).rowwise().sum().array();
            }
            double trace = inverse_diagonal.dot(
                model.factor_shrinkage_target.diagonal);
            if (model.factor_shrinkage_target.factor.cols() > 0) {
                trace += (model.factor_shrinkage_target.factor.transpose()
                    * solver.solve_matrix(
                        model.factor_shrinkage_target.factor)).trace();
            }
            out -= 0.5 * strength * (solver.log_determinant() + trace);
        }
        return out;
    }
    for (const auto& covariance : model.covariances) {
        Eigen::LLT<Eigen::MatrixXd> llt(covariance);
        if (llt.info() != Eigen::Success) {
            return -std::numeric_limits<double>::infinity();
        }
        const Eigen::MatrixXd lower = llt.matrixL();
        const double logdet = 2.0 * lower.diagonal().array().log().sum();
        out -= 0.5 * strength * (logdet
            + (llt.solve(model.shrinkage_target)).trace());
    }
    return out;
}

int32_t active_component_count(const Model& model) {
    return static_cast<int32_t>((model.weights.array() > 0.0).count());
}

double membership_epsilon(int32_t documents) {
    return std::max(1e-12, 64.0
        * std::numeric_limits<double>::epsilon() * documents);
}

int32_t map_start_seed(int32_t seed, int32_t start) {
    constexpr uint64_t modulus = 2147483647ull;
    const uint64_t value = static_cast<uint32_t>(seed)
        + 104729ull * static_cast<uint32_t>(start);
    return static_cast<int32_t>(value % modulus);
}

Eigen::VectorXd count_log_likelihood_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    if (coordinates.cols() != helmert.rows()
        || basis.probabilities.cols() != helmert.cols()
        || document.ids.size() != document.cnts.size()) {
        throw std::invalid_argument("Invalid UAC batched likelihood input");
    }
    const Eigen::Index samples = coordinates.rows();
    Eigen::MatrixXd logits = coordinates * helmert;
    for (Eigen::Index s = 0; s < samples; ++s) {
        logits.row(s).array() -= logits.row(s).maxCoeff();
        logits.row(s) = logits.row(s).array().exp();
        logits.row(s) /= logits.row(s).sum();
    }
    if (document.ids.empty()) return Eigen::VectorXd::Zero(samples);
    RowMajorMatrixXd observed(document.ids.size(), basis.probabilities.cols());
    for (size_t j = 0; j < document.ids.size(); ++j) {
        const uint32_t feature = document.ids[j];
        const double count = document.cnts[j];
        if (feature >= static_cast<uint32_t>(basis.probabilities.rows())) {
            throw std::runtime_error("UAC document feature index is out of range");
        }
        if (!std::isfinite(count) || count < 0.0) {
            throw std::runtime_error("UAC document count is invalid");
        }
        observed.row(j) = basis.probabilities.row(feature);
    }
    const Eigen::MatrixXd probability = observed * logits.transpose();
    Eigen::VectorXd out = Eigen::VectorXd::Zero(samples);
    for (size_t j = 0; j < document.ids.size(); ++j) {
        if (!(document.cnts[j] > 0.0)) continue;
        for (Eigen::Index s = 0; s < samples; ++s) {
            const double value = probability(j, s);
            if (!(value > 0.0) || !std::isfinite(value)) {
                out(s) = -std::numeric_limits<double>::infinity();
            } else if (std::isfinite(out(s))) {
                out(s) += document.cnts[j] * std::log(value);
            }
        }
    }
    return out;
}

const char* adaptive_particle_mode_name(
    const AdaptiveParticleOptions& options) {
    if (options.responsibility_se_target.has_value()
        && options.moment_ess_target.has_value()) return "both";
    if (options.responsibility_se_target.has_value()) return "resp";
    if (options.moment_ess_target.has_value()) return "moment";
    return "off";
}

double optional_target_or_zero(const std::optional<double>& value) {
    return value.value_or(0.0);
}

} // namespace uac::detail
