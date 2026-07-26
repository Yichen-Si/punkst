#include "clustering/uac_internal.hpp"
#include "clustering/uac_stream.hpp"

#include "clustering_core/cosine_clustering.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <variant>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace uac {
namespace {

constexpr double kLog2Pi = 1.83787706640934548356;

double logsumexp(const Eigen::Ref<const Eigen::VectorXd>& values) {
    const double maximum = values.maxCoeff();
    if (!std::isfinite(maximum)) return maximum;
    return maximum + std::log((values.array() - maximum).exp().sum());
}

double logaddexp(double left, double right) {
    if (!std::isfinite(left)) return right;
    if (!std::isfinite(right)) return left;
    const double maximum = std::max(left, right);
    return maximum + std::log(
        std::exp(left - maximum) + std::exp(right - maximum));
}

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

bool positive_definite(const Eigen::MatrixXd& covariance) {
    return covariance.rows() > 0 && covariance.rows() == covariance.cols()
        && covariance.allFinite()
        && (covariance - covariance.transpose()).cwiseAbs().maxCoeff()
            <= 1e-8
        && Eigen::LLT<Eigen::MatrixXd>(covariance).info()
            == Eigen::Success;
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
    bool require_checksum = true) {
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
        if (model.covariances.size() != static_cast<size_t>(components)
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
    auto valid_factor = [&](const LowRankDiagonalCovariance& covariance) {
        return covariance.diagonal.size() == dimension
            && covariance.factor.rows() == dimension
            && covariance.factor.cols() == rank
            && covariance.diagonal.allFinite()
            && covariance.factor.allFinite()
            && (covariance.diagonal.array() > 0.0).all();
    };
    if (!valid_factor(model.factor_shrinkage_target)) {
        throw std::invalid_argument(
            "Invalid UAC factor shrinkage target");
    }
    for (const auto& covariance : model.factor_covariances) {
        if (!valid_factor(covariance)) {
            throw std::invalid_argument("Invalid UAC factor covariance");
        }
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
    const Eigen::MatrixXd expected_helmert =
        normalized_helmert(dimension + 1);
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
        || state.leiden_knn_backend == CosineKnnBackend::Flat;
    if (!valid_handoff || !valid_proposal || !valid_start_method
        || !valid_knn_backend
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
        || !state.helmert.allFinite()
        || (state.helmert - expected_helmert)
            .cwiseAbs().maxCoeff() > 1e-12
        || !(state.center_floor > 0.0)
        || !(state.target_relative_floor > 0.0)
        || !(state.covariance_floor > 0.0)
        || !(state.objective_change_tolerance > 0.0)
        || !(state.responsibility_change_tolerance > 0.0)
        || !(state.particle_variance_change_tolerance >= 0.0)
        || !(state.initialization_ridge_precision >= 0.0)
        || !(state.leiden_knn_epsilon >= 0.0)
        || !(state.leiden_resolution > 0.0)
        || !(state.covariance_shrinkage_strength >= 0.0)
        || !(state.fisher_broadening > 0.0)
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

struct ScreenedComponents {
    Eigen::VectorXd score;
    std::vector<int32_t> evaluated;
    double log_mass = -std::numeric_limits<double>::infinity();
    double log_upper_mass = -std::numeric_limits<double>::infinity();
    double omitted_mass_bound = 0.0;
    bool full = true;
    bool bound_violation = false;
};

struct ComponentScreeningWorkspace {
    std::vector<int32_t> order;
    std::vector<double> suffix;
};

template<class Evaluate>
ScreenedComponents screen_component_scores(
    const Eigen::Ref<const Eigen::VectorXd>& upper,
    const ComponentScreeningOptions& options, bool enabled,
    Evaluate&& evaluate, ComponentScreeningWorkspace* supplied = nullptr) {
    const int32_t components = static_cast<int32_t>(upper.size());
    ScreenedComponents out;
    out.score = Eigen::VectorXd::Constant(
        components, -std::numeric_limits<double>::infinity());
    out.evaluated.reserve(components);
    ComponentScreeningWorkspace local;
    ComponentScreeningWorkspace& workspace =
        supplied == nullptr ? local : *supplied;
    std::vector<int32_t>& order = workspace.order;
    order.clear();
    order.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        if (std::isfinite(upper(c))) order.push_back(c);
    }
    if (order.empty()) {
        throw std::runtime_error("UAC component screen has no active component");
    }
    std::stable_sort(order.begin(), order.end(), [&](int32_t left,
            int32_t right) {
        return upper(left) == upper(right)
            ? left < right : upper(left) > upper(right);
    });
    std::vector<double>& suffix = workspace.suffix;
    suffix.assign(order.size() + 1,
        -std::numeric_limits<double>::infinity());
    for (size_t i = order.size(); i > 0; --i) {
        suffix[i - 1] = logaddexp(upper(order[i - 1]), suffix[i]);
    }
    const int32_t minimum = enabled
        ? std::min<int32_t>(options.minimum_components, order.size())
        : static_cast<int32_t>(order.size());
    const int32_t maximum =
        enabled && options.mode == ComponentScreeningMode::On
            && options.maximum_components > 0
        ? std::min<int32_t>(options.maximum_components, order.size())
        : static_cast<int32_t>(order.size());
    for (size_t rank = 0; rank < order.size(); ++rank) {
        const int32_t component = order[rank];
        const double exact = evaluate(component);
        out.score(component) = exact;
        out.evaluated.push_back(component);
        out.log_mass = logaddexp(out.log_mass, exact);
        const double tolerance = 1e-10
            * std::max(1.0, std::abs(upper(component)));
        if (exact > upper(component) + tolerance) {
            out.bound_violation = true;
            enabled = false;
        }
        const bool reached_maximum =
            static_cast<int32_t>(out.evaluated.size()) >= maximum;
        if (reached_maximum) {
            out.log_upper_mass = suffix[rank + 1];
            const double combined =
                logaddexp(out.log_mass, out.log_upper_mass);
            out.omitted_mass_bound = std::isfinite(out.log_upper_mass)
                ? std::exp(out.log_upper_mass - combined) : 0.0;
            if (out.bound_violation && rank + 1 < order.size()) {
                out.omitted_mass_bound = 1.0;
            }
            break;
        }
        if (!enabled
            || static_cast<int32_t>(out.evaluated.size()) < minimum) {
            continue;
        }
        out.log_upper_mass = suffix[rank + 1];
        const double combined =
            logaddexp(out.log_mass, out.log_upper_mass);
        out.omitted_mass_bound = std::isfinite(out.log_upper_mass)
            ? std::exp(out.log_upper_mass - combined) : 0.0;
        if (out.omitted_mass_bound <= options.tail_mass) break;
    }
    const bool forced_maximum =
        options.mode == ComponentScreeningMode::On
        && options.maximum_components > 0;
    if (!enabled && !forced_maximum
        && out.evaluated.size() < order.size()) {
        for (size_t rank = out.evaluated.size(); rank < order.size(); ++rank) {
            const int32_t component = order[rank];
            const double exact = evaluate(component);
            out.score(component) = exact;
            out.evaluated.push_back(component);
            out.log_mass = logaddexp(out.log_mass, exact);
        }
        out.log_upper_mass = -std::numeric_limits<double>::infinity();
        out.omitted_mass_bound = 0.0;
    }
    out.full = out.evaluated.size() == order.size();
    if (out.full) {
        out.log_upper_mass = -std::numeric_limits<double>::infinity();
        out.omitted_mass_bound = 0.0;
    }
    return out;
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

Eigen::MatrixXd floor_covariance(const Eigen::Ref<const Eigen::MatrixXd>& input,
    double floor) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        0.5 * (input + input.transpose()));
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("UAC covariance eigendecomposition failed");
    }
    return solver.eigenvectors()
        * solver.eigenvalues().cwiseMax(floor).asDiagonal()
        * solver.eigenvectors().transpose();
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

struct DenseGaussianSolver {
    Eigen::VectorXd mean;
    Eigen::MatrixXd lower;
    double log_determinant = 0.0;

    DenseGaussianSolver() = default;

    DenseGaussianSolver(const Eigen::Ref<const Eigen::VectorXd>& input_mean,
        const Eigen::Ref<const Eigen::MatrixXd>& covariance)
        : mean(input_mean) {
        Eigen::LLT<Eigen::MatrixXd> llt(covariance);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error("UAC covariance is not positive definite");
        }
        lower = llt.matrixL();
        log_determinant = 2.0 * lower.diagonal().array().log().sum();
    }

    double log_density(
        const Eigen::Ref<const Eigen::VectorXd>& value) const {
        Eigen::VectorXd standardized = value - mean;
        lower.triangularView<Eigen::Lower>().solveInPlace(standardized);
        return -0.5 * (value.size() * kLog2Pi + log_determinant
            + standardized.squaredNorm());
    }

    Eigen::VectorXd log_density_rows(
        const Eigen::Ref<const RowMajorMatrixXd>& values) const {
        Eigen::MatrixXd standardized =
            (values.rowwise() - mean.transpose()).transpose();
        lower.triangularView<Eigen::Lower>().solveInPlace(standardized);
        return (-0.5 * (mean.size() * kLog2Pi + log_determinant
            + standardized.colwise().squaredNorm().array())).matrix();
    }
};

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
    return model.covariance_kind == CovarianceKind::Dense
        ? model.covariances[component]
        : model.factor_covariances[component].dense();
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
        && model.factor_covariances.front().factor.cols()
            != reference.factor_covariances.front().factor.cols()) {
            throw std::invalid_argument(
                "Particle initial model factor rank differs");
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
            value = covariance.diagonal.maxCoeff();
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

double covariance_prior(const Model& model, double strength) {
    double out = 0.0;
    if (model.covariance_kind == CovarianceKind::FactorAnalytic) {
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

Eigen::VectorXd composition_from_coordinate(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    Eigen::VectorXd logits = helmert.transpose() * coordinate;
    logits.array() -= logits.maxCoeff();
    Eigen::VectorXd values = logits.array().exp();
    values /= values.sum();
    return values;
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

FisherApproximation fisher_approximation_impl(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal) {
    if (coordinate.size() != helmert.rows()
        || basis.probabilities.cols() != helmert.cols()
        || document.ids.size() != document.cnts.size()
        || (proposal != ProposalKind::ExactFisher
            && proposal != ProposalKind::SparseEmpiricalFisher)) {
        throw std::invalid_argument("Invalid UAC Fisher input");
    }
    const Eigen::VectorXd composition = composition_from_coordinate(
        coordinate, helmert);
    Eigen::MatrixXd simplex_covariance = composition.asDiagonal();
    simplex_covariance.noalias() -= composition * composition.transpose();
    const Eigen::MatrixXd simplex_derivative =
        simplex_covariance * helmert.transpose();
    FisherApproximation out;
    out.gradient = Eigen::VectorXd::Zero(coordinate.size());
    out.information = Eigen::MatrixXd::Zero(
        coordinate.size(), coordinate.size());
    if (proposal == ProposalKind::SparseEmpiricalFisher) {
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const uint32_t feature = document.ids[j];
            const double count = document.cnts[j];
            if (feature >= static_cast<uint32_t>(basis.probabilities.rows())) {
                throw std::runtime_error(
                    "UAC document feature index is out of range");
            }
            if (!std::isfinite(count) || count < 0.0) {
                throw std::runtime_error("UAC document count is invalid");
            }
            if (!(count > 0.0)) continue;
            const double probability =
                basis.probabilities.row(feature).dot(composition);
            if (!(probability > 0.0) || !std::isfinite(probability)) {
                throw std::runtime_error(
                    "UAC observed feature has zero Fisher probability");
            }
            const Eigen::VectorXd derivative = (
                basis.probabilities.row(feature) * simplex_derivative)
                .transpose();
            const Eigen::VectorXd score = derivative / probability;
            out.gradient.noalias() += count * score;
            out.information.noalias() += count
                * score * score.transpose();
        }
    } else {
        const Eigen::VectorXd probability = (basis.probabilities * composition)
            .array().max(1e-300).matrix();
        const RowMajorMatrixXd probability_derivative =
            basis.probabilities * simplex_derivative;
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const uint32_t feature = document.ids[j];
            const double count = document.cnts[j];
            if (feature >= static_cast<uint32_t>(probability.size())) {
                throw std::runtime_error(
                    "UAC document feature index is out of range");
            }
            if (!std::isfinite(count) || count < 0.0) {
                throw std::runtime_error("UAC document count is invalid");
            }
            if (!(count > 0.0)) continue;
            out.gradient.noalias() += count
                / probability(feature)
                * probability_derivative.row(feature).transpose();
        }
        const double total = std::accumulate(document.cnts.begin(),
            document.cnts.end(), 0.0);
        const Eigen::VectorXd curvature =
            Eigen::VectorXd::Constant(probability.size(), total).array()
                / probability.array();
        out.information.noalias() = probability_derivative.transpose()
            * curvature.asDiagonal() * probability_derivative;
    }
    out.information = 0.5 * (
        out.information + out.information.transpose());
    return out;
}

struct PilotCache {
    std::vector<Eigen::MatrixXd> inverse_covariances;
    Eigen::VectorXd log_determinants;

    explicit PilotCache(const Pilot& pilot)
        : log_determinants(pilot.weights.size()) {
        inverse_covariances.reserve(pilot.weights.size());
        for (Eigen::Index c = 0; c < pilot.weights.size(); ++c) {
            Eigen::LLT<Eigen::MatrixXd> llt(pilot.covariances[c]);
            if (llt.info() != Eigen::Success) {
                throw std::runtime_error(
                    "UAC Fisher pilot covariance is not positive definite");
            }
            inverse_covariances.push_back(llt.solve(
                Eigen::MatrixXd::Identity(
                    pilot.covariances[c].rows(),
                    pilot.covariances[c].cols())));
            const Eigen::MatrixXd lower = llt.matrixL();
            log_determinants(c) =
                2.0 * lower.diagonal().array().log().sum();
        }
    }
};

struct DocumentProposal {
    Eigen::VectorXd weights;
    std::vector<int32_t> component_ids;
    std::vector<Eigen::VectorXd> means;
    std::vector<Eigen::MatrixXd> precision_lower;
    Eigen::VectorXd log_precision_determinants;
    double broadening = 1.0;
    double precision_fallback_seconds = 0.0;
    int64_t precision_fallbacks = 0;
};

DocumentProposal fisher_proposal(
    const Eigen::Ref<const Eigen::VectorXd>& center,
    const FisherApproximation& fisher, const Pilot& pilot,
    const PilotCache& cache, double broadening,
    const std::vector<int32_t>* candidate_components = nullptr) {
    if (!(broadening > 0.0) || !std::isfinite(broadening)) {
        throw std::invalid_argument("Invalid UAC Fisher broadening");
    }
    const int32_t dimension = static_cast<int32_t>(center.size());
    DocumentProposal out;
    out.broadening = broadening;
    if (candidate_components) {
        out.component_ids = *candidate_components;
    } else {
        out.component_ids.resize(pilot.weights.size());
        std::iota(out.component_ids.begin(), out.component_ids.end(), 0);
    }
    if (out.component_ids.empty()) {
        throw std::runtime_error("UAC proposal has no candidate component");
    }
    out.weights.resize(out.component_ids.size());
    out.means.reserve(out.component_ids.size());
    out.precision_lower.reserve(out.component_ids.size());
    out.log_precision_determinants.resize(out.component_ids.size());
    Eigen::VectorXd log_weight(out.component_ids.size());
    for (size_t j = 0; j < out.component_ids.size(); ++j) {
        const int32_t c = out.component_ids[j];
        if (c < 0 || c >= pilot.weights.size()) {
            throw std::runtime_error(
                "UAC proposal candidate component is out of range");
        }
        if (!(pilot.weights(c) > 0.0)) {
            out.means.push_back(center);
            out.precision_lower.push_back(Eigen::MatrixXd::Identity(
                dimension, dimension));
            out.log_precision_determinants(j) = 0.0;
            log_weight(j) = -std::numeric_limits<double>::infinity();
            continue;
        }
        const Eigen::MatrixXd& inverse_covariance =
            cache.inverse_covariances[c];
        const double pilot_logdet = cache.log_determinants(c);
        const Eigen::VectorXd pilot_mean = pilot.means.row(c).transpose();
        const Eigen::VectorXd residual = center - pilot_mean;
        const Eigen::VectorXd b = fisher.gradient
            - inverse_covariance * residual;
        const Eigen::MatrixXd raw_precision = 0.5
            * (fisher.information + inverse_covariance
                + fisher.information.transpose()
                + inverse_covariance.transpose());
        Eigen::LLT<Eigen::MatrixXd> precision_llt(raw_precision);
        Eigen::MatrixXd precision_lower;
        bool fallback = precision_llt.info() != Eigen::Success;
        if (!fallback) {
            precision_lower = precision_llt.matrixL();
            fallback = !precision_lower.allFinite()
                || (precision_lower.diagonal().array() <= 0.0).any();
        }
        if (fallback) {
            const auto fallback_start = std::chrono::steady_clock::now();
            const Eigen::MatrixXd repaired =
                floor_covariance(raw_precision, 1e-8);
            precision_llt.compute(repaired);
            precision_lower = precision_llt.matrixL();
            out.precision_fallback_seconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - fallback_start).count();
            ++out.precision_fallbacks;
        }
        if (precision_llt.info() != Eigen::Success
            || !precision_lower.allFinite()
            || (precision_lower.diagonal().array() <= 0.0).any()) {
            throw std::runtime_error(
                "UAC Fisher precision is not positive definite");
        }
        const Eigen::VectorXd step = precision_llt.solve(b);
        const double precision_logdet = 2.0
            * precision_lower.diagonal().array().log().sum();
        out.means.push_back(center + step);
        out.precision_lower.push_back(std::move(precision_lower));
        out.log_precision_determinants(j) = precision_logdet;
        log_weight(j) = std::log(pilot.weights(c))
            - 0.5 * (dimension * kLog2Pi + pilot_logdet
                + residual.dot(inverse_covariance * residual))
            + 0.5 * b.dot(step) + 0.5 * dimension * kLog2Pi
            - 0.5 * precision_logdet;
    }
    out.weights = (log_weight.array() - logsumexp(log_weight)).exp();
    return out;
}

Eigen::VectorXd proposal_log_density_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    const DocumentProposal& proposal) {
    Eigen::MatrixXd terms(proposal.weights.size(), values.rows());
    for (Eigen::Index j = 0; j < proposal.weights.size(); ++j) {
        if (!(proposal.weights(j) > 0.0)) {
            terms.row(j).setConstant(
                -std::numeric_limits<double>::infinity());
            continue;
        }
        const Eigen::MatrixXd residual =
            (values.rowwise() - proposal.means[j].transpose()).transpose();
        const Eigen::MatrixXd transformed =
            proposal.precision_lower[j].transpose() * residual;
        const double covariance_logdet = values.cols()
            * std::log(proposal.broadening)
            - proposal.log_precision_determinants(j);
        terms.row(j) = (std::log(proposal.weights(j)) - 0.5
            * (values.cols() * kLog2Pi + covariance_logdet
                + transformed.colwise().squaredNorm().array()
                    / proposal.broadening)).matrix();
    }
    Eigen::VectorXd out(values.rows());
    for (Eigen::Index s = 0; s < values.rows(); ++s) {
        out(s) = logsumexp(terms.col(s));
    }
    return out;
}

struct ProposalScreeningPlan {
    bool enabled = false;
    double planning_seconds = 0.0;
    double predicted_work_ratio = 1.0;
    int32_t active_components = 0;
    std::vector<std::vector<int32_t>> candidates;
    std::vector<int32_t> audit_documents;
    int32_t audit_violations = 0;
    double maximum_audit_omitted_mass = 0.0;
};

void add_screening_metrics(ScoreResult& score,
    const ComponentScreeningOptions& requested,
    const ProposalScreeningPlan& proposal,
    const ComponentScreeningOptions& particle) {
    score.component_screening_options = requested;
    score.proposal_component_screening = proposal.enabled;
    score.particle_component_screening =
        particle.mode == ComponentScreeningMode::On;
    score.proposal_screening_seconds = proposal.planning_seconds;
    score.proposal_audit_documents =
        checked_int32(proposal.audit_documents.size(),
            "proposal audit document count");
    score.proposal_audit_violations = proposal.audit_violations;
    score.proposal_audit_maximum_omitted_mass =
        proposal.maximum_audit_omitted_mass;
}

double document_effective_total(const Dataset& data, int32_t document) {
    if (data.effective_totals.size() == data.coordinates.rows()) {
        return data.effective_totals(document);
    }
    if (document < static_cast<int32_t>(data.counts.size())) {
        return std::accumulate(data.counts[document].cnts.begin(),
            data.counts[document].cnts.end(), 0.0);
    }
    return 0.0;
}

ProposalScreeningPlan make_proposal_screening_plan(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& cache,
    ProposalKind proposal_kind, double broadening, uint64_t seed,
    const ComponentScreeningOptions& options,
    const IndexedDocumentSource* count_source = nullptr) {
    validate_component_screening(options);
    ProposalScreeningPlan out;
    if (options.mode == ComponentScreeningMode::Off) return out;
    const auto planning_start = std::chrono::steady_clock::now();
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(pilot.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    std::vector<int32_t> active;
    for (int32_t c = 0; c < components; ++c) {
        if (pilot.weights(c) > 0.0) active.push_back(c);
    }
    if (active.empty()) {
        throw std::runtime_error(
            "UAC proposal screening has no active pilot component");
    }
    out.active_components = static_cast<int32_t>(active.size());
    out.candidates.resize(documents);
    std::vector<double> proxy_entropy(documents, 0.0);
    std::vector<std::vector<int32_t>> groups(components);
    for (int32_t d = 0; d < documents; ++d) {
        Eigen::VectorXd proxy = Eigen::VectorXd::Constant(
            components, -std::numeric_limits<double>::infinity());
        const Eigen::VectorXd center =
            data.coordinates.row(d).transpose();
        for (const int32_t c : active) {
            const Eigen::VectorXd residual =
                center - pilot.means.row(c).transpose();
            proxy(c) = std::log(pilot.weights(c)) - 0.5
                * (dimension * kLog2Pi + cache.log_determinants(c)
                    + residual.dot(cache.inverse_covariances[c] * residual));
        }
        const double normalizer = logsumexp(proxy);
        std::vector<int32_t> order = active;
        std::stable_sort(order.begin(), order.end(), [&](int32_t left,
                int32_t right) {
            return proxy(left) == proxy(right)
                ? left < right : proxy(left) > proxy(right);
        });
        groups[order.front()].push_back(d);
        double cumulative = 0.0;
        const int32_t minimum = std::min<int32_t>(
            options.minimum_components, active.size());
        for (const int32_t c : order) {
            const double probability = std::exp(proxy(c) - normalizer);
            if (probability > 0.0) {
                proxy_entropy[d] -= probability * std::log(probability);
            }
            const bool below_maximum =
                options.mode != ComponentScreeningMode::On
                || options.maximum_components == 0
                || static_cast<int32_t>(out.candidates[d].size())
                    < options.maximum_components;
            if (below_maximum
                && (static_cast<int32_t>(out.candidates[d].size()) < minimum
                    || cumulative
                        < 1.0 - options.proposal_proxy_tail_mass)) {
                out.candidates[d].push_back(c);
            }
            cumulative += probability;
        }
    }

    const int32_t requested = options.audit_documents > 0
        ? options.audit_documents
        : std::min(256, std::max(16, 2 * out.active_components));
    const int32_t budget = std::min(documents, requested);
    std::vector<int32_t> represented;
    for (const int32_t c : active) {
        if (!groups[c].empty()) represented.push_back(c);
    }
    if (budget < static_cast<int32_t>(represented.size())) {
        std::stable_sort(represented.begin(), represented.end(),
            [&](int32_t left, int32_t right) {
                return hash_string(seed, std::to_string(left))
                    < hash_string(seed, std::to_string(right));
            });
        represented.resize(budget);
    }
    std::vector<uint8_t> selected(documents, 0);
    auto add_document = [&](int32_t document) {
        if (static_cast<int32_t>(out.audit_documents.size()) >= budget
            || selected[document]) {
            return;
        }
        selected[document] = 1;
        out.audit_documents.push_back(document);
    };
    for (const int32_t c : represented) {
        const auto found = std::max_element(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return proxy_entropy[left] < proxy_entropy[right];
            });
        if (found != groups[c].end()) add_document(*found);
    }
    for (const int32_t c : represented) {
        const auto found = std::min_element(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return document_effective_total(data, left)
                    < document_effective_total(data, right);
            });
        if (found != groups[c].end()) add_document(*found);
    }
    std::vector<size_t> cursor(components, 0);
    for (const int32_t c : represented) {
        std::stable_sort(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return hash_string(seed, data.identifiers[left])
                    < hash_string(seed, data.identifiers[right]);
            });
    }
    while (static_cast<int32_t>(out.audit_documents.size()) < budget) {
        bool added = false;
        for (const int32_t c : represented) {
            while (cursor[c] < groups[c].size()
                && selected[groups[c][cursor[c]]]) {
                ++cursor[c];
            }
            if (cursor[c] < groups[c].size()) {
                add_document(groups[c][cursor[c]++]);
                added = true;
                if (static_cast<int32_t>(out.audit_documents.size())
                    >= budget) {
                    break;
                }
            }
        }
        if (!added) break;
    }

    for (const int32_t d : out.audit_documents) {
        DocumentBlock count_block;
        if (count_source) {
            count_source->read_range(d, 1, count_block);
        }
        const Eigen::VectorXd center =
            data.coordinates.row(d).transpose();
        const FisherApproximation fisher = fisher_approximation_impl(
            center,
            count_source ? count_block.counts.front() : data.counts[d],
            basis, helmert, proposal_kind);
        const DocumentProposal full = fisher_proposal(
            center, fisher, pilot, cache, broadening);
        std::vector<uint8_t> retained(components, 0);
        for (const int32_t c : out.candidates[d]) retained[c] = 1;
        double omitted = 0.0;
        for (Eigen::Index j = 0; j < full.weights.size(); ++j) {
            if (!retained[full.component_ids[j]]) {
                omitted += full.weights(j);
            }
        }
        out.maximum_audit_omitted_mass = std::max(
            out.maximum_audit_omitted_mass, omitted);
        if (omitted > options.proposal_proxy_tail_mass * (1.0 + 1e-8)) {
            ++out.audit_violations;
        }
    }
    double mean_candidates = 0.0;
    for (const auto& candidates : out.candidates) {
        mean_candidates += candidates.size();
    }
    mean_candidates /= std::max(1, documents);
    out.predicted_work_ratio = 1.0 / std::max(1, dimension)
        + mean_candidates / out.active_components;
    const int32_t audit_count =
        static_cast<int32_t>(out.audit_documents.size());
    const int32_t audit_successes =
        audit_count - out.audit_violations;
    // Strictly more than 90%; exact 90% does not pass.
    const bool audit_passed = audit_count > 0
        && static_cast<int64_t>(audit_successes) * 10
            > static_cast<int64_t>(audit_count) * 9;
    out.enabled = options.mode == ComponentScreeningMode::On
        || (audit_passed
            && out.predicted_work_ratio
                <= 1.0 - options.minimum_work_reduction);
    out.planning_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - planning_start).count();
    return out;
}

struct Expectation {
    int32_t documents = 0;
    RowMajorMatrixXd responsibilities;
    Eigen::VectorXd membership;
    RowMajorMatrixXd first;
    std::vector<Eigen::MatrixXd> second;
    RowMajorMatrixXd sum_y2;
    RowMajorMatrixXd sum_f;
    std::vector<Eigen::MatrixXd> sum_ff;
    std::vector<Eigen::MatrixXd> sum_yf;
    std::vector<ParticleDiagnostic> particle_diagnostics;
    double log_likelihood = 0.0;
    double log_likelihood_upper = 0.0;
    double responsibility_entropy_sum = 0.0;
    double gaussian_seconds = 0.0;
    double component_bound_seconds = 0.0;
    double moment_seconds = 0.0;
    uint64_t peak_workspace_bytes = 0;
    uint64_t peak_particle_bytes = 0;
    int32_t parallel_workers = 0;
    int64_t evaluated_component_documents = 0;
    int64_t possible_component_documents = 0;
    int32_t full_component_documents = 0;
    int32_t component_bound_violations = 0;
    double omitted_component_mass_sum = 0.0;
    double maximum_omitted_component_mass = 0.0;
    double mean_max_responsibility_change =
        std::numeric_limits<double>::quiet_NaN();
    bool has_responsibility_change = false;
    std::vector<int32_t> per_document_evaluated_components;
    std::vector<double> per_document_omitted_component_mass;
};

struct ExpectationRequest {
    bool store_responsibilities = false;
    bool collect_diagnostics = false;
    bool accumulate_moments = true;
};

struct ExpectationBlock {
    Eigen::VectorXd membership;
    RowMajorMatrixXd first;
    std::vector<Eigen::MatrixXd> second;
    RowMajorMatrixXd sum_y2;
    RowMajorMatrixXd sum_f;
    std::vector<Eigen::MatrixXd> sum_ff;
    std::vector<Eigen::MatrixXd> sum_yf;
    double log_likelihood = 0.0;
    double log_likelihood_upper = 0.0;
    double responsibility_entropy_sum = 0.0;
    double component_bound_seconds = 0.0;
    int64_t evaluated_component_documents = 0;
    int64_t possible_component_documents = 0;
    int32_t full_component_documents = 0;
    int32_t component_bound_violations = 0;
    double omitted_component_mass_sum = 0.0;
    double maximum_omitted_component_mass = 0.0;

    ExpectationBlock(int32_t components, int32_t dimension,
        int32_t factor_rank = -1, bool accumulate_moments = true) {
        if (!accumulate_moments) return;
        membership = Eigen::VectorXd::Zero(components);
        first = RowMajorMatrixXd::Zero(components, dimension);
        if (factor_rank < 0) {
            second.assign(components,
                Eigen::MatrixXd::Zero(dimension, dimension));
        } else {
            sum_y2 = RowMajorMatrixXd::Zero(components, dimension);
            sum_f = RowMajorMatrixXd::Zero(components, factor_rank);
            sum_ff.assign(components,
                Eigen::MatrixXd::Zero(factor_rank, factor_rank));
            sum_yf.assign(components,
                Eigen::MatrixXd::Zero(dimension, factor_rank));
        }
    }
};

int32_t expectation_shards(int32_t documents, int32_t components,
    int32_t dimension, int32_t factor_rank) {
    constexpr uint64_t memory_budget = 64ull * 1024ull * 1024ull;
    constexpr int32_t maximum_shards = 32;
    uint64_t bytes = sizeof(double) * static_cast<uint64_t>(components)
        * (1 + dimension);
    if (factor_rank < 0) {
        bytes += sizeof(double) * static_cast<uint64_t>(components)
            * dimension * dimension;
    } else {
        bytes += sizeof(double) * static_cast<uint64_t>(components)
            * (dimension + factor_rank + factor_rank * factor_rank
                + dimension * factor_rank);
    }
    const int32_t memory_shards = static_cast<int32_t>(std::max<uint64_t>(
        1, memory_budget / std::max<uint64_t>(1, bytes)));
    return std::max(1, std::min({documents, maximum_shards, memory_shards}));
}

uint64_t expectation_block_bytes(int32_t components, int32_t dimension,
    int32_t factor_rank) {
    uint64_t values = static_cast<uint64_t>(components) * (1 + dimension);
    if (factor_rank < 0) {
        values += static_cast<uint64_t>(components) * dimension * dimension;
    } else {
        values += static_cast<uint64_t>(components)
            * (dimension + factor_rank + factor_rank * factor_rank
                + dimension * factor_rank);
    }
    return sizeof(double) * values;
}

void reduce_expectation_blocks(Expectation& out,
    const std::vector<ExpectationBlock>& blocks) {
    for (const auto& block : blocks) {
        if (block.membership.size() > 0) {
            out.membership += block.membership;
            out.first += block.first;
        }
        out.log_likelihood += block.log_likelihood;
        out.log_likelihood_upper += block.log_likelihood_upper;
        out.responsibility_entropy_sum +=
            block.responsibility_entropy_sum;
        out.component_bound_seconds += block.component_bound_seconds;
        out.evaluated_component_documents +=
            block.evaluated_component_documents;
        out.possible_component_documents +=
            block.possible_component_documents;
        out.full_component_documents += block.full_component_documents;
        out.component_bound_violations +=
            block.component_bound_violations;
        out.omitted_component_mass_sum +=
            block.omitted_component_mass_sum;
        out.maximum_omitted_component_mass = std::max(
            out.maximum_omitted_component_mass,
            block.maximum_omitted_component_mass);
        if (block.membership.size() == 0) {
            continue;
        } else if (out.sum_y2.size() > 0) {
            out.sum_y2 += block.sum_y2;
            out.sum_f += block.sum_f;
            for (size_t c = 0; c < out.sum_ff.size(); ++c) {
                out.sum_ff[c] += block.sum_ff[c];
                out.sum_yf[c] += block.sum_yf[c];
            }
        } else {
            for (size_t c = 0; c < out.second.size(); ++c) {
                out.second[c] += block.second[c];
            }
        }
    }
}

Expectation empty_expectation(int32_t documents, int32_t components,
    int32_t dimension, int32_t factor_rank = -1,
    bool accumulate_moments = true) {
    Expectation out;
    out.documents = documents;
    if (!accumulate_moments) return out;
    out.membership = Eigen::VectorXd::Zero(components);
    out.first = RowMajorMatrixXd::Zero(components, dimension);
    out.second.resize(components);
    if (factor_rank < 0) {
        for (auto& value : out.second) {
            value = Eigen::MatrixXd::Zero(dimension, dimension);
        }
    } else {
        out.sum_y2 = RowMajorMatrixXd::Zero(components, dimension);
        out.sum_f = RowMajorMatrixXd::Zero(components, factor_rank);
        out.sum_ff.assign(components,
            Eigen::MatrixXd::Zero(factor_rank, factor_rank));
        out.sum_yf.assign(components,
            Eigen::MatrixXd::Zero(dimension, factor_rank));
    }
    return out;
}

void accumulate_expectation(Expectation& target, const Expectation& source) {
    target.membership += source.membership;
    target.first += source.first;
    target.log_likelihood += source.log_likelihood;
    target.log_likelihood_upper += source.log_likelihood_upper;
    target.responsibility_entropy_sum +=
        source.responsibility_entropy_sum;
    target.component_bound_seconds += source.component_bound_seconds;
    target.evaluated_component_documents +=
        source.evaluated_component_documents;
    target.possible_component_documents +=
        source.possible_component_documents;
    target.full_component_documents += source.full_component_documents;
    target.component_bound_violations +=
        source.component_bound_violations;
    target.omitted_component_mass_sum +=
        source.omitted_component_mass_sum;
    target.maximum_omitted_component_mass = std::max(
        target.maximum_omitted_component_mass,
        source.maximum_omitted_component_mass);
    if (target.sum_y2.size() > 0) {
        target.sum_y2 += source.sum_y2;
        target.sum_f += source.sum_f;
        for (size_t c = 0; c < target.sum_ff.size(); ++c) {
            target.sum_ff[c] += source.sum_ff[c];
            target.sum_yf[c] += source.sum_yf[c];
        }
    } else {
        for (size_t c = 0; c < target.second.size(); ++c) {
            target.second[c] += source.second[c];
        }
    }
}

Expectation map_expectation(const Dataset& data, const Model& model,
    const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {}) {
    validate_component_screening(screening);
    const bool screen =
        screening.mode != ComponentScreeningMode::Off;
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    const int32_t factor_rank = model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(model.factor_covariances.front().factor.cols())
        : -1;
    Expectation out = empty_expectation(documents, components, dimension,
        factor_rank, request.accumulate_moments);
    if (request.store_responsibilities) {
        out.responsibilities.resize(documents, components);
        out.per_document_evaluated_components.resize(documents);
        out.per_document_omitted_component_mass.resize(documents);
    }
    std::vector<LowRankDiagonalSolver> factor_solvers;
    std::vector<Eigen::MatrixXd> factor_beta, factor_conditional;
    if (factor_rank >= 0) {
        factor_solvers.reserve(components);
        for (int32_t c = 0; c < components; ++c) {
            const auto& covariance = model.factor_covariances[c];
            factor_solvers.emplace_back(
                covariance.diagonal, covariance.factor);
            if (request.accumulate_moments) {
                factor_beta.push_back(factor_solvers.back().solve_matrix(
                    covariance.factor).transpose());
                factor_conditional.push_back(
                    Eigen::MatrixXd::Identity(factor_rank, factor_rank)
                    - factor_beta.back() * covariance.factor);
            }
        }
    }
    const std::vector<DenseGaussianSolver> dense_solvers =
        dense_model_solvers(model);
    const std::vector<double> eigenvalue_upper = screen
        ? model_eigenvalue_upper_bounds(model) : std::vector<double>{};
    Eigen::VectorXd gaussian_constant = Eigen::VectorXd::Constant(
        components, -std::numeric_limits<double>::infinity());
    for (int32_t c = 0; c < components; ++c) {
        if (!(model.weights(c) > 0.0)) continue;
        const double logdet = factor_rank < 0
            ? dense_solvers[c].log_determinant
            : factor_solvers[c].log_determinant();
        gaussian_constant(c) = std::log(model.weights(c))
            - 0.5 * (dimension * kLog2Pi + logdet);
    }
    const int32_t possible_components = active_component_count(model);
    const int32_t requested_blocks = expectation_shards(
        documents, components, dimension, factor_rank);
    const int32_t block_size =
        (documents + requested_blocks - 1) / requested_blocks;
    const int32_t n_blocks = (documents + block_size - 1) / block_size;
    std::vector<ExpectationBlock> blocks;
    blocks.reserve(n_blocks);
    for (int32_t block = 0; block < n_blocks; ++block) {
        blocks.emplace_back(components, dimension, factor_rank,
            request.accumulate_moments);
    }
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        ExpectationBlock& block = blocks[block_index];
        Eigen::VectorXd responsibility(components);
        Eigen::VectorXd upper(components);
        ComponentScreeningWorkspace screening_workspace;
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        for (int32_t d = begin; d < end; ++d) {
            const Eigen::VectorXd value = data.coordinates.row(d).transpose();
            auto exact_score = [&](int32_t c) {
                    if (factor_rank < 0) {
                        return std::log(model.weights(c))
                            + dense_solvers[c].log_density(value);
                    }
                    const Eigen::VectorXd residual = value
                        - model.means.row(c).transpose();
                    return std::log(model.weights(c)) - 0.5
                        * (dimension * kLog2Pi
                            + factor_solvers[c].log_determinant()
                            + factor_solvers[c].quadratic(residual));
                };
            ScreenedComponents selected;
            if (screen) {
                const auto bound_start = std::chrono::steady_clock::now();
                upper.setConstant(-std::numeric_limits<double>::infinity());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    upper(c) = gaussian_constant(c) - 0.5
                        * (value - model.means.row(c).transpose()).squaredNorm()
                        / eigenvalue_upper[c];
                }
                block.component_bound_seconds +=
                    std::chrono::duration<double>(
                        std::chrono::steady_clock::now()
                        - bound_start).count();
                selected = screen_component_scores(
                    upper, screening, true, exact_score,
                    &screening_workspace);
            } else {
                selected.score = Eigen::VectorXd::Constant(
                    components, -std::numeric_limits<double>::infinity());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    selected.score(c) = exact_score(c);
                    selected.evaluated.push_back(c);
                }
                selected.log_mass = logsumexp(selected.score);
                selected.full = true;
            }
            const double normalizer = selected.log_mass;
            const double upper_normalizer = logaddexp(
                selected.log_mass, selected.log_upper_mass);
            block.log_likelihood += normalizer;
            block.log_likelihood_upper += upper_normalizer;
            block.evaluated_component_documents += selected.evaluated.size();
            block.possible_component_documents += possible_components;
            block.full_component_documents += selected.full ? 1 : 0;
            block.component_bound_violations +=
                selected.bound_violation ? 1 : 0;
            block.omitted_component_mass_sum +=
                selected.omitted_mass_bound;
            block.maximum_omitted_component_mass = std::max(
                block.maximum_omitted_component_mass,
                selected.omitted_mass_bound);
            responsibility = (selected.score.array() - normalizer).exp();
            if (request.store_responsibilities) {
                out.responsibilities.row(d) = responsibility.transpose();
                out.per_document_evaluated_components[d] =
                    static_cast<int32_t>(selected.evaluated.size());
                out.per_document_omitted_component_mass[d] =
                    selected.omitted_mass_bound;
            }
            if (!request.accumulate_moments) continue;
            for (const int32_t c : selected.evaluated) {
                const double weight = responsibility(c);
                if (!(weight > 0.0)) continue;
                block.membership(c) += weight;
                block.first.row(c) += weight * value.transpose();
                if (factor_rank < 0) {
                    block.second[c].noalias() += weight
                        * value * value.transpose();
                } else {
                    const Eigen::VectorXd factor = factor_beta[c]
                        * (value - model.means.row(c).transpose());
                    block.sum_y2.row(c).array() += weight
                        * value.array().square().transpose();
                    block.sum_f.row(c) += weight * factor.transpose();
                    block.sum_ff[c].noalias() += weight
                        * (factor_conditional[c]
                            + factor * factor.transpose());
                    block.sum_yf[c].noalias() += weight
                        * value * factor.transpose();
                }
            }
        }
    });
    reduce_expectation_blocks(out, blocks);
    return out;
}

bool resolve_map_component_screening(const Dataset& data,
    const Model& model, const ComponentScreeningOptions& requested,
    uint64_t seed) {
    if (requested.mode != ComponentScreeningMode::Auto) {
        return requested.mode == ComponentScreeningMode::On;
    }
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    const int32_t active = active_component_count(model);
    if (documents == 0 || active <= requested.minimum_components) {
        return false;
    }
    const int32_t factor_rank = model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(model.factor_covariances.front().factor.cols())
        : -1;
    const std::vector<DenseGaussianSolver> dense_solvers =
        dense_model_solvers(model);
    std::vector<LowRankDiagonalSolver> factor_solvers;
    if (factor_rank >= 0) {
        factor_solvers.reserve(components);
        for (int32_t c = 0; c < components; ++c) {
            const auto& covariance = model.factor_covariances[c];
            factor_solvers.emplace_back(
                covariance.diagonal, covariance.factor);
        }
    }
    const std::vector<double> eigenvalue_upper =
        model_eigenvalue_upper_bounds(model);
    Eigen::VectorXd constant = Eigen::VectorXd::Constant(
        components, -std::numeric_limits<double>::infinity());
    for (int32_t c = 0; c < components; ++c) {
        if (!(model.weights(c) > 0.0)) continue;
        const double logdet = factor_rank < 0
            ? dense_solvers[c].log_determinant
            : factor_solvers[c].log_determinant();
        constant(c) = std::log(model.weights(c))
            - 0.5 * (dimension * kLog2Pi + logdet);
    }
    std::vector<std::vector<int32_t>> groups(components);
    std::vector<double> entropy(documents, 0.0);
    for (int32_t d = 0; d < documents; ++d) {
        Eigen::VectorXd upper = Eigen::VectorXd::Constant(
            components, -std::numeric_limits<double>::infinity());
        const Eigen::VectorXd value =
            data.coordinates.row(d).transpose();
        int32_t top = -1;
        for (int32_t c = 0; c < components; ++c) {
            if (!(model.weights(c) > 0.0)) continue;
            upper(c) = constant(c) - 0.5
                * (value - model.means.row(c).transpose()).squaredNorm()
                    / eigenvalue_upper[c];
            if (top < 0 || upper(c) > upper(top)) top = c;
        }
        const double normalizer = logsumexp(upper);
        for (int32_t c = 0; c < components; ++c) {
            if (!std::isfinite(upper(c))) continue;
            const double probability = std::exp(upper(c) - normalizer);
            if (probability > 0.0) {
                entropy[d] -= probability * std::log(probability);
            }
        }
        groups[top].push_back(d);
    }
    const int32_t requested_budget = requested.audit_documents > 0
        ? requested.audit_documents
        : std::min(256, std::max(16, 2 * active));
    const int32_t budget = std::min(documents, requested_budget);
    std::vector<int32_t> audit;
    std::vector<uint8_t> chosen(documents, 0);
    auto add = [&](int32_t d) {
        if (static_cast<int32_t>(audit.size()) < budget && !chosen[d]) {
            chosen[d] = 1;
            audit.push_back(d);
        }
    };
    std::vector<int32_t> represented;
    for (int32_t c = 0; c < components; ++c) {
        if (!groups[c].empty()) represented.push_back(c);
    }
    std::stable_sort(represented.begin(), represented.end(),
        [&](int32_t left, int32_t right) {
            return hash_string(seed, std::to_string(left))
                < hash_string(seed, std::to_string(right));
        });
    if (static_cast<int32_t>(represented.size()) > budget) {
        represented.resize(budget);
    }
    for (const int32_t c : represented) {
        add(*std::max_element(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return entropy[left] < entropy[right];
            }));
    }
    for (const int32_t c : represented) {
        add(*std::min_element(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return document_effective_total(data, left)
                    < document_effective_total(data, right);
            }));
    }
    std::vector<int32_t> remaining(documents);
    std::iota(remaining.begin(), remaining.end(), int32_t{0});
    std::stable_sort(remaining.begin(), remaining.end(),
        [&](int32_t left, int32_t right) {
            return hash_string(seed, data.identifiers[left])
                < hash_string(seed, data.identifiers[right]);
        });
    for (const int32_t d : remaining) add(d);

    ComponentScreeningOptions enabled = requested;
    enabled.mode = ComponentScreeningMode::On;
    enabled.maximum_components = 0;
    int64_t evaluated = 0;
    int32_t violations = 0;
    Eigen::VectorXd upper(components);
    ComponentScreeningWorkspace screening_workspace;
    for (const int32_t d : audit) {
        const Eigen::VectorXd value =
            data.coordinates.row(d).transpose();
        upper.setConstant(-std::numeric_limits<double>::infinity());
        for (int32_t c = 0; c < components; ++c) {
            if (model.weights(c) > 0.0) {
                upper(c) = constant(c) - 0.5
                    * (value - model.means.row(c).transpose()).squaredNorm()
                        / eigenvalue_upper[c];
            }
        }
        auto exact = [&](int32_t c) {
            if (factor_rank < 0) {
                return std::log(model.weights(c))
                    + dense_solvers[c].log_density(value);
            }
            const Eigen::VectorXd residual =
                value - model.means.row(c).transpose();
            return constant(c) - 0.5
                * factor_solvers[c].quadratic(residual);
        };
        const ScreenedComponents selected = screen_component_scores(
            upper, enabled, true, exact, &screening_workspace);
        evaluated += selected.evaluated.size();
        violations += selected.bound_violation ? 1 : 0;
    }
    const double mean_evaluated = static_cast<double>(evaluated)
        / std::max<size_t>(1, audit.size());
    const double exact_cost = factor_rank < 0
        ? static_cast<double>(dimension) * dimension
        : static_cast<double>(dimension) * factor_rank
            + factor_rank * factor_rank + dimension;
    const double ratio = (active * static_cast<double>(dimension)
            + mean_evaluated * exact_cost)
        / (active * exact_cost);
    return violations == 0
        && ratio <= 1.0 - requested.minimum_work_reduction;
}

template<class ParticleCollection>
Expectation particle_expectation_impl(const ParticleCollection& particles,
    const Model& model, const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {},
    int32_t forced_blocks = 0,
    ExpectationBlock* external_block = nullptr) {
    const bool accumulate_moments = request.accumulate_moments;
    validate_component_screening(screening);
    const bool screen =
        screening.mode != ComponentScreeningMode::Off;
    const int32_t documents = particles.documents;
    const int32_t maximum_samples = [&]() {
        if constexpr (std::is_same_v<ParticleCollection, ParticleSet>) {
            return particles.samples;
        } else {
            return particles.maximum_samples;
        }
    }();
    const int32_t dimension = particles.dimension;
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t factor_rank = model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(model.factor_covariances.front().factor.cols())
        : -1;
    Expectation out;
    if (accumulate_moments) {
        out = empty_expectation(
            documents, components, dimension, factor_rank);
    } else {
        out.documents = documents;
    }
    if (request.store_responsibilities) {
        out.responsibilities.resize(documents, components);
    }
    if (request.store_responsibilities || request.collect_diagnostics) {
        out.per_document_evaluated_components.resize(documents);
        out.per_document_omitted_component_mass.resize(documents);
    }
    if (request.collect_diagnostics) {
        out.particle_diagnostics.resize(documents);
    }
    std::vector<LowRankDiagonalSolver> factor_solvers;
    std::vector<Eigen::MatrixXd> factor_beta, factor_conditional;
    if (factor_rank >= 0) {
        factor_solvers.reserve(components);
        for (int32_t c = 0; c < components; ++c) {
            const auto& covariance = model.factor_covariances[c];
            factor_solvers.emplace_back(
                covariance.diagonal, covariance.factor);
            if (accumulate_moments) {
                factor_beta.push_back(factor_solvers.back().solve_matrix(
                    covariance.factor).transpose());
                factor_conditional.push_back(
                    Eigen::MatrixXd::Identity(factor_rank, factor_rank)
                    - factor_beta.back() * covariance.factor);
            }
        }
    }
    const std::vector<DenseGaussianSolver> dense_solvers =
        dense_model_solvers(model);
    const std::vector<double> eigenvalue_upper = screen
        ? model_eigenvalue_upper_bounds(model) : std::vector<double>{};
    Eigen::VectorXd gaussian_constant = Eigen::VectorXd::Constant(
        components, -std::numeric_limits<double>::infinity());
    for (int32_t c = 0; c < components; ++c) {
        if (!(model.weights(c) > 0.0)) continue;
        const double logdet = factor_rank < 0
            ? dense_solvers[c].log_determinant
            : factor_solvers[c].log_determinant();
        gaussian_constant(c) = std::log(model.weights(c))
            - 0.5 * (dimension * kLog2Pi + logdet);
    }
    const int32_t possible_components = active_component_count(model);
    const int32_t requested_blocks = external_block
        ? 1 : forced_blocks > 0
        ? std::min(documents, forced_blocks)
        : expectation_shards(
            documents, components, dimension, factor_rank);
    const int32_t block_size =
        (documents + requested_blocks - 1) / requested_blocks;
    const int32_t n_blocks = (documents + block_size - 1) / block_size;
    uint64_t block_workspace_values =
        static_cast<uint64_t>(components) * maximum_samples
        + 2 * static_cast<uint64_t>(components);
    if (screen) {
        block_workspace_values +=
            static_cast<uint64_t>(maximum_samples) * dimension
            + maximum_samples + components;
    }
    out.peak_workspace_bytes = static_cast<uint64_t>(n_blocks)
        * (sizeof(double) * block_workspace_values
            + (accumulate_moments && !external_block
                ? expectation_block_bytes(
                    components, dimension, factor_rank)
                : 0));
    std::vector<ExpectationBlock> blocks;
    if (!external_block) {
        blocks.reserve(n_blocks);
        for (int32_t block = 0; block < n_blocks; ++block) {
            blocks.emplace_back(
                components, dimension, factor_rank, accumulate_moments);
        }
    }
    std::atomic<int64_t> gaussian_nanoseconds{0};
    std::atomic<int64_t> moment_nanoseconds{0};
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        ExpectationBlock& block = external_block
            ? *external_block : blocks[block_index];
        Eigen::MatrixXd log_tilt(components, maximum_samples);
        Eigen::VectorXd evidence(components);
        Eigen::VectorXd responsibility(components);
        RowMajorMatrixXd bound_residual;
        Eigen::VectorXd bound_term;
        Eigen::VectorXd upper;
        if (screen) {
            bound_residual.resize(maximum_samples, dimension);
            bound_term.resize(maximum_samples);
            upper.resize(components);
        }
        ComponentScreeningWorkspace screening_workspace;
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        int64_t local_gaussian_nanoseconds = 0;
        int64_t local_moment_nanoseconds = 0;
        double local_bound_seconds = 0.0;
        for (int32_t d = begin; d < end; ++d) {
            const int32_t samples = particles.samples_for_document(d);
            const auto gaussian_start = std::chrono::steady_clock::now();
            const auto values = particles.values_for_document(d);
            const Eigen::VectorXd base =
                particles.log_likelihood_for_document(d)
                - particles.log_proposal_for_document(d)
                - Eigen::VectorXd::Constant(samples, std::log(samples));
            evidence.setConstant(
                -std::numeric_limits<double>::infinity());
            auto exact_score = [&](int32_t c) {
                if (factor_rank < 0) {
                    log_tilt.row(c).head(samples) = (base
                        + dense_solvers[c].log_density_rows(values)).transpose();
                } else {
                    const RowMajorMatrixXd residual =
                        values.rowwise() - model.means.row(c);
                    log_tilt.row(c).head(samples) = (base.array()
                        - 0.5 * (dimension * kLog2Pi
                            + factor_solvers[c].log_determinant()
                            + factor_solvers[c].quadratic_rows(
                                residual).array())).matrix().transpose();
                }
                evidence(c) = logsumexp(
                    log_tilt.row(c).head(samples).transpose());
                return std::log(model.weights(c)) + evidence(c);
            };
            ScreenedComponents selected;
            if (screen) {
                const auto bound_start = std::chrono::steady_clock::now();
                upper.setConstant(-std::numeric_limits<double>::infinity());
                // Round the squared-distance contribution downward so the
                // vectorized floating-point calculation remains conservative.
                const double squared_distance_roundoff = std::max(0.5,
                    1.0 - 16.0 * (dimension + 1)
                        * std::numeric_limits<double>::epsilon());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    auto residual = bound_residual.topRows(samples);
                    residual = values.rowwise() - model.means.row(c);
                    bound_term.head(samples).array() = base.array()
                        + gaussian_constant(c)
                        - 0.5 * squared_distance_roundoff
                            * residual.rowwise().squaredNorm().array()
                            / eigenvalue_upper[c];
                    upper(c) = logsumexp(bound_term.head(samples));
                }
                local_bound_seconds += std::chrono::duration<double>(
                    std::chrono::steady_clock::now()
                    - bound_start).count();
                selected = screen_component_scores(
                    upper, screening, true, exact_score,
                    &screening_workspace);
            } else {
                selected.score = Eigen::VectorXd::Constant(
                    components, -std::numeric_limits<double>::infinity());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    selected.score(c) = exact_score(c);
                    selected.evaluated.push_back(c);
                }
                selected.log_mass = logsumexp(selected.score);
                selected.full = true;
            }
            local_gaussian_nanoseconds +=
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - gaussian_start).count();
            const double normalizer = selected.log_mass;
            block.log_likelihood += normalizer;
            block.log_likelihood_upper += logaddexp(
                selected.log_mass, selected.log_upper_mass);
            block.evaluated_component_documents += selected.evaluated.size();
            block.possible_component_documents += possible_components;
            block.full_component_documents += selected.full ? 1 : 0;
            block.component_bound_violations +=
                selected.bound_violation ? 1 : 0;
            block.omitted_component_mass_sum +=
                selected.omitted_mass_bound;
            block.maximum_omitted_component_mass = std::max(
                block.maximum_omitted_component_mass,
                selected.omitted_mass_bound);
            if (accumulate_moments || request.store_responsibilities
                || request.collect_diagnostics) {
                responsibility =
                    (selected.score.array() - normalizer).exp();
            }
            if (request.store_responsibilities) {
                out.responsibilities.row(d) = responsibility.transpose();
            }
            if (request.store_responsibilities
                || request.collect_diagnostics) {
                out.per_document_evaluated_components[d] =
                    static_cast<int32_t>(selected.evaluated.size());
                out.per_document_omitted_component_mass[d] =
                    selected.omitted_mass_bound;
            }
            if (request.collect_diagnostics) {
                Eigen::VectorXd log_weight(samples);
                Eigen::VectorXd component_weight = Eigen::VectorXd::Constant(
                    components, -std::numeric_limits<double>::infinity());
                for (int32_t s = 0; s < samples; ++s) {
                    for (const int32_t c : selected.evaluated) {
                        component_weight(c) =
                            std::log(model.weights(c)) + log_tilt(c, s);
                    }
                    log_weight(s) = logsumexp(component_weight);
                }
                const double weight_normalizer = logsumexp(log_weight);
                const Eigen::VectorXd probability =
                    (log_weight.array() - weight_normalizer).exp();
                auto& diagnostic = out.particle_diagnostics[d];
                diagnostic.relative_ess = 1.0
                    / (samples * probability.squaredNorm());
                diagnostic.maximum_weight = probability.maxCoeff();
                const auto document_likelihood =
                    particles.log_likelihood_for_document(d);
                const auto document_proposal =
                    particles.log_proposal_for_document(d);
                diagnostic.log_likelihood_range =
                    document_likelihood.maxCoeff()
                    - document_likelihood.minCoeff();
                diagnostic.log_proposal_range = document_proposal.maxCoeff()
                    - document_proposal.minCoeff();
                const Eigen::VectorXd log_target = log_weight
                    + document_proposal
                    + Eigen::VectorXd::Constant(samples, std::log(samples));
                diagnostic.hpd80_log_density_threshold =
                    weighted_hpd_threshold(log_target, probability, 0.8);
                diagnostic.hpd95_log_density_threshold =
                    weighted_hpd_threshold(log_target, probability, 0.95);
            }
            if (accumulate_moments) {
                const auto moment_start = std::chrono::steady_clock::now();
                for (const int32_t c : selected.evaluated) {
                    const double component_responsibility = responsibility(c);
                    if (!(component_responsibility > 0.0)) continue;
                    block.membership(c) += component_responsibility;
                    const Eigen::VectorXd tau =
                        (log_tilt.row(c).head(samples).transpose().array()
                            - evidence(c)).exp();
                    if (factor_rank < 0) {
                        block.first.row(c).noalias() +=
                            component_responsibility
                            * (values.transpose() * tau).transpose();
                        RowMajorMatrixXd weighted = values;
                        weighted.array().colwise() *= tau.array().sqrt();
                        block.second[c].noalias() += component_responsibility
                            * weighted.transpose() * weighted;
                    } else {
                        const Eigen::VectorXd weight =
                            component_responsibility * tau;
                        const RowMajorMatrixXd residual =
                            values.rowwise() - model.means.row(c);
                        const RowMajorMatrixXd factors =
                            residual * factor_beta[c].transpose();
                        block.first.row(c).noalias() +=
                            weight.transpose() * values;
                        block.sum_y2.row(c).array() +=
                            (weight.transpose()
                                * values.array().square().matrix()).array();
                        block.sum_f.row(c).noalias() +=
                            weight.transpose() * factors;
                        RowMajorMatrixXd weighted_factors = factors;
                        weighted_factors.array().colwise() *=
                            weight.array().sqrt();
                        block.sum_ff[c].noalias() += weight.sum()
                            * factor_conditional[c]
                            + weighted_factors.transpose() * weighted_factors;
                        RowMajorMatrixXd weighted_values = values;
                        weighted_values.array().colwise() *= weight.array();
                        block.sum_yf[c].noalias() +=
                            weighted_values.transpose() * factors;
                    }
                }
                local_moment_nanoseconds +=
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now()
                        - moment_start).count();
            }
        }
        gaussian_nanoseconds.fetch_add(
            local_gaussian_nanoseconds, std::memory_order_relaxed);
        moment_nanoseconds.fetch_add(
            local_moment_nanoseconds, std::memory_order_relaxed);
        block.component_bound_seconds += local_bound_seconds;
    });
    if (!external_block) {
        reduce_expectation_blocks(out, blocks);
    }
    out.gaussian_seconds = 1e-9 * gaussian_nanoseconds.load();
    out.moment_seconds = 1e-9 * moment_nanoseconds.load();
    return out;
}

Expectation particle_expectation(const ParticleSet& particles,
    const Model& model, const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {}) {
    return particle_expectation_impl(
        particles, model, request, screening);
}

Expectation particle_expectation(const RaggedParticleSet& particles,
    const Model& model, const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {}) {
    return particle_expectation_impl(
        particles, model, request, screening);
}

template<class ParticleCollection>
bool resolve_particle_component_screening(
    const ParticleCollection& particles, const Model& model,
    const ComponentScreeningOptions& requested,
    const std::vector<int32_t>& audit_documents) {
    if (requested.mode != ComponentScreeningMode::Auto) {
        return requested.mode == ComponentScreeningMode::On;
    }
    ComponentScreeningOptions enabled = requested;
    enabled.mode = ComponentScreeningMode::On;
    enabled.maximum_components = 0;
    RaggedParticleSet audit;
    audit.documents = static_cast<int32_t>(audit_documents.size());
    audit.dimension = particles.dimension;
    audit.offsets.assign(audit.documents + 1, 0);
    audit.proposal_candidates.resize(audit.documents);
    for (int32_t local = 0; local < audit.documents; ++local) {
        const int32_t document = audit_documents[local];
        if (document < 0 || document >= particles.documents) {
            throw std::runtime_error(
                "UAC particle screening audit document is out of range");
        }
        const int32_t samples = particles.samples_for_document(document);
        audit.offsets[local + 1] = audit.offsets[local] + samples;
        audit.maximum_samples = std::max(audit.maximum_samples, samples);
        const auto values = particles.values_for_document(document);
        audit.values.insert(audit.values.end(), values.data(),
            values.data() + static_cast<int64_t>(samples)
                * particles.dimension);
        const auto likelihood =
            particles.log_likelihood_for_document(document);
        audit.log_likelihood.insert(audit.log_likelihood.end(),
            likelihood.data(), likelihood.data() + samples);
        const auto proposal =
            particles.log_proposal_for_document(document);
        audit.log_proposal.insert(audit.log_proposal.end(),
            proposal.data(), proposal.data() + samples);
        const auto origins =
            particles.proposal_origins_for_document(document);
        audit.proposal_origins.insert(audit.proposal_origins.end(),
            origins.data(), origins.data() + samples);
        audit.proposal_candidates[local] =
            particles.proposal_candidates[document];
    }
    const Expectation probe = particle_expectation_impl(
        audit, model, ExpectationRequest{false, false, false}, enabled);
    if (probe.component_bound_violations > 0
        || probe.possible_component_documents == 0) {
        return false;
    }
    const double exact_fraction =
        static_cast<double>(probe.evaluated_component_documents)
        / probe.possible_component_documents;
    const double dimension = std::max(1, audit.dimension);
    const double bound_fraction = 1.0 / dimension;
    return exact_fraction <= 0.5
        && bound_fraction + exact_fraction
            <= 1.0 - requested.minimum_work_reduction;
}

struct HardPartitionMoments {
    Eigen::VectorXi counts;
    RowMajorMatrixXd means;
    std::vector<Eigen::MatrixXd> scatter;
    Eigen::MatrixXd pooled_scatter;
};

HardPartitionMoments hard_partition_moments(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (assignments.size() != documents || components <= 0) {
        throw std::invalid_argument("Invalid UAC hard partition");
    }
    HardPartitionMoments out;
    out.counts = Eigen::VectorXi::Zero(components);
    out.means = RowMajorMatrixXd::Zero(components, dimension);
    for (int32_t d = 0; d < documents; ++d) {
        const int32_t component = assignments(d);
        if (component < 0 || component >= components) {
            throw std::invalid_argument(
                "UAC initial partition label is out of range");
        }
        ++out.counts(component);
        out.means.row(component) += data.coordinates.row(d);
    }
    for (int32_t c = 0; c < components; ++c) {
        if (out.counts(c) <= 0) {
            throw std::runtime_error(
                "UAC initial partition produced an empty component");
        }
        out.means.row(c) /= out.counts(c);
    }
    out.scatter.assign(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    for (int32_t d = 0; d < documents; ++d) {
        const int32_t component = assignments(d);
        const Eigen::VectorXd residual =
            data.coordinates.row(d).transpose()
            - out.means.row(component).transpose();
        out.scatter[component].noalias() +=
            residual * residual.transpose();
    }
    out.pooled_scatter =
        Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& scatter : out.scatter) {
        out.pooled_scatter += scatter;
    }
    out.pooled_scatter /= documents;
    out.pooled_scatter = 0.5
        * (out.pooled_scatter + out.pooled_scatter.transpose());
    return out;
}

Eigen::MatrixXd measurement_covariance(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision) {
    const FisherApproximation fisher = fisher_approximation_impl(
        coordinate, document, basis, helmert, ProposalKind::ExactFisher);
    Eigen::MatrixXd precision =
        fisher.information + regularizing_precision;
    precision = 0.5 * (precision + precision.transpose());
    Eigen::LLT<Eigen::MatrixXd> llt(precision);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "UAC deconvolution measurement precision is not positive definite");
    }
    Eigen::MatrixXd covariance = llt.solve(
        Eigen::MatrixXd::Identity(precision.rows(), precision.cols()));
    covariance = 0.5 * (covariance + covariance.transpose());
    if (!covariance.allFinite()) {
        throw std::runtime_error(
            "UAC deconvolution measurement covariance is nonfinite");
    }
    return covariance;
}

Eigen::MatrixXd measurement_covariance(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    int32_t document) {
    return measurement_covariance(
        data.coordinates.row(document).transpose(),
        data.counts[document], basis, helmert, regularizing_precision);
}

Eigen::MatrixXd shared_measurement_precision(
    const std::vector<HardPartitionMoments>& moments,
    double scalar_precision, double relative_floor) {
    if (moments.empty() || !(scalar_precision >= 0.0)
        || !std::isfinite(scalar_precision) || !(relative_floor > 0.0)) {
        throw std::invalid_argument(
            "Invalid UAC deconvolution regularizing precision");
    }
    const int32_t dimension = static_cast<int32_t>(
        moments.front().pooled_scatter.rows());
    if (scalar_precision > 0.0) {
        return scalar_precision
            * Eigen::MatrixXd::Identity(dimension, dimension);
    }
    Eigen::MatrixXd pooled =
        Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& value : moments) {
        if (value.pooled_scatter.rows() != dimension
            || value.pooled_scatter.cols() != dimension) {
            throw std::runtime_error(
                "Incompatible UAC start scatters");
        }
        pooled += value.pooled_scatter;
    }
    pooled /= moments.size();
    const double floor = std::max(1e-12,
        relative_floor * pooled.trace() / dimension);
    pooled = floor_covariance(pooled, floor);
    Eigen::LLT<Eigen::MatrixXd> llt(pooled);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "UAC shared deconvolution scatter is not positive definite");
    }
    return llt.solve(
        Eigen::MatrixXd::Identity(dimension, dimension));
}

std::vector<std::vector<Eigen::MatrixXd>>
measurement_sums_by_partition(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const std::vector<Eigen::VectorXi>& assignments,
    int32_t components,
    const IndexedDocumentSource* count_source = nullptr) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    std::vector<std::vector<Eigen::MatrixXd>> out(
        assignments.size(), std::vector<Eigen::MatrixXd>(
            components, Eigen::MatrixXd::Zero(dimension, dimension)));
    for (const auto& assignment : assignments) {
        if (assignment.size() != documents) {
            throw std::invalid_argument(
                "Invalid UAC measurement partition");
        }
    }
    constexpr int32_t kCountBlock = 64;
    for (int32_t first = 0; first < documents; first += kCountBlock) {
        const int32_t count =
            std::min(kCountBlock, documents - first);
        DocumentBlock block;
        if (count_source) {
            count_source->read_range(first, count, block);
        }
        for (int32_t local = 0; local < count; ++local) {
            const int32_t d = first + local;
            const Document& document = count_source
                ? block.counts[local] : data.counts[d];
            const Eigen::MatrixXd covariance = measurement_covariance(
                data.coordinates.row(d).transpose(), document,
                basis, helmert, regularizing_precision);
            for (size_t start = 0; start < assignments.size(); ++start) {
                const int32_t component = assignments[start](d);
                if (component < 0 || component >= components) {
                    throw std::invalid_argument(
                        "UAC measurement partition label is out of range");
                }
                out[start][component] += covariance;
            }
        }
    }
    return out;
}

Model initialize_model_from_partition(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments, int32_t components,
    double shrinkage, double covariance_floor, double relative_floor) {
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (assignments.size() != data.coordinates.rows() || components <= 0
        || !(shrinkage >= 0.0) || !(relative_floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC initial partition");
    }
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(components);
    Model model;
    model.means = RowMajorMatrixXd::Zero(components, dimension);
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        const int32_t component = assignments(d);
        if (component < 0 || component >= components) {
            throw std::invalid_argument(
                "UAC initial partition label is out of range");
        }
        ++counts(component);
        model.means.row(component) += data.coordinates.row(d);
    }
    for (int32_t c = 0; c < components; ++c) {
        if (counts(c) <= 0) {
            throw std::runtime_error(
                "UAC initial partition produced an empty component");
        }
        model.means.row(c) /= counts(c);
    }
    std::vector<Eigen::MatrixXd> scatter(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        const int32_t component = assignments(d);
        const Eigen::VectorXd residual = data.coordinates.row(d).transpose()
            - model.means.row(component).transpose();
        scatter[component].noalias() += residual * residual.transpose();
    }
    model.shrinkage_target = Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& value : scatter) model.shrinkage_target += value;
    model.shrinkage_target /= data.coordinates.rows();
    const double target_floor = std::max(1e-12,
        relative_floor * model.shrinkage_target.trace() / dimension);
    model.shrinkage_target = floor_covariance(
        model.shrinkage_target, target_floor);
    model.weights = counts.cast<double>() / data.coordinates.rows();
    model.covariances.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        model.covariances.push_back(floor_covariance(
            (scatter[c] + shrinkage * model.shrinkage_target)
                / (counts(c) + shrinkage),
            covariance_floor));
    }
    return model;
}

Model initialize_model_from_corrected_moments(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    const HardPartitionMoments& moments,
    const std::vector<Eigen::MatrixXd>& measurement_sum,
    double shrinkage, double covariance_floor) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(moments.counts.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (assignments.size() != documents
        || moments.means.rows() != components
        || moments.means.cols() != dimension
        || measurement_sum.size() != static_cast<size_t>(components)
        || !(shrinkage >= 0.0) || !(covariance_floor > 0.0)) {
        throw std::invalid_argument(
            "Invalid UAC corrected-moment initialization");
    }
    Eigen::MatrixXd corrected_target = moments.pooled_scatter;
    for (const auto& sum : measurement_sum) {
        corrected_target -= sum / documents;
    }
    corrected_target = floor_covariance(
        corrected_target, covariance_floor);

    Model model;
    model.weights = moments.counts.cast<double>() / documents;
    model.means = moments.means;
    model.shrinkage_target = corrected_target;
    model.covariances.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        Eigen::MatrixXd numerator =
            moments.scatter[c] - measurement_sum[c]
            + shrinkage * corrected_target;
        model.covariances.push_back(floor_covariance(
            numerator / (moments.counts(c) + shrinkage),
            covariance_floor));
    }
    return model;
}

Pilot pilot_from_map(const Dataset& data, const Model& model,
    const Expectation& expectation,
    double relative_floor) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (expectation.documents != data.coordinates.rows()
        || expectation.membership.size() != components
        || expectation.first.rows() != components
        || expectation.second.size() != static_cast<size_t>(components)
        || !(relative_floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC winning MAP responsibilities");
    }
    Pilot out;
    out.weights = model.weights;
    out.means = model.means;
    std::vector<Eigen::MatrixXd> raw_covariances(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    const Eigen::VectorXd& membership = expectation.membership;
    for (int32_t c = 0; c < components; ++c) {
        const Eigen::VectorXd mean = model.means.row(c).transpose();
        raw_covariances[c] = expectation.second[c]
            - expectation.first.row(c).transpose() * mean.transpose()
            - mean * expectation.first.row(c)
            + membership(c) * mean * mean.transpose();
        raw_covariances[c] = 0.5 * (raw_covariances[c]
            + raw_covariances[c].transpose());
    }
    out.pooled_covariance = Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& scatter : raw_covariances) {
        out.pooled_covariance += scatter;
    }
    const double total_membership = membership.sum();
    if (!(total_membership > 0.0)) {
        throw std::runtime_error("UAC winning MAP has no active membership");
    }
    out.pooled_covariance /= total_membership;
    const double target_floor = std::max(1e-12,
        relative_floor * out.pooled_covariance.trace() / dimension);
    out.pooled_covariance = floor_covariance(
        out.pooled_covariance, target_floor);
    out.covariances = model.covariances;
    const double epsilon = membership_epsilon(data.coordinates.rows());
    for (int32_t c = 0; c < components; ++c) {
        if (membership(c) > epsilon && model.weights(c) > 0.0) {
            raw_covariances[c] /= membership(c);
            raw_covariances[c] = 0.5 * (raw_covariances[c]
                + raw_covariances[c].transpose());
        } else {
            out.covariances[c] = out.pooled_covariance;
        }
    }
    return out;
}

Pilot pilot_from_model(const Model& model) {
    if (model.covariance_kind != CovarianceKind::Dense
        || model.covariances.size()
            != static_cast<size_t>(model.weights.size())) {
        throw std::invalid_argument(
            "UAC corrected-moment pilot requires dense covariance");
    }
    Pilot out;
    out.weights = model.weights;
    out.means = model.means;
    out.covariances = model.covariances;
    out.pooled_covariance = model.shrinkage_target;
    return out;
}

struct DeconvolutionScore {
    double log_likelihood = 0.0;
    double responsibility_entropy_sum = 0.0;
    double gaussian_seconds = 0.0;
};

std::vector<DeconvolutionScore> deconvolution_marginal_scores(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const std::vector<Model>& models,
    const IndexedDocumentSource* count_source = nullptr) {
    if (models.empty()) return {};
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components =
        static_cast<int32_t>(models.front().weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if ((!count_source
            && data.counts.size() != static_cast<size_t>(documents))
        || (count_source && count_source->documents() != documents)
        || regularizing_precision.rows() != dimension
        || regularizing_precision.cols() != dimension) {
        throw std::invalid_argument(
            "Invalid UAC deconvolution score input");
    }
    for (const auto& model : models) {
        if (model.covariance_kind != CovarianceKind::Dense
            || model.weights.size() != components
            || model.means.rows() != components
            || model.means.cols() != dimension
            || model.covariances.size()
                != static_cast<size_t>(components)) {
            throw std::invalid_argument(
                "Incompatible UAC deconvolution candidate");
        }
    }
    const int32_t requested_blocks = expectation_shards(
        documents, components, dimension, -1);
    const int32_t block_size =
        (documents + requested_blocks - 1) / requested_blocks;
    const int32_t n_blocks =
        (documents + block_size - 1) / block_size;
    std::vector<std::vector<double>> block_likelihood(
        n_blocks, std::vector<double>(models.size(), 0.0));
    std::vector<std::vector<double>> block_entropy(
        n_blocks, std::vector<double>(models.size(), 0.0));
    std::vector<double> block_seconds(n_blocks, 0.0);
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        std::vector<Eigen::VectorXd> log_score(
            models.size(), Eigen::VectorXd(components));
        std::vector<std::vector<Eigen::LLT<Eigen::MatrixXd>>> solvers(
            models.size(),
            std::vector<Eigen::LLT<Eigen::MatrixXd>>(components));
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        DocumentBlock count_block;
        if (count_source) {
            count_source->read_range(begin, end - begin, count_block);
        }
        const auto work_start = std::chrono::steady_clock::now();
        for (int32_t d = begin; d < end; ++d) {
            const Eigen::VectorXd observed =
                data.coordinates.row(d).transpose();
            const Eigen::MatrixXd measurement = measurement_covariance(
                observed,
                count_source
                    ? count_block.counts[d - begin] : data.counts[d],
                basis, helmert, regularizing_precision);
            for (size_t candidate = 0; candidate < models.size();
                    ++candidate) {
                const Model& model = models[candidate];
                log_score[candidate].setConstant(
                    -std::numeric_limits<double>::infinity());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    Eigen::MatrixXd marginal =
                        model.covariances[c] + measurement;
                    marginal = 0.5 * (
                        marginal + marginal.transpose());
                    solvers[candidate][c].compute(marginal);
                    if (solvers[candidate][c].info()
                            != Eigen::Success) {
                        throw std::runtime_error(
                            "UAC deconvolution marginal covariance is not "
                            "positive definite");
                    }
                    const Eigen::MatrixXd lower =
                        solvers[candidate][c].matrixL();
                    const double log_determinant =
                        2.0 * lower.diagonal().array().log().sum();
                    const Eigen::VectorXd residual = observed
                        - model.means.row(c).transpose();
                    log_score[candidate](c) = std::log(model.weights(c))
                        - 0.5 * (dimension * kLog2Pi + log_determinant
                            + residual.dot(
                                solvers[candidate][c].solve(residual)));
                }
                const double normalizer =
                    logsumexp(log_score[candidate]);
                if (!std::isfinite(normalizer)) {
                    throw std::runtime_error(
                        "UAC deconvolution has no finite component "
                        "evidence");
                }
                const Eigen::VectorXd responsibility =
                    (log_score[candidate].array() - normalizer).exp();
                block_likelihood[block_index][candidate] += normalizer;
                for (int32_t c = 0; c < components; ++c) {
                    const double weight = responsibility(c);
                    if (weight > 0.0) {
                        block_entropy[block_index][candidate] -=
                            weight * std::log(weight);
                    }
                }
            }
        }
        block_seconds[block_index] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - work_start).count();
    });
    std::vector<DeconvolutionScore> out(models.size());
    const double total_seconds =
        std::accumulate(block_seconds.begin(), block_seconds.end(), 0.0);
    for (size_t candidate = 0; candidate < models.size(); ++candidate) {
        for (int32_t block = 0; block < n_blocks; ++block) {
            out[candidate].log_likelihood +=
                block_likelihood[block][candidate];
            out[candidate].responsibility_entropy_sum +=
                block_entropy[block][candidate];
        }
        out[candidate].gaussian_seconds =
            total_seconds / models.size();
    }
    return out;
}

struct ModelUpdate {
    bool valid = false;
    int32_t active_components = 0;
};

LowRankDiagonalCovariance shrink_factor_covariance(
    const LowRankDiagonalCovariance& raw,
    const LowRankDiagonalCovariance& target, double membership,
    double shrinkage, int32_t rank, double floor) {
    const double alpha = membership / (membership + shrinkage);
    LowRankDiagonalCovariance out;
    out.diagonal = (alpha * raw.diagonal
        + (1.0 - alpha) * target.diagonal).cwiseMax(floor);
    const int32_t columns = static_cast<int32_t>(raw.factor.cols()
        + target.factor.cols());
    if (rank == 0 || columns == 0) {
        out.factor = RowMajorMatrixXd(raw.diagonal.size(), 0);
        if (columns > 0) {
            out.diagonal.array() += alpha
                * raw.factor.array().square().rowwise().sum();
            out.diagonal.array() += (1.0 - alpha)
                * target.factor.array().square().rowwise().sum();
        }
        return out;
    }
    Eigen::MatrixXd combined(raw.diagonal.size(), columns);
    if (raw.factor.cols() > 0) {
        combined.leftCols(raw.factor.cols()) = std::sqrt(alpha) * raw.factor;
    }
    if (target.factor.cols() > 0) {
        combined.rightCols(target.factor.cols()) = std::sqrt(1.0 - alpha)
            * target.factor;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(combined,
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    const int32_t retained = std::min<int32_t>(rank,
        static_cast<int32_t>(svd.singularValues().size()));
    out.factor = svd.matrixU().leftCols(retained)
        * svd.singularValues().head(retained).asDiagonal();
    if (retained < svd.singularValues().size()) {
        const Eigen::MatrixXd discarded = svd.matrixU().middleCols(
            retained, svd.matrixU().cols() - retained)
            * svd.singularValues().segment(retained,
                svd.singularValues().size() - retained).asDiagonal();
        out.diagonal.array() += discarded.array().square().rowwise().sum();
    }
    out.diagonal = out.diagonal.cwiseMax(floor);
    return out;
}

ModelUpdate update_model(Model& model, const Expectation& expectation,
    double shrinkage, double covariance_floor, bool adaptive_target = false) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t documents = expectation.documents;
    const double epsilon = membership_epsilon(documents);
    if (!expectation.membership.allFinite()
        || (expectation.membership.array() < 0.0).any()) {
        return {};
    }
    double active_mass = 0.0;
    for (int32_t c = 0; c < components; ++c) {
        if (expectation.membership(c) > epsilon) {
            active_mass += expectation.membership(c);
        }
    }
    if (!(active_mass > 0.0) || !std::isfinite(active_mass)) return {};

    Model next = model;
    next.weights.setZero();
    ModelUpdate result;
    std::vector<Eigen::MatrixXd> raw_dense(components);
    std::vector<LowRankDiagonalCovariance> raw_factor(components);
    for (int32_t c = 0; c < components; ++c) {
        const double membership = expectation.membership(c);
        if (!(membership > epsilon)) continue;
        if (!expectation.first.row(c).allFinite()
            || (model.covariance_kind == CovarianceKind::Dense
                && !expectation.second[c].allFinite())) {
            return {};
        }
        next.weights(c) = membership / active_mass;
        if (model.covariance_kind == CovarianceKind::FactorAnalytic) {
            const int32_t rank = static_cast<int32_t>(
                model.factor_covariances[c].factor.cols());
            if (expectation.sum_y2.cols() != model.means.cols()
                || expectation.sum_f.cols() != rank) {
                return {};
            }
            Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(
                rank + 1, rank + 1);
            gram(0, 0) = membership;
            if (rank > 0) {
                gram.block(0, 1, 1, rank) = expectation.sum_f.row(c);
                gram.block(1, 0, rank, 1) =
                    expectation.sum_f.row(c).transpose();
                gram.bottomRightCorner(rank, rank) =
                    expectation.sum_ff[c];
            }
            Eigen::LLT<Eigen::MatrixXd> regression_llt(gram);
            if (regression_llt.info() != Eigen::Success) return {};
            Eigen::MatrixXd cross(model.means.cols(), rank + 1);
            cross.col(0) = expectation.first.row(c).transpose();
            if (rank > 0) cross.rightCols(rank) = expectation.sum_yf[c];
            const Eigen::MatrixXd coefficient = regression_llt.solve(
                cross.transpose()).transpose();
            const Eigen::VectorXd mean = coefficient.col(0);
            const Eigen::MatrixXd loading = coefficient.rightCols(rank);
            const Eigen::MatrixXd centered_yf = expectation.sum_yf[c]
                - mean * expectation.sum_f.row(c);
            Eigen::VectorXd centered_y2 =
                expectation.sum_y2.row(c).transpose();
            centered_y2.array() -= 2.0 * mean.array()
                * expectation.first.row(c).transpose().array();
            centered_y2.array() += membership * mean.array().square();
            Eigen::VectorXd residual = centered_y2;
            residual.array() -= 2.0
                * (loading.cwiseProduct(centered_yf)).rowwise().sum().array();
            residual.array() += (loading * expectation.sum_ff[c])
                .cwiseProduct(loading).rowwise().sum().array();
            raw_factor[c].diagonal =
                (residual / membership).cwiseMax(covariance_floor);
            raw_factor[c].factor = loading;
            next.means.row(c) = mean.transpose();
            ++result.active_components;
            continue;
        }
        next.means.row(c) = expectation.first.row(c) / membership;
        const Eigen::VectorXd mean = next.means.row(c).transpose();
        Eigen::MatrixXd scatter = expectation.second[c]
            - membership * mean * mean.transpose();
        scatter = 0.5 * (scatter + scatter.transpose());
        if (!scatter.allFinite()) return {};
        raw_dense[c] = scatter / membership;
        ++result.active_components;
    }
    if (result.active_components == 0 || !next.weights.allFinite()
        || !next.means.allFinite()) {
        return {};
    }
    try {
        if (adaptive_target && shrinkage > 0.0) {
            Eigen::MatrixXd pooled = Eigen::MatrixXd::Zero(
                model.means.cols(), model.means.cols());
            for (int32_t c = 0; c < components; ++c) {
                const double membership = expectation.membership(c);
                if (!(membership > epsilon)) continue;
                pooled += membership * (model.covariance_kind
                        == CovarianceKind::Dense
                    ? raw_dense[c] : raw_factor[c].dense());
            }
            next.shrinkage_target = floor_covariance(
                pooled / active_mass, covariance_floor);
            if (model.covariance_kind == CovarianceKind::FactorAnalytic) {
                const int32_t rank = static_cast<int32_t>(
                    model.factor_covariances.front().factor.cols());
                next.factor_shrinkage_target = factorize_covariance(
                    next.shrinkage_target, rank, covariance_floor);
            }
        }
        for (int32_t c = 0; c < components; ++c) {
            const double membership = expectation.membership(c);
            if (!(membership > epsilon)) {
                if (model.covariance_kind == CovarianceKind::Dense) {
                    next.covariances[c] = next.shrinkage_target;
                } else {
                    next.factor_covariances[c] =
                        next.factor_shrinkage_target;
                }
                continue;
            }
            if (model.covariance_kind == CovarianceKind::Dense) {
                next.covariances[c] = floor_covariance(
                    (membership * raw_dense[c]
                        + shrinkage * next.shrinkage_target)
                        / (membership + shrinkage),
                    covariance_floor);
            } else {
                const int32_t rank = static_cast<int32_t>(
                    model.factor_covariances[c].factor.cols());
                next.factor_covariances[c] = shrink_factor_covariance(
                    raw_factor[c], next.factor_shrinkage_target, membership,
                    shrinkage, rank, covariance_floor);
            }
        }
    } catch (const std::runtime_error&) {
        return {};
    }
    result.valid = true;
    model = std::move(next);
    return result;
}

struct Candidate {
    Model model;
    RestartTrace trace;
    double objective = -std::numeric_limits<double>::infinity();
};

double mean_max_responsibility_change(
    const Eigen::Ref<const RowMajorMatrixXd>& current,
    const Eigen::Ref<const RowMajorMatrixXd>& previous) {
    if (current.rows() != previous.rows() || current.cols() != previous.cols()
        || current.rows() == 0) {
        throw std::invalid_argument(
            "Incompatible UAC responsibility convergence matrices");
    }
    double total = 0.0;
    for (Eigen::Index d = 0; d < current.rows(); ++d) {
        total += (current.row(d) - previous.row(d)).cwiseAbs().maxCoeff();
    }
    return total / current.rows();
}

void record_trace_point(RestartTrace& trace, const FitOptions& options,
    TraceEvent event, int32_t completed_updates, double objective,
    int32_t active_components,
    double relative_objective_change, double responsibility_change,
    double variance_change =
        std::numeric_limits<double>::quiet_NaN(),
    double mean_responsibility_entropy =
        std::numeric_limits<double>::quiet_NaN()) {
    RestartTrace::Point point;
    point.event = event;
    point.completed_updates = completed_updates;
    point.objective = objective;
    point.active_components = active_components;
    point.relative_objective_change = relative_objective_change;
    point.mean_max_responsibility_change = responsibility_change;
    point.median_absolute_relative_variance_change = variance_change;
    point.mean_responsibility_entropy = mean_responsibility_entropy;
    trace.points.push_back(std::move(point));
    if (options.iteration_callback) {
        options.iteration_callback({trace.phase, event, trace.start,
            completed_updates, relative_objective_change,
            responsibility_change, variance_change,
            mean_responsibility_entropy});
    }
}

void accumulate_estep_work(
    RestartTrace& trace, const Expectation& expectation) {
    trace.estep_work.gaussian_seconds += expectation.gaussian_seconds;
    trace.estep_work.component_bound_seconds +=
        expectation.component_bound_seconds;
    trace.estep_work.moment_seconds += expectation.moment_seconds;
    trace.estep_work.document_evaluations += expectation.documents;
    trace.estep_work.evaluated_component_documents +=
        expectation.evaluated_component_documents;
    trace.estep_work.possible_component_documents +=
        expectation.possible_component_documents;
    trace.estep_work.full_component_documents +=
        expectation.full_component_documents;
    trace.estep_work.component_bound_violations +=
        expectation.component_bound_violations;
}

Candidate fit_map_candidate(const Dataset& data, Model initial,
    const FitOptions& options, const RestartTrace& metadata) {
    Candidate out;
    out.trace = metadata;
    out.trace.handoff = HandoffMode::Map;
    out.trace.phase = TracePhase::PointMapEm;
    out.model = std::move(initial);
    ComponentScreeningOptions map_screening = options.component_screening;
    if (map_screening.mode == ComponentScreeningMode::Auto) {
        const bool enabled = resolve_map_component_screening(
            data, out.model, map_screening,
            static_cast<uint64_t>(metadata.seed));
        apply_auto_component_screening_resolution(
            map_screening, enabled);
    }
    const double shrinkage = options.adaptive_covariance_shrinkage
        ? options.covariance_shrinkage_strength : 0.0;
    RowMajorMatrixXd previous_responsibilities;
    double converged_log_likelihood =
        -std::numeric_limits<double>::infinity();
    double previous_objective_lower =
        -std::numeric_limits<double>::infinity();
    double previous_objective_upper =
        -std::numeric_limits<double>::infinity();
    double previous_omitted_mass = 0.0;
    bool reuse_converged_expectation = false;
    for (int32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        Expectation expectation = map_expectation(data, out.model,
            ExpectationRequest{true, false, true}, map_screening);
        accumulate_estep_work(out.trace, expectation);
        const double objective = expectation.log_likelihood
            + covariance_prior(out.model, shrinkage);
        const double objective_upper = expectation.log_likelihood_upper
            + covariance_prior(out.model, shrinkage);
        double relative_change = std::numeric_limits<double>::quiet_NaN();
        double responsibility_change =
            std::numeric_limits<double>::quiet_NaN();
        if (previous_responsibilities.size() > 0) {
            responsibility_change = mean_max_responsibility_change(
                expectation.responsibilities, previous_responsibilities)
                + expectation.maximum_omitted_component_mass
                + previous_omitted_mass;
        }
        if (std::isfinite(previous_objective_lower)) {
            relative_change = std::max(
                std::abs(objective_upper - previous_objective_lower),
                std::abs(objective - previous_objective_upper))
                / std::max({1.0, std::abs(previous_objective_lower),
                    std::abs(previous_objective_upper)});
        }
        record_trace_point(out.trace, options, TraceEvent::Evaluation,
            out.trace.completed_updates, objective,
            active_component_count(out.model), relative_change,
            responsibility_change);
        if (iteration > 0
            && (relative_change < options.objective_change_tolerance
                || responsibility_change
                    < options.responsibility_change_tolerance)) {
                out.trace.converged = true;
                converged_log_likelihood = expectation.log_likelihood;
                reuse_converged_expectation = true;
                break;
        }
        previous_responsibilities = expectation.responsibilities;
        previous_objective_lower = objective;
        previous_objective_upper = objective_upper;
        previous_omitted_mass =
            expectation.maximum_omitted_component_mass;
        const ModelUpdate update = update_model(out.model, expectation,
            shrinkage, options.covariance_floor);
        if (!update.valid) {
            out.trace.collapsed = true;
            return out;
        }
        ++out.trace.completed_updates;
    }
    if (reuse_converged_expectation) {
        out.objective = converged_log_likelihood;
    } else {
        const Expectation final_expectation = map_expectation(data, out.model,
            ExpectationRequest{false, false, false}, map_screening);
        accumulate_estep_work(out.trace, final_expectation);
        out.objective = final_expectation.log_likelihood;
    }
    out.trace.selection_objective = out.objective;
    record_trace_point(out.trace, options, TraceEvent::Terminal,
        out.trace.completed_updates,
        out.objective + covariance_prior(out.model, shrinkage),
        active_component_count(out.model),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());
    out.trace.succeeded = true;
    return out;
}

void score_corrected_moment_candidates(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const FitOptions& options, std::vector<Candidate>& candidates,
    const IndexedDocumentSource* count_source = nullptr) {
    std::vector<size_t> candidate_index;
    std::vector<Model> models;
    for (size_t index = 0; index < candidates.size(); ++index) {
        if (!candidates[index].trace.collapsed) {
            candidate_index.push_back(index);
            models.push_back(candidates[index].model);
        }
    }
    const std::vector<DeconvolutionScore> scores =
        deconvolution_marginal_scores(data, basis, helmert,
            regularizing_precision, models, count_source);
    for (size_t local = 0; local < scores.size(); ++local) {
        Candidate& candidate = candidates[candidate_index[local]];
        const DeconvolutionScore& score = scores[local];
        candidate.trace.estep_work.gaussian_seconds +=
            score.gaussian_seconds;
        candidate.trace.estep_work.document_evaluations +=
            data.coordinates.rows();
        candidate.objective = score.log_likelihood;
        record_trace_point(candidate.trace, options,
            TraceEvent::CandidateScore, 0, candidate.objective,
            active_component_count(candidate.model),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            score.responsibility_entropy_sum
                / std::max<Eigen::Index>(1, data.coordinates.rows()));
        candidate.trace.succeeded = true;
        candidate.trace.selection_objective = candidate.objective;
    }
}

Candidate fit_particle_candidate(
    const std::function<Expectation(const Model&)>& expectation_function,
    Model initial, const FitOptions& options,
    const RestartTrace& initialization_trace) {
    Candidate out;
    out.trace = initialization_trace;
    out.trace.points.clear();
    out.trace.model_trace.clear();
    out.trace.estep_work = {};
    out.trace.converged = false;
    out.trace.collapsed = false;
    out.trace.handoff = HandoffMode::Particle;
    out.trace.phase = TracePhase::ParticleEm;
    out.trace.fixed_em_iteration_schedule =
        options.particle_em_fixed_iterations > 0;
    out.trace.completed_updates = 0;
    out.model = std::move(initial);
    const double shrinkage = options.adaptive_covariance_shrinkage
        ? options.covariance_shrinkage_strength : 0.0;
    RowMajorMatrixXd previous_responsibilities;
    double converged_log_likelihood =
        -std::numeric_limits<double>::infinity();
    double previous_objective_lower =
        -std::numeric_limits<double>::infinity();
    double previous_objective_upper =
        -std::numeric_limits<double>::infinity();
    double previous_omitted_mass = 0.0;
    std::optional<Model> previous_variance_model;
    bool reuse_converged_expectation = false;
    bool adaptive_update_completed = false;
    int32_t model_iteration = 0;
    auto record_model = [&](bool final, double update_shrinkage_strength) {
        if (!options.capture_model_trace) return;
        ModelTraceEntry entry;
        entry.completed_updates = model_iteration;
        entry.event = final ? TraceEvent::Terminal : TraceEvent::Evaluation;
        entry.update_shrinkage_strength = update_shrinkage_strength;
        entry.model = out.model;
        out.trace.model_trace.push_back(std::move(entry));
    };
    if (options.adaptive_covariance_shrinkage) {
        record_model(false, 0.0);
        Expectation bootstrap = expectation_function(out.model);
        accumulate_estep_work(out.trace, bootstrap);
        record_trace_point(out.trace, options, TraceEvent::Evaluation,
            out.trace.completed_updates, bootstrap.log_likelihood,
            active_component_count(out.model),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN());
        previous_responsibilities = bootstrap.responsibilities;
        previous_objective_lower = bootstrap.log_likelihood;
        previous_objective_upper = bootstrap.log_likelihood_upper;
        previous_omitted_mass =
            bootstrap.maximum_omitted_component_mass;
        previous_variance_model = out.model;
        const ModelUpdate update = update_model(out.model, bootstrap, 0.0,
            options.covariance_floor);
        if (!update.valid) {
            out.trace.collapsed = true;
            record_model(true,
                std::numeric_limits<double>::quiet_NaN());
            return out;
        }
        ++model_iteration;
        ++out.trace.completed_updates;
    }
    const int32_t update_budget = options.particle_em_fixed_iterations > 0
        ? options.particle_em_fixed_iterations
        : options.max_iterations;
    const int32_t regular_iterations = std::max(
        0, update_budget - out.trace.completed_updates);
    for (int32_t iteration = 0; iteration < regular_iterations; ++iteration) {
        record_model(false, shrinkage);
        Expectation expectation = expectation_function(out.model);
        accumulate_estep_work(out.trace, expectation);
        const double objective = expectation.log_likelihood;
        const double objective_upper = expectation.log_likelihood_upper;
        double relative_change = std::numeric_limits<double>::quiet_NaN();
        double responsibility_change =
            std::numeric_limits<double>::quiet_NaN();
        double variance_change =
            std::numeric_limits<double>::quiet_NaN();
        if (std::isfinite(previous_objective_lower)) {
            relative_change = std::max(
                std::abs(objective_upper - previous_objective_lower),
                std::abs(objective - previous_objective_upper))
                / std::max({1.0, std::abs(previous_objective_lower),
                    std::abs(previous_objective_upper)});
        }
        if (expectation.has_responsibility_change) {
            responsibility_change =
                expectation.mean_max_responsibility_change
                + expectation.maximum_omitted_component_mass
                + previous_omitted_mass;
        } else if (previous_responsibilities.size() > 0) {
            responsibility_change = mean_max_responsibility_change(
                expectation.responsibilities, previous_responsibilities)
                + expectation.maximum_omitted_component_mass
                + previous_omitted_mass;
        }
        if (previous_variance_model.has_value()) {
            variance_change = median_absolute_relative_variance_change(
                out.model, *previous_variance_model,
                options.covariance_floor);
        }
        record_trace_point(out.trace, options, TraceEvent::Evaluation,
            out.trace.completed_updates, objective,
            active_component_count(out.model), relative_change,
            responsibility_change, variance_change);
        const bool convergence_eligible =
            !options.adaptive_covariance_shrinkage
            || adaptive_update_completed;
        const bool variance_converged =
            options.particle_variance_change_tolerance == 0.0
            || (std::isfinite(variance_change)
                && variance_change
                    < options.particle_variance_change_tolerance);
        if (!out.trace.fixed_em_iteration_schedule
            && convergence_eligible && std::isfinite(relative_change)
            && variance_converged
            && (relative_change < options.objective_change_tolerance
                || responsibility_change
                    < options.responsibility_change_tolerance)) {
            out.trace.converged = true;
            converged_log_likelihood = expectation.log_likelihood;
            reuse_converged_expectation = true;
            break;
        }
        previous_responsibilities = expectation.responsibilities;
        previous_objective_lower = objective;
        previous_objective_upper = objective_upper;
        previous_omitted_mass =
            expectation.maximum_omitted_component_mass;
        previous_variance_model = out.model;
        const ModelUpdate update = update_model(out.model, expectation,
            shrinkage, options.covariance_floor,
            options.adaptive_covariance_shrinkage);
        if (!update.valid) {
            out.trace.collapsed = true;
            record_model(true,
                std::numeric_limits<double>::quiet_NaN());
            return out;
        }
        ++model_iteration;
        ++out.trace.completed_updates;
        if (options.adaptive_covariance_shrinkage) {
            adaptive_update_completed = true;
        }
    }
    if (reuse_converged_expectation) {
        out.objective = converged_log_likelihood;
    } else {
        const Expectation final_expectation = expectation_function(out.model);
        accumulate_estep_work(out.trace, final_expectation);
        out.objective = final_expectation.log_likelihood;
    }
    record_trace_point(out.trace, options, TraceEvent::Terminal,
        out.trace.completed_updates, out.objective,
        active_component_count(out.model),
        std::numeric_limits<double>::quiet_NaN(),
        std::numeric_limits<double>::quiet_NaN());
    record_model(true, std::numeric_limits<double>::quiet_NaN());
    out.trace.succeeded = true;
    return out;
}

template<class ParticleCollection>
ScoreResult score_particles_impl(
    const ParticleCollection& particles, const Model& model,
    const ComponentScreeningOptions& screening = {}) {
    ScoreResult out;
    Expectation expectation = particle_expectation(particles, model,
        ExpectationRequest{true, true, false}, screening);
    out.responsibilities = std::move(expectation.responsibilities);
    out.particle_diagnostics = std::move(expectation.particle_diagnostics);
    out.gaussian_seconds = expectation.gaussian_seconds;
    out.moment_seconds = expectation.moment_seconds;
    out.estimated_peak_expectation_workspace_bytes =
        expectation.peak_workspace_bytes;
    out.sampling_seconds = particles.sampling_seconds;
    out.likelihood_seconds = particles.likelihood_seconds;
    out.fisher_work_seconds = particles.fisher_work_seconds;
    out.proposal_component_work_seconds =
        particles.proposal_component_work_seconds;
    out.proposal_draw_density_work_seconds =
        particles.proposal_draw_density_work_seconds;
    out.proposal_precision_fallback_seconds =
        particles.proposal_precision_fallback_seconds;
    out.proposal_precision_fallbacks =
        particles.proposal_precision_fallbacks;
    out.estimated_peak_proposal_workspace_bytes =
        particles.proposal_workspace_bytes;
    out.component_screening_options = screening;
    out.particle_component_screening =
        screening.mode != ComponentScreeningMode::Off;
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
    out.proposal_components_constructed =
        particles.proposal_components_constructed;
    out.proposal_components_possible =
        particles.proposal_components_possible;
    out.per_document_proposal_components = particles.proposal_candidates;
    int64_t total_samples = 0;
    out.per_document_particles.resize(particles.documents);
    for (int32_t d = 0; d < particles.documents; ++d) {
        const int32_t samples = particles.samples_for_document(d);
        out.per_document_particles[d] = samples;
        total_samples += samples;
    }
    out.particle_samples = total_samples;
    out.resident_particle_bytes = sizeof(double)
            * static_cast<uint64_t>(total_samples)
            * (particles.dimension + 2)
        + sizeof(int32_t) * (
            static_cast<uint64_t>(total_samples)
            + static_cast<uint64_t>(particles.documents));
    if constexpr (std::is_same_v<ParticleCollection, RaggedParticleSet>) {
        out.resident_particle_bytes += sizeof(int64_t)
            * static_cast<uint64_t>(particles.offsets.size());
    }
    out.particle_generation_passes = 1;
    return out;
}

ScoreResult score_particles(const ParticleSet& particles, const Model& model,
    const ComponentScreeningOptions& screening = {}) {
    return score_particles_impl(particles, model, screening);
}

ScoreResult score_particles(
    const RaggedParticleSet& particles, const Model& model,
    const ComponentScreeningOptions& screening = {}) {
    ScoreResult out = score_particles_impl(particles, model, screening);
    out.calibration_seconds = particles.calibration_seconds;
    out.calibration_samples = particles.calibration_samples;
    out.reused_calibration_samples = particles.reused_calibration_samples;
    out.adaptive_particle_diagnostics = particles.adaptive_diagnostics;
    return out;
}

std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> out;
    std::string token;
    std::istringstream input(line);
    while (input >> token) out.push_back(token);
    return out;
}

double entropy(const Eigen::Ref<const Eigen::RowVectorXd>& probability) {
    double value = 0.0;
    for (Eigen::Index i = 0; i < probability.size(); ++i) {
        if (probability(i) > 0.0) value -= probability(i) * std::log(probability(i));
    }
    return value;
}

const char* adaptive_particle_mode_name(
    const AdaptiveParticleOptions& options) {
    if (options.responsibility_se_target.has_value()
        && options.moment_ess_target.has_value()) {
        return "responsibility_moment";
    }
    if (options.responsibility_se_target.has_value()) {
        return "responsibility";
    }
    if (options.moment_ess_target.has_value()) return "moment";
    return "fixed";
}

double optional_target_or_zero(const std::optional<double>& value) {
    return value.value_or(0.0);
}

int32_t parse_state_int32(const std::string& text) {
    size_t consumed = 0;
    long long value = 0;
    try {
        value = std::stoll(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid UAC state integer: " + text);
    }
    if (consumed != text.size()
        || value < std::numeric_limits<int32_t>::min()
        || value > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("Invalid UAC state integer: " + text);
    }
    return static_cast<int32_t>(value);
}

uint64_t parse_state_uint64(const std::string& text) {
    if (text.empty() || text.front() == '-') {
        throw std::runtime_error(
            "Invalid UAC state unsigned integer: " + text);
    }
    size_t consumed = 0;
    unsigned long long value = 0;
    try {
        value = std::stoull(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error(
            "Invalid UAC state unsigned integer: " + text);
    }
    if (consumed != text.size()) {
        throw std::runtime_error(
            "Invalid UAC state unsigned integer: " + text);
    }
    return static_cast<uint64_t>(value);
}

double parse_state_double(const std::string& text) {
    size_t consumed = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid UAC state number: " + text);
    }
    if (consumed != text.size() || !std::isfinite(value)) {
        throw std::runtime_error("Invalid UAC state number: " + text);
    }
    return value;
}

bool parse_state_bool(const std::string& text) {
    if (text == "0") return false;
    if (text == "1") return true;
    throw std::runtime_error("Invalid UAC state boolean: " + text);
}

} // namespace

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
            model_covariance_dense(current, static_cast<int32_t>(c));
        const Eigen::MatrixXd previous_covariance =
            model_covariance_dense(previous, static_cast<int32_t>(c));
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

FisherApproximation fisher_approximation(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal) {
    return fisher_approximation_impl(
        coordinate, document, basis, helmert, proposal);
}

Eigen::Map<const Eigen::VectorXd> ParticleSet::value(int32_t document,
    int32_t sample) const {
    return Eigen::Map<const Eigen::VectorXd>(
        values.data() + (static_cast<int64_t>(document) * samples + sample)
            * dimension,
        dimension);
}

double ParticleSet::log_q(int32_t document, int32_t sample) const {
    return log_proposal(document, sample);
}

int32_t ParticleSet::samples_for_document(int32_t) const {
    return samples;
}

Eigen::Map<const RowMajorMatrixXd> ParticleSet::values_for_document(
    int32_t document) const {
    return Eigen::Map<const RowMajorMatrixXd>(
        values.data() + static_cast<int64_t>(document) * samples * dimension,
        samples, dimension);
}

Eigen::Map<const Eigen::VectorXd> ParticleSet::log_likelihood_for_document(
    int32_t document) const {
    return Eigen::Map<const Eigen::VectorXd>(
        log_likelihood.data() + static_cast<int64_t>(document) * samples,
        samples);
}

Eigen::Map<const Eigen::VectorXd> ParticleSet::log_proposal_for_document(
    int32_t document) const {
    return Eigen::Map<const Eigen::VectorXd>(
        log_proposal.data() + static_cast<int64_t>(document) * samples,
        samples);
}

Eigen::Map<const Eigen::VectorXi>
ParticleSet::proposal_origins_for_document(int32_t document) const {
    return Eigen::Map<const Eigen::VectorXi>(
        proposal_origins.data()
            + static_cast<int64_t>(document) * samples,
        samples);
}

int32_t RaggedParticleSet::samples_for_document(int32_t document) const {
    return static_cast<int32_t>(offsets.at(document + 1)
        - offsets.at(document));
}

Eigen::Map<const RowMajorMatrixXd> RaggedParticleSet::values_for_document(
    int32_t document) const {
    const int64_t offset = offsets.at(document);
    return Eigen::Map<const RowMajorMatrixXd>(
        values.data() + offset * dimension,
        samples_for_document(document), dimension);
}

Eigen::Map<const Eigen::VectorXd>
RaggedParticleSet::log_likelihood_for_document(int32_t document) const {
    const int64_t offset = offsets.at(document);
    return Eigen::Map<const Eigen::VectorXd>(log_likelihood.data() + offset,
        samples_for_document(document));
}

Eigen::Map<const Eigen::VectorXd>
RaggedParticleSet::log_proposal_for_document(int32_t document) const {
    const int64_t offset = offsets.at(document);
    return Eigen::Map<const Eigen::VectorXd>(log_proposal.data() + offset,
        samples_for_document(document));
}

Eigen::Map<const Eigen::VectorXi>
RaggedParticleSet::proposal_origins_for_document(int32_t document) const {
    const int64_t offset = offsets.at(document);
    return Eigen::Map<const Eigen::VectorXi>(
        proposal_origins.data() + offset, samples_for_document(document));
}

Eigen::MatrixXd normalized_helmert(int32_t topics) {
    if (topics < 2) throw std::invalid_argument("UAC requires at least two topics");
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(topics - 1, topics);
    for (int32_t row = 0; row < topics - 1; ++row) {
        const double denominator = std::sqrt((row + 1.0) * (row + 2.0));
        out.block(row, 0, 1, row + 1).setConstant(1.0 / denominator);
        out(row, row + 1) = -(row + 1.0) / denominator;
    }
    return out;
}

RowMajorMatrixXd ilr_transform(const Eigen::Ref<const RowMajorMatrixXd>& values,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, double floor) {
    if (values.cols() != helmert.cols() || floor <= 0.0) {
        throw std::invalid_argument("Invalid UAC ILR transform dimensions or floor");
    }
    RowMajorMatrixXd out(values.rows(), helmert.rows());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        Eigen::VectorXd normalized = values.row(row).transpose().array().max(floor);
        normalized /= normalized.sum();
        out.row(row) = (helmert * normalized.array().log().matrix()).transpose();
    }
    return out;
}

RowMajorMatrixXd ilr_inverse(const Eigen::Ref<const RowMajorMatrixXd>& values,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    if (values.cols() != helmert.rows()) {
        throw std::invalid_argument("Invalid UAC inverse ILR dimensions");
    }
    RowMajorMatrixXd out(values.rows(), helmert.cols());
    for (Eigen::Index row = 0; row < values.rows(); ++row) {
        out.row(row) = composition_from_coordinate(
            values.row(row).transpose(), helmert).transpose();
    }
    return out;
}

void normalize_basis(Basis& basis) {
    if (basis.probabilities.rows() == 0 || basis.probabilities.cols() < 2
        || basis.features.size() != static_cast<size_t>(basis.probabilities.rows())
        || basis.topics.size() != static_cast<size_t>(basis.probabilities.cols())
        || !basis.probabilities.allFinite()
        || (basis.probabilities.array() < 0.0).any()) {
        throw std::invalid_argument("Invalid UAC topic basis");
    }
    for (Eigen::Index topic = 0; topic < basis.probabilities.cols(); ++topic) {
        const double total = basis.probabilities.col(topic).sum();
        if (!(total > 0.0)) throw std::invalid_argument("UAC basis has an empty topic");
        basis.probabilities.col(topic) /= total;
    }
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
    for (const auto& name : basis.features) value = hash_string(value, name);
    for (const auto& name : basis.topics) value = hash_string(value, name);
    for (Eigen::Index row = 0; row < basis.probabilities.rows(); ++row) {
        for (Eigen::Index column = 0; column < basis.probabilities.cols(); ++column) {
            const double number = basis.probabilities(row, column);
            value = fnv_append(value, &number, sizeof(number));
        }
    }
    return value;
}

ParticleSet make_particle_range(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    const PilotCache& pilot_cache,
    ProposalKind proposal_kind, int32_t samples, uint64_t seed,
    double fisher_broadening, int32_t n_threads,
    const ProposalScreeningPlan* screening_plan, int32_t first_document,
    int32_t documents, int32_t global_first_document = -1) {
    const int32_t global_first = global_first_document >= 0
        ? global_first_document : first_document;
    if (samples <= 0 || data.counts.size() != data.identifiers.size()
        || data.coordinates.rows() != static_cast<Eigen::Index>(data.counts.size())
        || basis.probabilities.cols() != helmert.cols()
        || first_document < 0 || documents <= 0
        || static_cast<int64_t>(first_document) + documents
            > data.coordinates.rows()
        || !(fisher_broadening > 0.0)
        || !std::isfinite(fisher_broadening)) {
        throw std::invalid_argument("Invalid UAC particle input");
    }
    ParticleSet out;
    out.first_document = global_first;
    out.documents = documents;
    out.samples = samples;
    out.dimension = static_cast<int32_t>(helmert.rows());
    const size_t particle_rows = checked_mul(
        static_cast<size_t>(out.documents), static_cast<size_t>(samples),
        "UAC particle rows");
    checked_mul(particle_rows, static_cast<size_t>(out.dimension),
        "UAC particle values");
    if (particle_rows > static_cast<size_t>(
        std::numeric_limits<Eigen::Index>::max())) {
        throw std::overflow_error(
            "UAC particle row count exceeds Eigen index range");
    }
    out.values.resize(static_cast<Eigen::Index>(particle_rows), out.dimension);
    out.log_likelihood.resize(out.documents, samples);
    out.log_proposal.resize(out.documents, samples);
    out.proposal_origins.resize(particle_rows);
    out.proposal_candidates.resize(out.documents);
    const uint64_t proposal_workspace = sizeof(double)
        * static_cast<uint64_t>(pilot.weights.size())
        * (static_cast<uint64_t>(out.dimension) * out.dimension
            + out.dimension + 3);
    out.proposal_workspace_bytes = proposal_workspace;
    std::atomic<int64_t> fisher_nanoseconds{0};
    std::atomic<int64_t> proposal_nanoseconds{0};
    std::atomic<int64_t> draw_nanoseconds{0};
    std::atomic<int64_t> fallback_nanoseconds{0};
    std::atomic<int64_t> fallbacks{0};
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    const auto sampling_start = std::chrono::steady_clock::now();
    tbb::parallel_for(int32_t{0}, out.documents, [&](int32_t local_document) {
        const int32_t document = first_document + local_document;
        const int32_t global_document = global_first + local_document;
        const Eigen::VectorXd center =
            data.coordinates.row(document).transpose();
        const auto fisher_start = std::chrono::steady_clock::now();
        const FisherApproximation fisher = fisher_approximation_impl(center,
            data.counts[document], basis, helmert, proposal_kind);
        const auto proposal_start = std::chrono::steady_clock::now();
        fisher_nanoseconds.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                proposal_start - fisher_start).count(),
            std::memory_order_relaxed);
        const std::vector<int32_t>* candidates =
            screening_plan && screening_plan->enabled
            ? &screening_plan->candidates[global_document] : nullptr;
        const DocumentProposal proposal = fisher_proposal(center, fisher,
            pilot, pilot_cache, fisher_broadening, candidates);
        out.proposal_candidates[local_document] =
            static_cast<int32_t>(proposal.weights.size());
        fallback_nanoseconds.fetch_add(static_cast<int64_t>(
            proposal.precision_fallback_seconds * 1e9),
            std::memory_order_relaxed);
        fallbacks.fetch_add(proposal.precision_fallbacks,
            std::memory_order_relaxed);
        const auto draw_start = std::chrono::steady_clock::now();
        proposal_nanoseconds.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                draw_start - proposal_start).count(),
            std::memory_order_relaxed);
        const uint64_t document_seed = hash_string(
            seed ^ 0x9e3779b97f4a7c15ull, data.identifiers[document]);
        std::mt19937_64 engine(document_seed);
        std::discrete_distribution<int32_t> choose(proposal.weights.data(),
            proposal.weights.data() + proposal.weights.size());
        std::normal_distribution<double> normal(0.0, 1.0);
        for (int32_t sample = 0; sample < samples; ++sample) {
            const int32_t component = choose(engine);
            out.proposal_origins[
                static_cast<size_t>(local_document) * samples + sample] =
                proposal.component_ids[component];
            Eigen::VectorXd draw(out.dimension);
            for (int32_t dim = 0; dim < out.dimension; ++dim) {
                draw(dim) = normal(engine);
            }
            proposal.precision_lower[component].transpose()
                .triangularView<Eigen::Upper>().solveInPlace(draw);
            const Eigen::VectorXd value = proposal.means[component]
                + std::sqrt(proposal.broadening) * draw;
            out.values.row(
                static_cast<Eigen::Index>(local_document) * samples + sample)
                = value.transpose();
        }
        const auto values = out.values.middleRows(
            static_cast<Eigen::Index>(local_document) * samples, samples);
        out.log_proposal.row(local_document) =
            proposal_log_density_rows(values, proposal).transpose();
        draw_nanoseconds.fetch_add(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - draw_start).count(),
            std::memory_order_relaxed);
    });
    out.sampling_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - sampling_start).count();
    out.fisher_work_seconds = 1e-9 * fisher_nanoseconds.load();
    out.proposal_component_work_seconds =
        1e-9 * proposal_nanoseconds.load();
    out.proposal_draw_density_work_seconds =
        1e-9 * draw_nanoseconds.load();
    out.proposal_precision_fallback_seconds =
        1e-9 * fallback_nanoseconds.load();
    out.proposal_precision_fallbacks = fallbacks.load();
    out.proposal_components_constructed = std::accumulate(
        out.proposal_candidates.begin(), out.proposal_candidates.end(),
        int64_t{0});
    const int32_t active_components = static_cast<int32_t>(
        (pilot.weights.array() > 0.0).count());
    out.proposal_components_possible = static_cast<int64_t>(out.documents)
        * active_components;
    const auto likelihood_start = std::chrono::steady_clock::now();
    tbb::parallel_for(int32_t{0}, out.documents, [&](int32_t local_document) {
        const int32_t document = first_document + local_document;
        const auto values = out.values.middleRows(
            static_cast<Eigen::Index>(local_document) * samples, samples);
        out.log_likelihood.row(local_document) = count_log_likelihood_rows(
            values, data.counts[document], basis, helmert).transpose();
    });
    out.likelihood_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - likelihood_start).count();
    return out;
}

template<class Matrix>
void draw_proposal_values(const DocumentProposal& proposal,
    uint64_t document_seed, Matrix& values, int32_t* origins = nullptr) {
    std::mt19937_64 engine(document_seed);
    std::discrete_distribution<int32_t> choose(proposal.weights.data(),
        proposal.weights.data() + proposal.weights.size());
    std::normal_distribution<double> normal(0.0, 1.0);
    Eigen::VectorXd draw(values.cols());
    for (Eigen::Index sample = 0; sample < values.rows(); ++sample) {
        const int32_t component = choose(engine);
        if (origins) origins[sample] = proposal.component_ids[component];
        for (Eigen::Index dim = 0; dim < values.cols(); ++dim) {
            draw(dim) = normal(engine);
        }
        proposal.precision_lower[component].transpose()
            .triangularView<Eigen::Upper>().solveInPlace(draw);
        values.row(sample) = (proposal.means[component]
            + std::sqrt(proposal.broadening) * draw).transpose();
    }
}

struct AdaptiveCountResult {
    int32_t particles = 0;
    AdaptiveParticleDiagnostic diagnostic;
};

AdaptiveCountResult adaptive_particle_count(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    const Eigen::Ref<const Eigen::VectorXd>& log_likelihood,
    const Eigen::Ref<const Eigen::VectorXd>& log_proposal,
    const Model& model,
    const std::vector<DenseGaussianSolver>& solvers,
    const AdaptiveParticleOptions& options,
    int32_t maximum_particles) {
    const int32_t samples = static_cast<int32_t>(values.rows());
    const int32_t components = static_cast<int32_t>(model.weights.size());
    if (samples < 2 || log_likelihood.size() != samples
        || log_proposal.size() != samples
        || static_cast<int32_t>(solvers.size()) != components) {
        throw std::invalid_argument("Invalid adaptive particle calibration");
    }
    const Eigen::VectorXd base = log_likelihood - log_proposal
        - Eigen::VectorXd::Constant(samples, std::log(samples));
    Eigen::MatrixXd log_tilt(components, samples);
    Eigen::VectorXd evidence(components), score(components);
    for (int32_t c = 0; c < components; ++c) {
        if (!(model.weights(c) > 0.0)) {
            evidence(c) = -std::numeric_limits<double>::infinity();
            score(c) = -std::numeric_limits<double>::infinity();
            log_tilt.row(c).setConstant(
                -std::numeric_limits<double>::infinity());
            continue;
        }
        log_tilt.row(c) = (base
            + solvers[c].log_density_rows(values)).transpose();
        evidence(c) = logsumexp(log_tilt.row(c).transpose());
        score(c) = std::log(model.weights(c)) + evidence(c);
    }
    const double normalizer = logsumexp(score);
    const Eigen::VectorXd responsibility =
        (score.array() - normalizer).exp();
    std::vector<int32_t> order;
    order.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        if (model.weights(c) > 0.0) order.push_back(c);
    }
    std::sort(order.begin(), order.end(), [&](int32_t left, int32_t right) {
        return responsibility(left) > responsibility(right);
    });
    if (order.empty()) {
        throw std::runtime_error("Adaptive calibration has no active component");
    }
    std::vector<int32_t> plausible;
    double cumulative = 0.0;
    for (size_t rank = 0; rank < order.size(); ++rank) {
        const int32_t c = order[rank];
        if (cumulative < options.plausible_mass
            || responsibility(c) >= options.plausible_responsibility) {
            plausible.push_back(c);
        }
        cumulative += responsibility(c);
    }
    AdaptiveCountResult out;
    out.diagnostic.preliminary_maximum_responsibility =
        responsibility.maxCoeff();
    for (int32_t c = 0; c < components; ++c) {
        if (responsibility(c) > 0.0) {
            out.diagnostic.preliminary_entropy -= responsibility(c)
                * std::log(responsibility(c));
        }
    }
    out.diagnostic.plausible_components =
        static_cast<int32_t>(plausible.size());

    Eigen::MatrixXd tau = Eigen::MatrixXd::Zero(components, samples);
    for (const int32_t c : order) {
        tau.row(c) = (log_tilt.row(c).array() - evidence(c)).exp();
    }
    double required = options.minimum_particles;
    AdaptiveParticleBinding binding = AdaptiveParticleBinding::Minimum;
    auto update_required = [&](double candidate,
            AdaptiveParticleBinding candidate_binding) {
        if (candidate > required) {
            required = candidate;
            binding = candidate_binding;
        }
    };
    if (options.responsibility_se_target.has_value()) {
        Eigen::RowVectorXd mixture_tau = Eigen::RowVectorXd::Zero(samples);
        for (const int32_t c : order) {
            mixture_tau += responsibility(c) * tau.row(c);
        }
        for (const int32_t c : order) {
            const double responsibility_se = responsibility(c)
                * std::sqrt(samples / (samples - 1.0)
                    * (tau.row(c) - mixture_tau).squaredNorm());
            out.diagnostic.maximum_responsibility_se = std::max(
                out.diagnostic.maximum_responsibility_se,
                responsibility_se);
        }
        out.diagnostic.projected_responsibility_particles = samples
            * std::pow(out.diagnostic.maximum_responsibility_se
                / *options.responsibility_se_target, 2.0);
        update_required(out.diagnostic.projected_responsibility_particles,
            AdaptiveParticleBinding::Responsibility);
    }
    if (options.moment_ess_target.has_value()) {
        for (const int32_t c : plausible) {
            const double relative_ess = 1.0
                / (samples * tau.row(c).squaredNorm());
            out.diagnostic.projected_moment_particles = std::max(
                out.diagnostic.projected_moment_particles,
                *options.moment_ess_target
                    / std::max(1e-12, relative_ess));
        }
        update_required(out.diagnostic.projected_moment_particles,
            AdaptiveParticleBinding::MomentEss);
    }
    int32_t selected = options.minimum_particles;
    while (selected < maximum_particles && selected < required) {
        selected = std::min(maximum_particles, selected * 2);
    }
    out.particles = selected;
    out.diagnostic.selected_particles = selected;
    out.diagnostic.binding = binding;
    return out;
}

RaggedParticleSet make_adaptive_particle_range(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal_kind, uint64_t seed, double fisher_broadening,
    int32_t n_threads, const Model& calibration_model,
    const AdaptiveParticleOptions& options,
    int32_t maximum_particles,
    const ProposalScreeningPlan* screening_plan,
    int32_t first_document, int32_t documents,
    int32_t global_first_document = -1) {
    const int32_t global_first = global_first_document >= 0
        ? global_first_document : first_document;
    const int32_t total_documents =
        static_cast<int32_t>(data.coordinates.rows());
    const int32_t dimension = static_cast<int32_t>(helmert.rows());
    if (!options.enabled() || documents <= 0 || first_document < 0
        || first_document > total_documents
        || documents > total_documents - first_document
        || options.calibration_particles < 2
        || options.minimum_particles <= 0
        || options.minimum_particles < options.calibration_particles
        || maximum_particles < options.minimum_particles
        || (options.responsibility_se_target.has_value()
            && !(*options.responsibility_se_target > 0.0))
        || !(options.plausible_mass > 0.0 && options.plausible_mass <= 1.0)
        || !(options.plausible_responsibility >= 0.0
            && options.plausible_responsibility <= 1.0)
        || (options.moment_ess_target.has_value()
            && !(*options.moment_ess_target > 0.0))) {
        throw std::invalid_argument("Invalid adaptive particle options");
    }
    std::vector<DenseGaussianSolver> calibration_solvers;
    calibration_solvers.reserve(calibration_model.weights.size());
    for (Eigen::Index c = 0; c < calibration_model.weights.size(); ++c) {
        if (calibration_model.weights(c) > 0.0) {
            calibration_solvers.emplace_back(
                calibration_model.means.row(c).transpose(),
                model_covariance_dense(calibration_model, c));
        } else {
            calibration_solvers.emplace_back();
        }
    }
    RaggedParticleSet out;
    out.first_document = global_first;
    out.documents = documents;
    out.dimension = dimension;
    out.maximum_samples = maximum_particles;
    out.offsets.assign(static_cast<size_t>(documents) + 1, 0);
    out.adaptive_diagnostics.resize(documents);
    out.proposal_candidates.resize(documents);
    out.calibration_samples = static_cast<int64_t>(documents)
        * options.calibration_particles;
    out.reused_calibration_samples = out.calibration_samples;
    const size_t minimum_total = checked_mul(static_cast<size_t>(documents),
        static_cast<size_t>(options.minimum_particles),
        "UAC adaptive minimum particle count");
    out.values.reserve(checked_mul(minimum_total,
        static_cast<size_t>(dimension), "UAC adaptive particle values"));
    out.log_likelihood.reserve(minimum_total);
    out.log_proposal.reserve(minimum_total);
    out.proposal_workspace_bytes = sizeof(double)
        * static_cast<uint64_t>(pilot.weights.size())
        * (static_cast<uint64_t>(dimension) * dimension + dimension + 3);
    std::atomic<int64_t> fisher_nanoseconds{0};
    std::atomic<int64_t> proposal_nanoseconds{0};
    std::atomic<int64_t> draw_nanoseconds{0};
    std::atomic<int64_t> fallback_nanoseconds{0};
    std::atomic<int64_t> fallbacks{0};
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, n_threads));
    constexpr int32_t kGenerationChunk = 128;
    for (int32_t begin = 0; begin < documents; begin += kGenerationChunk) {
        const int32_t end = std::min(documents, begin + kGenerationChunk);
        const int32_t size = end - begin;
        std::vector<DocumentProposal> proposals(size);
        std::vector<int32_t> counts(size, options.minimum_particles);
        std::vector<RowMajorMatrixXd> calibration_values(size);
        std::vector<std::vector<int32_t>> calibration_origins(size);
        std::vector<Eigen::VectorXd> calibration_log_q(size);
        std::vector<Eigen::VectorXd> calibration_log_likelihood(size);
        const auto calibration_start = std::chrono::steady_clock::now();
        tbb::parallel_for(int32_t{0}, size, [&](int32_t local) {
            const int32_t local_document = begin + local;
            const int32_t document = first_document + local_document;
            const int32_t global_document = global_first + local_document;
            const Eigen::VectorXd center =
                data.coordinates.row(document).transpose();
            const auto fisher_start = std::chrono::steady_clock::now();
            const FisherApproximation fisher = fisher_approximation_impl(
                center, data.counts[document], basis, helmert, proposal_kind);
            const auto proposal_start = std::chrono::steady_clock::now();
            fisher_nanoseconds.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    proposal_start - fisher_start).count(),
                std::memory_order_relaxed);
            const std::vector<int32_t>* candidates =
                screening_plan && screening_plan->enabled
                ? &screening_plan->candidates[global_document] : nullptr;
            proposals[local] = fisher_proposal(center, fisher, pilot,
                pilot_cache, fisher_broadening, candidates);
            out.proposal_candidates[local_document] =
                static_cast<int32_t>(proposals[local].weights.size());
            fallback_nanoseconds.fetch_add(static_cast<int64_t>(
                proposals[local].precision_fallback_seconds * 1e9),
                std::memory_order_relaxed);
            fallbacks.fetch_add(proposals[local].precision_fallbacks,
                std::memory_order_relaxed);
            proposal_nanoseconds.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - proposal_start).count(),
                std::memory_order_relaxed);
            RowMajorMatrixXd& calibration = calibration_values[local];
            calibration.resize(options.calibration_particles, dimension);
            calibration_origins[local].resize(
                options.calibration_particles);
            const uint64_t calibration_seed = hash_string(
                seed ^ 0x6a09e667f3bcc909ull,
                data.identifiers[document]);
            draw_proposal_values(
                proposals[local], calibration_seed, calibration,
                calibration_origins[local].data());
            calibration_log_q[local] = proposal_log_density_rows(
                calibration, proposals[local]);
            calibration_log_likelihood[local] = count_log_likelihood_rows(
                calibration, data.counts[document], basis, helmert);
            const AdaptiveCountResult allocation = adaptive_particle_count(
                calibration, calibration_log_likelihood[local],
                calibration_log_q[local], calibration_model,
                calibration_solvers, options, maximum_particles);
            counts[local] = allocation.particles;
            out.adaptive_diagnostics[local_document] =
                allocation.diagnostic;
        });
        out.calibration_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - calibration_start).count();
        for (int32_t local = 0; local < size; ++local) {
            out.offsets[begin + local + 1] = out.offsets[begin + local]
                + counts[local];
        }
        const int64_t total_samples = out.offsets[end];
        out.values.resize(checked_mul(static_cast<size_t>(total_samples),
            static_cast<size_t>(dimension), "UAC adaptive particle values"));
        out.log_likelihood.resize(total_samples);
        out.log_proposal.resize(total_samples);
        out.proposal_origins.resize(total_samples);
        const auto sampling_start = std::chrono::steady_clock::now();
        tbb::parallel_for(int32_t{0}, size, [&](int32_t local) {
            const int32_t local_document = begin + local;
            const int32_t document = first_document + local_document;
            const int64_t offset = out.offsets[local_document];
            const int32_t samples = counts[local];
            Eigen::Map<RowMajorMatrixXd> values(
                out.values.data() + offset * dimension, samples, dimension);
            const int32_t calibration_samples =
                options.calibration_particles;
            values.topRows(calibration_samples) = calibration_values[local];
            Eigen::Map<Eigen::VectorXd> stored_log_q(
                out.log_proposal.data() + offset, samples);
            stored_log_q.head(calibration_samples) =
                calibration_log_q[local];
            std::copy(calibration_origins[local].begin(),
                calibration_origins[local].end(),
                out.proposal_origins.begin() + offset);
            Eigen::Map<Eigen::VectorXd> stored_log_likelihood(
                out.log_likelihood.data() + offset, samples);
            stored_log_likelihood.head(calibration_samples) =
                calibration_log_likelihood[local];
            const auto draw_start = std::chrono::steady_clock::now();
            const uint64_t document_seed = hash_string(
                seed ^ 0x9e3779b97f4a7c15ull,
                data.identifiers[document]);
            auto additional = values.bottomRows(
                samples - calibration_samples);
            draw_proposal_values(proposals[local], document_seed, additional,
                out.proposal_origins.data() + offset + calibration_samples);
            stored_log_q.tail(samples - calibration_samples) =
                proposal_log_density_rows(additional, proposals[local]);
            draw_nanoseconds.fetch_add(
                std::chrono::duration_cast<std::chrono::nanoseconds>(
                    std::chrono::steady_clock::now() - draw_start).count(),
                std::memory_order_relaxed);
        });
        out.sampling_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - sampling_start).count();
        const auto likelihood_start = std::chrono::steady_clock::now();
        tbb::parallel_for(int32_t{0}, size, [&](int32_t local) {
            const int32_t local_document = begin + local;
            const int32_t document = first_document + local_document;
            const int64_t offset = out.offsets[local_document];
            const int32_t samples = counts[local];
            const int32_t additional_samples = samples
                - options.calibration_particles;
            if (additional_samples == 0) return;
            const Eigen::Map<const RowMajorMatrixXd> values(
                out.values.data() + (offset + options.calibration_particles)
                    * dimension,
                additional_samples, dimension);
            Eigen::Map<Eigen::VectorXd>(out.log_likelihood.data() + offset
                    + options.calibration_particles, additional_samples) =
                count_log_likelihood_rows(
                    values, data.counts[document], basis, helmert);
        });
        out.likelihood_seconds += std::chrono::duration<double>(
            std::chrono::steady_clock::now() - likelihood_start).count();
    }
    out.fisher_work_seconds = 1e-9 * fisher_nanoseconds.load();
    out.proposal_component_work_seconds =
        1e-9 * proposal_nanoseconds.load();
    out.proposal_draw_density_work_seconds =
        1e-9 * draw_nanoseconds.load();
    out.proposal_precision_fallback_seconds =
        1e-9 * fallback_nanoseconds.load();
    out.proposal_precision_fallbacks = fallbacks.load();
    out.proposal_components_constructed = std::accumulate(
        out.proposal_candidates.begin(), out.proposal_candidates.end(),
        int64_t{0});
    out.proposal_components_possible = static_cast<int64_t>(documents)
        * (pilot.weights.array() > 0.0).count();
    return out;
}

RaggedParticleSet make_adaptive_particles(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& pilot_cache,
    ProposalKind proposal_kind, uint64_t seed, double fisher_broadening,
    int32_t n_threads, const Model& calibration_model,
    const AdaptiveParticleOptions& options,
    int32_t maximum_particles,
    const ProposalScreeningPlan* screening_plan) {
    return make_adaptive_particle_range(data, basis, helmert, pilot,
        pilot_cache, proposal_kind, seed, fisher_broadening, n_threads,
        calibration_model, options, maximum_particles, screening_plan, 0,
        static_cast<int32_t>(data.coordinates.rows()));
}

ParticleSet make_particles(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert, const Pilot& pilot,
    ProposalKind proposal_kind, int32_t samples, uint64_t seed,
    double fisher_broadening, int32_t n_threads) {
    const PilotCache pilot_cache(pilot);
    return make_particle_range(data, basis, helmert, pilot, pilot_cache,
        proposal_kind,
        samples, seed, fisher_broadening, n_threads, nullptr, 0,
        static_cast<int32_t>(data.coordinates.rows()));
}

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

struct ParticleCacheMetrics {
    double sampling_seconds = 0.0;
    double likelihood_seconds = 0.0;
    double fisher_work_seconds = 0.0;
    double proposal_component_work_seconds = 0.0;
    double proposal_draw_density_work_seconds = 0.0;
    double proposal_precision_fallback_seconds = 0.0;
    double calibration_seconds = 0.0;
    int64_t proposal_precision_fallbacks = 0;
    uint64_t peak_bytes = 0;
    uint64_t proposal_workspace_bytes = 0;
    int64_t proposal_components_constructed = 0;
    int64_t proposal_components_possible = 0;
    int64_t calibration_samples = 0;
    int64_t reused_calibration_samples = 0;
    int32_t generation_passes = 0;

    void add(const ParticleSet& particles) {
        sampling_seconds += particles.sampling_seconds;
        likelihood_seconds += particles.likelihood_seconds;
        fisher_work_seconds += particles.fisher_work_seconds;
        proposal_component_work_seconds +=
            particles.proposal_component_work_seconds;
        proposal_draw_density_work_seconds +=
            particles.proposal_draw_density_work_seconds;
        proposal_precision_fallback_seconds +=
            particles.proposal_precision_fallback_seconds;
        proposal_precision_fallbacks +=
            particles.proposal_precision_fallbacks;
        proposal_components_constructed +=
            particles.proposal_components_constructed;
        proposal_components_possible +=
            particles.proposal_components_possible;
        peak_bytes = std::max(peak_bytes, particle_set_bytes(particles));
        proposal_workspace_bytes = std::max(
            proposal_workspace_bytes, particles.proposal_workspace_bytes);
    }

    void add(const RaggedParticleSet& particles) {
        sampling_seconds += particles.sampling_seconds;
        likelihood_seconds += particles.likelihood_seconds;
        fisher_work_seconds += particles.fisher_work_seconds;
        proposal_component_work_seconds +=
            particles.proposal_component_work_seconds;
        proposal_draw_density_work_seconds +=
            particles.proposal_draw_density_work_seconds;
        proposal_precision_fallback_seconds +=
            particles.proposal_precision_fallback_seconds;
        proposal_precision_fallbacks +=
            particles.proposal_precision_fallbacks;
        calibration_seconds += particles.calibration_seconds;
        calibration_samples += particles.calibration_samples;
        reused_calibration_samples += particles.reused_calibration_samples;
        proposal_components_constructed +=
            particles.proposal_components_constructed;
        proposal_components_possible +=
            particles.proposal_components_possible;
        const uint64_t samples = particles.offsets.empty()
            ? 0 : static_cast<uint64_t>(particles.offsets.back());
        const uint64_t bytes = sizeof(double) * samples
                * (particles.dimension + 2)
            + sizeof(int32_t) * (
                samples + static_cast<uint64_t>(particles.documents))
            + sizeof(int64_t)
                * static_cast<uint64_t>(particles.offsets.size());
        peak_bytes = std::max(peak_bytes, bytes);
        proposal_workspace_bytes = std::max(
            proposal_workspace_bytes, particles.proposal_workspace_bytes);
    }
};

constexpr uint64_t kParticleCacheMagic = 0x3148434143504341ull;
constexpr uint32_t kParticleCacheVersion = 1;

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

uint64_t cache_hash_bytes(uint64_t hash, const void* data, size_t bytes) {
    const auto* value = static_cast<const unsigned char*>(data);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= value[i];
        hash *= 1099511628211ull;
    }
    return hash;
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
            hash = cache_hash_bytes(
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
    return cache_hash_bytes(hash, &value, sizeof(value));
}

template<class Value>
void write_cache_values(std::ostream& out, const Value* values, size_t count,
    uint64_t& checksum) {
    if (count == 0) return;
    const size_t bytes = sizeof(Value) * count;
    out.write(reinterpret_cast<const char*>(values), bytes);
    checksum = cache_hash_bytes(checksum, values, bytes);
}

template<class Value>
void read_cache_values(std::istream& in, Value* values, size_t count,
    uint64_t& checksum) {
    if (count == 0) return;
    const size_t bytes = sizeof(Value) * count;
    in.read(reinterpret_cast<char*>(values), bytes);
    if (!in) throw std::runtime_error("Truncated UAC particle cache shard");
    checksum = cache_hash_bytes(checksum, values, bytes);
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
    int32_t global_document) {
    const Eigen::VectorXd center =
        data.coordinates.row(data_document).transpose();
    const FisherApproximation fisher = fisher_approximation_impl(
        center, data.counts[data_document], basis, helmert, proposal_kind);
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
    for (int32_t local = 0; local < particles.documents; ++local) {
        const int32_t global_document = particles.first_document + local;
        const int32_t data_document = data_is_local_block
            ? local : global_document;
        const int32_t samples = particles.samples_for_document(local);
        const int32_t calibration_samples = ragged
            ? std::min(samples, adaptive.calibration_particles) : 0;
        const DocumentProposal proposal = particle_cache_proposal(
            data, basis, helmert, pilot, pilot_cache, proposal_kind,
            broadening, screening_plan, data_document, global_document);
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
        uint64_t position_checksum = cache_hash_bytes(
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

using CachedParticleShard = std::variant<ParticleSet, RaggedParticleSet>;

CachedParticleShard read_particle_cache(const std::filesystem::path& path) {
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
        std::vector<int64_t> offsets(
            static_cast<size_t>(header.documents) + 1, 0);
        std::vector<double> values;
        std::vector<double> log_likelihood;
        std::vector<double> log_proposal;
        std::vector<int32_t> origins;
        std::vector<int32_t> proposal_candidates;
        std::vector<AdaptiveParticleDiagnostic> adaptive_diagnostics;
        values.reserve(static_cast<size_t>(header.total_samples)
            * header.dimension);
        log_likelihood.reserve(header.total_samples);
        log_proposal.reserve(header.total_samples);
        origins.reserve(header.total_samples);
        proposal_candidates.reserve(header.documents);
        if (ragged) adaptive_diagnostics.reserve(header.documents);
        for (int32_t document = 0;
                document < header.documents; ++document) {
            int32_t samples = 0;
            int32_t calibration_samples = 0;
            int32_t proposal_components = 0;
            uint64_t document_seed = 0;
            uint64_t calibration_seed = 0;
            double broadening = 0.0;
            read_cache_values(in, &samples, 1, checksum);
            read_cache_values(in, &calibration_samples, 1, checksum);
            read_cache_values(in, &proposal_components, 1, checksum);
            read_cache_values(in, &document_seed, 1, checksum);
            read_cache_values(in, &calibration_seed, 1, checksum);
            read_cache_values(in, &broadening, 1, checksum);
            if (samples <= 0 || samples > header.samples
                || calibration_samples < 0
                || calibration_samples > samples
                || proposal_components <= 0
                || !(broadening > 0.0)) {
                throw std::runtime_error(
                    "Invalid UAC factor particle cache record");
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
            uint64_t expected_position_checksum = 0;
            read_cache_values(in, &expected_position_checksum, 1, checksum);
            RowMajorMatrixXd regenerated(samples, header.dimension);
            std::vector<int32_t> document_origins(samples);
            if (ragged) {
                if (calibration_samples > 0) {
                    auto calibration =
                        regenerated.topRows(calibration_samples);
                    draw_proposal_values(proposal, calibration_seed,
                        calibration, document_origins.data());
                }
                if (samples > calibration_samples) {
                    auto additional = regenerated.bottomRows(
                        samples - calibration_samples);
                    draw_proposal_values(proposal, document_seed,
                        additional,
                        document_origins.data() + calibration_samples);
                }
            } else {
                draw_proposal_values(proposal, document_seed,
                    regenerated, document_origins.data());
            }
            const uint64_t actual_position_checksum = cache_hash_bytes(
                1469598103934665603ull, regenerated.data(),
                sizeof(double) * regenerated.size());
            if (actual_position_checksum != expected_position_checksum) {
                throw std::runtime_error(
                    "UAC factor particle regeneration checksum mismatch: "
                    + path.string());
            }
            values.insert(values.end(), regenerated.data(),
                regenerated.data() + regenerated.size());
            const size_t old_log_size = log_likelihood.size();
            log_likelihood.resize(old_log_size + samples);
            log_proposal.resize(old_log_size + samples);
            read_cache_values(in, log_likelihood.data() + old_log_size,
                samples, checksum);
            read_cache_values(in, log_proposal.data() + old_log_size,
                samples, checksum);
            origins.insert(origins.end(), document_origins.begin(),
                document_origins.end());
            proposal_candidates.push_back(proposal_components);
            if (ragged) {
                AdaptiveParticleDiagnostic diagnostic;
                read_cache_values(in, &diagnostic, 1, checksum);
                adaptive_diagnostics.push_back(diagnostic);
            }
            offsets[document + 1] = offsets[document] + samples;
        }
        if (offsets.back() != header.total_samples
            || checksum != header.payload_checksum || in.peek() != EOF) {
            throw std::runtime_error(
                "UAC factor particle cache checksum mismatch: "
                + path.string());
        }
        if (ragged) {
            RaggedParticleSet out;
            out.first_document = header.first_document;
            out.documents = header.documents;
            out.dimension = header.dimension;
            out.maximum_samples = header.samples;
            out.offsets = std::move(offsets);
            out.values = std::move(values);
            out.log_likelihood = std::move(log_likelihood);
            out.log_proposal = std::move(log_proposal);
            out.proposal_origins = std::move(origins);
            out.proposal_candidates =
                std::move(proposal_candidates);
            out.adaptive_diagnostics =
                std::move(adaptive_diagnostics);
            return out;
        }
        ParticleSet out;
        out.first_document = header.first_document;
        out.documents = header.documents;
        out.dimension = header.dimension;
        out.samples = header.samples;
        out.values = Eigen::Map<RowMajorMatrixXd>(
            values.data(), header.total_samples, header.dimension);
        out.log_likelihood = Eigen::Map<RowMajorMatrixXd>(
            log_likelihood.data(), header.documents, header.samples);
        out.log_proposal = Eigen::Map<RowMajorMatrixXd>(
            log_proposal.data(), header.documents, header.samples);
        out.proposal_origins = std::move(origins);
        out.proposal_candidates = std::move(proposal_candidates);
        return out;
    }
    if (header.ragged != 1) {
        throw std::runtime_error("Unknown UAC particle cache storage kind");
    }
    RaggedParticleSet out;
    out.first_document = header.first_document;
    out.documents = header.documents;
    out.dimension = header.dimension;
    out.maximum_samples = header.samples;
    out.offsets.resize(static_cast<size_t>(header.documents) + 1);
    out.values.resize(static_cast<size_t>(header.total_samples)
        * header.dimension);
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

struct ParticleCache {
    std::filesystem::path directory;
    std::vector<std::filesystem::path> shards;
    std::vector<int32_t> first_documents;
    std::vector<int32_t> document_counts;
    std::vector<int32_t> arithmetic_shards;
    ParticleCacheMetrics metrics;
    int32_t documents = 0;
    int32_t dimension = 0;
    bool adaptive = false;
    bool reused = false;
    int32_t rebuilds = 0;
    bool auto_component_screening_enabled = false;
    uint64_t bytes = 0;
    StreamingParticleStorage storage = StreamingParticleStorage::Positions;
};

uint64_t particle_cache_key(const Dataset& data, const Basis& basis,
    const Pilot& pilot, const Model& initial_model, ProposalKind proposal,
    int32_t maximum_samples, uint64_t seed, double broadening,
    const AdaptiveParticleOptions& adaptive,
    const ComponentScreeningOptions& screening,
    StreamingParticleStorage storage, int32_t block_documents,
    const IndexedDocumentSource* count_source = nullptr) {
    uint64_t hash = 1469598103934665603ull;
    const char runtime[] =
#if defined(__clang__)
        "clang-" __clang_version__;
#elif defined(__GNUC__)
        "gcc-" __VERSION__;
#else
        "unknown-compiler";
#endif
    hash = cache_hash_bytes(hash, runtime, sizeof(runtime));
    const int32_t eigen_version[] = {
        EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION};
    hash = cache_hash_bytes(hash, eigen_version, sizeof(eigen_version));
    const bool fast_math =
#if defined(__FAST_MATH__)
        true;
#else
        false;
#endif
    hash = cache_hash_value(hash, fast_math);
    const bool compiled_fma =
#if defined(__FMA__)
        true;
#else
        false;
#endif
    hash = cache_hash_value(hash, compiled_fma);
    hash = cache_hash_value(hash, basis.checksum);
    hash = cache_hash_value(hash, proposal);
    hash = cache_hash_value(hash, maximum_samples);
    hash = cache_hash_value(hash, seed);
    hash = cache_hash_value(hash, broadening);
    hash = cache_hash_value(hash, storage);
    hash = cache_hash_value(hash, block_documents);
    hash = cache_hash_value(hash, adaptive.calibration_particles);
    hash = cache_hash_value(hash, adaptive.minimum_particles);
    const bool has_responsibility_target =
        adaptive.responsibility_se_target.has_value();
    const bool has_moment_target = adaptive.moment_ess_target.has_value();
    hash = cache_hash_value(hash, has_responsibility_target);
    hash = cache_hash_value(hash,
        adaptive.responsibility_se_target.value_or(0.0));
    hash = cache_hash_value(hash, has_moment_target);
    hash = cache_hash_value(hash,
        adaptive.moment_ess_target.value_or(0.0));
    hash = cache_hash_value(hash, adaptive.plausible_mass);
    hash = cache_hash_value(hash, adaptive.plausible_responsibility);
    hash = cache_hash_value(hash, screening.mode);
    hash = cache_hash_value(hash, screening.tail_mass);
    hash = cache_hash_value(hash, screening.proposal_proxy_tail_mass);
    hash = cache_hash_value(hash, screening.minimum_components);
    hash = cache_hash_value(hash, screening.maximum_components);
    hash = cache_hash_value(hash, screening.audit_documents);
    hash = cache_hash_value(hash, screening.minimum_work_reduction);
    for (const auto& identifier : data.identifiers) {
        hash = cache_hash_bytes(hash, identifier.data(), identifier.size());
        const unsigned char delimiter = 0xff;
        hash = cache_hash_value(hash, delimiter);
    }
    hash = cache_hash_bytes(hash, data.coordinates.data(),
        sizeof(double) * data.coordinates.size());
    constexpr int32_t kHashBlock = 256;
    for (int32_t first = 0;
            first < static_cast<int32_t>(data.identifiers.size());
            first += kHashBlock) {
        const int32_t count = std::min<int32_t>(
            kHashBlock,
            static_cast<int32_t>(data.identifiers.size()) - first);
        DocumentBlock block;
        if (count_source) {
            count_source->read_range(first, count, block);
        }
        for (int32_t local = 0; local < count; ++local) {
            const Document& document = count_source
                ? block.counts[local] : data.counts[first + local];
            const uint64_t entries = document.ids.size();
            hash = cache_hash_value(hash, entries);
            hash = cache_hash_bytes(hash, document.ids.data(),
                sizeof(uint32_t) * document.ids.size());
            hash = cache_hash_bytes(hash, document.cnts.data(),
                sizeof(double) * document.cnts.size());
        }
    }
    hash = cache_hash_bytes(hash, pilot.weights.data(),
        sizeof(double) * pilot.weights.size());
    hash = cache_hash_bytes(hash, pilot.means.data(),
        sizeof(double) * pilot.means.size());
    for (const auto& covariance : pilot.covariances) {
        hash = cache_hash_bytes(hash, covariance.data(),
            sizeof(double) * covariance.size());
    }
    hash = cache_hash_bytes(hash, pilot.pooled_covariance.data(),
        sizeof(double) * pilot.pooled_covariance.size());
    hash = cache_hash_bytes(hash, initial_model.weights.data(),
        sizeof(double) * initial_model.weights.size());
    hash = cache_hash_bytes(hash, initial_model.means.data(),
        sizeof(double) * initial_model.means.size());
    if (initial_model.covariance_kind == CovarianceKind::Dense) {
        for (const auto& covariance : initial_model.covariances) {
            hash = cache_hash_bytes(hash, covariance.data(),
                sizeof(double) * covariance.size());
        }
    } else {
        for (const auto& covariance : initial_model.factor_covariances) {
            hash = cache_hash_bytes(hash, covariance.diagonal.data(),
                sizeof(double) * covariance.diagonal.size());
            hash = cache_hash_bytes(hash, covariance.factor.data(),
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
    const IndexedDocumentSource* count_source = nullptr) {
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
    cache.directory = root / cache_key_text(key);
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
        if (options.rebuild_cache || !std::filesystem::exists(complete)) {
            return false;
        }
        try {
            {
                std::ifstream marker(complete);
                std::string version;
                uint64_t expected_manifest_hash = 0;
                if (!(marker >> version >> std::hex
                        >> expected_manifest_hash)
                    || version != "uac-particle-cache-v1"
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
                cache.auto_component_screening_enabled = enabled != 0;
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
            for (const auto& shard : cache.shards) {
                const CachedParticleShard value =
                    read_particle_cache(shard);
                std::visit([&](const auto& particles) {
                    cache.metrics.peak_bytes = std::max(
                        cache.metrics.peak_bytes,
                        particle_set_bytes(particles));
                }, value);
                cache.bytes += std::filesystem::file_size(shard);
            }
            cache.reused = true;
            return true;
        } catch (const std::exception&) {
            return false;
        }
    };
    const bool existing_entry = std::filesystem::exists(complete);
    if (validate_existing()) return cache;
    if (existing_entry) {
        ++cache.rebuilds;
    }

    if (std::filesystem::exists(cache.directory)) {
        std::filesystem::remove_all(cache.directory);
    }
    std::filesystem::create_directories(root);
    const std::filesystem::path temporary =
        cache.directory.string() + ".tmp";
    if (std::filesystem::exists(temporary)) {
        std::filesystem::remove_all(temporary);
    }
    std::filesystem::create_directories(temporary);
    for (size_t shard = 0; shard < cache.shards.size(); ++shard) {
        cache.shards[shard] = temporary
            / cache.shards[shard].filename();
    }
    auto load_particle_block = [&](int32_t first, int32_t count) {
        Dataset block;
        DocumentBlock count_block;
        count_source->read_range(first, count, count_block);
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
        cache.auto_component_screening_enabled =
            resolve_particle_component_screening(
                audit, initial_model, screening, local_audit);
    }
    {
        std::ofstream metadata(temporary / "manifest.tsv");
        metadata << "auto_component_screening_enabled\t"
            << static_cast<int32_t>(
                cache.auto_component_screening_enabled) << "\n"
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
    return cache;
}

struct CachedDocumentMetadata {
    std::vector<ParticleDiagnostic> diagnostics;
    std::vector<int32_t> particles;
    std::vector<int32_t> proposal_components;
    std::vector<AdaptiveParticleDiagnostic> adaptive;
};

class CachedResponsibilityState {
public:
    explicit CachedResponsibilityState(
        const std::filesystem::path& directory)
        : previous_(directory / "responsibilities.previous.bin"),
          current_(directory / "responsibilities.current.bin") {
        std::error_code error;
        std::filesystem::remove(previous_, error);
        std::filesystem::remove(current_, error);
    }

    ~CachedResponsibilityState() {
        std::error_code error;
        std::filesystem::remove(previous_, error);
        std::filesystem::remove(current_, error);
    }

    bool has_previous() const { return has_previous_; }
    const std::filesystem::path& previous_path() const {
        return previous_;
    }
    const std::filesystem::path& current_path() const {
        return current_;
    }

    void commit() {
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

private:
    std::filesystem::path previous_;
    std::filesystem::path current_;
    bool has_previous_ = false;
};

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
    CachedDocumentMetadata* metadata = nullptr,
    CachedResponsibilityState* responsibility_state = nullptr,
    const std::filesystem::path* responsibility_spool = nullptr,
    Eigen::VectorXd* effective_membership = nullptr) {
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
                    Expectation local = particle_expectation_impl(
                        particles, model, local_request, screening, 1,
                        &blocks[arithmetic]);
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
            Eigen::VectorXd row(components);
            for (int32_t c = 0; c < components; ++c) {
                std::ifstream in(
                    *responsibility_spool, std::ios::binary);
                if (!in) {
                    throw std::runtime_error(
                        "Cannot read final streaming UAC responsibilities");
                }
                for (int32_t d = 0; d < cache.documents; ++d) {
                    in.read(reinterpret_cast<char*>(row.data()),
                        sizeof(double) * components);
                    if (!in) {
                        throw std::runtime_error(
                            "Truncated final streaming UAC "
                            "responsibilities");
                    }
                    (*effective_membership)(c) += row(c);
                }
            }
        }
    }
    return out;
}

ScoreResult score_particle_cache(const ParticleCache& cache,
    const Model& model, const ComponentScreeningOptions& screening,
    bool materialize_responsibilities, int32_t n_threads) {
    CachedDocumentMetadata metadata;
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const std::filesystem::path sidecar =
        cache.directory / "score-responsibilities.bin";
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
    return out;
}

FitResult fit_impl(const Dataset& data, Dataset* mutable_data,
    const Basis* basis, const FitOptions& options,
    const IndexedDocumentSource* count_source = nullptr) {
    validate_dataset(data,
        options.handoff == HandoffMode::Particle && !count_source);
    validate_component_screening(options.component_screening);
    const int64_t total_starts = static_cast<int64_t>(options.kmeans_starts)
        + options.leiden_starts;
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
        || !(options.target_relative_floor > 0.0)
        || !(options.covariance_floor > 0.0)
        || !(options.covariance_shrinkage_strength >= 0.0)
        || !std::isfinite(options.covariance_shrinkage_strength)
        || !(options.fisher_broadening > 0.0)
        || !std::isfinite(options.fisher_broadening)) {
        throw std::invalid_argument("Invalid UAC fit options or dataset");
    }
    if (options.leiden_starts > 0
        && (options.leiden_neighbors <= 0
            || options.leiden_neighbors >= data.centers.rows()
            || options.leiden_max_iterations == 0
            || !(options.leiden_knn_epsilon >= 0.0)
            || !std::isfinite(options.leiden_knn_epsilon)
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
    tbb::global_control control(tbb::global_control::max_allowed_parallelism,
        std::max(1, options.n_threads));
    FitResult result;
    struct StartPartition {
        Eigen::VectorXi assignments;
        RestartTrace metadata;
    };
    std::vector<StartPartition> starts;
    starts.reserve(static_cast<size_t>(total_starts));
    auto append_start = [&](Eigen::VectorXi assignments,
                            RestartTrace metadata) {
        metadata.handoff = options.handoff;
        metadata.phase = options.handoff == HandoffMode::Particle
            ? TracePhase::CorrectedMomScore : TracePhase::PointMapEm;
        starts.push_back({
            std::move(assignments), std::move(metadata)});
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
        DenseKMeansResult clustering = cosine_dense_kmeans(
            data.centers, kmeans);
        append_start(std::move(clustering.assignments), metadata);
    }

    if (options.leiden_starts > 0) {
        CosineKnnOptions knn_options;
        knn_options.n_neighbors = options.leiden_neighbors;
        knn_options.knn_search_epsilon = options.leiden_knn_epsilon;
        knn_options.backend = options.leiden_knn_backend;
        knn_options.n_threads = options.n_threads;
        const CosineKnnResult knn = cosine_knn(data.centers, knn_options);
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
            const LeidenResult leiden = leiden_cluster(knn.graph.n_nodes,
                knn.graph.edges, knn.graph.weights, leiden_options);
            metadata.raw_communities = leiden.n_communities;
            metadata.reconciliation_count = std::abs(
                leiden.n_communities - options.n_components);
            DenseKMeansOptions reconcile_options;
            reconcile_options.n_clusters = options.n_components;
            reconcile_options.max_iterations = options.kmeans_max_iterations;
            reconcile_options.seed = metadata.seed;
            Eigen::VectorXi assignments = reconcile_cosine_communities(
                leiden.membership, leiden.n_communities,
                options.n_components, data.centers, reconcile_options);
            append_start(std::move(assignments), metadata);

            if (!adapting) continue;
            if (leiden.n_communities < options.n_components) {
                last_under_resolution = resolution;
                resolution = detail::increased_leiden_resolution(resolution,
                    leiden.n_communities, options.n_components);
            } else if (leiden.n_communities == options.n_components) {
                adapting = false;
            } else {
                if (last_under_resolution > 0.0) {
                    resolution = detail::midpoint_leiden_resolution(
                        last_under_resolution, resolution);
                }
                adapting = false;
            }
        }
    }

    std::vector<Candidate> candidates;
    candidates.reserve(static_cast<size_t>(total_starts));
    Eigen::MatrixXd initialization_precision;
    const Eigen::MatrixXd helmert =
        normalized_helmert(data.centers.cols());
    std::vector<HardPartitionMoments> partition_moments;
    std::vector<std::vector<Eigen::MatrixXd>> measurement_sums;
    if (options.handoff == HandoffMode::Particle) {
        partition_moments.reserve(starts.size());
        for (const auto& start : starts) {
            partition_moments.push_back(hard_partition_moments(
                data, start.assignments, options.n_components));
        }
        initialization_precision = shared_measurement_precision(
            partition_moments, options.initialization_ridge_precision,
            options.target_relative_floor);
        std::vector<Eigen::VectorXi> assignments;
        assignments.reserve(starts.size());
        for (const auto& start : starts) {
            assignments.push_back(start.assignments);
        }
        measurement_sums = measurement_sums_by_partition(
            data, *basis, helmert, initialization_precision, assignments,
            options.n_components, count_source);
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
                    options.covariance_floor);
                candidates.push_back(std::move(candidate));
            }
        } catch (const std::exception&) {
            Candidate failed;
            failed.trace = start.metadata;
            failed.trace.collapsed = true;
            candidates.push_back(std::move(failed));
        }
    }
    if (options.handoff == HandoffMode::Particle) {
        score_corrected_moment_candidates(data, *basis, helmert,
            initialization_precision, options, candidates, count_source);
        result.initialization_measurement_covariance_evaluations =
            2 * static_cast<int64_t>(data.coordinates.rows());
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
        selected->model.covariance_kind = CovarianceKind::FactorAnalytic;
        selected->model.factor_shrinkage_target = factorize_covariance(
            selected->model.shrinkage_target, rank,
            options.covariance_floor);
        selected->model.factor_covariances.clear();
        selected->model.factor_covariances.reserve(options.n_components);
        for (const auto& covariance : selected->model.covariances) {
            selected->model.factor_covariances.push_back(
                factorize_covariance(covariance, rank,
                    options.covariance_floor));
        }
        if (options.handoff == HandoffMode::Map) {
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

    if (options.handoff == HandoffMode::Map) {
        result.model = selected->model;
        result.score = score_map(data, result.model, options.n_threads,
            options.component_screening);
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
            particle_seed, options.component_screening, count_source);
    ComponentScreeningOptions particle_screening =
        options.component_screening;
    if (particle_screening.mode == ComponentScreeningMode::Auto) {
        apply_auto_component_screening_resolution(
            particle_screening, false);
    }
    Candidate particle;
    try {
        if (options.particle_engine == ParticleEngine::Stream) {
            ParticleCache cache = open_or_build_particle_cache(
                data, *basis, helmert, result.pilot, pilot_cache,
                options.proposal, options.n_particles, particle_seed,
                options.fisher_broadening, options.n_threads,
                particle_initial, options.adaptive_particles,
                &proposal_screening, options.component_screening,
                options.streaming, count_source);
            if (options.streaming.count_storage
                    == StreamingCountStorage::Source
                && mutable_data) {
                mutable_data->counts.clear();
                mutable_data->counts.shrink_to_fit();
            }
            if (options.component_screening.mode
                    == ComponentScreeningMode::Auto) {
                apply_auto_component_screening_resolution(
                    particle_screening,
                    cache.auto_component_screening_enabled);
            }
            CachedResponsibilityState responsibility_state(
                cache.directory);
            auto expectation_function = [&](const Model& model) {
                return cached_particle_expectation(cache, model,
                    particle_screening,
                    ExpectationRequest{false, false, true},
                    options.n_threads,
                    nullptr, &responsibility_state);
            };
            particle = fit_particle_candidate(expectation_function,
                particle_initial, options, selected->trace);
            if (!particle.trace.collapsed) {
                const auto score_start = std::chrono::steady_clock::now();
                result.score = score_particle_cache(
                    cache, particle.model, particle_screening,
                    options.streaming.count_storage
                        == StreamingCountStorage::Memory,
                    options.n_threads);
                add_screening_metrics(result.score,
                    options.component_screening, proposal_screening,
                    particle_screening);
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
                result.score.scoring_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - score_start).count();
            }
        } else if (options.adaptive_particles.enabled()) {
            const RaggedParticleSet particles = make_adaptive_particles(
                data, *basis, helmert, result.pilot, pilot_cache,
                options.proposal, particle_seed, options.fisher_broadening,
                options.n_threads, particle_initial,
                options.adaptive_particles, options.n_particles,
                &proposal_screening);
            if (options.component_screening.mode
                    == ComponentScreeningMode::Auto) {
                const bool enabled = resolve_particle_component_screening(
                    particles, particle_initial,
                    options.component_screening,
                    proposal_screening.audit_documents);
                apply_auto_component_screening_resolution(
                    particle_screening, enabled);
            }
            auto expectation_function = [&](const Model& model) {
                return particle_expectation(particles, model,
                    ExpectationRequest{true, false, true},
                    particle_screening);
            };
            particle = fit_particle_candidate(expectation_function,
                particle_initial, options, selected->trace);
            if (!particle.trace.collapsed) {
                const auto score_start = std::chrono::steady_clock::now();
                result.score = score_particles(
                    particles, particle.model, particle_screening);
                add_screening_metrics(result.score,
                    options.component_screening, proposal_screening,
                    particle_screening);
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
            }
        } else {
            const ParticleSet particles = make_particle_range(data, *basis,
                helmert, result.pilot, pilot_cache, options.proposal,
                options.n_particles, particle_seed,
                options.fisher_broadening, options.n_threads,
                &proposal_screening, 0,
                static_cast<int32_t>(data.coordinates.rows()));
            if (options.component_screening.mode
                    == ComponentScreeningMode::Auto) {
                const bool enabled = resolve_particle_component_screening(
                    particles, particle_initial,
                    options.component_screening,
                    proposal_screening.audit_documents);
                apply_auto_component_screening_resolution(
                    particle_screening, enabled);
            }
            auto expectation_function = [&](const Model& model) {
                return particle_expectation(particles, model,
                    ExpectationRequest{true, false, true},
                    particle_screening);
            };
            particle = fit_particle_candidate(expectation_function,
                particle_initial, options, selected->trace);
            if (!particle.trace.collapsed) {
                const auto score_start = std::chrono::steady_clock::now();
                result.score = score_particles(
                    particles, particle.model, particle_screening);
                add_screening_metrics(result.score,
                    options.component_screening, proposal_screening,
                    particle_screening);
                result.score.map_component_screening =
                    selected_map_screening.mode
                    == ComponentScreeningMode::On;
                result.score.particle_generation_seconds =
                    particles.sampling_seconds + particles.likelihood_seconds;
                result.score.scoring_seconds = std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - score_start).count();
            }
        }
    } catch (const std::exception& exception) {
        throw std::runtime_error(
            "Selected UAC initializer failed during particle EM: "
            + std::string(exception.what()));
    }
    result.traces.push_back(particle.trace);
    if (particle.trace.collapsed) {
        throw std::runtime_error(
            "Selected UAC initializer collapsed during particle EM");
    }
    result.model = particle.model;
    result.converged = particle.trace.converged;
    return result;
}

FitResult fit(Dataset& data, const Basis* basis,
    const FitOptions& options) {
    return fit_impl(data, &data, basis, options);
}

FitResult fit(const Dataset& data, const Basis* basis,
    const FitOptions& options) {
    return fit_impl(data, nullptr, basis, options);
}

FitResult fit_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source, const FitOptions& options) {
    return fit_impl(data, nullptr, &basis, options, &source);
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
    validate_dataset(data, !count_source);
    validate_basis(basis,
        checked_int32(data.centers.cols(), "topic count"));
    if (!count_source) validate_count_features(data, basis);
    validate_state(state);
    validate_component_screening(component_screening);
    validate_adaptive_particles(adaptive_particles, particles);
    if (particles <= 0 || state.basis_checksum != basis.checksum
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
            pilot_cache, proposal, state.fisher_broadening, particle_seed,
            component_screening, count_source);
    ComponentScreeningOptions particle_screening = component_screening;
    if (particle_screening.mode == ComponentScreeningMode::Auto) {
        apply_auto_component_screening_resolution(
            particle_screening, false);
    }
    if (options.particle_engine == ParticleEngine::Stream) {
        ParticleCache cache = open_or_build_particle_cache(
            data, basis, state.helmert, state.pilot, pilot_cache,
            proposal, particles, particle_seed, state.fisher_broadening,
            n_threads, state.model, adaptive_particles,
            &proposal_screening, component_screening, options.streaming,
            count_source);
        if (options.streaming.count_storage
                == StreamingCountStorage::Source
            && mutable_data) {
            mutable_data->counts.clear();
            mutable_data->counts.shrink_to_fit();
        }
        if (component_screening.mode == ComponentScreeningMode::Auto) {
            apply_auto_component_screening_resolution(
                particle_screening,
                cache.auto_component_screening_enabled);
        }
        const auto score_start = std::chrono::steady_clock::now();
        ScoreResult out = score_particle_cache(
            cache, state.model, particle_screening,
            options.streaming.count_storage
                == StreamingCountStorage::Memory,
            n_threads);
        add_screening_metrics(out, component_screening,
            proposal_screening, particle_screening);
        out.adaptive_particle_options = adaptive_particles;
        out.streaming_count_storage = options.streaming.count_storage;
        if (count_source) {
            out.streaming_count_spool_bytes =
                count_source->storage_bytes();
            out.streaming_peak_count_block_bytes =
                count_source->peak_block_bytes();
            out.streaming_external_count_parses = 1;
        }
        out.scoring_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - score_start).count();
        return out;
    }
    if (adaptive_particles.enabled()) {
        const auto particle_start = std::chrono::steady_clock::now();
        const RaggedParticleSet set = make_adaptive_particles(data, basis,
            state.helmert, state.pilot, pilot_cache, proposal,
            particle_seed,
            state.fisher_broadening, n_threads, state.model,
            adaptive_particles, particles, &proposal_screening);
        if (component_screening.mode == ComponentScreeningMode::Auto) {
            const bool enabled = resolve_particle_component_screening(
                set, state.model, component_screening,
                proposal_screening.audit_documents);
            apply_auto_component_screening_resolution(
                particle_screening, enabled);
        }
        const double particle_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - particle_start).count();
        const auto score_start = std::chrono::steady_clock::now();
        ScoreResult out = score_particles(
            set, state.model, particle_screening);
        add_screening_metrics(out, component_screening,
            proposal_screening, particle_screening);
        out.adaptive_particle_options = adaptive_particles;
        out.particle_generation_seconds = particle_seconds;
        out.scoring_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - score_start).count();
        return out;
    }
    const auto particle_start = std::chrono::steady_clock::now();
    const ParticleSet set = make_particle_range(data, basis, state.helmert,
        state.pilot, pilot_cache, proposal, particles, particle_seed,
        state.fisher_broadening, n_threads, &proposal_screening, 0,
        static_cast<int32_t>(data.coordinates.rows()));
    if (component_screening.mode == ComponentScreeningMode::Auto) {
        const bool enabled = resolve_particle_component_screening(
            set, state.model, component_screening,
            proposal_screening.audit_documents);
        apply_auto_component_screening_resolution(
            particle_screening, enabled);
    }
    const double particle_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - particle_start).count();
    const auto score_start = std::chrono::steady_clock::now();
    ScoreResult out = score_particles(set, state.model, particle_screening);
    add_screening_metrics(out, component_screening,
        proposal_screening, particle_screening);
    out.particle_generation_seconds = particle_seconds;
    out.scoring_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - score_start).count();
    return out;
}

ScoreResult score_particle(Dataset& data, const Basis& basis,
    const State& state, const ParticleScoreOptions& options) {
    return score_particle_impl(data, &data, basis, state, options);
}

ScoreResult score_particle(const Dataset& data, const Basis& basis,
    const State& state, const ParticleScoreOptions& options) {
    return score_particle_impl(data, nullptr, basis, state, options);
}

ScoreResult score_particle_indexed(const Dataset& data, const Basis& basis,
    IndexedDocumentSource& source, const State& state,
    const ParticleScoreOptions& options) {
    return score_particle_impl(
        data, nullptr, basis, state, options, &source);
}

State make_state(const FitResult& fit_result, const FitOptions& options,
    const StateMetadata& metadata) {
    validate_model(fit_result.model);
    const int32_t components =
        checked_int32(fit_result.model.weights.size(), "component count");
    const int32_t dimension =
        checked_int32(fit_result.model.means.cols(), "model dimension");
    validate_pilot(fit_result.pilot, components, dimension);
    const Eigen::MatrixXd expected_helmert =
        normalized_helmert(dimension + 1);
    if (metadata.topics.size()
            != static_cast<size_t>(dimension + 1)
        || metadata.helmert.rows() != dimension
        || metadata.helmert.cols() != dimension + 1
        || !metadata.helmert.allFinite()
        || (metadata.helmert - expected_helmert)
            .cwiseAbs().maxCoeff() > 1e-12
        || !(metadata.center_floor > 0.0)
        || !std::isfinite(metadata.center_floor)
        || !metadata.feature_weights.allFinite()
        || (metadata.feature_weights.array() < 0.0).any()
        || (metadata.feature_weights.size() > 0
            && !metadata.weighted_counts
            && !(metadata.feature_weights.array() == 1.0).all())
        || (options.handoff == HandoffMode::Particle
            && metadata.basis_checksum == 0)) {
        throw std::invalid_argument("Invalid UAC state metadata");
    }
    State state;
    state.handoff = options.handoff;
    state.proposal = options.proposal;
    state.n_particles = options.n_particles;
    state.seed = options.seed;
    state.cluster_covariance_rank = fit_result.model.covariance_kind
            == CovarianceKind::FactorAnalytic
        ? static_cast<int32_t>(
            fit_result.model.factor_covariances.front().factor.cols())
        : -1;
    state.kmeans_starts = options.kmeans_starts;
    state.leiden_starts = options.leiden_starts;
    state.kmeans_max_iterations = options.kmeans_max_iterations;
    state.leiden_neighbors = options.leiden_neighbors;
    state.leiden_knn_backend = options.leiden_knn_backend;
    state.leiden_max_iterations = options.leiden_max_iterations;
    state.selected_start = fit_result.selected_start;
    state.selected_start_method = fit_result.selected_start_method;
    state.selected_leiden_resolution =
        fit_result.selected_leiden_resolution;
    state.converged = fit_result.converged;
    state.center_floor = metadata.center_floor;
    state.target_relative_floor = options.target_relative_floor;
    state.leiden_knn_epsilon = options.leiden_knn_epsilon;
    state.leiden_resolution = options.leiden_resolution;
    state.covariance_floor = options.covariance_floor;
    state.objective_change_tolerance = options.objective_change_tolerance;
    state.responsibility_change_tolerance =
        options.responsibility_change_tolerance;
    state.particle_variance_change_tolerance =
        options.particle_variance_change_tolerance;
    state.initialization_ridge_precision =
        options.initialization_ridge_precision;
    state.adaptive_covariance_shrinkage =
        options.adaptive_covariance_shrinkage;
    state.covariance_shrinkage_strength =
        options.covariance_shrinkage_strength;
    state.fisher_broadening = options.fisher_broadening;
    state.fit_adaptive_particles = options.adaptive_particles;
    state.component_screening = options.component_screening;
    state.fit_map_component_screening =
        fit_result.score.map_component_screening;
    state.fit_proposal_component_screening =
        fit_result.score.proposal_component_screening;
    state.fit_particle_component_screening =
        fit_result.score.particle_component_screening;
    state.weighted_counts = metadata.weighted_counts;
    state.feature_weights = metadata.feature_weights;
    if (state.feature_weights.size() > 0
        && (state.feature_weights.array() == 1.0).all()) {
        state.feature_weights.resize(0);
    }
    state.pilot = fit_result.pilot;
    state.model = fit_result.model;
    state.topics = metadata.topics;
    state.basis_checksum = metadata.basis_checksum;
    state.helmert = metadata.helmert;
    validate_state(state);
    return state;
}

void write_state(const std::string& path, const State& state) {
    validate_state(state);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC state: " + path);
    const int32_t components = static_cast<int32_t>(state.model.weights.size());
    const int32_t dimension = static_cast<int32_t>(state.model.means.cols());
    out << "##punkst_uac_state_v10\n"
        << "##handoff\t" << handoff_name(state.handoff) << "\n"
        << "##proposal\t" << proposal_name(state.proposal) << "\n"
        << "##particles\t" << state.n_particles << "\n"
        << "##seed\t" << state.seed << "\n"
        << "##cluster_covariance_rank\t"
        << state.cluster_covariance_rank << "\n"
        << "##kmeans_starts\t" << state.kmeans_starts << "\n"
        << "##leiden_starts\t" << state.leiden_starts << "\n"
        << "##kmeans_max_iterations\t"
        << state.kmeans_max_iterations << "\n"
        << "##leiden_neighbors\t" << state.leiden_neighbors << "\n"
        << "##leiden_knn_backend\t"
        << cosine_knn_backend_name(state.leiden_knn_backend) << "\n"
        << "##leiden_max_iterations\t"
        << state.leiden_max_iterations << "\n"
        << "##selected_start\t" << state.selected_start << "\n"
        << "##selected_start_method\t"
        << start_method_name(state.selected_start_method) << "\n"
        << "##converged\t" << static_cast<int32_t>(state.converged) << "\n"
        << "##components\t" << components << "\n"
        << "##dimension\t" << dimension << "\n"
        << "##basis_checksum\t" << state.basis_checksum << "\n"
        << "##weighted_counts\t" << static_cast<int32_t>(state.weighted_counts) << "\n"
        << "##count_likelihood\t"
        << (state.weighted_counts ? "weighted_multinomial_kernel" : "multinomial")
        << "\n"
        << std::setprecision(17)
        << "##center_floor\t" << state.center_floor << "\n"
        << "##target_relative_floor\t"
        << state.target_relative_floor << "\n"
        << "##leiden_knn_epsilon\t" << state.leiden_knn_epsilon << "\n"
        << "##leiden_resolution\t" << state.leiden_resolution << "\n"
        << "##selected_leiden_resolution\t"
        << state.selected_leiden_resolution << "\n"
        << "##covariance_floor\t" << state.covariance_floor << "\n"
        << "##objective_change_tolerance\t"
        << state.objective_change_tolerance << "\n"
        << "##responsibility_change_tolerance\t"
        << state.responsibility_change_tolerance << "\n"
        << "##particle_variance_change_tolerance\t"
        << state.particle_variance_change_tolerance << "\n"
        << "##initialization_ridge_precision\t"
        << state.initialization_ridge_precision << "\n"
        << "##covariance_shrinkage\t"
        << (state.adaptive_covariance_shrinkage
            ? "adaptive_particle" : "none") << "\n"
        << "##covariance_shrinkage_strength\t"
        << state.covariance_shrinkage_strength << "\n"
        << "##fisher_broadening\t" << state.fisher_broadening << "\n"
        << "##component_screening\t"
        << component_screening_mode_name(
            state.component_screening.mode) << "\n"
        << "##component_tail_mass\t"
        << state.component_screening.tail_mass << "\n"
        << "##proposal_tail_mass\t"
        << state.component_screening.proposal_proxy_tail_mass << "\n"
        << "##component_minimum\t"
        << state.component_screening.minimum_components << "\n"
        << "##component_maximum\t"
        << state.component_screening.maximum_components << "\n"
        << "##component_audit_documents\t"
        << state.component_screening.audit_documents << "\n"
        << "##component_min_work_reduction\t"
        << state.component_screening.minimum_work_reduction << "\n"
        << "##fit_map_component_screening\t"
        << static_cast<int32_t>(state.fit_map_component_screening) << "\n"
        << "##fit_proposal_component_screening\t"
        << static_cast<int32_t>(
            state.fit_proposal_component_screening) << "\n"
        << "##fit_particle_component_screening\t"
        << static_cast<int32_t>(
            state.fit_particle_component_screening) << "\n"
        << "##particle_adapt_mode\t"
        << adaptive_particle_mode_name(state.fit_adaptive_particles) << "\n"
        << "##particle_adapt_resp\t"
        << optional_target_or_zero(
            state.fit_adaptive_particles.responsibility_se_target) << "\n"
        << "##particle_adapt_moment\t"
        << optional_target_or_zero(
            state.fit_adaptive_particles.moment_ess_target) << "\n"
        << "##particle_adapt_calibration\t"
        << state.fit_adaptive_particles.calibration_particles << "\n"
        << "##particle_adapt_min\t"
        << state.fit_adaptive_particles.minimum_particles << "\n"
        << "##particle_adapt_plausible_mass\t"
        << state.fit_adaptive_particles.plausible_mass << "\n"
        << "##particle_adapt_plausible_resp\t"
        << state.fit_adaptive_particles.plausible_responsibility << "\n";
    out << "TOPICS";
    for (const auto& topic : state.topics) out << "\t" << topic;
    out << "\nFEATURE_WEIGHTS";
    for (Eigen::Index i = 0; i < state.feature_weights.size(); ++i) {
        out << "\t" << state.feature_weights(i);
    }
    out << "\nMODEL_WEIGHTS";
    for (Eigen::Index c = 0; c < state.model.weights.size(); ++c) out << "\t" << state.model.weights(c);
    out << "\nPILOT_WEIGHTS";
    for (Eigen::Index c = 0; c < state.pilot.weights.size(); ++c) out << "\t" << state.pilot.weights(c);
    out << "\n";
    for (Eigen::Index r = 0; r < state.helmert.rows(); ++r) {
        out << "HELMERT\t" << r;
        for (Eigen::Index j = 0; j < state.helmert.cols(); ++j) out << "\t" << state.helmert(r, j);
        out << "\n";
    }
    for (int32_t c = 0; c < components; ++c) {
        out << "MODEL_MEAN\t" << c;
        for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.model.means(c, j);
        out << "\nPILOT_MEAN\t" << c;
        for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.pilot.means(c, j);
        out << "\n";
        for (int32_t r = 0; r < dimension; ++r) {
            if (state.cluster_covariance_rank < 0) {
                out << "MODEL_COV\t" << c << "\t" << r;
                for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.model.covariances[c](r, j);
                out << "\n";
            } else {
                out << "MODEL_FACTOR\t" << c << "\t" << r;
                for (int32_t j = 0; j < state.cluster_covariance_rank; ++j) {
                    out << "\t" << state.model.factor_covariances[c].factor(r, j);
                }
                out << "\n";
            }
            out << "PILOT_COV\t" << c << "\t" << r;
            for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.pilot.covariances[c](r, j);
            out << "\n";
        }
    }
    for (int32_t r = 0; r < dimension; ++r) {
        if (state.cluster_covariance_rank < 0) {
            out << "SHRINKAGE_TARGET\t" << r;
            for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.model.shrinkage_target(r, j);
        } else {
            out << "FA_DIAGONALS\t" << r;
            for (int32_t c = 0; c < components; ++c) {
                out << "\t" << state.model.factor_covariances[c].diagonal(r);
            }
            out << "\nFA_TARGET\t" << r << "\t"
                << state.model.factor_shrinkage_target.diagonal(r);
            for (int32_t j = 0; j < state.cluster_covariance_rank; ++j) {
                out << "\t" << state.model.factor_shrinkage_target.factor(r, j);
            }
        }
        out << "\nPILOT_POOLED\t" << r;
        for (int32_t j = 0; j < dimension; ++j) out << "\t" << state.pilot.pooled_covariance(r, j);
        out << "\n";
    }
}

State read_state(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot read UAC state: " + path);
    State state;
    std::string line;
    int32_t components = -1, dimension = -1;
    int32_t state_version = 0;
    bool saw_proposal = false;
    bool saw_fisher_broadening = false;
    bool saw_initialization_ridge_precision = false;
    bool saw_kmeans_starts = false, saw_leiden_starts = false;
    bool saw_selected_start = false, saw_selected_start_method = false;
    bool saw_target_relative_floor = false;
    bool saw_cluster_covariance_rank = false;
    bool saw_objective_change_tolerance = false;
    bool saw_responsibility_change_tolerance = false;
    bool saw_particle_variance_change_tolerance = false;
    bool saw_covariance_shrinkage = false;
    bool saw_covariance_shrinkage_strength = false;
    bool saw_particle_adapt_mode = false, saw_particle_adapt_resp = false;
    bool saw_particle_adapt_moment = false;
    bool saw_particle_adapt_calibration = false;
    bool saw_particle_adapt_min = false;
    bool saw_particle_adapt_plausible_mass = false;
    bool saw_particle_adapt_plausible_resp = false;
    std::unordered_map<std::string, int32_t> metadata_count;
    std::string count_likelihood;
    std::string particle_adapt_mode;
    std::vector<std::vector<std::string>> records;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        std::vector<std::string> token = fields(line);
        if (token.empty()) continue;
        if (state_version == 0
            && token[0] != "##punkst_uac_state_v10") {
            throw std::runtime_error(
                "UAC state must begin with the v10 header");
        }
        if (token[0] == "##punkst_uac_state_v10") {
            if (state_version != 0) {
                throw std::runtime_error("Duplicate UAC state version");
            }
            state_version = 10;
            continue;
        }
        if (token[0].rfind("##punkst_uac_state_v", 0) == 0) {
            throw std::runtime_error(
                "Unsupported UAC state version; only v10 is accepted");
        }
        if (token[0].rfind("##", 0) == 0) {
            if (token.size() != 2) throw std::runtime_error("Malformed UAC state metadata");
            const std::string key = token[0].substr(2);
            if (++metadata_count[key] != 1) {
                throw std::runtime_error(
                    "Duplicate UAC state metadata: " + key);
            }
            if (key == "handoff") state.handoff = parse_handoff(token[1]);
            else if (key == "proposal") {
                state.proposal = parse_proposal(token[1]);
                saw_proposal = true;
            }
            else if (key == "particles") {
                state.n_particles = parse_state_int32(token[1]);
            }
            else if (key == "seed") {
                state.seed = parse_state_int32(token[1]);
            }
            else if (key == "cluster_covariance_rank") {
                state.cluster_covariance_rank =
                    parse_state_int32(token[1]);
                saw_cluster_covariance_rank = true;
            }
            else if (key == "kmeans_starts") {
                state.kmeans_starts = parse_state_int32(token[1]);
                saw_kmeans_starts = true;
            }
            else if (key == "leiden_starts") {
                state.leiden_starts = parse_state_int32(token[1]);
                saw_leiden_starts = true;
            }
            else if (key == "kmeans_max_iterations") {
                state.kmeans_max_iterations =
                    parse_state_int32(token[1]);
            }
            else if (key == "leiden_neighbors") {
                state.leiden_neighbors = parse_state_int32(token[1]);
            }
            else if (key == "leiden_knn_backend") {
                state.leiden_knn_backend = parse_cosine_knn_backend(token[1]);
            }
            else if (key == "leiden_max_iterations") {
                state.leiden_max_iterations =
                    parse_state_int32(token[1]);
            }
            else if (key == "selected_start") {
                state.selected_start = parse_state_int32(token[1]);
                saw_selected_start = true;
            }
            else if (key == "selected_start_method") {
                state.selected_start_method =
                    parse_start_method(token[1]);
                saw_selected_start_method = true;
            }
            else if (key == "converged") {
                state.converged = parse_state_bool(token[1]);
            }
            else if (key == "components") {
                components = parse_state_int32(token[1]);
            }
            else if (key == "dimension") {
                dimension = parse_state_int32(token[1]);
            }
            else if (key == "basis_checksum") {
                state.basis_checksum = parse_state_uint64(token[1]);
            }
            else if (key == "weighted_counts") {
                state.weighted_counts = parse_state_bool(token[1]);
            }
            else if (key == "count_likelihood") count_likelihood = token[1];
            else if (key == "center_floor") {
                state.center_floor = parse_state_double(token[1]);
            }
            else if (key == "target_relative_floor") {
                state.target_relative_floor =
                    parse_state_double(token[1]);
                saw_target_relative_floor = true;
            }
            else if (key == "leiden_knn_epsilon") {
                state.leiden_knn_epsilon =
                    parse_state_double(token[1]);
            }
            else if (key == "leiden_resolution") {
                state.leiden_resolution =
                    parse_state_double(token[1]);
            }
            else if (key == "selected_leiden_resolution") {
                state.selected_leiden_resolution =
                    parse_state_double(token[1]);
            }
            else if (key == "covariance_floor") {
                state.covariance_floor = parse_state_double(token[1]);
            }
            else if (key == "objective_change_tolerance") {
                state.objective_change_tolerance =
                    parse_state_double(token[1]);
                saw_objective_change_tolerance = true;
            }
            else if (key == "responsibility_change_tolerance") {
                state.responsibility_change_tolerance =
                    parse_state_double(token[1]);
                saw_responsibility_change_tolerance = true;
            }
            else if (key == "particle_variance_change_tolerance") {
                state.particle_variance_change_tolerance =
                    parse_state_double(token[1]);
                saw_particle_variance_change_tolerance = true;
            }
            else if (key == "initialization_ridge_precision") {
                state.initialization_ridge_precision =
                    parse_state_double(token[1]);
                saw_initialization_ridge_precision = true;
            }
            else if (key == "covariance_shrinkage") {
                if (token[1] == "adaptive_particle") {
                    state.adaptive_covariance_shrinkage = true;
                } else if (token[1] == "none") {
                    state.adaptive_covariance_shrinkage = false;
                } else {
                    throw std::runtime_error(
                        "Unsupported UAC covariance shrinkage mode");
                }
                saw_covariance_shrinkage = true;
            }
            else if (key == "covariance_shrinkage_strength") {
                state.covariance_shrinkage_strength =
                    parse_state_double(token[1]);
                if (!(state.covariance_shrinkage_strength >= 0.0)
                    || !std::isfinite(
                        state.covariance_shrinkage_strength)) {
                    throw std::runtime_error(
                        "Invalid UAC covariance shrinkage strength");
                }
                saw_covariance_shrinkage_strength = true;
            }
            else if (key == "fisher_broadening") {
                state.fisher_broadening = parse_state_double(token[1]);
                saw_fisher_broadening = true;
            }
            else if (key == "component_screening") {
                state.component_screening.mode =
                    parse_component_screening_mode(token[1]);
            }
            else if (key == "component_tail_mass") {
                state.component_screening.tail_mass =
                    parse_state_double(token[1]);
            }
            else if (key == "proposal_tail_mass") {
                state.component_screening.proposal_proxy_tail_mass =
                    parse_state_double(token[1]);
            }
            else if (key == "component_minimum") {
                state.component_screening.minimum_components =
                    parse_state_int32(token[1]);
            }
            else if (key == "component_maximum") {
                state.component_screening.maximum_components =
                    parse_state_int32(token[1]);
            }
            else if (key == "component_audit_documents") {
                state.component_screening.audit_documents =
                    parse_state_int32(token[1]);
            }
            else if (key == "component_min_work_reduction") {
                state.component_screening.minimum_work_reduction =
                    parse_state_double(token[1]);
            }
            else if (key == "fit_map_component_screening") {
                state.fit_map_component_screening =
                    parse_state_bool(token[1]);
            }
            else if (key == "fit_proposal_component_screening") {
                state.fit_proposal_component_screening =
                    parse_state_bool(token[1]);
            }
            else if (key == "fit_particle_component_screening") {
                state.fit_particle_component_screening =
                    parse_state_bool(token[1]);
            }
            else if (key == "particle_adapt_mode") {
                particle_adapt_mode = token[1];
                if (particle_adapt_mode != "fixed"
                    && particle_adapt_mode != "responsibility"
                    && particle_adapt_mode != "moment"
                    && particle_adapt_mode != "responsibility_moment") {
                    throw std::runtime_error(
                        "Unknown UAC adaptive particle mode");
                }
                saw_particle_adapt_mode = true;
            }
            else if (key == "particle_adapt_resp") {
                state.fit_adaptive_particles.responsibility_se_target =
                    parse_state_double(token[1]);
                saw_particle_adapt_resp = true;
            }
            else if (key == "particle_adapt_moment") {
                state.fit_adaptive_particles.moment_ess_target =
                    parse_state_double(token[1]);
                saw_particle_adapt_moment = true;
            }
            else if (key == "particle_adapt_calibration") {
                state.fit_adaptive_particles.calibration_particles =
                    parse_state_int32(token[1]);
                saw_particle_adapt_calibration = true;
            }
            else if (key == "particle_adapt_min") {
                state.fit_adaptive_particles.minimum_particles =
                    parse_state_int32(token[1]);
                saw_particle_adapt_min = true;
            }
            else if (key == "particle_adapt_plausible_mass") {
                state.fit_adaptive_particles.plausible_mass =
                    parse_state_double(token[1]);
                saw_particle_adapt_plausible_mass = true;
            }
            else if (key == "particle_adapt_plausible_resp") {
                state.fit_adaptive_particles.plausible_responsibility =
                    parse_state_double(token[1]);
                saw_particle_adapt_plausible_resp = true;
            }
            else {
                throw std::runtime_error(
                    "Unknown UAC state metadata: " + key);
            }
            continue;
        }
        records.push_back(std::move(token));
    }
    if (state_version == 0 || !saw_proposal || !saw_fisher_broadening
        || !saw_kmeans_starts
        || !saw_leiden_starts || !saw_selected_start
        || !saw_selected_start_method || !saw_target_relative_floor
        || !saw_cluster_covariance_rank
        || !saw_objective_change_tolerance
        || !saw_responsibility_change_tolerance
        || !saw_particle_variance_change_tolerance
        || !saw_initialization_ridge_precision
        || !saw_covariance_shrinkage
        || !saw_covariance_shrinkage_strength
        || !saw_particle_adapt_mode || !saw_particle_adapt_resp
        || !saw_particle_adapt_moment || !saw_particle_adapt_calibration
        || !saw_particle_adapt_min || !saw_particle_adapt_plausible_mass
        || !saw_particle_adapt_plausible_resp
        || components <= 0 || dimension <= 0) {
        throw std::runtime_error("Invalid, stale, or unsupported UAC state");
    }
    std::vector<std::string> required_metadata = {
        "handoff", "initialization_ridge_precision", "proposal",
        "particles", "seed",
        "cluster_covariance_rank", "kmeans_starts", "leiden_starts",
        "kmeans_max_iterations", "leiden_neighbors", "leiden_knn_backend",
        "leiden_max_iterations", "selected_start",
        "selected_start_method",
        "converged", "components", "dimension", "basis_checksum",
        "weighted_counts", "count_likelihood", "center_floor",
        "target_relative_floor", "leiden_knn_epsilon", "leiden_resolution",
        "selected_leiden_resolution", "covariance_floor",
        "objective_change_tolerance", "responsibility_change_tolerance",
        "covariance_shrinkage", "covariance_shrinkage_strength",
        "fisher_broadening",
        "component_screening", "component_tail_mass",
        "proposal_tail_mass", "component_minimum", "component_maximum",
        "component_audit_documents", "component_min_work_reduction",
        "fit_map_component_screening",
        "fit_proposal_component_screening",
        "fit_particle_component_screening", "particle_adapt_mode",
        "particle_adapt_resp", "particle_adapt_moment",
        "particle_adapt_calibration", "particle_adapt_min",
        "particle_adapt_plausible_mass", "particle_adapt_plausible_resp",
    };
    required_metadata.push_back(
        "particle_variance_change_tolerance");
    for (const auto& key : required_metadata) {
        if (metadata_count.find(key) == metadata_count.end()) {
            throw std::runtime_error(
                "Missing UAC state metadata: " + key);
        }
    }
    if (particle_adapt_mode == "fixed") {
        state.fit_adaptive_particles.responsibility_se_target.reset();
        state.fit_adaptive_particles.moment_ess_target.reset();
    } else if (particle_adapt_mode == "responsibility") {
        state.fit_adaptive_particles.moment_ess_target.reset();
    } else if (particle_adapt_mode == "moment") {
        state.fit_adaptive_particles.responsibility_se_target.reset();
    }
    const std::string expected_likelihood = state.weighted_counts
        ? "weighted_multinomial_kernel" : "multinomial";
    if (count_likelihood != expected_likelihood) {
        throw std::runtime_error("Inconsistent UAC count likelihood metadata");
    }
    state.helmert = Eigen::MatrixXd::Zero(dimension, dimension + 1);
    state.model.weights = Eigen::VectorXd::Zero(components);
    state.model.covariance_kind = state.cluster_covariance_rank < 0
        ? CovarianceKind::Dense : CovarianceKind::FactorAnalytic;
    state.model.means = RowMajorMatrixXd::Zero(components, dimension);
    state.model.covariances.assign(components, Eigen::MatrixXd::Zero(dimension, dimension));
    state.model.shrinkage_target = Eigen::MatrixXd::Zero(dimension, dimension);
    if (state.cluster_covariance_rank >= 0) {
        if (state.cluster_covariance_rank > dimension) {
            throw std::runtime_error("Invalid UAC state factor rank");
        }
        state.model.factor_covariances.resize(components);
        for (auto& covariance : state.model.factor_covariances) {
            covariance.diagonal = Eigen::VectorXd::Zero(dimension);
            covariance.factor = RowMajorMatrixXd::Zero(
                dimension, state.cluster_covariance_rank);
        }
        state.model.factor_shrinkage_target.diagonal =
            Eigen::VectorXd::Zero(dimension);
        state.model.factor_shrinkage_target.factor = RowMajorMatrixXd::Zero(
            dimension, state.cluster_covariance_rank);
    }
    state.pilot.weights = Eigen::VectorXd::Zero(components);
    state.pilot.means = RowMajorMatrixXd::Zero(components, dimension);
    state.pilot.covariances.assign(components, Eigen::MatrixXd::Zero(dimension, dimension));
    state.pilot.pooled_covariance = Eigen::MatrixXd::Zero(dimension, dimension);
    bool saw_topics = false, saw_feature_weights = false;
    bool saw_model_weights = false, saw_pilot_weights = false;
    std::vector<uint8_t> saw_helmert(dimension, 0);
    std::vector<uint8_t> saw_model_mean(components, 0);
    std::vector<uint8_t> saw_pilot_mean(components, 0);
    std::vector<uint8_t> saw_model_cov(
        static_cast<size_t>(components) * dimension, 0);
    std::vector<uint8_t> saw_model_factor(
        static_cast<size_t>(components) * dimension, 0);
    std::vector<uint8_t> saw_pilot_cov(
        static_cast<size_t>(components) * dimension, 0);
    std::vector<uint8_t> saw_shrinkage_target(dimension, 0);
    std::vector<uint8_t> saw_fa_diagonals(dimension, 0);
    std::vector<uint8_t> saw_fa_target(dimension, 0);
    std::vector<uint8_t> saw_pilot_pooled(dimension, 0);
    auto check_index = [](int32_t value, int32_t size,
                           const char* name) {
        if (value < 0 || value >= size) {
            throw std::runtime_error(
                std::string("UAC state ") + name + " index is out of range");
        }
    };
    auto mark = [](uint8_t& seen, const char* name) {
        if (seen) {
            throw std::runtime_error(
                std::string("Duplicate UAC state ") + name + " record");
        }
        seen = 1;
    };
    for (const auto& token : records) {
        auto values = [&](size_t offset, Eigen::Ref<Eigen::VectorXd> target) {
            if (token.size() != offset + static_cast<size_t>(target.size())) throw std::runtime_error("Malformed UAC state row");
            for (Eigen::Index j = 0; j < target.size(); ++j) {
                target(j) = parse_state_double(token[offset + j]);
            }
        };
        if (token[0] == "TOPICS") {
            if (saw_topics) {
                throw std::runtime_error("Duplicate UAC state TOPICS record");
            }
            saw_topics = true;
            state.topics.assign(token.begin() + 1, token.end());
        }
        else if (token[0] == "FEATURE_WEIGHTS") {
            if (saw_feature_weights) {
                throw std::runtime_error(
                    "Duplicate UAC state FEATURE_WEIGHTS record");
            }
            saw_feature_weights = true;
            state.feature_weights.resize(token.size() - 1);
            for (size_t j = 1; j < token.size(); ++j) {
                state.feature_weights(j - 1) =
                    parse_state_double(token[j]);
            }
        } else if (token[0] == "MODEL_WEIGHTS") {
            if (saw_model_weights) {
                throw std::runtime_error(
                    "Duplicate UAC state MODEL_WEIGHTS record");
            }
            saw_model_weights = true;
            values(1, state.model.weights);
        } else if (token[0] == "PILOT_WEIGHTS") {
            if (saw_pilot_weights) {
                throw std::runtime_error(
                    "Duplicate UAC state PILOT_WEIGHTS record");
            }
            saw_pilot_weights = true;
            values(1, state.pilot.weights);
        }
        else if (token[0] == "HELMERT") {
            if (token.size() < 2) {
                throw std::runtime_error("Malformed UAC state HELMERT row");
            }
            const int32_t row = parse_state_int32(token[1]);
            check_index(row, dimension, "HELMERT");
            mark(saw_helmert[row], "HELMERT");
            Eigen::VectorXd target(dimension + 1);
            values(2, target);
            state.helmert.row(row) = target.transpose();
        } else if (token[0] == "MODEL_MEAN" || token[0] == "PILOT_MEAN") {
            if (token.size() < 2) {
                throw std::runtime_error("Malformed UAC state mean row");
            }
            const int32_t c = parse_state_int32(token[1]);
            check_index(c, components, "mean component");
            Eigen::VectorXd target(dimension);
            values(2, target);
            if (token[0] == "MODEL_MEAN") {
                mark(saw_model_mean[c], "MODEL_MEAN");
                state.model.means.row(c) = target.transpose();
            } else {
                mark(saw_pilot_mean[c], "PILOT_MEAN");
                state.pilot.means.row(c) = target.transpose();
            }
        } else if (token[0] == "MODEL_FACTOR") {
            if (state.cluster_covariance_rank < 0 || token.size() < 3) {
                throw std::runtime_error("Unexpected UAC MODEL_FACTOR row");
            }
            const int32_t c = parse_state_int32(token[1]);
            const int32_t row = parse_state_int32(token[2]);
            check_index(c, components, "MODEL_FACTOR component");
            check_index(row, dimension, "MODEL_FACTOR row");
            mark(saw_model_factor[
                static_cast<size_t>(c) * dimension + row], "MODEL_FACTOR");
            Eigen::VectorXd target(state.cluster_covariance_rank);
            values(3, target);
            state.model.factor_covariances[c].factor.row(row) =
                target.transpose();
        } else if (token[0] == "FA_DIAGONALS") {
            if (state.cluster_covariance_rank < 0 || token.size() < 2) {
                throw std::runtime_error("Unexpected UAC FA_DIAGONALS row");
            }
            const int32_t row = parse_state_int32(token[1]);
            check_index(row, dimension, "FA_DIAGONALS");
            mark(saw_fa_diagonals[row], "FA_DIAGONALS");
            Eigen::VectorXd target(components);
            values(2, target);
            for (int32_t c = 0; c < components; ++c) {
                state.model.factor_covariances[c].diagonal(row) = target(c);
            }
        } else if (token[0] == "FA_TARGET") {
            if (state.cluster_covariance_rank < 0 || token.size() < 2) {
                throw std::runtime_error("Unexpected UAC FA_TARGET row");
            }
            const int32_t row = parse_state_int32(token[1]);
            check_index(row, dimension, "FA_TARGET");
            mark(saw_fa_target[row], "FA_TARGET");
            Eigen::VectorXd target(state.cluster_covariance_rank + 1);
            values(2, target);
            state.model.factor_shrinkage_target.diagonal(row) = target(0);
            if (state.cluster_covariance_rank > 0) {
                state.model.factor_shrinkage_target.factor.row(row) =
                    target.tail(state.cluster_covariance_rank).transpose();
            }
        } else if (token[0] == "MODEL_COV"
                || token[0] == "PILOT_COV") {
            if (token.size() < 3
                || (token[0] == "MODEL_COV"
                    && state.cluster_covariance_rank >= 0)) {
                throw std::runtime_error("Unexpected UAC covariance row");
            }
            const int32_t c = parse_state_int32(token[1]);
            const int32_t row = parse_state_int32(token[2]);
            check_index(c, components, "covariance component");
            check_index(row, dimension, "covariance row");
            Eigen::VectorXd target(dimension);
            values(3, target);
            const size_t index = static_cast<size_t>(c) * dimension + row;
            if (token[0] == "MODEL_COV") {
                mark(saw_model_cov[index], "MODEL_COV");
                state.model.covariances[c].row(row) = target.transpose();
            } else {
                mark(saw_pilot_cov[index], "PILOT_COV");
                state.pilot.covariances[c].row(row) = target.transpose();
            }
        } else if (token[0] == "SHRINKAGE_TARGET" || token[0] == "PILOT_POOLED") {
            if (token.size() < 2
                || (token[0] == "SHRINKAGE_TARGET"
                    && state.cluster_covariance_rank >= 0)) {
                throw std::runtime_error(
                    "Unexpected UAC target covariance row");
            }
            const int32_t row = parse_state_int32(token[1]);
            check_index(row, dimension, "target covariance row");
            Eigen::VectorXd target(dimension);
            values(2, target);
            if (token[0] == "SHRINKAGE_TARGET") {
                mark(saw_shrinkage_target[row], "SHRINKAGE_TARGET");
                state.model.shrinkage_target.row(row) = target.transpose();
            } else {
                mark(saw_pilot_pooled[row], "PILOT_POOLED");
                state.pilot.pooled_covariance.row(row) = target.transpose();
            }
        } else {
            throw std::runtime_error(
                "Unknown UAC state record: " + token[0]);
        }
    }
    auto all_seen = [](const std::vector<uint8_t>& seen) {
        return std::all_of(seen.begin(), seen.end(),
            [](uint8_t value) { return value != 0; });
    };
    const bool common_records_complete = saw_topics && saw_feature_weights
        && saw_model_weights && saw_pilot_weights
        && all_seen(saw_helmert) && all_seen(saw_model_mean)
        && all_seen(saw_pilot_mean) && all_seen(saw_pilot_cov)
        && all_seen(saw_pilot_pooled);
    const bool covariance_records_complete =
        state.cluster_covariance_rank < 0
        ? all_seen(saw_model_cov) && all_seen(saw_shrinkage_target)
        : all_seen(saw_model_factor) && all_seen(saw_fa_diagonals)
            && all_seen(saw_fa_target);
    if (!common_records_complete || !covariance_records_complete) {
        throw std::runtime_error("Incomplete UAC state records");
    }
    try {
        validate_state(state);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Incomplete UAC state");
    }
    return state;
}

} // namespace uac
