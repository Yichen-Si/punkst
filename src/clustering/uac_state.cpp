
#include "clustering/uac_common_internal.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

namespace uac {
namespace {


std::vector<std::string> fields(const std::string& line) {
    std::vector<std::string> out;
    std::string token;
    std::istringstream input(line);
    while (input >> token) out.push_back(token);
    return out;
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

const char* state_adaptive_particle_mode_name(
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

} // namespace


State make_state(const FitResult& fit_result, const FitOptions& options,
    const StateMetadata& metadata) {
    detail::validate_model(fit_result.model);
    const int32_t components =
        detail::checked_int32(fit_result.model.weights.size(), "component count");
    const int32_t dimension =
        detail::checked_int32(fit_result.model.means.cols(), "model dimension");
    detail::validate_pilot(fit_result.pilot, components, dimension);
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
    detail::validate_state(state);
    return state;
}

void write_state(const std::string& path, const State& state) {
    detail::validate_state(state);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write UAC state: " + path);
    const int32_t components = static_cast<int32_t>(state.model.weights.size());
    const int32_t dimension = static_cast<int32_t>(state.model.means.cols());
    out << "##punkst_uac_state_v11\n"
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
        << "##canonical_basis_checksum\t" << state.basis_checksum << "\n"
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
        << state_adaptive_particle_mode_name(
            state.fit_adaptive_particles) << "\n"
        << "##particle_adapt_resp\t"
        << detail::optional_target_or_zero(
            state.fit_adaptive_particles.responsibility_se_target) << "\n"
        << "##particle_adapt_moment\t"
        << detail::optional_target_or_zero(
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
            && token[0] != "##punkst_uac_state_v11") {
            throw std::runtime_error(
                "UAC state must begin with the v11 header");
        }
        if (token[0] == "##punkst_uac_state_v11") {
            if (state_version != 0) {
                throw std::runtime_error("Duplicate UAC state version");
            }
            state_version = 11;
            continue;
        }
        if (token[0].rfind("##punkst_uac_state_v", 0) == 0) {
            throw std::runtime_error(
                "Unsupported UAC state version; only v11 is accepted");
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
            else if (key == "canonical_basis_checksum") {
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
        "converged", "components", "dimension",
        "canonical_basis_checksum",
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
        detail::validate_state(state);
    } catch (const std::invalid_argument&) {
        throw std::runtime_error("Incomplete UAC state");
    }
    return state;
}

} // namespace uac
