#include "gamma_pois_cluster.hpp"
#include "dense_kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include <Eigen/Eigenvalues>

Eigen::MatrixXd GammaPoissonTopicCovariance::dense() const {
    if (diagonal.size() == 0) {
        return Eigen::MatrixXd();
    }
    Eigen::MatrixXd out = diagonal.asDiagonal();
    if (factor.cols() > 0) {
        if (factor.rows() != diagonal.size()) {
            throw std::invalid_argument("Gamma-Poisson covariance dimensions do not match");
        }
        out.noalias() += factor * factor.transpose();
    }
    return out;
}

Eigen::MatrixXd GammaPoissonTopicCovariance::transformed(
    const Eigen::Ref<const Eigen::MatrixXd>& basis) const {
    if (basis.cols() != diagonal.size()) {
        throw std::invalid_argument("Gamma-Poisson covariance basis has wrong dimension");
    }
    Eigen::MatrixXd out = (basis.array().rowwise() * diagonal.transpose().array()).matrix()
        * basis.transpose();
    if (factor.cols() > 0) {
        Eigen::MatrixXd projected = basis * factor;
        out.noalias() += projected * projected.transpose();
    }
    return out;
}

GammaPoissonLogMoments gamma_poisson_log_moments(
    const Eigen::Ref<const Eigen::VectorXd>& shape,
    const Eigen::Ref<const Eigen::VectorXd>& rate,
    const Eigen::Ref<const Eigen::VectorXd>& topic_capacity) {
    if (shape.size() == 0 || rate.size() != shape.size()
        || topic_capacity.size() != shape.size()) {
        throw std::invalid_argument("Gamma-Poisson posterior moment dimensions do not match");
    }
    GammaPoissonLogMoments out;
    out.mean.resize(shape.size());
    out.variance.resize(shape.size());
    for (Eigen::Index k = 0; k < shape.size(); ++k) {
        if (!(shape(k) > 0.0) || !(rate(k) > 0.0)
            || !(topic_capacity(k) > 0.0)
            || !std::isfinite(shape(k)) || !std::isfinite(rate(k))
            || !std::isfinite(topic_capacity(k))) {
            throw std::invalid_argument("Gamma-Poisson posterior moments require finite positive inputs");
        }
        out.mean(k) = psi(shape(k)) - std::log(rate(k)) + std::log(topic_capacity(k));
        out.variance(k) = trigamma(shape(k));
    }
    return out;
}

Eigen::MatrixXd normalized_helmert_basis(int32_t n_topics) {
    if (n_topics < 2) {
        throw std::invalid_argument("Helmert basis requires at least two topics");
    }
    Eigen::MatrixXd out = Eigen::MatrixXd::Zero(n_topics - 1, n_topics);
    for (int32_t j = 1; j < n_topics; ++j) {
        const double denom = std::sqrt(static_cast<double>(j) * (j + 1));
        for (int32_t r = 0; r < j; ++r) {
            out(j - 1, r) = 1.0 / denom;
        }
        out(j - 1, j) = -static_cast<double>(j) / denom;
    }
    return out;
}

GammaPoissonBasisAccumulator::GammaPoissonBasisAccumulator(int32_t n_topics)
    : n_topics_(n_topics), dim_(n_topics - 1),
      helmert_(normalized_helmert_basis(n_topics)),
      sum_u_(Eigen::VectorXd::Zero(dim_)),
      sum_uu_(Eigen::MatrixXd::Zero(dim_, dim_)),
      sum_uncertainty_(Eigen::MatrixXd::Zero(dim_, dim_)) {}

void GammaPoissonBasisAccumulator::add(
    const Eigen::Ref<const Eigen::VectorXd>& log_mean,
    const GammaPoissonTopicCovariance& covariance) {
    if (log_mean.size() != n_topics_ || covariance.diagonal.size() != n_topics_) {
        throw std::invalid_argument("Gamma-Poisson basis accumulator dimensions do not match");
    }
    Eigen::VectorXd u = helmert_ * log_mean;
    sum_u_ += u;
    sum_uu_.noalias() += u * u.transpose();
    sum_uncertainty_ += covariance.transformed(helmert_);
    ++count_;
}

GammaPoissonLogRatioBasis GammaPoissonBasisAccumulator::finish() const {
    if (count_ == 0) {
        throw std::runtime_error("Cannot construct a Gamma-Poisson basis without documents");
    }
    GammaPoissonLogRatioBasis out;
    out.helmert = helmert_;
    out.mean_helmert = sum_u_ / static_cast<double>(count_);
    out.observed_covariance = sum_uu_ / static_cast<double>(count_)
        - out.mean_helmert * out.mean_helmert.transpose();
    out.mean_uncertainty = sum_uncertainty_ / static_cast<double>(count_);
    Eigen::MatrixXd raw_signal = 0.5 * (
        out.observed_covariance - out.mean_uncertainty
        + (out.observed_covariance - out.mean_uncertainty).transpose());

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(raw_signal);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("Gamma-Poisson signal covariance eigendecomposition failed");
    }
    std::vector<int32_t> order(dim_);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return solver.eigenvalues()(a) > solver.eigenvalues()(b);
    });

    out.rotation.resize(dim_, dim_);
    out.signal_eigenvalues.resize(dim_);
    for (int32_t j = 0; j < dim_; ++j) {
        Eigen::VectorXd q = solver.eigenvectors().col(order[j]);
        Eigen::Index pivot = 0;
        q.cwiseAbs().maxCoeff(&pivot);
        if (q(pivot) < 0.0) {
            q = -q;
        }
        out.rotation.col(j) = q;
        out.signal_eigenvalues(j) = std::max(0.0, solver.eigenvalues()(order[j]));
    }
    out.signal_covariance = out.rotation * out.signal_eigenvalues.asDiagonal()
        * out.rotation.transpose();
    out.basis = out.rotation.transpose() * helmert_;
    return out;
}

GammaPoissonClusterDataset load_gamma_poisson_cluster_dataset(
    const std::string& posterior_path, const std::string& dispersion_path,
    uint64_t expected_state_checksum,
    const Eigen::Ref<const Eigen::VectorXd>& topic_capacity,
    const std::vector<std::string>& expected_topic_names) {
    GammaPoissonPosteriorReader posterior_reader(
        posterior_path, expected_state_checksum);
    const auto& header = posterior_reader.header();
    if (topic_capacity.size() != header.n_topics
        || expected_topic_names != header.topic_names) {
        throw std::invalid_argument("Topic capacity does not match posterior topic count");
    }

    std::unique_ptr<GammaPoissonDispersionReader> dispersion_reader;
    if (!dispersion_path.empty()) {
        dispersion_reader = std::make_unique<GammaPoissonDispersionReader>(
            dispersion_path, header.state_checksum);
        if (dispersion_reader->header().n_topics != header.n_topics) {
            throw std::runtime_error("Dispersion sidecar topic count does not match posterior");
        }
        if (header.dispersion_sidecar_id.empty()
            || dispersion_reader->header().artifact_id
                != header.dispersion_sidecar_id) {
            throw std::runtime_error(
                "Dispersion sidecar ID does not match posterior");
        }
    }

    GammaPoissonClusterDataset out;
    out.posterior_header = header;
    if (dispersion_reader) {
        out.uncertainty_model = dispersion_reader->header().rank == 0
            ? GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionDiagonal
            : GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionLowRank;
    }
    std::vector<Eigen::VectorXd> means;
    GammaPoissonPosteriorRow row;
    while (posterior_reader.read_next(row)) {
        GammaPoissonLogMoments moments = gamma_poisson_log_moments(
            row.posterior.shape, row.posterior.rate, topic_capacity);
        GammaPoissonTopicCovariance covariance;
        covariance.diagonal = std::move(moments.variance);
        covariance.factor.resize(header.n_topics, 0);
        if (dispersion_reader) {
            GammaPoissonDispersionApproximation approximation;
            if (!dispersion_reader->read_next(approximation)) {
                throw std::runtime_error("Dispersion sidecar has fewer rows than posterior");
            }
            covariance.diagonal += approximation.residual_diagonal;
            covariance.factor = std::move(approximation.factor);
        }
        out.identifiers.push_back(row.identifiers);
        means.push_back(std::move(moments.mean));
        out.topic_covariance.push_back(std::move(covariance));
    }
    if (dispersion_reader) {
        GammaPoissonDispersionApproximation extra;
        if (dispersion_reader->read_next(extra)) {
            throw std::runtime_error("Dispersion sidecar has more rows than posterior");
        }
        if (dispersion_reader->header().record_count != means.size()) {
            throw std::runtime_error("Dispersion sidecar record count does not match posterior");
        }
    }
    if (means.empty()) {
        throw std::runtime_error("Gamma-Poisson posterior contains no documents");
    }
    out.log_mean.resize(means.size(), header.n_topics);
    for (size_t d = 0; d < means.size(); ++d) {
        out.log_mean.row(d) = means[d].transpose();
    }
    return out;
}

GammaPoissonClusterCoordinates make_gamma_poisson_cluster_coordinates(
    const GammaPoissonClusterDataset& dataset) {
    if (dataset.log_mean.rows() == 0
        || dataset.topic_covariance.size()
            != static_cast<size_t>(dataset.log_mean.rows())) {
        throw std::invalid_argument("Invalid Gamma-Poisson clustering dataset");
    }
    const int32_t n_topics = static_cast<int32_t>(dataset.log_mean.cols());
    GammaPoissonBasisAccumulator accumulator(n_topics);
    for (Eigen::Index d = 0; d < dataset.log_mean.rows(); ++d) {
        accumulator.add(dataset.log_mean.row(d).transpose(),
            dataset.topic_covariance[d]);
    }

    GammaPoissonLogRatioBasis log_ratio_basis = accumulator.finish();
    GammaPoissonClusterCoordinates out = make_gamma_poisson_cluster_coordinates(
        dataset, log_ratio_basis.basis);
    out.log_ratio_basis = std::move(log_ratio_basis);
    return out;
}

GammaPoissonClusterCoordinates make_gamma_poisson_cluster_coordinates(
    const GammaPoissonClusterDataset& dataset,
    const Eigen::Ref<const Eigen::MatrixXd>& basis) {
    if (dataset.log_mean.rows() == 0
        || dataset.topic_covariance.size()
            != static_cast<size_t>(dataset.log_mean.rows())) {
        throw std::invalid_argument("Invalid Gamma-Poisson clustering dataset");
    }
    const int32_t n_topics = static_cast<int32_t>(dataset.log_mean.cols());
    if (basis.rows() != n_topics - 1 || basis.cols() != n_topics
        || !basis.allFinite()
        || (basis * basis.transpose() - Eigen::MatrixXd::Identity(
                n_topics - 1, n_topics - 1)).norm() > 1e-8
        || (basis * Eigen::VectorXd::Ones(n_topics)).norm() > 1e-8) {
        throw std::invalid_argument("Invalid fixed Gamma-Poisson log-ratio basis");
    }

    GammaPoissonClusterCoordinates out;
    out.log_ratio_basis.basis = basis;
    out.mean = dataset.log_mean * out.log_ratio_basis.basis.transpose();
    out.uncertainty_diagonal.resize(dataset.log_mean.rows(), n_topics - 1);
    out.uncertainty_rank = dataset.topic_covariance.front().factor.cols();
    out.uncertainty_factor.resize(dataset.log_mean.rows(),
        (n_topics - 1) * out.uncertainty_rank);
    for (Eigen::Index d = 0; d < dataset.log_mean.rows(); ++d) {
        const auto& covariance = dataset.topic_covariance[d];
        if (covariance.factor.cols() != out.uncertainty_rank) {
            throw std::invalid_argument("Document uncertainty factors have inconsistent ranks");
        }
        out.uncertainty_diagonal.row(d) = (
            out.log_ratio_basis.basis.array().square().matrix()
            * covariance.diagonal).transpose();
        if (out.uncertainty_rank > 0) {
            Eigen::Map<RowMajorMatrixXd> factor(
                out.uncertainty_factor.row(d).data(), n_topics - 1,
                out.uncertainty_rank);
            factor = out.log_ratio_basis.basis * covariance.factor;
        }
    }
    return out;
}

Eigen::Map<const RowMajorMatrixXd> GammaPoissonClusterCoordinates::factor(
    Eigen::Index document) const {
    if (document < 0 || document >= mean.rows()) {
        throw std::out_of_range("Document uncertainty factor index is out of range");
    }
    return Eigen::Map<const RowMajorMatrixXd>(
        uncertainty_factor.row(document).data(), mean.cols(), uncertainty_rank);
}

namespace {

std::vector<std::string> split_cluster_tabs(const std::string& line) {
    std::vector<std::string> out;
    size_t begin = 0;
    while (begin <= line.size()) {
        const size_t end = line.find('\t', begin);
        if (end == std::string::npos) {
            out.push_back(line.substr(begin));
            break;
        }
        out.push_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    return out;
}

int32_t parse_cluster_i32(const std::string& value, const std::string& label) {
    size_t used = 0;
    long parsed = 0;
    try {
        parsed = std::stol(value, &used);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson cluster state");
    }
    if (used != value.size() || parsed < std::numeric_limits<int32_t>::min()
        || parsed > std::numeric_limits<int32_t>::max()) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson cluster state");
    }
    return static_cast<int32_t>(parsed);
}

uint64_t parse_cluster_u64(const std::string& value, const std::string& label) {
    size_t used = 0;
    uint64_t parsed = 0;
    try {
        parsed = std::stoull(value, &used);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson cluster state");
    }
    if (used != value.size()) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson cluster state");
    }
    return parsed;
}

double parse_cluster_double(const std::string& value, const std::string& label) {
    size_t used = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(value, &used);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson cluster state");
    }
    if (used != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson cluster state");
    }
    return parsed;
}

void validate_cluster_state(const GammaPoissonClusterState& state) {
    const auto& model = state.model;
    const Eigen::Index components = model.dirichlet_parameters.size();
    const Eigen::Index dim = state.basis.rows();
    const Eigen::Index topics = state.basis.cols();
    const Eigen::Index rank = model.orientation.cols();
    const std::set<std::string> unique_topics(
        state.topic_names.begin(), state.topic_names.end());
    const int32_t uncertainty_model =
        static_cast<int32_t>(state.document_uncertainty_model);
    if (state.topic_state_checksum == 0 || topics < 2 || dim != topics - 1
        || state.topic_names.size() != static_cast<size_t>(topics)
        || unique_topics.size() != state.topic_names.size()
        || unique_topics.count("") != 0
        || components <= 0
        || model.means.rows() != components || model.means.cols() != dim
        || model.variances.rows() != components || model.variances.cols() != dim
        || model.orientation.rows() != dim || rank < 0 || rank > dim
        || model.low_rank_variances.rows() != components
        || model.low_rank_variances.cols() != rank
        || state.document_uncertainty_rank < 0
        || uncertainty_model < static_cast<int32_t>(
            GammaPoissonClusterDataset::UncertaintyModel::MeanField)
        || uncertainty_model > static_cast<int32_t>(
            GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionLowRank)
        || (state.document_uncertainty_model
                == GammaPoissonClusterDataset::UncertaintyModel::MeanField
            && state.document_uncertainty_rank != 0)
        || (state.document_uncertainty_model
                == GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionDiagonal
            && state.document_uncertainty_rank != 0)
        || (state.document_uncertainty_model
                == GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionLowRank
            && state.document_uncertainty_rank <= 0)
        || state.active.size() != static_cast<size_t>(components)
        || !std::isfinite(state.min_cluster_size) || state.min_cluster_size < 0.0
        || (state.input_row_order != "input"
            && state.input_row_order != "randomized")
        || (state.diagnostics.covariance_accumulation != "diagonal"
            && state.diagnostics.covariance_accumulation != "dense"
            && state.diagnostics.covariance_accumulation != "compact")
        || (state.diagnostics.optimizer != "batch"
            && state.diagnostics.optimizer != "svi")
        || state.diagnostics.candidate_components < 0
        || state.diagnostics.candidate_dimensions < 0
        || state.diagnostics.candidate_dimensions > dim
        || (state.diagnostics.candidate_components > 0
            && (state.diagnostics.optimizer != "svi"
                || state.diagnostics.candidate_components < 3
                || state.diagnostics.candidate_components >= components
                || state.diagnostics.candidate_dimensions <= 0
                || (state.diagnostics.candidate_search != "linear"
                    && state.diagnostics.candidate_search != "kdtree")))
        || (state.diagnostics.candidate_components == 0
            && (state.diagnostics.candidate_dimensions != 0
                || state.diagnostics.full_refreshes != 0
                || state.diagnostics.candidate_search != "none"))
        || state.diagnostics.full_refreshes < 0
        || (state.diagnostics.optimizer == "batch"
            && (state.diagnostics.candidate_components != 0
                || state.diagnostics.candidate_dimensions != 0
                || state.diagnostics.full_refreshes != 0))
        || !state.basis.allFinite()
        || !model.dirichlet_parameters.allFinite() || !model.means.allFinite()
        || !model.variances.allFinite() || !model.orientation.allFinite()
        || !model.low_rank_variances.allFinite()
        || (model.dirichlet_parameters.array() <= 0.0).any()
        || (model.variances.array() <= 0.0).any()
        || (rank > 0 && (model.low_rank_variances.array() <= 0.0).any())
        || (state.basis * state.basis.transpose()
            - Eigen::MatrixXd::Identity(dim, dim)).norm() > 1e-8
        || (state.basis * Eigen::VectorXd::Ones(topics)).norm() > 1e-8
        || (rank > 0 && (model.orientation.transpose() * model.orientation
            - Eigen::MatrixXd::Identity(rank, rank)).norm() > 1e-8)
        || std::none_of(state.active.begin(), state.active.end(),
            [](uint8_t value) { return value != 0; })) {
        throw std::runtime_error("Invalid Gamma-Poisson cluster state");
    }
    for (uint8_t value : state.active) {
        if (value > 1) throw std::runtime_error("Invalid active component flag");
    }
}

} // namespace

void write_gamma_poisson_cluster_state(const std::string& path,
    const GammaPoissonClusterState& state) {
    validate_cluster_state(state);
    const auto& model = state.model;
    const auto& diagnostics = state.diagnostics;
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot open clustering state for writing: " + path);
    out << std::scientific << std::setprecision(17);
    out << "##punkst_gamma_pois_cluster\n";
    out << "##state_checksum\t" << state.topic_state_checksum << "\n";
    out << "##n_topics\t" << state.basis.cols() << "\n";
    out << "##n_dimensions\t" << state.basis.rows() << "\n";
    out << "##n_components\t" << model.dirichlet_parameters.size() << "\n";
    out << "##document_uncertainty_model\t";
    switch (state.document_uncertainty_model) {
        case GammaPoissonClusterDataset::UncertaintyModel::MeanField:
            out << "mean_field"; break;
        case GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionDiagonal:
            out << "mean_field_plus_dispersion_diagonal"; break;
        case GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionLowRank:
            out << "mean_field_plus_dispersion_low_rank"; break;
    }
    out << "\n##document_uncertainty_rank\t" << state.document_uncertainty_rank << "\n";
    out << "##cluster_covariance_rank\t" << model.orientation.cols() << "\n";
    out << "##optimizer\t" << diagnostics.optimizer << "\n";
    out << "##covariance_accumulation\t"
        << diagnostics.covariance_accumulation << "\n";
    out << "##input_row_order\t" << state.input_row_order << "\n";
    out << "##min_cluster_size\t" << state.min_cluster_size << "\n";
    out << "##active_components\t"
        << std::count(state.active.begin(), state.active.end(), uint8_t{1}) << "\n";
    out << "##elbo\t" << diagnostics.elbo << "\n";
    out << "##log_likelihood\t" << diagnostics.log_likelihood << "\n";
    out << "##iterations\t" << diagnostics.iterations << "\n";
    out << "##diagonal_warmup_iterations\t" << diagnostics.warmup_iterations << "\n";
    out << "##structured_iterations\t" << diagnostics.structured_iterations << "\n";
    out << "##orientation_updates\t" << diagnostics.orientation_updates << "\n";
    out << "##epochs\t" << diagnostics.epochs << "\n";
    out << "##svi_updates\t" << diagnostics.svi_updates << "\n";
    out << "##refinement_iterations\t"
        << diagnostics.refinement_iterations << "\n";
    out << "##candidate_components\t" << diagnostics.candidate_components << "\n";
    out << "##candidate_dimensions\t" << diagnostics.candidate_dimensions << "\n";
    out << "##candidate_search\t" << diagnostics.candidate_search << "\n";
    out << "##full_refreshes\t" << diagnostics.full_refreshes << "\n";
    out << "##orientation_converged\t" << (diagnostics.orientation_converged ? 1 : 0) << "\n";
    out << "##orientation_change\t" << diagnostics.orientation_change << "\n";
    out << "##converged\t" << (diagnostics.converged ? 1 : 0) << "\n";
    out << "##svi_converged\t" << (diagnostics.svi_converged ? 1 : 0) << "\n";
    out << "##termination\t" << (diagnostics.converged ? "converged" : "max_iterations") << "\n";
    out << "##relative_elbo_change\t" << diagnostics.relative_elbo_change << "\n";
    out << "##relative_predictive_log_likelihood_change\t"
        << diagnostics.relative_predictive_log_likelihood_change << "\n";
    out << "##mean_responsibility_linf_change\t" << diagnostics.mean_responsibility_linf_change << "\n";
    out << "##p90_responsibility_linf_change\t" << diagnostics.p90_responsibility_linf_change << "\n";
    out << "##top_assignment_change_fraction\t" << diagnostics.top_assignment_change_fraction << "\n";
    out << "##max_standardized_center_change\t" << diagnostics.max_standardized_center_change << "\n";
    out << "##max_weight_change\t" << diagnostics.max_weight_change << "\n";
    out << "##max_log_variance_change\t" << diagnostics.max_log_variance_change << "\n";
    out << "##kmeans_iterations\t" << diagnostics.kmeans_iterations << "\n";
    out << "##kmeans_converged\t" << (diagnostics.kmeans_converged ? 1 : 0) << "\n";
    out << "##kmeans_inertia\t" << diagnostics.kmeans_inertia << "\n";
    out << "#record\tindex";
    for (const auto& topic : state.topic_names) out << "\t" << topic;
    out << "\n";
    for (Eigen::Index j = 0; j < state.basis.rows(); ++j) {
        out << "basis\t" << j;
        for (Eigen::Index k = 0; k < state.basis.cols(); ++k) out << "\t" << state.basis(j, k);
        out << "\n";
    }
    for (Eigen::Index q = 0; q < model.orientation.cols(); ++q) {
        out << "orientation\t" << q;
        for (Eigen::Index j = 0; j < model.orientation.rows(); ++j) out << "\t" << model.orientation(j, q);
        out << "\n";
    }
    for (Eigen::Index c = 0; c < model.dirichlet_parameters.size(); ++c) {
        out << "dirichlet\t" << c << "\t" << model.dirichlet_parameters(c) << "\n";
        out << "active\t" << c << "\t" << static_cast<int32_t>(state.active[c]) << "\n";
        out << "mean\t" << c;
        for (Eigen::Index j = 0; j < model.means.cols(); ++j) out << "\t" << model.means(c, j);
        out << "\nvariance\t" << c;
        for (Eigen::Index j = 0; j < model.variances.cols(); ++j) out << "\t" << model.variances(c, j);
        out << "\nlow_rank_variance\t" << c;
        for (Eigen::Index q = 0; q < model.low_rank_variances.cols(); ++q) out << "\t" << model.low_rank_variances(c, q);
        out << "\n";
    }
    if (!out) throw std::runtime_error("Failed writing Gamma-Poisson cluster state");
}

GammaPoissonClusterState read_gamma_poisson_cluster_state(
    const std::string& path, uint64_t expected_topic_state_checksum) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open Gamma-Poisson cluster state: " + path);
    std::string line;
    if (!std::getline(in, line) || line != "##punkst_gamma_pois_cluster") {
        throw std::runtime_error("Invalid Gamma-Poisson cluster state marker");
    }
    std::map<std::string, std::string> metadata;
    std::vector<std::string> records;
    std::vector<std::string> topic_names;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line.rfind("##", 0) == 0) {
            const auto fields = split_cluster_tabs(line.substr(2));
            if (fields.size() != 2
                || !metadata.emplace(fields[0], fields[1]).second) {
                throw std::runtime_error(
                    "Invalid or duplicate Gamma-Poisson cluster state metadata");
            }
        } else if (line.rfind("#record\tindex", 0) == 0) {
            if (saw_header) {
                throw std::runtime_error(
                    "Duplicate Gamma-Poisson cluster state header");
            }
            const auto fields = split_cluster_tabs(line.substr(1));
            if (fields.size() < 3 || fields[0] != "record" || fields[1] != "index") {
                throw std::runtime_error("Invalid Gamma-Poisson cluster state header");
            }
            topic_names.assign(fields.begin() + 2, fields.end());
            std::set<std::string> unique_topics(topic_names.begin(), topic_names.end());
            if (unique_topics.size() != topic_names.size()
                || unique_topics.count("") != 0) {
                throw std::runtime_error(
                    "Invalid Gamma-Poisson cluster state topic names");
            }
            saw_header = true;
        } else {
            if (!saw_header) throw std::runtime_error("Missing Gamma-Poisson cluster state header");
            records.push_back(line);
        }
    }
    const auto required = [&](const std::string& key) -> const std::string& {
        auto it = metadata.find(key);
        if (it == metadata.end()) throw std::runtime_error("Missing " + key + " in Gamma-Poisson cluster state");
        return it->second;
    };
    GammaPoissonClusterState state;
    state.topic_state_checksum = parse_cluster_u64(required("state_checksum"), "state checksum");
    const int32_t topics = parse_cluster_i32(required("n_topics"), "topic count");
    const int32_t dim = parse_cluster_i32(required("n_dimensions"), "dimension count");
    const int32_t components = parse_cluster_i32(required("n_components"), "component count");
    const int32_t document_rank = parse_cluster_i32(required("document_uncertainty_rank"), "document rank");
    const int32_t cluster_rank = parse_cluster_i32(required("cluster_covariance_rank"), "cluster rank");
    const int32_t active_components = parse_cluster_i32(
        required("active_components"), "active component count");
    state.min_cluster_size = parse_cluster_double(required("min_cluster_size"), "minimum cluster size");
    const std::string& uncertainty_model = required("document_uncertainty_model");
    if (uncertainty_model == "mean_field") {
        state.document_uncertainty_model =
            GammaPoissonClusterDataset::UncertaintyModel::MeanField;
    } else if (uncertainty_model == "mean_field_plus_dispersion_diagonal") {
        state.document_uncertainty_model = GammaPoissonClusterDataset::
            UncertaintyModel::MeanFieldPlusDispersionDiagonal;
    } else if (uncertainty_model == "mean_field_plus_dispersion_low_rank") {
        state.document_uncertainty_model = GammaPoissonClusterDataset::
            UncertaintyModel::MeanFieldPlusDispersionLowRank;
    } else {
        throw std::runtime_error(
            "Invalid document uncertainty model in Gamma-Poisson cluster state");
    }
    state.document_uncertainty_rank = document_rank;
    state.input_row_order = required("input_row_order");
    state.topic_names = topic_names;
    if (!saw_header || topics < 2 || dim != topics - 1 || components <= 0
        || document_rank < 0 || cluster_rank < 0 || cluster_rank > dim
        || topic_names.size() != static_cast<size_t>(topics)
        || (state.input_row_order != "input" && state.input_row_order != "randomized")) {
        throw std::runtime_error("Invalid Gamma-Poisson cluster state dimensions");
    }
    if (expected_topic_state_checksum != 0
        && state.topic_state_checksum != expected_topic_state_checksum) {
        throw std::runtime_error("Gamma-Poisson cluster state checksum mismatch");
    }
    auto& model = state.model;
    model.dirichlet_parameters = Eigen::VectorXd::Constant(
        components, std::numeric_limits<double>::quiet_NaN());
    model.means = RowMajorMatrixXd::Constant(components, dim, std::numeric_limits<double>::quiet_NaN());
    model.variances = model.means;
    model.orientation = RowMajorMatrixXd::Constant(dim, cluster_rank, std::numeric_limits<double>::quiet_NaN());
    model.low_rank_variances = RowMajorMatrixXd::Constant(components, cluster_rank, std::numeric_limits<double>::quiet_NaN());
    state.basis = Eigen::MatrixXd::Constant(dim, topics, std::numeric_limits<double>::quiet_NaN());
    state.active.assign(components, 2);
    std::vector<uint8_t> seen_basis(dim), seen_orientation(cluster_rank),
        seen_dirichlet(components), seen_active(components), seen_mean(components),
        seen_variance(components), seen_low_rank(components);
    for (const auto& record : records) {
        const auto fields = split_cluster_tabs(record);
        if (fields.size() < 2) throw std::runtime_error("Truncated Gamma-Poisson cluster state record");
        const int32_t index = parse_cluster_i32(fields[1], "record index");
        auto read_vector = [&](Eigen::Ref<Eigen::VectorXd> target, std::vector<uint8_t>& seen) {
            if (index < 0 || index >= static_cast<int32_t>(seen.size()) || seen[index]
                || fields.size() != static_cast<size_t>(target.size() + 2)) {
                throw std::runtime_error("Invalid Gamma-Poisson cluster state record");
            }
            for (Eigen::Index j = 0; j < target.size(); ++j) target(j) = parse_cluster_double(fields[j + 2], fields[0]);
            seen[index] = 1;
        };
        if (fields[0] == "basis") {
            if (index < 0 || index >= dim || seen_basis[index]
                || fields.size() != static_cast<size_t>(topics + 2)) throw std::runtime_error("Invalid basis record");
            for (int32_t k = 0; k < topics; ++k) state.basis(index, k) = parse_cluster_double(fields[k + 2], "basis");
            seen_basis[index] = 1;
        } else if (fields[0] == "orientation") {
            if (index < 0 || index >= cluster_rank || seen_orientation[index]
                || fields.size() != static_cast<size_t>(dim + 2)) throw std::runtime_error("Invalid orientation record");
            for (int32_t j = 0; j < dim; ++j) model.orientation(j, index) = parse_cluster_double(fields[j + 2], "orientation");
            seen_orientation[index] = 1;
        } else if (fields[0] == "dirichlet") {
            Eigen::VectorXd target(1); read_vector(target, seen_dirichlet); model.dirichlet_parameters(index) = target(0);
        } else if (fields[0] == "active") {
            if (index < 0 || index >= components || seen_active[index] || fields.size() != 3) throw std::runtime_error("Invalid active record");
            const int32_t value = parse_cluster_i32(fields[2], "active flag");
            if (value != 0 && value != 1) throw std::runtime_error("Invalid active flag");
            state.active[index] = static_cast<uint8_t>(value); seen_active[index] = 1;
        } else if (fields[0] == "mean") {
            Eigen::VectorXd target(dim); read_vector(target, seen_mean); model.means.row(index) = target.transpose();
        } else if (fields[0] == "variance") {
            Eigen::VectorXd target(dim); read_vector(target, seen_variance); model.variances.row(index) = target.transpose();
        } else if (fields[0] == "low_rank_variance") {
            Eigen::VectorXd target(cluster_rank); read_vector(target, seen_low_rank); model.low_rank_variances.row(index) = target.transpose();
        } else {
            throw std::runtime_error("Unknown Gamma-Poisson cluster state record: " + fields[0]);
        }
    }
    const auto all_seen = [](const std::vector<uint8_t>& seen) {
        return std::all_of(seen.begin(), seen.end(), [](uint8_t value) { return value != 0; });
    };
    if (!all_seen(seen_basis) || !all_seen(seen_orientation)
        || !all_seen(seen_dirichlet) || !all_seen(seen_active) || !all_seen(seen_mean)
        || !all_seen(seen_variance) || !all_seen(seen_low_rank)) {
        throw std::runtime_error("Incomplete Gamma-Poisson cluster state");
    }
    if (active_components != std::count(
            state.active.begin(), state.active.end(), uint8_t{1})) {
        throw std::runtime_error(
            "Gamma-Poisson cluster state active component count mismatch");
    }
    auto parse_diagnostic = [&](const std::string& key) {
        size_t used = 0;
        const std::string& value = required(key);
        double parsed = 0.0;
        try {
            parsed = std::stod(value, &used);
        } catch (const std::exception&) {
            throw std::runtime_error("Invalid " + key
                + " in Gamma-Poisson cluster state");
        }
        if (used != value.size() || std::isnan(parsed)) {
            throw std::runtime_error("Invalid " + key
                + " in Gamma-Poisson cluster state");
        }
        return parsed;
    };
    auto& diagnostics = state.diagnostics;
    diagnostics.optimizer = required("optimizer");
    diagnostics.covariance_accumulation = required("covariance_accumulation");
    diagnostics.elbo = parse_diagnostic("elbo");
    diagnostics.log_likelihood = parse_diagnostic("log_likelihood");
    diagnostics.iterations = parse_cluster_i32(required("iterations"), "iterations");
    diagnostics.warmup_iterations = parse_cluster_i32(
        required("diagonal_warmup_iterations"), "warmup iterations");
    diagnostics.structured_iterations = parse_cluster_i32(
        required("structured_iterations"), "structured iterations");
    diagnostics.orientation_updates = parse_cluster_i32(
        required("orientation_updates"), "orientation updates");
    diagnostics.epochs = parse_cluster_i32(required("epochs"), "epochs");
    diagnostics.svi_updates = parse_cluster_i32(
        required("svi_updates"), "SVI updates");
    diagnostics.refinement_iterations = parse_cluster_i32(
        required("refinement_iterations"), "refinement iterations");
    diagnostics.candidate_components = parse_cluster_i32(
        required("candidate_components"), "candidate components");
    diagnostics.candidate_dimensions = parse_cluster_i32(
        required("candidate_dimensions"), "candidate dimensions");
    diagnostics.candidate_search = required("candidate_search");
    diagnostics.full_refreshes = parse_cluster_i32(
        required("full_refreshes"), "full refreshes");
    const auto parse_boolean = [&](const std::string& key) {
        const int32_t value = parse_cluster_i32(required(key), key);
        if (value != 0 && value != 1) {
            throw std::runtime_error("Invalid " + key
                + " in Gamma-Poisson cluster state");
        }
        return value != 0;
    };
    diagnostics.orientation_converged = parse_boolean("orientation_converged");
    diagnostics.orientation_change = parse_diagnostic("orientation_change");
    diagnostics.converged = parse_boolean("converged");
    diagnostics.svi_converged = parse_boolean("svi_converged");
    const std::string& termination = required("termination");
    if (termination != (diagnostics.converged
            ? "converged" : "max_iterations")) {
        throw std::runtime_error(
            "Gamma-Poisson cluster state termination metadata mismatch");
    }
    diagnostics.relative_elbo_change = parse_diagnostic("relative_elbo_change");
    diagnostics.relative_predictive_log_likelihood_change = parse_diagnostic(
        "relative_predictive_log_likelihood_change");
    diagnostics.mean_responsibility_linf_change = parse_diagnostic(
        "mean_responsibility_linf_change");
    diagnostics.p90_responsibility_linf_change = parse_diagnostic(
        "p90_responsibility_linf_change");
    diagnostics.top_assignment_change_fraction = parse_diagnostic(
        "top_assignment_change_fraction");
    diagnostics.max_standardized_center_change = parse_diagnostic(
        "max_standardized_center_change");
    diagnostics.max_weight_change = parse_diagnostic("max_weight_change");
    diagnostics.max_log_variance_change = parse_diagnostic(
        "max_log_variance_change");
    diagnostics.kmeans_iterations = parse_cluster_i32(
        required("kmeans_iterations"), "k-means iterations");
    diagnostics.kmeans_converged = parse_boolean("kmeans_converged");
    diagnostics.kmeans_inertia = parse_diagnostic("kmeans_inertia");
    validate_cluster_state(state);
    return state;
}

namespace {

GammaPoissonResponsibilityChange responsibility_change_impl(
    const Eigen::Ref<const RowMajorMatrixXd>& previous,
    const Eigen::Ref<const RowMajorMatrixXd>& current) {
    GammaPoissonResponsibilityChange out;
    std::vector<double> changes(current.rows());
    int64_t top_changes = 0;
    double sum = 0.0;
    for (Eigen::Index d = 0; d < current.rows(); ++d) {
        changes[d] = (current.row(d) - previous.row(d)).cwiseAbs().maxCoeff();
        sum += changes[d];
        Eigen::Index previous_top = 0;
        Eigen::Index current_top = 0;
        previous.row(d).maxCoeff(&previous_top);
        current.row(d).maxCoeff(&current_top);
        top_changes += previous_top != current_top;
    }
    out.mean_linf = sum / static_cast<double>(current.rows());
    const size_t p90_index = static_cast<size_t>(
        std::ceil(0.9 * changes.size())) - 1;
    std::nth_element(changes.begin(), changes.begin() + p90_index, changes.end());
    out.p90_linf = changes[p90_index];
    out.top_assignment_fraction = static_cast<double>(top_changes) / current.rows();
    return out;
}

} // namespace

GammaPoissonResponsibilityChange gamma_poisson_responsibility_change(
    const Eigen::Ref<const RowMajorMatrixXd>& previous,
    const Eigen::Ref<const RowMajorMatrixXd>& current) {
    if (previous.rows() == 0 || previous.rows() != current.rows()
        || previous.cols() != current.cols() || !previous.allFinite()
        || !current.allFinite()) {
        throw std::invalid_argument("Invalid responsibility matrices for convergence comparison");
    }
    return responsibility_change_impl(previous, current);
}

std::vector<uint8_t> gamma_poisson_active_components(
    const Eigen::Ref<const Eigen::VectorXd>& expected_size,
    double min_cluster_size) {
    if (expected_size.size() == 0 || !expected_size.allFinite()
        || (expected_size.array() < 0.0).any()
        || !std::isfinite(min_cluster_size) || min_cluster_size < 0.0) {
        throw std::invalid_argument("Invalid Gamma-Poisson cluster activation input");
    }
    std::vector<uint8_t> active(expected_size.size());
    for (Eigen::Index c = 0; c < expected_size.size(); ++c) {
        active[c] = expected_size(c) >= min_cluster_size;
    }
    if (std::none_of(active.begin(), active.end(),
            [](uint8_t value) { return value != 0; })) {
        Eigen::Index largest = 0;
        expected_size.maxCoeff(&largest);
        active[largest] = 1;
    }
    return active;
}

Eigen::VectorXd gamma_poisson_effective_membership_size(
    const Eigen::Ref<const RowMajorMatrixXd>& responsibilities) {
    if (responsibilities.rows() == 0 || responsibilities.cols() == 0
        || !responsibilities.allFinite()
        || (responsibilities.array() < 0.0).any()) {
        throw std::invalid_argument("Invalid responsibilities for effective membership size");
    }
    Eigen::VectorXd out(responsibilities.cols());
    for (Eigen::Index c = 0; c < responsibilities.cols(); ++c) {
        const double sum = responsibilities.col(c).sum();
        const double sum_square = responsibilities.col(c).squaredNorm();
        out(c) = sum_square > 0.0 ? sum * sum / sum_square : 0.0;
    }
    return out;
}
