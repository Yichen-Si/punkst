#include "clustering/gamma_pois_cluster.hpp"
#include "clustering_core/dense_kmeans.hpp"

#include <algorithm>
#include <array>
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
    const std::vector<std::string>& expected_topic_names,
    const std::string& coordinate_model) {
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
    out.topic_capacity = topic_capacity;
    if (dispersion_reader) {
        out.uncertainty_model = dispersion_reader->header().rank == 0
            ? GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionDiagonal
            : GammaPoissonClusterDataset::UncertaintyModel::MeanFieldPlusDispersionLowRank;
    }
    const bool retain_all = coordinate_model.empty();
    const bool retain_log_moments = retain_all
        || coordinate_model == "log-ratio";
    const bool retain_posterior_parameters = retain_all
        || coordinate_model == "power-ilr";
    const bool retain_combined_covariance = retain_all
        || coordinate_model == "l2" || coordinate_model == "log-ratio";
    const bool retain_dispersion_covariance = retain_all
        || coordinate_model == "power-ilr";
    std::vector<Eigen::VectorXd> means;
    std::vector<Eigen::VectorXd> topic_means;
    std::vector<Eigen::VectorXd> posterior_shapes;
    std::vector<Eigen::VectorXd> posterior_rates;
    GammaPoissonPosteriorRow row;
    while (posterior_reader.read_next(row)) {
        GammaPoissonLogMoments moments;
        if (retain_log_moments || retain_combined_covariance) {
            moments = gamma_poisson_log_moments(
                row.posterior.shape, row.posterior.rate, topic_capacity);
        }
        GammaPoissonTopicCovariance covariance;
        if (retain_combined_covariance) {
            covariance.diagonal = moments.variance;
        }
        covariance.factor.resize(header.n_topics, 0);
        GammaPoissonTopicCovariance dispersion_covariance;
        if (dispersion_reader) {
            GammaPoissonDispersionApproximation approximation;
            if (!dispersion_reader->read_next(approximation)) {
                throw std::runtime_error("Dispersion sidecar has fewer rows than posterior");
            }
            if (retain_dispersion_covariance) {
                dispersion_covariance.diagonal = approximation.residual_diagonal;
                dispersion_covariance.factor = approximation.factor;
            }
            if (retain_combined_covariance) {
                covariance.diagonal += approximation.residual_diagonal;
                covariance.factor = std::move(approximation.factor);
            }
        }
        out.identifiers.push_back(row.identifiers);
        out.posterior_rows.push_back(row.row);
        Eigen::VectorXd topic_mean = (row.posterior.shape.array()
            / row.posterior.rate.array()) * topic_capacity.array();
        const double topic_sum = topic_mean.sum();
        if (!topic_mean.allFinite() || topic_sum <= 0.0) {
            throw std::runtime_error("Invalid posterior topic mean");
        }
        topic_mean /= topic_sum;
        topic_means.push_back(std::move(topic_mean));
        if (retain_posterior_parameters) {
            posterior_shapes.push_back(row.posterior.shape);
            posterior_rates.push_back(row.posterior.rate);
        }
        if (retain_log_moments) means.push_back(std::move(moments.mean));
        if (retain_combined_covariance) {
            out.topic_covariance.push_back(std::move(covariance));
        }
        if (retain_dispersion_covariance) {
            out.dispersion_covariance.push_back(std::move(dispersion_covariance));
        }
    }
    if (dispersion_reader) {
        GammaPoissonDispersionApproximation extra;
        if (dispersion_reader->read_next(extra)) {
            throw std::runtime_error("Dispersion sidecar has more rows than posterior");
        }
        if (dispersion_reader->header().record_count != topic_means.size()) {
            throw std::runtime_error("Dispersion sidecar record count does not match posterior");
        }
    }
    if (topic_means.empty()) {
        throw std::runtime_error("Gamma-Poisson posterior contains no documents");
    }
    if (retain_log_moments) {
        out.log_mean.resize(topic_means.size(), header.n_topics);
    }
    out.topic_mean.resize(topic_means.size(), header.n_topics);
    if (retain_posterior_parameters) {
        out.posterior_shape.resize(topic_means.size(), header.n_topics);
        out.posterior_rate.resize(topic_means.size(), header.n_topics);
    }
    for (size_t d = 0; d < topic_means.size(); ++d) {
        if (retain_log_moments) out.log_mean.row(d) = means[d].transpose();
        out.topic_mean.row(d) = topic_means[d].transpose();
        if (retain_posterior_parameters) {
            out.posterior_shape.row(d) = posterior_shapes[d].transpose();
            out.posterior_rate.row(d) = posterior_rates[d].transpose();
        }
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
    if (dataset.topic_mean.rows() == dataset.log_mean.rows()
        && dataset.topic_mean.cols() == dataset.log_mean.cols()) {
        out.initialization_mean = dataset.topic_mean;
    }
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
    out.diagnostics.mean_covariance_trace =
        out.uncertainty_diagonal.sum() / out.mean.rows();
    if (out.uncertainty_rank > 0) {
        out.diagnostics.mean_covariance_trace +=
            out.uncertainty_factor.squaredNorm() / out.mean.rows();
    }
    out.diagnostics.observed_coordinate_trace =
        (out.mean.rowwise() - out.mean.colwise().mean()).squaredNorm()
        / out.mean.rows();
    return out;
}

GammaPoissonClusterCoordinates make_gamma_poisson_theta_l2_coordinates(
    const GammaPoissonClusterDataset& dataset) {
    if (dataset.topic_mean.rows() == 0
        || dataset.topic_mean.cols() < 2
        || dataset.topic_covariance.size()
            != static_cast<size_t>(dataset.topic_mean.rows())
        || !dataset.topic_mean.allFinite()) {
        throw std::invalid_argument("Invalid Gamma-Poisson topic proportions");
    }
    GammaPoissonClusterCoordinates out;
    out.mean = dataset.topic_mean;
    for (Eigen::Index d = 0; d < out.mean.rows(); ++d) {
        const double norm = out.mean.row(d).norm();
        if (!std::isfinite(norm) || norm <= 0.0) {
            throw std::invalid_argument("Invalid zero Gamma-Poisson topic row");
        }
        out.mean.row(d) /= norm;
    }
    out.initialization_mean = out.mean;
    out.log_ratio_basis.basis = Eigen::MatrixXd::Identity(
        out.mean.cols(), out.mean.cols());
    out.uncertainty_diagonal.resize(out.mean.rows(), out.mean.cols());
    out.uncertainty_rank = 0;
    out.uncertainty_factor.resize(out.mean.rows(), 0);

    // Delta method: theta = q / ||q||_2, with log-topic covariance V.
    // d theta / d log(q) = (I - theta theta') diag(q) / ||q||_2.
    for (Eigen::Index d = 0; d < out.mean.rows(); ++d) {
        const Eigen::VectorXd q = dataset.topic_mean.row(d).transpose();
        const double norm = q.norm();
        const Eigen::VectorXd theta = out.mean.row(d).transpose();
        const Eigen::MatrixXd jacobian =
            (Eigen::MatrixXd::Identity(q.size(), q.size())
                - theta * theta.transpose())
            * q.asDiagonal() / norm;
        const auto& covariance = dataset.topic_covariance[d];
        Eigen::VectorXd diagonal = jacobian.array().square().matrix()
            * covariance.diagonal;
        if (covariance.factor.cols() > 0) {
            const Eigen::MatrixXd transformed = jacobian * covariance.factor;
            diagonal += transformed.array().square().rowwise().sum().matrix();
        }
        out.uncertainty_diagonal.row(d) =
            diagonal.array().max(1e-10).transpose();
    }
    out.diagnostics.mean_covariance_trace =
        out.uncertainty_diagonal.sum() / out.mean.rows();
    out.diagnostics.observed_coordinate_trace =
        (out.mean.rowwise() - out.mean.colwise().mean()).squaredNorm()
        / out.mean.rows();
    return out;
}

namespace {

uint64_t coordinate_mix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ull;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
}

class CoordinatePhiloxStream {
public:
    CoordinatePhiloxStream(uint64_t key_first, uint64_t key_second,
        uint64_t sample, uint64_t topic)
        : key_{static_cast<uint32_t>(key_first),
              static_cast<uint32_t>(key_first >> 32)},
          counter_{static_cast<uint32_t>(sample),
              static_cast<uint32_t>(sample >> 32),
              static_cast<uint32_t>(topic),
              static_cast<uint32_t>(topic >> 32)},
          counter_high_(key_second) {}

    double uniform() {
        const uint64_t first = next_u32();
        const uint64_t second = next_u32();
        const uint64_t bits = (first << 21) ^ (second & 0x1fffffull);
        return (static_cast<double>(bits) + 0.5) * 0x1.0p-53;
    }

    double normal() {
        const double radius = std::sqrt(-2.0 * std::log(uniform()));
        const double angle = 2.0 * std::acos(-1.0) * uniform();
        return radius * std::cos(angle);
    }

private:
    static std::pair<uint32_t, uint32_t> multiply_high_low(
        uint32_t first, uint32_t second) {
        const uint64_t product = static_cast<uint64_t>(first) * second;
        return {static_cast<uint32_t>(product >> 32),
            static_cast<uint32_t>(product)};
    }

    void refill() {
        std::array<uint32_t, 4> value = counter_;
        value[1] ^= static_cast<uint32_t>(counter_high_);
        value[3] ^= static_cast<uint32_t>(counter_high_ >> 32);
        std::array<uint32_t, 2> key = key_;
        for (int32_t round = 0; round < 10; ++round) {
            const auto first = multiply_high_low(0xd2511f53u, value[0]);
            const auto second = multiply_high_low(0xcd9e8d57u, value[2]);
            value = {second.first ^ value[1] ^ key[0], second.second,
                first.first ^ value[3] ^ key[1], first.second};
            key[0] += 0x9e3779b9u;
            key[1] += 0xbb67ae85u;
        }
        buffer_ = value;
        buffer_index_ = 0;
        if (++counter_[0] == 0 && ++counter_[1] == 0
            && ++counter_[2] == 0) {
            ++counter_[3];
        }
    }

    uint32_t next_u32() {
        if (buffer_index_ == buffer_.size()) refill();
        return buffer_[buffer_index_++];
    }

    std::array<uint32_t, 2> key_{};
    std::array<uint32_t, 4> counter_{};
    uint64_t counter_high_ = 0;
    std::array<uint32_t, 4> buffer_{};
    size_t buffer_index_ = 4;
};

double log_gamma_unit_rate(double shape, CoordinatePhiloxStream& random) {
    if (!(shape > 0.0) || !std::isfinite(shape)) {
        throw std::invalid_argument("Gamma shape must be finite and positive");
    }
    if (shape < 1.0) {
        return log_gamma_unit_rate(shape + 1.0, random)
            + std::log(random.uniform()) / shape;
    }
    const double d = shape - 1.0 / 3.0;
    const double c = 1.0 / std::sqrt(9.0 * d);
    for (;;) {
        const double normal = random.normal();
        const double base = 1.0 + c * normal;
        if (base <= 0.0) continue;
        const double cube = base * base * base;
        const double uniform = random.uniform();
        if (uniform < 1.0 - 0.0331 * normal * normal * normal * normal
            || std::log(uniform) < 0.5 * normal * normal
                + d * (1.0 - cube + std::log(cube))) {
            return std::log(d) + 3.0 * std::log(base);
        }
    }
}

Eigen::VectorXd power_coordinate_from_log_intensity(
    const Eigen::Ref<const Eigen::VectorXd>& log_intensity,
    const Eigen::Ref<const Eigen::MatrixXd>& basis, double lambda,
    Eigen::VectorXd* powered_composition = nullptr) {
    Eigen::VectorXd scaled = lambda * log_intensity;
    scaled.array() -= scaled.maxCoeff();
    Eigen::VectorXd powered = scaled.array().exp();
    powered /= powered.sum();
    if (powered_composition) *powered_composition = powered;
    return (static_cast<double>(log_intensity.size()) / lambda)
        * basis * powered;
}

void compress_covariance_factor(const Eigen::Ref<const Eigen::MatrixXd>& source,
    int32_t rank, Eigen::VectorXd& residual, Eigen::MatrixXd& factor) {
    const Eigen::Index dim = source.rows();
    const int32_t retained = std::min<int32_t>(
        std::max(0, rank), std::min(source.rows(), source.cols()));
    const Eigen::VectorXd exact_diagonal =
        source.array().square().rowwise().sum().matrix();
    factor = Eigen::MatrixXd::Zero(dim, std::max(0, rank));
    if (retained > 0 && source.cols() > 0) {
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(
            source, Eigen::ComputeThinU | Eigen::ComputeThinV);
        if (svd.info() != Eigen::Success) {
            throw std::runtime_error("Coordinate covariance SVD failed");
        }
        factor.leftCols(retained) = svd.matrixU().leftCols(retained)
            * svd.singularValues().head(retained).asDiagonal();
    }
    residual = (exact_diagonal
        - factor.array().square().rowwise().sum().matrix()).cwiseMax(0.0);
}

Eigen::VectorXd project_simplex(const Eigen::Ref<const Eigen::VectorXd>& value) {
    std::vector<double> sorted(value.data(), value.data() + value.size());
    std::sort(sorted.begin(), sorted.end(), std::greater<double>());
    double cumulative = 0.0;
    int32_t support = 0;
    for (int32_t j = 0; j < static_cast<int32_t>(sorted.size()); ++j) {
        cumulative += sorted[j];
        if (sorted[j] - (cumulative - 1.0) / (j + 1) > 0.0) {
            support = j + 1;
        }
    }
    cumulative = std::accumulate(sorted.begin(), sorted.begin() + support, 0.0);
    const double threshold = (cumulative - 1.0) / support;
    return (value.array() - threshold).max(0.0).matrix();
}

Eigen::MatrixXd dispersion_source_factor(
    const GammaPoissonTopicCovariance& dispersion,
    const Eigen::Ref<const Eigen::MatrixXd>& jacobian) {
    const Eigen::Index topics = jacobian.cols();
    const bool has_diagonal = dispersion.diagonal.size() == topics
        && dispersion.diagonal.maxCoeff() > 0.0;
    const Eigen::Index diagonal_columns = has_diagonal ? topics : 0;
    Eigen::MatrixXd source(jacobian.rows(),
        diagonal_columns + dispersion.factor.cols());
    if (has_diagonal) {
        source.leftCols(diagonal_columns) = jacobian
            * dispersion.diagonal.cwiseMax(0.0).cwiseSqrt().asDiagonal();
    }
    if (dispersion.factor.cols() > 0) {
        source.rightCols(dispersion.factor.cols()) =
            jacobian * dispersion.factor;
    }
    return source;
}

} // namespace

Eigen::VectorXd gamma_poisson_power_ilr_transform(
    const Eigen::Ref<const Eigen::VectorXd>& topic_intensity,
    double lambda) {
    if (topic_intensity.size() < 2 || !topic_intensity.allFinite()
        || (topic_intensity.array() <= 0.0).any()
        || !(lambda > 0.0) || lambda > 1.0 || !std::isfinite(lambda)) {
        throw std::invalid_argument("Invalid power-ILR input");
    }
    return power_coordinate_from_log_intensity(
        topic_intensity.array().log().matrix(),
        normalized_helmert_basis(topic_intensity.size()), lambda);
}

Eigen::VectorXd gamma_poisson_power_ilr_inverse(
    const Eigen::Ref<const Eigen::MatrixXd>& basis,
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    double lambda) {
    const Eigen::Index topics = basis.cols();
    if (topics < 2 || basis.rows() != topics - 1
        || coordinate.size() != topics - 1 || !basis.allFinite()
        || !coordinate.allFinite() || !(lambda > 0.0) || lambda > 1.0
        || !std::isfinite(lambda)) {
        throw std::invalid_argument("Invalid inverse power-ILR input");
    }
    Eigen::VectorXd powered = Eigen::VectorXd::Constant(topics, 1.0 / topics)
        + (lambda / topics) * basis.transpose() * coordinate;
    powered = project_simplex(powered);
    Eigen::VectorXd composition = powered.array().pow(1.0 / lambda);
    composition /= composition.sum();
    return composition;
}

GammaPoissonClusterCoordinates make_gamma_poisson_power_ilr_coordinates(
    const GammaPoissonClusterDataset& dataset,
    const GammaPoissonPowerIlrOptions& options) {
    const Eigen::Index documents = dataset.posterior_shape.rows();
    const Eigen::Index topics = dataset.posterior_shape.cols();
    const Eigen::Index dim = topics - 1;
    const bool valid_samples = options.samples == 8 || options.samples == 16
        || options.samples == 32 || options.samples == 64;
    if (documents == 0 || topics < 2
        || dataset.posterior_rate.rows() != documents
        || dataset.posterior_rate.cols() != topics
        || dataset.topic_mean.rows() != documents
        || dataset.topic_capacity.size() != topics
        || dataset.posterior_rows.size() != static_cast<size_t>(documents)
        || dataset.dispersion_covariance.size()
            != static_cast<size_t>(documents)
        || !(options.lambda > 0.0) || options.lambda > 1.0
        || !std::isfinite(options.lambda) || !valid_samples
        || options.covariance_rank < 0 || options.covariance_rank > dim
        || options.n_threads <= 0) {
        throw std::invalid_argument("Invalid power-ILR clustering options or dataset");
    }
    GammaPoissonClusterCoordinates out;
    out.log_ratio_basis.helmert = normalized_helmert_basis(topics);
    out.log_ratio_basis.rotation = Eigen::MatrixXd::Identity(dim, dim);
    out.log_ratio_basis.basis = out.log_ratio_basis.helmert;
    out.mean.resize(documents, dim);
    out.initialization_mean = dataset.topic_mean;
    out.uncertainty_rank = options.covariance_rank;
    out.uncertainty_diagonal.resize(documents, dim);
    out.uncertainty_factor.resize(
        documents, dim * options.covariance_rank);
    const uint64_t artifact_first = dataset.posterior_header.artifact_id.empty()
        ? dataset.posterior_header.state_checksum
        : dataset.posterior_header.artifact_id.words[0];
    const uint64_t artifact_second = dataset.posterior_header.artifact_id.empty()
        ? coordinate_mix64(dataset.posterior_header.state_checksum)
        : dataset.posterior_header.artifact_id.words[1];
    int64_t fallback_rows = 0;
    double total_trace = 0.0;
#ifdef _OPENMP
#pragma omp parallel for num_threads(options.n_threads) schedule(static) \
    reduction(+:fallback_rows,total_trace)
#endif
    for (Eigen::Index d = 0; d < documents; ++d) {
        Eigen::MatrixXd samples(dim, options.samples);
        const uint64_t row_key = coordinate_mix64(
            artifact_first ^ dataset.posterior_rows[d] ^ options.seed);
        const uint64_t row_key_second = coordinate_mix64(
            artifact_second ^ dataset.posterior_rows[d]
            ^ (options.seed << 1));
        for (int32_t sample = 0; sample < options.samples; ++sample) {
            Eigen::VectorXd log_intensity(topics);
            for (Eigen::Index k = 0; k < topics; ++k) {
                CoordinatePhiloxStream random(row_key, row_key_second,
                    sample, static_cast<uint64_t>(k));
                log_intensity(k) = log_gamma_unit_rate(
                    dataset.posterior_shape(d, k), random)
                    - std::log(dataset.posterior_rate(d, k))
                    + std::log(dataset.topic_capacity(k));
            }
            samples.col(sample) = power_coordinate_from_log_intensity(
                log_intensity, out.log_ratio_basis.basis, options.lambda);
        }
        Eigen::VectorXd mean = samples.rowwise().mean();
        if (!mean.allFinite()) {
            ++fallback_rows;
            Eigen::VectorXd fallback = dataset.topic_mean.row(d).transpose();
            fallback = fallback.array().max(std::numeric_limits<double>::min());
            mean = gamma_poisson_power_ilr_transform(fallback, options.lambda);
            samples.colwise() = mean;
        }
        out.mean.row(d) = mean.transpose();
        Eigen::MatrixXd centered = samples.colwise() - mean;
        centered /= std::sqrt(static_cast<double>(options.samples - 1));

        const Eigen::VectorXd mean_intensity =
            (dataset.posterior_shape.row(d).array()
                / dataset.posterior_rate.row(d).array()
                * dataset.topic_capacity.transpose().array()).matrix();
        Eigen::VectorXd powered_center;
        power_coordinate_from_log_intensity(
            mean_intensity.array().log().matrix(),
            out.log_ratio_basis.basis, options.lambda, &powered_center);
        const Eigen::MatrixXd jacobian = static_cast<double>(topics)
            * out.log_ratio_basis.basis
            * (powered_center.asDiagonal().toDenseMatrix()
                - powered_center * powered_center.transpose());
        Eigen::MatrixXd dispersion_factor = dispersion_source_factor(
            dataset.dispersion_covariance[d], jacobian);
        Eigen::MatrixXd source(dim, centered.cols() + dispersion_factor.cols());
        source << centered, dispersion_factor;
        Eigen::VectorXd residual;
        Eigen::MatrixXd factor;
        compress_covariance_factor(source, options.covariance_rank,
            residual, factor);
        out.uncertainty_diagonal.row(d) = residual.cwiseMax(1e-10).transpose();
        if (options.covariance_rank > 0) {
            Eigen::Map<RowMajorMatrixXd>(out.uncertainty_factor.row(d).data(),
                dim, options.covariance_rank) = factor;
        }
        total_trace += residual.sum() + factor.squaredNorm();
    }
    out.diagnostics.fallback_rows = fallback_rows;
    out.diagnostics.mean_covariance_trace = total_trace / documents;
    out.diagnostics.observed_coordinate_trace =
        (out.mean.rowwise() - out.mean.colwise().mean()).squaredNorm()
        / documents;
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
    const bool log_ratio = state.coordinate_model == "log-ratio";
    const bool power_ilr = state.coordinate_model == "power-ilr";
    const bool compositional = log_ratio || power_ilr;
    const bool theta_l2 = state.coordinate_model == "l2";
    if (state.topic_state_checksum == 0 || topics < 2
        || (!compositional && !theta_l2)
        || (compositional && dim != topics - 1)
        || (theta_l2 && dim != topics)
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
        || state.document_uncertainty_rank > dim
        || state.coordinate_fallback_rows < 0
        || !std::isfinite(state.mean_document_uncertainty_trace)
        || state.mean_document_uncertainty_trace < 0.0
        || !std::isfinite(state.observed_coordinate_trace)
        || state.observed_coordinate_trace < 0.0
        || (power_ilr && (!(state.power_ilr_lambda > 0.0)
            || state.power_ilr_lambda > 1.0
            || (state.posterior_coordinate_samples != 8
                && state.posterior_coordinate_samples != 16
                && state.posterior_coordinate_samples != 32
                && state.posterior_coordinate_samples != 64)
            || state.coordinate_covariance_rank != state.document_uncertainty_rank
            || state.coordinate_sampler != "philox-v1"
            || state.coordinate_rotation != "none"))
        || state.active.size() != static_cast<size_t>(components)
        || !std::isfinite(state.min_cluster_size) || state.min_cluster_size < 0.0
        || (state.input_row_order != "input"
            && state.input_row_order != "randomized")
        || (state.diagnostics.covariance_accumulation != "diagonal"
            && state.diagnostics.covariance_accumulation != "diagonal-fallback"
            && state.diagnostics.covariance_accumulation != "dense"
            && state.diagnostics.covariance_accumulation != "compact")
        || (state.diagnostics.optimizer != "batch"
            && state.diagnostics.optimizer != "svi")
        || (state.diagnostics.initializer != "kmeans++"
            && state.diagnostics.initializer != "leiden")
        || state.diagnostics.initializer_communities <= 0
        || !std::isfinite(state.diagnostics.initializer_quality)
        || (state.diagnostics.initializer == "leiden"
            && ((state.diagnostics.knn_backend_requested != "auto"
                    && state.diagnostics.knn_backend_requested != "kdtree"
                    && state.diagnostics.knn_backend_requested != "flat")
                || (state.diagnostics.knn_backend_resolved != "kdtree"
                    && state.diagnostics.knn_backend_resolved != "flat")
                || state.diagnostics.knn_neighbors <= 0
                || !std::isfinite(state.diagnostics.knn_search_epsilon)
                || state.diagnostics.knn_search_epsilon < 0.0))
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
        || (compositional
            && (state.basis * Eigen::VectorXd::Ones(topics)).norm() > 1e-8)
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
    out << std::scientific << std::setprecision(4);
    out << "##punkst_gamma_pois_cluster\n";
    out << "##state_checksum\t" << state.topic_state_checksum << "\n";
    out << "##n_topics\t" << state.basis.cols() << "\n";
    out << "##n_dimensions\t" << state.basis.rows() << "\n";
    out << "##n_components\t" << model.dirichlet_parameters.size() << "\n";
    out << "##coordinate_model\t" << state.coordinate_model << "\n";
    out << "##power_ilr_lambda\t" << state.power_ilr_lambda << "\n";
    out << "##posterior_coordinate_samples\t"
        << state.posterior_coordinate_samples << "\n";
    out << "##coordinate_covariance_rank\t"
        << state.coordinate_covariance_rank << "\n";
    out << "##coordinate_seed\t" << state.coordinate_seed << "\n";
    out << "##coordinate_sampler\t" << state.coordinate_sampler << "\n";
    out << "##coordinate_rotation\t" << state.coordinate_rotation << "\n";
    out << "##coordinate_fallback_rows\t"
        << state.coordinate_fallback_rows << "\n";
    out << "##mean_document_uncertainty_trace\t"
        << state.mean_document_uncertainty_trace << "\n";
    out << "##observed_coordinate_trace\t"
        << state.observed_coordinate_trace << "\n";
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
    out << "##initializer\t" << diagnostics.initializer << "\n";
    out << "##initializer_communities\t"
        << diagnostics.initializer_communities << "\n";
    out << "##initializer_quality\t" << diagnostics.initializer_quality << "\n";
    out << "##knn_backend_requested\t"
        << diagnostics.knn_backend_requested << "\n";
    out << "##knn_backend_resolved\t"
        << diagnostics.knn_backend_resolved << "\n";
    out << "##knn_neighbors\t" << diagnostics.knn_neighbors << "\n";
    out << "##knn_search_epsilon\t" << diagnostics.knn_search_epsilon << "\n";
    out << "##covariance_accumulation\t"
        << diagnostics.covariance_accumulation << "\n";
    out << "##input_row_order\t" << state.input_row_order << "\n";
    out << "##min_cluster_size\t" << state.min_cluster_size << "\n";
    out << "##active_components\t"
        << std::count(state.active.begin(), state.active.end(), uint8_t{1}) << "\n";
    out << "##elbo\t" << diagnostics.elbo << "\n";
    out << "##log_likelihood\t" << diagnostics.log_likelihood << "\n";
    const double initial_log_likelihood =
        diagnostics.predictive_log_likelihood_trace.empty()
        ? diagnostics.log_likelihood
        : diagnostics.predictive_log_likelihood_trace.front();
    const double initial_entropy =
        diagnostics.mean_responsibility_entropy_trace.empty()
        ? 0.0 : diagnostics.mean_responsibility_entropy_trace.front();
    const double final_entropy =
        diagnostics.mean_responsibility_entropy_trace.empty()
        ? 0.0 : diagnostics.mean_responsibility_entropy_trace.back();
    out << "##initial_log_likelihood\t" << initial_log_likelihood << "\n";
    out << "##log_likelihood_gain\t"
        << diagnostics.log_likelihood - initial_log_likelihood << "\n";
    out << "##initial_mean_responsibility_entropy\t"
        << initial_entropy << "\n";
    out << "##final_mean_responsibility_entropy\t"
        << final_entropy << "\n";
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
    out << "##structured_covariance_fallback\t"
        << (diagnostics.structured_covariance_fallback ? 1 : 0) << "\n";
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
    const auto coordinate_it = metadata.find("coordinate_model");
    state.coordinate_model = coordinate_it == metadata.end()
        ? "log-ratio" : coordinate_it->second;
    // States written before the CLI spelling cleanup remain readable.
    if (state.coordinate_model == "theta-l2") state.coordinate_model = "l2";
    const auto optional_metadata = [&](const std::string& key) -> const std::string* {
        const auto it = metadata.find(key);
        return it == metadata.end() ? nullptr : &it->second;
    };
    if (const std::string* value = optional_metadata("power_ilr_lambda")) {
        state.power_ilr_lambda = parse_cluster_double(*value, "power-ILR lambda");
    }
    if (const std::string* value = optional_metadata("posterior_coordinate_samples")) {
        state.posterior_coordinate_samples = parse_cluster_i32(
            *value, "posterior coordinate samples");
    }
    if (const std::string* value = optional_metadata("coordinate_covariance_rank")) {
        state.coordinate_covariance_rank = parse_cluster_i32(
            *value, "coordinate covariance rank");
    }
    if (const std::string* value = optional_metadata("coordinate_seed")) {
        state.coordinate_seed = parse_cluster_u64(*value, "coordinate seed");
    }
    if (const std::string* value = optional_metadata("coordinate_sampler")) {
        state.coordinate_sampler = *value;
    }
    if (const std::string* value = optional_metadata("coordinate_rotation")) {
        state.coordinate_rotation = *value;
    }
    if (const std::string* value = optional_metadata("coordinate_fallback_rows")) {
        state.coordinate_fallback_rows = parse_cluster_u64(
            *value, "coordinate fallback rows");
    }
    if (const std::string* value = optional_metadata("mean_document_uncertainty_trace")) {
        state.mean_document_uncertainty_trace = parse_cluster_double(
            *value, "mean document uncertainty trace");
    }
    if (const std::string* value = optional_metadata("observed_coordinate_trace")) {
        state.observed_coordinate_trace = parse_cluster_double(
            *value, "observed coordinate trace");
    }
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
    const bool valid_dimension = state.coordinate_model == "l2"
        ? dim == topics
        : (state.coordinate_model == "log-ratio"
            || state.coordinate_model == "power-ilr")
            && dim == topics - 1;
    if (!saw_header || topics < 2 || !valid_dimension || components <= 0
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
    const auto initializer_it = metadata.find("initializer");
    diagnostics.initializer = initializer_it == metadata.end()
        ? "kmeans++" : initializer_it->second;
    const auto communities_it = metadata.find("initializer_communities");
    diagnostics.initializer_communities = communities_it == metadata.end()
        ? components : parse_cluster_i32(
            communities_it->second, "initializer communities");
    const auto quality_it = metadata.find("initializer_quality");
    diagnostics.initializer_quality = quality_it == metadata.end()
        ? 0.0 : parse_cluster_double(
            quality_it->second, "initializer quality");
    const bool leiden_initializer = diagnostics.initializer == "leiden";
    const auto knn_requested_it = metadata.find("knn_backend_requested");
    diagnostics.knn_backend_requested = knn_requested_it == metadata.end()
        ? "auto" : knn_requested_it->second;
    const auto knn_resolved_it = metadata.find("knn_backend_resolved");
    diagnostics.knn_backend_resolved = knn_resolved_it == metadata.end()
        ? (leiden_initializer ? "kdtree" : "none")
        : knn_resolved_it->second;
    const auto knn_neighbors_it = metadata.find("knn_neighbors");
    diagnostics.knn_neighbors = knn_neighbors_it == metadata.end()
        ? (leiden_initializer ? 15 : 0)
        : parse_cluster_i32(knn_neighbors_it->second, "k-NN neighbors");
    const auto knn_epsilon_it = metadata.find("knn_search_epsilon");
    diagnostics.knn_search_epsilon = knn_epsilon_it == metadata.end()
        ? 0.0 : parse_cluster_double(
            knn_epsilon_it->second, "k-NN epsilon");
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
    const auto structured_fallback_it = metadata.find(
        "structured_covariance_fallback");
    if (structured_fallback_it != metadata.end()) {
        const int32_t value = parse_cluster_i32(
            structured_fallback_it->second,
            "structured covariance fallback");
        if (value != 0 && value != 1) {
            throw std::runtime_error(
                "Invalid structured covariance fallback flag");
        }
        diagnostics.structured_covariance_fallback = value != 0;
    }
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
    // Text states intentionally use compact precision. Restore the exact
    // geometric invariants required by scoring after parsing rounded values.
    if (state.coordinate_model == "log-ratio"
        || state.coordinate_model == "power-ilr") {
        const Eigen::VectorXd common = Eigen::VectorXd::Ones(topics);
        for (int32_t row = 0; row < dim; ++row) {
            Eigen::VectorXd value = state.basis.row(row).transpose();
            value.noalias() -= common * (common.dot(value) / topics);
            for (int32_t previous = 0; previous < row; ++previous) {
                value.noalias() -= state.basis.row(previous).transpose()
                    * state.basis.row(previous).dot(value);
            }
            const double norm = value.norm();
            if (!std::isfinite(norm) || norm <= 1e-12) {
                throw std::runtime_error(
                    "Degenerate rounded Gamma-Poisson cluster basis");
            }
            state.basis.row(row) = (value / norm).transpose();
        }
    }
    for (int32_t column = 0; column < cluster_rank; ++column) {
        Eigen::VectorXd value = model.orientation.col(column);
        for (int32_t previous = 0; previous < column; ++previous) {
            value.noalias() -= model.orientation.col(previous)
                * model.orientation.col(previous).dot(value);
        }
        const double norm = value.norm();
        if (!std::isfinite(norm) || norm <= 1e-12) {
            throw std::runtime_error(
                "Degenerate rounded Gamma-Poisson cluster orientation");
        }
        model.orientation.col(column) = value / norm;
    }
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
