#include "partition_classifier_lrvb.hpp"

#include "gamma_pois_common.hpp"
#include "numerical_utils.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace punkst::partition_classifier {
namespace {

double scaled_fixed_point_change(
        const Eigen::Ref<const Eigen::VectorXd>& next,
        const Eigen::Ref<const Eigen::VectorXd>& previous) {
    if (next.size() != previous.size() || next.size() == 0) {
        throw std::invalid_argument("Invalid fixed-point vectors");
    }
    const double scale = static_cast<double>(next.size())
        + std::max(next.cwiseAbs().sum(), previous.cwiseAbs().sum());
    return (next - previous).cwiseAbs().sum() / scale;
}

struct LdaLocalState {
    Eigen::VectorXd a;
    RowMajorMatrixXd phi;
    int32_t iterations = 0;
    double residual = std::numeric_limits<double>::quiet_NaN();
    bool converged = false;
};

LdaLocalState refine_lda(const Eigen::Ref<const Eigen::VectorXd>& assigned,
        const Document& document,
        const Eigen::Ref<const Eigen::MatrixXd>& beta,
        double alpha, double tolerance, int32_t maximum_iterations) {
    const int32_t topics = static_cast<int32_t>(beta.rows());
    if (assigned.size() != topics || beta.cols() <= 0
            || document.ids.size() != document.cnts.size()
            || !(alpha > 0.0) || !(tolerance > 0.0)
            || maximum_iterations <= 0
            || !assigned.allFinite() || (assigned.array() < 0.0).any()) {
        throw std::invalid_argument("Invalid local LDA posterior");
    }
    Eigen::VectorXd current = assigned;
    LdaLocalState state;
    state.phi.resize(document.ids.size(), topics);
    for (int32_t iteration = 0; iteration < maximum_iterations; ++iteration) {
        state.a = current.array() + alpha;
        const Eigen::VectorXd theta_kernel =
            dirichlet_expectation_1d(state.a, 0.0);
        Eigen::VectorXd next = Eigen::VectorXd::Zero(topics);
        for (size_t feature = 0; feature < document.ids.size(); ++feature) {
            if (document.ids[feature]
                    >= static_cast<uint32_t>(beta.cols())
                    || !(document.cnts[feature] >= 0.0)
                    || !std::isfinite(document.cnts[feature])) {
                throw std::invalid_argument("Invalid local LDA document");
            }
            state.phi.row(feature) = (theta_kernel.array()
                * beta.col(document.ids[feature]).array()).matrix().transpose();
            const double total = state.phi.row(feature).sum();
            if (!(total > 0.0) || !std::isfinite(total)) {
                throw std::runtime_error("Nonfinite local LDA allocation");
            }
            state.phi.row(feature) /= total;
            next.noalias() += document.cnts[feature]
                * state.phi.row(feature).transpose();
        }
        state.residual = scaled_fixed_point_change(next, current);
        current = std::move(next);
        state.iterations = iteration + 1;
        if (state.residual <= tolerance) {
            state.a = current.array() + alpha;
            const Eigen::VectorXd final_kernel =
                dirichlet_expectation_1d(state.a, 0.0);
            for (size_t feature = 0; feature < document.ids.size(); ++feature) {
                state.phi.row(feature) = (final_kernel.array()
                    * beta.col(document.ids[feature]).array()).matrix().transpose();
                state.phi.row(feature) /= state.phi.row(feature).sum();
            }
            state.converged = true;
            break;
        }
    }
    return state;
}

struct LdaCurvature {
    Eigen::VectorXd trigamma_a;
    double trigamma_total = 0.0;
    const Document& document;
    const RowMajorMatrixXd& phi;
    Eigen::VectorXd base_diagonal;
    double jitter = 0.0;

    LdaCurvature(const Eigen::Ref<const Eigen::VectorXd>& a,
            const Document& document_, const RowMajorMatrixXd& phi_)
        : trigamma_a(a.unaryExpr(
              [](double value) { return trigamma(value); })),
          trigamma_total(trigamma(a.sum())), document(document_), phi(phi_) {
        Eigen::VectorXd rdiag = Eigen::VectorXd::Zero(a.size());
        for (Eigen::Index feature = 0; feature < phi.rows(); ++feature) {
            rdiag.array() += document.cnts[static_cast<size_t>(feature)]
                * phi.row(feature).array().transpose()
                * (1.0 - phi.row(feature).array().transpose());
        }
        base_diagonal = trigamma_a
            - trigamma_a.array().square().matrix().cwiseProduct(rdiag)
            - Eigen::VectorXd::Constant(a.size(), trigamma_total);
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& input) const {
        Eigen::VectorXd tx = trigamma_a.array() * input.array();
        Eigen::VectorXd rtx = Eigen::VectorXd::Zero(input.size());
        for (Eigen::Index feature = 0; feature < phi.rows(); ++feature) {
            const auto allocation = phi.row(feature);
            const double projection = allocation.dot(tx);
            rtx.noalias() += document.cnts[static_cast<size_t>(feature)]
                * (allocation.array().transpose()
                    * (tx.array() - projection)).matrix();
        }
        return tx - (trigamma_a.array() * rtx.array()).matrix()
            - Eigen::VectorXd::Constant(input.size(),
                trigamma_total * input.sum()) + jitter * input;
    }

};

struct GammaPoisLocalState {
    Eigen::VectorXd shape;
    Eigen::VectorXd rate;
    RowMajorMatrixXd phi;
    Eigen::VectorXd epsilon;
    int32_t iterations = 0;
    double residual = std::numeric_limits<double>::quiet_NaN();
    bool converged = false;
};

void normalize_gamma_poisson_allocation(
        const Eigen::Ref<const Eigen::VectorXd>& theta_log,
        const Eigen::Ref<const Eigen::VectorXd>& theta_kernel,
        uint32_t feature,
        const Eigen::Ref<const Eigen::MatrixXd>& beta_kernel,
        const GammaPoisson4HexInterface& model,
        Eigen::Ref<Eigen::RowVectorXd> allocation) {
    allocation = (theta_kernel.array()
        * beta_kernel.col(feature).array()).matrix().transpose();
    const double total = allocation.sum();
    if (total > 0.0 && std::isfinite(total)) {
        allocation /= total;
        return;
    }
    Eigen::VectorXd stable;
    model.normalizeTopicAllocation(theta_log,
        static_cast<int32_t>(feature), stable);
    allocation = stable.transpose();
}

GammaPoisLocalState refine_gamma_poisson(
        const GammaPoissonDocumentPosterior& posterior,
        const Document& document,
        const GammaPoisson4HexInterface& model,
        const Eigen::Ref<const Eigen::VectorXd>& capacity,
        const Eigen::Ref<const Eigen::MatrixXd>& beta_kernel,
        const Eigen::Ref<const Eigen::MatrixXd>& beta_mean,
        double prior_shape, const Eigen::Ref<const Eigen::VectorXd>& prior_rate,
        const Eigen::VectorXd* dispersion, double tolerance,
        int32_t maximum_iterations) {
    const int32_t topics = capacity.size();
    if (topics < 2 || posterior.shape.size() != topics
            || posterior.rate.size() != topics || prior_rate.size() != topics
            || beta_kernel.rows() != topics || beta_mean.rows() != topics
            || beta_kernel.cols() != beta_mean.cols()
            || document.ids.size() != document.cnts.size()
            || !(prior_shape > 0.0) || !(tolerance > 0.0)
            || maximum_iterations <= 0
            || !posterior.shape.allFinite()
            || (posterior.shape.array() <= 0.0).any()
            || !posterior.rate.allFinite()
            || (posterior.rate.array() <= 0.0).any()
            || !capacity.allFinite() || (capacity.array() <= 0.0).any()
            || !prior_rate.allFinite() || (prior_rate.array() <= 0.0).any()
            || !std::isfinite(posterior.exposure)
            || posterior.exposure < 0.0
            || (dispersion != nullptr
                && dispersion->size() != beta_mean.cols())) {
        throw std::invalid_argument("Invalid local Gamma-Poisson posterior");
    }
    GammaPoisLocalState state;
    state.shape = posterior.shape;
    state.rate = posterior.rate;
    state.phi.resize(document.ids.size(), topics);
    state.epsilon = Eigen::VectorXd::Ones(document.ids.size());
    for (int32_t iteration = 0; iteration < maximum_iterations; ++iteration) {
        const Eigen::VectorXd previous_shape = state.shape;
        const Eigen::VectorXd previous_rate = state.rate;
        if (dispersion != nullptr) {
            const Eigen::VectorXd mean = state.shape.array()
                / state.rate.array().max(1e-12);
            state.rate = prior_rate + posterior.exposure * capacity;
            for (size_t feature = 0; feature < document.ids.size(); ++feature) {
                const uint32_t word = document.ids[feature];
                if (word >= static_cast<uint32_t>(beta_mean.cols())) {
                    throw std::invalid_argument(
                        "Gamma-Poisson document feature is out of range");
                }
                const double tau = (*dispersion)(word);
                if (!(tau > 0.0) || !std::isfinite(tau)) {
                    throw std::invalid_argument(
                        "Invalid Gamma-Poisson feature dispersion");
                }
                const double intensity = beta_mean.col(word).dot(mean);
                state.epsilon(feature) = (tau + document.cnts[feature])
                    / std::max(tau + posterior.exposure * intensity, 1e-12);
                state.rate.noalias() += posterior.exposure
                    * (state.epsilon(feature) - 1.0) * beta_mean.col(word);
            }
        } else {
            state.rate = prior_rate + posterior.exposure * capacity;
        }
        state.rate = state.rate.array().max(1e-12);
        Eigen::VectorXd theta_log(topics);
        double maximum = -std::numeric_limits<double>::infinity();
        for (int32_t topic = 0; topic < topics; ++topic) {
            theta_log(topic) = psi(state.shape(topic))
                - std::log(state.rate(topic));
            maximum = std::max(maximum, theta_log(topic));
        }
        const Eigen::VectorXd theta_kernel =
            (theta_log.array() - maximum).exp();
        Eigen::VectorXd assigned = Eigen::VectorXd::Zero(topics);
        for (size_t feature = 0; feature < document.ids.size(); ++feature) {
            const uint32_t word = document.ids[feature];
            if (word >= static_cast<uint32_t>(beta_mean.cols())
                    || !(document.cnts[feature] >= 0.0)
                    || !std::isfinite(document.cnts[feature])) {
                throw std::invalid_argument(
                    "Invalid local Gamma-Poisson document");
            }
            normalize_gamma_poisson_allocation(theta_log, theta_kernel, word,
                beta_kernel, model, state.phi.row(feature));
            assigned.noalias() += document.cnts[feature]
                * state.phi.row(feature).transpose();
        }
        const Eigen::VectorXd next_shape = assigned.array() + prior_shape;
        const double shape_residual = scaled_fixed_point_change(
            next_shape, previous_shape);
        const double rate_residual = scaled_fixed_point_change(
            state.rate, previous_rate);
        state.shape = next_shape;
        state.residual = std::max(shape_residual, rate_residual);
        state.iterations = iteration + 1;
        if (state.residual <= tolerance) {
            if (dispersion != nullptr) {
                const Eigen::VectorXd mean = state.shape.array()
                    / state.rate.array().max(1e-12);
                for (size_t feature = 0; feature < document.ids.size(); ++feature) {
                    const uint32_t word = document.ids[feature];
                    const double tau = (*dispersion)(word);
                    const double intensity = beta_mean.col(word).dot(mean);
                    state.epsilon(feature) = (tau + document.cnts[feature])
                        / std::max(tau + posterior.exposure * intensity, 1e-12);
                }
            }
            double final_maximum = -std::numeric_limits<double>::infinity();
            for (int32_t topic = 0; topic < topics; ++topic) {
                theta_log(topic) = psi(state.shape(topic))
                    - std::log(state.rate(topic));
                final_maximum = std::max(final_maximum, theta_log(topic));
            }
            const Eigen::VectorXd final_theta_kernel =
                (theta_log.array() - final_maximum).exp();
            for (size_t feature = 0; feature < document.ids.size(); ++feature) {
                const uint32_t word = document.ids[feature];
                normalize_gamma_poisson_allocation(theta_log,
                    final_theta_kernel, word, beta_kernel, model,
                    state.phi.row(feature));
            }
            state.converged = true;
            break;
        }
    }
    return state;
}

struct GammaPoisCurvature {
    Eigen::VectorXd t;
    Eigen::VectorXd u;
    Eigen::VectorXd v;
    const Document& document;
    const RowMajorMatrixXd& phi;
    const Eigen::MatrixXd& beta_mean;
    Eigen::VectorXd q_coefficients;
    Eigen::VectorXd base_diagonal;
    double jitter = 0.0;

    GammaPoisCurvature(const GammaPoisLocalState& local,
            const Document& document_, const Eigen::MatrixXd& beta_mean_,
            const Eigen::VectorXd* dispersion, double exposure)
        : t(local.shape.unaryExpr(
              [](double value) { return trigamma(value); })),
          u(local.rate.cwiseInverse()),
          v((local.shape.array() / local.rate.array().square()).matrix()),
          document(document_), phi(local.phi), beta_mean(beta_mean_),
          q_coefficients(Eigen::VectorXd::Zero(document.ids.size())) {
        Eigen::VectorXd rdiag = Eigen::VectorXd::Zero(t.size());
        for (Eigen::Index feature = 0; feature < phi.rows(); ++feature) {
            const auto allocation = phi.row(feature);
            rdiag.array() += document.cnts[static_cast<size_t>(feature)]
                * allocation.array().transpose()
                * (1.0 - allocation.array().transpose());
        }
        Eigen::VectorXd qdiag = Eigen::VectorXd::Zero(t.size());
        if (dispersion != nullptr) {
            const Eigen::VectorXd mean = local.shape.array()
                / local.rate.array();
            if (beta_mean.rows() != mean.size()) {
                throw std::invalid_argument(
                    "Gamma-Poisson beta mean has the wrong topic dimension");
            }
            for (size_t feature = 0; feature < document.ids.size(); ++feature) {
                const uint32_t word = document.ids[feature];
                const double tau = (*dispersion)(word);
                const double denominator = std::max(tau + exposure
                    * beta_mean.col(word).dot(mean), 1e-12);
                q_coefficients(feature) = -exposure * exposure
                    * (tau + document.cnts[feature])
                    / (denominator * denominator);
                qdiag.array() += q_coefficients(feature)
                    * beta_mean.col(word).array().square();
            }
        }
        base_diagonal.resize(2 * t.size());
        base_diagonal.head(t.size()) = t
            - t.array().square().matrix().cwiseProduct(rdiag)
            - u.array().square().matrix().cwiseProduct(qdiag);
        base_diagonal.tail(t.size()) = v
            - u.array().square().matrix().cwiseProduct(rdiag)
            - v.array().square().matrix().cwiseProduct(qdiag);
    }

    Eigen::VectorXd apply_r(const Eigen::VectorXd& input) const {
        Eigen::VectorXd output = Eigen::VectorXd::Zero(input.size());
        for (Eigen::Index feature = 0; feature < phi.rows(); ++feature) {
            const auto allocation = phi.row(feature);
            const double projection = allocation.dot(input);
            output.noalias() += document.cnts[feature]
                * (allocation.array().transpose()
                    * (input.array() - projection)).matrix();
        }
        return output;
    }

    Eigen::VectorXd apply_q(const Eigen::VectorXd& input) const {
        Eigen::VectorXd output = Eigen::VectorXd::Zero(input.size());
        if (beta_mean.rows() != input.size()) {
            throw std::invalid_argument(
                "Gamma-Poisson curvature vector has the wrong dimension: beta="
                + std::to_string(beta_mean.rows()) + ", input="
                + std::to_string(input.size()));
        }
        for (size_t feature = 0; feature < document.ids.size(); ++feature) {
            const uint32_t word = document.ids[feature];
            output.noalias() += q_coefficients(feature) * beta_mean.col(word)
                * beta_mean.col(word).dot(input);
        }
        return output;
    }

    Eigen::VectorXd apply(const Eigen::VectorXd& input) const {
        const int32_t topics = t.size();
        const Eigen::VectorXd x = input.head(topics);
        const Eigen::VectorXd y = input.tail(topics);
        const Eigen::VectorXd tx = t.array() * x.array();
        const Eigen::VectorXd uy = u.array() * y.array();
        const Eigen::VectorXd ux = u.array() * x.array();
        const Eigen::VectorXd vy = v.array() * y.array();
        const Eigen::VectorXd rtx = apply_r(tx);
        const Eigen::VectorXd quy = apply_q(ux);
        const Eigen::VectorXd ruy = apply_r(uy);
        const Eigen::VectorXd qvy = apply_q(vy);
        Eigen::VectorXd output(2 * topics);
        output.head(topics) = tx
            - (t.array() * rtx.array()).matrix()
            - (u.array() * quy.array()).matrix()
            + (t.array() * ruy.array()).matrix()
            + (u.array() * qvy.array()).matrix() - uy;
        output.tail(topics) =
            (u.array() * rtx.array()).matrix()
            + (v.array() * quy.array()).matrix() - ux
            + vy - (u.array() * ruy.array()).matrix()
            - (v.array() * qvy.array()).matrix();
        output.noalias() += jitter * input;
        return output;
    }

};

template<class Curvature>
bool preconditioned_cg(const Curvature& curvature,
        const Eigen::Ref<const Eigen::VectorXd>& diagonal,
        const Eigen::VectorXd& right, double tolerance, int32_t maximum,
        Eigen::VectorXd& solution, int32_t& iterations) {
    if (!diagonal.allFinite() || (diagonal.array() <= 0.0).any()) return false;
    solution = Eigen::VectorXd::Zero(right.size());
    Eigen::VectorXd residual = right;
    Eigen::VectorXd z = residual.array() / diagonal.array();
    Eigen::VectorXd direction = z;
    double rz = residual.dot(z);
    const double target = tolerance * std::max(1.0, right.norm());
    if (residual.norm() <= target) {
        iterations = 0;
        return true;
    }
    for (iterations = 1; iterations <= maximum; ++iterations) {
        const Eigen::VectorXd hd = curvature.apply(direction);
        const double denominator = direction.dot(hd);
        if (!(denominator > 0.0) || !std::isfinite(denominator)) return false;
        solution.noalias() += (rz / denominator) * direction;
        residual.noalias() -= (rz / denominator) * hd;
        if (!solution.allFinite() || !residual.allFinite()) return false;
        if (residual.norm() <= target) return true;
        z = residual.array() / diagonal.array();
        const double next = residual.dot(z);
        if (!(next >= 0.0) || !std::isfinite(next)) return false;
        direction = z + (next / rz) * direction;
        rz = next;
    }
    return false;
}

std::vector<int32_t> select_candidates(const Eigen::VectorXd& probability,
        double target) {
    std::vector<int32_t> order(probability.size());
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int32_t left,
            int32_t right) { return probability(left) > probability(right); });
    double mass = 0.0;
    size_t take = 0;
    while (take < order.size() && (take < 2 || mass < target)) {
        mass += probability(order[take]);
        ++take;
    }
    order.resize(take);
    return order;
}

Eigen::VectorXd conditional_softmax(const Eigen::VectorXd& contrasts) {
    Eigen::VectorXd logits(contrasts.size() + 1);
    logits.head(contrasts.size()) = contrasts;
    logits(logits.size() - 1) = 0.0;
    softmaxInPlace(logits);
    return logits;
}

Eigen::MatrixXd covariance_factor(const Eigen::MatrixXd& covariance) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        0.5 * (covariance + covariance.transpose()));
    if (solver.info() != Eigen::Success || !solver.eigenvalues().allFinite()) {
        throw std::runtime_error("Projected covariance eigendecomposition failed");
    }
    const double scale = std::max(1.0,
        solver.eigenvalues().cwiseAbs().maxCoeff());
    if (solver.eigenvalues().minCoeff() < -1e-8 * scale) {
        throw std::runtime_error("Projected covariance is not PSD");
    }
    return solver.eigenvectors()
        * solver.eigenvalues().cwiseMax(0.0).cwiseSqrt().asDiagonal();
}

bool delta_probabilities(const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance, Eigen::VectorXd& output) {
    const int32_t dimension = static_cast<int32_t>(mean.size());
    const Eigen::VectorXd plugin = conditional_softmax(mean);
    Eigen::MatrixXd full = Eigen::MatrixXd::Zero(
        dimension + 1, dimension + 1);
    full.topLeftCorner(dimension, dimension) = covariance;
    const double common = (plugin.array()
        * full.diagonal().array()).sum()
        - plugin.dot(full * plugin);
    output = plugin;
    for (int32_t component = 0; component <= dimension; ++component) {
        Eigen::VectorXd direction = -plugin;
        direction(component) += 1.0;
        output(component) += 0.5 * plugin(component)
            * (direction.dot(full * direction) - common);
    }
    const double change = (output - plugin).cwiseAbs().sum();
    return output.allFinite() && (output.array() >= 0.0).all()
        && std::abs(output.sum() - 1.0) <= 1e-8 && change <= 0.05;
}

Eigen::VectorXd integrate_probabilities(const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance) {
    const int32_t dimension = static_cast<int32_t>(mean.size());
    const Eigen::MatrixXd factor = covariance_factor(covariance);
    Eigen::VectorXd output = Eigen::VectorXd::Zero(dimension + 1);
    if (dimension <= 2) {
        static constexpr std::array<double, 15> nodes{
            -4.499990707309392, -3.669950373404453, -2.967166927905603,
            -2.325732486173858, -1.719992575186489, -1.136115585210921,
            -0.565069583255576, 0.0, 0.565069583255576,
            1.136115585210921, 1.719992575186489, 2.325732486173858,
            2.967166927905603, 3.669950373404453, 4.499990707309392};
        static constexpr std::array<double, 15> weights{
            1.522475804253517e-9, 1.059115547711067e-6,
            1.000044412324999e-4, 2.778068842912776e-3,
            3.078003387254608e-2, 1.584889157959358e-1,
            4.120286874988986e-1, 5.641003087264175e-1,
            4.120286874988986e-1, 1.584889157959358e-1,
            3.078003387254608e-2, 2.778068842912776e-3,
            1.000044412324999e-4, 1.059115547711067e-6,
            1.522475804253517e-9};
        const double inverse_sqrt_pi = 1.0 / std::sqrt(M_PI);
        if (dimension == 1) {
            for (size_t first = 0; first < nodes.size(); ++first) {
                const Eigen::VectorXd point = mean
                    + std::sqrt(2.0) * factor.col(0) * nodes[first];
                output.noalias() += weights[first] * inverse_sqrt_pi
                    * conditional_softmax(point);
            }
        } else {
            for (size_t first = 0; first < nodes.size(); ++first) {
                for (size_t second = 0; second < nodes.size(); ++second) {
                    const Eigen::VectorXd point = mean + std::sqrt(2.0)
                        * (factor.col(0) * nodes[first]
                            + factor.col(1) * nodes[second]);
                    output.noalias() += weights[first] * weights[second]
                        * inverse_sqrt_pi * inverse_sqrt_pi
                        * conditional_softmax(point);
                }
            }
        }
    } else {
        const double radius = std::sqrt(static_cast<double>(dimension));
        for (int32_t axis = 0; axis < dimension; ++axis) {
            output.noalias() += conditional_softmax(
                mean + radius * factor.col(axis));
            output.noalias() += conditional_softmax(
                mean - radius * factor.col(axis));
        }
        output /= 2.0 * dimension;
    }
    return output / output.sum();
}

Eigen::VectorXd propagate_softmax_probabilities(const Eigen::VectorXd& mean,
        const Eigen::MatrixXd& covariance, std::string& method) {
    const int32_t dimension = static_cast<int32_t>(mean.size());
    if (dimension <= 2) {
        method = "quadrature";
        return integrate_probabilities(mean, covariance);
    }
    Eigen::VectorXd output;
    if (delta_probabilities(mean, covariance, output)) {
        method = "delta";
        return output;
    }
    method = "cubature";
    return integrate_probabilities(mean, covariance);
}

} // namespace

namespace testing {

void run_lrvb_numerical_tests() {
    const auto check = [](bool condition, const char* message) {
        if (!condition) throw std::runtime_error(message);
    };
    for (const int32_t dimension : {1, 2}) {
        const Eigen::VectorXd mean = Eigen::VectorXd::Zero(dimension);
        const Eigen::MatrixXd covariance =
            0.1 * Eigen::MatrixXd::Identity(dimension, dimension);
        std::string method;
        const Eigen::VectorXd probability = propagate_softmax_probabilities(
            mean, covariance, method);
        check(method == "quadrature"
                && probability.allFinite()
                && std::abs(probability.sum() - 1.0) < 1e-12,
            "One- or two-contrast propagation did not use quadrature");
    }
    {
        const Eigen::VectorXd mean = Eigen::VectorXd::Zero(3);
        const Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(3, 3);
        std::string method;
        (void)propagate_softmax_probabilities(mean, covariance, method);
        check(method == "delta",
            "Higher-dimensional propagation did not retain guarded delta");
    }
    Eigen::VectorXd unchanged(3);
    unchanged << 10.0, 20.0, 30.0;
    check(scaled_fixed_point_change(unchanged, unchanged) == 0.0,
        "Scaled fixed-point residual is nonzero without a change");
    Eigen::VectorXd first(3), second(3);
    first << 1000.0, 2000.0, 3000.0;
    second << 1001.0, 1998.0, 3003.0;
    const double residual = scaled_fixed_point_change(second, first);
    first *= 100.0;
    second *= 100.0;
    check(std::abs(scaled_fixed_point_change(second, first) - residual)
            < 1e-6,
        "Scaled fixed-point residual depends materially on coordinate scale");

    Document lda_document;
    lda_document.ids = {0, 1};
    lda_document.cnts = {2.0, 3.0};
    Eigen::VectorXd lda_a(3);
    lda_a << 8.0, 6.0, 5.0;
    RowMajorMatrixXd lda_phi(2, 3);
    lda_phi << 0.6, 0.3, 0.1,
               0.2, 0.5, 0.3;
    LdaCurvature lda_curvature(lda_a, lda_document, lda_phi);
    const Eigen::VectorXd lda_t = lda_curvature.trigamma_a;
    Eigen::MatrixXd lda_r = Eigen::MatrixXd::Zero(3, 3);
    for (Eigen::Index feature = 0; feature < lda_phi.rows(); ++feature) {
        const Eigen::VectorXd allocation = lda_phi.row(feature).transpose();
        Eigen::MatrixXd allocation_diagonal = allocation.asDiagonal();
        lda_r.noalias() += lda_document.cnts[feature]
            * (allocation_diagonal - allocation * allocation.transpose());
    }
    const Eigen::MatrixXd lda_t_diagonal = lda_t.asDiagonal();
    Eigen::MatrixXd lda_dense = lda_t_diagonal
        - lda_t_diagonal * lda_r * lda_t_diagonal
        - lda_curvature.trigamma_total * Eigen::MatrixXd::Ones(3, 3);
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> lda_eigen(lda_dense);
    check(lda_eigen.info() == Eigen::Success,
        "LDA curvature eigendecomposition failed in numerical test");
    lda_curvature.jitter = std::max(0.0,
        1e-3 - lda_eigen.eigenvalues().minCoeff());
    lda_dense.diagonal().array() += lda_curvature.jitter;
    Eigen::VectorXd lda_input(3);
    lda_input << 0.4, -0.7, 0.2;
    check(lda_curvature.apply(lda_input).isApprox(
            lda_dense * lda_input, 1e-10),
        "LDA Hessian-vector product disagrees with dense curvature");
    Eigen::VectorXd lda_solution;
    int32_t lda_iterations = 0;
    const Eigen::VectorXd lda_diagonal =
        lda_curvature.base_diagonal.array() + lda_curvature.jitter;
    check(preconditioned_cg(lda_curvature, lda_diagonal, lda_input,
            1e-10, 100, lda_solution, lda_iterations)
            && lda_solution.isApprox(lda_dense.ldlt().solve(lda_input), 1e-7),
        "LDA PCG solve disagrees with dense solve");

    GammaPoisLocalState gp_local;
    gp_local.shape.resize(3);
    gp_local.shape << 5.0, 4.0, 3.0;
    gp_local.rate.resize(3);
    gp_local.rate << 2.0, 2.5, 3.0;
    gp_local.phi = lda_phi;
    Eigen::MatrixXd beta_mean(3, 2);
    beta_mean << 0.7, 0.2,
                 0.2, 0.6,
                 0.1, 0.2;
    Eigen::VectorXd dispersion(2);
    dispersion << 10.0, 12.0;
    GammaPoisCurvature gp_curvature(gp_local, lda_document, beta_mean,
        &dispersion, 1.3);
    Eigen::MatrixXd gp_r = lda_r;
    Eigen::MatrixXd gp_q = Eigen::MatrixXd::Zero(3, 3);
    for (size_t feature = 0; feature < lda_document.ids.size(); ++feature) {
        const Eigen::VectorXd beta = beta_mean.col(
            lda_document.ids[feature]);
        gp_q.noalias() += gp_curvature.q_coefficients(feature)
            * beta * beta.transpose();
    }
    const Eigen::MatrixXd t = gp_curvature.t.asDiagonal();
    const Eigen::MatrixXd u = gp_curvature.u.asDiagonal();
    const Eigen::MatrixXd v = gp_curvature.v.asDiagonal();
    Eigen::MatrixXd gp_dense(6, 6);
    gp_dense.topLeftCorner(3, 3) = t - t * gp_r * t - u * gp_q * u;
    gp_dense.topRightCorner(3, 3) = t * gp_r * u + u * gp_q * v - u;
    gp_dense.bottomLeftCorner(3, 3) = u * gp_r * t + v * gp_q * u - u;
    gp_dense.bottomRightCorner(3, 3) = v - u * gp_r * u - v * gp_q * v;
    check(gp_dense.isApprox(gp_dense.transpose(), 1e-12),
        "Gamma-Poisson dense curvature is not symmetric");
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> gp_eigen(gp_dense);
    check(gp_eigen.info() == Eigen::Success,
        "Gamma-Poisson curvature eigendecomposition failed in numerical test");
    gp_curvature.jitter = std::max(0.0,
        1e-3 - gp_eigen.eigenvalues().minCoeff());
    gp_dense.diagonal().array() += gp_curvature.jitter;
    Eigen::VectorXd gp_input(6);
    gp_input << 0.4, -0.7, 0.2, -0.1, 0.5, 0.3;
    check(gp_curvature.apply(gp_input).isApprox(
            gp_dense * gp_input, 1e-10),
        "Gamma-Poisson Hessian-vector product disagrees with dense curvature");
    Eigen::VectorXd gp_solution;
    int32_t gp_iterations = 0;
    const Eigen::VectorXd gp_diagonal =
        gp_curvature.base_diagonal.array() + gp_curvature.jitter;
    check(preconditioned_cg(gp_curvature, gp_diagonal, gp_input,
            1e-10, 200, gp_solution, gp_iterations)
            && gp_solution.isApprox(gp_dense.ldlt().solve(gp_input), 1e-7),
        "Gamma-Poisson PCG solve disagrees with dense solve");
}

} // namespace testing

PropagatedPrediction propagate_lda(const Model& classifier,
        const Eigen::Ref<const Eigen::VectorXd>& assigned_counts,
        const Document& document,
        const Eigen::Ref<const Eigen::MatrixXd>& lda_allocation_kernel,
        double alpha, const PropagationOptions& options) {
    classifier.validate();
    if (classifier.topics.size()
            != static_cast<size_t>(lda_allocation_kernel.rows())) {
        throw std::invalid_argument(
            "Classifier topics do not match LDA state topics");
    }
    if (!(options.fixed_point_tolerance > 0.0)
            || !std::isfinite(options.fixed_point_tolerance)
            || options.fixed_point_max_iterations <= 0) {
        throw std::invalid_argument("Invalid fixed-point options");
    }
    PropagatedPrediction result;
    Eigen::VectorXd fallback_probabilities;
    try {
        Eigen::VectorXd initial_a = assigned_counts.array() + alpha;
        result.probabilities = classifier.probabilities(initial_a);
        fallback_probabilities = result.probabilities;
        Eigen::Index initial_leading = 0;
        const double initial_maximum =
            result.probabilities.maxCoeff(&initial_leading);
        if (options.plugin_only) {
            result.lrvb_status = "plugin_only";
            return result;
        }
        if (!options.lrvb_all
                && initial_maximum >= options.ambiguity_threshold) {
            result.lrvb_status = "skipped_confident";
            return result;
        }
        result.lrvb_attempted = true;
        const LdaLocalState local = refine_lda(assigned_counts, document,
            lda_allocation_kernel, alpha, options.fixed_point_tolerance,
            options.fixed_point_max_iterations);
        result.fixed_point_iterations = local.iterations;
        result.fixed_point_residual = local.residual;
        if (!local.converged) {
            result.lrvb_status = "local_nonconvergence";
            result.lrvb_failed = true;
            return result;
        }
        const Eigen::VectorXd mean = local.a / local.a.sum();
        fallback_probabilities = classifier.probabilities(mean);
        result.probabilities = fallback_probabilities;
        const std::vector<int32_t> candidates = select_candidates(
            result.probabilities, options.candidate_mass);
        result.candidate_count = candidates.size();
        double candidate_mass = 0.0;
        for (const int32_t component : candidates) {
            candidate_mass += result.probabilities(component);
        }
        result.held_fixed_tail_mass = std::max(0.0, 1.0 - candidate_mass);
        const int32_t dimension = static_cast<int32_t>(candidates.size()) - 1;
        const int32_t baseline = candidates.back();
        Eigen::MatrixXd d(dimension, local.a.size());
        Eigen::VectorXd contrast_mean(dimension);
        const Eigen::VectorXd all_logits = classifier.logits(mean);
        for (int32_t contrast = 0; contrast < dimension; ++contrast) {
            const Eigen::RowVectorXd coefficient =
                (classifier.coefficients.row(candidates[contrast])
                    - classifier.coefficients.row(baseline))
                / classifier.temperature;
            d.row(contrast) = (coefficient.array()
                - coefficient.dot(mean)).matrix() / local.a.sum();
            contrast_mean(contrast) = all_logits(candidates[contrast])
                - all_logits(baseline);
        }

        LdaCurvature curvature(local.a, document, local.phi);
        std::vector<double> positive_diagonal;
        for (Eigen::Index topic = 0;
                topic < curvature.base_diagonal.size(); ++topic) {
            const double value = curvature.base_diagonal(topic);
            if (value > 0.0 && std::isfinite(value)) {
                positive_diagonal.push_back(value);
            }
        }
        if (positive_diagonal.empty()) {
            throw std::runtime_error("nonpositive_curvature");
        }
        const double scale = median(positive_diagonal);
        Eigen::MatrixXd solved(local.a.size(), dimension);
        bool solved_all = false;
        int32_t maximum_iterations = std::min<int32_t>(
            8 * local.a.size(), 1000);
        for (const double multiplier :
                {0.0, 1e-12, 1e-10, 1e-8, 1e-6}) {
            curvature.jitter = multiplier * scale;
            const Eigen::VectorXd diagonal = curvature.base_diagonal.array()
                + curvature.jitter;
            solved_all = true;
            int32_t maximum_used = 0;
            for (int32_t contrast = 0; contrast < dimension; ++contrast) {
                int32_t iterations = 0;
                Eigen::VectorXd solution;
                if (!preconditioned_cg(curvature, diagonal,
                        d.row(contrast).transpose(),
                        options.cg_tolerance, maximum_iterations,
                        solution, iterations)) {
                    solved_all = false;
                    break;
                }
                solved.col(contrast) = solution;
                maximum_used = std::max(maximum_used, iterations);
            }
            if (solved_all) {
                result.cg_iterations = maximum_used;
                result.curvature_jitter = curvature.jitter;
                break;
            }
        }
        if (!solved_all) throw std::runtime_error("curvature_solve_failed");
        Eigen::MatrixXd covariance = d * solved;
        covariance = 0.5 * (covariance + covariance.transpose());
        (void)covariance_factor(covariance);
        const Eigen::VectorXd conditional = propagate_softmax_probabilities(
            contrast_mean, covariance, result.method);
        Eigen::VectorXd propagated = fallback_probabilities;
        for (int32_t index = 0; index < static_cast<int32_t>(candidates.size());
                ++index) {
            propagated(candidates[index]) =
                candidate_mass * conditional(index);
        }
        propagated /= propagated.sum();
        result.probabilities = std::move(propagated);
        result.lrvb_status = "ok";
    } catch (const std::exception& exception) {
        if (fallback_probabilities.size() != 0) {
            result.probabilities = fallback_probabilities;
        } else if (result.probabilities.size() == 0) {
            Eigen::VectorXd center = assigned_counts.array() + alpha;
            result.probabilities = classifier.probabilities(center);
        }
        result.method = "plugin";
        result.lrvb_status = exception.what();
        result.lrvb_attempted = true;
        result.lrvb_failed = true;
    }
    return result;
}

PropagatedPrediction propagate_gamma_poisson(const Model& classifier,
        const GammaPoissonDocumentPosterior& posterior,
        const Document& document,
        const GammaPoisson4HexInterface& model,
        const PropagationOptions& options,
        const Eigen::VectorXd* initial_composition) {
    const Eigen::VectorXd& topic_capacity = model.getTopicCapacity();
    const Eigen::MatrixXd& beta_allocation_kernel =
        model.getBetaAllocationKernel();
    const Eigen::MatrixXd& expected_beta = model.getExpectedBeta();
    const double prior_shape = model.getThetaPriorShape();
    const Eigen::VectorXd prior_rate = model.getThetaPriorRate();
    const Eigen::VectorXd* feature_dispersion = model.hasFeatureDispersion()
        ? &model.getFeatureDispersion() : nullptr;
    classifier.validate();
    if (classifier.topics.size() != static_cast<size_t>(topic_capacity.size())) {
        throw std::invalid_argument(
            "Classifier topics do not match Gamma-Poisson state topics");
    }
    if (!(options.fixed_point_tolerance > 0.0)
            || !std::isfinite(options.fixed_point_tolerance)
            || options.fixed_point_max_iterations <= 0) {
        throw std::invalid_argument("Invalid fixed-point options");
    }
    PropagatedPrediction result;
    Eigen::VectorXd fallback_probabilities;
    try {
        if (initial_composition != nullptr) {
            result.probabilities = classifier.probabilities(
                *initial_composition);
        } else {
            const Eigen::VectorXd initial_abundance = topic_capacity.array()
                * posterior.shape.array() / posterior.rate.array();
            result.probabilities = classifier.probabilities(initial_abundance);
        }
        fallback_probabilities = result.probabilities;
        Eigen::Index initial_leading = 0;
        const double initial_maximum =
            result.probabilities.maxCoeff(&initial_leading);
        if (options.plugin_only) {
            result.lrvb_status = "plugin_only";
            return result;
        }
        if (!options.lrvb_all
                && initial_maximum >= options.ambiguity_threshold) {
            result.lrvb_status = "skipped_confident";
            return result;
        }
        result.lrvb_attempted = true;
        const GammaPoisLocalState local = refine_gamma_poisson(posterior,
            document, model, topic_capacity, beta_allocation_kernel,
            expected_beta, prior_shape, prior_rate, feature_dispersion,
            options.fixed_point_tolerance,
            options.fixed_point_max_iterations);
        result.fixed_point_iterations = local.iterations;
        result.fixed_point_residual = local.residual;
        if (!local.converged) {
            result.lrvb_status = "local_nonconvergence";
            result.lrvb_failed = true;
            return result;
        }
        Eigen::VectorXd abundance = topic_capacity.array()
            * local.shape.array() / local.rate.array();
        if (!(abundance.sum() > 0.0) || !abundance.allFinite()) {
            throw std::runtime_error("invalid_posterior_mean");
        }
        const Eigen::VectorXd mean = abundance / abundance.sum();
        fallback_probabilities = classifier.probabilities(mean);
        result.probabilities = fallback_probabilities;
        const std::vector<int32_t> candidates = select_candidates(
            result.probabilities, options.candidate_mass);
        result.candidate_count = candidates.size();
        double candidate_mass = 0.0;
        for (const int32_t component : candidates) {
            candidate_mass += result.probabilities(component);
        }
        result.held_fixed_tail_mass = std::max(0.0, 1.0 - candidate_mass);
        const int32_t dimension = static_cast<int32_t>(candidates.size()) - 1;
        const int32_t baseline = candidates.back();
        const int32_t topics = topic_capacity.size();
        Eigen::MatrixXd d(dimension, 2 * topics);
        Eigen::VectorXd contrast_mean(dimension);
        const Eigen::VectorXd all_logits = classifier.logits(mean);
        const double abundance_total = abundance.sum();
        for (int32_t contrast = 0; contrast < dimension; ++contrast) {
            const Eigen::RowVectorXd coefficient =
                (classifier.coefficients.row(candidates[contrast])
                    - classifier.coefficients.row(baseline))
                / classifier.temperature;
            const Eigen::VectorXd centered =
                coefficient.transpose().array() - coefficient.dot(mean);
            d.row(contrast).head(topics) =
                (centered.array() * topic_capacity.array()
                    / local.rate.array() / abundance_total).matrix();
            d.row(contrast).tail(topics) =
                (-centered.array() * topic_capacity.array()
                    * local.shape.array() / local.rate.array().square()
                    / abundance_total).matrix();
            contrast_mean(contrast) = all_logits(candidates[contrast])
                - all_logits(baseline);
        }

        GammaPoisCurvature curvature(local, document, expected_beta,
            feature_dispersion, posterior.exposure);
        std::vector<double> positive_diagonal;
        const Eigen::VectorXd& initial_diagonal = curvature.base_diagonal;
        for (Eigen::Index coordinate = 0;
                coordinate < initial_diagonal.size(); ++coordinate) {
            if (initial_diagonal(coordinate) > 0.0
                    && std::isfinite(initial_diagonal(coordinate))) {
                positive_diagonal.push_back(initial_diagonal(coordinate));
            }
        }
        if (positive_diagonal.empty()) {
            throw std::runtime_error("nonpositive_curvature");
        }
        const double scale = median(positive_diagonal);
        Eigen::MatrixXd solved(2 * topics, dimension);
        bool solved_all = false;
        const int32_t maximum_iterations = std::min(16 * topics, 1000);
        for (const double multiplier :
                {0.0, 1e-12, 1e-10, 1e-8, 1e-6}) {
            curvature.jitter = multiplier * scale;
            const Eigen::VectorXd diagonal = curvature.base_diagonal.array()
                + curvature.jitter;
            solved_all = true;
            int32_t maximum_used = 0;
            for (int32_t contrast = 0; contrast < dimension; ++contrast) {
                Eigen::VectorXd solution;
                int32_t iterations = 0;
                if (!preconditioned_cg(curvature, diagonal,
                        d.row(contrast).transpose(), options.cg_tolerance,
                        maximum_iterations, solution, iterations)) {
                    solved_all = false;
                    break;
                }
                solved.col(contrast) = solution;
                maximum_used = std::max(maximum_used, iterations);
            }
            if (solved_all) {
                result.cg_iterations = maximum_used;
                result.curvature_jitter = curvature.jitter;
                break;
            }
        }
        if (!solved_all) throw std::runtime_error("curvature_solve_failed");
        Eigen::MatrixXd covariance = d * solved;
        covariance = 0.5 * (covariance + covariance.transpose());
        (void)covariance_factor(covariance);
        const Eigen::VectorXd conditional = propagate_softmax_probabilities(
            contrast_mean, covariance, result.method);
        Eigen::VectorXd propagated = fallback_probabilities;
        for (int32_t index = 0; index < static_cast<int32_t>(candidates.size());
                ++index) {
            propagated(candidates[index]) =
                candidate_mass * conditional(index);
        }
        propagated /= propagated.sum();
        result.probabilities = std::move(propagated);
        result.lrvb_status = "ok";
    } catch (const std::exception& exception) {
        if (fallback_probabilities.size() != 0) {
            result.probabilities = fallback_probabilities;
        } else if (result.probabilities.size() == 0) {
            if (initial_composition != nullptr) {
                result.probabilities = classifier.probabilities(
                    *initial_composition);
            } else {
                const Eigen::VectorXd abundance = topic_capacity.array()
                    * posterior.shape.array() / posterior.rate.array();
                result.probabilities = classifier.probabilities(abundance);
            }
        }
        result.method = "plugin";
        result.lrvb_status = exception.what();
        result.lrvb_attempted = true;
        result.lrvb_failed = true;
    }
    return result;
}

PropagatedPrediction propagate_lda_from_composition(
        const Model& classifier,
        const Eigen::Ref<const Eigen::VectorXd>& composition,
        const Document& document,
        const Eigen::Ref<const Eigen::MatrixXd>& lda_allocation_kernel,
        double alpha, const PropagationOptions& options) {
    if (composition.size() != lda_allocation_kernel.rows()
            || !composition.allFinite()
            || (composition.array() < 0.0).any()
            || !(composition.sum() > 0.0)) {
        throw std::invalid_argument("Invalid LDA warm-start composition");
    }
    const double total = document.ct_tot >= 0.0
        ? document.ct_tot
        : std::accumulate(document.cnts.begin(), document.cnts.end(), 0.0);
    if (!(total >= 0.0) || !std::isfinite(total)) {
        throw std::invalid_argument("Invalid LDA warm-start document total");
    }
    const Eigen::VectorXd assigned = total * composition / composition.sum();
    return propagate_lda(classifier, assigned, document,
        lda_allocation_kernel, alpha, options);
}

PropagatedPrediction propagate_gamma_poisson_from_composition(
        const Model& classifier,
        const Eigen::Ref<const Eigen::VectorXd>& composition,
        const Document& document,
        const GammaPoisson4HexInterface& model,
        const PropagationOptions& options) {
    const Eigen::VectorXd& topic_capacity = model.getTopicCapacity();
    const double prior_shape = model.getThetaPriorShape();
    const Eigen::VectorXd prior_rate = model.getThetaPriorRate();
    const double size_factor = model.getSizeFactor();
    const int32_t topics = static_cast<int32_t>(topic_capacity.size());
    if (composition.size() != topics || prior_rate.size() != topics
            || !composition.allFinite()
            || (composition.array() < 0.0).any()
            || !(composition.sum() > 0.0)
            || !(prior_shape > 0.0) || !(size_factor > 0.0)
            || !std::isfinite(size_factor)
            || !topic_capacity.allFinite()
            || (topic_capacity.array() <= 0.0).any()) {
        throw std::invalid_argument(
            "Invalid Gamma-Poisson warm-start composition");
    }
    const double total = document.ct_tot >= 0.0
        ? document.ct_tot
        : std::accumulate(document.cnts.begin(), document.cnts.end(), 0.0);
    if (!(total >= 0.0) || !std::isfinite(total)) {
        throw std::invalid_argument(
            "Invalid Gamma-Poisson warm-start document total");
    }
    GammaPoissonDocumentPosterior posterior;
    posterior.exposure = total / size_factor;
    posterior.rate = prior_rate + posterior.exposure * topic_capacity;
    const Eigen::VectorXd normalized = composition / composition.sum();
    Eigen::VectorXd allocation = normalized.array()
        * posterior.rate.array() / topic_capacity.array();
    const double allocation_total = allocation.sum();
    if (!(allocation_total > 0.0) || !allocation.allFinite()) {
        throw std::invalid_argument(
            "Invalid Gamma-Poisson reconstructed warm start");
    }
    allocation /= allocation_total;
    posterior.shape = Eigen::VectorXd::Constant(topics, prior_shape)
        + total * allocation;
    return propagate_gamma_poisson(classifier, posterior, document, model,
        options, &normalized);
}

} // namespace punkst::partition_classifier
