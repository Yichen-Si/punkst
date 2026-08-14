#include "clustering_core/qda_projection.hpp"
#include "clustering_core/projection.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/QR>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace punkst::projection {
namespace {

struct SufficientStatistics {
    Eigen::VectorXd global_mean;
    Eigen::MatrixXd global_covariance;
    Eigen::MatrixXd topic_helmert;
    RowMajorMatrixXd means;
    std::vector<Eigen::MatrixXd> covariances;
    Eigen::VectorXi counts;
};

struct ObjectiveResult {
    double loss = std::numeric_limits<double>::infinity();
    double log_loss = std::numeric_limits<double>::infinity();
    double quartimax_score = 0.0;
    Eigen::MatrixXd gradient;
};

Eigen::MatrixXd symmetrize(const Eigen::Ref<const Eigen::MatrixXd>& value) {
    return 0.5 * (value + value.transpose());
}

void validate_labels(const Eigen::Ref<const Eigen::VectorXi>& labels,
        int32_t components, int32_t minimum_count, const char* name) {
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(components);
    for (Eigen::Index row = 0; row < labels.size(); ++row) {
        const int32_t label = labels(row);
        if (label < 0 || label >= components) {
            throw std::invalid_argument(std::string(name)
                + " labels are outside [0, components)");
        }
        ++counts(label);
    }
    if ((counts.array() < minimum_count).any()) {
        throw std::invalid_argument(std::string(name)
            + " has too few rows in at least one class");
    }
}

SufficientStatistics summarize(
        const Eigen::Ref<const RowMajorMatrixXd>& values,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        int32_t components) {
    const Eigen::Index n = values.rows();
    const Eigen::Index p = values.cols();
    SufficientStatistics out;
    out.topic_helmert = normalized_helmert(static_cast<int32_t>(p + 1));
    out.global_mean = values.colwise().mean();
    Eigen::MatrixXd centered = values;
    centered.rowwise() -= out.global_mean.transpose();
    out.global_covariance = symmetrize(
        centered.transpose() * centered / static_cast<double>(n - 1));
    out.means = RowMajorMatrixXd::Zero(components, p);
    out.covariances.resize(static_cast<size_t>(components),
        Eigen::MatrixXd::Zero(p, p));
    out.counts = Eigen::VectorXi::Zero(components);
    for (Eigen::Index row = 0; row < n; ++row) {
        out.means.row(labels(row)) += values.row(row);
        ++out.counts(labels(row));
    }
    for (int32_t component = 0; component < components; ++component) {
        out.means.row(component) /= out.counts(component);
    }
    for (Eigen::Index row = 0; row < n; ++row) {
        const int32_t component = labels(row);
        const Eigen::VectorXd difference = values.row(row).transpose()
            - out.means.row(component).transpose();
        out.covariances[static_cast<size_t>(component)].noalias() +=
            difference * difference.transpose();
    }
    for (int32_t component = 0; component < components; ++component) {
        out.covariances[static_cast<size_t>(component)] /=
            static_cast<double>(out.counts(component) - 1);
        out.covariances[static_cast<size_t>(component)] = symmetrize(
            out.covariances[static_cast<size_t>(component)]);
    }
    return out;
}

void positive_qr(const Eigen::Ref<const Eigen::MatrixXd>& input,
        Eigen::MatrixXd& q, Eigen::MatrixXd& r) {
    const Eigen::Index p = input.rows();
    const Eigen::Index d = input.cols();
    Eigen::HouseholderQR<Eigen::MatrixXd> qr(input);
    q = qr.householderQ() * Eigen::MatrixXd::Identity(p, d);
    r = qr.matrixQR().topRows(d).template triangularView<Eigen::Upper>();
    for (Eigen::Index axis = 0; axis < d; ++axis) {
        if (r(axis, axis) < 0.0) {
            q.col(axis) *= -1.0;
            r.row(axis) *= -1.0;
        }
    }
    const double tolerance = 128.0 * std::numeric_limits<double>::epsilon()
        * std::max(1.0, input.norm());
    if ((r.diagonal().array().abs() <= tolerance).any()) {
        throw std::runtime_error("QDA projection QR factor is rank deficient");
    }
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

double deterministic_normal(uint64_t seed, uint64_t counter) {
    constexpr double scale = 1.0 / 9007199254740992.0;
    const double first = (static_cast<double>(
        splitmix64(seed ^ (2 * counter)) >> 11) + 0.5) * scale;
    const double second = (static_cast<double>(
        splitmix64(seed ^ (2 * counter + 1)) >> 11) + 0.5) * scale;
    return std::sqrt(-2.0 * std::log(first))
        * std::cos(2.0 * std::acos(-1.0) * second);
}

Eigen::MatrixXd fisher_initializer(const SufficientStatistics& statistics,
        int32_t dimensions, uint64_t random_seed) {
    const Eigen::Index p = statistics.means.cols();
    const int32_t components = static_cast<int32_t>(statistics.counts.size());
    Eigen::MatrixXd within = Eigen::MatrixXd::Zero(p, p);
    Eigen::MatrixXd between = Eigen::MatrixXd::Zero(p, p);
    const double total = statistics.counts.cast<double>().sum();
    for (int32_t component = 0; component < components; ++component) {
        within.noalias() += (statistics.counts(component) - 1.0)
            * statistics.covariances[static_cast<size_t>(component)];
        const Eigen::VectorXd difference =
            statistics.means.row(component).transpose()
            - statistics.global_mean;
        between.noalias() += statistics.counts(component)
            * difference * difference.transpose();
    }
    within /= std::max(1.0, total - components);
    between /= total;
    const double scale = within.trace() > 0.0
        ? within.trace() / static_cast<double>(p) : 1.0;
    within.diagonal().array() += 1e-5 * scale;
    Eigen::GeneralizedSelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        symmetrize(between), symmetrize(within));
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("QDA Fisher initialization failed");
    }
    Eigen::MatrixXd initial(p, dimensions);
    uint64_t counter = 0;
    for (Eigen::Index row = 0; row < p; ++row) {
        for (int32_t axis = 0; axis < dimensions; ++axis) {
            initial(row, axis) = deterministic_normal(random_seed, counter++);
        }
    }
    const int32_t fisher_dimensions = std::min({dimensions,
        components - 1, static_cast<int32_t>(p)});
    for (int32_t axis = 0; axis < fisher_dimensions; ++axis) {
        Eigen::VectorXd direction = solver.eigenvectors().col(p - 1 - axis);
        Eigen::Index pivot = 0;
        direction.cwiseAbs().maxCoeff(&pivot);
        if (direction(pivot) < 0.0) direction *= -1.0;
        initial.col(axis) = direction;
    }
    return initial;
}

ObjectiveResult objective(const Eigen::Ref<const Eigen::MatrixXd>& q,
        const Eigen::Ref<const RowMajorMatrixXd>& values,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const SufficientStatistics& statistics,
        const QdaProjectionOptions& options, bool need_gradient) {
    const Eigen::Index n = values.rows();
    const Eigen::Index p = values.cols();
    const Eigen::Index d = q.cols();
    const int32_t components = static_cast<int32_t>(statistics.counts.size());
    const Eigen::MatrixXd projected = values * q;
    const RowMajorMatrixXd means = statistics.means * q;
    const double raw_scale = (q.transpose()
        * statistics.global_covariance * q).trace() / d;
    const double scale = std::max(1e-6, raw_scale);
    const double ridge_scale = (1.0 + options.covariance_shrinkage)
        * options.ridge * scale;
    const double constant = d * std::log(2.0 * std::acos(-1.0));

    Eigen::MatrixXd logits(n, components);
    std::vector<Eigen::MatrixXd> precisions(components);
    std::vector<Eigen::MatrixXd> weighted_differences(components);
    std::vector<Eigen::MatrixXd> base_covariances(components);
    tbb::global_control thread_control(
        tbb::global_control::max_allowed_parallelism,
        static_cast<size_t>(options.n_threads));
    tbb::parallel_for(int32_t{0}, components, [&](int32_t component) {
        Eigen::MatrixXd base = (1.0 - options.covariance_shrinkage)
            * statistics.covariances[static_cast<size_t>(component)]
            + options.covariance_shrinkage * statistics.global_covariance;
        Eigen::MatrixXd covariance = symmetrize(q.transpose() * base * q);
        covariance.diagonal().array() += ridge_scale;
        Eigen::LLT<Eigen::MatrixXd> solver(covariance);
        if (solver.info() != Eigen::Success) {
            throw std::runtime_error(
                "QDA projected covariance is not positive definite");
        }
        const double log_determinant = 2.0
            * solver.matrixL().toDenseMatrix().diagonal().array().log().sum();
        Eigen::MatrixXd precision = solver.solve(
            Eigen::MatrixXd::Identity(d, d));
        Eigen::MatrixXd difference = projected;
        difference.rowwise() -= means.row(component);
        Eigen::MatrixXd weighted = difference * precision;
        logits.col(component) = Eigen::VectorXd::Constant(n,
            std::log(static_cast<double>(statistics.counts(component))
                / statistics.counts.cast<double>().sum())
            - 0.5 * (constant + log_determinant));
        logits.col(component).array() -= 0.5
            * (difference.array() * weighted.array()).rowwise().sum();
        precisions[static_cast<size_t>(component)] = std::move(precision);
        weighted_differences[static_cast<size_t>(component)] =
            std::move(weighted);
        base_covariances[static_cast<size_t>(component)] = std::move(base);
    });

    ObjectiveResult out;
    Eigen::MatrixXd derivatives(n, components);
    out.log_loss = 0.0;
    for (Eigen::Index row = 0; row < n; ++row) {
        const double maximum = logits.row(row).maxCoeff();
        Eigen::ArrayXd probabilities =
            (logits.row(row).array() - maximum).exp();
        probabilities /= probabilities.sum();
        const double selected = probabilities(labels(row));
        if (!(selected > 0.0) || !std::isfinite(selected)) {
            throw std::runtime_error("Non-finite QDA conditional loss");
        }
        out.log_loss -= std::log(selected) / static_cast<double>(n);
        derivatives.row(row) = probabilities.matrix().transpose()
            / static_cast<double>(n);
        derivatives(row, labels(row)) -= 1.0 / static_cast<double>(n);
    }
    const Eigen::MatrixXd topic_contrasts =
        statistics.topic_helmert.transpose() * q;
    out.quartimax_score = quartimax_objective(topic_contrasts)
        / static_cast<double>(d);
    out.loss = out.log_loss
        - options.sparsity_strength * out.quartimax_score;
    if (!need_gradient) return out;

    std::vector<Eigen::MatrixXd> projected_gradients(components);
    std::vector<Eigen::MatrixXd> q_gradients(components);
    std::vector<double> scale_gradients(components, 0.0);
    tbb::parallel_for(int32_t{0}, components, [&](int32_t component) {
        const Eigen::VectorXd derivative = derivatives.col(component);
        const Eigen::MatrixXd& weighted =
            weighted_differences[static_cast<size_t>(component)];
        projected_gradients[static_cast<size_t>(component)] =
            -(weighted.array().colwise() * derivative.array()).matrix();
        const Eigen::VectorXd mean_gradient =
            weighted.transpose() * derivative;
        Eigen::MatrixXd q_gradient = statistics.means.row(component).transpose()
            * mean_gradient.transpose();
        const Eigen::MatrixXd scaled_weighted =
            (weighted.array().colwise() * derivative.array()).matrix();
        Eigen::MatrixXd covariance_gradient = 0.5 * (
            weighted.transpose() * scaled_weighted
            - derivative.sum()
                * precisions[static_cast<size_t>(component)]);
        covariance_gradient = symmetrize(covariance_gradient);
        q_gradient.noalias() += 2.0
            * base_covariances[static_cast<size_t>(component)]
            * q * covariance_gradient;
        q_gradients[static_cast<size_t>(component)] = std::move(q_gradient);
        scale_gradients[static_cast<size_t>(component)] =
            (1.0 + options.covariance_shrinkage)
            * options.ridge * covariance_gradient.trace();
    });
    Eigen::MatrixXd projected_gradient = Eigen::MatrixXd::Zero(n, d);
    out.gradient = Eigen::MatrixXd::Zero(p, d);
    double scale_gradient = 0.0;
    for (int32_t component = 0; component < components; ++component) {
        projected_gradient += projected_gradients[static_cast<size_t>(component)];
        out.gradient += q_gradients[static_cast<size_t>(component)];
        scale_gradient += scale_gradients[static_cast<size_t>(component)];
    }
    out.gradient.noalias() += values.transpose() * projected_gradient;
    if (raw_scale > 1e-6) {
        out.gradient.noalias() += (2.0 * scale_gradient / d)
            * statistics.global_covariance * q;
    }
    if (options.sparsity_strength > 0.0) {
        out.gradient.noalias() -=
            (4.0 * options.sparsity_strength / static_cast<double>(d))
            * statistics.topic_helmert
            * topic_contrasts.array().cube().matrix();
    }
    return out;
}

Eigen::MatrixXd qr_backward(const Eigen::Ref<const Eigen::MatrixXd>& q,
        const Eigen::Ref<const Eigen::MatrixXd>& r,
        const Eigen::Ref<const Eigen::MatrixXd>& q_gradient) {
    Eigen::MatrixXd projected = q_gradient
        - q * (q.transpose() * q_gradient);
    return r.template triangularView<Eigen::Upper>()
        .solve(projected.transpose()).transpose();
}

void canonicalize(Eigen::MatrixXd& q,
        const SufficientStatistics& statistics) {
    const Eigen::Index d = q.cols();
    Eigen::MatrixXd between = Eigen::MatrixXd::Zero(q.rows(), q.rows());
    const double total = statistics.counts.cast<double>().sum();
    for (Eigen::Index component = 0;
            component < statistics.counts.size(); ++component) {
        const Eigen::VectorXd difference =
            statistics.means.row(component).transpose()
            - statistics.global_mean;
        between.noalias() += statistics.counts(component)
            * difference * difference.transpose() / total;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
        symmetrize(q.transpose() * between * q));
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error("QDA axis canonicalization failed");
    }
    Eigen::MatrixXd rotation(d, d);
    for (Eigen::Index axis = 0; axis < d; ++axis) {
        rotation.col(axis) = solver.eigenvectors().col(d - 1 - axis);
    }
    q *= rotation;

    const Eigen::MatrixXd helmert = normalized_helmert(
        static_cast<int32_t>(q.rows() + 1));
    Eigen::MatrixXd topic_contrasts = helmert.transpose() * q;
    quartimax_rotate(q, topic_contrasts);

    std::vector<Eigen::Index> order(static_cast<size_t>(d));
    std::vector<double> scatter(static_cast<size_t>(d));
    std::vector<double> concentration(static_cast<size_t>(d));
    for (Eigen::Index axis = 0; axis < d; ++axis) {
        order[static_cast<size_t>(axis)] = axis;
        scatter[static_cast<size_t>(axis)] =
            q.col(axis).dot(between * q.col(axis));
        concentration[static_cast<size_t>(axis)] =
            topic_contrasts.col(axis).array().square().square().sum();
    }
    std::stable_sort(order.begin(), order.end(),
        [&](Eigen::Index left, Eigen::Index right) {
            const double left_scatter = scatter[static_cast<size_t>(left)];
            const double right_scatter = scatter[static_cast<size_t>(right)];
            if (left_scatter != right_scatter) {
                return left_scatter > right_scatter;
            }
            const double left_concentration =
                concentration[static_cast<size_t>(left)];
            const double right_concentration =
                concentration[static_cast<size_t>(right)];
            if (left_concentration != right_concentration) {
                return left_concentration > right_concentration;
            }
            return left < right;
        });
    Eigen::MatrixXd ordered_q(q.rows(), d);
    Eigen::MatrixXd ordered_contrasts(topic_contrasts.rows(), d);
    for (Eigen::Index axis = 0; axis < d; ++axis) {
        const Eigen::Index source = order[static_cast<size_t>(axis)];
        ordered_q.col(axis) = q.col(source);
        ordered_contrasts.col(axis) = topic_contrasts.col(source);
    }
    q = std::move(ordered_q);
    topic_contrasts = std::move(ordered_contrasts);
    for (Eigen::Index axis = 0; axis < d; ++axis) {
        Eigen::Index pivot = 0;
        topic_contrasts.col(axis).cwiseAbs().maxCoeff(&pivot);
        if (topic_contrasts(pivot, axis) < 0.0) q.col(axis) *= -1.0;
    }
}

} // namespace

QdaProjectionResult fit_qda_projection(
    const Eigen::Ref<const RowMajorMatrixXd>& training,
    const Eigen::Ref<const Eigen::VectorXi>& training_labels,
    const Eigen::Ref<const RowMajorMatrixXd>& validation,
    const Eigen::Ref<const Eigen::VectorXi>& validation_labels,
    int32_t components, const QdaProjectionOptions& options) {
    if (training.rows() <= 0 || validation.rows() <= 0
        || training.cols() <= 1 || training.cols() != validation.cols()
        || training.rows() != training_labels.size()
        || validation.rows() != validation_labels.size()
        || !training.allFinite() || !validation.allFinite()
        || components < 2 || options.dimensions <= 0
        || options.dimensions >= training.cols() || options.epochs <= 0
        || options.restarts <= 0 || options.evaluate_every <= 0
        || options.n_threads <= 0
        || options.patience_checks <= 0 || !(options.learning_rate > 0.0)
        || !(options.covariance_shrinkage >= 0.0)
        || !(options.covariance_shrinkage <= 1.0)
        || !(options.ridge > 0.0)
        || !(options.improvement_tolerance >= 0.0)
        || !(options.sparsity_strength >= 0.0)
        || !std::isfinite(options.sparsity_strength)) {
        throw std::invalid_argument("Invalid QDA projection input or options");
    }
    validate_labels(training_labels, components, 2, "Training");
    validate_labels(validation_labels, components, 1, "Validation");
    const SufficientStatistics statistics = summarize(
        training, training_labels, components);
    const Eigen::Index p = training.cols();
    const Eigen::Index d = options.dimensions;
    QdaProjectionResult best;
    best.validation_loss = std::numeric_limits<double>::infinity();
    for (int32_t restart = 0; restart < options.restarts; ++restart) {
        const uint64_t random_seed = splitmix64(
            static_cast<uint64_t>(static_cast<uint32_t>(options.seed))
            ^ (static_cast<uint64_t>(options.dimensions) << 32)
            ^ static_cast<uint64_t>(restart) ^ 0x514441ULL);
        Eigen::MatrixXd b;
        if (restart == 0) {
            b = fisher_initializer(statistics, options.dimensions, random_seed);
            uint64_t counter = static_cast<uint64_t>(p * d);
            for (Eigen::Index row = 0; row < p; ++row) {
                for (Eigen::Index axis = 0; axis < d; ++axis) {
                    b(row, axis) += 1e-3
                        * deterministic_normal(random_seed, counter++);
                }
            }
        } else {
            b.resize(p, d);
            uint64_t counter = 0;
            for (Eigen::Index row = 0; row < p; ++row) {
                for (Eigen::Index axis = 0; axis < d; ++axis) {
                    b(row, axis) = deterministic_normal(
                        random_seed, counter++);
                }
            }
        }
        Eigen::MatrixXd first_moment = Eigen::MatrixXd::Zero(p, d);
        Eigen::MatrixXd second_moment = Eigen::MatrixXd::Zero(p, d);
        double restart_best = std::numeric_limits<double>::infinity();
        int32_t no_improvement = 0;
        for (int32_t epoch = 0; epoch < options.epochs; ++epoch) {
            Eigen::MatrixXd q, r;
            positive_qr(b, q, r);
            const ObjectiveResult fit = objective(q, training,
                training_labels, statistics, options, true);
            const Eigen::MatrixXd gradient = qr_backward(q, r, fit.gradient);
            first_moment = 0.9 * first_moment + 0.1 * gradient;
            second_moment = 0.999 * second_moment
                + 0.001 * gradient.array().square().matrix();
            const double first_correction = 1.0 - std::pow(0.9, epoch + 1);
            const double second_correction = 1.0 - std::pow(0.999, epoch + 1);
            b.array() -= options.learning_rate
                * (first_moment.array() / first_correction)
                / ((second_moment.array() / second_correction).sqrt() + 1e-8);

            if (epoch % options.evaluate_every == 0
                || epoch == options.epochs - 1) {
                positive_qr(b, q, r);
                const double validation_loss = objective(q, validation,
                    validation_labels, statistics, options, false).loss;
                if (validation_loss < restart_best
                        - options.improvement_tolerance) {
                    restart_best = validation_loss;
                    no_improvement = 0;
                    if (validation_loss < best.validation_loss) {
                        best.projection = q;
                        best.validation_loss = validation_loss;
                        best.restart = restart;
                        best.epoch = epoch;
                    }
                } else if (++no_improvement >= options.patience_checks) {
                    break;
                }
            }
        }
    }
    if (best.restart < 0 || best.projection.size() == 0) {
        throw std::runtime_error("QDA projection optimization produced no result");
    }
    canonicalize(best.projection, statistics);
    const ObjectiveResult training_fit = objective(best.projection, training,
        training_labels, statistics, options, false);
    const ObjectiveResult validation_fit = objective(best.projection,
        validation, validation_labels, statistics, options, false);
    best.training_loss = training_fit.log_loss;
    best.validation_loss = validation_fit.log_loss;
    best.quartimax_score = training_fit.quartimax_score;
    best.training_objective = training_fit.loss;
    best.validation_objective = validation_fit.loss;
    return best;
}

double qda_projection_log_loss(
    const Eigen::Ref<const Eigen::MatrixXd>& projection,
    const Eigen::Ref<const RowMajorMatrixXd>& training,
    const Eigen::Ref<const Eigen::VectorXi>& training_labels,
    const Eigen::Ref<const RowMajorMatrixXd>& evaluation,
    const Eigen::Ref<const Eigen::VectorXi>& evaluation_labels,
    int32_t components, const QdaProjectionOptions& options) {
    if (training.rows() <= 0 || evaluation.rows() <= 0
        || training.cols() <= 1 || training.cols() != evaluation.cols()
        || projection.rows() != training.cols() || projection.cols() <= 0
        || projection.cols() >= projection.rows()
        || training.rows() != training_labels.size()
        || evaluation.rows() != evaluation_labels.size()
        || !training.allFinite() || !evaluation.allFinite()
        || !projection.allFinite() || components < 2
        || options.n_threads <= 0
        || !(options.covariance_shrinkage >= 0.0)
        || !(options.covariance_shrinkage <= 1.0)
        || !(options.ridge > 0.0)
        || !(options.sparsity_strength >= 0.0)
        || !std::isfinite(options.sparsity_strength)) {
        throw std::invalid_argument("Invalid QDA projection evaluation input");
    }
    validate_labels(training_labels, components, 2, "Training");
    validate_labels(evaluation_labels, components, 1, "Evaluation");
    const SufficientStatistics statistics = summarize(
        training, training_labels, components);
    return objective(projection, evaluation, evaluation_labels,
        statistics, options, false).log_loss;
}

} // namespace punkst::projection
