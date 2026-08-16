#include "partition_classifier.hpp"

#include "lbfgs_history.hpp"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace punkst::partition_classifier {
namespace {

void report_progress(const FitOptions& options, const std::string& message) {
    if (options.progress_callback) options.progress_callback(message);
}

struct Objective {
    const RowMajorMatrixXd& x;
    const Eigen::VectorXi& y;
    const Eigen::VectorXd& weights;
    const std::vector<int32_t>& rows;
    const Eigen::MatrixXd& class_helmert;
    int32_t classes = 0;
    int32_t predictors = 0;
    double ridge = 0.0;

    double operator()(const Eigen::VectorXd& parameters,
            Eigen::VectorXd* gradient) const {
        const int32_t class_coordinates = classes - 1;
        const Eigen::Map<const Eigen::VectorXd> intercept(
            parameters.data(), class_coordinates);
        const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
            Eigen::Dynamic, Eigen::RowMajor>> coefficients(
                parameters.data() + class_coordinates,
                class_coordinates, predictors);
        Eigen::VectorXd intercept_gradient = Eigen::VectorXd::Zero(
            class_coordinates);
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
            coefficient_gradient = decltype(coefficient_gradient)::Zero(
                class_coordinates, predictors);
        double total_weight = 0.0;
        double loss = 0.0;
        Eigen::VectorXd coordinate_logits(class_coordinates);
        Eigen::VectorXd logits(classes);
        Eigen::VectorXd probabilities(classes);
        for (const int32_t row : rows) {
            const double weight = weights(row);
            coordinate_logits.noalias() = intercept
                + coefficients * x.row(row).transpose();
            logits.noalias() = class_helmert.transpose() * coordinate_logits;
            const double maximum = logits.maxCoeff();
            probabilities = (logits.array() - maximum).exp();
            const double normalization = probabilities.sum();
            probabilities /= normalization;
            loss += weight * (maximum + std::log(normalization)
                - logits(y(row)));
            probabilities(y(row)) -= 1.0;
            const Eigen::VectorXd coordinate_error =
                class_helmert * probabilities;
            intercept_gradient.noalias() += weight * coordinate_error;
            coefficient_gradient.noalias() += weight * coordinate_error
                * x.row(row);
            total_weight += weight;
        }
        if (!(total_weight > 0.0)) {
            throw std::invalid_argument("Classifier objective has no weight");
        }
        loss /= total_weight;
        loss += 0.5 * ridge * coefficients.squaredNorm();
        if (gradient != nullptr) {
            gradient->resize(parameters.size());
            Eigen::Map<Eigen::VectorXd> output_intercept(
                gradient->data(), class_coordinates);
            Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic,
                Eigen::Dynamic, Eigen::RowMajor>> output_coefficients(
                    gradient->data() + class_coordinates,
                    class_coordinates, predictors);
            output_intercept = intercept_gradient / total_weight;
            output_coefficients = coefficient_gradient / total_weight
                + ridge * coefficients;
        }
        return loss;
    }
};

Eigen::VectorXd optimize_lbfgs(const Objective& objective,
        Eigen::VectorXd parameters, const FitOptions& options) {
    Eigen::VectorXd gradient;
    double value = objective(parameters, &gradient);
    punkst::LbfgsHistory<Eigen::VectorXd> history(options.lbfgs_history);
    for (int32_t iteration = 0; iteration < options.max_iterations;
            ++iteration) {
        if (!std::isfinite(value) || !gradient.allFinite()) {
            throw std::runtime_error("Nonfinite classifier objective");
        }
        if (gradient.lpNorm<Eigen::Infinity>()
                <= options.gradient_tolerance) return parameters;

        Eigen::VectorXd direction = -history.apply(gradient);
        double directional_derivative = gradient.dot(direction);
        if (!(directional_derivative < 0.0)
                || !std::isfinite(directional_derivative)) {
            direction = -gradient;
            directional_derivative = -gradient.squaredNorm();
            history.clear();
        }

        double step = 1.0;
        Eigen::VectorXd candidate;
        Eigen::VectorXd candidate_gradient;
        double candidate_value = std::numeric_limits<double>::infinity();
        for (int32_t line_search = 0; line_search < 40; ++line_search) {
            candidate = parameters + step * direction;
            candidate_value = objective(candidate, &candidate_gradient);
            if (std::isfinite(candidate_value)
                    && candidate_value <= value
                        + 1e-4 * step * directional_derivative) break;
            step *= 0.5;
        }
        if (!(step > std::ldexp(1.0, -40))
                || !std::isfinite(candidate_value)) {
            throw std::runtime_error("Classifier L-BFGS line search failed");
        }
        Eigen::VectorXd s = candidate - parameters;
        Eigen::VectorXd y_delta = candidate_gradient - gradient;
        history.update(std::move(s), std::move(y_delta));
        parameters = std::move(candidate);
        gradient = std::move(candidate_gradient);
        value = candidate_value;
    }
    if (gradient.lpNorm<Eigen::Infinity>()
            <= options.gradient_tolerance) return parameters;
    std::ostringstream message;
    message << "Classifier L-BFGS did not converge after "
        << options.max_iterations << " iterations; final gradient infinity norm "
        << gradient.lpNorm<Eigen::Infinity>();
    throw std::runtime_error(message.str());
}

Eigen::VectorXd fit_parameters(const RowMajorMatrixXd& x,
        const Eigen::VectorXi& labels, const Eigen::VectorXd& weights,
        const std::vector<int32_t>& rows, int32_t classes, double ridge,
        const FitOptions& options) {
    const Eigen::MatrixXd class_helmert = normalized_helmert(classes);
    Objective objective{x, labels, weights, rows, class_helmert, classes,
        static_cast<int32_t>(x.cols()), ridge};
    Eigen::VectorXd parameters = Eigen::VectorXd::Zero(
        (classes - 1) * (x.cols() + 1));
    Eigen::VectorXd class_weight = Eigen::VectorXd::Zero(classes);
    for (const int32_t row : rows) class_weight(labels(row)) += weights(row);
    class_weight = (class_weight.array() + 0.5)
        / (class_weight.sum() + 0.5 * classes);
    parameters.head(classes - 1) = class_helmert
        * class_weight.array().log().matrix();
    return optimize_lbfgs(objective, std::move(parameters), options);
}

RowMajorMatrixXd parameter_logits(const RowMajorMatrixXd& x,
        const std::vector<int32_t>& rows, const Eigen::VectorXd& parameters,
        int32_t classes) {
    const int32_t class_coordinates = classes - 1;
    const Eigen::Map<const Eigen::VectorXd> intercept(
        parameters.data(), class_coordinates);
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
        Eigen::Dynamic, Eigen::RowMajor>> coefficients(
            parameters.data() + class_coordinates,
            class_coordinates, x.cols());
    const Eigen::MatrixXd class_helmert = normalized_helmert(classes);
    RowMajorMatrixXd output(rows.size(), classes);
    for (size_t index = 0; index < rows.size(); ++index) {
        output.row(index) = (class_helmert.transpose()
            * (intercept + coefficients
                * x.row(rows[index]).transpose())).transpose();
    }
    return output;
}

Model make_model(const Eigen::VectorXd& parameters,
        const std::vector<std::string>& topics,
        const std::vector<std::string>& classes, double ridge) {
    const int32_t class_count = static_cast<int32_t>(classes.size());
    const int32_t topic_count = static_cast<int32_t>(topics.size());
    const Eigen::MatrixXd topic_helmert = normalized_helmert(topic_count);
    const Eigen::MatrixXd class_helmert = normalized_helmert(class_count);
    const Eigen::Map<const Eigen::VectorXd> contrast_intercept(
        parameters.data(), class_count - 1);
    const Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic,
        Eigen::Dynamic, Eigen::RowMajor>> contrast_coefficients(
            parameters.data() + class_count - 1,
            class_count - 1, topic_count - 1);
    Model model;
    model.topics = topics;
    model.classes = classes;
    model.intercepts = class_helmert.transpose() * contrast_intercept;
    model.coefficients = class_helmert.transpose()
        * contrast_coefficients * topic_helmert;
    model.ridge = ridge;
    return model;
}

double temperature_loss(double log_temperature,
        const Eigen::Ref<const RowMajorMatrixXd>& logits,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const Eigen::Ref<const Eigen::VectorXd>& weights) {
    const double inverse_temperature = std::exp(-log_temperature);
    double loss = 0.0;
    double total_weight = 0.0;
    for (Eigen::Index row = 0; row < logits.rows(); ++row) {
        const Eigen::ArrayXd scaled = logits.row(row).array()
            * inverse_temperature;
        const double maximum = scaled.maxCoeff();
        loss += weights(row) * (maximum
            + std::log((scaled - maximum).exp().sum())
            - scaled(labels(row)));
        total_weight += weights(row);
    }
    return loss / total_weight;
}

std::vector<int32_t> all_rows(int32_t count) {
    std::vector<int32_t> out(static_cast<size_t>(count));
    std::iota(out.begin(), out.end(), 0);
    return out;
}

uint64_t identifier_hash(const std::string& identifier, uint64_t seed) {
    uint64_t value = 1469598103934665603ULL ^ seed;
    for (const unsigned char byte : identifier) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

struct FoldAssignment {
    Eigen::VectorXi by_row;
    int32_t folds = 0;
};

FoldAssignment make_stratified_folds(
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const std::vector<int32_t>& rows,
        const std::vector<std::string>& identifiers,
        int32_t classes, int32_t requested_folds, uint64_t seed) {
    std::vector<std::vector<int32_t>> by_class(
        static_cast<size_t>(classes));
    for (const int32_t row : rows) {
        if (row < 0 || row >= labels.size() || labels(row) < 0
                || labels(row) >= classes) {
            throw std::invalid_argument("Invalid crossfit row or label");
        }
        by_class[static_cast<size_t>(labels(row))].push_back(row);
    }
    int32_t folds = requested_folds;
    for (const auto& class_rows : by_class) {
        folds = std::min(folds, static_cast<int32_t>(class_rows.size()));
    }
    if (folds < 2) {
        throw std::invalid_argument(
            "Crossfit requires at least two represented rows per class");
    }
    FoldAssignment output;
    output.by_row = Eigen::VectorXi::Constant(labels.size(), -1);
    output.folds = folds;
    for (int32_t component = 0; component < classes; ++component) {
        auto& class_rows = by_class[static_cast<size_t>(component)];
        std::sort(class_rows.begin(), class_rows.end(),
            [&](int32_t left, int32_t right) {
                const uint64_t left_hash = identifier_hash(
                    identifiers[static_cast<size_t>(left)], seed);
                const uint64_t right_hash = identifier_hash(
                    identifiers[static_cast<size_t>(right)], seed);
                return left_hash == right_hash ? left < right
                    : left_hash < right_hash;
            });
        for (size_t index = 0; index < class_rows.size(); ++index) {
            output.by_row(class_rows[index]) =
                static_cast<int32_t>(index % folds);
        }
    }
    return output;
}

struct NestedModelFit {
    Model model;
    int32_t inner_folds = 0;
};

NestedModelFit fit_nested_model(const RowMajorMatrixXd& x,
        const Eigen::VectorXi& labels, const Eigen::VectorXd& weights,
        const std::vector<int32_t>& rows,
        const std::vector<std::string>& identifiers,
        const std::vector<std::string>& topics,
        const std::vector<std::string>& classes, uint64_t seed,
        const FitOptions& options) {
    const int32_t class_count = static_cast<int32_t>(classes.size());
    const FoldAssignment assignment = make_stratified_folds(
        labels, rows, identifiers, class_count, options.folds, seed);
    double best_loss = std::numeric_limits<double>::infinity();
    double selected_ridge = 0.0;
    RowMajorMatrixXd selected_oof;
    Eigen::VectorXi compact_labels(rows.size());
    Eigen::VectorXd compact_weights(rows.size());
    for (size_t index = 0; index < rows.size(); ++index) {
        compact_labels(index) = labels(rows[index]);
        compact_weights(index) = weights(rows[index]);
    }
    for (const double ridge : options.ridge_grid) {
        if (!(ridge >= 0.0) || !std::isfinite(ridge)) {
            throw std::invalid_argument(
                "Ridge grid must be finite and nonnegative");
        }
        RowMajorMatrixXd oof(rows.size(), class_count);
        for (int32_t fold = 0; fold < assignment.folds; ++fold) {
            std::vector<int32_t> training;
            std::vector<int32_t> validation;
            for (const int32_t row : rows) {
                (assignment.by_row(row) == fold ? validation : training)
                    .push_back(row);
            }
            const Eigen::VectorXd parameters = fit_parameters(
                x, labels, weights, training, class_count, ridge, options);
            const RowMajorMatrixXd logits = parameter_logits(
                x, validation, parameters, class_count);
            size_t output_index = 0;
            for (size_t index = 0; index < rows.size(); ++index) {
                if (assignment.by_row(rows[index]) == fold) {
                    oof.row(index) = logits.row(output_index++);
                }
            }
        }
        const double loss = evaluate(probabilities_from_logits(oof, 1.0),
            compact_labels, compact_weights).log_loss;
        if (loss < best_loss) {
            best_loss = loss;
            selected_ridge = ridge;
            selected_oof = std::move(oof);
        }
    }
    const double temperature = fit_temperature(
        selected_oof, compact_labels, compact_weights);
    const Eigen::VectorXd parameters = fit_parameters(x, labels, weights,
        rows, class_count, selected_ridge, options);
    NestedModelFit output;
    output.model = make_model(parameters, topics, classes, selected_ridge);
    output.model.temperature = temperature;
    output.model.folds = assignment.folds;
    output.inner_folds = assignment.folds;
    output.model.validate();
    return output;
}

void require_unique_nonempty(const std::vector<std::string>& values,
        const char* what) {
    std::unordered_set<std::string> seen;
    for (const std::string& value : values) {
        if (value.empty() || !seen.insert(value).second) {
            throw std::invalid_argument(std::string(what)
                + " must be nonempty and unique");
        }
    }
}

} // namespace

namespace testing {

void run_classifier_gradient_test() {
    RowMajorMatrixXd x(5, 3);
    x << 0.2, -0.4, 0.7,
         -0.1, 0.5, 0.3,
         0.8, 0.1, -0.2,
         -0.5, 0.6, 0.4,
         0.3, -0.2, 0.9;
    Eigen::VectorXi labels(5);
    labels << 0, 1, 2, 1, 0;
    Eigen::VectorXd weights(5);
    weights << 1.0, 0.7, 1.3, 0.9, 1.1;
    const std::vector<int32_t> rows{0, 1, 2, 3, 4};
    const Eigen::MatrixXd helmert = normalized_helmert(3);
    const Objective objective{x, labels, weights, rows, helmert, 3, 3, 0.07};
    Eigen::VectorXd parameters(8);
    parameters << 0.1, -0.2, 0.3, -0.1, 0.2, -0.4, 0.05, 0.15;
    Eigen::VectorXd analytic;
    (void)objective(parameters, &analytic);
    Eigen::VectorXd numerical(parameters.size());
    const double step = 1e-6;
    for (Eigen::Index coordinate = 0;
            coordinate < parameters.size(); ++coordinate) {
        Eigen::VectorXd plus = parameters;
        Eigen::VectorXd minus = parameters;
        plus(coordinate) += step;
        minus(coordinate) -= step;
        numerical(coordinate) =
            (objective(plus, nullptr) - objective(minus, nullptr))
            / (2.0 * step);
    }
    const double scale = std::max(1.0, numerical.cwiseAbs().maxCoeff());
    if ((analytic - numerical).cwiseAbs().maxCoeff() > 1e-6 * scale) {
        throw std::runtime_error(
            "Classifier analytic gradient failed finite-difference test");
    }
}

} // namespace testing

Eigen::VectorXd Model::logits(
        const Eigen::Ref<const Eigen::VectorXd>& composition) const {
    if (composition.size() != coefficients.cols()
            || !composition.allFinite()
            || (composition.array() < 0.0).any()
            || !(composition.sum() > 0.0)) {
        throw std::invalid_argument("Invalid classifier composition");
    }
    return (intercepts + coefficients
        * (composition / composition.sum())) / temperature;
}

Eigen::VectorXd Model::probabilities(
        const Eigen::Ref<const Eigen::VectorXd>& composition) const {
    Eigen::VectorXd output = logits(composition);
    softmaxInPlace(output);
    return output;
}

void Model::validate(double tolerance) const {
    require_unique_nonempty(topics, "Classifier topics");
    require_unique_nonempty(classes, "Classifier classes");
    if (topics.size() < 2 || classes.size() < 2
            || intercepts.size() != static_cast<Eigen::Index>(classes.size())
            || coefficients.rows()
                != static_cast<Eigen::Index>(classes.size())
            || coefficients.cols()
                != static_cast<Eigen::Index>(topics.size())
            || !intercepts.allFinite() || !coefficients.allFinite()
            || !(temperature > 0.0) || !std::isfinite(temperature)
            || !(ridge >= 0.0) || !std::isfinite(ridge)) {
        throw std::invalid_argument("Invalid partition classifier model");
    }
    const double scale = std::max({1.0, intercepts.cwiseAbs().maxCoeff(),
        coefficients.cwiseAbs().maxCoeff()});
    if (std::abs(intercepts.sum()) > tolerance * scale
            || coefficients.rowwise().sum().cwiseAbs().maxCoeff()
                > tolerance * scale
            || coefficients.colwise().sum().cwiseAbs().maxCoeff()
                > tolerance * scale) {
        throw std::invalid_argument(
            "Partition classifier coefficients are not centered");
    }
}

void Model::write(const std::string& path) const {
    validate();
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write classifier: " + path);
    output << "#partition_classifier\t" << MODEL_SCHEMA_VERSION << '\n'
        << "#ridge\t" << std::scientific
        << std::setprecision(std::numeric_limits<double>::max_digits10)
        << ridge << '\n'
        << "#temperature\t" << temperature << '\n'
        << "#matched_rows\t" << matched_rows << '\n'
        << "#sampled_rows\t" << sampled_rows << '\n'
        << "#folds\t" << folds << '\n'
        << "#minimum_per_class\t" << minimum_per_class << '\n'
        << "#sampling_seed\t" << sampling_seed << '\n';
    for (size_t topic = 0; topic < topics.size(); ++topic) {
        output << "topic\t" << topic << '\t' << topics[topic] << '\n';
    }
    for (Eigen::Index component = 0; component < coefficients.rows();
            ++component) {
        output << "class\t" << component << '\t'
            << classes[static_cast<size_t>(component)] << '\t'
            << intercepts(component);
        for (Eigen::Index topic = 0; topic < coefficients.cols(); ++topic) {
            output << '\t' << coefficients(component, topic);
        }
        output << '\n';
    }
}

Model Model::read(const std::string& path) {
    TextLineReader input(path);
    std::string line;
    Model model;
    bool version_seen = false;
    std::vector<std::vector<std::string>> class_rows;
    while (input.getline(line)) {
        if (line.empty()) continue;
        const std::vector<std::string> fields = split_delimited(line, '\t');
        if (fields[0] == "#partition_classifier") {
            int32_t version = 0;
            if (fields.size() != 2 || !str2int32(fields[1], version)
                    || version != MODEL_SCHEMA_VERSION) {
                throw std::runtime_error("Unsupported classifier schema");
            }
            version_seen = true;
        } else if (fields[0] == "#ridge" && fields.size() == 2) {
            if (!str2double(fields[1], model.ridge))
                throw std::runtime_error("Invalid classifier ridge");
        } else if (fields[0] == "#temperature" && fields.size() == 2) {
            if (!str2double(fields[1], model.temperature))
                throw std::runtime_error("Invalid classifier temperature");
        } else if (fields[0] == "#matched_rows" && fields.size() == 2) {
            if (!str2uint64(fields[1], model.matched_rows))
                throw std::runtime_error("Invalid classifier matched rows");
        } else if (fields[0] == "#sampled_rows" && fields.size() == 2) {
            if (!str2uint64(fields[1], model.sampled_rows))
                throw std::runtime_error("Invalid classifier sampled rows");
        } else if (fields[0] == "#folds" && fields.size() == 2) {
            if (!str2int32(fields[1], model.folds))
                throw std::runtime_error("Invalid classifier folds");
        } else if (fields[0] == "#minimum_per_class" && fields.size() == 2) {
            if (!str2int32(fields[1], model.minimum_per_class))
                throw std::runtime_error("Invalid classifier minimum");
        } else if (fields[0] == "#sampling_seed" && fields.size() == 2) {
            if (!str2uint64(fields[1], model.sampling_seed))
                throw std::runtime_error("Invalid classifier seed");
        } else if (fields[0] == "topic") {
            int32_t index = -1;
            if (fields.size() != 3 || !str2int32(fields[1], index)
                    || index != static_cast<int32_t>(model.topics.size())) {
                throw std::runtime_error("Invalid classifier topic row");
            }
            model.topics.push_back(fields[2]);
        } else if (fields[0] == "class") {
            class_rows.push_back(fields);
        } else if (fields[0][0] != '#') {
            throw std::runtime_error("Unknown classifier row: " + fields[0]);
        }
    }
    if (!version_seen || model.topics.empty() || class_rows.empty()) {
        throw std::runtime_error("Incomplete classifier model: " + path);
    }
    model.intercepts.resize(class_rows.size());
    model.coefficients.resize(class_rows.size(), model.topics.size());
    for (size_t component = 0; component < class_rows.size(); ++component) {
        const std::vector<std::string>& fields = class_rows[component];
        int32_t index = -1;
        if (fields.size() != model.topics.size() + 4
                || !str2int32(fields[1], index)
                || index != static_cast<int32_t>(component)
                || !str2double(fields[3], model.intercepts(index))) {
            throw std::runtime_error("Invalid classifier class row");
        }
        model.classes.push_back(fields[2]);
        for (size_t topic = 0; topic < model.topics.size(); ++topic) {
            if (!str2double(fields[topic + 4],
                    model.coefficients(index, topic))) {
                throw std::runtime_error("Invalid classifier coefficient");
            }
        }
    }
    model.validate(1e-7);
    return model;
}

void CrossfitBundle::validate(double tolerance) const {
    full_model.validate(tolerance);
    if (fold_models.size() < 2
            || heldout_fold_by_identifier.size() != full_model.sampled_rows) {
        throw std::invalid_argument("Invalid classifier crossfit bundle");
    }
    for (const Model& model : fold_models) {
        model.validate(tolerance);
        if (model.topics != full_model.topics
                || model.classes != full_model.classes) {
            throw std::invalid_argument(
                "Crossfit fold model topics or classes differ");
        }
    }
    for (const auto& route : heldout_fold_by_identifier) {
        if (route.first.empty() || route.second < 0
                || route.second >= static_cast<int32_t>(fold_models.size())) {
            throw std::invalid_argument("Invalid classifier crossfit route");
        }
    }
}

const Model& CrossfitBundle::model_for(const std::string& identifier,
        bool use_crossfit, int32_t* heldout_fold) const {
    if (heldout_fold != nullptr) *heldout_fold = -1;
    if (use_crossfit) {
        const auto found = heldout_fold_by_identifier.find(identifier);
        if (found != heldout_fold_by_identifier.end()) {
            if (heldout_fold != nullptr) *heldout_fold = found->second;
            return fold_models[static_cast<size_t>(found->second)];
        }
    }
    return full_model;
}

void CrossfitBundle::write(const std::string& path) const {
    validate();
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot write classifier crossfit bundle: "
            + path);
    }
    output << "#partition_classifier_crossfit\t" << CROSSFIT_SCHEMA_VERSION
        << '\n' << "#outer_folds\t" << fold_models.size()
        << '\n' << "#matched_rows\t" << full_model.matched_rows
        << '\n' << "#sampled_rows\t" << full_model.sampled_rows
        << '\n' << "#minimum_per_class\t" << full_model.minimum_per_class
        << '\n' << "#sampling_seed\t" << full_model.sampling_seed << '\n'
        << std::scientific
        << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (size_t topic = 0; topic < full_model.topics.size(); ++topic) {
        output << "topic\t" << topic << '\t' << full_model.topics[topic]
            << '\n';
    }
    for (size_t component = 0; component < full_model.classes.size();
            ++component) {
        output << "class_name\t" << component << '\t'
            << full_model.classes[component] << '\n';
    }
    auto write_model = [&](const char* name, const Model& model) {
        output << "model\t" << name << '\t' << model.ridge << '\t'
            << model.temperature << '\t' << model.folds << '\n';
        for (Eigen::Index component = 0;
                component < model.intercepts.size(); ++component) {
            output << "intercept\t" << name << '\t' << component << '\t'
                << model.intercepts(component) << '\n';
            for (Eigen::Index topic = 0; topic < model.coefficients.cols();
                    ++topic) {
                output << "coefficient\t" << name << '\t' << component
                    << '\t' << topic << '\t'
                    << model.coefficients(component, topic) << '\n';
            }
        }
    };
    write_model("full", full_model);
    std::vector<std::string> fold_names(fold_models.size());
    for (size_t fold = 0; fold < fold_models.size(); ++fold) {
        fold_names[fold] = std::to_string(fold);
        write_model(fold_names[fold].c_str(), fold_models[fold]);
    }
    std::vector<std::pair<std::string, int32_t>> routes(
        heldout_fold_by_identifier.begin(),
        heldout_fold_by_identifier.end());
    std::sort(routes.begin(), routes.end());
    for (const auto& route : routes) {
        output << "route\t" << route.first << '\t' << route.second << '\n';
    }
}

CrossfitBundle CrossfitBundle::read(const std::string& path) {
    TextLineReader input(path);
    std::string line;
    CrossfitBundle bundle;
    bool version_seen = false;
    int32_t outer_folds = -1;
    std::vector<std::string> topics;
    std::vector<std::string> classes;
    auto resolve_model = [&](const std::string& name) -> Model& {
        if (name == "full") return bundle.full_model;
        int32_t fold = -1;
        if (!str2int32(name, fold) || fold < 0 || fold >= outer_folds) {
            throw std::runtime_error("Invalid crossfit model index");
        }
        return bundle.fold_models[static_cast<size_t>(fold)];
    };
    while (input.getline(line)) {
        if (line.empty()) continue;
        const std::vector<std::string> fields = split_delimited(line, '\t');
        if (fields[0] == "#partition_classifier_crossfit") {
            int32_t version = 0;
            if (fields.size() != 2 || !str2int32(fields[1], version)
                    || version != CROSSFIT_SCHEMA_VERSION) {
                throw std::runtime_error("Unsupported classifier crossfit schema");
            }
            version_seen = true;
        } else if (fields[0] == "#outer_folds") {
            if (fields.size() != 2 || !str2int32(fields[1], outer_folds)
                    || outer_folds < 2) {
                throw std::runtime_error("Invalid crossfit fold count");
            }
            bundle.fold_models.resize(static_cast<size_t>(outer_folds));
        } else if (fields[0] == "#matched_rows" && fields.size() == 2) {
            if (!str2uint64(fields[1], bundle.full_model.matched_rows))
                throw std::runtime_error("Invalid crossfit matched rows");
        } else if (fields[0] == "#sampled_rows" && fields.size() == 2) {
            if (!str2uint64(fields[1], bundle.full_model.sampled_rows))
                throw std::runtime_error("Invalid crossfit sampled rows");
        } else if (fields[0] == "#minimum_per_class"
                && fields.size() == 2) {
            if (!str2int32(fields[1], bundle.full_model.minimum_per_class))
                throw std::runtime_error("Invalid crossfit minimum");
        } else if (fields[0] == "#sampling_seed" && fields.size() == 2) {
            if (!str2uint64(fields[1], bundle.full_model.sampling_seed))
                throw std::runtime_error("Invalid crossfit seed");
        } else if (fields[0] == "topic") {
            int32_t index = -1;
            if (fields.size() != 3 || !str2int32(fields[1], index)
                    || index != static_cast<int32_t>(topics.size())) {
                throw std::runtime_error("Invalid crossfit topic row");
            }
            topics.push_back(fields[2]);
        } else if (fields[0] == "class_name") {
            int32_t index = -1;
            if (fields.size() != 3 || !str2int32(fields[1], index)
                    || index != static_cast<int32_t>(classes.size())) {
                throw std::runtime_error("Invalid crossfit class row");
            }
            classes.push_back(fields[2]);
        } else if (fields[0] == "model") {
            if (fields.size() != 5 || topics.empty() || classes.empty()
                    || outer_folds < 2) {
                throw std::runtime_error("Invalid crossfit model row");
            }
            Model& model = resolve_model(fields[1]);
            model.topics = topics;
            model.classes = classes;
            model.intercepts = Eigen::VectorXd::Zero(classes.size());
            model.coefficients = RowMajorMatrixXd::Zero(
                classes.size(), topics.size());
            if (!str2double(fields[2], model.ridge)
                    || !str2double(fields[3], model.temperature)
                    || !str2int32(fields[4], model.folds)) {
                throw std::runtime_error("Invalid crossfit model metadata");
            }
        } else if (fields[0] == "intercept") {
            int32_t component = -1;
            double value = 0.0;
            if (fields.size() != 4) {
                throw std::runtime_error("Invalid crossfit intercept row");
            }
            Model& model = resolve_model(fields[1]);
            if (!str2int32(fields[2], component)
                    || component < 0 || component >= model.intercepts.size()
                    || !str2double(fields[3], value)) {
                throw std::runtime_error("Invalid crossfit intercept row");
            }
            model.intercepts(component) = value;
        } else if (fields[0] == "coefficient") {
            int32_t component = -1, topic = -1;
            double value = 0.0;
            if (fields.size() != 5) {
                throw std::runtime_error("Invalid crossfit coefficient row");
            }
            Model& model = resolve_model(fields[1]);
            if (!str2int32(fields[2], component)
                    || !str2int32(fields[3], topic) || component < 0
                    || component >= model.coefficients.rows() || topic < 0
                    || topic >= model.coefficients.cols()
                    || !str2double(fields[4], value)) {
                throw std::runtime_error("Invalid crossfit coefficient row");
            }
            model.coefficients(component, topic) = value;
        } else if (fields[0] == "route") {
            int32_t fold = -1;
            if (fields.size() != 3 || fields[1].empty()
                    || !str2int32(fields[2], fold) || fold < 0
                    || fold >= outer_folds
                    || !bundle.heldout_fold_by_identifier.emplace(
                        fields[1], fold).second) {
                throw std::runtime_error("Invalid crossfit route row");
            }
        } else if (fields[0][0] != '#') {
            throw std::runtime_error("Unknown crossfit row: " + fields[0]);
        }
    }
    if (!version_seen || outer_folds < 2) {
        throw std::runtime_error("Incomplete classifier crossfit bundle: "
            + path);
    }
    bundle.full_model.folds = outer_folds;
    for (Model& model : bundle.fold_models) {
        model.matched_rows = bundle.full_model.matched_rows;
        model.sampled_rows = bundle.full_model.sampled_rows;
        model.minimum_per_class = bundle.full_model.minimum_per_class;
        model.sampling_seed = bundle.full_model.sampling_seed;
    }
    bundle.validate(1e-7);
    return bundle;
}

bool CrossfitBundle::is_bundle(const std::string& path) {
    TextLineReader input(path);
    std::string line;
    while (input.getline(line)) {
        if (line.empty()) continue;
        const std::vector<std::string> fields = split_delimited(line, '\t');
        return fields.size() == 2
            && fields[0] == "#partition_classifier_crossfit";
    }
    throw std::runtime_error("Classifier model is empty: " + path);
}

Metrics evaluate(const Eigen::Ref<const RowMajorMatrixXd>& probabilities,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const Eigen::Ref<const Eigen::VectorXd>& weights) {
    if (probabilities.rows() != labels.size()
            || labels.size() != weights.size() || probabilities.cols() < 2) {
        throw std::invalid_argument("Invalid classifier metric dimensions");
    }
    Metrics output;
    for (Eigen::Index row = 0; row < probabilities.rows(); ++row) {
        const double weight = weights(row);
        if (!(weight >= 0.0) || !std::isfinite(weight)
                || labels(row) < 0 || labels(row) >= probabilities.cols()) {
            throw std::invalid_argument("Invalid classifier metric input");
        }
        output.log_loss -= weight * std::log(std::max(
            probabilities(row, labels(row)), 1e-300));
        for (Eigen::Index component = 0; component < probabilities.cols();
                ++component) {
            const double residual = probabilities(row, component)
                - (component == labels(row) ? 1.0 : 0.0);
            output.brier += weight * residual * residual;
        }
        Eigen::Index prediction = 0;
        probabilities.row(row).maxCoeff(&prediction);
        output.accuracy += weight * (prediction == labels(row));
        output.weight += weight;
    }
    if (!(output.weight > 0.0)) {
        throw std::invalid_argument("Classifier metrics have no weight");
    }
    output.log_loss /= output.weight;
    output.brier /= output.weight;
    output.accuracy /= output.weight;
    return output;
}

RowMajorMatrixXd probabilities_from_logits(
        const Eigen::Ref<const RowMajorMatrixXd>& logits,
        double temperature) {
    if (logits.cols() < 2 || !logits.allFinite()
            || !(temperature > 0.0) || !std::isfinite(temperature)) {
        throw std::invalid_argument("Invalid classifier logits");
    }
    RowMajorMatrixXd output = logits / temperature;
    rowSoftmaxInPlace(output);
    return output;
}

double fit_temperature(const Eigen::Ref<const RowMajorMatrixXd>& logits,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const Eigen::Ref<const Eigen::VectorXd>& weights) {
    if (logits.rows() != labels.size() || labels.size() != weights.size()
            || logits.cols() < 2) {
        throw std::invalid_argument("Invalid temperature-fitting input");
    }
    double left = -5.0;
    double right = 5.0;
    constexpr double ratio = 0.6180339887498948482;
    double x1 = right - ratio * (right - left);
    double x2 = left + ratio * (right - left);
    double f1 = temperature_loss(x1, logits, labels, weights);
    double f2 = temperature_loss(x2, logits, labels, weights);
    for (int32_t iteration = 0; iteration < 100; ++iteration) {
        if (f1 > f2) {
            left = x1;
            x1 = x2;
            f1 = f2;
            x2 = left + ratio * (right - left);
            f2 = temperature_loss(x2, logits, labels, weights);
        } else {
            right = x2;
            x2 = x1;
            f2 = f1;
            x1 = right - ratio * (right - left);
            f1 = temperature_loss(x1, logits, labels, weights);
        }
    }
    return std::exp(0.5 * (left + right));
}

FitResult fit(const Eigen::Ref<const RowMajorMatrixXd>& compositions,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const Eigen::Ref<const Eigen::VectorXd>& weights,
        const std::vector<std::string>& identifiers,
        const std::vector<std::string>& topics,
        const std::vector<std::string>& classes,
        uint64_t seed,
        const FitOptions& options) {
    const int32_t rows = static_cast<int32_t>(compositions.rows());
    const int32_t topic_count = static_cast<int32_t>(compositions.cols());
    const int32_t class_count = static_cast<int32_t>(classes.size());
    if (rows < 4 || topic_count != static_cast<int32_t>(topics.size())
            || topic_count < 2 || class_count < 2 || labels.size() != rows
            || weights.size() != rows
            || identifiers.size() != static_cast<size_t>(rows)
            || !compositions.allFinite()
            || (compositions.array() < 0.0).any()
            || (compositions.rowwise().sum().array() <= 0.0).any()
            || !weights.allFinite() || (weights.array() <= 0.0).any()
            || options.ridge_grid.empty() || options.folds < 2
            || options.max_iterations <= 0 || options.lbfgs_history <= 0
            || !(options.gradient_tolerance > 0.0)) {
        throw std::invalid_argument("Invalid classifier fit input");
    }
    require_unique_nonempty(identifiers, "Classifier identifiers");
    require_unique_nonempty(topics, "Classifier topics");
    require_unique_nonempty(classes, "Classifier classes");
    RowMajorMatrixXd normalized = compositions;
    for (Eigen::Index row = 0; row < normalized.rows(); ++row) {
        normalized.row(row) /= normalized.row(row).sum();
    }
    const Eigen::MatrixXd topic_helmert = normalized_helmert(topic_count);
    const RowMajorMatrixXd x = normalized * topic_helmert.transpose();

    const std::vector<int32_t> complete = all_rows(rows);
    const FoldAssignment assignment = make_stratified_folds(labels, complete,
        identifiers, class_count, options.folds,
        seed ^ 0x6f7264696e617279ULL);
    const int32_t folds = assignment.folds;
    const Eigen::VectorXi& fold_by_row = assignment.by_row;

    {
        std::ostringstream message;
        message << "Partition classifier CV started: "
            << options.ridge_grid.size() << " ridge values, " << folds
            << " folds, " << rows << " rows";
        report_progress(options, message.str());
    }

    FitResult result;
    result.cv.reserve(options.ridge_grid.size());
    size_t selected = 0;
    double best_loss = std::numeric_limits<double>::infinity();
    for (const double ridge : options.ridge_grid) {
        if (!(ridge >= 0.0) || !std::isfinite(ridge)) {
            throw std::invalid_argument("Ridge grid must be finite and nonnegative");
        }
        RowMajorMatrixXd oof(rows, class_count);
        for (int32_t fold = 0; fold < folds; ++fold) {
            std::vector<int32_t> training;
            std::vector<int32_t> validation;
            training.reserve(rows);
            validation.reserve(rows / folds + class_count);
            for (int32_t row = 0; row < rows; ++row) {
                (fold_by_row(row) == fold ? validation : training).push_back(row);
            }
            const Eigen::VectorXd parameters = fit_parameters(x, labels,
                weights, training, class_count, ridge, options);
            const RowMajorMatrixXd logits = parameter_logits(
                x, validation, parameters, class_count);
            for (size_t index = 0; index < validation.size(); ++index) {
                oof.row(validation[index]) = logits.row(index);
            }
        }
        CvResult cv;
        cv.ridge = ridge;
        cv.metrics = evaluate(probabilities_from_logits(oof, 1.0),
            labels, weights);
        result.cv.push_back(cv);
        if (cv.metrics.log_loss < best_loss) {
            best_loss = cv.metrics.log_loss;
            selected = result.cv.size() - 1;
            result.oof_logits = std::move(oof);
        }
    }
    result.cv[selected].selected = true;
    {
        std::ostringstream message;
        message << "Partition classifier CV parameter selection finished: "
            << "ridge " << result.cv[selected].ridge
            << ", OOF log loss " << result.cv[selected].metrics.log_loss;
        report_progress(options, message.str());
    }
    result.calibration.stored_temperature = fit_temperature(
        result.oof_logits, labels, weights);
    {
        std::ostringstream message;
        message << "Partition classifier OOF temperature calibration finished: "
            << result.calibration.stored_temperature;
        report_progress(options, message.str());
    }

    result.cross_fitted_probabilities.resize(rows, class_count);
    for (int32_t fold = 0; fold < folds; ++fold) {
        std::vector<int32_t> training;
        std::vector<int32_t> validation;
        for (int32_t row = 0; row < rows; ++row) {
            (fold_by_row(row) == fold ? validation : training).push_back(row);
        }
        RowMajorMatrixXd training_logits(training.size(), class_count);
        Eigen::VectorXi training_labels(training.size());
        Eigen::VectorXd training_weights(training.size());
        for (size_t index = 0; index < training.size(); ++index) {
            training_logits.row(index) = result.oof_logits.row(training[index]);
            training_labels(index) = labels(training[index]);
            training_weights(index) = weights(training[index]);
        }
        const double temperature = fit_temperature(
            training_logits, training_labels, training_weights);
        RowMajorMatrixXd validation_logits(validation.size(), class_count);
        for (size_t index = 0; index < validation.size(); ++index) {
            validation_logits.row(index) = result.oof_logits.row(validation[index]);
        }
        const RowMajorMatrixXd probability = probabilities_from_logits(
            validation_logits, temperature);
        for (size_t index = 0; index < validation.size(); ++index) {
            result.cross_fitted_probabilities.row(validation[index]) =
                probability.row(index);
        }
    }
    result.calibration.cross_fitted = evaluate(
        result.cross_fitted_probabilities, labels, weights);
    result.calibration.classwise.resize(class_count);
    for (int32_t component = 0; component < class_count; ++component) {
        Metrics metric;
        for (int32_t row = 0; row < rows; ++row) {
            const double target = labels(row) == component ? 1.0 : 0.0;
            const double probability =
                result.cross_fitted_probabilities(row, component);
            metric.log_loss -= weights(row) * (target > 0.5
                ? std::log(std::max(probability, 1e-300))
                : std::log(std::max(1.0 - probability, 1e-300)));
            metric.brier += weights(row)
                * (probability - target) * (probability - target);
            metric.accuracy += weights(row)
                * ((probability >= 0.5) == (target > 0.5));
            metric.weight += weights(row);
        }
        metric.log_loss /= metric.weight;
        metric.brier /= metric.weight;
        metric.accuracy /= metric.weight;
        result.calibration.classwise[component] = metric;
    }
    result.calibration.bins.resize(15);
    for (int32_t bin = 0; bin < 15; ++bin) {
        result.calibration.bins[bin].bin = bin;
        result.calibration.bins[bin].lower = bin / 15.0;
        result.calibration.bins[bin].upper = (bin + 1) / 15.0;
    }
    for (int32_t row = 0; row < rows; ++row) {
        Eigen::Index prediction = 0;
        const double confidence = result.cross_fitted_probabilities.row(row)
            .maxCoeff(&prediction);
        const int32_t bin = std::min(14,
            static_cast<int32_t>(confidence * 15.0));
        CalibrationBin& summary = result.calibration.bins[bin];
        summary.weight += weights(row);
        summary.mean_confidence += weights(row) * confidence;
        summary.accuracy += weights(row) * (prediction == labels(row));
    }
    for (CalibrationBin& bin : result.calibration.bins) {
        if (bin.weight > 0.0) {
            bin.mean_confidence /= bin.weight;
            bin.accuracy /= bin.weight;
        }
    }

    {
        std::ostringstream message;
        message << "Partition classifier full model fitting started: "
            << rows << " rows, ridge " << result.cv[selected].ridge;
        report_progress(options, message.str());
    }
    const Eigen::VectorXd parameters = fit_parameters(x, labels, weights,
        complete, class_count, result.cv[selected].ridge, options);
    result.model = make_model(parameters, topics, classes,
        result.cv[selected].ridge);
    result.model.temperature = result.calibration.stored_temperature;
    result.model.folds = folds;
    result.model.validate();
    report_progress(options,
        "Partition classifier full model fitting finished");
    return result;
}

CrossfitResult fit_crossfit(
        const Eigen::Ref<const RowMajorMatrixXd>& compositions,
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const Eigen::Ref<const Eigen::VectorXd>& weights,
        const std::vector<std::string>& identifiers,
        const std::vector<std::string>& topics,
        const std::vector<std::string>& classes, uint64_t seed,
        const FitOptions& options) {
    const int32_t rows = static_cast<int32_t>(compositions.rows());
    const int32_t class_count = static_cast<int32_t>(classes.size());
    if (rows < 4 || identifiers.size() != static_cast<size_t>(rows)
            || labels.size() != rows || weights.size() != rows
            || compositions.cols() != static_cast<Eigen::Index>(topics.size())
            || topics.size() < 2 || classes.size() < 2
            || !compositions.allFinite()
            || (compositions.array() < 0.0).any()
            || (compositions.rowwise().sum().array() <= 0.0).any()
            || !weights.allFinite() || (weights.array() <= 0.0).any()
            || options.ridge_grid.empty() || options.folds < 2) {
        throw std::invalid_argument("Invalid classifier crossfit input");
    }
    require_unique_nonempty(identifiers, "Crossfit identifiers");
    require_unique_nonempty(topics, "Classifier topics");
    require_unique_nonempty(classes, "Classifier classes");
    RowMajorMatrixXd normalized = compositions;
    for (Eigen::Index row = 0; row < normalized.rows(); ++row) {
        normalized.row(row) /= normalized.row(row).sum();
    }
    const RowMajorMatrixXd x = normalized
        * normalized_helmert(static_cast<int32_t>(topics.size())).transpose();
    const std::vector<int32_t> complete = all_rows(rows);
    const FoldAssignment outer = make_stratified_folds(labels, complete,
        identifiers, class_count, options.folds, seed ^ 0x6f75746572ULL);
    for (int32_t component = 0; component < class_count; ++component) {
        int32_t count = 0;
        for (int32_t row = 0; row < rows; ++row) {
            count += labels(row) == component;
        }
        if (count < 3) {
            throw std::invalid_argument(
                "Nested crossfit requires at least three rows per class");
        }
    }

    {
        std::ostringstream message;
        message << "Partition classifier nested crossfit started: "
            << outer.folds << " outer folds, " << rows << " rows";
        report_progress(options, message.str());
    }

    CrossfitResult result;
    result.fold_by_row = outer.by_row;
    result.probabilities.resize(rows, class_count);
    result.fold_models.reserve(outer.folds);
    result.diagnostics.reserve(outer.folds);
    for (int32_t fold = 0; fold < outer.folds; ++fold) {
        std::vector<int32_t> training;
        std::vector<int32_t> heldout;
        for (int32_t row = 0; row < rows; ++row) {
            (outer.by_row(row) == fold ? heldout : training).push_back(row);
        }
        {
            std::ostringstream message;
            message << "Partition classifier crossfit fold " << (fold + 1)
                << '/' << outer.folds
                << " started: nested parameter selection and final fitting on "
                << training.size() << " rows";
            report_progress(options, message.str());
        }
        const NestedModelFit fitted = fit_nested_model(x, labels, weights,
            training, identifiers, topics, classes,
            seed ^ (static_cast<uint64_t>(fold) << 32)
                ^ 0x696e6e6572ULL,
            options);
        result.fold_models.push_back(fitted.model);
        {
            std::ostringstream message;
            message << "Partition classifier crossfit fold " << (fold + 1)
                << '/' << outer.folds << " final model fitting finished: "
                << training.size() << " training rows, " << heldout.size()
                << " held-out rows, " << fitted.inner_folds
                << " inner folds, ridge " << fitted.model.ridge
                << ", temperature " << fitted.model.temperature;
            report_progress(options, message.str());
        }
        RowMajorMatrixXd fold_probabilities(heldout.size(), class_count);
        Eigen::VectorXi fold_labels(heldout.size());
        Eigen::VectorXd fold_weights(heldout.size());
        for (size_t index = 0; index < heldout.size(); ++index) {
            const int32_t row = heldout[index];
            fold_probabilities.row(index) = fitted.model.probabilities(
                normalized.row(row).transpose()).transpose();
            result.probabilities.row(row) = fold_probabilities.row(index);
            fold_labels(index) = labels(row);
            fold_weights(index) = weights(row);
        }
        CrossfitFoldDiagnostic diagnostic;
        diagnostic.fold = fold;
        diagnostic.training_rows = training.size();
        diagnostic.heldout_rows = heldout.size();
        diagnostic.inner_folds = fitted.inner_folds;
        diagnostic.ridge = fitted.model.ridge;
        diagnostic.temperature = fitted.model.temperature;
        diagnostic.metrics = evaluate(
            fold_probabilities, fold_labels, fold_weights);
        result.diagnostics.push_back(diagnostic);
    }
    result.overall = evaluate(result.probabilities, labels, weights);
    {
        std::ostringstream message;
        message << "Partition classifier nested crossfit finished: OOS log loss "
            << result.overall.log_loss << ", accuracy "
            << result.overall.accuracy;
        report_progress(options, message.str());
    }
    return result;
}

} // namespace punkst::partition_classifier
