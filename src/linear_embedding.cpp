#include "linear_embedding.hpp"

#include "clustering_core/qda_projection.hpp"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using punkst::linear_embedding::ProjectionData;
using punkst::linear_embedding::ProjectionSpace;
using punkst::linear_embedding::projection_space_name;

constexpr int32_t minimum_qda_cluster_rows = 11;

RowMajorMatrixXd normalize_factor_proportions(
        const Eigen::Ref<const RowMajorMatrixXd>& values) {
    RowMajorMatrixXd out = values;
    for (Eigen::Index row = 0; row < out.rows(); ++row) {
        const double scale = out.row(row).maxCoeff();
        if (!(scale > 0.0) || !std::isfinite(scale)) {
            throw std::invalid_argument(
                "Projection requires positive finite factor rows");
        }
        out.row(row) /= scale;
        const double total = out.row(row).sum();
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::invalid_argument(
                "Projection requires positive finite factor rows");
        }
        out.row(row) /= total;
    }
    return out;
}

template<class ResolveValueRow>
Eigen::MatrixXd aggregate_cluster_factors_impl(
        const Eigen::Ref<const RowMajorMatrixXd>& values,
        const Eigen::Ref<const Eigen::VectorXi>& assignments,
        int32_t components, ResolveValueRow&& resolve_value_row) {
    if (values.cols() <= 0 || assignments.size() <= 0 || components <= 0) {
        throw std::invalid_argument("Invalid cluster factor input");
    }
    Eigen::MatrixXd sums = Eigen::MatrixXd::Zero(
        values.cols(), components);
    for (Eigen::Index row = 0; row < assignments.size(); ++row) {
        const int32_t cluster = assignments(row);
        const Eigen::Index value_row = resolve_value_row(row);
        if (cluster < 0 || cluster >= components
                || value_row < 0 || value_row >= values.rows()) {
            throw std::invalid_argument(
                "Cluster factor assignment or row is outside its range");
        }
        sums.col(cluster) += values.row(value_row).transpose();
    }
    return sums;
}

struct ProjectionOutput {
    ProjectionSpace space = ProjectionSpace::Linear;
    const punkst::linear_embedding::ProjectionData* data = nullptr;
    punkst::projection::VisualizationResult visualization;
};

struct ProjectionInput {
    ProjectionSpace space = ProjectionSpace::Linear;
    punkst::linear_embedding::ProjectionData data;
    RowMajorMatrixXd matched_coordinates;
    std::optional<punkst::projection::VisualizationSampleMoments> sample_moments;
};

struct QdaRows {
    std::vector<int32_t> training;
    std::vector<int32_t> validation;
};

struct QdaSparsityCvEntry {
    double strength = 0.0;
    double mean_heldout_loss = 0.0;
    double standard_error = 0.0;
    double loss_increase = 0.0;
    double mean_quartimax_score = 0.0;
    double mean_training_loss = 0.0;
    double mean_validation_loss = 0.0;
    double mean_validation_objective = 0.0;
    size_t heldout_rows = 0;
    bool eligible = false;
    bool selected = false;
};

struct QdaSparsityCvResult {
    std::vector<QdaSparsityCvEntry> entries;
    double selected_strength = 0.0;
    double eligibility_threshold = 0.0;
    int32_t folds = 0;
    size_t population_rows = 0;
};

int32_t adaptive_qda_training_cap(
        int32_t input_dimensions, int32_t output_dimensions,
        int32_t components) {
    const int64_t grassmann_degrees = static_cast<int64_t>(output_dimensions)
        * (input_dimensions - output_dimensions);
    const int64_t class_degrees = static_cast<int64_t>(components)
        * output_dimensions * (output_dimensions + 3) / 2;
    const int64_t total_degrees = grassmann_degrees + class_degrees
        + components - 1;
    const int64_t target = std::max<int64_t>(
        static_cast<int64_t>(50) * components, 3 * total_degrees);
    if (target > std::numeric_limits<int32_t>::max()) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(target);
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

void deterministic_order(std::vector<int32_t>& rows, uint64_t seed) {
    std::sort(rows.begin(), rows.end(), [seed](int32_t left, int32_t right) {
        const uint64_t left_key = splitmix64(seed
            ^ static_cast<uint64_t>(static_cast<uint32_t>(left)));
        const uint64_t right_key = splitmix64(seed
            ^ static_cast<uint64_t>(static_cast<uint32_t>(right)));
        return left_key == right_key ? left < right : left_key < right_key;
    });
}

std::vector<int32_t> stratified_cap(const std::vector<int32_t>& rows,
        const Eigen::Ref<const Eigen::VectorXi>& assignments,
        int32_t components, int32_t maximum_rows, int32_t minimum_per_class,
        uint64_t seed) {
    if (maximum_rows <= 0 || static_cast<int32_t>(rows.size()) <= maximum_rows) {
        std::vector<int32_t> out = rows;
        std::sort(out.begin(), out.end());
        return out;
    }
    if (maximum_rows < components * minimum_per_class) {
        throw std::invalid_argument(
            "QDA row cap is too small for the represented classes");
    }
    std::vector<std::vector<int32_t>> by_class(components);
    for (const int32_t row : rows) {
        by_class[static_cast<size_t>(assignments(row))].push_back(row);
    }
    std::vector<int32_t> take(components, minimum_per_class);
    int32_t allocated = components * minimum_per_class;
    while (allocated < maximum_rows) {
        int32_t best = -1;
        double best_deficit = -std::numeric_limits<double>::infinity();
        for (int32_t component = 0; component < components; ++component) {
            if (take[component]
                    >= static_cast<int32_t>(by_class[component].size())) continue;
            const double target = static_cast<double>(maximum_rows)
                * by_class[component].size() / rows.size();
            const double deficit = target - take[component];
            if (deficit > best_deficit) {
                best_deficit = deficit;
                best = component;
            }
        }
        if (best < 0) break;
        ++take[best];
        ++allocated;
    }
    std::vector<int32_t> out;
    out.reserve(allocated);
    for (int32_t component = 0; component < components; ++component) {
        deterministic_order(by_class[component], seed
            ^ (static_cast<uint64_t>(component) << 32));
        out.insert(out.end(), by_class[component].begin(),
            by_class[component].begin() + take[component]);
    }
    std::sort(out.begin(), out.end());
    return out;
}

QdaRows make_qda_rows(const Eigen::Ref<const Eigen::VectorXi>& assignments,
        int32_t components, double validation_fraction,
        int32_t training_cap, int32_t validation_cap, int32_t seed) {
    std::vector<std::vector<int32_t>> by_class(components);
    for (Eigen::Index row = 0; row < assignments.size(); ++row) {
        by_class[static_cast<size_t>(assignments(row))].push_back(
            static_cast<int32_t>(row));
    }
    QdaRows out;
    for (int32_t component = 0; component < components; ++component) {
        if (by_class[component].size() < 3) {
            throw std::invalid_argument(
                "QDA projection requires at least three matched rows per class");
        }
        deterministic_order(by_class[component],
            static_cast<uint64_t>(static_cast<uint32_t>(seed))
                ^ (static_cast<uint64_t>(component) << 32));
        int32_t validation_count = static_cast<int32_t>(std::llround(
            validation_fraction * by_class[component].size()));
        validation_count = std::max(1, std::min(validation_count,
            static_cast<int32_t>(by_class[component].size()) - 2));
        out.validation.insert(out.validation.end(),
            by_class[component].begin(),
            by_class[component].begin() + validation_count);
        out.training.insert(out.training.end(),
            by_class[component].begin() + validation_count,
            by_class[component].end());
    }
    out.training = stratified_cap(out.training, assignments, components,
        training_cap, 2, static_cast<uint64_t>(seed) ^ 0x747261696eULL);
    out.validation = stratified_cap(out.validation, assignments, components,
        validation_cap, 1, static_cast<uint64_t>(seed) ^ 0x76616cULL);
    return out;
}

RowMajorMatrixXd select_rows(
        const Eigen::Ref<const RowMajorMatrixXd>& values,
        const std::vector<int32_t>& rows) {
    RowMajorMatrixXd out(rows.size(), values.cols());
    for (size_t index = 0; index < rows.size(); ++index) {
        out.row(static_cast<Eigen::Index>(index)) = values.row(rows[index]);
    }
    return out;
}

Eigen::VectorXi select_labels(
        const Eigen::Ref<const Eigen::VectorXi>& labels,
        const std::vector<int32_t>& rows) {
    Eigen::VectorXi out(rows.size());
    for (size_t index = 0; index < rows.size(); ++index) {
        out(static_cast<Eigen::Index>(index)) = labels(rows[index]);
    }
    return out;
}

QdaSparsityCvResult cross_validate_qda_sparsity(
        const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
        const Eigen::Ref<const Eigen::VectorXi>& assignments,
        int32_t components, const std::vector<double>& strengths,
        int32_t requested_folds, double inner_validation_fraction,
        int32_t training_cap, int32_t validation_cap, int32_t seed,
        const punkst::projection::QdaProjectionOptions& base_options) {
    std::vector<int32_t> population(coordinates.rows());
    for (Eigen::Index row = 0; row < coordinates.rows(); ++row) {
        population[static_cast<size_t>(row)] = static_cast<int32_t>(row);
    }
    int32_t population_cap = 0;
    if (validation_cap > 0) {
        const int64_t total_cap = static_cast<int64_t>(training_cap)
            + validation_cap;
        population_cap = total_cap > std::numeric_limits<int32_t>::max()
            ? std::numeric_limits<int32_t>::max()
            : static_cast<int32_t>(total_cap);
    }
    population = stratified_cap(population, assignments, components,
        population_cap, 4, static_cast<uint64_t>(seed) ^ 0x6376706f6f6cULL);
    const RowMajorMatrixXd cv_coordinates = select_rows(
        coordinates, population);
    const Eigen::VectorXi cv_assignments = select_labels(
        assignments, population);

    std::vector<std::vector<int32_t>> by_class(
        static_cast<size_t>(components));
    for (Eigen::Index row = 0; row < cv_assignments.size(); ++row) {
        by_class[static_cast<size_t>(cv_assignments(row))].push_back(
            static_cast<int32_t>(row));
    }
    int32_t folds = requested_folds;
    for (const auto& rows : by_class) {
        folds = std::min(folds, static_cast<int32_t>(rows.size()));
    }
    if (folds < 2) {
        throw std::runtime_error(
            "QDA sparsity cross-validation requires at least two folds");
    }
    Eigen::VectorXi fold_by_row(cv_assignments.size());
    for (int32_t component = 0; component < components; ++component) {
        deterministic_order(by_class[static_cast<size_t>(component)],
            static_cast<uint64_t>(seed)
                ^ (static_cast<uint64_t>(component) << 32)
                ^ 0x6376666f6c64ULL);
        const auto& rows = by_class[static_cast<size_t>(component)];
        for (size_t index = 0; index < rows.size(); ++index) {
            fold_by_row(rows[index]) = static_cast<int32_t>(index % folds);
        }
    }

    QdaSparsityCvResult out;
    out.folds = folds;
    out.population_rows = population.size();
    out.entries.reserve(strengths.size());
    for (const double strength : strengths) {
        QdaSparsityCvEntry entry;
        entry.strength = strength;
        std::vector<double> fold_losses;
        fold_losses.reserve(static_cast<size_t>(folds));
        double weighted_loss = 0.0;
        for (int32_t fold = 0; fold < folds; ++fold) {
            std::vector<int32_t> outer_training_rows;
            std::vector<int32_t> outer_validation_rows;
            outer_training_rows.reserve(static_cast<size_t>(
                cv_coordinates.rows()));
            for (Eigen::Index row = 0; row < cv_coordinates.rows(); ++row) {
                (fold_by_row(row) == fold
                    ? outer_validation_rows : outer_training_rows)
                    .push_back(static_cast<int32_t>(row));
            }
            const RowMajorMatrixXd outer_training = select_rows(
                cv_coordinates, outer_training_rows);
            const RowMajorMatrixXd outer_validation = select_rows(
                cv_coordinates, outer_validation_rows);
            const Eigen::VectorXi outer_training_labels = select_labels(
                cv_assignments, outer_training_rows);
            const Eigen::VectorXi outer_validation_labels = select_labels(
                cv_assignments, outer_validation_rows);
            const int32_t fold_seed = static_cast<int32_t>(splitmix64(
                static_cast<uint64_t>(static_cast<uint32_t>(seed))
                    ^ static_cast<uint64_t>(fold) ^ 0x696e6e6572ULL)
                & 0x7fffffffULL);
            const QdaRows inner_rows = make_qda_rows(
                outer_training_labels, components,
                inner_validation_fraction, training_cap, validation_cap,
                fold_seed);
            const RowMajorMatrixXd inner_training = select_rows(
                outer_training, inner_rows.training);
            const RowMajorMatrixXd inner_validation = select_rows(
                outer_training, inner_rows.validation);
            const Eigen::VectorXi inner_training_labels = select_labels(
                outer_training_labels, inner_rows.training);
            const Eigen::VectorXi inner_validation_labels = select_labels(
                outer_training_labels, inner_rows.validation);
            punkst::projection::QdaProjectionOptions fold_options =
                base_options;
            fold_options.seed = fold_seed;
            fold_options.sparsity_strength = strength;
            const punkst::projection::QdaProjectionResult fit =
                punkst::projection::fit_qda_projection(
                    inner_training, inner_training_labels,
                    inner_validation, inner_validation_labels,
                    components, fold_options);
            const double heldout_loss =
                punkst::projection::qda_projection_log_loss(
                    fit.projection, outer_training, outer_training_labels,
                    outer_validation, outer_validation_labels, components,
                    fold_options);
            fold_losses.push_back(heldout_loss);
            weighted_loss += heldout_loss * outer_validation.rows();
            entry.heldout_rows += static_cast<size_t>(
                outer_validation.rows());
            entry.mean_quartimax_score += fit.quartimax_score;
            entry.mean_training_loss += fit.training_loss;
            entry.mean_validation_loss += fit.validation_loss;
            entry.mean_validation_objective += fit.validation_objective;
        }
        entry.mean_heldout_loss = weighted_loss
            / static_cast<double>(entry.heldout_rows);
        const double fold_mean = std::accumulate(
            fold_losses.begin(), fold_losses.end(), 0.0)
            / static_cast<double>(folds);
        double squared_deviation = 0.0;
        for (const double loss : fold_losses) {
            squared_deviation += (loss - fold_mean) * (loss - fold_mean);
        }
        entry.standard_error = std::sqrt(squared_deviation
            / static_cast<double>(folds * (folds - 1)));
        entry.mean_quartimax_score /= folds;
        entry.mean_training_loss /= folds;
        entry.mean_validation_loss /= folds;
        entry.mean_validation_objective /= folds;
        notice("QDA sparsity CV lambda %.10g: held-out log loss %.10g (SE %.10g), quartimax %.10g",
            strength, entry.mean_heldout_loss, entry.standard_error,
            entry.mean_quartimax_score);
        out.entries.push_back(entry);
    }
    const auto baseline = std::find_if(out.entries.begin(), out.entries.end(),
        [](const QdaSparsityCvEntry& entry) { return entry.strength == 0.0; });
    if (baseline == out.entries.end()) {
        throw std::logic_error("QDA sparsity CV grid has no zero baseline");
    }
    out.eligibility_threshold = baseline->mean_heldout_loss
        + baseline->standard_error;
    size_t selected = static_cast<size_t>(baseline - out.entries.begin());
    for (size_t index = 0; index < out.entries.size(); ++index) {
        QdaSparsityCvEntry& entry = out.entries[index];
        entry.loss_increase = entry.mean_heldout_loss
            - baseline->mean_heldout_loss;
        entry.eligible = entry.mean_heldout_loss
            <= out.eligibility_threshold
                + 64.0 * std::numeric_limits<double>::epsilon()
                    * std::max(1.0, std::abs(out.eligibility_threshold));
        if (entry.eligible
                && entry.strength > out.entries[selected].strength) {
            selected = index;
        }
    }
    out.entries[selected].selected = true;
    out.selected_strength = out.entries[selected].strength;
    return out;
}

void truncate_projection(punkst::projection::VisualizationProjection& projection,
        int32_t dimensions) {
    projection.eigenvalues.conservativeResize(dimensions);
    if (projection.axis_scores.size() > 0) {
        projection.axis_scores.conservativeResize(dimensions);
    }
    projection.projection.conservativeResize(
        projection.projection.rows(), dimensions);
    projection.topic_contrasts.conservativeResize(
        projection.topic_contrasts.rows(), dimensions);
    projection.component_means.conservativeResize(
        projection.component_means.rows(), dimensions);
    for (Eigen::MatrixXd& covariance : projection.component_covariances) {
        covariance = covariance.topLeftCorner(dimensions, dimensions).eval();
    }
}

int32_t positive_rank(const Eigen::Ref<const Eigen::VectorXd>& eigenvalues) {
    if (eigenvalues.size() == 0) return 0;
    const double scale = std::max(1.0,
        eigenvalues.cwiseAbs().maxCoeff());
    const double tolerance = 256.0
        * std::numeric_limits<double>::epsilon() * scale;
    int32_t rank = 0;
    for (Eigen::Index axis = 0; axis < eigenvalues.size(); ++axis) {
        if (eigenvalues(axis) > tolerance) ++rank;
    }
    return rank;
}

void write_axis_weights(const std::string& path,
    const std::vector<std::string>& topics,
    const punkst::projection::VisualizationProjection& projection) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write embedding axis weights: " + path);
    }
    if (projection.topic_contrasts.rows()
            != static_cast<Eigen::Index>(topics.size())) {
        throw std::invalid_argument("Invalid embedding axis-weight dimensions");
    }
    out << "#Cluster";
    for (Eigen::Index axis = 0;
            axis < projection.topic_contrasts.cols(); ++axis) {
        out << "\tw" << axis + 1 << "_p\tw" << axis + 1 << "_n";
    }
    out << "\n" << std::scientific << std::setprecision(10);
    Eigen::VectorXd scales(projection.topic_contrasts.cols());
    for (Eigen::Index axis = 0;
            axis < projection.topic_contrasts.cols(); ++axis) {
        scales(axis) = projection.topic_contrasts.col(axis)
            .cwiseMax(0.0).sum();
        if (!(scales(axis) > 0.0) || !std::isfinite(scales(axis))) {
            throw std::runtime_error(
                "Embedding contrast has no positive mass");
        }
    }
    for (Eigen::Index topic = 0;
            topic < projection.topic_contrasts.rows(); ++topic) {
        out << topics[static_cast<size_t>(topic)];
        for (Eigen::Index axis = 0;
                axis < projection.topic_contrasts.cols(); ++axis) {
            const double coefficient =
                projection.topic_contrasts(topic, axis);
            out << '\t' << std::max(0.0, coefficient) / scales(axis)
                << '\t' << std::max(0.0, -coefficient) / scales(axis);
        }
        out << '\n';
    }
}

void write_coordinates(const std::string& path,
    const std::vector<std::string>& identifiers,
    const std::vector<ProjectionOutput>& projections,
    const punkst::linear_embedding::ProjectionData* qda_data = nullptr,
    const Eigen::MatrixXd* qda_projection = nullptr) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error(
            "Cannot write linear embedding coordinates: " + path);
    }
    out << "#id";
    for (const ProjectionOutput& projection : projections) {
        const char* space = projection_space_name(projection.space);
        for (Eigen::Index axis = 0;
                axis < projection.visualization.mean.projection.cols();
                ++axis) {
            out << '\t' << space << "_mean_" << axis + 1;
        }
        for (Eigen::Index axis = 0;
                axis < projection.visualization.full.projection.cols();
                ++axis) {
            out << '\t' << space << "_full_" << axis + 1;
        }
    }
    if (qda_projection != nullptr) {
        for (Eigen::Index axis = 0; axis < qda_projection->cols(); ++axis) {
            out << "\tlinear_qda_" << axis + 1;
        }
    }
    out << "\n" << std::scientific << std::setprecision(4);
    for (Eigen::Index document = 0;
            document < static_cast<Eigen::Index>(identifiers.size());
            ++document) {
        out << identifiers[static_cast<size_t>(document)];
        for (const ProjectionOutput& projection : projections) {
            if (projection.data == nullptr
                || projection.data->coordinates.rows()
                    != static_cast<Eigen::Index>(identifiers.size())) {
                throw std::invalid_argument(
                    "Invalid embedding coordinate dimensions");
            }
            for (Eigen::Index axis = 0;
                    axis < projection.visualization.mean.projection.cols();
                    ++axis) {
                out << '\t' << projection.data->coordinates.row(document).dot(
                    projection.visualization.mean.projection.col(axis));
            }
            for (Eigen::Index axis = 0;
                    axis < projection.visualization.full.projection.cols();
                    ++axis) {
                out << '\t' << projection.data->coordinates.row(document).dot(
                    projection.visualization.full.projection.col(axis));
            }
        }
        if (qda_projection != nullptr) {
            if (qda_data == nullptr || qda_data->coordinates.rows()
                    != static_cast<Eigen::Index>(identifiers.size())
                || qda_data->coordinates.cols() != qda_projection->rows()) {
                throw std::invalid_argument("Invalid QDA coordinate dimensions");
            }
            for (Eigen::Index axis = 0; axis < qda_projection->cols(); ++axis) {
                out << '\t' << qda_data->coordinates.row(document).dot(
                    qda_projection->col(axis));
            }
        }
        out << '\n';
    }
}

void write_qda_transform(const std::string& path,
        const std::vector<std::string>& topics,
        const Eigen::Ref<const Eigen::MatrixXd>& projection,
        const Eigen::Ref<const Eigen::MatrixXd>& topic_contrasts) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write QDA transform: " + path);
    out << "#axis\tbasis\tindex\tname\tcoefficient\n"
        << std::scientific << std::setprecision(10);
    for (Eigen::Index axis = 0; axis < projection.cols(); ++axis) {
        for (Eigen::Index coordinate = 0; coordinate < projection.rows();
                ++coordinate) {
            out << axis + 1 << "\thelmert\t" << coordinate
                << "\thelmert_" << coordinate << '\t'
                << projection(coordinate, axis) << '\n';
        }
        for (Eigen::Index topic = 0; topic < topic_contrasts.rows(); ++topic) {
            out << axis + 1 << "\ttopic\t" << topic << '\t'
                << topics[static_cast<size_t>(topic)] << '\t'
                << topic_contrasts(topic, axis) << '\n';
        }
    }
}

void write_qda_sparsity_cv(const std::string& path,
        const QdaSparsityCvResult& result) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Cannot write QDA sparsity CV trace: " + path);
    }
    out << "#lambda\tfolds\tcv_rows\theldout_rows"
        "\tmean_heldout_logloss\tse_heldout_logloss"
        "\tloss_increase_from_zero\tmean_quartimax_score"
        "\tmean_inner_training_logloss\tmean_inner_validation_logloss"
        "\tmean_inner_validation_objective\teligibility_threshold"
        "\teligible\tselected\n"
        << std::scientific << std::setprecision(10);
    for (const QdaSparsityCvEntry& entry : result.entries) {
        out << entry.strength << '\t' << result.folds << '\t'
            << result.population_rows << '\t' << entry.heldout_rows << '\t'
            << entry.mean_heldout_loss << '\t' << entry.standard_error << '\t'
            << entry.loss_increase << '\t' << entry.mean_quartimax_score << '\t'
            << entry.mean_training_loss << '\t'
            << entry.mean_validation_loss << '\t'
            << entry.mean_validation_objective << '\t'
            << result.eligibility_threshold << '\t'
            << static_cast<int32_t>(entry.eligible) << '\t'
            << static_cast<int32_t>(entry.selected) << '\n';
    }
}

void write_qda_diagnostics(const std::string& path,
        const punkst::projection::QdaProjectionResult& result,
        const punkst::projection::QdaProjectionOptions& options,
        size_t training_rows, size_t validation_rows,
        int32_t training_hard_cap, int32_t training_adaptive_cap,
        int32_t training_effective_cap, const char* sparsity_selection) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write QDA diagnostics: " + path);
    out << "#training_rows\tvalidation_rows\tdimensions\trestart\tepoch"
        "\ttraining_hard_cap\ttraining_adaptive_cap\ttraining_effective_cap"
        "\ttraining_logloss\tvalidation_logloss\tseed\tepochs\trestarts"
        "\tlearning_rate\tcovariance_shrinkage\tridge\tevaluate_every"
        "\tpatience_checks\tsparsity_selection\tsparsity_strength"
        "\tquartimax_score\ttraining_objective\tvalidation_objective\n"
        << training_rows << '\t' << validation_rows << '\t'
        << result.projection.cols() << '\t' << result.restart << '\t'
        << result.epoch << '\t' << training_hard_cap << '\t'
        << training_adaptive_cap << '\t' << training_effective_cap << '\t'
        << std::scientific << std::setprecision(10)
        << result.training_loss << '\t' << result.validation_loss << '\t'
        << options.seed << '\t' << options.epochs << '\t' << options.restarts
        << '\t' << options.learning_rate << '\t'
        << options.covariance_shrinkage << '\t' << options.ridge << '\t'
        << options.evaluate_every << '\t' << options.patience_checks << '\t'
        << sparsity_selection << '\t' << options.sparsity_strength << '\t'
        << result.quartimax_score << '\t' << result.training_objective << '\t'
        << result.validation_objective << '\n';
}

} // namespace

const char* punkst::linear_embedding::projection_space_name(
        ProjectionSpace space) {
    return space == ProjectionSpace::Linear ? "linear" : "ilr";
}

punkst::linear_embedding::ProjectionData
punkst::linear_embedding::prepare_projection(
        const Eigen::Ref<const RowMajorMatrixXd>& values,
        ProjectionSpace space,
        const Eigen::Ref<const Eigen::MatrixXd>& helmert,
        double center_floor) {
    ProjectionData out;
    if (space == ProjectionSpace::Linear) {
        out.centers = normalize_factor_proportions(values);
        out.coordinates = out.centers * helmert.transpose();
    } else {
        out.centers = values;
        normalize_compositions(out.centers, center_floor);
        out.coordinates = ilr_transform(out.centers, helmert);
    }
    return out;
}

Eigen::MatrixXd punkst::linear_embedding::aggregate_cluster_factors(
        const Eigen::Ref<const RowMajorMatrixXd>& values,
        const Eigen::Ref<const Eigen::VectorXi>& assignments,
        int32_t components) {
    if (assignments.size() != values.rows()) {
        throw std::invalid_argument(
            "Aligned cluster factor rows and assignments differ in size");
    }
    return aggregate_cluster_factors_impl(values, assignments, components,
        [](Eigen::Index row) { return row; });
}

Eigen::MatrixXd punkst::linear_embedding::aggregate_cluster_factors(
        const Eigen::Ref<const RowMajorMatrixXd>& values,
        const std::vector<int32_t>& value_rows,
        const Eigen::Ref<const Eigen::VectorXi>& assignments,
        int32_t components) {
    if (value_rows.size() != static_cast<size_t>(assignments.size())) {
        throw std::invalid_argument(
            "Matched cluster factor rows and assignments differ in size");
    }
    return aggregate_cluster_factors_impl(values, assignments, components,
        [&value_rows](Eigen::Index row) {
            return static_cast<Eigen::Index>(
                value_rows[static_cast<size_t>(row)]);
        });
}

void punkst::linear_embedding::write_cluster_factors(
        const std::string& path,
        const std::vector<std::string>& factor_names,
        const Eigen::Ref<const Eigen::MatrixXd>& sums) {
    if (sums.rows() != static_cast<Eigen::Index>(factor_names.size())
            || sums.cols() <= 0 || !sums.allFinite()) {
        throw std::invalid_argument("Invalid cluster factor sums");
    }
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error(
            "Cannot open cluster factor output: " + path);
    }
    output << "#factor";
    for (Eigen::Index cluster = 0; cluster < sums.cols(); ++cluster) {
        output << "\tcluster_" << cluster;
    }
    output << '\n' << std::setprecision(6);
    for (Eigen::Index factor = 0; factor < sums.rows(); ++factor) {
        output << factor_names[static_cast<size_t>(factor)];
        for (Eigen::Index cluster = 0; cluster < sums.cols(); ++cluster) {
            output << '\t' << sums(factor, cluster);
        }
        output << '\n';
    }
}

void punkst::linear_embedding::Options::validate() {
    if (dimensions <= 0 || threads <= 0
            || !(covariance_floor > 0.0)
            || !std::isfinite(covariance_floor)) {
        throw std::invalid_argument("Invalid linear embedding options");
    }
    if (projection_spaces.empty()
            || std::find(projection_spaces.begin(), projection_spaces.end(),
                ProjectionSpace::Linear) == projection_spaces.end()) {
        throw std::invalid_argument(
            "Linear embedding requires the linear projection space");
    }
    if (std::find(projection_spaces.begin(), projection_spaces.end(),
            ProjectionSpace::Ilr) != projection_spaces.end()
            && (!(center_floor > 0.0) || !std::isfinite(center_floor))) {
        throw std::invalid_argument(
            "Projection center floor must be positive and finite");
    }
    if (!qda_projection) return;
    if (qda_train_max_rows < 0 || qda_validation_max_rows < 0
            || !(qda_validation_fraction > 0.0)
            || !(qda_validation_fraction < 1.0)
            || qda_epochs <= 0 || qda_restarts <= 0
            || qda_evaluate_every <= 0 || qda_patience <= 0
            || qda_seed < 0 || !(qda_learning_rate > 0.0)
            || !(qda_covariance_shrinkage >= 0.0)
            || !(qda_covariance_shrinkage <= 1.0)
            || !(qda_ridge > 0.0)) {
        throw std::invalid_argument("Invalid QDA projection options");
    }
    if (!(qda_sparsity_strength >= 0.0)
            || !std::isfinite(qda_sparsity_strength)
            || qda_sparsity_cv_folds < 2) {
        throw std::invalid_argument("Invalid QDA sparsity options");
    }
    if (qda_sparsity_cv && qda_sparsity_grid.empty()) {
        qda_sparsity_grid = {
            0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0};
    }
    if (qda_sparsity_cv) {
        for (const double strength : qda_sparsity_grid) {
            if (!(strength >= 0.0) || !std::isfinite(strength)) {
                throw std::invalid_argument(
                    "QDA sparsity grid must be finite and nonnegative");
            }
        }
        std::sort(qda_sparsity_grid.begin(), qda_sparsity_grid.end());
        const auto duplicate = std::adjacent_find(
            qda_sparsity_grid.begin(), qda_sparsity_grid.end());
        if (duplicate != qda_sparsity_grid.end()
                || qda_sparsity_grid.size() < 2
                || qda_sparsity_grid.front() != 0.0
                || !(qda_sparsity_grid.back() > 0.0)) {
            throw std::invalid_argument(
                "QDA sparsity grid requires unique values including zero and a positive candidate");
        }
    }
}

void punkst::linear_embedding::run_partition(
        const TopicCenterTable& theta,
        const std::vector<int32_t>& matched_theta_rows,
        const Eigen::Ref<const Eigen::VectorXi>& assignments,
        int32_t components, const std::string& partition_label,
        const std::string& output_prefix,
        const Options& supplied_options) {
    Options pipeline = supplied_options;
    pipeline.validate();
    if (theta.values.rows() != static_cast<Eigen::Index>(
            theta.identifiers.size())
            || theta.values.cols() != static_cast<Eigen::Index>(
                theta.topics.size())
            || assignments.size() != static_cast<Eigen::Index>(
                matched_theta_rows.size())
            || components < 2) {
        throw std::invalid_argument("Invalid linear embedding partition input");
    }
    for (Eigen::Index row = 0; row < assignments.size(); ++row) {
        if (assignments(row) < 0 || assignments(row) >= components
                || matched_theta_rows[static_cast<size_t>(row)] < 0
                || matched_theta_rows[static_cast<size_t>(row)]
                    >= theta.values.rows()) {
            throw std::invalid_argument(
                "Invalid linear embedding assignment or row mapping");
        }
    }
    const int32_t topics = static_cast<int32_t>(theta.values.cols());
    if (topics < 2 || assignments.size() < topics) {
        throw std::runtime_error(
            "Linear embedding partition has too few matched units");
    }
    const Eigen::MatrixXd helmert = normalized_helmert(topics);
    std::vector<ProjectionInput> inputs;
    inputs.reserve(pipeline.projection_spaces.size());
    for (const ProjectionSpace space : pipeline.projection_spaces) {
        ProjectionInput input;
        input.space = space;
        input.data = prepare_projection(theta.values, space, helmert,
            pipeline.center_floor);
        input.matched_coordinates.resize(
            matched_theta_rows.size(), input.data.coordinates.cols());
        for (size_t matched = 0; matched < matched_theta_rows.size();
                ++matched) {
            input.matched_coordinates.row(static_cast<Eigen::Index>(matched)) =
                input.data.coordinates.row(matched_theta_rows[matched]);
        }
        if (pipeline.whitening
                == punkst::projection::VisualizationWhitening::Sample) {
            input.sample_moments =
                punkst::projection::summarize_visualization_sample(
                    input.data.coordinates, pipeline.threads);
        }
        inputs.push_back(std::move(input));
    }

    const ProjectionInput* linear_input = nullptr;
    for (const ProjectionInput& input : inputs) {
        if (input.space == ProjectionSpace::Linear) {
            linear_input = &input;
            break;
        }
    }
    std::optional<punkst::projection::QdaProjectionResult> qda_result;
    std::optional<QdaRows> qda_rows;
    punkst::projection::QdaProjectionOptions qda_options;
    if (pipeline.qda_projection && topics < 3) {
        warning("QDA projection partition %s omitted: at least three topics are required",
            partition_label.c_str());
    } else if (pipeline.qda_projection) {
        Eigen::VectorXi cluster_counts = Eigen::VectorXi::Zero(components);
        for (Eigen::Index row = 0; row < assignments.size(); ++row) {
            ++cluster_counts(assignments(row));
        }
        std::vector<int32_t> qda_component_map(
            static_cast<size_t>(components), -1);
        int32_t qda_components = 0;
        for (int32_t component = 0; component < components; ++component) {
            if (cluster_counts(component) >= minimum_qda_cluster_rows) {
                qda_component_map[static_cast<size_t>(component)] =
                    qda_components++;
            }
        }
        notice("QDA projection partition %s: %d of %d clusters enter optimization; clusters with <= 10 matched rows are discarded",
            partition_label.c_str(), qda_components, components);
        if (qda_components < 2) {
            warning("QDA projection partition %s omitted: fewer than two clusters have more than 10 matched rows",
                partition_label.c_str());
        } else {
            std::vector<int32_t> qda_documents;
            qda_documents.reserve(static_cast<size_t>(assignments.size()));
            for (Eigen::Index document = 0; document < assignments.size();
                    ++document) {
                if (qda_component_map[static_cast<size_t>(
                        assignments(document))] >= 0) {
                    qda_documents.push_back(static_cast<int32_t>(document));
                }
            }
            const RowMajorMatrixXd qda_coordinates = select_rows(
                linear_input->matched_coordinates, qda_documents);
            Eigen::VectorXi qda_assignments(qda_documents.size());
            for (size_t row = 0; row < qda_documents.size(); ++row) {
                qda_assignments(static_cast<Eigen::Index>(row)) =
                    qda_component_map[static_cast<size_t>(assignments(
                        qda_documents[row]))];
            }
            qda_options.dimensions = std::min(
                pipeline.dimensions, topics - 2);
            qda_options.epochs = pipeline.qda_epochs;
            qda_options.learning_rate = pipeline.qda_learning_rate;
            qda_options.covariance_shrinkage =
                pipeline.qda_covariance_shrinkage;
            qda_options.ridge = pipeline.qda_ridge;
            qda_options.restarts = pipeline.qda_restarts;
            qda_options.evaluate_every = pipeline.qda_evaluate_every;
            qda_options.patience_checks = pipeline.qda_patience;
            qda_options.seed = pipeline.qda_seed;
            qda_options.n_threads = pipeline.threads;
            qda_options.sparsity_strength =
                pipeline.qda_sparsity_strength;
            if (qda_options.dimensions < pipeline.dimensions) {
                notice("QDA projection partition %s resolved dimensions: %d (requested %d)",
                    partition_label.c_str(), qda_options.dimensions,
                    pipeline.dimensions);
            }
            const int32_t adaptive_training_cap = adaptive_qda_training_cap(
                static_cast<int32_t>(qda_coordinates.cols()),
                qda_options.dimensions, qda_components);
            const int32_t effective_training_cap =
                pipeline.qda_train_max_rows > 0
                ? std::min(pipeline.qda_train_max_rows,
                    adaptive_training_cap)
                : adaptive_training_cap;
            notice("QDA projection partition %s training cap: adaptive %d, hard %d, effective %d",
                partition_label.c_str(), adaptive_training_cap,
                pipeline.qda_train_max_rows, effective_training_cap);
            std::optional<QdaSparsityCvResult> cv_result;
            if (pipeline.qda_sparsity_cv) {
                notice("QDA sparsity CV partition %s: testing %zu strengths with up to %d outer folds",
                    partition_label.c_str(),
                    pipeline.qda_sparsity_grid.size(),
                    pipeline.qda_sparsity_cv_folds);
                cv_result = cross_validate_qda_sparsity(
                    qda_coordinates, qda_assignments, qda_components,
                    pipeline.qda_sparsity_grid,
                    pipeline.qda_sparsity_cv_folds,
                    pipeline.qda_validation_fraction,
                    effective_training_cap,
                    pipeline.qda_validation_max_rows,
                    pipeline.qda_seed, qda_options);
                qda_options.sparsity_strength = cv_result->selected_strength;
                notice("QDA sparsity CV partition %s: selected lambda %.10g; eligibility threshold %.10g; folds %d; rows %zu",
                    partition_label.c_str(), cv_result->selected_strength,
                    cv_result->eligibility_threshold, cv_result->folds,
                    cv_result->population_rows);
            }
            qda_rows = make_qda_rows(qda_assignments, qda_components,
                pipeline.qda_validation_fraction, effective_training_cap,
                pipeline.qda_validation_max_rows, pipeline.qda_seed);
            const RowMajorMatrixXd training = select_rows(
                qda_coordinates, qda_rows->training);
            const RowMajorMatrixXd validation = select_rows(
                qda_coordinates, qda_rows->validation);
            const Eigen::VectorXi training_labels = select_labels(
                qda_assignments, qda_rows->training);
            const Eigen::VectorXi validation_labels = select_labels(
                qda_assignments, qda_rows->validation);
            qda_result = punkst::projection::fit_qda_projection(
                training, training_labels, validation, validation_labels,
                qda_components, qda_options);
            const Eigen::MatrixXd topic_contrasts = helmert.transpose()
                * qda_result->projection;
            write_qda_transform(output_prefix + ".linear.qda.transform.tsv",
                theta.topics, qda_result->projection, topic_contrasts);
            punkst::projection::VisualizationProjection axes;
            axes.projection = qda_result->projection;
            axes.topic_contrasts = topic_contrasts;
            write_axis_weights(output_prefix + ".linear.qda.axes.tsv",
                theta.topics, axes);
            if (cv_result) {
                write_qda_sparsity_cv(
                    output_prefix + ".linear.qda.sparsity_cv.tsv",
                    *cv_result);
            }
            write_qda_diagnostics(
                output_prefix + ".linear.qda.diagnostics.tsv",
                *qda_result, qda_options, qda_rows->training.size(),
                qda_rows->validation.size(), pipeline.qda_train_max_rows,
                adaptive_training_cap, effective_training_cap,
                pipeline.qda_sparsity_cv ? "cv"
                    : qda_options.sparsity_strength > 0.0 ? "fixed" : "none");
        }
    }

    punkst::projection::VisualizationOptions visualization_options;
    visualization_options.whitening = pipeline.whitening;
    visualization_options.dimensions = std::min(
        pipeline.dimensions, topics - 1);
    visualization_options.n_threads = pipeline.threads;
    visualization_options.covariance_floor = pipeline.covariance_floor;
    visualization_options.include_full = pipeline.include_full;
    std::vector<ProjectionOutput> projections;
    projections.reserve(inputs.size());
    for (ProjectionInput& input : inputs) {
        punkst::projection::VisualizationResult visualization;
        if (pipeline.include_full) {
            const punkst::projection::VisualizationMoments moments =
                punkst::projection::summarize_hard_partition(
                    input.matched_coordinates, assignments,
                    components, pipeline.threads);
            visualization = input.sample_moments
                ? punkst::projection::make_visualization(
                    input.data.coordinates, moments, helmert,
                    visualization_options, *input.sample_moments)
                : punkst::projection::make_visualization(
                    input.data.coordinates, moments, helmert,
                    visualization_options);
        } else {
            const punkst::projection::VisualizationMeans means =
                punkst::projection::summarize_hard_partition_means(
                    input.matched_coordinates, assignments, components);
            punkst::projection::VisualizationOptions mean_options =
                visualization_options;
            mean_options.dimensions = std::min(
                mean_options.dimensions, components - 1);
            const punkst::projection::VisualizationSampleMoments moments =
                input.sample_moments ? *input.sample_moments
                : punkst::projection::summarize_visualization_sample(
                    input.matched_coordinates, pipeline.threads);
            visualization = punkst::projection::make_mean_visualization(
                means, helmert, mean_options, moments);
        }
        const int32_t mean_dimensions = pipeline.include_full
            ? std::min({visualization_options.dimensions, components - 1,
                positive_rank(visualization.mean.eigenvalues)})
            : static_cast<int32_t>(visualization.mean.projection.cols());
        if (mean_dimensions <= 0) {
            throw std::runtime_error(
                "Hard partition has no positive mean-separation axis");
        }
        if (pipeline.include_full) {
            truncate_projection(visualization.mean, mean_dimensions);
        }
        int32_t full_dimensions = 0;
        if (pipeline.include_full) {
            full_dimensions = std::min(visualization_options.dimensions,
                positive_rank(visualization.full.eigenvalues));
            if (full_dimensions <= 0) {
                throw std::runtime_error(
                    "Hard partition has no positive full-separation axis");
            }
            truncate_projection(visualization.full, full_dimensions);
        }
        punkst::projection::quartimax_rotate(visualization.mean);
        if (pipeline.include_full) {
            punkst::projection::quartimax_rotate(visualization.full);
        }
        if (mean_dimensions < pipeline.dimensions
                || (pipeline.include_full
                    && full_dimensions < pipeline.dimensions)) {
            if (pipeline.include_full) {
                notice("Linear embedding partition %s %s-space resolved dimensions: mean %d, full %d (requested %d)",
                    partition_label.c_str(),
                    projection_space_name(input.space), mean_dimensions,
                    full_dimensions, pipeline.dimensions);
            } else {
                notice("Linear embedding partition %s %s-space resolved dimensions: mean %d (requested %d)",
                    partition_label.c_str(),
                    projection_space_name(input.space), mean_dimensions,
                    pipeline.dimensions);
            }
        }
        const std::string space_prefix = output_prefix + "."
            + projection_space_name(input.space);
        punkst::projection::write_visualization_axes(
            space_prefix + ".transform.tsv", theta.topics, visualization);
        write_axis_weights(space_prefix + ".mean.axes.tsv",
            theta.topics, visualization.mean);
        if (pipeline.include_full) {
            write_axis_weights(space_prefix + ".full.axes.tsv",
                theta.topics, visualization.full);
        }
        projections.push_back(
            {input.space, &input.data, std::move(visualization)});
    }
    write_coordinates(output_prefix + ".results.tsv", theta.identifiers,
        projections, qda_result ? &linear_input->data : nullptr,
        qda_result ? &qda_result->projection : nullptr);
    notice("Linear embedding wrote %zu projection space(s) under %s and coordinates to %s.results.tsv",
        projections.size(), output_prefix.c_str(), output_prefix.c_str());
}
