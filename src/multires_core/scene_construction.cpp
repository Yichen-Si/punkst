#include "multires_core/scene_construction.hpp"

#include "multires_core/raw_affinity_graph.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

int32_t canonical_scene_count(const std::vector<int32_t>& membership) {
    if (membership.empty()) {
        throw std::invalid_argument("Scene membership is empty");
    }
    const int32_t maximum = *std::max_element(
        membership.begin(), membership.end());
    if (maximum < 0) {
        throw std::invalid_argument("Scene membership must be nonnegative");
    }
    std::vector<uint8_t> used(static_cast<size_t>(maximum + 1), 0);
    for (const int32_t scene : membership) {
        if (scene < 0) {
            throw std::invalid_argument(
                "Scene membership must be nonnegative");
        }
        used[static_cast<size_t>(scene)] = 1;
    }
    if (std::find(used.begin(), used.end(), uint8_t{0}) != used.end()) {
        throw std::invalid_argument(
            "Scene membership labels must be contiguous");
    }
    return maximum + 1;
}

void validate_scene_options(const SceneConstructionOptions& options) {
    if (options.minimum_scene_core_members < 1
        || !(options.halo_minimum_score >= 0.0)
        || !(options.halo_minimum_score <= 1.0)
        || !(options.halo_relative_to_core >= 0.0)
        || options.maximum_halo_scenes < 0
        || !(options.portal_minimum_child_fraction >= 0.0)
        || !(options.portal_minimum_child_fraction <= 1.0)
        || !std::isfinite(options.halo_minimum_score)
        || !std::isfinite(options.halo_relative_to_core)
        || !std::isfinite(options.portal_minimum_child_fraction)) {
        throw std::invalid_argument("Invalid scene construction options");
    }
}

void validate_fine_raw_graph(const RawClusteringGraph& graph) {
    if (graph.n_nodes <= 0
        || graph.edges.size() != graph.weights.size()
        || graph.component_labels.size()
            != static_cast<size_t>(graph.n_nodes)) {
        throw std::invalid_argument("Invalid fine raw-affinity graph");
    }
    if (std::any_of(graph.component_labels.begin(),
        graph.component_labels.end(), [](int32_t component) {
            return component < 0;
        })) {
        throw std::invalid_argument("Invalid fine raw-affinity component");
    }
    for (size_t edge = 0; edge < graph.edges.size(); ++edge) {
        const int32_t first = graph.edges[edge].first;
        const int32_t second = graph.edges[edge].second;
        if (first < 0 || second < first || second >= graph.n_nodes
            || !(graph.weights[edge] >= 0.0)
            || !std::isfinite(graph.weights[edge])
            || graph.component_labels[static_cast<size_t>(first)]
                != graph.component_labels[static_cast<size_t>(second)]) {
            throw std::invalid_argument("Invalid fine raw-affinity edge");
        }
    }
}

std::vector<int32_t> microcluster_components(
        const std::vector<int32_t>& fine_to_microcluster,
        const std::vector<int32_t>& fine_components,
        int32_t microclusters) {
    if (fine_to_microcluster.size() != fine_components.size()) {
        throw std::invalid_argument(
            "Fine memberships and components must align");
    }
    std::vector<int32_t> output(static_cast<size_t>(microclusters), -1);
    for (size_t point = 0; point < fine_to_microcluster.size(); ++point) {
        const int32_t microcluster = fine_to_microcluster[point];
        const int32_t component = fine_components[point];
        if (microcluster < 0 || microcluster >= microclusters
            || component < 0) {
            throw std::invalid_argument(
                "Fine microcluster/component label is invalid");
        }
        int32_t& stored = output[static_cast<size_t>(microcluster)];
        if (stored < 0) stored = component;
        else if (stored != component) {
            throw std::invalid_argument(
                "A microcluster crosses non-bridge components");
        }
    }
    if (std::find(output.begin(), output.end(), -1) != output.end()) {
        throw std::invalid_argument("A microcluster has no fine points");
    }
    return output;
}

std::vector<int32_t> scene_components(
        const std::vector<int32_t>& micro_membership,
        const std::vector<int32_t>& micro_components,
        int32_t scenes) {
    std::vector<int32_t> output(static_cast<size_t>(scenes), -1);
    for (size_t microcluster = 0;
            microcluster < micro_membership.size(); ++microcluster) {
        const int32_t scene = micro_membership[microcluster];
        int32_t& stored = output[static_cast<size_t>(scene)];
        const int32_t component = micro_components[microcluster];
        if (stored < 0) stored = component;
        else if (stored != component) {
            throw std::invalid_argument(
                "A scene crosses non-bridge components");
        }
    }
    return output;
}

std::vector<uint8_t> tail_scenes(
        const std::vector<int64_t>& counts) {
    std::vector<int32_t> order(counts.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t first, int32_t second) {
        return counts[static_cast<size_t>(first)]
                == counts[static_cast<size_t>(second)]
            ? first < second
            : counts[static_cast<size_t>(first)]
                > counts[static_cast<size_t>(second)];
    });
    const int64_t total = std::accumulate(
        counts.begin(), counts.end(), int64_t{0});
    int64_t cumulative = 0;
    std::vector<uint8_t> tail(counts.size(), 1);
    for (const int32_t scene : order) {
        if (static_cast<long double>(cumulative)
                >= 0.9L * static_cast<long double>(total)) break;
        tail[static_cast<size_t>(scene)] = 0;
        cumulative += counts[static_cast<size_t>(scene)];
    }
    return tail;
}

std::vector<std::unordered_map<int32_t, double>> raw_halo_scores(
        const RawClusteringGraph& graph,
        const std::vector<int32_t>& core_membership) {
    std::vector<std::unordered_map<int32_t, double>> scores(
        static_cast<size_t>(graph.n_nodes));
    std::vector<double> totals(static_cast<size_t>(graph.n_nodes), 0.0);
    for (size_t index = 0; index < graph.edges.size(); ++index) {
        const int32_t first = graph.edges[index].first;
        const int32_t second = graph.edges[index].second;
        const double weight = graph.weights[index];
        if (first == second) {
            scores[static_cast<size_t>(first)][
                core_membership[static_cast<size_t>(first)]] += weight;
            totals[static_cast<size_t>(first)] += weight;
        } else {
            scores[static_cast<size_t>(first)][
                core_membership[static_cast<size_t>(second)]] += weight;
            scores[static_cast<size_t>(second)][
                core_membership[static_cast<size_t>(first)]] += weight;
            totals[static_cast<size_t>(first)] += weight;
            totals[static_cast<size_t>(second)] += weight;
        }
    }
    for (int32_t point = 0; point < graph.n_nodes; ++point) {
        const double total = totals[static_cast<size_t>(point)];
        if (!(total > 0.0)) continue;
        for (auto& item : scores[static_cast<size_t>(point)]) {
            item.second /= total;
        }
    }
    return scores;
}

std::vector<int32_t> probability_predictions(
        const Eigen::Ref<const RowMajorMatrixXd>& probabilities,
        const std::vector<int32_t>& components,
        const std::vector<int32_t>& scene_component,
        const std::vector<uint8_t>& eligible,
        const std::vector<int32_t>* inherited = nullptr) {
    std::vector<int32_t> output(static_cast<size_t>(probabilities.rows()), -1);
    for (int32_t row = 0; row < probabilities.rows(); ++row) {
        if (inherited != nullptr
            && !eligible[static_cast<size_t>((*inherited)[row])]) {
            output[static_cast<size_t>(row)] = (*inherited)[row];
            continue;
        }
        int32_t selected = -1;
        double best = -1.0;
        double total = 0.0;
        for (int32_t scene = 0; scene < probabilities.cols(); ++scene) {
            if (!eligible[static_cast<size_t>(scene)]
                || scene_component[static_cast<size_t>(scene)]
                    != components[static_cast<size_t>(row)]) continue;
            const double value = probabilities(row, scene);
            total += value;
            if (value > best || (value == best && scene < selected)) {
                best = value;
                selected = scene;
            }
        }
        if ((selected < 0 || !(total > 0.0)) && inherited != nullptr) {
            selected = (*inherited)[row];
        }
        output[static_cast<size_t>(row)] = selected;
    }
    return output;
}

} // namespace

const char* scene_core_mode_name(SceneCoreMode mode) {
    switch (mode) {
        case SceneCoreMode::Inherit: return "inherit";
        case SceneCoreMode::ClassifierPlugin: return "classifier-plugin";
        case SceneCoreMode::ClassifierLrvb: return "classifier-lrvb";
    }
    throw std::invalid_argument("Unknown scene core mode");
}

SceneClassifierResult fit_plugin_scene_classifier(
        const Eigen::Ref<const RowMajorMatrixXd>& fine_compositions,
        const std::vector<int32_t>& representative_rows,
        const std::vector<int32_t>& microcluster_membership,
        const std::vector<int64_t>& microcluster_fine_counts,
        const std::vector<int32_t>& fine_component_labels,
        const SceneClassifierOptions& options) {
    SceneClassifierResult result;
    result.mode = SceneCoreMode::ClassifierPlugin;
    result.attempted = true;
    const int32_t microclusters = static_cast<int32_t>(
        microcluster_membership.size());
    const int32_t scenes = canonical_scene_count(microcluster_membership);
    if (fine_compositions.rows() < 2 || fine_compositions.cols() < 2
        || representative_rows.size() != static_cast<size_t>(microclusters)
        || microcluster_fine_counts.size()
            != static_cast<size_t>(microclusters)
        || fine_component_labels.size()
            != static_cast<size_t>(fine_compositions.rows())
        || options.minimum_representatives_per_scene < 3
        || !(options.minimum_crossfit_ari >= -1.0)
        || !(options.minimum_crossfit_ari <= 1.0)
        || !(options.minimum_scene_recall >= 0.0)
        || !(options.minimum_scene_recall <= 1.0)
        || !fine_compositions.allFinite()
        || (fine_compositions.array() < 0.0).any()
        || (fine_compositions.rowwise().sum().array() <= 0.0).any()) {
        throw std::invalid_argument("Invalid scene classifier input");
    }
    std::vector<int32_t> representative_components(
        static_cast<size_t>(microclusters));
    std::vector<int32_t> per_scene(static_cast<size_t>(scenes), 0);
    for (int32_t microcluster = 0; microcluster < microclusters;
            ++microcluster) {
        const int32_t row = representative_rows[static_cast<size_t>(microcluster)];
        if (row < 0 || row >= fine_compositions.rows()
            || microcluster_fine_counts[static_cast<size_t>(microcluster)] <= 0) {
            throw std::invalid_argument(
                "Invalid representative row or microcluster count");
        }
        representative_components[static_cast<size_t>(microcluster)] =
            fine_component_labels[static_cast<size_t>(row)];
        ++per_scene[static_cast<size_t>(
            microcluster_membership[static_cast<size_t>(microcluster)])];
    }
    const std::vector<int32_t> per_scene_component = scene_components(
        microcluster_membership, representative_components, scenes);
    result.eligible_scene.assign(static_cast<size_t>(scenes), 0);
    std::unordered_map<int32_t, int32_t> eligible_per_component;
    for (int32_t scene = 0; scene < scenes; ++scene) {
        if (per_scene[static_cast<size_t>(scene)]
                >= options.minimum_representatives_per_scene) {
            ++eligible_per_component[
                per_scene_component[static_cast<size_t>(scene)]];
        }
    }
    std::vector<int32_t> scene_to_class(static_cast<size_t>(scenes), -1);
    std::vector<int32_t> class_to_scene;
    for (int32_t scene = 0; scene < scenes; ++scene) {
        const int32_t component = per_scene_component[static_cast<size_t>(scene)];
        if (per_scene[static_cast<size_t>(scene)]
                >= options.minimum_representatives_per_scene
            && eligible_per_component[component] >= 2) {
            scene_to_class[static_cast<size_t>(scene)] =
                static_cast<int32_t>(class_to_scene.size());
            class_to_scene.push_back(scene);
            result.eligible_scene[static_cast<size_t>(scene)] = 1;
        }
    }
    if (class_to_scene.size() < 2) {
        result.fallback_reason = "fewer_than_two_estimable_scenes";
        return result;
    }
    std::vector<int32_t> selected_microclusters;
    for (int32_t microcluster = 0; microcluster < microclusters;
            ++microcluster) {
        if (result.eligible_scene[static_cast<size_t>(
                microcluster_membership[static_cast<size_t>(microcluster)])]) {
            selected_microclusters.push_back(microcluster);
        }
    }
    RowMajorMatrixXd x(selected_microclusters.size(), fine_compositions.cols());
    Eigen::VectorXi labels(selected_microclusters.size());
    Eigen::VectorXd weights = Eigen::VectorXd::Ones(
        selected_microclusters.size());
    std::vector<std::string> identifiers(selected_microclusters.size());
    std::vector<int64_t> audit_counts(selected_microclusters.size());
    for (size_t index = 0; index < selected_microclusters.size(); ++index) {
        const int32_t microcluster = selected_microclusters[index];
        x.row(index) = fine_compositions.row(
            representative_rows[static_cast<size_t>(microcluster)]);
        labels(index) = scene_to_class[static_cast<size_t>(
            microcluster_membership[static_cast<size_t>(microcluster)])];
        identifiers[index] = "microcluster_" + std::to_string(microcluster);
        audit_counts[index] =
            microcluster_fine_counts[static_cast<size_t>(microcluster)];
    }
    std::vector<std::string> topics(static_cast<size_t>(fine_compositions.cols()));
    for (size_t topic = 0; topic < topics.size(); ++topic) {
        topics[topic] = "factor_" + std::to_string(topic);
    }
    std::vector<std::string> classes(class_to_scene.size());
    for (size_t index = 0; index < classes.size(); ++index) {
        classes[index] = "scene_" + std::to_string(class_to_scene[index]);
    }
    punkst::partition_classifier::CrossfitResult crossfit;
    try {
        crossfit = punkst::partition_classifier::fit_crossfit(
            x, labels, weights, identifiers, topics, classes,
            options.seed, options.fit);
    } catch (const std::exception& error) {
        result.fallback_reason =
            std::string("classifier_crossfit_failed: ") + error.what();
        return result;
    }
    std::vector<int32_t> truth(selected_microclusters.size());
    std::vector<int32_t> predicted_class(selected_microclusters.size());
    result.scene_recall.assign(static_cast<size_t>(scenes),
        std::numeric_limits<double>::quiet_NaN());
    std::vector<int64_t> correct(static_cast<size_t>(scenes), 0);
    std::vector<int64_t> total(static_cast<size_t>(scenes), 0);
    for (size_t index = 0; index < selected_microclusters.size(); ++index) {
        const int32_t microcluster = selected_microclusters[index];
        const int32_t component =
            representative_components[static_cast<size_t>(microcluster)];
        int32_t selected_class = -1;
        double best = -1.0;
        for (int32_t candidate = 0;
                candidate < crossfit.probabilities.cols(); ++candidate) {
            const int32_t scene = class_to_scene[static_cast<size_t>(candidate)];
            if (per_scene_component[static_cast<size_t>(scene)] != component) {
                continue;
            }
            const double value = crossfit.probabilities(index, candidate);
            if (value > best) {
                best = value;
                selected_class = candidate;
            }
        }
        if (selected_class < 0) {
            throw std::runtime_error(
                "Classifier audit has no component-local candidate");
        }
        truth[index] = labels(index);
        predicted_class[index] = selected_class;
        const int32_t scene = class_to_scene[static_cast<size_t>(labels(index))];
        total[static_cast<size_t>(scene)] += audit_counts[index];
        if (selected_class == labels(index)) {
            correct[static_cast<size_t>(scene)] += audit_counts[index];
        }
    }
    result.crossfit_ari = fine_weighted_adjusted_rand_index(
        truth, predicted_class, audit_counts);
    bool recall_passed = true;
    for (const int32_t scene : class_to_scene) {
        result.scene_recall[static_cast<size_t>(scene)] =
            static_cast<double>(correct[static_cast<size_t>(scene)])
            / static_cast<double>(total[static_cast<size_t>(scene)]);
        recall_passed = recall_passed
            && result.scene_recall[static_cast<size_t>(scene)]
                >= options.minimum_scene_recall;
    }
    if (result.crossfit_ari < options.minimum_crossfit_ari
        || !recall_passed) {
        result.fallback_reason = result.crossfit_ari
                < options.minimum_crossfit_ari
            ? "crossfit_ari_below_threshold"
            : "scene_recall_below_threshold";
        return result;
    }
    punkst::partition_classifier::FitResult fitted;
    try {
        fitted = punkst::partition_classifier::fit(
            x, labels, weights, identifiers, topics, classes,
            options.seed, options.fit);
    } catch (const std::exception& error) {
        result.fallback_reason =
            std::string("classifier_refit_failed: ") + error.what();
        return result;
    }
    result.fine_probabilities = RowMajorMatrixXd::Zero(
        fine_compositions.rows(), scenes);
    for (int32_t point = 0; point < fine_compositions.rows(); ++point) {
        const Eigen::VectorXd probabilities = fitted.model.probabilities(
            fine_compositions.row(point).transpose());
        double normalization = 0.0;
        for (int32_t candidate = 0; candidate < probabilities.size();
                ++candidate) {
            const int32_t scene = class_to_scene[static_cast<size_t>(candidate)];
            if (per_scene_component[static_cast<size_t>(scene)]
                    != fine_component_labels[static_cast<size_t>(point)]) {
                continue;
            }
            result.fine_probabilities(point, scene) = probabilities(candidate);
            normalization += probabilities(candidate);
        }
        if (normalization > 0.0) {
            result.fine_probabilities.row(point) /= normalization;
        }
    }
    result.plugin_model = fitted.model;
    result.model_class_to_scene = class_to_scene;
    result.plugin_model_available = true;
    result.passed = true;
    return result;
}

SceneClassifierResult use_lrvb_scene_probabilities(
        const SceneClassifierResult& audited_plugin,
        const Eigen::Ref<const RowMajorMatrixXd>& fine_probabilities) {
    if (!audited_plugin.attempted || !audited_plugin.passed
        || !audited_plugin.plugin_model_available
        || audited_plugin.mode != SceneCoreMode::ClassifierPlugin
        || fine_probabilities.rows()
            != audited_plugin.fine_probabilities.rows()
        || fine_probabilities.cols()
            != audited_plugin.fine_probabilities.cols()
        || !fine_probabilities.allFinite()
        || (fine_probabilities.array() < 0.0).any()
        || (fine_probabilities.rowwise().sum().array() <= 0.0).any()) {
        throw std::invalid_argument("Invalid LRVB scene probabilities");
    }
    SceneClassifierResult output = audited_plugin;
    output.mode = SceneCoreMode::ClassifierLrvb;
    output.fine_probabilities = fine_probabilities;
    return output;
}

FineSceneLevel construct_fine_scene_level(
        const SceneLevelMetadata& metadata,
        const std::vector<int32_t>& microcluster_membership,
        const std::vector<int32_t>& fine_to_microcluster,
        const std::vector<int32_t>& fine_component_labels,
        const RawClusteringGraph& fine_raw_graph,
        SceneCoreMode requested_mode,
        const SceneClassifierResult* classifier,
        const SceneConstructionOptions& options) {
    validate_scene_options(options);
    validate_fine_raw_graph(fine_raw_graph);
    const int32_t microclusters = static_cast<int32_t>(
        microcluster_membership.size());
    const int32_t scenes = canonical_scene_count(microcluster_membership);
    if (metadata.level <= 0 || fine_raw_graph.n_nodes <= 0
        || fine_to_microcluster.size()
            != static_cast<size_t>(fine_raw_graph.n_nodes)
        || fine_component_labels.size() != fine_to_microcluster.size()
        || fine_raw_graph.component_labels != fine_component_labels) {
        throw std::invalid_argument("Invalid fine scene-level input");
    }
    const std::vector<int32_t> micro_components = microcluster_components(
        fine_to_microcluster, fine_component_labels, microclusters);
    const std::vector<int32_t> per_scene_component = scene_components(
        microcluster_membership, micro_components, scenes);
    std::vector<int32_t> inherited(fine_to_microcluster.size());
    for (size_t point = 0; point < inherited.size(); ++point) {
        inherited[point] = microcluster_membership[static_cast<size_t>(
            fine_to_microcluster[point])];
    }

    FineSceneLevel result;
    result.metadata = metadata;
    result.n_partition_clusters = scenes;
    result.requested_core_mode = requested_mode;
    result.applied_core_mode = SceneCoreMode::Inherit;
    result.fine_partition_membership = inherited;
    if (requested_mode != SceneCoreMode::Inherit) {
        if (classifier == nullptr || !classifier->attempted
            || !classifier->passed || classifier->mode != requested_mode
            || classifier->eligible_scene.size()
                != static_cast<size_t>(scenes)
            || classifier->fine_probabilities.rows()
                != fine_raw_graph.n_nodes
            || classifier->fine_probabilities.cols() != scenes) {
            result.classifier_fallback = true;
            if (classifier != nullptr) {
                result.classifier = *classifier;
            } else {
                result.classifier.mode = requested_mode;
                result.classifier.fallback_reason =
                    "classifier_result_unavailable";
            }
        } else {
            result.classifier = *classifier;
            result.fine_partition_membership = probability_predictions(
                classifier->fine_probabilities, fine_component_labels,
                per_scene_component, classifier->eligible_scene, &inherited);
            result.applied_core_mode = requested_mode;
        }
    }
    std::vector<int64_t> partition_counts(static_cast<size_t>(scenes), 0);
    for (size_t point = 0;
            point < result.fine_partition_membership.size(); ++point) {
        const int32_t scene = result.fine_partition_membership[point];
        if (scene < 0 || scene >= scenes
            || per_scene_component[static_cast<size_t>(scene)]
                != fine_component_labels[point]) {
            throw std::runtime_error(
                "Fine scene assignment crossed a component");
        }
        ++partition_counts[static_cast<size_t>(scene)];
    }
    if (result.applied_core_mode != SceneCoreMode::Inherit
        && std::find(partition_counts.begin(), partition_counts.end(),
            int64_t{0}) != partition_counts.end()) {
        result.classifier_fallback = true;
        result.applied_core_mode = SceneCoreMode::Inherit;
        result.fine_partition_membership = inherited;
        partition_counts.assign(static_cast<size_t>(scenes), 0);
        for (const int32_t scene : inherited) {
            ++partition_counts[static_cast<size_t>(scene)];
        }
        result.classifier.fallback_reason =
            "classifier_refit_erased_scene";
    }

    std::vector<int32_t> partition_to_scene(static_cast<size_t>(scenes), -1);
    for (int32_t partition = 0; partition < scenes; ++partition) {
        if (partition_counts[static_cast<size_t>(partition)]
                < options.minimum_scene_core_members) continue;
        partition_to_scene[static_cast<size_t>(partition)] = result.n_scenes++;
        result.scene_source_clusters.push_back(partition);
        result.scene_fine_counts.push_back(
            partition_counts[static_cast<size_t>(partition)]);
        result.scene_component_labels.push_back(
            per_scene_component[static_cast<size_t>(partition)]);
    }
    result.fine_core_membership.resize(result.fine_partition_membership.size());
    for (size_t point = 0;
            point < result.fine_partition_membership.size(); ++point) {
        const int32_t scene = partition_to_scene[static_cast<size_t>(
            result.fine_partition_membership[point])];
        result.fine_core_membership[point] = scene;
        result.excluded_fine_points += scene < 0;
    }
    result.tail_scene = tail_scenes(result.scene_fine_counts);

    const auto raw_scores = raw_halo_scores(
        fine_raw_graph, result.fine_core_membership);
    result.memberships.reserve(result.fine_core_membership.size()
        * static_cast<size_t>(options.maximum_halo_scenes + 1));
    for (int32_t point = 0; point < fine_raw_graph.n_nodes; ++point) {
        const int32_t core = result.fine_core_membership[static_cast<size_t>(point)];
        if (core < 0) continue;
        const int32_t source_core = result.scene_source_clusters[
            static_cast<size_t>(core)];
        const bool classifier_point =
            result.applied_core_mode != SceneCoreMode::Inherit
            && result.classifier.eligible_scene[
                static_cast<size_t>(source_core)];
        std::vector<std::pair<int32_t, double>> candidates;
        double core_score = 0.0;
        if (classifier_point) {
            double normalization = 0.0;
            for (int32_t scene = 0; scene < scenes; ++scene) {
                if (result.classifier.eligible_scene[
                        static_cast<size_t>(scene)]
                    && partition_to_scene[static_cast<size_t>(scene)] >= 0
                    && per_scene_component[static_cast<size_t>(scene)]
                        == fine_component_labels[static_cast<size_t>(point)]) {
                    normalization +=
                        result.classifier.fine_probabilities(point, scene);
                }
            }
            for (int32_t scene = 0; scene < scenes; ++scene) {
                if (!result.classifier.eligible_scene[
                        static_cast<size_t>(scene)]
                    || partition_to_scene[static_cast<size_t>(scene)] < 0
                    || per_scene_component[static_cast<size_t>(scene)]
                        != fine_component_labels[static_cast<size_t>(point)]) {
                    continue;
                }
                const double score = normalization > 0.0
                    ? result.classifier.fine_probabilities(point, scene)
                        / normalization
                    : 0.0;
                const int32_t created_scene = partition_to_scene[
                    static_cast<size_t>(scene)];
                if (created_scene == core) core_score = score;
                else if (score > 0.0) {
                    candidates.emplace_back(created_scene, score);
                }
            }
        } else {
            for (const auto& item : raw_scores[static_cast<size_t>(point)]) {
                if (item.first == core) core_score = item.second;
                else if (item.first >= 0
                    && result.scene_component_labels[
                        static_cast<size_t>(item.first)]
                        == fine_component_labels[static_cast<size_t>(point)]) {
                    candidates.push_back(item);
                }
            }
        }
        result.memberships.push_back({point, core, core_score, 0, true});
        std::sort(candidates.begin(), candidates.end(),
            [](const auto& first, const auto& second) {
                return first.second == second.second
                    ? first.first < second.first
                    : first.second > second.second;
            });
        int32_t added = 0;
        for (const auto& candidate : candidates) {
            if (added >= options.maximum_halo_scenes) break;
            if (candidate.second < options.halo_minimum_score
                || candidate.second
                    < options.halo_relative_to_core * core_score) continue;
            result.memberships.push_back(
                {point, candidate.first, candidate.second, added + 1, false});
            ++added;
        }
    }
    return result;
}

SceneDag build_scene_dag(
        const std::vector<FineSceneLevel>& levels,
        const SceneConstructionOptions& options) {
    validate_scene_options(options);
    if (levels.empty() || levels.front().fine_core_membership.empty()) {
        throw std::invalid_argument("Scene DAG requires at least one level");
    }
    const int64_t fine_nodes = levels.front().fine_core_membership.size();
    for (size_t index = 0; index < levels.size(); ++index) {
        if (levels[index].metadata.level != static_cast<int32_t>(index + 1)
            || levels[index].fine_core_membership.size()
                != static_cast<size_t>(fine_nodes)) {
            throw std::invalid_argument(
                "Scene DAG levels must be ordered and aligned");
        }
    }
    SceneDag dag;
    dag.fine_nodes = fine_nodes;
    SceneDagNode root;
    root.id = 0;
    root.level = 0;
    root.scene = 0;
    root.fine_count = fine_nodes;
    dag.nodes.push_back(root);
    std::vector<std::vector<int32_t>> node_id(levels.size());
    for (size_t level = 0; level < levels.size(); ++level) {
        node_id[level].resize(static_cast<size_t>(levels[level].n_scenes));
        for (int32_t scene = 0; scene < levels[level].n_scenes; ++scene) {
            SceneDagNode node;
            node.id = static_cast<int32_t>(dag.nodes.size());
            node.level = levels[level].metadata.level;
            node.scene = scene;
            node.source_cluster = levels[level].scene_source_clusters[
                static_cast<size_t>(scene)];
            node.fine_count =
                levels[level].scene_fine_counts[static_cast<size_t>(scene)];
            node.component = levels[level].scene_component_labels[
                static_cast<size_t>(scene)];
            node.tail = levels[level].tail_scene[static_cast<size_t>(scene)];
            node.resolution = levels[level].metadata.resolution;
            node.c90 = levels[level].metadata.c90;
            node.plateau_index = levels[level].metadata.plateau_index;
            node.plateau_fallback = levels[level].metadata.plateau_fallback;
            node.core_mode = levels[level].applied_core_mode;
            node.classifier_fallback = levels[level].classifier_fallback;
            node_id[level][static_cast<size_t>(scene)] = node.id;
            dag.nodes.push_back(node);
        }
    }
    for (size_t level = 0; level < levels.size(); ++level) {
        if (level == 0) {
            for (int32_t scene = 0; scene < levels[level].n_scenes; ++scene) {
                const int32_t child = node_id[level][static_cast<size_t>(scene)];
                const int64_t overlap = dag.nodes[static_cast<size_t>(child)].fine_count;
                dag.edges.push_back({0, child, overlap, 1.0, true, false});
            }
            continue;
        }
        std::vector<std::unordered_map<int32_t, int64_t>> overlap(
            static_cast<size_t>(levels[level].n_scenes));
        for (int64_t point = 0; point < fine_nodes; ++point) {
            const int32_t child = levels[level].fine_core_membership[
                static_cast<size_t>(point)];
            const int32_t parent = levels[level - 1].fine_core_membership[
                static_cast<size_t>(point)];
            if (child < 0 || parent < 0) continue;
            ++overlap[static_cast<size_t>(child)][parent];
        }
        for (int32_t child_scene = 0;
                child_scene < levels[level].n_scenes; ++child_scene) {
            const auto& candidates = overlap[static_cast<size_t>(child_scene)];
            if (candidates.empty()) {
                throw std::runtime_error("Scene DAG child has no parent overlap");
            }
            int32_t major_parent = -1;
            int64_t major_overlap = -1;
            for (const auto& item : candidates) {
                if (item.second > major_overlap
                    || (item.second == major_overlap
                        && item.first < major_parent)) {
                    major_parent = item.first;
                    major_overlap = item.second;
                }
            }
            const int32_t child = node_id[level][
                static_cast<size_t>(child_scene)];
            const double denominator = static_cast<double>(
                dag.nodes[static_cast<size_t>(child)].fine_count);
            for (const auto& item : candidates) {
                const double fraction = item.second / denominator;
                const bool major = item.first == major_parent;
                if (!major
                    && fraction < options.portal_minimum_child_fraction) {
                    continue;
                }
                dag.edges.push_back({
                    node_id[level - 1][static_cast<size_t>(item.first)],
                    child, item.second, fraction, major, !major});
            }
        }
    }
    for (const SceneDagEdge& edge : dag.edges) {
        SceneDagNode& parent = dag.nodes[static_cast<size_t>(edge.parent)];
        SceneDagNode& child = dag.nodes[static_cast<size_t>(edge.child)];
        ++parent.child_count;
        ++child.parent_count;
        if (edge.major) child.major_parent = edge.parent;
    }
    for (SceneDagNode& node : dag.nodes) {
        node.merge = node.parent_count > 1;
        node.split = node.child_count > 1;
    }
    return dag;
}
