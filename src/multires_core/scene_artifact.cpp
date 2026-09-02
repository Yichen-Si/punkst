#include "multires_core/scene_artifact.hpp"

#include "multires_core/graph_artifact_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace punkst::multires {
namespace {

std::vector<int32_t> identity_mapping(int32_t size) {
    std::vector<int32_t> output(static_cast<size_t>(size));
    std::iota(output.begin(), output.end(), 0);
    return output;
}

int32_t canonical_count(const std::vector<int32_t>& membership) {
    if (membership.empty()
            || std::any_of(membership.begin(), membership.end(),
                [](int32_t value) { return value < 0; })) {
        throw std::runtime_error("Partition membership is invalid");
    }
    const int32_t count = 1 + *std::max_element(
        membership.begin(), membership.end());
    std::vector<uint8_t> used(static_cast<size_t>(count), 0);
    for (const int32_t value : membership) used[static_cast<size_t>(value)] = 1;
    if (std::find(used.begin(), used.end(), uint8_t{0}) != used.end()) {
        throw std::runtime_error("Partition labels are not canonical");
    }
    return count;
}

ThetaReadOptions theta_options_from_graph(const json& graph_manifest) {
    const json& input = graph_manifest.at("resolved_request").at("input");
    ThetaReadOptions options;
    options.identifier_column = input.value("identifier_column", 0);
    options.factor_column_start = input.at("factor_column_start").is_null()
        ? -1 : input.at("factor_column_start").get<int32_t>();
    options.factor_column_end = input.at("factor_column_end").is_null()
        ? -1 : input.at("factor_column_end").get<int32_t>();
    options.factor_weight_threshold = input.value(
        "factor_weight_threshold", 1e-5);
    return options;
}

json classifier_json(const SceneClassifierResult& classifier) {
    json recall = json::array();
    for (const double value : classifier.scene_recall) {
        recall.push_back(std::isfinite(value) ? json(value) : json(nullptr));
    }
    return {
        {"attempted", classifier.attempted},
        {"passed", classifier.passed},
        {"fallback_reason", classifier.fallback_reason.empty()
            ? json(nullptr) : json(classifier.fallback_reason)},
        {"crossfit_ari", classifier.attempted
            ? json(classifier.crossfit_ari) : json(nullptr)},
        {"scene_recall", recall},
        {"eligible_scene", classifier.eligible_scene},
        {"plugin_model_available", classifier.plugin_model_available},
        {"model_class_to_scene", classifier.model_class_to_scene}
    };
}

json level_summary(const FineSceneLevel& level) {
    int64_t halos = 0;
    for (const auto& row : level.memberships) halos += !row.core;
    int32_t tails = 0;
    for (const uint8_t tail : level.tail_scene) tails += tail != 0;
    return {
        {"level", level.metadata.level},
        {"resolution", level.metadata.resolution},
        {"selection_c90", level.metadata.c90},
        {"partition_clusters", level.n_partition_clusters},
        {"scenes", level.n_scenes},
        {"tail_scenes", tails},
        {"requested_core_mode", scene_core_mode_name(
            level.requested_core_mode)},
        {"applied_core_mode", scene_core_mode_name(level.applied_core_mode)},
        {"classifier_fallback", level.classifier_fallback},
        {"core_memberships", static_cast<int64_t>(
            level.fine_core_membership.size()) - level.excluded_fine_points},
        {"excluded_fine_points", level.excluded_fine_points},
        {"halo_memberships", halos},
        {"mean_memberships_per_point",
            static_cast<double>(level.memberships.size())
                / static_cast<double>(level.fine_core_membership.size())},
        {"classifier", classifier_json(level.classifier)}
    };
}

void require_stream(const std::ofstream& stream, const fs::path& path) {
    if (!stream) {
        throw std::runtime_error("Failed writing table: " + path.string());
    }
}

} // namespace

SceneArtifactResult run_multires_scenes(
        const SceneArtifactOptions& options) {
    const KnnGraphArtifactData graph = load_knn_graph_artifact(
        options.graph_manifest);
    const fs::path selection_path = resolve_artifact_manifest(
        options.selection_manifest);
    const json selection = load_verified_artifact(
        selection_path, "punkst.multires.selection");
    if (selection.at("source").at("graph_fingerprint").get<std::string>()
            != graph.fingerprint) {
        throw std::runtime_error(
            "Selection artifact was built from a different graph");
    }
    const std::string population = selection.at(
        "resolved_scan_population").get<std::string>();
    if (population != "points" && population != "microclusters") {
        throw std::runtime_error("Selection scan population is invalid");
    }
    if (options.core_mode == SceneCoreMode::ClassifierLrvb) {
        throw std::invalid_argument(
            "LRVB scene assignment requires explicit posterior input and is "
            "not supported by this artifact command");
    }
    if (options.core_mode == SceneCoreMode::ClassifierPlugin
            && population != "microclusters") {
        throw std::invalid_argument(
            "Classifier scene assignment requires a microcluster scan");
    }

    SceneArtifactResult result;
    result.graph_fingerprint = graph.fingerprint;
    result.selection_fingerprint =
        selection.at("fingerprint").get<std::string>();
    result.identifiers = graph.identifiers;
    const fs::path selection_root = selection_path.parent_path();
    const json& selected_levels = selection.at("levels");
    if (!selected_levels.is_array() || selected_levels.empty()) {
        throw std::runtime_error("Selection artifact has no selected levels");
    }

    std::optional<ThetaTable> theta;
    if (options.core_mode == SceneCoreMode::ClassifierPlugin) {
        if (!graph.coarse.has_value()) {
            throw std::runtime_error(
                "Classifier assignment requires graph coarsening");
        }
        for (const json& level : selected_levels) {
            if (level.value("full_data_refined", false)) {
                throw std::invalid_argument(
                    "Classifier assignment cannot replace a refined "
                    "full-data partition");
            }
        }
        const fs::path theta_path = graph.manifest.at("source")
            .at("theta_path").get<std::string>();
        if (sha256_file(theta_path) != graph.manifest.at("source")
                .at("theta_sha256").get<std::string>()) {
            throw std::runtime_error("Source theta checksum does not match");
        }
        theta = read_theta_table(theta_path,
            theta_options_from_graph(graph.manifest));
        if (theta->identifiers != graph.identifiers) {
            throw std::runtime_error(
                "Reloaded theta identifiers do not match the graph artifact");
        }
    }

    const std::vector<int32_t> identity = identity_mapping(
        graph.fine.graph.n_nodes);
    for (size_t index = 0; index < selected_levels.size(); ++index) {
        const json& selected = selected_levels[index];
        SceneLevelMetadata metadata;
        metadata.level = selected.at("level").get<int32_t>();
        if (metadata.level != static_cast<int32_t>(index + 1)) {
            throw std::runtime_error("Selected levels are not consecutive");
        }
        metadata.resolution = selected.at("resolution").get<double>();
        metadata.c90 = selected.at("selection_c90").get<int32_t>();
        metadata.plateau_index = selected.at("plateau").is_null()
            ? -1 : selected.at("plateau").get<int32_t>();
        metadata.plateau_fallback = selected.at("fallback").get<bool>();
        const std::vector<int32_t> full_membership = read_int32_array(
            selection_root, selected.at("full_membership"));
        canonical_count(full_membership);

        if (options.core_mode == SceneCoreMode::ClassifierPlugin) {
            const std::vector<int32_t> scan_membership = read_int32_array(
                selection_root, selected.at("scan_membership"));
            canonical_count(scan_membership);
            SceneClassifierResult classifier = fit_plugin_scene_classifier(
                theta->values, graph.representative_rows, scan_membership,
                graph.coarse->fine_point_counts,
                graph.fine.graph.component_labels, options.classifier);
            result.levels.push_back(construct_fine_scene_level(
                metadata, scan_membership, graph.fine_to_microcluster,
                graph.fine.graph.component_labels, graph.fine.graph,
                SceneCoreMode::ClassifierPlugin, &classifier,
                options.scenes));
        } else {
            result.levels.push_back(construct_fine_scene_level(
                metadata, full_membership, identity,
                graph.fine.graph.component_labels, graph.fine.graph,
                SceneCoreMode::Inherit, nullptr, options.scenes));
        }
    }
    result.dag = build_scene_dag(result.levels, options.scenes);
    return result;
}

void write_scene_artifact(const fs::path& output,
        const json& resolved_request,
        const SceneArtifactOptions& options,
        const SceneArtifactResult& result) {
    publish_directory_atomic(fs::absolute(output), [&](const fs::path& root) {
        fs::create_directories(root / "levels");
        fs::create_directories(root / "internal");
        const fs::path membership_path = root / "scene_memberships.tsv";
        std::ofstream memberships(membership_path);
        memberships << std::setprecision(17)
            << "level\tid\tscene\tscore\trank\tcore\n";
        json level_manifest = json::array();
        for (const FineSceneLevel& level : result.levels) {
            std::vector<double> core_scores(
                level.fine_core_membership.size(), 0.0);
            std::vector<int32_t> halo_counts(
                level.fine_core_membership.size(), 0);
            for (const SceneHaloMembership& row : level.memberships) {
                memberships << level.metadata.level << '\t'
                    << result.identifiers[static_cast<size_t>(row.fine_node)]
                    << '\t' << row.scene << '\t' << row.score << '\t'
                    << row.rank << '\t' << static_cast<int32_t>(row.core)
                    << '\n';
                if (row.core) {
                    core_scores[static_cast<size_t>(row.fine_node)] = row.score;
                } else {
                    ++halo_counts[static_cast<size_t>(row.fine_node)];
                }
            }
            const std::string stem = "level" + std::to_string(
                level.metadata.level);
            const fs::path assignment_path = root / (
                "levels/" + stem + "_assignment.tsv");
            std::ofstream assignment(assignment_path);
            assignment << std::setprecision(17)
                << "id\tpartition_cluster\tcore_scene\tcore_score"
                   "\thalo_count\n";
            for (size_t point = 0; point < result.identifiers.size(); ++point) {
                assignment << result.identifiers[point] << '\t'
                    << level.fine_partition_membership[point] << '\t';
                if (level.fine_core_membership[point] < 0) {
                    assignment << ".\t.";
                } else {
                    assignment << level.fine_core_membership[point] << '\t'
                        << core_scores[point];
                }
                assignment << '\t' << halo_counts[point] << '\n';
            }
            assignment.close();
            require_stream(assignment, assignment_path);

            const fs::path scenes_path = root / (
                "levels/" + stem + "_scenes.tsv");
            std::ofstream scenes(scenes_path);
            scenes << "scene\tsource_cluster\tfine_count\tcomponent\ttail\n";
            for (int32_t scene = 0; scene < level.n_scenes; ++scene) {
                scenes << scene << '\t'
                    << level.scene_source_clusters[static_cast<size_t>(scene)]
                    << '\t' << level.scene_fine_counts[static_cast<size_t>(scene)]
                    << '\t' << level.scene_component_labels[
                        static_cast<size_t>(scene)]
                    << '\t' << static_cast<int32_t>(level.tail_scene[
                        static_cast<size_t>(scene)]) << '\n';
            }
            scenes.close();
            require_stream(scenes, scenes_path);

            json summary = level_summary(level);
            summary["assignment_table"] = fs::relative(
                assignment_path, root).string();
            summary["assignment_sha256"] = sha256_file(assignment_path);
            summary["scenes_table"] = fs::relative(
                scenes_path, root).string();
            summary["scenes_sha256"] = sha256_file(scenes_path);
            if (level.classifier.plugin_model_available) {
                const fs::path model_path = root / (
                    "internal/" + stem + "_classifier.tsv");
                level.classifier.plugin_model.write(model_path.string());
                summary["classifier"]["model"] = fs::relative(
                    model_path, root).string();
                summary["classifier"]["model_sha256"] =
                    sha256_file(model_path);
            } else {
                summary["classifier"]["model"] = nullptr;
                summary["classifier"]["model_sha256"] = nullptr;
            }
            level_manifest.push_back(std::move(summary));
        }
        memberships.close();
        require_stream(memberships, membership_path);

        const fs::path node_path = root / "scene_nodes.tsv";
        std::ofstream nodes(node_path);
        nodes << std::setprecision(17)
            << "node\tlevel\tscene\tsource_cluster\tfine_count\tcomponent"
               "\ttail\tresolution\tselection_c90\tplateau\tfallback"
               "\tcore_mode\tclassifier_fallback\tmajor_parent"
               "\tparent_count\tchild_count\tmerge\tsplit\n";
        for (const SceneDagNode& node : result.dag.nodes) {
            nodes << node.id << '\t' << node.level << '\t' << node.scene << '\t';
            if (node.source_cluster < 0) nodes << '.';
            else nodes << node.source_cluster;
            nodes << '\t' << node.fine_count << '\t';
            if (node.component < 0) nodes << '.';
            else nodes << node.component;
            nodes << '\t' << static_cast<int32_t>(node.tail) << '\t'
                << node.resolution << '\t' << node.c90 << '\t';
            if (node.plateau_index < 0) nodes << '.';
            else nodes << node.plateau_index;
            nodes << '\t' << static_cast<int32_t>(node.plateau_fallback)
                << '\t' << scene_core_mode_name(node.core_mode) << '\t'
                << static_cast<int32_t>(node.classifier_fallback) << '\t';
            if (node.major_parent < 0) nodes << '.';
            else nodes << node.major_parent;
            nodes << '\t' << node.parent_count << '\t' << node.child_count
                << '\t' << static_cast<int32_t>(node.merge) << '\t'
                << static_cast<int32_t>(node.split) << '\n';
        }
        nodes.close();
        require_stream(nodes, node_path);

        const fs::path edge_path = root / "scene_edges.tsv";
        std::ofstream edges(edge_path);
        edges << std::setprecision(17)
            << "parent\tchild\toverlap\tchild_fraction\tmajor\tportal\n";
        for (const SceneDagEdge& edge : result.dag.edges) {
            edges << edge.parent << '\t' << edge.child << '\t'
                << edge.overlap << '\t' << edge.child_fraction << '\t'
                << static_cast<int32_t>(edge.major) << '\t'
                << static_cast<int32_t>(edge.portal) << '\n';
        }
        edges.close();
        require_stream(edges, edge_path);

        int32_t portals = 0;
        int32_t merges = 0;
        int32_t splits = 0;
        for (const SceneDagEdge& edge : result.dag.edges) portals += edge.portal;
        for (const SceneDagNode& node : result.dag.nodes) {
            merges += node.merge;
            splits += node.split;
        }
        json manifest = {
            {"artifact_type", "punkst.multires.scenes"},
            {"schema_version", 1},
            {"resolved_request", resolved_request},
            {"source", {
                {"graph_manifest", resolve_artifact_manifest(
                    options.graph_manifest).string()},
                {"graph_fingerprint", result.graph_fingerprint},
                {"selection_manifest", resolve_artifact_manifest(
                    options.selection_manifest).string()},
                {"selection_fingerprint", result.selection_fingerprint}
            }},
            {"fine_points", result.dag.fine_nodes},
            {"levels", level_manifest},
            {"dag", {
                {"nodes", result.dag.nodes.size()},
                {"edges", result.dag.edges.size()},
                {"portal_edges", portals},
                {"merge_nodes", merges},
                {"split_nodes", splits}
            }},
            {"tables", {
                {"memberships", "scene_memberships.tsv"},
                {"memberships_sha256", sha256_file(membership_path)},
                {"nodes", "scene_nodes.tsv"},
                {"nodes_sha256", sha256_file(node_path)},
                {"edges", "scene_edges.tsv"},
                {"edges_sha256", sha256_file(edge_path)}
            }}
        };
        manifest["fingerprint"] = artifact_fingerprint(manifest, root, {});
        write_json(root / "manifest.json", manifest, 2);
    });
}

} // namespace punkst::multires
