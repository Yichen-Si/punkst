#include "multires_core/scene_projection_artifact.hpp"

#include "clustering_core/leiden.hpp"
#include "multires_core/diffusion_graph.hpp"
#include "multires_core/graph_artifact_io.hpp"
#include "multires_core/scene_artifact_io.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace punkst::multires {
namespace {

using Clock = std::chrono::steady_clock;

struct CluePartition {
    std::string source;
    std::vector<int32_t> core_rows;
    Eigen::VectorXi assignments;
    int32_t groups = 0;
    int32_t excluded_rows = 0;
    bool fallback_attempted = false;
    int32_t fallback_seed = 0;
    int32_t fallback_clusters = 0;
    double fallback_quality = 0.0;
    int32_t fallback_iterations = 0;
    bool fallback_converged = false;
    double fallback_seconds = 0.0;
};

struct ViewFile {
    bool available = false;
    int32_t dimensions = 0;
    std::string coordinates;
    std::string coordinates_sha256;
    std::string axes;
    std::string axes_sha256;
    std::string omitted_reason;
    int32_t fit_rows = 0;
    int32_t represented_groups = 0;
    double retained_subspace_variance_fraction = 0.0;
    double quartimax_objective = 0.0;
};

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

int32_t scene_seed(uint64_t seed, int32_t node) {
    const uint64_t mixed = splitmix64(seed
        ^ (static_cast<uint64_t>(static_cast<uint32_t>(node)) << 1));
    return static_cast<int32_t>(mixed % 2147483646ULL + 1ULL);
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

std::vector<std::vector<SceneHaloMembership>> group_memberships(
        const FineSceneLevel& level) {
    std::vector<std::vector<SceneHaloMembership>> grouped(
        static_cast<size_t>(level.n_scenes));
    for (const SceneHaloMembership& row : level.memberships) {
        grouped[static_cast<size_t>(row.scene)].push_back(row);
    }
    for (auto& scene : grouped) {
        std::sort(scene.begin(), scene.end(),
            [](const SceneHaloMembership& left,
               const SceneHaloMembership& right) {
                return left.fine_node < right.fine_node;
            });
        for (size_t index = 1; index < scene.size(); ++index) {
            if (scene[index - 1].fine_node == scene[index].fine_node) {
                throw std::runtime_error(
                    "Scene contains a duplicate point membership");
            }
        }
    }
    return grouped;
}

RowMajorMatrixXd scene_values(const ThetaTable& theta,
        const std::vector<SceneHaloMembership>& members) {
    RowMajorMatrixXd output(members.size(), theta.values.cols());
    for (size_t row = 0; row < members.size(); ++row) {
        output.row(static_cast<Eigen::Index>(row)) = theta.values.row(
            members[row].fine_node);
    }
    return output;
}

std::vector<int32_t> core_local_rows(
        const std::vector<SceneHaloMembership>& members) {
    std::vector<int32_t> output;
    for (size_t row = 0; row < members.size(); ++row) {
        if (members[row].core) output.push_back(static_cast<int32_t>(row));
    }
    return output;
}

CluePartition retained_partition(
        const std::vector<int32_t>& labels,
        const std::vector<int32_t>& core_rows,
        int32_t minimum_members, const std::string& source) {
    if (labels.size() != core_rows.size()) {
        throw std::invalid_argument("Clue labels and core rows do not align");
    }
    std::map<int32_t, int32_t> counts;
    for (const int32_t label : labels) {
        if (label >= 0) ++counts[label];
    }
    std::map<int32_t, int32_t> canonical;
    for (const auto& item : counts) {
        if (item.second >= minimum_members) {
            canonical.emplace(item.first,
                static_cast<int32_t>(canonical.size()));
        }
    }
    CluePartition output;
    output.source = source;
    output.groups = static_cast<int32_t>(canonical.size());
    std::vector<int32_t> assignments;
    for (size_t index = 0; index < labels.size(); ++index) {
        const auto found = canonical.find(labels[index]);
        if (found == canonical.end()) {
            ++output.excluded_rows;
            continue;
        }
        output.core_rows.push_back(core_rows[index]);
        assignments.push_back(found->second);
    }
    output.assignments.resize(assignments.size());
    for (size_t index = 0; index < assignments.size(); ++index) {
        output.assignments(static_cast<Eigen::Index>(index)) = assignments[index];
    }
    return output;
}

CluePartition child_clues(
        const std::vector<SceneHaloMembership>& members,
        const std::vector<int32_t>& core_rows,
        const FineSceneLevel* children, int32_t minimum_members) {
    if (children == nullptr) return {};
    std::vector<int32_t> labels;
    labels.reserve(core_rows.size());
    for (const int32_t local : core_rows) {
        labels.push_back(children->fine_core_membership[
            static_cast<size_t>(members[static_cast<size_t>(local)].fine_node)]);
    }
    return retained_partition(
        labels, core_rows, minimum_members, "child_scenes");
}

CluePartition fallback_clues(
        const RowMajorMatrixXd& values,
        const std::vector<int32_t>& core_rows,
        const SceneProjectionArtifactOptions& options,
        int32_t node) {
    if (core_rows.size() < static_cast<size_t>(
            2 * options.minimum_clue_members)) {
        CluePartition output;
        output.source = "fallback_unavailable";
        output.excluded_rows = static_cast<int32_t>(core_rows.size());
        return output;
    }
    const auto begin = Clock::now();
    const RowMajorMatrixXd core = [&]() {
        RowMajorMatrixXd selected(core_rows.size(), values.cols());
        for (size_t row = 0; row < core_rows.size(); ++row) {
            selected.row(static_cast<Eigen::Index>(row)) =
                values.row(core_rows[row]);
        }
        return selected;
    }();
    HellingerKnnGraphOptions graph_options;
    graph_options.knn.n_neighbors = std::min<int32_t>(
        options.local_neighbors, static_cast<int32_t>(core.rows()) - 1);
    graph_options.knn.n_threads = options.projection.n_threads;
    graph_options.knn.ann_seed = scene_seed(options.seed, node);
    graph_options.build_bridge_candidates = false;
    graph_options.retain_diffusion_geometry = false;
    const HellingerKnnGraph graph = build_hellinger_knn_graph(
        core, graph_options);
    LeidenOptions leiden_options;
    leiden_options.resolution = options.fallback_resolution;
    leiden_options.max_iterations = -1;
    leiden_options.seed = scene_seed(options.seed, node);
    const LeidenResult leiden = leiden_cluster(
        graph.n_nodes, graph.edges, graph.raw_affinities, leiden_options);
    std::vector<int32_t> labels(static_cast<size_t>(leiden.membership.size()));
    for (Eigen::Index row = 0; row < leiden.membership.size(); ++row) {
        labels[static_cast<size_t>(row)] = leiden.membership(row);
    }
    CluePartition output = retained_partition(labels, core_rows,
        options.minimum_clue_members, "fallback_leiden");
    output.fallback_attempted = true;
    output.fallback_seed = leiden_options.seed;
    output.fallback_clusters = leiden.n_communities;
    output.fallback_quality = leiden.quality;
    output.fallback_iterations = leiden.iterations;
    output.fallback_converged = leiden.converged;
    output.fallback_seconds = std::chrono::duration<double>(
        Clock::now() - begin).count();
    return output;
}

void require_stream(const std::ofstream& output, const fs::path& path) {
    if (!output) {
        throw std::runtime_error("Failed writing table: " + path.string());
    }
}

ViewFile write_view(const fs::path& root, const std::string& view,
        int32_t level, int32_t scene,
        const std::vector<SceneHaloMembership>& members,
        const std::vector<std::string>& identifiers,
        const std::vector<std::string>& factor_names,
        const SceneProjectionView& fitted) {
    const std::string stem = "level" + std::to_string(level)
        + "_scene" + std::to_string(scene);
    const fs::path directory = root / "views" / view;
    fs::create_directories(directory);
    const fs::path coordinates = directory / (stem + ".coordinates.tsv");
    std::ofstream coordinate_output(coordinates);
    coordinate_output << "id\tcore\tmembership_score\tmembership_rank";
    for (Eigen::Index axis = 0; axis < fitted.coordinates.cols(); ++axis) {
        coordinate_output << "\taxis_" << axis;
    }
    coordinate_output << '\n' << std::setprecision(9);
    for (size_t row = 0; row < members.size(); ++row) {
        const SceneHaloMembership& member = members[row];
        coordinate_output << identifiers[static_cast<size_t>(member.fine_node)]
            << '\t' << static_cast<int32_t>(member.core)
            << '\t' << member.score << '\t' << member.rank;
        for (Eigen::Index axis = 0; axis < fitted.coordinates.cols(); ++axis) {
            coordinate_output << '\t'
                << fitted.coordinates(static_cast<Eigen::Index>(row), axis);
        }
        coordinate_output << '\n';
    }
    coordinate_output.close();
    require_stream(coordinate_output, coordinates);

    const fs::path axes = directory / (stem + ".axes.tsv");
    std::ofstream axis_output(axes);
    axis_output << "axis\tfactor\tcoefficient\tpositive_weight"
                   "\tnegative_weight\tscore\n"
                << std::setprecision(17);
    for (Eigen::Index axis = 0; axis < fitted.topic_contrasts.cols(); ++axis) {
        const Eigen::VectorXd values = fitted.topic_contrasts.col(axis);
        const double positive = values.cwiseMax(0.0).sum();
        const double negative = (-values).cwiseMax(0.0).sum();
        for (Eigen::Index factor = 0; factor < values.size(); ++factor) {
            axis_output << axis << '\t'
                << factor_names[static_cast<size_t>(factor)] << '\t'
                << values(factor) << '\t'
                << (positive > 0.0 ? std::max(0.0, values(factor)) / positive : 0.0)
                << '\t'
                << (negative > 0.0 ? std::max(0.0, -values(factor)) / negative : 0.0)
                << '\t' << fitted.axis_scores(axis) << '\n';
        }
    }
    axis_output.close();
    require_stream(axis_output, axes);

    ViewFile output;
    output.available = true;
    output.dimensions = static_cast<int32_t>(fitted.coordinates.cols());
    output.coordinates = fs::relative(coordinates, root).string();
    output.coordinates_sha256 = sha256_file(coordinates);
    output.axes = fs::relative(axes, root).string();
    output.axes_sha256 = sha256_file(axes);
    output.fit_rows = fitted.fit_rows;
    output.represented_groups = fitted.represented_groups;
    output.retained_subspace_variance_fraction =
        fitted.retained_subspace_variance_fraction;
    output.quartimax_objective = fitted.quartimax_objective;
    return output;
}

json view_json(const ViewFile& view) {
    return {
        {"available", view.available},
        {"dimensions", view.dimensions},
        {"coordinates", view.available ? json(view.coordinates) : json(nullptr)},
        {"coordinates_sha256", view.available
            ? json(view.coordinates_sha256) : json(nullptr)},
        {"axes", view.available ? json(view.axes) : json(nullptr)},
        {"axes_sha256", view.available
            ? json(view.axes_sha256) : json(nullptr)},
        {"fit_rows", view.available ? json(view.fit_rows) : json(nullptr)},
        {"represented_groups", view.available
            ? json(view.represented_groups) : json(nullptr)},
        {"retained_subspace_variance_fraction", view.available
            ? json(view.retained_subspace_variance_fraction) : json(nullptr)},
        {"quartimax_objective", view.available
            ? json(view.quartimax_objective) : json(nullptr)},
        {"omitted_reason", view.omitted_reason.empty()
            ? json(nullptr) : json(view.omitted_reason)}
    };
}

} // namespace

SceneProjectionArtifactSummary write_scene_projection_artifact(
        const fs::path& output, const json& resolved_request,
        const SceneProjectionArtifactOptions& options) {
    validate_scene_projection_options(options.projection);
    if (options.local_neighbors <= 0 || options.minimum_clue_members <= 0
            || !(options.fallback_resolution > 0.0)
            || !std::isfinite(options.fallback_resolution)) {
        throw std::invalid_argument("Invalid scene projection artifact options");
    }
    const KnnGraphArtifactData graph = load_knn_graph_artifact(
        options.graph_manifest);
    const LoadedSceneArtifact scenes = load_scene_artifact(
        options.scenes_manifest);
    if (scenes.graph_fingerprint != graph.fingerprint
            || scenes.fine_points != graph.fine.graph.n_nodes) {
        throw std::runtime_error(
            "Scene and graph artifacts do not describe the same points");
    }
    const fs::path theta_path = graph.manifest.at("source")
        .at("theta_path").get<std::string>();
    if (sha256_file(theta_path) != graph.manifest.at("source")
            .at("theta_sha256").get<std::string>()) {
        throw std::runtime_error("Source theta checksum does not match");
    }
    const ThetaTable theta = read_theta_table(
        theta_path, theta_options_from_graph(graph.manifest));
    if (theta.identifiers != graph.identifiers) {
        throw std::runtime_error(
            "Reloaded theta identifiers do not match the graph artifact");
    }

    SceneProjectionArtifactSummary summary;
    publish_directory_atomic(fs::absolute(output), [&](const fs::path& root) {
        fs::create_directories(root / "views");
        const fs::path index_path = root / "scene_projections.tsv";
        std::ofstream index(index_path);
        index << "node\tlevel\tscene\tmembers\tcore_members\tclue_source"
                 "\tclue_groups\tclue_fit_rows\texcluded_clue_rows"
                 "\tsupervised_dimensions\tquartimax_pca_dimensions\n";
        json records = json::array();
        for (size_t level_index = 0;
                level_index < scenes.levels.size(); ++level_index) {
            const FineSceneLevel& level = scenes.levels[level_index];
            const FineSceneLevel* children = level_index + 1 < scenes.levels.size()
                ? &scenes.levels[level_index + 1] : nullptr;
            const auto grouped = group_memberships(level);
            for (int32_t scene = 0; scene < level.n_scenes; ++scene) {
                ++summary.scenes;
                const int32_t node = scene_dag_node_id(
                    scenes, level.metadata.level, scene);
                const auto& members = grouped[static_cast<size_t>(scene)];
                const std::vector<int32_t> core_rows = core_local_rows(members);
                if (members.empty() || core_rows.empty()) {
                    throw std::runtime_error("Scene has no members or core");
                }
                const RowMajorMatrixXd values = scene_values(theta, members);
                CluePartition clues = child_clues(members, core_rows, children,
                    options.minimum_clue_members);
                if (clues.groups < 2) {
                    clues = fallback_clues(values, core_rows, options, node);
                    if (clues.fallback_attempted) {
                        ++summary.fallback_leiden_runs;
                    }
                }

                ViewFile supervised;
                ViewFile quartimax_pca;
                std::optional<SceneComposition> composition;
                try {
                    composition = prepare_scene_composition(values,
                        theta.factor_names, core_rows, options.projection);
                } catch (const std::exception& error) {
                    supervised.omitted_reason = error.what();
                    quartimax_pca.omitted_reason = error.what();
                }
                if (composition.has_value()) {
                    if (clues.groups >= 2) {
                        try {
                            const SceneProjectionView fitted =
                                fit_scene_mean_separation(*composition,
                                    clues.core_rows, clues.assignments,
                                    clues.groups, options.projection);
                            supervised = write_view(root, "supervised",
                                level.metadata.level, scene, members,
                                theta.identifiers, theta.factor_names, fitted);
                            ++summary.supervised_views;
                        } catch (const std::exception& error) {
                            supervised.omitted_reason = error.what();
                        }
                    } else {
                        supervised.omitted_reason =
                            "fewer than two clue clusters meet the minimum size";
                    }
                    try {
                        const SceneProjectionView fitted = fit_scene_quartimax_pca(
                            *composition, core_rows, options.projection);
                        quartimax_pca = write_view(root, "quartimax_pca",
                            level.metadata.level, scene, members,
                            theta.identifiers, theta.factor_names, fitted);
                        ++summary.quartimax_pca_views;
                    } catch (const std::exception& error) {
                        quartimax_pca.omitted_reason = error.what();
                    }
                }
                index << node << '\t' << level.metadata.level << '\t' << scene
                    << '\t' << members.size() << '\t' << core_rows.size()
                    << '\t' << clues.source << '\t' << clues.groups
                    << '\t' << clues.core_rows.size() << '\t'
                    << clues.excluded_rows << '\t' << supervised.dimensions
                    << '\t' << quartimax_pca.dimensions << '\n';
                records.push_back({
                    {"node", node}, {"level", level.metadata.level},
                    {"scene", scene}, {"members", members.size()},
                    {"core_members", core_rows.size()},
                    {"factor_selection", composition.has_value() ? json{
                        {"retained_factors", composition->retained_factors},
                        {"retained_factor_names",
                            composition->retained_factor_names},
                        {"retained_core_mass_proportion",
                            composition->retained_core_mass_proportion},
                        {"minimum_factors_restored",
                            composition->minimum_factors_restored}
                    } : json(nullptr)},
                    {"clues", {
                        {"source", clues.source}, {"groups", clues.groups},
                        {"fit_rows", clues.core_rows.size()},
                        {"excluded_rows", clues.excluded_rows},
                        {"fallback_attempted", clues.fallback_attempted},
                        {"fallback_resolution", clues.fallback_attempted
                            ? json(options.fallback_resolution) : json(nullptr)},
                        {"fallback_seed", clues.fallback_attempted
                            ? json(clues.fallback_seed) : json(nullptr)},
                        {"fallback_clusters", clues.fallback_attempted
                            ? json(clues.fallback_clusters) : json(nullptr)},
                        {"fallback_quality", clues.fallback_attempted
                            ? json(clues.fallback_quality) : json(nullptr)},
                        {"fallback_iterations", clues.fallback_attempted
                            ? json(clues.fallback_iterations) : json(nullptr)},
                        {"fallback_converged", clues.fallback_attempted
                            ? json(clues.fallback_converged) : json(nullptr)},
                        {"fallback_seconds", clues.fallback_attempted
                            ? json(clues.fallback_seconds) : json(nullptr)}
                    }},
                    {"supervised", view_json(supervised)},
                    {"quartimax_pca", view_json(quartimax_pca)}
                });
            }
        }
        index.close();
        require_stream(index, index_path);
        const std::string index_sha256 = sha256_file(index_path);
        json content_identity = {
            {"artifact_type",
                "punkst.multires.scene_projections.content_identity"},
            {"schema_version", 1},
            {"graph_fingerprint", graph.fingerprint},
            {"scenes_fingerprint", scenes.fingerprint},
            {"index_sha256", index_sha256},
            {"scenes", json::array()}
        };
        for (const json& record : records) {
            json encoded = {
                {"node", record.at("node")},
                {"level", record.at("level")},
                {"scene", record.at("scene")}
            };
            for (const char* name : {"supervised", "quartimax_pca"}) {
                const json& view = record.at(name);
                encoded[name] = {
                    {"available", view.at("available")},
                    {"dimensions", view.at("dimensions")},
                    {"coordinates_sha256", view.at("coordinates_sha256")},
                    {"axes_sha256", view.at("axes_sha256")}
                };
            }
            content_identity["scenes"].push_back(std::move(encoded));
        }
        json manifest = {
            {"artifact_type", "punkst.multires.scene_projections"},
            {"schema_version", 1},
            {"resolved_request", resolved_request},
            {"source", {
                {"graph_manifest", resolve_artifact_manifest(
                    options.graph_manifest).string()},
                {"graph_fingerprint", graph.fingerprint},
                {"scenes_manifest", resolve_artifact_manifest(
                    options.scenes_manifest).string()},
                {"scenes_fingerprint", scenes.fingerprint}
            }},
            {"summary", {
                {"scenes", summary.scenes},
                {"supervised_views", summary.supervised_views},
                {"quartimax_pca_views", summary.quartimax_pca_views},
                {"fallback_leiden_runs", summary.fallback_leiden_runs}
            }},
            {"index", "scene_projections.tsv"},
            {"index_sha256", index_sha256},
            {"content_identity_fingerprint", artifact_fingerprint(
                content_identity, root, {})},
            {"scenes", std::move(records)}
        };
        summary.fingerprint = artifact_fingerprint(manifest, root, {});
        manifest["fingerprint"] = summary.fingerprint;
        write_json(root / "manifest.json", manifest, 2);
    });
    return summary;
}

} // namespace punkst::multires
