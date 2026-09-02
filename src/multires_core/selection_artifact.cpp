#include "multires_core/selection_artifact.hpp"
#include "multires_core/graph_artifact_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace punkst::multires {
namespace {

json nullable_number(double value) {
    return std::isfinite(value) ? json(value) : json(nullptr);
}

void write_diagnostics(const fs::path& root,
        const ResolutionSelectionResult& result) {
    fs::create_directories(root / "diagnostics");
    {
        std::ofstream output(root / "diagnostics/scout.tsv");
        output << std::setprecision(17)
            << "scout\tresolution\tc90\tcommunities\tquality\titerations"
               "\tconverged\n";
        for (size_t index = 0; index < result.scout_evaluations.size(); ++index) {
            const auto& value = result.scout_evaluations[index];
            output << index << '\t' << value.resolution << '\t' << value.c90
                << '\t' << value.n_communities << '\t' << value.quality
                << '\t' << value.iterations << '\t' << value.converged << '\n';
        }
    }
    {
        std::ofstream output(root / "diagnostics/evaluations.tsv");
        output << std::setprecision(17)
            << "evaluation\tresolution\tc90\tcommunities\tmean_seed_ari"
               "\tminimum_seed_ari\tpersistence_from_previous"
               "\tmedoid_restart\n";
        for (size_t index = 0; index < result.evaluations.size(); ++index) {
            const auto& value = result.evaluations[index];
            output << index << '\t' << value.resolution << '\t' << value.c90
                << '\t' << value.n_communities << '\t'
                << value.mean_pairwise_ari << '\t'
                << value.minimum_pairwise_ari << '\t';
            if (std::isfinite(value.persistence_from_previous)) {
                output << value.persistence_from_previous;
            } else {
                output << '.';
            }
            output << '\t' << value.medoid_restart << '\n';
        }
    }
    {
        std::ofstream output(root / "diagnostics/restarts.tsv");
        output << std::setprecision(17)
            << "evaluation\trestart\tseed\tcommunities\tquality\titerations"
               "\tconverged\n";
        for (size_t evaluation = 0;
                evaluation < result.evaluations.size(); ++evaluation) {
            const auto& value = result.evaluations[evaluation];
            for (size_t restart = 0;
                    restart < value.restart_seeds.size(); ++restart) {
                output << evaluation << '\t' << restart << '\t'
                    << value.restart_seeds[restart] << '\t'
                    << value.restart_n_communities[restart] << '\t'
                    << value.restart_quality[restart] << '\t'
                    << value.restart_iterations[restart] << '\t'
                    << static_cast<int32_t>(value.restart_converged[restart])
                    << '\n';
            }
        }
    }
    {
        std::ofstream output(root / "diagnostics/pairwise_ari.tsv");
        output << std::setprecision(17)
            << "evaluation\tfirst_restart\tsecond_restart\tari\n";
        for (size_t evaluation = 0;
                evaluation < result.evaluations.size(); ++evaluation) {
            const auto& value = result.evaluations[evaluation];
            size_t pair = 0;
            for (size_t first = 0; first < value.restart_seeds.size(); ++first) {
                for (size_t second = first + 1;
                        second < value.restart_seeds.size(); ++second) {
                    output << evaluation << '\t' << first << '\t' << second
                        << '\t' << value.pairwise_ari[pair++] << '\n';
                }
            }
        }
    }
    {
        std::ofstream output(root / "diagnostics/plateaus.tsv");
        output << std::setprecision(17)
            << "plateau\tfirst_evaluation\tlast_evaluation"
               "\trepresentative_evaluation\tfirst_resolution"
               "\tlast_resolution\trepresentative_resolution\tc90"
               "\tcommunities\tminimum_seed_stability"
               "\tminimum_adjacent_persistence\n";
        for (size_t index = 0; index < result.plateaus.size(); ++index) {
            const auto& value = result.plateaus[index];
            output << index << '\t' << value.first_evaluation << '\t'
                << value.last_evaluation << '\t'
                << value.representative_evaluation << '\t'
                << value.first_resolution << '\t' << value.last_resolution
                << '\t' << value.representative_resolution << '\t'
                << value.c90 << '\t' << value.n_communities << '\t'
                << value.minimum_seed_stability << '\t'
                << value.minimum_adjacent_persistence << '\n';
        }
    }
    {
        std::ofstream output(root / "selected_levels.tsv");
        output << std::setprecision(17)
            << "level\tresolution\tc90\tcommunities\tevaluation\tplateau"
               "\tstable_plateau\tfallback\tfallback_ceiling_relaxed"
               "\tmean_seed_ari\tminimum_seed_ari\n";
        for (const auto& value : result.levels) {
            output << value.level << '\t' << value.resolution << '\t'
                << value.c90 << '\t' << value.n_communities << '\t'
                << value.evaluation << '\t' << value.plateau << '\t'
                << value.stable_plateau << '\t' << value.fallback << '\t'
                << value.fallback_ceiling_relaxed << '\t'
                << value.mean_pairwise_ari << '\t'
                << value.minimum_pairwise_ari << '\n';
        }
    }
}

json selection_options_json(const SelectionArtifactOptions& options) {
    const auto& value = options.selection;
    return {
        {"level1_c90_minimum", value.level1_c90_minimum},
        {"level1_c90_maximum", value.level1_c90_maximum},
        {"min_level", value.minimum_levels},
        {"max_level", value.maximum_levels},
        {"next_level_c90_multiplier", value.next_level_c90_multiplier},
        {"fallback_c90_max_multiplier", value.fallback_c90_max_multiplier},
        {"scout_max_iterations", value.scout_max_iterations},
        {"maximum_scout_steps", value.maximum_scout_steps},
        {"maximum_midpoints", value.maximum_midpoints},
        {"final_restarts", value.final_restarts},
        {"stop_c90", value.stop_c90},
        {"maximum_scan_steps", value.maximum_scan_steps},
        {"initial_resolution", value.initial_resolution},
        {"scout_resolution_factor", value.scout_resolution_factor},
        {"scan_resolution_factor", value.scan_resolution_factor},
        {"seed_stability_threshold", value.seed_stability_threshold},
        {"persistence_threshold", value.persistence_threshold},
        {"minimum_resolution", value.minimum_resolution},
        {"maximum_resolution", value.maximum_resolution},
        {"seed", value.seed},
        {"threads", value.n_threads}
    };
}

json load_diffusion_selection_identity(const fs::path& input) {
    const fs::path path = resolve_artifact_manifest(input);
    json manifest = read_json(path);
    if (manifest.value("artifact_type", "") != "punkst.multires.diffusion"
            || manifest.value("schema_version", 0) != 1) {
        throw std::runtime_error(
            "Unexpected diffusion artifact type or schema: " + path.string());
    }
    if (!manifest.contains("selection_identity_fingerprint")) {
        // Compatibility for early schema-v1 artifacts whose full manifest can
        // be reproduced byte-for-byte by this runtime.
        return load_verified_artifact(input, "punkst.multires.diffusion");
    }
    const std::string declared = manifest.at(
        "selection_identity_fingerprint").get<std::string>();
    const std::string full_fingerprint = manifest.value("fingerprint", "");
    if (declared.size() != 64 || full_fingerprint.size() != 64
            || !manifest.contains("source")
            || !manifest.at("source").is_object()
            || !manifest.contains("populations")
            || !manifest.at("populations").is_object()) {
        throw std::runtime_error(
            "Diffusion selection identity is malformed: " + path.string());
    }
    std::vector<std::string> populations;
    for (const auto& item : manifest.at("populations").items()) {
        if (item.key() != "points" && item.key() != "microclusters") {
            throw std::runtime_error(
                "Unknown diffusion population: " + item.key());
        }
        populations.push_back(item.key());
    }
    if (populations.empty()) {
        throw std::runtime_error("Diffusion artifact has no populations");
    }
    std::sort(populations.begin(), populations.end());
    const json identity = {
        {"artifact_type", "punkst.multires.diffusion.selection_identity"},
        {"schema_version", 1},
        {"graph_fingerprint", manifest.at("source").at(
            "graph_fingerprint")},
        {"populations", populations}
    };
    const std::string actual = artifact_fingerprint(identity,
        path.parent_path(), {});
    if (actual != declared) {
        throw std::runtime_error(
            "Diffusion selection identity does not match: " + path.string());
    }
    return manifest;
}

} // namespace

const char* scan_population_name(ScanPopulation population) {
    switch (population) {
        case ScanPopulation::Auto: return "auto";
        case ScanPopulation::Points: return "points";
        case ScanPopulation::Microclusters: return "microclusters";
    }
    throw std::invalid_argument("Unknown scan population");
}

ScanPopulation parse_scan_population(const std::string& value) {
    if (value == "auto") return ScanPopulation::Auto;
    if (value == "points") return ScanPopulation::Points;
    if (value == "microclusters") return ScanPopulation::Microclusters;
    throw std::invalid_argument(
        "scan_population must be auto, points, or microclusters");
}

SelectionArtifactResult run_multires_selection(
        const SelectionArtifactOptions& options) {
    const KnnGraphArtifactData graph_source = load_knn_graph_artifact(
        options.graph_manifest);
    SelectionArtifactResult out;
    out.graph_fingerprint = graph_source.fingerprint;
    out.identifiers = graph_source.identifiers;
    const int32_t points = graph_source.fine.graph.n_nodes;

    json diffusion;
    if (options.diffusion_manifest.has_value()) {
        diffusion = load_diffusion_selection_identity(
            *options.diffusion_manifest);
        out.diffusion_fingerprint =
            diffusion.at("fingerprint").get<std::string>();
        if (diffusion.at("source").at("graph_fingerprint")
                .get<std::string>() != out.graph_fingerprint) {
            throw std::runtime_error(
                "Diffusion artifact was built from a different graph");
        }
    }

    out.resolved_population = options.scan_population;
    if (out.resolved_population == ScanPopulation::Auto) {
        if (diffusion.is_null()) {
            throw std::invalid_argument(
                "scan_population=auto requires a diffusion artifact");
        }
        const json& populations = diffusion.at("populations");
        out.resolved_population = populations.contains("microclusters")
            ? ScanPopulation::Microclusters : ScanPopulation::Points;
    }
    if (out.resolved_population == ScanPopulation::Points
            && options.refine_on_full_graph) {
        throw std::invalid_argument(
            "Full-data refinement is only meaningful after a microcluster scan");
    }

    RawClusteringGraph scan_graph;
    std::vector<int64_t> scan_counts;
    std::vector<int32_t> fine_to_microcluster;
    if (out.resolved_population == ScanPopulation::Microclusters) {
        if (!graph_source.coarse.has_value()) {
            throw std::invalid_argument(
                "Microcluster scanning requires graph coarsening");
        }
        scan_graph = graph_source.coarse->graph;
        scan_counts = graph_source.coarse->fine_point_counts;
        fine_to_microcluster = graph_source.fine_to_microcluster;
    } else {
        scan_graph = graph_source.fine.graph;
        scan_counts = graph_source.fine.fine_point_counts;
    }

    out.selection = select_stable_resolutions(
        scan_graph, scan_counts, options.selection);
    RawClusteringGraph fine_graph;
    if (out.resolved_population == ScanPopulation::Microclusters
            && options.refine_on_full_graph) {
        fine_graph = graph_source.fine.graph;
    }
    for (const SelectedResolutionLevel& level : out.selection.levels) {
        out.scan_memberships.push_back(level.membership);
        std::vector<int32_t> full;
        if (out.resolved_population == ScanPopulation::Points) {
            full = level.membership;
        } else {
            full.resize(static_cast<size_t>(points));
            for (int32_t point = 0; point < points; ++point) {
                const int32_t microcluster =
                    fine_to_microcluster[static_cast<size_t>(point)];
                if (microcluster < 0
                        || microcluster >= static_cast<int32_t>(
                            level.membership.size())) {
                    throw std::runtime_error(
                        "Fine-to-microcluster membership is invalid");
                }
                full[static_cast<size_t>(point)] =
                    level.membership[static_cast<size_t>(microcluster)];
            }
            if (options.refine_on_full_graph) {
                full = refine_partition_on_full_graph(fine_graph, full,
                    level.resolution,
                    options.refinement_seed + level.level - 1);
            }
        }
        out.full_memberships.push_back(std::move(full));
    }
    return out;
}

void write_selection_artifact(const fs::path& output,
        const json& resolved_request,
        const SelectionArtifactOptions& options,
        const SelectionArtifactResult& result) {
    publish_directory_atomic(fs::absolute(output), [&](const fs::path& root) {
        std::vector<json> arrays;
        std::vector<json> levels;
        fs::create_directories(root / "partitions");
        for (size_t index = 0; index < result.selection.levels.size(); ++index) {
            const SelectedResolutionLevel& selected =
                result.selection.levels[index];
            const std::string stem = "level" + std::to_string(selected.level);
            const json scan_spec = write_array(root,
                "partitions/" + stem + "_scan.i32",
                result.scan_memberships[index]);
            const json full_spec = write_array(root,
                "partitions/" + stem + "_full.i32",
                result.full_memberships[index]);
            arrays.push_back(scan_spec);
            arrays.push_back(full_spec);
            const fs::path table_path = root / ("partitions/" + stem + ".tsv");
            std::ofstream table(table_path);
            if (!table) {
                throw std::runtime_error(
                    "Cannot write partition table: " + table_path.string());
            }
            table << "id\tcluster\n";
            for (size_t row = 0; row < result.identifiers.size(); ++row) {
                table << result.identifiers[row] << '\t'
                    << result.full_memberships[index][row] << '\n';
            }
            table.close();
            if (!table) {
                throw std::runtime_error(
                    "Failed writing partition table: " + table_path.string());
            }
            const ResolutionEvaluation& evaluation = result.selection.evaluations[
                static_cast<size_t>(selected.evaluation)];
            std::vector<int64_t> unit_counts(
                result.full_memberships[index].size(), 1);
            const int32_t full_c90 = weighted_c90(
                result.full_memberships[index], unit_counts);
            const int32_t full_communities = 1 + *std::max_element(
                result.full_memberships[index].begin(),
                result.full_memberships[index].end());
            levels.push_back({
                {"level", selected.level},
                {"resolution", selected.resolution},
                {"selection_c90", selected.c90},
                {"scan_communities", selected.n_communities},
                {"full_c90", full_c90},
                {"full_communities", full_communities},
                {"evaluation", selected.evaluation},
                {"plateau", selected.plateau < 0
                    ? json(nullptr) : json(selected.plateau)},
                {"stable_plateau", selected.stable_plateau},
                {"fallback", selected.fallback},
                {"fallback_ceiling_relaxed",
                    selected.fallback_ceiling_relaxed},
                {"mean_seed_ari", evaluation.mean_pairwise_ari},
                {"minimum_seed_ari", evaluation.minimum_pairwise_ari},
                {"scan_membership", scan_spec},
                {"full_membership", full_spec},
                {"partition_table", "partitions/" + stem + ".tsv"},
                {"partition_sha256", sha256_file(table_path)},
                {"full_data_refined", options.refine_on_full_graph},
                {"full_data_refinement_seed", options.refine_on_full_graph
                    ? json(options.refinement_seed + selected.level - 1)
                    : json(nullptr)}
            });
        }
        write_diagnostics(root, result.selection);

        json evaluations = json::array();
        for (const auto& value : result.selection.evaluations) {
            evaluations.push_back({
                {"resolution", value.resolution},
                {"c90", value.c90},
                {"communities", value.n_communities},
                {"mean_seed_ari", value.mean_pairwise_ari},
                {"minimum_seed_ari", value.minimum_pairwise_ari},
                {"persistence_from_previous",
                    nullable_number(value.persistence_from_previous)},
                {"medoid_restart", value.medoid_restart},
                {"restart_seeds", value.restart_seeds},
                {"restart_communities", value.restart_n_communities},
                {"pairwise_ari", value.pairwise_ari},
                {"restart_quality", value.restart_quality},
                {"restart_iterations", value.restart_iterations},
                {"restart_converged", value.restart_converged}
            });
        }
        json manifest = {
            {"artifact_type", "punkst.multires.selection"},
            {"schema_version", 1},
            {"resolved_request", resolved_request},
            {"source", {
                {"graph_manifest", resolve_artifact_manifest(
                    options.graph_manifest).string()},
                {"graph_fingerprint", result.graph_fingerprint},
                {"diffusion_manifest", options.diffusion_manifest.has_value()
                    ? json(resolve_artifact_manifest(
                        *options.diffusion_manifest).string())
                    : json(nullptr)},
                {"diffusion_fingerprint",
                    result.diffusion_fingerprint.has_value()
                        ? json(*result.diffusion_fingerprint) : json(nullptr)}
            }},
            {"requested_scan_population",
                scan_population_name(options.scan_population)},
            {"resolved_scan_population",
                scan_population_name(result.resolved_population)},
            {"selection_options", selection_options_json(options)},
            {"anchor_resolution", result.selection.anchor_resolution},
            {"selection_seconds", result.selection.selection_seconds},
            {"levels", levels},
            {"diagnostics", {
                {"scout_table", "diagnostics/scout.tsv"},
                {"scout_sha256", sha256_file(
                    root / "diagnostics/scout.tsv")},
                {"evaluations_table", "diagnostics/evaluations.tsv"},
                {"evaluations_sha256", sha256_file(
                    root / "diagnostics/evaluations.tsv")},
                {"restarts_table", "diagnostics/restarts.tsv"},
                {"restarts_sha256", sha256_file(
                    root / "diagnostics/restarts.tsv")},
                {"pairwise_ari_table", "diagnostics/pairwise_ari.tsv"},
                {"pairwise_ari_sha256", sha256_file(
                    root / "diagnostics/pairwise_ari.tsv")},
                {"plateaus_table", "diagnostics/plateaus.tsv"},
                {"plateaus_sha256", sha256_file(
                    root / "diagnostics/plateaus.tsv")},
                {"selected_levels_table", "selected_levels.tsv"},
                {"selected_levels_sha256", sha256_file(
                    root / "selected_levels.tsv")},
                {"evaluations", evaluations}
            }}
        };
        manifest["fingerprint"] = artifact_fingerprint(manifest, root, arrays);
        write_json(root / "manifest.json", manifest, 2);
    });
}

} // namespace punkst::multires
