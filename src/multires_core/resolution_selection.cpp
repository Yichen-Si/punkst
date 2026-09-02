#include "multires_core/resolution_selection.hpp"

#include "clustering_core/leiden.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

double elapsed_seconds(const Clock::time_point& begin) {
    return std::chrono::duration<double>(Clock::now() - begin).count();
}

uint64_t splitmix64(uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

int32_t resolution_seed(
        int32_t base, double resolution, int32_t restart, uint64_t phase) {
    uint64_t bits = 0;
    std::memcpy(&bits, &resolution, sizeof(bits));
    uint64_t value = splitmix64(static_cast<uint32_t>(base) ^ phase);
    value = splitmix64(value ^ bits);
    value = splitmix64(value ^ static_cast<uint64_t>(restart + 1));
    return static_cast<int32_t>(value & 0x7fffffffULL);
}

std::vector<int32_t> membership_of(const LeidenResult& result) {
    return std::vector<int32_t>(result.membership.data(),
        result.membership.data() + result.membership.size());
}

void validate_options(const ResolutionSelectionOptions& options) {
    if (options.level1_c90_minimum <= 0
        || options.level1_c90_maximum < options.level1_c90_minimum
        || options.minimum_levels <= 0
        || options.maximum_levels < options.minimum_levels
        || !(options.next_level_c90_multiplier > 1.0)
        || !(options.fallback_c90_max_multiplier
            >= options.next_level_c90_multiplier)
        || options.scout_max_iterations <= 0
        || options.maximum_scout_steps <= 0
        || options.maximum_midpoints < 0 || options.final_restarts <= 0
        || options.stop_c90 < options.level1_c90_minimum
        || options.maximum_scan_steps < 2
        || !(options.initial_resolution > 0.0)
        || !(options.scout_resolution_factor > 1.0)
        || !(options.scan_resolution_factor > 1.0)
        || !(options.seed_stability_threshold >= 0.0)
        || !(options.seed_stability_threshold <= 1.0)
        || !(options.persistence_threshold >= 0.0)
        || !(options.persistence_threshold <= 1.0)
        || !(options.minimum_resolution > 0.0)
        || !(options.maximum_resolution >= options.minimum_resolution)
        || options.initial_resolution < options.minimum_resolution
        || options.initial_resolution > options.maximum_resolution
        || options.seed < 0 || options.n_threads <= 0
        || !std::isfinite(options.next_level_c90_multiplier)
        || !std::isfinite(options.fallback_c90_max_multiplier)
        || !std::isfinite(options.initial_resolution)
        || !std::isfinite(options.scout_resolution_factor)
        || !std::isfinite(options.scan_resolution_factor)
        || !std::isfinite(options.seed_stability_threshold)
        || !std::isfinite(options.persistence_threshold)
        || !std::isfinite(options.minimum_resolution)
        || !std::isfinite(options.maximum_resolution)) {
        throw std::invalid_argument("Invalid resolution-selection options");
    }
}

void validate_graph(const RawClusteringGraph& graph,
        const std::vector<int64_t>& fine_counts) {
    if (graph.n_nodes < 2 || graph.edges.empty()
        || graph.edges.size() != graph.weights.size()
        || graph.component_labels.size() != static_cast<size_t>(graph.n_nodes)
        || fine_counts.size() != static_cast<size_t>(graph.n_nodes)) {
        throw std::invalid_argument("Invalid resolution-selection graph");
    }
    std::pair<int32_t, int32_t> previous{-1, -1};
    for (size_t index = 0; index < graph.edges.size(); ++index) {
        const auto edge = graph.edges[index];
        const double weight = graph.weights[index];
        if (edge.first < 0 || edge.second < edge.first
            || edge.second >= graph.n_nodes || edge < previous
            || !(weight > 0.0) || !std::isfinite(weight)
            || graph.component_labels[static_cast<size_t>(edge.first)] < 0
            || graph.component_labels[static_cast<size_t>(edge.first)]
                != graph.component_labels[static_cast<size_t>(edge.second)]) {
            throw std::invalid_argument(
                "Resolution-selection graph edges are invalid");
        }
        previous = edge;
    }
    if (std::any_of(graph.component_labels.begin(),
            graph.component_labels.end(), [](int32_t value) {
                return value < 0;
            })) {
        throw std::invalid_argument(
            "Resolution-selection component labels must be nonnegative");
    }
    for (const int64_t count : fine_counts) {
        if (count <= 0) {
            throw std::invalid_argument(
                "Resolution-selection fine-point counts must be positive");
        }
    }
}

void audit_components(const std::vector<int32_t>& membership,
        const std::vector<int32_t>& components) {
    std::unordered_map<int32_t, int32_t> community_component;
    for (size_t node = 0; node < membership.size(); ++node) {
        const auto inserted = community_component.emplace(
            membership[node], components[node]);
        if (!inserted.second && inserted.first->second != components[node]) {
            throw std::runtime_error(
                "Selected partition crossed a non-bridge component");
        }
    }
}

struct ActiveGraph {
    int32_t original_nodes = 0;
    int32_t n_nodes = 0;
    std::vector<std::pair<int32_t, int32_t>> edges;
    std::vector<double> weights;
    std::vector<int32_t> active_to_original;
    std::vector<int32_t> isolated;
};

ActiveGraph make_active_graph(const RawClusteringGraph& graph) {
    ActiveGraph out;
    out.original_nodes = graph.n_nodes;
    std::vector<uint8_t> incident(static_cast<size_t>(graph.n_nodes), 0);
    for (const auto& edge : graph.edges) {
        incident[static_cast<size_t>(edge.first)] = 1;
        incident[static_cast<size_t>(edge.second)] = 1;
    }
    std::vector<int32_t> original_to_active(
        static_cast<size_t>(graph.n_nodes), -1);
    for (int32_t node = 0; node < graph.n_nodes; ++node) {
        if (incident[static_cast<size_t>(node)]) {
            original_to_active[static_cast<size_t>(node)] = out.n_nodes++;
            out.active_to_original.push_back(node);
        } else {
            out.isolated.push_back(node);
        }
    }
    out.edges.reserve(graph.edges.size());
    out.weights = graph.weights;
    for (const auto& edge : graph.edges) {
        out.edges.emplace_back(
            original_to_active[static_cast<size_t>(edge.first)],
            original_to_active[static_cast<size_t>(edge.second)]);
    }
    return out;
}

LeidenResult restore_isolated(LeidenResult result, const ActiveGraph& graph) {
    if (graph.isolated.empty()) return result;
    Eigen::VectorXi membership(graph.original_nodes);
    for (int32_t active = 0; active < graph.n_nodes; ++active) {
        membership(graph.active_to_original[static_cast<size_t>(active)]) =
            result.membership(active);
    }
    int32_t next = result.n_communities;
    for (const int32_t node : graph.isolated) membership(node) = next++;
    result.membership = std::move(membership);
    result.n_communities = next;
    return result;
}

struct ScoutRecord {
    ResolutionScoutEvaluation diagnostics;
    std::vector<int32_t> membership;
};

struct FinalRecord {
    ResolutionEvaluation diagnostics;
    std::vector<int32_t> medoid_membership;
};

bool more_stable(const FinalRecord& candidate, const FinalRecord& incumbent) {
    if (candidate.diagnostics.minimum_pairwise_ari
            != incumbent.diagnostics.minimum_pairwise_ari) {
        return candidate.diagnostics.minimum_pairwise_ari
            > incumbent.diagnostics.minimum_pairwise_ari;
    }
    if (candidate.diagnostics.mean_pairwise_ari
            != incumbent.diagnostics.mean_pairwise_ari) {
        return candidate.diagnostics.mean_pairwise_ari
            > incumbent.diagnostics.mean_pairwise_ari;
    }
    if (candidate.diagnostics.c90 != incumbent.diagnostics.c90) {
        return candidate.diagnostics.c90 < incumbent.diagnostics.c90;
    }
    return candidate.diagnostics.resolution
        < incumbent.diagnostics.resolution;
}

int32_t c90_distance(int32_t value, int32_t minimum, int32_t maximum) {
    if (value < minimum) return minimum - value;
    if (value > maximum) return value - maximum;
    return 0;
}

int32_t ceiling_product(int32_t value, double multiplier) {
    const long double product = static_cast<long double>(value) * multiplier;
    if (product > std::numeric_limits<int32_t>::max()) {
        return std::numeric_limits<int32_t>::max();
    }
    return static_cast<int32_t>(std::ceil(product));
}

SelectedResolutionLevel make_level(int32_t level, int32_t evaluation,
        int32_t plateau, bool stable, bool fallback, bool relaxed,
        const std::vector<FinalRecord>& finals) {
    const FinalRecord& source = finals[static_cast<size_t>(evaluation)];
    SelectedResolutionLevel out;
    out.level = level;
    out.evaluation = evaluation;
    out.plateau = plateau;
    out.resolution = source.diagnostics.resolution;
    out.c90 = source.diagnostics.c90;
    out.n_communities = source.diagnostics.n_communities;
    out.mean_pairwise_ari = source.diagnostics.mean_pairwise_ari;
    out.minimum_pairwise_ari = source.diagnostics.minimum_pairwise_ari;
    out.stable_plateau = stable;
    out.fallback = fallback;
    out.fallback_ceiling_relaxed = relaxed;
    out.membership = source.medoid_membership;
    return out;
}

} // namespace

ResolutionSelectionResult select_stable_resolutions(
        const RawClusteringGraph& graph,
        const std::vector<int64_t>& fine_point_counts,
        const ResolutionSelectionOptions& options) {
    const Clock::time_point begin = Clock::now();
    validate_options(options);
    validate_graph(graph, fine_point_counts);
    const ActiveGraph active_graph = make_active_graph(graph);

    std::map<double, ScoutRecord> scouts;
    auto evaluate_scout = [&](double resolution) -> ScoutRecord& {
        const auto found = scouts.find(resolution);
        if (found != scouts.end()) return found->second;
        LeidenOptions leiden_options;
        leiden_options.resolution = resolution;
        leiden_options.max_iterations = options.scout_max_iterations;
        leiden_options.seed = resolution_seed(
            options.seed, resolution, 0, 0x53454c53434f5554ULL);
        LeidenResult run = restore_isolated(leiden_cluster(
            active_graph.n_nodes, active_graph.edges, active_graph.weights,
            leiden_options), active_graph);
        ScoutRecord record;
        record.membership = membership_of(run);
        audit_components(record.membership, graph.component_labels);
        record.diagnostics = {resolution,
            weighted_c90(record.membership, fine_point_counts),
            run.n_communities, run.quality, run.iterations, run.converged};
        return scouts.emplace(resolution, std::move(record)).first->second;
    };

    ScoutRecord* lower = nullptr;
    ScoutRecord* upper = nullptr;
    ScoutRecord* current = &evaluate_scout(options.initial_resolution);
    for (int32_t step = 0; step < options.maximum_scout_steps; ++step) {
        const bool qualifies = current->diagnostics.c90
            >= options.level1_c90_minimum;
        const double next_resolution = std::clamp(
            qualifies
                ? current->diagnostics.resolution
                    / options.scout_resolution_factor
                : current->diagnostics.resolution
                    * options.scout_resolution_factor,
            options.minimum_resolution, options.maximum_resolution);
        if (next_resolution == current->diagnostics.resolution) break;
        ScoutRecord* next = &evaluate_scout(next_resolution);
        const bool next_qualifies = next->diagnostics.c90
            >= options.level1_c90_minimum;
        if (qualifies != next_qualifies) {
            lower = qualifies ? next : current;
            upper = qualifies ? current : next;
            break;
        }
        current = next;
    }
    if (lower != nullptr && upper != nullptr) {
        for (int32_t step = 0; step < options.maximum_midpoints; ++step) {
            const double midpoint = std::sqrt(
                lower->diagnostics.resolution
                * upper->diagnostics.resolution);
            ScoutRecord* trial = &evaluate_scout(midpoint);
            if (trial->diagnostics.c90 >= options.level1_c90_minimum) {
                upper = trial;
            } else {
                lower = trial;
            }
        }
    }
    if (upper == nullptr) {
        for (auto& item : scouts) {
            if (item.second.diagnostics.c90
                    < options.level1_c90_minimum) continue;
            if (upper == nullptr || item.first < upper->diagnostics.resolution) {
                upper = &item.second;
            }
        }
    }
    if (upper == nullptr) {
        throw std::runtime_error(
            "Resolution scout could not reach the Level-1 C90 range");
    }
    if (lower == nullptr) {
        for (auto& item : scouts) {
            if (item.first >= upper->diagnostics.resolution) continue;
            if (lower == nullptr || item.first > lower->diagnostics.resolution) {
                lower = &item.second;
            }
        }
    }

    std::vector<double> resolutions;
    if (lower != nullptr) {
        resolutions.push_back(lower->diagnostics.resolution);
    } else {
        const double context = std::max(options.minimum_resolution,
            upper->diagnostics.resolution / options.scan_resolution_factor);
        if (context < upper->diagnostics.resolution) {
            resolutions.push_back(context);
        }
    }
    resolutions.push_back(upper->diagnostics.resolution);
    std::sort(resolutions.begin(), resolutions.end());
    resolutions.erase(std::unique(resolutions.begin(), resolutions.end()),
        resolutions.end());

    std::vector<FinalRecord> finals;
    auto evaluate_final = [&](double resolution) {
        std::vector<int32_t> seeds(static_cast<size_t>(options.final_restarts));
        for (int32_t restart = 0; restart < options.final_restarts; ++restart) {
            seeds[static_cast<size_t>(restart)] = resolution_seed(
                options.seed, resolution, restart, 0x53454c46494e414cULL);
        }
        LeidenOptions leiden_options;
        leiden_options.resolution = resolution;
        leiden_options.max_iterations = -1;
        std::vector<LeidenResult> runs = leiden_cluster_restarts(
            active_graph.n_nodes, active_graph.edges, active_graph.weights,
            leiden_options, seeds, options.n_threads);
        for (LeidenResult& run : runs) {
            run = restore_isolated(std::move(run), active_graph);
        }
        std::vector<std::vector<int32_t>> memberships;
        memberships.reserve(runs.size());
        for (const LeidenResult& run : runs) {
            memberships.push_back(membership_of(run));
            audit_components(memberships.back(), graph.component_labels);
        }
        std::vector<double> medoid_mean(runs.size(), 1.0);
        std::vector<double> pairwise;
        double pairwise_sum = 0.0;
        double pairwise_minimum = 1.0;
        int32_t pair_count = 0;
        if (runs.size() > 1) {
            std::fill(medoid_mean.begin(), medoid_mean.end(), 0.0);
            for (size_t first = 0; first < runs.size(); ++first) {
                for (size_t second = first + 1; second < runs.size(); ++second) {
                    const double ari = fine_weighted_adjusted_rand_index(
                        memberships[first], memberships[second],
                        fine_point_counts);
                    pairwise.push_back(ari);
                    pairwise_sum += ari;
                    pairwise_minimum = std::min(pairwise_minimum, ari);
                    medoid_mean[first] += ari;
                    medoid_mean[second] += ari;
                    ++pair_count;
                }
            }
            for (double& value : medoid_mean) {
                value /= static_cast<double>(runs.size() - 1);
            }
        }
        int32_t medoid = 0;
        for (int32_t restart = 1; restart < options.final_restarts; ++restart) {
            if (medoid_mean[static_cast<size_t>(restart)]
                    > medoid_mean[static_cast<size_t>(medoid)]
                || (medoid_mean[static_cast<size_t>(restart)]
                        == medoid_mean[static_cast<size_t>(medoid)]
                    && runs[static_cast<size_t>(restart)].quality
                        > runs[static_cast<size_t>(medoid)].quality)) {
                medoid = restart;
            }
        }
        FinalRecord record;
        record.medoid_membership = memberships[static_cast<size_t>(medoid)];
        auto& diagnostics = record.diagnostics;
        diagnostics.resolution = resolution;
        diagnostics.c90 = weighted_c90(
            record.medoid_membership, fine_point_counts);
        diagnostics.n_communities = runs[static_cast<size_t>(medoid)].n_communities;
        diagnostics.mean_pairwise_ari = pair_count > 0
            ? pairwise_sum / pair_count : 1.0;
        diagnostics.minimum_pairwise_ari = pairwise_minimum;
        diagnostics.medoid_restart = medoid;
        diagnostics.restart_seeds = seeds;
        diagnostics.pairwise_ari = std::move(pairwise);
        for (const LeidenResult& run : runs) {
            diagnostics.restart_n_communities.push_back(run.n_communities);
            diagnostics.restart_quality.push_back(run.quality);
            diagnostics.restart_iterations.push_back(run.iterations);
            diagnostics.restart_converged.push_back(run.converged);
        }
        if (!finals.empty()) {
            diagnostics.persistence_from_previous =
                fine_weighted_adjusted_rand_index(
                    finals.back().medoid_membership,
                    record.medoid_membership, fine_point_counts);
        }
        finals.push_back(std::move(record));
    };

    for (const double resolution : resolutions) evaluate_final(resolution);
    std::vector<int32_t> singleton_membership(
        static_cast<size_t>(graph.n_nodes));
    std::iota(singleton_membership.begin(), singleton_membership.end(), 0);
    const int32_t effective_stop_c90 = std::min(options.stop_c90,
        weighted_c90(singleton_membership, fine_point_counts));
    double resolution = resolutions.back();
    while (static_cast<int32_t>(finals.size()) < options.maximum_scan_steps
            && finals.back().diagnostics.c90 < effective_stop_c90) {
        const double next = std::min(options.maximum_resolution,
            resolution * options.scan_resolution_factor);
        if (next == resolution) break;
        evaluate_final(next);
        resolution = next;
    }

    std::vector<ResolutionPlateau> plateaus;
    for (size_t index = 1; index < finals.size(); ++index) {
        const FinalRecord& previous = finals[index - 1];
        const FinalRecord& current_final = finals[index];
        const bool stable_pair =
            previous.diagnostics.minimum_pairwise_ari
                >= options.seed_stability_threshold
            && current_final.diagnostics.minimum_pairwise_ari
                >= options.seed_stability_threshold
            && current_final.diagnostics.persistence_from_previous
                >= options.persistence_threshold;
        if (!stable_pair) continue;
        const size_t representative = more_stable(current_final, previous)
            ? index : index - 1;
        if (!plateaus.empty()) {
            ResolutionPlateau& incumbent = plateaus.back();
            const double ari = fine_weighted_adjusted_rand_index(
                incumbent.membership,
                finals[representative].medoid_membership,
                fine_point_counts);
            if (ari >= options.persistence_threshold
                    && incumbent.last_evaluation
                        == static_cast<int32_t>(index - 1)) {
                incumbent.last_evaluation = static_cast<int32_t>(index);
                incumbent.last_resolution = current_final.diagnostics.resolution;
                incumbent.minimum_seed_stability = std::min(
                    incumbent.minimum_seed_stability,
                    std::min(previous.diagnostics.minimum_pairwise_ari,
                        current_final.diagnostics.minimum_pairwise_ari));
                incumbent.minimum_adjacent_persistence = std::min(
                    incumbent.minimum_adjacent_persistence,
                    current_final.diagnostics.persistence_from_previous);
                if (more_stable(finals[representative],
                        finals[static_cast<size_t>(
                            incumbent.representative_evaluation)])) {
                    incumbent.representative_evaluation =
                        static_cast<int32_t>(representative);
                    incumbent.representative_resolution =
                        finals[representative].diagnostics.resolution;
                    incumbent.c90 = finals[representative].diagnostics.c90;
                    incumbent.n_communities =
                        finals[representative].diagnostics.n_communities;
                    incumbent.membership =
                        finals[representative].medoid_membership;
                }
                continue;
            }
        }
        ResolutionPlateau plateau;
        plateau.first_evaluation = static_cast<int32_t>(index - 1);
        plateau.last_evaluation = static_cast<int32_t>(index);
        plateau.representative_evaluation =
            static_cast<int32_t>(representative);
        plateau.first_resolution = previous.diagnostics.resolution;
        plateau.last_resolution = current_final.diagnostics.resolution;
        plateau.representative_resolution =
            finals[representative].diagnostics.resolution;
        plateau.c90 = finals[representative].diagnostics.c90;
        plateau.n_communities =
            finals[representative].diagnostics.n_communities;
        plateau.minimum_seed_stability = std::min(
            previous.diagnostics.minimum_pairwise_ari,
            current_final.diagnostics.minimum_pairwise_ari);
        plateau.minimum_adjacent_persistence =
            current_final.diagnostics.persistence_from_previous;
        plateau.membership = finals[representative].medoid_membership;
        plateaus.push_back(std::move(plateau));
    }

    std::vector<SelectedResolutionLevel> levels;
    int32_t level1_eval = -1;
    int32_t level1_plateau = -1;
    for (size_t plateau = 0; plateau < plateaus.size(); ++plateau) {
        for (int32_t evaluation = plateaus[plateau].first_evaluation;
                evaluation <= plateaus[plateau].last_evaluation; ++evaluation) {
            const int32_t c90 = finals[static_cast<size_t>(evaluation)]
                .diagnostics.c90;
            if (c90 < options.level1_c90_minimum
                    || c90 > options.level1_c90_maximum) continue;
            if (level1_eval < 0 || more_stable(
                    finals[static_cast<size_t>(evaluation)],
                    finals[static_cast<size_t>(level1_eval)])) {
                level1_eval = evaluation;
                level1_plateau = static_cast<int32_t>(plateau);
            }
        }
    }
    if (level1_eval >= 0) {
        levels.push_back(make_level(1, level1_eval, level1_plateau,
            true, false, false, finals));
    } else {
        int32_t fallback = -1;
        bool any_in_range = std::any_of(finals.begin(), finals.end(),
            [&](const FinalRecord& value) {
                return c90_distance(value.diagnostics.c90,
                    options.level1_c90_minimum,
                    options.level1_c90_maximum) == 0;
            });
        for (size_t index = 0; index < finals.size(); ++index) {
            const int32_t distance = c90_distance(finals[index].diagnostics.c90,
                options.level1_c90_minimum, options.level1_c90_maximum);
            if (any_in_range && distance != 0) continue;
            if (fallback < 0) {
                fallback = static_cast<int32_t>(index);
                continue;
            }
            const int32_t incumbent_distance = c90_distance(
                finals[static_cast<size_t>(fallback)].diagnostics.c90,
                options.level1_c90_minimum, options.level1_c90_maximum);
            if (distance < incumbent_distance
                || (distance == incumbent_distance
                    && more_stable(finals[index],
                        finals[static_cast<size_t>(fallback)]))) {
                fallback = static_cast<int32_t>(index);
            }
        }
        if (fallback < 0) {
            throw std::runtime_error("Resolution scan produced no partition");
        }
        levels.push_back(make_level(1, fallback, -1,
            false, true, false, finals));
    }

    while (static_cast<int32_t>(levels.size()) < options.maximum_levels) {
        const SelectedResolutionLevel& previous = levels.back();
        const int32_t lower_c90 = ceiling_product(
            previous.c90, options.next_level_c90_multiplier);
        int32_t stable_eval = -1;
        int32_t stable_plateau = -1;
        for (size_t plateau = 0; plateau < plateaus.size(); ++plateau) {
            for (int32_t evaluation = plateaus[plateau].first_evaluation;
                    evaluation <= plateaus[plateau].last_evaluation;
                    ++evaluation) {
                const FinalRecord& candidate =
                    finals[static_cast<size_t>(evaluation)];
                if (evaluation <= previous.evaluation
                        || candidate.diagnostics.c90 < lower_c90) continue;
                if (fine_weighted_adjusted_rand_index(previous.membership,
                        candidate.medoid_membership, fine_point_counts)
                        >= options.persistence_threshold) continue;
                if (stable_eval < 0
                    || candidate.diagnostics.c90
                        < finals[static_cast<size_t>(stable_eval)].diagnostics.c90
                    || (candidate.diagnostics.c90
                            == finals[static_cast<size_t>(stable_eval)]
                                .diagnostics.c90
                        && more_stable(candidate,
                            finals[static_cast<size_t>(stable_eval)]))) {
                    stable_eval = evaluation;
                    stable_plateau = static_cast<int32_t>(plateau);
                }
            }
        }
        if (stable_eval >= 0) {
            levels.push_back(make_level(
                static_cast<int32_t>(levels.size() + 1), stable_eval,
                stable_plateau, true, false, false, finals));
            continue;
        }
        if (static_cast<int32_t>(levels.size()) >= options.minimum_levels) {
            break;
        }

        const int32_t upper_c90 = ceiling_product(
            previous.c90, options.fallback_c90_max_multiplier);
        int32_t fallback = -1;
        for (size_t index = 0; index < finals.size(); ++index) {
            const FinalRecord& candidate = finals[index];
            if (static_cast<int32_t>(index) <= previous.evaluation
                    || candidate.diagnostics.c90 < lower_c90
                    || candidate.diagnostics.c90 > upper_c90
                    || fine_weighted_adjusted_rand_index(previous.membership,
                        candidate.medoid_membership, fine_point_counts)
                        >= options.persistence_threshold) continue;
            if (fallback < 0 || more_stable(candidate,
                    finals[static_cast<size_t>(fallback)])) {
                fallback = static_cast<int32_t>(index);
            }
        }
        bool relaxed = false;
        if (fallback < 0) {
            relaxed = true;
            for (size_t index = 0; index < finals.size(); ++index) {
                const FinalRecord& candidate = finals[index];
                if (static_cast<int32_t>(index) <= previous.evaluation
                        || candidate.diagnostics.c90 < lower_c90
                        || fine_weighted_adjusted_rand_index(
                            previous.membership, candidate.medoid_membership,
                            fine_point_counts)
                            >= options.persistence_threshold) continue;
                if (fallback < 0 || more_stable(candidate,
                        finals[static_cast<size_t>(fallback)])) {
                    fallback = static_cast<int32_t>(index);
                }
            }
        }
        if (fallback < 0) {
            throw std::runtime_error(
                "Resolution scan cannot satisfy minimum_levels");
        }
        levels.push_back(make_level(
            static_cast<int32_t>(levels.size() + 1), fallback, -1,
            false, true, relaxed, finals));
    }

    ResolutionSelectionResult result;
    result.anchor_resolution = upper->diagnostics.resolution;
    if (!resolutions.empty()
            && resolutions.front() < upper->diagnostics.resolution) {
        result.has_lower_context = true;
        result.lower_context_resolution = resolutions.front();
    }
    result.selection_seconds = elapsed_seconds(begin);
    for (const auto& item : scouts) {
        result.scout_evaluations.push_back(item.second.diagnostics);
    }
    result.evaluations.reserve(finals.size());
    for (const FinalRecord& item : finals) {
        result.evaluations.push_back(item.diagnostics);
    }
    result.plateaus = std::move(plateaus);
    result.levels = std::move(levels);
    return result;
}

std::vector<int32_t> refine_partition_on_full_graph(
        const RawClusteringGraph& graph,
        const std::vector<int32_t>& initial_membership,
        double resolution,
        int32_t seed) {
    std::vector<int64_t> unit_counts(static_cast<size_t>(graph.n_nodes), 1);
    validate_graph(graph, unit_counts);
    if (initial_membership.size() != static_cast<size_t>(graph.n_nodes)
            || std::any_of(initial_membership.begin(),
                initial_membership.end(), [](int32_t label) {
                    return label < 0;
                })
            || !(resolution > 0.0) || !std::isfinite(resolution)
            || seed < 0) {
        throw std::invalid_argument("Invalid full-graph refinement input");
    }
    const ActiveGraph active = make_active_graph(graph);
    std::vector<double> strengths(static_cast<size_t>(active.n_nodes), 0.0);
    for (size_t index = 0; index < active.edges.size(); ++index) {
        const auto edge = active.edges[index];
        const double weight = active.weights[index];
        if (edge.first == edge.second) {
            strengths[static_cast<size_t>(edge.first)] += 2.0 * weight;
        } else {
            strengths[static_cast<size_t>(edge.first)] += weight;
            strengths[static_cast<size_t>(edge.second)] += weight;
        }
    }
    if (std::any_of(strengths.begin(), strengths.end(), [](double value) {
            return !(value > 0.0) || !std::isfinite(value);
        })) {
        throw std::invalid_argument(
            "Full-graph refinement requires positive node strengths");
    }
    LeidenOptions options;
    options.resolution = resolution;
    options.max_iterations = -1;
    options.seed = seed;
    std::vector<int32_t> active_initial(static_cast<size_t>(active.n_nodes));
    for (int32_t node = 0; node < active.n_nodes; ++node) {
        active_initial[static_cast<size_t>(node)] = initial_membership[
            static_cast<size_t>(active.active_to_original[
                static_cast<size_t>(node)])];
    }
    LeidenResult result = restore_isolated(leiden_cluster(
        active.n_nodes, active.edges, active.weights, strengths,
        active_initial, options), active);
    std::vector<int32_t> membership = membership_of(result);
    audit_components(membership, graph.component_labels);
    return membership;
}
