#pragma once

#include "multires_core/artifacts.hpp"
#include "multires_core/diffusion_graph.hpp"
#include "multires_core/graph_coarsening.hpp"

#include <filesystem>
#include <optional>

namespace punkst::multires {

struct MetricGraphOptions {
    ThetaReadOptions theta;
    HellingerKnnGraphOptions graph;
    bool coarsening_enabled = false;
    GraphCoarseningOptions coarsening;
    bool diffusion_sidecar = false;
};

struct MetricGraphTimings {
    double input_seconds = 0.0;
    double graph_seconds = 0.0;
    double coarsening_seconds = 0.0;
    double total_seconds = 0.0;
};

struct MetricGraphResult {
    ThetaTable theta;
    HellingerKnnGraph graph;
    std::optional<GraphCoarseningResult> coarsening;
    MetricGraphTimings timings;
};

MetricGraphResult build_metric_graph(
    const std::filesystem::path& theta_path,
    const MetricGraphOptions& options = MetricGraphOptions());

void write_metric_graph_artifact(
    const std::filesystem::path& output,
    const std::filesystem::path& theta_path,
    const json& resolved_request,
    const MetricGraphOptions& options,
    const MetricGraphResult& result);

} // namespace punkst::multires
