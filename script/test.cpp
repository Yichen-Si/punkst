#include "punkst.h"
#include "tiles2minibatch.hpp"
#include "utils_sys.hpp"

#include "nlohmann/json.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace {

using json = nlohmann::json;

std::vector<float> build_evenly_spaced_thin3d_zlevels(double zMin, double zMax, int32_t nLevels) {
    if (!(zMax > zMin) || nLevels <= 1) {
        error("%s: invalid thin 3D z-level configuration", __func__);
    }
    std::vector<float> zLevels(static_cast<size_t>(nLevels));
    const double zStep = (zMax - zMin) / static_cast<double>(nLevels);
    for (int32_t i = 0; i < nLevels; ++i) {
        zLevels[static_cast<size_t>(i)] = static_cast<float>(zMin + (static_cast<double>(i) + 0.5) * zStep);
    }
    return zLevels;
}

double nth_nearest_zlevel_distance(const std::vector<float>& zLevels, double z, int32_t nPick) {
    std::vector<double> dists;
    dists.reserve(zLevels.size());
    for (float level : zLevels) {
        dists.push_back(std::abs(z - static_cast<double>(level)));
    }
    std::nth_element(dists.begin(), dists.begin() + (nPick - 1), dists.end());
    return dists[static_cast<size_t>(nPick - 1)];
}

double thin3d_default_zreach(const std::vector<float>& zLevels, int32_t nPick, double zMin, double zMax) {
    if (zLevels.size() <= 1) {
        return 0.0;
    }
    nPick = std::min<int32_t>(std::max<int32_t>(nPick, 1), static_cast<int32_t>(zLevels.size()));
    std::vector<double> probes;
    probes.reserve(zLevels.size() * 2 + 2);
    probes.push_back(zMin);
    probes.push_back(zMax);
    for (size_t i = 0; i < zLevels.size(); ++i) {
        probes.push_back(static_cast<double>(zLevels[i]));
        if (i + 1 < zLevels.size()) {
            probes.push_back(0.5 * (static_cast<double>(zLevels[i]) + static_cast<double>(zLevels[i + 1])));
        }
    }
    double maxDist = 0.0;
    for (double z : probes) {
        maxDist = std::max(maxDist, nth_nearest_zlevel_distance(zLevels, z, nPick));
    }
    return maxDist;
}

double compute_dist_nu(double halfLifeDist) {
    if (!(halfLifeDist > 0.0 && halfLifeDist < 1.0)) {
        error("%s: --half-life-dist must be in (0, 1)", __func__);
    }
    return std::log(0.5) / std::log(halfLifeDist);
}

void ensure_parent_dir(const std::string& pathStr) {
    const std::filesystem::path path(pathStr);
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

struct ProbeParams {
    std::string outPrefix;
    double hexSize = -1.0;
    double hexGridDist = -1.0;
    double anchorDist = -1.0;
    double radius = -1.0;
    double halfLifeDist = 0.7;
    double zMin = std::numeric_limits<double>::quiet_NaN();
    double zMax = std::numeric_limits<double>::quiet_NaN();
    std::vector<float> thin3DZLevels;
    int32_t thin3DNZLevels = -1;
    int32_t nMoves = -1;
    double pixelDensity = 1.0;
    double pixelResolution = 1.0;
    double pixelResolutionZ = 1.0;
    double xmin = 0.0;
    double xmax = -1.0;
    double ymin = 0.0;
    double ymax = -1.0;
    double minInitCount = 10.0;
    bool ignoreOutsideZrange = false;
    bool writeHtml = false;
};

struct RawPoint {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    uint32_t feature = 0;
    uint32_t count = 0;
};

struct InitCandidateRow {
    size_t pointIndex = 0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    uint32_t feature = 0;
    double count = 0.0;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
};

struct AnchorRow {
    size_t anchorIndex = 0;
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    double totalCount = 0.0;
};

struct PixelAssignmentRow {
    size_t pixelIndex = 0;
    int32_t pxIndex = 0;
    int32_t pyIndex = 0;
    int32_t pzIndex = 0;
    double px = 0.0;
    double py = 0.0;
    double pz = 0.0;
    double pixelCount = 0.0;
    size_t anchorIndex = 0;
    double ax = 0.0;
    double ay = 0.0;
    double az = 0.0;
    double weight = 0.0;
};

class Thin3DProbeHarness final : public Tiles2MinibatchBase<float> {
public:
    using Base = Tiles2MinibatchBase<float>;
    using typename Base::vec2f_t;

    Thin3DProbeHarness(TileReader& tileReader, lineParserUnival& parser, double pixelResolution)
        : Base(1, 0.0, tileReader, "", &parser, MinibatchIoConfig{}, nullptr, pixelResolution, 0) {}

    int32_t getFactorCount() const override { return 1; }

    void processTile(TileData<float>&, int, int, vec2f_t*) override {}

    int32_t buildAnchorsThin3DPublic(TileData<float>& tileData,
        std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents,
        const HexGrid& hexGrid, int32_t nMoves, double minCount,
        double supportRadius, double distNu) {
        return this->buildAnchorsThin3D(tileData, anchors, documents, hexGrid, nMoves,
            minCount, supportRadius, distNu);
    }

    double buildMinibatchThin3DPublic(TileData<float>& tileData,
        std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
        double supportRadius, double distNu) {
        return this->buildMinibatchCore3D(tileData, anchors, minibatch, nullptr, supportRadius, distNu);
    }

    void forEachThin3DAnchorWithinRadiusPublic(float x, float y, float z,
        const HexGrid& hexGrid, int32_t nMoves, double supportRadius,
        const std::function<void(float, float, float)>& emit) const {
        this->forEachThin3DAnchorWithinRadius(x, y, z, hexGrid, nMoves, supportRadius,
            [&](const AnchorKey2D&, float ax, float ay, float az, double) {
                emit(ax, ay, az);
            });
    }
};

ProbeParams parse_params(int32_t argc, char** argv) {
    ProbeParams params;
    ParamList pl;
    pl.add_option("out-prefix", "Output prefix for generated TSV/JSON/HTML files", params.outPrefix, true)
      .add_option("hex-size", "Hexagon side length (alternative to --hex-grid-dist)", params.hexSize)
      .add_option("hex-grid-dist", "Hexagon center-to-center distance", params.hexGridDist)
      .add_option("anchor-dist", "Distance between adjacent x-y anchors", params.anchorDist)
      .add_option("radius", "Support radius for pixel-to-anchor assignment", params.radius)
      .add_option("half-life-dist", "Weight half-life ratio used by pixel-decode", params.halfLifeDist)
      .add_option("zmin", "Minimum z coordinate", params.zMin, true)
      .add_option("zmax", "Maximum z coordinate", params.zMax, true)
      .add_option("thin-3d-z-levels", "Explicit thin-3D z levels", params.thin3DZLevels)
      .add_option("thin-3d-n-z-levels", "Number of evenly spaced thin-3D z levels", params.thin3DNZLevels)
      .add_option("n-moves", "Number of x-y grid shifts used for anchor placement", params.nMoves)
      .add_option("pixel-density", "Synthetic sample density per pixel edge relative to pixel resolution", params.pixelDensity)
      .add_option("pixel-res", "Pixel resolution in x/y", params.pixelResolution)
      .add_option("pixel-res-z", "Pixel resolution in z", params.pixelResolutionZ)
      .add_option("xmin", "Minimum x coordinate of the synthetic domain", params.xmin)
      .add_option("xmax", "Maximum x coordinate of the synthetic domain", params.xmax)
      .add_option("ymin", "Minimum y coordinate of the synthetic domain", params.ymin)
      .add_option("ymax", "Maximum y coordinate of the synthetic domain", params.ymax)
      .add_option("min-init-count", "Minimum accumulated count for a thin-3D anchor to be retained", params.minInitCount)
      .add_option("ignore-outside-zrange", "Ignore raw points outside [zmin, zmax]", params.ignoreOutsideZrange)
      .add_option("write-html", "Write a Plotly HTML visualization beside the TSV outputs", params.writeHtml);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        throw;
    }

    if (!(params.zMax > params.zMin)) {
        error("%s: --zmax must be greater than --zmin", __func__);
    }
    if (params.pixelDensity <= 0.0) {
        error("%s: --pixel-density must be positive", __func__);
    }
    if (params.pixelResolution <= 0.0 || params.pixelResolutionZ <= 0.0) {
        error("%s: --pixel-res and --pixel-res-z must be positive", __func__);
    }
    const bool hasExplicitZLevels = !params.thin3DZLevels.empty();
    const bool hasGeneratedZLevels = params.thin3DNZLevels > 0;
    if (hasExplicitZLevels && hasGeneratedZLevels) {
        warning("Both --thin-3d-z-levels and --thin-3d-n-z-levels are set. Ignoring --thin-3d-n-z-levels.");
    }
    if (!hasExplicitZLevels) {
        if (params.thin3DNZLevels <= 1) {
            error("%s: thin 3D probe requires either --thin-3d-z-levels or --thin-3d-n-z-levels > 1", __func__);
        }
        params.thin3DZLevels = build_evenly_spaced_thin3d_zlevels(params.zMin, params.zMax, params.thin3DNZLevels);
    }

    if (params.hexSize <= 0.0 && params.hexGridDist <= 0.0) {
        error("%s: --hex-size or --hex-grid-dist is required", __func__);
    }
    if (params.nMoves <= 0 && params.anchorDist <= 0.0) {
        error("%s: --n-moves or --anchor-dist is required", __func__);
    }
    if (params.hexSize > 0.0) {
        params.hexGridDist = params.hexSize * std::sqrt(3.0);
    } else {
        params.hexSize = params.hexGridDist / std::sqrt(3.0);
    }
    if (params.nMoves <= 0) {
        params.nMoves = std::max<int32_t>(
            static_cast<int32_t>(std::ceil(params.hexGridDist / params.anchorDist)), 1);
    } else {
        params.anchorDist = params.hexGridDist / static_cast<double>(params.nMoves);
    }
    if (params.nMoves <= 1) {
        error("%s: thin 3D probe requires --n-moves > 1", __func__);
    }
    if (params.radius <= 0.0) {
        const double zReach = thin3d_default_zreach(
            params.thin3DZLevels, 1, params.zMin, params.zMax);
        params.radius = std::sqrt(params.anchorDist * params.anchorDist + zReach * zReach) * 1.2;
    }
    if (!(params.xmax > params.xmin)) {
        params.xmax = params.xmin + 4.0 * params.hexGridDist;
    }
    if (!(params.ymax > params.ymin)) {
        params.ymax = params.ymin + 4.0 * params.hexGridDist;
    }
    return params;
}

void write_dummy_tile_files(const ScopedTempDir& tmpDir, std::string& tsvFile, std::string& indexFile) {
    tsvFile = (tmpDir.path / "dummy.tsv").string();
    indexFile = (tmpDir.path / "dummy.index").string();

    {
        std::ofstream out(tsvFile);
        if (!out.is_open()) {
            error("%s: failed to create %s", __func__, tsvFile.c_str());
        }
        out << "# thin3d probe\n";
    }
    {
        std::ofstream out(indexFile);
        if (!out.is_open()) {
            error("%s: failed to create %s", __func__, indexFile.c_str());
        }
        out << "# tilesize 1\n";
        out << "0 0 0 0\n";
    }
}

std::vector<RawPoint> generate_raw_points(const ProbeParams& params) {
    std::vector<RawPoint> points;
    const double stepXY = params.pixelResolution / params.pixelDensity;
    const double stepZ = params.pixelResolutionZ / params.pixelDensity;
    if (stepXY <= 0.0 || stepZ <= 0.0) {
        error("%s: invalid synthetic sampling step", __func__);
    }

    for (double z = params.zMin + 0.5 * stepZ; z < params.zMax; z += stepZ) {
        for (double y = params.ymin + 0.5 * stepXY; y < params.ymax; y += stepXY) {
            for (double x = params.xmin + 0.5 * stepXY; x < params.xmax; x += stepXY) {
                points.push_back(RawPoint{x, y, z, 0u, 1u});
            }
        }
    }
    return points;
}

void populate_tile_data(const std::vector<RawPoint>& points, TileData<float>& tileData, const ProbeParams& params) {
    tileData.clear();
    tileData.xmin = static_cast<float>(params.xmin);
    tileData.xmax = static_cast<float>(params.xmax);
    tileData.ymin = static_cast<float>(params.ymin);
    tileData.ymax = static_cast<float>(params.ymax);
    auto& input = tileData.emplaceStandard3D();
    input.pts3d.reserve(points.size());
    for (const auto& pt : points) {
        input.pts3d.push_back(RecordT3D<float>{
            static_cast<float>(pt.x),
            static_cast<float>(pt.y),
            static_cast<float>(pt.z),
            pt.feature,
            pt.count
        });
    }
}

std::vector<InitCandidateRow> collect_init_candidates(const Thin3DProbeHarness& harness,
    const std::vector<RawPoint>& rawPoints, const HexGrid& hexGrid,
    int32_t nMoves, double supportRadius) {
    std::vector<InitCandidateRow> rows;
    for (size_t i = 0; i < rawPoints.size(); ++i) {
        const RawPoint& pt = rawPoints[i];
        harness.forEachThin3DAnchorWithinRadiusPublic(
            static_cast<float>(pt.x), static_cast<float>(pt.y), static_cast<float>(pt.z),
            hexGrid, nMoves, supportRadius, [&](float ax, float ay, float az) {
            rows.push_back(InitCandidateRow{
                i, pt.x, pt.y, pt.z, pt.feature, static_cast<double>(pt.count), ax, ay, az
            });
        });
    }
    return rows;
}

void write_input_points_tsv(const std::string& file, const std::vector<RawPoint>& points) {
    std::ofstream out(file);
    if (!out.is_open()) {
        error("%s: failed to open %s", __func__, file.c_str());
    }
    out << "point_index\tx\ty\tz\tfeature\tcount\n";
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < points.size(); ++i) {
        const auto& pt = points[i];
        out << i << '\t'
            << pt.x << '\t' << pt.y << '\t' << pt.z << '\t'
            << pt.feature << '\t' << pt.count << '\n';
    }
}

void write_init_candidates_tsv(const std::string& file, const std::vector<InitCandidateRow>& rows) {
    std::ofstream out(file);
    if (!out.is_open()) {
        error("%s: failed to open %s", __func__, file.c_str());
    }
    out << "point_index\tpx\tpy\tpz\tfeature\tcount\tax\tay\taz\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.pointIndex << '\t'
            << row.px << '\t' << row.py << '\t' << row.pz << '\t'
            << row.feature << '\t' << row.count << '\t'
            << row.ax << '\t' << row.ay << '\t' << row.az << '\n';
    }
}

void write_anchor_tsv(const std::string& file,
    const std::vector<AnchorPoint>& anchors, const std::vector<SparseObs>& docs) {
    std::ofstream out(file);
    if (!out.is_open()) {
        error("%s: failed to open %s", __func__, file.c_str());
    }
    out << "anchor_index\tx\ty\tz\ttotal_count\n";
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < anchors.size(); ++i) {
        const double total = (i < docs.size()) ? static_cast<double>(docs[i].ct_tot) : 0.0;
        out << i << '\t'
            << anchors[i].x << '\t' << anchors[i].y << '\t' << anchors[i].z << '\t'
            << total << '\n';
    }
}

std::vector<PixelAssignmentRow> collect_pixel_assignments(
    const TileData<float>& tileData, const std::vector<AnchorPoint>& anchors,
    const Minibatch& minibatch, const ProbeParams& params) {
    std::vector<PixelAssignmentRow> rows;
    const double res = params.pixelResolution;
    const double resZ = params.pixelResolutionZ;
    for (int row = 0; row < minibatch.wij.outerSize(); ++row) {
        if (row >= static_cast<int>(tileData.coords3d.size())) {
            break;
        }
        const auto& coord = tileData.coords3d[static_cast<size_t>(row)];
        const double px = static_cast<double>(coord.x) * res;
        const double py = static_cast<double>(coord.y) * res;
        const double pz = static_cast<double>(coord.z) * resZ;

        double pixelCount = 0.0;
        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator itM(minibatch.mtx, row); itM; ++itM) {
            pixelCount += static_cast<double>(itM.value());
        }

        for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(minibatch.wij, row); it; ++it) {
            const int anchorIndex = it.col();
            if (anchorIndex < 0 || anchorIndex >= static_cast<int>(anchors.size())) {
                continue;
            }
            const auto& anchor = anchors[static_cast<size_t>(anchorIndex)];
            rows.push_back(PixelAssignmentRow{
                static_cast<size_t>(row),
                coord.x, coord.y, coord.z,
                px, py, pz,
                pixelCount,
                static_cast<size_t>(anchorIndex),
                anchor.x, anchor.y, anchor.z,
                static_cast<double>(it.value())
            });
        }
    }
    return rows;
}

void write_pixel_tsv(const std::string& file, const std::vector<PixelAssignmentRow>& rows) {
    std::ofstream out(file);
    if (!out.is_open()) {
        error("%s: failed to open %s", __func__, file.c_str());
    }
    out << "pixel_index\tpx_index\tpy_index\tpz_index\tpx\tpy\tpz\tpixel_count\tanchor_index\tax\tay\taz\tweight\n";
    out << std::fixed << std::setprecision(6);
    for (const auto& row : rows) {
        out << row.pixelIndex << '\t'
            << row.pxIndex << '\t' << row.pyIndex << '\t' << row.pzIndex << '\t'
            << row.px << '\t' << row.py << '\t' << row.pz << '\t'
            << row.pixelCount << '\t'
            << row.anchorIndex << '\t'
            << row.ax << '\t' << row.ay << '\t' << row.az << '\t'
            << row.weight << '\n';
    }
}

void write_metadata_json(const std::string& file, const ProbeParams& params,
    const std::vector<float>& thin3DZLevels, double distNu,
    size_t rawPointCount, size_t anchorCount, size_t uniquePixelCount, double avgDegree) {
    json meta;
    meta["hex_size"] = params.hexSize;
    meta["hex_grid_dist"] = params.hexGridDist;
    meta["anchor_dist"] = params.anchorDist;
    meta["radius"] = params.radius;
    meta["half_life_dist"] = params.halfLifeDist;
    meta["dist_nu"] = distNu;
    meta["zmin"] = params.zMin;
    meta["zmax"] = params.zMax;
    meta["thin_3d_z_levels"] = thin3DZLevels;
    meta["n_moves"] = params.nMoves;
    meta["thin_3d_init_uses_radius"] = true;
    meta["pixel_density"] = params.pixelDensity;
    meta["pixel_resolution"] = params.pixelResolution;
    meta["pixel_resolution_z"] = params.pixelResolutionZ;
    meta["xmin"] = params.xmin;
    meta["xmax"] = params.xmax;
    meta["ymin"] = params.ymin;
    meta["ymax"] = params.ymax;
    meta["min_init_count"] = params.minInitCount;
    meta["ignore_outside_zrange"] = params.ignoreOutsideZrange;
    meta["raw_point_count"] = rawPointCount;
    meta["anchor_count"] = anchorCount;
    meta["unique_pixel_count"] = uniquePixelCount;
    meta["avg_degree"] = avgDegree;

    std::ofstream out(file);
    if (!out.is_open()) {
        error("%s: failed to open %s", __func__, file.c_str());
    }
    out << meta.dump(2) << '\n';
}

std::string relation_anchor_key(double x, double y, double z) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << x << '|' << y << '|' << z;
    return ss.str();
}

json build_init_anchor_json(const std::vector<InitCandidateRow>& rows) {
    json anchorsJson = json::array();
    std::set<std::string> seen;
    for (const auto& row : rows) {
        const std::string key = relation_anchor_key(row.ax, row.ay, row.az);
        if (!seen.insert(key).second) {
            continue;
        }
        std::ostringstream label;
        label << "Init anchor @ z=" << std::fixed << std::setprecision(2) << row.az;
        anchorsJson.push_back({
            {"anchor_key", key},
            {"label", label.str()},
            {"x", row.ax},
            {"y", row.ay},
            {"z", row.az}
        });
    }
    return anchorsJson;
}

json build_init_relation_json(const std::vector<InitCandidateRow>& rows) {
    json relationsJson = json::array();
    for (const auto& row : rows) {
        relationsJson.push_back({
            {"point_index", row.pointIndex},
            {"px", row.px},
            {"py", row.py},
            {"pz", row.pz},
            {"anchor_key", relation_anchor_key(row.ax, row.ay, row.az)}
        });
    }
    return relationsJson;
}

json build_iter_anchor_json(const std::vector<AnchorPoint>& anchors) {
    json anchorsJson = json::array();
    for (size_t i = 0; i < anchors.size(); ++i) {
        std::ostringstream label;
        label << "Anchor " << i << " @ z=" << std::fixed << std::setprecision(2) << anchors[i].z;
        anchorsJson.push_back({
            {"anchor_key", std::to_string(i)},
            {"label", label.str()},
            {"x", anchors[i].x},
            {"y", anchors[i].y},
            {"z", anchors[i].z}
        });
    }
    return anchorsJson;
}

json build_iter_relation_json(const std::vector<PixelAssignmentRow>& rows) {
    json relationsJson = json::array();
    for (const auto& row : rows) {
        relationsJson.push_back({
            {"point_index", row.pixelIndex},
            {"px", row.px},
            {"py", row.py},
            {"pz", row.pz},
            {"anchor_key", std::to_string(row.anchorIndex)}
        });
    }
    return relationsJson;
}

void write_relation_plotly_html(const std::string& file, const std::string& title,
    const json& anchorsJson, const json& relationsJson, const std::string& pointLabel) {
    if (anchorsJson.empty()) {
        return;
    }
    std::ofstream out(file);
    if (!out.is_open()) {
        error("%s: failed to open %s", __func__, file.c_str());
    }

    out << "<!doctype html>\n<html><head><meta charset=\"utf-8\">"
        << "<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>"
        << "<title>" << title << "</title></head><body>\n"
        << "<div id=\"plot\" style=\"width:100%;height:92vh;\"></div>\n"
        << "<script>\n"
        << "const anchors = " << anchorsJson.dump() << ";\n"
        << "const relations = " << relationsJson.dump() << ";\n"
        << "const colors = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];\n"
        << "const xs = anchors.map(a => a.x);\n"
        << "const ys = anchors.map(a => a.y);\n"
        << "const xmin = Math.min(...xs), xmax = Math.max(...xs);\n"
        << "const ymin = Math.min(...ys), ymax = Math.max(...ys);\n"
        << "const xMargin = Math.max((xmax - xmin) * 0.12, 1e-6);\n"
        << "const yMargin = Math.max((ymax - ymin) * 0.12, 1e-6);\n"
        << "const interior = anchors.filter(a => a.x > xmin + xMargin && a.x < xmax - xMargin && a.y > ymin + yMargin && a.y < ymax - yMargin);\n"
        << "const candidateAnchors = interior.length ? interior : anchors;\n"
        << "const shuffled = candidateAnchors.slice();\n"
        << "for (let i = shuffled.length - 1; i > 0; --i) {\n"
        << "  const j = Math.floor(Math.random() * (i + 1));\n"
        << "  const tmp = shuffled[i]; shuffled[i] = shuffled[j]; shuffled[j] = tmp;\n"
        << "}\n"
        << "const zPlanes = [...new Set(shuffled.map(a => a.z.toFixed(6)))];\n"
        << "const nXBins = 3, nYBins = 3;\n"
        << "const usedBins = new Set();\n"
        << "const usedPlanes = new Set();\n"
        << "const selected = [];\n"
        << "for (const a of shuffled) {\n"
        << "  const planeKey = a.z.toFixed(6);\n"
        << "  const xBin = Math.max(0, Math.min(nXBins - 1, Math.floor(((a.x - xmin) / Math.max(xmax - xmin, 1e-6)) * nXBins)));\n"
        << "  const yBin = Math.max(0, Math.min(nYBins - 1, Math.floor(((a.y - ymin) / Math.max(ymax - ymin, 1e-6)) * nYBins)));\n"
        << "  const binKey = `${xBin}:${yBin}`;\n"
        << "  if (usedPlanes.has(planeKey) || usedBins.has(binKey)) continue;\n"
        << "  selected.push(a);\n"
        << "  usedPlanes.add(planeKey);\n"
        << "  usedBins.add(binKey);\n"
        << "  if (selected.length >= Math.min(9, zPlanes.length)) break;\n"
        << "}\n"
        << "if (!selected.length) {\n"
        << "  for (let i = 0; i < Math.min(6, shuffled.length); ++i) selected.push(shuffled[i]);\n"
        << "}\n"
        << "const traces = [{\n"
        << "  type: 'scatter3d', mode: 'markers', name: 'All anchors',\n"
        << "  x: anchors.map(a => a.x), y: anchors.map(a => a.y), z: anchors.map(a => a.z),\n"
        << "  marker: {size: 4, color: 'rgba(0,0,0,0.45)', symbol: 'circle'}\n"
        << "}, {\n"
        << "  type: 'scatter3d', mode: 'markers', name: 'All " << pointLabel << "', showlegend: false,\n"
        << "  x: relations.map(r => r.px), y: relations.map(r => r.py), z: relations.map(r => r.pz),\n"
        << "  marker: {size: 2, color: 'rgba(150,150,150,0.18)', symbol: 'circle'}\n"
        << "}];\n"
        << "selected.forEach((anchor, idx) => {\n"
        << "  const related = relations.filter(r => r.anchor_key === anchor.anchor_key);\n"
        << "  if (!related.length) return;\n"
        << "  const color = colors[idx % colors.length];\n"
        << "  traces.push({type:'scatter3d', mode:'markers', name: anchor.label,\n"
        << "    x: related.map(r => r.px), y: related.map(r => r.py), z: related.map(r => r.pz),\n"
        << "    marker:{size:3, color:color, opacity:0.70}});\n"
        << "  traces.push({type:'scatter3d', mode:'markers', showlegend:false,\n"
        << "    x:[anchor.x], y:[anchor.y], z:[anchor.z], marker:{size:11, color:'black', symbol:'diamond', line:{color:'black', width:1}}});\n"
        << "});\n"
        << "Plotly.newPlot('plot', traces, {title:" << json(title).dump()
        << ", scene:{"
        << "xaxis:{title:'X', showgrid:false, showbackground:true, backgroundcolor:'rgba(245,245,245,0.75)', zeroline:false},"
        << "yaxis:{title:'Y', showgrid:false, showbackground:true, backgroundcolor:'rgba(245,245,245,0.75)', zeroline:false},"
        << "zaxis:{title:'Z', showgrid:false, showbackground:true, backgroundcolor:'rgba(245,245,245,0.75)', zeroline:false},"
        << "aspectmode:'data'}, margin:{l:0,r:0,b:0,t:40}});\n"
        << "</script></body></html>\n";
}

} // namespace

int32_t test(int32_t argc, char** argv) {
    try {
        ProbeParams params = parse_params(argc, argv);
        ensure_parent_dir(params.outPrefix);

        const double distNu = compute_dist_nu(params.halfLifeDist);
        ScopedTempDir tmpDir(std::filesystem::temp_directory_path());
        std::string dummyTsv;
        std::string dummyIndex;
        write_dummy_tile_files(tmpDir, dummyTsv, dummyIndex);

        lineParserUnival parser(0, 1, 3, 4);
        parser.setZ(2);
        parser.setFeatureDict(std::vector<std::string>{"synthetic_feature"});

        TileReader tileReader(dummyTsv, dummyIndex, nullptr, 1, false);
        Thin3DProbeHarness harness(tileReader, parser, params.pixelResolution);
        harness.setFeatureNames(std::vector<std::string>{"synthetic_feature"});
        harness.set3Dparameters(true, params.zMin, params.zMax,
            static_cast<float>(params.pixelResolutionZ),
            params.ignoreOutsideZrange, -1.0f, params.thin3DZLevels);

        const HexGrid hexGrid(params.hexSize);
        const std::vector<RawPoint> rawPoints = generate_raw_points(params);
        TileData<float> tileData;
        populate_tile_data(rawPoints, tileData, params);

        const std::vector<InitCandidateRow> initCandidates =
            collect_init_candidates(harness, rawPoints, hexGrid, params.nMoves, params.radius);

        std::vector<AnchorPoint> anchors;
        std::vector<SparseObs> documents;
        harness.buildAnchorsThin3DPublic(
            tileData, anchors, documents, hexGrid, params.nMoves, params.minInitCount,
            params.radius, distNu);

        Minibatch minibatch;
        minibatch.n = static_cast<int>(anchors.size());
        const double avgDegree = harness.buildMinibatchThin3DPublic(
            tileData, anchors, minibatch, params.radius, distNu);
        const std::vector<PixelAssignmentRow> pixelRows =
            collect_pixel_assignments(tileData, anchors, minibatch, params);

        const std::string inputFile = params.outPrefix + ".input_points.tsv";
        const std::string initFile = params.outPrefix + ".init_candidates.tsv";
        const std::string anchorFile = params.outPrefix + ".anchors.tsv";
        const std::string pixelFile = params.outPrefix + ".pixels.tsv";
        const std::string metaFile = params.outPrefix + ".meta.json";
        const std::string initHtmlFile = params.outPrefix + ".init.html";
        const std::string iterHtmlFile = params.outPrefix + ".iter.html";

        write_input_points_tsv(inputFile, rawPoints);
        write_init_candidates_tsv(initFile, initCandidates);
        write_anchor_tsv(anchorFile, anchors, documents);
        write_pixel_tsv(pixelFile, pixelRows);
        write_metadata_json(metaFile, params, params.thin3DZLevels, distNu,
            rawPoints.size(), anchors.size(),
            tileData.coords3d.size(), avgDegree);
        if (params.writeHtml) {
            write_relation_plotly_html(initHtmlFile,
                "Thin-3D Probe: Initialization Relations",
                build_init_anchor_json(initCandidates),
                build_init_relation_json(initCandidates),
                "init points");
            write_relation_plotly_html(iterHtmlFile,
                "Thin-3D Probe: Iteration Connectivity",
                build_iter_anchor_json(anchors),
                build_iter_relation_json(pixelRows),
                "connected pixels");
        }

        std::cout << "Thin-3D probe complete\n";
        std::cout << "Raw points: " << rawPoints.size() << "\n";
        std::cout << "Initial candidate edges: " << initCandidates.size() << "\n";
        std::cout << "Retained anchors: " << anchors.size() << "\n";
        std::cout << "Unique pixels: " << tileData.coords3d.size() << "\n";
        std::cout << "Pixel-anchor edges: " << pixelRows.size() << "\n";
        std::cout << "Average anchors per pixel: " << avgDegree << "\n";
        std::cout << "Wrote: " << inputFile << "\n";
        std::cout << "Wrote: " << initFile << "\n";
        std::cout << "Wrote: " << anchorFile << "\n";
        std::cout << "Wrote: " << pixelFile << "\n";
        std::cout << "Wrote: " << metaFile << "\n";
        if (params.writeHtml) {
            std::cout << "Wrote: " << initHtmlFile << "\n";
            std::cout << "Wrote: " << iterHtmlFile << "\n";
        }
    } catch (const std::exception& ex) {
        std::cerr << "Thin-3D probe failed: " << ex.what() << "\n";
        return 1;
    }
    return 0;
}
