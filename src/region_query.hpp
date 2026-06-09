#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "clipper2/clipper.h"
#include "geometry_utils.hpp"
#include "json.hpp"
#include "tile_io.hpp"

enum class RegionTileState : uint8_t { Outside, Inside, Partial };
enum class RegionPixelState : uint8_t { Outside, Interior, Boundary };

struct RegionBox64 {
    int64_t xmin = 0;
    int64_t ymin = 0;
    int64_t xmax = -1;
    int64_t ymax = -1;

    bool valid() const {
        return xmin <= xmax && ymin <= ymax;
    }
};

struct PreparedRegionMask2D {
    int64_t scale = 10;
    int32_t tileSize = 0;
    Rectangle<float> bbox_f;
    Clipper2Lib::Paths64 union_paths;
    std::vector<RegionBox64> comp_bbox;
    std::unordered_map<TileKey, std::vector<uint32_t>, TileKeyHash> tile_bins;
    mutable std::unordered_map<TileKey, RegionTileState, TileKeyHash> tile_state_cache;

    bool empty() const { return union_paths.empty(); }

    bool containsPoint(float x, float y, const TileKey* tile_hint = nullptr) const;
    RegionTileState classifyTile(const TileKey& tile) const;
};

struct PreparedRegionRasterMask2D {
    PreparedRegionMask2D outer_mask;
    PreparedRegionMask2D inner_mask;
    float pixelResolution = 1.0f;
    double delta = 0.0;

    RegionPixelState classifyPixel(int32_t pixX, int32_t pixY,
                                   const TileKey* tile_hint = nullptr) const;
};

struct PreparedGeoJSONFeature2D {
    std::string id;
    PreparedRegionMask2D region;
    float x = 0.0f;
    float y = 0.0f;
};

struct SimplePolygonRingRecord {
    std::string polygonId;
    size_t partIndex = 0;
    uint32_t assignedId = 0;
    uint64_t featureId = 0;
    std::vector<std::pair<double, double>> ring;
    std::pair<double, double> center{0.0, 0.0};
};

struct SimplePolygonTableReadOptions {
    int32_t idCol = 0;
    int32_t xCol = 1;
    int32_t yCol = 2;
    int32_t orderCol = -1;
    bool requireConsecutiveIds = false;
    bool idIsU32 = false;
};

PreparedRegionMask2D prepareRegionFromPaths(const Clipper2Lib::Paths64& paths,
                                        int32_t tileSize,
                                        int64_t scale = 10);
PreparedRegionMask2D prepareRegionFromRectangle(const Rectangle<float>& rect,
                                                int32_t tileSize,
                                                int64_t scale = 10);
PreparedRegionMask2D prepareRegionFromGeoJSONGeometry(const nlohmann::json& geometry,
                                                      int32_t tileSize,
                                                      int64_t scale = 10);

PreparedRegionMask2D loadPreparedRegionGeoJSON(const std::string& geojsonFile,
                                           int32_t tileSize,
                                           int64_t scale = 10);

std::vector<PreparedGeoJSONFeature2D> loadPreparedGeoJSONFeatures(
    const std::string& geojsonFile,
    int32_t tileSize,
    int64_t scale = 10,
    const std::string& idProperty = "title");

std::pair<double, double> centroidForSimpleRing(
    const std::vector<std::pair<double, double>>& ring);
std::pair<double, double> centroidForSimpleRings(
    const std::vector<std::vector<std::pair<double, double>>>& rings);

std::vector<SimplePolygonRingRecord> readSimplePolygonGeoJSON(
    const std::string& geojsonFile,
    const std::string& idProperty = "cell_id",
    bool idIsU32 = false);
std::vector<SimplePolygonRingRecord> readSimplePolygonTable(
    const std::string& tableFile,
    const SimplePolygonTableReadOptions& options = {});
std::vector<SimplePolygonRingRecord> readSimplePolygonsAuto(
    const std::string& path,
    const std::string& format,
    const std::string& idProperty,
    const SimplePolygonTableReadOptions& tableOptions = {});

PreparedRegionRasterMask2D prepareRegionRasterMask2D(const PreparedRegionMask2D& region,
                                                     float pixelResolution,
                                                     double delta = -1.0);
