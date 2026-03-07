#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "clipper2/clipper.h"
#include "img_utils.hpp"
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

PreparedRegionMask2D prepareRegionFromPaths(const Clipper2Lib::Paths64& paths,
                                        int32_t tileSize,
                                        int64_t scale = 10);

PreparedRegionMask2D loadPreparedRegionGeoJSON(const std::string& geojsonFile,
                                           int32_t tileSize,
                                           int64_t scale = 10);

PreparedRegionRasterMask2D prepareRegionRasterMask2D(const PreparedRegionMask2D& region,
                                                     float pixelResolution,
                                                     double delta = -1.0);
