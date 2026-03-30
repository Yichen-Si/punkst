#pragma once

#include "json.hpp"
#include "mlt_utils.hpp"
#include "pmtiles_utils.hpp"
#include "tile_io.hpp"

#include <cstdint>
#include <limits>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace simple_polygon_pmtiles {

struct PolygonFeatureProperties {
    std::vector<std::optional<int32_t>> intValues;
    std::vector<std::optional<float>> floatValues;
    std::vector<std::optional<std::string>> stringValues;
};

struct SingleZoomPolygonWriterOptions {
    uint8_t zoom = 0;
    uint32_t extent = 4096;
    double coordScale = 1.0;
    double tileBufferPixels = 5.0;
    int64_t clipScale = 1024;
    int32_t threads = 1;
};

struct PolygonWriteSummary {
    uint64_t featureCount = 0;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
};

void append_simple_polygon_feature(std::map<TileKey, mlt_pmtiles::PolygonTileData>& tileMap,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const std::vector<std::pair<double, double>>& outerRing,
    const PolygonFeatureProperties& properties,
    const SingleZoomPolygonWriterOptions& options,
    PolygonWriteSummary& summary);

std::vector<mlt_pmtiles::EncodedTilePayload> encode_polygon_tile_map(
    std::map<TileKey, mlt_pmtiles::PolygonTileData>& tileMap,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::GlobalStringDictionary* stringDictionary,
    const SingleZoomPolygonWriterOptions& options);

nlohmann::json build_simple_polygon_metadata(const std::string& sourceFamily,
    const std::vector<std::string>& canonicalIdFields,
    const SingleZoomPolygonWriterOptions& options,
    const nlohmann::json& sourceMetadata);

} // namespace simple_polygon_pmtiles
