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

enum class PolygonBoundaryMode {
    BufferClipDuplicate,
    NoClippingDuplicate,
    SingleTileNoDuplication,
};

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
    PolygonBoundaryMode boundaryMode = PolygonBoundaryMode::BufferClipDuplicate;
};

struct PolygonWriteSummary {
    uint64_t featureCount = 0;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
};

size_t append_simple_polygon_feature_to_tile(mlt_pmtiles::PolygonTileData& outTile,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const std::vector<std::pair<double, double>>& outerRing,
    uint64_t featureId,
    const PolygonFeatureProperties& properties,
    uint32_t tileX,
    uint32_t tileY,
    const SingleZoomPolygonWriterOptions& options);

size_t append_simple_polygon_global_feature_to_tile(mlt_pmtiles::PolygonTileData& outTile,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const std::vector<std::pair<int64_t, int64_t>>& globalRing,
    uint64_t featureId,
    const PolygonFeatureProperties& properties,
    uint32_t tileX,
    uint32_t tileY,
    uint8_t sourceZoom,
    const SingleZoomPolygonWriterOptions& options);

void append_simple_polygon_feature(std::map<TileKey, mlt_pmtiles::PolygonTileData>& tileMap,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const std::vector<std::pair<double, double>>& outerRing,
    uint64_t featureId,
    const PolygonFeatureProperties& properties,
    const SingleZoomPolygonWriterOptions& options,
    PolygonWriteSummary& summary,
    const std::optional<std::pair<double, double>>& assignmentCenter = std::nullopt);

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
