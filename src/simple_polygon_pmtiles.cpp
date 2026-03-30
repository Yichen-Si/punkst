#include "simple_polygon_pmtiles.hpp"

#include "clipper2/clipper.h"
#include "pmtiles_utils.hpp"
#include "utils.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace simple_polygon_pmtiles {

namespace {

using Clipper2Lib::Area;
using Clipper2Lib::Path64;
using Clipper2Lib::Paths64;
using Clipper2Lib::Point64;
using Clipper2Lib::Rect64;
using Clipper2Lib::RectClip;

void ensure_tile_columns(mlt_pmtiles::PolygonTileData& tile,
    const mlt_pmtiles::FeatureTableSchema& schema) {
    if (!tile.columns.empty()) {
        return;
    }
    tile.ringOffsets.push_back(0u);
    tile.columns.reserve(schema.columns.size());
    for (const auto& columnSchema : schema.columns) {
        tile.columns.emplace_back(columnSchema.type, columnSchema.nullable);
    }
}

int64_t to_scaled_world(double value, int64_t clipScale) {
    return static_cast<int64_t>(std::llround(value * static_cast<double>(clipScale)));
}

double from_scaled_world(int64_t value, int64_t clipScale) {
    return static_cast<double>(value) / static_cast<double>(clipScale);
}

double tile_buffer_world_units(const SingleZoomPolygonWriterOptions& options) {
    const double localUnitsPerPixel = static_cast<double>(options.extent) / 256.0;
    return options.tileBufferPixels * localUnitsPerPixel *
        mlt_pmtiles::epsg3857_scale_factor(options.zoom);
}

void append_property_row(const mlt_pmtiles::FeatureTableSchema& schema,
    mlt_pmtiles::PolygonTileData& tile,
    const PolygonFeatureProperties& properties) {
    size_t intIdx = 0;
    size_t floatIdx = 0;
    size_t stringIdx = 0;
    for (size_t colIdx = 0; colIdx < schema.columns.size(); ++colIdx) {
        const auto& colSchema = schema.columns[colIdx];
        auto& col = tile.columns[colIdx];
        switch (colSchema.type) {
        case mlt_pmtiles::ScalarType::INT_32: {
            if (intIdx >= properties.intValues.size()) {
                error("%s: integer property count mismatch", __func__);
            }
            const auto& value = properties.intValues[intIdx++];
            if (colSchema.nullable) {
                col.present.push_back(value.has_value());
            }
            col.intValues.push_back(value.value_or(0));
            break;
        }
        case mlt_pmtiles::ScalarType::FLOAT: {
            if (floatIdx >= properties.floatValues.size()) {
                error("%s: float property count mismatch", __func__);
            }
            const auto& value = properties.floatValues[floatIdx++];
            if (colSchema.nullable) {
                col.present.push_back(value.has_value());
            }
            col.floatValues.push_back(value.value_or(0.0f));
            break;
        }
        case mlt_pmtiles::ScalarType::STRING: {
            if (stringIdx >= properties.stringValues.size()) {
                error("%s: string property count mismatch", __func__);
            }
            const auto& value = properties.stringValues[stringIdx++];
            if (colSchema.nullable) {
                col.present.push_back(value.has_value());
            }
            col.stringValues.push_back(value.value_or(std::string()));
            break;
        }
        case mlt_pmtiles::ScalarType::BOOLEAN:
        default:
            error("%s: unsupported polygon property scalar type", __func__);
        }
    }
    if (intIdx != properties.intValues.size() ||
        floatIdx != properties.floatValues.size() ||
        stringIdx != properties.stringValues.size()) {
        error("%s: polygon property vector count mismatch", __func__);
    }
}

double signed_area_local(const std::vector<int32_t>& xs,
    const std::vector<int32_t>& ys,
    uint32_t beg, uint32_t end) {
    double twiceArea = 0.0;
    for (uint32_t i = beg; i < end; ++i) {
        const uint32_t j = (i + 1u == end) ? beg : (i + 1u);
        twiceArea += static_cast<double>(xs[i]) * static_cast<double>(ys[j]) -
            static_cast<double>(ys[i]) * static_cast<double>(xs[j]);
    }
    return twiceArea * 0.5;
}

std::vector<std::pair<int32_t, int32_t>> convert_clipped_ring_to_local(
    const Path64& clipped,
    int64_t tileX, int64_t tileY,
    const SingleZoomPolygonWriterOptions& options) {
    double tileMinX = 0.0;
    double tileMaxY = 0.0;
    double tileMaxX = 0.0;
    double tileMinY = 0.0;
    mlt_pmtiles::tilecoord_to_epsg3857(tileX, tileY, 0.0, 0.0, options.zoom, tileMinX, tileMaxY);
    mlt_pmtiles::tilecoord_to_epsg3857(tileX, tileY, 256.0, 256.0, options.zoom, tileMaxX, tileMinY);
    const double tileWidth = tileMaxX - tileMinX;
    const double tileHeight = tileMaxY - tileMinY;

    std::vector<std::pair<int32_t, int32_t>> out;
    out.reserve(clipped.size());
    for (const auto& pt : clipped) {
        const double x = from_scaled_world(pt.x, options.clipScale);
        const double y = from_scaled_world(pt.y, options.clipScale);
        const int32_t localX = static_cast<int32_t>(std::llround(
            (x - tileMinX) * static_cast<double>(options.extent) / tileWidth));
        const int32_t localY = static_cast<int32_t>(std::llround(
            (tileMaxY - y) * static_cast<double>(options.extent) / tileHeight));
        if (!out.empty() && out.back().first == localX && out.back().second == localY) {
            continue;
        }
        out.emplace_back(localX, localY);
    }
    if (out.size() >= 2 && out.front() == out.back()) {
        out.pop_back();
    }
    return out;
}

void update_geo_bounds(const std::vector<std::pair<double, double>>& outerRing,
    PolygonWriteSummary& summary) {
    for (const auto& pt : outerRing) {
        summary.geoMinX = std::min(summary.geoMinX, pt.first);
        summary.geoMinY = std::min(summary.geoMinY, pt.second);
        summary.geoMaxX = std::max(summary.geoMaxX, pt.first);
        summary.geoMaxY = std::max(summary.geoMaxY, pt.second);
    }
}

void append_local_polygon_fragment(mlt_pmtiles::PolygonTileData& tile,
    const std::vector<std::pair<int32_t, int32_t>>& localRing,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const PolygonFeatureProperties& properties) {
    if (localRing.size() < 3) {
        return;
    }
    ensure_tile_columns(tile, schema);
    const uint32_t beg = static_cast<uint32_t>(tile.localX.size());
    for (const auto& pt : localRing) {
        tile.localX.push_back(pt.first);
        tile.localY.push_back(pt.second);
    }
    const uint32_t end = static_cast<uint32_t>(tile.localX.size());
    if (end - beg < 3u) {
        tile.localX.resize(beg);
        tile.localY.resize(beg);
        return;
    }
    if (signed_area_local(tile.localX, tile.localY, beg, end) < 0.0) {
        std::reverse(tile.localX.begin() + beg, tile.localX.begin() + end);
        std::reverse(tile.localY.begin() + beg, tile.localY.begin() + end);
    }
    tile.ringOffsets.push_back(end);
    append_property_row(schema, tile, properties);
}

mlt_pmtiles::EncodedTilePayload encode_polygon_tile_payload(const TileKey& tileKey,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile,
    const mlt_pmtiles::GlobalStringDictionary* stringDictionary,
    uint8_t zoom) {
    const std::string raw = mlt_pmtiles::encode_polygon_tile(schema, tile, stringDictionary);
    mlt_pmtiles::EncodedTilePayload payload;
    payload.tileId = pmtiles::zxy_to_tileid(zoom, static_cast<uint32_t>(tileKey.col), static_cast<uint32_t>(tileKey.row));
    payload.z = zoom;
    payload.x = static_cast<uint32_t>(tileKey.col);
    payload.y = static_cast<uint32_t>(tileKey.row);
    payload.featureCount = static_cast<uint32_t>(tile.size());
    payload.compressedData = mlt_pmtiles::gzip_compress(raw);
    return payload;
}

} // namespace

void append_simple_polygon_feature(std::map<TileKey, mlt_pmtiles::PolygonTileData>& tileMap,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const std::vector<std::pair<double, double>>& outerRing,
    const PolygonFeatureProperties& properties,
    const SingleZoomPolygonWriterOptions& options,
    PolygonWriteSummary& summary) {
    if (outerRing.size() < 3) {
        return;
    }
    if (options.extent == 0) {
        error("%s: extent must be positive", __func__);
    }
    if (options.clipScale <= 0) {
        error("%s: clipScale must be positive", __func__);
    }

    update_geo_bounds(outerRing, summary);
    ++summary.featureCount;

    Path64 source;
    source.reserve(outerRing.size());
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (const auto& pt : outerRing) {
        minX = std::min(minX, pt.first);
        minY = std::min(minY, pt.second);
        maxX = std::max(maxX, pt.first);
        maxY = std::max(maxY, pt.second);
        source.push_back(Point64(to_scaled_world(pt.first, options.clipScale),
            to_scaled_world(pt.second, options.clipScale)));
    }
    if (source.size() < 3 || std::llround(std::abs(Area(source))) == 0) {
        return;
    }

    const double bufferWorld = tile_buffer_world_units(options);
    const double minXBuffered = minX - bufferWorld;
    const double minYBuffered = minY - bufferWorld;
    const double maxXBuffered = maxX + bufferWorld;
    const double maxYBuffered = maxY + bufferWorld;

    int64_t tileX0 = 0, tileY0 = 0, tileX1 = 0, tileY1 = 0;
    double localXDummy = 0.0, localYDummy = 0.0;
    mlt_pmtiles::epsg3857_to_tilecoord(minXBuffered, maxYBuffered, options.zoom, tileX0, tileY0, localXDummy, localYDummy);
    mlt_pmtiles::epsg3857_to_tilecoord(maxXBuffered, minYBuffered, options.zoom, tileX1, tileY1, localXDummy, localYDummy);
    const int64_t maxTileIndex = static_cast<int64_t>((uint64_t{1} << options.zoom) - 1u);
    tileX0 = std::clamp(tileX0, int64_t{0}, maxTileIndex);
    tileX1 = std::clamp(tileX1, int64_t{0}, maxTileIndex);
    tileY0 = std::clamp(tileY0, int64_t{0}, maxTileIndex);
    tileY1 = std::clamp(tileY1, int64_t{0}, maxTileIndex);

    for (int64_t tileY = tileY0; tileY <= tileY1; ++tileY) {
        for (int64_t tileX = tileX0; tileX <= tileX1; ++tileX) {
            double worldMinX = 0.0;
            double worldMaxY = 0.0;
            double worldMaxX = 0.0;
            double worldMinY = 0.0;
            mlt_pmtiles::tilecoord_to_epsg3857(tileX, tileY, 0.0, 0.0, options.zoom, worldMinX, worldMaxY);
            mlt_pmtiles::tilecoord_to_epsg3857(tileX, tileY, 256.0, 256.0, options.zoom, worldMaxX, worldMinY);
            const Rect64 clipRect(
                to_scaled_world(worldMinX - bufferWorld, options.clipScale),
                to_scaled_world(worldMinY - bufferWorld, options.clipScale),
                to_scaled_world(worldMaxX + bufferWorld, options.clipScale),
                to_scaled_world(worldMaxY + bufferWorld, options.clipScale));
            const Paths64 clipped = RectClip(clipRect, Paths64{source});
            if (clipped.empty()) {
                continue;
            }
            TileKey key{static_cast<int32_t>(tileY), static_cast<int32_t>(tileX)};
            auto& outTile = tileMap[key];
            for (const auto& path : clipped) {
                const auto localRing = convert_clipped_ring_to_local(path, tileX, tileY, options);
                append_local_polygon_fragment(outTile, localRing, schema, properties);
            }
        }
    }
}

std::vector<mlt_pmtiles::EncodedTilePayload> encode_polygon_tile_map(
    std::map<TileKey, mlt_pmtiles::PolygonTileData>& tileMap,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::GlobalStringDictionary* stringDictionary,
    const SingleZoomPolygonWriterOptions& options) {
    struct OutputTile {
        TileKey key;
        const mlt_pmtiles::PolygonTileData* data = nullptr;
    };
    std::vector<OutputTile> outputs;
    outputs.reserve(tileMap.size());
    for (auto& kv : tileMap) {
        if (kv.second.size() == 0) {
            continue;
        }
        outputs.push_back(OutputTile{kv.first, &kv.second});
    }
    std::vector<mlt_pmtiles::EncodedTilePayload> encoded(outputs.size());
    if (options.threads > 1 && outputs.size() > 1) {
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
            static_cast<size_t>(options.threads));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, outputs.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    encoded[i] = encode_polygon_tile_payload(outputs[i].key, schema,
                        *outputs[i].data, stringDictionary, options.zoom);
                }
            });
    } else {
        for (size_t i = 0; i < outputs.size(); ++i) {
            encoded[i] = encode_polygon_tile_payload(outputs[i].key, schema,
                *outputs[i].data, stringDictionary, options.zoom);
        }
    }
    return encoded;
}

nlohmann::json build_simple_polygon_metadata(const std::string& sourceFamily,
    const std::vector<std::string>& canonicalIdFields,
    const SingleZoomPolygonWriterOptions& options,
    const nlohmann::json& sourceMetadata) {
    nlohmann::json out;
    out["polygon_topology"] = {
        {"simple_polygon", true},
        {"holes", false},
        {"multipolygon", false},
    };
    out["polygon_pyramid_hint"] = {
        {"strategy", "canonical_id_reclip"},
        {"buffer_screen_px", options.tileBufferPixels},
        {"source_family", sourceFamily},
        {"canonical_id_fields", canonicalIdFields},
    };
    out["polygon_source"] = {
        {"family", sourceFamily},
        {"reconstructible", true},
        {"canonical_id_fields", canonicalIdFields},
        {"parameters", sourceMetadata},
    };
    return out;
}

} // namespace simple_polygon_pmtiles
