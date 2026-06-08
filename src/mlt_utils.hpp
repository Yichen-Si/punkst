#pragma once

#include "vector_tile_utils.hpp"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlt_pmtiles {

using pm_vector::DecodedPointTile;
using pm_vector::DecodedPolygonTile;
using pm_vector::FeatureTableSchema;
using pm_vector::GlobalStringDictionary;
using pm_vector::PointTileData;
using pm_vector::PolygonTileData;
using pm_vector::ColumnSchema;
using pm_vector::PropertyColumn;
using pm_vector::ScalarType;
using pm_vector::encode_zigzag32;

std::string encode_bool_rle(const std::vector<bool>& present);
std::vector<bool> decode_bool_rle(const uint8_t* data, size_t len, size_t count);

std::string encode_point_tile(const FeatureTableSchema& schema,
    const PointTileData& tile,
    const GlobalStringDictionary* stringDictionary);
std::string encode_point_tile_prefix(const FeatureTableSchema& schema,
    const PointTileData& tile,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary);
std::string encode_point_tile_subset(const FeatureTableSchema& schema,
    const PointTileData& tile,
    const std::vector<uint32_t>& order,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary);
std::string encode_polygon_tile(const FeatureTableSchema& schema,
    const PolygonTileData& tile,
    const GlobalStringDictionary* stringDictionary);
// Output: one MLT tile for feature 0..rowCount-1 (for pyramid building)
std::string encode_polygon_tile_prefix(const FeatureTableSchema& schema,
    const PolygonTileData& tile,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary);
std::string encode_polygon_tile_subset(const FeatureTableSchema& schema,
    const PolygonTileData& tile,
    const std::vector<uint32_t>& order,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary);
std::string rewrite_point_tile_layer_name(const std::string& rawTile,
    const std::string& newLayerName);
DecodedPointTile decode_point_tile(const std::string& rawTile);
DecodedPolygonTile decode_polygon_tile(const std::string& rawTile);

} // namespace mlt_pmtiles
