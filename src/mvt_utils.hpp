#pragma once

#include "vector_tile_utils.hpp"

#include <cstddef>
#include <string>
#include <vector>

namespace mvt_pmtiles {

using pm_vector::DecodedPointTile;
using pm_vector::DecodedPolygonTile;
using pm_vector::FeatureTableSchema;
using pm_vector::GlobalStringDictionary;
using pm_vector::PointTileData;
using pm_vector::PolygonTileData;

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
std::string encode_polygon_tile_prefix(const FeatureTableSchema& schema,
    const PolygonTileData& tile,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary);
std::string encode_polygon_tile_subset(const FeatureTableSchema& schema,
    const PolygonTileData& tile,
    const std::vector<uint32_t>& order,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary);

DecodedPointTile decode_point_tile(const std::string& rawTile);
DecodedPolygonTile decode_polygon_tile(const std::string& rawTile);
uint64_t count_features(const std::string& rawTile);

} // namespace mvt_pmtiles
