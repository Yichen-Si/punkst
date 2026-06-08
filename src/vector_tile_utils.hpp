#pragma once

#include <cstddef>
#include <cstdint>
#include <mlt/metadata/tileset.hpp>
#include <string>
#include <vector>

namespace pm_vector {

using ScalarType = mlt::metadata::tileset::ScalarType;

struct ColumnSchema {
    std::string name;
    ScalarType type;
    bool nullable = false;
};

struct FeatureTableSchema {
    std::string layerName;
    uint32_t extent = 4096;
    bool hasIdColumn = false;
    bool idIsUint64 = false;
    std::vector<ColumnSchema> columns;
};

// This is not mapped to MLT schema but a temporary helper
struct GlobalStringDictionary {
    std::vector<std::string> values;
    const std::string& lookup(uint32_t code) const;
};

struct PropertyColumn {
    ScalarType type;
    bool nullable = false;
    std::vector<bool> present;
    std::vector<int32_t> length;
    std::vector<uint32_t> offset;
    std::vector<int32_t> intValues;
    std::vector<float> floatValues;
    std::vector<uint32_t> stringCodes;
    std::vector<std::string> stringValues;
    std::vector<bool> boolValues;
    PropertyColumn() = default;
    PropertyColumn(ScalarType _type, bool _nullable = false) : type(_type), nullable(_nullable) {}
};

struct PointTileData {
    std::vector<uint64_t> featureIds;
    std::vector<int32_t> localX;
    std::vector<int32_t> localY;
    std::vector<PropertyColumn> columns;

    size_t size() const { return localX.size(); }
};

struct DecodedPointTile {
    FeatureTableSchema schema;
    PointTileData tile;
};

struct PolygonTileData {
    // currently supports exactly one outer ring per feature
    // with no holes and no multipolygon
    // ringOffsets has size featureCount + 1 and stores half-open vertex ranges
    // figure i uses vertices in localX[ringOffsets[i] : ringOffsets[i+1]]
    std::vector<uint64_t> featureIds;
    std::vector<uint32_t> ringOffsets;
    std::vector<int32_t> localX;
    std::vector<int32_t> localY;
    std::vector<PropertyColumn> columns;

    size_t size() const { return ringOffsets.empty() ? 0u : ringOffsets.size() - 1u; }
    size_t vertexCount() const { return localX.size(); }
};

struct DecodedPolygonTile {
    FeatureTableSchema schema;
    PolygonTileData tile;
};

inline uint32_t encode_zigzag32(int32_t value) {
    return (static_cast<uint32_t>(value) << 1u) ^ static_cast<uint32_t>(value >> 31);
}

int32_t remap_child_local_to_parent_local(int32_t childLocal, uint32_t childIndex,
    uint32_t extent);
void append_child_row_to_parent_tile(const DecodedPointTile& child,
    size_t row,
    uint32_t childX,
    uint32_t childY,
    uint32_t parentX,
    uint32_t parentY,
    PointTileData& parentOut);

} // namespace pm_vector
