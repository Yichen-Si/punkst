#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace mlt_pmtiles {

enum class ColumnType {
    Int32,
    Float32,
    StringDictionary,
};

struct ColumnSchema {
    std::string name;
    ColumnType type = ColumnType::Int32;
    bool nullable = false;
};

struct Schema {
    std::string layerName;
    uint32_t extent = 4096;
    std::vector<ColumnSchema> columns;
};

struct GlobalStringDictionary {
    std::vector<std::string> values;
    const std::string& lookup(uint32_t code) const;
};

struct PropertyColumn {
    ColumnType type = ColumnType::Int32;
    bool nullable = false;
    std::vector<bool> present;
    std::vector<int32_t> intValues;
    std::vector<float> floatValues;
    std::vector<uint32_t> stringCodes;
};

struct PointTileData {
    std::vector<int32_t> localX;
    std::vector<int32_t> localY;
    std::vector<PropertyColumn> columns;

    size_t size() const { return localX.size(); }
};

std::string encode_point_tile(const Schema& schema,
    const PointTileData& tile,
    const GlobalStringDictionary* stringDictionary);

} // namespace mlt_pmtiles
