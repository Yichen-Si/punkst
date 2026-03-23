#include "mlt_encoder.hpp"

#include "mlt_pmtiles_utils.hpp"
#include "utils.h"

#include <cstring>

namespace mlt_pmtiles {

namespace {

inline void append_varint(std::string& out, uint64_t value) {
    do {
        uint8_t byte = static_cast<uint8_t>(value & 0x7F);
        value >>= 7;
        if (value > 0) {
            byte |= 0x80;
        }
        out.push_back(static_cast<char>(byte));
    } while (value > 0);
}

inline uint32_t encode_zigzag32(int32_t value) {
    return (static_cast<uint32_t>(value) << 1u) ^ static_cast<uint32_t>(value >> 31);
}

inline void validate_column_lengths(const PropertyColumn& column, size_t rowCount, const char* funcName) {
    if (!column.present.empty() && column.present.size() != rowCount) {
        error("%s: PRESENT stream length mismatch", funcName);
    }
    switch (column.type) {
    case ColumnType::Int32:
        if (column.intValues.size() != rowCount) {
            error("%s: INT column length mismatch", funcName);
        }
        break;
    case ColumnType::Float32:
        if (column.floatValues.size() != rowCount) {
            error("%s: FLOAT column length mismatch", funcName);
        }
        break;
    case ColumnType::StringDictionary:
        if (column.stringCodes.size() != rowCount) {
            error("%s: STRING column length mismatch", funcName);
        }
        break;
    }
}

} // namespace

const std::string& GlobalStringDictionary::lookup(uint32_t code) const {
    if (code >= values.size()) {
        error("%s: string dictionary code %u is out of range for dictionary size %zu",
            __func__, code, values.size());
    }
    return values[code];
}

std::string encode_point_tile(const Schema& schema,
    const PointTileData& tile,
    const GlobalStringDictionary* stringDictionary) {
    const size_t rowCount = tile.size();
    if (tile.localY.size() != rowCount) {
        error("%s: geometry column length mismatch", __func__);
    }
    if (schema.columns.size() != tile.columns.size()) {
        error("%s: schema/column count mismatch", __func__);
    }
    for (size_t i = 0; i < tile.columns.size(); ++i) {
        validate_column_lengths(tile.columns[i], rowCount, __func__);
    }

    std::string layer;
    append_varint(layer, schema.layerName.size());
    layer.append(schema.layerName);
    append_varint(layer, schema.extent);
    append_varint(layer, 1 + schema.columns.size());
    append_varint(layer, 4); // geometry column
    for (size_t i = 0; i < schema.columns.size(); ++i) {
        const auto& columnSchema = schema.columns[i];
        int typeCode = 0;
        switch (columnSchema.type) {
        case ColumnType::Int32:
            // The current pmpoint export path recognizes integer columns in the
            // 20..23 family. We use that compatible tag range for bounded k*
            // attributes while keeping the data stream as zigzag-varint int32.
            typeCode = columnSchema.nullable ? 21 : 20;
            break;
        case ColumnType::Float32:
            typeCode = columnSchema.nullable ? 25 : 24;
            break;
        case ColumnType::StringDictionary:
            typeCode = columnSchema.nullable ? 29 : 28;
            break;
        }
        append_varint(layer, static_cast<uint64_t>(typeCode));
        append_varint(layer, columnSchema.name.size());
        layer.append(columnSchema.name);
    }

    append_varint(layer, 2); // geometry stream count
    layer.push_back(0x10);
    layer.push_back(0x02);
    append_varint(layer, rowCount);
    append_varint(layer, rowCount);
    for (size_t i = 0; i < rowCount; ++i) {
        layer.push_back(0); // point geometry
    }

    std::string vertexData;
    for (size_t i = 0; i < rowCount; ++i) {
        append_varint(vertexData, encode_zigzag32(tile.localX[i]));
        append_varint(vertexData, encode_zigzag32(tile.localY[i]));
    }
    layer.push_back(0x13);
    layer.push_back(0x02);
    append_varint(layer, rowCount * 2);
    append_varint(layer, vertexData.size());
    layer.append(vertexData);

    for (size_t colIdx = 0; colIdx < schema.columns.size(); ++colIdx) {
        const auto& columnSchema = schema.columns[colIdx];
        const auto& column = tile.columns[colIdx];
        const bool nullable = columnSchema.nullable;

        std::vector<bool> present;
        present.reserve(rowCount);
        if (nullable) {
            if (!column.present.empty()) {
                present = column.present;
            } else {
                present.assign(rowCount, true);
            }
        }

        switch (columnSchema.type) {
        case ColumnType::Int32: {
            std::string intData;
            size_t nonNullCount = 0;
            for (size_t row = 0; row < rowCount; ++row) {
                if (nullable && !present[row]) {
                    continue;
                }
                ++nonNullCount;
                append_varint(intData, encode_zigzag32(column.intValues[row]));
            }
            if (nullable) {
                const std::string rle = encode_bool_rle(present);
                layer.push_back(0x00);
                layer.push_back(0x02);
                append_varint(layer, rowCount);
                append_varint(layer, rle.size());
                layer.append(rle);
            }
            layer.push_back(0x10);
            layer.push_back(0x02);
            append_varint(layer, nonNullCount);
            append_varint(layer, intData.size());
            layer.append(intData);
            break;
        }
        case ColumnType::Float32: {
            std::string floatData;
            size_t nonNullCount = 0;
            for (size_t row = 0; row < rowCount; ++row) {
                if (nullable && !present[row]) {
                    continue;
                }
                ++nonNullCount;
                uint32_t bits = 0;
                static_assert(sizeof(float) == sizeof(uint32_t));
                std::memcpy(&bits, &column.floatValues[row], sizeof(bits));
                floatData.push_back(static_cast<char>(bits & 0xFFu));
                floatData.push_back(static_cast<char>((bits >> 8u) & 0xFFu));
                floatData.push_back(static_cast<char>((bits >> 16u) & 0xFFu));
                floatData.push_back(static_cast<char>((bits >> 24u) & 0xFFu));
            }
            if (nullable) {
                const std::string rle = encode_bool_rle(present);
                layer.push_back(0x00);
                layer.push_back(0x02);
                append_varint(layer, rowCount);
                append_varint(layer, rle.size());
                layer.append(rle);
            }
            layer.push_back(0x10);
            layer.push_back(0x00);
            append_varint(layer, nonNullCount);
            append_varint(layer, floatData.size());
            layer.append(floatData);
            break;
        }
        case ColumnType::StringDictionary: {
            if (stringDictionary == nullptr) {
                error("%s: string dictionary is required for STRING columns", __func__);
            }
            std::string lengths;
            std::string bytes;
            size_t nonNullCount = 0;
            for (size_t row = 0; row < rowCount; ++row) {
                if (nullable && !present[row]) {
                    continue;
                }
                ++nonNullCount;
                const std::string& value = stringDictionary->lookup(column.stringCodes[row]);
                append_varint(lengths, value.size());
                bytes.append(value);
            }
            append_varint(layer, nullable ? 3 : 2);
            if (nullable) {
                const std::string rle = encode_bool_rle(present);
                layer.push_back(0x00);
                layer.push_back(0x02);
                append_varint(layer, rowCount);
                append_varint(layer, rle.size());
                layer.append(rle);
            }
            layer.push_back(0x30);
            layer.push_back(0x02);
            append_varint(layer, nonNullCount);
            append_varint(layer, lengths.size());
            layer.append(lengths);

            layer.push_back(0x10);
            layer.push_back(0x00);
            append_varint(layer, 0);
            append_varint(layer, bytes.size());
            layer.append(bytes);
            break;
        }
        }
    }

    std::string out;
    append_varint(out, 1 + layer.size());
    out.push_back(1);
    out.append(layer);
    return out;
}

} // namespace mlt_pmtiles
