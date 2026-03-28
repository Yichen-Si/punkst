#include "mlt_utils.hpp"

#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mlt/metadata/type_map.hpp>
#include <zlib.h>

namespace mlt_pmtiles {

namespace {

constexpr double kEpsg3857Radius = 6378137.0;
constexpr double kEpsg3857Bound = 20037508.3428;

using SpecColumn = mlt::metadata::tileset::Column;
using SpecColumnScope = mlt::metadata::tileset::ColumnScope;
using SpecScalarColumn = mlt::metadata::tileset::ScalarColumn;
using SpecTypeMap = mlt::metadata::type_map::Tag0x01;

// pack uint64 to protobuf-style varint encoding
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

inline std::string encode_varint_bytes(uint64_t value) {
    std::string out;
    append_varint(out, value);
    return out;
}

inline uint32_t encode_zigzag32(int32_t value) {
    return (static_cast<uint32_t>(value) << 1u) ^ static_cast<uint32_t>(value >> 31);
}

inline int32_t decode_zigzag32(uint64_t value) {
    return static_cast<int32_t>((value >> 1u) ^ (~(value & 1u) + 1u));
}

uint64_t read_varint(const uint8_t*& ptr, const uint8_t* end, const char* funcName) {
    uint64_t value = 0;
    int shift = 0;
    while (ptr < end) {
        const uint8_t byte = *ptr++;
        value |= static_cast<uint64_t>(byte & 0x7Fu) << shift;
        if ((byte & 0x80u) == 0u) {
            return value;
        }
        shift += 7;
        if (shift >= 64) {
            error("%s: malformed varint", funcName);
        }
    }
    error("%s: truncated varint", funcName);
    return 0;
}

const uint8_t* checked_advance(const uint8_t* ptr, const uint8_t* end, uint64_t bytes, const char* funcName) {
    if (bytes > static_cast<uint64_t>(end - ptr)) {
        error("%s: truncated MLT payload", funcName);
    }
    return ptr + static_cast<size_t>(bytes);
}

inline void validate_column_lengths(const PropertyColumn& column, size_t rowCount, const char* funcName) {
    if (!column.present.empty() && column.present.size() != rowCount) {
        error("%s: PRESENT stream length mismatch", funcName);
    }
    switch (column.type) {
    case ScalarType::BOOLEAN:
        if (column.boolValues.size() != rowCount) {
            error("%s: BOOLEAN column length mismatch", funcName);
        }
        break;
    case ScalarType::INT_32:
        if (column.intValues.size() != rowCount) {
            error("%s: INT column length mismatch", funcName);
        }
        break;
    case ScalarType::FLOAT:
        if (column.floatValues.size() != rowCount) {
            error("%s: FLOAT column length mismatch", funcName);
        }
        break;
    case ScalarType::STRING:
        if (!column.stringCodes.empty() && !column.stringValues.empty()) {
            error("%s: STRING column cannot mix dictionary codes with plain strings", funcName);
        }
        if (!column.stringCodes.empty()) {
            if (column.stringCodes.size() != rowCount) {
                error("%s: STRING column length mismatch", funcName);
            }
        } else if (!column.stringValues.empty()) {
            if (column.stringValues.size() != rowCount) {
                error("%s: STRING column length mismatch", funcName);
            }
        } else {
            error("%s: STRING column length mismatch", funcName);
        }
        break;
    default:
        error("%s: unsupported scalar column type %d", funcName, static_cast<int>(column.type));
    }
}

SpecColumn make_spec_column(const ColumnSchema& columnSchema) {
    return SpecColumn{
        .name = columnSchema.name,
        .nullable = columnSchema.nullable,
        .columnScope = SpecColumnScope::FEATURE,
        .type = SpecScalarColumn{.type = columnSchema.type},
    };
}

uint32_t encode_column_type(const ColumnSchema& columnSchema) {
    const SpecColumn specColumn = make_spec_column(columnSchema);
    const auto typeCode = SpecTypeMap::encodeColumnType(
        specColumn.getScalarType().getPhysicalType(),
        std::nullopt,
        std::nullopt,
        std::nullopt,
        specColumn.nullable,
        false,
        specColumn.getScalarType().hasLongID);
    if (!typeCode.has_value()) {
        error("%s: failed to encode MLT type code", __func__);
    }
    return *typeCode;
}

ColumnSchema decode_column_type(uint32_t typeCode) {
    for (ScalarType candidate : {
            ScalarType::BOOLEAN,
            ScalarType::INT_32,
            ScalarType::FLOAT,
            ScalarType::STRING}) {
        for (bool nullable : {false, true}) {
            ColumnSchema schema;
            schema.type = candidate;
            schema.nullable = nullable;
            if (encode_column_type(schema) == typeCode) {
                return schema;
            }
        }
    }
    error("%s: unsupported MLT type code %u", __func__, typeCode);
    return ColumnSchema{};
}

std::vector<bool> resolve_present_bitmap(const PropertyColumn& column, bool nullable, size_t rowCount) {
    std::vector<bool> present;
    if (!nullable) {
        return present;
    }
    present.reserve(rowCount);
    if (!column.present.empty()) {
        present = column.present;
    } else {
        present.assign(rowCount, true);
    }
    return present;
}

void append_stream(std::string& layer, uint8_t streamTag, uint8_t encodingTag,
    uint64_t numValues, const std::string& data) {
    layer.push_back(static_cast<char>(streamTag));
    layer.push_back(static_cast<char>(encodingTag));
    append_varint(layer, numValues);
    append_varint(layer, data.size());
    layer.append(data);
}

void append_present_stream(std::string& layer, const std::vector<bool>& present, size_t rowCount) {
    const std::string rle = encode_bool_rle(present);
    append_stream(layer, 0x00, 0x02, rowCount, rle);
}

void encode_boolean_column(std::string& layer, const PropertyColumn& column,
    const std::vector<bool>& present, bool nullable, size_t rowCount) {
    std::vector<bool> boolValues;
    boolValues.reserve(nullable ? rowCount : column.boolValues.size());
    for (size_t row = 0; row < rowCount; ++row) {
        if (nullable && !present[row]) {
            continue;
        }
        boolValues.push_back(column.boolValues[row]);
    }
    if (nullable) {
        append_present_stream(layer, present, rowCount);
    }
    const std::string rle = encode_bool_rle(boolValues);
    append_stream(layer, 0x10, 0x02, boolValues.size(), rle);
}

void encode_int32_column(std::string& layer, const PropertyColumn& column,
    const std::vector<bool>& present, bool nullable, size_t rowCount) {
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
        append_present_stream(layer, present, rowCount);
    }
    append_stream(layer, 0x10, 0x02, nonNullCount, intData);
}

void encode_float_column(std::string& layer, const PropertyColumn& column,
    const std::vector<bool>& present, bool nullable, size_t rowCount) {
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
        append_present_stream(layer, present, rowCount);
    }
    append_stream(layer, 0x10, 0x00, nonNullCount, floatData);
}

void encode_string_column(std::string& layer, const PropertyColumn& column,
    const std::vector<bool>& present, bool nullable, size_t rowCount,
    const GlobalStringDictionary* stringDictionary) {
    const bool useDictionary = !column.stringCodes.empty();
    if (useDictionary && stringDictionary == nullptr) {
        error("%s: string dictionary is required for dictionary-coded STRING columns", __func__);
    }
    std::string lengths;
    std::string bytes;
    size_t nonNullCount = 0;
    for (size_t row = 0; row < rowCount; ++row) {
        if (nullable && !present[row]) {
            continue;
        }
        ++nonNullCount;
        const std::string& value = useDictionary
            ? stringDictionary->lookup(column.stringCodes[row])
            : column.stringValues[row];
        append_varint(lengths, value.size());
        bytes.append(value);
    }
    append_varint(layer, nullable ? 3 : 2);
    if (nullable) {
        append_present_stream(layer, present, rowCount);
    }
    append_stream(layer, 0x30, 0x02, nonNullCount, lengths);
    append_stream(layer, 0x10, 0x00, 0, bytes);
}

void decode_boolean_column(PropertyColumn& column, const uint8_t*& ptr, const uint8_t* end, size_t rowCount) {
    std::vector<bool> present(rowCount, true);
    const size_t expectedStreams = column.nullable ? 2 : 1;
    for (size_t streamIdx = 0; streamIdx < expectedStreams; ++streamIdx) {
        if (ptr + 2 > end) {
            error("%s: truncated BOOLEAN stream header", __func__);
        }
        const uint8_t h0 = *ptr++;
        ptr++;
        const uint64_t numValues = read_varint(ptr, end, __func__);
        const uint64_t byteLen = read_varint(ptr, end, __func__);
        const uint8_t* streamData = ptr;
        ptr = checked_advance(ptr, end, byteLen, __func__);
        const uint8_t physicalType = (h0 >> 4u) & 0x0Fu;
        if (physicalType == 0) {
            present = decode_bool_rle(streamData, static_cast<size_t>(byteLen), rowCount);
            column.present = present;
        } else if (physicalType == 1) {
            const std::vector<bool> values = decode_bool_rle(streamData, static_cast<size_t>(byteLen),
                static_cast<size_t>(numValues));
            column.boolValues.assign(rowCount, false);
            size_t valueIdx = 0;
            for (size_t row = 0; row < rowCount; ++row) {
                if (!present[row]) {
                    continue;
                }
                if (valueIdx >= values.size()) {
                    error("%s: BOOLEAN value stream underflow", __func__);
                }
                column.boolValues[row] = values[valueIdx++];
            }
            if (valueIdx != values.size()) {
                error("%s: BOOLEAN value stream overflow", __func__);
            }
        } else {
            error("%s: unsupported BOOLEAN physical stream %u", __func__, physicalType);
        }
    }
    if (column.boolValues.size() != rowCount) {
        error("%s: BOOLEAN row count mismatch", __func__);
    }
}

void decode_int32_column(PropertyColumn& column, const uint8_t*& ptr, const uint8_t* end, size_t rowCount) {
    std::vector<bool> present(rowCount, true);
    const size_t expectedStreams = column.nullable ? 2 : 1;
    for (size_t streamIdx = 0; streamIdx < expectedStreams; ++streamIdx) {
        if (ptr + 2 > end) {
            error("%s: truncated INT stream header", __func__);
        }
        const uint8_t h0 = *ptr++;
        ptr++;
        const uint64_t numValues = read_varint(ptr, end, __func__);
        const uint64_t byteLen = read_varint(ptr, end, __func__);
        const uint8_t* streamData = ptr;
        ptr = checked_advance(ptr, end, byteLen, __func__);
        const uint8_t physicalType = (h0 >> 4u) & 0x0Fu;
        if (physicalType == 0) {
            present = decode_bool_rle(streamData, static_cast<size_t>(byteLen), rowCount);
            column.present = present;
        } else if (physicalType == 1) {
            column.intValues.assign(rowCount, 0);
            const uint8_t* valuePtr = streamData;
            const uint8_t* valueEnd = streamData + byteLen;
            size_t row = 0;
            for (uint64_t valueIdx = 0; valueIdx < numValues; ++valueIdx) {
                while (row < rowCount && !present[row]) {
                    ++row;
                }
                if (row >= rowCount) {
                    error("%s: INT value stream overflow", __func__);
                }
                column.intValues[row++] = decode_zigzag32(read_varint(valuePtr, valueEnd, __func__));
            }
            while (row < rowCount && !present[row]) {
                ++row;
            }
            if (row != rowCount || valuePtr != valueEnd) {
                error("%s: INT stream length mismatch", __func__);
            }
        } else {
            error("%s: unsupported INT physical stream %u", __func__, physicalType);
        }
    }
    if (column.intValues.size() != rowCount) {
        error("%s: INT row count mismatch", __func__);
    }
}

void decode_float_column(PropertyColumn& column, const uint8_t*& ptr, const uint8_t* end, size_t rowCount) {
    std::vector<bool> present(rowCount, true);
    const size_t expectedStreams = column.nullable ? 2 : 1;
    for (size_t streamIdx = 0; streamIdx < expectedStreams; ++streamIdx) {
        if (ptr + 2 > end) {
            error("%s: truncated FLOAT stream header", __func__);
        }
        const uint8_t h0 = *ptr++;
        ptr++;
        const uint64_t numValues = read_varint(ptr, end, __func__);
        const uint64_t byteLen = read_varint(ptr, end, __func__);
        const uint8_t* streamData = ptr;
        ptr = checked_advance(ptr, end, byteLen, __func__);
        const uint8_t physicalType = (h0 >> 4u) & 0x0Fu;
        if (physicalType == 0) {
            present = decode_bool_rle(streamData, static_cast<size_t>(byteLen), rowCount);
            column.present = present;
        } else if (physicalType == 1) {
            if (byteLen != numValues * sizeof(uint32_t)) {
                error("%s: FLOAT byte length mismatch", __func__);
            }
            column.floatValues.assign(rowCount, 0.0f);
            const uint8_t* valuePtr = streamData;
            size_t row = 0;
            for (uint64_t valueIdx = 0; valueIdx < numValues; ++valueIdx) {
                while (row < rowCount && !present[row]) {
                    ++row;
                }
                if (row >= rowCount) {
                    error("%s: FLOAT value stream overflow", __func__);
                }
                uint32_t bits = 0;
                std::memcpy(&bits, valuePtr, sizeof(bits));
                valuePtr += sizeof(bits);
                float value = 0.0f;
                std::memcpy(&value, &bits, sizeof(value));
                column.floatValues[row++] = value;
            }
            while (row < rowCount && !present[row]) {
                ++row;
            }
            if (row != rowCount || valuePtr != streamData + byteLen) {
                error("%s: FLOAT stream length mismatch", __func__);
            }
        } else {
            error("%s: unsupported FLOAT physical stream %u", __func__, physicalType);
        }
    }
    if (column.floatValues.size() != rowCount) {
        error("%s: FLOAT row count mismatch", __func__);
    }
}

void decode_string_column(PropertyColumn& column, const uint8_t*& ptr, const uint8_t* end, size_t rowCount) {
    std::vector<bool> present(rowCount, true);
    const uint64_t numStreams = read_varint(ptr, end, __func__);
    std::vector<uint64_t> lengths;
    const uint8_t* stringBytes = nullptr;
    size_t stringByteLen = 0;
    for (uint64_t streamIdx = 0; streamIdx < numStreams; ++streamIdx) {
        if (ptr + 2 > end) {
            error("%s: truncated STRING stream header", __func__);
        }
        const uint8_t h0 = *ptr++;
        ptr++;
        const uint64_t numValues = read_varint(ptr, end, __func__);
        const uint64_t byteLen = read_varint(ptr, end, __func__);
        const uint8_t* streamData = ptr;
        ptr = checked_advance(ptr, end, byteLen, __func__);
        const uint8_t physicalType = (h0 >> 4u) & 0x0Fu;
        if (physicalType == 0) {
            present = decode_bool_rle(streamData, static_cast<size_t>(byteLen), rowCount);
            column.present = present;
        } else if (physicalType == 3) {
            lengths.clear();
            lengths.reserve(static_cast<size_t>(numValues));
            const uint8_t* valuePtr = streamData;
            const uint8_t* valueEnd = streamData + byteLen;
            for (uint64_t valueIdx = 0; valueIdx < numValues; ++valueIdx) {
                lengths.push_back(read_varint(valuePtr, valueEnd, __func__));
            }
            if (valuePtr != valueEnd) {
                error("%s: STRING length stream mismatch", __func__);
            }
        } else if (physicalType == 1) {
            stringBytes = streamData;
            stringByteLen = static_cast<size_t>(byteLen);
        } else {
            error("%s: unsupported STRING physical stream %u", __func__, physicalType);
        }
    }
    column.stringValues.assign(rowCount, std::string());
    const uint8_t* dataPtr = stringBytes;
    const uint8_t* dataEnd = stringBytes + stringByteLen;
    size_t row = 0;
    for (size_t valueIdx = 0; valueIdx < lengths.size(); ++valueIdx) {
        while (row < rowCount && !present[row]) {
            ++row;
        }
        if (row >= rowCount) {
            error("%s: STRING value stream overflow", __func__);
        }
        const uint64_t len = lengths[valueIdx];
        if (dataPtr == nullptr || len > static_cast<uint64_t>(dataEnd - dataPtr)) {
            error("%s: truncated STRING data", __func__);
        }
        column.stringValues[row++] = std::string(reinterpret_cast<const char*>(dataPtr), static_cast<size_t>(len));
        dataPtr += static_cast<size_t>(len);
    }
    while (row < rowCount && !present[row]) {
        ++row;
    }
    if (row != rowCount || dataPtr != dataEnd) {
        error("%s: STRING stream length mismatch", __func__);
    }
}

} // namespace

std::string gzip_compress(const std::string& data) {
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 | 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        error("%s: deflateInit2 failed", __func__);
    }
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
    zs.avail_in = static_cast<uInt>(data.size());

    int ret = Z_OK;
    char outbuffer[32768];
    std::string out;
    do {
        zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
        zs.avail_out = sizeof(outbuffer);
        ret = deflate(&zs, Z_FINISH);
        if (ret != Z_OK && ret != Z_STREAM_END) {
            deflateEnd(&zs);
            error("%s: deflate failed", __func__);
        }
        if (out.size() < zs.total_out) {
            out.append(outbuffer, zs.total_out - out.size());
        }
    } while (ret == Z_OK);

    deflateEnd(&zs);
    return out;
}

std::string gzip_decompress(const std::string& data) {
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
    zs.avail_in = static_cast<uInt>(data.size());
    if (inflateInit2(&zs, 15 | 32) != Z_OK) {
        error("%s: inflateInit2 failed", __func__);
    }

    int ret = Z_OK;
    char outbuffer[32768];
    std::string out;
    while (ret != Z_STREAM_END) {
        zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
        zs.avail_out = sizeof(outbuffer);
        ret = inflate(&zs, Z_NO_FLUSH);
        if (ret != Z_OK && ret != Z_STREAM_END) {
            inflateEnd(&zs);
            error("%s: inflate failed", __func__);
        }
        out.append(outbuffer, sizeof(outbuffer) - zs.avail_out);
    }

    inflateEnd(&zs);
    return out;
}

std::string encode_bool_rle(const std::vector<bool>& present) {
    const size_t n = present.size();
    const size_t numBytes = (n + 7u) / 8u;
    std::vector<uint8_t> packed(numBytes, 0);
    for (size_t i = 0; i < n; ++i) {
        if (present[i]) {
            packed[i / 8u] |= static_cast<uint8_t>(1u << (i % 8u));
        }
    }

    std::string out;
    size_t offset = 0;
    while (offset < numBytes) {
        const size_t chunk = std::min<size_t>(128, numBytes - offset);
        out.push_back(static_cast<char>(static_cast<uint8_t>(256 - chunk)));
        for (size_t i = 0; i < chunk; ++i) {
            out.push_back(static_cast<char>(packed[offset + i]));
        }
        offset += chunk;
    }
    return out;
}

std::vector<bool> decode_bool_rle(const uint8_t* data, size_t len, size_t count) {
    std::vector<bool> result;
    result.reserve(count);
    size_t offset = 0;
    while (offset < len && result.size() < count) {
        const uint8_t header = data[offset++];
        if (header >= 128) {
            const size_t runLen = 256u - header;
            for (size_t i = 0; i < runLen && offset < len && result.size() < count; ++i, ++offset) {
                const uint8_t byte = data[offset];
                for (int bit = 0; bit < 8 && result.size() < count; ++bit) {
                    result.push_back(((byte >> bit) & 1u) != 0u);
                }
            }
        } else {
            const size_t runLen = header + 3u;
            if (offset >= len) {
                error("%s: malformed repeated RLE block", __func__);
            }
            const uint8_t byte = data[offset++];
            for (size_t i = 0; i < runLen && result.size() < count; ++i) {
                for (int bit = 0; bit < 8 && result.size() < count; ++bit) {
                    result.push_back(((byte >> bit) & 1u) != 0u);
                }
            }
        }
    }
    if (result.size() < count) {
        result.resize(count, true);
    }
    return result;
}

void epsg3857_to_wgs84(double x, double y, double& lon, double& lat) {
    lon = (x / kEpsg3857Radius) * (180.0 / M_PI);
    lat = (2.0 * std::atan(std::exp(y / kEpsg3857Radius)) - M_PI / 2.0) * (180.0 / M_PI);
}

double epsg3857_scale_factor(uint8_t zoom) {
    return 2.0 * kEpsg3857Bound / static_cast<double>(uint64_t{1} << (zoom + 12));
}

void epsg3857_to_tilecoord(double x, double y, uint8_t zoom,
    int64_t& tileX, int64_t& tileY, double& localX, double& localY) {
    if (!std::isfinite(x)) x = 40000000.0;
    if (!std::isfinite(y)) y = 40000000.0;

    constexpr double tileSize = 256.0;
    const uint64_t numTiles = uint64_t{1} << zoom;
    tileX = static_cast<int64_t>((x + kEpsg3857Bound) / (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)));
    tileY = static_cast<int64_t>((kEpsg3857Bound - y) / (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)));

    const double tileOriginX = static_cast<double>(tileX) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)) - kEpsg3857Bound;
    const double tileOriginY = kEpsg3857Bound - static_cast<double>(tileY) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles));
    localX = (x - tileOriginX) / (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
    localY = (tileOriginY - y) / (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
}

void tilecoord_to_epsg3857(int64_t tileX, int64_t tileY,
    double localX, double localY, uint8_t zoom,
    double& x, double& y) {
    constexpr double tileSize = 256.0;
    const uint64_t numTiles = uint64_t{1} << zoom;
    const double tileOriginX = static_cast<double>(tileX) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)) - kEpsg3857Bound;
    const double tileOriginY = kEpsg3857Bound - static_cast<double>(tileY) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles));
    x = tileOriginX + localX * (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
    y = tileOriginY - localY * (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
}

const std::string& GlobalStringDictionary::lookup(uint32_t code) const {
    if (code >= values.size()) {
        error("%s: string dictionary code %u is out of range for dictionary size %zu",
            __func__, code, values.size());
    }
    return values[code];
}

std::string encode_point_tile(const FeatureTableSchema& schema,
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
        append_varint(layer, encode_column_type(columnSchema));
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
        const std::vector<bool> present = resolve_present_bitmap(column, nullable, rowCount);

        switch (columnSchema.type) {
        case ScalarType::BOOLEAN:
            encode_boolean_column(layer, column, present, nullable, rowCount);
            break;
        case ScalarType::INT_32:
            encode_int32_column(layer, column, present, nullable, rowCount);
            break;
        case ScalarType::FLOAT:
            encode_float_column(layer, column, present, nullable, rowCount);
            break;
        case ScalarType::STRING:
            encode_string_column(layer, column, present, nullable, rowCount, stringDictionary);
            break;
        default:
            error("%s: unsupported scalar column type %d",
                __func__, static_cast<int>(columnSchema.type));
        }
    }

    std::string out;
    append_varint(out, 1 + layer.size());
    out.push_back(1);
    out.append(layer);
    return out;
}

std::string rewrite_point_tile_layer_name(const std::string& rawTile,
    const std::string& newLayerName) {
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(rawTile.data());
    const uint8_t* end = ptr + rawTile.size();
    if (ptr == end) {
        error("%s: empty MLT tile payload", __func__);
    }

    const uint8_t* outerLenPtr = ptr;
    const uint64_t outerLen = read_varint(ptr, end, __func__);
    (void)outerLen;
    const uint8_t* afterOuterLen = ptr;
    if (ptr >= end) {
        error("%s: truncated MLT tile payload", __func__);
    }
    if (*ptr++ != 1) {
        error("%s: expected single-layer tag 1", __func__);
    }

    const uint8_t* layerStart = ptr;
    const uint64_t oldNameLen = read_varint(ptr, end, __func__);
    if (oldNameLen > static_cast<uint64_t>(end - ptr)) {
        error("%s: truncated MLT layer name", __func__);
    }
    ptr += static_cast<size_t>(oldNameLen);
    const uint8_t* rest = ptr;

    std::string layer;
    layer.reserve(rawTile.size() + newLayerName.size());
    layer.append(encode_varint_bytes(newLayerName.size()));
    layer.append(newLayerName);
    layer.append(reinterpret_cast<const char*>(rest),
        static_cast<size_t>(end - rest));

    std::string out;
    out.reserve(rawTile.size() + newLayerName.size());
    out.append(reinterpret_cast<const char*>(outerLenPtr),
        static_cast<size_t>(afterOuterLen - outerLenPtr));
    out.clear();
    out.append(encode_varint_bytes(1 + layer.size()));
    out.push_back(1);
    out.append(layer);
    if (reinterpret_cast<const char*>(layerStart) != rawTile.data() +
            static_cast<std::ptrdiff_t>(afterOuterLen - outerLenPtr + 1)) {
        // The current encoder emits a single point layer starting immediately
        // after the outer tag; reject other layouts instead of guessing.
        error("%s: unexpected MLT layer layout", __func__);
    }
    return out;
}

DecodedPointTile decode_point_tile(const std::string& rawTile) {
    DecodedPointTile out;
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(rawTile.data());
    const uint8_t* end = ptr + rawTile.size();
    if (ptr == end) {
        return out;
    }

    const uint64_t layerLen = read_varint(ptr, end, __func__);
    if (layerLen == 0 || ptr >= end) {
        error("%s: malformed tile layer header", __func__);
    }
    if (layerLen > static_cast<uint64_t>(end - ptr)) {
        error("%s: truncated tile layer", __func__);
    }
    const uint8_t* layerEnd = ptr + static_cast<size_t>(layerLen);
    const uint8_t tag = *ptr++;
    if (tag != 1) {
        error("%s: expected point layer tag 1, found %u", __func__, static_cast<unsigned>(tag));
    }

    const uint64_t layerNameLen = read_varint(ptr, layerEnd, __func__);
    if (layerNameLen > static_cast<uint64_t>(layerEnd - ptr)) {
        error("%s: truncated layer name", __func__);
    }
    out.schema.layerName.assign(reinterpret_cast<const char*>(ptr), static_cast<size_t>(layerNameLen));
    ptr += static_cast<size_t>(layerNameLen);
    out.schema.extent = static_cast<uint32_t>(read_varint(ptr, layerEnd, __func__));
    const uint64_t numColumns = read_varint(ptr, layerEnd, __func__);
    if (numColumns == 0) {
        error("%s: MLT layer has no geometry column", __func__);
    }

    const uint64_t geomTypeCode = read_varint(ptr, layerEnd, __func__);
    if (geomTypeCode != 4) {
        error("%s: unsupported geometry column type %llu", __func__,
            static_cast<unsigned long long>(geomTypeCode));
    }

    out.schema.columns.reserve(static_cast<size_t>(numColumns - 1));
    out.tile.columns.reserve(static_cast<size_t>(numColumns - 1));
    for (uint64_t col = 1; col < numColumns; ++col) {
        ColumnSchema schema = decode_column_type(static_cast<uint32_t>(read_varint(ptr, layerEnd, __func__)));
        const uint64_t colNameLen = read_varint(ptr, layerEnd, __func__);
        if (colNameLen > static_cast<uint64_t>(layerEnd - ptr)) {
            error("%s: truncated column name", __func__);
        }
        schema.name.assign(reinterpret_cast<const char*>(ptr), static_cast<size_t>(colNameLen));
        ptr += static_cast<size_t>(colNameLen);
        out.schema.columns.push_back(schema);
        out.tile.columns.emplace_back(schema.type, schema.nullable);
    }

    const uint64_t geomNumStreams = read_varint(ptr, layerEnd, __func__);
    size_t rowCount = 0;
    for (uint64_t streamIdx = 0; streamIdx < geomNumStreams; ++streamIdx) {
        if (ptr + 2 > layerEnd) {
            error("%s: truncated geometry stream header", __func__);
        }
        const uint8_t h0 = *ptr++;
        ptr++;
        const uint64_t numValues = read_varint(ptr, layerEnd, __func__);
        const uint64_t byteLen = read_varint(ptr, layerEnd, __func__);
        const uint8_t* streamData = ptr;
        ptr = checked_advance(ptr, layerEnd, byteLen, __func__);
        const uint8_t physicalType = (h0 >> 4u) & 0x0Fu;
        const uint8_t dictionaryType = h0 & 0x0Fu;
        if (physicalType == 1 && dictionaryType == 0) {
            rowCount = static_cast<size_t>(numValues);
        } else if (physicalType == 1 && dictionaryType == 3) {
            if ((numValues % 2u) != 0u) {
                error("%s: vertex stream length must be even", __func__);
            }
            rowCount = static_cast<size_t>(numValues / 2u);
            out.tile.localX.reserve(rowCount);
            out.tile.localY.reserve(rowCount);
            const uint8_t* valuePtr = streamData;
            const uint8_t* valueEnd = streamData + byteLen;
            for (size_t row = 0; row < rowCount; ++row) {
                out.tile.localX.push_back(decode_zigzag32(read_varint(valuePtr, valueEnd, __func__)));
                out.tile.localY.push_back(decode_zigzag32(read_varint(valuePtr, valueEnd, __func__)));
            }
            if (valuePtr != valueEnd) {
                error("%s: vertex stream length mismatch", __func__);
            }
        }
    }
    if (out.tile.localX.size() != rowCount || out.tile.localY.size() != rowCount) {
        error("%s: geometry row count mismatch", __func__);
    }

    for (size_t colIdx = 0; colIdx < out.schema.columns.size(); ++colIdx) {
        auto& column = out.tile.columns[colIdx];
        switch (out.schema.columns[colIdx].type) {
        case ScalarType::BOOLEAN:
            decode_boolean_column(column, ptr, layerEnd, rowCount);
            break;
        case ScalarType::INT_32:
            decode_int32_column(column, ptr, layerEnd, rowCount);
            break;
        case ScalarType::FLOAT:
            decode_float_column(column, ptr, layerEnd, rowCount);
            break;
        case ScalarType::STRING:
            decode_string_column(column, ptr, layerEnd, rowCount);
            break;
        default:
            error("%s: unsupported scalar column type %d", __func__,
                static_cast<int>(out.schema.columns[colIdx].type));
        }
    }

    if (ptr != layerEnd || layerEnd != end) {
        error("%s: unexpected trailing MLT data", __func__);
    }
    return out;
}

} // namespace mlt_pmtiles
