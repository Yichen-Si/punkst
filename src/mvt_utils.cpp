#include "mvt_utils.hpp"

#include "utils.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <limits>
#include <unordered_map>
#include <unordered_set>

namespace {

using pm_vector::ColumnSchema;
using pm_vector::FeatureTableSchema;
using pm_vector::GlobalStringDictionary;
using pm_vector::PointTileData;
using pm_vector::PolygonTileData;
using pm_vector::PropertyColumn;
using pm_vector::ScalarType;

enum WireType : uint8_t {
    WT_VARINT = 0,
    WT_FIXED64 = 1,
    WT_LEN = 2,
    WT_FIXED32 = 5,
};

enum GeomType : uint32_t {
    GEOM_POINT = 1,
    GEOM_POLYGON = 3,
};

void write_varint(std::string& out, uint64_t v) {
    do {
        uint8_t b = static_cast<uint8_t>(v & 0x7fu);
        v >>= 7u;
        if (v) {
            b |= 0x80u;
        }
        out.push_back(static_cast<char>(b));
    } while (v);
}

uint64_t read_varint(const std::string& data, size_t& pos) {
    uint64_t out = 0;
    uint32_t shift = 0;
    while (pos < data.size() && shift <= 63) {
        const uint8_t b = static_cast<uint8_t>(data[pos++]);
        out |= (static_cast<uint64_t>(b & 0x7fu) << shift);
        if ((b & 0x80u) == 0) {
            return out;
        }
        shift += 7u;
    }
    error("%s: truncated or invalid varint", __func__);
    return 0;
}

int32_t decode_zigzag32(uint32_t value) {
    return static_cast<int32_t>((value >> 1u) ^ (~(value & 1u) + 1u));
}

void write_key(std::string& out, uint32_t field, WireType wt) {
    write_varint(out, (static_cast<uint64_t>(field) << 3u) | static_cast<uint64_t>(wt));
}

void add_uint32(std::string& out, uint32_t field, uint32_t value) {
    write_key(out, field, WT_VARINT);
    write_varint(out, value);
}

void add_uint64(std::string& out, uint32_t field, uint64_t value) {
    write_key(out, field, WT_VARINT);
    write_varint(out, value);
}

void add_sint64(std::string& out, uint32_t field, int64_t value) {
    const uint64_t zz = (static_cast<uint64_t>(value) << 1u) ^
        static_cast<uint64_t>(value >> 63);
    write_key(out, field, WT_VARINT);
    write_varint(out, zz);
}

void add_bool(std::string& out, uint32_t field, bool value) {
    write_key(out, field, WT_VARINT);
    write_varint(out, value ? 1u : 0u);
}

void add_float(std::string& out, uint32_t field, float value) {
    uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    write_key(out, field, WT_FIXED32);
    for (int i = 0; i < 4; ++i) {
        out.push_back(static_cast<char>((bits >> (8 * i)) & 0xffu));
    }
}

void add_string(std::string& out, uint32_t field, const std::string& value) {
    write_key(out, field, WT_LEN);
    write_varint(out, value.size());
    out.append(value);
}

void add_bytes(std::string& out, uint32_t field, const std::string& value) {
    add_string(out, field, value);
}

void add_packed_uint32(std::string& out, uint32_t field,
    const std::vector<uint32_t>& values) {
    std::string payload;
    for (uint32_t v : values) {
        write_varint(payload, v);
    }
    add_bytes(out, field, payload);
}

std::string read_len_payload(const std::string& data, size_t& pos) {
    const uint64_t len = read_varint(data, pos);
    if (len > data.size() - pos) {
        error("%s: truncated length-delimited field", __func__);
    }
    std::string out(data.data() + pos, data.data() + pos + len);
    pos += static_cast<size_t>(len);
    return out;
}

void skip_field(const std::string& data, size_t& pos, uint32_t wireType) {
    switch (wireType) {
    case WT_VARINT:
        (void)read_varint(data, pos);
        break;
    case WT_FIXED64:
        if (data.size() - pos < 8) {
            error("%s: truncated fixed64 field", __func__);
        }
        pos += 8;
        break;
    case WT_LEN: {
        const uint64_t len = read_varint(data, pos);
        if (len > data.size() - pos) {
            error("%s: truncated skipped field", __func__);
        }
        pos += static_cast<size_t>(len);
        break;
    }
    case WT_FIXED32:
        if (data.size() - pos < 4) {
            error("%s: truncated fixed32 field", __func__);
        }
        pos += 4;
        break;
    default:
        error("%s: unsupported protobuf wire type %u", __func__, wireType);
    }
}

std::vector<uint32_t> read_packed_uint32(const std::string& payload) {
    std::vector<uint32_t> out;
    size_t pos = 0;
    while (pos < payload.size()) {
        out.push_back(static_cast<uint32_t>(read_varint(payload, pos)));
    }
    return out;
}

struct MvtValue {
    ScalarType type = ScalarType::STRING;
    int32_t intValue = 0;
    float floatValue = 0.0f;
    std::string stringValue;
    bool boolValue = false;
    std::string raw;
};

std::string canonical_value_key(const MvtValue& value) {
    switch (value.type) {
    case ScalarType::INT_32:
        return "i:" + std::to_string(value.intValue);
    case ScalarType::FLOAT: {
        uint32_t bits = 0;
        std::memcpy(&bits, &value.floatValue, sizeof(bits));
        return "f:" + std::to_string(bits);
    }
    case ScalarType::BOOLEAN:
        return value.boolValue ? "b:1" : "b:0";
    case ScalarType::STRING:
    default:
        return "s:" + value.stringValue;
    }
}

MvtValue decode_value_message(const std::string& raw) {
    MvtValue out;
    out.raw = raw;
    size_t pos = 0;
    bool seen = false;
    while (pos < raw.size()) {
        const uint64_t key = read_varint(raw, pos);
        const uint32_t tag = static_cast<uint32_t>(key >> 3u);
        const uint32_t wt = static_cast<uint32_t>(key & 0x7u);
        switch (tag) {
        case 1:
            if (wt != WT_LEN) { error("%s: invalid string value wire type", __func__); }
            out.type = ScalarType::STRING;
            out.stringValue = read_len_payload(raw, pos);
            seen = true;
            break;
        case 2: {
            if (wt != WT_FIXED32 || raw.size() - pos < 4) { error("%s: invalid float value", __func__); }
            uint32_t bits = 0;
            for (int i = 0; i < 4; ++i) {
                bits |= static_cast<uint32_t>(static_cast<uint8_t>(raw[pos++])) << (8 * i);
            }
            out.type = ScalarType::FLOAT;
            std::memcpy(&out.floatValue, &bits, sizeof(out.floatValue));
            seen = true;
            break;
        }
        case 3: {
            if (wt != WT_FIXED64 || raw.size() - pos < 8) { error("%s: invalid double value", __func__); }
            uint64_t bits = 0;
            for (int i = 0; i < 8; ++i) {
                bits |= static_cast<uint64_t>(static_cast<uint8_t>(raw[pos++])) << (8 * i);
            }
            double d = 0.0;
            std::memcpy(&d, &bits, sizeof(d));
            out.type = ScalarType::FLOAT;
            out.floatValue = static_cast<float>(d);
            seen = true;
            break;
        }
        case 4:
            if (wt != WT_VARINT) { error("%s: invalid int value", __func__); }
            out.type = ScalarType::INT_32;
            out.intValue = static_cast<int32_t>(read_varint(raw, pos));
            seen = true;
            break;
        case 5:
            if (wt != WT_VARINT) { error("%s: invalid uint value", __func__); }
            out.type = ScalarType::INT_32;
            out.intValue = static_cast<int32_t>(read_varint(raw, pos));
            seen = true;
            break;
        case 6: {
            if (wt != WT_VARINT) { error("%s: invalid sint value", __func__); }
            const uint64_t zz = read_varint(raw, pos);
            out.type = ScalarType::INT_32;
            out.intValue = static_cast<int32_t>((zz >> 1u) ^ (~(zz & 1u) + 1u));
            seen = true;
            break;
        }
        case 7:
            if (wt != WT_VARINT) { error("%s: invalid bool value", __func__); }
            out.type = ScalarType::BOOLEAN;
            out.boolValue = read_varint(raw, pos) != 0;
            seen = true;
            break;
        default:
            skip_field(raw, pos, wt);
            break;
        }
    }
    if (!seen) {
        out.type = ScalarType::STRING;
        out.stringValue.clear();
    }
    return out;
}

std::string encode_value_message(const MvtValue& value) {
    std::string out;
    switch (value.type) {
    case ScalarType::INT_32:
        add_sint64(out, 6, value.intValue);
        break;
    case ScalarType::FLOAT:
        add_float(out, 2, value.floatValue);
        break;
    case ScalarType::BOOLEAN:
        add_bool(out, 7, value.boolValue);
        break;
    case ScalarType::STRING:
    default:
        add_string(out, 1, value.stringValue);
        break;
    }
    return out;
}

bool column_value_present(const ColumnSchema& schema,
    const PropertyColumn& column, size_t row) {
    return !schema.nullable || column.present.empty() || column.present[row];
}

MvtValue value_from_column(const ColumnSchema& schema,
    const PropertyColumn& column,
    const GlobalStringDictionary* stringDictionary,
    size_t row) {
    MvtValue out;
    out.type = schema.type;
    switch (schema.type) {
    case ScalarType::INT_32:
        out.intValue = column.intValues[row];
        break;
    case ScalarType::FLOAT:
        out.floatValue = column.floatValues[row];
        break;
    case ScalarType::BOOLEAN:
        out.boolValue = column.boolValues[row];
        break;
    case ScalarType::STRING:
    default:
        if (!column.stringCodes.empty()) {
            if (stringDictionary == nullptr) {
                error("%s: string-code column requires a dictionary", __func__);
            }
            out.stringValue = stringDictionary->lookup(column.stringCodes[row]);
        } else {
            out.stringValue = column.stringValues[row];
        }
        break;
    }
    return out;
}

struct LayerTable {
    std::vector<std::string> keys;
    std::vector<MvtValue> values;
    std::unordered_map<std::string, uint32_t> valueIndex;
};

std::vector<uint32_t> build_feature_tags(LayerTable& table,
    const FeatureTableSchema& schema,
    const std::vector<PropertyColumn>& columns,
    const GlobalStringDictionary* stringDictionary,
    size_t row) {
    std::vector<uint32_t> tags;
    for (size_t c = 0; c < schema.columns.size(); ++c) {
        if (!column_value_present(schema.columns[c], columns[c], row)) {
            continue;
        }
        const MvtValue value = value_from_column(schema.columns[c], columns[c], stringDictionary, row);
        const std::string key = canonical_value_key(value);
        auto it = table.valueIndex.find(key);
        uint32_t valueIdx = 0;
        if (it == table.valueIndex.end()) {
            valueIdx = static_cast<uint32_t>(table.values.size());
            table.values.push_back(value);
            table.valueIndex.emplace(key, valueIdx);
        } else {
            valueIdx = it->second;
        }
        tags.push_back(static_cast<uint32_t>(c));
        tags.push_back(valueIdx);
    }
    return tags;
}

std::vector<uint32_t> point_geometry(int32_t x, int32_t y) {
    return {
        (1u << 3u) | 1u,
        pm_vector::encode_zigzag32(x),
        pm_vector::encode_zigzag32(y),
    };
}

double signed_area(const std::vector<std::pair<int32_t, int32_t>>& ring) {
    double area = 0.0;
    for (size_t i = 0; i < ring.size(); ++i) {
        const auto& a = ring[i];
        const auto& b = ring[(i + 1u) % ring.size()];
        area += static_cast<double>(a.first) * static_cast<double>(b.second) -
            static_cast<double>(b.first) * static_cast<double>(a.second);
    }
    return area * 0.5;
}

std::vector<uint32_t> polygon_geometry(std::vector<std::pair<int32_t, int32_t>> ring) {
    while (ring.size() > 1u && ring.front() == ring.back()) {
        ring.pop_back();
    }
    if (ring.size() < 3u || std::abs(signed_area(ring)) < 0.5) {
        return {};
    }
    if (signed_area(ring) < 0.0) {
        std::reverse(ring.begin(), ring.end());
    }
    std::vector<uint32_t> geom;
    geom.reserve(3u + ring.size() * 2u);
    geom.push_back((1u << 3u) | 1u);
    int32_t cx = 0;
    int32_t cy = 0;
    geom.push_back(pm_vector::encode_zigzag32(ring[0].first - cx));
    geom.push_back(pm_vector::encode_zigzag32(ring[0].second - cy));
    cx = ring[0].first;
    cy = ring[0].second;
    geom.push_back((static_cast<uint32_t>(ring.size() - 1u) << 3u) | 2u);
    for (size_t i = 1; i < ring.size(); ++i) {
        geom.push_back(pm_vector::encode_zigzag32(ring[i].first - cx));
        geom.push_back(pm_vector::encode_zigzag32(ring[i].second - cy));
        cx = ring[i].first;
        cy = ring[i].second;
    }
    geom.push_back((1u << 3u) | 7u);
    return geom;
}

void write_layer(std::string& tileOut,
    const FeatureTableSchema& schema,
    LayerTable& table,
    const std::vector<std::string>& featureMessages) {
    std::string layer;
    add_uint32(layer, 15, 2);
    add_string(layer, 1, schema.layerName);
    for (const auto& feature : featureMessages) {
        add_bytes(layer, 2, feature);
    }
    for (const auto& key : table.keys) {
        add_string(layer, 3, key);
    }
    for (const auto& value : table.values) {
        add_bytes(layer, 4, encode_value_message(value));
    }
    add_uint32(layer, 5, schema.extent);
    add_bytes(tileOut, 3, layer);
}

std::string encode_point_tile_impl(const FeatureTableSchema& schema,
    const PointTileData& tile,
    const std::vector<uint32_t>* order,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary) {
    if (rowCount > tile.size()) {
        error("%s: requested row count exceeds point tile size", __func__);
    }
    LayerTable table;
    for (const auto& col : schema.columns) {
        table.keys.push_back(col.name);
    }
    std::vector<std::string> features;
    features.reserve(rowCount);
    for (size_t outRow = 0; outRow < rowCount; ++outRow) {
        const size_t row = order == nullptr ? outRow : (*order)[outRow];
        const std::vector<uint32_t> tags =
            build_feature_tags(table, schema, tile.columns, stringDictionary, row);
        std::string feature;
        if (!tile.featureIds.empty()) {
            add_uint64(feature, 1, tile.featureIds[row]);
        }
        if (!tags.empty()) {
            add_packed_uint32(feature, 2, tags);
        }
        add_uint32(feature, 3, GEOM_POINT);
        add_packed_uint32(feature, 4, point_geometry(tile.localX[row], tile.localY[row]));
        features.push_back(std::move(feature));
    }
    std::string out;
    write_layer(out, schema, table, features);
    return out;
}

std::string encode_polygon_tile_impl(const FeatureTableSchema& schema,
    const PolygonTileData& tile,
    const std::vector<uint32_t>* order,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary) {
    if (rowCount > tile.size()) {
        error("%s: requested row count exceeds polygon tile size", __func__);
    }
    LayerTable table;
    for (const auto& col : schema.columns) {
        table.keys.push_back(col.name);
    }
    std::vector<std::string> features;
    features.reserve(rowCount);
    for (size_t outRow = 0; outRow < rowCount; ++outRow) {
        const size_t row = order == nullptr ? outRow : (*order)[outRow];
        std::vector<std::pair<int32_t, int32_t>> ring;
        for (uint32_t i = tile.ringOffsets[row]; i < tile.ringOffsets[row + 1u]; ++i) {
            ring.emplace_back(tile.localX[i], tile.localY[i]);
        }
        const std::vector<uint32_t> geom = polygon_geometry(std::move(ring));
        if (geom.empty()) {
            continue;
        }
        const std::vector<uint32_t> tags =
            build_feature_tags(table, schema, tile.columns, stringDictionary, row);
        std::string feature;
        if (!tile.featureIds.empty()) {
            add_uint64(feature, 1, tile.featureIds[row]);
        }
        if (!tags.empty()) {
            add_packed_uint32(feature, 2, tags);
        }
        add_uint32(feature, 3, GEOM_POLYGON);
        add_packed_uint32(feature, 4, geom);
        features.push_back(std::move(feature));
    }
    std::string out;
    write_layer(out, schema, table, features);
    return out;
}

struct DecodedFeature {
    uint64_t id = 0;
    bool hasId = false;
    uint32_t type = 0;
    std::vector<uint32_t> tags;
    std::vector<uint32_t> geometry;
};

struct DecodedLayer {
    std::string name;
    uint32_t extent = 4096;
    std::vector<std::string> keys;
    std::vector<MvtValue> values;
    std::vector<DecodedFeature> features;
};

DecodedFeature decode_feature_message(const std::string& raw) {
    DecodedFeature out;
    size_t pos = 0;
    while (pos < raw.size()) {
        const uint64_t key = read_varint(raw, pos);
        const uint32_t tag = static_cast<uint32_t>(key >> 3u);
        const uint32_t wt = static_cast<uint32_t>(key & 0x7u);
        switch (tag) {
        case 1:
            out.id = read_varint(raw, pos);
            out.hasId = true;
            break;
        case 2:
            out.tags = read_packed_uint32(read_len_payload(raw, pos));
            break;
        case 3:
            out.type = static_cast<uint32_t>(read_varint(raw, pos));
            break;
        case 4:
            out.geometry = read_packed_uint32(read_len_payload(raw, pos));
            break;
        default:
            skip_field(raw, pos, wt);
            break;
        }
    }
    return out;
}

DecodedLayer decode_layer_message(const std::string& raw) {
    DecodedLayer out;
    size_t pos = 0;
    while (pos < raw.size()) {
        const uint64_t key = read_varint(raw, pos);
        const uint32_t tag = static_cast<uint32_t>(key >> 3u);
        const uint32_t wt = static_cast<uint32_t>(key & 0x7u);
        switch (tag) {
        case 1:
            out.name = read_len_payload(raw, pos);
            break;
        case 2:
            out.features.push_back(decode_feature_message(read_len_payload(raw, pos)));
            break;
        case 3:
            out.keys.push_back(read_len_payload(raw, pos));
            break;
        case 4:
            out.values.push_back(decode_value_message(read_len_payload(raw, pos)));
            break;
        case 5:
            out.extent = static_cast<uint32_t>(read_varint(raw, pos));
            break;
        default:
            skip_field(raw, pos, wt);
            break;
        }
    }
    return out;
}

std::vector<DecodedLayer> decode_layers(const std::string& rawTile) {
    std::vector<DecodedLayer> out;
    size_t pos = 0;
    while (pos < rawTile.size()) {
        const uint64_t key = read_varint(rawTile, pos);
        const uint32_t tag = static_cast<uint32_t>(key >> 3u);
        const uint32_t wt = static_cast<uint32_t>(key & 0x7u);
        if (tag == 3) {
            out.push_back(decode_layer_message(read_len_payload(rawTile, pos)));
        } else {
            skip_field(rawTile, pos, wt);
        }
    }
    return out;
}

std::vector<std::pair<int32_t, int32_t>> decode_point_geometry(
    const std::vector<uint32_t>& geom) {
    std::vector<std::pair<int32_t, int32_t>> out;
    int32_t x = 0;
    int32_t y = 0;
    size_t pos = 0;
    while (pos < geom.size()) {
        const uint32_t cmd = geom[pos++];
        const uint32_t id = cmd & 0x7u;
        const uint32_t count = cmd >> 3u;
        if (id != 1u) {
            error("%s: point geometry must contain MoveTo commands only", __func__);
        }
        for (uint32_t i = 0; i < count; ++i) {
            if (pos + 1u >= geom.size()) {
                error("%s: truncated point geometry", __func__);
            }
            x += decode_zigzag32(geom[pos++]);
            y += decode_zigzag32(geom[pos++]);
            out.emplace_back(x, y);
        }
    }
    return out;
}

std::vector<std::pair<int32_t, int32_t>> decode_polygon_geometry(
    const std::vector<uint32_t>& geom) {
    std::vector<std::pair<int32_t, int32_t>> ring;
    int32_t x = 0;
    int32_t y = 0;
    int32_t startX = 0;
    int32_t startY = 0;
    bool haveRing = false;
    bool closed = false;
    size_t pos = 0;
    while (pos < geom.size()) {
        const uint32_t cmd = geom[pos++];
        const uint32_t id = cmd & 0x7u;
        const uint32_t count = cmd >> 3u;
        if (id == 1u) {
            if (count != 1u || haveRing) {
                error("%s: only one simple polygon ring is supported", __func__);
            }
            if (pos + 1u >= geom.size()) { error("%s: truncated polygon MoveTo", __func__); }
            x += decode_zigzag32(geom[pos++]);
            y += decode_zigzag32(geom[pos++]);
            startX = x;
            startY = y;
            ring.emplace_back(x, y);
            haveRing = true;
        } else if (id == 2u) {
            if (!haveRing) { error("%s: polygon LineTo before MoveTo", __func__); }
            for (uint32_t i = 0; i < count; ++i) {
                if (pos + 1u >= geom.size()) { error("%s: truncated polygon LineTo", __func__); }
                x += decode_zigzag32(geom[pos++]);
                y += decode_zigzag32(geom[pos++]);
                ring.emplace_back(x, y);
            }
        } else if (id == 7u) {
            if (count != 1u) { error("%s: invalid polygon ClosePath count", __func__); }
            closed = true;
        } else {
            error("%s: unsupported polygon geometry command %u", __func__, id);
        }
    }
    if (!closed) {
        error("%s: polygon ring is not closed", __func__);
    }
    if (!ring.empty() && ring.back() != std::make_pair(startX, startY)) {
        ring.emplace_back(startX, startY);
    }
    if (ring.size() < 4u) {
        error("%s: polygon ring has fewer than three vertices", __func__);
    }
    return ring;
}

void initialize_decoded_schema(const DecodedLayer& layer, FeatureTableSchema& schema,
    std::vector<PropertyColumn>& columns) {
    schema.layerName = layer.name;
    schema.extent = layer.extent;
    schema.columns.clear();
    columns.clear();
    std::vector<bool> observed(layer.keys.size(), false);
    std::vector<ScalarType> types(layer.keys.size(), ScalarType::STRING);
    for (const auto& feature : layer.features) {
        for (size_t i = 0; i + 1u < feature.tags.size(); i += 2u) {
            const uint32_t keyIdx = feature.tags[i];
            const uint32_t valIdx = feature.tags[i + 1u];
            if (keyIdx >= layer.keys.size() || valIdx >= layer.values.size()) {
                error("%s: MVT tag references out-of-range key/value", __func__);
            }
            if (!observed[keyIdx]) {
                types[keyIdx] = layer.values[valIdx].type;
                observed[keyIdx] = true;
            } else if (types[keyIdx] != layer.values[valIdx].type) {
                types[keyIdx] = ScalarType::STRING;
            }
        }
    }
    auto inferred_type_for_name = [](const std::string& name) {
        if (name == "feature" || name == "gene") {
            return ScalarType::STRING;
        }
        if (name == "ct" || name == "topK" || name == "hex_q" || name == "hex_r") {
            return ScalarType::INT_32;
        }
        if (name == "z" || name == "topP") {
            return ScalarType::FLOAT;
        }
        std::string suffix = name;
        const size_t underscore = suffix.rfind('_');
        if (underscore != std::string::npos && underscore + 1u < suffix.size()) {
            suffix = suffix.substr(underscore + 1u);
        }
        if (suffix.size() > 1 && suffix[0] == 'K' &&
            std::all_of(suffix.begin() + 1, suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
            return ScalarType::INT_32;
        }
        if (suffix.size() > 1 && suffix[0] == 'P' &&
            std::all_of(suffix.begin() + 1, suffix.end(), [](unsigned char c) { return std::isdigit(c); })) {
            return ScalarType::FLOAT;
        }
        if (!name.empty() &&
            std::all_of(name.begin(), name.end(), [](unsigned char c) { return std::isdigit(c); })) {
            return ScalarType::FLOAT;
        }
        return ScalarType::STRING;
    };
    for (size_t i = 0; i < layer.keys.size(); ++i) {
        if (!observed[i]) {
            types[i] = inferred_type_for_name(layer.keys[i]);
        }
        ColumnSchema col{layer.keys[i], types[i], true};
        schema.columns.push_back(col);
        columns.emplace_back(types[i], true);
    }
}

std::vector<int32_t> feature_value_indices(const DecodedLayer& layer,
    const DecodedFeature& feature) {
    std::vector<int32_t> out(layer.keys.size(), -1);
    for (size_t i = 0; i + 1u < feature.tags.size(); i += 2u) {
        const uint32_t keyIdx = feature.tags[i];
        const uint32_t valIdx = feature.tags[i + 1u];
        if (keyIdx >= layer.keys.size() || valIdx >= layer.values.size()) {
            error("%s: MVT tag references out-of-range key/value", __func__);
        }
        out[keyIdx] = static_cast<int32_t>(valIdx);
    }
    return out;
}

void append_decoded_properties(const DecodedLayer& layer,
    const std::vector<int32_t>& valueIdx,
    std::vector<PropertyColumn>& columns) {
    for (size_t c = 0; c < columns.size(); ++c) {
        const bool present = valueIdx[c] >= 0;
        columns[c].present.push_back(present);
        MvtValue value;
        if (present) {
            value = layer.values[static_cast<size_t>(valueIdx[c])];
        }
        switch (columns[c].type) {
        case ScalarType::INT_32:
            columns[c].intValues.push_back(present ? value.intValue : 0);
            break;
        case ScalarType::FLOAT:
            columns[c].floatValues.push_back(present ? value.floatValue : 0.0f);
            break;
        case ScalarType::BOOLEAN:
            columns[c].boolValues.push_back(present ? value.boolValue : false);
            break;
        case ScalarType::STRING:
        default:
            columns[c].stringValues.push_back(present ? value.stringValue : std::string());
            break;
        }
    }
}

const DecodedLayer& require_single_layer(const std::vector<DecodedLayer>& layers) {
    if (layers.empty()) {
        error("%s: MVT tile has no layers", __func__);
    }
    if (layers.size() > 1u) {
        error("%s: only single-layer MVT tiles are supported", __func__);
    }
    return layers.front();
}

} // namespace

namespace mvt_pmtiles {

std::string encode_point_tile(const FeatureTableSchema& schema,
    const PointTileData& tile,
    const GlobalStringDictionary* stringDictionary) {
    return encode_point_tile_impl(schema, tile, nullptr, tile.size(), stringDictionary);
}

std::string encode_point_tile_prefix(const FeatureTableSchema& schema,
    const PointTileData& tile,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary) {
    return encode_point_tile_impl(schema, tile, nullptr, rowCount, stringDictionary);
}

std::string encode_point_tile_subset(const FeatureTableSchema& schema,
    const PointTileData& tile,
    const std::vector<uint32_t>& order,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary) {
    if (rowCount > order.size()) {
        error("%s: requested row count exceeds order size", __func__);
    }
    return encode_point_tile_impl(schema, tile, &order, rowCount, stringDictionary);
}

std::string encode_polygon_tile(const FeatureTableSchema& schema,
    const PolygonTileData& tile,
    const GlobalStringDictionary* stringDictionary) {
    return encode_polygon_tile_impl(schema, tile, nullptr, tile.size(), stringDictionary);
}

std::string encode_polygon_tile_prefix(const FeatureTableSchema& schema,
    const PolygonTileData& tile,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary) {
    return encode_polygon_tile_impl(schema, tile, nullptr, rowCount, stringDictionary);
}

std::string encode_polygon_tile_subset(const FeatureTableSchema& schema,
    const PolygonTileData& tile,
    const std::vector<uint32_t>& order,
    size_t rowCount,
    const GlobalStringDictionary* stringDictionary) {
    if (rowCount > order.size()) {
        error("%s: requested row count exceeds order size", __func__);
    }
    return encode_polygon_tile_impl(schema, tile, &order, rowCount, stringDictionary);
}

DecodedPointTile decode_point_tile(const std::string& rawTile) {
    const std::vector<DecodedLayer> layers = decode_layers(rawTile);
    const DecodedLayer& layer = require_single_layer(layers);
    DecodedPointTile out;
    initialize_decoded_schema(layer, out.schema, out.tile.columns);
    for (const auto& feature : layer.features) {
        if (feature.type != GEOM_POINT) {
            error("%s: expected point MVT feature, got type=%u", __func__, feature.type);
        }
        const auto points = decode_point_geometry(feature.geometry);
        if (points.size() != 1u) {
            error("%s: only one point per MVT feature is supported", __func__);
        }
        if (feature.hasId) {
            out.tile.featureIds.push_back(feature.id);
        }
        out.tile.localX.push_back(points[0].first);
        out.tile.localY.push_back(points[0].second);
        append_decoded_properties(layer, feature_value_indices(layer, feature), out.tile.columns);
    }
    return out;
}

DecodedPointTile decode_point_tile_select(
    const std::string& rawTile,
    const std::vector<std::string>& includeColumns) {
    const std::unordered_set<std::string> include(includeColumns.begin(), includeColumns.end());
    const std::vector<DecodedLayer> layers = decode_layers(rawTile);
    const DecodedLayer& layer = require_single_layer(layers);
    FeatureTableSchema fullSchema;
    std::vector<PropertyColumn> unusedColumns;
    initialize_decoded_schema(layer, fullSchema, unusedColumns);

    DecodedPointTile out;
    out.schema.layerName = fullSchema.layerName;
    out.schema.extent = fullSchema.extent;
    out.schema.hasIdColumn = fullSchema.hasIdColumn;
    out.schema.idIsUint64 = fullSchema.idIsUint64;
    std::vector<size_t> selectedSourceColumns;
    for (size_t c = 0; c < fullSchema.columns.size(); ++c) {
        if (include.count(fullSchema.columns[c].name) == 0) {
            continue;
        }
        selectedSourceColumns.push_back(c);
        out.schema.columns.push_back(fullSchema.columns[c]);
        out.tile.columns.emplace_back(fullSchema.columns[c].type, fullSchema.columns[c].nullable);
    }

    for (const auto& feature : layer.features) {
        if (feature.type != GEOM_POINT) {
            error("%s: expected point MVT feature, got type=%u", __func__, feature.type);
        }
        const auto points = decode_point_geometry(feature.geometry);
        if (points.size() != 1u) {
            error("%s: only one point per MVT feature is supported", __func__);
        }
        if (feature.hasId) {
            out.tile.featureIds.push_back(feature.id);
        }
        out.tile.localX.push_back(points[0].first);
        out.tile.localY.push_back(points[0].second);
        const std::vector<int32_t> valueIdx = feature_value_indices(layer, feature);
        for (size_t outIdx = 0; outIdx < selectedSourceColumns.size(); ++outIdx) {
            const size_t srcIdx = selectedSourceColumns[outIdx];
            PropertyColumn& column = out.tile.columns[outIdx];
            const bool present = valueIdx[srcIdx] >= 0;
            column.present.push_back(present);
            MvtValue value;
            if (present) {
                value = layer.values[static_cast<size_t>(valueIdx[srcIdx])];
            }
            switch (column.type) {
            case ScalarType::INT_32:
                column.intValues.push_back(present ? value.intValue : 0);
                break;
            case ScalarType::FLOAT:
                column.floatValues.push_back(present ? value.floatValue : 0.0f);
                break;
            case ScalarType::BOOLEAN:
                column.boolValues.push_back(present ? value.boolValue : false);
                break;
            case ScalarType::STRING:
            default:
                column.stringValues.push_back(present ? value.stringValue : std::string());
                break;
            }
        }
    }
    return out;
}

DecodedPolygonTile decode_polygon_tile(const std::string& rawTile) {
    const std::vector<DecodedLayer> layers = decode_layers(rawTile);
    const DecodedLayer& layer = require_single_layer(layers);
    DecodedPolygonTile out;
    initialize_decoded_schema(layer, out.schema, out.tile.columns);
    out.tile.ringOffsets.push_back(0);
    for (const auto& feature : layer.features) {
        if (feature.type != GEOM_POLYGON) {
            error("%s: expected polygon MVT feature, got type=%u", __func__, feature.type);
        }
        const auto ring = decode_polygon_geometry(feature.geometry);
        if (feature.hasId) {
            out.tile.featureIds.push_back(feature.id);
            out.schema.hasIdColumn = true;
            out.schema.idIsUint64 = true;
        }
        for (const auto& pt : ring) {
            out.tile.localX.push_back(pt.first);
            out.tile.localY.push_back(pt.second);
        }
        out.tile.ringOffsets.push_back(static_cast<uint32_t>(out.tile.localX.size()));
        append_decoded_properties(layer, feature_value_indices(layer, feature), out.tile.columns);
    }
    return out;
}

uint64_t count_features(const std::string& rawTile) {
    uint64_t out = 0;
    for (const auto& layer : decode_layers(rawTile)) {
        out += layer.features.size();
    }
    return out;
}

} // namespace mvt_pmtiles
