#include "pmtiles_pyramid.hpp"

#include "hexgrid.h"
#include "mlt_utils.hpp"
#include "pmtiles_utils.hpp"
#include "simple_polygon_pmtiles.hpp"
#include "utils.h"
#include "clipper2/clipper.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <fcntl.h>
#include <functional>
#include <limits>
#include <numeric>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace pmtiles_pyramid {

namespace {

using Clipper2Lib::Area;
using Clipper2Lib::Path64;
using Clipper2Lib::Paths64;
using Clipper2Lib::Point64;
using Clipper2Lib::Union;

char infer_simple_polygon_delimiter(const std::string& line) {
    if (line.find('\t') != std::string::npos) {
        return '\t';
    }
    if (line.find(',') != std::string::npos) {
        return ',';
    }
    error("%s: could not infer polygon-table delimiter from header line", __func__);
    return '\t';
}

double compute_simple_polygon_area_abs(const std::vector<std::pair<int64_t, int64_t>>& ring) {
    if (ring.size() < 3) {
        return 0.0;
    }
    const size_t n = (ring.size() >= 2 && ring.front() == ring.back()) ? ring.size() - 1 : ring.size();
    if (n < 3) {
        return 0.0;
    }
    double twiceArea = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const size_t j = (i + 1u == n) ? 0u : (i + 1u);
        twiceArea += static_cast<double>(ring[i].first) * static_cast<double>(ring[j].second) -
            static_cast<double>(ring[i].second) * static_cast<double>(ring[j].first);
    }
    return std::abs(twiceArea) * 0.5;
}

double compute_path64_area_abs(const Path64& path) {
    return std::abs(Area(path));
}

} // namespace

SimplePolygonTableIndex::SimplePolygonTableIndex(const std::string& path,
    const SimplePolygonTableReadOptions& options,
    uint8_t sourceZoom,
    uint32_t extent) {
    TextLineReader reader(path);
    std::string line;
    if (options.icolId < 0 || options.icolX < 0 || options.icolY < 0) {
        error("%s: --icol-id, --icol-x, and --icol-y must be non-negative", __func__);
    }

    struct VertexRow {
        int64_t order = 0;
        double x = 0.0;
        double y = 0.0;
    };

    std::unordered_map<std::string, std::vector<VertexRow>> staged;
    uint64_t lineNo = 0;
    bool sawData = false;
    bool firstDataRowHandled = false;
    while (reader.getline(line)) {
        ++lineNo;
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const char delimiter = (options.delimiter != '\0') ?
            options.delimiter : infer_simple_polygon_delimiter(line);
        const std::string delims(1, delimiter);
        std::vector<std::string> fields;
        split(fields, delims, line, UINT_MAX, true, false, false, false);
        const int32_t maxRequired = std::max({options.icolId, options.icolX, options.icolY, options.icolOrder});
        if (static_cast<int32_t>(fields.size()) <= maxRequired) {
            if (!firstDataRowHandled) {
                error("%s: source polygon file row %" PRIu64 " has fewer than %d required columns",
                    __func__, lineNo, maxRequired + 1);
            }
            error("%s: invalid source polygon row %" PRIu64 ": expected at least %d columns, found %zu",
                __func__, lineNo, maxRequired + 1, fields.size());
        }
        VertexRow row;
        const bool xOk = str2double(fields[options.icolX], row.x);
        const bool yOk = str2double(fields[options.icolY], row.y);
        if (!firstDataRowHandled) {
            firstDataRowHandled = true;
            if (!xOk || !yOk) {
                continue;
            }
        } else if (!xOk || !yOk) {
            error("%s: invalid x/y token in source polygon row %" PRIu64,
                __func__, lineNo);
        }
        sawData = true;
        std::string polygonId = fields[options.icolId];
        if (polygonId.empty()) {
            error("%s: empty polygon ID in polygon table row %" PRIu64, __func__, lineNo);
        }
        auto& rows = staged[polygonId];
        if (options.icolOrder >= 0) {
            if (!str2int64(fields[options.icolOrder], row.order)) {
                error("%s: failed parsing polygon vertex order at row %" PRIu64,
                    __func__, lineNo);
            }
        } else {
            row.order = static_cast<int64_t>(rows.size());
        }
        rows.push_back(row);
    }
    if (!sawData) {
        error("%s: polygon vertex table %s contains no usable data rows", __func__, path.c_str());
    }

    for (auto& kv : staged) {
        auto& vertices = kv.second;
        std::sort(vertices.begin(), vertices.end(),
            [](const VertexRow& lhs, const VertexRow& rhs) {
                return lhs.order < rhs.order;
            });
        SimplePolygonRecord record;
        record.polygonId = kv.first;
        record.globalRing.reserve(vertices.size());
        for (size_t i = 0; i < vertices.size(); ++i) {
            if (i > 0 && vertices[i].order == vertices[i - 1].order) {
                error("%s: duplicate vertex order %" PRId64 " for polygon %s",
                    __func__, vertices[i].order, kv.first.c_str());
            }
            int64_t tileX = 0;
            int64_t tileY = 0;
            double localX = 0.0;
            double localY = 0.0;
            mlt_pmtiles::epsg3857_to_tilecoord(vertices[i].x, vertices[i].y, sourceZoom,
                tileX, tileY, localX, localY);
            const int64_t globalX = tileX * static_cast<int64_t>(extent) +
                static_cast<int64_t>(std::llround(localX * static_cast<double>(extent) / 256.0));
            const int64_t globalY = tileY * static_cast<int64_t>(extent) +
                static_cast<int64_t>(std::llround(localY * static_cast<double>(extent) / 256.0));
            record.globalRing.emplace_back(globalX, globalY);
        }
        record.area = compute_simple_polygon_area_abs(record.globalRing);
        polygons_.emplace(record.polygonId, std::move(record));
    }
}

void SimplePolygonTableIndex::add_fragment(const std::string& polygonId,
    const std::vector<std::pair<int64_t, int64_t>>& globalRing) {
    if (polygonId.empty() || globalRing.size() < 3) {
        return;
    }
    pendingFragments_[polygonId].push_back(globalRing);
}

void SimplePolygonTableIndex::finalize_repairs() {
    polygons_.clear();
    issues_.clear();
    polygons_.reserve(pendingFragments_.size());
    for (auto& kv : pendingFragments_) {
        Paths64 subject;
        subject.reserve(kv.second.size());
        for (const auto& ring : kv.second) {
            if (ring.size() < 3) {
                continue;
            }
            Path64 path;
            path.reserve(ring.size());
            for (const auto& pt : ring) {
                path.push_back(Point64(pt.first, pt.second));
            }
            if (path.size() >= 3 && std::llround(std::abs(Area(path))) != 0) {
                subject.push_back(std::move(path));
            }
        }
        if (subject.empty()) {
            issues_[kv.first] = SimplePolygonIssueReason::Degenerate;
            continue;
        }
        const Paths64 united = Union(subject, Clipper2Lib::FillRule::NonZero);
        if (united.empty()) {
            issues_[kv.first] = SimplePolygonIssueReason::Degenerate;
            continue;
        }
        SimplePolygonIssueReason issue = SimplePolygonIssueReason::None;
        size_t bestIndex = 0;
        double bestArea = -1.0;
        if (united.size() != 1u) {
            issue = SimplePolygonIssueReason::Multipolygon;
        }
        for (size_t i = 0; i < united.size(); ++i) {
            const double area = compute_path64_area_abs(united[i]);
            if (area > bestArea) {
                bestArea = area;
                bestIndex = i;
            }
        }
        const Path64& path = united[bestIndex];
        if (path.size() < 3) {
            issues_[kv.first] = (issue == SimplePolygonIssueReason::Multipolygon) ?
                SimplePolygonIssueReason::Multipolygon :
                SimplePolygonIssueReason::Degenerate;
            continue;
        }
        SimplePolygonRecord record;
        record.polygonId = kv.first;
        record.globalRing.reserve(path.size());
        for (const auto& pt : path) {
            record.globalRing.emplace_back(pt.x, pt.y);
        }
        record.area = compute_simple_polygon_area_abs(record.globalRing);
        if (record.area <= 0.0) {
            issues_[kv.first] = (issue == SimplePolygonIssueReason::Multipolygon) ?
                SimplePolygonIssueReason::Multipolygon :
                SimplePolygonIssueReason::Degenerate;
            continue;
        }
        polygons_.emplace(record.polygonId, std::move(record));
        if (issue != SimplePolygonIssueReason::None) {
            issues_[kv.first] = issue;
        }
    }
    pendingFragments_.clear();
}

const SimplePolygonRecord* SimplePolygonTableIndex::find(const std::string& polygonId) const {
    const auto it = polygons_.find(polygonId);
    if (it == polygons_.end()) {
        return nullptr;
    }
    return &it->second;
}

SimplePolygonIssueReason SimplePolygonTableIndex::issue_reason(const std::string& polygonId) const {
    const auto it = issues_.find(polygonId);
    if (it == issues_.end()) {
        return SimplePolygonIssueReason::None;
    }
    return it->second;
}

namespace {

constexpr double kSqrt3 = 1.73205080756887729353;
constexpr double kPi = 3.14159265358979323846;
constexpr uint64_t kFnvOffsetBasis = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

struct ParentBuildInput {
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t tileId = 0;
    std::vector<const mlt_pmtiles::StoredTilePayloadRef*> children;
};

struct PointParentTileCandidate {
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t tileId = 0;
    mlt_pmtiles::PointTileData data;
    std::vector<uint64_t> priorities;
    uint32_t rawFeatureCount = 0;
    size_t estimatedBytes = 0;
};

struct PolygonParentTileCandidate {
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t tileId = 0;
    mlt_pmtiles::PolygonTileData data;
    std::vector<uint64_t> priorities;
    uint32_t rawFeatureCount = 0;
    size_t estimatedBytes = 0;
    uint32_t droppedDegenerate = 0;
    uint32_t repairedMultipolygon = 0;
};

struct EncodedParentTile {
    mlt_pmtiles::EncodedTilePayload payload;
    std::vector<uint64_t> priorities;
};

struct ResolvedCanonicalField {
    size_t columnIndex = 0;
    mlt_pmtiles::ScalarType type = mlt_pmtiles::ScalarType::INT_32;
    std::string name;
};

enum class PolygonGeometryBackendKind {
    Hexgrid,
    PolygonTable,
};

struct PolygonSourceDescriptor {
    PolygonGeometryBackendKind backend = PolygonGeometryBackendKind::Hexgrid;
    std::string backendName;
    std::string hexOrientation;
    double hexGridDistScaled = 0.0;
    PolygonPriorityMode priorityMode = PolygonPriorityMode::Random;
    uint8_t sourceZoom = 0;
    uint32_t sourceExtent = 0;
    std::vector<ResolvedCanonicalField> canonicalFields;
    size_t hexQColumn = std::numeric_limits<size_t>::max();
    size_t hexRColumn = std::numeric_limits<size_t>::max();
    size_t polygonIdColumn = std::numeric_limits<size_t>::max();
    std::shared_ptr<SimplePolygonTableIndex> polygonTable;
    simple_polygon_pmtiles::SingleZoomPolygonWriterOptions writerOptions;
};

int32_t resolve_thread_count(int32_t requested) {
    if (requested > 0) {
        return requested;
    }
    const unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int32_t>(hw) : 4;
}

bool schemas_equal(const mlt_pmtiles::FeatureTableSchema& lhs,
    const mlt_pmtiles::FeatureTableSchema& rhs) {
    if (lhs.layerName != rhs.layerName ||
        lhs.extent != rhs.extent ||
        lhs.columns.size() != rhs.columns.size()) {
        return false;
    }
    for (size_t i = 0; i < lhs.columns.size(); ++i) {
        const auto& lc = lhs.columns[i];
        const auto& rc = rhs.columns[i];
        if (lc.name != rc.name || lc.type != rc.type || lc.nullable != rc.nullable) {
            return false;
        }
    }
    return true;
}

void validate_schema_equal(const mlt_pmtiles::FeatureTableSchema& expected,
    const mlt_pmtiles::FeatureTableSchema& observed,
    const char* context) {
    if (!schemas_equal(expected, observed)) {
        error("%s: inconsistent MLT schema encountered while %s", __func__, context);
    }
}

std::string read_compressed_tile_blob(flexio::FlexReader& reader,
    const pmtiles::entry_zxy& entry) {
    std::string compressed(static_cast<size_t>(entry.length), '\0');
    if (!reader.read_at(entry.offset, compressed.size(), compressed)) {
        error("%s: failed to read PMTiles tile z=%u x=%u y=%u", __func__,
            static_cast<unsigned>(entry.z), entry.x, entry.y);
    }
    return compressed;
}

std::string read_stored_tile_blob(int fd, const mlt_pmtiles::StoredTilePayloadRef& tile) {
    std::string compressed(static_cast<size_t>(tile.dataLength), '\0');
    size_t offset = 0;
    while (offset < compressed.size()) {
        const ssize_t nread = pread(fd, compressed.data() + offset,
            compressed.size() - offset, static_cast<off_t>(tile.dataOffset + offset));
        if (nread <= 0) {
            error("%s: failed to read temporary tile blob at offset %" PRIu64,
                __func__, tile.dataOffset + offset);
        }
        offset += static_cast<size_t>(nread);
    }
    return compressed;
}

uint64_t append_blob_to_store(int fd, std::atomic<uint64_t>& storeSize,
    const std::string& blob) {
    const uint64_t offset = storeSize.fetch_add(blob.size());
    size_t written = 0;
    while (written < blob.size()) {
        const ssize_t nwritten = pwrite(fd, blob.data() + written,
            blob.size() - written, static_cast<off_t>(offset + written));
        if (nwritten <= 0) {
            error("%s: failed to append temporary tile blob", __func__);
        }
        written += static_cast<size_t>(nwritten);
    }
    return offset;
}

std::vector<uint64_t> read_priority_store(int fd, const mlt_pmtiles::StoredTilePayloadRef& tile) {
    std::vector<uint64_t> priorities(tile.priorityCount, 0);
    size_t offset = 0;
    const size_t nBytes = priorities.size() * sizeof(uint64_t);
    char* data = reinterpret_cast<char*>(priorities.data());
    while (offset < nBytes) {
        const ssize_t nread = pread(fd, data + offset, nBytes - offset,
            static_cast<off_t>(tile.priorityOffset + offset));
        if (nread <= 0) {
            error("%s: failed to read temporary priority blob at offset %" PRIu64,
                __func__, tile.priorityOffset + offset);
        }
        offset += static_cast<size_t>(nread);
    }
    return priorities;
}

uint64_t append_priorities_to_store(int fd, std::atomic<uint64_t>& storeSize,
    const std::vector<uint64_t>& priorities) {
    const size_t nBytes = priorities.size() * sizeof(uint64_t);
    const uint64_t offset = storeSize.fetch_add(nBytes);
    size_t written = 0;
    const char* data = reinterpret_cast<const char*>(priorities.data());
    while (written < nBytes) {
        const ssize_t nwritten = pwrite(fd, data + written, nBytes - written,
            static_cast<off_t>(offset + written));
        if (nwritten <= 0) {
            error("%s: failed to append temporary priority blob", __func__);
        }
        written += static_cast<size_t>(nwritten);
    }
    return offset;
}

mlt_pmtiles::PointTileData make_empty_point_tile_data(
    const mlt_pmtiles::FeatureTableSchema& schema) {
    mlt_pmtiles::PointTileData out;
    out.columns.reserve(schema.columns.size());
    for (const auto& columnSchema : schema.columns) {
        out.columns.emplace_back(columnSchema.type, columnSchema.nullable);
    }
    return out;
}

mlt_pmtiles::PolygonTileData make_empty_polygon_tile_data(
    const mlt_pmtiles::FeatureTableSchema& schema) {
    mlt_pmtiles::PolygonTileData out;
    out.ringOffsets.push_back(0u);
    out.columns.reserve(schema.columns.size());
    for (const auto& columnSchema : schema.columns) {
        out.columns.emplace_back(columnSchema.type, columnSchema.nullable);
    }
    return out;
}

bool column_value_present(const mlt_pmtiles::PropertyColumn& column, size_t row) {
    if (column.present.empty()) {
        return true;
    }
    return column.present[row];
}

size_t varint_size(uint64_t value) {
    size_t size = 1;
    while (value >= 0x80u) {
        value >>= 7u;
        ++size;
    }
    return size;
}

size_t estimate_bool_rle_payload_bytes(size_t valueCount) {
    if (valueCount == 0) {
        return 0;
    }
    const size_t packedBytes = (valueCount + 7u) / 8u;
    const size_t chunkCount = (packedBytes + 127u) / 128u;
    return packedBytes + chunkCount;
}

size_t estimate_stream_bytes(uint64_t numValues, size_t payloadBytes) {
    return 2u + varint_size(numValues) + varint_size(payloadBytes) + payloadBytes;
}

size_t estimate_property_columns_bytes(const mlt_pmtiles::FeatureTableSchema& schema,
    const std::vector<mlt_pmtiles::PropertyColumn>& columns,
    const std::vector<uint32_t>* order,
    size_t rowCount) {
    auto row_at = [&](size_t i) -> uint32_t {
        return order == nullptr ? static_cast<uint32_t>(i) : (*order)[i];
    };

    size_t layerBytes = 0;
    for (size_t colIdx = 0; colIdx < schema.columns.size(); ++colIdx) {
        const auto& columnSchema = schema.columns[colIdx];
        const auto& column = columns[colIdx];
        size_t nonNullCount = 0;
        size_t payloadBytes = 0;
        size_t valueBytes = 0;
        for (size_t i = 0; i < rowCount; ++i) {
            const uint32_t row = row_at(i);
            const bool present = column_value_present(column, row);
            if (!present) {
                continue;
            }
            ++nonNullCount;
            switch (columnSchema.type) {
            case mlt_pmtiles::ScalarType::BOOLEAN:
                break;
            case mlt_pmtiles::ScalarType::INT_32:
                payloadBytes += varint_size(mlt_pmtiles::encode_zigzag32(column.intValues[row]));
                break;
            case mlt_pmtiles::ScalarType::FLOAT:
                payloadBytes += sizeof(uint32_t);
                break;
            case mlt_pmtiles::ScalarType::STRING:
                valueBytes += column.stringValues[row].size();
                payloadBytes += varint_size(column.stringValues[row].size());
                break;
            default:
                error("%s: unsupported scalar column type %d", __func__,
                    static_cast<int>(columnSchema.type));
            }
        }

        if (columnSchema.type == mlt_pmtiles::ScalarType::STRING) {
            layerBytes += varint_size(columnSchema.nullable ? 3u : 2u);
            if (columnSchema.nullable) {
                layerBytes += estimate_stream_bytes(rowCount, estimate_bool_rle_payload_bytes(rowCount));
            }
            layerBytes += estimate_stream_bytes(nonNullCount, payloadBytes);
            layerBytes += estimate_stream_bytes(0u, valueBytes);
            continue;
        }

        if (columnSchema.nullable) {
            layerBytes += estimate_stream_bytes(rowCount, estimate_bool_rle_payload_bytes(rowCount));
        }
        if (columnSchema.type == mlt_pmtiles::ScalarType::BOOLEAN) {
            layerBytes += estimate_stream_bytes(nonNullCount, estimate_bool_rle_payload_bytes(nonNullCount));
        } else {
            layerBytes += estimate_stream_bytes(nonNullCount, payloadBytes);
        }
    }

    return layerBytes;
}

size_t estimate_point_tile_bytes_impl(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PointTileData& tile,
    const std::vector<uint32_t>* order,
    size_t rowCount) {
    if (rowCount == 0) {
        return 0;
    }
    auto row_at = [&](size_t i) -> uint32_t {
        return order == nullptr ? static_cast<uint32_t>(i) : (*order)[i];
    };

    size_t layerBytes = 0;
    layerBytes += varint_size(schema.layerName.size()) + schema.layerName.size();
    layerBytes += varint_size(schema.extent);
    layerBytes += varint_size(1u + schema.columns.size());
    layerBytes += varint_size(4u);
    for (const auto& columnSchema : schema.columns) {
        switch (columnSchema.type) {
        case mlt_pmtiles::ScalarType::BOOLEAN:
        case mlt_pmtiles::ScalarType::INT_32:
        case mlt_pmtiles::ScalarType::FLOAT:
        case mlt_pmtiles::ScalarType::STRING:
            break;
        default:
            error("%s: unsupported scalar column type %d", __func__,
                static_cast<int>(columnSchema.type));
        }
        layerBytes += 1u;
        layerBytes += varint_size(columnSchema.name.size()) + columnSchema.name.size();
    }

    layerBytes += varint_size(2u);
    layerBytes += estimate_stream_bytes(rowCount, rowCount);
    size_t vertexPayload = 0;
    int32_t prevX = 0;
    int32_t prevY = 0;
    for (size_t i = 0; i < rowCount; ++i) {
        const uint32_t row = row_at(i);
        const int32_t x = tile.localX[row];
        const int32_t y = tile.localY[row];
        vertexPayload += varint_size(mlt_pmtiles::encode_zigzag32(x - prevX));
        vertexPayload += varint_size(mlt_pmtiles::encode_zigzag32(y - prevY));
        prevX = x;
        prevY = y;
    }
    layerBytes += estimate_stream_bytes(rowCount * 2u, vertexPayload);
    layerBytes += estimate_property_columns_bytes(schema, tile.columns, order, rowCount);
    return varint_size(1u + layerBytes) + 1u + layerBytes;
}

size_t estimate_point_tile_bytes_full(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PointTileData& tile) {
    return estimate_point_tile_bytes_impl(schema, tile, nullptr, tile.size());
}

size_t estimate_point_tile_bytes_prefix(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PointTileData& tile, size_t rowCount) {
    return estimate_point_tile_bytes_impl(schema, tile, nullptr, rowCount);
}

size_t estimate_polygon_tile_bytes_impl(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile,
    const std::vector<uint32_t>* order,
    size_t rowCount) {
    if (rowCount == 0) {
        return 0;
    }
    auto row_at = [&](size_t i) -> uint32_t {
        return order == nullptr ? static_cast<uint32_t>(i) : (*order)[i];
    };

    size_t layerBytes = 0;
    layerBytes += varint_size(schema.layerName.size()) + schema.layerName.size();
    layerBytes += varint_size(schema.extent);
    layerBytes += varint_size(1u + schema.columns.size());
    layerBytes += varint_size(4u);
    for (const auto& columnSchema : schema.columns) {
        switch (columnSchema.type) {
        case mlt_pmtiles::ScalarType::BOOLEAN:
        case mlt_pmtiles::ScalarType::INT_32:
        case mlt_pmtiles::ScalarType::FLOAT:
        case mlt_pmtiles::ScalarType::STRING:
            break;
        default:
            error("%s: unsupported scalar column type %d", __func__,
                static_cast<int>(columnSchema.type));
        }
        layerBytes += 1u;
        layerBytes += varint_size(columnSchema.name.size()) + columnSchema.name.size();
    }

    layerBytes += varint_size(4u);
    layerBytes += estimate_stream_bytes(rowCount, rowCount);

    size_t partPayload = 0;
    size_t ringPayload = 0;
    size_t vertexPayload = 0;
    size_t totalVertices = 0;
    int32_t prevX = 0;
    int32_t prevY = 0;
    for (size_t i = 0; i < rowCount; ++i) {
        const uint32_t row = row_at(i);
        const uint32_t beg = tile.ringOffsets[row];
        const uint32_t end = tile.ringOffsets[row + 1u];
        partPayload += varint_size(1u);
        ringPayload += varint_size(end - beg);
        totalVertices += static_cast<size_t>(end - beg);
        for (uint32_t v = beg; v < end; ++v) {
            const int32_t x = tile.localX[v];
            const int32_t y = tile.localY[v];
            vertexPayload += varint_size(mlt_pmtiles::encode_zigzag32(x - prevX));
            vertexPayload += varint_size(mlt_pmtiles::encode_zigzag32(y - prevY));
            prevX = x;
            prevY = y;
        }
    }
    layerBytes += estimate_stream_bytes(rowCount, partPayload);
    layerBytes += estimate_stream_bytes(rowCount, ringPayload);
    layerBytes += estimate_stream_bytes(totalVertices * 2u, vertexPayload);
    layerBytes += estimate_property_columns_bytes(schema, tile.columns, order, rowCount);
    return varint_size(1u + layerBytes) + 1u + layerBytes;
}

size_t estimate_polygon_tile_bytes_full(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile) {
    return estimate_polygon_tile_bytes_impl(schema, tile, nullptr, tile.size());
}

size_t estimate_polygon_tile_bytes_prefix(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile, size_t rowCount) {
    return estimate_polygon_tile_bytes_impl(schema, tile, nullptr, rowCount);
}

std::vector<uint64_t> build_seed_row_priorities(uint32_t nRows, uint64_t seed) {
    std::vector<uint64_t> priorities(nRows, 0);
    std::mt19937_64 rng(seed);
    for (uint32_t i = 0; i < nRows; ++i) {
        priorities[i] = rng();
    }
    return priorities;
}

uint32_t shrink_target_count(uint32_t current, size_t budget, size_t observed) {
    if (current <= 1u || observed == 0u) {
        return current;
    }
    const double ratio = static_cast<double>(budget) / static_cast<double>(observed);
    uint32_t next = static_cast<uint32_t>(std::floor(static_cast<double>(current) * ratio * 0.95));
    if (next >= current) {
        next = current - 1u;
    }
    return std::max<uint32_t>(1u, next);
}

nlohmann::json build_pyramid_metadata(nlohmann::json metadata,
    const std::string& outPmtiles,
    uint8_t minZoom,
    uint8_t maxZoom,
    const std::string& generator) {
    if (!metadata.is_object()) {
        metadata = nlohmann::json::object();
    }
    metadata["name"] = basename(outPmtiles, true);
    metadata["generator"] = generator;
    metadata["zoom"] = static_cast<int32_t>(maxZoom);
    if (metadata.contains("vector_layers") && metadata["vector_layers"].is_array()) {
        for (auto& layer : metadata["vector_layers"]) {
            if (!layer.is_object()) {
                continue;
            }
            layer["minzoom"] = static_cast<int32_t>(minZoom);
            layer["maxzoom"] = static_cast<int32_t>(maxZoom);
        }
    }
    if (metadata.contains("tilestats") &&
        metadata["tilestats"].contains("layers") &&
        metadata["tilestats"]["layers"].is_array()) {
        for (auto& layer : metadata["tilestats"]["layers"]) {
            if (!layer.is_object()) {
                continue;
            }
            layer["minzoom"] = static_cast<int32_t>(minZoom);
            layer["maxzoom"] = static_cast<int32_t>(maxZoom);
        }
    }
    return metadata;
}

std::vector<ParentBuildInput> enumerate_parent_tiles(
    const std::vector<mlt_pmtiles::StoredTilePayloadRef>& currentLevel) {
    std::vector<ParentBuildInput> parents;
    parents.reserve(currentLevel.size());
    std::unordered_map<uint64_t, size_t> parentIndex;
    parentIndex.reserve(currentLevel.size());

    for (const auto& child : currentLevel) {
        if (child.z == 0) {
            continue;
        }
        const uint8_t pz = static_cast<uint8_t>(child.z - 1u);
        const uint32_t px = child.x / 2u;
        const uint32_t py = child.y / 2u;
        const uint64_t tileId = pmtiles::zxy_to_tileid(pz, px, py);
        auto [it, inserted] = parentIndex.emplace(tileId, parents.size());
        if (inserted) {
            ParentBuildInput parent;
            parent.z = pz;
            parent.x = px;
            parent.y = py;
            parent.tileId = tileId;
            parents.push_back(std::move(parent));
            it->second = parents.size() - 1u;
        }
        parents[it->second].children.push_back(&child);
    }
    return parents;
}

void parallel_for_tiles(size_t n, int32_t threads,
    const std::function<void(size_t)>& fn) {
    if (n == 0) {
        return;
    }
    if (threads > 1 && n > 1) {
        tbb::global_control limit(tbb::global_control::max_allowed_parallelism,
            static_cast<size_t>(threads));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, n),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    fn(i);
                }
            });
    } else {
        for (size_t i = 0; i < n; ++i) {
            fn(i);
        }
    }
}

PointParentTileCandidate build_point_parent_candidate(const ParentBuildInput& input,
    const mlt_pmtiles::FeatureTableSchema& schema,
    int blobFd,
    int priorityFd) {
    PointParentTileCandidate candidate;
    candidate.z = input.z;
    candidate.x = input.x;
    candidate.y = input.y;
    candidate.tileId = input.tileId;
    candidate.data = make_empty_point_tile_data(schema);
    struct ChildSequence {
        mlt_pmtiles::DecodedPointTile decoded;
        std::vector<uint64_t> priorities;
        std::vector<uint32_t> order;
        bool useIdentityOrder = false;
        size_t totalRows = 0;
    };
    std::vector<ChildSequence> children;
    children.reserve(input.children.size());
    size_t totalRows = 0;
    for (const auto* child : input.children) {
        const std::string compressed = read_stored_tile_blob(blobFd, *child);
        const std::string raw = mlt_pmtiles::gzip_decompress(compressed);
        ChildSequence childSeq;
        childSeq.decoded = mlt_pmtiles::decode_point_tile(raw);
        validate_schema_equal(schema, childSeq.decoded.schema, "building point pyramid parent tiles");
        childSeq.priorities = read_priority_store(priorityFd, *child);
        childSeq.totalRows = childSeq.decoded.tile.size();
        if (childSeq.priorities.size() != childSeq.totalRows) {
            error("%s: priority count mismatch for child tile z=%u x=%u y=%u",
                __func__, static_cast<unsigned>(child->z), child->x, child->y);
        }
        if (child->prioritiesSorted) {
            childSeq.useIdentityOrder = true;
        } else {
            childSeq.order.resize(childSeq.totalRows);
            std::iota(childSeq.order.begin(), childSeq.order.end(), 0u);
            std::sort(childSeq.order.begin(), childSeq.order.end(),
                [&](uint32_t lhs, uint32_t rhs) {
                    return childSeq.priorities[lhs] < childSeq.priorities[rhs];
                });
        }
        totalRows += childSeq.totalRows;
        children.push_back(std::move(childSeq));
    }
    candidate.priorities.reserve(totalRows);
    candidate.data.localX.reserve(totalRows);
    candidate.data.localY.reserve(totalRows);
    for (auto& column : candidate.data.columns) {
        if (column.nullable) {
            column.present.reserve(totalRows);
        }
        switch (column.type) {
        case mlt_pmtiles::ScalarType::BOOLEAN:
            column.boolValues.reserve(totalRows);
            break;
        case mlt_pmtiles::ScalarType::INT_32:
            column.intValues.reserve(totalRows);
            break;
        case mlt_pmtiles::ScalarType::FLOAT:
            column.floatValues.reserve(totalRows);
            break;
        case mlt_pmtiles::ScalarType::STRING:
            column.stringValues.reserve(totalRows);
            break;
        default:
            error("%s: unsupported scalar column type %d", __func__,
                static_cast<int>(column.type));
        }
    }

    struct HeapItem {
        uint64_t priority = 0;
        size_t childIndex = 0;
        size_t position = 0;
    };
    struct HeapGreater {
        bool operator()(const HeapItem& lhs, const HeapItem& rhs) const {
            if (lhs.priority != rhs.priority) {
                return lhs.priority > rhs.priority;
            }
            if (lhs.childIndex != rhs.childIndex) {
                return lhs.childIndex > rhs.childIndex;
            }
            return lhs.position > rhs.position;
        }
    };
    auto child_row_at = [&](const ChildSequence& childSeq, size_t pos) -> uint32_t {
        return childSeq.useIdentityOrder ? static_cast<uint32_t>(pos) : childSeq.order[pos];
    };
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapGreater> heap;
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i].totalRows == 0) {
            continue;
        }
        const uint32_t row = child_row_at(children[i], 0);
        heap.push(HeapItem{children[i].priorities[row], i, 0});
    }
    while (!heap.empty()) {
        const HeapItem item = heap.top();
        heap.pop();
        const auto* child = input.children[item.childIndex];
        const ChildSequence& childSeq = children[item.childIndex];
        const uint32_t row = child_row_at(childSeq, item.position);
        mlt_pmtiles::append_child_row_to_parent_tile(
            childSeq.decoded, row, child->x, child->y, input.x, input.y, candidate.data);
        candidate.priorities.push_back(item.priority);
        const size_t nextPos = item.position + 1u;
        if (nextPos < childSeq.totalRows) {
            const uint32_t nextRow = child_row_at(childSeq, nextPos);
            heap.push(HeapItem{childSeq.priorities[nextRow], item.childIndex, nextPos});
        }
    }

    candidate.rawFeatureCount = static_cast<uint32_t>(candidate.data.size());
    if (candidate.rawFeatureCount > 0) {
        candidate.estimatedBytes = estimate_point_tile_bytes_full(schema, candidate.data);
    }
    return candidate;
}

std::optional<EncodedParentTile> encode_point_parent_candidate(
    const PointParentTileCandidate& candidate,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const BuildOptions& options,
    double levelRatio) {
    if (candidate.rawFeatureCount == 0) {
        return std::nullopt;
    }

    uint32_t targetCount = static_cast<uint32_t>(
        std::floor(levelRatio * static_cast<double>(candidate.rawFeatureCount)));
    targetCount = std::min<uint32_t>(targetCount, static_cast<uint32_t>(options.maxTileFeatures));
    if (targetCount == 0) {
        targetCount = 1;
    }
    targetCount = std::min<uint32_t>(targetCount, candidate.rawFeatureCount);

    const size_t estimatedBudget = static_cast<size_t>(
        std::floor(static_cast<double>(options.maxTileBytes) * options.scaleFactorCompression));

    while (targetCount > 0) {
        const size_t estimatedBytes = estimate_point_tile_bytes_prefix(
            schema, candidate.data, targetCount);
        if (targetCount > 1u && estimatedBytes > estimatedBudget) {
            targetCount = shrink_target_count(targetCount, estimatedBudget, estimatedBytes);
            continue;
        }

        const std::string raw = mlt_pmtiles::encode_point_tile_prefix(
            schema, candidate.data, targetCount, nullptr);
        std::string compressed = mlt_pmtiles::gzip_compress(raw);
        if (compressed.size() <= static_cast<size_t>(options.maxTileBytes) || targetCount == 1u) {
            if (compressed.size() > static_cast<size_t>(options.maxTileBytes) && targetCount == 1u) {
                warning("%s: tile z=%u x=%u y=%u still exceeds --max-tile-bytes with one row (%zu bytes)",
                    __func__, static_cast<unsigned>(candidate.z), candidate.x, candidate.y, compressed.size());
            }
            EncodedParentTile encoded;
            encoded.payload.z = candidate.z;
            encoded.payload.x = candidate.x;
            encoded.payload.y = candidate.y;
            encoded.payload.tileId = candidate.tileId;
            encoded.payload.featureCount = targetCount;
            encoded.payload.compressedData = std::move(compressed);
            encoded.priorities.assign(candidate.priorities.begin(),
                candidate.priorities.begin() + targetCount);
            return encoded;
        }
        targetCount = shrink_target_count(targetCount,
            static_cast<size_t>(options.maxTileBytes), compressed.size());
    }

    return std::nullopt;
}

uint64_t fnv1a_append(uint64_t state, const void* data, size_t nBytes) {
    const auto* bytes = static_cast<const uint8_t*>(data);
    for (size_t i = 0; i < nBytes; ++i) {
        state ^= static_cast<uint64_t>(bytes[i]);
        state *= kFnvPrime;
    }
    return state;
}

uint64_t hash_priority_key(const std::string& key) {
    return fnv1a_append(kFnvOffsetBasis, key.data(), key.size());
}

double compute_ring_area_abs(const std::vector<std::pair<double, double>>& ring) {
    if (ring.size() < 3) {
        return 0.0;
    }
    const size_t n = (ring.size() >= 2 && ring.front() == ring.back()) ? ring.size() - 1 : ring.size();
    if (n < 3) {
        return 0.0;
    }
    double twiceArea = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const size_t j = (i + 1u == n) ? 0u : (i + 1u);
        twiceArea += ring[i].first * ring[j].second - ring[i].second * ring[j].first;
    }
    return std::abs(twiceArea) * 0.5;
}

uint64_t make_random_priority(const std::string& canonicalKey) {
    return hash_priority_key(canonicalKey);
}

uint64_t make_area_priority(const std::string& canonicalKey, double area) {
    if (!(area > 0.0) || !std::isfinite(area)) {
        return make_random_priority(canonicalKey);
    }
    const uint64_t areaBits = std::bit_cast<uint64_t>(area);
    const uint64_t major = (~areaBits) & 0xffffffffffff0000ull;
    return major | (hash_priority_key(canonicalKey) & 0xffffull);
}

PolygonGeometryBackendKind parse_polygon_geometry_backend_kind(const std::string& name) {
    if (name == "hexgrid") {
        return PolygonGeometryBackendKind::Hexgrid;
    }
    if (name == "polygon_table") {
        return PolygonGeometryBackendKind::PolygonTable;
    }
    error("%s: unsupported polygon geometry backend %s", __func__, name.c_str());
    return PolygonGeometryBackendKind::Hexgrid;
}

size_t find_schema_column_by_name(const mlt_pmtiles::FeatureTableSchema& schema,
    const std::string& name) {
    for (size_t i = 0; i < schema.columns.size(); ++i) {
        if (schema.columns[i].name == name) {
            return i;
        }
    }
    return schema.columns.size();
}

std::string canonical_key_for_polygon_row(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile,
    size_t row,
    const PolygonSourceDescriptor& descriptor) {
    std::string key;
    key.reserve(descriptor.canonicalFields.size() * 16u);
    for (const auto& field : descriptor.canonicalFields) {
        const auto& column = tile.columns[field.columnIndex];
        const bool present = column_value_present(column, row);
        key.push_back(static_cast<char>(field.type));
        key.push_back(present ? '\x01' : '\x00');
        key.push_back('\x1f');
        key.append(field.name);
        key.push_back('\x1e');
        if (!present) {
            key.push_back('\x00');
            continue;
        }
        switch (field.type) {
        case mlt_pmtiles::ScalarType::INT_32: {
            const int32_t value = column.intValues[row];
            key.append(reinterpret_cast<const char*>(&value), sizeof(value));
            break;
        }
        case mlt_pmtiles::ScalarType::FLOAT: {
            const uint32_t bits = std::bit_cast<uint32_t>(column.floatValues[row]);
            key.append(reinterpret_cast<const char*>(&bits), sizeof(bits));
            break;
        }
        case mlt_pmtiles::ScalarType::STRING: {
            const auto& value = column.stringValues[row];
            const uint32_t len = static_cast<uint32_t>(value.size());
            key.append(reinterpret_cast<const char*>(&len), sizeof(len));
            key.append(value);
            break;
        }
        default:
            error("%s: unsupported canonical field type for polygon pyramid", __func__);
        }
        key.push_back('\x1d');
    }
    return key;
}

int32_t require_polygon_row_int(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile,
    size_t row,
    size_t columnIndex) {
    if (columnIndex >= schema.columns.size()) {
        error("%s: polygon canonical column index out of range", __func__);
    }
    if (schema.columns[columnIndex].type != mlt_pmtiles::ScalarType::INT_32) {
        error("%s: polygon canonical column %s must be INT_32", __func__,
            schema.columns[columnIndex].name.c_str());
    }
    const auto& column = tile.columns[columnIndex];
    if (!column_value_present(column, row)) {
        error("%s: polygon canonical column %s is unexpectedly null", __func__,
            schema.columns[columnIndex].name.c_str());
    }
    return column.intValues[row];
}

std::string require_polygon_row_string(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile,
    size_t row,
    size_t columnIndex) {
    if (columnIndex >= schema.columns.size()) {
        error("%s: polygon canonical column index out of range", __func__);
    }
    if (schema.columns[columnIndex].type != mlt_pmtiles::ScalarType::STRING) {
        error("%s: polygon canonical column %s must be STRING", __func__,
            schema.columns[columnIndex].name.c_str());
    }
    const auto& column = tile.columns[columnIndex];
    if (!column_value_present(column, row)) {
        error("%s: polygon canonical column %s is unexpectedly null", __func__,
            schema.columns[columnIndex].name.c_str());
    }
    return column.stringValues[row];
}

std::vector<std::pair<double, double>> extract_polygon_row_world_ring(
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile,
    size_t row,
    uint8_t z,
    uint32_t x,
    uint32_t y) {
    if (row + 1u >= tile.ringOffsets.size()) {
        error("%s: polygon row %zu is out of range for ringOffsets", __func__, row);
    }
    const uint32_t beg = tile.ringOffsets[row];
    const uint32_t end = tile.ringOffsets[row + 1u];
    if (end < beg || end > tile.localX.size() || end > tile.localY.size()) {
        error("%s: invalid ring offset range [%u, %u)", __func__, beg, end);
    }
    double tileMinX = 0.0;
    double tileMaxY = 0.0;
    double tileMaxX = 0.0;
    double tileMinY = 0.0;
    mlt_pmtiles::tilecoord_to_epsg3857(x, y, 0.0, 0.0, z, tileMinX, tileMaxY);
    mlt_pmtiles::tilecoord_to_epsg3857(x, y, 256.0, 256.0, z, tileMaxX, tileMinY);
    const double tileWidth = tileMaxX - tileMinX;
    const double tileHeight = tileMaxY - tileMinY;
    std::vector<std::pair<double, double>> ring;
    ring.reserve(end - beg);
    for (uint32_t i = beg; i < end; ++i) {
        const double worldX = tileMinX +
            (static_cast<double>(tile.localX[i]) / static_cast<double>(schema.extent)) * tileWidth;
        const double worldY = tileMaxY -
            (static_cast<double>(tile.localY[i]) / static_cast<double>(schema.extent)) * tileHeight;
        ring.emplace_back(worldX, worldY);
    }
    return ring;
}

simple_polygon_pmtiles::PolygonFeatureProperties extract_polygon_properties(
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile,
    size_t row) {
    simple_polygon_pmtiles::PolygonFeatureProperties props;
    for (size_t colIdx = 0; colIdx < schema.columns.size(); ++colIdx) {
        const auto& colSchema = schema.columns[colIdx];
        const auto& col = tile.columns[colIdx];
        const bool present = column_value_present(col, row);
        switch (colSchema.type) {
        case mlt_pmtiles::ScalarType::INT_32:
            props.intValues.push_back(present ? std::optional<int32_t>(col.intValues[row]) : std::nullopt);
            break;
        case mlt_pmtiles::ScalarType::FLOAT:
            props.floatValues.push_back(present ? std::optional<float>(col.floatValues[row]) : std::nullopt);
            break;
        case mlt_pmtiles::ScalarType::STRING:
            props.stringValues.push_back(present ? std::optional<std::string>(col.stringValues[row]) : std::nullopt);
            break;
        default:
            error("%s: unsupported polygon property type %d", __func__,
                static_cast<int>(colSchema.type));
        }
    }
    return props;
}

std::vector<std::pair<double, double>> build_pointy_hex_ring(double centerX,
    double centerY,
    double hexGridDistScaled) {
    const double edge = hexGridDistScaled / kSqrt3;
    std::vector<std::pair<double, double>> out;
    out.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double angle = (30.0 + 60.0 * static_cast<double>(i)) * kPi / 180.0;
        out.emplace_back(centerX + edge * std::cos(angle),
            centerY + edge * std::sin(angle));
    }
    return out;
}

std::vector<std::pair<double, double>> reconstruct_polygon_ring(
    const PolygonSourceDescriptor& descriptor,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PolygonTileData& tile,
    size_t row) {
    switch (descriptor.backend) {
    case PolygonGeometryBackendKind::Hexgrid: {
        if (descriptor.hexOrientation != "pointy") {
            error("%s: unsupported hex orientation %s", __func__,
                descriptor.hexOrientation.c_str());
        }
        const int32_t hexQ = require_polygon_row_int(schema, tile, row, descriptor.hexQColumn);
        const int32_t hexR = require_polygon_row_int(schema, tile, row, descriptor.hexRColumn);
        HexGrid scaledGrid(descriptor.hexGridDistScaled / kSqrt3, true);
        double centerX = 0.0;
        double centerY = 0.0;
        scaledGrid.axial_to_cart(centerX, centerY, hexQ, hexR);
        return build_pointy_hex_ring(centerX, centerY, descriptor.hexGridDistScaled);
    }
    case PolygonGeometryBackendKind::PolygonTable: {
        if (!descriptor.polygonTable) {
            error("%s: polygon_table backend requires a loaded polygon table", __func__);
        }
        const std::string polygonId = require_polygon_row_string(
            schema, tile, row, descriptor.polygonIdColumn);
        const SimplePolygonRecord* record = descriptor.polygonTable->find(polygonId);
        if (record == nullptr) {
            error("%s: polygon ID %s was not found in the source geometry table",
                __func__, polygonId.c_str());
        }
        std::vector<std::pair<double, double>> ring;
        ring.reserve(record->globalRing.size());
        const double worldPerGlobal =
            40075016.68557849 /
            (static_cast<double>(uint64_t{1} << descriptor.sourceZoom) * static_cast<double>(descriptor.sourceExtent));
        for (const auto& pt : record->globalRing) {
            const double worldX = static_cast<double>(pt.first) * worldPerGlobal - 20037508.342789244;
            const double worldY = 20037508.342789244 - static_cast<double>(pt.second) * worldPerGlobal;
            ring.emplace_back(worldX, worldY);
        }
        return ring;
    }
    }
    error("%s: unsupported polygon geometry backend", __func__);
    return {};
}

PolygonSourceDescriptor resolve_polygon_source_descriptor(
    const nlohmann::json& metadata,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const BuildOptions& options) {
    if (!metadata.is_object()) {
        error("%s: polygon PMTiles metadata must be a JSON object", __func__);
    }
    if (!metadata.contains("polygon_source") || !metadata["polygon_source"].is_object()) {
        error("%s: polygon PMTiles metadata is missing polygon_source", __func__);
    }
    if (!metadata.contains("polygon_pyramid_hint") || !metadata["polygon_pyramid_hint"].is_object()) {
        error("%s: polygon PMTiles metadata is missing polygon_pyramid_hint", __func__);
    }
    const auto& source = metadata["polygon_source"];
    const auto& hint = metadata["polygon_pyramid_hint"];

    PolygonSourceDescriptor out;
    out.backendName = source.value("geometry_backend",
        source.value("family", std::string()));
    if (out.backendName.empty()) {
        error("%s: polygon_source.geometry_backend (or family) is required", __func__);
    }
    out.backend = parse_polygon_geometry_backend_kind(out.backendName);
    out.priorityMode = options.polygonPriorityMode;
    out.writerOptions.extent = schema.extent;
    out.writerOptions.tileBufferPixels = hint.value("buffer_screen_px", 5.0);
    out.writerOptions.clipScale = 1024;
    out.writerOptions.coordScale = metadata.value("coord_scale", 1.0);
    std::vector<std::string> canonicalFieldNames;
    if (hint.contains("canonical_id_fields") && hint["canonical_id_fields"].is_array()) {
        for (const auto& item : hint["canonical_id_fields"]) {
            if (!item.is_string()) {
                error("%s: canonical_id_fields entries must be strings", __func__);
            }
            canonicalFieldNames.push_back(item.get<std::string>());
        }
    }
    if (canonicalFieldNames.empty() && out.backend == PolygonGeometryBackendKind::PolygonTable) {
        canonicalFieldNames.push_back(options.polygonIdColumn);
    }
    if (canonicalFieldNames.empty()) {
        error("%s: polygon_pyramid_hint.canonical_id_fields is required", __func__);
    }
    for (const auto& name : canonicalFieldNames) {
        bool found = false;
        for (size_t colIdx = 0; colIdx < schema.columns.size(); ++colIdx) {
            if (schema.columns[colIdx].name == name) {
                out.canonicalFields.push_back(ResolvedCanonicalField{
                    colIdx, schema.columns[colIdx].type, name
                });
                found = true;
                break;
            }
        }
        if (!found) {
            error("%s: canonical polygon ID field %s was not found in the tile schema",
                __func__, name.c_str());
        }
    }

    if (out.backend == PolygonGeometryBackendKind::Hexgrid) {
        if (!source.contains("parameters") || !source["parameters"].is_object()) {
            error("%s: polygon_source.parameters is required for hexgrid polygon pyramids", __func__);
        }
        const auto& params = source["parameters"];
        out.hexGridDistScaled = params.value("hex_grid_dist_scaled", -1.0);
        if (!(out.hexGridDistScaled > 0.0)) {
            error("%s: polygon_source.parameters.hex_grid_dist_scaled must be positive", __func__);
        }
        out.hexOrientation = params.value("hex_orientation", std::string("pointy"));
        out.hexQColumn = find_schema_column_by_name(schema, "hex_q");
        out.hexRColumn = find_schema_column_by_name(schema, "hex_r");
        if (out.hexQColumn == std::numeric_limits<size_t>::max() ||
            out.hexRColumn == std::numeric_limits<size_t>::max()) {
            error("%s: hexgrid polygon pyramids require schema columns hex_q and hex_r", __func__);
        }
    } else if (out.backend == PolygonGeometryBackendKind::PolygonTable) {
        if (out.canonicalFields.size() != 1u ||
            out.canonicalFields.front().type != mlt_pmtiles::ScalarType::STRING) {
            error("%s: polygon_table backend requires exactly one STRING canonical ID field",
                __func__);
        }
        out.polygonIdColumn = out.canonicalFields.front().columnIndex;
    } else {
        error("%s: polygon pyramid builder currently supports only known polygon geometry backends", __func__);
    }

    return out;
}

std::vector<uint64_t> build_polygon_seed_priorities(
    const mlt_pmtiles::DecodedPolygonTile& decoded,
    const PolygonSourceDescriptor& descriptor) {
    std::vector<uint64_t> priorities(decoded.tile.size(), 0);
    for (size_t row = 0; row < decoded.tile.size(); ++row) {
        const std::string key = canonical_key_for_polygon_row(
            decoded.schema, decoded.tile, row, descriptor);
        if (descriptor.priorityMode == PolygonPriorityMode::Area) {
            double area = 0.0;
            if (descriptor.backend == PolygonGeometryBackendKind::PolygonTable) {
                const std::string polygonId = require_polygon_row_string(
                    decoded.schema, decoded.tile, row, descriptor.polygonIdColumn);
                const SimplePolygonRecord* record = descriptor.polygonTable ?
                    descriptor.polygonTable->find(polygonId) : nullptr;
                if (record != nullptr) {
                    area = record->area;
                } else {
                    priorities[row] = make_random_priority(key);
                    continue;
                }
            } else {
                const auto ring = reconstruct_polygon_ring(descriptor, decoded.schema, decoded.tile, row);
                area = compute_ring_area_abs(ring);
            }
            priorities[row] = make_area_priority(key, area);
        } else {
            priorities[row] = make_random_priority(key);
        }
    }
    return priorities;
}

std::vector<std::pair<int64_t, int64_t>> extract_polygon_row_global_ring(
    const mlt_pmtiles::DecodedPolygonTile& decoded,
    size_t row,
    uint8_t z,
    uint32_t x,
    uint32_t y) {
    if (row + 1u >= decoded.tile.ringOffsets.size()) {
        error("%s: polygon row %zu is out of range for ringOffsets", __func__, row);
    }
    const uint32_t beg = decoded.tile.ringOffsets[row];
    const uint32_t end = decoded.tile.ringOffsets[row + 1u];
    std::vector<std::pair<int64_t, int64_t>> ring;
    ring.reserve(end - beg);
    const int64_t tileOffsetX = static_cast<int64_t>(x) * static_cast<int64_t>(decoded.schema.extent);
    const int64_t tileOffsetY = static_cast<int64_t>(y) * static_cast<int64_t>(decoded.schema.extent);
    for (uint32_t i = beg; i < end; ++i) {
        ring.emplace_back(tileOffsetX + static_cast<int64_t>(decoded.tile.localX[i]),
            tileOffsetY + static_cast<int64_t>(decoded.tile.localY[i]));
    }
    return ring;
}

PolygonParentTileCandidate build_polygon_parent_candidate(const ParentBuildInput& input,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const PolygonSourceDescriptor& descriptor,
    int blobFd,
    int priorityFd) {
    PolygonParentTileCandidate candidate;
    candidate.z = input.z;
    candidate.x = input.x;
    candidate.y = input.y;
    candidate.tileId = input.tileId;
    candidate.data = make_empty_polygon_tile_data(schema);

    struct ChildSequence {
        mlt_pmtiles::DecodedPolygonTile decoded;
        std::vector<uint64_t> priorities;
        std::vector<uint32_t> order;
        bool useIdentityOrder = false;
        size_t totalRows = 0;
    };
    std::vector<ChildSequence> children;
    children.reserve(input.children.size());
    size_t totalRows = 0;
    for (const auto* child : input.children) {
        const std::string compressed = read_stored_tile_blob(blobFd, *child);
        const std::string raw = mlt_pmtiles::gzip_decompress(compressed);
        ChildSequence childSeq;
        childSeq.decoded = mlt_pmtiles::decode_polygon_tile(raw);
        validate_schema_equal(schema, childSeq.decoded.schema, "building polygon pyramid parent tiles");
        childSeq.priorities = read_priority_store(priorityFd, *child);
        childSeq.totalRows = childSeq.decoded.tile.size();
        if (childSeq.priorities.size() != childSeq.totalRows) {
            error("%s: priority count mismatch for polygon child tile z=%u x=%u y=%u",
                __func__, static_cast<unsigned>(child->z), child->x, child->y);
        }
        if (child->prioritiesSorted) {
            childSeq.useIdentityOrder = true;
        } else {
            childSeq.order.resize(childSeq.totalRows);
            std::iota(childSeq.order.begin(), childSeq.order.end(), 0u);
            std::sort(childSeq.order.begin(), childSeq.order.end(),
                [&](uint32_t lhs, uint32_t rhs) {
                    return childSeq.priorities[lhs] < childSeq.priorities[rhs];
                });
        }
        totalRows += childSeq.totalRows;
        children.push_back(std::move(childSeq));
    }

    struct HeapItem {
        uint64_t priority = 0;
        size_t childIndex = 0;
        size_t position = 0;
    };
    struct HeapGreater {
        bool operator()(const HeapItem& lhs, const HeapItem& rhs) const {
            if (lhs.priority != rhs.priority) {
                return lhs.priority > rhs.priority;
            }
            if (lhs.childIndex != rhs.childIndex) {
                return lhs.childIndex > rhs.childIndex;
            }
            return lhs.position > rhs.position;
        }
    };
    auto child_row_at = [&](const ChildSequence& childSeq, size_t pos) -> uint32_t {
        return childSeq.useIdentityOrder ? static_cast<uint32_t>(pos) : childSeq.order[pos];
    };
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapGreater> heap;
    for (size_t i = 0; i < children.size(); ++i) {
        if (children[i].totalRows == 0) {
            continue;
        }
        const uint32_t row = child_row_at(children[i], 0);
        heap.push(HeapItem{children[i].priorities[row], i, 0});
    }

    std::unordered_set<std::string> seen;
    seen.reserve(totalRows);
    while (!heap.empty()) {
        const HeapItem item = heap.top();
        heap.pop();
        const ChildSequence& childSeq = children[item.childIndex];
        const uint32_t row = child_row_at(childSeq, item.position);
        const std::string key = canonical_key_for_polygon_row(
            childSeq.decoded.schema, childSeq.decoded.tile, row, descriptor);
        if (seen.insert(key).second) {
            const auto props = extract_polygon_properties(childSeq.decoded.schema,
                childSeq.decoded.tile, row);
            auto writerOptions = descriptor.writerOptions;
            writerOptions.zoom = input.z;
            size_t appended = 0;
            bool skippedForSourceIssue = false;
            if (descriptor.backend == PolygonGeometryBackendKind::PolygonTable) {
                const std::string polygonId = require_polygon_row_string(
                    childSeq.decoded.schema, childSeq.decoded.tile, row, descriptor.polygonIdColumn);
                const SimplePolygonIssueReason issue = descriptor.polygonTable ?
                    descriptor.polygonTable->issue_reason(polygonId) :
                    SimplePolygonIssueReason::None;
                if (issue == SimplePolygonIssueReason::Multipolygon) {
                    ++candidate.repairedMultipolygon;
                }
                const SimplePolygonRecord* record = descriptor.polygonTable ?
                    descriptor.polygonTable->find(polygonId) : nullptr;
                if (record == nullptr) {
                    ++candidate.droppedDegenerate;
                    skippedForSourceIssue = true;
                }
                if (!skippedForSourceIssue) {
                    appended = simple_polygon_pmtiles::append_simple_polygon_global_feature_to_tile(
                        candidate.data, schema, record->globalRing, props, input.x, input.y,
                        descriptor.sourceZoom, writerOptions);
                }
            } else {
                auto ring = reconstruct_polygon_ring(descriptor, childSeq.decoded.schema,
                    childSeq.decoded.tile, row);
                appended = simple_polygon_pmtiles::append_simple_polygon_feature_to_tile(
                    candidate.data, schema, ring, props, input.x, input.y, writerOptions);
            }
            if (!skippedForSourceIssue && appended == 0) {
                ++candidate.droppedDegenerate;
            }
            for (size_t i = 0; i < appended; ++i) {
                candidate.priorities.push_back(item.priority);
            }
        }
        const size_t nextPos = item.position + 1u;
        if (nextPos < childSeq.totalRows) {
            const uint32_t nextRow = child_row_at(childSeq, nextPos);
            heap.push(HeapItem{childSeq.priorities[nextRow], item.childIndex, nextPos});
        }
    }

    candidate.rawFeatureCount = static_cast<uint32_t>(candidate.data.size());
    if (candidate.rawFeatureCount > 0) {
        candidate.estimatedBytes = estimate_polygon_tile_bytes_full(schema, candidate.data);
    }
    return candidate;
}

std::optional<EncodedParentTile> encode_polygon_parent_candidate(
    const PolygonParentTileCandidate& candidate,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const BuildOptions& options,
    double levelRatio) {
    if (candidate.rawFeatureCount == 0) {
        return std::nullopt;
    }

    uint32_t targetCount = static_cast<uint32_t>(
        std::floor(levelRatio * static_cast<double>(candidate.rawFeatureCount)));
    targetCount = std::min<uint32_t>(targetCount, static_cast<uint32_t>(options.maxTileFeatures));
    if (targetCount == 0) {
        targetCount = 1;
    }
    targetCount = std::min<uint32_t>(targetCount, candidate.rawFeatureCount);

    const size_t estimatedBudget = static_cast<size_t>(
        std::floor(static_cast<double>(options.maxTileBytes) * options.scaleFactorCompression));

    while (targetCount > 0) {
        const size_t estimatedBytes = estimate_polygon_tile_bytes_prefix(
            schema, candidate.data, targetCount);
        if (targetCount > 1u && estimatedBytes > estimatedBudget) {
            targetCount = shrink_target_count(targetCount, estimatedBudget, estimatedBytes);
            continue;
        }

        const std::string raw = mlt_pmtiles::encode_polygon_tile_prefix(
            schema, candidate.data, targetCount, nullptr);
        std::string compressed = mlt_pmtiles::gzip_compress(raw);
        if (compressed.size() <= static_cast<size_t>(options.maxTileBytes) || targetCount == 1u) {
            if (compressed.size() > static_cast<size_t>(options.maxTileBytes) && targetCount == 1u) {
                warning("%s: polygon tile z=%u x=%u y=%u still exceeds --max-tile-bytes with one feature (%zu bytes)",
                    __func__, static_cast<unsigned>(candidate.z), candidate.x, candidate.y, compressed.size());
            }
            EncodedParentTile encoded;
            encoded.payload.z = candidate.z;
            encoded.payload.x = candidate.x;
            encoded.payload.y = candidate.y;
            encoded.payload.tileId = candidate.tileId;
            encoded.payload.featureCount = targetCount;
            encoded.payload.compressedData = std::move(compressed);
            encoded.priorities.assign(candidate.priorities.begin(),
                candidate.priorities.begin() + targetCount);
            return encoded;
        }
        targetCount = shrink_target_count(targetCount,
            static_cast<size_t>(options.maxTileBytes), compressed.size());
    }

    return std::nullopt;
}

void validate_common_pyramid_inputs(const std::string& inPmtiles,
    const std::string& outPmtiles,
    const BuildOptions& options,
    const char* fn) {
    if (outPmtiles.empty() || outPmtiles == "-") {
        error("%s: build pyramid requires a concrete output PMTiles path", fn);
    }
    if (options.maxTileBytes <= 0) {
        error("%s: maxTileBytes must be positive", fn);
    }
    if (options.maxTileFeatures <= 0) {
        error("%s: maxTileFeatures must be positive", fn);
    }
    if (options.scaleFactorCompression <= 0.0) {
        error("%s: scaleFactorCompression must be positive", fn);
    }
    if (inPmtiles == outPmtiles) {
        error("%s: input and output PMTiles paths must differ", fn);
    }
}

struct TempStores {
    std::string blobFile;
    int blobFd = -1;
    std::atomic<uint64_t> blobSize{0};
    std::string priorityFile;
    int priorityFd = -1;
    std::atomic<uint64_t> prioritySize{0};

    TempStores() = default;
    TempStores(const TempStores&) = delete;
    TempStores& operator=(const TempStores&) = delete;
    TempStores(TempStores&& other) noexcept
        : blobFile(std::move(other.blobFile)),
          blobFd(other.blobFd),
          blobSize(other.blobSize.load()),
          priorityFile(std::move(other.priorityFile)),
          priorityFd(other.priorityFd),
          prioritySize(other.prioritySize.load()) {
        other.blobFd = -1;
        other.priorityFd = -1;
        other.blobSize.store(0);
        other.prioritySize.store(0);
    }
    TempStores& operator=(TempStores&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (blobFd >= 0) {
            close(blobFd);
        }
        if (priorityFd >= 0) {
            close(priorityFd);
        }
        blobFile = std::move(other.blobFile);
        blobFd = other.blobFd;
        blobSize.store(other.blobSize.load());
        priorityFile = std::move(other.priorityFile);
        priorityFd = other.priorityFd;
        prioritySize.store(other.prioritySize.load());
        other.blobFd = -1;
        other.priorityFd = -1;
        other.blobSize.store(0);
        other.prioritySize.store(0);
        return *this;
    }

    ~TempStores() {
        if (blobFd >= 0) {
            close(blobFd);
        }
        if (priorityFd >= 0) {
            close(priorityFd);
        }
    }
};

TempStores open_temp_stores(const std::string& outPmtiles) {
    TempStores stores;
    stores.blobFile = outPmtiles + ".tiledata." + std::to_string(getpid()) + ".tmp";
    stores.blobFd = open(stores.blobFile.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (stores.blobFd < 0) {
        error("%s: cannot create temporary tile blob file %s", __func__, stores.blobFile.c_str());
    }
    stores.priorityFile = outPmtiles + ".priority." + std::to_string(getpid()) + ".tmp";
    stores.priorityFd = open(stores.priorityFile.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (stores.priorityFd < 0) {
        error("%s: cannot create temporary priority file %s", __func__, stores.priorityFile.c_str());
    }
    return stores;
}

void finalize_pyramid_archive(const mlt_pmtiles::LoadedPmtilesArchive& archive,
    const std::string& outPmtiles,
    uint8_t minZoom,
    uint8_t maxZoom,
    const std::string& generator,
    TempStores& stores,
    std::vector<mlt_pmtiles::StoredTilePayloadRef> allTileRefs) {
    mlt_pmtiles::ArchiveOptions archiveOptions;
    archiveOptions.tileType = archive.header.tile_type;
    archiveOptions.minZoom = minZoom;
    archiveOptions.maxZoom = maxZoom;
    archiveOptions.centerZoom = static_cast<uint8_t>(
        std::clamp<int32_t>(archive.header.center_zoom, static_cast<int32_t>(minZoom),
            static_cast<int32_t>(maxZoom)));
    archiveOptions.clustered = true;
    archiveOptions.hasGeographicBounds = true;
    archiveOptions.minLonE7 = archive.header.min_lon_e7;
    archiveOptions.minLatE7 = archive.header.min_lat_e7;
    archiveOptions.maxLonE7 = archive.header.max_lon_e7;
    archiveOptions.maxLatE7 = archive.header.max_lat_e7;
    archiveOptions.centerLonE7 = archive.header.center_lon_e7;
    archiveOptions.centerLatE7 = archive.header.center_lat_e7;
    archiveOptions.metadata = build_pyramid_metadata(
        archive.metadata, outPmtiles, minZoom, maxZoom, generator);

    notice("%s: writing %zu total tiles to %s",
        __func__, allTileRefs.size(), outPmtiles.c_str());
    close(stores.blobFd);
    stores.blobFd = -1;
    close(stores.priorityFd);
    stores.priorityFd = -1;
    mlt_pmtiles::write_pmtiles_archive_from_blob_file(
        outPmtiles, stores.blobFile, std::move(allTileRefs), archiveOptions);
    unlink(stores.blobFile.c_str());
    unlink(stores.priorityFile.c_str());
}

} // namespace

void build_point_pmtiles_pyramid(const std::string& inPmtiles,
    const std::string& outPmtiles,
    const BuildOptions& options) {
    validate_common_pyramid_inputs(inPmtiles, outPmtiles, options, __func__);
    const mlt_pmtiles::LoadedPmtilesArchive archive =
        mlt_pmtiles::load_pmtiles_archive(inPmtiles);
    if (archive.header.tile_type != pmtiles::TILETYPE_MLT) {
        error("%s: expected MLT PMTiles input, got tile_type=%u", __func__,
            static_cast<unsigned>(archive.header.tile_type));
    }
    if (archive.header.tile_compression != pmtiles::COMPRESSION_GZIP) {
        error("%s: only gzip-compressed PMTiles tiles are currently supported", __func__);
    }

    const uint8_t maxZoom = archive.header.max_zoom;
    const uint8_t minZoom = static_cast<uint8_t>(
        std::clamp(options.minZoom, 0, static_cast<int32_t>(maxZoom)));
    const int32_t threads = resolve_thread_count(options.threads);
    uint8_t existingMinZoom = maxZoom;
    for (const auto& entry : archive.entries) {
        existingMinZoom = std::min(existingMinZoom, entry.z);
    }
    if (existingMinZoom <= minZoom) {
        warning("%s: input already contains zoom levels down to z%u, which is <= requested min zoom z%u; nothing to do",
            __func__, static_cast<unsigned>(existingMinZoom), static_cast<unsigned>(minZoom));
        return;
    }
    TempStores stores = open_temp_stores(outPmtiles);
    if (minZoom != options.minZoom) {
        notice("%s: clamped requested min zoom %d to %u",
            __func__, options.minZoom, static_cast<unsigned>(minZoom));
    }

    notice("%s: building point PMTiles pyramid from existing z%u down to z%u with %d thread(s)",
        __func__, static_cast<unsigned>(existingMinZoom), static_cast<unsigned>(minZoom), threads);

    mlt_pmtiles::FeatureTableSchema schema;
    bool haveSchema = false;
    std::vector<mlt_pmtiles::StoredTilePayloadRef> currentLevel;
    currentLevel.reserve(archive.entries.size());
    std::vector<mlt_pmtiles::StoredTilePayloadRef> allTileRefs;
    allTileRefs.reserve(archive.entries.size() * 2u);

    for (const auto& entry : archive.entries) {
        mlt_pmtiles::StoredTilePayloadRef tile;
        tile.z = entry.z;
        tile.x = entry.x;
        tile.y = entry.y;
        tile.tileId = pmtiles::zxy_to_tileid(entry.z, entry.x, entry.y);
        const std::string compressed = read_compressed_tile_blob(*archive.reader, entry);
        tile.dataLength = static_cast<uint32_t>(compressed.size());
        tile.dataOffset = append_blob_to_store(stores.blobFd, stores.blobSize, compressed);
        if (entry.z == existingMinZoom) {
            const std::string raw = mlt_pmtiles::gzip_decompress(compressed);
            const mlt_pmtiles::DecodedPointTile decoded = mlt_pmtiles::decode_point_tile(raw);
            if (!haveSchema) {
                schema = decoded.schema;
                haveSchema = true;
            } else {
                validate_schema_equal(schema, decoded.schema, "seeding point pyramid from existing min zoom");
            }
            tile.featureCount = static_cast<uint32_t>(decoded.tile.size());
            const std::vector<uint64_t> priorities =
                build_seed_row_priorities(tile.featureCount, tile.tileId);
            tile.priorityCount = static_cast<uint32_t>(priorities.size());
            tile.priorityOffset = append_priorities_to_store(
                stores.priorityFd, stores.prioritySize, priorities);
            tile.prioritiesSorted = false;
            currentLevel.push_back(tile);
        }
        allTileRefs.push_back(std::move(tile));
    }
    if (!haveSchema || currentLevel.empty()) {
        error("%s: PMTiles archive %s has no tiles at existing min zoom %u",
            __func__, inPmtiles.c_str(), static_cast<unsigned>(existingMinZoom));
    }

    for (int32_t z = static_cast<int32_t>(existingMinZoom) - 1;
        z >= static_cast<int32_t>(minZoom); --z) {
        const std::vector<ParentBuildInput> parents = enumerate_parent_tiles(currentLevel);
        if (parents.empty()) {
            notice("%s: no parent tiles found at z%d; stopping early", __func__, z);
            break;
        }

        std::vector<PointParentTileCandidate> candidates(parents.size());
        parallel_for_tiles(parents.size(), threads, [&](size_t i) {
            candidates[i] = build_point_parent_candidate(
                parents[i], schema, stores.blobFd, stores.priorityFd);
        });

        uint32_t maxRawFeatures = 0;
        size_t maxEstimatedBytes = 0;
        for (const auto& candidate : candidates) {
            maxRawFeatures = std::max(maxRawFeatures, candidate.rawFeatureCount);
            maxEstimatedBytes = std::max(maxEstimatedBytes, candidate.estimatedBytes);
        }

        double ratioFeatures = 1.0;
        if (maxRawFeatures > static_cast<uint32_t>(options.maxTileFeatures)) {
            ratioFeatures = static_cast<double>(options.maxTileFeatures) /
                static_cast<double>(maxRawFeatures);
        }
        double ratioBytes = 1.0;
        const double estimatedBudget =
            static_cast<double>(options.maxTileBytes) * options.scaleFactorCompression;
        if (maxEstimatedBytes > static_cast<size_t>(estimatedBudget) && estimatedBudget > 0.0) {
            ratioBytes = estimatedBudget / static_cast<double>(maxEstimatedBytes);
        }
        const double levelRatio = std::min(ratioFeatures, ratioBytes);

        notice("%s: z%d parent pass: %zu tiles, max_raw=%u, max_estimated=%zu, level_ratio=%.6f",
            __func__, z, candidates.size(), maxRawFeatures, maxEstimatedBytes, levelRatio);

        std::vector<std::optional<mlt_pmtiles::StoredTilePayloadRef>> nextLevel(candidates.size());
        parallel_for_tiles(candidates.size(), threads, [&](size_t i) {
            std::optional<EncodedParentTile> encoded =
                encode_point_parent_candidate(candidates[i], schema, options, levelRatio);
            if (!encoded.has_value()) {
                return;
            }
            mlt_pmtiles::StoredTilePayloadRef stored;
            stored.tileId = encoded->payload.tileId;
            stored.z = encoded->payload.z;
            stored.x = encoded->payload.x;
            stored.y = encoded->payload.y;
            stored.featureCount = encoded->payload.featureCount;
            stored.dataLength = static_cast<uint32_t>(encoded->payload.compressedData.size());
            stored.dataOffset = append_blob_to_store(
                stores.blobFd, stores.blobSize, encoded->payload.compressedData);
            stored.priorityCount = static_cast<uint32_t>(encoded->priorities.size());
            stored.priorityOffset = append_priorities_to_store(
                stores.priorityFd, stores.prioritySize, encoded->priorities);
            stored.prioritiesSorted = true;
            nextLevel[i] = stored;
        });

        currentLevel.clear();
        for (auto& maybeTile : nextLevel) {
            if (!maybeTile.has_value()) {
                continue;
            }
            currentLevel.push_back(*maybeTile);
            allTileRefs.push_back(std::move(*maybeTile));
        }

        uint64_t levelFeatureCount = 0;
        for (const auto& tile : currentLevel) {
            levelFeatureCount += tile.featureCount;
        }
        notice("%s: z%d wrote %zu tiles with %llu retained features",
            __func__, z, currentLevel.size(),
            static_cast<unsigned long long>(levelFeatureCount));
    }

    finalize_pyramid_archive(archive, outPmtiles, minZoom, maxZoom,
        "punkst build-pyramid --point", stores, std::move(allTileRefs));
}

void build_polygon_pmtiles_pyramid(const std::string& inPmtiles,
    const std::string& outPmtiles,
    const BuildOptions& options) {
    validate_common_pyramid_inputs(inPmtiles, outPmtiles, options, __func__);
    const mlt_pmtiles::LoadedPmtilesArchive archive =
        mlt_pmtiles::load_pmtiles_archive(inPmtiles);
    if (archive.header.tile_type != pmtiles::TILETYPE_MLT) {
        error("%s: expected MLT PMTiles input, got tile_type=%u", __func__,
            static_cast<unsigned>(archive.header.tile_type));
    }
    if (archive.header.tile_compression != pmtiles::COMPRESSION_GZIP) {
        error("%s: only gzip-compressed PMTiles tiles are currently supported", __func__);
    }

    const uint8_t maxZoom = archive.header.max_zoom;
    const uint8_t minZoom = static_cast<uint8_t>(
        std::clamp(options.minZoom, 0, static_cast<int32_t>(maxZoom)));
    const int32_t threads = resolve_thread_count(options.threads);
    uint8_t existingMinZoom = maxZoom;
    for (const auto& entry : archive.entries) {
        existingMinZoom = std::min(existingMinZoom, entry.z);
    }
    if (existingMinZoom <= minZoom) {
        warning("%s: input already contains zoom levels down to z%u, which is <= requested min zoom z%u; nothing to do",
            __func__, static_cast<unsigned>(existingMinZoom), static_cast<unsigned>(minZoom));
        return;
    }
    TempStores stores = open_temp_stores(outPmtiles);
    if (minZoom != options.minZoom) {
        notice("%s: clamped requested min zoom %d to %u",
            __func__, options.minZoom, static_cast<unsigned>(minZoom));
    }

    notice("%s: building polygon PMTiles pyramid from existing z%u down to z%u with %d thread(s)",
        __func__, static_cast<unsigned>(existingMinZoom), static_cast<unsigned>(minZoom), threads);

    mlt_pmtiles::FeatureTableSchema schema;
    PolygonSourceDescriptor sourceDescriptor;
    bool haveSchema = false;
    std::vector<mlt_pmtiles::StoredTilePayloadRef> currentLevel;
    currentLevel.reserve(archive.entries.size());
    std::vector<mlt_pmtiles::StoredTilePayloadRef> allTileRefs;
    allTileRefs.reserve(archive.entries.size() * 2u);

    for (const auto& entry : archive.entries) {
        mlt_pmtiles::StoredTilePayloadRef tile;
        tile.z = entry.z;
        tile.x = entry.x;
        tile.y = entry.y;
        tile.tileId = pmtiles::zxy_to_tileid(entry.z, entry.x, entry.y);
        const std::string compressed = read_compressed_tile_blob(*archive.reader, entry);
        tile.dataLength = static_cast<uint32_t>(compressed.size());
        tile.dataOffset = append_blob_to_store(stores.blobFd, stores.blobSize, compressed);
        if (entry.z == existingMinZoom || entry.z == maxZoom) {
            const std::string raw = mlt_pmtiles::gzip_decompress(compressed);
            const mlt_pmtiles::DecodedPolygonTile decoded = mlt_pmtiles::decode_polygon_tile(raw);
            if (!haveSchema) {
                schema = decoded.schema;
                sourceDescriptor = resolve_polygon_source_descriptor(archive.metadata, schema, options);
                sourceDescriptor.sourceZoom = maxZoom;
                sourceDescriptor.sourceExtent = schema.extent;
                if (sourceDescriptor.backend == PolygonGeometryBackendKind::PolygonTable) {
                    if (!options.polygonSourcePath.empty()) {
                        SimplePolygonTableReadOptions readOptions;
                        readOptions.icolId = options.polygonSourceIcolId;
                        readOptions.icolX = options.polygonSourceIcolX;
                        readOptions.icolY = options.polygonSourceIcolY;
                        readOptions.icolOrder = options.polygonSourceIcolOrder;
                        sourceDescriptor.polygonTable = std::make_shared<SimplePolygonTableIndex>(
                            options.polygonSourcePath, readOptions, sourceDescriptor.sourceZoom,
                            sourceDescriptor.sourceExtent);
                    } else {
                        sourceDescriptor.polygonTable = std::make_shared<SimplePolygonTableIndex>();
                    }
                }
                haveSchema = true;
            } else {
                validate_schema_equal(schema, decoded.schema, "seeding polygon pyramid from existing min zoom");
            }
            if (sourceDescriptor.backend == PolygonGeometryBackendKind::PolygonTable &&
                sourceDescriptor.polygonTable &&
                options.polygonSourcePath.empty() &&
                entry.z == maxZoom) {
                for (size_t row = 0; row < decoded.tile.size(); ++row) {
                    const std::string polygonId = require_polygon_row_string(
                        decoded.schema, decoded.tile, row, sourceDescriptor.polygonIdColumn);
                    sourceDescriptor.polygonTable->add_fragment(polygonId,
                        extract_polygon_row_global_ring(decoded, row, entry.z, entry.x, entry.y));
                }
            }
            if (entry.z == existingMinZoom) {
                tile.featureCount = static_cast<uint32_t>(decoded.tile.size());
                currentLevel.push_back(tile);
            }
        }
        allTileRefs.push_back(std::move(tile));
    }
    if (sourceDescriptor.backend == PolygonGeometryBackendKind::PolygonTable &&
        sourceDescriptor.polygonTable && options.polygonSourcePath.empty()) {
        sourceDescriptor.polygonTable->finalize_repairs();
    }
    if (!haveSchema || currentLevel.empty()) {
        error("%s: PMTiles archive %s has no tiles at existing min zoom %u",
            __func__, inPmtiles.c_str(), static_cast<unsigned>(existingMinZoom));
    }
    if (sourceDescriptor.backend == PolygonGeometryBackendKind::PolygonTable &&
        (!sourceDescriptor.polygonTable || sourceDescriptor.polygonTable->size() == 0)) {
        error("%s: no source polygons could be recovered for polygon_table backend", __func__);
    }
    for (auto& tile : currentLevel) {
        const std::string compressed = read_stored_tile_blob(stores.blobFd, tile);
        const std::string raw = mlt_pmtiles::gzip_decompress(compressed);
        const mlt_pmtiles::DecodedPolygonTile decoded = mlt_pmtiles::decode_polygon_tile(raw);
        validate_schema_equal(schema, decoded.schema, "seeding polygon priorities from existing min zoom");
        const std::vector<uint64_t> priorities =
            build_polygon_seed_priorities(decoded, sourceDescriptor);
        tile.priorityCount = static_cast<uint32_t>(priorities.size());
        tile.priorityOffset = append_priorities_to_store(
            stores.priorityFd, stores.prioritySize, priorities);
        tile.prioritiesSorted = false;
    }

    for (int32_t z = static_cast<int32_t>(existingMinZoom) - 1;
        z >= static_cast<int32_t>(minZoom); --z) {
        const std::vector<ParentBuildInput> parents = enumerate_parent_tiles(currentLevel);
        if (parents.empty()) {
            notice("%s: no parent tiles found at z%d; stopping early", __func__, z);
            break;
        }

        std::vector<PolygonParentTileCandidate> candidates(parents.size());
        parallel_for_tiles(parents.size(), threads, [&](size_t i) {
            candidates[i] = build_polygon_parent_candidate(
                parents[i], schema, sourceDescriptor, stores.blobFd, stores.priorityFd);
        });

        uint32_t maxRawFeatures = 0;
        size_t maxEstimatedBytes = 0;
        uint64_t droppedDegenerate = 0;
        uint64_t repairedMultipolygon = 0;
        for (const auto& candidate : candidates) {
            maxRawFeatures = std::max(maxRawFeatures, candidate.rawFeatureCount);
            maxEstimatedBytes = std::max(maxEstimatedBytes, candidate.estimatedBytes);
            droppedDegenerate += candidate.droppedDegenerate;
            repairedMultipolygon += candidate.repairedMultipolygon;
        }

        double ratioFeatures = 1.0;
        if (maxRawFeatures > static_cast<uint32_t>(options.maxTileFeatures)) {
            ratioFeatures = static_cast<double>(options.maxTileFeatures) /
                static_cast<double>(maxRawFeatures);
        }
        double ratioBytes = 1.0;
        const double estimatedBudget =
            static_cast<double>(options.maxTileBytes) * options.scaleFactorCompression;
        if (maxEstimatedBytes > static_cast<size_t>(estimatedBudget) && estimatedBudget > 0.0) {
            ratioBytes = estimatedBudget / static_cast<double>(maxEstimatedBytes);
        }
        const double levelRatio = std::min(ratioFeatures, ratioBytes);

        notice("%s: z%d polygon parent pass: %zu tiles, max_raw=%u, max_estimated=%zu, level_ratio=%.6f",
            __func__, z, candidates.size(), maxRawFeatures, maxEstimatedBytes, levelRatio);

        std::vector<std::optional<mlt_pmtiles::StoredTilePayloadRef>> nextLevel(candidates.size());
        parallel_for_tiles(candidates.size(), threads, [&](size_t i) {
            std::optional<EncodedParentTile> encoded =
                encode_polygon_parent_candidate(candidates[i], schema, options, levelRatio);
            if (!encoded.has_value()) {
                return;
            }
            mlt_pmtiles::StoredTilePayloadRef stored;
            stored.tileId = encoded->payload.tileId;
            stored.z = encoded->payload.z;
            stored.x = encoded->payload.x;
            stored.y = encoded->payload.y;
            stored.featureCount = encoded->payload.featureCount;
            stored.dataLength = static_cast<uint32_t>(encoded->payload.compressedData.size());
            stored.dataOffset = append_blob_to_store(
                stores.blobFd, stores.blobSize, encoded->payload.compressedData);
            stored.priorityCount = static_cast<uint32_t>(encoded->priorities.size());
            stored.priorityOffset = append_priorities_to_store(
                stores.priorityFd, stores.prioritySize, encoded->priorities);
            stored.prioritiesSorted = true;
            nextLevel[i] = stored;
        });

        currentLevel.clear();
        for (auto& maybeTile : nextLevel) {
            if (!maybeTile.has_value()) {
                continue;
            }
            currentLevel.push_back(*maybeTile);
            allTileRefs.push_back(std::move(*maybeTile));
        }

        uint64_t levelFeatureCount = 0;
        for (const auto& tile : currentLevel) {
            levelFeatureCount += tile.featureCount;
        }
        notice("%s: z%d wrote %zu polygon tiles with %llu retained features",
            __func__, z, currentLevel.size(),
            static_cast<unsigned long long>(levelFeatureCount));
        if (droppedDegenerate > 0 || repairedMultipolygon > 0) {
            warning("%s: z%d encountered problematic polygons (dropped_degenerate=%llu, repaired_multipolygon=%llu)",
                __func__, z,
                static_cast<unsigned long long>(droppedDegenerate),
                static_cast<unsigned long long>(repairedMultipolygon));
        }
    }

    finalize_pyramid_archive(archive, outPmtiles, minZoom, maxZoom,
        "punkst build-pyramid --polygon", stores, std::move(allTileRefs));
}

} // namespace pmtiles_pyramid
