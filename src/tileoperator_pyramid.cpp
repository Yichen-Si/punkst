#include "tileoperator.hpp"

#include "mlt_utils.hpp"
#include "pmtiles_utils.hpp"
#include "utils.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <filesystem>
#include <fcntl.h>
#include <functional>
#include <queue>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <unistd.h>

#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>

namespace {

struct ParentBuildInput {
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t tileId = 0;
    std::vector<const mlt_pmtiles::StoredTilePayloadRef*> children;
};

struct ParentTileCandidate {
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t tileId = 0;
    mlt_pmtiles::PointTileData data;
    std::vector<uint64_t> priorities;
    uint32_t rawFeatureCount = 0;
    size_t estimatedBytes = 0;
};

struct EncodedParentTile {
    mlt_pmtiles::EncodedTilePayload payload;
    std::vector<uint64_t> priorities;
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

size_t estimate_point_tile_bytes_impl(const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::PointTileData& tile,
    const std::vector<uint32_t>* order, size_t rowCount) {
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
    for (size_t i = 0; i < rowCount; ++i) {
        const uint32_t row = row_at(i);
        vertexPayload += varint_size(mlt_pmtiles::encode_zigzag32(tile.localX[row]));
        vertexPayload += varint_size(mlt_pmtiles::encode_zigzag32(tile.localY[row]));
    }
    layerBytes += estimate_stream_bytes(rowCount * 2u, vertexPayload);

    for (size_t colIdx = 0; colIdx < schema.columns.size(); ++colIdx) {
        const auto& columnSchema = schema.columns[colIdx];
        const auto& column = tile.columns[colIdx];
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

std::vector<uint64_t> build_seed_row_priorities(uint32_t nRows, uint64_t seed) {
    std::vector<uint64_t> priorities(nRows, 0);
    std::mt19937_64 rng(seed);
    for (uint32_t i = 0; i < nRows; ++i) {
        priorities[i] = rng();
    }
    return priorities;
}

ParentTileCandidate build_parent_candidate(const ParentBuildInput& input,
    const mlt_pmtiles::FeatureTableSchema& schema, int blobFd, int priorityFd) {
    ParentTileCandidate candidate;
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
        validate_schema_equal(schema, childSeq.decoded.schema, "building pyramid parent tiles");
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

std::optional<EncodedParentTile> encode_parent_candidate(
    const ParentTileCandidate& candidate,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const TileOperator::BuildPmtilesPyramidOptions& options,
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

nlohmann::json build_pyramid_metadata(nlohmann::json metadata,
    const std::string& outPmtiles, uint8_t minZoom, uint8_t maxZoom) {
    if (!metadata.is_object()) {
        metadata = nlohmann::json::object();
    }
    metadata["name"] = basename(outPmtiles, true);
    metadata["generator"] = "punkst tile-op --build-pyramid";
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

} // namespace

void TileOperator::buildPmtilesPyramid(const std::string& inPmtiles,
    const std::string& outPmtiles,
    const BuildPmtilesPyramidOptions& options) {
    if (outPmtiles.empty() || outPmtiles == "-") {
        error("%s: buildPmtilesPyramid requires a concrete output PMTiles path", __func__);
    }
    if (options.maxTileBytes <= 0) {
        error("%s: maxTileBytes must be positive", __func__);
    }
    if (options.maxTileFeatures <= 0) {
        error("%s: maxTileFeatures must be positive", __func__);
    }
    if (options.scaleFactorCompression <= 0.0) {
        error("%s: scaleFactorCompression must be positive", __func__);
    }
    if (inPmtiles == outPmtiles) {
        error("%s: input and output PMTiles paths must differ", __func__);
    }
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
    const uint8_t minZoom = static_cast<uint8_t>(std::clamp(options.minZoom, 0, static_cast<int32_t>(maxZoom)));
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
    const std::string blobFile = outPmtiles + ".tiledata." + std::to_string(getpid()) + ".tmp";
    const int blobFd = open(blobFile.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (blobFd < 0) {
        error("%s: cannot create temporary tile blob file %s", __func__, blobFile.c_str());
    }
    std::atomic<uint64_t> blobSize{0};
    const std::string priorityFile = outPmtiles + ".priority." + std::to_string(getpid()) + ".tmp";
    const int priorityFd = open(priorityFile.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (priorityFd < 0) {
        close(blobFd);
        error("%s: cannot create temporary priority file %s", __func__, priorityFile.c_str());
    }
    std::atomic<uint64_t> prioritySize{0};
    if (minZoom != options.minZoom) {
        notice("%s: clamped requested min zoom %d to %u",
            __func__, options.minZoom, static_cast<unsigned>(minZoom));
    }

    notice("%s: building PMTiles pyramid from existing z%u down to z%u with %d thread(s)",
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
        tile.dataOffset = append_blob_to_store(blobFd, blobSize, compressed);
        if (entry.z == existingMinZoom) {
            const std::string raw = mlt_pmtiles::gzip_decompress(compressed);
            const mlt_pmtiles::DecodedPointTile decoded = mlt_pmtiles::decode_point_tile(raw);
            if (!haveSchema) {
                schema = decoded.schema;
                haveSchema = true;
            } else {
                validate_schema_equal(schema, decoded.schema, "seeding pyramid from existing min zoom");
            }
            tile.featureCount = static_cast<uint32_t>(decoded.tile.size());
            const std::vector<uint64_t> priorities =
                build_seed_row_priorities(tile.featureCount, tile.tileId);
            tile.priorityCount = static_cast<uint32_t>(priorities.size());
            tile.priorityOffset = append_priorities_to_store(priorityFd, prioritySize, priorities);
            tile.prioritiesSorted = false;
            currentLevel.push_back(tile);
        }
        allTileRefs.push_back(std::move(tile));
    }
    if (!haveSchema || currentLevel.empty()) {
        error("%s: PMTiles archive %s has no tiles at existing min zoom %u",
            __func__, inPmtiles.c_str(), static_cast<unsigned>(existingMinZoom));
    }

    for (int32_t z = static_cast<int32_t>(existingMinZoom) - 1; z >= static_cast<int32_t>(minZoom); --z) {
        const std::vector<ParentBuildInput> parents = enumerate_parent_tiles(currentLevel);
        if (parents.empty()) {
            notice("%s: no parent tiles found at z%d; stopping early", __func__, z);
            break;
        }

        std::vector<ParentTileCandidate> candidates(parents.size());
        parallel_for_tiles(parents.size(), threads, [&](size_t i) {
            candidates[i] = build_parent_candidate(parents[i], schema, blobFd, priorityFd);
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
                encode_parent_candidate(candidates[i], schema, options, levelRatio);
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
            stored.dataOffset = append_blob_to_store(blobFd, blobSize, encoded->payload.compressedData);
            stored.priorityCount = static_cast<uint32_t>(encoded->priorities.size());
            stored.priorityOffset = append_priorities_to_store(
                priorityFd, prioritySize, encoded->priorities);
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
        archive.metadata, outPmtiles, minZoom, maxZoom);

    notice("%s: writing %zu total tiles to %s",
        __func__, allTileRefs.size(), outPmtiles.c_str());
    close(blobFd);
    close(priorityFd);
    mlt_pmtiles::write_pmtiles_archive_from_blob_file(
        outPmtiles, blobFile, std::move(allTileRefs), archiveOptions);
    unlink(blobFile.c_str());
    unlink(priorityFile.c_str());
}
