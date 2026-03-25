#include "tileoperator.hpp"

#include "json.hpp"
#include "mlt_encoder.hpp"
#include "mlt_pmtiles_utils.hpp"
#include "pmtiles_writer.hpp"
#include "PMTiles/pmtiles.hpp"

#include <algorithm>
#include <atomic>
#include <cinttypes>
#include <cmath>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace {

struct AccumTileData {
    bool hasZ = false;
    std::vector<int32_t> localX;
    std::vector<int32_t> localY;
    std::vector<uint32_t> featureCodes;
    std::vector<float> zValues;
    std::vector<std::vector<int32_t>> kValues;
    std::vector<std::vector<bool>> kPresent;
    std::vector<std::vector<float>> pValues;
    std::vector<std::vector<bool>> pPresent;

    AccumTileData() = default;
    AccumTileData(size_t kCount, bool withZ)
        : hasZ(withZ),
          kValues(kCount),
          kPresent(kCount),
          pValues(kCount),
          pPresent(kCount) {}

    size_t size() const {
        return localX.size();
    }

    void append(int32_t x, int32_t y, uint32_t featureCode, float z,
        const std::vector<int32_t>& ks, const std::vector<float>& ps,
        int32_t totalFactors) {
        if (ks.size() != kValues.size() || ps.size() != pValues.size()) {
            error("%s: top-k payload length mismatch", __func__);
        }
        localX.push_back(x);
        localY.push_back(y);
        featureCodes.push_back(featureCode);
        if (hasZ) {
            zValues.push_back(z);
        }
        for (size_t i = 0; i < kValues.size(); ++i) {
            const bool present = ks[i] >= 0;
            if (present) {
                if (totalFactors > 0 && ks[i] >= totalFactors) {
                    error("%s: factor index %d is out of range for K=%d",
                        __func__, ks[i], totalFactors);
                }
                if (!(ps[i] >= 0.0f && ps[i] <= 1.0f)) {
                    error("%s: factor probability %.6g is outside [0, 1]",
                        __func__, static_cast<double>(ps[i]));
                }
                kValues[i].push_back(ks[i]);
                pValues[i].push_back(ps[i]);
            } else {
                kValues[i].push_back(0);
                pValues[i].push_back(0.0f);
            }
            kPresent[i].push_back(present);
            pPresent[i].push_back(present);
        }
    }

    void appendFrom(const AccumTileData& other) {
        if (hasZ != other.hasZ || kValues.size() != other.kValues.size()) {
            error("%s: incompatible tile accumulators", __func__);
        }
        localX.insert(localX.end(), other.localX.begin(), other.localX.end());
        localY.insert(localY.end(), other.localY.begin(), other.localY.end());
        featureCodes.insert(featureCodes.end(), other.featureCodes.begin(), other.featureCodes.end());
        if (hasZ) {
            zValues.insert(zValues.end(), other.zValues.begin(), other.zValues.end());
        }
        for (size_t i = 0; i < kValues.size(); ++i) {
            kValues[i].insert(kValues[i].end(), other.kValues[i].begin(), other.kValues[i].end());
            kPresent[i].insert(kPresent[i].end(), other.kPresent[i].begin(), other.kPresent[i].end());
            pValues[i].insert(pValues[i].end(), other.pValues[i].begin(), other.pValues[i].end());
            pPresent[i].insert(pPresent[i].end(), other.pPresent[i].begin(), other.pPresent[i].end());
        }
    }
};

struct OutputTileInfo {
    TileKey sourceKey{};
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t tileId = 0;
    AccumTileData* data = nullptr;
};

struct GeoTileRect {
    double xmin = 0.0;
    double ymin = 0.0;
    double xmax = 0.0;
    double ymax = 0.0;
};

struct SpillFragmentIndex {
    TileKey tile{};
    uint64_t dataOffset = 0;
    uint64_t dataBytes = 0;
    uint32_t rowCount = 0;
};

struct WorkerSpillShard {
    std::string dataPath;
    int fdData = -1;
    uint64_t dataSize = 0;
    std::vector<SpillFragmentIndex> fragments;
};

struct WorkerScanResult {
    std::vector<mlt_pmtiles::EncodedTilePayload> completedTiles;
    WorkerSpillShard spill;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
    uint64_t totalRecordCount = 0;
};

struct SpillFragmentRef {
    size_t shardId = 0;
    uint64_t dataOffset = 0;
    uint64_t dataBytes = 0;
    uint32_t rowCount = 0;
};

int32_t floor_div_int32(int32_t value, int32_t divisor) {
    if (divisor <= 0) {
        error("%s: divisor must be positive", __func__);
    }
    int32_t q = value / divisor;
    int32_t r = value % divisor;
    if (r != 0 && ((r < 0) != (divisor < 0))) {
        --q;
    }
    return q;
}

int32_t scale_coord_to_int(float value, double coordScale) {
    const long double scaled = static_cast<long double>(value) * static_cast<long double>(coordScale);
    if (scaled < static_cast<long double>(std::numeric_limits<int32_t>::min()) ||
        scaled > static_cast<long double>(std::numeric_limits<int32_t>::max())) {
        error("%s: scaled coordinate %.8Lf is outside int32 range", __func__, scaled);
    }
    return static_cast<int32_t>(std::trunc(scaled));
}

std::string stem_from_path(const std::string& path) {
    std::string base = path;
    const size_t slash = base.find_last_of("/\\");
    if (slash != std::string::npos) {
        base = base.substr(slash + 1);
    }
    const size_t dot = base.find_last_of('.');
    if (dot != std::string::npos) {
        base = base.substr(0, dot);
    }
    if (base.empty()) {
        base = "data";
    }
    return base;
}

template<typename T>
void reorder_vector(std::vector<T>& values, const std::vector<size_t>& order) {
    std::vector<T> reordered;
    reordered.reserve(values.size());
    for (size_t idx : order) {
        reordered.push_back(values[idx]);
    }
    values.swap(reordered);
}

AccumTileData sort_tile_rows(const AccumTileData& in) {
    AccumTileData out = in;
    std::vector<size_t> order(out.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        if (out.localY[lhs] != out.localY[rhs]) {
            return out.localY[lhs] < out.localY[rhs];
        }
        if (out.localX[lhs] != out.localX[rhs]) {
            return out.localX[lhs] < out.localX[rhs];
        }
        return out.featureCodes[lhs] < out.featureCodes[rhs];
    });

    reorder_vector(out.localX, order);
    reorder_vector(out.localY, order);
    reorder_vector(out.featureCodes, order);
    if (out.hasZ) {
        reorder_vector(out.zValues, order);
    }
    for (size_t i = 0; i < out.kValues.size(); ++i) {
        reorder_vector(out.kValues[i], order);
        reorder_vector(out.kPresent[i], order);
        reorder_vector(out.pValues[i], order);
        reorder_vector(out.pPresent[i], order);
    }
    return out;
}

mlt_pmtiles::PointTileData build_point_tile_data(const AccumTileData& in) {
    mlt_pmtiles::PointTileData out;
    out.localX = in.localX;
    out.localY = in.localY;
    out.columns.reserve(1 + (in.hasZ ? 1 : 0) + in.kValues.size() * 2);

    mlt_pmtiles::PropertyColumn featureCol;
    featureCol.type = mlt_pmtiles::ColumnType::StringDictionary;
    featureCol.nullable = false;
    featureCol.stringCodes = in.featureCodes;
    out.columns.push_back(std::move(featureCol));

    if (in.hasZ) {
        mlt_pmtiles::PropertyColumn zCol;
        zCol.type = mlt_pmtiles::ColumnType::Float32;
        zCol.nullable = false;
        zCol.floatValues = in.zValues;
        out.columns.push_back(std::move(zCol));
    }

    for (size_t i = 0; i < in.kValues.size(); ++i) {
        mlt_pmtiles::PropertyColumn kCol;
        kCol.type = mlt_pmtiles::ColumnType::Int32;
        kCol.nullable = true;
        kCol.present = in.kPresent[i];
        kCol.intValues = in.kValues[i];
        out.columns.push_back(std::move(kCol));

        mlt_pmtiles::PropertyColumn pCol;
        pCol.type = mlt_pmtiles::ColumnType::Float32;
        pCol.nullable = true;
        pCol.present = in.pPresent[i];
        pCol.floatValues = in.pValues[i];
        out.columns.push_back(std::move(pCol));
    }
    return out;
}

uint8_t choose_single_zoom(uint64_t width, uint64_t height) {
    uint64_t span = std::max(width, height);
    uint8_t zoom = 0;
    uint64_t capacity = 1;
    while (capacity < span) {
        if (zoom == std::numeric_limits<uint8_t>::max()) {
            error("%s: tile grid span is too large for PMTiles zoom encoding", __func__);
        }
        ++zoom;
        capacity <<= 1;
    }
    return zoom;
}

template<typename T>
void append_scalar(std::vector<char>& out, const T& value) {
    const char* ptr = reinterpret_cast<const char*>(&value);
    out.insert(out.end(), ptr, ptr + sizeof(T));
}

template<typename T>
void append_array(std::vector<char>& out, const std::vector<T>& values) {
    if (values.empty()) {
        return;
    }
    const char* ptr = reinterpret_cast<const char*>(values.data());
    out.insert(out.end(), ptr, ptr + sizeof(T) * values.size());
}

std::vector<char> serialize_accum_tile_data(const AccumTileData& tile) {
    std::vector<char> out;
    const uint32_t rowCount = static_cast<uint32_t>(tile.size());
    append_scalar(out, rowCount);
    append_array(out, tile.localX);
    append_array(out, tile.localY);
    append_array(out, tile.featureCodes);
    if (tile.hasZ) {
        append_array(out, tile.zValues);
    }
    for (size_t i = 0; i < tile.kValues.size(); ++i) {
        for (bool present : tile.kPresent[i]) {
            out.push_back(present ? 1 : 0);
        }
        append_array(out, tile.kValues[i]);
        append_array(out, tile.pValues[i]);
    }
    return out;
}

template<typename T>
const char* read_array_into(const char* ptr, const char* end, std::vector<T>& out, size_t n, const char* funcName) {
    const size_t bytes = sizeof(T) * n;
    if (static_cast<size_t>(end - ptr) < bytes) {
        error("%s: truncated spill fragment", funcName);
    }
    out.resize(n);
    if (bytes > 0) {
        std::memcpy(out.data(), ptr, bytes);
    }
    return ptr + bytes;
}

AccumTileData deserialize_accum_tile_data(const std::vector<char>& bytes, size_t kCount, bool hasZ) {
    const char* ptr = bytes.data();
    const char* end = ptr + bytes.size();
    if (static_cast<size_t>(end - ptr) < sizeof(uint32_t)) {
        error("%s: truncated spill fragment", __func__);
    }
    uint32_t rowCount = 0;
    std::memcpy(&rowCount, ptr, sizeof(rowCount));
    ptr += sizeof(rowCount);

    AccumTileData out(kCount, hasZ);
    ptr = read_array_into(ptr, end, out.localX, rowCount, __func__);
    ptr = read_array_into(ptr, end, out.localY, rowCount, __func__);
    ptr = read_array_into(ptr, end, out.featureCodes, rowCount, __func__);
    if (hasZ) {
        ptr = read_array_into(ptr, end, out.zValues, rowCount, __func__);
    }
    for (size_t i = 0; i < kCount; ++i) {
        if (static_cast<size_t>(end - ptr) < rowCount) {
            error("%s: truncated PRESENT bytes in spill fragment", __func__);
        }
        out.kPresent[i].reserve(rowCount);
        out.pPresent[i].reserve(rowCount);
        for (uint32_t j = 0; j < rowCount; ++j) {
            const bool present = ptr[j] != 0;
            out.kPresent[i].push_back(present);
            out.pPresent[i].push_back(present);
        }
        ptr += rowCount;
        ptr = read_array_into(ptr, end, out.kValues[i], rowCount, __func__);
        ptr = read_array_into(ptr, end, out.pValues[i], rowCount, __func__);
    }
    if (ptr != end) {
        error("%s: unexpected trailing bytes in spill fragment", __func__);
    }
    return out;
}

mlt_pmtiles::EncodedTilePayload encode_accum_tile_payload(
    const TileKey& tileKey,
    uint8_t zoom,
    const mlt_pmtiles::Schema& schema,
    const AccumTileData& data,
    const mlt_pmtiles::GlobalStringDictionary& featureDictionary) {
    const AccumTileData sorted = sort_tile_rows(data);
    const auto tile = build_point_tile_data(sorted);
    const std::string raw = mlt_pmtiles::encode_point_tile(schema, tile, &featureDictionary);
    mlt_pmtiles::EncodedTilePayload encoded;
    encoded.tileId = pmtiles::zxy_to_tileid(zoom, static_cast<uint32_t>(tileKey.col), static_cast<uint32_t>(tileKey.row));
    encoded.z = zoom;
    encoded.x = static_cast<uint32_t>(tileKey.col);
    encoded.y = static_cast<uint32_t>(tileKey.row);
    encoded.featureCount = static_cast<uint32_t>(tile.size());
    encoded.compressedData = mlt_pmtiles::gzip_compress(raw);
    return encoded;
}

GeoTileRect source_tile_rect_epsg3857(const TileKey& sourceKey, int32_t sourceTileSize, double coordScale) {
    const double x0 = static_cast<double>(sourceKey.col) * static_cast<double>(sourceTileSize) * coordScale;
    const double x1 = static_cast<double>(sourceKey.col + 1) * static_cast<double>(sourceTileSize) * coordScale;
    const double y0 = static_cast<double>(sourceKey.row) * static_cast<double>(sourceTileSize) * coordScale;
    const double y1 = static_cast<double>(sourceKey.row + 1) * static_cast<double>(sourceTileSize) * coordScale;
    GeoTileRect rect;
    rect.xmin = std::min(x0, x1);
    rect.xmax = std::max(x0, x1);
    rect.ymin = std::min(y0, y1);
    rect.ymax = std::max(y0, y1);
    return rect;
}

GeoTileRect destination_tile_rect_epsg3857(const TileKey& tileKey, uint8_t zoom) {
    double x0 = 0.0;
    double y0 = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    mlt_pmtiles::tilecoord_to_epsg3857(tileKey.col, tileKey.row, 0.0, 0.0, zoom, x0, y0);
    mlt_pmtiles::tilecoord_to_epsg3857(tileKey.col, tileKey.row, 256.0, 256.0, zoom, x1, y1);
    GeoTileRect rect;
    rect.xmin = std::min(x0, x1);
    rect.xmax = std::max(x0, x1);
    rect.ymin = std::min(y0, y1);
    rect.ymax = std::max(y0, y1);
    return rect;
}

bool is_interior_destination_tile(const TileKey& tileKey, const GeoTileRect& sourceRect, uint8_t zoom) {
    const GeoTileRect dst = destination_tile_rect_epsg3857(tileKey, zoom);
    const double tol = std::max(1e-6, mlt_pmtiles::epsg3857_scale_factor(zoom));
    return dst.xmin > sourceRect.xmin + tol &&
           dst.xmax < sourceRect.xmax - tol &&
           dst.ymin > sourceRect.ymin + tol &&
           dst.ymax < sourceRect.ymax - tol;
}

void close_spill_shard(WorkerSpillShard& shard) {
    if (shard.fdData >= 0) {
        ::close(shard.fdData);
        shard.fdData = -1;
    }
}

void spill_accum_tile(WorkerSpillShard& shard, const TileKey& tileKey, const AccumTileData& tile) {
    std::vector<char> bytes = serialize_accum_tile_data(tile);
    if (!bytes.empty() && !write_all(shard.fdData, bytes.data(), bytes.size())) {
        error("%s: failed writing spill shard %s", __func__, shard.dataPath.c_str());
    }
    SpillFragmentIndex idx;
    idx.tile = tileKey;
    idx.dataOffset = shard.dataSize;
    idx.dataBytes = bytes.size();
    idx.rowCount = static_cast<uint32_t>(tile.size());
    shard.dataSize += idx.dataBytes;
    shard.fragments.push_back(idx);
}

std::vector<mlt_pmtiles::EncodedTilePayload> run_epsg3857_parallel_pipeline(
    const std::string& dataFile,
    const std::vector<TileInfo>& blocks,
    const IndexHeader& formatInfo,
    bool hasZ,
    int32_t kValueCount,
    int32_t totalFactorCount,
    int32_t threadCount,
    const std::function<void(std::ifstream&, float&, float&, float&, uint32_t&, std::vector<int32_t>&, std::vector<float>&)>& readRecord,
    const mlt_pmtiles::Schema& schema,
    const mlt_pmtiles::GlobalStringDictionary& featureDictionary,
    const std::vector<std::string>& featureNames,
    double coordScale,
    uint8_t zoom,
    uint64_t& totalRecordCount,
    double& geoMinX,
    double& geoMinY,
    double& geoMaxX,
    double& geoMaxY) {
    if (formatInfo.tileSize <= 0) {
        error("%s: input tile size must be positive", __func__);
    }

    const size_t workerCount = static_cast<size_t>(std::max(1, threadCount));
    ScopedTempDir tmpDirScope(std::filesystem::temp_directory_path());

    std::vector<WorkerScanResult> workerResults(workerCount);
    for (size_t i = 0; i < workerCount; ++i) {
        auto& spill = workerResults[i].spill;
        spill.dataPath = (tmpDirScope.path / ("worker." + std::to_string(i) + ".mltspill.dat")).string();
        spill.fdData = ::open(spill.dataPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (spill.fdData < 0) {
            error("%s: failed opening spill shard %s", __func__, spill.dataPath.c_str());
        }
    }

    std::atomic<size_t> nextBlock{0};
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    for (size_t workerId = 0; workerId < workerCount; ++workerId) {
        workers.emplace_back([&, workerId]() {
            auto& result = workerResults[workerId];
            std::ifstream dataStream(dataFile, std::ios::binary);
            if (!dataStream.is_open()) {
                error("%s: cannot open binary input %s", __func__, dataFile.c_str());
            }

            for (;;) {
                const size_t blockIdx = nextBlock.fetch_add(1, std::memory_order_relaxed);
                if (blockIdx >= blocks.size()) {
                    break;
                }
                const auto& blk = blocks[blockIdx];
                const uint64_t len = blk.idx.ed - blk.idx.st;
                if (len == 0) {
                    continue;
                }
                if (formatInfo.recordSize == 0 || (len % formatInfo.recordSize) != 0) {
                    error("%s: block length %" PRIu64 " is inconsistent with record size %u",
                        __func__, len, formatInfo.recordSize);
                }
                dataStream.clear();
                dataStream.seekg(static_cast<std::streamoff>(blk.idx.st));
                const uint64_t nRecords = len / formatInfo.recordSize;
                const TileKey sourceKey{blk.row, blk.col};
                const GeoTileRect sourceRect = source_tile_rect_epsg3857(sourceKey, formatInfo.tileSize, coordScale);

                std::unordered_map<TileKey, AccumTileData, TileKeyHash> localTiles;
                localTiles.reserve(64);
                for (uint64_t i = 0; i < nRecords; ++i) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    uint32_t featureIdx = 0;
                    std::vector<int32_t> ks;
                    std::vector<float> ps;
                    readRecord(dataStream, x, y, z, featureIdx, ks, ps);

                    if (featureIdx >= featureNames.size()) {
                        error("%s: feature index %u is out of range for dictionary size %zu",
                            __func__, featureIdx, featureNames.size());
                    }

                    const double scaledX = static_cast<double>(x) * coordScale;
                    const double scaledY = static_cast<double>(y) * coordScale;
                    result.geoMinX = std::min(result.geoMinX, scaledX);
                    result.geoMinY = std::min(result.geoMinY, scaledY);
                    result.geoMaxX = std::max(result.geoMaxX, scaledX);
                    result.geoMaxY = std::max(result.geoMaxY, scaledY);

                    int64_t tx = 0;
                    int64_t ty = 0;
                    double tileLocalX = 0.0;
                    double tileLocalY = 0.0;
                    mlt_pmtiles::epsg3857_to_tilecoord(scaledX, scaledY, zoom, tx, ty, tileLocalX, tileLocalY);
                    if (tx < std::numeric_limits<int32_t>::min() || tx > std::numeric_limits<int32_t>::max() ||
                        ty < std::numeric_limits<int32_t>::min() || ty > std::numeric_limits<int32_t>::max()) {
                        error("%s: output tile coordinate is out of int32 range", __func__);
                    }
                    TileKey tileKey{static_cast<int32_t>(ty), static_cast<int32_t>(tx)};
                    int32_t localX = static_cast<int32_t>(std::llround(tileLocalX * static_cast<double>(schema.extent) / 256.0));
                    int32_t localY = static_cast<int32_t>(std::llround(tileLocalY * static_cast<double>(schema.extent) / 256.0));
                    localX = std::clamp(localX, 0, static_cast<int32_t>(schema.extent) - 1);
                    localY = std::clamp(localY, 0, static_cast<int32_t>(schema.extent) - 1);

                    auto it = localTiles.find(tileKey);
                    if (it == localTiles.end()) {
                        it = localTiles.emplace(tileKey, AccumTileData(static_cast<size_t>(kValueCount), hasZ)).first;
                    }
                    it->second.append(localX, localY, featureIdx, z, ks, ps, totalFactorCount);
                    ++result.totalRecordCount;
                }

                for (const auto& kv : localTiles) {
                    if (is_interior_destination_tile(kv.first, sourceRect, zoom)) {
                        result.completedTiles.push_back(
                            encode_accum_tile_payload(kv.first, zoom, schema, kv.second, featureDictionary));
                    } else {
                        spill_accum_tile(result.spill, kv.first, kv.second);
                    }
                }
            }

            close_spill_shard(result.spill);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    totalRecordCount = 0;
    geoMinX = std::numeric_limits<double>::infinity();
    geoMinY = std::numeric_limits<double>::infinity();
    geoMaxX = -std::numeric_limits<double>::infinity();
    geoMaxY = -std::numeric_limits<double>::infinity();

    std::vector<mlt_pmtiles::EncodedTilePayload> encodedTiles;
    for (const auto& worker : workerResults) {
        totalRecordCount += worker.totalRecordCount;
        geoMinX = std::min(geoMinX, worker.geoMinX);
        geoMinY = std::min(geoMinY, worker.geoMinY);
        geoMaxX = std::max(geoMaxX, worker.geoMaxX);
        geoMaxY = std::max(geoMaxY, worker.geoMaxY);
        encodedTiles.insert(encodedTiles.end(), worker.completedTiles.begin(), worker.completedTiles.end());
    }

    std::map<TileKey, std::vector<SpillFragmentRef>> groupedFragments;
    for (size_t shardId = 0; shardId < workerResults.size(); ++shardId) {
        for (const auto& fragment : workerResults[shardId].spill.fragments) {
            groupedFragments[fragment.tile].push_back(
                SpillFragmentRef{shardId, fragment.dataOffset, fragment.dataBytes, fragment.rowCount});
        }
    }
    if (groupedFragments.empty()) {
        return encodedTiles;
    }

    std::vector<std::pair<TileKey, std::vector<SpillFragmentRef>>> mergeTasks;
    mergeTasks.reserve(groupedFragments.size());
    for (auto& kv : groupedFragments) {
        mergeTasks.emplace_back(kv.first, std::move(kv.second));
    }

    std::vector<std::vector<mlt_pmtiles::EncodedTilePayload>> mergeOutputs(workerCount);
    std::atomic<size_t> nextTask{0};
    workers.clear();
    workers.reserve(workerCount);
    for (size_t workerId = 0; workerId < workerCount; ++workerId) {
        workers.emplace_back([&, workerId]() {
            std::vector<std::ifstream> spillStreams(workerResults.size());
            for (size_t shardId = 0; shardId < workerResults.size(); ++shardId) {
                if (!workerResults[shardId].spill.fragments.empty()) {
                    spillStreams[shardId].open(workerResults[shardId].spill.dataPath, std::ios::binary);
                    if (!spillStreams[shardId].is_open()) {
                        error("%s: failed opening spill shard %s", __func__, workerResults[shardId].spill.dataPath.c_str());
                    }
                }
            }

            auto& out = mergeOutputs[workerId];
            for (;;) {
                const size_t taskIdx = nextTask.fetch_add(1, std::memory_order_relaxed);
                if (taskIdx >= mergeTasks.size()) {
                    break;
                }
                const TileKey tileKey = mergeTasks[taskIdx].first;
                const auto& fragments = mergeTasks[taskIdx].second;
                AccumTileData merged(static_cast<size_t>(kValueCount), hasZ);
                for (const auto& fragment : fragments) {
                    std::ifstream& in = spillStreams[fragment.shardId];
                    in.clear();
                    in.seekg(static_cast<std::streamoff>(fragment.dataOffset));
                    std::vector<char> buf(fragment.dataBytes);
                    if (!buf.empty()) {
                        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
                        if (in.gcount() != static_cast<std::streamsize>(buf.size())) {
                            error("%s: failed reading spill fragment", __func__);
                        }
                    }
                    const AccumTileData part = deserialize_accum_tile_data(buf, static_cast<size_t>(kValueCount), hasZ);
                    merged.appendFrom(part);
                }
                out.push_back(encode_accum_tile_payload(tileKey, zoom, schema, merged, featureDictionary));
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    for (const auto& vec : mergeOutputs) {
        encodedTiles.insert(encodedTiles.end(), vec.begin(), vec.end());
    }
    return encodedTiles;
}

} // namespace

void TileOperator::writeMltPmtiles(const std::string& outFile,
    const std::string& featureDictFile, double coordScale,
    bool epsg3857Mode, int32_t zoom, int32_t targetTileSize) {
    (void) featureDictFile;
    if (isTextInput()) {
        error("%s: MLT-PMTiles export currently supports binary input only", __func__);
    }
    if (!hasFeatureIndex()) {
        error("%s: MLT-PMTiles export requires feature-bearing binary input with stored feature indices; "
              "for pixel-decode outputs use --single-molecule together with --output-binary", __func__);
    }
    if (!(coordScale > 0.0)) {
        error("%s: coordScale must be positive", __func__);
    }
    if (k_ <= 0) {
        error("%s: input does not carry top-k payloads; export requires binary decode output with factor assignments", __func__);
    }
    if (epsg3857Mode) {
        if (zoom < 0 || zoom > 31) {
            error("%s: EPSG:3857 mode requires a zoom in [0, 31]", __func__);
        }
        if (targetTileSize > 0) {
            warning("%s: targetTileSize is ignored in EPSG:3857 mode", __func__);
        }
    } else {
        if ((mode_ & 0x8u) != 0) {
            error("%s: generic MLT-PMTiles export requires regular tile input", __func__);
        }
        if (formatInfo_.tileSize <= 0) {
            error("%s: input tile size must be positive", __func__);
        }
    }

    const std::vector<std::string> featureNames = loadFeatureNames("");
    mlt_pmtiles::GlobalStringDictionary featureDictionary;
    featureDictionary.values = featureNames;

    const bool hasZ = (coord_dim_ == 3);
    int32_t inputTileSizeScaled = -1;
    int32_t adjustedTargetTileSize = targetTileSize;
    uint32_t extent = 4096;
    if (!epsg3857Mode) {
        const double scaledTileSize = static_cast<double>(formatInfo_.tileSize) * coordScale;
        if (scaledTileSize <= 0.0) {
            error("%s: scaled input tile size must be positive", __func__);
        }
        inputTileSizeScaled = static_cast<int32_t>(std::trunc(scaledTileSize));
        if (inputTileSizeScaled <= 0) {
            error("%s: scaled input tile size must be positive after truncation", __func__);
        }
        if (adjustedTargetTileSize <= 0) {
            adjustedTargetTileSize = inputTileSizeScaled;
        }
        if (adjustedTargetTileSize > inputTileSizeScaled) {
            error("%s: targetTileSize %d exceeds scaled input tile size %d",
                __func__, adjustedTargetTileSize, inputTileSizeScaled);
        }
        const int32_t divisor = std::max<int32_t>(1, inputTileSizeScaled / adjustedTargetTileSize);
        adjustedTargetTileSize = inputTileSizeScaled / divisor;
        if (adjustedTargetTileSize <= 0) {
            error("%s: adjusted target tile size is invalid", __func__);
        }
        extent = static_cast<uint32_t>(adjustedTargetTileSize);
    }

    mlt_pmtiles::Schema schema;
    schema.layerName = stem_from_path(outFile);
    schema.extent = extent;
    schema.columns.push_back({"feature", mlt_pmtiles::ColumnType::StringDictionary, false});
    if (hasZ) {
        schema.columns.push_back({"z", mlt_pmtiles::ColumnType::Float32, false});
    }
    for (int32_t i = 0; i < k_; ++i) {
        schema.columns.push_back({"k" + std::to_string(i + 1), mlt_pmtiles::ColumnType::Int32, true});
        schema.columns.push_back({"p" + std::to_string(i + 1), mlt_pmtiles::ColumnType::Float32, true});
    }

    std::map<TileKey, AccumTileData> tileMap;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
    uint64_t totalRecordCount = 0;
    uint8_t outputZoom = 0;
    int32_t genericOriginRow = 0;
    int32_t genericOriginCol = 0;

    std::vector<mlt_pmtiles::EncodedTilePayload> encodedTiles;
    if (epsg3857Mode && threads_ > 1) {
        outputZoom = static_cast<uint8_t>(zoom);
        const auto readRecord = [this, hasZ](std::ifstream& dataStream, float& x, float& y, float& z,
                                   uint32_t& featureIdx, std::vector<int32_t>& ks, std::vector<float>& ps) {
            if (hasZ) {
                PixTopProbsFeature3D<float> rec;
                if (!readBinaryRecord3D(dataStream, rec, false)) {
                    error("%s: failed to read 3D feature record", __func__);
                }
                x = rec.x;
                y = rec.y;
                z = rec.z;
                featureIdx = rec.featureIdx;
                ks = std::move(rec.ks);
                ps = std::move(rec.ps);
            } else {
                PixTopProbsFeature<float> rec;
                if (!readBinaryRecord2D(dataStream, rec, false)) {
                    error("%s: failed to read 2D feature record", __func__);
                }
                x = rec.x;
                y = rec.y;
                z = 0.0f;
                featureIdx = rec.featureIdx;
                ks = std::move(rec.ks);
                ps = std::move(rec.ps);
            }
        };
        encodedTiles = run_epsg3857_parallel_pipeline(dataFile_, blocks_, formatInfo_, hasZ, k_, K_, threads_,
            readRecord, schema, featureDictionary, featureNames, coordScale, outputZoom,
            totalRecordCount, geoMinX, geoMinY, geoMaxX, geoMaxY);
    } else {
        std::ifstream dataStream(dataFile_, std::ios::binary);
        if (!dataStream.is_open()) {
            error("%s: cannot open binary input %s", __func__, dataFile_.c_str());
        }

        for (const auto& blk : blocks_) {
            const uint64_t len = blk.idx.ed - blk.idx.st;
            if (len == 0) {
                continue;
            }
            if (formatInfo_.recordSize == 0 || (len % formatInfo_.recordSize) != 0) {
                error("%s: block length %" PRIu64 " is inconsistent with record size %u",
                    __func__, len, formatInfo_.recordSize);
            }
            dataStream.clear();
            dataStream.seekg(static_cast<std::streamoff>(blk.idx.st));
            const uint64_t nRecords = len / formatInfo_.recordSize;
            for (uint64_t i = 0; i < nRecords; ++i) {
                float x = 0.0f;
                float y = 0.0f;
                float z = 0.0f;
                uint32_t featureIdx = 0;
                std::vector<int32_t> ks;
                std::vector<float> ps;

                if (hasZ) {
                    PixTopProbsFeature3D<float> rec;
                    if (!readBinaryRecord3D(dataStream, rec, false)) {
                        error("%s: failed to read 3D feature record", __func__);
                    }
                    x = rec.x;
                    y = rec.y;
                    z = rec.z;
                    featureIdx = rec.featureIdx;
                    ks = std::move(rec.ks);
                    ps = std::move(rec.ps);
                } else {
                    PixTopProbsFeature<float> rec;
                    if (!readBinaryRecord2D(dataStream, rec, false)) {
                        error("%s: failed to read 2D feature record", __func__);
                    }
                    x = rec.x;
                    y = rec.y;
                    featureIdx = rec.featureIdx;
                    ks = std::move(rec.ks);
                    ps = std::move(rec.ps);
                }

                if (featureIdx >= featureNames.size()) {
                    error("%s: feature index %u is out of range for dictionary size %zu",
                        __func__, featureIdx, featureNames.size());
                }

                TileKey tileKey{};
                int32_t localX = 0;
                int32_t localY = 0;
                if (epsg3857Mode) {
                    const double scaledX = static_cast<double>(x) * coordScale;
                    const double scaledY = static_cast<double>(y) * coordScale;
                    geoMinX = std::min(geoMinX, scaledX);
                    geoMinY = std::min(geoMinY, scaledY);
                    geoMaxX = std::max(geoMaxX, scaledX);
                    geoMaxY = std::max(geoMaxY, scaledY);
                    int64_t tx = 0;
                    int64_t ty = 0;
                    double tileLocalX = 0.0;
                    double tileLocalY = 0.0;
                    mlt_pmtiles::epsg3857_to_tilecoord(scaledX, scaledY, static_cast<uint8_t>(zoom),
                        tx, ty, tileLocalX, tileLocalY);
                    if (tx < std::numeric_limits<int32_t>::min() || tx > std::numeric_limits<int32_t>::max() ||
                        ty < std::numeric_limits<int32_t>::min() || ty > std::numeric_limits<int32_t>::max()) {
                        error("%s: output tile coordinate is out of int32 range", __func__);
                    }
                    tileKey.row = static_cast<int32_t>(ty);
                    tileKey.col = static_cast<int32_t>(tx);
                    localX = static_cast<int32_t>(std::llround(tileLocalX * static_cast<double>(extent) / 256.0));
                    localY = static_cast<int32_t>(std::llround(tileLocalY * static_cast<double>(extent) / 256.0));
                    localX = std::clamp(localX, 0, static_cast<int32_t>(extent) - 1);
                    localY = std::clamp(localY, 0, static_cast<int32_t>(extent) - 1);
                } else {
                    const int32_t scaledX = scale_coord_to_int(x, coordScale);
                    const int32_t scaledY = scale_coord_to_int(y, coordScale);
                    tileKey.row = floor_div_int32(scaledY, adjustedTargetTileSize);
                    tileKey.col = floor_div_int32(scaledX, adjustedTargetTileSize);
                    localX = scaledX - tileKey.col * adjustedTargetTileSize;
                    localY = scaledY - tileKey.row * adjustedTargetTileSize;
                    if (localX < 0 || localX >= adjustedTargetTileSize ||
                        localY < 0 || localY >= adjustedTargetTileSize) {
                        error("%s: tile-local coordinate is outside the target tile extent", __func__);
                    }
                }

                auto it = tileMap.find(tileKey);
                if (it == tileMap.end()) {
                    it = tileMap.emplace(tileKey, AccumTileData(static_cast<size_t>(k_), hasZ)).first;
                }
                it->second.append(localX, localY, featureIdx, z, ks, ps, K_);
                ++totalRecordCount;
            }
        }

        if (epsg3857Mode) {
            outputZoom = static_cast<uint8_t>(zoom);
        } else if (!tileMap.empty()) {
            int32_t minRow = std::numeric_limits<int32_t>::max();
            int32_t maxRow = std::numeric_limits<int32_t>::min();
            int32_t minCol = std::numeric_limits<int32_t>::max();
            int32_t maxCol = std::numeric_limits<int32_t>::min();
            for (const auto& kv : tileMap) {
                minRow = std::min(minRow, kv.first.row);
                maxRow = std::max(maxRow, kv.first.row);
                minCol = std::min(minCol, kv.first.col);
                maxCol = std::max(maxCol, kv.first.col);
            }
            genericOriginRow = minRow;
            genericOriginCol = minCol;
            const uint64_t width = static_cast<uint64_t>(static_cast<int64_t>(maxCol) - static_cast<int64_t>(minCol) + 1);
            const uint64_t height = static_cast<uint64_t>(static_cast<int64_t>(maxRow) - static_cast<int64_t>(minRow) + 1);
            outputZoom = choose_single_zoom(width, height);
        }

        std::vector<OutputTileInfo> outputTiles;
        outputTiles.reserve(tileMap.size());
        for (auto& kv : tileMap) {
            OutputTileInfo info;
            info.sourceKey = kv.first;
            info.z = outputZoom;
            if (epsg3857Mode) {
                info.x = static_cast<uint32_t>(kv.first.col);
                info.y = static_cast<uint32_t>(kv.first.row);
            } else {
                const int64_t x = static_cast<int64_t>(kv.first.col) - static_cast<int64_t>(genericOriginCol);
                const int64_t y = static_cast<int64_t>(kv.first.row) - static_cast<int64_t>(genericOriginRow);
                if (x < 0 || y < 0) {
                    error("%s: normalized generic tile coordinate underflow", __func__);
                }
                info.x = static_cast<uint32_t>(x);
                info.y = static_cast<uint32_t>(y);
            }
            info.tileId = pmtiles::zxy_to_tileid(info.z, info.x, info.y);
            info.data = &kv.second;
            outputTiles.push_back(info);
        }

        encodedTiles.resize(outputTiles.size());
        if (threads_ > 1 && outputTiles.size() > 1) {
            tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
                static_cast<size_t>(threads_));
            tbb::parallel_for(tbb::blocked_range<size_t>(0, outputTiles.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i != range.end(); ++i) {
                        const auto& outTile = outputTiles[i];
                        auto& encoded = encodedTiles[i];
                        encoded = encode_accum_tile_payload(
                            TileKey{static_cast<int32_t>(outTile.y), static_cast<int32_t>(outTile.x)},
                            outTile.z, schema, *outTile.data, featureDictionary);
                    }
                });
        } else {
            for (size_t i = 0; i < outputTiles.size(); ++i) {
                const auto& outTile = outputTiles[i];
                encodedTiles[i] = encode_accum_tile_payload(
                    TileKey{static_cast<int32_t>(outTile.y), static_cast<int32_t>(outTile.x)},
                    outTile.z, schema, *outTile.data, featureDictionary);
            }
        }
    }

    nlohmann::json metadata;
    metadata["name"] = schema.layerName;
    metadata["type"] = "overlay";
    metadata["version"] = "2";
    metadata["format"] = "pbf";
    metadata["description"] = "Generated PMTiles by punkst for MLT point features";
    metadata["generator"] = "punkst tile-op --write-mlt-pmtiles";
    metadata["coord_scale"] = coordScale;
    metadata["feature_dictionary_size"] = featureNames.size();

    if (epsg3857Mode) {
        metadata["punkst_coordinate_mode"] = "epsg3857";
        metadata["zoom"] = outputZoom;
    } else {
        metadata["punkst_coordinate_mode"] = "generic-scaled-grid";
        metadata["source_tile_size_scaled"] = inputTileSizeScaled;
        metadata["target_tile_size"] = adjustedTargetTileSize;
        metadata["grid_zoom"] = outputZoom;
        metadata["grid_origin_row"] = genericOriginRow;
        metadata["grid_origin_col"] = genericOriginCol;
    }

    nlohmann::json fields = nlohmann::json::object();
    fields["feature"] = "String";
    if (hasZ) {
        fields["z"] = "Number";
    }
    for (int32_t i = 0; i < k_; ++i) {
        fields["k" + std::to_string(i + 1)] = "Number";
        fields["p" + std::to_string(i + 1)] = "Number";
    }
    nlohmann::json vectorLayer;
    vectorLayer["id"] = schema.layerName;
    vectorLayer["fields"] = fields;
    vectorLayer["minzoom"] = outputZoom;
    vectorLayer["maxzoom"] = outputZoom;
    metadata["vector_layers"] = nlohmann::json::array({vectorLayer});

    nlohmann::json tilestatsLayer;
    tilestatsLayer["layer"] = schema.layerName;
    tilestatsLayer["count"] = totalRecordCount;
    tilestatsLayer["geometry"] = "Point";
    tilestatsLayer["attributeCount"] = fields.size();
    nlohmann::json attributes = nlohmann::json::array();
    for (auto it = fields.begin(); it != fields.end(); ++it) {
        nlohmann::json attr;
        attr["attribute"] = it.key();
        attr["type"] = (it.value() == "String") ? "string" : "number";
        attributes.push_back(attr);
    }
    tilestatsLayer["attributes"] = attributes;
    nlohmann::json tilestats;
    tilestats["layerCount"] = 1;
    tilestats["layers"] = nlohmann::json::array({tilestatsLayer});
    metadata["tilestats"] = tilestats;

    mlt_pmtiles::ArchiveOptions archiveOptions;
    archiveOptions.tileType = pmtiles::TILETYPE_MLT;
    archiveOptions.minZoom = outputZoom;
    archiveOptions.maxZoom = outputZoom;
    archiveOptions.centerZoom = outputZoom;
    archiveOptions.clustered = true;
    archiveOptions.metadata = metadata;

    if (epsg3857Mode && std::isfinite(geoMinX) && std::isfinite(geoMinY) &&
        std::isfinite(geoMaxX) && std::isfinite(geoMaxY)) {
        double minLon = 0.0;
        double minLat = 0.0;
        double maxLon = 0.0;
        double maxLat = 0.0;
        mlt_pmtiles::epsg3857_to_wgs84(geoMinX, geoMinY, minLon, minLat);
        mlt_pmtiles::epsg3857_to_wgs84(geoMaxX, geoMaxY, maxLon, maxLat);
        archiveOptions.hasGeographicBounds = true;
        archiveOptions.minLonE7 = static_cast<int32_t>(minLon * 10000000.0);
        archiveOptions.minLatE7 = static_cast<int32_t>(minLat * 10000000.0);
        archiveOptions.maxLonE7 = static_cast<int32_t>(maxLon * 10000000.0);
        archiveOptions.maxLatE7 = static_cast<int32_t>(maxLat * 10000000.0);
        if (!encodedTiles.empty()) {
            const auto centerIt = std::max_element(encodedTiles.begin(), encodedTiles.end(),
                [](const mlt_pmtiles::EncodedTilePayload& lhs, const mlt_pmtiles::EncodedTilePayload& rhs) {
                    return lhs.featureCount < rhs.featureCount;
                });
            double centerX = 0.0;
            double centerY = 0.0;
            mlt_pmtiles::tilecoord_to_epsg3857(centerIt->x, centerIt->y, 128.0, 128.0,
                centerIt->z, centerX, centerY);
            double centerLon = 0.0;
            double centerLat = 0.0;
            mlt_pmtiles::epsg3857_to_wgs84(centerX, centerY, centerLon, centerLat);
            archiveOptions.centerLonE7 = static_cast<int32_t>(centerLon * 10000000.0);
            archiveOptions.centerLatE7 = static_cast<int32_t>(centerLat * 10000000.0);
        }
    } else {
        archiveOptions.hasGeographicBounds = true;
        archiveOptions.minLonE7 = -1800000000;
        archiveOptions.minLatE7 = -850000000;
        archiveOptions.maxLonE7 = 1800000000;
        archiveOptions.maxLatE7 = 850000000;
        archiveOptions.centerLonE7 = 0;
        archiveOptions.centerLatE7 = 0;
    }

    notice("%s: writing %zu PMTiles tiles to %s", __func__, encodedTiles.size(), outFile.c_str());
    mlt_pmtiles::write_pmtiles_archive(outFile, std::move(encodedTiles), archiveOptions);
}
