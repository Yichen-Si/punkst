#pragma once

#include "tileoperator.hpp"

#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace tileoperator_detail {

namespace fmt {

template<typename... Args>
inline void append_format(std::string& out, const char* fmtStr, Args... args) {
    char stackBuf[256];
    int n = std::snprintf(stackBuf, sizeof(stackBuf), fmtStr, args...);
    if (n < 0) {
        error("%s: snprintf failed", __func__);
    }
    if (static_cast<size_t>(n) < sizeof(stackBuf)) {
        out.append(stackBuf, static_cast<size_t>(n));
        return;
    }
    std::string temp(static_cast<size_t>(n), '\0');
    if (std::snprintf(temp.data(), temp.size() + 1, fmtStr, args...) != n) {
        error("%s: snprintf failed", __func__);
    }
    out.append(temp);
}

} // namespace fmt

namespace io {

struct TileWriteResult {
    TileKey tile;
    uint32_t nMain = 0;
    uint32_t n = 0;
    std::string textData;
    std::vector<char> binaryData;
};

template<typename T>
inline void append_binary_value(std::vector<char>& out, const T& value) {
    const size_t oldSize = out.size();
    out.resize(oldSize + sizeof(T));
    std::memcpy(out.data() + oldSize, &value, sizeof(T));
}

template<typename T>
inline void append_binary_span(std::vector<char>& out, const std::vector<T>& values) {
    if (values.empty()) {
        return;
    }
    const size_t byteCount = values.size() * sizeof(T);
    const size_t oldSize = out.size();
    out.resize(oldSize + byteCount);
    std::memcpy(out.data() + oldSize, values.data(), byteCount);
}

inline void append_pix_top_probs_binary(std::vector<char>& out,
    int32_t x, int32_t y, const TopProbs& probs) {
    append_binary_value(out, x);
    append_binary_value(out, y);
    append_binary_span(out, probs.ks);
    append_binary_span(out, probs.ps);
}

inline void append_pix_top_probs3d_binary(std::vector<char>& out,
    int32_t x, int32_t y, int32_t z, const TopProbs& probs) {
    append_binary_value(out, x);
    append_binary_value(out, y);
    append_binary_value(out, z);
    append_binary_span(out, probs.ks);
    append_binary_span(out, probs.ps);
}

inline void append_pix_top_probs_feature_binary(std::vector<char>& out,
    int32_t x, int32_t y, uint32_t featureIdx, const TopProbs& probs) {
    append_binary_value(out, x);
    append_binary_value(out, y);
    append_binary_value(out, featureIdx);
    append_binary_span(out, probs.ks);
    append_binary_span(out, probs.ps);
}

inline void append_pix_top_probs_feature3d_binary(std::vector<char>& out,
    int32_t x, int32_t y, int32_t z, uint32_t featureIdx, const TopProbs& probs) {
    append_binary_value(out, x);
    append_binary_value(out, y);
    append_binary_value(out, z);
    append_binary_value(out, featureIdx);
    append_binary_span(out, probs.ks);
    append_binary_span(out, probs.ps);
}

inline void write_tile_result(const TileWriteResult& result, bool binaryOutput,
    FILE* fp, int fdMain, int fdIndex, long& currentOffset, int32_t tileSize) {
    IndexEntryF newEntry(result.tile.row, result.tile.col);
    newEntry.st = currentOffset;
    newEntry.n = result.n;
    tile2bound(result.tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, tileSize);
    if (binaryOutput) {
        if (!result.binaryData.empty() && !write_all(fdMain, result.binaryData.data(), result.binaryData.size())) {
            error("%s: Write error", __func__);
        }
        currentOffset += static_cast<long>(result.binaryData.size());
    } else {
        if (!result.textData.empty() &&
            std::fwrite(result.textData.data(), 1, result.textData.size(), fp) != result.textData.size()) {
            error("%s: Write error", __func__);
        }
        currentOffset = std::ftell(fp);
    }
    newEntry.ed = currentOffset;
    if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) {
        error("%s: Index entry write error", __func__);
    }
}

} // namespace io

namespace merge {

inline void append_top_probs_prefix(TopProbs& out, const TopProbs& src, uint32_t keepK) {
    const size_t keep = std::min<size_t>(keepK, std::min(src.ks.size(), src.ps.size()));
    out.ks.insert(out.ks.end(), src.ks.begin(), src.ks.begin() + keep);
    out.ps.insert(out.ps.end(), src.ps.begin(), src.ps.begin() + keep);
}

inline void append_placeholder_pairs(TopProbs& out, uint32_t keepK) {
    out.ks.insert(out.ks.end(), keepK, -1);
    out.ps.insert(out.ps.end(), keepK, 0.0f);
}

inline TileKey tile_key_from_source_xy(int32_t x, int32_t y, float resXY, int32_t tileSize) {
    const double worldX = static_cast<double>(x) * static_cast<double>(resXY);
    const double worldY = static_cast<double>(y) * static_cast<double>(resXY);
    return TileKey{
        static_cast<int32_t>(std::floor(worldY / static_cast<double>(tileSize))),
        static_cast<int32_t>(std::floor(worldX / static_cast<double>(tileSize)))
    };
}

} // namespace merge

namespace cellagg {

using FactorSums = std::pair<std::unordered_map<int32_t, double>, int32_t>;

struct CellAgg {
    FactorSums sums;
    std::map<std::string, FactorSums> compSums;
    bool boundary = false;
};

inline void write_top_factors(FILE* fp, const FactorSums& sums, uint32_t k_out) {
    std::vector<std::pair<int32_t, double>> items;
    items.reserve(sums.first.size());
    for (const auto& kv : sums.first) {
        if (kv.second != 0.0) {
            items.emplace_back(kv.first, kv.second / sums.second);
        }
    }
    uint32_t keep = std::min<uint32_t>(k_out, static_cast<uint32_t>(items.size()));
    if (keep > 0) {
        std::partial_sort(items.begin(), items.begin() + keep, items.end(),
            [](const auto& a, const auto& b) {
                if (a.second == b.second) return a.first < b.first;
                return a.second > b.second;
            });
    }
    std::fprintf(fp, "\t%d", sums.second);
    for (uint32_t i = 0; i < keep; ++i) {
        std::fprintf(fp, "\t%d\t%.4e", items[i].first, items[i].second);
    }
    for (uint32_t i = keep; i < k_out; ++i) {
        std::fprintf(fp, "\t-1\t0");
    }
}

inline void write_cell_row(FILE* fp, const std::string& cellId, const std::string& comp,
    const FactorSums& sums, uint32_t k_out, bool writeComp) {
    if (writeComp) {
        std::fprintf(fp, "%s\t%s", cellId.c_str(), comp.c_str());
    } else {
        std::fprintf(fp, "%s", cellId.c_str());
    }
    write_top_factors(fp, sums, k_out);
    std::fprintf(fp, "\n");
}

template<typename K, typename V>
inline void add_numeric_map(std::map<K, V>& dst, const std::map<K, V>& src) {
    for (const auto& kv : src) {
        dst[kv.first] += kv.second;
    }
}

template<typename K1, typename K2, typename V>
inline void add_nested_map(std::map<K1, std::map<K2, V>>& dst, const std::map<K1, std::map<K2, V>>& src) {
    for (const auto& kv : src) {
        add_numeric_map(dst[kv.first], kv.second);
    }
}

} // namespace cellagg

namespace parallel {

template<typename T>
class BoundedResultQueue {
public:
    explicit BoundedResultQueue(size_t capacity) : capacity_(std::max<size_t>(1, capacity)) {}

    bool push(T value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cvNotFull_.wait(lock, [&]() {
            return aborted_ || queue_.size() < capacity_;
        });
        if (aborted_) {
            return false;
        }
        queue_.push_back(std::move(value));
        cvNotEmpty_.notify_one();
        return true;
    }

    bool pop(T& value) {
        std::unique_lock<std::mutex> lock(mutex_);
        cvNotEmpty_.wait(lock, [&]() {
            return aborted_ || closed_ || !queue_.empty();
        });
        if (aborted_) {
            return false;
        }
        if (queue_.empty()) {
            return false;
        }
        value = std::move(queue_.front());
        queue_.pop_front();
        cvNotFull_.notify_one();
        return true;
    }

    void close() {
        std::lock_guard<std::mutex> lock(mutex_);
        closed_ = true;
        cvNotEmpty_.notify_all();
        cvNotFull_.notify_all();
    }

    void abort() {
        std::lock_guard<std::mutex> lock(mutex_);
        aborted_ = true;
        cvNotEmpty_.notify_all();
        cvNotFull_.notify_all();
    }

private:
    size_t capacity_;
    std::deque<T> queue_;
    std::mutex mutex_;
    std::condition_variable cvNotEmpty_;
    std::condition_variable cvNotFull_;
    bool closed_ = false;
    bool aborted_ = false;
};

template<typename MakeWorkerStateFn, typename BuildTileResultFn, typename WriteResultFn>
inline void process_tile_results_parallel(const std::vector<TileKey>& tiles, int32_t threads,
    MakeWorkerStateFn&& makeWorkerState, BuildTileResultFn&& buildTileResult,
    WriteResultFn&& writeResult) {
    using Result = io::TileWriteResult;
    const bool useParallel = (threads > 1 && tiles.size() > 1);
    if (!useParallel) {
        auto workerState = makeWorkerState();
        for (const auto& tile : tiles) {
            Result result = buildTileResult(tile, workerState);
            if (result.n > 0) {
                writeResult(result);
            }
        }
        return;
    }

    const size_t chunkTileCount = std::max<size_t>(
        (tiles.size() + static_cast<size_t>(threads) - 1) / static_cast<size_t>(threads), 1);
    const size_t nChunks = (tiles.size() + chunkTileCount - 1) / chunkTileCount;
    BoundedResultQueue<Result> queue(std::max<size_t>(4, static_cast<size_t>(threads) * 2));
    std::exception_ptr writerError;
    std::mutex writerErrorMutex;
    std::thread writer([&]() {
        try {
            Result result;
            while (queue.pop(result)) {
                writeResult(result);
            }
        } catch (...) {
            {
                std::lock_guard<std::mutex> lock(writerErrorMutex);
                writerError = std::current_exception();
            }
            queue.abort();
        }
    });

    try {
        tbb::global_control globalLimit(
            tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, nChunks),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t chunkIdx = range.begin(); chunkIdx < range.end(); ++chunkIdx) {
                    auto workerState = makeWorkerState();
                    const size_t begin = chunkIdx * chunkTileCount;
                    const size_t end = std::min(tiles.size(), begin + chunkTileCount);
                    for (size_t ti = begin; ti < end; ++ti) {
                        Result result = buildTileResult(tiles[ti], workerState);
                        if (result.n > 0 && !queue.push(std::move(result))) {
                            return;
                        }
                    }
                }
            });
    } catch (...) {
        queue.abort();
        if (writer.joinable()) {
            writer.join();
        }
        throw;
    }

    queue.close();
    writer.join();
    {
        std::lock_guard<std::mutex> lock(writerErrorMutex);
        if (writerError) {
            std::rethrow_exception(writerError);
        }
    }
}

} // namespace parallel

} // namespace tileoperator_detail
