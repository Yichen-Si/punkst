#pragma once

#include "tileoperator.hpp"

#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <exception>
#include <map>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <vector>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
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

struct TextOutputHandle {
    FILE* fp = stdout;
    int fdIndex = -1;
    std::string outFile = "stdout";
    std::string outIndex;

    bool writeStdout() const {
        return fp == stdout;
    }
};

inline TextOutputHandle open_text_output(const std::string& outPrefix,
    const char* funcName) {
    TextOutputHandle out;
    if (outPrefix.empty() || outPrefix == "-") {
        return out;
    }
    out.outFile = outPrefix + ".tsv";
    out.outIndex = outPrefix + ".index";
    out.fp = std::fopen(out.outFile.c_str(), "w");
    if (!out.fp) {
        error("%s: Cannot open output file %s", funcName, out.outFile.c_str());
    }
    out.fdIndex = open(out.outIndex.c_str(),
        O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (out.fdIndex < 0) {
        error("%s: Cannot open output index %s", funcName, out.outIndex.c_str());
    }
    return out;
}

inline void close_text_output(TextOutputHandle& out) {
    if (out.fp != nullptr && out.fp != stdout) {
        std::fclose(out.fp);
        out.fp = nullptr;
    }
    if (out.fdIndex >= 0) {
        close(out.fdIndex);
        out.fdIndex = -1;
    }
}

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
        if (fdIndex >= 0) {
            currentOffset = std::ftell(fp);
        } else {
            currentOffset += static_cast<long>(result.textData.size());
        }
    }
    newEntry.ed = currentOffset;
    if (fdIndex >= 0 && !write_all(fdIndex, &newEntry, sizeof(newEntry))) {
        error("%s: Index entry write error", __func__);
    }
}

} // namespace io

namespace merge {

inline void check_k2keep(std::vector<uint32_t>& k2keep, std::vector<TileOperator*>& opPtrs) {
    size_t nSources = opPtrs.size();
    if (!k2keep.empty() && k2keep.size() != nSources) {
        error("%s: expected %zu K values in k2keep, got %zu",
            __func__, nSources, k2keep.size());
    }
    if (k2keep.empty()) {
        for (const auto* op : opPtrs) {
            k2keep.push_back(op->getK());
        }
    } else {
        for (uint32_t i = 0; i < nSources; ++i) {
            if (k2keep[i] > opPtrs[i]->getK()) {
                warning("%s: Invalid value k (%d) specified for the %d-th source", __func__, k2keep[i], i);
                k2keep[i] = opPtrs[i]->getK();
            }
        }
    }
    if (nSources > 7) {
        const int32_t k = *std::min_element(k2keep.begin(), k2keep.end());
        k2keep.assign(nSources, static_cast<uint32_t>(k));
        warning("%s: More than 7 files to merge, keep %d values each", __func__, k);
    }
}

inline std::vector<std::string> build_merge_column_names(
    const std::vector<uint32_t>& k2keep,
    const std::vector<std::string>& mergePrefixes) {
    std::vector<std::string> out;
    size_t totalPairs = 0;
    for (uint32_t keep : k2keep) {
        totalPairs += static_cast<size_t>(keep);
    }
    out.reserve(totalPairs * 2);
    if (!mergePrefixes.empty() && mergePrefixes.size() != k2keep.size()) {
        error("%s: expected %zu merge prefixes, got %zu",
            __func__, k2keep.size(), mergePrefixes.size());
    }
    if (mergePrefixes.empty()) {
        uint32_t idx = 1;
        for (uint32_t keep : k2keep) {
            for (uint32_t j = 0; j < keep; ++j) {
                out.push_back("K" + std::to_string(idx));
                out.push_back("P" + std::to_string(idx));
                ++idx;
            }
        }
        return out;
    }
    for (size_t srcIdx = 0; srcIdx < k2keep.size(); ++srcIdx) {
        const std::string& prefix = mergePrefixes[srcIdx];
        if (prefix.empty()) {
            error("%s: merge prefixes must be non-empty", __func__);
        }
        for (uint32_t j = 0; j < k2keep[srcIdx]; ++j) {
            const uint32_t slot = j + 1;
            out.push_back(prefix + "_K" + std::to_string(slot));
            out.push_back(prefix + "_P" + std::to_string(slot));
        }
    }
    return out;
}

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

namespace feature {

inline std::unordered_map<std::string, uint32_t> build_feature_index_map(
    const std::vector<std::string>& featureNames) {
    std::unordered_map<std::string, uint32_t> out;
    out.reserve(featureNames.size());
    for (uint32_t i = 0; i < featureNames.size(); ++i) {
        const auto ret = out.emplace(featureNames[i], i);
        if (!ret.second) {
            error("%s: duplicate feature name '%s' in dictionary",
                __func__, featureNames[i].c_str());
        }
    }
    return out;
}

struct FeatureRemapPlan {
    std::vector<std::string> canonicalNames;
    std::unordered_map<std::string, uint32_t> canonicalIndex;
    std::vector<std::vector<int32_t>> localToCanonical;
    std::vector<std::vector<int32_t>> canonicalToLocal;
};

inline bool is_identity_feature_remap(const std::vector<int32_t>& localToCanonical) {
    for (size_t i = 0; i < localToCanonical.size(); ++i) {
        if (localToCanonical[i] != static_cast<int32_t>(i)) {
            return false;
        }
    }
    return true;
}

template<size_t FeatureIndexPos, typename Key>
inline void remap_feature_map_to_canonical(std::map<Key, TopProbs>& pixelMap,
    const std::vector<int32_t>& localToCanonical, const char* funcName) {
    if (pixelMap.empty() || localToCanonical.empty() ||
        is_identity_feature_remap(localToCanonical)) {
        return;
    }
    std::map<Key, TopProbs> remapped;
    for (auto& kv : pixelMap) {
        const uint32_t localFeatureIdx = std::get<FeatureIndexPos>(kv.first);
        if (localFeatureIdx >= localToCanonical.size()) {
            error("%s: local feature index %u exceeds remap size %zu",
                funcName, localFeatureIdx, localToCanonical.size());
        }
        const int32_t canonicalFeatureIdx = localToCanonical[localFeatureIdx];
        if (canonicalFeatureIdx < 0) {
            error("%s: missing canonical mapping for local feature index %u",
                funcName, localFeatureIdx);
        }
        Key remappedKey = kv.first;
        std::get<FeatureIndexPos>(remappedKey) = static_cast<uint32_t>(canonicalFeatureIdx);
        remapped.emplace(std::move(remappedKey), std::move(kv.second));
    }
    pixelMap.swap(remapped);
}

inline FeatureRemapPlan build_feature_remap_plan(
    const std::vector<TileOperator*>& opPtrs, const char* funcName) {
    if (opPtrs.empty()) {
        error("%s: no operators provided for feature remapping", funcName);
    }
    FeatureRemapPlan plan;
    plan.localToCanonical.resize(opPtrs.size());
    plan.canonicalToLocal.resize(opPtrs.size());
    plan.canonicalNames = opPtrs[0]->getFeatureNames();
    if (plan.canonicalNames.empty()) {
        error("%s: main input is feature-bearing but has no embedded feature dictionary",
            funcName);
    }
    plan.canonicalIndex = build_feature_index_map(plan.canonicalNames);
    for (size_t srcIdx = 1; srcIdx < opPtrs.size(); ++srcIdx) {
        const TileOperator* op = opPtrs[srcIdx];
        if (!op->hasFeatureIndex()) {
            continue;
        }
        if (op->getFeatureNames().empty()) {
            error("%s: source %zu is feature-bearing but has no embedded feature dictionary",
                funcName, srcIdx);
        }
        for (const auto& featureName : op->getFeatureNames()) {
            if (plan.canonicalIndex.emplace(featureName,
                    static_cast<uint32_t>(plan.canonicalNames.size())).second) {
                plan.canonicalNames.push_back(featureName);
            }
        }
    }
    for (size_t srcIdx = 0; srcIdx < opPtrs.size(); ++srcIdx) {
        const TileOperator* op = opPtrs[srcIdx];
        if (!op->hasFeatureIndex()) {
            continue;
        }
        const auto& sourceNames = op->getFeatureNames();
        plan.localToCanonical[srcIdx].assign(sourceNames.size(), -1);
        plan.canonicalToLocal[srcIdx].assign(plan.canonicalNames.size(), -1);
        for (size_t localIdx = 0; localIdx < sourceNames.size(); ++localIdx) {
            const auto it = plan.canonicalIndex.find(sourceNames[localIdx]);
            if (it == plan.canonicalIndex.end()) {
                error("%s: feature '%s' is missing from canonical dictionary",
                    funcName, sourceNames[localIdx].c_str());
            }
            const int32_t canonicalIdx = static_cast<int32_t>(it->second);
            plan.localToCanonical[srcIdx][localIdx] = canonicalIdx;
            plan.canonicalToLocal[srcIdx][canonicalIdx] = static_cast<int32_t>(localIdx);
        }
    }
    return plan;
}

} // namespace feature

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


struct TileOperator::MergedAnnotate2DCounts {
    uint32_t nMain = 0;
    uint32_t nEmit = 0;
};

template<typename OnEmitFn>
inline uint32_t TileOperator::annotateTile2DPlainShared(
    TileReader& reader, const TileKey& tile, std::ifstream& tileStream,
    uint32_t ntok, int32_t icol_x, int32_t icol_y,
    float resXY, bool annoKeepAll, uint32_t placeholderK,
    OnEmitFn&& onEmit, const char* funcName) const
{
    uint32_t nEmit = 0;
    std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
    if (loadTileToMap(tile, pixelMap, nullptr, &tileStream) <= 0 && !annoKeepAll) {
        return nEmit;
    }
    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {
        return nEmit;
    }
    std::string s;
    while (it->next(s)) {
        if (s.empty() || s[0] == '#') {
            continue;
        }
        std::vector<std::string> tokens;
        split(tokens, "\t", s, ntok + 1, true, true, true);
        if (tokens.size() < ntok) {
            error("%s: Invalid line: %s", funcName, s.c_str());
        }
        float x = 0.0f;
        float y = 0.0f;
        if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
            error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
        }
        const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
        const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
        auto pit = pixelMap.find({ix, iy});
        if (pit == pixelMap.end() && !annoKeepAll) {
            continue;
        }
        TopProbs placeholder;
        const TopProbs* probs = nullptr;
        if (pit == pixelMap.end()) {
            tileoperator_detail::merge::append_placeholder_pairs(placeholder, placeholderK);
            probs = &placeholder;
        } else {
            probs = &pit->second;
        }
        if (onEmit(s, tokens, x, y, ix, iy, *probs)) {
            ++nEmit;
        }
    }
    return nEmit;
}

template<typename OnEmitFn>
inline uint32_t TileOperator::annotateTile3DPlainShared(
    TileReader& reader, const TileKey& tile, std::ifstream& tileStream,
    uint32_t ntok, int32_t icol_x, int32_t icol_y, int32_t icol_z,
    float resXY, float resZ, bool annoKeepAll, uint32_t placeholderK,
    OnEmitFn&& onEmit, const char* funcName) const
{
    uint32_t nEmit = 0;
    std::map<PixelKey3, TopProbs> pixelMap;
    if (loadTileToMap3D(tile, pixelMap, &tileStream) <= 0 && !annoKeepAll) {
        return nEmit;
    }
    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {
        return nEmit;
    }
    std::string s;
    while (it->next(s)) {
        if (s.empty() || s[0] == '#') {
            continue;
        }
        std::vector<std::string> tokens;
        split(tokens, "\t", s, ntok + 1, true, true, true);
        if (tokens.size() < ntok) {
            error("%s: Invalid line: %s", funcName, s.c_str());
        }
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        if (!str2float(tokens[icol_x], x) ||
            !str2float(tokens[icol_y], y) ||
            !str2float(tokens[icol_z], z)) {
            error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
        }
        const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
        const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
        const int32_t iz = static_cast<int32_t>(std::floor(z / resZ));
        auto pit = pixelMap.find(std::make_tuple(ix, iy, iz));
        if (pit == pixelMap.end() && !annoKeepAll) {
            continue;
        }
        TopProbs placeholder;
        const TopProbs* probs = nullptr;
        if (pit == pixelMap.end()) {
            tileoperator_detail::merge::append_placeholder_pairs(placeholder, placeholderK);
            probs = &placeholder;
        } else {
            probs = &pit->second;
        }
        if (onEmit(s, tokens, x, y, z, ix, iy, iz, *probs)) {
            ++nEmit;
        }
    }
    return nEmit;
}

template<typename OnEmitFn>
inline TileOperator::MergedAnnotate2DCounts TileOperator::annotateMergedTile2DPlainShared(
    TileReader& reader, const TileKey& tile,
    std::vector<std::ifstream>& streams,
    const std::vector<MergeSourcePlan>& mergePlans,
    uint32_t ntok, int32_t icol_x, int32_t icol_y, float resXY,
    bool keepAllMain, bool keepAll, bool annoKeepAll,
    size_t totalK, OnEmitFn&& onEmit, const char* funcName) const
{
    MergedAnnotate2DCounts counts;
    const size_t nSources = mergePlans.size();
    std::vector<std::map<TileKey, std::map<std::pair<int32_t, int32_t>, TopProbs>>> auxTileCaches(nSources);
    std::vector<std::set<TileKey>> missingAuxTiles(nSources);
    auto findAuxRecord = [&](size_t srcIdx, int32_t mainX, int32_t mainY) -> const TopProbs* {
        const MergeSourcePlan& plan = mergePlans[srcIdx];
        const std::pair<int32_t, int32_t> auxKey{
            floorDivInt32(mainX, plan.ratioXY),
            floorDivInt32(mainY, plan.ratioXY)
        };
        const TileKey auxTile = tileoperator_detail::merge::tile_key_from_source_xy(
            auxKey.first, auxKey.second, plan.srcResXY, plan.tileSize);
        if (missingAuxTiles[srcIdx].count(auxTile) > 0) {
            return nullptr;
        }
        auto tileIt = auxTileCaches[srcIdx].find(auxTile);
        if (tileIt == auxTileCaches[srcIdx].end()) {
            std::map<std::pair<int32_t, int32_t>, TopProbs> auxMap;
            if (plan.op->loadTileToMap(auxTile, auxMap, nullptr, &streams[srcIdx]) == 0) {
                missingAuxTiles[srcIdx].insert(auxTile);
                return nullptr;
            }
            tileIt = auxTileCaches[srcIdx].emplace(auxTile, std::move(auxMap)).first;
        }
        auto recIt = tileIt->second.find(auxKey);
        return (recIt == tileIt->second.end()) ? nullptr : &recIt->second;
    };

    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {
        return counts;
    }
    std::string s;
    while (it->next(s)) {
        if (s.empty() || s[0] == '#') {
            continue;
        }
        std::vector<std::string> tokens;
        split(tokens, "\t", s, ntok + 1, true, true, true);
        if (tokens.size() < ntok) {
            error("%s: Invalid line: %s", funcName, s.c_str());
        }
        float x = 0.0f;
        float y = 0.0f;
        if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
            error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
        }
        const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
        const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
        TopProbs merged;
        merged.ks.reserve(totalK);
        merged.ps.reserve(totalK);
        bool anyFound = false;
        bool allFound = true;
        bool mainFound = false;
        for (size_t srcIdx = 0; srcIdx < nSources; ++srcIdx) {
            const TopProbs* rec = findAuxRecord(srcIdx, ix, iy);
            if (rec != nullptr) {
                anyFound = true;
                if (srcIdx == 0) {
                    mainFound = true;
                }
                tileoperator_detail::merge::append_top_probs_prefix(
                    merged, *rec, mergePlans[srcIdx].keepK);
            } else {
                allFound = false;
                tileoperator_detail::merge::append_placeholder_pairs(
                    merged, mergePlans[srcIdx].keepK);
            }
        }
        if (mainFound) {
            ++counts.nMain;
        }
        const bool emit = annoKeepAll ||
            (keepAll ? anyFound : (keepAllMain ? mainFound : (mainFound && allFound)));
        if (!emit) {
            continue;
        }
        if (onEmit(s, tokens, x, y, ix, iy, merged)) {
            ++counts.nEmit;
        }
    }
    return counts;
}

template<typename OnEmitFn>
inline TileOperator::MergedAnnotate2DCounts TileOperator::annotateMergedTile3DPlainShared(
    TileReader& reader, const TileKey& tile,
    std::vector<std::ifstream>& streams,
    const std::vector<MergeSourcePlan>& mergePlans,
    uint32_t ntok, int32_t icol_x, int32_t icol_y, int32_t icol_z,
    float resXY, float resZ,
    bool keepAllMain, bool keepAll, bool annoKeepAll,
    size_t totalK, OnEmitFn&& onEmit, const char* funcName) const
{
    MergedAnnotate2DCounts counts;
    const size_t nSources = mergePlans.size();
    std::vector<std::map<TileKey, std::map<std::pair<int32_t, int32_t>, TopProbs>>> auxTileCaches2D(nSources);
    std::vector<std::set<TileKey>> missingAuxTiles2D(nSources);
    std::vector<std::map<TileKey, std::map<PixelKey3, TopProbs>>> auxTileCaches3D(nSources);
    std::vector<std::set<TileKey>> missingAuxTiles3D(nSources);
    auto findAuxRecord = [&](size_t srcIdx, int32_t mainX, int32_t mainY, int32_t mainZ) -> const TopProbs* {
        const MergeSourcePlan& plan = mergePlans[srcIdx];
        const int32_t auxX = floorDivInt32(mainX, plan.ratioXY);
        const int32_t auxY = floorDivInt32(mainY, plan.ratioXY);
        const TileKey auxTile = tileoperator_detail::merge::tile_key_from_source_xy(
            auxX, auxY, plan.srcResXY, plan.tileSize);
        if (plan.relation == MergeSourceRelation::Broadcast2DTo3D) {
            if (missingAuxTiles2D[srcIdx].count(auxTile) > 0) {
                return nullptr;
            }
            auto tileIt = auxTileCaches2D[srcIdx].find(auxTile);
            if (tileIt == auxTileCaches2D[srcIdx].end()) {
                std::map<std::pair<int32_t, int32_t>, TopProbs> auxMap;
                if (plan.op->loadTileToMap(auxTile, auxMap, nullptr, &streams[srcIdx]) == 0) {
                    missingAuxTiles2D[srcIdx].insert(auxTile);
                    return nullptr;
                }
                tileIt = auxTileCaches2D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
            }
            auto recIt = tileIt->second.find({auxX, auxY});
            return (recIt == tileIt->second.end()) ? nullptr : &recIt->second;
        }
        const int32_t auxZ = floorDivInt32(mainZ, plan.ratioZ);
        if (missingAuxTiles3D[srcIdx].count(auxTile) > 0) {
            return nullptr;
        }
        auto tileIt = auxTileCaches3D[srcIdx].find(auxTile);
        if (tileIt == auxTileCaches3D[srcIdx].end()) {
            std::map<PixelKey3, TopProbs> auxMap;
            if (plan.op->loadTileToMap3D(auxTile, auxMap, &streams[srcIdx]) == 0) {
                missingAuxTiles3D[srcIdx].insert(auxTile);
                return nullptr;
            }
            tileIt = auxTileCaches3D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
        }
        auto recIt = tileIt->second.find({auxX, auxY, auxZ});
        return (recIt == tileIt->second.end()) ? nullptr : &recIt->second;
    };

    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {
        return counts;
    }
    std::string s;
    while (it->next(s)) {
        if (s.empty() || s[0] == '#') {
            continue;
        }
        std::vector<std::string> tokens;
        split(tokens, "\t", s, ntok + 1, true, true, true);
        if (tokens.size() < ntok) {
            error("%s: Invalid line: %s", funcName, s.c_str());
        }
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
        if (!str2float(tokens[icol_x], x) ||
            !str2float(tokens[icol_y], y) ||
            !str2float(tokens[icol_z], z)) {
            error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
        }
        const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
        const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
        const int32_t iz = static_cast<int32_t>(std::floor(z / resZ));
        TopProbs merged;
        merged.ks.reserve(totalK);
        merged.ps.reserve(totalK);
        bool anyFound = false;
        bool allFound = true;
        bool mainFound = false;
        for (size_t srcIdx = 0; srcIdx < nSources; ++srcIdx) {
            const TopProbs* rec = findAuxRecord(srcIdx, ix, iy, iz);
            if (rec != nullptr) {
                anyFound = true;
                if (srcIdx == 0) {
                    mainFound = true;
                }
                tileoperator_detail::merge::append_top_probs_prefix(
                    merged, *rec, mergePlans[srcIdx].keepK);
            } else {
                allFound = false;
                tileoperator_detail::merge::append_placeholder_pairs(
                    merged, mergePlans[srcIdx].keepK);
            }
        }
        if (mainFound) {
            ++counts.nMain;
        }
        const bool emit = annoKeepAll ||
            (keepAll ? anyFound : (keepAllMain ? mainFound : (mainFound && allFound)));
        if (!emit) {
            continue;
        }
        if (onEmit(s, tokens, x, y, z, ix, iy, iz, merged)) {
            ++counts.nEmit;
        }
    }
    return counts;
}

template<typename OnEmitFn>
inline uint32_t TileOperator::annotateSingleTile2DShared(
    TileReader& reader, const TileKey& tile, std::ifstream& tileStream,
    const std::unordered_map<std::string, uint32_t>& featureIndex,
    uint32_t ntok, int32_t icol_x, int32_t icol_y, int32_t icol_f,
    float resXY, bool annoKeepAll, uint32_t placeholderK,
    OnEmitFn&& onEmit, const char* funcName) const {
    uint32_t nEmit = 0;
    std::map<PixelFeatureKey2, TopProbs> pixelMap;
    // Locate the tile in the query
    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) { return nEmit; }
    // Load the tile from the main source
    if (loadTileToMapFeature(tile, pixelMap, &tileStream) <= 0 && !annoKeepAll) { return nEmit; }
    std::string s;
    while (it->next(s)) { // For each line in the query
        if (s.empty() || s[0] == '#') {continue;}
        std::vector<std::string> tokens;
        split(tokens, "\t", s, ntok + 1, true, true, true);
        if (tokens.size() < ntok) {
            error("%s: Invalid line: %s", funcName, s.c_str());
        }
        float x = 0.0f, y = 0.0f;
        if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
            error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
        }
        const std::string& featureName = tokens[icol_f];
        const auto fit = featureIndex.find(featureName);
        const bool featureKnown = (fit != featureIndex.end());
        if (!featureKnown && !annoKeepAll) {continue;}
        const uint32_t featureIdx = featureKnown ? fit->second : 0u;
        const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
        const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
        auto pit = featureKnown ? pixelMap.find(std::make_tuple(ix, iy, featureIdx)) : pixelMap.end();
        if (pit == pixelMap.end() && !annoKeepAll) {continue;}
        TopProbs placeholder;
        const TopProbs* probs = nullptr;
        if (pit == pixelMap.end()) {
            tileoperator_detail::merge::append_placeholder_pairs(placeholder, placeholderK);
            probs = &placeholder;
        } else {
            probs = &pit->second;
        }
        if (onEmit(s, tokens, featureName, featureKnown, featureIdx, x, y, ix, iy, *probs)) {
            ++nEmit;
        }
    }
    return nEmit;
}

template<typename OnEmitFn>
inline uint32_t TileOperator::annotateSingleTile3DShared(
    TileReader& reader,
    const TileKey& tile,
    std::ifstream& tileStream,
    const std::unordered_map<std::string, uint32_t>& featureIndex,
    uint32_t ntok,
    int32_t icol_x, int32_t icol_y, int32_t icol_z, int32_t icol_f,
    float resXY, float resZ,
    bool annoKeepAll,
    uint32_t placeholderK,
    OnEmitFn&& onEmit,
    const char* funcName) const {
    uint32_t nEmit = 0;
    std::map<PixelFeatureKey3, TopProbs> pixelMap;
    if (loadTileToMapFeature3D(tile, pixelMap, &tileStream) <= 0 && !annoKeepAll) {
        return nEmit;
    }
    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {
        return nEmit;
    }
    std::string s;
    while (it->next(s)) {
        if (s.empty() || s[0] == '#') {
            continue;
        }
        std::vector<std::string> tokens;
        split(tokens, "\t", s, ntok + 1, true, true, true);
        if (tokens.size() < ntok) {
            error("%s: Invalid line: %s", funcName, s.c_str());
        }
        float x = 0.0f, y = 0.0f, z = 0.0f;
        if (!str2float(tokens[icol_x], x) ||
            !str2float(tokens[icol_y], y) ||
            !str2float(tokens[icol_z], z)) {
            error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
        }
        const std::string& featureName = tokens[icol_f];
        const auto fit = featureIndex.find(featureName);
        if (fit == featureIndex.end() && !annoKeepAll) {
            continue;
        }
        const bool featureKnown = (fit != featureIndex.end());
        const uint32_t featureIdx = featureKnown ? fit->second : 0u;
        const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
        const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
        const int32_t iz = static_cast<int32_t>(std::floor(z / resZ));
        auto pit = featureKnown ? pixelMap.find(std::make_tuple(ix, iy, iz, featureIdx)) : pixelMap.end();
        if (pit == pixelMap.end() && !annoKeepAll) {
            continue;
        }
        TopProbs placeholder;
        const TopProbs* probs = nullptr;
        if (pit == pixelMap.end()) {
            tileoperator_detail::merge::append_placeholder_pairs(placeholder, placeholderK);
            probs = &placeholder;
        } else {
            probs = &pit->second;
        }
        if (onEmit(s, tokens, featureName, featureKnown, featureIdx, x, y, z, ix, iy, iz, *probs)) {
            ++nEmit;
        }
    }
    return nEmit;
}

template<typename OnEmitFn>
inline TileOperator::MergedAnnotate2DCounts TileOperator::annotateMergedTile2DShared(
    TileReader& reader,
    const TileKey& tile,
    std::vector<std::ifstream>& streams,
    const std::vector<MergeSourcePlan>& mergePlans,
    const tileoperator_detail::feature::FeatureRemapPlan& featureRemap,
    const std::unordered_map<std::string, uint32_t>& featureIndex,
    uint32_t ntok,
    int32_t icol_x, int32_t icol_y, int32_t icol_f,
    float resXY,
    bool keepAllMain, bool keepAll, bool annoKeepAll,
    size_t totalK,
    OnEmitFn&& onEmit,
    const char* funcName) const {
    MergedAnnotate2DCounts counts;
    const size_t nSources = mergePlans.size();
    std::vector<bool> loaded(nSources, false);
    std::vector<bool> missing(nSources, false);
    std::vector<std::map<PixelFeatureKey2, TopProbs>> sourceFeature2D(nSources);
    std::vector<std::map<std::pair<int32_t, int32_t>, TopProbs>> sourcePlain2D(nSources);
    auto ensure_loaded = [&](size_t srcIdx) {
        if (loaded[srcIdx]) {
            return;
        }
        loaded[srcIdx] = true;
        const MergeSourcePlan& plan = mergePlans[srcIdx];
        if (plan.op->hasFeatureIndex()) {
            if (plan.op->loadTileToMapFeature(tile, sourceFeature2D[srcIdx], &streams[srcIdx]) == 0) {
                missing[srcIdx] = true;
                return;
            }
            tileoperator_detail::feature::remap_feature_map_to_canonical<2>(
                sourceFeature2D[srcIdx], featureRemap.localToCanonical[srcIdx], funcName);
        } else if (plan.op->loadTileToMap(tile, sourcePlain2D[srcIdx], nullptr, &streams[srcIdx]) == 0) {
            missing[srcIdx] = true;
        }
    };
    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {
        return counts;
    }
    std::string s;
    while (it->next(s)) {
        if (s.empty() || s[0] == '#') {
            continue;
        }
        std::vector<std::string> tokens;
        split(tokens, "\t", s, ntok + 1, true, true, true);
        if (tokens.size() < ntok) {
            error("%s: Invalid line: %s", funcName, s.c_str());
        }
        float x = 0.0f, y = 0.0f;
        if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
            error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
        }
        const std::string& featureName = tokens[icol_f];
        const auto fit = featureIndex.find(featureName);
        if (fit == featureIndex.end() && !annoKeepAll) {
            continue;
        }
        const bool featureKnown = (fit != featureIndex.end());
        const uint32_t featureIdx = featureKnown ? fit->second : 0u;
        const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
        const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
        TopProbs merged;
        merged.ks.reserve(totalK);
        merged.ps.reserve(totalK);
        bool anyFound = false;
        bool allFound = true;
        bool mainFound = false;
        for (size_t srcIdx = 0; srcIdx < nSources; ++srcIdx) {
            const MergeSourcePlan& plan = mergePlans[srcIdx];
            const int32_t auxX = floorDivInt32(ix, plan.ratioXY);
            const int32_t auxY = floorDivInt32(iy, plan.ratioXY);
            const TopProbs* rec = nullptr;
            if (!(plan.op->hasFeatureIndex() && !featureKnown)) {
                ensure_loaded(srcIdx);
            }
            if (!missing[srcIdx]) {
                if (plan.op->hasFeatureIndex()) {
                    auto recIt = sourceFeature2D[srcIdx].find(std::make_tuple(auxX, auxY, featureIdx));
                    if (recIt != sourceFeature2D[srcIdx].end()) {
                        rec = &recIt->second;
                    }
                } else {
                    auto recIt = sourcePlain2D[srcIdx].find({auxX, auxY});
                    if (recIt != sourcePlain2D[srcIdx].end()) {
                        rec = &recIt->second;
                    }
                }
            }
            if (rec == nullptr) {
                allFound = false;
                tileoperator_detail::merge::append_placeholder_pairs(merged, mergePlans[srcIdx].keepK);
            } else {
                anyFound = true;
                if (srcIdx == 0) {
                    mainFound = true;
                }
                tileoperator_detail::merge::append_top_probs_prefix(
                    merged, *rec, mergePlans[srcIdx].keepK);
            }
        }
        if (mainFound) {
            ++counts.nMain;
        }
        const bool emit = annoKeepAll ||
            (keepAll ? anyFound : (keepAllMain ? mainFound : (mainFound && allFound)));
        if (!emit) {
            continue;
        }
        if (onEmit(s, tokens, featureName, featureKnown, featureIdx, x, y, ix, iy, merged)) {
            ++counts.nEmit;
        }
    }
    return counts;
}

template<typename OnEmitFn>
inline TileOperator::MergedAnnotate2DCounts TileOperator::annotateMergedTile3DShared(
    TileReader& reader,
    const TileKey& tile,
    std::vector<std::ifstream>& streams,
    const std::vector<MergeSourcePlan>& mergePlans,
    const tileoperator_detail::feature::FeatureRemapPlan& featureRemap,
    const std::unordered_map<std::string, uint32_t>& featureIndex,
    uint32_t ntok,
    int32_t icol_x, int32_t icol_y, int32_t icol_z, int32_t icol_f,
    float resXY, float resZ,
    bool keepAllMain, bool keepAll, bool annoKeepAll,
    size_t totalK,
    OnEmitFn&& onEmit,
    const char* funcName) const {
    MergedAnnotate2DCounts counts;
    const size_t nSources = mergePlans.size();
    std::vector<bool> loaded(nSources, false);
    std::vector<bool> missing(nSources, false);
    std::vector<std::map<PixelFeatureKey2, TopProbs>> sourceFeature2D(nSources);
    std::vector<std::map<PixelFeatureKey3, TopProbs>> sourceFeature3D(nSources);
    std::vector<std::map<std::pair<int32_t, int32_t>, TopProbs>> sourcePlain2D(nSources);
    std::vector<std::map<PixelKey3, TopProbs>> sourcePlain3D(nSources);
    auto ensure_loaded = [&](size_t srcIdx) {
        if (loaded[srcIdx]) {
            return;
        }
        loaded[srcIdx] = true;
        const MergeSourcePlan& plan = mergePlans[srcIdx];
        if (plan.op->hasFeatureIndex()) {
            if (plan.relation == MergeSourceRelation::Broadcast2DTo3D) {
                if (plan.op->loadTileToMapFeature(tile, sourceFeature2D[srcIdx], &streams[srcIdx]) == 0) {
                    missing[srcIdx] = true;
                    return;
                }
                tileoperator_detail::feature::remap_feature_map_to_canonical<2>(
                    sourceFeature2D[srcIdx], featureRemap.localToCanonical[srcIdx], funcName);
            } else {
                if (plan.op->loadTileToMapFeature3D(tile, sourceFeature3D[srcIdx], &streams[srcIdx]) == 0) {
                    missing[srcIdx] = true;
                    return;
                }
                tileoperator_detail::feature::remap_feature_map_to_canonical<3>(
                    sourceFeature3D[srcIdx], featureRemap.localToCanonical[srcIdx], funcName);
            }
        } else if (plan.relation == MergeSourceRelation::Broadcast2DTo3D) {
            if (plan.op->loadTileToMap(tile, sourcePlain2D[srcIdx], nullptr, &streams[srcIdx]) == 0) {
                missing[srcIdx] = true;
            }
        } else if (plan.op->loadTileToMap3D(tile, sourcePlain3D[srcIdx], &streams[srcIdx]) == 0) {
            missing[srcIdx] = true;
        }
    };
    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {
        return counts;
    }
    std::string s;
    while (it->next(s)) {
        if (s.empty() || s[0] == '#') {
            continue;
        }
        std::vector<std::string> tokens;
        split(tokens, "\t", s, ntok + 1, true, true, true);
        if (tokens.size() < ntok) {
            error("%s: Invalid line: %s", funcName, s.c_str());
        }
        float x = 0.0f, y = 0.0f, z = 0.0f;
        if (!str2float(tokens[icol_x], x) ||
            !str2float(tokens[icol_y], y) ||
            !str2float(tokens[icol_z], z)) {
            error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
        }
        const std::string& featureName = tokens[icol_f];
        const auto fit = featureIndex.find(featureName);
        if (fit == featureIndex.end() && !annoKeepAll) {
            continue;
        }
        const bool featureKnown = (fit != featureIndex.end());
        const uint32_t featureIdx = featureKnown ? fit->second : 0u;
        const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
        const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
        const int32_t iz = static_cast<int32_t>(std::floor(z / resZ));
        TopProbs merged;
        merged.ks.reserve(totalK);
        merged.ps.reserve(totalK);
        bool anyFound = false;
        bool allFound = true;
        bool mainFound = false;
        for (size_t srcIdx = 0; srcIdx < nSources; ++srcIdx) {
            const MergeSourcePlan& plan = mergePlans[srcIdx];
            const int32_t auxX = floorDivInt32(ix, plan.ratioXY);
            const int32_t auxY = floorDivInt32(iy, plan.ratioXY);
            const TopProbs* rec = nullptr;
            if (!(plan.op->hasFeatureIndex() && !featureKnown)) {
                ensure_loaded(srcIdx);
            }
            if (!missing[srcIdx]) {
                if (plan.op->hasFeatureIndex()) {
                    if (plan.relation == MergeSourceRelation::Broadcast2DTo3D) {
                        auto recIt = sourceFeature2D[srcIdx].find(std::make_tuple(auxX, auxY, featureIdx));
                        if (recIt != sourceFeature2D[srcIdx].end()) {
                            rec = &recIt->second;
                        }
                    } else {
                        const int32_t auxZ = floorDivInt32(iz, plan.ratioZ);
                        auto recIt = sourceFeature3D[srcIdx].find(std::make_tuple(auxX, auxY, auxZ, featureIdx));
                        if (recIt != sourceFeature3D[srcIdx].end()) {
                            rec = &recIt->second;
                        }
                    }
                } else if (plan.relation == MergeSourceRelation::Broadcast2DTo3D) {
                    auto recIt = sourcePlain2D[srcIdx].find({auxX, auxY});
                    if (recIt != sourcePlain2D[srcIdx].end()) {
                        rec = &recIt->second;
                    }
                } else {
                    const int32_t auxZ = floorDivInt32(iz, plan.ratioZ);
                    auto recIt = sourcePlain3D[srcIdx].find(std::make_tuple(auxX, auxY, auxZ));
                    if (recIt != sourcePlain3D[srcIdx].end()) {
                        rec = &recIt->second;
                    }
                }
            }
            if (rec == nullptr) {
                allFound = false;
                tileoperator_detail::merge::append_placeholder_pairs(
                    merged, mergePlans[srcIdx].keepK);
            } else {
                anyFound = true;
                if (srcIdx == 0) {
                    mainFound = true;
                }
                tileoperator_detail::merge::append_top_probs_prefix(
                    merged, *rec, mergePlans[srcIdx].keepK);
            }
        }
        if (mainFound) {
            ++counts.nMain;
        }
        const bool emit = annoKeepAll ||
            (keepAll ? anyFound : (keepAllMain ? mainFound : (mainFound && allFound)));
        if (!emit) {
            continue;
        }
        if (onEmit(s, tokens, featureName, featureKnown, featureIdx, x, y, z, ix, iy, iz, merged)) {
            ++counts.nEmit;
        }
    }
    return counts;
}
