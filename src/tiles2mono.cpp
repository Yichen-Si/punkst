#include "tiles2mono.hpp"

#include "image_utils.hpp"
#include "threads.hpp"
#include "tilereader.hpp"
#include "utils.h"
#include "utils_sys.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <map>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tiles2mono {

namespace {

constexpr int32_t kRasterTileSize = 256;

using TileBuffer = std::vector<uint16_t>;
using TileMap = std::unordered_map<mlt_pmtiles::RasterTileKey, TileBuffer,
    mlt_pmtiles::RasterTileKeyHash>;

mlt_pmtiles::RasterBounds resolve_bounds_from_inputs(const Options& options) {
    if (options.bounds.valid()) {
        return options.bounds;
    }
    mlt_pmtiles::RasterBounds bounds;
    if (!options.rangeFile.empty() && file_exists(options.rangeFile)) {
        readCoordRange(options.rangeFile, bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax);
        if (bounds.valid()) {
            return bounds;
        }
    }
    const LoadedTileIndexData loaded = loadTileIndexData(options.indexFile);
    const IndexHeader& h = loaded.header;
    if (h.xmax > h.xmin && h.ymax > h.ymin) {
        bounds.xmin = h.xmin;
        bounds.xmax = h.xmax;
        bounds.ymin = h.ymin;
        bounds.ymax = h.ymax;
        return bounds;
    }
    if (loaded.globalBox.proper()) {
        bounds.xmin = loaded.globalBox.xmin;
        bounds.xmax = loaded.globalBox.xmax;
        bounds.ymin = loaded.globalBox.ymin;
        bounds.ymax = loaded.globalBox.ymax;
        return bounds;
    }
    error("%s: cannot determine bounds from %s or %s",
        __func__, options.rangeFile.c_str(), options.indexFile.c_str());
    return bounds;
}

void add_count(TileBuffer& buffer, int32_t px, int32_t py, double count) {
    if (!(count > 0.0)) {
        return;
    }
    const size_t idx = static_cast<size_t>(py) * static_cast<size_t>(kRasterTileSize) +
        static_cast<size_t>(px);
    const uint32_t add = static_cast<uint32_t>(std::llround(count));
    const uint32_t next = static_cast<uint32_t>(buffer[idx]) + add;
    buffer[idx] = static_cast<uint16_t>(std::min<uint32_t>(next, 255u));
}

void merge_tile_map(TileMap& dst, const TileMap& src) {
    for (const auto& kv : src) {
        auto it = dst.find(kv.first);
        if (it == dst.end()) {
            dst.emplace(kv.first, kv.second);
            continue;
        }
        TileBuffer& out = it->second;
        const TileBuffer& in = kv.second;
        for (size_t i = 0; i < out.size(); ++i) {
            out[i] = static_cast<uint16_t>(
                std::min<uint32_t>(static_cast<uint32_t>(out[i]) + in[i], 255u));
        }
    }
}

int32_t quantile_threshold_from_histogram(const std::array<uint64_t, 256>& hist,
    double quantile) {
    uint64_t total = 0;
    for (int32_t i = 1; i <= 255; ++i) {
        total += hist[static_cast<size_t>(i)];
    }
    if (total == 0) {
        return 255;
    }
    const double q = std::clamp(quantile, 0.0, 1.0);
    const long double target = static_cast<long double>(q) * static_cast<long double>(total);
    uint64_t cumulative = 0;
    int32_t threshold = 255;
    for (int32_t i = 1; i <= 255; ++i) {
        cumulative += hist[static_cast<size_t>(i)];
        if (static_cast<long double>(cumulative) > target) {
            threshold = i;
            break;
        }
    }
    return std::max(1, threshold);
}

uint8_t adjusted_intensity(uint16_t raw, bool autoAdjust, int32_t threshold) {
    if (raw == 0) {
        return 0;
    }
    const uint16_t capped = std::min<uint16_t>(raw, 255);
    if (!autoAdjust) {
        return static_cast<uint8_t>(capped);
    }
    if (capped > threshold) {
        return 255;
    }
    return static_cast<uint8_t>(
        std::min<int32_t>(255, static_cast<int32_t>(capped) * 255 / threshold));
}

TileMap accumulate_zoom(const TileReader& reader,
    const std::vector<TileInfo>& blocks,
    const Options& options,
    const mlt_pmtiles::RasterBounds& bounds,
    int32_t zoom) {
    const int32_t nThreads = std::max<int32_t>(1, options.threads);
    std::vector<TileMap> localMaps(static_cast<size_t>(nThreads));
    ThreadSafeQueue<TileInfo> queue;
    std::vector<std::thread> workers;
    workers.reserve(static_cast<size_t>(nThreads));
    for (int32_t tid = 0; tid < nThreads; ++tid) {
        workers.emplace_back([&, tid]() {
            TileInfo block;
            while (queue.pop(block)) {
                auto iter = reader.get_block_iterator(block);
                if (!iter) {
                    continue;
                }
                std::string line;
                std::vector<std::string> fields;
                const int32_t maxCol = std::max({options.icolX, options.icolY, options.icolCount});
                while (iter->next(line)) {
                    if (line.empty() || line.front() == '#') {
                        continue;
                    }
                    split(fields, "\t", line, UINT_MAX, true, false, false, false);
                    if (static_cast<int32_t>(fields.size()) <= maxCol) {
                        continue;
                    }
                    double x = 0.0;
                    double y = 0.0;
                    double count = 1.0;
                    if (!str2double(fields[static_cast<size_t>(options.icolX)], x) ||
                        !str2double(fields[static_cast<size_t>(options.icolY)], y) ||
                        !str2double(fields[static_cast<size_t>(options.icolCount)], count)) {
                        continue;
                    }
                    if (!(count > 0.0) || x < bounds.xmin || x > bounds.xmax ||
                        y < bounds.ymin || y > bounds.ymax) {
                        continue;
                    }
                    const mlt_pmtiles::RasterPixelCoord pix =
                        mlt_pmtiles::epsg3857_to_raster_pixel(x, y, zoom);
                    TileBuffer& buffer = localMaps[static_cast<size_t>(tid)][pix.key];
                    if (buffer.empty()) {
                        buffer.assign(static_cast<size_t>(kRasterTileSize) *
                            static_cast<size_t>(kRasterTileSize), 0);
                    }
                    add_count(buffer, pix.px, pix.py, count);
                }
            }
        });
    }
    for (const TileInfo& block : blocks) {
        queue.push(block);
    }
    queue.set_done();
    for (auto& worker : workers) {
        worker.join();
    }

    TileMap merged;
    for (const TileMap& local : localMaps) {
        merge_tile_map(merged, local);
    }
    return merged;
}

} // namespace

void write_tiles2mono_pmtiles(const Options& options) {
    if (options.dataFile.empty() || options.indexFile.empty() || options.outFile.empty()) {
        error("%s: input TSV, index, and output PMTiles are required", __func__);
    }
    if (options.icolX < 0 || options.icolY < 0 || options.icolCount < 0) {
        error("%s: column indices must be non-negative", __func__);
    }
    const mlt_pmtiles::RasterBounds bounds = resolve_bounds_from_inputs(options);
    mlt_pmtiles::validate_raster_archive_options(bounds, options.minZoom, options.maxZoom, __func__);
    TileReader reader(options.dataFile, options.indexFile);
    if (!reader.isValid()) {
        error("%s: failed to initialize TileReader for %s", __func__, options.dataFile.c_str());
    }
    std::vector<TileInfo> blocks;
    reader.getBlockList(blocks);
    if (blocks.empty()) {
        error("%s: no input tiles found in %s", __func__, options.indexFile.c_str());
    }

    const std::string tempBlobFile = options.tempBlobFile.empty()
        ? options.outFile + ".blob.tmp"
        : options.tempBlobFile;
    std::filesystem::path blobPath(tempBlobFile);
    if (blobPath.has_parent_path()) {
        std::filesystem::create_directories(blobPath.parent_path());
    }
    std::ofstream blob(blobPath, std::ios::binary | std::ios::trunc);
    if (!blob.is_open()) {
        error("%s: cannot open temporary PMTiles blob %s", __func__, tempBlobFile.c_str());
    }

    std::vector<mlt_pmtiles::StoredTilePayloadRef> tiles;
    uint64_t dataOffset = 0;
    for (int32_t z = options.minZoom; z <= options.maxZoom; ++z) {
        TileMap accum = accumulate_zoom(reader, blocks, options, bounds, z);
        std::array<uint64_t, 256> hist{};
        for (const auto& kv : accum) {
            for (uint16_t raw : kv.second) {
                const uint16_t capped = std::min<uint16_t>(raw, 255);
                hist[static_cast<size_t>(capped)] += 1;
            }
        }
        const int32_t threshold = options.autoAdjust
            ? quantile_threshold_from_histogram(hist, options.adjustQuantile)
            : 255;
        std::map<mlt_pmtiles::RasterTileKey, TileBuffer> sorted(accum.begin(), accum.end());
        for (const auto& kv : sorted) {
            Image2D<Rgb8> tile(kRasterTileSize, kRasterTileSize, Rgb8{0, 0, 0});
            bool nonEmpty = false;
            for (int32_t y = 0; y < kRasterTileSize; ++y) {
                for (int32_t x = 0; x < kRasterTileSize; ++x) {
                    const uint16_t raw = kv.second[static_cast<size_t>(y) *
                        static_cast<size_t>(kRasterTileSize) + static_cast<size_t>(x)];
                    const uint8_t gray = adjusted_intensity(raw, options.autoAdjust, threshold);
                    if (gray != 0) {
                        nonEmpty = true;
                    }
                    tile(y, x) = Rgb8{gray, gray, gray};
                }
            }
            if (!nonEmpty) {
                continue;
            }
            const std::string encoded = encode_png_rgb8(tile);
            mlt_pmtiles::append_png_tile_to_blob(
                blob, tempBlobFile, encoded, dataOffset, kv.first, tiles);
        }
        notice("%s: z%d wrote %zu mono raster tile(s); auto_adjust_threshold=%d",
            __func__, z, accum.size(), threshold);
    }
    blob.close();
    if (tiles.empty()) {
        error("%s: no raster tiles generated from %s", __func__, options.dataFile.c_str());
    }

    mlt_pmtiles::write_png_raster_pmtiles_archive_from_blob(
        options.outFile, tempBlobFile, std::move(tiles), bounds,
        options.minZoom, options.maxZoom);
    std::error_code ec;
    std::filesystem::remove(blobPath, ec);
}

} // namespace tiles2mono
