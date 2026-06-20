#include "image_pmtiles.hpp"

#include "geometry_utils.hpp"
#include "image_utils.hpp"
#include "pmtiles_utils.hpp"
#include "utils.h"
#include "utils_sys.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cinttypes>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <list>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;
using image_tiff::Gray16Bounds;
using image_tiff::TiffInfo;
using image_tiff::TiffLevel;
using image_tiff::decode_tiff_tile;
using image_tiff::estimate_gray16_bounds;
using image_tiff::parse_tiff_info;

namespace {

constexpr int32_t kRasterTileSize = 256;
constexpr double kEpsg3857Span = 40075016.6856;

struct RasterSample {
    Rgba8 value;
    bool inside = false;
};

class RasterSource {
public:
    virtual ~RasterSource() = default;
    virtual int width() const = 0;
    virtual int height() const = 0;
    virtual RasterSample sample_nearest(double px, double py) = 0;
    virtual bool supports_source_tile_planning() const { return false; }
    virtual std::vector<size_t> source_tiles_for_source_bbox(double x0, double y0,
        double x1, double y1) const {
        (void)x0;
        (void)y0;
        (void)x1;
        (void)y1;
        return {};
    }
    virtual void set_source_tile_remaining_counts(std::unordered_map<size_t, uint32_t> counts) {
        (void)counts;
    }
    virtual void release_source_tiles(const std::vector<size_t>& tileIds) {
        (void)tileIds;
    }
};

bool has_extension(const fs::path& path, const std::vector<std::string>& exts) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

class PngRasterSource : public RasterSource {
public:
    explicit PngRasterSource(const fs::path& path) : image_(load_png_rgba8(path.string())) {}

    int width() const override { return image_.width(); }
    int height() const override { return image_.height(); }

    RasterSample sample_nearest(double px, double py) override {
        if (!(px >= 0.0 && py >= 0.0 && px < static_cast<double>(image_.width()) &&
                py < static_cast<double>(image_.height()))) {
            return RasterSample{};
        }
        int32_t ix = std::clamp(static_cast<int32_t>(std::floor(px)), 0, image_.width() - 1);
        int32_t iy = std::clamp(static_cast<int32_t>(std::floor(py)), 0, image_.height() - 1);
        return RasterSample{image_(iy, ix), true};
    }

private:
    Image2D<Rgba8> image_;
};

class TiledTiffRasterSource : public RasterSource {
public:
    TiledTiffRasterSource(const fs::path& path,
        const image_pmtiles::Options& options,
        int levelIndex)
        : file_(path), info_(parse_tiff_info(file_)), levelIndex_(levelIndex) {
        if (levelIndex_ < 0 || static_cast<size_t>(levelIndex_) >= info_.levels.size()) {
            error("%s: TIFF source level %d is out of range for %s",
                __func__, levelIndex_, path.string().c_str());
        }
        const TiffLevel& level = selected_level();
        if (level.width > static_cast<uint64_t>(std::numeric_limits<int>::max()) ||
                level.height > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
            error("%s: TIFF dimensions exceed supported image size: %s", __func__, path.string().c_str());
        }
        bytesPerSample_ = level.bitsPerSample.front() / 8u;
        bytesPerPixel_ = checked_mul(static_cast<size_t>(level.samplesPerPixel), bytesPerSample_, __func__);
        tileBytes_ = checked_mul(checked_mul(static_cast<size_t>(level.tileWidth),
            static_cast<size_t>(level.tileHeight), __func__), bytesPerPixel_, __func__);
        tilesX_ = (level.width + level.tileWidth - 1) / level.tileWidth;
        gray_ = level.photometric == 0 || level.photometric == 1;
        gray16_ = gray_ && bytesPerSample_ == 2;
        if (gray16_) {
            grayBounds_ = estimate_gray16_bounds(file_, level, tileBytes_,
                info_.littleEndian,
                options.grayLowPercentile, options.grayHighPercentile,
                options.graySampleFraction, options.graySampleTiles, options.graySampleSeed);
        }
        cacheMaxBytes_ = options.tileCacheMb > 0
            ? static_cast<size_t>(options.tileCacheMb) * 1024u * 1024u
            : 0;
    }

    const TiffInfo& info() const { return info_; }
    int level_index() const { return levelIndex_; }

    int width() const override { return static_cast<int>(selected_level().width); }
    int height() const override { return static_cast<int>(selected_level().height); }

    RasterSample sample_nearest(double px, double py) override {
        const TiffLevel& level = selected_level();
        if (!(px >= 0.0 && py >= 0.0 && px < static_cast<double>(level.width) &&
                py < static_cast<double>(level.height))) {
            return RasterSample{};
        }
        const uint64_t ix = static_cast<uint64_t>(std::floor(px));
        const uint64_t iy = static_cast<uint64_t>(std::floor(py));
        const uint64_t tx = ix / level.tileWidth;
        const uint64_t ty = iy / level.tileHeight;
        const uint64_t localX = ix - tx * level.tileWidth;
        const uint64_t localY = iy - ty * level.tileHeight;
        const size_t tileIdx = static_cast<size_t>(ty * tilesX_ + tx);
        const Image2D<Rgba8>& tile = get_tile(tileIdx);
        return RasterSample{tile(static_cast<int>(localY), static_cast<int>(localX)), true};
    }

    bool supports_source_tile_planning() const override { return true; }

    std::vector<size_t> source_tiles_for_source_bbox(double x0, double y0,
        double x1, double y1) const override {
        const TiffLevel& level = selected_level();
        if (!std::isfinite(x0) || !std::isfinite(y0) || !std::isfinite(x1) || !std::isfinite(y1)) {
            return {};
        }
        const double minX = std::max(0.0, std::min(x0, x1));
        const double minY = std::max(0.0, std::min(y0, y1));
        const double maxX = std::min(static_cast<double>(level.width), std::max(x0, x1));
        const double maxY = std::min(static_cast<double>(level.height), std::max(y0, y1));
        if (!(minX < maxX && minY < maxY)) {
            return {};
        }

        const uint64_t px0 = static_cast<uint64_t>(std::floor(minX));
        const uint64_t py0 = static_cast<uint64_t>(std::floor(minY));
        const uint64_t px1 = static_cast<uint64_t>(std::ceil(maxX) - 1.0);
        const uint64_t py1 = static_cast<uint64_t>(std::ceil(maxY) - 1.0);
        const uint64_t tx0 = px0 / level.tileWidth;
        const uint64_t ty0 = py0 / level.tileHeight;
        const uint64_t tx1 = std::min<uint64_t>(tilesX_ - 1u, px1 / level.tileWidth);
        const uint64_t tilesY = (level.height + level.tileHeight - 1) / level.tileHeight;
        const uint64_t ty1 = std::min<uint64_t>(tilesY - 1u, py1 / level.tileHeight);

        std::vector<size_t> out;
        out.reserve(static_cast<size_t>((tx1 - tx0 + 1u) * (ty1 - ty0 + 1u)));
        for (uint64_t ty = ty0; ty <= ty1; ++ty) {
            for (uint64_t tx = tx0; tx <= tx1; ++tx) {
                out.push_back(static_cast<size_t>(ty * tilesX_ + tx));
            }
        }
        return out;
    }

    void set_source_tile_remaining_counts(std::unordered_map<size_t, uint32_t> counts) override {
        plannedRemaining_ = std::move(counts);
    }

    void release_source_tiles(const std::vector<size_t>& tileIds) override {
        for (size_t tileIdx : tileIds) {
            auto it = plannedRemaining_.find(tileIdx);
            if (it == plannedRemaining_.end() || it->second == 0) {
                continue;
            }
            --it->second;
            if (it->second == 0) {
                evict_tile(tileIdx);
                plannedRemaining_.erase(it);
            }
        }
    }

private:
    struct CacheEntry {
        Image2D<Rgba8> tile;
        std::list<size_t>::iterator lruIt;
        size_t bytes = 0;
    };

    const TiffLevel& selected_level() const {
        return info_.levels[static_cast<size_t>(levelIndex_)];
    }

    Image2D<Rgba8> decode_rgba_tile(size_t tileIdx) {
        const TiffLevel& level = selected_level();
        const std::vector<uint8_t> raw = decode_tiff_tile(file_, level, tileIdx, tileBytes_);
        Image2D<Rgba8> out(static_cast<int>(level.tileHeight), static_cast<int>(level.tileWidth), Rgba8{0, 0, 0, 0});
        for (uint64_t py = 0; py < level.tileHeight; ++py) {
            for (uint64_t px = 0; px < level.tileWidth; ++px) {
                const size_t srcOff = (static_cast<size_t>(py) * static_cast<size_t>(level.tileWidth) +
                    static_cast<size_t>(px)) * bytesPerPixel_;
                Rgba8 p{0, 0, 0, 255};
                if (gray_) {
                    uint8_t g = 0;
                    if (bytesPerSample_ == 1) {
                        g = raw[srcOff];
                    } else {
                        const uint16_t v = read_u16(raw, srcOff, info_.littleEndian);
                        const double scaled = (static_cast<double>(v) - grayBounds_.low) /
                            static_cast<double>(grayBounds_.high - grayBounds_.low) * 255.0;
                        g = clamp_u8(static_cast<float>(scaled));
                    }
                    if (level.photometric == 0) {
                        g = static_cast<uint8_t>(255u - g);
                    }
                    p = Rgba8{g, g, g, 255};
                } else {
                    p = Rgba8{raw[srcOff], raw[srcOff + 1], raw[srcOff + 2],
                        level.samplesPerPixel == 4 ? raw[srcOff + 3] : uint8_t{255}};
                }
                out(static_cast<int>(py), static_cast<int>(px)) = p;
            }
        }
        return out;
    }

    const Image2D<Rgba8>& get_tile(size_t tileIdx) {
        auto it = cache_.find(tileIdx);
        if (it != cache_.end()) {
            lru_.splice(lru_.begin(), lru_, it->second.lruIt);
            it->second.lruIt = lru_.begin();
            return it->second.tile;
        }
        Image2D<Rgba8> tile = decode_rgba_tile(tileIdx);
        const size_t bytes = tile.data().size() * sizeof(Rgba8);
        lru_.push_front(tileIdx);
        CacheEntry entry{std::move(tile), lru_.begin(), bytes};
        cacheBytes_ += bytes;
        auto inserted = cache_.emplace(tileIdx, std::move(entry)).first;
        enforce_cache_guard(tileIdx);
        return inserted->second.tile;
    }

    bool tile_still_planned(size_t tileIdx) const {
        const auto it = plannedRemaining_.find(tileIdx);
        return it != plannedRemaining_.end() && it->second > 0;
    }

    void evict_tile(size_t tileIdx) {
        auto it = cache_.find(tileIdx);
        if (it == cache_.end()) {
            return;
        }
        lru_.erase(it->second.lruIt);
        cacheBytes_ -= it->second.bytes;
        cache_.erase(it);
    }

    void enforce_cache_guard(size_t protectedTile) {
        if (cacheMaxBytes_ == 0) {
            return;
        }
        bool evicted = true;
        while (cacheBytes_ > cacheMaxBytes_ && cache_.size() > 1 && evicted) {
            evicted = false;
            for (auto lit = lru_.rbegin(); lit != lru_.rend(); ++lit) {
                const size_t victim = *lit;
                if (victim == protectedTile || tile_still_planned(victim)) {
                    continue;
                }
                evict_tile(victim);
                evicted = true;
                break;
            }
        }
        if (cacheBytes_ > cacheMaxBytes_ && !warnedCacheGuard_) {
            warnedCacheGuard_ = true;
            notice("%s: TIFF tile cache guard exceeded because planned source tiles are still active",
                __func__);
        }
    }

    RandomAccessFile file_;
    TiffInfo info_;
    int levelIndex_ = 0;
    size_t bytesPerSample_ = 1;
    size_t bytesPerPixel_ = 1;
    size_t tileBytes_ = 0;
    uint64_t tilesX_ = 0;
    bool gray_ = false;
    bool gray16_ = false;
    Gray16Bounds grayBounds_;
    size_t cacheMaxBytes_ = 0;
    size_t cacheBytes_ = 0;
    bool warnedCacheGuard_ = false;
    std::list<size_t> lru_;
    std::unordered_map<size_t, CacheEntry> cache_;
    std::unordered_map<size_t, uint32_t> plannedRemaining_;
};

bool invert_3x3(const std::array<double, 9>& m, std::array<double, 9>& inv) {
    const double det =
        m[0] * (m[4] * m[8] - m[5] * m[7]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);
    if (std::abs(det) < 1e-15 || !std::isfinite(det)) {
        return false;
    }
    const double r = 1.0 / det;
    inv[0] = (m[4] * m[8] - m[5] * m[7]) * r;
    inv[1] = (m[2] * m[7] - m[1] * m[8]) * r;
    inv[2] = (m[1] * m[5] - m[2] * m[4]) * r;
    inv[3] = (m[5] * m[6] - m[3] * m[8]) * r;
    inv[4] = (m[0] * m[8] - m[2] * m[6]) * r;
    inv[5] = (m[2] * m[3] - m[0] * m[5]) * r;
    inv[6] = (m[3] * m[7] - m[4] * m[6]) * r;
    inv[7] = (m[1] * m[6] - m[0] * m[7]) * r;
    inv[8] = (m[0] * m[4] - m[1] * m[3]) * r;
    return true;
}

std::pair<double, double> apply_transform(const std::array<double, 9>& m, double x, double y) {
    const double w = m[6] * x + m[7] * y + m[8];
    if (std::abs(w) < 1e-15) {
        return {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::quiet_NaN()};
    }
    return {
        (m[0] * x + m[1] * y + m[2]) / w,
        (m[3] * x + m[4] * y + m[5]) / w
    };
}

pm_raster::RasterBounds transform_bounds(const std::array<double, 9>& m, int width, int height) {
    std::array<std::pair<double, double>, 4> pts{
        apply_transform(m, 0.0, 0.0),
        apply_transform(m, static_cast<double>(width), 0.0),
        apply_transform(m, static_cast<double>(width), static_cast<double>(height)),
        apply_transform(m, 0.0, static_cast<double>(height))
    };
    pm_raster::RasterBounds b;
    b.xmin = b.ymin = std::numeric_limits<double>::infinity();
    b.xmax = b.ymax = -std::numeric_limits<double>::infinity();
    for (const auto& p : pts) {
        b.xmin = std::min(b.xmin, p.first);
        b.xmax = std::max(b.xmax, p.first);
        b.ymin = std::min(b.ymin, p.second);
        b.ymax = std::max(b.ymax, p.second);
    }
    return b;
}

uint32_t clamp_tile_coord_local(int64_t value, int32_t z) {
    const int64_t limit = (int64_t{1} << z) - 1;
    return static_cast<uint32_t>(std::clamp(value, int64_t{0}, limit));
}

std::array<double, 9> scaled_transform_for_level(const std::array<double, 9>& baseTransform,
    double sx, double sy) {
    return {
        baseTransform[0] * sx, baseTransform[1] * sy, baseTransform[2],
        baseTransform[3] * sx, baseTransform[4] * sy, baseTransform[5],
        baseTransform[6] * sx, baseTransform[7] * sy, baseTransform[8]
    };
}

double average_pixel_size(const std::array<double, 9>& transform) {
    const double sx = std::hypot(transform[0], transform[3]);
    const double sy = std::hypot(transform[1], transform[4]);
    if (sx > 0.0 && sy > 0.0) {
        return (sx + sy) * 0.5;
    }
    return std::max(sx, sy);
}

int choose_tiff_level(const TiffInfo& info, const image_pmtiles::Options& options,
    const std::array<double, 9>& baseTransform) {
    if (options.tiffSourceLevel == "base") {
        return 0;
    }
    if (options.tiffSourceLevel != "auto") {
        try {
            const int level = std::stoi(options.tiffSourceLevel);
            if (level < 0 || static_cast<size_t>(level) >= info.levels.size()) {
                error("%s: tiff source level %d is out of range", __func__, level);
            }
            return level;
        } catch (const std::exception&) {
            error("%s: --tiff-source-level must be auto, base, or a numeric level", __func__);
        }
    }
    if (info.levels.size() <= 1) {
        return 0;
    }
    const double basePixel = average_pixel_size(baseTransform);
    if (!(basePixel > 0.0)) {
        return 0;
    }
    const double outputPixel = kEpsg3857Span /
        (static_cast<double>(uint64_t{1} << options.maxZoom) * static_cast<double>(kRasterTileSize));
    int best = 0;
    for (size_t i = 1; i < info.levels.size(); ++i) {
        const TiffLevel& level = info.levels[i];
        const double sx = static_cast<double>(info.levels[0].width) / static_cast<double>(level.width);
        const double sy = static_cast<double>(info.levels[0].height) / static_cast<double>(level.height);
        const double levelPixel = basePixel * (sx + sy) * 0.5;
        if (levelPixel <= outputPixel * 1.25) {
            best = static_cast<int>(i);
        }
    }
    return best;
}

using TilePayloadMap = pm_raster::EncodedRasterTileMap;

struct PlannedImageTile {
    pm_raster::RasterTileKey key;
    std::vector<size_t> sourceTiles;
    uint64_t order = 0;
};

struct SourceBbox {
    double x0 = 0.0;
    double y0 = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    bool valid = false;
};

SourceBbox estimate_source_bbox_for_tile(const pm_raster::RasterTileKey& key,
    const std::array<double, 9>& inv) {
    SourceBbox out;
    std::array<std::pair<double, double>, 4> corners{};
    std::array<std::pair<double, double>, 4> pixels{
        std::pair<double, double>{0.0, 0.0},
        std::pair<double, double>{static_cast<double>(kRasterTileSize), 0.0},
        std::pair<double, double>{static_cast<double>(kRasterTileSize), static_cast<double>(kRasterTileSize)},
        std::pair<double, double>{0.0, static_cast<double>(kRasterTileSize)}
    };
    for (size_t i = 0; i < pixels.size(); ++i) {
        double x = 0.0, y = 0.0;
        pm_core::tilecoord_to_epsg3857(key.x, key.y, pixels[i].first, pixels[i].second,
            key.z, x, y);
        corners[i] = apply_transform(inv, x, y);
        if (!std::isfinite(corners[i].first) || !std::isfinite(corners[i].second)) {
            return out;
        }
    }

    out.x0 = out.y0 = std::numeric_limits<double>::infinity();
    out.x1 = out.y1 = -std::numeric_limits<double>::infinity();
    for (const auto& p : corners) {
        out.x0 = std::min(out.x0, p.first);
        out.x1 = std::max(out.x1, p.first);
        out.y0 = std::min(out.y0, p.second);
        out.y1 = std::max(out.y1, p.second);
    }
    out.x0 -= 1.0;
    out.y0 -= 1.0;
    out.x1 += 1.0;
    out.y1 += 1.0;
    out.valid = true;
    return out;
}

std::vector<PlannedImageTile> plan_max_zoom_tiles(RasterSource& src,
    const pm_raster::RasterBounds& bounds,
    const std::array<double, 9>& inv,
    int32_t maxZoom) {
    int64_t txA = 0, tyA = 0, txB = 0, tyB = 0;
    double lx = 0.0, ly = 0.0;
    pm_core::epsg3857_to_tilecoord(bounds.xmin, bounds.ymax, static_cast<uint8_t>(maxZoom), txA, tyA, lx, ly);
    pm_core::epsg3857_to_tilecoord(bounds.xmax, bounds.ymin, static_cast<uint8_t>(maxZoom), txB, tyB, lx, ly);
    const uint32_t tx0 = clamp_tile_coord_local(std::min(txA, txB), maxZoom);
    const uint32_t tx1 = clamp_tile_coord_local(std::max(txA, txB), maxZoom);
    const uint32_t ty0 = clamp_tile_coord_local(std::min(tyA, tyB), maxZoom);
    const uint32_t ty1 = clamp_tile_coord_local(std::max(tyA, tyB), maxZoom);

    std::vector<PlannedImageTile> planned;
    planned.reserve(static_cast<size_t>(tx1 - tx0 + 1u) * static_cast<size_t>(ty1 - ty0 + 1u));
    std::unordered_map<size_t, uint32_t> remainingCounts;
    bool canPlanSourceTiles = src.supports_source_tile_planning();
    for (uint32_t ty = ty0; ty <= ty1; ++ty) {
        for (uint32_t tx = tx0; tx <= tx1; ++tx) {
            PlannedImageTile plannedTile;
            plannedTile.key = pm_raster::RasterTileKey{static_cast<uint8_t>(maxZoom), tx, ty};
            plannedTile.order = hilbert_index_2d(tx, ty, maxZoom);
            if (canPlanSourceTiles) {
                const SourceBbox srcBbox = estimate_source_bbox_for_tile(plannedTile.key, inv);
                if (srcBbox.valid) {
                    plannedTile.sourceTiles = src.source_tiles_for_source_bbox(srcBbox.x0, srcBbox.y0,
                        srcBbox.x1, srcBbox.y1);
                    for (size_t sourceTile : plannedTile.sourceTiles) {
                        ++remainingCounts[sourceTile];
                    }
                } else {
                    canPlanSourceTiles = false;
                    remainingCounts.clear();
                    for (PlannedImageTile& prior : planned) {
                        prior.sourceTiles.clear();
                    }
                }
            }
            planned.push_back(std::move(plannedTile));
        }
    }

    std::sort(planned.begin(), planned.end(), [](const PlannedImageTile& a, const PlannedImageTile& b) {
        if (a.order != b.order) {
            return a.order < b.order;
        }
        if (a.key.y != b.key.y) {
            return a.key.y < b.key.y;
        }
        return a.key.x < b.key.x;
    });
    if (canPlanSourceTiles) {
        src.set_source_tile_remaining_counts(std::move(remainingCounts));
        notice("%s: z%d planned %zu image tile(s) in Hilbert order with source-tile coverage eviction",
            __func__, maxZoom, planned.size());
    } else {
        notice("%s: z%d planned %zu image tile(s) in Hilbert order",
            __func__, maxZoom, planned.size());
    }
    return planned;
}

void append_encoded_tile(std::ofstream& blob,
    const fs::path& tempBlobFile,
    const pm_raster::RasterTileKey& key,
    const std::string& encoded,
    uint64_t& dataOffset,
    std::vector<pm_core::StoredTilePayloadRef>& tiles) {
    pm_raster::append_png_tile_to_blob(blob, tempBlobFile.string(), encoded, dataOffset, key, tiles);
}

TilePayloadMap write_max_zoom_tiles(RasterSource& src,
    const std::vector<PlannedImageTile>& plannedTiles,
    const pm_raster::RasterBounds& bounds,
    const std::array<double, 9>& inv,
    int32_t maxZoom,
    std::ofstream& blob,
    const fs::path& tempBlobFile,
    uint64_t& dataOffset,
    std::vector<pm_core::StoredTilePayloadRef>& tiles) {
    TilePayloadMap out;
    size_t nTiles = 0;
    for (const PlannedImageTile& plannedTile : plannedTiles) {
        const uint32_t tx = plannedTile.key.x;
        const uint32_t ty = plannedTile.key.y;
        Image2D<Rgba8> tile(kRasterTileSize, kRasterTileSize, Rgba8{0, 0, 0, 0});
        bool nonEmpty = false;
        for (int32_t py = 0; py < kRasterTileSize; ++py) {
            for (int32_t px = 0; px < kRasterTileSize; ++px) {
                double x = 0.0, y = 0.0;
                pm_core::tilecoord_to_epsg3857(tx, ty,
                    static_cast<double>(px) + 0.5,
                    static_cast<double>(py) + 0.5,
                    static_cast<uint8_t>(maxZoom), x, y);
                if (x < bounds.xmin || x > bounds.xmax || y < bounds.ymin || y > bounds.ymax) {
                    continue;
                }
                const auto srcPt = apply_transform(inv, x, y);
                RasterSample sample = src.sample_nearest(srcPt.first, srcPt.second);
                if (!sample.inside) {
                    continue;
                }
                tile(py, px) = sample.value;
                if (sample.value.a != 0) {
                    nonEmpty = true;
                }
            }
        }
        if (!nonEmpty) {
            src.release_source_tiles(plannedTile.sourceTiles);
            continue;
        }
        std::string encoded = encode_png_rgba8(tile);
        append_encoded_tile(blob, tempBlobFile, plannedTile.key, encoded, dataOffset, tiles);
        out.emplace(plannedTile.key, std::move(encoded));
        src.release_source_tiles(plannedTile.sourceTiles);
        ++nTiles;
    }
    notice("%s: z%d wrote %zu image raster tile(s)",
        __func__, maxZoom, nTiles);
    return out;
}

void write_transformed_image_pmtiles(RasterSource& src,
    const fs::path& outFile,
    const fs::path& tempBlobFile,
    const std::array<double, 9>& transform,
    int32_t minZoom,
    int32_t maxZoom) {
    std::array<double, 9> inv;
    if (!invert_3x3(transform, inv)) {
        error("%s: image transform is singular", __func__);
    }
    const pm_raster::RasterBounds bounds = transform_bounds(transform, src.width(), src.height());
    pm_raster::validate_raster_archive_options(bounds, minZoom, maxZoom, __func__);

    if (tempBlobFile.has_parent_path()) {
        fs::create_directories(tempBlobFile.parent_path());
    }
    std::ofstream blob(tempBlobFile, std::ios::binary | std::ios::trunc);
    if (!blob.is_open()) {
        error("%s: cannot open temporary PMTiles blob %s", __func__, tempBlobFile.string().c_str());
    }

    std::vector<pm_core::StoredTilePayloadRef> tiles;
    uint64_t dataOffset = 0;
    std::vector<PlannedImageTile> plannedTiles = plan_max_zoom_tiles(src, bounds, inv, maxZoom);
    TilePayloadMap current = write_max_zoom_tiles(src, plannedTiles, bounds, inv, maxZoom, blob, tempBlobFile,
        dataOffset, tiles);
    for (int32_t z = maxZoom - 1; z >= minZoom; --z) {
        current = pm_raster::write_rgba_png_parent_zoom(static_cast<uint8_t>(z), current, blob,
            tempBlobFile.string(),
            dataOffset, tiles);
        notice("%s: z%d wrote %zu image raster tile(s)",
            __func__, z, current.size());
        if (current.empty()) {
            break;
        }
    }
    blob.close();
    pm_raster::write_png_raster_pmtiles_archive_from_blob(outFile.string(),
        tempBlobFile.string(), std::move(tiles), bounds, minZoom, maxZoom);
    std::error_code ec;
    fs::remove(tempBlobFile, ec);
}

json make_image_asset_json(const image_pmtiles::Asset& asset) {
    json out = json::object();
    out[asset.id] = asset.pmtilesPath.filename().string();
    return out;
}

void write_image_asset_json(const image_pmtiles::Asset& asset, bool overwrite) {
    write_text_checked(asset.assetJson, make_image_asset_json(asset).dump(4) + "\n",
        overwrite, "image asset JSON");
}

std::array<double, 9> resolve_base_transform(const image_pmtiles::Options& options) {
    if (options.hasTransform) {
        return options.transform;
    }
    return {
        options.micronsPerPixel, 0.0, options.offsetXUm,
        0.0, options.micronsPerPixel, options.offsetYUm,
        0.0, 0.0, 1.0
    };
}

std::array<double, 9> resolve_pix_reference_transform(
    const image_pmtiles::Options& options, double width, double height) {
    if (!(width > 0.0 && height > 0.0)) {
        error("%s: image dimensions must be positive for pix reference alignment", __func__);
    }
    const double x1 = options.pixZero[0];
    const double y1 = options.pixZero[1];
    const double x2 = options.pixMax[0];
    const double y2 = options.pixMax[1];
    if (!std::isfinite(x1) || !std::isfinite(y1) ||
        !std::isfinite(x2) || !std::isfinite(y2)) {
        error("%s: --pix-zero and --pix-max values must be finite", __func__);
    }
    const double dx = x2 - x1;
    const double dy = y2 - y1;
    if (!(dx != 0.0 || dy != 0.0)) {
        error("%s: --pix-zero and --pix-max must specify distinct micron coordinates", __func__);
    }
    const double denom = width * width + height * height;
    const double a = (width * dx + height * dy) / denom;
    const double b = (-height * dx + width * dy) / denom;
    const double scale = std::hypot(a, b);
    if (!(scale > 0.0) || !std::isfinite(scale)) {
        error("%s: invalid pix reference alignment scale", __func__);
    }
    return {
        a, -b, x1,
        b, a, y1,
        0.0, 0.0, 1.0
    };
}

std::array<double, 9> resolve_dimension_dependent_base_transform(
    const image_pmtiles::Options& options, double width, double height) {
    if (options.hasPixZero || options.hasPixMax) {
        return resolve_pix_reference_transform(options, width, height);
    }
    return resolve_base_transform(options);
}

std::unique_ptr<RasterSource> make_raster_source(
    const image_pmtiles::Options& options,
    std::array<double, 9>& sourceTransform) {
    if (has_extension(options.inImage, {".png"})) {
        std::unique_ptr<RasterSource> source(new PngRasterSource(options.inImage));
        sourceTransform = resolve_dimension_dependent_base_transform(options,
            static_cast<double>(source->width()),
            static_cast<double>(source->height()));
        return source;
    }
    if (has_extension(options.inImage, {".tif", ".tiff", ".btf"})) {
        RandomAccessFile probe(options.inImage);
        TiffInfo info = parse_tiff_info(probe);
        const TiffLevel& base = info.levels.front();
        const std::array<double, 9> baseTransform = resolve_dimension_dependent_base_transform(options,
            static_cast<double>(base.width),
            static_cast<double>(base.height));
        const int levelIndex = choose_tiff_level(info, options, baseTransform);
        const TiffLevel& level = info.levels[static_cast<size_t>(levelIndex)];
        const double sx = static_cast<double>(base.width) / static_cast<double>(level.width);
        const double sy = static_cast<double>(base.height) / static_cast<double>(level.height);
        sourceTransform = scaled_transform_for_level(baseTransform, sx, sy);
        notice("%s: using TIFF source level %d (%" PRIu64 "x%" PRIu64 ") for %s",
            __func__, levelIndex, level.width, level.height, options.inImage.string().c_str());
        return std::unique_ptr<RasterSource>(new TiledTiffRasterSource(options.inImage, options, levelIndex));
    }
    error("%s: unsupported image extension for %s", __func__, options.inImage.string().c_str());
    return nullptr;
}

} // namespace

void image_pmtiles::validate_options(const Options& options) {
    if (options.id.empty()) {
        error("%s: --id is required", __func__);
    }
    if (options.inImage.empty()) {
        error("%s: --in-image is required", __func__);
    }
    require_file(options.inImage, "image input");
    if (options.outPrefix.empty()) {
        error("%s: --out-prefix is required", __func__);
    }
    if (options.minZoom < 0 || options.maxZoom < options.minZoom || options.maxZoom > 30) {
        error("%s: invalid zoom range %d..%d", __func__, options.minZoom, options.maxZoom);
    }
    if (!(options.grayLowPercentile >= 0.0 &&
            options.grayLowPercentile < options.grayHighPercentile &&
            options.grayHighPercentile <= 100.0)) {
        error("%s: invalid grayscale percentile range %.6g..%.6g; require 0 <= low < high <= 100",
            __func__, options.grayLowPercentile, options.grayHighPercentile);
    }
    if (!(options.graySampleFraction >= 0.0 && options.graySampleFraction <= 1.0)) {
        error("%s: --gray-sample-fraction must be in [0, 1]", __func__);
    }
    if (options.graySampleTiles < 0) {
        error("%s: --gray-sample-tiles must be non-negative", __func__);
    }
    if (options.tileCacheMb < 0) {
        error("%s: --tile-cache-mb must be non-negative", __func__);
    }
    if (options.hasPixZero != options.hasPixMax) {
        error("%s: --pix-zero and --pix-max must be specified together", __func__);
    }
    const bool hasPixReference = options.hasPixZero && options.hasPixMax;
    const bool hasScaleOnly = options.hasMicronsPerPixel;
    const int transformSources = (options.hasTransform ? 1 : 0) +
        (hasScaleOnly ? 1 : 0) + (hasPixReference ? 1 : 0);
    if (transformSources != 1) {
        error("%s: provide exactly one image alignment source: --microns-per-pixel, --transform, or --pix-zero with --pix-max",
            __func__);
    }
    if ((options.hasOffsetXUm || options.hasOffsetYUm) && !hasScaleOnly) {
        error("%s: --offset-x-um and --offset-y-um require --microns-per-pixel", __func__);
    }
    if (hasPixReference && (options.hasOffsetXUm || options.hasOffsetYUm)) {
        error("%s: --pix-zero/--pix-max cannot be combined with --offset-x-um or --offset-y-um",
            __func__);
    }
    if (hasScaleOnly && !(options.micronsPerPixel > 0.0)) {
        error("%s: --microns-per-pixel must be positive", __func__);
    }
}

image_pmtiles::Asset image_pmtiles::write_image_pmtiles(const Options& options) {
    validate_options(options);
    fs::path outFile = options.outPrefix;
    outFile += ".pmtiles";
    if (!options.overwrite && file_exists(outFile)) {
        notice("%s: %s already exists; skipping image PMTiles", __func__, outFile.string().c_str());
    } else {
        std::array<double, 9> sourceTransform{
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0};
        std::unique_ptr<RasterSource> source = make_raster_source(options, sourceTransform);
        const fs::path blobFile = options.outPrefix.string() + ".blob.tmp";
        write_transformed_image_pmtiles(*source, outFile, blobFile, sourceTransform,
            options.minZoom, options.maxZoom);
    }

    Asset asset;
    asset.id = options.id;
    asset.pmtilesPath = outFile;
    asset.assetJson = options.assetJson.empty()
        ? fs::path(options.outPrefix.string() + "_assets.json")
        : options.assetJson;
    write_image_asset_json(asset, options.overwrite);
    return asset;
}
