#include "pmtiles_utils.hpp"

#include "image_utils.hpp"
#include "PMTiles/pmtiles.hpp"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <tuple>
#include <zlib.h>

namespace {

constexpr int kRasterTileSize = 256;

std::string maybe_decompress_pmtiles_blob(const std::string& data, uint8_t compression) {
    switch (compression) {
    case pmtiles::COMPRESSION_NONE:
        return data;
    case pmtiles::COMPRESSION_GZIP:
        return pm_core::gzip_decompress(data);
    default:
        error("%s: unsupported PMTiles compression mode %u", __func__, static_cast<unsigned>(compression));
        return std::string();
    }
}

const char* geometry_type_name(pm_vector::VectorGeometryType geometryType) {
    switch (geometryType) {
    case pm_vector::VectorGeometryType::Point:
        return "Point";
    case pm_vector::VectorGeometryType::Polygon:
        return "Polygon";
    default:
        error("%s: unsupported vector geometry type", __func__);
        return "";
    }
}

int32_t lonlat_e7(double value) {
    return static_cast<int32_t>(std::llround(value * 10000000.0));
}

uint32_t clamp_tile_coord(int64_t value, int32_t z) {
    const int64_t limit = (int64_t{1} << z) - 1;
    return static_cast<uint32_t>(std::clamp(value, int64_t{0}, limit));
}

Rgb8 sample_nearest(const Image2D<Rgb8>& src, double x, double y, const pm_raster::RasterBounds& bounds) {
    const double sx = (x - bounds.xmin) / (bounds.xmax - bounds.xmin) * static_cast<double>(src.width() - 1);
    const double sy = (y - bounds.ymin) / (bounds.ymax - bounds.ymin) * static_cast<double>(src.height() - 1);
    int32_t ix = static_cast<int32_t>(std::llround(sx));
    int32_t iy = static_cast<int32_t>(std::llround(sy));
    ix = std::max(0, std::min(src.width() - 1, ix));
    iy = std::max(0, std::min(src.height() - 1, iy));
    return src(iy, ix);
}

} // namespace

std::string pm_core::gzip_compress(const std::string& data) {
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

std::string pm_core::gzip_decompress(const std::string& data) {
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

void pm_core::epsg3857_to_wgs84(double x, double y, double& lon, double& lat) {
    constexpr double kEpsg3857Radius = 6378137.0;
    lon = (x / kEpsg3857Radius) * (180.0 / M_PI);
    lat = (2.0 * std::atan(std::exp(y / kEpsg3857Radius)) - M_PI / 2.0) * (180.0 / M_PI);
}

double pm_core::epsg3857_scale_factor(uint8_t zoom) {
    constexpr double kEpsg3857Bound = 20037508.3428;
    return 2.0 * kEpsg3857Bound / static_cast<double>(uint64_t{1} << (zoom + 12));
}

void pm_core::epsg3857_to_tilecoord(double x, double y, uint8_t zoom,
    int64_t& tileX, int64_t& tileY, double& localX, double& localY) {
    if (!std::isfinite(x)) x = 40000000.0;
    if (!std::isfinite(y)) y = 40000000.0;

    constexpr double kEpsg3857Bound = 20037508.3428;
    constexpr double tileSize = 256.0;
    const uint64_t numTiles = uint64_t{1} << zoom;
    tileX = static_cast<int64_t>((x + kEpsg3857Bound) / (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)));
    tileY = static_cast<int64_t>((kEpsg3857Bound - y) / (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)));

    const double tileOriginX = static_cast<double>(tileX) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)) - kEpsg3857Bound;
    const double tileOriginY = kEpsg3857Bound - static_cast<double>(tileY) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles));
    localX = (x - tileOriginX) / (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
    localY = (tileOriginY - y) / (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
}

void pm_core::tilecoord_to_epsg3857(int64_t tileX, int64_t tileY,
    double localX, double localY, uint8_t zoom,
    double& x, double& y) {
    constexpr double kEpsg3857Bound = 20037508.3428;
    constexpr double tileSize = 256.0;
    const uint64_t numTiles = uint64_t{1} << zoom;
    const double tileOriginX = static_cast<double>(tileX) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)) - kEpsg3857Bound;
    const double tileOriginY = kEpsg3857Bound - static_cast<double>(tileY) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles));
    x = tileOriginX + localX * (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
    y = tileOriginY - localY * (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
}

size_t pm_raster::RasterTileKeyHash::operator()(const RasterTileKey& key) const {
    uint64_t v = static_cast<uint64_t>(key.z);
    v = (v << 32u) ^ static_cast<uint64_t>(key.x);
    v = (v << 32u) ^ static_cast<uint64_t>(key.y);
    v ^= v >> 33u;
    v *= 0xff51afd7ed558ccdULL;
    v ^= v >> 33u;
    return static_cast<size_t>(v);
}

void pm_raster::validate_raster_archive_options(const RasterBounds& bounds,
    int32_t minZoom,
    int32_t maxZoom,
    const char* context) {
    if (!bounds.valid()) {
        error("%s: invalid raster bounds", context);
    }
    if (minZoom < 0 || maxZoom < minZoom || maxZoom > 30) {
        error("%s: invalid zoom range %d..%d", context, minZoom, maxZoom);
    }
}

pm_raster::RasterPixelCoord pm_raster::epsg3857_to_raster_pixel(double x, double y, int32_t zoom) {
    int64_t tx = 0;
    int64_t ty = 0;
    double localX = 0.0;
    double localY = 0.0;
    pm_core::epsg3857_to_tilecoord(x, y, static_cast<uint8_t>(zoom), tx, ty, localX, localY);
    RasterPixelCoord out;
    out.key.z = static_cast<uint8_t>(zoom);
    out.key.x = clamp_tile_coord(tx, zoom);
    out.key.y = clamp_tile_coord(ty, zoom);
    out.px = std::clamp(static_cast<int32_t>(std::floor(localX)), 0, kRasterTileSize - 1);
    out.py = std::clamp(static_cast<int32_t>(std::floor(localY)), 0, kRasterTileSize - 1);
    return out;
}

void pm_raster::append_png_tile_to_blob(std::ofstream& blob,
    const std::string& tempBlobFile,
    const std::string& encoded,
    uint64_t& dataOffset,
    const RasterTileKey& key,
    std::vector<pm_core::StoredTilePayloadRef>& tiles) {
    blob.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
    if (!blob.good()) {
        error("%s: failed writing temporary PMTiles blob %s", __func__, tempBlobFile.c_str());
    }
    pm_core::StoredTilePayloadRef ref;
    ref.z = key.z;
    ref.x = key.x;
    ref.y = key.y;
    ref.featureCount = 1;
    ref.tileId = pmtiles::zxy_to_tileid(ref.z, ref.x, ref.y);
    ref.dataOffset = dataOffset;
    ref.dataLength = static_cast<uint32_t>(encoded.size());
    dataOffset += encoded.size();
    tiles.push_back(ref);
}

void pm_raster::write_png_raster_pmtiles_archive_from_blob(const std::string& outFile,
    const std::string& tempBlobFile,
    std::vector<pm_core::StoredTilePayloadRef> tiles,
    const RasterBounds& bounds,
    int32_t minZoom,
    int32_t maxZoom) {
    validate_raster_archive_options(bounds, minZoom, maxZoom, __func__);
    if (tiles.empty()) {
        error("%s: no raster tiles generated for %s", __func__, outFile.c_str());
    }

    double minLon = 0.0, minLat = 0.0, maxLon = 0.0, maxLat = 0.0;
    pm_core::epsg3857_to_wgs84(bounds.xmin, bounds.ymin, minLon, minLat);
    pm_core::epsg3857_to_wgs84(bounds.xmax, bounds.ymax, maxLon, maxLat);
    nlohmann::json metadata = {
        {"version", "1.1"},
        {"name", std::filesystem::path(outFile).filename().string()},
        {"description", std::filesystem::path(outFile).filename().string()},
        {"format", "png"},
        {"type", "overlay"},
        {"minzoom", minZoom},
        {"maxzoom", maxZoom}
    };

    pm_core::ArchiveOptions options;
    options.tileType = pmtiles::TILETYPE_PNG;
    options.tileCompression = pmtiles::COMPRESSION_NONE;
    options.minZoom = static_cast<uint8_t>(minZoom);
    options.maxZoom = static_cast<uint8_t>(maxZoom);
    options.centerZoom = static_cast<uint8_t>(minZoom);
    options.hasGeographicBounds = true;
    options.minLonE7 = lonlat_e7(minLon);
    options.minLatE7 = lonlat_e7(minLat);
    options.maxLonE7 = lonlat_e7(maxLon);
    options.maxLatE7 = lonlat_e7(maxLat);
    options.centerLonE7 = lonlat_e7((minLon + maxLon) * 0.5);
    options.centerLatE7 = lonlat_e7((minLat + maxLat) * 0.5);
    options.metadata = std::move(metadata);
    pm_core::write_pmtiles_archive_from_blob_file(outFile, tempBlobFile, std::move(tiles), options);
}

namespace {

Image2D<Rgba8> build_rgba_parent_tile(uint32_t parentX, uint32_t parentY,
    const pm_raster::EncodedRasterTileMap& children,
    uint8_t childZoom) {
    std::map<pm_raster::RasterTileKey, Image2D<Rgba8>> decodedChildren;
    for (uint32_t cy = 0; cy < 2; ++cy) {
        for (uint32_t cx = 0; cx < 2; ++cx) {
            const pm_raster::RasterTileKey childKey{
                childZoom,
                parentX * 2u + cx,
                parentY * 2u + cy};
            auto it = children.find(childKey);
            if (it != children.end()) {
                decodedChildren.emplace(childKey, decode_png_rgba8(it->second));
            }
        }
    }
    Image2D<Rgba8> parent(kRasterTileSize, kRasterTileSize, Rgba8{0, 0, 0, 0});
    for (int32_t py = 0; py < kRasterTileSize; ++py) {
        for (int32_t px = 0; px < kRasterTileSize; ++px) {
            uint32_t sumA = 0;
            uint32_t sumPremulR = 0;
            uint32_t sumPremulG = 0;
            uint32_t sumPremulB = 0;
            for (uint32_t dy = 0; dy < 2; ++dy) {
                for (uint32_t dx = 0; dx < 2; ++dx) {
                    const uint64_t globalX = static_cast<uint64_t>(parentX) * 512u +
                        static_cast<uint64_t>(px) * 2u + dx;
                    const uint64_t globalY = static_cast<uint64_t>(parentY) * 512u +
                        static_cast<uint64_t>(py) * 2u + dy;
                    const pm_raster::RasterTileKey childKey{
                        childZoom,
                        static_cast<uint32_t>(globalX / kRasterTileSize),
                        static_cast<uint32_t>(globalY / kRasterTileSize)};
                    auto it = decodedChildren.find(childKey);
                    if (it == decodedChildren.end()) {
                        continue;
                    }
                    const Rgba8 c = it->second(static_cast<int>(globalY % kRasterTileSize),
                        static_cast<int>(globalX % kRasterTileSize));
                    sumA += c.a;
                    sumPremulR += static_cast<uint32_t>(c.r) * c.a;
                    sumPremulG += static_cast<uint32_t>(c.g) * c.a;
                    sumPremulB += static_cast<uint32_t>(c.b) * c.a;
                }
            }
            if (sumA == 0) {
                continue;
            }
            const uint8_t a = static_cast<uint8_t>((sumA + 2u) / 4u);
            parent(py, px) = Rgba8{
                static_cast<uint8_t>((sumPremulR + sumA / 2u) / sumA),
                static_cast<uint8_t>((sumPremulG + sumA / 2u) / sumA),
                static_cast<uint8_t>((sumPremulB + sumA / 2u) / sumA),
                a};
        }
    }
    return parent;
}

bool tile_has_alpha(const Image2D<Rgba8>& tile) {
    for (const Rgba8& p : tile.data()) {
        if (p.a != 0) {
            return true;
        }
    }
    return false;
}

} // namespace

pm_raster::EncodedRasterTileMap pm_raster::write_rgba_png_parent_zoom(uint8_t parentZoom,
    const EncodedRasterTileMap& children,
    std::ofstream& blob,
    const std::string& tempBlobFile,
    uint64_t& dataOffset,
    std::vector<pm_core::StoredTilePayloadRef>& tiles) {
    std::set<std::pair<uint32_t, uint32_t>> parents;
    for (const auto& kv : children) {
        parents.insert({kv.first.x / 2u, kv.first.y / 2u});
    }
    EncodedRasterTileMap out;
    for (const auto& parentKey : parents) {
        Image2D<Rgba8> tile = build_rgba_parent_tile(parentKey.first, parentKey.second, children,
            static_cast<uint8_t>(parentZoom + 1u));
        if (!tile_has_alpha(tile)) {
            continue;
        }
        RasterTileKey key{parentZoom, parentKey.first, parentKey.second};
        std::string encoded = encode_png_rgba8(tile);
        append_png_tile_to_blob(blob, tempBlobFile, encoded, dataOffset, key, tiles);
        out.emplace(key, std::move(encoded));
    }
    return out;
}

pm_core::LoadedPmtilesArchive pm_core::load_pmtiles_archive(const std::string& inFile) {
    auto reader = flexio::FlexReaderFactory::create_reader(inFile);
    if (reader == nullptr || !reader->is_open()) {
        error("%s: cannot open PMTiles source %s", __func__, inFile.c_str());
    }

    std::string headerBytes(127, '\0');
    if (!reader->read_at(0, headerBytes.size(), headerBytes)) {
        error("%s: failed to read PMTiles header from %s", __func__, inFile.c_str());
    }

    LoadedPmtilesArchive out;
    out.reader = std::move(reader);
    out.header = pmtiles::deserialize_header(headerBytes);
    std::string directoryBytes(static_cast<size_t>(out.header.tile_data_offset), '\0');
    if (!out.reader->read_at(0, directoryBytes.size(), directoryBytes)) {
        error("%s: failed to read PMTiles directory bytes from %s", __func__, inFile.c_str());
    }

    const auto decompressFn = [](const std::string& data, uint8_t compression) {
        return maybe_decompress_pmtiles_blob(data, compression);
    };
    out.entries = pmtiles::entries_tms(decompressFn, directoryBytes.c_str());
    const std::string compressedMetadata = directoryBytes.substr(
        static_cast<size_t>(out.header.json_metadata_offset),
        static_cast<size_t>(out.header.json_metadata_bytes));
    out.metadata = nlohmann::json::parse(
        maybe_decompress_pmtiles_blob(compressedMetadata, out.header.internal_compression));
    return out;
}

std::string pm_core::read_pmtiles_tile_payload(flexio::FlexReader& reader,
    const pmtiles::headerv3& header,
    const pmtiles::entry_zxy& entry) {
    std::string compressed(static_cast<size_t>(entry.length), '\0');
    if (!reader.read_at(entry.offset, compressed.size(), compressed)) {
        error("%s: failed to read PMTiles tile z=%u x=%u y=%u", __func__,
            static_cast<unsigned>(entry.z), entry.x, entry.y);
    }
    return maybe_decompress_pmtiles_blob(compressed, header.tile_compression);
}

void pm_core::write_pmtiles_archive(const std::string& outFile,
    std::vector<EncodedTilePayload> tiles,
    const ArchiveOptions& options) {
    std::sort(tiles.begin(), tiles.end(), [](const EncodedTilePayload& lhs, const EncodedTilePayload& rhs) {
        return lhs.tileId < rhs.tileId;
    });

    std::vector<pmtiles::entryv3> entries;
    entries.reserve(tiles.size());
    uint64_t currentTileOffset = 0;
    for (const auto& tile : tiles) {
        entries.emplace_back(tile.tileId, currentTileOffset,
            static_cast<uint32_t>(tile.compressedData.size()), 1);
        currentTileOffset += tile.compressedData.size();
    }

    const auto compressFn = [](const std::string& data, uint8_t /*compression*/) {
        return gzip_compress(data);
    };
    std::string rootBytes;
    std::string leavesBytes;
    int numLeaves = 0;
    std::tie(rootBytes, leavesBytes, numLeaves) =
        pmtiles::make_root_leaves(compressFn, pmtiles::COMPRESSION_GZIP, entries);
    (void)numLeaves;

    const std::string metadataJson = options.metadata.dump();
    const std::string compressedMetadata = gzip_compress(metadataJson);

    pmtiles::headerv3 header{};
    header.root_dir_offset = 127;
    header.root_dir_bytes = rootBytes.size();
    header.json_metadata_offset = header.root_dir_offset + header.root_dir_bytes;
    header.json_metadata_bytes = compressedMetadata.size();
    header.leaf_dirs_offset = header.json_metadata_offset + header.json_metadata_bytes;
    header.leaf_dirs_bytes = leavesBytes.size();
    header.tile_data_offset = header.leaf_dirs_offset + header.leaf_dirs_bytes;
    header.tile_data_bytes = currentTileOffset;
    header.addressed_tiles_count = entries.size();
    header.tile_entries_count = entries.size();
    header.tile_contents_count = entries.size();
    header.clustered = options.clustered;
    header.internal_compression = pmtiles::COMPRESSION_GZIP;
    header.tile_compression = options.tileCompression;
    header.tile_type = options.tileType;
    header.min_zoom = options.minZoom;
    header.max_zoom = options.maxZoom;
    header.center_zoom = options.centerZoom;
    if (options.hasGeographicBounds) {
        header.min_lon_e7 = options.minLonE7;
        header.min_lat_e7 = options.minLatE7;
        header.max_lon_e7 = options.maxLonE7;
        header.max_lat_e7 = options.maxLatE7;
        header.center_lon_e7 = options.centerLonE7;
        header.center_lat_e7 = options.centerLatE7;
    } else {
        // Generic tiled coordinate spaces still need a non-zero PMTiles bbox
        // for validators and clients that require one.
        header.min_lon_e7 = -1800000000;
        header.min_lat_e7 = -850000000;
        header.max_lon_e7 = 1800000000;
        header.max_lat_e7 = 850000000;
        header.center_lon_e7 = 0;
        header.center_lat_e7 = 0;
    }

    std::ofstream out(outFile, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        error("%s: cannot open output PMTiles file %s", __func__, outFile.c_str());
    }
    const std::string headerBytes = header.serialize();
    out.write(headerBytes.data(), static_cast<std::streamsize>(headerBytes.size()));
    out.write(rootBytes.data(), static_cast<std::streamsize>(rootBytes.size()));
    out.write(compressedMetadata.data(), static_cast<std::streamsize>(compressedMetadata.size()));
    if (!leavesBytes.empty()) {
        out.write(leavesBytes.data(), static_cast<std::streamsize>(leavesBytes.size()));
    }
    for (const auto& tile : tiles) {
        out.write(tile.compressedData.data(), static_cast<std::streamsize>(tile.compressedData.size()));
    }
    if (!out.good()) {
        error("%s: write failure while writing %s", __func__, outFile.c_str());
    }
    out.close();
}

void pm_core::write_pmtiles_archive_from_blob_file(const std::string& outFile,
    const std::string& blobFile,
    std::vector<StoredTilePayloadRef> tiles,
    const ArchiveOptions& options) {
    std::sort(tiles.begin(), tiles.end(), [](const StoredTilePayloadRef& lhs, const StoredTilePayloadRef& rhs) {
        return lhs.tileId < rhs.tileId;
    });

    std::vector<pmtiles::entryv3> entries;
    entries.reserve(tiles.size());
    uint64_t currentTileOffset = 0;
    for (const auto& tile : tiles) {
        entries.emplace_back(tile.tileId, currentTileOffset, tile.dataLength, 1);
        currentTileOffset += tile.dataLength;
    }

    const auto compressFn = [](const std::string& data, uint8_t /*compression*/) {
        return gzip_compress(data);
    };
    std::string rootBytes;
    std::string leavesBytes;
    int numLeaves = 0;
    std::tie(rootBytes, leavesBytes, numLeaves) =
        pmtiles::make_root_leaves(compressFn, pmtiles::COMPRESSION_GZIP, entries);
    (void)numLeaves;

    const std::string metadataJson = options.metadata.dump();
    const std::string compressedMetadata = gzip_compress(metadataJson);

    pmtiles::headerv3 header{};
    header.root_dir_offset = 127;
    header.root_dir_bytes = rootBytes.size();
    header.json_metadata_offset = header.root_dir_offset + header.root_dir_bytes;
    header.json_metadata_bytes = compressedMetadata.size();
    header.leaf_dirs_offset = header.json_metadata_offset + header.json_metadata_bytes;
    header.leaf_dirs_bytes = leavesBytes.size();
    header.tile_data_offset = header.leaf_dirs_offset + header.leaf_dirs_bytes;
    header.tile_data_bytes = currentTileOffset;
    header.addressed_tiles_count = entries.size();
    header.tile_entries_count = entries.size();
    header.tile_contents_count = entries.size();
    header.clustered = options.clustered;
    header.internal_compression = pmtiles::COMPRESSION_GZIP;
    header.tile_compression = options.tileCompression;
    header.tile_type = options.tileType;
    header.min_zoom = options.minZoom;
    header.max_zoom = options.maxZoom;
    header.center_zoom = options.centerZoom;
    if (options.hasGeographicBounds) {
        header.min_lon_e7 = options.minLonE7;
        header.min_lat_e7 = options.minLatE7;
        header.max_lon_e7 = options.maxLonE7;
        header.max_lat_e7 = options.maxLatE7;
        header.center_lon_e7 = options.centerLonE7;
        header.center_lat_e7 = options.centerLatE7;
    } else {
        header.min_lon_e7 = -1800000000;
        header.min_lat_e7 = -850000000;
        header.max_lon_e7 = 1800000000;
        header.max_lat_e7 = 850000000;
        header.center_lon_e7 = 0;
        header.center_lat_e7 = 0;
    }

    std::ifstream blob(blobFile, std::ios::binary);
    if (!blob.is_open()) {
        error("%s: cannot open blob input file %s", __func__, blobFile.c_str());
    }
    std::ofstream out(outFile, std::ios::binary | std::ios::trunc);
    if (!out.is_open()) {
        error("%s: cannot open output PMTiles file %s", __func__, outFile.c_str());
    }

    const std::string headerBytes = header.serialize();
    out.write(headerBytes.data(), static_cast<std::streamsize>(headerBytes.size()));
    out.write(rootBytes.data(), static_cast<std::streamsize>(rootBytes.size()));
    out.write(compressedMetadata.data(), static_cast<std::streamsize>(compressedMetadata.size()));
    if (!leavesBytes.empty()) {
        out.write(leavesBytes.data(), static_cast<std::streamsize>(leavesBytes.size()));
    }

    std::array<char, 1 << 20> buffer{};
    for (const auto& tile : tiles) {
        blob.clear();
        blob.seekg(static_cast<std::streamoff>(tile.dataOffset), std::ios::beg);
        if (!blob.good()) {
            error("%s: failed to seek blob input file %s to offset %" PRIu64,
                __func__, blobFile.c_str(), tile.dataOffset);
        }
        uint64_t remaining = tile.dataLength;
        while (remaining > 0) {
            const std::streamsize chunk = static_cast<std::streamsize>(
                std::min<uint64_t>(remaining, buffer.size()));
            blob.read(buffer.data(), chunk);
            if (blob.gcount() != chunk) {
                error("%s: failed to read %" PRIu64 " bytes from blob input file %s at offset %" PRIu64,
                    __func__, static_cast<uint64_t>(chunk), blobFile.c_str(),
                    tile.dataOffset + static_cast<uint64_t>(tile.dataLength) - remaining);
            }
            out.write(buffer.data(), chunk);
            remaining -= static_cast<uint64_t>(chunk);
        }
    }

    if (!out.good()) {
        error("%s: write failure while writing %s", __func__, outFile.c_str());
    }
    out.close();
}

nlohmann::json pm_vector::build_schema_fields_json(const FeatureTableSchema& schema) {
    nlohmann::json fields = nlohmann::json::object();
    for (const auto& col : schema.columns) {
        switch (col.type) {
        case ScalarType::STRING:
            fields[col.name] = "String";
            break;
        case ScalarType::BOOLEAN:
            fields[col.name] = "Boolean";
            break;
        default:
            fields[col.name] = "Number";
            break;
        }
    }
    return fields;
}

nlohmann::json pm_vector::build_exact_schema_json(const FeatureTableSchema& schema,
    VectorGeometryType geometryType) {
    auto scalar_type_name = [](ScalarType type) {
        switch (type) {
        case ScalarType::STRING:
            return "string";
        case ScalarType::BOOLEAN:
            return "boolean";
        case ScalarType::INT_32:
            return "int32";
        case ScalarType::FLOAT:
            return "float";
        default:
            error("%s: unsupported scalar column type %d", __func__, static_cast<int>(type));
            return "";
        }
    };

    nlohmann::json columns = nlohmann::json::array();
    for (const auto& col : schema.columns) {
        columns.push_back({
            {"name", col.name},
            {"type", scalar_type_name(col.type)},
            {"nullable", col.nullable},
        });
    }

    return {
        {"version", 1},
        {"layer", schema.layerName},
        {"geometry", geometry_type_name(geometryType)},
        {"extent", schema.extent},
        {"has_id_column", schema.hasIdColumn},
        {"id_is_uint64", schema.idIsUint64},
        {"columns", columns},
    };
}

bool pm_vector::parse_exact_schema_json(const nlohmann::json& metadata,
    const std::string& layerName,
    FeatureTableSchema& schema,
    VectorGeometryType* geometryType) {
    auto parse_scalar_type = [](const std::string& name, ScalarType& out) {
        if (name == "string") {
            out = ScalarType::STRING;
            return true;
        }
        if (name == "boolean") {
            out = ScalarType::BOOLEAN;
            return true;
        }
        if (name == "int32") {
            out = ScalarType::INT_32;
            return true;
        }
        if (name == "float") {
            out = ScalarType::FLOAT;
            return true;
        }
        return false;
    };
    auto parse_geometry_type = [](const std::string& name, VectorGeometryType& out) {
        if (name == "Point") {
            out = VectorGeometryType::Point;
            return true;
        }
        if (name == "Polygon") {
            out = VectorGeometryType::Polygon;
            return true;
        }
        return false;
    };

    if (!metadata.is_object() || !metadata.contains(PUNKST_VECTOR_SCHEMA_METADATA_KEY)) {
        return false;
    }
    const nlohmann::json& root = metadata[PUNKST_VECTOR_SCHEMA_METADATA_KEY];
    const nlohmann::json* item = nullptr;
    if (root.is_object()) {
        item = &root;
    } else if (root.is_array()) {
        for (const auto& candidate : root) {
            if (!candidate.is_object()) {
                continue;
            }
            if (layerName.empty() || candidate.value("layer", std::string()) == layerName) {
                item = &candidate;
                break;
            }
        }
    }
    if (item == nullptr || !item->is_object()) {
        return false;
    }
    if (!layerName.empty() && item->value("layer", std::string()) != layerName) {
        return false;
    }
    if (!item->contains("columns") || !(*item)["columns"].is_array()) {
        return false;
    }

    FeatureTableSchema parsed;
    parsed.layerName = item->value("layer", layerName);
    parsed.extent = item->value("extent", 4096u);
    parsed.hasIdColumn = item->value("has_id_column", false);
    parsed.idIsUint64 = item->value("id_is_uint64", false);
    parsed.columns.reserve((*item)["columns"].size());
    for (const auto& colJson : (*item)["columns"]) {
        if (!colJson.is_object() || !colJson.contains("name") || !colJson["name"].is_string() ||
            !colJson.contains("type") || !colJson["type"].is_string()) {
            return false;
        }
        ScalarType type = ScalarType::STRING;
        if (!parse_scalar_type(colJson["type"].get<std::string>(), type)) {
            return false;
        }
        parsed.columns.push_back({
            colJson["name"].get<std::string>(),
            type,
            colJson.value("nullable", false),
        });
    }

    if (geometryType != nullptr && item->contains("geometry") && (*item)["geometry"].is_string()) {
        VectorGeometryType parsedGeometry = VectorGeometryType::Point;
        if (parse_geometry_type((*item)["geometry"].get<std::string>(), parsedGeometry)) {
            *geometryType = parsedGeometry;
        }
    }
    schema = std::move(parsed);
    return true;
}

void pm_vector::write_single_layer_vector_pmtiles_archive(const std::string& outFile,
    std::vector<pm_core::EncodedTilePayload> encodedTiles,
    const SingleLayerVectorPmtilesOptions& options) {
    nlohmann::json metadata;
    metadata["name"] = options.schema.layerName;
    metadata["type"] = "overlay";
    metadata["version"] = "2";
    metadata["format"] = "pbf";
    const bool isMvt = options.tileType == pmtiles::TILETYPE_MVT;
    metadata["description"] = options.description.empty()
        ? ("Generated PMTiles by punkst for " + std::string(isMvt ? "MVT " : "MLT ") +
            std::string(to_lower(geometry_type_name(options.geometryType))))
        : options.description;
    metadata["generator"] = options.generator;
    if (options.coordScale > 0) {
        metadata["coord_scale"] = options.coordScale;
    }
    metadata["feature_dictionary_size"] = options.featureDictionarySize;
    metadata["coordinate_mode"] = "epsg3857";
    metadata["zoom"] = options.outputZoom;

    const nlohmann::json fields = build_schema_fields_json(options.schema);
    nlohmann::json vectorLayer;
    vectorLayer["id"] = options.schema.layerName;
    vectorLayer["fields"] = fields;
    vectorLayer["minzoom"] = options.outputZoom;
    vectorLayer["maxzoom"] = options.outputZoom;
    metadata["vector_layers"] = nlohmann::json::array({vectorLayer});
    metadata[PUNKST_VECTOR_SCHEMA_METADATA_KEY] =
        build_exact_schema_json(options.schema, options.geometryType);

    nlohmann::json tilestatsLayer;
    tilestatsLayer["layer"] = options.schema.layerName;
    tilestatsLayer["count"] = options.totalRecordCount;
    tilestatsLayer["geometry"] = geometry_type_name(options.geometryType);
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

    if (options.extraMetadata.is_object()) {
        for (auto it = options.extraMetadata.begin(); it != options.extraMetadata.end(); ++it) {
            metadata[it.key()] = it.value();
        }
    }

    pm_core::ArchiveOptions archiveOptions;
    archiveOptions.tileType = options.tileType;
    archiveOptions.minZoom = options.outputZoom;
    archiveOptions.maxZoom = options.outputZoom;
    archiveOptions.centerZoom = options.outputZoom;
    archiveOptions.clustered = true;
    archiveOptions.metadata = metadata;

    if (std::isfinite(options.geoMinX) && std::isfinite(options.geoMinY) &&
        std::isfinite(options.geoMaxX) && std::isfinite(options.geoMaxY)) {
        double minLon = 0.0;
        double minLat = 0.0;
        double maxLon = 0.0;
        double maxLat = 0.0;
        pm_core::epsg3857_to_wgs84(options.geoMinX, options.geoMinY, minLon, minLat);
        pm_core::epsg3857_to_wgs84(options.geoMaxX, options.geoMaxY, maxLon, maxLat);
        archiveOptions.hasGeographicBounds = true;
        archiveOptions.minLonE7 = static_cast<int32_t>(minLon * 10000000.0);
        archiveOptions.minLatE7 = static_cast<int32_t>(minLat * 10000000.0);
        archiveOptions.maxLonE7 = static_cast<int32_t>(maxLon * 10000000.0);
        archiveOptions.maxLatE7 = static_cast<int32_t>(maxLat * 10000000.0);
        if (!encodedTiles.empty()) {
            const auto centerIt = std::max_element(encodedTiles.begin(), encodedTiles.end(),
                [](const pm_core::EncodedTilePayload& lhs, const pm_core::EncodedTilePayload& rhs) {
                    return lhs.featureCount < rhs.featureCount;
                });
            double centerX = 0.0;
            double centerY = 0.0;
            pm_core::tilecoord_to_epsg3857(centerIt->x, centerIt->y, 128.0, 128.0,
                centerIt->z, centerX, centerY);
            double centerLon = 0.0;
            double centerLat = 0.0;
            pm_core::epsg3857_to_wgs84(centerX, centerY, centerLon, centerLat);
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
    pm_core::write_pmtiles_archive(outFile, std::move(encodedTiles), archiveOptions);
}

void pm_raster::write_png_raster_pmtiles_archive(const std::string& pngFile,
    const std::string& outFile,
    const std::string& tempBlobFile,
    const RasterBounds& bounds,
    int32_t minZoom,
    int32_t maxZoom) {
    validate_raster_archive_options(bounds, minZoom, maxZoom, __func__);

    Image2D<Rgb8> src = load_png_rgb8(pngFile);
    std::filesystem::path blobPath(tempBlobFile);
    if (blobPath.has_parent_path()) {
        std::filesystem::create_directories(blobPath.parent_path());
    }
    std::ofstream blob(blobPath, std::ios::binary | std::ios::trunc);
    if (!blob.is_open()) {
        error("%s: cannot open temporary PMTiles blob %s", __func__, tempBlobFile.c_str());
    }

    std::vector<pm_core::StoredTilePayloadRef> tiles;
    uint64_t dataOffset = 0;
    for (int32_t z = minZoom; z <= maxZoom; ++z) {
        int64_t txA = 0, tyA = 0, txB = 0, tyB = 0;
        double lx = 0.0, ly = 0.0;
        pm_core::epsg3857_to_tilecoord(bounds.xmin, bounds.ymax, static_cast<uint8_t>(z), txA, tyA, lx, ly);
        pm_core::epsg3857_to_tilecoord(bounds.xmax, bounds.ymin, static_cast<uint8_t>(z), txB, tyB, lx, ly);
        uint32_t tx0 = clamp_tile_coord(std::min(txA, txB), z);
        uint32_t tx1 = clamp_tile_coord(std::max(txA, txB), z);
        uint32_t ty0 = clamp_tile_coord(std::min(tyA, tyB), z);
        uint32_t ty1 = clamp_tile_coord(std::max(tyA, tyB), z);

        for (uint32_t ty = ty0; ty <= ty1; ++ty) {
            for (uint32_t tx = tx0; tx <= tx1; ++tx) {
                Image2D<Rgb8> tile(kRasterTileSize, kRasterTileSize, Rgb8{0, 0, 0});
                bool nonEmpty = false;
                for (int32_t py = 0; py < kRasterTileSize; ++py) {
                    for (int32_t px = 0; px < kRasterTileSize; ++px) {
                        double mx = 0.0;
                        double my = 0.0;
                        pm_core::tilecoord_to_epsg3857(tx, ty, px + 0.5, py + 0.5,
                            static_cast<uint8_t>(z), mx, my);
                        if (mx < bounds.xmin || mx > bounds.xmax || my < bounds.ymin || my > bounds.ymax) {
                            continue;
                        }
                        tile(py, px) = sample_nearest(src, mx, my, bounds);
                        nonEmpty = true;
                    }
                }
                if (!nonEmpty) {
                    continue;
                }
                const std::string encoded = encode_png_rgb8(tile);
                append_png_tile_to_blob(blob, tempBlobFile, encoded, dataOffset,
                    RasterTileKey{static_cast<uint8_t>(z), tx, ty}, tiles);
            }
        }
    }
    blob.close();
    write_png_raster_pmtiles_archive_from_blob(
        outFile, tempBlobFile, std::move(tiles), bounds, minZoom, maxZoom);
    std::error_code ec;
    std::filesystem::remove(blobPath, ec);
}
