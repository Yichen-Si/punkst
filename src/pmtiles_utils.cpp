#include "pmtiles_utils.hpp"

#include "image_utils.hpp"
#include "mlt_utils.hpp"
#include "PMTiles/pmtiles.hpp"
#include "utils.h"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <tuple>

namespace mlt_pmtiles {

namespace {

constexpr int kRasterTileSize = 256;

std::string maybe_decompress_pmtiles_blob(const std::string& data, uint8_t compression) {
    switch (compression) {
    case pmtiles::COMPRESSION_NONE:
        return data;
    case pmtiles::COMPRESSION_GZIP:
        return gzip_decompress(data);
    default:
        error("%s: unsupported PMTiles compression mode %u", __func__, static_cast<unsigned>(compression));
        return std::string();
    }
}

const char* geometry_type_name(VectorGeometryType geometryType) {
    switch (geometryType) {
    case VectorGeometryType::Point:
        return "Point";
    case VectorGeometryType::Polygon:
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

Rgb8 sample_nearest(const Image2D<Rgb8>& src, double x, double y, const RasterBounds& bounds) {
    const double sx = (x - bounds.xmin) / (bounds.xmax - bounds.xmin) * static_cast<double>(src.width() - 1);
    const double sy = (y - bounds.ymin) / (bounds.ymax - bounds.ymin) * static_cast<double>(src.height() - 1);
    int32_t ix = static_cast<int32_t>(std::llround(sx));
    int32_t iy = static_cast<int32_t>(std::llround(sy));
    ix = std::max(0, std::min(src.width() - 1, ix));
    iy = std::max(0, std::min(src.height() - 1, iy));
    return src(iy, ix);
}

} // namespace

LoadedPmtilesArchive load_pmtiles_archive(const std::string& inFile) {
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

std::string read_pmtiles_tile_payload(flexio::FlexReader& reader,
    const pmtiles::headerv3& header,
    const pmtiles::entry_zxy& entry) {
    std::string compressed(static_cast<size_t>(entry.length), '\0');
    if (!reader.read_at(entry.offset, compressed.size(), compressed)) {
        error("%s: failed to read PMTiles tile z=%u x=%u y=%u", __func__,
            static_cast<unsigned>(entry.z), entry.x, entry.y);
    }
    return maybe_decompress_pmtiles_blob(compressed, header.tile_compression);
}

void write_pmtiles_archive(const std::string& outFile,
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

void write_pmtiles_archive_from_blob_file(const std::string& outFile,
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

nlohmann::json build_schema_fields_json(const FeatureTableSchema& schema) {
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

void write_single_layer_vector_pmtiles_archive(const std::string& outFile,
    std::vector<EncodedTilePayload> encodedTiles,
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

    ArchiveOptions archiveOptions;
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
        epsg3857_to_wgs84(options.geoMinX, options.geoMinY, minLon, minLat);
        epsg3857_to_wgs84(options.geoMaxX, options.geoMaxY, maxLon, maxLat);
        archiveOptions.hasGeographicBounds = true;
        archiveOptions.minLonE7 = static_cast<int32_t>(minLon * 10000000.0);
        archiveOptions.minLatE7 = static_cast<int32_t>(minLat * 10000000.0);
        archiveOptions.maxLonE7 = static_cast<int32_t>(maxLon * 10000000.0);
        archiveOptions.maxLatE7 = static_cast<int32_t>(maxLat * 10000000.0);
        if (!encodedTiles.empty()) {
            const auto centerIt = std::max_element(encodedTiles.begin(), encodedTiles.end(),
                [](const EncodedTilePayload& lhs, const EncodedTilePayload& rhs) {
                    return lhs.featureCount < rhs.featureCount;
                });
            double centerX = 0.0;
            double centerY = 0.0;
            tilecoord_to_epsg3857(centerIt->x, centerIt->y, 128.0, 128.0,
                centerIt->z, centerX, centerY);
            double centerLon = 0.0;
            double centerLat = 0.0;
            epsg3857_to_wgs84(centerX, centerY, centerLon, centerLat);
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
    write_pmtiles_archive(outFile, std::move(encodedTiles), archiveOptions);
}

void write_png_raster_pmtiles_archive(const std::string& pngFile,
    const std::string& outFile,
    const std::string& tempBlobFile,
    const RasterBounds& bounds,
    int32_t minZoom,
    int32_t maxZoom) {
    if (!bounds.valid()) {
        error("%s: invalid raster bounds for %s", __func__, pngFile.c_str());
    }
    if (minZoom < 0 || maxZoom < minZoom || maxZoom > 30) {
        error("%s: invalid zoom range %d..%d", __func__, minZoom, maxZoom);
    }

    Image2D<Rgb8> src = load_png_rgb8(pngFile);
    std::filesystem::path blobPath(tempBlobFile);
    if (blobPath.has_parent_path()) {
        std::filesystem::create_directories(blobPath.parent_path());
    }
    std::ofstream blob(blobPath, std::ios::binary | std::ios::trunc);
    if (!blob.is_open()) {
        error("%s: cannot open temporary PMTiles blob %s", __func__, tempBlobFile.c_str());
    }

    std::vector<StoredTilePayloadRef> tiles;
    uint64_t dataOffset = 0;
    for (int32_t z = minZoom; z <= maxZoom; ++z) {
        int64_t txA = 0, tyA = 0, txB = 0, tyB = 0;
        double lx = 0.0, ly = 0.0;
        epsg3857_to_tilecoord(bounds.xmin, bounds.ymax, static_cast<uint8_t>(z), txA, tyA, lx, ly);
        epsg3857_to_tilecoord(bounds.xmax, bounds.ymin, static_cast<uint8_t>(z), txB, tyB, lx, ly);
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
                        tilecoord_to_epsg3857(tx, ty, px + 0.5, py + 0.5,
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
                blob.write(encoded.data(), static_cast<std::streamsize>(encoded.size()));
                if (!blob.good()) {
                    error("%s: failed writing temporary PMTiles blob %s", __func__, tempBlobFile.c_str());
                }

                StoredTilePayloadRef ref;
                ref.z = static_cast<uint8_t>(z);
                ref.x = tx;
                ref.y = ty;
                ref.featureCount = 1;
                ref.tileId = pmtiles::zxy_to_tileid(ref.z, ref.x, ref.y);
                ref.dataOffset = dataOffset;
                ref.dataLength = static_cast<uint32_t>(encoded.size());
                dataOffset += encoded.size();
                tiles.push_back(ref);
            }
        }
    }
    blob.close();
    if (tiles.empty()) {
        error("%s: no raster tiles generated for %s", __func__, pngFile.c_str());
    }

    double minLon = 0.0, minLat = 0.0, maxLon = 0.0, maxLat = 0.0;
    epsg3857_to_wgs84(bounds.xmin, bounds.ymin, minLon, minLat);
    epsg3857_to_wgs84(bounds.xmax, bounds.ymax, maxLon, maxLat);
    nlohmann::json metadata = {
        {"version", "1.1"},
        {"name", std::filesystem::path(outFile).filename().string()},
        {"description", std::filesystem::path(outFile).filename().string()},
        {"format", "png"},
        {"type", "overlay"},
        {"minzoom", minZoom},
        {"maxzoom", maxZoom}
    };

    ArchiveOptions options;
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
    write_pmtiles_archive_from_blob_file(outFile, tempBlobFile, std::move(tiles), options);
    std::error_code ec;
    std::filesystem::remove(blobPath, ec);
}

} // namespace mlt_pmtiles
