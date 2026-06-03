#pragma once

#include "flex_io.hpp"
#include "json.hpp"
#include "mlt_utils.hpp"
#include "PMTiles/pmtiles.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <vector>

namespace mlt_pmtiles {

struct EncodedTilePayload {
    uint64_t tileId = 0;
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t featureCount = 0;
    std::string compressedData;
};

struct StoredTilePayloadRef {
    uint64_t tileId = 0;
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t featureCount = 0;
    uint64_t dataOffset = 0;
    uint32_t dataLength = 0;
    uint64_t priorityOffset = 0;
    uint32_t priorityCount = 0;
    bool prioritiesSorted = false;
};

struct ArchiveOptions {
    uint8_t tileType = 0x06;
    uint8_t tileCompression = pmtiles::COMPRESSION_GZIP;
    uint8_t minZoom = 0;
    uint8_t maxZoom = 0;
    uint8_t centerZoom = 0;
    bool clustered = true;
    bool hasGeographicBounds = false;
    int32_t minLonE7 = 0;
    int32_t minLatE7 = 0;
    int32_t maxLonE7 = 0;
    int32_t maxLatE7 = 0;
    int32_t centerLonE7 = 0;
    int32_t centerLatE7 = 0;
    nlohmann::json metadata;
};

struct RasterBounds {
    double xmin = std::numeric_limits<double>::quiet_NaN();
    double xmax = std::numeric_limits<double>::quiet_NaN();
    double ymin = std::numeric_limits<double>::quiet_NaN();
    double ymax = std::numeric_limits<double>::quiet_NaN();

    bool valid() const {
        return xmax > xmin && ymax > ymin;
    }
};

struct RasterTileKey {
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;

    bool operator==(const RasterTileKey& other) const {
        return z == other.z && x == other.x && y == other.y;
    }

    bool operator<(const RasterTileKey& other) const {
        return z < other.z ||
            (z == other.z && (x < other.x || (x == other.x && y < other.y)));
    }
};

struct RasterTileKeyHash {
    size_t operator()(const RasterTileKey& key) const;
};

struct RasterPixelCoord {
    RasterTileKey key;
    int32_t px = 0;
    int32_t py = 0;
};

enum class VectorGeometryType {
    Point,
    Polygon,
};

struct SingleLayerVectorPmtilesOptions {
    FeatureTableSchema schema;
    VectorGeometryType geometryType = VectorGeometryType::Point;
    uint8_t tileType = 0x06;
    uint64_t totalRecordCount = 0;
    double coordScale = -1.0;
    size_t featureDictionarySize = 0;
    uint8_t outputZoom = 0;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
    std::string generator;
    std::string description;
    nlohmann::json extraMetadata;
};

struct LoadedPmtilesArchive {
    std::unique_ptr<flexio::FlexReader> reader;
    pmtiles::headerv3 header{};
    nlohmann::json metadata;
    std::vector<pmtiles::entry_zxy> entries;
};

LoadedPmtilesArchive load_pmtiles_archive(const std::string& inFile);
std::string read_pmtiles_tile_payload(flexio::FlexReader& reader,
    const pmtiles::headerv3& header,
    const pmtiles::entry_zxy& entry);

void write_pmtiles_archive(const std::string& outFile,
    std::vector<EncodedTilePayload> tiles,
    const ArchiveOptions& options);

void write_pmtiles_archive_from_blob_file(const std::string& outFile,
    const std::string& blobFile,
    std::vector<StoredTilePayloadRef> tiles,
    const ArchiveOptions& options);

nlohmann::json build_schema_fields_json(const FeatureTableSchema& schema);

void write_single_layer_vector_pmtiles_archive(const std::string& outFile,
    std::vector<EncodedTilePayload> encodedTiles,
    const SingleLayerVectorPmtilesOptions& options);

void write_png_raster_pmtiles_archive(const std::string& pngFile,
    const std::string& outFile,
    const std::string& tempBlobFile,
    const RasterBounds& bounds,
    int32_t minZoom,
    int32_t maxZoom);

void validate_raster_archive_options(const RasterBounds& bounds,
    int32_t minZoom,
    int32_t maxZoom,
    const char* context);

RasterPixelCoord epsg3857_to_raster_pixel(double x, double y, int32_t zoom);

void append_png_tile_to_blob(std::ofstream& blob,
    const std::string& tempBlobFile,
    const std::string& encoded,
    uint64_t& dataOffset,
    const RasterTileKey& key,
    std::vector<StoredTilePayloadRef>& tiles);

void write_png_raster_pmtiles_archive_from_blob(const std::string& outFile,
    const std::string& tempBlobFile,
    std::vector<StoredTilePayloadRef> tiles,
    const RasterBounds& bounds,
    int32_t minZoom,
    int32_t maxZoom);

} // namespace mlt_pmtiles
