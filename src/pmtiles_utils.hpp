#pragma once

#include "flex_io.hpp"
#include "json.hpp"
#include "vector_tile_utils.hpp"
#include "PMTiles/pmtiles.hpp"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace pm_core {

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

struct LoadedPmtilesArchive {
    std::unique_ptr<flexio::FlexReader> reader;
    pmtiles::headerv3 header{};
    nlohmann::json metadata;
    std::vector<pmtiles::entry_zxy> entries;
};

std::string gzip_compress(const std::string& data);
std::string gzip_decompress(const std::string& data);

void epsg3857_to_wgs84(double x, double y, double& lon, double& lat);
double epsg3857_scale_factor(uint8_t zoom);
void epsg3857_to_tilecoord(double x, double y, uint8_t zoom,
    int64_t& tileX, int64_t& tileY, double& localX, double& localY);
void tilecoord_to_epsg3857(int64_t tileX, int64_t tileY,
    double localX, double localY, uint8_t zoom,
    double& x, double& y);

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

} // namespace pm_core

namespace pm_vector {

enum class VectorGeometryType {
    Point,
    Polygon,
};

inline constexpr const char* PUNKST_VECTOR_SCHEMA_METADATA_KEY = "punkst_vector_schema";

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

nlohmann::json build_schema_fields_json(const FeatureTableSchema& schema);
nlohmann::json build_exact_schema_json(const FeatureTableSchema& schema,
    VectorGeometryType geometryType);
bool parse_exact_schema_json(const nlohmann::json& metadata,
    const std::string& layerName,
    FeatureTableSchema& schema,
    VectorGeometryType* geometryType = nullptr);

void write_single_layer_vector_pmtiles_archive(const std::string& outFile,
    std::vector<pm_core::EncodedTilePayload> encodedTiles,
    const SingleLayerVectorPmtilesOptions& options);

} // namespace pm_vector

namespace pm_raster {

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

using EncodedRasterTileMap = std::unordered_map<RasterTileKey, std::string, RasterTileKeyHash>;

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
    std::vector<pm_core::StoredTilePayloadRef>& tiles);

void write_png_raster_pmtiles_archive_from_blob(const std::string& outFile,
    const std::string& tempBlobFile,
    std::vector<pm_core::StoredTilePayloadRef> tiles,
    const RasterBounds& bounds,
    int32_t minZoom,
    int32_t maxZoom);

EncodedRasterTileMap write_rgba_png_parent_zoom(uint8_t parentZoom,
    const EncodedRasterTileMap& children,
    std::ofstream& blob,
    const std::string& tempBlobFile,
    uint64_t& dataOffset,
    std::vector<pm_core::StoredTilePayloadRef>& tiles);

} // namespace pm_raster
