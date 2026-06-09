#include "punkst.h"

#include "dataunits.hpp"
#include "factor_polygon_pmtiles.hpp"
#include "json.hpp"
#include "mlt_utils.hpp"
#include "mvt_utils.hpp"
#include "pmtiles_pyramid.hpp"
#include "pmtiles_utils.hpp"
#include "region_query.hpp"
#include "simple_polygon_pmtiles.hpp"
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
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct TopFactors {
    std::vector<int32_t> k;
    std::vector<float> p;
};

struct CellFactorRow {
    std::string cellId;
    TopFactors top;
};

struct CellCenter {
    double x = 0.0;
    double y = 0.0;
};

struct CellGeometry {
    std::string cellId;
    std::vector<std::vector<std::pair<double, double>>> rings;
    CellCenter center;
};

char infer_delim(const std::string& line) {
    if (line.find('\t') != std::string::npos) {
        return '\t';
    }
    if (line.find(',') != std::string::npos) {
        return ',';
    }
    return '\t';
}

void require_col_index(int32_t col, const char* optName) {
    if (col < 0) {
        error("%s: %s must be a non-negative 0-based column index", __func__, optName);
    }
}

void require_fields(const std::vector<std::string>& fields, int32_t maxCol,
    const char* context, uint64_t rowNo) {
    if (maxCol < 0 || fields.size() <= static_cast<size_t>(maxCol)) {
        error("%s: malformed row %" PRIu64 " with fewer than %d columns",
            context, rowNo, maxCol + 1);
    }
}

std::unordered_map<std::string, CellFactorRow> read_factor_rows(
    const std::string& path,
    const std::string& idColName,
    int32_t topK) {
    UnitFactorResultReadOptions opts;
    opts.unitIdColName = idColName;
    opts.xColName.clear();
    opts.yColName.clear();
    opts.topKColName = "topK";
    opts.topPColName = "topP";
    opts.denseTopK = topK;
    opts.requireFactorValues = false;
    opts.allowKpColumns = true;
    opts.autoDetectDelimiter = true;
    UnitFactorResultReader reader(path, opts);
    std::unordered_map<std::string, CellFactorRow> out;
    UnitFactorResultRow parsed;
    while (reader.next(parsed)) {
        CellFactorRow row;
        row.cellId = parsed.unitId;
        row.top.k.reserve(parsed.topFactors.size());
        row.top.p.reserve(parsed.topFactors.size());
        for (const UnitTopFactor& tf : parsed.topFactors) {
            row.top.k.push_back(tf.k);
            row.top.p.push_back(tf.p);
        }
        out[row.cellId] = std::move(row);
    }
    if (out.empty()) {
        error("%s: no factor rows found in %s", __func__, path.c_str());
    }
    return out;
}

std::vector<CellGeometry> read_cell_geometries_auto(const std::string& path,
    const std::string& format, const std::string& idProperty,
    int32_t idCol, int32_t xCol, int32_t yCol) {
    SimplePolygonTableReadOptions tableOpts;
    tableOpts.idCol = idCol;
    tableOpts.xCol = xCol;
    tableOpts.yCol = yCol;
    tableOpts.requireConsecutiveIds = true;
    const auto records = readSimplePolygonsAuto(path, format, idProperty, tableOpts);
    std::vector<CellGeometry> out;
    std::unordered_map<std::string, size_t> index;
    for (const auto& rec : records) {
        auto it = index.find(rec.polygonId);
        if (it == index.end()) {
            CellGeometry geom;
            geom.cellId = rec.polygonId;
            out.push_back(std::move(geom));
            it = index.emplace(rec.polygonId, out.size() - 1).first;
        }
        out[it->second].rings.push_back(rec.ring);
    }
    for (auto& geom : out) {
        const auto center = centroidForSimpleRings(geom.rings);
        geom.center = {center.first, center.second};
    }
    return out;
}

std::unordered_map<std::string, CellCenter> read_centers(const std::string& path, int32_t idCol, int32_t xCol, int32_t yCol) {
    std::unordered_map<std::string, CellCenter> out;
    if (path.empty()) {
        return out;
    }
    require_col_index(idCol, "--c-icol-id");
    require_col_index(xCol, "--c-icol-x");
    require_col_index(yCol, "--c-icol-y");
    TextLineReader reader(path);
    std::string line;
    uint64_t rowNo = 0;
    if (!read_next_data_line(reader, line, rowNo)) {
        error("%s: empty centers input %s", __func__, path.c_str());
    }
    const char delim = infer_delim(line);
    const int32_t maxCol = std::max({idCol, xCol, yCol});
    auto parse_center_row = [&](const std::string& rowLine, uint64_t row, bool allowHeader) {
        std::vector<std::string> fields;
        split(fields, std::string_view(&delim, 1), rowLine);
        require_fields(fields, maxCol, __func__, row);
        CellCenter c;
        if (!str2double(fields[static_cast<size_t>(xCol)], c.x) ||
            !str2double(fields[static_cast<size_t>(yCol)], c.y)) {
            if (allowHeader) {
                return false;
            }
            error("%s: failed parsing center x/y at row %" PRIu64, __func__, row);
        }
        out[fields[static_cast<size_t>(idCol)]] = c;
        return true;
    };
    parse_center_row(line, rowNo, true);
    while (reader.getline(line)) {
        ++rowNo;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        parse_center_row(line, rowNo, false);
    }
    if (out.empty()) {
        error("%s: no cell centers found in %s", __func__, path.c_str());
    }
    return out;
}

pm_vector::FeatureTableSchema build_cell_schema(const std::string& layerName,
    uint32_t extent, bool point, size_t nTop) {
    pm_vector::FeatureTableSchema schema;
    schema.layerName = layerName;
    schema.extent = extent;
    schema.hasIdColumn = true;
    schema.idIsUint64 = true;
    schema.columns.push_back({"cell_id", pm_vector::ScalarType::STRING, false});
    schema.columns.push_back({"topK", point ? pm_vector::ScalarType::INT_32 : pm_vector::ScalarType::STRING, false});
    schema.columns.push_back({"topP", pm_vector::ScalarType::FLOAT, false});
    for (size_t i = 2; i <= nTop; ++i) {
        schema.columns.push_back({"K" + std::to_string(i), point ? pm_vector::ScalarType::INT_32 : pm_vector::ScalarType::STRING, true});
        schema.columns.push_back({"P" + std::to_string(i), pm_vector::ScalarType::FLOAT, true});
    }
    return schema;
}

simple_polygon_pmtiles::PolygonFeatureProperties make_polygon_props(
    const std::string& cellId, const TopFactors& top, size_t nTop, double probThreshold) {
    simple_polygon_pmtiles::PolygonFeatureProperties props;
    const int32_t topK = top.k.empty() ? -1 : top.k[0];
    const float topP = top.p.empty() ? 0.0f : top.p[0];
    props.stringValues = {
        cellId,
        std::to_string(topK),
    };
    props.floatValues = {topP};
    for (size_t i = 1; i < nTop; ++i) {
        const bool present = i < top.k.size() && top.k[i] >= 0 &&
            i < top.p.size() && top.p[i] >= static_cast<float>(probThreshold);
        props.stringValues.push_back(present ? std::optional<std::string>(std::to_string(top.k[i])) : std::nullopt);
        props.floatValues.push_back(present ? std::optional<float>(top.p[i]) : std::nullopt);
    }
    return props;
}

uint64_t stable_id(size_t cellIdx, size_t partIdx) {
    return (static_cast<uint64_t>(cellIdx) << 20u) | static_cast<uint64_t>(partIdx);
}

void append_point_column_values(pm_vector::PointTileData& tile,
    const std::string& cellId, const TopFactors& top, size_t nTop, double probThreshold) {
    const int32_t topK = top.k.empty() ? -1 : top.k[0];
    const float topP = top.p.empty() ? 0.0f : top.p[0];
    tile.columns[0].stringValues.push_back(cellId);
    tile.columns[1].intValues.push_back(topK);
    tile.columns[2].floatValues.push_back(topP);
    for (size_t i = 1; i < nTop; ++i) {
        const bool present = i < top.k.size() && top.k[i] >= 0 &&
            i < top.p.size() && top.p[i] >= static_cast<float>(probThreshold);
        auto& kCol = tile.columns[1 + i * 2];
        auto& pCol = tile.columns[2 + i * 2];
        kCol.present.push_back(present);
        pCol.present.push_back(present);
        kCol.intValues.push_back(present ? top.k[i] : -1);
        pCol.floatValues.push_back(present ? top.p[i] : 0.0f);
    }
}

std::vector<pm_core::EncodedTilePayload> encode_point_tiles(
    std::map<TileKey, pm_vector::PointTileData>& tileMap,
    const pm_vector::FeatureTableSchema& schema, uint8_t zoom, bool useMvt) {
    std::vector<pm_core::EncodedTilePayload> out;
    out.reserve(tileMap.size());
    for (auto& kv : tileMap) {
        const std::string raw = useMvt
            ? mvt_pmtiles::encode_point_tile(schema, kv.second, nullptr)
            : mlt_pmtiles::encode_point_tile(schema, kv.second, nullptr);
        pm_core::EncodedTilePayload payload;
        payload.z = zoom;
        payload.x = static_cast<uint32_t>(kv.first.col);
        payload.y = static_cast<uint32_t>(kv.first.row);
        payload.tileId = pmtiles::zxy_to_tileid(payload.z, payload.x, payload.y);
        payload.featureCount = static_cast<uint32_t>(kv.second.size());
        payload.compressedData = pm_core::gzip_compress(raw);
        out.push_back(std::move(payload));
    }
    return out;
}

void append_cell_point(std::map<TileKey, pm_vector::PointTileData>& tileMap,
    const pm_vector::FeatureTableSchema& schema,
    const CellCenter& center, uint64_t featureId, const std::string& cellId,
    const TopFactors& top, size_t nTop, double probThreshold,
    double coordScale, uint8_t zoom,
    double& minX, double& minY, double& maxX, double& maxY) {
    const double x = center.x * coordScale;
    const double y = center.y * coordScale;
    minX = std::min(minX, x);
    minY = std::min(minY, y);
    maxX = std::max(maxX, x);
    maxY = std::max(maxY, y);
    int64_t tx = 0, ty = 0;
    double lx = 0.0, ly = 0.0;
    pm_core::epsg3857_to_tilecoord(x, y, zoom, tx, ty, lx, ly);
    TileKey key{static_cast<int32_t>(ty), static_cast<int32_t>(tx)};
    auto it = tileMap.find(key);
    if (it == tileMap.end()) {
        pm_vector::PointTileData empty;
        for (const auto& col : schema.columns) {
            empty.columns.emplace_back(col.type, col.nullable);
        }
        it = tileMap.emplace(key, std::move(empty)).first;
    }
    pm_vector::PointTileData& tile = it->second;
    tile.featureIds.push_back(featureId);
    int32_t localX = static_cast<int32_t>(std::llround(lx * static_cast<double>(schema.extent) / 256.0));
    int32_t localY = static_cast<int32_t>(std::llround(ly * static_cast<double>(schema.extent) / 256.0));
    tile.localX.push_back(std::clamp(localX, 0, static_cast<int32_t>(schema.extent) - 1));
    tile.localY.push_back(std::clamp(localY, 0, static_cast<int32_t>(schema.extent) - 1));
    append_point_column_values(tile, cellId, top, nTop, probThreshold);
}

void write_pmtiles_archive(const fs::path& outFile,
    const pm_vector::FeatureTableSchema& schema,
    pm_vector::VectorGeometryType geomType,
    std::vector<pm_core::EncodedTilePayload> tiles,
    uint64_t totalRecords, double coordScale, uint8_t zoom,
    double minX, double minY, double maxX, double maxY, bool useMvt,
    const std::string& generator, const json& extra = json::object()) {
    pm_vector::SingleLayerVectorPmtilesOptions options;
    options.schema = schema;
    options.geometryType = geomType;
    options.tileType = useMvt ? pmtiles::TILETYPE_MVT : pmtiles::TILETYPE_MLT;
    options.totalRecordCount = totalRecords;
    options.coordScale = coordScale;
    options.outputZoom = zoom;
    options.geoMinX = minX;
    options.geoMinY = minY;
    options.geoMaxX = maxX;
    options.geoMaxY = maxY;
    options.generator = generator;
    options.description = useMvt
        ? "Generated PMTiles by punkst for MVT cell assets"
        : "Generated PMTiles by punkst for MLT cell assets";
    options.extraMetadata = extra;
    pm_vector::write_single_layer_vector_pmtiles_archive(outFile.string(), std::move(tiles), options);
}

void build_pyramid(const fs::path& pmtilesPath, const fs::path& tmpDir,
    bool point, int32_t minZoom, int32_t maxTileBytes, int32_t maxTileFeatures,
    double compressionScale, int32_t threads) {
    fs::path tmpOut = tmpDir / (pmtilesPath.filename().string() + ".pyramid.pmtiles");
    pmtiles_pyramid::BuildOptions options;
    options.minZoom = minZoom;
    options.maxTileBytes = maxTileBytes;
    options.maxTileFeatures = maxTileFeatures;
    options.scaleFactorCompression = compressionScale;
    options.threads = threads;
    if (point) {
        pmtiles_pyramid::build_point_pmtiles_pyramid(pmtilesPath.string(), tmpOut.string(), options);
    } else {
        pmtiles_pyramid::build_polygon_pmtiles_pyramid(pmtilesPath.string(), tmpOut.string(), options);
    }
    if (fs::exists(tmpOut)) {
        fs::rename(tmpOut, pmtilesPath);
    }
}

} // namespace

int32_t cmdCells2Pmtiles(int32_t argc, char** argv) {
    std::string inResults;
    std::string inBoundaries;
    std::string inCenters;
    std::string outPrefix;
    std::string format = "MVT";
    std::string boundaryFormat = "auto";
    std::string boundaryIdProp = "cell_id";
    std::string resultIdCol = "cell_id";
    int32_t minZoom = 10;
    int32_t maxZoom = 18;
    int32_t maxPointTileBytes = 500000;
    int32_t maxPointTileFeatures = 50000;
    int32_t maxPolygonTileBytes = 500000;
    int32_t maxPolygonTileFeatures = 5000;
    int32_t topK = 3;
    int32_t bIcolId = 0;
    int32_t bIcolX = 1;
    int32_t bIcolY = 2;
    int32_t cIcolId = 0;
    int32_t cIcolX = 1;
    int32_t cIcolY = 2;
    int32_t extent = 4096;
    int32_t threads = 1;
    double coordScale = 1.0;
    double probThreshold = 1e-4;
    double compressionScale = 10.0;
    double tileBufferPixels = 5.0;
    int64_t clipScale = 1024;
    bool overwrite = false;

    ParamList pl;
    pl.add_option("in-results", "Cell projection result TSV", inResults, true)
      .add_option("in-boundaries", "Cell boundary GeoJSON/JSON or flat TSV/CSV table", inBoundaries, true)
      .add_option("in-centers", "Optional cell center TSV/CSV", inCenters)
      .add_option("out-prefix", "Output prefix", outPrefix, true)
      .add_option("format", "Tile encoding format: MLT or MVT", format)
      .add_option("boundary-format", "Boundary input format: auto, geojson, json, table, tsv, or csv", boundaryFormat)
      .add_option("id-col", "Cell ID column name in projection results", resultIdCol)
      .add_option("top-k", "Number of top factors to keep from dense factor columns", topK)
      .add_option("b-icol-id", "0-based cell ID column in table boundaries", bIcolId)
      .add_option("b-icol-x", "0-based X column in table boundaries", bIcolX)
      .add_option("b-icol-y", "0-based Y column in table boundaries", bIcolY)
      .add_option("c-icol-id", "0-based cell ID column in --in-centers", cIcolId)
      .add_option("c-icol-x", "0-based center X column in --in-centers", cIcolX)
      .add_option("c-icol-y", "0-based center Y column in --in-centers", cIcolY)
      .add_option("boundary-id-prop", "Cell ID property in boundary GeoJSON features", boundaryIdProp)
      .add_option("min-zoom", "Minimum zoom for pyramids", minZoom)
      .add_option("max-zoom", "Maximum zoom for initial export", maxZoom)
      .add_option("max-point-tile-bytes", "Maximum compressed bytes per point tile", maxPointTileBytes)
      .add_option("max-point-tile-features", "Maximum point features per tile", maxPointTileFeatures)
      .add_option("max-polygon-tile-bytes", "Maximum compressed bytes per polygon tile", maxPolygonTileBytes)
      .add_option("max-polygon-tile-features", "Maximum polygon features per tile", maxPolygonTileFeatures)
      .add_option("coord-scale", "Scale applied to input coordinates before tiling", coordScale)
      .add_option("prob-thres", "Minimum probability retained for nullable K/P fields", probThreshold)
      .add_option("scale-factor-compression", "Pyramid compression aggressiveness estimate", compressionScale)
      .add_option("tile-buffer-px", "Polygon tile buffer in screen pixels", tileBufferPixels)
      .add_option("clip-scale", "Integer scale used for polygon clipping", clipScale)
      .add_option("extent", "Vector tile extent", extent)
      .add_option("threads", "Number of threads", threads)
      .add_option("overwrite", "Overwrite PMTiles outputs", overwrite);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    format = to_upper(format);
    if (format != "MVT" && format != "MLT") {
        error("%s: --format must be MVT or MLT", __func__);
    }
    if (minZoom < 0 || maxZoom < minZoom || maxZoom > 31) {
        error("%s: invalid zoom range %d..%d", __func__, minZoom, maxZoom);
    }
    const bool useMvt = format == "MVT";
    fs::path prefix(outPrefix);
    fs::create_directories(prefix.parent_path().empty() ? fs::path(".") : prefix.parent_path());
    const fs::path tmpDir = prefix.parent_path() / "tmp";
    fs::create_directories(tmpDir);

    auto factors = read_factor_rows(inResults, resultIdCol, topK);
    auto geoms = read_cell_geometries_auto(inBoundaries, boundaryFormat, boundaryIdProp,
        bIcolId, bIcolX, bIcolY);
    auto centers = read_centers(inCenters, cIcolId, cIcolX, cIcolY);
    size_t nTop = 1;
    for (const auto& kv : factors) {
        nTop = std::max(nTop, kv.second.top.k.size());
    }

    pm_vector::FeatureTableSchema pointSchema =
        build_cell_schema(prefix.filename().string() + "-cells", static_cast<uint32_t>(extent), true, nTop);
    std::map<TileKey, pm_vector::PointTileData> pointTiles;
    double pointMinX = std::numeric_limits<double>::infinity();
    double pointMinY = std::numeric_limits<double>::infinity();
    double pointMaxX = -std::numeric_limits<double>::infinity();
    double pointMaxY = -std::numeric_limits<double>::infinity();
    uint64_t pointCount = 0;

    for (size_t i = 0; i < geoms.size(); ++i) {
        const CellGeometry& geom = geoms[i];
        auto fit = factors.find(geom.cellId);
        if (fit == factors.end()) {
            warning("%s: skipping geometry for cell %s because it is absent from projection results",
                __func__, geom.cellId.c_str());
            continue;
        }
        const TopFactors& top = fit->second.top;
        CellCenter center = geom.center;
        auto cit = centers.find(geom.cellId);
        if (cit != centers.end()) {
            center = cit->second;
        }
        append_cell_point(pointTiles, pointSchema, center, static_cast<uint64_t>(i),
            geom.cellId, top, nTop, probThreshold, coordScale, static_cast<uint8_t>(maxZoom),
            pointMinX, pointMinY, pointMaxX, pointMaxY);
        ++pointCount;
    }

    const fs::path boundariesPmtiles = outPrefix + "-boundaries.pmtiles";
    factor_polygon_pmtiles::Options polyOpts;
    polyOpts.inTsv = inResults;
    polyOpts.inGeom = inBoundaries;
    polyOpts.outFile = boundariesPmtiles.string();
    polyOpts.format = format;
    polyOpts.idColName = resultIdCol;
    polyOpts.geomFormat = boundaryFormat;
    polyOpts.geomIdProp = boundaryIdProp;
    polyOpts.geomIdCol = bIcolId;
    polyOpts.geomXCol = bIcolX;
    polyOpts.geomYCol = bIcolY;
    polyOpts.zoom = maxZoom;
    polyOpts.coordScale = coordScale;
    polyOpts.probThreshold = probThreshold;
    polyOpts.topK = topK;
    polyOpts.tileBufferPixels = tileBufferPixels;
    polyOpts.clipScale = clipScale;
    polyOpts.extent = extent;
    polyOpts.threads = threads;
    polyOpts.cartoscopeBoundary = true;
    if (factor_polygon_pmtiles::write(polyOpts) != 0) {
        error("%s: failed writing cell boundary PMTiles", __func__);
    }
    build_pyramid(boundariesPmtiles, tmpDir, false, minZoom, maxPolygonTileBytes,
        maxPolygonTileFeatures, compressionScale, threads);

    const fs::path cellsPmtiles = outPrefix + "-cells.pmtiles";
    auto encodedPoints = encode_point_tiles(pointTiles, pointSchema, static_cast<uint8_t>(maxZoom), useMvt);
    write_pmtiles_archive(cellsPmtiles, pointSchema, pm_vector::VectorGeometryType::Point,
        std::move(encodedPoints), pointCount, coordScale, static_cast<uint8_t>(maxZoom),
        pointMinX, pointMinY, pointMaxX, pointMaxY, useMvt, "punkst cells2pmtiles");
    build_pyramid(cellsPmtiles, tmpDir, true, minZoom, maxPointTileBytes,
        maxPointTileFeatures, compressionScale, threads);

    fs::remove_all(tmpDir);
    notice("%s: wrote cell PMTiles with prefix %s", __func__, outPrefix.c_str());
    return 0;
}
