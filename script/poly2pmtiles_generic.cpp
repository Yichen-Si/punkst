#include "punkst.h"

#include "json.hpp"
#include "mvt_utils.hpp"
#include "pmtiles_utils.hpp"
#include "region_query.hpp"
#include "simple_polygon_pmtiles.hpp"
#include "utils.h"
#include "utils_sys.hpp"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct Options {
    std::string inTsv;
    std::string inGeom;
    std::string outFile;
    std::string layerName;
    std::string format = "MVT";
    std::string idColName;
    std::string geomFormat = "auto";
    std::string geomIdProp = "cell_id";
    std::vector<std::string> intCols;
    std::vector<std::string> floatCols;
    std::vector<std::string> stringCols;
    int32_t geomIdCol = 0;
    int32_t geomXCol = 1;
    int32_t geomYCol = 2;
    int32_t geomOrderCol = -1;
    int32_t zoom = -1;
    int32_t extent = 4096;
    int32_t threads = 1;
    double coordScale = 1.0;
    double tileBufferPixels = 5.0;
    int64_t clipScale = 1024;
    bool noClipping = false;
    bool noDuplication = false;
    bool idIsU32 = false;
};

struct RingRecord {
    uint64_t featureId = 0;
    std::string polygonId;
    std::vector<std::pair<double, double>> ring;
};

struct ColumnSpec {
    std::string name;
    int32_t sourceCol = -1;
    pm_vector::ScalarType type = pm_vector::ScalarType::STRING;
};

bool is_missing_token(const std::string& token) {
    const std::string t = trim(token);
    return t.empty() || t == "NA" || t == "NaN" || t == "nan" || t == "NULL" || t == "null";
}

std::string clean_header_name(std::string name) {
    if (!name.empty() && name.front() == '#') {
        name.erase(name.begin());
    }
    return name;
}

std::vector<std::string> split_column_names(const std::vector<std::string>& values) {
    std::vector<std::string> out;
    for (const auto& value : values) {
        std::vector<std::string> parts;
        split(parts, ",", value, UINT_MAX, true, false, true);
        for (const auto& part : parts) {
            if (!part.empty()) {
                out.push_back(part);
            }
        }
    }
    return out;
}

void scale_ring_in_place(std::vector<std::pair<double, double>>& ring, double coordScale) {
    for (auto& pt : ring) {
        pt.first *= coordScale;
        pt.second *= coordScale;
    }
}

int32_t require_header_col(const std::vector<std::string>& header, const std::string& name) {
    const int32_t col = find_header_column_exact(header, name);
    if (col < 0) {
        error("%s: column '%s' not found in input TSV", __func__, name.c_str());
    }
    return col;
}

void add_column_specs(const std::vector<std::string>& names,
    pm_vector::ScalarType type,
    const std::vector<std::string>& header,
    std::vector<ColumnSpec>& specs,
    std::set<std::string>& seen) {
    for (const auto& name : names) {
        if (!seen.insert(name).second) {
            error("%s: duplicate output property column '%s'", __func__, name.c_str());
        }
        specs.push_back({name, require_header_col(header, name), type});
    }
}

pm_vector::FeatureTableSchema build_schema(const std::string& layerName,
    uint32_t extent,
    const std::vector<ColumnSpec>& specs) {
    pm_vector::FeatureTableSchema schema;
    schema.layerName = layerName;
    schema.extent = extent;
    schema.hasIdColumn = true;
    schema.idIsUint64 = true;
    for (const auto& spec : specs) {
        schema.columns.push_back({spec.name, spec.type, true});
    }
    return schema;
}

simple_polygon_pmtiles::PolygonFeatureProperties row_props(
    const std::vector<std::string>& fields,
    const std::vector<ColumnSpec>& specs,
    uint64_t rowNo) {
    simple_polygon_pmtiles::PolygonFeatureProperties props;
    for (const auto& spec : specs) {
        require_fields(fields, spec.sourceCol, __func__, rowNo);
        const std::string& token = fields[static_cast<size_t>(spec.sourceCol)];
        if (spec.type == pm_vector::ScalarType::STRING) {
            props.stringValues.push_back(is_missing_token(token)
                ? std::optional<std::string>() : std::optional<std::string>(token));
        } else if (spec.type == pm_vector::ScalarType::INT_32) {
            int32_t value = 0;
            if (is_missing_token(token)) {
                props.intValues.push_back(std::nullopt);
            } else if (str2int32(token, value)) {
                props.intValues.push_back(value);
            } else {
                error("%s: failed parsing INT_32 column '%s' value '%s' at row %" PRIu64,
                    __func__, spec.name.c_str(), token.c_str(), rowNo);
            }
        } else if (spec.type == pm_vector::ScalarType::FLOAT) {
            float value = 0.0f;
            if (is_missing_token(token)) {
                props.floatValues.push_back(std::nullopt);
            } else if (str2float(token, value)) {
                props.floatValues.push_back(std::isfinite(value) ? std::optional<float>(value) : std::nullopt);
            } else {
                error("%s: failed parsing FLOAT column '%s' value '%s' at row %" PRIu64,
                    __func__, spec.name.c_str(), token.c_str(), rowNo);
            }
        } else {
            error("%s: unsupported property type", __func__);
        }
    }
    return props;
}

std::vector<RingRecord> read_geometry(const Options& options) {
    SimplePolygonTableReadOptions tableOpts;
    tableOpts.idCol = options.geomIdCol;
    tableOpts.xCol = options.geomXCol;
    tableOpts.yCol = options.geomYCol;
    tableOpts.orderCol = options.geomOrderCol;
    tableOpts.idIsU32 = options.idIsU32;
    const auto simpleRecords = readSimplePolygonsAuto(
        options.inGeom, to_lower(options.geomFormat), options.geomIdProp, tableOpts);

    std::vector<RingRecord> out;
    out.reserve(simpleRecords.size());
    for (const auto& src : simpleRecords) {
        RingRecord rec;
        rec.featureId = src.featureId;
        rec.polygonId = src.polygonId;
        rec.ring = src.ring;
        out.push_back(std::move(rec));
    }
    return out;
}

int run(const Options& options) {
    std::string format = to_upper(options.format);
    if (format != "MLT" && format != "MVT") {
        error("%s: --format must be MLT or MVT", __func__);
    }
    if (options.zoom < 0 || options.zoom > 31) {
        error("%s: --pmtiles-zoom must be in [0,31]", __func__);
    }
    if (options.idColName.empty()) {
        error("%s: --id-col is required", __func__);
    }
    if (options.coordScale <= 0.0 || options.extent <= 0 || options.clipScale <= 0 ||
        options.tileBufferPixels < 0.0) {
        error("%s: invalid scale/extent/buffer options", __func__);
    }
    if (options.noClipping && options.noDuplication) {
        error("%s: --no-clipping and --no-duplication are mutually exclusive", __func__);
    }

    TextLineReader reader(options.inTsv);
    std::string headerLine;
    uint64_t rowNo = 0;
    if (!reader.getline(headerLine)) {
        error("%s: empty input TSV: %s", __func__, options.inTsv.c_str());
    }
    ++rowNo;
    headerLine = strip_leading_hash(headerLine);
    const char delim = infer_table_delimiter(headerLine);
    std::vector<std::string> header = split_delimited(headerLine, delim);
    for (auto& name : header) {
        name = clean_header_name(name);
    }

    std::vector<ColumnSpec> specs;
    std::set<std::string> seen;
    seen.insert(options.idColName);
    specs.push_back({options.idColName, require_header_col(header, options.idColName),
        pm_vector::ScalarType::STRING});
    add_column_specs(split_column_names(options.stringCols), pm_vector::ScalarType::STRING,
        header, specs, seen);
    add_column_specs(split_column_names(options.intCols), pm_vector::ScalarType::INT_32,
        header, specs, seen);
    add_column_specs(split_column_names(options.floatCols), pm_vector::ScalarType::FLOAT,
        header, specs, seen);

    std::string layerName = options.layerName;
    if (layerName.empty()) {
        layerName = basename(options.outFile, true);
    }
    fs::path outPath(options.outFile);
    if (outPath.has_parent_path()) {
        fs::create_directories(outPath.parent_path());
    }

    const auto geomRecords = read_geometry(options);
    std::unordered_map<std::string, std::vector<RingRecord>> geomById;
    for (const auto& rec : geomRecords) {
        geomById[rec.polygonId].push_back(rec);
    }

    const pm_vector::FeatureTableSchema schema =
        build_schema(layerName, static_cast<uint32_t>(options.extent), specs);

    simple_polygon_pmtiles::SingleZoomPolygonWriterOptions writerOptions;
    writerOptions.zoom = static_cast<uint8_t>(options.zoom);
    writerOptions.extent = static_cast<uint32_t>(options.extent);
    writerOptions.coordScale = options.coordScale;
    writerOptions.tileBufferPixels = options.tileBufferPixels;
    writerOptions.clipScale = options.clipScale;
    writerOptions.threads = options.threads;
    writerOptions.boundaryMode = options.noDuplication ?
        simple_polygon_pmtiles::PolygonBoundaryMode::SingleTileNoDuplication :
        (options.noClipping ? simple_polygon_pmtiles::PolygonBoundaryMode::NoClippingDuplicate :
            simple_polygon_pmtiles::PolygonBoundaryMode::BufferClipDuplicate);

    std::map<TileKey, pm_vector::PolygonTileData> tileMap;
    simple_polygon_pmtiles::PolygonWriteSummary summary;
    uint64_t nRows = 0;
    uint64_t nParts = 0;
    std::vector<std::string> fields;
    std::string line;
    while (reader.getline(line)) {
        ++rowNo;
        if (line.empty()) {
            continue;
        }
        fields = split_delimited(line, delim);
        const int32_t idCol = specs.front().sourceCol;
        require_fields(fields, idCol, __func__, rowNo);
        const std::string polygonId = fields[static_cast<size_t>(idCol)];
        auto git = geomById.find(polygonId);
        if (git == geomById.end()) {
            error("%s: polygon ID '%s' not found in geometry", __func__, polygonId.c_str());
        }
        const auto props = row_props(fields, specs, rowNo);
        for (const auto& rec : git->second) {
            std::vector<std::pair<double, double>> scaledRing = rec.ring;
            scale_ring_in_place(scaledRing, options.coordScale);
            simple_polygon_pmtiles::append_simple_polygon_feature(
                tileMap, schema, scaledRing, rec.featureId, props, writerOptions, summary);
            ++nParts;
        }
        ++nRows;
    }

    std::vector<pm_core::EncodedTilePayload> encodedTiles;
    if (format == "MVT") {
        for (const auto& kv : tileMap) {
            pm_core::EncodedTilePayload payload;
            payload.z = static_cast<uint8_t>(options.zoom);
            payload.x = static_cast<uint32_t>(kv.first.col);
            payload.y = static_cast<uint32_t>(kv.first.row);
            payload.tileId = pmtiles::zxy_to_tileid(payload.z, payload.x, payload.y);
            payload.featureCount = static_cast<uint32_t>(kv.second.size());
            payload.compressedData = pm_core::gzip_compress(
                mvt_pmtiles::encode_polygon_tile(schema, kv.second, nullptr));
            encodedTiles.push_back(std::move(payload));
        }
    } else {
        encodedTiles = simple_polygon_pmtiles::encode_polygon_tile_map(
            tileMap, schema, nullptr, writerOptions);
    }

    json sourceMetadata = {
        {"coord_scale_applied", options.coordScale},
        {"statistics_id_column", options.idColName},
    };
    pm_vector::SingleLayerVectorPmtilesOptions archiveOptions;
    archiveOptions.schema = schema;
    archiveOptions.geometryType = pm_vector::VectorGeometryType::Polygon;
    archiveOptions.tileType = format == "MVT" ? pmtiles::TILETYPE_MVT : pmtiles::TILETYPE_MLT;
    archiveOptions.totalRecordCount = summary.featureCount;
    archiveOptions.coordScale = options.coordScale;
    archiveOptions.featureDictionarySize = 0;
    archiveOptions.outputZoom = static_cast<uint8_t>(options.zoom);
    archiveOptions.geoMinX = summary.geoMinX;
    archiveOptions.geoMinY = summary.geoMinY;
    archiveOptions.geoMaxX = summary.geoMaxX;
    archiveOptions.geoMaxY = summary.geoMaxY;
    archiveOptions.generator = "punkst poly2pmtiles-generic";
    archiveOptions.description = "Generated PMTiles by punkst poly2pmtiles-generic";
    archiveOptions.extraMetadata = simple_polygon_pmtiles::build_simple_polygon_metadata(
        "polygon_table", {options.idColName}, writerOptions, sourceMetadata);
    pm_vector::write_single_layer_vector_pmtiles_archive(
        options.outFile, std::move(encodedTiles), archiveOptions);
    notice("%s: wrote %" PRIu64 " rows / %" PRIu64 " polygon parts into %s (%zu destination tiles)",
        __func__, nRows, nParts, options.outFile.c_str(), tileMap.size());
    return 0;
}

} // namespace

int32_t cmdPoly2PmtilesGeneric(int32_t argc, char** argv) {
    Options opts;
    ParamList pl;
    pl.add_option("in-tsv", "Input property TSV", opts.inTsv, true)
      .add_option("in-geom", "Polygon geometry TSV/CSV/GeoJSON/JSON", opts.inGeom, true)
      .add_option("out", "Output single-zoom PMTiles file", opts.outFile, true)
      .add_option("format", "Tile encoding format: MLT or MVT", opts.format)
      .add_option("layer-name", "Optional PMTiles layer name (default: basename of --out)", opts.layerName)
      .add_option("id-col", "Polygon ID column name in the property TSV", opts.idColName, true)
      .add_option("int-cols", "Integer property columns; repeat or comma-separate names", opts.intCols)
      .add_option("float-cols", "Float property columns; repeat or comma-separate names", opts.floatCols)
      .add_option("string-cols", "String property columns; repeat or comma-separate names", opts.stringCols)
      .add_option("pmtiles-zoom", "Web Mercator zoom level for PMTiles export", opts.zoom, true)
      .add_option("coord-scale", "Scale factor applied to input coordinates before tiling", opts.coordScale)
      .add_option("tile-buffer-px", "Tile buffer in screen pixels for clipped polygon output", opts.tileBufferPixels)
      .add_option("no-clipping", "Duplicate polygons across touched tiles without clipping", opts.noClipping)
      .add_option("no-duplication", "Store each polygon intact in exactly one tile", opts.noDuplication)
      .add_option("clip-scale", "Integer scale used internally for polygon clipping", opts.clipScale)
      .add_option("extent", "Vector tile extent", opts.extent)
      .add_option("threads", "Number of encode threads", opts.threads)
      .add_option("geom-format", "Geometry format: auto, table, geojson, or json", opts.geomFormat)
      .add_option("geom-id-prop", "Polygon ID property in GeoJSON/JSON features", opts.geomIdProp)
      .add_option("g-icol-id", "0-based polygon ID column in table geometry", opts.geomIdCol)
      .add_option("g-icol-x", "0-based x coordinate column in table geometry", opts.geomXCol)
      .add_option("g-icol-y", "0-based y coordinate column in table geometry", opts.geomYCol)
      .add_option("g-icol-order", "Optional 0-based vertex-order column in table geometry", opts.geomOrderCol)
      .add_option("id-is-u32", "Parse polygon IDs directly as u32 feature IDs where possible", opts.idIsU32);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    return run(opts);
}
