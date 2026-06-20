#include "factor_polygon_pmtiles.hpp"
#include "dataunits.hpp"
#include "hexgrid.h"
#include "json.hpp"
#include "mlt_utils.hpp"
#include "mvt_utils.hpp"
#include "pmtiles_utils.hpp"
#include "simple_polygon_pmtiles.hpp"
#include "region_query.hpp"
#include "utils.h"
#include "utils_sys.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cinttypes>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

constexpr double kSqrt3 = 1.73205080756887729353;
constexpr double kPi = 3.14159265358979323846;

struct TopFactor {
    int32_t k = -1;
    float p = 0.0f;
};

struct StatRow {
    std::string id;
    double x = 0.0;
    double y = 0.0;
    std::vector<TopFactor> top;
    std::vector<float> dense;
};

struct StatLayout {
    std::vector<std::string> header;
    int32_t idCol = -1;
    int32_t xCol = -1;
    int32_t yCol = -1;
    int32_t topKCol = -1;
    int32_t topPCol = -1;
    std::vector<std::pair<int32_t, int32_t>> denseCols; // factor, column
    std::vector<std::pair<int32_t, int32_t>> kpCols;    // K column, P column
};

struct RingRecord {
    uint64_t featureId = 0;
    uint32_t assignedId = 0;
    std::string polygonId;
    size_t partIndex = 0;
    std::vector<std::pair<double, double>> ring;
    std::pair<double, double> center{0.0, 0.0};
};

std::vector<std::pair<double, double>> build_pointy_hex_ring(double centerX, double centerY, double hexGridDistScaled) {
    const double edge = hexGridDistScaled / kSqrt3;
    std::vector<std::pair<double, double>> out;
    out.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double angle = (30.0 + 60.0 * static_cast<double>(i)) * kPi / 180.0;
        out.emplace_back(centerX + edge * std::cos(angle), centerY + edge * std::sin(angle));
    }
    return out;
}

std::string valid_original_id_column_name(const std::string& requested) {
    if (requested.empty()) return "ID_org";
    const unsigned char first = static_cast<unsigned char>(requested.front());
    if (!(std::isalpha(first) || first == '_')) return "ID_org";
    for (unsigned char ch : requested) {
        if (!(std::isalnum(ch) || ch == '_')) return "ID_org";
    }
    return requested;
}

pm_vector::FeatureTableSchema build_schema(const std::string& layerName,
    uint32_t extent, const StatLayout& layout, bool hexMode,
    bool cartoscopeBoundary, bool keepOrgId,
    const std::string& idColName, size_t nTop) {
    pm_vector::FeatureTableSchema schema;
    schema.layerName = layerName;
    schema.extent = extent;
    schema.hasIdColumn = true;
    schema.idIsUint64 = true;
    if (cartoscopeBoundary) {
        schema.columns.push_back({"cell_id", pm_vector::ScalarType::STRING, false});
        schema.columns.push_back({"topK", pm_vector::ScalarType::STRING, false});
        schema.columns.push_back({"topP", pm_vector::ScalarType::FLOAT, false});
        for (size_t i = 2; i <= nTop; ++i) {
            schema.columns.push_back({"K" + std::to_string(i), pm_vector::ScalarType::STRING, true});
            schema.columns.push_back({"P" + std::to_string(i), pm_vector::ScalarType::FLOAT, true});
        }
        return schema;
    }
    if (keepOrgId && !hexMode) {
        schema.columns.push_back({valid_original_id_column_name(idColName), pm_vector::ScalarType::STRING, true});
    }
    if (hexMode) {
        schema.columns.push_back({"hex_q", pm_vector::ScalarType::INT_32, false});
        schema.columns.push_back({"hex_r", pm_vector::ScalarType::INT_32, false});
    }
    schema.columns.push_back({"topK", pm_vector::ScalarType::INT_32, false});
    schema.columns.push_back({"topP", pm_vector::ScalarType::FLOAT, false});
    for (const auto& fc : layout.denseCols) {
        schema.columns.push_back({std::to_string(fc.first), pm_vector::ScalarType::FLOAT, true});
    }
    return schema;
}

simple_polygon_pmtiles::PolygonFeatureProperties make_props(const StatRow& row,
    const StatLayout& layout, bool hexMode,
    bool cartoscopeBoundary, bool keepOrgId,
    const std::string& polygonId, int32_t hexQ, int32_t hexR,
    double probThreshold, size_t nTopSchema) {
    simple_polygon_pmtiles::PolygonFeatureProperties props;
    if (cartoscopeBoundary) {
        const TopFactor first = row.top.empty() ? TopFactor{} : row.top.front();
        props.stringValues.push_back(polygonId);
        props.stringValues.push_back(std::to_string(first.k));
        props.floatValues.push_back(first.p);
        for (size_t i = 1; i < nTopSchema; ++i) {
            const bool present = i < row.top.size() && row.top[i].k >= 0 &&
                row.top[i].p >= static_cast<float>(probThreshold);
            props.stringValues.push_back(present ? std::optional<std::string>(std::to_string(row.top[i].k)) : std::nullopt);
            props.floatValues.push_back(present ? std::optional<float>(row.top[i].p) : std::nullopt);
        }
        return props;
    }
    if (keepOrgId && !hexMode) props.stringValues.push_back(polygonId);
    if (hexMode) {
        props.intValues.push_back(hexQ);
        props.intValues.push_back(hexR);
    }
    const TopFactor first = row.top.empty() ? TopFactor{} : row.top.front();
    props.intValues.push_back(first.k);
    props.floatValues.push_back(first.p);
    for (float prob : row.dense) {
        props.floatValues.push_back(std::isfinite(prob) && prob >= static_cast<float>(probThreshold)
            ? std::optional<float>(prob) : std::nullopt);
    }
    return props;
}

void scale_ring_in_place(std::vector<std::pair<double, double>>& ring, double coordScale) {
    for (auto& pt : ring) {
        pt.first *= coordScale;
        pt.second *= coordScale;
    }
}

void write_sidecar(const std::string& path, const std::vector<RingRecord>& records) {
    if (path.empty()) return;
    std::ofstream out(path);
    if (!out.is_open()) error("%s: cannot open %s", __func__, path.c_str());
    out << "polygon_id\tpart_index\tfeature_id\tcenter_x\tcenter_y\n";
    for (const auto& rec : records) {
        out << rec.polygonId << "\t" << rec.partIndex << "\t" << rec.featureId
            << "\t" << rec.center.first << "\t" << rec.center.second << "\n";
    }
    if (!out.good()) error("%s: failed writing %s", __func__, path.c_str());
}

} // namespace

namespace factor_polygon_pmtiles {

int write(const Options& options) {
    std::string format = options.format;
    std::string geomFormat = options.geomFormat;
    std::string layerName = options.layerName;
    bool keepOrgId = options.keepOrgId;
    format = to_upper(format);
    geomFormat = to_lower(geomFormat);
    if (format != "MLT" && format != "MVT") error("%s: --format must be MLT or MVT", __func__);
    if (options.zoom < 0 || options.zoom > 31) error("%s: --pmtiles-zoom must be in [0,31]", __func__);
    if (options.topK <= 0) error("%s: --top-k must be positive", __func__);
    if (options.coordScale <= 0.0 || options.extent <= 0 || options.clipScale <= 0 ||
        options.tileBufferPixels < 0.0) {
        error("%s: invalid scale/extent/buffer options", __func__);
    }
    if (options.noClipping && options.noDuplication) {
        error("%s: --no-clipping and --no-duplication are mutually exclusive", __func__);
    }
    const bool useFactorRange = options.factorColBegin >= 0 || options.factorColEnd >= 0;
    if (useFactorRange && (options.factorColBegin < 0 || options.factorColEnd < 0)) {
        error("%s: --factor-col-begin and --factor-col-end must be provided together", __func__);
    }
    if (useFactorRange && options.factorColEnd < options.factorColBegin) {
        error("%s: --factor-col-end must be >= --factor-col-begin", __func__);
    }
    const bool genericMode = !options.inGeom.empty();
    if (genericMode && options.idColName.empty()) error("%s: --id-col is required with --in-geom", __func__);
    if (!genericMode && options.hexGridDist <= 0.0) error("%s: --hex-grid-dist must be positive in hex mode", __func__);
    if (options.idIsU32 && keepOrgId) error("%s: --keep-org-id cannot be used with --id-is-u32", __func__);
    if (options.cartoscopeBoundary && !keepOrgId) keepOrgId = true;

    UnitFactorResultReadOptions factorOpts;
    factorOpts.unitIdColName = genericMode ? options.idColName : std::string();
    factorOpts.xColName = genericMode ? std::string() : options.xColName;
    factorOpts.yColName = genericMode ? std::string() : options.yColName;
    factorOpts.topKColName = options.topKColName;
    factorOpts.topPColName = options.topPColName;
    if (useFactorRange) {
        factorOpts.factorColBegin = options.factorColBegin;
        factorOpts.factorColEnd = options.factorColEnd + 1;
    }
    factorOpts.denseTopK = options.topK;
    factorOpts.requireFactorValues = false;
    factorOpts.allowKpColumns = true;
    factorOpts.autoDetectDelimiter = true;
    UnitFactorResultReader reader(options.inTsv, factorOpts);
    StatLayout layout;
    layout.header = reader.header().columns;
    layout.idCol = reader.header().unitIdCol;
    layout.xCol = reader.header().xCol;
    layout.yCol = reader.header().yCol;
    layout.topKCol = reader.header().topKCol;
    layout.topPCol = reader.header().topPCol;
    layout.denseCols = reader.header().factorCols;
    layout.kpCols = reader.header().topPairCols;

    if (layerName.empty()) layerName = basename(options.outFile, true);
    fs::path outPath(options.outFile);
    if (outPath.has_parent_path()) fs::create_directories(outPath.parent_path());

    std::vector<RingRecord> geomRecords;
    std::unordered_map<std::string, std::vector<RingRecord>> geomById;
    if (genericMode) {
        SimplePolygonTableReadOptions tableOpts;
        tableOpts.idCol = options.geomIdCol;
        tableOpts.xCol = options.geomXCol;
        tableOpts.yCol = options.geomYCol;
        tableOpts.orderCol = options.geomOrderCol;
        tableOpts.idIsU32 = options.idIsU32;
        const auto simpleRecords = readSimplePolygonsAuto(
            options.inGeom, geomFormat, options.geomIdProp, tableOpts);
        geomRecords.reserve(simpleRecords.size());
        for (const auto& src : simpleRecords) {
            RingRecord rec;
            rec.featureId = src.featureId;
            rec.assignedId = src.assignedId;
            rec.polygonId = src.polygonId;
            rec.partIndex = src.partIndex;
            rec.ring = src.ring;
            rec.center = src.center;
            geomRecords.push_back(std::move(rec));
        }
        for (const auto& rec : geomRecords) geomById[rec.polygonId].push_back(rec);
    }

    size_t nTopSchema = 1;
    if (!layout.kpCols.empty()) {
        nTopSchema = layout.kpCols.size();
    } else if (!layout.denseCols.empty()) {
        nTopSchema = static_cast<size_t>(options.topK);
    }
    const pm_vector::FeatureTableSchema schema =
        build_schema(layerName, static_cast<uint32_t>(options.extent), layout, !genericMode,
            options.cartoscopeBoundary, keepOrgId, options.idColName, nTopSchema);

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
    std::vector<RingRecord> emitted;
    HexGrid inputGrid;
    HexGrid scaledGrid;
    double scaledGridDist = 0.0;
    if (!genericMode) {
        inputGrid = HexGrid(options.hexGridDist / kSqrt3, true);
        scaledGrid = HexGrid((options.hexGridDist * options.coordScale) / kSqrt3, true);
        scaledGridDist = options.hexGridDist * options.coordScale;
    }

    UnitFactorResultRow parsed;
    StatRow row;
    uint64_t nRows = 0;
    while (reader.next(parsed)) {
        row = StatRow{};
        row.id = parsed.unitId;
        row.x = parsed.x;
        row.y = parsed.y;
        row.dense = parsed.factorValues;
        row.top.reserve(parsed.topFactors.size());
        for (const UnitTopFactor& tf : parsed.topFactors) {
            row.top.push_back({tf.k, tf.p});
        }
        if (genericMode) {
            auto git = geomById.find(row.id);
            if (git == geomById.end()) error("%s: polygon ID %s not found in geometry", __func__, row.id.c_str());
            for (RingRecord rec : git->second) {
                std::vector<std::pair<double, double>> scaledRing = rec.ring;
                scale_ring_in_place(scaledRing, options.coordScale);
                auto props = make_props(row, layout, false, options.cartoscopeBoundary, keepOrgId,
                    rec.polygonId, 0, 0, options.probThreshold, nTopSchema);
                simple_polygon_pmtiles::append_simple_polygon_feature(
                    tileMap, schema, scaledRing, rec.featureId, props, writerOptions, summary);
                emitted.push_back(std::move(rec));
            }
        } else {
            int32_t hexQ = 0, hexR = 0;
            inputGrid.cart_to_axial(hexQ, hexR, row.x, row.y);
            double centerX = 0.0, centerY = 0.0;
            scaledGrid.axial_to_cart(centerX, centerY, hexQ, hexR);
            std::vector<std::pair<double, double>> ring = build_pointy_hex_ring(centerX, centerY, scaledGridDist);
            const uint64_t featureId =
                (static_cast<uint64_t>(static_cast<uint32_t>(hexQ)) << 32u) |
                static_cast<uint64_t>(static_cast<uint32_t>(hexR));
            const std::string polygonId = std::to_string(hexQ) + "_" + std::to_string(hexR);
            auto props = make_props(row, layout, true, options.cartoscopeBoundary, keepOrgId,
                polygonId, hexQ, hexR, options.probThreshold, nTopSchema);
            simple_polygon_pmtiles::append_simple_polygon_feature(
                tileMap, schema, ring, featureId, props, writerOptions, summary,
                std::make_optional(std::make_pair(centerX, centerY)));
            RingRecord rec;
            rec.polygonId = polygonId;
            rec.featureId = featureId;
            rec.center = {centerX / options.coordScale, centerY / options.coordScale};
            emitted.push_back(std::move(rec));
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
        encodedTiles = simple_polygon_pmtiles::encode_polygon_tile_map(tileMap, schema, nullptr, writerOptions);
    }

    json sourceMetadata = {
        {"coord_scale_applied", options.coordScale},
        {"statistics_id_column", options.idColName},
        {"cartoscope_boundary", options.cartoscopeBoundary},
    };
    if (!genericMode) {
        sourceMetadata["hex_grid_dist_scaled"] = scaledGridDist;
        sourceMetadata["hex_orientation"] = "pointy";
    }
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
    archiveOptions.generator = "punkst poly2pmtiles";
    archiveOptions.description = "Generated PMTiles by punkst poly2pmtiles";
    archiveOptions.extraMetadata = simple_polygon_pmtiles::build_simple_polygon_metadata(
        genericMode ? "polygon_table" : "hexgrid", {}, writerOptions, sourceMetadata);
    pm_vector::write_single_layer_vector_pmtiles_archive(options.outFile, std::move(encodedTiles), archiveOptions);
    write_sidecar(options.outSidecar, emitted);
    notice("%s: wrote %" PRIu64 " statistics rows into %s (%zu destination tiles)",
        __func__, nRows, options.outFile.c_str(), tileMap.size());
    return 0;
}

} // namespace factor_polygon_pmtiles
