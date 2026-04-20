#include "punkst.h"

#include "dataunits.hpp"
#include "hexgrid.h"
#include "pmtiles_utils.hpp"
#include "simple_polygon_pmtiles.hpp"
#include "utils.h"

#include <cinttypes>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <unordered_map>
#include <unordered_set>

namespace {

constexpr double kSqrt3 = 1.73205080756887729353;
constexpr double kPi = 3.14159265358979323846;

std::vector<std::pair<double, double>> build_pointy_hex_ring(double centerX, double centerY, double hexGridDistScaled) {
    const double edge = hexGridDistScaled / kSqrt3;
    std::vector<std::pair<double, double>> out;
    out.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double angle = (30.0 + 60.0 * static_cast<double>(i)) * kPi / 180.0;
        out.emplace_back(centerX + edge * std::cos(angle),
            centerY + edge * std::sin(angle));
    }
    return out;
}

std::string build_hex_polygon_id(int32_t hexQ, int32_t hexR) {
    return std::to_string(hexQ) + "_" + std::to_string(hexR);
}

uint64_t pack_hex_polygon_id(int32_t hexQ, int32_t hexR) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(hexQ)) << 32u) |
        static_cast<uint64_t>(static_cast<uint32_t>(hexR));
}

struct SimplePolygonWorldRecord {
    std::string polygonId;
    uint32_t assignedId = 0;
    std::vector<std::pair<double, double>> outerRing;
};

struct SimplePolygonGeomReadOptions {
    int32_t icolId = 0;
    int32_t icolX = 1;
    int32_t icolY = 2;
    int32_t icolOrder = -1;
    char delimiter = '\0';
};

char infer_delimiter_or_die(const std::string& line) {
    if (line.find('\t') != std::string::npos) {
        return '\t';
    }
    if (line.find(',') != std::string::npos) {
        return ',';
    }
    error("%s: could not infer delimiter from line: %s", __func__, line.c_str());
    return '\t';
}

std::unordered_map<std::string, SimplePolygonWorldRecord> read_simple_polygon_geometry_map(
    const std::string& path,
    const SimplePolygonGeomReadOptions& options,
    bool idIsU32) {
    if (options.icolId < 0 || options.icolX < 0 || options.icolY < 0) {
        error("%s: --icol-id-geom, --icol-x-geom, and --icol-y-geom must be non-negative", __func__);
    }
    TextLineReader reader(path);
    std::unordered_map<std::string, std::vector<std::tuple<int64_t, double, double>>> staged;
    std::unordered_map<std::string, uint32_t> assignedIdsByPolygonId;
    uint32_t nextAssignedId = 0;
    std::string line;
    uint64_t lineNo = 0;
    bool firstDataRowHandled = false;
    bool sawData = false;
    std::unordered_set<uint32_t> seenAssignedIds;
    while (reader.getline(line)) {
        ++lineNo;
        if (line.empty() || line.front() == '#') {
            continue;
        }
        const char delimiter = options.delimiter != '\0' ? options.delimiter : infer_delimiter_or_die(line);
        const std::string delims(1, delimiter);
        std::vector<std::string> fields;
        split(fields, delims, line, UINT_MAX, true, false, false, false);
        const int32_t maxRequired = std::max({options.icolId, options.icolX, options.icolY, options.icolOrder});
        if (static_cast<int32_t>(fields.size()) <= maxRequired) {
            if (!firstDataRowHandled) {
                error("%s: geometry file row %" PRIu64 " has fewer than %d required columns",
                    __func__, lineNo, maxRequired + 1);
            }
            error("%s: invalid geometry row %" PRIu64 ": expected at least %d columns, found %zu",
                __func__, lineNo, maxRequired + 1, fields.size());
        }
        double x = 0.0;
        double y = 0.0;
        const bool xOk = str2double(fields[options.icolX], x);
        const bool yOk = str2double(fields[options.icolY], y);
        if (!firstDataRowHandled) {
            firstDataRowHandled = true;
            if (!xOk || !yOk) {
                continue;
            }
        } else if (!xOk || !yOk) {
            error("%s: invalid x/y token in geometry row %" PRIu64, __func__, lineNo);
        }
        sawData = true;
        const std::string polygonId = fields[options.icolId];
        if (polygonId.empty()) {
            error("%s: empty polygon ID in geometry row %" PRIu64, __func__, lineNo);
        }
        if (assignedIdsByPolygonId.count(polygonId) == 0) {
            uint32_t assignedId = 0;
            if (idIsU32) {
                if (!str2uint32(polygonId, assignedId)) {
                    error("%s: failed parsing polygon ID %s as u32", __func__, polygonId.c_str());
                }
                if (!seenAssignedIds.insert(assignedId).second) {
                    error("%s: duplicate parsed u32 polygon ID %u", __func__, assignedId);
                }
            } else {
                assignedId = nextAssignedId++;
            }
            assignedIdsByPolygonId.emplace(polygonId, assignedId);
        }
        auto& rows = staged[polygonId];
        int64_t order = 0;
        if (options.icolOrder >= 0) {
            if (!str2int64(fields[options.icolOrder], order)) {
                error("%s: invalid vertex order in geometry row %" PRIu64, __func__, lineNo);
            }
        } else {
            order = static_cast<int64_t>(rows.size());
        }
        rows.emplace_back(order, x, y);
    }
    if (!sawData) {
        error("%s: geometry file %s contains no usable data rows", __func__, path.c_str());
    }

    std::unordered_map<std::string, SimplePolygonWorldRecord> out;
    out.reserve(staged.size());
    for (auto& kv : staged) {
        auto& rows = kv.second;
        std::sort(rows.begin(), rows.end(),
            [](const auto& lhs, const auto& rhs) {
                return std::get<0>(lhs) < std::get<0>(rhs);
            });
        SimplePolygonWorldRecord rec;
        rec.polygonId = kv.first;
        auto idIt = assignedIdsByPolygonId.find(kv.first);
        if (idIt == assignedIdsByPolygonId.end()) {
            error("%s: internal polygon ID assignment missing for %s", __func__, kv.first.c_str());
        }
        rec.assignedId = idIt->second;
        rec.outerRing.reserve(rows.size());
        for (size_t i = 0; i < rows.size(); ++i) {
            if (i > 0 && std::get<0>(rows[i]) == std::get<0>(rows[i - 1])) {
                error("%s: duplicate vertex order %" PRId64 " for polygon %s",
                    __func__, std::get<0>(rows[i]), kv.first.c_str());
            }
            rec.outerRing.emplace_back(std::get<1>(rows[i]), std::get<2>(rows[i]));
        }
        if (rec.outerRing.size() < 3) {
            error("%s: polygon %s has fewer than 3 vertices in geometry file",
                __func__, kv.first.c_str());
        }
        out.emplace(rec.polygonId, std::move(rec));
    }
    return out;
}

bool is_valid_mlt_column_name(const std::string& name) {
    if (name.empty()) {
        return false;
    }
    const unsigned char first = static_cast<unsigned char>(name.front());
    if (!(std::isalpha(first) || first == '_')) {
        return false;
    }
    for (unsigned char ch : name) {
        if (!(std::isalnum(ch) || ch == '_')) {
            return false;
        }
    }
    return true;
}

std::string choose_original_id_column_name(const std::string& requested) {
    return is_valid_mlt_column_name(requested) ? requested : std::string("ID_org");
}

mlt_pmtiles::FeatureTableSchema build_polygon_schema(const std::string& layerName,
    uint32_t extent,
    const std::vector<std::pair<int32_t, int32_t>>& factorCols,
    bool includeHexCoords,
    bool idIsUint64,
    const std::string& originalIdColumnName = std::string()) {
    mlt_pmtiles::FeatureTableSchema schema;
    schema.layerName = layerName;
    schema.extent = extent;
    schema.hasIdColumn = true;
    schema.idIsUint64 = idIsUint64;
    if (!originalIdColumnName.empty()) {
        schema.columns.push_back({originalIdColumnName, mlt_pmtiles::ScalarType::STRING, true});
    }
    if (includeHexCoords) {
        schema.columns.push_back({"hex_q", mlt_pmtiles::ScalarType::INT_32, false});
        schema.columns.push_back({"hex_r", mlt_pmtiles::ScalarType::INT_32, false});
    }
    schema.columns.push_back({"topK", mlt_pmtiles::ScalarType::INT_32, false});
    schema.columns.push_back({"topP", mlt_pmtiles::ScalarType::FLOAT, false});
    for (const auto& kv : factorCols) {
        schema.columns.push_back({std::to_string(kv.first), mlt_pmtiles::ScalarType::FLOAT, true});
    }
    return schema;
}

} // namespace

int32_t cmdHex2PmtilesMlt(int32_t argc, char** argv) {
    std::string inTsv;
    std::string inGeom;
    std::string outFile;
    std::string layerName;
    std::string xColName = "x";
    std::string yColName = "y";
    std::string topKColName = "topK";
    std::string topPColName = "topP";
    std::string idColName;
    int32_t icolIdGeom = 0;
    int32_t icolXGeom = 1;
    int32_t icolYGeom = 2;
    int32_t icolOrderGeom = -1;
    double hexGridDist = -1.0;
    double coordScale = 1.0;
    double probThreshold = 1e-4;
    double tileBufferPixels = 5.0;
    bool noClipping = false;
    bool noDuplication = false;
    bool idIsU32 = false;
    bool keepOrgId = false;
    int64_t clipScale = 1024;
    int32_t zoom = -1;
    int32_t extent = 4096;
    int32_t threads = 1;

    ParamList pl;
    pl.add_option("in-tsv", "Input TSV(.gz)", inTsv, true)
      .add_option("in-geom", "Optional polygon geometry TSV/CSV(.gz) for generic simple-polygon export", inGeom)
      .add_option("out", "Output single-zoom PMTiles file", outFile, true)
      .add_option("layer-name", "Optional PMTiles layer name (default: basename of --out)", layerName)
      .add_option("hex-grid-dist", "Distance between adjacent hex centers in the input coordinate system (hex mode only)", hexGridDist)
      .add_option("pmtiles-zoom", "Web Mercator zoom level for PMTiles export", zoom, true)
      .add_option("coord-scale", "Scale factor applied to input coordinates before EPSG:3857 tiling", coordScale)
      .add_option("prob-thres", "Minimum factor probability retained as a nullable property", probThreshold)
      .add_option("tile-buffer-px", "Tile buffer in screen pixels for buffered clipping", tileBufferPixels)
      .add_option("no-clipping", "Duplicate polygons across touched tiles without clipping them to tile boundaries", noClipping)
      .add_option("no-duplication", "Store each polygon intact in exactly one tile per zoom level", noDuplication)
      .add_option("clip-scale", "Integer scale used internally for polygon clipping", clipScale)
      .add_option("extent", "Vector tile extent", extent)
      .add_option("threads", "Number of encode threads", threads)
      .add_option("x-col", "Input column name for x coordinate", xColName)
      .add_option("y-col", "Input column name for y coordinate", yColName)
      .add_option("topk-col", "Input column name for top factor", topKColName)
      .add_option("topp-col", "Input column name for top factor probability", topPColName)
      .add_option("id-col", "Polygon ID column name in the factor-probability file (generic polygon mode)", idColName)
      .add_option("id-is-u32", "Parse polygon IDs directly as u32 instead of assigning IDs by geometry-file order", idIsU32)
      .add_option("keep-org-id", "Keep the original generic polygon ID as a string property column when generated u32 IDs are used", keepOrgId)
      .add_option("icol-id-geom", "0-based polygon ID column index in the geometry file", icolIdGeom)
      .add_option("icol-x-geom", "0-based x coordinate column index in the geometry file", icolXGeom)
      .add_option("icol-y-geom", "0-based y coordinate column index in the geometry file", icolYGeom)
      .add_option("icol-order-geom", "Optional 0-based vertex-order column index in the geometry file", icolOrderGeom);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (zoom < 0 || zoom > 31) {
        error("%s: --pmtiles-zoom must be in [0, 31]", __func__);
    }
    if (coordScale <= 0.0) {
        error("%s: --coord-scale must be positive", __func__);
    }
    if (extent <= 0) {
        error("%s: --extent must be positive", __func__);
    }
    if (tileBufferPixels < 0.0) {
        error("%s: --tile-buffer-px must be non-negative", __func__);
    }
    if (noClipping && noDuplication) {
        error("%s: --no-clipping cannot be used together with --no-duplication", __func__);
    }
    if (clipScale <= 0) {
        error("%s: --clip-scale must be positive", __func__);
    }
    const bool genericPolygonMode = !inGeom.empty();
    if (genericPolygonMode) {
        if (idColName.empty()) {
            error("%s: --id-col is required when --in-geom is provided", __func__);
        }
        if (idIsU32 && keepOrgId) {
            error("%s: --keep-org-id cannot be used together with --id-is-u32", __func__);
        }
    } else {
        if (hexGridDist <= 0.0) {
            error("%s: --hex-grid-dist must be positive in hex mode", __func__);
        }
        const double localCenterDist = (hexGridDist * coordScale * static_cast<double>(extent)) /
            (mlt_pmtiles::epsg3857_scale_factor(static_cast<uint8_t>(zoom)) * 4096.0);
        const double localHexEdge = localCenterDist / kSqrt3;
        constexpr double kDegenerateCenterDistThreshold = 2.0;
        if (localCenterDist < kDegenerateCenterDistThreshold) {
            const double zoomThreshold = std::log2(
                (kDegenerateCenterDistThreshold * 40075016.6856) /
                (hexGridDist * coordScale * static_cast<double>(extent)));
            warning("%s: --pmtiles-zoom %d may be too coarse for hex-grid-dist %.6g with extent %d "
                    "(tile-local center spacing %.3f, edge length %.3f). "
                    "Substantial hexagon degeneration starts below about z=%.2f for these parameters",
                __func__, zoom, hexGridDist, extent, localCenterDist, localHexEdge, zoomThreshold);
        }
    }

    std::filesystem::path outPath(outFile);
    if (outPath.has_parent_path()) {
        std::filesystem::create_directories(outPath.parent_path());
    }
    if (layerName.empty()) {
        layerName = basename(outFile, true);
    }

    UnitFactorResultReadOptions readOptions;
    readOptions.topKColName = topKColName;
    readOptions.topPColName = topPColName;
    if (genericPolygonMode) {
        readOptions.unitIdColName = idColName;
        readOptions.xColName.clear();
        readOptions.yColName.clear();
    } else {
        readOptions.xColName = xColName;
        readOptions.yColName = yColName;
    }
    UnitFactorResultReader reader(inTsv, readOptions);
    const UnitFactorResultHeader& layout = reader.header();
    const std::string originalIdColumnName =
        (genericPolygonMode && keepOrgId && !idIsU32) ? choose_original_id_column_name(idColName) : std::string();
    const mlt_pmtiles::FeatureTableSchema schema =
        build_polygon_schema(layerName, static_cast<uint32_t>(extent), layout.factorCols,
            !genericPolygonMode, !genericPolygonMode, originalIdColumnName);
    simple_polygon_pmtiles::SingleZoomPolygonWriterOptions writerOptions;
    writerOptions.zoom = static_cast<uint8_t>(zoom);
    writerOptions.extent = static_cast<uint32_t>(extent);
    writerOptions.coordScale = coordScale;
    writerOptions.tileBufferPixels = tileBufferPixels;
    writerOptions.clipScale = clipScale;
    writerOptions.threads = threads;
    writerOptions.boundaryMode = noDuplication ?
        simple_polygon_pmtiles::PolygonBoundaryMode::SingleTileNoDuplication :
        (noClipping ?
            simple_polygon_pmtiles::PolygonBoundaryMode::NoClippingDuplicate :
            simple_polygon_pmtiles::PolygonBoundaryMode::BufferClipDuplicate);

    std::map<TileKey, mlt_pmtiles::PolygonTileData> tileMap;
    simple_polygon_pmtiles::PolygonWriteSummary summary;
    HexGrid inputGrid;
    HexGrid scaledGrid;
    double scaledGridDist = 0.0;
    std::unordered_map<std::string, SimplePolygonWorldRecord> geometryMap;
    if (genericPolygonMode) {
        SimplePolygonGeomReadOptions geomOptions;
        geomOptions.icolId = icolIdGeom;
        geomOptions.icolX = icolXGeom;
        geomOptions.icolY = icolYGeom;
        geomOptions.icolOrder = icolOrderGeom;
        geometryMap = read_simple_polygon_geometry_map(inGeom, geomOptions, idIsU32);
    } else {
        inputGrid = HexGrid(hexGridDist / kSqrt3, true);
        scaledGrid = HexGrid((hexGridDist * coordScale) / kSqrt3, true);
        scaledGridDist = hexGridDist * coordScale;
    }
    uint64_t nRows = 0;
    uint64_t nSnapped = 0;
    UnitFactorResultRow row;
    while (reader.next(row)) {
        simple_polygon_pmtiles::PolygonFeatureProperties props;
        std::vector<std::pair<double, double>> ring;
        uint64_t featureId = 0;
        if (genericPolygonMode) {
            const std::string& polygonId = row.unitId;
            auto it = geometryMap.find(polygonId);
            if (it == geometryMap.end()) {
                error("%s: polygon ID %s from data file was not found in geometry file",
                    __func__, polygonId.c_str());
            }
            ring = it->second.outerRing;
            for (auto& pt : ring) {
                pt.first *= coordScale;
                pt.second *= coordScale;
            }
            featureId = static_cast<uint64_t>(it->second.assignedId);
            if (keepOrgId && !idIsU32) {
                props.stringValues = {polygonId};
            }
            props.intValues = {row.topK};
        } else {
            int32_t hexQ = 0;
            int32_t hexR = 0;
            inputGrid.cart_to_axial(hexQ, hexR, row.x, row.y);
            double snappedX = 0.0;
            double snappedY = 0.0;
            inputGrid.axial_to_cart(snappedX, snappedY, hexQ, hexR);
            if (std::abs(snappedX - row.x) > 1e-6 || std::abs(snappedY - row.y) > 1e-6) {
                ++nSnapped;
            }
            double centerX = 0.0;
            double centerY = 0.0;
            scaledGrid.axial_to_cart(centerX, centerY, hexQ, hexR);
            ring = build_pointy_hex_ring(centerX, centerY, scaledGridDist);
            featureId = pack_hex_polygon_id(hexQ, hexR);
            props.intValues = {hexQ, hexR, row.topK};
        }
        props.floatValues.reserve(1u + layout.factorCols.size());
        props.floatValues.push_back(row.topP);
        for (float prob : row.factorValues) {
            if (std::isfinite(prob) && prob >= static_cast<float>(probThreshold)) {
                props.floatValues.push_back(prob);
            } else {
                props.floatValues.push_back(std::nullopt);
            }
        }
        if (genericPolygonMode) {
            simple_polygon_pmtiles::append_simple_polygon_feature(
                tileMap, schema, ring, featureId, props, writerOptions, summary);
        } else {
            const int32_t hexQ = props.intValues[0].value_or(0);
            const int32_t hexR = props.intValues[1].value_or(0);
            double centerX = 0.0;
            double centerY = 0.0;
            scaledGrid.axial_to_cart(centerX, centerY, hexQ, hexR);
            simple_polygon_pmtiles::append_simple_polygon_feature(
                tileMap, schema, ring, featureId, props, writerOptions, summary,
                std::make_optional(std::make_pair(centerX, centerY)));
        }
        ++nRows;
    }

    std::vector<mlt_pmtiles::EncodedTilePayload> encodedTiles =
        simple_polygon_pmtiles::encode_polygon_tile_map(tileMap, schema, nullptr, writerOptions);
    nlohmann::json sourceMetadata;
    std::string sourceFamily;
    if (genericPolygonMode) {
        sourceFamily = "polygon_table";
        sourceMetadata = {
            {"coord_scale_applied", coordScale},
            {"data_id_column", idColName},
            {"geom_icol_id", icolIdGeom},
            {"geom_icol_x", icolXGeom},
            {"geom_icol_y", icolYGeom},
            {"geom_icol_order", icolOrderGeom},
        };
    } else {
        sourceFamily = "hexgrid";
        sourceMetadata = {
            {"hex_grid_dist_input", hexGridDist},
            {"hex_grid_dist_scaled", scaledGridDist},
            {"hex_orientation", "pointy"},
            {"coord_scale_applied", coordScale},
            {"input_columns", {
                {"x", xColName},
                {"y", yColName},
                {"topK", topKColName},
                {"topP", topPColName},
            }},
        };
    }
    nlohmann::json extraMetadata = simple_polygon_pmtiles::build_simple_polygon_metadata(
        sourceFamily, {}, writerOptions, sourceMetadata);
    extraMetadata["tile_buffer_rule"] =
        writerOptions.boundaryMode == simple_polygon_pmtiles::PolygonBoundaryMode::SingleTileNoDuplication ?
            "single_tile_no_duplication" :
        (writerOptions.boundaryMode == simple_polygon_pmtiles::PolygonBoundaryMode::NoClippingDuplicate ?
            "no_clipping_duplicate" : "tippecanoe_default_like");
    extraMetadata["tile_buffer_screen_px"] = tileBufferPixels;

    mlt_pmtiles::SingleLayerVectorPmtilesOptions archiveOptions;
    archiveOptions.schema = schema;
    archiveOptions.geometryType = mlt_pmtiles::VectorGeometryType::Polygon;
    archiveOptions.totalRecordCount = summary.featureCount;
    archiveOptions.coordScale = coordScale;
    archiveOptions.featureDictionarySize = 0;
    archiveOptions.outputZoom = static_cast<uint8_t>(zoom);
    archiveOptions.geoMinX = summary.geoMinX;
    archiveOptions.geoMinY = summary.geoMinY;
    archiveOptions.geoMaxX = summary.geoMaxX;
    archiveOptions.geoMaxY = summary.geoMaxY;
    archiveOptions.generator = genericPolygonMode ? "punkst hex2pmtiles --in-geom" : "punkst hex2pmtiles";
    archiveOptions.description = "Generated PMTiles by punkst for MLT polygons";
    archiveOptions.extraMetadata = extraMetadata;
    mlt_pmtiles::write_single_layer_vector_pmtiles_archive(outFile, std::move(encodedTiles), archiveOptions);

    if (genericPolygonMode) {
        notice("%s: wrote %" PRIu64 " source polygons into %s (%zu destination tiles)",
            __func__, nRows, outFile.c_str(), tileMap.size());
        if (!idIsU32) {
            const std::string idMapPath = std::filesystem::path(outFile).replace_extension("").string() + ".idmap.tsv";
            std::ofstream ofs(idMapPath);
            if (!ofs.good()) {
                error("%s: failed opening polygon ID map output %s", __func__, idMapPath.c_str());
            }
            ofs << "input_id\tassigned_u32\n";
            std::vector<std::pair<std::string, uint32_t>> mappings;
            mappings.reserve(geometryMap.size());
            for (const auto& kv : geometryMap) {
                mappings.emplace_back(kv.first, kv.second.assignedId);
            }
            std::sort(mappings.begin(), mappings.end(),
                [](const auto& lhs, const auto& rhs) {
                    return lhs.second < rhs.second;
                });
            for (const auto& kv : mappings) {
                ofs << kv.first << '\t' << kv.second << '\n';
            }
            if (!ofs.good()) {
                error("%s: failed writing polygon ID map to %s", __func__, idMapPath.c_str());
            }
            notice("%s: wrote polygon ID mapping to %s", __func__, idMapPath.c_str());
        }
    } else {
        notice("%s: wrote %" PRIu64 " source hexagons into %s (%zu destination tiles, %" PRIu64 " snapped centers)",
            __func__, nRows, outFile.c_str(), tileMap.size(), nSnapped);
    }
    return 0;
}
