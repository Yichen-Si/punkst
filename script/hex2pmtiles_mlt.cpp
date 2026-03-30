#include "punkst.h"

#include "dataunits.hpp"
#include "hexgrid.h"
#include "pmtiles_utils.hpp"
#include "simple_polygon_pmtiles.hpp"

#include <cinttypes>
#include <cmath>
#include <filesystem>

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

mlt_pmtiles::FeatureTableSchema build_hex_schema(const std::string& layerName, uint32_t extent, const std::vector<std::pair<int32_t, int32_t>>& factorCols) {
    mlt_pmtiles::FeatureTableSchema schema;
    schema.layerName = layerName;
    schema.extent = extent;
    schema.columns.push_back({"hex_q", mlt_pmtiles::ScalarType::INT_32, false});
    schema.columns.push_back({"hex_r", mlt_pmtiles::ScalarType::INT_32, false});
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
    std::string outFile;
    std::string layerName;
    std::string xColName = "x";
    std::string yColName = "y";
    std::string topKColName = "topK";
    std::string topPColName = "topP";
    double hexGridDist = -1.0;
    double coordScale = 1.0;
    double probThreshold = 1e-4;
    double tileBufferPixels = 5.0;
    int64_t clipScale = 1024;
    int32_t zoom = -1;
    int32_t extent = 4096;
    int32_t threads = 1;

    ParamList pl;
    pl.add_option("in-tsv", "Input TSV(.gz)", inTsv, true)
      .add_option("out", "Output single-zoom PMTiles file", outFile, true)
      .add_option("layer-name", "Optional PMTiles layer name (default: basename of --out)", layerName)
      .add_option("hex-grid-dist", "Distance between adjacent hex centers in the input coordinate system", hexGridDist, true)
      .add_option("pmtiles-zoom", "Web Mercator zoom level for PMTiles export", zoom, true)
      .add_option("coord-scale", "Scale factor applied to input coordinates before EPSG:3857 tiling", coordScale)
      .add_option("prob-thres", "Minimum factor probability retained as a nullable property", probThreshold)
      .add_option("tile-buffer-px", "Tile buffer in screen pixels for buffered clipping", tileBufferPixels)
      .add_option("clip-scale", "Integer scale used internally for polygon clipping", clipScale)
      .add_option("extent", "Vector tile extent", extent)
      .add_option("threads", "Number of encode threads", threads)
      .add_option("x-col", "Input column name for x coordinate", xColName)
      .add_option("y-col", "Input column name for y coordinate", yColName)
      .add_option("topk-col", "Input column name for top factor", topKColName)
      .add_option("topp-col", "Input column name for top factor probability", topPColName);

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
    if (hexGridDist <= 0.0) {
        error("%s: --hex-grid-dist must be positive", __func__);
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
    if (clipScale <= 0) {
        error("%s: --clip-scale must be positive", __func__);
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

    std::filesystem::path outPath(outFile);
    if (outPath.has_parent_path()) {
        std::filesystem::create_directories(outPath.parent_path());
    }
    if (layerName.empty()) {
        layerName = basename(outFile, true);
    }

    UnitFactorResultReader reader(inTsv, xColName, yColName, topKColName, topPColName);
    const UnitFactorResultHeader& layout = reader.header();

    const mlt_pmtiles::FeatureTableSchema schema =
        build_hex_schema(layerName, static_cast<uint32_t>(extent), layout.factorCols);
    simple_polygon_pmtiles::SingleZoomPolygonWriterOptions writerOptions;
    writerOptions.zoom = static_cast<uint8_t>(zoom);
    writerOptions.extent = static_cast<uint32_t>(extent);
    writerOptions.coordScale = coordScale;
    writerOptions.tileBufferPixels = tileBufferPixels;
    writerOptions.clipScale = clipScale;
    writerOptions.threads = threads;

    std::map<TileKey, mlt_pmtiles::PolygonTileData> tileMap;
    simple_polygon_pmtiles::PolygonWriteSummary summary;
    HexGrid inputGrid(hexGridDist / kSqrt3, true);
    HexGrid scaledGrid((hexGridDist * coordScale) / kSqrt3, true);
    const double scaledGridDist = hexGridDist * coordScale;
    uint64_t nRows = 0;
    uint64_t nSnapped = 0;
    UnitFactorResultRow row;
    while (reader.next(row)) {
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
        const auto ring = build_pointy_hex_ring(centerX, centerY, scaledGridDist);

        simple_polygon_pmtiles::PolygonFeatureProperties props;
        props.intValues = {hexQ, hexR, row.topK};
        props.floatValues.reserve(1u + layout.factorCols.size());
        props.floatValues.push_back(row.topP);
        for (float prob : row.factorValues) {
            if (std::isfinite(prob) && prob >= static_cast<float>(probThreshold)) {
                props.floatValues.push_back(prob);
            } else {
                props.floatValues.push_back(std::nullopt);
            }
        }
        simple_polygon_pmtiles::append_simple_polygon_feature(tileMap, schema, ring, props, writerOptions, summary);
        ++nRows;
    }

    std::vector<mlt_pmtiles::EncodedTilePayload> encodedTiles =
        simple_polygon_pmtiles::encode_polygon_tile_map(tileMap, schema, nullptr, writerOptions);
    const nlohmann::json sourceMetadata = {
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
    nlohmann::json extraMetadata = simple_polygon_pmtiles::build_simple_polygon_metadata(
        "hexgrid", {"hex_q", "hex_r"}, writerOptions, sourceMetadata);
    extraMetadata["tile_buffer_rule"] = "tippecanoe_default_like";
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
    archiveOptions.generator = "punkst hex2pmtiles-mlt";
    archiveOptions.description = "Generated PMTiles by punkst for MLT polygons";
    archiveOptions.extraMetadata = extraMetadata;
    mlt_pmtiles::write_single_layer_vector_pmtiles_archive(outFile, std::move(encodedTiles), archiveOptions);

    notice("%s: wrote %" PRIu64 " source hexagons into %s (%zu destination tiles, %" PRIu64 " snapped centers)",
        __func__, nRows, outFile.c_str(), tileMap.size(), nSnapped);
    return 0;
}
