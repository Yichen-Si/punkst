#include <cmath>

#include "pmtiles_pyramid.hpp"
#include "punkst.h"

int32_t cmdBuildPmtilesPyramid(int32_t argc, char** argv) {
    std::string inPrefix;
    std::string inData;
    std::string outPmtiles;
    bool pointMode = false;
    bool polygonMode = false;
    int32_t threads = 1;
    std::string polygonPriorityMode = "area";
    pmtiles_pyramid::BuildOptions options;

    ParamList pl;
    pl.add_option("in", "Input PMTiles file", inPrefix)
      .add_option("in-data", "Input PMTiles file", inData)
      .add_option("out", "Output PMTiles file", outPmtiles, true)
      .add_option("point", "Build a multi-zoom PMTiles pyramid for point-only MLT tiles", pointMode)
      .add_option("polygon", "Build a multi-zoom PMTiles pyramid for simple-polygon MLT tiles", polygonMode)
      .add_option("min-zoom", "Minimum zoom level to build", options.minZoom)
      .add_option("max-tile-bytes", "Maximum compressed tile bytes", options.maxTileBytes)
      .add_option("max-tile-features", "Maximum features per tile", options.maxTileFeatures)
      .add_option("scale-factor-compression", "Compression aggressiveness estimate", options.scaleFactorCompression)
      .add_option("polygon-priority", "Polygon retention priority: random or area", polygonPriorityMode)
      .add_option("polygon-id-col", "Hard override polygon ID property column name for generic polygon inputs", options.polygonIdColumn)
      .add_option("polygon-source", "Optional source polygon vertex table override for generic polygon inputs", options.polygonSourcePath)
      .add_option("icol-id", "0-based polygon source ID column index", options.polygonSourceIcolId)
      .add_option("icol-x", "0-based polygon source X column index", options.polygonSourceIcolX)
      .add_option("icol-y", "0-based polygon source Y column index", options.polygonSourceIcolY)
      .add_option("icol-order", "Optional 0-based polygon source vertex-order column index", options.polygonSourceIcolOrder)
      .add_option("polygon-source-coord-scale", "Override scale applied to --polygon-source coordinates before EPSG:3857 tiling (default: archive coord_scale metadata)", options.polygonSourceCoordScale)
      .add_option("tile-buffer-px", "Override polygon tile buffer in screen pixels for parent construction", options.tileBufferPixels)
      .add_option("no-clipping", "Duplicate polygons across touched tiles without clipping them to tile boundaries", options.polygonNoClipping)
      .add_option("no-duplication", "Store each polygon intact in exactly one tile per zoom level", options.polygonNoDuplication)
      .add_option("threads", "Number of threads to use", threads);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (pointMode == polygonMode) {
        error("%s: specify exactly one of --point or --polygon", __func__);
    }
    if (!inPrefix.empty()) {
        if (!inData.empty() && inData != inPrefix) {
            error("%s: --in and --in-data refer to different inputs", __func__);
        }
        inData = inPrefix;
    }
    if (inData.empty()) {
        error("%s: either --in or --in-data must be specified", __func__);
    }

    options.threads = threads;
    if (!std::isnan(options.polygonSourceCoordScale) &&
        !(options.polygonSourceCoordScale > 0.0)) {
        error("%s: --polygon-source-coord-scale must be positive", __func__);
    }
    if (!std::isnan(options.tileBufferPixels) &&
        !(options.tileBufferPixels >= 0.0)) {
        error("%s: --tile-buffer-px must be non-negative", __func__);
    }
    if (options.polygonNoClipping && options.polygonNoDuplication) {
        error("%s: --no-clipping cannot be used together with --no-duplication", __func__);
    }
    if (polygonPriorityMode == "random") {
        options.polygonPriorityMode = pmtiles_pyramid::PolygonPriorityMode::Random;
    } else if (polygonPriorityMode == "area") {
        options.polygonPriorityMode = pmtiles_pyramid::PolygonPriorityMode::Area;
    } else {
        error("%s: --polygon-priority must be one of random or area", __func__);
    }
    if (pointMode) {
        pmtiles_pyramid::build_point_pmtiles_pyramid(inData, outPmtiles, options);
    } else {
        pmtiles_pyramid::build_polygon_pmtiles_pyramid(inData, outPmtiles, options);
    }
    return 0;
}
