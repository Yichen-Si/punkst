#include <limits>

#include "punkst.h"
#include "tileoperator.hpp"

int32_t cmdExportPmtiles(int32_t argc, char** argv) {
    std::string inPrefix;
    std::string inData;
    std::string outPrefix;
    int32_t tileSize = -1;
    int32_t probDigits = 4;
    int32_t coordDigits = 2;
    std::string extractRegionGeoJSON;
    int64_t extractRegionScale = 10;
    float xmin = 0.0f;
    float xmax = -1.0f;
    float ymin = 0.0f;
    float ymax = -1.0f;
    float zmin = std::numeric_limits<float>::quiet_NaN();
    float zmax = std::numeric_limits<float>::quiet_NaN();

    TileOperator::ExportPmtilesOptions options;
    ParamList pl;
    pl.add_option("in", "Input PMTiles file", inPrefix)
      .add_option("in-data", "Input PMTiles file", inData)
      .add_option("out", "Output prefix", outPrefix, true)
      .add_option("tile-size", "Tile size in the exported TileOperator index", tileSize, true)
      .add_option("prob-digits", "Number of decimal digits to output for probabilities", probDigits)
      .add_option("coord-digits", "Number of decimal digits to output for coordinates", coordDigits)
      .add_option("extract-region-geojson", "Export only rows inside a GeoJSON Polygon/MultiPolygon region", extractRegionGeoJSON)
      .add_option("extract-region-scale", "Integer scale for GeoJSON region snapping", extractRegionScale)
      .add_option("xmin", "Minimum x coordinate for export filtering", xmin)
      .add_option("xmax", "Maximum x coordinate for export filtering", xmax)
      .add_option("ymin", "Minimum y coordinate for export filtering", ymin)
      .add_option("ymax", "Maximum y coordinate for export filtering", ymax)
      .add_option("zmin", "Minimum z coordinate for 3D export filtering ([zmin, zmax))", zmin)
      .add_option("zmax", "Maximum z coordinate for 3D export filtering ([zmin, zmax))", zmax);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
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

    options.tileSize = tileSize;
    options.probDigits = probDigits;
    options.coordDigits = coordDigits;
    options.geojsonFile = extractRegionGeoJSON;
    options.geojsonScale = extractRegionScale;
    options.xmin = xmin;
    options.xmax = xmax;
    options.ymin = ymin;
    options.ymax = ymax;
    options.zmin = zmin;
    options.zmax = zmax;
    TileOperator::exportPMTiles(inData, outPrefix, options);
    return 0;
}
