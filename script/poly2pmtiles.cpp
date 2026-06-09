#include "punkst.h"

#include "factor_polygon_pmtiles.hpp"
#include "utils.h"

int32_t cmdPoly2Pmtiles(int32_t argc, char** argv) {
    factor_polygon_pmtiles::Options opts;

    ParamList pl;
    pl.add_option("in-tsv", "Input statistics TSV", opts.inTsv, true)
      .add_option("in-geom", "Optional polygon geometry TSV/CSV/GeoJSON/JSON", opts.inGeom)
      .add_option("out", "Output single-zoom PMTiles file", opts.outFile, true)
      .add_option("format", "Tile encoding format: MLT or MVT", opts.format)
      .add_option("layer-name", "Optional PMTiles layer name (default: basename of --out)", opts.layerName)
      .add_option("hex-grid-dist", "Distance between adjacent hex centers in input coordinates (hex mode)", opts.hexGridDist)
      .add_option("pmtiles-zoom", "Web Mercator zoom level for PMTiles export", opts.zoom, true)
      .add_option("coord-scale", "Scale factor applied to input coordinates before tiling", opts.coordScale)
      .add_option("prob-thres", "Minimum probability retained for nullable factor properties", opts.probThreshold)
      .add_option("tile-buffer-px", "Tile buffer in screen pixels for clipped polygon output", opts.tileBufferPixels)
      .add_option("no-clipping", "Duplicate polygons across touched tiles without clipping", opts.noClipping)
      .add_option("no-duplication", "Store each polygon intact in exactly one tile", opts.noDuplication)
      .add_option("clip-scale", "Integer scale used internally for polygon clipping", opts.clipScale)
      .add_option("extent", "Vector tile extent", opts.extent)
      .add_option("threads", "Number of encode threads", opts.threads)
      .add_option("x-col", "Input column name for x coordinate", opts.xColName)
      .add_option("y-col", "Input column name for y coordinate", opts.yColName)
      .add_option("topk-col", "Optional input column name for top factor", opts.topKColName)
      .add_option("topp-col", "Optional input column name for top factor probability", opts.topPColName)
      .add_option("id-col", "Polygon ID column name in the statistics TSV (generic polygon mode)", opts.idColName)
      .add_option("top-k", "Number of top factors to keep from dense factor columns", opts.topK)
      .add_option("geom-format", "Geometry format: auto, table, geojson, or json", opts.geomFormat)
      .add_option("geom-id-prop", "Polygon ID property in GeoJSON/JSON features", opts.geomIdProp)
      .add_option("g-icol-id", "0-based polygon ID column in table geometry", opts.geomIdCol)
      .add_option("g-icol-x", "0-based x coordinate column in table geometry", opts.geomXCol)
      .add_option("g-icol-y", "0-based y coordinate column in table geometry", opts.geomYCol)
      .add_option("g-icol-order", "Optional 0-based vertex-order column in table geometry", opts.geomOrderCol)
      .add_option("id-is-u32", "Parse polygon IDs directly as u32 feature IDs where possible", opts.idIsU32)
      .add_option("keep-org-id", "Keep original polygon ID as a string property in generic mode", opts.keepOrgId)
      .add_option("cartoscope-boundary", "Emit CartoScope boundary schema", opts.cartoscopeBoundary)
      .add_option("out-sidecar-tsv", "Optional polygon ID / feature ID / center sidecar TSV", opts.outSidecar);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    return factor_polygon_pmtiles::write(opts);
}
