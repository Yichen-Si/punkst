#include "punkst.h"

#include "tiles2mono.hpp"
#include "utils.h"

#include <exception>
#include <iostream>

int32_t cmdTiles2Mono(int32_t argc, char** argv) {
    std::string inPrefix;
    std::string displayTransform = "linear";
    bool noAutoAdjust = false;
    tiles2mono::Options options;

    ParamList pl;
    pl.add_option("in", "Input prefix (equivalent to --in-tsv <in>.tsv --in-index <in>.index)", inPrefix)
      .add_option("in-tsv", "Input tiled TSV from pts2tiles", options.dataFile)
      .add_option("in-data", "Input tiled TSV from pts2tiles", options.dataFile)
      .add_option("in-index", "Input tile index from pts2tiles", options.indexFile)
      .add_option("range", "Coordinate range TSV from pts2tiles", options.rangeFile)
      .add_option("icol-x", "0-based x coordinate column", options.icolX)
      .add_option("icol-y", "0-based y coordinate column", options.icolY)
      .add_option("icol-count", "0-based count column", options.icolCount)
      .add_option("min-zoom", "Minimum PMTiles zoom", options.minZoom)
      .add_option("max-zoom", "Maximum PMTiles zoom", options.maxZoom)
      .add_option("max-zoom-from-raw", "Parse raw data for zoom levels >= this value; derive lower zooms from parent layers", options.maxZoomFromRaw)
      .add_option("adjust-quantile", "Quantile for draw-xy-compatible auto-adjustment", options.adjustQuantile)
      .add_option("display-transform", "Display transform for mono raster intensity: linear or log1p", displayTransform)
      .add_option("no-auto-adjust", "Disable draw-xy-compatible intensity auto-adjustment", noAutoAdjust)
      .add_option("threads", "Number of worker threads", options.threads)
      .add_option("out", "Output mono raster PMTiles", options.outFile, true);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (!inPrefix.empty()) {
        options.dataFile = inPrefix + ".tsv";
        options.indexFile = inPrefix + ".index";
        if (options.rangeFile.empty()) {
            options.rangeFile = inPrefix + ".coord_range.tsv";
        }
    }
    if (options.dataFile.empty() || options.indexFile.empty()) {
        error("%s: provide --in or both --in-tsv/--in-data and --in-index", __func__);
    }
    if (noAutoAdjust) {
        options.autoAdjust = false;
    }
    options.displayTransform = tiles2mono::parse_display_transform(displayTransform);
    if (options.rangeFile.empty()) {
        std::string prefix = options.dataFile;
        constexpr const char* suffix = ".tsv";
        if (prefix.size() > 4 && prefix.substr(prefix.size() - 4) == suffix) {
            prefix.resize(prefix.size() - 4);
            options.rangeFile = prefix + ".coord_range.tsv";
        }
    }
    options.tempBlobFile = options.outFile + ".blob.tmp";
    tiles2mono::write_tiles2mono_pmtiles(options);
    return 0;
}
