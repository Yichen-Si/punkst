#include "punkst.h"

#include "image_pmtiles.hpp"
#include "utils.h"

#include <array>
#include <exception>
#include <iostream>
#include <sstream>
#include <vector>

namespace {

std::array<double, 9> parse_transform_string(const std::string& raw) {
    std::string text = raw;
    for (char& c : text) {
        if (c == ',' || c == ';') {
            c = ' ';
        }
    }
    std::istringstream iss(text);
    std::array<double, 9> out{};
    for (size_t i = 0; i < out.size(); ++i) {
        if (!(iss >> out[i])) {
            error("%s: --transform must contain 9 numeric values", __func__);
        }
    }
    double extra = 0.0;
    if (iss >> extra) {
        error("%s: --transform must contain exactly 9 numeric values", __func__);
    }
    return out;
}

} // namespace

int32_t cmdImage2Pmtiles(int32_t argc, char** argv) {
    image_pmtiles::Options options;
    std::string transform;
    std::vector<double> grayPercentiles;

    ParamList pl;
    pl.add_option("in-image", "Input PNG or supported tiled TIFF", options.inImage, true)
      .add_option("out-prefix", "Output prefix for .pmtiles and _assets.json", options.outPrefix, true)
      .add_option("asset-json", "Output asset JSON path (default: <out-prefix>_assets.json)", options.assetJson)
      .add_option("id", "Image/basemap ID", options.id, true)
      .add_option("min-zoom", "Minimum PMTiles zoom", options.minZoom)
      .add_option("max-zoom", "Maximum PMTiles zoom", options.maxZoom)
      .add_option("microns-per-pixel", "Pixel size in punkst coordinate units", options.micronsPerPixel)
      .add_option("offset-x-um", "X offset for scale-only transform", options.offsetXUm)
      .add_option("offset-y-um", "Y offset for scale-only transform", options.offsetYUm)
      .add_option("transform", "3x3 pixel-to-micron transform as 9 comma/space-separated values", transform)
      .add_option("gray-percentiles", "LOW HIGH percentile bounds for 16-bit grayscale TIFF scaling", grayPercentiles)
      .add_option("gray-sample-fraction", "Fraction of TIFF tiles sampled for 16-bit grayscale scaling", options.graySampleFraction)
      .add_option("gray-sample-tiles", "Minimum number of TIFF tiles sampled for 16-bit grayscale scaling; 0 scans all tiles", options.graySampleTiles)
      .add_option("gray-sample-seed", "Random seed for 16-bit grayscale TIFF tile sampling", options.graySampleSeed)
      .add_option("tile-cache-mb", "Optional decoded TIFF tile cache guard in MB; 0 disables the guard", options.tileCacheMb)
      .add_option("tiff-source-level", "TIFF source level: auto, base, or numeric IFD/SubIFD level", options.tiffSourceLevel)
      .add_option("overwrite", "Overwrite existing outputs", options.overwrite);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (!transform.empty()) {
        options.hasTransform = true;
        options.transform = parse_transform_string(transform);
    }
    if (!grayPercentiles.empty()) {
        if (grayPercentiles.size() != 2) {
            error("%s: --gray-percentiles requires exactly two values", __func__);
        }
        options.grayLowPercentile = grayPercentiles[0];
        options.grayHighPercentile = grayPercentiles[1];
    }
    image_pmtiles::write_image_pmtiles(options);
    return 0;
}
