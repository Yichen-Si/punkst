#pragma once

#include "pmtiles_utils.hpp"

#include <cstdint>
#include <string>

namespace tiles2mono {

enum class DisplayTransform {
    Linear,
    Log1p
};

struct Options {
    std::string dataFile;
    std::string indexFile;
    std::string rangeFile;
    std::string outFile;
    std::string tempBlobFile;
    int32_t icolX = 0;
    int32_t icolY = 1;
    int32_t icolCount = 3;
    int32_t minZoom = 7;
    int32_t maxZoom = 18;
    int32_t maxZoomFromRaw = -1;
    int32_t threads = 1;
    bool autoAdjust = true;
    double adjustQuantile = 0.99;
    DisplayTransform displayTransform = DisplayTransform::Linear;
    pm_raster::RasterBounds bounds;
};

DisplayTransform parse_display_transform(const std::string& value);
const char* display_transform_name(DisplayTransform transform);

void write_tiles2mono_pmtiles(const Options& options);

} // namespace tiles2mono
