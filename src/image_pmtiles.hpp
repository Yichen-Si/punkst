#pragma once

#include <array>
#include <cstdint>
#include <filesystem>
#include <string>

namespace image_pmtiles {

struct Options {
    std::string id;
    std::filesystem::path inImage;
    std::filesystem::path outPrefix;
    std::filesystem::path assetJson;
    int32_t minZoom = 7;
    int32_t maxZoom = 18;
    double micronsPerPixel = 0.0;
    double offsetXUm = 0.0;
    double offsetYUm = 0.0;
    bool hasMicronsPerPixel = false;
    bool hasOffsetXUm = false;
    bool hasOffsetYUm = false;
    bool hasTransform = false;
    bool hasPixZero = false;
    bool hasPixMax = false;
    std::array<double, 2> pixZero{0.0, 0.0};
    std::array<double, 2> pixMax{0.0, 0.0};
    double grayLowPercentile = 1.0;
    double grayHighPercentile = 99.0;
    double graySampleFraction = 0.05;
    int32_t graySampleTiles = 10;
    uint32_t graySampleSeed = 1;
    int32_t tileCacheMb = 0;
    std::string tiffSourceLevel = "auto";
    std::array<double, 9> transform{
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0};
    bool overwrite = false;
};

struct Asset {
    std::string id;
    std::filesystem::path pmtilesPath;
    std::filesystem::path assetJson;
};

void validate_options(const Options& options);
Asset write_image_pmtiles(const Options& options);

} // namespace image_pmtiles
