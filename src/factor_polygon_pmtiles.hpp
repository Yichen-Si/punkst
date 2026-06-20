#pragma once

#include <cstdint>
#include <string>

namespace factor_polygon_pmtiles {

struct Options {
    std::string inTsv;
    std::string inGeom;
    std::string outFile;
    std::string layerName;
    std::string format = "MLT";
    std::string xColName = "x";
    std::string yColName = "y";
    std::string topKColName = "topK";
    std::string topPColName = "topP";
    std::string idColName;
    std::string geomFormat = "auto";
    std::string geomIdProp = "cell_id";
    std::string outSidecar;
    int32_t geomIdCol = 0;
    int32_t geomXCol = 1;
    int32_t geomYCol = 2;
    int32_t geomOrderCol = -1;
    int32_t factorColBegin = -1;
    int32_t factorColEnd = -1;
    int32_t topK = 3;
    int32_t zoom = -1;
    int32_t extent = 4096;
    int32_t threads = 1;
    double hexGridDist = -1.0;
    double coordScale = 1.0;
    double probThreshold = 1e-4;
    double tileBufferPixels = 5.0;
    int64_t clipScale = 1024;
    bool noClipping = false;
    bool noDuplication = false;
    bool idIsU32 = false;
    bool keepOrgId = false;
    bool cartoscopeBoundary = false;
};

int write(const Options& options);

} // namespace factor_polygon_pmtiles
