#pragma once

#include "json.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace mlt_pmtiles {

struct EncodedTilePayload {
    uint64_t tileId = 0;
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t featureCount = 0;
    std::string compressedData;
};

struct ArchiveOptions {
    uint8_t tileType = 0x06;
    uint8_t minZoom = 0;
    uint8_t maxZoom = 0;
    uint8_t centerZoom = 0;
    bool clustered = true;
    bool hasGeographicBounds = false;
    int32_t minLonE7 = 0;
    int32_t minLatE7 = 0;
    int32_t maxLonE7 = 0;
    int32_t maxLatE7 = 0;
    int32_t centerLonE7 = 0;
    int32_t centerLatE7 = 0;
    nlohmann::json metadata;
};

void write_pmtiles_archive(const std::string& outFile,
    std::vector<EncodedTilePayload> tiles,
    const ArchiveOptions& options);

} // namespace mlt_pmtiles
