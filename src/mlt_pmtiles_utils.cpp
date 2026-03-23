#include "mlt_pmtiles_utils.hpp"

#include "utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <zlib.h>

namespace mlt_pmtiles {

namespace {

constexpr double kEpsg3857Radius = 6378137.0;
constexpr double kEpsg3857Bound = 20037508.3428;

} // namespace

std::string gzip_compress(const std::string& data) {
    z_stream zs;
    std::memset(&zs, 0, sizeof(zs));
    if (deflateInit2(&zs, Z_DEFAULT_COMPRESSION, Z_DEFLATED, 15 | 16, 8, Z_DEFAULT_STRATEGY) != Z_OK) {
        error("%s: deflateInit2 failed", __func__);
    }
    zs.next_in = reinterpret_cast<Bytef*>(const_cast<char*>(data.data()));
    zs.avail_in = static_cast<uInt>(data.size());

    int ret = Z_OK;
    char outbuffer[32768];
    std::string out;
    do {
        zs.next_out = reinterpret_cast<Bytef*>(outbuffer);
        zs.avail_out = sizeof(outbuffer);
        ret = deflate(&zs, Z_FINISH);
        if (ret != Z_OK && ret != Z_STREAM_END) {
            deflateEnd(&zs);
            error("%s: deflate failed", __func__);
        }
        if (out.size() < zs.total_out) {
            out.append(outbuffer, zs.total_out - out.size());
        }
    } while (ret == Z_OK);

    deflateEnd(&zs);
    return out;
}

std::string encode_bool_rle(const std::vector<bool>& present) {
    const size_t n = present.size();
    const size_t numBytes = (n + 7u) / 8u;
    std::vector<uint8_t> packed(numBytes, 0);
    for (size_t i = 0; i < n; ++i) {
        if (present[i]) {
            packed[i / 8u] |= static_cast<uint8_t>(1u << (i % 8u));
        }
    }

    std::string out;
    size_t offset = 0;
    while (offset < numBytes) {
        const size_t chunk = std::min<size_t>(128, numBytes - offset);
        out.push_back(static_cast<char>(static_cast<uint8_t>(256 - chunk)));
        for (size_t i = 0; i < chunk; ++i) {
            out.push_back(static_cast<char>(packed[offset + i]));
        }
        offset += chunk;
    }
    return out;
}

void epsg3857_to_wgs84(double x, double y, double& lon, double& lat) {
    lon = (x / kEpsg3857Radius) * (180.0 / M_PI);
    lat = (2.0 * std::atan(std::exp(y / kEpsg3857Radius)) - M_PI / 2.0) * (180.0 / M_PI);
}

double epsg3857_scale_factor(uint8_t zoom) {
    return 2.0 * kEpsg3857Bound / static_cast<double>(uint64_t{1} << (zoom + 12));
}

void epsg3857_to_tilecoord(double x, double y, uint8_t zoom,
    int64_t& tileX, int64_t& tileY, double& localX, double& localY) {
    if (!std::isfinite(x)) x = 40000000.0;
    if (!std::isfinite(y)) y = 40000000.0;

    constexpr double tileSize = 256.0;
    const uint64_t numTiles = uint64_t{1} << zoom;
    tileX = static_cast<int64_t>((x + kEpsg3857Bound) / (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)));
    tileY = static_cast<int64_t>((kEpsg3857Bound - y) / (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)));

    const double tileOriginX = static_cast<double>(tileX) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)) - kEpsg3857Bound;
    const double tileOriginY = kEpsg3857Bound - static_cast<double>(tileY) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles));
    localX = (x - tileOriginX) / (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
    localY = (tileOriginY - y) / (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
}

void tilecoord_to_epsg3857(int64_t tileX, int64_t tileY,
    double localX, double localY, uint8_t zoom,
    double& x, double& y) {
    constexpr double tileSize = 256.0;
    const uint64_t numTiles = uint64_t{1} << zoom;
    const double tileOriginX = static_cast<double>(tileX) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles)) - kEpsg3857Bound;
    const double tileOriginY = kEpsg3857Bound - static_cast<double>(tileY) * (2.0 * kEpsg3857Bound / static_cast<double>(numTiles));
    x = tileOriginX + localX * (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
    y = tileOriginY - localY * (2.0 * kEpsg3857Bound / (static_cast<double>(numTiles) * tileSize));
}

} // namespace mlt_pmtiles
