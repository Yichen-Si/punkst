#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace mlt_pmtiles {

std::string gzip_compress(const std::string& data);
std::string encode_bool_rle(const std::vector<bool>& present);

void epsg3857_to_wgs84(double x, double y, double& lon, double& lat);
double epsg3857_scale_factor(uint8_t zoom);
void epsg3857_to_tilecoord(double x, double y, uint8_t zoom,
    int64_t& tileX, int64_t& tileY, double& localX, double& localY);
void tilecoord_to_epsg3857(int64_t tileX, int64_t tileY,
    double localX, double localY, uint8_t zoom,
    double& x, double& y);

} // namespace mlt_pmtiles
