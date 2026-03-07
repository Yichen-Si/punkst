#include "punkst.h"
#include "region_query.hpp"
#include "utils.h"
#include "nlohmann/json.hpp"

#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

namespace {

using json = nlohmann::json;

struct MultiPolygonStats {
    std::size_t polygons = 0;
    std::size_t rings = 0;
    std::size_t points = 0;
    double xmin = std::numeric_limits<double>::infinity();
    double xmax = -std::numeric_limits<double>::infinity();
    double ymin = std::numeric_limits<double>::infinity();
    double ymax = -std::numeric_limits<double>::infinity();
    double firstX = 0.0;
    double firstY = 0.0;
    bool hasFirstPoint = false;
};

std::string get_type_string(const json &obj) {
    if (!obj.is_object() || !obj.contains("type") || !obj["type"].is_string()) {
        return "";
    }
    return obj["type"].get<std::string>();
}

bool validate_point(const json &pt, MultiPolygonStats &stats) {
    if (!pt.is_array() || pt.size() < 2 || !pt[0].is_number() || !pt[1].is_number()) {
        return false;
    }
    const double x = pt[0].get<double>();
    const double y = pt[1].get<double>();
    if (!stats.hasFirstPoint) {
        stats.firstX = x;
        stats.firstY = y;
        stats.hasFirstPoint = true;
    }
    stats.xmin = std::min(stats.xmin, x);
    stats.xmax = std::max(stats.xmax, x);
    stats.ymin = std::min(stats.ymin, y);
    stats.ymax = std::max(stats.ymax, y);
    ++stats.points;
    return true;
}

bool validate_ring(const json &ring, MultiPolygonStats &stats) {
    if (!ring.is_array() || ring.size() < 4) {
        return false;
    }
    MultiPolygonStats tmp = stats;
    for (std::size_t i = 0; i < ring.size(); ++i) {
        if (!validate_point(ring[i], tmp)) {
            return false;
        }
    }
    const double fx = ring.front()[0].get<double>();
    const double fy = ring.front()[1].get<double>();
    const double lx = ring.back()[0].get<double>();
    const double ly = ring.back()[1].get<double>();
    if (std::abs(fx - lx) > 1e-9 || std::abs(fy - ly) > 1e-9) {
        return false;
    }
    ++tmp.rings;
    stats = tmp;
    return true;
}

bool collect_polygon_stats(const json& polygon, MultiPolygonStats& stats) {
    if (!polygon.is_array() || polygon.empty()) {
        return false;
    }
    MultiPolygonStats tmp = stats;
    for (const auto& ring : polygon) {
        if (!validate_ring(ring, tmp)) {
            return false;
        }
    }
    ++tmp.polygons;
    stats = tmp;
    return true;
}

void collect_geometry_stats(const json& obj, MultiPolygonStats& stats) {
    if (!obj.is_object()) {
        return;
    }
    const std::string type = get_type_string(obj);
    if (type == "Polygon") {
        const auto coords_it = obj.find("coordinates");
        if (coords_it != obj.end()) {
            (void) collect_polygon_stats(*coords_it, stats);
        }
        return;
    }
    if (type == "MultiPolygon") {
        const auto coords_it = obj.find("coordinates");
        if (coords_it != obj.end() && coords_it->is_array()) {
            for (const auto& polygon : *coords_it) {
                (void) collect_polygon_stats(polygon, stats);
            }
        }
        return;
    }
    if (type == "Feature") {
        const auto geom_it = obj.find("geometry");
        if (geom_it != obj.end() && geom_it->is_object()) {
            collect_geometry_stats(*geom_it, stats);
        }
        return;
    }
    if (type == "FeatureCollection") {
        const auto features_it = obj.find("features");
        if (features_it != obj.end() && features_it->is_array()) {
            for (const auto& feature : *features_it) {
                collect_geometry_stats(feature, stats);
            }
        }
        return;
    }
    const auto geom_it = obj.find("geometry");
    if (geom_it != obj.end() && geom_it->is_object()) {
        collect_geometry_stats(*geom_it, stats);
    }
    const auto features_it = obj.find("features");
    if (features_it != obj.end() && features_it->is_array()) {
        for (const auto& feature : *features_it) {
            collect_geometry_stats(feature, stats);
        }
    }
}

} // namespace

int32_t test(int32_t argc, char** argv) {
    std::string inFile = "/net/wonderland/home/ycsi/tmp/test_geojson/test_region.geojson";
    int32_t tileSize = 32;
    int64_t scale = 10;

    ParamList pl;
    pl.add_option("input", "Input GeoJSON file containing a Polygon or MultiPolygon", inFile)
      .add_option("tile-size", "Tile size for prepared region tests", tileSize)
      .add_option("scale", "Integer coordinate scale for prepared region tests", scale);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    try {
        std::ifstream ifs(inFile);
        if (!ifs) {
            throw std::runtime_error("Failed to open input file: " + inFile);
        }

        json root;
        ifs >> root;

        MultiPolygonStats stats;
        collect_geometry_stats(root, stats);
        if (stats.polygons == 0) {
            throw std::runtime_error("No valid polygons found in the input JSON");
        }

        PreparedRegionMask2D region = loadPreparedRegionGeoJSON(inFile, tileSize, scale);
        if (region.empty()) {
            throw std::runtime_error("Prepared region is empty");
        }
        if (!stats.hasFirstPoint || !region.containsPoint(static_cast<float>(stats.firstX), static_cast<float>(stats.firstY))) {
            throw std::runtime_error("Prepared region does not contain the first polygon point");
        }
        if (region.containsPoint(static_cast<float>(stats.xmax + tileSize * 2.0),
                                 static_cast<float>(stats.ymax + tileSize * 2.0))) {
            throw std::runtime_error("Prepared region incorrectly contains a far-away point");
        }

        std::cout << "GeoJSON region parse test passed\n";
        std::cout << "Input: " << inFile << "\n";
        std::cout << "Polygons: " << stats.polygons
                  << ", Rings: " << stats.rings
                  << ", Points: " << stats.points << "\n";
        std::cout << "Bounding box: ["
                  << stats.xmin << ", " << stats.ymin << "] -> ["
                  << stats.xmax << ", " << stats.ymax << "]\n";
        std::cout << "Prepared components: " << region.union_paths.size() << "\n";
    } catch (const std::exception &ex) {
        std::cerr << "GeoJSON region parse test failed: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}
