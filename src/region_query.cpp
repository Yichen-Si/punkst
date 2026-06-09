#include "region_query.hpp"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>

#include "json.hpp"
#include "error.hpp"
#include "utils.h"
#include "utils_sys.hpp"

namespace {

using json = nlohmann::json;
using Clipper2Lib::Difference;
using Clipper2Lib::FillRule;
using Clipper2Lib::InflatePaths;
using Clipper2Lib::IsPositive;
using Clipper2Lib::JoinType;
using Clipper2Lib::Path64;
using Clipper2Lib::Paths64;
using Clipper2Lib::Point64;
using Clipper2Lib::PointInPolygon;
using Clipper2Lib::PointInPolygonResult;
using Clipper2Lib::Rect64;
using Clipper2Lib::RectClip;
using Clipper2Lib::SegmentsIntersect;
using Clipper2Lib::SimplifyPaths;
using Clipper2Lib::TrimCollinear;
using Clipper2Lib::Union;
using Clipper2Lib::EndType;

constexpr int64_t kDefaultScale = 10;

int64_t floor_div64(int64_t a, int64_t b) {
    if (b <= 0) {
        throw std::runtime_error("floor_div64 requires positive divisor");
    }
    int64_t q = a / b;
    int64_t r = a % b;
    if (r != 0 && ((r > 0) != (b > 0))) {
        --q;
    }
    return q;
}

bool point_eq(const Point64& a, const Point64& b) {
    return a.x == b.x && a.y == b.y;
}

bool point_in_box_inclusive(const Point64& p, const Rect64& box) {
    return p.x >= box.left && p.x <= box.right &&
           p.y >= box.top && p.y <= box.bottom;
}

bool point_on_segment(const Point64& p, const Point64& a, const Point64& b) {
    const int64_t cross = (p.x - a.x) * (b.y - a.y) - (p.y - a.y) * (b.x - a.x);
    if (cross != 0) {
        return false;
    }
    const int64_t minx = std::min(a.x, b.x);
    const int64_t maxx = std::max(a.x, b.x);
    const int64_t miny = std::min(a.y, b.y);
    const int64_t maxy = std::max(a.y, b.y);
    return p.x >= minx && p.x <= maxx && p.y >= miny && p.y <= maxy;
}

bool segment_touches_rect(const Point64& a, const Point64& b, const Rect64& rect) {
    if (point_in_box_inclusive(a, rect) || point_in_box_inclusive(b, rect)) {
        return true;
    }
    const Point64 tl(rect.left, rect.top);
    const Point64 tr(rect.right, rect.top);
    const Point64 br(rect.right, rect.bottom);
    const Point64 bl(rect.left, rect.bottom);
    const Point64 rect_pts[4] = {tl, tr, br, bl};
    for (size_t i = 0; i < 4; ++i) {
        const Point64& p0 = rect_pts[i];
        const Point64& p1 = rect_pts[(i + 1) % 4];
        if (SegmentsIntersect(a, b, p0, p1, true)) {
            return true;
        }
        if (point_on_segment(p0, a, b) || point_on_segment(p1, a, b)) {
            return true;
        }
    }
    return false;
}

bool path_touches_rect(const Path64& path, const Rect64& rect) {
    if (path.size() < 2) {
        return false;
    }
    for (size_t i = 0; i < path.size(); ++i) {
        const Point64& p0 = path[i];
        const Point64& p1 = path[(i + 1) % path.size()];
        if (segment_touches_rect(p0, p1, rect)) {
            return true;
        }
    }
    return false;
}

RegionBox64 path_bbox(const Path64& path) {
    RegionBox64 box;
    box.xmin = std::numeric_limits<int64_t>::max();
    box.ymin = std::numeric_limits<int64_t>::max();
    box.xmax = std::numeric_limits<int64_t>::lowest();
    box.ymax = std::numeric_limits<int64_t>::lowest();
    for (const auto& pt : path) {
        box.xmin = std::min(box.xmin, pt.x);
        box.ymin = std::min(box.ymin, pt.y);
        box.xmax = std::max(box.xmax, pt.x);
        box.ymax = std::max(box.ymax, pt.y);
    }
    return box;
}

bool boxes_overlap(const RegionBox64& box, const Rect64& rect) {
    return box.valid() &&
           box.xmin <= rect.right && box.xmax >= rect.left &&
           box.ymin <= rect.bottom && box.ymax >= rect.top;
}

Path64 normalize_ring(const json& ring, int64_t scale) {
    if (!ring.is_array() || ring.size() < 4) {
        return {};
    }
    Path64 path;
    path.reserve(ring.size());
    for (const auto& pt : ring) {
        if (!pt.is_array() || pt.size() < 2 || !pt[0].is_number() || !pt[1].is_number()) {
            return {};
        }
        const double x = pt[0].get<double>();
        const double y = pt[1].get<double>();
        path.emplace_back(static_cast<int64_t>(std::llround(x * static_cast<double>(scale))),
                          static_cast<int64_t>(std::llround(y * static_cast<double>(scale))));
    }
    while (path.size() > 1 && point_eq(path.front(), path.back())) {
        path.pop_back();
    }
    Path64 deduped;
    deduped.reserve(path.size());
    for (const auto& pt : path) {
        if (!deduped.empty() && point_eq(deduped.back(), pt)) {
            continue;
        }
        deduped.push_back(pt);
    }
    if (deduped.size() < 3) {
        return {};
    }
    deduped = TrimCollinear(deduped, false);
    if (deduped.size() < 3) {
        return {};
    }
    return deduped;
}

bool has_self_intersection(const Path64& path) {
    if (path.size() < 4) {
        return false;
    }
    const size_t n = path.size();
    for (size_t i = 0; i < n; ++i) {
        const Point64& a0 = path[i];
        const Point64& a1 = path[(i + 1) % n];
        for (size_t j = i + 1; j < n; ++j) {
            if (j == i) {
                continue;
            }
            if (j == i + 1 || (i == 0 && j == n - 1)) {
                continue;
            }
            const Point64& b0 = path[j];
            const Point64& b1 = path[(j + 1) % n];
            if (point_eq(a0, b0) || point_eq(a0, b1) || point_eq(a1, b0) || point_eq(a1, b1)) {
                return true;
            }
            if (SegmentsIntersect(a0, a1, b0, b1, true)) {
                return true;
            }
        }
    }
    return false;
}

bool append_polygon_rings(Paths64& rings, const json& polygon, int64_t scale,
                          bool rejectSelfIntersections = true) {
    if (!polygon.is_array() || polygon.empty()) {
        return false;
    }
    Paths64 polygon_rings;
    for (const auto& ring : polygon) {
        Path64 path = normalize_ring(ring, scale);
        if (path.empty()) {
            return false;
        }
        if (rejectSelfIntersections && has_self_intersection(path)) {
            return false;
        }
        if (std::llround(std::abs(Clipper2Lib::Area(path))) == 0) {
            return false;
        }
        // Standalone polygon rings are treated as filled regions regardless of
        // input winding. Normalize them before union so overlapping polygons
        // never cancel each other under FillRule::NonZero.
        if (!IsPositive(path)) {
            std::reverse(path.begin(), path.end());
        }
        polygon_rings.push_back(std::move(path));
    }
    if (polygon_rings.empty()) {
        return false;
    }
    rings.insert(rings.end(),
                 std::make_move_iterator(polygon_rings.begin()),
                 std::make_move_iterator(polygon_rings.end()));
    return true;
}

Paths64 repair_region_paths_with_clipper(const Paths64& input_rings,
                                         const std::string& featureId) {
    if (input_rings.empty()) {
        return {};
    }
    try {
        Paths64 out = Union(input_rings, FillRule::NonZero);
        if (!out.empty()) {
            return out;
        }
    } catch (const std::exception& e) {
        warning("%s: Clipper union failed for GeoJSON feature '%s': %s",
            __func__, featureId.c_str(), e.what());
    } catch (...) {
        warning("%s: Clipper union failed for GeoJSON feature '%s'",
            __func__, featureId.c_str());
    }

    try {
        Paths64 simplified = SimplifyPaths(input_rings, 1.0, true);
        Paths64 cleaned;
        cleaned.reserve(simplified.size());
        for (auto& path : simplified) {
            path = TrimCollinear(path, false);
            if (path.size() < 3) {
                continue;
            }
            if (std::llround(std::abs(Clipper2Lib::Area(path))) == 0) {
                continue;
            }
            if (!IsPositive(path)) {
                std::reverse(path.begin(), path.end());
            }
            cleaned.push_back(std::move(path));
        }
        if (!cleaned.empty()) {
            Paths64 out = Union(cleaned, FillRule::NonZero);
            if (!out.empty()) {
                warning("%s: Repaired GeoJSON feature '%s' using Clipper simplification",
                    __func__, featureId.c_str());
                return out;
            }
        }
    } catch (const std::exception& e) {
        warning("%s: Clipper simplification repair failed for GeoJSON feature '%s': %s",
            __func__, featureId.c_str(), e.what());
    } catch (...) {
        warning("%s: Clipper simplification repair failed for GeoJSON feature '%s'",
            __func__, featureId.c_str());
    }

    try {
        Paths64 out = Union(input_rings, FillRule::EvenOdd);
        if (!out.empty()) {
            warning("%s: Repaired GeoJSON feature '%s' using even-odd fill",
                __func__, featureId.c_str());
            return out;
        }
    } catch (const std::exception& e) {
        warning("%s: Clipper even-odd repair failed for GeoJSON feature '%s': %s",
            __func__, featureId.c_str(), e.what());
    } catch (...) {
        warning("%s: Clipper even-odd repair failed for GeoJSON feature '%s'",
            __func__, featureId.c_str());
    }
    return {};
}

void append_geometry_rings(Paths64& rings, const json& obj, int64_t scale,
                           bool rejectSelfIntersections = true) {
    if (!obj.is_object()) {
        return;
    }
    const auto type_it = obj.find("type");
    if (type_it == obj.end() || !type_it->is_string()) {
        const auto geom_it = obj.find("geometry");
        if (geom_it != obj.end() && geom_it->is_object()) {
            append_geometry_rings(rings, *geom_it, scale, rejectSelfIntersections);
        }
        const auto features_it = obj.find("features");
        if (features_it != obj.end() && features_it->is_array()) {
            for (const auto& feature : *features_it) {
                append_geometry_rings(rings, feature, scale, rejectSelfIntersections);
            }
        }
        return;
    }
    const std::string type = type_it->get<std::string>();
    if (type == "Polygon") {
        const auto coords_it = obj.find("coordinates");
        if (coords_it != obj.end()) {
            (void) append_polygon_rings(rings, *coords_it, scale, rejectSelfIntersections);
        }
        return;
    }
    if (type == "MultiPolygon") {
        const auto coords_it = obj.find("coordinates");
        if (coords_it == obj.end() || !coords_it->is_array()) {
            return;
        }
        for (const auto& polygon : *coords_it) {
            (void) append_polygon_rings(rings, polygon, scale, rejectSelfIntersections);
        }
        return;
    }
    if (type == "Feature") {
        const auto geom_it = obj.find("geometry");
        if (geom_it != obj.end() && geom_it->is_object()) {
            append_geometry_rings(rings, *geom_it, scale, rejectSelfIntersections);
        }
        return;
    }
    if (type == "FeatureCollection") {
        const auto features_it = obj.find("features");
        if (features_it == obj.end() || !features_it->is_array()) {
            return;
        }
        for (const auto& feature : *features_it) {
            append_geometry_rings(rings, feature, scale, rejectSelfIntersections);
        }
    }
}

TileKey point_to_tile(const Point64& pt, int64_t tileSizeScaled) {
    TileKey tile;
    tile.col = static_cast<int32_t>(floor_div64(pt.x, tileSizeScaled));
    tile.row = static_cast<int32_t>(floor_div64(pt.y, tileSizeScaled));
    return tile;
}

Rect64 tile_to_scaled_rect(const TileKey& tile, int64_t tileSizeScaled) {
    const int64_t left = static_cast<int64_t>(tile.col) * tileSizeScaled;
    const int64_t top = static_cast<int64_t>(tile.row) * tileSizeScaled;
    return Rect64(left, top, left + tileSizeScaled, top + tileSizeScaled);
}

std::vector<uint32_t> collect_candidates(
    const PreparedRegionMask2D& region, const TileKey& tile) {
    const auto it = region.tile_bins.find(tile);
    if (it == region.tile_bins.end()) {
        return {};
    }
    return it->second;
}

bool point_in_region_paths(const Point64& pt, const Paths64& paths, const std::vector<uint32_t>& ids) {
    int winding = 0;
    for (uint32_t idx : ids) {
        if (idx >= paths.size()) {
            continue;
        }
        const auto pip = PointInPolygon(pt, paths[idx]);
        if (pip == PointInPolygonResult::IsOn) {
            return true;
        }
        if (pip == PointInPolygonResult::IsInside) {
            winding += IsPositive(paths[idx]) ? 1 : -1;
        }
    }
    return winding != 0;
}

} // namespace

PreparedRegionMask2D prepareRegionFromPaths(const Paths64& paths,
                                        int32_t tileSize,
                                        int64_t scale) {
    if (tileSize <= 0) {
        throw std::runtime_error("tileSize must be positive");
    }
    if (scale <= 0) {
        scale = kDefaultScale;
    }
    if (paths.empty()) {
        PreparedRegionMask2D emptyRegion;
        emptyRegion.scale = scale;
        emptyRegion.tileSize = tileSize;
        return emptyRegion;
    }

    PreparedRegionMask2D region;
    region.scale = scale;
    region.tileSize = tileSize;
    region.union_paths = paths;

    const int64_t tileSizeScaled = static_cast<int64_t>(tileSize) * scale;
    int64_t global_xmin = std::numeric_limits<int64_t>::max();
    int64_t global_ymin = std::numeric_limits<int64_t>::max();
    int64_t global_xmax = std::numeric_limits<int64_t>::lowest();
    int64_t global_ymax = std::numeric_limits<int64_t>::lowest();

    region.comp_bbox.reserve(region.union_paths.size());
    for (size_t i = 0; i < region.union_paths.size(); ++i) {
        const RegionBox64 box = path_bbox(region.union_paths[i]);
        region.comp_bbox.push_back(box);
        global_xmin = std::min(global_xmin, box.xmin);
        global_ymin = std::min(global_ymin, box.ymin);
        global_xmax = std::max(global_xmax, box.xmax);
        global_ymax = std::max(global_ymax, box.ymax);

        const int64_t col0 = floor_div64(box.xmin, tileSizeScaled);
        const int64_t col1 = floor_div64(box.xmax, tileSizeScaled);
        const int64_t row0 = floor_div64(box.ymin, tileSizeScaled);
        const int64_t row1 = floor_div64(box.ymax, tileSizeScaled);
        for (int64_t row = row0; row <= row1; ++row) {
            for (int64_t col = col0; col <= col1; ++col) {
                region.tile_bins[TileKey{static_cast<int32_t>(row), static_cast<int32_t>(col)}]
                    .push_back(static_cast<uint32_t>(i));
            }
        }
    }

    region.bbox_f = Rectangle<float>(
        static_cast<float>(static_cast<double>(global_xmin) / static_cast<double>(scale)),
        static_cast<float>(static_cast<double>(global_ymin) / static_cast<double>(scale)),
        static_cast<float>(static_cast<double>(global_xmax + 1) / static_cast<double>(scale)),
        static_cast<float>(static_cast<double>(global_ymax + 1) / static_cast<double>(scale)));
    return region;
}

PreparedRegionMask2D prepareRegionFromRectangle(const Rectangle<float>& rect,
                                                int32_t tileSize,
                                                int64_t scale) {
    if (!rect.proper()) {
        throw std::runtime_error("Rectangle region must be proper");
    }
    if (scale <= 0) {
        scale = kDefaultScale;
    }
    Path64 ring;
    ring.reserve(4);
    ring.emplace_back(
        static_cast<int64_t>(std::llround(static_cast<double>(rect.xmin) * static_cast<double>(scale))),
        static_cast<int64_t>(std::llround(static_cast<double>(rect.ymin) * static_cast<double>(scale))));
    ring.emplace_back(
        static_cast<int64_t>(std::llround(static_cast<double>(rect.xmax) * static_cast<double>(scale))),
        static_cast<int64_t>(std::llround(static_cast<double>(rect.ymin) * static_cast<double>(scale))));
    ring.emplace_back(
        static_cast<int64_t>(std::llround(static_cast<double>(rect.xmax) * static_cast<double>(scale))),
        static_cast<int64_t>(std::llround(static_cast<double>(rect.ymax) * static_cast<double>(scale))));
    ring.emplace_back(
        static_cast<int64_t>(std::llround(static_cast<double>(rect.xmin) * static_cast<double>(scale))),
        static_cast<int64_t>(std::llround(static_cast<double>(rect.ymax) * static_cast<double>(scale))));
    return prepareRegionFromPaths(Paths64{std::move(ring)}, tileSize, scale);
}

PreparedRegionMask2D prepareRegionFromGeoJSONGeometry(const json& geometry,
                                                      int32_t tileSize,
                                                      int64_t scale) {
    if (tileSize <= 0) {
        throw std::runtime_error("tileSize must be positive");
    }
    if (scale <= 0) {
        scale = kDefaultScale;
    }

    Paths64 input_rings;
    append_geometry_rings(input_rings, geometry, scale);
    if (input_rings.empty()) {
        throw std::runtime_error("No valid Polygon or MultiPolygon rings found in GeoJSON geometry");
    }

    const Paths64 region_union = Union(input_rings, FillRule::NonZero);
    if (region_union.empty()) {
        throw std::runtime_error("Polygon union is empty after preprocessing");
    }
    return prepareRegionFromPaths(region_union, tileSize, scale);
}

PreparedRegionMask2D loadPreparedRegionGeoJSON(const std::string& geojsonFile,
                                           int32_t tileSize,
                                           int64_t scale) {
    if (tileSize <= 0) {
        throw std::runtime_error("tileSize must be positive");
    }
    if (scale <= 0) {
        scale = kDefaultScale;
    }

    std::ifstream in(geojsonFile);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open GeoJSON file: " + geojsonFile);
    }

    json root;
    in >> root;
    return prepareRegionFromGeoJSONGeometry(root, tileSize, scale);
}

std::vector<PreparedGeoJSONFeature2D> loadPreparedGeoJSONFeatures(
    const std::string& geojsonFile,
    int32_t tileSize,
    int64_t scale,
    const std::string& idProperty) {
    if (tileSize <= 0) {
        throw std::runtime_error("tileSize must be positive");
    }
    if (scale <= 0) {
        scale = kDefaultScale;
    }
    if (idProperty.empty()) {
        throw std::runtime_error("GeoJSON ID property must not be empty");
    }

    std::ifstream in(geojsonFile);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open GeoJSON file: " + geojsonFile);
    }

    json root;
    in >> root;
    if (!root.is_object()) {
        throw std::runtime_error("GeoJSON root must be an object");
    }
    const auto type_it = root.find("type");
    if (type_it == root.end() || !type_it->is_string()) {
        throw std::runtime_error("GeoJSON root is missing string field 'type'");
    }

    std::vector<const json*> features;
    const std::string rootType = type_it->get<std::string>();
    if (rootType == "FeatureCollection") {
        const auto features_it = root.find("features");
        if (features_it == root.end() || !features_it->is_array()) {
            throw std::runtime_error("GeoJSON FeatureCollection is missing array-valued 'features'");
        }
        features.reserve(features_it->size());
        for (const auto& feature : *features_it) {
            features.push_back(&feature);
        }
    } else if (rootType == "Feature") {
        features.push_back(&root);
    } else {
        throw std::runtime_error("Expected GeoJSON FeatureCollection or Feature root");
    }

    std::vector<PreparedGeoJSONFeature2D> out;
    out.reserve(features.size());
    std::unordered_map<std::string, uint32_t> seenIds;
    for (size_t i = 0; i < features.size(); ++i) {
        const json& feature = *features[i];
        if (!feature.is_object()) {
            throw std::runtime_error("GeoJSON feature must be an object");
        }
        const auto feature_type_it = feature.find("type");
        if (feature_type_it == feature.end() || !feature_type_it->is_string() ||
            feature_type_it->get<std::string>() != "Feature") {
            throw std::runtime_error("Expected GeoJSON Feature entries");
        }
        const auto props_it = feature.find("properties");
        if (props_it == feature.end() || !props_it->is_object()) {
            throw std::runtime_error("GeoJSON feature is missing object-valued 'properties'");
        }
        const auto id_it = props_it->find(idProperty);
        if (id_it == props_it->end() || !id_it->is_string()) {
            throw std::runtime_error("GeoJSON feature is missing string property '" + idProperty + "'");
        }
        const std::string id = id_it->get<std::string>();
        if (id.empty()) {
            throw std::runtime_error("GeoJSON feature has empty string property '" + idProperty + "'");
        }
        if (!seenIds.emplace(id, static_cast<uint32_t>(i)).second) {
            throw std::runtime_error("Duplicate GeoJSON feature ID: " + id);
        }
        const auto geom_it = feature.find("geometry");
        if (geom_it == feature.end() || !geom_it->is_object()) {
            throw std::runtime_error("GeoJSON feature '" + id + "' is missing object-valued geometry");
        }

        Paths64 input_rings;
        append_geometry_rings(input_rings, *geom_it, scale, false);
        if (input_rings.empty()) {
            warning("%s: Skipping GeoJSON feature '%s': no valid Polygon or MultiPolygon rings found",
                __func__, id.c_str());
            continue;
        }
        const Paths64 region_union = repair_region_paths_with_clipper(input_rings, id);
        if (region_union.empty()) {
            warning("%s: Skipping GeoJSON feature '%s': polygon repair produced an empty geometry",
                __func__, id.c_str());
            continue;
        }

        try {
            PreparedGeoJSONFeature2D prepared;
            prepared.id = id;
            prepared.region = prepareRegionFromPaths(region_union, tileSize, scale);
            prepared.x = 0.5f * (prepared.region.bbox_f.xmin + prepared.region.bbox_f.xmax);
            prepared.y = 0.5f * (prepared.region.bbox_f.ymin + prepared.region.bbox_f.ymax);
            out.push_back(std::move(prepared));
        } catch (const std::exception& e) {
            warning("%s: Skipping GeoJSON feature '%s': %s", __func__, id.c_str(), e.what());
        }
    }
    return out;
}

namespace {

std::optional<std::string> simple_json_scalar_to_string(const json& value) {
    if (value.is_string()) {
        return value.get<std::string>();
    }
    if (value.is_number_integer()) {
        return std::to_string(value.get<int64_t>());
    }
    if (value.is_number_unsigned()) {
        return std::to_string(value.get<uint64_t>());
    }
    return std::nullopt;
}

std::vector<std::pair<double, double>> parse_simple_json_ring(const json& ring) {
    if (!ring.is_array()) {
        error("%s: GeoJSON ring must be an array", __func__);
    }
    std::vector<std::pair<double, double>> out;
    out.reserve(ring.size());
    for (const auto& coord : ring) {
        if (!coord.is_array() || coord.size() < 2 ||
            !coord[0].is_number() || !coord[1].is_number()) {
            error("%s: GeoJSON coordinate must have numeric x/y", __func__);
        }
        out.emplace_back(coord[0].get<double>(), coord[1].get<double>());
    }
    if (out.size() >= 2 && out.front() == out.back()) {
        out.pop_back();
    }
    if (out.size() < 3) {
        error("%s: polygon ring has fewer than 3 vertices", __func__);
    }
    return out;
}

std::vector<const json*> collect_simple_features(const json& root) {
    std::vector<const json*> out;
    if (!root.is_object()) {
        error("%s: GeoJSON root must be an object", __func__);
    }
    const std::string type = root.value("type", std::string());
    if (type == "FeatureCollection") {
        const auto& features = root.at("features");
        if (!features.is_array()) {
            error("%s: FeatureCollection.features must be an array", __func__);
        }
        for (const auto& feature : features) {
            out.push_back(&feature);
        }
    } else if (type == "Feature" || root.contains("geometry")) {
        out.push_back(&root);
    } else if (root.contains("features") && root["features"].is_array()) {
        for (const auto& feature : root["features"]) {
            out.push_back(&feature);
        }
    } else {
        error("%s: expected FeatureCollection, Feature, or object with features/geometry", __func__);
    }
    return out;
}

uint64_t simple_part_feature_id(uint32_t assignedId, size_t partIndex, size_t nParts) {
    if (nParts <= 1) {
        return static_cast<uint64_t>(assignedId);
    }
    return (static_cast<uint64_t>(assignedId) << 20u) | static_cast<uint64_t>(partIndex);
}

void assign_simple_record_ids(SimplePolygonRingRecord& rec, const std::string& context,
    bool idIsU32, uint32_t nextId, std::set<uint32_t>& seenNumericIds) {
    rec.assignedId = nextId;
    if (idIsU32) {
        if (!str2uint32(rec.polygonId, rec.assignedId)) {
            error("%s: failed parsing polygon ID %s as u32", context.c_str(), rec.polygonId.c_str());
        }
        if (!seenNumericIds.insert(rec.assignedId).second) {
            error("%s: duplicate numeric polygon ID %u", context.c_str(), rec.assignedId);
        }
    }
}

} // namespace

std::pair<double, double> centroidForSimpleRing(
    const std::vector<std::pair<double, double>>& ring) {
    double twiceArea = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double minX = std::numeric_limits<double>::infinity();
    double minY = std::numeric_limits<double>::infinity();
    double maxX = -std::numeric_limits<double>::infinity();
    double maxY = -std::numeric_limits<double>::infinity();
    for (size_t i = 0; i < ring.size(); ++i) {
        const auto& a = ring[i];
        const auto& b = ring[(i + 1) % ring.size()];
        const double cross = a.first * b.second - b.first * a.second;
        twiceArea += cross;
        cx += (a.first + b.first) * cross;
        cy += (a.second + b.second) * cross;
        minX = std::min(minX, a.first);
        minY = std::min(minY, a.second);
        maxX = std::max(maxX, a.first);
        maxY = std::max(maxY, a.second);
    }
    if (std::abs(twiceArea) > 1e-12) {
        return {cx / (3.0 * twiceArea), cy / (3.0 * twiceArea)};
    }
    return {0.5 * (minX + maxX), 0.5 * (minY + maxY)};
}

std::pair<double, double> centroidForSimpleRings(
    const std::vector<std::vector<std::pair<double, double>>>& rings) {
    double totalArea = 0.0;
    double sx = 0.0;
    double sy = 0.0;
    for (const auto& ring : rings) {
        double twiceArea = 0.0;
        for (size_t i = 0; i < ring.size(); ++i) {
            const auto& a = ring[i];
            const auto& b = ring[(i + 1) % ring.size()];
            twiceArea += a.first * b.second - b.first * a.second;
        }
        const double area = std::abs(0.5 * twiceArea);
        const auto c = centroidForSimpleRing(ring);
        sx += c.first * area;
        sy += c.second * area;
        totalArea += area;
    }
    if (totalArea > 0.0) {
        return {sx / totalArea, sy / totalArea};
    }
    return rings.empty() ? std::pair<double, double>{0.0, 0.0} : centroidForSimpleRing(rings.front());
}

std::vector<SimplePolygonRingRecord> readSimplePolygonGeoJSON(
    const std::string& geojsonFile,
    const std::string& idProperty,
    bool idIsU32) {
    std::ifstream in(geojsonFile);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, geojsonFile.c_str());
    }
    json root;
    in >> root;
    std::vector<SimplePolygonRingRecord> out;
    uint32_t nextId = 0;
    std::set<uint32_t> seenNumericIds;
    for (const json* fptr : collect_simple_features(root)) {
        const json& feature = *fptr;
        const json* props = feature.contains("properties") && feature["properties"].is_object()
            ? &feature["properties"] : nullptr;
        if (props == nullptr || !props->contains(idProperty)) {
            error("%s: GeoJSON feature missing property '%s'", __func__, idProperty.c_str());
        }
        auto id = simple_json_scalar_to_string((*props)[idProperty]);
        if (!id || id->empty()) {
            error("%s: invalid GeoJSON polygon ID", __func__);
        }
        const json& geom = feature.contains("geometry") ? feature["geometry"] : feature;
        const std::string type = geom.value("type", std::string());
        std::vector<std::vector<std::pair<double, double>>> rings;
        if (type == "Polygon") {
            rings.push_back(parse_simple_json_ring(geom.at("coordinates").at(0)));
        } else if (type == "MultiPolygon") {
            for (const auto& poly : geom.at("coordinates")) {
                if (poly.is_array() && !poly.empty()) {
                    rings.push_back(parse_simple_json_ring(poly.at(0)));
                }
            }
        } else {
            error("%s: expected Polygon or MultiPolygon geometry", __func__);
        }
        for (size_t i = 0; i < rings.size(); ++i) {
            SimplePolygonRingRecord rec;
            rec.polygonId = *id;
            rec.partIndex = i;
            rec.ring = std::move(rings[i]);
            rec.center = centroidForSimpleRing(rec.ring);
            assign_simple_record_ids(rec, __func__, idIsU32, nextId, seenNumericIds);
            rec.featureId = simple_part_feature_id(rec.assignedId, i, rings.size());
            out.push_back(std::move(rec));
        }
        ++nextId;
    }
    if (out.empty()) {
        error("%s: no geometry records found in %s", __func__, geojsonFile.c_str());
    }
    return out;
}

std::vector<SimplePolygonRingRecord> readSimplePolygonTable(
    const std::string& tableFile,
    const SimplePolygonTableReadOptions& options) {
    const int32_t maxCol = std::max({options.idCol, options.xCol, options.yCol, options.orderCol});
    if (options.idCol < 0 || options.xCol < 0 || options.yCol < 0) {
        error("%s: geometry column indexes must be non-negative", __func__);
    }
    TextLineReader reader(tableFile);
    std::string line;
    uint64_t rowNo = 0;
    if (!read_next_data_line(reader, line, rowNo)) {
        error("%s: empty geometry input %s", __func__, tableFile.c_str());
    }
    const char delim = infer_table_delimiter(line);
    std::map<std::string, std::vector<std::tuple<int64_t, double, double>>> staged;
    std::vector<std::string> idOrder;
    std::set<std::string> closedIds;
    std::string currentId;
    bool haveCurrent = false;

    auto parse_row = [&](const std::string& rowLine, uint64_t row, bool allowHeader) {
        std::vector<std::string> fields = split_delimited(rowLine, delim);
        require_fields(fields, maxCol, __func__, row);
        double x = 0.0;
        double y = 0.0;
        if (!str2double(fields[static_cast<size_t>(options.xCol)], x) ||
            !str2double(fields[static_cast<size_t>(options.yCol)], y)) {
            if (allowHeader) {
                return false;
            }
            error("%s: invalid x/y in geometry row %" PRIu64, __func__, row);
        }
        const std::string id = fields[static_cast<size_t>(options.idCol)];
        if (options.requireConsecutiveIds) {
            if (haveCurrent && currentId != id) {
                closedIds.insert(currentId);
            }
            if (closedIds.find(id) != closedIds.end()) {
                error("%s: polygon ID '%s' appears in more than one block", __func__, id.c_str());
            }
            currentId = id;
            haveCurrent = true;
        }
        if (staged.find(id) == staged.end()) {
            idOrder.push_back(id);
        }
        int64_t order = static_cast<int64_t>(staged[id].size());
        if (options.orderCol >= 0 &&
            !str2int64(fields[static_cast<size_t>(options.orderCol)], order)) {
            error("%s: invalid vertex order in geometry row %" PRIu64, __func__, row);
        }
        staged[id].emplace_back(order, x, y);
        return true;
    };

    parse_row(line, rowNo, true);
    while (reader.getline(line)) {
        ++rowNo;
        if (line.empty() || is_comment_line(line)) {
            continue;
        }
        parse_row(line, rowNo, false);
    }
    if (staged.empty()) {
        error("%s: no geometry records found in %s", __func__, tableFile.c_str());
    }

    std::vector<SimplePolygonRingRecord> out;
    uint32_t nextId = 0;
    std::set<uint32_t> seenNumericIds;
    for (const std::string& id : idOrder) {
        auto& rows = staged[id];
        std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
            return std::get<0>(a) < std::get<0>(b);
        });
        SimplePolygonRingRecord rec;
        rec.polygonId = id;
        rec.partIndex = 0;
        rec.ring.reserve(rows.size());
        for (const auto& row : rows) {
            rec.ring.emplace_back(std::get<1>(row), std::get<2>(row));
        }
        if (rec.ring.size() < 3) {
            error("%s: polygon %s has fewer than 3 vertices", __func__, id.c_str());
        }
        rec.center = centroidForSimpleRing(rec.ring);
        assign_simple_record_ids(rec, __func__, options.idIsU32, nextId, seenNumericIds);
        rec.featureId = rec.assignedId;
        out.push_back(std::move(rec));
        ++nextId;
    }
    if (options.idIsU32) {
        std::sort(out.begin(), out.end(), [](const auto& a, const auto& b) {
            return a.assignedId < b.assignedId;
        });
    }
    return out;
}

std::vector<SimplePolygonRingRecord> readSimplePolygonsAuto(
    const std::string& path,
    const std::string& format,
    const std::string& idProperty,
    const SimplePolygonTableReadOptions& tableOptions) {
    const std::string fmt = to_lower(format);
    const std::string effectiveFmt = fmt == "auto"
        ? infer_table_or_json_format_from_extension(path, "geom-format")
        : fmt;
    if (effectiveFmt == "geojson" || effectiveFmt == "json") {
        return readSimplePolygonGeoJSON(path, idProperty, tableOptions.idIsU32);
    }
    if (effectiveFmt == "table" || effectiveFmt == "tsv" || effectiveFmt == "csv") {
        return readSimplePolygonTable(path, tableOptions);
    }
    error("%s: format must be auto, geojson, json, table, tsv, or csv", __func__);
    return {};
}

bool PreparedRegionMask2D::containsPoint(float x, float y, const TileKey* tile_hint) const {
    if (empty()) {
        return false;
    }
    const Point64 pt(static_cast<int64_t>(std::llround(static_cast<double>(x) * static_cast<double>(scale))),
                     static_cast<int64_t>(std::llround(static_cast<double>(y) * static_cast<double>(scale))));
    TileKey tile = tile_hint ? *tile_hint :
        point_to_tile(pt, static_cast<int64_t>(tileSize) * scale);
    const std::vector<uint32_t> ids = collect_candidates(*this, tile);
    if (ids.empty()) {
        return false;
    }
    return point_in_region_paths(pt, union_paths, ids);
}

RegionTileState PreparedRegionMask2D::classifyTile(const TileKey& tile) const {
    const auto cache_it = tile_state_cache.find(tile);
    if (cache_it != tile_state_cache.end()) {
        return cache_it->second;
    }

    const std::vector<uint32_t> ids = collect_candidates(*this, tile);
    if (ids.empty()) {
        tile_state_cache[tile] = RegionTileState::Outside;
        return RegionTileState::Outside;
    }

    const int64_t tileSizeScaled = static_cast<int64_t>(tileSize) * scale;
    const Rect64 rect = tile_to_scaled_rect(tile, tileSizeScaled);
    Paths64 candidates;
    candidates.reserve(ids.size());
    for (uint32_t idx : ids) {
        if (idx >= union_paths.size()) {
            continue;
        }
        if (!boxes_overlap(comp_bbox[idx], rect)) {
            continue;
        }
        candidates.push_back(union_paths[idx]);
    }
    if (candidates.empty()) {
        tile_state_cache[tile] = RegionTileState::Outside;
        return RegionTileState::Outside;
    }

    const Paths64 uncovered = Difference(Paths64{rect.AsPath()}, candidates, FillRule::NonZero);
    if (uncovered.empty()) {
        tile_state_cache[tile] = RegionTileState::Inside;
        return RegionTileState::Inside;
    }

    const Point64 corners[4] = {
        Point64(rect.left, rect.top),
        Point64(rect.right, rect.top),
        Point64(rect.right, rect.bottom),
        Point64(rect.left, rect.bottom)
    };
    const std::vector<uint32_t> local_ids = [&]() {
        std::vector<uint32_t> out;
        out.reserve(candidates.size());
        for (uint32_t i = 0; i < candidates.size(); ++i) {
            out.push_back(i);
        }
        return out;
    }();
    for (const auto& corner : corners) {
        if (point_in_region_paths(corner, candidates, local_ids)) {
            tile_state_cache[tile] = RegionTileState::Partial;
            return RegionTileState::Partial;
        }
    }
    for (const auto& path : candidates) {
        if (path_touches_rect(path, rect)) {
            tile_state_cache[tile] = RegionTileState::Partial;
            return RegionTileState::Partial;
        }
    }

    tile_state_cache[tile] = RegionTileState::Outside;
    return RegionTileState::Outside;
}

PreparedRegionRasterMask2D prepareRegionRasterMask2D(const PreparedRegionMask2D& region,
                                                     float pixelResolution,
                                                     double delta) {
    PreparedRegionRasterMask2D mask;
    mask.pixelResolution = pixelResolution > 0.0f ? pixelResolution : 1.0f;
    mask.delta = (delta > 0.0) ? delta : (2.0 * static_cast<double>(mask.pixelResolution));
    if (region.empty()) {
        return mask;
    }
    const Paths64 outer = InflatePaths(region.union_paths, mask.delta, JoinType::Round, EndType::Polygon);
    const Paths64 inner = InflatePaths(region.union_paths, -mask.delta, JoinType::Round, EndType::Polygon);
    mask.outer_mask = prepareRegionFromPaths(outer, region.tileSize, region.scale);
    mask.inner_mask = prepareRegionFromPaths(inner, region.tileSize, region.scale);
    return mask;
}

RegionPixelState PreparedRegionRasterMask2D::classifyPixel(int32_t pixX, int32_t pixY,
                                                           const TileKey* tile_hint) const {
    const float x = static_cast<float>(pixX) * pixelResolution;
    const float y = static_cast<float>(pixY) * pixelResolution;
    if (outer_mask.empty() || !outer_mask.containsPoint(x, y, tile_hint)) {
        return RegionPixelState::Outside;
    }
    if (!inner_mask.empty() && inner_mask.containsPoint(x, y, tile_hint)) {
        return RegionPixelState::Interior;
    }
    return RegionPixelState::Boundary;
}
