#include "region_query.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>

#include "json.hpp"

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

bool append_polygon_rings(Paths64& rings, const json& polygon, int64_t scale) {
    if (!polygon.is_array() || polygon.empty()) {
        return false;
    }
    Paths64 polygon_rings;
    for (const auto& ring : polygon) {
        Path64 path = normalize_ring(ring, scale);
        if (path.empty()) {
            return false;
        }
        if (has_self_intersection(path)) {
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

void append_geometry_rings(Paths64& rings, const json& obj, int64_t scale) {
    if (!obj.is_object()) {
        return;
    }
    const auto type_it = obj.find("type");
    if (type_it == obj.end() || !type_it->is_string()) {
        const auto geom_it = obj.find("geometry");
        if (geom_it != obj.end() && geom_it->is_object()) {
            append_geometry_rings(rings, *geom_it, scale);
        }
        const auto features_it = obj.find("features");
        if (features_it != obj.end() && features_it->is_array()) {
            for (const auto& feature : *features_it) {
                append_geometry_rings(rings, feature, scale);
            }
        }
        return;
    }
    const std::string type = type_it->get<std::string>();
    if (type == "Polygon") {
        const auto coords_it = obj.find("coordinates");
        if (coords_it != obj.end()) {
            (void) append_polygon_rings(rings, *coords_it, scale);
        }
        return;
    }
    if (type == "MultiPolygon") {
        const auto coords_it = obj.find("coordinates");
        if (coords_it == obj.end() || !coords_it->is_array()) {
            return;
        }
        for (const auto& polygon : *coords_it) {
            (void) append_polygon_rings(rings, polygon, scale);
        }
        return;
    }
    if (type == "Feature") {
        const auto geom_it = obj.find("geometry");
        if (geom_it != obj.end() && geom_it->is_object()) {
            append_geometry_rings(rings, *geom_it, scale);
        }
        return;
    }
    if (type == "FeatureCollection") {
        const auto features_it = obj.find("features");
        if (features_it == obj.end() || !features_it->is_array()) {
            return;
        }
        for (const auto& feature : *features_it) {
            append_geometry_rings(rings, feature, scale);
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

    Paths64 input_rings;
    append_geometry_rings(input_rings, root, scale);
    if (input_rings.empty()) {
        throw std::runtime_error("No valid Polygon or MultiPolygon rings found in GeoJSON");
    }

    const Paths64 region_union = Union(input_rings, FillRule::NonZero);
    if (region_union.empty()) {
        throw std::runtime_error("Polygon union is empty after preprocessing");
    }
    return prepareRegionFromPaths(region_union, tileSize, scale);
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
