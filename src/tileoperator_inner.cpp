#include "tileoperator.hpp"
#include "tileoperator_common.hpp"
#include "numerical_utils.hpp"
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <set>
#include <memory>
#include <unordered_map>
#include <limits>

namespace {

using tileoperator_detail::fmt::append_format;
using tileoperator_detail::io::append_binary_span;
using tileoperator_detail::io::append_binary_value;
using tileoperator_detail::io::append_pix_top_probs3d_binary;
using tileoperator_detail::io::append_pix_top_probs_binary;
using tileoperator_detail::io::TileWriteResult;
using tileoperator_detail::io::write_tile_result;
using tileoperator_detail::merge::append_placeholder_pairs;
using tileoperator_detail::merge::append_top_probs_prefix;
using tileoperator_detail::merge::tile_key_from_source_xy;
using tileoperator_detail::parallel::process_tile_results_parallel;

using Clipper2Lib::Area;
using Clipper2Lib::ClipType;
using Clipper2Lib::Clipper64;
using Clipper2Lib::FillRule;
using Clipper2Lib::IsPositive;
using Clipper2Lib::Path64;
using Clipper2Lib::Paths64;
using Clipper2Lib::Point64;
using Clipper2Lib::PolyPath64;
using Clipper2Lib::PolyTree64;
using Clipper2Lib::SimplifyPaths;
using Clipper2Lib::TrimCollinear;
using json = nlohmann::json;

template<typename K, typename V>
void add_numeric_map(std::map<K, V>& dst, const std::map<K, V>& src) {
    for (const auto& kv : src) {
        dst[kv.first] += kv.second;
    }
}

template<typename K1, typename K2, typename V>
void add_nested_map(std::map<K1, std::map<K2, V>>& dst, const std::map<K1, std::map<K2, V>>& src) {
    for (const auto& kv : src) {
        add_numeric_map(dst[kv.first], kv.second);
    }
}

int32_t floor_div_int32(int32_t value, int32_t divisor) {
    assert(divisor > 0);
    int32_t q = value / divisor;
    const int32_t r = value % divisor;
    if (r != 0 && ((r > 0) != (divisor > 0))) {
        --q;
    }
    return q;
}

int64_t rounded_abs_area64(const Path64& path) {
    return static_cast<int64_t>(std::llround(std::abs(Area(path))));
}

void append_filtered_polytree_paths(const PolyPath64& node, uint32_t minHoleArea, Paths64& out) {
    if (!node.IsHole()) {
        Path64 shell = TrimCollinear(node.Polygon(), false);
        if (shell.size() >= 3) {
            out.push_back(std::move(shell));
        }
        for (const auto& child : node) {
            const int64_t holeArea = rounded_abs_area64(child->Polygon());
            if (holeArea >= static_cast<int64_t>(minHoleArea)) {
                Path64 hole = TrimCollinear(child->Polygon(), false);
                if (hole.size() >= 3) {
                    out.push_back(std::move(hole));
                }
            }
            for (const auto& grandchild : *child) {
                append_filtered_polytree_paths(*grandchild, minHoleArea, out);
            }
        }
    } else {
        for (const auto& child : node) {
            append_filtered_polytree_paths(*child, minHoleArea, out);
        }
    }
}

json path64_to_json_ring(const Path64& path, double scale, bool wantPositive) {
    Path64 ringPath = path;
    const bool isPositive = IsPositive(ringPath);
    if (isPositive != wantPositive) {
        std::reverse(ringPath.begin(), ringPath.end());
    }
    json ring = json::array();
    for (const Point64& pt : ringPath) {
        ring.push_back({static_cast<double>(pt.x) * scale, static_cast<double>(pt.y) * scale});
    }
    if (!ringPath.empty()) {
        const Point64& pt0 = ringPath.front();
        ring.push_back({static_cast<double>(pt0.x) * scale, static_cast<double>(pt0.y) * scale});
    }
    return ring;
}

void append_geojson_polygons(const PolyPath64& node, double scale, json& multipolygon) {
    if (!node.IsHole()) {
        json polygon = json::array();
        polygon.push_back(path64_to_json_ring(node.Polygon(), scale, true));
        for (const auto& child : node) {
            polygon.push_back(path64_to_json_ring(child->Polygon(), scale, false));
        }
        multipolygon.push_back(std::move(polygon));
        for (const auto& child : node) {
            for (const auto& grandchild : *child) {
                append_geojson_polygons(*grandchild, scale, multipolygon);
            }
        }
    } else {
        for (const auto& child : node) {
            append_geojson_polygons(*child, scale, multipolygon);
        }
    }
}

} // namespace

void TileOperator::loadDenseTile(const TileInfo& blk, std::ifstream& in, DenseTile& out, uint8_t bg,
    uint64_t& nOutOfRangeIgnored, uint64_t& nBadLabelIgnored) const {
    if (coord_dim_ != 2) {
        error("%s: Only 2D records are supported", __func__);
    }
    initTileGeom(blk, out.geom);
    const size_t nPix = out.geom.W * out.geom.H;
    out.lab.assign(nPix, bg);
    out.boundary.clear();

    if (usesRasterResolutionOverride()) {
        std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
        loadTileToMap(out.geom.key, pixelMap, nullptr, &in);
        for (const auto& kv : pixelMap) {
            if (kv.second.ks.empty() || kv.second.ps.empty()) {
                continue;
            }
            const int32_t xpix = kv.first.first;
            const int32_t ypix = kv.first.second;
            if (xpix < out.geom.pixX0 || xpix >= out.geom.pixX1 || ypix < out.geom.pixY0 || ypix >= out.geom.pixY1) {
                ++nOutOfRangeIgnored;
                continue;
            }
            const int32_t k = kv.second.ks[0];
            if (k < 0 || k >= static_cast<int32_t>(bg)) {
                ++nBadLabelIgnored;
                continue;
            }
            const size_t x0 = static_cast<size_t>(xpix - out.geom.pixX0);
            const size_t y0 = static_cast<size_t>(ypix - out.geom.pixY0);
            out.lab[y0 * out.geom.W + x0] = static_cast<uint8_t>(k);
        }
        return;
    }

    in.clear();
    in.seekg(blk.idx.st);
    if (!in.good()) {
        error("%s: Failed seeking input stream to tile (%d, %d)", __func__, out.geom.key.row, out.geom.key.col);
    }

    TopProbs rec;
    int32_t xpix = 0;
    int32_t ypix = 0;
    uint64_t pos = blk.idx.st;
    while (readNextRecord2DAsPixel(in, pos, blk.idx.ed, xpix, ypix, rec)) {
        if (rec.ks.empty() || rec.ps.empty()) {
            continue;
        }
        if (xpix < out.geom.pixX0 || xpix >= out.geom.pixX1 || ypix < out.geom.pixY0 || ypix >= out.geom.pixY1) {
            ++nOutOfRangeIgnored;
            continue;
        }
        const int32_t k = rec.ks[0];
        if (k < 0 || k >= static_cast<int32_t>(bg)) {
            ++nBadLabelIgnored;
            continue;
        }
        const size_t x0 = static_cast<size_t>(xpix - out.geom.pixX0);
        const size_t y0 = static_cast<size_t>(ypix - out.geom.pixY0);
        out.lab[y0 * out.geom.W + x0] = static_cast<uint8_t>(k);
    }
}

TileOperator::TileCCL TileOperator::tileLocalCCL(const DenseTile& tile, uint8_t bg,
    const std::vector<uint8_t>* boundaryMask, bool keepPixelCid) const {
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    TileCCL out;
    out.pixX0 = tile.geom.pixX0;
    out.pixX1 = tile.geom.pixX1;
    out.pixY0 = tile.geom.pixY0;
    out.pixY1 = tile.geom.pixY1;
    if (tile.geom.W == 0 || tile.geom.H == 0) {
        return out;
    }
    const size_t nPix = tile.geom.W * tile.geom.H;
    std::vector<uint32_t> parent(nPix, INVALID);
    std::vector<uint8_t> rankv(nPix, 0);

    auto findRoot = [&](uint32_t x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    };
    auto unite = [&](uint32_t a, uint32_t b) {
        uint32_t ra = findRoot(a);
        uint32_t rb = findRoot(b);
        if (ra == rb) return;
        if (rankv[ra] < rankv[rb]) std::swap(ra, rb);
        parent[rb] = ra;
        if (rankv[ra] == rankv[rb]) rankv[ra]++;
    };
    // 1st pass: left/up unions
    for (size_t y = 0; y < tile.geom.H; ++y) {
        const size_t row = y * tile.geom.W;
        for (size_t x = 0; x < tile.geom.W; ++x) {
            const size_t idx = row + x;
            const uint8_t lbl = tile.lab[idx];
            if (lbl == bg) continue;
            parent[idx] = static_cast<uint32_t>(idx);
            if (x > 0 && tile.lab[idx - 1] == lbl) {
                unite(static_cast<uint32_t>(idx), static_cast<uint32_t>(idx - 1));
            }
            if (y > 0 && tile.lab[idx - tile.geom.W] == lbl) {
                unite(static_cast<uint32_t>(idx), static_cast<uint32_t>(idx - tile.geom.W));
            }
        }
    }
    // 2nd pass: pixel -> compact component ID
    std::vector<uint32_t> root2cid(nPix, INVALID);
    if (keepPixelCid) {
        out.pixelCid.assign(nPix, INVALID);
    }
    for (size_t idx = 0; idx < nPix; ++idx) {
        if (parent[idx] == INVALID) continue;
        uint32_t r = findRoot(static_cast<uint32_t>(idx));
        uint32_t cid = root2cid[r];
        const size_t y = idx / tile.geom.W;
        const size_t x = idx - y * tile.geom.W;
        const int32_t gx = tile.geom.pixX0 + static_cast<int32_t>(x);
        const int32_t gy = tile.geom.pixY0 + static_cast<int32_t>(y);
        if (cid == INVALID) {
            cid = static_cast<uint32_t>(out.compSize.size());
            root2cid[r] = cid;
            out.compSize.push_back(0);
            out.compLabel.push_back(tile.lab[idx]);
            out.compSumX.push_back(0);
            out.compSumY.push_back(0);
            out.compBox.push_back(PixBox{});
        }
        out.compSize[cid] += 1;
        out.compSumX[cid] += static_cast<uint64_t>(gx);
        out.compSumY[cid] += static_cast<uint64_t>(gy);
        out.compBox[cid].include(gx, gy);
        if (keepPixelCid) {
            out.pixelCid[idx] = cid;
        }
    }
    out.ncomp = static_cast<uint32_t>(out.compSize.size());
    auto cidAt = [&](size_t idx) -> uint32_t {
        if (parent[idx] == INVALID) return INVALID;
        uint32_t r = findRoot((uint32_t)idx);
        return root2cid[r];
    };
    // borders
    out.leftCid.assign(tile.geom.H, INVALID);
    out.rightCid.assign(tile.geom.H, INVALID);
    for (size_t y = 0; y < tile.geom.H; ++y) {
        const size_t row = y * tile.geom.W;
        out.leftCid[y] = cidAt(row);
        out.rightCid[y] = cidAt(row + tile.geom.W - 1);
    }
    out.topCid.assign(tile.geom.W, INVALID);
    out.bottomCid.assign(tile.geom.W, INVALID);
    const size_t bottomRow = (tile.geom.H - 1) * tile.geom.W;
    for (size_t x = 0; x < tile.geom.W; ++x) {
        out.topCid[x] = cidAt(x);
        out.bottomCid[x] = cidAt(bottomRow + x);
    }
    if (boundaryMask != nullptr) {
        if (boundaryMask->size() != nPix) {
            error("%s: Boundary mask size mismatch", __func__);
        }
        out.bndPix.reserve(tile.geom.W * 2 + tile.geom.H * 2);
        out.bndCid.reserve(tile.geom.W * 2 + tile.geom.H * 2);
        for (size_t idx = 0; idx < nPix; ++idx) {
            if (tile.lab[idx] == bg || (*boundaryMask)[idx] == 0) continue;
            const uint32_t cid = cidAt(idx);
            if (cid == INVALID || cid >= out.ncomp) continue;
            out.bndPix.push_back(static_cast<uint32_t>(idx));
            out.bndCid.push_back(cid);
        }
    }
    return out;
}

void TileOperator::computeTileBoundaryMasks(std::vector<DenseTile>& tiles) const {
    // Intra-tile boundaries
    for (auto& tile : tiles) {
        const size_t nPix = tile.geom.W * tile.geom.H;
        tile.boundary.assign(nPix, 0);
        for (size_t y = 0; y < tile.geom.H; ++y) {
            const size_t row = y * tile.geom.W;
            for (size_t x = 0; x < tile.geom.W; ++x) {
                const size_t idx = row + x;
                const uint8_t c = tile.lab[idx];
                bool isBoundary = false;
                if (x > 0 && tile.lab[idx - 1] != c) isBoundary = true;
                if (x + 1 < tile.geom.W && tile.lab[idx + 1] != c) isBoundary = true;
                if (y > 0 && tile.lab[idx - tile.geom.W] != c) isBoundary = true;
                if (y + 1 < tile.geom.H && tile.lab[idx + tile.geom.W] != c) isBoundary = true;
                tile.boundary[idx] = isBoundary ? 1 : 0;
            }
        }
    }
    // right/down inter-tile boundaries
    for (size_t i = 0; i < tiles.size(); ++i) {
        const TileKey key = tiles[i].geom.key;
        const auto rightIt = tile_lookup_.find(TileKey{key.row, key.col + 1});
        if (rightIt != tile_lookup_.end()) {
            size_t j = rightIt->second;
            if (j < tiles.size() && tiles[i].geom.W > 0 && tiles[j].geom.W > 0) {
                auto& a = tiles[i];
                auto& b = tiles[j];
                const int32_t y0 = std::max(a.geom.pixY0, b.geom.pixY0);
                const int32_t y1 = std::min(a.geom.pixY1, b.geom.pixY1);
                for (int32_t gy = y0; gy < y1; ++gy) {
                    const size_t ay = static_cast<size_t>(gy - a.geom.pixY0);
                    const size_t by = static_cast<size_t>(gy - b.geom.pixY0);
                    const size_t ia = ay * a.geom.W + (a.geom.W - 1);
                    const size_t ib = by * b.geom.W;
                    if (a.lab[ia] != b.lab[ib]) {
                        a.boundary[ia] = 1;
                        b.boundary[ib] = 1;
                    }
                }
            }
        }

        const auto downIt = tile_lookup_.find(TileKey{key.row + 1, key.col});
        if (downIt != tile_lookup_.end()) {
            size_t j = downIt->second;
            if (j < tiles.size() && tiles[i].geom.H > 0 && tiles[j].geom.H > 0) {
                auto& a = tiles[i];
                auto& b = tiles[j];
                const int32_t x0 = std::max(a.geom.pixX0, b.geom.pixX0);
                const int32_t x1 = std::min(a.geom.pixX1, b.geom.pixX1);
                for (int32_t gx = x0; gx < x1; ++gx) {
                    const size_t ax = static_cast<size_t>(gx - a.geom.pixX0);
                    const size_t bx = static_cast<size_t>(gx - b.geom.pixX0);
                    const size_t ia = (a.geom.H - 1) * a.geom.W + ax;
                    const size_t ib = bx;
                    if (a.lab[ia] != b.lab[ib]) {
                        a.boundary[ia] = 1;
                        b.boundary[ib] = 1;
                    }
                }
            }
        }
    }
}

TileOperator::BorderRemapInfo TileOperator::remapTileToBorderComponents(TileCCL& t, uint32_t invalid) const {
    BorderRemapInfo out;
    out.oldCompSize = std::move(t.compSize);
    out.oldCompLabel = std::move(t.compLabel);
    out.oldCompSumX = std::move(t.compSumX);
    out.oldCompSumY = std::move(t.compSumY);
    out.oldCompBox = std::move(t.compBox);
    if (out.oldCompSize.size() != out.oldCompLabel.size()) {
        error("%s: Component size/label length mismatch", __func__);
    }
    if (out.oldCompSize.size() != out.oldCompSumX.size() ||
        out.oldCompSize.size() != out.oldCompSumY.size() ||
        out.oldCompSize.size() != out.oldCompBox.size()) {
        error("%s: Component stat length mismatch", __func__);
    }
    const size_t oldN = out.oldCompSize.size();
    out.remap.assign(oldN, invalid);

    std::vector<uint8_t> isBorder(oldN, 0);
    auto markBorder = [&](const std::vector<uint32_t>& edgeCids) {
        for (uint32_t cid : edgeCids) {
            if (cid != invalid && cid < oldN) {
                isBorder[cid] = 1;
            }
        }
    };
    markBorder(t.leftCid);
    markBorder(t.rightCid);
    markBorder(t.topCid);
    markBorder(t.bottomCid);

    std::vector<uint32_t> borderCompSize;
    std::vector<uint8_t> borderCompLabel;
    std::vector<uint64_t> borderCompSumX;
    std::vector<uint64_t> borderCompSumY;
    std::vector<PixBox> borderCompBox;
    borderCompSize.reserve(oldN);
    borderCompLabel.reserve(oldN);
    borderCompSumX.reserve(oldN);
    borderCompSumY.reserve(oldN);
    borderCompBox.reserve(oldN);
    for (uint32_t cid = 0; cid < oldN; ++cid) {
        if (!isBorder[cid]) continue;
        out.remap[cid] = static_cast<uint32_t>(borderCompSize.size());
        borderCompSize.push_back(out.oldCompSize[cid]);
        borderCompLabel.push_back(out.oldCompLabel[cid]);
        borderCompSumX.push_back(out.oldCompSumX[cid]);
        borderCompSumY.push_back(out.oldCompSumY[cid]);
        borderCompBox.push_back(out.oldCompBox[cid]);
    }

    auto remapEdge = [&](std::vector<uint32_t>& edgeCids) {
        for (auto& cid : edgeCids) {
            if (cid == invalid) continue;
            if (cid >= out.remap.size() || out.remap[cid] == invalid) {
                error("%s: Unexpected non-border component on tile boundary", __func__);
            }
            cid = out.remap[cid];
        }
    };
    remapEdge(t.leftCid);
    remapEdge(t.rightCid);
    remapEdge(t.topCid);
    remapEdge(t.bottomCid);

    t.compSize.swap(borderCompSize);
    t.compLabel.swap(borderCompLabel);
    t.compSumX.swap(borderCompSumX);
    t.compSumY.swap(borderCompSumY);
    t.compBox.swap(borderCompBox);
    t.ncomp = static_cast<uint32_t>(t.compSize.size());
    return out;
}

TileOperator::BorderDSUState TileOperator::mergeBorderComponentsWithDSU(
    const std::vector<TileCCL>& perTile, uint8_t bg, uint32_t invalid) const {
    BorderDSUState out;
    if (perTile.size() != blocks_.size()) {
        error("%s: perTile/block size mismatch", __func__);
    }
    out.globalBase.assign(perTile.size() + 1, 0);
    for (size_t i = 0; i < perTile.size(); ++i) {
        out.globalBase[i + 1] = out.globalBase[i] + static_cast<size_t>(perTile[i].ncomp);
    }
    const size_t nGlobalComp = out.globalBase.back();
    out.rootSize.assign(nGlobalComp, 0);
    out.rootLabel.assign(nGlobalComp, bg);
    out.rootSumX.assign(nGlobalComp, 0);
    out.rootSumY.assign(nGlobalComp, 0);
    out.rootBox.assign(nGlobalComp, PixBox{});
    out.tileRoot.resize(perTile.size());
    for (size_t i = 0; i < perTile.size(); ++i) {
        out.tileRoot[i].assign(static_cast<size_t>(perTile[i].ncomp), 0);
    }
    if (nGlobalComp == 0) return out;

    if (nGlobalComp > std::numeric_limits<uint32_t>::max()) {
        warning("%s: Total number of border components %zu exceeds uint32_t max", __func__, nGlobalComp);
    }

    DisjointSet dsu(nGlobalComp);
    for (size_t i = 0; i < perTile.size(); ++i) {
        const TileCCL& a = perTile[i];
        const TileKey key{blocks_[i].idx.row, blocks_[i].idx.col};
        const auto rightIt = tile_lookup_.find(TileKey{key.row, key.col + 1});
        if (rightIt != tile_lookup_.end()) {
            const size_t j = rightIt->second;
            if (j < perTile.size()) {
                const TileCCL& b = perTile[j];
                const int32_t y0 = std::max(a.pixY0, b.pixY0);
                const int32_t y1 = std::min(a.pixY1, b.pixY1);
                uint32_t last_ca = invalid, last_cb = invalid;
                for (int32_t gy = y0; gy < y1; ++gy) {
                    const uint32_t ca = a.rightCid[static_cast<size_t>(gy - a.pixY0)];
                    const uint32_t cb = b.leftCid[static_cast<size_t>(gy - b.pixY0)];
                    if (ca == invalid || cb == invalid || (ca == last_ca && cb == last_cb)) continue;
                    last_ca = ca; last_cb = cb;
                    if (a.compLabel[ca] != b.compLabel[cb]) continue;
                    dsu.unite(out.globalBase[i] + static_cast<size_t>(ca),
                              out.globalBase[j] + static_cast<size_t>(cb));
                }
            }
        }
        const auto downIt = tile_lookup_.find(TileKey{key.row + 1, key.col});
        if (downIt != tile_lookup_.end()) {
            const size_t j = downIt->second;
            if (j < perTile.size()) {
                const TileCCL& b = perTile[j];
                const int32_t x0 = std::max(a.pixX0, b.pixX0);
                const int32_t x1 = std::min(a.pixX1, b.pixX1);
                uint32_t last_ca = invalid, last_cb = invalid;
                for (int32_t gx = x0; gx < x1; ++gx) {
                    const uint32_t ca = a.bottomCid[static_cast<size_t>(gx - a.pixX0)];
                    const uint32_t cb = b.topCid[static_cast<size_t>(gx - b.pixX0)];
                    if (ca == invalid || cb == invalid || (ca == last_ca && cb == last_cb)) continue;
                    last_ca = ca; last_cb = cb;
                    if (a.compLabel[ca] != b.compLabel[cb]) continue;
                    dsu.unite(out.globalBase[i] + static_cast<size_t>(ca),
                              out.globalBase[j] + static_cast<size_t>(cb));
                }
            }
        }
    }

    dsu.compress_all();
    for (size_t i = 0; i < perTile.size(); ++i) {
        const TileCCL& t = perTile[i];
        for (size_t cid = 0; cid < t.ncomp; ++cid) {
            const size_t gid = out.globalBase[i] + cid;
            const size_t root = dsu.parent[gid];
            out.tileRoot[i][cid] = root;
            out.rootSize[root] += static_cast<uint64_t>(t.compSize[cid]);
            out.rootSumX[root] += t.compSumX[cid];
            out.rootSumY[root] += t.compSumY[cid];
            out.rootBox[root].include(t.compBox[cid]);
            if (out.rootLabel[root] == bg) {
                out.rootLabel[root] = t.compLabel[cid];
            } else if (out.rootLabel[root] != t.compLabel[cid]) {
                error("%s: Label mismatch after seam union", __func__);
            }
        }
    }
    return out;
}

void TileOperator::mergeTiles2D(const std::vector<TileKey>& mainTiles,
    const std::vector<MergeSourcePlan>& mergePlans, bool keepAllMain, bool binaryOutput,
    FILE* fp, int fdMain, int fdIndex, long& currentOffset) {
    const char* funcName = __func__;
    const size_t nSources = mergePlans.size();
    const size_t totalK = std::accumulate(mergePlans.begin(), mergePlans.end(), size_t(0),
        [](size_t sum, const MergeSourcePlan& plan) {
            return sum + static_cast<size_t>(plan.keepK);
        });
    auto buildTileResult = [&](const TileKey& tile, std::vector<std::ifstream>& streams) {
        TileWriteResult result;
        result.tile = tile;
        std::map<std::pair<int32_t, int32_t>, TopProbs> mainMap;
        if (mergePlans[0].op->loadTileToMap(tile, mainMap, nullptr, &streams[0]) == 0) {
            warning("%s: Main tile (%d, %d) has no data", funcName, tile.row, tile.col);
            return result;
        }
        if (mainMap.empty()) {
            return result;
        }
        result.nMain = static_cast<uint32_t>(mainMap.size());

        std::vector<std::map<TileKey, std::map<std::pair<int32_t, int32_t>, TopProbs>>> auxTileCaches(nSources);
        std::vector<std::set<TileKey>> missingAuxTiles(nSources);
        auto findAuxRecord = [&](size_t srcIdx, int32_t mainX, int32_t mainY) -> const TopProbs* {
            const MergeSourcePlan& plan = mergePlans[srcIdx];
            const std::pair<int32_t, int32_t> auxKey{
                floor_div_int32(mainX, plan.ratioXY),
                floor_div_int32(mainY, plan.ratioXY)
            };
            const TileKey auxTile = tile_key_from_source_xy(auxKey.first, auxKey.second,
                plan.srcResXY, plan.tileSize);
            if (missingAuxTiles[srcIdx].count(auxTile) > 0) {
                return nullptr;
            }
            auto tileIt = auxTileCaches[srcIdx].find(auxTile);
            if (tileIt == auxTileCaches[srcIdx].end()) {
                std::map<std::pair<int32_t, int32_t>, TopProbs> auxMap;
                if (plan.op->loadTileToMap(auxTile, auxMap, nullptr, &streams[srcIdx]) == 0) {
                    missingAuxTiles[srcIdx].insert(auxTile);
                    return nullptr;
                }
                tileIt = auxTileCaches[srcIdx].emplace(auxTile, std::move(auxMap)).first;
            }
            auto recIt = tileIt->second.find(auxKey);
            if (recIt == tileIt->second.end()) {
                return nullptr;
            }
            return &recIt->second;
        };

        const size_t bytesPerRecord = 2 * sizeof(int32_t) +
            totalK * (sizeof(int32_t) + sizeof(float));
        if (binaryOutput) {
            result.binaryData.reserve(mainMap.size() * bytesPerRecord);
        }
        for (const auto& kv : mainMap) {
            TopProbs merged;
            merged.ks.reserve(totalK);
            merged.ps.reserve(totalK);
            append_top_probs_prefix(merged, kv.second, mergePlans[0].keepK);
            bool keep = true;
            for (size_t i = 1; i < nSources; ++i) {
                const TopProbs* aux = findAuxRecord(i, kv.first.first, kv.first.second);
                if (aux == nullptr) {
                    if (!keepAllMain) {
                        keep = false;
                        break;
                    }
                    append_placeholder_pairs(merged, mergePlans[i].keepK);
                } else {
                    append_top_probs_prefix(merged, *aux, mergePlans[i].keepK);
                }
            }
            if (!keep) {
                continue;
            }
            ++result.n;
            if (binaryOutput) {
                append_pix_top_probs_binary(result.binaryData, kv.first.first, kv.first.second, merged);
            } else {
                append_format(result.textData, "%d\t%d", kv.first.first, kv.first.second);
                for (size_t i = 0; i < merged.ks.size(); ++i) {
                    append_format(result.textData, "\t%d\t%.4e", merged.ks[i], merged.ps[i]);
                }
                result.textData.push_back('\n');
            }
        }
        return result;
    };

    auto writeResult = [&](const TileWriteResult& result) {
        write_tile_result(result, binaryOutput, fp, fdMain, fdIndex, currentOffset, formatInfo_.tileSize);
        notice("%s: Merged tile (%d, %d) with %u output pixels from %u main-input pixels",
            funcName, result.tile.row, result.tile.col, result.n, result.nMain);
    };
    process_tile_results_parallel(mainTiles, threads_,
        [&]() { return std::vector<std::ifstream>(nSources); },
        buildTileResult, writeResult);
}

void TileOperator::mergeTiles3D(const std::vector<TileKey>& mainTiles,
    const std::vector<MergeSourcePlan>& mergePlans, bool keepAllMain, bool binaryOutput,
    FILE* fp, int fdMain, int fdIndex, long& currentOffset) {
    const char* funcName = __func__;
    const size_t nSources = mergePlans.size();
    const size_t totalK = std::accumulate(mergePlans.begin(), mergePlans.end(), size_t(0),
        [](size_t sum, const MergeSourcePlan& plan) {
            return sum + static_cast<size_t>(plan.keepK);
        });
    auto buildTileResult = [&](const TileKey& tile, std::vector<std::ifstream>& streams) {
        TileWriteResult result;
        result.tile = tile;
        std::map<PixelKey3, TopProbs> mainMap;
        if (mergePlans[0].op->loadTileToMap3D(tile, mainMap, &streams[0]) == 0) {
            warning("%s: Main tile (%d, %d) has no data", funcName, tile.row, tile.col);
            return result;
        }
        if (mainMap.empty()) {
            return result;
        }
        result.nMain = static_cast<uint32_t>(mainMap.size());

        std::vector<std::map<TileKey, std::map<std::pair<int32_t, int32_t>, TopProbs>>> auxTileCaches2D(nSources);
        std::vector<std::set<TileKey>> missingAuxTiles2D(nSources);
        std::vector<std::map<TileKey, std::map<PixelKey3, TopProbs>>> auxTileCaches3D(nSources);
        std::vector<std::set<TileKey>> missingAuxTiles3D(nSources);
        auto findAuxRecord = [&](size_t srcIdx, int32_t mainX, int32_t mainY, int32_t mainZ) -> const TopProbs* {
            const MergeSourcePlan& plan = mergePlans[srcIdx];
            const int32_t auxX = floor_div_int32(mainX, plan.ratioXY);
            const int32_t auxY = floor_div_int32(mainY, plan.ratioXY);
            const TileKey auxTile = tile_key_from_source_xy(auxX, auxY, plan.srcResXY, plan.tileSize);
            if (plan.relation == MergeSourceRelation::Broadcast2DTo3D) {
                if (missingAuxTiles2D[srcIdx].count(auxTile) > 0) {
                    return nullptr;
                }
                auto tileIt = auxTileCaches2D[srcIdx].find(auxTile);
                if (tileIt == auxTileCaches2D[srcIdx].end()) {
                    std::map<std::pair<int32_t, int32_t>, TopProbs> auxMap;
                    if (plan.op->loadTileToMap(auxTile, auxMap, nullptr, &streams[srcIdx]) == 0) {
                        missingAuxTiles2D[srcIdx].insert(auxTile);
                        return nullptr;
                    }
                    tileIt = auxTileCaches2D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
                }
                auto recIt = tileIt->second.find({auxX, auxY});
                if (recIt == tileIt->second.end()) {
                    return nullptr;
                }
                return &recIt->second;
            }
            const int32_t auxZ = floor_div_int32(mainZ, plan.ratioZ);
            if (missingAuxTiles3D[srcIdx].count(auxTile) > 0) {
                return nullptr;
            }
            auto tileIt = auxTileCaches3D[srcIdx].find(auxTile);
            if (tileIt == auxTileCaches3D[srcIdx].end()) {
                std::map<PixelKey3, TopProbs> auxMap;
                if (plan.op->loadTileToMap3D(auxTile, auxMap, &streams[srcIdx]) == 0) {
                    missingAuxTiles3D[srcIdx].insert(auxTile);
                    return nullptr;
                }
                tileIt = auxTileCaches3D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
            }
            auto recIt = tileIt->second.find({auxX, auxY, auxZ});
            if (recIt == tileIt->second.end()) {
                return nullptr;
            }
            return &recIt->second;
        };

        const size_t bytesPerRecord = 3 * sizeof(int32_t) +
            totalK * (sizeof(int32_t) + sizeof(float));
        if (binaryOutput) {
            result.binaryData.reserve(mainMap.size() * bytesPerRecord);
        }
        for (const auto& kv : mainMap) {
            TopProbs merged;
            merged.ks.reserve(totalK);
            merged.ps.reserve(totalK);
            append_top_probs_prefix(merged, kv.second, mergePlans[0].keepK);
            bool keep = true;
            for (size_t i = 1; i < nSources; ++i) {
                const TopProbs* aux = findAuxRecord(i,
                    std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first));
                if (aux == nullptr) {
                    if (!keepAllMain) {
                        keep = false;
                        break;
                    }
                    append_placeholder_pairs(merged, mergePlans[i].keepK);
                } else {
                    append_top_probs_prefix(merged, *aux, mergePlans[i].keepK);
                }
            }
            if (!keep) {
                continue;
            }
            ++result.n;
            if (binaryOutput) {
                append_pix_top_probs3d_binary(result.binaryData,
                    std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first), merged);
            } else {
                append_format(result.textData, "%d\t%d\t%d",
                    std::get<0>(kv.first), std::get<1>(kv.first), std::get<2>(kv.first));
                for (size_t i = 0; i < merged.ks.size(); ++i) {
                    append_format(result.textData, "\t%d\t%.4e", merged.ks[i], merged.ps[i]);
                }
                result.textData.push_back('\n');
            }
        }
        return result;
    };

    auto writeResult = [&](const TileWriteResult& result) {
        write_tile_result(result, binaryOutput, fp, fdMain, fdIndex, currentOffset, formatInfo_.tileSize);
        notice("%s: Merged tile (%d, %d) with %u output pixels from %u main-input pixels",
            funcName, result.tile.row, result.tile.col, result.n, result.nMain);
    };
    process_tile_results_parallel(mainTiles, threads_,
        [&]() { return std::vector<std::ifstream>(nSources); },
        buildTileResult, writeResult);
}

void TileOperator::annotateTiles2D(const std::vector<TileKey>& tiles,
    TileReader& reader, uint32_t icol_x, uint32_t icol_y,
    uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset) {
    const char* funcName = __func__;
    notice("%s: Start annotating query with %lu tiles", funcName, tiles.size());
    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;
    auto buildTileResult = [&](const TileKey& tile, std::ifstream& tileStream) {
        TileWriteResult result;
        result.tile = tile;
        std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
        if (loadTileToMap(tile, pixelMap, nullptr, &tileStream) <= 0) {
            debug("%s: Query tile (%d, %d) is not in the annotation dataset", funcName, tile.row, tile.col);
            return result;
        }
        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) {
            return result;
        }
        std::string s;
        while (it->next(s)) {
            if (s.empty() || s[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", s, ntok + 1, true, true, true);
            if (tokens.size() < ntok) {
                error("%s: Invalid line: %s", __func__, s.c_str());
            }
            float x, y;
            if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
                error("%s: Invalid coordinates in line: %s", __func__, s.c_str());
            }
            int32_t ix = static_cast<int32_t>(std::floor(x / res));
            int32_t iy = static_cast<int32_t>(std::floor(y / res));
            auto pit = pixelMap.find({ix, iy});
            if (pit == pixelMap.end()) {
                continue;
            }
            result.textData += s;
            for (size_t k = 0; k < pit->second.ks.size(); ++k) {
                append_format(result.textData, "\t%d\t%.4e", pit->second.ks[k], pit->second.ps[k]);
            }
            result.textData.push_back('\n');
            result.n++;
        }
        return result;
    };

    auto writeResult = [&](const TileWriteResult& result) {
        write_tile_result(result, false, fp, -1, fdIndex, currentOffset, formatInfo_.tileSize);
        notice("%s: Annotated tile (%d, %d) with %u points",
            funcName, result.tile.row, result.tile.col, result.n);
    };
    process_tile_results_parallel(tiles, threads_,
        [&]() { return std::ifstream(); },
        buildTileResult, writeResult);
}

void TileOperator::annotateTiles3D(const std::vector<TileKey>& tiles,
    TileReader& reader, uint32_t icol_x, uint32_t icol_y, uint32_t icol_z,
    uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset) {
    const char* funcName = __func__;
    notice("%s: Start annotating query with %lu tiles", funcName, tiles.size());
    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;
    auto buildTileResult = [&](const TileKey& tile, std::ifstream& tileStream) {
        TileWriteResult result;
        result.tile = tile;
        std::map<PixelKey3, TopProbs> pixelMap3d;
        if (loadTileToMap3D(tile, pixelMap3d, &tileStream) <= 0) {
            debug("%s: Query tile (%d, %d) is not in the annotation dataset", funcName, tile.row, tile.col);
            return result;
        }
        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) {
            return result;
        }
        std::string s;
        while (it->next(s)) {
            if (s.empty() || s[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", s, ntok + 1, true, true, true);
            if (tokens.size() < ntok) {
                error("%s: Invalid line: %s", __func__, s.c_str());
            }
            float x, y, z;
            if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
                error("%s: Invalid coordinates in line: %s", __func__, s.c_str());
            }
            if (!str2float(tokens[icol_z], z)) {
                error("%s: Invalid z coordinate in line: %s", __func__, s.c_str());
            }
            int32_t ix = static_cast<int32_t>(std::floor(x / res));
            int32_t iy = static_cast<int32_t>(std::floor(y / res));
            int32_t iz = static_cast<int32_t>(std::floor(z / res));
            auto pit = pixelMap3d.find(std::make_tuple(ix, iy, iz));
            if (pit == pixelMap3d.end()) {
                continue;
            }
            result.textData += s;
            for (size_t k = 0; k < pit->second.ks.size(); ++k) {
                append_format(result.textData, "\t%d\t%.4e", pit->second.ks[k], pit->second.ps[k]);
            }
            result.textData.push_back('\n');
            result.n++;
        }
        return result;
    };

    auto writeResult = [&](const TileWriteResult& result) {
        write_tile_result(result, false, fp, -1, fdIndex, currentOffset, formatInfo_.tileSize);
        notice("%s: Annotated tile (%d, %d) with %u points",
            funcName, result.tile.row, result.tile.col, result.n);
    };
    process_tile_results_parallel(tiles, threads_,
        [&]() { return std::ifstream(); },
        buildTileResult, writeResult);
}

void TileOperator::probDotTiles2D(const std::set<TileKey>& commonTiles,
    const std::vector<TileOperator*>& opPtrs,
    const std::vector<uint32_t>& k2keep, const std::vector<uint32_t>& offsets,
    std::vector<std::map<int32_t, double>>& marginals,
    std::vector<std::map<std::pair<int32_t, int32_t>, double>>& internalDots,
    std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>>& crossDots, size_t& count) {
    uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    size_t nSets = k2keep.size();
    const std::vector<TileKey> tileVec(commonTiles.begin(), commonTiles.end());
    if (tileVec.empty()) {
        return;
    }
    struct LocalAccum {
        std::vector<std::map<int32_t, double>> marginals;
        std::vector<std::map<std::pair<int32_t, int32_t>, double>> internalDots;
        std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>> crossDots;
        size_t count = 0;

        explicit LocalAccum(size_t nSets_)
            : marginals(nSets_), internalDots(nSets_) {}
    };
    const bool useParallel = (threads_ > 1 && tileVec.size() > 1);
    const size_t chunkTileCount = useParallel
        ? std::max<size_t>((tileVec.size() + static_cast<size_t>(threads_) - 1) / static_cast<size_t>(threads_), 1)
        : static_cast<size_t>(tileVec.size());
    const size_t nChunks = (tileVec.size() + chunkTileCount - 1) / chunkTileCount;
    std::vector<LocalAccum> partials;
    partials.reserve(nChunks);
    for (size_t i = 0; i < nChunks; ++i) {
        partials.emplace_back(nSets);
    }

    auto processChunk = [&](size_t chunkIdx) {
        const size_t begin = chunkIdx * chunkTileCount;
        const size_t end = std::min(tileVec.size(), begin + chunkTileCount);
        LocalAccum& local = partials[chunkIdx];
        std::vector<std::ifstream> streams(nSources);
        for (size_t ti = begin; ti < end; ++ti) {
            const auto& tile = tileVec[ti];
            std::map<std::pair<int32_t, int32_t>, TopProbs> mergedMap;
            bool first = true;
            for (uint32_t i = 0; i < nSources; ++i) {
                TileOperator* op = opPtrs[i];
                std::map<std::pair<int32_t, int32_t>, TopProbs> currentMap;
                if (op->loadTileToMap(tile, currentMap, nullptr, &streams[i]) == 0) {
                    mergedMap.clear();
                    break;
                }
                if (first) {
                    if (op->getK() > static_cast<int32_t>(k2keep[i])) {
                        for (auto& kv : currentMap) {
                            kv.second.ks.resize(k2keep[i]);
                            kv.second.ps.resize(k2keep[i]);
                        }
                    }
                    mergedMap = std::move(currentMap);
                    first = false;
                    continue;
                }
                auto it = mergedMap.begin();
                while (it != mergedMap.end()) {
                    auto it2 = currentMap.find(it->first);
                    if (it2 == currentMap.end()) {
                        it = mergedMap.erase(it);
                        continue;
                    }
                    it->second.ks.insert(it->second.ks.end(),
                        it2->second.ks.begin(), it2->second.ks.begin() + k2keep[i]);
                    it->second.ps.insert(it->second.ps.end(),
                        it2->second.ps.begin(), it2->second.ps.begin() + k2keep[i]);
                    ++it;
                }
            }
            if (mergedMap.empty()) {
                continue;
            }
            local.count += mergedMap.size();
            for (const auto& kv : mergedMap) {
                const auto& ks = kv.second.ks;
                const auto& ps = kv.second.ps;
                for (size_t s1 = 0; s1 < nSets; ++s1) {
                    const uint32_t off1 = offsets[s1];
                    for (uint32_t i = 0; i < k2keep[s1]; ++i) {
                        const int32_t k1 = ks[off1 + i];
                        const float p1 = ps[off1 + i];
                        if (k1 < 0 || p1 <= 0.0f) { continue; }
                        local.marginals[s1][k1] += p1;
                        for (uint32_t j = i; j < k2keep[s1]; ++j) {
                            const int32_t k2 = ks[off1 + j];
                            const float p2 = ps[off1 + j];
                            if (k2 < 0 || p2 <= 0.0f) { continue; }
                            const auto k12 = (k1 <= k2) ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
                            local.internalDots[s1][k12] += static_cast<double>(p1) * p2;
                        }
                        for (size_t s2 = s1 + 1; s2 < nSets; ++s2) {
                            const uint32_t off2 = offsets[s2];
                            auto& cross = local.crossDots[std::make_pair(s1, s2)];
                            for (uint32_t j = 0; j < k2keep[s2]; ++j) {
                                const int32_t k2 = ks[off2 + j];
                                const float p2 = ps[off2 + j];
                                if (k2 < 0 || p2 <= 0.0f) { continue; }
                                cross[std::make_pair(k1, k2)] += static_cast<double>(p1) * p2;
                            }
                        }
                    }
                }
            }
            notice("%s: Processed tile (%d, %d) with %lu pixels shared by all sources", __func__, tile.row, tile.col, mergedMap.size());
        }
    };

    if (useParallel) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, nChunks),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t chunkIdx = range.begin(); chunkIdx < range.end(); ++chunkIdx) {
                    processChunk(chunkIdx);
                }
            });
    } else {
        processChunk(0);
    }

    for (const auto& local : partials) {
        count += local.count;
        for (size_t s = 0; s < nSets; ++s) {
            add_numeric_map(marginals[s], local.marginals[s]);
            add_numeric_map(internalDots[s], local.internalDots[s]);
        }
        add_nested_map(crossDots, local.crossDots);
    }
}

void TileOperator::probDotTiles3D(const std::set<TileKey>& commonTiles,
    const std::vector<TileOperator*>& opPtrs,
    const std::vector<uint32_t>& k2keep, const std::vector<uint32_t>& offsets,
    std::vector<std::map<int32_t, double>>& marginals,
    std::vector<std::map<std::pair<int32_t, int32_t>, double>>& internalDots,
    std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>>& crossDots, size_t& count) {
    uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    size_t nSets = k2keep.size();
    const std::vector<TileKey> tileVec(commonTiles.begin(), commonTiles.end());
    if (tileVec.empty()) {
        return;
    }
    struct LocalAccum {
        std::vector<std::map<int32_t, double>> marginals;
        std::vector<std::map<std::pair<int32_t, int32_t>, double>> internalDots;
        std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>> crossDots;
        size_t count = 0;

        explicit LocalAccum(size_t nSets_)
            : marginals(nSets_), internalDots(nSets_) {}
    };
    const bool useParallel = (threads_ > 1 && tileVec.size() > 1);
    const size_t chunkTileCount = useParallel
        ? std::max<size_t>((tileVec.size() + static_cast<size_t>(threads_) - 1) / static_cast<size_t>(threads_), 1)
        : static_cast<size_t>(tileVec.size());
    const size_t nChunks = (tileVec.size() + chunkTileCount - 1) / chunkTileCount;
    std::vector<LocalAccum> partials;
    partials.reserve(nChunks);
    for (size_t i = 0; i < nChunks; ++i) {
        partials.emplace_back(nSets);
    }

    auto processChunk = [&](size_t chunkIdx) {
        const size_t begin = chunkIdx * chunkTileCount;
        const size_t end = std::min(tileVec.size(), begin + chunkTileCount);
        LocalAccum& local = partials[chunkIdx];
        std::vector<std::ifstream> streams(nSources);
        for (size_t ti = begin; ti < end; ++ti) {
            const auto& tile = tileVec[ti];
            std::map<PixelKey3, TopProbs> mergedMap;
            bool first = true;
            for (uint32_t i = 0; i < nSources; ++i) {
                TileOperator* op = opPtrs[i];
                std::map<PixelKey3, TopProbs> currentMap;
                if (op->loadTileToMap3D(tile, currentMap, &streams[i]) == 0) {
                    mergedMap.clear();
                    break;
                }
                if (first) {
                    if (op->getK() > static_cast<int32_t>(k2keep[i])) {
                        for (auto& kv : currentMap) {
                            kv.second.ks.resize(k2keep[i]);
                            kv.second.ps.resize(k2keep[i]);
                        }
                    }
                    mergedMap = std::move(currentMap);
                    first = false;
                    continue;
                }
                auto it = mergedMap.begin();
                while (it != mergedMap.end()) {
                    auto it2 = currentMap.find(it->first);
                    if (it2 == currentMap.end()) {
                        it = mergedMap.erase(it);
                        continue;
                    }
                    it->second.ks.insert(it->second.ks.end(),
                        it2->second.ks.begin(), it2->second.ks.begin() + k2keep[i]);
                    it->second.ps.insert(it->second.ps.end(),
                        it2->second.ps.begin(), it2->second.ps.begin() + k2keep[i]);
                    ++it;
                }
            }
            if (mergedMap.empty()) {
                continue;
            }
            local.count += mergedMap.size();
            for (const auto& kv : mergedMap) {
                const auto& ks = kv.second.ks;
                const auto& ps = kv.second.ps;
                for (size_t s1 = 0; s1 < nSets; ++s1) {
                    const uint32_t off1 = offsets[s1];
                    for (uint32_t i = 0; i < k2keep[s1]; ++i) {
                        const int32_t k1 = ks[off1 + i];
                        const float p1 = ps[off1 + i];
                        if (k1 < 0 || p1 <= 0.0f) { continue; }
                        local.marginals[s1][k1] += p1;
                        for (uint32_t j = i; j < k2keep[s1]; ++j) {
                            const int32_t k2 = ks[off1 + j];
                            const float p2 = ps[off1 + j];
                            if (k2 < 0 || p2 <= 0.0f) { continue; }
                            const auto k12 = (k1 <= k2) ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
                            local.internalDots[s1][k12] += static_cast<double>(p1) * p2;
                        }
                        for (size_t s2 = s1 + 1; s2 < nSets; ++s2) {
                            const uint32_t off2 = offsets[s2];
                            auto& cross = local.crossDots[std::make_pair(s1, s2)];
                            for (uint32_t j = 0; j < k2keep[s2]; ++j) {
                                const int32_t k2 = ks[off2 + j];
                                const float p2 = ps[off2 + j];
                                if (k2 < 0 || p2 <= 0.0f) { continue; }
                                cross[std::make_pair(k1, k2)] += static_cast<double>(p1) * p2;
                            }
                        }
                    }
                }
            }
            notice("%s: Processed tile (%d, %d) with %lu pixels shared by all sources", __func__, tile.row, tile.col, mergedMap.size());
        }
    };

    if (useParallel) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, nChunks),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t chunkIdx = range.begin(); chunkIdx < range.end(); ++chunkIdx) {
                    processChunk(chunkIdx);
                }
            });
    } else {
        processChunk(0);
    }

    for (const auto& local : partials) {
        count += local.count;
        for (size_t s = 0; s < nSets; ++s) {
            add_numeric_map(marginals[s], local.marginals[s]);
            add_numeric_map(internalDots[s], local.internalDots[s]);
        }
        add_nested_map(crossDots, local.crossDots);
    }
}

void TileOperator::classifyBlocks(int32_t tileSize) {
    if (blocks_.empty()) return;

    for (auto& b : blocks_) {
        // Check if block is strictly contained in a tile
        float cx = (b.idx.xmin + b.idx.xmax) / 2.0f;
        float cy = (b.idx.ymin + b.idx.ymax) / 2.0f;

        int32_t c = static_cast<int32_t>(std::floor(cx / tileSize));
        int32_t r = static_cast<int32_t>(std::floor(cy / tileSize));

        float tileX0 = c * tileSize;
        float tileX1 = (c + 1) * tileSize;
        float tileY0 = r * tileSize;
        float tileY1 = (r + 1) * tileSize;

        float tol = 1.0f; // 1 unit tolerance

        bool crossesX = (b.idx.xmin < tileX0 - tol) || (b.idx.xmax > tileX1 + tol);
        bool crossesY = (b.idx.ymin < tileY0 - tol) || (b.idx.ymax > tileY1 + tol);

        if (!crossesX && !crossesY) {
            b.contained = true;
            b.row = r;
            b.col = c;
        } else {
            b.contained = false;
        }
    }
}

void TileOperator::initTileGeom(const TileInfo& blk, TileGeom& out) const {
    const bool coordScaled = (mode_ & 0x2) != 0;
    out.key = TileKey{blk.idx.row, blk.idx.col};
    tile2bound(out.key, out.pixX0, out.pixX1, out.pixY0, out.pixY1, formatInfo_.tileSize);
    if (coordScaled) {
        out.pixX0 = coord2pix(out.pixX0);
        out.pixX1 = coord2pix(out.pixX1);
        out.pixY0 = coord2pix(out.pixY0);
        out.pixY1 = coord2pix(out.pixY1);
    }
    if (usesRasterResolutionOverride()) {
        out.pixX0 = mapPixelToRasterFloor(out.pixX0);
        out.pixX1 = mapPixelToRasterCeil(out.pixX1);
        out.pixY0 = mapPixelToRasterFloor(out.pixY0);
        out.pixY1 = mapPixelToRasterCeil(out.pixY1);
    }
    if (out.pixX1 <= out.pixX0 || out.pixY1 <= out.pixY0) {
        error("%s: Invalid raster bounds in tile (%d, %d)", __func__, out.key.row, out.key.col);
    }
    out.W = static_cast<size_t>(out.pixX1 - out.pixX0);
    out.H = static_cast<size_t>(out.pixY1 - out.pixY0);
    if (out.H > 0 && out.W > std::numeric_limits<size_t>::max() / out.H) {
        error("%s: Raster size overflow in tile (%d, %d)", __func__, out.key.row, out.key.col);
    }
}

void TileOperator::loadSoftMaskTileData(const TileInfo& blk, std::ifstream& in,
    float minPixelProb, bool keepRecords, bool keepFactorEntries, bool keepFactorMass, SoftMaskTileData& out,
    uint64_t& nOutOfRangeIgnored, uint64_t& nBadFactorIgnored,
    uint64_t* nCollisionIgnored, std::vector<double>* histGlobal) const {
    initTileGeom(blk, out.geom);
    const size_t nPix = out.geom.W * out.geom.H;
    out.seenLocal.assign(nPix, 0);
    out.factorEntries.clear();
    out.factorMass.clear();
    out.records.clear();
    if (keepRecords) {
        out.records.reserve(static_cast<size_t>(blk.idx.n));
    }

    in.clear();
    in.seekg(static_cast<std::streamoff>(blk.idx.st));
    if (!in.good()) {
        error("%s: Failed seeking input stream to tile (%d, %d)", __func__, out.geom.key.row, out.geom.key.col);
    }

    if (histGlobal && histGlobal->size() != static_cast<size_t>(K_)) {
        error("%s: histGlobal size mismatch", __func__);
    }

    if (usesRasterResolutionOverride()) {
        std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
        loadTileToMap(out.geom.key, pixelMap, nullptr, &in);
        for (const auto& kv : pixelMap) {
            const int32_t xpix = kv.first.first;
            const int32_t ypix = kv.first.second;
            if (xpix < out.geom.pixX0 || xpix >= out.geom.pixX1 ||
                ypix < out.geom.pixY0 || ypix >= out.geom.pixY1) {
                ++nOutOfRangeIgnored;
                continue;
            }
            const size_t x0 = static_cast<size_t>(xpix - out.geom.pixX0);
            const size_t y0 = static_cast<size_t>(ypix - out.geom.pixY0);
            const uint32_t localIdx = static_cast<uint32_t>(y0 * out.geom.W + x0);
            if (out.seenLocal[localIdx]) {
                if (nCollisionIgnored) {
                    ++(*nCollisionIgnored);
                }
                continue;
            }
            out.seenLocal[localIdx] = 1;

            const TopProbs& rec = kv.second;
            for (size_t i = 0; i < rec.ks.size() && i < rec.ps.size(); ++i) {
                const int32_t k = rec.ks[i];
                const float p = rec.ps[i];
                if (k < 0 || k >= K_) {
                    if (k >= 0) {
                        ++nBadFactorIgnored;
                    }
                    continue;
                }
                if (histGlobal) {
                    (*histGlobal)[static_cast<size_t>(k)] += static_cast<double>(p);
                }
                if (p < minPixelProb) {
                    continue;
                }
                if (keepFactorEntries) {
                    out.factorEntries[k].emplace_back(localIdx, p);
                }
                if (keepFactorMass) {
                    out.factorMass[k] += static_cast<double>(p);
                }
            }

            if (keepRecords) {
                SoftMaskSparseRec stored;
                stored.localIdx = localIdx;
                stored.rec = rec;
                out.records.push_back(std::move(stored));
            }
        }
        return;
    }

    TopProbs rec;
    int32_t xpix = 0;
    int32_t ypix = 0;
    uint64_t pos = blk.idx.st;
    while (readNextRecord2DAsPixel(in, pos, blk.idx.ed, xpix, ypix, rec)) {
        if (xpix < out.geom.pixX0 || xpix >= out.geom.pixX1 ||
            ypix < out.geom.pixY0 || ypix >= out.geom.pixY1) {
            ++nOutOfRangeIgnored;
            continue;
        }
        const size_t x0 = static_cast<size_t>(xpix - out.geom.pixX0);
        const size_t y0 = static_cast<size_t>(ypix - out.geom.pixY0);
        const uint32_t localIdx = static_cast<uint32_t>(y0 * out.geom.W + x0);
        if (out.seenLocal[localIdx]) {
            if (nCollisionIgnored) {
                ++(*nCollisionIgnored);
            }
            continue;
        }
        out.seenLocal[localIdx] = 1;

        for (size_t i = 0; i < rec.ks.size() && i < rec.ps.size(); ++i) {
            const int32_t k = rec.ks[i];
            const float p = rec.ps[i];
            if (k < 0 || k >= K_) {
                ++nBadFactorIgnored;
                continue;
            }
            if (histGlobal) {
                (*histGlobal)[static_cast<size_t>(k)] += static_cast<double>(p);
            }
            if (p < minPixelProb) {
                continue;
            }
            if (keepFactorEntries) {
                out.factorEntries[k].emplace_back(localIdx, p);
            }
            if (keepFactorMass) {
                out.factorMass[k] += static_cast<double>(p);
            }
        }

        if (keepRecords) {
            SoftMaskSparseRec stored;
            stored.localIdx = localIdx;
            stored.rec = std::move(rec);
            out.records.push_back(std::move(stored));
        }
    }
}

void TileOperator::buildDenseFactorRaster(const std::vector<std::pair<uint32_t, float>>& entries,
    size_t nPix, std::vector<float>& dense) const {
    dense.assign(nPix, 0.0f);
    for (const auto& kv : entries) {
        dense[kv.first] += kv.second;
    }
}

void TileOperator::applyBinaryMorphologyStep(std::vector<uint8_t>& mask, size_t W, size_t H, int32_t step) const {
    if (mask.size() != W * H) {
        error("%s: Mask size mismatch", __func__);
    }
    if (step == 0) {
        error("%s: morphology step must not be 0", __func__);
    }
    const int32_t kernelSize = std::abs(step);
    if (kernelSize <= 0 || (kernelSize % 2) == 0) {
        error("%s: morphology kernel size must be a positive odd integer", __func__);
    }
    if (kernelSize == 1 || W == 0 || H == 0) {
        return;
    }

    const int32_t radius = (kernelSize - 1) / 2;
    std::vector<uint32_t> summed;
    ::boxFilterSum2D<uint8_t, uint32_t, uint32_t>(mask, W, H, radius, summed);

    std::vector<uint8_t> out(mask.size(), 0);
    const size_t r = static_cast<size_t>(radius);
    for (size_t idx = 0; idx < mask.size(); ++idx) {
        const size_t y = idx / W;
        const size_t x = idx - y * W;
        const size_t x0 = (x > r) ? (x - r) : 0;
        const size_t x1 = std::min(W - 1, x + r);
        const size_t y0 = (y > r) ? (y - r) : 0;
        const size_t y1 = std::min(H - 1, y + r);
        const uint32_t clippedArea = static_cast<uint32_t>((x1 - x0 + 1) * (y1 - y0 + 1));
        if (step > 0) {
            out[idx] = (summed[idx] > 0) ? 1 : 0;
        } else {
            out[idx] = (summed[idx] == clippedArea) ? 1 : 0;
        }
    }
    mask.swap(out);
}

void TileOperator::buildSoftMaskBinary(const std::vector<float>& dense,
    const std::vector<uint8_t>& seenLocal,
    size_t W, size_t H, int32_t radius, double neighborhoodThreshold,
    const SoftMaskThresholdConfig& config,
    std::vector<uint8_t>& mask, std::vector<float>* filteredOut,
    std::vector<float>* observedFilteredOut) const {
    if (dense.size() != W * H || seenLocal.size() != W * H) {
        error("%s: Input size mismatch", __func__);
    }
    std::vector<float> filtered;
    ::boxFilterSum2D<float, float, double>(dense, W, H, radius, filtered);
    if (filteredOut) {
        *filteredOut = filtered;
    }

    mask.assign(W * H, 0);
    if (!config.useObservedDenominator) {
        for (size_t idx = 0; idx < mask.size(); ++idx) {
            mask[idx] = (filtered[idx] >= neighborhoodThreshold) ? 1 : 0;
        }
        if (observedFilteredOut) {
            observedFilteredOut->clear();
        }
    } else {
        std::vector<float> observed(mask.size(), 0.0f);
        for (size_t idx = 0; idx < observed.size(); ++idx) {
            observed[idx] = seenLocal[idx] ? 1.0f : 0.0f;
        }
        std::vector<float> observedFiltered;
        ::boxFilterSum2D<float, float, double>(observed, W, H, radius, observedFiltered);
        if (observedFilteredOut) {
            *observedFilteredOut = observedFiltered;
        }

        const size_t r = static_cast<size_t>(std::max(radius, 0));
        for (size_t idx = 0; idx < mask.size(); ++idx) {
            const size_t y = idx / W;
            const size_t x = idx - y * W;
            const size_t x0 = (x > r) ? (x - r) : 0;
            const size_t x1 = std::min(W - 1, x + r);
            const size_t y0 = (y > r) ? (y - r) : 0;
            const size_t y1 = std::min(H - 1, y + r);
            const double observedCount = static_cast<double>(observedFiltered[idx]);
            const double windowMass = static_cast<double>(filtered[idx]);
            bool sparseEmptyCenter = false;
            if (config.applySparseEmptyCenterRule && !seenLocal[idx]) {
                int32_t nonEmptyAdjacent = 0;
                if (x > 0 && seenLocal[idx - 1]) ++nonEmptyAdjacent;
                if (x + 1 < W && seenLocal[idx + 1]) ++nonEmptyAdjacent;
                if (y > 0 && seenLocal[idx - W]) ++nonEmptyAdjacent;
                if (y + 1 < H && seenLocal[idx + W]) ++nonEmptyAdjacent;
                sparseEmptyCenter = (nonEmptyAdjacent <= 2);
            }
            mask[idx] = (!sparseEmptyCenter && observedCount > 0.0 &&
                windowMass >= config.minWindowMass &&
                windowMass >= neighborhoodThreshold * observedCount) ? 1 : 0;
        }
    }

    for (int32_t step : config.morphologySteps) {
        if (step != 0) {
            applyBinaryMorphologyStep(mask, W, H, step);
        }
    }
}

uint64_t TileOperator::filterMaskByMinComponentArea4(std::vector<uint8_t>& mask, size_t W, size_t H, uint32_t minComponentArea) const {
    if (mask.size() != W * H) {
        error("%s: Mask size mismatch", __func__);
    }
    if (minComponentArea <= 1) {
        uint64_t kept = 0;
        for (uint8_t v : mask) kept += (v != 0);
        return kept;
    }
    std::vector<uint8_t> seen(mask.size(), 0);
    std::vector<uint32_t> queue;
    queue.reserve(std::min<size_t>(mask.size(), 4096));
    uint64_t kept = 0;
    for (uint32_t start = 0; start < mask.size(); ++start) {
        if (mask[start] == 0 || seen[start]) continue;
        queue.clear();
        queue.push_back(start);
        seen[start] = 1;
        for (size_t head = 0; head < queue.size(); ++head) {
            const uint32_t idx = queue[head];
            const size_t x = idx % W;
            const size_t y = idx / W;
            auto tryPush = [&](uint32_t nidx) {
                if (mask[nidx] == 0 || seen[nidx]) return;
                seen[nidx] = 1;
                queue.push_back(nidx);
            };
            if (x > 0) tryPush(idx - 1);
            if (x + 1 < W) tryPush(idx + 1);
            if (y > 0) tryPush(static_cast<uint32_t>(idx - W));
            if (y + 1 < H) tryPush(static_cast<uint32_t>(idx + W));
        }
        if (queue.size() < minComponentArea) {
            for (uint32_t idx : queue) {
                mask[idx] = 0;
            }
        } else {
            kept += static_cast<uint64_t>(queue.size());
        }
    }
    return kept;
}

TileOperator::BinaryMaskCCL TileOperator::buildBinaryMaskCCL4(std::vector<uint8_t>& mask, const std::vector<float>& mass,
    size_t W, size_t H, double minComponentMass) const {
    if (mask.size() != W * H || mass.size() != W * H) {
        error("%s: Mask size mismatch", __func__);
    }
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    BinaryMaskCCL out;
    out.W = W;
    out.H = H;
    out.pixelCid.assign(mask.size(), INVALID);
    if (W == 0 || H == 0) {
        return out;
    }

    std::vector<uint8_t> seen(mask.size(), 0);
    std::vector<uint32_t> queue;
    queue.reserve(std::min<size_t>(mask.size(), 4096));
    for (uint32_t start = 0; start < mask.size(); ++start) {
        if (mask[start] == 0 || seen[start]) continue;
        queue.clear();
        queue.push_back(start);
        seen[start] = 1;
        PixBox box;
        bool touchesBorder = false;
        double compMass = 0.0;
        for (size_t head = 0; head < queue.size(); ++head) {
            const uint32_t idx = queue[head];
            const size_t x = idx % W;
            const size_t y = idx / W;
            box.include(static_cast<int32_t>(x), static_cast<int32_t>(y));
            touchesBorder = touchesBorder || (x == 0 || y == 0 || x + 1 == W || y + 1 == H);
            compMass += static_cast<double>(mass[idx]);
            auto tryPush = [&](uint32_t nidx) {
                if (mask[nidx] == 0 || seen[nidx]) return;
                seen[nidx] = 1;
                queue.push_back(nidx);
            };
            if (x > 0) tryPush(idx - 1);
            if (x + 1 < W) tryPush(idx + 1);
            if (y > 0) tryPush(static_cast<uint32_t>(idx - W));
            if (y + 1 < H) tryPush(static_cast<uint32_t>(idx + W));
        }
        if (compMass < minComponentMass) {
            for (uint32_t idx : queue) {
                mask[idx] = 0;
            }
            continue;
        }
        const uint32_t cid = out.ncomp++;
        out.compArea.push_back(static_cast<uint32_t>(queue.size()));
        out.compBox.push_back(box);
        out.compTouchesBorder.push_back(touchesBorder ? 1 : 0);
        out.keptArea += static_cast<uint64_t>(queue.size());
        for (uint32_t idx : queue) {
            out.pixelCid[idx] = cid;
        }
    }

    out.leftCid.assign(H, INVALID);
    out.rightCid.assign(H, INVALID);
    for (size_t y = 0; y < H; ++y) {
        out.leftCid[y] = out.pixelCid[y * W];
        out.rightCid[y] = out.pixelCid[y * W + (W - 1)];
    }
    out.topCid.assign(W, INVALID);
    out.bottomCid.assign(W, INVALID);
    const size_t bottomRow = (H - 1) * W;
    for (size_t x = 0; x < W; ++x) {
        out.topCid[x] = out.pixelCid[x];
        out.bottomCid[x] = out.pixelCid[bottomRow + x];
    }
    return out;
}

Clipper2Lib::Paths64 TileOperator::normalizeMaskPolygons(const Clipper2Lib::Paths64& paths) const {
    Paths64 trimmed;
    trimmed.reserve(paths.size());
    for (const Path64& path : paths) {
        Path64 path2 = TrimCollinear(path, false);
        if (path2.size() >= 3 && rounded_abs_area64(path2) > 0) {
            trimmed.push_back(std::move(path2));
        }
    }
    if (trimmed.empty()) {
        return {};
    }

    Paths64 cleaned = Clipper2Lib::Union(trimmed, FillRule::NonZero);
    Paths64 out;
    out.reserve(cleaned.size());
    for (const Path64& path : cleaned) {
        Path64 path2 = TrimCollinear(path, false);
        if (path2.size() >= 3 && rounded_abs_area64(path2) > 0) {
            out.push_back(std::move(path2));
        }
    }
    return out;
}

Clipper2Lib::Paths64 TileOperator::cleanupMaskPolygons(const Clipper2Lib::Paths64& paths, uint32_t minHoleArea, double simplifyTolerance) const {
    Paths64 cleaned = normalizeMaskPolygons(paths);
    if (cleaned.empty()) {
        return {};
    }
    if (simplifyTolerance > 0.0) {
        cleaned = SimplifyPaths(cleaned, simplifyTolerance);
        cleaned = normalizeMaskPolygons(cleaned);
    }
    if (minHoleArea == 0) {
        return cleaned;
    }

    PolyTree64 tree;
    Paths64 open_paths;
    Clipper64 clipper;
    clipper.AddSubject(cleaned);
    clipper.Execute(ClipType::Union, FillRule::NonZero, tree, open_paths);
    Paths64 filtered;
    for (const auto& child : tree) {
        append_filtered_polytree_paths(*child, minHoleArea, filtered);
    }
    Paths64 out;
    out.reserve(filtered.size());
    for (const Path64& path : filtered) {
        Path64 path2 = TrimCollinear(path, false);
        if (path2.size() >= 3 && rounded_abs_area64(path2) > 0) {
            out.push_back(std::move(path2));
        }
    }
    return out;
}

Clipper2Lib::Paths64 TileOperator::buildMaskComponentRuns(const BinaryMaskCCL& ccl, uint32_t cid, const TileGeom& geom) const {
    if (cid >= ccl.ncomp) {
        error("%s: Component id out of range", __func__);
    }
    Paths64 runs;
    const PixBox& box = ccl.compBox[cid];
    if (box.minX > box.maxX || box.minY > box.maxY) {
        return {};
    }
    runs.reserve(static_cast<size_t>(box.maxY - box.minY + 1));
    for (int32_t ly = box.minY; ly <= box.maxY; ++ly) {
        const size_t row = static_cast<size_t>(ly) * ccl.W;
        int32_t lx = box.minX;
        while (lx <= box.maxX) {
            const size_t idx = row + static_cast<size_t>(lx);
            if (ccl.pixelCid[idx] != cid) {
                ++lx;
                continue;
            }
            int32_t lx1 = lx + 1;
            while (lx1 <= box.maxX &&
                   ccl.pixelCid[row + static_cast<size_t>(lx1)] == cid) {
                ++lx1;
            }
            const int64_t gx0 = static_cast<int64_t>(geom.pixX0 + lx);
            const int64_t gx1 = static_cast<int64_t>(geom.pixX0 + lx1);
            const int64_t gy0 = static_cast<int64_t>(geom.pixY0 + ly);
            const int64_t gy1 = gy0 + 1;
            runs.push_back(Path64{
                Point64(gx0, gy0),
                Point64(gx1, gy0),
                Point64(gx1, gy1),
                Point64(gx0, gy1)
            });
            lx = lx1;
        }
    }
    return runs;
}

Clipper2Lib::Paths64 TileOperator::buildMaskComponentPolygons(const BinaryMaskCCL& ccl, uint32_t cid, const TileGeom& geom,
    uint32_t minHoleArea, double simplifyTolerance) const {
    return cleanupMaskPolygons(buildMaskComponentRuns(ccl, cid, geom), minHoleArea, simplifyTolerance);
}

Clipper2Lib::Paths64 TileOperator::buildLabelComponentPolygons(const std::vector<uint32_t>& pixelCid, size_t W, uint32_t cid,
    const PixBox& box, const TileGeom& geom, uint32_t minHoleArea, double simplifyTolerance) const {
    if (W == 0 || pixelCid.empty()) {
        return {};
    }
    Paths64 runs;
    if (box.minX > box.maxX || box.minY > box.maxY) {
        return {};
    }
    runs.reserve(static_cast<size_t>(box.maxY - box.minY + 1));
    for (int32_t gy = box.minY; gy <= box.maxY; ++gy) {
        const int32_t ly = gy - geom.pixY0;
        if (ly < 0) continue;
        const size_t row = static_cast<size_t>(ly) * W;
        int32_t gx = box.minX;
        while (gx <= box.maxX) {
            const int32_t lx = gx - geom.pixX0;
            if (lx < 0) {
                ++gx;
                continue;
            }
            const size_t idx = row + static_cast<size_t>(lx);
            if (idx >= pixelCid.size() || pixelCid[idx] != cid) {
                ++gx;
                continue;
            }
            int32_t gx1 = gx + 1;
            while (gx1 <= box.maxX) {
                const int32_t lx1 = gx1 - geom.pixX0;
                if (lx1 < 0) break;
                const size_t idx1 = row + static_cast<size_t>(lx1);
                if (idx1 >= pixelCid.size() || pixelCid[idx1] != cid) break;
                ++gx1;
            }
            runs.push_back(Path64{
                Point64(static_cast<int64_t>(gx), static_cast<int64_t>(gy)),
                Point64(static_cast<int64_t>(gx1), static_cast<int64_t>(gy)),
                Point64(static_cast<int64_t>(gx1), static_cast<int64_t>(gy + 1)),
                Point64(static_cast<int64_t>(gx), static_cast<int64_t>(gy + 1))
            });
            gx = gx1;
        }
    }
    return cleanupMaskPolygons(runs, minHoleArea, simplifyTolerance);
}

nlohmann::json TileOperator::maskPathsToMultiPolygonGeoJSON(const Clipper2Lib::Paths64& paths) const {
    json multipolygon = json::array();
    if (paths.empty()) {
        return json{{"type", "MultiPolygon"}, {"coordinates", multipolygon}};
    }
    const double scale = static_cast<double>(getRasterPixelResolution() > 0.0f ? getRasterPixelResolution() : 1.0f);
    PolyTree64 tree;
    Paths64 open_paths;
    Clipper64 clipper;
    clipper.AddSubject(paths);
    clipper.Execute(ClipType::Union, FillRule::NonZero, tree, open_paths);
    for (const auto& child : tree) {
        append_geojson_polygons(*child, scale, multipolygon);
    }
    return json{{"type", "MultiPolygon"}, {"coordinates", std::move(multipolygon)}};
}

int32_t TileOperator::floorDivInt32(int32_t value, int32_t divisor) {
    if (divisor <= 0) {
        error("%s: divisor must be positive", __func__);
    }
    int32_t q = value / divisor;
    const int32_t r = value % divisor;
    if (r < 0) {
        --q;
    }
    return q;
}

int32_t TileOperator::ceilDivInt32(int32_t value, int32_t divisor) {
    return -floorDivInt32(-value, divisor);
}

int32_t TileOperator::mapPixelToRasterFloor(int32_t value) const {
    if (!hasRasterResolutionOverride_) {
        return value;
    }
    return floorDivInt32(value, rasterRatioXY_);
}

int32_t TileOperator::mapPixelToRasterCeil(int32_t value) const {
    if (!hasRasterResolutionOverride_) {
        return value;
    }
    return ceilDivInt32(value, rasterRatioXY_);
}
