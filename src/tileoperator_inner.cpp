#include "tileoperator.hpp"
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

void TileOperator::loadDenseTile(const TileInfo& blk, std::ifstream& in, DenseTile& out, uint8_t bg,
    uint64_t& nOutOfRangeIgnored, uint64_t& nBadLabelIgnored) const {
    if (coord_dim_ != 2) {
        error("%s: Only 2D records are supported", __func__);
    }
    const bool coordScaled = (mode_ & 0x2) != 0;
    out.key = TileKey{blk.idx.row, blk.idx.col};
    tile2bound(out.key, out.pixX0, out.pixX1, out.pixY0, out.pixY1, formatInfo_.tileSize);
    if (coordScaled) {
        out.pixX0 = coord2pix(out.pixX0);
        out.pixX1 = coord2pix(out.pixX1);
        out.pixY0 = coord2pix(out.pixY0);
        out.pixY1 = coord2pix(out.pixY1);
    }
    if (out.pixX1 <= out.pixX0 || out.pixY1 <= out.pixY0) {
        error("%s: Invalid raster bounds in tile (%d, %d)", __func__, out.key.row, out.key.col);
    }
    out.W = static_cast<size_t>(out.pixX1 - out.pixX0);
    out.H = static_cast<size_t>(out.pixY1 - out.pixY0);
    if (out.H > 0 && out.W > std::numeric_limits<size_t>::max() / out.H) {
        error("%s: Raster size overflow in tile (%d, %d)", __func__, out.key.row, out.key.col);
    }
    const size_t nPix = out.W * out.H;
    out.lab.assign(nPix, bg);
    out.boundary.clear();

    in.clear();
    in.seekg(blk.idx.st);
    if (!in.good()) {
        error("%s: Failed seeking input stream to tile (%d, %d)", __func__, out.key.row, out.key.col);
    }

    TopProbs rec;
    int32_t xpix = 0;
    int32_t ypix = 0;
    uint64_t pos = blk.idx.st;
    while (readNextRecord2DAsPixel(in, pos, blk.idx.ed, xpix, ypix, rec)) {
        if (rec.ks.empty() || rec.ps.empty()) {
            continue;
        }
        if (xpix < out.pixX0 || xpix >= out.pixX1 || ypix < out.pixY0 || ypix >= out.pixY1) {
            ++nOutOfRangeIgnored;
            continue;
        }
        const int32_t k = rec.ks[0];
        if (k < 0 || k >= static_cast<int32_t>(bg)) {
            ++nBadLabelIgnored;
            continue;
        }
        const size_t x0 = static_cast<size_t>(xpix - out.pixX0);
        const size_t y0 = static_cast<size_t>(ypix - out.pixY0);
        out.lab[y0 * out.W + x0] = static_cast<uint8_t>(k);
    }
}

TileOperator::TileCCL TileOperator::tileLocalCCL(const DenseTile& tile, uint8_t bg, const std::vector<uint8_t>* boundaryMask) const {
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    TileCCL out;
    out.pixX0 = tile.pixX0;
    out.pixX1 = tile.pixX1;
    out.pixY0 = tile.pixY0;
    out.pixY1 = tile.pixY1;
    if (tile.W == 0 || tile.H == 0) {
        return out;
    }
    const size_t nPix = tile.W * tile.H;
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
    for (size_t y = 0; y < tile.H; ++y) {
        const size_t row = y * tile.W;
        for (size_t x = 0; x < tile.W; ++x) {
            const size_t idx = row + x;
            const uint8_t lbl = tile.lab[idx];
            if (lbl == bg) continue;
            parent[idx] = static_cast<uint32_t>(idx);
            if (x > 0 && tile.lab[idx - 1] == lbl) {
                unite(static_cast<uint32_t>(idx), static_cast<uint32_t>(idx - 1));
            }
            if (y > 0 && tile.lab[idx - tile.W] == lbl) {
                unite(static_cast<uint32_t>(idx), static_cast<uint32_t>(idx - tile.W));
            }
        }
    }
    // 2nd pass: pixel -> compact component ID
    std::vector<uint32_t> root2cid(nPix, INVALID);
    for (size_t idx = 0; idx < nPix; ++idx) {
        if (parent[idx] == INVALID) continue;
        uint32_t r = findRoot(static_cast<uint32_t>(idx));
        uint32_t cid = root2cid[r];
        const size_t y = idx / tile.W;
        const size_t x = idx - y * tile.W;
        const int32_t gx = tile.pixX0 + static_cast<int32_t>(x);
        const int32_t gy = tile.pixY0 + static_cast<int32_t>(y);
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
    }
    out.ncomp = static_cast<uint32_t>(out.compSize.size());
    auto cidAt = [&](size_t idx) -> uint32_t {
        if (parent[idx] == INVALID) return INVALID;
        uint32_t r = findRoot((uint32_t)idx);
        return root2cid[r];
    };
    // borders
    out.leftCid.assign(tile.H, INVALID);
    out.rightCid.assign(tile.H, INVALID);
    for (size_t y = 0; y < tile.H; ++y) {
        const size_t row = y * tile.W;
        out.leftCid[y] = cidAt(row);
        out.rightCid[y] = cidAt(row + tile.W - 1);
    }
    out.topCid.assign(tile.W, INVALID);
    out.bottomCid.assign(tile.W, INVALID);
    const size_t bottomRow = (tile.H - 1) * tile.W;
    for (size_t x = 0; x < tile.W; ++x) {
        out.topCid[x] = cidAt(x);
        out.bottomCid[x] = cidAt(bottomRow + x);
    }
    if (boundaryMask != nullptr) {
        if (boundaryMask->size() != nPix) {
            error("%s: Boundary mask size mismatch", __func__);
        }
        out.bndPix.reserve(tile.W * 2 + tile.H * 2);
        out.bndCid.reserve(tile.W * 2 + tile.H * 2);
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
        const size_t nPix = tile.W * tile.H;
        tile.boundary.assign(nPix, 0);
        for (size_t y = 0; y < tile.H; ++y) {
            const size_t row = y * tile.W;
            for (size_t x = 0; x < tile.W; ++x) {
                const size_t idx = row + x;
                const uint8_t c = tile.lab[idx];
                bool isBoundary = false;
                if (x > 0 && tile.lab[idx - 1] != c) isBoundary = true;
                if (x + 1 < tile.W && tile.lab[idx + 1] != c) isBoundary = true;
                if (y > 0 && tile.lab[idx - tile.W] != c) isBoundary = true;
                if (y + 1 < tile.H && tile.lab[idx + tile.W] != c) isBoundary = true;
                tile.boundary[idx] = isBoundary ? 1 : 0;
            }
        }
    }
    // right/down inter-tile boundaries
    for (size_t i = 0; i < tiles.size(); ++i) {
        const TileKey key = tiles[i].key;
        const auto rightIt = tile_lookup_.find(TileKey{key.row, key.col + 1});
        if (rightIt != tile_lookup_.end()) {
            size_t j = rightIt->second;
            if (j < tiles.size() && tiles[i].W > 0 && tiles[j].W > 0) {
                auto& a = tiles[i];
                auto& b = tiles[j];
                const int32_t y0 = std::max(a.pixY0, b.pixY0);
                const int32_t y1 = std::min(a.pixY1, b.pixY1);
                for (int32_t gy = y0; gy < y1; ++gy) {
                    const size_t ay = static_cast<size_t>(gy - a.pixY0);
                    const size_t by = static_cast<size_t>(gy - b.pixY0);
                    const size_t ia = ay * a.W + (a.W - 1);
                    const size_t ib = by * b.W;
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
            if (j < tiles.size() && tiles[i].H > 0 && tiles[j].H > 0) {
                auto& a = tiles[i];
                auto& b = tiles[j];
                const int32_t x0 = std::max(a.pixX0, b.pixX0);
                const int32_t x1 = std::min(a.pixX1, b.pixX1);
                for (int32_t gx = x0; gx < x1; ++gx) {
                    const size_t ax = static_cast<size_t>(gx - a.pixX0);
                    const size_t bx = static_cast<size_t>(gx - b.pixX0);
                    const size_t ia = (a.H - 1) * a.W + ax;
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

void TileOperator::mergeTiles2D(const std::set<TileKey>& commonTiles,
    const std::vector<TileOperator*>& opPtrs,
    const std::vector<uint32_t>& k2keep, bool binaryOutput,
    FILE* fp, int fdMain, int fdIndex, long& currentOffset) {
    uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    for (const auto& tile : commonTiles) {
        std::map<std::pair<int32_t, int32_t>, TopProbs> mergedMap;
        bool first = true;
        for (uint32_t i = 0; i < nSources; ++i) {
            TileOperator* op = opPtrs[i];
            std::map<std::pair<int32_t, int32_t>, TopProbs> currentMap;
            if (op->loadTileToMap(tile, currentMap) == 0) {
                warning("%s: Tile (%d, %d) has no data in source %d", __func__, tile.row, tile.col, i);
                mergedMap.clear();
                break;
            }
            notice("%s: Loaded tile (%d, %d) from source %d with %lu pixels",
                   __func__, tile.row, tile.col, i, currentMap.size());
            if (first) {
                if (k_ > k2keep[i]) { // Trim to k2keep[i]
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
                    it = mergedMap.erase(it); // Intersect
                    continue;
                }
                // Concatenate
                it->second.ks.insert(it->second.ks.end(),
                    it2->second.ks.begin(), it2->second.ks.begin() + k2keep[i]);
                it->second.ps.insert(it->second.ps.end(),
                    it2->second.ps.begin(), it2->second.ps.begin() + k2keep[i]);
                ++it;
            }
        }
        notice("%s: Merged tile (%d, %d) with %lu pixels shared by all sources", __func__, tile.row, tile.col, mergedMap.size());
        if (mergedMap.empty()) {
            continue;
        }
        // Write tile
        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n  = mergedMap.size();
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, formatInfo_.tileSize);
        for (const auto& kv : mergedMap) {
            const auto& p = kv.second;
            const auto& key = kv.first;
            if (binaryOutput) {
                PixTopProbs<int32_t> outRec;
                outRec.x = key.first;
                outRec.y = key.second;
                outRec.ks = p.ks;
                outRec.ps = p.ps;
                if (outRec.write(fdMain) < 0) error("Write error");
            } else {
                fprintf(fp, "%d\t%d", key.first, key.second);
                for (size_t i = 0; i < p.ks.size(); ++i) {
                    fprintf(fp, "\t%d\t%.4e", p.ks[i], p.ps[i]);
                }
                fprintf(fp, "\n");
            }
        }
        if (binaryOutput) {
            currentOffset = lseek(fdMain, 0, SEEK_CUR);
        } else {
            currentOffset = ftell(fp);
        }
        newEntry.ed = currentOffset;
        if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index entry write error");
    }
}

void TileOperator::mergeTiles3D(const std::set<TileKey>& commonTiles,
    const std::vector<TileOperator*>& opPtrs,
    const std::vector<uint32_t>& k2keep, bool binaryOutput,
    FILE* fp, int fdMain, int fdIndex, long& currentOffset) {
    uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    for (const auto& tile : commonTiles) {
        std::map<PixelKey3, TopProbs> mergedMap;
        bool first = true;
        for (uint32_t i = 0; i < nSources; ++i) {
            TileOperator* op = opPtrs[i];
            std::map<PixelKey3, TopProbs> currentMap;
            if (op->loadTileToMap3D(tile, currentMap) == 0) {
                warning("%s: Tile (%d, %d) has no data in source %d", __func__, tile.row, tile.col, i);
                mergedMap.clear();
                break;
            }
            notice("%s: Loaded tile (%d, %d) from source %d with %lu pixels",
                   __func__, tile.row, tile.col, i, currentMap.size());
            if (first) {
                if (k_ > k2keep[i]) { // Trim to k2keep[i]
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
                    it = mergedMap.erase(it); // Intersect
                    continue;
                }
                // Concatenate
                it->second.ks.insert(it->second.ks.end(),
                    it2->second.ks.begin(), it2->second.ks.begin() + k2keep[i]);
                it->second.ps.insert(it->second.ps.end(),
                    it2->second.ps.begin(), it2->second.ps.begin() + k2keep[i]);
                ++it;
            }
        }
        notice("%s: Merged tile (%d, %d) with %lu pixels shared by all sources", __func__, tile.row, tile.col, mergedMap.size());
        if (mergedMap.empty()) {
            continue;
        }
        // Write tile
        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n  = mergedMap.size();
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, formatInfo_.tileSize);
        for (const auto& kv : mergedMap) {
            const auto& p = kv.second;
            const auto& key = kv.first;
            if (binaryOutput) {
                PixTopProbs3D<int32_t> outRec;
                outRec.x = std::get<0>(key);
                outRec.y = std::get<1>(key);
                outRec.z = std::get<2>(key);
                outRec.ks = p.ks;
                outRec.ps = p.ps;
                if (outRec.write(fdMain) < 0) error("Write error");
            } else {
                fprintf(fp, "%d\t%d\t%d", std::get<0>(key), std::get<1>(key), std::get<2>(key));
                for (size_t i = 0; i < p.ks.size(); ++i) {
                    fprintf(fp, "\t%d\t%.4e", p.ks[i], p.ps[i]);
                }
                fprintf(fp, "\n");
            }
        }
        if (binaryOutput) {
            currentOffset = lseek(fdMain, 0, SEEK_CUR);
        } else {
            currentOffset = ftell(fp);
        }
        newEntry.ed = currentOffset;
        if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index entry write error");
    }
}

void TileOperator::annotateTiles2D(const std::vector<TileKey>& tiles,
    TileReader& reader, uint32_t icol_x, uint32_t icol_y,
    uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset) {
    notice("%s: Start annotating query with %lu tiles", __func__, tiles.size());
    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;
    for (const auto& tile : tiles) {
        std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
        if (loadTileToMap(tile, pixelMap) <= 0) {
            debug("%s: Query tile (%d, %d) is not in the annotation dataset", __func__, tile.row, tile.col);
            continue;
        }
        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) continue;
        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n = 0;
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, formatInfo_.tileSize);
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
            fprintf(fp, "%s", s.c_str());
            for (size_t k = 0; k < pit->second.ks.size(); ++k) {
                fprintf(fp, "\t%d\t%.4e", pit->second.ks[k], pit->second.ps[k]);
            }
            fprintf(fp, "\n");
            newEntry.n++;
        }
        notice("%s: Annotated tile (%d, %d) with %u points", __func__, tile.row, tile.col, newEntry.n);
        currentOffset = ftell(fp);
        newEntry.ed = currentOffset;
        if (newEntry.n > 0) {
             if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index entry write error");
        }
    }
}

void TileOperator::annotateTiles3D(const std::vector<TileKey>& tiles,
    TileReader& reader, uint32_t icol_x, uint32_t icol_y, uint32_t icol_z,
    uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset) {
    notice("%s: Start annotating query with %lu tiles", __func__, tiles.size());
    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;
    for (const auto& tile : tiles) {
        std::map<PixelKey3, TopProbs> pixelMap3d;
        if (loadTileToMap3D(tile, pixelMap3d) <= 0) {
            debug("%s: Query tile (%d, %d) is not in the annotation dataset", __func__, tile.row, tile.col);
            continue;
        }
        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) continue;
        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n = 0;
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, formatInfo_.tileSize);
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
            fprintf(fp, "%s", s.c_str());
            for (size_t k = 0; k < pit->second.ks.size(); ++k) {
                fprintf(fp, "\t%d\t%.4e", pit->second.ks[k], pit->second.ps[k]);
            }
            fprintf(fp, "\n");
            newEntry.n++;
        }
        notice("%s: Annotated tile (%d, %d) with %u points", __func__, tile.row, tile.col, newEntry.n);
        currentOffset = ftell(fp);
        newEntry.ed = currentOffset;
        if (newEntry.n > 0) {
             if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index entry write error");
        }
    }
}

void TileOperator::probDotTiles2D(const std::set<TileKey>& commonTiles,
    const std::vector<TileOperator*>& opPtrs,
    const std::vector<uint32_t>& k2keep, const std::vector<uint32_t>& offsets,
    std::vector<std::map<int32_t, double>>& marginals,
    std::vector<std::map<std::pair<int32_t, int32_t>, double>>& internalDots,
    std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>>& crossDots, size_t& count) {
    uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    size_t nSets = k2keep.size();
    for (const auto& tile : commonTiles) {
        std::map<std::pair<int32_t, int32_t>, TopProbs> mergedMap;
        bool first = true;

        for (uint32_t i = 0; i < nSources; ++i) {
            TileOperator* op = opPtrs[i];
            std::map<std::pair<int32_t, int32_t>, TopProbs> currentMap;
            if (op->loadTileToMap(tile, currentMap) == 0) { // Should not happen
                 mergedMap.clear();
                 break;
            }
            if (first) {
                if (op->getK() > (int32_t)k2keep[i]) { // Trim to k2keep[i]
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
                    it = mergedMap.erase(it); // Intersect
                    continue;
                }
                // Concatenate
                it->second.ks.insert(it->second.ks.end(),
                    it2->second.ks.begin(), it2->second.ks.begin() + k2keep[i]);
                it->second.ps.insert(it->second.ps.end(),
                    it2->second.ps.begin(), it2->second.ps.begin() + k2keep[i]);
                ++it;
            }
        }
        if (mergedMap.empty()) continue;
        count += mergedMap.size();

        // Accumulate stats
        for (const auto& kv : mergedMap) {
            const auto& ks = kv.second.ks;
            const auto& ps = kv.second.ps;
            for (size_t s1 = 0; s1 < nSets; ++s1) {
                uint32_t off1 = offsets[s1];
                for (uint32_t i = 0; i < k2keep[s1]; ++i) {
                    int32_t k1 = ks[off1 + i];
                    float p1 = ps[off1 + i];
                    marginals[s1][k1] += p1;
                    // Internal
                    for (uint32_t j = i; j < k2keep[s1]; ++j) {
                        int32_t k2 = ks[off1 + j];
                        float p2 = ps[off1 + j];
                        std::pair<int32_t, int32_t> k12 = (k1 <= k2) ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
                        internalDots[s1][k12] += (double)p1 * p2;
                    }
                    // Cross
                    for (size_t s2 = s1 + 1; s2 < nSets; ++s2) {
                        uint32_t off2 = offsets[s2];
                        for (uint32_t j = 0; j < k2keep[s2]; ++j) {
                            int32_t k2 = ks[off2 + j];
                            float p2 = ps[off2 + j];
                            crossDots[std::make_pair(s1, s2)][std::make_pair(k1, k2)] += (double)p1 * p2;
                        }
                    }
                }
            }
        }
        notice("%s: Processed tile (%d, %d) with %lu pixels shared by all sources", __func__, tile.row, tile.col, mergedMap.size());
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
    for (const auto& tile : commonTiles) {
        std::map<PixelKey3, TopProbs> mergedMap;
        bool first = true;

        for (uint32_t i = 0; i < nSources; ++i) {
            TileOperator* op = opPtrs[i];
            std::map<PixelKey3, TopProbs> currentMap;
            if (op->loadTileToMap3D(tile, currentMap) == 0) { // Should not happen
                 mergedMap.clear();
                 break;
            }
            if (first) {
                if (op->getK() > (int32_t)k2keep[i]) { // Trim to k2keep[i]
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
                    it = mergedMap.erase(it); // Intersect
                    continue;
                }
                // Concatenate
                it->second.ks.insert(it->second.ks.end(),
                    it2->second.ks.begin(), it2->second.ks.begin() + k2keep[i]);
                it->second.ps.insert(it->second.ps.end(),
                    it2->second.ps.begin(), it2->second.ps.begin() + k2keep[i]);
                ++it;
            }
        }
        if (mergedMap.empty()) continue;
        count += mergedMap.size();

        // Accumulate stats
        for (const auto& kv : mergedMap) {
            const auto& ks = kv.second.ks;
            const auto& ps = kv.second.ps;
            for (size_t s1 = 0; s1 < nSets; ++s1) {
                uint32_t off1 = offsets[s1];
                for (uint32_t i = 0; i < k2keep[s1]; ++i) {
                    int32_t k1 = ks[off1 + i];
                    float p1 = ps[off1 + i];
                    marginals[s1][k1] += p1;
                    // Internal
                    for (uint32_t j = i; j < k2keep[s1]; ++j) {
                        int32_t k2 = ks[off1 + j];
                        float p2 = ps[off1 + j];
                        std::pair<int32_t, int32_t> k12 = (k1 <= k2) ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
                        internalDots[s1][k12] += (double)p1 * p2;
                    }
                    // Cross
                    for (size_t s2 = s1 + 1; s2 < nSets; ++s2) {
                        uint32_t off2 = offsets[s2];
                        for (uint32_t j = 0; j < k2keep[s2]; ++j) {
                            int32_t k2 = ks[off2 + j];
                            float p2 = ps[off2 + j];
                            crossDots[std::make_pair(s1, s2)][std::make_pair(k1, k2)] += (double)p1 * p2;
                        }
                    }
                }
            }
        }
        notice("%s: Processed tile (%d, %d) with %lu pixels shared by all sources", __func__, tile.row, tile.col, mergedMap.size());
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
