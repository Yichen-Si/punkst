#include "tileoperator.hpp"
#include "numerical_utils.hpp"
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <set>
#include <memory>

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
