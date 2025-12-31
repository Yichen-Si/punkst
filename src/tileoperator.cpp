#include "tileoperator.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <memory>

void TileOperator::mergeTiles2D(const std::set<TileKey>& commonTiles,
    const std::vector<TileOperator*>& opPtrs,
    const std::vector<uint32_t>& k2keep,
    bool binaryOutput,
    FILE* fp, int fdMain, int fdIndex,
    long& currentOffset) {
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
    const std::vector<uint32_t>& k2keep,
    bool binaryOutput,
    FILE* fp, int fdMain, int fdIndex,
    long& currentOffset) {
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

void TileOperator::merge(const std::vector<std::string>& otherFiles, const std::string& outPrefix, std::vector<uint32_t> k2keep, bool binaryOutput) {
    std::string outIndex = outPrefix + ".index";
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());
    if (k2keep.size() > 0) {assert(k2keep.size() == otherFiles.size() + 1);}

    // 1. Setup operators
    std::vector<std::unique_ptr<TileOperator>> ops;
    std::vector<TileOperator*> opPtrs;
    opPtrs.push_back(this); // current object
    for (const auto& f : otherFiles) {
        std::string idxFile = f.substr(0, f.find_last_of('.')) + ".index";
        struct stat buffer;
        if (stat(idxFile.c_str(), &buffer) != 0) {
            error("%s: Index file %s not found", __func__, idxFile.c_str());
        }
        std::string df = f;
        ops.push_back(std::make_unique<TileOperator>(df, idxFile));
        opPtrs.push_back(ops.back().get());
    }
    uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    if (k2keep.size() == 0) {
        for (auto* op : opPtrs) {
            k2keep.push_back(op->getK());
        }
    } else {
        for (uint32_t i = 0; i < nSources; ++i) {
            if (k2keep[i] > opPtrs[i]->getK()) {
                warning("%s: Invalid value k (%d) specified for the %d-th source", __func__, k2keep[i], i);
                k2keep[i] = opPtrs[i]->getK();
            }
        }
    }
    if (nSources > 7) {
        int32_t k = *std::min_element(k2keep.begin(), k2keep.end());
        k2keep.assign(nSources, k);
        warning("%s: More than 7 files to merge, keep %d values each", __func__, k);
    }
    int32_t totalK = std::accumulate(k2keep.begin(), k2keep.end(), 0);
    bool use3d = (coord_dim_ == 3);
    for (auto* op : opPtrs) {
        if (op->coord_dim_ != coord_dim_) {
            error("%s: Mixed 2D/3D inputs are not supported", __func__);
        }
    }

    // 2. Identify common tiles (Intersection)
    std::set<TileKey> commonTiles;
    if (opPtrs[0]->tile_lookup_.empty()) {
        warning("%s: No tiles in the base dataset", __func__);
        return;
    }
    for (const auto& kv : opPtrs[0]->tile_lookup_) {
        commonTiles.insert(kv.first);
    }
    for (uint32_t i = 1; i < nSources; ++i) {
        std::set<TileKey> currentTiles;
        for (const auto& kv : opPtrs[i]->tile_lookup_) {
            if (commonTiles.count(kv.first)) {
                currentTiles.insert(kv.first);
            }
        }
        commonTiles = currentTiles;
    }
    if (commonTiles.empty()) {
        warning("%s: No overlapping tiles found for merge", __func__);
        return;
    }

    // 3. Prepare output
    std::string outFile;
    FILE* fp = nullptr;
    int fdMain = -1;
    long currentOffset = 0;

    // Index metadata
    IndexHeader idxHeader = formatInfo_;
    idxHeader.mode |= 0x4; // int32
    idxHeader.coordType = 1;
    if (!idxHeader.packKvec(k2keep)) {
        warning("%s: Too many input fields", __func__);
    }

    if (binaryOutput) {
        outFile = outPrefix + ".bin";
        fdMain = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (fdMain < 0) error("Cannot open output file %s", outFile.c_str());
        idxHeader.mode |= 0x1; // Binary mode
        uint32_t coordCount = use3d ? 3 : 2;
        idxHeader.recordSize = sizeof(int32_t) * coordCount + sizeof(int32_t) * totalK + sizeof(float) * totalK;
        currentOffset = 0;
    } else {
        outFile = outPrefix + ".tsv";
        fp = fopen(outFile.c_str(), "w");
        if (!fp) error("Cannot open output file %s", outFile.c_str());
        idxHeader.mode &= ~0x1; // Text mode
        idxHeader.recordSize = 0;

        // TSV header
        std::string headerStr = "#x\ty";
        if (use3d) {
            headerStr += "\tz";
        }
        uint32_t idx = 1;
        for (uint32_t i = 0; i < nSources; ++i) {
            for (int j = 0; j < k2keep[i]; ++j) {
                headerStr += "\tK" + std::to_string(idx) + "\tP" + std::to_string(idx);
                idx++;
            }
        }
        headerStr += "\n";
        fprintf(fp, "%s", headerStr.c_str());
        currentOffset = ftell(fp);
    }

    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) error("Index write error");

    // 4. Process tiles
    notice("%s: Start merging %u files", __func__, nSources);
    if (use3d) {
        mergeTiles3D(commonTiles, opPtrs, k2keep, binaryOutput, fp, fdMain, fdIndex, currentOffset);
    } else {
        mergeTiles2D(commonTiles, opPtrs, k2keep, binaryOutput, fp, fdMain, fdIndex, currentOffset);
    }

    if (binaryOutput) {
        close(fdMain);
    } else {
        fclose(fp);
    }
    close(fdIndex);
    notice("Merged %u files (%lu shared tiles) to %s", nSources, commonTiles.size(), outFile.c_str());
}

void TileOperator::annotate(const std::string& ptPrefix, const std::string& outPrefix, uint32_t icol_x, uint32_t icol_y, uint32_t icol_z) {
    std::string ptData = ptPrefix + ".tsv";
    std::string ptIndex = ptPrefix + ".index";
    TileReader reader(ptData, ptIndex);
    assert(reader.getTileSize() == formatInfo_.tileSize);
    assert(icol_x != icol_y);
    if (coord_dim_ == 3) {assert(icol_z != std::numeric_limits<uint32_t>::max() && icol_z != icol_x && icol_z != icol_y);}
    bool use3d = (coord_dim_ == 3);
    std::string outFile = outPrefix + ".tsv";
    std::string outIndex = outPrefix + ".index";
    FILE* fp = fopen(outFile.c_str(), "w");
    if (!fp) error("Cannot open output file %s", outFile.c_str());
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());
    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;
    uint32_t ntok = std::max(icol_x, icol_y);
    if (use3d) {ntok = std::max(ntok, icol_z);}
    ntok += 1;
    // Header?
    if (!reader.headerLine.empty()) {
        std::string headerStr = reader.headerLine;
        for (uint32_t i = 1; i <= k_; ++i) {
            headerStr += "\tK" + std::to_string(i) + "\tP" + std::to_string(i);
        }
        fprintf(fp, "%s\n", headerStr.c_str());
    }
    long currentOffset = ftell(fp);
    // Write index header
    IndexHeader idxHeader = formatInfo_;
    idxHeader.mode &= ~(0x7);
    idxHeader.recordSize = 0;
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) error("Index header write error");

    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    if (use3d) {
        annotateTiles3D(tiles, reader, icol_x, icol_y, icol_z, ntok, fp, fdIndex, currentOffset);
    } else {
        annotateTiles2D(tiles, reader, icol_x, icol_y, ntok, fp, fdIndex, currentOffset);
    }

    fclose(fp);
    close(fdIndex);
    notice("Annotation finished, data written to %s", outFile.c_str());
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

void TileOperator::reorgTiles(const std::string& outPrefix, int32_t tileSize) {
    if (blocks_.empty()) {
        error("No blocks found in index");
    }
    if (tileSize <= 0) {
        tileSize = formatInfo_.tileSize;
    }
    assert(tileSize > 0);

    classifyBlocks(tileSize);
    openDataStream();

    if (mode_ & 0x1) {reorgTilesBinary(outPrefix, tileSize); return;}

    std::map<TileKey, std::vector<size_t>> tileMainBlocks;
    std::map<TileKey, std::vector<std::string>> boundaryLines;
    std::map<TileKey, IndexEntryF> boundaryInfo;

    int boundaryBlocksCount = 0;
    int mainBlocksCount = 0;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto& b = blocks_[i];
        if (b.contained) {
            TileKey key{b.row, b.col};
            tileMainBlocks[key].push_back(i);
            mainBlocksCount++;
            continue;
        }

        boundaryBlocksCount++;
        dataStream_.seekg(b.idx.st);
        size_t len = b.idx.ed - b.idx.st;
        if (len == 0) continue;
        std::vector<char> data(len);
        dataStream_.read(data.data(), len);
        const char* ptr = data.data();
        const char* end = ptr + len;
        const char* lineStart = ptr;
        std::vector<std::string> tokens;

        while (lineStart < end) {
            const char* lineEnd = static_cast<const char*>(memchr(lineStart, '\n', end - lineStart));
            if (!lineEnd) lineEnd = end;

            size_t lineLen = lineEnd - lineStart;
            if (lineLen > 0 && lineStart[lineLen - 1] == '\r') lineLen--;
            if (lineLen == 0 || lineStart[0] == '#') {
                lineStart = lineEnd + 1;
                continue;
            }
            std::string_view lineView(lineStart, lineLen);
            split(tokens, "\t", lineView);
            if (tokens.size() < icol_max_ + 1) {
                error("Insufficient tokens (%lu) in line (block %lu): %.*s.", tokens.size(), i, (int)lineLen, lineStart);
            }

            float x, y;
            if (!str2float(tokens[icol_x_], x) || !str2float(tokens[icol_y_], y)) {
                error("Invalid coordinate values in line: %.*s", (int)lineLen, lineStart);
            }
            if (mode_ & 0x2) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
            }

            int32_t c = static_cast<int32_t>(std::floor(x / tileSize));
            int32_t r = static_cast<int32_t>(std::floor(y / tileSize));
            TileKey key{r, c};
            boundaryLines[key].emplace_back(lineStart, lineLen);
            if (boundaryInfo.find(key) == boundaryInfo.end()) {
                IndexEntryF idx(r, c);
                tile2bound(key, idx.xmin, idx.xmax, idx.ymin, idx.ymax, tileSize);
                boundaryInfo.emplace(key, std::move(idx));
            }
            lineStart = lineEnd + 1;
        }
    }

    notice("Found %d main blocks and %d boundary blocks", mainBlocksCount, boundaryBlocksCount);

    std::string outFile = outPrefix + ".tsv";
    std::string outIndex = outPrefix + ".index";

    int fdMain = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdMain < 0) error("Cannot open output file %s", outFile.c_str());

    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());

    // Write index header
    IndexHeader idxHeader = formatInfo_;
    idxHeader.mode &= ~0x8;
    idxHeader.tileSize = tileSize;
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
        error("Error writing header to index output file: %s", outIndex.c_str());
    }

    if (!write_all(fdMain, headerLine_.data(), headerLine_.size())) {
        error("Error writing header");
    }

    size_t currentOffset = headerLine_.size();

    // 1. Process tiles with main blocks
    for (const auto& kv : tileMainBlocks) {
        TileKey tile = kv.first;
        const auto& indices = kv.second;

        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n = 0;
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, tileSize);

        // Write main blocks
        for (size_t i : indices) {
            const auto& mb = blocks_[i];
            size_t len = mb.idx.ed - mb.idx.st;
            if (len > 0) {
                dataStream_.seekg(mb.idx.st);
                size_t copied = 0;
                const size_t bufSz = 1024 * 1024;
                std::vector<char> buffer(bufSz);
                while (copied < len) {
                    size_t toRead = std::min(bufSz, len - copied);
                    dataStream_.read(buffer.data(), toRead);
                    if (!write_all(fdMain, buffer.data(), toRead)) error("Write error");
                    copied += toRead;
                }
                newEntry.n += mb.idx.n;
            }
        }

        // Append boundary lines if any
        if (boundaryLines.count(tile)) {
            const auto& lines = boundaryLines[tile];
            for (size_t i = 0; i < lines.size(); ++i) {
                const std::string& l = lines[i];
                if (!write_all(fdMain, l.data(), l.size())) error("Write error");
                if (!write_all(fdMain, "\n", 1)) error("Write error");

                newEntry.n++;
            }
            boundaryLines.erase(tile);
        }

        newEntry.ed = lseek(fdMain, 0, SEEK_CUR);
        currentOffset = newEntry.ed;
        if (newEntry.n > 0) {
            if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index write error");
        }
    }

    // 2. Process remaining boundary-only tiles
    for (const auto& kv : boundaryLines) {
        TileKey tile = kv.first;
        const auto& lines = kv.second;
        IndexEntryF& newEntry = boundaryInfo[tile];
        newEntry.st = currentOffset;
        for (size_t i = 0; i < lines.size(); ++i) {
            const std::string& l = lines[i];
            if (!write_all(fdMain, l.data(), l.size())) error("Write error");
            if (!write_all(fdMain, "\n", 1)) error("Write error");
            newEntry.n++;
        }
        newEntry.ed = lseek(fdMain, 0, SEEK_CUR);
        currentOffset = newEntry.ed;
        if (newEntry.n > 0) {
            if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index write error");
        }
    }

    close(fdMain);
    close(fdIndex);
    notice("Reorganization completed. Output written to %s\n Index written to %s", outFile.c_str(), outIndex.c_str());
}

void TileOperator::reorgTilesBinary(const std::string& outPrefix, int32_t tileSize) {
    std::map<TileKey, std::vector<size_t>> tileMainBlocks;
    std::map<TileKey, std::vector<char>> boundaryData;
    std::map<TileKey, IndexEntryF> boundaryInfo;

    int boundaryBlocksCount = 0;
    int mainBlocksCount = 0;
    uint32_t recordSize = formatInfo_.recordSize;
    std::vector<char> recBuf(recordSize);

    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto& b = blocks_[i];
        if (b.contained) {
            TileKey key{b.row, b.col};
            tileMainBlocks[key].push_back(i);
            mainBlocksCount++;
            continue;
        }

        boundaryBlocksCount++;
        dataStream_.seekg(b.idx.st);
        size_t len = b.idx.ed - b.idx.st;
        size_t nRecords = len / recordSize;

        for (size_t j = 0; j < nRecords; ++j) {
            if (!dataStream_.read(recBuf.data(), recordSize))
                error("%s: Read error block %lu", __func__, i);

            float x, y;
            if (mode_ & 0x4) {
                int32_t xi = *reinterpret_cast<int32_t*>(recBuf.data());
                int32_t yi = *reinterpret_cast<int32_t*>(recBuf.data() + 4);
                x = static_cast<float>(xi);
                y = static_cast<float>(yi);
            } else {
                x = *reinterpret_cast<float*>(recBuf.data());
                y = *reinterpret_cast<float*>(recBuf.data() + 4);
            }
            if (mode_ & 0x2) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
            }

            int32_t c = static_cast<int32_t>(std::floor(x / tileSize));
            int32_t r = static_cast<int32_t>(std::floor(y / tileSize));
            TileKey key{r, c};
            auto& d = boundaryData[key];
            size_t csz = d.size();
            d.resize(csz + recordSize);
            std::memcpy(d.data() + csz, recBuf.data(), recordSize);

            if (boundaryInfo.find(key) == boundaryInfo.end()) {
                IndexEntryF idx(r, c);
                tile2bound(key, idx.xmin, idx.xmax, idx.ymin, idx.ymax, tileSize);
                boundaryInfo.emplace(key, std::move(idx));
            }
        }
    }

    notice("Found %d main blocks and %d boundary blocks", mainBlocksCount, boundaryBlocksCount);

    std::string outFile = outPrefix + ".bin";
    std::string outIndex = outPrefix + ".index";

    int fdMain = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdMain < 0) error("Cannot open output file %s", outFile.c_str());
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());

    IndexHeader idxHeader = formatInfo_;
    idxHeader.mode &= ~0x8;
    idxHeader.tileSize = tileSize;
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) error("Index header write error");

    size_t currentOffset = 0;
    std::vector<TileKey> allKeys;
    for (const auto& kv : tileMainBlocks) allKeys.push_back(kv.first);
    for (const auto& kv : boundaryData) allKeys.push_back(kv.first);
    std::sort(allKeys.begin(), allKeys.end());
    allKeys.erase(std::unique(allKeys.begin(), allKeys.end()), allKeys.end());
    int32_t nTiles = 0;
    for (const auto& tile : allKeys) {
        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n = 0;
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, tileSize);

        if (tileMainBlocks.count(tile)) {
            for (size_t i : tileMainBlocks[tile]) {
                const auto& mb = blocks_[i];
                size_t len = mb.idx.ed - mb.idx.st;
                if (len > 0) {
                    dataStream_.seekg(mb.idx.st);
                    size_t copied = 0;
                    const size_t bufSz = 1024 * 1024;
                    std::vector<char> buffer(bufSz);
                    while (copied < len) {
                        size_t toRead = std::min(bufSz, len - copied);
                        dataStream_.read(buffer.data(), toRead);
                        if (!write_all(fdMain, buffer.data(), toRead)) error("Write error");
                        copied += toRead;
                    }
                    newEntry.n += mb.idx.n;
                }
            }
        }

        if (boundaryData.count(tile)) {
            const auto& d = boundaryData[tile];
            if (!d.empty()) {
                if (!write_all(fdMain, d.data(), d.size())) error("Write error");
                newEntry.n += d.size() / recordSize;
            }
        }

        newEntry.ed = lseek(fdMain, 0, SEEK_CUR);
        currentOffset = newEntry.ed;
        if (newEntry.n > 0) {
            if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index write error");
            nTiles++;
        }
    }
    close(fdMain);
    close(fdIndex);
    notice("Reorganized into %d tiles. Output written to %s\n Index written to %s", nTiles, outFile.c_str(), outIndex.c_str());
    return;
}

void TileOperator::probDot(const std::string& outPrefix, int32_t probDigits) {
    if (k_ <= 0 || kvec_.empty()) {
        warning("%s: k is 0 or unknown, nothing to do", __func__);
        return;
    }
    size_t nSets = kvec_.size();

    // Accumulators
    std::vector<std::map<int32_t, double>> marginals(nSets);
    std::vector<std::map<std::pair<int32_t, int32_t>, double>> internalDots(nSets);
    std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>> crossDots;

    PixTopProbs<float> recFloat;
    PixTopProbs<int32_t> recInt;
    std::vector<int32_t>* ksPtr = nullptr;
    std::vector<float>* psPtr = nullptr;

    // Precompute offsets
    std::vector<uint32_t> offsets(nSets + 1, 0);
    for (size_t s = 0; s < nSets; ++s) offsets[s+1] = offsets[s] + kvec_[s];

    resetReader();
    size_t count = 0;
    while (true) {
        int32_t ret = 0;
        if (mode_ & 0x4) {
            ret = next(recInt);
            ksPtr = &recInt.ks;
            psPtr = &recInt.ps;
        } else {
            ret = next(recFloat, true);
            ksPtr = &recFloat.ks;
            psPtr = &recFloat.ps;
        }
        if (ret == -1) break;
        if (ret == 0) continue;
        std::vector<int32_t>& ks = *ksPtr;
        std::vector<float>& ps = *psPtr;

        count++;

        for (size_t s1 = 0; s1 < nSets; ++s1) {
            uint32_t off1 = offsets[s1];
            for (uint32_t i = 0; i < kvec_[s1]; ++i) {
                int32_t k1 = ks[off1 + i];
                float p1 = ps[off1 + i];
                marginals[s1][k1] += p1;
                // Internal
                for (uint32_t j = i; j < kvec_[s1]; ++j) {
                    int32_t k2 = ks[off1 + j];
                    float p2 = ps[off1 + j];
                    // Unordered pair within set
                    int32_t ka = std::min(k1, k2);
                    int32_t kb = std::max(k1, k2);
                    internalDots[s1][{ka, kb}] += (double)p1 * p2;
                }
                // Cross with s2 > s1
                for (size_t s2 = s1 + 1; s2 < nSets; ++s2) {
                    uint32_t off2 = offsets[s2];
                    for (uint32_t j = 0; j < kvec_[s2]; ++j) {
                        int32_t k2 = ks[off2 + j];
                        float p2 = ps[off2 + j];
                        // Ordered pair (k1 from s1, k2 from s2)
                        crossDots[{s1, s2}][{k1, k2}] += (double)p1 * p2;
                    }
                }
            }
        }
    }

    notice("%s: Processed %lu records", __func__, count);

    // Write outputs
    for (size_t s = 0; s < nSets; ++s) {
        std::string fn = outPrefix + ".";
        if (nSets > 1) {fn += std::to_string(s) + ".marginal.tsv";}
        else {fn += "marginal.tsv";}
        FILE* fp = fopen(fn.c_str(), "w");
        if (!fp) error("Cannot open output file %s", fn.c_str());
        fprintf(fp, "#K\tSum\n");
        for (const auto& kv : marginals[s]) {
             fprintf(fp, "%d\t%.*e\n", kv.first, probDigits, kv.second);
        }
        fclose(fp);
    }

    for (size_t s = 0; s < nSets; ++s) {
        std::string fn = outPrefix + ".";
        if (nSets > 1) {fn += std::to_string(s) + ".joint.tsv";}
        else fn += "joint.tsv";
        FILE* fp = fopen(fn.c_str(), "w");
        if (!fp) error("Cannot open output file %s", fn.c_str());
        fprintf(fp, "#K1\tK2\tJoint\n");
        for (const auto& kv : internalDots[s]) {
             fprintf(fp, "%d\t%d\t%.*e\n", kv.first.first, kv.first.second, probDigits, kv.second);
        }
        fclose(fp);
    }

    for (auto const& [setPair, mapVal] : crossDots) {
        std::string fn = outPrefix + "." + std::to_string(setPair.first) + "v" + std::to_string(setPair.second) + ".cross.tsv";
        FILE* fp = fopen(fn.c_str(), "w");
        if (!fp) error("Cannot open output file %s", fn.c_str());
        fprintf(fp, "#K1\tK2\tJoint\n");
        for (const auto& kv : mapVal) {
             fprintf(fp, "%d\t%d\t%.*e\n", kv.first.first, kv.first.second, probDigits, kv.second);
        }
        fclose(fp);
    }
}

void TileOperator::probDotTiles2D(const std::set<TileKey>& commonTiles,
    const std::vector<TileOperator*>& opPtrs,
    const std::vector<uint32_t>& k2keep,
    const std::vector<uint32_t>& offsets,
    std::vector<std::map<int32_t, double>>& marginals,
    std::vector<std::map<std::pair<int32_t, int32_t>, double>>& internalDots,
    std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>>& crossDots,
    size_t& count) {
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
                        int32_t ka = std::min(k1, k2);
                        int32_t kb = std::max(k1, k2);
                        internalDots[s1][{ka, kb}] += (double)p1 * p2;
                    }
                    // Cross
                    for (size_t s2 = s1 + 1; s2 < nSets; ++s2) {
                        uint32_t off2 = offsets[s2];
                        for (uint32_t j = 0; j < k2keep[s2]; ++j) {
                            int32_t k2 = ks[off2 + j];
                            float p2 = ps[off2 + j];
                            crossDots[{s1, s2}][{k1, k2}] += (double)p1 * p2;
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
    const std::vector<uint32_t>& k2keep,
    const std::vector<uint32_t>& offsets,
    std::vector<std::map<int32_t, double>>& marginals,
    std::vector<std::map<std::pair<int32_t, int32_t>, double>>& internalDots,
    std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>>& crossDots,
    size_t& count) {
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
                        int32_t ka = std::min(k1, k2);
                        int32_t kb = std::max(k1, k2);
                        internalDots[s1][{ka, kb}] += (double)p1 * p2;
                    }
                    // Cross
                    for (size_t s2 = s1 + 1; s2 < nSets; ++s2) {
                        uint32_t off2 = offsets[s2];
                        for (uint32_t j = 0; j < k2keep[s2]; ++j) {
                            int32_t k2 = ks[off2 + j];
                            float p2 = ps[off2 + j];
                            crossDots[{s1, s2}][{k1, k2}] += (double)p1 * p2;
                        }
                    }
                }
            }
        }
        notice("%s: Processed tile (%d, %d) with %lu pixels shared by all sources", __func__, tile.row, tile.col, mergedMap.size());
    }
}

void TileOperator::probDot_multi(const std::vector<std::string>& otherFiles, const std::string& outPrefix, std::vector<uint32_t> k2keep, int32_t probDigits) {
    if (k2keep.size() > 0) {assert(k2keep.size() == otherFiles.size() + 1);}
    assert(!otherFiles.empty());

    // 1. Setup operators
    std::vector<std::unique_ptr<TileOperator>> ops;
    std::vector<TileOperator*> opPtrs;
    opPtrs.push_back(this); // current object
    for (const auto& f : otherFiles) {
        std::string idxFile = f.substr(0, f.find_last_of('.')) + ".index";
        struct stat buffer;
        if (stat(idxFile.c_str(), &buffer) != 0) {
            error("%s: Index file %s not found", __func__, idxFile.c_str());
        }
        std::string df = f;
        ops.push_back(std::make_unique<TileOperator>(df, idxFile));
        opPtrs.push_back(ops.back().get());
    }
    uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    if (k2keep.size() == 0) {
        for (auto* op : opPtrs) {
            k2keep.push_back(op->getK());
        }
    } else {
        for (uint32_t i = 0; i < nSources; ++i) {
            if (k2keep[i] > opPtrs[i]->getK()) {
                warning("%s: Invalid value k (%d) specified for the %d-th source", __func__, k2keep[i], i);
                k2keep[i] = opPtrs[i]->getK();
            }
        }
    }

    size_t nSets = nSources;
    bool use3d = (coord_dim_ == 3);
    for (auto* op : opPtrs) {
        if (op->coord_dim_ != coord_dim_) {
            error("%s: Mixed 2D/3D inputs are not supported", __func__);
        }
    }
    // Accumulators
    std::vector<std::map<int32_t, double>> marginals(nSets);
    std::vector<std::map<std::pair<int32_t, int32_t>, double>> internalDots(nSets);
    std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>> crossDots;

    // Precompute offsets
    std::vector<uint32_t> offsets(nSets + 1, 0);
    for (size_t s = 0; s < nSets; ++s) offsets[s+1] = offsets[s] + k2keep[s];

    // 2. Identify common tiles (Intersection)
    std::set<TileKey> commonTiles;
    if (opPtrs[0]->tile_lookup_.empty()) {
        warning("%s: No tiles in the base dataset", __func__);
        return;
    }
    for (const auto& kv : opPtrs[0]->tile_lookup_) {
        commonTiles.insert(kv.first);
    }
    for (uint32_t i = 1; i < nSources; ++i) {
        std::set<TileKey> currentTiles;
        for (const auto& kv : opPtrs[i]->tile_lookup_) {
            if (commonTiles.count(kv.first)) {
                currentTiles.insert(kv.first);
            }
        }
        commonTiles = currentTiles;
    }
    if (commonTiles.empty()) {
        warning("%s: No overlapping tiles found for merge", __func__);
        return;
    }

    notice("%s: Start computing on %u files (%lu shared tiles)", __func__, nSources, commonTiles.size());

    // 3. Process tiles
    size_t count = 0;
    if (use3d) {
        probDotTiles3D(commonTiles, opPtrs, k2keep, offsets, marginals, internalDots, crossDots, count);
    } else {
        probDotTiles2D(commonTiles, opPtrs, k2keep, offsets, marginals, internalDots, crossDots, count);
    }

    notice("%s: Processed %lu shared pixels", __func__, count);

    // Write outputs
    for (size_t s = 0; s < nSets; ++s) {
        std::string fn = outPrefix + "." + std::to_string(s) + ".marginal.tsv";
        FILE* fp = fopen(fn.c_str(), "w");
        if (!fp) error("Cannot open output file %s", fn.c_str());
        fprintf(fp, "#K\tSum\n");
        for (const auto& kv : marginals[s]) {
             fprintf(fp, "%d\t%.*e\n", kv.first, probDigits, kv.second);
        }
        fclose(fp);
    }

    for (size_t s = 0; s < nSets; ++s) {
        std::string fn = outPrefix + "." + std::to_string(s) + ".joint.tsv";
        FILE* fp = fopen(fn.c_str(), "w");
        if (!fp) error("Cannot open output file %s", fn.c_str());
        fprintf(fp, "#K1\tK2\tJoint\n");
        for (const auto& kv : internalDots[s]) {
             fprintf(fp, "%d\t%d\t%.*e\n", kv.first.first, kv.first.second, probDigits, kv.second);
        }
        fclose(fp);
    }

    for (auto const& [setPair, mapVal] : crossDots) {
        std::string fn = outPrefix + "." + std::to_string(setPair.first) + "v" + std::to_string(setPair.second) + ".cross.tsv";
        FILE* fp = fopen(fn.c_str(), "w");
        if (!fp) error("Cannot open output file %s", fn.c_str());
        fprintf(fp, "#K1\tK2\tJoint\n");
        for (const auto& kv : mapVal) {
             fprintf(fp, "%d\t%d\t%.*e\n", kv.first.first, kv.first.second, probDigits, kv.second);
        }
        fclose(fp);
    }
}
