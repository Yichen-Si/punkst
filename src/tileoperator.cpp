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
#include <unordered_set>

namespace {

using FactorSums = std::pair<std::unordered_map<int32_t, double>, int32_t>;

struct CellAgg {
    FactorSums sums;
    std::map<std::string, FactorSums> compSums;
    bool boundary = false;
};

void write_top_factors(FILE* fp, const FactorSums& sums, uint32_t k_out) {
    std::vector<std::pair<int32_t, double>> items;
    items.reserve(sums.first.size());
    for (const auto& kv : sums.first) {
        if (kv.second != 0.0) {
            items.emplace_back(kv.first, kv.second / sums.second);
        }
    }
    uint32_t keep = std::min<uint32_t>(k_out, static_cast<uint32_t>(items.size()));
    if (keep > 0) {
        std::partial_sort(items.begin(), items.begin() + keep, items.end(),
            [](const auto& a, const auto& b) {
                if (a.second == b.second) return a.first < b.first;
                return a.second > b.second;
            });
    }
    fprintf(fp, "\t%d", sums.second);
    for (uint32_t i = 0; i < keep; ++i) {
        fprintf(fp, "\t%d\t%.4e", items[i].first, items[i].second);
    }
    for (uint32_t i = keep; i < k_out; ++i) {
        fprintf(fp, "\t-1\t0");
    }
}

void write_cell_row(FILE* fp, const std::string& cellId, const std::string& comp, const FactorSums& sums, uint32_t k_out, bool writeComp) {
    if (writeComp) {
        fprintf(fp, "%s\t%s", cellId.c_str(), comp.c_str());
    } else {
        fprintf(fp, "%s", cellId.c_str());
    }
    write_top_factors(fp, sums, k_out);
    fprintf(fp, "\n");
}

} // namespace

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
        ops.push_back(std::make_unique<TileOperator>(f, idxFile));
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

void TileOperator::annotate(const std::string& ptPrefix, const std::string& outPrefix, uint32_t icol_x, uint32_t icol_y, int32_t icol_z) {
    std::string ptData = ptPrefix + ".tsv";
    std::string ptIndex = ptPrefix + ".index";
    TileReader reader(ptData, ptIndex);
    assert(reader.getTileSize() == formatInfo_.tileSize);
    assert(icol_x != icol_y);
    if (coord_dim_ == 3) {assert(icol_z >= 0 && icol_z != icol_x && icol_z != icol_y);}
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
    if (use3d) {ntok = std::max(ntok, (uint32_t) icol_z);}
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
        annotateTiles3D(tiles, reader, icol_x, icol_y, (uint32_t) icol_z, ntok, fp, fdIndex, currentOffset);
    } else {
        annotateTiles2D(tiles, reader, icol_x, icol_y, ntok, fp, fdIndex, currentOffset);
    }

    fclose(fp);
    close(fdIndex);
    notice("Annotation finished, data written to %s", outFile.c_str());
}

void TileOperator::pix2cell(const std::string& ptPrefix, const std::string& outPrefix, uint32_t icol_c, uint32_t icol_x, uint32_t icol_y, int32_t icol_s, int32_t icol_z, uint32_t k_out, float max_cell_diameter) {
    std::string ptData = ptPrefix + ".tsv";
    std::string ptIndex = ptPrefix + ".index";
    TileReader reader(ptData, ptIndex);
    assert(reader.getTileSize() == formatInfo_.tileSize);
    assert(icol_c >= 0 && icol_c != icol_x && icol_c != icol_y);
    assert(icol_x != icol_y);
    if (coord_dim_ == 3) {assert(icol_z >= 0);}
    bool use3d = (coord_dim_ == 3);
    bool hasComp = (icol_s >= 0);
    if (k_out == 0) {k_out = static_cast<uint32_t>(k_);}
    if (k_out == 0) {error("%s: Invalid k_out value", __func__);}

    std::string outFile = outPrefix + ".tsv";
    FILE* fp = fopen(outFile.c_str(), "w");
    if (!fp) error("Cannot open output file %s", outFile.c_str());
    std::string headerStr = hasComp ? "#CellID\tCellComp\tnPixel" : "#CellID\tnPixel";
    for (uint32_t i = 1; i <= k_out; ++i) {
        headerStr += "\tK" + std::to_string(i) + "\tP" + std::to_string(i);
    }
    headerStr += "\n";
    fprintf(fp, "%s", headerStr.c_str());

    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;
    int32_t tileSize =  formatInfo_.tileSize;
    bool enableEarlyFlush = (max_cell_diameter > 0);

    uint32_t ntok = std::max(icol_c, std::max(icol_x, icol_y));
    if (use3d) {
        ntok = std::max(ntok, static_cast<uint32_t>(icol_z));
    }
    if (hasComp) {
        ntok = std::max(ntok, static_cast<uint32_t>(icol_s));
    }
    ntok += 1;

    std::map<std::string, CellAgg> cellAggs;
    std::map<std::string, FactorSums> compTotals;

    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    int32_t nTilesDone = 0;
    int32_t nTiles = static_cast<int32_t>(tiles.size());
    notice("%s: Start processing %d tiles", __func__, nTiles);
    for (const auto& tile : tiles) {
        nTilesDone++;
        std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
        std::map<PixelKey3, TopProbs> pixelMap3d;
        if (use3d) {
            if (loadTileToMap3D(tile, pixelMap3d) <= 0) {
                continue;
            }
        } else {
            if (loadTileToMap(tile, pixelMap) <= 0) {
                continue;
            }
        }

        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) continue;

        float tileX0 = tile.col * tileSize;
        float tileX1 = (tile.col + 1) * tileSize;
        float tileY0 = tile.row * tileSize;
        float tileY1 = (tile.row + 1) * tileSize;

        std::unordered_set<std::string> tileCells;
        std::string s;
        while (it->next(s)) {
            if (s.empty() || s[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", s, ntok + 1, true, true, true);
            if (tokens.size() < ntok) {
                error("%s: Invalid line: %s", __func__, s.c_str());
            }
            float x, y, z = 0.0f;
            if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
                error("%s: Invalid coordinates in line: %s", __func__, s.c_str());
            }
            if (use3d && !str2float(tokens[icol_z], z)) {
                error("%s: Invalid z coordinate in line: %s", __func__, s.c_str());
            }

            int32_t ix = static_cast<int32_t>(std::floor(x / res));
            int32_t iy = static_cast<int32_t>(std::floor(y / res));
            int32_t iz = 0;
            const TopProbs* probs = nullptr;
            if (use3d) {
                iz = static_cast<int32_t>(std::floor(z / res));
                auto pit = pixelMap3d.find(std::make_tuple(ix, iy, iz));
                if (pit == pixelMap3d.end()) {
                    continue;
                }
                probs = &pit->second;
            } else {
                auto pit = pixelMap.find({ix, iy});
                if (pit == pixelMap.end()) {
                    continue;
                }
                probs = &pit->second;
            }

            const std::string& cellId = tokens[icol_c];
            const std::string compartment = hasComp ? tokens[icol_s] : std::string();
            auto& agg = cellAggs[cellId];
            agg.sums.second += 1;
            if (hasComp) {
                compTotals[compartment].second += 1;
                agg.compSums[compartment].second += 1;
            }

            if (enableEarlyFlush) {
                tileCells.insert(cellId);
                float minDist = std::min(
                    std::min(x - tileX0, tileX1 - x),
                    std::min(y - tileY0, tileY1 - y));
                if (minDist <= max_cell_diameter) {
                    agg.boundary = true;
                }
            }

            for (size_t i = 0; i < probs->ks.size(); ++i) {
                int32_t k = probs->ks[i];
                double p = probs->ps[i];
                agg.sums.first[k] += p;
                if (hasComp) {
                    agg.compSums[compartment].first[k] += p;
                    compTotals[compartment].first[k] += p;
                }
            }
        }

        if (!enableEarlyFlush || tileCells.empty()) {
            notice("%s: Processed tile [%d, %d] (%d/%d, %lu)", __func__, tile.row, tile.col, nTilesDone, nTiles, cellAggs.size());
            continue;
        }

        int32_t nFlushed = 0;
        for (const auto& cellId : tileCells) {
            auto itCell = cellAggs.find(cellId);
            if (itCell == cellAggs.end()) continue;
            if (itCell->second.boundary) continue;
            if (hasComp) {
                write_cell_row(fp, cellId, "ALL", itCell->second.sums, k_out, true);
                for (const auto& kv : itCell->second.compSums) {
                    write_cell_row(fp, cellId, kv.first, kv.second, k_out, true);
                }
            } else {
                write_cell_row(fp, cellId, "", itCell->second.sums, k_out, false);
            }
            cellAggs.erase(itCell);
            nFlushed++;
        }
        notice("%s: Processed tile [%d, %d] (%d/%d) with %lu cells, output %d, %lu in buffer", __func__, tile.row, tile.col, nTilesDone, nTiles, tileCells.size(), nFlushed, cellAggs.size());
    }

    for (const auto& kv : cellAggs) {
        if (hasComp) {
            write_cell_row(fp, kv.first, "ALL", kv.second.sums, k_out, true);
            for (const auto& comp : kv.second.compSums) {
                write_cell_row(fp, kv.first, comp.first, comp.second, k_out, true);
            }
        } else {
            write_cell_row(fp, kv.first, "", kv.second.sums, k_out, false);
        }
    }

    fclose(fp);
    notice("%s: Cell aggregation written to %s", __func__, outFile.c_str());
    if (!hasComp) {
        return;
    }

    std::string pbFile = outPrefix + ".pseudobulk.tsv";
    FILE* pb = fopen(pbFile.c_str(), "w");
    if (!pb) error("Cannot open output file %s", pbFile.c_str());
    fprintf(pb, "Factor");
    for (const auto& kv : compTotals) {
        fprintf(pb, "\t%s", kv.first.c_str());
    }
    fprintf(pb, "\n");
    std::set<int32_t> factors;
    for (const auto& kv : compTotals) {
        for (const auto& fv : kv.second.first) {
            if (fv.second != 0.0) {
                factors.insert(fv.first);
            }
        }
    }
    for (int32_t k : factors) {
        fprintf(pb, "%d", k);
        for (const auto& kv : compTotals) {
            double v = 0.0;
            auto it = kv.second.first.find(k);
            if (it != kv.second.first.end()) v = it->second;
            fprintf(pb, "\t%.4e", v);
        }
        fprintf(pb, "\n");
    }
    fclose(pb);
    notice("%s: Pseudobulk matrix written to %s", __func__, pbFile.c_str());
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
                    std::pair<int32_t, int32_t> k12 = (k1 <= k2) ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
                    internalDots[s1][k12] += (double) p1 * p2;
                }
                // Cross with s2 > s1
                for (size_t s2 = s1 + 1; s2 < nSets; ++s2) {
                    uint32_t off2 = offsets[s2];
                    for (uint32_t j = 0; j < kvec_[s2]; ++j) {
                        int32_t k2 = ks[off2 + j];
                        float p2 = ps[off2 + j];
                        // Ordered pair (k1 from s1, k2 from s2)
                        crossDots[std::make_pair(s1, s2)][std::make_pair(k1, k2)] += (double)p1 * p2;
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
        std::map<int32_t, double> rowSums;
        std::map<int32_t, double> colSums;
        double total = 0.0;
        double pseudo = 0.5;
        for (const auto& kv : mapVal) {
            rowSums[kv.first.first] += kv.second;
            colSums[kv.first.second] += kv.second;
            total += kv.second;
            if (kv.second > 0 && kv.second < pseudo) {
                pseudo = kv.second;
            }
        }
        pseudo *= 0.5;
        std::map<int32_t, double> colFreq = colSums;
        for (auto& kv : colFreq) {kv.second /= total;}
        fprintf(fp, "#K1\tK2\tJoint\tlog10pval\tlog2OR\n");
        for (const auto& kv : mapVal) {
             double a = kv.second;
             double rowSum = rowSums[kv.first.first];
             double colSum = colSums[kv.first.second];
             double b = std::max(0.0, rowSum - a);
             double c = std::max(0.0, colSum - a);
             double d = std::max(0.0, total - rowSum - colSum + a);
             auto stats = chisq2x2_log10p(a, b, c, d, pseudo);
             double log2OR = std::log2(a+pseudo) - std::log2(rowSums[kv.first.first] * colFreq[kv.first.second] + pseudo);
             fprintf(fp, "%d\t%d\t%.*e\t%.4e\t%.4e\n",
                 kv.first.first, kv.first.second, probDigits, kv.second, stats.second, log2OR);
        }
        fclose(fp);
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
        std::map<int32_t, double> rowSums;
        std::map<int32_t, double> colSums;
        double total = 0.0;
        double pseudo = 0.5;
        for (const auto& kv : mapVal) {
            rowSums[kv.first.first] += kv.second;
            colSums[kv.first.second] += kv.second;
            total += kv.second;
            if (kv.second > 0 && kv.second < pseudo) {
                pseudo = kv.second;
            }
        }
        std::map<int32_t, double> colFreq = colSums;
        for (auto& kv : colFreq) {kv.second /= total;}
        pseudo *= 0.5;
        fprintf(fp, "#K1\tK2\tJoint\tlog10pval\tlog2OR\n");
        for (const auto& kv : mapVal) {
             double a = kv.second;
             double rowSum = rowSums[kv.first.first];
             double colSum = colSums[kv.first.second];
             double b = std::max(0.0, rowSum - a);
             double c = std::max(0.0, colSum - a);
             double d = std::max(0.0, total - rowSum - colSum + a);
             double log2OR = std::log2(a+pseudo) - std::log2(rowSums[kv.first.first] * colFreq[kv.first.second] + pseudo);
             auto stats = chisq2x2_log10p(a, b, c, d, pseudo);
             fprintf(fp, "%d\t%d\t%.*e\t%.4e\t%.4e\n",
                 kv.first.first, kv.first.second, probDigits, kv.second, stats.second, log2OR);
        }
        fclose(fp);
    }
}

std::unordered_map<int32_t, TileOperator::Slice> TileOperator::aggOneTile(TileReader& reader, lineParserUnival& parser, TileKey tile, double gridSize, double minProb) const {
    if (coord_dim_ == 3) {error("%s does not support 3D data yet", __func__);}
    assert(reader.getTileSize() == formatInfo_.tileSize);
    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;

    std::unordered_map<int32_t, Slice> tileAgg; // k -> unit key ->
    std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
    if (loadTileToMap(tile, pixelMap) == 0) {return tileAgg;}
    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {return tileAgg;}

    std::string line;
    RecordT<float> rec;
    // parse data with (x,y,feature_id,count)
    while (it->next(line)) {
        if (line.empty() || line[0] == '#') continue;
        int32_t idx = parser.parse(rec, line);
        if (idx < 0) continue;
        int32_t ix = static_cast<int32_t>(std::floor(rec.x / res));
        int32_t iy = static_cast<int32_t>(std::floor(rec.y / res));
        auto pixIt = pixelMap.find({ix, iy});
        if (pixIt == pixelMap.end()) continue;
        int32_t ux = static_cast<int32_t>(std::floor(rec.x / gridSize));
        int32_t uy = static_cast<int32_t>(std::floor(rec.y / gridSize));
        auto& anno = pixIt->second;
        for (size_t i = 0; i < anno.ks.size(); ++i) {
            if (anno.ps[i] < minProb) continue;
            int32_t k = anno.ks[i];
            float p = anno.ps[i];
            auto aggIt = tileAgg.find(k);
            if (aggIt == tileAgg.end()) {
                aggIt = tileAgg.emplace(k, Slice()).first;
            }
            auto& oneSlice = aggIt->second;
            auto unitIt = oneSlice.find({ux, uy});
            if (unitIt == oneSlice.end()) {
                unitIt = oneSlice.emplace(std::make_pair(ux, uy), SparseObsDict()).first;
            }
            unitIt->second.add(rec.idx, rec.ct * p);
        }
    }
    return tileAgg;
}
