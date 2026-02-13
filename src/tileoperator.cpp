#include "tileoperator.hpp"
#include "numerical_utils.hpp"
#include "img_utils.hpp"
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <set>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <numeric>
#include <limits>
#include <chrono>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

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

struct SpatialMetricsAccum {
    int32_t K = 0;
    std::vector<uint64_t> area; // pixel count
    std::vector<uint64_t> perim; // shared edge count with all other labels
    std::vector<uint64_t> perim_bg; // shared edge count with background
    std::vector<uint64_t> shared_ij; // asymmetric

    explicit SpatialMetricsAccum(int32_t K_)
        : K(K_),
          area(static_cast<size_t>(K_), 0),
          perim(static_cast<size_t>(K_), 0),
          perim_bg(static_cast<size_t>(K_), 0),
          shared_ij(static_cast<size_t>(K_ + 1) * static_cast<size_t>(K_) / 2, 0) {}

    size_t triIndex(int32_t i, int32_t j) const {
        const size_t base = static_cast<size_t>(i) * static_cast<size_t>(2 * K - i + 1) / 2;
        return base + static_cast<size_t>(j - i - 1);
    }

    uint64_t& shared(int32_t i, int32_t j) {
        if (i > j) std::swap(i, j);
        return shared_ij[triIndex(i, j)];
    }

    void add(const SpatialMetricsAccum& other) {
        for (size_t i = 0; i < area.size(); ++i) area[i] += other.area[i];
        for (size_t i = 0; i < perim.size(); ++i) perim[i] += other.perim[i];
        for (size_t i = 0; i < perim_bg.size(); ++i) perim_bg[i] += other.perim_bg[i];
        for (size_t i = 0; i < shared_ij.size(); ++i) shared_ij[i] += other.shared_ij[i];
    }
};

inline void spatialAccumulateEdge(SpatialMetricsAccum& m, uint8_t a, uint8_t b) {
    if (a == b) return;
    const uint8_t BG = static_cast<uint8_t>(m.K);
    m.shared(static_cast<int32_t>(a), static_cast<int32_t>(b))++;
    if (a < BG) m.perim[a]++;
    if (b < BG) m.perim[b]++;
    if (a == BG && b < BG) m.perim_bg[b]++;
    else if (b == BG && a < BG) m.perim_bg[a]++;
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

    if (headerLine_.empty()) {
        error("%s: Missing TSV header line; cannot reorganize text input", __func__);
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

void TileOperator::smoothTopLabels2D(const std::string& outPrefix, int32_t islandSmoothRounds, bool fillEmptyIslands) {
    if (!regular_labeled_raster_ || coord_dim_ != 2) {
        error("%s only supports raster 2D data with regular tiles", __func__);
    }
    if (islandSmoothRounds <= 0) {
        error("%s: islandSmoothRounds must be > 0", __func__);
    }
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    const bool coordScaled = (mode_ & 0x2) != 0;

    std::string outFile = outPrefix + ".bin";
    std::string outIndex = outPrefix + ".index";
    int fdMain = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdMain < 0) {
        error("%s: Cannot open output file %s", __func__, outFile.c_str());
    }
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) {
        close(fdMain);
        error("%s: Cannot open output index %s", __func__, outIndex.c_str());
    }

    IndexHeader idxHeader = formatInfo_;
    std::vector<uint32_t> outKvec{1};
    idxHeader.packKvec(outKvec);
    idxHeader.mode = (K_ << 16) | (mode_ & 0x2) | 0x5;
    idxHeader.coordType = 1;
    idxHeader.recordSize = sizeof(int32_t) * 2 + sizeof(int32_t) + sizeof(float);
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
        close(fdMain);
        close(fdIndex);
        error("%s: Failed writing index header", __func__);
    }

    struct SmoothTileResult {
        std::vector<char> data;
        uint32_t nOut = 0;
        size_t nCollisionIgnored = 0;
        size_t nOutOfRangeIgnored = 0;
        int32_t row = 0;
        int32_t col = 0;
    };

    const bool useParallel = (threads_ > 1 && blocks_.size() > 1);
    const size_t recSize = static_cast<size_t>(idxHeader.recordSize);
    const size_t chunkTileCount = useParallel
        ? std::max<size_t>(static_cast<size_t>(threads_) * 4, 1)
        : static_cast<size_t>(1);

    auto processTile = [&](size_t bi, std::ifstream& in, SmoothTileResult& out) {
        const TileInfo& blk = blocks_[bi];
        const TileKey tile{blk.idx.row, blk.idx.col};
        out.row = tile.row;
        out.col = tile.col;
        int32_t pixX0, pixX1, pixY0, pixY1; // Tile bounds (global pix coord)
        tile2bound(tile, pixX0, pixX1, pixY0, pixY1, formatInfo_.tileSize);
        if (coordScaled) {
            pixX0 = coord2pix(pixX0);
            pixX1 = coord2pix(pixX1);
            pixY0 = coord2pix(pixY0);
            pixY1 = coord2pix(pixY1);
        }
        if (pixX1 <= pixX0 || pixY1 <= pixY0) {
            error("%s: Invalid raster bounds in tile (%d, %d)", __func__, tile.row, tile.col);
        }
        const size_t width = static_cast<size_t>(pixX1 - pixX0);
        const size_t height = static_cast<size_t>(pixY1 - pixY0);
        if (height > 0 && width > std::numeric_limits<size_t>::max() / height) {
            error("%s: Raster size overflow in tile (%d, %d)", __func__, tile.row, tile.col);
        }
        const size_t nPixels = width * height;
        std::vector<int32_t> labels(nPixels, -1);
        std::vector<float> probs(nPixels, 0.0f);

        in.clear();
        in.seekg(static_cast<std::streamoff>(blk.idx.st));
        if (!in.good()) {
            error("%s: Failed seeking input stream to tile %lu", __func__, bi);
        }
        TopProbs rec;
        int32_t xpix = 0;
        int32_t ypix = 0;
        uint64_t pos = blk.idx.st;
        while (readNextRecord2DAsPixel(in, pos, blk.idx.ed, xpix, ypix, rec)) {
            if (rec.ks.empty() || rec.ps.empty()) {
                continue;
            }
            if (xpix < pixX0 || xpix >= pixX1 || ypix < pixY0 || ypix >= pixY1) {
                ++out.nOutOfRangeIgnored;
                continue;
            }
            const size_t x0 = static_cast<size_t>(xpix - pixX0);
            const size_t y0 = static_cast<size_t>(ypix - pixY0);
            const size_t idx = y0 * width + x0;
            if (labels[idx] != -1) {
                ++out.nCollisionIgnored;
                continue;
            }
            labels[idx] = rec.ks[0];
            probs[idx] = rec.ps[0];
        }

        island_smoothing::Options smoothOpts;
        smoothOpts.fillEmpty = fillEmptyIslands;
        smoothOpts.updateProbFromWinnerMin = true;
        island_smoothing::Result ret = island_smoothing::smoothLabels8Neighborhood(
            labels, &probs, width, height, islandSmoothRounds, smoothOpts);
        (void)ret;

        size_t nOutLocal = 0;
        for (size_t i = 0; i < nPixels; ++i) {
            if (labels[i] >= 0) {
                ++nOutLocal;
            }
        }
        if (nOutLocal > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
            error("%s: Too many output records in tile (%d, %d)", __func__, tile.row, tile.col);
        }
        out.nOut = static_cast<uint32_t>(nOutLocal);
        out.data.resize(nOutLocal * recSize);
        size_t off = 0;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t idx = y * width + x;
                const int32_t label = labels[idx];
                if (label < 0) {
                    continue;
                }
                const int32_t outX = static_cast<int32_t>(x) + pixX0;
                const int32_t outY = static_cast<int32_t>(y) + pixY0;
                const float outP = probs[idx];
                std::memcpy(out.data.data() + off, &outX, sizeof(int32_t));
                std::memcpy(out.data.data() + off + sizeof(int32_t), &outY, sizeof(int32_t));
                std::memcpy(out.data.data() + off + sizeof(int32_t) * 2, &label, sizeof(int32_t));
                std::memcpy(out.data.data() + off + sizeof(int32_t) * 3, &outP, sizeof(float));
                off += recSize;
            }
        }
    };

    size_t currentOffset = 0;
    const size_t nTiles = blocks_.size();
    size_t nProcessed = 0;
    std::unique_ptr<tbb::global_control> globalLimit;
    if (useParallel) {
        globalLimit = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
    }
    for (size_t chunkBegin = 0; chunkBegin < nTiles; chunkBegin += chunkTileCount) {
        const size_t chunkEnd = std::min(nTiles, chunkBegin + chunkTileCount);
        std::vector<SmoothTileResult> chunkResults(chunkEnd - chunkBegin);
        if (useParallel && (chunkEnd - chunkBegin) > 1) {
            tbb::parallel_for(tbb::blocked_range<size_t>(chunkBegin, chunkEnd),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::ifstream in;
                    if (mode_ & 0x1) {
                        in.open(dataFile_, std::ios::binary);
                    } else {
                        in.open(dataFile_);
                    }
                    if (!in.is_open()) {
                        error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
                    }
                    for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                        processTile(bi, in, chunkResults[bi - chunkBegin]);
                    }
                });
        } else {
            std::ifstream in;
            if (mode_ & 0x1) {
                in.open(dataFile_, std::ios::binary);
            } else {
                in.open(dataFile_);
            }
            if (!in.is_open()) {
                error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
            }
            for (size_t bi = chunkBegin; bi < chunkEnd; ++bi) {
                processTile(bi, in, chunkResults[bi - chunkBegin]);
            }
        }

        for (size_t bi = chunkBegin; bi < chunkEnd; ++bi) {
            const auto& res = chunkResults[bi - chunkBegin];
            const TileInfo& blk = blocks_[bi];
            IndexEntryF outEntry = blk.idx;
            outEntry.st = currentOffset;
            outEntry.n = res.nOut;
            if (!res.data.empty()) {
                if (!write_all(fdMain, res.data.data(), res.data.size())) {
                    close(fdMain);
                    close(fdIndex);
                    error("%s: Failed writing output data", __func__);
                }
            }
            outEntry.ed = currentOffset + res.data.size();
            currentOffset = outEntry.ed;
            if (!write_all(fdIndex, &outEntry, sizeof(outEntry))) {
                close(fdMain);
                close(fdIndex);
                error("%s: Failed writing output index entry", __func__);
            }
            if (res.nCollisionIgnored > 0) {
                warning("%s: Ignored %lu colliding records in tile (%d, %d)",
                    __func__, res.nCollisionIgnored, res.row, res.col);
            }
            if (res.nOutOfRangeIgnored > 0) {
                warning("%s: Ignored %lu out-of-tile records in tile (%d, %d)",
                    __func__, res.nOutOfRangeIgnored, res.row, res.col);
            }
            ++nProcessed;
            if (nProcessed % 10 == 0) {
                notice("%s: Processing tile %lu/%lu", __func__, nProcessed, nTiles);
            }
        }
    }

    close(fdMain); close(fdIndex);
    notice("%s: Wrote smoothed output to %s (index: %s)", __func__, outFile.c_str(), outIndex.c_str());
}

void TileOperator::spatialMetricsBasic(const std::string& outPrefix) {
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (!regular_labeled_raster_ || coord_dim_ != 2) {
        error("%s only supports raster 2D data with regular tiles", __func__);
    }
    if (K_ <= 0 || K_ > 255) {
        error("%s: K must be in [1, 255], got %d", __func__, K_);
    }
    const int32_t K = K_;
    const uint8_t BG = static_cast<uint8_t>(K);
    auto processTile = [&](const TileInfo& blk,
            std::ifstream& in, SpatialMetricsAccum& out,
            uint64_t& nOutOfRangeIgnored, uint64_t& nBadLabelIgnored) {
        DenseTile dense;
        loadDenseTile(blk, in, dense, BG, nOutOfRangeIgnored, nBadLabelIgnored);
        const size_t width = dense.W;
        const size_t height = dense.H;
        const size_t nPixels = width * height;
        const std::vector<uint8_t>& labels = dense.lab;

        for (size_t i = 0; i < nPixels; ++i) {
            uint8_t a = labels[i];
            if (a < BG) out.area[a]++;
        }

        if (width > 1) { // right edge
            for (size_t y = 0; y < height; ++y) {
                const size_t row = y * width;
                for (size_t x = 0; x + 1 < width; ++x) {
                    spatialAccumulateEdge(out, labels[row + x], labels[row + x + 1]);
                }
            }
        }
        if (height > 1) { // down edge
            for (size_t y = 0; y + 1 < height; ++y) {
                const size_t row = y * width;
                const size_t rowDown = row + width;
                for (size_t x = 0; x < width; ++x) {
                    spatialAccumulateEdge(out, labels[row + x], labels[rowDown + x]);
                }
            }
        }
    };

    SpatialMetricsAccum total(K);
    uint64_t totalOutOfRangeIgnored = 0;
    uint64_t totalBadLabelIgnored = 0;
    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<SpatialMetricsAccum> tls([&] { return SpatialMetricsAccum(K); });
        tbb::combinable<std::pair<uint64_t, uint64_t>> tlsIgnored(
            [] { return std::make_pair(0ULL, 0ULL); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in; // thread-local input stream
                if (mode_ & 0x1) {
                    in.open(dataFile_, std::ios::binary);
                } else {
                    in.open(dataFile_);
                }
                if (!in.is_open()) {
                    error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
                }
                auto& local = tls.local();
                auto& localIgnored = tlsIgnored.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processTile(blocks_[bi], in, local, localIgnored.first, localIgnored.second);
                }
            });
        tls.combine_each([&](const SpatialMetricsAccum& local) {
            total.add(local);
        });
        tlsIgnored.combine_each([&](const std::pair<uint64_t, uint64_t>& localIgnored) {
            totalOutOfRangeIgnored += localIgnored.first;
            totalBadLabelIgnored += localIgnored.second;
        });
    } else {
        std::ifstream in;
        if (mode_ & 0x1) {
            in.open(dataFile_, std::ios::binary);
        } else {
            in.open(dataFile_);
        }
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        for (const auto& blk : blocks_) {
            processTile(blk, in, total, totalOutOfRangeIgnored, totalBadLabelIgnored);
        }
    }

    std::string outSingle = outPrefix + ".stats.single.tsv";
    FILE* fpSingle = fopen(outSingle.c_str(), "w");
    if (!fpSingle) {
        error("%s: Cannot open output file %s", __func__, outSingle.c_str());
    }
    fprintf(fpSingle, "#k\tarea\tperim\tperim_bg\n");
    for (int32_t k = 0; k < K; ++k) {
        fprintf(fpSingle, "%d\t%llu\t%llu\t%llu\n",
            k,
            static_cast<unsigned long long>(total.area[k]),
            static_cast<unsigned long long>(total.perim[k]),
            static_cast<unsigned long long>(total.perim_bg[k]));
    }
    fclose(fpSingle);
    notice("%s: Wrote single-channel metrics to %s", __func__, outSingle.c_str());

    std::string outPairwise = outPrefix + ".stats.pairwise.tsv";
    FILE* fpPairwise = fopen(outPairwise.c_str(), "w");
    if (!fpPairwise) {
        error("%s: Cannot open output file %s", __func__, outPairwise.c_str());
    }
    fprintf(fpPairwise, "#k\tl\tcontact\tfrac0\tfrac1\tfrac2\tdensity\n");
    for (int32_t k = 0; k < K; ++k) {
        double ak = std::max(1., static_cast<double>(total.area[k]));
        double pk = std::max(1., static_cast<double>(total.perim[k]));
        for (int32_t l = k + 1; l <= K; ++l) {
            double al = std::max(1., static_cast<double>(total.area[l]));
            double pl = std::max(1., static_cast<double>(total.perim[l]));
            double pkl = static_cast<double>(total.shared_ij[total.triIndex(k, l)]);
            fprintf(fpPairwise, "%d\t%d\t%llu\t%.2e\t%.2e\t%.2e\t%.2e\n",
                k, l,
                static_cast<unsigned long long>(total.shared_ij[total.triIndex(k, l)]),
                pkl / (pk + pl - pkl), pkl / pk, pkl / pl, pkl / (ak + al)
            );
        }
    }
    fclose(fpPairwise);
    notice("%s: Wrote pairwise metrics to %s", __func__, outPairwise.c_str());
    if (totalOutOfRangeIgnored > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(totalOutOfRangeIgnored));
    }
    if (totalBadLabelIgnored > 0) {
        warning("%s: Ignored %llu records with invalid labels",
            __func__, static_cast<unsigned long long>(totalBadLabelIgnored));
    }
}

void TileOperator::connectedComponents(const std::string& outPrefix, uint32_t minSize) {
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (!regular_labeled_raster_ || coord_dim_ != 2) {
        error("%s only supports raster 2D data with regular tiles", __func__);
    }
    if (K_ <= 0 || K_ > 255) {
        error("%s: K must be in [1, 255], got %d", __func__, K_);
    }
    const int32_t K = K_;
    const uint8_t BG = static_cast<uint8_t>(K);
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    const auto tStage1Start = std::chrono::steady_clock::now();

    std::vector<TileCCL> perTile(blocks_.size());
    uint64_t totalOutOfRangeIgnored = 0;
    uint64_t totalBadLabelIgnored = 0;
    // Stage 1: tile-local CCL
    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<std::pair<uint64_t, uint64_t>> tlsIgnored(
            [] { return std::make_pair(0ULL, 0ULL); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in;
                if (mode_ & 0x1) {
                    in.open(dataFile_, std::ios::binary);
                } else {
                    in.open(dataFile_);
                }
                if (!in.is_open()) {
                    error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
                }
                auto& localIgnored = tlsIgnored.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    DenseTile dense;
                    loadDenseTile(blocks_[bi], in, dense, BG, localIgnored.first, localIgnored.second);
                    perTile[bi] = tileLocalCCL(dense, BG);
                }
            });
        tlsIgnored.combine_each([&](const std::pair<uint64_t, uint64_t>& localIgnored) {
            totalOutOfRangeIgnored += localIgnored.first;
            totalBadLabelIgnored += localIgnored.second;
        });
    } else {
        std::ifstream in;
        if (mode_ & 0x1) {
            in.open(dataFile_, std::ios::binary);
        } else {
            in.open(dataFile_);
        }
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            DenseTile dense;
            loadDenseTile(blocks_[bi], in, dense, BG, totalOutOfRangeIgnored, totalBadLabelIgnored);
            perTile[bi] = tileLocalCCL(dense, BG);
        }
    }
    const auto tStage1End = std::chrono::steady_clock::now();

    struct CCOutRec {
        uint64_t size = 0;
        uint64_t sumX = 0;
        uint64_t sumY = 0;
        PixBox box;
    };
    std::vector<std::vector<CCOutRec>> perLabelComponents(static_cast<size_t>(K));
    std::vector<std::unordered_map<uint64_t, uint64_t>> perLabelHist(static_cast<size_t>(K));
    auto addFinalComponent = [&](uint8_t lbl, uint64_t size,
                                 uint64_t sumX, uint64_t sumY,
                                 const PixBox& box) {
        if (lbl >= BG || size == 0) return;
        perLabelHist[static_cast<size_t>(lbl)][size] += 1;
        if (size >= static_cast<uint64_t>(minSize)) {
            perLabelComponents[static_cast<size_t>(lbl)].push_back(CCOutRec{size, sumX, sumY, box});
        }
    };

    // Stage 1.5: local-finalize non-border components
    const auto tStage15Start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < perTile.size(); ++i) {
        auto& t = perTile[i];
        if (t.ncomp == 0) continue;
        const BorderRemapInfo remapInfo = remapTileToBorderComponents(t, INVALID);
        for (uint32_t cid = 0; cid < remapInfo.remap.size(); ++cid) {
            if (remapInfo.remap[cid] != INVALID) continue;
            addFinalComponent(
                remapInfo.oldCompLabel[cid],
                static_cast<uint64_t>(remapInfo.oldCompSize[cid]),
                remapInfo.oldCompSumX[cid],
                remapInfo.oldCompSumY[cid],
                remapInfo.oldCompBox[cid]);
        }
    }
    const auto tStage15End = std::chrono::steady_clock::now();

    // Stage 2: seam union on border-touching components only
    const auto tStage2Start = std::chrono::steady_clock::now();
    const BorderDSUState dsuState = mergeBorderComponentsWithDSU(perTile, BG, INVALID);
    for (size_t i = 0; i < dsuState.rootSize.size(); ++i) {
        if (dsuState.rootSize[i] == 0 || dsuState.rootLabel[i] >= BG) continue;
        addFinalComponent(
            dsuState.rootLabel[i],
            dsuState.rootSize[i],
            dsuState.rootSumX[i],
            dsuState.rootSumY[i],
            dsuState.rootBox[i]);
    }
    const auto tStage2End = std::chrono::steady_clock::now();
    const auto stage1Ms = std::chrono::duration_cast<std::chrono::milliseconds>(tStage1End - tStage1Start).count();
    const auto stage15Ms = std::chrono::duration_cast<std::chrono::milliseconds>(tStage15End - tStage15Start).count();
    const auto stage2Ms = std::chrono::duration_cast<std::chrono::milliseconds>(tStage2End - tStage2Start).count();
    notice("%s: timing(ms): stage1=%lld stage1.5=%lld stage2=%lld",
        __func__,
        static_cast<long long>(stage1Ms),
        static_cast<long long>(stage15Ms),
        static_cast<long long>(stage2Ms));

    std::string outCc = outPrefix + ".connected_components.tsv";
    FILE* fpCc = fopen(outCc.c_str(), "w");
    if (!fpCc) {
        error("%s: Cannot open output file %s", __func__, outCc.c_str());
    }
    fprintf(fpCc, "#k\tcc_idx\tsize\tcentroid_x\tcentroid_y\txmin\txmax\tymin\tymax\n");
    for (int32_t k = 0; k < K; ++k) {
        auto& vec = perLabelComponents[static_cast<size_t>(k)];
        std::sort(vec.begin(), vec.end(),
            [](const CCOutRec& a, const CCOutRec& b) {
                if (a.size != b.size) return a.size > b.size;
                if (a.box.minX != b.box.minX) return a.box.minX < b.box.minX;
                if (a.box.minY != b.box.minY) return a.box.minY < b.box.minY;
                if (a.box.maxX != b.box.maxX) return a.box.maxX < b.box.maxX;
                return a.box.maxY < b.box.maxY;
            });
        for (size_t i = 0; i < vec.size(); ++i) {
            const double cx = static_cast<double>(vec[i].sumX) / static_cast<double>(vec[i].size);
            const double cy = static_cast<double>(vec[i].sumY) / static_cast<double>(vec[i].size);
            fprintf(fpCc, "%d\t%zu\t%llu\t%.6f\t%.6f\t%d\t%d\t%d\t%d\n",
                k, i,
                static_cast<unsigned long long>(vec[i].size),
                cx,
                cy,
                vec[i].box.minX,
                vec[i].box.maxX,
                vec[i].box.minY,
                vec[i].box.maxY);
        }
    }
    fclose(fpCc);
    notice("%s: Wrote connected components to %s", __func__, outCc.c_str());

    std::string outHist = outPrefix + ".connected_components_hist.tsv";
    FILE* fpHist = fopen(outHist.c_str(), "w");
    if (!fpHist) {
        error("%s: Cannot open output file %s", __func__, outHist.c_str());
    }
    fprintf(fpHist, "#k\tsize\tn_components\n");
    for (int32_t k = 0; k < K; ++k) {
        const auto& hist = perLabelHist[static_cast<size_t>(k)];
        for (const auto& kv : hist) {
            fprintf(fpHist, "%d\t%llu\t%llu\n",
                k,
                static_cast<unsigned long long>(kv.first),
                static_cast<unsigned long long>(kv.second));
        }
    }
    fclose(fpHist);
    notice("%s: Wrote connected component size histogram to %s", __func__, outHist.c_str());

    if (totalOutOfRangeIgnored > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(totalOutOfRangeIgnored));
    }
    if (totalBadLabelIgnored > 0) {
        warning("%s: Ignored %llu records with invalid labels",
            __func__, static_cast<unsigned long long>(totalBadLabelIgnored));
    }
}

void TileOperator::profileShellAndSurface(const std::string& outPrefix,
    const std::vector<int32_t>& radii, int32_t dMax,
    uint32_t minCompSize, uint32_t minPixPerTilePerLabel) {
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (!regular_labeled_raster_ || coord_dim_ != 2) {
        error("%s only supports raster 2D data with regular tiles", __func__);
    }
    if (K_ <= 0 || K_ > 255) {
        error("%s: K must be in [1, 255], got %d", __func__, K_);
    }
    if (dMax < 0) {
        error("%s: dMax must be >= 0", __func__);
    }
    std::vector<int32_t> radiiSorted;
    radiiSorted.reserve(radii.size());
    for (int32_t r : radii) {
        if (r < 0) continue;
        radiiSorted.push_back(r);
    }
    if (radiiSorted.empty()) {
        error("%s: At least one non-negative radius is required", __func__);
    }
    std::sort(radiiSorted.begin(), radiiSorted.end());
    radiiSorted.erase(std::unique(radiiSorted.begin(), radiiSorted.end()), radiiSorted.end());
    const int32_t rMax = radiiSorted.back();
    const int32_t D = std::max(rMax, dMax + 1);
    if (D > 65534) {
        error("%s: Distance cap too large (%d), must be <= 65534", __func__, D);
    }

    const int32_t K = K_;
    const uint8_t BG = static_cast<uint8_t>(K);
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();

    // Stage 1: load all tiles and collect global/tile label counts
    std::vector<DenseTile> tiles(blocks_.size());
    std::vector<std::vector<uint32_t>> tileLabelCount(
        blocks_.size(), std::vector<uint32_t>(static_cast<size_t>(K), 0));
    std::vector<uint64_t> areaTot(static_cast<size_t>(K + 1), 0);
    uint64_t totalOutOfRangeIgnored = 0;
    uint64_t totalBadLabelIgnored = 0;

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<std::vector<uint64_t>> tlsArea([&] {
            return std::vector<uint64_t>(static_cast<size_t>(K + 1), 0);
        });
        tbb::combinable<std::pair<uint64_t, uint64_t>> tlsIgnored(
            [] { return std::make_pair(0ULL, 0ULL); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in;
                if (mode_ & 0x1) {
                    in.open(dataFile_, std::ios::binary);
                } else {
                    in.open(dataFile_);
                }
                if (!in.is_open()) {
                    error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
                }
                auto& localArea = tlsArea.local();
                auto& localIgnored = tlsIgnored.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    loadDenseTile(blocks_[bi], in, tiles[bi], BG, localIgnored.first, localIgnored.second);
                    if (tiles[bi].lab.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                        error("%s: Tile too large for local indexing", __func__);
                    }
                    auto& counts = tileLabelCount[bi];
                    for (size_t li = 0; li < tiles[bi].lab.size(); ++li) {
                        const uint8_t lbl = tiles[bi].lab[li];
                        localArea[lbl] += 1;
                        if (lbl < BG) counts[lbl] += 1;
                    }
                }
            });
        tlsArea.combine_each([&](const std::vector<uint64_t>& localArea) {
            for (size_t i = 0; i < areaTot.size(); ++i) areaTot[i] += localArea[i];
        });
        tlsIgnored.combine_each([&](const std::pair<uint64_t, uint64_t>& localIgnored) {
            totalOutOfRangeIgnored += localIgnored.first;
            totalBadLabelIgnored += localIgnored.second;
        });
    } else {
        std::ifstream in;
        if (mode_ & 0x1) {
            in.open(dataFile_, std::ios::binary);
        } else {
            in.open(dataFile_);
        }
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            loadDenseTile(blocks_[bi], in, tiles[bi], BG, totalOutOfRangeIgnored, totalBadLabelIgnored);
            if (tiles[bi].lab.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
                error("%s: Tile too large for local indexing", __func__);
            }
            auto& counts = tileLabelCount[bi];
            for (size_t li = 0; li < tiles[bi].lab.size(); ++li) {
                const uint8_t lbl = tiles[bi].lab[li];
                areaTot[lbl] += 1;
                if (lbl < BG) counts[lbl] += 1;
            }
        }
    }

    // Stage 1.25: compute seam-aware boundary masks, then local CCL with boundary->cid tracking.
    computeTileBoundaryMasks(tiles);
    std::vector<TileCCL> perTile(tiles.size());
    if (threads_ > 1 && tiles.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, tiles.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    perTile[bi] = tileLocalCCL(tiles[bi], BG, &tiles[bi].boundary);
                }
            });
    } else {
        for (size_t bi = 0; bi < tiles.size(); ++bi) {
            perTile[bi] = tileLocalCCL(tiles[bi], BG, &tiles[bi].boundary);
        }
    }

    // Stage 1.5: finalize non-border components and keep remapped border components.
    std::vector<std::vector<PixelRef>> seedsBoundaryBig(static_cast<size_t>(K));
    std::vector<std::vector<uint8_t>> tileCompBig(tiles.size());
    for (size_t i = 0; i < perTile.size(); ++i) {
        auto& t = perTile[i];
        if (t.ncomp == 0) continue;
        const BorderRemapInfo remapInfo = remapTileToBorderComponents(t, INVALID);

        std::vector<uint32_t> bndPixKeep;
        std::vector<uint32_t> bndCidKeep;
        bndPixKeep.reserve(t.bndPix.size());
        bndCidKeep.reserve(t.bndCid.size());
        for (size_t bi = 0; bi < t.bndPix.size(); ++bi) {
            const uint32_t oldCid = t.bndCid[bi];
            if (oldCid >= remapInfo.remap.size()) {
                error("%s: Boundary cid out of range", __func__);
            }
            const uint32_t pix = t.bndPix[bi];
            const uint8_t lbl = tiles[i].lab[pix];
            if (lbl >= BG) continue;
            if (remapInfo.remap[oldCid] == INVALID) {
                if (remapInfo.oldCompSize[oldCid] < minCompSize) continue;
                if (minPixPerTilePerLabel > 0 &&
                    tileLabelCount[i][static_cast<size_t>(lbl)] < minPixPerTilePerLabel) {
                    continue;
                }
                seedsBoundaryBig[static_cast<size_t>(lbl)].push_back(
                    PixelRef{static_cast<uint32_t>(i), pix});
            } else {
                bndPixKeep.push_back(pix);
                bndCidKeep.push_back(remapInfo.remap[oldCid]);
            }
        }
        t.bndPix.swap(bndPixKeep);
        t.bndCid.swap(bndCidKeep);
        tileCompBig[i].assign(static_cast<size_t>(t.ncomp), 0);
    }

    // Stage 2: seam union for border components only and finalize big flags/seeds.
    const BorderDSUState dsuState = mergeBorderComponentsWithDSU(perTile, BG, INVALID);
    for (size_t i = 0; i < perTile.size(); ++i) {
        auto& t = perTile[i];
        auto& big = tileCompBig[i];
        if (big.size() != static_cast<size_t>(t.ncomp)) {
            big.assign(static_cast<size_t>(t.ncomp), 0);
        }
        const auto& tileRoot = dsuState.tileRoot[i];
        for (size_t cid = 0; cid < t.ncomp; ++cid) {
            if (cid >= tileRoot.size()) {
                error("%s: Missing root id for border component", __func__);
            }
            const size_t root = tileRoot[cid];
            big[cid] = (root < dsuState.rootSize.size() && dsuState.rootSize[root] >= minCompSize) ? 1 : 0;
        }
        for (size_t bi = 0; bi < t.bndPix.size(); ++bi) {
            const uint32_t cid = t.bndCid[bi];
            if (cid >= big.size() || !big[cid]) continue;
            const uint32_t pix = t.bndPix[bi];
            const uint8_t lbl = tiles[i].lab[pix];
            if (lbl >= BG) continue;
            if (minPixPerTilePerLabel > 0 &&
                tileLabelCount[i][static_cast<size_t>(lbl)] < minPixPerTilePerLabel) {
                continue;
            }
            seedsBoundaryBig[static_cast<size_t>(lbl)].push_back(
                PixelRef{static_cast<uint32_t>(i), pix});
        }
    }

    struct TileNeighbor {
        int32_t left = -1;
        int32_t right = -1;
        int32_t up = -1;
        int32_t down = -1;
    };
    struct TileSeamJump {
        std::vector<uint32_t> left;
        std::vector<uint32_t> right;
        std::vector<uint32_t> up;
        std::vector<uint32_t> down;
    };
    std::vector<TileNeighbor> nbr(tiles.size());
    std::vector<TileSeamJump> seam(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
        const TileKey key = tiles[i].key;
        auto it = tile_lookup_.find(TileKey{key.row, key.col - 1});
        if (it != tile_lookup_.end()) nbr[i].left = static_cast<int32_t>(it->second);
        it = tile_lookup_.find(TileKey{key.row, key.col + 1});
        if (it != tile_lookup_.end()) nbr[i].right = static_cast<int32_t>(it->second);
        it = tile_lookup_.find(TileKey{key.row - 1, key.col});
        if (it != tile_lookup_.end()) nbr[i].up = static_cast<int32_t>(it->second);
        it = tile_lookup_.find(TileKey{key.row + 1, key.col});
        if (it != tile_lookup_.end()) nbr[i].down = static_cast<int32_t>(it->second);

        auto& sj = seam[i];
        sj.left.assign(tiles[i].H, INVALID);
        sj.right.assign(tiles[i].H, INVALID);
        sj.up.assign(tiles[i].W, INVALID);
        sj.down.assign(tiles[i].W, INVALID);

        if (nbr[i].left >= 0) {
            const DenseTile& nt = tiles[static_cast<size_t>(nbr[i].left)];
            for (size_t y = 0; y < tiles[i].H; ++y) {
                const int32_t ngx = tiles[i].pixX0 - 1;
                const int32_t ngy = tiles[i].pixY0 + static_cast<int32_t>(y);
                if (ngx < nt.pixX0 || ngx >= nt.pixX1 || ngy < nt.pixY0 || ngy >= nt.pixY1) continue;
                const size_t nx = static_cast<size_t>(ngx - nt.pixX0);
                const size_t ny = static_cast<size_t>(ngy - nt.pixY0);
                sj.left[y] = static_cast<uint32_t>(ny * nt.W + nx);
            }
        }
        if (nbr[i].right >= 0) {
            const DenseTile& nt = tiles[static_cast<size_t>(nbr[i].right)];
            for (size_t y = 0; y < tiles[i].H; ++y) {
                const int32_t ngx = tiles[i].pixX1;
                const int32_t ngy = tiles[i].pixY0 + static_cast<int32_t>(y);
                if (ngx < nt.pixX0 || ngx >= nt.pixX1 || ngy < nt.pixY0 || ngy >= nt.pixY1) continue;
                const size_t nx = static_cast<size_t>(ngx - nt.pixX0);
                const size_t ny = static_cast<size_t>(ngy - nt.pixY0);
                sj.right[y] = static_cast<uint32_t>(ny * nt.W + nx);
            }
        }
        if (nbr[i].up >= 0) {
            const DenseTile& nt = tiles[static_cast<size_t>(nbr[i].up)];
            for (size_t x = 0; x < tiles[i].W; ++x) {
                const int32_t ngx = tiles[i].pixX0 + static_cast<int32_t>(x);
                const int32_t ngy = tiles[i].pixY0 - 1;
                if (ngx < nt.pixX0 || ngx >= nt.pixX1 || ngy < nt.pixY0 || ngy >= nt.pixY1) continue;
                const size_t nx = static_cast<size_t>(ngx - nt.pixX0);
                const size_t ny = static_cast<size_t>(ngy - nt.pixY0);
                sj.up[x] = static_cast<uint32_t>(ny * nt.W + nx);
            }
        }
        if (nbr[i].down >= 0) {
            const DenseTile& nt = tiles[static_cast<size_t>(nbr[i].down)];
            for (size_t x = 0; x < tiles[i].W; ++x) {
                const int32_t ngx = tiles[i].pixX0 + static_cast<int32_t>(x);
                const int32_t ngy = tiles[i].pixY1;
                if (ngx < nt.pixX0 || ngx >= nt.pixX1 || ngy < nt.pixY0 || ngy >= nt.pixY1) continue;
                const size_t nx = static_cast<size_t>(ngx - nt.pixX0);
                const size_t ny = static_cast<size_t>(ngy - nt.pixY0);
                sj.down[x] = static_cast<uint32_t>(ny * nt.W + nx);
            }
        }
    }

    const uint16_t INF = static_cast<uint16_t>(D + 1);
    struct TileDistBuf {
        std::vector<uint16_t> dist;
        std::vector<uint32_t> touched;
    };
    std::vector<TileDistBuf> distBuf(tiles.size());
    for (size_t i = 0; i < tiles.size(); ++i) {
        distBuf[i].dist.assign(tiles[i].lab.size(), INF);
        distBuf[i].touched.reserve(std::min<size_t>(tiles[i].lab.size(), 4096));
    }
    std::vector<uint8_t> tileActiveMark(tiles.size(), 0);
    std::vector<uint32_t> activeTiles;
    activeTiles.reserve(std::min<size_t>(tiles.size(), 1024));

    std::string outShell = outPrefix + ".shell.tsv";
    FILE* fpShell = fopen(outShell.c_str(), "w");
    if (!fpShell) {
        error("%s: Cannot open output file %s", __func__, outShell.c_str());
    }
    fprintf(fpShell, "#K_focal\tK2\tr\tn_within\tn_K2_total\n");

    std::string outSurface = outPrefix + ".surface.tsv";
    FILE* fpSurface = fopen(outSurface.c_str(), "w");
    if (!fpSurface) {
        fclose(fpShell);
        error("%s: Cannot open output file %s", __func__, outSurface.c_str());
    }
    fprintf(fpSurface, "#from_K1\tto_K2\td\tcount\n");

    auto shellIndex = [&](int32_t b, int32_t d) -> size_t {
        return static_cast<size_t>(b) * static_cast<size_t>(rMax + 1) + static_cast<size_t>(d);
    };
    auto surfIndex = [&](int32_t b, int32_t d) -> size_t {
        return static_cast<size_t>(b) * static_cast<size_t>(dMax + 1) + static_cast<size_t>(d);
    };

    std::vector<PixelRef> frontier;
    std::vector<PixelRef> nextFrontier;
    for (int32_t A = 0; A < K; ++A) {
        for (uint32_t ti : activeTiles) {
            auto& db = distBuf[ti];
            if (db.touched.size() > db.dist.size() / 2) {
                std::fill(db.dist.begin(), db.dist.end(), INF);
            } else {
                for (uint32_t idx : db.touched) db.dist[idx] = INF;
            }
            tileActiveMark[ti] = 0;
        }
        activeTiles.clear();

        frontier.clear();
        nextFrontier.clear();
        const auto& seedsA = seedsBoundaryBig[static_cast<size_t>(A)];
        frontier.reserve(seedsA.size());
        for (const auto& ref : seedsA) {
            auto& db = distBuf[ref.tileIdx];
            if (db.dist[ref.localIdx] == 0) continue;
            db.dist[ref.localIdx] = 0;
            db.touched.push_back(ref.localIdx);
            if (!tileActiveMark[ref.tileIdx]) {
                tileActiveMark[ref.tileIdx] = 1;
                activeTiles.push_back(ref.tileIdx);
            }
            frontier.push_back(ref);
        }
        if (frontier.empty()) continue;

        std::vector<uint64_t> shellCount(static_cast<size_t>(K + 1) * static_cast<size_t>(rMax + 1), 0);
        std::vector<uint64_t> surfHist(static_cast<size_t>(K + 1) * static_cast<size_t>(dMax + 1), 0);

        for (int32_t d = 0; d <= D && !frontier.empty(); ++d) {
            nextFrontier.clear();
            auto tryPush = [&](uint32_t nti, uint32_t nli, uint16_t nd) {
                if (tiles[nti].lab[nli] == static_cast<uint8_t>(A)) return;
                auto& db = distBuf[nti];
                if (db.dist[nli] != INF) return;
                db.dist[nli] = nd;
                db.touched.push_back(nli);
                if (!tileActiveMark[nti]) {
                    tileActiveMark[nti] = 1;
                    activeTiles.push_back(nti);
                }
                nextFrontier.push_back(PixelRef{nti, nli});
            };

            for (const auto& ref : frontier) {
                const DenseTile& tile = tiles[ref.tileIdx];
                const size_t li = ref.localIdx;
                const uint8_t B = tile.lab[li];
                if (B != static_cast<uint8_t>(A)) {
                    if (d <= rMax) {
                        shellCount[shellIndex(B, d)] += 1;
                    }
                    if (tile.boundary[li]) {
                        const int32_t dReport = (d > 0) ? (d - 1) : 0;
                        if (dReport <= dMax) {
                            surfHist[surfIndex(B, dReport)] += 1;
                        }
                    }
                }
                if (d == D) continue;

                const size_t x = li % tile.W;
                const size_t y = li / tile.W;
                const uint16_t nd = static_cast<uint16_t>(d + 1);
                if (x > 0) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li - 1), nd);
                } else if (nbr[ref.tileIdx].left >= 0) {
                    const uint32_t nli = seam[ref.tileIdx].left[y];
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].left), nli, nd);
                    }
                }
                if (x + 1 < tile.W) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li + 1), nd);
                } else if (nbr[ref.tileIdx].right >= 0) {
                    const uint32_t nli = seam[ref.tileIdx].right[y];
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].right), nli, nd);
                    }
                }
                if (y > 0) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li - tile.W), nd);
                } else if (nbr[ref.tileIdx].up >= 0) {
                    const uint32_t nli = seam[ref.tileIdx].up[x];
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].up), nli, nd);
                    }
                }
                if (y + 1 < tile.H) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li + tile.W), nd);
                } else if (nbr[ref.tileIdx].down >= 0) {
                    const uint32_t nli = seam[ref.tileIdx].down[x];
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].down), nli, nd);
                    }
                }
            }
            frontier.swap(nextFrontier);
        }

        for (int32_t b = 0; b <= K; ++b) {
            for (int32_t d = 1; d <= rMax; ++d) {
                shellCount[shellIndex(b, d)] += shellCount[shellIndex(b, d - 1)];
            }
        }
        for (int32_t b = 0; b <= K; ++b) {
            if (b == A) continue;
            const uint64_t totalB = areaTot[static_cast<size_t>(b)];
            for (int32_t r : radiiSorted) {
                const uint64_t within = shellCount[shellIndex(b, r)];
                fprintf(fpShell, "%d\t%d\t%d\t%llu\t%llu\n",
                    A, b, r,
                    static_cast<unsigned long long>(within),
                    static_cast<unsigned long long>(totalB));
            }
        }
        for (int32_t b = 0; b <= K; ++b) {
            if (b == A) continue;
            for (int32_t d = 0; d <= dMax; ++d) {
                const uint64_t ct = surfHist[surfIndex(b, d)];
                if (ct == 0) continue;
                fprintf(fpSurface, "%d\t%d\t%d\t%llu\n",
                    b, A, d, static_cast<unsigned long long>(ct));
            }
        }
    }

    fclose(fpShell);
    fclose(fpSurface);
    notice("%s: Wrote shell occupancy to %s", __func__, outShell.c_str());
    notice("%s: Wrote surface distance histogram to %s", __func__, outSurface.c_str());
    if (totalOutOfRangeIgnored > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(totalOutOfRangeIgnored));
    }
    if (totalBadLabelIgnored > 0) {
        warning("%s: Ignored %llu records with invalid labels",
            __func__, static_cast<unsigned long long>(totalBadLabelIgnored));
    }
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

std::unordered_map<int32_t, TileOperator::Slice> TileOperator::aggOneTile(
    std::map<std::pair<int32_t, int32_t>, TopProbs>& pixelMap,
    TileReader& reader, lineParserUnival& parser, TileKey tile, double gridSize, double minProb, int32_t union_key) const {
    if (coord_dim_ == 3) {error("%s does not support 3D data yet", __func__);}
    assert(reader.getTileSize() == formatInfo_.tileSize);

    std::unordered_map<int32_t, Slice> tileAgg; // k -> unit key ->
    if (pixelMap.size() == 0) {return tileAgg;}
    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {return tileAgg;}
    auto aggIt0 = tileAgg.emplace(union_key, Slice()).first;
    auto& oneSlice0 = aggIt0->second;

    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;

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
        if (union_key == 0) continue;
        auto unitIt = oneSlice0.find({ux, uy});
        if (unitIt == oneSlice0.end()) {
            unitIt = oneSlice0.emplace(std::make_pair(ux, uy), SparseObsDict()).first;
        }
        unitIt->second.add(rec.idx, rec.ct);
    }
    return tileAgg;
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TileOperator::computeConfusionMatrix(double resolution, const char* outPref, int32_t probDigits) const {
    if (coord_dim_ != 2) {error("%s: only 2D data are supported", __func__);}
    if (K_ <= 0) {error("%s: K is 0 or unknown", __func__);}
    const int32_t K = K_;
    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;
    if (resolution > 0) res /= resolution;
    std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
    int32_t nTiles = 0;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> confusion;
    confusion.setZero(K, K);
    for (const auto& tileInfo : blocks_) {
        nTiles++;
        if (nTiles % 10 == 0) {
            notice("%s: Processed %d tiles...", __func__, nTiles);
        }
        TileKey tile{tileInfo.row, tileInfo.col};
        if (loadTileToMap(tile, pixelMap) <= 0) {
            continue;
        }
        if (resolution > 0) {
            std::unordered_map<std::pair<int32_t, int32_t>, Eigen::VectorXd, PairHash> squareSums;
            for (const auto& kv : pixelMap) {
                const auto& coord = kv.first;
                int32_t sx = static_cast<int32_t>(std::floor(coord.first * res));
                int32_t sy = static_cast<int32_t>(std::floor(coord.second* res));
                auto& merged = squareSums[std::make_pair(sx, sy)];
                if (merged.size() == 0) {
                    merged = Eigen::VectorXd::Zero(K);
                }
                const TopProbs& tp = kv.second;
                for (size_t i = 0; i < tp.ks.size(); ++i) {
                    int32_t k = tp.ks[i];
                    if (k < 0 || k >= K) {
                        error("%s: factor index %d out of range [0, %d)", __func__, k, K);
                    }
                    merged[k] += tp.ps[i];
                }
            }
            for (auto& kv : squareSums) {
                auto& merged = kv.second;
                double w = merged.sum();
                if (w == 0.0) continue;
                merged = merged.array() / w;
                confusion += merged * merged.transpose() * w;
            }
        } else {
            for (const auto& kv : pixelMap) {
                const TopProbs& tp = kv.second;
                for (size_t i = 0; i < tp.ks.size(); ++i) {
                    int32_t k1 = tp.ks[i];
                    float p1 = tp.ps[i];
                    for (size_t j = 0; j < tp.ks.size(); ++j) {
                        int32_t k2 = tp.ks[j];
                        float p2 = tp.ps[j];
                        confusion(k1, k2) += static_cast<double>(p1 * p2);
                    }
                }
            }
        }
    }

    if (outPref) {
        std::vector<std::string> factorNames(K);
        for (int32_t k = 0; k < K; ++k) {factorNames[k] = std::to_string(k);}
        std::string outFile(outPref);
        outFile += ".confusion.tsv";
        write_matrix_to_file(outFile, confusion, probDigits, true, factorNames, "K", &factorNames);
        notice("Confusion matrix written to %s", outFile.c_str());
    }

    return confusion;
}
