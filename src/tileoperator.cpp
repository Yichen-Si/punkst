#include "tileoperator.hpp"
#include "region_query.hpp"
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

void write_factor_mask_info(const std::string& outHist,
    const std::vector<uint64_t>& tileCount,
    const std::vector<uint64_t>& maskArea,
    const std::vector<uint64_t>& nComponents) {
    FILE* fp = fopen(outHist.c_str(), "w");
    if (!fp) {
        error("%s: Cannot open output file %s", __func__, outHist.c_str());
    }
    fprintf(fp, "k\tn_tiles\tmask_area_pix\tn_components\n");
    for (size_t k = 0; k < maskArea.size(); ++k) {
        fprintf(fp, "%zu\t%llu\t%llu\t%llu\n",
            k,
            static_cast<unsigned long long>(tileCount[k]),
            static_cast<unsigned long long>(maskArea[k]),
            static_cast<unsigned long long>(nComponents[k]));
    }
    fclose(fp);
}

void write_factor_mask_component_histogram(const std::string& outHist,
    const std::vector<std::unordered_map<uint64_t, uint64_t>>& perFactorHist) {
    FILE* fp = fopen(outHist.c_str(), "w");
    if (!fp) {
        error("%s: Cannot open output file %s", __func__, outHist.c_str());
    }
    fprintf(fp, "k\tsize\tn_components\n");
    for (size_t k = 0; k < perFactorHist.size(); ++k) {
        std::vector<std::pair<uint64_t, uint64_t>> entries(
            perFactorHist[k].begin(), perFactorHist[k].end());
        std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& kv : entries) {
            fprintf(fp, "%zu\t%llu\t%llu\n",
                k,
                static_cast<unsigned long long>(kv.first),
                static_cast<unsigned long long>(kv.second));
        }
    }
    fclose(fp);
}

nlohmann::json load_template_json(const std::string& templateFile) {
    std::ifstream in(templateFile);
    if (!in.is_open()) {
        error("%s: Cannot open template GeoJSON file %s", __func__, templateFile.c_str());
    }
    nlohmann::json j;
    in >> j;
    return j;
}

void write_factor_template_json(const std::string& outFile, const nlohmann::json& templateJson,
    int32_t factor, const nlohmann::json& feature) {
    nlohmann::json outJson = templateJson;
    outJson["title"] = std::to_string(factor);
    outJson["geometry"] = feature;
    std::ofstream out(outFile);
    if (!out.is_open()) {
        error("%s: Cannot open output file %s", __func__, outFile.c_str());
    }
    out << outJson.dump();
    out << "\n";
    out.close();
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

size_t pair_index_upper(size_t i, size_t j, size_t n) {
    return i * (2 * n - i - 1) / 2 + (j - i - 1);
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
            decodeBinaryXY(recBuf.data(), x, y);

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

void TileOperator::profileSoftFactorMasks(const std::string& outPrefix,
    int32_t focalK, int32_t radius, double neighborhoodThreshold,
    double minFactorFrac, float minPixelProb, uint32_t minComponentArea, bool skipMaskOverlap) {
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (coord_dim_ != 2 || (mode_ & 0x8) != 0) {
        error("%s: Only 2D regular tiles are supported", __func__);
    }
    if (isTextInput() && !canSeekTextInput()) {
        error("%s: This operation requires seekable input", __func__);
    }
    if (K_ <= 0) {
        error("%s: K must be known and positive", __func__);
    }
    if (focalK < 0 || focalK >= K_) {
        error("%s: focalK=%d out of range [0, %d)", __func__, focalK, K_);
    }
    if (radius < 0) {
        error("%s: radius must be >= 0", __func__);
    }
    if (neighborhoodThreshold < 0.0) {
        error("%s: neighborhoodThreshold must be >= 0", __func__);
    }
    if (minFactorFrac < 0.0 || minFactorFrac >= 1.0) {
        error("%s: minFactorFrac must be in [0,1)", __func__);
    }
    if (minPixelProb < 0.0) {
        error("%s: minPixelProb must be >= 0", __func__);
    }

    struct Stage1Accum {
        std::vector<double> histMask;
        std::vector<double> histGlobal;
        uint64_t focalArea = 0;
        uint64_t outOfRange = 0;
        uint64_t badFactor = 0;
        uint64_t collisions = 0;

        explicit Stage1Accum(int32_t K)
            : histMask(static_cast<size_t>(K), 0.0),
              histGlobal(static_cast<size_t>(K), 0.0) {}
    };

    const auto t0 = std::chrono::steady_clock::now();
    Stage1Accum stage1Global(K_);

    auto openTileStream = [&]() {
        std::ifstream in;
        if (mode_ & 0x1) in.open(dataFile_, std::ios::binary);
        else in.open(dataFile_);
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        return in;
    };

    auto processStage1Tile = [&](size_t bi, std::ifstream& in, Stage1Accum& accum) {
        const TileInfo& blk = blocks_[bi];
        SoftMaskTileData tileData;
        loadSoftMaskTileData(blk, in, minPixelProb, true, tileData,
            accum.outOfRange, accum.badFactor, &accum.collisions, &accum.histGlobal);
        const size_t nPix = tileData.geom.W * tileData.geom.H;
        std::vector<float> denseFocal;
        std::vector<uint8_t> focalMask;
        auto focalIt = tileData.factorEntries.find(focalK);
        if (focalIt != tileData.factorEntries.end()) {
            buildDenseFactorRaster(focalIt->second, nPix, denseFocal);
        } else {
            denseFocal.assign(nPix, 0.0f);
        }
        buildSoftMaskBinary(denseFocal, tileData.seenLocal, tileData.geom.W, tileData.geom.H,
            radius, neighborhoodThreshold, SoftMaskThresholdConfig{}, focalMask);
        const uint64_t keptArea = filterMaskByMinComponentArea4(
            focalMask, tileData.geom.W, tileData.geom.H, minComponentArea);
        accum.focalArea += keptArea;
        for (const auto& stored : tileData.records) {
            if (!focalMask[stored.localIdx]) continue;
            const TopProbs& top = stored.rec;
            for (size_t i = 0; i < top.ks.size() && i < top.ps.size(); ++i) {
                const int32_t k = top.ks[i];
                if (k < 0 || k >= K_) continue;
                accum.histMask[static_cast<size_t>(k)] += static_cast<double>(top.ps[i]);
            }
        }

    };

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<Stage1Accum> tls([&] { return Stage1Accum(K_); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in = openTileStream();
                auto& local = tls.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processStage1Tile(bi, in, local);
                }
            });
        tls.combine_each([&](const Stage1Accum& local) {
            for (size_t k = 0; k < stage1Global.histMask.size(); ++k) {
                stage1Global.histMask[k] += local.histMask[k];
                stage1Global.histGlobal[k] += local.histGlobal[k];
            }
            stage1Global.focalArea += local.focalArea;
            stage1Global.outOfRange += local.outOfRange;
            stage1Global.badFactor += local.badFactor;
            stage1Global.collisions += local.collisions;
        });
    } else {
        std::ifstream in = openTileStream();
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            processStage1Tile(bi, in, stage1Global);
        }
    }
    const auto tStage1 = std::chrono::steady_clock::now();

    double histMaskTotal = std::accumulate(stage1Global.histMask.begin(), stage1Global.histMask.end(), 0.0);
    double histGlobalTotal = std::accumulate(stage1Global.histGlobal.begin(), stage1Global.histGlobal.end(), 0.0);
    std::string outHist = outPrefix + ".factor_hist.tsv";
    FILE* fpHist = fopen(outHist.c_str(), "w");
    if (!fpHist) {
        error("%s: Cannot open output file %s", __func__, outHist.c_str());
    }
    fprintf(fpHist, "k\tmass_in_mask\tfrac_in_mask\tmass_global\tfrac_global\n");
    for (int32_t k = 0; k < K_; ++k) {
        const double fracMask = (histMaskTotal > 0.0) ? stage1Global.histMask[static_cast<size_t>(k)] / histMaskTotal : 0.0;
        const double fracGlobal = (histGlobalTotal > 0.0) ? stage1Global.histGlobal[static_cast<size_t>(k)] / histGlobalTotal : 0.0;
        fprintf(fpHist, "%d\t%.3e\t%.3e\t%.3e\t%.3e\n",
            k,
            stage1Global.histMask[static_cast<size_t>(k)],
            fracMask,
            stage1Global.histGlobal[static_cast<size_t>(k)],
            fracGlobal);
    }
    fclose(fpHist);

    if (skipMaskOverlap) {
        return;
    }

    std::vector<int32_t> selectedFactors;
    selectedFactors.reserve(static_cast<size_t>(K_));
    selectedFactors.push_back(focalK);
    if (histMaskTotal > 0.0) {
        for (int32_t k = 0; k < K_; ++k) {
            if (k == focalK) continue;
            if (stage1Global.histMask[static_cast<size_t>(k)] / histMaskTotal > minFactorFrac) {
                selectedFactors.push_back(k);
            }
        }
    }
    const size_t nSelected = selectedFactors.size();
    if (nSelected < 2) {
        notice("%s: No factors passed the minFactorFrac threshold", __func__);
        return;
    }

    std::string outPairwise = outPrefix + ".pairwise.tsv";
    FILE* fpPairwise = fopen(outPairwise.c_str(), "w");
    if (!fpPairwise) {
        error("%s: Cannot open output file %s", __func__, outPairwise.c_str());
    }
    fprintf(fpPairwise, "k1\tk2\tarea1_pix\tarea2_pix\tarea_ovlp_pix\tarea_ovlp_f1\tarea_ovlp_f2\tarea_jaccard\tmass1_in_ovlp\tmass2_in_ovlp\tmass_ovlp_f1\tmass_ovlp_f2\n");

    if (histMaskTotal <= 0.0 || nSelected < 2) {
        fclose(fpPairwise);
        const auto tDone = std::chrono::steady_clock::now();
        notice("%s: Wrote histogram to %s", __func__, outHist.c_str());
        notice("%s: Wrote empty pairwise summary to %s", __func__, outPairwise.c_str());
        notice("%s: timing(ms): stage1=%lld total=%lld n_selected=%zu focal_area=%llu",
            __func__,
            static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage1 - t0).count()),
            static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tDone - t0).count()),
            nSelected,
            static_cast<unsigned long long>(stage1Global.focalArea));
        if (stage1Global.outOfRange > 0) {
            warning("%s: Ignored %llu out-of-tile records", __func__, static_cast<unsigned long long>(stage1Global.outOfRange));
        }
        if (stage1Global.badFactor > 0) {
            warning("%s: Ignored %llu invalid factor entries", __func__, static_cast<unsigned long long>(stage1Global.badFactor));
        }
        if (stage1Global.collisions > 0) {
            warning("%s: Encountered %llu duplicate pixel records", __func__, static_cast<unsigned long long>(stage1Global.collisions));
        }
        return;
    }


    const size_t nPairs = nSelected * (nSelected - 1) / 2;
    std::vector<int32_t> factorToSelected(static_cast<size_t>(K_), -1);
    for (size_t s = 0; s < nSelected; ++s) {
        factorToSelected[static_cast<size_t>(selectedFactors[s])] = static_cast<int32_t>(s);
    }

    struct Stage3Accum {
        std::vector<uint64_t> maskArea;
        std::vector<uint64_t> areaIntersection;
        std::vector<double> mass1Intersection;
        std::vector<double> mass2Intersection;
        uint64_t outOfRange = 0;
        uint64_t badFactor = 0;

        Stage3Accum(size_t nSel, size_t nPair)
            : maskArea(nSel, 0),
              areaIntersection(nPair, 0),
              mass1Intersection(nPair, 0.0),
              mass2Intersection(nPair, 0.0) {}
    };

    Stage3Accum stage3Global(nSelected, nPairs);

    auto processStage3Tile = [&](size_t bi, std::ifstream& in, Stage3Accum& accum) {
        const TileInfo& blk = blocks_[bi];
        SoftMaskTileData tileData;
        loadSoftMaskTileData(blk, in, minPixelProb, true, tileData,
            accum.outOfRange, accum.badFactor, nullptr, nullptr);
        const size_t nPix = tileData.geom.W * tileData.geom.H;
        if (nPix == 0) return;

        std::vector<std::vector<std::pair<uint32_t, float>>> entries(nSelected);
        for (size_t s = 0; s < nSelected; ++s) {
            auto it = tileData.factorEntries.find(selectedFactors[s]);
            if (it != tileData.factorEntries.end()) {
                entries[s] = it->second;
            }
        }

        std::vector<std::vector<uint8_t>> builtMasks(nSelected);
        std::vector<const std::vector<uint8_t>*> maskPtrs(nSelected, nullptr);

        std::vector<float> dense;
        for (size_t s = 0; s < nSelected; ++s) {
            buildDenseFactorRaster(entries[s], nPix, dense);
            buildSoftMaskBinary(dense, tileData.seenLocal, tileData.geom.W, tileData.geom.H,
                radius, neighborhoodThreshold, SoftMaskThresholdConfig{}, builtMasks[s]);
            const uint64_t keptArea = filterMaskByMinComponentArea4(
                builtMasks[s], tileData.geom.W, tileData.geom.H, minComponentArea);
            accum.maskArea[s] += keptArea;
            maskPtrs[s] = &builtMasks[s];
        }

        for (size_t idx = 0; idx < nPix; ++idx) {
            for (size_t i = 0; i < nSelected; ++i) {
                if (!(*maskPtrs[i])[idx]) continue;
                for (size_t j = i + 1; j < nSelected; ++j) {
                    if ((*maskPtrs[j])[idx]) {
                        accum.areaIntersection[pair_index_upper(i, j, nSelected)]++;
                    }
                }
            }
        }

        std::vector<double> selectedProb(nSelected, 0.0);
        for (const auto& stored : tileData.records) {
            const uint32_t localIdx = stored.localIdx;
            std::fill(selectedProb.begin(), selectedProb.end(), 0.0);
            const TopProbs& top = stored.rec;
            for (size_t i = 0; i < top.ks.size() && i < top.ps.size(); ++i) {
                const int32_t k = top.ks[i];
                if (k < 0 || k >= K_) continue;
                const double p = static_cast<double>(top.ps[i]);
                const int32_t selIdx = factorToSelected[static_cast<size_t>(k)];
                if (selIdx >= 0) {
                    selectedProb[static_cast<size_t>(selIdx)] += p;
                }
            }
            for (size_t i = 0; i < nSelected; ++i) {
                if (!(*maskPtrs[i])[localIdx]) continue;
                for (size_t j = i + 1; j < nSelected; ++j) {
                    if (!(*maskPtrs[j])[localIdx]) continue;
                    const size_t pidx = pair_index_upper(i, j, nSelected);
                    accum.mass1Intersection[pidx] += selectedProb[i];
                    accum.mass2Intersection[pidx] += selectedProb[j];
                }
            }
        }
    };

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<Stage3Accum> tls([&] { return Stage3Accum(nSelected, nPairs); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in = openTileStream();
                auto& local = tls.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processStage3Tile(bi, in, local);
                }
            });
        tls.combine_each([&](const Stage3Accum& local) {
            for (size_t i = 0; i < nSelected; ++i) {
                stage3Global.maskArea[i] += local.maskArea[i];
            }
            for (size_t i = 0; i < nPairs; ++i) {
                stage3Global.areaIntersection[i] += local.areaIntersection[i];
                stage3Global.mass1Intersection[i] += local.mass1Intersection[i];
                stage3Global.mass2Intersection[i] += local.mass2Intersection[i];
            }
            stage3Global.outOfRange += local.outOfRange;
            stage3Global.badFactor += local.badFactor;
        });
    } else {
        std::ifstream in = openTileStream();
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            processStage3Tile(bi, in, stage3Global);
        }
    }
    const auto tStage3 = std::chrono::steady_clock::now();

    auto safe_ratio = [](double num, double den) {
        return (den > 0.0) ? (num / den) : 0.0;
    };
    for (size_t i = 0; i < nSelected; ++i) {
        for (size_t j = i + 1; j < nSelected; ++j) {
            const size_t pidx = pair_index_upper(i, j, nSelected);
            const double area1 = static_cast<double>(stage3Global.maskArea[i]);
            const double area2 = static_cast<double>(stage3Global.maskArea[j]);
            const double areaI = static_cast<double>(stage3Global.areaIntersection[pidx]);
            const double areaJ = safe_ratio(areaI, area1 + area2 - areaI);
            const double mass1I = stage3Global.mass1Intersection[pidx];
            const double mass2I = stage3Global.mass2Intersection[pidx];
            const double totalMass1 = stage1Global.histGlobal[static_cast<size_t>(selectedFactors[i])];
            const double totalMass2 = stage1Global.histGlobal[static_cast<size_t>(selectedFactors[j])];
            fprintf(fpPairwise, "%d\t%d\t%llu\t%llu\t%llu\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n",
                selectedFactors[i], selectedFactors[j],
                static_cast<unsigned long long>(stage3Global.maskArea[i]),
                static_cast<unsigned long long>(stage3Global.maskArea[j]),
                static_cast<unsigned long long>(stage3Global.areaIntersection[pidx]),
                safe_ratio(areaI, area1), safe_ratio(areaI, area2), areaJ,
                mass1I, mass2I,
                safe_ratio(mass1I, totalMass1),
                safe_ratio(mass2I, totalMass2));
        }
    }
    fclose(fpPairwise);

    const auto tDone = std::chrono::steady_clock::now();
    notice("%s: Wrote histogram to %s", __func__, outHist.c_str());
    notice("%s: Wrote pairwise mask summary to %s", __func__, outPairwise.c_str());
    notice("%s: timing(ms): stage1=%lld stage3=%lld total=%lld n_selected=%zu focal_area=%llu",
        __func__,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage1 - t0).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage3 - tStage1).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tDone - t0).count()),
        nSelected,
        static_cast<unsigned long long>(stage3Global.maskArea[0]));
    if (stage1Global.outOfRange + stage3Global.outOfRange > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(stage1Global.outOfRange + stage3Global.outOfRange));
    }
    if (stage1Global.badFactor + stage3Global.badFactor > 0) {
        warning("%s: Ignored %llu invalid factor entries",
            __func__, static_cast<unsigned long long>(stage1Global.badFactor + stage3Global.badFactor));
    }
    if (stage1Global.collisions > 0) {
        warning("%s: Encountered %llu duplicate pixel records",
            __func__, static_cast<unsigned long long>(stage1Global.collisions));
    }
}

void TileOperator::softFactorMask(const std::string& outPrefix,
    int32_t radius, double neighborhoodThreshold,
    float minPixelProb, double minTileFactorMass,
    uint32_t minComponentArea, uint32_t minHoleArea,
    double simplifyTolerance, bool skipBoundaries,
    const std::string& templateGeoJSON, const std::string& templateOutPrefix) {
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (coord_dim_ != 2 || (mode_ & 0x8) != 0) {
        error("%s: Only 2D regular tiles are supported", __func__);
    }
    if (isTextInput() && !canSeekTextInput()) {
        error("%s: This operation requires seekable input", __func__);
    }
    if (K_ <= 0) {
        error("%s: K must be known and positive", __func__);
    }
    if (radius < 0) {
        error("%s: radius must be >= 0", __func__);
    }
    if (neighborhoodThreshold < 0.0 || neighborhoodThreshold > 1.0) {
        error("%s: neighborhoodThreshold must be in [0,1]", __func__);
    }
    if (minPixelProb < 0.0f) {
        error("%s: minPixelProb must be >= 0", __func__);
    }
    if (minTileFactorMass < 0.0) {
        error("%s: minTileFactorMass must be >= 0", __func__);
    }
    if (simplifyTolerance < 0.0) {
        error("%s: simplifyTolerance must be >= 0", __func__);
    }

    struct StageAccum {
        std::vector<uint64_t> tileCount;
        uint64_t outOfRange = 0;
        uint64_t badFactor = 0;
        uint64_t collisions = 0;

        explicit StageAccum(int32_t K)
            : tileCount(static_cast<size_t>(K), 0) {}
    };

    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<SoftMaskTileResult> perTile(blocks_.size());
    StageAccum globalAccum(K_);

    auto openTileStream = [&]() {
        std::ifstream in;
        if (mode_ & 0x1) in.open(dataFile_, std::ios::binary);
        else in.open(dataFile_);
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        return in;
    };

    auto processTile = [&](size_t bi, std::ifstream& in, StageAccum& accum) {
        const TileInfo& blk = blocks_[bi];
        SoftMaskTileResult tileOut;
        initTileGeom(blk, tileOut.geom);
        const size_t nPix = tileOut.geom.W * tileOut.geom.H;
        if (nPix == 0) {
            perTile[bi] = std::move(tileOut);
            return;
        }

        SoftMaskTileData tileData;
        loadSoftMaskTileData(blk, in, minPixelProb, false, tileData,
            accum.outOfRange, accum.badFactor, &accum.collisions, nullptr);

        std::vector<int32_t> factorsToProcess;
        factorsToProcess.reserve(tileData.factorMass.size());
        for (const auto& kv : tileData.factorMass) {
            if (kv.second >= minTileFactorMass) {
                factorsToProcess.push_back(kv.first);
                accum.tileCount[static_cast<size_t>(kv.first)] += 1;
            }
        }
        std::sort(factorsToProcess.begin(), factorsToProcess.end());
        if (factorsToProcess.empty()) {
            perTile[bi] = std::move(tileOut);
            return;
        }

        std::vector<float> dense;
        std::vector<uint8_t> mask;
        const SoftMaskThresholdConfig maskConfig{true, true, 0.4, 1.0};

        for (int32_t factor : factorsToProcess) {
            auto it = tileData.factorEntries.find(factor);
            if (it == tileData.factorEntries.end()) {
                continue;
            }
            buildDenseFactorRaster(it->second, nPix, dense);
            buildSoftMaskBinary(dense, tileData.seenLocal, tileOut.geom.W, tileOut.geom.H,
                radius, neighborhoodThreshold, maskConfig, mask);

            BinaryMaskCCL ccl = buildBinaryMaskCCL4(mask, dense, tileOut.geom.W, tileOut.geom.H,
                static_cast<double>(minComponentArea));
            if (ccl.ncomp == 0 || ccl.keptArea == 0) continue;

            SoftMaskTileFactorResult factorOut;
            factorOut.factor = factor;
            factorOut.maskArea = ccl.keptArea;
            std::vector<uint32_t> borderRemap(ccl.ncomp, INVALID);
            std::vector<uint32_t> borderCidOrder;
            if (!skipBoundaries) {
                factorOut.borderPolys.resize(ccl.ncomp);
            }
            for (uint32_t cid = 0; cid < ccl.ncomp; ++cid) {
                if (ccl.compTouchesBorder[cid]) {
                    borderRemap[cid] = static_cast<uint32_t>(factorOut.borderCompAreas.size());
                    borderCidOrder.push_back(cid);
                    factorOut.borderCompAreas.push_back(ccl.compArea[cid]);
                } else {
                    factorOut.interiorCompAreas.push_back(ccl.compArea[cid]);
                }
                if (!skipBoundaries) {
                    Clipper2Lib::Paths64 polys = buildMaskComponentPolygons(ccl, cid, tileOut.geom, minHoleArea, simplifyTolerance);
                    if (ccl.compTouchesBorder[cid]) factorOut.borderPolys[static_cast<size_t>(cid)] = std::move(polys);
                    else if (!polys.empty()) factorOut.interiorPolys.push_back(std::move(polys));
                }
            }
            if (factorOut.interiorCompAreas.empty() && factorOut.borderCompAreas.empty()) continue;

            factorOut.leftCid.assign(ccl.leftCid.size(), INVALID);
            factorOut.rightCid.assign(ccl.rightCid.size(), INVALID);
            factorOut.topCid.assign(ccl.topCid.size(), INVALID);
            factorOut.bottomCid.assign(ccl.bottomCid.size(), INVALID);
            for (size_t i = 0; i < ccl.leftCid.size(); ++i) {
                const uint32_t cid = ccl.leftCid[i];
                if (cid != INVALID && cid < borderRemap.size()) factorOut.leftCid[i] = borderRemap[cid];
            }
            for (size_t i = 0; i < ccl.rightCid.size(); ++i) {
                const uint32_t cid = ccl.rightCid[i];
                if (cid != INVALID && cid < borderRemap.size()) factorOut.rightCid[i] = borderRemap[cid];
            }
            for (size_t i = 0; i < ccl.topCid.size(); ++i) {
                const uint32_t cid = ccl.topCid[i];
                if (cid != INVALID && cid < borderRemap.size()) factorOut.topCid[i] = borderRemap[cid];
            }
            for (size_t i = 0; i < ccl.bottomCid.size(); ++i) {
                const uint32_t cid = ccl.bottomCid[i];
                if (cid != INVALID && cid < borderRemap.size()) factorOut.bottomCid[i] = borderRemap[cid];
            }

            if (!skipBoundaries && !factorOut.borderCompAreas.empty()) {
                std::vector<Clipper2Lib::Paths64> compactBorderPolys(factorOut.borderCompAreas.size());
                for (size_t i = 0; i < borderCidOrder.size(); ++i) {
                    compactBorderPolys[i] = std::move(factorOut.borderPolys[borderCidOrder[i]]);
                }
                factorOut.borderPolys.swap(compactBorderPolys);
            } else {
                factorOut.borderPolys.clear();
            }

            tileOut.factorToIndex[factor] = tileOut.factors.size();
            tileOut.factors.push_back(std::move(factorOut));
        }

        perTile[bi] = std::move(tileOut);
    };

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<StageAccum> tls([&] { return StageAccum(K_); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in = openTileStream();
                auto& local = tls.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processTile(bi, in, local);
                }
            });
        tls.combine_each([&](const StageAccum& local) {
            for (size_t k = 0; k < globalAccum.tileCount.size(); ++k) {
                globalAccum.tileCount[k] += local.tileCount[k];
            }
            globalAccum.outOfRange += local.outOfRange;
            globalAccum.badFactor += local.badFactor;
            globalAccum.collisions += local.collisions;
        });
    } else {
        std::ifstream in = openTileStream();
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            processTile(bi, in, globalAccum);
        }
    }
    const auto tStage1 = std::chrono::steady_clock::now();

    std::vector<Clipper2Lib::Paths64> finalPathsByFactor(static_cast<size_t>(K_));
    std::vector<uint64_t> finalArea(static_cast<size_t>(K_), 0);
    std::vector<uint64_t> finalComponents(static_cast<size_t>(K_), 0);
    std::vector<std::unordered_map<uint64_t, uint64_t>> componentHist(static_cast<size_t>(K_));
    std::vector<std::vector<size_t>> borderBase(perTile.size());
    size_t totalBorderComps = 0;

    for (size_t ti = 0; ti < perTile.size(); ++ti) {
        const SoftMaskTileResult& tile = perTile[ti];
        borderBase[ti].assign(tile.factors.size(), std::numeric_limits<size_t>::max());
        for (size_t fi = 0; fi < tile.factors.size(); ++fi) {
            const SoftMaskTileFactorResult& factorRes = tile.factors[fi];
            finalArea[static_cast<size_t>(factorRes.factor)] += factorRes.maskArea;
            for (uint32_t area : factorRes.interiorCompAreas) {
                finalComponents[static_cast<size_t>(factorRes.factor)] += 1;
                componentHist[static_cast<size_t>(factorRes.factor)][static_cast<uint64_t>(area)] += 1;
            }
            if (!skipBoundaries) {
                for (const auto& polys : factorRes.interiorPolys) {
                    finalPathsByFactor[static_cast<size_t>(factorRes.factor)].insert(
                        finalPathsByFactor[static_cast<size_t>(factorRes.factor)].end(),
                        polys.begin(), polys.end());
                }
            }
            if (!factorRes.borderCompAreas.empty()) {
                borderBase[ti][fi] = totalBorderComps;
                totalBorderComps += factorRes.borderCompAreas.size();
            }
        }
    }

    DisjointSet dsu(totalBorderComps);
    for (size_t ti = 0; ti < perTile.size(); ++ti) {
        const SoftMaskTileResult& aTile = perTile[ti];
        const TileKey key{blocks_[ti].idx.row, blocks_[ti].idx.col};
        const auto rightIt = tile_lookup_.find(TileKey{key.row, key.col + 1});
        if (rightIt != tile_lookup_.end()) {
            const size_t tj = rightIt->second;
            const SoftMaskTileResult& bTile = perTile[tj];
            for (size_t afi = 0; afi < aTile.factors.size(); ++afi) {
                const SoftMaskTileFactorResult& a = aTile.factors[afi];
                if (a.borderCompAreas.empty()) continue;
                const auto bIt = bTile.factorToIndex.find(a.factor);
                if (bIt == bTile.factorToIndex.end()) continue;
                const size_t bfi = bIt->second;
                const SoftMaskTileFactorResult& b = bTile.factors[bfi];
                if (b.borderCompAreas.empty()) continue;
                const int32_t y0 = std::max(aTile.geom.pixY0, bTile.geom.pixY0);
                const int32_t y1 = std::min(aTile.geom.pixY1, bTile.geom.pixY1);
                uint32_t lastA = INVALID, lastB = INVALID;
                for (int32_t gy = y0; gy < y1; ++gy) {
                    const uint32_t ca = a.rightCid[static_cast<size_t>(gy - aTile.geom.pixY0)];
                    const uint32_t cb = b.leftCid[static_cast<size_t>(gy - bTile.geom.pixY0)];
                    if (ca == INVALID || cb == INVALID || (ca == lastA && cb == lastB)) continue;
                    lastA = ca;
                    lastB = cb;
                    dsu.unite(borderBase[ti][afi] + static_cast<size_t>(ca),
                        borderBase[tj][bfi] + static_cast<size_t>(cb));
                }
            }
        }

        const auto downIt = tile_lookup_.find(TileKey{key.row + 1, key.col});
        if (downIt != tile_lookup_.end()) {
            const size_t tj = downIt->second;
            const SoftMaskTileResult& bTile = perTile[tj];
            for (size_t afi = 0; afi < aTile.factors.size(); ++afi) {
                const SoftMaskTileFactorResult& a = aTile.factors[afi];
                if (a.borderCompAreas.empty()) continue;
                const auto bIt = bTile.factorToIndex.find(a.factor);
                if (bIt == bTile.factorToIndex.end()) continue;
                const size_t bfi = bIt->second;
                const SoftMaskTileFactorResult& b = bTile.factors[bfi];
                if (b.borderCompAreas.empty()) continue;
                const int32_t x0 = std::max(aTile.geom.pixX0, bTile.geom.pixX0);
                const int32_t x1 = std::min(aTile.geom.pixX1, bTile.geom.pixX1);
                uint32_t lastA = INVALID, lastB = INVALID;
                for (int32_t gx = x0; gx < x1; ++gx) {
                    const uint32_t ca = a.bottomCid[static_cast<size_t>(gx - aTile.geom.pixX0)];
                    const uint32_t cb = b.topCid[static_cast<size_t>(gx - bTile.geom.pixX0)];
                    if (ca == INVALID || cb == INVALID || (ca == lastA && cb == lastB)) continue;
                    lastA = ca;
                    lastB = cb;
                    dsu.unite(borderBase[ti][afi] + static_cast<size_t>(ca),
                        borderBase[tj][bfi] + static_cast<size_t>(cb));
                }
            }
        }
    }
    if (totalBorderComps > 0) dsu.compress_all();

    std::vector<uint64_t> rootArea(totalBorderComps, 0);
    std::vector<Clipper2Lib::Paths64> rootPaths(totalBorderComps);
    std::vector<int32_t> rootFactor(totalBorderComps, -1);
    for (size_t ti = 0; ti < perTile.size(); ++ti) {
        const SoftMaskTileResult& tile = perTile[ti];
        for (size_t fi = 0; fi < tile.factors.size(); ++fi) {
            const SoftMaskTileFactorResult& factorRes = tile.factors[fi];
            if (factorRes.borderCompAreas.empty()) continue;
            const size_t base = borderBase[ti][fi];
            for (size_t cid = 0; cid < factorRes.borderCompAreas.size(); ++cid) {
                const size_t gid = base + cid;
                const size_t root = dsu.parent[gid];
                rootArea[root] += static_cast<uint64_t>(factorRes.borderCompAreas[cid]);
                if (rootFactor[root] < 0) rootFactor[root] = factorRes.factor;
                else if (rootFactor[root] != factorRes.factor) {
                    error("%s: Border component factor mismatch after DSU merge", __func__);
                }
                if (!skipBoundaries && cid < factorRes.borderPolys.size()) {
                    rootPaths[root].insert(rootPaths[root].end(),
                        factorRes.borderPolys[cid].begin(), factorRes.borderPolys[cid].end());
                }
            }
        }
    }
    for (size_t root = 0; root < rootFactor.size(); ++root) {
        if (rootFactor[root] < 0 || rootArea[root] == 0) continue;
        finalComponents[static_cast<size_t>(rootFactor[root])] += 1;
        componentHist[static_cast<size_t>(rootFactor[root])][rootArea[root]] += 1;
        if (skipBoundaries || rootPaths[root].empty()) continue;
        Clipper2Lib::Paths64 merged = cleanupMaskPolygons(rootPaths[root], minHoleArea, simplifyTolerance);
        if (merged.empty()) continue;
        finalPathsByFactor[static_cast<size_t>(rootFactor[root])].insert(
            finalPathsByFactor[static_cast<size_t>(rootFactor[root])].end(),
            merged.begin(), merged.end());
    }
    const auto tStage2 = std::chrono::steady_clock::now();

    std::string outHist = outPrefix + ".factor_summary.tsv";
    write_factor_mask_info(outHist, globalAccum.tileCount, finalArea, finalComponents);
    std::string outCompHist = outPrefix + ".component_hist.tsv";
    write_factor_mask_component_histogram(outCompHist, componentHist);

    std::string outGeoJSON = outPrefix + ".geojson";
    if (!skipBoundaries) {
        nlohmann::json templateJson;
        const bool writeTemplateJSON = !templateGeoJSON.empty();
        const std::string factorOutPrefix = templateOutPrefix.empty() ? outPrefix : templateOutPrefix;
        if (writeTemplateJSON) {
            templateJson = load_template_json(templateGeoJSON);
        }
        nlohmann::json features = nlohmann::json::array();
        for (int32_t k = 0; k < K_; ++k) {
            if (finalPathsByFactor[static_cast<size_t>(k)].empty()) continue;
            Clipper2Lib::Paths64 factorPaths = cleanupMaskPolygons(
                finalPathsByFactor[static_cast<size_t>(k)], minHoleArea, simplifyTolerance);
            if (factorPaths.empty()) continue;
            nlohmann::json feature;
            feature["type"] = "Feature";
            feature["properties"] = {
                {"Factor", k},
                {"n_tiles", globalAccum.tileCount[static_cast<size_t>(k)]},
                {"mask_area_pix", finalArea[static_cast<size_t>(k)]},
                {"n_components", finalComponents[static_cast<size_t>(k)]}
            };
            feature["geometry"] = maskPathsToMultiPolygonGeoJSON(factorPaths);
            if (writeTemplateJSON) {
                const std::string outFactor = factorOutPrefix + ".k" + std::to_string(k) + ".json";
                write_factor_template_json(outFactor, templateJson, k, feature);
            }
            features.push_back(std::move(feature));
        }
        nlohmann::json featureCollection;
        featureCollection["type"] = "FeatureCollection";
        featureCollection["features"] = std::move(features);

        std::ofstream geojsonOut(outGeoJSON);
        if (!geojsonOut.is_open()) {
            error("%s: Cannot open output file %s", __func__, outGeoJSON.c_str());
        }
        geojsonOut << featureCollection.dump();
        geojsonOut << "\n";
        geojsonOut.close();
    }

    const auto tDone = std::chrono::steady_clock::now();
    notice("%s: Wrote factor histogram to %s", __func__, outHist.c_str());
    notice("%s: Wrote component size histogram to %s", __func__, outCompHist.c_str());
    if (!skipBoundaries) {
        notice("%s: Wrote soft-mask GeoJSON to %s", __func__, outGeoJSON.c_str());
    }
    notice("%s: timing(ms): stage1=%lld stage2=%lld total=%lld",
        __func__,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage1 - t0).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage2 - tStage1).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tDone - t0).count()));
    if (globalAccum.outOfRange > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(globalAccum.outOfRange));
    }
    if (globalAccum.badFactor > 0) {
        warning("%s: Ignored %llu invalid factor entries",
            __func__, static_cast<unsigned long long>(globalAccum.badFactor));
    }
    if (globalAccum.collisions > 0) {
        warning("%s: Encountered %llu duplicate pixel records",
            __func__, static_cast<unsigned long long>(globalAccum.collisions));
    }
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
        const size_t width = dense.geom.W;
        const size_t height = dense.geom.H;
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

void TileOperator::hardFactorMask(const std::string& outPrefix, uint32_t minSize, bool skipBoundaries, const std::string& templateGeoJSON, const std::string& templateOutPrefix) {
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
    const auto t0 = std::chrono::steady_clock::now();

    struct StageAccum {
        std::vector<uint64_t> tileCount;
        uint64_t outOfRange = 0;
        uint64_t badLabel = 0;

        explicit StageAccum(int32_t K_) : tileCount(static_cast<size_t>(K_), 0) {}
    };

    std::vector<TileCCL> perTile(blocks_.size());
    StageAccum globalAccum(K_);

    auto openTileStream = [&]() {
        std::ifstream in;
        if (mode_ & 0x1) in.open(dataFile_, std::ios::binary);
        else in.open(dataFile_);
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        return in;
    };

    auto processTile = [&](size_t bi, std::ifstream& in, StageAccum& accum) {
        DenseTile dense;
        loadDenseTile(blocks_[bi], in, dense, BG, accum.outOfRange, accum.badLabel);
        perTile[bi] = tileLocalCCL(dense, BG);
        std::vector<uint8_t> seen(static_cast<size_t>(K), 0);
        for (uint8_t lbl : dense.lab) {
            if (lbl < BG) seen[static_cast<size_t>(lbl)] = 1;
        }
        for (size_t k = 0; k < seen.size(); ++k) {
            accum.tileCount[k] += static_cast<uint64_t>(seen[k]);
        }
    };

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<StageAccum> tls([&] { return StageAccum(K_); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in = openTileStream();
                auto& local = tls.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processTile(bi, in, local);
                }
            });
        tls.combine_each([&](const StageAccum& local) {
            for (size_t k = 0; k < globalAccum.tileCount.size(); ++k) {
                globalAccum.tileCount[k] += local.tileCount[k];
            }
            globalAccum.outOfRange += local.outOfRange;
            globalAccum.badLabel += local.badLabel;
        });
    } else {
        std::ifstream in = openTileStream();
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            processTile(bi, in, globalAccum);
        }
    }
    const auto tStage1 = std::chrono::steady_clock::now();

    std::vector<Clipper2Lib::Paths64> finalPathsByFactor(static_cast<size_t>(K), Clipper2Lib::Paths64{});
    std::vector<uint64_t> finalArea(static_cast<size_t>(K), 0);
    std::vector<uint64_t> finalComponents(static_cast<size_t>(K), 0);
    std::vector<std::unordered_map<uint64_t, uint64_t>> componentHist(static_cast<size_t>(K));
    std::vector<SoftMaskTileResult> borderTiles(blocks_.size());

    for (size_t ti = 0; ti < perTile.size(); ++ti) {
        TileCCL& t = perTile[ti];
        if (t.ncomp == 0) continue;
        TileGeom geom;
        initTileGeom(blocks_[ti], geom);

        std::vector<Clipper2Lib::Paths64> oldPolys;
        if (!skipBoundaries) {
            oldPolys.resize(t.ncomp);
            for (uint32_t cid = 0; cid < t.ncomp; ++cid) {
                oldPolys[cid] = buildLabelComponentPolygons(
                    t.pixelCid, geom.W, cid, t.compBox[cid], geom, 0, 0.0);
            }
        }

        const BorderRemapInfo remapInfo = remapTileToBorderComponents(t, INVALID);
        SoftMaskTileResult tileOut;
        tileOut.geom = geom;
        std::vector<uint32_t> newCidToOld(t.ncomp, INVALID);
        for (uint32_t oldCid = 0; oldCid < remapInfo.remap.size(); ++oldCid) {
            const uint8_t lbl = remapInfo.oldCompLabel[oldCid];
            const uint64_t size = static_cast<uint64_t>(remapInfo.oldCompSize[oldCid]);
            if (lbl >= BG || size == 0) continue;
            const uint32_t newCid = remapInfo.remap[oldCid];
            if (newCid == INVALID) {
                if (size < static_cast<uint64_t>(minSize)) continue;
                finalArea[static_cast<size_t>(lbl)] += size;
                finalComponents[static_cast<size_t>(lbl)] += 1;
                componentHist[static_cast<size_t>(lbl)][size] += 1;
                if (!skipBoundaries && oldCid < oldPolys.size() && !oldPolys[oldCid].empty()) {
                    finalPathsByFactor[static_cast<size_t>(lbl)].insert(
                        finalPathsByFactor[static_cast<size_t>(lbl)].end(),
                        oldPolys[oldCid].begin(), oldPolys[oldCid].end());
                }
            } else if (newCid < newCidToOld.size()) {
                newCidToOld[newCid] = oldCid;
            }
        }

        std::vector<uint32_t> newCidToFactorLocal(t.ncomp, INVALID);
        std::vector<int32_t> newCidToFactorIndex(t.ncomp, -1);
        for (uint32_t cid = 0; cid < t.ncomp; ++cid) {
            const uint8_t lbl = t.compLabel[cid];
            if (lbl >= BG) continue;
            auto it = tileOut.factorToIndex.find(static_cast<int32_t>(lbl));
            size_t fi = 0;
            if (it == tileOut.factorToIndex.end()) {
                fi = tileOut.factors.size();
                tileOut.factorToIndex[static_cast<int32_t>(lbl)] = fi;
                tileOut.factors.push_back(SoftMaskTileFactorResult{});
                tileOut.factors.back().factor = static_cast<int32_t>(lbl);
                tileOut.factors.back().leftCid.assign(geom.H, INVALID);
                tileOut.factors.back().rightCid.assign(geom.H, INVALID);
                tileOut.factors.back().topCid.assign(geom.W, INVALID);
                tileOut.factors.back().bottomCid.assign(geom.W, INVALID);
            } else {
                fi = it->second;
            }
            SoftMaskTileFactorResult& factorRes = tileOut.factors[fi];
            newCidToFactorIndex[cid] = static_cast<int32_t>(fi);
            newCidToFactorLocal[cid] = static_cast<uint32_t>(factorRes.borderCompAreas.size());
            factorRes.borderCompAreas.push_back(t.compSize[cid]);
            factorRes.maskArea += static_cast<uint64_t>(t.compSize[cid]);
            if (!skipBoundaries && cid < newCidToOld.size()) {
                const uint32_t oldCid = newCidToOld[cid];
                if (oldCid != INVALID && oldCid < oldPolys.size()) {
                    factorRes.borderPolys.push_back(std::move(oldPolys[oldCid]));
                } else {
                    factorRes.borderPolys.emplace_back();
                }
            }
        }

        for (size_t y = 0; y < t.leftCid.size(); ++y) {
            const uint32_t cid = t.leftCid[y];
            if (cid != INVALID && cid < newCidToFactorIndex.size() && newCidToFactorIndex[cid] >= 0) {
                tileOut.factors[static_cast<size_t>(newCidToFactorIndex[cid])].leftCid[y] = newCidToFactorLocal[cid];
            }
        }
        for (size_t y = 0; y < t.rightCid.size(); ++y) {
            const uint32_t cid = t.rightCid[y];
            if (cid != INVALID && cid < newCidToFactorIndex.size() && newCidToFactorIndex[cid] >= 0) {
                tileOut.factors[static_cast<size_t>(newCidToFactorIndex[cid])].rightCid[y] = newCidToFactorLocal[cid];
            }
        }
        for (size_t x = 0; x < t.topCid.size(); ++x) {
            const uint32_t cid = t.topCid[x];
            if (cid != INVALID && cid < newCidToFactorIndex.size() && newCidToFactorIndex[cid] >= 0) {
                tileOut.factors[static_cast<size_t>(newCidToFactorIndex[cid])].topCid[x] = newCidToFactorLocal[cid];
            }
        }
        for (size_t x = 0; x < t.bottomCid.size(); ++x) {
            const uint32_t cid = t.bottomCid[x];
            if (cid != INVALID && cid < newCidToFactorIndex.size() && newCidToFactorIndex[cid] >= 0) {
                tileOut.factors[static_cast<size_t>(newCidToFactorIndex[cid])].bottomCid[x] = newCidToFactorLocal[cid];
            }
        }

        borderTiles[ti] = std::move(tileOut);
    }

    std::vector<std::vector<size_t>> borderBase(borderTiles.size());
    size_t totalBorderComps = 0;
    for (size_t ti = 0; ti < borderTiles.size(); ++ti) {
        const SoftMaskTileResult& tile = borderTiles[ti];
        borderBase[ti].assign(tile.factors.size(), std::numeric_limits<size_t>::max());
        for (size_t fi = 0; fi < tile.factors.size(); ++fi) {
            const auto& factorRes = tile.factors[fi];
            if (factorRes.borderCompAreas.empty()) continue;
            borderBase[ti][fi] = totalBorderComps;
            totalBorderComps += factorRes.borderCompAreas.size();
        }
    }

    DisjointSet dsu(totalBorderComps);
    for (size_t ti = 0; ti < borderTiles.size(); ++ti) {
        const SoftMaskTileResult& aTile = borderTiles[ti];
        const TileKey key{blocks_[ti].idx.row, blocks_[ti].idx.col};
        const auto rightIt = tile_lookup_.find(TileKey{key.row, key.col + 1});
        if (rightIt != tile_lookup_.end()) {
            const size_t tj = rightIt->second;
            const SoftMaskTileResult& bTile = borderTiles[tj];
            for (size_t afi = 0; afi < aTile.factors.size(); ++afi) {
                const SoftMaskTileFactorResult& a = aTile.factors[afi];
                if (a.borderCompAreas.empty()) continue;
                const auto bIt = bTile.factorToIndex.find(a.factor);
                if (bIt == bTile.factorToIndex.end()) continue;
                const SoftMaskTileFactorResult& b = bTile.factors[bIt->second];
                if (b.borderCompAreas.empty()) continue;
                const int32_t y0 = std::max(aTile.geom.pixY0, bTile.geom.pixY0);
                const int32_t y1 = std::min(aTile.geom.pixY1, bTile.geom.pixY1);
                uint32_t lastA = INVALID, lastB = INVALID;
                for (int32_t gy = y0; gy < y1; ++gy) {
                    const uint32_t ca = a.rightCid[static_cast<size_t>(gy - aTile.geom.pixY0)];
                    const uint32_t cb = b.leftCid[static_cast<size_t>(gy - bTile.geom.pixY0)];
                    if (ca == INVALID || cb == INVALID || (ca == lastA && cb == lastB)) continue;
                    lastA = ca;
                    lastB = cb;
                    dsu.unite(borderBase[ti][afi] + static_cast<size_t>(ca),
                        borderBase[tj][bIt->second] + static_cast<size_t>(cb));
                }
            }
        }
        const auto downIt = tile_lookup_.find(TileKey{key.row + 1, key.col});
        if (downIt != tile_lookup_.end()) {
            const size_t tj = downIt->second;
            const SoftMaskTileResult& bTile = borderTiles[tj];
            for (size_t afi = 0; afi < aTile.factors.size(); ++afi) {
                const SoftMaskTileFactorResult& a = aTile.factors[afi];
                if (a.borderCompAreas.empty()) continue;
                const auto bIt = bTile.factorToIndex.find(a.factor);
                if (bIt == bTile.factorToIndex.end()) continue;
                const SoftMaskTileFactorResult& b = bTile.factors[bIt->second];
                if (b.borderCompAreas.empty()) continue;
                const int32_t x0 = std::max(aTile.geom.pixX0, bTile.geom.pixX0);
                const int32_t x1 = std::min(aTile.geom.pixX1, bTile.geom.pixX1);
                uint32_t lastA = INVALID, lastB = INVALID;
                for (int32_t gx = x0; gx < x1; ++gx) {
                    const uint32_t ca = a.bottomCid[static_cast<size_t>(gx - aTile.geom.pixX0)];
                    const uint32_t cb = b.topCid[static_cast<size_t>(gx - bTile.geom.pixX0)];
                    if (ca == INVALID || cb == INVALID || (ca == lastA && cb == lastB)) continue;
                    lastA = ca;
                    lastB = cb;
                    dsu.unite(borderBase[ti][afi] + static_cast<size_t>(ca),
                        borderBase[tj][bIt->second] + static_cast<size_t>(cb));
                }
            }
        }
    }
    if (totalBorderComps > 0) dsu.compress_all();

    std::vector<uint64_t> rootArea(totalBorderComps, 0);
    std::vector<Clipper2Lib::Paths64> rootPaths(totalBorderComps);
    std::vector<int32_t> rootFactor(totalBorderComps, -1);
    for (size_t ti = 0; ti < borderTiles.size(); ++ti) {
        const SoftMaskTileResult& tile = borderTiles[ti];
        for (size_t fi = 0; fi < tile.factors.size(); ++fi) {
            const auto& factorRes = tile.factors[fi];
            if (factorRes.borderCompAreas.empty()) continue;
            const size_t base = borderBase[ti][fi];
            for (size_t cid = 0; cid < factorRes.borderCompAreas.size(); ++cid) {
                const size_t gid = base + cid;
                const size_t root = dsu.parent[gid];
                rootArea[root] += static_cast<uint64_t>(factorRes.borderCompAreas[cid]);
                if (rootFactor[root] < 0) rootFactor[root] = factorRes.factor;
                else if (rootFactor[root] != factorRes.factor) {
                    error("%s: Label mismatch after seam union", __func__);
                }
                if (!skipBoundaries && cid < factorRes.borderPolys.size()) {
                    rootPaths[root].insert(rootPaths[root].end(),
                        factorRes.borderPolys[cid].begin(), factorRes.borderPolys[cid].end());
                }
            }
        }
    }

    for (size_t root = 0; root < rootFactor.size(); ++root) {
        if (rootFactor[root] < 0 || rootArea[root] < static_cast<uint64_t>(minSize)) continue;
        finalArea[static_cast<size_t>(rootFactor[root])] += rootArea[root];
        finalComponents[static_cast<size_t>(rootFactor[root])] += 1;
        componentHist[static_cast<size_t>(rootFactor[root])][rootArea[root]] += 1;
        if (skipBoundaries || rootPaths[root].empty()) continue;
        Clipper2Lib::Paths64 merged = cleanupMaskPolygons(rootPaths[root], 0, 0.0);
        if (merged.empty()) continue;
        finalPathsByFactor[static_cast<size_t>(rootFactor[root])].insert(
            finalPathsByFactor[static_cast<size_t>(rootFactor[root])].end(),
            merged.begin(), merged.end());
    }
    const auto tStage2 = std::chrono::steady_clock::now();

    std::string outHist = outPrefix + ".factor_summary.tsv";
    write_factor_mask_info(outHist, globalAccum.tileCount, finalArea, finalComponents);
    std::string outCompHist = outPrefix + ".component_hist.tsv";
    write_factor_mask_component_histogram(outCompHist, componentHist);

    std::string outGeoJSON = outPrefix + ".geojson";
    if (!skipBoundaries) {
        nlohmann::json templateJson;
        const bool writeTemplateJSON = !templateGeoJSON.empty();
        const std::string factorOutPrefix = templateOutPrefix.empty() ? outPrefix : templateOutPrefix;
        if (writeTemplateJSON) {
            templateJson = load_template_json(templateGeoJSON);
        }
        nlohmann::json features = nlohmann::json::array();
        for (int32_t k = 0; k < K; ++k) {
            if (finalPathsByFactor[static_cast<size_t>(k)].empty()) continue;
            Clipper2Lib::Paths64 factorPaths = cleanupMaskPolygons(finalPathsByFactor[static_cast<size_t>(k)], 0, 0.0);
            if (factorPaths.empty()) continue;
            nlohmann::json feature;
            feature["type"] = "Feature";
            feature["properties"] = {
                {"Factor", k},
                {"n_tiles", globalAccum.tileCount[static_cast<size_t>(k)]},
                {"mask_area_pix", finalArea[static_cast<size_t>(k)]},
                {"n_components", finalComponents[static_cast<size_t>(k)]}
            };
            feature["geometry"] = maskPathsToMultiPolygonGeoJSON(factorPaths);
            if (writeTemplateJSON) {
                const std::string outFactor = factorOutPrefix + ".k" + std::to_string(k) + ".json";
                write_factor_template_json(outFactor, templateJson, k, feature);
            }
            features.push_back(std::move(feature));
        }
        nlohmann::json featureCollection;
        featureCollection["type"] = "FeatureCollection";
        featureCollection["features"] = std::move(features);
        std::ofstream geojsonOut(outGeoJSON);
        if (!geojsonOut.is_open()) {
            error("%s: Cannot open output file %s", __func__, outGeoJSON.c_str());
        }
        geojsonOut << featureCollection.dump();
        geojsonOut << "\n";
        geojsonOut.close();
    }

    const auto tDone = std::chrono::steady_clock::now();
    notice("%s: Wrote factor histogram to %s", __func__, outHist.c_str());
    notice("%s: Wrote component size histogram to %s", __func__, outCompHist.c_str());
    if (!skipBoundaries) {
        notice("%s: Wrote hard-mask GeoJSON to %s", __func__, outGeoJSON.c_str());
    }
    notice("%s: timing(ms): stage1=%lld stage2=%lld total=%lld",
        __func__,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage1 - t0).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage2 - tStage1).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tDone - t0).count()));
    if (globalAccum.outOfRange > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(globalAccum.outOfRange));
    }
    if (globalAccum.badLabel > 0) {
        warning("%s: Ignored %llu records with invalid labels",
            __func__, static_cast<unsigned long long>(globalAccum.badLabel));
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
        const TileKey key = tiles[i].geom.key;
        auto it = tile_lookup_.find(TileKey{key.row, key.col - 1});
        if (it != tile_lookup_.end()) nbr[i].left = static_cast<int32_t>(it->second);
        it = tile_lookup_.find(TileKey{key.row, key.col + 1});
        if (it != tile_lookup_.end()) nbr[i].right = static_cast<int32_t>(it->second);
        it = tile_lookup_.find(TileKey{key.row - 1, key.col});
        if (it != tile_lookup_.end()) nbr[i].up = static_cast<int32_t>(it->second);
        it = tile_lookup_.find(TileKey{key.row + 1, key.col});
        if (it != tile_lookup_.end()) nbr[i].down = static_cast<int32_t>(it->second);

        auto& sj = seam[i];
        sj.left.assign(tiles[i].geom.H, INVALID);
        sj.right.assign(tiles[i].geom.H, INVALID);
        sj.up.assign(tiles[i].geom.W, INVALID);
        sj.down.assign(tiles[i].geom.W, INVALID);

        if (nbr[i].left >= 0) {
            const DenseTile& nt = tiles[static_cast<size_t>(nbr[i].left)];
            for (size_t y = 0; y < tiles[i].geom.H; ++y) {
                const int32_t ngx = tiles[i].geom.pixX0 - 1;
                const int32_t ngy = tiles[i].geom.pixY0 + static_cast<int32_t>(y);
                if (ngx < nt.geom.pixX0 || ngx >= nt.geom.pixX1 || ngy < nt.geom.pixY0 || ngy >= nt.geom.pixY1) continue;
                const size_t nx = static_cast<size_t>(ngx - nt.geom.pixX0);
                const size_t ny = static_cast<size_t>(ngy - nt.geom.pixY0);
                sj.left[y] = static_cast<uint32_t>(ny * nt.geom.W + nx);
            }
        }
        if (nbr[i].right >= 0) {
            const DenseTile& nt = tiles[static_cast<size_t>(nbr[i].right)];
            for (size_t y = 0; y < tiles[i].geom.H; ++y) {
                const int32_t ngx = tiles[i].geom.pixX1;
                const int32_t ngy = tiles[i].geom.pixY0 + static_cast<int32_t>(y);
                if (ngx < nt.geom.pixX0 || ngx >= nt.geom.pixX1 || ngy < nt.geom.pixY0 || ngy >= nt.geom.pixY1) continue;
                const size_t nx = static_cast<size_t>(ngx - nt.geom.pixX0);
                const size_t ny = static_cast<size_t>(ngy - nt.geom.pixY0);
                sj.right[y] = static_cast<uint32_t>(ny * nt.geom.W + nx);
            }
        }
        if (nbr[i].up >= 0) {
            const DenseTile& nt = tiles[static_cast<size_t>(nbr[i].up)];
            for (size_t x = 0; x < tiles[i].geom.W; ++x) {
                const int32_t ngx = tiles[i].geom.pixX0 + static_cast<int32_t>(x);
                const int32_t ngy = tiles[i].geom.pixY0 - 1;
                if (ngx < nt.geom.pixX0 || ngx >= nt.geom.pixX1 || ngy < nt.geom.pixY0 || ngy >= nt.geom.pixY1) continue;
                const size_t nx = static_cast<size_t>(ngx - nt.geom.pixX0);
                const size_t ny = static_cast<size_t>(ngy - nt.geom.pixY0);
                sj.up[x] = static_cast<uint32_t>(ny * nt.geom.W + nx);
            }
        }
        if (nbr[i].down >= 0) {
            const DenseTile& nt = tiles[static_cast<size_t>(nbr[i].down)];
            for (size_t x = 0; x < tiles[i].geom.W; ++x) {
                const int32_t ngx = tiles[i].geom.pixX0 + static_cast<int32_t>(x);
                const int32_t ngy = tiles[i].geom.pixY1;
                if (ngx < nt.geom.pixX0 || ngx >= nt.geom.pixX1 || ngy < nt.geom.pixY0 || ngy >= nt.geom.pixY1) continue;
                const size_t nx = static_cast<size_t>(ngx - nt.geom.pixX0);
                const size_t ny = static_cast<size_t>(ngy - nt.geom.pixY0);
                sj.down[x] = static_cast<uint32_t>(ny * nt.geom.W + nx);
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

                const size_t x = li % tile.geom.W;
                const size_t y = li / tile.geom.W;
                const uint16_t nd = static_cast<uint16_t>(d + 1);
                if (x > 0) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li - 1), nd);
                } else if (nbr[ref.tileIdx].left >= 0) {
                    const uint32_t nli = seam[ref.tileIdx].left[y];
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].left), nli, nd);
                    }
                }
                if (x + 1 < tile.geom.W) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li + 1), nd);
                } else if (nbr[ref.tileIdx].right >= 0) {
                    const uint32_t nli = seam[ref.tileIdx].right[y];
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].right), nli, nd);
                    }
                }
                if (y > 0) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li - tile.geom.W), nd);
                } else if (nbr[ref.tileIdx].up >= 0) {
                    const uint32_t nli = seam[ref.tileIdx].up[x];
                    if (nli != INVALID) {
                        tryPush(static_cast<uint32_t>(nbr[ref.tileIdx].up), nli, nd);
                    }
                }
                if (y + 1 < tile.geom.H) {
                    tryPush(ref.tileIdx, static_cast<uint32_t>(li + tile.geom.W), nd);
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

std::unordered_map<int32_t, TileOperator::Slice> TileOperator::aggOneTileRegion(
    const std::map<std::pair<int32_t, int32_t>, TopProbs>& pixelMap,
    const std::unordered_map<std::pair<int32_t, int32_t>, RegionPixelState, PairHash>& pixelState,
    TileReader& reader, lineParserUnival& parser, TileKey tile,
    const PreparedRegionMask2D& region, double gridSize, double minProb,
    int32_t union_key, Eigen::MatrixXd* confusion, double* residualAccum) const {
    if (coord_dim_ == 3) {
        error("%s does not support 3D data yet", __func__);
    }
    assert(reader.getTileSize() == formatInfo_.tileSize);

    std::unordered_map<int32_t, Slice> tileAgg;
    if (pixelMap.empty()) {
        return tileAgg;
    }
    auto it = reader.get_tile_iterator(tile.row, tile.col);
    if (!it) {
        return tileAgg;
    }
    auto aggIt0 = tileAgg.emplace(union_key, Slice()).first;
    auto& oneSlice0 = aggIt0->second;

    const float res = formatInfo_.pixelResolution > 0.0f ? formatInfo_.pixelResolution : 1.0f;
    auto add_confusion = [&](const TopProbs& anno) {
        if (confusion == nullptr || residualAccum == nullptr) {
            return;
        }
        double residual = 1.0;
        for (size_t ii = 0; ii < anno.ks.size(); ++ii) {
            residual -= anno.ps[ii];
            for (size_t jj = ii; jj < anno.ks.size(); ++jj) {
                (*confusion)(anno.ks[ii], anno.ks[jj]) += anno.ps[ii] * anno.ps[jj];
            }
        }
        if (residual > 0.0) {
            *residualAccum += residual * residual;
        }
    };

    std::string line;
    RecordT<float> rec;
    while (it->next(line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        const int32_t idx = parser.parse(rec, line);
        if (idx < 0) {
            continue;
        }
        const int32_t ix = static_cast<int32_t>(std::floor(rec.x / res));
        const int32_t iy = static_cast<int32_t>(std::floor(rec.y / res));
        const auto pixIt = pixelMap.find({ix, iy});
        if (pixIt == pixelMap.end()) {
            continue;
        }
        const auto stateIt = pixelState.find({ix, iy});
        RegionPixelState state = RegionPixelState::Boundary;
        if (stateIt != pixelState.end()) {
            state = stateIt->second;
        }
        if (state == RegionPixelState::Outside) {
            continue;
        }
        if (state == RegionPixelState::Boundary && !region.containsPoint(rec.x, rec.y, &tile)) {
            continue;
        }

        const auto& anno = pixIt->second;
        add_confusion(anno);
        const int32_t ux = static_cast<int32_t>(std::floor(rec.x / gridSize));
        const int32_t uy = static_cast<int32_t>(std::floor(rec.y / gridSize));
        for (size_t i = 0; i < anno.ks.size(); ++i) {
            if (anno.ps[i] < minProb) {
                continue;
            }
            const int32_t k = anno.ks[i];
            const float p = anno.ps[i];
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
        if (union_key == 0) {
            continue;
        }
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
