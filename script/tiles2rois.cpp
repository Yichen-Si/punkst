#include <algorithm>
#include <atomic>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "json.hpp"
#include "punkst.h"
#include "region_query.hpp"
#include "threads.hpp"
#include "tilereader.hpp"

namespace {

struct RoiCounts {
    uint64_t total = 0;
    std::map<uint32_t, uint64_t> featureCounts;

    void add(uint32_t feature, uint64_t count) {
        if (count == 0) {
            return;
        }
        total += count;
        featureCounts[feature] += count;
    }

    void merge(const RoiCounts& other) {
        total += other.total;
        for (const auto& kv : other.featureCounts) {
            featureCounts[kv.first] += kv.second;
        }
    }
};

struct ParsedPoint {
    float x = 0.0f;
    float y = 0.0f;
    uint32_t feature = 0;
    uint64_t count = 1;
};

struct RoiTileTask {
    TileKey tile;
    std::vector<uint32_t> roiIds;
};

bool passes_min_count(const RoiCounts& counts, uint64_t minCount) {
    if (minCount == 0) {
        return true;
    }
    for (const auto& kv : counts.featureCounts) {
        if (kv.second >= minCount) {
            return true;
        }
    }
    return false;
}

void record_count_hist(std::map<uint32_t, uint64_t>& hist, uint32_t binSize, uint64_t count) {
    const uint32_t binStart = static_cast<uint32_t>((count / binSize) * binSize);
    hist[binStart] += 1;
}

void write_roi_row(std::ostream& out,
                   const PreparedGeoJSONFeature2D& roi,
                   const RoiCounts& counts) {
    out << roi.id << "\t"
        << std::fixed << std::setprecision(4) << roi.x << "\t" << roi.y << "\t"
        << counts.featureCounts.size() << "\t" << counts.total;
    for (const auto& kv : counts.featureCounts) {
        if (kv.second > 0) {
            out << "\t" << kv.first << " " << kv.second;
        }
    }
    out << "\n";
}

std::unordered_map<std::string, uint32_t> read_feature_dict(const std::string& dictFile) {
    std::unordered_map<std::string, uint32_t> dict;
    if (dictFile.empty()) {
        return dict;
    }
    std::ifstream in(dictFile);
    if (!in) {
        error("Error opening feature dictionary file: %s", dictFile.c_str());
    }
    std::string line;
    uint32_t n = 0;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        size_t pos = line.find_first_of(" \t");
        if (pos != std::string::npos) {
            line = line.substr(0, pos);
        }
        if (line.empty()) {
            continue;
        }
        dict.emplace(line, n++);
    }
    if (dict.empty()) {
        error("Error reading feature dictionary file: %s", dictFile.c_str());
    }
    notice("Read %zu features from dictionary file", dict.size());
    return dict;
}

bool parse_point_line(const std::string& line,
                      uint32_t ntok,
                      int32_t icolX,
                      int32_t icolY,
                      int32_t icolFeature,
                      int32_t icolCount,
                      const std::unordered_map<std::string, uint32_t>& featureDict,
                      ParsedPoint& out) {
    if (line.empty() || line[0] == '#') {
        return false;
    }
    std::vector<std::string> tokens;
    split(tokens, "\t", line, ntok + 1, true, true, true);
    if (tokens.size() < ntok) {
        error("%s: Invalid line: %s", __func__, line.c_str());
    }
    if (!str2float(tokens[icolX], out.x) || !str2float(tokens[icolY], out.y)) {
        error("%s: Invalid coordinates in line: %s", __func__, line.c_str());
    }
    if (featureDict.empty()) {
        if (!str2uint32(tokens[icolFeature], out.feature)) {
            error("%s: Invalid integer feature in line: %s", __func__, line.c_str());
        }
    } else {
        const auto it = featureDict.find(tokens[icolFeature]);
        if (it == featureDict.end()) {
            return false;
        }
        out.feature = it->second;
    }
    out.count = 1;
    if (icolCount >= 0) {
        int64_t count = 0;
        if (!str2num<int64_t>(tokens[icolCount], count)) {
            error("%s: Invalid count in line: %s", __func__, line.c_str());
        }
        if (count <= 0) {
            return false;
        }
        out.count = static_cast<uint64_t>(count);
    }
    return true;
}

} // namespace

int32_t cmdTiles2Rois(int32_t argc, char** argv) {
    std::string inTsv, inIndex, outPrefix, geojsonFile, idProperty = "title", dictFile;
    int32_t icolX = -1, icolY = -1, icolFeature = -1, icolCount = -1;
    int32_t threads = 1;
    int64_t geojsonScale = 100;
    uint64_t minCount = 1;
    int32_t verbose = 1000000;

    ParamList pl;
    pl.add_option("in-tsv", "Input tiled transcript TSV file", inTsv)
      .add_option("in-index", "Input tile index file", inIndex)
      .add_option("out", "Output prefix", outPrefix)
      .add_option("geojson", "GeoJSON FeatureCollection/Feature with Polygon/MultiPolygon ROI geometries", geojsonFile)
      .add_option("geojson-id-prop", "GeoJSON feature property used as ROI ID", idProperty)
      .add_option("geojson-scale", "Integer scale for GeoJSON region snapping", geojsonScale)
      .add_option("icol-x", "Column index for x coordinate (0-based)", icolX)
      .add_option("icol-y", "Column index for y coordinate (0-based)", icolY)
      .add_option("icol-feature", "Column index for feature/gene (0-based)", icolFeature)
      .add_option("icol-count", "Optional column index for count/value (0-based); default count is 1", icolCount)
      .add_option("feature-dict", "If feature column is not integer, provide the list of feature names", dictFile)
      .add_option("min-count", "Minimum count for at least one feature to emit an ROI row", minCount)
      .add_option("threads", "Number of threads to use", threads)
      .add_option("verbose", "Verbose", verbose);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (inTsv.empty() || inIndex.empty() || outPrefix.empty() || geojsonFile.empty()) {
        error("--in-tsv, --in-index, --out, and --geojson are required");
    }
    if (icolX < 0 || icolY < 0 || icolFeature < 0) {
        error("--icol-x, --icol-y, and --icol-feature are required");
    }
    if (icolX == icolY || icolX == icolFeature || icolY == icolFeature ||
        (icolCount >= 0 && (icolCount == icolX || icolCount == icolY || icolCount == icolFeature))) {
        error("Input column indices must be distinct");
    }
    if (geojsonScale <= 0) {
        error("--geojson-scale must be positive");
    }
    threads = std::max(1, threads);

    TileReader reader(inTsv, inIndex);
    if (!reader.isValid()) {
        error("Error opening input file: %s", inTsv.c_str());
    }
    const int32_t tileSize = reader.getTileSize();
    std::vector<PreparedGeoJSONFeature2D> rois;
    try {
        rois = loadPreparedGeoJSONFeatures(geojsonFile, tileSize, geojsonScale, idProperty);
    } catch (const std::exception& e) {
        error("%s: %s", __func__, e.what());
    }
    if (rois.empty()) {
        error("No GeoJSON ROI features found");
    }
    const auto featureDict = read_feature_dict(dictFile);

    Rectangle<double> globalRoiBox;
    bool hasBox = false;
    for (const auto& roi : rois) {
        const Rectangle<double> box(
            static_cast<double>(roi.region.bbox_f.xmin),
            static_cast<double>(roi.region.bbox_f.ymin),
            static_cast<double>(roi.region.bbox_f.xmax),
            static_cast<double>(roi.region.bbox_f.ymax));
        if (!hasBox) {
            globalRoiBox = box;
            hasBox = true;
        } else {
            globalRoiBox.xmin = std::min(globalRoiBox.xmin, box.xmin);
            globalRoiBox.ymin = std::min(globalRoiBox.ymin, box.ymin);
            globalRoiBox.xmax = std::max(globalRoiBox.xmax, box.xmax);
            globalRoiBox.ymax = std::max(globalRoiBox.ymax, box.ymax);
        }
    }
    if (!hasBox || !globalRoiBox.proper()) {
        error("Invalid GeoJSON ROI bounding box");
    }

    std::vector<Rectangle<double>> roiBounds{globalRoiBox};
    std::vector<TileKey> bboxTiles;
    std::vector<bool> bboxContained;
    reader.getTileList(roiBounds, bboxTiles, bboxContained);

    std::unordered_map<TileKey, std::vector<uint32_t>, TileKeyHash> tileToRois;
    std::vector<uint32_t> roiTileCounts(rois.size(), 0);
    for (uint32_t roiIdx = 0; roiIdx < rois.size(); ++roiIdx) {
        for (const auto& kv : rois[roiIdx].region.tile_bins) {
            tileToRois[kv.first].push_back(roiIdx);
        }
        roiTileCounts[roiIdx] = static_cast<uint32_t>(rois[roiIdx].region.tile_bins.size());
    }

    std::vector<RoiTileTask> tasks;
    tasks.reserve(bboxTiles.size());
    for (const auto& tile : bboxTiles) {
        auto it = tileToRois.find(tile);
        if (it == tileToRois.end() || it->second.empty()) {
            continue;
        }
        std::sort(it->second.begin(), it->second.end());
        it->second.erase(std::unique(it->second.begin(), it->second.end()), it->second.end());
        tasks.push_back(RoiTileTask{tile, it->second});
    }
    notice("%s: Loaded %zu ROIs; processing %zu tiles within the ROI bounding box",
        __func__, rois.size(), tasks.size());

    const std::string outTsv = outPrefix + ".tsv";
    std::ofstream out(outTsv);
    if (!out) {
        error("Error opening output file %s for writing", outTsv.c_str());
    }
    out << std::setprecision(4) << std::fixed;

    const uint32_t countHistBinSize = 5;
    std::map<uint32_t, uint64_t> countHist;
    std::mutex outMutex;
    std::mutex mergeMutex;
    std::vector<RoiCounts> globalCounts(rois.size());
    std::vector<uint8_t> emitted(rois.size(), 0);
    std::atomic<uint32_t> maxFeatureIdx{0};
    std::atomic<uint64_t> nTilesDone{0};
    ThreadSafeQueue<RoiTileTask> queue;

    uint32_t ntok = static_cast<uint32_t>(std::max({icolX, icolY, icolFeature, icolCount}));
    ntok += 1;

    auto worker = [&](int32_t threadId) {
        RoiTileTask task;
        std::unordered_map<uint32_t, RoiCounts> localMultiTile;
        while (queue.pop(task)) {
            auto iter = reader.get_tile_iterator(task.tile.row, task.tile.col);
            if (!iter) {
                continue;
            }

            std::unordered_map<uint32_t, RoiCounts> localSingleTile;
            std::string line;
            while (iter->next(line)) {
                ParsedPoint pt;
                if (!parse_point_line(line, ntok, icolX, icolY, icolFeature, icolCount, featureDict, pt)) {
                    continue;
                }
                uint32_t expected = maxFeatureIdx.load(std::memory_order_relaxed);
                while (pt.feature >= expected &&
                       !maxFeatureIdx.compare_exchange_weak(expected, pt.feature + 1,
                           std::memory_order_relaxed)) {}
                for (uint32_t roiIdx : task.roiIds) {
                    const auto& roi = rois[roiIdx];
                    if (!roi.region.containsPoint(pt.x, pt.y, &task.tile)) {
                        continue;
                    }
                    if (roiTileCounts[roiIdx] == 1) {
                        localSingleTile[roiIdx].add(pt.feature, pt.count);
                    } else {
                        localMultiTile[roiIdx].add(pt.feature, pt.count);
                    }
                }
            }

            if (!localSingleTile.empty()) {
                std::lock_guard<std::mutex> lock(outMutex);
                for (const auto& kv : localSingleTile) {
                    if (!passes_min_count(kv.second, minCount)) {
                        continue;
                    }
                    write_roi_row(out, rois[kv.first], kv.second);
                    record_count_hist(countHist, countHistBinSize, kv.second.total);
                    emitted[kv.first] = 1;
                }
                out.flush();
            }
            const uint64_t done = ++nTilesDone;
            if (verbose > 0 && (done % static_cast<uint64_t>(verbose)) == 0) {
                notice("Thread %d: processed %lu/%zu tiles", threadId, done, tasks.size());
            }
        }
        if (!localMultiTile.empty()) {
            std::lock_guard<std::mutex> lock(mergeMutex);
            for (const auto& kv : localMultiTile) {
                globalCounts[kv.first].merge(kv.second);
            }
        }
    };

    std::vector<std::thread> workers;
    workers.reserve(threads);
    for (int32_t i = 0; i < threads; ++i) {
        workers.emplace_back(worker, i);
    }
    for (const auto& task : tasks) {
        queue.push(task);
    }
    queue.set_done();
    for (auto& t : workers) {
        t.join();
    }

    uint64_t nUnits = 0;
    {
        std::lock_guard<std::mutex> lock(outMutex);
        for (uint32_t roiIdx = 0; roiIdx < rois.size(); ++roiIdx) {
            if (emitted[roiIdx]) {
                ++nUnits;
                continue;
            }
            const RoiCounts& counts = globalCounts[roiIdx];
            if (!passes_min_count(counts, minCount)) {
                continue;
            }
            write_roi_row(out, rois[roiIdx], counts);
            record_count_hist(countHist, countHistBinSize, counts.total);
            ++nUnits;
        }
    }
    out.close();

    const std::string histFile = outPrefix + ".count_hist.tsv";
    std::ofstream histOut(histFile);
    if (!histOut) {
        error("Error opening count histogram file %s for output", histFile.c_str());
    }
    histOut << "count_min\tcount_max\tn_units\n";
    for (const auto& kv : countHist) {
        histOut << kv.first << "\t" << (kv.first + countHistBinSize - 1) << "\t" << kv.second << "\n";
    }
    histOut.close();

    nlohmann::json meta;
    meta["unit_type"] = "geojson_roi";
    meta["geojson_file"] = geojsonFile;
    meta["geojson_id_property"] = idProperty;
    meta["geojson_scale"] = geojsonScale;
    meta["coord_dim"] = 2;
    meta["n_units"] = nUnits;
    meta["n_rois"] = rois.size();
    meta["n_features"] = featureDict.empty() ? maxFeatureIdx.load() : featureDict.size();
    meta["count_hist_bin_size"] = countHistBinSize;
    meta["icol_x"] = icolX;
    meta["icol_y"] = icolY;
    meta["icol_feature"] = icolFeature;
    meta["icol_count"] = icolCount;
    meta["n_modalities"] = 1;
    meta["offset_data"] = 3;
    meta["header_info"] = {"roi_id", "x", "y"};
    if (!featureDict.empty()) {
        nlohmann::json dictJson;
        for (const auto& kv : featureDict) {
            dictJson[kv.first] = kv.second;
        }
        meta["dictionary"] = std::move(dictJson);
    }

    const std::string metaFile = outPrefix + ".json";
    std::ofstream metaOut(metaFile);
    if (!metaOut) {
        error("Error opening metadata file %s for output", metaFile.c_str());
    }
    metaOut << std::setw(4) << meta << std::endl;
    metaOut.close();

    notice("Processing completed. Output is written to %s", outTsv.c_str());
    return 0;
}
