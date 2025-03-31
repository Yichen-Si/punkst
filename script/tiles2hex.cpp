#include "punkst.h"
#include "dataunits.hpp"
#include "hexgrid.h"
#include "tilereader.hpp"
#include "threads.hpp"
#include "utils.h"
#include <atomic>
#include <random>
#include <utility>
#include <iomanip>
#include <filesystem>
#include "json.hpp"

class Tiles2Hex {

public:

    Tiles2Hex(int32_t nThreads, std::string& tmpDir, std::string& outFile, HexGrid& hexGrid, TileReader& tileReader, lineParser& parser, std::vector<int32_t> minCounts = {})
        : nThreads(nThreads), tmpDir(tmpDir), outFile(outFile), hexGrid(hexGrid), tileReader(tileReader), parser(parser), minCounts(minCounts), nUnits(0), nFeatures(0) {
        if (!createDirectory(tmpDir)) {
            error("Error creating temporary directory (or the existing directory is not empty): %s", tmpDir.c_str());
        }
        if (tmpDir.back() != '/') {
            tmpDir += "/";
        }
        nLayer = parser.n_ints;
        mainOut.open(outFile, std::ios::out);
        if (!mainOut) {
            error("Error opening output file %s for writing", outFile.c_str());
            return;
        }
        if (minCounts.size() != nLayer) {
            minCounts.resize(nLayer);
            std::fill(minCounts.begin(), minCounts.end(), 1);
        }
    }
    ~Tiles2Hex() {
        if (mainOut.is_open()) {
            mainOut.close();
        }
    }

    bool run() {
        if (nThreads <= 1) {
            std::vector<TileKey> tileList;
            tileReader.getTileList(tileList);
            for (const auto& tile : tileList) {
                tileQueue.push(tile);
            }
            tileQueue.set_done();
            // Call worker directly instead of spawning a new thread.
            worker(0);
        } else {
            // Launch worker threads to process tiles.
            notice("Launching %d worker threads...", nThreads);
            if (!launchWorkerThreads()) {
                error("Error launching worker threads");
                return false;
            }
            // Wait for all worker threads to finish.
            if (!joinWorkerThreads()) {
                error("Error joining worker threads");
                return false;
            }
            notice("All worker threads finished");
        }
        // Merge boundary hexagons from temporary files and append to main output.
        notice("Merging boundary hexagons...");
        if (!mergeBoundaryHexagons()) {
            error("Error merging boundary hexagons");
            return false;
        }
        return true;
    }

    void writeMetadata() {
        meta["hex_size"] = hexGrid.size;
        meta["n_units"] = nUnits;
        meta["n_layers"] = nLayer;
        if (parser.isFeatureDict) {
            meta["n_features"] = parser.featureDict.size();
            for (const auto& entry : parser.featureDict) {
                meta["dictionary"][entry.second] = entry.first;
            }
        } else {
            meta["n_features"] = nFeatures;
        }
        // check if outFile has an extension
        std::string metaFile;
        size_t pos = outFile.find_last_of(".");
        if (pos != std::string::npos) {
            metaFile = outFile.substr(0, pos) + ".json";
        } else {
            metaFile = outFile + ".meat.json";
        }
        std::ofstream metaOut(metaFile);
        if (!metaOut) {
            error("Error opening metadata file %s for output", metaFile.c_str());
            return;
        }
        metaOut << std::setw(4) << meta << std::endl;
        metaOut.close();
    }

private:
    int32_t nThreads;
    std::string outFile, tmpDir;
    int32_t nLayer, nUnits, nFeatures;
    std::vector<int32_t> minCounts;
    nlohmann::json meta;

    lineParser parser;
    HexGrid hexGrid;
    TileReader tileReader;

    ThreadSafeQueue<TileKey> tileQueue;
    std::vector<std::thread> threads;
    std::mutex mainOutMutex;
    std::ofstream mainOut;

    void addPixelToUnitMaps(PixelValues& pixel, int32_t hx, int32_t hy, std::unordered_map<int64_t, UnitValues>& Units) {
        int64_t key = (static_cast<int64_t>(hx) << 32) | hy;
        auto it = Units.find(key);
        if (it == Units.end()) {
            Units.insert({key, UnitValues(hx, hy, pixel)});
        } else {
            it->second.addPixel(pixel);
        }
    }

    void worker(int threadId) {
        // Set up a per-thread random engine.
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);
        // temporary file
        std::string partialFile = tmpDir + std::to_string(threadId) + ".txt";
        std::ofstream outFile(partialFile, std::ios::out);
        if (!outFile) {
            error("Error opening temporary file: %s", partialFile.c_str());
        }
        uint32_t maxFeatureIdxLocal = 0;
        // Process one tile at a time
        TileKey tile;
        int tileSize = tileReader.getTileSize();
        while (tileQueue.pop(tile)) {

            // Compute tile’s cartesian bounding box.
            double tile_xmin = tile.col * tileSize;
            double tile_xmax = (tile.col + 1) * tileSize;
            double tile_ymin = tile.row * tileSize;
            double tile_ymax = (tile.row + 1) * tileSize;

            // Get an iterator for points in this tile.
            std::unique_ptr<BoundedReadline> iter;
            try {
                iter = tileReader.get_tile_iterator(tile.row, tile.col);
            } catch (const std::exception &e) {
                std::cerr << "Tile (" << tile.row << "," << tile.col
                    << ") iterator error: " << e.what() << "\n";
                continue;
            }

            // Local collection of hexagons
            std::unordered_map<int64_t, UnitValues> unitBuffer;

            std::string line;
            while (iter->next(line)) {
                PixelValues pixel;
                int32_t ret = parser.parse(pixel, line);
                if (!parser.isFeatureDict && pixel.feature >= maxFeatureIdxLocal) {
                    maxFeatureIdxLocal = pixel.feature + 1;
                }
                if (ret < 0) {
                    error("Error parsing line: %s", line.c_str());
                }
                if (ret == 0) {
                    continue;
                }
                // Convert point to hex axial coordinate & add to buffer
                int32_t hx, hy;
                hexGrid.cart_to_axial(hx, hy, pixel.x, pixel.y);
                addPixelToUnitMaps(pixel, hx, hy, unitBuffer);
            } // end while reading lines in tile
            if (unitBuffer.empty()) {
                continue;
            }
            { // Lock the global output file.
                std::lock_guard<std::mutex> lock(mainOutMutex);
                for (const auto &entry : unitBuffer) {
                    // Get the hexagon’s bounding box
                    double hex_xmin, hex_xmax, hex_ymin, hex_ymax;
                    hexGrid.hex_bounding_box_axial(hex_xmin, hex_xmax, hex_ymin, hex_ymax, entry.second.x, entry.second.y);
                    // Determine if the hexagon is fully contained inside the tile
                    bool isInternal = (
                        hex_xmin >= tile_xmin && hex_xmax < tile_xmax &&
                        hex_ymin >= tile_ymin && hex_ymax < tile_ymax);
                    if (isInternal) {
                        bool flag = false;
                        for (size_t i = 0; i < nLayer; ++i) {
                            if (entry.second.valsums[i] >= minCounts[i]) {
                                flag = true;
                                break;
                            }
                        }
                        if (!flag) {
                            continue;
                        }
                        uint32_t iden = rdUnif(gen);
                        entry.second.writeToFile(mainOut, iden);
                        nUnits++;
                    } else {
                        entry.second.writeToFile(outFile, 0);
                    }
                }
                mainOut.flush();
                if (!parser.isFeatureDict && nFeatures < maxFeatureIdxLocal) {
                    nFeatures = maxFeatureIdxLocal;
                }
            }
        } // end while tiles
        outFile.close();
    }

    bool launchWorkerThreads() {
        for (int i = 0; i < nThreads; ++i) {
            threads.emplace_back(&Tiles2Hex::worker, this, i);
        }
        std::vector<TileKey> tileList;
        tileReader.getTileList(tileList);
        for (const auto& tile : tileList) {
            tileQueue.push(tile);
        }
        tileQueue.set_done();
        return true;
    }

    bool joinWorkerThreads() {
        for (auto& t : threads) {
            t.join();
        }
        return true;
    }

    // Merge boundary hexagons from all temporary files and append to mainOut
    bool mergeBoundaryHexagons() {
        std::unordered_map<int64_t, UnitValues> mergedUnits;
        for (int i = 0; i < nThreads; ++i) {
            std::string fname = tmpDir + std::to_string(i) + ".txt";
            std::ifstream ifs(fname);
            if (!ifs) {
                continue; // unlikely?
            }
            std::string line;
            int32_t nunits = 0, nnew = 0;
            while (std::getline(ifs, line)) {
                if (line.empty()) continue;
                UnitValues unit(0, 0); // temporary initialization
                if (!unit.readFromLine(line, nLayer)) {
                    std::cerr << "Error parsing boundary line: " << line << "\n";
                    continue;
                }
                nunits++;
                int64_t key = (static_cast<int64_t>(unit.x) << 32) | unit.y;
                auto it = mergedUnits.find(key);
                if (it == mergedUnits.end()) {
                    mergedUnits.insert({key, unit});
                    nnew++;
                } else {
                    it->second.mergeUnits(unit);
                }
            }
            ifs.close();
            std::filesystem::remove(fname);
            notice("Merging temporary file from thread %d... %d units, %d new", i, nunits, nnew);
        }
        // Append merged boundary units to main output.
        {
            std::lock_guard<std::mutex> lock(mainOutMutex);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);
            int32_t nUnits0 = nUnits;
            for (const auto &entry : mergedUnits) {
                bool flag = false;
                for (size_t i = 0; i < nLayer; ++i) {
                    if (entry.second.valsums[i] >= minCounts[i]) {
                        flag = true;
                        break;
                    }
                }
                if (!flag) {
                    continue;
                }
                uint32_t rnd = rdUnif(gen);
                entry.second.writeToFile(mainOut, rnd);
                nUnits++;
            }
            mainOut.flush();
            notice("Wrote %d/%zu boundary hexagons to main output", nUnits-nUnits0, mergedUnits.size());
        }
        mainOut.close();
        return true;
    }
};


int32_t cmdTiles2HexTxt(int32_t argc, char** argv) {

    std::string inTsv, inIndex, outFile, tmpDir, dictFile;
    int nThreads = 1, debug = 0, verbose = 1000000;
    int icol_x, icol_y, icol_feature;
    double hexSize;
    std::vector<int32_t> icol_ints;
    std::vector<int32_t> min_counts;

	paramList pl;
	BEGIN_LONG_PARAMS(longParameters)
		LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("in-tsv", &inTsv, "Input TSV file. Header must begin with #")
        LONG_STRING_PARAM("in-index", &inIndex, "Input index file")
        LONG_INT_PARAM("icol-x", &icol_x, "Column index for x coordinate (0-based)")
        LONG_INT_PARAM("icol-y", &icol_y, "Column index for y coordinate (0-based)")
        LONG_INT_PARAM("icol-feature", &icol_feature, "Column index for feature (0-based)")
        LONG_STRING_PARAM("feature-dict", &dictFile, "If feature column is not integer, provide a dictionary/list of all possible values")
        LONG_MULTI_INT_PARAM("icol-int", &icol_ints, "Column index for integer values (0-based)")
        LONG_DOUBLE_PARAM("hex-size", &hexSize, "Hexagon size (size length) in microns")
        LONG_STRING_PARAM("temp-dir", &tmpDir, "Directory to store temporary files")
        LONG_INT_PARAM("threads", &nThreads, "Number of threads to use (default: 1)")
		LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("out", &outFile, "Output TSV file")
        LONG_MULTI_INT_PARAM("min-count", &min_counts, "Minimum count for each integer column, applied with OR")
        LONG_INT_PARAM("verbose", &verbose, "Verbose")
        LONG_INT_PARAM("debug", &debug, "Debug")
    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

    HexGrid hexGrid(hexSize);
    TileReader tileReader(inTsv, inIndex);
    if (!tileReader.isValid()) {
        error("Error opening input file: %s", inTsv.c_str());
        return 1;
    }
    lineParser parser(icol_x, icol_y, icol_feature, icol_ints, dictFile);
    if (parser.n_ints == 0) {
        error("No integer columns specified");
    }
    Tiles2Hex tiles2Hex(nThreads, tmpDir, outFile, hexGrid, tileReader, parser, min_counts);
    if (!tiles2Hex.run()) {
        return 1;
    }
    tiles2Hex.writeMetadata();
    notice("Processing completed. Output is written to %s", outFile.c_str());

    return 0;
}
