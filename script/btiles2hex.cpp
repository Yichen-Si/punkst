#include "punkst.h"
#include "hexgrid.h"
#include "tilereader.hpp"
#include "threads.hpp"
#include "utils.h"
#include <atomic>
#include <random>
#include <utility>
#include <iomanip>
#include <filesystem>
#include "qgenlib/qgen_utils.h"

class BinaryTiles2Hex {

public:
    struct PixelValues {
        double x, y;
        uint32_t catval;
        std::vector<int32_t> intvals;
        PixelValues() {}
    };

    struct UnitValues {
        int32_t nPixel;
        int32_t x, y;
        uint32_t nInt;
        std::unordered_map<uint32_t, std::vector<int32_t>> intvals;
        UnitValues(int32_t hx, int32_t hy, uint32_t ni) : nPixel(0), nInt(ni), x(hx), y(hy) {
            assert(nInt > 0);
        }
        UnitValues(int32_t hx, int32_t hy, const PixelValues& pixel) : nPixel(1), x(hx), y(hy) {
            intvals[pixel.catval] = pixel.intvals;
            nInt = pixel.intvals.size();
        }

        void addPixel(const PixelValues& pixel) {
            ++nPixel;
            assert(nInt == pixel.intvals.size());
            auto ptr = intvals.find(pixel.catval);
            if (ptr == intvals.end()) {
                intvals[pixel.catval] = pixel.intvals;
            } else {
                for (size_t i = 0; i < pixel.intvals.size(); ++i) {
                    if (pixel.intvals[i] > 0)
                        ptr->second[i] += pixel.intvals[i];
                }
            }
        }
        bool mergeUnits(const UnitValues& other) {
            if (x != other.x || y != other.y) {
                return false;
            }
            assert(nInt == other.nInt);
            nPixel += other.nPixel;
            for (const auto& entry : other.intvals) {
                auto it = intvals.find(entry.first);
                if (it == intvals.end()) {
                    intvals[entry.first] = entry.second;
                } else {
                    for (size_t i = 0; i < nInt; ++i) {
                        it->second[i] += entry.second[i];
                    }
                }
            }
            return true;
        }
        bool writeToFile(std::ostream& os, uint32_t key) const {
            os << uint32toHex(key) << "\t" << x << "\t" << y << "\t" << nPixel;
            os << "\t" << intvals.size() << "\t" << nInt;
            for (const auto& entry : intvals) {
                os << "\t" << entry.first;
                for (size_t i = 0; i < nInt; ++i) {
                    os << " " << entry.second[i];
                }
            }
            os << "\n";
            return os.good();
        }
    };

    BinaryTiles2Hex(int32_t nThreads, std::string& tmpDir, std::string& outFile, HexGrid& hexGrid, BinaryTileReader& tileReader, int32_t iCat, std::vector<int32_t>& iVals)
        : numThreads(nThreads), tmpDir(tmpDir), outFile(outFile), hexGrid(hexGrid), tileReader(tileReader), iCat(iCat), iVals(iVals) {
        if (!createDirectory(tmpDir)) {
            error("Error creating temporary directory (or the existing directory is not empty): %s", tmpDir.c_str());
        }
        if (tmpDir.back() != '/') {
            tmpDir += "/";
        }
        nVals = iVals.size();
        mainOut.open(outFile, std::ios::out);
        if (!mainOut) {
            error("Error opening output file: %s", outFile.c_str());
        }
    }
    ~BinaryTiles2Hex() {
        if (mainOut.is_open()) {
            mainOut.close();
        }
    }

    bool mapreduce() {
        // Launch worker threads to process tiles.
        if (!launchWorkerThreads()) {
            error("Error launching worker threads");
            return false;
        }
        // Wait for all worker threads to finish.
        if (!joinWorkerThreads()) {
            error("Error joining worker threads");
            return false;
        }
        // Merge boundary hexagons from temporary files and append to main output.
        if (!mergeBoundaryHexagons()) {
            error("Error merging boundary hexagons");
            return false;
        }
        return true;
    }

private:
    int32_t numThreads;
    std::string outFile, tmpDir;
    int32_t nVals;
    int32_t iCat;
    std::vector<int32_t> iVals;

    HexGrid hexGrid;
    BinaryTileReader tileReader;

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
            {
                // Lock the global output file.
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
                        uint32_t iden = rdUnif(gen);
                        entry.second.writeToFile(mainOut, iden);
                    } else {
                        entry.second.writeToFile(outFile, 0);
                    }
                }
                mainOut.flush();
            }
        } // end while tiles
        outFile.close();
    }

    bool launchWorkerThreads() {
        for (int i = 0; i < numThreads; ++i) {
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
        for (int i = 0; i < numThreads; ++i) {
            std::string fname = tmpDir + std::to_string(i) + ".txt";
            std::ifstream ifs(fname);
            if (!ifs) {
                continue; // unlikely?
            }
            std::string line;
            while (std::getline(ifs, line)) {
                if (line.empty()) continue;
                int64_t key;
                UnitValues unit(0, 0); // temporary initialization
                if (!parseBoundaryLine(line, key, unit)) {
                    std::cerr << "Error parsing boundary line: " << line << "\n";
                    continue;
                }
                auto it = mergedUnits.find(key);
                if (it == mergedUnits.end()) {
                    mergedUnits.insert({key, unit});
                } else {
                    it->second.mergeUnits(unit);
                }
            }
            ifs.close();
            std::filesystem::remove(fname);
        }
        // Append merged boundary units to main output.
        {
            std::lock_guard<std::mutex> lock(mainOutMutex);
            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);
            for (const auto &entry : mergedUnits) {
                uint32_t rnd = rdUnif(gen);
                entry.second.writeToFile(mainOut, rnd);
            }
            mainOut.flush();
        }
        mainOut.close();
        return true;
    }

    // Helper function to parse a boundary hexagon record
    bool parseBoundaryLine(std::string &line, int64_t &key, UnitValues &unit) {
        std::istringstream iss(line);
        // key, x, y, nfeature1, ...
        std::string hexKey;
        int32_t hx, hy;
        if (!(iss >> hexKey >> hx >> hy)) {
            return false;
        }
        unit = UnitValues(hx, hy);
        unit.vals.resize(nLayer);
        std::vector<int32_t> nfeatures(nLayer, 0);
        for (int i = 0; i < nLayer; ++i) {
            if (!(iss >> nfeatures[i])) {
                return false;
            }
        }
        for (size_t i = 0; i < nLayer; ++i) {
            for (int j = 0; j < nfeatures[i]; ++j) {
                std::string feature;
                int32_t value;
                if (!(iss >> feature >> value)) {
                    return false;
                }
                unit.vals[i][feature] = value;
            }
        }
        key = (static_cast<int64_t>(hx) << 32) | hy;
        return true;
    }

};
