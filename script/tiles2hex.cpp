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

class Tiles2Hex {

public:
    struct PixelValues {
        double x, y;
        std::string feature;
        std::vector<int32_t> intvals;
        PixelValues() {}
    };

    struct UnitValues {
        int32_t nPixel;
        uint32_t totVal;
        int32_t x, y;
        std::vector<std::unordered_map<std::string, uint32_t>> vals;
        UnitValues(int32_t hx, int32_t hy) : nPixel(0), totVal(0), x(hx), y(hy) {}
        UnitValues(int32_t hx, int32_t hy, const PixelValues& pixel) : nPixel(1), totVal(0), x(hx), y(hy) {
            vals.resize(pixel.intvals.size());
            for (size_t i = 0; i < pixel.intvals.size(); ++i) {
                if (pixel.intvals[i] > 0) {
                    vals[i][pixel.feature] = pixel.intvals[i];
                    totVal += pixel.intvals[i];
                }
            }
        }
        void addPixel(const PixelValues& pixel) {
            ++nPixel;
            for (size_t i = 0; i < pixel.intvals.size(); ++i) {
                if (pixel.intvals[i] > 0)
                    vals[i][pixel.feature] += pixel.intvals[i];
            }
        }
        bool mergeUnits(const UnitValues& other) {
            if (x != other.x || y != other.y) {
                return false;
            }
            nPixel += other.nPixel;
            for (size_t i = 0; i < vals.size(); ++i) {
                for (const auto& entry : other.vals[i]) {
                    if (entry.second > 0)
                        vals[i][entry.first] += entry.second;
                }
            }
            return true;
        }
        bool writeToFile(std::ostream& os, uint32_t key) const {
            os << uint32toHex(key) << "\t" << x << "\t" << y;
            for (const auto& val : vals) {
                os << "\t" << val.size();
            }
            for (const auto& val : vals) {
                for (const auto& entry : val) {
                    os << "\t" << entry.first << " " << entry.second;
                }
            }
            os << "\n";
            return os.good();
        }
    };

    struct lineParser {
        size_t icol_x, icol_y, icol_feature;
        std::vector<size_t> icol_ints;
        size_t n_ints;
        size_t n_tokens;

        lineParser() {}
        lineParser(size_t _ix, size_t _iy, size_t _iz, const std::vector<int32_t>& _ivals) {
            init(_ix, _iy, _iz, _ivals);
        }
        void init(size_t _ix, size_t _iy, size_t _iz, const std::vector<int32_t>& _ivals) {
            icol_x = _ix;
            icol_y = _iy;
            icol_feature = _iz;
            n_ints = _ivals.size();
            icol_ints.resize(n_ints);
            n_tokens = icol_feature;
            for (size_t i = 0; i < n_ints; ++i) {
                icol_ints[i] = (size_t) _ivals[i];
                n_tokens = std::max(n_tokens, icol_ints[i]);
            }
            n_tokens += 1;
        }

        int32_t parse(PixelValues& pixel, std::string& line) {
            std::vector<std::string> tokens;
            split(tokens, "\t", line);
            if (tokens.size() < n_tokens) {
                return -1;
            }
            pixel.x = std::stod(tokens[icol_x]);
            pixel.y = std::stod(tokens[icol_y]);
            pixel.feature = tokens[icol_feature];
            pixel.intvals.resize(n_ints);
            int32_t totVal = 0;
            for (size_t i = 0; i < n_ints; ++i) {
                if (!str2int32(tokens[icol_ints[i]], pixel.intvals[i])) {
                    return -1;
                }
                totVal += pixel.intvals[i];
            }
            return totVal;
        }
    };

    Tiles2Hex(int32_t nThreads, std::string& tmpDir, std::string& outFile, HexGrid& hexGrid, TileReader& tileReader, lineParser& parser)
        : numThreads(nThreads), tmpDir(tmpDir), outFile(outFile), hexGrid(hexGrid), tileReader(tileReader), parser(parser) {
        if (!createDirectory(tmpDir)) {
            error("Error creating temporary directory (or the existing directory is not empty): %s", tmpDir.c_str());
        }
        if (tmpDir.back() != '/') {
            tmpDir += "/";
        }
        nLayer = parser.n_ints;
        mainOut.open(outFile, std::ios::out);
    }
    ~Tiles2Hex() {
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
    int32_t nLayer;

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


int32_t cmdTiles2HexTxt(int32_t argc, char** argv) {

    std::string inTsv, inIndex, outFile, tmpDir;
    int nThreads = 1, debug = 0, verbose = 1000000;
    int icol_x, icol_y, icol_feature;
    double hexSize;
    std::vector<int32_t> icol_ints;

	paramList pl;
	BEGIN_LONG_PARAMS(longParameters)
		LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("in-tsv", &inTsv, "Input TSV file. Header must begin with #")
        LONG_STRING_PARAM("in-index", &inIndex, "Input index file")
        LONG_INT_PARAM("icol-x", &icol_x, "Column index for x coordinate (0-based)")
        LONG_INT_PARAM("icol-y", &icol_y, "Column index for y coordinate (0-based)")
        LONG_INT_PARAM("icol-feature", &icol_feature, "Column index for feature (0-based)")
        LONG_MULTI_INT_PARAM("icol-int", &icol_ints, "Column index for integer values (0-based)")
        LONG_DOUBLE_PARAM("hex-size", &hexSize, "Hexagon size (size length) in microns")
        LONG_STRING_PARAM("temp-dir", &tmpDir, "Directory to store temporary files")
        LONG_INT_PARAM("threads", &nThreads, "Number of threads to use (default: 1)")
		LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("out", &outFile, "Output TSV file")
        LONG_INT_PARAM("verbose", &verbose, "Verbose")
        LONG_INT_PARAM("debug", &debug, "Debug")
    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0 || numThreads > nThreads) {
        numThreads = nThreads;
    }

    HexGrid hexGrid(hexSize);
    TileReader tileReader(inTsv, inIndex);
    if (!tileReader.isValid()) {
        error("Error opening input file: %s", inTsv.c_str());
        return 1;
    }
    Tiles2Hex::lineParser parser(icol_x, icol_y, icol_feature, icol_ints);
    if (parser.n_ints == 0) {
        error("No integer columns specified");
    }
    Tiles2Hex tiles2Hex(numThreads, tmpDir, outFile, hexGrid, tileReader, parser);
    if (!tiles2Hex.mapreduce()) {
        return 1;
    }
    notice("Processing completed. Output is written to %s", outFile.c_str());

    return 0;
}
