#include "punkst.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <thread>
#include <mutex>

class Pts2Tiles {
public:
    Pts2Tiles(int32_t nthreads, const std::string& inFile, std::string& tmpdir, const std::string& outPref, int32_t buff = 1000, int32_t nskip = 0) : nThreads(nthreads), inFile(inFile), tmpDir(tmpdir), outPref(outPref), tileBuffer(buff), nskip(nskip) {
        if (!createDirectory(tmpDir)) {
            throw std::runtime_error("Error creating temporary directory (or the existing directory is not empty): " + tmpDir);
        }
        if (tmpDir.back() != '/') {
            tmpDir += "/";
        }
    }

    void initLineParser(int32_t tileSize, int32_t icol_x, int32_t icol_y) {
        parser.init(tileSize, icol_x, icol_y);
    }

    const std::unordered_map<int64_t, int64_t>& getGlobalTiles() const {
        return globalTiles;
    }

    bool run() {
        notice("Launching %d worker threads", nThreads);
        if (!launchWorkerThreads()) {
            error("Error launching worker threads");
            return false;
        }
        if (!joinWorkerThreads()) {
            error("Error joining worker threads");
            return false;
        }
        notice("Merging temporary files and writing index");
        if (!mergeAndWriteIndex()) {
            error("Error merging temporary files and writing index");
            return false;
        }
        return true;
    }

    struct lineParser {
        int32_t tileSize;
        int32_t icol_x;
        int32_t icol_y;

        lineParser() {}
        lineParser(int32_t tileSize, int32_t icol_x, int32_t icol_y)
            : tileSize(tileSize), icol_x(icol_x), icol_y(icol_y) {}
        void init(int32_t tileSize, int32_t icol_x, int32_t icol_y) {
            this->tileSize = tileSize;
            this->icol_x = icol_x;
            this->icol_y = icol_y;
        }

        int64_t parse(std::string& line) {
            std::istringstream iss(line);
            std::string token;
            double x, y;
            int32_t i = 0;
            while (std::getline(iss, token, '\t')) {
                if (i == icol_x) {
                    x = std::stod(token);
                } else if (i == icol_y) {
                    y = std::stod(token);
                }
                ++i;
            }
            int32_t row = static_cast<int32_t>(std::floor(y / tileSize));
            int32_t col = static_cast<int32_t>(std::floor(x / tileSize));
            return static_cast<int64_t>(row) << 32 | col ;
        }
    };


private:
    int32_t nThreads;
    std::string tmpDir, inFile, outPref;
    int32_t tileBuffer, nskip;

    std::mutex globalTilesMutex;
    std::unordered_map<int64_t, int64_t> globalTiles;

    std::vector<std::thread> threads;

    lineParser parser;

    // Worker thread function
    void worker(int threadId, std::streampos start, std::streampos end) {
        std::ifstream file(inFile);
        if (!file) {
            error("Error opening file in worker");
            return;
        }
        file.seekg(start);
        std::string line;
        // Map: tile id -> vector of lines (buffer)
        std::unordered_map<int64_t, std::vector<std::string>> buffers;
        while (file.tellg() < end && std::getline(file, line)) {
            int64_t tileId = parser.parse(line);
            buffers[tileId].push_back(line);
            // Flush if the buffer is large enough.
            if (buffers[tileId].size() >= tileBuffer) {
                { // update globalTiles
                    std::lock_guard<std::mutex> lock(globalTilesMutex);
                    int64_t npt = buffers[tileId].size();
                    if (globalTiles.find(tileId) == globalTiles.end()) {
                        globalTiles[tileId] = npt;
                    } else {
                        globalTiles[tileId] += npt;
                    }
                }
                std::string tmpFilename = tmpDir + std::to_string(tileId) + "_" + std::to_string(threadId) + ".tsv";
                std::ofstream out(tmpFilename, std::ios::app);
                for (const auto& bufferedLine : buffers[tileId]) {
                    out << bufferedLine << "\n";
                }
                out.close();
                buffers[tileId].clear();
            }
        }

        // Flush any remaining data in the buffers.
        for (auto& pair : buffers) {
            if (!pair.second.empty()) {
                { // update globalTiles
                    std::lock_guard<std::mutex> lock(globalTilesMutex);
                    int64_t npt = pair.second.size();
                    if (globalTiles.find(pair.first) == globalTiles.end()) {
                        globalTiles[pair.first] = npt;
                    } else {
                        globalTiles[pair.first] += npt;
                    }
                }
                std::string tmpFilename = tmpDir + std::to_string(pair.first) + "_" + std::to_string(threadId) + ".tsv";
                std::ofstream out(tmpFilename, std::ios::app);
                for (const auto& bufferedLine : pair.second) {
                    out << bufferedLine << "\n";
                }
                out.close();
            }
        }
    }

    bool launchWorkerThreads() {
        std::vector<std::pair<std::streampos, std::streampos>> blocks;
        computeBlocks(blocks, inFile, nThreads, nskip);
        for (size_t i = 0; i < blocks.size(); i++) {
            threads.emplace_back(&Pts2Tiles::worker, this, static_cast<int>(i), blocks[i].first, blocks[i].second);
        }
        return true;
    }

    bool joinWorkerThreads() {
        for (auto& t : threads) {
            t.join();
        }
        return true;
    }

    void mergeTmpFileToOutput(int64_t tileId, std::ofstream& outfile) {
        for (uint32_t threadId = 0; threadId < nThreads; ++threadId) {
            std::string tmpFilename = tmpDir + std::to_string(tileId) + "_" + std::to_string(threadId) + ".tsv";
            std::ifstream tmpFile(tmpFilename, std::ios::binary);
            if (tmpFile) {
                outfile << tmpFile.rdbuf();
                tmpFile.close();
                // Remove temporary file after merging.
                std::remove(tmpFilename.c_str());
            }
        }
    }

    // Merge temporary files by tile and write index file
    bool mergeAndWriteIndex() {
        std::ofstream outfile(outPref + ".tsv", std::ios::binary);
        if (!outfile) {
            error("Error opening output file: %s.tsv", outPref.c_str());
        }

        std::string indexFilename = outPref + ".index";
        std::ofstream indexfile(indexFilename);
        if (!indexfile) {
            error("Error opening index file: %s", indexFilename.c_str());
        }
        indexfile << "# tilesize\t" << parser.tileSize << "\n";

        int64_t nPoints = 0;
        std::vector<int64_t> sortedTiles;
        for (const auto& pair : globalTiles) {
            sortedTiles.push_back(pair.first);
            nPoints += pair.second;
        }
        indexfile << "# npixels\t" << nPoints << "\n";
        std::sort(sortedTiles.begin(), sortedTiles.end());

        for (const auto& tileId : sortedTiles) {
            std::streampos startOffset = outfile.tellp();
            mergeTmpFileToOutput(tileId, outfile);
            std::streampos endOffset = outfile.tellp();
            indexfile << (int32_t)(tileId >> 32) << "\t" << (int32_t)(tileId & 0xFFFFFFFF) << "\t" << startOffset << "\t" << endOffset << "\t" << globalTiles[tileId] << "\n";
        }

        outfile.close();
        indexfile.close();
        return true;
    }
};

int32_t cmdPts2TilesTsv(int32_t argc, char** argv) {

    std::string inTsv, outPref, tmpDir;
    int nThreads0 = 1, tileSize = 500000;
    int debug = 0, tileBuffer = 1000, verbose = 1000000;
    int icol_x, icol_y, nskip = 0;

	paramList pl;
	BEGIN_LONG_PARAMS(longParameters)
		LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("in-tsv", &inTsv, "Input TSV file. Header must begin with #")
        LONG_INT_PARAM("icol-x", &icol_x, "Column index for x coordinate (0-based)")
        LONG_INT_PARAM("icol-y", &icol_y, "Column index for y coordinate (0-based)")
        LONG_INT_PARAM("skip", &nskip, "Number of lines to skip in the input file (default: 0)")
        LONG_STRING_PARAM("temp-dir", &tmpDir, "Directory to store temporary files")
        LONG_INT_PARAM("tile-size", &tileSize, "Tile size in units (default: 300 um)")
        LONG_INT_PARAM("tile-buffer", &tileBuffer, "Buffer size per tile per thread (default: 1000 lines)")
        LONG_INT_PARAM("threads", &nThreads0, "Number of threads to use (default: 1)")
		LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("out-prefix", &outPref, "Output TSV file")
        LONG_INT_PARAM("verbose", &verbose, "Verbose")
        LONG_INT_PARAM("debug", &debug, "Debug")
    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

    // Determine the number of threads to use.
    unsigned int nThreads = std::thread::hardware_concurrency();
    if (nThreads == 0 || nThreads >= nThreads0) {
        nThreads = nThreads0;
    }
    notice("Using %u threads for processing", nThreads);

    Pts2Tiles pts2Tiles(nThreads, inTsv, tmpDir, outPref, tileBuffer, nskip);
    pts2Tiles.initLineParser(tileSize, icol_x, icol_y);
    if (!pts2Tiles.run()) {
        return 1;
    }

    notice("Processing completed. Output is written to %s.tsv and index file is written to %s.index", outPref.c_str(), outPref.c_str());
    return 0;
}
