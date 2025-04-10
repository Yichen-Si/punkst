#include "punkst.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <memory>
#include "nanoflann.hpp"
#include "nanoflann_utils.h"

class Pts2Tiles {
public:
    Pts2Tiles(int32_t nthreads, const std::string& inFile, std::string& tmpdir, const std::string& outPref, int32_t tileSize, int32_t icol_x, int32_t icol_y, int32_t nskip = 0, int32_t buff = 1000) : nThreads(nthreads), inFile(inFile), tmpDir(tmpdir), outPref(outPref), tileSize(tileSize), icol_x(icol_x), icol_y(icol_y), nskip(nskip), tileBuffer(buff) {
        if (!createDirectory(tmpDir)) {
            throw std::runtime_error("Error creating temporary directory (or the existing directory is not empty): " + tmpDir);
        }
        if (tmpDir.back() != '/') {
            tmpDir += "/";
        }
    }
    virtual ~Pts2Tiles() {}

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

protected:
    int32_t nThreads;
    std::string tmpDir, inFile, outPref;
    int32_t tileBuffer, nskip;
    int32_t tileSize;
    int32_t icol_x, icol_y;

    std::mutex globalTilesMutex;
    std::unordered_map<int64_t, int64_t> globalTiles;

    std::vector<std::thread> threads;

    virtual int64_t parse(std::string& line) {
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

    // Worker thread function
    virtual void worker(int threadId, std::streampos start, std::streampos end) {
        std::ifstream file(inFile);
        if (!file) {
            error("Thread %d: Error opening file input file", threadId);
        }
        file.seekg(start);
        std::string line;
        // Map: tile id -> vector of lines (buffer)
        std::unordered_map<int64_t, std::vector<std::string>> buffers;
        while (file.tellg() < end && std::getline(file, line)) {
            int64_t tileId = parse(line);
            if (tileId == -1) {
                continue;
            }
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
            error("Error opening output file for writing: %s.tsv", outPref.c_str());
        }

        std::string indexFilename = outPref + ".index";
        std::ofstream indexfile(indexFilename);
        if (!indexfile) {
            error("Error opening index file for writing: %s", indexFilename.c_str());
        }
        indexfile << "# tilesize\t" << tileSize << "\n";

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

// NOT TESTED YET
class Pts2TilesAnno2D: public Pts2Tiles {
public:
    std::vector<std::unique_ptr<kd_tree_f2_t> > trees;
    int32_t dist_out_precision;

    void add_refpts(const std::string& file, int32_t f = 4) {
        PointCloud<float> cloud;
        std::ifstream ifs(file);
        if (!ifs) {
            error("Error opening reference points file: %s", file.c_str());
        }
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            float x, y;
            iss >> x >> y;
            cloud.pts.push_back({x, y});
        }
        trees.push_back(std::unique_ptr<kd_tree_f2_t>(new kd_tree_f2_t(2, cloud, {10})));
        dist_out_precision = f;
    }
    void initialize_refpts(const std::vector<std::string>& files, int32_t f = 4) {
        for (const auto& file : files) {
            add_refpts(file, f);
        }
    }

    ~Pts2TilesAnno2D() {}

protected:
    int64_t parse(std::string& line, float& x, float& y) {
        std::istringstream iss(line);
        std::string token;
        int32_t i = 0;
        while (std::getline(iss, token, '\t')) {
            if (i == icol_x) {
                x = std::stof(token);
            } else if (i == icol_y) {
                y = std::stof(token);
            }
            ++i;
        }
        int32_t row = static_cast<int32_t>(std::floor(y / tileSize));
        int32_t col = static_cast<int32_t>(std::floor(x / tileSize));
        return static_cast<int64_t>(row) << 32 | col ;
    }

    // Worker thread function
    virtual void worker(int threadId, std::streampos start, std::streampos end) {
        std::ifstream file(inFile);
        if (!file) {
            error("Thread %d: Error opening file input file", threadId);
        }
        file.seekg(start);
        std::string line;
        // Map: tile id -> vector of lines (buffer)
        std::unordered_map<int64_t, std::vector<std::string>> buffers;
        while (file.tellg() < end && std::getline(file, line)) {
            float pt[2];
            int64_t tileId = parse(line, pt[0], pt[1]);
            if (tileId == -1) {
                continue;
            }
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(dist_out_precision);
            // fine the distance to the nearest reference point in each kdtree
            for (size_t i = 0; i < trees.size(); ++i) {
                std::vector<uint32_t> indices(1);
                std::vector<float> dists(1);
                trees[i]->knnSearch(pt, 1, &indices[0], &dists[0]);
                oss << "\t" << dists[0];
            }
            line += oss.str();
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

};


int32_t cmdPts2TilesTsv(int32_t argc, char** argv) {

    std::string inTsv, outPref, tmpDir;
    int nThreads0 = 1, tileSize = -1;
    int debug = 0, tileBuffer = 1000, verbose = 1000000;
    int icol_x, icol_y, nskip = 0;

    ParamList pl;
    // Input Options
    pl.add_option("in-tsv", "Input TSV file.", inTsv)
      .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x)
      .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y)
      .add_option("skip", "Number of lines to skip in the input file (default: 0)", nskip)
      .add_option("temp-dir", "Directory to store temporary files", tmpDir)
      .add_option("tile-size", "Tile size (in the same unit as the input coordinates)", tileSize)
      .add_option("tile-buffer", "Buffer size per tile per thread (default: 1000 lines)", tileBuffer)
      .add_option("threads", "Number of threads to use (default: 1)", nThreads0);
    // Output Options
    pl.add_option("out-prefix", "Output TSV file", outPref)
      .add_option("verbose", "Verbose", verbose)
      .add_option("debug", "Debug", debug);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (tileSize <= 0) {
        error("Tile size is required");
    }

    // Determine the number of threads to use.
    unsigned int nThreads = std::thread::hardware_concurrency();
    if (nThreads == 0 || nThreads >= nThreads0) {
        nThreads = nThreads0;
    }
    notice("Using %u threads for processing", nThreads);

    Pts2Tiles pts2Tiles(nThreads, inTsv, tmpDir, outPref, tileSize, icol_x, icol_y, nskip, tileBuffer);
    if (!pts2Tiles.run()) {
        return 1;
    }

    notice("Processing completed. Output is written to %s.tsv and index file is written to %s.index", outPref.c_str(), outPref.c_str());
    return 0;
}
