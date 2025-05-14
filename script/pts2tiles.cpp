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
    Pts2Tiles(int32_t nthreads, const std::string& inFile, std::string& tmpdir, const std::string& outPref, int32_t tileSize, int32_t icol_x, int32_t icol_y, int32_t icol_g = -1, std::vector<int32_t> icol_ints = {}, int32_t nskip = 0, int32_t buff = 1000) : nThreads(nthreads),
    inFile(inFile), tmpDir(tmpdir), outPref(outPref), tileSize(tileSize),
    icol_x(icol_x), icol_y(icol_y), icol_feature(icol_g), icol_ints(icol_ints),
    nskip(nskip), tileBuffer(buff) {
        minX = std::numeric_limits<double>::infinity();
        minY = std::numeric_limits<double>::infinity();
        maxX = -std::numeric_limits<double>::infinity();
        maxY = -std::numeric_limits<double>::infinity();
        ntokens = std::max(icol_x, icol_y);
        if (icol_feature >= 0) {
            ntokens = std::max(ntokens, icol_feature);
            for (const auto& icol : icol_ints) {
                ntokens = std::max(ntokens, icol);
            }
        }
        ntokens += 1;
        notice("Created temporary directory: %s", tmpDir.path.string().c_str());
    }
    virtual ~Pts2Tiles() {}

    const std::unordered_map<uint64_t, uint64_t>& getGlobalTiles() const {
        return globalTiles;
    }

    bool run() {
        notice("Launching %d worker threads", nThreads);
        if (!launchWorkerThreads()) {
            error("Error launching worker threads");
        }
        if (!joinWorkerThreads()) {
            error("Error joining worker threads");
        }
        notice("Merging temporary files and writing index");
        if (!mergeAndWriteIndex()) {
            error("Error merging temporary files and writing index");
        }
        if (!writeAuxiliaryFiles()) {
            warning("Error writing auxiliary files");
            return false;
        }
        return true;
    }

protected:
    int32_t nThreads;
    std::string inFile, outPref;
    ScopedTempDir tmpDir;
    int32_t tileBuffer, nskip;
    int32_t tileSize;
    int32_t icol_x, icol_y, icol_feature;
    std::vector<int32_t> icol_ints;
    int32_t ntokens;

    double minX, minY, maxX, maxY;
    std::unordered_map<std::string, std::vector<int32_t>> featureCounts;
    std::mutex minmaxMutex;

    std::mutex globalTilesMutex;
    std::unordered_map<uint64_t, uint64_t> globalTiles;

    std::vector<std::thread> threads;

    struct PtRecord {
        double x, y;
        std::string feature;
        std::vector<int32_t> vals;
    };

    virtual uint64_t parse(std::string& line, PtRecord& pt) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < ntokens) {
            error("Error parsing line: %s", line.c_str());
        }
        pt.x = std::stod(tokens[icol_x]);
        pt.y = std::stod(tokens[icol_y]);
        if (icol_feature >= 0) {
            pt.feature = tokens[icol_feature];
            if (icol_ints.size() > 0) {
                pt.vals.resize(icol_ints.size());
                for (size_t i = 0; i < icol_ints.size(); ++i) {
                    if (!str2int32(tokens[icol_ints[i]], pt.vals[i])) {
                        error("Error parsing the %d-th token to integer (%s) at line %s", icol_ints[i], tokens[icol_ints[i]].c_str(), line.c_str());
                    }
                }
            }
        }
        uint32_t row = static_cast<uint32_t>(std::floor(pt.y / tileSize));
        uint32_t col = static_cast<uint32_t>(std::floor(pt.x / tileSize));
        return ((static_cast<uint64_t>(row) << 32) | col);
    }

    virtual uint64_t parse(std::string& line, double& x, double& y) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < ntokens) {
            error("Error parsing line: %s", line.c_str());
        }
        x = std::stod(tokens[icol_x]);
        y = std::stod(tokens[icol_y]);
        uint32_t row = static_cast<uint32_t>(std::floor(y / tileSize));
        uint32_t col = static_cast<uint32_t>(std::floor(x / tileSize));
        return ((static_cast<uint64_t>(row) << 32) | col);
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
        std::unordered_map<uint64_t, std::vector<std::string>> buffers;
        double localMinX = std::numeric_limits<double>::infinity();
        double localMinY = std::numeric_limits<double>::infinity();
        double localMaxX = -std::numeric_limits<double>::infinity();
        double localMaxY = -std::numeric_limits<double>::infinity();
        std::unordered_map<std::string, std::vector<int32_t>> localCounts;
        while (file.tellg() < end && std::getline(file, line)) {
            PtRecord pt;
            uint64_t tileId = parse(line, pt);
            if (tileId == -1) {
                continue;
            }
            if (icol_feature >= 0) {
                auto it = localCounts.find(pt.feature);
                if (it == localCounts.end()) {
                    localCounts[pt.feature] = pt.vals;
                } else {
                    for (size_t i = 0; i < icol_ints.size(); ++i) {
                        it->second[i] += pt.vals[i];
                    }
                }
            }
            localMinX = std::min(localMinX, pt.x);
            localMaxX = std::max(localMaxX, pt.x);
            localMinY = std::min(localMinY, pt.y);
            localMaxY = std::max(localMaxY, pt.y);
            buffers[tileId].push_back(line);
            // Flush if the buffer is large enough.
            if (buffers[tileId].size() >= tileBuffer) {
                { // update globalTiles
                    std::lock_guard<std::mutex> lock(globalTilesMutex);
                    uint64_t npt = buffers[tileId].size();
                    if (globalTiles.find(tileId) == globalTiles.end()) {
                        globalTiles[tileId] = npt;
                    } else {
                        globalTiles[tileId] += npt;
                    }
                }
                auto tmpFilename = tmpDir.path / (std::to_string(tileId) + "_" + std::to_string(threadId) + ".tsv");
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
                    uint64_t npt = pair.second.size();
                    if (globalTiles.find(pair.first) == globalTiles.end()) {
                        globalTiles[pair.first] = npt;
                    } else {
                        globalTiles[pair.first] += npt;
                    }
                }
                auto tmpFilename = tmpDir.path / (std::to_string(pair.first) + "_" + std::to_string(threadId) + ".tsv");
                std::ofstream out(tmpFilename, std::ios::app);
                for (const auto& bufferedLine : pair.second) {
                    out << bufferedLine << "\n";
                }
                out.close();
            }
        }
        {
            std::lock_guard<std::mutex> lock(minmaxMutex);
            minX = std::min(minX, localMinX);
            maxX = std::max(maxX, localMaxX);
            minY = std::min(minY, localMinY);
            maxY = std::max(maxY, localMaxY);
            for (const auto& pair : localCounts) {
                auto it = featureCounts.find(pair.first);
                if (it == featureCounts.end()) {
                    featureCounts[pair.first] = pair.second;
                } else {
                    for (size_t i = 0; i < icol_ints.size(); ++i) {
                        it->second[i] += pair.second[i];
                    }
                }
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

    void mergeTmpFileToOutput(uint64_t tileId, std::ofstream& outfile) {
        for (uint32_t threadId = 0; threadId < nThreads; ++threadId) {
            auto tmpFilename = tmpDir.path / (std::to_string(tileId) + "_" + std::to_string(threadId) + ".tsv");
            std::ifstream tmpFile(tmpFilename, std::ios::binary);
            if (tmpFile) {
                outfile << tmpFile.rdbuf();
                tmpFile.close();
                // Remove temporary file after merging.
                std::filesystem::remove(tmpFilename);
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

        uint64_t nPoints = 0;
        std::vector<uint64_t> sortedTiles;
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
            indexfile << static_cast<int32_t>(tileId >> 32) << "\t" << static_cast<int32_t>(tileId & 0xFFFFFFFFULL) << "\t" << startOffset << "\t" << endOffset << "\t" << globalTiles[tileId] << "\n";
        }

        outfile.close();
        indexfile.close();
        return true;
    }

    bool writeAuxiliaryFiles() {
        size_t pos = outPref.find_last_of("/\\");
        std::string outDir = (pos == std::string::npos) ? "" : outPref.substr(0, pos + 1);
        std::string outFile = outDir + "coord_range.txt";
        std::ofstream out(outFile);
        if (!out) {
            warning("Error opening output file for writing: %s", outFile.c_str());
            return false;
        }
        out << "xmin\t" << minX << "\n"
            << "xmax\t" << maxX << "\n"
            << "ymin\t" << minY << "\n"
            << "ymax\t" << maxY << "\n";
        out.close();
        if (icol_feature >= 0) {
            std::string outFile = outDir + "features.tsv";
            out.open(outFile);
            if (!out) {
                warning("Error opening output file for writing: %s", outFile.c_str());
                return false;
            }
            for (const auto& pair : featureCounts) {
                out << pair.first;
                for (const auto& val : pair.second) {
                    out << "\t" << val;
                }
                out << "\n";
            }
            out.close();
        }
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
    uint64_t parse(std::string& line, float& x, float& y) {
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
        uint32_t row = static_cast<uint32_t>(std::floor(y / tileSize));
        uint32_t col = static_cast<uint32_t>(std::floor(x / tileSize));
        return ((static_cast<uint64_t>(row) << 32) | col);
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
        std::unordered_map<uint64_t, std::vector<std::string>> buffers;
        while (file.tellg() < end && std::getline(file, line)) {
            float pt[2];
            uint64_t tileId = parse(line, pt[0], pt[1]);
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
                    uint64_t npt = buffers[tileId].size();
                    if (globalTiles.find(tileId) == globalTiles.end()) {
                        globalTiles[tileId] = npt;
                    } else {
                        globalTiles[tileId] += npt;
                    }
                }
                auto tmpFilename = tmpDir.path / (std::to_string(tileId) + "_" + std::to_string(threadId) + ".tsv");
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
                    uint64_t npt = pair.second.size();
                    if (globalTiles.find(pair.first) == globalTiles.end()) {
                        globalTiles[pair.first] = npt;
                    } else {
                        globalTiles[pair.first] += npt;
                    }
                }
                auto tmpFilename = tmpDir.path / (std::to_string(pair.first) + "_" + std::to_string(threadId) + ".tsv");
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
    int icol_feature = -1;
    std::vector<int32_t> icol_ints;

    ParamList pl;
    // Input Options
    pl.add_option("in-tsv", "Input TSV file.", inTsv)
      .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x)
      .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y)
      .add_option("icol-feature", "Column index for feature (0-based)", icol_feature)
      .add_option("icol-int", "Column index for integer values (0-based)", icol_ints)
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

    Pts2Tiles pts2Tiles(nThreads, inTsv, tmpDir, outPref, tileSize, icol_x, icol_y, icol_feature, icol_ints, nskip, tileBuffer);
    if (!pts2Tiles.run()) {
        return 1;
    }

    notice("Processing completed. Output is written to %s.tsv and index file is written to %s.index", outPref.c_str(), outPref.c_str());
    return 0;
}
