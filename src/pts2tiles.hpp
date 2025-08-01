#pragma once

#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <array>
#include "zlib.h"
#include "threads.hpp"
#include "nanoflann.hpp"
#include "nanoflann_utils.h"

class Pts2Tiles {
public:
    Pts2Tiles(int32_t nthreads,
        const std::string& _inFile, std::string& _tmpDir,
        const std::string& _outPref, int32_t _tileSize,
        int32_t icol_x, int32_t icol_y, int32_t icol_g = -1, std::vector<int32_t> icol_ints = {}, int32_t _nskip = 0,
        bool _streamingMode = false,
        int32_t _tileBuffer = 1000, int32_t _batchSize = 10000, double _scale = 0, int _digits=2) :
        nThreads(nthreads),
        inFile(_inFile), tmpDir(_tmpDir),
        outPref(_outPref), tileSize(_tileSize),
        icol_x(icol_x), icol_y(icol_y), icol_feature(icol_g),
        icol_ints(icol_ints), nskip(_nskip),
        streamingMode(_streamingMode),
        tileBuffer(_tileBuffer), batchSize(_batchSize),
        scale(_scale), digits(_digits) {
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
        scaling = (std::abs(scale) > 1e-20);
        ntokens += 1;
        notice("Created temporary directory: %s", tmpDir.path.string().c_str());
    }
    virtual ~Pts2Tiles() {
        if (gz) {
            gzclose(gz);
        }
        if (inPtr && inPtr != &std::cin) {
            delete inPtr;
        }
    }

    const std::unordered_map<uint64_t, uint64_t>& getGlobalTiles() const {
        return globalTiles;
    }

    bool run() {
        if (!streamingMode) {
            if (!launchWorkerThreads()) {
                error("Error launching worker threads");
            }
        } else {
            openInput();
            for (int i = 0; i < nThreads; ++i) {
                threads.emplace_back(&Pts2Tiles::streamingWorker, this, i);
            }
        }
        notice("Launched %d worker threads", nThreads);
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
    int32_t tileBuffer, batchSize;
    int32_t tileSize;
    int32_t icol_x, icol_y, icol_feature;
    std::vector<int32_t> icol_ints;
    int32_t ntokens;
    int32_t nskip, nskipped = 0;
    double scale;
    bool scaling;
    int32_t digits;

    double minX, minY, maxX, maxY;
    std::unordered_map<std::string, std::vector<int32_t>> featureCounts;
    std::mutex minmaxMutex;

    std::mutex globalTilesMutex;
    std::unordered_map<uint64_t, uint64_t> globalTiles;

    std::vector<std::thread> threads;
    bool streamingMode;
    gzFile         gz    = nullptr;
    std::istream*  inPtr = nullptr;
    std::mutex readMutex;

    struct PtRecord {
        double x, y;
        std::string feature;
        std::vector<int32_t> vals;
    };

    void openInput() {
        if (inFile == "-") {
            inPtr = &std::cin;
        }
        else if (ends_with(inFile, ".gz")) {
            gz = gzopen(inFile.c_str(), "rb");
            if (gz == Z_NULL) {
                error("Error opening gzipped input file: %s", inFile.c_str());
            }
        }
        else {
            warning("%s: the input is not stdin or gzipped file but the streaming mode is used, assuming it is a plain tsv file", __FUNCTION__);
            inPtr = new std::ifstream(inFile);
            if (!inPtr || !static_cast<std::ifstream*>(inPtr)->is_open()) {
                error("Error opening input file: %s", inFile.c_str());
            }
        }
    }

    virtual uint64_t parse(std::string& line, PtRecord& pt) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < ntokens) {
            error("Error parsing line: %s", line.c_str());
        }
        pt.x = std::stod(tokens[icol_x]);
        pt.y = std::stod(tokens[icol_y]);
        if (scaling) {
            pt.x *= scale;
            pt.y *= scale;
            tokens[icol_x] = fp_to_string(pt.x, digits);
            tokens[icol_y] = fp_to_string(pt.y, digits);
            line = join(tokens, "\t");
        }
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
        if (scaling) {
            x *= scale;
            y *= scale;
            tokens[icol_x] = fp_to_string(x, digits);
            tokens[icol_y] = fp_to_string(y, digits);
            line = join(tokens, "\t");
        }
        uint32_t row = static_cast<uint32_t>(std::floor(y / tileSize));
        uint32_t col = static_cast<uint32_t>(std::floor(x / tileSize));
        return ((static_cast<uint64_t>(row) << 32) | col);
    }

    void consumeLine(int threadId, std::string &line,
            std::unordered_map<uint64_t, std::vector<std::string>> &buffers,
            double &localMinX,  double &localMaxX,
            double &localMinY,  double &localMaxY,
            std::unordered_map<std::string,std::vector<int32_t>> &localCounts) {
        PtRecord pt;
        uint64_t tileId = parse(line, pt);
        if (tileId == uint64_t(-1)) return;

        // feature counts
        if (icol_feature >= 0) {
            auto &v = localCounts[pt.feature];
            if (v.empty()) v = pt.vals;
            else for (size_t i = 0; i < pt.vals.size(); ++i)
                    v[i] += pt.vals[i];
        }

        // min/max
        localMinX = std::min(localMinX, pt.x);
        localMaxX = std::max(localMaxX, pt.x);
        localMinY = std::min(localMinY, pt.y);
        localMaxY = std::max(localMaxY, pt.y);

        // buffer + flush
        auto &buf = buffers[tileId];
        buf.push_back(line);
        if (buf.size() >= static_cast<size_t>(tileBuffer))
            flushBuffer(threadId, tileId, buf);
    }

    void flushBuffer(int threadId, uint64_t tileId,
                     std::vector<std::string> &buf) {
        {
            std::lock_guard lk(globalTilesMutex);
            globalTiles[tileId] += buf.size();
        }
        auto fn = tmpDir.path / (std::to_string(tileId) + "_"
                                + std::to_string(threadId) + ".tsv");
        std::ofstream out(fn, std::ios::app);
        for (auto &l : buf) out << l << "\n";
        out.close();
        buf.clear();
    }

    void flushAll(int threadId,
            std::unordered_map<uint64_t,std::vector<std::string>> &buffers) {
        for (auto &p : buffers)
        if (!p.second.empty())
            flushBuffer(threadId, p.first, p.second);
    }

    void mergeLocalStats(double localMinX, double localMaxX,
                         double localMinY, double localMaxY,
            std::unordered_map<std::string,std::vector<int32_t>> &localCounts) {
        std::lock_guard lk(minmaxMutex);
        minX = std::min(minX, localMinX);
        maxX = std::max(maxX, localMaxX);
        minY = std::min(minY, localMinY);
        maxY = std::max(maxY, localMaxY);
        for (auto &p : localCounts) {
            auto &glob = featureCounts[p.first];
            if (glob.empty()) {
                glob = std::move(p.second);
            } else {
                for (size_t i = 0; i < glob.size(); ++i) {
                    glob[i] += p.second[i];}
            }
        }
    }

    // Worker thread function
    virtual void worker(int threadId, std::streampos start, std::streampos end) {
        // Map: tile id -> vector of lines (buffer)
        std::unordered_map<uint64_t,std::vector<std::string>> buffers;
        std::unordered_map<std::string,std::vector<int32_t>> localCounts;
        double localMinX = +INFINITY, localMaxX = -INFINITY;
        double localMinY = +INFINITY, localMaxY = -INFINITY;

        std::ifstream file(inFile);
        if (!file) {
            error("Thread %d: Error opening file input file", threadId);
        }
        file.seekg(start);
        std::string line;

        while (file.tellg() < end && std::getline(file, line)) {
            consumeLine(threadId, line,
                        buffers,
                        localMinX, localMaxX,
                        localMinY, localMaxY,
                        localCounts);
        }
        flushAll(threadId, buffers);
        mergeLocalStats(localMinX, localMaxX,
                        localMinY, localMaxY,
                        localCounts);
    }

    void streamingWorker(int threadId) {
        std::unordered_map<uint64_t,std::vector<std::string>> buffers;
        std::unordered_map<std::string,std::vector<int32_t>> localCounts;
        double localMinX = +INFINITY, localMaxX = -INFINITY;
        double localMinY = +INFINITY, localMaxY = -INFINITY;

        std::vector<std::string> batch;
        batch.reserve(batchSize);

        {
            std::lock_guard lk(readMutex);
            if (nskipped < nskip) { // Skip initial lines
                if (gz) {
                    char buf[1<<16];
                    while (nskipped < nskip && gzgets(gz, buf, sizeof(buf))) {
                        nskipped++;
                    }
                } else {
                    std::string line;
                    while (nskipped < nskip && std::getline(*inPtr, line)) {
                        nskipped++;
                    }
                }
            }
        }
        while (true) {
            batch.clear();
            { // —— fill one batch under lock ——
                std::lock_guard lk(readMutex);
                if (gz) {
                    for (int i = 0; i < batchSize; ++i) {
                        char buf[1<<16];
                        if (!gzgets(gz, buf, sizeof(buf))) break;
                        size_t len = strlen(buf);
                        if (len > 0 && buf[len-1] == '\n') {
                            buf[--len] = '\0';
                            if (len > 0 && buf[len-1] == '\r')
                                buf[--len] = '\0';
                            batch.emplace_back(buf);
                        } else {
                            // Continue reading
                            std::string line(buf, len);
                            while (true) {
                                if (!gzgets(gz, buf, sizeof(buf))) break;
                                size_t chunkLen = strlen(buf);
                                bool gotNL = chunkLen > 0 && buf[chunkLen-1] == '\n';
                                // append without the trailing '\n'
                                line.append(buf, gotNL ? chunkLen-1 : chunkLen);
                                if (gotNL) break;
                            }
                            // strip a final '\r' if present
                            if (!line.empty() && line.back() == '\r')
                                line.pop_back();
                            batch.push_back(std::move(line));
                        }
                    }
                } else {
                    for (int i = 0; i < batchSize; ++i) {
                        std::string line;
                        if (!std::getline(*inPtr, line)) break;
                        if (!line.empty() && line.back()=='\r')
                            line.pop_back();
                        batch.push_back(std::move(line));
                    }
                }
            }
            if (batch.empty()) break;
            for (auto &ln : batch) {
                consumeLine(threadId, ln,
                            buffers,
                            localMinX, localMaxX,
                            localMinY, localMaxY,
                            localCounts);
            }
            flushAll(threadId, buffers);
        }
        mergeLocalStats(localMinX, localMaxX,
                        localMinY, localMaxY,
                        localCounts);
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
        indexfile << "# xmin\t" << minX << "\n"
                  << "# xmax\t" << maxX << "\n"
                  << "# ymin\t" << minY << "\n"
                  << "# ymax\t" << maxY << "\n";
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
        std::string outFile = outPref + ".coord_range.tsv";
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
            std::string outFile = outPref + ".features.tsv";
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
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < ntokens) {
            error("Error parsing line: %s", line.c_str());
        }
        x = std::stod(tokens[icol_x]);
        y = std::stod(tokens[icol_y]);
        if (scaling) {
            x *= scale;
            y *= scale;
            tokens[icol_x] = fp_to_string(x, digits);
            tokens[icol_y] = fp_to_string(y, digits);
            line = join(tokens, "\t");
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
