#pragma once

#include <cmath>
#include <random>
#include <cassert>
#include <stdexcept>
#include "qgenlib/qgen_error.h"
#include "nanoflann.hpp"
#include "nanoflann_utils.h"
#include "dataunits.hpp"
#include "tilereader.hpp"
#include "hexgrid.h"
#include "utils.h"
#include "threads.hpp"
#include "lda.hpp"
#include "slda.hpp"

#include "Eigen/Dense"
#include "Eigen/Sparse"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

// to serialize to temporary files
#pragma pack(push, 1)
struct Record {
    int32_t x;
    int32_t y;
    uint32_t idx;
    uint32_t ct;
};
#pragma pack(pop)

struct lineParserLocal : public lineParser {
    size_t icol_val;
    lineParserLocal() {}
    lineParserLocal(size_t _ix, size_t _iy, size_t _iz, size_t _ival, std::string& _dfile) {
        icol_val = _ival;
        std::vector<int32_t> _ivals = { (int32_t) _ival};
        init(_ix, _iy, _iz, _ivals, _dfile);
    }
    void init2(size_t _ix, size_t _iy, size_t _iz, size_t _ival, std::string& _dfile) {
        icol_val = _ival;
        std::vector<int32_t> _ivals = { (int32_t) _ival};
        init(_ix, _iy, _iz, _ivals, _dfile);
    }

    int32_t parse(Record& rec, std::string& line) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < n_tokens) {
            return -2;
        }
        if (isFeatureDict) {
            auto it = featureDict.find(tokens[icol_feature]);
            if (it == featureDict.end()) {
                return -1;
            }
            rec.idx = it->second;
        } else {
            rec.idx = std::stoul(tokens[icol_feature]);
        }
        rec.x = std::stoi(tokens[icol_x]);
        rec.y = std::stoi(tokens[icol_y]);
        rec.ct = std::stoi(tokens[icol_val]);
        return rec.idx;
    }
};

struct TileData {
    std::unordered_map<uint32_t, std::vector<Record>> buffers; // local buffer to accumulate records to be written to temporary files
    std::vector<Record> pts; // original data points
    std::vector<int32_t> idxinternal; // indices for internal data points to output
    std::vector<int32_t> orgpts2pixel; // map from original points to indices of the pixels used in the model. -1 for not used
    std::vector<std::pair<double, double>> coords; // unique coordinates
    void clear() {
        buffers.clear();
        pts.clear();
        idxinternal.clear();
        orgpts2pixel.clear();
        coords.clear();
    }
};

// manage one temporary buffer
struct BoundaryBuffer {
    uint32_t key;
    std::string tmpFile;
    uint8_t nTiles;
    std::shared_ptr<std::mutex> mutex;
    BoundaryBuffer(uint32_t _key, const std::string& _tmpFile) : key(_key), tmpFile(_tmpFile), nTiles(0) {
        mutex = std::make_shared<std::mutex>();
        std::ofstream ofs(tmpFile, std::ios::binary);
        if (!ofs) {
            throw std::runtime_error("Error creating temporary file: " + tmpFile);
        }
        ofs.close();
    }
    bool finished() {
        if ((key & 0x1) && nTiles == 2) { // Vertical buffer
            return true;
        }
        if ((key & 0x1) == 0 && nTiles == 6) { // Horizontal buffer
            return true;
        }
    }
    void writeToFile(const std::vector<Record>& records) {
        std::lock_guard<std::mutex> lock(*mutex);
        std::ofstream ofs(tmpFile, std::ios::binary | std::ios::app);
        if (!ofs) {
            throw std::runtime_error("Error creating temporary file: " + tmpFile);
        }
        ofs.write(reinterpret_cast<const char*>(records.data()), records.size() * sizeof(Record));
        ofs.close();
        nTiles++;
    }
};

class Tiles2MinibatchBase {

public:

    Tiles2MinibatchBase(int nThreads, double r, const std::string& outputFile, const std::string& _tmpDir, TileReader& tileReader)
    : nThreads(nThreads), r(r), outputFile(outputFile), tmpDir(_tmpDir), tileReader(tileReader) {
        if (!createDirectory(tmpDir)) {
            throw std::runtime_error("Error creating temporary directory (or the existing directory is not empty): " + tmpDir);
        }
        if (tmpDir.back() != '/') {
            tmpDir.push_back('/');
        }
        mainOut.open(this->outputFile, std::ios::out);
        if (!mainOut) {
            error("Error opening main output file: %s", outputFile.c_str());
        } // Assume tileSize is provided by the TileReader.
        tileSize = tileReader.getTileSize();
    }

    ~Tiles2MinibatchBase() {
        if (mainOut.is_open()) {
            mainOut.close();
        }
    }

    virtual void run() = 0;

protected:

    int nThreads; // Number of worker threads
    int tileSize; // Tile size (square)
    double r;     // Processing radius (padding width)
    std::string outputFile, tmpDir;
    TileReader& tileReader;

    std::ofstream mainOut;
    std::mutex mainOutMutex;  // Protects writing to mainOut
    std::unordered_map<uint32_t, std::shared_ptr<BoundaryBuffer>> boundaryBuffers;
    std::mutex boundaryBuffersMapMutex; // Protects modifying boundaryBuffers
    ThreadSafeQueue<TileKey> tileQueue;
    ThreadSafeQueue<std::shared_ptr<BoundaryBuffer>> bufferQueue;
    std::vector<std::thread> workThreads;

    std::shared_ptr<BoundaryBuffer> getBoundaryBuffer(uint32_t key) {
        std::lock_guard<std::mutex> lock(boundaryBuffersMapMutex);
        auto it = boundaryBuffers.find(key);
        if (it == boundaryBuffers.end()) {
            std::string tmpFile = tmpDir + std::to_string(key);
            auto buffer = std::make_shared<BoundaryBuffer>(key, tmpFile);
            boundaryBuffers[key] = buffer;
            return buffer;
        }
        return it->second;
    }

    bool decodeTempFileKey(uint32_t key, int32_t& R, int32_t& C) {
        R = static_cast<int32_t>(key >> 16);
        C = static_cast<int32_t>((key >> 1) & 0x7FFF);
        return (key & 0x1) != 0;
    }

    uint32_t encodeTempFileKey(bool isVertical, int32_t R, int32_t C) {
        return (static_cast<uint32_t>(R) << 16) | ((static_cast<uint32_t>(C) & 0x7FFF) << 1) | static_cast<uint32_t>(isVertical);
    }

    int32_t pts2buffer(std::vector<uint32_t>& bufferidx, int32_t x0, int32_t y0, TileKey tile) {
        // convert to local coordinates
        int32_t x = x0 - tile.col * tileSize;
        int32_t y = y0 - tile.row * tileSize;
        bufferidx.clear();
        if (x > tileSize - 2 * r && tile.col < tileReader.maxcol) {
            bufferidx.push_back(encodeTempFileKey(true, tile.row, tile.col));
        }
        if (x < 2 * r && tile.col > tileReader.mincol) {
            bufferidx.push_back(encodeTempFileKey(true, tile.row, tile.col - 1));
        }
        if (y < 2 * r && tile.row > tileReader.minrow) {
            bufferidx.push_back(encodeTempFileKey(false, tile.row - 1, tile.col));
        }
        if (y > tileSize - 2 * r) {
            bufferidx.push_back(encodeTempFileKey(false, tile.row, tile.col));
        }
        if (x < r && tile.col > tileReader.mincol) {
            if (y < 2 * r && tile.row > tileReader.minrow) {
                bufferidx.push_back(encodeTempFileKey(false, tile.row - 1, tile.col - 1));
            }
            if (y > tileSize - 2 * r) {
                bufferidx.push_back(encodeTempFileKey(false, tile.row, tile.col - 1));
            }
        }
        if (x > tileSize - r) {
            if (y < 2 * r && tile.row > tileReader.minrow && tile.col < tileReader.maxcol) {
                bufferidx.push_back(encodeTempFileKey(false, tile.row - 1, tile.col + 1));
            }
            if (y > tileSize - 2 * r && tile.col < tileReader.maxcol) {
                bufferidx.push_back(encodeTempFileKey(false, tile.row, tile.col + 1));
            }
        }
        bool xTrue = x >= r && x < tileSize - r;
        bool yTrue = y >= r && y < tileSize - r;
        if (xTrue && yTrue) {
            return 1; // internal
        }
        // corners
        if (tile.row == tileReader.minrow && tile.col == tileReader.mincol) {
            return x < tileSize - r && y < tileSize - r;
        }
        if (tile.row == tileReader.minrow && tile.col == tileReader.maxcol) {
            return x >= r && y < tileSize - r;
        }
        if (tile.row == tileReader.maxrow && tile.col == tileReader.mincol) {
            return x < tileSize - r && y >= r;
        }
        if (tile.row == tileReader.maxrow && tile.col == tileReader.maxcol) {
            return x >= r && y >= r;
        }
        // non-corner edges
        if (tile.row == tileReader.minrow) {
            return xTrue && y < tileSize - r;
        }
        if (tile.col == tileReader.mincol) {
            return yTrue && x < tileSize - r;
        }
        if (tile.row == tileReader.maxrow) {
            return xTrue && y >= r;
        }
        if (tile.col == tileReader.maxcol) {
            return yTrue && x >= r;
        }
        return 0;
    }

    bool isInternalToBuffer(int32_t x, int32_t y, uint32_t bufferId) {
        int32_t bufRow, bufCol;
        bool isVertical = decodeTempFileKey(bufferId, bufRow, bufCol);
        if (isVertical) {
            double x_min = (bufCol + 1.) * tileSize - r;
            double x_max = (bufCol + 1.) * tileSize + r;
            double y_min = bufRow * tileSize + r;
            double y_max = bufRow * tileSize + tileSize - r;
            if (bufRow == tileReader.minrow) {
                return (x > x_min && x < x_max && y < y_max);
            }
            return (x > x_min && x < x_max && y > y_min && y < y_max);
        } else {
            double x_min = bufCol * tileSize;
            double x_max = bufCol * tileSize + tileSize;
            double y_min = (bufRow + 1) * tileSize - r;
            double y_max = (bufRow + 1) * tileSize + r;
            return (x >= x_min && x < x_max && y > y_min && y < y_max);
        }
    }

    virtual void tileWorker(int threadId) = 0;
    virtual void boundaryWorker(int threadId) = 0;
};

class Tiles2Minibatch : public Tiles2MinibatchBase {

public:
    Tiles2Minibatch(int nThreads, double r,
        const std::string& outputFile, const std::string& _tmpDir,
        LatentDirichletAllocation& _lda,
        TileReader& tileReader, lineParserLocal& lineParser,
        HexGrid& hexGrid, int32_t nMoves,
        unsigned int seed = std::random_device{}(),
        double c = 20, double h = 0.7, double res = 1, bool outorg = true, int32_t M = 0, int32_t N = 0, int32_t k = 3, int32_t verbose = 0, int32_t debug = 0) :
        Tiles2MinibatchBase(nThreads, r + hexGrid.size, outputFile, _tmpDir, tileReader), distR(r), lda(_lda), lineParser(lineParser), hexGrid(hexGrid), nMoves(nMoves), anchorMinCount(c), pixelResolution(res), outpurOriginalData(outorg), M_(M), topk_(k), debug_(debug) {
        if (pixelResolution <= 0) {
            pixelResolution = 1;
        }
        weighted = lineParser.weighted;
        M_ = lda.get_n_features();
        if (lineParser.isFeatureDict) {
            assert((M_ == lineParser.featureDict.size()) && "Feature number does not match");
            featureNames.resize(M_);
            for (const auto& entry : lineParser.featureDict) {
                featureNames[entry.second] = entry.first;
            }
        } else if (lineParser.weighted) {
            assert(M_ == lineParser.weights.size() && "Feature number does not match");
        }
        lda.set_nthreads(1); // because we parallelize by tile
        K_ = lda.get_n_topics();
        distNu = std::log(0.5) / std::log(h);
        if (N <= 0) {
            N = lda.get_N_global() * 100;
        }
        slda.init(K_, M_, N, seed);
        slda.init_global_parameter(lda.get_model());
        slda.verbose_ = verbose;
        if (featureNames.size() == 0) {
            featureNames.resize(M_);
            for (int32_t i = 0; i < M_; ++i) {
                featureNames[i] = std::to_string(i);
            }
        }
        notice("Initialized Tiles2Minibatch");
    }

    void run() override {
        // Phase 1: Process tiles
        notice("Phase 1 Launching %d worker threads", nThreads);
        for (int i = 0; i < nThreads; ++i) {
            workThreads.push_back(std::thread(&Tiles2Minibatch::tileWorker, this, i));
        }
        std::vector<TileKey> tileList;
        tileReader.getTileList(tileList);
        for (const auto &tile : tileList) {
            tileQueue.push(tile);
        }
        tileQueue.set_done();
        for (auto &t : workThreads) {
            t.join();
        }
        workThreads.clear();
        // Phase 2: Process boundary buffers
        notice("Phase 2 Launching %d worker threads", nThreads);
        { // Enqueue all boundary buffers from the global map.
            std::lock_guard<std::mutex> lock(boundaryBuffersMapMutex);
            for (const auto &entry : boundaryBuffers) {
                bufferQueue.push(entry.second);
            }
            bufferQueue.set_done();
        }
        for (int i = 0; i < nThreads; ++i) {
            workThreads.push_back(std::thread(&Tiles2Minibatch::boundaryWorker, this, i));
        }
        for (auto &t : workThreads) {
            t.join();
        }
    }

    void setFeatureNames(const std::vector<std::string>& names) {
        assert(names.size() == M_);
        featureNames = names;
    }

private:
    int32_t debug_;
    int32_t M_, K_, topk_;
    bool weighted, outpurOriginalData;
    LatentDirichletAllocation& lda;
    OnlineSLDA slda;
    lineParserLocal& lineParser;
    HexGrid hexGrid;
    int32_t nMoves;
    double anchorMinCount, distNu, distR;
    float pixelResolution;
    std::vector<std::string> featureNames;

    // Parse pixels from one tile
    int32_t parseOneTile(TileData& tileData, TileKey tile);
    // Parse a binary temporary file (written by BoundaryBuffer)
    int32_t parseBoundaryFile(TileData& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);

    int32_t initAnchors(TileData& tileData, PointCloud<double>& anchors, Minibatch& minibatch);
    int32_t makeMinibatch(TileData& tileData, PointCloud<double>& anchors, Minibatch& minibatch);

    // Write the results of internal points
    int32_t outputOriginalDataWithPixelResult(const TileData& tileData, const MatrixXd& topVals, const Eigen::MatrixXi& topIds);
    int32_t outputPixelResult(const TileData& tileData, const MatrixXd& topVals, const Eigen::MatrixXi& topIds);

    void processTile(TileData &tileData, int threadId=0);

    void tileWorker(int threadId) override {
        TileKey tile;
        while (tileQueue.pop(tile)) {
            TileData tileData;
            int32_t ret = parseOneTile(tileData, tile);
            notice("Thread %d read tile (%d, %d) with %d internal pixels", threadId, tile.row, tile.col, ret);
            if (ret <= 10) continue;
            processTile(tileData, threadId);
        }
    }

    void boundaryWorker(int threadId) override {
        std::shared_ptr<BoundaryBuffer> bufferPtr;
        while (bufferQueue.pop(bufferPtr)) {
            TileData tileData;
            int32_t ret = parseBoundaryFile(tileData, bufferPtr);
            notice("Thread %d read boundary buffer (%d, %d, %d) with %d internal pixels", threadId, (bufferPtr->key & 0x1), bufferPtr->key >> 16, (bufferPtr->key >> 1) & 0x7FFF, ret);
            if (ret >10) {
                processTile(tileData, threadId);
            }
            std::remove(bufferPtr->tmpFile.c_str());
        }
    }

};
