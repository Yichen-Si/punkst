#pragma once

#include <cmath>
#include <random>
#include <cassert>
#include <stdexcept>
#include <atomic>
#include "error.hpp"
#include "json.hpp"
#include "nanoflann.hpp"
#include "nanoflann_utils.h"
#include <opencv2/imgproc.hpp>
#include "dataunits.hpp"
#include "tilereader.hpp"
#include "hexgrid.h"
#include "utils.h"
#include "threads.hpp"
#include "lda.hpp"
#include "slda.hpp"

#include "Eigen/Dense"
#include "Eigen/Sparse"
using Eigen::MatrixXf;

enum class FieldType : uint8_t { INT32, FLOAT, STRING };
struct FieldDef {
    FieldType  type;
    size_t     size;    // STRING: fixed byte length; INT32/FLOAT: sizeof()
    size_t     offset;  // filled in after we know all fields
};

template<typename T>
struct TileData {
    float xmin, xmax, ymin, ymax;
    std::unordered_map<uint32_t, std::vector<RecordT<T>>> buffers; // local buffer to accumulate records to be written to temporary files
    std::unordered_map<uint32_t, std::vector<RecordExtendedT<T>>> extBuffers;
    std::vector<RecordT<T>> pts; // original data points
    std::vector<RecordExtendedT<T>> extPts; // original data points with extended info fields
    std::vector<int32_t> idxinternal; // indices for internal data points to output
    std::vector<int32_t> orgpts2pixel; // map from original points to indices of the pixels used in the model. -1 for not used
    std::vector<std::pair<double, double>> coords; // unique coordinates
    void clear() {
        buffers.clear();
        pts.clear();
        extPts.clear();
        idxinternal.clear();
        orgpts2pixel.clear();
        coords.clear();
    }
    void setBounds(float _xmin, float _xmax, float _ymin, float _ymax) {
        xmin = _xmin;
        xmax = _xmax;
        ymin = _ymin;
        ymax = _ymax;
    }
};

// manage one temporary buffer
struct BoundaryBuffer {
    uint32_t key; // row|col|isVertical
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
        return false;
    }

    template<typename T>
    void writeToFile(const std::vector<RecordT<T>>& records) {
        std::lock_guard<std::mutex> lock(*mutex);
        std::ofstream ofs(tmpFile, std::ios::binary | std::ios::app);
        if (!ofs) {
            error("%s: error opening temporary file %s", __FUNCTION__, tmpFile.c_str());
        }
        ofs.write(reinterpret_cast<const char*>(records.data()), records.size() * sizeof(RecordT<T>));
        ofs.close();
        nTiles++;
    }

    template<typename T>
    void writeToFileExtended(
        const std::vector<RecordExtendedT<T>>& recs,
        const std::vector<FieldDef>&           schema,
        size_t                                 recordSize
    ) {
        if (recs.empty()) return;
        std::lock_guard<std::mutex> lock(*mutex);
        // pack into one big byte‐array
        std::vector<uint8_t> buf;
        buf.reserve(recs.size() * recordSize);
        for (auto &r : recs) {
            size_t baseOff = buf.size();
            buf.resize(buf.size() + recordSize);
            // 1) the base blob
            std::memcpy(buf.data() + baseOff, &r.recBase, sizeof(r.recBase));
            // 2) each extra field by schema
            uint32_t i_int = 0, i_flt = 0, i_str = 0;
            for (size_t fi = 0; fi < schema.size(); ++fi) {
                auto const &f = schema[fi];
                auto dst = buf.data() + baseOff + f.offset;
                switch (f.type) {
                    case FieldType::INT32: {
                        int32_t v = r.intvals[i_int++];
                        std::memcpy(dst, &v, sizeof(v));
                    } break;
                    case FieldType::FLOAT: {
                        float v = r.floatvals[i_flt++];
                        std::memcpy(dst, &v, sizeof(v));
                    } break;
                    case FieldType::STRING: {
                        auto &s = r.strvals[i_str++];
                        std::memcpy(dst, s.data(), std::min(s.size(), f.size));
                        if (s.size() < f.size) {
                            std::memset(dst + s.size(), 0, f.size - s.size());
                        }
                    } break;
                }
            }
        }
        std::ofstream ofs(tmpFile, std::ios::binary | std::ios::app);
        if (!ofs) {
            error("%s: error opening temporary file %s", __FUNCTION__, tmpFile.c_str());
        }
        ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
        ofs.close();
        nTiles++;
    }
};

/* Implement the logic of processing tiles while resolving boundary issues */
class Tiles2MinibatchBase {

public:

    Tiles2MinibatchBase(int nThreads, double r, const std::string& _outPref, const std::string& _tmpDirPath, TileReader& tileReader)
    : nThreads(nThreads), r(r), outPref(_outPref), tmpDir(_tmpDirPath), tileReader(tileReader) {
        std::string outputFile = outPref + ".tsv";
        mainOut.open(outputFile, std::ios::out);
        if (!mainOut) {
            error("Error opening main output file: %s", outputFile.c_str());
        } // Assume tileSize is provided by the TileReader.
        mainOut.close();
        tileSize = tileReader.getTileSize();
        notice("Created temporary directory: %s", tmpDir.path.string().c_str());
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
    std::string outPref;
    ScopedTempDir tmpDir;
    TileReader& tileReader;

    std::ofstream mainOut;
    std::mutex mainOutMutex;  // Protects writing to mainOut
    std::map<uint32_t, std::shared_ptr<BoundaryBuffer>> boundaryBuffers;
    std::mutex boundaryBuffersMapMutex; // Protects modifying boundaryBuffers
    ThreadSafeQueue<std::pair<TileKey, int32_t> > tileQueue;
    ThreadSafeQueue<std::pair<std::shared_ptr<BoundaryBuffer>, int32_t>> bufferQueue;
    std::vector<std::thread> workThreads;

    std::shared_ptr<BoundaryBuffer> getBoundaryBuffer(uint32_t key) {
        std::lock_guard<std::mutex> lock(boundaryBuffersMapMutex);
        auto it = boundaryBuffers.find(key);
        if (it == boundaryBuffers.end()) {
            auto tmpFile = tmpDir.path / std::to_string(key);
            auto buffer = std::make_shared<BoundaryBuffer>(key, tmpFile.string());
            boundaryBuffers[key] = buffer;
            return buffer;
        }
        return it->second;
    }

    bool decodeTempFileKey(uint32_t key, int32_t& R, int32_t& C) {
        R = static_cast<int16_t>( key >> 16 );
        uint32_t rawC = (key >> 1) & 0x7FFF;
        if (rawC & 0x4000) {
            C = static_cast<int32_t>(rawC | 0xFFFF8000u);
        } else {
            C = static_cast<int32_t>(rawC);
        }
        return (key & 0x1) != 0;
    }

    uint32_t encodeTempFileKey(bool isVertical, int32_t R, int32_t C) {
        return (static_cast<uint32_t>(R) << 16) | ((static_cast<uint32_t>(C) & 0x7FFF) << 1) | static_cast<uint32_t>(isVertical);
    }

    // given (x, y) and its tile, compute all buffers it walls into
    // and whether it is "internal" to the tile
    template<typename T>
    int32_t pt2buffer(std::vector<uint32_t>& bufferidx, T x0, T y0, TileKey tile) {
        // convert to local coordinates
        T x = x0 - tile.col * tileSize;
        T y = y0 - tile.row * tileSize;
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

    template<typename T>
    void pt2tile(T x, T y, TileKey &tile) const {
        tile.row = static_cast<int32_t>(std::floor(y / tileSize));
        tile.col = static_cast<int32_t>(std::floor(x / tileSize));
    }
    template<typename T>
    void tile2bound(TileKey &tile, T& xmin, T& xmax, T& ymin, T& ymax) const {
        xmin = static_cast<T>(tile.col * tileSize);
        xmax = static_cast<T>((tile.col + 1) * tileSize);
        ymin = static_cast<T>(tile.row * tileSize);
        ymax = static_cast<T>((tile.row + 1) * tileSize);
    }

    template<typename T>
    void buffer2bound(bool isVertical, int32_t& bufRow, int32_t& bufCol, T& xmin, T& xmax, T& ymin, T& ymax) {
        if (isVertical) {
            xmin = (bufCol + 1.) * tileSize - 2 * r;
            xmax = (bufCol + 1.) * tileSize + 2 * r;
            ymin = bufRow * tileSize;
            ymax = bufRow * tileSize + tileSize;
        } else {
            xmin = bufCol * tileSize - r;
            xmax = bufCol * tileSize + tileSize + r;
            ymin = (bufRow + 1) * tileSize - 2 * r;
            ymax = (bufRow + 1) * tileSize + 2 * r;
        }
    }
    template<typename T>
    void bufferId2bound(uint32_t bufferId, T& xmin, T& xmax, T& ymin, T& ymax) {
        int32_t bufRow, bufCol;
        bool isVertical = decodeTempFileKey(bufferId, bufRow, bufCol);
        buffer2bound(isVertical, bufRow, bufCol, xmin, xmax, ymin, ymax);
    }
    template<typename T>
    bool isInternalToBuffer(T x, T y, uint32_t bufferId) {
        int32_t bufRow, bufCol;
        bool isVertical = decodeTempFileKey(bufferId, bufRow, bufCol);
        if (isVertical) {
            float x_min = (bufCol + 1.) * tileSize - r;
            float x_max = (bufCol + 1.) * tileSize + r;
            float y_min = bufRow * tileSize + r;
            float y_max = bufRow * tileSize + tileSize - r;
            if (bufRow == tileReader.minrow) {
                return (x > x_min && x < x_max && y < y_max);
            }
            return (x > x_min && x < x_max && y > y_min && y < y_max);
        } else {
            float x_min = bufCol * tileSize;
            float x_max = bufCol * tileSize + tileSize;
            float y_min = (bufRow + 1) * tileSize - r;
            float y_max = (bufRow + 1) * tileSize + r;
            return (x >= x_min && x < x_max && y > y_min && y < y_max);
        }
    }

    virtual void tileWorker(int threadId) = 0;
    virtual void boundaryWorker(int threadId) = 0;
};

template<typename T>
class Tiles2Minibatch : public Tiles2MinibatchBase {

public:
    Tiles2Minibatch(int nThreads, double r,
        const std::string& _outPref, const std::string& _tmpDir,
        LatentDirichletAllocation& _lda,
        TileReader& tileReader, lineParserUnival& lineParser,
        HexGrid& hexGrid, int32_t nMoves,
        unsigned int seed = std::random_device{}(),
        double c = 20, double h = 0.7, double res = 1,
        int32_t M = 0, int32_t N = 0, int32_t k = 3,
        int32_t verbose = 0, int32_t debug = 0) :
        Tiles2MinibatchBase(nThreads, r + hexGrid.size, _outPref, _tmpDir, tileReader), distR(r), lda(_lda), lineParser(lineParser), hexGrid(hexGrid), nMoves(nMoves), anchorMinCount(c), pixelResolution(res), M_(M), topk_(k), debug_(debug),
        outputOriginalData(false), useExtended_(lineParser.isExtended),
        useTicketSystem(false), currentTicket(0) {
        // check type consistency
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            assert(tileReader.getCoordType() == CoordType::FLOAT && "Template type does not match with TileReader coordinate type");
        } else if constexpr (std::is_same_v<T, int32_t>) {
            assert(tileReader.getCoordType() == CoordType::INTEGER && "Template type does not match with TileReader coordinate type");
        } else {
            // static_assert(false, "Unsupported coordinate type"); // need newer gcc
            error("Unsupported coordinate type");
        }
        if (pixelResolution <= 0) {
            pixelResolution = 1;
        }
        if (useExtended_) {
            setExtendedSchema();
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
        pseudobulk = MatrixXf::Zero(K_, M_);
        slda.init(K_, M_, N, seed);
        slda.init_global_parameter(lda.get_model());
        slda.verbose_ = verbose;
        slda.debug_ = debug;
        if (featureNames.size() == 0) {
            featureNames.resize(M_);
            for (int32_t i = 0; i < M_; ++i) {
                featureNames[i] = std::to_string(i);
            }
        }
        if (debug_ > 0) {
            std::cout << "Check model initialization\n" << std::fixed << std::setprecision(2);
            const auto& lambda = slda.get_lambda();
            const auto& Elog_beta = slda.get_Elog_beta();
            for (int32_t i = 0; i < std::min(3, K_) ; ++i) {
                std::cout << "\tLambda " << i << ": ";
                for (int32_t j = 0; j < std::min(5, M_); ++j) {
                    std::cout << lambda(i, j) << " ";
                }
                std::cout << "\n\tElog_beta: ";
                for (int32_t j = 0; j < std::min(5, M_); ++j) {
                    std::cout << Elog_beta(i, j) << " ";
                }
                std::cout << "\n";
            }
        }

        notice("Initialized Tiles2Minibatch");
    }

    void setFeatureNames(const std::vector<std::string>& names) {
        assert(names.size() == M_);
        featureNames = names;
    }
    void setLloydIter(int32_t nIter) {
        nLloydIter = nIter;
    }
    void setOutputCoordDigits(int32_t digits) {
        floatCoordDigits = digits;
    }
    void setOutputProbDigits(int32_t digits) {
        probDigits = digits;
    }
    void setOutputOptions(bool includeOrg, bool useTicket) {
        outputOriginalData = includeOrg;
        useTicketSystem = useTicket;
    }

    int32_t loadAnchors(const std::string& anchorFile);

    void run() override {
        setupOutput();
        // Phase 1: Process tiles
        notice("Phase 1 Launching %d worker threads", nThreads);
        for (int i = 0; i < nThreads; ++i) {
            workThreads.push_back(std::thread(&Tiles2Minibatch::tileWorker, this, i));
        }
        std::vector<TileKey> tileList;
        tileReader.getTileList(tileList);
        std::sort(tileList.begin(), tileList.end());
        currentTicket.store(0);
        int32_t ticket = 0;
        // Enqueue all tiles to the queue in a deterministic order
        for (const auto &tile : tileList) {
            tileQueue.push(std::make_pair(tile, ticket++));
            if (debug_ > 0 && ticket >= debug_) {
                break;
            }
        }
        tileQueue.set_done();
        for (auto &t : workThreads) {
            t.join();
        }
        workThreads.clear();

        // Phase 2: Process boundary buffers
        notice("Phase 2 Launching %d worker threads", nThreads);
        std::vector<std::shared_ptr<BoundaryBuffer>> buffers;
        buffers.reserve(boundaryBuffers.size());
        for (auto &kv : boundaryBuffers) {
            buffers.push_back(kv.second);
        }
        std::sort(buffers.begin(), buffers.end(),
            [](auto const &A, auto const &B){
                return A->key < B->key;
        });
        currentTicket.store(0);
        ticket = 0;
        // Enqueue all boundary buffers from the global map.
        for (auto &bufferPtr : buffers) {
            bufferQueue.push(std::make_pair(bufferPtr, ticket++));
        }
        bufferQueue.set_done();
        for (int i = 0; i < nThreads; ++i) {
            workThreads.push_back(std::thread(&Tiles2Minibatch::boundaryWorker, this, i));
        }
        for (auto &t : workThreads) {
            t.join();
        }
        mainOut.close();
        indexOut.close();
        writeHeaderToJson();
        writePseudobulkToTsv();
    }

protected:
    int32_t debug_;
    int32_t M_, K_, topk_;
    bool weighted, outputOriginalData;
    bool useTicketSystem;
    std::atomic<int> currentTicket;
    std::mutex ticketMutex;
    std::condition_variable ticketCondition;
    bool useExtended_;
    size_t recordSize_ = 0;
    std::vector<FieldDef> schema_;
    LatentDirichletAllocation& lda;
    OnlineSLDA slda;
    lineParserUnival& lineParser;
    HexGrid hexGrid;
    int32_t nMoves;
    double anchorMinCount, distNu, distR;
    float pixelResolution;
    double eps_;
    std::vector<std::string> featureNames;
    using vec2f_t = std::vector<std::vector<float>>;
    std::unordered_map<TileKey, vec2f_t, TileKeyHash> fixedAnchorForTile; // we may need more than one set of pre-defined anchors in the future
    std::unordered_map<uint32_t, vec2f_t> fixedAnchorForBoundary;
    int32_t nLloydIter = 1;
    std::ofstream indexOut;
    size_t outputSize = 0;
    int32_t floatCoordDigits = 4, probDigits = 4;
    MatrixXf pseudobulk; // K x M
    std::mutex pseudobulkMutex; // Protects pseudobulk

    // Parse pixels from one tile
    int32_t parseOneTile(TileData<T>& tileData, TileKey tile);
    // Parse a binary temporary file (written by BoundaryBuffer)
    int32_t parseBoundaryFile(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);
    int32_t parseBoundaryFileExtended(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);

    int32_t initAnchorsHybrid(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch, const vec2f_t* fixedAnchors = nullptr);
    int32_t initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch);
    int32_t makeMinibatch(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch);

    // Write the results of internal points
    int32_t outputOriginalDataWithPixelResult(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds);
    int32_t outputPixelResult(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds);

    // Process one tile or one boundary buffer
    void processTile(TileData<T> &tileData, int threadId=0, int ticket = 0, vec2f_t* anchorPtr = nullptr);

    // write output column info to a json file
    void writeHeaderToJson();
    // write posterior pseudobulk to a tsv file
    void writePseudobulkToTsv();

    // Call *before* run()
    void setExtendedSchema() {
        if (!lineParser.isExtended) {
            useExtended_ = false;
            return;
        }
        useExtended_ = true;
        schema_.clear();
        size_t offset = sizeof(RecordT<T>);
        size_t n_ints = lineParser.icol_ints.size();
        size_t n_floats = lineParser.icol_floats.size();
        size_t n_strs = lineParser.icol_strs.size();
        for (size_t i = 0; i < n_ints; ++i) {
            schema_.push_back({FieldType::INT32, sizeof(int32_t), 0});
        }
        for (size_t i = 0; i < n_floats; ++i) {
            schema_.push_back({FieldType::FLOAT, sizeof(float), 0});
        }
        for (size_t i = 0; i < n_strs; ++i) {
            schema_.push_back({FieldType::STRING, lineParser.str_lens[i], 0});
        }
        // now fix up offsets and total size
        for (auto &f : schema_) {
            f.offset = offset;
            offset  += f.size;
        }
        recordSize_ = offset;
    }

    void tileWorker(int threadId) override {
        std::pair<TileKey, int32_t> tileTicket;
        TileKey tile;
        int32_t ticket;
        while (tileQueue.pop(tileTicket)) {
            tile = tileTicket.first;
            ticket = tileTicket.second;
            TileData<T> tileData;
            int32_t ret = parseOneTile(tileData, tile);
            notice("%s: Thread %d (ticket %d) read tile (%d, %d) with %d internal pixels", __FUNCTION__, threadId, ticket, tile.row, tile.col, ret);
            if (ret <= 10) {
                waitToAdvanceTicket(ticket);
                continue;
            }
            vec2f_t* anchorPtr = nullptr;
            if (fixedAnchorForTile.find(tile) != fixedAnchorForTile.end()) {
                anchorPtr = &fixedAnchorForTile[tile];
            }
            processTile(tileData, threadId, ticket, anchorPtr);
        }
    }

    void boundaryWorker(int threadId) override {
        std::pair<std::shared_ptr<BoundaryBuffer>, int32_t> bufferTicket;
        std::shared_ptr<BoundaryBuffer> bufferPtr;
        int32_t ticket;
        while (bufferQueue.pop(bufferTicket)) {
            bufferPtr = bufferTicket.first;
            ticket = bufferTicket.second;
            TileData<T> tileData;
            int32_t ret;
            if (useExtended_) {
                ret = parseBoundaryFileExtended(tileData, bufferPtr);
            } else {
                ret = parseBoundaryFile(tileData, bufferPtr);
            }
            notice("%s: Thread %d (ticket %d) read boundary buffer (%d) with %d internal pixels", __FUNCTION__, threadId, ticket, bufferPtr->key, ret);
            if (ret <= 10) {
                std::remove(bufferPtr->tmpFile.c_str());
                waitToAdvanceTicket(ticket);
                continue;
            }
            vec2f_t* anchorPtr = nullptr;
            if (fixedAnchorForBoundary.find(bufferPtr->key) != fixedAnchorForBoundary.end()) {
                anchorPtr = &fixedAnchorForBoundary[bufferPtr->key];
            }
            processTile(tileData, threadId, ticket, anchorPtr);
            std::remove(bufferPtr->tmpFile.c_str());
        }
    }

    void setupOutput() {
        std::string outputFile = outPref + ".tsv";
        mainOut.open(outputFile, std::ios::out);
        if (!mainOut) {
            error("Error opening main output file: %s", outputFile.c_str());
        }
        // write header
        if (outputOriginalData) {
            mainOut << "#x\ty\tfeature\tct";
        } else {
            mainOut << "#x\ty";
        }
        for (int32_t i = 0; i < topk_; ++i) {
            mainOut << "\tK" << i+1;
        }
        for (int32_t i = 0; i < topk_; ++i) {
            mainOut << "\tP" << i+1;
        }
        if (useExtended_) {
            for (const auto &v : lineParser.name_ints) {
                mainOut << "\t" << v;
            }
            for (const auto &v : lineParser.name_floats) {
                mainOut << "\t" << v;
            }
            for (const auto &v : lineParser.name_strs) {
                mainOut << "\t" << v;
            }
        }
        mainOut << "\n";
        std::string indexFile = outPref + ".index";
        indexOut.open(indexFile, std::ios::out | std::ios::binary);
        if (!indexOut) {
            error("Error opening index output file: %s", indexFile.c_str());
        }
        indexOut << std::fixed << std::setprecision(floatCoordDigits);
    }

    void advanceTicket() {
        if (!useTicketSystem) {
            return;
        }
        currentTicket.fetch_add(1, std::memory_order_release);
        ticketCondition.notify_all();
    }
    void waitToAdvanceTicket(int ticket) {
        if (!useTicketSystem) {
            return;
        }
        std::unique_lock<std::mutex> lock(ticketMutex);
        ticketCondition.wait(lock, [this, ticket]() {
            return ticket == currentTicket.load(std::memory_order_acquire);
        });
        advanceTicket();
    }

};
