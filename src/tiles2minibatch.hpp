#pragma once

#include <cmath>
#include <random>
#include <cassert>
#include <stdexcept>
#include <atomic>
#include <tuple>
#include <variant>
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

struct IBoundaryStorage {
    virtual ~IBoundaryStorage() = default;
};

template<typename T>
struct InMemoryStorageStandard : public IBoundaryStorage {
    std::vector<RecordT<T>> data;
};

template<typename T>
struct InMemoryStorageExtended : public IBoundaryStorage {
    std::vector<RecordExtendedT<T>> dataExtended;
};

// manage one temporary buffer
struct BoundaryBuffer {
    uint32_t key; // row|col|isVertical
    uint8_t nTiles;
    std::shared_ptr<std::mutex> mutex;
    // either a temporary file path or an in-memory storage
    std::variant<std::string, std::unique_ptr<IBoundaryStorage>> storage;

    BoundaryBuffer(uint32_t _key,
        std::optional<std::reference_wrapper<std::string>> tmpFilePtr = std::nullopt) : key(_key), nTiles(0) {
        mutex = std::make_shared<std::mutex>();
        if (tmpFilePtr && !(tmpFilePtr->get()).empty()) {
            storage = tmpFilePtr->get();
            std::ofstream ofs(tmpFilePtr->get(), std::ios::binary);
            if (!ofs) {
                throw std::runtime_error("Error creating temporary file: " + tmpFilePtr->get());
            }
            ofs.close();
        } else {
            storage = std::unique_ptr<IBoundaryStorage>(nullptr);
        }
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
    void addRecords(const std::vector<RecordT<T>>& recs) {
        if (recs.empty()) return;
        std::lock_guard<std::mutex> lock(*mutex);

        if (auto* storagePtr = std::get_if<std::unique_ptr<IBoundaryStorage>>(&storage)) {
            // In-memory
            if (!*storagePtr) { // First write, create the correct storage object
                *storagePtr = std::make_unique<InMemoryStorageStandard<T>>();
            }
            // Cast to the concrete type and append data
            if (auto* memStore = dynamic_cast<InMemoryStorageStandard<T>*>(storagePtr->get())) {
                memStore->data.insert(memStore->data.end(), recs.begin(), recs.end());
            } else {
                throw std::runtime_error("Mismatched storage type in BoundaryBuffer::addRecords");
            }

        } else if (auto* tmpFile = std::get_if<std::string>(&storage)) {
            std::ofstream ofs(*tmpFile, std::ios::binary | std::ios::app);
            if (!ofs) {
                error("%s: error opening temporary file %s", __FUNCTION__, tmpFile->c_str());
            }
            ofs.write(reinterpret_cast<const char*>(recs.data()), recs.size() * sizeof(RecordT<T>));
            ofs.close();
        }
        nTiles++;
    }

    template<typename T>
    void addRecordsExtended(
        const std::vector<RecordExtendedT<T>>& recs,
        const std::vector<FieldDef>&           schema,
        size_t                                 recordSize
    ) {
        if (recs.empty()) return;
        std::lock_guard<std::mutex> lock(*mutex);

        if (auto* storagePtr = std::get_if<std::unique_ptr<IBoundaryStorage>>(&storage)) {
            // In-memory
            if (!*storagePtr) { // First write, create the correct storage object
                *storagePtr = std::make_unique<InMemoryStorageExtended<T>>();
            }
            // Cast to the concrete type and append data
            if (auto* memStore = dynamic_cast<InMemoryStorageExtended<T>*>(storagePtr->get())) {
                memStore->dataExtended.insert(memStore->dataExtended.end(), recs.begin(), recs.end());
            } else {
                throw std::runtime_error("Mismatched storage type in BoundaryBuffer::addRecords (Extended)");
            }
        } else if (auto* filePath = std::get_if<std::string>(&storage)) {
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
            std::ofstream ofs(*filePath, std::ios::binary | std::ios::app);
            if (!ofs) {
                error("%s: error opening temporary file %s", __FUNCTION__, filePath->c_str());
            }
            ofs.write(reinterpret_cast<const char*>(buf.data()), buf.size());
            ofs.close();
        }
        nTiles++;
    }
};

/* Implement the logic of processing tiles while resolving boundary issues */
class Tiles2MinibatchBase {

public:

    Tiles2MinibatchBase(int nThreads, double r, TileReader& tileReader, const std::string& _outPref, const std::string* opt = nullptr)
    : nThreads(nThreads), r(r), tileReader(tileReader), outPref(_outPref) {
        std::string outputFile = outPref + ".tsv";
        mainOut.open(outputFile, std::ios::out);
        if (!mainOut) {
            error("Error opening main output file: %s", outputFile.c_str());
        } // Assume tileSize is provided by the TileReader.
        mainOut.close();
        tileSize = tileReader.getTileSize();
        if (opt && !(*opt).empty()) {
            useMemoryBuffer_ = false;
            tmpDir.init(*opt);
            notice("Created temporary directory: %s", tmpDir.path.string().c_str());
        } else {
            useMemoryBuffer_ = true;
        }
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
    TileReader& tileReader;
    ScopedTempDir tmpDir;
    bool useMemoryBuffer_;

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
            std::string tmpFile;
            if (!useMemoryBuffer_) {
                tmpFile = (tmpDir.path / std::to_string(key)).string();
            }
            auto buffer = std::make_shared<BoundaryBuffer>(key, tmpFile);
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
        Tiles2MinibatchBase(nThreads, r + hexGrid.size, tileReader, _outPref, &_tmpDir), distR(r), lda(_lda), lineParser(lineParser), hexGrid(hexGrid), nMoves(nMoves), anchorMinCount(c), pixelResolution(res), M_(M), topk_(k), debug_(debug),
        outputOriginalData(false), useExtended_(lineParser.isExtended),
        resultQueue(static_cast<size_t>(std::max(1, nThreads))),
        useTicketSystem(false) {
        // check type consistency
        if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
            assert(tileReader.getCoordType() == CoordType::FLOAT && "Template type does not match with TileReader coordinate type");
        } else if constexpr (std::is_same_v<T, int32_t>) {
            assert(tileReader.getCoordType() == CoordType::INTEGER && "Template type does not match with TileReader coordinate type");
        } else {
            error("%s: Unsupported coordinate type", __func__);
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

        std::thread writer(&Tiles2Minibatch::writerWorker, this);

        for (int i = 0; i < nThreads; ++i) {
            workThreads.push_back(std::thread(&Tiles2Minibatch::tileWorker, this, i));
        }
        std::vector<TileKey> tileList;
        tileReader.getTileList(tileList);
        std::sort(tileList.begin(), tileList.end());
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

        resultQueue.set_done();
        writer.join();
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

    // buffer output results for one tile
    struct ProcessedResult {
        int32_t ticket;
        float xmin, xmax, ymin, ymax;
        std::vector<std::string> outputLines;
        uint32_t npts;
        ProcessedResult(int32_t t = 0, float _xmin = 0, float _xmax = 0, float _ymin = 0, float _ymax = 0)
        : ticket(t), npts(0), xmin(_xmin), xmax(_xmax), ymin(_ymin), ymax(_ymax) {}
        bool operator>(const ProcessedResult& other) const {
            return ticket > other.ticket;
        }
    };
    ThreadSafeQueue<ProcessedResult> resultQueue;

    // Parse pixels from one tile
    int32_t parseOneTile(TileData<T>& tileData, TileKey tile);
    // Parse a binary temporary file (written by BoundaryBuffer)
    int32_t parseBoundaryFile(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);
    int32_t parseBoundaryFileExtended(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);
    // Parse bufferred boundary data in memory
    int32_t parseBoundaryMemoryStandard(TileData<T>& tileData,
        InMemoryStorageStandard<T>* memStore, uint32_t bufferKey);
    int32_t parseBoundaryMemoryExtended(TileData<T>& tileData,
        InMemoryStorageExtended<T>* memStore, uint32_t bufferKey);

    int32_t initAnchorsHybrid(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch, const vec2f_t* fixedAnchors = nullptr);
    int32_t initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch);
    int32_t makeMinibatch(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch);

    // Prepare results of internal points
    ProcessedResult formatPixelResultWithOriginalData(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket);
    ProcessedResult formatPixelResult(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket);

    // Process one tile or one boundary buffer
    void processTile(TileData<T> &tileData, int threadId=0, int ticket = 0, vec2f_t* anchorPtr = nullptr);

    // write output column info to a json file
    void writeHeaderToJson();
    // write posterior pseudobulk to a tsv file
    void writePseudobulkToTsv();

    // Call *before* run()
    void setExtendedSchema();

    void tileWorker(int threadId) override;

    void boundaryWorker(int threadId) override;

    void writerWorker();

    void setupOutput();

};
