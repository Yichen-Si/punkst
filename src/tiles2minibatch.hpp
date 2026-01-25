#pragma once

#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include <stdexcept>
#include <atomic>
#include <tuple>
#include <variant>
#include <cassert>
#include "error.hpp"
#include "json.hpp"
#include "nanoflann.hpp"
#include "nanoflann_utils.h"
#include <opencv2/imgproc.hpp>
#include "dataunits.hpp"
#include "tile_io.hpp"
#include "tilereader.hpp"
#include "hexgrid.h"
#include "utils.h"
#include "utils_sys.hpp"
#include "threads.hpp"

#include "Eigen/Dense"
#include "Eigen/Sparse"
using Eigen::MatrixXf;
using Eigen::MatrixXi;
using Eigen::SparseMatrix;
using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct Minibatch {
    // Required:
    int n  = 0;    // number of anchors
    int N  = 0;    // number of pixels
    int M  = 0;    // number of features
    SparseMatrix<float, Eigen::RowMajor> mtx;  // (N x M); observed data matrix
    SparseMatrix<float, Eigen::RowMajor> wij; // (N x n)
    RowMajorMatrixXf gamma; // (n x K); ~P(k|j)
    RowMajorMatrixXf theta; // (n x K); only for em-nmf
    // Does not need to be initialized:
    SparseMatrix<float, Eigen::RowMajor> psi; // (N x n)
    RowMajorMatrixXf phi;   // (N x K); ~P(k|i)
    double ll = 0.0;
};

enum class FieldType : uint8_t { INT32, FLOAT, STRING };
struct FieldDef {
    FieldType  type;
    size_t     size;    // STRING: fixed byte length; INT32/FLOAT: sizeof()
    size_t     offset;  // filled in after we know all fields
};

enum class MinibatchInputMode : uint8_t { Standard, Extended };
enum class MinibatchOutputMode : uint8_t { Standard, Original, Binary };
enum class MinibatchCoordDim : uint8_t { Dim2 = 2, Dim3 = 3 };

struct MinibatchIoConfig {
    MinibatchInputMode input = MinibatchInputMode::Standard;
    MinibatchOutputMode output = MinibatchOutputMode::Standard;
    bool outputAnchor = false;
    bool useTicketSystem = false;
    MinibatchCoordDim coordDim = MinibatchCoordDim::Dim2;
};

using AnchorPoint = PointCloud<float>::Point;

// manage one temporary buffer
struct BoundaryBuffer {
    uint32_t key; // row|col|isVertical
    uint8_t nTiles;
    std::shared_ptr<std::mutex> mutex;
    // Either a temporary file path or an in-memory storage
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

    // Write (to file) or store (in memory) records with fixed fields
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

    // Write (to file) or store (in memory) records with fixed fields (3D)
    template<typename T>
    void addRecords3D(const std::vector<RecordT3D<T>>& recs) {
        if (recs.empty()) return;
        std::lock_guard<std::mutex> lock(*mutex);

        if (auto* storagePtr = std::get_if<std::unique_ptr<IBoundaryStorage>>(&storage)) {
            if (!*storagePtr) {
                *storagePtr = std::make_unique<InMemoryStorageStandard3D<T>>();
            }
            if (auto* memStore = dynamic_cast<InMemoryStorageStandard3D<T>*>(storagePtr->get())) {
                memStore->data.insert(memStore->data.end(), recs.begin(), recs.end());
            } else {
                throw std::runtime_error("Mismatched storage type in BoundaryBuffer::addRecords3D");
            }
        } else if (auto* tmpFile = std::get_if<std::string>(&storage)) {
            std::ofstream ofs(*tmpFile, std::ios::binary | std::ios::app);
            if (!ofs) {
                error("%s: error opening temporary file %s", __FUNCTION__, tmpFile->c_str());
            }
            ofs.write(reinterpret_cast<const char*>(recs.data()), recs.size() * sizeof(RecordT3D<T>));
            ofs.close();
        }
        nTiles++;
    }

    // Write (to file) or store (in memory) records with additional fields
    template<typename T>
    void addRecordsExtended(
        const std::vector<RecordExtendedT<T>>& recs,
        const std::vector<FieldDef>&           schema,
        size_t                                 recordSize) {
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

    // Write (to file) or store (in memory) records with additional fields (3D)
    template<typename T>
    void addRecordsExtended3D(
        const std::vector<RecordExtendedT3D<T>>& recs,
        const std::vector<FieldDef>&             schema,
        size_t                                   recordSize) {
        if (recs.empty()) return;
        std::lock_guard<std::mutex> lock(*mutex);

        if (auto* storagePtr = std::get_if<std::unique_ptr<IBoundaryStorage>>(&storage)) {
            if (!*storagePtr) {
                *storagePtr = std::make_unique<InMemoryStorageExtended3D<T>>();
            }
            if (auto* memStore = dynamic_cast<InMemoryStorageExtended3D<T>*>(storagePtr->get())) {
                memStore->dataExtended.insert(memStore->dataExtended.end(), recs.begin(), recs.end());
            } else {
                throw std::runtime_error("Mismatched storage type in BoundaryBuffer::addRecords3D (Extended)");
            }
        } else if (auto* filePath = std::get_if<std::string>(&storage)) {
            std::vector<uint8_t> buf;
            buf.reserve(recs.size() * recordSize);
            for (auto &r : recs) {
                size_t baseOff = buf.size();
                buf.resize(buf.size() + recordSize);
                std::memcpy(buf.data() + baseOff, &r.recBase, sizeof(r.recBase));
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
template<typename T>
class Tiles2MinibatchBase {

public:

    Tiles2MinibatchBase(int nThreads, double r,
        TileReader& tileReader, const std::string& _outPref,
        lineParserUnival* parser, const MinibatchIoConfig& ioConfig,
        const std::string* opt = nullptr, double res = 1, int32_t debug = 0)
    : nThreads(nThreads), r(r), tileReader(tileReader), outPref(_outPref),
      lineParserPtr(parser), pixelResolution_(res), debug_(debug),
      useTicketSystem_(ioConfig.useTicketSystem),
      outputAnchor_(ioConfig.outputAnchor),
      inputMode_(ioConfig.input), outputMode_(ioConfig.output),
      coordDim_(ioConfig.coordDim) {
        tileSize = tileReader.getTileSize();
        if (opt && !(*opt).empty()) {
            useMemoryBuffer_ = false;
            tmpDir.init(*opt);
            notice("Created temporary directory: %s", tmpDir.path.string().c_str());
        } else {
            useMemoryBuffer_ = true;
        }
        resultQueue.set_capacity(static_cast<size_t>(std::max(1, nThreads)));
        if (outputAnchor_) {
            anchorQueue.set_capacity(static_cast<size_t>(std::max(1, nThreads)));
        }
        if (!lineParserPtr) {
            error("%s: lineParser is required", __func__);
        }
        if (pixelResolution_ <= 0) {
            pixelResolution_ = 1;
        }
        configureInputMode();
        configureOutputMode();
    }

    virtual ~Tiles2MinibatchBase() {
        closeOutput();
    }

    void run();
    // Load fixed anchor points from a file and assign to tiles/boundaries
    int32_t loadAnchors(const std::string& anchorFile);

    void setFeatureNames(const std::vector<std::string>& names) {
        if (M_ > 0) {assert((int32_t) names.size() == M_);}
        else {M_ = names.size();}
        featureNames = names;
    }
    void setOutputProbDigits(int32_t digits) { probDigits = digits; }
    void setOutputCoordDigits(int32_t digits) { floatCoordDigits = digits; }
    virtual int32_t getFactorCount() const = 0;

protected:

    // buffer output results (for one tile)
    struct ResultBuf {
        int32_t ticket;
        float xmin, xmax, ymin, ymax;
        std::vector<std::string> outputLines;
        std::vector<PixTopProbs<int32_t>> outputObjs;
        std::vector<PixTopProbs3D<int32_t>> outputObjs3d;
        uint32_t npts;
        bool useObj = false;
        ResultBuf(int32_t t=0, float x1=0, float x2=0, float y1=0, float y2=0)
        : ticket(t), xmin(x1), xmax(x2), ymin(y1), ymax(y2), npts(0) {}
        bool operator>(const ResultBuf& other) const {
            return ticket > other.ticket;
        }
    };

    using ParseTileFn = int32_t (Tiles2MinibatchBase::*)(TileData<T>& tileData, TileKey tile);
    using ParseBoundaryFileFn = int32_t (Tiles2MinibatchBase::*)(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);
    using ParseBoundaryMemoryFn = int32_t (Tiles2MinibatchBase::*)(TileData<T>& tileData, IBoundaryStorage* storage, uint32_t bufferKey);
    using FormatPixelFn = ResultBuf (Tiles2MinibatchBase::*)(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0);

    int nThreads; // Number of worker threads
    int tileSize; // Tile size (square)
    double r;     // Processing radius (padding width)
    std::string outPref;
    TileReader& tileReader;
    ScopedTempDir tmpDir;
    bool useMemoryBuffer_;
    bool useTicketSystem_;
    int32_t debug_;

    int fdMain = -1;
    int fdIndex = -1;
    int fdAnchor = -1;
    size_t outputSize = 0;
    size_t headerSize = 0;
    size_t anchorHeaderSize = 0;
    size_t anchorOutputSize = 0;
    std::map<uint32_t, std::shared_ptr<BoundaryBuffer>> boundaryBuffers;
    std::mutex boundaryBuffersMapMutex; // Protects modifying boundaryBuffers
    ThreadSafeQueue<std::pair<TileKey, int32_t> > tileQueue;
    ThreadSafeQueue<std::pair<std::shared_ptr<BoundaryBuffer>, int32_t>> bufferQueue;
    std::vector<std::thread> workThreads;
    ThreadSafeQueue<ResultBuf> resultQueue;
    ThreadSafeQueue<ResultBuf> anchorQueue;
    // Anchors (optionally preloaded from files)
    // (we may need more than one set of pre-defined anchors in the future)
    using vec2f_t = std::vector<std::vector<float>>;
    std::unordered_map<TileKey, vec2f_t, TileKeyHash> fixedAnchorForTile;
    std::unordered_map<uint32_t, vec2f_t> fixedAnchorForBoundary;
    // Output/formatting related
    bool outputBinary_ = false;
    size_t outputRecordSize_ = 0;
    bool outputOriginalData_ = false;
    bool outputBackgroundProbDense_ = false;
    bool outputBackgroundProbExpand_ = false;
    bool outputAnchor_ = false;
    MinibatchInputMode inputMode_ = MinibatchInputMode::Standard;
    MinibatchOutputMode outputMode_ = MinibatchOutputMode::Standard;
    MinibatchCoordDim coordDim_ = MinibatchCoordDim::Dim2;
    ParseTileFn parseTileFn_ = &Tiles2MinibatchBase::parseOneTileStandard;
    ParseBoundaryFileFn parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFile;
    ParseBoundaryMemoryFn parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemoryStandardWrapper;
    FormatPixelFn formatPixelFn_ = &Tiles2MinibatchBase::formatPixelResultStandard;
    std::vector<std::string> featureNames;
    float pixelResolution_ = 1.0f;
    int32_t floatCoordDigits = 2, probDigits = 4;
    int32_t topk_ = 3;
    bool useExtended_ = false;
    lineParserUnival* lineParserPtr = nullptr;
    std::vector<FieldDef> schema_;
    size_t recordSize_ = 0;
    int32_t M_ = 0;

    /* Worker */

    virtual void onWorkerStart(int threadId) { (void)threadId; }

    void tileWorker(int threadId);
    void boundaryWorker(int threadId);
    void writerWorker();
    void anchorWriterWorker();

    /* Key logic */
    void configureInputMode();
    void configureOutputMode();

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

    // given (x, y) and its tile, compute all buffers it falls into
    // and whether it is "internal" to the tile
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

    void pt2tile(T x, T y, TileKey &tile) const {
        tile.row = static_cast<int32_t>(std::floor(y / tileSize));
        tile.col = static_cast<int32_t>(std::floor(x / tileSize));
    }

    void buffer2bound(bool isVertical, int32_t& bufRow, int32_t& bufCol, float& xmin, float& xmax, float& ymin, float& ymax) {
        if (isVertical) {
            xmin = static_cast<float>((bufCol + 1.) * tileSize - 2 * r);
            xmax = static_cast<float>((bufCol + 1.) * tileSize + 2 * r);
            ymin = static_cast<float>(bufRow * tileSize);
            ymax = static_cast<float>(bufRow * tileSize + tileSize);
        } else {
            xmin = static_cast<float>(bufCol * tileSize - r);
            xmax = static_cast<float>(bufCol * tileSize + tileSize + r);
            ymin = static_cast<float>((bufRow + 1) * tileSize - 2 * r);
            ymax = static_cast<float>((bufRow + 1) * tileSize + 2 * r);
        }
    }
    void bufferId2bound(uint32_t bufferId, float& xmin, float& xmax, float& ymin, float& ymax) {
        int32_t bufRow, bufCol;
        bool isVertical = decodeTempFileKey(bufferId, bufRow, bufCol);
        buffer2bound(isVertical, bufRow, bufCol, xmin, xmax, ymin, ymax);
    }

    bool isInternal(T x0, T y0, TileKey tile) {
        T x = x0 - tile.col * tileSize;
        T y = y0 - tile.row * tileSize;
        return (x > r && x < tileSize - r && y > r && y < tileSize - r);
    }
    bool isInternal(T x, T y, TileData<T>& tileData) {
        return (x > tileData.xmin + r && x < tileData.xmax - r &&
                y > tileData.ymin + r && y < tileData.ymax - r);
    }
    bool isInternalToBuffer(float x, float y, uint32_t bufferId) {
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

    virtual void processTile(TileData<T>& tileData, int threadId, int ticket, vec2f_t* anchorPtr) = 0;
    virtual void postRun() {}

    vec2f_t* lookupTileAnchors(const TileKey& tile) {
        auto it = fixedAnchorForTile.find(tile);
        if (it == fixedAnchorForTile.end()) {
            return nullptr;
        }
        return &it->second;
    }

    vec2f_t* lookupBoundaryAnchors(uint32_t key) {
        auto it = fixedAnchorForBoundary.find(key);
        if (it == fixedAnchorForBoundary.end()) {
            return nullptr;
        }
        return &it->second;
    }

    /* I/O */
    // Given data and anchor pos, build pixels & pixel-anchor relations
    int32_t buildMinibatchCore(TileData<T>& tileData,
        std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
        double distR, double distNu);
    int32_t buildMinibatchCore3D(TileData<T>& tileData,
        std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
        double distR, double distNu);
    // Create anchor grid and initialize anchor level counts
    int32_t buildAnchors(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, HexGrid& hexGrid_, int32_t nMoves_, double minCount = 0);
    int32_t buildAnchors3D(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, HexGrid& hexGrid_, int32_t nMoves_, double minCount = 0);
    // Parsing helpers
    int32_t parseOneTile(TileData<T>& tileData, TileKey tile);
    int32_t parseOneTileStandard(TileData<T>& tileData, TileKey tile);
    int32_t parseOneTileExtended(TileData<T>& tileData, TileKey tile);
    int32_t parseOneTileStandard3D(TileData<T>& tileData, TileKey tile);
    int32_t parseOneTileExtended3D(TileData<T>& tileData, TileKey tile);
    int32_t parseBoundaryFile(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);
    int32_t parseBoundaryFileExtended(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);
    int32_t parseBoundaryFile3D(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);
    int32_t parseBoundaryFileExtended3D(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr);
    int32_t parseBoundaryMemoryStandard(TileData<T>& tileData,
        InMemoryStorageStandard<T>* memStore, uint32_t bufferKey);
    int32_t parseBoundaryMemoryExtended(TileData<T>& tileData,
        InMemoryStorageExtended<T>* memStore, uint32_t bufferKey);
    int32_t parseBoundaryMemoryStandard3D(TileData<T>& tileData,
        InMemoryStorageStandard3D<T>* memStore, uint32_t bufferKey);
    int32_t parseBoundaryMemoryExtended3D(TileData<T>& tileData,
        InMemoryStorageExtended3D<T>* memStore, uint32_t bufferKey);
    int32_t parseBoundaryMemoryStandardWrapper(TileData<T>& tileData,
        IBoundaryStorage* storage, uint32_t bufferKey);
    int32_t parseBoundaryMemoryExtendedWrapper(TileData<T>& tileData,
        IBoundaryStorage* storage, uint32_t bufferKey);
    int32_t parseBoundaryMemoryStandard3DWrapper(TileData<T>& tileData,
        IBoundaryStorage* storage, uint32_t bufferKey);
    int32_t parseBoundaryMemoryExtended3DWrapper(TileData<T>& tileData,
        IBoundaryStorage* storage, uint32_t bufferKey);
    // Output helpers
    void setExtendedSchema(size_t offset);
    void setupOutput();
    void closeOutput();
    std::string composeHeader();
    ResultBuf formatAnchorResult(const std::vector<AnchorPoint>& anchors, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, float xmin, float xmax, float ymin, float ymax);
    ResultBuf formatPixelResult(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0 = nullptr);
    ResultBuf formatPixelResultStandard(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0 = nullptr);
    ResultBuf formatPixelResultWithOriginalData(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0 = nullptr);
    ResultBuf formatPixelResultWithBackground(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0 = nullptr);
    ResultBuf formatPixelResultBinary(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0 = nullptr);
    ResultBuf formatPixelResultStandard3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0 = nullptr);
    ResultBuf formatPixelResultWithOriginalData3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0 = nullptr);
    ResultBuf formatPixelResultWithBackground3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0 = nullptr);
    ResultBuf formatPixelResultBinary3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0 = nullptr);

};
