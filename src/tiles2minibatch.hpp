#pragma once

#include <cmath>
#include <algorithm>
#include <random>
#include <cassert>
#include <stdexcept>
#include <atomic>
#include <functional>
#include <limits>
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
#include "bccgrid.hpp"
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
    enum class StorageMode : uint8_t { GenericSparse, SingleMolecule };
    StorageMode storageMode = StorageMode::GenericSparse;
    SparseMatrix<float, Eigen::RowMajor> mtx;  // (N x M); observed data matrix
    SparseMatrix<float, Eigen::RowMajor> wij; // (N x n)
    SparseMatrix<float, Eigen::RowMajor> psi; // (N x n)
    // SingleMolecule mode specific fields
    std::vector<uint32_t> featureIdx; // (N); one feature id per pixel
    std::vector<float> featureWeight; // (N); count of the feature per pixel
    std::vector<uint32_t> rowOffsets; // (N + 1); CSR offsets for anchor edges
    std::vector<uint32_t> edgeAnchorIdx; // CSR anchor indices
    std::vector<float> wijVal; // CSR edge values; raw weights in builders, logits in model-specific consumers
    std::vector<float> psiVal; // CSR normalized edge weights
    // Filled during inference
    RowMajorMatrixXf gamma; // (n x K); ~P(k|j)
    RowMajorMatrixXf phi;   // (N x K); ~P(k|i)
    RowMajorMatrixXf theta; // (n x K), em-nmf specific
    double ll = 0.0;

    void clearDataSgl() {
        featureIdx.clear(); featureWeight.clear();
        rowOffsets.clear(); edgeAnchorIdx.clear();
        wijVal.clear(); psiVal.clear();
    }
    void clearDataMtx() {
        mtx.resize(0, 0);
        wij.resize(0, 0);
        psi.resize(0, 0);
    }
    bool checkIfReadySgl() const {
        if (rowOffsets.size() != static_cast<size_t>(N + 1)) {
            return false;
        }
        size_t expectedEdges = rowOffsets.back();
        return featureIdx.size() == static_cast<size_t>(N) &&
               featureWeight.size() == static_cast<size_t>(N) &&
               edgeAnchorIdx.size() == expectedEdges &&
               wijVal.size() == expectedEdges &&
               psiVal.size() == expectedEdges;
    }
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
    bool singleMolecule = false;
    bool outputAnchor = false;
    bool useTicketSystem = false;
    bool nativeRegularTiles = false;
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
      singleMolecule_(ioConfig.singleMolecule),
      useTicketSystem_(ioConfig.useTicketSystem),
      outputAnchor_(ioConfig.outputAnchor),
      inputMode_(ioConfig.input), outputMode_(ioConfig.output),
      nativeRegularTiles_(ioConfig.nativeRegularTiles),
      coordDim_(MinibatchCoordDim::Dim2) {
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
        nativeBinaryRegularTiles_ = nativeRegularTiles_ && outputBinary_;
        workerOutputContext_.resize(static_cast<size_t>(std::max(1, nThreads)));
    }

    virtual ~Tiles2MinibatchBase() {closeOutput();}

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
    void set3Dparameters(bool isThin,
        double zMin = std::numeric_limits<double>::quiet_NaN(),
        double zMax = std::numeric_limits<double>::quiet_NaN(),
        float zRes = -1.0f, bool enforceZrange = false,
        float standard3DBccGridDist = -1.0f,
        const std::vector<float>& thin3DZLevels = {});
    virtual int32_t getFactorCount() const = 0;

protected:

    using AnchorKey2D = std::tuple<int32_t, int32_t, int32_t, int32_t>;
    using AnchorKey3D = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>;

    enum class OutputSourceKind : uint8_t { None = 0, MainTile = 1, Boundary = 2 };

    struct ShardFragmentIndex {
        uint8_t kind = 0;
        uint8_t reserved0 = 0;
        uint8_t reserved1 = 0;
        uint8_t reserved2 = 0;
        int32_t row = 0;
        int32_t col = 0;
        uint32_t boundaryKey = 0;
        uint32_t npts = 0;
        uint64_t dataOffset = 0;
        uint64_t dataBytes = 0;
    };

    struct FragmentSpan {
        size_t shardId = 0;
        uint64_t dataOffset = 0;
        uint64_t dataBytes = 0;
        uint32_t npts = 0;
    };

    struct WorkerShardFiles {
        std::string mainDataPath;
        std::string mainIndexPath;
        std::string boundaryDataPath;
        std::string boundaryIndexPath;
        int fdMainData = -1;
        int fdMainIndex = -1;
        int fdBoundaryData = -1;
        int fdBoundaryIndex = -1;
        uint64_t mainDataSize = 0;
        uint64_t boundaryDataSize = 0;
    };

    struct WorkerOutputContext {
        OutputSourceKind kind = OutputSourceKind::None;
        TileKey tile{0, 0};
        uint32_t boundaryKey = 0;
    };

    // buffer output results (for one tile)
    struct ResultBuf {
        int32_t ticket;
        float xmin, xmax, ymin, ymax;
        std::vector<std::string> outputLines;
        std::vector<PixTopProbs<int32_t>> outputObjs;
        std::vector<PixTopProbsFeature<int32_t>> outputObjsFeature;
        std::vector<PixTopProbs3D<int32_t>> outputObjs3d;
        std::vector<PixTopProbsFeature3D<int32_t>> outputObjsFeature3d;
        uint32_t npts;
        bool useObj = false; // true for binary output
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
    bool nativeRegularTiles_ = false;
    bool nativeBinaryRegularTiles_ = false;
    size_t outputRecordSize_ = 0;
    bool outputOriginalData_ = false;
    bool outputBackgroundProbDense_ = false;
    bool outputBackgroundProbExpand_ = false;
    bool outputAnchor_ = false;
    bool singleMolecule_ = false;
    MinibatchInputMode inputMode_ = MinibatchInputMode::Standard;
    MinibatchOutputMode outputMode_ = MinibatchOutputMode::Standard;
    MinibatchCoordDim coordDim_ = MinibatchCoordDim::Dim2;
    bool ignoreOutsideZrange_ = false;
    bool useThin3DAnchors_ = false;
    double zMin_ = std::numeric_limits<double>::quiet_NaN();
    double zMax_ = std::numeric_limits<double>::quiet_NaN();
    std::vector<float> thin3DZLevels_;
    int32_t nZLevels_ = 0;
    uint64_t thin3DHashSeed_ = 0x8f3f73b5cf1c9d4bull;
    float standard3DBccSize_ = -1.0f;
    float standard3DBccGridDist_ = -1.0f;
    ParseTileFn parseTileFn_ = &Tiles2MinibatchBase::parseOneTileStandard;
    ParseBoundaryFileFn parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFile;
    ParseBoundaryMemoryFn parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemoryStandardWrapper;
    FormatPixelFn formatPixelFn_ = &Tiles2MinibatchBase::formatPixelResultStandard;
    std::vector<std::string> featureNames;
    float pixelResolution_ = 1.0f;
    float pixelResolutionZ_ = 1.0f;
    int32_t floatCoordDigits = 2, probDigits = 4;
    int32_t topk_ = 3;
    bool useExtended_ = false;
    lineParserUnival* lineParserPtr = nullptr;
    std::vector<FieldDef> schema_;
    size_t recordSize_ = 0;
    int32_t M_ = 0;
    std::vector<WorkerShardFiles> workerShards_;
    std::vector<WorkerOutputContext> workerOutputContext_;

    /* Worker */

    virtual void onWorkerStart(int threadId) { (void)threadId; }

    void tileWorker(int threadId);
    void boundaryWorker(int threadId);
    void writerWorker();
    void anchorWriterWorker();
    void submitPixelResult(ResultBuf&& result, int threadId);

    /* Key logic */
    void configureInputMode();
    void configureOutputMode();
    void setupNativeBinaryShards();
    void closeNativeBinaryShards();
    void mergeNativeBinaryShards();
    IndexHeader buildIndexHeader(bool fragmented) const;
    std::vector<char> serializeBinaryResult(const ResultBuf& result) const;
    void appendNativeBinaryResult(const ResultBuf& result, int threadId);

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
        // vertical, right
        if (x > tileSize - 2 * r && tile.col < tileReader.maxcol) {
            bufferidx.push_back(encodeTempFileKey(true, tile.row, tile.col));
        }
        // vertical, left
        if (x < 2 * r && tile.col > tileReader.mincol) {
            bufferidx.push_back(encodeTempFileKey(true, tile.row, tile.col - 1));
        }
        // horizontal, up
        if (y < 2 * r && tile.row > tileReader.minrow) {
            bufferidx.push_back(encodeTempFileKey(false, tile.row - 1, tile.col));
        }
        // horizontal, down
        if (y > tileSize - 2 * r && tile.row < tileReader.maxrow) {
            bufferidx.push_back(encodeTempFileKey(false, tile.row, tile.col));
        }
        if (x < r && tile.col > tileReader.mincol) {
            if (y < 2 * r && tile.row > tileReader.minrow) {
                bufferidx.push_back(encodeTempFileKey(false, tile.row - 1, tile.col - 1));
            }
            if (y > tileSize - 2 * r && tile.row < tileReader.maxrow) {
                bufferidx.push_back(encodeTempFileKey(false, tile.row, tile.col - 1));
            }
        }
        if (x > tileSize - r && tile.col < tileReader.maxcol) {
            if (y < 2 * r && tile.row > tileReader.minrow) {
                bufferidx.push_back(encodeTempFileKey(false, tile.row - 1, tile.col + 1));
            }
            if (y > tileSize - 2 * r && tile.row < tileReader.maxrow) {
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
            if (bufRow == tileReader.maxrow) {
                return (x > x_min && x < x_max && y > y_min);
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

    // Given data and anchor pos, build pixels & pixel-anchor relations
    double buildMinibatchCore(TileData<T>& tileData,
        std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
        double supportRadius, double distNu);
    double buildMinibatchCore3D(TileData<T>& tileData,
        std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
        const BCCGrid* bccGrid, double supportRadius, double distNu);
    double buildMinibatchCoreSingleMolecule(TileData<T>& tileData,
        std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
        double supportRadius, double distNu);
    double buildMinibatchCoreSingleMolecule3D(TileData<T>& tileData,
        std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
        const BCCGrid* bccGrid, double supportRadius, double distNu);
    int32_t thin3DAnchorZIndexForKey(const AnchorKey2D& key) const;
    void thin3DAnchorKeyToFineAxial(int32_t& u, int32_t& v, const AnchorKey2D& key, int32_t nMoves_) const;
    AnchorKey2D thin3DFineAxialToAnchorKey(int32_t u, int32_t v, int32_t nMoves_) const;
    void forEachThin3DAnchorWithinRadius(float x, float y, float z,
        const HexGrid& hexGrid_, int32_t nMoves_, float supportRadius,
        const std::function<void(const AnchorKey2D&, float, float, float, float)>& emit) const;
    // Choose a set of neighboring anchors for each point
    void forEachAnchorCandidate2D(const TileData<T>& tileData,
        const HexGrid& hexGrid_, int32_t nMoves_, const std::function<void(uint32_t, float, const AnchorKey2D&)>& emit) const;
    void forEachAnchorCandidate3D(const TileData<T>& tileData,
        const std::function<void(uint32_t, float, const AnchorKey3D&)>& emit) const;
    void forEachAnchorCandidate3D(const TileData<T>& tileData,
        const BCCGrid& bccGrid, const std::function<void(uint32_t, float, const AnchorKey3D&)>& emit) const;
    void forEachAnchorCandidateThin3D(const TileData<T>& tileData,
        const HexGrid& hexGrid_, int32_t nMoves_, double supportRadius, double distNu,
        const std::function<void(uint32_t, float, const AnchorKey2D&)>& emit) const;
    // Anchor key helper
    void anchorKeyToCoord2D(float& x, float& y, const AnchorKey2D& key, const HexGrid& hexGrid_, int32_t nMoves_) const;
    void anchorKeyToCoord3D(float& x, float& y, float& z, const AnchorKey3D& key) const;
    void anchorKeyToCoord3D(float& x, float& y, float& z, const AnchorKey3D& key, const BCCGrid& bccGrid) const;
    void anchorKeyToCoordThin3D(float& x, float& y, float& z, const AnchorKey2D& key, const HexGrid& hexGrid_, int32_t nMoves_) const;
    // Build anchor and aggregate counts
    int32_t buildAnchors(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, const HexGrid& hexGrid_, int32_t nMoves_, double minCount = 0);
    int32_t buildAnchors3D(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, double minCount = 0);
    int32_t buildAnchors3D(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, const BCCGrid& bccGrid, double minCount = 0);
    int32_t buildAnchorsThin3D(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents,
        const HexGrid& hexGrid_, int32_t nMoves_, double minCount, double supportRadius, double distNu);

    /* I/O */
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
    float anchor_distance_weight(float dist, float radius, float nu = 1.943f) const;
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
