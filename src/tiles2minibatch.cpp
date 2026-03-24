#include "tiles2minibatch.hpp"
#include "numerical_utils.hpp"
#include <cmath>
#include <numeric>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <thread>
#include <map>
#include <algorithm>

namespace {

template<typename CoordT>
void append_record_bytes(std::vector<char>& bytes, const PixTopProbs<CoordT>& obj, size_t recordSize) {
    const size_t off = bytes.size();
    bytes.resize(off + recordSize);
    char* dst = bytes.data() + off;
    std::memcpy(dst, &obj.x, sizeof(obj.x));
    std::memcpy(dst + sizeof(obj.x), &obj.y, sizeof(obj.y));
    if (!obj.ks.empty()) {
        std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y),
            obj.ks.data(), obj.ks.size() * sizeof(int32_t));
    }
    if (!obj.ps.empty()) {
        std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y) + obj.ks.size() * sizeof(int32_t),
            obj.ps.data(), obj.ps.size() * sizeof(float));
    }
}

template<typename CoordT>
void append_record_bytes(std::vector<char>& bytes, const PixTopProbs3D<CoordT>& obj, size_t recordSize) {
    const size_t off = bytes.size();
    bytes.resize(off + recordSize);
    char* dst = bytes.data() + off;
    std::memcpy(dst, &obj.x, sizeof(obj.x));
    std::memcpy(dst + sizeof(obj.x), &obj.y, sizeof(obj.y));
    std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y), &obj.z, sizeof(obj.z));
    if (!obj.ks.empty()) {
        std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y) + sizeof(obj.z),
            obj.ks.data(), obj.ks.size() * sizeof(int32_t));
    }
    if (!obj.ps.empty()) {
        std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y) + sizeof(obj.z) + obj.ks.size() * sizeof(int32_t),
            obj.ps.data(), obj.ps.size() * sizeof(float));
    }
}

template<typename CoordT>
void append_record_bytes(std::vector<char>& bytes, const PixTopProbsFeature<CoordT>& obj, size_t recordSize) {
    const size_t off = bytes.size();
    bytes.resize(off + recordSize);
    char* dst = bytes.data() + off;
    std::memcpy(dst, &obj.x, sizeof(obj.x));
    std::memcpy(dst + sizeof(obj.x), &obj.y, sizeof(obj.y));
    std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y), &obj.featureIdx, sizeof(obj.featureIdx));
    if (!obj.ks.empty()) {
        std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y) + sizeof(obj.featureIdx),
            obj.ks.data(), obj.ks.size() * sizeof(int32_t));
    }
    if (!obj.ps.empty()) {
        std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y) + sizeof(obj.featureIdx) + obj.ks.size() * sizeof(int32_t),
            obj.ps.data(), obj.ps.size() * sizeof(float));
    }
}

template<typename CoordT>
void append_record_bytes(std::vector<char>& bytes, const PixTopProbsFeature3D<CoordT>& obj, size_t recordSize) {
    const size_t off = bytes.size();
    bytes.resize(off + recordSize);
    char* dst = bytes.data() + off;
    std::memcpy(dst, &obj.x, sizeof(obj.x));
    std::memcpy(dst + sizeof(obj.x), &obj.y, sizeof(obj.y));
    std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y), &obj.z, sizeof(obj.z));
    std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y) + sizeof(obj.z), &obj.featureIdx, sizeof(obj.featureIdx));
    if (!obj.ks.empty()) {
        std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y) + sizeof(obj.z) + sizeof(obj.featureIdx),
            obj.ks.data(), obj.ks.size() * sizeof(int32_t));
    }
    if (!obj.ps.empty()) {
        std::memcpy(dst + sizeof(obj.x) + sizeof(obj.y) + sizeof(obj.z) + sizeof(obj.featureIdx) + obj.ks.size() * sizeof(int32_t),
            obj.ps.data(), obj.ps.size() * sizeof(float));
    }
}

inline TileKey boundary_tile_key(const PixTopProbs<int32_t>& obj, int tileSize, float pixelResolution) {
    const float x = static_cast<float>(obj.x) * pixelResolution;
    const float y = static_cast<float>(obj.y) * pixelResolution;
    return TileKey{
        static_cast<int32_t>(std::floor(y / tileSize)),
        static_cast<int32_t>(std::floor(x / tileSize))
    };
}

inline TileKey boundary_tile_key(const PixTopProbs3D<int32_t>& obj, int tileSize, float pixelResolution) {
    const float x = static_cast<float>(obj.x) * pixelResolution;
    const float y = static_cast<float>(obj.y) * pixelResolution;
    return TileKey{
        static_cast<int32_t>(std::floor(y / tileSize)),
        static_cast<int32_t>(std::floor(x / tileSize))
    };
}

inline TileKey boundary_tile_key(const PixTopProbsFeature<int32_t>& obj, int tileSize, float pixelResolution) {
    const float x = static_cast<float>(obj.x) * pixelResolution;
    const float y = static_cast<float>(obj.y) * pixelResolution;
    return TileKey{
        static_cast<int32_t>(std::floor(y / tileSize)),
        static_cast<int32_t>(std::floor(x / tileSize))
    };
}

inline TileKey boundary_tile_key(const PixTopProbsFeature3D<int32_t>& obj, int tileSize, float pixelResolution) {
    const float x = static_cast<float>(obj.x) * pixelResolution;
    const float y = static_cast<float>(obj.y) * pixelResolution;
    return TileKey{
        static_cast<int32_t>(std::floor(y / tileSize)),
        static_cast<int32_t>(std::floor(x / tileSize))
    };
}

inline TileKey boundary_tile_key(const PixTopProbsFeature<float>& obj, int tileSize, float /*pixelResolution*/) {
    return TileKey{
        static_cast<int32_t>(std::floor(obj.y / tileSize)),
        static_cast<int32_t>(std::floor(obj.x / tileSize))
    };
}

inline TileKey boundary_tile_key(const PixTopProbsFeature3D<float>& obj, int tileSize, float /*pixelResolution*/) {
    return TileKey{
        static_cast<int32_t>(std::floor(obj.y / tileSize)),
        static_cast<int32_t>(std::floor(obj.x / tileSize))
    };
}

template<typename PayloadT>
void write_bucketed_binary_payload(int fdMain, int fdIndex, size_t& outputSize,
    const PayloadT& data, size_t recordSize, int tileSize, float pixelResolution) {
    std::map<TileKey, std::pair<std::vector<char>, uint32_t>> buckets;
    for (const auto& obj : data) {
        TileKey tile = boundary_tile_key(obj, tileSize, pixelResolution);
        auto& bucket = buckets[tile];
        append_record_bytes(bucket.first, obj, recordSize);
        bucket.second++;
    }
    for (const auto& kv : buckets) {
        const auto& tile = kv.first;
        const auto& bytes = kv.second.first;
        if (bytes.empty()) {
            continue;
        }
        IndexEntryF entry(tile.row, tile.col);
        entry.st = outputSize;
        entry.n = kv.second.second;
        tile2bound(tile, entry.xmin, entry.xmax, entry.ymin, entry.ymax, tileSize);
        if (!write_all(fdMain, bytes.data(), bytes.size())) {
            error("%s: Error writing bucketed binary records to main output file", __func__);
        }
        outputSize += bytes.size();
        entry.ed = outputSize;
        if (!write_all(fdIndex, &entry, sizeof(entry))) {
            error("%s: Error writing bucketed index entry", __func__);
        }
    }
}

} // namespace

template<typename T>
float Tiles2MinibatchBase<T>::anchor_distance_weight(float dist, float radius, float nu) const {
    if (radius <= 0 || nu < 0) {
        error("%s: invalid radius or nu parameters", __func__);
    }
    float weight = 1. - std::pow(dist / radius, nu);
    return std::clamp(weight, 0.05f, 0.95f);
}

template<typename T>
void Tiles2MinibatchBase<T>::run() {
    setupOutput();
    const bool useLegacyWriter = !nativeBinaryRegularTiles_;
    std::thread writer;
    if (useLegacyWriter) {
        writer = std::thread(&Tiles2MinibatchBase<T>::writerWorker, this);
    }
    std::thread anchorWriter;
    if (outputAnchor_) {
        anchorWriter = std::thread(&Tiles2MinibatchBase<T>::anchorWriterWorker, this);
    }

    notice("Phase 1 Launching %d worker threads", nThreads);
    workThreads.clear();
    workThreads.reserve(static_cast<size_t>(nThreads));
    for (int i = 0; i < nThreads; ++i) {
        workThreads.emplace_back(&Tiles2MinibatchBase<T>::tileWorker, this, i);
    }

    std::vector<TileKey> tileList;
    tileReader.getTileList(tileList);
    std::sort(tileList.begin(), tileList.end());
    int32_t ticket = 0;
    for (const auto& tile : tileList) {
        tileQueue.push(std::make_pair(tile, ticket++));
        if (debug_ > 0 && ticket >= debug_) {
            break;
        }
    }
    tileQueue.set_done();
    for (auto& t : workThreads) {
        t.join();
    }
    workThreads.clear();

    notice("Phase 2 Launching %d worker threads", nThreads);
    std::vector<std::shared_ptr<BoundaryBuffer>> buffers;
    buffers.reserve(boundaryBuffers.size());
    for (auto &kv : boundaryBuffers) {
        buffers.push_back(kv.second);
    }
    std::sort(buffers.begin(), buffers.end(),
        [](const std::shared_ptr<BoundaryBuffer>& A,
           const std::shared_ptr<BoundaryBuffer>& B) {
            return A->key < B->key;
        });
    for (auto &bufferPtr : buffers) {
        bufferQueue.push(std::make_pair(bufferPtr, ticket++));
    }
    bufferQueue.set_done();
    workThreads.reserve(static_cast<size_t>(nThreads));
    for (int i = 0; i < nThreads; ++i) {
        workThreads.emplace_back(&Tiles2MinibatchBase<T>::boundaryWorker, this, i);
    }
    for (auto& t : workThreads) {
        t.join();
    }
    workThreads.clear();

    if (useLegacyWriter) {
        resultQueue.set_done();
    }
    if (outputAnchor_) {
        anchorQueue.set_done();
    }
    notice("%s: all workers done", __func__);

    if (useLegacyWriter && writer.joinable()) {
        writer.join();
    }
    if (anchorWriter.joinable()) {
        anchorWriter.join();
    }
    if (nativeBinaryRegularTiles_) {
        closeNativeBinaryShards();
        mergeNativeBinaryShards();
    }
    closeOutput();
    notice("%s: writer threads done", __func__);

    postRun();
}

template<typename T>
void Tiles2MinibatchBase<T>::configureInputMode() {
    if (!lineParserPtr) {
        error("%s: lineParser is required", __func__);
    }
    if (usesFeatureSpecificMode() && inputMode_ != MinibatchInputMode::Standard) {
        error("%s: feature-row modes currently do not support --ext-col-* output carry-through", __func__);
    }
    if (coordDim_ == MinibatchCoordDim::Dim3 && !lineParserPtr->hasZCoord()) {
        error("%s: 3D mode requires a z column", __func__);
    }
    if (inputMode_ == MinibatchInputMode::Extended) {
        if (!lineParserPtr->isExtended) {
            error("%s: extended input mode requires extended parser", __func__);
        }
        if (coordDim_ == MinibatchCoordDim::Dim3) {
            setExtendedSchema(sizeof(RecordT3D<T>));
            parseTileFn_ = &Tiles2MinibatchBase::parseOneTileExtended3D;
            parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFileExtended3D;
            parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemoryExtended3DWrapper;
        } else {
            setExtendedSchema(sizeof(RecordT<T>));
            parseTileFn_ = &Tiles2MinibatchBase::parseOneTileExtended;
            parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFileExtended;
            parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemoryExtendedWrapper;
        }
    } else {
        if (lineParserPtr->isExtended) {
            warning("%s: extended columns detected but input mode is standard; extra fields will be ignored", __func__);
        }
        useExtended_ = false;
        if (coordDim_ == MinibatchCoordDim::Dim3) {
            if (isSingleMoleculeMode()) {
                parseTileFn_ = &Tiles2MinibatchBase::parseOneTileSingleMolecule3D;
                parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFileSingleMolecule3D;
                parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemorySingleMolecule3DWrapper;
            } else {
                parseTileFn_ = &Tiles2MinibatchBase::parseOneTileStandard3D;
                parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFile3D;
                parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemoryStandard3DWrapper;
            }
        } else {
            if (isSingleMoleculeMode()) {
                parseTileFn_ = &Tiles2MinibatchBase::parseOneTileSingleMolecule;
                parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFileSingleMolecule;
                parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemorySingleMoleculeWrapper;
            } else {
                parseTileFn_ = &Tiles2MinibatchBase::parseOneTileStandard;
                parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFile;
                parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemoryStandardWrapper;
            }
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::configureOutputMode() {
    if (usesFeatureSpecificMode()) {
        if (outputMode_ != MinibatchOutputMode::Binary) {
            error("%s: feature-row modes currently support binary output only", __func__);
        }
        if (outputBackgroundProbDense_ || outputBackgroundProbExpand_) {
            error("%s: feature-row modes do not support background-expanded output", __func__);
        }
    }
    if (outputMode_ == MinibatchOutputMode::Binary) {
        outputBinary_ = true;
        outputOriginalData_ = false;
        formatPixelFn_ = (coordDim_ == MinibatchCoordDim::Dim3)
            ? &Tiles2MinibatchBase::formatPixelResultBinary3D
            : &Tiles2MinibatchBase::formatPixelResultBinary;
    } else if (outputMode_ == MinibatchOutputMode::Original) {
        outputBinary_ = false;
        outputOriginalData_ = true;
        formatPixelFn_ = (coordDim_ == MinibatchCoordDim::Dim3)
            ? &Tiles2MinibatchBase::formatPixelResultWithOriginalData3D
            : &Tiles2MinibatchBase::formatPixelResultWithOriginalData;
    } else {
        outputBinary_ = false;
        outputOriginalData_ = false;
        if (outputBackgroundProbExpand_) {
            formatPixelFn_ = (coordDim_ == MinibatchCoordDim::Dim3)
                ? &Tiles2MinibatchBase::formatPixelResultWithBackground3D
                : &Tiles2MinibatchBase::formatPixelResultWithBackground;
        } else {
            formatPixelFn_ = (coordDim_ == MinibatchCoordDim::Dim3)
                ? &Tiles2MinibatchBase::formatPixelResultStandard3D
                : &Tiles2MinibatchBase::formatPixelResultStandard;
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::tileWorker(int threadId) {
    onWorkerStart(threadId);
    std::pair<TileKey, int32_t> tileTicket;
    TileKey tile;
    int32_t ticket;
    while (tileQueue.pop(tileTicket)) {
        tile = tileTicket.first;
        ticket = tileTicket.second;
        if (nativeBinaryRegularTiles_) {
            workerOutputContext_[static_cast<size_t>(threadId)].kind = OutputSourceKind::MainTile;
            workerOutputContext_[static_cast<size_t>(threadId)].tile = tile;
            workerOutputContext_[static_cast<size_t>(threadId)].boundaryKey = 0;
        }
        TileData<T> tileData;
        int32_t ret = parseOneTile(tileData, tile);
        notice("%s: Thread %d (ticket %d) read tile (%d, %d) with %d internal pixels",
            __func__, threadId, ticket, tile.row, tile.col, ret);
        if (ret <= 10) {
            continue;
        }
        vec2f_t* anchorPtr = lookupTileAnchors(tile);
        processTile(tileData, threadId, ticket, anchorPtr);
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::boundaryWorker(int threadId) {
    onWorkerStart(threadId);
    std::pair<std::shared_ptr<BoundaryBuffer>, int32_t> bufferTicket;
    std::shared_ptr<BoundaryBuffer> bufferPtr;
    int32_t ticket;
    while (bufferQueue.pop(bufferTicket)) {
        bufferPtr = bufferTicket.first;
        ticket = bufferTicket.second;
        if (nativeBinaryRegularTiles_) {
            workerOutputContext_[static_cast<size_t>(threadId)].kind = OutputSourceKind::Boundary;
            workerOutputContext_[static_cast<size_t>(threadId)].boundaryKey = bufferPtr->key;
        }
        TileData<T> tileData;
        int32_t ret = 0;

        if (auto* storagePtr = std::get_if<std::unique_ptr<IBoundaryStorage>>(&(bufferPtr->storage))) {
            ret = (this->*parseBoundaryMemoryFn_)(tileData, storagePtr->get(), bufferPtr->key);
        } else if (auto* filePath = std::get_if<std::string>(&(bufferPtr->storage))) {
            ret = (this->*parseBoundaryFileFn_)(tileData, bufferPtr);
            std::remove(filePath->c_str());
        }
        notice("%s: Thread %d (ticket %d) read boundary buffer (%d) with %d internal pixels",
            __func__, threadId, ticket, bufferPtr->key, ret);
        vec2f_t* anchorPtr = lookupBoundaryAnchors(bufferPtr->key);
        processTile(tileData, threadId, ticket, anchorPtr);
    }
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::loadAnchors(const std::string& anchorFile) {
    std::ifstream inFile(anchorFile);
    if (!inFile) {
        error("Error opening anchors file: %s", anchorFile.c_str());
    }
    std::string line;
    std::vector<std::string> tokens;
    int32_t nAnchors = 0;
    while (std::getline(inFile, line)) {
        if (line.empty() || line[0] == '#') continue;
        split(tokens, "\t", line);
        if (tokens.size() < 2 || (coordDim_ == MinibatchCoordDim::Dim3 && tokens.size() < 3)) {
            error("Error reading anchors file at line: %s", line.c_str());
        }
        float x, y, z = 0.0f;
        bool valid = str2float(tokens[0], x) && str2float(tokens[1], y);
        if (coordDim_ == MinibatchCoordDim::Dim3) {
            valid = valid && str2float(tokens[2], z);
        }
        if (!valid) {
            error("Error reading anchors file at line: %s", line.c_str());
        }
        TileKey tile;
        if (!tileReader.pt2tile(x, y, tile)) {
            continue;
        }
        if (coordDim_ == MinibatchCoordDim::Dim3) {
            fixedAnchorForTile[tile].emplace_back(std::vector<float>{x, y, z});
        } else {
            fixedAnchorForTile[tile].emplace_back(std::vector<float>{x, y});
        }
        std::vector<uint32_t> bufferidx;
        int32_t ret = pt2buffer(bufferidx, x, y, tile);
        (void)ret;
        for (const auto& key : bufferidx) {
            if (coordDim_ == MinibatchCoordDim::Dim3) {
                fixedAnchorForBoundary[key].emplace_back(std::vector<float>{x, y, z});
            } else {
                fixedAnchorForBoundary[key].emplace_back(std::vector<float>{x, y});
            }
        }
        nAnchors++;
    }
    inFile.close();
    if (fixedAnchorForTile.empty()) {
        error("No anchors fall in the region of the input pixel data, please make sure the anchors are in the same coordinate system as the input data");
    }
    return nAnchors;
}

template<typename T>
void Tiles2MinibatchBase<T>::forEachAnchorCandidate2D(const TileData<T>& tileData, const HexGrid& hexGrid_, int32_t nMoves_, const std::function<void(uint32_t, float, const AnchorKey2D&)>& emit) const
{
    auto assign_pt = [&](float x, float y, uint32_t idx, float ct) {
        for (int32_t ir = 0; ir < nMoves_; ++ir) {
            for (int32_t ic = 0; ic < nMoves_; ++ic) {
                int32_t hx, hy;
                hexGrid_.cart_to_axial(hx, hy, x, y, ic * 1.0 / nMoves_, ir * 1.0 / nMoves_);
                emit(idx, ct, AnchorKey2D{hx, hy, ic, ir});
            }
        }
    };
    if (useExtended_) {
        for (const auto& pt : tileData.extended2D().extPts) {
            assign_pt(static_cast<float>(pt.recBase.x), static_cast<float>(pt.recBase.y),
                pt.recBase.idx, static_cast<float>(pt.recBase.ct));
        }
    } else if (isSingleMoleculeMode()) {
        const auto& smInput = tileData.singleMolecule2D();
        for (size_t i = 0; i < smInput.coordsFloat.size(); ++i) {
            const auto& coord = smInput.coordsFloat[i];
            assign_pt(coord.first, coord.second, smInput.featureIdx[i], smInput.obsWeight[i]);
        }
    } else {
        for (const auto& pt : tileData.standard2D().pts) {
            assign_pt(static_cast<float>(pt.x), static_cast<float>(pt.y), pt.idx, static_cast<float>(pt.ct));
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::anchorKeyToCoord2D(float& x, float& y, const AnchorKey2D& key, const HexGrid& hexGrid_, int32_t nMoves_) const {
    const int32_t hx = std::get<0>(key);
    const int32_t hy = std::get<1>(key);
    const int32_t ic = std::get<2>(key);
    const int32_t ir = std::get<3>(key);
    hexGrid_.axial_to_cart(x, y, hx, hy, ic * 1.0 / nMoves_, ir * 1.0 / nMoves_);
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::buildAnchors(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, const HexGrid& hexGrid_, int32_t nMoves_, double minCount) {
    if (coordDim_ == MinibatchCoordDim::Dim3) {
        if (useThin3DAnchors_) {
            error("%s: thin 3D anchor initialization requires supportRadius/distNu and must call buildAnchorsThin3D directly", __func__);
        }
        return buildAnchors3D(tileData, anchors, documents, minCount);
    }
    anchors.clear();
    documents.clear();
    std::map<AnchorKey2D, std::unordered_map<uint32_t, float>> hexAggregation;
    forEachAnchorCandidate2D(tileData, hexGrid_, nMoves_, [&](uint32_t idx, float ct, const AnchorKey2D& key) {
        hexAggregation[key][idx] += ct;
    });

    for (auto& entry : hexAggregation) {
        float sum = std::accumulate(entry.second.begin(), entry.second.end(), 0.0f,
            [](float acc, const auto& p) { return acc + p.second; });
        if (sum < minCount) {
            continue;
        }
        SparseObs obs;
        Document& doc = obs.doc;
        obs.ct_tot = sum;
        for (auto& featurePair : entry.second) {
            doc.ids.push_back(featurePair.first);
            doc.cnts.push_back(featurePair.second);
        }
        if (lineParserPtr->weighted) {
            for (size_t i = 0; i < doc.ids.size(); ++i) {
                doc.cnts[i] *= lineParserPtr->weights[doc.ids[i]];
            }
        }
        documents.push_back(std::move(obs));
        const auto& key = entry.first;
        float x, y;
        anchorKeyToCoord2D(x, y, key, hexGrid_, nMoves_);
        anchors.emplace_back(x, y);
    }
    return documents.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseOneTile(TileData<T>& tileData, TileKey tile) {
    return (this->*parseTileFn_)(tileData, tile);
}

template<typename T>
double Tiles2MinibatchBase<T>::buildMinibatchCoreSingleFeaturePixel(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    double supportRadius, double distNu) {
    struct SingleMoleculeKey2D {
        int32_t x = 0;
        int32_t y = 0;
        uint32_t feature = 0;
        bool operator==(const SingleMoleculeKey2D& other) const {
            return x == other.x && y == other.y && feature == other.feature;
        }
    };
    struct SingleMoleculeKey2DHash {
        size_t operator()(const SingleMoleculeKey2D& key) const {
            size_t h = std::hash<int32_t>()(key.x);
            h ^= std::hash<int32_t>()(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>()(key.feature) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct SingleMoleculePixel2D {
        int32_t x = 0;
        int32_t y = 0;
        uint32_t feature = 0;
        float weight = 0.0f;
        std::vector<uint32_t> originalIdx;
    };

    PointCloud<float> pc;
    pc.pts = std::move(anchors);
    kd_tree_f2_t kdtree(2, pc, {10});
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;

    const float res = pixelResolution_;
    const float l2radius = static_cast<float>(supportRadius * supportRadius);

    minibatch.storageMode = Minibatch::StorageMode::SingleMolecule;
    minibatch.clearDataSgl();
    minibatch.clearDataMtx();

    std::unordered_map<SingleMoleculeKey2D, uint32_t, SingleMoleculeKey2DHash> pixelLookup;
    std::vector<SingleMoleculePixel2D> groupedPixels;
    uint32_t idxOriginal = 0;
    uint32_t nPixels = 0;
    const auto& stdInput = tileData.standard2D();
    tileData.orgpts2pixel.assign(stdInput.pts.size(), -1);
    for (const auto& pt : stdInput.pts) {
        const int32_t x = static_cast<int32_t>(std::floor(pt.x / res));
        const int32_t y = static_cast<int32_t>(std::floor(pt.y / res));
        const SingleMoleculeKey2D key{x, y, pt.idx};
        auto it = pixelLookup.find(key);
        if (it == pixelLookup.end()) {
            SingleMoleculePixel2D pix;
            pix.x = x;
            pix.y = y;
            pix.feature = pt.idx;
            groupedPixels.push_back(std::move(pix));
            pixelLookup.emplace(key, nPixels++);
            it = pixelLookup.find(key);
        }
        SingleMoleculePixel2D& pix = groupedPixels[it->second];
        float weight = static_cast<float>(pt.ct);
        if (lineParserPtr->weighted) {
            weight *= static_cast<float>(lineParserPtr->weights[pt.idx]);
        }
        pix.weight += weight;
        pix.originalIdx.push_back(idxOriginal++);
    }

    tileData.coords.clear();
    tileData.rowFeatureIdx.clear();
    tileData.coords.reserve(nPixels);
    tileData.rowFeatureIdx.reserve(nPixels);
    minibatch.featureIdx.reserve(nPixels);
    minibatch.featureWeight.reserve(nPixels);
    minibatch.rowOffsets.reserve(nPixels + 1);
    minibatch.rowOffsets.push_back(0);
    minibatch.edgeAnchorIdx.reserve(nPixels * 4);

    uint32_t npt = 0;
    for (const auto& pix : groupedPixels) {
        float xy[2] = {pix.x * res, pix.y * res};
        const size_t n = kdtree.radiusSearch(xy, l2radius, indices_dists);
        if (n == 0) {
            continue;
        }

        float weightSum = 0.0f;
        const size_t edgeStart = minibatch.edgeAnchorIdx.size();
        for (size_t i = 0; i < n; ++i) {
            const float dist = std::sqrt(indices_dists[i].second);
            const float rawWeight = anchor_distance_weight(dist, supportRadius, distNu);
            minibatch.edgeAnchorIdx.push_back(indices_dists[i].first);
            minibatch.wijVal.push_back(rawWeight);
            minibatch.psiVal.push_back(rawWeight);
            weightSum += rawWeight;
        }
        for (size_t e = edgeStart; e < minibatch.psiVal.size(); ++e) {
            minibatch.psiVal[e] /= weightSum;
        }

        tileData.coords.emplace_back(pix.x, pix.y);
        tileData.rowFeatureIdx.push_back(pix.feature);
        minibatch.featureIdx.push_back(pix.feature);
        minibatch.featureWeight.push_back(pix.weight);
        for (auto v : pix.originalIdx) {
            tileData.orgpts2pixel[v] = static_cast<int32_t>(npt);
        }
        ++npt;
        minibatch.rowOffsets.push_back(static_cast<uint32_t>(minibatch.edgeAnchorIdx.size()));
    }
    anchors = std::move(pc.pts);
    const double avgDegree = (npt > 0)
        ? static_cast<double>(minibatch.edgeAnchorIdx.size()) / static_cast<double>(npt)
        : 0.0;
    debug("%s: created %zu single-feature-pixel edges between %u rows and %zu anchors, average degree %.2f",
        __func__, minibatch.edgeAnchorIdx.size(), npt, anchors.size(), avgDegree);

    minibatch.N = static_cast<int32_t>(npt);
    minibatch.M = M_;
    return avgDegree;
}

template<typename T>
double Tiles2MinibatchBase<T>::buildMinibatchCoreSingleMolecule(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    double supportRadius, double distNu) {
    PointCloud<float> pc;
    pc.pts = std::move(anchors);
    kd_tree_f2_t kdtree(2, pc, {10});
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;

    const float l2radius = static_cast<float>(supportRadius * supportRadius);

    minibatch.storageMode = Minibatch::StorageMode::SingleMolecule;
    minibatch.clearDataSgl();
    minibatch.clearDataMtx();

    tileData.coords.clear();
    const auto& smInput = tileData.singleMolecule2D();
    auto& smInputMut = tileData.singleMolecule2D();
    const size_t nObs = smInput.coordsFloat.size();
    minibatch.rowOffsets.reserve(nObs + 1);
    minibatch.rowOffsets.push_back(0);
    minibatch.edgeAnchorIdx.reserve(nObs * 4);

    size_t npt = 0;
    tileData.orgpts2pixel.assign(nObs, -1);
    for (size_t i = 0; i < nObs; ++i) {
        const auto coord = smInput.coordsFloat[i];
        const float xy[2] = {coord.first, coord.second};
        const size_t n = kdtree.radiusSearch(xy, l2radius, indices_dists);
        if (n == 0) {
            continue;
        }

        float weightSum = 0.0f;
        const size_t edgeStart = minibatch.edgeAnchorIdx.size();
        for (size_t i = 0; i < n; ++i) {
            const float dist = std::sqrt(indices_dists[i].second);
            const float rawWeight = anchor_distance_weight(dist, supportRadius, distNu);
            minibatch.edgeAnchorIdx.push_back(indices_dists[i].first);
            minibatch.wijVal.push_back(rawWeight);
            minibatch.psiVal.push_back(rawWeight);
            weightSum += rawWeight;
        }
        for (size_t e = edgeStart; e < minibatch.psiVal.size(); ++e) {
            minibatch.psiVal[e] /= weightSum;
        }

        smInputMut.coordsFloat[npt] = coord;
        smInputMut.featureIdx[npt] = smInput.featureIdx[i];
        smInputMut.obsWeight[npt] = smInput.obsWeight[i];
        tileData.orgpts2pixel[i] = static_cast<int32_t>(npt);
        ++npt;
        minibatch.rowOffsets.push_back(static_cast<uint32_t>(minibatch.edgeAnchorIdx.size()));
    }
    smInputMut.coordsFloat.resize(npt);
    smInputMut.featureIdx.resize(npt);
    smInputMut.obsWeight.resize(npt);
    minibatch.featureIdx = smInputMut.featureIdx;
    minibatch.featureWeight = std::move(smInputMut.obsWeight);
    anchors = std::move(pc.pts);
    const double avgDegree = (npt > 0)
        ? static_cast<double>(minibatch.edgeAnchorIdx.size()) / static_cast<double>(npt)
        : 0.0;
    debug("%s: created %zu single-molecule edges between %zu rows and %zu anchors, average degree %.2f",
        __func__, minibatch.edgeAnchorIdx.size(), npt, anchors.size(), avgDegree);

    minibatch.N = static_cast<int32_t>(npt);
    minibatch.M = M_;
    return avgDegree;
}

template<typename T>
double Tiles2MinibatchBase<T>::buildMinibatchCore(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    double supportRadius, double distNu) {
    const size_t nObs = isSingleMoleculeMode()
        ? tileData.singleMolecule2D().coordsFloat.size()
        : (useExtended_ ? tileData.extended2D().extPts.size() : tileData.standard2D().pts.size());
    debug("%s: building minibatch with %zu anchors and %zu documents", __func__, anchors.size(), nObs);
    if (minibatch.n <= 0) {
        return 0.0;
    }
    assert(supportRadius > 0.0 && distNu >= 0.0);

    if (coordDim_ == MinibatchCoordDim::Dim3) {
        error("%s: 3D inference requires an explicit buildMinibatchCore3D() call", __func__);
    }

    if (isSingleFeaturePixelMode()) {
        return buildMinibatchCoreSingleFeaturePixel(tileData, anchors, minibatch, supportRadius, distNu);
    }
    if (isSingleMoleculeMode()) {
        return buildMinibatchCoreSingleMolecule(tileData, anchors, minibatch, supportRadius, distNu);
    }
    minibatch.storageMode = Minibatch::StorageMode::GenericSparse;
    minibatch.clearDataSgl();

    PointCloud<float> pc;
    pc.pts = std::move(anchors);
    kd_tree_f2_t kdtree(2, pc, {10});
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;

    const float res = pixelResolution_;
    const float l2radius = static_cast<float>(supportRadius * supportRadius);

    std::vector<Eigen::Triplet<float>> tripletsMtx;
    std::vector<Eigen::Triplet<float>> tripletsWij;
    const size_t standardObs = useExtended_ ? tileData.extended2D().extPts.size() : tileData.standard2D().pts.size();
    tripletsMtx.reserve(standardObs);
    tripletsWij.reserve(standardObs);

    std::unordered_map<uint64_t, std::pair<std::unordered_map<uint32_t, float>, std::vector<uint32_t>>> pixAgg;
    uint32_t idxOriginal = 0;

    if (useExtended_) {
        const auto& extInput = tileData.extended2D();
        tileData.orgpts2pixel.assign(extInput.extPts.size(), -1);
        for (const auto& pt : extInput.extPts) {
            int32_t x = static_cast<int32_t>(std::floor(pt.recBase.x / res));
            int32_t y = static_cast<int32_t>(std::floor(pt.recBase.y / res));
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) | static_cast<uint32_t>(y);
            pixAgg[key].first[pt.recBase.idx] += pt.recBase.ct;
            pixAgg[key].second.push_back(idxOriginal++);
        }
    } else {
        const auto& stdInput2 = tileData.standard2D();
        tileData.orgpts2pixel.assign(stdInput2.pts.size(), -1);
        for (const auto& pt : stdInput2.pts) {
            int32_t x = static_cast<int32_t>(std::floor(pt.x / res));
            int32_t y = static_cast<int32_t>(std::floor(pt.y / res));
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) | static_cast<uint32_t>(y);
            pixAgg[key].first[pt.idx] += pt.ct;
            pixAgg[key].second.push_back(idxOriginal++);
        }
    }

    tileData.coords.clear();
    tileData.coords.reserve(pixAgg.size());
    uint32_t npt = 0;
    for (auto& kv : pixAgg) {
        int32_t px = static_cast<int32_t>(kv.first >> 32);
        int32_t py = static_cast<int32_t>(kv.first & 0xFFFFFFFFu);
        float xy[2] = {px * res, py * res};
        size_t n = kdtree.radiusSearch(xy, l2radius, indices_dists);
        if (n == 0) {
            continue;
        }

        if (lineParserPtr->weighted) {
            for (auto& kv2 : kv.second.first) {
                kv2.second *= static_cast<float>(lineParserPtr->weights[kv2.first]);
                tripletsMtx.emplace_back(npt, static_cast<int>(kv2.first), kv2.second);
            }
        } else {
            for (auto& kv2 : kv.second.first) {
                tripletsMtx.emplace_back(npt, static_cast<int>(kv2.first), kv2.second);
            }
        }

        tileData.coords.emplace_back(px, py);
        for (auto v : kv.second.second) {
            tileData.orgpts2pixel[v] = static_cast<int32_t>(npt);
        }

        for (size_t i = 0; i < n; ++i) {
            uint32_t idx = indices_dists[i].first;
            const float dist = std::sqrt(indices_dists[i].second);
            tripletsWij.emplace_back(
                npt, static_cast<int>(idx),
                anchor_distance_weight(dist, supportRadius, distNu));
        }

        ++npt;
    }
    anchors = std::move(pc.pts);
    double avgDegree = (npt > 0)
        ? static_cast<double>(tripletsWij.size()) / static_cast<double>(npt)
        : 0.0;
    debug("%s: created %zu edges between %zu pixels and %zu anchors, average degree %.2f", __func__, tripletsWij.size(), npt, anchors.size(), avgDegree);

    minibatch.N = static_cast<int32_t>(npt);
    minibatch.M = M_;
    minibatch.mtx.resize(npt, M_);
    minibatch.mtx.setFromTriplets(tripletsMtx.begin(), tripletsMtx.end());
    minibatch.mtx.makeCompressed();

    minibatch.wij.resize(npt, minibatch.n);
    minibatch.wij.setFromTriplets(tripletsWij.begin(), tripletsWij.end());
    minibatch.wij.makeCompressed();

    minibatch.psi = minibatch.wij;
    rowNormalizeInPlace(minibatch.psi);

    return avgDegree;
}

template<typename T>
void Tiles2MinibatchBase<T>::writerWorker() {
    // A priority queue to buffer out-of-order results
    std::priority_queue<ResultBuf, std::vector<ResultBuf>, std::greater<ResultBuf>> outOfOrderBuffer;
    int nextTicketToWrite = 0;
    ResultBuf result;
    // Loop until the queue is marked as done and is empty
    while (resultQueue.pop(result)) {
        outOfOrderBuffer.push(std::move(result));
        // Write all results that are now ready in sequential order
        while (!outOfOrderBuffer.empty() &&
               (!useTicketSystem_ ||
                outOfOrderBuffer.top().ticket == nextTicketToWrite)) {
            const auto& readyToWrite = outOfOrderBuffer.top();
            if (readyToWrite.empty()) {
                outOfOrderBuffer.pop();
                nextTicketToWrite++;
                continue;
            }
            size_t st = outputSize;
            size_t ed = st;
            std::visit([&](const auto& data) {
                using PayloadT = std::decay_t<decltype(data)>;
                if constexpr (std::is_same_v<PayloadT, typename ResultBuf::TextLines>) {
                    for (const auto& line : data) {
                        if (!write_all(fdMain, line.data(), line.size())) {
                            error("Error writing to main output file");
                        }
                        ed += line.size();
                    }
                } else {
                    write_bucketed_binary_payload(fdMain, fdIndex, outputSize,
                        data, outputRecordSize_, tileSize, pixelResolution_);
                    ed = outputSize;
                }
            }, readyToWrite.payload);
            if (readyToWrite.isText()) {
                IndexEntryF e(st, ed, readyToWrite.size(),
                    static_cast<int32_t>(std::floor(readyToWrite.xmin)),
                    static_cast<int32_t>(std::ceil(readyToWrite.xmax)),
                    static_cast<int32_t>(std::floor(readyToWrite.ymin)),
                    static_cast<int32_t>(std::ceil(readyToWrite.ymax)));
                if (!write_all(fdIndex, &e, sizeof(e))) {
                    error("Error writing to index output file");
                }
            }
            outputSize = ed;
            outOfOrderBuffer.pop();
            nextTicketToWrite++;
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::submitPixelResult(ResultBuf&& result, int threadId) {
    if (!nativeBinaryRegularTiles_) {
        resultQueue.push(std::move(result));
        return;
    }
    appendNativeBinaryResult(result, threadId);
}

template<typename T>
IndexHeader Tiles2MinibatchBase<T>::buildIndexHeader(bool fragmented) const {
    IndexHeader idxHeader;
    idxHeader.magic = PUNKST_INDEX_MAGIC;
    idxHeader.mode = (getFactorCount() << 16);
    if (fragmented) {
        idxHeader.mode |= 0x8;
    }
    idxHeader.tileSize = tileSize;
    idxHeader.topK = topk_;
    idxHeader.pixelResolution = usesFloatFeatureCoords() ? -1.0f : pixelResolution_;
    idxHeader.pixelResolutionZ = -1.0f;
    const auto& box = tileReader.getGlobalBox();
    idxHeader.xmin = box.xmin;
    idxHeader.xmax = box.xmax;
    idxHeader.ymin = box.ymin;
    idxHeader.ymax = box.ymax;
    if (outputBinary_) {
        idxHeader.mode |= 0x1;
        if (!usesFloatFeatureCoords()) {
            idxHeader.mode |= 0x6;
        }
        idxHeader.recordSize = outputRecordSize_;
    }
    if (usesFeatureSpecificMode()) {
        idxHeader.mode |= 0x40u;
    }
    if (coordDim_ == MinibatchCoordDim::Dim3) {
        idxHeader.mode |= 0x10;
        if (!usesFloatFeatureCoords()) {
            idxHeader.mode |= 0x20u;
            idxHeader.pixelResolutionZ = pixelResolutionZ_;
        }
    }
    configureFeatureDictionaryHeader(idxHeader, featureNames, __func__);
    return idxHeader;
}

template<typename T>
void Tiles2MinibatchBase<T>::setupNativeBinaryShards() {
    if (!nativeBinaryRegularTiles_) {
        return;
    }
    if (!outputBinary_) {
        error("%s: native regular tile mode currently supports binary output only", __func__);
    }
    if (!tmpDir.enabled) {
        tmpDir.init(std::filesystem::temp_directory_path());
        notice("Created temporary directory: %s", tmpDir.path.string().c_str());
    }
    const size_t nWorkers = static_cast<size_t>(std::max(1, nThreads));
    workerShards_.assign(nWorkers, WorkerShardFiles{});
    for (size_t i = 0; i < nWorkers; ++i) {
        WorkerShardFiles& shard = workerShards_[i];
        shard.mainDataPath = (tmpDir.path / ("worker." + std::to_string(i) + ".main.dat")).string();
        shard.mainIndexPath = (tmpDir.path / ("worker." + std::to_string(i) + ".main.idx")).string();
        shard.boundaryDataPath = (tmpDir.path / ("worker." + std::to_string(i) + ".boundary.dat")).string();
        shard.boundaryIndexPath = (tmpDir.path / ("worker." + std::to_string(i) + ".boundary.idx")).string();
        shard.fdMainData = ::open(shard.mainDataPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        shard.fdMainIndex = ::open(shard.mainIndexPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        shard.fdBoundaryData = ::open(shard.boundaryDataPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        shard.fdBoundaryIndex = ::open(shard.boundaryIndexPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (shard.fdMainData < 0 || shard.fdMainIndex < 0 ||
            shard.fdBoundaryData < 0 || shard.fdBoundaryIndex < 0) {
            error("%s: failed opening native shard files in %s", __func__, tmpDir.path.c_str());
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::closeNativeBinaryShards() {
    for (auto& shard : workerShards_) {
        if (shard.fdMainData >= 0) {
            ::close(shard.fdMainData);
            shard.fdMainData = -1;
        }
        if (shard.fdMainIndex >= 0) {
            ::close(shard.fdMainIndex);
            shard.fdMainIndex = -1;
        }
        if (shard.fdBoundaryData >= 0) {
            ::close(shard.fdBoundaryData);
            shard.fdBoundaryData = -1;
        }
        if (shard.fdBoundaryIndex >= 0) {
            ::close(shard.fdBoundaryIndex);
            shard.fdBoundaryIndex = -1;
        }
    }
}

template<typename T>
std::vector<char> Tiles2MinibatchBase<T>::serializeBinaryResult(const ResultBuf& result) const {
    if (result.isText()) {
        error("%s: native binary serialization expects object records", __func__);
    }
    std::vector<char> bytes;
    bytes.reserve(result.size() * outputRecordSize_);
    std::visit([&](const auto& data) {
        using PayloadT = std::decay_t<decltype(data)>;
        if constexpr (std::is_same_v<PayloadT, typename ResultBuf::TextLines>) {
            error("%s: native binary serialization expects object records", __func__);
        } else {
            for (const auto& obj : data) {
                append_record_bytes(bytes, obj, outputRecordSize_);
            }
        }
    }, result.payload);
    return bytes;
}

template<typename T>
void Tiles2MinibatchBase<T>::appendNativeBinaryResult(const ResultBuf& result, int threadId) {
    if (result.empty()) {
        return;
    }
    const size_t workerId = static_cast<size_t>(threadId);
    if (workerId >= workerShards_.size()) {
        error("%s: invalid worker id %d", __func__, threadId);
    }
    const WorkerOutputContext& ctx = workerOutputContext_[workerId];
    if (ctx.kind == OutputSourceKind::None) {
        error("%s: missing output context for worker %d", __func__, threadId);
    }
    WorkerShardFiles& shard = workerShards_[workerId];
    if (ctx.kind == OutputSourceKind::MainTile) {
        std::vector<char> bytes = serializeBinaryResult(result);
        if (bytes.empty()) {
            return;
        }
        ShardFragmentIndex idx;
        idx.kind = static_cast<uint8_t>(ctx.kind);
        idx.npts = result.size();
        idx.boundaryKey = 0;
        idx.row = ctx.tile.row;
        idx.col = ctx.tile.col;
        idx.dataOffset = shard.mainDataSize;
        idx.dataBytes = bytes.size();
        if (!write_all(shard.fdMainData, bytes.data(), bytes.size()) ||
            !write_all(shard.fdMainIndex, &idx, sizeof(idx))) {
            error("%s: failed writing main shard data for worker %d", __func__, threadId);
        }
        shard.mainDataSize += idx.dataBytes;
    } else if (ctx.kind == OutputSourceKind::Boundary) {
        std::map<TileKey, std::pair<std::vector<char>, uint32_t>> buckets;
        std::visit([&](const auto& data) {
            using PayloadT = std::decay_t<decltype(data)>;
            if constexpr (std::is_same_v<PayloadT, typename ResultBuf::TextLines>) {
                error("%s: boundary native binary output expects object records", __func__);
            } else {
                for (const auto& obj : data) {
                    TileKey tile = boundary_tile_key(obj, tileSize, pixelResolution_);
                    auto& bucket = buckets[tile];
                    append_record_bytes(bucket.first, obj, outputRecordSize_);
                    bucket.second++;
                }
            }
        }, result.payload);
        for (const auto& kv : buckets) {
            const auto& tile = kv.first;
            const auto& bytes = kv.second.first;
            if (bytes.empty()) {
                continue;
            }
            ShardFragmentIndex idx;
            idx.kind = static_cast<uint8_t>(ctx.kind);
            idx.row = tile.row;
            idx.col = tile.col;
            idx.boundaryKey = ctx.boundaryKey;
            idx.npts = kv.second.second;
            idx.dataOffset = shard.boundaryDataSize;
            idx.dataBytes = bytes.size();
            if (!write_all(shard.fdBoundaryData, bytes.data(), bytes.size()) ||
                !write_all(shard.fdBoundaryIndex, &idx, sizeof(idx))) {
                error("%s: failed writing boundary shard data for worker %d", __func__, threadId);
            }
            shard.boundaryDataSize += idx.dataBytes;
        }
    } else {
        error("%s: unsupported output source kind", __func__);
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::mergeNativeBinaryShards() {
    if (!nativeBinaryRegularTiles_) {
        return;
    }

    std::map<TileKey, std::vector<FragmentSpan>> tileMainSpans;
    std::map<TileKey, std::vector<FragmentSpan>> tileBoundarySpans;
    for (size_t shardId = 0; shardId < workerShards_.size(); ++shardId) {
        std::ifstream idxIn(workerShards_[shardId].mainIndexPath, std::ios::binary);
        if (!idxIn.is_open()) {
            error("%s: failed opening shard index %s", __func__, workerShards_[shardId].mainIndexPath.c_str());
        }
        ShardFragmentIndex idx;
        while (idxIn.read(reinterpret_cast<char*>(&idx), sizeof(idx))) {
            if (idx.kind != static_cast<uint8_t>(OutputSourceKind::MainTile)) {
                error("%s: invalid fragment kind in main shard index", __func__);
            }
            TileKey tile{idx.row, idx.col};
            tileMainSpans[tile].push_back(FragmentSpan{shardId, idx.dataOffset, idx.dataBytes, idx.npts});
        }
    }
    for (size_t shardId = 0; shardId < workerShards_.size(); ++shardId) {
        std::ifstream idxIn(workerShards_[shardId].boundaryIndexPath, std::ios::binary);
        if (!idxIn.is_open()) {
            error("%s: failed opening boundary shard index %s", __func__, workerShards_[shardId].boundaryIndexPath.c_str());
        }
        ShardFragmentIndex idx;
        while (idxIn.read(reinterpret_cast<char*>(&idx), sizeof(idx))) {
            if (idx.kind != static_cast<uint8_t>(OutputSourceKind::Boundary)) {
                error("%s: invalid fragment kind in boundary shard index", __func__);
            }
            if (idx.dataBytes == 0 || idx.npts == 0) {
                continue;
            }
            TileKey tile{idx.row, idx.col};
            tileBoundarySpans[tile].push_back(FragmentSpan{shardId, idx.dataOffset, idx.dataBytes, idx.npts});
        }
    }

    std::string outFile = outPref + ".bin";
    std::string outIndex = outPref + ".index";
    int fdOut = ::open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    int fdIdx = ::open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdOut < 0 || fdIdx < 0) {
        if (fdOut >= 0) ::close(fdOut);
        if (fdIdx >= 0) ::close(fdIdx);
        error("%s: failed opening final output %s / %s", __func__, outFile.c_str(), outIndex.c_str());
    }
    IndexHeader idxHeader = buildIndexHeader(false);
    if (!write_all(fdIdx, &idxHeader, sizeof(idxHeader))) {
        ::close(fdOut);
        ::close(fdIdx);
        error("%s: failed writing final index header", __func__);
    }
    if (!writeFeatureDictionaryPayload(fdIdx, idxHeader, featureNames)) {
        ::close(fdOut);
        ::close(fdIdx);
        error("%s: failed writing embedded feature dictionary", __func__);
    }

    std::vector<std::ifstream> mainInputs;
    std::vector<std::ifstream> boundaryInputs;
    mainInputs.reserve(workerShards_.size());
    boundaryInputs.reserve(workerShards_.size());
    for (const auto& shard : workerShards_) {
        mainInputs.emplace_back(shard.mainDataPath, std::ios::binary);
        if (!mainInputs.back().is_open()) {
            ::close(fdOut);
            ::close(fdIdx);
            error("%s: failed opening main shard data %s", __func__, shard.mainDataPath.c_str());
        }
        boundaryInputs.emplace_back(shard.boundaryDataPath, std::ios::binary);
        if (!boundaryInputs.back().is_open()) {
            ::close(fdOut);
            ::close(fdIdx);
            error("%s: failed opening boundary shard data %s", __func__, shard.boundaryDataPath.c_str());
        }
    }

    auto copySpan = [&](std::ifstream& in, const FragmentSpan& span) {
        static constexpr size_t kBufSize = 1 << 20;
        std::vector<char> buf(kBufSize);
        in.clear();
        in.seekg(static_cast<std::streamoff>(span.dataOffset));
        if (!in.good()) {
            error("%s: failed seeking shard input", __func__);
        }
        uint64_t copied = 0;
        while (copied < span.dataBytes) {
            const size_t toRead = static_cast<size_t>(
                std::min<uint64_t>(static_cast<uint64_t>(buf.size()), span.dataBytes - copied));
            if (!in.read(buf.data(), static_cast<std::streamsize>(toRead))) {
                error("%s: failed reading shard payload", __func__);
            }
            if (!write_all(fdOut, buf.data(), toRead)) {
                error("%s: failed writing final payload", __func__);
            }
            copied += static_cast<uint64_t>(toRead);
        }
    };

    std::set<TileKey> allTiles;
    std::vector<TileKey> sourceTiles;
    tileReader.getTileList(sourceTiles);
    for (const auto& tile : sourceTiles) {
        allTiles.insert(tile);
    }
    for (const auto& kv : tileMainSpans) {
        allTiles.insert(kv.first);
    }
    for (const auto& kv : tileBoundarySpans) {
        allTiles.insert(kv.first);
    }

    uint64_t currentOffset = 0;
    int32_t nTiles = 0;
    for (const auto& tile : allTiles) {
        IndexEntryF outEntry(tile.row, tile.col);
        outEntry.st = currentOffset;
        outEntry.n = 0;
        tile2bound(tile, outEntry.xmin, outEntry.xmax, outEntry.ymin, outEntry.ymax, tileSize);

        auto mainIt = tileMainSpans.find(tile);
        if (mainIt != tileMainSpans.end()) {
            for (const auto& span : mainIt->second) {
                copySpan(mainInputs[span.shardId], span);
                currentOffset += span.dataBytes;
                outEntry.n += span.npts;
            }
        }
        auto boundaryIt = tileBoundarySpans.find(tile);
        if (boundaryIt != tileBoundarySpans.end()) {
            for (const auto& span : boundaryIt->second) {
                copySpan(boundaryInputs[span.shardId], span);
                currentOffset += span.dataBytes;
                outEntry.n += span.npts;
            }
        }

        outEntry.ed = currentOffset;
        if (outEntry.n > 0) {
            if (!write_all(fdIdx, &outEntry, sizeof(outEntry))) {
                ::close(fdOut);
                ::close(fdIdx);
                error("%s: failed writing final tile index", __func__);
            }
            ++nTiles;
        }
    }

    ::close(fdOut);
    ::close(fdIdx);
    notice("%s: native regularization wrote %d tiles to %s", __func__, nTiles, outFile.c_str());
}

template<typename T>
void Tiles2MinibatchBase<T>::anchorWriterWorker() {
    if (fdAnchor < 0) return;
    std::priority_queue<ResultBuf, std::vector<ResultBuf>, std::greater<ResultBuf>> outOfOrderBuffer;
    int nextTicketToWrite = 0;
    ResultBuf result;
    while (anchorQueue.pop(result)) {
        outOfOrderBuffer.push(std::move(result));
        while (!outOfOrderBuffer.empty() &&
               (!useTicketSystem_ ||
                outOfOrderBuffer.top().ticket == nextTicketToWrite)) {
            const auto& readyToWrite = outOfOrderBuffer.top();
            if (readyToWrite.empty()) {
                outOfOrderBuffer.pop();
                nextTicketToWrite++;
                continue;
            }
            size_t totalLen = 0;
            std::visit([&](const auto& data) {
                using PayloadT = std::decay_t<decltype(data)>;
                if constexpr (std::is_same_v<PayloadT, typename ResultBuf::TextLines>) {
                    for (const auto& line : data) {
                        if (!write_all(fdAnchor, line.data(), line.size())) {
                            error("Error writing to anchor output file");
                        }
                        totalLen += line.size();
                    }
                } else {
                    for (const auto& obj : data) {
                        int32_t s0 = obj.write(fdAnchor);
                        if (s0 < 0) {
                            error("Error writing to anchor output file");
                        }
                        totalLen += s0;
                    }
                }
            }, readyToWrite.payload);
            anchorOutputSize += totalLen;
            outOfOrderBuffer.pop();
            nextTicketToWrite++;
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::setExtendedSchema(size_t offset) {
    if (!lineParserPtr->isExtended) {
        useExtended_ = false; return;
    }
    useExtended_ = true;
    schema_.clear();
    size_t n_ints = lineParserPtr->icol_ints.size();
    size_t n_floats = lineParserPtr->icol_floats.size();
    size_t n_strs = lineParserPtr->icol_strs.size();
    for (size_t i = 0; i < n_ints; ++i)
        schema_.push_back({FieldType::INT32, sizeof(int32_t), 0});
    for (size_t i = 0; i < n_floats; ++i)
        schema_.push_back({FieldType::FLOAT, sizeof(float), 0});
    for (size_t i = 0; i < n_strs; ++i)
        schema_.push_back({FieldType::STRING, lineParserPtr->str_lens[i], 0});
    for (auto &f : schema_) {
        f.offset = offset; offset += f.size;
    }
    recordSize_ = offset;
}


template<typename T>
void Tiles2MinibatchBase<T>::closeOutput() {
    if (fdMain >= 0) { ::close(fdMain); fdMain = -1; }
    if (fdIndex >= 0) { ::close(fdIndex); fdIndex = -1; }
    if (fdAnchor >= 0) { ::close(fdAnchor); fdAnchor = -1; }
    closeNativeBinaryShards();
}

// Include template implementations to keep definitions in one TU.
#include "tiles2minibatch_io.cpp"
#include "tiles2minibatch_3d.cpp"

template class Tiles2MinibatchBase<int32_t>;
template class Tiles2MinibatchBase<float>;
