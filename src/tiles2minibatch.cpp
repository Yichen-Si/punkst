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

float anchor_distance_weight(double dist, double refDist, double weightAtRefDist) {
    if (!(refDist > 0)) {
        error("%s: reference anchor distance must be positive", __func__);
    }
    if (!(weightAtRefDist > 0.0 && weightAtRefDist < 1.0)) {
        error("%s: weight-at-anchor-dist must be in (0, 1)", __func__);
    }
    const double weight = std::exp(std::log(weightAtRefDist) * dist / refDist);
    return static_cast<float>(std::clamp(weight, 1e-4, 1.0 - 1e-4));
}

} // namespace

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
            parseTileFn_ = &Tiles2MinibatchBase::parseOneTileStandard3D;
            parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFile3D;
            parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemoryStandard3DWrapper;
        } else {
            parseTileFn_ = &Tiles2MinibatchBase::parseOneTileStandard;
            parseBoundaryFileFn_ = &Tiles2MinibatchBase::parseBoundaryFile;
            parseBoundaryMemoryFn_ = &Tiles2MinibatchBase::parseBoundaryMemoryStandardWrapper;
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::configureOutputMode() {
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
void Tiles2MinibatchBase<T>::forEachAnchorCandidate2D(const TileData<T>& tileData, const HexGrid& hexGrid_, int32_t nMoves_,
    const std::function<void(uint32_t, float, const AnchorKey2D&)>& emit) const
{
    auto assign_pt = [&](const auto& pt) {
        for (int32_t ir = 0; ir < nMoves_; ++ir) {
            for (int32_t ic = 0; ic < nMoves_; ++ic) {
                int32_t hx, hy;
                hexGrid_.cart_to_axial(hx, hy, pt.x, pt.y, ic * 1.0 / nMoves_, ir * 1.0 / nMoves_);
                emit(pt.idx, static_cast<float>(pt.ct), AnchorKey2D{hx, hy, ic, ir});
            }
        }
    };
    if (useExtended_) {
        for (const auto& pt : tileData.extPts) {
            assign_pt(pt.recBase);
        }
    } else {
        for (const auto& pt : tileData.pts) {
            assign_pt(pt);
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
            return buildAnchorsThin3D(tileData, anchors, documents, hexGrid_, nMoves_, minCount);
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
double Tiles2MinibatchBase<T>::buildMinibatchCore(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    double supportRadius, double refDist, double weightAtRefDist) {
    debug("%s: building minibatch with %zu anchors and %zu documents", __func__, anchors.size(), tileData.pts.size() + tileData.extPts.size());
    if (minibatch.n <= 0) {
        return 0.0;
    }
    assert(supportRadius > 0.0 && refDist > 0.0 && weightAtRefDist > 0.0 && weightAtRefDist < 1.0);

    if (coordDim_ == MinibatchCoordDim::Dim3) {
        return buildMinibatchCore3D(tileData, anchors, minibatch, supportRadius, refDist, weightAtRefDist);
    }

    PointCloud<float> pc;
    pc.pts = std::move(anchors);
    kd_tree_f2_t kdtree(2, pc, {10});
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;

    const float res = pixelResolution_;
    const float l2radius = static_cast<float>(supportRadius * supportRadius);

    std::vector<Eigen::Triplet<float>> tripletsMtx;
    std::vector<Eigen::Triplet<float>> tripletsWij;
    tripletsMtx.reserve(tileData.pts.size() + tileData.extPts.size());
    tripletsWij.reserve(tileData.pts.size() + tileData.extPts.size());

    std::unordered_map<uint64_t, std::pair<std::unordered_map<uint32_t, float>, std::vector<uint32_t>>> pixAgg;
    uint32_t idxOriginal = 0;

    if (useExtended_) {
        tileData.orgpts2pixel.assign(tileData.extPts.size(), -1);
        for (const auto& pt : tileData.extPts) {
            int32_t x = static_cast<int32_t>(std::floor(pt.recBase.x / res));
            int32_t y = static_cast<int32_t>(std::floor(pt.recBase.y / res));
            uint64_t key = (static_cast<uint64_t>(static_cast<uint32_t>(x)) << 32) | static_cast<uint32_t>(y);
            pixAgg[key].first[pt.recBase.idx] += pt.recBase.ct;
            pixAgg[key].second.push_back(idxOriginal++);
        }
    } else {
        tileData.orgpts2pixel.assign(tileData.pts.size(), -1);
        for (const auto& pt : tileData.pts) {
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
                anchor_distance_weight(dist, refDist, weightAtRefDist));
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
            if (readyToWrite.npts == 0) {
                outOfOrderBuffer.pop();
                nextTicketToWrite++;
                continue;
            }
            size_t st = outputSize;
            size_t ed = st;
            if (readyToWrite.useObj) {
                if (coordDim_ == MinibatchCoordDim::Dim3) {
                    for (const auto& obj : readyToWrite.outputObjs3d) {
                        int32_t s0 = obj.write(fdMain);
                        if (s0 < 0) {
                            error("Error writing to main output file");
                        }
                        ed += s0;
                    }
                } else {
                    for (const auto& obj : readyToWrite.outputObjs) {
                        int32_t s0 = obj.write(fdMain);
                        if (s0 < 0) {
                            error("Error writing to main output file");
                        }
                        ed += s0;
                    }
                }
            } else {
                for (const auto& line : readyToWrite.outputLines) {
                    if (!write_all(fdMain, line.data(), line.size())) {
                        error("Error writing to main output file");
                    }
                    ed += line.size();
                }
            }
            IndexEntryF e(st, ed, readyToWrite.npts,
                (int32_t) std::floor(readyToWrite.xmin),
                (int32_t) std::ceil(readyToWrite.xmax),
                (int32_t) std::floor(readyToWrite.ymin),
                (int32_t) std::ceil(readyToWrite.ymax));
            if (!write_all(fdIndex, &e, sizeof(e))) {
                error("Error writing to index output file");
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
    idxHeader.pixelResolution = pixelResolution_;
    idxHeader.pixelResolutionZ = pixelResolution_;
    const auto& box = tileReader.getGlobalBox();
    idxHeader.xmin = box.xmin;
    idxHeader.xmax = box.xmax;
    idxHeader.ymin = box.ymin;
    idxHeader.ymax = box.ymax;
    if (outputBinary_) {
        idxHeader.mode |= 0x7;
        idxHeader.recordSize = outputRecordSize_;
    }
    if (coordDim_ == MinibatchCoordDim::Dim3) {
        idxHeader.mode |= 0x10;
        idxHeader.mode |= 0x20u;
        idxHeader.pixelResolutionZ = pixelResolutionZ_;
    }
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
    if (!result.useObj) {
        error("%s: native binary serialization expects object records", __func__);
    }
    std::vector<char> bytes;
    bytes.reserve(static_cast<size_t>(result.npts) * outputRecordSize_);
    auto appendObjectBytes2D = [&](const PixTopProbs<int32_t>& obj) {
        const size_t off = bytes.size();
        bytes.resize(off + outputRecordSize_);
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
    };
    auto appendObjectBytes3D = [&](const PixTopProbs3D<int32_t>& obj) {
        const size_t off = bytes.size();
        bytes.resize(off + outputRecordSize_);
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
    };
    if (coordDim_ == MinibatchCoordDim::Dim3) {
        for (const auto& obj : result.outputObjs3d) {
            appendObjectBytes3D(obj);
        }
    } else {
        for (const auto& obj : result.outputObjs) {
            appendObjectBytes2D(obj);
        }
    }
    return bytes;
}

template<typename T>
void Tiles2MinibatchBase<T>::appendNativeBinaryResult(const ResultBuf& result, int threadId) {
    if (result.npts == 0) {
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
    auto appendObjectBytes2D = [&](std::vector<char>& bytes, const PixTopProbs<int32_t>& obj) {
        const size_t off = bytes.size();
        bytes.resize(off + outputRecordSize_);
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
    };
    auto appendObjectBytes3D = [&](std::vector<char>& bytes, const PixTopProbs3D<int32_t>& obj) {
        const size_t off = bytes.size();
        bytes.resize(off + outputRecordSize_);
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
    };
    if (ctx.kind == OutputSourceKind::MainTile) {
        std::vector<char> bytes = serializeBinaryResult(result);
        if (bytes.empty()) {
            return;
        }
        ShardFragmentIndex idx;
        idx.kind = static_cast<uint8_t>(ctx.kind);
        idx.npts = result.npts;
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
        if (coordDim_ == MinibatchCoordDim::Dim3) {
            for (const auto& obj : result.outputObjs3d) {
                const float x = static_cast<float>(obj.x) * pixelResolution_;
                const float y = static_cast<float>(obj.y) * pixelResolution_;
                TileKey tile{
                    static_cast<int32_t>(std::floor(y / tileSize)),
                    static_cast<int32_t>(std::floor(x / tileSize))
                };
                auto& bucket = buckets[tile];
                appendObjectBytes3D(bucket.first, obj);
                bucket.second++;
            }
        } else {
            for (const auto& obj : result.outputObjs) {
                const float x = static_cast<float>(obj.x) * pixelResolution_;
                const float y = static_cast<float>(obj.y) * pixelResolution_;
                TileKey tile{
                    static_cast<int32_t>(std::floor(y / tileSize)),
                    static_cast<int32_t>(std::floor(x / tileSize))
                };
                auto& bucket = buckets[tile];
                appendObjectBytes2D(bucket.first, obj);
                bucket.second++;
            }
        }
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
            if (readyToWrite.npts == 0) {
                outOfOrderBuffer.pop();
                nextTicketToWrite++;
                continue;
            }
            size_t totalLen = 0;
            if (readyToWrite.useObj) { // currently always false
                if (coordDim_ == MinibatchCoordDim::Dim3) {
                    for (const auto& obj : readyToWrite.outputObjs3d) {
                        int32_t s0 = obj.write(fdAnchor);
                        if (s0 < 0) {
                            error("Error writing to anchor output file");
                        }
                        totalLen += s0;
                    }
                } else {
                    for (const auto& obj : readyToWrite.outputObjs) {
                        int32_t s0 = obj.write(fdAnchor);
                        if (s0 < 0) {
                            error("Error writing to anchor output file");
                        }
                        totalLen += s0;
                    }
                }
            } else {
                for (const auto& line : readyToWrite.outputLines) {
                    if (!write_all(fdAnchor, line.data(), line.size())) {
                        error("Error writing to anchor output file");
                    }
                    totalLen += line.size();
                }
            }
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
