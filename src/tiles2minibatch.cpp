#include "tiles2minibatch.hpp"
#include "numerical_utils.hpp"
#include <cmath>
#include <numeric>
#include <fcntl.h>
#include <thread>
#include <algorithm>

template<typename T>
void Tiles2MinibatchBase<T>::run() {
    setupOutput();
    std::thread writer(&Tiles2MinibatchBase<T>::writerWorker, this);
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

    resultQueue.set_done();
    if (outputAnchor_) {
        anchorQueue.set_done();
    }
    // notice("%s: all workers done, waiting for writer to finish", __func__);

    writer.join();
    if (anchorWriter.joinable()) {
        anchorWriter.join();
    }
    closeOutput();
    postRun();
}

template<typename T>
void Tiles2MinibatchBase<T>::configureInputMode() {
    if (!lineParserPtr) {
        error("%s: lineParser is required", __func__);
    }
    if (lineParserPtr->hasZCoord()) {
        coordDim_ = MinibatchCoordDim::Dim3;
    } else if (coordDim_ == MinibatchCoordDim::Dim3) {
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
    std::pair<TileKey, int32_t> tileTicket;
    TileKey tile;
    int32_t ticket;
    while (tileQueue.pop(tileTicket)) {
        tile = tileTicket.first;
        ticket = tileTicket.second;
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
    std::pair<std::shared_ptr<BoundaryBuffer>, int32_t> bufferTicket;
    std::shared_ptr<BoundaryBuffer> bufferPtr;
    int32_t ticket;
    while (bufferQueue.pop(bufferTicket)) {
        bufferPtr = bufferTicket.first;
        ticket = bufferTicket.second;
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
int32_t Tiles2MinibatchBase<T>::buildAnchors(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, HexGrid& hexGrid_, int32_t nMoves_, double minCount) {
    if (coordDim_ == MinibatchCoordDim::Dim3) {
        return buildAnchors3D(tileData, anchors, documents, hexGrid_, nMoves_, minCount);
    }
    anchors.clear();
    documents.clear();
    std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, std::unordered_map<uint32_t, float>> hexAggregation;
    auto assign_pt = [&](const auto& pt) {
        for (int32_t ir = 0; ir < nMoves_; ++ir) {
            for (int32_t ic = 0; ic < nMoves_; ++ic) {
                int32_t hx, hy;
                hexGrid_.cart_to_axial(hx, hy, pt.x, pt.y, ic * 1. / nMoves_, ir * 1. / nMoves_);
                auto key = std::make_tuple(hx, hy, ic, ir);
                hexAggregation[key][pt.idx] += pt.ct;
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
        int32_t hx = std::get<0>(key);
        int32_t hy = std::get<1>(key);
        int32_t ic = std::get<2>(key);
        int32_t ir = std::get<3>(key);
        float x, y;
        hexGrid_.axial_to_cart(x, y, hx, hy, ic * 1. / nMoves_, ir * 1. / nMoves_);
        anchors.emplace_back(x, y);
    }
    return documents.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseOneTile(TileData<T>& tileData, TileKey tile) {
    return (this->*parseTileFn_)(tileData, tile);
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::buildMinibatchCore(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    double distR, double distNu) {

    if (minibatch.n <= 0) {
        return 0;
    }
    assert(distR > 0.0 && distNu > 0.0);

    if (coordDim_ == MinibatchCoordDim::Dim3) {
        return buildMinibatchCore3D(tileData, anchors, minibatch, distR, distNu);
    }

    PointCloud<float> pc;
    pc.pts = std::move(anchors);
    kd_tree_f2_t kdtree(2, pc, {10});
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;

    const float res = pixelResolution_;
    const float radius = static_cast<float>(distR);
    const float l2radius = radius * radius;
    const float nu = static_cast<float>(distNu);

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
            float dist = std::pow(indices_dists[i].second, 0.5f);
            dist = std::max(std::min(1.f - std::pow(dist / radius, nu), 0.95f), 0.05f);
            tripletsWij.emplace_back(npt, static_cast<int>(idx), dist);
        }

        ++npt;
    }
    anchors = std::move(pc.pts);

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

    return minibatch.N;
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
            if (readyToWrite.npts > 0) {
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
            }
            outOfOrderBuffer.pop();
            nextTicketToWrite++;
        }
    }
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
            if (readyToWrite.npts > 0) {
                size_t totalLen = 0;
                if (readyToWrite.useObj) {
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
                if (totalLen > 0 && anchorHeaderSize > 0 && anchorOutputSize == 0) {
                    anchorOutputSize += anchorHeaderSize;
                }
                anchorOutputSize += totalLen;
            }
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
}

// Include template implementations to keep definitions in one TU.
#include "tiles2minibatch_io.cpp"
#include "tiles2minibatch_3d.cpp"

template class Tiles2MinibatchBase<int32_t>;
template class Tiles2MinibatchBase<float>;
