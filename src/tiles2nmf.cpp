#include "tiles2nmf.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <cstdint>
#include <utility>
#include <unordered_map>

template<typename T>
Tiles2NMF<T>::Tiles2NMF(int nThreads, double r,
        const std::string& outPref, const std::string& tmpDir,
        PoissonLog1pNMF& nmf, EMPoisReg& empois,
        TileReader& tileReader, lineParserUnival& lineParser,
        HexGrid& hexGrid, int32_t nMoves,
        unsigned int seed,
        double c, double h, double res, int32_t topk,
        int32_t verbose, int32_t debug)
    : Tiles2MinibatchBase(nThreads, r + hexGrid.size, tileReader, outPref, &tmpDir), distR_(r),
        nmf_(nmf), empois_(empois),
        lineParser_(lineParser),
        hexGrid_(hexGrid), nMoves_(nMoves),
        seed_(seed), anchorMinCount_(c),
        pixelResolution_(static_cast<float>(res)),
        debug_(debug), verbose_(verbose)
{
    lineParserPtr = &lineParser_;
    useExtended_ = lineParser_.isExtended;
    topk_ = topk;

    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        assert(tileReader.getCoordType() == CoordType::FLOAT && "Template type does not match TileReader coordinate type");
    } else if constexpr (std::is_same_v<T, int32_t>) {
        assert(tileReader.getCoordType() == CoordType::INTEGER && "Template type does not match TileReader coordinate type");
    } else {
        error("%s: Unsupported coordinate type", __func__);
    }

    if (h <= 0.0 || h >= 1.0) {
        error("%s: smoothing parameter h must be in (0, 1)", __func__);
    }
    distNu_ = std::log(0.5) / std::log(h);

    if (pixelResolution_ <= 0.f) {
        pixelResolution_ = 1.f;
    }

    if (useExtended_) {
         Tiles2MinibatchBase::setExtendedSchema(sizeof(RecordT<T>));
    }

    nmf_.set_nthreads(1);
    const RowMajorMatrixXd& beta = nmf_.get_model();
    M_ = static_cast<int32_t>(beta.rows());
    K_ = static_cast<int32_t>(beta.cols());
    if (M_ <= 0 || K_ <= 0) {
        error("%s: Invalid beta dimensions (%d x %d)", __func__, M_, K_);
    }

    empois_.init_global_parameter(beta);

    if (lineParser_.isFeatureDict) {
        if (static_cast<int32_t>(lineParser_.featureDict.size()) != M_) {
            error("%s: feature dictionary size mismatch (%zu vs %d)",
                  __func__, lineParser_.featureDict.size(), M_);
        }
        featureNames.resize(M_);
        for (const auto& entry : lineParser_.featureDict) {
            featureNames[entry.second] = entry.first;
        }
    } else if (lineParser_.weighted) {
        if (static_cast<int32_t>(lineParser_.weights.size()) != M_) {
            error("%s: feature weight size mismatch (%zu vs %d)",
                  __func__, lineParser_.weights.size(), M_);
        }
    }

    if (featureNames.empty()) {
        featureNames.resize(M_);
        for (int32_t i = 0; i < M_; ++i) {
            featureNames[i] = std::to_string(i);
        }
    }

    notice("Initialized Tiles2NMF");
}

template<typename T>
int32_t Tiles2NMF<T>::makeMinibatch(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {
    int32_t nPixels = buildMinibatchCore(
        tileData, anchors, minibatch,
        pixelResolution_, distR_, distNu_);
    if (nPixels <= 0) {
        return nPixels;
    }

    minibatch.psi = minibatch.psi.transpose(); // n x N
    minibatch.psi.makeCompressed();

    minibatch.wij.unaryExpr([](float val) {return log(val);});
    minibatch.wij = minibatch.wij.transpose();
    minibatch.wij.makeCompressed();

    return nPixels;
}

template<typename T>
int32_t Tiles2NMF<T>::initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {
    std::vector<SparseObs> documents;
    Tiles2MinibatchBase::buildAnchors(tileData, anchors, documents, minibatch, hexGrid_, nMoves_, anchorMinCount_);
    if (documents.empty()) {
        return 0;
    }
    for (auto& docObs : documents) {
        docObs.c = docObs.ct_tot / empois_.size_factor_;
    }

    std::vector<MLEStats> stats;
    minibatch.theta = nmf_.transform(documents, empois_.mle_opts_, stats).template cast<float>();
    minibatch.n = documents.size();
    return anchors.size();
}

template<typename T>
void Tiles2NMF<T>::processTile(TileData<T>& tileData, int threadId, int ticket) {
    if (tileData.pts.empty() && tileData.extPts.empty()) {
        return;
    }
    std::vector<cv::Point2f> anchors;
    Minibatch minibatch;
    int32_t nAnchors = initAnchors(tileData, anchors, minibatch);
    debug("%s: Thread %d (ticket %d) initialized %d anchors", __func__, threadId, ticket, nAnchors);
    if (nAnchors <= 0) {
        return;
    }
    int32_t nPixels = makeMinibatch(tileData, anchors, minibatch);
    debug("%s: Thread %d (ticket %d) made minibatch with %d pixels", __func__, threadId, ticket, nPixels);
    if (nPixels < 10) {
        return;
    }
    empois_.run_em(minibatch, max_iter_, tol_);
    debug("%s: Thread %d (ticket %d) finished EM", __func__, threadId, ticket);

    MatrixXf topVals;
    Eigen::MatrixXi topIds;
    findTopK(topVals, topIds, minibatch.phi, topk_);
    ProcessedResult result;
    if (outputOriginalData) {
        result = Tiles2MinibatchBase::formatPixelResultWithOriginalData(tileData, topVals, topIds, ticket);
    } else {
        result = Tiles2MinibatchBase::formatPixelResult(tileData, topVals, topIds, ticket);
    }
    notice("Thread %d (ticket %d) fit minibatch with %d anchors and output %lu internal pixels", threadId, ticket, nAnchors, result.npts);
    resultQueue.push(std::move(result));
}

template<typename T>
void Tiles2NMF<T>::tileWorker(int threadId) {
    std::pair<TileKey, int32_t> tileTicket;
    TileKey tile;
    int32_t ticket;
    while (tileQueue.pop(tileTicket)) {
        tile = tileTicket.first;
        ticket = tileTicket.second;
        TileData<T> tileData;
        int32_t ret = Tiles2MinibatchBase::parseOneTile<T>(tileData, tile);
        notice("%s: Thread %d (ticket %d) read tile (%d, %d) with %d internal pixels", __func__, threadId, ticket, tile.row, tile.col, ret);
        if (ret <= 10) {
            continue;
        }
        processTile(tileData, threadId, ticket);
    }
}

template<typename T>
void Tiles2NMF<T>::boundaryWorker(int threadId) {
    std::pair<std::shared_ptr<BoundaryBuffer>, int32_t> bufferTicket;
    std::shared_ptr<BoundaryBuffer> bufferPtr;
    int32_t ticket;
    while (bufferQueue.pop(bufferTicket)) {
        bufferPtr = bufferTicket.first;
        ticket = bufferTicket.second;
        TileData<T> tileData;
        int32_t ret = 0;

        if (auto* storagePtr = std::get_if<std::unique_ptr<IBoundaryStorage>>(&(bufferPtr->storage))) {
            // --- IN-MEMORY PATH ---
            if (auto* extStore = dynamic_cast<InMemoryStorageExtended<T>*>(storagePtr->get())) {
                ret = Tiles2MinibatchBase::parseBoundaryMemoryExtended(tileData, extStore, bufferPtr->key);
            } else if (auto* stdStore = dynamic_cast<InMemoryStorageStandard<T>*>(storagePtr->get())) {
                ret = Tiles2MinibatchBase::parseBoundaryMemoryStandard(tileData, stdStore, bufferPtr->key);
            }
        } else if (auto* filePath = std::get_if<std::string>(&(bufferPtr->storage))) {
            // --- DISK I/O PATH ---
            if (useExtended_) {
                ret = Tiles2MinibatchBase::parseBoundaryFileExtended(tileData, bufferPtr);
            } else {
                ret = Tiles2MinibatchBase::parseBoundaryFile(tileData, bufferPtr);
            }
            // Clean up the temporary file
            std::remove(filePath->c_str());
        }
        notice("%s: Thread %d (ticket %d) read boundary buffer (%d) with %d internal pixels", __func__, threadId, ticket, bufferPtr->key, ret);
        processTile(tileData, threadId, ticket);
    }
}

template<typename T>
void Tiles2NMF<T>::run() {
    setupOutput();
    std::thread writer(&Tiles2NMF::writerWorker, this);

    // Phase 1: Process tiles
    notice("Phase 1 Launching %d worker threads", nThreads);
    for (int i = 0; i < nThreads; ++i) {
        workThreads.push_back(std::thread(&Tiles2NMF::tileWorker, this, i));
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
        workThreads.push_back(std::thread(&Tiles2NMF::boundaryWorker, this, i));
    }
    for (auto &t : workThreads) {
        t.join();
    }

    resultQueue.set_done();
    notice("%s: all workers done, waiting for writer to finish", __func__);

    writer.join();
    closeOutput();
    writeHeaderToJson();
}

// explicit instantiations
template class Tiles2NMF<int32_t>;
template class Tiles2NMF<float>;
