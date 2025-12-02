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
            if (auto* extStore = dynamic_cast<InMemoryStorageExtended<T>*>(storagePtr->get())) {
                ret = parseBoundaryMemoryExtended(tileData, extStore, bufferPtr->key);
            } else if (auto* stdStore = dynamic_cast<InMemoryStorageStandard<T>*>(storagePtr->get())) {
                ret = parseBoundaryMemoryStandard(tileData, stdStore, bufferPtr->key);
            }
        } else if (auto* filePath = std::get_if<std::string>(&(bufferPtr->storage))) {
            if (useExtended_) {
                ret = parseBoundaryFileExtended(tileData, bufferPtr);
            } else {
                ret = parseBoundaryFile(tileData, bufferPtr);
            }
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
        if (line.empty()) continue;
        split(tokens, "\t", line);
        if (tokens.size() < 2) {
            error("Error reading anchors file at line: %s", line.c_str());
        }
        float x, y;
        bool valid = str2float(tokens[0], x) && str2float(tokens[1], y);
        if (!valid) {
            error("Error reading anchors file at line: %s", line.c_str());
        }
        TileKey tile;
        if (!tileReader.pt2tile(x, y, tile)) {
            continue;
        }
        fixedAnchorForTile[tile].emplace_back(std::vector<float>{x, y});
        std::vector<uint32_t> bufferidx;
        int32_t ret = pt2buffer(bufferidx, x, y, tile);
        (void)ret;
        for (const auto& key : bufferidx) {
            fixedAnchorForBoundary[key].emplace_back(std::vector<float>{x, y});
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
int32_t Tiles2MinibatchBase<T>::buildAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, std::vector<SparseObs>& documents, HexGrid& hexGrid_, int32_t nMoves_, double minCount) {
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
        float sum = std::accumulate(entry.second.begin(), entry.second.end(), 0.0, [](float acc, const auto& p) { return acc + p.second; });
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
        // Unpack the key to get hex coordinates and move indices
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
    std::unique_ptr<BoundedReadline> iter;
    try {
        iter = tileReader.get_tile_iterator(tile.row, tile.col);
    } catch (const std::exception &e) {
        warning("%s", e.what());
        return -1;
    }
    tileData.clear();
    tile2bound(tile, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    if (M_ == 0) {
        M_ = lineParserPtr->featureDict.size();
    }

    std::string line;
    int32_t npt = 0;
    if (useExtended_) {
        while (iter->next(line)) {
            RecordExtendedT<T> recExt;
            int32_t idx = lineParserPtr->parse(recExt, line);
            if (idx < -1) {
                error("Error parsing line: %s", line.c_str());
            }
            if (idx == -1 || idx >= M_) {
                continue;
            }
            tileData.extPts.push_back(recExt);
            std::vector<uint32_t> bufferidx;
            if (pt2buffer(bufferidx, recExt.recBase.x, recExt.recBase.y, tile) == 1) {
                tileData.idxinternal.push_back(npt);
            }
            for (const auto& key : bufferidx) {
                tileData.extBuffers[key].push_back(recExt);
            }
            npt++;
        }
        for (const auto& entry : tileData.extBuffers) {
            auto buffer = getBoundaryBuffer(entry.first);
            buffer->addRecordsExtended(entry.second, schema_, recordSize_);
        }
        tileData.extBuffers.clear();
    } else {
        while (iter->next(line)) {
            RecordT<T> rec;
            int32_t idx = lineParserPtr->parse<T>(rec, line);
            if (idx < -1) {
                error("Error parsing line: %s", line.c_str());
            }
            if (idx == -1 || idx >= M_) {
                continue;
            }
            tileData.pts.push_back(rec);
            std::vector<uint32_t> bufferidx;
            if (pt2buffer(bufferidx, rec.x, rec.y, tile) == 1) {
                tileData.idxinternal.push_back(npt);
            }
            for (const auto& key : bufferidx) {
                tileData.buffers[key].push_back(rec);
            }
            npt++;
        }
        // write buffered records to temporary files
        for (const auto& entry : tileData.buffers) {
            auto buffer = getBoundaryBuffer(entry.first);
            buffer->addRecords(entry.second);
        }
        tileData.buffers.clear();
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryFile(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
    std::lock_guard<std::mutex> lock(*(bufferPtr->mutex));
    std::ifstream ifs;
    if (auto* tmpFile = std::get_if<std::string>(&(bufferPtr->storage))) {
        ifs.open(*tmpFile, std::ios::binary);
        if (!ifs) {
            warning("%s: Failed to open temporary file %s", __func__, tmpFile->c_str());
            return -1;
        }
    } else {
        error("%s cannot be called when buffer is in memory", __func__);
    }
    int npt = 0;
    tileData.clear();
    bufferId2bound(bufferPtr->key, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    while (true) {
        RecordT<T> rec;
        ifs.read(reinterpret_cast<char*>(&rec), sizeof(RecordT<T>));
        if (ifs.gcount() != sizeof(RecordT<T>)) break;
        tileData.pts.push_back(rec);
        if (isInternalToBuffer(rec.x, rec.y, bufferPtr->key)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryFileExtended(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
    std::lock_guard<std::mutex> lock(*(bufferPtr->mutex));
    std::ifstream ifs;
    if (auto* tmpFile = std::get_if<std::string>(&(bufferPtr->storage))) {
        ifs.open(*tmpFile, std::ios::binary);
        if (!ifs) {
            warning("%s: Failed to open temporary file %s", __func__, tmpFile->c_str());
            return -1;
        }
    } else {
        error("%s cannot be called when buffer is in memory", __func__);
    }
    int npt = 0;
    tileData.clear();
    bufferId2bound(bufferPtr->key, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    while (true) {
        std::vector<uint8_t> buf(recordSize_);
        ifs.read(reinterpret_cast<char*>(buf.data()), recordSize_);
        if (ifs.gcount() != recordSize_) break;
        auto *ptr = buf.data();
        RecordExtendedT<T> r;
        // a) base part
        std::memcpy(&r.recBase, ptr, sizeof(r.recBase));
        // b) each extra
        for (auto &f : schema_) {
            auto *fp = ptr + f.offset;
            switch (f.type) {
                case FieldType::INT32: {
                    int32_t v;
                    std::memcpy(&v, fp, sizeof(v));
                    r.intvals.push_back(v);
                } break;
                case FieldType::FLOAT: {
                    float v;
                    std::memcpy(&v, fp, sizeof(v));
                    r.floatvals.push_back(v);
                } break;
                case FieldType::STRING: {
                    std::string s((char*)fp, f.size);
                    // trim trailing NULs
                    auto pos = s.find('\0');
                    if (pos!=std::string::npos) s.resize(pos);
                    r.strvals.push_back(s);
                } break;
            }
        }
        tileData.extPts.push_back(std::move(r));
        if (isInternalToBuffer(r.recBase.x, r.recBase.y, bufferPtr->key)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryStandard(TileData<T>& tileData, InMemoryStorageStandard<T>* memStore, uint32_t bufferKey) {
    tileData.clear();
    bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    // Directly copy the data from the in-memory vector.
    tileData.pts = std::move(memStore->data);
    // Mark internal points (for output)
    int npt = 0;
    for(const auto& rec : tileData.pts) {
        if (isInternalToBuffer(rec.x, rec.y, bufferKey)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryExtended(TileData<T>& tileData, InMemoryStorageExtended<T>* memStore, uint32_t bufferKey) {
    tileData.clear();
    bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    tileData.extPts = std::move(memStore->dataExtended);
    int npt = 0;
    for(const auto& rec : tileData.extPts) {
        if (isInternalToBuffer(rec.recBase.x, rec.recBase.y, bufferKey)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
typename Tiles2MinibatchBase<T>::ProcessedResult Tiles2MinibatchBase<T>::formatAnchorResult(const std::vector<cv::Point2f>& anchors, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket, float xmin, float xmax, float ymin, float ymax) {
    size_t nrows = std::min((size_t) topVals.rows(), anchors.size());
    if (topVals.rows() != nrows || topIds.rows() != nrows || anchors.size() != nrows) {
        error("%s: size mismatch: topVals.rows()=%d, topIds.rows()=%d, anchors.size()=%d",
            __func__, topVals.rows(), topIds.rows(), anchors.size());
    }
    ProcessedResult result(ticket, xmin, xmax, ymin, ymax);
    char buf[512];
    for (size_t i = 0; i < nrows; ++i) {
        if (anchors[i].x < xmin + r || anchors[i].x >= xmax - r ||
            anchors[i].y < ymin + r || anchors[i].y >= ymax - r) {
            continue; // only write internal anchors
        }
        int len = std::snprintf(
            buf, sizeof(buf),
            "%.*f\t%.*f",
            floatCoordDigits,
            anchors[i].x,
            floatCoordDigits,
            anchors[i].y
        );
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len,
                sizeof(buf) - len,
                "\t%d",
                topIds(i, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing anchor output line", __func__);
            }
            len += n;
        }
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len,
                sizeof(buf) - len,
                "\t%.*e",
                probDigits,
                topVals(i, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing anchor output line", __func__);
            }
            len += n;
        }
        buf[len++] = '\n';
        result.outputLines.emplace_back(buf, len);
    }
    result.npts = result.outputLines.size();
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ProcessedResult Tiles2MinibatchBase<T>::formatPixelResultWithOriginalData(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket,
std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    if (outputBackgroundProb_) {
        assert(phi0 != nullptr && phi0->size() == size_t(topVals.rows()));
    }
    ProcessedResult result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    int32_t nrows = topVals.rows();
    char buf[65536];
    for (size_t i = 0; i < tileData.idxinternal.size(); ++i) {
        int32_t idxorg = tileData.idxinternal[i]; // index in the original data
        int32_t idx = tileData.orgpts2pixel[idxorg]; // index in the pixel minibatch
        if (idx < 0 || idx >= nrows) {
            continue;
        }
        const RecordT<T>* recPtr;
        if (useExtended_) {
            recPtr = &tileData.extPts[idxorg].recBase;
        } else {
            recPtr = &tileData.pts[idxorg];
        }
        const RecordT<T>& rec = *recPtr;
        int len = 0;
        if constexpr (std::is_same_v<T, int32_t>) {
            len = std::snprintf(
                buf, sizeof(buf), "%d\t%d\t",
                rec.x, rec.y
            );
        } else {
            len = std::snprintf(
                buf, sizeof(buf),
                "%.*f\t%.*f\t",
                floatCoordDigits,
                rec.x,
                floatCoordDigits,
                rec.y
            );
        }
        len += std::snprintf(
            buf + len, sizeof(buf) - len, "%s\t%d",
            featureNames[rec.idx].c_str(), rec.ct
        );
        // write background probability if available
        if (phi0 != nullptr) {
            float bgprob = 0.0f;
            auto& phi0_map = (*phi0)[idx];
            auto it = phi0_map.find(rec.idx);
            if (it != phi0_map.end()) {
                bgprob = it->second;
            }
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%.*e",
                probDigits, bgprob
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing background probability", __func__);
            }
            len += n;
        }
        // write the top‑k IDs
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%d",
                topIds(idx, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing output line", __func__);
            }
            len += n;
        }
        // write the top‑k probabilities in scientific form
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%.*e",
                probDigits, topVals(idx, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing output line", __func__);
            }
            len += n;
        }
        // write the extra fields
        if (useExtended_) {
            const RecordExtendedT<T>& recExt = tileData.extPts[idxorg];
            for (auto v : recExt.intvals) {
                len += std::snprintf(buf + len, sizeof(buf) - len, "\t%d", v);
                if (len >= int(sizeof(buf))) {
                    error("%s: buffer overflow while writing intvals", __func__);
                }
            }
            for (auto v : recExt.floatvals) {
                len += std::snprintf(buf + len, sizeof(buf) - len, "\t%f", v);
                if (len >= int(sizeof(buf))) {
                    error("%s: buffer overflow while writing floatvals", __func__);
                }
            }
            for (auto& v : recExt.strvals) {
                len += std::snprintf(buf + len, sizeof(buf) - len, "\t%s", v.c_str());
                if (len >= int(sizeof(buf))) {
                    error("%s: buffer overflow while writing strvals", __func__);
                }
            }
        }
        buf[len++] = '\n';
        result.outputLines.emplace_back(buf, len);
    }
    result.npts = result.outputLines.size();
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ProcessedResult Tiles2MinibatchBase<T>::formatPixelResult(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket) {
    ProcessedResult result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    size_t N = tileData.coords.size();
    std::vector<bool> internal(N, 0);
    for (auto j : tileData.idxinternal) {
        if (tileData.orgpts2pixel[j] < 0) {
            continue;
        }
        internal[tileData.orgpts2pixel[j]] = true;
    }
    char buf[512];
    for (size_t j = 0; j < N; ++j) {
        if (!internal[j]) {
            continue;
        }
        int len = len = std::snprintf(
            buf, sizeof(buf), "%.*f\t%.*f",
            floatCoordDigits, tileData.coords[j].first,
            floatCoordDigits, tileData.coords[j].second
        );
        // write the top‑k IDs
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%d",
                topIds(j, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing output line", __func__);
            }
            len += n;
        }
        // write the top‑k probabilities in scientific form
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%.*e",
                probDigits, topVals(j, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing output line", __func__);
            }
            len += n;
        }
        buf[len++] = '\n';
        result.outputLines.emplace_back(buf, len);
    }
    result.npts = result.outputLines.size();
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ProcessedResult Tiles2MinibatchBase<T>::formatPixelResultWithBackground(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket,
std::vector<std::unordered_map<uint32_t, float>>& phi0) {
    if (outputBackgroundProb_) {
        assert(phi0.size() == size_t(topVals.rows()));
    }
    ProcessedResult result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    size_t N = tileData.coords.size();
    std::vector<bool> internal(N, 0);
    for (auto j : tileData.idxinternal) {
        if (tileData.orgpts2pixel[j] < 0) {
            continue;
        }
        internal[tileData.orgpts2pixel[j]] = true;
    }
    char buf[512];
    for (size_t j = 0; j < N; ++j) {
        if (!internal[j]) {
            continue;
        }
        for (const auto& kv : phi0[j]) {
            int len = len = std::snprintf(
                buf, sizeof(buf), "%.*f\t%.*f",
                floatCoordDigits, tileData.coords[j].first,
                floatCoordDigits, tileData.coords[j].second
            );
            // write feature name and background probability
            len += std::snprintf(
                buf + len, sizeof(buf) - len, "\t%s\t%.*f",
                featureNames[kv.first].c_str(),
                floatCoordDigits, kv.second
            );
            // write the top‑k IDs
            for (int32_t k = 0; k < topk_; ++k) {
                int n = std::snprintf(
                    buf + len, sizeof(buf) - len, "\t%d",
                    topIds(j, k)
                );
                if (n < 0 || n >= int(sizeof(buf) - len)) {
                    error("%s: error writing output line", __func__);
                }
                len += n;
            }
            // write the top‑k probabilities in scientific form
            for (int32_t k = 0; k < topk_; ++k) {
                int n = std::snprintf(
                    buf + len, sizeof(buf) - len, "\t%.*e",
                    probDigits, topVals(j, k)
                );
                if (n < 0 || n >= int(sizeof(buf) - len)) {
                    error("%s: error writing output line", __func__);
                }
                len += n;
            }
            buf[len++] = '\n';
            result.outputLines.emplace_back(buf, len);
        }

    }
    result.npts = result.outputLines.size();
    return result;
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::buildMinibatchCore( TileData<T>& tileData,
    std::vector<cv::Point2f>& anchors, Minibatch& minibatch,
    double pixelResolution, double distR, double distNu) {

    if (minibatch.n <= 0) {
        return 0;
    }
    assert(distR > 0.0 && distNu > 0.0);

    PointCloudCV<float> pc;
    pc.pts = std::move(anchors);
    kd_tree_cv2f_t kdtree(2, pc, {10});
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;

    const float res = static_cast<float>(pixelResolution);
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
            int32_t x = static_cast<int32_t>(pt.recBase.x / res);
            int32_t y = static_cast<int32_t>(pt.recBase.y / res);
            uint64_t key = (static_cast<uint64_t>(x) << 32) | static_cast<uint32_t>(y);
            pixAgg[key].first[pt.recBase.idx] += pt.recBase.ct;
            pixAgg[key].second.push_back(idxOriginal++);
        }
    } else {
        tileData.orgpts2pixel.assign(tileData.pts.size(), -1);
        for (const auto& pt : tileData.pts) {
            int32_t x = static_cast<int32_t>(pt.x / res);
            int32_t y = static_cast<int32_t>(pt.y / res);
            uint64_t key = (static_cast<uint64_t>(x) << 32) | static_cast<uint32_t>(y);
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

        tileData.coords.emplace_back(static_cast<double>(xy[0]), static_cast<double>(xy[1]));
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
void Tiles2MinibatchBase<T>::writeHeaderToJson() {
    std::string jsonFile = outPref + ".json";
    std::ofstream jsonOut(jsonFile);
    if (!jsonOut) {
        error("Error opening json output file: %s", jsonFile.c_str());
    }
    nlohmann::json header;
    header["x"] = 0;
    header["y"] = 1;
    int32_t idx = 2;
    if (outputOriginalData_) {
        header["feature"] = 2;
        header["ct"] = 3;
        idx += 2;
    }
    if (outputBackgroundProb_) {
        header["p0"] = idx++;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header["K" + std::to_string(i+1)] = idx++;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header["P" + std::to_string(i+1)] = idx++;
    }
    if (useExtended_ && lineParserPtr) {
        for (const auto& v : lineParserPtr->name_ints) {
            header[v] = idx++;
        }
        for (const auto& v : lineParserPtr->name_floats) {
            header[v] = idx++;
        }
        for (const auto& v : lineParserPtr->name_strs) {
            header[v] = idx++;
        }
    }
    jsonOut << std::setw(4) << header << std::endl;
    jsonOut.close();
}

template<typename T>
void Tiles2MinibatchBase<T>::writerWorker() {
    // A priority queue to buffer out-of-order results
    std::priority_queue<ProcessedResult, std::vector<ProcessedResult>, std::greater<ProcessedResult>> outOfOrderBuffer;
    int nextTicketToWrite = 0;
    ProcessedResult result;
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
                size_t totalLen = 0;
                for (const auto& line : readyToWrite.outputLines) {
                    if (!write_all(fdMain, line.data(), line.size())) {
                        error("Error writing to main output file");
                    }
                    totalLen += line.size();
                }
                size_t ed = st + totalLen;
                if (st == 0) ed += headerSize;
                IndexEntry<float> e{st, ed, readyToWrite.npts,
                    readyToWrite.xmin, readyToWrite.xmax,
                    readyToWrite.ymin, readyToWrite.ymax};
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
    std::priority_queue<ProcessedResult, std::vector<ProcessedResult>, std::greater<ProcessedResult>> outOfOrderBuffer;
    int nextTicketToWrite = 0;
    ProcessedResult result;
    while (anchorQueue.pop(result)) {
        outOfOrderBuffer.push(std::move(result));
        while (!outOfOrderBuffer.empty() &&
               (!useTicketSystem_ ||
                outOfOrderBuffer.top().ticket == nextTicketToWrite)) {
            const auto& readyToWrite = outOfOrderBuffer.top();
            if (readyToWrite.npts > 0) {
                size_t totalLen = 0;
                for (const auto& line : readyToWrite.outputLines) {
                    if (!write_all(fdAnchor, line.data(), line.size())) {
                        error("Error writing to anchor output file");
                    }
                    totalLen += line.size();
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
void Tiles2MinibatchBase<T>::setupOutput() {
    #if !defined(_WIN32)
        // ensure includes present
    #endif
    std::string outputFile = outPref + ".tsv";
    fdMain = ::open(outputFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdMain < 0) {
        error("Error opening main output file: %s", outputFile.c_str());
    }
    // compose header
    std::string jsonFile = outPref + ".json";
    std::ofstream jsonOut(jsonFile);
    if (!jsonOut) {
        error("Error opening json output file: %s", jsonFile.c_str());
    }
    nlohmann::json header;
    std::string header_str = "#x\ty";
    header["x"] = 0;
    header["y"] = 1;
    int32_t idx = 2;
    if (outputOriginalData_) {
        header_str += "\tfeature\tct";
        header["feature"] = 2;
        header["ct"] = 3;
        idx += 2;
    }
    if (outputBackgroundProb_) {
        header_str += "\tp0";
        header["p0"] = idx++;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header_str += "\tK" + std::to_string(i+1);
        header["K" + std::to_string(i+1)] = idx++;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header_str += "\tP" + std::to_string(i+1);
        header["P" + std::to_string(i+1)] = idx++;
    }
    if (useExtended_ && lineParserPtr) {
        for (const auto& v : lineParserPtr->name_ints) {
            header_str += "\t" + v;
            header[v] = idx++;
        }
        for (const auto& v : lineParserPtr->name_floats) {
            header_str += "\t" + v;
            header[v] = idx++;
        }
        for (const auto& v : lineParserPtr->name_strs) {
            header_str += "\t" + v;
            header[v] = idx++;
        }
    }
    jsonOut << std::setw(4) << header << std::endl;
    jsonOut.close();
    header_str += "\n";
    if (!write_all(fdMain, header_str.data(), header_str.size())) {
        error("Error writing header_str to main output file: %s", outputFile.c_str());
    }
    headerSize = header_str.size();
    std::string indexFile = outPref + ".index";
    fdIndex = ::open(indexFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) {
        error("Error opening index output file: %s", indexFile.c_str());
    }
    if (!outputAnchor_) return;
    // setup anchor output
    std::string anchorFile = outPref + ".anchors.tsv";
    fdAnchor = ::open(anchorFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdAnchor < 0) {
        error("Error opening anchor output file: %s", anchorFile.c_str());
    }
    header_str = "#x\ty";
    for (int32_t i = 0; i < topk_; ++i) header_str += "\tK" + std::to_string(i+1);
    for (int32_t i = 0; i < topk_; ++i) header_str += "\tP" + std::to_string(i+1);
    header_str += "\n";
    if (!write_all(fdAnchor, header_str.data(), header_str.size())) {
        error("Error writing header_str to anchor output file: %s", anchorFile.c_str());
    }
    anchorHeaderSize = header_str.size();
}

template<typename T>
void Tiles2MinibatchBase<T>::closeOutput() {
    if (fdMain >= 0) { ::close(fdMain); fdMain = -1; }
    if (fdIndex >= 0) { ::close(fdIndex); fdIndex = -1; }
    if (fdAnchor >= 0) { ::close(fdAnchor); fdAnchor = -1; }
}

template class Tiles2MinibatchBase<int32_t>;
template class Tiles2MinibatchBase<float>;
