#include "tiles2minibatch.hpp"

template<typename T>
int32_t Tiles2Minibatch<T>::loadAnchors(const std::string& anchorFile) {
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
int32_t Tiles2Minibatch<T>::parseOneTile(TileData<T>& tileData, TileKey tile) {
    std::unique_ptr<BoundedReadline> iter;
    try {
        iter = tileReader.get_tile_iterator(tile.row, tile.col);
    } catch (const std::exception &e) {
        warning("%s", e.what());
        return -1;
    }
    tileData.clear();
    tile2bound(tile, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);

    std::string line;
    int32_t npt = 0;
    if (useExtended_) {
        while (iter->next(line)) {
            RecordExtendedT<T> recExt;
            int32_t idx = lineParser.parse(recExt, line);
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
            int32_t idx = lineParser.parse<T>(rec, line);
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
int32_t Tiles2Minibatch<T>::parseBoundaryFile(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
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
int32_t Tiles2Minibatch<T>::parseBoundaryFileExtended(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
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
int32_t Tiles2Minibatch<T>::parseBoundaryMemoryStandard(TileData<T>& tileData, InMemoryStorageStandard<T>* memStore, uint32_t bufferKey) {
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
int32_t Tiles2Minibatch<T>::parseBoundaryMemoryExtended(TileData<T>& tileData, InMemoryStorageExtended<T>* memStore, uint32_t bufferKey) {
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
int32_t Tiles2Minibatch<T>::initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {
    anchors.clear();
    std::vector<Document> documents;
    std::map<std::tuple<int32_t, int32_t, int32_t, int32_t>, std::unordered_map<uint32_t, float>> hexAggregation;
    auto assign_pt = [&](const auto& pt) {
        for (int32_t ir = 0; ir < nMoves; ++ir) {
            for (int32_t ic = 0; ic < nMoves; ++ic) {
                int32_t hx, hy;
                hexGrid.cart_to_axial(hx, hy, pt.x, pt.y, ic * 1. / nMoves, ir * 1. / nMoves);
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
        if (sum < anchorMinCount) {
            continue;
        }
        Document doc;
        for (auto& featurePair : entry.second) {
            doc.ids.push_back(featurePair.first);
            doc.cnts.push_back(featurePair.second);
        }
        if (weighted) {
            for (size_t i = 0; i < doc.ids.size(); ++i) {
                doc.cnts[i] *= lineParser.weights[doc.ids[i]];
            }
        }
        documents.push_back(std::move(doc));
        // Unpack the key to get hex coordinates and move indices
        const auto& key = entry.first;
        int32_t hx = std::get<0>(key);
        int32_t hy = std::get<1>(key);
        int32_t ic = std::get<2>(key);
        int32_t ir = std::get<3>(key);
        float x, y;
        hexGrid.axial_to_cart(x, y, hx, hy, ic * 1. / nMoves, ir * 1. / nMoves);
        anchors.emplace_back(x, y);
    }

    if (documents.empty()) {
        return 0;
    }
    minibatch.gamma = lda.transform(documents).template cast<float>();
    // TODO: need to test if scaling/normalizing gamma is better
    // scale each row so that the mean is 1
    for (int i = 0; i < minibatch.gamma.rows(); ++i) {
        float sum = minibatch.gamma.row(i).sum();
        if (sum > 0) {
            minibatch.gamma.row(i) /= sum / K_;
        }
    }
    minibatch.n = documents.size();
    minibatch.M = M_;
    return anchors.size();
}

template<typename T>
int32_t Tiles2Minibatch<T>::makeMinibatch(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {

    PointCloudCV<float> pc;
    pc.pts = std::move(anchors);
    kd_tree_cv2f_t kdtree(2, pc, {10});
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;

    float l2radius = distR * distR;
    std::vector<Eigen::Triplet<float>> triplets4mtx;
    std::vector<Eigen::Triplet<float>> triplets4wij;
    std::vector<Eigen::Triplet<float>> triplets4psi;
    uint32_t npt = 0;
    std::unordered_map<uint64_t, std::pair<std::unordered_map<uint32_t, float>, std::vector<uint32_t>> > pixAgg;
    uint32_t i = 0;
    if (useExtended_) {
        tileData.orgpts2pixel.resize(tileData.extPts.size(), -1);
        for (const auto& pt : tileData.extPts) {
            int32_t x = int32_t (pt.recBase.x / pixelResolution);
            int32_t y = int32_t (pt.recBase.y / pixelResolution);
            uint64_t key = (static_cast<uint64_t>(x) << 32) | (static_cast<uint32_t>(y));
            pixAgg[key].first[pt.recBase.idx] += pt.recBase.ct;
            pixAgg[key].second.push_back(i); // list of original points' indices
            i++;
        }
    } else {
        tileData.orgpts2pixel.resize(tileData.pts.size(), -1);
        for (const auto& pt : tileData.pts) {
            int32_t x = int32_t (pt.x / pixelResolution);
            int32_t y = int32_t (pt.y / pixelResolution);
            uint64_t key = (static_cast<uint64_t>(x) << 32) | (static_cast<uint32_t>(y));
            pixAgg[key].first[pt.idx] += pt.ct;
            pixAgg[key].second.push_back(i); // list of original points' indices
            i++;
        }
    }
    // vector of unique coordinates
    tileData.coords.reserve(pixAgg.size());
    for (auto & kv : pixAgg) {
        int32_t x = static_cast<int32_t>(kv.first >> 32);
        int32_t y = static_cast<int32_t>(kv.first & 0xFFFFFFFF);
        float xy[2] = {x * pixelResolution, y * pixelResolution};
        size_t n = kdtree.radiusSearch(xy, l2radius, indices_dists);
        if (n == 0) {
            continue;
        }
        if (weighted) {
            for (auto & kv2 : kv.second.first) {
                kv2.second *= lineParser.weights[kv2.first];
                triplets4mtx.emplace_back(npt, kv2.first, kv2.second);
            }
        } else {
            for (auto & kv2 : kv.second.first) {
                triplets4mtx.emplace_back(npt, kv2.first, kv2.second);
            }
        }
        tileData.coords.emplace_back(xy[0], xy[1]);
        for (auto & v : kv.second.second) {
            tileData.orgpts2pixel[v] = npt;
        }
        std::vector<float> dvec(n, 0);
        for (size_t i = 0; i < n; ++i) {
            uint32_t idx = indices_dists[i].first;
            float dist = indices_dists[i].second;
            dist = std::max(std::min(1. - pow(dist / distR, distNu), 0.95), 0.05);
            dvec[i] = dist;
            triplets4wij.emplace_back(npt, idx, logit(dist));
        }
        float rowsum = std::accumulate(dvec.begin(), dvec.end(), 0.0);
        for (size_t i = 0; i < n; ++i) {
            triplets4psi.emplace_back(npt, indices_dists[i].first, dvec[i] / rowsum);
        }
        npt++;
    }
    minibatch.N = npt;
    minibatch.mtx.resize(npt, M_);
    minibatch.mtx.setFromTriplets(triplets4mtx.begin(), triplets4mtx.end());
    minibatch.mtx.makeCompressed();
    triplets4mtx.clear();
    minibatch.logitwij.resize(npt, minibatch.n);
    minibatch.logitwij.setFromTriplets(triplets4wij.begin(), triplets4wij.end());
    minibatch.logitwij.makeCompressed();
    triplets4wij.clear();
    minibatch.psi.resize(npt, minibatch.n);
    minibatch.psi.setFromTriplets(triplets4psi.begin(), triplets4psi.end());
    minibatch.psi.makeCompressed();
    return npt;
}

template<typename T>
Tiles2Minibatch<T>::ProcessedResult Tiles2Minibatch<T>::formatPixelResultWithOriginalData(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket) {
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
                buf, sizeof(buf),
                "%d\t%d\t",
                rec.x,
                rec.y
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
            buf + len, sizeof(buf) - len,
            "%s\t%d",
            featureNames[rec.idx].c_str(),
            rec.ct
        );
        // write the top‑k IDs
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len,
                sizeof(buf) - len,
                "\t%d",
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
                buf + len,
                sizeof(buf) - len,
                "\t%.*e",
                probDigits,
                topVals(idx, k)
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
Tiles2Minibatch<T>::ProcessedResult Tiles2Minibatch<T>::formatPixelResult(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket) {
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
            buf, sizeof(buf),
            "%.*f\t%.*f",
            floatCoordDigits,
            tileData.coords[j].first,
            floatCoordDigits,
            tileData.coords[j].second
        );
        // write the top‑k IDs
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len,
                sizeof(buf) - len,
                "\t%d",
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
                buf + len,
                sizeof(buf) - len,
                "\t%.*e",
                probDigits,
                topVals(j, k)
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
void Tiles2Minibatch<T>::processTile(TileData<T> &tileData, int threadId, int ticket, vec2f_t* anchorPtr) {
    if (tileData.pts.empty() && tileData.extPts.empty()) {
        return;
    }
    std::vector<cv::Point2f> anchors;
    Minibatch minibatch;
    int32_t nAnchors = initAnchorsHybrid(tileData, anchors, minibatch, anchorPtr);

    if (debug_) {
        std::cout << "Thread " << threadId << " initialized " << nAnchors << " anchors" << std::endl << std::flush;
    }
    if (nAnchors == 0) {
        return;
    }
    int32_t nPixels = makeMinibatch(tileData, anchors, minibatch);
    if (debug_) {
        std::cout << "Thread " << threadId << " made minibatch with " << nPixels << " pixels" << std::endl << std::flush;
    }
    if (nPixels < 10) {
        return;
    }
    auto smtx = slda.do_e_step(minibatch, true);
    {
        std::lock_guard<std::mutex> lock(pseudobulkMutex);
        pseudobulk += smtx;
        if (debug_) {
            std::cout << "Thread " << threadId << " updated pseudobulk.\n";
            std::cout << "    Peek: " << std::fixed << std::setprecision(0);
            for (int32_t i = 0; i < std::min(3, K_); ++i) {
                for (int32_t j = 0; j < std::min(5, M_); ++j) {
                    std::cout << smtx(i, j) << " ";
                }
                std::cout << "\n    ";
            }
            std::cout << "    Current sums: ";
            auto rowsums = pseudobulk.rowwise().sum();
            // sort rowsums in descending order
            std::vector<float> sortedRowsums(rowsums.size());
            for (int32_t i = 0; i < rowsums.size(); ++i) {
                sortedRowsums[i] = rowsums(i);
            }
            std::sort(sortedRowsums.begin(), sortedRowsums.end(), std::greater<float>());
            for (int32_t i = 0; i < K_; ++i) {
                std::cout << sortedRowsums[i] << " ";
            }
            std::cout << std::endl << std::flush;
        }
    }
    if (debug_) {
        std::cout << "Thread " << threadId << " finished decoding" << std::endl << std::flush;
    }
    MatrixXf topVals;
    Eigen::MatrixXi topIds;
    findTopK(topVals, topIds, minibatch.phi, topk_);
    if (debug_) {
        std::cout << "Thread " << threadId << " start writing to output" << std::endl << std::flush;
    }

    ProcessedResult result;
    if (outputOriginalData) {
        result = formatPixelResultWithOriginalData(tileData, topVals, topIds, ticket);
    } else {
        result = formatPixelResult(tileData, topVals, topIds, ticket);
    }
    resultQueue.push(std::move(result));

    notice("Thread %d (ticket %d) fit minibatch with %d anchors and output %lu internal pixels", threadId, ticket, nAnchors, result.npts);
}

template<typename T>
void Tiles2Minibatch<T>::writeHeaderToJson() {
    std::string jsonFile = outPref + ".json";
    std::ofstream jsonOut(jsonFile);
    if (!jsonOut) {
        error("Error opening json output file: %s", jsonFile.c_str());
    }
    nlohmann::json header;
    header["x"] = 0;
    header["y"] = 1;
    int32_t idx = 2;
    if (outputOriginalData) {
        header["feature"] = 2;
        header["ct"] = 3;
        idx = 4;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header["K" + std::to_string(i+1)] = idx++;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header["P" + std::to_string(i+1)] = idx++;
    }
    if (useExtended_) {
        for (const auto& v : lineParser.name_ints) {
            header[v] = idx++;
        }
        for (const auto& v : lineParser.name_floats) {
            header[v] = idx++;
        }
        for (const auto& v : lineParser.name_strs) {
            header[v] = idx++;
        }
    }
    jsonOut << std::setw(4) << header << std::endl;
    jsonOut.close();
}

template<typename T>
void Tiles2Minibatch<T>::writePseudobulkToTsv() {
    std::string pseudobulkFile = outPref + ".pseudobulk.tsv";
    std::ofstream oss(pseudobulkFile, std::ios::out);
    if (!oss) {
        error("Error opening pseudobulk output file: %s", pseudobulkFile.c_str());
    }
    oss << "Feature";
    const auto factorNames = lda.get_topic_names();
    for (int32_t i = 0; i < K_; ++i) {
        oss << "\t" << factorNames[i];
    }
    oss << "\n" << std::setprecision(probDigits) << std::fixed;
    for (int32_t i = 0; i < M_; ++i) {
        oss << featureNames[i];
        for (int32_t j = 0; j < K_; ++j) {
            oss << "\t" << pseudobulk(j, i);
        }
        oss << "\n";
    }
    oss.close();
}

template<typename T>
int32_t Tiles2Minibatch<T>::initAnchorsHybrid(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch, const vec2f_t* fixedAnchors) {
    if ((fixedAnchors == nullptr) || fixedAnchors->empty()) {
        return initAnchors(tileData, anchors, minibatch);
    }
    anchors.clear();
    for (const auto& pt : *fixedAnchors) {
        anchors.emplace_back(pt[0], pt[1]);
    }
    size_t nFixed = anchors.size();
    if (nFixed == 0) {
        return initAnchors(tileData, anchors, minibatch);
    }

    // 1 Initialize hexagonal lattice
    vec2f_t lattice;
    double gridDist = hexGrid.size/nMoves;
    double buff = gridDist / 4.;
    hex_grid_cart<float>(lattice, tileData.xmin + buff, tileData.xmax - buff, tileData.ymin + buff, tileData.ymax - buff, gridDist);
    // 2 Remove lattice points too close to any fixed anchors
    KDTreeVectorOfVectorsAdaptor<vec2f_t, float> reftree(2, *fixedAnchors, 10);
    float l2radius = gridDist * gridDist / 4.;
    int32_t nRemoved = 0;
    for (const auto& pt : lattice) {
        std::vector<size_t> ret_indexes(1);
        std::vector<float> out_dists_sqr(1);
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        reftree.index->findNeighbors(resultSet, pt.data());
        if (out_dists_sqr[0] < l2radius) {
            nRemoved++;
            continue;
        }
        anchors.emplace_back(pt[0], pt[1]);
    }
    size_t nAnchors = anchors.size();
    notice("Initialized %zu fixed anchors and %zu-%d lattice points", nFixed, nAnchors, nRemoved);

    std::vector<std::unordered_map<uint32_t, float>> docAgg;

    // 3 Iterative refinement (weighted Lloyd's / K-means)
    for (int32_t t = 0; t < nLloydIter; ++t) {
        // Build a k-d tree on the current anchor positions
        PointCloudCV<float> pc;
        pc.pts = anchors;
        kd_tree_cv2f_t kdtree(2, pc, {10});
        docAgg.assign(nAnchors, std::unordered_map<uint32_t, float>());
        std::vector<cv::Point2f> newAnchorCoords(nAnchors, cv::Point2f(0, 0));
        std::vector<float> totalCounts(nAnchors, 0.0f);
        // E-Step: Assign each data point to its closest anchor
        auto assign_pt = [&](const auto& pt) {
            float query_pt[2] = {(float) pt.x, (float) pt.y};
            // Find the single nearest anchor (k=1 search is fast)
            size_t ret_index;
            float out_dist_sqr;
            nanoflann::KNNResultSet<float> resultSet(1);
            resultSet.init(&ret_index, &out_dist_sqr);
            kdtree.findNeighbors(resultSet, query_pt);
            // Aggregate data and coordinates to the assigned anchor
            docAgg[ret_index][pt.idx] += pt.ct;
            newAnchorCoords[ret_index].x += pt.x * pt.ct;
            newAnchorCoords[ret_index].y += pt.y * pt.ct;
            totalCounts[ret_index] += pt.ct;
        };
        if (useExtended_) {
            for (const auto& rec : tileData.extPts) {
                assign_pt(rec.recBase);
            }
        } else {
            for (const auto& rec : tileData.pts) {
                assign_pt(rec);
            }
        }
        // M-Step: Recalculate centroids for non-fixed anchors
        for (size_t i = nFixed; i < nAnchors; ++i) {
            if (totalCounts[i] > 0) {
                anchors[i].x = newAnchorCoords[i].x / totalCounts[i];
                anchors[i].y = newAnchorCoords[i].y / totalCounts[i];
            }
        }
    }

    // 4 Aggregate pixels and initialize anchors
    std::vector<Document> docs;
    std::vector<cv::Point2f> finalAnchors;
    for (int32_t j = 0; j < nAnchors; ++j) {
        if (docAgg[j].empty()) continue;
        float sum = std::accumulate(docAgg[j].begin(), docAgg[j].end(), 0.0,
                                    [](float a, const auto& b) { return a + b.second; });
        if (sum < anchorMinCount) continue;
        finalAnchors.push_back(anchors[j]);
        Document doc;
        if (weighted) {
            for (const auto& item : docAgg[j]) {
                doc.ids.push_back(item.first);
                doc.cnts.push_back(item.second * lineParser.weights[item.first]);
            }
        } else {
            for (const auto& item : docAgg[j]) {
                doc.ids.push_back(item.first);
                doc.cnts.push_back(item.second);
            }
        }
        docs.push_back(std::move(doc));
    }
    if (docs.empty()) return 0;

    anchors = std::move(finalAnchors);
    minibatch.gamma = lda.transform(docs).template cast<float>();
    for (int i = 0; i < minibatch.gamma.rows(); ++i) {
        float sum = minibatch.gamma.row(i).sum();
        if (sum > 0) {
            minibatch.gamma.row(i) /= sum / K_;
        }
    }
    minibatch.n = docs.size();
    minibatch.M = M_;
    return anchors.size();
}

template<typename T>
void Tiles2Minibatch<T>::tileWorker(int threadId) {
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
            continue;
        }
        vec2f_t* anchorPtr = nullptr;
        if (fixedAnchorForTile.find(tile) != fixedAnchorForTile.end()) {
            anchorPtr = &fixedAnchorForTile[tile];
        }
        processTile(tileData, threadId, ticket, anchorPtr);
    }
}

template<typename T>
void Tiles2Minibatch<T>::boundaryWorker(int threadId) {
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
                ret = parseBoundaryMemoryExtended(tileData, extStore, bufferPtr->key);
            } else if (auto* stdStore = dynamic_cast<InMemoryStorageStandard<T>*>(storagePtr->get())) {
                ret = parseBoundaryMemoryStandard(tileData, stdStore, bufferPtr->key);
            }
        } else if (auto* filePath = std::get_if<std::string>(&(bufferPtr->storage))) {
            // --- DISK I/O PATH ---
            if (useExtended_) {
                ret = parseBoundaryFileExtended(tileData, bufferPtr);
            } else {
                ret = parseBoundaryFile(tileData, bufferPtr);
            }
            // Clean up the temporary file
            std::remove(filePath->c_str());
        }
        notice("%s: Thread %d (ticket %d) read boundary buffer (%d) with %d internal pixels", __FUNCTION__, threadId, ticket, bufferPtr->key, ret);
        vec2f_t* anchorPtr = nullptr;
        if (fixedAnchorForBoundary.find(bufferPtr->key) != fixedAnchorForBoundary.end()) {
            anchorPtr = &fixedAnchorForBoundary[bufferPtr->key];
        }
        processTile(tileData, threadId, ticket, anchorPtr);
    }
}

template<typename T>
void Tiles2Minibatch<T>::writerWorker() {
    // A priority queue to buffer out-of-order results
    std::priority_queue<ProcessedResult, std::vector<ProcessedResult>, std::greater<ProcessedResult>> outOfOrderBuffer;
    int nextTicketToWrite = 0;
    ProcessedResult result;
    // Loop until the queue is marked as done and is empty
    while (resultQueue.pop(result)) {
        outOfOrderBuffer.push(std::move(result));
        // Write all results that are now ready in sequential order
        while (!outOfOrderBuffer.empty() &&
               (!useTicketSystem ||
                outOfOrderBuffer.top().ticket == nextTicketToWrite)) {
            const auto& readyToWrite = outOfOrderBuffer.top();
            if (readyToWrite.npts > 0) {
                for (const auto& line : readyToWrite.outputLines) {
                    mainOut.write(line.c_str(), line.length());
                }
                size_t endPos = mainOut.tellp();
                IndexEntry<float> e{outputSize, endPos, readyToWrite.npts,
                    readyToWrite.xmin, readyToWrite.xmax,
                    readyToWrite.ymin, readyToWrite.ymax};
                indexOut.write(reinterpret_cast<char*>(&e), sizeof(e));
                outputSize = endPos;
            }
            outOfOrderBuffer.pop();
            nextTicketToWrite++;
        }
    }
}

template<typename T>
void Tiles2Minibatch<T>::setExtendedSchema() {
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

template<typename T>
void Tiles2Minibatch<T>::setupOutput() {
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

template class Tiles2Minibatch<int32_t>;
template class Tiles2Minibatch<float>;
