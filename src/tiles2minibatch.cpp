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
        buffer->writeToFile(entry.second);
    }
    tileData.buffers.clear();
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2Minibatch<T>::parseBoundaryFile(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
    std::lock_guard<std::mutex> lock(*(bufferPtr->mutex));
    std::ifstream ifs(bufferPtr->tmpFile, std::ios::binary);
    int32_t bufRow, bufCol;
    bool isVertical = decodeTempFileKey(bufferPtr->key, bufRow, bufCol);
    if (!ifs) {
        warning("Error opening temporary file (%d, %d, %d): %s", int32_t (isVertical), bufRow, bufCol, (bufferPtr->tmpFile).c_str());
        return -1;
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
    notice("Read boundary buffer (%d, %d, %d) with %d internal pixels", isVertical, bufRow, bufCol, npt);
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2Minibatch<T>::initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {

    anchors.clear();
    std::vector<Document> documents;
    for (int32_t ir = 0; ir < nMoves; ++ir) {
        for (int32_t ic = 0; ic < nMoves; ++ic) {
            std::unordered_map<int64_t, std::unordered_map<uint32_t, float>> hexAggregation;
            for (const auto& pt : tileData.pts) {
                int32_t hx, hy;
                hexGrid.cart_to_axial(hx, hy, pt.x, pt.y, ic*1./nMoves, ir*1./nMoves);
                int64_t key = (static_cast<int64_t>(hx) << 32) | (static_cast<uint32_t>(hy));
                hexAggregation[key][pt.idx] += pt.ct;
            }
            // Create the vector of Document from the aggregated data.
            for (auto& hexEntry : hexAggregation) {
                double sum = std::accumulate(hexEntry.second.begin(), hexEntry.second.end(), 0.0, [](double acc, const auto& p) { return acc + p.second; });
                if (sum < anchorMinCount) {
                    continue;
                }
                Document doc;
                for (auto& featurePair : hexEntry.second) {
                    doc.ids.push_back(featurePair.first);
                    doc.cnts.push_back(featurePair.second);
                }
                if (weighted) {
                    for (size_t i = 0; i < doc.ids.size(); ++i) {
                        doc.cnts[i] *= lineParser.weights[doc.ids[i]];
                    }
                }
                documents.push_back(std::move(doc));
                float x, y;
                int32_t hx = static_cast<int32_t>(hexEntry.first >> 32);
                int32_t hy = static_cast<int32_t>(hexEntry.first & 0xFFFFFFFF);
                hexGrid.axial_to_cart(x, y, hx, hy, ic*1./nMoves, ir*1./nMoves);
                anchors.emplace_back(x, y);
            }
        }
    }
    if (documents.empty()) {
        return 0;
    }
    minibatch.gamma = lda.transform(documents);
    // TODO: need to test if scaling/normalizing gamma is better
    // scale each row so that the mean is 1
    for (int i = 0; i < minibatch.gamma.rows(); ++i) {
        double sum = minibatch.gamma.row(i).sum();
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

    double l2radius = distR * distR;
    std::vector<Eigen::Triplet<double>> triplets4mtx;
    std::vector<Eigen::Triplet<double>> triplets4wij;
    std::vector<Eigen::Triplet<double>> triplets4psi;
    uint32_t npt = 0;
    tileData.orgpts2pixel.resize(tileData.pts.size(), -1);
    std::unordered_map<uint64_t, std::pair<std::unordered_map<uint32_t, double>, std::vector<uint32_t>> > pixAgg;
    uint32_t i = 0;
    for (const auto& pt : tileData.pts) {
        int32_t x = int32_t (pt.x / pixelResolution);
        int32_t y = int32_t (pt.y / pixelResolution);
        uint64_t key = (static_cast<uint64_t>(x) << 32) | (static_cast<uint32_t>(y));
        pixAgg[key].first[pt.idx] += pt.ct;
        pixAgg[key].second.push_back(i); // list of original points' indices
        i++;
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
        std::vector<double> dvec(n, 0);
        for (size_t i = 0; i < n; ++i) {
            uint32_t idx = indices_dists[i].first;
            double dist = indices_dists[i].second;
            dist = std::max(std::min(1. - pow(dist / distR, distNu), 0.95), 0.05);
            dvec[i] = dist;
            triplets4wij.emplace_back(npt, idx, logit(dist));
        }
        double rowsum = std::accumulate(dvec.begin(), dvec.end(), 0.0);
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
int32_t Tiles2Minibatch<T>::outputOriginalDataWithPixelResult(const TileData<T>& tileData, const MatrixXd& topVals, const Eigen::MatrixXi& topIds) {
    std::lock_guard<std::mutex> lock(mainOutMutex);
    uint32_t npts = 0;
    int32_t nrows = topVals.rows();
    char buf[65536];
    for (size_t i = 0; i < tileData.idxinternal.size(); ++i) {
        int32_t idxorg = tileData.idxinternal[i]; // index in the original data
        int32_t idx = tileData.orgpts2pixel[idxorg]; // index in the pixel minibatch
        if (idx < 0 || idx >= nrows) {
            continue;
        }
        const RecordT<T>& rec = tileData.pts[idxorg];
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
        buf[len++] = '\n';
        mainOut.write(buf, len);
        npts++;
    }
    size_t endPos = mainOut.tellp();
    IndexEntry<float> e{outputSize, endPos, npts,
        tileData.xmin, tileData.xmax,
        tileData.ymin, tileData.ymax};
    indexOut.write(reinterpret_cast<char*>(&e), sizeof(e));
    outputSize += endPos;
    return npts;
}

template<typename T>
int32_t Tiles2Minibatch<T>::outputPixelResult(const TileData<T>& tileData, const MatrixXd& topVals, const Eigen::MatrixXi& topIds) {
    std::lock_guard<std::mutex> lock(mainOutMutex);
    size_t N = tileData.coords.size();
    std::vector<bool> internal(N, 0);
    for (auto j : tileData.idxinternal) {
        if (tileData.orgpts2pixel[j] < 0) {
            continue;
        }
        internal[tileData.orgpts2pixel[j]] = true;
    }
    uint32_t npts = 0;
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
        mainOut.write(buf, len);
        npts++;
    }
    size_t endPos = mainOut.tellp();
    IndexEntry<float> e{outputSize, endPos, npts,
        tileData.xmin, tileData.xmax,
        tileData.ymin, tileData.ymax};
    indexOut.write(reinterpret_cast<char*>(&e), sizeof(e));
    outputSize += endPos;
    return npts;
}

template<typename T>
void Tiles2Minibatch<T>::processTile(TileData<T> &tileData, int threadId, int ticket, vec2f_t* anchorPtr) {
    if (tileData.pts.empty()) {
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
    }
    if (debug_) {
        std::cout << "Thread " << threadId << " finished decoding" << std::endl << std::flush;
    }
    MatrixXd topVals;
    Eigen::MatrixXi topIds;
    findTopK(topVals, topIds, minibatch.phi, topk_);
    if (debug_) {
        std::cout << "Thread " << threadId << " start writing to output" << std::endl << std::flush;
    }
    uint32_t npts;
    if (useTicketSystem) {
        std::unique_lock<std::mutex> lock(ticketMutex);
        ticketCondition.wait(lock, [this, ticket]() {
            return ticket == currentTicket.load(std::memory_order_acquire);
        });
        if (outputOriginalData) {
            npts = outputOriginalDataWithPixelResult(tileData, topVals, topIds);
        } else {
            npts = outputPixelResult(tileData, topVals, topIds);
        }
        currentTicket.fetch_add(1, std::memory_order_release);
        lock.unlock();
        ticketCondition.notify_all();
    } else {
        if (outputOriginalData) {
            npts = outputOriginalDataWithPixelResult(tileData, topVals, topIds);
        } else {
            npts = outputPixelResult(tileData, topVals, topIds);
        }
    }


    notice("Thread %d fit minibatch with %d anchors and output %d internal pixels", threadId, nAnchors, npts);
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
    KDTreeVectorOfVectorsAdaptor<vec2f_t, float> reftree(2, *fixedAnchors, {10});
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
    // 3 Lloyd iteration
    float eps = 1.;
    float xmin = tileData.xmin - eps;
    float ymin = tileData.ymin - eps;
    float xrange = tileData.xmax - tileData.xmin + eps * 2;
    float yrange = tileData.ymax - tileData.ymin + eps * 2;
    cv::Rect2f rect(xmin, ymin, xrange, yrange);
    std::vector<int32_t> indices;
    for (int32_t t = 0; t < nLloydIter; ++t) {
        try {
            cv::Subdiv2D subdiv(rect);
            std::vector<std::vector<cv::Point2f>> facets;
            std::vector<cv::Point2f> facetCenters;
            subdiv.insert(anchors);
            subdiv.getVoronoiFacetList(indices, facets, facetCenters);
            for (size_t i = nFixed; i < nAnchors; ++i) {
                if (i >= facets.size() || facets[i].empty())
                    continue;
                std::vector<cv::Point2f> poly = clipPolygonToRect(facets[i], rect);
                cv::Point2f centroid = centroidOfPolygon(poly);
                anchors[i] = centroid;
            }
        } catch (...) {
            warning("Error in the %d-th Lloyd iteration", t+1);
        }

    }
    // 4 Aggregate pixels and initialize anchors
    PointCloudCV<float> pc;
    pc.pts = std::move(anchors);
    kd_tree_cv2f_t kdtree(2, pc, {10});
    l2radius = hexGrid.size * hexGrid.size * 0.827;
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;
    std::vector<std::unordered_map<uint32_t, float>> docAgg(nAnchors);
    for (const auto& pt : tileData.pts) {
        float xy[2] = {(float) pt.x, (float) pt.y};
        size_t n = kdtree.radiusSearch(xy, l2radius, indices_dists);
        if (n == 0) {
            continue;
        }
        for (size_t i = 0; i < n; ++i) {
            docAgg[indices_dists[i].first][pt.idx] += pt.ct;
        }
    }
    std::vector<Document> docs;
    anchors.clear();
    for (int32_t j = 0; j < nAnchors; ++j) {
        auto& kv = docAgg[j];
        if (kv.empty()) {
            continue;
        }
        double sum = std::accumulate(kv.begin(), kv.end(), 0.0, [](double a, const auto& b) { return a + b.second; });
        if (sum < anchorMinCount) {
            continue;
        }
        anchors.push_back(pc.pts[j]);
        Document doc;
        if (weighted) {
            for (const auto& item : kv) {
                doc.ids.push_back(item.first);
                doc.cnts.push_back(item.second * lineParser.weights[item.first]);
            }
        } else {
            for (const auto& item : kv) {
                doc.ids.push_back(item.first);
                doc.cnts.push_back(item.second);
            }
        }
        docs.push_back(std::move(doc));
    }
    minibatch.gamma = lda.transform(docs);
    for (int i = 0; i < minibatch.gamma.rows(); ++i) {
        double sum = minibatch.gamma.row(i).sum();
        if (sum > 0) {
            minibatch.gamma.row(i) /= sum / K_;
        }
    }
    minibatch.n = docs.size();
    minibatch.M = M_;
    return anchors.size();
}

template class Tiles2Minibatch<int32_t>;
template class Tiles2Minibatch<float>;
