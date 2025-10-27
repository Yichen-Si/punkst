#include "tiles2slda.hpp"

template<typename T>
Tiles2SLDA<T>::Tiles2SLDA(int nThreads, double r,
        const std::string& _outPref, const std::string& _tmpDir,
        LatentDirichletAllocation& _lda,
        TileReader& tileReader, lineParserUnival& _lineParser,
        HexGrid& _hexGrid, int32_t _nMoves,
        unsigned int seed,
        double c, double h, double res,
        int32_t M, int32_t N, int32_t k,
        int32_t verbose, int32_t debug)
    : Tiles2MinibatchBase(nThreads, r + _hexGrid.size, tileReader, _outPref, &_tmpDir),
      distR(r), lda(_lda), lineParser(_lineParser), hexGrid(_hexGrid), nMoves(_nMoves), anchorMinCount(c), pixelResolution(res), M_(M), debug_(debug)
{
    topk_ = k;
    lineParserPtr = &this->lineParser;
    useExtended_ = this->lineParser.isExtended;
    outputOriginalData = false;

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

template<typename T>
void Tiles2SLDA<T>::run() {
    setupOutput();
    std::thread writer(&Tiles2SLDA::writerWorker, this);

    // Phase 1: Process tiles
    notice("Phase 1 Launching %d worker threads", nThreads);
    for (int i = 0; i < nThreads; ++i) {
        workThreads.push_back(std::thread(&Tiles2SLDA::tileWorker, this, i));
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
        workThreads.push_back(std::thread(&Tiles2SLDA::boundaryWorker, this, i));
    }
    for (auto &t : workThreads) {
        t.join();
    }

    resultQueue.set_done();
notice("%s: all workers done, waiting for writer to finish", __func__);

    writer.join();
    closeOutput();
    writeHeaderToJson();
    writePseudobulkToTsv();
}

template<typename T>
int32_t Tiles2SLDA<T>::initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {
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
int32_t Tiles2SLDA<T>::makeMinibatch(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {

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
int32_t Tiles2SLDA<T>::initAnchorsHybrid(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch, const vec2f_t* fixedAnchors) {
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
void Tiles2SLDA<T>::tileWorker(int threadId) {
    std::pair<TileKey, int32_t> tileTicket;
    TileKey tile;
    int32_t ticket;
    while (tileQueue.pop(tileTicket)) {
        tile = tileTicket.first;
        ticket = tileTicket.second;
        TileData<T> tileData;
        int32_t ret = Tiles2MinibatchBase::parseOneTile<T>(tileData, tile, lineParser, M_, useExtended_, schema_, recordSize_);
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
void Tiles2SLDA<T>::boundaryWorker(int threadId) {
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
                ret = Tiles2MinibatchBase::parseBoundaryFileExtended(tileData, bufferPtr, schema_, recordSize_);
            } else {
                ret = Tiles2MinibatchBase::parseBoundaryFile(tileData, bufferPtr);
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
void Tiles2SLDA<T>::processTile(TileData<T> &tileData, int threadId, int ticket, vec2f_t* anchorPtr) {
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
        result = Tiles2MinibatchBase::formatPixelResultWithOriginalData(tileData, topVals, topIds, ticket);
    } else {
        result = Tiles2MinibatchBase::formatPixelResult(tileData, topVals, topIds, ticket);
    }

    notice("Thread %d (ticket %d) fit minibatch with %d anchors and output %lu internal pixels", threadId, ticket, nAnchors, result.npts);
    resultQueue.push(std::move(result));
}

template<typename T>
void Tiles2SLDA<T>::writePseudobulkToTsv() {
    std::string pseudobulkFile = outPref + ".pseudobulk.tsv";
    std::ofstream oss(pseudobulkFile, std::ios::out);
    if (!oss) error("Error opening pseudobulk output file: %s", pseudobulkFile.c_str());
    oss << "Feature";
    const auto factorNames = lda.get_topic_names();
    for (int32_t i = 0; i < K_; ++i) oss << "\t" << factorNames[i];
    oss << "\n" << std::setprecision(probDigits) << std::fixed;
    for (int32_t i = 0; i < M_; ++i) {
        oss << featureNames[i];
        for (int32_t j = 0; j < K_; ++j) oss << "\t" << pseudobulk(j, i);
        oss << "\n";
    }
    oss.close();
}

template<typename T>
void Tiles2SLDA<T>::setExtendedSchema() {
    if (!lineParser.isExtended) { useExtended_ = false; return; }
    useExtended_ = true; schema_.clear(); size_t offset = sizeof(RecordT<T>);
    size_t n_ints = lineParser.icol_ints.size(); size_t n_floats = lineParser.icol_floats.size(); size_t n_strs = lineParser.icol_strs.size();
    for (size_t i = 0; i < n_ints; ++i) schema_.push_back({FieldType::INT32, sizeof(int32_t), 0});
    for (size_t i = 0; i < n_floats; ++i) schema_.push_back({FieldType::FLOAT, sizeof(float), 0});
    for (size_t i = 0; i < n_strs; ++i) schema_.push_back({FieldType::STRING, lineParser.str_lens[i], 0});
    for (auto &f : schema_) { f.offset = offset; offset += f.size; }
    recordSize_ = offset;
}

// explicit instantiations
template class Tiles2SLDA<int32_t>;
template class Tiles2SLDA<float>;
