#include "tiles2minibatch.hpp"

int32_t Tiles2Minibatch::parseOneTile(TileData& tileData, TileKey tile) {
    std::unique_ptr<BoundedReadline> iter;
    try {
        iter = tileReader.get_tile_iterator(tile.row, tile.col);
    } catch (const std::exception &e) {
        warning("%s", e.what());
        return -1;
    }
    tileData.clear();
    std::string line;
    int32_t npt = 0;
    while (iter->next(line)) {
        Record rec;
        int32_t idx = lineParser.parse(rec, line);
        if (idx < -1) {
            error("Error parsing line: %s", line.c_str());
        }
        if (idx == -1 || idx >= M_) {
            continue;
        }
        tileData.pts.push_back(rec);
        std::vector<uint32_t> bufferidx;
        if (pts2buffer(bufferidx, rec.x, rec.y, tile) == 1) {
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

int32_t Tiles2Minibatch::parseBoundaryFile(TileData& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
    std::lock_guard<std::mutex> lock(*(bufferPtr->mutex));
    std::ifstream ifs(bufferPtr->tmpFile, std::ios::binary);
    if (!ifs) {
        int32_t bufRow, bufCol;
        bool isVertical = decodeTempFileKey(bufferPtr->key, bufRow, bufCol);
        warning("Error opening temporary file (%d, %d, %d): %s", int32_t (isVertical), bufRow, bufCol, (bufferPtr->tmpFile).c_str());
        return -1;
    }
    int npt = 0;
    tileData.clear();
    while (true) {
        Record rec;
        ifs.read(reinterpret_cast<char*>(&rec), sizeof(Record));
        if (ifs.gcount() != sizeof(Record)) break;
        tileData.pts.push_back(rec);
        if (isInternalToBuffer(rec.x, rec.y, bufferPtr->key)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

int32_t Tiles2Minibatch::initAnchors(TileData& tileData, PointCloud<double>& anchors, Minibatch& minibatch) {

    anchors.pts.clear();
    std::vector<Document> documents;
    for (int32_t ir = 0; ir < nMoves; ++ir) {
        for (int32_t ic = 0; ic < nMoves; ++ic) {
            std::unordered_map<int64_t, std::unordered_map<uint32_t, double>> hexAggregation;
            for (const auto& pt : tileData.pts) {
                int32_t hx, hy;
                hexGrid.cart_to_axial(hx, hy, (double) pt.x, (double) pt.y, ic*1./nMoves, ir*1./nMoves);
                int64_t key = (static_cast<int64_t>(hx) << 32) | (static_cast<uint32_t>(hy));
                hexAggregation[key][pt.idx] += pt.ct;
            }
            // Create the vector of Document from the aggregated data.
            for (auto& hexEntry : hexAggregation) {
                Document doc;
                for (auto& featurePair : hexEntry.second) {
                    doc.ids.push_back(featurePair.first);
                    doc.cnts.push_back(featurePair.second);
                }
                double sum = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
                if (sum < anchorMinCount) {
                    continue;
                }
                if (weighted) {
                    for (size_t i = 0; i < doc.ids.size(); ++i) {
                        doc.cnts[i] *= lineParser.weights[doc.ids[i]];
                    }
                }
                documents.push_back(std::move(doc));
                double x, y;
                int32_t hx = static_cast<int32_t>(hexEntry.first >> 32);
                int32_t hy = static_cast<int32_t>(hexEntry.first & 0xFFFFFFFF);
                hexGrid.axial_to_cart(x, y, hx, hy, ic*1./nMoves, ir*1./nMoves);
                anchors.pts.emplace_back(x, y);
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
    return anchors.pts.size();
}

int32_t Tiles2Minibatch::makeMinibatch(TileData& tileData, PointCloud<double>& anchors, Minibatch& minibatch) {

    nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>,
        PointCloud<double>, 2> kdtree(2, anchors, {10});
    std::vector<nanoflann::ResultItem<uint32_t, double>> indices_dists;

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
        double xy[2] = {x * pixelResolution, y * pixelResolution};
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

int32_t Tiles2Minibatch::outputOriginalDataWithPixelResult(const TileData& tileData, const MatrixXd& topVals, const Eigen::MatrixXi& topIds) {
    std::lock_guard<std::mutex> lock(mainOutMutex);
    int32_t npts = 0;
    int32_t nrows = topVals.rows();
    for (size_t i = 0; i < tileData.idxinternal.size(); ++i) {
        int32_t idxorg = tileData.idxinternal[i]; // index in the original data
        int32_t idx = tileData.orgpts2pixel[idxorg]; // index in the pixel minibatch
        if (idx < 0 || idx >= nrows) {
            continue;
        }
        const Record& rec = tileData.pts[idxorg];
        std::stringstream ss;
        // set precision to 4 decimal for double/float
        ss.precision(4);
        ss << rec.x << "\t" << rec.y << "\t" << featureNames[rec.idx] << "\t" << rec.ct;
        for (int32_t j = 0; j < topk_; ++j) {
            ss << "\t" << topIds(idx, j);
        }
        for (int32_t j = 0; j < topk_; ++j) {
            ss << "\t" << topVals(idx, j);
        }
        ss << "\n";
        mainOut << ss.rdbuf();
        npts++;
    }
    return npts;
}

int32_t Tiles2Minibatch::outputPixelResult(const TileData& tileData, const MatrixXd& topVals, const Eigen::MatrixXi& topIds) {
    std::lock_guard<std::mutex> lock(mainOutMutex);
    size_t N = tileData.coords.size();
    std::vector<bool> internal(N, 0);
    for (auto j : tileData.idxinternal) {
        if (tileData.orgpts2pixel[j] < 0) {
            continue;
        }
        internal[tileData.orgpts2pixel[j]] = true;
    }
    int32_t npts = 0;
    for (size_t j = 0; j < N; ++j) {
        if (!internal[j]) {
            continue;
        }
        std::stringstream ss;
        // set precision to %.2f for coordinates
        ss.precision(2);
        ss << std::fixed << tileData.coords[j].first << "\t" << tileData.coords[j].second;
        // set precision to 4 decimal for probabilities
        ss.precision(4);
        for (int32_t k = 0; k < topk_; ++k) {
            ss << "\t" << topIds(j, k);
        }
        for (int32_t k = 0; k < topk_; ++k) {
            ss << "\t" << topVals(j, k);
        }
        ss << "\n";
        mainOut << ss.rdbuf();
        npts++;
    }
    return npts;
}

void Tiles2Minibatch::processTile(TileData &tileData, int threadId) {
    if (tileData.pts.empty()) {
        return;
    }
    PointCloud<double> anchors;
    Minibatch minibatch;
    int32_t nAnchors = initAnchors(tileData, anchors, minibatch);
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
    auto ret = slda.do_e_step(minibatch, false);
    if (debug_) {
        std::cout << "Thread " << threadId << " finished decoding" << std::endl << std::flush;
    }
    MatrixXd topVals;
    Eigen::MatrixXi topIds;
    findTopK(topVals, topIds, minibatch.phi, topk_);
    if (debug_) {
        std::cout << "Thread " << threadId << " start writing to output" << std::endl << std::flush;
    }
    int32_t npts;
    if (outputOriginalData) {
        npts = outputOriginalDataWithPixelResult(tileData, topVals, topIds);
    } else {
        npts = outputPixelResult(tileData, topVals, topIds);
    }
    notice("Thread %d fit minibatch with %d anchors and output %d internal pixels", threadId, nAnchors, npts);
}

void Tiles2Minibatch::writeHeaderToJson() {
    size_t pos = outputFile.find_last_of(".");
    std::string jsonFile;
    if (pos != std::string::npos) {
        jsonFile = outputFile.substr(0, pos) + ".json";
    } else {
        jsonFile = outputFile + ".json";
    }
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
