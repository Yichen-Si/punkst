#include "tiles2slda.hpp"
#include <algorithm>

template<typename T>
Tiles2SLDA<T>::Tiles2SLDA(int nThreads, double r,
        const std::string& outPref, const std::string& tmpDir,
        LatentDirichletAllocation& lda,
        TileReader& tileReader, lineParserUnival& lineParser,
        HexGrid& hexGrid, int32_t nMoves,
        unsigned int seed,
        double c, double h, double res, int32_t N, int32_t k,
        int32_t verbose, int32_t debug)
    : Tiles2MinibatchBase<T>(nThreads, r + hexGrid.size, tileReader, outPref, &tmpDir, debug),
      distR_(r), lda_(lda), lineParser_(lineParser), hexGrid_(hexGrid), nMoves_(nMoves), anchorMinCount_(c), pixelResolution_(res)
{
    lineParserPtr = &lineParser_;
    useExtended_ = lineParser_.isExtended;
    topk_ = k;

    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        assert(tileReader.getCoordType() == CoordType::FLOAT && "Template type does not match with TileReader coordinate type");
    } else if constexpr (std::is_same_v<T, int32_t>) {
        assert(tileReader.getCoordType() == CoordType::INTEGER && "Template type does not match with TileReader coordinate type");
    } else {
        error("%s: Unsupported coordinate type", __func__);
    }
    if (pixelResolution_ <= 0) {
        pixelResolution_ = 1;
    }
    if (useExtended_) {
        Base::setExtendedSchema(sizeof(RecordT<T>));
    }
    M_ = lda_.get_n_features();
    if (lineParser_.isFeatureDict) {
        assert((M_ == lineParser_.featureDict.size()) && "Feature number does not match");
        featureNames.resize(M_);
        for (const auto& entry : lineParser_.featureDict) {
            featureNames[entry.second] = entry.first;
        }
    } else if (lineParser_.weighted) {
        assert(M_ == lineParser_.weights.size() && "Feature number does not match");
    }
    lda_.set_nthreads(1); // because we parallelize by tile
    K_ = lda_.get_n_topics();
    distNu_ = std::log(0.5) / std::log(h);
    if (N <= 0) {
        N = lda_.get_N_global() * 100;
    }
    pseudobulk_ = MatrixXf::Zero(M_, K_);
    slda_.init(K_, M_, N, seed);
    slda_.init_global_parameter(lda_.get_model());
    slda_.verbose_ = verbose;
    slda_.debug_ = debug;
    if (featureNames.size() == 0) {
        featureNames.resize(M_);
        for (int32_t i = 0; i < M_; ++i) {
            featureNames[i] = std::to_string(i);
        }
    }

    if (debug_ > 0) {
        std::cout << "Check model initialization\n" << std::fixed << std::setprecision(4);
        const auto& lambda = slda_.get_lambda();
        const auto& Elog_beta = slda_.get_Elog_beta(); // M x K
        for (int32_t i = 0; i < std::min(3, K_) ; ++i) {
            std::cout << "\tLambda " << i << ": ";
            for (int32_t j = 0; j < std::min(5, M_); ++j) {
                std::cout << lambda(j, i) << " ";
            }
            std::cout << "\n\tElog_beta: ";
            for (int32_t j = 0; j < std::min(5, M_); ++j) {
                std::cout << Elog_beta(j, i) << " ";
            }
            std::cout << "\n";
        }
    }

    notice("Initialized Tiles2Minibatch");
}

template<typename T>
int32_t Tiles2SLDA<T>::initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {
    std::vector<SparseObs> documents;
    Base::buildAnchors(tileData, anchors, documents, hexGrid_, nMoves_, anchorMinCount_);
    if (documents.empty()) {
        return 0;
    }

    minibatch.gamma = lda_.transform(documents).template cast<float>();
    // TODO: need to test if scaling/normalizing gamma is better
    // scale each row so that the mean is 1
    for (int i = 0; i < minibatch.gamma.rows(); ++i) {
        float sum = minibatch.gamma.row(i).sum();
        if (sum > 0) {
            minibatch.gamma.row(i) /= sum / K_;
        }
    }
    minibatch.n = documents.size();
    return anchors.size();
}

template<typename T>
int32_t Tiles2SLDA<T>::makeMinibatch(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {
    int32_t nPixels = Base::buildMinibatchCore(
        tileData, anchors, minibatch,
        pixelResolution_, distR_, distNu_);

    if (nPixels <= 0) {
        return nPixels;
    }

    for (int i = 0; i < minibatch.wij.outerSize(); ++i) {
        for (typename SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(minibatch.wij, i); it; ++it) {
            it.valueRef() = logit(it.value());
        }
    }

    return nPixels;
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
    double gridDist = hexGrid_.size/nMoves_;
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
    for (int32_t t = 0; t < nLloydIter_; ++t) {
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
        if (sum < anchorMinCount_) continue;
        finalAnchors.push_back(anchors[j]);
        Document doc;
        if (lineParser_.weighted) {
            for (const auto& item : docAgg[j]) {
                doc.ids.push_back(item.first);
                doc.cnts.push_back(item.second * lineParser_.weights[item.first]);
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
    minibatch.gamma = lda_.transform(docs).template cast<float>();
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
void Tiles2SLDA<T>::processTile(TileData<T> &tileData, int threadId, int ticket, vec2f_t* anchorPtr) {
    if (tileData.pts.empty() && tileData.extPts.empty()) {
        return;
    }
    std::vector<cv::Point2f> anchors;
    Minibatch minibatch;
    int32_t nAnchors = initAnchorsHybrid(tileData, anchors, minibatch, anchorPtr);

    if (nAnchors == 0) {
        return;
    }
    int32_t nPixels = makeMinibatch(tileData, anchors, minibatch);
    if (debug_) {
        std::cout << "Thread " << threadId << " made minibatch with " << nPixels << " pixels. ";
        int32_t nnz = minibatch.wij.nonZeros();
        std::cout << nnz << " " << (float) nnz / minibatch.n << std::endl << std::flush;
        std::vector<float> colsums(minibatch.n, 0.0f);
        for (int i = 0; i < minibatch.wij.outerSize(); ++i) {
            for (typename SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(minibatch.wij, i); it; ++it) {
                colsums[it.col()] += it.value();
            }
        }
        std::cout << "  " << std::accumulate(colsums.begin(), colsums.end(), 0.0f) / minibatch.n << " " << std::max_element(colsums.begin(), colsums.end())[0] << std::endl << std::flush;
    }
    if (nPixels < 10) {
        return;
    }
    int32_t n_iter = 0;
    double delta = 0.0;
    auto smtx = slda_.do_e_step(minibatch, true, &n_iter, &delta); // M x K
    {
        std::lock_guard<std::mutex> lock(pseudobulkMutex_);
        pseudobulk_ += smtx;
        if (debug_) {
            std::cout << "Thread " << threadId << " updated pseudobulk.\n";
            // std::cout << "    Peek: " << std::fixed << std::setprecision(0);
            // for (int32_t i = 0; i < std::min(3, K_); ++i) {
            //     for (int32_t j = 0; j < std::min(5, M_); ++j) {
            //         std::cout << smtx(i, j) << " ";
            //     }
            //     std::cout << "\n    ";
            // }
            std::cout << "    Current sums: ";
            auto colsums = pseudobulk_.colwise().sum();
            for (int32_t i = 0; i < K_; ++i) {
                std::cout << colsums(i) << " ";
            }
            std::cout << std::endl << std::flush;
        }
    }

    MatrixXf topVals;
    Eigen::MatrixXi topIds;
    findTopK(topVals, topIds, minibatch.phi, topk_);
    ProcessedResult result;
    if (outputOriginalData_) {
        result = Base::formatPixelResultWithOriginalData(tileData, topVals, topIds, ticket);
    } else {
        result = Base::formatPixelResult(tileData, topVals, topIds, ticket);
    }
    resultQueue.push(std::move(result));
    if (outputAnchor_) {
        MatrixXf prob = rowNormalize(minibatch.gamma);
        MatrixXf anchorTopVals;
        Eigen::MatrixXi anchorTopIds;
        findTopK(anchorTopVals, anchorTopIds, prob, topk_);
        auto anchorResult = Base::formatAnchorResult(
            anchors, anchorTopVals, anchorTopIds, ticket,
            tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
        anchorQueue.push(std::move(anchorResult));
    }
    notice("Thread %d (ticket %d) fit minibatch with %d anchors and output %lu internal pixels in %d iterations. Final mean max change in phi: %.1e", threadId, ticket, nAnchors, result.npts, n_iter, delta);
}

template<typename T>
void Tiles2SLDA<T>::postRun() {
    writePseudobulkToTsv();
}

template<typename T>
void Tiles2SLDA<T>::writePseudobulkToTsv() {
    std::string pseudobulkFile = outPref + ".pseudobulk.tsv";
    std::ofstream oss(pseudobulkFile, std::ios::out);
    if (!oss) error("Error opening pseudobulk output file: %s", pseudobulkFile.c_str());
    oss << "Feature";
    const auto factorNames = lda_.get_topic_names();
    for (int32_t i = 0; i < K_; ++i) oss << "\t" << factorNames[i];
    oss << "\n" << std::setprecision(probDigits) << std::fixed;
    for (int32_t i = 0; i < M_; ++i) {
        oss << featureNames[i];
        for (int32_t j = 0; j < K_; ++j) oss << "\t" << pseudobulk_(i, j);
        oss << "\n";
    }
    oss.close();
}

// explicit instantiations
template class Tiles2SLDA<int32_t>;
template class Tiles2SLDA<float>;
