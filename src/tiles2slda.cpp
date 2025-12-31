#include "tiles2slda.hpp"
#include <algorithm>

template<typename T>
Tiles2SLDA<T>::Tiles2SLDA(int nThreads, double r,
        const std::string& outPref, const std::string& tmpDir,
        LatentDirichletAllocation& lda,
        TileReader& tileReader, lineParserUnival& lineParser,
        const MinibatchIoConfig& ioConfig,
        HexGrid& hexGrid, int32_t nMoves,
        unsigned int seed,
        double c, double h, double res, int32_t N, int32_t k,
        int32_t verbose, int32_t debug)
    : Tiles2MinibatchBase<T>(nThreads, r + hexGrid.size, tileReader, outPref, &lineParser, ioConfig, &tmpDir, res, debug),
      distR_(r), lda_(lda), lineParser_(lineParser), hexGrid_(hexGrid), nMoves_(nMoves), anchorMinCount_(c)
{
    topk_ = k;
    if constexpr (std::is_same_v<T, float> || std::is_same_v<T, double>) {
        assert(tileReader.getCoordType() == CoordType::FLOAT && "Template type does not match with TileReader coordinate type");
    } else if constexpr (std::is_same_v<T, int32_t>) {
        assert(tileReader.getCoordType() == CoordType::INTEGER && "Template type does not match with TileReader coordinate type");
    } else {
        error("%s: Unsupported coordinate type", __func__);
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
void Tiles2SLDA<T>::set_background_prior(VectorXf& eta0, double a0, double b0, bool outputExpand) {
    if (eta0.size() != M_) {
        error("%s: size of background prior (%d) does not match feature number (%d)", __func__, (int32_t)eta0.size(), M_);
    }
    slda_.set_background_prior(eta0, a0, b0);
    fitBackground_ = true;
    Base::outputBackgroundProbDense_ = !outputExpand;
    Base::outputBackgroundProbExpand_ = outputExpand;
    Base::configureOutputMode();
}

template<typename T>
void Tiles2SLDA<T>::set_background_prior(std::string& bgModelFile, double a0, double b0, bool outputExpand) {
    VectorXf eta0 = VectorXf::Zero(M_);
    std::ifstream fin(bgModelFile);
    if (!fin.is_open()) {
        error("%s: cannot open background model file %s", __func__, bgModelFile.c_str());
    }
    std::string line;
    std::vector<std::string> tokens;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        split(tokens, "\t ", line, 3, true, true, true);
        if (tokens.size() < 2) continue;
        std::string& feature = tokens[0];
        auto it = lineParser_.featureDict.find(feature);
        if (it == lineParser_.featureDict.end()) {
            continue;
        }
        if (!str2float(tokens[1], eta0(it->second))) {
            error("%s: invalid value for feature %s in background model file (%s)", __func__, feature.c_str(), line.c_str());
        }
    }
    fin.close();
    slda_.set_background_prior(eta0, a0, b0);
    fitBackground_ = true;
    Base::outputBackgroundProbDense_ = !outputExpand;
    Base::outputBackgroundProbExpand_ = outputExpand;
    Base::configureOutputMode();
}

template<typename T>
int32_t Tiles2SLDA<T>::initAnchors(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, Minibatch& minibatch) {
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
int32_t Tiles2SLDA<T>::makeMinibatch(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, Minibatch& minibatch) {
    int32_t nPixels = Base::buildMinibatchCore(
        tileData, anchors, minibatch, distR_, distNu_);

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
int32_t Tiles2SLDA<T>::initAnchorsHybrid(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, Minibatch& minibatch, const vec2f_t* fixedAnchors) {
    if (Base::coordDim_ == MinibatchCoordDim::Dim3) {
        if (fixedAnchors && !fixedAnchors->empty()) {
            warning("%s: fixed anchors are ignored in 3D mode", __func__);
        }
        return initAnchors(tileData, anchors, minibatch);
    }
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
        PointCloud<float> pc;
        pc.pts = anchors;
        kd_tree_f2_t kdtree(2, pc, {10});
        docAgg.assign(nAnchors, std::unordered_map<uint32_t, float>());
        std::vector<AnchorPoint> newAnchorCoords(nAnchors, AnchorPoint(0.f, 0.f, 0.f));
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
    std::vector<AnchorPoint> finalAnchors;
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
    if (tileData.pts.empty() && tileData.extPts.empty() &&
        tileData.pts3d.empty() && tileData.extPts3d.empty()) {
        return;
    }
    std::vector<AnchorPoint> anchors;
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
    RowMajorMatrixXf smtx(M_, K_);
    float f0;
    // store background probability (per pixel per feature)
    std::vector<std::unordered_map<uint32_t, float>> phi0;
    std::vector<std::unordered_map<uint32_t, float>>* phi0_ptr = nullptr;
    if (fitBackground_) {
        f0 = slda_.do_e_step_bg(minibatch, phi0, &smtx, &n_iter, &delta);
        phi0_ptr = &phi0;
    } else {
        f0 = slda_.do_e_step_standard(minibatch, &smtx, &n_iter, &delta);
    }
    {
        std::lock_guard<std::mutex> lock(pseudobulkMutex_);
        pseudobulk_ += smtx;
        if (debug_) {
            std::cout << "Thread " << threadId << " updated pseudobulk.\n";
            std::cout << "    Current sums: ";
            auto colsums = pseudobulk_.colwise().sum();
            for (int32_t i = 0; i < K_; ++i) {
                std::cout << colsums(i) << " ";
            }
            std::cout << std::endl << std::flush;
        }
    }

    MatrixXf topVals;
    MatrixXi topIds;
    findTopK(topVals, topIds, minibatch.phi, topk_);
    ResultBuf result = Base::formatPixelResult(tileData, topVals, topIds, ticket, phi0_ptr);
    resultQueue.push(std::move(result));
    if (Base::outputAnchor_) {
        MatrixXf prob = rowNormalize(minibatch.gamma);
        MatrixXf anchorTopVals;
        MatrixXi anchorTopIds;
        findTopK(anchorTopVals, anchorTopIds, prob, topk_);
        auto anchorResult = Base::formatAnchorResult(
            anchors, anchorTopVals, anchorTopIds, ticket,
            tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
        anchorQueue.push(std::move(anchorResult));
    }

    char buf[256];
    int l = snprintf(buf, sizeof(buf), "Thread %d (ticket %d) fit minibatch with %d anchors and output %u internal pixels in %d iterations. Final mean max change in phi: %.1e", threadId, ticket, nAnchors, result.npts, n_iter, delta);
    if (fitBackground_) {
        l += snprintf(buf + l, sizeof(buf) - l, " (%.3f background)", f0);
    }
    notice("%s", buf);
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
    if (fitBackground_) {oss << "\tBackground";}
    const VectorXf& bgProb = slda_.get_background_count();
    const auto factorNames = lda_.get_topic_names();
    for (int32_t i = 0; i < K_; ++i) oss << "\t" << factorNames[i];
    oss << "\n" << std::setprecision(probDigits) << std::fixed;
    for (int32_t i = 0; i < M_; ++i) {
        oss << featureNames[i];
        if (fitBackground_) {oss << "\t" << bgProb(i);}
        for (int32_t j = 0; j < K_; ++j) oss << "\t" << pseudobulk_(i, j);
        oss << "\n";
    }
    oss.close();
}

// explicit instantiations
template class Tiles2SLDA<int32_t>;
template class Tiles2SLDA<float>;
