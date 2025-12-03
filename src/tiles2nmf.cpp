#include "tiles2nmf.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <type_traits>
#include <cstdint>
#include <utility>
#include <unordered_map>

void PixelEM::run_em_mlr(Minibatch& batch, EMstats& stats, int max_iter, double tol) const {
    if (!mlr_initialized_) {
        error("%s: not initialized, run init_global_parameter first", __func__);
    }
    if (batch.M != M_) {
        error("%s: dimension M mismatch (%d vs %d)", __func__, batch.M, M_);
    }
    max_iter = (max_iter > 0) ? max_iter : max_iter_;
    tol = (tol > 0) ? tol : tol_;

    bool theta_init = true;
    if (batch.theta.cols() != K_) {
        batch.theta = RowMajorMatrixXf::Zero(batch.n, K_);
        theta_init = false;
    }
    RowMajorMatrixXf theta_old = batch.theta;
    RowMajorMatrixXf Xb = batch.mtx * logBeta_; // N x K
    double meanchange_theta = tol + 1.0;
    debug("%s: Starting EM", __func__);
    int it = 0;
    for (; it < max_iter; it++) {
        SparseRowMatf ymtx = batch.psi.transpose() * batch.mtx; // n x M
        batch.theta = mlr_.predict(ymtx); // n x K
        for (int j = 0; j < batch.N; ++j) {
            for (SparseRowMatf::InnerIterator it1(batch.psi, j), it2(batch.wij, j); it1 && it2; ++it1, ++it2) {
                int i = it1.col();
                it1.valueRef() = Xb.row(j).dot(batch.theta.row(i)) + it2.value();
            }
        }
        expitAndRowNormalize(batch.psi);

        // check convergence
        if (theta_init) {
            meanchange_theta = mean_max_row_change(theta_old, batch.theta);
        } else {
            theta_init = true;
        }
        if (meanchange_theta < tol && it > 0) {
            break;
        }
        theta_old = batch.theta;
        debug("%s: EM iter %d, mean max change in theta: %.4e", __func__, it, meanchange_theta);
    }
    stats.niter = it;
    stats.last_change = meanchange_theta;
    batch.phi = batch.psi * batch.theta; // N x K
    notice("%s: Finished EM with %d iterations. Final mean max change in theta: %.4e", __func__, it, meanchange_theta);
}

void PixelEM::run_em_pnmf(Minibatch& batch, EMstats& stats, int max_iter, double tol, std::vector<std::unordered_map<uint32_t, float>>* phi0) const {
    if (!pnmf_initialized_) {
        error("%s: not initialized, run init_global_parameter first", __func__);
    }
    if (batch.M != M_) {
        error("%s: dimension M mismatch (%d vs %d)", __func__, batch.M, M_);
    }
    max_iter = (max_iter > 0) ? max_iter : max_iter_;
    tol = (tol > 0) ? tol : tol_;
    SparseRowMatf* mtx = &(batch.mtx);
    SparseRowMatf  mtx_local;
    if (fit_background_) {
        assert(phi0 != nullptr);
        mtx_local = batch.mtx;
        for (int i = 0; i < batch.N; ++i) {
            for (SparseRowMatf::InnerIterator it(mtx_local, i); it; ++it) {
                it.valueRef() *= (1.0f - pi_);
            }
        }
        mtx = &mtx_local;
        phi0->reserve(batch.mtx.rows());
        for (int i = 0; i < batch.mtx.rows(); ++i) {
            std::unordered_map<uint32_t, float> phi0_i;
            for (SparseRowMatf::InnerIterator it(batch.mtx, i); it; ++it) {
                phi0_i[it.col()] = 0;
            }
            phi0->push_back(std::move(phi0_i));
        }
    }

    bool theta_init = true;
    if (batch.theta.cols() != K_) {
        batch.theta = RowMajorMatrixXf::Zero(batch.n, K_);
        theta_init = false;
    }
    std::vector<bool> theta_valid(batch.n, theta_init);
    RowMajorMatrixXf theta_old = batch.theta;
    double meanchange_theta = tol + 1.0;
    double avg_iters = 0.0f;
    // RowMajorMatrixXf Xb = batch.mtx * logBeta_; // N x K
    debug("%s: Starting EM", __func__);
    int it = 0;
    for (; it < max_iter; it++) {
        SparseRowMatf ymtx = batch.psi.transpose() * (*mtx); // n x M
        std::vector<int32_t> niters(batch.n, 0);
        float nkept = 0;
        MLEStats stats;
        for (int j = 0; j < batch.n; j++) { // solve for \theta_j
            float rowsum = 0.0;
            size_t nnz = ymtx.row(j).nonZeros();
            Document y;
            y.ids.reserve(nnz);
            y.cnts.reserve(nnz);
            for (SparseRowMatf::InnerIterator it(ymtx, j); it; ++it) {
                if (it.value() <= 0.f) {
                    continue;
                }
                y.ids.push_back(it.col());
                y.cnts.push_back(it.value());
                rowsum += it.value();
            }
            if (rowsum <= min_ct_) {
                batch.theta.row(j).setZero();
                theta_valid[j] = false;
                continue;
            }
            nkept += 1;
            double c = rowsum / size_factor_;
            VectorXd b; // K
            if (theta_valid[j]) {
                b = batch.theta.row(j).transpose().cast<double>();
            }
            if (exact_) {
                pois_log1p_mle_exact(beta_, y, c, mle_opts_, b, stats);
            } else {
                pois_log1p_mle(beta_, y, c, mle_opts_, b, stats);
            }
            niters[j] = stats.optim.niters;
            batch.theta.row(j) = b.transpose().cast<float>();
            if (batch.theta.row(j).minCoeff() > 1e-8)
                theta_valid[j] = true;
        }
        MatrixXf loglam = (((beta_f_ * batch.theta.transpose()).array().exp() - 1.0)).max(1e-12).log(); // M x n

        RowMajorMatrixXf theta_norm = rowNormalize(batch.theta);
        for (int i = 0; i < batch.N; ++i) {
            for (SparseRowMatf::InnerIterator it1(batch.psi, i), it2(batch.wij, i); it1 && it2; ++it1, ++it2) {
                int j = it1.col();
                it1.valueRef() = mtx->row(i).dot(loglam.col(j)) + it2.value();
                // if (!theta_valid[j]) {
                //     it1.valueRef() = -std::numeric_limits<float>::infinity();
                //     continue;
                // }
                // it1.valueRef() = Xb.row(i).dot(theta_norm.row(j)); //+ it2.value();
            }
        }

        int32_t total_iters = std::accumulate(niters.begin(), niters.end(), 0);
        avg_iters = static_cast<double>(total_iters) / nkept;
        int32_t nskip = (int32_t) (batch.n - nkept);
        debug("%s: Completed %d-th EM iteration with %d anchors (skipped %d), average %.1f iterations per optim", __func__, it, batch.n, nskip, avg_iters);

        // rowSoftmaxInPlace(batch.psi);
        expitAndRowNormalize(batch.psi);

        if (fit_background_) {
            double x1 = 0., xtot = 0.;
            loglam = loglam.array().exp();
            for (int i = 0; i < batch.N; ++i) {
                for (SparseRowMatf::InnerIterator it1(mtx_local, i), it2(batch.mtx, i); it1 && it2; ++it1, ++it2) {
                    int m = it1.col();
                    float p1 = 0.;
                    for (SparseRowMatf::InnerIterator it(batch.psi, i); it; ++it) {
                        int j = it.col();
                        p1 += it.value() * loglam(m, j) / size_factor_;
                    }
                    p1 = (1.0f - pi_) * p1;
                    float p0 = pi_ * beta0_(m);
                    p1 = (p0 + p1 > 0) ? p1 / (p0 + p1) : 1.0f;
                    it1.valueRef() = it2.value() * p1;
                    xtot += it2.value();
                    x1 += it1.value();
                    (*phi0)[i][m] = 1.0f - p1;
                }
            }
            double f0 = 1. - x1 / xtot;
            notice("%s: Average background fraction = %.4f", __func__, f0);
        }

        if (theta_init) {
            meanchange_theta = mean_max_row_change(theta_old, batch.theta);
        } else {
            theta_init = true;
        }
        if (meanchange_theta < tol && it > 0) {
            break;
        }
        theta_old = batch.theta;
        debug("%s: EM iter %d, mean max change in theta: %.4e", __func__, it, meanchange_theta);
    }
    stats.niter = it;
    stats.last_avg_internal_niters = avg_iters;
    stats.last_change = meanchange_theta;
    rowNormalizeInPlace(batch.theta);
    batch.phi = batch.psi * batch.theta; // N x K
    debug("%s: Finished EM", __func__);

}

template<typename T>
Tiles2NMF<T>::Tiles2NMF(int nThreads, double r,
        const std::string& outPref, const std::string& tmpDir,
        PixelEM& empois, TileReader& tileReader, lineParserUnival& lineParser, HexGrid& hexGrid, int32_t nMoves,
        unsigned int seed, double c, double h, double res, int32_t topk,
        int32_t verbose, int32_t debug)
    : Tiles2MinibatchBase<T>(nThreads, r + hexGrid.size, tileReader, outPref, &tmpDir, debug), distR_(r),
        empois_(empois), lineParser_(lineParser),
        hexGrid_(hexGrid), nMoves_(nMoves),
        seed_(seed), anchorMinCount_(c),
        pixelResolution_(static_cast<float>(res)),
        verbose_(verbose)
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
         Base::setExtendedSchema(sizeof(RecordT<T>));
    }

    M_ = empois_.get_M();
    K_ = empois_.get_K();
    if (M_ <= 0 || K_ <= 0) {
        error("%s: Invalid beta dimensions (%d x %d)", __func__, M_, K_);
    }

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
    int32_t nPixels = Base::buildMinibatchCore(
        tileData, anchors, minibatch,
        pixelResolution_, distR_, distNu_);
    if (nPixels <= 0) {
        return nPixels;
    }

    for (int i = 0; i < minibatch.wij.outerSize(); ++i) {
        for (typename SparseRowMatf::InnerIterator it(minibatch.wij, i); it; ++it) {
            it.valueRef() = log(it.value());
        }
    }

    return nPixels;
}

template<typename T>
int32_t Tiles2NMF<T>::initAnchors(TileData<T>& tileData, std::vector<cv::Point2f>& anchors, Minibatch& minibatch) {
    anchors.clear();
    std::unordered_map<HexGrid::GridKey, uint32_t, HexGrid::GridKeyHash> GridPts;
    auto assign_pt = [&](const auto& pt) {
        for (int32_t ir = 0; ir < nMoves_; ++ir) {
            for (int32_t ic = 0; ic < nMoves_; ++ic) {
                int32_t hx, hy;
                hexGrid_.cart_to_axial(hx, hy, pt.x, pt.y, ic * 1. / nMoves_, ir * 1. / nMoves_);
                GridPts[std::make_tuple(hx, hy, ic, ir)] += pt.ct;
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
    for (auto& kv : GridPts) {
        if (kv.second < anchorMinCount_) {
            continue;
        }
        auto& key = kv.first;
        int32_t hx = std::get<0>(key);
        int32_t hy = std::get<1>(key);
        int32_t ic = std::get<2>(key);
        int32_t ir = std::get<3>(key);
        float x, y;
        hexGrid_.axial_to_cart(x, y, hx, hy, ic * 1. / nMoves_, ir * 1. / nMoves_);
        anchors.emplace_back(x, y);
    }
    minibatch.n = anchors.size();
    return anchors.size();
}

template<typename T>
void Tiles2NMF<T>::processTile(TileData<T>& tileData, int threadId, int ticket, vec2f_t* anchorPtr) {
    (void)anchorPtr;
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
    PixelEM::EMstats stats;
    // store background probability (per pixel per feature)
    std::vector<std::unordered_map<uint32_t, float>> phi0;
    empois_.run_em(minibatch, stats, -1, -1, &phi0);
    debug("%s: Thread %d (ticket %d) finished EM", __func__, threadId, ticket);

    MatrixXf topVals;
    Eigen::MatrixXi topIds;
    findTopK(topVals, topIds, minibatch.phi, topk_);
    ProcessedResult result = Base::formatPixelResult(tileData, topVals, topIds, ticket, &phi0);
    resultQueue.push(std::move(result));
    if (Base::outputAnchor_) {
        MatrixXf anchorTopVals;
        Eigen::MatrixXi anchorTopIds;
        findTopK(anchorTopVals, anchorTopIds, minibatch.theta, topk_);
        auto anchorResult = Base::formatAnchorResult(
            anchors, anchorTopVals, anchorTopIds, ticket,
            tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
        anchorQueue.push(std::move(anchorResult));
    }

    notice("Thread %d (ticket %d) fit minibatch with %d anchors and output %lu internal pixels in %d iterations. Final mean max change in theta: %.1e (final averaged inner iterations %.1f)", threadId, ticket, nAnchors, result.npts, stats.niter, stats.last_change, stats.last_avg_internal_niters);

}

// explicit instantiations
template class Tiles2NMF<int32_t>;
template class Tiles2NMF<float>;
