#include "poisnmf.hpp"

void PoissonLog1pNMF::set_nthreads(int nThreads) {
    nThreads_ = nThreads;
    if (nThreads_ > 0) {
        tbb_ctrl_ = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism,
            std::size_t(nThreads_));
    } else {
        tbb_ctrl_.reset();
    }
    nThreads_ = int( tbb::global_control::active_value(
            tbb::global_control::max_allowed_parallelism) );
    notice("PoissonLog1pNMF: Requested %d threads, actual number of threads: %d", nThreads, nThreads_);
}

void PoissonLog1pNMF::fit(const std::vector<SparseObs>& docs,
    const MLEOptions mle_opts, NmfFitOptions nmf_opts, bool reset) {
    size_t N = docs.size();
    if (N == 0) {
        warning("%s: Empty input", __func__);
        return;
    }
    auto t0 = std::chrono::steady_clock::now();
    size_t grainsize_n = std::min(64, std::max(1, int(N / (2 * nThreads_))) );
    size_t grainsize_m = std::min(64, std::max(1, int(M_/ (2 * nThreads_))) );
    VectorXd cvec;
    P_ = (int) docs[0].covar.size();
    std::vector<Document> mtx_t = transpose_data(docs, cvec);
    RowMajorMatrixXd X;
    std::gamma_distribution<double> dist(100.0, 0.01);
    theta_ = RowMajorMatrixXd::Zero(N, K_);
    if (reset || beta_.rows() != M_ || beta_.cols() != K_) {
        beta_  = RowMajorMatrixXd::Zero(M_,K_);
        for(int k=0; k<K_; ++k) { // Initialize beta
            for(int j=0; j<M_; ++j)
                beta_(j,k) = dist(rng_);
        }
    }
    MLEOptions mle_opts_bcov = mle_opts;
    MLEOptions mle_opts_fit = mle_opts;
    mle_opts_fit.mle_only_mode();
    mle_opts_bcov.mle_only_mode();
    std::vector<double> offset;
    double objective_old = std::numeric_limits<double>::max()-1;
    if (P_ > 0) {
        Bcov_.resize(M_, P_);
        X.resize(N, P_); // covariate matrix
        for (int i = 0; i < N; ++i) {
            X.row(i) = docs[i].covar.transpose();
        }
        mle_opts_bcov.optim.set_bounds(nmf_opts.covar_coef_min, nmf_opts.covar_coef_max, P_);
        // Do one round of regression for covariates only, with a mean offset
        notice("Fit initial regressions for covariates");
        objective_old = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, M_, grainsize_m), 0.0,
            [&](const tbb::blocked_range<int>& r, double local_sum) {
            MLEStats stats{};
            for (int j = r.begin(); j != r.end(); ++j) {
                const auto& colj = mtx_t[j];
                if (colj.ids.empty()) {
                    Bcov_.row(j).setZero();
                    continue;
                }
                double mu_j = 0.0;
                for (size_t i = 0; i < colj.ids.size(); ++i) {
                    mu_j += std::log1p(colj.cnts[i] / cvec[colj.ids[i]]);
                }
                mu_j /= N;
                VectorXd o = VectorXd::Constant(N, mu_j);
                VectorXd bj; // empty
                if (exact_) {
                    local_sum += pois_log1p_mle_exact(X, mtx_t[j], cvec, &o, mle_opts_bcov, bj, stats, debug_);
                } else {
                    local_sum += pois_log1p_mle(X, mtx_t[j], cvec, &o, mle_opts_bcov, bj, stats, debug_);
                }
                Bcov_.row(j) = bj.transpose();
            }
            return local_sum;
        }, std::plus<double>());
        objective_old /= N;
    }

    notice("%s: Starting fit %d x %d matrix with K=%d, (P=%d)", __func__, N, M_, K_, P_);
    int epoch = 0;

    if (nmf_opts.n_mb_epoch > 0) {
        tr_delta_beta_.assign(M_, 1.0);
        tr_delta_bcov_.assign(M_, 1.0);
        size_t batch_size = (nmf_opts.batch_size < 1) ? 1024 : nmf_opts.batch_size;
        if (nmf_opts.t0 < 0) {
            nmf_opts.t0 = std::min(10., 2. * N / (double) batch_size);
            notice("Set decay parameter t0 = %.1f", nmf_opts.t0);
        }
        std::vector<int> order(N);
        std::iota(order.begin(), order.end(), 0);
        size_t n_mb = 0;
        for (; epoch < nmf_opts.n_mb_epoch; ++epoch) {
            if (nmf_opts.shuffle) std::shuffle(order.begin(), order.end(), rng_);
            for (size_t start = 0; start < N; start += batch_size) {
                size_t end = std::min(N, start + batch_size);
                std::vector<int> batch(order.begin() + start, order.begin() + end);
                double rho_t = nmf_opts.use_decay ? std::pow(nmf_opts.t0 + n_mb, -nmf_opts.kappa) : -1.;
                partial_fit(docs, mtx_t,
                            mle_opts_fit, mle_opts_bcov,
                            batch, rho_t, n_mb, epoch);
                n_mb += 1;
            }
            if (epoch % nmf_opts.rescale_period == 0) {
                rescale_matrices();
            }
        }
    }

    int convergence_counter = 0;
    std::vector<int32_t> niters_reg_theta(N, 0);
    std::vector<int32_t> niters_reg_beta(M_, 0);
    std::vector<int32_t> niters_reg_bcov(M_, 0);
    for (; epoch < nmf_opts.max_iter; ++epoch) {
        // --- Update theta (document-topic matrix) ---
        double objective_current, rel_change;
        int32_t niters_theta, niters_beta, niters_bcov;
        objective_current = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, N, grainsize_n), 0.0,
            [&](const tbb::blocked_range<int>& r, double local_sum) {
            MLEStats stats;
            VectorXd cM(M_), oM;
            if (P_ > 0) oM.resize(M_);
            for (int i = r.begin(); i != r.end(); ++i) {
                cM.setConstant(docs[i].c);
                VectorXd* oM_ptr = nullptr;
                if (P_ > 0) { // Calculate offsets if covariates are present
                    oM.noalias() = Bcov_ * docs[i].covar;
                    oM_ptr = &oM;
                }
                VectorXd b;
                if (epoch > 0) {
                    b = theta_.row(i).transpose();
                }
                if (exact_) {
                    local_sum += pois_log1p_mle_exact(beta_, docs[i].doc, cM, oM_ptr, mle_opts_fit, b, stats, debug_);
                } else {
                    local_sum += pois_log1p_mle(beta_, docs[i].doc, cM, oM_ptr, mle_opts_fit, b, stats, debug_);
                }
                theta_.row(i) = b.transpose();
                niters_reg_theta[i] += stats.optim.niters;
            }
            return local_sum;
        }, std::plus<double>());
        objective_current /= N;
        niters_theta = std::accumulate(niters_reg_theta.begin(), niters_reg_theta.end(), 0) / N;

        rel_change = std::abs(objective_current - objective_old) / (std::abs(objective_old) + 1e-9);
        notice("Iteration %d, update theta: Objective = %.3e, Rel. Change = %.6e, avg niters = %d", epoch + 1, objective_current, rel_change, niters_theta);
        objective_old = objective_current;

        // --- Update beta and bcov holding theta fixed ---
        objective_current = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, M_, grainsize_m), 0.0,
            [&](const tbb::blocked_range<int>& r, double local_sum) {
                MLEStats stats_b{}, stats_beta{};
                VectorXd oN(N); // offset (Xb for beta, \theta \beta for b)
                for (int j = r.begin(); j != r.end(); ++j) {
                    // (a) update b_j with A = X, offset = theta * beta_j
                    if (P_ > 0) {
                        oN.noalias() = theta_ * beta_.row(j).transpose(); // N
                        VectorXd bj = Bcov_.row(j).transpose();
                        double fval;
                        if (exact_) {
                            fval = pois_log1p_mle_exact(
                            X, mtx_t[j], cvec, &oN, mle_opts_bcov, bj, stats_b, debug_);
                        } else {
                            fval = pois_log1p_mle(
                            X, mtx_t[j], cvec, &oN, mle_opts_bcov, bj, stats_b, debug_);
                        }
                        Bcov_.row(j) = bj.transpose();
                        niters_reg_bcov[j] += stats_b.optim.niters;
                    }
                    // (b) update beta_j with A = theta, offset = X * b_j
                    VectorXd beta_j = beta_.row(j).transpose();
                    const VectorXd* off_ptr = nullptr;
                    if (P_ > 0) {
                        oN.noalias() = X * Bcov_.row(j).transpose(); // N
                        off_ptr = &oN;
                    }
                    if (exact_) {
                        local_sum += pois_log1p_mle_exact(
                            theta_, mtx_t[j], cvec, off_ptr, mle_opts_fit, beta_j, stats_beta, debug_);
                    } else {
                        local_sum += pois_log1p_mle(
                            theta_, mtx_t[j], cvec, off_ptr, mle_opts_fit, beta_j, stats_beta, debug_);
                    }
                    beta_.row(j) = beta_j.transpose();
                    niters_reg_beta[j] += stats_beta.optim.niters;
                }
                return local_sum;
        }, std::plus<double>());

        objective_current /= N;
        niters_beta = std::accumulate(niters_reg_beta.begin(), niters_reg_beta.end(), 0) / M_;

        // --- Check for convergence ---
        rel_change = std::abs(objective_current - objective_old) / (std::abs(objective_old) + 1e-9);
        if (P_ > 0) {
            niters_bcov = std::accumulate(niters_reg_bcov.begin(), niters_reg_bcov.end(), 0) / M_;
            notice("Iteration %d, update beta:  Objective = %.3e, Rel. Change = %.6e, avg niters(beta) = %d, avg niters(b) = %d", epoch + 1, objective_current, rel_change, niters_beta, niters_bcov);
            std::fill(niters_reg_bcov.begin(), niters_reg_bcov.end(), 0);
        } else {
            notice("Iteration %d, update beta:  Objective = %.3e, Rel. Change = %.6e, avg niters = %d", epoch + 1, objective_current, rel_change, niters_beta);
        }

        if (rel_change < nmf_opts.tol) {
            convergence_counter++;
        } else {
            convergence_counter = 0;
        }
        if (convergence_counter >= 3) {
            notice("Converged after %d iterations.", epoch + 1);
            break;
        }
        objective_old = objective_current;
        std::fill(niters_reg_theta.begin(), niters_reg_theta.end(), 0);
        std::fill(niters_reg_beta.begin(), niters_reg_beta.end(), 0);

        // --- Rescale factors for numerical stability ---
        if (epoch % nmf_opts.rescale_period == 0) {
            rescale_matrices();
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
    int32_t minutes = static_cast<int32_t>(sec / 60.0);
    notice("%s: Model fitting took %.3f seconds (%dmin %.3fs)", __func__, sec, minutes, sec - minutes * 60);

    rescale_beta_to_const_sum(M_);

    if (mle_opts.se_flag == 0 && !mle_opts.compute_residual) {
        return;
    }

    t0 = std::chrono::steady_clock::now();
    cov_fisher_.clear(); cov_robust_.clear();
    if (mle_opts.se_flag & 0x1) {
        se_fisher_.resize(M_, K_);
        if (mle_opts.store_cov) cov_fisher_.resize(M_);
    }
    if (mle_opts.se_flag & 0x2) {
        se_robust_.resize(M_, K_);
        if (mle_opts.store_cov) cov_robust_.resize(M_);
    }
    if (mle_opts.compute_residual) {
        feature_residuals_.resize(M_);
    }
    notice("%s: Start computing Cov/SE/r2 for %d features", __func__, M_);
    tbb::parallel_for(0, M_, [&](int j) {
        const Eigen::Map<const Eigen::VectorXd> beta_j(beta_.row(j).data(), beta_.cols());
        VectorXd oN; VectorXd* off_ptr = nullptr;
        if (P_ > 0) {
            oN.noalias() = X * Bcov_.row(j).transpose(); // N
            off_ptr = &oN;
        }
        if (mle_opts.compute_residual) {
            feature_residuals_[j] = pois_log1p_residual(theta_, mtx_t[j], cvec, off_ptr, beta_j).sum();
        }
        if (mle_opts.se_flag) {
            MLEStats st;
            pois_log1p_compute_se(theta_, mtx_t[j], cvec, off_ptr, mle_opts, beta_j, st);
            if (mle_opts.se_flag & 0x1)
                se_fisher_.row(j) = st.se_fisher.transpose();
            if (mle_opts.se_flag & 0x2)
                se_robust_.row(j) = st.se_robust.transpose();
            if (mle_opts.store_cov) {
                if (mle_opts.se_flag & 0x1)
                    cov_fisher_[j] = std::move(st.cov_fisher);
                if (mle_opts.se_flag & 0x2)
                    cov_robust_[j] = std::move(st.cov_robust);
            }
        }
    });
    t1 = std::chrono::steady_clock::now();
    sec = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
    minutes = static_cast<int32_t>(sec / 60.0);
    notice("%s: Computing Cov/SE/r2 took %.3f seconds (%dmin %.3fs)", __func__, sec, minutes, sec - minutes * 60);

    notice("%s: Done", __func__);
}

RowMajorMatrixXd PoissonLog1pNMF::transform(std::vector<SparseObs>& docs, const MLEOptions mle_opts, std::vector<MLEStats>& res, ArrayXd* fres_ptr) {
    int N = docs.size();
    if (N == 0) return RowMajorMatrixXd(0, K_);
    if (beta_.rows() != M_ || beta_.cols() != K_) {
        error("%s: Model not fitted yet, call fit() first.", __func__);
        return RowMajorMatrixXd(N, K_);
    }
    res.clear();
    res.reserve(N);
    size_t grainsize = std::min(64, std::max(1, int(N / (2 * nThreads_))) );
    RowMajorMatrixXd new_theta(N, K_);
    ArrayXd feature_residuals;
    if (fres_ptr) feature_residuals = ArrayXd::Zero(M_);
    std::mutex mtx_residual_;
    tbb::parallel_for(tbb::blocked_range<int>(0, N, grainsize), [&](const tbb::blocked_range<int>& r) {
        VectorXd cM(M_);
        VectorXd oM; if (Bcov_.size() > 0) oM.resize(M_);
        VectorXd* oM_ptr = (Bcov_.size() > 0) ? &oM : nullptr;
        ArrayXd local_residuals;
        ArrayXd* local_fres_ptr = nullptr;
        if (fres_ptr) {
            local_residuals = ArrayXd::Zero(M_);
            local_fres_ptr = &local_residuals;
        }
        for (int i = r.begin(); i != r.end(); ++i) {
            cM.setConstant(docs[i].c);
            if (Bcov_.size() > 0) {
                oM.noalias() = Bcov_ * docs[i].covar;
            }
            VectorXd b;
            double obj;
            MLEStats stats;
            if (exact_) {
                obj = pois_log1p_mle_exact(beta_, docs[i].doc, cM, oM_ptr, mle_opts, b, stats, debug_, local_fres_ptr);
            } else {
                obj = pois_log1p_mle(beta_, docs[i].doc, cM, oM_ptr, mle_opts, b, stats, debug_, local_fres_ptr);
            }
            res.push_back(std::move(stats));
            new_theta.row(i) = b.transpose();
        }
        if (fres_ptr) {
            std::lock_guard<std::mutex> lock(mtx_residual_);
            feature_residuals += local_residuals;
        }
    });
    if (fres_ptr) {
        *fres_ptr = std::move(feature_residuals);
    }
    notice("%s: Done", __func__);
    return new_theta;
}

void PoissonLog1pNMF::partial_fit(
        const std::vector<SparseObs>& docs, const std::vector<Document>& mtx_t,
        const MLEOptions mle_opts, const MLEOptions mle_opts_bcov,
        const std::vector<int>& batch_indices, double rho_t,
        int32_t mb_count, int32_t total_epoch) {
    const int B = static_cast<int>(batch_indices.size());
    if (B == 0) return;

    // Build inverse map from global doc id -> local batch row
    std::vector<int> inv_map(docs.size(), -1);
    for (int r = 0; r < B; ++r) inv_map[ batch_indices[r] ] = r;

    // Build local matrices
    VectorXd cB(B);
    RowMajorMatrixXd XB;
    for (int r = 0; r < B; ++r) {
        int i = batch_indices[r];
        cB[r] = docs[i].c;
    }
    if (P_ > 0) {
        XB.resize(B, P_);
        for (int r = 0; r < B; ++r) {
            XB.row(r) = docs[ batch_indices[r] ].covar.transpose();
        }
    }
    // Update theta
    int grainsize = std::min(64, std::max(1, int(B / (2 * nThreads_))) );
    tbb::combinable<int> niters_local(0);
    tbb::combinable<double> obj_local(0.0);
    tbb::parallel_for(tbb::blocked_range<int>(0, B, grainsize),
                    [&](const tbb::blocked_range<int>& r) {
        VectorXd cM(M_), oM;
        if (P_ > 0) oM.resize(M_);
        for (int rr = r.begin(); rr != r.end(); ++rr) {
            int i = batch_indices[rr];
            // cM is M-length vector equal to c_i
            cM.setConstant(docs[i].c);
            VectorXd* oM_ptr = nullptr;
            if (P_ > 0) {
                oM.noalias() = Bcov_ * docs[i].covar;
                oM_ptr = &oM;
            }
            VectorXd b;
            if (total_epoch > 0) b = theta_.row(i).transpose();
            MLEStats stats;
            if (exact_) {
                obj_local.local() += pois_log1p_mle_exact(beta_, docs[i].doc, cM, oM_ptr, mle_opts, b, stats, debug_);
            } else {
                obj_local.local() += pois_log1p_mle(beta_, docs[i].doc, cM, oM_ptr, mle_opts, b, stats, debug_);
            }
            theta_.row(i) = b.transpose();
            niters_local.local() += stats.optim.niters;
        }
    });
    int niters_theta = niters_local.combine(std::plus<int>()) / B;
    double obj1 = obj_local.combine(std::plus<double>()) / B;

    OptimOptions beta_opts_mb = mle_opts.optim;
    OptimOptions bcov_opts_mb = mle_opts_bcov.optim;
    beta_opts_mb.max_iters = 1;
    bcov_opts_mb.max_iters = 1;

    auto subset_doc_to_batch = [&](const Document& full_col) {
        Document out;
        out.ids.reserve(full_col.ids.size());
        out.cnts.reserve(full_col.cnts.size());
        for (size_t t = 0; t < full_col.ids.size(); ++t) {
            int gid = full_col.ids[t];
            int lid = (gid >= 0 && gid < (int)inv_map.size()) ? inv_map[gid] : -1;
            if (lid >= 0) {
                out.ids.push_back(lid);
                out.cnts.push_back(full_col.cnts[t]);
            }
        }
        return out;
    };
    RowMajorMatrixXd ThetaB(B, K_);
    for (int r = 0; r < B; ++r) {
        ThetaB.row(r) = theta_.row(batch_indices[r]);
    }

    // Update beta and bcov
    grainsize = std::min(64, std::max(1, int(M_ / (2 * nThreads_))) );
    tbb::combinable<double> obj_local2(0.0);
    tbb::parallel_for(tbb::blocked_range<int>(0, M_, grainsize),
                    [&](const tbb::blocked_range<int>& r) {
        OptimStats stats_b{}, stats_beta{};
        VectorXd oN(B);
        for (int j = r.begin(); j != r.end(); ++j) {
            Document yB = subset_doc_to_batch(mtx_t[j]);
            // (a) Update b_cov[j] first (A = X_B, offset = Θ_B * beta_j)
            if (P_ > 0) {
                oN.noalias() = ThetaB * beta_.row(j).transpose();
                VectorXd bj = Bcov_.row(j).transpose();
                double fval;
                bcov_opts_mb.tron.delta_init = tr_delta_bcov_[j];
                if (exact_) {
                    PoisRegExactProblem P_b(XB, yB, cB, &oN, mle_opts_bcov);
                    fval = tron_solve(P_b, bj, bcov_opts_mb, stats_b, debug_, &tr_delta_bcov_[j], rho_t);
                } else {
                    PoisRegSparseProblem P_b(XB, yB, cB, &oN, mle_opts_bcov);
                    fval = tron_solve(P_b, bj, bcov_opts_mb, stats_b, debug_, &tr_delta_bcov_[j], rho_t);
                }
                (void)fval;
                Bcov_.row(j) = bj.transpose();
            }

            // (b) Update beta_j (A = Θ_B, offset = X_B * b_cov[j])
            VectorXd betaj = beta_.row(j).transpose();
            VectorXd* off_ptr = nullptr;
            beta_opts_mb.tron.delta_init = tr_delta_beta_[j];
            if (P_ > 0) {
                oN.noalias() = XB * Bcov_.row(j).transpose();
                off_ptr = &oN;
            }
            if (exact_) {
                PoisRegExactProblem P_beta(ThetaB, yB, cB, off_ptr, mle_opts);
                obj_local2.local() += tron_solve(P_beta, betaj, beta_opts_mb, stats_beta, debug_, &tr_delta_beta_[j], rho_t);
            } else {
                PoisRegSparseProblem P_beta(ThetaB, yB, cB, off_ptr, mle_opts);
                obj_local2.local() += tron_solve(P_beta, betaj, beta_opts_mb, stats_beta, debug_, &tr_delta_beta_[j], rho_t);
            }
            beta_.row(j) = betaj.transpose();
        }
    });
    double obj2 = obj_local2.combine(std::plus<double>()) / B;
    notice("Mini-batch %d, epoch %d: avg objective after theta update = %.3e, after beta update = %.3e, avg niters for theta = %d",
        mb_count, total_epoch, obj1, obj2, niters_theta);
}


std::vector<TestResult>
PoissonLog1pNMF::test_beta_vs_null(int flag) {
    if (!(flag == 1 || flag == 2)) {
        warning("%s: flag must be 1 (Fisher) or 2 (robust).", __func__);
        return {};
    }
    auto t0 = std::chrono::steady_clock::now();
    const bool use_fisher = (flag == 1);
    const auto& C = use_fisher ? cov_fisher_ : cov_robust_;
    if (C.empty() || C.size() != (size_t)M_ || C[0].size() == 0) {
        warning("%s: requested covariance (%s) not available; store_cov in MLEOptions provided to fit() must have store_cov=true.", __func__, use_fisher ? "Fisher" : "robust");
        return {};
    }
    const double eps = 1e-12;
    bool sums_known = feature_sums_.size() == (size_t)M_;

    tbb::combinable<std::vector<TestResult>> tls;
    const int grainsize = std::max(1, std::min(64, M_ / std::max(1, 2 * nThreads_)));

    tbb::parallel_for(tbb::blocked_range<int>(0, M_, grainsize),
        [&](const tbb::blocked_range<int>& r) {
            auto& out = tls.local();
            VectorXd ones = VectorXd::Ones(K_);
            for (int m = r.begin(); m != r.end(); ++m) {
                if (sums_known && feature_sums_[m] < min_ct_) continue;
                const MatrixXd& V = C[m];
                const VectorXd  b = beta_.row(m).transpose();
                // --- GLS weights w = V^{-1}1 / (1^T V^{-1}1)
                Eigen::LDLT<MatrixXd> ldlt(V);
                if (ldlt.info() != Eigen::Success) {
                    MatrixXd Vr = V;
                    Vr.diagonal().array() += eps * (V.trace() / K_ + 1.0);
                    ldlt.compute(Vr);
                }
                if (ldlt.info() != Eigen::Success) continue;
                VectorXd Vinv1 = ldlt.solve(ones);
                double denom = ones.dot(Vinv1);
                if (denom < eps) {
                    Vinv1 = ones;
                    denom = static_cast<double>(K_);
                }
                VectorXd w = Vinv1 / denom;
                double beta0 = w.dot(b);
                // Precompute V*w and w^T V w
                VectorXd Vw = V * w;
                double wVw = w.dot(Vw);
                for (int k = 0; k < K_; ++k) {
                    const double est = beta_(m, k) - beta0;
                    if (est < 0) {continue;}
                    double fc = (std::exp(beta_(m, k))-1)/(std::exp(beta0)-1);
                    if (fc < min_fc_) {continue;}
                    // contrast c = e_k - w
                    // var = c_k^T V c_k = V_kk - 2(Vw)_k + w^T V w
                    double var = V(k,k) - 2.0 * Vw(k) + wVw;
                    if (var <= 0) continue;
                    double se = std::sqrt(var);
                    double p  = normal_log10sf(est / se);
                    if (p > max_log10p_) continue;
                    out.push_back(TestResult{m, k, -1, est, p, fc});
                }
            }
        });

    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
    int32_t minutes = static_cast<int32_t>(sec / 60.0);
    std::string method = use_fisher ? "Fisher" : "robust";
    notice("%s: Computing 1 vs avg DE stats (%s) took %.3f seconds (%dmin %.3fs)", __func__, method.c_str(), sec, minutes, sec - minutes * 60);

    std::vector<TestResult> res;
    tls.combine_each([&](const std::vector<TestResult>& v){ res.insert(res.end(), v.begin(), v.end()); });
    return res;
}


std::vector<Document> PoissonLog1pNMF::transpose_data(const std::vector<SparseObs>& docs, VectorXd& cvec) {
    feature_sums_.assign(M_, 0.0);
    cvec.resize(docs.size());
    std::vector<Document> mtx_t(M_, Document{});
    size_t N = docs.size();
    for (int i = 0; i < N; ++i) {
        const auto& doc = docs[i].doc;
        cvec(i) = docs[i].c;
        for (size_t k = 0; k < doc.ids.size(); ++k) {
            int j = doc.ids[k];
            if (j >= M_) {continue;}
            feature_sums_[j] += doc.cnts[k];
            mtx_t[j].ids.push_back(i);
            mtx_t[j].cnts.push_back(doc.cnts[k]);
        }
    }
    return mtx_t;
}

double PoissonLog1pNMF::calculate_global_objective(const std::vector<SparseObs>& docs, RowMajorMatrixXd* Xptr) {
    size_t N = docs.size();
    if (N == 0) return 0.0;
    size_t grainsize = std::min(64, std::max(1, int(N / (2 * nThreads_))) );
    auto objective_part = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, N, grainsize), 0.0,
        [&](const tbb::blocked_range<int>& r, double local_sum) {
            for (int i = r.begin(); i != r.end(); ++i) {
                const auto& doc = docs[i].doc;
                for (size_t k = 0; k < doc.ids.size(); ++k) {
                    int j = doc.ids[k];
                    double eta = theta_.row(i).dot(beta_.row(j));
                    if (P_ > 0) {
                        eta += Xptr->row(i).dot(Bcov_.row(j));
                    }
                    double lambda = docs[i].c * (std::exp(eta) - 1.0);
                    lambda = std::max(lambda, 1e-12);
                    local_sum += -(doc.cnts[k] * std::log(lambda) - lambda);
                }
            }
            return local_sum;
        },
        std::plus<double>()
    );
    return objective_part;
}

void PoissonLog1pNMF::rescale_matrices() {
    VectorXd theta_col_means = theta_.colwise().mean();
    VectorXd beta_col_means = beta_.colwise().mean();
    for (int k = 0; k < K_; ++k) {
        double c_mean = theta_col_means(k);
        double r_mean = beta_col_means(k);
        if (c_mean < 1e-9 || r_mean < 1e-9) {
            warning("%d-th column of theta or beta has near-zero mean, skipping rescale (%.4e, %.4e).", k, c_mean, r_mean);
            continue;
        }
        double scale_factor = std::sqrt(r_mean / c_mean);
        // Rescale column k of theta and beta
        theta_.col(k) *= scale_factor;
        beta_.col(k) /= scale_factor;
    }
}

void PoissonLog1pNMF::rescale_beta_to_const_sum(double c) {
    VectorXd scale = beta_.colwise().sum() / c;
    for (int k = 0; k < K_; ++k) {
        theta_.col(k) *= scale(k);
        beta_.col(k) /= scale(k);
    }
}

void PoissonLog1pNMF::rescale_theta_to_const_sum(double c) {
    VectorXd scale = theta_.colwise().sum() / c;
    for (int k = 0; k < K_; ++k) {
        beta_.col(k) *= scale(k);
        theta_.col(k) /= scale(k);
    }
}

RowMajorMatrixXd PoissonLog1pNMF::convert_to_factor_loading() {
    Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor> wk = beta_.colwise().sum().array(); // 1 x K
    RowMajorMatrixXd wik = theta_.array().rowwise() * wk;
    wik = wik.array().colwise() / wik.rowwise().sum().array();
    return wik;
}

void PoissonLog1pNMF::set_de_parameters(double min_ct, double min_fc, double max_p) {
    min_ct_ = min_ct; min_fc_ = min_fc; max_p_ = max_p;
    if (max_p <= 0) {
        warning("%s: max_p must be positive, replacing %.4f with 0.05", __func__, max_p);
        max_p_ = 0.05;
    }
    if (min_fc < 1) {
        warning("%s: min_fc must be >= 1, replacing %.4f with 1.5", __func__, min_fc);
        min_fc_ = 1.5;
    }
    min_logfc_ = std::abs(std::log(min_fc_));
    max_log10p_ = std::log10(max_p_);
}


// TODO: not evaluated yet
std::vector<TestResult>
PoissonLog1pNMF::test_beta_pairwise(int flag, std::pair<int,int> kpair) {
    const int k1 = kpair.first, k2 = kpair.second;
    if (k1 < 0 || k1 >= K_ || k2 < 0 || k2 >= K_ || k1 == k2) {
        warning("%s: invalid factor indices (%d,%d).", __func__, k1, k2);
        return {};
    }
    const auto& C = (flag == 1) ? cov_fisher_ : cov_robust_;
    ArrayXd fc = (beta_.col(k1).array().exp() - 1.0).cwiseMax(1e-12) /
                 (beta_.col(k2).array().exp() - 1.0).cwiseMax(1e-12);
    ArrayXd abslogfc = fc.log().abs();
    ArrayXd est = beta_.col(k1).array() - beta_.col(k2).array();
    bool sums_known = feature_sums_.size() == (size_t)M_;

    tbb::combinable<std::vector<TestResult>> tls;
    const int grainsize = std::max(1, std::min(64, M_ / std::max(1, 2 * nThreads_)));
    tbb::parallel_for(tbb::blocked_range<int>(0, M_, grainsize),
        [&](const tbb::blocked_range<int>& r) {
            auto& out = tls.local();
            for (int m = r.begin(); m != r.end(); ++m) {
                if (sums_known &&
                    (feature_sums_[m] < min_ct_ || abslogfc(m) < min_logfc_)) continue;
                const MatrixXd& V = C[m];
                double var = V(k1,k1) + V(k2,k2) - 2.0 * V(k1,k2);
                if (var <= 0) continue;
                double se = std::sqrt(var);
                double p  = log10_twosided_p_from_z(est(m) / se);
                if (p > max_log10p_) continue;
                out.push_back(TestResult{m, k1, k2, est(m), p, fc(m)});
            }
        });

    std::vector<TestResult> res;
    tls.combine_each([&](const std::vector<TestResult>& v){ res.insert(res.end(), v.begin(), v.end()); });
    return res;
}

std::vector<TestResult>
PoissonLog1pNMF::test_beta_pairwise(int flag) {
    if (!(flag == 1 || flag == 2)) {
        warning("%s: flag must be 1 (Fisher) or 2 (robust).", __func__);
        return {};
    }
    const bool use_fisher = (flag == 1);
    const auto& C = use_fisher ? cov_fisher_ : cov_robust_;
    if (C.empty() || C.size() != (size_t)M_ || C[0].size() == 0) {
        warning("%s: requested covariance (%s) not available; rerun fit() with store_cov=true.", __func__, use_fisher ? "Fisher" : "robust");
        return {};
    }
    std::vector<TestResult> all;
    all.reserve(std::max(0, M_ * K_ * (K_ - 1) / 4)); // rough prealloc
    for (int k1 = 0; k1 < K_; ++k1) {
        for (int k2 = k1 + 1; k2 < K_; ++k2) {
            auto part = test_beta_pairwise(flag, {k1, k2});
            all.insert(all.end(),
                       std::make_move_iterator(part.begin()),
                       std::make_move_iterator(part.end()));
        }
    }
    return all;
}
