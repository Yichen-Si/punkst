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
    notice("Requested %d threads, actual number of threads: %d", nThreads, nThreads_);
}

void PoissonLog1pNMF::fit(const std::vector<SparseObs>& docs,
    const MLEOptions mle_opts, int max_iter, double tol,
    double covar_coef_min, double covar_coef_max) {
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
    beta_  = RowMajorMatrixXd::Zero(M_,K_);
    for(int k=0; k<K_; ++k) { // Initialize beta
        for(int j=0; j<M_; ++j)
            beta_(j,k) = dist(rng_);
    }
    MLEOptions mle_opts_bcov = mle_opts;
    MLEOptions mle_opts_fit = mle_opts;
    mle_opts_fit.se_flag = 0; // no SE during fitting
    mle_opts_bcov.se_flag = 0;
    mle_opts_fit.compute_residual = false;
    mle_opts_bcov.compute_residual = false;
    std::vector<double> offset;
    double objective_old = std::numeric_limits<double>::max()-1;
    if (P_ > 0) {
        Bcov_.resize(M_, P_);
        X.resize(N, P_); // covariate matrix
        for (int i = 0; i < N; ++i) {
            X.row(i) = docs[i].covar.transpose();
        }
        mle_opts_bcov.optim.set_bounds(covar_coef_min, covar_coef_max, P_);
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
    int convergence_counter = 0;
    std::vector<int32_t> niters_reg_theta(N, 0);
    std::vector<int32_t> niters_reg_beta(M_, 0);
    std::vector<int32_t> niters_reg_bcov(M_, 0);
    for (int iter = 0; iter < max_iter; ++iter) {
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
                if (iter > 0) {
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
        notice("Iteration %d, update theta: Objective = %.3e, Rel. Change = %.6e, avg niters = %d", iter + 1, objective_current, rel_change, niters_theta);
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
            notice("Iteration %d, update beta:  Objective = %.3e, Rel. Change = %.6e, avg niters(beta) = %d, avg niters(b) = %d", iter + 1, objective_current, rel_change, niters_beta, niters_bcov);
            std::fill(niters_reg_bcov.begin(), niters_reg_bcov.end(), 0);
        } else {
            notice("Iteration %d, update beta:  Objective = %.3e, Rel. Change = %.6e, avg niters = %d", iter + 1, objective_current, rel_change, niters_beta);
        }

        if (rel_change < tol) {
            convergence_counter++;
        } else {
            convergence_counter = 0;
        }
        if (convergence_counter >= 3) {
            notice("Converged after %d iterations.", iter + 1);
            break;
        }
        objective_old = objective_current;
        std::fill(niters_reg_theta.begin(), niters_reg_theta.end(), 0);
        std::fill(niters_reg_beta.begin(), niters_reg_beta.end(), 0);

        // --- Rescale factors for numerical stability ---
        rescale_matrices();
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

RowMajorMatrixXd PoissonLog1pNMF::transform(std::vector<SparseObs>& docs, const MLEOptions mle_opts, std::vector<MLEStats>& res) {
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
    tbb::parallel_for(tbb::blocked_range<int>(0, N, grainsize), [&](const tbb::blocked_range<int>& r) {
        VectorXd cM(M_);
        VectorXd oM; if (Bcov_.size() > 0) oM.resize(M_);
        VectorXd* oM_ptr = (Bcov_.size() > 0) ? &oM : nullptr;
        for (int i = r.begin(); i != r.end(); ++i) {
            cM.setConstant(docs[i].c);
            if (Bcov_.size() > 0) {
                oM.noalias() = Bcov_ * docs[i].covar;
            }
            VectorXd b;
            double obj;
            MLEStats stats;
            if (exact_) {
                obj = pois_log1p_mle_exact(beta_, docs[i].doc, cM, oM_ptr, mle_opts, b, stats, debug_);
            } else {
                obj = pois_log1p_mle(beta_, docs[i].doc, cM, oM_ptr, mle_opts, b, stats, debug_);
            }
            res.push_back(std::move(stats));
            new_theta.row(i) = b.transpose();
        }
    });
    return new_theta;
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

    tbb::combinable<std::vector<TestResult>> tls;
    const int grainsize = std::max(1, std::min(64, M_ / std::max(1, 2 * nThreads_)));

    tbb::parallel_for(tbb::blocked_range<int>(0, M_, grainsize),
        [&](const tbb::blocked_range<int>& r) {
            auto& out = tls.local();
            VectorXd ones = VectorXd::Ones(K_);
            for (int m = r.begin(); m != r.end(); ++m) {
                if (feature_sums_[m] < min_ct_) continue;
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

    tbb::combinable<std::vector<TestResult>> tls;
    const int grainsize = std::max(1, std::min(64, M_ / std::max(1, 2 * nThreads_)));
    tbb::parallel_for(tbb::blocked_range<int>(0, M_, grainsize),
        [&](const tbb::blocked_range<int>& r) {
            auto& out = tls.local();
            for (int m = r.begin(); m != r.end(); ++m) {
                if (feature_sums_[m] < min_ct_ || abslogfc(m) < min_logfc_) continue;
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
