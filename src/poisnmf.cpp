#include "poisnmf.hpp"
#include <numeric>

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
    const MLEOptions& mle_opts, NmfFitOptions& nmf_opts, bool reset, std::vector<int32_t>* labels) {

    N_ = docs.size();
    if (N_ == 0) {
        warning("%s: Empty input", __func__);
        return;
    }
    if (labels != nullptr && labels->size() != N_) {
        error("%s: Number of labels (%lu) does not match number of documents (%lu)", __func__, labels->size(), N_);
    }
    P_ = (int) docs[0].covar.size();

    mle_opts_bcov_ = mle_opts;
    mle_opts_fit_  = mle_opts;
    mle_opts_fit_.mle_only_mode();
    mle_opts_bcov_.mle_only_mode();
    mle_opts_bcov_.optim.set_bounds(nmf_opts.covar_coef_min, nmf_opts.covar_coef_max, P_);

    // Initialize data objects/storage
    double obj_old = init_data(docs, reset, labels);

    auto t0 = std::chrono::steady_clock::now();
    notice("%s: Start fitting %d x %d matrix with K=%d, (P=%d)", __func__, N_, M_, K_, P_);
    epoch_ = 0;

    // Minibatch training (fast initialization)
    if (nmf_opts.n_mb_epoch > 0) {
        tr_delta_beta_.assign(M_, 1.0);
        tr_delta_bcov_.assign(M_, 1.0);
        size_t batch_size = (nmf_opts.batch_size < 1) ? 1024 : nmf_opts.batch_size;
        if (nmf_opts.t0 < 0) {
            nmf_opts.t0 = std::min(10., 2. * N_ / (double) batch_size);
            notice("Set decay parameter t0 = %.1f", nmf_opts.t0);
        }
        std::vector<uint32_t> order(N_);
        std::iota(order.begin(), order.end(), (uint32_t)0);
        minibatch_ = 0;
        for (; epoch_ < nmf_opts.n_mb_epoch; ++epoch_) {
            if (nmf_opts.shuffle) {
                std::shuffle(order.begin(), order.end(), rng_);
            }
            for (size_t start = 0; start < N_; start += batch_size) {
                size_t end = std::min(N_, start + batch_size);
                std::vector<uint32_t> batch(order.begin() + start, order.begin() + end);
                double rho_t = nmf_opts.use_decay ? std::pow(nmf_opts.t0 + minibatch_, -nmf_opts.kappa) : -1.;
                if (partial_fit(docs, batch, rho_t) < 0) {
                    break;
                }
                minibatch_ += 1;
            }
            theta_valid_.assign(N_, false);
            if (epoch_ % nmf_opts.rescale_period == 0) {
                rescale_beta_to_const_sum(M_);
            }
        }
    }

    // Batch training
    int convergence_counter = 0;
    for (; epoch_ < nmf_opts.max_iter; ++epoch_) {
        double obj_nmf, rel_change;
        double obj_phi = 0;
        int32_t niters_theta, niters_beta, niters_bcov;

        // --- Update theta ---
        obj_nmf = update_local(docs, niters_theta, &obj_phi);

        // --- Update beta and bcov ---
        double obj2 = update_bcov(niters_bcov);
        obj_nmf = update_beta(niters_beta);
        double obj_cur = obj_nmf + obj_phi;

        // --- Check for convergence ---
        rel_change = (obj_cur - obj_old) / (std::abs(obj_old) + 1e-9);
        if (P_ > 0) {
            notice("[%d] Updated beta: avg obj = %.3e, rel. change = %.3e, avg niters(beta) = %d, avg niters(b) = %d", epoch_, obj_cur, rel_change, niters_beta, niters_bcov);
        } else {
            notice("[%d] Updated beta: avg obj = %.3e, rel. change = %.3e, avg niters = %d", epoch_, obj_cur, rel_change, niters_beta);
        }
        if (std::abs(rel_change) < nmf_opts.tol) {
            convergence_counter++;
        } else {
            convergence_counter = 0;
        }
        if (convergence_counter >= 2) {
            notice("Converged after %d iterations.", epoch_ + 1);
            break;
        }
        obj_old = obj_cur;

        // --- Rescale factors for numerical stability ---
        if (epoch_ % nmf_opts.rescale_period == 0) {
            rescale_matrices();
        }
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
    int32_t minutes = static_cast<int32_t>(sec / 60.0);
    notice("%s: Model fitting took %dmin %.3fs", __func__, minutes, sec - minutes * 60);

    rescale_beta_to_const_sum(M_);

    if (mle_opts.se_flag == 0 && !mle_opts.compute_residual) {
        return;
    }
    compute_beta_se(mle_opts);
}

RowMajorMatrixXd PoissonLog1pNMF::transform(std::vector<SparseObs>& docs, const MLEOptions mle_opts, std::vector<MLEStats>& res, ArrayXd* fres_ptr) {
    int N = docs.size();
    if (N == 0) return RowMajorMatrixXd(0, K_);
    if (beta_.rows() != M_ || beta_.cols() != K_) {
        error("%s: Model not fitted yet, call fit() first.", __func__);
        return RowMajorMatrixXd(N, K_);
    }
    res.clear();
    res.resize(N);
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
            cM.setConstant(docs[i].ct_tot / size_factor_);
            if (Bcov_.size() > 0) {
                oM.noalias() = Bcov_ * docs[i].covar;
            }
            VectorXd b;
            double obj;
            MLEStats& stats = res[i];
            if (exact_) {
                obj = pois_log1p_mle_exact(beta_, docs[i].doc, cM, oM_ptr, mle_opts, b, stats, debug_, local_fres_ptr);
            } else {
                obj = pois_log1p_mle(beta_, docs[i].doc, cM, oM_ptr, mle_opts, b, stats, debug_, local_fres_ptr);
            }
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

std::vector<TestResult> PoissonLog1pNMF::test_beta_vs_null(int flag) {
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

void PoissonLog1pNMF::compute_beta_se(const MLEOptions& mle_opts) {
    auto t0 = std::chrono::steady_clock::now();
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
            oN.noalias() = X_ * Bcov_.row(j).transpose(); // N_
            off_ptr = &oN;
        }
        if (mle_opts.compute_residual) {
            feature_residuals_[j] = pois_log1p_residual(theta_, mtx_t_[j], cvec_, off_ptr, beta_j).sum();
        }
        if (mle_opts.se_flag) {
            MLEStats st;
            pois_log1p_compute_se(theta_, mtx_t_[j], cvec_, off_ptr, mle_opts, beta_j, st);
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
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
    int32_t minutes = static_cast<int32_t>(sec / 60.0);
    notice("%s: Computing Cov/SE/r2 took %dmin %.3fs", __func__, minutes, sec - minutes * 60);
}

double PoissonLog1pNMF::update_beta(int32_t& niters_beta) {
    std::vector<int32_t> niters_reg_beta(M_, 0);
    tbb::combinable<int> nkept(0.0);
    size_t grainsize_m = std::min(64, std::max(1, int(M_/ (2 * nThreads_))) );
    double objective_current = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, M_, grainsize_m), 0.0,
        [&](const tbb::blocked_range<int>& r, double local_sum) {
            MLEStats stats_b{}, stats_beta{};
            VectorXd oN(N_); // offset (Xb for beta, \theta \beta for b)
            for (int j = r.begin(); j != r.end(); ++j) {
                // (b) update beta_j with A = theta, offset = X_ * b_j
                VectorXd beta_j = beta_.row(j).transpose();
                const VectorXd* off_ptr = nullptr;
                if (P_ > 0) {
                    oN.noalias() = X_ * Bcov_.row(j).transpose(); // N_
                    off_ptr = &oN;
                }
                double obj;
                if (exact_) {
                    obj = pois_log1p_mle_exact(
                        theta_, mtx_t_[j], cvec_, off_ptr, mle_opts_fit_, beta_j, stats_beta, debug_);
                } else {
                    obj = pois_log1p_mle(
                        theta_, mtx_t_[j], cvec_, off_ptr, mle_opts_fit_, beta_j, stats_beta, debug_);
                }
                if (std::isnan(obj) || std::isinf(obj)) {
                    warning("%s: NaN/Inf objective encountered when updating beta", __func__);
                    continue;
                }
                nkept.local() += 1;
                local_sum += obj;
                beta_.row(j) = beta_j.transpose();
                niters_reg_beta[j] += stats_beta.optim.niters;
            }
            return local_sum;
    }, std::plus<double>());
    int n = nkept.combine(std::plus<int>());
    if (n == 0) {
        niters_beta = -1;
        return -1;
    }

    niters_beta = std::accumulate(niters_reg_beta.begin(), niters_reg_beta.end(), 0) / n;
    return objective_current / N_;
}

double PoissonLog1pNMF::update_bcov(int32_t& niters_bcov) {
    if (P_ <= 0) {
        return -1;
    }
    std::vector<int32_t> niters_reg_bcov(M_, 0);
    tbb::combinable<int> nkept(0.0);
    size_t grainsize_m = std::min(64, std::max(1, int(M_/ (2 * nThreads_))) );
    double objective_current = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, M_, grainsize_m), 0.0,
        [&](const tbb::blocked_range<int>& r, double local_sum) {
            MLEStats stats_b{}, stats_beta{};
            VectorXd oN(N_); // offset (Xb for beta, \theta \beta for b)
            for (int j = r.begin(); j != r.end(); ++j) {
                // (a) update b_j with A = X_, offset = theta * beta_j
                oN.noalias() = theta_ * beta_.row(j).transpose(); // N_
                VectorXd bj = Bcov_.row(j).transpose();
                double obj;
                if (exact_) {
                    obj = pois_log1p_mle_exact(
                    X_, mtx_t_[j], cvec_, &oN, mle_opts_bcov_, bj, stats_b, debug_);
                } else {
                    obj = pois_log1p_mle(
                    X_, mtx_t_[j], cvec_, &oN, mle_opts_bcov_, bj, stats_b, debug_);
                }
                if (std::isnan(obj) || std::isinf(obj)) {
                    warning("%s: NaN/Inf objective encountered when updating bcov", __func__);
                    continue;
                }
                nkept.local() += 1;
                local_sum += obj;
                Bcov_.row(j) = bj.transpose();
                niters_reg_bcov[j] += stats_b.optim.niters;
            }
            return local_sum;
    }, std::plus<double>());
    int n = nkept.combine(std::plus<int>());
    if (n == 0) {
        niters_bcov = -1;
        return -1;
    }
    niters_bcov = std::accumulate(niters_reg_bcov.begin(), niters_reg_bcov.end(), 0) / n;
    return objective_current / N_;
}

double PoissonLog1pNMF::update_theta(const std::vector<SparseObs>& docs,
    int32_t& niters, const std::vector<uint32_t>* batch_indices) {
    size_t N = batch_indices == nullptr? docs.size() : batch_indices->size();
    size_t grainsize_n = std::min(64, std::max(1, int(N / (2 * nThreads_))) );
    std::vector<int32_t> niters_reg_theta(N, 0);
    tbb::combinable<int> nkept(0);
    auto global_idx = [&](int i) {
        return (batch_indices != nullptr) ? (*batch_indices)[i] : i;
    };
    double obj_avg = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, N, grainsize_n), 0.0,
        [&](const tbb::blocked_range<int>& r, double local_sum) {
        MLEStats stats;
        VectorXd cM(M_), oM;
        if (P_ > 0) oM.resize(M_);
        for (int rr = r.begin(); rr != r.end(); ++rr) {
            int i = global_idx(rr);
            cM.setConstant(cvec_[i]);
            VectorXd* oM_ptr = nullptr;
            if (P_ > 0) { // Calculate offsets if covariates are present
                oM.noalias() = Bcov_ * docs[i].covar;
                oM_ptr = &oM;
            }
            VectorXd b;
            if (theta_valid_[i]) {
                b = theta_.row(i).transpose();
            }
            double obj;
            if (exact_) {
                obj = pois_log1p_mle_exact(beta_, docs[i].doc, cM, oM_ptr, mle_opts_fit_, b, stats, debug_);
            } else {
                obj = pois_log1p_mle(beta_, docs[i].doc, cM, oM_ptr, mle_opts_fit_, b, stats, debug_);
            }
            if (std::isnan(obj) || std::isinf(obj)) {
                warning("%s: NaN/Inf objective encountered when updating theta", __func__);
                theta_valid_[i] = false;
                continue;
            }
            nkept.local() += 1;
            local_sum += obj;
            theta_.row(i) = b.transpose();
            niters_reg_theta[rr] += stats.optim.niters;
            theta_valid_[i] = true;
        }
        return local_sum;
    }, std::plus<double>());
    int n = nkept.combine(std::plus<int>());
    if (n == 0) {
        niters = -1;
        return 0;
    }
    niters = std::accumulate(niters_reg_theta.begin(), niters_reg_theta.end(), 0) / n;
    obj_avg /= n;
    notice("[%d] Updated theta: avg obj = %.3e, avg niters = %d", epoch_, obj_avg, niters);

    return obj_avg;
}

void PoissonLog1pNMF::transpose_data(const std::vector<SparseObs>& docs, std::vector<Document>& mtx_t, const std::vector<uint32_t>* row_idx) {
    transpose_data_common(
        mtx_t, docs, row_idx,
        [](const SparseObs& obs) -> const Document& { return obs.doc; },
        [](const SparseObs& obs, const Document& doc) {
            return (obs.ct_tot >= 0) ? obs.ct_tot
                : std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
        });
}

void PoissonLog1pNMF::transpose_data(const std::vector<Document>& docs, std::vector<Document>& mtx_t, const std::vector<uint32_t>* row_idx) {
    transpose_data_common(
        mtx_t, docs, row_idx,
        [](const Document& doc) -> const Document& { return doc; },
        [](const Document&, const Document& doc) {
            return (doc.ct_tot >= 0) ? doc.ct_tot
                : std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
        });
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
                    double lambda = cvec_[i] * (std::exp(eta) - 1.0);
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

void PoissonLog1pNMF::init_denoise(const std::vector<SparseObs>& docs) {
    mtx_.reserve(M_);
    phi_.resize(N_, pi0_);
    for (const auto& doc : docs) {
        mtx_.push_back(doc.doc);
    }
    if (beta0_.size() != (size_t)M_) {
        // set beta0_ to the relative feature abundance
        if (feature_sums_.size() == (size_t)M_) {
            beta0_.resize(M_);
            for (size_t j = 0; j < (size_t)M_; ++j) {
                beta0_(j) = feature_sums_[j];
            }
            beta0_ /= beta0_.sum();
        } else if (mtx_t_.size() == (size_t)M_) {
            beta0_ = VectorXd::Zero(M_);
            for (size_t j = 0; j < (size_t)M_; ++j) {
                beta0_(j) = std::accumulate(
                    mtx_t_[j].cnts.begin(), mtx_t_[j].cnts.end(), 0.0);
            }
            beta0_ /= beta0_.sum();
        } else {
            error("%s: feature_sums_ not available; init_data (thus transpose_data) should be called first.", __func__);
        }
    }
}

double PoissonLog1pNMF::init_data(const std::vector<SparseObs>& docs, bool reset_beta, std::vector<int32_t>* labels) {
    cvec_ = VectorXd(N_);
    for (size_t i = 0; i < N_; ++i) {
        cvec_(i) = docs[i].ct_tot / size_factor_;
    }
    transpose_data(docs, mtx_t_);
    if (fit_background_) {
        init_denoise(docs);
    }
    double obj = 0.0;
    // Initialize beta
    std::gamma_distribution<double> dist(100.0, 0.01);
    if (reset_beta || beta_.rows() != M_ || beta_.cols() != K_) {
        beta_  = RowMajorMatrixXd::Zero(M_,K_);
        for(int k=0; k<K_; ++k) {
            for(int j=0; j<M_; ++j)
                beta_(j,k) = dist(rng_);
        }
    }
    // Initialize theta
    theta_ = RowMajorMatrixXd::Zero(N_, K_);
    theta_valid_.assign(N_, false);
    if (labels != nullptr) {
        for (int32_t n = 0; n < N_; ++n) {
            int32_t k = (*labels)[n];
            if (k < 0 || k >= K_) {
                for (int32_t kk = 0; kk < K_; ++kk) {
                    theta_(n, kk) = 1.0 / K_;
                }
                continue;
            }
            theta_(n, k) = 1.0;
            theta_valid_[n] = true;
        }
        int32_t niters_beta;
        obj = update_beta(niters_beta);
        notice("Initialized theta & beta from labels, objective = %.3e", obj);
    }
    notice("Initialized data with N=%d, M=%d", N_, M_);
    if (P_ <= 0) {
        return obj;
    }
    // Initialize covariate matrix X_ and coefficient Bcov_
    X_.resize(N_, P_);
    for (size_t i = 0; i < N_; ++i) {
        X_.row(i) = docs[i].covar.transpose();
    }
    Bcov_.resize(M_, P_);
    // Do one round of regression for covariates only, with a mean offset
    notice("Fit initial regressions for covariates");
    size_t grainsize_m = std::min(64, std::max(1, int(M_/ (2 * nThreads_))) );
    obj = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, M_, grainsize_m), 0.0,
        [&](const tbb::blocked_range<int>& r, double local_sum) {
        MLEStats stats{};
        for (int j = r.begin(); j != r.end(); ++j) {
            const auto& colj = mtx_t_[j];
            if (colj.ids.empty()) {
                Bcov_.row(j).setZero();
                continue;
            }
            double mu_j = 0.0;
            for (size_t i = 0; i < colj.ids.size(); ++i) {
                mu_j += std::log1p(colj.cnts[i] / cvec_[colj.ids[i]]);
            }
            mu_j /= N_;
            VectorXd o = VectorXd::Constant(N_, mu_j);
            VectorXd bj; // empty
            if (exact_) {
                local_sum += pois_log1p_mle_exact(X_, mtx_t_[j], cvec_, &o, mle_opts_bcov_, bj, stats, debug_);
            } else {
                local_sum += pois_log1p_mle(X_, mtx_t_[j], cvec_, &o, mle_opts_bcov_, bj, stats, debug_);
            }
            Bcov_.row(j) = bj.transpose();
        }
        return local_sum;
    }, std::plus<double>());
    return obj / N_;
}

std::vector<TestResult> PoissonLog1pNMF::test_beta_pairwise(int flag) {
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

double PoissonLog1pNMF::update_local(const std::vector<SparseObs>& docs,
    int32_t& niters, double* obj0, const std::vector<uint32_t>* batch_indices, double rho_t)
{
    if (!fit_background_) {
        return update_theta(docs, niters, batch_indices);
    }

    size_t N = batch_indices == nullptr? N_ : batch_indices->size();
    size_t grainsize_n = std::min(64, std::max(1, int(N / (2 * nThreads_))) );
    std::vector<int32_t> niters_reg_theta(N, 0);
    tbb::combinable<int> nkept(0);
    auto global_idx = [&](int i) {
        return (batch_indices != nullptr) ? (*batch_indices)[i] : i;
    };
    double obj_avg = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, N, grainsize_n), 0.0,
        [&](const tbb::blocked_range<int>& r, double local_sum) {
        MLEStats stats;
        VectorXd cM(M_), oM;
        if (P_ > 0) oM.resize(M_);
        for (int rr = r.begin(); rr != r.end(); ++rr) {
            int i = global_idx(rr);
            cM.setConstant(cvec_[i]);
            VectorXd* oM_ptr = nullptr;
            if (P_ > 0) { // Calculate offsets if covariates are present
                oM.noalias() = Bcov_ * X_.row(i).transpose();
                oM_ptr = &oM;
            }
            VectorXd b;
            if (theta_valid_[i]) {
                b = theta_.row(i).transpose();
            }
            double obj;
            obj = pois_log1p_mle(beta_, mtx_[i], cM, oM_ptr, mle_opts_fit_, b, stats, debug_);
            if (std::isnan(obj) || std::isinf(obj)) {
                warning("%s: NaN/Inf objective encountered when updating theta", __func__);
                theta_valid_[i] = false;
                continue;
            }
            nkept.local() += 1;
            local_sum += obj;
            theta_.row(i) = b.transpose();
            niters_reg_theta[rr] += stats.optim.niters;
            theta_valid_[i] = true;
        }
        return local_sum;
    }, std::plus<double>());
    int n = nkept.combine(std::plus<int>());
    if (n == 0) {
        niters = -1;
        return 0;
    }
    niters = std::accumulate(niters_reg_theta.begin(), niters_reg_theta.end(), 0) / n;
    obj_avg /= n;
    tbb::combinable<double> fg_acc(0.0), bg_acc(0.0);
    tbb::combinable<VectorXd> beta0_acc([&]() {return VectorXd::Zero(M_);});
    // Update background proportions phi_ and denoised counts mtx_
    double obj_phi = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, N, grainsize_n), 0.0,
        [&](const tbb::blocked_range<int>& r, double q0) {
        auto& beta0 = beta0_acc.local();
        double& fg = fg_acc.local();
        double& bg = bg_acc.local();
        for (int rr = r.begin(); rr != r.end(); ++rr) {
            int i = global_idx(rr);
            if (!theta_valid_[i]) {
                continue;
            }
            double phi_i = 0.0;
            for (size_t idx = 0; idx < mtx_[i].ids.size(); ++idx) {
                int m = mtx_[i].ids[idx];
                double y = docs[i].doc.cnts[idx];
                double lam = std::max(std::exp(beta_.row(m).dot(theta_.row(i))) - 1.0, 1e-12) / size_factor_;
                double phi = pi_ * beta0_(m);
                double denom = phi + (1 - pi_) * lam;
                phi = denom > 0 ? phi / denom : 0.0;
                double y0 = y * phi;
                double y1 = y * (1-phi);
                fg += y1;
                bg += y0;
                mtx_[i].cnts[idx] = y1;
                phi_i += y0;
                beta0(m) += y0;
                q0 += y0 * (log_pi_ + std::log(beta0_(m)))
                    + y1 * log_1mpi_ - log_factorial(y1);
            }
            phi_i /= docs[i].ct_tot;
            phi_[i] = phi_i;
        }
        return q0;
    }, std::plus<double>());
    VectorXd beta0_local = beta0_acc.combine(
        [](const VectorXd& a, const VectorXd& b) {return a + b;});
    beta0_local /= beta0_local.sum();
    *obj0 = - obj_phi / n;
    double fg = fg_acc.combine(std::plus<double>());
    double bg = bg_acc.combine(std::plus<double>());
    double f0 = bg / (fg + bg);

    if (batch_indices == nullptr) {
        transpose_data(mtx_, mtx_t_); // TODO - could be slow
    }

    if (rho_t < 0) {
        beta0_ = std::move(beta0_local);
        pi_ = bg / (fg + bg);
    } else {
        beta0_ = (1.0 - rho_t) * beta0_ + rho_t * beta0_local;
        pi_ = (1.0 - rho_t) * pi_ + rho_t * (bg / (fg + bg));
    }
    notice("[%d] Updated theta & phi: avg obj = %.3e (%.3e), avg niters = %d, avg pi = %.3e", epoch_, obj_avg, *obj0, niters, f0);

    return obj_avg;
}

int32_t PoissonLog1pNMF::partial_fit(
        const std::vector<SparseObs>& docs,
        const std::vector<uint32_t>& batch_indices, double rho_t)
{
    const int N = static_cast<int>(docs.size());
    const int B = static_cast<int>(batch_indices.size());
    if (B == 0) return 0;
    std::vector<Document> mtx_t; // feature-major matrix for this mini-batch

    // Update theta (& background proportions)
    int niters_theta; double obj_phi;
    double obj1 = update_local(docs, niters_theta, &obj_phi, &batch_indices, rho_t);
    if (niters_theta < 0) {
        warning("%s: No documents were kept after updating theta in partial_fit", __func__);
        return -1;
    }
    if (fit_background_) {
        transpose_data(mtx_, mtx_t, &batch_indices);
    } else {
        transpose_data(docs, mtx_t, &batch_indices);
    }

    // Build local matrices
    VectorXd cB(B);
    RowMajorMatrixXd XB;
    for (int r = 0; r < B; ++r) {
        cB[r] = cvec_[ batch_indices[r] ];
    }
    if (P_ > 0) {
        XB.resize(B, P_);
        for (int r = 0; r < B; ++r) {
            XB.row(r) = X_.row( batch_indices[r] );
        }
    }
    RowMajorMatrixXd ThetaB(B, K_);
    for (int r = 0; r < B; ++r) {
        ThetaB.row(r) = theta_.row(batch_indices[r]);
    }

    // Update beta and bcov
    OptimOptions beta_opts_mb = mle_opts_fit_.optim;
    OptimOptions bcov_opts_mb = mle_opts_bcov_.optim;
    beta_opts_mb.max_iters = 1;
    bcov_opts_mb.max_iters = 1;
    size_t grainsize = std::min(64, std::max(1, int(M_ / (2 * nThreads_))) );
    tbb::combinable<double> obj_local(0.0);
    tbb::parallel_for(tbb::blocked_range<int>(0, M_, grainsize),
                    [&](const tbb::blocked_range<int>& r) {
        OptimStats stats_b{}, stats_beta{};
        VectorXd oN(B);
        for (int j = r.begin(); j != r.end(); ++j) {
            // (a) Update b_cov[j] first (A = X_B, offset = Θ_B * beta_j)
            if (P_ > 0) {
                oN.noalias() = ThetaB * beta_.row(j).transpose();
                VectorXd bj = Bcov_.row(j).transpose();
                bcov_opts_mb.tron.delta_init = tr_delta_bcov_[j];
                PoisRegSparseProblem P_b(XB, mtx_t[j], cB, &oN, mle_opts_bcov_);
                double fval = tron_solve(P_b, bj, bcov_opts_mb, stats_b, debug_, &tr_delta_bcov_[j], rho_t);
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
            PoisRegSparseProblem P_beta(ThetaB, mtx_t[j], cB, off_ptr, mle_opts_fit_);
            double obj = tron_solve(P_beta, betaj, beta_opts_mb, stats_beta, debug_, &tr_delta_beta_[j], rho_t);
            if (std::isnan(obj) || std::isinf(obj)) {
                warning("%s: NaN/Inf objective encountered when updating beta", __func__);
                continue;
            }
            obj_local.local() += obj;
            beta_.row(j) = betaj.transpose();
        }
    });
    double obj2 = obj_local.combine(std::plus<double>()) / B;
    notice("[%d] Mini-batch %d (rho=%.2e): avg obj = %.3e, avg niters for theta = %d",
        epoch_, minibatch_, rho_t, obj2+obj_phi, niters_theta);
    return 1;
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
