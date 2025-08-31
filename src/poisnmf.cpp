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
    MLEOptions mle_opts, int max_iter, double tol,
    double covar_coef_min, double covar_coef_max) {
    size_t N = docs.size();
    if (N == 0) {
        warning("%s: Empty input", __func__);
        return;
    }
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
    mle_opts.set_bounds(0.0, std::numeric_limits<double>::infinity(), K_);
    std::vector<double> offset;
    double objective_old = std::numeric_limits<double>::max()-1;
    if (P_ > 0) {
        Bcov_.resize(M_, P_);
        X.resize(N, P_); // covariate matrix
        for (int i = 0; i < N; ++i) {
            X.row(i) = docs[i].covar.transpose();
        }
        mle_opts_bcov.set_bounds(covar_coef_min, covar_coef_max, P_);
        // Do one round of regression for covariates only, with a mean offset
        notice("Fit initial regressions for covariates");
        objective_old = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, M_), 0.0,
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
                    local_sum += pois_log1p_mle_exact(X, mtx_t[j], &cvec, &o, mle_opts_bcov, bj, stats);
                } else {
                    local_sum += pois_log1p_mle(X, mtx_t[j], &cvec, &o, mle_opts_bcov, bj, stats);
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
            tbb::blocked_range<int>(0, N), 0.0,
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
                    local_sum += pois_log1p_mle_exact(beta_, docs[i].doc, &cM, oM_ptr, mle_opts, b, stats);
                } else {
                    local_sum += pois_log1p_mle(beta_, docs[i].doc, &cM, oM_ptr, mle_opts, b, stats);
                }
                theta_.row(i) = b.transpose();
                niters_reg_theta[i] += stats.niters;
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
            tbb::blocked_range<int>(0, M_), 0.0,
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
                            X, mtx_t[j], &cvec, &oN, mle_opts_bcov, bj, stats_b);
                        } else {
                            fval = pois_log1p_mle(
                            X, mtx_t[j], &cvec, &oN, mle_opts_bcov, bj, stats_b);
                        }
                        Bcov_.row(j) = bj.transpose();
                        niters_reg_bcov[j] += stats_b.niters;
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
                            theta_, mtx_t[j], &cvec, off_ptr, mle_opts, beta_j, stats_beta);
                    } else {
                        local_sum += pois_log1p_mle(
                            theta_, mtx_t[j], &cvec, off_ptr, mle_opts, beta_j, stats_beta);
                    }
                    beta_.row(j) = beta_j.transpose();
                    niters_reg_beta[j] += stats_beta.niters;
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
}

RowMajorMatrixXd PoissonLog1pNMF::transform(const std::vector<SparseObs>& docs, const MLEOptions mle_opts) {
    int N = docs.size();
    if (N == 0) return RowMajorMatrixXd(0, K_);
    if (beta_.rows() != M_ || beta_.cols() != K_) {
        error("%s: Model not fitted yet, call fit() first.", __func__);
        return RowMajorMatrixXd(N, K_);
    }
    RowMajorMatrixXd new_theta(N, K_);
    tbb::parallel_for(tbb::blocked_range<int>(0, N), [&](const tbb::blocked_range<int>& r) {
        MLEStats stats;
        Eigen::VectorXd cM(M_);
        Eigen::VectorXd oM; if (Bcov_.size() > 0) oM.resize(M_);
        VectorXd* oM_ptr = (Bcov_.size() > 0) ? &oM : nullptr;
        for (int i = r.begin(); i != r.end(); ++i) {
            cM.setConstant(docs[i].c);
            if (Bcov_.size() > 0) {
                oM.noalias() = Bcov_ * docs[i].covar;
            }
            VectorXd b;
            double obj;
            if (exact_) {
                obj = pois_log1p_mle_exact(beta_, docs[i].doc, &cM, oM_ptr, mle_opts, b, stats);
            } else {
                obj = pois_log1p_mle(beta_, docs[i].doc, &cM, oM_ptr, mle_opts, b, stats);
            }
            new_theta.row(i) = b.transpose();
        }
    });
    return new_theta;
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
            if (j < M_) {
                mtx_t[j].ids.push_back(i);
                mtx_t[j].cnts.push_back(doc.cnts[k]);
            }
        }
    }
    return mtx_t;
}

double PoissonLog1pNMF::calculate_global_objective(const std::vector<SparseObs>& docs, RowMajorMatrixXd* Xptr) {
    size_t N = docs.size();
    if (N == 0) return 0.0;
    auto objective_part = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, N), 0.0,
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

void PoissonLog1pNMF::rescale_beta_to_unit_sum() {
    VectorXd beta_col_means = beta_.colwise().sum();
    for (int k = 0; k < K_; ++k) {
        theta_.col(k) *= beta_col_means(k);
        beta_.col(k) /= beta_col_means(k);
    }
}

void PoissonLog1pNMF::rescale_theta_to_unit_sum() {
    VectorXd theta_col_means = theta_.colwise().sum();
    for (int k = 0; k < K_; ++k) {
        beta_.col(k) *= theta_col_means(k);
        theta_.col(k) /= theta_col_means(k);
    }
}

RowMajorMatrixXd PoissonLog1pNMF::convert_to_factor_loading() {
    Eigen::Array<double, 1, Eigen::Dynamic, Eigen::RowMajor> wk = beta_.colwise().sum().array(); // 1 x K
    RowMajorMatrixXd wik = theta_.array().rowwise() * wk;
    wik = wik.array().colwise() / wik.rowwise().sum().array();
    return wik;
}
