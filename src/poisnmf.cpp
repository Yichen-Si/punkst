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

void PoissonLog1pNMF::fit(const std::vector<Document>& docs, const MLEOptions mle_opts, int max_iter, double tol) {
    size_t N = docs.size();
    if (N == 0) {
        warning("%s: Empty input", __func__);
        return;
    }
    std::vector<Document> mtx_t = transpose_data(docs);
    // std::uniform_real_distribution<double> dist(0.1, 1.0);
    std::gamma_distribution<double> dist(100.0, 0.01);
    theta_ = RowMajorMatrixXd::Zero(N, K_);
    beta_  = RowMajorMatrixXd::Zero(M_,K_);
    for(int k=0; k<K_; ++k) for(int j=0; j<M_; ++j) beta_(j,k) = dist(rng_);

    double objective_old = std::numeric_limits<double>::max();
    int convergence_counter = 0;

    notice("%s: Starting fit %d x %d matrix with K=%d, c=%.2f", __func__, N, M_, K_, c_);
    std::vector<int32_t> niters_reg_theta(N, 0);
    std::vector<int32_t> niters_reg_beta(M_, 0);
    for (int iter = 0; iter < max_iter; ++iter) {
        // --- Update theta (document-topic matrix) holding beta fixed ---
        double objective_current, rel_change;
        int32_t niters_theta, niters_beta;
        objective_current = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, N), 0.0,
            [&](const tbb::blocked_range<int>& r, double local_sum) {
            MLEStats stats;
            for (int i = r.begin(); i != r.end(); ++i) {
                VectorXd b;
                if (iter > 0) {
                    b = theta_.row(i).transpose();
                }
                local_sum += pois_log1p_mle(beta_, docs[i], c_, mle_opts, b, stats);
                theta_.row(i) = b.transpose();
                niters_reg_theta[i] += stats.niters;
            }
            return local_sum;
        }, std::plus<double>());
        objective_current /= M_;
        niters_theta = std::accumulate(niters_reg_theta.begin(), niters_reg_theta.end(), 0) / N;

        rel_change = std::abs(objective_current - objective_old) / (std::abs(objective_old) + 1e-9);
        notice("Iteration %d, update theta: Objective = %.3e, Rel. Change = %.6e, avg niters = %d", iter + 1, objective_current, rel_change, niters_theta);
        objective_old = objective_current;

        // --- Update beta (topic-word matrix) holding theta fixed ---
        objective_current = tbb::parallel_reduce(
            tbb::blocked_range<int>(0, M_), 0.0,
            [&](const tbb::blocked_range<int>& r, double local_sum) {
                MLEStats stats;
                for (int j = r.begin(); j != r.end(); ++j) {
                    VectorXd b = beta_.row(j);
                    local_sum += pois_log1p_mle(theta_, mtx_t[j], c_, mle_opts, b, stats);
                    beta_.row(j) = b.transpose();
                    niters_reg_beta[j] += stats.niters;
                }
            return local_sum;
        }, std::plus<double>());
        objective_current /= M_;
        niters_beta = std::accumulate(niters_reg_beta.begin(), niters_reg_beta.end(), 0) / M_;

        // --- Check for convergence ---
        rel_change = std::abs(objective_current - objective_old) / (std::abs(objective_old) + 1e-9);
        notice("Iteration %d, update beta: Objective = %.3e, Rel. Change = %.6e, avg niters = %d", iter + 1, objective_current, rel_change, niters_beta);

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

RowMajorMatrixXd PoissonLog1pNMF::transform(const std::vector<Document>& docs, const MLEOptions mle_opts) {
    int N = docs.size();
    if (N == 0) return RowMajorMatrixXd(0, K_);
    if (beta_.rows() != M_ || beta_.cols() != K_) {
        error("%s: Model not fitted yet, call fit() first.", __func__);
        return RowMajorMatrixXd(N, K_);
    }
    RowMajorMatrixXd new_theta(N, K_);
    tbb::parallel_for(tbb::blocked_range<int>(0, N), [&](const tbb::blocked_range<int>& r) {
        MLEStats stats;
        for (int i = r.begin(); i != r.end(); ++i) {
            VectorXd b;
            double obj = pois_log1p_mle(beta_, docs[i], c_, mle_opts, b, stats);
            new_theta.row(i) = b.transpose();
        }
    });
    return new_theta;
}

std::vector<Document> PoissonLog1pNMF::transpose_data(const std::vector<Document>& docs) {
    std::vector<Document> mtx_t(M_, Document{});
    size_t N = docs.size();
    for (int i = 0; i < N; ++i) {
        for (size_t k = 0; k < docs[i].ids.size(); ++k) {
            int j = docs[i].ids[k];
            if (j < M_) {
                mtx_t[j].ids.push_back(i);
                mtx_t[j].cnts.push_back(docs[i].cnts[k]);
            }
        }
    }
    return mtx_t;
}

double PoissonLog1pNMF::calculate_global_objective(const std::vector<Document>& docs) {
    size_t N = docs.size();
    if (N == 0) return 0.0;
    auto objective_part = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, N), 0.0,
        [&](const tbb::blocked_range<int>& r, double local_sum) {
            for (int i = r.begin(); i != r.end(); ++i) {
                for (size_t k = 0; k < docs[i].ids.size(); ++k) {
                    int j = docs[i].ids[k];
                    double eta = theta_.row(i).dot(beta_.row(j));
                    double lambda = c_ * (std::exp(eta) - 1.0);
                    lambda = std::max(lambda, 1e-8);
                    local_sum += -(docs[i].cnts[k] * std::log(lambda) - lambda);
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
