#pragma once

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include <tuple>
#include "numerical_utils.hpp"
#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "poisreg.hpp"
#include "tiles2minibatch.hpp"

// Using Eigen types for convenience
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::RowVectorXf;
using Eigen::ArrayXd;
using Eigen::ArrayXf;
using Eigen::SparseMatrix;
using RowMajorMatrixXf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class OnlineSLDA {
public:
    int verbose_, debug_;
    // Constructor
    OnlineSLDA() {}
    OnlineSLDA(int K, int M, int N,
        unsigned int seed = std::random_device{}(),
        VectorXf* alpha = nullptr, VectorXf* eta = nullptr,
        double tau0 = 9.0, double kappa = 0.7,
        int iter_inner = 50, double tol = 1e-4,
        int verbose = 0, int debug = 0) {
        init(K, M, N, seed, alpha, eta, tau0, kappa, iter_inner, tol, verbose, debug);
    }

    const RowMajorMatrixXf& get_lambda() const {return lambda_;}
    const RowMajorMatrixXf& get_Elog_beta() const {return Elog_beta_;}

    void init(int K, int M, int N,
        unsigned int seed = std::random_device{}(),
        VectorXf* alpha = nullptr, VectorXf* eta = nullptr,
        double tau0 = 9.0, double kappa = 0.7,
        int iter_inner = 50, double tol = 1e-4,
        int verbose = 0, int debug = 0);

    // Initialize the global variational parameter lambda for topics.
    template<typename Derived>
    void init_global_parameter(const Eigen::MatrixBase<Derived>& lambda) {
        lambda_ = lambda.template cast<float>();
        if (lambda_.rows() == M_ && lambda_.cols() == K_) {
            lambda_ = lambda.transpose().template cast<float>();
        } else if (lambda_.rows() != K_ || lambda_.cols() != M_) {
            throw std::invalid_argument(
                "Invalid dimensions of lambda (expecting K x M)");
        }
        // finally recompute
        Elog_beta_ = dirichlet_entropy_2d(lambda_);
    }

    // If a matrix pointer is provided, use it; otherwise initialize randomly.
    void init_global_parameter(const RowMajorMatrixXf* m_lambda = nullptr);

    // Perform the E-step for a given minibatch.
    // Returns sufficient statistics (K x M) computed from the minibatch.
    // Single-threaded, supposed to be run in parallel in spatial patches
    RowMajorMatrixXf do_e_step(Minibatch& batch, bool return_ss = true);

    void approx_ll(Minibatch& batch);

    // Update the global lambda parameter with one minibatch.
    void update_lambda(Minibatch& batch);

    // Compute approximate scores for monitoring progress.
    std::tuple<double, double, double, double, double> approx_score(Minibatch& batch);

private:
    int K_;   // number of topics
    int M_;   // number of features
    int N_;   // total number of pixels (global)
    double tau0_;
    double kappa_;
    int updatect_;
    int max_iter_inner_;
    double tol_;

    RowMajorMatrixXf Elog_beta_; // K x M: expectation of log β
    RowMajorMatrixXf lambda_;    // K x M: variational parameter for β
    RowVectorXf alpha_;  // 1 X K: global prior for θ
    RowVectorXf eta_;    // 1 x M: prior for β (stored as a row vector)
    std::mt19937 rng_;
};





class EMPoisReg {
public:
    int debug_;
    MLEOptions mle_opts_;
    double size_factor_ = 1000.;

    EMPoisReg() {}
    EMPoisReg(MLEOptions opts, double L = 1000, bool exact = false, int debug = 0)
        : mle_opts_(opts), size_factor_(L), exact_(exact), debug_(debug) {}

    void init(MLEOptions opts, double L = 1000, bool exact = false, int debug = 0) {
        mle_opts_ = opts;
        size_factor_ = L;
        exact_ = exact;
        debug_ = debug;
    }

    template<typename Derived>
    void init_global_parameter(const Eigen::MatrixBase<Derived>& beta) {
        beta_ = beta.template cast<double>();
        M_ = beta_.rows();
        K_ = beta_.cols();
        if (M_ < 1 || K_ < 1) {
            error("%s: Invalid dimensions of beta", __func__);
        }
        initialized_ = true;
    }

    void run_em(Minibatch& batch, int max_iter = 20, double tol = 1e-4) {
        if (!initialized_) {
            error("%s: not initialized, run init_global_parameter first", __func__);
        }
        if (batch.M != M_) {
            error("%s: dimension M mismatch (%d vs %d)", __func__, batch.M, M_);
        }

        bool theta_init = true;
        if (batch.theta.cols() != K_) {
            batch.theta = RowMajorMatrixXf::Zero(batch.n, K_);
            theta_init = false;
        }
        RowMajorMatrixXf theta_old = batch.theta;
        double meanchange_theta = tol + 1.0;
        VectorXf ones = VectorXf::Ones(M_);
        for (int it = 0; it < max_iter; it++) {
            SparseMatrix<float, Eigen::RowMajor> ymtx = batch.psi * batch.mtx; // n x M
            VectorXf rowsums = ymtx * ones; // n
            MLEStats stats;
            for (int j = 0; j < batch.n; j++) { // solve pnmf for \theta_j
                size_t nnz = ymtx.row(j).nonZeros();
                Document y;
                y.ids.reserve(nnz);
                y.cnts.reserve(nnz);
                for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(ymtx, j); it; ++it) {
                    y.ids.push_back(it.col());
                    y.cnts.push_back(it.value());
                }
                VectorXd b; // K
                if (theta_init) {
                    b = batch.theta.row(j).transpose().cast<double>();
                }
                double c = rowsums[j] / size_factor_;
                if (exact_) {
                    pois_log1p_mle_exact(beta_, y, c, mle_opts_, b, stats);
                } else {
                    pois_log1p_mle(beta_, y, c, mle_opts_, b, stats);
                }
                batch.theta.row(j) = b.transpose().cast<float>();
                VectorXf lam = (beta_ * b).cast<float>(); // M
                lam = (((lam.array().exp() - 1.0)).max(mle_opts_.ridge) * c).log();
                for (Eigen::SparseMatrix<float, Eigen::RowMajor>::InnerIterator it(batch.psi, j); it; ++it) {
                    // sum_m x_{im} log λ_{im}
                    it.valueRef() = batch.mtx.row(it.col()).dot(lam);
                }
            }
            batch.psi = (batch.psi + batch.wij).unaryExpr([](float x) { return std::exp(x);});
            colNormalizeInPlace(batch.psi);
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
    }

private:
    int32_t K_ = 0, M_ = 0;
    RowMajorMatrixXd beta_;  // M x K
    bool exact_ = false;
    bool initialized_ = false;

};
