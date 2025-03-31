#pragma once

#include <vector>
#include <string>
#include <random>
#include <cmath>
#include <iostream>
#include <tuple>
#include "numerical_utils.hpp"
#include "Eigen/Dense"
#include "Eigen/Sparse"

// Using Eigen types for convenience
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

// Structure holding a minibatch
struct Minibatch {
    // Required:
    int n;        // number of anchors
    int N;        // number of pixels
    int M;        // number of features
    SparseMatrix<double> mtx;  // (N x M) observed data matrix
    SparseMatrix<double, Eigen::RowMajor> logitwij;  // (N x n); d_psi E_q[log P(Cij)]
    MatrixXd gamma; // (n x K); ~P(k|j)
    // Does not need to be initialized:
    SparseMatrix<double, Eigen::RowMajor> psi;   // (N x n); ~P(j|i)
    MatrixXd phi;   // (N x K); ~P(k|i)
    double ll;      // Log-likelihood value computed during the E-step
};

class OnlineSLDA {
public:
    int verbose_;
    // Constructor
    OnlineSLDA() {}
    OnlineSLDA(int K, int M, int N,
        unsigned int seed = std::random_device{}(),
        VectorXd* alpha = nullptr, VectorXd* eta = nullptr,
        double tau0 = 9.0, double kappa = 0.7,
        int iter_inner = 50, double tol = 1e-4,
        int verbose = 0) {
        init(K, M, N, seed, alpha, eta, tau0, kappa, iter_inner, tol, verbose);
    }
    void init(int K, int M, int N,
        unsigned int seed = std::random_device{}(),
        VectorXd* alpha = nullptr, VectorXd* eta = nullptr,
        double tau0 = 9.0, double kappa = 0.7,
        int iter_inner = 50, double tol = 1e-4,
        int verbose = 0) {
        K_ = K;
        M_ = M;
        N_ = N;
        rng_.seed(seed);
        tau0_ = std::max(tau0 + 1., 1.0);
        kappa_ = kappa;
        updatect_ = 0;
        max_iter_inner_ = iter_inner;
        tol_ = tol;
        verbose_ = verbose;
        // Global alpha: expect a K-dimensional vector (K x 1)
        if (alpha && alpha->size() == K_) {
            alpha_ = (*alpha).transpose();
        } else {
            alpha_ = Eigen::RowVectorXd::Constant(K_, 1./K_);
        }
        // Global eta: stored as a row vector (1 x M)
        if (eta && eta->size() == M_) {
            eta_ = *eta;
            eta_ = eta_.transpose();
        } else {
            eta_ = VectorXd::Constant(M_, 1./K_).transpose();
        }
    }

    // Initialize the global variational parameter lambda for topics.
    void init_global_parameter(const MatrixXd& lambda) {
        lambda_ = lambda;
        if (lambda_.rows() == M_ && lambda_.cols() == K_) {
            lambda_ = lambda_.transpose();
        } else if (lambda_.rows() != K_ || lambda_.cols() != M_) {
            throw std::invalid_argument("Invalid dimensions of lambda (expecting K x M)");
        }
        // Compute E[log(beta)]
        Elog_beta_ = dirichlet_entropy_2d(lambda_); // (K x M)
    }
    // If a matrix pointer is provided, use it; otherwise initialize randomly.
    void init_global_parameter(const MatrixXd* m_lambda = nullptr) {
        if (m_lambda == nullptr) {
            lambda_.resize(K_, M_);
            std::gamma_distribution<double> gamma_dist(100.0, 1.0/100.0);
            for (int i = 0; i < K_; i++) {
                for (int j = 0; j < M_; j++) {
                    lambda_(i, j) = gamma_dist(rng_);
                }
            }
            Elog_beta_ = dirichlet_entropy_2d(lambda_); // (K x M)
        } else {
            init_global_parameter(*m_lambda);
        }
    }

    // Perform the E-step for a given minibatch.
    // Returns sufficient statistics (K x M) computed from the minibatch.
    MatrixXd do_e_step(Minibatch& batch, bool return_ss = true) {
        // (N x M) * (M x K) = (N x K)
        MatrixXd Xb = batch.mtx * Elog_beta_.transpose();
        // If gamma is not set, randomly initialize
        if (batch.gamma.size() == 0) {
            batch.gamma.resize(batch.n, K_);
            std::gamma_distribution<double> gamma_dist(100.0, 1.0/100.0);
            for (int i = 0; i < batch.n; i++) {
                for (int j = 0; j < K_; j++) {
                    batch.gamma(i, j) = gamma_dist(rng_);
                }
            }
        }

        MatrixXd gamma_old = batch.gamma;
        MatrixXd phi_old = batch.phi;
        MatrixXd Elog_theta = dirichlet_entropy_2d(batch.gamma); // (n x K)
        if (batch.psi.size() == 0) {
            batch.psi = batch.logitwij; // (N x n)
            expitAndRowNormalize(batch.psi);
        }
        if (verbose_ > 1) {
            approx_ll(batch);
            printf("Initialize E-step, E_q[ll] %.4e\n", batch.ll);
        }

        double meanchange = tol_ + 1;
        double meanchange_phi = tol_ + 1;
        int it = 0;
        while (it < max_iter_inner_ && meanchange > tol_) {
            it++;
            // Update phi.
            // psi: (N x n) * Elog_theta (n x K) → phi: (N x K)
            batch.phi = batch.psi * Elog_theta + Xb;
            // exponentiate and row-normalize
            for (int i = 0; i < batch.phi.rows(); i++) {
                batch.phi.row(i) = (batch.phi.row(i).array() - logsumexp(batch.phi.row(i)).first).exp();
            }
            // Update psi. (psi always has the same sparsity pattern as logitwij)
            // we could parallelize the row-wise computation
            // #pragma omp parallel for schedule(dynamic)
            for (int i = 0; i < batch.psi.rows(); ++i) {
                double extra = (batch.phi.row(i).array() * Xb.row(i).array()).sum();
                // Iterate over the nonzero entries
                for (SparseMatrix<double, Eigen::RowMajor>::InnerIterator it1(batch.psi, i), it2(batch.logitwij, i); it1 && it2; ++it1, ++it2) {
                    int j = it1.col();
                    // dot product or i-th row of phi with j-th column of Elog_theta
                    double sum_val = batch.phi.row(i).dot(Elog_theta.row(j));
                    // Updated value at (i,j)
                    batch.psi.coeffRef(i, j) = it2.value() + sum_val + extra;
                }
            }
            expitAndRowNormalize(batch.psi);
            // Update gamma: gamma = alpha + psi^T * phi.
            // psi^T: (n x N), phi: (N x K) → gamma: (n x K)
            // broadcast alpha_
            batch.gamma = batch.psi.transpose() * batch.phi;
            batch.gamma += alpha_.replicate(batch.n, 1);
            // Update E[log θ] using the new gamma.
            Elog_theta = dirichlet_entropy_2d(batch.gamma);
            // Check convergence
            meanchange = mean_max_row_change(batch.gamma, gamma_old);
            gamma_old = batch.gamma;
            meanchange_phi = mean_max_row_change(batch.phi, phi_old);
            phi_old = batch.phi;
            if (verbose_ > 2 || (verbose_ > 1 && it % 10 == 0)) {
                printf("E-step, iteration %d, mean change in gamma: %.4e; mean change in phi: %.4e\n", it, meanchange, meanchange_phi);
            }
        }
        if (verbose_ > 1) {
            approx_ll(batch);
            printf("E-step finished in %d iterations. E_q[ll] %.4e\n", it, batch.ll);
        }

        if (return_ss) {
            // Compute sufficient statistics: sstats = phi^T * mtx → (K x M)
            MatrixXd sstats = batch.phi.transpose() * batch.mtx;
            return sstats;
        } else {
            // Return an empty matrix
            return MatrixXd::Zero(0, 0);
        }
    }

    void approx_ll(Minibatch& batch) {
        // Compute the log-likelihood for a minibatch.
        MatrixXd Xb = batch.mtx * Elog_beta_.transpose(); // (N x K)
        MatrixXd ll_mat = batch.psi.transpose() * Xb;
        double ll_tot = 0.0;
        for (int i = 0; i < ll_mat.rows(); i++) {
            double lse = logsumexp(ll_mat.row(i)).first;
            ll_tot += (ll_mat.row(i).array() - lse).sum();
        }
        batch.ll = ll_tot / batch.n;
    }

    // Update the global lambda parameter with one minibatch.
    void update_lambda(Minibatch& batch) {
        MatrixXd sstats = do_e_step(batch);
        if (verbose_ > 0) {
            auto scores = approx_score(batch);
            printf("%d-th global update. Scores: %.4e, %.4e, %.4e, %.4e, %.4e\n", updatect_, std::get<0>(scores), std::get<1>(scores), std::get<2>(scores), std::get<3>(scores), std::get<4>(scores));

        }
        double rhot = std::pow(tau0_ + updatect_, -kappa_);
        // Update rule: λ = (1 - rhot) * λ + rhot * ((N / batch.N) * (η + sstats))
        double scale = static_cast<double>(N_) / batch.N;
        // We assume η is stored as a row vector; replicate it to match λ’s dimensions.
        lambda_ = (1 - rhot) * lambda_ + rhot * ((eta_.replicate(K_, 1) + sstats) * scale);
        Elog_beta_ = dirichlet_entropy_2d(lambda_);
        updatect_++;
    }

    // Compute approximate scores for monitoring progress.
    std::tuple<double, double, double, double, double> approx_score(Minibatch& batch) {
        // Score for pixels: sum( φ .* (mtx * Elog_beta_.transpose())/batch.N )
        MatrixXd X = batch.mtx * Elog_beta_.transpose();
        double score_pixel = (batch.phi.array() * (X.array() / batch.N)).sum();
        double score_patch = batch.ll;

        // Score for gamma: E[log p(θ | α) - log q(θ | γ)]
        MatrixXd Elog_theta = dirichlet_entropy_2d(batch.gamma);
        double score_gamma = ((alpha_.replicate(batch.n, 1) - batch.gamma).array() * Elog_theta.array()).sum();
        for (int i = 0; i < batch.gamma.rows(); i++) {
            double row_sum = batch.gamma.row(i).sum();
            for (int k = 0; k < batch.gamma.cols(); k++) {
                score_gamma += std::lgamma(batch.gamma(i, k));
            }
            score_gamma -= std::lgamma(row_sum);
        }
        score_gamma /= batch.n;

        // Score for beta: E[log p(β | η) - log q(β | λ)]
        double score_beta = ((eta_.replicate(K_,1) - lambda_).array() * Elog_beta_.array()).sum();
        for (int i = 0; i < lambda_.rows(); i++) {
            double row_sum = lambda_.row(i).sum();
            for (int j = 0; j < lambda_.cols(); j++) {
                score_beta += std::lgamma(lambda_(i, j));
            }
            score_beta -= std::lgamma(row_sum);
        }
        double total_score = N_*1./batch.N * batch.n * (score_patch + score_gamma) + score_beta;
        return std::make_tuple(score_pixel, score_patch, score_gamma, score_beta, total_score);
    }

private:
    int K_;   // number of topics
    int M_;   // number of features
    int N_;   // total number of pixels (global)
    double tau0_;
    double kappa_;
    int updatect_;
    int max_iter_inner_;
    double tol_;

    MatrixXd Elog_beta_; // K x M: expectation of log β
    MatrixXd lambda_;    // K x M: variational parameter for β
    Eigen::RowVectorXd alpha_;     // 1 X K: global prior for θ
    MatrixXd eta_;       // 1 x M: prior for β (stored as a row vector)
    std::mt19937 rng_;
};
