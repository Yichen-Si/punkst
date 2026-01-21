#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <memory>
#include "numerical_utils.hpp"
#include "dataunits.hpp"
#include "error.hpp"
#include "newtons.hpp"
#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <Eigen/Dense>
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::MatrixXd;
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// \lambda_i = c_i * \sum_k \theta_{ik} * beta_k, beta_k >= 0
// \sum_i\lambda_i = \sum_k [(\sum_i c_i * \theta_{ik}) * beta_k]
//                := \sum_k a_sum_k * beta_k
class MixPoisRegProblem {
public:

    MixPoisRegProblem(const RowMajorMatrixXd& theta,  // N x K
                    const VectorXd& c,       // N
                    const VectorXd& a_sum,   // K
                    const std::vector<uint32_t>& ids,
                    const std::vector<double>& cnts,
                    double lambda_floor = 1e-20)
        : theta_(theta), c_(c), a_sum_(a_sum), ids_(ids), cnts_(cnts),
          N_(static_cast<int>(theta.rows())),
          K_(static_cast<int>(theta.cols())),
          lambda_floor_(lambda_floor) {}

    // beta_k >= 0, a_sum_k := \sum_i c_i * theta_{ik}
    double f(const VectorXd& beta) const {
        double val = beta.dot(a_sum_);
        const int nnz = static_cast<int>(ids_.size());
        for (int j = 0; j < nnz; ++j) {
            const double x = cnts_[j];
            uint32_t i = ids_[j];
            // \lambda_i = \sum_k c_i * theta_{ik} * beta_k
            const double lam = std::max(lambda_floor_, c_[i] * theta_.row(i).dot(beta));
            val -= x * std::log(lam);
        }
        return val;
    }

    void grad(const VectorXd& beta, VectorXd& g_out) const {
        eval(beta, nullptr, &g_out, nullptr, nullptr);
    }

    // Computes f, g, q=diag(H), and w = x/lam^2 for Hv.
    void eval(const VectorXd& beta, double* f_out,
        VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const {
        const int nnz = static_cast<int>(ids_.size());

        if (f_out) *f_out = beta.dot(a_sum_);
        if (g_out) *g_out = a_sum_;
        if (q_out) q_out->setZero(K_);
        if (w_out) {if (w_out->size() != nnz) w_out->resize(nnz);}

        for (int j = 0; j < nnz; ++j) {
            const int i = static_cast<int>(ids_[j]);
            const double x = cnts_[j];
            const double ci = c_[i];
            auto row = theta_.row(i);
            double lam = std::max(lambda_floor_, ci * row.dot(beta));
            const double w_fisher = x / lam / lam;
            if (f_out) *f_out -= x * std::log(lam);
            if (g_out) {g_out->noalias() -= (ci*x/lam) * row.transpose(); }
            if (q_out) {q_out->array() += (ci*ci*w_fisher) * row.array().square().transpose(); }
            if (w_out) (*w_out)[j] = w_fisher;
        }
    }

    // Hv(v) = sum_i w_i * a_i * (a_i^T v)
    auto make_Hv(const ArrayXd& w) const {
        return [this, &w](const VectorXd& v) -> VectorXd {
            VectorXd out = VectorXd::Zero(K_);
            const int nnz = static_cast<int>(ids_.size());
            for (int j = 0; j < nnz; ++j) {
                const int i = static_cast<int>(ids_[j]);
                double t = theta_.row(i).dot(v);
                out.noalias() += (c_[i] * c_[i] * w[j] * t) * theta_.row(i).transpose();
            }
            return out;
        };
    }

private:
    const RowMajorMatrixXd& theta_;
    const VectorXd& c_;
    const VectorXd& a_sum_;
    const std::vector<uint32_t>& ids_;
    const std::vector<double>& cnts_;
    int N_, K_;
    double lambda_floor_;
};

enum class MixPoisLink { Log, Log1p };

// x_im ~ Poisson(lambda_im), \lambda_im = c_i * \sum_k \theta_ik * g(eta_km)
// g(eta) = exp(eta) - 1 (log1p link) or exp(eta) (log link)
// precomputed a_sum_k := \sum_i c_i * \theta_ik
class MixPoisReg {
public:
    MixPoisReg(const RowMajorMatrixXd& theta, // N x K
               const VectorXd& c,     // N
               const VectorXd& a_sum, // K
               const OptimOptions& opt,
               MixPoisLink link)
        : theta_(theta), c_(c), a_sum_(a_sum), opt_(opt), link_(link) {
        N_ = static_cast<int>(theta_.rows());
        K_ = static_cast<int>(theta_.cols());
        opt_.tron.enabled = true;
    }

    // Fit one column m given sparse x_{*m} and eta init (K).
    VectorXd fit_one(const std::vector<uint32_t>& ids,
                     const std::vector<double>& cnts,
                     const VectorXd& eta0) const {
        if (link_ == MixPoisLink::Log1p) {
            return fit_one_log1p(ids, cnts, eta0);
        }
        return fit_one_log(ids, cnts, eta0);
    }

    // Return: K x M
    MatrixXd transform(const std::vector<Document>& X_cols,
                       MatrixXd& Eta, int n_threads = 1) const {
        const int M = static_cast<int>(X_cols.size());
        if (Eta.rows() != K_ || Eta.cols() != M) {
            error("%s: wrong size of Eta (%d x %d vs %d x %d)", __func__,
                  (int)Eta.rows(), (int)Eta.cols(), K_, M);
        }
        if (n_threads == 1) {
            for (int m = 0; m < M; ++m) {
                VectorXd eta0 = Eta.col(m);
                Eta.col(m) = fit_one(X_cols[m].ids, X_cols[m].cnts, eta0);
            }
            return Eta;
        }

        std::unique_ptr<tbb::global_control> ctrl;
        ctrl = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, n_threads);

        tbb::parallel_for(tbb::blocked_range<int>(0, M),
                          [&](const tbb::blocked_range<int>& r) {
            for (int m = r.begin(); m < r.end(); ++m) {
                VectorXd eta0 = Eta.col(m);
                Eta.col(m) = fit_one(X_cols[m].ids, X_cols[m].cnts, eta0);
            }
        });

        return Eta;
    }

private:
    VectorXd fit_one_log1p(const std::vector<uint32_t>& ids,
                           const std::vector<double>& cnts,
                           const VectorXd& eta0) const {
        if (ids.empty()) {
            return VectorXd::Zero(K_);
        }
        VectorXd eta = eta0;
        if (eta.size() != K_) {
            error("%s: wrong size of initial eta (%d vs %d)", __func__, (int)eta.size(), K_);
        }
        eta = eta.array().max(0.0);

        // Work variable: beta := exp(eta) - 1 >= 0
        VectorXd beta(K_);
        for (int k = 0; k < K_; ++k) {
            const double e = std::exp(std::min(eta[k], exp_cap_));
            beta[k] = std::max(0.0, e - 1.0);
        }

        MixPoisRegProblem P(theta_, c_, a_sum_, ids, cnts, lambda_floor_);
        OptimStats stats{}; //
        tron_solve(P, beta, opt_, stats); // use non-negative bound to fit beta

        // Back-transform: eta = log(1 + beta)
        for (int k = 0; k < K_; ++k) {
            eta[k] = std::log1p(std::max(0.0, beta[k]));
        }
        eta = eta.array().max(0.0);
        return eta;
    }

    VectorXd fit_one_log(const std::vector<uint32_t>& ids,
                         const std::vector<double>& cnts,
                         const VectorXd& eta0) const {
        if (ids.empty()) {
            return VectorXd::Zero(K_);
        }
        VectorXd eta = eta0;
        if (eta.size() != K_) {
            error("%s: wrong size of initial eta (%d vs %d)", __func__, (int)eta.size(), K_);
        }

        // Work variable: beta := exp(eta) > 0
        VectorXd beta(K_);
        for (int k = 0; k < K_; ++k) {
            const double e = std::exp(std::min(eta[k], exp_cap_));
            beta[k] = std::max(beta_floor_, e);
        }

        MixPoisRegProblem P(theta_, c_, a_sum_, ids, cnts, lambda_floor_);
        OptimStats stats{}; //
        tron_solve(P, beta, opt_, stats); // use non-negative bound to fit beta

        // Back-transform: eta = log(beta)
        for (int k = 0; k < K_; ++k) {
            eta[k] = std::log(std::max(beta_floor_, beta[k]));
        }
        return eta;
    }
    const RowMajorMatrixXd& theta_;  // N x K
    const VectorXd& c_;      // N
    const VectorXd& a_sum_;  // K
    OptimOptions opt_;
    MixPoisLink link_;
    double lambda_floor_ = 1e-30;
    double beta_floor_   = 1e-30;
    double exp_cap_      = 40.0;
    int N_ = 0;
    int K_ = 0;
};
