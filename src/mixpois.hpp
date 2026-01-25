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

struct MLEOptions;
struct MLEStats;

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

/*
    Poisson mixture regression
    \lambda_i = c_i * \sum_k a_{ik} (\exp(o_{ik} + x_i b_k) - 1)
    Assuming x_i \in \set{-1,1}
*/

// Precomputed context for one contrast
class MixPoisLog1pSparseContext {
public:
    const RowMajorMatrixXd& A;   // N x K, rows sum to 1
    const VectorXd& x;           // N
    const VectorXd& c;           // N
    const MLEOptions& opt;
    const int N_active; // active sample count for this contrast
    const int N, K;
    bool lda_uncertainty = false;
    double size_factor = 1.0;
    bool has_s2 = false;
    bool has_m2 = false;
    bool has_a_unc = false;
    bool has_a2 = false;
    bool has_a2_full = false;

    const VectorXd* s1p = nullptr;
    const VectorXd* s1m = nullptr;
    const VectorXd* s2p = nullptr;
    const VectorXd* s2m = nullptr;
    const MatrixXd* m2p = nullptr;
    const MatrixXd* m2m = nullptr;
    const VectorXd* a1p = nullptr;
    const VectorXd* a1m = nullptr;
    const VectorXd* a2p = nullptr;
    const VectorXd* a2m = nullptr;
    const MatrixXd* a2p_full = nullptr;
    const MatrixXd* a2m_full = nullptr;

    MixPoisLog1pSparseContext(const RowMajorMatrixXd& A_,
            const VectorXd& x_, const VectorXd& c_, const MLEOptions& opt_,
            bool robust_se_full, int N_active_ = -1,
            bool lda_uncertainty_ = false, double size_factor_ = 1.0);

    MixPoisLog1pSparseContext(const RowMajorMatrixXd& A_,
            const VectorXd& x_, const VectorXd& c_, const MLEOptions& opt_,
            const VectorXd& s1p_, const VectorXd& s1m_,
            const VectorXd* s2p_ = nullptr, const VectorXd* s2m_ = nullptr,
            const MatrixXd* m2p_ = nullptr, const MatrixXd* m2m_ = nullptr,
            int N_active_ = -1, bool lda_uncertainty_ = false,
            double size_factor_ = 1.0);

private:
    VectorXd s1p_store, s1m_store;
    VectorXd s2p_store, s2m_store;
    MatrixXd m2p_store, m2m_store;
    VectorXd a1p_store, a1m_store;
    VectorXd a2p_store, a2m_store;
    MatrixXd a2p_full_store, a2m_full_store;
};

class MixPoisLog1pSparseProblem {
public:
    const RowMajorMatrixXd& A;   // A_rows x K, rows sum to 1
    const VectorXd& x;           // N
    const VectorXd& c;           // N
    const MLEOptions& opt;
    bool robust_se_diagonal_only = true;

    const int N_active; // active sample count for this contrast
    const int N, K;
    int n = 0; // nnz
    RowMajorMatrixXd Anz;  // n x K
    VectorXd xS, cS;       // n

    // Sums of c_i a_{ik} over Zero-set split by sign of x
    VectorXd Sz_plus, Sz_minus;
    // Sums of (c_i a_{ik})^2 over Zero-set split by sign of x
    VectorXd Sz2_plus,  Sz2_minus;
    // Full second-moment matrices over Zero-set split by sign of x
    MatrixXd Mz_plus, Mz_minus;
    // Uncertainty aggregates over Zero-set split by sign of x
    VectorXd Az1_plus, Az1_minus;
    VectorXd Az2_plus, Az2_minus;
    MatrixXd Az2_full_plus, Az2_full_minus;

    // Cached (depends on b, updated in eval)
    mutable MatrixXd Vnz;       // n x K, v_{ik} = a_{ik} * exp(o_k + x_i b_k)
    mutable ArrayXd w_cache;    // n, w_i = (c_i x_i)^2 / lambda_i
    mutable VectorXd lam_cache; // n, lambda_i for nonzero rows
    mutable VectorXd last_qZ;
    mutable ArrayXd slope_cache;

    MixPoisLog1pSparseProblem(const MixPoisLog1pSparseContext& ctx_);

    void reset_feature(const std::vector<uint32_t>& ids_,
        const std::vector<double>& cnts_, const VectorXd& oK_);

    double f(const VectorXd& b) const {
        double fval;
        eval(b, &fval, nullptr, nullptr, nullptr);
        return fval;
    }

    auto make_Hv(const ArrayXd& w) const {
        return [this, &w](const VectorXd& v) -> VectorXd {
            VectorXd tmp = Vnz * v; // n
            VectorXd Hv_nz = Vnz.transpose() * (w * tmp.array()).matrix();
            VectorXd Hv = Hv_nz;
            Hv.array() += last_qZ.array() * v.array();
            return Hv;
        };
    }

    void eval(const VectorXd& b, double* f_out,
              VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const {
        eval_safe(b, f_out, g_out, q_out, w_out);
    }
    void eval_safe(const VectorXd& b, double* f_out,
              VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const;

    void compute_se(const VectorXd& b_hat, const MLEOptions& opt, MLEStats& stats) const;

    const VectorXd& oK_ref() const { return *oK; }
    const double* y_ptr() const { return yptr; }
    Eigen::Index y_size() const { return ysize; }

private:
    const MixPoisLog1pSparseContext& ctx;
    const VectorXd* oK = nullptr;
    const double* yptr = nullptr;
    Eigen::Index ysize = 0;
};

double mix_pois_log1p_mle(const RowMajorMatrixXd& A,
    const std::vector<uint32_t>& ids, const std::vector<double>& cnts,
    const VectorXd& x, const VectorXd& c, const VectorXd& oK,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats,
    VectorXd& s1p, VectorXd& s1m,
    VectorXd* s2p = nullptr, VectorXd* s2m = nullptr, int N_active = -1);

/*
  Poisson mixture regression with log link + size factor (Unknown: b)
    lambda_i = c_i * sum_k a_{ik} * exp(o_k + x_i b_k)
*/
class MixPoisLogRegProblem {
public:
    const RowMajorMatrixXd& A;  // N x K (a_{ik})
    const VectorXd& y;          // N
    const VectorXd& x;          // N
    const VectorXd& c;          // N (size factors)
    const VectorXd& oK;         // K (offsets)
    const MLEOptions& opt;
    const double ridge;

    const int N_active; // active sample count for this contrast
    const int N, K;

    // Cached at current b (for Hv)
    mutable MatrixXd V;         // N x K, v_{ik} = a_{ik} * exp(o_k + x_i b_k)
    mutable ArrayXd  w_cache;   // N, w_i = (c_i x_i)^2 / lambda_i
    mutable VectorXd lam_cache; // N

    MixPoisLogRegProblem(const RowMajorMatrixXd& A_,
        const VectorXd& y_, const VectorXd& x_,
        const VectorXd& c_, const VectorXd& oK_, const MLEOptions& opt_,
        int N_active_ = -1);

    double f(const VectorXd& b) const {
        double fval;
        eval(b, &fval, nullptr, nullptr, nullptr);
        return fval;
    }

    auto make_Hv(const ArrayXd& w) const {
        const double ridge_local = ridge;
        return [this, &w, ridge_local](const VectorXd& v) -> VectorXd {
            VectorXd tmp = V * v; // N
            VectorXd Hv  = V.transpose() * (w * tmp.array()).matrix(); // K
            if (ridge_local > 0.0) Hv.noalias() += ridge_local * v;
            return Hv;
        };
    }

    void eval(const VectorXd& b, double* f_out,
        VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const;

    void compute_se(const VectorXd& b_hat, const MLEOptions& opt, MLEStats& stats) const;
};

double mix_pois_log_mle(const RowMajorMatrixXd& A,
    const VectorXd& y, const VectorXd& x,
    const VectorXd& c, const VectorXd& oK,
    MLEOptions& opt, VectorXd& b, MLEStats& stats, int N_active = -1);

/*
  NB2 mixture regression with log link + size factor (Unknown: b)
    lambda_i = c_i * sum_k a_{ik} * exp(o_k + x_i b_k)
*/
class MixNBLogRegProblem {
public:
    const RowMajorMatrixXd& A;  // N x K (a_{ik})
    const VectorXd& y;          // N
    const VectorXd& x;          // N
    const VectorXd& c;          // N (size factors)
    const VectorXd& oK;         // K (offsets)
    const MLEOptions& opt;
    const double ridge;
    const double alpha;

    const int N_active; // active sample count for this contrast
    const int N, K;

    // Cached at current b (for Hv)
    mutable MatrixXd V;         // N x K, v_{ik} = a_{ik} * exp(o_k + x_i b_k)
    mutable ArrayXd  w_cache;   // N, w_i = (c_i x_i)^2 / (lambda_i (1 + alpha lambda_i))
    mutable VectorXd lam_cache; // N

    MixNBLogRegProblem(const RowMajorMatrixXd& A_, int N_active_,
        const VectorXd& y_, const VectorXd& x_,
        const VectorXd& c_, const VectorXd& oK_, double alpha_,
        const MLEOptions& opt_);

    double f(const VectorXd& b) const {
        double fval;
        eval(b, &fval, nullptr, nullptr, nullptr);
        return fval;
    }

    auto make_Hv(const ArrayXd& w) const {
        const double ridge_local = ridge;
        return [this, &w, ridge_local](const VectorXd& v) -> VectorXd {
            VectorXd tmp = V * v; // N
            VectorXd Hv  = V.transpose() * (w * tmp.array()).matrix(); // K
            if (ridge_local > 0.0) Hv.noalias() += ridge_local * v;
            return Hv;
        };
    }

    void eval(const VectorXd& b, double* f_out,
        VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const;
    void compute_se(const VectorXd& b_hat, const MLEOptions& opt,
        MLEStats& stats) const;
};

// NB log1p mixture regression with sparse exact nonzeros and approximate zeros
class MixNBLog1pSparseApproxProblem {
public:
    const RowMajorMatrixXd& A;   // N x K, rows sum to 1
    const VectorXd& x;           // N
    const VectorXd& c;           // N
    const VectorXd& oK;          // K
    const std::vector<uint32_t>& ids; // indices with y>0
    Eigen::Map<const VectorXd> yvec;  // counts for ids
    const OptimOptions& opt;
    const double ridge;
    const double soft_tau;
    const double alpha; // NB dispersion

    const int N; // active sample count for this contrast
    const int K;
    const int n; // nnz
    RowMajorMatrixXd Anz;  // n x K
    VectorXd xS, cS;       // n

    // Zero-set aggregates split by sign of x
    VectorXd Sz_plus, Sz_minus;         // sum c_i a_ik
    VectorXd Szc2_plus, Szc2_minus;     // sum c_i^2 a_ik
    VectorXd Sz2_plus, Sz2_minus;       // sum c_i^2 a_ik^2
    double Cz, Cz2;                     // sum c_i and c_i^2 over zero-set

    // Cached (depends on b, updated in eval)
    mutable MatrixXd Vnz;       // n x K, v_{ik} = a_{ik} * exp(o_k + x_i b_k)
    mutable ArrayXd w_cache;    // n, w_i = (c_i x_i)^2 / (lambda_i (1 + alpha lambda_i))
    mutable VectorXd lam_cache; // n, lambda_i for nonzero rows
    mutable VectorXd last_qZ;
    mutable ArrayXd slope_cache;
    static constexpr double kZeroApproxTau = 0.2;

    MixNBLog1pSparseApproxProblem(const RowMajorMatrixXd& A_, int N_,
            const std::vector<uint32_t>& ids_, const std::vector<double>& cnts_,
            const VectorXd& x_, const VectorXd& c_, const VectorXd& oK_,
            double alpha_,
            const OptimOptions& opt_, double ridge_, double soft_tau_,
            const VectorXd& s1p, const VectorXd& s1m,
            const VectorXd& s1p_c2, const VectorXd& s1m_c2,
            const VectorXd& s2p, const VectorXd& s2m,
            double csum_p, double csum_m, double c2sum_p, double c2sum_m);

    double f(const VectorXd& b) const {
        double fval;
        eval(b, &fval, nullptr, nullptr, nullptr);
        return fval;
    }

    auto make_Hv(const ArrayXd& w) const {
        return [this, &w](const VectorXd& v) -> VectorXd {
            VectorXd tmp = Vnz * v; // n
            VectorXd Hv_nz = Vnz.transpose() * (w * tmp.array()).matrix();
            VectorXd Hv = Hv_nz;
            Hv.array() += last_qZ.array() * v.array();
            return Hv;
        };
    }

    void eval(const VectorXd& b, double* f_out,
              VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const {
        eval_safe(b, f_out, g_out, q_out, w_out);
    }

    void eval_safe(const VectorXd& b, double* f_out,
              VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const;
    void compute_se(const VectorXd& b_hat, const MLEOptions& opt, MLEStats& stats) const;
};
