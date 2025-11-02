// argmax_{b} Σ_i [ y_i log λ_i − λ_i ], log(1+λ_i/c_i) = (A b)_i (A, b ≥ 0)

#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <random>
#include <optional>
#include <algorithm>
#include "numerical_utils.hpp"
#include "dataunits.hpp"
#include "error.hpp"
#include "newtons.hpp"

#include <Eigen/Core>
#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Eigen::ArrayXd;

struct MLEOptions {
	OptimOptions optim;
	bool  exact_zero = false;
	double ridge     = 1e-12;
	double soft_tau  = 1e-3;    // for softplus if needed
	uint32_t se_flag = 0;       // 0: none, 1: fisher, 2: robust, 3: both
	uint32_t hc_type = 1;
	bool store_cov   = false;   // Store covariance matrices for estimates
    bool compute_residual = false;
    bool compute_var_mu = false;
    MLEOptions() = default;
    MLEOptions(const OptimOptions& optim_) : optim(optim_) {}
    void mle_only_mode() {
        se_flag = 0;
        store_cov = false;
        compute_residual = false;
        compute_var_mu = false;
    }
};

struct MLEStats {
	OptimStats optim;
	// Optional standard errors and/or covariance
	VectorXd se_fisher;
	VectorXd se_robust;
	MatrixXd cov_fisher;
    MatrixXd cov_robust;
	// Goodness-of-fit
	double pll; // per-token log-likelihood
	double residual, var_mu;
    MLEStats() = default;
    MLEStats(const OptimStats& optim_) : optim(optim_) {}
};

// ---------------- Problem Definition for pois_log1p_mle_exact --------------
class PoisRegExactProblem {
public:
    const RowMajorMatrixXd& A; // N x K
    const VectorXd& c;
    const VectorXd* o;
    const MLEOptions& opt;
    const bool has_offset;
    bool nonnegative;
    VectorXd yvec;

    void init(const std::vector<uint32_t>& ids_,
             const std::vector<double>& cnts_) {
        yvec = VectorXd::Zero(A.rows());
        for (size_t j = 0; j < ids_.size(); ++j) {
            yvec[ids_[j]] = cnts_[j];
        }
        double min_lower_bound = 0;
        if (opt.optim.b_min) min_lower_bound = opt.optim.b_min->minCoeff();
        nonnegative = min_lower_bound >= 0.0;
        if (has_offset) {
            nonnegative = nonnegative && (o->minCoeff() >= 0.0);
        }
    }

    PoisRegExactProblem(const RowMajorMatrixXd& A_,
        const std::vector<uint32_t>& ids_, const std::vector<double>& cnts_,
        const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_), c(c_), o(o_), opt(opt_), has_offset(o != nullptr) {
        init(ids_, cnts_);
    }
    PoisRegExactProblem(const RowMajorMatrixXd& A_, const Document& y_,
        const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_), c(c_), o(o_), opt(opt_), has_offset(o != nullptr) {
        init(y_.ids, y_.cnts);
    }

    void eval(const VectorXd& bvec,
              double* f_out = nullptr,   // f = -logP
              VectorXd* g_out = nullptr, // df/db
              VectorXd* q_out = nullptr, // d2f/db2 (diag)
              ArrayXd* w_out = nullptr) const;
    void eval_safe(const VectorXd& bvec,
              double* f_out = nullptr,
              VectorXd* g_out = nullptr,
              VectorXd* q_out = nullptr,
              ArrayXd* w_out = nullptr) const;

    double f(const VectorXd& bvec) const {
        double f_val;
        eval(bvec, &f_val, nullptr, nullptr, nullptr);
        return f_val;
    }

    void grad(const VectorXd& bvec, VectorXd& gout) const {
        eval(bvec, nullptr, &gout, nullptr, nullptr);
    }

    auto make_Hv(const ArrayXd& w) const {
        return [&](const VectorXd& v) -> VectorXd {
            VectorXd Av = A * v;
            return A.transpose() * (w * Av.array()).matrix();
        };
    }

    ArrayXd residual(const VectorXd& bvec) const;

};

// ---------------- Problem Definition for pois_log1p_mle (sparse) ----------
class PoisRegSparseProblem {
public:
    // Pointers to original data
    const RowMajorMatrixXd& A;
    const std::vector<uint32_t>& ids;
    const VectorXd& c;
    const VectorXd* o;
    const MLEOptions& opt;

    // Precomputed values
    const bool has_offset;
    const size_t n;
    RowMajorMatrixXd Anz;
    MatrixXd AsqnzT;
    Eigen::Map<const VectorXd> yvec;
    VectorXd cS;
    VectorXd oS;
    VectorXd zak;
    MatrixXd zakl;
    VectorXd dZ;
    VectorXd zoak;
    bool nonnegative;

    void init() {
        const int K = static_cast<int>(A.cols());

        // Build submatrices for non-zero counts
        for (size_t j = 0; j < n; ++j) {
            Anz.row(j) = A.row(ids[j]);
        }
        AsqnzT = Anz.array().square().matrix().transpose();

        if (has_offset) oS.resize(n);
        for (size_t t = 0; t < n; ++t) {
            const int i = ids[t];
            cS[t] = c[i];
            if (has_offset) oS[t] = (*o)[i];
        }

        // Precompute for zero-count contributions
        VectorXd zmask = c;
        for (size_t j = 0; j < n; ++j) {
            zmask[ids[j]] = 0.0;
        }
        zak = A.transpose() * zmask;
        zakl = A.transpose() * (zmask.asDiagonal() * A);
        dZ = zakl.diagonal();
        zoak = VectorXd::Zero(K);
        if (has_offset) {
            zoak = A.transpose() * (zmask.array() * o->array()).matrix();
        }

        // Check if we need safe/heuristic to enforce non-negativity
        double min_lower_bound = 0;
        if (opt.optim.b_min) min_lower_bound = opt.optim.b_min->minCoeff();
        nonnegative = min_lower_bound >= 0.0;
        if (has_offset) {
            nonnegative = nonnegative && (o->minCoeff() >= 0.0);
        }
    }

    PoisRegSparseProblem(const RowMajorMatrixXd& A_, const Document& y_,
        const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_), ids(y_.ids), c(c_), o(o_), opt(opt_), has_offset(o != nullptr),
          n(y_.ids.size()), Anz(n, A.cols()), yvec(y_.cnts.data(), n), cS(n) {
        init();
    }
    PoisRegSparseProblem(const RowMajorMatrixXd& A_,
        const std::vector<uint32_t>& ids_, const std::vector<double>& cnts_,
        const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_), ids(ids_), c(c_), o(o_), opt(opt_), has_offset(o != nullptr),
          n(ids.size()), Anz(n, A.cols()), yvec(cnts_.data(), n), cS(n) {
        init();
    }

    void eval(const VectorXd& bvec,
              double* f_out = nullptr,
              VectorXd* g_out = nullptr,
              VectorXd* q_out = nullptr,
              ArrayXd* w_out = nullptr) const;
    void eval_safe(const VectorXd& bvec,
              double* f_out = nullptr,
              VectorXd* g_out = nullptr,
              VectorXd* q_out = nullptr,
              ArrayXd* w_out = nullptr) const;

    double f(const VectorXd& bvec) const {
        double f_val;
        eval(bvec, &f_val, nullptr, nullptr, nullptr);
        return f_val;
    }

    void grad(const VectorXd& bvec, VectorXd& gout) const {
        eval(bvec, nullptr, &gout, nullptr, nullptr);
    }

    auto make_Hv(const ArrayXd& w) const {
        return [&](const VectorXd& v) -> VectorXd {
            VectorXd Av = Anz * v;
            VectorXd Hv_nz = Anz.transpose() * (w * Av.array()).matrix();
            return Hv_nz + zakl * v;
        };
    }

    ArrayXd residual(const VectorXd& bvec) const;
};

// Poisson regression with log(1+λ/c) link
// Return -logP (final objective)

double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A,
    const Document& y,
    const VectorXd& c,    // length N, scaling factors
    const VectorXd* o,    // length N, offset
    const MLEOptions& opt,
    VectorXd& b, MLEStats& stats, int32_t debug_ = 0,
    ArrayXd* res_ptr = nullptr);

// With 2nd order approximation for zero-set speed up
double pois_log1p_mle(
    const RowMajorMatrixXd& A,
    const Document& y,
    const VectorXd& c,
    const VectorXd* o,
    const MLEOptions& opt,
    VectorXd& b, MLEStats& stats, int32_t debug_ = 0,
    ArrayXd* res_ptr = nullptr);

// Wrappers for using a constant scaling factor
double pois_log1p_mle(
    const RowMajorMatrixXd& A, const Document& y, double c,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_ = 0, ArrayXd* res_ptr = nullptr);

double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A, const Document& y, double c,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_ = 0, ArrayXd* res_ptr = nullptr);

// Compute Fisher and/or robust SE for Poisson with log1p link.
void pois_log1p_compute_se(
    const RowMajorMatrixXd& X, const Document& y, const VectorXd& c,
    const VectorXd* o, const MLEOptions& opt, const Eigen::Ref<const VectorXd>& b, MLEStats& stats);

// (dense y is used only if opt.se_flag & 0x2)
void pois_log1p_compute_se(
    const RowMajorMatrixXd& X, const VectorXd& y, const VectorXd& c,
    const VectorXd* o, const MLEOptions& opt, const Eigen::Ref<const VectorXd>& b, MLEStats& stats);

ArrayXd pois_log1p_residual(
    const RowMajorMatrixXd& X, const Document& y, const VectorXd& c,
    const VectorXd* o, const Eigen::Ref<const VectorXd>& b);

ArrayXd pois_log1p_residual(
    const RowMajorMatrixXd& X, const VectorXd& y, const VectorXd& c,
    const VectorXd* o, const Eigen::Ref<const VectorXd>& b);
