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
#include <type_traits>
#include "numerical_utils.hpp"
#include "dataunits.hpp"
#include "error.hpp"
#include "newtons.hpp"

#include <Eigen/Core>
#include "Eigen/Dense"
#include <Eigen/Sparse>
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Eigen::ArrayXd;
using Eigen::SparseMatrix;

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

/*
    Poisson log(1+lambda/c) regression
    \lambda_i = c_i * \exp(o_i + \sum_k a_{ik} b_k)
*/
class PoisLog1pRegExactProblem {
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

    PoisLog1pRegExactProblem(const RowMajorMatrixXd& A_,
        const std::vector<uint32_t>& ids_, const std::vector<double>& cnts_,
        const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_), c(c_), o(o_), opt(opt_), has_offset(o != nullptr) {
        init(ids_, cnts_);
    }
    PoisLog1pRegExactProblem(const RowMajorMatrixXd& A_, const Document& y_,
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
        return [this, &w](const VectorXd& v) -> VectorXd {
            VectorXd Av = A * v;
            return A.transpose() * (w * Av.array()).matrix();
        };
    }

    ArrayXd residual(const VectorXd& bvec) const;

};

class PoisLog1pRegSparseProblem {
public:
    // Pointers to original data
    const RowMajorMatrixXd& A;
    std::vector<uint32_t> ids_storage; // only used when input is Eigen sparse
    std::vector<double> cnts_storage;
    const std::vector<uint32_t>& ids;
    const VectorXd& c;
    const VectorXd* o;
    const MLEOptions& opt;

    // Precomputed values
    const bool has_offset;
    const size_t n; // number of non-zero values
    RowMajorMatrixXd Anz; // rows of A for non-zero values, n x K
    MatrixXd AsqnzT;
    Eigen::Map<const VectorXd> yvec;
    VectorXd cS; // n (c)
    VectorXd oS; // n (offset)
    VectorXd zak;
    MatrixXd zakl;
    VectorXd dZ;
    VectorXd zoak;
    bool nonnegative;

    void init();

    PoisLog1pRegSparseProblem(const RowMajorMatrixXd& A_, const Document& y_,
        const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_), ids_storage(), cnts_storage(),
          ids(y_.ids), c(c_), o(o_), opt(opt_), has_offset(o != nullptr),
          n(y_.ids.size()), Anz(n, A.cols()), yvec(y_.cnts.data(), n), cS(n) {
        init();
    }
    PoisLog1pRegSparseProblem(const RowMajorMatrixXd& A_,
        const std::vector<uint32_t>& ids_, const std::vector<double>& cnts_,
        const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_), ids_storage(), cnts_storage(),
          ids(ids_), c(c_), o(o_), opt(opt_), has_offset(o != nullptr),
          n(ids.size()), Anz(n, A.cols()), yvec(cnts_.data(), n), cS(n) {
        init();
    }
    template <typename SparseDerived>
    PoisLog1pRegSparseProblem(const RowMajorMatrixXd& A_,
        const Eigen::SparseMatrixBase<SparseDerived>& y_sparse,
        const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_),
          ids_storage(static_cast<size_t>(y_sparse.derived().nonZeros())),
          cnts_storage(static_cast<size_t>(y_sparse.derived().nonZeros())),
          ids(ids_storage), c(c_), o(o_), opt(opt_), has_offset(o != nullptr),
          n(ids_storage.size()), Anz(n, A.cols()), yvec(cnts_storage.data(), n), cS(n) {
        if (y_sparse.size() != A.rows()) {
            error("%s: sparse observation length (%ld) must match rows of A (%ld)",
                  __func__, static_cast<long>(y_sparse.size()),
                  static_cast<long>(A.rows()));
        }
        fill_from_sparse_vector(y_sparse.derived());
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
        return [this, &w](const VectorXd& v) -> VectorXd {
            VectorXd Av = Anz * v;
            VectorXd Hv_nz = Anz.transpose() * (w * Av.array()).matrix();
            return Hv_nz + zakl * v;
        };
    }

    ArrayXd residual(const VectorXd& bvec) const;

private:
    template <typename SparseDerived>
    void fill_from_sparse_vector(const Eigen::SparseMatrixBase<SparseDerived>& y_sparse) {
        using Plain = typename Eigen::SparseMatrixBase<std::decay_t<SparseDerived>>::PlainObject;
        Plain y_eval = y_sparse.derived();
        const bool is_row_vector = (y_eval.rows() == 1 && y_eval.cols() != 1);
        const bool is_col_vector = (y_eval.cols() == 1);
        if (!Plain::IsVectorAtCompileTime && !(is_row_vector || is_col_vector)) {
            error("%s: sparse input must be a vector or a single row/column", __func__);
        }
        size_t pos = 0;
        for (Eigen::Index outer = 0; outer < y_eval.outerSize(); ++outer) {
            for (typename Plain::InnerIterator it(y_eval, outer); it; ++it) {
                if (pos >= ids_storage.size()) {
                    error("%s: more non-zero entries than expected (%zu)", __func__, ids_storage.size());
                }
                Eigen::Index idx;
                if constexpr (Plain::IsVectorAtCompileTime) {
                    idx = it.index();
                } else if (is_row_vector) {
                    idx = it.col();
                } else {
                    idx = it.row();
                }
                ids_storage[pos] = static_cast<uint32_t>(idx);
                cnts_storage[pos] = static_cast<double>(it.value());
                ++pos;
            }
        }
        if (pos != ids_storage.size()) {
            error("%s: expected %zu non-zero entries but found %zu", __func__, ids_storage.size(), pos);
        }
    }
};

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



// TO TEST
template <typename SparseDerived>
double pois_log1p_mle(
    const RowMajorMatrixXd& A,
    const Eigen::SparseMatrixBase<SparseDerived>& y_sparse,
    const VectorXd& c,
    const VectorXd* o,
    const MLEOptions& opt,
    VectorXd& b, MLEStats& stats, int32_t debug_ = 0, ArrayXd* res_ptr = nullptr)
{
    if (c.size() != A.rows())
        error("%s: c is missing or has wrong size", __func__);
    if (o && o->size() != A.rows())
        error("%s: o has wrong size", __func__);
    if (opt.exact_zero) {
        error("%s: exact_zero option not supported for sparse input", __func__);
    }
    const int K = static_cast<int>(A.cols());
    PoisLog1pRegSparseProblem P(A, y_sparse, c, o, opt);

    // Initialization
    if (b.size() != K) {
        double numer = (P.yvec.array() / P.cS.array() + 1.0).log().sum();
        if (o) {
            numer -= o->sum();
        }
        double denom = P.Anz.sum();
        double init_val = (denom > 1e-8) ? (numer / denom) : 1e-6;
        b = VectorXd::Constant(K, init_val);
    }

    if (debug_ > 0) {
        double f_cur = P.f(b);
        std::cout << "Initial b (" << b.size() << ", " << b.norm() << ") " << f_cur << std::endl;
    }

    double final_obj;
    if (opt.optim.tron.enabled) {
        final_obj = tron_solve(P, b, opt.optim, stats.optim, debug_);
    } else if (opt.optim.acg.enabled) {
        final_obj = acg_solve(P, b, opt.optim, stats.optim, debug_);
    } else {
        final_obj = newton_solve(P, b, opt.optim, stats.optim, debug_);
    }

    if (debug_ > 0) {
        std::cout << "Finished in " << stats.optim.niters << " iterations" << std::endl;
        std::cout << "Final b (" << b.size() << ", " << b.norm() << ") " << final_obj << std::endl;
    }

    stats.pll = (-final_obj
        - P.yvec.unaryExpr([](double n){ return log_factorial(n); }).sum()
        )/ P.yvec.sum();
    // if (opt.se_flag != 0) {
        // TODO
    // }
    if (opt.compute_residual) {
        ArrayXd res = P.residual(b);
        stats.residual = res.sum() / A.rows();
        if (res_ptr != nullptr && res_ptr->size() == res.size())
            *res_ptr += res;
    }

    return final_obj;
}

template <typename SparseDerived>
double pois_log1p_mle(
    const RowMajorMatrixXd& A, const Eigen::SparseMatrixBase<SparseDerived>& y_sparse, double c,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_ = 0, ArrayXd* res_ptr = nullptr)
{
    if (c <= 0.0) error("%s: c must be positive", __func__);
    VectorXd cvec = VectorXd::Constant((int)A.rows(), c);
    return pois_log1p_mle(A, y_sparse, cvec, nullptr, opt, b, stats, debug_, res_ptr);
}
