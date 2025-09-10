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

// ---------------- Problem Definition for pois_log1p_mle_exact --------------
class PoisRegExactProblem {
public:
    const RowMajorMatrixXd& A;
    const Document& y;
    const VectorXd& c;
    const VectorXd* o;
    const MLEOptions& opt;
    const bool has_offset;
    bool nonnegative;
    VectorXd yvec;

    PoisRegExactProblem(const RowMajorMatrixXd& A_, const Document& y_,
                        const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_), y(y_), c(c_), o(o_), opt(opt_), has_offset(o != nullptr)
    {
        yvec = VectorXd::Zero(A.rows());
        for (size_t j = 0; j < y.ids.size(); ++j) {
            yvec[y.ids[j]] = y.cnts[j];
        }
        double min_lower_bound = 0;
        if (opt.b_min) min_lower_bound = opt.b_min->minCoeff();
        nonnegative = min_lower_bound >= 0.0;
        if (has_offset) {
            nonnegative = nonnegative && (o->minCoeff() >= 0.0);
        }
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
            VectorXd Av = A * v;
            return A.transpose() * (w * Av.array()).matrix();
        };
    }
};

// ---------------- Problem Definition for pois_log1p_mle (sparse) ----------
class PoisRegSparseProblem {
public:
    // Pointers to original data
    const RowMajorMatrixXd& A;
    const Document& y;
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

    PoisRegSparseProblem(const RowMajorMatrixXd& A_, const Document& y_,
                         const VectorXd& c_, const VectorXd* o_, const MLEOptions& opt_)
        : A(A_), y(y_), c(c_), o(o_), opt(opt_),
          has_offset(o != nullptr),
          n(y.ids.size()),
          Anz(n, A.cols()),
          yvec(y.cnts.data(), n),
          cS(n)
    {
        const int K = static_cast<int>(A.cols());

        // Build submatrices for non-zero counts
        for (size_t j = 0; j < n; ++j) {
            Anz.row(j) = A.row(y.ids[j]);
        }
        AsqnzT = Anz.array().square().matrix().transpose();

        if (has_offset) oS.resize(n);
        for (size_t t = 0; t < n; ++t) {
            const int i = y.ids[t];
            cS[t] = c[i];
            if (has_offset) oS[t] = (*o)[i];
        }

        // Precompute for zero-count contributions
        VectorXd zmask = c;
        for (size_t j = 0; j < n; ++j) {
            zmask[y.ids[j]] = 0.0;
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
        if (opt.b_min) min_lower_bound = opt.b_min->minCoeff();
        nonnegative = min_lower_bound >= 0.0;
        if (has_offset) {
            nonnegative = nonnegative && (o->minCoeff() >= 0.0);
        }
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
};


double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A,
    const Document& y,
    const VectorXd* c,    // length N, scaling factors
    const VectorXd* o,    // length N, offset
    const MLEOptions& opt,
    VectorXd& b, MLEStats& stats, int32_t debug_ = 0);

// With 2nd order approximation for zero-set speed up
double pois_log1p_mle(
    const RowMajorMatrixXd& A,
    const Document& y,
    const VectorXd* c,
    const VectorXd* o,
    const MLEOptions& opt,
    VectorXd& b, MLEStats& stats, int32_t debug_ = 0);

// Wrappers for using a constant scaling factor
double pois_log1p_mle(
    const RowMajorMatrixXd& A, const Document& y, double c,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_ = 0);

double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A, const Document& y, double c,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_ = 0);

// Compute Fisher and/or robust SE for Poisson with log1p link.
void pois_log1p_compute_se(
    const RowMajorMatrixXd& X, const Document& y, const VectorXd* c,
    const VectorXd* o, const MLEOptions& opt, VectorXd& b, MLEStats& stats);
