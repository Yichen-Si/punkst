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

#include <Eigen/Core>
#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Eigen::ArrayXd;

struct LineSearchOptions {
    bool   enabled        = false;
    double beta           = 0.5;    // step shrink factor
    double c1             = 1e-4;   // Armijo constant
    int    max_backtracks = 20;
};

struct ACGOptions {
    bool   enabled    = false;
    double L0         = 1.0;     // initial Lipschitz guess (ignored if alpha>0)
    double bt_inc     = 2.0;     // multiply L by this on backtracking
    bool   monotone   = true;    // monotone FISTA
    bool   restart    = true;    // gradient-based restart
};

struct TrustRegionOptions {
    bool   enabled      = false;
    double delta_init   = 1.0;     // initial trust radius
    double delta_max    = 1e6;     // max trust radius
    double eta          = 1e-4;    // acceptance threshold
    double cg_tol       = 1e-4;    // relative CG tolerance
    int    cg_max_iter  = -1;      // default: 2*K
};

struct MLEOptions {
    bool  exact_zero = false;
    int    max_iters = 50;
    double tol       = 1e-6;
    double alpha     = 1.0;     // initial step size
    double ridge     = 1e-12;
    double eps       = 1e-12;
    bool use_agd     = false;
    LineSearchOptions ls{};
    ACGOptions acg{};
    TrustRegionOptions tron{};
};

struct MLEStats {
    int niters = 0;
    double obj, diff_obj;
    double diff_b, rel_diff_b;
};

inline double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A, // N x K
    const Document& y, double c, const MLEOptions& opt,
    VectorXd& b, MLEStats& stats, int32_t debug_ = 0);

/// argmax Σ_i [ y_i log λ_i − λ_i ], log(1+λ_i/c) = (A b)_i (A, b ≥ 0)
/// With 2nd order approximation for zero-set speed up
inline double pois_log1p_mle(
    const RowMajorMatrixXd& A, // N x K
    const Document& y, double c,
    const MLEOptions& opt,
    VectorXd& b, MLEStats& stats, int32_t debug_ = 0)
{
    if (c <= 0.0) error("%s: c must be positive", __func__);
    if (opt.exact_zero) {
        return pois_log1p_mle_exact(A, y, c, opt, b, stats, debug_);
    }

    const size_t n = y.ids.size();
    const int N = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());
    // Build a submatrix of A
    RowMajorMatrixXd Anz(n, K);
    for (size_t j = 0; j < n; ++j) {
        Anz.row(j) = A.row(y.ids[j]);
    }
    MatrixXd Asqnz = Anz.array().square().matrix().transpose(); // K x n
    Eigen::Map<const Eigen::VectorXd> yvec(y.cnts.data(), n);

    // Precompute zak, zakl
    VectorXd zmask = VectorXd::Ones(N);
    for (size_t j = 0; j < y.ids.size(); ++j) {
        zmask[y.ids[j]] = 0.0;
    }
    const VectorXd zak  = A.transpose() * zmask;                     // K
    const MatrixXd zakl = A.transpose() * (zmask.asDiagonal() * A);  // KxK
    const VectorXd dZ   = zakl.diagonal();                           // K

    // Objective f(b) = -logl(b)
    auto f_objective = [&](const VectorXd& bvec) -> double {
        // Nonzero part: exact
        const ArrayXd lam = ((Anz * bvec).array().exp() - 1.0) * c;
        const ArrayXd lam_pos = lam.max(1e-8);
        const double sumNZ = yvec.dot(lam_pos.log().matrix()) - lam.sum();
        // Zero set: 2nd order Taylor
        double sumZ = -c * (bvec.dot(zak) + 0.5 * bvec.dot(zakl * bvec));
        return -(sumNZ + sumZ);
    };

    // Hessian-vector product H·v
    auto make_Hv = [&](const ArrayXd& w) {
        return [&](const VectorXd& v) -> VectorXd {
            // NZ: Anzᵀ( w ∘ (Anz v) )
            VectorXd Av = Anz * v;
            VectorXd Hv_nz = Anz.transpose() * (w * Av.array()).matrix();
            // Z: c * zakl * v
            VectorXd Hv_z  = c * (zakl * v);
            return Hv_nz + Hv_z;
        };
    };

    if (b.size() != K) {
        double numer = (yvec.array() / c + 1.0).log().sum();
        double denom = Anz.sum();
        double init_val = (denom > 1e-8) ? (numer / denom) : 1e-6;
        b = VectorXd::Constant(K, init_val);
    }
    double f_cur = f_objective(b);

if (debug_ > 0) {
    std::cout << "Initial b (" << b.size() << ", " << b.norm() << ") " << f_cur << std::endl;
}

    VectorXd g(K), q(K);
    int obj_stable_count = 0;
    int it = 0;
    double f_diff, b_diff, f_rel, b_rel;

    // ---------------- Trust region ----------------
    if (opt.tron.enabled) {
        double delta = opt.tron.delta_init;
        for (it = 0; it < opt.max_iters; ++it) {
            // NZ exact parts
            const ArrayXd u   = (Anz * b).array().exp(); // n
            const ArrayXd um1 = (u - 1.0).max(opt.eps);
            const ArrayXd t   = c * u - yvec.array() * (u / um1);
            const ArrayXd w   = yvec.array() * (u / (um1 * um1)) + c * u;

            // Gradient & diag(H)
            g.noalias() = Anz.transpose() * t.matrix() + c * (zak + zakl * b);
            q.noalias() = Asqnz * w.matrix() + c * dZ;
            q = q.array().max(opt.ridge).matrix();

            // Free set
            Eigen::Array<bool, Eigen::Dynamic, 1> free_mask =
                (b.array() > 0.0) || (g.array() < 0.0);

            // CG init
            auto Hv_op = make_Hv(w);
            VectorXd d = VectorXd::Zero(K), r = VectorXd::Zero(K),
                     z = VectorXd::Zero(K), p = VectorXd::Zero(K);

            double gF2 = 0.0;
            for (int k = 0; k < K; ++k)
                if (free_mask[k]) {
                    r[k] = g[k];
                    z[k] = r[k] / q[k];
                    p[k] = -z[k];
                    gF2 += g[k]*g[k];
                }
            double rz = r.dot(z);
            const double cg_tol = std::max(opt.eps, opt.tron.cg_tol * std::sqrt(rz));
            const int cg_maxit = (opt.tron.cg_max_iter > 0) ? opt.tron.cg_max_iter : (2 * K);

            bool hit_boundary = false;
            for (int itcg = 0; itcg < cg_maxit; ++itcg) {
                if (std::sqrt(rz) <= cg_tol) break;
                VectorXd p_masked = VectorXd::Zero(K);
                for (int j = 0; j < K; ++j) if (free_mask[j]) p_masked[j] = p[j];
                VectorXd Hp = Hv_op(p_masked);
                for (int j = 0; j < K; ++j) if (!free_mask[j]) Hp[j] = 0.0;

                const double pHp = p.dot(Hp);
                if (pHp <= 0.0) {
                    const double p2 = p.squaredNorm(), dTp = d.dot(p);
                    const double tau = (-dTp + std::sqrt(std::max(0.0, dTp*dTp + p2*(delta*delta - d.squaredNorm())))) / std::max(1e-32, p2);
                    d += tau * p; hit_boundary = true; break;
                }
                const double alpha_cg = rz / std::max(1e-32, pHp);
                VectorXd d_next = d + alpha_cg * p;
                if (d_next.norm() >= delta) {
                    const double p2 = p.squaredNorm(), dTp = d.dot(p), rad2 = delta*delta;
                    const double under = dTp*dTp + p2*(rad2 - d.squaredNorm());
                    const double tau   = (-dTp + std::sqrt(std::max(0.0, under))) / std::max(1e-32, p2);
                    d += tau * p; hit_boundary = true; break;
                }
                d = std::move(d_next);
                r += alpha_cg * Hp;
                if (r.norm() <= cg_tol) break;
                VectorXd z_next = VectorXd::Zero(K);
                for (int j = 0; j < K; ++j) if (free_mask[j]) z_next[j] = r[j] / q[j];
                const double rz_next = r.dot(z_next);
                const double beta = rz_next / std::max(1e-32, rz);
                p = -z_next + beta * p;
                z.swap(z_next);
                rz = rz_next;
            }

            VectorXd b_trial = (b + d).cwiseMax(0.0);
            VectorXd d_proj  = b_trial - b;

            VectorXd Hdproj = make_Hv(w)(d_proj);
            const double m_pred   = g.dot(d_proj) + 0.5 * d_proj.dot(Hdproj);
            const double pred_red = -m_pred;

            const double f_new = f_objective(b_trial);
            const double act_red = f_cur - f_new;
            const double rho = (pred_red > 0.0) ? (act_red / pred_red) : -std::numeric_limits<double>::infinity();

            if (rho < 0.25)       delta *= 0.25;
            else if (rho > 0.75 && hit_boundary) delta = std::min(2.0 * delta, opt.tron.delta_max);

            f_diff = std::abs(f_new - f_cur);
            f_rel = f_diff / (f_new + f_cur) * 2;
            b_diff = d_proj.norm();
            b_rel = b_diff / std::max(1.0, b.norm());

            if (rho > opt.tron.eta) {
                b.swap(b_trial);
                f_cur = f_new;
            }

            if (debug_ > 0) {
                std::cout << "TRON it=" << it << ", obj=" << f_new << ", delta=" << delta << ", diff=" << b_rel << std::endl;
            }

            const double gF_norm = std::sqrt(gF2);
            if (b_rel <= opt.tol) break;
            if (gF_norm <= opt.tol) break;
            if (delta < 1e-12) break;
        }
    }
    // ---------------- Monotone FISTA ----------------
    else if (opt.acg.enabled) {
        VectorXd b_prev = b;
        VectorXd yk = b;
        double t = 1.0;
        double L = std::max(1e-12, opt.acg.L0);
        VectorXd gk(K);
        auto grad_at = [&](const VectorXd& bvec, VectorXd& gout) {
            const ArrayXd u   = (Anz * bvec).array().exp();
            const ArrayXd um1 = (u - 1.0).max(opt.eps);
            const ArrayXd t   = c * u - yvec.array() * (u / um1);
            gout.noalias() = Anz.transpose() * t.matrix() + c * (zak + zakl * bvec);
        };
        for (it = 0; it < opt.max_iters; ++it) {
            grad_at(yk, gk);
            const double f_yk = f_objective(yk);
            // Backtracking from yk
            VectorXd b_trial, diff;
            double f_new = 0.0;
            while (true) {
                b_trial = (yk - (1.0 / L) * gk).cwiseMax(0.0);
                diff = b_trial - yk;
                const double rhs = f_yk + gk.dot(diff) + 0.5 * L * diff.squaredNorm();
                f_new = f_objective(b_trial);
                if (f_new <= rhs) break;
                L *= opt.acg.bt_inc;
            }

            // Monotone safeguard: fallback from b if needed
            if (opt.acg.monotone && f_new > f_cur) {
                grad_at(b, gk);
                const double f_b = f_cur;
                while (true) {
                    b_trial = (b - (1.0 / L) * gk).cwiseMax(0.0);
                    diff = b_trial - b;
                    const double rhs = f_b + gk.dot(diff) + 0.5 * L * diff.squaredNorm();
                    f_new = f_objective(b_trial);
                    if (f_new <= rhs) break;
                    L *= opt.acg.bt_inc;
                }
                // restart momentum
                t = 1.0;
                yk = b;
            }

            // Accept
            b_prev.swap(b);
            b.swap(b_trial);

            // Nesterov momentum
            const double t_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
            VectorXd y_next = b + ((t - 1.0) / t_next) * (b - b_prev);

            if (opt.acg.restart) {
                // gradient-based restart
                if ((b - b_prev).dot(y_next - b) > 0.0) {
                    t = 1.0;
                    yk = b;
                } else {
                    t = t_next;
                    yk.swap(y_next);
                }
            } else {
                t = t_next;
                yk.swap(y_next);
            }

            f_diff = std::abs(f_new - f_cur);
            f_rel = f_diff / (f_new + f_cur) * 2;
            b_diff = (b - b_prev).norm();
            b_rel = b_diff / std::max(1.0, b_prev.norm());

            if (debug_ > 0) {
                double obj = f_objective(b);
                std::cout << "ACG it=" << it << ", obj=" << f_new << ", L=" << L << ", diff=" << b_rel << std::endl;
            }

            if (f_rel < opt.tol) {
                obj_stable_count++;
                if (obj_stable_count >= 3) break;
            } else {
                obj_stable_count = 0;
            }
            if (b_rel < opt.tol) break;
            f_cur = f_new;
        }
    }
    // ---------------- Diagonal Newton + LS ----------------
    else {
        for (it = 0; it < opt.max_iters; ++it) {
            // Nonzero part
            const ArrayXd u   = (Anz * b).array().exp(); // n
            const ArrayXd um1 = (u - 1.0).max(opt.eps);  // n
            const ArrayXd t = c * u - yvec.array() * (u / um1);
            const ArrayXd w = yvec.array() * (u / (um1 * um1)) + c * u;
            // Gradient
            g.noalias() = Anz.transpose() * t.matrix() + c * (zak + zakl * b);
            // Diag(H)
            q.noalias() = Asqnz * w.matrix() + c * dZ;
            q = q.array().max(opt.ridge).matrix();
            const VectorXd dir = (-g.array() / q.array()).matrix();
            double alpha = opt.alpha;
            VectorXd b_trial = (b + alpha * dir).cwiseMax(0.0);
            double f_new = f_objective(b_trial);
            if (opt.ls.enabled) { // optional line search
                int bt = 0;
                for (;; ++bt) {
                    const double armijo_rhs = f_cur + opt.ls.c1 * g.dot(b_trial - b);
                    if (f_new <= armijo_rhs || bt >= opt.ls.max_backtracks) break;
                    alpha *= opt.ls.beta;
                    b_trial = (b + alpha * dir).cwiseMax(0.0);
                    f_new = f_objective(b_trial);
                }
            }
            // Convergence
            f_diff = std::abs(f_new - f_cur);
            f_rel = f_diff / (f_new + f_cur) * 2;
            b_diff = (b_trial - b).norm();
            b_rel = b_diff / std::max(1.0, b.norm());

            b.swap(b_trial);

            if (debug_ > 0) {
                std::cout << "GD(LS) it=" << it << ", obj=" << f_new << ", alpha=" << alpha << ", diff=" << b_rel << std::endl;
            }

            if (f_rel < opt.tol) {
                obj_stable_count++;
                if (obj_stable_count >= 3) break;
            } else {
                obj_stable_count = 0;
            }
            if (b_rel < opt.tol) {
                break;
            }
            f_cur = f_new;
        }
    }
if (debug_ > 0) {
    std::cout << "Finished in " << it << " iterations" << std::endl;
    std::cout << "Final b (" << b.size() << ", " << b.norm() << ") " << f_cur << std::endl;
}
    stats.niters = it + 1;
    stats.obj = f_cur;
    stats.diff_obj = f_diff;
    stats.rel_diff_b = b_rel;
    stats.diff_b = b_diff;

    return f_cur;
}










inline double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A, // N x K
    const Document& y, double c, const MLEOptions& opt,
    VectorXd& b, MLEStats& stats, int32_t debug_)
{
    if (c <= 0.0) throw std::invalid_argument("c must be positive");
    if (!opt.exact_zero) {
        return pois_log1p_mle(A, y, c, opt, b, stats, debug_);
    }

    const size_t n = y.ids.size();
    const int N = static_cast<int>(A.rows());
    const int K = static_cast<int>(A.cols());

    VectorXd yvec = VectorXd::Zero(N);
    for (size_t j = 0; j < n; ++j) {
        yvec[y.ids[j]] = y.cnts[j];
    }

    // Objective f(b) = -ℓ(b) (exact over all rows)
    auto f_objective = [&](const VectorXd& bvec) -> double {
        const ArrayXd  lam = ((A * bvec).array().exp() - 1.0) * c;
        const ArrayXd  lam_pos = lam.max(1e-8);
        return -(yvec.dot(lam_pos.log().matrix()) - lam.sum());
    };

    // Gradient at b (exact): g = Aᵀ t,  with t_i = c u_i − y_i u_i/(u_i−1)
    auto grad_at = [&](const VectorXd& bvec, VectorXd& gout, ArrayXd* w_out=nullptr) {
        const VectorXd eta = A * bvec;
        const ArrayXd  u   = eta.array().exp();
        const ArrayXd  um1 = (u - 1.0).max(opt.eps);

        ArrayXd t = c * u - yvec.array() * (u / um1);
        gout.noalias() = A.transpose() * t.matrix();

        if (w_out) {
            ArrayXd w = c * u + yvec.array() * (u / (um1 * um1));
            *w_out = std::move(w);
        }
    };

    // Hv operator (exact): Hv = Aᵀ( w ∘ (A v) ), with w from current point
    auto make_Hv = [&](const ArrayXd& w) {
        return [&](const VectorXd& v) -> VectorXd {
            VectorXd Av = A * v;                            // N
            return A.transpose() * (w * Av.array()).matrix();
        };
    };

    // Initialization
    if (b.size() != K) {
        double numer = (yvec.array() / c + 1.0).log().sum();
        VectorXd A_row_sum = A.rowwise().sum();
        double denom = 0.;
        for (auto i : y.ids) {
            denom += A_row_sum[i];
        }
        double init_val = (denom > 1e-8) ? (numer / denom) : 1e-6;
        b = VectorXd::Constant(K, init_val);
    }
    double f_cur = f_objective(b);

if (debug_ > 0) {
    std::cout << "Initial b (" << b.size() << ", " << b.norm() << ") " << f_cur << std::endl;
}

    // Buffers
    VectorXd g(K), q(K);
    int obj_stable_count = 0;
    int it = 0;
    double f_diff, b_diff, f_rel, b_rel;

    // ---------------- Trust region ----------------
    if (opt.tron.enabled) {
        double delta = opt.tron.delta_init;
        for (it = 0; it < opt.max_iters; ++it) {
            ArrayXd wN;
            grad_at(b, g, &wN);  // get g and w

            // Build preconditioner diag(H) exactly: q_k = Σ_i w_i A_{ik}^2
            q = (A.array().square().colwise() * wN).colwise().sum().transpose();
            q = q.array().max(opt.ridge).matrix();

            // Free set
            Eigen::Array<bool, Eigen::Dynamic, 1> free_mask =
                (b.array() > 0.0) || (g.array() < 0.0);

            // CG init
            auto Hv_op = make_Hv(wN);
            VectorXd d = VectorXd::Zero(K), r = VectorXd::Zero(K),
                     z = VectorXd::Zero(K), p = VectorXd::Zero(K);

            double gF2 = 0.0;
            for (int k = 0; k < K; ++k)
                if (free_mask[k]) {
                    r[k] = g[k];
                    z[k] = r[k] / q[k];
                    p[k] = -z[k];
                    gF2 += g[k]*g[k];
                }
            double rz = r.dot(z);
            const double cg_tol = std::max(opt.eps, opt.tron.cg_tol * std::sqrt(rz));
            const int cg_maxit = (opt.tron.cg_max_iter > 0) ? opt.tron.cg_max_iter : (2 * K);

            bool hit_boundary = false;
            for (int itcg = 0; itcg < cg_maxit; ++itcg) {
                if (std::sqrt(rz) <= cg_tol) break;
                VectorXd p_masked = VectorXd::Zero(K);
                for (int j = 0; j < K; ++j) if (free_mask[j]) p_masked[j] = p[j];
                VectorXd Hp = Hv_op(p_masked);
                for (int j = 0; j < K; ++j) if (!free_mask[j]) Hp[j] = 0.0;

                const double pHp = p.dot(Hp);
                if (pHp <= 0.0) {
                    const double p2 = p.squaredNorm(), dTp = d.dot(p);
                    const double tau = (-dTp + std::sqrt(std::max(0.0, dTp*dTp + p2*(delta*delta - d.squaredNorm())))) / std::max(1e-32, p2);
                    d += tau * p; hit_boundary = true; break;
                }
                const double alpha_cg = rz / std::max(1e-32, pHp);
                VectorXd d_next = d + alpha_cg * p;
                if (d_next.norm() >= delta) {
                    const double p2 = p.squaredNorm(), dTp = d.dot(p), rad2 = delta*delta;
                    const double under = dTp*dTp + p2*(rad2 - d.squaredNorm());
                    const double tau   = (-dTp + std::sqrt(std::max(0.0, under))) / std::max(1e-32, p2);
                    d += tau * p; hit_boundary = true; break;
                }
                d = std::move(d_next);
                r += alpha_cg * Hp;
                if (r.norm() <= cg_tol) break;
                VectorXd z_next = VectorXd::Zero(K);
                for (int j = 0; j < K; ++j) if (free_mask[j]) z_next[j] = r[j] / q[j];
                const double rz_next = r.dot(z_next);
                const double beta = rz_next / std::max(1e-32, rz);
                p = -z_next + beta * p;
                z.swap(z_next);
                rz = rz_next;
            }

            VectorXd b_trial = (b + d).cwiseMax(0.0);
            VectorXd d_proj  = b_trial - b;

            // Predicted vs actual reduction
            VectorXd Hdproj = make_Hv(wN)(d_proj);
            const double m_pred   = g.dot(d_proj) + 0.5 * d_proj.dot(Hdproj);
            const double pred_red = -m_pred;
            const double f_new = f_objective(b_trial);
            const double act_red = f_cur - f_new;
            const double rho = (pred_red > 0.0) ? (act_red / pred_red) : -std::numeric_limits<double>::infinity();

            if (rho < 0.25)       delta *= 0.25;
            else if (rho > 0.75 && hit_boundary) delta = std::min(2.0 * delta, opt.tron.delta_max);

            f_diff = std::abs(f_new - f_cur);
            f_rel = f_diff / (f_new + f_cur) * 2;
            b_diff = d_proj.norm();
            b_rel = b_diff / std::max(1.0, b.norm());

            if (rho > opt.tron.eta) {
                b.swap(b_trial);
                f_cur = f_new;
            }

            if (debug_ > 0) {
                std::cout << "TRON it=" << it << ", obj=" << f_new << ", delta=" << delta << ", diff=" << b_rel << std::endl;
            }

            const double gF_norm = std::sqrt(gF2);
            if (b_rel <= opt.tol) break;
            if (gF_norm <= opt.tol) break;
            if (delta < 1e-12) break;
        }
    }
    // ---------------- Monotone FISTA ----------------
    else if (opt.acg.enabled) {
        VectorXd b_prev = b, yk = b;
        double t = 1.0;
        double L = std::max(1e-12, opt.acg.L0);
        VectorXd gk(K);

        for (it = 0; it < opt.max_iters; ++it) {
            grad_at(yk, gk, nullptr);
            const double f_yk = f_objective(yk);

            // Backtracking from yk
            VectorXd b_trial, diff;
            double f_new = 0.0;
            while (true) {
                b_trial = (yk - (1.0 / L) * gk).cwiseMax(0.0);
                diff = b_trial - yk;
                const double rhs = f_yk + gk.dot(diff) + 0.5 * L * diff.squaredNorm();
                f_new = f_objective(b_trial);
                if (f_new <= rhs) break;
                L *= opt.acg.bt_inc;
            }

            // Monotone safeguard
            if (opt.acg.monotone && f_new > f_cur) {
                grad_at(b, gk, nullptr);
                const double f_b = f_cur;
                while (true) {
                    b_trial = (b - (1.0 / L) * gk).cwiseMax(0.0);
                    diff = b_trial - b;
                    const double rhs = f_b + gk.dot(diff) + 0.5 * L * diff.squaredNorm();
                    f_new = f_objective(b_trial);
                    if (f_new <= rhs) break;
                    L *= opt.acg.bt_inc;
                }
                t = 1.0;
                yk = b;
            }

            b_prev.swap(b);
            b.swap(b_trial);

            const double t_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
            VectorXd y_next = b + ((t - 1.0) / t_next) * (b - b_prev);

            if (opt.acg.restart) {
                if ((b - b_prev).dot(y_next - b) > 0.0) {
                    t = 1.0;
                    yk = b;
                } else {
                    t = t_next;
                    yk.swap(y_next);
                }
            } else {
                t = t_next;
                yk.swap(y_next);
            }

            f_diff = std::abs(f_new - f_cur);
            f_rel = f_diff / (f_new + f_cur) * 2;
            b_diff = (b - b_prev).norm();
            b_rel = b_diff / std::max(1.0, b_prev.norm());

            if (debug_ > 0) {
                std::cout << "ACG it=" << it << ", obj=" << f_new << ", L=" << L << ", diff=" << b_rel << std::endl;
            }

            if (f_rel < opt.tol) {
                obj_stable_count++;
                if (obj_stable_count >= 3) break;
            } else {
                obj_stable_count = 0;
            }
            if (b_rel < opt.tol) break;
            f_cur = f_new;
        }
    }
    // ---------------- Diagonal Newton + LS ----------------
    else {
        for (it = 0; it < opt.max_iters; ++it) {
            ArrayXd wN;
            grad_at(b, g, &wN); // g and wN (H diag weights)

            q = (A.array().square().colwise() * wN).colwise().sum().transpose();
            q = q.array().max(opt.ridge).matrix();

            const VectorXd dir = (-g.array() / q.array()).matrix();
            double alpha = opt.alpha;
            VectorXd b_trial = (b + alpha * dir).cwiseMax(0.0);
            double f_new = f_objective(b_trial);
            if (opt.ls.enabled) {
                int bt = 0;
                for (;; ++bt) {
                    const double rhs = f_cur + opt.ls.c1 * g.dot(b_trial - b);
                    if (f_new <= rhs || bt >= opt.ls.max_backtracks) break;
                    alpha *= opt.ls.beta;
                    b_trial = (b + alpha * dir).cwiseMax(0.0);
                    f_new = f_objective(b_trial);
                }
            }
            // Convergence
            f_diff = std::abs(f_new - f_cur);
            f_rel = f_diff / (f_new + f_cur) * 2;
            b_diff = (b_trial - b).norm();
            b_rel = b_diff / std::max(1.0, b.norm());

            b.swap(b_trial);

            if (debug_ > 0) {
                double obj = f_objective(b);
                std::cout << "GD(LS) it=" << it << ", obj=" << obj << ", alpha=" << alpha << ", diff=" << b_rel << std::endl;
            }

            if (f_rel < opt.tol) {
                obj_stable_count++;
                if (obj_stable_count >= 3) break;
            } else {
                obj_stable_count = 0;
            }
            if (b_rel < opt.tol) {
                break;
            }
            f_cur = f_new;
        }
    }

    stats.niters = it + 1;
    stats.obj = f_cur;
    stats.diff_obj = f_diff;
    stats.rel_diff_b = b_rel;
    stats.diff_b = b_diff;

    return f_cur;
}
