#include "poisreg.hpp"

// Exact

void PoisRegExactProblem::eval(const VectorXd& bvec, double* f_out,
    VectorXd* g_out, VectorXd* q_out, ArrayXd* w_out) const
{
    if (!nonnegative) {
        eval_safe(bvec, f_out, g_out, q_out, w_out);
        return;
    }
    ArrayXd u = (A * bvec).array();
    if (has_offset) u += o->array();
    u = u.exp(); // exp(Ab+o)
    ArrayXd um1 = (u - 1.0).max(opt.optim.eps); // \lambda / c

    if (f_out) {
        ArrayXd lam = um1 * c.array();
        *f_out = -(yvec.dot(lam.log().matrix()) - lam.sum());
    }
    if (!(g_out || q_out || w_out)) return;

    if (g_out) {
        ArrayXd t = c.array() * u - yvec.array() * (u / um1);
        g_out->noalias() = A.transpose() * t.matrix();
    }
    if (q_out) {
        ArrayXd w = c.array() * u + yvec.array() * (u / (um1 * um1));
        *q_out = ((A.array().square().colwise() * w).colwise().sum().transpose()).array().max(opt.ridge).matrix();
        if (w_out) *w_out = std::move(w);
    }
}

void PoisRegExactProblem::eval_safe(const VectorXd& bvec, double* f_out,
    VectorXd* g_out, VectorXd* q_out, ArrayXd* w_out) const
{
    ArrayXd u = (A * bvec).array();
    if (has_offset) u += o->array();
    u = u.exp(); // exp(Ab+o)
    ArrayXd s = u - 1.0;
    double eps = opt.optim.eps;
    double tau = opt.soft_tau;

    ArrayXd z = (s - eps) / tau;
    // softplus(s) = log1p(exp(z))
    ArrayXd softplus = (z > 0.0).select( z + (-z).exp().log1p(), z.exp().log1p() );

    ArrayXd phi   = eps + tau * softplus;            // > eps
    ArrayXd sig   = 1.0 / (1.0 + (-z).exp());        // expit(z) in (0,1)
    ArrayXd phi_p = sig;                             // φ'(s)
    ArrayXd phi_pp = (sig * (1.0 - sig)) / tau;      // φ''(s)

    // f(b)
    if (f_out) {
      ArrayXd lam = phi * c.array();
      *f_out = -(yvec.dot(lam.log().matrix()) - lam.sum());
    }
    if (!(g_out || q_out || w_out)) return;

    // gradient wrt η
    ArrayXd C = c.array() - yvec.array() / phi;      // C_i = c_i - y_i/φ_i
    ArrayXd dfdeta = u * phi_p * C;

    if (g_out) {
      g_out->noalias() = A.transpose() * dfdeta.matrix();
    }

    if (q_out) {
      // w = ∂²f/∂η² (per-row curvature)
      ArrayXd w = (phi_pp * u.square() + phi_p * u) * C
                + (yvec.array() * (phi_p.square() * u.square() / phi.square()));
      // diag(H) = (A.^2)^T w
      *q_out = ((A.array().square().colwise() * w).colwise().sum()
               .transpose()).array().max(opt.ridge).matrix();
      if (w_out) *w_out = std::move(w);
    }
}

ArrayXd PoisRegExactProblem::residual(const VectorXd& bvec) const {
    ArrayXd mu = (A * bvec).array();
    if (o) mu += o->array();
    mu = (mu.exp() - 1.0).max(0.) * c.array();
    return (yvec.array() - mu).square() / mu.max(1e-5);
}

ArrayXd pois_log1p_residual(
    const RowMajorMatrixXd& X, const VectorXd& y, const VectorXd& c,
    const VectorXd* o, const Eigen::Ref<const VectorXd>& b) {
    ArrayXd mu = (X * b).array();
    if (o) mu += o->array();
    mu = (mu.exp() - 1.0).max(0.) * c.array();
    return (y.array() - mu).square() / mu.max(1e-5);
}

double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A, const Document& y, double c,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_, ArrayXd* res_ptr)
{
    if (c <= 0.0) error("%s: c must be positive", __func__);
    VectorXd cvec = VectorXd::Constant((int)A.rows(), c);
    return pois_log1p_mle_exact(A, y, cvec, nullptr, opt, b, stats, debug_, res_ptr);
}

double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A, const Document& y,
    const VectorXd& c, const VectorXd* o,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_, ArrayXd* res_ptr)
{
    if (c.size() != A.rows())
        error("%s: c is missing or has wrong size", __func__);
    if (o && o->size() != A.rows())
        error("%s: o has wrong size", __func__);
    if (!opt.exact_zero) {
        return pois_log1p_mle(A, y, c, o, opt, b, stats, debug_, res_ptr);
    }

    const int K = static_cast<int>(A.cols());
    PoisRegExactProblem P(A, y, c, o, opt);

    // Initialization
    if (b.size() != K) {
        double numer = (P.yvec.array() / c.array() + 1.0).log().sum();
        if (o) {
            numer -= o->sum();
        }
        VectorXd A_row_sum = A.rowwise().sum();
        double denom = 0.;
        for (auto i : y.ids) {
            denom += A_row_sum[i];
        }
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
        ) / P.yvec.sum();
    if (opt.se_flag != 0) {
        pois_log1p_compute_se(A, P.yvec, c, o, opt, b, stats);
    }
    if (opt.compute_residual) {
        ArrayXd res = P.residual(b);
        stats.residual = res.sum() / A.rows();
        if (res_ptr != nullptr && res_ptr->size() == res.size())
            *res_ptr += res;
    }

    return final_obj;
}

// Approximate

void PoisRegSparseProblem::eval(const VectorXd& bvec, double* f_out,
    VectorXd* g_out, VectorXd* q_out, ArrayXd* w_out) const
{
    if (!nonnegative) {
        eval_safe(bvec, f_out, g_out, q_out, w_out);
        return;
    }
    ArrayXd u = (Anz * bvec).array();
    if (has_offset) u += oS.array();
    u = u.exp();
    ArrayXd um1 = (u - 1.0).max(opt.optim.eps);

    if (f_out) {
        ArrayXd lam = cS.array() * um1;
        double sumNZ = yvec.dot(lam.log().matrix()) - lam.sum();
        double sumZ = -(bvec.dot(zak + zoak) + 0.5 * bvec.dot(zakl * bvec));
        *f_out = -(sumNZ + sumZ);
    }
    if (!(g_out || q_out || w_out)) return;

    if (g_out) {
        ArrayXd t = (cS.array() * u) - (yvec.array() * (u / um1));
        g_out->noalias() = Anz.transpose() * t.matrix();
        g_out->noalias() += (zak + zoak + zakl * bvec);
    }
    if (q_out) {
        ArrayXd w = (cS.array() * u) + (yvec.array() * (u / (um1 * um1)));
        q_out->noalias() = AsqnzT * w.matrix() + dZ;
        *q_out = q_out->array().max(opt.ridge).matrix();
        if (w_out) *w_out = std::move(w);
    }
}

void PoisRegSparseProblem::eval_safe(const VectorXd& bvec, double* f_out,
    VectorXd* g_out, VectorXd* q_out, ArrayXd* w_out) const
{
    ArrayXd u = (Anz * bvec).array();
    if (has_offset) u += oS.array();
    u = u.exp();
    ArrayXd s = u - 1.0;
    double eps = opt.optim.eps;
    double tau = opt.soft_tau;

    ArrayXd z = (s - eps) / tau;
    ArrayXd softplus = (z > 0.0).select( z + (-z).exp().log1p(), z.exp().log1p() );
    ArrayXd phi   = eps + tau * softplus;
    ArrayXd sig   = 1.0 / (1.0 + (-z).exp());
    ArrayXd phi_p = sig;
    ArrayXd phi_pp = (sig * (1.0 - sig)) / tau;

    if (f_out) {
        ArrayXd lam_nz = cS.array() * phi;
        double sumNZ = yvec.dot(lam_nz.log().matrix()) - lam_nz.sum();
        double sumZ = -(bvec.dot(zak + zoak) + 0.5 * bvec.dot(zakl * bvec));
        *f_out = -(sumNZ + sumZ);
    }
    if (!(g_out || q_out || w_out)) return;

    ArrayXd C = cS.array() - yvec.array() / phi;   // c - y/φ
    ArrayXd dfdeta = u * phi_p * C;

    if (g_out) {
        g_out->noalias() = Anz.transpose() * dfdeta.matrix();
        g_out->noalias() += (zak + zoak + zakl * bvec);
    }

    if (q_out) {
        ArrayXd w = (phi_pp * u.square() + phi_p * u) * C
                + (yvec.array() * (phi_p.square() * u.square() / phi.square()));
        q_out->noalias() = AsqnzT * w.matrix() + dZ;
        *q_out = q_out->array().max(opt.ridge).matrix();
        if (w_out) *w_out = std::move(w);
    }
}

ArrayXd PoisRegSparseProblem::residual(const VectorXd& bvec) const {
    ArrayXd mu = (A * bvec).array();
    if (o) mu += o->array();
    mu = ((mu.exp() - 1.0) * c.array()).max(0.);
    Eigen::Map<const Eigen::Array<uint32_t, Eigen::Dynamic, 1>> ids_map(y.ids.data(), n);
    Eigen::ArrayXd mu_nz = mu(ids_map.cast<Eigen::Index>());
    mu(ids_map.cast<Eigen::Index>()) = (yvec.array() - mu_nz).square() / mu_nz.max(1e-5);
    return mu;
}

ArrayXd pois_log1p_residual(
    const RowMajorMatrixXd& X, const Document& y,
    const VectorXd& c, const VectorXd* o, const Eigen::Ref<const VectorXd>& b) {
    ArrayXd mu = (X * b).array();
    if (o) mu += o->array();
    mu = (mu.exp() - 1.0).max(0.) * c.array();
    size_t n = y.ids.size();
    Eigen::Map<const Eigen::Array<uint32_t, Eigen::Dynamic, 1>> ids_map(y.ids.data(), n);
    Eigen::ArrayXd mu_nz = mu(ids_map.cast<Eigen::Index>());
    Eigen::Map<const VectorXd> yvec(y.cnts.data(), n);
    mu(ids_map.cast<Eigen::Index>()) = (yvec.array() - mu_nz).square() / mu_nz.max(1e-5);
    return mu;
}

double pois_log1p_mle(
    const RowMajorMatrixXd& A, const Document& y, double c,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_, ArrayXd* res_ptr)
{
    if (c <= 0.0) error("%s: c must be positive", __func__);
    VectorXd cvec = VectorXd::Constant((int)A.rows(), c);
    return pois_log1p_mle(A, y, cvec, nullptr, opt, b, stats, debug_, res_ptr);
}

double pois_log1p_mle(
    const RowMajorMatrixXd& A, const Document& y,
    const VectorXd& c, const VectorXd* o,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_, ArrayXd* res_ptr)
{
    if (c.size() != A.rows())
        error("%s: c is missing or has wrong size", __func__);
    if (o && o->size() != A.rows())
        error("%s: o has wrong size", __func__);
    if (opt.exact_zero) {
        return pois_log1p_mle_exact(A, y, c, o, opt, b, stats, debug_, res_ptr);
    }

    const int K = static_cast<int>(A.cols());
    PoisRegSparseProblem P(A, y, c, o, opt);

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
    if (opt.se_flag != 0) {
        pois_log1p_compute_se(A, y, c, o, opt, b, stats);
    }
    if (opt.compute_residual) {
        ArrayXd res = P.residual(b);
        stats.residual = res.sum() / A.rows();
        if (res_ptr != nullptr && res_ptr->size() == res.size())
            *res_ptr += res;
    }

    return final_obj;
}

void pois_log1p_compute_se(
    const RowMajorMatrixXd& X, const Document& y, const VectorXd& c,
    const VectorXd* o, const MLEOptions& opt, const Eigen::Ref<const VectorXd>& b, MLEStats& stats) {
    VectorXd y_dense;
    if (opt.se_flag & 0x2) {
        y_dense = y.to_dense(X.rows());
    }
    pois_log1p_compute_se(X, y_dense, c, o, opt, b, stats);
}

void pois_log1p_compute_se(
    const RowMajorMatrixXd& X, const VectorXd& y, const VectorXd& c,
    const VectorXd* o, const MLEOptions& opt, const Eigen::Ref<const VectorXd>& b, MLEStats& stats) {
    if (opt.se_flag == 0) return;
    const int n = static_cast<int>(X.rows());
    const int p = static_cast<int>(X.cols());
    // 1) η, μ, ∂μ/∂η
    ArrayXd eta = (X * b).array();
    if (o) eta += o->array();
    ArrayXd u   = eta.exp();                // e^{η}
    ArrayXd um1 = (u - 1.0).max(opt.optim.eps);   // (e^{η}-1)
    ArrayXd mu  = c.array() * um1;         // μ = c*(e^{η}-1)
    ArrayXd dmu = c.array() + mu;          // ∂μ/∂η = c*e^{η} = c + μ
    // 2) A = X'WX with W = (dμ/dη)^2 / V(μ) = (dmu^2) / μ
    ArrayXd w_fisher = (dmu.square() / mu.max(opt.optim.eps));
    // Scale X[i,:] by \sqrt(w_i)
    VectorXd sqrt_w = w_fisher.sqrt().matrix();
    RowMajorMatrixXd Xw = X;
    for (int j = 0; j < p; ++j) {
        Xw.col(j).array() *= sqrt_w.array();
    }
    MatrixXd XtWX = Xw.transpose() * Xw;
    XtWX.diagonal().array() += opt.ridge;
    // Invert
    Eigen::LDLT<MatrixXd> ldlt(XtWX);
    if (ldlt.info() != Eigen::Success) {
        error("%s: LDLT failed on XtWX", __func__);
    }
    MatrixXd XtWX_inv = ldlt.solve(MatrixXd::Identity(p, p));
    MatrixXd Vrob;
    if (opt.se_flag & 0x1) { // Fisher
        stats.se_fisher = XtWX_inv.diagonal().array().sqrt().matrix();
    } else {
        stats.se_fisher.resize(0);
    }
    if (opt.se_flag & 0x2) { // Sandwich robust
        // 3) B = X' diag(s^2) X with s = (y-μ)*(dμ/dη)/μ
        ArrayXd s = (y.array() - mu) * (dmu / mu.max(opt.optim.eps));
        if (opt.hc_type == 1) {
            if (n > p) {
                s *= std::sqrt(double(n) / double(n - p));
            }
        } else if (opt.hc_type >= 2) {
            Eigen::MatrixXd Z = Xw * XtWX_inv;
            Eigen::VectorXd h = (Z.cwiseProduct(Xw)).rowwise().sum();
            if (opt.hc_type == 2) {
                s /= ((1.0 - h.array()).max(opt.optim.eps)).sqrt();
            } else {
                s /= (1.0 - h.array()).max(opt.optim.eps);
            }
        }
        VectorXd abs_s = s.abs().matrix();
        RowMajorMatrixXd Xs = X;
        for (int j = 0; j < p; ++j) {
            Xs.col(j).array() *= abs_s.array(); // row-scale by |s|
        }
        MatrixXd B = Xs.transpose() * Xs;    // X' diag(s^2) X
        // Var(β̂)_robust ≈ (X'WX)^{-1} * B * (X'WX)^{-1}
        Vrob = XtWX_inv * B * XtWX_inv;
        stats.se_robust = Vrob.diagonal().array().sqrt().matrix();
    } else {
        stats.se_robust.resize(0);
    }

    if (opt.compute_var_mu) {
        ArrayXd var_eta = ArrayXd::Zero(n);
        if (opt.se_flag & 0x2) {
            for (int i = 0; i < n; ++i) {
                var_eta[i] = X.row(i) * Vrob * X.row(i).transpose();
            }
        } else {
            for (int i = 0; i < n; ++i) {
                var_eta[i] = X.row(i) * XtWX_inv * X.row(i).transpose();
            }
        }
        // ArrayXd var_mu = (c.array().square()) * (var_eta.exp() - 1.) * (2 * eta + var_eta).exp();
        ArrayXd var_mu = dmu.square() * var_eta; // delta method
        stats.var_mu = var_mu.sum();
    }

    if (opt.store_cov) {
        if (opt.se_flag & 0x1) {
            stats.cov_fisher = std::move(XtWX_inv);
        }
        if (opt.se_flag & 0x2) {
            stats.cov_robust = std::move(Vrob);
        }
    } else {
        stats.cov_fisher.resize(0,0);
        stats.cov_fisher.resize(0,0);
    }
}
