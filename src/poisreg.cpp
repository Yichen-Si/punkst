#include "poisreg.hpp"

// Exact

void PoisLog1pRegExactProblem::eval(const VectorXd& bvec, double* f_out,
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

void PoisLog1pRegExactProblem::eval_safe(const VectorXd& bvec, double* f_out,
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

ArrayXd PoisLog1pRegExactProblem::residual(const VectorXd& bvec) const {
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
    PoisLog1pRegExactProblem P(A, y, c, o, opt);

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
void PoisLog1pRegSparseProblem::init() {
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

void PoisLog1pRegSparseProblem::eval(const VectorXd& bvec, double* f_out,
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

void PoisLog1pRegSparseProblem::eval_safe(const VectorXd& bvec, double* f_out,
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

ArrayXd PoisLog1pRegSparseProblem::residual(const VectorXd& bvec) const {
    ArrayXd mu = (A * bvec).array();
    if (o) mu += o->array();
    mu = ((mu.exp() - 1.0) * c.array()).max(0.);
    Eigen::Map<const Eigen::Array<uint32_t, Eigen::Dynamic, 1>> ids_map(ids.data(), n);
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
    PoisLog1pRegSparseProblem P(A, y, c, o, opt);

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

// Mixture regression
MixPoisLog1pSparseProblem::MixPoisLog1pSparseProblem(const RowMajorMatrixXd& A_,
        const std::vector<uint32_t>& ids_, const std::vector<double>& cnts_,
        const VectorXd& x_, const VectorXd& c_, const VectorXd& oK_,
        const MLEOptions& opt_, const VectorXd& s1p, const VectorXd& s1m,
        VectorXd* s2p, VectorXd* s2m)
    : A(A_), x(x_), c(c_), oK(oK_), ids(ids_),
    yvec(cnts_.data(), (Eigen::Index)cnts_.size()), opt(opt_),
    N((int)A_.rows()), K((int)A_.cols()), n((int)ids_.size()),
    Anz(n, K), xS(n), cS(n),
    Sz_plus(K), Sz_minus(K), Sz2_plus(K), Sz2_minus(K),
    Vnz(n, K), w_cache(n), lam_cache(n), last_qZ(K), slope_cache(n)
{
    if (s1p.size() != K || s1m.size() != K) {
        error("%s: need valid vectors for Sz1_plus and Sz1_minus", __func__);
    }
    // Build Anz, xS, cS
    for (int t = 0; t < n; ++t) {
        int i = (int)ids[t];
        Anz.row(t) = A.row(i);
        xS[t] = x[i];
        cS[t] = c[i];
    }
    VectorXd cnz_plus  = (cS.array() * (xS.array() > 0).cast<double>()).matrix(); // n
    VectorXd cnz_minus = (cS.array() * (xS.array() < 0).cast<double>()).matrix(); // n
    Sz_plus.noalias()  = s1p - Anz.transpose() * cnz_plus;   // K
    Sz_minus.noalias() = s1m - Anz.transpose() * cnz_minus;  // K
    Sz_plus  = Sz_plus.cwiseMax(0.0);
    Sz_minus = Sz_minus.cwiseMax(0.0);

    if (opt.se_flag & 0x2) {
        if (s2p == nullptr || s2m == nullptr || s2p->size() != K || s2m->size() != K) {
            error("%s: need valid pointers for Sz2_plus and Sz2_minus when robust SE is requested", __func__);
        }
        Sz2_plus  = *s2p;
        Sz2_minus = *s2m;
        // subtract nonzero rows to get Z-only
        for (int t = 0; t < n; ++t) {
            if (cS[t] <= 0 || xS[t] == 0) continue;
            double ci2 = cS[t] * cS[t];
            if (xS[t] > 0) {
                Sz2_plus.array()  -= ci2 * Anz.row(t).array().square();
            } else {
                Sz2_minus.array() -= ci2 * Anz.row(t).array().square();
            }
        }
        Sz2_plus  = Sz2_plus.cwiseMax(0.0);
        Sz2_minus = Sz2_minus.cwiseMax(0.0);
    }
}

void MixPoisLog1pSparseProblem::eval_safe(const VectorXd& b, double* f_out,
              VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const
{
    const double eps = opt.optim.eps;
    const double ridge = opt.ridge;

    // ---- ZERO-set exact contributions (diagonal in k) ----
    // ep_k = exp(o_k + b_k), em_k = exp(o_k - b_k)
    ArrayXd ep = (oK.array() + b.array()).min(40.0).exp();
    ArrayXd em = (oK.array() - b.array()).min(40.0).exp();
    // diagonal curvature
    last_qZ.array() = ep * Sz_plus.array() + em * Sz_minus.array() + ridge;

    // ---- NONZERO-set exact contributions ----
    // Eta = xS * b^T + 1 * oK^T
    MatrixXd Eta(n, K);
    Eta.noalias() = xS * b.transpose();
    Eta.rowwise() += oK.transpose();
    Eta = Eta.array().min(40.0);

    Vnz = (Anz.array() * Eta.array().exp()).matrix();        // n x K
    ArrayXd m = Vnz.rowwise().sum().array();                 // n
    ArrayXd lambda_raw = cS.array() * (m - 1.0);             // n

    ArrayXd tau = opt.soft_tau * cS.array().max(1.0);
    ArrayXd z   = (lambda_raw - eps) / tau;
    ArrayXd sp  = softplus_stable(z);
    ArrayXd lambda = eps + tau * sp;
    ArrayXd slope  = sigmoid_stable(z);
    lam_cache = lambda.matrix();
    slope_cache = slope;

    if (f_out) {
        double fZ = 0.0;
        fZ = ((ep - 1.0) * Sz_plus.array() + (em - 1.0) * Sz_minus.array()).sum();
        const double fN = lambda.sum() - (yvec.array() * lambda.log()).sum();
        *f_out = fZ + fN + 0.5 * ridge * b.squaredNorm();
    }

    const ArrayXd cx = cS.array() * xS.array();
    if (g_out) {
        ArrayXd g = ep * Sz_plus.array() - em * Sz_minus.array();
        const ArrayXd alpha = cx * slope * (1.0 - yvec.array() / lambda);
        g += (Vnz.transpose() * alpha.matrix()).array();
        g.array() += ridge * b.array();
        *g_out = std::move(g);
    }

    w_cache = (slope * cx).square() / lambda;
    if (w_out) *w_out = w_cache;

    if (q_out) {
        VectorXd diagN = (Vnz.array().square().matrix().transpose() * w_cache.matrix()); // non-zero contribution to q
        VectorXd q = last_qZ + diagN; // last_qZ includes ridge
        *q_out = std::move(q);
    }
}

double mix_pois_log1p_mle(const RowMajorMatrixXd& A,
    const std::vector<uint32_t>& ids, const std::vector<double>& cnts,
    const VectorXd& x, const VectorXd& c, const VectorXd& oK,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats,
    VectorXd& s1p, VectorXd& s1m, VectorXd* s2p, VectorXd* s2m)
{
    if (c.size() != A.rows())
        error("%s: c has wrong size", __func__);
    if (oK.size() != A.cols())
        error("%s: oK has wrong size", __func__);
    const int K = static_cast<int>(A.cols());
    MixPoisLog1pSparseProblem P(A, ids, cnts, x, c, oK, opt, s1p, s1m, s2p, s2m);
    if (b.size() != K) {
        b = VectorXd::Zero(K);
    }
    double final_obj = tron_solve(P, b, opt.optim, stats.optim);
    if (opt.se_flag != 0) {
        mix_pois_log1p_compute_se(P, b, opt, stats);
    }
    return final_obj;
}

void mix_pois_log1p_compute_se(const MixPoisLog1pSparseProblem& P,
    const VectorXd& b_hat, const MLEOptions& opt, MLEStats& stats)
{
    // Ensure caches correspond to b_hat
    double ftmp; VectorXd gtmp, qtmp; ArrayXd wtmp;
    P.eval(b_hat, &ftmp, &gtmp, &qtmp, &wtmp);

    const int K = (int)b_hat.size();
    VectorXd sqrt_w = wtmp.sqrt().matrix();
    MatrixXd Xw = P.Vnz;
    for (int j = 0; j < K; ++j) Xw.col(j).array() *= sqrt_w.array();

    MatrixXd H = Xw.transpose() * Xw;
    H.diagonal() = qtmp.array(); // add back zero-set curvature
    H.diagonal().array() += 1e-8; // for safe inverse
    Eigen::LDLT<MatrixXd> ldlt(H);
    if (ldlt.info() != Eigen::Success) {
        error("%s: LDLT failed on Fisher Hessian", __func__);
    }
    MatrixXd Hinv = ldlt.solve(MatrixXd::Identity(K, K));


    // -------- Fisher --------
    if (opt.se_flag & 0x1) {
        stats.se_fisher = Hinv.diagonal().array().sqrt().matrix();
        if (opt.store_cov) stats.cov_fisher = Hinv;
        else stats.cov_fisher.resize(0,0);
    } else {
        stats.se_fisher.resize(0);
        stats.cov_fisher.resize(0,0);
    }

    // -------- Robust --------
    if (opt.se_flag & 0x2) {
        int n = P.n;
        double eps = opt.optim.eps;
        ArrayXd lam = P.lam_cache.array().max(eps);
        // slope=1 if not using safe; otherwise use cached slope
        ArrayXd slope = ArrayXd::Ones(n);
        if (P.slope_cache.size() == n && P.slope_cache.allFinite()) slope = P.slope_cache;

        const ArrayXd cx = P.cS.array() * P.xS.array();
        ArrayXd s = ((1 - P.yvec.array()/lam) * (slope * cx)).abs(); // n

        // HC adjustments
        if (opt.hc_type >= 1) { // HC1
            if (n > K) s *= std::sqrt(double(n) / double(n - K));
        }
        // Meat from NONZERO rows
        MatrixXd Vs = P.Vnz; // a_ik e_ik
        for (int j = 0; j < K; ++j) Vs.col(j).array() *= s;
        MatrixXd B = Vs.transpose() * Vs; // K x K: \sum_i g_i g_i^T

        // ZERO-set meat approximation: only diagonal
        if (P.Sz2_plus.size() == K && P.Sz2_minus.size() == K) {
            ArrayXd op2 = (2.0 * (P.oK.array() + b_hat.array())).min(40.0);
            ArrayXd om2 = (2.0 * (P.oK.array() - b_hat.array())).min(40.0);
            ArrayXd ep2 = op2.exp();
            ArrayXd em2 = om2.exp();
            VectorXd Bz_diag = (ep2 * P.Sz2_plus.array() + em2 * P.Sz2_minus.array()).matrix();
            B.diagonal().array() += Bz_diag.array();
        }
        MatrixXd Vrob = Hinv * B * Hinv;
        stats.se_robust = Vrob.diagonal().array().sqrt().matrix();
        if (opt.store_cov) stats.cov_robust = std::move(Vrob);
        else stats.cov_robust.resize(0,0);
    } else {
        stats.se_robust.resize(0);
        stats.cov_robust.resize(0,0);
    }
}


void MixPoisLogRegProblem::eval(const VectorXd& b, double* f_out,
        VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const
{
    const double ridge = opt.ridge;

    ArrayXd s = ArrayXd::Zero(N); // s_i = sum_k a_ik * exp(eta_ik)
    for (int k = 0; k < K; ++k) {
        ArrayXd eta = (oK[k] + x.array() * b[k]).min(40.0);
        ArrayXd vk  = A.col(k).array() * eta.exp(); // v_{ik} over i
        V.col(k) = vk.matrix();
        s += vk;
    }

    ArrayXd lam = (c.array() * s).max(opt.optim.eps);
    lam_cache = lam.matrix();

    if (f_out) {
        double nll = lam.sum() - (y.array() * lam.log()).sum();
        if (ridge > 0.0) nll += 0.5 * ridge * b.squaredNorm();
        *f_out = nll;
    }

    const ArrayXd cx = c.array() * x.array();
    const ArrayXd alpha = cx * (1.0 - y.array() / lam);

    if (g_out) {
        VectorXd g = V.transpose() * alpha.matrix();
        if (ridge > 0.0) g.noalias() += ridge * b;
        *g_out = std::move(g);
    }

    w_cache = cx.square() / lam;
    if (w_out) *w_out = w_cache;

    if (q_out) {
        VectorXd q = (V.array().square().matrix().transpose() * w_cache.matrix());
        q.array() += ridge;
        *q_out = std::move(q);
    }
}


double mix_pois_log_mle(const RowMajorMatrixXd& A,
    const VectorXd& y, const VectorXd& x,
    const VectorXd& c, const VectorXd& oK,
    MLEOptions& opt, VectorXd& b, MLEStats& stats,
    int32_t init_newton, bool init_polish)
{
    const int K = (int)A.cols();
    if (!opt.optim.b_min && !opt.optim.b_max) {
        double bd = std::log(100.0);
        warning("%s: no bounds specified, assuming FC within 100", __func__);
        opt.optim.set_bounds(-bd, bd, K);
    }
    MixPoisLogRegProblem P(A, y, x, c, oK, opt);
    if (b.size() != K) {
        if (init_newton > 0) {
            b = init_mix_pois_log_vec(A, y, x, c, oK, opt.optim.eps,
                                      init_newton,init_polish,&opt);
        } else {
            b = VectorXd::Zero(K);
        }
    }
    double final_obj = tron_solve(P, b, opt.optim, stats.optim);
    if (opt.se_flag != 0) {
        mix_pois_log_compute_se(P, b, opt, stats);
    }
    return final_obj;
}

void mix_pois_log_compute_se(const MixPoisLogRegProblem& P,
    const VectorXd& b_hat, const MLEOptions& opt, MLEStats& stats)
{
    if (opt.se_flag == 0) return;

    const int K = (int)b_hat.size();
    const int N = (int)P.V.rows();
    const double eps = opt.optim.eps;

    // Refresh caches at b_hat (fills P.V and P.lam_cache)
    double ftmp; VectorXd gtmp, qtmp; ArrayXd wtmp;
    P.eval(b_hat, &ftmp, &gtmp, &qtmp, &wtmp);

    // --- Bread: Fisher Hessian H = V^T diag(w) V + jitter ---
    // Here w_i = (c_i x_i)^2 / lambda_i is returned as wtmp by eval()
    VectorXd sqrt_w = wtmp.sqrt().matrix();
    MatrixXd Xw = P.V; // N x K
    for (int j = 0; j < K; ++j) {
        Xw.col(j).array() *= sqrt_w.array();
    }

    MatrixXd H = Xw.transpose() * Xw;
    H.diagonal() = qtmp.array(); // uses diagN + ridge from eval()
    H.diagonal().array() += 1e-8;
    Eigen::LDLT<MatrixXd> ldlt(H);
    if (ldlt.info() != Eigen::Success) {
        error("%s: LDLT failed on Fisher Hessian", __func__);
    }
    MatrixXd Hinv = ldlt.solve(MatrixXd::Identity(K, K));

    // Fisher SE / cov
    if (opt.se_flag & 0x1) {
        stats.se_fisher = Hinv.diagonal().array().sqrt().matrix();
        if (opt.store_cov) stats.cov_fisher = Hinv;
        else stats.cov_fisher.resize(0,0);
    } else {
        stats.se_fisher.resize(0);
        stats.cov_fisher.resize(0,0);
    }

    // Robust SE / cov
    if (opt.se_flag & 0x2) {
        // s_i = ((y_i - lambda_i)/lambda_i) * (c_i x_i)
        const ArrayXd lam = P.lam_cache.array().max(eps);
        const ArrayXd cx  = (P.c.array() * P.x.array());
        ArrayXd s = (P.y.array() - lam) * (cx / lam);

        // HC adjustments
        if (opt.hc_type == 1) { // HC1
            if (N > K) s *= std::sqrt(double(N) / double(N - K));
        } else if (opt.hc_type >= 2) { // HC2 / HC3
            // leverage h_i = xw_i^T Hinv xw_i, with Xw = sqrt(w) * V
            MatrixXd Z = Xw * Hinv; // N x K
            VectorXd h = (Z.cwiseProduct(Xw)).rowwise().sum(); // N
            if (opt.hc_type == 2) {
                s /= (1.0 - h.array()).max(eps).sqrt();
            } else {
                s /= (1.0 - h.array()).max(eps);
            }
        }

        // Meat: B = V^T diag(s^2) V  (compute by row-scaling V by |s|)
        VectorXd abs_s = s.abs().matrix();
        MatrixXd Vs = P.V;
        for (int j = 0; j < K; ++j) {
            Vs.col(j).array() *= abs_s.array();
        }
        MatrixXd B = Vs.transpose() * Vs;

        MatrixXd Vrob = Hinv * B * Hinv;
        stats.se_robust = Vrob.diagonal().array().sqrt().matrix();

        if (opt.store_cov) stats.cov_robust = Vrob;
        else stats.cov_robust.resize(0,0);
    } else {
        stats.se_robust.resize(0);
        stats.cov_robust.resize(0,0);
    }
}

double init_b0_loglink_1d(const RowMajorMatrixXd& A,
    const VectorXd& y, const VectorXd& x,
    const VectorXd& c, const VectorXd& oK, double eps, int iters) {
    const int N = (int)A.rows();
    VectorXd eo = oK.array().exp().matrix(); // K
    ArrayXd s  = (A * eo).array().max(eps);  // N, s_i = sum_k a_ik exp(o_k)
    double b0 = 0.0;
    for (int it = 0; it < iters; ++it) {
        ArrayXd lam = (c.array() * s) * (x.array() * b0).exp();
        lam = lam.max(eps);

        double num = (x.array() * (y.array() - lam)).sum();
        double den = (x.array().square() * lam).sum();
        if (den <= eps) break;

        double step = num / den;
        b0 += step;
        if (std::abs(step) < 1e-8) break;
    }
    return b0;
}

VectorXd init_mix_pois_log_vec(const RowMajorMatrixXd& A,
    const VectorXd& y, const VectorXd& x,
    const VectorXd& c, const VectorXd& oK,
    double eps, int newton_steps_per_k,
    bool do_diag_polish, const MLEOptions* opt_for_polish)
{
    const int N = (int)A.rows();
    const int K = (int)A.cols();
    if (y.size() != N || x.size() != N || c.size() != N || oK.size() != K)
        error("%s: dimension mismatch", __func__);

    // 1) shared slope b0
    double b0 = init_b0_loglink_1d(A, y, x, c, oK, eps, 20);
    VectorXd b = VectorXd::Constant(K, b0);
    ArrayXd eo = oK.array().exp();
    RowMajorMatrixXd Ae = A; // N x K : a_ik * exp(o_k)
    for (int k = 0; k < K; ++k) Ae.col(k).array() *= eo[k];

    // 2) compute expectation
    MatrixXd Eta_xb = x * b.transpose(); // N x K
    MatrixXd Exb = Eta_xb.array().exp().matrix(); // x * b^T
    MatrixXd V = (Ae.array() * Exb.array()).matrix(); // N x K, A*exp(o+x b^T)
    ArrayXd s = V.rowwise().sum().array().max(eps);
    MatrixXd P = (V.array().colwise() / s).matrix(); // N x K
    // yhat_ik = y_i * P_ik
    MatrixXd Yhat = (P.array().colwise() * y.array()).matrix(); // N x K

    // 3) Per-k 1D Newton steps for b_k using Yhat
    // mu_ik = c_i * a_ik exp(o_k + x_i b_k)
    ArrayXd cx = c.array();
    for (int it = 0; it < newton_steps_per_k; ++it) {
        Eta_xb.noalias() = x * b.transpose();
        Exb = Eta_xb.array().exp().matrix();
        V = (Ae.array() * Exb.array()).matrix();
        MatrixXd MU = (V.array().colwise() * cx).matrix();
        // Newton numerator per k: sum_i x_i (yhat_ik - mu_ik)
        // denominator per k:      sum_i x_i^2 mu_ik
        VectorXd num = ( (Yhat - MU).array().colwise() * x.array() ).colwise().sum().transpose();
        VectorXd den = ( MU.array().colwise() * x.array().square() ).colwise().sum().transpose();
        for (int k = 0; k < K; ++k) {
            double d = std::max(eps, den[k]);
            b[k] += num[k] / d;
        }
    }

    // diagonal Newton polish
    if (do_diag_polish) {
        if (!opt_for_polish) {return b;}
        MLEOptions opt = *opt_for_polish;
        opt.optim.set_bounds(-1e30, 1e30, K); // unbounded
        MixPoisLogRegProblem Pprob(A, y, x, c, oK, opt);
        for (int t = 0; t < 2; ++t) {
            VectorXd g, q; ArrayXd w;
            Pprob.eval(b, nullptr, &g, &q, &w);
            VectorXd step = g.cwiseQuotient(q.array().max(eps).matrix());
            b -= step;
            project_to_box(b, opt.optim);
            if (step.norm() / std::max(1.0, b.norm()) < 1e-6) break;
        }
    }

    return b;
}
