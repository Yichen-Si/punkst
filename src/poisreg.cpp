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

namespace {

template <typename Problem>
double solve_pois_log1p_problem(
    const RowMajorMatrixXd& A,
    const VectorXd& c,
    const VectorXd* o,
    const MLEOptions& opt,
    Problem& P,
    VectorXd& b,
    MLEStats& stats,
    int32_t debug_,
    ArrayXd* res_ptr,
    bool exact_problem)
{
    const int K = static_cast<int>(A.cols());
    if (b.size() != K) {
        double numer = (P.yvec.array() / P.cS.array() + 1.0).log().sum();
        if (o) {
            numer -= o->sum();
        }
        double denom = exact_problem ? 0.0 : P.Anz.sum();
        if (exact_problem) {
            VectorXd A_row_sum = A.rowwise().sum();
            for (auto i : P.ids) {
                denom += A_row_sum[i];
            }
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

    const double ysum = P.yvec.sum();
    stats.pll = ysum > 0
        ? (-final_obj - P.yvec.unaryExpr([](double n){ return log_factorial(n); }).sum()) / ysum
        : 0.0;
    if (opt.compute_residual) {
        ArrayXd res = P.residual(b);
        stats.residual = res.sum() / A.rows();
        if (res_ptr != nullptr && res_ptr->size() == res.size()) {
            *res_ptr += res;
        }
    }
    return final_obj;
}

} // namespace

double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A, const Document& y, double c,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_, ArrayXd* res_ptr)
{
    if (c <= 0.0) error("%s: c must be positive", __func__);
    VectorXd cvec = VectorXd::Constant((int)A.rows(), c);
    MLEOptions opt_exact = opt;
    opt_exact.exact_zero = true;
    return pois_log1p_mle(A, y, cvec, nullptr, opt_exact, b, stats, debug_, res_ptr);
}

double pois_log1p_mle_exact(
    const RowMajorMatrixXd& A, const Document& y,
    const VectorXd& c, const VectorXd* o,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_, ArrayXd* res_ptr)
{
    MLEOptions opt_exact = opt;
    opt_exact.exact_zero = true;
    if (c.size() != A.rows()) error("%s: c is missing or has wrong size", __func__);
    if (o && o->size() != A.rows()) error("%s: o has wrong size", __func__);
    PoisLog1pRegExactProblem P(A, y, c, o, opt_exact);
    const int K = static_cast<int>(A.cols());
    if (b.size() != K) {
        double numer = (P.yvec.array() / c.array() + 1.0).log().sum();
        if (o) {
            numer -= o->sum();
        }
        VectorXd A_row_sum = A.rowwise().sum();
        double denom = 0.0;
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
    if (opt_exact.optim.tron.enabled) {
        final_obj = tron_solve(P, b, opt_exact.optim, stats.optim, debug_);
    } else if (opt_exact.optim.acg.enabled) {
        final_obj = acg_solve(P, b, opt_exact.optim, stats.optim, debug_);
    } else {
        final_obj = newton_solve(P, b, opt_exact.optim, stats.optim, debug_);
    }
    if (debug_ > 0) {
        std::cout << "Finished in " << stats.optim.niters << " iterations" << std::endl;
        std::cout << "Final b (" << b.size() << ", " << b.norm() << ") " << final_obj << std::endl;
    }
    const double ysum = P.yvec.sum();
    stats.pll = ysum > 0
        ? (-final_obj - P.yvec.unaryExpr([](double n){ return log_factorial(n); }).sum()) / ysum
        : 0.0;
    if (opt_exact.compute_residual) {
        ArrayXd res = P.residual(b);
        stats.residual = res.sum() / A.rows();
        if (res_ptr != nullptr && res_ptr->size() == res.size()) {
            *res_ptr += res;
        }
    }
    if (opt.se_flag != 0) {
        pois_log1p_compute_se(A, P.yvec, c, o, opt_exact, b, stats);
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

PoisLog1pSparseContext::PoisLog1pSparseContext(const RowMajorMatrixXd& A_,
    const VectorXd& c_, const MLEOptions& opt_)
    : A(A_), c(c_), opt(opt_) {
    if (c.size() != A.rows()) {
        error("%s: c is missing or has wrong size", __func__);
    }
    zak_all = A.transpose() * c;
    zakl_all = A.transpose() * (c.asDiagonal() * A);
    dZ_all = zakl_all.diagonal();
}

PoisLog1pRegSparseCachedProblem::PoisLog1pRegSparseCachedProblem(
    const PoisLog1pSparseContext& ctx_, const Document& y_, const VectorXd* o_)
    : ctx(ctx_), A(ctx_.A), c(ctx_.c), o(o_), opt(ctx_.opt),
      has_offset(o_ != nullptr), ids(y_.ids), cnts(y_.cnts),
      n(y_.ids.size()), Anz(n, A.cols()), yvec(y_.cnts.data(), n), cS(n) {
    init();
}

void PoisLog1pRegSparseCachedProblem::init() {
    const int K = static_cast<int>(A.cols());
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

    zak = ctx.zak_all;
    zakl = ctx.zakl_all;
    for (size_t t = 0; t < n; ++t) {
        const int i = ids[t];
        const double ci = c[i];
        zak.noalias() -= ci * A.row(i).transpose();
        zakl.noalias() -= ci * (A.row(i).transpose() * A.row(i));
    }
    dZ = zakl.diagonal();

    zoak = VectorXd::Zero(K);
    if (has_offset) {
        zoak = A.transpose() * (c.array() * o->array()).matrix();
        for (size_t t = 0; t < n; ++t) {
            const int i = ids[t];
            zoak.noalias() -= c[i] * (*o)[i] * A.row(i).transpose();
        }
    }

    double min_lower_bound = 0;
    if (opt.optim.b_min) min_lower_bound = opt.optim.b_min->minCoeff();
    nonnegative = min_lower_bound >= 0.0;
    if (has_offset) {
        nonnegative = nonnegative && (o->minCoeff() >= 0.0);
    }
}

void PoisLog1pRegSparseCachedProblem::eval(const VectorXd& bvec, double* f_out,
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

void PoisLog1pRegSparseCachedProblem::eval_safe(const VectorXd& bvec, double* f_out,
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

    ArrayXd C = cS.array() - yvec.array() / phi;
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

ArrayXd PoisLog1pRegSparseCachedProblem::residual(const VectorXd& bvec) const {
    ArrayXd mu = (A * bvec).array();
    if (o) mu += o->array();
    mu = ((mu.exp() - 1.0) * c.array()).max(0.);
    Eigen::Map<const Eigen::Array<uint32_t, Eigen::Dynamic, 1>> ids_map(ids.data(), n);
    Eigen::ArrayXd mu_nz = mu(ids_map.cast<Eigen::Index>());
    mu(ids_map.cast<Eigen::Index>()) = (yvec.array() - mu_nz).square() / mu_nz.max(1e-5);
    return mu;
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

    PoisLog1pRegSparseProblem P(A, y, c, o, opt);
    double final_obj = solve_pois_log1p_problem(A, c, o, opt, P, b, stats, debug_, res_ptr, false);
    if (opt.se_flag != 0) {
        pois_log1p_compute_se(A, y, c, o, opt, b, stats);
    }

    return final_obj;
}

double pois_log1p_mle(
    const PoisLog1pSparseContext& ctx, const Document& y, const VectorXd* o,
    const MLEOptions& opt, VectorXd& b, MLEStats& stats, int32_t debug_, ArrayXd* res_ptr)
{
    if (&opt != &ctx.opt) {
        warning("%s: using options from context; passed options object differs", __func__);
    }
    if (o && o->size() != ctx.A.rows()) {
        error("%s: o has wrong size", __func__);
    }
    if (ctx.opt.exact_zero) {
        return pois_log1p_mle_exact(ctx.A, y, ctx.c, o, ctx.opt, b, stats, debug_, res_ptr);
    }
    PoisLog1pRegSparseCachedProblem P(ctx, y, o);
    double final_obj = solve_pois_log1p_problem(ctx.A, ctx.c, o, ctx.opt, P, b, stats, debug_, res_ptr, false);
    if (ctx.opt.se_flag != 0) {
        pois_log1p_compute_se(ctx.A, y, ctx.c, o, ctx.opt, b, stats);
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
