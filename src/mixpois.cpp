#include "mixpois.hpp"
#include "poisreg.hpp"

// Mixture regression (log1p link, sparse)
MixPoisLog1pSparseProblem::MixPoisLog1pSparseProblem(const RowMajorMatrixXd& A_,
        const std::vector<uint32_t>& ids_, const std::vector<double>& cnts_,
        const VectorXd& x_, const VectorXd& c_, const VectorXd& oK_,
        const MLEOptions& opt_, const VectorXd& s1p, const VectorXd& s1m,
        VectorXd* s2p, VectorXd* s2m, int N_active_)
    : A(A_), x(x_), c(c_), oK(oK_), ids(ids_),
    yvec(cnts_.data(), (Eigen::Index)cnts_.size()), opt(opt_),
    N_active(N_active_ < 0 ? (int)A_.rows() : N_active_),
    N((int)A_.rows()), K((int)A_.cols()), n((int)ids_.size()),
    Anz(n, K), xS(n), cS(n),
    Sz_plus(K), Sz_minus(K), Sz2_plus(K), Sz2_minus(K),
    Vnz(n, K), w_cache(n), lam_cache(n), last_qZ(K), slope_cache(n)
{
    if (N_active < 0 || N_active > N) {
        error("%s: N_active must be between 0 and A.rows()", __func__);
    }
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
        g_out->noalias() = g.matrix();
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
    VectorXd& s1p, VectorXd& s1m, VectorXd* s2p, VectorXd* s2m, int N_active)
{
    if (c.size() != A.rows())
        error("%s: c has wrong size", __func__);
    if (oK.size() != A.cols())
        error("%s: oK has wrong size", __func__);
    const int K = static_cast<int>(A.cols());
    MixPoisLog1pSparseProblem P(A, ids, cnts, x, c, oK, opt, s1p, s1m, s2p, s2m, N_active);
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
        const int n_nz = P.n;
        const int n_eff = P.N_active;
        double eps = opt.optim.eps;
        ArrayXd lam = P.lam_cache.array().max(eps);
        // slope=1 if not using safe; otherwise use cached slope
        ArrayXd slope = ArrayXd::Ones(n_nz);
        if (P.slope_cache.size() == n_nz && P.slope_cache.allFinite()) slope = P.slope_cache;

        const ArrayXd cx = P.cS.array() * P.xS.array();
        ArrayXd s = ((1 - P.yvec.array()/lam) * (slope * cx)).abs(); // nnz

        // HC adjustments
        if (opt.hc_type >= 1) { // HC1
            if (n_eff > K) s *= std::sqrt(double(n_eff) / double(n_eff - K));
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

MixPoisLogRegProblem::MixPoisLogRegProblem(const RowMajorMatrixXd& A_,
    const VectorXd& y_, const VectorXd& x_,
    const VectorXd& c_, const VectorXd& oK_, const MLEOptions& opt_,
    int N_active_)
    : A(A_), y(y_), x(x_), c(c_), oK(oK_), opt(opt_), ridge(opt_.ridge),
      N_active(N_active_ < 0 ? (int)A_.rows() : N_active_),
      N((int)A_.rows()), K((int)A_.cols()),
      V(N, K), w_cache(N), lam_cache(N)
{
    if (y.size() != N)  error("%s: y has wrong size", __func__);
    if (x.size() != N)  error("%s: x has wrong size", __func__);
    if (c.size() != N)  error("%s: c has wrong size", __func__);
    if (oK.size() != K) error("%s: o has wrong size", __func__);
    if (N_active < 0 || N_active > N) {
        error("%s: N_active must be between 0 and A.rows()", __func__);
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
    int32_t init_newton, bool init_polish, int N_active)
{
    const int K = (int)A.cols();
    if (!opt.optim.b_min && !opt.optim.b_max) {
        double bd = std::log(100.0);
        warning("%s: no bounds specified, assuming FC within 100", __func__);
        opt.optim.set_bounds(-bd, bd, K);
    }
    MixPoisLogRegProblem P(A, y, x, c, oK, opt, N_active);
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
    const int N_eff = P.N_active;
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
            if (N_eff > K) s *= std::sqrt(double(N_eff) / double(N_eff - K));
        } else if (opt.hc_type >= 2) { // HC2 / HC3
            // leverage h_i = xw_i^T Hinv xw_i, with Xw = sqrt(w) * V
            MatrixXd Z = Xw * Hinv; // rows in V x K
            VectorXd h = (Z.cwiseProduct(Xw)).rowwise().sum(); // rows in V
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

// NB log mixture regression
MixNBLogRegProblem::MixNBLogRegProblem(const RowMajorMatrixXd& A_, int N_active_,
    const VectorXd& y_, const VectorXd& x_,
    const VectorXd& c_, const VectorXd& oK_, double alpha_,
    const MLEOptions& opt_)
    : A(A_), y(y_), x(x_), c(c_), oK(oK_), opt(opt_),
      ridge(opt_.ridge), alpha(alpha_),
      N_active(N_active_), N((int)A_.rows()), K((int)A_.cols()),
      V(N, K), w_cache(N), lam_cache(N)
{
    if (y.size() != N)  error("%s: y has wrong size", __func__);
    if (x.size() != N)  error("%s: x has wrong size", __func__);
    if (c.size() != N)  error("%s: c has wrong size", __func__);
    if (oK.size() != K) error("%s: o has wrong size", __func__);
    if (N_active < 0 || N_active > N) {
        error("%s: N_active must be between 0 and A.rows()", __func__);
    }
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        error("%s: alpha must be positive and finite", __func__);
    }
}

void MixNBLogRegProblem::eval(const VectorXd& b, double* f_out,
        VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const
{
    const double eps = opt.optim.eps;

    ArrayXd s = ArrayXd::Zero(N); // s_i = sum_k a_ik * exp(eta_ik)
    for (int k = 0; k < K; ++k) {
        ArrayXd eta = (oK[k] + x.array() * b[k]).min(40.0);
        ArrayXd vk  = A.col(k).array() * eta.exp(); // v_{ik} over i
        V.col(k) = vk.matrix();
        s += vk;
    }

    ArrayXd lam = (c.array() * s).max(eps);
    lam_cache = lam.matrix();

    if (f_out) {
        const double r = 1.0 / alpha;
        const ArrayXd log1p_term = (alpha * lam).log1p();
        double nll = ((y.array() + r) * log1p_term - y.array() * lam.log()).sum();
        if (ridge > 0.0) nll += 0.5 * ridge * b.squaredNorm();
        *f_out = nll;
    }

    const ArrayXd cx = c.array() * x.array();
    const ArrayXd denom = lam * (1.0 + alpha * lam);
    const ArrayXd dlam = (lam - y.array()) / denom;

    if (g_out) {
        VectorXd g = V.transpose() * (cx * dlam).matrix();
        if (ridge > 0.0) g.noalias() += ridge * b;
        *g_out = std::move(g);
    }

    w_cache = cx.square() / denom;
    if (w_out) *w_out = w_cache;

    if (q_out) {
        VectorXd q = (V.array().square().matrix().transpose() * w_cache.matrix());
        q.array() += ridge;
        *q_out = std::move(q);
    }
}

void MixNBLogRegProblem::compute_se(
    const VectorXd& b_hat, const MLEOptions& opt_se, MLEStats& stats) const
{
    if (opt_se.se_flag == 0) return;

    const int K = (int)b_hat.size();
    const int N = N_active;
    const double eps = opt_se.optim.eps;

    // Refresh caches at b_hat (fills P.V and P.lam_cache)
    double ftmp; VectorXd gtmp, qtmp; ArrayXd wtmp;
    this->eval(b_hat, &ftmp, &gtmp, &qtmp, &wtmp);

    // --- Bread: Fisher Hessian H = V^T diag(w) V + jitter ---
    VectorXd sqrt_w = wtmp.sqrt().matrix();
    MatrixXd Xw = V; // rows in V x K
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
    if (opt_se.se_flag & 0x1) {
        stats.se_fisher = Hinv.diagonal().array().sqrt().matrix();
        if (opt_se.store_cov) stats.cov_fisher = Hinv;
        else stats.cov_fisher.resize(0,0);
    } else {
        stats.se_fisher.resize(0);
        stats.cov_fisher.resize(0,0);
    }

    // Robust SE / cov
    if (opt_se.se_flag & 0x2) {
        const ArrayXd lam = lam_cache.array().max(eps);
        const ArrayXd cx  = (c.array() * x.array());
        const ArrayXd denom = lam * (1.0 + alpha * lam);
        ArrayXd s = (y.array() - lam) * (cx / denom);

        if (opt_se.hc_type == 1) { // HC1
            if (N > K) s *= std::sqrt(double(N) / double(N - K));
        } else if (opt_se.hc_type >= 2) { // HC2 / HC3
            MatrixXd Z = Xw * Hinv; // N x K
            VectorXd h = (Z.cwiseProduct(Xw)).rowwise().sum(); // N
            if (opt_se.hc_type == 2) {
                s /= (1.0 - h.array()).max(eps).sqrt();
            } else {
                s /= (1.0 - h.array()).max(eps);
            }
        }

        VectorXd abs_s = s.abs().matrix();
        MatrixXd Vs = V;
        for (int j = 0; j < K; ++j) {
            Vs.col(j).array() *= abs_s.array();
        }
        MatrixXd B = Vs.transpose() * Vs;

        MatrixXd Vrob = Hinv * B * Hinv;
        stats.se_robust = Vrob.diagonal().array().sqrt().matrix();

        if (opt_se.store_cov) stats.cov_robust = Vrob;
        else stats.cov_robust.resize(0,0);
    } else {
        stats.se_robust.resize(0);
        stats.cov_robust.resize(0,0);
    }
}

// NB log1p mixture regression with sparse exact nonzeros and approximate zeros
MixNBLog1pSparseApproxProblem::MixNBLog1pSparseApproxProblem(
        const RowMajorMatrixXd& A_, int N_,
        const std::vector<uint32_t>& ids_, const std::vector<double>& cnts_,
        const VectorXd& x_, const VectorXd& c_, const VectorXd& oK_,
        double alpha_,
        const OptimOptions& opt_, double ridge_, double soft_tau_,
        const VectorXd& s1p, const VectorXd& s1m,
        const VectorXd& s1p_c2, const VectorXd& s1m_c2,
        const VectorXd& s2p, const VectorXd& s2m,
        double csum_p, double csum_m, double c2sum_p, double c2sum_m)
    : A(A_), x(x_), c(c_), oK(oK_), ids(ids_), N(N_),
      yvec(cnts_.data(), (Eigen::Index)cnts_.size()),
      opt(opt_), ridge(ridge_), soft_tau(soft_tau_), alpha(alpha_),
      K((int)A_.cols()), n((int)ids_.size()),
      Anz(n, K), xS(n), cS(n),
      Sz_plus(K), Sz_minus(K), Szc2_plus(K), Szc2_minus(K),
      Sz2_plus(K), Sz2_minus(K), Cz(0.0), Cz2(0.0),
      Vnz(n, K), w_cache(n), lam_cache(n), last_qZ(K), slope_cache(n)
{
    if (!(alpha > 0.0) || !std::isfinite(alpha)) {
        error("%s: alpha must be positive and finite", __func__);
    }
    if (N < n) {
        error("%s: N is the number of (active) samples, must be >= the number of non-zero values", __func__);
    }
    if (s1p.size() != K || s1m.size() != K) {
        error("%s: need valid vectors for Sz1_plus and Sz1_minus", __func__);
    }
    if (s1p_c2.size() != K || s1m_c2.size() != K) {
        error("%s: need valid vectors for Szc2_plus and Szc2_minus", __func__);
    }
    if (s2p.size() != K || s2m.size() != K) {
        error("%s: need valid vectors for Sz2_plus and Sz2_minus", __func__);
    }
    // Build Anz, xS, cS
    for (int t = 0; t < n; ++t) {
        int i = (int)ids[t];
        Anz.row(t) = A.row(i);
        xS[t] = x[i];
        cS[t] = c[i];
    }
    VectorXd cnz_plus  = (cS.array() * (xS.array() > 0).cast<double>()).matrix();
    VectorXd cnz_minus = (cS.array() * (xS.array() < 0).cast<double>()).matrix();
    Sz_plus.noalias()  = s1p - Anz.transpose() * cnz_plus;   // K
    Sz_minus.noalias() = s1m - Anz.transpose() * cnz_minus;  // K
    Sz_plus  = Sz_plus.cwiseMax(0.0);
    Sz_minus = Sz_minus.cwiseMax(0.0);

    VectorXd cnz_c2_plus  = (cS.array().square() * (xS.array() > 0).cast<double>()).matrix();
    VectorXd cnz_c2_minus = (cS.array().square() * (xS.array() < 0).cast<double>()).matrix();
    Szc2_plus.noalias()  = s1p_c2 - Anz.transpose() * cnz_c2_plus;
    Szc2_minus.noalias() = s1m_c2 - Anz.transpose() * cnz_c2_minus;
    Szc2_plus  = Szc2_plus.cwiseMax(0.0);
    Szc2_minus = Szc2_minus.cwiseMax(0.0);

    Sz2_plus  = s2p;
    Sz2_minus = s2m;
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

    const double cnz_sum = cS.sum();
    const double cnz2_sum = cS.array().square().sum();
    Cz  = (csum_p + csum_m) - cnz_sum;
    Cz2 = (c2sum_p + c2sum_m) - cnz2_sum;
    if (Cz < 0.0) Cz = 0.0;
    if (Cz2 < 0.0) Cz2 = 0.0;
}

void MixNBLog1pSparseApproxProblem::eval_safe(const VectorXd& b, double* f_out,
              VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const
{
    const double eps = opt.eps;
    const double r = 1.0 / alpha;

    // ---- ZERO-set approximate contributions ----
    ArrayXd ep = (oK.array() + b.array()).min(40.0).exp();
    ArrayXd em = (oK.array() - b.array()).min(40.0).exp();

    last_qZ.array() = ep * Sz_plus.array() + em * Sz_minus.array() + ridge;
    const double Lz = (ep * Sz_plus.array() + em * Sz_minus.array()).sum() - Cz;

    const int nZero = N - n;

    bool use_second_order = false;
    if (nZero > 0 && Lz > 0.0) {
        const double meanLamZ = Lz / (double)nZero;
        use_second_order = std::isfinite(meanLamZ) && (alpha * meanLamZ <= kZeroApproxTau);
    }

    double Qz = 0.0;
    ArrayXd ep2, em2;
    if (use_second_order) {
        ep2 = ep.square();
        em2 = em.square();
        Qz = (ep2 * Sz2_plus.array() + em2 * Sz2_minus.array()).sum()
           - 2.0 * (ep * Szc2_plus.array() + em * Szc2_minus.array()).sum()
           + Cz2;
    }

    // ---- NONZERO-set exact contributions ----
    MatrixXd Eta(n, K);
    Eta.noalias() = xS * b.transpose();
    Eta.rowwise() += oK.transpose();
    Eta = Eta.array().min(40.0);

    Vnz = (Anz.array() * Eta.array().exp()).matrix();        // n x K
    ArrayXd m = Vnz.rowwise().sum().array();                 // n
    ArrayXd lambda_raw = cS.array() * (m - 1.0);             // n

    ArrayXd tau = soft_tau * cS.array().max(1.0);
    ArrayXd z   = (lambda_raw - eps) / tau;
    ArrayXd sp  = softplus_stable(z);
    ArrayXd lambda = eps + tau * sp;
    ArrayXd slope  = sigmoid_stable(z);
    lam_cache = lambda.matrix();
    slope_cache = slope;

    if (f_out) {
        const double fZ = use_second_order ? (Lz - 0.5 * alpha * Qz) : Lz;
        const ArrayXd log1p_term = (alpha * lambda).log1p();
        const double fN = ((yvec.array() + r) * log1p_term
                         - yvec.array() * lambda.log()).sum();
        *f_out = fZ + fN + 0.5 * ridge * b.squaredNorm();
    }

    const ArrayXd cx = cS.array() * xS.array();
    const ArrayXd denom = lambda * (1.0 + alpha * lambda);

    if (g_out) {
        ArrayXd g = ep * Sz_plus.array() - em * Sz_minus.array();

        if (use_second_order) {
            ArrayXd g2 = 2.0 * (ep2 * Sz2_plus.array() - em2 * Sz2_minus.array())
                       - 2.0 * (ep  * Szc2_plus.array() - em  * Szc2_minus.array());
            g -= 0.5 * alpha * g2;
        }

        const ArrayXd dlam = (lambda - yvec.array()) / denom;
        const ArrayXd alpha_nz = cx * slope * dlam;
        g += (Vnz.transpose() * alpha_nz.matrix()).array();
        g.array() += ridge * b.array();

        g_out->noalias() = g.matrix();
    }

    w_cache = (slope * cx).square() / denom;
    if (w_out) *w_out = w_cache;

    if (q_out) {
        VectorXd diagN = (Vnz.array().square().matrix().transpose() * w_cache.matrix());
        VectorXd q = last_qZ + diagN;
        *q_out = std::move(q);
    }
}

void MixNBLog1pSparseApproxProblem::compute_se(
    const VectorXd& b_hat, const MLEOptions& opt_se, MLEStats& stats) const
{
    if (opt_se.se_flag == 0) return;

    double ftmp;
    VectorXd gtmp, qtmp;
    ArrayXd wtmp;
    this->eval(b_hat, &ftmp, &gtmp, &qtmp, &wtmp); // fills caches: Vnz, lam_cache, slope_cache, last_qZ

    const int K = (int)b_hat.size();

    // -------- Fisher / GN SE --------
    VectorXd sqrt_w = wtmp.sqrt().matrix();
    MatrixXd Xw = this->Vnz;
    for (int j = 0; j < K; ++j) Xw.col(j).array() *= sqrt_w.array();
    MatrixXd H = Xw.transpose() * Xw;

    // qtmp is diagN + last_qZ (from eval), so setting diagonal is equivalent to adding last_qZ
    H.diagonal() = qtmp.array();
    H.diagonal().array() += 1e-8;

    Eigen::LDLT<MatrixXd> ldlt(H);
    if (ldlt.info() != Eigen::Success) {
        error("%s: LDLT failed on Fisher Hessian", __func__);
    }
    MatrixXd Hinv = ldlt.solve(MatrixXd::Identity(K, K));

    if (opt_se.se_flag & 0x1) {
        stats.se_fisher = Hinv.diagonal().array().sqrt().matrix();
        if (opt_se.store_cov) stats.cov_fisher = Hinv;
        else stats.cov_fisher.resize(0,0);
    } else {
        stats.se_fisher.resize(0);
        stats.cov_fisher.resize(0,0);
    }

    // -------- Robust / Sandwich SE --------
    if (opt_se.se_flag & 0x2) {
        const int n = this->n;
        const double eps = this->opt.eps; // match eval_safe()
        ArrayXd lam = this->lam_cache.array().max(eps);

        ArrayXd slope = ArrayXd::Ones(n);
        if (this->slope_cache.size() == n && this->slope_cache.allFinite()) {
            slope = this->slope_cache;
        }

        const ArrayXd cx = this->cS.array() * this->xS.array();
        const ArrayXd denom = lam * (1.0 + this->alpha * lam);

        // nnz score scaling per row
        ArrayXd s = (lam - this->yvec.array()) * (slope * cx / denom);

        // HC1: use active sample count rather than nnz
        if (opt_se.hc_type >= 1) {
            const int n_eff = this->N;
            if (n_eff > K) s *= std::sqrt(double(n_eff) / double(n_eff - K));
        }

        // B from nnz rows: V^T diag(s^2) V
        MatrixXd Vs = this->Vnz;
        VectorXd abs_s = s.abs().matrix();
        for (int j = 0; j < K; ++j) Vs.col(j).array() *= abs_s.array();
        MatrixXd B = Vs.transpose() * Vs;

        // Add diagonal-only approximation for zeros, matching eval_safe() scaling/capping
        if (this->Sz2_plus.size() == K && this->Sz2_minus.size() == K) {
            ArrayXd ep = (this->oK.array() + b_hat.array()).min(40.0).exp();
            ArrayXd em = (this->oK.array() - b_hat.array()).min(40.0).exp();
            ArrayXd ep2 = ep.square();
            ArrayXd em2 = em.square();

            // Determine whether second-order correction is in effect (same rule as eval_safe)
            const double Lz = (ep * this->Sz_plus.array() + em * this->Sz_minus.array()).sum() - this->Cz;
            const int nZero = std::max(0, this->N - this->n);

            bool use_second_order = false;
            double zscale = 1.0;
            if (nZero > 0 && Lz > 0.0) {
                const double meanLamZ = Lz / (double)nZero;
                use_second_order = std::isfinite(meanLamZ) && (this->alpha * meanLamZ <= kZeroApproxTau);
                if (use_second_order) {
                    zscale = std::max(0.0, 1.0 - this->alpha * meanLamZ);
                }
            }

            VectorXd Bz_diag = (ep2 * this->Sz2_plus.array() + em2 * this->Sz2_minus.array()).matrix();
            B.diagonal().array() += (zscale * zscale) * Bz_diag.array();
        }

        MatrixXd Vrob = Hinv * B * Hinv;
        stats.se_robust = Vrob.diagonal().array().sqrt().matrix();
        if (opt_se.store_cov) stats.cov_robust = std::move(Vrob);
        else stats.cov_robust.resize(0,0);
    } else {
        stats.se_robust.resize(0);
        stats.cov_robust.resize(0,0);
    }
}
