#include "mixpois.hpp"
#include "poisreg.hpp"

// Mixture regression (log1p link, sparse)
MixPoisLog1pSparseContext::MixPoisLog1pSparseContext(const RowMajorMatrixXd& A_,
        const VectorXd& x_, const VectorXd& c_, const MLEOptions& opt_,
        bool robust_se_full, int N_active_, bool lda_uncertainty_,
        double size_factor_)
    : A(A_), x(x_), c(c_), opt(opt_),
    N_active(N_active_ < 0 ? (int)A_.rows() : N_active_),
    N((int)A_.rows()), K((int)A_.cols()),
    lda_uncertainty(lda_uncertainty_), size_factor(size_factor_)
{
    if (x.size() != N)  error("%s: x has wrong size", __func__);
    if (c.size() != N)  error("%s: c has wrong size", __func__);
    if (N_active < 0 || N_active > N) {
        error("%s: N_active must be between 0 and A.rows()", __func__);
    }

    VectorXd wp = (c.array() * (x.array() > 0).cast<double>()).matrix();
    VectorXd wm = (c.array() * (x.array() < 0).cast<double>()).matrix();
    s1p_store = A.transpose() * wp;
    s1m_store = A.transpose() * wm;
    s1p = &s1p_store;
    s1m = &s1m_store;

    if (opt.se_flag & 0x2) {
        has_s2 = true;
        if (robust_se_full) {
            s2p_store = VectorXd::Zero(K);
            s2m_store = VectorXd::Zero(K);
            m2p_store = MatrixXd::Zero(K, K);
            m2m_store = MatrixXd::Zero(K, K);
            for (int i = 0; i < N; ++i) {
                if (c[i] <= 0.0 || !(x[i] > 0 || x[i] < 0)) continue;
                const double ci2 = c[i] * c[i];
                const auto ai = A.row(i);
                if (x[i] > 0.0) {
                    s2p_store.array() += ci2 * ai.array().square();
                    m2p_store.noalias() += ci2 * (ai.transpose() * ai);
                } else {
                    s2m_store.array() += ci2 * ai.array().square();
                    m2m_store.noalias() += ci2 * (ai.transpose() * ai);
                }
            }
            m2p_store = 0.5 * (m2p_store + m2p_store.transpose());
            m2m_store = 0.5 * (m2m_store + m2m_store.transpose());
            m2p = &m2p_store;
            m2m = &m2m_store;
            has_m2 = true;
        } else {
            s2p_store = VectorXd::Zero(K);
            s2m_store = VectorXd::Zero(K);
            for (int i = 0; i < N; ++i) {
                if (c[i] <= 0.0 || !(x[i] > 0 || x[i] < 0)) continue;
                const double ci2 = c[i] * c[i];
                const auto ai = A.row(i);
                if (x[i] > 0.0) {
                    s2p_store.array() += ci2 * ai.array().square();
                } else {
                    s2m_store.array() += ci2 * ai.array().square();
                }
            }
        }
        s2p = &s2p_store;
        s2m = &s2m_store;
    }

    if (lda_uncertainty && (opt.se_flag & 0x2)) {
        if (size_factor <= 0.0) {
            error("%s: size_factor must be positive when LDA uncertainty is enabled", __func__);
        }
        has_a_unc = true;
        has_a2 = true;
        a1p_store = VectorXd::Zero(K);
        a1m_store = VectorXd::Zero(K);
        a2p_store = VectorXd::Zero(K);
        a2m_store = VectorXd::Zero(K);
        if (robust_se_full) {
            a2p_full_store = MatrixXd::Zero(K, K);
            a2m_full_store = MatrixXd::Zero(K, K);
        }
        for (int i = 0; i < N; ++i) {
            if (c[i] <= 0.0 || !(x[i] > 0 || x[i] < 0)) continue;
            const double denom = 2.0 + c[i] * size_factor;
            if (denom <= 0.0) continue;
            const double wi = (c[i] * c[i]) / denom;
            const auto ai = A.row(i);
            if (x[i] > 0.0) {
                a1p_store.array() += wi * ai.array();
                a2p_store.array() += wi * ai.array().square();
                if (robust_se_full) {
                    a2p_full_store.noalias() += wi * (ai.transpose() * ai);
                }
            } else {
                a1m_store.array() += wi * ai.array();
                a2m_store.array() += wi * ai.array().square();
                if (robust_se_full) {
                    a2m_full_store.noalias() += wi * (ai.transpose() * ai);
                }
            }
        }
        a1p = &a1p_store;
        a1m = &a1m_store;
        a2p = &a2p_store;
        a2m = &a2m_store;
        if (robust_se_full) {
            a2p_full_store = 0.5 * (a2p_full_store + a2p_full_store.transpose());
            a2m_full_store = 0.5 * (a2m_full_store + a2m_full_store.transpose());
            a2p_full = &a2p_full_store;
            a2m_full = &a2m_full_store;
            has_a2_full = true;
        }
    }
}

MixPoisLog1pSparseContext::MixPoisLog1pSparseContext(const RowMajorMatrixXd& A_,
        const VectorXd& x_, const VectorXd& c_, const MLEOptions& opt_,
        const VectorXd& s1p_, const VectorXd& s1m_,
        const VectorXd* s2p_, const VectorXd* s2m_,
        const MatrixXd* m2p_, const MatrixXd* m2m_, int N_active_,
        bool lda_uncertainty_, double size_factor_)
    : A(A_), x(x_), c(c_), opt(opt_),
    N_active(N_active_ < 0 ? (int)A_.rows() : N_active_),
    N((int)A_.rows()), K((int)A_.cols()),
    lda_uncertainty(lda_uncertainty_), size_factor(size_factor_),
    s1p(&s1p_), s1m(&s1m_), s2p(s2p_), s2m(s2m_),
    m2p(m2p_), m2m(m2m_)
{
    if (x.size() != N)  error("%s: x has wrong size", __func__);
    if (c.size() != N)  error("%s: c has wrong size", __func__);
    if (N_active < 0 || N_active > N) {
        error("%s: N_active must be between 0 and A.rows()", __func__);
    }
    if (s1p_.size() != K || s1m_.size() != K) {
        error("%s: need valid vectors for Sz1_plus and Sz1_minus", __func__);
    }
    if (opt.se_flag & 0x2) {
        if (s2p_ == nullptr || s2m_ == nullptr || s2p_->size() != K || s2m_->size() != K) {
            error("%s: need valid pointers for Sz2_plus and Sz2_minus when robust SE is requested", __func__);
        }
        has_s2 = true;
        if (m2p_ && m2m_) {
            if (m2p_->rows() != K || m2p_->cols() != K ||
                m2m_->rows() != K || m2m_->cols() != K) {
                error("%s: m2p_all/m2m_all must be KxK", __func__);
            }
            has_m2 = true;
        }
    }
    if (lda_uncertainty && (opt.se_flag & 0x2)) {
        if (size_factor <= 0.0) {
            error("%s: size_factor must be positive when LDA uncertainty is enabled", __func__);
        }
        has_a_unc = true;
        has_a2 = true;
        a1p_store = VectorXd::Zero(K);
        a1m_store = VectorXd::Zero(K);
        a2p_store = VectorXd::Zero(K);
        a2m_store = VectorXd::Zero(K);
        const bool robust_se_full = (m2p_ && m2m_);
        if (robust_se_full) {
            a2p_full_store = MatrixXd::Zero(K, K);
            a2m_full_store = MatrixXd::Zero(K, K);
        }
        for (int i = 0; i < N; ++i) {
            if (c[i] <= 0.0 || !(x[i] > 0 || x[i] < 0)) continue;
            const double denom = 2.0 + c[i] * size_factor;
            if (denom <= 0.0) continue;
            const double wi = (c[i] * c[i]) / denom;
            const auto ai = A.row(i);
            if (x[i] > 0.0) {
                a1p_store.array() += wi * ai.array();
                a2p_store.array() += wi * ai.array().square();
                if (robust_se_full) {
                    a2p_full_store.noalias() += wi * (ai.transpose() * ai);
                }
            } else {
                a1m_store.array() += wi * ai.array();
                a2m_store.array() += wi * ai.array().square();
                if (robust_se_full) {
                    a2m_full_store.noalias() += wi * (ai.transpose() * ai);
                }
            }
        }
        a1p = &a1p_store;
        a1m = &a1m_store;
        a2p = &a2p_store;
        a2m = &a2m_store;
        if (robust_se_full) {
            a2p_full_store = 0.5 * (a2p_full_store + a2p_full_store.transpose());
            a2m_full_store = 0.5 * (a2m_full_store + a2m_full_store.transpose());
            a2p_full = &a2p_full_store;
            a2m_full = &a2m_full_store;
            has_a2_full = true;
        }
    }
}

MixPoisLog1pSparseProblem::MixPoisLog1pSparseProblem(const MixPoisLog1pSparseContext& ctx_)
    : A(ctx_.A), x(ctx_.x), c(ctx_.c), opt(ctx_.opt),
    N_active(ctx_.N_active),
    N(ctx_.N), K(ctx_.K), n(0),
    Anz(0, ctx_.K), xS(0), cS(0),
    Sz_plus(ctx_.K), Sz_minus(ctx_.K), Sz2_plus(0), Sz2_minus(0),
    Mz_plus(0, 0), Mz_minus(0, 0),
    Az1_plus(0), Az1_minus(0), Az2_plus(0), Az2_minus(0),
    Az2_full_plus(0, 0), Az2_full_minus(0, 0),
    Vnz(0, ctx_.K), w_cache(0), lam_cache(0), last_qZ(ctx_.K), slope_cache(0),
    ctx(ctx_)
{
    if (!ctx.s1p || !ctx.s1m) {
        error("%s: missing s1p/s1m precompute", __func__);
    }
    if (ctx.has_s2) {
        Sz2_plus.resize(K);
        Sz2_minus.resize(K);
    }
    if (ctx.has_m2) {
        Mz_plus.resize(K, K);
        Mz_minus.resize(K, K);
        robust_se_diagonal_only = false;
    }
    if (ctx.has_a_unc) {
        Az1_plus.resize(K);
        Az1_minus.resize(K);
        if (ctx.has_a2) {
            Az2_plus.resize(K);
            Az2_minus.resize(K);
        }
        if (ctx.has_a2_full) {
            Az2_full_plus.resize(K, K);
            Az2_full_minus.resize(K, K);
        }
    }
}

void MixPoisLog1pSparseProblem::reset_feature(const std::vector<uint32_t>& ids_,
        const std::vector<double>& cnts_, const VectorXd& oK_)
{
    if (oK_.size() != K) {
        error("%s: oK has wrong size", __func__);
    }
    if (cnts_.size() != ids_.size()) {
        error("%s: ids and cnts must have the same size", __func__);
    }
    oK = &oK_;
    yptr = cnts_.data();
    ysize = (Eigen::Index)cnts_.size();
    n = (int)ids_.size();

    Anz.resize(n, K);
    xS.resize(n);
    cS.resize(n);
    Vnz.resize(n, K);
    w_cache.resize(n);
    lam_cache.resize(n);
    slope_cache.resize(n);

    // Build Anz, xS, cS
    for (int t = 0; t < n; ++t) {
        int i = (int)ids_[t];
        Anz.row(t) = A.row(i);
        xS[t] = x[i];
        cS[t] = c[i];
    }
    VectorXd cnz_plus  = (cS.array() * (xS.array() > 0).cast<double>()).matrix(); // n
    VectorXd cnz_minus = (cS.array() * (xS.array() < 0).cast<double>()).matrix(); // n
    Sz_plus.noalias()  = (*ctx.s1p) - Anz.transpose() * cnz_plus;   // K
    Sz_minus.noalias() = (*ctx.s1m) - Anz.transpose() * cnz_minus;  // K
    Sz_plus  = Sz_plus.cwiseMax(0.0);
    Sz_minus = Sz_minus.cwiseMax(0.0);

    if (opt.se_flag & 0x2) {
        if (!ctx.has_s2 || ctx.s2p == nullptr || ctx.s2m == nullptr) {
            error("%s: need valid pointers for Sz2_plus and Sz2_minus when robust SE is requested", __func__);
        }
        Sz2_plus  = *ctx.s2p;
        Sz2_minus = *ctx.s2m;
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

    if ((opt.se_flag & 0x2) && ctx.has_m2) {
        Mz_plus  = *ctx.m2p;
        Mz_minus = *ctx.m2m;
        // subtract nonzero rows to get Z-only
        for (int t = 0; t < n; ++t) {
            if (cS[t] <= 0 || xS[t] == 0) continue;
            const double ci2 = cS[t] * cS[t];
            if (xS[t] > 0) {
                Mz_plus.noalias()  -= ci2 * (Anz.row(t).transpose() * Anz.row(t));
            } else {
                Mz_minus.noalias() -= ci2 * (Anz.row(t).transpose() * Anz.row(t));
            }
        }
        Mz_plus  = 0.5 * (Mz_plus + Mz_plus.transpose());
        Mz_minus = 0.5 * (Mz_minus + Mz_minus.transpose());
        Mz_plus  = Mz_plus.cwiseMax(0.0);
        Mz_minus = Mz_minus.cwiseMax(0.0);
    }

    if (ctx.has_a_unc) {
        if (!ctx.a1p || !ctx.a1m || !ctx.a2p || !ctx.a2m) {
            error("%s: missing LDA-uncertainty precompute", __func__);
        }
        ArrayXd denom = 2.0 + cS.array() * ctx.size_factor;
        ArrayXd wS = cS.array().square() / denom.max(1e-12);
        wS *= (cS.array() > 0).cast<double>();
        VectorXd wnz_plus  = (wS * (xS.array() > 0).cast<double>()).matrix();
        VectorXd wnz_minus = (wS * (xS.array() < 0).cast<double>()).matrix();
        Az1_plus.noalias()  = (*ctx.a1p) - Anz.transpose() * wnz_plus;
        Az1_minus.noalias() = (*ctx.a1m) - Anz.transpose() * wnz_minus;
        Az1_plus  = Az1_plus.cwiseMax(0.0);
        Az1_minus = Az1_minus.cwiseMax(0.0);

        Az2_plus  = *ctx.a2p;
        Az2_minus = *ctx.a2m;
        for (int t = 0; t < n; ++t) {
            if (xS[t] == 0 || wS[t] <= 0.0) continue;
            const double wt = wS[t];
            if (xS[t] > 0) {
                Az2_plus.array()  -= wt * Anz.row(t).array().square();
            } else {
                Az2_minus.array() -= wt * Anz.row(t).array().square();
            }
        }
        Az2_plus  = Az2_plus.cwiseMax(0.0);
        Az2_minus = Az2_minus.cwiseMax(0.0);

        if (ctx.has_a2_full) {
            if (!ctx.a2p_full || !ctx.a2m_full) {
                error("%s: missing full LDA-uncertainty precompute", __func__);
            }
            Az2_full_plus  = *ctx.a2p_full;
            Az2_full_minus = *ctx.a2m_full;
            for (int t = 0; t < n; ++t) {
                if (xS[t] == 0 || wS[t] <= 0.0) continue;
                const double wt = wS[t];
                if (xS[t] > 0) {
                    Az2_full_plus.noalias()  -= wt * (Anz.row(t).transpose() * Anz.row(t));
                } else {
                    Az2_full_minus.noalias() -= wt * (Anz.row(t).transpose() * Anz.row(t));
                }
            }
            Az2_full_plus  = 0.5 * (Az2_full_plus + Az2_full_plus.transpose());
            Az2_full_minus = 0.5 * (Az2_full_minus + Az2_full_minus.transpose());
            Az2_full_plus  = Az2_full_plus.cwiseMax(0.0);
            Az2_full_minus = Az2_full_minus.cwiseMax(0.0);
        }
    }
}

void MixPoisLog1pSparseProblem::eval_safe(const VectorXd& b, double* f_out,
              VectorXd* g_out, VectorXd* q_out, ArrayXd*  w_out) const
{
    const double eps = opt.optim.eps;
    const double ridge = opt.ridge;
    const VectorXd& oK_ref = *oK;
    Eigen::Map<const VectorXd> yvec(yptr, ysize);

    // ---- ZERO-set exact contributions (diagonal in k) ----
    // ep_k = exp(o_k + b_k), em_k = exp(o_k - b_k)
    ArrayXd ep = (oK_ref.array() + b.array()).min(40.0).exp();
    ArrayXd em = (oK_ref.array() - b.array()).min(40.0).exp();
    // diagonal curvature
    last_qZ.array() = ep * Sz_plus.array() + em * Sz_minus.array() + ridge;

    // ---- NONZERO-set exact contributions ----
    // Eta = xS * b^T + 1 * oK^T
    MatrixXd Eta(n, K);
    Eta.noalias() = xS * b.transpose();
    Eta.rowwise() += oK_ref.transpose();
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

void MixPoisLog1pSparseProblem::compute_se(const VectorXd& b_hat, const MLEOptions& opt, MLEStats& stats) const {
    // Ensure caches correspond to b_hat
    double ftmp; VectorXd gtmp, qtmp; ArrayXd wtmp;
    eval(b_hat, &ftmp, &gtmp, &qtmp, &wtmp);

    const int K = (int)b_hat.size();
    VectorXd sqrt_w = wtmp.sqrt().matrix();
    MatrixXd Xw = Vnz;
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
        const int n_nz = n;
        const int n_eff = N_active;
        double eps = opt.optim.eps;
        ArrayXd lam = lam_cache.array().max(eps);
        Eigen::Map<const VectorXd> yvec(y_ptr(), y_size());
        // slope=1 if not using safe; otherwise use cached slope
        ArrayXd slope = ArrayXd::Ones(n_nz);
        if (slope_cache.size() == n_nz && slope_cache.allFinite()) slope = slope_cache;

        const ArrayXd cx = cS.array() * xS.array();
        ArrayXd s_signed = (1 - yvec.array()/lam) * (slope * cx); // nnz

        // HC adjustments
        double hc_mult = 1.0;
        if (opt.hc_type == 1) { // HC1
            if (n_eff > K) hc_mult = double(n_eff) / double(n_eff - K);
            s_signed *= std::sqrt(hc_mult);
        } else if (opt.hc_type >= 2) { // HC2 / HC3
            // leverage h_i = xw_i^T Hinv xw_i  (xw_i is row i of Xw)
            MatrixXd Z = Xw * Hinv; // n x K
            VectorXd h = (Z.cwiseProduct(Xw)).rowwise().sum(); // n
            ArrayXd denom = (1.0 - h.array()).max(eps);        // avoid blowups
            if (opt.hc_type == 2) s_signed /= denom.sqrt();
            else                  s_signed /= denom;
        }
        ArrayXd s = s_signed.abs();

        // Meat from NONZERO rows
        MatrixXd Vs = Vnz; // a_ik e_ik
        for (int j = 0; j < K; ++j) Vs.col(j).array() *= s;
        MatrixXd B = Vs.transpose() * Vs; // K x K: \sum_i g_i g_i^T

        // ZERO-set meat: full matrix if available, otherwise fallback to diagonal-only
        if (!robust_se_diagonal_only) {
            const VectorXd& oK_ref = *oK;
            VectorXd up = (oK_ref.array() + b_hat.array()).min(40.0).exp().matrix();
            VectorXd um = (oK_ref.array() - b_hat.array()).min(40.0).exp().matrix();
            MatrixXd outp = up * up.transpose();
            MatrixXd outm = um * um.transpose();
            B.array() += hc_mult * (outp.array() * Mz_plus.array()
                                  + outm.array() * Mz_minus.array());
        } else if (Sz2_plus.size() == K && Sz2_minus.size() == K) {
            const VectorXd& oK_ref = *oK;
            ArrayXd op2 = (2.0 * (oK_ref.array() + b_hat.array())).min(40.0);
            ArrayXd om2 = (2.0 * (oK_ref.array() - b_hat.array())).min(40.0);
            ArrayXd ep2 = op2.exp();
            ArrayXd em2 = om2.exp();
            VectorXd Bz_diag = (ep2 * Sz2_plus.array() + em2 * Sz2_minus.array()).matrix();
            B.diagonal().array() += hc_mult * Bz_diag.array();
        }
        if (ctx.has_a_unc) {
            const VectorXd& oK_ref = *oK;
            ArrayXd up = (oK_ref.array() + b_hat.array()).min(40.0).exp();
            ArrayXd um = (oK_ref.array() - b_hat.array()).min(40.0).exp();
            if (!robust_se_diagonal_only && ctx.has_a2_full) {
                MatrixXd outp = up.matrix() * up.matrix().transpose();
                MatrixXd outm = um.matrix() * um.matrix().transpose();
                B.diagonal().array() += hc_mult * (up.square() * Az1_plus.array()
                                                 + um.square() * Az1_minus.array());
                B.array() -= hc_mult * (outp.array() * Az2_full_plus.array()
                                      + outm.array() * Az2_full_minus.array());
            } else {
                ArrayXd Blda_diag = up.square() * (Az1_plus - Az2_plus).array()
                                  + um.square() * (Az1_minus - Az2_minus).array();
                B.diagonal().array() += hc_mult * Blda_diag.array();
            }
        }
        MatrixXd Vrob = Hinv * B * Hinv;
        if (opt.se_stabilize > 0.0) {
            // Effective exposure per topic: E_k = sum_i c_i a_{ik}
            VectorXd E = (*ctx.s1p) + (*ctx.s1m);  // K
            const double tiny = 1e-30;
            ArrayXd gamma = opt.se_stabilize /
                            (opt.se_stabilize + E.array().max(0.0) + tiny);
            VectorXd diag_f = Hinv.diagonal();
            VectorXd diag_r = Vrob.diagonal();
            // Only lift (never shrink)
            VectorXd lift = (diag_f - diag_r).cwiseMax(0.0);
            Vrob.diagonal().array() += gamma * lift.array();
        }
        stats.se_robust = Vrob.diagonal().array().sqrt().matrix();
        if (opt.store_cov) stats.cov_robust = std::move(Vrob);
        else stats.cov_robust.resize(0,0);
    } else {
        stats.se_robust.resize(0);
        stats.cov_robust.resize(0,0);
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
    MixPoisLog1pSparseContext ctx(A, x, c, opt, s1p, s1m, s2p, s2m,
        nullptr, nullptr, N_active);
    MixPoisLog1pSparseProblem P(ctx);
    P.reset_feature(ids, cnts, oK);
    if (b.size() != K) {
        b = VectorXd::Zero(K);
    }
    double final_obj = tron_solve(P, b, opt.optim, stats.optim);
    if (opt.se_flag != 0) {
        P.compute_se(b, opt, stats);
    }
    return final_obj;
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

void MixPoisLogRegProblem::compute_se(const VectorXd& b_hat, const MLEOptions& opt, MLEStats& stats) const
{
    if (opt.se_flag == 0) return;

    const int K = (int)b_hat.size();
    const int N_eff = N_active;
    const double eps = opt.optim.eps;

    // Refresh caches at b_hat (fills V and lam_cache)
    double ftmp; VectorXd gtmp, qtmp; ArrayXd wtmp;
    eval(b_hat, &ftmp, &gtmp, &qtmp, &wtmp);

    // --- Bread: Fisher Hessian H = V^T diag(w) V + jitter ---
    // Here w_i = (c_i x_i)^2 / lambda_i is returned as wtmp by eval()
    VectorXd sqrt_w = wtmp.sqrt().matrix();
    MatrixXd Xw = V; // N x K
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
        const ArrayXd lam = lam_cache.array().max(eps);
        const ArrayXd cx  = (c.array() * x.array());
        ArrayXd s = (y.array() - lam) * (cx / lam);

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
        MatrixXd Vs = V;
        for (int j = 0; j < K; ++j) {
            Vs.col(j).array() *= abs_s.array();
        }
        MatrixXd B = Vs.transpose() * Vs;

        MatrixXd Vrob = Hinv * B * Hinv;
        if (opt.se_stabilize > 0.0) {
            // Effective exposure per topic: E_k = sum_i c_i a_{ik}
            VectorXd E = A.transpose() * c;  // K
            const double tiny = 1e-30;
            ArrayXd gamma = opt.se_stabilize /
                            (opt.se_stabilize + E.array().max(0.0) + tiny);
            VectorXd diag_f = Hinv.diagonal();
            VectorXd diag_r = Vrob.diagonal();
            // Only lift (never shrink)
            VectorXd lift = (diag_f - diag_r).cwiseMax(0.0);
            Vrob.diagonal().array() += gamma * lift.array();
        }
        stats.se_robust = Vrob.diagonal().array().sqrt().matrix();

        if (opt.store_cov) stats.cov_robust = Vrob;
        else stats.cov_robust.resize(0,0);
    } else {
        stats.se_robust.resize(0);
        stats.cov_robust.resize(0,0);
    }
}

double mix_pois_log_mle(const RowMajorMatrixXd& A,
    const VectorXd& y, const VectorXd& x,
    const VectorXd& c, const VectorXd& oK,
    MLEOptions& opt, VectorXd& b, MLEStats& stats, int N_active)
{
    const int K = (int)A.cols();
    if (!opt.optim.b_min && !opt.optim.b_max) {
        double bd = std::log(100.0);
        warning("%s: no bounds specified, assuming FC within 100", __func__);
        opt.optim.set_bounds(-bd, bd, K);
    }
    MixPoisLogRegProblem P(A, y, x, c, oK, opt, N_active);
    if (b.size() != K) {
        b = VectorXd::Zero(K);
    }
    double final_obj = tron_solve(P, b, opt.optim, stats.optim);
    if (opt.se_flag != 0) {
        P.compute_se(b, opt, stats);
    }
    return final_obj;
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
