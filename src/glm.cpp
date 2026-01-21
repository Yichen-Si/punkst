#include "glm.hpp"

void MultiSliceUnitCache::add_unit(int k, int group, double n,
    const std::unordered_map<int32_t, double>& feat_map) {
    if (k < 0 || k >= K_) throw std::out_of_range("slice index out of range");
    if (n <= 0.0 || n < min_unit_total_) return;

    auto& U = units_[k];
    auto& F = feat_ids_[k];
    auto& C = feat_counts_[k];

    const uint32_t off0 = (uint32_t)F.size();
    uint32_t len = 0;

    for (const auto& kv : feat_map) {
        const int f = (int)kv.first;
        const double y = kv.second;
        if (y <= 0.0) continue;
        if (f < 0 || f >= M_) continue;
        F.push_back((int32_t)f);
        C.push_back(y);
        ++len;
    }

    U.push_back(Unit{(float)n, off0, len, group});
}

void MultiSliceUnitCache::merge_from(const MultiSliceUnitCache& other) {
    if (K_ != other.K_ || M_ != other.M_) {
        throw std::invalid_argument("MultiSliceUnitCache merge dimension mismatch");
    }
    for (int k = 0; k < K_; ++k) {
        auto& U = units_[k];
        auto& F = feat_ids_[k];
        auto& C = feat_counts_[k];

        const uint32_t base_off = (uint32_t)F.size();
        // append features
        F.insert(F.end(), other.feat_ids_[k].begin(), other.feat_ids_[k].end());
        C.insert(C.end(), other.feat_counts_[k].begin(), other.feat_counts_[k].end());
        // append units with offset shift
        U.reserve(U.size() + other.units_[k].size());
        for (const auto& u : other.units_[k]) {
            Unit v = u;
            v.off += base_off;
            U.push_back(v);
        }
    }
}

void PairwiseBinomRobust::add_unit(int group, double n, const std::vector<int>& feat_ids, const std::vector<double>& feat_counts) {
    if (group < 0 || group >= G_) throw std::out_of_range("group out of range");
    if (feat_ids.size() != feat_counts.size())
        throw std::invalid_argument("feat_ids and feat_counts size mismatch");
    if (n <= 0.0 || n < min_unit_total_) return;
    N_[group] += n;
    C_[group] += n*n;
    n_[group] += 1;
    const int base = group * M_;
    for (size_t k = 0; k < feat_ids.size(); ++k) {
        const double y = feat_counts[k];
        if (y <= 0.0) continue;
        const int f = feat_ids[k];
        if (f < 0 || f >= M_) throw std::out_of_range("feature id out of range");
        if (!touched_[f]) { touched_[f] = 1; active_.push_back(f); }
        const double y2 = y*y;
        Y_[base + f] += y;
        A_[base + f] += y2;
        B_[base + f] += n * y;
    }
}

void PairwiseBinomRobust::add_unit(int group, double n,
    const std::unordered_map<int32_t, double>& feat_map) {
    if (group < 0 || group >= G_) throw std::out_of_range("group out of range");
    if (n <= 0.0 || n < min_unit_total_) return;
    N_[group] += n;
    C_[group] += n*n;
    n_[group] += 1;
    const int base = group * M_;
    for (const auto& kv : feat_map) {
        const int f = kv.first;
        const double y = kv.second;
        if (y <= 0.0) continue;
        if (f < 0 || f >= M_) throw std::out_of_range("feature id out of range");
        if (!touched_[f]) { touched_[f] = 1; active_.push_back(f); }
        const double y2 = y*y;
        Y_[base + f] += y;
        A_[base + f] += y2;
        B_[base + f] += n * y;
    }
}

void PairwiseBinomRobust::merge_from(const PairwiseBinomRobust& other) {
    if (G_ != other.G_ || M_ != other.M_) {
        throw std::invalid_argument("PairwiseBinomRobust merge dimension mismatch");
    }
    for (int g = 0; g < G_; ++g) {
        N_[g] += other.N_[g];
        C_[g] += other.C_[g];
        n_[g] += other.n_[g];
    }
    for (int f : other.active_) {
        if (!touched_[f]) { touched_[f] = 1; active_.push_back(f); }
        for (int g = 0; g < G_; ++g) {
            const int idx = g * M_ + f;
            Y_[idx] += other.Y_[idx];
            A_[idx] += other.A_[idx];
            B_[idx] += other.B_[idx];
        }
    }
}

bool PairwiseBinomRobust::compute_one_test(int f, int g0, int g1,
    PairwiseOneResult& out,
    double min_total_pair, double pi_eps, bool use_hc1) const
{
    const double N0 = N_[g0], N1 = N_[g1];
    if (N0 <= 0.0 || N1 <= 0.0) return false;

    const int i0 = g0 * M_ + f;
    const int i1 = g1 * M_ + f;

    const double Y0 = Y_[i0], Y1 = Y_[i1];
    out.tot = Y0 + Y1;
    if (out.tot < min_total_pair) return false;

    const double pi0 = clamp(Y0 / N0, pi_eps, 1.0 - pi_eps);
    const double pi1 = clamp(Y1 / N1, pi_eps, 1.0 - pi_eps);
    out.pi0  = pi0;
    out.pi1  = pi1;
    out.beta = logit(pi1) - logit(pi0);

    const double S0 = N0 * (pi0 * (1.0 - pi0));
    const double S1 = N1 * (pi1 * (1.0 - pi1));
    if (S0 <= 0.0 || S1 <= 0.0) return false;

    const double R0 = A_[i0] - 2.0*pi0*B_[i0] + (pi0*pi0)*C_[g0];
    const double R1 = A_[i1] - 2.0*pi1*B_[i1] + (pi1*pi1)*C_[g1];

    out.varb = (R0/(S0*S0)) + (R1/(S1*S1));
    if (out.varb <= 0.0) return false;

    if (use_hc1) {
        const int m_pair = n_[g0] + n_[g1];
        const int df = std::max(1, m_pair - 2);
        out.varb *= (double)m_pair / (double)df;
    }
    return true;
}

bool PairwiseBinomRobust::compute_one_test_aggregate(int f,
    const std::vector<int32_t>& g0s, const std::vector<int32_t>& g1s,
    PairwiseOneResult& out,
    double min_total_pair, double pi_eps, bool use_hc1) const
{
    if (g0s.empty() || g1s.empty()) return false;

    double N0 = 0.0, N1 = 0.0;
    double C0 = 0.0, C1 = 0.0;
    int n0 = 0, n1 = 0;
    double Y0 = 0.0, Y1 = 0.0;
    double A0 = 0.0, A1 = 0.0;
    double B0 = 0.0, B1 = 0.0;

    for (int g : g0s) {
        if (g < 0 || g >= G_) throw std::out_of_range("group out of range");
        N0 += N_[g];
        C0 += C_[g];
        n0 += n_[g];
        const int idx = g * M_ + f;
        Y0 += Y_[idx];
        A0 += A_[idx];
        B0 += B_[idx];
    }
    for (int g : g1s) {
        if (g < 0 || g >= G_) throw std::out_of_range("group out of range");
        N1 += N_[g];
        C1 += C_[g];
        n1 += n_[g];
        const int idx = g * M_ + f;
        Y1 += Y_[idx];
        A1 += A_[idx];
        B1 += B_[idx];
    }

    if (N0 <= 0.0 || N1 <= 0.0) return false;

    out.tot = Y0 + Y1;
    if (out.tot < min_total_pair) return false;

    const double pi0 = clamp(Y0 / N0, pi_eps, 1.0 - pi_eps);
    const double pi1 = clamp(Y1 / N1, pi_eps, 1.0 - pi_eps);
    out.pi0  = pi0;
    out.pi1  = pi1;
    out.beta = logit(pi1) - logit(pi0);

    const double S0 = N0 * (pi0 * (1.0 - pi0));
    const double S1 = N1 * (pi1 * (1.0 - pi1));
    if (S0 <= 0.0 || S1 <= 0.0) return false;

    const double R0 = A0 - 2.0*pi0*B0 + (pi0*pi0)*C0;
    const double R1 = A1 - 2.0*pi1*B1 + (pi1*pi1)*C1;

    out.varb = (R0/(S0*S0)) + (R1/(S1*S1));
    if (out.varb <= 0.0) return false;

    if (use_hc1) {
        const int m_pair = n0 + n1;
        const int df = std::max(1, m_pair - 2);
        out.varb *= (double)m_pair / (double)df;
    }
    return true;
}

void ContrastPrecomp::init() {
    if (C0.rows() != K || C0.cols() != K || C1.rows() != K || C1.cols() != K)
        throw std::invalid_argument("ContrastPrecomp::init: C size mismatch");
    if (w0c.size() != K || w1c.size() != K)
        throw std::invalid_argument("ContrastPrecomp::init: weight vector size mismatch");
    // allocate scratch
    p0_obs.resize(K); p1_obs.resize(K);
    alpha.resize(K); beta.resize(K);
    z0.resize(K); z1.resize(K);
    p0.resize(K); p1.resize(K);
    d0.resize(K); d1.resize(K);
    r0.resize(K); r1.resize(K);
    wr0.resize(K); wr1.resize(K);
    u0.resize(K); u1.resize(K);
    g_alpha.resize(K); g_beta.resize(K);
    S0.resize(K, K); S1.resize(K, K);
    H.resize(2*K, 2*K);
    rhs.resize(2*K);
    delta.resize(2*K);

    w0 = w0c; w1 = w1c;
    compute_CtWC();
}

MultiSlicePairwiseBinom::MultiSlicePairwiseBinom(int K, int G, int M, double min_unit_total)
    : K_(K), G_(G), M_(M),
      slices_(), touched_union_(size_t(M), 0), active_union_()
{
    if (K_ <= 0) throw std::invalid_argument("K must be positive");
    if (G_ <  2) throw std::invalid_argument("Need at least 2 groups");
    if (M_ <= 0) throw std::invalid_argument("Number of features must be positive");
    slices_.reserve(K_);
    for (int k = 0; k < K_; ++k) {
        slices_.emplace_back(G_, M_, min_unit_total);
    }
    confusionMatrices_.assign(G, Eigen::MatrixXd::Zero(K_, K_));
}

void MultiSlicePairwiseBinom::merge_from(const MultiSlicePairwiseBinom& other) {
    if (K_ != other.K_ || G_ != other.G_ || M_ != other.M_) {
        throw std::invalid_argument("MultiSlicePairwiseBinom merge dimension mismatch");
    }
    for (int k = 0; k < K_; ++k) {
        slices_[k].merge_from(other.slices_[k]);
    }
    for (int g = 0; g < G_; ++g) {
        confusionMatrices_[g] += other.confusionMatrices_[g];
    }
}

void MultiSlicePairwiseBinom::finished_adding_data() {
    std::fill(touched_union_.begin(), touched_union_.end(), uint8_t{0});
    active_union_.clear();
    for (int k = 0; k < K_; ++k) {
        for (const int f : slices_[k].get_active_features()) {
            if (!touched_union_[f]) {
                touched_union_[f] = 1;
                active_union_.push_back(f);
            }
        }
    }
    std::sort(active_union_.begin(), active_union_.end());
}

ContrastPrecomp MultiSlicePairwiseBinom::prepare_contrast(
    const std::vector<int32_t>& g0s, const std::vector<int32_t>& g1s,
    int max_iter, double tol, double pi_eps,
    double lambda_beta, double lambda_alpha, double lm_damping) const
{
    if (g0s.empty() || g1s.empty()) {
        throw std::invalid_argument("prepare_contrast requires non-empty group lists");
    }
    for (int32_t g : g0s) {
        if (g < 0 || g >= G_) throw std::out_of_range("group index out of range");
    }
    for (int32_t g : g1s) {
        if (g < 0 || g >= G_) throw std::out_of_range("group index out of range");
    }

    ContrastPrecomp pc(K_, max_iter, tol, pi_eps, lambda_beta, lambda_alpha, lm_damping);
    pc.C0 = get_mixing_prob(g0s);
    pc.C1 = get_mixing_prob(g1s);

    for (int k = 0; k < K_; ++k) {
        double N0k = 0.0;
        double N1k = 0.0;
        for (int32_t g : g0s) N0k += slices_[k].get_group_totals(g);
        for (int32_t g : g1s) N1k += slices_[k].get_group_totals(g);
        pc.w0c(k) = std::max(0.0, N0k);
        pc.w1c(k) = std::max(0.0, N1k);
    }
    pc.init();
    return pc;
}

// -------- Deconvolution (jointly adjust parameters across slices) --------
// Model:
//   p0_true = sigmoid(alpha - 0.5*beta)
//   p1_true = sigmoid(alpha + 0.5*beta)
//   p0_obs  ~ C0 * p0_true
//   p1_obs  ~ C1 * p1_true
//   beta is the desired logit-diff per slice: logit(p1_true)-logit(p0_true)
// Weighted NLS + ridge + LM damping
// Returns false on failure
bool MultiSlicePairwiseBinom::deconvolution(ContrastPrecomp& pc,
    const Eigen::VectorXd& p0_obs_in, const Eigen::VectorXd& p1_obs_in,
    Eigen::VectorXd& beta_out)
{
    const int K = pc.K;
    if (p0_obs_in.size() != K || p1_obs_in.size() != K) return false;

    // Clamp inputs into pc buffers
    double wsum = 0.0;
    for (int k = 0; k < K; ++k) {
        pc.p0_obs(k) = clamp(p0_obs_in(k), pc.pi_eps, 1.0 - pc.pi_eps);
        pc.p1_obs(k) = clamp(p1_obs_in(k), pc.pi_eps, 1.0 - pc.pi_eps);
        wsum += pc.w0(k) + pc.w1(k);
    }
    if (!(wsum > 0.0)) return false;

    // init alpha/beta
    for (int k = 0; k < K; ++k) {
        const double t0 = logit(pc.p0_obs(k));
        const double t1 = logit(pc.p1_obs(k));
        pc.alpha(k) = 0.5 * (t0 + t1);
        pc.beta(k)  = (t1 - t0);
    }

    const int P = 2 * K;
    for (int it = 0; it < pc.max_iter; ++it) {
        pc.z0 = pc.alpha - 0.5 * pc.beta;
        pc.z1 = pc.alpha + 0.5 * pc.beta;

        for (int k = 0; k < K; ++k) {
            pc.p0(k) = sigmoid_stable(pc.z0(k));
            pc.p1(k) = sigmoid_stable(pc.z1(k));
            pc.d0(k) = pc.p0(k) * (1.0 - pc.p0(k));
            pc.d1(k) = pc.p1(k) * (1.0 - pc.p1(k));
        }

        pc.r0.noalias() = (pc.C0 * pc.p0) - pc.p0_obs;
        pc.r1.noalias() = (pc.C1 * pc.p1) - pc.p1_obs;

        pc.wr0 = pc.w0.array() * pc.r0.array();
        pc.wr1 = pc.w1.array() * pc.r1.array();

        pc.u0.noalias() = pc.C0.transpose() * pc.wr0;
        pc.u1.noalias() = pc.C1.transpose() * pc.wr1;

        pc.g_alpha = (pc.d0.array() * pc.u0.array()).matrix()
                   + (pc.d1.array() * pc.u1.array()).matrix()
                   + pc.lambda_alpha * pc.alpha;

        pc.g_beta  = ((-0.5 * pc.d0.array() * pc.u0.array()) +
                      ( 0.5 * pc.d1.array() * pc.u1.array())).matrix()
                   + pc.lambda_beta * pc.beta;

        pc.S0 = pc.T0;
        pc.S0.array().colwise() *= pc.d0.array();
        pc.S0.array().rowwise() *= pc.d0.transpose().array();

        pc.S1 = pc.T1;
        pc.S1.array().colwise() *= pc.d1.array();
        pc.S1.array().rowwise() *= pc.d1.transpose().array();

        pc.H.setZero();
        pc.H.topLeftCorner(K, K) = pc.S0 + pc.S1;
        pc.H.topRightCorner(K, K) = 0.5 * (pc.S1 - pc.S0);
        pc.H.bottomLeftCorner(K, K) = pc.H.topRightCorner(K, K);
        pc.H.bottomRightCorner(K, K) = 0.25 * (pc.S0 + pc.S1);
        pc.H.topLeftCorner(K, K).diagonal().array()     += pc.lambda_alpha;
        pc.H.bottomRightCorner(K, K).diagonal().array() += pc.lambda_beta;

        const double tr = pc.H.diagonal().sum();
        const double mu = pc.lm_damping * (tr / (double)P + 1.0);
        pc.H.diagonal().array() += mu;

        pc.rhs.head(K) = -pc.g_alpha;
        pc.rhs.tail(K) = -pc.g_beta;

        Eigen::LDLT<Eigen::MatrixXd> ldlt(pc.H);
        if (ldlt.info() != Eigen::Success) return false;

        pc.delta = ldlt.solve(pc.rhs);
        if (!pc.delta.allFinite()) return false;

        pc.alpha += pc.delta.head(K);
        pc.beta  += pc.delta.tail(K);

        if (pc.delta.squaredNorm() < pc.tol * pc.tol) break;
    }

    beta_out = pc.beta;
    return beta_out.allFinite();
}

bool MultiSlicePairwiseBinom::compute_one_test_aggregate(int f,
    const std::vector<int32_t>& g0s, const std::vector<int32_t>& g1s,
    ContrastPrecomp& pc, MultiSliceOneResult& out,
    double min_total_pair, double pi_eps, bool use_hc1, double deconv_hit_p)
{
    if (f < 0 || f >= M_) throw std::out_of_range("feature id out of range");
    if (g0s.empty() || g1s.empty()) return false;
    if (pc.K != K_) throw std::invalid_argument("precomp K mismatch");

    out.feature = f;
    out.resize(K_);
    out.tot_sum = 0.0;

    int n_ok = 0;
    int n_hit = 0;
    bool check_hits = (deconv_hit_p > 0.0) && std::isfinite(deconv_hit_p);
    double log10_thresh = 0.0;
    if (check_hits) {
        log10_thresh = (deconv_hit_p >= 1.0) ? 0.0 : -std::log10(deconv_hit_p);
    }
    for (int k = 0; k < K_; ++k) {
        PairwiseBinomRobust::PairwiseOneResult r;
        const bool ok = slices_[k].compute_one_test_aggregate(
            f, g0s, g1s, r, min_total_pair, pi_eps, use_hc1
        );

        if (!ok) {
            out.slice_ok[(size_t)k] = 0;
            out.pi0_obs(k)  = 0.5;
            out.pi1_obs(k)  = 0.5;
            out.beta_obs(k) = 0.0;
            out.varb_obs(k) = std::numeric_limits<double>::infinity();
            out.log10p_obs(k) = -std::numeric_limits<double>::infinity();
            continue;
        }

        out.slice_ok[(size_t)k] = 1;
        out.pi0_obs(k)  = r.pi0;
        out.pi1_obs(k)  = r.pi1;
        out.beta_obs(k) = r.beta;
        out.varb_obs(k) = r.varb;
        out.tot_sum += r.tot;
        ++n_ok;

        const double se = std::sqrt(out.varb_obs(k));
        if (se > 0.0 && std::isfinite(se)) {
            const double z = out.beta_obs(k) / se;
            out.log10p_obs(k) = -log10_twosided_p_from_z(z);
        } else {
            out.log10p_obs(k) = -1.0;
        }
        if (check_hits && out.log10p_obs(k) >= log10_thresh) {
            ++n_hit;
        }

    }

    if (n_ok == 0) return false;

    out.beta_deconv = out.beta_obs;
    out.deconv_ok = false;

    const bool do_deconv = check_hits && (n_hit >= 2);
    if (do_deconv) {
        pc.reset_w();
        for (int k = 0; k < K_; ++k) {
            if (!out.slice_ok[(size_t)k]) {
                pc.w0(k) = 0.0;
                pc.w1(k) = 0.0;
            }
        }
        pc.compute_CtWC();
        Eigen::VectorXd beta_deconv(K_);
        const bool ok = deconvolution(pc, out.pi0_obs, out.pi1_obs, beta_deconv);
        if (ok) {
            out.beta_deconv = beta_deconv;
            out.deconv_ok = true;
        }
    }

    return true;
}
