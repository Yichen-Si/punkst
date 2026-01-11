#pragma once

#include <vector>
#include <cstdint>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <algorithm>
#include <unordered_map>
#include "utils.h"
#include "numerical_utils.hpp"

// Cache of per-unit summaries for permutation tests
class MultiSliceUnitCache {
public:
    struct Unit {
        float n = 0.0f;        // totalCount (trials)
        uint32_t off = 0;      // offset into feat_ids_/feat_counts_
        uint32_t len = 0;      // number of nonzero features for this unit
    };

    MultiSliceUnitCache(int K, int M, double min_unit_total = 1.0)
        : K_(K), M_(M), min_unit_total_(min_unit_total),
          units_(K_), feat_ids_(K_), feat_counts_(K_)
    {
        if (K_ <= 0) throw std::invalid_argument("K must be positive");
        if (M_ <= 0) throw std::invalid_argument("M must be positive");
    }

    int get_n_slices() const { return K_; }
    int get_n_features() const { return M_; }

    const std::vector<Unit>& slice_units(int k) const {
        if (k < 0 || k >= K_) throw std::out_of_range("slice index out of range");
        return units_[k];
    }
    const std::vector<int32_t>& slice_feat_ids(int k) const {
        if (k < 0 || k >= K_) throw std::out_of_range("slice index out of range");
        return feat_ids_[k];
    }
    const std::vector<double>& slice_feat_counts(int k) const {
        if (k < 0 || k >= K_) throw std::out_of_range("slice index out of range");
        return feat_counts_[k];
    }

    // Cache one unit (pooled across groups)
    void add_unit(int k, double n, const std::unordered_map<int32_t, double>& feat_map) {
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

        U.push_back(Unit{(float)n, off0, len});
    }

    // Merge thread-local cache into this cache.
    void merge_from(const MultiSliceUnitCache& other) {
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

private:
    int K_, M_;
    double min_unit_total_;
    std::vector<std::vector<Unit>> units_;
    std::vector<std::vector<int32_t>> feat_ids_;
    std::vector<std::vector<double>> feat_counts_;
};

class PairwiseBinomRobust {
public:
    PairwiseBinomRobust(int G, int M, double min_unit_total = 1.) : G_(G), M_(M), min_unit_total_(min_unit_total),
        N_(G_, 0.0), C_(G_, 0.0), n_(G_, 0),
        Y_(size_t(G_)*M_, 0.0),
        A_(size_t(G_)*M_, 0.0),
        B_(size_t(G_)*M_, 0.0),
        touched_(M_, 0)
    {
        if (G_ <  2) throw std::invalid_argument("Need at least 2 groups");
        if (M_ <= 0) throw std::invalid_argument("Number of features must be positive");
    }

    struct PairwiseOneResult {
        double beta = 0.0;   // logit(pi1) - logit(pi0)
        double varb = 0.0;   // robust variance of beta
        double pi0  = 0.0;
        double pi1  = 0.0;
        double tot  = 0.0;   // Y0 + Y1
    };

    int get_n_groups()   const { return G_; }
    int get_n_features() const { return M_; }
    const std::vector<double>& get_group_totals() const { return N_; }
    const std::vector<double>& get_group_counts() const { return Y_; }
    const std::vector<int>& get_group_unit_counts() const { return n_; }
    double get_group_totals(int g) const {
        if (g < 0 || g >= G_) throw std::out_of_range("group out of range");
        return N_[g];
    }
    const std::vector<int>& get_active_features() const { return active_; }

    // Add one unit
    void add_unit(int group, double n, const std::vector<int>& feat_ids,
                  const std::vector<double>& feat_counts) {
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

    void add_unit(int group, double n, const std::unordered_map<int32_t, double>& feat_map) {
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

    void merge_from(const PairwiseBinomRobust& other) {
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

    // Build list of pairs if not provided
    static std::vector<std::pair<int,int>> make_all_pairs(int G) {
        std::vector<std::pair<int,int>> pairs;
        pairs.reserve(size_t(G) * (G - 1) / 2);
        for (int g0 = 0; g0 < G; ++g0) {
            for (int g1 = g0 + 1; g1 < G; ++g1)
                pairs.emplace_back(g0, g1);
        }
        return pairs;
    }

    // Compute group effects with closed-form robust SE
    // logit(p_{ikm}) = a_{km} + y_i * b_{km}
    bool compute_one_test(int f, int g0, int g1, PairwiseOneResult& out,
        double min_total_pair, double pi_eps = 1e-8, bool use_hc1 = true) const
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

    // Compute tests for all active features and group pairs
    // emit(feature_id, g0, g1, b, se, pi0, pi1, total_pair_count)
    // pairs: optional list of (g0, g1) contrasts
    // min_total_pair: per-pair filter on Y[g0,f]+Y[g1,f]
    // pi_eps: bound y/n MLEs away from 0 and 1
    template <class EmitFn>
    void compute_tests(EmitFn&& emit, double min_total_pair,
            const std::vector<std::pair<int,int>>& pairs = {},
            double min_or = 1.1, double pi_eps = 1e-8, bool use_hc1 = true) const
    {
        std::vector<std::pair<int,int>> use_pairs;
        if (!pairs.empty()) {
            use_pairs = pairs;
            for (auto [g0,g1] : use_pairs) {
            if (g0 < 0 || g0 >= G_ || g1 < 0 || g1 >= G_ || g0 == g1)
                throw std::invalid_argument("Invalid pair in pairs list.");
            }
        } else {
            use_pairs = make_all_pairs(G_);
        }
        double min_log_or = std::log(min_or);
        PairwiseOneResult result;
        for (int f : active_) {
            for (auto [g0, g1] : use_pairs) {
                if (!compute_one_test(f, g0, g1, result, min_total_pair, pi_eps, use_hc1)) {continue;}
                if (std::abs(result.beta) < min_log_or) {continue;}
                emit(f, g0, g1, result.beta, std::sqrt(result.varb), result.pi0, result.pi1, result.tot);
            }
        }
    }

private:
    int G_, M_;
    double min_unit_total_;

    // Group totals
    std::vector<double> N_;   // sum n
    std::vector<double> C_;   // sum n^2
    std::vector<int>    n_;   // #units

    // Flattened group-feature aggregates at index g*F + f
    std::vector<double> Y_;   // sum y
    std::vector<double> A_;   // sum y^2
    std::vector<double> B_;   // sum n*y

    // Active feature tracking
    std::vector<uint8_t> touched_;
    std::vector<int> active_;
};




class MultiSlicePairwiseBinom {
public:
    MultiSlicePairwiseBinom(int K, int G, int M, double min_unit_total = 1.0)
    : K_(K), G_(G), M_(M),
      slices_(), touched_union_(size_t(M), 0), active_union_()
    {
        if (K_ <= 0) throw std::invalid_argument("K must be positive");
        if (G_ < 2)  throw std::invalid_argument("Need at least 2 groups");
        if (M_ <= 0) throw std::invalid_argument("Number of features must be positive");
        slices_.reserve(K_);
        for (int k = 0; k < K_; ++k) {
            slices_.emplace_back(G_, M_, min_unit_total);
        }
    }

    int get_n_slices()   const { return K_; }
    int get_n_groups()   const { return G_; }
    int get_n_features() const { return M_; }
    const std::vector<int>& get_active_features() const { return active_union_; }

    PairwiseBinomRobust& slice(int k) {
        if (k < 0 || k >= K_)
            throw std::out_of_range("slice index out of range");
        return slices_[k];
    }
    const PairwiseBinomRobust& slice(int k) const {
        if (k < 0 || k >= K_)
            throw std::out_of_range("slice index out of range");
        return slices_[k];
    }

    void merge_from(const MultiSlicePairwiseBinom& other) {
        if (K_ != other.K_ || G_ != other.G_ || M_ != other.M_) {
            throw std::invalid_argument("MultiSlicePairwiseBinom merge dimension mismatch");
        }
        for (int k = 0; k < K_; ++k) {
            slices_[k].merge_from(other.slices_[k]);
        }
    }

    // Call after adding all units
    void finished_adding_data() {
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

    // 1) slice marginal:
    // emit_slice(f, g0, g1, k, beta, se, -log10p, pi0, pi1, tot)
    // 2) global:
    // emit_g(f, g0, g1, ok_count, g, se_g, -log10p)
    // 3) deviation of slice from global:
    // emit_d(f, g0, g1, k, d, se_d, -log10p)
    template <class EmitSlice, class EmitG, class EmitD>
    void compute_tests(EmitSlice&& emit_slice, EmitG&& emit_g, EmitD&& emit_d,
            double min_total_pair_per_slice,
            const std::vector<std::pair<int,int>>& pairs,
            double min_or = 1.1, double pi_eps = 1e-8, bool use_hc1 = true,
            bool renormalize_pi_over_available = true,
            bool emit_marginal_slices = true, bool emit_decomp = true) const
    {
        if (active_union_.empty()) {
            throw std::runtime_error("active_union is empty. Call finalize_active_union() after adding all units.");
        }
        const double min_log_or = std::log(min_or);

        for (auto [g0, g1] : pairs) {
            if (g0 < 0 || g0 >= G_ || g1 < 0 || g1 >= G_ || g0 == g1)
                throw std::invalid_argument("Invalid pair in pairs list.");

            // base weights pi_k ∝ trials in the pair (N_{k,g0}+N_{k,g1})
            std::vector<double> base_pi(K_, 0.0);
            double base_sum = 0.0;
            for (int k = 0; k < K_; ++k) {
                const double wk = slices_[k].get_group_totals(g0) + slices_[k].get_group_totals(g1);
                base_pi[k] = wk;
                base_sum += wk;
            }
            if (base_sum <= 0.0) continue;
            for (int k = 0; k < K_; ++k) base_pi[k] /= base_sum;

            for (int f : active_union_) {

                // gather slice stats for this (f, pair)
                std::vector<double> beta(K_, 0.0), varb(K_, 0.0), pi0(K_, 0.0), pi1(K_, 0.0), tot(K_, 0.0);
                std::vector<uint8_t> ok(K_, 0);

                std::vector<double> pi = base_pi;
                double pisum_ok = 0.0;
                int ok_count = 0;

                for (int k = 0; k < K_; ++k) {
                    PairwiseBinomRobust::PairwiseOneResult r;
                    if (!slices_[k].compute_one_test(f, g0, g1, r,
                        min_total_pair_per_slice, pi_eps, use_hc1)) {
                        continue;
                    }
                    ok[k] = 1;
                    beta[k] = r.beta; varb[k] = r.varb;
                    pi0[k]  = r.pi0; pi1[k]  = r.pi1; tot[k]  = r.tot;
                    pisum_ok += pi[k];
                    ok_count++;
                }
                if (ok_count == 0 || pisum_ok == 0) continue;

                if (renormalize_pi_over_available) {
                    // renormalize pi over slices passed the feature filter
                    for (int k = 0; k < K_; ++k) if (ok[k]) pi[k] /= pisum_ok;
                }

                if (emit_marginal_slices) {
                    for (int k = 0; k < K_; ++k) {
                        if (!ok[k] || std::abs(beta[k]) < min_log_or) continue;
                        const double se = std::sqrt(varb[k]);
                        double z = beta[k] / se;
                        double log10p = -log10_twosided_p_from_z(z);
                        emit_slice(f, g0, g1, k, beta[k], se, log10p, pi0[k], pi1[k], tot[k]);
                    }
                }

                if (!emit_decomp) continue;

                // compute g = Σ pi_k beta_k
                double g = 0.0, varg = 0.0;
                for (int k = 0; k < K_; ++k) {
                    if (!ok[k]) continue;
                    g += pi[k] * beta[k];
                    varg += (pi[k] * pi[k]) * varb[k];
                }
                if (varg <= 0.0) continue;

                const double se_g = std::sqrt(varg);
                double z_g = g / se_g;
                double log10p_g = -log10_twosided_p_from_z(z_g);
                emit_g(f, g0, g1, ok_count, g, se_g, log10p_g);

                if (ok_count < 2) continue;

                // d_k = beta_k - g
                // Var(d_k) ≈ varb_k + varg - 2*pi_k*varb_k
                for (int k = 0; k < K_; ++k) {
                    if (!ok[k]) continue;
                    const double d = beta[k] - g;
                    if (std::abs(d) < min_log_or) {continue;}
                    double vard = varb[k] + varg - 2.0 * pi[k] * varb[k];
                    if (vard <= 0.0) continue;
                    const double se = std::sqrt(vard);
                    double z = beta[k] / se;
                    double log10p = -log10_twosided_p_from_z(z);
                    emit_d(f, g0, g1, k, d, se, log10p);
                }
            }
        }
    }

private:
    int K_, G_, M_;
    std::vector<PairwiseBinomRobust> slices_;
    std::vector<uint8_t> touched_union_;
    std::vector<int> active_union_;
};
