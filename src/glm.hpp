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

#include "Eigen/Dense"
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// Cache of per-unit summaries for permutation tests
class MultiSliceUnitCache {
public:
    struct Unit {
        float n = 0.0f;        // totalCount (trials)
        uint32_t off = 0;      // offset into feat_ids_/feat_counts_
        uint32_t len = 0;      // number of nonzero features for this unit
        int32_t group = -1;    // dataset index
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
        add_unit(k, -1, n, feat_map);
    }

    // Cache one unit with sample label
    void add_unit(int k, int group, double n, const std::unordered_map<int32_t, double>& feat_map);

    // Merge thread-local cache into this cache.
    void merge_from(const MultiSliceUnitCache& other);

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
                  const std::vector<double>& feat_counts);

    void add_unit(int group, double n, const std::unordered_map<int32_t, double>& feat_map);

    void merge_from(const PairwiseBinomRobust& other);

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
        double min_total_pair, double pi_eps = 1e-8, bool use_hc1 = true) const;

    // Compute test for aggregated group lists
    bool compute_one_test_aggregate(int f,
        const std::vector<int32_t>& g0s, const std::vector<int32_t>& g1s,
        PairwiseOneResult& out,
        double min_total_pair, double pi_eps = 1e-8, bool use_hc1 = true) const;

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


struct ContrastPrecomp {
    int K = 0;

    // Fixed per contrast
    RowMajorMatrixXd C0;   // KxK
    RowMajorMatrixXd C1;   // KxK

    // Hyperparameters per contrast
    int    max_iter     = 25;
    double tol          = 1e-6;
    double pi_eps       = 1e-8;
    double lambda_beta  = 1e-4;
    double lambda_alpha = 1e-6;
    double lm_damping   = 1e-4;

    // Scratch buffers reused per feature
    Eigen::VectorXd p0_obs, p1_obs;
    Eigen::VectorXd w0, w0c, w1, w1c;
    Eigen::VectorXd alpha, beta;
    Eigen::VectorXd z0, z1, p0, p1, d0, d1;
    Eigen::VectorXd r0, r1, wr0, wr1, u0, u1;
    Eigen::VectorXd g_alpha, g_beta;

    Eigen::MatrixXd T0, T1;
    Eigen::MatrixXd S0, S1;
    Eigen::MatrixXd H;
    Eigen::VectorXd rhs, delta;

    RowMajorMatrixXd W0C0, W1C1;

    ContrastPrecomp(int K_,
        int max_iter_ = 25, double tol_ = 1e-6, double pi_eps_ = 1e-8,
        double lambda_beta_ = 1e-2, double lambda_alpha_ = 1e-6,
        double lm_damping_ = 1e-4) :
        K(K_), max_iter(max_iter_), tol(tol_), pi_eps(pi_eps_),
        lambda_beta(lambda_beta_), lambda_alpha(lambda_alpha_),
        lm_damping(lm_damping_),
        w0c(K_), w1c(K_) {}

    void init();

    void compute_CtWC() {
        W0C0 = C0; W1C1 = C1;
        W0C0.array().colwise() *= w0.array();
        W1C1.array().colwise() *= w1.array();
        T0.noalias() = C0.transpose() * W0C0;
        T1.noalias() = C1.transpose() * W1C1;
    }

    void reset_w() {
        w0 = w0c; w1 = w1c;
    }
};

struct MultiSliceOneResult {
    int feature = -1;
    // slice_ok[k]=1 if slice k passed min_total_pair and S0,S1 checks
    std::vector<uint8_t> slice_ok;
    // Observed per-slice quantities (from robust binom aggregate)
    Eigen::VectorXd pi0_obs;     // size K
    Eigen::VectorXd pi1_obs;     // size K
    Eigen::VectorXd beta_obs;    // size K, logit(pi1)-logit(pi0)
    Eigen::VectorXd varb_obs;    // size K
    Eigen::VectorXd log10p_obs;  // size K, -log10 p-value from beta_obs/varb_obs
    // Deconvolution output
    bool deconv_ok = false;
    Eigen::VectorXd beta_deconv; // size K (decontaminated beta)
    // Convenience
    double tot_sum = 0.0; // sum_k (Y0_k + Y1_k) for slices that were ok

    MultiSliceOneResult(int K) {resize(K);}
    void resize(int K) {
        slice_ok.assign((size_t)K, uint8_t{0});
        pi0_obs.setZero(K); pi1_obs.setZero(K);
        beta_obs.setZero(K); varb_obs.setZero(K);
        log10p_obs.setConstant(K, -1);
        beta_deconv.setZero(K);
        tot_sum = 0.0; deconv_ok = false;
    }
};

class MultiSlicePairwiseBinom {

private:
    int K_, G_, M_;
    std::vector<PairwiseBinomRobust> slices_;
    std::vector<uint8_t> touched_union_;
    std::vector<int> active_union_;
    std::vector<Eigen::MatrixXd> confusionMatrices_;

public:
    MultiSlicePairwiseBinom(int K, int G, int M, double min_unit_total = 1.0);

    int get_n_slices()   const { return K_; }
    int get_n_groups()   const { return G_; }
    int get_n_features() const { return M_; }
    const std::vector<int>& get_active_features() const { return active_union_; }
    const std::vector<Eigen::MatrixXd>& get_confusion_matrices() const { return confusionMatrices_; }
    void add_to_confusion(int g, const Eigen::MatrixXd& mtx) {
        if (g < 0 || g >= G_)
            throw std::out_of_range("group index out of range");
        if (mtx.rows() != K_ || mtx.cols() != K_)
            throw std::invalid_argument("confusion matrix size mismatch");
        confusionMatrices_[g] += mtx;
    }

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

    void merge_from(const MultiSlicePairwiseBinom& other);

    // Call after adding all units
    void finished_adding_data();

    RowMajorMatrixXd get_mixing_prob(const std::vector<int32_t>& gs) const {
        RowMajorMatrixXd mtx = RowMajorMatrixXd::Zero(K_, K_);
        for (int g : gs) {
            if (g < 0 || g >= G_)
                throw std::out_of_range("group index out of range");
            mtx += confusionMatrices_[g];
        }
        rowNormalizeInPlace(mtx);
        return mtx;
    }

    ContrastPrecomp prepare_contrast(const std::vector<int32_t>& g0s,
        const std::vector<int32_t>& g1s,
        int max_iter = 25, double tol = 1e-6, double pi_eps = 1e-8,
        double lambda_beta = 1e-2, double lambda_alpha = 1e-6,
        double lm_damping = 1e-4) const;

    bool deconvolution(ContrastPrecomp& pc,
        const Eigen::VectorXd& p0_obs_in, const Eigen::VectorXd& p1_obs_in,
        Eigen::VectorXd& beta_out);

    bool compute_one_test_aggregate(int f,
        const std::vector<int32_t>& g0s, const std::vector<int32_t>& g1s,
        ContrastPrecomp& pc, MultiSliceOneResult& out,
        double min_total_pair, double pi_eps, bool use_hc1, double deconv_hit_p);

};
