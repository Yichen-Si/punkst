# pragma once

#include "concurrent_matrix.hpp"
#include "concurrent_tree.hpp"
#include "dynamic_matrix.hpp"
#include "numerical_utils.hpp"
#include "dataunits.hpp"
#include "error.hpp"
#include "utils.h"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>
#include <execution>
#include <cmath>

#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

struct nCrpLeaf {
    Document data;
    std::vector<int32_t> c; // topic assignment on each level
    std::vector<int32_t> z; // level assignment for each uniq word
    int leaf_id = -1;
    bool initialized = false;

    std::vector<uint32_t> reordered_ids; // re-ordered word ids
    std::vector<double>   reordered_cts;
    std::vector<uint32_t> offsets; // begin index in reordered_ids for level l

    nCrpLeaf() {}
    nCrpLeaf(Document& d) : data(std::move(d)) {}
    nCrpLeaf(const Document& d) : data(d) {}

    // Partition data.ids according to z so that words assigned to the same
    // layer are stored consecutively
    void partition_by_z(int L);
};

class HLDA {

public:

    HLDA(int L, int n_features,
        int seed = std::random_device{}(),
        int nThreads = 0, int debug = 0, int verbose = 0,
        int max_k = 1024, std::vector<double> log_gamma = {1e-30},
        int thr_heavy = 50, int thr_prune = 0, int s_resmpc = 5,
        std::vector<double> eta = {1., .5, .25},
        double alpha = 0.2, double m = -1, int D = 1000000) :
        tree_(L, log_gamma, max_k, thr_heavy, thr_prune, debug),
        L_(L), n_features_(n_features),
        seed_(seed), nThreads_(nThreads), debug_(debug), verbose_(verbose),
        max_n_nodes_(max_k), s_resmpc_(s_resmpc),
        eta_(eta), alpha_(alpha), m_(m), total_doc_count_(D) {
        assert(L_ > 0 && n_features_ > 0 && eta_.size() > 0 &&
               max_n_nodes_ > L_ && total_doc_count_ > 0);
        if (m_ > 0)
            assert(alpha_ > 0 && alpha_ < 1);
        set_nthreads(nThreads_);
        if (seed_ <= 0) {
            seed_ = std::random_device{}();
        }
        random_engine_.seed(seed_);
        nlwk_.reserve(L_);
        nlwk_.emplace_back(n_features_, 0);
        nlwk_[0].Grow(1);
        for (int l = 1; l < L_; ++l) {
            nlwk_.emplace_back(n_features_);
        }
        allow_new_topic_ = true;
        allow_branching_.resize(L_, 1);
        n_nodes_ = tree_.GetNumNodes();
        n_heavy_ = tree_.GetNumHeavy();
        offsets_.resize(L_ + 1);
        offsets_[0] = 0;
        for (int l = 0; l < L_; ++l) {
            offsets_[l + 1] = offsets_[l] + n_nodes_[l];
        }
        beta_.reserve(L_);
        logbeta_.reserve(L_);
        for (int l = 0; l < L_; ++l) {
            beta_.emplace_back(n_features_, 1);
            logbeta_.emplace_back(n_features_, 1);
        }
        if (eta_.size() < L_) {
            for (int l = eta_.size(); l < L_; ++l) {
                eta_.push_back(eta_.back());
            }
        }
        generators_.reserve(nThreads_);
        for (int i = 0; i < nThreads_; ++i) {
            generators_.emplace_back(random_engine_(), random_engine_());
        }
        z_prior_.resize(L_, 1);
        if (m_ > 0) {
            z_prior_[0] = alpha_;
            double psum = alpha_;
            for (int l = 1; l < L_-1; ++l) {
                z_prior_[l] = (1 - psum) * alpha;
                psum += z_prior_[l];
            }
            z_prior_[L_-1] = 1 - psum;
            d_z_prior_ = std::discrete_distribution<uint32_t>(z_prior_.begin(), z_prior_.end());
        }
    }

    void fit(std::vector<nCrpLeaf>& docs, int n_iters = 10, int n_mc_iters = 5, int n_mb_iters = 5, int bsize = 512, int csize = 512, int n_init = 1000);

    void sample_c(nCrpLeaf& doc, xorshift& rng, bool dec_count, bool inc_count = true, bool int_z = true);

    void sample_z(nCrpLeaf& doc, bool dec_count, bool inc_count = true);

    void set_nthreads(int nThreads);

private:
    ConcurrentTree tree_;
    // Per level global parameters
    std::vector<ConcurrentColMatrix<double>> nlwk_; // L x W x K_l for rct (rare)
    std::vector<int> n_nodes_; // number of nodes
    std::vector<int> n_heavy_; // number of heavy nodes
    std::vector<int> offsets_;
    std::vector<Matrix<double>> beta_, logbeta_; // L x W x Kl for sct (heavy)
    int nThreads_, seed_;
    int debug_, verbose_;
    int L_, n_features_;
    int s_resmpc_; // num of samples of z to do when approx P(c|w)
    int max_n_nodes_;
    std::mt19937 random_engine_;
    std::vector<xorshift> generators_;
    double m_; // If m_ > 0 \theta \sim GEM(m, \alpha)
    double alpha_; // else \theta \sim Dir(\alpha)
    std::vector<double> eta_; // \beta \sim Dir(\eta)
    int total_doc_count_;
    bool allow_new_topic_;
    MatrixXd nwk_global_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;
    std::vector<double> z_prior_; // P(z=l; m, \alpha)
    std::discrete_distribution<uint32_t> d_z_prior_;

    void fit_onepass(std::vector<nCrpLeaf>& docs, bool integrate_z);

    // single-threaded cgs for initialization
    void initialize(std::vector<nCrpLeaf>& docs, int n = 1000);

    void sample_z_dir(nCrpLeaf& doc, bool dec_count, bool inc_count = true);
    void sample_z_gem(nCrpLeaf& doc, bool dec_count, bool inc_count = true);

    // Given a doc's (z, w) & a freeze of tree, compute the logl of all paths
    // Including new paths stemming from each internal node
    // \log P(c_d | w,z,c) = \log P_{nCRP}(c_d | c_{-d}) + \log P(w_d|c,z,...)
    void compute_path_ll_(nCrpLeaf& doc, xorshift& rng, ConcurrentTree::RetTree& tree, float* pk);

    // Consolidate tree; update beta & logbeta; reset nlwk
    void update_global_();
    // Compute beta & logbeta without changing the list of heavy nodes
    void update_beta_();

    // \sum_{i:zi=l} x_{wi}*\log\beta_{wk} for one doc and all topics on level l
    // nt: number of topics to compute on level l
    // result: store the results for each topic
    void compute_node_ll_beta_(nCrpLeaf& doc, int l, int nt, float* result);
        // the collapsed version (from nwk instead of realized beta)
        // return in addition the log likelihood under a new topic at level l
    float compute_node_ll_nwk_(nCrpLeaf& doc, int l, int offset, int nt, float* result);

    // Update nwk for small/new topics (expensive)
    void update_nwk_(nCrpLeaf& doc, bool add);

    void remap_nwk_(const std::vector<ConcurrentTree::NodeRemap>& remaps);

    void remap_c_(std::vector<nCrpLeaf>& docs);

};
