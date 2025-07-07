# pragma once

#include "concurrent_matrix.hpp"
#include "concurrent_tree.hpp"
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
    std::vector<uint32_t> c; // topic assignment on each level
    std::vector<uint32_t> z; // level assignment for each uniq word
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
        int max_k = 1024, double gamma = 1e-6, int s_resmpc = 5,
        double alpha = 0.2, double eta = 0.1, int D = 1000000) :
        tree_(L, gamma, max_k),
        L_(L), n_features_(n_features),
        seed_(seed), nThreads_(nThreads), debug_(debug), verbose_(verbose),
        max_n_nodes_(max_k), gamma_(gamma), s_resmpc_(s_resmpc),
        alpha_(alpha), eta_(eta), total_doc_count_(D)
    {
        set_nthreads(nThreads_);
        if (seed_ <= 0) {
            seed_ = std::random_device{}();
        }
        random_engine_.seed(seed_);
        nlwk_.reserve(L_);
        for (int l = 0; l < L_; ++l) {
            nlwk_.emplace_back(n_features_);
        }
        allow_new_topic_ = true;
    }

    void fit(std::vector<nCrpLeaf>& docs, int n_iters = 10, int n_mc_iters = 5, int n_mb_iters = 5, int bsize = 512);
    void sample_c(nCrpLeaf& doc, xorshift& rng, bool dec_count, bool inc_count = true, bool int_z = true);
    void sample_z(nCrpLeaf& doc, bool dec_count, bool inc_count = true);

    void initialize();

    void set_nthreads(int nThreads);

private:
    ConcurrentTree tree_;
    // Per level global parameters
    std::vector<ConcurrentMatrix<double>> nlwk_; // L x W x K_l for rct (rare)
    std::vector<int> n_nodes_; // number of nodes
    std::vector<int> n_heavy_; // number of heavy nodes
    std::vector<int> offsets_;
    std::vector<Matrix<float>> beta_, logbeta_; // L x W x Kl for sct (heavy)
    int nThreads_, seed_;
    int debug_, verbose_;
    int L_, n_features_;
    int s_resmpc_; // num of samples of z to do when approx P(c|w)
    int max_n_nodes_;
    std::mt19937 random_engine_;
    std::vector<xorshift> generators_;
    double alpha_, eta_, gamma_;
    int total_doc_count_;
    bool allow_new_topic_;
    MatrixXd nwk_global_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;

    // Given a doc & a freeze of tree, compute the log likelihood of all paths
    // Including new paths stemming from each internal node
    // \log P(c_d | w,z,c) = \log P_{nCRP}(c_d | c_{-d}) + \log P(w_d|c,z,...)
    void compute_path_ll_(nCrpLeaf& doc, xorshift& rng, ConcurrentTree::RetTree& tree, float* pk);

    // Consolidate tree; update beta & logbeta; reset nlwk
    void update_global_(MatrixXd& nwk_local);

    // \sum_{i:zi=l} x_{wi}*\log\beta_{wk} for one doc and all topics on level l
    // nt: number of topics to compute on level l
    // result: store the results for each topic
    void compute_node_ll_beta_(nCrpLeaf& doc, int l, int nt, float* result);
        // the collapsed version (from nwk instead of realized beta)
        // return in addition the log likelihood under a new topic
    float compute_node_ll_nwk_(nCrpLeaf& doc, int l, int offset, int nt, float* result);

    // Update nwk for small/new topics (expensive)
    void update_nwk_(nCrpLeaf& doc, bool add);

};
