#include "hlda.hpp"

void nCrpLeaf::partition_by_z(int L) {
    if (z.size() != data.ids.size()) {
        error("%s: z has not been correctly initialized", __func__);
    }
    offsets.resize(L + 1, 0);
    uint32_t n = data.ids.size();
    reordered_ids.resize(n);
    reordered_cts.resize(n);
    for (auto k: z) {
        offsets[k + 1]++;
    } // o[k+1] = number of words assigned to level k
    for (int32_t l = 1; l <= L; ++l) {
        offsets[l] += offsets[l - 1];
    } // o[k] = index of the first word assigned to level k
    for (uint32_t i = 0; i < n; ++i) {
        reordered_ids[offsets[z[i]]] = data.ids[i];
        reordered_cts[offsets[z[i]]] = data.cnts[i];
        offsets[z[i]]++;
    }
    for (int32_t l = L; l > 0; --l) {
        offsets[l] = offsets[l - 1];
    } // restore o[k] to point to the first word assigned to level k
    offsets[0] = 0; // o[L] = n
}

void HLDA::sample_z(nCrpLeaf& doc, bool dec_count, bool inc_count) {
    uint32_t n = doc.data.ids.size();
    std::vector<double> ndl(L_);
    for (uint32_t i = 0; i < n; ++i) {
        ndl[doc.z[i]] += doc.data.cnts[i];
    }
    std::vector<bool> is_small(L_, false);
    for (int l = 0; l < L_; l++) {
        is_small[l] = doc.c[l] >= n_heavy_[l];
    }
    double Weta = n_features_ * eta_;
    std::vector<float> pl(L_);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t w = doc.data.ids[i];
        double   x = doc.data.cnts[i];
        uint32_t l = doc.z[i];
        auto &nwk_ = nlwk_[l];
        if (dec_count && is_small[l]) {
            nwk_.Dec(w, doc.c[l], x);
        }
        ndl[l] -= x;
        for (uint32_t j = 0; j < L_; ++j) {
            if (is_small[j]) {
                pl[j] = (ndl[j] + alpha_) *
                        (nwk_.Get(w, doc.c[j]) + eta_) /
                        (nwk_.GetSum(doc.c[j]) + Weta);
            } else {
                pl[j] = (ndl[j] + alpha_) * beta_[j](w, doc.c[j]);
            }
            std::discrete_distribution<uint32_t> d(pl.begin(), pl.end());
            l = d(random_engine_);
            doc.z[i] = l;
        }
        if (inc_count && is_small[l]) {
            nwk_.Inc(w, doc.c[l], x);
        }
        ndl[l] += x;
    }
}

void HLDA::sample_c(nCrpLeaf& doc, xorshift& rng, bool dec_count, bool inc_count, bool int_z) {
    if (dec_count) {
        update_nwk_(doc, false);
        tree_.DecNumDocs(doc.leaf_id);
    }
    ConcurrentTree::RetTree tree = tree_.GetTree(); // get a freeze of the tree
    int S = int_z ? std::max(s_resmpc_, 1) : 1;
    auto &nodes = tree.nodes;
    uint32_t n = nodes.size();
    std::vector<float> pk(n * S, -1e9f); // initially log(P(c|z,w))
    if (!int_z) {
        compute_path_ll_(doc, rng, tree, pk.data());
    } else {
        for (int s = 0; s < S; ++s) {
            for (auto &l : doc.z) { // l ~ Unif[L]
                l = (uint32_t) (((uint64_t) rng.sample() * L_) >> 32);
            }
            compute_path_ll_(doc, rng, tree, pk.data() + s * n);
        }
    }
    softmax(pk.begin(), pk.end()); // back to probability scale
    // Sample from all paths (including new ones)
    int node_id = discrete_sample(pk.begin(), pk.end(), rng) / S;
    int delta = inc_count ? 1 : 0;
    auto ret = tree_.IncNumDocs(node_id, delta);
    doc.leaf_id = ret.id;
    doc.c = std::move(ret.pos);
    if (inc_count) {
        update_nwk_(doc, true);
    }
}

void HLDA::fit(std::vector<nCrpLeaf>& docs, int n_iters, int n_mc_iters, int n_mb_iters, int bsize) {
    int32_t ndocs = docs.size();
    if (n_mb_iters == 0 || bsize > ndocs / 1.5) {
        bsize = ndocs;
    } else {
        bsize = std::min(bsize, ndocs / 2 + 1);
    }

    std::vector<uint32_t> doc_idx(ndocs);
    std::iota(doc_idx.begin(), doc_idx.end(), 0);

    // Get current tree statistics (freeze for each minibatch)
    n_heavy_ = tree_.GetNumHeavy();
    n_nodes_ = tree_.GetNumNodes();
    offsets_[0] = 0;
    for (int l = 0; l < L_; ++l) {
        offsets_[l + 1] = offsets_[l] + n_nodes_[l];
    }
    int it = 0;
    bool integrate_z = it < n_mc_iters;
    while (it < n_iters) {
        std::shuffle(doc_idx.begin(), doc_idx.end(), random_engine_);
        nwk_global_ = MatrixXd::Zero(n_features_, offsets_[L_]);

        for (uint32_t d_st = 0; d_st < ndocs; d_st += bsize) {
            uint32_t d_ed = std::min(d_st + bsize, (uint32_t) ndocs);
            // parallelized sampling of local parameters
            tbb::parallel_for(tbb::blocked_range<uint32_t>(d_st, d_ed), [&](const tbb::blocked_range<uint32_t>& r) {
                int tid = tbb::this_task_arena::current_thread_index();
                xorshift& rng = generators_[tid];
                for (uint32_t d = r.begin(); d < r.end(); ++d) {
                    auto &doc = docs[doc_idx[d]];
                    if (!doc.initialized) {
                        for (auto &l: doc.z) {
                            l = rng() % L_;
                        }
                    }
                    sample_c(doc, rng, doc.initialized, true, integrate_z);
                    sample_z(doc, true, true);
                    doc.initialized = true;
                }
            });
            // Compute nwk for all nodes
            tbb::combinable<MatrixXd> nwk_acc { // in old node order
                [&]{ return MatrixXd::Zero(n_features_, offsets_[L_]); }
            };
            tbb::combinable<VectorXd> nct_acc { // level marginal
                [&]{ return VectorXd::Zero(L_); }
            };
            tbb::parallel_for(tbb::blocked_range<uint32_t>(d_st, d_ed), [&](const tbb::blocked_range<uint32_t>& r) {
                auto& nwk_local = nwk_acc.local();
                auto& nct_local = nct_acc.local();
                for (uint32_t d = r.begin(); d < r.end(); ++d) {
                    auto &doc = docs[doc_idx[d]];
                    for (int i = 0; i < doc.z.size(); ++i) {
                        uint32_t l = doc.z[i];
                        nwk_local(doc.data.ids[i], offsets_[l] + doc.c[l]) += doc.data.cnts[i];
                        nct_local[l] += doc.data.cnts[i];
                    }
                }
            });
            MatrixXd nwk_local = nwk_acc.combine(
                [](const MatrixXd& a, const MatrixXd& b) {return a + b;}
            );
            VectorXd nct_local = nct_acc.combine(
                [](const VectorXd& a, const VectorXd& b) {return a + b;}
            );

            update_global_(nwk_local);
        }
        it++;
        integrate_z = it < n_mc_iters;
        if (it >= n_mb_iters) {
            bsize = ndocs;
        }
    }
}

void HLDA::compute_node_ll_beta_(nCrpLeaf& doc, int l, int nt, float *result) {
    if (nt <= 0) {return;}
    memset(result, 0, nt * sizeof(float));
    for (uint32_t i = doc.offsets[l]; i < doc.offsets[l + 1]; ++i) {
        auto *betaw = &logbeta_[l](doc.reordered_ids[i], 0);
        for (uint32_t k = 0; k < nt; ++k) {
            result[k] += betaw[k] * doc.reordered_cts[i];
        }
    }
}

float HLDA::compute_node_ll_nwk_(nCrpLeaf& doc, int l, int offset, int nt, float* result) {
    if (offset + nt > nlwk_[l].GetC()) {
        throw std::out_of_range("HLDA::compute_node_ll_nwk_: offset+nt exceeds number of topics on level l");
    }
    memset(result, 0, nt * sizeof(float));
    double Weta = n_features_ * eta_;
    float new_node_ll = 0.0f;
    const auto& nwk = nlwk_[l];
    double ndl = 0.;
    for (uint32_t i = doc.offsets[l]; i < doc.offsets[l + 1]; ++i) {
        double ndlw = doc.reordered_cts[i];
        double ndlw2 = ndlw/2-.5; // approximate sum with midpoint
        ndl += ndlw;
        for (uint32_t k = 0; k < nt; ++k) {
            auto n = nwk.Get(doc.reordered_ids[i], offset + k);
            result[k] += ndlw * logf(n + ndlw2 + eta_);
        }
        new_node_ll += ndlw * logf(ndlw2 + eta_);
    }
    for (uint32_t k = 0; k < nt; ++k) {
        double nk = nwk.GetSum(offset + k);
        result[k] += lgamma(nk + Weta) - lgamma(nk + ndl + Weta);
    }
    return new_node_ll + lgamma(Weta) - lgamma(ndl/2-.5 + Weta);
}

void HLDA::update_nwk_(nCrpLeaf& doc, bool add) {
    for (int l = 0; l < L_; ++l) {
        nlwk_[l].Grow(doc.c[l] + 1); // check if new topics are added
    }
    uint32_t n = doc.z.size();
    if (add) {
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t l = doc.z[i];
            if (doc.c[l] >= n_heavy_[l]) { // rct
                nlwk_[l].Inc(doc.data.ids[i], doc.c[l], doc.data.cnts[i]);
            }
        }
    } else {
        for (uint32_t i = 0; i < n; ++i) {
            uint32_t l = doc.z[i];
            if (doc.c[l] >= n_heavy_[l]) { // rct
                nlwk_[l].Dec(doc.data.ids[i], doc.c[l], doc.data.cnts[i]);
            }
        }
    }
}

void HLDA::compute_path_ll_(nCrpLeaf& doc, xorshift& rng, ConcurrentTree::RetTree& tree, float* pk) {
    doc.partition_by_z(L_);
    // Compute likelihood for slow changing topics (using beta)
    std::vector<std::vector<float>> scores(L_);
    for (int l = 0; l < L_; ++l) {
        uint32_t nheavy = n_heavy_[l]; // num of slow (initiated) topics at l
        scores[l].resize(nheavy);
        #pragma forceinline
        compute_node_ll_beta_(doc, l, nheavy, scores[l].data());
    }
    // Compute likelihood for rapid changing topics (using nlwk)
    auto &nodes = tree.nodes;
    uint32_t n = nodes.size();
    for (int l = 0; l < L_; ++l) {
        auto &score = scores[l];
        uint32_t nheavy = n_heavy_[l];
        uint32_t nsmall = tree.num_nodes[l] - nheavy;
        score.resize(tree.num_nodes[l] + 1);
        #pragma forceinline
        score.back() = compute_node_ll_nwk_(doc, l, nheavy, nsmall, score.data() + nheavy);
        if (!allow_new_topic_) {
            score.back() = -1e20f;
        }
    }
    // Compute path prior
    std::vector<float> sum_logpk(nodes.size()); // logP(w|c) for partial paths
    std::vector<float> empty_pl(L_, 0.0f);
    for (int l = L_ - 2; l >= 0; --l) {
        empty_pl[l] = empty_pl[l+1] + scores[l+1].back();
    }
    for (uint32_t i = 0; i < nodes.size(); i++) { // top-down
        auto &node = nodes[i];
        if (node.depth == 0) {
            sum_logpk[i] = scores[0][node.pos];
        } else {
            sum_logpk[i] = scores[node.depth][node.pos] + sum_logpk[node.parent_id];
        }
        // logP(w|c) + logP(c) for the path from i to root
        pk[i] = sum_logpk[i] + node.log_path_weight;
        if (node.depth < L_ - 1) { // new path from an internal node
            pk[i] += empty_pl[node.depth];
        }
    }
}

void HLDA::update_global_(MatrixXd& nwk_local) {
    // Consolidate the tree
    std::vector<std::vector<int>> pos_map;
    bool node_remap  = tree_.Consolidate(pos_map);
    auto n_nodes_new = tree_.GetNumNodesC();
    if (debug_) {
        ConcurrentTree::RetTree tree = tree_.GetTree();
        notice("%s: consolidated tree", __func__);
        std::cout << tree;
    }
    std::vector<int> offsets_new(L_ + 1, 0);
    for (int l = 0; l < L_; ++l) {
        offsets_new[l + 1] = offsets_new[l] + n_nodes_new[l];
    }
    MatrixXd nwk_new(n_features_, offsets_new[L_]);
    for (int l = 0; l < L_; ++l) {
        for (uint32_t i = 0; i < n_nodes_new[l]; ++i) { // within-level pos
            int i_new = offsets_new[l] + i;
            int k_old = node_remap ? pos_map[l][i] : i;
            int i_old = offsets_[l] + k_old;
            nwk_new.col(i_new) = nwk_local.col(i_old);
            if (k_old < n_nodes_[l]) { // o.w. this is a new node
                nwk_new.col(i_new) += nwk_global_.col(i_old);
            }
        }
    }
    nwk_global_ = std::move(nwk_new);
    VectorXd col_sums = nwk_global_.colwise().sum(); // n_k
    // Compute beta and logbeta
    for (int l = 0; l < L_; ++l) {
        int K = n_nodes_new[l];
        beta_[l].SetC(K, false, true);
        logbeta_[l].SetC(K);
        int o = offsets_new[l];
        std::vector<double> denom(K);
        for (int k = 0; k < K; ++k) { // n_k + W \eta
            denom[k] = 1. / (eta_ * n_features_ + col_sums(o + k));
        }
        int C = beta_[l].GetCcap();
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, n_features_), [&](const tbb::blocked_range<uint32_t>& r) {
            for (uint32_t w = r.begin(); w < r.end(); ++w) {
                for (int k = 0; k < K; ++k) { // E[\beta_{wk}]
                    beta_[l](w, k) = (nwk_global_(w, o + k) + eta_) * denom[k];
                }
                std::transform(
                    beta_[l].Data() + w*C, beta_[l].Data() + w*C + K,
                    logbeta_[l].Data() + w*C, [](float x) {return std::log(x);});
            }
        });
    }
    // Reset nlwk_
    auto n_heavy_new = tree_.GetNumHeavyC();
    for (int l = 0; l < L_; ++l) {
        nlwk_[l].Reset(n_nodes_new[l]);
    }
    if (node_remap || n_heavy_new != n_heavy_) {
        for (int l = 0; l < L_; ++l) {
            int o = offsets_new[l];
            tbb::parallel_for(tbb::blocked_range<uint32_t>(0, n_features_), [&](const tbb::blocked_range<uint32_t>& r) {
                for (uint32_t w = r.begin(); w < r.end(); ++w) {
                for (int k = n_heavy_new[l]; k < n_nodes_new[l]; ++k) {
                    if (nwk_global_(w, o + k) > 1e-6)
                        nlwk_[l].Set(w, k, nwk_global_(w, o + k));
                }
                }
            });
            for (int k = n_heavy_new[l]; k < n_nodes_new[l]; ++k) {
                nlwk_[l].SetSum(k, col_sums(o + k));
            }
        }
    }
    n_heavy_ = tree_.GetNumHeavy();
    n_nodes_ = tree_.GetNumNodes();
    offsets_ = std::move(offsets_new);
    if (tree_.GetSize() >= max_n_nodes_) {
        allow_new_topic_ = false;
    }
}

void HLDA::set_nthreads(int nThreads) {
    nThreads_ = nThreads;
    if (nThreads_ > 0) {
        tbb_ctrl_ = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism,
            std::size_t(nThreads_));
    } else {
        tbb_ctrl_.reset();
    }
}
