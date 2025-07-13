#include "hlda.hpp"

void nCrpLeaf::partition_by_z(int L) {
    if (z.size() != data.ids.size()) {
        error("%s: z has not been correctly initialized", __func__);
    }
    std::fill(offsets.begin(), offsets.end(), 0);
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
    if (m_ > 0) {
        sample_z_gem(doc, dec_count, inc_count);
    } else {
        sample_z_dir(doc, dec_count, inc_count);
    }
}

void HLDA::sample_z_dir(nCrpLeaf& doc, bool dec_count, bool inc_count) {
    uint32_t n = doc.data.ids.size();
    std::vector<bool> is_small(L_, false);
    for (int l = 0; l < L_; l++) {
        is_small[l] = doc.c[l] >= n_heavy_[l];
    }
    std::vector<double> ndl(L_);
    for (uint32_t i = 0; i < n; ++i) {
        ndl[doc.z[i]] += doc.data.cnts[i];
    }
    std::vector<float> pl(L_);
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t w = doc.data.ids[i];
        double   x = doc.data.cnts[i];
        uint32_t l = doc.z[i];
        if (dec_count && is_small[l]) {
            nlwk_[l].Dec(w, doc.c[l], x);
        }
        ndl[l] -= x;
        for (uint32_t j = 0; j < L_; ++j) {
            pl[j] = ndl[j] + alpha_;
            if (is_small[j]) {
                pl[j] *= (nlwk_[j].Get(w, doc.c[j]) + eta_[j]) /
                         (nlwk_[j].GetSum(doc.c[j]) + eta_[j] * n_features_);
            } else {
                pl[j] *= beta_[j](w, doc.c[j]);
            }
        }
        std::discrete_distribution<uint32_t> d(pl.begin(), pl.end());
        l = d(random_engine_);
        doc.z[i] = l;
        if (inc_count && is_small[l]) {
            nlwk_[l].Inc(w, doc.c[l], x);
        }
        ndl[l] += x;
    }
// std::cout << "sample_z done" << std::endl;
}

void HLDA::sample_z_gem(nCrpLeaf& doc, bool dec_count, bool inc_count) {
    uint32_t n = doc.data.ids.size();
    std::vector<bool> is_small(L_);
    for (int l = 0; l < L_; l++) {
        is_small[l] = doc.c[l] >= n_heavy_[l];
    }
    std::vector<double> ndl(L_, 0);
    std::vector<double> ndl_ge(L_+1, 0);
    for (uint32_t i = 0; i < n; ++i) {
        ndl[doc.z[i]] += doc.data.cnts[i];
    }
    for (int l = L_-1; l >= 0; l--) {
        ndl_ge[l] = ndl_ge[l + 1] + ndl[l];
    }
    std::vector<float> p_lt(L_, 1); // P(z>=l|...)=\prod_{l'<l} ...
    for (int l = 1; l < L_; l++) {
        p_lt[l] = p_lt[l-1]*(ndl_ge[l]+(1-alpha_)*m_)/(ndl_ge[l-1]+m_);
    }
    std::vector<float> pl(L_); // P(z=l|w,c)
    int32_t ndelta = 0;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t w = doc.data.ids[i];
        double   x = doc.data.cnts[i];
        uint32_t l = doc.z[i];
        if (dec_count && is_small[l]) {
            nlwk_[l].Dec(w, doc.c[l], x);
        }
        ndl[l] -= x;
        ndl_ge[l] -= x;
        for (uint32_t j = 0; j < L_; ++j) {
            pl[j] = (ndl[j] + alpha_ * m_) / (ndl_ge[j] + m_) * p_lt[j];
            if (is_small[j]) {
                pl[j] *= (nlwk_[j].Get(w, doc.c[j]) + eta_[j]) /
                         (nlwk_[j].GetSum(doc.c[j]) + eta_[j] * n_features_);
            } else {
                pl[j] *= beta_[j](w, doc.c[j]);
            }
        }
        std::discrete_distribution<uint32_t> d(pl.begin(), pl.end());
        uint32_t l1 = d(random_engine_);
        ndelta += (l1 != l);
        l = l1;
        doc.z[i] = l;
        if (inc_count && is_small[l]) {
            nlwk_[l].Inc(w, doc.c[l], x);
        }
        ndl[l] += x;
        ndl_ge[l] += x;
        if ((ndelta+1) % L_ == 0) {
            for (int l = 1; l < L_; l++) {
                p_lt[l] = p_lt[l-1]*(ndl_ge[l]+(1-alpha_)*m_)/(ndl_ge[l-1]+m_);
            }
        }
    }
// std::cout << "sample_z done" << std::endl;
}

void HLDA::sample_c(nCrpLeaf& doc, xorshift& rng, bool dec_count, bool inc_count, bool int_z) {
    assert(int_z || doc.z.size() == doc.data.ids.size());
    if (doc.z.size() != doc.data.ids.size()) {
        doc.z.resize(doc.data.ids.size());
    }
    if (dec_count) {
        update_nwk_(doc, false);
        tree_.DecNumDocs(doc.leaf_id);
    }
    ConcurrentTree::RetTree tree = tree_.GetTree(); // get a freeze of the tree
    int S = int_z ? std::max(s_resmpc_, 1) : 1;
    auto &nodes = tree.nodes;
// std::cout << "got tree " << nodes.size() << std::endl;
    uint32_t n = nodes.size();
    std::vector<float> pk(n * S, -1e9f); // initially storing log(P(c|z,w))
    if (!int_z) {
        compute_path_ll_(doc, rng, tree, pk.data());
    } else {
        for (int s = 0; s < S; ++s) {
            if (m_ > 0) { // z ~ E[GEM(m, \alpha)]
                for (auto &l : doc.z) {
                    l = d_z_prior_(rng);
                }
            } else { // z ~ Unif[L]
                for (auto &l : doc.z) {
                    l = (uint32_t) (((uint64_t) rng.sample() * L_) >> 32);
                }
            }
            compute_path_ll_(doc, rng, tree, pk.data() + s * n);
        }
    }
    softmax(pk.begin(), pk.end()); // back to probability scale
// if (debug_) {
//     std::vector<float> pk_nodes(n, 0);
//     float sum = 0;
//     for (int i = 0; i < n; ++i) {
//         for (int s = 0; s < S; ++s) {
//             pk_nodes[i] += pk[s * n + i];
//         }
//         sum += pk_nodes[i];
//     }
//     std::cout << "pk " << sum << std::endl;
//     for (int l = 0; l < L_; ++l) {
//         std::cout << "l=" << l << ": ";
//         for (int i = 0; i < n; ++i) {
//             if (nodes[i].depth == l)
//                 std::cout << nodes[i].n_docs << ", " << pk_nodes[i] / sum << "; ";
//         }
//         std::cout << std::endl;
//     }
// }
    // Sample from all paths (including new ones)
    int node_id = discrete_sample(pk.begin(), pk.end(), rng) % n;
    int delta = inc_count ? 1 : 0;
    auto ret = tree_.IncNumDocs(node_id, delta);
    doc.leaf_id = ret.id;
    doc.c = std::move(ret.pos);
// std::cout << "sampled node_id = " << node_id << "->" << ret.id << std::endl;
    if (inc_count) {
        update_nwk_(doc, true);
    }
// std::cout << "sample_c done" << std::endl;
}

void HLDA::fit(std::vector<nCrpLeaf>& docs, int n_iters, int n_mc_iters, int n_mb_iters, int bsize, int csize, int n_init) {
    int32_t ndocs = docs.size();
    if (n_mb_iters == 0 || bsize > ndocs / 1.5) {
        bsize = ndocs;
        csize = ndocs;
    } else {
        bsize = std::min(bsize, ndocs / 2 + 1);
    }
    std::vector<uint32_t> doc_idx(ndocs);
    std::iota(doc_idx.begin(), doc_idx.end(), 0);
    std::shuffle(doc_idx.begin(), doc_idx.end(), random_engine_);

    // single-thread CGS
    debug("%s: initializing with %d documents using CGS", __func__, n_init);
    initialize(docs, n_init);
    if (debug_) {
        std::cout << tree_.GetTree();
    }

    double tau = 10, kappa = 0.7;
    int mb_per_tc = std::max(csize / bsize, 1);

    int it = 0, n_mb = 0;
    bool integrate_z = it < n_mc_iters;
    debug("%s: starting iterations with minibatch size %d", __func__, bsize);
    nwk_global_ = MatrixXd::Zero(n_features_, offsets_[L_]);
    while (it < n_mb_iters) {
        std::shuffle(doc_idx.begin(), doc_idx.end(), random_engine_);

        for (uint32_t d_st = 0; d_st < ndocs; d_st += bsize) {
            uint32_t d_ed = std::min(d_st + bsize, (uint32_t) ndocs);
            // parallelized sampling of local parameters
            tbb::parallel_for(tbb::blocked_range<uint32_t>(d_st, d_ed), [&](const tbb::blocked_range<uint32_t>& r) {
                int tid = tbb::this_task_arena::current_thread_index();
                xorshift& rng = generators_[tid];
                for (uint32_t d = r.begin(); d < r.end(); ++d) {
                    auto &doc = docs[doc_idx[d]];
                    if (!doc.initialized && !integrate_z) {
                        doc.z.resize(doc.data.ids.size());
                        for (auto &l: doc.z) {
                            l = (uint32_t) (((uint64_t) rng.sample() * L_) >> 32);
                        }
                    }
                    sample_c(doc, rng, doc.initialized, true, integrate_z);
                    sample_z(doc, true, true);
                    doc.initialized = true;
                }
            });
            debug("%s: %d-th minibatch in %d-th iteration: sampled local parameters", __func__, n_mb, it);
            // Get new node number including ones created in this batch
            auto n_nodes_new = tree_.GetNumNodesC();
            std::vector<int> offsets_new(L_ + 1, 0);
            for (int l = 0; l < L_; ++l) {
                offsets_new[l + 1] = offsets_new[l] + n_nodes_new[l];
            }
            // Compute nwk for all nodes
            tbb::combinable<MatrixXd> nwk_acc { // in old node order
                [&]{ return MatrixXd::Zero(n_features_, offsets_new[L_]); }
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
                        nwk_local(doc.data.ids[i], offsets_new[l] + doc.c[l]) += doc.data.cnts[i];
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
            double rho = std::pow(n_mb + tau, -kappa);
            double scale = (double) ndocs / (d_ed - d_st);
            for (int l = 0; l < L_; ++l) {
                for (uint32_t i = 0; i < n_nodes_new[l]; ++i) { // within-level
                    if (i < n_nodes_[l]) { // node exists in the previous tree
                        nwk_local.col(offsets_new[l] + i) =
                        nwk_global_.col(offsets_[l] + i).array() * (1 - rho) +
                        nwk_local.col(offsets_new[l] + i).array() * rho * scale;
                    } else {
                        nwk_local.col(offsets_new[l] + i) =
                        nwk_local.col(offsets_new[l] + i).array() * rho * scale;
                    }
                }
            }
            nwk_global_ = std::move(nwk_local);
            offsets_ = std::move(offsets_new);
            debug("%s: %d-th minibatch in %d-th iteration: computed local summary statistics", __func__, n_mb, it);

            if (n_mb % mb_per_tc == 0) {
                // tree consolidation before updating global parameters
                update_global_();
                remap_c_(docs);
            } else {
                // Keep n_heavy_ and node orders. only update beta_, not nlwk_
                update_beta_();
            }
            debug("%s: %d-th minibatch in %d-th iteration: updated global parameters", __func__, n_mb, it);

            n_mb++;
        } // finish one pass
        it++;
        integrate_z = it < n_mc_iters;
        if (it == n_mb_iters) {
            bsize = ndocs;
            mb_per_tc = 1;
        }
    }
    while (it < n_iters) {
        fit_onepass(docs, integrate_z);
        it++;
        integrate_z = it < n_mc_iters;
    }
}

void HLDA::compute_node_ll_beta_(nCrpLeaf& doc, int l, int nt, float *result) {
    if (nt <= 0) {return;}
    memset(result, 0, nt * sizeof(float));
    for (uint32_t i = doc.offsets[l]; i < doc.offsets[l + 1]; ++i) {
        for (uint32_t k = 0; k < nt; ++k) {
            result[k] += logbeta_[l](doc.reordered_ids[i], k) * doc.reordered_cts[i];
        }
    }
}

float HLDA::compute_node_ll_nwk_(nCrpLeaf& doc, int l, int offset, int nt, float* result) {
    if (offset + nt > nlwk_[l].GetC()) {
        throw std::out_of_range("HLDA::compute_node_ll_nwk_: offset+nt exceeds number of topics on level l");
    }
    memset(result, 0, nt * sizeof(float));
    double Weta = n_features_ * eta_[l];
    float new_node_ll = 0.0f; // depends on l only through ndlw thus z, not c
    const auto& nwk = nlwk_[l];
    double ndl = 0.;
    for (uint32_t i = doc.offsets[l]; i < doc.offsets[l + 1]; ++i) {
        double ndlw = doc.reordered_cts[i];
        ndl += ndlw;
        double ndlw2 = std::max(ndlw/2-.5, 0.); // approx seq sum with midpoint
        for (uint32_t k = 0; k < nt; ++k) {
            auto n = nwk.Get(doc.reordered_ids[i], offset + k);
            result[k] += ndlw * log(n + ndlw2 + eta_[l]);
        }
        new_node_ll += ndlw * log(ndlw2 + eta_[l]);
    }
// std::cout << "HLDA::compute_node_ll_nwk_: (" << l << ", " << ndl << ") eta=" << eta_[l] << std::endl;
// for (uint32_t k = 0; k < nt; ++k) {
//     std::cout << " " << result[k];
// }
// std::cout << "; " << new_node_ll << std::endl;

    for (uint32_t k = 0; k < nt; ++k) {
        double nk = nwk.GetSum(offset + k);
        result[k] += lgamma(nk + Weta) - lgamma(nk + ndl + Weta);
    }
    new_node_ll += lgamma(Weta) - lgamma(ndl + Weta);
// for (uint32_t k = 0; k < nt; ++k) {
//     std::cout << " " << result[k];
// }
// std::cout << "; " << new_node_ll << std::endl;
    return new_node_ll;
}

void HLDA::update_nwk_(nCrpLeaf& doc, bool add) {
    if (doc.c.empty()) {
        return;
    }
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

void HLDA::remap_nwk_(const std::vector<ConcurrentTree::NodeRemap>& remaps) {
    if (remaps.empty()) return;

    const std::size_t nRows = nlwk_.front().GetR();
    struct Pending {
        int l, k;
        std::vector<double> col;
    };
    std::vector<Pending> pendings;
    pendings.reserve(remaps.size());
    auto n_heavy = tree_.GetNumHeavyC();
    for (const auto& v : remaps) {
        if (v.k_new < n_heavy[v.l_new]) {
            continue;
        }
        Pending p{v.l_new, v.k_new, std::vector<double>(nRows + 1, 0.0)};
        Eigen::Map<Eigen::VectorXd> acc(p.col.data(), nRows);
        for (std::size_t j = 0; j < v.l_old.size(); ++j) {
            int l = v.l_old[j];
            int k = v.k_old[j];
            if (k < n_heavy_[l]) { // was heavy before, so not in nlwk_
                acc += nwk_global_.col(offsets_[l] + k);
                p.col[nRows] += nwk_global_.col(offsets_[l] + k).sum();
            } else {
                const double* src =
                    reinterpret_cast<const double*>(nlwk_[l].ColPtrAtomic(k));
                acc += Eigen::Map<const Eigen::VectorXd>(src, nRows);
                p.col[nRows] += src[nRows];
            }
        }
        pendings.emplace_back(std::move(p));
    }

    for (int l = 0; l < L_; ++l) {
        for (int k = 0; k < nlwk_[l].GetC(); ++k) {
            if (nlwk_[l].GetSum(k) > 0.0) {
                nlwk_[l].ClearCol(k);
            }
        }
    }
    for (const auto& p : pendings) {
        double* dst = reinterpret_cast<double*>(
            nlwk_[p.l].ColPtrAtomic(static_cast<std::size_t>(p.k)));
        std::memcpy(dst, p.col.data(), sizeof(double) * (nRows + 1));
    }
}

void HLDA::compute_path_ll_(nCrpLeaf& doc, xorshift& rng, ConcurrentTree::RetTree& tree, float* pk) {
    doc.partition_by_z(L_);

    std::vector<std::vector<float>> scores(L_);
    // Compute likelihood for heavy topics (using beta)
    for (int l = 0; l < L_; ++l) { // P(w_{dl}|\beta_k) for heavy k on level l
        uint32_t nheavy = n_heavy_[l];
        scores[l].resize(tree.n_nodes[l] + 1);
        #pragma forceinline
        compute_node_ll_beta_(doc, l, nheavy, scores[l].data());
    }
    // Compute likelihood for rapid changing topics (using nlwk)
    auto &nodes = tree.nodes;
    uint32_t n = nodes.size();
    for (int l = 0; l < L_; ++l) { // P(w_{dl}|n_{wk}) for small k on level l
        auto &score = scores[l];
        uint32_t nheavy = n_heavy_[l];
        uint32_t nsmall = tree.n_nodes[l] - nheavy;
        #pragma forceinline
        score.back() = compute_node_ll_nwk_(doc, l, nheavy, nsmall, score.data() + nheavy); // P(w_{dl}|new k on l)
        if (!allow_new_topic_) {
            score.back() = -1e20f;
        }
    }
    // Compute path prior
    std::vector<float> sum_logpk(nodes.size()); // logP(w|c) for partial paths
    std::vector<float> empty_pl(L_, 0.0f); // P(w_{d,l'>l}|new k on all l'>l)
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
        // logP(w|c) + logP(c) for the path from root to i (and branching out)
        pk[i] = sum_logpk[i] + node.log_path_weight;
        if (node.depth < L_ - 1) { // new path from an internal node
            pk[i] += empty_pl[node.depth];
        }
// std::cout << i << " (" << node.depth << ", " << node.n_docs << ") " << (sum_logpk[i] + ((node.depth < L_ - 1) ? empty_pl[node.depth] : 0)) << " " << node.log_path_weight << " -> " << pk[i] << std::endl;
    }
}

void HLDA::update_global_() {
    // Consolidate the tree
    std::vector<ConcurrentTree::NodeRemap> pos_map;
    bool node_remap  = tree_.Consolidate(pos_map);
    n_nodes_ = tree_.GetNumNodes();
    if (debug_) {
        ConcurrentTree::RetTree tree = tree_.GetTree();
        notice("%s: consolidated tree (%d)", __func__, node_remap);
        std::cout << tree;
    }
    std::vector<int> offsets_new(L_ + 1, 0);
    for (int l = 0; l < L_; ++l) {
        offsets_new[l + 1] = offsets_new[l] + n_nodes_[l];
    }
    if (node_remap) {
        MatrixXd nwk_new(n_features_, offsets_new[L_]);
        for (const auto& v : pos_map) {
            int i_new = offsets_new[v.l_new] + v.k_new;
            for (size_t i = 0; i < v.k_old.size(); ++i) {
                int i_old = offsets_[v.l_old[i]] + v.k_old[i];
                nwk_new.col(i_new) += nwk_global_.col(i_old);
            }
        }
        nwk_global_ = std::move(nwk_new);
    }
    offsets_ = std::move(offsets_new);
    // Permute columns in nlwk_
    if (node_remap || tree_.GetNumHeavyC() != n_heavy_) {
        remap_nwk_(pos_map);
    }
    n_heavy_ = tree_.GetNumHeavy();
    // Update beta and logbeta
    update_beta_();

    if (tree_.GetSize() >= max_n_nodes_) {
        allow_new_topic_ = false;
    }
}

void HLDA::update_beta_() {
    VectorXd col_sums = nwk_global_.colwise().sum(); // n_k
    for (int l = 0; l < L_; ++l) {
        double Weta = n_features_ * eta_[l];
        int K = n_nodes_[l];
        beta_[l].SetC(K, false, true);
        logbeta_[l].SetC(K, false, true);
        int o = offsets_[l];
        std::vector<double> denom(K);
        tbb::parallel_for(tbb::blocked_range<uint32_t>(0, K), [&](const tbb::blocked_range<uint32_t>& r) {
            for (uint32_t k = r.begin(); k < r.end(); ++k) {
                denom[k] = 1. / (Weta + col_sums(o + k));
                Eigen::Map<Eigen::VectorXd> dst(beta_[l].ColPtr(k), n_features_);
                dst = (nwk_global_.col(o + k).array() + eta_[l]) * denom[k];
                std::transform(
                    beta_[l].ColPtr(k), beta_[l].ColPtr(k+1),
                    logbeta_[l].ColPtr(k), [](float x) {return std::log(x);});
            }
        });
    }
}

void HLDA::initialize(std::vector<nCrpLeaf>& docs, int n) {
    xorshift& rng = generators_[0];
    for (uint32_t d = 0; d < n && d < docs.size(); ++d) {
        auto &doc = docs[d];
        sample_c(doc, rng, doc.initialized);
        sample_z(doc, true, true);
        doc.initialized = true;
    }
    n_heavy_ = tree_.GetNumHeavy();
    n_nodes_ = tree_.GetNumNodes();
    for (int l = 0; l < L_; ++l) {
        offsets_[l + 1] = offsets_[l] + n_nodes_[l];
    }
}

void HLDA::fit_onepass(std::vector<nCrpLeaf>& docs, bool integrate_z) {
    int32_t ndocs = docs.size();
    std::vector<uint32_t> doc_idx(ndocs);
    std::iota(doc_idx.begin(), doc_idx.end(), 0);
    std::shuffle(doc_idx.begin(), doc_idx.end(), random_engine_);
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, ndocs), [&](const tbb::blocked_range<uint32_t>& r) {
        int tid = tbb::this_task_arena::current_thread_index();
        xorshift& rng = generators_[tid];
        for (uint32_t d = r.begin(); d < r.end(); ++d) {
            auto &doc = docs[doc_idx[d]];
            if (!doc.initialized && !integrate_z) {
                doc.z.resize(doc.data.ids.size());
                for (auto &l: doc.z) {
                    l = (uint32_t) (((uint64_t) rng.sample() * L_) >> 32);
                }
            }
            sample_c(doc, rng, doc.initialized, true, integrate_z);
            sample_z(doc, true, true);
            doc.initialized = true;
        }
    });
    n_nodes_ = tree_.GetNumNodes();
    for (int l = 0; l < L_; ++l) {
        offsets_[l + 1] = offsets_[l] + n_nodes_[l];
    }
    tbb::combinable<MatrixXd> nwk_acc { // in old node order
        [&]{ return MatrixXd::Zero(n_features_, offsets_[L_]); }
    };
    tbb::combinable<VectorXd> nct_acc { // level marginal
        [&]{ return VectorXd::Zero(L_); }
    };
    tbb::parallel_for(tbb::blocked_range<uint32_t>(0, ndocs), [&](const tbb::blocked_range<uint32_t>& r) {
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
    nwk_global_ = nwk_acc.combine(
        [](const MatrixXd& a, const MatrixXd& b) {return a + b;}
    );
    VectorXd nct = nct_acc.combine(
        [](const VectorXd& a, const VectorXd& b) {return a + b;}
    );
    update_global_();
    remap_c_(docs);
}

void HLDA::remap_c_(std::vector<nCrpLeaf>& docs) {
    auto& leaf_map = tree_.id_old_to_survived_anc;
    if (leaf_map.empty()) {
        return;
    }
    auto& pos_map = tree_.pos_old2new;
    for (auto& doc : docs) {
        if (doc.leaf_id < 0) continue;
        doc.leaf_id = leaf_map[doc.leaf_id];
        if (doc.c.empty()) {
            continue;
        }
        for (int l = 0; l < L_; ++l) {
            if (doc.c[l] < 0 || doc.c[l] >= pos_map[l].size()) {
                doc.c[l] = -1;
                continue;
            }
            doc.c[l] = pos_map[l][doc.c[l]];
        }
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
