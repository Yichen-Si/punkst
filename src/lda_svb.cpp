#include "lda.hpp"

void LatentDirichletAllocation::svb_partial_fit(const std::vector<Document>& docs) {
    int minibatch_size = docs.size();
    MatrixXd gamma = MatrixXd::Zero(minibatch_size, n_topics_);

    tbb::combinable<MatrixXd> ss_acc{
        [&]{ return MatrixXd::Zero(n_topics_, n_features_); }
    };
    tbb::combinable<std::vector<int32_t>> niters_acc{
        []{ return std::vector<int32_t>(); }
    };

    tbb::parallel_for(
        tbb::blocked_range<int>(0, minibatch_size),
        [&](const tbb::blocked_range<int>& range) {
        auto& local_ss   = ss_acc.local();
        auto& local_nits = niters_acc.local();
        VectorXd phi_k(n_topics_);
        for (int d = range.begin(); d < range.end(); ++d) {
            // document level variational parameters.
            const auto& doc = docs[d];
            int n_ids = doc.ids.size();
            VectorXd gamma_d, exp_Elog_theta_d;
            int iter = svb_fit_one_document(gamma_d, exp_Elog_theta_d, doc);
            gamma.row(d) = gamma_d.transpose();
            local_nits.push_back(iter);
            // update sufficient statistics.
            for (int j = 0; j < n_ids; j++) {
                int word_id = doc.ids[j];
                phi_k = exp_Elog_theta_d.array() * exp_Elog_beta_.col(word_id).array();
                phi_k /= (phi_k.sum() + eps_);
                local_ss.col(word_id) += phi_k * doc.cnts[j];
            }
        }
    });

    // 4) Merge thread‐local ss and niters into the global buffers
    MatrixXd ss = ss_acc.combine(
        [](const MatrixXd &A, const MatrixXd &B) {
            return A + B;
        }
    );
    std::vector<int32_t> niters = niters_acc.combine(
        [](const std::vector<int32_t> &a,
            const std::vector<int32_t> &b) {
            std::vector<int32_t> out = a;
            out.insert(out.end(), b.begin(), b.end());
            return out;
        }
    );

    if (verbose_ > 0) {
        int32_t fail_converge = 0;
        for (int i = 0; i < niters.size(); i++) {
            if (niters[i] >= max_doc_update_iter_) {
                fail_converge++;
            }
        }
        notice("Partial fit: %d documents. Average iterations per doc: %.2f, %d documents did not reach mean change %.1e in %d iterations.", minibatch_size, std::accumulate(niters.begin(), niters.end(), 0) / static_cast<double>(niters.size()), fail_converge, mean_change_tol_, max_doc_update_iter_);
        if (verbose_ > 2) {
            std::vector<double> scores = approx_bound(docs, gamma, false);
            scores[0] /= minibatch_size;
            scores[1] /= minibatch_size;
            notice("  Average log-likelihood: %.2f, average KL divergence to prior: %.2f", scores[0], -scores[1]);
            std::vector<double> weights;
            get_topic_abundance(weights);
            std::stringstream ss;
            ss.precision(4);
            for (const auto& w : weights) {
                ss << std::fixed << w << "\t";
            }
            notice("  Topic relative abundance: %s", ss.str().c_str());
        }
    }

    // Update the global parameters using an online learning rate.
    update_count_++;
    double rho = std::pow(learning_offset_ + update_count_, -learning_decay_);
    MatrixXd update_val =
            MatrixXd::Constant(n_topics_, n_features_, eta_) +
            (static_cast<double>(total_doc_count_) / minibatch_size) * ss;
    components_ = (1 - rho) * components_ + rho * update_val;
    exp_Elog_beta_ = dirichlet_expectation_2d(components_);
}

int32_t LatentDirichletAllocation::svb_fit_one_document(
    VectorXd& gamma, VectorXd& exp_Elog_theta, const Document &doc) {
    int n_ids = doc.ids.size();
    if (gamma.size() != n_topics_) {
        gamma.resize(n_topics_);
        if (n_ids == 0) {
            std::fill(gamma.data(), gamma.data() + n_topics_, alpha_);
        } else {
            std::gamma_distribution<double> gamma_dist(100.0, 0.01);
            for (int k = 0; k < n_topics_; k++) {
                gamma[k] = gamma_dist(random_engine_);
            }
        }
    }
    if (n_ids == 0) {
        exp_Elog_theta.resize(n_topics_);
        exp_Elog_theta.setConstant(1.0 / n_topics_);
        return 0;
    }

    exp_Elog_theta = dirichlet_expectation_1d(gamma, 0); // K x 1
    // Build a submatrix for the nonzero word indices in the document.
    MatrixXd exp_Elog_beta_local(n_topics_, n_ids);
    for (int j = 0; j < n_ids; j++) {
        exp_Elog_beta_local.col(j) = exp_Elog_beta_.col(doc.ids[j]);
    }
    Eigen::Map<const Eigen::VectorXd> doc_counts(doc.cnts.data(), n_ids);
    // Iterative update for the document.
    double diff = 1.;
    int iter = 0;
    while (iter < max_doc_update_iter_) {
        VectorXd last_gamma = gamma;
        // norm_phi: |ids| x 1
        VectorXd norm_phi = exp_Elog_beta_local.transpose() * exp_Elog_theta;
        norm_phi.array() += eps_;
        VectorXd ratio = doc_counts.array() / norm_phi.array();
        gamma = exp_Elog_theta.array() * (exp_Elog_beta_local * ratio).array();
        // Dirichlet expectation update:
        exp_Elog_theta = dirichlet_expectation_1d(gamma, alpha_);
        // Check convergence via mean absolute change.
        diff = (last_gamma - gamma).cwiseAbs().sum() / n_topics_;
        iter++;
        if (diff < mean_change_tol_) {
            break;
        }
    }
    if (verbose_ > 1) {
        notice("%s: finished after %d iterations, mean change %.1e", __FUNCTION__, iter, diff);
    }
    return iter;
}

void LatentDirichletAllocation::set_svb_parameters(int32_t max_iter, double tol) {
    max_doc_update_iter_ = max_iter;
    if (tol < 0.) {
        mean_change_tol_ = 0.001 / n_topics_;
    } else {
        mean_change_tol_ = tol;
    }
}

void LatentDirichletAllocation::set_background_prior(const VectorXd& eta0, double a0, double b0, bool fixed) {
    assert(algo_ == InferenceType::SVB_DN || algo_ == InferenceType::SVB);
    assert(eta0.size() == n_features_);
    eta0_ = eta0.array().max(1e-12);
    a0_ = a0; b0_ = b0;
    a_ = 0; b_ = 0;
    lambda0_ = eta0_;
    exp_Elog_beta0_ = dirichlet_expectation_1d(lambda0_, 0);
    algo_ = InferenceType::SVB_DN;
    fix_background_ = fixed;
}
void LatentDirichletAllocation::set_background_prior(const std::vector<double> eta0, double a0, double b0, bool fixed) {
    assert(algo_ == InferenceType::SVB_DN || algo_ == InferenceType::SVB);
    assert(eta0.size() == static_cast<size_t>(n_features_));
    eta0_ = VectorXd::Zero(eta0.size());
    for (size_t j = 0; j < eta0.size(); ++j) {
        eta0_(j) = std::max(eta0[j], 1e-12);
    }
    a0_ = a0; b0_ = b0;
    a_ = 0; b_ = 0;
    lambda0_ = eta0_;
    exp_Elog_beta0_ = dirichlet_expectation_1d(lambda0_, 0);
    algo_ = InferenceType::SVB_DN;
    fix_background_ = fixed;
}

std::vector<double> LatentDirichletAllocation::approx_bound(
    const std::vector<Document>& docs, const MatrixXd& doc_topic_distr, bool sub_sampling) {
    const int n_docs     = static_cast<int>(docs.size());
    const int n_topics   = n_topics_;
    const int n_features = n_features_;

    // 1) Expected log likelihood
    double score1 = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, n_docs), 0.0,
        [&](const tbb::blocked_range<int>& range, double local_sum) {
            for (int d = range.begin(); d != range.end(); ++d) {
                const auto& doc = docs[d];
                double sum_gamma = doc_topic_distr.row(d).sum();
                // for each word in doc
                for (size_t idx = 0; idx < doc.ids.size(); ++idx) {
                    int    word_id = doc.ids[idx];
                    double count   = doc.cnts[idx];
                    // log-sum-exp over topics
                    double max_val = -std::numeric_limits<double>::infinity();
                    std::vector<double> temp(n_topics);
                    for (int k = 0; k < n_topics; ++k) {
                        double doc_term   = psi(doc_topic_distr(d,k)) - psi(sum_gamma);
                        double row_sum    = components_.row(k).sum();
                        double topic_term = psi(components_(k,word_id)) - psi(row_sum);
                        double comb       = doc_term + topic_term;
                        temp[k] = comb;
                        if (comb > max_val) max_val = comb;
                    }
                    double sum_exp = 0.0;
                    for (int k = 0; k < n_topics; ++k)
                        sum_exp += std::exp(temp[k] - max_val);

                    double word_ll = max_val + std::log(sum_exp);
                    local_sum += count * word_ll;
                }
            }
            return local_sum;
        },
        std::plus<>()  // how to combine partial sums
    );

    // 2) Document–topic prior term
    double score2 = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, n_docs), 0.0,
        [&](const tbb::blocked_range<int>& range, double local_sum) {
            for (int d = range.begin(); d != range.end(); ++d) {
                Eigen::VectorXd gamma = doc_topic_distr.row(d).transpose();
                double sum_gamma  = gamma.sum();
                double psi_sum    = psi(sum_gamma);
                double doc_score  = 0.0;
                for (int k = 0; k < n_topics; ++k) {
                    double gmk = gamma[k];
                    doc_score += (alpha_ - gmk) * (psi(gmk) - psi_sum);
                    doc_score += std::lgamma(gmk) - std::lgamma(alpha_);
                }
                doc_score += std::lgamma(alpha_*n_topics)
                            - std::lgamma(sum_gamma);
                local_sum += doc_score;
            }
            return local_sum;
        },
        std::plus<>()
    );
    // optional scaling for subsampling
    if (sub_sampling) {
        double ratio = static_cast<double>(total_doc_count_) / n_docs;
        score1 *= ratio;
        score2 *= ratio;
    }

    // 3) Topic–word prior term
    double score3 = tbb::parallel_reduce(
        tbb::blocked_range<int>(0, n_topics),
        0.0,
        [&](const tbb::blocked_range<int>& range, double local_sum) {
            for (int k = range.begin(); k != range.end(); ++k) {
                double row_sum    = components_.row(k).sum();
                double psi_row    = psi(row_sum);
                double topic_score = 0.0;

                for (int j = 0; j < n_features; ++j) {
                    double lkj = components_(k,j);
                    topic_score += (eta_ - lkj)
                                    * (psi(lkj) - psi_row);
                    topic_score += std::lgamma(lkj)
                                    - std::lgamma(eta_);
                }
                topic_score += std::lgamma(eta_*n_features)
                                - std::lgamma(row_sum);

                local_sum += topic_score;
            }
            return local_sum;
        },
        std::plus<>()
    );

    return { score1, score2, score3 };
}

double LatentDirichletAllocation::_perplexity_precomp_distr(const std::vector<Document>& docs, const MatrixXd& doc_topic_distr, bool sub_sampling) {
    std::vector<double> scores = approx_bound(docs, doc_topic_distr, sub_sampling);
    double bound = scores[0] + scores[1] + scores[2];
    double word_cnt = 0.0;
    for (const auto& doc : docs) {
        for (double cnt : doc.cnts) {
            word_cnt += cnt;
        }
    }
    if (sub_sampling) {
        word_cnt *= static_cast<double>(total_doc_count_) / docs.size();
    }
    double perword_bound = bound / word_cnt;
    return std::exp(-perword_bound);
}

int32_t LatentDirichletAllocation::svbdn_fit_one_document(
    VectorXd& gamma, VectorXd& exp_Elog_theta, const Document &doc, ArrayXd& fg_counts) {
    int n_ids = doc.ids.size();
    if (gamma.size() != n_topics_) {
        gamma.resize(n_topics_);
        if (n_ids == 0) {
            std::fill(gamma.data(), gamma.data() + n_topics_, alpha_);
        } else {
            std::gamma_distribution<double> gamma_dist(100.0, 0.01);
            for (int k = 0; k < n_topics_; k++) {
                gamma[k] = gamma_dist(random_engine_);
            }
        }
    }
    if (n_ids == 0) {
        exp_Elog_theta.resize(n_topics_);
        exp_Elog_theta.setConstant(1.0 / n_topics_);
        return 0;
    }
    exp_Elog_theta = dirichlet_expectation_1d(gamma, 0); // K x 1
    // Build a submatrix for the nonzero word indices in the document.
    MatrixXd exp_Elog_beta_local(n_topics_, n_ids);
    ArrayXd Elog_beta0_local(n_ids);
    for (int j = 0; j < n_ids; j++) {
        exp_Elog_beta_local.col(j) = exp_Elog_beta_.col(doc.ids[j]);
        Elog_beta0_local[j] = std::log(exp_Elog_beta0_[doc.ids[j]]);
    }
    Eigen::Map<const ArrayXd> doc_counts(doc.cnts.data(), n_ids);
    fg_counts.resize(n_ids);
    double cnt_sum = doc_counts.sum();
    double aj = a0_, bj = b0_;
    double Elogit_pi = psi(a0_) - psi(b0_);

    // Iterative update for the document.
    double diff = 1.;
    int iter = 0;
    while (iter < max_doc_update_iter_) {
        VectorXd last_gamma = gamma;
        // norm_phi: |ids| x 1, \sum_k exp(E[log beta_km] + E[log theta_k] )
        VectorXd norm_phi = exp_Elog_beta_local.transpose() * exp_Elog_theta;
        norm_phi.array() += eps_;
        // Background level update
        ArrayXd phi0 = Elog_beta0_local + Elogit_pi - norm_phi.array().log();
        for (auto& v : phi0) {v = expit(v);}
        double phi0_sum = (phi0 * doc_counts).sum();
        aj = a0_ + phi0_sum;
        bj = b0_ + cnt_sum - phi0_sum;
        Elogit_pi = psi(aj) - psi(bj);
        fg_counts = doc_counts * (1. - phi0);
        // Topic assignment update
        VectorXd ratio = fg_counts / norm_phi.array();
        gamma = exp_Elog_theta.array() * (exp_Elog_beta_local * ratio).array();
        exp_Elog_theta = dirichlet_expectation_1d(gamma, alpha_);
        // Check convergence via mean absolute change.
        diff = (last_gamma - gamma).cwiseAbs().sum() / n_topics_;
        iter++;
        if (diff < mean_change_tol_) {
            break;
        }
    }
    if (verbose_ > 1) {
        double bg_frac = fg_counts.sum() / cnt_sum;
        notice("%s: finished after %d iterations, mean change %.1e, background fraction %.3f", __FUNCTION__, iter, diff, bg_frac);
    }
    return iter;
}


void LatentDirichletAllocation::svbdn_partial_fit(const std::vector<Document>& docs) {
    int minibatch_size = docs.size();
    MatrixXd gamma = MatrixXd::Zero(minibatch_size, n_topics_);

    tbb::combinable<MatrixXd> ss_acc{
        [&]{ return MatrixXd::Zero(n_topics_, n_features_); }
    };
    tbb::combinable<VectorXd> ss0_acc{
        [&]{ return VectorXd::Zero(n_features_); }
    };
    tbb::combinable<std::vector<int32_t>> niters_acc{
        []{ return std::vector<int32_t>(); }
    };
    tbb::combinable<double> phi0_acc{[]{ return 0.0; }};
    tbb::combinable<double> phi1_acc{[]{ return 0.0; }};

    tbb::parallel_for(
        tbb::blocked_range<int>(0, minibatch_size),
        [&](const tbb::blocked_range<int>& range) {
        auto& local_ss   = ss_acc.local();
        auto& local_nits = niters_acc.local();
        auto& local_ss0  = ss0_acc.local();
        auto& local_phi0 = phi0_acc.local();
        auto& local_phi1 = phi1_acc.local();
        VectorXd phi_k(n_topics_);
        for (int d = range.begin(); d < range.end(); ++d) {
            // document level variational parameters.
            const auto& doc = docs[d];
            int n_ids = doc.ids.size();
            VectorXd gamma_d, exp_Elog_theta_d;
            ArrayXd fg_counts;
            int iter = svbdn_fit_one_document(gamma_d, exp_Elog_theta_d, doc, fg_counts);
            gamma.row(d) = gamma_d.transpose();
            local_nits.push_back(iter);
            double c = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
            double c1 = fg_counts.sum();
            local_phi0 += c - c1;
            local_phi1 += c1;
            // update sufficient statistics.
            for (int j = 0; j < n_ids; j++) {
                int word_id = doc.ids[j];
                phi_k = exp_Elog_theta_d.array() * exp_Elog_beta_.col(word_id).array();
                phi_k /= (phi_k.sum() + eps_);
                local_ss.col(word_id) += phi_k * fg_counts[j];
                local_ss0[word_id] += doc.cnts[j] - fg_counts[j];
            }
        }
    });

    // 4) Merge thread‐local ss and niters into the global buffers
    MatrixXd ss = ss_acc.combine(
        [](const MatrixXd &A, const MatrixXd &B) {return A + B;}
    );
    VectorXd ss0 = ss0_acc.combine(
        [](const VectorXd &A, const VectorXd &B) {return A + B;}
    );
    std::vector<int32_t> niters = niters_acc.combine(
        [](const std::vector<int32_t> &a, const std::vector<int32_t> &b) {
            std::vector<int32_t> out = a;
            out.insert(out.end(), b.begin(), b.end());
            return out;
        }
    );
    double phi0 = phi0_acc.combine(
        [](double a, double b) {return a + b;}
    );
    double phi1 = phi1_acc.combine(
        [](double a, double b) {return a + b;}
    );
    a_ += phi0;
    b_ += phi1;

    if (verbose_ > 0) {
        int32_t fail_converge = 0;
        for (int i = 0; i < niters.size(); i++) {
            if (niters[i] >= max_doc_update_iter_) {
                fail_converge++;
            }
        }
        notice("Partial fit: %d documents. Average iterations per doc: %.2f, %d documents did not reach mean change %.1e in %d iterations. Average background fraction: %.3f", minibatch_size, std::accumulate(niters.begin(), niters.end(), 0) / static_cast<double>(niters.size()), fail_converge, mean_change_tol_, max_doc_update_iter_, phi0/(phi0 + phi1));
    }

    // Update the global parameters using an online learning rate.
    update_count_++;
    double rho = std::pow(learning_offset_ + update_count_, -learning_decay_);
    double scale = static_cast<double>(total_doc_count_) / minibatch_size;
    MatrixXd update_val =
            MatrixXd::Constant(n_topics_, n_features_, eta_) + scale * ss;
    components_ = (1 - rho) * components_ + rho * update_val;
    exp_Elog_beta_ = dirichlet_expectation_2d(components_);
    VectorXd update_val0 = eta0_ + scale * ss0;
    if (!fix_background_) {
        lambda0_ = (1 - rho) * lambda0_ + rho * update_val0;
        exp_Elog_beta0_ = dirichlet_expectation_1d(lambda0_, 0);
    }
}
