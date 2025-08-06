#include "lda.hpp"

void LatentDirichletAllocation::svb_partial_fit(const std::vector<Document>& docs) {
    int minibatch_size = docs.size();
    MatrixXd doc_topic_distr = MatrixXd::Zero(minibatch_size, n_topics_);

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
            VectorXd doc_topic, exp_doc;
            int iter = svb_fit_one_document(doc_topic, exp_doc, doc);
            doc_topic_distr.row(d) = doc_topic.transpose();
            local_nits.push_back(iter);
            // update sufficient statistics.
            for (int j = 0; j < n_ids; j++) {
                int word_id = doc.ids[j];
                phi_k = exp_doc.array() * exp_Elog_beta_.col(word_id).array();
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
            std::vector<double> scores = approx_bound(docs, doc_topic_distr, false);
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
            MatrixXd::Constant(n_topics_, n_features_, topic_word_prior_) +
            (static_cast<double>(total_doc_count_) / minibatch_size) * ss;
    components_ = (1 - rho) * components_ + rho * update_val;
    exp_Elog_beta_ = dirichlet_expectation_2d(components_);
}

int32_t LatentDirichletAllocation::svb_fit_one_document(
    VectorXd& doc_topic, VectorXd& exp_doc,
    const Document &doc, const std::optional<VectorXd>& doc_topic_) {
    int n_ids = doc.ids.size();
    if (!doc_topic_) {
        doc_topic.resize(n_topics_);
        if (n_ids == 0) {
            std::fill(doc_topic.data(), doc_topic.data() + n_topics_, doc_topic_prior_);
        } else {
            std::gamma_distribution<double> gamma_dist(100.0, 0.01);
            for (int k = 0; k < n_topics_; k++) {
                doc_topic[k] = gamma_dist(random_engine_);
            }
        }
    } else {
        doc_topic = *doc_topic_;
    }

    if (n_ids == 0) {
        exp_doc.resize(n_topics_);
        exp_doc.setConstant(1.0 / n_topics_);
        return 0;
    }

    exp_doc = dirichlet_expectation_1d(doc_topic, 0);
    // Build a submatrix for the nonzero word indices in the document.
    MatrixXd exp_topic_word(n_topics_, n_ids);
    for (int j = 0; j < n_ids; j++) {
        exp_topic_word.col(j) = exp_Elog_beta_.col(doc.ids[j]);
    }
    Eigen::Map<const Eigen::VectorXd> doc_counts(doc.cnts.data(), n_ids);
    // Iterative update for the document.
    double diff = 1.;
    int iter = 0;
    while (iter < max_doc_update_iter_) {
        // VectorXd last_doc = doc_topic; // Save the previous state.
        VectorXd last_doc = doc_topic;

        // norm_phi = exp_doc^T * exp_topic_word (|ids| x 1).
        VectorXd norm_phi = exp_topic_word.transpose() * exp_doc;
        norm_phi.array() += eps_; // Avoid division by zero.

        VectorXd ratio = doc_counts.array() / norm_phi.array();
        VectorXd new_vector = exp_topic_word * ratio; // K x 1
        doc_topic = exp_doc.array() * new_vector.array();
        // Dirichlet expectation update:
        // Add the prior and update exp_doc.
        exp_doc = dirichlet_expectation_1d(doc_topic, doc_topic_prior_);

        // Check convergence via mean absolute change.
        diff = (last_doc - doc_topic).cwiseAbs().sum() / n_topics_;

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
                    doc_score += (doc_topic_prior_ - gmk) * (psi(gmk) - psi_sum);
                    doc_score += std::lgamma(gmk) - std::lgamma(doc_topic_prior_);
                }
                doc_score += std::lgamma(doc_topic_prior_*n_topics)
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
                    topic_score += (topic_word_prior_ - lkj)
                                    * (psi(lkj) - psi_row);
                    topic_score += std::lgamma(lkj)
                                    - std::lgamma(topic_word_prior_);
                }
                topic_score += std::lgamma(topic_word_prior_*n_features)
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
