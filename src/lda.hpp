#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <random>
#include <optional>
#include <omp.h>
#include "numerical_utils.hpp"
#include "dataunits.hpp"
#include "qgenlib/qgen_error.h"

#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

class LatentDirichletAllocation {
public:
    LatentDirichletAllocation(int n_topics, int n_features,
                              int seed = std::random_device{}(),
                              int nThreads = 0, int verbose = 0,
                              double doc_topic_prior = -1., // alpha
                              double topic_word_prior = -1., // eta
                              int max_doc_update_iter = 100,
                              double mean_change_tol = -1.,
                              double learning_decay = 0.7, // kappa
                              double learning_offset = 10.0, // tau_0
                              int total_doc_count = 1000000,
                              const std::optional<MatrixXd>& topic_word_distr = std::nullopt)
        : n_topics_(n_topics), n_features_(n_features), seed_(seed),
        nThreads_(nThreads), verbose_(verbose),
        doc_topic_prior_(doc_topic_prior),
        topic_word_prior_(topic_word_prior),
        max_doc_update_iter_(max_doc_update_iter),
        mean_change_tol_(mean_change_tol),
        learning_decay_(learning_decay), learning_offset_(learning_offset),
        total_doc_count_(total_doc_count),
        update_count_(0) {
        init(topic_word_distr);
    }
    LatentDirichletAllocation(int n_topics, int n_features,
        int seed = std::random_device{}(),
        int nThreads = 0, int verbose = 0,
        const std::optional<MatrixXd>& topic_word_distr = std::nullopt,
        int max_doc_update_iter = 100,
        double mean_change_tol = -1.) : n_topics_(n_topics), n_features_(n_features), seed_(seed), nThreads_(nThreads), verbose_(verbose), max_doc_update_iter_(max_doc_update_iter),
        mean_change_tol_(mean_change_tol), update_count_(0) {
            doc_topic_prior_ = -1.;
            topic_word_prior_ = -1.;
            learning_decay_ = 0.7;
            learning_offset_ = 10.0;
            total_doc_count_ = 1000000;
            init(topic_word_distr);
    }

    const MatrixXd& get_model() const {
        return components_;
    }
    MatrixXd copy_model() const {
        return components_;
    }
    int32_t get_n_topics() const {
        return n_topics_;
    }
    int32_t get_n_features() const {
        return n_features_;
    }
    int32_t get_N_global() const {
        return total_doc_count_;
    }
    void get_topic_abundance(std::vector<double>& weights) const {
        weights.resize(n_topics_);
        for (int k = 0; k < n_topics_; k++) {
            weights[k] = components_.row(k).sum();
        }
        double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (int k = 0; k < n_topics_; k++) {
            weights[k] /= total;
        }
    }
    void set_nthreads(int nThreads) {
        nThreads_ = nThreads;
        if (nThreads_ > 0) {
            omp_set_num_threads(nThreads_);
        } else {
            nThreads_ = omp_get_max_threads();
        }
    }

    void set_model_from_matrix(std::vector<std::vector<double>>& lambdaVals) {
        if (lambdaVals.size() != n_topics_ || lambdaVals[0].size() != n_features_) {
            warning("Model matrix size mismatch, reset according to the provided global parameters. (%d x %d) -> (%d x %d)", n_topics_, n_features_, lambdaVals.size(), lambdaVals[0].size());
            n_topics_ = lambdaVals.size();
            n_features_ = lambdaVals[0].size();
        }
        notice("Global variational parameters are reset, but online training status (if any) is not. It is only safe for transform.");
        components_.resize(n_topics_, n_features_);
        for (int i = 0; i < n_topics_; i++) {
            for (int j = 0; j < n_features_; j++) {
                components_(i, j) = lambdaVals[i][j];
            }
        }
        exp_Elog_beta_ = dirichlet_expectation_2d(components_);
    }
    void set_model_from_matrix(MatrixXd& lambda) {
        if (lambda.rows() != n_topics_ || lambda.cols() != n_features_) {
            warning("Model matrix size mismatch, reset according to the provided global parameters. (%d x %d) -> (%d x %d)", n_topics_, n_features_, lambda.rows(), lambda.cols());
            n_topics_ = lambda.rows();
            n_features_ = lambda.cols();
        }
        notice("Global variational parameters are reset, but online training status (if any) is not. It is only safe for transform.");
        components_ = lambda;
        exp_Elog_beta_ = dirichlet_expectation_2d(components_);
    }

    // Online update
    // process a mini-batch of documents to update the global topic-word distribution.

    void partial_fit(const std::vector<Document>& docs) {
        int minibatch_size = docs.size();
        MatrixXd ss = MatrixXd::Zero(n_topics_, n_features_);
        MatrixXd doc_topic_distr = MatrixXd::Zero(minibatch_size, n_topics_);
        std::vector<int32_t> niters;
        #pragma omp parallel
        {
            MatrixXd local_ss = MatrixXd::Zero(n_topics_, n_features_);
            MatrixXd local_doc_topic_distr = MatrixXd::Zero(minibatch_size, n_topics_);
            std::vector<int32_t> local_niters;
            #pragma omp for
            for (int d = 0; d < minibatch_size; d++) {
                // Use the shared routine to get the document's variational parameters.
                VectorXd doc_topic, exp_doc;
                local_niters.push_back(fit_one_document(doc_topic, exp_doc, docs[d]));
                doc_topic_distr.row(d) = doc_topic.transpose();
                int n_ids = docs[d].ids.size();
                // For each nonzero word in the document, update sufficient statistics.
                for (int j = 0; j < n_ids; j++) {
                    int word_id = docs[d].ids[j];
                    double count = docs[d].cnts[j];
                    double norm_phi =eps_;
                    for (int k = 0; k < n_topics_; k++) {
                        norm_phi += exp_doc[k] * exp_Elog_beta_(k, word_id);
                    }
                    for (int k = 0; k < n_topics_; k++) {
                        double phi = (exp_doc[k] * exp_Elog_beta_(k, word_id)) / norm_phi;
                        local_ss(k, word_id) += count * phi;
                    }
                }
            }
            #pragma omp critical
            {
                ss += local_ss;
                doc_topic_distr += local_doc_topic_distr;
                niters.insert(niters.end(), local_niters.begin(), local_niters.end());
            }
        }
        if (verbose_ > 0) {
            int32_t fail_converge = 0;
            for (int i = 0; i < niters.size(); i++) {
                if (niters[i] >= max_doc_update_iter_) {
                    fail_converge++;
                }
            }
            notice("Partial fit: %d documents. Average iterations per doc: %.2f, %d documents did not reach mean change %.1e in %d iterations.", minibatch_size, std::accumulate(niters.begin(), niters.end(), 0) / static_cast<double>(niters.size()), fail_converge, mean_change_tol_, max_doc_update_iter_);
            if (verbose_ > 1) {
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

    // Parallelized transform: compute document-topic distributions for a list of documents.
    // For transform, we do not compute or return sufficient statistics.
    MatrixXd transform(const std::vector<Document>& docs) {
        int n_samples = docs.size();

        MatrixXd doc_topic_distr(n_samples, n_topics_);
        // Parallel update: process each document independently.
        #pragma omp parallel for schedule(dynamic)
        for (int d = 0; d < n_samples; d++) {
            // Update document d using the helper function.
            VectorXd updated_doc, exp_doc;
            int32_t niter = fit_one_document(updated_doc, exp_doc, docs[d]);
            doc_topic_distr.row(d) = updated_doc.transpose();
        }
        return doc_topic_distr;
    }

    // Compute the approximate variational bound
    std::vector<double> approx_bound(const std::vector<Document>& docs,
        const MatrixXd& doc_topic_distr, bool sub_sampling) {
        int n_docs = docs.size();
        int n_topics = n_topics_;
        int n_features = n_features_;

        double score1 = 0.0; // expected log likelihood
        double score2 = 0.0; // document-topic prior term
        double score3 = 0.0; // topic-word prior term

        // 1. Expected log likelihood: E[log p(docs | theta, beta)]
        #pragma omp parallel for reduction(+:score1)
        for (int d = 0; d < n_docs; d++) {
            const Document& doc = docs[d];
            // Compute sum of gamma for document d.
            double sum_gamma = doc_topic_distr.row(d).sum();
            for (size_t idx = 0; idx < doc.ids.size(); idx++) {
                int word_id = doc.ids[idx];
                double count = doc.cnts[idx];

                // Compute logsumexp over topics:
                double max_val = -std::numeric_limits<double>::infinity();
                std::vector<double> temp_vals(n_topics);
                for (int k = 0; k < n_topics; k++) {
                    // Document part: psi(gamma_dk) - psi(sum(gamma))
                    double doc_term = psi(doc_topic_distr(d, k)) - psi(sum_gamma);
                    // Topic-word part: psi(components_(k,word_id)) - psi(sum_j components_(k,j))
                    double row_sum = components_.row(k).sum();
                    double topic_term = psi(components_(k, word_id)) - psi(row_sum);
                    double combined = doc_term + topic_term;
                    temp_vals[k] = combined;
                    if (combined > max_val) {
                        max_val = combined;
                    }
                }
                double sum_exp = 0.0;
                for (int k = 0; k < n_topics; k++) {
                    sum_exp += std::exp(temp_vals[k] - max_val);
                }
                double word_log_sum_exp = max_val + std::log(sum_exp);
                score1 += count * word_log_sum_exp;
            }
        }

        // 2. Document-topic prior term: E[log p(theta | alpha) - log q(theta | gamma)]
        #pragma omp parallel for reduction(+:score2)
        for (int d = 0; d < n_docs; d++) {
            Eigen::VectorXd gamma = doc_topic_distr.row(d).transpose();
            double sum_gamma = gamma.sum();
            double psi_sum = psi(sum_gamma);
            double doc_score = 0.0;
            for (int k = 0; k < n_topics; k++) {
                double gamma_val = gamma[k];
                doc_score += (doc_topic_prior_ - gamma_val) * (psi(gamma_val) - psi_sum);
                doc_score += std::lgamma(gamma_val) - std::lgamma(doc_topic_prior_);
            }
            doc_score += std::lgamma(doc_topic_prior_ * n_topics) - std::lgamma(sum_gamma);
            score2 += doc_score;
        }
        if (sub_sampling) {
            double doc_ratio = static_cast<double>(total_doc_count_) / n_docs;
            score1 *= doc_ratio;
            score2 *= doc_ratio;
        }

        // 3. Topic-word prior term: E[log p(beta | eta) - log q(beta | lambda)]
        #pragma omp parallel for reduction(+:score3)
        for (int k = 0; k < n_topics; k++) {
            double row_sum = components_.row(k).sum();
            double psi_row_sum = psi(row_sum);
            double topic_score = 0.0;
            for (int j = 0; j < n_features; j++) {
                double lambda_val = components_(k, j);
                topic_score += (topic_word_prior_ - lambda_val) * (psi(lambda_val) - psi_row_sum);
                topic_score += std::lgamma(lambda_val) - std::lgamma(topic_word_prior_);
            }
            topic_score += std::lgamma(topic_word_prior_ * n_features) - std::lgamma(row_sum);
            score3 += topic_score;
        }

        return {score1, score2, score3};
    }

    // Compute the score (variational bound) for the given documents.
    std::vector<double> score(const std::vector<Document>& docs) {
        // Compute document-topic distributions using your transform() method.
        MatrixXd doc_topic_distr = transform(docs);
        return approx_bound(docs, doc_topic_distr, false);
    }

    // Compute perplexity for a set of documents.
    double perplexity(const std::vector<Document>& docs, bool sub_sampling = false) {
        MatrixXd doc_topic_distr = transform(docs);
        return _perplexity_precomp_distr(docs, doc_topic_distr, sub_sampling);
    }

private:

    int n_topics_, n_features_;
    int seed_;
    int total_doc_count_;
    int nThreads_;
    MatrixXd components_; // lambda, K x M
    MatrixXd exp_Elog_beta_; // exp(E[log beta])
    double doc_topic_prior_; // alpha
    double topic_word_prior_; // eta
    double eps_;
    int max_doc_update_iter_;
    double mean_change_tol_; // for per document inner loop
    double learning_decay_; // kappa
    double learning_offset_; // tau_0
    int update_count_;
    int32_t verbose_;
    std::mt19937 random_engine_;

    // Update a single document's topic distribution.
    // doc: sparse representation of the document.
    // doc_topic: current/prior document-topic vector (K x 1).
    // Returns the updated document-topic vectors (K x 1).
    int32_t fit_one_document(VectorXd& doc_topic, VectorXd& exp_doc,
        const Document &doc,
        const std::optional<VectorXd>& doc_topic_ = std::nullopt)
    {

        if (!doc_topic_) {
            doc_topic.resize(n_topics_);
            std::gamma_distribution<double> gamma_dist(100.0, 0.01);
            for (int k = 0; k < n_topics_; k++) {
                doc_topic[k] = gamma_dist(random_engine_);
            }
        } else {
            doc_topic = *doc_topic_;
        }

        exp_doc.resize(n_topics_); // exp(psi(gamma) - psi(sum(gamma)))
        double sum = doc_topic.sum();
        double psi_total = psi(sum);
        for (int k = 0; k < n_topics_; k++) {
            exp_doc[k] = std::exp(psi(doc_topic[k]) - psi_total);
        }
        // Build a submatrix for the nonzero word indices in the document.
        int n_ids = doc.ids.size();
        MatrixXd exp_topic_word(n_topics_, n_ids);
        for (int j = 0; j < n_ids; j++) {
            exp_topic_word.col(j) = exp_Elog_beta_.col(doc.ids[j]);
        }
        // Iterative update for the document.
        for (int iter = 0; iter < max_doc_update_iter_; iter++) {
            // VectorXd last_doc = doc_topic; // Save the previous state.
            VectorXd last_doc = exp_doc;

            // norm_phi = exp_doc^T * exp_topic_word (|ids| x 1).
            VectorXd norm_phi = exp_topic_word.transpose() * exp_doc;
            norm_phi.array() += eps_; // Avoid division by zero.

            VectorXd ratio(n_ids);
            for (int j = 0; j < n_ids; j++) {
                ratio[j] = doc.cnts[j] / norm_phi[j];
            }
            VectorXd new_vector = exp_topic_word * ratio; // K x 1

            doc_topic = exp_doc.array() * new_vector.array();
            // Dirichlet expectation update:
            // Add the prior, compute the total and update exp_doc.
            exp_doc = dirichlet_expectation_1d(doc_topic, doc_topic_prior_);

            // Check convergence via mean absolute change.
            double diff = (last_doc - exp_doc).cwiseAbs().sum() / n_topics_;
            if (diff < mean_change_tol_) {
                return iter;
            }
        }
        return max_doc_update_iter_;
    }

    // Compute perplexity from precomputed document-topic distributions.
    double _perplexity_precomp_distr(const std::vector<Document>& docs,
        const MatrixXd& doc_topic_distr, bool sub_sampling) {
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

    void init(const std::optional<MatrixXd>& topic_word_distr = std::nullopt) {
        if (nThreads_ > 0) {
            omp_set_num_threads(nThreads_);
        } else {
            nThreads_ = omp_get_max_threads();
        }
        if (mean_change_tol_ < 0) {
            mean_change_tol_ = 0.001 / n_topics_;
        }
        if (seed_ <= 0) {
            seed_ = std::random_device{}();
        }
        random_engine_.seed(seed_);
        eps_ = std::numeric_limits<double>::epsilon();

        if (topic_word_distr) {
            components_ = *topic_word_distr;
        } else {
            components_.resize(n_topics_, n_features_);
            std::gamma_distribution<double> gamma_dist(100.0, 0.01);
            for (int k = 0; k < n_topics_; k++) {
                for (int j = 0; j < n_features_; j++) {
                    components_(k, j) = gamma_dist(random_engine_);
                }
            }
        }
        exp_Elog_beta_ = dirichlet_expectation_2d(components_);
        if (doc_topic_prior_ < 0) {
            doc_topic_prior_ = 1. / n_topics_;
        }
        if (topic_word_prior_ < 0) {
            topic_word_prior_ = 1. / n_topics_;
        }
        if (total_doc_count_ <= 0) {
            total_doc_count_ = 1000000;
        }
        if (verbose_) {
            notice("LDA initialized with %d topics, %d features, %d threads", n_topics_, n_features_, nThreads_);
        }
    }

};
