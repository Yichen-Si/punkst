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

#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;

class LatentDirichletAllocation {
public:
    LatentDirichletAllocation(int n_topics, int n_features,
                              int seed = std::random_device{}(),
                              int nThreads = 0,
                              double doc_topic_prior = -1., // alpha
                              double topic_word_prior = -1., // eta
                              int max_doc_update_iter = 100,
                              double mean_change_tol = 1e-3,
                              double learning_decay = 0.7, // kappa
                              double learning_offset = 10.0, // tau_0
                              int total_doc_count = 1000000,
                              const std::optional<Eigen::MatrixXd>& topic_word_distr = std::nullopt)
        : n_topics_(n_topics), n_features_(n_features), nThreads_(nThreads),
        doc_topic_prior_(doc_topic_prior),
        topic_word_prior_(topic_word_prior),
        max_doc_update_iter_(max_doc_update_iter),
        mean_change_tol_(mean_change_tol),
        learning_decay_(learning_decay), learning_offset_(learning_offset),
        total_doc_count_(total_doc_count),
        update_count_(0)
    {
        if (nThreads_ > 0) {
            omp_set_num_threads(nThreads_);
        } else {
            nThreads_ = omp_get_max_threads();
        }
        random_engine_.seed(seed);
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
    }

    // Online update
    // process a mini-batch of documents to update the global topic-word distribution.
    void partial_fit(const std::vector<Document>& docs) {
        int minibatch_size = docs.size();
        MatrixXd ss = MatrixXd::Zero(n_topics_, n_features_);

        #pragma omp parallel
        {
            MatrixXd local_ss = MatrixXd::Zero(n_topics_, n_features_);
            #pragma omp for
            for (int d = 0; d < minibatch_size; d++) {
                // Use the shared routine to get the document's variational parameters.
                auto [doc_topic, exp_doc] = fit_one_document(docs[d]);
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
            VectorXd updated_doc = fit_one_document(docs[d]).first;
            doc_topic_distr.row(d) = updated_doc.transpose();
        }
        return doc_topic_distr;
    }

    const MatrixXd& get_model() const {
        return components_;
    }
    int32_t get_n_topics() const {
        return n_topics_;
    }
    int32_t get_n_features() const {
        return n_features_;
    }

private:

    int n_topics_, n_features_;
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
    std::mt19937 random_engine_;

    // Update a single document's topic distribution.
    // doc: sparse representation of the document.
    // doc_topic: current/prior document-topic vector (K x 1).
    // Returns the updated document-topic vectors (K x 1).
    std::pair<VectorXd, VectorXd> fit_one_document(const Document &doc,
        const std::optional<Eigen::VectorXd>& doc_topic_ = std::nullopt)
    {

        VectorXd doc_topic;
        if (!doc_topic_) {
            doc_topic = VectorXd(n_topics_);
            std::gamma_distribution<double> gamma_dist(100.0, 0.01);
            for (int k = 0; k < n_topics_; k++) {
                doc_topic[k] = gamma_dist(random_engine_);
            }
        } else {
            doc_topic = *doc_topic_;
        }

        VectorXd exp_doc(n_topics_); // exp(psi(gamma) - psi(sum(gamma)))
        double sum = doc_topic.sum();
        double psi_total = psi(sum);
        for (int k = 0; k < n_topics_; k++) {
            exp_doc[k] = std::exp(psi(doc_topic[k]) - psi_total);
        }

        // Build a submatrix for the nonzero word indices in the document.
        int n_ids = doc.ids.size();
        MatrixXd exp_topic_word(n_topics_, n_ids);
        for (int j = 0; j < n_ids; j++) {
            exp_topic_word.col(j) = components_.col(doc.ids[j]);
        }

        // Iterative update for the document.
        for (int iter = 0; iter < max_doc_update_iter_; iter++) {
            VectorXd last_doc = doc_topic; // Save the previous state.

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
            double diff = (last_doc - doc_topic).cwiseAbs().sum() / n_topics_;
            if (diff < mean_change_tol_) {
                break;
            }
        }
        return {doc_topic, exp_doc};
    }

};
