#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <random>
#include <optional>
#include "numerical_utils.hpp"
#include "dataunits.hpp"
#include "error.hpp"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>

#include "Eigen/Dense"
using Eigen::MatrixXf;
using Eigen::VectorXf;
using Eigen::RowVectorXf;
using Eigen::MatrixXd;
using Eigen::VectorXd;

enum class InferenceType { SVB, SCVB0 };

class LatentDirichletAllocation {
public:

    std::vector<std::string> topic_names_;
    std::vector<std::string> feature_names_;

    LatentDirichletAllocation(int n_topics, int n_features,
                              int seed = std::random_device{}(),
                              int nThreads = 0, int verbose = 0,
                              InferenceType algo = InferenceType::SVB,
                              double doc_topic_prior = -1., // alpha
                              double topic_word_prior = -1., // eta
                              double learning_decay = -1., // kappa
                              double learning_offset = -1., // tau
                              int total_doc_count = 1000000,
            std::optional<std::reference_wrapper<std::string>> mfileptr = std::nullopt,
            const std::optional<MatrixXf>& topic_word_distr = std::nullopt, double pariorScale = -1.)
        : n_topics_(n_topics), n_features_(n_features), seed_(seed),
        nThreads_(nThreads), verbose_(verbose), algo_(algo),
        doc_topic_prior_(doc_topic_prior),
        topic_word_prior_(topic_word_prior),
        learning_decay_(learning_decay), learning_offset_(learning_offset),
        total_doc_count_(total_doc_count),
        update_count_(0) {
        if (mfileptr && !(mfileptr->get()).empty()) {
            set_model_from_tsv(mfileptr->get(), pariorScale);
        } else {
            init_model(topic_word_distr);
        }
        init();
    }

    // Meant for loading a pre-trained model
    LatentDirichletAllocation(
        const std::string& modelFile,
        int seed = std::random_device{}(), int nThreads = 0, int verbose = 0,
        InferenceType algo = InferenceType::SVB) : seed_(seed),
        nThreads_(nThreads), verbose_(verbose), algo_(algo),
        total_doc_count_(-1), update_count_(0) {
        set_nthreads(nThreads);
        set_model_from_tsv(modelFile);
        init();
    }
    LatentDirichletAllocation(
        MatrixXf& modelMtx,
        int seed = std::random_device{}(), int nThreads = 0, int verbose = 0,
        InferenceType algo = InferenceType::SVB) : seed_(seed),
        nThreads_(nThreads), verbose_(verbose), algo_(algo),
        total_doc_count_(-1), update_count_(0) {
        set_nthreads(nThreads);
        init_model(modelMtx);
        init();
    }

    const MatrixXf& get_model() const {
        return components_;
    }
    MatrixXf copy_model() const {
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
    const std::vector<std::string>& get_topic_names() {
        if (topic_names_.empty()) {
            topic_names_.resize(n_topics_);
            for (int i = 0; i < n_topics_; i++) {
                topic_names_[i] = std::to_string(i);
            }
        }
        return topic_names_;
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
    void sort_topics() {
        // sort topics by decreasing total weight
        VectorXf topic_weights = components_.rowwise().sum();
        std::vector<int> indices(n_topics_);
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(),
                  [&topic_weights](int a, int b) {
            return topic_weights[a] > topic_weights[b];
        });
        // update components_, exp_Elog_beta_, or Nk_
        MatrixXf sorted_components(n_topics_, n_features_);
        for (int i = 0; i < n_topics_; i++) {
            sorted_components.row(i) = components_.row(indices[i]);
        }
        components_ = std::move(sorted_components);
        if (algo_ == InferenceType::SCVB0) {
            Nk_ = components_.rowwise().sum();
        } else if (algo_ == InferenceType::SVB) {
            exp_Elog_beta_ = dirichlet_expectation_2d(sorted_components);
        }
        // topic_names_
        if (!topic_names_.empty()) {
            std::vector<std::string> sorted_topic_names(n_topics_);
            for (int i = 0; i < n_topics_; i++) {
                sorted_topic_names[i] = topic_names_[indices[i]];
            }
            topic_names_ = std::move(sorted_topic_names);
        }
    }

    void set_nthreads(int nThreads) {
        nThreads_ = nThreads;
        if (nThreads_ > 0) {
            tbb_ctrl_ = std::make_unique<tbb::global_control>(
               tbb::global_control::max_allowed_parallelism,
               std::size_t(nThreads_));
          } else {
            tbb_ctrl_.reset();
          }
    }

    // Set engine specific parameters
    void set_svb_parameters(int32_t max_iter = 100, double tol = -1.);
    void set_scvb0_parameters(double s_beta = 1, double s_theta = 1, double tau_theta = 10, double kappa_theta = 0.9, int32_t burnin = 10);

    // Set the model matrix
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
        if (algo_ == InferenceType::SCVB0) {
            Nk_ = components_.rowwise().sum();
        } else {
            exp_Elog_beta_ = dirichlet_expectation_2d(components_);
        }
    }
    void set_model_from_matrix(MatrixXf& lambda) {
        if (lambda.rows() != n_topics_ || lambda.cols() != n_features_) {
            warning("Model matrix size mismatch, reset according to the provided global parameters. (%d x %d) -> (%d x %d)", n_topics_, n_features_, lambda.rows(), lambda.cols());
            n_topics_ = lambda.rows();
            n_features_ = lambda.cols();
        }
        notice("Global variational parameters are reset, but online training status (if any) is not. It is only safe for transform.");
        components_ = lambda;
        if (algo_ == InferenceType::SCVB0) {
            Nk_ = components_.rowwise().sum();
        } else {
            exp_Elog_beta_ = dirichlet_expectation_2d(components_);
        }
    }
    // Read a model matrix from file
    void set_model_from_tsv(const std::string& modelFile, double scalar = -1.) {
        std::ifstream modelIn(modelFile, std::ios::in);
        if (!modelIn) {
            error("Error opening model file: %s", modelFile.c_str());
        }
        std::string line;
        std::vector<std::string> tokens;
        std::getline(modelIn, line);
        split(tokens, "\t", line);
        int32_t K = tokens.size() - 1;
        feature_names_.clear();
        topic_names_.resize(K);
        for (int32_t i = 0; i < K; ++i) {
            topic_names_[i] = tokens[i + 1];
        }
        std::vector<std::vector<double>> modelValues;
        while (std::getline(modelIn, line)) {
            split(tokens, "\t", line);
            if (tokens.size() != K + 1) {
                error("Error reading model file at line ", line.c_str());
            }
            feature_names_.push_back(tokens[0]);
            std::vector<double> values(K);
            for (int32_t i = 0; i < K; ++i) {
                values[i] = std::stod(tokens[i + 1]);
            }
            modelValues.push_back(values);
        }
        modelIn.close();

        n_topics_ = K;
        n_features_ = feature_names_.size();
        notice("Read %d topics and %d features from model file", n_topics_, n_features_);

        components_.resize(n_topics_, n_features_);
        for (uint32_t i = 0; i < n_features_; ++i) {
            for (int32_t j = 0; j < K; ++j) {
                components_(j, i) = modelValues[i][j];
            }
        }
        if (scalar > 0.) {
            components_ *= scalar;
        }
        if (algo_ == InferenceType::SCVB0) {
            Nk_ = components_.rowwise().sum();
        } else {
            exp_Elog_beta_ = dirichlet_expectation_2d(components_);
        }
    }

    // process a mini-batch of documents to update the global topic-word distribution.
    void partial_fit(const std::vector<Document>& docs) {
        if (algo_ == InferenceType::SCVB0) {
            scvb0_partial_fit(docs);
        } else {
            svb_partial_fit(docs);
        }
    }
    void svb_partial_fit(const std::vector<Document>& docs);
    void scvb0_partial_fit(const std::vector<Document>& docs);

    // Transform: compute document-topic distributions for a list of documents
    // For transform, we do not compute or return sufficient statistics.
    MatrixXf transform(const std::vector<Document>& docs) {
        int n_docs = docs.size();
        MatrixXf doc_topic_distr(n_docs, n_topics_);
        // Parallel update: process each document independently
        if (algo_ == InferenceType::SCVB0) {
            tbb::parallel_for(0, n_docs, [&](int d) {
                // Update document d using the helper function.
                VectorXf hatNk;
                scvb0_fit_one_document(hatNk, docs[d]);
                doc_topic_distr.row(d) = hatNk.transpose();
            });
        } else {
            tbb::parallel_for(0, n_docs, [&](int d) {
                // Update document d using the helper function.
                VectorXf updated_doc, exp_doc;
                int32_t niter = svb_fit_one_document(updated_doc, exp_doc, docs[d]);
                doc_topic_distr.row(d) = updated_doc.transpose();
            });
        }
        return doc_topic_distr;
    }

    // SVB only
    // Compute the score (variational bound) for the given documents
    std::vector<double> score(const std::vector<Document>& docs) {
        assert(algo_ == InferenceType::SVB);
        // Compute document-topic distributions using your transform() method.
        MatrixXf doc_topic_distr = transform(docs);
        return approx_bound(docs, doc_topic_distr, false);
    }
    // Compute perplexity for a set of documents.
    double perplexity(const std::vector<Document>& docs, bool sub_sampling = false) {
        assert(algo_ == InferenceType::SVB);
        MatrixXf doc_topic_distr = transform(docs);
        return _perplexity_precomp_distr(docs, doc_topic_distr, sub_sampling);
    }

private:

    InferenceType algo_;
    int n_topics_ = -1, n_features_ = -1;
    int seed_;
    int total_doc_count_;
    int nThreads_;
    MatrixXf components_; // lambda in SVB or N_kw in SCVB, K x M
    double doc_topic_prior_  = -1; // alpha
    double topic_word_prior_ = -1; // eta
    double eps_;
    double learning_decay_  = -1; // kappa
    double learning_offset_ = -1; // tau
    int update_count_; // number of processed minibatches
    int32_t verbose_;
    std::mt19937 random_engine_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;

    // SVB specific parameters
    MatrixXf exp_Elog_beta_; // exp(E[log beta])
    int max_doc_update_iter_ = -1; // for per document inner loop
    double mean_change_tol_  = -1; // for per document inner loop

    // SCVB0 specific parameters
    double s_beta_ = 1, s_theta_ = 1;
    double tau_theta_ = 10, kappa_theta_ = 0.9;
    int32_t burn_in_ = 10;
    VectorXf Nk_;

    // Update a single document's topic distribution.
    // doc: sparse representation of the document.
    // doc_topic: current/prior document-topic vector (K x 1).
    // Returns the updated document-topic vectors (K x 1).
    int32_t svb_fit_one_document(VectorXf& doc_topic, VectorXf& exp_doc,
        const Document &doc,
        const std::optional<VectorXf>& doc_topic_ = std::nullopt);
    void scvb0_fit_one_document(MatrixXf& hatNkw, const Document& doc);
    void scvb0_fit_one_document(VectorXf& hatNk, const Document& doc);

    void init_model(const std::optional<MatrixXf>& topic_word_distr = std::nullopt, double scalar = -1.) {
        if (topic_word_distr && topic_word_distr->rows() > 0
                             && topic_word_distr->cols() > 0) {
            components_ = *topic_word_distr;
            n_topics_ = components_.rows();
            n_features_ = components_.cols();
            if (scalar > 0.) {
                components_ *= scalar;
            }
        } else {
            components_.resize(n_topics_, n_features_);
            std::gamma_distribution<double> gamma_dist(100.0, 0.01);
            for (int k = 0; k < n_topics_; k++) {
                for (int j = 0; j < n_features_; j++) {
                    components_(k, j) = gamma_dist(random_engine_);
                }
            }
        }
        if (algo_ == InferenceType::SCVB0) {
            Nk_ = components_.rowwise().sum();
        } else {
            exp_Elog_beta_ = dirichlet_expectation_2d(components_);
        }
    }

    void init() {
        set_nthreads(nThreads_);
        if (seed_ <= 0) {
            seed_ = std::random_device{}();
        }
        random_engine_.seed(seed_);
        eps_ = std::numeric_limits<double>::epsilon();
        if (doc_topic_prior_ < 0) {
            doc_topic_prior_ = 1. / n_topics_;
        }
        if (topic_word_prior_ < 0) {
            topic_word_prior_ = 1. / n_topics_;
        }
        if (total_doc_count_ <= 0) {
            total_doc_count_ = 1000000;
        }
        if (learning_decay_ <= 0) {
            learning_decay_ = 0.7; // kappa
        }
        if (learning_offset_ < 0) {
            learning_offset_ = 10.0; // tau
        }
        if (algo_ == InferenceType::SCVB0) {
            if (s_beta_ > std::pow(learning_offset_ + 1, learning_decay_) ) {
                s_beta_ = 1;
            }
        } else {
            if (mean_change_tol_ <= 0) {
                mean_change_tol_ = 0.001;
            }
            if (max_doc_update_iter_ <= 0) {
                max_doc_update_iter_ = 100;
            }
        }
        if (verbose_) {
            notice("LDA initialized with %d topics, %d features, %d threads", n_topics_, n_features_, nThreads_);
        }
    }

    // SCVB0 specific
    // SCVB0 latent variable update (Eq. 5 for gamma_{ijk} in the paper)
    inline void scvb0_one_word(
            const uint32_t w, const VectorXf& NTheta_j, VectorXf& phi) const {
        // (β_kw + η) / (n_k + M*η) * (N_jk + α)
        VectorXf model = (components_.col(w).array() + topic_word_prior_) / (Nk_.array() + n_features_ * topic_word_prior_);
        phi = (components_.col(w).array() + topic_word_prior_)
              * (NTheta_j.array() + doc_topic_prior_)
              / (Nk_.array() + n_features_ * topic_word_prior_);
        phi /= phi.sum();
    }

    // SVB specific
    // Compute the approximate variational bound
    std::vector<double> approx_bound(const std::vector<Document>& docs,
        const MatrixXf& doc_topic_distr, bool sub_sampling);
    // Compute perplexity from precomputed document-topic distributions.
    double _perplexity_precomp_distr(const std::vector<Document>& docs,
        const MatrixXf& doc_topic_distr, bool sub_sampling) {
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
};
