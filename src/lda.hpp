#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <cstdlib>
#include <iostream>
#include <random>
#include <optional>
#include <numeric>
#include "numerical_utils.hpp"
#include "dataunits.hpp"
#include "error.hpp"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>

#include <Eigen/Core>
#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;
using Eigen::RowVectorXd;
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

enum class InferenceType { SVB, SVB_DN, SCVB0 };

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
            const std::string* mfile = nullptr,
            const std::optional<RowMajorMatrixXd>& topic_word_distr = std::nullopt, double pariorScale = -1.)
        : n_topics_(n_topics), n_features_(n_features), seed_(seed),
        nThreads_(nThreads), verbose_(verbose), algo_(algo),
        alpha_(doc_topic_prior),
        eta_(topic_word_prior),
        learning_decay_(learning_decay), learning_offset_(learning_offset),
        total_doc_count_(total_doc_count),
        update_count_(0) {
        if (mfile && !(*mfile).empty()) {
            set_model_from_tsv(*mfile, pariorScale);
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
        RowMajorMatrixXd& modelMtx,
        int seed = std::random_device{}(), int nThreads = 0, int verbose = 0,
        InferenceType algo = InferenceType::SVB) : seed_(seed),
        nThreads_(nThreads), verbose_(verbose), algo_(algo),
        total_doc_count_(-1), update_count_(0) {
        set_nthreads(nThreads);
        set_model_from_matrix(modelMtx);
        init();
    }

    InferenceType get_algorithm() const {
        return algo_;
    }
    const RowMajorMatrixXd& get_model() const {
        return components_;
    }
    const VectorXd& get_background_model() const {
        assert(algo_ == InferenceType::SVB_DN);
        return lambda0_;
    }
    RowMajorMatrixXd copy_model() const {
        return components_;
    }
    VectorXd copy_background_model() const {
        assert(algo_ == InferenceType::SVB_DN);
        return lambda0_;
    }
    double get_forground_count() const {
        assert(algo_ == InferenceType::SVB_DN);
        return b_;
    }
    double get_background_count() const {
        assert(algo_ == InferenceType::SVB_DN);
        return a_;
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
    const std::vector<std::string>& get_topic_names();
    void get_topic_abundance(std::vector<double>& weights) const;
    void sort_topics();

    void set_nthreads(int nThreads);

    // Set engine specific parameters
    void set_svb_parameters(int32_t max_iter = 100, double tol = -1.);
    void set_scvb0_parameters(double s_beta = 1, double s_theta = 1, double tau_theta = 10, double kappa_theta = 0.9, int32_t burnin = 10);
    void set_background_prior(const VectorXd& eta0, double a0, double b0, bool fixed = false);
    void set_background_prior(const std::vector<double> eta0, double a0, double b0, bool fixed = false);

    // Set the model matrix
    void set_model_from_matrix(std::vector<std::vector<double>>& lambdaVals);
    void set_model_from_matrix(RowMajorMatrixXd& lambda);
    // Read a model matrix from file
    void set_model_from_tsv(const std::string& modelFile, double scalar = -1.);

    // process a mini-batch of documents to update the global topic-word distribution.
    void partial_fit(const std::vector<Document>& docs) {
        switch (algo_) {
            case InferenceType::SCVB0:
                scvb0_partial_fit(docs);
                break;
            case InferenceType::SVB_DN:
                svbdn_partial_fit(docs);
                break;
            case InferenceType::SVB:
                svb_partial_fit(docs);
                break;
            default:
                error("%s: Unknown inference type", __func__);
        }
    }
    void svb_partial_fit(const std::vector<Document>& docs);
    void svbdn_partial_fit(const std::vector<Document>& docs);
    void scvb0_partial_fit(const std::vector<Document>& docs);

    // Transform: compute document-topic distributions for a list of documents
    // For transform, we do not compute or return sufficient statistics.
    RowMajorMatrixXd transform(const std::vector<Document>& docs) {
        return transform_common(docs, [](const auto& d) -> const Document& { return d; });
    }
    RowMajorMatrixXd transform(const std::vector<SparseObs>& docs) {
        return transform_common(docs, [](const auto& d) -> const Document& { return d.doc; });
    }

    // SVB only
    // Compute the score (variational bound) for the given documents
    std::vector<double> score(const std::vector<Document>& docs) {
        assert(algo_ == InferenceType::SVB);
        // Compute document-topic distributions using your transform() method.
        MatrixXd gamma = transform(docs);
        return approx_bound(docs, gamma, false);
    }
    // Compute perplexity for a set of documents.
    double perplexity(const std::vector<Document>& docs, bool sub_sampling = false) {
        assert(algo_ == InferenceType::SVB);
        MatrixXd gamma = transform(docs);
        return _perplexity_precomp_distr(docs, gamma, sub_sampling);
    }

private:

    template <typename Docs, typename DocAccessor>
    RowMajorMatrixXd transform_common(const Docs& docs, DocAccessor&& doc_of) {
        const int n_docs = static_cast<int>(docs.size());
        const int ncol = algo_ == InferenceType::SVB_DN ? n_topics_ + 1 : n_topics_;
        RowMajorMatrixXd gamma(n_docs, ncol);

        auto process_doc = [&](int d) {
            const Document& doc = doc_of(docs[d]);
            switch (algo_) {
                case InferenceType::SCVB0: {
                    VectorXd hatNk;
                    scvb0_fit_one_document(hatNk, doc);
                    hatNk /= hatNk.sum();
                    gamma.row(d) = hatNk.transpose();
                    break;
                }
                case InferenceType::SVB_DN: {
                    VectorXd gamma_d, exp_Elog_theta_d;
                    ArrayXd fg_counts;
                    (void)svbdn_fit_one_document(gamma_d, exp_Elog_theta_d, doc, fg_counts);
                    gamma_d /= gamma_d.sum();
                    double c = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
                    double bg = 1. - fg_counts.sum() / c;
                    gamma(d, 0) = bg;
                    for (int32_t k = 0; k < n_topics_; ++k) {
                        gamma(d, k + 1) = gamma_d(k);
                    }
                    break;
                }
                case InferenceType::SVB: {
                    VectorXd gamma_d, exp_Elog_theta_d;
                    (void)svb_fit_one_document(gamma_d, exp_Elog_theta_d, doc);
                    gamma_d /= gamma_d.sum();
                    gamma.row(d) = gamma_d.transpose();
                    break;
                }
                default:
                    error("%s: Unknown inference type", __func__);
            }
        };

        if (nThreads_ == 1) {
            for (int d = 0; d < n_docs; ++d) process_doc(d);
        } else {
            tbb::parallel_for(0, n_docs, [&](int d) { process_doc(d); });
        }
        return gamma;
    }

    InferenceType algo_;
    int n_topics_ = -1, n_features_ = -1;
    int seed_;
    int total_doc_count_;
    int nThreads_;
    RowMajorMatrixXd components_; // lambda in SVB or N_kw in SCVB, K x M
    double alpha_ = -1; // prior for theta
    double eta_ = -1; // prior for beta
    double eps_;
    double learning_decay_  = -1; // kappa
    double learning_offset_ = -1; // tau
    int update_count_; // number of processed minibatches
    int32_t verbose_;
    std::mt19937 random_engine_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;

    // SVB specific parameters
    MatrixXd exp_Elog_beta_; // exp(E[log beta]), K x M
    int max_doc_update_iter_ = -1; // for per document inner loop
    double mean_change_tol_  = -1; // for per document inner loop
    // SVB_DN specific parameters
    bool fix_background_ = false;
    double a0_, b0_; // prior for background proportion pi
    double a_, b_;
    VectorXd eta0_; // prior for background distribution, M x 1
    VectorXd lambda0_, exp_Elog_beta0_; // background distribution, M x 1

    // SCVB0 specific parameters
    double s_beta_ = 1, s_theta_ = 1;
    double tau_theta_ = 10, kappa_theta_ = 0.9;
    int32_t burn_in_ = 10;
    VectorXd Nk_;

    // Update a single document's topic distribution.
    // doc: sparse representation of the document.
    // doc_topic: current/prior document-topic vector (K x 1).
    // Returns the updated document-topic vectors (K x 1).
    int32_t svb_fit_one_document(VectorXd& gamma, VectorXd& exp_Elog_theta,
        const Document &doc);
    int32_t svbdn_fit_one_document(VectorXd& gamma, VectorXd& exp_Elog_theta,
        const Document &doc, ArrayXd& fg_counts);
    void scvb0_fit_one_document(MatrixXd& hatNkw, const Document& doc);
    void scvb0_fit_one_document(VectorXd& hatNk, const Document& doc);

    void init_model(const std::optional<RowMajorMatrixXd>& topic_word_distr = std::nullopt, double scalar = -1.);
    void compute_global_mtx();
    void init();

    // SCVB0 specific
    // SCVB0 latent variable update (Eq. 5 for gamma_{ijk} in the paper)
    inline void scvb0_one_word(
            const uint32_t w, const VectorXd& NTheta_j, VectorXd& phi) const {
        // (β_kw + η) / (n_k + M*η) * (N_jk + α)
        VectorXd model = (components_.col(w).array() + eta_) / (Nk_.array() + n_features_ * eta_);
        phi = (components_.col(w).array() + eta_)
              * (NTheta_j.array() + alpha_)
              / (Nk_.array() + n_features_ * eta_);
        phi /= phi.sum();
    }

    // SVB specific
    // Compute the approximate variational bound
    std::vector<double> approx_bound(const std::vector<Document>& docs,
        const MatrixXd& gamma, bool sub_sampling);
    // Compute perplexity from precomputed document-topic distributions.
    double _perplexity_precomp_distr(const std::vector<Document>& docs,
        const MatrixXd& gamma, bool sub_sampling);
};
