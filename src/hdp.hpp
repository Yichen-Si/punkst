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
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::RowVectorXd;
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

class HDP {
public:

    std::vector<std::string> feature_names_;

    HDP(int32_t K, int32_t T, int32_t M, int32_t nThreads = 1,
        int seed = std::random_device{}(), int verbose = 0,
        double eta = 0.01, double alpha = 1, double omega = 1,
        int32_t total_docs = 1000000,
        double learning_offset = 10, double learning_decay = 0.9,
        int max_doc_update_iter = 100, double mean_change_tol = 0.001
        )
      : K_(K), T_(T), M_(M), nThreads_(nThreads),
        seed_(seed), verbose_(verbose),
        eta_(eta), alpha_(alpha), omega_(omega),
        total_doc_count_(total_docs),
        learning_offset_(learning_offset), learning_decay_(learning_decay),
        max_doc_update_iter_(max_doc_update_iter),
        mean_change_tol_(mean_change_tol),
        update_count_(0)
    {
        init();
    }

    const RowMajorMatrixXd& get_model() const {
        return lambda_;
    }
    RowMajorMatrixXd copy_model() const {
        return lambda_;
    }
    int32_t get_K() const {
        return K_;
    }
    int32_t get_n_features() const {
        return M_;
    }
    int32_t get_N_global() const {
        return total_doc_count_;
    }
    void get_topic_abundance(std::vector<double>& weights) const {
        weights.resize(K_);
        for (int k = 0; k < K_; k++) {
            weights[k] = lambda_.row(k).sum();
        }
        double total = std::accumulate(weights.begin(), weights.end(), 0.0);
        for (int k = 0; k < K_; k++) {
            weights[k] /= total;
        }
    }

    // Process a minibatch and update global parameters
    void partial_fit(const std::vector<Document>& docs);

    std::vector<int32_t> sort_topics();

    // Transform: compute document-topic distributions for a list of documents
    MatrixXd transform(const std::vector<Document>& docs);

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

private:

    int K_; // Maximum number of topics
    int T_; // Minimum number of topics per document
    int M_; // Vocabulary size
    int seed_;
    int nThreads_;
    double eps_;
    int32_t verbose_;
    std::mt19937 random_engine_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;
    std::vector<int32_t> sorted_indices_;

    // Priors/concentration parameters
    double eta_; // prior for beta (topic-word prior)
    double alpha_; // for local breaking probabilities (pi~Beta(1, alpha))
    double omega_; // for global breaking probabilities (nu~Beta(1, omega))
    // Global VB parameters
    RowMajorMatrixXd lambda_; // K x M, for beta (topic-word)
    MatrixXd Elog_beta_; // exp(E[log beta])
    VectorXd aK_, bK_; // K x 1, for nu (breaking prob)
    VectorXd Elog_sigma_K_; // E[log sigma(nu)], K x 1
    // Online learning parameters
    int total_doc_count_;    // expected total number of documents
    double learning_decay_ ; // kappa
    double learning_offset_; // tau
    int update_count_; // number of processed minibatches
    // Local iteration parameters
    int max_doc_update_iter_; // for per document inner loop
    double mean_change_tol_ ; // for per document inner loop

    // Check parameters and initialize global variables
    void init();

    // Update a single document's topic distribution (sparse version)
    // doc: sparse representation of the document
    // Return zeta: T x K; phi: M' x T
    VectorXd fit_one_document(MatrixXd& zeta, MatrixXd& phi, const Document &doc);

};
