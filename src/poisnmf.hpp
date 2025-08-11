#pragma once

#include "poisreg.hpp"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>

class PoissonLog1pNMF {
public:

    PoissonLog1pNMF(int K, int M, double c, int nThreads = -1, int seed = std::random_device{}()) : K_(K), M_(M), c_(c), seed_(seed) {
        set_nthreads(nThreads);
        rng_.seed(seed_);
    }

    void fit(const std::vector<Document>& docs,
        const MLEOptions mle_opts,
        int max_iter = 100, double tol = 1e-6);

    RowMajorMatrixXd transform(const std::vector<Document>& docs,
        const MLEOptions mle_opts);

    const RowMajorMatrixXd& get_model() const { return beta_; }
    const RowMajorMatrixXd& get_theta() const { return theta_; }
    void set_nthreads(int nThreads);
    void rescale_beta_to_unit_sum();
    void rescale_theta_to_unit_sum();
    RowMajorMatrixXd convert_to_factor_loading();

private:
    int K_, M_;
    int seed_;
    int nThreads_;
    double c_;
    RowMajorMatrixXd beta_;  // M x K
    RowMajorMatrixXd theta_; // N x K
    std::mt19937 rng_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;

    // transpose sparse representation
    std::vector<Document> transpose_data(const std::vector<Document>& docs);
    // negative log-likelihood
    double calculate_global_objective(const std::vector<Document>& docs);
    // rescales theta and beta to improve numerical stability
    void rescale_matrices();

};
