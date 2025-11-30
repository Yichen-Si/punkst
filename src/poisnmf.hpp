#pragma once

#include "poisreg.hpp"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>

struct TestResult {
  int m, k1, k2;    // for null-mean test, set k2 = -1
  double est;       // beta_k1 - beta_k2 or beta_k - beta0
  double log10p;      // log p value from Wald
  double fc;        // approximated fold-change
};

struct NmfFitOptions {
    int    max_iter     = 50;
    double tol          = 1e-4;
    double covar_coef_min = -1e6;
    double covar_coef_max =  1e6;
    int    n_mb_epoch   = 0;
    int    batch_size   = 1024;
    bool   shuffle      = true;
    bool   use_decay    = true;
    double t0           = -1;  // decay offset
    double kappa        = 0.7; // decay exponent
    // How often to rescale beta & theta
    int  rescale_period = 1;
};

class PoissonLog1pNMF {
public:

    PoissonLog1pNMF(int K, int M, int nThreads = -1,
        int seed = std::random_device{}(), bool exact = true, int debug = 0) :
        K_(K), M_(M), seed_(seed), exact_(exact), debug_(debug) {
        set_nthreads(nThreads);
        rng_.seed(seed_);
    }

    void fit(const std::vector<SparseObs>& docs,
        const MLEOptions mle_opts, NmfFitOptions nmf_opts, bool reset = false, std::vector<int32_t>* labels = nullptr);

    int32_t partial_fit(
        const std::vector<SparseObs>& docs, const std::vector<Document>& mtx_t,
        const MLEOptions mle_opts, const MLEOptions mle_opts_bcov,
        const std::vector<uint32_t>& batch_indices,
        double rho_t, int32_t mb_count, int32_t total_epoch);

    RowMajorMatrixXd transform(std::vector<SparseObs>& docs,
        const MLEOptions mle_opts, std::vector<MLEStats>& res, ArrayXd* fres_ptr = nullptr);

    std::vector<TestResult> test_beta_vs_null(int flag);
    std::vector<TestResult> test_beta_pairwise(int flag);
    std::vector<TestResult> test_beta_pairwise(int flag, std::pair<int, int> kpair);

    const RowMajorMatrixXd& get_model() const { return beta_; }
    const RowMajorMatrixXd& get_theta() const { return theta_; }
    const RowMajorMatrixXd& get_covar_coef() const { return Bcov_; }
    const RowMajorMatrixXd& get_se(bool robust = false) const { return robust ? se_robust_ : se_fisher_; }
    const std::vector<double>& get_feature_sums() const { return feature_sums_; }
    const std::vector<double>& get_feature_residuals() const { return feature_residuals_; }

    void set_nthreads(int nThreads);
    void set_de_parameters(double min_ct = 100, double min_fc = 1.5, double max_p = 0.05);
    void set_beta(RowMajorMatrixXd& beta) {
        beta_ = std::move(beta);
        M_ = beta_.rows();
        K_ = beta_.cols();
    }
    void set_beta_se(RowMajorMatrixXd& se_mtx, int32_t robust = 0) {
        if (se_mtx.rows() != M_ || se_mtx.cols() != K_) {
            error("%s: SE matrix has incorrect dimensions", __func__);
        }
        if (robust) se_robust_ = std::move(se_mtx);
        else se_fisher_ = std::move(se_mtx);
    }
    void set_covar_coef(RowMajorMatrixXd& Bcov) {
        if (Bcov.rows() != M_) {
            error("%s: Covariate coefficient matrix has incorrect number of rows", __func__);
        }
        Bcov_ = std::move(Bcov);
        P_ = Bcov_.cols();
    }
    void clear_theta() {
        theta_.resize(0,0);
        theta_valid_.clear();
    }

    RowMajorMatrixXd convert_to_factor_loading();

private:
    int K_, M_, P_ = 0;
    int seed_;
    int nThreads_;
    bool exact_;
    RowMajorMatrixXd beta_;  // M x K
    RowMajorMatrixXd theta_; // N x K
    RowMajorMatrixXd Bcov_;  // M x P (per-feature covariate coefficients)
    std::mt19937 rng_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;
    int debug_;
    RowMajorMatrixXd se_fisher_, se_robust_; // M x K, SE of beta
    std::vector<MatrixXd> cov_fisher_, cov_robust_;
    std::vector<double> feature_sums_;
    std::vector<double> feature_residuals_;
    std::vector<bool> theta_valid_;
    double min_ct_ = 100, min_fc_ = 1.5, max_p_ = 0.05; // for DE tests
    double min_logfc_ = 0.176, max_log10p_ = -1.3; // from above
    std::vector<double> tr_delta_bcov_, tr_delta_beta_; // for minibatch
    bool fit_background_ = false;
    double pi_;
    VectorXd beta0_;

    std::vector<Document> transpose_data(const std::vector<SparseObs>& docs, VectorXd& cvec);

    double update_beta(const std::vector<Document>& mtx_t,
        const VectorXd& cvec, RowMajorMatrixXd& X,
        const MLEOptions& mle_opts, int32_t& niters_beta);
    double update_bcov(const std::vector<Document>& mtx_t,
        const VectorXd& cvec, RowMajorMatrixXd& X,
        const MLEOptions& mle_opts, int32_t& niters_bcov);
    int32_t update_theta(const std::vector<SparseObs>& docs,
        const MLEOptions& mle_opts, int32_t& niters, double& obj_avg,
        const std::vector<uint32_t>* batch_indices = nullptr);

        // negative log-likelihood
    double calculate_global_objective(const std::vector<SparseObs>& docs, RowMajorMatrixXd* Xptr = nullptr);
    // rescales theta and beta to improve numerical stability
    void rescale_matrices();
    void rescale_beta_to_const_sum(double c = 1.0);
    void rescale_theta_to_const_sum(double c = 1.0);

};
