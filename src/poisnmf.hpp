#pragma once

#include <numeric>
#include "poisreg.hpp"
#include <tbb/parallel_for.h>
#include <tbb/parallel_reduce.h>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>

using SparseRowMat = Eigen::SparseMatrix<double, Eigen::RowMajor>;
using SparseColMat = Eigen::SparseMatrix<double>;

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

    PoissonLog1pNMF(int K, int M, int nThreads = -1, double L = 10000,
        int seed = std::random_device{}(), bool exact = true, int debug = 0) :
        K_(K), M_(M), size_factor_(L), seed_(seed), exact_(exact), debug_(debug) {
        set_nthreads(nThreads);
        rng_.seed(seed_);
    }

    void fit(const std::vector<SparseObs>& docs,
        const MLEOptions& mle_opts, NmfFitOptions& nmf_opts, bool reset = false, std::vector<int32_t>* labels = nullptr);

    RowMajorMatrixXd transform(std::vector<SparseObs>& docs,
        const MLEOptions mle_opts, std::vector<MLEStats>& res, ArrayXd* fres_ptr = nullptr);

    std::vector<TestResult> test_beta_vs_null(int flag);
    std::vector<TestResult> test_beta_pairwise(int flag);
    std::vector<TestResult> test_beta_pairwise(int flag, std::pair<int, int> kpair);

    const RowMajorMatrixXd& get_model() const { return beta_; }
    const RowMajorMatrixXd& get_theta() const { return theta_; }
    const RowMajorMatrixXd& get_covar_coef() const { return Bcov_; }
    const VectorXd& get_bg_model() const { return beta0_; }
    const std::vector<double>& get_bg_proportions() const { return phi_; }
    double get_pi() const { return pi_; }
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
    void set_background_model(double pi0, VectorXd* beta0 = nullptr, bool fixed = false) {
        fit_background_ = true;
        fix_background_ = fixed;
        assert(pi0 > 0 && pi0 < 1);
        pi0_ = pi0;
        pi_ = pi0; log_pi_ = std::log(pi0); log_1mpi_ = std::log(1 - pi0);
        if (beta0 != nullptr) {
            assert(beta0->size() == M_);
            beta0_ = *beta0;
        }
        beta0_ /= beta0_.sum();
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
    double size_factor_;
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
    MLEOptions mle_opts_bcov_, mle_opts_fit_;
    int32_t minibatch_ = 0;
    int32_t epoch_ = 0;

    size_t N_;
    std::vector<Document> mtx_t_;
    VectorXd cvec_;
    RowMajorMatrixXd X_;

    bool fit_background_ = false;
    bool fix_background_ = false;
    double pi0_, pi_, log_pi_, log_1mpi_;
    VectorXd beta0_;
    std::vector<double> phi_;
    std::vector<Document> mtx_;

    // M-step (returns -ll(nmf) / N_)
    double update_beta(int32_t& niters_beta);
    double update_bcov(int32_t& niters_bcov);
    // E-step
    double update_theta(const std::vector<SparseObs>& docs,
        int32_t& niters, const std::vector<uint32_t>* batch_indices = nullptr);
    // E-step with background proportions
    double update_local(const std::vector<SparseObs>& docs,
        int32_t& niters, double* obj0 = nullptr, const std::vector<uint32_t>* batch_indices = nullptr, double rho_t = -1);

    // Mini-batch update
    int32_t partial_fit(const std::vector<SparseObs>& docs,
        const std::vector<uint32_t>& batch_indices, double rho_t);
    // Initialize data objects
    double init_data(const std::vector<SparseObs>& docs,
        bool reset_beta = false, std::vector<int32_t>* labels = nullptr);
    // Initialization specific to the denoising mode
    void init_denoise(const std::vector<SparseObs>& docs);

    // Store feature-major data
    void transpose_data(const std::vector<SparseObs>& docs, std::vector<Document>& mtx_t, const std::vector<uint32_t>* row_idx = nullptr);
    void transpose_data(const std::vector<Document>& docs, std::vector<Document>& mtx_t, const std::vector<uint32_t>* row_idx = nullptr);

    // Compute SE of beta estimates
    void compute_beta_se(const MLEOptions& mle_opts);
    // negative log-likelihood
    double calculate_global_objective(const std::vector<SparseObs>& docs, RowMajorMatrixXd* Xptr = nullptr);
    // rescales theta and beta to improve numerical stability
    void rescale_matrices();
    void rescale_beta_to_const_sum(double c = 1.0);
    void rescale_theta_to_const_sum(double c = 1.0);

    template <typename Docs, typename DocAccessor, typename SumAccessor>
    void transpose_data_common(
        std::vector<Document>& mtx_t,
        const Docs& docs,
        const std::vector<uint32_t>* row_idx,
        DocAccessor&& doc_of,
        SumAccessor&& sum_of) {
        mtx_t.assign(M_, Document{});
        if (row_idx == nullptr) {
            feature_sums_.assign(M_, 0.0);
            const size_t N = docs.size();
            cvec_.resize(N);
            for (size_t i = 0; i < N; ++i) {
                const Document& doc = doc_of(docs[i]);
                cvec_(i) = sum_of(docs[i], doc) / size_factor_;
                for (size_t k = 0; k < doc.ids.size(); ++k) {
                    int j = doc.ids[k];
                    if (j >= M_) { continue; }
                    feature_sums_[j] += doc.cnts[k];
                    mtx_t[j].ids.push_back(static_cast<uint32_t>(i));
                    mtx_t[j].cnts.push_back(doc.cnts[k]);
                }
            }
            return;
        }
        const size_t N = row_idx->size();
        if (cvec_.size() < docs.size()) {
            cvec_.resize(docs.size());
        }
        for (size_t rr = 0; rr < N; ++rr) {
            const uint32_t i = (*row_idx)[rr];
            const Document& doc = doc_of(docs[i]);
            cvec_(i) = sum_of(docs[i], doc) / size_factor_;
            for (size_t k = 0; k < doc.ids.size(); ++k) {
                int j = doc.ids[k];
                if (j >= M_) { continue; }
                mtx_t[j].ids.push_back(static_cast<uint32_t>(rr));
                mtx_t[j].cnts.push_back(doc.cnts[k]);
            }
        }
    }

};
