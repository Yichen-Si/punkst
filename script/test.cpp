#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#include "punkst.h"
#include "utils.h"
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

#include "poisnmf.hpp"

/**
 * @brief Calculates the relative L2 error between two matrices.
 * Formula: ||true_matrix - est_matrix||_F / ||true_matrix||_F
 */
double calculate_relative_error(const RowMajorMatrixXd& true_matrix, const RowMajorMatrixXd& est_matrix) {
    if (true_matrix.rows() != est_matrix.rows() || true_matrix.cols() != est_matrix.cols()) {
        throw std::runtime_error("Matrix dimensions must match for error calculation.");
    }
    double norm_diff = (true_matrix - est_matrix).norm();
    double norm_true = true_matrix.norm();
    if (norm_true < 1e-9) {
        // If the true matrix is essentially zero, return the norm of the estimated one.
        return norm_diff;
    }
    return norm_diff / norm_true;
}


/**
 * @brief Rescales theta and beta to handle the inherent scaling ambiguity in NMF.
 * This function puts the matrices into a canonical form for meaningful comparison.
 * We normalize by making the columns of beta sum to 1.
 */
void rescale_for_ambiguity(RowMajorMatrixXd& theta, RowMajorMatrixXd& beta) {
    if (theta.cols() != beta.cols()) {
         throw std::runtime_error("Theta and Beta must have the same number of columns (K) for rescaling.");
    }
    for (int k = 0; k < beta.cols(); ++k) {
        double col_sum = beta.col(k).sum();
        if (col_sum > 1e-9) {
            theta.col(k) *= col_sum;
            beta.col(k) /= col_sum;
        }
    }
}

/**
 * @brief Calculates the global objective function (negative log-likelihood) for a given set of parameters.
 */
double calculate_global_objective_test(
    const std::vector<SparseObs>& docs,
    const RowMajorMatrixXd& theta,
    const RowMajorMatrixXd& beta,
    const RowMajorMatrixXd& Bcov,
    const RowMajorMatrixXd& X)
{
    double total_log_likelihood = 0.0;
    const int P = X.cols();
    int N = docs.size();

    for (size_t i = 0; i < docs.size(); ++i) {
        const auto& doc_obs = docs[i];
        for (size_t k = 0; k < doc_obs.doc.ids.size(); ++k) {
            int j = doc_obs.doc.ids[k];

            double eta = theta.row(i).dot(beta.row(j));
            if (P > 0) {
                eta += X.row(i).dot(Bcov.row(j));
            }
            eta = std::max(0.0, eta);

            double lambda = doc_obs.c * (std::exp(eta) - 1.0);
            lambda = std::max(1e-12, lambda); // for numerical stability

            total_log_likelihood += (doc_obs.doc.cnts[k] * std::log(lambda) - lambda) / N;
        }
    }
    // The objective is the negative log-likelihood
    return -total_log_likelihood;
}

/**
 * @brief Finds the best permutation of columns in est_matrix to match true_matrix.
 * This solves the column permutation ambiguity problem for error reporting.
 * @return A vector `p` where `p[i] = j` means true column `i` maps to estimated column `j`.
 */
std::vector<int> find_best_permutation(const RowMajorMatrixXd& true_matrix, const RowMajorMatrixXd& est_matrix) {
    const int K = true_matrix.cols();
    RowMajorMatrixXd similarity(K, K);

    // Calculate cosine similarity between columns
    for (int i = 0; i < K; ++i) {
        for (int j = 0; j < K; ++j) {
            double dot_product = true_matrix.col(i).dot(est_matrix.col(j));
            double norm_true = true_matrix.col(i).norm();
            double norm_est = est_matrix.col(j).norm();
            if (norm_true > 1e-9 && norm_est > 1e-9) {
                similarity(i, j) = dot_product / (norm_true * norm_est);
            } else {
                similarity(i, j) = 0.0;
            }
        }
    }

    std::vector<int> p(K);
    std::vector<bool> est_col_used(K, false);
    // Greedily find the best match for each true column
    for (int i = 0; i < K; ++i) {
        int best_j = -1;
        double max_sim = -2.0;
        for (int j = 0; j < K; ++j) {
            if (!est_col_used[j] && similarity(i, j) > max_sim) {
                max_sim = similarity(i, j);
                best_j = j;
            }
        }
        p[i] = best_j;
        est_col_used[best_j] = true;
    }
    return p;
}

/**
 * @brief Calculates the Mean Squared Error of the reconstructed linear predictor, eta.
 * This metric is invariant to NMF scaling ambiguity and is a better measure of model fit.
 */
double calculate_eta_mse(
    const RowMajorMatrixXd& theta_true, const RowMajorMatrixXd& beta_true, const RowMajorMatrixXd& Bcov_true,
    const RowMajorMatrixXd& theta_est, const RowMajorMatrixXd& beta_est, const RowMajorMatrixXd& Bcov_est,
    const RowMajorMatrixXd& X)
{
    RowMajorMatrixXd eta_true = theta_true * beta_true.transpose() + X * Bcov_true.transpose();
    RowMajorMatrixXd eta_est = theta_est * beta_est.transpose() + X * Bcov_est.transpose();

    return (eta_true - eta_est).squaredNorm() / eta_true.size();
}


// --- Data Simulation ---

/**
 * @brief Simulates data according to the Poisson NMF model with covariates.
 */
void simulate_nmf_data(
    int N, int M, int K, int P,
    std::vector<SparseObs>& docs,
    RowMajorMatrixXd& theta_true,
    RowMajorMatrixXd& beta_true,
    RowMajorMatrixXd& Bcov_true,
    RowMajorMatrixXd& X_true)
{
    std::mt19937 rng(12345); // Fixed seed for reproducibility
    std::gamma_distribution<double> gamma_dist(0.5, 1.0); // Use smaller values
    std::normal_distribution<double> normal_dist(0.0, 0.5); // Use smaller variance

    theta_true.resize(N, K);
    beta_true.resize(M, K);
    Bcov_true.resize(M, P);
    X_true.resize(N, P);

    for (int i = 0; i < N; ++i) for (int k = 0; k < K; ++k) theta_true(i, k) = gamma_dist(rng);
    for (int j = 0; j < M; ++j) for (int k = 0; k < K; ++k) beta_true(j, k) = gamma_dist(rng);
    for (int j = 0; j < M; ++j) for (int p = 0; p < P; ++p) Bcov_true(j, p) = normal_dist(rng);
    for (int i = 0; i < N; ++i) for (int p = 0; p < P; ++p) X_true(i, p) = normal_dist(rng);

    // NEW: Normalize the true factors to prevent numerical overflow during simulation
    double avg_eta_scale = (theta_true * beta_true.transpose() + X_true * Bcov_true.transpose()).mean();
    double scale_factor = std::sqrt(std::abs(avg_eta_scale));
    if (scale_factor > 1.0) {
        theta_true /= scale_factor;
        beta_true /= scale_factor;
        Bcov_true /= scale_factor; // Scale covariate effects as well
    }

    docs.resize(N);
    for (int i = 0; i < N; ++i) {
        docs[i].c = 1.0;
        docs[i].covar = X_true.row(i);

        for (int j = 0; j < M; ++j) {
            double eta = theta_true.row(i).dot(beta_true.row(j));
            if (P > 0) {
                eta += X_true.row(i).dot(Bcov_true.row(j));
            }
            eta = std::max(0.0, eta);
            double lambda = docs[i].c * (std::exp(eta) - 1.0);

            std::poisson_distribution<int> poisson_dist(lambda);
            int count = poisson_dist(rng);

            if (count > 0) {
                docs[i].doc.ids.push_back(j);
                docs[i].doc.cnts.push_back(static_cast<double>(count));
            }
        }
    }
}

int32_t test(int32_t argc, char** argv) {

    int32_t N, K, M, P;
    int32_t seed, debug_ = 0;
    int32_t threads = 1;
    bool exact = false;

    ParamList pl;
    // Input / sim options
    pl.add_option("N", "Number of rows (N)", N, true)
      .add_option("K", "Number of cols (K)", K, true)
      .add_option("M", "Number of features (M)", M, true)
      .add_option("P", "Number of covariates (P)", P, true)
      .add_option("seed", "Random seed", seed, true)
      .add_option("threads", "Number of threads", threads)
      .add_option("exact", "", exact);

    // Output options
    pl.add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    std::cout << "--- Poisson NMF Test Suite ---" << std::endl;
    std::cout << "Simulating data with N=" << N << ", M=" << M << ", K=" << K << ", P=" << P << std::endl;

// --- 1. Simulate Data ---
    std::vector<SparseObs> docs;
    RowMajorMatrixXd theta_true, beta_true, Bcov_true, X_true;
    simulate_nmf_data(N, M, K, P, docs, theta_true, beta_true, Bcov_true, X_true);
    double true_objective_canonical = calculate_global_objective_test(docs, theta_true, beta_true, Bcov_true, X_true);
    std::cout << "True objective: " << true_objective_canonical << std::endl;


    // --- 2. Configure and Fit Model ---
    PoissonLog1pNMF model(K, M, threads, seed, exact);

    MLEOptions mle_opts;
    mle_opts.optim.max_iters = 15; // A few more iterations can help
    mle_opts.optim.tol = 1e-5;
    mle_opts.optim.tron.enabled = true;

    std::cout << "\nFitting the model..." << std::endl;
    auto start_time = std::chrono::high_resolution_clock::now();
    model.fit(docs, mle_opts, 50, 1e-4, -1e6, 1e6);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> runtime = end_time - start_time;
    std::cout << "Fitting complete in " << runtime.count() << " ms." << std::endl;

    // --- 3. Evaluate Results ---
    RowMajorMatrixXd theta_est = model.get_theta();
    RowMajorMatrixXd beta_est = model.get_model();
    RowMajorMatrixXd Bcov_est = model.get_covar_coef();

    double final_objective = calculate_global_objective_test(docs, theta_est, beta_est, Bcov_est, X_true);
    double eta_error = calculate_eta_mse(theta_true, beta_true, Bcov_true, theta_est, beta_est, Bcov_est, X_true);
    // Create copies for canonical comparison
    RowMajorMatrixXd theta_true_canon = theta_true;
    RowMajorMatrixXd beta_true_canon = beta_true;
    rescale_for_ambiguity(theta_true_canon, beta_true_canon);

    RowMajorMatrixXd theta_est_canon = theta_est;
    RowMajorMatrixXd beta_est_canon = beta_est;
    rescale_for_ambiguity(theta_est_canon, beta_est_canon);

    // NEW: Find best permutation after canonical scaling
    std::vector<int> p = find_best_permutation(theta_true_canon, theta_est_canon);
    RowMajorMatrixXd theta_est_aligned(N, K);
    RowMajorMatrixXd beta_est_aligned(M, K);
    for(int k=0; k<K; ++k) {
        theta_est_aligned.col(k) = theta_est_canon.col(p[k]);
        beta_est_aligned.col(k) = beta_est_canon.col(p[k]);
    }

    true_objective_canonical = calculate_global_objective_test(docs, theta_true_canon, beta_true_canon, Bcov_true, X_true);

    // Calculate matrix errors after alignment
    double theta_error = calculate_relative_error(theta_true_canon, theta_est_aligned);
    double beta_error = calculate_relative_error(beta_true_canon, beta_est_aligned);
    double bcov_error = calculate_relative_error(Bcov_true, Bcov_est);

    std::cout << "\n--- Evaluation Results ---" << std::endl;
    printf("Total Runtime: %.2f ms\n", runtime.count());
    printf("------------------------------------------------\n");
    printf("Primary Metric (Invariant to Ambiguities):\n");
    printf("  - Linear Predictor (eta) MSE | %.6f\n", eta_error);
    printf("------------------------------------------------\n");
    printf("Objective Function (Comparison):\n");
    printf("  - True Objective (Canonical) | %.4e\n", true_objective_canonical);
    printf("  - Final Estimated Objective  | %.4e\n", final_objective);
    printf("  - Rel Difference             | %.4f\n", std::abs(final_objective - true_objective_canonical) / std::abs(true_objective_canonical));
    printf("------------------------------------------------\n");
    printf("Matrix Errors (After Alignment & Rescaling):\n");
    printf("  - Theta (NxK)    | %.6f\n", theta_error);
    printf("  - Beta (MxK)     | %.6f\n", beta_error);
    printf("  - Bcov (MxP)     | %.6f\n", bcov_error);
    std::cout << "------------------------------------------------" << std::endl;

    return 0;
}
