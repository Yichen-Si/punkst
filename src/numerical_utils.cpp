#include "numerical_utils.hpp"

// Calculate the mean absolute difference between two arrays.
double mean_change(const std::vector<double>& arr1, const std::vector<double>& arr2) {
    if (arr1.size() != arr2.size()) {
        return std::numeric_limits<double>::max();
    }
    double total = 0.0;
    size_t size = arr1.size();
    for (size_t i = 0; i < size; i++) {
        total += std::abs(arr1[i] - arr2[i]);
    }
    return total / size;
}

// Psi (digamma) function (not optimized for maximum accuracy)
double psi(double x) {
    const double EULER = 0.577215664901532860606512090082402431;
    if (x <= 1e-6) {
        // psi(x) ~ -EULER - 1/x when x is very small
        return -EULER - 1.0 / x;
    }
    double result = 0.0;
    // Increment x until it is large enough
    while (x < 6) {
        result -= 1.0 / x;
        x += 1;
    }
    double r = 1.0 / x;
    result += std::log(x) - 0.5 * r;
    r = r * r;
    result -= r * ((1.0/12.0) - r * ((1.0/120.0) - r * (1.0/252.0)));
    return result;
}

Eigen::VectorXd expect_log_sticks(const Eigen::VectorXd& alpha,
                                  const Eigen::VectorXd& beta) {
    assert(alpha.size() == beta.size() && "alpha and beta must have same length");
    int K = alpha.size();
    int N = K + 1;

    // ψ(α_j + β_j)
    Eigen::VectorXd dig_sum(K);
    for (int j = 0; j < K; ++j) {
        dig_sum[j] = psi(alpha[j] + beta[j]);
    }
    // ElogW_j    = ψ(α_j) - ψ(α_j + β_j)
    // Elog1_W_j  = ψ(β_j) - ψ(α_j + β_j)
    Eigen::VectorXd ElogW(K), Elog1_W(K);
    for (int j = 0; j < K; ++j) {
        ElogW[j]    = psi(alpha[j]) - dig_sum[j];
        Elog1_W[j]  = psi(beta[j])  - dig_sum[j];
    }
    // ElogSigma_k = ElogW_k + \sum_{l=1}^{k-1} Elog1_W_l
    Eigen::VectorXd result = ElogW;
    // accumulate Elog1_W into positions 1…K
    double running = 0.0;
    for (int j = 0; j < K - 1; ++j) {
        running += Elog1_W[j];
        result[j+1] += running;
    }
    return result;
}

// exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
// = exp(psi(\alpha_k) - psi(\alpha_0))
void dirichlet_expectation_1d(std::vector<double>& alpha, std::vector<double>& out, double offset) {
    if (offset < 1e-6) {
        offset = 1e-6;
    }
    size_t size = alpha.size();
    double total = 0.0;
    // Add the prior and compute total
    for (size_t i = 0; i < size; i++) {
        alpha[i] += offset;
        total += alpha[i];
    }
    double psi_total = psi(total);
    // Compute the exponentiated psi differences.
    for (size_t i = 0; i < size; i++) {
        out[i] = std::exp(psi(alpha[i]) - psi_total);
    }
}
