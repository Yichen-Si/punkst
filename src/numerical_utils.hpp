#include <vector>
#include <cmath>

#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;

// Calculate the mean absolute difference between two arrays.
double mean_change(const std::vector<double>& arr1, const std::vector<double>& arr2) {
    double total = 0.0;
    size_t size = arr1.size();
    for (size_t i = 0; i < size; i++) {
        total += std::fabs(arr1[i] - arr2[i]);
    }
    return total / size;
}

// Psi (digamma) function optimized for speed (not maximum accuracy).
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

// Dirichlet expectation for a single document.
// Given doc_topic vector and a prior, updates doc_topic in-place and writes
// exp(psi(doc_topic[i]) - psi(total)) into out vector.
void dirichlet_expectation_1d(std::vector<double>& doc_topic,
                              double doc_topic_prior,
                              std::vector<double>& out) {
    size_t size = doc_topic.size();
    double total = 0.0;
    // Add the prior and compute total
    for (size_t i = 0; i < size; i++) {
        doc_topic[i] += doc_topic_prior;
        total += doc_topic[i];
    }
    double psi_total = psi(total);
    // Compute the exponentiated psi differences.
    for (size_t i = 0; i < size; i++) {
        out[i] = std::exp(psi(doc_topic[i]) - psi_total);
    }
}

// Vector version
VectorXd dirichlet_expectation_1d(VectorXd& doc_topic, double doc_topic_prior) {
    size_t size = doc_topic.size();
    VectorXd out(size);
    doc_topic.array() += doc_topic_prior;
    double total = doc_topic.sum();
    double psi_total = psi(total);
    for (size_t i = 0; i < size; i++) {
        out[i] = std::exp(psi(doc_topic[i]) - psi_total);
    }
    return out;
}

MatrixXd dirichlet_expectation_2d(const MatrixXd& comp) {
    MatrixXd result(comp.rows(), comp.cols());
    for (int i = 0; i < comp.rows(); i++) {
        double row_sum = comp.row(i).sum();
        double psi_row_sum = psi(row_sum);
        for (int j = 0; j < comp.cols(); j++) {
            result(i, j) = std::exp(psi(comp(i, j)) - psi_row_sum);
        }
    }
    return result;
}
