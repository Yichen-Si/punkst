#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "Eigen/Dense"
#include "Eigen/Sparse"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::SparseMatrix;

// Calculate the mean absolute difference between two arrays.
double mean_change(const std::vector<double>& arr1, const std::vector<double>& arr2);

double mean_max_row_change(const MatrixXd& arr1, const MatrixXd& arr2);

// Psi (digamma) function (not optimized for maximum accuracy)
double psi(double x);

// exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
// = exp(psi(\alpha_k) - psi(\alpha_0))
void dirichlet_expectation_1d(std::vector<double>& alpha, std::vector<double>& out, double offset = 0);

// Vector version
VectorXd dirichlet_expectation_1d(VectorXd& alpha, double offset = 0);

// row-wise exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
MatrixXd dirichlet_expectation_2d(const MatrixXd& alpha);

// row-wise E[log X] for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
MatrixXd dirichlet_entropy_2d(const MatrixXd& alpha);

// If b is provided, it must be of the same size as a.
std::pair<double, double> logsumexp(const VectorXd &a, const VectorXd* b = nullptr);

inline double expit(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

inline double logit(double x) {
    if (x <= 0.0 || x >= 1.0) {
        throw std::out_of_range("Input to logit must be in (0, 1)");
    }
    return std::log(x / (1.0 - x));
}

// Element-wise expit then row-normalize
template<typename SparseMatrixType>
void expitAndRowNormalize(SparseMatrixType& mat) {
    static_assert(SparseMatrixType::IsRowMajor, "normalizeRows requires a row-major sparse matrix.");
    using Scalar = typename SparseMatrixType::Scalar;
    for (int i = 0; i < mat.rows(); ++i) {
        Scalar rowSums = Scalar(0);
        for (typename SparseMatrixType::InnerIterator it(mat, i); it; ++it) {
            double transformed = expit(it.value());
            it.valueRef() = Scalar(transformed);
            rowSums += transformed;
        }
        if (rowSums == Scalar(0)) {
            continue;
        }
        for (typename SparseMatrixType::InnerIterator it(mat, i); it; ++it) {
            it.valueRef() /= rowSums;
        }
    }
}

// row-normalize
template<typename SparseMatrixType>
void rowNormalize(SparseMatrixType& mat) {
    static_assert(SparseMatrixType::IsRowMajor, "normalizeRows requires a row-major sparse matrix.");
    using Scalar = typename SparseMatrixType::Scalar;
    for (int i = 0; i < mat.rows(); ++i) {
        Scalar rowSum = Scalar(0);
        for (typename SparseMatrixType::InnerIterator it(mat, i); it; ++it) {
            rowSum += it.value();
        }
        if (rowSum == Scalar(0)) {
            continue;
        }
        for (typename SparseMatrixType::InnerIterator it(mat, i); it; ++it) {
            it.valueRef() /= rowSum;
        }
    }
}

// find largest values and indices
void findTopK(MatrixXd& topVals, Eigen::MatrixXi& topIds, const MatrixXd& mtx, int32_t k);
