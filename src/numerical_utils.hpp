#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <cassert>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"

// Calculate the mean absolute difference between two arrays.
template<typename T>
T mean_change(const std::vector<T>& arr1, const std::vector<T>& arr2) {
    if (arr1.size() != arr2.size()) {
        return std::numeric_limits<T>::max();
    }
    T total = 0.0;
    size_t size = arr1.size();
    for (size_t i = 0; i < size; i++) {
        total += std::fabs(arr1[i] - arr2[i]);
    }
    return total / size;
}

template<typename Derived>
auto mean_max_row_change(const Eigen::MatrixBase<Derived>& arr1,
                         const Eigen::MatrixBase<Derived>& arr2)
    -> typename Derived::Scalar
{
    using Scalar = typename Derived::Scalar;
    if (arr1.rows() != arr2.rows() || arr1.cols() != arr2.cols()) {
        return std::numeric_limits<Scalar>::infinity();
    }
    Scalar total = Scalar(0);
    for (int i = 0; i < arr1.rows(); ++i) {
        total += (arr1.row(i) - arr2.row(i)).cwiseAbs().maxCoeff();
    }
    return total / arr1.rows();
}


// Psi (digamma) function (not optimized for maximum accuracy)
template<typename T>
T psi(T x) {
    const T EULER = 0.577215664901532860606512090082402431;
    if (x <= 1e-6) {
        // psi(x) ~ -EULER - 1/x when x is very small
        return -EULER - 1.0 / x;
    }
    T result = 0.0;
    // Increment x until it is large enough
    while (x < 6) {
        result -= 1.0 / x;
        x += 1;
    }
    T r = 1.0 / x;
    result += std::log(x) - 0.5 * r;
    r = r * r;
    result -= r * ((1.0/12.0) - r * ((1.0/120.0) - r * (1.0/252.0)));
    return result;
}

template<typename VectorType>
VectorType expect_log_sticks(const VectorType& alpha, const VectorType& beta) {
    assert(alpha.size() == beta.size() && "alpha and beta must have same length");
    using Scalar = typename VectorType::Scalar;
    const int K = alpha.size();
    // psi(alpha + beta)
    VectorType dig_sum = (alpha.array() + beta.array())
        .unaryExpr([](Scalar s){ return psi(s); });
    // ElogW_j    = psi(α_j) - psi(α_j + β_j)
    // Elog1_W_j  = psi(β_j) - psi(α_j + β_j)
    VectorType ElogW = alpha.array()
        .unaryExpr([](Scalar s){ return psi(s); }) - dig_sum.array();
    VectorType Elog1_W = beta.array()
        .unaryExpr([](Scalar s){ return psi(s); }) - dig_sum.array();
    // ElogSigma_k = ElogW_k + \sum_{l=1}^{k-1} Elog1_W_l
    VectorType result = ElogW;
    Scalar running_sum = 0;
    for (int j = 0; j < K - 1; ++j) {
        running_sum += Elog1_W(j);
        result(j + 1) += running_sum;
    }
    return result;
}

// exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
// = exp(psi(\alpha_k) - psi(\alpha_0))
template<typename T>
void dirichlet_expectation_1d(std::vector<T>& alpha, std::vector<T>& out, T offset = -1) {
    if (offset < 1e-6) {
        offset = 1e-6;
    }
    size_t size = alpha.size();
    T total = 0.0;
    // Add the prior and compute total
    for (size_t i = 0; i < size; i++) {
        alpha[i] += offset;
        total += alpha[i];
    }
    T psi_total = psi(total);
    // Compute the exponentiated psi differences.
    for (size_t i = 0; i < size; i++) {
        out[i] = std::exp(psi(alpha[i]) - psi_total);
    }
}

// Vector version
template<typename Derived>
Derived dirichlet_expectation_1d(const Eigen::MatrixBase<Derived>& alpha,
                                 typename Derived::Scalar offset = Derived::Scalar(-1)) {
    using Scalar = typename Derived::Scalar;
    if (offset < 0) {
        offset = 1e-6;
    }
    Derived tmp = alpha.array() + offset;
    Scalar psi_total = psi(tmp.sum());
    tmp = tmp.unaryExpr([](Scalar x) {return psi(x);});
    return (tmp.array() - psi_total).exp();
}

// row-wise exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
template<typename Derived>
Derived dirichlet_expectation_2d(const Eigen::MatrixBase<Derived>& alpha) {
    double eps = 1e-6;
    using Scalar = typename Derived::Scalar;
    int32_t K = alpha.cols();
    auto psi_totals = (alpha.rowwise().sum().array() + eps * K)
            .unaryExpr([](Scalar x){ return psi(x); }).matrix();
    Derived result = (alpha.array() + eps)
            .unaryExpr([](Scalar x){ return psi(x); });
    result.colwise() -= psi_totals;
    return result.array().exp();
}

// row-wise E[log X] for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
template<typename Derived>
Derived dirichlet_entropy_2d(const Eigen::MatrixBase<Derived>& alpha)
{
    double eps = 1e-6;
    using Scalar = typename Derived::Scalar;
    int32_t K = alpha.cols();
    auto psi_totals = (alpha.rowwise().sum().array() + eps * K)
            .unaryExpr([](Scalar x){ return psi(x); }).matrix();
    Derived result = (alpha.array() + eps)
            .unaryExpr([](Scalar x){ return psi(x); });
    result.colwise() -= psi_totals;
    return result;
}

template<typename Derived>
void rowwiseSoftmax(Eigen::MatrixBase<Derived>& mat)
{
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> maxCoeffs = mat.rowwise().maxCoeff();
    mat.colwise() -= maxCoeffs;
    mat = mat.array().exp();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> row_sums = mat.rowwise().sum();
    mat.array().colwise() /= row_sums.array();
}

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
template<typename Derived>
void findTopK(
    Eigen::Matrix<typename Derived::Scalar,
                  Eigen::Dynamic,Eigen::Dynamic>& topVals,
    Eigen::Matrix<int,Eigen::Dynamic,Eigen::Dynamic>& topIds,
    const Eigen::MatrixBase<Derived>& mtx,
    int k)
{
    using Scalar = typename Derived::Scalar;
    int nRows = mtx.rows();
    int nCols = mtx.cols();
    k = std::min(k, nCols);
    topVals.resize(nRows,k);
    topIds.resize(nRows,k);
    for (int i = 0; i < nRows; ++i) {
        std::vector<std::pair<Scalar,int>> rowData;
        rowData.reserve(nCols);
        for (int j = 0; j < nCols; ++j) {
            rowData.emplace_back(mtx(i,j), j);
        }
        std::partial_sort(rowData.begin(), rowData.begin()+k, rowData.end(),
            [](auto &a, auto &b){ return a.first > b.first; }
        );
        for (int j = 0; j < k; ++j) {
            topVals(i,j) = rowData[j].first;
            topIds(i,j) = rowData[j].second;
        }
    }
}
