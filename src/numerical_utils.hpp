#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"

// Calculate the mean absolute difference between two arrays.
double mean_change(const std::vector<double>& arr1, const std::vector<double>& arr2);

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
double psi(double x);

Eigen::VectorXd expect_log_sticks(const Eigen::VectorXd& alpha,
                                  const Eigen::VectorXd& beta);

// exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
// = exp(psi(\alpha_k) - psi(\alpha_0))
void dirichlet_expectation_1d(std::vector<double>& alpha, std::vector<double>& out, double offset = -1);

// Vector version
template<typename Derived>
Derived dirichlet_expectation_1d(const Eigen::MatrixBase<Derived>& alpha,
                                 typename Derived::Scalar offset = Derived::Scalar(-1))
{
    using Scalar = typename Derived::Scalar;
    Derived tmp = alpha.derived();
    if (offset < 1e-6) {
        offset = 1e-6;
    }
    tmp.array() += offset;
    Scalar total = tmp.sum();
    double psi_total = psi(static_cast<double>(total));
    for (int i = 0; i < tmp.size(); ++i) {
        tmp(i) = Scalar(std::exp(psi(static_cast<double>(tmp(i))) - psi_total));
    }
    return tmp;
}

// row-wise exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
template<typename Derived>
Derived dirichlet_expectation_2d(const Eigen::MatrixBase<Derived>& alpha)
{
    double eps = 1e-6;
    using Scalar = typename Derived::Scalar;
    Derived result(alpha.rows(), alpha.cols());
    for (int i = 0; i < alpha.rows(); ++i) {
        auto row = alpha.row(i);
        Scalar total = row.sum();
        double psi_total = psi(static_cast<double>(total) + eps);
        for (int j = 0; j < alpha.cols(); ++j) {
            result(i,j) = Scalar(
                std::exp(psi(static_cast<double>(alpha(i,j)) + eps) - psi_total)
            );
        }
    }
    return result;
}

// row-wise E[log X] for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
template<typename Derived>
Derived dirichlet_entropy_2d(const Eigen::MatrixBase<Derived>& alpha)
{
    double eps = 1e-6;
    using Scalar = typename Derived::Scalar;
    Derived result(alpha.rows(), alpha.cols());
    int32_t K = alpha.cols();
    for (int i = 0; i < alpha.rows(); ++i) {
        auto row = alpha.row(i);
        Scalar total = row.sum();
        double psi_total = psi(static_cast<double>(total) + eps * K);
        for (int j = 0; j < alpha.cols(); ++j) {
            result(i,j) = Scalar(
                psi(static_cast<double>(alpha(i,j)) + eps) - psi_total
            );
        }
    }
    return result;
}

// If b is provided, it must be of the same size as a.
template<typename Derived>
std::pair<typename Derived::Scalar, typename Derived::Scalar>
logsumexp(const Eigen::MatrixBase<Derived>& a,
          const Eigen::MatrixBase<Derived>* b = nullptr)
{
    using Scalar = typename Derived::Scalar;
    if (a.size() == 0) {
        return { -std::numeric_limits<Scalar>::infinity(), Scalar(0) };
    }
    Scalar maxVal = a.maxCoeff();
    if (!std::isfinite(maxVal)) {
        return { -std::numeric_limits<Scalar>::infinity(), Scalar(0) };
    }
    Scalar sumExp = Scalar(0);
    Scalar sign = Scalar(0);
    if (!b) {
        for (int i = 0; i < a.size(); ++i) {
            sumExp += std::exp(a(i) - maxVal);
        }
        sign = Scalar(1);
    } else {
        for (int i = 0; i < a.size(); ++i) {
            auto term = (*b)(i) * std::exp(a(i) - maxVal);
            sumExp += term;
        }
        sign = sumExp > 0 ? Scalar(1) : (sumExp < 0 ? Scalar(-1) : Scalar(0));
        sumExp = std::abs(sumExp);
    }
    Scalar lse = maxVal + std::log(sumExp);
    return { lse, sign };
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
