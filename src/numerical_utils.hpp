#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <array>
#include <type_traits>
#include "error.hpp"

#include "Eigen/Core"
#include "Eigen/Dense"
#include "Eigen/Sparse"

template<typename Scalar>
using RealScalar = typename Eigen::NumTraits<Scalar>::Real;

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
void dirichlet_expectation_1d(std::vector<T>& alpha, std::vector<T>& out, T eps = 1e-6) {
    size_t size = alpha.size();
    T total = 0.0;
    // Add the prior and compute total
    for (size_t i = 0; i < size; i++) {
        alpha[i] += eps;
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
                                 double eps = 1e-6) {
    using Scalar = typename Derived::Scalar;
    Derived tmp = alpha.array() + eps;
    Scalar psi_total = psi(tmp.sum());
    tmp = tmp.unaryExpr([](Scalar x) {return psi(x);});
    return (tmp.array() - psi_total).exp();
}

template<typename Derived>
Derived dirichlet_entropy_1d(const Eigen::MatrixBase<Derived>& alpha,
                             double eps = 1e-6) {
    using Scalar = typename Derived::Scalar;
    Derived tmp = alpha.array() + eps;
    Scalar psi_total = psi(tmp.sum());
    tmp = tmp.unaryExpr([](Scalar x) {return psi(x);});
    return tmp.array() - psi_total;
}

// exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
template<typename Derived>
Derived dirichlet_expectation_2d(const Eigen::MatrixBase<Derived>& alpha, bool colwise = false, double eps = 1e-6) {
    using Scalar = typename Derived::Scalar;
    if (colwise) {
        int32_t K = alpha.rows();
        auto psi_totals = (alpha.colwise().sum().array() + eps * K)
                .unaryExpr([](Scalar x){ return psi(x); }).matrix();
        Derived result = (alpha.array() + eps)
                .unaryExpr([](Scalar x){ return psi(x); });
        result.rowwise() -= psi_totals;
        return result.array().exp();
    }
    int32_t K = alpha.cols();
    auto psi_totals = (alpha.rowwise().sum().array() + eps * K)
            .unaryExpr([](Scalar x){ return psi(x); }).matrix();
    Derived result = (alpha.array() + eps)
            .unaryExpr([](Scalar x){ return psi(x); });
    result.colwise() -= psi_totals;
    return result.array().exp();
}

// E[log X] for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
template<typename Derived>
Derived dirichlet_entropy_2d(const Eigen::MatrixBase<Derived>& alpha, bool colwise = false, double eps = 1e-6) {
    using Scalar = typename Derived::Scalar;
    if (colwise) {
        int32_t K = alpha.rows();
        auto psi_totals = (alpha.colwise().sum().array() + eps * K)
                .unaryExpr([](Scalar x){ return psi(x); }).matrix();
        Derived result = (alpha.array() + eps)
                .unaryExpr([](Scalar x){ return psi(x); });
        result.rowwise() -= psi_totals;
        return result;
    }
    int32_t K = alpha.cols();
    auto psi_totals = (alpha.rowwise().sum().array() + eps * K)
            .unaryExpr([](Scalar x){ return psi(x); }).matrix();
    Derived result = (alpha.array() + eps)
            .unaryExpr([](Scalar x){ return psi(x); });
    result.colwise() -= psi_totals;
    return result;
}

template<typename Derived>
void rowSoftmaxInPlace(Eigen::MatrixBase<Derived>& mat)
{
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> maxCoeffs = mat.rowwise().maxCoeff();
    mat.colwise() -= maxCoeffs;
    mat = mat.array().exp();
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> row_sums = mat.rowwise().sum();
    // safe inverse
    for (int i = 0; i < row_sums.size(); ++i) {
        if (row_sums(i) < std::numeric_limits<Scalar>::epsilon()) {
            row_sums(i) = Scalar(1);
        } else {
            row_sums(i) = Scalar(1) / row_sums(i);
        }
    }
    mat.array().colwise() *= row_sums.array();
}

template<typename Derived>
void softmaxInPlace(Eigen::MatrixBase<Derived>& vec) {
    using Scalar = typename Derived::Scalar;
    Scalar maxCoeff = vec.maxCoeff();
    vec.array() -= maxCoeff;
    vec = vec.array().exp();
    Scalar sum = vec.sum();
    if (sum <= std::numeric_limits<Scalar>::epsilon()) {
        vec.setConstant(Scalar(1) / vec.size());
    } else {
        vec /= sum;
    }
}

template <typename T>
inline T expit(T x) {
    // Handles large |x| without overflow.
    if (x >= T(0)) {
        const T z = std::exp(-x);
        return T(1) / (T(1) + z);
    } else {
        const T z = std::exp(x);
        return z / (T(1) + z);
    }
}

double logit(double x);
float logit(float x);

template <typename Scalar, int StorageOrder>
void rowNormalizeInPlace(Eigen::SparseMatrix<Scalar, StorageOrder>& mat, bool nonNeg = true) {
    using Real = RealScalar<Scalar>;
    if constexpr (StorageOrder == Eigen::RowMajor) {
        for (int i = 0; i < mat.outerSize(); ++i) {
            Real rowNormL1(0);
            // First pass: compute L1 norm for row i
            if (nonNeg) {
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, i); it; ++it) {
                    rowNormL1 += it.value();
                }
            } else {
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, i); it; ++it) {
                    rowNormL1 += std::abs(it.value());
                }
            }
            if (rowNormL1 > std::numeric_limits<Real>::epsilon()) {
                Real invRowNormL1 = Real(1) / rowNormL1;
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, i); it; ++it) {
                    it.valueRef() *= invRowNormL1;
                }
            }
        }
    } else {
        Eigen::Matrix<Real, Eigen::Dynamic, 1> rowNormsL1 = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(mat.rows());
        if (nonNeg) {
            for (int j = 0; j < mat.outerSize(); ++j) {
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, j); it; ++it) {
                    rowNormsL1(it.row()) += it.value();
                }
            }
        } else {
            for (int j = 0; j < mat.outerSize(); ++j) {
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, j); it; ++it) {
                    rowNormsL1(it.row()) += std::abs(it.value());
                }
            }
        }
        Eigen::Matrix<Real, Eigen::Dynamic, 1> invRowNormsL1 = Eigen::Matrix<Real, Eigen::Dynamic, 1>::Zero(mat.rows());
        for (int i = 0; i < mat.rows(); ++i) {
            if (rowNormsL1(i) > std::numeric_limits<Real>::epsilon()) {
                invRowNormsL1(i) = Real(1) / rowNormsL1(i);
            }
        }
        for (int j = 0; j < mat.outerSize(); ++j) {
            for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, j); it; ++it) {
                it.valueRef() *= invRowNormsL1(it.row());
            }
        }
    }
}

template <typename Scalar>
void rowSoftmaxInPlace(Eigen::SparseMatrix<Scalar, Eigen::RowMajor>& mat) {
    using Real = RealScalar<Scalar>;
    for (int i = 0; i < mat.outerSize(); ++i) {
        // First pass: find max for numerical stability
        Real rowMax = -std::numeric_limits<Real>::infinity();
        for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
            if (it.value() > rowMax) {
                rowMax = it.value();
            }
        }
        // Second pass: compute exponentials and sum
        Real rowSum(0);
        for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
            double expVal = std::exp(it.value() - rowMax);
            it.valueRef() = Scalar(expVal);
            rowSum += expVal;
        }
        // Third pass: normalize
        if (rowSum > std::numeric_limits<Real>::epsilon()) {
            Real invRowSum = Real(1) / rowSum;
            for (typename Eigen::SparseMatrix<Scalar, Eigen::RowMajor>::InnerIterator it(mat, i); it; ++it) {
                it.valueRef() *= invRowSum;
            }
        }
    }
}

template <typename Derived>
void rowNormalizeInPlace(Eigen::MatrixBase<Derived>& mat, bool nonNeg = true) {
    using Real = RealScalar<typename Derived::Scalar>;
    using RealVector = Eigen::Matrix<Real, Derived::RowsAtCompileTime, 1>;
    RealVector rowNorms;
    if (nonNeg) {
        rowNorms = mat.rowwise().sum();
    } else {
        rowNorms = mat.rowwise().template lpNorm<1>();
    }
    RealVector invRowNorms = rowNorms.array().unaryExpr([](Real v) {
        return (v > std::numeric_limits<Real>::epsilon()) ? (Real(1) / v) : Real(0);
    });
    mat.array().colwise() *= invRowNorms.array();
}

template <typename Derived>
Eigen::Matrix<typename Derived::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
rowNormalize(const Eigen::MatrixBase<Derived>& X)
{
    using Scalar = typename Derived::Scalar;
    Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> out(X.rows(), X.cols());
    for (Eigen::Index i = 0; i < X.rows(); ++i) {
        Scalar s = X.row(i).sum();
        if (s != Scalar(0)) {
            out.row(i) = X.row(i) / s;
        } else {
            out.row(i).setZero();
        }
    }
    return out;
}


template <typename Scalar, int StorageOrder>
void colNormalizeInPlace(Eigen::SparseMatrix<Scalar, StorageOrder>& mat, bool nonNeg = true) {
    using Real = RealScalar<Scalar>;
    if constexpr (StorageOrder == Eigen::ColMajor) {
        for (int j = 0; j < mat.outerSize(); ++j) {
            Real colNormL1(0);
            if (nonNeg) {
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, j); it; ++it) {
                    colNormL1 += it.value();
                }
            } else {
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, j); it; ++it) {
                    colNormL1 += std::abs(it.value());
                }
            }
            if (colNormL1 > std::numeric_limits<Real>::epsilon()) {
                Real invColNormL1 = Real(1) / colNormL1;
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, j); it; ++it) {
                    it.valueRef() *= invColNormL1;
                }
            }
        }
    } else {
        Eigen::Matrix<Real, 1, Eigen::Dynamic> invColNormsL1 = Eigen::Matrix<Real, 1, Eigen::Dynamic>::Zero(mat.cols());
        if (nonNeg) {
            for (int i = 0; i < mat.outerSize(); ++i) {
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, i); it; ++it) {
                    invColNormsL1(it.col()) += it.value();
                }
            }
        } else {
            for (int i = 0; i < mat.outerSize(); ++i) {
                for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, i); it; ++it) {
                    invColNormsL1(it.col()) += std::abs(it.value());
                }
            }
        }
        for (int j = 0; j < mat.cols(); ++j) {
            if (invColNormsL1(j) > std::numeric_limits<Real>::epsilon()) {
                invColNormsL1(j) = Real(1) / invColNormsL1(j);
            } else {
                invColNormsL1(j) = Real(0);
            }
        }
        for (int i = 0; i < mat.outerSize(); ++i) {
            for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, i); it; ++it) {
                it.valueRef() *= invColNormsL1(it.col());
            }
        }
    }
}

template <typename Derived>
void colNormalizeInPlace(Eigen::MatrixBase<Derived>& mat, bool nonNeg = true) {
    using Real = RealScalar<typename Derived::Scalar>;
    using RealVector = Eigen::Matrix<Real, Derived::ColsAtCompileTime, 1>;
    RealVector colNorms;
    if (nonNeg) {
        colNorms = mat.colwise().sum();
    } else {
        colNorms = mat.colwise().template lpNorm<1>();
    }
    RealVector invColNorms = colNorms.array().unaryExpr([](Real v) {
        return (v > std::numeric_limits<Real>::epsilon()) ? (Real(1) / v) : Real(0);
    });
    mat.array().rowwise() *= invColNorms.transpose().array();
}

template <typename Scalar, int StorageOrder, typename UnaryOp>
void transformAndRowNormalize(Eigen::SparseMatrix<Scalar, StorageOrder>& mat,
                              UnaryOp&& op) {
    using Real = RealScalar<Scalar>;
    const Real zero(0);
    const Real eps = std::numeric_limits<Real>::min();

    if constexpr (StorageOrder == Eigen::RowMajor) {
        for (int i = 0; i < mat.outerSize(); ++i) {
            Real rowSum = zero;
            for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, i); it; ++it) {
                Real v = static_cast<Real>(it.value());
                Real transformed = static_cast<Real>(op(v));
                if (!std::isfinite(transformed) || transformed < zero) {
                    transformed = zero;
                }
                it.valueRef() = static_cast<Scalar>(transformed);
                rowSum += transformed;
            }

            // Treat tiny sums as zero to avoid 1 / tiny -> inf
            if (rowSum <= eps) {
                continue;
            }

            const Real invRowSum = Real(1) / rowSum;
            for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, i); it; ++it) {
                Real v = static_cast<Real>(it.value()) * invRowSum;
                if (!std::isfinite(v)) {
                    v = zero; // safety belt
                }
                it.valueRef() = static_cast<Scalar>(v);
            }
        }
    } else {
        Eigen::Matrix<Real, Eigen::Dynamic, 1> rowSum(mat.rows());
        rowSum.setZero();
        for (int j = 0; j < mat.outerSize(); ++j) {
            for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, j); it; ++it) {
                Real v = static_cast<Real>(it.value());
                Real transformed = static_cast<Real>(op(v));
                if (!std::isfinite(transformed) || transformed < zero) {
                    transformed = zero;
                }
                it.valueRef() = static_cast<Scalar>(transformed);
                rowSum(it.row()) += transformed;
            }
        }
        Eigen::Matrix<Real, Eigen::Dynamic, 1> invRowSum(mat.rows());
        invRowSum.setZero();
        for (int i = 0; i < mat.rows(); ++i) {
            if (rowSum(i) > eps) {
                invRowSum(i) = Real(1) / rowSum(i);
            }
        }
        for (int j = 0; j < mat.outerSize(); ++j) {
            for (typename Eigen::SparseMatrix<Scalar, StorageOrder>::InnerIterator it(mat, j); it; ++it) {
                Real v = static_cast<Real>(it.value()) * invRowSum(it.row());
                if (!std::isfinite(v)) {
                    v = zero;
                }
                it.valueRef() = static_cast<Scalar>(v);
            }
        }
    }
}

// expit + row-normalize wrapper
template <typename Scalar, int StorageOrder>
void expitAndRowNormalize(Eigen::SparseMatrix<Scalar, StorageOrder>& mat) {
    transformAndRowNormalize(mat, [](auto x) {
        using Real = RealScalar<decltype(x)>;
        return expit(static_cast<Real>(x));
    });
}

template <class Derived>
Eigen::Matrix<typename Eigen::NumTraits<typename Derived::Scalar>::Real,
    Eigen::Dynamic, 1> columnMedians(const Eigen::DenseBase<Derived>& X)
{
    using Scalar     = typename Derived::Scalar;
    using RealScalar = typename Eigen::NumTraits<Scalar>::Real;
    using Index      = typename Eigen::Index;
    // Reject complex types at compile time
    static_assert(!Eigen::NumTraits<Scalar>::IsComplex,
                  "columnMedians: complex types are not supported.");
    const Index nRows = X.rows();
    const Index nCols = X.cols();
    Eigen::Matrix<RealScalar, Eigen::Dynamic, 1> medians(nCols);
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> col(nRows);
    for (Index j = 0; j < nCols; ++j) {
        col = X.col(j);
        std::sort(col.data(), col.data() + nRows);
        if (nRows % 2 == 1) {
            medians(j) = static_cast<RealScalar>(col(nRows / 2));
        } else {
            const Index k = nRows / 2;
            medians(j) = RealScalar(0.5) * (static_cast<RealScalar>(col(k - 1)) + static_cast<RealScalar>(col(k)));
        }
    }
    return medians;
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

template<typename T>
inline T clamp(T x, T a, T b) {
    return x < a ? a : (x > b ? b : x);
}

double safe_log10(double x);
double normal_sf(double x);
double twosided_p_from_z(double z);
double normal_logsf(double x);
double log_twosided_p_from_z(double z);
double normal_log10sf(double x);
double log10_twosided_p_from_z(double z);
double chisq1_log10p(double chi2);
std::pair<double, double> chisq2x2_log10p(double a, double b, double c, double d, double pseudocount = 1e-8);
double cauchy_combination(const std::vector<double>& pval,
                          const std::vector<double>& weights_in = {},
                          double small_p_approx_thresh = 1e-8);
int32_t loess_quadratic_tricube(const std::vector<double>& x,
                                const std::vector<double>& y,
                                std::vector<double>& yhat, double span = 0.3);

template<typename T>
long double factorial(T n) {
    // Handle invalid input for which factorial is undefined.
    if (n < 0) {
        throw std::domain_error("Factorial is not defined for negative numbers.");
    }

    // --- Lookup Table (n <= 20) ---
    constexpr int PRECOMPUTED_LIMIT = 21;
    static const std::array<unsigned long long, PRECOMPUTED_LIMIT> small_factorials = {
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800,
        39916800, 479001600, 6227020800, 87178291200, 1307674368000,
        20922789888000, 355687428096000, 6402373705728000,
        121645100408832000, 2432902008176640000
    };

    if (n < PRECOMPUTED_LIMIT) {
        return static_cast<long double>(small_factorials[(uint32_t) n]);
    }

    // --- Stirling-Ramanujan Approximation (n > 20) ---
    // ln(n!) ≈ n*ln(n) - n + 0.5*ln(2*π*n) + 1/(12n)
    long double n_ld = static_cast<long double>(n);
    long double log_factorial = n_ld * std::log(n_ld) - n_ld +
                                0.5L * std::log(2 * M_PI * n_ld) +
                                1.0L / (12.0L * n_ld);

    return std::exp(log_factorial);
}

template<typename T>
long double log_factorial(T n) {
    // Handle invalid input for which factorial is undefined.
    if (n < 0) {
        throw std::domain_error("Factorial is not defined for negative numbers.");
    }

    // --- Lookup Table (n <= 20) ---
    constexpr int PRECOMPUTED_LIMIT = 21;
    static const std::array<unsigned long long, PRECOMPUTED_LIMIT> small_factorials = {
        1, 1, 2, 6, 24, 120, 720, 5040, 40320, 362880, 3628800,
        39916800, 479001600, 6227020800, 87178291200, 1307674368000,
        20922789888000, 355687428096000, 6402373705728000,
        121645100408832000, 2432902008176640000
    };

    if (n < PRECOMPUTED_LIMIT) {
        return std::log(static_cast<long double>(small_factorials[(uint32_t) n]));
    }

    // --- Stirling-Ramanujan Approximation (n > 20) ---
    // ln(n!) ≈ n*ln(n) - n + 0.5*ln(2*π*n) + 1/(12n)
    long double n_ld = static_cast<long double>(n);
    return n_ld * std::log(n_ld) - n_ld +
           0.5L * std::log(2 * M_PI * n_ld) +
           1.0L / (12.0L * n_ld);
}
