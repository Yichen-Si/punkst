#pragma once

#include <vector>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <cassert>
#include <algorithm>
#include <numeric>
#include <tbb/tbb.h>
#include <tbb/blocked_range.h>

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

template<typename T>
inline T clamp(T x, T a, T b) {
    return x < a ? a : (x > b ? b : x);
}

inline double safe_log10(double x) {
    return (x > 0.0) ? std::log10(x) : -std::numeric_limits<double>::infinity();
}

inline double normal_sf(double x) {
    return 0.5 * std::erfc(x / std::sqrt(2.0));
}

inline double twosided_p_from_z(double z) {
    return 2.0 * normal_sf(std::fabs(z));
}

inline double normal_logsf(double x) {
    const double threshold = 30.0;
    if (x > threshold) {
        // For large x, log(P(Z>x)) ≈ -0.5*x² - log(x) - 0.5*log(2π)
        // (asymptotic expansion)
        const double LOG_2_PI = 1.83787706640934548356;
        return -0.5 * x * x - std::log(x) - 0.5 * LOG_2_PI;
    } else {
        double erfc_val = std::erfc(x / std::sqrt(2.0));
        // In case erfc underflows before our threshold is met, log(0) is -inf.
        if (erfc_val <= 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        const double LOG_0_5 = -0.69314718055994530941;
        return LOG_0_5 + std::log(erfc_val);
    }
}

inline double log_twosided_p_from_z(double z) {
    const double LOG_2 = 0.69314718055994530941; // log(2.0)
    return LOG_2 + normal_logsf(std::fabs(z));
}

inline double normal_log10sf(double x) {
    const double LOG_10 = 2.30258509299404568402; // log(10.0)
    return normal_logsf(x) / LOG_10;
}

inline double log10_twosided_p_from_z(double z) {
    const double LOG10_2 = 0.3010299956639812; // log10(2.0)
    return LOG10_2 + normal_log10sf(std::fabs(z));
}

// Weighted quadratic local regression at each xi,
//   tricube weights over k-NN window.
// Input: x,y of length n. span \in (0,1].
// Outpu: fitted yhat at each x.
inline int32_t loess_quadratic_tricube(const std::vector<double>& x,
                                       const std::vector<double>& y,
                        std::vector<double>& yhat, double span = 0.3)
{
    const int n = (int)x.size();
    if (n != (int)y.size()) {
        throw std::invalid_argument("loess_quadratic_tricube: invalid input");
    }
    if (n < 3) {
        yhat = y;
        return 0;
    }
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(),
                     [&](int a, int b){ return x[a] < x[b]; });

    std::vector<double> xs(n), ys(n);
    for (int r = 0; r < n; ++r) { xs[r] = x[ord[r]]; ys[r] = y[ord[r]]; }

    const int k = std::max(3, (int)std::ceil(std::max(0.0, std::min(1.0, span)) * n));
    std::vector<double> yhat_sorted(n, 0.0);

    tbb::parallel_for(tbb::blocked_range<int>(0, n, 1024),
                      [&](const tbb::blocked_range<int>& range){
        for (int i = range.begin(); i < range.end(); ++i) {
            int L = std::max(0, i - k/2);
            int R = std::min(n - 1, L + k - 1);
            L = std::max(0, R - k + 1);
            const double xi = xs[i];
            const double dmax = std::max(xs[i] - xs[L], xs[R] - xs[i]);

            if (dmax <= 0.0) {
                double S0=0, T0=0;
                for (int j = L; j <= R; ++j) { S0 += 1.0; T0 += ys[j]; }
                yhat_sorted[i] = (S0 > 0.0) ? (T0 / S0) : 0.0;
                continue;
            }

            // normal eqs for quadratic
            double S0=0, S1=0, S2=0, S3=0, S4=0;
            double T0=0, T1=0, T2=0;
            for (int j = L; j <= R; ++j) {
                double u = std::abs(xs[j] - xi) / dmax;           // [0,1]
                double w = std::pow(1.0 - std::pow(u,3.0), 3.0);  // tricube
                double xj = xs[j], yj = ys[j], x2 = xj*xj;
                S0 += w;     S1 += w*xj;    S2 += w*x2;
                S3 += w*x2*xj; S4 += w*x2*x2;
                T0 += w*yj;  T1 += w*xj*yj; T2 += w*x2*yj;
            }

            // Solve 3x3 via adjugate
            double A00=S0, A01=S1, A02=S2,
                   A10=S1, A11=S2, A12=S3,
                   A20=S2, A21=S3, A22=S4;
            double c00 =  A11*A22 - A12*A21;
            double c01 = -(A10*A22 - A12*A20);
            double c02 =  A10*A21 - A11*A20;
            double c10 = -(A01*A22 - A02*A21);
            double c11 =  A00*A22 - A02*A20;
            double c12 = -(A00*A21 - A01*A20);
            double c20 =  A01*A12 - A02*A11;
            double c21 = -(A00*A12 - A02*A10);
            double c22 =  A00*A11 - A01*A10;
            double det = A00*c00 + A01*c01 + A02*c02;

            double yi;
            if (std::abs(det) <= 1e-20 || !std::isfinite(det)) {
                // fallback to local linear
                double W=0, Wx=0, Wy=0, Wxx=0, Wxy=0;
                for (int j = L; j <= R; ++j) {
                    double u = std::abs(xs[j] - xi) / dmax;
                    double w = std::pow(1.0 - std::pow(u,3.0), 3.0);
                    double xj = xs[j], yj = ys[j];
                    W += w; Wx += w*xj; Wy += w*yj; Wxx += w*xj*xj; Wxy += w*xj*yj;
                }
                double det2 = W*Wxx - Wx*Wx;
                yi = (std::abs(det2) <= 1e-20 || !std::isfinite(det2))
                     ? ((W>0)?(Wy/W):0.0)
                     : ((Wxx*Wy - Wx*Wxy)/det2 + (W*Wxy - Wx*Wy)/det2 * xi);
            } else {
                double a0 = (c00*T0 + c01*T1 + c02*T2) / det;
                double a1 = (c10*T0 + c11*T1 + c12*T2) / det;
                double a2 = (c20*T0 + c21*T1 + c22*T2) / det;
                yi = a0 + a1*xi + a2*xi*xi;
            }
            yhat_sorted[i] = yi;
        }
    });

    // unsort
    yhat.resize(n);
    for (int r = 0; r < n; ++r) yhat[ord[r]] = yhat_sorted[r];
    return 1;
}
