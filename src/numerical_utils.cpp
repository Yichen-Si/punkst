#include "numerical_utils.hpp"

// Calculate the mean absolute difference between two arrays.
double mean_change(const std::vector<double>& arr1, const std::vector<double>& arr2) {
    if (arr1.size() != arr2.size()) {
        return std::numeric_limits<double>::max();
    }
    double total = 0.0;
    size_t size = arr1.size();
    for (size_t i = 0; i < size; i++) {
        total += std::fabs(arr1[i] - arr2[i]);
    }
    return total / size;
}

double mean_max_row_change(const MatrixXd& arr1, const MatrixXd& arr2) {
    if (arr1.rows() != arr2.rows() || arr1.cols() != arr2.cols()) {
        return std::numeric_limits<double>::max();
    }
    double total = 0.0;
    size_t rows = arr1.rows();
    for (size_t i = 0; i < rows; i++) {
        total += (arr1.row(i) - arr2.row(i)).cwiseAbs().maxCoeff();
    }
    return total / rows;
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

// exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
// = exp(psi(\alpha_k) - psi(\alpha_0))
void dirichlet_expectation_1d(std::vector<double>& alpha, std::vector<double>& out, double offset) {
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

// Vector version
VectorXd dirichlet_expectation_1d(VectorXd& alpha, double offset) {
    size_t size = alpha.size();
    VectorXd out(size);
    alpha.array() += offset;
    double total = alpha.sum();
    double psi_total = psi(total);
    for (size_t i = 0; i < size; i++) {
        out[i] = std::exp(psi(alpha[i]) - psi_total);
    }
    return out;
}

// row-wise exp(E[log X]) for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
MatrixXd dirichlet_expectation_2d(const MatrixXd& alpha) {
    MatrixXd result(alpha.rows(), alpha.cols());
    for (int i = 0; i < alpha.rows(); i++) {
        double total = alpha.row(i).sum();
        double psi_total = psi(total);
        for (int j = 0; j < alpha.cols(); j++) {
            result(i, j) = std::exp(psi(alpha(i, j)) - psi_total);
        }
    }
    return result;
}

// row-wise E[log X] for X ~ Dir(\alpha), \alpha_0 := \sum_k \alpha_k
MatrixXd dirichlet_entropy_2d(const MatrixXd& alpha) {
    MatrixXd result(alpha.rows(), alpha.cols());
    for (int i = 0; i < alpha.rows(); i++) {
        double total = alpha.row(i).sum();
        double psi_total = psi(total);
        for (int j = 0; j < alpha.cols(); j++) {
            result(i, j) = psi(alpha(i, j)) - psi_total;
        }
    }
    return result;
}

// If b is provided, it must be of the same size as a.
std::pair<double, double> logsumexp(const VectorXd &a, const VectorXd* b) {
    if (a.size() == 0)
        return std::make_pair(-std::numeric_limits<double>::infinity(), 0.0);

    // Find the maximum value for numerical stability.
    double maxVal = a.maxCoeff();
    // If all entries are -infinity, then return -infinity.
    if (!std::isfinite(maxVal)) {
        return std::make_pair(-std::numeric_limits<double>::infinity(), 0.0);
    }

    double sumExp = 0.0;
    double sign = 0.0;
    if (b == nullptr) {
        // Standard logsumexp
        for (int i = 0; i < a.size(); ++i) {
            sumExp += std::exp(a(i) - maxVal);
        }
        sign = 1.0;
    } else {
        // Weighted version: log(sum(b * exp(a))).
        for (int i = 0; i < a.size(); ++i) {
            double term = (*b)(i) * std::exp(a(i) - maxVal);
            sumExp += term;
        }
        sign = (sumExp > 0 ? 1.0 : (sumExp < 0 ? -1.0 : 0.0));
        sumExp = std::abs(sumExp);
    }
    double lse = maxVal + std::log(sumExp);
    return std::make_pair(lse, sign);
}


// find largest values and indices
void findTopK(MatrixXd& topVals, Eigen::MatrixXi& topIds, const MatrixXd& mtx, int32_t k) {
    int32_t nRows = mtx.rows();
    int32_t nCols = mtx.cols();
    if (k > nCols) {
        k = nCols;
    }
    topVals.resize(nRows, k);
    topIds.resize(nRows, k);
    for (int32_t i = 0; i < nRows; ++i) {
        // Create a vector of (value, index) pairs for the current row.
        std::vector<std::pair<double, int>> rowData;
        rowData.reserve(nCols);
        for (int j = 0; j < nCols; ++j) {
            rowData.emplace_back(mtx(i, j), j);
        }
        // Partial sort to get the k largest values in descending order.
        std::partial_sort(rowData.begin(), rowData.begin() + k, rowData.end(),
            [](const std::pair<double, int>& a, const std::pair<double, int>& b) {
                return a.first > b.first;
            }
        );
        // Store the top k values and their column indices.
        for (int j = 0; j < k; ++j) {
            topVals(i, j) = rowData[j].first;
            topIds(i, j) = rowData[j].second;
        }
    }
}
