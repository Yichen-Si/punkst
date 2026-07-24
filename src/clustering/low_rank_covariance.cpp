#include "clustering/low_rank_covariance.hpp"

#include <cmath>
#include <stdexcept>

namespace {

void validate_covariance(const Eigen::Ref<const Eigen::VectorXd>& diagonal,
    const Eigen::Ref<const RowMajorMatrixXd>& factor) {
    if (diagonal.size() == 0 || factor.rows() != diagonal.size()
        || !diagonal.allFinite() || !factor.allFinite()
        || (diagonal.array() <= 0.0).any()) {
        throw std::invalid_argument("Invalid low-rank diagonal covariance");
    }
}

} // namespace

Eigen::MatrixXd LowRankDiagonalCovariance::dense() const {
    validate_covariance(diagonal, factor);
    Eigen::MatrixXd out = diagonal.asDiagonal();
    if (factor.cols() > 0) out.noalias() += factor * factor.transpose();
    return out;
}

Eigen::VectorXd LowRankDiagonalCovariance::apply_vector(
    const Eigen::Ref<const Eigen::VectorXd>& value) const {
    validate_covariance(diagonal, factor);
    if (value.size() != diagonal.size()) {
        throw std::invalid_argument("Low-rank covariance vector dimension mismatch");
    }
    Eigen::VectorXd out = diagonal.array() * value.array();
    if (factor.cols() > 0) out.noalias() += factor * (factor.transpose() * value);
    return out;
}

Eigen::MatrixXd LowRankDiagonalCovariance::apply_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& value) const {
    validate_covariance(diagonal, factor);
    if (value.rows() != diagonal.size()) {
        throw std::invalid_argument("Low-rank covariance matrix dimension mismatch");
    }
    Eigen::MatrixXd out = (value.array().colwise() * diagonal.array()).matrix();
    if (factor.cols() > 0) out.noalias() += factor * (factor.transpose() * value);
    return out;
}

LowRankDiagonalSolver::LowRankDiagonalSolver(
    const Eigen::Ref<const Eigen::VectorXd>& diagonal,
    const Eigen::Ref<const RowMajorMatrixXd>& factor) {
    compute(diagonal, factor);
}

void LowRankDiagonalSolver::compute(
    const Eigen::Ref<const Eigen::VectorXd>& diagonal,
    const Eigen::Ref<const RowMajorMatrixXd>& factor) {
    validate_covariance(diagonal, factor);
    inverse_diagonal_ = diagonal.cwiseInverse();
    inverse_diagonal_factor_ =
        (factor.array().colwise() * inverse_diagonal_.array()).matrix();
    Eigen::MatrixXd core = Eigen::MatrixXd::Identity(factor.cols(), factor.cols());
    if (factor.cols() > 0) {
        core.noalias() += factor.transpose() * inverse_diagonal_factor_;
        core_llt_.compute(core);
        if (core_llt_.info() != Eigen::Success) {
            throw std::runtime_error("Low-rank covariance Woodbury factorization failed");
        }
    }
    log_determinant_ = diagonal.array().log().sum();
    if (factor.cols() > 0) {
        log_determinant_ += 2.0
            * core_llt_.matrixL().toDenseMatrix().diagonal().array().log().sum();
    }
}

Eigen::MatrixXd LowRankDiagonalSolver::solve_core(
    const Eigen::Ref<const Eigen::MatrixXd>& value) const {
    if (value.rows() != inverse_diagonal_factor_.cols()) {
        throw std::invalid_argument("Low-rank covariance core solve dimension mismatch");
    }
    if (inverse_diagonal_factor_.cols() == 0) {
        return Eigen::MatrixXd(0, value.cols());
    }
    return core_llt_.solve(value);
}

Eigen::VectorXd LowRankDiagonalSolver::solve_vector(
    const Eigen::Ref<const Eigen::VectorXd>& value) const {
    if (value.size() != inverse_diagonal_.size()) {
        throw std::invalid_argument("Low-rank covariance solve dimension mismatch");
    }
    Eigen::VectorXd out = inverse_diagonal_.array() * value.array();
    if (inverse_diagonal_factor_.cols() > 0) {
        out.noalias() -= inverse_diagonal_factor_ * core_llt_.solve(
            inverse_diagonal_factor_.transpose() * value);
    }
    return out;
}

Eigen::MatrixXd LowRankDiagonalSolver::solve_matrix(
    const Eigen::Ref<const Eigen::MatrixXd>& value) const {
    if (value.rows() != inverse_diagonal_.size()) {
        throw std::invalid_argument("Low-rank covariance matrix solve dimension mismatch");
    }
    Eigen::MatrixXd out = (value.array().colwise() * inverse_diagonal_.array()).matrix();
    if (inverse_diagonal_factor_.cols() > 0) {
        out.noalias() -= inverse_diagonal_factor_ * core_llt_.solve(
            inverse_diagonal_factor_.transpose() * value);
    }
    return out;
}

double LowRankDiagonalSolver::quadratic(
    const Eigen::Ref<const Eigen::VectorXd>& value) const {
    return value.dot(solve_vector(value));
}

Eigen::VectorXd LowRankDiagonalSolver::quadratic_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& values) const {
    if (values.cols() != inverse_diagonal_.size()) {
        throw std::invalid_argument(
            "Low-rank covariance row matrix dimension mismatch");
    }
    Eigen::VectorXd out = (
        values.array().square().rowwise()
            * inverse_diagonal_.transpose().array()).rowwise().sum();
    if (inverse_diagonal_factor_.cols() > 0) {
        const Eigen::MatrixXd projected =
            values * inverse_diagonal_factor_;
        const Eigen::MatrixXd solved =
            core_llt_.solve(projected.transpose()).transpose();
        out.array() -=
            (projected.array() * solved.array()).rowwise().sum();
    }
    return out;
}
