#pragma once

// Low-rank-plus-diagonal covariance support for downstream clustering.

#include "numerical_utils.hpp"

struct LowRankDiagonalCovariance {
    Eigen::VectorXd diagonal;
    RowMajorMatrixXd factor;

    Eigen::MatrixXd dense() const;
    Eigen::VectorXd apply_vector(const Eigen::Ref<const Eigen::VectorXd>& value) const;
    Eigen::MatrixXd apply_matrix(const Eigen::Ref<const Eigen::MatrixXd>& value) const;
};

class LowRankDiagonalSolver {
public:
    LowRankDiagonalSolver() = default;
    LowRankDiagonalSolver(const Eigen::Ref<const Eigen::VectorXd>& diagonal,
        const Eigen::Ref<const RowMajorMatrixXd>& factor);
    void compute(const Eigen::Ref<const Eigen::VectorXd>& diagonal,
        const Eigen::Ref<const RowMajorMatrixXd>& factor);

    Eigen::VectorXd solve_vector(const Eigen::Ref<const Eigen::VectorXd>& value) const;
    Eigen::MatrixXd solve_matrix(const Eigen::Ref<const Eigen::MatrixXd>& value) const;
    double quadratic(const Eigen::Ref<const Eigen::VectorXd>& value) const;
    double log_determinant() const { return log_determinant_; }
    const Eigen::VectorXd& inverse_diagonal() const { return inverse_diagonal_; }
    const Eigen::MatrixXd& inverse_diagonal_factor() const {
        return inverse_diagonal_factor_;
    }
    Eigen::MatrixXd solve_core(
        const Eigen::Ref<const Eigen::MatrixXd>& value) const;

private:
    Eigen::VectorXd inverse_diagonal_;
    Eigen::MatrixXd inverse_diagonal_factor_;
    Eigen::LLT<Eigen::MatrixXd> core_llt_;
    double log_determinant_ = 0.0;
};
