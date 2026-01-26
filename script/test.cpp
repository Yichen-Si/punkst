#include "punkst.h"
#include "utils.h"
#include "numerical_utils.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

int test(int argc, char** argv) {
    std::string inFile, confusionFile, outFile;

    ParamList pl;
    pl.add_option("input", "Input data file", inFile, true)
      .add_option("confusion", "Confusion matrix file", confusionFile, true)
      .add_option("output", "Output data file", outFile, true);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }
    std::vector<std::string> rows, cols, rows_c, cols_c;
    RowMajorMatrixXd mat;
    read_matrix_from_file(inFile, mat, &rows, &cols);
    RowMajorMatrixXd C;
    read_matrix_from_file(confusionFile, C, &rows_c, &cols_c);
    int32_t K = mat.cols();
    int32_t M = mat.rows();
    Eigen::MatrixXd B = Eigen::MatrixXd(mat);
    Eigen::VectorXd colsums = B.colwise().sum();
    colNormalizeInPlace(B);
    Eigen::VectorXd w = colsums.array().sqrt();
    w = w.array() / w.sum() * K;
    // Eigen::VectorXd w = colsums.array() / colsums.sum() * K;
    rowNormalizeInPlace(C);
    NonnegRidgeResult denoise = solve_nonneg_weighted_ridge(C, B, w);
    Eigen::VectorXd delta = Eigen::VectorXd::Zero(K);
    for (int32_t k = 0; k < K; ++k) {
        delta(k) = (denoise.A.col(k).array() - B.col(k).array()).abs().sum();
        denoise.A.col(k) *= colsums(k);
        printf("%d: %.6f\t%.0f\n", k, delta(k), colsums(k));
    }
    printf("Total change in fraction: %.6f\n", delta.sum());
    delta = delta.array() * colsums.array();
    printf("Total absolute change: %.6f (%.6f)\n", delta.sum(), delta.sum() / colsums.sum());

    write_matrix_to_file(outFile, denoise.A, 4, true, rows, "Feature", &cols);

    return 0;
}
