#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#include "punkst.h"
#include "utils.h"
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

#include <Eigen/Dense>
#include <Eigen/Sparse>

using Clock = std::chrono::high_resolution_clock;

template <typename F>
double time_ms(F&& f, int iters = 5) {
    double best = 1e100;
    for (int i = 0; i < iters; ++i) {
        auto t0 = Clock::now();
        f();
        auto t1 = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        if (ms < best) best = ms;
    }
    return best;
}

template <typename SparseMat>
SparseMat make_random_sparse(int rows, int cols, double density, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist_val(-1.f, 1.f);
    std::uniform_real_distribution<double> coin(0.0, 1.0);

    std::vector<Eigen::Triplet<float>> trips;
    trips.reserve(static_cast<size_t>(rows * cols * density) + 1);

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (coin(gen) < density) {
                trips.emplace_back(i, j, dist_val(gen));
            }
        }
    }

    SparseMat A(rows, cols);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();
    return A;
}

template <typename DenseMat>
DenseMat make_random_dense(int rows, int cols, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    DenseMat M(rows, cols);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            M(i, j) = dist(gen);
    return M;
}



int32_t test(int32_t argc, char** argv) {

    int32_t N, K, M;
    int32_t seed, debug_ = 0;
    int32_t threads = 1;
    double densityA = 0.002, densityB = 0.002;
    bool SS, SD, DS, DD;

    ParamList pl;
    // Input / sim options
    pl.add_option("N", "Number of rows (N)", N, true)
        .add_option("K", "Number of cols (K)", K, true)
        .add_option("M", "Number of features (M)", M, true)
        .add_option("seed", "Random seed", seed, true)
        .add_option("threads", "Number of threads", threads);
    pl.add_option("densityA", "Density of matrix A", densityA)
        .add_option("densityB", "Density of matrix B", densityB);
    pl.add_option("debug", "Debug", debug_)
        .add_option("SS", "Test Sparse x Sparse", SS)
        .add_option("SD", "Test Sparse x Dense", SD)
        .add_option("DS", "Test Dense x Sparse", DS)
        .add_option("DD", "Test Dense x Dense", DD);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    std::cout << "Sizes: (" << M << "x" << K << ") * (" << K << "x" << N << ")\n";
    std::cout << "Sparse density A=" << densityA << ", B=" << densityB << "\n\n";

    using SpCol = Eigen::SparseMatrix<float, Eigen::ColMajor>;
    using SpRow = Eigen::SparseMatrix<float, Eigen::RowMajor>;
    using MatCol = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;
    using MatRow = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

    // operands
    SpCol A_sc = make_random_sparse<SpCol>(M, K, densityA, 1); // MxK
    SpRow A_sr = make_random_sparse<SpRow>(M, K, densityA, 2); // MxK
    SpCol B_sc = make_random_sparse<SpCol>(K, N, densityB, 3); // KxN
    SpRow B_sr = make_random_sparse<SpRow>(K, N, densityB, 4); // KxN

    MatCol D_dc = make_random_dense<MatCol>(M, K, 5); // MxK
    MatRow D_dr = make_random_dense<MatRow>(M, K, 6); // MxK
    MatCol E_dc = make_random_dense<MatCol>(K, N, 7); // KxN
    MatRow E_dr = make_random_dense<MatRow>(K, N, 8); // KxN

    // AB/AE/DB (M x K) * (K x N)

    // warm-up
    {
        Eigen::MatrixXf tmp = A_sc * E_dc;
        (void)tmp;
    }


if (SS) {    // =========================================================
    // 1. SPARSE x SPARSE -> SPARSE (8 cases)
    // naming: t_spsp_<lhs><rhs><res>, where c=col, r=row
    // =========================================================
    double t_spsp_rrr = time_ms([&]() {
        Eigen::SparseMatrix<float, Eigen::RowMajor> C = A_sr * B_sr;
        volatile int nnz = C.nonZeros(); (void)nnz;
    });
    double t_spsp_rrc = time_ms([&]() {
        Eigen::SparseMatrix<float> C = A_sr * B_sr;
        volatile int nnz = C.nonZeros(); (void)nnz;
    });
    double t_spsp_rcr = time_ms([&]() {
        Eigen::SparseMatrix<float, Eigen::RowMajor> C = A_sr * B_sc;
        volatile int nnz = C.nonZeros(); (void)nnz;
    });
    double t_spsp_rcc = time_ms([&]() {
        Eigen::SparseMatrix<float> C = A_sr * B_sc;
        volatile int nnz = C.nonZeros(); (void)nnz;
    });
    double t_spsp_crr = time_ms([&]() {
        Eigen::SparseMatrix<float, Eigen::RowMajor> C = A_sc * B_sr;
        volatile int nnz = C.nonZeros(); (void)nnz;
    });
    double t_spsp_crc = time_ms([&]() {
        Eigen::SparseMatrix<float> C = A_sc * B_sr;
        volatile int nnz = C.nonZeros(); (void)nnz;
    });
    double t_spsp_ccr = time_ms([&]() {
        Eigen::SparseMatrix<float, Eigen::RowMajor> C = A_sc * B_sc;
        volatile int nnz = C.nonZeros(); (void)nnz;
    });
    double t_spsp_ccc = time_ms([&]() {
        Eigen::SparseMatrix<float> C = A_sc * B_sc;
        volatile int nnz = C.nonZeros(); (void)nnz;
    });

    std::cout << "=== sparse x sparse (A*B -> S) ===\n";
    std::cout << "rrr: " << t_spsp_rrr << " ms\n";
    std::cout << "rrc: " << t_spsp_rrc << " ms\n";
    std::cout << "rcr: " << t_spsp_rcr << " ms\n";
    std::cout << "rcc: " << t_spsp_rcc << " ms\n";
    std::cout << "crr: " << t_spsp_crr << " ms\n";
    std::cout << "crc: " << t_spsp_crc << " ms\n";
    std::cout << "ccr: " << t_spsp_ccr << " ms\n";
    std::cout << "ccc: " << t_spsp_ccc << " ms\n\n";}

if (SD) {    // =========================================================
    // 2. SPARSE x DENSE -> DENSE (8 cases)
    // lhs sparse (r/c), rhs dense (r/c), result dense (r/c)
    // =========================================================
    double t_spd_rrr = time_ms([&]() {
        MatRow C = A_sr * E_dr;
        volatile float x = C(0,0); (void)x;
    });
    double t_spd_rrc = time_ms([&]() {
        MatCol C = A_sr * E_dr;
        volatile float x = C(0,0); (void)x;
    });
    double t_spd_rcr = time_ms([&]() {
        MatRow C = A_sr * E_dc;
        volatile float x = C(0,0); (void)x;
    });
    double t_spd_rcc = time_ms([&]() {
        MatCol C = A_sr * E_dc;
        volatile float x = C(0,0); (void)x;
    });
    double t_spd_crr = time_ms([&]() {
        MatRow C = A_sc * E_dr;
        volatile float x = C(0,0); (void)x;
    });
    double t_spd_crc = time_ms([&]() {
        MatCol C = A_sc * E_dr;
        volatile float x = C(0,0); (void)x;
    });
    double t_spd_ccr = time_ms([&]() {
        MatRow C = A_sc * E_dc;
        volatile float x = C(0,0); (void)x;
    });
    double t_spd_ccc = time_ms([&]() {
        MatCol C = A_sc * E_dc;
        volatile float x = C(0,0); (void)x;
    });

    std::cout << "=== sparse x dense (A*E -> D) ===\n";
    std::cout << "rrr: " << t_spd_rrr << " ms\n";
    std::cout << "rrc: " << t_spd_rrc << " ms\n";
    std::cout << "rcr: " << t_spd_rcr << " ms\n";
    std::cout << "rcc: " << t_spd_rcc << " ms\n";
    std::cout << "crr: " << t_spd_crr << " ms\n";
    std::cout << "crc: " << t_spd_crc << " ms\n";
    std::cout << "ccr: " << t_spd_ccr << " ms\n";
    std::cout << "ccc: " << t_spd_ccc << " ms\n\n";}

if (DS) {    // =========================================================
    // 3. DENSE x SPARSE -> DENSE (8 cases)
    // lhs dense (r/c), rhs sparse (r/c), result dense (r/c)
    // note: right sparse is KxN → must use B_sc/B_sr
    // =========================================================
    double t_dsp_rrr = time_ms([&]() {
        MatRow C = D_dr * B_sr;
        volatile float x = C(0,0); (void)x;
    });
    double t_dsp_rrc = time_ms([&]() {
        MatCol C = D_dr * B_sr;
        volatile float x = C(0,0); (void)x;
    });
    double t_dsp_rcr = time_ms([&]() {
        MatRow C = D_dr * B_sc;
        volatile float x = C(0,0); (void)x;
    });
    double t_dsp_rcc = time_ms([&]() {
        MatCol C = D_dr * B_sc;
        volatile float x = C(0,0); (void)x;
    });
    double t_dsp_crr = time_ms([&]() {
        MatRow C = D_dc * B_sr;
        volatile float x = C(0,0); (void)x;
    });
    double t_dsp_crc = time_ms([&]() {
        MatCol C = D_dc * B_sr;
        volatile float x = C(0,0); (void)x;
    });
    double t_dsp_ccr = time_ms([&]() {
        MatRow C = D_dc * B_sc;
        volatile float x = C(0,0); (void)x;
    });
    double t_dsp_ccc = time_ms([&]() {
        MatCol C = D_dc * B_sc;
        volatile float x = C(0,0); (void)x;
    });

    std::cout << "=== dense x sparse (D*B -> D) ===\n";
    std::cout << "rrr: " << t_dsp_rrr << " ms\n";
    std::cout << "rrc: " << t_dsp_rrc << " ms\n";
    std::cout << "rcr: " << t_dsp_rcr << " ms\n";
    std::cout << "rcc: " << t_dsp_rcc << " ms\n";
    std::cout << "crr: " << t_dsp_crr << " ms\n";
    std::cout << "crc: " << t_dsp_crc << " ms\n";
    std::cout << "ccr: " << t_dsp_ccr << " ms\n";
    std::cout << "ccc: " << t_dsp_ccc << " ms\n\n";}


if (DD) {    // =========================================================
    // 4. DENSE x DENSE -> DENSE (8 cases)
    // just for completeness
    // =========================================================
    double t_dd_rrr = time_ms([&]() {
        MatRow C = D_dr * E_dr;
        volatile float x = C(0,0); (void)x;
    });
    double t_dd_rrc = time_ms([&]() {
        MatCol C = D_dr * E_dr;
        volatile float x = C(0,0); (void)x;
    });
    double t_dd_rcr = time_ms([&]() {
        MatRow C = D_dr * E_dc;
        volatile float x = C(0,0); (void)x;
    });
    double t_dd_rcc = time_ms([&]() {
        MatCol C = D_dr * E_dc;
        volatile float x = C(0,0); (void)x;
    });
    double t_dd_crr = time_ms([&]() {
        MatRow C = D_dc * E_dr;
        volatile float x = C(0,0); (void)x;
    });
    double t_dd_crc = time_ms([&]() {
        MatCol C = D_dc * E_dr;
        volatile float x = C(0,0); (void)x;
    });
    double t_dd_ccr = time_ms([&]() {
        MatRow C = D_dc * E_dc;
        volatile float x = C(0,0); (void)x;
    });
    double t_dd_ccc = time_ms([&]() {
        MatCol C = D_dc * E_dc;
        volatile float x = C(0,0); (void)x;
    });

    std::cout << "=== dense x dense (D*E -> D) ===\n";
    std::cout << "rrr: " << t_dd_rrr << " ms\n";
    std::cout << "rrc: " << t_dd_rrc << " ms\n";
    std::cout << "rcr: " << t_dd_rcr << " ms\n";
    std::cout << "rcc: " << t_dd_rcc << " ms\n";
    std::cout << "crr: " << t_dd_crr << " ms\n";
    std::cout << "crc: " << t_dd_crc << " ms\n";
    std::cout << "ccr: " << t_dd_ccr << " ms\n";
    std::cout << "ccc: " << t_dd_ccc << " ms\n";}

    return 0;
}
