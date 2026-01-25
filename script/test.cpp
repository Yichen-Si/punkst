#include "poisreg.hpp"
#include "mixpois.hpp"
#include "punkst.h"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>

static RowMajorMatrixXd sample_dirichlet(int N, int K, double alpha, std::mt19937_64& rng) {
    std::gamma_distribution<double> gamma(alpha, 1.0);
    RowMajorMatrixXd A(N, K);
    for (int i = 0; i < N; ++i) {
        double sum = 0.0;
        for (int k = 0; k < K; ++k) {
            double v = gamma(rng);
            if (v <= 0.0) v = 1e-8;
            A(i, k) = v;
            sum += v;
        }
        if (sum <= 0.0) {
            A.row(i).setConstant(1.0 / static_cast<double>(K));
        } else {
            A.row(i) /= sum;
        }
    }
    return A;
}

static bool simulate_once(int N, int K, std::mt19937_64& rng,
                          Eigen::VectorXd& se_fisher,
                          Eigen::VectorXd& se_robust) {
    std::uniform_real_distribution<double> unif_c(0.5, 2.0);
    std::normal_distribution<double> norm_b(0.0, 0.2);
    std::normal_distribution<double> norm_o(0.0, 0.1);

    RowMajorMatrixXd A = sample_dirichlet(N, K, 0.7, rng);
    Eigen::VectorXd x = Eigen::VectorXd::Zero(N);
    Eigen::VectorXd c = Eigen::VectorXd::Zero(N);
    for (int i = 0; i < N; ++i) {
        x(i) = (i < N / 2) ? -1.0 : 1.0;
        c(i) = unif_c(rng);
    }
    std::shuffle(x.data(), x.data() + x.size(), rng);

    Eigen::VectorXd oK(K);
    Eigen::VectorXd b_true(K);
    for (int k = 0; k < K; ++k) {
        oK(k) = 1.3 + norm_o(rng);
        b_true(k) = norm_b(rng);
    }

    std::vector<uint32_t> ids;
    std::vector<double> cnts;
    ids.reserve(N);
    cnts.reserve(N);

    for (int i = 0; i < N; ++i) {
        double sum_exp = 0.0;
        for (int k = 0; k < K; ++k) {
            double eta = oK(k) + x(i) * b_true(k);
            double term = std::exp(std::min(eta, 40.0));
            sum_exp += A(i, k) * term;
        }
        double lambda = c(i) * (sum_exp - 1.0);
        if (!(lambda > 0.0)) {
            return false;
        }
        std::poisson_distribution<int> pois(lambda);
        int y = pois(rng);
        if (y > 0) {
            ids.push_back(static_cast<uint32_t>(i));
            cnts.push_back(static_cast<double>(y));
        }
    }

    if (ids.empty()) {
        return false;
    }

    MLEOptions opt;
    opt.se_flag = 3;
    opt.hc_type = 3;
    opt.se_stabilize = 10.0;

    MixPoisLog1pSparseContext ctx(A, x, c, opt, /*robust_se_full=*/true, N);
    MixPoisLog1pSparseProblem P(ctx);
    P.reset_feature(ids, cnts, oK);

    MLEStats stats;
    P.compute_se(b_true, opt, stats);

    se_fisher = stats.se_fisher;
    se_robust = stats.se_robust;
    if (se_fisher.size() != K || se_robust.size() != K) {
        return false;
    }
    return true;
}

int test(int argc, char** argv) {
    int N = 400;
    int K = 4;
    int reps = 200;
    uint64_t seed = 123;
    ParamList pl;
    pl.add_option("N", "Number of rows (N)", N)
      .add_option("K", "Number of cols (K)", K)
      .add_option("R", "Number of replicates", reps)
      .add_option("seed", "Random seed", seed);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }


    if (argc > 1) N = std::max(50, std::atoi(argv[1]));
    if (argc > 2) K = std::max(2, std::atoi(argv[2]));
    if (argc > 3) reps = std::max(20, std::atoi(argv[3]));
    if (argc > 4) seed = static_cast<uint64_t>(std::strtoull(argv[4], nullptr, 10));

    std::mt19937_64 rng(seed);
    double sum_ratio = 0.0;
    double sum_log_ratio = 0.0;
    int count = 0;
    int skipped = 0;

    for (int r = 0; r < reps; ++r) {
        Eigen::VectorXd se_fisher;
        Eigen::VectorXd se_robust;
        if (!simulate_once(N, K, rng, se_fisher, se_robust)) {
            skipped++;
            continue;
        }
        for (int k = 0; k < K; ++k) {
            double sf = se_fisher(k);
            double sr = se_robust(k);
            if (!std::isfinite(sf) || !std::isfinite(sr) || sf <= 0.0) {
                continue;
            }
            double ratio = sr / sf;
            if (!std::isfinite(ratio) || ratio <= 0.0) {
                continue;
            }
            sum_ratio += ratio;
            sum_log_ratio += std::log(ratio);
            count++;
        }
    }
    if (count == 0) {
        std::cout << "All SE failed.\n";
        return 0;
    }
    double mean_ratio = sum_ratio / static_cast<double>(count);
    double mean_log_ratio = sum_log_ratio / static_cast<double>(count);

    std::cout << "Simulations: " << reps << " (skipped " << skipped << ")\n";
    std::cout << "Mean ratio robust/fisher: " << mean_ratio << "\n";
    std::cout << "Mean log ratio: " << mean_log_ratio << "\n";

    return 0;
}
