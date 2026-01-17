#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <limits>

#include "punkst.h"
#include "mixpois.hpp"
#include "poisreg.hpp"

static double max_abs_diff(const VectorXd& a, const VectorXd& b) {
    if (a.size() != b.size() || a.size() == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return (a - b).cwiseAbs().maxCoeff();
}

static bool all_finite_positive(const VectorXd& v) {
    for (int i = 0; i < v.size(); ++i) {
        if (!(v[i] > 0.0) || !std::isfinite(v[i])) {
            return false;
        }
    }
    return true;
}

int32_t test(int32_t argc, char** argv) {
    int N = 3000;
    int K = 6;
    int seed = 7;

    ParamList pl;
    pl.add_option("N", "Number of rows", N)
      .add_option("K", "Number of mixture components", K)
      .add_option("seed", "Random seed", seed);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    std::mt19937 rng(seed);
    std::gamma_distribution<double> gamma(1.0, 1.0);
    std::uniform_real_distribution<double> unif_c(0.8, 2.0);
    std::uniform_real_distribution<double> unif_b0(0.7, 1.2);
    std::uniform_real_distribution<double> unif_b(-0.3, 0.3);
    std::normal_distribution<double> nrm(0.0, 0.1);
    std::bernoulli_distribution bern(0.5);

    RowMajorMatrixXd theta(N, K);
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) {
            double v = gamma(rng) + 1e-3;
            theta(i, k) = v;
            s += v;
        }
        theta.row(i) /= s;
    }

    VectorXd c(N);
    VectorXd x(N);
    for (int i = 0; i < N; ++i) {
        c[i] = unif_c(rng);
        x[i] = bern(rng) ? 1.0 : -1.0;
    }

    VectorXd b0_true(K);
    VectorXd b_true(K);
    for (int k = 0; k < K; ++k) {
        b0_true[k] = unif_b0(rng);
        double bk = unif_b(rng);
        double cap = 0.8 * b0_true[k];
        if (std::abs(bk) > cap) {
            bk = std::copysign(cap, bk);
        }
        b_true[k] = bk;
    }

    std::vector<uint32_t> ids;
    std::vector<double> cnts;
    ids.reserve(N);
    cnts.reserve(N);
    double total_count = 0.0;
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) {
            s += theta(i, k) * std::exp(b0_true[k] + x[i] * b_true[k]);
        }
        double lambda = c[i] * (s - 1.0);
        if (lambda < 1e-8) lambda = 1e-8;
        std::poisson_distribution<int> pois(lambda);
        int y = pois(rng);
        total_count += y;
        if (y > 0) {
            ids.push_back(static_cast<uint32_t>(i));
            cnts.push_back(static_cast<double>(y));
        }
    }

    std::cout << "nnz (y_i>0): " << ids.size() << " / " << N << "\n";
    std::cout << "total count: " << total_count << "\n";

    VectorXd a_sum = VectorXd::Zero(K);
    for (int i = 0; i < N; ++i) {
        a_sum.noalias() += c[i] * theta.row(i).transpose();
    }

    OptimOptions optim;
    optim.max_iters = 200;
    optim.tol = 1e-7;
    optim.tron.enabled = true;

    MixPoisReg baseline(theta, c, a_sum, optim);
    VectorXd eta_init = b0_true;
    for (int k = 0; k < K; ++k) {
        eta_init[k] = std::max(0.0, eta_init[k] + nrm(rng));
    }
    VectorXd b0_hat = baseline.fit_one(ids, cnts, eta_init);

    MixPoisRegProblem P_base(theta, c, a_sum, ids, cnts, 1e-30);
    VectorXd beta_init(K);
    VectorXd beta_hat(K);
    for (int k = 0; k < K; ++k) {
        double e0 = std::exp(std::min(eta_init[k], 40.0));
        double eh = std::exp(std::min(b0_hat[k], 40.0));
        beta_init[k] = std::max(0.0, e0 - 1.0);
        beta_hat[k] = std::max(0.0, eh - 1.0);
    }
    double f_init = P_base.f(beta_init);
    double f_hat = P_base.f(beta_hat);
    double b0_err = max_abs_diff(b0_hat, b0_true);

    std::cout << "baseline f(init)=" << f_init << " f(hat)=" << f_hat
              << " max|b0_hat-b0_true|=" << b0_err << "\n";

    VectorXd wp = (c.array() * (x.array() > 0).cast<double>()).matrix();
    VectorXd wm = (c.array() * (x.array() < 0).cast<double>()).matrix();
    VectorXd s1p = theta.transpose() * wp;
    VectorXd s1m = theta.transpose() * wm;

    VectorXd s2p = VectorXd::Zero(K);
    VectorXd s2m = VectorXd::Zero(K);
    for (int i = 0; i < N; ++i) {
        if (c[i] <= 0.0 || x[i] == 0.0) continue;
        double ci2 = c[i] * c[i];
        auto ai = theta.row(i).array();
        if (x[i] > 0.0) {
            s2p.array() += ci2 * ai.square();
        } else {
            s2m.array() += ci2 * ai.square();
        }
    }

    MLEOptions mle_opt(optim);
    mle_opt.se_flag = 3;
    mle_opt.optim.set_bounds(-b0_hat, b0_hat);

    MixPoisLog1pSparseProblem P(theta, ids, cnts, x, c, b0_hat, mle_opt, s1p, s1m, &s2p, &s2m);
    VectorXd b_hat = VectorXd::Zero(K);
    MLEStats stats;
    double f_log1p_init = P.f(b_hat);
    double f_log1p = tron_solve(P, b_hat, mle_opt.optim, stats.optim);
    mix_pois_log1p_compute_se(P, b_hat, mle_opt, stats);

    VectorXd g(K), q(K);
    ArrayXd w;
    P.eval(b_hat, nullptr, &g, &q, &w);
    double g_norm = g.norm();
    double b_err = max_abs_diff(b_hat, b_true);
    bool se_ok = (stats.se_fisher.size() == K) && (stats.se_robust.size() == K)
        && all_finite_positive(stats.se_fisher) && all_finite_positive(stats.se_robust);

    bool q_ok = true;
    for (int k = 0; k < K; ++k) {
        if (!(q[k] > 0.0) || !std::isfinite(q[k])) {
            q_ok = false;
            break;
        }
    }

    std::cout << "log1p f(init)=" << f_log1p_init << " f(hat)=" << f_log1p
              << " max|b_hat-b_true|=" << b_err
              << " |g|=" << g_norm << "\n";
    std::cout << "se_fisher_ok=" << (se_ok ? "yes" : "no")
              << " q_diag_ok=" << (q_ok ? "yes" : "no") << "\n";

    bool ok = true;
    if (ids.empty()) {
        std::cerr << "No nonzero counts generated.\n";
        ok = false;
    }
    if (f_hat > f_init + 1e-9) {
        std::cerr << "Baseline fit did not improve objective.\n";
        ok = false;
    }
    if (b0_err > 0.35) {
        std::cerr << "Baseline b0 error too large: " << b0_err << "\n";
        ok = false;
    }
    if (f_log1p > f_log1p_init + 1e-9) {
        std::cerr << "Log1p fit did not improve objective.\n";
        ok = false;
    }
    if (b_err > 0.35) {
        std::cerr << "Log1p b error too large: " << b_err << "\n";
        ok = false;
    }
    if (!se_ok) {
        std::cerr << "Invalid SE estimates.\n";
        ok = false;
    }
    if (!q_ok) {
        std::cerr << "Invalid diagonal curvature from MixPoisLog1pSparseProblem.\n";
        ok = false;
    }

    return ok ? 0 : 1;
}
