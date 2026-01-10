#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include "punkst.h"
#include "mixpois.hpp"


static double rel_err(double a, double b) {
    double denom = std::max(1.0, std::max(std::abs(a), std::abs(b)));
    return std::abs(a - b) / denom;
}

static double vec_rel_err_inf(const VectorXd& a, const VectorXd& b) {
    assert(a.size() == b.size());
    double maxe = 0.0;
    for (int i = 0; i < a.size(); ++i) {
        maxe = std::max(maxe, rel_err(a[i], b[i]));
    }
    return maxe;
}

// Finite difference gradient check (central differences)
static VectorXd finite_diff_grad(MixPoisRegProblem& P, const VectorXd& beta, double eps = 1e-6) {
    const int K = (int)beta.size();
    VectorXd g(K);
    for (int k = 0; k < K; ++k) {
        VectorXd bp = beta, bm = beta;
        double h = eps * std::max(1.0, std::abs(beta[k]));
        bp[k] += h;
        bm[k] = std::max(0.0, bm[k] - h); // keep feasible (beta>=0)

        double fp = P.f(bp);
        double fm = P.f(bm);
        // If we clipped bm, the step is not symmetric; still OK but less accurate.
        // Keep it simple here by ensuring beta[k] is not too close to 0 in test data.
        g[k] = (fp - fm) / (bp[k] - bm[k]);
    }
    return g;
}

// Finite difference Hv check using gradient differences
static VectorXd finite_diff_Hv(MixPoisRegProblem& P, const VectorXd& beta, const VectorXd& v, double eps = 1e-6) {
    double h = eps * std::max(1.0, beta.norm());
    VectorXd bp = beta + h * v;
    VectorXd bm = beta - h * v;

    // Keep feasibility beta>=0
    bp = bp.array().max(0.0);
    bm = bm.array().max(0.0);

    VectorXd gp(beta.size()), gm(beta.size());
    P.grad(bp, gp);
    P.grad(bm, gm);

    return (gp - gm) / (2.0 * h);
}

int32_t test(int32_t argc, char** argv) {
    int N = 500;
    int K = 20;
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
    std::normal_distribution<double> nrm(0.0, 1.0);

    // Random theta (row-stochastic)
    RowMajorMatrixXd theta(N, K);
    std::gamma_distribution<double> gamma(1.0, 1.0);
    for (int i = 0; i < N; ++i) {
        double s = 0.0;
        for (int k = 0; k < K; ++k) {
            double v = gamma(rng);
            theta(i, k) = v;
            s += v;
        }
        theta.row(i) /= s;
    }

    // Random c_i ~ Uniform(0.5, 2.0)
    VectorXd c(N);
    std::uniform_real_distribution<double> uni_c(0.5, 2.0);
    for (int i = 0; i < N; ++i) c[i] = uni_c(rng);

    // Compute a_sum_k = sum_i c_i * theta_{ik}
    VectorXd a_sum = VectorXd::Zero(K);
    for (int i = 0; i < N; ++i) {
        a_sum.noalias() += c[i] * theta.row(i).transpose();
    }

    // True eta, beta = exp(eta)-1 (keep away from 0 to avoid FD asymmetry)
    VectorXd eta_true(K);
    std::uniform_real_distribution<double> uni_eta(0.2, 1.0);
    for (int k = 0; k < K; ++k) eta_true[k] = uni_eta(rng);

    VectorXd beta_true(K);
    for (int k = 0; k < K; ++k) beta_true[k] = std::expm1(eta_true[k]); // > 0

    // Generate one synthetic column's x_i ~ Poisson(lambda_i)
    // lambda_i = c_i * theta_i^T beta_true
    std::vector<uint32_t> ids;
    std::vector<double> cnts;
    ids.reserve(N);
    cnts.reserve(N);

    for (int i = 0; i < N; ++i) {
        double lam = c[i] * theta.row(i).dot(beta_true);
        // keep lam in a comfortable range for stable FD
        if (lam < 1e-6) lam = 1e-6;

        std::poisson_distribution<int> pois(lam);
        int x = pois(rng);
        if (x > 0) {
            ids.push_back((uint32_t)i);
            cnts.push_back((double)x);
        }
    }

    std::cout << "nnz (x_i>0): " << ids.size() << " / " << N << "\n";

    // Construct problem with tiny floor so clamp is inactive
    MixPoisRegProblem P(theta, c, a_sum, ids, cnts, /*lambda_floor=*/1e-30);

    // ---- Test 1: f == eval(f_out) ----
    {
        double f1 = P.f(beta_true);

        double f2 = 0.0;
        VectorXd g(K), q(K);
        ArrayXd w;
        P.eval(beta_true, &f2, &g, &q, &w);
        std::cout << "f(beta_true): " << f1 << "  eval f_out: " << f2
                  << "  relerr: " << rel_err(f1, f2) << "\n";
    }

    // ---- Test 2: analytic gradient vs finite difference ----
    {
        VectorXd g_ana(K);
        P.grad(beta_true, g_ana);

        VectorXd g_fd = finite_diff_grad(P, beta_true, 1e-6);

        double err = vec_rel_err_inf(g_ana, g_fd);
        std::cout << "grad relerr_inf: " << err << "\n";
    }

    // ---- Test 3: Hv vs finite difference of gradient ----
    {
        // Get w at beta_true
        double f_out = 0.0;
        VectorXd g(K), q(K);
        ArrayXd w;
        P.eval(beta_true, &f_out, &g, &q, &w);

        // Random direction v
        VectorXd v(K);
        for (int k = 0; k < K; ++k) v[k] = nrm(rng);
        v.normalize();

        VectorXd Hv_ana = P.make_Hv(w)(v);
        VectorXd Hv_fd  = finite_diff_Hv(P, beta_true, v, 1e-6);

        double err = vec_rel_err_inf(Hv_ana, Hv_fd);
        std::cout << "Hv relerr_inf: " << err << "\n";
    }

    // ---- Optional Test 4: end-to-end TRON fit
    {
        OptimOptions opt;

        MixPoisReg fitter(theta, c, a_sum, opt);

        // Start near truth
        VectorXd eta0 = eta_true;
        for (int k = 0; k < K; ++k) eta0[k] = std::max(0.0, eta0[k] + 0.1 * nrm(rng));

        VectorXd eta_hat = fitter.fit_one(ids, cnts, eta0);

        // Compare objectives
        VectorXd beta0(K), betahat(K);
        for (int k = 0; k < K; ++k) beta0[k] = std::expm1(std::min(eta0[k], 40.0));
        for (int k = 0; k < K; ++k) betahat[k] = std::expm1(std::min(eta_hat[k], 40.0));

        double f0 = P.f(beta0);
        double fh = P.f(betahat);

        std::cout << "TRON objective: f(init)=" << f0 << "  f(hat)=" << fh << "\n";
        if(fh > f0 + 1e-9) {
            std::cerr << "TRON fit failed to improve objective: f(hat)=" << fh << " > f(init)=" << f0 << "\n";
        }
    }

    return 0;
}
