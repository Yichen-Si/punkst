#include "punkst.h"
#include "glm.hpp"

#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include <cassert>
#include <algorithm>
#include <string>
#include <unordered_map>

// ------------------ tiny test utilities ------------------
static void require(bool cond, const char* msg) {
    if (!cond) {
        std::cerr << "TEST FAILED: " << msg << "\n";
        std::abort();
    }
}

static void info_skip(const std::string& name, const std::string& why) {
    std::cout << "[skip] " << name << " (" << why << ")\n";
}

static double max_abs(const Eigen::VectorXd& v) {
    double m = 0.0;
    for (int i = 0; i < v.size(); ++i) m = std::max(m, std::abs(v(i)));
    return m;
}

static bool approx(double a, double b, double tol) {
    return std::abs(a - b) <= tol;
}

static bool approx_vec(const Eigen::VectorXd& a, const Eigen::VectorXd& b, double tol) {
    if (a.size() != b.size()) return false;
    for (int i = 0; i < a.size(); ++i)
        if (!approx(a(i), b(i), tol)) return false;
    return true;
}

// Parse integer from argv with default
static int parse_int_arg(int argc, char** argv, int idx, int def) {
    if (idx >= argc) return def;
    try {
        return std::stoi(argv[idx]);
    } catch (...) {
        return def;
    }
}

static void usage(const char* prog) {
    std::cout << "Usage: " << prog << " [K] [G]\n"
              << "  K = number of slices (>=2), default 5\n"
              << "  G = number of datasets/groups in PairwiseBinomRobust (>=2), default 2\n"
              << "Example: " << prog << " 8 6\n";
}

// Make a simple diagonally-dominant mixing matrix and row-normalize.
static RowMajorMatrixXd make_mixing(int K, double diag_mass, std::mt19937_64& rng) {
    std::uniform_real_distribution<double> unif(0.0, 1.0);
    RowMajorMatrixXd C(K, K);
    C.setZero();
    for (int i = 0; i < K; ++i) {
        double rest = 1.0 - diag_mass;
        double s = 0.0;
        for (int j = 0; j < K; ++j) {
            if (j == i) continue;
            double x = unif(rng);
            C(i, j) = x;
            s += x;
        }
        if (s > 0) {
            for (int j = 0; j < K; ++j) if (j != i) C(i, j) = rest * (C(i, j) / s);
        }
        C(i, i) = diag_mass;
    }
    for (int i = 0; i < K; ++i) {
        double rs = C.row(i).sum();
        require(rs > 0.0, "row sum must be >0");
        C.row(i) /= rs;
    }
    return C;
}

static Eigen::VectorXd sigmoid_vec(const Eigen::VectorXd& x) {
    Eigen::VectorXd y(x.size());
    for (int i = 0; i < x.size(); ++i) {
        double z = x(i);
        if (z >= 0) { double t = std::exp(-z); y(i) = 1.0 / (1.0 + t); }
        else { double t = std::exp(z); y(i) = t / (1.0 + t); }
    }
    return y;
}

static double logit_safe(double p) {
    p = std::max(1e-12, std::min(1.0 - 1e-12, p));
    return std::log(p) - std::log(1.0 - p);
}

static Eigen::VectorXd logit_vec(const Eigen::VectorXd& p) {
    Eigen::VectorXd y(p.size());
    for (int i = 0; i < p.size(); ++i) y(i) = logit_safe(p(i));
    return y;
}

// Simulate Binomial(n, p)
static int rbinom(std::mt19937_64& rng, int n, double p) {
    p = std::max(0.0, std::min(1.0, p));
    std::binomial_distribution<int> dist(n, p);
    return dist(rng);
}

// Convert row-major to col-major Eigen::MatrixXd
static Eigen::MatrixXd to_colmajor(const RowMajorMatrixXd& C) {
    Eigen::MatrixXd X = C;
    return X;
}

// Build group lists from G: g0s = [0..G0-1], g1s=[G0..G-1]
static void make_group_lists(int G, std::vector<int32_t>& g0s, std::vector<int32_t>& g1s) {
    int G0 = std::max(1, G / 2);
    int G1 = G - G0;
    require(G1 >= 1, "G must be >=2");
    g0s.clear(); g1s.clear();
    for (int i = 0; i < G0; ++i) g0s.push_back((int32_t)i);
    for (int i = G0; i < G; ++i) g1s.push_back((int32_t)i);
}

// ------------------ Tests ------------------

static void test_identity_mixing_matches_observed(int K, int G) {
    std::cout << "[test] identity mixing -> beta_deconv == beta_obs (K=" << K << ",G=" << G << ")\n";
    const int M = 1;

    MultiSlicePairwiseBinom ms(K, G, M, /*min_unit_total=*/1.0);

    // Identity mixing for all datasets
    RowMajorMatrixXd I = RowMajorMatrixXd::Identity(K, K);
    for (int g = 0; g < G; ++g) ms.add_to_confusion(g, to_colmajor(I));

    // True probs per slice (make something varying but valid for any K>=2)
    Eigen::VectorXd p0_true(K), p1_true(K);
    for (int k = 0; k < K; ++k) {
        p0_true(k) = 0.05 + 0.25 * (double)k / std::max(1, K-1);     // ~[0.05..0.30]
        p1_true(k) = 0.08 + 0.30 * (double)(K-1-k) / std::max(1, K-1); // ~[0.38..0.08]
        p0_true(k) = std::min(0.45, std::max(0.02, p0_true(k)));
        p1_true(k) = std::min(0.45, std::max(0.02, p1_true(k)));
    }
    Eigen::VectorXd beta_true = logit_vec(p1_true) - logit_vec(p0_true);

    std::mt19937_64 rng(123);

    // Contrast groups
    std::vector<int32_t> g0s, g1s;
    make_group_lists(G, g0s, g1s);
    int G0 = (int)g0s.size(), G1 = (int)g1s.size();

    const int total_units_per_group = 200; // per slice
    const int n_trials = 80;
    const int u0 = std::max(1, total_units_per_group / G0);
    const int u1 = std::max(1, total_units_per_group / G1);

    for (int k = 0; k < K; ++k) {
        // group0 datasets
        for (int di = 0; di < G0; ++di) {
            int g = g0s[di];
            for (int i = 0; i < u0; ++i) {
                int y0 = rbinom(rng, n_trials, p0_true(k));
                std::unordered_map<int32_t,double> fm0; fm0[0] = (double)y0;
                ms.slice(k).add_unit(g, (double)n_trials, fm0);
            }
        }
        // group1 datasets
        for (int di = 0; di < G1; ++di) {
            int g = g1s[di];
            for (int i = 0; i < u1; ++i) {
                int y1 = rbinom(rng, n_trials, p1_true(k));
                std::unordered_map<int32_t,double> fm1; fm1[0] = (double)y1;
                ms.slice(k).add_unit(g, (double)n_trials, fm1);
            }
        }
    }

    ms.finished_adding_data();

    ContrastPrecomp pc = ms.prepare_contrast(g0s, g1s,
        /*max_iter=*/30, /*tol=*/1e-8, /*pi_eps=*/1e-8,
        /*lambda_beta=*/1e-6, /*lambda_alpha=*/1e-8, /*lm_damping=*/1e-6);

    MultiSliceOneResult out(K);

    bool ok = ms.compute_one_test_aggregate(/*f=*/0, g0s, g1s, pc, out,
        /*min_total_pair=*/10.0, /*pi_eps=*/1e-8, /*use_hc1=*/true, /*do_deconv=*/true);
    require(ok, "compute_one_test_aggregate ok");
    require(out.deconv_ok, "deconv_ok");

    // Identity mixing => deconv should match observed
    require(approx_vec(out.beta_deconv, out.beta_obs, 1e-3), "beta_deconv ~= beta_obs (I mixing)");

    // Should also be close-ish to true (sampling noise)
    require(approx_vec(out.beta_deconv, beta_true, 0.20), "beta_deconv ~= beta_true within sampling tol");

    std::cout << "  slice betas:\n";
    for (int k = 0; k < K; ++k) {
        std::cout << "    slice " << k << ": beta_deconv=" << out.beta_deconv(k)
                  << ", beta_obs=" << out.beta_obs(k)
                  << ", beta_true=" << beta_true(k)
                  << ", diff=" << (out.beta_deconv(k) - beta_true(k)) << "\n";
    }
}

static void test_known_mixing_recovers_true_beta(int K, int G) {
    std::cout << "[test] known mixing -> deconv recovers beta_true (K=" << K << ",G=" << G << ")\n";
    const int M = 1;
    MultiSlicePairwiseBinom ms(K, G, M, 1.0);
    std::mt19937_64 rng(456);

    std::vector<int32_t> g0s, g1s;
    make_group_lists(G, g0s, g1s);
    int G0 = (int)g0s.size(), G1 = (int)g1s.size();

    // Different mixings for the two groups
    RowMajorMatrixXd C0 = make_mixing(K, /*diag_mass=*/0.85, rng);
    RowMajorMatrixXd C1 = make_mixing(K, /*diag_mass=*/0.75, rng);

    // Assign group-level mixing to each dataset
    for (int di = 0; di < G0; ++di) ms.add_to_confusion(g0s[di], to_colmajor(C0));
    for (int di = 0; di < G1; ++di) ms.add_to_confusion(g1s[di], to_colmajor(C1));

    // Choose alpha and beta_true with two nonzero slices (works for any K>=2)
    Eigen::VectorXd alpha = Eigen::VectorXd::Constant(K, -1.0);
    Eigen::VectorXd beta_true = Eigen::VectorXd::Zero(K);
    int up = std::min(1, K-1);
    int down = std::max(0, K-2);
    if (down == up) down = 0;              // if K==2, up=1 down=0
    beta_true(up) = 1.0;
    beta_true(down) = (down == up) ? 0.0 : -0.8;

    Eigen::VectorXd p0_true = sigmoid_vec(alpha - 0.5 * beta_true);
    Eigen::VectorXd p1_true = sigmoid_vec(alpha + 0.5 * beta_true);

    // Observed after mixing (this is what slice-level binomial sees)
    Eigen::VectorXd p0_obs = C0 * p0_true;
    Eigen::VectorXd p1_obs = C1 * p1_true;

    const int total_units_per_group = 400; // per slice
    const int n_trials = 60;
    const int u0 = std::max(1, total_units_per_group / G0);
    const int u1 = std::max(1, total_units_per_group / G1);

    for (int k = 0; k < K; ++k) {
        for (int di = 0; di < G0; ++di) {
            int g = g0s[di];
            for (int i = 0; i < u0; ++i) {
                int y0 = rbinom(rng, n_trials, p0_obs(k));
                std::unordered_map<int32_t,double> fm0; fm0[0] = (double)y0;
                ms.slice(k).add_unit(g, (double)n_trials, fm0);
            }
        }
        for (int di = 0; di < G1; ++di) {
            int g = g1s[di];
            for (int i = 0; i < u1; ++i) {
                int y1 = rbinom(rng, n_trials, p1_obs(k));
                std::unordered_map<int32_t,double> fm1; fm1[0] = (double)y1;
                ms.slice(k).add_unit(g, (double)n_trials, fm1);
            }
        }
    }

    ms.finished_adding_data();

    ContrastPrecomp pc = ms.prepare_contrast(g0s, g1s,
        /*max_iter=*/50, /*tol=*/1e-8, /*pi_eps=*/1e-8,
        /*lambda_beta=*/1e-3, /*lambda_alpha=*/1e-6, /*lm_damping=*/1e-4);

    MultiSliceOneResult out(K);
    bool ok = ms.compute_one_test_aggregate(0, g0s, g1s, pc, out,
        /*min_total_pair=*/10.0, /*pi_eps=*/1e-8, /*use_hc1=*/true, /*do_deconv=*/true);
    require(ok, "compute_one_test_aggregate ok");
    require(out.deconv_ok, "deconv_ok");

    // Expect near beta_true (sampling noise)
    require(approx_vec(out.beta_deconv, beta_true, 0.30), "beta_deconv ~= beta_true within tol");

    // Null slices should be near 0-ish
    for (int k = 0; k < K; ++k) {
        if (std::abs(beta_true(k)) < 1e-12) {
            require(std::abs(out.beta_deconv(k)) < 0.45, "null slice beta small-ish");
        }
        std::cout << "  slice " << k << ": beta_deconv=" << out.beta_deconv(k)
                  << ", beta_true=" << beta_true(k)
                  << ", diff=" << (out.beta_deconv(k) - beta_true(k)) << "\n";
    }
}

static void test_null_no_signal(int K, int G) {
    std::cout << "[test] null -> beta_deconv near 0 (K=" << K << ",G=" << G << ")\n";
    const int M = 1;
    MultiSlicePairwiseBinom ms(K, G, M, 1.0);
    std::mt19937_64 rng(789);

    std::vector<int32_t> g0s, g1s;
    make_group_lists(G, g0s, g1s);
    int G0 = (int)g0s.size(), G1 = (int)g1s.size();

    RowMajorMatrixXd C0 = make_mixing(K, 0.80, rng);
    RowMajorMatrixXd C1 = make_mixing(K, 0.80, rng);

    for (int di = 0; di < G0; ++di) ms.add_to_confusion(g0s[di], to_colmajor(C0));
    for (int di = 0; di < G1; ++di) ms.add_to_confusion(g1s[di], to_colmajor(C1));

    Eigen::VectorXd alpha = Eigen::VectorXd::Constant(K, -0.6);
    Eigen::VectorXd p_true = sigmoid_vec(alpha);
    Eigen::VectorXd p0_obs = C0 * p_true;
    Eigen::VectorXd p1_obs = C1 * p_true;

    const int total_units_per_group = 300;
    const int n_trials = 50;
    const int u0 = std::max(1, total_units_per_group / G0);
    const int u1 = std::max(1, total_units_per_group / G1);

    for (int k = 0; k < K; ++k) {
        for (int di = 0; di < G0; ++di) {
            int g = g0s[di];
            for (int i = 0; i < u0; ++i) {
                int y0 = rbinom(rng, n_trials, p0_obs(k));
                std::unordered_map<int32_t,double> fm0; fm0[0] = (double)y0;
                ms.slice(k).add_unit(g, (double)n_trials, fm0);
            }
        }
        for (int di = 0; di < G1; ++di) {
            int g = g1s[di];
            for (int i = 0; i < u1; ++i) {
                int y1 = rbinom(rng, n_trials, p1_obs(k));
                std::unordered_map<int32_t,double> fm1; fm1[0] = (double)y1;
                ms.slice(k).add_unit(g, (double)n_trials, fm1);
            }
        }
    }

    ms.finished_adding_data();

    ContrastPrecomp pc = ms.prepare_contrast(g0s, g1s,
        40, 1e-8, 1e-8, 1e-3, 1e-6, 1e-4);

    MultiSliceOneResult out(K);
    bool ok = ms.compute_one_test_aggregate(0, g0s, g1s, pc, out,
        10.0, 1e-8, true, true);
    require(ok, "compute_one_test_aggregate ok");
    require(out.deconv_ok, "deconv_ok");

    require(max_abs(out.beta_deconv) < 0.6, "null max |beta_deconv| not too large");

    std::cout << "  (max |beta_deconv| = " << max_abs(out.beta_deconv) << ")\n";
}

static void test_low_count_slice_masking(int K, int G) {
    if (K < 3) {
        info_skip("low_count_slice_masking", "requires K>=3");
        return;
    }
    std::cout << "[test] low-count slice masking doesn't pollute others (K=" << K << ",G=" << G << ")\n";

    const int M = 1;
    MultiSlicePairwiseBinom ms(K, G, M, 1.0);
    std::mt19937_64 rng(321);

    std::vector<int32_t> g0s, g1s;
    make_group_lists(G, g0s, g1s);
    int G0 = (int)g0s.size(), G1 = (int)g1s.size();

    RowMajorMatrixXd I = RowMajorMatrixXd::Identity(K, K);
    for (int g = 0; g < G; ++g) ms.add_to_confusion(g, to_colmajor(I));

    // Choose one signal slice and one low-count slice (distinct)
    int low = 0;
    int sig = std::min(2, K-1);
    if (sig == low) sig = 1;

    Eigen::VectorXd p0 = Eigen::VectorXd::Constant(K, 0.2);
    Eigen::VectorXd p1 = p0;
    p0(sig) = 0.05;
    p1(sig) = 0.20;

    const int total_units_per_group = 200;
    const int n_trials = 50;
    const int u0 = std::max(1, total_units_per_group / G0);
    const int u1 = std::max(1, total_units_per_group / G1);

    for (int k = 0; k < K; ++k) {
        int units_scale = (k == low) ? 1 : 1;  // we'll downsample by using 1 unit total effectively
        for (int di = 0; di < G0; ++di) {
            int g = g0s[di];
            int u = (k == low) ? 1 : u0;
            for (int i = 0; i < u; ++i) {
                int y0 = rbinom(rng, n_trials, p0(k));
                std::unordered_map<int32_t,double> fm0; fm0[0] = (double)y0;
                ms.slice(k).add_unit(g, (double)n_trials, fm0);
            }
        }
        for (int di = 0; di < G1; ++di) {
            int g = g1s[di];
            int u = (k == low) ? 1 : u1;
            for (int i = 0; i < u; ++i) {
                int y1 = rbinom(rng, n_trials, p1(k));
                std::unordered_map<int32_t,double> fm1; fm1[0] = (double)y1;
                ms.slice(k).add_unit(g, (double)n_trials, fm1);
            }
        }
    }

    ms.finished_adding_data();

    ContrastPrecomp pc = ms.prepare_contrast(g0s, g1s, 40, 1e-8, 1e-8, 1e-6, 1e-8, 1e-6);

    MultiSliceOneResult out(K);
    // set min_total_pair large to force low slice to fail
    bool ok = ms.compute_one_test_aggregate(0, g0s, g1s, pc, out,
        /*min_total_pair=*/ (double)(G * n_trials * 5), /*pi_eps=*/1e-8, /*use_hc1=*/true, /*do_deconv=*/true);
    require(ok, "compute_one_test_aggregate ok");

    require(out.slice_ok[(size_t)low] == 0, "low slice should be slice_ok=0");
    require(out.beta_deconv(sig) > 0.5, "signal slice beta positive");

    // pick another slice that's not low or signal and check near 0
    int other = 0;
    while (other == low || other == sig) ++other;
    require(other < K, "have an 'other' slice");
    require(std::abs(out.beta_deconv(other)) < 0.5, "other slice near 0");

    std::cout << "  (low slice " << low << " masked, sig slice " << sig
              << " beta=" << out.beta_deconv(sig)
              << ", other slice " << other
              << " beta=" << out.beta_deconv(other) << ")\n";
}


int32_t test(int32_t argc, char** argv) {
    int G = 2;
    int K = 10;
    int seed = 7;

    ParamList pl;
    pl.add_option("K", "Number of mixture components", K)
      .add_option("seed", "Random seed", seed);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (K < 2 || G < 2) {
        std::cerr << "Error: require K>=2 and G>=2\n";
        return 1;
    }

    test_identity_mixing_matches_observed(K, G);
    test_known_mixing_recovers_true_beta(K, G);
    test_null_no_signal(K, G);
    test_low_count_slice_masking(K, G);

    return 0;
}
