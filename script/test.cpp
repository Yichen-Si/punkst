#include <iostream>
#include <iomanip>
#include <random>
#include "punkst.h"
#include "utils.h"
#include "nanoflann.hpp"
#include "nanoflann_utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include "Eigen/Dense"

#include "poisreg.hpp"
#include "poisnmf.hpp"

struct SimConfig {
    int N = 2000;     // rows (documents)
    int M = 1000;     // columns (terms)
    int K = 32;       // factors
    double c = 0.1;  // Poisson scale
    double eta_cap = 2.0;     // cap max(η) to control rates/sparsity
    unsigned seed = 42;
    int nThreads = 1;
    int max_outer = 50;
    double tol_outer = 1e-4;
};

struct SimData {
    RowMajorMatrixXd Aeta_true;  // N x M of η = θ_true * β_true^T
    RowMajorMatrixXd Theta_true; // N x K
    RowMajorMatrixXd Beta_true;  // M x K
    std::vector<Document> docs;  // row-sparse counts
};

// Build row-sparse documents from dense counts
static std::vector<Document> build_docs(const RowMajorMatrixXd& Aeta,
                                        double c, unsigned seed) {
    const int N = (int)Aeta.rows();
    const int M = (int)Aeta.cols();
    std::mt19937_64 rng(seed);
    std::vector<Document> docs(N);
    docs.reserve(N);
    double avg_n = 0.0, avg_count = 0.0;
    for (int i = 0; i < N; ++i) {
        docs[i].ids.clear();
        docs[i].cnts.clear();
        // If your Document has a length field (e.g., docs[i].N), set it:
        // docs[i].N = M;
        for (int j = 0; j < M; ++j) {
            double eta = Aeta(i, j);
            double lam = c * (std::exp(eta) - 1.0);
            if (lam <= 0) continue;
            int y = (lam < 30.0)
                  ? std::poisson_distribution<int>(lam)(rng)
                  : (int)std::round(std::max(0.0, std::normal_distribution<double>(lam, std::sqrt(lam))(rng)));
            if (y > 0) {
                docs[i].ids.push_back((uint32_t)j);
                docs[i].cnts.push_back((double)y);
            }
        }
        avg_n += docs[i].ids.size();
        avg_count += std::accumulate(docs[i].cnts.begin(), docs[i].cnts.end(), 0.0);
    }
    avg_n /= (double)N;
    avg_count /= (double)N;
    std::cout << "Average non-zero words per document: " << avg_n << " /" << M << std::endl;
    std::cout << "Average total words per document: " << avg_count << std::endl;
    return docs;
}

// Simulate θ_true, β_true, then η = θ β^T, scaled to cap max(η)
static SimData simulate(const SimConfig& cfg) {
    std::mt19937_64 rng(cfg.seed);
    std::gamma_distribution<double> G(2.0, 0.5);
    const int N = cfg.N, M = cfg.M, K = cfg.K;

    RowMajorMatrixXd Theta_true(N, K), Beta_true(M, K);
    for (int i=0;i<N;++i) for (int k=0;k<K;++k) Theta_true(i,k) = std::max(1e-9, G(rng));
    for (int j=0;j<M;++j) for (int k=0;k<K;++k) Beta_true(j,k)  = std::max(1e-9, G(rng)) / std::sqrt((double)K);

    RowMajorMatrixXd Aeta = Theta_true * Beta_true.transpose(); // N x M
    // Cap η to control λ and sparsity
    double mx = Aeta.maxCoeff();
    if (mx > cfg.eta_cap) {
        double s = cfg.eta_cap / mx;
        Aeta *= s;
        Theta_true *= std::sqrt(s);
        Beta_true  *= std::sqrt(s); // keep product ≈ scaled
    }
    // Build docs
    auto docs = build_docs(Aeta, cfg.c, cfg.seed + 1);
    return {std::move(Aeta), std::move(Theta_true), std::move(Beta_true), std::move(docs)};
}

// Exact negative log-likelihood on row-sparse docs, given θ, β
static double nll_exact(const std::vector<Document>& docs,
                        const RowMajorMatrixXd& theta,
                        const RowMajorMatrixXd& beta,
                        double c) {
    const int N = (int)theta.rows();
    double total = 0.0;
    for (int i = 0; i < N; ++i) {
        const auto& di = docs[i];
        for (size_t t = 0; t < di.ids.size(); ++t) {
            int j = (int)di.ids[t];
            double eta = theta.row(i).dot(beta.row(j));
            double lam = c * (std::exp(eta) - 1.0);
            lam = std::max(lam, 1e-300);
            double y = di.cnts[t];
            total += -(y * std::log(lam) - lam);
        }
    }
    return total;
}

// RMSE on η over observed entries (NZ in docs)
static double rmse_eta_on_nz(const std::vector<Document>& docs,
                             const RowMajorMatrixXd& theta_hat,
                             const RowMajorMatrixXd& beta_hat,
                             const RowMajorMatrixXd& Aeta_true) {
    const int N = (int)theta_hat.rows();
    double se = 0.0;
    size_t nnz = 0;
    for (int i = 0; i < N; ++i) {
        const auto& di = docs[i];
        for (size_t t = 0; t < di.ids.size(); ++t) {
            int j = (int)di.ids[t];
            double ehat = theta_hat.row(i).dot(beta_hat.row(j));
            double etru = Aeta_true(i, j);
            double diff = ehat - etru;
            se += diff * diff;
        }
        nnz += di.ids.size();
    }
    return (nnz == 0) ? 0.0 : std::sqrt(se / (double)nnz);
}

struct RunSummary {
    std::string tag;           // approx:TRON / approx:FISTA / approx:DiagLS
    double time_ms = 0.0;
    double nll = 0.0;
    double rmse_eta = 0.0;
};

// Run one factorization with given inner solver options
static RunSummary run_one(const SimConfig& cfg,
                          const std::vector<Document>& docs,
                          const std::string& tag,
                          const MLEOptions& mopt,
                          PoissonLog1pNMF& nmf) {
    auto t0 = std::chrono::steady_clock::now();
    // Fit
    MLEOptions inner = mopt; // outer passes a base copy
    nmf.fit(docs, inner, cfg.max_outer, cfg.tol_outer);
    auto t1 = std::chrono::steady_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double nll = nll_exact(docs, nmf.get_theta(), nmf.get_model(), cfg.c);
    // For RMSE we need the truth; leave it to caller
    return {tag, ms, nll, 0.0};
}

int32_t test(int32_t argc, char** argv) {

	std::string intsv, outtsv;
    int32_t debug = 0, verbose = 500000;
    SimConfig cfg;

    ParamList pl;
    // Input Options
    pl.add_option("N", "Number of rows (N)", cfg.N, true)
      .add_option("M", "Number of cols (M)", cfg.M, true)
      .add_option("K", "Number of cols (K)", cfg.K, true)
      .add_option("c", "Constant c", cfg.c)
      .add_option("seed", "Random seed", cfg.seed)
      .add_option("threads", "Number of threads", cfg.nThreads);
    // Output Options
    pl.add_option("out-tsv", "Output TSV file", outtsv)
      .add_option("verbose", "Verbose", verbose)
      .add_option("debug", "Debug", debug);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }


    auto data = simulate(cfg);

    // Build three option profiles
    MLEOptions tron{};        tron.tron.enabled = true;  tron.acg.enabled = false; tron.ls.enabled = false;
    tron.max_iters = 20; tron.tol = 1e-6; tron.eps = 1e-12; tron.ridge = 1e-12;

    MLEOptions fista{};       fista.tron.enabled = false; fista.acg.enabled = true;  fista.ls.enabled = false;
    fista.max_iters = 50; fista.tol = 1e-6; fista.eps = 1e-12; fista.ridge = 1e-12;
    fista.acg.L0 = 1.0; fista.acg.bt_inc = 2.0; fista.acg.monotone = true; fista.acg.restart = true;

    MLEOptions diagls{};      diagls.tron.enabled = false; diagls.acg.enabled = false; diagls.ls.enabled = true;
    diagls.max_iters = 50; diagls.tol = 1e-6; diagls.eps = 1e-12; diagls.ridge = 1e-12;
    diagls.alpha = 1.0; diagls.ls.beta = 0.5; diagls.ls.c1 = 1e-4; diagls.ls.max_backtracks = 20;

    std::vector<RunSummary> results;

    // Approximate-Z version (recommended for large N)
    {
        std::cout << "\napprox:TRON:\n";
        PoissonLog1pNMF nmf(cfg.K, cfg.M, cfg.c, cfg.nThreads, (int)cfg.seed + 7);
        auto rs = run_one(cfg, data.docs, "approx:TRON", tron, nmf);
        rs.rmse_eta = rmse_eta_on_nz(data.docs, nmf.get_theta(), nmf.get_model(), data.Aeta_true);
        results.push_back(rs);
    }
    // {
    //     std::cout << "\napprox:FISTA:\n";
    //     PoissonLog1pNMF nmf(cfg.K, cfg.M, cfg.c, cfg.nThreads, (int)cfg.seed + 8);
    //     auto rs = run_one(cfg, data.docs, "approx:FISTA", fista, nmf);
    //     rs.rmse_eta = rmse_eta_on_nz(data.docs, nmf.get_theta(), nmf.get_model(), data.Aeta_true);
    //     results.push_back(rs);
    // }
    // {
    //     std::cout << "\napprox:DiagLS:\n";
    //     PoissonLog1pNMF nmf(cfg.K, cfg.M, cfg.c, cfg.nThreads, (int)cfg.seed + 9);
    //     auto rs = run_one(cfg, data.docs, "approx:DiagLS", diagls, nmf);
    //     rs.rmse_eta = rmse_eta_on_nz(data.docs, nmf.get_theta(), nmf.get_model(), data.Aeta_true);
    //     results.push_back(rs);
    // }

    // Exact-Z version (set exact flag; factorizer will pass through to exact solver)
    tron.exact_zero = true;   fista.exact_zero = true;   diagls.exact_zero = true;
    {
        std::cout << "\nexact:TRON:\n";
        PoissonLog1pNMF nmf(cfg.K, cfg.M, cfg.c, cfg.nThreads, (int)cfg.seed + 10);
        auto rs = run_one(cfg, data.docs, "exact:TRON", tron, nmf);
        rs.rmse_eta = rmse_eta_on_nz(data.docs, nmf.get_theta(), nmf.get_model(), data.Aeta_true);
        results.push_back(rs);
    }
    // {
    //     std::cout << "\nexact:FISTA:\n";
    //     PoissonLog1pNMF nmf(cfg.K, cfg.M, cfg.c, cfg.nThreads, (int)cfg.seed + 11);
    //     auto rs = run_one(cfg, data.docs, "exact:FISTA", fista, nmf);
    //     rs.rmse_eta = rmse_eta_on_nz(data.docs, nmf.get_theta(), nmf.get_model(), data.Aeta_true);
    //     results.push_back(rs);
    // }
    // {
    //     std::cout << "\nexact:DiagLS:\n";
    //     PoissonLog1pNMF nmf(cfg.K, cfg.M, cfg.c, cfg.nThreads, (int)cfg.seed + 12);
    //     auto rs = run_one(cfg, data.docs, "exact:DiagLS", diagls, nmf);
    //     rs.rmse_eta = rmse_eta_on_nz(data.docs, nmf.get_theta(), nmf.get_model(), data.Aeta_true);
    //     results.push_back(rs);
    // }

    // Write CSV
    std::ofstream ofs;
    if (outtsv.empty()) {
        ofs.basic_ios<char>::rdbuf(std::cout.rdbuf());
    } else {
        ofs.open(outtsv);
    }
    ofs << "tag,N,M,K,c,time_ms,nll,rmse_eta\n";
    for (auto const& r : results) {
        std::cout << std::setw(14) << r.tag
                  << "  time(ms)=" << std::fixed << std::setprecision(2) << r.time_ms
                  << "  nll=" << std::setprecision(6) << r.nll
                  << "  rmse_eta=" << std::setprecision(6) << r.rmse_eta << "\n";
        ofs << r.tag << "," << cfg.N << "," << cfg.M << "," << cfg.K << "," << cfg.c << ","
            << std::fixed << std::setprecision(3) << r.time_ms << ","
            << std::setprecision(9) << r.nll << ","
            << std::setprecision(9) << r.rmse_eta << "\n";
    }
    ofs.close();

    return 0;
}
