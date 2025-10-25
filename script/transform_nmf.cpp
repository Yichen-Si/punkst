#include <iostream>
#include <iomanip>
#include <random>
#include "punkst.h"
#include "utils.h"
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include "poisnmf.hpp"

int32_t cmdNmfTransform(int32_t argc, char** argv) {

    std::string inFile, metaFile, featureFile, covarFile, outPrefix;
    std::string modelFile, covarCoefFile;
    std::vector<uint32_t> covar_idx;
    int32_t seed = -1, nThreads = 1, debug_ = 0, debug_N = 0, verbose = 500000;
    int32_t minCount = 50;
    int32_t max_iter_inner = 20;
    double tol_inner = 1e-6;
    double size_factor = 10000, c = -1;
    bool allow_na = false;
    bool exact = false;
    int32_t se_method = 1;
    int32_t mode = 1;

    ParamList pl;
    // Input Options
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true)
      .add_option("in-covar", "Covariate file", covarFile)
      .add_option("allow-na", "Replace non-numerical values in covariates with zero", allow_na)
      .add_option("icol-covar", "Column indices (0-based) in --in-covar to use", covar_idx)
      .add_option("in-model", "Input model (beta) file", modelFile, true)
      .add_option("in-covar-coef", "Input covariate coefficients (Bcov) file", covarCoefFile);
    // Transform & Algorithm Options
    pl.add_option("mode", "Algorithm for regression (1:TRON, 2:FISTA, 3:DiagLS)", mode)
      .add_option("c", "Constant c in log(1+lambda/c)", c)
      .add_option("size-factor", "L: c_i=y_i/L, g()=log(1+lambda/c_i)", size_factor)
      .add_option("max-iter-inner", "Maximum inner iterations for transform", max_iter_inner)
      .add_option("tol-inner", "Inner tolerance for transform", tol_inner)
      .add_option("exact", "Exact, no approximation on zero terms", exact)
      .add_option("se-method", "Method for calculating SE of beta: 1: Fisher, 2: robust", se_method)
      .add_option("seed", "Random seed", seed)
      .add_option("threads", "Number of threads", nThreads);
    // Output Options
    pl.add_option("out-prefix", "Output prefix", outPrefix, true)
      .add_option("min-count", "Minimum total count per unit", minCount)
      .add_option("verbose", "Verbose", verbose)
      .add_option("debug-N", "Debug with the first N units", debug_N)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    // --- 1. Load Model ---
    notice("Loading model...");
    RowMajorMatrixXd beta, bcov;
    std::vector<std::string> model_features, covar_names_from_file;
    std::vector<std::string> tokens;

    read_matrix_from_file(modelFile, beta, &model_features, &tokens);
    const int32_t K = beta.cols();
    const int32_t M = beta.rows();
    notice("Loaded model with %d features and %d factors", M, K);

    PoissonLog1pNMF nmf(K, M, nThreads, seed, exact, debug_);
    nmf.set_beta(beta);

    if (!covarCoefFile.empty()) {
        read_matrix_from_file(covarCoefFile, bcov, &tokens, &covar_names_from_file);
        if (bcov.rows() != M) {
            error("Covariate coefficient matrix has %ld rows, but model has %d features.", bcov.rows(), M);
        }
        nmf.set_covar_coef(bcov);
    }

    MLEOptions opts{};
    opts.optim.max_iters = max_iter_inner;
    opts.optim.tol = tol_inner;
    if (mode == 1) {
        opts.optim.tron.enabled = true;
    } else if (mode == 2) {
        opts.optim.acg.enabled = true;
    } else if (mode == 3) {
        opts.optim.ls.enabled = true;
    }
    opts.compute_residual = true;
    opts.compute_var_mu = true;
    opts.se_flag = se_method == 2 ? 2 : 1;

    // --- 2. Load Input Data ---
    notice("Loading input data...");
    HexReader reader(metaFile);
    reader.setFeatureIndexRemap(model_features);
    std::vector<SparseObs> docs;
    std::vector<std::string> rnames, covar_names;
    int32_t N = read_sparse_obs(inFile, reader, docs,
        rnames, minCount, size_factor, c,
        &covarFile, &covar_idx, &covar_names,
        allow_na, debug_N);
    int32_t n_covar = static_cast<int32_t>(covar_idx.size());
    notice("Read %d documents with %d features", N, M);

    // --- 3. Transform Data ---
    std::vector<MLEStats> stats;
    ArrayXd resids;
    RowMajorMatrixXd theta = nmf.transform(docs, opts, stats, &resids);

    std::string outf;
    std::ofstream ofs;

    outf = outPrefix + ".feature_residuals.tsv";
    const auto& sums = reader.getFeatureSums();
    ofs.open(outf);
    if (!ofs) {
        error("Cannot open output file %s", outf.c_str());
    }
    ofs << "Feature\tTotalCount\tResidual\n";
    for (int32_t m = 0; m < M; ++m) {
        ofs << model_features[m] << "\t"
            << std::setprecision(0) << std::fixed << sums[m] << "\t"
            << std::setprecision(4) << std::defaultfloat << resids[m]/N << "\n";
    }
    ofs.close();
    notice("Wrote per-feature averaged residuals to %s", outf.c_str());

    outf = outPrefix + ".theta.tsv";
    write_matrix_to_file(outf, theta, 4, false, rnames, "Index");
    notice("Wrote theta to %s", outf.c_str());

    outf = outPrefix + ".fit_stats.tsv";
    ofs.open(outf);
    if (!ofs) {
        error("Cannot open output file %s", outf.c_str());
    }
    ofs << "Index\tTotalCount\tll\tResidual\tVarMu\n";
    for (size_t i = 0; i < N; i++) {
        ofs << rnames[i] << "\t"
            << std::setprecision(2) << std::fixed << docs[i].ct_tot << "\t"
            << std::setprecision(4) << std::defaultfloat << stats[i].pll << "\t"
            << std::setprecision(4) << stats[i].residual << "\t"
            << std::setprecision(4) << stats[i].var_mu << "\n";
    }
    ofs.close();
    notice("Wrote goodness of fit statistics to %s", outf.c_str());

    return 0;
}
