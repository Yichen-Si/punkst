#include <iostream>
#include <iomanip>
#include <random>
#include "punkst.h"
#include "utils.h"
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include "poisnmf.hpp"

int32_t cmdNmfPoisLog1p(int32_t argc, char** argv) {

	std::string inFile, metaFile, featureFile, covarFile, outPrefix;
    std::string include_ftr_regex, exclude_ftr_regex;
    std::vector<uint32_t> covar_idx;
    int32_t K;
    int32_t seed = -1, nThreads = 1, debug_ = 0, debug_N = 0, verbose = 500000;
    int32_t minCountTrain = 50, minCountFeature = 100;
    int32_t max_iter_outer = 20, max_iter_inner = 20;
    double covar_coef_min = -1e6, covar_coef_max = 1e6;
    double tol_outer = 1e-4, tol_inner = 1e-6;
    double size_factor = 10000, c = -1;
    bool allow_na = false;
    bool exact = false;
    bool write_se = false, compute_residual = false;
    bool test_beta_vs_null = false;
    bool transform = false;
    int32_t se_method = 1; // 1: fisher, 2: robust, 3: both
    double min_ct_de = 100, min_fc = 1.5, max_p = 0.05;
    int32_t mode = 1;

    ParamList pl;
    // Input Options
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true)
      .add_option("in-covar", "Covariate file", covarFile)
      .add_option("allow-na", "Replace non-numerical values in covariates with zero", allow_na)
      .add_option("icol-covar", "Column indices (0-based) in --in-covar to use", covar_idx);
    pl.add_option("K", "K", K, true)
      .add_option("mode", "Algorithm", mode)
      .add_option("c", "Constant c in log(1+lambda/c)", c)
      .add_option("size-factor", "L: c_i=y_i/L, g()=log(1+lambda/c_i)", size_factor)
      .add_option("max-iter-outer", "Maximum outer iterations", max_iter_outer)
      .add_option("max-iter-inner", "Maximum inner iterations", max_iter_inner)
      .add_option("tol-outer", "Outer tolerance", tol_outer)
      .add_option("tol-inner", "Inner tolerance", tol_inner)
      .add_option("exact", "Exact, no approximation on zero terms", exact)
      .add_option("covar-coef-min", "Lower bound for covariate coefficients", covar_coef_min)
      .add_option("covar-coef-max", "Upperbound for covariate coefficients", covar_coef_max)
      .add_option("seed", "Random seed", seed)
      .add_option("threads", "Number of threads", nThreads);
    pl.add_option("features", "Feature names and total counts file", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", minCountFeature)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex)
      .add_option("min-count-train", "Minimum total count per doc", minCountTrain);
    // DE test options
    pl.add_option("detest-vs-avg", "For each factor, test if each feature is  enriched compared to the average of the whole sample", test_beta_vs_null)
      .add_option("se-method", "Method for calculating SE of beta: 1: Fisher, 2: robust, 3: both", se_method)
      .add_option("min-fc", "Minimum fold-change for tests", min_fc)
      .add_option("max-p", "Maximum p-value for tests", max_p)
      .add_option("min-ct", "Minimum total count for a feature to be tested", min_ct_de);
    // Output Options
    pl.add_option("out-prefix", "Output prefix", outPrefix, true)
      .add_option("write-se", "Write standard errors for beta", write_se)
      .add_option("feature-residuals", "Compute and write per-feature residuals", compute_residual)
      .add_option("transform", "Transform the data after model fitting", transform)
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
    if (size_factor <= 0) {
        error("Error: --size-factor must be positive (Seurat uses 10000)");
    }
    bool per_doc_c = c <= 0;

    HexReader reader(metaFile);
    if (!featureFile.empty()) {
        reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
    }
    int32_t M = reader.nFeatures;

    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("Fail to open input file: %s", inFile.c_str());
    }
    std::string line, tmp;
    std::vector<std::string> tokens, covar_header;
    std::ifstream covarFileStream;
    int32_t n_covar = 0, n_tokens = 0;
    std::vector<std::string> covar_names;
    if (!covarFile.empty()) {
        covarFileStream.open(covarFile);
        if (!covarFileStream) {
            error("Fail to covariate file: %s", covarFile.c_str());
        }
        // Read the header line
        if (!std::getline(covarFileStream, line)) {
            error("Fail to parse covariate file: %s", covarFile.c_str());
        }
        split(covar_header, "\t ", line, UINT_MAX, true, true, true);
        n_tokens = static_cast<int32_t>(covar_header.size());
        if (covar_idx.empty()) {
            // Assuming use all columns except the first one
            n_covar = covar_header.size() - 1;
            for (int32_t i = 1; i < n_tokens; i++) {
                covar_idx.push_back(i);
                covar_names.push_back(covar_header[i]);
            }
        } else {
            n_covar = (int32_t) covar_idx.size();
            for (auto i : covar_idx) {
                if (i < 0 || i >= n_tokens) {
                    error("Covariate index %d is out of range [0,%lu)", i, covar_header.size());
                }
                covar_names.push_back(covar_header[i]);
            }
        }
        if (n_covar < 1) {
            error("The covariate file should have at least 2 columns");
        }
        notice("Covariate file has %d columns, using %d as covariates", n_tokens, n_covar);
    }

    std::vector<SparseObs> docs;
    if (reader.nUnits > 0) {
        docs.reserve(reader.nUnits);
    }
    std::vector<std::string> rnames;
    int32_t idx = 0;
    while (std::getline(inFileStream, line)) {
        idx++;
        if (idx % 10000 == 0) {
            notice("Read %d units...", idx);
        }
        SparseObs obs;
        int32_t ct = reader.parseLine(obs.doc, tmp, line);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        if (ct < minCountTrain) {
            continue;
        }
        obs.c = per_doc_c ? ct / size_factor : c;
        obs.ct_tot = ct;
        if (n_covar > 0) { // read covariates
            if (!std::getline(covarFileStream, line)) {
                error("The number of lines in covariate file is less than that in data file");
            }
            obs.covar = VectorXd::Zero(n_covar);
            split(tokens, "\t ", line, UINT_MAX, true, true, true);
            if (tokens.size() != n_tokens) {
                error("Number of columns (%lu) of line [%s] in covariate file does not match header (%d)", tokens.size(), line.c_str(), n_tokens);
            }
            for (int32_t i = 0; i < n_covar; i++) {
                if (!str2double(tokens[covar_idx[i]], obs.covar(i))) {
                    if (!allow_na)
                        error("Invalid value for %d-th covariate %s.", i+1, tokens[covar_idx[i]].c_str());
                    obs.covar(i) = 0;
                }
            }
        }
        docs.push_back(std::move(obs));
        rnames.emplace_back(std::to_string(idx-1));
        if (debug_N > 0 && idx > debug_N) {
            break;
        }
    }
    inFileStream.close();
    size_t N = docs.size();
    notice("Read %lu documents with %d features", N, M);

    PoissonLog1pNMF nmf(K, M, nThreads, seed, exact, debug_);

    // Set up MLE options (for subproblems)
    MLEOptions opts{};
    opts.optim.max_iters = max_iter_inner;
    opts.optim.tol = tol_inner;
    std::string model_name = "CD";
    if (mode == 1) {
        opts.optim.tron.enabled = true;
        model_name = "TRON";
    } else if (mode == 2) {
        opts.optim.acg.enabled = true;
        model_name = "FISTA";
    } else if (mode == 3) {
        opts.optim.ls.enabled = true;
        model_name = "DiagLS";
    }
    outPrefix += "." + model_name;
    if (test_beta_vs_null) {
        opts.store_cov = true;
        if (se_method == 0) {
            se_method = 1;
        }
        opts.se_flag = se_method;
        nmf.set_de_parameters(min_ct_de, min_fc, max_p);
    } else if (write_se) {
        opts.se_flag = se_method;
    }
    opts.compute_residual = compute_residual;

    // Fit the model
    nmf.fit(docs, opts, max_iter_outer, tol_outer);

    // Compute DE stats
    if (test_beta_vs_null) {
        for (uint32_t flag = 1; flag <= 2; flag++) {
            if ((se_method & flag) == 0) {
                continue;
            }
            auto results = nmf.test_beta_vs_null(flag);
            std::string outf = outPrefix + (flag == 1 ? ".detest.fisher.tsv" : ".detest.robust.tsv");
            std::ofstream ofs(outf);
            if (!ofs) {
                error("Cannot open output file %s", outf.c_str());
            }
            ofs << "Feature\tFactor\tDiff\tlog10Pval\tApproxFC\n";
            for (const auto& r : results) {
                ofs << reader.features[r.m] << "\t" << r.k1 << "\t"
                    << std::setprecision(6) << r.est << "\t"
                    << std::setprecision(4) << -r.log10p << "\t"
                    << std::setprecision(4) << r.fc << "\n";
            }
            ofs.close();
            notice("Wrote DE test results to %s", outf.c_str());
        }
    }

    std::string outf;
    outf = outPrefix + ".model.tsv";
    const auto& mat = nmf.get_model();
    write_matrix_to_file(outf, mat, 4, false, reader.features, "Feature");
    notice("Wrote model to %s", outf.c_str());
    if (write_se) {
        for (uint32_t flag = 1; flag <= 2; flag++) {
            if ((se_method & flag) == 0) {
                continue;
            }
            outf = outPrefix + (flag == 1 ? ".model.se.fisher.tsv" : ".model.se.robust.tsv");
            const auto& se_mat = nmf.get_se(flag == 2);
            write_matrix_to_file(outf, se_mat, 4, false, reader.features, "Feature");
            notice("Wrote standard errors of beta to %s", outf.c_str());
        }
    }
    if (compute_residual) {
        const auto& resids = nmf.get_feature_residuals();
        const auto& sums = nmf.get_feature_sums();
        outf = outPrefix + ".feature.residuals.tsv";
        std::ofstream ofs(outf);
        if (!ofs) {
            error("Cannot open output file %s", outf.c_str());
        }
        ofs << "Feature\tTotalCount\tResidual\n";
        for (int32_t i = 0; i < M; i++) {
            ofs << reader.features[i] << "\t"
             << std::setprecision(0) << sums[i] << "\t"
            << std::setprecision(4) << std::fixed << resids[i]/N << "\n";
        }
        ofs.close();
        notice("Wrote per-feature averaged residuals to %s", outf.c_str());
    }

    if (n_covar > 0) {
        const auto& bcov = nmf.get_covar_coef(); // M x P
        std::string outf = outPrefix + ".covar.tsv";
        write_matrix_to_file(outf, bcov, 6, false, reader.features, "Feature", &covar_names);
        notice("Wrote covariate coefficients to %s", outf.c_str());
    }

    if (!transform) {
        return 0;
    }

    opts.compute_residual = true;
    opts.se_flag = 0;

    std::vector<MLEStats> stats;
    RowMajorMatrixXd theta = nmf.transform(docs, opts, stats);

    outf = outPrefix + ".theta.tsv";
    write_matrix_to_file(outf, theta, 4, false, rnames, "Index");
    notice("Wrote theta to %s", outf.c_str());

    outf = outPrefix + ".fit_stats.tsv";
    std::ofstream ofs(outf);
    if (!ofs) {
        error("Cannot open output file %s", outf.c_str());
    }
    ofs << "Index\tTotalCount\tll\tResidual\n";
    for (size_t i = 0; i < N; i++) {
        ofs << rnames[i] << "\t"
            << std::setprecision(2) << std::fixed << docs[i].ct_tot << "\t"
            << std::setprecision(4) << stats[i].pll << "\t"
            << std::setprecision(4) << stats[i].residual << "\n";
    }
    ofs.close();
    notice("Wrote goodness of fit statistics to %s", outf.c_str());

    return 0;
}
