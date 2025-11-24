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
    std::string modelFile;
    std::string include_ftr_regex, exclude_ftr_regex;
    std::vector<uint32_t> covar_idx;
    int32_t label_idx = -1;
    int32_t K;
    int32_t seed = -1, nThreads = 1, debug_ = 0, debug_N = 0, verbose = 500000;
    int32_t minCountTrain = 50, minCountFeature = 100;
    double covar_coef_min = -1e6, covar_coef_max = 1e6;
    double size_factor = 10000, c = -1;
    bool allow_na = false;
    bool exact = false;
    bool write_se = false;
    bool test_beta_vs_null = false;
    bool transform = false;
    bool random_init_missing_features = false;
    int32_t se_method = 1; // 1: fisher, 2: robust, 3: both
    double min_ct_de = 100, min_fc = 1.5, max_p = 0.05;
    int32_t mode = 1;
    NmfFitOptions nmf_opts;
    nmf_opts.max_iter = 20;
    nmf_opts.tol = 1e-4;
    MLEOptions opts;
    opts.optim.max_iters = 50;
    opts.optim.tol = 1e-6;

    ParamList pl;
    // Input Options
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true)
      .add_option("in-covar", "Covariate file", covarFile)
      .add_option("allow-na", "Replace non-numerical values in covariates with zero", allow_na)
      .add_option("icol-covar", "Column indices (0-based) in --in-covar to use", covar_idx)
      .add_option("icol-label", "Column index (0-based) in --in-covar for labels", label_idx)
      .add_option("in-model", "Input model (beta) file", modelFile)
      .add_option("random-init-missing", "Randomly initialize features missing from the model", random_init_missing_features);
    pl.add_option("K", "K", K, true)
      .add_option("mode", "Algorithm", mode)
      .add_option("c", "Constant c in log(1+lambda/c)", c)
      .add_option("size-factor", "L: c_i=y_i/L, g()=log(1+lambda/c_i)", size_factor)
      .add_option("max-iter-outer", "Maximum outer iterations", nmf_opts.max_iter)
      .add_option("tol-outer", "Outer tolerance", nmf_opts.tol)
      .add_option("minibatch-epoch", "Number of minibatch epochs at the beginning", nmf_opts.n_mb_epoch)
      .add_option("minibatch-size", "Minibatch size", nmf_opts.batch_size)
      .add_option("t0", "Decay parameter t0 for minibatch", nmf_opts.t0)
      .add_option("kappa", "Decay parameter kappa for minibatch", nmf_opts.kappa)
      .add_option("max-iter-inner", "Maximum inner iterations", opts.optim.max_iters)
      .add_option("tol-inner", "Inner tolerance", opts.optim.tol)
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
      .add_option("feature-residuals", "Compute and write per-feature residuals", opts.compute_residual)
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
    std::mt19937 rng(seed > 0 ? seed : std::random_device{}());

    HexReader reader(metaFile);
    if (!featureFile.empty()) {
        reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
    }
    int32_t M = reader.nFeatures;
    notice("Number of features: %d", M);

    PoissonLog1pNMF nmf(K, M, nThreads, seed, exact, debug_);

    // Load model / warm start
    std::vector<std::string> factor_names;
    if (!modelFile.empty()) {
        RowMajorMatrixXd beta;
        std::vector<std::string> model_features, covar_names_from_file;
        read_matrix_from_file(modelFile, beta, &model_features, &factor_names);
        int32_t K1 = beta.cols();
        if (K1 != K) {
            error("Model has %d factors, but K=%d is specified.", K1, K);
        }
        int32_t M1 = beta.rows();
        notice("Loaded model with %d features and %d factors", M1, K);

        std::unordered_map<std::string, uint32_t> data_features;
        std::vector<std::string> kept_model_features;
        reader.featureDict(data_features);
        int32_t m = 0;
        for (uint32_t i = 0; i < model_features.size(); i++) {
            auto it = data_features.find(model_features[i]);
            if (it != data_features.end()) {
                kept_model_features.push_back(model_features[i]);
                beta.row(m) = beta.row(i);
                m++;
            }
        }
        M1 = m;
        bool keep_unmapped = !featureFile.empty() && random_init_missing_features;
        reader.setFeatureIndexRemap(kept_model_features, keep_unmapped);
        M = reader.nFeatures;
        RowMajorMatrixXd beta1 = RowMajorMatrixXd::Zero(M, K);
        beta1.topRows(M1) = beta.topRows(M1);
        if (M != M1) {
            auto colmed = columnMedians(beta1.topRows(M1));
            std::gamma_distribution<double> dist(100.0, 0.01);
            for (int32_t i = M1; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    beta1(i, k) = dist(rng) * colmed(k);
                }
            }
            notice("Filled in %d missing features with random initial values", M - M1);
        }
        nmf.set_beta(beta1);
    }

    // Load data
    std::vector<SparseObs> docs;
    std::vector<std::string> rnames, covar_names, labels;
    size_t N = read_sparse_obs(inFile, reader, docs,
        rnames, minCountTrain, size_factor, c,
        &covarFile, &covar_idx, &covar_names,
        allow_na, label_idx, &labels, debug_N);
    int32_t n_covar = static_cast<int32_t>(covar_idx.size());
    notice("Read %lu documents with %d features", N, M);

    std::vector<int32_t> labels_idx;
    if (labels.size() > 0) {
        if (labels.size() != N) {
            error("Number of labels (%lu) does not match number of units (%lu)", labels.size(), N);
        }
        std::unordered_map<std::string, int32_t> label_to_idx;
        std::vector<int32_t> label_counts(K, 0);
        if (!modelFile.empty()) {
            for (int32_t i = 0; i < K; i++) {
                label_to_idx[factor_names[i]] = i;
            }
        } else {
            std::unordered_map<std::string, int32_t> label_to_ct;
            for (const auto& lab : labels) {
                auto it = label_to_ct.find(lab);
                if (it != label_to_ct.end()) {
                    it->second++;
                } else {
                    label_to_ct[lab] = 1;
                }
            }
            if (label_to_ct.size() > K) {
                warning("Number of unique labels (%lu) is greater than K=%d; only the top K labels by counts will be used.",
                    label_to_ct.size(), K);
                // pick the K labels with the highest counts
                std::vector<std::pair<std::string, int32_t>> lab_ct_vec(label_to_ct.begin(), label_to_ct.end());
                std::sort(lab_ct_vec.begin(), lab_ct_vec.end(),
                    [](const auto& a, const auto& b) { return a.second > b.second; });
                for (int32_t i = 0; i < K; i++) {
                    label_to_idx[lab_ct_vec[i].first] = i;
                }
            } else {
                int32_t idx = 0;
                for (const auto& p : label_to_ct) {
                    label_to_idx[p.first] = idx++;
                }
            }
            factor_names.resize(K);
            for (const auto& p : label_to_idx) {
                factor_names[p.second] = p.first;
            }
        }
        int32_t skipped = 0;
        for (const auto& lab : labels) {
            auto it = label_to_idx.find(lab);
            if (it != label_to_idx.end()) {
                labels_idx.push_back(it->second);
                label_counts[it->second]++;
            } else {
                labels_idx.push_back(-1);
                skipped++;
            }
        }
        notice("Parsed labels for %lu units, skipped %d undefined labels", N, skipped);
    }

    // Set up MLE options (for subproblems)
    if (mode == 1) {
        opts.optim.tron.enabled = true;
    } else if (mode == 2) {
        opts.optim.acg.enabled = true;
    } else if (mode == 3) {
        opts.optim.ls.enabled = true;
    }
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

    // Fit the model
    std::vector<int32_t>* labels_ptr = nullptr;
    if (labels_idx.size() == N) {
        labels_ptr = &labels_idx;
    }
    nmf.fit(docs, opts, nmf_opts, false, labels_ptr);

    // Compute DE stats
    if (test_beta_vs_null) {
        for (uint32_t flag = 1; flag <= 2; flag++) {
            if ((se_method & flag) == 0) {
                continue;
            }
            auto results = nmf.test_beta_vs_null(flag);
            std::string outf = outPrefix + (flag == 1 ? ".de.fisher.tsv" : ".de.robust.tsv");
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

    // Write output
    std::string outf;
    outf = outPrefix + ".model.tsv";
    std::vector<std::string>* cnames_ptr = nullptr;
    if ((int32_t) factor_names.size() == K) {
        cnames_ptr = &factor_names;
    }
    const auto& mat = nmf.get_model();
    write_matrix_to_file(outf, mat, 4, false, reader.features, "Feature", cnames_ptr);
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
    if (opts.compute_residual) {
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
            << std::setprecision(0) << std::fixed << sums[i] << "\t"
            << std::setprecision(4) << std::defaultfloat << resids[i]/N << "\n";
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
    opts.compute_var_mu = true;
    opts.se_flag = se_method > 1 ? 2 : 1;

    std::vector<MLEStats> stats;
    RowMajorMatrixXd theta = nmf.transform(docs, opts, stats);

    outf = outPrefix + ".theta.tsv";
    write_matrix_to_file(outf, theta, 4, false, rnames, "#Index");
    notice("Wrote theta to %s", outf.c_str());

    outf = outPrefix + ".fit_stats.tsv";
    std::ofstream ofs(outf);
    if (!ofs) {
        error("Cannot open output file %s", outf.c_str());
    }
    ofs << "#Index\tTotalCount\tll\tResidual\tVarMu\n";
    for (size_t i = 0; i < N; i++) {
        ofs << rnames[i] << "\t"
            << std::setprecision(2) << std::fixed << docs[i].ct_tot << "\t"
            << std::setprecision(4) << stats[i].pll << "\t"
            << std::setprecision(4) << stats[i].residual << "\t"
            << std::setprecision(4) << stats[i].var_mu << "\n";
    }
    ofs.close();
    notice("Wrote goodness of fit statistics to %s", outf.c_str());

    return 0;
}
