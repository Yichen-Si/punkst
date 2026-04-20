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
    std::string modelFile, labelListFile;
    std::string dge_dir, in_bc, in_ft, in_mtx;
    std::string include_ftr_regex, exclude_ftr_regex;
    std::vector<uint32_t> covar_idx;
    int32_t label_idx = -1;
    std::string label_na = "";
    int32_t K;
    int32_t seed = -1, nThreads = 1, debug_ = 0, debug_N = 0, verbose = 500000;
    int32_t minCountTrain = 50, minCountFeature = 100;
    double covar_coef_min = -1e6, covar_coef_max = 1e6;
    double size_factor = 10000, c = -1;
    bool allow_na = false;
    bool exact = false;
    bool write_se = false;
    bool test_beta_vs_null = false;
    bool fit_stats = false;
    bool random_init_missing_features = false;
    int32_t se_method = 1; // 1: fisher, 2: robust, 3: both
    double min_ct_de = 100, min_fc = 1.5, max_p = 0.05;
    bool fit_background = false;
    bool fix_background = false;
    double pi0 = 0.1;
    int32_t mode = 1;
    NmfFitOptions nmf_opts;
    nmf_opts.max_iter = 20;
    nmf_opts.tol = 1e-4;
    nmf_opts.n_mb_epoch = 1;
    MLEOptions opts;
    opts.optim.max_iters = 50;
    opts.optim.tol = 1e-6;

    ParamList pl;
    // Input Options
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("in-covar", "Covariate file", covarFile)
      .add_option("allow-na", "Replace non-numerical values in covariates with zero", allow_na)
      .add_option("icol-covar", "Column indices (0-based) in --in-covar to use", covar_idx)
      .add_option("icol-label", "Column index (0-based) in --in-covar for labels", label_idx)
      .add_option("label-list", "List of unique labels", labelListFile)
      .add_option("label-na", "String for missing labels", label_na)
      .add_option("in-model", "Input model (beta) file", modelFile)
      .add_option("random-init-missing", "Randomly initialize features missing from the model", random_init_missing_features);
    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dir)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx);
	pl.add_option("K", "K", K, true)
      .add_option("mode", "Algorithm", mode)
      .add_option("fit-background", "Fit background noise", fit_background)
      .add_option("fix-background", "Fix background model during training", fix_background)
      .add_option("background-init", "Initial background proportion pi0", pi0)
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
    pl.add_option("features", "Feature list", featureFile)
	  .add_option("min-count-per-feature", "Min count for features to be included", minCountFeature)
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
      .add_option("fit-stats", "Compute goodness of fit statistics", fit_stats)
      .add_option("verbose", "Verbose", verbose)
      .add_option("debug-N", "Debug with the first N units", debug_N)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }
    if (size_factor <= 0) {
        error("Error: --size-factor must be positive (Seurat uses 10000)");
    }
    bool per_doc_c = c <= 0;
    std::mt19937 rng(seed > 0 ? seed : std::random_device{}());

    enum class TenXFeatureMode {
        Default,
        FeatureFile,
        ModelOnly,
        RegexOnly,
        PostloadCounts
    };

    std::unique_ptr<DGEReader10X> dge_ptr;
    HexReader reader;
    bool use_10x = !dge_dir.empty() || !in_bc.empty() || !in_ft.empty() || !in_mtx.empty();
    if (use_10x) {
        if (!inFile.empty()) {
            warning("Both --in-data and 10X inputs are provided; using 10X inputs and ignoring --in-data");
        }
        if (!dge_dir.empty() && (in_bc.empty() || in_ft.empty() || in_mtx.empty())) {
            if (dge_dir.back() == '/') {
                dge_dir.pop_back();
            }
            in_bc = dge_dir + "/barcodes.tsv.gz";
            in_ft = dge_dir + "/features.tsv.gz";
            in_mtx = dge_dir + "/matrix.mtx.gz";
        }
        if (in_bc.empty() || in_ft.empty() || in_mtx.empty()) {
            error("Missing required 10X inputs (--in-barcodes, --in-features, --in-matrix)");
        }
        dge_ptr = std::make_unique<DGEReader10X>(in_bc, in_ft, in_mtx);
        reader.initFromFeatures(dge_ptr->features, dge_ptr->nBarcodes);
    } else {
        if (metaFile.empty() || inFile.empty()) {
            error("Missing --in-data or --in-meta");
        }
        reader.readMetadata(metaFile);
    }

    TenXFeatureMode tenx_feature_mode = TenXFeatureMode::Default;
    if (use_10x) {
        const bool has_model_file = !modelFile.empty();
        if (!featureFile.empty()) {
            tenx_feature_mode = TenXFeatureMode::FeatureFile;
            reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex, true);
        } else if (has_model_file) {
            tenx_feature_mode = TenXFeatureMode::ModelOnly;
        } else if (minCountFeature <= 1) {
            tenx_feature_mode = TenXFeatureMode::RegexOnly;
            if (!include_ftr_regex.empty() || !exclude_ftr_regex.empty()) {
                reader.filterCurrentFeatures(1, include_ftr_regex, exclude_ftr_regex);
            }
        } else {
            tenx_feature_mode = TenXFeatureMode::PostloadCounts;
        }
    } else if (!featureFile.empty()) {
        reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
    }

    RowMajorMatrixXd beta_init;
    bool has_beta_init = false;
    std::vector<std::string> factor_names;
    if (!modelFile.empty()) {
        RowMajorMatrixXd beta;
        std::vector<std::string> model_features;
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
        int32_t M = reader.nFeatures;
        beta_init = RowMajorMatrixXd::Zero(M, K);
        beta_init.topRows(M1) = beta.topRows(M1);
        if (M != M1) {
            auto colmed = columnMedians(beta_init.topRows(M1));
            std::gamma_distribution<double> dist(100.0, 0.01);
            for (int32_t i = M1; i < M; i++) {
                for (int k = 0; k < K; k++) {
                    beta_init(i, k) = dist(rng) * colmed(k);
                }
            }
            notice("Filled in %d missing features with random initial values", M - M1);
        }
        has_beta_init = true;
    }

    int32_t M = reader.nFeatures;
    notice("Number of features: %d", M);

    // Load data
    // TODO: data loading with covariates is unsafe & messy
    std::vector<SparseObs> docs;
    std::vector<std::string> rnames, covar_names, labels;
    size_t N = 0;
    int32_t n_covar = 0;
    if (use_10x) {
        if (tenx_feature_mode == TenXFeatureMode::ModelOnly &&
            ((minCountFeature > 1) || !include_ftr_regex.empty() || !exclude_ftr_regex.empty())) {
            warning("Ignoring --min-count-per-feature and feature regex filters for 10X input because the input model defines the feature space");
        }

        std::vector<Document> dge_docs;
        std::vector<int32_t> dge_barcode_idx;
        auto reload_10x_docs = [&]() {
            int32_t n_overlap = dge_ptr->setFeatureIndexRemap(reader.features, false);
            if (n_overlap == 0) {
                error("No overlapping features found between 10X input and the configured feature space");
            }
            dge_ptr->readAll(dge_docs, dge_barcode_idx, 0);
            std::vector<double> feature_sums_raw(dge_ptr->feature_totals.begin(), dge_ptr->feature_totals.end());
            reader.setFeatureSums(feature_sums_raw, true);
            notice("Loaded %zu 10X units", dge_docs.size());
        };

        reload_10x_docs();
        if (tenx_feature_mode == TenXFeatureMode::PostloadCounts) {
            const int32_t nFeaturesPrev = reader.nFeatures;
            const int32_t nKept = reader.filterCurrentFeatures(minCountFeature, include_ftr_regex, exclude_ftr_regex);
            if (nKept == 0) {
                error("No features remain after applying feature filters");
            }
            if (reader.nFeatures != nFeaturesPrev) {
                reload_10x_docs();
            }
        }

        if (featureFile.empty()) {
            const std::string outFeatures = outPrefix + ".features.tsv";
            std::ofstream outFeatureStream(outFeatures);
            if (!outFeatureStream) {
                error("Error opening output file: %s for writing", outFeatures.c_str());
            }
            const auto& featureSums = reader.getFeatureSumsRaw();
            if (reader.features.size() != featureSums.size()) {
                error("Feature names and total counts have inconsistent sizes (%zu vs %zu)",
                    reader.features.size(), featureSums.size());
            }
            outFeatureStream << std::fixed << std::setprecision(0);
            for (size_t i = 0; i < reader.features.size(); ++i) {
                outFeatureStream << reader.features[i] << "\t" << featureSums[i] << "\n";
            }
            outFeatureStream.close();
            notice("Features and total counts written to %s", outFeatures.c_str());
        }

        std::ifstream covarStream;
        bool has_covar = false;
        bool has_labels = false;
        int32_t n_tokens = 0;
        if (!covarFile.empty() || label_idx >= 0) {
            if (covarFile.empty()) {
                error("Label index requires a non-empty covariate file");
            }
            covarStream.open(covarFile);
            if (!covarStream) {
                error("Fail to covariate file: %s", covarFile.c_str());
            }
            std::string line;
            if (!std::getline(covarStream, line)) {
                error("Fail to parse covariate file: %s", covarFile.c_str());
            }
            std::vector<std::string> covar_header;
            split(covar_header, "\t ", line, UINT_MAX, true, true, true);
            n_tokens = static_cast<int32_t>(covar_header.size());
            has_labels = (label_idx >= 0);
            if (covar_idx.empty() && !has_labels) {
                n_covar = n_tokens - 1;
                for (int32_t i = 1; i < n_tokens; i++) {
                    covar_idx.push_back(i);
                    covar_names.push_back(covar_header[i]);
                }
            } else {
                n_covar = static_cast<int32_t>(covar_idx.size());
                for (const auto i : covar_idx) {
                    if (i >= static_cast<uint32_t>(n_tokens)) {
                        error("Covariate index %d is out of range [0,%d)", static_cast<int32_t>(i), n_tokens);
                    }
                    covar_names.push_back(covar_header[i]);
                }
            }
            has_covar = (n_covar > 0 || has_labels);
            notice("Covariate file has %d columns, using %d as covariates and %d as label", n_tokens, n_covar, (int32_t) has_labels);
        }

        const size_t n_docs_raw = (debug_N > 0 && debug_N < static_cast<int32_t>(dge_docs.size()))
            ? static_cast<size_t>(debug_N) : dge_docs.size();
        docs.reserve(n_docs_raw);
        rnames.reserve(n_docs_raw);
        std::vector<std::string> tokens;
        for (size_t i = 0; i < n_docs_raw; ++i) {
            std::string covar_line;
            if (has_covar) {
                if (!std::getline(covarStream, covar_line)) {
                    error("The number of lines in covariate file is less than that in data file");
                }
            }

            SparseObs obs;
            obs.doc = std::move(dge_docs[i]);
            const double ct = obs.doc.get_sum();
            if (ct < minCountTrain) {
                continue;
            }
            obs.c = per_doc_c ? ct / size_factor : c;
            obs.ct_tot = ct;

            if (has_covar) {
                split(tokens, "\t ", covar_line, UINT_MAX, true, true, true);
                if (tokens.size() != static_cast<size_t>(n_tokens)) {
                    error("Number of columns (%lu) of line [%s] in covariate file does not match header (%d)",
                        tokens.size(), covar_line.c_str(), n_tokens);
                }
                if (n_covar > 0) {
                    obs.covar = Eigen::VectorXd::Zero(n_covar);
                    for (int32_t j = 0; j < n_covar; j++) {
                        if (!str2double(tokens[covar_idx[j]], obs.covar(j))) {
                            if (!allow_na) {
                                error("Invalid value for %d-th covariate %s.", j+1, tokens[covar_idx[j]].c_str());
                            }
                            obs.covar(j) = 0;
                        }
                    }
                }
                if (has_labels) {
                    if (tokens[label_idx] == label_na) {
                        continue;
                    }
                    labels.push_back(tokens[label_idx]);
                }
            }

            docs.push_back(std::move(obs));
            if (i < dge_barcode_idx.size() &&
                dge_barcode_idx[i] >= 0 &&
                dge_barcode_idx[i] < dge_ptr->nBarcodes) {
                rnames.push_back(dge_ptr->barcodes[dge_barcode_idx[i]]);
            } else {
                rnames.push_back(std::to_string(i));
            }
        }
        N = docs.size();
    } else {
        SparseObsMinibatchReader minibatch_reader(inFile, reader,
            minCountTrain, size_factor, c, debug_N);
        if (!covarFile.empty() || label_idx >= 0) {
            minibatch_reader.set_covariates(covarFile, &covar_idx, &covar_names,
                allow_na, label_idx, label_na, &labels);
        }
        N = minibatch_reader.readAll(docs, rnames);
        n_covar = static_cast<int32_t>(covar_idx.size());
    }
    M = reader.nFeatures;
    notice("Read %lu units with %d features", N, M);

    PoissonLog1pNMF nmf(K, M, nThreads, size_factor, seed, exact, debug_);
    if (fit_background) {
        nmf.set_background_model(pi0, nullptr, fix_background);
    }
    if (has_beta_init) {
        nmf.set_beta(beta_init);
    }

    std::vector<int32_t> labels_idx;
    if (labels.size() > 0) {
        if (labels.size() != N) {
            error("Number of labels (%lu) does not match number of units (%lu)", labels.size(), N);
        }
        std::unordered_map<std::string, int32_t> label_to_idx;
        if (!modelFile.empty()) {
            for (int32_t i = 0; i < K; i++) {
                label_to_idx[factor_names[i]] = i;
            }
        } else if (!labelListFile.empty()) {
            std::ifstream lstream(labelListFile);
            if (!lstream) {
                error("Failed to open file %s", labelListFile.c_str());
            }
            std::string label;
            int32_t idx = 0;
            while (lstream >> label) {
                label_to_idx[label] = idx++;
                lstream.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }
            if (idx != K) {
                error("Number of labels (%d) does not match specified K=%d", idx, K);
            }
            factor_names.resize(K);
            for (const auto& p : label_to_idx) {
                factor_names[p.second] = p.first;
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
            ofs << "Feature\tFactor\tDiff\tlog10pval\tApproxFC\n";
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
    if (fit_background) {
        const VectorXd& beta0 = nmf.get_bg_model();
        double pi = nmf.get_pi();
        outf = outPrefix + ".background.tsv";
        std::ofstream ofs(outf);
        if (!ofs) {
            error("Cannot open output file %s", outf.c_str());
        }
        ofs << "##pi=" << std::scientific << std::setprecision(4) << pi << "\n";
        ofs << "#Feature\tBackground\n";
        for (int32_t m = 0; m < M; m++) {
            ofs << reader.features[m] << "\t" << beta0(m) << "\n";
        }
        ofs.close();
        notice("Wrote background model to %s", outf.c_str());

        const std::vector<double>& phi = nmf.get_bg_proportions();
        outf = outPrefix + ".bgprob.tsv";
        ofs.open(outf);
        if (!ofs) {
            error("Cannot open output file %s", outf.c_str());
        }
        ofs << "#Index\tBackground\n" << std::fixed << std::setprecision(4);
        for (size_t i = 0; i < N; i++) {
            ofs << rnames[i] << "\t" << phi[i] << "\n";
        }
        ofs.close();
    }

    if (opts.optim.max_iters > 0) {
        const auto& theta_ = nmf.get_theta();
        outf = outPrefix + ".theta.tsv";
        write_matrix_to_file(outf, theta_, 4, false, rnames, "#Index");
        notice("Wrote theta to %s", outf.c_str());
    }

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

    // TODO: implement goodness of fit stats when background is fitted
    if (fit_background || !fit_stats) {
        return 0;
    }

    opts.compute_residual = true;
    opts.compute_var_mu = true;
    opts.se_flag = se_method > 1 ? 2 : 1;

    std::vector<MLEStats> stats;
    RowMajorMatrixXd theta = nmf.transform(docs, opts, stats);

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
