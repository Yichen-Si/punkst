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
    std::vector<std::string> dge_dirs, in_bc, in_ft, in_mtx, dataset_ids;
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
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("in-covar", "Covariate file", covarFile)
      .add_option("allow-na", "Replace non-numerical values in covariates with zero", allow_na)
      .add_option("icol-covar", "Column indices (0-based) in --in-covar to use", covar_idx)
      .add_option("in-model", "Input model (beta) file", modelFile, true)
      .add_option("in-covar-coef", "Input covariate coefficients (Bcov) file", covarCoefFile);
    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", dataset_ids);
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
        pl.print_help_noexit();
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

    PoissonLog1pNMF nmf(K, M, nThreads, size_factor, seed, exact, debug_);
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
    std::vector<SparseObs> docs;
    std::vector<std::string> rnames, covar_names;
    const auto dge_inputs = resolveDge10XInputs(dge_dirs, in_bc, in_ft, in_mtx, dataset_ids);
    const bool use_10x = !dge_inputs.empty();
    HexReader reader;
    int32_t N = 0;
    int32_t n_covar = 0;
    if (use_10x) {
        if (!inFile.empty()) {
            warning("Both --in-data and 10X inputs are provided; using 10X inputs and ignoring --in-data");
        }
        std::unique_ptr<DGEReader10X> dge_ptr;
        if (!in_bc.empty() || !in_ft.empty() || !in_mtx.empty()) {
            dge_ptr = std::make_unique<DGEReader10X>(in_bc, in_ft, in_mtx, dataset_ids);
        } else {
            dge_ptr = std::make_unique<DGEReader10X>(dge_dirs, dataset_ids);
        }
        reader.initFromFeatures(dge_ptr->features, dge_ptr->nBarcodes);
        reader.setFeatureIndexRemap(model_features);

        int32_t n_overlap = dge_ptr->setFeatureIndexRemap(reader.features, false);
        if (n_overlap == 0) {
            error("No overlapping features found between 10X input and the configured feature space");
        }

        std::vector<Document> dge_docs;
        std::vector<int32_t> dge_unit_idx;
        dge_ptr->readAll(dge_docs, dge_unit_idx, 0);
        std::vector<double> feature_sums_raw(dge_ptr->feature_totals.begin(), dge_ptr->feature_totals.end());
        reader.setFeatureSums(feature_sums_raw, true);

        std::ifstream covarStream;
        bool has_covar = false;
        int32_t n_tokens = 0;
        if (!covarFile.empty()) {
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
            if (covar_idx.empty()) {
                n_covar = n_tokens - 1;
                for (int32_t i = 1; i < n_tokens; ++i) {
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
            has_covar = (n_covar > 0);
            notice("Covariate file has %d columns, using %d as covariates", n_tokens, n_covar);
        }

        const size_t n_docs_raw = (debug_N > 0 && debug_N < static_cast<int32_t>(dge_docs.size()))
            ? static_cast<size_t>(debug_N) : dge_docs.size();
        docs.reserve(n_docs_raw);
        rnames.reserve(n_docs_raw);
        std::vector<std::string> tokens_local;
        for (size_t i = 0; i < n_docs_raw; ++i) {
            std::string covar_line;
            if (has_covar && !std::getline(covarStream, covar_line)) {
                error("The number of lines in covariate file is less than that in data file");
            }

            SparseObs obs;
            obs.doc = std::move(dge_docs[i]);
            const double raw_ct = (obs.doc.raw_ct_tot >= 0.0) ? obs.doc.raw_ct_tot : obs.doc.get_sum();
            if (raw_ct < minCount) {
                continue;
            }
            obs.c = c <= 0 ? raw_ct / size_factor : c;
            obs.ct_tot = raw_ct;

            if (has_covar) {
                split(tokens_local, "\t ", covar_line, UINT_MAX, true, true, true);
                if (tokens_local.size() != static_cast<size_t>(n_tokens)) {
                    error("Number of columns (%lu) of line [%s] in covariate file does not match header (%d)",
                        tokens_local.size(), covar_line.c_str(), n_tokens);
                }
                obs.covar = Eigen::VectorXd::Zero(n_covar);
                for (int32_t j = 0; j < n_covar; ++j) {
                    if (!str2double(tokens_local[covar_idx[j]], obs.covar(j))) {
                        if (!allow_na) {
                            error("Invalid value for %d-th covariate %s.", j + 1, tokens_local[covar_idx[j]].c_str());
                        }
                        obs.covar(j) = 0;
                    }
                }
            }

            docs.push_back(std::move(obs));
            if (i < dge_unit_idx.size() &&
                dge_unit_idx[i] >= 0 &&
                dge_unit_idx[i] < dge_ptr->nBarcodes) {
                rnames.push_back(dge_ptr->barcodes[dge_unit_idx[i]]);
            } else {
                rnames.push_back(std::to_string(i));
            }
        }
        N = static_cast<int32_t>(docs.size());
    } else {
        if (metaFile.empty() || inFile.empty()) {
            error("Missing --in-data or --in-meta");
        }
        reader = HexReader(metaFile);
        reader.setFeatureIndexRemap(model_features);
        SparseObsMinibatchReader minibatch_reader(inFile, reader,
            minCount, size_factor, c, debug_N);
        if (!covarFile.empty()) {
            minibatch_reader.set_covariates(covarFile, &covar_idx, &covar_names,
                allow_na, -1, "", nullptr);
        }
        N = minibatch_reader.readAll(docs, rnames);
        n_covar = static_cast<int32_t>(covar_idx.size());
    }
    notice("Read %d documents with %d features", N, M);

    // --- 3. Transform Data ---
    std::vector<MLEStats> stats;
    ArrayXd resids;
    RowMajorMatrixXd theta = nmf.transform(docs, opts, stats, &resids);
    const MatrixXd similarity = pairwiseCosineSimilarityRows(rowNormalize(beta.transpose()));
    const ThetaEntropyStats thetaStats = computeThetaEntropyStats(theta, similarity);

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
    write_matrix_to_file(outf, theta, 4, false, rnames, "#Index");
    notice("Wrote theta to %s", outf.c_str());

    outf = outPrefix + ".unit_stats.tsv";
    ofs.open(outf);
    if (!ofs) {
        error("Cannot open output file %s", outf.c_str());
    }
    ofs << "#Index\ttotal_count\tresidual\tll\tvar_mu\tentropy\tsh_lcr\tsh_q\n";
    for (size_t i = 0; i < N; i++) {
        ofs << rnames[i] << "\t"
            << std::setprecision(2) << std::fixed << docs[i].ct_tot << "\t"
            << std::setprecision(4) << stats[i].residual << "\t"
            << std::setprecision(4) << std::defaultfloat << stats[i].pll << "\t"
            << std::setprecision(4) << stats[i].var_mu << "\t"
            << std::setprecision(4) << thetaStats.entropy(i) << "\t"
            << std::setprecision(4) << thetaStats.sh_lcr(i) << "\t"
            << std::setprecision(4) << thetaStats.sh_q(i) << "\n";
    }
    ofs.close();
    notice("Wrote goodness of fit statistics to %s", outf.c_str());

    return 0;
}
