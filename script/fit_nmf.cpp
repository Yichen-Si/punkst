#include <iostream>
#include <iomanip>
#include <random>
#include "punkst.h"
#include "utils.h"
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include "poisnmf.hpp"

int32_t cmdNmfPoisLog1p(int32_t argc, char** argv) {

	std::string inFile, metaFile, featureFile, outPrefix;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t seed = -1, nThreads = 1, debug_ = 0, verbose = 500000;
    int32_t minCountTrain = 50, minCountFeature = 100;
    int32_t K;
    int32_t max_iter_outer = 50, max_iter_inner = 20;
    double tol_outer = 1e-5, tol_inner = 1e-6;
    double c = 0.01;
    int32_t mode = 0;

    ParamList pl;
    // Input Options
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true);
    pl.add_option("K", "K", K, true)
      .add_option("mode", "Algorithm", mode)
      .add_option("c", "Constant c", c)
      .add_option("max-iter-outer", "Maximum outer iterations", max_iter_outer)
      .add_option("max-iter-inner", "Maximum inner iterations", max_iter_inner)
      .add_option("tol-outer", "Outer tolerance", tol_outer)
      .add_option("tol-inner", "Inner tolerance", tol_inner)
      .add_option("seed", "Random seed", seed)
      .add_option("threads", "Number of threads", nThreads);
    pl.add_option("features", "Feature names and total counts file", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", minCountFeature)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex)
      .add_option("min-count-train", "Minimum total count per doc", minCountTrain);
    // Output Options
    pl.add_option("out-prefix", "Output prefix", outPrefix, true)
      .add_option("verbose", "Verbose", verbose)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }


    HexReader reader(metaFile);
    if (!featureFile.empty()) {
        reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
    }
    int32_t M = reader.nFeatures;

    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("%s: Error opening input file: %s", __func__, inFile.c_str());
    }
    std::vector<Document> docs;
    std::vector<std::string> rnames;
    std::string line, tmp;
    int32_t idx = 0;
    while (std::getline(inFileStream, line)) {
        idx++;
        Document doc;
        int32_t ct = reader.parseLine(doc, tmp, line);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        if (ct < minCountTrain) {
            continue;
        }
        docs.emplace_back(doc);
        rnames.emplace_back(std::to_string(idx-1));
        if (debug_ > 0 && idx > debug_) {
            break;
        }
    }
    inFileStream.close();
    notice("Read %lu documents with %d features", docs.size(), M);

    PoissonLog1pNMF nmf(K, M, c, nThreads, seed);

    // Set up MLE options (for subproblems)
    MLEOptions opts{};
    opts.max_iters = max_iter_inner;
    opts.tol = tol_inner;
    std::string model_name = "CD";
    if (mode == 1) {
        opts.tron.enabled = true;
        model_name = "TRON";
    } else if (mode == 2) {
        opts.acg.enabled = true;
        model_name = "FISTA";
    } else if (mode == 3) {
        opts.ls.enabled = true;
        model_name = "DiagLS";
    }
    outPrefix += "." + model_name;

    // Fit the model
    auto t0 = std::chrono::steady_clock::now();
    nmf.fit(docs, opts, max_iter_outer, tol_outer);
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double, std::milli>(t1 - t0).count() / 1000.0;
    int32_t minutes = static_cast<int32_t>(sec / 60.0);
    notice("Approximate %s took %.3f seconds (%dmin %.2fs)", model_name.c_str(), sec, minutes, sec - minutes * 60);

    std::string outf;
    outf = outPrefix + ".model.tsv";
    std::vector<std::string> vocab = reader.features;
    const auto& mat = nmf.get_model();
    write_matrix_to_file(outf, mat, 4, false, vocab, "Feature");
    notice("Wrote model to %s", outf.c_str());

    outf = outPrefix + ".theta.tsv";
    const auto& theta = nmf.get_theta();
    write_matrix_to_file(outf, theta, 4, false, rnames, "Index");
    notice("Wrote theta to %s", outf.c_str());

    outf = outPrefix + ".loadings.tsv";
    auto loadings = nmf.convert_to_factor_loading();
    write_matrix_to_file(outf, loadings, 4, true, rnames, "Index");
    notice("Wrote scaled factor loadings to %s", outf.c_str());

    return 0;
}
