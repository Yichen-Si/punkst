#include "cndtest.hpp"

int32_t cmdMultiConditionTest(int argc, char** argv) {

    std::string inFile, metaFile, labelFile, outPrefix, modelFile;
    int32_t nThreads = 0;
    int32_t seed = -1;
    int32_t nFold, nContrast;
    int32_t nSplit = 1;
    int32_t batchSize = 1024;
    int32_t maxIter = 100;
    double  mDelta = 1e-3;
    int32_t debug = 0, verbose = 10000;


    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true)
      .add_option("in-labels", "Labels file", labelFile, true)
      .add_option("model", "Model file", modelFile, true)
      .add_option("out-prefix", "Output prefix for model and results files", outPrefix, true)
      .add_option("n-fold", "Number of folds for leave-one-fold-out imputation", nFold, true)
      .add_option("n-contrast", "Number of contrasts to compute (<= number of columns in --in-labels)", nContrast, true)
      .add_option("threads", "Number of threads to use", nThreads)
      .add_option("seed", "Random seed", seed)
      .add_option("n-split", "Number of random splits (each split generates a set of test statistics for all genes", nSplit)
      .add_option("batch-size", "Batch size for processing", batchSize)
      .add_option("max-iter", "(LDA-SVB/HDP) Max iterations per doc", maxIter)
      .add_option("mean-change-tol", "(LDA-SVB/HDP) Convergence tolerance per doc", mDelta)
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

    std::string outf = outPrefix + ".score_test.tsv";
    FILE* wf = fopen(outf.c_str(), "w");
    if (!wf) {
        error("Failed to open output file %s for writing", outf.c_str());
    }
    fprintf(wf, "#Repeat\tFeature");
    for (int32_t c = 0; c < nContrast; ++c) {
        fprintf(wf, "\tC%d", c);
    }
    fprintf(wf, "\n");

    outf = outPrefix + ".split.tsv";
    FILE* splitInfo = fopen(outf.c_str(), "w");
    if (!splitInfo) {
        error("Failed to open split info file %s for writing", outf.c_str());
    }
    fprintf(splitInfo, "#Repeat\tFold\tHeldoutIndices\n");

    ConditionalDEtest obj(nFold, nContrast, nThreads, seed,
                          inFile, metaFile, labelFile, modelFile, maxIter, mDelta, batchSize, debug, verbose);
    int32_t M = obj.nFeatures();
    std::vector<std::string> featureNames = obj.getFeatureNames();

    int32_t r = 0;
    while (r < nSplit) {
        std::vector<Eigen::VectorXd> scores = obj.processAll();
        for (int j = 0; j < M; ++j) {
            fprintf(wf, "%d\t%s", r, featureNames[j].c_str());
            for (int32_t c = 0; c < nContrast; ++c) {
                fprintf(wf, "\t%.4f", scores[c][j]);
            }
            fprintf(wf, "\n");
        }
        const auto& heldoutIdx = obj.getHeldoutIndices();
        for (int32_t i = 0; i < nFold; ++i) {
            for (const auto& idx : heldoutIdx[i]) {
                fprintf(splitInfo, "%d\t%d\t%d\n", r, i, idx);
            }
        }
        r++;
        notice("Completed random split %d/%d", r, nSplit);

        if (r >= nSplit) {
            break;
        }
        obj.reset();
    }
    fclose(wf);
    fclose(splitInfo);

    return 0;
};
