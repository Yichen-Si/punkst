#include "punkst.h"
#include "hlda.hpp"

int32_t test(int32_t argc, char** argv) {

	std::string inFile, metaFile, weightFile, outPrefix;
    int32_t seed = -1, nThreads = 1, debug = 0, verbose = 500000;
    int32_t L = 4, max_k = 1024;
    int32_t thr_heavy = 50, thr_prune = 0;
    int32_t final_thr_prune = -1;
    int32_t sMC = 5, nIter = 10, nMCiter = 5, nMBiter = 5, nFixedIter = 3;
    int32_t bsize = 512, csize = 1024, nInit = 1000;
    int32_t minCountTrain = 50;
    std::vector<double> log_gamma = {};
    double gem_m = 1, alpha = 0.2;
    std::vector<double> eta = {1., .5, .25};
    std::vector<int> max_outdg = {};
    bool transform = false;

    ParamList pl;
    // Input Options
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true)
      .add_option("feature-weights", "Input weights file", weightFile);
    pl.add_option("levels", "", L)
      .add_option("max-nodes", "Max number of nodes", max_k)
      .add_option("max-children", "Max number of children for each node on each layer", max_outdg)
      .add_option("thr-heavy", "Threshold for use batch update for a topic", thr_heavy)
      .add_option("thr-prune", "Threshold for pruning a topic", thr_prune)
      .add_option("final-thr-prune", "Final threshold for pruning", final_thr_prune)
      .add_option("s-mc", "Number of samples for integrating out z in P(c|w)", sMC)
      .add_option("n-iter", "Number of total iterations", nIter)
      .add_option("n-mc-iter", "Number of iterations to sample from P(c|w) instead of P(c|w,z)", nMCiter)
      .add_option("n-mb-iter", "Number of minibatch iterations", nMBiter)
      .add_option("n-fixed-iter", "Number of iterations with fixed tree (at the end)", nFixedIter)
      .add_option("n-init", "Number of docs to for strict CGS initialization", nInit)
      .add_option("minibatch-size", "Minibatch size", bsize)
      .add_option("tree-prune-interval", "Consolidate tree every N docs", csize)
      .add_option("min-count-train", "Minimum total count for a document to be included in model training", minCountTrain)
      .add_option("log-gamma", "log(gamma) for gamma in CRP", log_gamma)
      .add_option("alpha", "Dirichlet prior for c", alpha)
      .add_option("gem-m", "m in GEM(alpha, m) prior for z", gem_m)
      .add_option("eta", "Dirichlet prior for beta", eta)
      .add_option("seed", "Random seed", seed)
      .add_option("threads", "Number of threads", nThreads);
    // Output Options
    pl.add_option("out-prefix", "Output prefix", outPrefix, true)
      .add_option("transform", "Compute probabilistic path assignments", transform)
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
    if (debug > 0) {
        punkst::Logger::getInstance().setLevel(punkst::LogLevel::DEBUG);
    }
    if (log_gamma.size() == 0) {
        log_gamma = std::vector<double>(L, -60);
    }
    final_thr_prune = (final_thr_prune < 0) ? thr_prune : final_thr_prune;

    HexReader reader(metaFile);
    int32_t M = reader.nFeatures;

    HLDA hlda(L, M, seed, nThreads, debug, verbose,
        max_k, log_gamma, thr_heavy, thr_prune, sMC, eta, alpha, gem_m, max_outdg);

    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("%s: Error opening input file: %s", __func__, inFile.c_str());
    }
    std::vector<nCrpLeaf> docs;
    std::string line, tmp;
    while (std::getline(inFileStream, line)) {
        Document doc;
        int32_t ct = reader.parseLine(doc, tmp, line);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        if (ct < minCountTrain) {
            continue;
        }
        docs.emplace_back(doc);
    }
    inFileStream.close();
    notice("Read %lu documents", docs.size());

    hlda.fit(docs, nIter, nMCiter, nMBiter, bsize, csize, nInit);

    if (nFixedIter > 0) {
        hlda.set_allow_new_topic(false);
        for (int i = 0; i < nFixedIter - 1; ++i) {
            hlda.fit_onepass(docs, false);
        }
        hlda.set_tree_threshold(thr_heavy, final_thr_prune);
        hlda.fit_onepass(docs, false);
        printTree(hlda.get_tree(), std::cout, true, true);
    }

    std::string outf;

    outf = outPrefix + ".model.tsv";
    std::vector<std::string> vocab = reader.features;
    if (vocab.size() != M) {
        warning("Did not find feature names in the input metadata, using numbers instead");
        vocab.resize(M);
        for (int i = 0; i < M; ++i) {
            vocab[i] = std::to_string(i);
        }
    }
    const MatrixXd& mat = hlda.get_nwk_global();
    hlda.write_model_to_file(outf, vocab, 0);

    outf = outPrefix + ".de_chisq.tsv";
    auto topgenes = chisq_from_matrix_marginal(
        mat, vocab, hlda.node_names, outf, 9, nThreads, 0.5, 10, 1.5, 0.05);

    auto tree = hlda.get_tree();
    int l = L-1;
    while (l > 1 && tree.n_nodes[l] > 10) {
        l--;
    }
    while (l < L-1) {
        outf = outPrefix + ".top_" + std::to_string(l+1) + ".dot";
        WriteTreeAsDot(tree, outf, &topgenes, l);
        l++;
    }
    outf = outPrefix + ".dot";
    WriteTreeAsDot(tree, outf, &topgenes, 3);

    WriteTreeAsTSV(tree, outPrefix);

    return 0;
}
