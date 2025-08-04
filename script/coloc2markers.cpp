#include "punkst.h"
#include "markerselection.hpp"

// TODO: run multiple experiments with different minCount after loading the mtx
int32_t cmdQ2Markers(int32_t argc, char** argv) {
    std::string inFile, infoFile, outPref, outFile;
    int32_t K, neighbors = -1;
    int32_t valueBytes = 8;
    bool binaryInput = false, denseInput = false;
    int32_t minCount = 1;
    std::vector<std::string> selectedMarkers;
    double maxRankFraction = 0.1;
    int32_t maxIter = 500;
    double tol = 1e-6;
    int32_t threads = -1;
    int32_t verbose = 0;
    bool recoverFactors = false, weightFactorsByCounts = false;
    bool findNeighbors = false;

    ParamList pl;
    pl.add_option("input", "Input co-occurrence matrix (binary or TSV)", inFile, true)
        .add_option("info", "Input gene info", infoFile, true)
        .add_option("K", "Total number of markers to select", K, true)
        .add_option("neighbors", "Number of top neighbors to find for each marker (default: 10)", neighbors)
        .add_option("binary", "Input matrix is in binary format", binaryInput)
        .add_option("dense", "Input matrix is dense", denseInput)
        .add_option("value-bytes", "Number of bytes for each value in the matrix (default: 8, only used for binary input)", valueBytes)
        .add_option("fixed", "Fixed markers", selectedMarkers)
        .add_option("min-count", "Minimum count for a feature to be considered as a marker (default: 1)", minCount);
    pl.add_option("recover-factors", "Recover factors from the co-occurrence matrix after selecting markers", recoverFactors)
        .add_option("threads", "Number of threads to use (only used if --neighbors > 0 or --recover-factors is set. Default: -1, auto)", threads)
        .add_option("max-iter", "Maximum number of iterations for factor recovery (default: 500)", maxIter)
        .add_option("tol", "Tolerance for convergence (default: 1e-6)", tol)
        .add_option("weight-by-counts", "Weight factors by counts (default: false)", weightFactorsByCounts);
    pl.add_option("out", "Output prefix", outPref, true)
        .add_option("find-neighbors", "Find neighbors for each marker", findNeighbors)
        .add_option("neighbor-max-rank-fraction", "Maximum fraction of rank to consider for (mutual) neighbors (default: 0.1)", maxRankFraction)
        .add_option("verbose", "Verbose level (default: 0)", verbose);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    MarkerSelector selector(infoFile, inFile, binaryInput, denseInput, valueBytes, minCount, verbose, &selectedMarkers);
    selector.selectMarkers(K, selectedMarkers);

    outFile = outPref + ".top.tsv";
    std::ofstream ofs(outFile);
    for (uint32_t i=0; i < selectedMarkers.size(); ++i) {
        ofs << i << "\t" << selectedMarkers[i] << "\n";
    }
    ofs.close();

    if (findNeighbors || neighbors > 0) {
        if (neighbors <= 0) {
            neighbors = 10;
        }
        std::vector<MarkerSelector::markerSetInfo> neighborLists = selector.findNeighborsToAnchors(neighbors, threads, maxRankFraction);

        outFile = outPref + ".pairs.tsv";
        std::ofstream ofs(outFile);
        outFile = outPref + ".short.txt";
        std::ofstream ofsShort(outFile);
        if (!ofs) {
            error("Cannot open output: %s", outFile.c_str());
        }
        uint32_t k = 0;
        for (auto &marker : neighborLists) {
            ofsShort << marker.name;
            for (auto &neighbor : marker.neighbors) {
                ofs << k << "\t" << marker.name << "\t" << marker.count << "\t"
                    << neighbor.name << "\t" << neighbor.count << "\t"
                    << neighbor.qij << "\t" << neighbor.qji << "\t"
                    << neighbor.rij << "\t" << neighbor.rji << "\n";
                ofsShort << " " << neighbor.name;
            }
            ofsShort << "\n";
            k++;
        }
        ofs.close();
        ofsShort.close();
    }

    if (!recoverFactors)
        return 0;

    selector.computeTopicDistribution(maxIter, tol, threads, weightFactorsByCounts);
    auto& features = selector.getFeatureInfo();
    outFile = outPref + ".factors.tsv";
    ofs.open(outFile);
    if (!ofs) {
        error("Cannot open output: %s", outFile.c_str());
    }
    uint32_t M = features.size();
    ofs << std::scientific << std::setprecision(4);
    ofs << "Feature";
    for (int i = 0; i < K; ++i) {
        ofs << "\t" << i;
    }
    ofs << "\n";
    for (uint32_t j = 0; j < M; ++j) {
        ofs << features.at(j).name;
        for (int i = 0; i < K; ++i) {
            ofs << "\t" << selector.betas(j, i);
        }
        ofs << "\n";
    }
    ofs.close();

    return 0;
}
