#include "punkst.h"
#include "markerselection.hpp"

int32_t cmdQ2Markers(int32_t argc, char** argv) {
    std::string inFile, infoFile, outPref, outFile;
    int32_t K, neighbors = 10;
    int32_t valueBytes = 8;
    bool binaryInput = false;
    int32_t minCount = 1;
    std::vector<std::string> selectedMarkers;
    int32_t verbose = 0;
    double maxRankFraction = 0.2;

    ParamList pl;
    pl.add_option("input", "Input co-occurrence matrix (binary or TSV)", inFile, true)
        .add_option("info", "Input gene info", infoFile, true)
        .add_option("K", "Total number of markers to select", K, true)
        .add_option("neighbors", "Number of top neighbors to find for each marker (default: 10)", neighbors)
        .add_option("binary", "Input matrix is in binary format", binaryInput)
        .add_option("value-bytes", "Number of bytes for each value in the matrix (default: 8, only used for binary input)", valueBytes)
        .add_option("fixed", "Fixed markers", selectedMarkers)
        .add_option("min-count", "Minimum count for a feature to be considered as a marker (default: 1)", minCount);
    pl.add_option("out", "Output prefix", outPref, true)
        .add_option("neighbor-max-rank-fraction", "Maximum fraction of rank to consider for (mutual) neighbors (default: 0.2)", maxRankFraction)
        .add_option("verbose", "Verbose level (default: 0)", verbose);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    MarkerSelector selector(infoFile, inFile, binaryInput, valueBytes, minCount, verbose);
    selector.selectMarkers(K, selectedMarkers);
    std::vector<MarkerSelector::markerSetInfo> neighborLists = selector.findNeighborsToAnchors(neighbors, maxRankFraction);

    outFile = outPref + ".tsv";
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

    return 0;
}
