#include "punkst.h"
#include "gene_bin_utils.hpp"

int32_t cmdGeneBins(int32_t argc, char** argv) {
    std::string inTsv, outFile;
    int32_t icolFeature = 0;
    int32_t icolCount = 1;
    int32_t nGeneBins = 50;
    int32_t skipLines = 0;
    uint64_t targetMolecules = 1000000;
    double singletonRatio = 1.0;
    std::string mode = "adaptive";

    ParamList pl;
    pl.add_option("in-tsv", "Input TSV with feature names and counts", inTsv, true)
      .add_option("out", "Output gene-bin JSON file", outFile, true)
      .add_option("icol-feature", "Feature-name column index, 0-based", icolFeature)
      .add_option("icol-count", "Feature-count column index, 0-based", icolCount)
      .add_option("n-gene-bins", "Maximum number of gene bins in adaptive mode; requested count in fixed mode", nGeneBins)
      .add_option("gene-bin-mode", "Gene-bin packing mode: adaptive or fixed", mode)
      .add_option("gene-bin-target-molecules", "Target molecules per adaptive gene bin", targetMolecules)
      .add_option("gene-bin-singleton-ratio", "Adaptive singleton threshold as a multiple of target molecules", singletonRatio)
      .add_option("skip-lines", "Number of initial lines to skip", skipLines);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    GeneBinBuildOptions options;
    options.nGeneBins = nGeneBins;
    options.icolFeature = icolFeature;
    options.icolCount = icolCount;
    options.skipLines = skipLines;
    options.targetMolecules = targetMolecules;
    options.singletonRatio = singletonRatio;
    options.mode = parse_gene_bin_mode(mode);
    GeneBinInfo info(inTsv, options);
    info.write_gene_bin_info_json(outFile);
    notice("Wrote %zu gene-bin entries to %s", info.entries.size(), outFile.c_str());
    return 0;
}
