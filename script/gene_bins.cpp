#include "punkst.h"
#include "gene_bin_utils.hpp"

int32_t cmdGeneBins(int32_t argc, char** argv) {
    std::string inTsv, outFile;
    int32_t icolFeature = 0;
    int32_t icolCount = 1;
    int32_t nGeneBins = 50;
    int32_t skipLines = 0;

    ParamList pl;
    pl.add_option("in-tsv", "Input TSV with feature names and counts", inTsv, true)
      .add_option("out", "Output gene-bin JSON file", outFile, true)
      .add_option("icol-feature", "Feature-name column index, 0-based", icolFeature)
      .add_option("icol-count", "Feature-count column index, 0-based", icolCount)
      .add_option("n-gene-bins", "Number of gene bins to create", nGeneBins)
      .add_option("skip-lines", "Number of initial lines to skip", skipLines);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    GeneBinInfo info(inTsv, nGeneBins, icolFeature, icolCount, skipLines);
    info.write_gene_bin_info_json(outFile);
    notice("Wrote %zu gene-bin entries to %s", info.entries.size(), outFile.c_str());
    return 0;
}
