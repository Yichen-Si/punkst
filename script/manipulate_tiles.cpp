#include "punkst.h"
#include "tileoperator.hpp"

int32_t cmdManipulateTiles(int32_t argc, char** argv) {
    std::string inPrefix, inTsv, inIndex, outPrefix;
    int32_t tileSize = -1;
    bool reorganize = false;
    bool printIndex = false;

    ParamList pl;
    pl.add_option("in-tsv", "Input TSV file. Header must begin with #", inTsv)
      .add_option("in-index", "Input index file", inIndex)
      .add_option("in", "Input prefix (equal to --in-tsv <in>.tsv --in-index <in>.index)", inPrefix)
      .add_option("out", "Output prefix", outPrefix)
      .add_option("tile-size", "Tile size used in the original data", tileSize)
      .add_option("reorganize", "Reorganize fragmented tiles", reorganize)
      .add_option("print-index", "Print the index entries to stdout", printIndex);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (!inPrefix.empty()) {
        inTsv = inPrefix + ".tsv";
        inIndex = inPrefix + ".index";
    } else {
        if (inTsv.empty() || inIndex.empty()) {
            error("Either --in or both --in-tsv and --in-index must be specified");
        }
    }

    TileOperator tileOp(inTsv, inIndex);

    if (reorganize) {
        if (tileSize <= 0) {
            error("Tile size is required and must be positive");
        }
        tileOp.reorgTiles(outPrefix, tileSize);
        return 0;
    }

    if(printIndex) {
        tileOp.printIndex();
        return 0;
    }

    return 0;
}
