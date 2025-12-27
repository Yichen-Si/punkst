#include "punkst.h"
#include "tileoperator.hpp"

/*
    Input:
        if --in is given, data file name is <in>.tsv/.bin depending on --binary, index file name is <in>.index
        else --in-data and --in-index must be given, and the file format is inferred from the index file
*/

int32_t cmdManipulateTiles(int32_t argc, char** argv) {
    std::string inPrefix, inData, inIndex, outPrefix;
    int32_t tileSize = -1;
    bool isBinary = false;
    bool reorganize = false;
    bool printIndex = false;
    bool dumpTSV = false;

    ParamList pl;
    pl.add_option("in-data", "Input data file", inData)
      .add_option("in-index", "Input index file", inIndex)
      .add_option("in", "Input prefix (equal to --in-tsv <in>.tsv/.bin --in-index <in>.index)", inPrefix)
      .add_option("binary", "Data file is in binary format", isBinary)
      .add_option("out", "Output prefix", outPrefix)
      .add_option("tile-size", "Tile size in the original data", tileSize)
      .add_option("reorganize", "Reorganize fragmented tiles", reorganize)
      .add_option("print-index", "Print the index entries to stdout", printIndex)
      .add_option("dump-tsv", "Dump all records to TSV format", dumpTSV);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (!inPrefix.empty()) {
        inData = inPrefix + (isBinary ? ".bin" : ".tsv");
        inIndex = inPrefix + ".index";
    } else if (inData.empty() || inIndex.empty()) {
        error("Either --in or both --in-data and --in-index must be specified");
    }

    TileOperator tileOp(inData, inIndex);

    if (reorganize) {
        tileOp.reorgTiles(outPrefix, tileSize);
    }

    if(printIndex) {
        tileOp.printIndex();
    }

    if (dumpTSV) {
        tileOp.dumpTSV(outPrefix);
    }

    return 0;
}
