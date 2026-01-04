#include "punkst.h"
#include "tileoperator.hpp"

/*
    Input:
        if --in is given, data file name is <in>.tsv/.bin depending on --binary, index file name is <in>.index
        else --in-data and --in-index must be given, and the file format is inferred from the index file
*/

int32_t cmdManipulateTiles(int32_t argc, char** argv) {
    std::string inPrefix, inData, inIndex, outPrefix;
    std::vector<std::string> inMergeEmbFiles;
    std::string inMergePtsPrefix;
    int32_t tileSize = -1;
    bool isBinary = false;
    bool reorganize = false;
    bool printIndex = false;
    bool dumpTSV = false;
    bool probDot = false;
    std::vector<uint32_t> k2keep;
    int32_t icol_x = -1;
    int32_t icol_y = -1;
    int32_t coordDigits = 2, probDigits = 4; // for dump-tsv
    bool binaryOut = false;

    ParamList pl;
    pl.add_option("in-data", "Input data file", inData)
      .add_option("in-index", "Input index file", inIndex)
      .add_option("in", "Input prefix (equal to --in-data <in>.tsv/.bin --in-index <in>.index)", inPrefix)
      .add_option("binary", "Data file is in binary format", isBinary)
      .add_option("out", "Output prefix", outPrefix)
      .add_option("tile-size", "Tile size in the original data", tileSize)
      .add_option("reorganize", "Reorganize fragmented tiles", reorganize)
      .add_option("print-index", "Print the index entries to stdout", printIndex)
      .add_option("dump-tsv", "Dump all records to TSV format", dumpTSV)
      .add_option("prob-dot", "Compute pairwise probability dot products", probDot)
      .add_option("merge-emb", "List of embedding files to merge", inMergeEmbFiles)
      .add_option("annotate-pts", "Prefix of the data file to annotate", inMergePtsPrefix)
      .add_option("k2keep", "Number of factors to keep from each source (merge only)", k2keep)
      .add_option("icol-x", "X coordinate column index, 0-based (annotate only)", icol_x)
      .add_option("icol-y", "Y coordinate column index, 0-based (annotate only)", icol_y)
      .add_option("binary-out", "Output in binary format (merge only)", binaryOut);
    pl.add_option("coord-digits", "Number of decimal digits to output for coordinates (for dump-tsv)", coordDigits)
      .add_option("prob-digits", "Number of decimal digits to output for probabilities (for dump-tsv)", probDigits);

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

    if(printIndex) {
        tileOp.printIndex();
    }

    if (reorganize) {
        tileOp.reorgTiles(outPrefix, tileSize);
        return 0;
    }

    if (dumpTSV) {
        tileOp.dumpTSV(outPrefix, probDigits, coordDigits);
        return 0;
    }

    if (probDot) {
        if (outPrefix.empty()) error("Output prefix must be specified for prob-dot");
        if (!inMergeEmbFiles.empty()) {
            tileOp.probDot_multi(inMergeEmbFiles, outPrefix, k2keep, probDigits);
        } else {
            tileOp.probDot(outPrefix, probDigits);
        }
        return 0;
    }

    if (!inMergeEmbFiles.empty()) {
        if (outPrefix.empty()) error("Output prefix must be specified for merge");
        tileOp.merge(inMergeEmbFiles, outPrefix, k2keep, binaryOut);
        return 0;
    }

    if (!inMergePtsPrefix.empty()) {
        if (icol_x < 0 || icol_y < 0) {
            error("icol-x and icol-y for --annotate-pts must be specified");
        }
        if (outPrefix.empty()) error("Output prefix must be specified for annotate");
        tileOp.annotate(inMergePtsPrefix, outPrefix, icol_x, icol_y);
        return 0;
    }

    return 0;
}
