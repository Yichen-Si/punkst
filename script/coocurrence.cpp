#include "tiles2cooccurrence.hpp"

int32_t cmdTiles2FeatureCooccurrence(int32_t argc, char** argv) {
    std::string inTsv, inIndex, outPref, dictFile;
    double radius, halflife = -1, localMin = 0;
    int nThreads = 1, debug = 0;
    int icol_x, icol_y, icol_feature;
    std::vector<int32_t> icol_ints;
    bool binaryOutput = false;
    int minNeighbor = 1;

    ParamList pl;
    // Input Options
    pl.add_option("in-tsv", "Input TSV file. Header must begin with #", inTsv, true)
        .add_option("in-index", "Input index file", inIndex, true)
        .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x, true)
        .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y, true)
        .add_option("icol-feature", "Column index for feature (0-based)", icol_feature)
        .add_option("feature-dict", "If feature column is not integer, provide a dictionary/list of all possible values", dictFile)
        .add_option("icol-int", "Column index for integer values (0-based)", icol_ints)
        .add_option("radius", "Radius to count coocurrence", radius, true)
        .add_option("halflife", "Halflife for exponential decay (default: -1, unweighted count)", halflife)
        .add_option("min-neighbor", "Minimum number of neighbors within the radius for a pixel to be included", minNeighbor)
        .add_option("local-min", "Minimum cooccurrence within a tile to record", localMin)
        .add_option("threads", "Number of threads to use (default: 1)", nThreads);
    // Output Options
    pl.add_option("out", "Output prefix", outPref, true)
        .add_option("binary", "Output in binary format (default: false)", binaryOutput)
        .add_option("debug", "Debug", debug);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    TileReader tileReader(inTsv, inIndex);
    if (!tileReader.isValid()) {
        error("Error opening input file: %s", inTsv.c_str());
        return 1;
    }
    lineParser parser(icol_x, icol_y, icol_feature, icol_ints, dictFile);

    Tiles2FeatureCooccurrence cooccurrence(nThreads, tileReader, parser, outPref, radius, halflife, localMin, binaryOutput, minNeighbor);

    cooccurrence.run();

    return 0;
}
