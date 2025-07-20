#include "punkst.h"
#include "pts2tiles.hpp"

int32_t cmdPts2TilesTsv(int32_t argc, char** argv) {

    std::string inTsv, outPref, tmpDir;
    int nThreads0 = 1, tileSize = -1;
    int tileBuffer = 1000, batchSize = 10000;
    int debug = 0, verbose = 1000000;
    int icol_x, icol_y, nskip = 0;
    int icol_feature = -1;
    double scale = 0;
    int digits = 2;
    std::vector<int32_t> icol_ints;

    ParamList pl;
    // Input Options
    pl.add_option("in-tsv", "Input TSV file.", inTsv)
      .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x)
      .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y)
      .add_option("icol-feature", "Column index for feature (0-based)", icol_feature)
      .add_option("icol-int", "Column index for integer values (0-based)", icol_ints)
      .add_option("skip", "Number of lines to skip in the input file (default: 0)", nskip)
      .add_option("temp-dir", "Directory to store temporary files", tmpDir)
      .add_option("tile-size", "Tile size (in the same unit as the input coordinates)", tileSize)
      .add_option("tile-buffer", "Buffer size per tile per thread (default: 1000 lines)", tileBuffer)
      .add_option("batch-size", "(Only used if the input is gzipped or a stdin stream.) Batch size in terms of the number of lines (default: 10000)", batchSize)
      .add_option("scale", "Scale the coordinates by this factor. This may not be very efficient (default: no scaling)", scale)
      .add_option("digits", "Precision for the output coordinates (only used when --scale is provided; default 2)", digits)
      .add_option("threads", "Number of threads to use (default: 1)", nThreads0);
    // Output Options
    pl.add_option("out-prefix", "Output TSV file", outPref)
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

    if (tileSize <= 0) {
        error("Tile size is required to be a positive integer");
    }

    // Determine the number of threads to use.
    unsigned int nThreads = std::thread::hardware_concurrency();
    if (nThreads == 0 || nThreads >= nThreads0) {
        nThreads = nThreads0;
    }
    notice("Using %u threads for processing", nThreads);

    // Determine the input type and thus the streaming mode
    bool streaming = false;
    if (inTsv == "-" ||
        (inTsv.size()>3 && inTsv.compare(inTsv.size()-3,3,".gz")==0))
    streaming = true;

    Pts2Tiles pts2Tiles(nThreads, inTsv, tmpDir, outPref, tileSize, icol_x, icol_y, icol_feature, icol_ints, nskip, streaming, tileBuffer, batchSize, scale, digits);
    if (!pts2Tiles.run()) {
        return 1;
    }

    notice("Processing completed. Output is written to %s.tsv and index file is written to %s.index", outPref.c_str(), outPref.c_str());
    return 0;
}
