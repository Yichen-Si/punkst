#include "punkst.h"
#include "preprocess_options.hpp"

int32_t cmdPts2TilesTsv(int32_t argc, char** argv) {

    Pts2TilesOptions opts;
    int debug = 0, verbose = 1000000;

    ParamList pl;
    // Input Options
    opts.addInputOptions(pl);
    // Output Options
    opts.addOutputOptions(pl)
      .add_option("verbose", "Verbose", verbose)
      .add_option("debug", "Debug", debug);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    opts.validateStandalone();

    // Determine the number of threads to use.
    unsigned int nThreads = opts.resolveThreads();
    notice("Using %u threads for processing", nThreads);

    Pts2Tiles pts2Tiles = opts.makeRunner(nThreads);
    if (!pts2Tiles.run()) {
        return 1;
    }

    notice("Processing completed. Output is written to %s.tsv and index file is written to %s.index", opts.outPref.c_str(), opts.outPref.c_str());
    return 0;
}
