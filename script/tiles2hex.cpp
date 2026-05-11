#include "preprocess_options.hpp"

int32_t cmdTiles2HexTxt(int32_t argc, char** argv) {

    Tiles2HexOptions opts;
    int debug = 0, verbose = 1000000;

    ParamList pl;
    opts.addStandaloneOptions(pl)
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

    opts.validateColumns();
    opts.resolveStandaloneGrid();
    HexGrid hexGrid = opts.makeGrid();
    std::vector<Rectangle<double>> rects = opts.parseBoundingBoxes();

    TileReader tileReader(opts.inTsv, opts.inIndex, &rects);
    if (!tileReader.isValid()) {
        error("Error opening input file: %s", opts.inTsv.c_str());
    }
    lineParser parser = opts.makeParser(&rects, true);

    if (opts.anchorFiles.empty()) {
        Tiles2Hex tiles2Hex(opts.nThreads, opts.tmpDir, opts.outFile, hexGrid, tileReader, parser, opts.min_counts, opts.seed, opts.bccSize);
        if (!tiles2Hex.run()) {
            return 1;
        }
        tiles2Hex.writeMetadata();
    } else {
        Tiles2UnitsByAnchor tiles2Hex(opts.nThreads, opts.tmpDir, opts.outFile, hexGrid, tileReader, parser, opts.anchorFiles, opts.radius, opts.min_counts, opts.noBackground, opts.seed);
        if (!tiles2Hex.run()) {
            return 1;
        }
        tiles2Hex.writeMetadata();
    }
    if (opts.randomize_output) {
        opts.sortOutput(false);
    }

    notice("Processing completed. Output is written to %s", opts.outFile.c_str());

    return 0;
}
