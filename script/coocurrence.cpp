#include "tiles2cooccurrence.hpp"

int32_t cmdTiles2FeatureCooccurrence(int32_t argc, char** argv) {
    std::string inTsv, inIndex, outPref, dictFile;
    double radius, halflife = -1, localMin = 0;
    int nThreads = 1, debug = 0;
    int icol_x, icol_y, icol_feature, icol_val;
    bool binaryOutput = false, weightByCount = false;
    int minNeighbor = 1;
    std::vector<double> boundingBoxes;

    ParamList pl;
    // Input Options
    pl.add_option("in-tsv", "Input TSV file. Header must begin with #", inTsv, true)
        .add_option("in-index", "Input index file", inIndex, true)
        .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x, true)
        .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y, true)
        .add_option("icol-feature", "Column index for feature (0-based)", icol_feature)
        .add_option("feature-dict", "If feature column is not integer, provide a dictionary/list of all possible values", dictFile)
        .add_option("icol-val", "Column index for the integer count (0-based)", icol_val)
        .add_option("bounding-boxes", "Rectangular query regions (xmin ymin xmax ymax)*", boundingBoxes)
        .add_option("weight-by-count", "Weight co-occurrence by the product of the number of transcripts (default: false)", weightByCount)
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

    std::vector<Rectangle<double>> rects;
    if (boundingBoxes.size() > 0) {
        int32_t nrects = parseCoordsToRects(rects, boundingBoxes);
        if (nrects <= 0) {
            error("Error parsing bounding boxes");
        }
        notice("Received %d bounding boxes", nrects);
    }

    TileReader tileReader(inTsv, inIndex, &rects);
    if (!tileReader.isValid()) {
        error("Error parsing input index file");
        return 1;
    }
    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val, dictFile, &rects);

    Tiles2FeatureCooccurrence cooccurrence(nThreads, tileReader, parser, outPref, radius, halflife, localMin, binaryOutput, minNeighbor, weightByCount);

    cooccurrence.run();

    return 0;
}
