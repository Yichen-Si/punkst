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
    std::string annotateHeaderFile;
    int32_t tileSize = -1;
    bool binaryOut = false;
    bool isBinary = false;
    bool reorganize = false;
    bool printIndex = false;
    bool dumpTSV = false;
    bool probDot = false;
    bool cellAnno = false;
    bool spatialMetrics = false;
    bool cnctComponents = false;
    bool profileShellSurface = false;
    uint32_t ccMinSize = 1;
    std::vector<int32_t> shellRadii;
    int32_t surfaceDmax = -1;
    uint32_t spatialMinPixPerTilePerLabel = 0;
    int32_t smoothTopLabelsRounds = 0;
    bool fillEmptyIslands = false;
    double confusionRes = -1.0;
    std::vector<uint32_t> k2keep;
    int32_t icol_x = -1, icol_y = -1, icol_z = -1;
    int32_t icol_c = -1, icol_s = -1;
    int32_t coordDigits = 2, probDigits = 4;
    int32_t kOut = 0;
    int32_t K = -1;
    float maxCellDiameter = 50;
    int32_t threads = 1;
    int32_t debug_ = 0;

    ParamList pl;
    pl.add_option("in-data", "Input data file", inData)
      .add_option("in-index", "Input index file", inIndex)
      .add_option("in", "Input prefix (equal to --in-data <in>.tsv/.bin --in-index <in>.index)", inPrefix)
      .add_option("binary", "Data file is in binary format", isBinary)
      .add_option("K", "Total number of factors in the data", K)
      .add_option("tile-size", "Tile size in the original data", tileSize);
    pl.add_option("print-index", "Print the index entries to stdout", printIndex)
      .add_option("reorganize", "Reorganize fragmented tiles", reorganize)
      .add_option("dump-tsv", "Dump all records to TSV format", dumpTSV)
      .add_option("smooth-top-labels", "Per-tile island smoothing of top labels (>0 to enable)", smoothTopLabelsRounds)
      .add_option("fill-empty-islands", "Fill empty pixels surrounded by consistent neighbors (only for --smooth-top-labels)", fillEmptyIslands)
      .add_option("spatial-metrics", "Compute area/perim metrics for single & pairwise channels", spatialMetrics)
      .add_option("connected-components", "Compute global connected-component sizes per label", cnctComponents)
      .add_option("cc-min-size", "Minimum size of connected components", ccMinSize)
      .add_option("shell-surface", "Compute shell occupancy and directional surface-distance histograms", profileShellSurface)
      .add_option("shell-radii", "Radii list for --spatial-shell-surface (pixel units)", shellRadii)
      .add_option("surface-dmax", "Maximum distance for surface histogram in --shell-surface", surfaceDmax)
      .add_option("spatial-min-pix-per-tile-label", "Only seed a label in a tile if nPixels(label,tile) >= this threshold", spatialMinPixPerTilePerLabel)
      .add_option("confusion", "Compute confusion matrix using r-by-r squares", confusionRes)
      .add_option("prob-dot", "Compute pairwise probability dot products", probDot)
      .add_option("annotate-cell", "Annotate factor composition per cell and subcellular component", cellAnno)
      .add_option("merge-emb", "List of embedding files to merge", inMergeEmbFiles)
      .add_option("annotate-pts", "Prefix of the data file to annotate", inMergePtsPrefix)
      .add_option("annotate-header-file", "Use first line of this file as header base for --annotate-pts output", annotateHeaderFile)
      .add_option("k2keep", "Number of factors to keep from each source (merge only)", k2keep)
      .add_option("icol-x", "X coordinate column index, 0-based", icol_x)
      .add_option("icol-y", "Y coordinate column index, 0-based", icol_y)
      .add_option("icol-z", "Z coordinate column index, 0-based", icol_z)
      .add_option("icol-c", "Cell ID column index, 0-based (for pix2cell)", icol_c)
      .add_option("icol-s", "Cell component column index, 0-based (for pix2cell)", icol_s)
      .add_option("k-out", "Number of top factors to output (for pix2cell)", kOut)
      .add_option("max-cell-diameter", "Maximum cell diameter in microns (for pix2cell)", maxCellDiameter);
    pl.add_option("out", "Output prefix", outPrefix)
      .add_option("coord-digits", "Number of decimal digits to output for coordinates (for dump-tsv)", coordDigits)
      .add_option("prob-digits", "Number of decimal digits to output for probabilities (for dump-tsv)", probDigits)
      .add_option("binary-out", "Output in binary format (merge only)", binaryOut)
      .add_option("threads", "Number of threads to use", threads)
      .add_option("debug", "Debug", debug_);

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
    if (K > 0) {tileOp.setFactorCount(K);}
    tileOp.setThreads(threads);

    if(printIndex) {
        tileOp.printIndex();
    }
    if (outPrefix.empty()) {
        return 0;
    }
    if (debug_ > 0) { // CAUTION
        tileOp.sampleTilesToDebug(debug_);
    }

    if (reorganize) {
        tileOp.reorgTiles(outPrefix, tileSize);
        return 0;
    }

    if (dumpTSV) {
        tileOp.dumpTSV(outPrefix, probDigits, coordDigits);
        return 0;
    }

    if (smoothTopLabelsRounds > 0) {
        tileOp.smoothTopLabels2D(outPrefix, smoothTopLabelsRounds, fillEmptyIslands);
        return 0;
    }

    if (spatialMetrics) {
        tileOp.spatialMetricsBasic(outPrefix);
        return 0;
    }
    if (cnctComponents) {
        tileOp.connectedComponents(outPrefix, ccMinSize);
        return 0;
    }
    if (profileShellSurface) {
        if (shellRadii.empty()) {
            error("--spatial-radii is required for --shell-surface");
        }
        if (surfaceDmax < 0) {
            error("--spatial-dmax (>=0) is required for --shell-surface");
        }
        tileOp.profileShellAndSurface(outPrefix, shellRadii, surfaceDmax, ccMinSize, spatialMinPixPerTilePerLabel);
        return 0;
    }

    if (confusionRes >= 0) {
        auto confusion = tileOp.computeConfusionMatrix(confusionRes, outPrefix.c_str(), probDigits);
        return 0;
    }

    if (probDot) {
        if (!inMergeEmbFiles.empty()) {
            tileOp.probDot_multi(inMergeEmbFiles, outPrefix, k2keep, probDigits);
        } else {
            tileOp.probDot(outPrefix, probDigits);
        }
        return 0;
    }

    if (cellAnno) {
        tileOp.pix2cell(inMergePtsPrefix, outPrefix, icol_c, icol_x, icol_y, icol_s, icol_z, kOut, maxCellDiameter);
        return 0;
    }

    if (!inMergeEmbFiles.empty()) {
        tileOp.merge(inMergeEmbFiles, outPrefix, k2keep, binaryOut);
        return 0;
    }

    if (!inMergePtsPrefix.empty()) {
        if (icol_x < 0 || icol_y < 0) {
            error("icol-x and icol-y for --annotate-pts must be specified");
        }
        tileOp.annotate(inMergePtsPrefix, outPrefix, icol_x, icol_y, icol_z, annotateHeaderFile);
        return 0;
    }

    return 0;
}
