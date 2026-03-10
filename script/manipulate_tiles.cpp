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
    bool binaryOut = false;
    bool isBinary = false;
    bool reorganize = false;
    bool printIndex = false;
    bool extractRegion = false;
    std::string extractRegionGeoJSON;
    int64_t extractRegionScale = 10;
    bool dumpTSV = false;
    bool probDot = false;
    bool cellAnno = false;
    bool spatialMetrics = false;
    bool profileShellSurface = false;
    bool oneFactorMask = false;
    bool softMask = false;
    bool hardMask = false;
    bool skipMaskOverlap = false;
    bool skipBoundaries = false;
    uint32_t ccMinSize = 1;
    uint32_t maskMinComponentArea = 5;
    uint32_t maskMinHoleArea = 5;
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
    int32_t focalK = -1;
    int32_t maskRadius = 0;
    float maxCellDiameter = 50;
    double maskThreshold = -1.0;
    double maskMinFrac = 0.05;
    float maskMinPixelProb = 0.01;
    double maskMinTileMass = 10.;
    double maskSimplify = 0.0;
    std::string templateGeoJSON;
    std::string templateOutPrefix;
    float xmin = 0.0f, xmax = -1.0f, ymin = 0.0f, ymax = -1.0f;
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
      .add_option("extract-region", "Extract all records within --xmin/--xmax/--ymin/--ymax and write a new indexed file pair", extractRegion)
      .add_option("extract-region-geojson", "Extract all records inside a GeoJSON Polygon/MultiPolygon region", extractRegionGeoJSON)
      .add_option("extract-region-scale", "Integer scale for GeoJSON region snapping", extractRegionScale)
      .add_option("reorganize", "Reorganize fragmented tiles", reorganize)
      .add_option("dump-tsv", "Dump all records to TSV format", dumpTSV)
      .add_option("smooth-top-labels", "Per-tile island smoothing of top labels (>0 to enable)", smoothTopLabelsRounds)
      .add_option("fill-empty-islands", "Fill empty pixels surrounded by consistent neighbors (for --smooth-top-labels)", fillEmptyIslands)
      .add_option("spatial-metrics", "Compute area/perim metrics for single & pairwise channels", spatialMetrics)
      .add_option("hard-factor-mask", "Build per-label hard masks from the top factor and report global summaries", hardMask)
      .add_option("profile-one-factor-mask", "Build thresholded neighborhood mask for one factor and report selected pairwise overlaps", oneFactorMask)
      .add_option("soft-factor-mask", "Build per-factor soft masks, polygonize them, and export merged boundaries as GeoJSON", softMask)
      .add_option("skip-mask-overlap", "Skip computing mask overlaps with top co-localized factors (for --profile-factor-masks)", skipMaskOverlap)
      .add_option("skip-boundaries", "Skip GeoJSON export and write only summary tables for hard/soft factor mask commands", skipBoundaries)
      .add_option("cc-min-size", "Minimum size of connected components", ccMinSize)
      .add_option("mask-min-component-area", "Minimum 4-connected component area retained in each per-factor mask", maskMinComponentArea)
      .add_option("mask-min-hole-area", "Minimum hole area retained in output polygons for --soft-factor-mask", maskMinHoleArea)
      .add_option("shell-surface", "Compute shell occupancy and directional surface-distance histograms", profileShellSurface)
      .add_option("shell-radii", "Radii list for --spatial-shell-surface (pixel units)", shellRadii)
      .add_option("surface-dmax", "Maximum distance for surface histogram in --shell-surface", surfaceDmax)
      .add_option("spatial-min-pix-per-tile-label", "Only seed a label in a tile if nPixels(label,tile) >= this threshold", spatialMinPixPerTilePerLabel)
      .add_option("confusion", "Compute confusion matrix using r-by-r squares", confusionRes)
      .add_option("prob-dot", "Compute pairwise probability dot products", probDot)
      .add_option("annotate-cell", "Annotate factor composition per cell and subcellular component", cellAnno)
      .add_option("merge-emb", "List of embedding files to merge", inMergeEmbFiles)
      .add_option("annotate-pts", "Prefix of the data file to annotate", inMergePtsPrefix)
      .add_option("k2keep", "Number of factors to keep from each source (merge only)", k2keep)
      .add_option("icol-x", "X coordinate column index, 0-based", icol_x)
      .add_option("icol-y", "Y coordinate column index, 0-based", icol_y)
      .add_option("icol-z", "Z coordinate column index, 0-based", icol_z)
      .add_option("icol-c", "Cell ID column index, 0-based (for pix2cell)", icol_c)
      .add_option("icol-s", "Cell component column index, 0-based (for pix2cell)", icol_s)
      .add_option("k-out", "Number of top factors to output (for pix2cell)", kOut)
      .add_option("focal-k", "Focal factor index for --profile-factor-masks", focalK)
      .add_option("mask-radius", "Neighborhood radius in pixels for --profile-factor-masks", maskRadius)
      .add_option("mask-threshold", "Neighborhood mass fraction threshold for --profile-factor-masks", maskThreshold)
      .add_option("mask-min-frac", "Minimum focal-mask mass fraction to keep a secondary factor in --profile-factor-masks", maskMinFrac)
      .add_option("mask-min-pixel-prob", "Minimum per-pixel factor probability used when constructing masks in --profile-factor-masks", maskMinPixelProb)
      .add_option("mask-min-tile-mass", "Skip factors whose total mass in a tile is below this threshold for --soft-factor-mask", maskMinTileMass)
      .add_option("mask-simplify", "Optional simplification tolerance applied to output polygons for --soft-factor-mask", maskSimplify)
      .add_option("template-geojson", "Optional template JSON/GeoJSON file used to write one additional per-factor file with replaced geometry and title", templateGeoJSON)
      .add_option("template-out-prefix", "Optional output prefix for per-factor files written from --template-geojson (defaults to --out)", templateOutPrefix)
      .add_option("max-cell-diameter", "Maximum cell diameter in microns (for pix2cell)", maxCellDiameter)
      .add_option("xmin", "Minimum x coordinate for --extract-region", xmin)
      .add_option("xmax", "Maximum x coordinate for --extract-region", xmax)
      .add_option("ymin", "Minimum y coordinate for --extract-region", ymin)
      .add_option("ymax", "Maximum y coordinate for --extract-region", ymax);
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

    if (extractRegion) {
        if (!extractRegionGeoJSON.empty()) {
            error("--extract-region and --extract-region-geojson are mutually exclusive");
        }
        if (xmin >= xmax || ymin >= ymax) {
            error("Valid --xmin/--xmax/--ymin/--ymax are required for --extract-region");
        }
        tileOp.extractRegion(outPrefix, xmin, xmax, ymin, ymax);
        return 0;
    }

    if (!extractRegionGeoJSON.empty()) {
        tileOp.extractRegionGeoJSON(outPrefix, extractRegionGeoJSON, extractRegionScale);
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
    if (oneFactorMask) {
        if (focalK < 0) {
            error("--focal-k is required for --profile-factor-masks");
        }
        if (maskRadius < 0) {
            error("--mask-radius must be >= 0");
        }
        if (maskThreshold < 0.0 || maskThreshold > 1) {
            error("--mask-threshold must be in (0, 1) for --profile-factor-masks");
        }
        if (maskMinFrac < 0.0 || maskMinFrac >= 1.0) {
            error("--mask-min-frac must be in [0,1)");
        }
        maskThreshold = maskThreshold * (2 * maskRadius + 1) * (2 * maskRadius + 1); // convert from fraction to absolute mass
        tileOp.profileSoftFactorMasks(outPrefix, focalK, maskRadius,
            maskThreshold, maskMinFrac, maskMinPixelProb,
            maskMinComponentArea, skipMaskOverlap);
        return 0;
    }
    if (softMask) {
        if (maskRadius < 0) {
            error("--mask-radius must be >= 0");
        }
        if (maskThreshold < 0.0 || maskThreshold > 1.0) {
            error("--mask-threshold must be in [0,1] for --soft-factor-mask");
        }
        if (maskMinPixelProb < 0.0f) {
            error("--mask-min-pixel-prob must be >= 0");
        }
        if (maskMinTileMass < 0.0) {
            error("--mask-min-tile-mass must be >= 0");
        }
        if (maskSimplify < 0.0) {
            error("--mask-simplify must be >= 0");
        }
        tileOp.softFactorMask(outPrefix, maskRadius, maskThreshold,
            maskMinPixelProb, maskMinTileMass, maskMinComponentArea,
            maskMinHoleArea, maskSimplify, skipBoundaries, templateGeoJSON, templateOutPrefix);
        return 0;
    }
    if (hardMask) {
        tileOp.hardFactorMask(outPrefix, ccMinSize, skipBoundaries, templateGeoJSON, templateOutPrefix);
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
        tileOp.annotate(inMergePtsPrefix, outPrefix, icol_x, icol_y);
        return 0;
    }

    return 0;
}
