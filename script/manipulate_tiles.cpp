#include <cmath>
#include <limits>

#include "punkst.h"
#include "tileoperator.hpp"

int32_t cmdManipulateTiles(int32_t argc, char** argv) {
    std::string inPrefix, inData, inIndex, outPrefix;
    std::vector<std::string> inMergeEmbFiles;
    std::vector<std::string> mergeEmbPrefixes;
    std::string inMergePtsPrefix;
    bool mergeKeepAllMain = false;
    bool mergeKeepAll = false;
    bool annoKeepAll = false;
    int32_t tileSize = -1;
    bool binaryOut = false;
    bool isBinary = false;
    bool reorganize = false;
    bool printIndex = false;
    bool extractRegion = false;
    std::string extractRegionGeoJSON;
    int64_t extractRegionScale = 10;
    bool dumpTSV = false;
    bool exportPMTiles = false;
    bool buildPyramid = false;
    bool probDot = false;
    bool cellAnno = false;
    bool spatialMetrics = false;
    bool profileShellSurface = false;
    bool profileOneFactorMask = false;
    bool runSoftFactorMask = false;
    bool runHardFactorMask = false;
    std::string softMaskCompositionGeoJSON;
    std::vector<int32_t> softMaskCompositionFocal;
    bool skipMaskOverlap = false;
    bool skipBoundaries = false;
    uint32_t minComponentSize = 1;
    uint32_t maskMinComponentArea = 5;
    uint32_t maskMinHoleArea = 5;
    std::vector<int32_t> maskMorphology;
    std::vector<int32_t> shellRadii;
    int32_t surfaceDmax = -1;
    uint32_t minPixPerTilePerLabel = 0;
    int32_t smoothTopLabelsRounds = 0;
    bool fillEmptyIslands = false;
    double confusionRes = -1.0;
    std::vector<uint32_t> k2keep;
    int32_t icol_x = -1, icol_y = -1, icol_z = -1;
    int32_t icol_f = -1;
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
    double minTileFactorMass = 10.;
    double maskSimplify = 0.0;
    float pixelResOverride = -1.0f;
    float pixelResZOverride = -1.0f;
    float rasterPixelRes = -1.0f;
    std::string templateGeoJSON;
    std::string templateOutPrefix;
    std::string nullK = "-1";
    std::string nullP = "0";
    float xmin = 0.0f, xmax = -1.0f, ymin = 0.0f, ymax = -1.0f;
    float zmin = std::numeric_limits<float>::quiet_NaN();
    float zmax = std::numeric_limits<float>::quiet_NaN();
    TileOperator::MltPmtilesOptions mltOptions;
    TileOperator::BuildPmtilesPyramidOptions pyramidOptions;
    int32_t threads = 1;
    int32_t debug_ = 0;

    ParamList pl;
    // Main input/output and runtime.
    pl.add_option("in-data", "Input data file", inData)
      .add_option("in-index", "Input index file", inIndex)
      .add_option("in", "Input prefix (equal to --in-data <in>.tsv/.bin --in-index <in>.index)", inPrefix)
      .add_option("binary", "Data file is in binary format", isBinary)
      .add_option("out", "Output prefix", outPrefix)
      .add_option("binary-out", "Output in binary format (merge only)", binaryOut)
      .add_option("K", "Total number of factors in the data", K)
      .add_option("tile-size", "Tile size in the original data", tileSize)
      .add_option("pixel-res-override", "Override the pixel resolution used to map raw float coordinates to integer pixels", pixelResOverride)
      .add_option("pixel-res-z-override", "Override the z pixel resolution used to map raw float 3D coordinates to integer pixels", pixelResZOverride)
      .add_option("raster-pixel-res", "Optional coarser raster resolution in original units for raster-style tile-op commands", rasterPixelRes)
      .add_option("threads", "Number of threads to use", threads)
      .add_option("debug", "Debug", debug_);

    // Basic inspection, conversion, and region query.
    pl.add_option("print-index", "Print the index entries to stdout", printIndex)
      .add_option("dump-tsv", "Dump all records to TSV format", dumpTSV)
      .add_option("export-pmtiles", "Export an input PMTiles archive to TSV plus TileOperator index", exportPMTiles)
      .add_option("build-pyramid", "Build a multi-zoom PMTiles pyramid from a max-zoom MLT PMTiles archive", buildPyramid)
      .add_option("write-mlt-pmtiles", "Write MLT-backed PMTiles", mltOptions.enabled)
      .add_option("min-zoom", "Minimum zoom level for --build-pyramid", pyramidOptions.minZoom)
      .add_option("max-tile-bytes", "Maximum compressed tile bytes for --build-pyramid", pyramidOptions.maxTileBytes)
      .add_option("max-tile-features", "Maximum features per tile for --build-pyramid", pyramidOptions.maxTileFeatures)
      .add_option("scale-factor-compression", "Compression aggressiveness estimate for --build-pyramid", pyramidOptions.scaleFactorCompression)
      .add_option("gene-bin-info", "JSON file with gene/count/bin rows; when provided with --write-mlt-pmtiles, gene-bin PMTiles packaging is activated", mltOptions.gene_bin_info_file)
      .add_option("feature-count-file", "Optional TSV with feature name in column 1 and total count in column 2; together with positive --n-gene-bins this activates gene-bin PMTiles packaging", mltOptions.feature_count_file)
      .add_option("n-gene-bins", "Positive number of gene bins to derive from --feature-count-file; zero disables TSV-derived gene-bin packaging", mltOptions.n_gene_bins)
      .add_option("coord-scale", "Scale factor applied to x/y before MLT-PMTiles export", mltOptions.coordScale)
      .add_option("encode-prob-min", "For MLT PMTiles export, encode P2+ values below this threshold as null together with their K values; negative disables pruning", mltOptions.encode_prob_min)
      .add_option("encode-prob-eps", "For MLT PMTiles export, mark P1 nullable and omit it when P1 > 1-eps; non-positive disables this rule", mltOptions.encode_prob_eps)
      .add_option("pmtiles-zoom", "Web Mercator zoom level for EPSG:3857 MLT-PMTiles export", mltOptions.zoom)
      .add_option("coord-digits", "Number of decimal digits to output for coordinates (for --dump-tsv)", coordDigits)
      .add_option("prob-digits", "Number of decimal digits to output for probabilities (for --dump-tsv)", probDigits)
      .add_option("reorganize", "Reorganize fragmented tiles", reorganize)
      .add_option("extract-region-geojson", "Extract all records inside a GeoJSON Polygon/MultiPolygon region", extractRegionGeoJSON)
      .add_option("extract-region", "Extract all records within --xmin/--xmax/--ymin/--ymax and write a new indexed file pair", extractRegion)
      .add_option("extract-region-scale", "Integer scale for GeoJSON region snapping", extractRegionScale)
      .add_option("xmin", "Minimum x coordinate for --extract-region", xmin)
      .add_option("xmax", "Maximum x coordinate for --extract-region", xmax)
      .add_option("ymin", "Minimum y coordinate for --extract-region", ymin)
      .add_option("ymax", "Maximum y coordinate for --extract-region", ymax)
      .add_option("zmin", "Minimum z coordinate for 3D region extraction ([zmin, zmax))", zmin)
      .add_option("zmax", "Maximum z coordinate for 3D region extraction ([zmin, zmax))", zmax);

    // Merge, join, and aggregation operations.
    pl.add_option("merge-emb", "List of embedding files to merge", inMergeEmbFiles)
      .add_option("emb-prefix", "Optional per-source prefixes for merged K/P TSV columns", mergeEmbPrefixes)
      .add_option("merge-keep-all-main", "Keep all main-input records in --merge-emb and fill missing source slots with (-1, 0)", mergeKeepAllMain)
      .add_option("merge-keep-all", "Keep any pixel observed in at least one source when merging", mergeKeepAll)
      .add_option("k2keep", "Number of factors to keep from each source (merge only)", k2keep)
      .add_option("null-k", "Placeholder printed for missing K values in TSV merge outputs", nullK)
      .add_option("null-p", "Placeholder printed for missing P values in TSV merge outputs", nullP)
      .add_option("annotate-pts", "Prefix of the data file to annotate", inMergePtsPrefix)
      .add_option("anno-keep-all", "Keep all query records in annotate outputs and use placeholders for missing annotations", annoKeepAll)
      .add_option("annotate-cell", "Annotate factor composition per cell and subcellular component", cellAnno)
      .add_option("icol-x", "X coordinate column index, 0-based", icol_x)
      .add_option("icol-y", "Y coordinate column index, 0-based", icol_y)
      .add_option("icol-z", "Z coordinate column index, 0-based", icol_z)
      .add_option("icol-feature", "Feature-name column index, 0-based (required for single-molecule annotate/pix2cell and annotate PMTiles packaging)", icol_f)
      .add_option("icol-count", "Count/value column index, 0-based (required for annotate PMTiles packaging)", mltOptions.icol_count)
      .add_option("ext-col-ints", "Additional integer query columns for PMTiles packaging, in the form of \"idx[:name[:nullval]]\"", mltOptions.ext_col_ints)
      .add_option("ext-col-floats", "Additional float query columns for PMTiles packaging, in the form of \"idx[:name[:nullval]]\"", mltOptions.ext_col_floats)
      .add_option("ext-col-strs", "Additional string query columns for PMTiles packaging, in the form of \"idx[:name[:nullval]]\"", mltOptions.ext_col_strs)
      .add_option("icol-c", "Cell ID column index, 0-based (for pix2cell)", icol_c)
      .add_option("icol-s", "Cell component column index, 0-based (for pix2cell)", icol_s)
      .add_option("k-out", "Number of top factors to output (for pix2cell)", kOut)
      .add_option("max-cell-diameter", "Maximum cell diameter in microns (for pix2cell)", maxCellDiameter);

    // Factor-distribution summaries.
    pl.add_option("prob-dot", "Compute pairwise probability dot products", probDot)
      .add_option("confusion", "Compute confusion matrix using r-by-r squares", confusionRes);

    // Raster-label and spatial profiling operations.
    pl.add_option("smooth-top-labels", "Per-tile island smoothing of top labels (>0 to enable)", smoothTopLabelsRounds)
      .add_option("fill-empty-islands", "Fill empty pixels surrounded by consistent neighbors (for --smooth-top-labels)", fillEmptyIslands)
      .add_option("spatial-metrics", "Compute area/perim metrics for single & pairwise channels", spatialMetrics)
      .add_option("shell-surface", "Compute shell occupancy and directional surface-distance histograms", profileShellSurface)
      .add_option("cc-min-size", "Minimum connected-component size for --shell-surface and --hard-factor-mask", minComponentSize)
      .add_option("shell-radii", "Radii list for --shell-surface (pixel units)", shellRadii)
      .add_option("surface-dmax", "Maximum distance for surface histogram in --shell-surface", surfaceDmax)
      .add_option("spatial-min-pix-per-tile-label", "Only seed a label in a tile if nPixels(label,tile) >= this threshold", minPixPerTilePerLabel);

    // Factor-mask operations
    pl.add_option("mask-radius", "Neighborhood radius in pixels for factor-mask operations", maskRadius)
      .add_option("mask-threshold", "Neighborhood mass-fraction threshold for factor-mask operations", maskThreshold)
      .add_option("mask-min-pixel-prob", "Minimum per-pixel factor probability used when constructing soft masks", maskMinPixelProb)
      .add_option("mask-morphology", "Optional post-threshold morphology steps for factor-mask operations; positive= dilation, negative= erosion, abs(value)= odd kernel size", maskMorphology)
      .add_option("mask-min-component-area", "Minimum per-tile 4-connected component cutoff retained in factor-mask operations", maskMinComponentArea)
      .add_option("skip-boundaries", "Skip GeoJSON export and write only summary tables for hard/soft factor mask commands", skipBoundaries)
      .add_option("template-geojson", "Optional template GeoJSON file used to write one additional per-factor file with replaced geometry and title", templateGeoJSON)
      .add_option("template-out-prefix", "Optional output prefix for per-factor GeoJSON files written from --template-geojson (defaults to --out)", templateOutPrefix)
      .add_option("profile-one-factor-mask", "Build a thresholded neighborhood mask for one factor and report selected pairwise overlaps", profileOneFactorMask)
      .add_option("focal-k", "Focal factor index for --profile-one-factor-mask", focalK)
      .add_option("mask-min-frac", "Minimum focal-mask mass fraction to keep a secondary factor in --profile-one-factor-mask", maskMinFrac)
      .add_option("skip-mask-overlap", "Skip computing mask overlaps with top co-localized factors (for --profile-one-factor-mask)", skipMaskOverlap)
      .add_option("soft-factor-mask", "Build per-factor soft masks, polygonize them, and export merged boundaries as GeoJSON", runSoftFactorMask)
      .add_option("soft-mask-composition", "Read the joined soft-mask GeoJSON and profile factor composition within each mask and globally", softMaskCompositionGeoJSON)
      .add_option("soft-mask-composition-focal", "Optional subset of focal factor IDs to profile from --soft-mask-composition", softMaskCompositionFocal)
      .add_option("mask-min-tile-mass", "Skip factors whose total mass in a tile is below this threshold for --soft-factor-mask", minTileFactorMass)
      .add_option("mask-min-hole-area", "Minimum hole area retained in output polygons for --soft-factor-mask", maskMinHoleArea)
      .add_option("mask-simplify", "Optional simplification tolerance applied to output polygons for --soft-factor-mask", maskSimplify)
      .add_option("hard-factor-mask", "Build per-label hard masks from the top factor and report global summaries", runHardFactorMask);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (exportPMTiles && buildPyramid) {
        error("--export-pmtiles and --build-pyramid are mutually exclusive");
    }

    if (!inPrefix.empty()) {
        if (exportPMTiles || buildPyramid) {
            inData = inPrefix;
        } else {
            inData = inPrefix + (isBinary ? ".bin" : ".tsv");
            inIndex = inPrefix + ".index";
        }
    } else if (inData.empty() || (!(exportPMTiles || buildPyramid) && inIndex.empty())) {
        if (exportPMTiles || buildPyramid) {
            error("Either --in or --in-data must be specified for the selected PMTiles operation");
        }
        error("Either --in or both --in-data and --in-index must be specified");
    }

    if (exportPMTiles) {
        TileOperator::ExportPmtilesOptions exportOptions;
        exportOptions.tileSize = tileSize;
        exportOptions.probDigits = probDigits;
        exportOptions.coordDigits = coordDigits;
        exportOptions.geojsonFile = extractRegionGeoJSON;
        exportOptions.geojsonScale = extractRegionScale;
        exportOptions.xmin = xmin;
        exportOptions.xmax = xmax;
        exportOptions.ymin = ymin;
        exportOptions.ymax = ymax;
        exportOptions.zmin = zmin;
        exportOptions.zmax = zmax;
        TileOperator::exportPMTiles(inData, outPrefix, exportOptions);
        return 0;
    }

    if (buildPyramid) {
        pyramidOptions.threads = threads;
        TileOperator::buildPmtilesPyramid(inData, outPrefix, pyramidOptions);
        return 0;
    }

    TileOperator tileOp(inData, inIndex);
    tileOp.setNullPlaceholders(nullK, nullP);
    if (pixelResZOverride > 0.0f && !(pixelResOverride > 0.0f)) {
        error("--pixel-res-z-override requires --pixel-res-override");
    }
    if (pixelResOverride > 0.0f) {
        tileOp.setPixelResolutionOverride(pixelResOverride, pixelResZOverride);
    }
    if (K > 0) {tileOp.setFactorCount(K);}
    tileOp.setThreads(threads);
    const bool hasFeatureIndex = tileOp.hasFeatureIndex();
    const bool hasRasterPixelResOverride = (rasterPixelRes > 0.0f);
    auto applyRasterPixelResOverride = [&]() {
        if (hasRasterPixelResOverride) {
            tileOp.setRasterPixelResolution(rasterPixelRes);
        }
    };
    int32_t nTiles = tileOp.query(xmin, xmax - 1e-6f, ymin, ymax - 1e-6f);
    if (nTiles >= 0) {
        notice("Operating on %d tiles within the specified range", nTiles);
    }

    if(printIndex) {
        tileOp.printIndex();
    }
    if (outPrefix.empty()) {
        return 0;
    }
    if (debug_ > 0) { // CAUTION
        tileOp.sampleTilesToDebug(debug_);
    }
    if (hasRasterPixelResOverride &&
        !(smoothTopLabelsRounds > 0 || spatialMetrics || profileShellSurface ||
          profileOneFactorMask || runSoftFactorMask || runHardFactorMask)) {
        error("--raster-pixel-res is currently supported only with raster-style tile-op commands");
    }
    mltOptions.enabled = mltOptions.zoom > 0;
    if (mltOptions.enabled) {
        if (mltOptions.encode_prob_min > 1.0) {
            error("--encode-prob-min must be <= 1");
        }
        if (mltOptions.encode_prob_eps > 1.0) {
            error("--encode-prob-eps must be <= 1");
        }
        if (inMergePtsPrefix.empty() && !inMergeEmbFiles.empty()) {
            error("--write-mlt-pmtiles with --merge-emb currently requires --annotate-pts");
        }
        if (mltOptions.zoom > 31) {
            error("--pmtiles-zoom must be between 0 and 31");
        }
    }
    const bool hasExtractRegionZMin = !std::isnan(zmin);
    const bool hasExtractRegionZMax = !std::isnan(zmax);
    if (hasExtractRegionZMin != hasExtractRegionZMax) {
        error("Both --zmin and --zmax must be provided together");
    }
    if ((hasExtractRegionZMin || hasExtractRegionZMax) &&
        !(extractRegion || !extractRegionGeoJSON.empty() || dumpTSV)) {
        error("--zmin/--zmax are only supported with --extract-region, --extract-region-geojson, or --dump-tsv with --extract-region-geojson");
    }
    if ((extractRegion || !extractRegionGeoJSON.empty() || dumpTSV) && hasExtractRegionZMin && !(zmax > zmin)) {
        error("--zmax must be greater than --zmin");
    }
    if (dumpTSV && (hasExtractRegionZMin || hasExtractRegionZMax) && extractRegionGeoJSON.empty()) {
        error("--zmin/--zmax require --extract-region-geojson when used with --dump-tsv");
    }

    if (reorganize) {
        tileOp.reorgTiles(outPrefix, tileSize);
        return 0;
    }

    if (extractRegion) {
        if (!extractRegionGeoJSON.empty()) {
            error("--extract-region and --extract-region-geojson are mutually exclusive");
        }
        tileOp.extractRegion(outPrefix, xmin, xmax - 1e-6f, ymin, ymax - 1e-6f, zmin, zmax);
        return 0;
    }

    if (!extractRegionGeoJSON.empty() && !dumpTSV) {
        tileOp.extractRegionGeoJSON(outPrefix, extractRegionGeoJSON, extractRegionScale, zmin, zmax);
        return 0;
    }

    if (dumpTSV) {
        tileOp.dumpTSV(outPrefix, probDigits, coordDigits,
            extractRegionGeoJSON, extractRegionScale, zmin, zmax, mergeEmbPrefixes);
        return 0;
    }
    if (mltOptions.enabled &&
        inMergePtsPrefix.empty() && inMergeEmbFiles.empty()) {
        tileOp.writeMltPmtiles(outPrefix, mltOptions, k2keep, mergeEmbPrefixes);
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
        if (hasFeatureIndex && icol_f < 0)
            error("valid --icol-feature is required for --annotate-cell on single-molecule input");
        tileOp.pix2cell(inMergePtsPrefix, outPrefix, icol_c, icol_x, icol_y,
            icol_s, icol_z, icol_f, kOut, maxCellDiameter);
        return 0;
    }

    if (!inMergeEmbFiles.empty() && !inMergePtsPrefix.empty()) {
        if (binaryOut) {
            error("--binary-out is not supported when using --merge-emb together with --annotate-pts");
        }
        if (hasFeatureIndex && icol_f < 0)
            error("valid --icol-feature is required for --annotate-pts on single-molecule input");
        tileOp.annotateMerged(inMergeEmbFiles, inMergePtsPrefix,
            outPrefix, k2keep, icol_x, icol_y, icol_z, icol_f,
            mergeKeepAllMain, mergeKeepAll, mergeEmbPrefixes,
            annoKeepAll, mltOptions);
        return 0;
    }

    if (!inMergeEmbFiles.empty()) {
        tileOp.merge(inMergeEmbFiles, outPrefix, k2keep, binaryOut, mergeKeepAllMain, mergeKeepAll, mergeEmbPrefixes);
        return 0;
    }

    if (!inMergePtsPrefix.empty()) {
        if (hasFeatureIndex && icol_f < 0)
            error("valid --icol-feature is required for --annotate-pts on single-molecule input");
        tileOp.annotate(inMergePtsPrefix, outPrefix, icol_x, icol_y, icol_z,
            icol_f, annoKeepAll, mergeEmbPrefixes, mltOptions);
        return 0;
    }

    // Raster-style operations
    if (hasFeatureIndex) {
        error("Raster image based operations do not support single-molecule input");
    }
    if (smoothTopLabelsRounds > 0) {
        applyRasterPixelResOverride();
        tileOp.smoothTopLabels2D(outPrefix, smoothTopLabelsRounds, fillEmptyIslands);
        return 0;
    }
    if (spatialMetrics) {
        applyRasterPixelResOverride();
        tileOp.spatialMetricsBasic(outPrefix);
        return 0;
    }
    if (profileShellSurface) {
        applyRasterPixelResOverride();
        tileOp.profileShellAndSurface(outPrefix, shellRadii, surfaceDmax, minComponentSize, minPixPerTilePerLabel);
        return 0;
    }
    if (profileOneFactorMask) {
        applyRasterPixelResOverride();
        tileOp.profileSoftFactorMasks(outPrefix, focalK, maskRadius,
            maskThreshold, maskMinFrac, maskMinPixelProb, maskMorphology,
            maskMinComponentArea, skipMaskOverlap);
        return 0;
    }
    if (runSoftFactorMask) {
        applyRasterPixelResOverride();
        tileOp.softFactorMask(outPrefix, maskRadius, maskThreshold,
            maskMinPixelProb, maskMorphology, minTileFactorMass, maskMinComponentArea,
            maskMinHoleArea, maskSimplify, skipBoundaries, templateGeoJSON, templateOutPrefix);
        return 0;
    }
    if (!softMaskCompositionGeoJSON.empty()) {
        tileOp.softMaskComposition(outPrefix, softMaskCompositionGeoJSON, softMaskCompositionFocal);
        return 0;
    }
    if (runHardFactorMask) {
        applyRasterPixelResOverride();
        tileOp.hardFactorMask(outPrefix, minComponentSize, skipBoundaries, templateGeoJSON, templateOutPrefix);
        return 0;
    }

    return 0;
}
