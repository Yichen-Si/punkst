#pragma once

#include "punkst.h"
#include "pts2tiles.hpp"
#include "tiles2bins.hpp"
#include <cmath>
#include <limits>

struct Pts2TilesOptions {
    std::string inTsv, outPref, tmpDir;
    int32_t nThreads0 = 1;
    int32_t tileSize = -1;
    int32_t tileBuffer = 1000;
    int32_t batchSize = 10000;
    int32_t icol_x = -1;
    int32_t icol_y = -1;
    int32_t icol_z = -1;
    int32_t icol_feature = -1;
    std::vector<int32_t> icol_ints;
    int32_t nskip = 0;
    bool skip_last_is_header = false;
    bool csv_input = false;
    std::vector<int32_t> keep_quotes;
    bool tile_op_factor_tsv = false;
    double scale = 1.0;
    double scale_x = std::numeric_limits<double>::quiet_NaN();
    double scale_y = std::numeric_limits<double>::quiet_NaN();
    double scale_z = std::numeric_limits<double>::quiet_NaN();
    int32_t digits = 2;
    std::vector<int32_t> include_cols;
    std::vector<int32_t> exclude_cols;

    ParamList& addInputOptions(ParamList& pl,
            bool requireCoords = false,
            bool requireFeature = false,
            bool allowTileOpFactorTsv = true) {
        pl.add_option("in-tsv", "Input delimited text file.", inTsv);
        return addProcessingOptions(pl, requireCoords, requireFeature, allowTileOpFactorTsv);
    }

    ParamList& addProcessingOptions(ParamList& pl,
            bool requireCoords = false,
            bool requireFeature = false,
            bool allowTileOpFactorTsv = true,
            bool includeTempDir = true,
            bool includeThreads = true) {
        pl
          .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x, requireCoords)
          .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y, requireCoords)
          .add_option("icol-z", "Column index for z coordinate (0-based, optional)", icol_z)
          .add_option("icol-feature", "Column index for feature (0-based)", icol_feature, requireFeature)
          .add_option("icol-int", "Column index for integer values (0-based)", icol_ints);
        if (allowTileOpFactorTsv) {
            pl.add_option("tile-op-factor-tsv", "Input is a TileOperator factor-probability TSV with x/y[/z] and K1/P1 columns; write a tile-op compatible index", tile_op_factor_tsv);
        }
        pl.add_option("csv", "Treat the input as comma-delimited CSV instead of tab-delimited text; otherwise inferred from input extension when possible", csv_input)
          .add_option("keep-quotes", "Columns to keep quoted in CSV output (0-based)", keep_quotes)
          .add_option("skip", "Number of lines to skip in the input file (default: 0)", nskip)
          .add_option("skip-last-is-header", "Treat the last skipped line as the header line", skip_last_is_header);
        if (includeTempDir) {
            pl.add_option("temp-dir", "Directory to store temporary files", tmpDir);
        }
        pl
          .add_option("tile-size", "Tile size (in the same unit as the input coordinates)", tileSize)
          .add_option("tile-buffer", "Buffer size per tile per thread (default: 1000 lines)", tileBuffer)
          .add_option("batch-size", "(Only used if the input is gzipped or a stdin stream.) Batch size in terms of the number of lines (default: 10000)", batchSize)
          .add_option("scale", "Uniformly scale x/y/(z) coordinates by this factor unless axis-specific overrides are provided", scale)
          .add_option("scale-x", "Scale factor for the x coordinate", scale_x)
          .add_option("scale-y", "Scale factor for the y coordinate", scale_y)
          .add_option("scale-z", "Scale factor for the z coordinate", scale_z)
          .add_option("digits", "Precision for rewritten output coordinates when scaling is applied (default 2)", digits)
          .add_option("include-cols", "Columns to include in the tiled output (0-based)", include_cols)
          .add_option("exclude-cols", "Columns to exclude from the tiled output (0-based)", exclude_cols);
        if (includeThreads) {
            pl.add_option("threads", "Number of threads to use (default: 1)", nThreads0);
        }
        return pl;
    }

    ParamList& addOutputOptions(ParamList& pl) {
        return pl.add_option("out-prefix", "Output TSV file", outPref);
    }

    void validateStandalone() {
        validateCommon();
        if (tile_op_factor_tsv && (icol_feature >= 0 || !icol_ints.empty())) {
            error("--tile-op-factor-tsv cannot be combined with --icol-feature or --icol-int");
        }
        if (!tile_op_factor_tsv && (icol_x < 0 || icol_y < 0)) {
            error("--icol-x and --icol-y are required unless --tile-op-factor-tsv is used");
        }
    }

    void validateCommon() {
        if (tileSize <= 0) {
            error("Tile size is required to be a positive integer");
        }
        if (skip_last_is_header && nskip <= 0) {
            error("--skip-last-is-header requires --skip to be greater than 0");
        }
        for (int32_t col : keep_quotes) {
            if (col < 0) {
                error("--keep-quotes requires non-negative 0-based column indices");
            }
        }
        if (!include_cols.empty() && !exclude_cols.empty()) {
            error("--include-cols and --exclude-cols are mutually exclusive");
        }
        if (tile_op_factor_tsv && (!include_cols.empty() || !exclude_cols.empty())) {
            error("--include-cols/--exclude-cols cannot be combined with --tile-op-factor-tsv");
        }
        if (inputDelimiter() != ',' && !keep_quotes.empty()) {
            error("--keep-quotes can only be used with CSV input");
        }
        normalizeScales();
    }

    void normalizeScales() {
        if (std::isnan(scale_x)) {
            scale_x = scale;
        }
        if (std::isnan(scale_y)) {
            scale_y = scale;
        }
        if (std::isnan(scale_z)) {
            scale_z = scale;
        }
    }

    unsigned int resolveThreads() const {
        unsigned int nThreads = std::thread::hardware_concurrency();
        if (nThreads == 0 || nThreads >= static_cast<unsigned int>(nThreads0)) {
            nThreads = nThreads0;
        }
        return nThreads;
    }

    bool streamingInput() const {
        return inTsv == "-" || (inTsv.size() > 3 && inTsv.compare(inTsv.size() - 3, 3, ".gz") == 0);
    }

    static size_t countDelimiterOutsideQuotes(const std::string& line, char delim) {
        size_t count = 0;
        bool inQuotes = false;
        for (size_t i = 0; i < line.size(); ++i) {
            const char c = line[i];
            if (c == '"') {
                if (inQuotes && i + 1 < line.size() && line[i + 1] == '"') {
                    ++i;
                    continue;
                }
                inQuotes = !inQuotes;
                continue;
            }
            if (!inQuotes && c == delim) {
                ++count;
            }
        }
        return count;
    }

    bool readDelimiterProbeLine(std::string& line) const {
        if (inTsv.empty() || inTsv == "-") {
            return false;
        }
        TextLineReader reader(inTsv);
        int32_t skipped = 0;
        while (reader.getline(line)) {
            if (skipped < nskip) {
                ++skipped;
                continue;
            }
            if (line.empty() || line[0] == '#') {
                continue;
            }
            return true;
        }
        return false;
    }

    char inputDelimiter() const {
        if (csv_input) {
            return ',';
        }
        char delim = '\t';
        bool inferred = false;
        if (path_has_suffix_ci(inTsv, ".csv") || path_has_suffix_ci(inTsv, ".csv.gz")) {
            delim = ',';
            inferred = true;
        } else if (path_has_suffix_ci(inTsv, ".tsv") || path_has_suffix_ci(inTsv, ".tsv.gz")) {
            delim = '\t';
            inferred = true;
        }
        std::string line;
        if (readDelimiterProbeLine(line)) {
            const size_t commas = countDelimiterOutsideQuotes(line, ',');
            const size_t tabs = countDelimiterOutsideQuotes(line, '\t');
            if (!inferred) {
                return commas > tabs ? ',' : '\t';
            }
            const size_t expected = delim == ',' ? commas : tabs;
            const size_t other = delim == ',' ? tabs : commas;
            if (expected == 0 && other > 0) {
                error("Input delimiter inferred from extension as %s, but first data line looks %s-delimited: %s",
                    delim == ',' ? "CSV" : "TSV",
                    delim == ',' ? "tab" : "comma",
                    inTsv.c_str());
            }
        }
        return delim;
    }

    Pts2Tiles makeRunner(unsigned int nThreads) {
        return Pts2Tiles(nThreads, inTsv, tmpDir, outPref, tileSize,
            icol_x, icol_y, icol_z, icol_feature, icol_ints, nskip,
            skip_last_is_header, streamingInput(), tileBuffer, batchSize,
            scale_x, scale_y, scale_z, digits, inputDelimiter(), tile_op_factor_tsv,
            include_cols, exclude_cols, keep_quotes);
    }

    int32_t remapColumn(int32_t col, bool includeImplicitCount = false) const {
        if (col < 0) {
            return -1;
        }
        if (include_cols.empty() && exclude_cols.empty()) {
            return col;
        }
        std::set<int32_t> forced;
        forced.insert(icol_x);
        forced.insert(icol_y);
        if (icol_z >= 0) forced.insert(icol_z);
        if (icol_feature >= 0) forced.insert(icol_feature);
        for (const auto& c : icol_ints) forced.insert(c);
        std::set<int32_t> includeSet(include_cols.begin(), include_cols.end());
        std::set<int32_t> excludeSet(exclude_cols.begin(), exclude_cols.end());
        int32_t out = 0;
        for (int32_t i = 0; i <= col; ++i) {
            bool keep = include_cols.empty() ? excludeSet.find(i) == excludeSet.end() : includeSet.find(i) != includeSet.end();
            keep = keep || forced.find(i) != forced.end();
            if (keep) {
                if (i == col) {
                    return out;
                }
                ++out;
            }
        }
        if (includeImplicitCount && icol_ints.empty()) {
            return out;
        }
        return -1;
    }

    std::vector<int32_t> remapIntColumns() const {
        std::vector<int32_t> out;
        for (const auto& c : icol_ints) {
            out.push_back(remapColumn(c));
        }
        return out;
    }
};

struct Tiles2HexOptions {
    std::string inTsv, inIndex, outFile, tmpDir, dictFile;
    std::vector<std::string> anchorFiles;
    std::vector<float> radius;
    int32_t nThreads = 1;
    int32_t seed = -1;
    int32_t icol_x = -1;
    int32_t icol_y = -1;
    int32_t icol_z = -1;
    int32_t icol_feature = -1;
    double hexSize = -1.0;
    double hexGridDist = -1.0;
    double bccSize = -1.0;
    double bccGridDist = -1.0;
    std::vector<int32_t> icol_ints;
    std::vector<int32_t> min_counts;
    bool noBackground = false;
    std::vector<double> boundingBoxes;
    bool randomize_output = false;
    std::string sort_mem;
    bool use_internal_sort = false;

    ParamList& addStandaloneOptions(ParamList& pl) {
        pl.add_option("in-tsv", "Input TSV file. Header must begin with #", inTsv)
          .add_option("in-index", "Input index file", inIndex)
          .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x)
          .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y)
          .add_option("icol-z", "Column index for z coordinate (0-based, enables 3D BCC aggregation)", icol_z)
          .add_option("icol-feature", "Column index for feature (0-based)", icol_feature)
          .add_option("feature-dict", "If feature column is not integer, provide the list of feature names", dictFile)
          .add_option("icol-int", "Column index for integer values (0-based)", icol_ints)
          .add_option("bounding-boxes", "Rectangular query regions (xmin ymin xmax ymax)*", boundingBoxes)
          .add_option("anchor-files", "Anchor files", anchorFiles)
          .add_option("radius", "Radius for each set of anchors", radius)
          .add_option("hex-size", "Hexagon size (size length)", hexSize)
          .add_option("hex-grid-dist", "Hexagon grid distance (center-to-center distance)", hexGridDist)
          .add_option("bcc-size", "BCC lattice size for 3D aggregation", bccSize)
          .add_option("bcc-grid-dist", "BCC grid distance (nearest center-to-center distance) for 3D aggregation", bccGridDist)
          .add_option("temp-dir", "Directory to store temporary files", tmpDir)
          .add_option("seed", "Random seed for randomized output keys", seed)
          .add_option("threads", "Number of threads to use (default: 1)", nThreads);
        return pl.add_option("out", "Output TSV file", outFile)
          .add_option("randomize", "Randomize output order", randomize_output)
          .add_option("sort-mem", "Memory to use for sorting, with units K, M, or G similar to -S in linux sort", sort_mem)
          .add_option("use-internal-sort", "Use internal sort instead of system sort command for randomization (default: false)", use_internal_sort)
          .add_option("min-count", "Minimum count for each integer column, applied with OR", min_counts)
          .add_option("ignore-background", "Ignore pixels not within radius of any of the anchors", noBackground);
    }

    ParamList& addMultisampleOptions(ParamList& pl,
            std::vector<double>& hexSizes,
            std::vector<double>& hexGridDists,
            std::vector<double>& bccSizes,
            std::vector<double>& bccGridDists,
            std::string& anchorListFile) {
        return pl.add_option("hex-size", "Hexagon size(s) (size length)", hexSizes)
          .add_option("hex-grid-dist", "Hexagon grid distance(s) (center-to-center distance)", hexGridDists)
          .add_option("bcc-size", "BCC lattice size(s) for 3D aggregation", bccSizes)
          .add_option("bcc-grid-dist", "BCC grid distance(s) (nearest center-to-center distance) for 3D aggregation", bccGridDists)
          .add_option("anchor-files", "Anchor files (one for each sample)", anchorFiles)
          .add_option("anchor-files-list", "A file containing the path to the anchor file for each sample", anchorListFile)
          .add_option("radius", "Radius for each set of anchors", radius)
          .add_option("min-count", "Minimum count for a unit to be included in output", min_counts)
          .add_option("ignore-background", "Ignore pixels not within radius of any of the anchors", noBackground)
          .add_option("seed", "Random seed for randomized output keys", seed)
          .add_option("sort-mem", "Memory to use for sorting, with units K, M, or G similar to -S in linux sort", sort_mem)
          .add_option("use-internal-sort", "Use internal sort instead of system sort command for randomization (default: false)", use_internal_sort);
    }

    int32_t minCount() const {
        return min_counts.empty() ? 1 : min_counts.front();
    }

    bool use3D() const {
        return icol_z >= 0;
    }

    void validateColumns() const {
        if (icol_x < 0 || icol_y < 0) {
            error("--icol-x and --icol-y are required");
        }
        if (icol_feature < 0) {
            error("--icol-feature is required");
        }
    }

    void resolveStandaloneGrid() {
        resolveGrid(use3D(), hexSize, hexGridDist, bccSize, bccGridDist, anchorFiles.empty());
    }

    HexGrid makeGrid() const {
        return HexGrid(use3D() ? bccSize : hexSize);
    }

    std::vector<Rectangle<double>> parseBoundingBoxes() const {
        std::vector<Rectangle<double>> rects;
        if (!boundingBoxes.empty()) {
            int32_t nrects = parseCoordsToRects(rects, boundingBoxes);
            if (nrects <= 0) {
                error("Error parsing bounding boxes");
            }
            notice("Received %d bounding boxes", nrects);
        }
        return rects;
    }

    lineParser makeParser(std::vector<Rectangle<double>>* rects = nullptr,
            bool allowImplicitCount = true) const {
        lineParser parser(icol_x, icol_y, icol_feature, icol_ints, dictFile, rects, allowImplicitCount);
        if (use3D()) {
            parser.setZ(static_cast<size_t>(icol_z));
        }
        return parser;
    }

    void sortOutput(bool fatal) const {
        if (use_internal_sort) {
            size_t maxMemBytes = ExternalSorter::parseMemoryString(sort_mem);
            try {
                ExternalSorter::sortBy1stColHex(outFile, outFile, maxMemBytes, tmpDir, nThreads);
            } catch (const std::exception& e) {
                if (fatal) {
                    error("Error sorting output %s: %s", outFile.c_str(), e.what());
                }
                warning("Error shuffling output %s: %s", outFile.c_str(), e.what());
            }
        } else {
            std::vector<std::string> sort_flags = {"-k1,1", "-o", outFile};
            if (!sort_mem.empty()) {
                sort_flags.insert(sort_flags.begin() + 1, "-S");
                sort_flags.insert(sort_flags.begin() + 2, sort_mem);
            }
            #if defined(__linux__)
            sort_flags.push_back("--parallel=" + std::to_string(nThreads));
            #endif
            if (sys_sort(outFile.c_str(), nullptr, sort_flags) != 0) {
                if (fatal) {
                    error("Error sorting output %s", outFile.c_str());
                }
                warning("Error shuffling output %s", outFile.c_str());
            }
        }
    }

    static void resolveGrid(bool use3D,
            double& hexSize,
            double& hexGridDist,
            double& bccSize,
            double& bccGridDist,
            bool noAnchors) {
        if (use3D) {
            if (!noAnchors) {
                error("Anchor-based aggregation is currently only supported for 2D input");
            }
            if (bccSize > 0 && bccGridDist > 0) {
                warning("If both --bcc-size and --bcc-grid-dist are specified, only --bcc-size will be used");
            }
            if (bccSize <= 0) {
                if (bccGridDist <= 0) {
                    error("3D input requires --bcc-size or --bcc-grid-dist");
                } else {
                    bccSize = 2.0 * bccGridDist / sqrt(3.0);
                }
            }
            if (hexSize > 0 || hexGridDist > 0) {
                warning("Ignoring --hex-size/--hex-grid-dist for 3D input; using --bcc-size/--bcc-grid-dist");
            }
        } else {
            if (bccSize > 0 || bccGridDist > 0) {
                warning("Ignoring --bcc-size/--bcc-grid-dist for 2D input");
            }
            if (hexSize <= 0) {
                if (hexGridDist <= 0) {
                    error("Hexagon size or hexagon grid distance must be specified");
                } else {
                    hexSize = hexGridDist / sqrt(3.0);
                }
            }
        }
    }

    static void resolveGridVectors(bool use3D,
            std::vector<double>& hexSizes,
            std::vector<double>& hexGridDists,
            std::vector<double>& bccSizes,
            std::vector<double>& bccGridDists,
            bool noAnchors) {
        if (use3D) {
            if (!noAnchors) {
                error("Anchor-based aggregation is currently only supported for 2D input");
            }
            if (!hexSizes.empty() || !hexGridDists.empty()) {
                warning("Ignoring --hex-size/--hex-grid-dist for 3D input; using --bcc-size/--bcc-grid-dist");
            }
            if (!bccSizes.empty() && !bccGridDists.empty()) {
                warning("If both --bcc-size and --bcc-grid-dist are specified, only --bcc-size will be used");
            }
            if (bccSizes.empty()) {
                if (bccGridDists.empty()) {
                    error("3D input requires --bcc-size or --bcc-grid-dist");
                }
                for (auto& v : bccGridDists) {
                    bccSizes.push_back(2.0 * v / sqrt(3.0));
                }
            }
            return;
        }
        if (!bccSizes.empty() || !bccGridDists.empty()) {
            warning("Ignoring --bcc-size/--bcc-grid-dist for 2D input");
        }
        if (!hexGridDists.empty() && !hexSizes.empty()) {
            warning("If both --hex-grid-dist and --hex-size are specified, only --hex-size will be used");
        }
        if (!hexGridDists.empty() && hexSizes.empty()) {
            for (auto& v : hexGridDists) {
                hexSizes.push_back(v / sqrt(3.0));
            }
        }
    }
};
