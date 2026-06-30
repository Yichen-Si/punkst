#include "punkst.h"
#include "multi_cde_pixel_common.hpp"
#include "region_query.hpp"
#include "tileoperator.hpp"
#include "utils.h"
#include "glm.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <memory>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace {

struct PixelContrastDesign {
    std::vector<std::string> annoPrefixes;
    std::vector<std::string> ptsPrefixes;
    std::vector<std::string> regionFiles;
    std::vector<std::string> labels;
    std::vector<std::string> confusionFiles;
    std::vector<ContrastDef> contrasts;
    bool hasRegion = false;
};

struct ResolvedAnnoInput {
    std::string data;
    std::string index;
};

struct LocalAgg {
    MultiSlicePairwiseBinom stat;
    MultiSliceUnitCache cache;
    PairwiseBinomRobust statu;
    LocalAgg(int K, int G, int M, double minCount)
        : stat(K, G, M, minCount), cache(K, M, minCount), statu(G, M, minCount) {}
};

struct RegionLocalAgg {
    MultiSlicePairwiseBinom stat;
    MultiSliceUnitCache cache;
    PairwiseBinomRobust statu;
    Eigen::MatrixXd confusion;
    double residual = 0.0;

    RegionLocalAgg(int K, int G, int M, double minCount)
        : stat(K, G, M, minCount),
          cache(K, M, minCount),
          statu(G, M, minCount),
          confusion(Eigen::MatrixXd::Zero(K, K)) {}
};

bool hasSuffix(const std::string& s, const std::string& suffix) {
    return s.size() >= suffix.size() &&
           s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

ResolvedAnnoInput resolveAnnoInput(const std::string& anno, bool isBinary) {
    ResolvedAnnoInput out;
    if (hasSuffix(anno, ".bin")) {
        out.data = anno;
        out.index = anno.substr(0, anno.size() - 4) + ".index";
    } else if (hasSuffix(anno, ".tsv")) {
        out.data = anno;
        out.index = anno.substr(0, anno.size() - 4) + ".index";
    } else {
        out.data = anno + (isBinary ? ".bin" : ".tsv");
        out.index = anno + ".index";
    }
    return out;
}

int32_t findHeaderColumn(const std::vector<std::string>& header,
                         const std::vector<std::string>& names) {
    int32_t out = -1;
    for (size_t i = 0; i < header.size(); ++i) {
        for (const auto& name : names) {
            if (header[i] != name) {
                continue;
            }
            if (out >= 0) {
                error("Contrast file has multiple columns matching '%s'", name.c_str());
            }
            out = static_cast<int32_t>(i);
        }
    }
    return out;
}

void splitContrastLine(std::vector<std::string>& tokens, const std::string& line,
                       bool whitespaceMode) {
    if (whitespaceMode) {
        split(tokens, "\t ", line, UINT_MAX, true, true, true);
    } else {
        split(tokens, "\t", line, UINT_MAX, true, false, true);
    }
}

void finalizeContrastDefs(std::vector<ContrastDef>& contrasts, int32_t n_samples) {
    if (n_samples == 0) {
        error("Contrast file has a header but no samples");
    }
    for (auto& contrast : contrasts) {
        if (contrast.labels.size() != static_cast<size_t>(n_samples)) {
            error("Contrast %s has %zu labels, expected %d samples",
                  contrast.name.c_str(), contrast.labels.size(), n_samples);
        }
        contrast.group_neg.clear();
        contrast.group_pos.clear();
        for (int32_t s = 0; s < n_samples; ++s) {
            const int8_t y = contrast.labels[s];
            if (y < 0) {
                contrast.group_neg.push_back(s);
            } else if (y > 0) {
                contrast.group_pos.push_back(s);
            }
        }
        if (contrast.group_neg.empty() || contrast.group_pos.empty()) {
            error("Contrast %s must have at least one sample in each group", contrast.name.c_str());
        }
        notice("Read contrast %s with %zu vs %zu samples",
               contrast.name.c_str(), contrast.group_pos.size(), contrast.group_neg.size());
    }
}

PixelContrastDesign readContrastDesignFile(const std::string& contrastFile) {
    PixelContrastDesign design;
    std::ifstream ifs(contrastFile);
    if (!ifs.is_open()) {
        error("Cannot open contrast file: %s", contrastFile.c_str());
    }
    std::string line;
    std::vector<std::string> tokens;
    if (!std::getline(ifs, line)) {
        error("Contrast file is empty: %s", contrastFile.c_str());
    }
    splitContrastLine(tokens, line, false);
    if (tokens.size() < 3) {
        error("Contrast file must have >= 3 columns (anno, pts, [region], [label], [confusion], contrast...): %s", contrastFile.c_str());
    }
    bool has_empty_header_token = false;
    bool has_region_header = false;
    for (const auto& token : tokens) {
        if (token.empty()) {
            has_empty_header_token = true;
        } else if (token == "region") {
            has_region_header = true;
        }
    }
    const bool whitespaceMode = !has_region_header && !has_empty_header_token;
    if (whitespaceMode) {
        splitContrastLine(tokens, line, true);
    }
    const size_t n_header_cols = tokens.size();
    const int32_t anno_col = findHeaderColumn(tokens, {"anno", "anno_prefix"});
    const int32_t pts_col = findHeaderColumn(tokens, {"pts", "pts_prefix"});
    const int32_t region_col = findHeaderColumn(tokens, {"region"});
    const int32_t label_col = findHeaderColumn(tokens, {"label"});
    const int32_t confusion_col = findHeaderColumn(tokens, {"confusion"});
    if (anno_col < 0 || pts_col < 0) {
        error("Contrast file must include anno and pts columns: %s", contrastFile.c_str());
    }
    design.hasRegion = (region_col >= 0);

    std::unordered_set<int32_t> meta_cols = {anno_col, pts_col};
    if (region_col >= 0) {
        meta_cols.insert(region_col);
    }
    if (label_col >= 0) {
        meta_cols.insert(label_col);
    }
    if (confusion_col >= 0) {
        meta_cols.insert(confusion_col);
    }

    std::vector<int32_t> contrast_cols;
    for (size_t i = 0; i < n_header_cols; ++i) {
        if (meta_cols.count(static_cast<int32_t>(i)) != 0) {
            continue;
        }
        contrast_cols.push_back(static_cast<int32_t>(i));
    }
    if (contrast_cols.empty()) {
        error("Contrast file must include at least one contrast column: %s", contrastFile.c_str());
    }
    design.contrasts.assign(contrast_cols.size(), ContrastDef{});
    std::unordered_set<std::string> seen_names;
    for (size_t c = 0; c < contrast_cols.size(); ++c) {
        const std::string name = tokens[contrast_cols[c]];
        if (name.empty()) {
            warning("Contrast %zu has an empty header name; using index instead", c);
            design.contrasts[c].name = std::to_string(c);
        } else {
            if (!seen_names.insert(name).second) {
                error("Contrast name '%s' appears more than once in header", name.c_str());
            }
            design.contrasts[c].name = name;
        }
    }

    int32_t n_samples = 0;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        splitContrastLine(tokens, line, whitespaceMode);
        if (tokens.size() != n_header_cols) {
            error("Contrast file row has %zu columns, expected %zu: %s",
                  tokens.size(), n_header_cols, line.c_str());
        }
        if (tokens[anno_col].empty()) {
            error("Contrast file row must have a non-empty anno column: %s", line.c_str());
        }
        design.annoPrefixes.push_back(tokens[anno_col]);
        design.ptsPrefixes.push_back(tokens[pts_col]);
        if (design.hasRegion) {
            if (tokens[region_col].empty()) {
                error("Contrast file row must have a non-empty region column in region-aware mode: %s", line.c_str());
            }
            design.regionFiles.push_back(tokens[region_col]);
        }
        if (label_col >= 0 && !tokens[label_col].empty()) {
            design.labels.push_back(tokens[label_col]);
        } else {
            design.labels.push_back(std::to_string(n_samples));
        }
        if (confusion_col >= 0) {
            design.confusionFiles.push_back(tokens[confusion_col]);
        }
        for (size_t c = 0; c < contrast_cols.size(); ++c) {
            int32_t y = 0;
            const std::string& token = tokens[contrast_cols[c]];
            if (!str2int32(token, y)) {
                error("Contrast value must be -1, 0, or 1: %s", token.c_str());
            }
            if (y < -1 || y > 1) {
                error("Contrast value must be -1, 0, or 1: %d", y);
            }
            design.contrasts[c].labels.push_back(static_cast<int8_t>(y));
        }
        n_samples++;
    }
    if (!design.hasRegion) {
        design.regionFiles.clear();
    }
    finalizeContrastDefs(design.contrasts, n_samples);
    return design;
}

} // namespace

/**
 * Join pixel level decoding results with original transcripts and perform
 * cluster/factor specific (conditional) DE test between groups
 */
int32_t cmdConditionalTest(int32_t argc, char** argv) {
    std::vector<std::string> inPrefix, inData, inIndex, inPtsPrefix, inConfusion;
    std::vector<std::string> dataLabels;
    std::string dictFile, outPrefix, outFile, contrastFile;
    std::string auxiSuff;
    PixelDETestOptions testOpts;
    bool isBinary = false;
    double gridSize;
    int32_t K = 0;
    int32_t icol_x = 0, icol_y = 1, icol_feature = 2, icol_val = 3;
    int64_t regionScale = 10;
    double minCount = 10.0;
    double minProb = 0.01;
    int32_t debug_ = 0;
    int32_t nThreads = 1;
    std::vector<ContrastDef> contrasts;
    bool hasRegionDesign = false;
    std::vector<std::string> regionFiles;

    ParamList pl;
    pl.add_option("anno-data", "Input pixel files", inData)
      .add_option("anno-index", "Input pixel index files", inIndex)
      .add_option("anno", "Prefixes of input pixel data files", inPrefix)
      .add_option("labels", "Labels for the datasets to show in pairwise output", dataLabels)
      .add_option("binary", "Data file is in binary format", isBinary)
      .add_option("K", "Number of factors in the annotation files; optional for indexed binary inputs with K in the header", K)
      .add_option("pts", "Prefixes of the transcript data files", inPtsPrefix)
      .add_option("contrast", "Contrast design TSV (anno, pts, [region], [label], [confusion], contrasts)", contrastFile)
      .add_option("confusion", "Per-dataset confusion matrices (TSV with K+1 rows/cols)", inConfusion)
      .add_option("features", "List of features to test; optional for single-molecule annotation inputs with embedded feature names", dictFile)
      .add_option("grid-size", "Grid size", gridSize, true)
      .add_option("pseudo-rel", "Relative pseudo count fraction w.r.t. null", testOpts.pseudoFracRel)
      .add_option("icol-x", "Column index for x coordinate for files in --pts (0-based)", icol_x)
      .add_option("icol-y", "Column index for y coordinate for files in --pts (0-based)", icol_y)
      .add_option("icol-feature", "Column index for feature for files in --pts (0-based)", icol_feature)
      .add_option("icol-val", "Column index for count/value for files in --pts (0-based)", icol_val)
      .add_option("region-scale", "Integer coordinate scale for GeoJSON preprocessing", regionScale)
      .add_option("threads", "Number of threads to use", nThreads)
      .add_option("out", "Output prefix", outPrefix, true)
      .add_option("aux-suff", "Suffix for auxiliary output files", auxiSuff)
      .add_option("min-prob", "Minimum probability for a pixel to be included (softly) in a slice", minProb)
      .add_option("min-count-per-feature", "Minimum total count for a feature to be considered", testOpts.minCountPerFeature)
      .add_option("max-pval", "Max p-value for output (default: 1)", testOpts.minPvalOutput)
      .add_option("max-pval-deconv", "If at least two slices reach this p-value, do deconvolution", testOpts.deconvHitP)
      .add_option("min-or", "Minimum odds ratio for output (default: 1)", testOpts.minOROutput)
      .add_option("min-or-perm", "Minimum odds ratio for doing permutation (default: 1.2)", testOpts.minORPerm)
      .add_option("min-count", "Minimum observed factor-specific count for a unit to be included (default: 10)", minCount)
      .add_option("perm", "Number of permutations for beta calibration (two-sample only)", testOpts.nPerm)
      .add_option("seed", "Seed for permutations", testOpts.seed)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }
    if (debug_ > 0) {
        logger::Logger::getInstance().setLevel(logger::LogLevel::DEBUG);
    }
    if (nThreads < 1) {
        nThreads = 1;
    }
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nThreads);

    // Collect input files and contrasts
    if (!contrastFile.empty()) {
        if (!inPrefix.empty() || !inData.empty() || !inIndex.empty()
                || !inPtsPrefix.empty() || !inConfusion.empty()) {
            error("--contrast cannot be combined with --anno/--anno-data/--anno-index/--pts/--confusion");
        }
        PixelContrastDesign design = readContrastDesignFile(contrastFile);
        inPrefix = std::move(design.annoPrefixes);
        inPtsPrefix = std::move(design.ptsPrefixes);
        inConfusion = std::move(design.confusionFiles);
        dataLabels = std::move(design.labels);
        regionFiles = std::move(design.regionFiles);
        contrasts = std::move(design.contrasts);
        hasRegionDesign = design.hasRegion;
    }
    if (!inPrefix.empty()) {
        inData.resize(inPrefix.size());
        inIndex.resize(inPrefix.size());
        for (size_t i = 0; i < inPrefix.size(); ++i) {
            ResolvedAnnoInput resolved = resolveAnnoInput(inPrefix[i], isBinary);
            inData[i] = std::move(resolved.data);
            inIndex[i] = std::move(resolved.index);
        }
    } else if (inData.empty() || inIndex.empty() || inData.size() != inIndex.size()) {
        error("Either --anno or both --anno-data and --anno-index must be specified");
    }
    uint32_t n_data = static_cast<uint32_t>(inData.size());
    if (hasRegionDesign && regionFiles.size() != n_data) {
        error("Region-aware contrast design must have one region per input row");
    }
    if (!inConfusion.empty() && inConfusion.size() != n_data) {
        error("--confusion must have %u entries to match datasets", n_data);
    }
    dataLabels.resize(n_data);
    for (uint32_t i = 0; i < n_data; ++i) {
        if (dataLabels[i].empty())
            dataLabels[i] = std::to_string(i);
    }
    if (contrastFile.empty()) {
        buildPairwiseContrasts(dataLabels, contrasts);
    }
    if (contrasts.empty()) {
        error("No contrasts defined");
    }

    if (inPtsPrefix.size() < n_data) {
        inPtsPrefix.resize(n_data);
    } else if (inPtsPrefix.size() != n_data) {
        error("--pts must match with --anno/--anno-data");
    }
    if (gridSize <= 0) {error("--grid-size must be positive");}

    std::vector<std::unique_ptr<TileOperator>> tileOps;
    for (uint32_t i = 0; i < n_data; ++i) {
        tileOps.emplace_back(std::make_unique<TileOperator>(inData[i], inIndex[i], "", nThreads));
    }
    std::vector<uint8_t> isFeatureInput(n_data, 0);
    for (uint32_t i = 0; i < n_data; ++i) {
        isFeatureInput[i] = tileOps[i]->hasFeatureIndex() ? 1 : 0;
        const int32_t headerK = tileOps[i]->getFactorCount();
        if (K <= 0 && headerK > 0) {
            K = headerK;
        }
        if (K > 0 && headerK > 0 && headerK != K) {
            error("Annotation input %u has K=%d in the index header, expected %d",
                  i, headerK, K);
        }
    }
    if (K <= 0) {
        error("--K must be specified when it cannot be read from annotation index headers");
    }

    std::vector<std::unique_ptr<TileReader>> readers(n_data);
    for (uint32_t i = 0; i < n_data; ++i) {
        if (isFeatureInput[i]) {
            continue;
        }
        if (inPtsPrefix[i].empty()) {
            error("--pts/pts column is required for pixel-mode annotation input %u", i);
        }
        readers[i] = std::make_unique<TileReader>(inPtsPrefix[i] + ".tsv", inPtsPrefix[i] + ".index");
        if (readers[i]->getTileSize() != tileOps[i]->getTileSize()) {
            error("Currently we require the tile size to be the same for each pair of annotation and transcript data. Invalid dataset %u (%u vs %u)", i, tileOps[i]->getTileSize(), readers[i]->getTileSize());
        }
    }

    std::vector<std::string> featureList;
    if (!dictFile.empty()) {
        lineParserUnival featureParser(icol_x, icol_y, icol_feature, icol_val, dictFile);
        featureParser.getFeatureList(featureList);
    } else {
        for (uint32_t i = 0; i < n_data; ++i) {
            if (!isFeatureInput[i]) {
                error("--features is required for pixel-mode annotation input %u", i);
            }
            if (!tileOps[i]->getFeatureNames().empty()) {
                featureList = tileOps[i]->getFeatureNames();
                break;
            }
        }
    }
    int32_t M = static_cast<int32_t>(featureList.size());
    if (M == 0) {error("No features found");}
    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val, dictFile);
    std::unordered_map<std::string, int32_t> featureIndex;
    for (int32_t m = 0; m < M; ++m) {
        featureIndex[featureList[m]] = m;
    }
    std::vector<std::vector<int32_t>> featureRemaps(n_data);
    for (uint32_t i = 0; i < n_data; ++i) {
        if (!isFeatureInput[i]) {
            continue;
        }
        const auto& annoFeatures = tileOps[i]->getFeatureNames();
        featureRemaps[i].assign(annoFeatures.size(), -1);
        int32_t nMapped = 0;
        for (size_t f = 0; f < annoFeatures.size(); ++f) {
            auto it = featureIndex.find(annoFeatures[f]);
            if (it == featureIndex.end()) {
                continue;
            }
            featureRemaps[i][f] = it->second;
            nMapped++;
        }
        if (nMapped == 0) {
            error("No embedded annotation features map to the requested feature list for input %u", i);
        }
        notice("Input %u: mapped %d / %zu embedded features", i, nMapped, annoFeatures.size());
    }

    MultiSlicePairwiseBinom statOp(K, n_data, M, minCount);
    PairwiseBinomRobust statUnion(n_data, M, minCount);
    MultiSliceUnitCache unitCache(K, M, minCount);

    std::vector<uint8_t> has_confusion(n_data, 0);
    if (!inConfusion.empty()) {
        int32_t n_has_confusion = 0;
        for (uint32_t i = 0; i < n_data; ++i) {
            if (inConfusion[i].empty()) {
                continue;
            }
            Eigen::MatrixXd confusion = readConfusionMatrixFile(inConfusion[i], K);
            statOp.add_to_confusion(static_cast<int>(i), confusion);
            has_confusion[i] = 1;
            n_has_confusion++;
        }
        notice("Loaded %d pre-computed confusion matrices", n_has_confusion);
    }

    if (hasRegionDesign) {
        for (uint32_t i = 0; i < n_data; ++i) {
            const bool use_confusion = (has_confusion[i] != 0);
            auto& tileOp = *tileOps[i];
            PreparedRegionMask2D region = loadPreparedRegionGeoJSON(regionFiles[i], tileOp.getTileSize(), regionScale);
            if (region.empty()) {
                error("Region is empty after preprocessing for input row %u: %s", i, regionFiles[i].c_str());
            }
            const float pixelResolution = tileOp.getPixelResolution() > 0.0f ? tileOp.getPixelResolution() : 1.0f;
            const PreparedRegionRasterMask2D mask = prepareRegionRasterMask2D(region, pixelResolution);
            std::vector<TileKey> tileList;
            std::unordered_map<TileKey, RegionTileState, TileKeyHash> featureTileStates;
            if (isFeatureInput[i]) {
                for (const auto& tileInfo : tileOp.getTileInfo()) {
                    TileKey tile{tileInfo.row, tileInfo.col};
                    RegionTileState tileState = region.classifyTile(tile);
                    if (tileState != RegionTileState::Outside) {
                        tileList.push_back(tile);
                        featureTileStates.emplace(tile, tileState);
                    }
                }
            } else {
                auto& reader = *readers[i];
                std::vector<Rectangle<double>> queryRects;
                queryRects.emplace_back(region.bbox_f.xmin, region.bbox_f.ymin,
                                        region.bbox_f.xmax, region.bbox_f.ymax);
                std::vector<bool> isContained;
                reader.getTileList(queryRects, tileList, isContained);
            }
            std::sort(tileList.begin(), tileList.end());
            tileList.erase(std::unique(tileList.begin(), tileList.end()), tileList.end());
            if (tileList.empty()) {
                error("No transcript tiles intersect region bounding box for input row %u: %s",
                      i, regionFiles[i].c_str());
            }
            const std::vector<Rectangle<float>> pixelRects = {region.bbox_f};
            notice("Processing %zu region tiles for input row %u (%s)", tileList.size(), i, dataLabels[i].c_str());
            std::atomic<int32_t> processed{0};
            tbb::enumerable_thread_specific<RegionLocalAgg> tls([&] {
                return RegionLocalAgg(K, (int)n_data, M, minCount);
            });
            size_t ntasks = tileList.size();
            if (debug_ > 0) {
                ntasks = std::min<size_t>(ntasks, static_cast<size_t>(debug_ * nThreads));
            }
            tbb::parallel_for(tbb::blocked_range<size_t>(0, ntasks),
                [&](const tbb::blocked_range<size_t>& range) {
                    auto& local = tls.local();
                    std::ifstream tileStream;
                    for (size_t ti = range.begin(); ti != range.end(); ++ti) {
                        const TileKey tile = tileList[ti];
                        if (isFeatureInput[i]) {
                            std::vector<PixTopProbsFeature<float>> records;
                            tileOp.loadTileFeatureRecords(tile, records, &tileStream);
                            if (records.empty()) {
                                const int32_t done = processed.fetch_add(1) + 1;
                                if (done % 10 == 0) {
                                    notice("... processed %d / %zu region tiles for input row %u", done, ntasks, i);
                                }
                                continue;
                            }
                            RegionTileState tileState = RegionTileState::Partial;
                            auto stateIt = featureTileStates.find(tile);
                            if (stateIt != featureTileStates.end()) {
                                tileState = stateIt->second;
                            }
                            auto tileAgg = aggOneFeatureTileRegion(
                                records, region, tile, tileState, featureRemaps[i],
                                gridSize, minProb, K,
                                use_confusion ? nullptr : &local.confusion,
                                use_confusion ? nullptr : &local.residual);
                            mergeTileAggGeneric(tileAgg, static_cast<int>(i), K, testOpts.nPerm, local);

                            const int32_t done = processed.fetch_add(1) + 1;
                            if (done % 10 == 0) {
                                notice("... processed %d / %zu region tiles for input row %u", done, ntasks, i);
                            }
                            continue;
                        }
                        auto& reader = *readers[i];
                        std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
                        tileOp.loadTileToMap(tile, pixelMap, &pixelRects, &tileStream);
                        if (pixelMap.empty()) {
                            const int32_t done = processed.fetch_add(1) + 1;
                            if (done % 10 == 0) {
                                notice("... processed %d / %zu region tiles for input row %u", done, ntasks, i);
                            }
                            continue;
                        }
                        std::unordered_map<std::pair<int32_t, int32_t>, RegionPixelState, PairHash> state;
                        state.reserve(pixelMap.size());
                        for (const auto& kv : pixelMap) {
                            state.emplace(kv.first, mask.classifyPixel(kv.first.first, kv.first.second, &tile));
                        }
                        auto tileAgg = tileOp.aggOneTileRegion(
                            pixelMap, state, reader, parser, tile, region,
                            gridSize, minProb, K,
                            use_confusion ? nullptr : &local.confusion,
                            use_confusion ? nullptr : &local.residual);
                        mergeTileAggGeneric(tileAgg, static_cast<int>(i), K, testOpts.nPerm, local);

                        const int32_t done = processed.fetch_add(1) + 1;
                        if (done % 10 == 0) {
                            notice("... processed %d / %zu region tiles for input row %u", done, ntasks, i);
                        }
                    }
                });
            Eigen::MatrixXd confusion = Eigen::MatrixXd::Zero(K, K);
            double residual = 0.0;
            for (auto& local : tls) {
                statOp.merge_from(local.stat);
                unitCache.merge_from(local.cache);
                statUnion.merge_from(local.statu);
                if (!use_confusion) {
                    confusion += local.confusion;
                    residual += local.residual;
                }
            }
            if (!use_confusion) {
                finalizeConfusionMatrix(confusion, residual, K);
                statOp.add_to_confusion(static_cast<int>(i), confusion);
            }
        }
        notice("Finished collecting sufficient statistics from all dataset-region inputs");
        return runConditionalPixelTests(outPrefix, auxiSuff, dataLabels, featureList,
                                        contrasts, statOp, statUnion, unitCache, testOpts);
    }

    // Process each dataset and collect sufficient statistics
    for (uint32_t i = 0; i < n_data; ++i) {
        const bool use_confusion = (has_confusion[i] != 0);
        auto& tileOp = *tileOps[i];
        const auto& tileList = tileOp.getTileInfo();
        int32_t nTiles = static_cast<int32_t>(tileList.size());
        notice("Processing %d tiles for dataset %u", nTiles, i);
        std::atomic<int32_t> processed{0};
        tbb::enumerable_thread_specific<LocalAgg> tls([&] {
            return LocalAgg(K, (int)n_data, M, minCount);
        });
        size_t ntasks = tileList.size();
        if (debug_) {
            ntasks = std::min<size_t>(ntasks, debug_ * nThreads);
        }
        tbb::parallel_for(tbb::blocked_range<size_t>(0, ntasks),
            [&](const tbb::blocked_range<size_t>& range)
        {
            auto& local = tls.local();
            std::ifstream tileStream;
            Eigen::MatrixXd confusion;
            double p_residual = 0.0;
            if (!use_confusion) {
                confusion = Eigen::MatrixXd::Zero(K, K);
            }
            for (size_t ti = range.begin(); ti != range.end(); ++ti) {
                const auto& tileInfo = tileList[ti];
                TileKey tile{tileInfo.row, tileInfo.col};
                if (isFeatureInput[i]) {
                    std::vector<PixTopProbsFeature<float>> records;
                    tileOp.loadTileFeatureRecords(tile, records, &tileStream);
                    auto tileAgg = aggOneFeatureTile(
                        records, featureRemaps[i],
                        gridSize, minProb, K,
                        use_confusion ? nullptr : &confusion,
                        use_confusion ? nullptr : &p_residual);
                    mergeTileAggGeneric(tileAgg, static_cast<int>(i), K, testOpts.nPerm, local);
                    int32_t done = processed.fetch_add(1) + 1;
                    if (done % 10 == 0) {
                        notice("... processed %d / %d tiles for dataset %u", done, nTiles, i);
                    }
                    continue;
                }
                auto& reader = *readers[i];
                std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
                tileOp.loadTileToMap(tile, pixelMap, nullptr, &tileStream);
                if (!use_confusion) {
                    for (const auto& kv : pixelMap) {
                        accumulateConfusionFromTopProbs(kv.second, confusion, p_residual);
                    }
                }
                auto tileAgg = tileOp.aggOneTile(pixelMap, reader, parser, tile, gridSize, minProb, K);
                mergeTileAggGeneric(tileAgg, static_cast<int>(i), K, testOpts.nPerm, local);
                int32_t done = processed.fetch_add(1) + 1;
                if (done % 10 == 0) {
                    notice("... processed %d / %d tiles for dataset %u", done, nTiles, i);
                }
            }
            if (!use_confusion) {
                finalizeConfusionMatrix(confusion, p_residual, K);
                local.stat.add_to_confusion(i, confusion);
            }
        });
        for (auto& local : tls) {
            statOp.merge_from(local.stat);
            unitCache.merge_from(local.cache);
            statUnion.merge_from(local.statu);
        }
    }
    notice("Finished collecting sufficient statistics from all datasets");
    return runConditionalPixelTests(outPrefix, auxiSuff, dataLabels, featureList,
                                    contrasts, statOp, statUnion, unitCache, testOpts);
}
