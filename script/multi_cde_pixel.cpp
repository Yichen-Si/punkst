#include "punkst.h"
#include "multi_cde_pixel_common.hpp"
#include "tileoperator.hpp"
#include "utils.h"
#include "glm.hpp"
#include <fstream>
#include <memory>
#include <atomic>
#include <unordered_set>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

static void readContrastDesignFile(const std::string& contrastFile,
        std::vector<std::string>& annoPrefixes,
        std::vector<std::string>& ptsPrefixes,
        std::vector<std::string>& confusionFiles,
        std::vector<ContrastDef>& contrasts);

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
    int32_t K;
    int32_t icol_x, icol_y, icol_feature, icol_val;
    float qxmin, qxmax, qymin, qymax;
    bool bounded = false;
    double minCount = 10.0;
    double minProb = 0.01;
    int32_t debug_ = 0;
    int32_t nThreads = 1;
    std::vector<ContrastDef> contrasts;

    ParamList pl;
    pl.add_option("anno-data", "Input pixel files", inData)
      .add_option("anno-index", "Input pixel index files", inIndex)
      .add_option("anno", "Prefixes of input pixel data files", inPrefix)
      .add_option("labels", "Labels for the datasets to show in pairwise output", dataLabels)
      .add_option("binary", "Data file is in binary format", isBinary)
      .add_option("K", "Number of factors in the annotation files", K, true)
      .add_option("pts", "Prefixes of the transcript data files", inPtsPrefix)
      .add_option("contrast", "Contrast design TSV (anno, pts, [confusion], contrasts)", contrastFile)
      .add_option("confusion", "Per-dataset confusion matrices (TSV with K+1 rows/cols)", inConfusion)
      .add_option("features", "List of features to test", dictFile, true)
      .add_option("grid-size", "Grid size", gridSize, true)
      .add_option("pseudo-rel", "Relative pseudo count fraction w.r.t. null", testOpts.pseudoFracRel)
      .add_option("icol-x", "Column index for x coordinate for files in --pts (0-based)", icol_x, true)
      .add_option("icol-y", "Column index for y coordinate for files in --pts (0-based)", icol_y, true)
      .add_option("icol-feature", "Column index for feature for files in --pts (0-based)", icol_feature, true)
      .add_option("icol-val", "Column index for count/value for files in --pts (0-based)", icol_val, true)
      .add_option("xmin", "Minimum x coordinate for subsetting", qxmin)
      .add_option("xmax", "Maximum x coordinate for subsetting", qxmax)
      .add_option("ymin", "Minimum y coordinate for subsetting", qymin)
      .add_option("ymax", "Maximum y coordinate for subsetting", qymax)
      .add_option("bounded", "Whether to subset to the bounding box defined by --xmin/--xmax/--ymin/--ymax", bounded)
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
        readContrastDesignFile(contrastFile, inPrefix, inPtsPrefix, inConfusion, contrasts);
    }
    if (!inPrefix.empty()) {
        inData.resize(inPrefix.size());
        inIndex.resize(inPrefix.size());
        for (size_t i = 0; i < inPrefix.size(); ++i) {
            inData[i] = inPrefix[i] + (isBinary ? ".bin" : ".tsv");
            inIndex[i] = inPrefix[i] + ".index";
        }
    } else if (inData.empty() || inIndex.empty() || inData.size() != inIndex.size()) {
        error("Either --anno or both --anno-data and --anno-index must be specified");
    }
    if (inPtsPrefix.empty() || inPtsPrefix.size() != inData.size()) {
        error("--pts must match with --anno/--anno-data");
    }
    uint32_t n_data = static_cast<uint32_t>(inData.size());
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

    if (gridSize <= 0) {error("--grid-size must be positive");}
    if (K <= 0) {error("--K must be positive");}
    if (bounded && (qxmin >= qxmax || qymin >= qymax)) {
        error("Invalid bounding box specified");
    }
    // Set up data readers
    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val, dictFile);
    std::vector<std::string> featureList;
    parser.getFeatureList(featureList);
    int32_t M = static_cast<int32_t>(featureList.size());
    if (M == 0) {error("No features found");}

    std::vector<std::unique_ptr<TileOperator>> tileOps;
    for (uint32_t i = 0; i < n_data; ++i) {
        tileOps.emplace_back(std::make_unique<TileOperator>(inData[i], inIndex[i]));
        if (bounded) {
            int32_t n = tileOps.back()->query(qxmin, qxmax, qymin, qymax);
            debug("Dataset %u: using %d tiles in the bounding box", i, n);
        }
    }

    std::vector<TileReader> readers;
    for (uint32_t i = 0; i < n_data; ++i) {
        readers.emplace_back(inPtsPrefix[i] + ".tsv", inPtsPrefix[i] + ".index");
        if (readers.back().getTileSize() != tileOps[i]->getTileSize()) {
            error("Currently we require the tile size to be the same for each pair of annotation and transcript data. Invalid dataset %u (%u vs %u)", i, tileOps[i]->getTileSize(), readers.back().getTileSize());
        }
    }

    MultiSlicePairwiseBinom statOp(K, n_data, M, minCount);
    PairwiseBinomRobust statUnion(n_data, M, minCount);
    MultiSliceUnitCache unitCache(K, M, minCount);
    struct LocalAgg {
        MultiSlicePairwiseBinom stat;
        MultiSliceUnitCache cache;
        PairwiseBinomRobust statu;
        LocalAgg(int K, int G, int M, double minCount)
            : stat(K, G, M, minCount), cache(K, M, minCount), statu(G, M, minCount) {}
    };

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

    // Process each dataset and collect sufficient statistics
    for (uint32_t i = 0; i < n_data; ++i) {
        const bool use_confusion = (has_confusion[i] != 0);
        auto& tileOp = *tileOps[i];
        const auto& tileList = tileOp.getTileInfo();
        int32_t nTiles = static_cast<int32_t>(tileList.size());
        notice("Processing %d tiles for dataset %u", nTiles, i);
        auto& reader = readers[i];
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
                std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
                tileOp.loadTileToMap(tile, pixelMap, nullptr, &tileStream);
                if (!use_confusion) {
                    for (const auto& kv : pixelMap) {
                        accumulateConfusionFromTopProbs(kv.second, confusion, p_residual);
                    }
                }
                auto tileAgg = tileOp.aggOneTile(pixelMap, reader, parser, tile, gridSize, minProb, K);
                for (const auto& kv : tileAgg) {
                    const int32_t k = kv.first;
                    if (k < 0 || k > K) continue;
                    if (k == K) {
                        for (const auto& unitKv : kv.second) {
                            const auto& obs = unitKv.second;
                            local.statu.add_unit(static_cast<int>(i), obs.totalCount, obs.featureCounts);
                        }
                        continue;
                    }
                    for (const auto& unitKv : kv.second) {
                        const auto& obs = unitKv.second;
                        local.stat.slice(k).add_unit(static_cast<int>(i), obs.totalCount, obs.featureCounts);
                        if (testOpts.nPerm > 0) {
                            local.cache.add_unit(k, static_cast<int>(i), obs.totalCount, obs.featureCounts);
                        }
                    }
                }
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


static void readContrastDesignFile(const std::string& contrastFile,
        std::vector<std::string>& annoPrefixes,
        std::vector<std::string>& ptsPrefixes,
        std::vector<std::string>& confusionFiles,
        std::vector<ContrastDef>& contrasts) {
    annoPrefixes.clear();
    ptsPrefixes.clear();
    confusionFiles.clear();
    std::ifstream ifs(contrastFile);
    if (!ifs.is_open()) {
        error("Cannot open contrast file: %s", contrastFile.c_str());
    }
    std::string line;
    std::vector<std::string> tokens;
    if (!std::getline(ifs, line)) {
        error("Contrast file is empty: %s", contrastFile.c_str());
    }
    split(tokens, "\t ", line, UINT_MAX, true, true, true);
    if (tokens.size() < 3) {
        error("Contrast file must have >= 3 columns (anno, pts, [confusion], contrast...): %s", contrastFile.c_str());
    }
    const size_t n_header_cols = tokens.size();
    int32_t confusion_col = -1;
    std::vector<int32_t> contrast_cols;
    contrast_cols.reserve(n_header_cols > 2 ? n_header_cols - 2 : 0);
    for (size_t i = 2; i < n_header_cols; ++i) {
        if (tokens[i] == "confusion") {
            if (confusion_col >= 0) {
                error("Contrast file has multiple 'confusion' columns: %s", contrastFile.c_str());
            }
            confusion_col = static_cast<int32_t>(i);
            continue;
        }
        contrast_cols.push_back(static_cast<int32_t>(i));
    }
    const int32_t C = static_cast<int32_t>(contrast_cols.size());
    if (C <= 0) {
        error("Contrast file must include at least one contrast column: %s", contrastFile.c_str());
    }
    contrasts.assign(C, ContrastDef{});
    std::unordered_set<std::string> seen_names;
    for (int32_t c = 0; c < C; ++c) {
        const std::string name = tokens[contrast_cols[c]];
        if (name.empty()) {
            warning("Contrast %d has an empty header name; using index instead", c);
            contrasts[c].name = std::to_string(c);
        } else {
            if (!seen_names.insert(name).second) {
                error("Contrast name '%s' appears more than once in header", name.c_str());
            }
            contrasts[c].name = name;
        }
    }

    int32_t n_samples = 0;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        split(tokens, "\t ", line, UINT_MAX, true, true, true);
        if (tokens.size() != n_header_cols) {
            error("Contrast file row has %zu columns, expected %zu: %s",
                tokens.size(), n_header_cols, line.c_str());
        }
        annoPrefixes.push_back(tokens[0]);
        ptsPrefixes.push_back(tokens[1]);
        if (confusion_col >= 0) {
            confusionFiles.push_back(tokens[confusion_col]);
        }
        for (int32_t c = 0; c < C; ++c) {
            int32_t y = 0;
            const std::string& token = tokens[contrast_cols[c]];
            if (!str2int32(token, y)) {
                error("Contrast value must be -1, 0, or 1: %s", token.c_str());
            }
            if (y < -1 || y > 1) {
                error("Contrast value must be -1, 0, or 1: %d", y);
            }
            contrasts[c].labels.push_back(static_cast<int8_t>(y));
        }
        n_samples++;
    }
    if (n_samples == 0) {
        error("Contrast file has a header but no samples: %s", contrastFile.c_str());
    }
    for (int32_t c = 0; c < C; ++c) {
        if (contrasts[c].labels.size() != static_cast<size_t>(n_samples)) {
            error("Contrast %s has %zu labels, expected %d samples",
                contrasts[c].name.c_str(), contrasts[c].labels.size(), n_samples);
        }
        contrasts[c].group_neg.clear();
        contrasts[c].group_pos.clear();
        for (int32_t s = 0; s < n_samples; ++s) {
            const int8_t y = contrasts[c].labels[s];
            if (y < 0) {
                contrasts[c].group_neg.push_back(s);
            } else if (y > 0) {
                contrasts[c].group_pos.push_back(s);
            }
        }
        if (contrasts[c].group_neg.empty() || contrasts[c].group_pos.empty()) {
            error("Contrast %s must have at least one sample in each group", contrasts[c].name.c_str());
        }
        notice("Read contrast %s with %zu vs %zu samples", contrasts[c].name.c_str(), contrasts[c].group_pos.size(), contrasts[c].group_neg.size());
    }
}
