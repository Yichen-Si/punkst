#include "punkst.h"
#include "multi_cde_pixel_common.hpp"
#include "region_query.hpp"
#include "tileoperator.hpp"
#include "utils.h"
#include "glm.hpp"

#include <algorithm>
#include <atomic>
#include <memory>
#include <unordered_map>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace {

struct RegionTileAggLocal {
    MultiSlicePairwiseBinom stat;
    MultiSliceUnitCache cache;
    PairwiseBinomRobust statu;
    Eigen::MatrixXd confusion_neg;
    Eigen::MatrixXd confusion_pos;
    double residual_neg = 0.0;
    double residual_pos = 0.0;

    RegionTileAggLocal(int K, int M, double minCount)
        : stat(K, 2, M, minCount),
          cache(K, M, minCount),
          statu(2, M, minCount),
          confusion_neg(Eigen::MatrixXd::Zero(K, K)),
          confusion_pos(Eigen::MatrixXd::Zero(K, K)) {}
};

void mergeTileAgg(const std::unordered_map<int32_t, TileOperator::Slice>& tileAgg,
                  int group,
                  int K,
                  int nPerm,
                  RegionTileAggLocal& local) {
    for (const auto& kv : tileAgg) {
        const int32_t k = kv.first;
        if (k < 0 || k > K) {
            continue;
        }
        if (k == K) {
            for (const auto& unitKv : kv.second) {
                const auto& obs = unitKv.second;
                local.statu.add_unit(group, obs.totalCount, obs.featureCounts);
            }
            continue;
        }
        for (const auto& unitKv : kv.second) {
            const auto& obs = unitKv.second;
            local.stat.slice(k).add_unit(group, obs.totalCount, obs.featureCounts);
            if (nPerm > 0) {
                local.cache.add_unit(k, group, obs.totalCount, obs.featureCounts);
            }
        }
    }
}

} // namespace

int32_t cmdConditionalTestRegionPixel(int32_t argc, char** argv) {
    std::string annoPrefix, annoData, annoIndex;
    std::string ptsPrefix;
    std::string dictFile, outPrefix;
    std::string regionNegFile, regionPosFile;
    std::string regionLabelNeg = "0";
    std::string regionLabelPos = "1";
    std::string auxiSuff;
    PixelDETestOptions testOpts;
    bool isBinary = false;
    double gridSize = 0.0;
    int32_t K = 0;
    int32_t icol_x = 0, icol_y = 0, icol_feature = 0, icol_val = 0;
    double minCount = 10.0;
    double minProb = 0.01;
    int32_t debug_ = 0;
    int32_t nThreads = 1;
    int64_t regionScale = 10;

    ParamList pl;
    pl.add_option("anno-data", "Input pixel file", annoData)
      .add_option("anno-index", "Input pixel index file", annoIndex)
      .add_option("anno", "Prefix of the input pixel data file", annoPrefix)
      .add_option("pts", "Prefix of the transcript data file", ptsPrefix, true)
      .add_option("region-neg", "GeoJSON file for the negative/reference region", regionNegFile, true)
      .add_option("region-pos", "GeoJSON file for the positive/comparison region", regionPosFile, true)
      .add_option("region-scale", "Integer coordinate scale for GeoJSON preprocessing", regionScale)
      .add_option("region-label-neg", "Label for the negative/reference region", regionLabelNeg)
      .add_option("region-label-pos", "Label for the positive/comparison region", regionLabelPos)
      .add_option("binary", "Annotation file is in binary format", isBinary)
      .add_option("K", "Number of factors in the annotation file", K, true)
      .add_option("features", "List of features to test", dictFile, true)
      .add_option("grid-size", "Grid size", gridSize, true)
      .add_option("pseudo-rel", "Relative pseudo count fraction w.r.t. null", testOpts.pseudoFracRel)
      .add_option("icol-x", "Column index for x coordinate for files in --pts (0-based)", icol_x, true)
      .add_option("icol-y", "Column index for y coordinate for files in --pts (0-based)", icol_y, true)
      .add_option("icol-feature", "Column index for feature for files in --pts (0-based)", icol_feature, true)
      .add_option("icol-val", "Column index for count/value for files in --pts (0-based)", icol_val, true)
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
      .add_option("perm", "Number of permutations for beta calibration", testOpts.nPerm)
      .add_option("seed", "Seed for permutations", testOpts.seed)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
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

    if (!annoPrefix.empty()) {
        if (!annoData.empty() || !annoIndex.empty()) {
            error("--anno cannot be combined with --anno-data/--anno-index");
        }
        annoData = annoPrefix + (isBinary ? ".bin" : ".tsv");
        annoIndex = annoPrefix + ".index";
    } else if (annoData.empty() || annoIndex.empty()) {
        error("Either --anno or both --anno-data and --anno-index must be specified");
    }
    if (gridSize <= 0) {
        error("--grid-size must be positive");
    }
    if (K <= 0) {
        error("--K must be positive");
    }

    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val, dictFile);
    std::vector<std::string> featureList;
    parser.getFeatureList(featureList);
    const int32_t M = static_cast<int32_t>(featureList.size());
    if (M == 0) {
        error("No features found");
    }

    TileOperator tileOp(annoData, annoIndex, "", nThreads);
    TileReader reader(ptsPrefix + ".tsv", ptsPrefix + ".index");
    if (reader.getTileSize() != tileOp.getTileSize()) {
        error("Currently we require the tile size to be the same for the annotation and transcript data (%d vs %d)",
              tileOp.getTileSize(), reader.getTileSize());
    }

    PreparedRegionMask2D regionNeg = loadPreparedRegionGeoJSON(regionNegFile, tileOp.getTileSize(), regionScale);
    PreparedRegionMask2D regionPos = loadPreparedRegionGeoJSON(regionPosFile, tileOp.getTileSize(), regionScale);
    if (regionNeg.empty()) {
        error("Negative region is empty after preprocessing: %s", regionNegFile.c_str());
    }
    if (regionPos.empty()) {
        error("Positive region is empty after preprocessing: %s", regionPosFile.c_str());
    }

    const float pixelResolution = tileOp.getPixelResolution() > 0.0f ? tileOp.getPixelResolution() : 1.0f;
    const PreparedRegionRasterMask2D maskNeg = prepareRegionRasterMask2D(regionNeg, pixelResolution);
    const PreparedRegionRasterMask2D maskPos = prepareRegionRasterMask2D(regionPos, pixelResolution);

    std::vector<Rectangle<double>> queryRects;
    queryRects.emplace_back(regionNeg.bbox_f.xmin, regionNeg.bbox_f.ymin,
                            regionNeg.bbox_f.xmax, regionNeg.bbox_f.ymax);
    queryRects.emplace_back(regionPos.bbox_f.xmin, regionPos.bbox_f.ymin,
                            regionPos.bbox_f.xmax, regionPos.bbox_f.ymax);
    std::vector<TileKey> tileList;
    std::vector<bool> isContained;
    reader.getTileList(queryRects, tileList, isContained);
    std::sort(tileList.begin(), tileList.end());
    tileList.erase(std::unique(tileList.begin(), tileList.end()), tileList.end());
    if (tileList.empty()) {
        error("No transcript tiles intersect the two region bounding boxes");
    }

    const std::vector<Rectangle<float>> pixelRects = {regionNeg.bbox_f, regionPos.bbox_f};
    MultiSlicePairwiseBinom statOp(K, 2, M, minCount);
    PairwiseBinomRobust statUnion(2, M, minCount);
    MultiSliceUnitCache unitCache(K, M, minCount);
    Eigen::MatrixXd confusionNeg = Eigen::MatrixXd::Zero(K, K);
    Eigen::MatrixXd confusionPos = Eigen::MatrixXd::Zero(K, K);
    double residualNeg = 0.0;
    double residualPos = 0.0;
    std::atomic<int32_t> processed{0};

    size_t ntasks = tileList.size();
    if (debug_ > 0) {
        ntasks = std::min<size_t>(ntasks, static_cast<size_t>(debug_ * nThreads));
    }

    tbb::enumerable_thread_specific<RegionTileAggLocal> tls([&] {
        return RegionTileAggLocal(K, M, minCount);
    });

    tbb::parallel_for(tbb::blocked_range<size_t>(0, ntasks),
        [&](const tbb::blocked_range<size_t>& range) {
            auto& local = tls.local();
            std::ifstream tileStream;
            for (size_t ti = range.begin(); ti != range.end(); ++ti) {
                const TileKey tile = tileList[ti];
                std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
                tileOp.loadTileToMap(tile, pixelMap, &pixelRects, &tileStream);
                if (pixelMap.empty()) {
                    const int32_t done = processed.fetch_add(1) + 1;
                    if (done % 10 == 0) {
                        notice("... processed %d / %zu region tiles", done, ntasks);
                    }
                    continue;
                }

                std::unordered_map<std::pair<int32_t, int32_t>, RegionPixelState, PairHash> stateNeg;
                std::unordered_map<std::pair<int32_t, int32_t>, RegionPixelState, PairHash> statePos;
                stateNeg.reserve(pixelMap.size());
                statePos.reserve(pixelMap.size());
                for (const auto& kv : pixelMap) {
                    stateNeg.emplace(kv.first, maskNeg.classifyPixel(kv.first.first, kv.first.second, &tile));
                    statePos.emplace(kv.first, maskPos.classifyPixel(kv.first.first, kv.first.second, &tile));
                }

                auto tileAggNeg = tileOp.aggOneTileRegion(
                    pixelMap, stateNeg, reader, parser, tile, regionNeg,
                    gridSize, minProb, K, &local.confusion_neg, &local.residual_neg);
                auto tileAggPos = tileOp.aggOneTileRegion(
                    pixelMap, statePos, reader, parser, tile, regionPos,
                    gridSize, minProb, K, &local.confusion_pos, &local.residual_pos);

                mergeTileAgg(tileAggNeg, 0, K, testOpts.nPerm, local);
                mergeTileAgg(tileAggPos, 1, K, testOpts.nPerm, local);

                const int32_t done = processed.fetch_add(1) + 1;
                if (done % 10 == 0) {
                    notice("... processed %d / %zu region tiles", done, ntasks);
                }
            }
        });

    for (auto& local : tls) {
        statOp.merge_from(local.stat);
        unitCache.merge_from(local.cache);
        statUnion.merge_from(local.statu);
        confusionNeg += local.confusion_neg;
        confusionPos += local.confusion_pos;
        residualNeg += local.residual_neg;
        residualPos += local.residual_pos;
    }
    finalizeConfusionMatrix(confusionNeg, residualNeg, K);
    finalizeConfusionMatrix(confusionPos, residualPos, K);
    statOp.add_to_confusion(0, confusionNeg);
    statOp.add_to_confusion(1, confusionPos);
    notice("Finished collecting sufficient statistics from the two regions");

    std::vector<std::string> dataLabels = {regionLabelNeg, regionLabelPos};
    ContrastDef contrast;
    contrast.name = regionLabelNeg + "_vs_" + regionLabelPos;
    contrast.labels = {-1, 1};
    contrast.group_neg = {0};
    contrast.group_pos = {1};

    return runConditionalPixelTests(outPrefix, auxiSuff, dataLabels, featureList,
                                    {contrast}, statOp, statUnion, unitCache, testOpts);
}
