#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "glm.hpp"
#include "region_query.hpp"
#include "tile_io.hpp"
#include "tileoperator.hpp"

struct ContrastDef {
    std::string name;
    std::vector<int8_t> labels;
    std::vector<int32_t> group_neg;
    std::vector<int32_t> group_pos;
};

struct PixelDETestOptions {
    double minCountPerFeature = 100.0;
    double pseudoFracRel = 0.05;
    double minPvalOutput = 1.0;
    double deconvHitP = 0.05;
    double minOROutput = 1.0;
    double minORPerm = 1.2;
    int32_t nPerm = 0;
    uint64_t seed = 1;
};

void buildPairwiseContrasts(const std::vector<std::string>& dataLabels,
                            std::vector<ContrastDef>& contrasts);

Eigen::MatrixXd readConfusionMatrixFile(const std::string& path, int K);

void accumulateConfusionFromTopProbs(const TopProbs& tp,
                                     Eigen::MatrixXd& confusion,
                                     double& residualAccum);

void finalizeConfusionMatrix(Eigen::MatrixXd& confusion,
                             double residualAccum,
                             int K);

template<typename LocalAggT>
void mergeTileAggGeneric(const std::unordered_map<int32_t, TileOperator::Slice>& tileAgg,
                         int group,
                         int K,
                         int nPerm,
                         LocalAggT& local) {
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

std::unordered_map<int32_t, TileOperator::Slice> aggOneFeatureTile(
    const std::vector<PixTopProbsFeature<float>>& records,
    const std::vector<int32_t>& featureRemap,
    double gridSize,
    double minProb,
    int32_t union_key,
    Eigen::MatrixXd* confusion,
    double* residualAccum);

std::unordered_map<int32_t, TileOperator::Slice> aggOneFeatureTileRegion(
    const std::vector<PixTopProbsFeature<float>>& records,
    const PreparedRegionMask2D& region,
    const TileKey& tile,
    RegionTileState tileState,
    const std::vector<int32_t>& featureRemap,
    double gridSize,
    double minProb,
    int32_t union_key,
    Eigen::MatrixXd* confusion,
    double* residualAccum);

int32_t runConditionalPixelTests(const std::string& outPrefix,
                                 const std::string& auxSuffix,
                                 const std::vector<std::string>& dataLabels,
                                 const std::vector<std::string>& featureList,
                                 const std::vector<ContrastDef>& contrasts,
                                 MultiSlicePairwiseBinom& statOp,
                                 PairwiseBinomRobust& statUnion,
                                 MultiSliceUnitCache& unitCache,
                                 const PixelDETestOptions& opts);
