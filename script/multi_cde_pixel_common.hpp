#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "glm.hpp"
#include "tile_io.hpp"

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

int32_t runConditionalPixelTests(const std::string& outPrefix,
                                 const std::string& auxSuffix,
                                 const std::vector<std::string>& dataLabels,
                                 const std::vector<std::string>& featureList,
                                 const std::vector<ContrastDef>& contrasts,
                                 MultiSlicePairwiseBinom& statOp,
                                 PairwiseBinomRobust& statUnion,
                                 MultiSliceUnitCache& unitCache,
                                 const PixelDETestOptions& opts);
