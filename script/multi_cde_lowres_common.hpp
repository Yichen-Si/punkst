#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "dataunits.hpp"
#include "Eigen/Dense"

using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Eigen::VectorXd;

enum class LowresPoisLink { Log, Log1p };

struct LowresScaleOptions {
    std::string link = "log1p";
    double sizeFactor = 10000.0;
    double cScale = -1.0;

    LowresPoisLink parsedLink() const;
    bool useLog1p() const;
    double linkScale() const;
    void validate(bool usesLdaUncertainty = false) const;
};

struct LowresContrastDesign {
    std::vector<std::string> inFile;
    std::vector<std::string> metaFile;
    std::vector<std::string> dataLabels;
    std::vector<std::vector<int32_t>> contrasts;
    std::vector<std::string> contrastNames;

    int32_t nSamples() const { return static_cast<int32_t>(inFile.size()); }
    int32_t nContrasts() const { return static_cast<int32_t>(contrasts.size()); }
};

struct LowresInputData {
    int32_t K = 0;
    int32_t M = 0;
    int32_t N_total = 0;
    std::vector<std::string> factorNames;
    std::vector<std::string> featureNames;
    RowMajorMatrixXd A_all;
    VectorXd cvec_all;
    std::vector<std::vector<Document>> docsT;
    std::vector<int32_t> nUnits;
    std::vector<int32_t> offsets;
    std::vector<VectorXd> cThetaSums;
};

struct LowresContrastData {
    int32_t n = 0;
    VectorXd xvec;
    VectorXd cvecMasked;
    VectorXd aSum;
    double cSum = 0.0;
};

struct LowresFeatureObs {
    int32_t feature = -1;
    int32_t nnz = 0;
    int32_t totalCount = 0;
    double ysum0 = 0.0;
    double ysum1 = 0.0;
    Document y;
};

LowresContrastDesign loadLowresContrastDesign(
    const std::string& contrastFile,
    const std::vector<std::string>& inFile,
    const std::vector<std::string>& metaFile,
    const std::vector<std::string>& dataLabels);

LowresInputData loadLowresInputData(
    const std::string& modelFile,
    int32_t seed,
    int32_t nThreads,
    const LowresContrastDesign& design,
    const LowresScaleOptions& scale,
    int32_t minCount,
    int32_t debug,
    RowMajorMatrixXd& eta0);

LowresContrastData buildLowresContrastData(
    const LowresInputData& data,
    const std::vector<int32_t>& contrast);

bool buildLowresFeatureObs(
    const LowresInputData& data,
    const std::vector<int32_t>& contrast,
    int32_t feature,
    int32_t minUnits,
    int32_t minCountFeature,
    LowresFeatureObs& out);

std::vector<int32_t> shuffledFeatureOrder(int32_t M, int32_t seed);

void writeLowresEta0(
    const std::string& path,
    const RowMajorMatrixXd& eta0,
    bool useLog1p,
    const std::vector<std::string>& featureNames,
    const std::vector<std::string>& factorNames);
