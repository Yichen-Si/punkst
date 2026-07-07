#pragma once

#include "topic_svb.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace factoreval {

struct FactorEvalCommonOptions {
    std::string inFile;
    std::string metaFile;
    std::string modelFile;
    std::string outPrefix;
    std::string featureFile;
    std::string weightFile;
    std::vector<std::string> dgeDirs;
    std::vector<std::string> inBarcodes;
    std::vector<std::string> inFeatures;
    std::vector<std::string> inMatrix;
    std::vector<std::string> datasetIds;
    std::string includeFeatureRegex;
    std::string excludeFeatureRegex;
    std::string unitFactorStatsOut;
    int32_t seed = -1;
    int32_t nThreads = 1;
    int32_t modal = 0;
    int32_t minCountFeature = 1;
    int32_t icolWeight = 1;
    int32_t debug = 0;
    int32_t verbose = 0;
    int32_t maxIter = 100;
    double minCount = 20.0;
    double minCountFactor = 50.0;
    double defaultWeight = 1.0;
    double meanChangeTol = 1e-3;
    double candidateThreshold = 0.005;
    double alpha = -1.0;
    bool writeUnitFactorStats = false;
    bool writeTheta = false;
};

struct EvaluationData {
    std::vector<Document> docs;
};

struct UnitGofStats {
    VectorXd residuals;
    VectorXd cosineSim;
    VectorXd entropy;
    VectorXd shLcr;
    VectorXd shQ;
};

struct FactorCandidateStats {
    std::vector<double> fractions;
    std::vector<double> thetaSums;
    std::vector<double> weightedCounts;
    std::vector<std::vector<int32_t>> thetaHist;
    std::vector<int32_t> candidates;
    double totalCount = 0.0;
};

void addFactorEvalCommonOptions(ParamList& pl, FactorEvalCommonOptions& opt,
    const std::string& unitStatsDescription,
    const std::string& unitStatsOutDescription);

void validateFactorEvalCommonOptions(const FactorEvalCommonOptions& opt);

void applyWeights(std::vector<Document>& docs, const LDA4Hex& lda);

EvaluationData loadEvaluationData(const FactorEvalCommonOptions& opt,
    LDA4Hex& lda, DGEReader10X* dge);

UnitGofStats computeUnitGof(const std::vector<Document>& docs,
    const RowMajorMatrixXd& theta, const RowMajorMatrixXd& betaNorm);

void writeUnitGof(const std::string& path, const std::vector<Document>& docs,
    const UnitGofStats& stats);

void writeThetaMatrix(const std::string& path, const RowMajorMatrixXd& theta);

RowMajorMatrixXd removeFactorRows(const RowMajorMatrixXd& mtx, int32_t drop);

std::vector<int32_t> reducedFactorMap(int32_t K, int32_t drop);

RowMajorMatrixXd reducedBetaRows(const RowMajorMatrixXd& beta,
    const std::vector<int32_t>& factors);

FactorCandidateStats computeFactorCandidateStats(const std::vector<Document>& docs,
    const RowMajorMatrixXd& theta, double candidateThreshold, double minCountFactor);

void writeThetaHist(const std::string& path,
    const std::vector<std::vector<int32_t>>& thetaHist);

} // namespace factoreval
