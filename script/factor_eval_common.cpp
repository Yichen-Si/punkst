#include "factor_eval_common.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>

namespace factoreval {

void addFactorEvalCommonOptions(ParamList& pl, FactorEvalCommonOptions& opt,
        const std::string& unitStatsDescription,
        const std::string& unitStatsOutDescription) {
    pl.add_option("in-data", "Input hex file", opt.inFile)
      .add_option("in-meta", "Metadata file", opt.metaFile)
      .add_option("in-model", "Input model matrix (topic-word) file", opt.modelFile, true)
      .add_option("out-prefix", "Output prefix for evaluation files", opt.outPrefix, true)
      .add_option("modal", "Modality to use (0-based)", opt.modal)
      .add_option("threads", "Number of threads", opt.nThreads)
      .add_option("seed", "Random seed", opt.seed)
      .add_option("verbose", "Verbose level", opt.verbose)
      .add_option("debug", "If >0, only process this many units", opt.debug);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", opt.dgeDirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", opt.inBarcodes)
      .add_option("in-features", "Input features.tsv.gz", opt.inFeatures)
      .add_option("in-matrix", "Input matrix.mtx.gz", opt.inMatrix)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", opt.datasetIds);

    pl.add_option("features", "Feature names and total counts file", opt.featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", opt.minCountFeature)
      .add_option("min-count-per-factor", "Minimum total count for a factor to be kept in evaluation", opt.minCountFactor)
      .add_option("min-count", "Minimum total feature count for a unit to be kept", opt.minCount)
      .add_option("feature-weights", "Input weights file", opt.weightFile)
      .add_option("default-weight", "Default weight for features not in weight file", opt.defaultWeight)
      .add_option("icol-weight", "0-based column index for weight in --feature-weights (feature name/index is column 0)", opt.icolWeight)
      .add_option("include-feature-regex", "Regex for including features", opt.includeFeatureRegex)
      .add_option("exclude-feature-regex", "Regex for excluding features", opt.excludeFeatureRegex);

    pl.add_option("max-iter", "Max iterations per document", opt.maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per document", opt.meanChangeTol)
      .add_option("candidate-threshold", "Maximum count-weighted theta abundance fraction for candidate factors", opt.candidateThreshold)
      .add_option("alpha", "Document-topic prior override for LDA transform", opt.alpha)
      .add_option("write-theta", "Write unit-by-factor theta matrix", opt.writeTheta)
      .add_option("write-unit-factor-stats", unitStatsDescription, opt.writeUnitFactorStats)
      .add_option("unit-factor-stats-out", unitStatsOutDescription, opt.unitFactorStatsOut);
}

void validateFactorEvalCommonOptions(const FactorEvalCommonOptions& opt) {
    if (opt.candidateThreshold <= 0.0) {
        error("--candidate-threshold must be greater than 0");
    }
    if (opt.candidateThreshold > 1.0) {
        error("--candidate-threshold is a fractional abundance threshold and must be <= 1");
    }
    if (opt.minCountFactor < 0.0) {
        error("--min-count-per-factor must be non-negative");
    }
    if (opt.alpha != -1.0 && opt.alpha <= 0.0) {
        error("--alpha must be positive");
    }
}

void applyWeights(std::vector<Document>& docs, const LDA4Hex& lda) {
    for (auto& doc : docs) {
        lda.applyWeights(doc);
    }
}

EvaluationData loadEvaluationData(const FactorEvalCommonOptions& opt,
        LDA4Hex& lda, DGEReader10X* dge) {
    EvaluationData data;
    const int32_t minCountInt = opt.minCount > 0 ? static_cast<int32_t>(std::ceil(opt.minCount)) : 0;
    const int32_t maxUnits = opt.debug > 0 ? opt.debug : INT32_MAX;
    if (dge != nullptr) {
        dge->readAll(data.docs, minCountInt);
        if (static_cast<int32_t>(data.docs.size()) > maxUnits) {
            data.docs.resize(maxUnits);
        }
        applyWeights(data.docs, lda);
        return data;
    }

    lda.readAllDocuments(data.docs, opt.inFile, minCountInt, maxUnits);
    return data;
}

UnitGofStats computeUnitGof(const std::vector<Document>& docs,
        const RowMajorMatrixXd& theta, const RowMajorMatrixXd& betaNorm) {
    const int32_t N = static_cast<int32_t>(docs.size());
    const int32_t M = static_cast<int32_t>(betaNorm.cols());
    UnitGofStats stats;
    stats.residuals = VectorXd::Zero(N);
    stats.cosineSim = VectorXd::Zero(N);
    const MatrixXd topicSimilarity = pairwiseCosineSimilarityRows(betaNorm);
    const ThetaEntropyStats thetaStats = computeThetaEntropyStats(theta, topicSimilarity);
    stats.entropy = thetaStats.entropy;
    stats.shLcr = thetaStats.sh_lcr;
    stats.shQ = thetaStats.sh_q;

    RowVectorXd expected = RowVectorXd::Zero(M);
    for (int32_t i = 0; i < N; ++i) {
        const Document& doc = docs[i];
        const double weightedTotal = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
        expected.noalias() = theta.row(i) * betaNorm;
        expected *= weightedTotal;

        double cosine = 0.0;
        double observedNormSq = 0.0;
        const double expectedNormSq = expected.squaredNorm();
        double docResidual = expected.sum();
        for (size_t j = 0; j < doc.ids.size(); ++j) {
            const uint32_t m = doc.ids[j];
            const double observed = doc.cnts[j];
            const double estimate = expected(m);
            docResidual += std::abs(estimate - observed) - estimate;
            cosine += estimate * observed;
            observedNormSq += observed * observed;
        }
        if (expectedNormSq > 0.0 && observedNormSq > 0.0) {
            cosine /= std::sqrt(expectedNormSq * observedNormSq);
        }
        stats.residuals(i) = docResidual;
        stats.cosineSim(i) = cosine;
    }
    return stats;
}

void writeUnitGof(const std::string& path, const std::vector<Document>& docs,
        const UnitGofStats& stats) {
    std::ofstream out(path);
    if (!out) {
        error("Error opening output file: %s for writing", path.c_str());
    }
    out << "#unit_id\ttotal_count\tresidual\tcosine_sim\tentropy\tsh_lcr\tsh_q\n";
    out << std::fixed;
    for (int32_t i = 0; i < static_cast<int32_t>(docs.size()); ++i) {
        out << i
            << "\t" << static_cast<int64_t>(std::llround(docs[i].get_raw_sum()))
            << "\t" << std::setprecision(2) << stats.residuals(i)
            << "\t" << std::setprecision(4) << stats.cosineSim(i)
            << "\t" << std::setprecision(4) << stats.entropy(i)
            << "\t" << std::setprecision(4) << stats.shLcr(i)
            << "\t" << std::setprecision(4) << stats.shQ(i) << "\n";
    }
}

void writeThetaMatrix(const std::string& path, const RowMajorMatrixXd& theta) {
    std::vector<std::string> unitNames(theta.rows());
    for (int32_t i = 0; i < theta.rows(); ++i) {
        unitNames[i] = std::to_string(i);
    }
    std::vector<std::string> factorNames(theta.cols());
    for (int32_t k = 0; k < theta.cols(); ++k) {
        factorNames[k] = std::to_string(k);
    }
    write_matrix_to_file(path, theta, 8, false, unitNames, "#unit_id", &factorNames);
}

RowMajorMatrixXd removeFactorRows(const RowMajorMatrixXd& mtx, int32_t drop) {
    RowMajorMatrixXd out(mtx.rows() - 1, mtx.cols());
    int32_t r = 0;
    for (int32_t k = 0; k < mtx.rows(); ++k) {
        if (k == drop) {
            continue;
        }
        out.row(r++) = mtx.row(k);
    }
    return out;
}

std::vector<int32_t> reducedFactorMap(int32_t K, int32_t drop) {
    std::vector<int32_t> factors;
    factors.reserve(K - 1);
    for (int32_t k = 0; k < K; ++k) {
        if (k != drop) {
            factors.push_back(k);
        }
    }
    return factors;
}

RowMajorMatrixXd reducedBetaRows(const RowMajorMatrixXd& beta,
        const std::vector<int32_t>& factors) {
    RowMajorMatrixXd out(factors.size(), beta.cols());
    for (int32_t j = 0; j < static_cast<int32_t>(factors.size()); ++j) {
        out.row(j) = beta.row(factors[j]);
    }
    return out;
}

FactorCandidateStats computeFactorCandidateStats(const std::vector<Document>& docs,
        const RowMajorMatrixXd& theta, double candidateThreshold, double minCountFactor) {
    const int32_t K = static_cast<int32_t>(theta.cols());
    FactorCandidateStats stats;
    stats.fractions.assign(K, 0.0);
    stats.thetaSums.assign(K, 0.0);
    stats.weightedCounts.assign(K, 0.0);
    stats.thetaHist.assign(K, std::vector<int32_t>(100, 0));

    for (int32_t i = 0; i < theta.rows(); ++i) {
        const double ni = docs[i].get_raw_sum();
        stats.totalCount += ni;
        for (int32_t k = 0; k < K; ++k) {
            const double thetaIk = theta(i, k);
            stats.thetaSums[k] += thetaIk;
            stats.weightedCounts[k] += thetaIk * ni;
            const int32_t bin = std::min(99,
                std::max(0, static_cast<int32_t>(std::floor(thetaIk * 100.0))));
            ++stats.thetaHist[k][bin];
        }
    }
    if (stats.totalCount <= 0.0) {
        error("Total count is not positive");
    }
    for (int32_t k = 0; k < K; ++k) {
        if (stats.weightedCounts[k] < minCountFactor) {
            continue;
        }
        stats.fractions[k] = stats.weightedCounts[k] / stats.totalCount;
        if (stats.fractions[k] <= candidateThreshold) {
            stats.candidates.push_back(k);
        }
    }
    return stats;
}

void writeThetaHist(const std::string& path,
        const std::vector<std::vector<int32_t>>& thetaHist) {
    std::ofstream out(path);
    if (!out) {
        error("Error opening output file: %s for writing", path.c_str());
    }
    out << "factor\tbin_start\tbin_end\tN\n";
    for (int32_t k = 0; k < static_cast<int32_t>(thetaHist.size()); ++k) {
        for (int32_t b = 0; b < 100; ++b) {
            out << k
                << "\t" << std::setprecision(2) << (static_cast<double>(b) / 100.0)
                << "\t" << std::setprecision(2) << (static_cast<double>(b + 1) / 100.0)
                << "\t" << thetaHist[k][b] << "\n";
        }
    }
}

} // namespace factoreval
