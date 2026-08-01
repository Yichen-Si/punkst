#include "topic_svb.hpp"
#include "transform_helper.hpp"

#include <fstream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <limits>
#include <numeric>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace {

using transform_pseudobulk::Mode;
using transform_pseudobulk::SpecialBatch;
using transform_helpers::TransformBatch;
using transform_helpers::applyWeights;
using transform_helpers::assignBarcodeIds;
using transform_helpers::readSpecialDgeMinibatch;
using transform_helpers::readSpecialHexMinibatch;
using transform_helpers::writeUnitIdHeader;
using feature_diagnostics::PullRecord;
using feature_diagnostics::ResidualLocalAgg;

void writeResultHeader(std::ostream& out, LDA4Hex& lda, int32_t topkOnly) {
    if (topkOnly < 0) {
        lda.writeUnitHeader(out);
        return;
    }
    const int32_t topk = std::min(topkOnly, lda.getNumTopics());
    if (topk <= 0) {
        out << "\n";
        return;
    }
    out << "K1";
    for (int32_t i = 1; i < topk; ++i) {
        out << "\tK" << (i + 1);
    }
    for (int32_t i = 0; i < topk; ++i) {
        out << "\tP" << (i + 1);
    }
    out << "\n";
}

struct ResidualState : feature_diagnostics::FeatureResidualState {
    RowMajorMatrixXd betaNormRow;
    MatrixXd betaNormCol;
    MatrixXd betaAllocationKernel;

    ResidualState(const RowMajorMatrixXd& model, bool cheapDiagnostics,
            bool similarityDiagnostics, bool useTrainingPrevalence,
            const std::string& tempDir)
        : FeatureResidualState(
              static_cast<int32_t>(model.rows()),
              static_cast<int32_t>(model.cols()),
              useTrainingPrevalence
                  ? feature_diagnostics::make_cofeature_model(
                      model, model.rowwise().sum())
                  : feature_diagnostics::CofeatureModel{},
              cheapDiagnostics, useTrainingPrevalence, tempDir),
          betaNormRow(rowNormalize(model)),
          betaNormCol(betaNormRow),
          betaAllocationKernel(dirichlet_expectation_2d(model)) {
        if (similarityDiagnostics) {
            feature_diagnostics::initialize_topic_similarity(
                *this, betaNormRow);
        }
    }
};

struct TransformOutputs {
    std::string resultsPath;
    std::string unitStatsPath;
    std::ofstream results;
    std::ofstream unitStats;

    TransformOutputs(const std::string& outPrefix, bool writeUnitStats)
        : resultsPath(outPrefix + ".results.tsv"),
          unitStatsPath(outPrefix + ".unit_stats.tsv"),
          results(resultsPath) {
        if (!results) {
            error("Error opening output file: %s for writing", resultsPath.c_str());
        }
        if (writeUnitStats) {
            unitStats.open(unitStatsPath);
            if (!unitStats) {
                error("Error opening output file: %s for writing", unitStatsPath.c_str());
            }
        }
    }
};

class TransformBatchProcessor {
public:
    TransformBatchProcessor(LDA4Hex& lda_, std::ofstream& resultsStream_,
            std::ofstream* unitMetaStream_, RowMajorMatrixXd& pseudobulk_,
            MatrixXd& specialPseudobulk_, Mode pseudobulkMode_,
            ResidualState* residualState_, int32_t topkOnly_,
            bool similarityDiagnostics_, int32_t nThreads_)
        : lda(lda_),
          resultsStream(resultsStream_),
          unitMetaStream(unitMetaStream_),
          pseudobulk(pseudobulk_),
          specialPseudobulk(specialPseudobulk_),
          pseudobulkMode(pseudobulkMode_),
          residualState(residualState_),
          topkOnly(topkOnly_),
          similarityDiagnostics(similarityDiagnostics_),
          threadHint(std::max<int32_t>(1, nThreads_)),
          M(lda_.nFeatures()),
          K(lda_.getNumTopics()) {
        if (pseudobulkMode == Mode::Standard) {
            standardTls = std::make_unique<StandardTls>([this] {
                return StandardLocalAgg(M, K);
            });
        }
        if (residualState != nullptr) {
            residualTls = std::make_unique<ResidualTls>([this] {
                return ResidualLocalAgg(K);
            });
        }
    }

    void process(TransformBatch& batch) {
        if (batch.empty()) {
            return;
        }
        if (pseudobulkMode != Mode::Standard) {
            error("%s: standard batch used with a special pseudobulk mode", __func__);
        }

        RowMajorMatrixXd gamma;
        RowMajorMatrixXd doc_topic =
            inferTopics(DocumentView(batch.docs), gamma);
        writeTopicRows(batch.ids, doc_topic);

        const size_t grainSize = std::max<size_t>(
            1, batch.size() / (2 * static_cast<size_t>(threadHint)));
        tbb::parallel_for(tbb::blocked_range<size_t>(
                0, batch.size(), grainSize),
            [&](const tbb::blocked_range<size_t>& range) {
                auto& local = standardTls->local();
                for (size_t idx = range.begin(); idx < range.end(); ++idx) {
                    const int32_t i = static_cast<int32_t>(idx);
                    const Document& doc = batch.docs[idx];
                    for (size_t j = 0; j < doc.ids.size(); ++j) {
                        const uint32_t m = doc.ids[j];
                        const double cnt = doc.cnts[j];
                        const double raw_count = lda.rawCountFor(m, cnt, doc.counts_weighted);
                        for (int32_t k = 0; k < K; ++k) {
                            local.pseudobulk(m, k) += raw_count * doc_topic(i, k);
                        }
                    }
                }
            });

        processResiduals(batch.docs, batch.ids, doc_topic, gamma);
    }

    void process(SpecialBatch& batch) {
        if (batch.empty()) {
            return;
        }
        if (pseudobulkMode == Mode::Standard) {
            error("%s: special batch used with standard pseudobulk mode", __func__);
        }

        RowMajorMatrixXd gamma;
        RowMajorMatrixXd doc_topic =
            inferTopics(DocumentView(batch.modelDocs), gamma);
        writeTopicRows(batch.ids, doc_topic);
        transform_pseudobulk::accumulate(
            specialPseudobulk, batch, doc_topic, pseudobulkMode);
        processResiduals(batch.modelDocs, batch.ids, doc_topic, gamma);
    }

    void finalize() {
        if (pseudobulkMode == Mode::Standard) {
            for (auto& local : *standardTls) {
                pseudobulk += local.pseudobulk;
            }
        }
        if (residualState != nullptr) {
            for (auto& local : *residualTls) {
                residualState->topicExposureTotals +=
                    local.topicExposureTotals;
            }
            finalizeResiduals();
        }
    }

private:
    struct StandardLocalAgg {
        RowMajorMatrixXd pseudobulk;

        StandardLocalAgg(int32_t M, int32_t K)
            : pseudobulk(RowMajorMatrixXd::Zero(M, K)) {}
    };

    using StandardTls = tbb::enumerable_thread_specific<StandardLocalAgg>;
    using ResidualTls = tbb::enumerable_thread_specific<ResidualLocalAgg>;

    RowMajorMatrixXd inferTopics(
            DocumentView docs, RowMajorMatrixXd& gamma) {
        if (residualState == nullptr) {
            return lda.do_transform(docs);
        }
        gamma = lda.do_transform_gamma(docs);
        RowMajorMatrixXd topics = gamma;
        for (int32_t d = 0; d < topics.rows(); ++d) {
            const double total = topics.row(d).sum();
            if (total > 0.0 && std::isfinite(total)) {
                topics.row(d) /= total;
            } else {
                topics.row(d).setConstant(
                    1.0 / static_cast<double>(K));
            }
        }
        return topics;
    }

    void processResiduals(const std::vector<Document>& docs,
            const std::vector<std::string>& ids,
            const RowMajorMatrixXd& docTopic,
            const RowMajorMatrixXd& gamma) {
        if (!residualState) return;
        const int32_t nDocs = static_cast<int32_t>(docs.size());
        const std::vector<size_t> documentOffsets =
            feature_diagnostics::make_document_offsets(docs);
        const size_t nCells = documentOffsets.back();
        std::vector<double> expectedCells(nCells, 0.0);
        feature_diagnostics::CofeatureBatchContext cofeatureContext;
        if (residualState->useTrainingPrevalence) {
            cofeatureContext =
                feature_diagnostics::make_cofeature_batch_context(
                    docs, residualState->cofeatureModel, threadHint);
        }
        RowMajorMatrixXd thetaKernel(nDocs, K);
        VectorXd unitResidual = VectorXd::Zero(nDocs);
        VectorXd unitCosine;
        VectorXd unitEntropy = VectorXd::Zero(nDocs);
        ThetaEntropyStats similarityEntropy;
        VectorXd expectedNormSq;
        const double alpha = lda.get_doc_topic_prior();
        const size_t grainSize = std::max<size_t>(
            1, docs.size() / (2 * static_cast<size_t>(threadHint)));
        VectorXd documentTotals;
        if (similarityDiagnostics) {
            unitCosine = VectorXd::Zero(nDocs);
            documentTotals.resize(nDocs);
            tbb::parallel_for(tbb::blocked_range<size_t>(
                    0, docs.size(), grainSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t d = range.begin(); d < range.end(); ++d) {
                        documentTotals(static_cast<int32_t>(d)) =
                            std::accumulate(
                                docs[d].cnts.begin(), docs[d].cnts.end(), 0.0);
                    }
                });
            similarityEntropy = computeThetaEntropyStats(
                docTopic, residualState->topicSimilarity);
            unitEntropy = similarityEntropy.entropy;
            RowMajorMatrixXd expectedTopicWeights = docTopic;
            for (int32_t d = 0; d < nDocs; ++d) {
                expectedTopicWeights.row(d) *= documentTotals(d);
            }
            expectedNormSq = rowQuadraticForms(
                expectedTopicWeights, residualState->profileGram);
        }

        tbb::parallel_for(tbb::blocked_range<size_t>(
                0, docs.size(), grainSize),
            [&](const tbb::blocked_range<size_t>& range) {
                ResidualLocalAgg& local = residualTls->local();
                for (size_t d = range.begin(); d < range.end(); ++d) {
                    const Document& doc = docs[d];
                    const double documentTotal = similarityDiagnostics
                        ? documentTotals(static_cast<int32_t>(d))
                        : std::accumulate(
                            doc.cnts.begin(), doc.cnts.end(), 0.0);
                    local.topicExposureTotals.noalias() +=
                        documentTotal
                        * docTopic.row(static_cast<int32_t>(d)).transpose();

                    double maxLog = -std::numeric_limits<double>::infinity();
                    for (int32_t k = 0; k < K; ++k) {
                        const double value = psi(std::max(
                            gamma(static_cast<int32_t>(d), k) + alpha,
                            1e-12));
                        thetaKernel(static_cast<int32_t>(d), k) = value;
                        maxLog = std::max(maxLog, value);
                    }
                    for (int32_t k = 0; k < K; ++k) {
                        thetaKernel(static_cast<int32_t>(d), k) = std::exp(
                            thetaKernel(static_cast<int32_t>(d), k) - maxLog);
                    }

                    const VectorXd theta =
                        docTopic.row(static_cast<int32_t>(d)).transpose();
                    if (!similarityDiagnostics) {
                        double entropy = 0.0;
                        for (int32_t k = 0; k < K; ++k) {
                            const double probability = theta(k);
                            if (probability > 0.0) {
                                entropy -= probability
                                    * std::log(probability);
                            }
                        }
                        unitEntropy(static_cast<int32_t>(d)) = entropy;
                    }
                    double residual = documentTotal;
                    double dotProduct = 0.0;
                    double observedNormSq = 0.0;
                    size_t positiveCells = 0;
                    for (double count : doc.cnts) {
                        positiveCells += count > 0.0;
                    }
                    const bool sparsePrediction =
                        3.0 * static_cast<double>(positiveCells + K) < M;
                    if (!sparsePrediction) {
                        if (local.denseExpected.size() != M) {
                            local.denseExpected.resize(M);
                        }
                        local.denseExpected.noalias() =
                            docTopic.row(static_cast<int32_t>(d))
                            * residualState->betaNormRow;
                        local.denseExpected *= documentTotal;
                    }
                    for (size_t j = 0; j < doc.ids.size(); ++j) {
                        const uint32_t w = doc.ids[j];
                        const double observed = doc.cnts[j];
                        const double expected = sparsePrediction
                            ? documentTotal * theta.dot(
                                residualState->betaNormCol.col(w))
                            : local.denseExpected(w);
                        expectedCells[documentOffsets[d] + j] = expected;
                        residual += std::abs(expected - observed) - expected;
                        if (similarityDiagnostics) {
                            dotProduct += expected * observed;
                            observedNormSq += observed * observed;
                        }
                    }
                    if (similarityDiagnostics
                            && expectedNormSq(static_cast<int32_t>(d)) > 0.0
                            && observedNormSq > 0.0) {
                        unitCosine(static_cast<int32_t>(d)) =
                            dotProduct
                            / std::sqrt(
                                expectedNormSq(static_cast<int32_t>(d))
                                * observedNormSq);
                    }
                    unitResidual(static_cast<int32_t>(d)) =
                        std::max(0.0, residual);
                }
            });

        const feature_diagnostics::FeatureCellIndex featureIndex =
            feature_diagnostics::make_feature_cell_index(docs, M, nCells);

        std::vector<PullRecord> batchPullRecords;
        if (residualState->diagnosticSpool.storesPullRecords()) {
            batchPullRecords.resize(nCells);
        }
        const size_t featureGrain = std::max<size_t>(
            1, static_cast<size_t>(M)
                / (2 * static_cast<size_t>(threadHint)));
        tbb::parallel_for(tbb::blocked_range<size_t>(
                0, static_cast<size_t>(M), featureGrain),
            [&](const tbb::blocked_range<size_t>& range) {
                VectorXd allocation(K);
                VectorXd deletedTopic(K);
                VectorXd assigned = VectorXd::Zero(K);
                for (size_t w0 = range.begin(); w0 < range.end(); ++w0) {
                    if (featureIndex.featureOffsets[w0]
                            == featureIndex.featureOffsets[w0 + 1]) {
                        continue;
                    }
                    const int32_t w = static_cast<int32_t>(w0);
                    double correction = 0.0;
                    double total = 0.0;
                    double conditionalLogTerm = 0.0;
                    double deletionNumerator = 0.0;
                    feature_diagnostics::CofeatureLiftSums cofeatureSums;
                    int64_t units = 0;
                    assigned.setZero();

                    for (size_t p = featureIndex.featureOffsets[w0];
                            p < featureIndex.featureOffsets[w0 + 1]; ++p) {
                        const feature_diagnostics::CellRef ref =
                            featureIndex.cellsByFeature[p];
                        const size_t d = ref.document;
                        const size_t j = ref.offset;
                        const Document& doc = docs[d];
                        const double observed = doc.cnts[j];
                        const double expected =
                            expectedCells[documentOffsets[d] + j];
                        correction +=
                            std::abs(expected - observed) - expected;
                        total += observed;
                        if (observed <= 0.0) continue;
                        if (!std::isfinite(expected) || expected <= 0.0) {
                            error("%s: non-positive fitted mean for feature %d",
                                __func__, w);
                        }
                        ++units;
                        conditionalLogTerm +=
                            observed * (std::log(observed)
                                - std::log(expected));
                        if (residualState->useTrainingPrevalence) {
                            feature_diagnostics::accumulate_cofeature_lift(
                                residualState->cofeatureModel,
                                cofeatureContext, d, w, cofeatureSums);
                        }

                        allocation =
                            thetaKernel.row(static_cast<int32_t>(d))
                                .transpose().array()
                            * residualState->betaAllocationKernel.col(w).array();
                        const double allocationTotal = allocation.sum();
                        if (!std::isfinite(allocationTotal)
                                || allocationTotal <= 0.0) {
                            error("%s: invalid topic allocation for feature %d",
                                __func__, w);
                        }
                        allocation /= allocationTotal;
                        assigned.noalias() += observed * allocation;

                        double deletedTotal = 0.0;
                        for (int32_t k = 0; k < K; ++k) {
                            deletedTopic(k) = std::max(0.0,
                                gamma(static_cast<int32_t>(d), k)
                                - observed * allocation(k));
                            deletedTotal += deletedTopic(k);
                        }
                        if (deletedTotal > 0.0
                                && std::isfinite(deletedTotal)) {
                            deletedTopic /= deletedTotal;
                        } else {
                            deletedTopic.setConstant(
                                1.0 / static_cast<double>(K));
                        }
                        double deletionVariation = 0.0;
                        for (int32_t k = 0; k < K; ++k) {
                            const double deletedProbability =
                                deletedTopic(k);
                            deletionVariation += std::abs(
                                deletedProbability
                                - docTopic(static_cast<int32_t>(d), k));
                        }
                        deletionVariation *= 0.5;
                        deletionNumerator += deletionVariation;

                        if (residualState->useTrainingPrevalence
                                || residualState->diagnosticSpool
                                    .storesPullRecords()) {
                            const double totalVariation = 0.5
                                * (allocation - deletedTopic)
                                    .cwiseAbs().sum();
                            if (residualState->useTrainingPrevalence) {
                                residualState->pullNumerators(w) +=
                                    std::abs(observed - expected)
                                    * totalVariation;
                            } else {
                                PullRecord& record = batchPullRecords[
                                    documentOffsets[d] + j];
                                record.feature = static_cast<uint32_t>(w);
                                record.observed = observed;
                                record.mean = expected;
                                record.totalVariation = totalVariation;
                            }
                        }
                    }

                    residualState->featureCorrections(w) += correction;
                    residualState->featureTotals(w) += total;
                    residualState->conditionalLogTerms(w) +=
                        conditionalLogTerm;
                    residualState->deletionNumerators(w) +=
                        deletionNumerator;
                    residualState->cofeatureCorroborationSums(w) +=
                        cofeatureSums.corroboration;
                    residualState->cofeatureConflictSums(w) +=
                        cofeatureSums.conflict;
                    residualState->cofeatureContextUnits[w0] +=
                        cofeatureSums.contextUnits;
                    residualState->featureUnits[w0] += units;
                    residualState->allocatedTopicCounts.col(w).noalias() +=
                        assigned;
                }
            });

        residualState->diagnosticSpool.append(
            docs, documentOffsets, batchPullRecords);

        transform_helpers::writeUnitStatsRows(
            *unitMetaStream, docs, ids, unitResidual, unitCosine,
            unitEntropy, similarityEntropy.sh_lcr, similarityEntropy.sh_q,
            similarityDiagnostics);
    }

    void finalizeResiduals() {
        feature_diagnostics::finalize_feature_residuals(
            *residualState, residualState->betaNormRow,
            residualState->betaNormCol,
            residualState->topicExposureTotals, threadHint);
    }

    void writeTopicRows(const std::vector<std::string>& ids, const RowMajorMatrixXd& doc_topic) {
        if (topkOnly > 0) {
            writeTopKRows(ids, doc_topic);
            return;
        }
        transform_helpers::writeTopicRows(resultsStream, ids, doc_topic);
    }

    void writeTopKRows(const std::vector<std::string>& ids, const RowMajorMatrixXd& doc_topic) {
        const int32_t topk = std::min(topkOnly, K);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!ids[i].empty()) {
                resultsStream << ids[i] << "\t";
            }
            std::vector<std::pair<double, int32_t>> ranked;
            ranked.reserve(K);
            for (int32_t k = 0; k < K; ++k) {
                ranked.emplace_back(doc_topic(i, k), k);
            }
            std::partial_sort(ranked.begin(), ranked.begin() + topk, ranked.end(),
                [](const auto& a, const auto& b) {
                    if (a.first != b.first) {
                        return a.first > b.first;
                    }
                    return a.second < b.second;
                });

            resultsStream << ranked[0].second;
            for (int32_t j = 1; j < topk; ++j) {
                resultsStream << "\t" << ranked[j].second;
            }
            for (int32_t j = 0; j < topk; ++j) {
                resultsStream << "\t" << ranked[j].first;
            }
            resultsStream << "\n";
        }
    }

    LDA4Hex& lda;
    std::ofstream& resultsStream;
    std::ofstream* unitMetaStream;
    RowMajorMatrixXd& pseudobulk;
    MatrixXd& specialPseudobulk;
    Mode pseudobulkMode;
    ResidualState* residualState;
    int32_t topkOnly;
    bool similarityDiagnostics;
    int32_t threadHint;
    int32_t M;
    int32_t K;
    std::unique_ptr<StandardTls> standardTls;
    std::unique_ptr<ResidualTls> residualTls;
};

} // namespace

int32_t cmdLDATransform(int argc, char** argv) {
    std::string inFile, metaFile, modelFile, outPrefix, featureFile, temp_dir;
    std::vector<std::string> dge_dirs, in_bc, in_ft, in_mtx, dataset_ids;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t seed = -1;
    int32_t batchSize = 1024;
    int32_t nThreads = 1;
    int32_t modal = 0;
    int32_t minCountFeature = 1;
    int32_t icolWeight = -1;
    double minCount = 20;
    int32_t debug_ = 0;
    int32_t verbose = 0;
    double defaultWeight = -1.0;
    int32_t maxIter = 100;
    double mDelta = 1e-3;
    int32_t topk_only = -1;
    bool computeResiduals = false;
    bool cheap_feature_diagnostics = false;
    bool use_training_prevalence = false;
    bool unit_similarity_diagnostics = false;
    bool sorted_by_barcode = false;
    bool keep_barcodes = false;
    bool pseudobulk_all_features = false;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("in-model", "Input model matrix (topic-word) file", modelFile, true)
      .add_option("out-prefix", "Output prefix for results files", outPrefix, true)
      .add_option("minibatch-size", "Minibatch size", batchSize)
      .add_option("modal", "Modality to use (0-based)", modal)
      .add_option("threads", "Number of threads", nThreads)
      .add_option("temp-dir", "Directory to store temporary files", temp_dir)
      .add_option("seed", "Random seed", seed)
      .add_option("verbose", "Verbose level", verbose)
      .add_option("debug", "If >0, only process this many units", debug_);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", dataset_ids)
      .add_option("sorted-by-barcode", "Input matrix is sorted by barcode, use streaming mode", sorted_by_barcode)
      .add_option("keep-barcodes", "For 10X input, write IDs from barcodes.tsv.gz instead of 0-based barcode indices", keep_barcodes);

    pl.add_option("features", "Feature names and total counts file", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", minCountFeature)
      .add_option("min-count", "Minimum total feature count for a unit to be kept", minCount)
      .add_option("default-weight", "Default weight for model features missing from --features when feature weights are active; <0 drops missing features", defaultWeight)
      .add_option("icol-weight", "0-based column index for feature weight in --features; <0 disables feature weights", icolWeight)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex);

    pl.add_option("max-iter", "Max iterations per document", maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per document", mDelta)
      .add_option("feature-residuals", "Compute per-feature and per-unit residuals (backward compatibility)", computeResiduals)
      .add_option("residuals", "Compute per-feature and per-unit residuals", computeResiduals)
      .add_option("feature-diagnostics-cheap", "Skip spool-dependent gain-adjusted feature residual and Pull diagnostics", cheap_feature_diagnostics)
      .add_option("use-training-prevalence", "Use fitted training prevalence for feature diagnostics", use_training_prevalence)
      .add_option("unit-diagnostics-similarity", "Add cosine and similarity-adjusted entropy unit diagnostics", unit_similarity_diagnostics)
      .add_option("pseudobulk-all-features", "Include all retained input features in pseudobulk output", pseudobulk_all_features)
      .add_option("topk-only", "Write only top-k factor indices/probabilities to results.tsv", topk_only);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (batchSize <= 0) {
        batchSize = 512;
        warning("Minibatch size must be greater than 0, using default value of %d", batchSize);
    }
    if (topk_only == 0) {
        error("--topk-only must be a positive integer");
    }
    if (cheap_feature_diagnostics && !computeResiduals) {
        error("--feature-diagnostics-cheap requires --residuals");
    }
    if (use_training_prevalence && !computeResiduals) {
        error("--use-training-prevalence requires --residuals");
    }
    if (use_training_prevalence && cheap_feature_diagnostics) {
        warning("--feature-diagnostics-cheap has no effect with "
            "--use-training-prevalence");
        cheap_feature_diagnostics = false;
    }
    if (unit_similarity_diagnostics && !computeResiduals) {
        error("--unit-diagnostics-similarity requires --residuals");
    }
    if (seed <= 0) {
        seed = std::random_device{}();
    }
    const bool weights_active = !featureFile.empty() && icolWeight >= 0;
    if (defaultWeight < 0.0) {
        defaultWeight = -1.0;
    }

    const auto dge_inputs = resolveDge10XInputs(dge_dirs, in_bc, in_ft, in_mtx, dataset_ids);
    bool use_10x = !dge_inputs.empty();
    if (use_10x && !inFile.empty()) {
        warning("Both --in-data and 10X inputs are provided; using 10X inputs and ignoring --in-data");
    }
    if (!use_10x && inFile.empty()) {
        error("Either --in-data or 10X inputs must be provided");
    }
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    if (use_10x) {
        dge_ptr = makeDGEReader10X(dge_dirs, in_bc, in_ft, in_mtx, dataset_ids, keep_barcodes);
        reader.initFromFeatures(dge_ptr->features, dge_ptr->nBarcodes);
    } else {
        if (metaFile.empty()) {
            error("Missing required --in-meta for non-10X input");
        }
        reader.readMetadata(metaFile);
    }
    if (!featureFile.empty()) {
        if (weights_active) {
            reader.setFeatureFilterAndWeights(featureFile, minCountFeature,
                include_ftr_regex, exclude_ftr_regex, icolWeight, defaultWeight,
                defaultWeight >= 0.0);
        } else {
            reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
        }
    }

    const Mode pseudobulkMode =
        transform_pseudobulk::selectMode(pseudobulk_all_features, weights_active);
    std::unique_ptr<HexReader> rawReader;
    std::vector<std::string> retainedInputFeatures;
    if (pseudobulkMode != Mode::Standard) {
        rawReader = std::make_unique<HexReader>(reader);
        rawReader->clearFeatureWeights();
        retainedInputFeatures = rawReader->features;
    }

    std::string info_header;
    if (!use_10x) {
        reader.getInfoHeaderStr(info_header);
    }

    LDA4Hex lda(reader, modal, verbose);
    lda.initialize_transform(modelFile,
        seed, nThreads, verbose, maxIter, mDelta);

    const int32_t M = lda.nFeatures();
    const int32_t K = lda.getNumTopics();
    const std::vector<std::string> modelFeatureNames = lda.getFeatureNames();
    if (topk_only > 0 && topk_only > K-1) {
        warning("--topk-only is >= the number of topics (%d); writing all topics", K);
        topk_only = -1;
    }

    std::vector<std::string> pseudobulkFeatureNames = modelFeatureNames;
    std::vector<int32_t> inputToModel;
    if (pseudobulkMode == Mode::WeightedModel) {
        rawReader->setFeatureIndexRemap(pseudobulkFeatureNames, false);
    } else if (pseudobulkMode == Mode::AllFeatures) {
        pseudobulkFeatureNames = retainedInputFeatures;
        inputToModel = transform_pseudobulk::mapInputFeaturesToModel(
            pseudobulkFeatureNames, modelFeatureNames);
    }

    RowMajorMatrixXd pseudobulk;
    MatrixXd specialPseudobulk;
    if (pseudobulkMode == Mode::Standard) {
        pseudobulk = RowMajorMatrixXd::Zero(M, K);
    } else {
        specialPseudobulk = MatrixXd::Zero(
            static_cast<int32_t>(pseudobulkFeatureNames.size()), K);
    }
    std::unique_ptr<ResidualState> residualState;
    if (computeResiduals) {
        residualState = std::make_unique<ResidualState>(
            lda.get_model_matrix(), cheap_feature_diagnostics,
            unit_similarity_diagnostics, use_training_prevalence,
            temp_dir);
    }

    TransformOutputs outputs(outPrefix, computeResiduals);
    writeUnitIdHeader(outputs.results, use_10x, info_header);
    writeResultHeader(outputs.results, lda, topk_only);
    outputs.results << std::fixed << std::setprecision(4);
    if (computeResiduals) {
        writeUnitIdHeader(outputs.unitStats, use_10x, info_header);
        outputs.unitStats << "total_count\tresidual\tentropy";
        if (unit_similarity_diagnostics) {
            outputs.unitStats << "\tcosine_sim\tsh_lcr\tsh_q";
        }
        outputs.unitStats << "\n";
        outputs.unitStats << std::fixed;
    }

    TransformBatchProcessor processor(lda, outputs.results,
        computeResiduals ? &outputs.unitStats : nullptr,
        pseudobulk, specialPseudobulk, pseudobulkMode,
        residualState.get(), topk_only,
        unit_similarity_diagnostics, nThreads);

    bool fileopen = true;
    int32_t processed = 0;
    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    TransformBatch batch;
    const int32_t minCountInt = minCount > 0 ? static_cast<int32_t>(std::ceil(minCount)) : 0;

    if (pseudobulkMode == Mode::Standard) {
        if (use_10x) {
            DGEReader10X& dge = *dge_ptr;
            const std::vector<std::string> model_features = lda.getFeatureNames();
            int32_t n_overlap = dge.setFeatureIndexRemap(model_features, false);
            if (n_overlap == 0) {
                error("No overlapping features found between 10X input and model metadata");
            }

            std::vector<int32_t> barcode_idx;

            if (sorted_by_barcode) {
                while (fileopen && processed < maxUnits) {
                    batch.clear();
                    const int32_t remaining = maxUnits - processed;
                    fileopen = dge.readMinibatch(batch.docs, barcode_idx, batchSize, remaining, minCountInt);
                    if (batch.empty()) {
                        break;
                    }
                    applyWeights(batch.docs, lda);
                    assignBarcodeIds(dge, barcode_idx, batch.ids);
                    processor.process(batch);
                    processed += static_cast<int32_t>(batch.size());
                }
            } else {
                std::vector<Document> all_docs;
                std::vector<int32_t> all_barcode_idx;
                dge.readAll(all_docs, all_barcode_idx, minCountInt);
                applyWeights(all_docs, lda);
                std::vector<std::string> all_ids;
                assignBarcodeIds(dge, all_barcode_idx, all_ids);
                size_t cursor = 0;
                while (cursor < all_docs.size() && processed < maxUnits) {
                    batch.clear();
                    const int32_t remaining = maxUnits - processed;
                    size_t take = std::min(static_cast<size_t>(batchSize), all_docs.size() - cursor);
                    if (take > static_cast<size_t>(remaining)) {
                        take = static_cast<size_t>(remaining);
                    }
                    batch.docs.insert(batch.docs.end(),
                        std::make_move_iterator(all_docs.begin() + cursor),
                        std::make_move_iterator(all_docs.begin() + cursor + take));
                    batch.ids.insert(batch.ids.end(),
                        std::make_move_iterator(all_ids.begin() + cursor),
                        std::make_move_iterator(all_ids.begin() + cursor + take));
                    cursor += take;
                    if (batch.empty()) {
                        break;
                    }
                    processor.process(batch);
                    processed += static_cast<int32_t>(batch.size());
                }
            }
        } else {
            std::ifstream inFileStream(inFile);
            if (!inFileStream) error("Error opening input file: %s", inFile.c_str());
            while (fileopen && processed < maxUnits) {
                batch.clear();
                const int32_t remaining = maxUnits - processed;
                fileopen = lda.readMinibatch(inFileStream, batch.docs, batch.ids, batchSize, minCountInt, remaining);
                if (batch.empty()) break;
                processor.process(batch);
                processed += static_cast<int32_t>(batch.size());
            }
            inFileStream.close();
        }
    } else {
        SpecialBatch specialBatch;
        if (use_10x) {
            DGEReader10X& dge = *dge_ptr;
            const std::vector<std::string>& readFeatures = pseudobulkFeatureNames;
            int32_t n_overlap = dge.setFeatureIndexRemap(readFeatures, false);
            if (n_overlap == 0) {
                error("No retained input features overlap with 10X input");
            }

            if (sorted_by_barcode) {
                while (fileopen && processed < maxUnits) {
                    const int32_t remaining = maxUnits - processed;
                    fileopen = readSpecialDgeMinibatch(dge, lda, specialBatch,
                        pseudobulkMode, inputToModel, batchSize, remaining,
                        minCountInt);
                    if (specialBatch.empty()) {
                        break;
                    }
                    processor.process(specialBatch);
                    processed += static_cast<int32_t>(specialBatch.size());
                }
            } else {
                std::vector<Document> allRawDocs;
                std::vector<int32_t> allUnitIndices;
                dge.readAll(allRawDocs, allUnitIndices, 0);
                size_t cursor = 0;
                while (cursor < allRawDocs.size() && processed < maxUnits) {
                    specialBatch.clear();
                    const int32_t target = std::min(batchSize, maxUnits - processed);
                    while (cursor < allRawDocs.size() &&
                            static_cast<int32_t>(specialBatch.size()) < target) {
                        const int32_t unitIndex = allUnitIndices[cursor];
                        transform_pseudobulk::appendDocument(
                            std::move(allRawDocs[cursor]), dge.getUnitId(unitIndex),
                            specialBatch, pseudobulkMode, inputToModel, lda,
                            minCountInt);
                        ++cursor;
                    }
                    if (specialBatch.empty()) {
                        continue;
                    }
                    processor.process(specialBatch);
                    processed += static_cast<int32_t>(specialBatch.size());
                }
            }
        } else {
            std::ifstream inFileStream(inFile);
            if (!inFileStream) error("Error opening input file: %s", inFile.c_str());
            while (fileopen && processed < maxUnits) {
                const int32_t remaining = maxUnits - processed;
                fileopen = readSpecialHexMinibatch(inFileStream, *rawReader,
                    lda, specialBatch, pseudobulkMode, inputToModel, modal,
                    batchSize, remaining, minCountInt);
                if (specialBatch.empty()) {
                    break;
                }
                processor.process(specialBatch);
                processed += static_cast<int32_t>(specialBatch.size());
            }
            inFileStream.close();
        }
    }
    processor.finalize();
    outputs.results.close();
    notice("Transformation results written to %s", outputs.resultsPath.c_str());
    if (computeResiduals) {
        outputs.unitStats.close();
        notice("Per-unit residuals written to %s", outputs.unitStatsPath.c_str());
    }

    std::string outFile = outPrefix + ".pseudobulk.tsv";
    std::ofstream outFileStream(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());
    outFileStream << "Feature\t";
    lda.writeModelHeader(outFileStream);
    transform_helpers::writePseudobulkRows(
        outFileStream, pseudobulkFeatureNames, pseudobulk,
        specialPseudobulk, pseudobulkMode, K);
    outFileStream.close();
    notice("Pseudobulk counts written to %s", outFile.c_str());

    if (!computeResiduals) return 0;
    outFile = outPrefix + ".feature_residuals.tsv";
    outFileStream.open(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());
    feature_diagnostics::write_feature_residuals(
        outFileStream, modelFeatureNames, *residualState);
    outFileStream.close();
    notice("Per-feature residuals written to %s", outFile.c_str());

    return 0;
}
