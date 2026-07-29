#include "gamma_pois_topic.hpp"
#include "transform_pseudobulk.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace {

using transform_pseudobulk::Mode;
using transform_pseudobulk::SpecialBatch;

struct TransformBatch {
    std::vector<Document> docs;
    std::vector<std::string> ids;

    void clear() {
        docs.clear();
        ids.clear();
    }
    bool empty() const {
        return docs.empty();
    }
    size_t size() const {
        return docs.size();
    }
};

void writeUnitIdHeader(std::ostream& out, bool use_10x, const std::string& info_header) {
    if (use_10x) {
        out << "#barcode\t";
        return;
    }
    out << "#";
    if (!info_header.empty()) {
        out << info_header << "\t";
    }
}

void assignBarcodeIds(const DGEReader10X& dge,
    const std::vector<int32_t>& barcode_idx, std::vector<std::string>& ids) {
    ids.clear();
    ids.reserve(barcode_idx.size());
    for (auto idx : barcode_idx) {
        if (idx >= 0 && idx < static_cast<int32_t>(dge.barcodes.size())) {
            ids.push_back(dge.barcodes[idx]);
        } else {
            ids.push_back(std::to_string(idx));
        }
    }
}

void applyWeights(std::vector<Document>& docs, GammaPoisson4Hex& gp) {
    for (auto& doc : docs) {
        gp.applyWeights(doc);
    }
}

int64_t rawTotalCount(const Document& doc) {
    const double rawTotal = doc.raw_ct_tot >= 0.0
        ? doc.raw_ct_tot
        : std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
    return static_cast<int64_t>(std::llround(rawTotal));
}

struct PullRecord {
    uint32_t feature = 0;
    double observed = 0.0;
    double mean = 0.0;
    double totalVariation = 0.0;
};

class PullSpool {
public:
    explicit PullSpool(bool enabled) : enabled_(enabled) {
        if (enabled_) {
            file_ = std::tmpfile();
            if (!file_) {
                error("Unable to create temporary feature diagnostic spool; "
                    "use --feature-diagnostics-cheap to disable spool-dependent statistics");
            }
        }
    }

    ~PullSpool() {
        if (file_) {
            std::fclose(file_);
        }
    }

    PullSpool(const PullSpool&) = delete;
    PullSpool& operator=(const PullSpool&) = delete;

    bool enabled() const {
        return enabled_;
    }

    void append(const std::vector<PullRecord>& records) {
        if (!enabled_ || records.empty()) return;
        if (std::fwrite(records.data(), sizeof(PullRecord), records.size(), file_)
                != records.size()) {
            error("Error writing temporary feature diagnostic spool");
        }
    }

    void accumulate(const VectorXd& gain, VectorXd& adjustedResidual,
            VectorXd& pull) {
        if (!enabled_) return;
        if (std::fflush(file_) != 0 || std::fseek(file_, 0, SEEK_SET) != 0) {
            error("Error rewinding temporary feature diagnostic spool");
        }
        std::vector<PullRecord> buffer(4096);
        while (true) {
            const size_t n = std::fread(
                buffer.data(), sizeof(PullRecord), buffer.size(), file_);
            for (size_t i = 0; i < n; ++i) {
                const PullRecord& record = buffer[i];
                if (record.feature >= static_cast<uint32_t>(gain.size())) {
                    error("Invalid feature index in temporary diagnostic spool");
                }
                const double a = gain(record.feature);
                if (!std::isfinite(a) || a < 0.0) continue;
                const double difference =
                    std::abs(record.observed - a * record.mean);
                adjustedResidual(record.feature) += difference;
                pull(record.feature) +=
                    difference * record.totalVariation;
            }
            if (n < buffer.size()) {
                if (std::ferror(file_)) {
                    error("Error reading temporary feature diagnostic spool");
                }
                break;
            }
        }
    }

private:
    bool enabled_ = false;
    std::FILE* file_ = nullptr;
};

struct ResidualState {
    const MatrixXd& expectedBeta;
    const MatrixXd& betaAllocationKernel;
    const VectorXd& topicCapacity;
    const VectorXd& featureDispersion;
    bool hasFeatureDispersion;
    MatrixXd profileGram;
    MatrixXd topicSimilarity;
    MatrixXd allocatedTopicCounts;
    VectorXd featureCorrections;
    VectorXd featureTotals;
    VectorXd topicExposureTotals;
    VectorXd conditionalLogTerms;
    VectorXd deletionNumerators;
    std::vector<int64_t> featureUnits;
    PullSpool pullSpool;

    VectorXd predictedTotals;
    VectorXd log2Gain;
    VectorXd marginalDeviance;
    VectorXd conditionalDeviance;
    VectorXd topicDeviance;
    VectorXd deletionTV;
    VectorXd adjustedResidualPerCount;
    VectorXd pull;

    ResidualState(GammaPoisson4Hex& gp, bool cheapDiagnostics,
            bool similarityDiagnostics)
        : expectedBeta(gp.getExpectedBeta()),
          betaAllocationKernel(gp.getBetaAllocationKernel()),
          topicCapacity(gp.getTopicCapacity()),
          featureDispersion(gp.getFeatureDispersion()),
          hasFeatureDispersion(gp.hasFeatureDispersion()),
          allocatedTopicCounts(MatrixXd::Zero(
              expectedBeta.rows(), expectedBeta.cols())),
          featureCorrections(VectorXd::Zero(expectedBeta.cols())),
          featureTotals(VectorXd::Zero(expectedBeta.cols())),
          topicExposureTotals(VectorXd::Zero(expectedBeta.rows())),
          conditionalLogTerms(VectorXd::Zero(expectedBeta.cols())),
          deletionNumerators(VectorXd::Zero(expectedBeta.cols())),
          featureUnits(static_cast<size_t>(expectedBeta.cols()), 0),
          pullSpool(!cheapDiagnostics) {
        if (!similarityDiagnostics) {
            return;
        }
        profileGram.noalias() =
            expectedBeta * expectedBeta.transpose();
        topicSimilarity = MatrixXd::Zero(
            expectedBeta.rows(), expectedBeta.rows());
        VectorXd norms = profileGram.diagonal().array().max(0.0).sqrt();
        for (int32_t k = 0; k < expectedBeta.rows(); ++k) {
            topicSimilarity(k, k) = 1.0;
            for (int32_t l = k + 1; l < expectedBeta.rows(); ++l) {
                const double denom = norms(k) * norms(l);
                const double similarity = denom > 0.0
                    ? std::clamp(profileGram(k, l) / denom, 0.0, 1.0)
                    : 0.0;
                topicSimilarity(k, l) = similarity;
                topicSimilarity(l, k) = similarity;
            }
        }
    }
};

void randomizeDocuments(std::vector<Document>& docs, std::vector<std::string>& ids,
    std::mt19937& random_engine) {
    std::vector<size_t> order(docs.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), random_engine);
    std::vector<Document> shuffled_docs;
    std::vector<std::string> shuffled_ids;
    shuffled_docs.reserve(docs.size());
    shuffled_ids.reserve(ids.size());
    for (size_t i : order) {
        shuffled_docs.push_back(std::move(docs[i]));
        shuffled_ids.push_back(std::move(ids[i]));
    }
    docs = std::move(shuffled_docs);
    ids = std::move(shuffled_ids);
}

class GammaPoisTransformBatchProcessor {
public:
    GammaPoisTransformBatchProcessor(GammaPoisson4Hex& gp_,
        std::ostream& results_, RowMajorMatrixXd& pseudobulk_,
        MatrixXd& specialPseudobulk_, Mode pseudobulkMode_,
        std::ostream* unitStats_, ResidualState* residualState_,
        bool similarityDiagnostics_, int32_t nThreads_)
        : gp(gp_), results(results_), pseudobulk(pseudobulk_),
          specialPseudobulk(specialPseudobulk_), pseudobulkMode(pseudobulkMode_),
          unitStats(unitStats_), residualState(residualState_),
          similarityDiagnostics(similarityDiagnostics_),
          threadHint(std::max(1, nThreads_)),
          M(gp_.nFeatures()), K(gp_.getNumTopics()) {
        if (residualState) {
            residualTls = std::make_unique<ResidualTls>([this] {
                return ResidualLocalAgg(K);
            });
        }
    }

    void process(TransformBatch& batch) {
        if (batch.empty()) {
            return;
        }
        RowMajorMatrixXd doc_topic;
        std::vector<GammaPoissonDocumentPosterior> posteriors;
        if (residualState) {
            gp.transformWithPosteriors(
                DocumentView(batch.docs), doc_topic, posteriors);
        } else {
            doc_topic = gp.transformMeans(DocumentView(batch.docs));
        }
        writeTopicRows(batch.ids, doc_topic);
        processResiduals(batch.docs, batch.ids, doc_topic, posteriors);
        for (size_t i = 0; i < batch.docs.size(); ++i) {
            const Document& doc = batch.docs[i];
            for (size_t j = 0; j < doc.ids.size(); ++j) {
                const uint32_t m = doc.ids[j];
                const double raw_count = gp.rawCountFor(m, doc.cnts[j], doc.counts_weighted);
                for (int32_t k = 0; k < K; ++k) {
                    pseudobulk(m, k) += raw_count * doc_topic(i, k);
                }
            }
        }
    }

    void process(SpecialBatch& batch) {
        if (batch.empty()) {
            return;
        }
        if (pseudobulkMode == Mode::Standard) {
            error("%s: special batch used with standard pseudobulk mode", __func__);
        }
        RowMajorMatrixXd doc_topic;
        std::vector<GammaPoissonDocumentPosterior> posteriors;
        if (residualState) {
            gp.transformWithPosteriors(
                DocumentView(batch.modelDocs), doc_topic, posteriors);
        } else {
            doc_topic = gp.transformMeans(DocumentView(batch.modelDocs));
        }
        writeTopicRows(batch.ids, doc_topic);
        processResiduals(batch.modelDocs, batch.ids, doc_topic, posteriors);
        transform_pseudobulk::accumulate(
            specialPseudobulk, batch, doc_topic, pseudobulkMode);
    }

    void finalizeResiduals() {
        if (!residualState) return;
        for (auto& local : *residualTls) {
            residualState->topicExposureTotals += local.topicExposureTotals;
        }
        residualState->predictedTotals.noalias() =
            residualState->expectedBeta.transpose()
            * residualState->topicExposureTotals;
        residualState->featureCorrections.noalias() +=
            residualState->predictedTotals;

        const double nan = std::numeric_limits<double>::quiet_NaN();
        residualState->log2Gain = VectorXd::Constant(M, nan);
        residualState->marginalDeviance = VectorXd::Constant(M, nan);
        residualState->conditionalDeviance = VectorXd::Constant(M, nan);
        residualState->topicDeviance = VectorXd::Constant(M, nan);
        residualState->deletionTV = VectorXd::Constant(M, nan);
        residualState->adjustedResidualPerCount =
            VectorXd::Constant(M, nan);
        residualState->pull = VectorXd::Constant(M, nan);

        VectorXd gain = VectorXd::Constant(M, nan);
        for (int32_t w = 0; w < M; ++w) {
            const double observed = residualState->featureTotals(w);
            const double predicted = residualState->predictedTotals(w);
            if (!std::isfinite(observed) || observed < 0.0
                    || !std::isfinite(predicted) || predicted <= 1e-300) {
                continue;
            }
            gain(w) = observed / predicted;
            if (observed == 0.0) {
                residualState->log2Gain(w) =
                    -std::numeric_limits<double>::infinity();
                residualState->marginalDeviance(w) = 2.0 * predicted;
                residualState->conditionalDeviance(w) = 0.0;
                residualState->topicDeviance(w) = 0.0;
                continue;
            }

            residualState->log2Gain(w) = std::log2(gain(w));
            residualState->marginalDeviance(w) = std::max(0.0,
                2.0 * (observed * std::log(observed / predicted)
                    - (observed - predicted)));
            residualState->conditionalDeviance(w) = std::max(0.0,
                2.0 * (residualState->conditionalLogTerms(w)
                    - observed * std::log(gain(w))));
            double topicDeviance = 0.0;
            for (int32_t k = 0; k < K; ++k) {
                const double allocated =
                    residualState->allocatedTopicCounts(k, w);
                if (allocated <= 0.0) continue;
                const double expected = gain(w)
                    * residualState->expectedBeta(k, w)
                    * residualState->topicExposureTotals(k);
                if (expected <= 0.0 || !std::isfinite(expected)) {
                    topicDeviance =
                        std::numeric_limits<double>::infinity();
                    break;
                }
                topicDeviance +=
                    2.0 * allocated * std::log(allocated / expected);
            }
            residualState->topicDeviance(w) =
                std::max(0.0, topicDeviance);
            residualState->deletionTV(w) =
                residualState->deletionNumerators(w) / observed;
        }

        VectorXd adjustedNumerator = VectorXd::Zero(M);
        VectorXd pullNumerator = VectorXd::Zero(M);
        residualState->pullSpool.accumulate(
            gain, adjustedNumerator, pullNumerator);
        if (residualState->pullSpool.enabled()) {
            for (int32_t w = 0; w < M; ++w) {
                const double observed = residualState->featureTotals(w);
                if (observed > 0.0 && std::isfinite(gain(w))) {
                    residualState->adjustedResidualPerCount(w) =
                        adjustedNumerator(w) / observed;
                    residualState->pull(w) =
                        pullNumerator(w) / observed;
                }
            }
        }
    }

private:
    struct ResidualLocalAgg {
        VectorXd topicExposureTotals;
        RowVectorXd denseExpected;

        explicit ResidualLocalAgg(int32_t nTopics)
            : topicExposureTotals(VectorXd::Zero(nTopics)) {}
    };

    struct CellRef {
        uint32_t document = 0;
        uint32_t offset = 0;
    };

    using ResidualTls =
        tbb::enumerable_thread_specific<ResidualLocalAgg>;

    void writeTopicRows(const std::vector<std::string>& ids,
        const RowMajorMatrixXd& doc_topic) {
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!ids[i].empty()) {
                results << ids[i] << "\t";
            }
            results << doc_topic(i, 0);
            for (int32_t k = 1; k < K; ++k) {
                results << "\t" << doc_topic(i, k);
            }
            results << "\n";
        }
    }

    void processResiduals(const std::vector<Document>& docs,
        const std::vector<std::string>& ids,
        const RowMajorMatrixXd& docTopic,
        const std::vector<GammaPoissonDocumentPosterior>& posteriors) {
        if (!residualState) return;
        const int32_t nDocs = static_cast<int32_t>(docs.size());
        std::vector<size_t> documentOffsets(
            static_cast<size_t>(nDocs) + 1, 0);
        for (int32_t d = 0; d < nDocs; ++d) {
            documentOffsets[static_cast<size_t>(d) + 1] =
                documentOffsets[static_cast<size_t>(d)]
                + docs[static_cast<size_t>(d)].ids.size();
        }
        const size_t nCells = documentOffsets.back();
        std::vector<double> expectedCells(nCells, 0.0);
        RowMajorMatrixXd thetaKernel(nDocs, K);
        VectorXd unitResidual = VectorXd::Zero(nDocs);
        VectorXd unitCosine;
        VectorXd unitEntropy = VectorXd::Zero(nDocs);
        ThetaEntropyStats similarityEntropy;
        RowMajorMatrixXd expectedTopicWeights;
        VectorXd expectedNormSq;
        const size_t grainSize = std::max<size_t>(
            1, docs.size() / (2 * static_cast<size_t>(threadHint)));
        if (similarityDiagnostics) {
            unitCosine = VectorXd::Zero(nDocs);
            similarityEntropy = computeThetaEntropyStats(
                docTopic, residualState->topicSimilarity);
            unitEntropy = similarityEntropy.entropy;
            expectedTopicWeights.resize(nDocs, K);
            tbb::parallel_for(tbb::blocked_range<size_t>(
                    0, docs.size(), grainSize),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t i = range.begin(); i < range.end(); ++i) {
                        const GammaPoissonDocumentPosterior& posterior =
                            posteriors[i];
                        expectedTopicWeights.row(
                            static_cast<int32_t>(i)) =
                            posterior.exposure
                            * (posterior.shape.array()
                                / posterior.rate.array().max(1e-12))
                                .matrix().transpose();
                    }
                });
            expectedNormSq = rowQuadraticForms(
                expectedTopicWeights, residualState->profileGram);
        }

        tbb::parallel_for(tbb::blocked_range<size_t>(
                0, docs.size(), grainSize),
            [&](const tbb::blocked_range<size_t>& range) {
                ResidualLocalAgg& local = residualTls->local();
                VectorXd exposureTheta(K);
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const Document& doc = docs[i];
                    const GammaPoissonDocumentPosterior& posterior =
                        posteriors[i];
                    if (similarityDiagnostics) {
                        exposureTheta =
                            expectedTopicWeights.row(
                                static_cast<int32_t>(i)).transpose();
                    } else {
                        exposureTheta =
                            posterior.exposure
                            * (posterior.shape.array()
                                / posterior.rate.array().max(1e-12)).matrix();
                        double entropy = 0.0;
                        for (int32_t k = 0; k < K; ++k) {
                            const double probability =
                                docTopic(static_cast<int32_t>(i), k);
                            if (probability > 0.0) {
                                entropy -= probability
                                    * std::log(probability);
                            }
                        }
                        unitEntropy(static_cast<int32_t>(i)) = entropy;
                    }
                    double maxLog = -std::numeric_limits<double>::infinity();
                    for (int32_t k = 0; k < K; ++k) {
                        const double value = psi(posterior.shape(k))
                            - std::log(std::max(posterior.rate(k), 1e-12));
                        thetaKernel(static_cast<int32_t>(i), k) = value;
                        maxLog = std::max(maxLog, value);
                    }
                    for (int32_t k = 0; k < K; ++k) {
                        thetaKernel(static_cast<int32_t>(i), k) = std::exp(
                            thetaKernel(static_cast<int32_t>(i), k) - maxLog);
                    }
                    local.topicExposureTotals.noalias() +=
                        exposureTheta;

                    double residual =
                        exposureTheta.dot(residualState->topicCapacity);
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
                            exposureTheta.transpose()
                            * residualState->expectedBeta;
                    }
                    for (size_t j = 0; j < doc.ids.size(); ++j) {
                        const uint32_t w = doc.ids[j];
                        const double observed = doc.cnts[j];
                        const double expected = sparsePrediction
                            ? exposureTheta.dot(
                                residualState->expectedBeta.col(w))
                            : local.denseExpected(w);
                        expectedCells[documentOffsets[i] + j] = expected;
                        const double correction =
                            std::abs(expected - observed) - expected;
                        residual += correction;
                        if (similarityDiagnostics) {
                            dotProduct += expected * observed;
                            observedNormSq += observed * observed;
                        }
                    }
                    if (similarityDiagnostics
                            && expectedNormSq(static_cast<int32_t>(i)) > 0.0
                            && observedNormSq > 0.0) {
                        unitCosine(static_cast<int32_t>(i)) =
                            dotProduct
                            / std::sqrt(
                                expectedNormSq(static_cast<int32_t>(i))
                                * observedNormSq);
                    }
                    unitResidual(static_cast<int32_t>(i)) =
                        std::max(0.0, residual);
                }
            });

        std::vector<size_t> featureOffsets(
            static_cast<size_t>(M) + 1, 0);
        for (const Document& doc : docs) {
            for (uint32_t w : doc.ids) {
                if (w >= static_cast<uint32_t>(M)) {
                    error("%s: feature index %u is out of range", __func__, w);
                }
                ++featureOffsets[static_cast<size_t>(w) + 1];
            }
        }
        for (int32_t w = 0; w < M; ++w) {
            featureOffsets[static_cast<size_t>(w) + 1] +=
                featureOffsets[static_cast<size_t>(w)];
        }
        std::vector<size_t> featureCursor = featureOffsets;
        std::vector<CellRef> cellsByFeature(nCells);
        for (int32_t d = 0; d < nDocs; ++d) {
            const Document& doc = docs[static_cast<size_t>(d)];
            for (size_t j = 0; j < doc.ids.size(); ++j) {
                const uint32_t w = doc.ids[j];
                cellsByFeature[featureCursor[w]++] = {
                    static_cast<uint32_t>(d), static_cast<uint32_t>(j)};
            }
        }

        std::vector<PullRecord> batchPullRecords;
        if (residualState->pullSpool.enabled()) {
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
                    if (featureOffsets[w0] == featureOffsets[w0 + 1]) {
                        continue;
                    }
                    const int32_t w = static_cast<int32_t>(w0);
                    double correction = 0.0;
                    double total = 0.0;
                    double conditionalLogTerm = 0.0;
                    double deletionNumerator = 0.0;
                    int64_t units = 0;
                    assigned.setZero();

                    for (size_t p = featureOffsets[w0];
                            p < featureOffsets[w0 + 1]; ++p) {
                        const CellRef ref = cellsByFeature[p];
                        const size_t d = ref.document;
                        const size_t j = ref.offset;
                        const Document& doc = docs[d];
                        const GammaPoissonDocumentPosterior& posterior =
                            posteriors[d];
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

                        double epsilon = 1.0;
                        if (residualState->hasFeatureDispersion) {
                            const double tau =
                                residualState->featureDispersion(w);
                            epsilon = (tau + observed)
                                / std::max(tau + expected, 1e-12);
                        }
                        double deletedTotal = 0.0;
                        for (int32_t k = 0; k < K; ++k) {
                            const double deletedShape =
                                posterior.shape(k)
                                - observed * allocation(k);
                            const double deletedRate =
                                posterior.rate(k)
                                - posterior.exposure * epsilon
                                    * residualState->expectedBeta(k, w);
                            if (!std::isfinite(deletedShape)
                                    || deletedShape <= 0.0
                                    || !std::isfinite(deletedRate)
                                    || deletedRate <= 0.0) {
                                error("%s: invalid one-step deletion "
                                    "posterior for feature %d", __func__, w);
                            }
                            deletedTopic(k) =
                                deletedShape / deletedRate
                                * residualState->topicCapacity(k);
                            deletedTotal += deletedTopic(k);
                        }
                        if (!std::isfinite(deletedTotal)
                                || deletedTotal <= 0.0) {
                            error("%s: invalid one-step deletion topic "
                                "total for feature %d", __func__, w);
                        }
                        deletedTopic /= deletedTotal;
                        deletionNumerator += observed * 0.5
                            * (deletedTopic
                                - docTopic.row(static_cast<int32_t>(d))
                                    .transpose()).cwiseAbs().sum();

                        if (residualState->pullSpool.enabled()) {
                            const double totalVariation = 0.5
                                * (allocation
                                    - docTopic.row(static_cast<int32_t>(d))
                                        .transpose()).cwiseAbs().sum();
                            PullRecord& record =
                                batchPullRecords[documentOffsets[d] + j];
                            record.feature = static_cast<uint32_t>(w);
                            record.observed = observed;
                            record.mean = expected;
                            record.totalVariation = totalVariation;
                        }
                    }

                    residualState->featureCorrections(w) += correction;
                    residualState->featureTotals(w) += total;
                    residualState->conditionalLogTerms(w) +=
                        conditionalLogTerm;
                    residualState->deletionNumerators(w) +=
                        deletionNumerator;
                    residualState->featureUnits[w0] += units;
                    residualState->allocatedTopicCounts.col(w).noalias() +=
                        assigned;
                }
            });

        if (residualState->pullSpool.enabled()) {
            std::vector<PullRecord> positiveRecords;
            positiveRecords.reserve(nCells);
            for (const PullRecord& record : batchPullRecords) {
                if (record.mean > 0.0) {
                    positiveRecords.push_back(record);
                }
            }
            residualState->pullSpool.append(positiveRecords);
        }

        for (int32_t i = 0; i < nDocs; ++i) {
            if (!ids[i].empty()) {
                *unitStats << ids[i] << "\t";
            }
            *unitStats << rawTotalCount(docs[i])
                << "\t" << std::setprecision(2) << unitResidual(i)
                << "\t" << std::setprecision(4) << unitEntropy(i);
            if (similarityDiagnostics) {
                *unitStats
                    << "\t" << std::setprecision(4) << unitCosine(i)
                    << "\t" << similarityEntropy.sh_lcr(i)
                    << "\t" << similarityEntropy.sh_q(i);
            }
            *unitStats << "\n";
        }
    }

    GammaPoisson4Hex& gp;
    std::ostream& results;
    RowMajorMatrixXd& pseudobulk;
    MatrixXd& specialPseudobulk;
    Mode pseudobulkMode;
    std::ostream* unitStats;
    ResidualState* residualState;
    bool similarityDiagnostics;
    int32_t threadHint;
    std::unique_ptr<ResidualTls> residualTls;
    int32_t M;
    int32_t K;
};

bool readSpecialHexMinibatch(std::ifstream& input, HexReader& rawReader,
        GammaPoisson4Hex& gp, SpecialBatch& batch, Mode mode,
        const std::vector<int32_t>& inputToModel, int32_t modal,
        int32_t batchSize, int32_t maxUnits, int32_t minCount) {
    batch.clear();
    const int32_t target = std::min(batchSize, maxUnits);
    std::string line;
    while (static_cast<int32_t>(batch.size()) < target) {
        if (!std::getline(input, line)) {
            return false;
        }
        Document rawDoc;
        std::string id;
        if (rawReader.parseLine(rawDoc, id, line, modal, false) < 0) {
            error("%s: error parsing input line", __func__);
        }
        transform_pseudobulk::appendDocument(std::move(rawDoc), std::move(id),
            batch, mode, inputToModel, gp, minCount);
    }
    return true;
}

bool readSpecialDgeMinibatch(DGEReader10X& dge, GammaPoisson4Hex& gp,
        SpecialBatch& batch, Mode mode,
        const std::vector<int32_t>& inputToModel, int32_t batchSize,
        int32_t maxUnits, int32_t minCount) {
    batch.clear();
    const int32_t target = std::min(batchSize, maxUnits);
    while (static_cast<int32_t>(batch.size()) < target) {
        Document rawDoc;
        int32_t unitIndex = -1;
        if (!dge.next(rawDoc, &unitIndex, nullptr)) {
            return false;
        }
        if (unitIndex < 0) {
            continue;
        }
        transform_pseudobulk::appendDocument(std::move(rawDoc),
            dge.getUnitId(unitIndex), batch, mode, inputToModel, gp, minCount);
    }
    return true;
}

} // namespace

int32_t cmdGammaPoisTransform(int argc, char** argv) {
    std::string inFile, metaFile, stateFile, outPrefix, featureFile;
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
    bool sorted_by_barcode = false;
    bool keep_barcodes = false;
    bool randomize_output = false;
    bool pseudobulk_all_features = false;
    bool compute_residuals = false;
    bool cheap_feature_diagnostics = false;
    bool unit_similarity_diagnostics = false;
    bool full_model = false;
    bool use_stored_dispersion = false;
    int32_t dispersion_min_positive = 10;
    int32_t dispersion_mu_bins = 32;
    double dispersion_loess_span = 0.3;
    double dispersion_delta_min = 1e-8;
    double dispersion_delta_max = 1e4;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("in-state", "Input Gamma-Poisson state file", stateFile, true)
      .add_option("out-prefix", "Output prefix for results files", outPrefix, true)
      .add_option("minibatch-size", "Minibatch size", batchSize)
      .add_option("modal", "Modality to use (0-based)", modal)
      .add_option("threads", "Number of threads", nThreads)
      .add_option("seed", "Random seed", seed)
      .add_option("verbose", "Verbose level", verbose)
      .add_option("debug", "If >0, only process this many units", debug_)
      .add_option("randomize-output", "Randomize document output order", randomize_output)
      .add_option("use-stored-dispersion", "Use dispersion from the fitted state instead of estimating it in the transform data", use_stored_dispersion);

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
      .add_option("full-model", "Use every fitted model feature, treating missing input features as measured zeros", full_model)
      .add_option("pseudobulk-all-features", "Include all retained input features in pseudobulk output", pseudobulk_all_features)
      .add_option("feature-residuals", "Compute per-feature and per-unit residuals (backward compatibility)", compute_residuals)
      .add_option("residuals", "Compute per-feature and per-unit residuals", compute_residuals)
      .add_option("feature-diagnostics-cheap", "Skip spool-dependent gain-adjusted feature residual and Pull diagnostics", cheap_feature_diagnostics)
      .add_option("unit-diagnostics-similarity", "Add cosine and similarity-adjusted entropy unit diagnostics", unit_similarity_diagnostics);

    pl.add_option("dispersion-loess-span", "LOESS span for transform-data dispersion estimation", dispersion_loess_span)
      .add_option("dispersion-min-positive", "Minimum positive cells for a raw transform-data dispersion estimate", dispersion_min_positive)
      .add_option("dispersion-mu-bins", "Log-mean bins per feature during transform-data dispersion estimation", dispersion_mu_bins)
      .add_option("dispersion-delta-min", "Lower bound for estimated inverse dispersion", dispersion_delta_min)
      .add_option("dispersion-delta-max", "Upper bound for estimated inverse dispersion", dispersion_delta_max);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (batchSize <= 0) batchSize = 512;
    if (seed <= 0) seed = std::random_device{}();
    if (randomize_output && sorted_by_barcode) {
        error("--randomize-output and --sorted-by-barcode are mutually exclusive");
    }
    if (cheap_feature_diagnostics && !compute_residuals) {
        error("--feature-diagnostics-cheap requires --residuals");
    }
    if (unit_similarity_diagnostics && !compute_residuals) {
        error("--unit-diagnostics-similarity requires --residuals");
    }
    if (dispersion_min_positive < 1 || dispersion_mu_bins < 1
        || !std::isfinite(dispersion_loess_span)
        || dispersion_loess_span <= 0.0 || dispersion_loess_span > 1.0
        || !std::isfinite(dispersion_delta_min)
        || !std::isfinite(dispersion_delta_max)
        || dispersion_delta_min <= 0.0
        || dispersion_delta_max < dispersion_delta_min) {
        error("Invalid transform dispersion estimation options");
    }
    std::mt19937 output_random_engine(static_cast<uint32_t>(seed));
    const bool transform_weights_provided =
        !featureFile.empty() && icolWeight >= 0;
    if (full_model && transform_weights_provided) {
        error("--full-model cannot be combined with transform-time feature weights; "
            "the fitted state weights are used");
    }
    if (defaultWeight < 0.0) defaultWeight = -1.0;

    std::unique_ptr<GammaPoissonTopicModel> loadedModel =
        GammaPoissonTopicModel::load_state_deferred(
            stateFile, seed, nThreads, verbose);
    const std::vector<std::string>& stateFeatureNames =
        loadedModel->get_feature_names();
    const bool stateWeightsActive =
        loadedModel->feature_weights_active();
    const std::vector<double>& stateFeatureWeights =
        loadedModel->get_feature_weight();
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    const bool use_10x = initHexOrDgeInput(reader, dge_ptr, inFile, metaFile,
        dge_dirs, in_bc, in_ft, in_mtx, dataset_ids, keep_barcodes);
    if (full_model && !featureFile.empty()) {
        reader.readFeatureTotals(featureFile);
        if (minCountFeature != 1 || !include_ftr_regex.empty()
            || !exclude_ftr_regex.empty()) {
            warning("--full-model ignores transform feature filtering options");
        }
    } else if (!featureFile.empty()) {
        if (transform_weights_provided) {
            reader.setFeatureFilterAndWeights(featureFile, minCountFeature,
                include_ftr_regex, exclude_ftr_regex, icolWeight, defaultWeight,
                defaultWeight >= 0.0);
        } else {
            reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
        }
    }

    std::vector<int32_t> keptModelFeatures;
    std::vector<std::string> modelFeatures = stateFeatureNames;
    const bool orderedFullPanel =
        reader.features == stateFeatureNames;
    bool allModelFeaturesPresent = full_model || orderedFullPanel;
    if (!allModelFeaturesPresent) {
        std::unordered_set<std::string> inputFeatureSet(
            reader.features.begin(), reader.features.end());
        allModelFeaturesPresent = std::all_of(
            stateFeatureNames.begin(), stateFeatureNames.end(),
            [&](const std::string& feature) {
                return inputFeatureSet.find(feature) != inputFeatureSet.end();
            });
    }
    const bool orderedFullUnweightedFastPath =
        !transform_weights_provided
        && orderedFullPanel
        && !stateWeightsActive;
    if (!orderedFullUnweightedFastPath) {
        std::unordered_map<std::string, int32_t> stateFeatureIndex;
        stateFeatureIndex.reserve(stateFeatureNames.size());
        for (int32_t w = 0;
                w < static_cast<int32_t>(stateFeatureNames.size()); ++w) {
            const bool inserted =
                stateFeatureIndex.emplace(stateFeatureNames[w], w).second;
            if (!inserted) {
                error("Duplicate feature %s in Gamma-Poisson state",
                    stateFeatureNames[w].c_str());
            }
        }
        const std::vector<double> requestedWeights =
            reader.getFeatureWeights();
        std::vector<double> storedInputWeights(reader.features.size(), 1.0);
        std::vector<char> measuredStateFeature(stateFeatureNames.size(), 0);
        for (size_t i = 0; i < reader.features.size(); ++i) {
            const auto found = stateFeatureIndex.find(reader.features[i]);
            if (found == stateFeatureIndex.end()) continue;
            const int32_t w = found->second;
            measuredStateFeature[w] = 1;
            storedInputWeights[i] = stateWeightsActive
                ? stateFeatureWeights[w] : 1.0;
            if (transform_weights_provided) {
                const double scale = std::max(
                    {1.0, std::abs(requestedWeights[i]),
                        std::abs(storedInputWeights[i])});
                if (std::abs(requestedWeights[i] - storedInputWeights[i])
                        > 1e-12 * scale) {
                    error("Transform feature weight for %s (%.17g) conflicts with "
                        "the fitted state weight (%.17g)",
                        reader.features[i].c_str(), requestedWeights[i],
                        storedInputWeights[i]);
                }
            }
        }
        if (!transform_weights_provided) {
            reader.setFeatureWeights(storedInputWeights);
        }
        keptModelFeatures.reserve(stateFeatureNames.size());
        modelFeatures.clear();
        modelFeatures.reserve(stateFeatureNames.size());
        for (int32_t w = 0;
                w < static_cast<int32_t>(stateFeatureNames.size()); ++w) {
            if (allModelFeaturesPresent || measuredStateFeature[w]) {
                keptModelFeatures.push_back(w);
                modelFeatures.push_back(stateFeatureNames[w]);
            }
        }
        if (keptModelFeatures.empty()) {
            error("No overlapping measured features found between input and "
                "Gamma-Poisson state");
        }
    }

    const bool weights_active = reader.hasFeatureWeights();
    const Mode pseudobulkMode =
        transform_pseudobulk::selectMode(pseudobulk_all_features, weights_active);
    std::unique_ptr<HexReader> rawReader;
    std::vector<std::string> retainedInputFeatures;
    if (pseudobulkMode != Mode::Standard) {
        rawReader = std::make_unique<HexReader>(reader);
        rawReader->clearFeatureWeights();
        retainedInputFeatures = rawReader->features;
    }
    reader.setFeatureIndexRemap(modelFeatures, false);

    std::string info_header;
    if (!use_10x) {
        reader.getInfoHeaderStr(info_header);
    }

    GammaPoisson4Hex gp(reader, modal, verbose);
    gp.initialize_transform(std::move(loadedModel), maxIter, mDelta,
        keptModelFeatures);

    const int32_t M = gp.nFeatures();
    const int32_t K = gp.getNumTopics();
    std::vector<std::string> pseudobulkFeatureNames = gp.getFeatureNames();
    std::vector<int32_t> inputToModel;
    if (pseudobulkMode == Mode::WeightedModel) {
        rawReader->setFeatureIndexRemap(pseudobulkFeatureNames, false);
    } else if (pseudobulkMode == Mode::AllFeatures) {
        inputToModel = transform_pseudobulk::mapInputFeaturesToModel(
            retainedInputFeatures, pseudobulkFeatureNames);
        pseudobulkFeatureNames = std::move(retainedInputFeatures);
    }

    RowMajorMatrixXd pseudobulk;
    MatrixXd specialPseudobulk;
    if (pseudobulkMode == Mode::Standard) {
        pseudobulk = RowMajorMatrixXd::Zero(M, K);
    } else {
        specialPseudobulk = MatrixXd::Zero(
            static_cast<int32_t>(pseudobulkFeatureNames.size()), K);
    }

    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    const int32_t minCountInt =
        minCount > 0 ? static_cast<int32_t>(std::ceil(minCount)) : 0;
    if (!use_stored_dispersion) {
        GammaPoissonDispersionOptions options;
        options.min_positive = dispersion_min_positive;
        options.mu_bins = dispersion_mu_bins;
        options.loess_span = dispersion_loess_span;
        options.delta_min = dispersion_delta_min;
        options.delta_max = dispersion_delta_max;
        gp.clearFeatureDispersion();
        GammaPoissonDispersionResult estimated;
        if (use_10x) {
            DGEReader10X& dge = *dge_ptr;
            const int32_t overlap =
                dge.setFeatureIndexRemap(modelFeatures, false);
            if (overlap == 0) {
                error("No measured model features overlap with 10X input");
            }
            estimated = gp.estimateFeatureDispersion10X(options, dge,
                batchSize, minCountInt, maxUnits);
        } else {
            estimated = gp.estimateFeatureDispersion(options, inFile,
                batchSize, minCountInt, maxUnits);
        }
        const std::string dispersionPath = outPrefix + ".dispersion.tsv";
        write_gamma_poisson_dispersion_diagnostics(
            dispersionPath, gp.getFeatureNames(), estimated);
        notice("Estimated transform-data dispersion from %d documents; "
            "diagnostics written to %s",
            estimated.n_documents, dispersionPath.c_str());
    } else {
        notice(gp.hasFeatureDispersion()
            ? "Using per-feature dispersion stored in the fitted state"
            : "Using the stored Poisson model (state has no feature dispersion)");
    }

    const std::string resultsPath = outPrefix + ".results.tsv";
    std::ofstream results(resultsPath);
    if (!results) {
        error("Error opening output file: %s for writing", resultsPath.c_str());
    }
    writeUnitIdHeader(results, use_10x, info_header);
    gp.writeUnitHeader(results);
    results << std::scientific << std::setprecision(4);

    std::unique_ptr<ResidualState> residualState;
    std::unique_ptr<std::ofstream> unitStats;
    if (compute_residuals) {
        residualState =
            std::make_unique<ResidualState>(
                gp, cheap_feature_diagnostics,
                unit_similarity_diagnostics);
        const std::string unitStatsPath =
            outPrefix + ".unit_stats.tsv";
        unitStats = std::make_unique<std::ofstream>(unitStatsPath);
        if (!*unitStats) {
            error("Error opening output file: %s for writing",
                unitStatsPath.c_str());
        }
        writeUnitIdHeader(*unitStats, use_10x, info_header);
        *unitStats << "total_count\tresidual\tentropy";
        if (unit_similarity_diagnostics) {
            *unitStats << "\tcosine_sim\tsh_lcr\tsh_q";
        }
        *unitStats << "\n" << std::fixed;
    }

    GammaPoisTransformBatchProcessor processor(gp, results, pseudobulk,
        specialPseudobulk, pseudobulkMode,
        unitStats.get(), residualState.get(),
        unit_similarity_diagnostics, nThreads);
    bool fileopen = true;
    int32_t processed = 0;
    TransformBatch batch;

    if (pseudobulkMode == Mode::Standard) {
    if (use_10x) {
        DGEReader10X& dge = *dge_ptr;
        int32_t n_overlap = dge.setFeatureIndexRemap(modelFeatures, false);
        if (n_overlap == 0) {
            error("No overlapping features found between 10X input and model state");
        }
        std::vector<int32_t> barcode_idx;
        if (sorted_by_barcode) {
            while (fileopen && processed < maxUnits) {
                batch.clear();
                const int32_t remaining = maxUnits - processed;
                fileopen = dge.readMinibatch(batch.docs, barcode_idx,
                    batchSize, remaining, minCountInt);
                if (batch.empty()) break;
                applyWeights(batch.docs, gp);
                assignBarcodeIds(dge, barcode_idx, batch.ids);
                processor.process(batch);
                processed += static_cast<int32_t>(batch.size());
            }
        } else {
            std::vector<Document> all_docs;
            std::vector<int32_t> all_barcode_idx;
            dge.readAll(all_docs, all_barcode_idx, minCountInt);
            applyWeights(all_docs, gp);
            std::vector<std::string> all_ids;
            assignBarcodeIds(dge, all_barcode_idx, all_ids);
            if (randomize_output) {
                randomizeDocuments(all_docs, all_ids, output_random_engine);
            }
            size_t cursor = 0;
            while (cursor < all_docs.size() && processed < maxUnits) {
                batch.clear();
                const int32_t remaining = maxUnits - processed;
                size_t take = std::min(static_cast<size_t>(batchSize),
                    all_docs.size() - cursor);
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
                if (batch.empty()) break;
                processor.process(batch);
                processed += static_cast<int32_t>(batch.size());
            }
        }
    } else {
        std::ifstream inFileStream(inFile);
        if (!inFileStream) {
            error("Error opening input file: %s", inFile.c_str());
        }
        if (randomize_output) {
            std::vector<Document> all_docs;
            std::vector<std::string> all_ids;
            while (fileopen && static_cast<int32_t>(all_docs.size()) < maxUnits) {
                batch.clear();
                const int32_t remaining = maxUnits - static_cast<int32_t>(all_docs.size());
                fileopen = gp.readMinibatch(inFileStream, batch.docs, batch.ids,
                    batchSize, minCountInt, remaining);
                all_docs.insert(all_docs.end(),
                    std::make_move_iterator(batch.docs.begin()),
                    std::make_move_iterator(batch.docs.end()));
                all_ids.insert(all_ids.end(),
                    std::make_move_iterator(batch.ids.begin()),
                    std::make_move_iterator(batch.ids.end()));
                if (batch.empty()) break;
            }
            randomizeDocuments(all_docs, all_ids, output_random_engine);
            size_t cursor = 0;
            while (cursor < all_docs.size()) {
                batch.clear();
                const size_t take = std::min(static_cast<size_t>(batchSize),
                    all_docs.size() - cursor);
                batch.docs.insert(batch.docs.end(),
                    std::make_move_iterator(all_docs.begin() + cursor),
                    std::make_move_iterator(all_docs.begin() + cursor + take));
                batch.ids.insert(batch.ids.end(),
                    std::make_move_iterator(all_ids.begin() + cursor),
                    std::make_move_iterator(all_ids.begin() + cursor + take));
                cursor += take;
                processor.process(batch);
                processed += static_cast<int32_t>(batch.size());
            }
        } else while (fileopen && processed < maxUnits) {
            batch.clear();
            const int32_t remaining = maxUnits - processed;
            fileopen = gp.readMinibatch(inFileStream, batch.docs, batch.ids,
                batchSize, minCountInt, remaining);
            if (batch.empty()) break;
            processor.process(batch);
            processed += static_cast<int32_t>(batch.size());
        }
    }
    } else {
        SpecialBatch specialBatch;
        if (use_10x) {
            DGEReader10X& dge = *dge_ptr;
            const std::vector<std::string>& readFeatures = pseudobulkFeatureNames;
            const int32_t n_overlap =
                dge.setFeatureIndexRemap(readFeatures, false);
            if (n_overlap == 0) {
                error("No retained input features overlap with 10X input");
            }

            if (sorted_by_barcode) {
                while (fileopen && processed < maxUnits) {
                    const int32_t remaining = maxUnits - processed;
                    fileopen = readSpecialDgeMinibatch(dge, gp, specialBatch,
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
                SpecialBatch allSpecial;
                for (size_t i = 0; i < allRawDocs.size(); ++i) {
                    transform_pseudobulk::appendDocument(
                        std::move(allRawDocs[i]),
                        dge.getUnitId(allUnitIndices[i]), allSpecial,
                        pseudobulkMode, inputToModel, gp, minCountInt);
                }
                if (randomize_output) {
                    transform_pseudobulk::randomize(
                        allSpecial, output_random_engine);
                }
                size_t cursor = 0;
                while (cursor < allSpecial.size() && processed < maxUnits) {
                    size_t take = std::min(static_cast<size_t>(batchSize),
                        allSpecial.size() - cursor);
                    take = std::min(take,
                        static_cast<size_t>(maxUnits - processed));
                    transform_pseudobulk::moveRange(
                        allSpecial, cursor, take, specialBatch);
                    cursor += take;
                    processor.process(specialBatch);
                    processed += static_cast<int32_t>(specialBatch.size());
                }
            }
        } else {
            std::ifstream inFileStream(inFile);
            if (!inFileStream) {
                error("Error opening input file: %s", inFile.c_str());
            }
            if (randomize_output) {
                SpecialBatch allSpecial;
                while (fileopen &&
                        static_cast<int32_t>(allSpecial.size()) < maxUnits) {
                    const int32_t remaining =
                        maxUnits - static_cast<int32_t>(allSpecial.size());
                    fileopen = readSpecialHexMinibatch(inFileStream, *rawReader,
                        gp, specialBatch, pseudobulkMode, inputToModel, modal,
                        batchSize, remaining, minCountInt);
                    const size_t take = specialBatch.size();
                    transform_pseudobulk::appendMoved(
                        allSpecial, specialBatch);
                    if (take == 0) {
                        break;
                    }
                }
                transform_pseudobulk::randomize(
                    allSpecial, output_random_engine);
                size_t cursor = 0;
                while (cursor < allSpecial.size()) {
                    const size_t take = std::min(
                        static_cast<size_t>(batchSize),
                        allSpecial.size() - cursor);
                    transform_pseudobulk::moveRange(
                        allSpecial, cursor, take, specialBatch);
                    cursor += take;
                    processor.process(specialBatch);
                    processed += static_cast<int32_t>(specialBatch.size());
                }
            } else {
                while (fileopen && processed < maxUnits) {
                    const int32_t remaining = maxUnits - processed;
                    fileopen = readSpecialHexMinibatch(inFileStream, *rawReader,
                        gp, specialBatch, pseudobulkMode, inputToModel, modal,
                        batchSize, remaining, minCountInt);
                    if (specialBatch.empty()) {
                        break;
                    }
                    processor.process(specialBatch);
                    processed += static_cast<int32_t>(specialBatch.size());
                }
            }
        }
    }
    processor.finalizeResiduals();
    results.close();
    if (unitStats) {
        unitStats->close();
        notice("Per-unit residuals written to %s",
            (outPrefix + ".unit_stats.tsv").c_str());
    }
    notice("Transformation results written to %s", resultsPath.c_str());

    const std::string pseudobulkPath = outPrefix + ".pseudobulk.tsv";
    std::ofstream pseudobulkOut(pseudobulkPath);
    if (!pseudobulkOut) {
        error("Error opening output file: %s for writing", pseudobulkPath.c_str());
    }
    pseudobulkOut << "Feature\t";
    gp.writeModelHeader(pseudobulkOut);
    pseudobulkOut << std::fixed << std::setprecision(3);
    for (int32_t w = 0;
            w < static_cast<int32_t>(pseudobulkFeatureNames.size()); ++w) {
        pseudobulkOut << pseudobulkFeatureNames[w];
        for (int32_t k = 0; k < K; ++k) {
            const double value = pseudobulkMode == Mode::Standard
                ? pseudobulk(w, k)
                : specialPseudobulk(w, k);
            pseudobulkOut << "\t" << value;
        }
        pseudobulkOut << "\n";
    }
    pseudobulkOut.close();
    notice("Pseudobulk counts written to %s", pseudobulkPath.c_str());

    if (residualState) {
        const std::string featureResidualPath =
            outPrefix + ".feature_residuals.tsv";
        std::ofstream featureResidualOut(featureResidualPath);
        if (!featureResidualOut) {
            error("Error opening output file: %s for writing",
                featureResidualPath.c_str());
        }
        featureResidualOut
            << "Feature\tabsDiff\tabsDiffRate"
            << "\ttotCount\tnUnits\tlog2Gain"
            << "\tmarginalDev\tconditionalDev"
            << "\tfactorDrift\tdeletionTV"
            << (cheap_feature_diagnostics
                ? "\n"
                : "\tadjAbsDiffRate\tpull\n");
        auto writeDiagnostic = [&](double value) {
            if (std::isnan(value)) {
                featureResidualOut << "NA";
            } else if (std::isinf(value)) {
                featureResidualOut << (value < 0.0 ? "-inf" : "inf");
            } else {
                featureResidualOut << std::scientific
                    << std::setprecision(4) << value;
            }
        };
        for (int32_t w = 0; w < M; ++w) {
            const double total = residualState->featureTotals(w);
            const double difference = std::max(
                0.0, residualState->featureCorrections(w));
            const double ratio =
                total > 0.0 ? difference / total : 0.0;
            featureResidualOut << modelFeatures[w]
                << "\t" << std::fixed << std::setprecision(3)
                << difference
                << "\t" << std::setprecision(6) << ratio
                << "\t" << std::llround(total)
                << "\t" << residualState->featureUnits[
                    static_cast<size_t>(w)] << "\t";
            writeDiagnostic(residualState->log2Gain(w));
            featureResidualOut << "\t";
            writeDiagnostic(residualState->marginalDeviance(w));
            featureResidualOut << "\t";
            writeDiagnostic(residualState->conditionalDeviance(w));
            featureResidualOut << "\t";
            writeDiagnostic(residualState->topicDeviance(w));
            featureResidualOut << "\t";
            writeDiagnostic(residualState->deletionTV(w));
            if (!cheap_feature_diagnostics) {
                featureResidualOut << "\t";
                writeDiagnostic(
                    residualState->adjustedResidualPerCount(w));
                featureResidualOut << "\t";
                writeDiagnostic(residualState->pull(w));
            }
            featureResidualOut << "\n";
        }
        featureResidualOut.close();
        notice("Per-feature residuals written to %s",
            featureResidualPath.c_str());
    }
    return 0;
}
