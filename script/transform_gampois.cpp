#include "gamma_pois_topic.hpp"
#include "gamma_pois_posterior_io.hpp"
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
    const RowMajorMatrixXd& topicProfiles;
    const MatrixXd& expectedBeta;
    const MatrixXd& betaAllocationKernel;
    const VectorXd& featureDispersion;
    const VectorXd& topicCapacity;
    bool hasDispersion;
    MatrixXd profileGram;
    MatrixXd topicSimilarity;
    MatrixXd allocatedTopicCounts;
    VectorXd featureCorrections;
    VectorXd featureTotals;
    VectorXd topicExposureTotals;
    VectorXd conditionalLogTerms;
    VectorXd leverageNumerators;
    VectorXd cookCorrections;
    VectorXd cookBaseline;
    std::vector<int64_t> featureUnits;
    PullSpool pullSpool;

    VectorXd predictedTotals;
    VectorXd log2Gain;
    VectorXd marginalDeviance;
    VectorXd conditionalDeviance;
    VectorXd topicDeviance;
    VectorXd topicLeverage;
    VectorXd adjustedResidualPerCount;
    VectorXd pull;
    VectorXd cookScore;

    ResidualState(GammaPoisson4Hex& gp, bool cheapDiagnostics)
        : topicProfiles(gp.get_model_matrix()),
          expectedBeta(gp.getExpectedBeta()),
          betaAllocationKernel(gp.getBetaAllocationKernel()),
          featureDispersion(gp.getFeatureDispersion()),
          topicCapacity(gp.getTopicCapacity()),
          hasDispersion(gp.hasFeatureDispersion()),
          profileGram(topicProfiles * topicProfiles.transpose()),
          topicSimilarity(MatrixXd::Zero(
              topicProfiles.rows(), topicProfiles.rows())),
          allocatedTopicCounts(MatrixXd::Zero(
              topicProfiles.rows(), topicProfiles.cols())),
          featureCorrections(VectorXd::Zero(topicProfiles.cols())),
          featureTotals(VectorXd::Zero(topicProfiles.cols())),
          topicExposureTotals(VectorXd::Zero(topicProfiles.rows())),
          conditionalLogTerms(VectorXd::Zero(topicProfiles.cols())),
          leverageNumerators(VectorXd::Zero(topicProfiles.cols())),
          cookCorrections(VectorXd::Zero(topicProfiles.cols())),
          cookBaseline(VectorXd::Zero(topicProfiles.rows())),
          featureUnits(static_cast<size_t>(topicProfiles.cols()), 0),
          pullSpool(!cheapDiagnostics) {
        VectorXd norms = profileGram.diagonal().array().max(0.0).sqrt();
        for (int32_t k = 0; k < topicProfiles.rows(); ++k) {
            topicSimilarity(k, k) = 1.0;
            for (int32_t l = k + 1; l < topicProfiles.rows(); ++l) {
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
        std::ostream* posterior_, GammaPoissonDispersionWriter* dispersion_,
        uint64_t state_checksum_, std::ostream* unitStats_,
        ResidualState* residualState_, int32_t nThreads_)
        : gp(gp_), results(results_), pseudobulk(pseudobulk_),
          specialPseudobulk(specialPseudobulk_), pseudobulkMode(pseudobulkMode_),
          posterior(posterior_), dispersion(dispersion_),
          state_checksum(state_checksum_), unitStats(unitStats_),
          residualState(residualState_), threadHint(std::max(1, nThreads_)),
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
        gp.transformWithPosteriors(DocumentView(batch.docs), doc_topic, posteriors);
        writeTopicRows(batch.ids, doc_topic);
        writePosteriorRows(batch.docs, batch.ids, posteriors);
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
        gp.transformWithPosteriors(
            DocumentView(batch.modelDocs), doc_topic, posteriors);
        writeTopicRows(batch.ids, doc_topic);
        writePosteriorRows(batch.modelDocs, batch.ids, posteriors);
        processResiduals(batch.modelDocs, batch.ids, doc_topic, posteriors);
        transform_pseudobulk::accumulate(
            specialPseudobulk, batch, doc_topic, pseudobulkMode);
    }

    void finalizeResiduals() {
        if (!residualState) return;
        for (auto& local : *residualTls) {
            residualState->topicExposureTotals += local.topicExposureTotals;
            residualState->cookBaseline += local.cookBaseline;
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
        residualState->topicLeverage = VectorXd::Constant(M, nan);
        residualState->adjustedResidualPerCount =
            VectorXd::Constant(M, nan);
        residualState->pull = VectorXd::Constant(M, nan);
        residualState->cookScore =
            residualState->expectedBeta.array().square().matrix().transpose()
            * residualState->cookBaseline;
        residualState->cookScore += residualState->cookCorrections;
        residualState->cookScore =
            0.5 * residualState->cookScore.cwiseMax(0.0);

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
            residualState->topicLeverage(w) =
                residualState->leverageNumerators(w) / observed;
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
        VectorXd cookBaseline;

        explicit ResidualLocalAgg(int32_t nTopics)
            : topicExposureTotals(VectorXd::Zero(nTopics)),
              cookBaseline(VectorXd::Zero(nTopics)) {}
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

    void writePosteriorRows(const std::vector<Document>& docs,
        const std::vector<std::string>& ids,
        const std::vector<GammaPoissonDocumentPosterior>& posteriors) {
        if (!posterior) return;
        for (size_t i = 0; i < posteriors.size(); ++i) {
            if (!ids[i].empty()) {
                *posterior << ids[i] << "\t";
            }
            const GammaPoissonDocumentPosterior& local = posteriors[i];
            *posterior << row_index << "\t" << local.exposure;
            for (int32_t k = 0; k < K; ++k) {
                *posterior << "\t" << local.shape(k);
            }
            for (int32_t k = 0; k < K; ++k) {
                *posterior << "\t" << local.rate(k);
            }
            *posterior << "\n";

            if (dispersion) {
                GammaPoissonDispersionApproximation approximation;
                gp.dispersionCovarianceApproximation(docs[i], local,
                    dispersion->rank(), state_checksum ^ row_index, approximation);
                dispersion->append(approximation);
            }
            ++row_index;
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
        RowMajorMatrixXd thetaMean(nDocs, K);
        RowMajorMatrixXd thetaKernel(nDocs, K);
        VectorXd unitResidual = VectorXd::Zero(nDocs);
        VectorXd unitCosine = VectorXd::Zero(nDocs);
        const ThetaEntropyStats entropyStats =
            computeThetaEntropyStats(
                docTopic, residualState->topicSimilarity);
        const size_t grainSize = std::max<size_t>(
            1, docs.size() / (2 * static_cast<size_t>(threadHint)));

        tbb::parallel_for(tbb::blocked_range<size_t>(
                0, docs.size(), grainSize),
            [&](const tbb::blocked_range<size_t>& range) {
                ResidualLocalAgg& local = residualTls->local();
                VectorXd intensity(K);
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const Document& doc = docs[i];
                    const GammaPoissonDocumentPosterior& posterior =
                        posteriors[i];
                    thetaMean.row(static_cast<int32_t>(i)) =
                        (posterior.shape.array()
                        / posterior.rate.array().max(1e-12)).matrix().transpose();
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
                    intensity = thetaMean.row(
                        static_cast<int32_t>(i)).transpose();
                    local.topicExposureTotals.noalias() +=
                        posterior.exposure * intensity;
                    for (int32_t k = 0; k < K; ++k) {
                        local.cookBaseline(k) +=
                            posterior.exposure * posterior.exposure
                            * intensity(k) * intensity(k)
                            / std::max(posterior.shape(k), 1e-12);
                    }
                    intensity.array() *=
                        residualState->topicCapacity.array();
                    intensity *= posterior.exposure;

                    double residual = intensity.sum();
                    double dotProduct = 0.0;
                    double observedNormSq = 0.0;
                    const double expectedNormSq = std::max(
                        0.0, intensity.dot(
                            residualState->profileGram * intensity));
                    for (size_t j = 0; j < doc.ids.size(); ++j) {
                        const uint32_t w = doc.ids[j];
                        const double observed = doc.cnts[j];
                        const double expected = intensity.dot(
                            residualState->topicProfiles.col(w));
                        expectedCells[documentOffsets[i] + j] = expected;
                        const double correction =
                            std::abs(expected - observed) - expected;
                        residual += correction;
                        dotProduct += expected * observed;
                        observedNormSq += observed * observed;
                    }
                    if (expectedNormSq > 0.0 && observedNormSq > 0.0) {
                        unitCosine(static_cast<int32_t>(i)) =
                            dotProduct
                            / std::sqrt(expectedNormSq * observedNormSq);
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
                VectorXd assigned = VectorXd::Zero(K);
                for (size_t w0 = range.begin(); w0 < range.end(); ++w0) {
                    const int32_t w = static_cast<int32_t>(w0);
                    double correction = 0.0;
                    double total = 0.0;
                    double conditionalLogTerm = 0.0;
                    double leverage = 0.0;
                    double cookCorrection = 0.0;
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
                        const double totalVariation = 0.5
                            * (allocation
                                - docTopic.row(static_cast<int32_t>(d))
                                    .transpose()).cwiseAbs().sum();
                        leverage += observed * totalVariation;

                        double epsilon = 1.0;
                        if (residualState->hasDispersion) {
                            const double tau =
                                residualState->featureDispersion(w);
                            epsilon = (tau + observed)
                                / std::max(tau + expected, 1e-12);
                        }
                        for (int32_t k = 0; k < K; ++k) {
                            const double theta =
                                thetaMean(static_cast<int32_t>(d), k);
                            const double baseline = posterior.exposure
                                * theta * residualState->expectedBeta(k, w);
                            const double actual = observed * allocation(k)
                                - epsilon * baseline;
                            cookCorrection +=
                                (actual * actual - baseline * baseline)
                                / std::max(posterior.shape(k), 1e-12);
                        }

                        if (residualState->pullSpool.enabled()) {
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
                    residualState->leverageNumerators(w) += leverage;
                    residualState->cookCorrections(w) += cookCorrection;
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
                << "\t" << std::setprecision(4) << unitCosine(i)
                << "\t" << std::setprecision(4)
                << entropyStats.entropy(i)
                << "\t" << entropyStats.sh_lcr(i)
                << "\t" << entropyStats.sh_q(i) << "\n";
        }
    }

    GammaPoisson4Hex& gp;
    std::ostream& results;
    RowMajorMatrixXd& pseudobulk;
    MatrixXd& specialPseudobulk;
    Mode pseudobulkMode;
    std::ostream* posterior;
    GammaPoissonDispersionWriter* dispersion;
    uint64_t state_checksum;
    std::ostream* unitStats;
    ResidualState* residualState;
    int32_t threadHint;
    std::unique_ptr<ResidualTls> residualTls;
    uint64_t row_index = 0;
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
    bool skip_posterior = false;
    bool randomize_output = false;
    bool pseudobulk_all_features = false;
    bool compute_residuals = false;
    bool cheap_feature_diagnostics = false;
    int32_t posterior_dispersion_rank = 0;

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
      .add_option("skip-posterior", "Skip local Gamma posterior output", skip_posterior)
      .add_option("randomize-output", "Randomize document output order for downstream streaming clustering", randomize_output)
      .add_option("posterior-dispersion-rank", "Rank of optional dispersion covariance sidecar; 0 keeps only the diagonal, <0 disables it", posterior_dispersion_rank);

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
      .add_option("pseudobulk-all-features", "Include all retained input features in pseudobulk output", pseudobulk_all_features)
      .add_option("feature-residuals", "Compute per-feature and per-unit residuals (backward compatibility)", compute_residuals)
      .add_option("residuals", "Compute per-feature and per-unit residuals", compute_residuals)
      .add_option("feature-diagnostics-cheap", "Skip spool-dependent gain-adjusted feature residual and Pull diagnostics", cheap_feature_diagnostics);

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
    std::mt19937 output_random_engine(static_cast<uint32_t>(seed));
    const bool weights_active = !featureFile.empty() && icolWeight >= 0;
    if (defaultWeight < 0.0) defaultWeight = -1.0;

    const std::vector<std::string> modelFeatures =
        GammaPoissonTopicModel::read_state_feature_names(stateFile);
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    const bool use_10x = initHexOrDgeInput(reader, dge_ptr, inFile, metaFile,
        dge_dirs, in_bc, in_ft, in_mtx, dataset_ids, keep_barcodes);
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
    std::vector<std::string> modelFeaturesMutable = modelFeatures;
    reader.setFeatureIndexRemap(modelFeaturesMutable, false);

    std::string info_header;
    if (!use_10x) {
        reader.getInfoHeaderStr(info_header);
    }

    GammaPoisson4Hex gp(reader, modal, verbose);
    gp.initialize_transform(stateFile, seed, nThreads, verbose, maxIter, mDelta);

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
                gp, cheap_feature_diagnostics);
        const std::string unitStatsPath =
            outPrefix + ".unit_stats.tsv";
        unitStats = std::make_unique<std::ofstream>(unitStatsPath);
        if (!*unitStats) {
            error("Error opening output file: %s for writing",
                unitStatsPath.c_str());
        }
        writeUnitIdHeader(*unitStats, use_10x, info_header);
        *unitStats
            << "total_count\tresidual\tcosine_sim\tentropy\tsh_lcr\tsh_q\n"
            << std::fixed;
    }

    const uint64_t stateChecksum = gamma_poisson_state_checksum(stateFile);
    std::unique_ptr<std::ofstream> posterior;
    std::unique_ptr<GammaPoissonDispersionWriter> dispersion;
    if (!skip_posterior) {
        const GammaPoissonArtifactId posteriorId =
            generate_gamma_poisson_artifact_id();
        GammaPoissonArtifactId sidecarId;
        if (gp.hasFeatureDispersion() && posterior_dispersion_rank >= 0) {
            const int32_t actualRank = std::min(posterior_dispersion_rank, K);
            sidecarId = generate_gamma_poisson_artifact_id();
            dispersion = std::make_unique<GammaPoissonDispersionWriter>(
                outPrefix + ".posterior-dispersion.bin", K, actualRank,
                stateChecksum, sidecarId);
        }
        const std::string posteriorPath = outPrefix + ".posterior.tsv";
        posterior = std::make_unique<std::ofstream>(posteriorPath);
        if (!*posterior) {
            error("Error opening output file: %s for writing", posteriorPath.c_str());
        }
        *posterior << "##punkst_gamma_pois_posterior_v2\n";
        *posterior << "##n_topics\t" << K << "\n";
        *posterior << "##state_checksum\t" << stateChecksum << "\n";
        *posterior << "##artifact_id\t"
            << gamma_poisson_artifact_id_string(posteriorId) << "\n";
        *posterior << "##row_order\t" << (randomize_output ? "randomized" : "input") << "\n";
        if (!sidecarId.empty()) {
            *posterior << "##dispersion_sidecar_id\t"
                << gamma_poisson_artifact_id_string(sidecarId) << "\n";
        }
        writeUnitIdHeader(*posterior, use_10x, info_header);
        *posterior << "gp_row\tgp_exposure";
        const auto& topicNames = gp.get_topic_names();
        for (int32_t k = 0; k < K; ++k) {
            *posterior << "\tgp_shape_" << topicNames[k];
        }
        for (int32_t k = 0; k < K; ++k) {
            *posterior << "\tgp_rate_" << topicNames[k];
        }
        *posterior << "\n" << std::scientific << std::setprecision(4);
        notice("Gamma-Poisson local posterior will be written to %s", posteriorPath.c_str());
    }

    GammaPoisTransformBatchProcessor processor(gp, results, pseudobulk,
        specialPseudobulk, pseudobulkMode,
        posterior.get(), dispersion.get(), stateChecksum,
        unitStats.get(), residualState.get(), nThreads);
    bool fileopen = true;
    int32_t processed = 0;
    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    const int32_t minCountInt = minCount > 0 ? static_cast<int32_t>(std::ceil(minCount)) : 0;
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
    if (posterior) posterior->close();
    if (dispersion) dispersion->close();
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
            << "Feature\tAbsDiff\tAbsDiffPerCount"
            << "\tcount\tn_units\tlog2_gain"
            << "\tmarginal_deviance\tconditional_deviance"
            << "\ttopic_deviance\ttopic_leverage"
            << (cheap_feature_diagnostics
                ? "\tcook_score\n"
                : "\tgain_adjusted_absdiff_per_count\tpull\tcook_score\n");
        auto writeDiagnostic = [&](double value) {
            if (std::isnan(value)) {
                featureResidualOut << "NA";
            } else if (std::isinf(value)) {
                featureResidualOut << (value < 0.0 ? "-inf" : "inf");
            } else {
                featureResidualOut << std::scientific
                    << std::setprecision(8) << value;
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
                << "\t" << std::setprecision(6) << total
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
            writeDiagnostic(residualState->topicLeverage(w));
            if (!cheap_feature_diagnostics) {
                featureResidualOut << "\t";
                writeDiagnostic(
                    residualState->adjustedResidualPerCount(w));
                featureResidualOut << "\t";
                writeDiagnostic(residualState->pull(w));
            }
            featureResidualOut << "\t";
            writeDiagnostic(residualState->cookScore(w));
            featureResidualOut << "\n";
        }
        featureResidualOut.close();
        notice("Per-feature residuals written to %s",
            featureResidualPath.c_str());
    }
    return 0;
}
