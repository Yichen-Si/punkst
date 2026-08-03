#pragma once

#include "dataunits.hpp"
#include "error.hpp"
#include "numerical_utils.hpp"
#include "utils_sys.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <ostream>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace transform_helpers {

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

inline void writeUnitIdHeader(
        std::ostream& out, bool use10x, const std::string& infoHeader) {
    if (use10x) {
        out << "#barcode\t";
        return;
    }
    out << "#";
    if (!infoHeader.empty()) {
        out << infoHeader << "\t";
    }
}

inline int64_t rawTotalCount(const Document& doc) {
    const double rawTotal = doc.raw_ct_tot >= 0.0
        ? doc.raw_ct_tot
        : std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
    return static_cast<int64_t>(std::llround(rawTotal));
}

inline void assignBarcodeIds(const DGEReader10X& dge,
        const std::vector<int32_t>& barcodeIndices,
        std::vector<std::string>& ids) {
    ids.clear();
    ids.reserve(barcodeIndices.size());
    for (int32_t index : barcodeIndices) {
        if (index >= 0
                && index < static_cast<int32_t>(dge.barcodes.size())) {
            ids.push_back(dge.barcodes[index]);
        } else {
            ids.push_back(std::to_string(index));
        }
    }
}

template <typename Model>
inline void applyWeights(std::vector<Document>& docs, Model& model) {
    for (Document& doc : docs) {
        model.applyWeights(doc);
    }
}

inline void writeTopicRows(std::ostream& out,
        const std::vector<std::string>& ids,
        const RowMajorMatrixXd& documentTopics) {
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!ids[i].empty()) {
            out << ids[i] << "\t";
        }
        out << documentTopics(static_cast<int32_t>(i), 0);
        for (int32_t k = 1; k < documentTopics.cols(); ++k) {
            out << "\t" << documentTopics(static_cast<int32_t>(i), k);
        }
        out << "\n";
    }
}

inline void writeUnitStatsRows(std::ostream& out,
        const std::vector<Document>& docs,
        const std::vector<std::string>& ids,
        const VectorXd& residuals, const VectorXd& cosine,
        const VectorXd& entropy, const VectorXd& sensitiveEntropyLcr,
        const VectorXd& sensitiveEntropyQ, bool similarityDiagnostics) {
    for (size_t i = 0; i < docs.size(); ++i) {
        if (!ids[i].empty()) {
            out << ids[i] << "\t";
        }
        out << rawTotalCount(docs[i])
            << "\t" << std::setprecision(2) << residuals(i)
            << "\t" << std::setprecision(4) << entropy(i);
        if (similarityDiagnostics) {
            out << "\t" << std::setprecision(4) << cosine(i)
                << "\t" << std::setprecision(4) << sensitiveEntropyLcr(i)
                << "\t" << std::setprecision(4) << sensitiveEntropyQ(i);
        }
        out << "\n";
    }
}

} // namespace transform_helpers

namespace feature_diagnostics {

struct CofeatureModel {
    VectorXd topicReference;
    MatrixXd featureTopic;
    VectorXd topicInformation;
};

inline CofeatureModel make_cofeature_model(
        const MatrixXd& topicFeature,
        const VectorXd& topicReference) {
    const int32_t topics = static_cast<int32_t>(topicFeature.rows());
    const int32_t features = static_cast<int32_t>(topicFeature.cols());
    CofeatureModel result;
    result.topicReference = topicReference;
    if (result.topicReference.size() != topics
            || !result.topicReference.allFinite()
            || (result.topicReference.array() < 0.0).any()
            || result.topicReference.sum() <= 0.0) {
        result.topicReference =
            VectorXd::Constant(topics, 1.0 / static_cast<double>(topics));
    }
    const double floor = std::numeric_limits<double>::epsilon();
    result.topicReference =
        result.topicReference.array().max(floor);
    result.topicReference /= result.topicReference.sum();

    MatrixXd normalizedTopic = topicFeature.cwiseMax(0.0);
    for (int32_t k = 0; k < topics; ++k) {
        const double total = normalizedTopic.row(k).sum();
        if (std::isfinite(total) && total > 0.0) {
            normalizedTopic.row(k) /= total;
        } else {
            normalizedTopic.row(k).setConstant(
                1.0 / static_cast<double>(features));
        }
    }

    result.featureTopic.resize(topics, features);
    result.topicInformation = VectorXd::Zero(features);
    for (int32_t w = 0; w < features; ++w) {
        const double marginal =
            result.topicReference.dot(normalizedTopic.col(w));
        if (!std::isfinite(marginal) || marginal <= 0.0) {
            result.featureTopic.col(w) = result.topicReference;
            continue;
        }
        result.featureTopic.col(w) =
            result.topicReference.array()
            * normalizedTopic.col(w).array() / marginal;
        for (int32_t k = 0; k < topics; ++k) {
            const double probability = result.featureTopic(k, w);
            if (probability > 0.0) {
                result.topicInformation(w) += probability
                    * std::log(
                        probability / result.topicReference(k));
            }
        }
        result.topicInformation(w) =
            std::max(0.0, result.topicInformation(w));
    }
    return result;
}

template <typename Derived>
inline double cofeature_log_lift(
        const MatrixXd& featureTopic,
        const VectorXd& topicReference,
        const Eigen::MatrixBase<Derived>& documentFeatureTopicSum,
        int32_t feature, int32_t positiveFeatures) {
    if (positiveFeatures <= 1) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    double overlap = 0.0;
    for (int32_t k = 0; k < featureTopic.rows(); ++k) {
        const double rest = std::max(0.0,
            documentFeatureTopicSum(k) - featureTopic(k, feature))
            / static_cast<double>(positiveFeatures - 1);
        overlap += featureTopic(k, feature) * rest
            / topicReference(k);
    }
    return std::log(std::max(
        overlap, std::numeric_limits<double>::min()));
}

struct PullRecord {
    uint32_t feature = 0;
    double observed = 0.0;
    double mean = 0.0;
    double totalVariation = 0.0;
};

struct CellRef {
    uint32_t document = 0;
    uint32_t offset = 0;
};

struct FeatureCellIndex {
    std::vector<size_t> featureOffsets;
    std::vector<CellRef> cellsByFeature;
};

inline std::vector<size_t> make_document_offsets(
        const std::vector<Document>& docs) {
    std::vector<size_t> offsets(docs.size() + 1, 0);
    for (size_t d = 0; d < docs.size(); ++d) {
        offsets[d + 1] = offsets[d] + docs[d].ids.size();
    }
    return offsets;
}

inline FeatureCellIndex make_feature_cell_index(
        const std::vector<Document>& docs, int32_t features,
        size_t cells) {
    FeatureCellIndex result;
    result.featureOffsets.assign(static_cast<size_t>(features) + 1, 0);
    for (const Document& doc : docs) {
        for (uint32_t feature : doc.ids) {
            if (feature >= static_cast<uint32_t>(features)) {
                error("%s: feature index %u is out of range",
                    __func__, feature);
            }
            ++result.featureOffsets[static_cast<size_t>(feature) + 1];
        }
    }
    for (int32_t feature = 0; feature < features; ++feature) {
        result.featureOffsets[static_cast<size_t>(feature) + 1] +=
            result.featureOffsets[static_cast<size_t>(feature)];
    }
    std::vector<size_t> cursor = result.featureOffsets;
    result.cellsByFeature.resize(cells);
    for (size_t d = 0; d < docs.size(); ++d) {
        const Document& doc = docs[d];
        for (size_t j = 0; j < doc.ids.size(); ++j) {
            const uint32_t feature = doc.ids[j];
            result.cellsByFeature[cursor[feature]++] = {
                static_cast<uint32_t>(d), static_cast<uint32_t>(j)};
        }
    }
    return result;
}

struct CofeatureBatchContext {
    RowMajorMatrixXd featureTopicSums;
    std::vector<int32_t> positiveFeatureCounts;
};

inline CofeatureBatchContext make_cofeature_batch_context(
        const std::vector<Document>& docs, const CofeatureModel& model,
        int32_t threadHint) {
    const int32_t documents = static_cast<int32_t>(docs.size());
    const int32_t topics = static_cast<int32_t>(model.featureTopic.rows());
    CofeatureBatchContext result{
        RowMajorMatrixXd::Zero(documents, topics),
        std::vector<int32_t>(docs.size(), 0)};
    const size_t grainSize = std::max<size_t>(
        1, docs.size() / (2 * static_cast<size_t>(threadHint)));
    tbb::parallel_for(tbb::blocked_range<size_t>(
            0, docs.size(), grainSize),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t d = range.begin(); d < range.end(); ++d) {
                const Document& doc = docs[d];
                for (size_t j = 0; j < doc.ids.size(); ++j) {
                    if (doc.cnts[j] <= 0.0) continue;
                    const int32_t feature =
                        static_cast<int32_t>(doc.ids[j]);
                    result.featureTopicSums.row(static_cast<int32_t>(d))
                        += model.featureTopic.col(feature).transpose();
                    ++result.positiveFeatureCounts[d];
                }
            }
        });
    return result;
}

struct CofeatureLiftSums {
    double corroboration = 0.0;
    double conflict = 0.0;
    int64_t contextUnits = 0;
};

inline void accumulate_cofeature_lift(
        const CofeatureModel& model, const CofeatureBatchContext& context,
        size_t document, int32_t feature, CofeatureLiftSums& sums) {
    const double lift = cofeature_log_lift(
        model.featureTopic, model.topicReference,
        context.featureTopicSums.row(
            static_cast<int32_t>(document)).transpose(),
        feature, context.positiveFeatureCounts[document]);
    if (std::isfinite(lift)) {
        sums.corroboration += std::max(0.0, lift);
        sums.conflict += std::max(0.0, -lift);
        ++sums.contextUnits;
    }
}

class DiagnosticSpool {
public:
    DiagnosticSpool(bool useTrainingPrevalence, bool cheapDiagnostics,
            const std::string& tempDir)
        : enabled_(!useTrainingPrevalence),
          storesPullRecords_(!useTrainingPrevalence && !cheapDiagnostics) {
        if (enabled_) {
            const std::filesystem::path parent = tempDir.empty()
                ? std::filesystem::temp_directory_path()
                : std::filesystem::path(tempDir);
            tempDirectory_.init(parent);
            const std::filesystem::path spoolPath =
                tempDirectory_.path / "feature_diagnostics.bin";
            file_ = std::fopen(spoolPath.c_str(), "w+b");
            if (!file_) {
                error("Unable to create temporary feature diagnostic spool: %s",
                    spoolPath.c_str());
            }
        }
    }

    ~DiagnosticSpool() {
        if (file_) {
            std::fclose(file_);
        }
    }

    DiagnosticSpool(const DiagnosticSpool&) = delete;
    DiagnosticSpool& operator=(const DiagnosticSpool&) = delete;

    bool enabled() const {
        return enabled_;
    }

    bool storesPullRecords() const {
        return storesPullRecords_;
    }

    void append(const std::vector<Document>& docs,
            const std::vector<size_t>& documentOffsets,
            const std::vector<PullRecord>& pullRecords) {
        if (!enabled_) return;
        if (documentOffsets.size() != docs.size() + 1) {
            error("Invalid document offsets for feature diagnostic spool");
        }
        if (storesPullRecords_
                && pullRecords.size() != documentOffsets.back()) {
            error("Pull records do not align with feature diagnostic documents");
        }
        for (size_t d = 0; d < docs.size(); ++d) {
            const Document& doc = docs[d];
            uint32_t positive = 0;
            for (double count : doc.cnts) {
                positive += count > 0.0;
            }
            if (positive == 0 || (!storesPullRecords_ && positive <= 1)) {
                continue;
            }
            write(&positive, sizeof(positive), 1);
            for (size_t j = 0; j < doc.ids.size(); ++j) {
                if (doc.cnts[j] <= 0.0) continue;
                if (storesPullRecords_) {
                    const PullRecord& record =
                        pullRecords[documentOffsets[d] + j];
                    if (record.feature != doc.ids[j]
                            || !std::isfinite(record.mean)
                            || record.mean <= 0.0) {
                        error("Invalid pull record in feature diagnostic spool");
                    }
                    write(&record, sizeof(record), 1);
                } else {
                    const uint32_t feature = doc.ids[j];
                    write(&feature, sizeof(feature), 1);
                }
            }
        }
    }

    void accumulate(const CofeatureModel& model, const VectorXd& gain,
            VectorXd& cofeatureCorroborationSums,
            VectorXd& cofeatureConflictSums,
            std::vector<int64_t>& cofeatureContextUnits,
            VectorXd& adjustedResidual, VectorXd& pull,
            int32_t threadHint) {
        if (!enabled_) return;
        if (std::fflush(file_) != 0 || std::fseek(file_, 0, SEEK_SET) != 0) {
            error("Error rewinding temporary feature diagnostic spool");
        }
        const int32_t features = static_cast<int32_t>(gain.size());
        constexpr size_t maxDocuments = 1024;
        while (true) {
            std::vector<Document> docs;
            std::vector<PullRecord> records;
            docs.reserve(maxDocuments);
            bool reachedEnd = false;
            while (docs.size() < maxDocuments) {
                uint32_t positive = 0;
                const size_t n = std::fread(
                    &positive, sizeof(positive), 1, file_);
                if (n == 0) {
                    if (std::ferror(file_)) {
                        error("Error reading temporary feature diagnostic spool");
                    }
                    reachedEnd = true;
                    break;
                }
                if (positive == 0) {
                    error("Invalid empty document in feature diagnostic spool");
                }
                Document doc;
                doc.ids.resize(positive);
                doc.cnts.assign(positive, 1.0);
                if (storesPullRecords_) {
                    const size_t begin = records.size();
                    records.resize(begin + positive);
                    if (std::fread(records.data() + begin, sizeof(PullRecord),
                            positive, file_) != positive) {
                        error("Truncated pull records in feature diagnostic spool");
                    }
                    for (uint32_t j = 0; j < positive; ++j) {
                        doc.ids[j] = records[begin + j].feature;
                    }
                } else if (std::fread(doc.ids.data(), sizeof(uint32_t),
                        positive, file_) != positive) {
                    error("Truncated feature context in diagnostic spool");
                }
                for (uint32_t feature : doc.ids) {
                    if (feature >= static_cast<uint32_t>(features)) {
                        error("Feature index out of range in diagnostic spool");
                    }
                }
                docs.push_back(std::move(doc));
            }
            if (docs.empty()) break;

            const std::vector<size_t> documentOffsets =
                make_document_offsets(docs);
            const CofeatureBatchContext context =
                make_cofeature_batch_context(
                    docs, model, std::max(1, threadHint));
            const FeatureCellIndex featureIndex =
                make_feature_cell_index(
                    docs, features, documentOffsets.back());
            const size_t grain = std::max<size_t>(1,
                static_cast<size_t>(features)
                    / (2 * static_cast<size_t>(std::max(1, threadHint))));
            tbb::parallel_for(tbb::blocked_range<size_t>(
                    0, static_cast<size_t>(features), grain),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t w0 = range.begin(); w0 < range.end(); ++w0) {
                        CofeatureLiftSums cofeatureSums;
                        double adjusted = 0.0;
                        double pullSum = 0.0;
                        for (size_t p = featureIndex.featureOffsets[w0];
                                p < featureIndex.featureOffsets[w0 + 1]; ++p) {
                            const CellRef ref = featureIndex.cellsByFeature[p];
                            accumulate_cofeature_lift(model, context,
                                ref.document, static_cast<int32_t>(w0),
                                cofeatureSums);
                            if (storesPullRecords_) {
                                const PullRecord& record = records[
                                    documentOffsets[ref.document] + ref.offset];
                                const double a = gain(static_cast<int32_t>(w0));
                                if (std::isfinite(a) && a >= 0.0) {
                                    const double difference = std::abs(
                                        record.observed - a * record.mean);
                                    adjusted += difference;
                                    pullSum += difference
                                        * record.totalVariation;
                                }
                            }
                        }
                        cofeatureCorroborationSums(
                            static_cast<int32_t>(w0)) +=
                            cofeatureSums.corroboration;
                        cofeatureConflictSums(static_cast<int32_t>(w0)) +=
                            cofeatureSums.conflict;
                        cofeatureContextUnits[w0] +=
                            cofeatureSums.contextUnits;
                        if (storesPullRecords_) {
                            adjustedResidual(static_cast<int32_t>(w0)) +=
                                adjusted;
                            pull(static_cast<int32_t>(w0)) += pullSum;
                        }
                    }
                });
            if (reachedEnd) break;
        }
    }

private:
    void write(const void* data, size_t width, size_t count) {
        if (std::fwrite(data, width, count, file_) != count) {
            error("Error writing temporary feature diagnostic spool");
        }
    }

    bool enabled_ = false;
    bool storesPullRecords_ = false;
    ScopedTempDir tempDirectory_;
    std::FILE* file_ = nullptr;
};

struct FeatureResidualState {
    CofeatureModel cofeatureModel;
    MatrixXd profileGram;
    MatrixXd topicSimilarity;
    MatrixXd allocatedTopicCounts;
    VectorXd featureCorrections;
    VectorXd featureTotals;
    VectorXd topicExposureTotals;
    VectorXd conditionalLogTerms;
    VectorXd deletionNumerators;
    VectorXd cofeatureCorroborationSums;
    VectorXd cofeatureConflictSums;
    std::vector<int64_t> cofeatureContextUnits;
    std::vector<int64_t> featureUnits;
    VectorXd pullNumerators;
    bool useTrainingPrevalence;
    bool cheapDiagnostics;
    DiagnosticSpool diagnosticSpool;

    VectorXd predictedTotals;
    VectorXd log2Gain;
    VectorXd marginalDeviance;
    VectorXd conditionalDeviance;
    VectorXd topicDeviance;
    VectorXd deletionTV;
    VectorXd cofeatureCorroboration;
    VectorXd cofeatureConflict;
    VectorXd adjustedResidualPerCount;
    VectorXd pull;

    FeatureResidualState(int32_t topics, int32_t features,
            CofeatureModel model, bool cheapDiagnostics_,
            bool useTrainingPrevalence_, const std::string& tempDir)
        : cofeatureModel(std::move(model)),
          allocatedTopicCounts(MatrixXd::Zero(topics, features)),
          featureCorrections(VectorXd::Zero(features)),
          featureTotals(VectorXd::Zero(features)),
          topicExposureTotals(VectorXd::Zero(topics)),
          conditionalLogTerms(VectorXd::Zero(features)),
          deletionNumerators(VectorXd::Zero(features)),
          cofeatureCorroborationSums(VectorXd::Zero(features)),
          cofeatureConflictSums(VectorXd::Zero(features)),
          cofeatureContextUnits(static_cast<size_t>(features), 0),
          featureUnits(static_cast<size_t>(features), 0),
          pullNumerators(VectorXd::Zero(features)),
          useTrainingPrevalence(useTrainingPrevalence_),
          cheapDiagnostics(cheapDiagnostics_),
          diagnosticSpool(
              useTrainingPrevalence_, cheapDiagnostics_, tempDir) {}
};

struct FeatureVarianceDiagnostics {
    VectorXd factorialMoment;
    VectorXd adjustedTopicSecondMoment;
    VectorXd depthSecondMoment;
    VectorXd excessVarianceExplainedByStructure;
    VectorXd totalVarianceExplainedByStructure;
    std::optional<VectorXd> uncertainty;

    explicit FeatureVarianceDiagnostics(
            int32_t features, bool writeUncertainty_ = false)
        : factorialMoment(VectorXd::Zero(features)),
          adjustedTopicSecondMoment(VectorXd::Constant(
              features, std::numeric_limits<double>::quiet_NaN())),
          depthSecondMoment(VectorXd::Constant(
              features, std::numeric_limits<double>::quiet_NaN())),
          excessVarianceExplainedByStructure(VectorXd::Constant(
              features, std::numeric_limits<double>::quiet_NaN())),
          totalVarianceExplainedByStructure(VectorXd::Constant(
              features, std::numeric_limits<double>::quiet_NaN())) {
        if (writeUncertainty_) {
            uncertainty.emplace(VectorXd::Constant(
                features, std::numeric_limits<double>::quiet_NaN()));
        }
    }
};

inline bool clamp_tiny_negative(double& value, double scale) {
    if (value >= 0.0) return true;
    if (value >= -1e-12 * std::max(1.0, scale)) {
        value = 0.0;
        return true;
    }
    return false;
}

inline double positive_raw_dispersion(double factorial, double topicSecond) {
    return std::isfinite(factorial) && std::isfinite(topicSecond)
            && topicSecond > 0.0
        ? std::max(factorial / topicSecond - 1.0, 0.0)
        : std::numeric_limits<double>::quiet_NaN();
}

inline void store_variance_decomposition(FeatureVarianceDiagnostics& diagnostics,
        int32_t feature, double topicSecond, double depthSecond) {
    if (!std::isfinite(topicSecond) || topicSecond < 0.0
            || !std::isfinite(depthSecond) || depthSecond < 0.0) {
        return;
    }
    diagnostics.adjustedTopicSecondMoment(feature) = topicSecond;
    diagnostics.depthSecondMoment(feature) = depthSecond;
    const double extra = diagnostics.factorialMoment(feature) - depthSecond;
    if (std::isfinite(extra) && extra > 0.0) {
        diagnostics.excessVarianceExplainedByStructure(feature) =
            (topicSecond - depthSecond) / extra;
    }
}

inline void accumulate_weighted_topic_sum(const RowMajorMatrixXd& topics,
        const VectorXd& weights, VectorXd& total) {
    if (topics.rows() != weights.size() || topics.cols() != total.size()) {
        error("%s: incompatible topic moment dimensions", __func__);
    }
    total.noalias() += topics.transpose() * weights;
}

inline void accumulate_weighted_topic_second_moment(
        const RowMajorMatrixXd& topics, const VectorXd& weights,
        MatrixXd& total, RowMajorMatrixXd& weightedScratch) {
    if (topics.rows() != weights.size()
            || topics.cols() != total.rows()
            || total.rows() != total.cols()) {
        error("%s: incompatible topic moment dimensions", __func__);
    }
    weightedScratch = topics;
    weightedScratch.array().colwise() *= weights.array();
    total.noalias() += topics.transpose() * weightedScratch;
}

template <typename Derived>
inline void initialize_topic_similarity(
        FeatureResidualState& state,
        const Eigen::MatrixBase<Derived>& topicFeature) {
    state.profileGram.noalias() =
        topicFeature * topicFeature.transpose();
    state.topicSimilarity = MatrixXd::Zero(
        topicFeature.rows(), topicFeature.rows());
    const VectorXd norms =
        state.profileGram.diagonal().array().max(0.0).sqrt();
    for (int32_t k = 0; k < topicFeature.rows(); ++k) {
        state.topicSimilarity(k, k) = 1.0;
        for (int32_t l = k + 1; l < topicFeature.rows(); ++l) {
            const double denominator = norms(k) * norms(l);
            const double similarity = denominator > 0.0
                ? std::clamp(
                    state.profileGram(k, l) / denominator, 0.0, 1.0)
                : 0.0;
            state.topicSimilarity(k, l) = similarity;
            state.topicSimilarity(l, k) = similarity;
        }
    }
}

template <typename PredictionDerived, typename FactorDerived>
inline void finalize_feature_residuals(FeatureResidualState& state,
        const Eigen::MatrixBase<PredictionDerived>& predictionTopicFeature,
        const Eigen::MatrixBase<FactorDerived>& factorTopicFeature,
        const VectorXd& transformTopicReference, int32_t threadHint) {
    const int32_t topics =
        static_cast<int32_t>(factorTopicFeature.rows());
    const int32_t features =
        static_cast<int32_t>(factorTopicFeature.cols());
    state.predictedTotals.noalias() =
        predictionTopicFeature.transpose() * state.topicExposureTotals;
    state.featureCorrections.noalias() += state.predictedTotals;

    const double nan = std::numeric_limits<double>::quiet_NaN();
    state.log2Gain = VectorXd::Constant(features, nan);
    state.marginalDeviance = VectorXd::Constant(features, nan);
    state.conditionalDeviance = VectorXd::Constant(features, nan);
    state.topicDeviance = VectorXd::Constant(features, nan);
    state.deletionTV = VectorXd::Constant(features, nan);
    state.cofeatureCorroboration = VectorXd::Constant(features, nan);
    state.cofeatureConflict = VectorXd::Constant(features, nan);
    state.adjustedResidualPerCount = VectorXd::Constant(features, nan);
    state.pull = VectorXd::Constant(features, nan);

    VectorXd gain = VectorXd::Constant(features, nan);
    for (int32_t feature = 0; feature < features; ++feature) {
        const double observed = state.featureTotals(feature);
        const double predicted = state.predictedTotals(feature);
        if (!std::isfinite(observed) || observed < 0.0
                || !std::isfinite(predicted) || predicted <= 1e-300) {
            continue;
        }
        gain(feature) = observed / predicted;
        if (observed == 0.0) {
            state.log2Gain(feature) =
                -std::numeric_limits<double>::infinity();
            state.marginalDeviance(feature) = 2.0 * predicted;
            state.conditionalDeviance(feature) = 0.0;
            state.topicDeviance(feature) = 0.0;
            continue;
        }

        state.log2Gain(feature) = std::log2(gain(feature));
        state.marginalDeviance(feature) = std::max(0.0,
            2.0 * (observed * std::log(observed / predicted)
                - (observed - predicted)));
        state.conditionalDeviance(feature) = std::max(0.0,
            2.0 * (state.conditionalLogTerms(feature)
                - observed * std::log(gain(feature))));
        double topicDeviance = 0.0;
        for (int32_t k = 0; k < topics; ++k) {
            const double allocated =
                state.allocatedTopicCounts(k, feature);
            if (allocated <= 0.0) continue;
            const double expected = gain(feature)
                * factorTopicFeature(k, feature)
                * state.topicExposureTotals(k);
            if (expected <= 0.0 || !std::isfinite(expected)) {
                topicDeviance = std::numeric_limits<double>::infinity();
                break;
            }
            topicDeviance +=
                2.0 * allocated * std::log(allocated / expected);
        }
        state.topicDeviance(feature) = std::max(0.0, topicDeviance);
        const int64_t expressingUnits =
            state.featureUnits[static_cast<size_t>(feature)];
        if (expressingUnits > 0) {
            state.deletionTV(feature) = state.deletionNumerators(feature)
                / static_cast<double>(expressingUnits);
        }
    }

    VectorXd adjustedNumerator = VectorXd::Zero(features);
    VectorXd pullNumerator = VectorXd::Zero(features);
    if (!state.useTrainingPrevalence) {
        state.cofeatureModel = make_cofeature_model(
            factorTopicFeature, transformTopicReference);
        state.diagnosticSpool.accumulate(
            state.cofeatureModel, gain,
            state.cofeatureCorroborationSums,
            state.cofeatureConflictSums,
            state.cofeatureContextUnits,
            adjustedNumerator, pullNumerator, threadHint);
    }
    for (int32_t feature = 0; feature < features; ++feature) {
        const int64_t contextUnits =
            state.cofeatureContextUnits[static_cast<size_t>(feature)];
        if (contextUnits > 0) {
            state.cofeatureCorroboration(feature) =
                state.cofeatureCorroborationSums(feature)
                / static_cast<double>(contextUnits);
            state.cofeatureConflict(feature) =
                state.cofeatureConflictSums(feature)
                / static_cast<double>(contextUnits);
        }
    }
    if (state.useTrainingPrevalence) {
        for (int32_t feature = 0; feature < features; ++feature) {
            const double observed = state.featureTotals(feature);
            if (observed > 0.0) {
                state.pull(feature) =
                    state.pullNumerators(feature) / observed;
            }
        }
    } else if (state.diagnosticSpool.storesPullRecords()) {
        for (int32_t feature = 0; feature < features; ++feature) {
            const double observed = state.featureTotals(feature);
            if (observed > 0.0 && std::isfinite(gain(feature))) {
                state.adjustedResidualPerCount(feature) =
                    adjustedNumerator(feature) / observed;
                state.pull(feature) =
                    pullNumerator(feature) / observed;
            }
        }
    }
}

inline void write_diagnostic(std::ostream& out, double value) {
    if (std::isnan(value)) {
        out << "NA";
    } else if (std::isinf(value)) {
        out << (value < 0.0 ? "-inf" : "inf");
    } else {
        out << std::scientific << std::setprecision(4) << value;
    }
}

inline void write_diagnostic_precise(std::ostream& out, double value) {
    if (std::isnan(value)) {
        out << "NA";
    } else if (std::isinf(value)) {
        out << (value < 0.0 ? "-inf" : "inf");
    } else {
        out << std::scientific
            << std::setprecision(std::numeric_limits<double>::max_digits10)
            << value;
    }
}

inline void write_feature_residuals(std::ostream& out,
        const std::vector<std::string>& featureNames,
        const FeatureResidualState& state,
        const FeatureVarianceDiagnostics* variance = nullptr) {
    out << "Feature\tabsDiff\tabsDiffRate"
        << "\ttotCount\tnUnits\tlog2Gain"
        << "\tmarginalDev\tconditionalDev"
        << "\tfactorDrift\tdeletionTV"
        << "\ttopicInformation"
        << "\tcofeatureCorroboration"
        << "\tcofeatureConflict";
    if (variance) {
        out << "\tF_w\tQa_w\tQ0_w\tEVES_w\tTVES_w";
        if (variance->uncertainty.has_value()) {
            out << "\tU_w";
        }
    }
    if (state.useTrainingPrevalence) {
        out << "\tpull\n";
    } else {
        out << (state.cheapDiagnostics
            ? "\n" : "\tadjAbsDiffRate\tpull\n");
    }
    for (int32_t feature = 0;
            feature < static_cast<int32_t>(featureNames.size()); ++feature) {
        const double total = state.featureTotals(feature);
        const double difference =
            std::max(0.0, state.featureCorrections(feature));
        const double ratio = total > 0.0 ? difference / total : 0.0;
        out << featureNames[feature]
            << "\t" << std::fixed << std::setprecision(3) << difference
            << "\t" << std::fixed << std::setprecision(6) << ratio
            << "\t" << std::llround(total)
            << "\t" << state.featureUnits[
                static_cast<size_t>(feature)] << "\t";
        write_diagnostic(out, state.log2Gain(feature));
        out << "\t";
        write_diagnostic(out, state.marginalDeviance(feature));
        out << "\t";
        write_diagnostic(out, state.conditionalDeviance(feature));
        out << "\t";
        write_diagnostic(out, state.topicDeviance(feature));
        out << "\t";
        write_diagnostic(out, state.deletionTV(feature));
        out << "\t";
        write_diagnostic(out, state.cofeatureModel.topicInformation(feature));
        out << "\t";
        write_diagnostic(out, state.cofeatureCorroboration(feature));
        out << "\t";
        write_diagnostic(out, state.cofeatureConflict(feature));
        if (variance) {
            out << "\t";
            write_diagnostic_precise(
                out, variance->factorialMoment(feature));
            out << "\t";
            write_diagnostic_precise(
                out, variance->adjustedTopicSecondMoment(feature));
            out << "\t";
            write_diagnostic_precise(
                out, variance->depthSecondMoment(feature));
            out << "\t";
            write_diagnostic(out,
                variance->excessVarianceExplainedByStructure(feature));
            out << "\t";
            write_diagnostic(out,
                variance->totalVarianceExplainedByStructure(feature));
            if (variance->uncertainty.has_value()) {
                out << "\t";
                write_diagnostic_precise(
                    out, (*variance->uncertainty)(feature));
            }
        }
        if (state.useTrainingPrevalence) {
            out << "\t";
            write_diagnostic(out, state.pull(feature));
        } else if (!state.cheapDiagnostics) {
            out << "\t";
            write_diagnostic(out, state.adjustedResidualPerCount(feature));
            out << "\t";
            write_diagnostic(out, state.pull(feature));
        }
        out << "\n";
    }
}

} // namespace feature_diagnostics

namespace transform_pseudobulk {

enum class Mode {
    Standard,
    WeightedModel,
    AllFeatures
};

struct SpecialBatch {
    std::vector<Document> modelDocs;
    std::vector<std::string> ids;
    std::vector<Document> rawDocs;
    std::vector<std::vector<double>> rawModelCounts;

    void clear() {
        modelDocs.clear();
        ids.clear();
        rawDocs.clear();
        rawModelCounts.clear();
    }

    bool empty() const {
        return modelDocs.empty();
    }

    size_t size() const {
        return modelDocs.size();
    }
};

inline Mode selectMode(bool allFeatures, bool weightsActive) {
    if (allFeatures) {
        return Mode::AllFeatures;
    }
    return weightsActive ? Mode::WeightedModel : Mode::Standard;
}

inline std::vector<int32_t> mapInputFeaturesToModel(
        const std::vector<std::string>& inputFeatures,
        const std::vector<std::string>& modelFeatures) {
    std::unordered_map<std::string, int32_t> modelIndex;
    modelIndex.reserve(modelFeatures.size());
    for (size_t i = 0; i < modelFeatures.size(); ++i) {
        modelIndex.emplace(modelFeatures[i], static_cast<int32_t>(i));
    }

    std::vector<int32_t> inputToModel(inputFeatures.size(), -1);
    for (size_t i = 0; i < inputFeatures.size(); ++i) {
        const auto it = modelIndex.find(inputFeatures[i]);
        if (it != modelIndex.end()) {
            inputToModel[i] = it->second;
        }
    }
    return inputToModel;
}

template <typename Model>
bool appendDocument(Document&& rawDoc, std::string id, SpecialBatch& batch,
        Mode mode, const std::vector<int32_t>& inputToModel, Model& model,
        int32_t minCount, bool preserveRawModelCounts = false) {
    Document modelDoc;
    if (mode == Mode::AllFeatures) {
        modelDoc.ids.reserve(rawDoc.ids.size());
        modelDoc.cnts.reserve(rawDoc.cnts.size());
        double modelTotal = 0.0;
        for (size_t j = 0; j < rawDoc.ids.size(); ++j) {
            const uint32_t inputFeature = rawDoc.ids[j];
            if (inputFeature >= inputToModel.size()) {
                continue;
            }
            const int32_t modelFeature = inputToModel[inputFeature];
            if (modelFeature < 0) {
                continue;
            }
            modelDoc.ids.push_back(static_cast<uint32_t>(modelFeature));
            modelDoc.cnts.push_back(rawDoc.cnts[j]);
            modelTotal += rawDoc.cnts[j];
        }
        modelDoc.raw_ct_tot = modelTotal;
        modelDoc.ct_tot = modelTotal;
    } else {
        modelDoc = std::move(rawDoc);
    }

    if (minCount > 0 && modelDoc.get_raw_sum() < minCount) {
        return false;
    }

    if (mode == Mode::WeightedModel || preserveRawModelCounts) {
        batch.rawModelCounts.push_back(modelDoc.cnts);
    }
    if (mode == Mode::AllFeatures) {
        batch.rawDocs.push_back(std::move(rawDoc));
    } else if (mode != Mode::WeightedModel) {
        error("%s: special batch used in standard mode", __func__);
    }
    model.applyWeights(modelDoc);
    batch.modelDocs.push_back(std::move(modelDoc));
    batch.ids.push_back(std::move(id));
    return true;
}

inline void accumulate(MatrixXd& pseudobulk, const SpecialBatch& batch,
        const RowMajorMatrixXd& docTopic, Mode mode) {
    if (docTopic.rows() != static_cast<int32_t>(batch.size())) {
        error("%s: topic rows do not match document count", __func__);
    }
    if (mode == Mode::WeightedModel
            && batch.rawModelCounts.size() != batch.size()) {
        error("%s: raw-count sidecar does not match document count", __func__);
    }
    if (!batch.rawModelCounts.empty()
            && batch.rawModelCounts.size() != batch.size()) {
        error("%s: partial raw-count sidecar", __func__);
    }
    if (mode == Mode::AllFeatures && batch.rawDocs.size() != batch.size()) {
        error("%s: all-feature documents do not match document count", __func__);
    }

    const int32_t K = static_cast<int32_t>(docTopic.cols());
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, K, 1),
        [&](const tbb::blocked_range<int32_t>& range) {
            for (int32_t k = range.begin(); k < range.end(); ++k) {
                if (mode == Mode::WeightedModel) {
                    for (size_t i = 0; i < batch.size(); ++i) {
                        const Document& doc = batch.modelDocs[i];
                        const auto& rawCounts = batch.rawModelCounts[i];
                        if (rawCounts.size() != doc.ids.size()) {
                            error("%s: raw counts do not align with model document",
                                __func__);
                        }
                        const double theta = docTopic(static_cast<int32_t>(i), k);
                        for (size_t j = 0; j < doc.ids.size(); ++j) {
                            pseudobulk(doc.ids[j], k) += rawCounts[j] * theta;
                        }
                    }
                } else if (mode == Mode::AllFeatures) {
                    for (size_t i = 0; i < batch.size(); ++i) {
                        const Document& doc = batch.rawDocs[i];
                        const double theta = docTopic(static_cast<int32_t>(i), k);
                        for (size_t j = 0; j < doc.ids.size(); ++j) {
                            pseudobulk(doc.ids[j], k) += doc.cnts[j] * theta;
                        }
                    }
                }
            }
        });
}

template <typename RandomEngine>
void randomize(SpecialBatch& batch, RandomEngine& randomEngine) {
    if (!batch.rawModelCounts.empty()
            && batch.rawModelCounts.size() != batch.size()) {
        error("%s: partial raw-count sidecar", __func__);
    }
    std::vector<size_t> order(batch.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), randomEngine);

    SpecialBatch shuffled;
    shuffled.modelDocs.reserve(batch.modelDocs.size());
    shuffled.ids.reserve(batch.ids.size());
    shuffled.rawDocs.reserve(batch.rawDocs.size());
    shuffled.rawModelCounts.reserve(batch.rawModelCounts.size());
    for (size_t i : order) {
        shuffled.modelDocs.push_back(std::move(batch.modelDocs[i]));
        shuffled.ids.push_back(std::move(batch.ids[i]));
        if (!batch.rawDocs.empty()) {
            shuffled.rawDocs.push_back(std::move(batch.rawDocs[i]));
        }
        if (!batch.rawModelCounts.empty()) {
            shuffled.rawModelCounts.push_back(std::move(batch.rawModelCounts[i]));
        }
    }
    batch = std::move(shuffled);
}

inline void appendMoved(SpecialBatch& destination, SpecialBatch& source) {
    if ((!destination.rawModelCounts.empty()
            && destination.rawModelCounts.size() != destination.size())
            || (!source.rawModelCounts.empty()
                && source.rawModelCounts.size() != source.size())) {
        error("%s: partial raw-count sidecar", __func__);
    }
    if (!destination.empty() && !source.empty()
            && (destination.rawModelCounts.empty()
                != source.rawModelCounts.empty())) {
        error("%s: incompatible raw-count sidecars", __func__);
    }
    destination.modelDocs.insert(destination.modelDocs.end(),
        std::make_move_iterator(source.modelDocs.begin()),
        std::make_move_iterator(source.modelDocs.end()));
    destination.ids.insert(destination.ids.end(),
        std::make_move_iterator(source.ids.begin()),
        std::make_move_iterator(source.ids.end()));
    destination.rawDocs.insert(destination.rawDocs.end(),
        std::make_move_iterator(source.rawDocs.begin()),
        std::make_move_iterator(source.rawDocs.end()));
    destination.rawModelCounts.insert(destination.rawModelCounts.end(),
        std::make_move_iterator(source.rawModelCounts.begin()),
        std::make_move_iterator(source.rawModelCounts.end()));
    source.clear();
}

inline void moveRange(SpecialBatch& source, size_t begin, size_t count,
        SpecialBatch& destination) {
    if (!source.rawModelCounts.empty()
            && source.rawModelCounts.size() != source.size()) {
        error("%s: partial raw-count sidecar", __func__);
    }
    destination.clear();
    const size_t end = begin + count;
    destination.modelDocs.insert(destination.modelDocs.end(),
        std::make_move_iterator(source.modelDocs.begin() + begin),
        std::make_move_iterator(source.modelDocs.begin() + end));
    destination.ids.insert(destination.ids.end(),
        std::make_move_iterator(source.ids.begin() + begin),
        std::make_move_iterator(source.ids.begin() + end));
    if (!source.rawDocs.empty()) {
        destination.rawDocs.insert(destination.rawDocs.end(),
            std::make_move_iterator(source.rawDocs.begin() + begin),
            std::make_move_iterator(source.rawDocs.begin() + end));
    }
    if (!source.rawModelCounts.empty()) {
        destination.rawModelCounts.insert(destination.rawModelCounts.end(),
            std::make_move_iterator(source.rawModelCounts.begin() + begin),
            std::make_move_iterator(source.rawModelCounts.begin() + end));
    }
}

} // namespace transform_pseudobulk

namespace transform_helpers {

template <typename Model>
bool readSpecialHexMinibatch(std::ifstream& input, HexReader& rawReader,
        Model& model, transform_pseudobulk::SpecialBatch& batch,
        transform_pseudobulk::Mode mode,
        const std::vector<int32_t>& inputToModel, int32_t modal,
        int32_t batchSize, int32_t maxUnits, int32_t minCount,
        bool preserveRawModelCounts = false) {
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
        transform_pseudobulk::appendDocument(
            std::move(rawDoc), std::move(id), batch, mode,
            inputToModel, model, minCount, preserveRawModelCounts);
    }
    return true;
}

template <typename Model>
bool readSpecialDgeMinibatch(DGEReader10X& dge, Model& model,
        transform_pseudobulk::SpecialBatch& batch,
        transform_pseudobulk::Mode mode,
        const std::vector<int32_t>& inputToModel, int32_t batchSize,
        int32_t maxUnits, int32_t minCount,
        bool preserveRawModelCounts = false) {
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
        transform_pseudobulk::appendDocument(
            std::move(rawDoc), dge.getUnitId(unitIndex), batch, mode,
            inputToModel, model, minCount, preserveRawModelCounts);
    }
    return true;
}

inline void writePseudobulkRows(std::ostream& out,
        const std::vector<std::string>& featureNames,
        const RowMajorMatrixXd& standardPseudobulk,
        const MatrixXd& specialPseudobulk,
        transform_pseudobulk::Mode mode, int32_t topics) {
    out << std::fixed << std::setprecision(3);
    for (int32_t feature = 0;
            feature < static_cast<int32_t>(featureNames.size()); ++feature) {
        out << featureNames[feature];
        for (int32_t k = 0; k < topics; ++k) {
            const double value =
                mode == transform_pseudobulk::Mode::Standard
                ? standardPseudobulk(feature, k)
                : specialPseudobulk(feature, k);
            out << "\t" << value;
        }
        out << "\n";
    }
}

} // namespace transform_helpers
