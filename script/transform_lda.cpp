#include "topic_svb.hpp"
#include "transform_helper.hpp"
#include "partition_classifier.hpp"
#include "partition_classifier_lrvb.hpp"

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

struct ClassifierOutput {
    std::string path;
    std::ofstream stream;
    punkst::partition_classifier::Model model;
    punkst::partition_classifier::PropagationOptions propagation;
    int32_t topK = 3;
    bool dense = false;
    uint64_t attempted = 0;
    uint64_t failed = 0;
    uint64_t localNonconvergence = 0;
    uint64_t curvatureFailures = 0;
    uint64_t otherFailures = 0;
    uint64_t successful = 0;
    uint64_t fixedPointIterations = 0;
    uint64_t cgIterations = 0;
    int32_t maximumFixedPointIterations = 0;
    int32_t maximumCgIterations = 0;
    double maximumFixedPointResidual = 0.0;
    double maximumCurvatureJitter = 0.0;

    ClassifierOutput(const std::string& outPrefix,
            const std::string& modelPath,
            const punkst::partition_classifier::PropagationOptions& options,
            int32_t top_k, bool dense_probabilities)
        : path(outPrefix + ".classifications.tsv"), stream(path),
          model(punkst::partition_classifier::Model::read(modelPath)),
          propagation(options), topK(top_k), dense(dense_probabilities) {
        if (!stream) {
            throw std::runtime_error("Cannot write classifications: " + path);
        }
        if (topK <= 0) {
            throw std::invalid_argument("--classifier-top-k must be positive");
        }
    }

    void writeHeader(bool use10x, const std::string& infoHeader) {
        writeUnitIdHeader(stream, use10x, infoHeader);
        stream << "prediction\tmaximum_probability\tentropy"
            "\tpropagation_method\tcandidate_count"
            "\theld_fixed_candidate_tail_mass"
            "\toutput_topk_tail_probability\tlrvb_status";
        if (dense) {
            for (size_t component = 0; component < model.classes.size();
                    ++component) stream << "\tP" << component;
        } else {
            for (int32_t rank = 1; rank <= std::min<int32_t>(
                    topK, model.classes.size()); ++rank) {
                stream << "\tC" << rank << "\tP" << rank;
            }
        }
        stream << "\tfixed_point_iterations\tfixed_point_residual"
            "\tcg_iterations\tcurvature_jitter";
        stream << '\n' << std::scientific << std::setprecision(4);
    }

    void write(const std::string& id,
            const punkst::partition_classifier::PropagatedPrediction& prediction) {
        if (prediction.lrvb_attempted) ++attempted;
        if (prediction.lrvb_attempted) {
            fixedPointIterations += prediction.fixed_point_iterations;
            maximumFixedPointIterations = std::max(
                maximumFixedPointIterations,
                prediction.fixed_point_iterations);
            if (std::isfinite(prediction.fixed_point_residual)) {
                maximumFixedPointResidual = std::max(
                    maximumFixedPointResidual,
                    prediction.fixed_point_residual);
            }
        }
        if (prediction.lrvb_failed) {
            ++failed;
            if (prediction.lrvb_status == "local_nonconvergence") {
                ++localNonconvergence;
            } else if (prediction.lrvb_status.find("curvature")
                    != std::string::npos
                    || prediction.lrvb_status.find("PSD")
                        != std::string::npos
                    || prediction.lrvb_status.find("eigendecomposition")
                        != std::string::npos) {
                ++curvatureFailures;
            } else {
                ++otherFailures;
            }
        } else if (prediction.lrvb_status == "ok") {
            ++successful;
            cgIterations += prediction.cg_iterations;
            maximumCgIterations = std::max(
                maximumCgIterations, prediction.cg_iterations);
            maximumCurvatureJitter = std::max(
                maximumCurvatureJitter, prediction.curvature_jitter);
        }
        std::vector<int32_t> order(prediction.probabilities.size());
        std::iota(order.begin(), order.end(), 0);
        std::stable_sort(order.begin(), order.end(), [&](int32_t left,
                int32_t right) {
            return prediction.probabilities(left)
                > prediction.probabilities(right);
        });
        double entropy = 0.0;
        for (Eigen::Index component = 0;
                component < prediction.probabilities.size(); ++component) {
            const double probability = prediction.probabilities(component);
            if (probability > 0.0) entropy -= probability * std::log(probability);
        }
        const int32_t output_count = std::min<int32_t>(topK, order.size());
        double output_mass = 0.0;
        for (int32_t rank = 0; rank < output_count; ++rank) {
            output_mass += prediction.probabilities(order[rank]);
        }
        stream << id << '\t' << model.classes[order[0]] << '\t'
            << prediction.probabilities(order[0]) << '\t' << entropy << '\t'
            << prediction.method << '\t' << prediction.candidate_count << '\t'
            << prediction.held_fixed_tail_mass << '\t'
            << std::max(0.0, 1.0 - output_mass) << '\t'
            << prediction.lrvb_status;
        if (dense) {
            for (Eigen::Index component = 0;
                    component < prediction.probabilities.size(); ++component) {
                stream << '\t' << prediction.probabilities(component);
            }
        } else {
            for (int32_t rank = 0; rank < output_count; ++rank) {
                stream << '\t' << model.classes[order[rank]] << '\t'
                    << prediction.probabilities(order[rank]);
            }
        }
        stream << '\t' << prediction.fixed_point_iterations << '\t'
            << prediction.fixed_point_residual << '\t'
            << prediction.cg_iterations << '\t'
            << prediction.curvature_jitter;
        stream << '\n';
    }
};

struct ResidualLocalAgg {
    VectorXd topicExposureTotals;
    RowVectorXd denseExpected;

    explicit ResidualLocalAgg(int32_t topics)
        : topicExposureTotals(VectorXd::Zero(topics)) {}
};

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
    feature_diagnostics::FeatureVarianceDiagnostics varianceDiagnostics;
    VectorXd rawFeatureTotals;
    VectorXd countWeightedTopicTotals;
    MatrixXd factorialCountTopicSecondMoment;
    VectorXd unitTopicTotals;
    MatrixXd unitTopicSecondMoment;
    VectorXd inverseCountTopicTotals;
    MatrixXd inverseCountTopicSecondMoment;
    VectorXd uncertaintyTopicTotals;
    MatrixXd uncertaintyTopicSecondMoment;
    double rawCountTotal = 0.0;
    double rawFactorialCountTotal = 0.0;
    int64_t positiveRawCountUnits = 0;

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
          betaAllocationKernel(dirichlet_expectation_2d(model)),
          varianceDiagnostics(static_cast<int32_t>(model.cols()), true),
          rawFeatureTotals(VectorXd::Zero(model.cols())),
          countWeightedTopicTotals(VectorXd::Zero(model.rows())),
          factorialCountTopicSecondMoment(
              MatrixXd::Zero(model.rows(), model.rows())),
          unitTopicTotals(VectorXd::Zero(model.rows())),
          unitTopicSecondMoment(MatrixXd::Zero(model.rows(), model.rows())),
          inverseCountTopicTotals(VectorXd::Zero(model.rows())),
          inverseCountTopicSecondMoment(
              MatrixXd::Zero(model.rows(), model.rows())),
          uncertaintyTopicTotals(VectorXd::Zero(model.rows())),
          uncertaintyTopicSecondMoment(
              MatrixXd::Zero(model.rows(), model.rows())) {
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
            bool similarityDiagnostics_, int32_t nThreads_,
            ClassifierOutput* classifierOutput_ = nullptr)
        : lda(lda_),
          resultsStream(resultsStream_),
          unitMetaStream(unitMetaStream_),
          pseudobulk(pseudobulk_),
          specialPseudobulk(specialPseudobulk_),
          pseudobulkMode(pseudobulkMode_),
          residualState(residualState_),
          topkOnly(topkOnly_),
          similarityDiagnostics(similarityDiagnostics_),
          classifierOutput(classifierOutput_),
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
        writeClassificationRows(batch.docs, batch.ids, gamma);

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

        processResiduals(batch.docs, batch.ids, doc_topic, gamma, nullptr);
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
        writeClassificationRows(batch.modelDocs, batch.ids, gamma);
        transform_pseudobulk::accumulate(
            specialPseudobulk, batch, doc_topic, pseudobulkMode);
        processResiduals(batch.modelDocs, batch.ids, doc_topic, gamma,
            batch.rawModelCounts.empty() ? nullptr : &batch.rawModelCounts);
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
        if (residualState == nullptr && classifierOutput == nullptr) {
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

    void writeClassificationRows(const std::vector<Document>& docs,
            const std::vector<std::string>& ids,
            const RowMajorMatrixXd& gamma) {
        if (classifierOutput == nullptr) return;
        if (gamma.rows() != static_cast<Eigen::Index>(docs.size())
                || ids.size() != docs.size()) {
            error("%s: classifier posterior batch dimensions do not match",
                __func__);
        }
        std::vector<punkst::partition_classifier::PropagatedPrediction>
            predictions(docs.size());
        tbb::parallel_for(0, static_cast<int32_t>(docs.size()),
            [&](int32_t document) {
                predictions[static_cast<size_t>(document)] =
                    punkst::partition_classifier::propagate_lda(
                    classifierOutput->model,
                    gamma.row(document).transpose(),
                    docs[static_cast<size_t>(document)],
                    lda.get_allocation_kernel(),
                    lda.get_doc_topic_prior(), classifierOutput->propagation);
            });
        for (size_t document = 0; document < docs.size(); ++document) {
            classifierOutput->write(ids[document], predictions[document]);
        }
    }

    void processResiduals(const std::vector<Document>& docs,
            const std::vector<std::string>& ids,
            const RowMajorMatrixXd& docTopic,
            const RowMajorMatrixXd& gamma,
            const std::vector<std::vector<double>>* rawModelCounts) {
        if (!residualState) return;
        const int32_t nDocs = static_cast<int32_t>(docs.size());
        if (rawModelCounts != nullptr
                && rawModelCounts->size() != docs.size()) {
            error("%s: raw-count sidecar does not match document count",
                __func__);
        }
        const double alpha = lda.get_doc_topic_prior();
        VectorXd rawTotals = VectorXd::Zero(nDocs);
        VectorXd rawFactorials = VectorXd::Zero(nDocs);
        VectorXd positiveUnits = VectorXd::Zero(nDocs);
        VectorXd inverseRawTotals = VectorXd::Zero(nDocs);
        VectorXd uncertaintyWeights = VectorXd::Zero(nDocs);
        RowMajorMatrixXd posteriorMeans = RowMajorMatrixXd::Zero(nDocs, K);
        for (int32_t d = 0; d < nDocs; ++d) {
            const Document& doc = docs[static_cast<size_t>(d)];
            const std::vector<double>* rawCounts = rawModelCounts == nullptr
                ? nullptr
                : &(*rawModelCounts)[static_cast<size_t>(d)];
            if (rawCounts != nullptr
                    && rawCounts->size() != doc.ids.size()) {
                error("%s: raw counts do not align with model document",
                    __func__);
            }
            double rawTotal = 0.0;
            for (size_t j = 0; j < doc.ids.size(); ++j) {
                const double rawObserved = rawCounts == nullptr
                    ? doc.cnts[j]
                    : (*rawCounts)[j];
                if (!std::isfinite(rawObserved) || rawObserved < 0.0) {
                    error("%s: invalid raw count", __func__);
                }
                const uint32_t w = doc.ids[j];
                residualState->rawFeatureTotals(w) += rawObserved;
                residualState->varianceDiagnostics.factorialMoment(w) +=
                    rawObserved * (rawObserved - 1.0);
                rawTotal += rawObserved;
            }

            if (!(rawTotal > 0.0) || !std::isfinite(rawTotal)) {
                continue;
            }
            const double rawFactorial = rawTotal * (rawTotal - 1.0);
            rawTotals(d) = rawTotal;
            rawFactorials(d) = rawFactorial;
            positiveUnits(d) = 1.0;
            inverseRawTotals(d) = 1.0 / rawTotal;
            residualState->rawCountTotal += rawTotal;
            residualState->rawFactorialCountTotal += rawFactorial;
            ++residualState->positiveRawCountUnits;

            const double concentration = gamma.row(d).sum()
                + static_cast<double>(K) * alpha;
            if (std::isfinite(concentration) && concentration > 0.0) {
                posteriorMeans.row(d) =
                    (gamma.row(d).array() + alpha) / concentration;
                if (posteriorMeans.row(d).allFinite()
                        && (posteriorMeans.row(d).array() >= 0.0).all()) {
                    uncertaintyWeights(d) =
                        rawFactorial / (concentration + 1.0);
                } else {
                    posteriorMeans.row(d).setZero();
                }
            }
        }
        RowMajorMatrixXd weightedTopics;
        feature_diagnostics::accumulate_weighted_topic_sum(
            docTopic, rawTotals,
            residualState->countWeightedTopicTotals);
        feature_diagnostics::accumulate_weighted_topic_second_moment(
            docTopic, rawFactorials,
            residualState->factorialCountTopicSecondMoment, weightedTopics);
        feature_diagnostics::accumulate_weighted_topic_sum(
            docTopic, positiveUnits, residualState->unitTopicTotals);
        feature_diagnostics::accumulate_weighted_topic_second_moment(
            docTopic, positiveUnits,
            residualState->unitTopicSecondMoment, weightedTopics);
        feature_diagnostics::accumulate_weighted_topic_sum(
            docTopic, inverseRawTotals,
            residualState->inverseCountTopicTotals);
        feature_diagnostics::accumulate_weighted_topic_second_moment(
            docTopic, inverseRawTotals,
            residualState->inverseCountTopicSecondMoment, weightedTopics);
        feature_diagnostics::accumulate_weighted_topic_sum(
            posteriorMeans, uncertaintyWeights,
            residualState->uncertaintyTopicTotals);
        feature_diagnostics::accumulate_weighted_topic_second_moment(
            posteriorMeans, uncertaintyWeights,
            residualState->uncertaintyTopicSecondMoment, weightedTopics);
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
        EntropyStats similarityEntropy;
        VectorXd expectedNormSq;
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
        finalizeVarianceDiagnostics();
    }

    void finalizeVarianceDiagnostics() {
        auto& state = *residualState;
        auto& diagnostics = state.varianceDiagnostics;
        if (!(state.rawCountTotal > 0.0)
                || state.positiveRawCountUnits <= 0) {
            return;
        }

        const VectorXd predicted = state.betaNormCol.transpose()
            * state.countWeightedTopicTotals;
        const MatrixXd factorialCross =
            state.factorialCountTopicSecondMoment * state.betaNormCol;
        const MatrixXd unitCross =
            state.unitTopicSecondMoment * state.betaNormCol;
        const MatrixXd inverseCountCross =
            state.inverseCountTopicSecondMoment * state.betaNormCol;
        const MatrixXd uncertaintyCross =
            state.uncertaintyTopicSecondMoment * state.betaNormCol;
        const double positiveUnits =
            static_cast<double>(state.positiveRawCountUnits);
        const double nan = std::numeric_limits<double>::quiet_NaN();

        for (int32_t w = 0; w < M; ++w) {
            const double observed = state.rawFeatureTotals(w);
            if (!std::isfinite(observed) || observed < 0.0
                    || !std::isfinite(predicted(w))
                    || !(predicted(w) > 1e-300)) {
                continue;
            }
            const VectorXd beta = state.betaNormCol.col(w);
            const double gain = observed / predicted(w);
            const double gainSquared = gain * gain;
            const double topicQ = beta.dot(factorialCross.col(w));
            const double qa = gainSquared * topicQ;
            const double marginal = observed / state.rawCountTotal;
            const double q0 = marginal * marginal
                * state.rawFactorialCountTotal;
            feature_diagnostics::store_variance_decomposition(
                diagnostics, w, qa, q0);

            const double adjustedUnitQ = gainSquared
                * beta.dot(unitCross.col(w));
            const double adjustedUnitMean = gain
                * beta.dot(state.unitTopicTotals);
            double structuredVariance = adjustedUnitQ
                - adjustedUnitMean * adjustedUnitMean / positiveUnits;
            const double structuredScale = std::max(
                std::abs(adjustedUnitQ),
                adjustedUnitMean * adjustedUnitMean / positiveUnits);

            double samplingVariance = gain
                    * beta.dot(state.inverseCountTopicTotals)
                - gainSquared * beta.dot(inverseCountCross.col(w));
            const double samplingScale = std::max(
                std::abs(gain
                    * beta.dot(state.inverseCountTopicTotals)),
                std::abs(gainSquared
                    * beta.dot(inverseCountCross.col(w))));

            double dispersionQuadratic = gainSquared * beta.dot(
                unitCross.col(w) - inverseCountCross.col(w));
            const double dispersionScale = gainSquared * std::max(
                std::abs(beta.dot(unitCross.col(w))),
                std::abs(beta.dot(inverseCountCross.col(w))));

            if (qa > 0.0 && std::isfinite(qa)
                    && feature_diagnostics::clamp_tiny_negative(
                        structuredVariance, structuredScale)
                    && feature_diagnostics::clamp_tiny_negative(
                        samplingVariance, samplingScale)
                    && feature_diagnostics::clamp_tiny_negative(
                        dispersionQuadratic, dispersionScale)) {
                const double positiveDispersion =
                    feature_diagnostics::positive_raw_dispersion(
                        diagnostics.factorialMoment(w), qa);
                const double denominator = structuredVariance
                    + samplingVariance
                    + positiveDispersion * dispersionQuadratic;
                if (std::isfinite(denominator) && denominator > 0.0) {
                    diagnostics.totalVarianceExplainedByStructure(w) =
                        structuredVariance / denominator;
                }
            }

            double uncertainty = gainSquared * (
                beta.array().square().matrix().dot(
                    state.uncertaintyTopicTotals)
                - beta.dot(uncertaintyCross.col(w)));
            const double uncertaintyScale = gainSquared * std::max(
                std::abs(beta.array().square().matrix().dot(
                    state.uncertaintyTopicTotals)),
                std::abs(beta.dot(uncertaintyCross.col(w))));
            (*diagnostics.uncertainty)(w) =
                std::isfinite(uncertainty)
                    && feature_diagnostics::clamp_tiny_negative(
                        uncertainty, uncertaintyScale)
                ? uncertainty
                : nan;
        }
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
    ClassifierOutput* classifierOutput;
    int32_t threadHint;
    int32_t M;
    int32_t K;
    std::unique_ptr<StandardTls> standardTls;
    std::unique_ptr<ResidualTls> residualTls;
};

} // namespace

int32_t cmdLDATransform(int argc, char** argv) {
    std::string inFile, metaFile, modelFile, stateFile, outPrefix, featureFile, temp_dir;
    std::string classifier_model;
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
    double alpha = -1.0;
    double classifier_ambiguity_threshold = 0.95;
    double classifier_candidate_mass = 0.999;
    double classifier_max_failure_rate = 0.01;
    double classifier_fixed_point_tolerance = 1e-7;
    int32_t classifier_top_k = 3;
    int32_t classifier_bootstrap_draws = 64;
    int32_t classifier_fixed_point_max_iterations = 5000;
    int32_t topk_only = -1;
    bool computeResiduals = false;
    bool cheap_feature_diagnostics = false;
    bool use_training_prevalence = false;
    bool unit_similarity_diagnostics = false;
    bool sorted_by_barcode = false;
    bool keep_barcodes = false;
    bool pseudobulk_all_features = false;
    bool classifier_dense = false;
    bool classifier_lrvb_all = false;
    bool classifier_plugin_only = false;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("in-model", "Legacy input model matrix (topic-word) file", modelFile)
      .add_option("in-state", "Versioned LDA SVB state", stateFile)
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

    pl.add_option("alpha", "Document-topic prior required for legacy-model LRVB", alpha)
      .add_option("classifier-model", "Partition classifier model", classifier_model)
      .add_option("classifier-top-k", "Classification class/probability pairs", classifier_top_k)
      .add_option("classifier-dense-probabilities", "Write dense classification probabilities", classifier_dense)
      .add_option("classifier-ambiguity-threshold", "Skip LRVB above this leading probability", classifier_ambiguity_threshold)
      .add_option("classifier-candidate-mass", "Candidate-class probability mass", classifier_candidate_mass)
      .add_option("classifier-lrvb-all", "Attempt LRVB for all units", classifier_lrvb_all)
      .add_option("classifier-plugin-only", "Disable classifier uncertainty propagation", classifier_plugin_only)
      .add_option("classifier-fixed-point-tol", "Scale-aware classifier fixed-point tolerance", classifier_fixed_point_tolerance)
      .add_option("classifier-fixed-point-max-iter", "Maximum classifier fixed-point iterations", classifier_fixed_point_max_iterations)
      .add_option("classifier-max-lrvb-failure-rate", "Maximum attempted-unit LRVB failure rate", classifier_max_failure_rate)
      .add_option("classifier-bootstrap-draws", "Reserved one-step bootstrap draw count", classifier_bootstrap_draws);

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
    if (modelFile.empty() == stateFile.empty()) {
        error("Exactly one of --in-model or --in-state is required");
    }
    if (!classifier_model.empty()
            && !(classifier_ambiguity_threshold > 0.0
                && classifier_ambiguity_threshold <= 1.0
                && classifier_candidate_mass > 0.0
                && classifier_candidate_mass <= 1.0
                && std::isfinite(classifier_fixed_point_tolerance)
                && classifier_fixed_point_tolerance > 0.0
                && classifier_fixed_point_max_iterations > 0
                && classifier_max_failure_rate >= 0.0
                && classifier_max_failure_rate <= 1.0
                && classifier_bootstrap_draws > 0)) {
        error("Invalid classifier propagation option");
    }
    if (!classifier_model.empty() && stateFile.empty()
            && !classifier_plugin_only && !(alpha > 0.0)) {
        error("LDA LRVB with --in-model requires explicit positive --alpha");
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
    std::optional<LdaState> ldaState;
    if (!stateFile.empty()) {
        ldaState = LdaState::read(stateFile);
        lda.initialize_transform(*ldaState,
            seed, nThreads, verbose, maxIter, mDelta);
    } else {
        lda.initialize_transform(modelFile,
            seed, nThreads, verbose, maxIter, mDelta, alpha);
    }
    lda.set_reproducible_init(true);

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
    const bool preserveRawModelCounts = computeResiduals && weights_active;

    TransformOutputs outputs(outPrefix, computeResiduals);
    writeUnitIdHeader(outputs.results, use_10x, info_header);
    writeResultHeader(outputs.results, lda, topk_only);
    outputs.results << std::scientific << std::setprecision(4);
    std::unique_ptr<ClassifierOutput> classifierOutput;
    if (!classifier_model.empty()) {
        punkst::partition_classifier::PropagationOptions propagation;
        propagation.ambiguity_threshold = classifier_ambiguity_threshold;
        propagation.candidate_mass = classifier_candidate_mass;
        propagation.lrvb_all = classifier_lrvb_all;
        propagation.plugin_only = classifier_plugin_only;
        propagation.fixed_point_tolerance =
            classifier_fixed_point_tolerance;
        propagation.fixed_point_max_iterations =
            classifier_fixed_point_max_iterations;
        classifierOutput = std::make_unique<ClassifierOutput>(outPrefix,
            classifier_model, propagation, classifier_top_k, classifier_dense);
        if (classifierOutput->model.topics != lda.get_topic_names()) {
            error("Classifier topics do not exactly match LDA topic names/order");
        }
        classifierOutput->writeHeader(use_10x, info_header);
    }
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
        unit_similarity_diagnostics, nThreads, classifierOutput.get());

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
                        minCountInt, preserveRawModelCounts);
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
                            minCountInt, preserveRawModelCounts);
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
                    batchSize, remaining, minCountInt,
                    preserveRawModelCounts);
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
    bool classifier_failure = false;
    if (classifierOutput) {
        classifierOutput->stream.close();
        notice("Partition classifications written to %s",
            classifierOutput->path.c_str());
        const double failure_rate = classifierOutput->attempted > 0
            ? static_cast<double>(classifierOutput->failed)
                / classifierOutput->attempted : 0.0;
        notice("LDA classifier LRVB attempted %zu units; %zu failed (%.6g)",
            classifierOutput->attempted, classifierOutput->failed, failure_rate);
        const std::string diagnosticPath =
            outPrefix + ".classification_diagnostics.tsv";
        std::ofstream diagnostic(diagnosticPath);
        if (!diagnostic) {
            error("Cannot write classification diagnostics: %s",
                diagnosticPath.c_str());
        }
        const double meanFixedPointIterations = classifierOutput->attempted > 0
            ? static_cast<double>(classifierOutput->fixedPointIterations)
                / classifierOutput->attempted : 0.0;
        const double meanCgIterations = classifierOutput->successful > 0
            ? static_cast<double>(classifierOutput->cgIterations)
                / classifierOutput->successful : 0.0;
        diagnostic << "#attempted\tfailed\tfailure_rate\tmaximum_failure_rate"
            "\tlocal_nonconvergence\tcurvature_failures\tother_failures"
            "\tmean_fixed_point_iterations\tmax_fixed_point_iterations"
            "\tmax_fixed_point_residual\tmean_cg_iterations"
            "\tmax_cg_iterations\tmax_curvature_jitter\n"
            << classifierOutput->attempted << '\t' << classifierOutput->failed
            << '\t' << std::scientific << std::setprecision(4)
            << failure_rate << '\t' << classifier_max_failure_rate << '\t'
            << classifierOutput->localNonconvergence << '\t'
            << classifierOutput->curvatureFailures << '\t'
            << classifierOutput->otherFailures << '\t'
            << meanFixedPointIterations << '\t'
            << classifierOutput->maximumFixedPointIterations << '\t'
            << classifierOutput->maximumFixedPointResidual << '\t'
            << meanCgIterations << '\t'
            << classifierOutput->maximumCgIterations << '\t'
            << classifierOutput->maximumCurvatureJitter << '\n';
        classifier_failure = failure_rate > classifier_max_failure_rate;
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

    if (!computeResiduals) return classifier_failure ? 1 : 0;
    outFile = outPrefix + ".feature_residuals.tsv";
    outFileStream.open(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());
    feature_diagnostics::write_feature_residuals(
        outFileStream, modelFeatureNames, *residualState,
        &residualState->varianceDiagnostics);
    outFileStream.close();
    notice("Per-feature residuals written to %s", outFile.c_str());

    return classifier_failure ? 1 : 0;
}
