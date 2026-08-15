#include "gamma_pois_topic.hpp"
#include "transform_helper.hpp"
#include "partition_classifier.hpp"
#include "partition_classifier_lrvb.hpp"
#include "factor_result_stream.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
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
using transform_helpers::TransformBatch;
using transform_helpers::applyWeights;
using transform_helpers::assignBarcodeIds;
using transform_helpers::readSpecialDgeMinibatch;
using transform_helpers::readSpecialHexMinibatch;
using transform_helpers::writeUnitIdHeader;
using feature_diagnostics::PullRecord;

struct GammaClassifierOutput {
    std::string path;
    std::ofstream stream;
    punkst::partition_classifier::Model model;
    std::unique_ptr<punkst::partition_classifier::CrossfitBundle> bundle;
    bool useCrossfit = false;
    uint64_t heldoutModelRows = 0;
    uint64_t fullModelRows = 0;
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

    GammaClassifierOutput(const std::string& prefix, const std::string& modelPath,
            const punkst::partition_classifier::PropagationOptions& options,
            int32_t top_k, bool dense_probabilities, bool use_crossfit)
        : path(prefix + ".classifications.tsv"), stream(path),
          useCrossfit(use_crossfit), propagation(options), topK(top_k),
          dense(dense_probabilities) {
        if (punkst::partition_classifier::CrossfitBundle::is_bundle(modelPath)) {
            bundle = std::make_unique<
                punkst::partition_classifier::CrossfitBundle>(
                    punkst::partition_classifier::CrossfitBundle::read(modelPath));
            model = bundle->full_model;
        } else {
            if (useCrossfit) {
                throw std::invalid_argument(
                    "--classifier-crossfit requires a crossfit classifier bundle");
            }
            model = punkst::partition_classifier::Model::read(modelPath);
        }
        if (!stream) throw std::runtime_error("Cannot write classifications: " + path);
        if (topK <= 0) throw std::invalid_argument("--classifier-top-k must be positive");
    }

    const punkst::partition_classifier::Model& modelFor(
            const std::string& id, int32_t* heldout_fold) const {
        if (bundle) return bundle->model_for(id, useCrossfit, heldout_fold);
        if (heldout_fold != nullptr) *heldout_fold = -1;
        return model;
    }

    void writeHeader(bool use10x, const std::string& infoHeader) {
        writeUnitIdHeader(stream, use10x, infoHeader);
        stream << "classifier_model_source\tentropy"
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
            const punkst::partition_classifier::PropagatedPrediction& prediction,
            int32_t heldout_fold) {
        if (useCrossfit && heldout_fold >= 0) ++heldoutModelRows;
        else ++fullModelRows;
        attempted += prediction.lrvb_attempted;
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
                int32_t right) { return prediction.probabilities(left)
                    > prediction.probabilities(right); });
        double entropy = 0.0;
        for (const double probability : prediction.probabilities) {
            if (probability > 0.0) entropy -= probability * std::log(probability);
        }
        const int32_t output_count = std::min<int32_t>(topK, order.size());
        double output_mass = 0.0;
        for (int32_t rank = 0; rank < output_count; ++rank) {
            output_mass += prediction.probabilities(order[rank]);
        }
        stream << id << '\t';
        if (useCrossfit) {
            if (heldout_fold >= 0) stream << "heldout_fold_" << heldout_fold;
            else stream << "full_unseen";
        } else {
            stream << "full";
        }
        stream << '\t' << entropy << '\t' << prediction.method << '\t'
            << prediction.candidate_count << '\t'
            << prediction.held_fixed_tail_mass << '\t'
            << std::max(0.0, 1.0 - output_mass) << '\t'
            << prediction.lrvb_status;
        if (dense) {
            for (const double probability : prediction.probabilities) {
                stream << '\t' << probability;
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
    RowVectorXd denseExpected;

    explicit ResidualLocalAgg(int32_t) {}
};

VectorXd gammaPoissonTopicReference(GammaPoisson4Hex& gp) {
    std::vector<double> abundance;
    gp.get_topic_abundance(abundance);
    VectorXd reference(static_cast<int32_t>(abundance.size()));
    for (int32_t k = 0; k < reference.size(); ++k) {
        reference(k) = abundance[static_cast<size_t>(k)];
    }
    return reference;
}

struct ResidualState : feature_diagnostics::FeatureResidualState {
    const MatrixXd& expectedBeta;
    const MatrixXd& betaAllocationKernel;
    const VectorXd& topicCapacity;
    const VectorXd& featureDispersion;
    bool hasFeatureDispersion;
    bool featureWeightsActive;
    bool requiresRawCountSidecar = false;
    std::vector<double> featureWeights;
    feature_diagnostics::FeatureVarianceDiagnostics varianceDiagnostics;
    VectorXd rawFeatureTotals;
    MatrixXd topicSecondMoment;
    MatrixXd rateSecondMoment;
    VectorXd rateTopicTotals;
    VectorXd inverseExposureRateTopicTotals;
    double exposureTotal = 0.0;
    double exposureSquaredTotal = 0.0;
    int64_t positiveExposureUnits = 0;

    ResidualState(GammaPoisson4Hex& gp, bool cheapDiagnostics,
            bool similarityDiagnostics, bool useTrainingPrevalence,
            const std::string& tempDir)
        : FeatureResidualState(
              static_cast<int32_t>(gp.getExpectedBeta().rows()),
              static_cast<int32_t>(gp.getExpectedBeta().cols()),
              useTrainingPrevalence
                  ? feature_diagnostics::make_cofeature_model(
                      gp.getExpectedBeta(), gammaPoissonTopicReference(gp))
                  : feature_diagnostics::CofeatureModel{},
              cheapDiagnostics, useTrainingPrevalence, tempDir),
          expectedBeta(gp.getExpectedBeta()),
          betaAllocationKernel(gp.getBetaAllocationKernel()),
          topicCapacity(gp.getTopicCapacity()),
          featureDispersion(gp.getFeatureDispersion()),
          hasFeatureDispersion(gp.hasFeatureDispersion()),
          featureWeightsActive(gp.featureWeightsActive()),
          featureWeights(featureWeightsActive
              ? gp.getFeatureWeights()
              : std::vector<double>(
                  static_cast<size_t>(gp.getExpectedBeta().cols()), 1.0)),
          varianceDiagnostics(
              static_cast<int32_t>(gp.getExpectedBeta().cols())),
          rawFeatureTotals(VectorXd::Zero(gp.getExpectedBeta().cols())),
          topicSecondMoment(MatrixXd::Zero(
              gp.getExpectedBeta().rows(), gp.getExpectedBeta().rows())),
          rateSecondMoment(MatrixXd::Zero(
              gp.getExpectedBeta().rows(), gp.getExpectedBeta().rows())),
          rateTopicTotals(VectorXd::Zero(gp.getExpectedBeta().rows())),
          inverseExposureRateTopicTotals(
              VectorXd::Zero(gp.getExpectedBeta().rows())) {
        if (similarityDiagnostics) {
            feature_diagnostics::initialize_topic_similarity(
                *this, expectedBeta);
        }
        if (featureWeights.size()
                != static_cast<size_t>(gp.getExpectedBeta().cols())) {
            error("%s: invalid Gamma-Poisson feature weights", __func__);
        }
        for (double weight : featureWeights) {
            if (!std::isfinite(weight) || weight < 0.0) {
                error("%s: invalid Gamma-Poisson feature weight", __func__);
            }
            requiresRawCountSidecar = requiresRawCountSidecar || weight == 0.0;
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
        bool similarityDiagnostics_, int32_t nThreads_,
        GammaClassifierOutput* classifierOutput_ = nullptr,
        punkst::FactorResultStream* factorResults_ = nullptr)
        : gp(gp_), results(results_), pseudobulk(pseudobulk_),
          specialPseudobulk(specialPseudobulk_), pseudobulkMode(pseudobulkMode_),
          unitStats(unitStats_), residualState(residualState_),
          similarityDiagnostics(similarityDiagnostics_),
          classifierOutput(classifierOutput_),
          factorResults(factorResults_),
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
        if (factorResults != nullptr) {
            const Eigen::MatrixXd compositions =
                factorResults->next_batch(batch.ids);
            writeWarmStartClassificationRows(
                batch.docs, batch.ids, compositions);
            return;
        }
        RowMajorMatrixXd doc_topic;
        std::vector<GammaPoissonDocumentPosterior> posteriors;
        if (residualState || classifierOutput) {
            gp.transformWithPosteriors(
                DocumentView(batch.docs), doc_topic, posteriors);
        } else {
            doc_topic = gp.transformMeans(DocumentView(batch.docs));
        }
        writeTopicRows(batch.ids, doc_topic);
        writeClassificationRows(batch.docs, batch.ids, posteriors);
        processResiduals(batch.docs, batch.ids, doc_topic, posteriors, nullptr);
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
        if (factorResults != nullptr) {
            const Eigen::MatrixXd compositions =
                factorResults->next_batch(batch.ids);
            writeWarmStartClassificationRows(
                batch.modelDocs, batch.ids, compositions);
            return;
        }
        if (pseudobulkMode == Mode::Standard) {
            error("%s: special batch used with standard pseudobulk mode", __func__);
        }
        RowMajorMatrixXd doc_topic;
        std::vector<GammaPoissonDocumentPosterior> posteriors;
        if (residualState || classifierOutput) {
            gp.transformWithPosteriors(
                DocumentView(batch.modelDocs), doc_topic, posteriors);
        } else {
            doc_topic = gp.transformMeans(DocumentView(batch.modelDocs));
        }
        writeTopicRows(batch.ids, doc_topic);
        writeClassificationRows(batch.modelDocs, batch.ids, posteriors);
        processResiduals(batch.modelDocs, batch.ids, doc_topic, posteriors,
            batch.rawModelCounts.empty() ? nullptr : &batch.rawModelCounts);
        transform_pseudobulk::accumulate(
            specialPseudobulk, batch, doc_topic, pseudobulkMode);
    }

    void finalizeResiduals() {
        if (factorResults != nullptr) return;
        if (!residualState) return;
        feature_diagnostics::finalize_feature_residuals(
            *residualState, residualState->expectedBeta,
            residualState->expectedBeta,
            (residualState->topicExposureTotals.array()
                * residualState->topicCapacity.array()).matrix(),
            threadHint);
        finalizeVarianceDiagnostics();
    }

private:
    using ResidualTls =
        tbb::enumerable_thread_specific<ResidualLocalAgg>;

    void writeTopicRows(const std::vector<std::string>& ids,
        const RowMajorMatrixXd& doc_topic) {
        transform_helpers::writeTopicRows(results, ids, doc_topic);
    }

    void writeClassificationRows(const std::vector<Document>& docs,
            const std::vector<std::string>& ids,
            const std::vector<GammaPoissonDocumentPosterior>& posteriors) {
        if (classifierOutput == nullptr) return;
        if (docs.size() != ids.size() || docs.size() != posteriors.size()) {
            error("%s: classifier posterior batch dimensions do not match",
                __func__);
        }
        const VectorXd priorRate = gp.getThetaPriorRate();
        const VectorXd* dispersion = gp.hasFeatureDispersion()
            ? &gp.getFeatureDispersion() : nullptr;
        std::vector<punkst::partition_classifier::PropagatedPrediction>
            predictions(docs.size());
        std::vector<const punkst::partition_classifier::Model*>
            classifierModels(docs.size());
        std::vector<int32_t> heldoutFolds(docs.size(), -1);
        for (size_t document = 0; document < docs.size(); ++document) {
            classifierModels[document] = &classifierOutput->modelFor(
                ids[document], &heldoutFolds[document]);
        }
        tbb::parallel_for(0, static_cast<int32_t>(docs.size()),
            [&](int32_t document) {
                predictions[static_cast<size_t>(document)] =
                    punkst::partition_classifier::propagate_gamma_poisson(
                    *classifierModels[static_cast<size_t>(document)],
                    posteriors[static_cast<size_t>(document)],
                    docs[static_cast<size_t>(document)], gp.getTopicCapacity(),
                    gp.getBetaAllocationKernel(), gp.getExpectedBeta(),
                    gp.getThetaPriorShape(), priorRate, dispersion,
                    classifierOutput->propagation);
            });
        for (size_t document = 0; document < docs.size(); ++document) {
            classifierOutput->write(ids[document], predictions[document],
                heldoutFolds[document]);
        }
    }

    void writeWarmStartClassificationRows(
            const std::vector<Document>& docs,
            const std::vector<std::string>& ids,
            const Eigen::Ref<const Eigen::MatrixXd>& compositions) {
        if (classifierOutput == nullptr) return;
        if (compositions.rows() != static_cast<Eigen::Index>(docs.size())
                || compositions.cols() != K || ids.size() != docs.size()) {
            error("%s: classifier warm-start batch dimensions do not match",
                __func__);
        }
        const VectorXd priorRate = gp.getThetaPriorRate();
        const VectorXd* dispersion = gp.hasFeatureDispersion()
            ? &gp.getFeatureDispersion() : nullptr;
        std::vector<punkst::partition_classifier::PropagatedPrediction>
            predictions(docs.size());
        std::vector<const punkst::partition_classifier::Model*>
            classifierModels(docs.size());
        std::vector<int32_t> heldoutFolds(docs.size(), -1);
        for (size_t document = 0; document < docs.size(); ++document) {
            classifierModels[document] = &classifierOutput->modelFor(
                ids[document], &heldoutFolds[document]);
        }
        tbb::parallel_for(0, static_cast<int32_t>(docs.size()),
            [&](int32_t document) {
                predictions[static_cast<size_t>(document)] =
                    punkst::partition_classifier::
                        propagate_gamma_poisson_from_composition(
                    *classifierModels[static_cast<size_t>(document)],
                    compositions.row(document).transpose(),
                    docs[static_cast<size_t>(document)], gp.getTopicCapacity(),
                    gp.getBetaAllocationKernel(), gp.getExpectedBeta(),
                    gp.getThetaPriorShape(), priorRate, gp.getSizeFactor(),
                    dispersion, classifierOutput->propagation);
            });
        for (size_t document = 0; document < docs.size(); ++document) {
            classifierOutput->write(ids[document], predictions[document],
                heldoutFolds[document]);
        }
    }

    void finalizeVarianceDiagnostics() {
        auto& state = *residualState;
        auto& diagnostics = state.varianceDiagnostics;
        const double nan = std::numeric_limits<double>::quiet_NaN();
        diagnostics.adjustedTopicSecondMoment.setConstant(nan);
        diagnostics.depthSecondMoment.setConstant(nan);
        diagnostics.excessVarianceExplainedByStructure.setConstant(nan);
        diagnostics.totalVarianceExplainedByStructure.setConstant(nan);

        const MatrixXd topicCross =
            state.topicSecondMoment * state.expectedBeta;
        const MatrixXd rateCross =
            state.rateSecondMoment * state.expectedBeta;
        for (int32_t w = 0; w < M; ++w) {
            const double observed = state.featureTotals(w);
            const double featureWeight = state.featureWeights[
                static_cast<size_t>(w)];
            const double rawObserved = state.rawFeatureTotals(w);
            if (!std::isfinite(observed) || observed < 0.0
                    || !std::isfinite(rawObserved) || rawObserved < 0.0
                    || !std::isfinite(state.exposureTotal)
                    || state.exposureTotal <= 0.0
                    || !std::isfinite(state.exposureSquaredTotal)
                    || state.exposureSquaredTotal < 0.0) {
                continue;
            }
            const double depthQ = rawObserved * rawObserved
                * state.exposureSquaredTotal
                / (state.exposureTotal * state.exposureTotal);
            if (std::isfinite(depthQ) && depthQ >= 0.0) {
                diagnostics.depthSecondMoment(w) = depthQ;
            }
            if (!(featureWeight > 0.0)) continue;
            const double predicted = state.predictedTotals(w) / featureWeight;
            if (!std::isfinite(predicted) || predicted <= 1e-300) {
                continue;
            }
            const double gain = rawObserved / predicted;
            const double topicQ = state.expectedBeta.col(w).dot(
                topicCross.col(w)) / (featureWeight * featureWeight);
            const double adjustedTopicQ = gain * gain * topicQ;
            if (!std::isfinite(adjustedTopicQ) || adjustedTopicQ < 0.0) {
                continue;
            }
            feature_diagnostics::store_variance_decomposition(
                diagnostics, w, adjustedTopicQ, depthQ);

            if (!(adjustedTopicQ > 0.0)
                    || state.positiveExposureUnits <= 0) {
                continue;
            }
            const double rateQ = state.expectedBeta.col(w).dot(
                rateCross.col(w)) / (featureWeight * featureWeight);
            const double adjustedRateQ = gain * gain * rateQ;
            const double adjustedRateMean = gain
                * state.expectedBeta.col(w).dot(state.rateTopicTotals)
                / featureWeight;
            double rateVariance = adjustedRateQ
                - adjustedRateMean * adjustedRateMean
                    / static_cast<double>(state.positiveExposureUnits);
            const bool validRateVariance =
                feature_diagnostics::clamp_tiny_negative(
                    rateVariance, adjustedRateQ);
            const double poissonVariance = gain
                * state.expectedBeta.col(w).dot(
                    state.inverseExposureRateTopicTotals)
                / featureWeight;
            const double phi = feature_diagnostics::positive_raw_dispersion(
                diagnostics.factorialMoment(w), adjustedTopicQ);
            const double denominator = rateVariance + poissonVariance
                + phi * adjustedRateQ;
            if (validRateVariance
                    && std::isfinite(rateVariance) && rateVariance >= 0.0
                    && std::isfinite(poissonVariance)
                    && poissonVariance >= 0.0
                    && std::isfinite(adjustedRateQ)
                    && adjustedRateQ >= 0.0
                    && std::isfinite(denominator) && denominator > 0.0) {
                diagnostics.totalVarianceExplainedByStructure(w) =
                    rateVariance / denominator;
            }
        }
    }

    void processResiduals(const std::vector<Document>& docs,
        const std::vector<std::string>& ids,
        const RowMajorMatrixXd& docTopic,
        const std::vector<GammaPoissonDocumentPosterior>& posteriors,
        const std::vector<std::vector<double>>* rawModelCounts) {
        if (!residualState) return;
        const int32_t nDocs = static_cast<int32_t>(docs.size());
        if (rawModelCounts != nullptr
                && rawModelCounts->size() != docs.size()) {
            error("%s: raw-count sidecar does not match document count",
                __func__);
        }
        if (residualState->requiresRawCountSidecar
                && rawModelCounts == nullptr) {
            error("%s: zero-weight Gamma-Poisson diagnostics require raw counts",
                __func__);
        }
        if (rawModelCounts != nullptr) {
            for (size_t d = 0; d < docs.size(); ++d) {
                if ((*rawModelCounts)[d].size() != docs[d].ids.size()) {
                    error("%s: raw counts do not align with model document",
                        __func__);
                }
            }
        }
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
        RowMajorMatrixXd expectedTopicWeights(nDocs, K);
        RowMajorMatrixXd rateTopicMeans =
            RowMajorMatrixXd::Zero(nDocs, K);
        VectorXd exposures = VectorXd::Zero(nDocs);
        VectorXd inverseExposures = VectorXd::Zero(nDocs);
        VectorXd expectedNormSq;
        const size_t grainSize = std::max<size_t>(
            1, docs.size() / (2 * static_cast<size_t>(threadHint)));
        if (similarityDiagnostics) {
            unitCosine = VectorXd::Zero(nDocs);
            similarityEntropy = computeThetaEntropyStats(
                docTopic, residualState->topicSimilarity);
            unitEntropy = similarityEntropy.entropy;
        }

        tbb::parallel_for(tbb::blocked_range<size_t>(
                0, docs.size(), grainSize),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    const GammaPoissonDocumentPosterior& posterior =
                        posteriors[i];
                    const RowVectorXd theta =
                        (posterior.shape.array()
                            / posterior.rate.array().max(1e-12))
                            .matrix().transpose();
                    expectedTopicWeights.row(static_cast<int32_t>(i)) =
                        posterior.exposure * theta;
                    exposures(static_cast<int32_t>(i)) = posterior.exposure;
                    if (posterior.exposure > 0.0) {
                        rateTopicMeans.row(static_cast<int32_t>(i)) = theta;
                        inverseExposures(static_cast<int32_t>(i)) =
                            1.0 / posterior.exposure;
                    }
                }
            });
        residualState->topicExposureTotals.noalias() +=
            expectedTopicWeights.colwise().sum().transpose();
        residualState->topicSecondMoment.noalias() +=
            expectedTopicWeights.transpose() * expectedTopicWeights;
        residualState->rateTopicTotals.noalias() +=
            rateTopicMeans.colwise().sum().transpose();
        residualState->rateSecondMoment.noalias() +=
            rateTopicMeans.transpose() * rateTopicMeans;
        residualState->inverseExposureRateTopicTotals.noalias() +=
            rateTopicMeans.transpose() * inverseExposures;
        residualState->exposureTotal += exposures.sum();
        residualState->exposureSquaredTotal += exposures.squaredNorm();
        residualState->positiveExposureUnits +=
            (exposures.array() > 0.0).count();
        if (similarityDiagnostics) {
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
                    exposureTheta = expectedTopicWeights.row(
                        static_cast<int32_t>(i)).transpose();
                    if (!similarityDiagnostics) {
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
                        const GammaPoissonDocumentPosterior& posterior =
                            posteriors[d];
                        const double observed = doc.cnts[j];
                        const double rawObserved = rawModelCounts == nullptr
                            ? gp.rawCountFor(w, observed, doc.counts_weighted)
                            : (*rawModelCounts)[d][j];
                        if (!std::isfinite(rawObserved) || rawObserved < 0.0) {
                            error("%s: invalid raw count", __func__);
                        }
                        const double expected =
                            expectedCells[documentOffsets[d] + j];
                        correction +=
                            std::abs(expected - observed) - expected;
                        total += observed;
                        residualState->rawFeatureTotals(w) += rawObserved;
                        residualState->varianceDiagnostics
                            .factorialMoment(w) +=
                                rawObserved * (rawObserved - 1.0);
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
            *unitStats, docs, ids, unitResidual, unitCosine,
            unitEntropy, similarityEntropy.sh_lcr, similarityEntropy.sh_q,
            similarityDiagnostics);
    }

    GammaPoisson4Hex& gp;
    std::ostream& results;
    RowMajorMatrixXd& pseudobulk;
    MatrixXd& specialPseudobulk;
    Mode pseudobulkMode;
    std::ostream* unitStats;
    ResidualState* residualState;
    bool similarityDiagnostics;
    GammaClassifierOutput* classifierOutput;
    punkst::FactorResultStream* factorResults;
    int32_t threadHint;
    std::unique_ptr<ResidualTls> residualTls;
    int32_t M;
    int32_t K;
};

} // namespace

int32_t cmdGammaPoisTransform(int argc, char** argv) {
    std::string inFile, metaFile, stateFile, outPrefix,
        classifierOutPrefix, featureFile, temp_dir, inTransformResults,
        inTransformDispersion;
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
    bool sorted_by_barcode = false;
    bool keep_barcodes = false;
    bool randomize_output = false;
    bool pseudobulk_all_features = false;
    bool compute_residuals = false;
    bool cheap_feature_diagnostics = false;
    bool use_training_prevalence = false;
    bool unit_similarity_diagnostics = false;
    bool full_model = false;
    bool use_stored_dispersion = false;
    std::string dispersion_estimator = "factorial";
    double dispersion_min_information = 8.0;
    double dispersion_outlier_sd = 2.0;
    double dispersion_loess_span = 0.3;
    double dispersion_delta_min = 1e-8;
    double dispersion_delta_max = 1e4;
    double classifier_ambiguity_threshold = 0.95;
    double classifier_candidate_mass = 0.999;
    double classifier_max_failure_rate = 0.01;
    double classifier_fixed_point_tolerance = 1e-7;
    int32_t classifier_top_k = 3;
    int32_t classifier_bootstrap_draws = 64;
    int32_t classifier_fixed_point_max_iterations = 5000;
    bool classifier_dense = false;
    bool classifier_lrvb_all = false;
    bool classifier_plugin_only = false;
    bool classifier_crossfit = false;
    bool classifier_only = false;
    bool factor_is_in_sample = false;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("in-state", "Input Gamma-Poisson state file", stateFile, true)
      .add_option("out-prefix", "Output prefix for results files", outPrefix, true)
      .add_option("out-prefix-classifier", "Output prefix for classification files; defaults to --out-prefix", classifierOutPrefix)
      .add_option("classifier-only", "Classify from prior dense transform results without repeating factor inference", classifier_only)
      .add_option("in-transform-results", "Dense prior transform results used to warm-start classifier-only inference", inTransformResults)
      .add_option("in-transform-dispersion", "Prior transform dispersion diagnostics used by classifier-only inference", inTransformDispersion)
      .add_option("minibatch-size", "Minibatch size", batchSize)
      .add_option("modal", "Modality to use (0-based)", modal)
      .add_option("threads", "Number of threads", nThreads)
      .add_option("temp-dir", "Directory to store temporary files", temp_dir)
      .add_option("seed", "Random seed", seed)
      .add_option("verbose", "Verbose level", verbose)
      .add_option("debug", "If >0, only process this many units", debug_)
      .add_option("randomize-output", "Randomize document output order", randomize_output)
      .add_option("use-stored-dispersion", "Use dispersion from the fitted state instead of estimating it in the transform data", use_stored_dispersion)
      .add_option("factor-is-in-sample", "Input is the factor-fitting sample; enables stored dispersion and training prevalence", factor_is_in_sample);

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
      .add_option("use-training-prevalence", "Treat input as fitted training data for feature diagnostics and dispersion marginal calibration", use_training_prevalence)
      .add_option("unit-diagnostics-similarity", "Add cosine and similarity-adjusted entropy unit diagnostics", unit_similarity_diagnostics);

    pl.add_option("dispersion-estimator", "All-cell transform-data moment estimator: factorial or residual", dispersion_estimator)
      .add_option("dispersion-loess-span", "LOESS span for transform-data dispersion estimation", dispersion_loess_span)
      .add_option("dispersion-min-information", "Minimum adjusted squared-mean information Q for a raw transform-data dispersion estimate", dispersion_min_information)
      .add_option("dispersion-outlier-sd", "Standard-deviation threshold for retaining high-dispersion outliers", dispersion_outlier_sd)
      .add_option("dispersion-delta-min", "Lower bound for estimated NB2 dispersion phi", dispersion_delta_min)
      .add_option("dispersion-delta-max", "Upper bound for estimated NB2 dispersion phi", dispersion_delta_max);

    pl.add_option("classifier-model", "Partition classifier model or crossfit bundle", classifier_model)
      .add_option("classifier-crossfit", "Route classifier-training IDs through held-out fold models", classifier_crossfit)
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
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (factor_is_in_sample) {
        use_stored_dispersion = true;
        use_training_prevalence = true;
        notice("--factor-is-in-sample enabled stored dispersion and training prevalence");
    }
    if (batchSize <= 0) batchSize = 512;
    if (classifierOutPrefix.empty()) classifierOutPrefix = outPrefix;
    if (classifier_only != !inTransformResults.empty()) {
        error("--classifier-only and --in-transform-results must be supplied together");
    }
    if (classifier_only && classifier_model.empty()) {
        error("--classifier-only requires --classifier-model");
    }
    if (!classifier_only && !inTransformDispersion.empty()) {
        error("--in-transform-dispersion requires --classifier-only");
    }
    if (!inTransformDispersion.empty() && use_stored_dispersion) {
        error("--in-transform-dispersion conflicts with stored dispersion options");
    }
    if (classifier_only && !use_stored_dispersion
            && inTransformDispersion.empty()) {
        error("Gamma-Poisson --classifier-only requires either stored dispersion or --in-transform-dispersion");
    }
    if (classifier_only && (compute_residuals || pseudobulk_all_features
            || randomize_output)) {
        error("--classifier-only cannot be combined with residual, pseudobulk-all-features, or randomized factor output options");
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
    if (classifier_crossfit && classifier_model.empty()) {
        error("--classifier-crossfit requires --classifier-model");
    }
    if (seed <= 0) seed = std::random_device{}();
    if (randomize_output && sorted_by_barcode) {
        error("--randomize-output and --sorted-by-barcode are mutually exclusive");
    }
    if (cheap_feature_diagnostics && !compute_residuals) {
        error("--feature-diagnostics-cheap requires --residuals");
    }
    if (use_training_prevalence && cheap_feature_diagnostics) {
        warning("--feature-diagnostics-cheap has no effect with "
            "--use-training-prevalence");
        cheap_feature_diagnostics = false;
    }
    if (unit_similarity_diagnostics && !compute_residuals) {
        error("--unit-diagnostics-similarity requires --residuals");
    }
    if (dispersion_estimator != "factorial" && dispersion_estimator != "residual") {
        error("--dispersion-estimator must be factorial or residual");
    }
    if (!std::isfinite(dispersion_min_information)
        || dispersion_min_information <= 0.0
        || !std::isfinite(dispersion_outlier_sd) || dispersion_outlier_sd < 0.0
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
    if (classifier_only && !inTransformDispersion.empty()) {
        gp.clearFeatureDispersion();
        gp.setFeatureDispersion(read_gamma_poisson_dispersion(
            inTransformDispersion, gp.getFeatureNames()));
        notice("Using transform dispersion from %s",
            inTransformDispersion.c_str());
    } else if (!use_stored_dispersion) {
        GammaPoissonDispersionOptions options;
        options.estimator = dispersion_estimator == "factorial"
            ? GammaPoissonDispersionEstimatorKind::Factorial
            : GammaPoissonDispersionEstimatorKind::Residual;
        options.min_information = dispersion_min_information;
        options.outlier_sd = dispersion_outlier_sd;
        options.loess_span = dispersion_loess_span;
        options.delta_min = dispersion_delta_min;
        options.delta_max = dispersion_delta_max;
        options.adjust_marginal_gain = !use_training_prevalence;
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
    std::ofstream results;
    if (!classifier_only) {
        results.open(resultsPath);
        if (!results) {
            error("Error opening output file: %s for writing", resultsPath.c_str());
        }
        writeUnitIdHeader(results, use_10x, info_header);
        gp.writeUnitHeader(results);
        results << std::scientific << std::setprecision(4);
    }

    std::unique_ptr<GammaClassifierOutput> classifierOutput;
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
        classifierOutput = std::make_unique<GammaClassifierOutput>(classifierOutPrefix,
            classifier_model, propagation, classifier_top_k, classifier_dense,
            classifier_crossfit);
        if (classifierOutput->model.topics != gp.get_topic_names()) {
            error("Classifier topics do not exactly match Gamma-Poisson topic names/order");
        }
        classifierOutput->writeHeader(use_10x, info_header);
    }

    std::unique_ptr<ResidualState> residualState;
    std::unique_ptr<std::ofstream> unitStats;
    if (compute_residuals) {
        residualState =
            std::make_unique<ResidualState>(
                gp, cheap_feature_diagnostics,
                unit_similarity_diagnostics, use_training_prevalence,
                temp_dir);
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
    const bool preserveRawModelCounts = compute_residuals && weights_active;

    std::unique_ptr<punkst::FactorResultStream> factorResults;
    if (classifier_only) {
        factorResults = std::make_unique<punkst::FactorResultStream>(
            inTransformResults, gp.get_topic_names());
    }

    GammaPoisTransformBatchProcessor processor(gp, results, pseudobulk,
        specialPseudobulk, pseudobulkMode,
        unitStats.get(), residualState.get(),
        unit_similarity_diagnostics, nThreads, classifierOutput.get(),
        factorResults.get());
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
                SpecialBatch allSpecial;
                for (size_t i = 0; i < allRawDocs.size(); ++i) {
                    transform_pseudobulk::appendDocument(
                        std::move(allRawDocs[i]),
                        dge.getUnitId(allUnitIndices[i]), allSpecial,
                        pseudobulkMode, inputToModel, gp, minCountInt,
                        preserveRawModelCounts);
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
                        batchSize, remaining, minCountInt,
                        preserveRawModelCounts);
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
                        batchSize, remaining, minCountInt,
                        preserveRawModelCounts);
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
    if (factorResults) {
        factorResults->require_finished(debug_ > 0);
    }
    if (!classifier_only) results.close();
    if (unitStats) {
        unitStats->close();
        notice("Per-unit residuals written to %s",
            (outPrefix + ".unit_stats.tsv").c_str());
    }
    if (!classifier_only) {
        notice("Transformation results written to %s", resultsPath.c_str());
    }
    bool classifierFailure = false;
    if (classifierOutput) {
        classifierOutput->stream.close();
        notice("Partition classifications written to %s",
            classifierOutput->path.c_str());
        const double failureRate = classifierOutput->attempted > 0
            ? static_cast<double>(classifierOutput->failed)
                / classifierOutput->attempted : 0.0;
        notice("Gamma-Poisson classifier LRVB attempted %zu units; %zu failed (%.6g)",
            classifierOutput->attempted, classifierOutput->failed, failureRate);
        if (classifierOutput->useCrossfit
                && classifierOutput->heldoutModelRows == 0) {
            warning("--classifier-crossfit matched no classifier-training IDs; "
                "all rows used the full model");
        }
        const std::string diagnosticPath =
            classifierOutPrefix + ".classification_diagnostics.tsv";
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
            "\tmax_cg_iterations\tmax_curvature_jitter"
            "\tclassifier_prediction_mode\theldout_model_rows"
            "\tfull_model_rows\n"
            << classifierOutput->attempted << '\t' << classifierOutput->failed
            << '\t' << std::scientific << std::setprecision(4)
            << failureRate << '\t' << classifier_max_failure_rate << '\t'
            << classifierOutput->localNonconvergence << '\t'
            << classifierOutput->curvatureFailures << '\t'
            << classifierOutput->otherFailures << '\t'
            << meanFixedPointIterations << '\t'
            << classifierOutput->maximumFixedPointIterations << '\t'
            << classifierOutput->maximumFixedPointResidual << '\t'
            << meanCgIterations << '\t'
            << classifierOutput->maximumCgIterations << '\t'
            << classifierOutput->maximumCurvatureJitter << '\t'
            << (classifierOutput->useCrossfit ? "crossfit" : "full") << '\t'
            << classifierOutput->heldoutModelRows << '\t'
            << classifierOutput->fullModelRows << '\n';
        classifierFailure = failureRate > classifier_max_failure_rate;
    }

    if (classifier_only) return classifierFailure ? 1 : 0;

    const std::string pseudobulkPath = outPrefix + ".pseudobulk.tsv";
    std::ofstream pseudobulkOut(pseudobulkPath);
    if (!pseudobulkOut) {
        error("Error opening output file: %s for writing", pseudobulkPath.c_str());
    }
    pseudobulkOut << "Feature\t";
    gp.writeModelHeader(pseudobulkOut);
    transform_helpers::writePseudobulkRows(
        pseudobulkOut, pseudobulkFeatureNames, pseudobulk,
        specialPseudobulk, pseudobulkMode, K);
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
        feature_diagnostics::write_feature_residuals(
            featureResidualOut, modelFeatures, *residualState,
            &residualState->varianceDiagnostics);
        featureResidualOut.close();
        notice("Per-feature residuals written to %s",
            featureResidualPath.c_str());
    }
    return classifierFailure ? 1 : 0;
}
