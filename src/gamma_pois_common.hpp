#pragma once

#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "gamma_pois_dispersion.hpp"
#include "topic_svb.hpp"

struct GammaPoissonStateFeatureInfo {
    std::vector<std::string> names;
    std::vector<double> training_count;
    bool feature_weights_active = false;
    std::vector<double> feature_weight;
};

class GammaPoisson4HexInterface : public TopicModelWrapper {
public:
    GammaPoisson4HexInterface(HexReader& reader, int32_t modal = 0,
        int32_t verbose = 0) : TopicModelWrapper(reader, modal, verbose) {}

    virtual void setFeatureDispersion(const std::vector<double>& tau) = 0;
    virtual void clearFeatureDispersion() = 0;
    virtual GammaPoissonDispersionResult estimateFeatureDispersion(
        const GammaPoissonDispersionOptions& options, const std::string& inFile,
        int32_t batchSize, int32_t minCountTrain, int32_t maxUnits) = 0;
    virtual GammaPoissonDispersionResult estimateFeatureDispersion(
        const GammaPoissonDispersionOptions& options,
        punkst::DocumentBlockSource& source,
        int32_t batchSize, int32_t maxUnits) = 0;
    virtual GammaPoissonDispersionResult estimateFeatureDispersion(
        const GammaPoissonDispersionOptions& options,
        const std::vector<std::vector<Document>>& batches,
        int32_t maxUnits) = 0;
    virtual GammaPoissonDispersionResult estimateFeatureDispersion10X(
        const GammaPoissonDispersionOptions& options, int32_t batchSize,
        int32_t maxUnits) = 0;
    virtual GammaPoissonDispersionResult estimateFeatureDispersion10X(
        const GammaPoissonDispersionOptions& options, DGEReader10X& dge,
        int32_t batchSize, int32_t minCount, int32_t maxUnits) = 0;
    virtual RowMajorMatrixXd transformMeans(DocumentView batch) const = 0;
    virtual void transformWithPosteriors(DocumentView batch,
        RowMajorMatrixXd& topics,
        std::vector<GammaPoissonDocumentPosterior>& posteriors) const = 0;
    virtual bool hasFeatureDispersion() const = 0;
    virtual const VectorXd& getTopicCapacity() const = 0;
    virtual const MatrixXd& getExpectedBeta() const = 0;
    virtual const MatrixXd& getBetaAllocationKernel() const = 0;
    virtual void normalizeTopicAllocation(
        const Eigen::Ref<const VectorXd>& theta_log,
        int32_t feature, VectorXd& allocation) const {
        const MatrixXd& kernel = getBetaAllocationKernel();
        if (theta_log.size() != kernel.rows()
                || feature < 0 || feature >= kernel.cols()) {
            throw std::invalid_argument(
                "Gamma-Poisson topic allocation dimensions do not match");
        }
        const double max_log = theta_log.maxCoeff();
        allocation = (theta_log.array() - max_log).exp()
            * kernel.col(feature).array();
        const double total = allocation.sum();
        if (!std::isfinite(total) || total <= 0.0) {
            throw std::runtime_error(
                "Gamma-Poisson topic allocation is not positive and finite");
        }
        allocation /= total;
    }
    virtual const VectorXd& getFeatureDispersion() const = 0;
    virtual double getSizeFactor() const = 0;
    virtual double getThetaPriorShape() const = 0;
    virtual VectorXd getThetaPriorRate() const = 0;
    virtual bool featureWeightsActive() const = 0;
    virtual const std::vector<double>& getFeatureWeights() const = 0;
    virtual void get_topic_abundance(std::vector<double>& topic_weights) = 0;
};
