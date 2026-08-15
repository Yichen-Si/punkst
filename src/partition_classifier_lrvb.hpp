#pragma once

#include "dataunits.hpp"
#include "partition_classifier.hpp"

#include <limits>
#include <string>

struct GammaPoissonDocumentPosterior;

namespace punkst::partition_classifier {

struct PropagationOptions {
    double ambiguity_threshold = 0.95;
    double candidate_mass = 0.999;
    bool lrvb_all = false;
    bool plugin_only = false;
    double fixed_point_tolerance = 1e-7;
    int32_t fixed_point_max_iterations = 5000;
    double cg_tolerance = 1e-7;
};

struct PropagatedPrediction {
    Eigen::VectorXd probabilities;
    std::string method = "plugin";
    std::string lrvb_status = "not_attempted";
    int32_t candidate_count = 0;
    int32_t fixed_point_iterations = 0;
    int32_t cg_iterations = 0;
    double fixed_point_residual =
        std::numeric_limits<double>::quiet_NaN();
    double curvature_jitter = 0.0;
    double held_fixed_tail_mass = 0.0;
    bool lrvb_attempted = false;
    bool lrvb_failed = false;
};

PropagatedPrediction propagate_lda(const Model& classifier,
    const Eigen::Ref<const Eigen::VectorXd>& assigned_counts,
    const Document& document,
    const Eigen::Ref<const Eigen::MatrixXd>& lda_allocation_kernel,
    double alpha, const PropagationOptions& options = {});

PropagatedPrediction propagate_lda_from_composition(
    const Model& classifier,
    const Eigen::Ref<const Eigen::VectorXd>& composition,
    const Document& document,
    const Eigen::Ref<const Eigen::MatrixXd>& lda_allocation_kernel,
    double alpha, const PropagationOptions& options = {});

PropagatedPrediction propagate_gamma_poisson(const Model& classifier,
    const GammaPoissonDocumentPosterior& posterior,
    const Document& document,
    const Eigen::Ref<const Eigen::VectorXd>& topic_capacity,
    const Eigen::MatrixXd& beta_allocation_kernel,
    const Eigen::MatrixXd& expected_beta,
    double prior_shape,
    const Eigen::Ref<const Eigen::VectorXd>& prior_rate,
    const Eigen::VectorXd* feature_dispersion,
    const PropagationOptions& options = {},
    const Eigen::VectorXd* initial_composition = nullptr);

PropagatedPrediction propagate_gamma_poisson_from_composition(
    const Model& classifier,
    const Eigen::Ref<const Eigen::VectorXd>& composition,
    const Document& document,
    const Eigen::Ref<const Eigen::VectorXd>& topic_capacity,
    const Eigen::MatrixXd& beta_allocation_kernel,
    const Eigen::MatrixXd& expected_beta,
    double prior_shape,
    const Eigen::Ref<const Eigen::VectorXd>& prior_rate,
    double size_factor,
    const Eigen::VectorXd* feature_dispersion,
    const PropagationOptions& options = {});

namespace testing {

void run_lrvb_numerical_tests();

} // namespace testing

} // namespace punkst::partition_classifier
