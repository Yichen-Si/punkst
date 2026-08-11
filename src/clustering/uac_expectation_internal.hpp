#pragma once

#include "clustering/uac_particles_internal.hpp"
#include "clustering/uac_screening_internal.hpp"

#include <cstdint>
#include <limits>
#include <vector>

namespace uac::detail {


struct Expectation {
    int32_t documents = 0;
    RowMajorMatrixXd responsibilities;
    Eigen::VectorXd membership;
    Eigen::VectorXd membership_weight_squared;
    RowMajorMatrixXd first;
    std::vector<Eigen::MatrixXd> second;
    RowMajorMatrixXd sum_y2;
    RowMajorMatrixXd sum_f;
    std::vector<Eigen::MatrixXd> sum_ff;
    std::vector<Eigen::MatrixXd> sum_yf;
    std::vector<ParticleDiagnostic> particle_diagnostics;
    double log_likelihood = 0.0;
    double log_likelihood_upper = 0.0;
    double responsibility_entropy_sum = 0.0;
    double gaussian_seconds = 0.0;
    double component_bound_seconds = 0.0;
    double moment_seconds = 0.0;
    uint64_t peak_workspace_bytes = 0;
    uint64_t peak_particle_bytes = 0;
    int32_t parallel_workers = 0;
    int64_t evaluated_component_documents = 0;
    int64_t possible_component_documents = 0;
    int32_t full_component_documents = 0;
    int32_t component_bound_violations = 0;
    double omitted_component_mass_sum = 0.0;
    double maximum_omitted_component_mass = 0.0;
    double mean_max_responsibility_change =
        std::numeric_limits<double>::quiet_NaN();
    bool has_responsibility_change = false;
    std::vector<int32_t> per_document_evaluated_components;
    std::vector<double> per_document_omitted_component_mass;
    std::vector<uint16_t> subsample_strata;
    std::vector<int32_t> subsample_document_samples;
    Eigen::VectorXi subsample_stratum_documents;
    Eigen::VectorXd subsample_stratum_purity;
    Eigen::MatrixXd subsample_transfer;
    Eigen::VectorXd subsample_stratum_bytes;
};

struct ExpectationRequest {
    bool store_responsibilities = false;
    bool collect_diagnostics = false;
    bool accumulate_moments = true;
    bool collect_subsample_statistics = false;
};

struct ExpectationBlock {
    Eigen::VectorXd membership;
    Eigen::VectorXd membership_weight_squared;
    RowMajorMatrixXd first;
    std::vector<Eigen::MatrixXd> second;
    RowMajorMatrixXd sum_y2;
    RowMajorMatrixXd sum_f;
    std::vector<Eigen::MatrixXd> sum_ff;
    std::vector<Eigen::MatrixXd> sum_yf;
    double log_likelihood = 0.0;
    double log_likelihood_upper = 0.0;
    double responsibility_entropy_sum = 0.0;
    double component_bound_seconds = 0.0;
    int64_t evaluated_component_documents = 0;
    int64_t possible_component_documents = 0;
    int32_t full_component_documents = 0;
    int32_t component_bound_violations = 0;
    double omitted_component_mass_sum = 0.0;
    double maximum_omitted_component_mass = 0.0;
    Eigen::VectorXi subsample_stratum_documents;
    Eigen::VectorXd subsample_stratum_purity;
    Eigen::MatrixXd subsample_transfer;
    Eigen::VectorXd subsample_stratum_bytes;

    ExpectationBlock(int32_t components, int32_t dimension,
        int32_t factor_rank = -1, bool accumulate_moments = true,
        bool weighted_documents = false,
        bool collect_subsample_statistics = false) {
        if (collect_subsample_statistics) {
            subsample_stratum_documents = Eigen::VectorXi::Zero(components);
            subsample_stratum_purity = Eigen::VectorXd::Zero(components);
            subsample_transfer = Eigen::MatrixXd::Zero(components, components);
            subsample_stratum_bytes = Eigen::VectorXd::Zero(components);
        }
        if (!accumulate_moments) return;
        membership = Eigen::VectorXd::Zero(components);
        if (weighted_documents) {
            membership_weight_squared = Eigen::VectorXd::Zero(components);
        }
        first = RowMajorMatrixXd::Zero(components, dimension);
        if (factor_rank < 0) {
            second.assign(components,
                Eigen::MatrixXd::Zero(dimension, dimension));
        } else {
            sum_y2 = RowMajorMatrixXd::Zero(components, dimension);
            sum_f = RowMajorMatrixXd::Zero(components, factor_rank);
            sum_ff.assign(components,
                Eigen::MatrixXd::Zero(factor_rank, factor_rank));
            sum_yf.assign(components,
                Eigen::MatrixXd::Zero(dimension, factor_rank));
        }
    }
};

uint64_t expectation_block_bytes(
    int32_t components, int32_t dimension, int32_t factor_rank);
uint64_t particle_expectation_peak_bytes(int32_t documents,
    int32_t components, int32_t dimension, int32_t factor_rank,
    int32_t maximum_samples, bool screen);
int32_t expectation_shards(int32_t documents, int32_t components,
    int32_t dimension, int32_t factor_rank);
void reduce_expectation_blocks(
    Expectation& out, const std::vector<ExpectationBlock>& blocks);
Expectation empty_expectation(int32_t documents, int32_t components,
    int32_t dimension, int32_t factor_rank = -1,
    bool accumulate_moments = true);
void accumulate_expectation(Expectation& target, const Expectation& source);
Expectation map_expectation(const Dataset& data, const Model& model,
    const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {});
bool resolve_map_component_screening(
    const Dataset& data, const Model& model,
    const ComponentScreeningOptions& requested, uint64_t seed);
Expectation particle_expectation(const ParticleSet& particles,
    const Model& model, const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {},
    const Eigen::VectorXd* document_weights = nullptr);
Expectation particle_expectation(const RaggedParticleSet& particles,
    const Model& model, const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {},
    const Eigen::VectorXd* document_weights = nullptr);
Expectation particle_expectation(const IndexedFixedParticleView& particles,
    const Model& model, const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {},
    const Eigen::VectorXd* document_weights = nullptr);
Expectation particle_expectation(const IndexedRaggedParticleView& particles,
    const Model& model, const ExpectationRequest& request = {},
    const ComponentScreeningOptions& screening = {},
    const Eigen::VectorXd* document_weights = nullptr);
Expectation particle_expectation_into(const ParticleSet& particles,
    const Model& model, const ExpectationRequest& request,
    const ComponentScreeningOptions& screening, ExpectationBlock& block);
Expectation particle_expectation_into(const RaggedParticleSet& particles,
    const Model& model, const ExpectationRequest& request,
    const ComponentScreeningOptions& screening, ExpectationBlock& block);
bool resolve_particle_component_screening(
    const ParticleSet& particles, const Model& model,
    const ComponentScreeningOptions& requested,
    const std::vector<int32_t>& audit_documents);
bool resolve_particle_component_screening(
    const RaggedParticleSet& particles, const Model& model,
    const ComponentScreeningOptions& requested,
    const std::vector<int32_t>& audit_documents);

} // namespace uac::detail
