#pragma once

#include "clustering/uac.hpp"
#include "clustering/uac_stream.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace uac::detail {

inline constexpr double kLog2Pi = 1.83787706640934548356;

double logsumexp(const Eigen::Ref<const Eigen::VectorXd>& values);
double logaddexp(double left, double right);

void validate_component_screening(
    const ComponentScreeningOptions& options);
int32_t checked_int32(Eigen::Index value, const char* name);
bool positive_definite(const Eigen::MatrixXd& covariance);
void validate_dataset(const Dataset& data, bool require_counts);
void validate_basis(const Basis& basis, int32_t topics,
    bool require_checksum = true);
void validate_count_features(const Dataset& data, const Basis& basis);
void validate_model(const Model& model);
void validate_pilot(const Pilot& pilot, int32_t components,
    int32_t dimension);
void validate_adaptive_particles(const AdaptiveParticleOptions& options,
    int32_t maximum_particles);
void validate_state(const State& state);

DocumentBlock read_aligned_document_range(const Dataset& data,
    const IndexedDocumentSource& source, int32_t first, int32_t count);
bool has_nonidentity_feature_weights(const State& state);
void prepare_particle_score_counts(Dataset& data, const State& state);

class ValidatingIndexedDocumentSource final
    : public IndexedDocumentSource {
public:
    ValidatingIndexedDocumentSource(IndexedDocumentSource& source,
        bool require_weighted);

    int64_t documents() const override;
    int64_t features() const override;
    uint64_t storage_bytes() const override;
    uint64_t content_checksum() const override;
    uint64_t peak_block_bytes() const override;
    void reset() override;
    bool next(DocumentBlock& block, int32_t maximum_documents) override;
    void read_range(int64_t first_document, int32_t documents,
        DocumentBlock& block) const override;

private:
    void validate_weighting(const DocumentBlock& block) const;

    IndexedDocumentSource& source_;
    bool require_weighted_ = false;
};

double weighted_hpd_threshold(
    const Eigen::Ref<const Eigen::VectorXd>& log_density,
    const Eigen::Ref<const Eigen::VectorXd>& probability, double level);
uint64_t fnv_append(uint64_t value, const void* data, size_t size);
uint64_t hash_string(uint64_t value, const std::string& text);
Eigen::MatrixXd floor_covariance(
    const Eigen::Ref<const Eigen::MatrixXd>& input, double floor);
double log_gaussian(const Eigen::Ref<const Eigen::VectorXd>& value,
    const Eigen::Ref<const Eigen::VectorXd>& mean,
    const Eigen::Ref<const Eigen::MatrixXd>& covariance);

struct DenseGaussianSolver {
    Eigen::VectorXd mean;
    Eigen::MatrixXd lower;
    double log_determinant = 0.0;

    DenseGaussianSolver() = default;
    DenseGaussianSolver(const Eigen::Ref<const Eigen::VectorXd>& input_mean,
        const Eigen::Ref<const Eigen::MatrixXd>& covariance);
    double log_density(
        const Eigen::Ref<const Eigen::VectorXd>& value) const;
    Eigen::VectorXd log_density_rows(
        const Eigen::Ref<const RowMajorMatrixXd>& values) const;
};

std::vector<DenseGaussianSolver> dense_model_solvers(const Model& model);
Eigen::MatrixXd model_covariance_dense(
    const Model& model, int32_t component);
void validate_particle_initial_model(
    const Model& model, const Model& reference);
std::vector<double> model_eigenvalue_upper_bounds(const Model& model);
LowRankDiagonalCovariance factorize_covariance(
    const Eigen::Ref<const Eigen::MatrixXd>& covariance, int32_t rank,
    double floor);
double covariance_prior(const Model& model, double strength);
int32_t active_component_count(const Model& model);
double membership_epsilon(int32_t documents);
int32_t map_start_seed(int32_t seed, int32_t start);
Eigen::VectorXd composition_from_coordinate(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert);
Eigen::VectorXd count_log_likelihood_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert);

void apply_auto_component_screening_resolution(
    ComponentScreeningOptions& options, bool enabled);
const char* adaptive_particle_mode_name(
    const AdaptiveParticleOptions& options);
double optional_target_or_zero(const std::optional<double>& value);

double increased_leiden_resolution(double resolution, int32_t raw_communities,
    int32_t requested_communities);
double midpoint_leiden_resolution(double lower, double upper);
void prepare_counts(std::vector<Document>& documents, int32_t feature_count,
    const Eigen::VectorXd* feature_weights, Eigen::VectorXd& raw_totals,
    Eigen::VectorXd& effective_totals);

} // namespace uac::detail
