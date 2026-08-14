#pragma once

#include "numerical_utils.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace punkst::projection {

enum class VisualizationWhitening {
    Sample,
    Mixture,
};

enum class VisualizationView {
    Mean,
    Full,
};

struct VisualizationOptions {
    VisualizationWhitening whitening = VisualizationWhitening::Mixture;
    int32_t dimensions = 2;
    int32_t n_threads = 1;
    double covariance_floor = 1e-5;
    bool include_full = false;
};

struct VisualizationProjection {
    VisualizationView view = VisualizationView::Mean;
    Eigen::VectorXd eigenvalues;
    Eigen::VectorXd axis_scores;
    Eigen::MatrixXd projection;
    Eigen::MatrixXd topic_contrasts;
    RowMajorMatrixXd component_means;
    std::vector<Eigen::MatrixXd> component_covariances;
};

struct VisualizationResult {
    VisualizationWhitening whitening = VisualizationWhitening::Mixture;
    Eigen::MatrixXd whitening_covariance;
    VisualizationProjection mean;
    VisualizationProjection full;
};

struct VisualizationMoments {
    Eigen::VectorXd weights;
    RowMajorMatrixXd means;
    std::vector<Eigen::MatrixXd> covariances;
};

struct VisualizationMeans {
    Eigen::VectorXd weights;
    RowMajorMatrixXd means;
};

struct VisualizationSampleMoments {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
};

const char* visualization_whitening_name(VisualizationWhitening value);
const char* visualization_view_name(VisualizationView value);
VisualizationWhitening parse_visualization_whitening(
    const std::string& value);

double quartimax_objective(
    const Eigen::Ref<const Eigen::MatrixXd>& loadings);
void quartimax_rotate(Eigen::MatrixXd& basis, Eigen::MatrixXd& loadings);
void quartimax_rotate(VisualizationProjection& projection);

VisualizationResult make_visualization(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const VisualizationMoments& moments,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options = {});
VisualizationResult make_visualization(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const VisualizationMoments& moments,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options,
    const VisualizationSampleMoments& sample_moments);
VisualizationResult make_mean_visualization(
    const VisualizationMeans& means,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options,
    const VisualizationSampleMoments& sample_moments);
VisualizationMeans summarize_hard_partition_means(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components);
VisualizationMoments summarize_hard_partition(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components, int32_t n_threads = 1);
VisualizationSampleMoments summarize_visualization_sample(
    const Eigen::Ref<const RowMajorMatrixXd>& coordinates,
    int32_t n_threads = 1);

void write_visualization_axes(const std::string& path,
    const std::vector<std::string>& topics,
    const VisualizationResult& visualization);

} // namespace punkst::projection
