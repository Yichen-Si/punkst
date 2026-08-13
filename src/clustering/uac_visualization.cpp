#include "clustering/uac.hpp"
#include "clustering/uac_common_internal.hpp"

namespace uac {

VisualizationWhitening parse_visualization_whitening(
        const std::string& value) {
    if (value == "sample") return VisualizationWhitening::Sample;
    if (value == "mixture") return VisualizationWhitening::Mixture;
    throw std::invalid_argument(
        "UAC visualization whitening must be sample or mixture");
}

VisualizationResult make_visualization(const Dataset& data,
    const Model& model, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options) {
    detail::validate_model(model);
    VisualizationMoments moments;
    moments.weights = model.weights;
    moments.means = model.means;
    moments.covariances.reserve(model.weights.size());
    for (int32_t component = 0; component < model.weights.size();
            ++component) {
        moments.covariances.push_back(
            detail::model_covariance_dense(model, component));
    }
    return punkst::projection::make_visualization(
        data.coordinates, moments, helmert, options);
}

VisualizationResult make_visualization(const Dataset& data,
    const VisualizationMoments& moments,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options) {
    return punkst::projection::make_visualization(
        data.coordinates, moments, helmert, options);
}

VisualizationResult make_visualization(const Dataset& data,
    const VisualizationMoments& moments,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const VisualizationOptions& options,
    const VisualizationSampleMoments& sample_moments) {
    return punkst::projection::make_visualization(
        data.coordinates, moments, helmert, options, sample_moments);
}

} // namespace uac
