#include "clustering/gamma_pois_cluster_internal.hpp"

#include "nanoflann.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace gamma_pois_cluster_detail {
namespace {

using Score = std::pair<double, int32_t>;

bool better_score(const Score& first, const Score& second) {
    return first.first > second.first
        || (first.first == second.first && first.second < second.second);
}

void document_marginal_variance(
    const GammaPoissonClusterCoordinates& coordinates, int32_t document,
    int32_t used_dimensions, Eigen::Ref<Eigen::VectorXd> out) {
    const auto factor = coordinates.factor(document);
    for (int32_t j = 0; j < used_dimensions; ++j) {
        out(j) = coordinates.uncertainty_diagonal(document, j);
        if (coordinates.uncertainty_rank > 0) {
            out(j) += factor.row(j).squaredNorm();
        }
    }
}

double proxy_score(const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model,
    const PreparedEStepModel& prepared_model, int32_t document,
    int32_t component, int32_t used_dimensions,
    const Eigen::Ref<const Eigen::VectorXd>& document_variance) {
    double score = std::log(std::max(
        prepared_model.weights(component), 1e-300));
    for (int32_t j = 0; j < used_dimensions; ++j) {
        const double variance = document_variance(j)
            + prepared_model.marginal_variances(component, j);
        const double residual = coordinates.mean(document, j)
            - model.means(component, j);
        score -= 0.5 * (std::log(variance)
            + residual * residual / variance);
    }
    return score;
}

int32_t retain_previous_top(const std::vector<int32_t>& previous_top,
    const std::vector<uint8_t>& dormant, int32_t document,
    int32_t candidate_count, std::vector<uint8_t>& used, int32_t* out) {
    int32_t filled = 0;
    for (int32_t top = 0; top < 3 && filled < candidate_count; ++top) {
        const int32_t component = previous_top[
            static_cast<size_t>(document) * 3 + top];
        if (component >= 0 && component < static_cast<int32_t>(used.size())
            && !dormant[component] && !used[component]) {
            out[filled++] = component;
            used[component] = 1;
        }
    }
    return filled;
}

void append_best_scores(std::vector<Score>& scores, int32_t needed,
    int32_t& filled, int32_t* out) {
    needed = std::min<int32_t>(needed, scores.size());
    if (needed < static_cast<int32_t>(scores.size())) {
        std::nth_element(scores.begin(), scores.begin() + needed,
            scores.end(), better_score);
        scores.resize(needed);
    }
    std::sort(scores.begin(), scores.end(), better_score);
    for (const Score& score : scores) out[filled++] = score.second;
}

} // namespace

std::vector<int32_t> select_candidate_components_linear(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model,
    const PreparedEStepModel& prepared_model, const int32_t* documents,
    int32_t n_documents, int32_t candidate_count, int32_t candidate_dimensions,
    const std::vector<int32_t>& previous_top,
    const std::vector<uint8_t>& dormant) {
    const int32_t components = static_cast<int32_t>(model.means.rows());
    const int32_t dim = static_cast<int32_t>(model.means.cols());
    const int32_t used_dimensions = candidate_dimensions == 0
        ? dim : candidate_dimensions;
    std::vector<int32_t> out(
        static_cast<size_t>(n_documents) * candidate_count, -1);
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, n_documents),
        [&](const tbb::blocked_range<int32_t>& range) {
            std::vector<uint8_t> used(components);
            std::vector<Score> scores;
            scores.reserve(components);
            Eigen::VectorXd document_variance(used_dimensions);
            for (int32_t position = range.begin(); position < range.end();
                 ++position) {
                const int32_t document = documents[position];
                std::fill(used.begin(), used.end(), uint8_t{0});
                int32_t* selected = out.data()
                    + static_cast<size_t>(position) * candidate_count;
                int32_t filled = retain_previous_top(previous_top, dormant,
                    document, candidate_count, used, selected);
                document_marginal_variance(coordinates, document,
                    used_dimensions, document_variance);
                scores.clear();
                for (int32_t component = 0; component < components;
                     ++component) {
                    if (dormant[component] || used[component]) continue;
                    scores.emplace_back(proxy_score(coordinates, model,
                        prepared_model, document, component, used_dimensions,
                        document_variance), component);
                }
                append_best_scores(scores, candidate_count - filled,
                    filled, selected);
                if (filled == 0) {
                    throw std::runtime_error(
                        "No active Gamma-Poisson candidate components remain");
                }
            }
        });
    return out;
}

std::vector<int32_t> select_candidate_components_kdtree(
    const GammaPoissonClusterCoordinates& coordinates,
    const GammaPoissonClusterModel& model,
    const PreparedEStepModel& prepared_model, const int32_t* documents,
    int32_t n_documents, int32_t candidate_count, int32_t candidate_dimensions,
    int32_t pool_multiplier, const std::vector<int32_t>& previous_top,
    const std::vector<uint8_t>& dormant) {
    const int32_t components = static_cast<int32_t>(model.means.rows());
    const int32_t dim = static_cast<int32_t>(model.means.cols());
    const int32_t used_dimensions = candidate_dimensions == 0
        ? dim : candidate_dimensions;
    if (pool_multiplier <= 0) {
        throw std::invalid_argument("Candidate pool multiplier must be positive");
    }
    std::vector<int32_t> active;
    for (int32_t component = 0; component < components; ++component) {
        if (!dormant[component]) active.push_back(component);
    }
    if (active.empty()) {
        throw std::runtime_error(
            "No active Gamma-Poisson candidate components remain");
    }

    Eigen::VectorXd scale = Eigen::VectorXd::Zero(used_dimensions);
    Eigen::VectorXd document_variance(used_dimensions);
    for (int32_t position = 0; position < n_documents; ++position) {
        document_marginal_variance(coordinates, documents[position],
            used_dimensions, document_variance);
        scale += document_variance;
    }
    scale /= static_cast<double>(n_documents);
    for (int32_t component : active) {
        scale.noalias() += prepared_model.weights(component)
            * prepared_model.marginal_variances.row(component)
                .head(used_dimensions).transpose();
    }
    scale = scale.array().max(1e-12).sqrt().inverse();

    RowMajorMatrixXd centers(active.size(), used_dimensions);
    for (size_t row = 0; row < active.size(); ++row) {
        centers.row(row) = model.means.row(active[row]).head(used_dimensions)
            .array() * scale.transpose().array();
    }
    using Index = nanoflann::KDTreeEigenMatrixAdaptor<
        RowMajorMatrixXd, -1, nanoflann::metric_L2, true>;
    Index index(used_dimensions, std::cref(centers), 10);
    const int32_t pool_size = std::min<int32_t>(active.size(),
        std::max(candidate_count, pool_multiplier * candidate_count));

    std::vector<int32_t> out(
        static_cast<size_t>(n_documents) * candidate_count, -1);
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, n_documents),
        [&](const tbb::blocked_range<int32_t>& range) {
            std::vector<uint8_t> used(components);
            std::vector<Score> scores;
            scores.reserve(pool_size + 3);
            Eigen::VectorXd query(used_dimensions);
            Eigen::VectorXd local_document_variance(used_dimensions);
            std::vector<Eigen::Index> neighbors(pool_size);
            std::vector<double> distances(pool_size);
            for (int32_t position = range.begin(); position < range.end();
                 ++position) {
                const int32_t document = documents[position];
                std::fill(used.begin(), used.end(), uint8_t{0});
                int32_t* selected = out.data()
                    + static_cast<size_t>(position) * candidate_count;
                int32_t filled = retain_previous_top(previous_top, dormant,
                    document, candidate_count, used, selected);
                query = coordinates.mean.row(document).head(used_dimensions)
                    .transpose().array() * scale.array();
                index.query(query.data(), pool_size,
                    neighbors.data(), distances.data());
                document_marginal_variance(coordinates, document,
                    used_dimensions, local_document_variance);
                scores.clear();
                for (Eigen::Index neighbor : neighbors) {
                    const int32_t component = active[neighbor];
                    if (used[component]) continue;
                    used[component] = 1;
                    scores.emplace_back(proxy_score(coordinates, model,
                        prepared_model, document, component, used_dimensions,
                        local_document_variance), component);
                }
                append_best_scores(scores, candidate_count - filled,
                    filled, selected);
                if (filled == 0) {
                    throw std::runtime_error(
                        "No active Gamma-Poisson candidate components remain");
                }
            }
        });
    return out;
}

} // namespace gamma_pois_cluster_detail
