#include "dense_kmeans.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace {

void assign_to_centers(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const Eigen::Ref<const RowMajorMatrixXd>& centers,
    Eigen::VectorXi& assignments, Eigen::VectorXd& distances,
    Eigen::VectorXi& counts, double& inertia) {
    const Eigen::Index n = observations.rows();
    assignments.resize(n);
    distances.resize(n);
    counts = Eigen::VectorXi::Zero(centers.rows());
    inertia = 0.0;
    for (Eigen::Index d = 0; d < n; ++d) {
        Eigen::Index best = 0;
        double best_distance = (observations.row(d) - centers.row(0)).squaredNorm();
        for (Eigen::Index c = 1; c < centers.rows(); ++c) {
            const double distance = (observations.row(d) - centers.row(c)).squaredNorm();
            if (distance < best_distance) {
                best = c;
                best_distance = distance;
            }
        }
        assignments(d) = static_cast<int32_t>(best);
        distances(d) = best_distance;
        ++counts(best);
        inertia += best_distance;
    }
}

} // namespace

DenseKMeansResult dense_kmeans(
    const Eigen::Ref<const RowMajorMatrixXd>& observations,
    const DenseKMeansOptions& options) {
    if (observations.rows() == 0 || observations.cols() == 0
        || !observations.allFinite() || options.n_clusters <= 0
        || options.n_clusters > observations.rows()
        || options.max_iterations <= 0) {
        throw std::invalid_argument("Invalid dense k-means input or options");
    }

    const Eigen::Index n = observations.rows();
    const Eigen::Index dim = observations.cols();
    const int32_t clusters = options.n_clusters;
    DenseKMeansResult out;
    out.centers.resize(clusters, dim);

    std::mt19937 random_engine(static_cast<uint32_t>(options.seed));
    std::uniform_int_distribution<Eigen::Index> first_distribution(0, n - 1);
    std::vector<Eigen::Index> seeds;
    seeds.reserve(clusters);
    seeds.push_back(first_distribution(random_engine));
    Eigen::VectorXd min_distance = Eigen::VectorXd::Constant(
        n, std::numeric_limits<double>::infinity());
    while (static_cast<int32_t>(seeds.size()) < clusters) {
        const Eigen::Index latest = seeds.back();
        for (Eigen::Index d = 0; d < n; ++d) {
            min_distance(d) = std::min(min_distance(d),
                (observations.row(d) - observations.row(latest)).squaredNorm());
        }
        const double total = min_distance.sum();
        Eigen::Index selected = -1;
        if (total > 0.0 && std::isfinite(total)) {
            std::uniform_real_distribution<double> draw(0.0, total);
            const double target = draw(random_engine);
            double cumulative = 0.0;
            for (Eigen::Index d = 0; d < n; ++d) {
                cumulative += min_distance(d);
                if (min_distance(d) > 0.0 && cumulative >= target) {
                    selected = d;
                    break;
                }
            }
        }
        if (selected < 0) {
            for (Eigen::Index d = 0; d < n; ++d) {
                if (std::find(seeds.begin(), seeds.end(), d) == seeds.end()) {
                    selected = d;
                    break;
                }
            }
        }
        seeds.push_back(selected);
    }
    for (int32_t c = 0; c < clusters; ++c) {
        out.centers.row(c) = observations.row(seeds[c]);
    }

    Eigen::VectorXi previous = Eigen::VectorXi::Constant(n, -1);
    Eigen::VectorXd distances;
    for (int32_t iteration = 0; iteration < options.max_iterations; ++iteration) {
        assign_to_centers(observations, out.centers, out.assignments,
            distances, out.counts, out.inertia);
        bool changed = (out.assignments.array() != previous.array()).any();
        RowMajorMatrixXd sums = RowMajorMatrixXd::Zero(clusters, dim);
        for (Eigen::Index d = 0; d < n; ++d) {
            sums.row(out.assignments(d)) += observations.row(d);
        }

        std::vector<bool> reseeded(n, false);
        for (int32_t empty = 0; empty < clusters; ++empty) {
            if (out.counts(empty) > 0) continue;
            Eigen::Index farthest = -1;
            double farthest_distance = -1.0;
            for (Eigen::Index d = 0; d < n; ++d) {
                const int32_t donor = out.assignments(d);
                if (!reseeded[d] && out.counts(donor) > 1
                    && distances(d) > farthest_distance) {
                    farthest = d;
                    farthest_distance = distances(d);
                }
            }
            if (farthest < 0) {
                throw std::runtime_error("Cannot reseed empty dense k-means cluster");
            }
            const int32_t donor = out.assignments(farthest);
            sums.row(donor) -= observations.row(farthest);
            --out.counts(donor);
            out.assignments(farthest) = empty;
            sums.row(empty) = observations.row(farthest);
            out.counts(empty) = 1;
            distances(farthest) = 0.0;
            reseeded[farthest] = true;
            changed = true;
        }

        for (int32_t c = 0; c < clusters; ++c) {
            out.centers.row(c) = sums.row(c) / static_cast<double>(out.counts(c));
        }
        out.iterations = iteration + 1;
        if (!changed) {
            out.converged = true;
            break;
        }
        previous = out.assignments;
    }
    out.inertia = 0.0;
    for (Eigen::Index d = 0; d < n; ++d) {
        out.inertia += (observations.row(d)
            - out.centers.row(out.assignments(d))).squaredNorm();
    }
    return out;
}
