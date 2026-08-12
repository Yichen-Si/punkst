#include "clustering/uac_initialization_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <tbb/parallel_for.h>

namespace uac::detail {




HardPartitionMoments hard_partition_moments(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    int32_t components) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (assignments.size() != documents || components <= 0) {
        throw std::invalid_argument("Invalid UAC hard partition");
    }
    HardPartitionMoments out;
    out.counts = Eigen::VectorXi::Zero(components);
    out.means = RowMajorMatrixXd::Zero(components, dimension);
    for (int32_t d = 0; d < documents; ++d) {
        const int32_t component = assignments(d);
        if (component < 0 || component >= components) {
            throw std::invalid_argument(
                "UAC initial partition label is out of range");
        }
        ++out.counts(component);
        out.means.row(component) += data.coordinates.row(d);
    }
    for (int32_t c = 0; c < components; ++c) {
        if (out.counts(c) <= 0) {
            throw std::runtime_error(
                "UAC initial partition produced an empty component");
        }
        out.means.row(c) /= out.counts(c);
    }
    out.scatter.assign(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    for (int32_t d = 0; d < documents; ++d) {
        const int32_t component = assignments(d);
        const Eigen::VectorXd residual =
            data.coordinates.row(d).transpose()
            - out.means.row(component).transpose();
        out.scatter[component].noalias() +=
            residual * residual.transpose();
    }
    out.pooled_scatter =
        Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& scatter : out.scatter) {
        out.pooled_scatter += scatter;
    }
    out.pooled_scatter /= documents;
    out.pooled_scatter = 0.5
        * (out.pooled_scatter + out.pooled_scatter.transpose());
    return out;
}

Eigen::MatrixXd measurement_covariance(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    ProposalKind proposal, FisherWorkspace* fisher_workspace = nullptr,
    Eigen::LLT<Eigen::MatrixXd>* supplied_solver = nullptr) {
    const FisherApproximation fisher = fisher_approximation_impl(
        coordinate, document, basis, helmert, proposal, false,
        fisher_workspace);
    Eigen::MatrixXd precision =
        fisher.information + regularizing_precision;
    precision = 0.5 * (precision + precision.transpose());
    Eigen::LLT<Eigen::MatrixXd> local_solver;
    Eigen::LLT<Eigen::MatrixXd>& solver =
        supplied_solver ? *supplied_solver : local_solver;
    solver.compute(precision);
    if (solver.info() != Eigen::Success) {
        throw std::runtime_error(
            "UAC deconvolution measurement precision is not positive definite");
    }
    Eigen::MatrixXd covariance = solver.solve(
        Eigen::MatrixXd::Identity(precision.rows(), precision.cols()));
    covariance = 0.5 * (covariance + covariance.transpose());
    if (!covariance.allFinite()) {
        throw std::runtime_error(
            "UAC deconvolution measurement covariance is nonfinite");
    }
    return covariance;
}

Eigen::MatrixXd measurement_covariance(const Dataset& data,
    const Basis& basis, const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    ProposalKind proposal, int32_t document) {
    return measurement_covariance(
        data.coordinates.row(document).transpose(),
        data.counts[document], basis, helmert, regularizing_precision,
        proposal);
}

Eigen::MatrixXd shared_measurement_precision(
    const std::vector<HardPartitionMoments>& moments,
    double scalar_precision, double relative_floor) {
    if (moments.empty() || !(scalar_precision >= 0.0)
        || !std::isfinite(scalar_precision) || !(relative_floor > 0.0)) {
        throw std::invalid_argument(
            "Invalid UAC deconvolution regularizing precision");
    }
    const int32_t dimension = static_cast<int32_t>(
        moments.front().pooled_scatter.rows());
    if (scalar_precision > 0.0) {
        return scalar_precision
            * Eigen::MatrixXd::Identity(dimension, dimension);
    }
    Eigen::MatrixXd pooled =
        Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& value : moments) {
        if (value.pooled_scatter.rows() != dimension
            || value.pooled_scatter.cols() != dimension) {
            throw std::runtime_error(
                "Incompatible UAC start scatters");
        }
        pooled += value.pooled_scatter;
    }
    pooled /= moments.size();
    const double floor = std::max(1e-12,
        relative_floor * pooled.trace() / dimension);
    pooled = floor_covariance(pooled, floor);
    Eigen::LLT<Eigen::MatrixXd> llt(pooled);
    if (llt.info() != Eigen::Success) {
        throw std::runtime_error(
            "UAC shared deconvolution scatter is not positive definite");
    }
    return llt.solve(
        Eigen::MatrixXd::Identity(dimension, dimension));
}

std::vector<std::vector<Eigen::MatrixXd>>
measurement_sums_by_partition(const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const std::vector<Eigen::VectorXi>& assignments,
    int32_t components, ProposalKind proposal,
    const IndexedDocumentSource* count_source) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    using PartitionSums = std::vector<std::vector<Eigen::MatrixXd>>;
    auto empty_sums = [&]() {
        return PartitionSums(assignments.size(),
            std::vector<Eigen::MatrixXd>(components,
                Eigen::MatrixXd::Zero(dimension, dimension)));
    };
    PartitionSums out = empty_sums();
    for (const auto& assignment : assignments) {
        if (assignment.size() != documents) {
            throw std::invalid_argument(
                "Invalid UAC measurement partition");
        }
    }
    constexpr uint64_t kScratchBudget = 64ull * 1024ull * 1024ull;
    constexpr int32_t kMaximumBlocks = 32;
    const uint64_t accumulator_values =
        static_cast<uint64_t>(assignments.size()) * components
        * dimension * dimension;
    const uint64_t fisher_values =
        proposal == ProposalKind::ExactFisher
        ? static_cast<uint64_t>(basis.probabilities.rows()) * dimension
            + basis.probabilities.rows()
            + static_cast<uint64_t>(basis.probabilities.cols()) * dimension
            + static_cast<uint64_t>(basis.probabilities.cols())
                * basis.probabilities.cols()
        : static_cast<uint64_t>(dimension) * dimension
            + static_cast<uint64_t>(basis.probabilities.cols()) * dimension;
    const uint64_t bytes_per_block = sizeof(double)
        * std::max<uint64_t>(1, accumulator_values + fisher_values);
    const int32_t memory_blocks = static_cast<int32_t>(
        std::max<uint64_t>(1, kScratchBudget / bytes_per_block));
    const int32_t requested_blocks = std::max(1, std::min({
        documents, kMaximumBlocks, memory_blocks}));
    const int32_t block_size =
        (documents + requested_blocks - 1) / requested_blocks;
    const int32_t n_blocks =
        (documents + block_size - 1) / block_size;
    std::vector<PartitionSums> block_sums;
    block_sums.reserve(n_blocks);
    for (int32_t block = 0; block < n_blocks; ++block) {
        block_sums.push_back(empty_sums());
    }
    constexpr int32_t kCountBlock = 64;
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        PartitionSums& local_sums = block_sums[block_index];
        FisherWorkspace fisher_workspace;
        Eigen::LLT<Eigen::MatrixXd> measurement_solver;
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        for (int32_t first = begin; first < end; first += kCountBlock) {
            const int32_t count =
                std::min(kCountBlock, end - first);
            DocumentBlock block;
            if (count_source) {
                block = read_aligned_document_range(
                    data, *count_source, first, count);
            }
            for (int32_t local = 0; local < count; ++local) {
                const int32_t d = first + local;
                const Document& document = count_source
                    ? block.counts[local] : data.counts[d];
                const Eigen::MatrixXd covariance = measurement_covariance(
                    data.coordinates.row(d).transpose(), document,
                    basis, helmert, regularizing_precision, proposal,
                    &fisher_workspace, &measurement_solver);
                for (size_t start = 0; start < assignments.size(); ++start) {
                    const int32_t component = assignments[start](d);
                    if (component < 0 || component >= components) {
                        throw std::invalid_argument(
                            "UAC measurement partition label is out of range");
                    }
                    local_sums[start][component] += covariance;
                }
            }
        }
    });
    for (const PartitionSums& block : block_sums) {
        for (size_t start = 0; start < assignments.size(); ++start) {
            for (int32_t component = 0;
                    component < components; ++component) {
                out[start][component] += block[start][component];
            }
        }
    }
    return out;
}

namespace {

double initialization_sample_uniform(int32_t seed, int32_t document) {
    uint64_t value = static_cast<uint32_t>(seed);
    value ^= (static_cast<uint64_t>(static_cast<uint32_t>(document)) + 1)
        * 0x9e3779b97f4a7c15ULL;
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<double>(value >> 11)
        * (1.0 / 9007199254740992.0);
}

Eigen::VectorXd initialization_inclusion_probabilities(
    const std::vector<Eigen::VectorXi>& assignments,
    const std::vector<HardPartitionMoments>& moments,
    int32_t documents, int32_t target) {
    Eigen::VectorXd probability = Eigen::VectorXd::Ones(documents);
    if (target <= 0) return probability;
    probability.setZero();
    for (size_t start = 0; start < assignments.size(); ++start) {
        for (int32_t d = 0; d < documents; ++d) {
            const int32_t component = assignments[start](d);
            probability(d) = std::max(probability(d), std::min(
                1.0, static_cast<double>(target)
                    / moments[start].counts(component)));
        }
    }
    return probability;
}

double minimum_partition_effective_size(
    const std::vector<Eigen::VectorXi>& assignments, int32_t components,
    const Eigen::Ref<const Eigen::VectorXd>& probability,
    const std::vector<uint8_t>& selected) {
    double minimum = std::numeric_limits<double>::infinity();
    for (const auto& assignment : assignments) {
        Eigen::VectorXd sum = Eigen::VectorXd::Zero(components);
        Eigen::VectorXd square = Eigen::VectorXd::Zero(components);
        for (int32_t d = 0; d < assignment.size(); ++d) {
            if (!selected[d]) continue;
            const double weight = 1.0 / probability(d);
            sum(assignment(d)) += weight;
            square(assignment(d)) += weight * weight;
        }
        for (int32_t c = 0; c < components; ++c) {
            if (!(square(c) > 0.0)) {
                throw std::runtime_error(
                    "UAC initialization sample omitted a start/component stratum");
            }
            minimum = std::min(minimum,
                sum(c) * sum(c) / square(c));
        }
    }
    return minimum;
}

void pack_symmetric(const Eigen::Ref<const Eigen::MatrixXd>& matrix,
    Eigen::Ref<Eigen::RowVectorXd> packed) {
    Eigen::Index offset = 0;
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col <= row; ++col) {
            packed(offset++) = matrix(row, col);
        }
    }
}

void unpack_symmetric(const Eigen::Ref<const Eigen::RowVectorXd>& packed,
    Eigen::MatrixXd& matrix) {
    Eigen::Index offset = 0;
    for (Eigen::Index row = 0; row < matrix.rows(); ++row) {
        for (Eigen::Index col = 0; col <= row; ++col) {
            matrix(row, col) = packed(offset);
            matrix(col, row) = packed(offset++);
        }
    }
}

} // namespace

InitializationMeasurements collect_initialization_measurements(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const std::vector<Eigen::VectorXi>& assignments,
    const std::vector<HardPartitionMoments>& moments,
    int32_t components, ProposalKind proposal,
    InitializationMeasurementMode mode, int32_t measurement_target,
    int32_t score_target, int32_t seed,
    const IndexedDocumentSource* count_source) {
    const auto work_start = std::chrono::steady_clock::now();
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (assignments.empty() || assignments.size() != moments.size()
        || components <= 0 || measurement_target <= 0 || score_target < 0) {
        throw std::invalid_argument(
            "Invalid UAC initialization measurement sampling options");
    }
    const bool sampled_moments = mode
        == InitializationMeasurementMode::HorvitzThompson;
    const Eigen::VectorXd measurement_probability =
        initialization_inclusion_probabilities(assignments, moments,
            documents, sampled_moments ? measurement_target : 0);
    const Eigen::VectorXd score_probability =
        initialization_inclusion_probabilities(assignments, moments,
            documents, score_target);
    std::vector<uint8_t> measurement_selected(documents, 0);
    std::vector<uint8_t> score_selected(documents, 0);
    std::vector<int32_t> score_row(documents, -1);
    InitializationMeasurements out;
    for (int32_t d = 0; d < documents; ++d) {
        const double uniform = initialization_sample_uniform(seed, d);
        measurement_selected[d] = uniform < measurement_probability(d);
        score_selected[d] = uniform < score_probability(d);
        if (measurement_selected[d]) ++out.measurement_documents;
        if (score_selected[d]) {
            score_row[d] = static_cast<int32_t>(out.score_documents.size());
            out.score_documents.push_back(d);
        }
    }
    out.score_probabilities.resize(out.score_documents.size());
    for (size_t row = 0; row < out.score_documents.size(); ++row) {
        out.score_probabilities(row) =
            score_probability(out.score_documents[row]);
    }
    out.maximum_measurement_weight =
        1.0 / measurement_probability.minCoeff();
    out.maximum_score_weight = 1.0 / score_probability.minCoeff();
    out.minimum_measurement_effective_size =
        minimum_partition_effective_size(assignments, components,
            measurement_probability, measurement_selected);
    out.minimum_score_effective_size =
        minimum_partition_effective_size(assignments, components,
            score_probability, score_selected);

    const int64_t packed_values = static_cast<int64_t>(dimension)
        * (dimension + 1) / 2;
    out.packed_score_covariances.resize(
        out.score_documents.size(), packed_values);
    out.cache_bytes = static_cast<uint64_t>(
        out.packed_score_covariances.size()) * sizeof(double);
    using PartitionSums = std::vector<std::vector<Eigen::MatrixXd>>;
    auto empty_sums = [&]() {
        return PartitionSums(assignments.size(),
            std::vector<Eigen::MatrixXd>(components,
                Eigen::MatrixXd::Zero(dimension, dimension)));
    };
    constexpr int32_t kMaximumBlocks = 32;
    const int32_t n_blocks = std::min(documents, kMaximumBlocks);
    const int32_t block_size = (documents + n_blocks - 1) / n_blocks;
    std::vector<PartitionSums> block_sums;
    block_sums.reserve(n_blocks);
    for (int32_t block = 0; block < n_blocks; ++block) {
        block_sums.push_back(empty_sums());
    }
    std::vector<int64_t> block_evaluations(n_blocks, 0);
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        FisherWorkspace fisher_workspace;
        Eigen::LLT<Eigen::MatrixXd> measurement_solver;
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        DocumentBlock count_block;
        if (count_source) {
            count_block = read_aligned_document_range(
                data, *count_source, begin, end - begin);
        }
        for (int32_t d = begin; d < end; ++d) {
            if (!measurement_selected[d] && !score_selected[d]) continue;
            const Eigen::MatrixXd covariance = measurement_covariance(
                data.coordinates.row(d).transpose(),
                count_source ? count_block.counts[d - begin]
                             : data.counts[d],
                basis, helmert, regularizing_precision, proposal,
                &fisher_workspace, &measurement_solver);
            ++block_evaluations[block_index];
            if (measurement_selected[d]) {
                const double weight = 1.0 / measurement_probability(d);
                for (size_t start = 0; start < assignments.size(); ++start) {
                    block_sums[block_index][start][assignments[start](d)]
                        .noalias() += weight * covariance;
                }
            }
            if (score_selected[d]) {
                Eigen::RowVectorXd packed =
                    out.packed_score_covariances.row(score_row[d]);
                pack_symmetric(covariance, packed);
                out.packed_score_covariances.row(score_row[d]) = packed;
            }
        }
    });
    out.sums = empty_sums();
    for (int32_t block = 0; block < n_blocks; ++block) {
        out.covariance_evaluations += block_evaluations[block];
        for (size_t start = 0; start < assignments.size(); ++start) {
            for (int32_t component = 0; component < components; ++component) {
                out.sums[start][component] +=
                    block_sums[block][start][component];
            }
        }
    }
    out.seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - work_start).count();
    return out;
}

Model initialize_model_from_partition(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments, int32_t components,
    double shrinkage, double covariance_floor, double relative_floor) {
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (assignments.size() != data.coordinates.rows() || components <= 0
        || !(shrinkage >= 0.0) || !(relative_floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC initial partition");
    }
    Eigen::VectorXi counts = Eigen::VectorXi::Zero(components);
    Model model;
    model.means = RowMajorMatrixXd::Zero(components, dimension);
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        const int32_t component = assignments(d);
        if (component < 0 || component >= components) {
            throw std::invalid_argument(
                "UAC initial partition label is out of range");
        }
        ++counts(component);
        model.means.row(component) += data.coordinates.row(d);
    }
    for (int32_t c = 0; c < components; ++c) {
        if (counts(c) <= 0) {
            throw std::runtime_error(
                "UAC initial partition produced an empty component");
        }
        model.means.row(c) /= counts(c);
    }
    std::vector<Eigen::MatrixXd> scatter(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    for (int32_t d = 0; d < data.coordinates.rows(); ++d) {
        const int32_t component = assignments(d);
        const Eigen::VectorXd residual = data.coordinates.row(d).transpose()
            - model.means.row(component).transpose();
        scatter[component].noalias() += residual * residual.transpose();
    }
    model.shrinkage_target = Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& value : scatter) model.shrinkage_target += value;
    model.shrinkage_target /= data.coordinates.rows();
    const double target_floor = std::max(1e-12,
        relative_floor * model.shrinkage_target.trace() / dimension);
    model.shrinkage_target = floor_covariance(
        model.shrinkage_target, target_floor);
    model.weights = counts.cast<double>() / data.coordinates.rows();
    model.covariances.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        model.covariances.push_back(floor_covariance(
            (scatter[c] + shrinkage * model.shrinkage_target)
                / (counts(c) + shrinkage),
            covariance_floor));
    }
    return model;
}

Model initialize_model_from_corrected_moments(const Dataset& data,
    const Eigen::Ref<const Eigen::VectorXi>& assignments,
    const HardPartitionMoments& moments,
    const std::vector<Eigen::MatrixXd>& measurement_sum,
    double shrinkage, double covariance_floor,
    int32_t* floor_activations) {
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(moments.counts.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (assignments.size() != documents
        || moments.means.rows() != components
        || moments.means.cols() != dimension
        || measurement_sum.size() != static_cast<size_t>(components)
        || !(shrinkage >= 0.0) || !(covariance_floor > 0.0)) {
        throw std::invalid_argument(
            "Invalid UAC corrected-moment initialization");
    }
    Eigen::MatrixXd corrected_target = moments.pooled_scatter;
    for (const auto& sum : measurement_sum) {
        corrected_target -= sum / documents;
    }
    auto floor_and_count = [&](const Eigen::MatrixXd& covariance) {
        if (floor_activations) {
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(
                0.5 * (covariance + covariance.transpose()),
                Eigen::EigenvaluesOnly);
            if (solver.info() == Eigen::Success
                && solver.eigenvalues().minCoeff() < covariance_floor) {
                ++*floor_activations;
            }
        }
        return floor_covariance(covariance, covariance_floor);
    };
    corrected_target = floor_and_count(corrected_target);

    Model model;
    model.weights = moments.counts.cast<double>() / documents;
    model.means = moments.means;
    model.shrinkage_target = corrected_target;
    model.covariances.reserve(components);
    for (int32_t c = 0; c < components; ++c) {
        Eigen::MatrixXd numerator =
            moments.scatter[c] - measurement_sum[c]
            + shrinkage * corrected_target;
        model.covariances.push_back(floor_and_count(
            numerator / (moments.counts(c) + shrinkage)));
    }
    return model;
}

Pilot pilot_from_map(const Dataset& data, const Model& model,
    const Expectation& expectation,
    double relative_floor) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if (expectation.documents != data.coordinates.rows()
        || expectation.membership.size() != components
        || expectation.first.rows() != components
        || expectation.second.size() != static_cast<size_t>(components)
        || !(relative_floor > 0.0)) {
        throw std::invalid_argument("Invalid UAC winning MAP responsibilities");
    }
    Pilot out;
    out.weights = model.weights;
    out.means = model.means;
    std::vector<Eigen::MatrixXd> raw_covariances(components,
        Eigen::MatrixXd::Zero(dimension, dimension));
    const Eigen::VectorXd& membership = expectation.membership;
    for (int32_t c = 0; c < components; ++c) {
        const Eigen::VectorXd mean = model.means.row(c).transpose();
        raw_covariances[c] = expectation.second[c]
            - expectation.first.row(c).transpose() * mean.transpose()
            - mean * expectation.first.row(c)
            + membership(c) * mean * mean.transpose();
        raw_covariances[c] = 0.5 * (raw_covariances[c]
            + raw_covariances[c].transpose());
    }
    out.pooled_covariance = Eigen::MatrixXd::Zero(dimension, dimension);
    for (const auto& scatter : raw_covariances) {
        out.pooled_covariance += scatter;
    }
    const double total_membership = membership.sum();
    if (!(total_membership > 0.0)) {
        throw std::runtime_error("UAC winning MAP has no active membership");
    }
    out.pooled_covariance /= total_membership;
    const double target_floor = std::max(1e-12,
        relative_floor * out.pooled_covariance.trace() / dimension);
    out.pooled_covariance = floor_covariance(
        out.pooled_covariance, target_floor);
    out.covariances = model.covariances;
    const double epsilon = membership_epsilon(data.coordinates.rows());
    for (int32_t c = 0; c < components; ++c) {
        if (membership(c) > epsilon && model.weights(c) > 0.0) {
            raw_covariances[c] /= membership(c);
            raw_covariances[c] = 0.5 * (raw_covariances[c]
                + raw_covariances[c].transpose());
        } else {
            out.covariances[c] = out.pooled_covariance;
        }
    }
    return out;
}

Pilot pilot_from_model(const Model& model) {
    if ((model.covariance_kind == CovarianceKind::Dense
            && model.covariances.size()
                != static_cast<size_t>(model.weights.size()))
        || (model.covariance_kind == CovarianceKind::FactorAnalytic
            && model.factor_covariances.size()
                != static_cast<size_t>(model.weights.size()))) {
        throw std::invalid_argument(
            "UAC pilot requires a complete covariance model");
    }
    Pilot out;
    out.weights = model.weights;
    out.means = model.means;
    if (model.covariance_kind == CovarianceKind::Dense) {
        out.covariances = model.covariances;
        out.pooled_covariance = model.shrinkage_target;
    } else {
        out.pooled_covariance = Eigen::MatrixXd::Zero(
            model.means.cols(), model.means.cols());
        out.covariances.reserve(model.weights.size());
        for (int32_t c = 0; c < model.weights.size(); ++c) {
            out.covariances.push_back(model_covariance_dense(model, c));
            out.pooled_covariance.noalias() +=
                model.weights(c) * out.covariances.back();
        }
    }
    return out;
}

struct DeconvolutionScore {
    double log_likelihood = 0.0;
    double top_probability_sum = 0.0;
    double gaussian_seconds = 0.0;
};

std::vector<DeconvolutionScore> deconvolution_marginal_scores(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const std::vector<Model>& models, ProposalKind proposal,
    const IndexedDocumentSource* count_source = nullptr) {
    if (models.empty()) return {};
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components =
        static_cast<int32_t>(models.front().weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    if ((!count_source
            && data.counts.size() != static_cast<size_t>(documents))
        || (count_source && count_source->documents() != documents)
        || regularizing_precision.rows() != dimension
        || regularizing_precision.cols() != dimension) {
        throw std::invalid_argument(
            "Invalid UAC deconvolution score input");
    }
    for (const auto& model : models) {
        if (model.covariance_kind != CovarianceKind::Dense
            || model.weights.size() != components
            || model.means.rows() != components
            || model.means.cols() != dimension
            || model.covariances.size()
                != static_cast<size_t>(components)) {
            throw std::invalid_argument(
                "Incompatible UAC deconvolution candidate");
        }
    }
    constexpr uint64_t kScratchBudget = 64ull * 1024ull * 1024ull;
    constexpr int32_t kMaximumBlocks = 32;
    const uint64_t fisher_values =
        proposal == ProposalKind::ExactFisher
        ? static_cast<uint64_t>(basis.probabilities.rows()) * dimension
            + basis.probabilities.rows()
            + static_cast<uint64_t>(basis.probabilities.cols()) * dimension
            + static_cast<uint64_t>(basis.probabilities.cols())
                * basis.probabilities.cols()
        : static_cast<uint64_t>(dimension) * dimension
            + static_cast<uint64_t>(basis.probabilities.cols()) * dimension;
    const uint64_t solver_values =
        4 * static_cast<uint64_t>(dimension) * dimension
        + static_cast<uint64_t>(models.size()) * components;
    const uint64_t bytes_per_block = sizeof(double)
        * std::max<uint64_t>(1, fisher_values + solver_values);
    const int32_t memory_blocks = static_cast<int32_t>(
        std::max<uint64_t>(1, kScratchBudget / bytes_per_block));
    const int32_t requested_blocks = std::max(1, std::min({
        documents, kMaximumBlocks, memory_blocks}));
    const int32_t block_size =
        (documents + requested_blocks - 1) / requested_blocks;
    const int32_t n_blocks =
        (documents + block_size - 1) / block_size;
    std::vector<std::vector<double>> block_likelihood(
        n_blocks, std::vector<double>(models.size(), 0.0));
    std::vector<std::vector<double>> block_top_probability(
        n_blocks, std::vector<double>(models.size(), 0.0));
    std::vector<double> block_seconds(n_blocks, 0.0);
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        std::vector<Eigen::VectorXd> log_score(
            models.size(), Eigen::VectorXd(components));
        FisherWorkspace fisher_workspace;
        Eigen::LLT<Eigen::MatrixXd> measurement_solver;
        Eigen::LLT<Eigen::MatrixXd> solver;
        Eigen::MatrixXd marginal(dimension, dimension);
        Eigen::VectorXd residual(dimension);
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(documents, begin + block_size);
        DocumentBlock count_block;
        if (count_source) {
            count_block = read_aligned_document_range(
                data, *count_source, begin, end - begin);
        }
        const auto work_start = std::chrono::steady_clock::now();
        for (int32_t d = begin; d < end; ++d) {
            const Eigen::VectorXd observed =
                data.coordinates.row(d).transpose();
            const Eigen::MatrixXd measurement = measurement_covariance(
                observed,
                count_source
                    ? count_block.counts[d - begin] : data.counts[d],
                basis, helmert, regularizing_precision, proposal,
                &fisher_workspace, &measurement_solver);
            for (size_t candidate = 0; candidate < models.size();
                    ++candidate) {
                const Model& model = models[candidate];
                log_score[candidate].setConstant(
                    -std::numeric_limits<double>::infinity());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    marginal = model.covariances[c] + measurement;
                    marginal = 0.5 * (
                        marginal + marginal.transpose());
                    solver.compute(marginal);
                    if (solver.info() != Eigen::Success) {
                        throw std::runtime_error(
                            "UAC deconvolution marginal covariance is not "
                            "positive definite");
                    }
                    const double log_determinant =
                        2.0 * solver.matrixLLT().diagonal()
                            .array().log().sum();
                    residual = observed - model.means.row(c).transpose();
                    log_score[candidate](c) = std::log(model.weights(c))
                        - 0.5 * (dimension * kLog2Pi + log_determinant
                            + residual.dot(
                                solver.solve(residual)));
                }
                const double normalizer =
                    logsumexp(log_score[candidate]);
                if (!std::isfinite(normalizer)) {
                    throw std::runtime_error(
                        "UAC deconvolution has no finite component "
                        "evidence");
                }
                const Eigen::VectorXd responsibility =
                    (log_score[candidate].array() - normalizer).exp();
                block_likelihood[block_index][candidate] += normalizer;
                block_top_probability[block_index][candidate] +=
                    responsibility.maxCoeff();
            }
        }
        block_seconds[block_index] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - work_start).count();
    });
    std::vector<DeconvolutionScore> out(models.size());
    const double total_seconds =
        std::accumulate(block_seconds.begin(), block_seconds.end(), 0.0);
    for (size_t candidate = 0; candidate < models.size(); ++candidate) {
        for (int32_t block = 0; block < n_blocks; ++block) {
            out[candidate].log_likelihood +=
                block_likelihood[block][candidate];
            out[candidate].top_probability_sum +=
                block_top_probability[block][candidate];
        }
        out[candidate].gaussian_seconds =
            total_seconds / models.size();
    }
    return out;
}



LowRankDiagonalCovariance shrink_factor_covariance(
    const LowRankDiagonalCovariance& raw,
    const LowRankDiagonalCovariance& target, double membership,
    double shrinkage, int32_t rank, double floor) {
    const double alpha = membership / (membership + shrinkage);
    LowRankDiagonalCovariance out;
    out.diagonal = (alpha * raw.diagonal
        + (1.0 - alpha) * target.diagonal).cwiseMax(floor);
    const int32_t columns = static_cast<int32_t>(raw.factor.cols()
        + target.factor.cols());
    if (rank == 0 || columns == 0) {
        out.factor = RowMajorMatrixXd(raw.diagonal.size(), 0);
        if (columns > 0) {
            out.diagonal.array() += alpha
                * raw.factor.array().square().rowwise().sum();
            out.diagonal.array() += (1.0 - alpha)
                * target.factor.array().square().rowwise().sum();
        }
        return out;
    }
    Eigen::MatrixXd combined(raw.diagonal.size(), columns);
    if (raw.factor.cols() > 0) {
        combined.leftCols(raw.factor.cols()) = std::sqrt(alpha) * raw.factor;
    }
    if (target.factor.cols() > 0) {
        combined.rightCols(target.factor.cols()) = std::sqrt(1.0 - alpha)
            * target.factor;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(combined,
        Eigen::ComputeThinU | Eigen::ComputeThinV);
    const int32_t retained = std::min<int32_t>(rank,
        static_cast<int32_t>(svd.singularValues().size()));
    out.factor = svd.matrixU().leftCols(retained)
        * svd.singularValues().head(retained).asDiagonal();
    if (retained < svd.singularValues().size()) {
        const Eigen::MatrixXd discarded = svd.matrixU().middleCols(
            retained, svd.matrixU().cols() - retained)
            * svd.singularValues().segment(retained,
                svd.singularValues().size() - retained).asDiagonal();
        out.diagonal.array() += discarded.array().square().rowwise().sum();
    }
    out.diagonal = out.diagonal.cwiseMax(floor);
    return out;
}

ModelUpdate update_model(Model& model, const Expectation& expectation,
    double shrinkage, double covariance_floor, bool adaptive_target,
    bool update_weights, bool allow_extinction,
    const Eigen::VectorXd* shrinkage_membership) {
    const int32_t components = static_cast<int32_t>(model.weights.size());
    const int32_t documents = expectation.documents;
    const double epsilon = membership_epsilon(documents);
    if (!expectation.membership.allFinite()
        || (expectation.membership.array() < 0.0).any()) {
        return {};
    }
    if (shrinkage_membership != nullptr
        && (shrinkage_membership->size() != components
            || !shrinkage_membership->allFinite()
            || (shrinkage_membership->array() < 0.0).any())) {
        return {};
    }
    double active_mass = 0.0;
    for (int32_t c = 0; c < components; ++c) {
        if (expectation.membership(c) > epsilon) {
            active_mass += expectation.membership(c);
        }
    }
    if (!(active_mass > 0.0) || !std::isfinite(active_mass)) return {};

    Model next = model;
    if (update_weights) next.weights.setZero();
    ModelUpdate result;
    std::vector<Eigen::MatrixXd> raw_dense(components);
    std::vector<LowRankDiagonalCovariance> raw_factor(components);
    for (int32_t c = 0; c < components; ++c) {
        const double membership = expectation.membership(c);
        if (!(membership > epsilon)) {
            if (!allow_extinction && model.weights(c) > 0.0) {
                ++result.active_components;
            }
            continue;
        }
        if (!expectation.first.row(c).allFinite()
            || (model.covariance_kind == CovarianceKind::Dense
                && !expectation.second[c].allFinite())) {
            return {};
        }
        if (update_weights) next.weights(c) = membership / active_mass;
        if (model.covariance_kind == CovarianceKind::FactorAnalytic) {
            const int32_t rank = static_cast<int32_t>(
                model.factor_covariances[c].factor.cols());
            if (expectation.sum_y2.cols() != model.means.cols()
                || expectation.sum_f.cols() != rank) {
                return {};
            }
            Eigen::MatrixXd gram = Eigen::MatrixXd::Zero(
                rank + 1, rank + 1);
            gram(0, 0) = membership;
            if (rank > 0) {
                gram.block(0, 1, 1, rank) = expectation.sum_f.row(c);
                gram.block(1, 0, rank, 1) =
                    expectation.sum_f.row(c).transpose();
                gram.bottomRightCorner(rank, rank) =
                    expectation.sum_ff[c];
                if (model.factor_diagonal_mode
                        == FactorDiagonalMode::Shared
                    && shrinkage > 0.0) {
                    gram.bottomRightCorner(rank, rank).diagonal().array()
                        += shrinkage;
                }
            }
            Eigen::LLT<Eigen::MatrixXd> regression_llt(gram);
            if (regression_llt.info() != Eigen::Success) return {};
            Eigen::MatrixXd cross(model.means.cols(), rank + 1);
            cross.col(0) = expectation.first.row(c).transpose();
            if (rank > 0) cross.rightCols(rank) = expectation.sum_yf[c];
            const Eigen::MatrixXd coefficient = regression_llt.solve(
                cross.transpose()).transpose();
            const Eigen::VectorXd mean = coefficient.col(0);
            const Eigen::MatrixXd loading = coefficient.rightCols(rank);
            const Eigen::MatrixXd centered_yf = expectation.sum_yf[c]
                - mean * expectation.sum_f.row(c);
            Eigen::VectorXd centered_y2 =
                expectation.sum_y2.row(c).transpose();
            centered_y2.array() -= 2.0 * mean.array()
                * expectation.first.row(c).transpose().array();
            centered_y2.array() += membership * mean.array().square();
            Eigen::VectorXd residual = centered_y2;
            residual.array() -= 2.0
                * (loading.cwiseProduct(centered_yf)).rowwise().sum().array();
            residual.array() += (loading * expectation.sum_ff[c])
                .cwiseProduct(loading).rowwise().sum().array();
            raw_factor[c].diagonal =
                model.factor_diagonal_mode == FactorDiagonalMode::Shared
                ? residual : (residual / membership).cwiseMax(covariance_floor);
            raw_factor[c].factor = loading;
            next.means.row(c) = mean.transpose();
            ++result.active_components;
            continue;
        }
        next.means.row(c) = expectation.first.row(c) / membership;
        const Eigen::VectorXd mean = next.means.row(c).transpose();
        Eigen::MatrixXd scatter = expectation.second[c]
            - membership * mean * mean.transpose();
        scatter = 0.5 * (scatter + scatter.transpose());
        if (!scatter.allFinite()) return {};
        raw_dense[c] = scatter / membership;
        ++result.active_components;
    }
    if (result.active_components == 0 || !next.weights.allFinite()
        || !next.means.allFinite()) {
        return {};
    }
    if (model.covariance_kind == CovarianceKind::FactorAnalytic
        && model.factor_diagonal_mode == FactorDiagonalMode::Shared) {
        const int32_t dimension = static_cast<int32_t>(model.means.cols());
        const int32_t rank = static_cast<int32_t>(
            model.factor_covariances.front().factor.cols());
        Eigen::VectorXd residual = Eigen::VectorXd::Zero(dimension);
        for (int32_t c = 0; c < components; ++c) {
            if (!(expectation.membership(c) > epsilon)) continue;
            residual += raw_factor[c].diagonal;
            if (shrinkage > 0.0 && rank > 0) {
                residual.array() += shrinkage
                    * raw_factor[c].factor.array().square().rowwise().sum();
            }
        }
        next.factor_diagonal_mode = FactorDiagonalMode::Shared;
        next.shared_factor_diagonal =
            (residual / active_mass).cwiseMax(covariance_floor);
        next.factor_shrinkage_target.diagonal =
            next.shared_factor_diagonal;
        next.factor_shrinkage_target.factor =
            RowMajorMatrixXd::Zero(dimension, rank);
        for (int32_t c = 0; c < components; ++c) {
            next.factor_covariances[c].diagonal.resize(0);
            if (!(expectation.membership(c) > epsilon)) {
                if (allow_extinction || !(model.weights(c) > 0.0)) {
                    next.factor_covariances[c].factor =
                        RowMajorMatrixXd::Zero(dimension, rank);
                }
            } else {
                next.factor_covariances[c].factor = raw_factor[c].factor;
            }
        }
        if (!next.shared_factor_diagonal.allFinite()
            || !next.factor_shrinkage_target.factor.allFinite()) {
            return {};
        }
        result.valid = true;
        model = std::move(next);
        return result;
    }
    try {
        if (adaptive_target && shrinkage > 0.0) {
            Eigen::MatrixXd pooled = Eigen::MatrixXd::Zero(
                model.means.cols(), model.means.cols());
            for (int32_t c = 0; c < components; ++c) {
                const double membership = expectation.membership(c);
                if (!(membership > epsilon)) continue;
                pooled += membership * (model.covariance_kind
                        == CovarianceKind::Dense
                    ? raw_dense[c] : raw_factor[c].dense());
            }
            next.shrinkage_target = floor_covariance(
                pooled / active_mass, covariance_floor);
            if (model.covariance_kind == CovarianceKind::FactorAnalytic) {
                const int32_t rank = static_cast<int32_t>(
                    model.factor_covariances.front().factor.cols());
                next.factor_shrinkage_target = factorize_covariance(
                    next.shrinkage_target, rank, covariance_floor);
            }
        }
        for (int32_t c = 0; c < components; ++c) {
            const double membership = expectation.membership(c);
            if (!(membership > epsilon)) {
                if (!allow_extinction && model.weights(c) > 0.0) continue;
                if (model.covariance_kind == CovarianceKind::Dense) {
                    next.covariances[c] = next.shrinkage_target;
                } else {
                    next.factor_covariances[c] =
                        next.factor_shrinkage_target;
                }
                continue;
            }
            const double effective_membership = shrinkage_membership == nullptr
                ? membership : (*shrinkage_membership)(c);
            if (model.covariance_kind == CovarianceKind::Dense) {
                next.covariances[c] = floor_covariance(
                    (effective_membership * raw_dense[c]
                        + shrinkage * next.shrinkage_target)
                        / (effective_membership + shrinkage),
                    covariance_floor);
            } else {
                const int32_t rank = static_cast<int32_t>(
                    model.factor_covariances[c].factor.cols());
                next.factor_covariances[c] = shrink_factor_covariance(
                    raw_factor[c], next.factor_shrinkage_target,
                    effective_membership,
                    shrinkage, rank, covariance_floor);
            }
        }
    } catch (const std::runtime_error&) {
        return {};
    }
    result.valid = true;
    model = std::move(next);
    return result;
}



void score_corrected_moment_candidates(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Eigen::Ref<const Eigen::MatrixXd>& regularizing_precision,
    const FitOptions& options, std::vector<Candidate>& candidates,
    const IndexedDocumentSource* count_source) {
    std::vector<size_t> candidate_index;
    std::vector<Model> models;
    for (size_t index = 0; index < candidates.size(); ++index) {
        if (!candidates[index].trace.collapsed) {
            candidate_index.push_back(index);
            models.push_back(candidates[index].model);
        }
    }
    const std::vector<DeconvolutionScore> scores =
        deconvolution_marginal_scores(data, basis, helmert,
            regularizing_precision, models, options.proposal, count_source);
    for (size_t local = 0; local < scores.size(); ++local) {
        Candidate& candidate = candidates[candidate_index[local]];
        const DeconvolutionScore& score = scores[local];
        candidate.trace.estep_work.gaussian_seconds +=
            score.gaussian_seconds;
        candidate.trace.estep_work.document_evaluations +=
            data.coordinates.rows();
        candidate.objective = score.log_likelihood;
        record_trace_point(candidate.trace, options,
            TraceEvent::CandidateScore, 0, candidate.objective,
            active_component_count(candidate.model),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            score.top_probability_sum
                / std::max<Eigen::Index>(1, data.coordinates.rows()));
        candidate.trace.succeeded = true;
        candidate.trace.selection_objective = candidate.objective;
    }
}

void score_corrected_moment_candidates(
    const Dataset& data, const FitOptions& options,
    const InitializationMeasurements& measurements,
    std::vector<Candidate>& candidates) {
    std::vector<size_t> candidate_index;
    std::vector<Model> models;
    for (size_t index = 0; index < candidates.size(); ++index) {
        if (!candidates[index].trace.collapsed) {
            candidate_index.push_back(index);
            models.push_back(candidates[index].model);
        }
    }
    if (models.empty()) return;
    const int32_t components = static_cast<int32_t>(models[0].weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    const int32_t sampled = static_cast<int32_t>(
        measurements.score_documents.size());
    const int64_t packed_values = static_cast<int64_t>(dimension)
        * (dimension + 1) / 2;
    if (sampled <= 0
        || measurements.score_probabilities.size() != sampled
        || measurements.packed_score_covariances.rows() != sampled
        || measurements.packed_score_covariances.cols() != packed_values) {
        throw std::invalid_argument(
            "Invalid cached UAC initialization candidate score input");
    }
    constexpr int32_t kMaximumBlocks = 32;
    const int32_t n_blocks = std::min(sampled, kMaximumBlocks);
    const int32_t block_size = (sampled + n_blocks - 1) / n_blocks;
    std::vector<std::vector<double>> block_likelihood(
        n_blocks, std::vector<double>(models.size(), 0.0));
    std::vector<std::vector<double>> block_top_probability(
        n_blocks, std::vector<double>(models.size(), 0.0));
    std::vector<double> block_weight(n_blocks, 0.0);
    std::vector<double> block_seconds(n_blocks, 0.0);
    tbb::parallel_for(int32_t{0}, n_blocks, [&](int32_t block_index) {
        std::vector<Eigen::VectorXd> log_score(
            models.size(), Eigen::VectorXd(components));
        Eigen::LLT<Eigen::MatrixXd> solver;
        Eigen::MatrixXd measurement(dimension, dimension);
        Eigen::MatrixXd marginal(dimension, dimension);
        Eigen::VectorXd residual(dimension);
        const int32_t begin = block_index * block_size;
        const int32_t end = std::min(sampled, begin + block_size);
        const auto work_start = std::chrono::steady_clock::now();
        for (int32_t row = begin; row < end; ++row) {
            unpack_symmetric(
                measurements.packed_score_covariances.row(row), measurement);
            const int32_t document = measurements.score_documents[row];
            const Eigen::VectorXd observed =
                data.coordinates.row(document).transpose();
            const double survey_weight =
                1.0 / measurements.score_probabilities(row);
            block_weight[block_index] += survey_weight;
            for (size_t candidate = 0; candidate < models.size(); ++candidate) {
                const Model& model = models[candidate];
                log_score[candidate].setConstant(
                    -std::numeric_limits<double>::infinity());
                for (int32_t c = 0; c < components; ++c) {
                    if (!(model.weights(c) > 0.0)) continue;
                    marginal = model.covariances[c] + measurement;
                    marginal = 0.5 * (marginal + marginal.transpose());
                    solver.compute(marginal);
                    if (solver.info() != Eigen::Success) {
                        throw std::runtime_error(
                            "UAC cached deconvolution marginal covariance is not positive definite");
                    }
                    const double log_determinant =
                        2.0 * solver.matrixLLT().diagonal()
                            .array().log().sum();
                    residual = observed - model.means.row(c).transpose();
                    log_score[candidate](c) = std::log(model.weights(c))
                        - 0.5 * (dimension * kLog2Pi + log_determinant
                            + residual.dot(solver.solve(residual)));
                }
                const double normalizer = logsumexp(log_score[candidate]);
                if (!std::isfinite(normalizer)) {
                    throw std::runtime_error(
                        "UAC cached deconvolution has no finite component evidence");
                }
                const Eigen::VectorXd responsibility =
                    (log_score[candidate].array() - normalizer).exp();
                block_likelihood[block_index][candidate] +=
                    survey_weight * normalizer;
                block_top_probability[block_index][candidate] +=
                    survey_weight * responsibility.maxCoeff();
            }
        }
        block_seconds[block_index] = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - work_start).count();
    });
    const double total_seconds =
        std::accumulate(block_seconds.begin(), block_seconds.end(), 0.0);
    const double total_weight =
        std::accumulate(block_weight.begin(), block_weight.end(), 0.0);
    for (size_t local = 0; local < models.size(); ++local) {
        Candidate& candidate = candidates[candidate_index[local]];
        double top_probability_sum = 0.0;
        candidate.objective = 0.0;
        for (int32_t block = 0; block < n_blocks; ++block) {
            candidate.objective += block_likelihood[block][local];
            top_probability_sum += block_top_probability[block][local];
        }
        candidate.trace.estep_work.gaussian_seconds +=
            total_seconds / models.size();
        candidate.trace.estep_work.document_evaluations += sampled;
        record_trace_point(candidate.trace, options,
            TraceEvent::CandidateScore, 0, candidate.objective,
            active_component_count(candidate.model),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            std::numeric_limits<double>::quiet_NaN(),
            top_probability_sum / std::max(1.0, total_weight));
        candidate.trace.succeeded = true;
        candidate.trace.selection_objective = candidate.objective;
    }
}

} // namespace uac::detail
