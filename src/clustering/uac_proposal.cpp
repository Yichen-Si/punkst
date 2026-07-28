#include "clustering/uac_proposal_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace uac::detail {


FisherApproximation fisher_approximation_impl(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal, bool compute_gradient,
    FisherWorkspace* supplied_workspace) {
    if (coordinate.size() != helmert.rows()
        || basis.probabilities.cols() != helmert.cols()
        || document.ids.size() != document.cnts.size()
        || (proposal != ProposalKind::ExactFisher
            && proposal != ProposalKind::SparseEmpiricalFisher)) {
        throw std::invalid_argument("Invalid UAC Fisher input");
    }
    FisherWorkspace local_workspace;
    FisherWorkspace& workspace =
        supplied_workspace ? *supplied_workspace : local_workspace;
    workspace.composition = composition_from_coordinate(coordinate, helmert);
    const Eigen::VectorXd& composition = workspace.composition;
    workspace.simplex_covariance = composition.asDiagonal();
    Eigen::MatrixXd& simplex_covariance = workspace.simplex_covariance;
    simplex_covariance.noalias() -= composition * composition.transpose();
    workspace.simplex_derivative.noalias() =
        simplex_covariance * helmert.transpose();
    const Eigen::MatrixXd& simplex_derivative =
        workspace.simplex_derivative;
    FisherApproximation out;
    if (compute_gradient) {
        out.gradient = Eigen::VectorXd::Zero(coordinate.size());
    }
    out.information = Eigen::MatrixXd::Zero(
        coordinate.size(), coordinate.size());
    if (proposal == ProposalKind::SparseEmpiricalFisher) {
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const uint32_t feature = document.ids[j];
            const double count = document.cnts[j];
            if (feature >= static_cast<uint32_t>(basis.probabilities.rows())) {
                throw std::runtime_error(
                    "UAC document feature index is out of range");
            }
            if (!std::isfinite(count) || count < 0.0) {
                throw std::runtime_error("UAC document count is invalid");
            }
            if (!(count > 0.0)) continue;
            const double probability =
                basis.probabilities.row(feature).dot(composition);
            if (!(probability > 0.0) || !std::isfinite(probability)) {
                throw std::runtime_error(
                    "UAC observed feature has zero Fisher probability");
            }
            workspace.derivative = (
                basis.probabilities.row(feature) * simplex_derivative)
                .transpose();
            workspace.score = workspace.derivative / probability;
            const Eigen::VectorXd& score = workspace.score;
            if (compute_gradient) {
                out.gradient.noalias() += count * score;
            }
            out.information.selfadjointView<Eigen::Lower>().rankUpdate(
                score, count);
        }
        for (Eigen::Index row = 0; row < out.information.rows(); ++row) {
            for (Eigen::Index col = row + 1;
                    col < out.information.cols(); ++col) {
                out.information(row, col) = out.information(col, row);
            }
        }
    } else {
        workspace.probability = (basis.probabilities * composition)
            .array().max(1e-300).matrix();
        const Eigen::VectorXd& probability = workspace.probability;
        workspace.probability_derivative =
            basis.probabilities * simplex_derivative;
        RowMajorMatrixXd& probability_derivative =
            workspace.probability_derivative;
        double total = 0.0;
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const uint32_t feature = document.ids[j];
            const double count = document.cnts[j];
            if (feature >= static_cast<uint32_t>(probability.size())) {
                throw std::runtime_error(
                    "UAC document feature index is out of range");
            }
            if (!std::isfinite(count) || count < 0.0) {
                throw std::runtime_error("UAC document count is invalid");
            }
            total += count;
            if (compute_gradient && count > 0.0) {
                out.gradient.noalias() += count
                    / probability(feature)
                    * probability_derivative.row(feature).transpose();
            }
        }
        workspace.scale =
            (Eigen::VectorXd::Constant(probability.size(), total).array()
                / probability.array()).sqrt();
        probability_derivative.array().colwise() *= workspace.scale.array();
        out.information.noalias() = probability_derivative.transpose()
            * probability_derivative;
        out.information = 0.5 * (
            out.information + out.information.transpose());
    }
    return out;
}

PilotCache::PilotCache(const Pilot& pilot)
    : log_determinants(pilot.weights.size()) {
    inverse_covariances.reserve(pilot.weights.size());
    for (Eigen::Index c = 0; c < pilot.weights.size(); ++c) {
        Eigen::LLT<Eigen::MatrixXd> llt(pilot.covariances[c]);
        if (llt.info() != Eigen::Success) {
            throw std::runtime_error(
                "UAC Fisher pilot covariance is not positive definite");
        }
        inverse_covariances.push_back(llt.solve(
            Eigen::MatrixXd::Identity(
                pilot.covariances[c].rows(),
                pilot.covariances[c].cols())));
        const Eigen::MatrixXd lower = llt.matrixL();
        log_determinants(c) =
            2.0 * lower.diagonal().array().log().sum();
    }
}



DocumentProposal fisher_proposal(
    const Eigen::Ref<const Eigen::VectorXd>& center,
    const FisherApproximation& fisher, const Pilot& pilot,
    const PilotCache& cache, double broadening,
    const std::vector<int32_t>* candidate_components) {
    if (!(broadening > 0.0) || !std::isfinite(broadening)) {
        throw std::invalid_argument("Invalid UAC Fisher broadening");
    }
    const int32_t dimension = static_cast<int32_t>(center.size());
    DocumentProposal out;
    out.broadening = broadening;
    if (candidate_components) {
        out.component_ids = *candidate_components;
    } else {
        out.component_ids.resize(pilot.weights.size());
        std::iota(out.component_ids.begin(), out.component_ids.end(), 0);
    }
    if (out.component_ids.empty()) {
        throw std::runtime_error("UAC proposal has no candidate component");
    }
    out.weights.resize(out.component_ids.size());
    out.means.reserve(out.component_ids.size());
    out.precision_lower.reserve(out.component_ids.size());
    out.log_precision_determinants.resize(out.component_ids.size());
    Eigen::VectorXd log_weight(out.component_ids.size());
    for (size_t j = 0; j < out.component_ids.size(); ++j) {
        const int32_t c = out.component_ids[j];
        if (c < 0 || c >= pilot.weights.size()) {
            throw std::runtime_error(
                "UAC proposal candidate component is out of range");
        }
        if (!(pilot.weights(c) > 0.0)) {
            out.means.push_back(center);
            out.precision_lower.push_back(Eigen::MatrixXd::Identity(
                dimension, dimension));
            out.log_precision_determinants(j) = 0.0;
            log_weight(j) = -std::numeric_limits<double>::infinity();
            continue;
        }
        const Eigen::MatrixXd& inverse_covariance =
            cache.inverse_covariances[c];
        const double pilot_logdet = cache.log_determinants(c);
        const Eigen::VectorXd pilot_mean = pilot.means.row(c).transpose();
        const Eigen::VectorXd residual = center - pilot_mean;
        const Eigen::VectorXd b = fisher.gradient
            - inverse_covariance * residual;
        const Eigen::MatrixXd raw_precision = 0.5
            * (fisher.information + inverse_covariance
                + fisher.information.transpose()
                + inverse_covariance.transpose());
        Eigen::LLT<Eigen::MatrixXd> precision_llt(raw_precision);
        Eigen::MatrixXd precision_lower;
        bool fallback = precision_llt.info() != Eigen::Success;
        if (!fallback) {
            precision_lower = precision_llt.matrixL();
            fallback = !precision_lower.allFinite()
                || (precision_lower.diagonal().array() <= 0.0).any();
        }
        if (fallback) {
            const auto fallback_start = std::chrono::steady_clock::now();
            const Eigen::MatrixXd repaired =
                floor_covariance(raw_precision, 1e-8);
            precision_llt.compute(repaired);
            precision_lower = precision_llt.matrixL();
            out.precision_fallback_seconds +=
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - fallback_start).count();
            ++out.precision_fallbacks;
        }
        if (precision_llt.info() != Eigen::Success
            || !precision_lower.allFinite()
            || (precision_lower.diagonal().array() <= 0.0).any()) {
            throw std::runtime_error(
                "UAC Fisher precision is not positive definite");
        }
        const Eigen::VectorXd step = precision_llt.solve(b);
        const double precision_logdet = 2.0
            * precision_lower.diagonal().array().log().sum();
        out.means.push_back(center + step);
        out.precision_lower.push_back(std::move(precision_lower));
        out.log_precision_determinants(j) = precision_logdet;
        log_weight(j) = std::log(pilot.weights(c))
            - 0.5 * (dimension * kLog2Pi + pilot_logdet
                + residual.dot(inverse_covariance * residual))
            + 0.5 * b.dot(step) + 0.5 * dimension * kLog2Pi
            - 0.5 * precision_logdet;
    }
    out.weights = (log_weight.array() - logsumexp(log_weight)).exp();
    return out;
}

Eigen::VectorXd proposal_log_density_rows(
    const Eigen::Ref<const RowMajorMatrixXd>& values,
    const DocumentProposal& proposal) {
    Eigen::MatrixXd terms(proposal.weights.size(), values.rows());
    for (Eigen::Index j = 0; j < proposal.weights.size(); ++j) {
        if (!(proposal.weights(j) > 0.0)) {
            terms.row(j).setConstant(
                -std::numeric_limits<double>::infinity());
            continue;
        }
        const Eigen::MatrixXd residual =
            (values.rowwise() - proposal.means[j].transpose()).transpose();
        const Eigen::MatrixXd transformed =
            proposal.precision_lower[j].transpose() * residual;
        const double covariance_logdet = values.cols()
            * std::log(proposal.broadening)
            - proposal.log_precision_determinants(j);
        terms.row(j) = (std::log(proposal.weights(j)) - 0.5
            * (values.cols() * kLog2Pi + covariance_logdet
                + transformed.colwise().squaredNorm().array()
                    / proposal.broadening)).matrix();
    }
    Eigen::VectorXd out(values.rows());
    for (Eigen::Index s = 0; s < values.rows(); ++s) {
        out(s) = logsumexp(terms.col(s));
    }
    return out;
}



void add_screening_metrics(ScoreResult& score,
    const ComponentScreeningOptions& requested,
    const ProposalScreeningPlan& proposal,
    const ComponentScreeningOptions& particle) {
    score.component_screening_options = requested;
    score.proposal_component_screening = proposal.enabled;
    score.particle_component_screening =
        particle.mode == ComponentScreeningMode::On;
    score.proposal_screening_seconds = proposal.planning_seconds;
    score.proposal_audit_documents =
        checked_int32(proposal.audit_documents.size(),
            "proposal audit document count");
    score.proposal_audit_represented_components =
        proposal.audit_represented_components;
    score.proposal_audit_covered_components =
        proposal.audit_covered_components;
    score.proposal_audit_violations = proposal.audit_violations;
    score.proposal_audit_maximum_omitted_mass =
        proposal.maximum_audit_omitted_mass;
}

double document_effective_total(const Dataset& data, int32_t document) {
    if (data.effective_totals.size() == data.coordinates.rows()) {
        return data.effective_totals(document);
    }
    if (document < static_cast<int32_t>(data.counts.size())) {
        return std::accumulate(data.counts[document].cnts.begin(),
            data.counts[document].cnts.end(), 0.0);
    }
    return 0.0;
}

ProposalScreeningPlan make_proposal_screening_plan(
    const Dataset& data, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    const Pilot& pilot, const PilotCache& cache,
    ProposalKind proposal_kind, double broadening, uint64_t seed,
    const ComponentScreeningOptions& options,
    const IndexedDocumentSource* count_source) {
    validate_component_screening(options);
    ProposalScreeningPlan out;
    if (options.mode == ComponentScreeningMode::Off) return out;
    const auto planning_start = std::chrono::steady_clock::now();
    const int32_t documents = static_cast<int32_t>(data.coordinates.rows());
    const int32_t components = static_cast<int32_t>(pilot.weights.size());
    const int32_t dimension = static_cast<int32_t>(data.coordinates.cols());
    std::vector<int32_t> active;
    for (int32_t c = 0; c < components; ++c) {
        if (pilot.weights(c) > 0.0) active.push_back(c);
    }
    if (active.empty()) {
        throw std::runtime_error(
            "UAC proposal screening has no active pilot component");
    }
    out.active_components = static_cast<int32_t>(active.size());
    out.candidates.resize(documents);
    std::vector<double> proxy_entropy(documents, 0.0);
    std::vector<std::vector<int32_t>> groups(components);
    for (int32_t d = 0; d < documents; ++d) {
        Eigen::VectorXd proxy = Eigen::VectorXd::Constant(
            components, -std::numeric_limits<double>::infinity());
        const Eigen::VectorXd center =
            data.coordinates.row(d).transpose();
        for (const int32_t c : active) {
            const Eigen::VectorXd residual =
                center - pilot.means.row(c).transpose();
            proxy(c) = std::log(pilot.weights(c)) - 0.5
                * (dimension * kLog2Pi + cache.log_determinants(c)
                    + residual.dot(cache.inverse_covariances[c] * residual));
        }
        const double normalizer = logsumexp(proxy);
        std::vector<int32_t> order = active;
        std::stable_sort(order.begin(), order.end(), [&](int32_t left,
                int32_t right) {
            return proxy(left) == proxy(right)
                ? left < right : proxy(left) > proxy(right);
        });
        groups[order.front()].push_back(d);
        double cumulative = 0.0;
        const int32_t minimum = std::min<int32_t>(
            options.minimum_components, active.size());
        for (const int32_t c : order) {
            const double probability = std::exp(proxy(c) - normalizer);
            if (probability > 0.0) {
                proxy_entropy[d] -= probability * std::log(probability);
            }
            const bool below_maximum =
                options.mode != ComponentScreeningMode::On
                || options.maximum_components == 0
                || static_cast<int32_t>(out.candidates[d].size())
                    < options.maximum_components;
            if (below_maximum
                && (static_cast<int32_t>(out.candidates[d].size()) < minimum
                    || cumulative
                        < 1.0 - options.proposal_proxy_tail_mass)) {
                out.candidates[d].push_back(c);
            }
            cumulative += probability;
        }
    }

    const int32_t requested = options.audit_documents > 0
        ? options.audit_documents
        : std::min(256, std::max(16, 2 * out.active_components));
    const int32_t budget = std::min(documents, requested);
    std::vector<int32_t> represented;
    for (const int32_t c : active) {
        if (!groups[c].empty()) represented.push_back(c);
    }
    out.audit_represented_components =
        checked_int32(represented.size(),
            "proposal audit represented component count");
    const bool audit_covers_represented =
        static_cast<int32_t>(represented.size()) <= budget;
    if (budget < static_cast<int32_t>(represented.size())) {
        std::stable_sort(represented.begin(), represented.end(),
            [&](int32_t left, int32_t right) {
                return hash_string(seed, std::to_string(left))
                    < hash_string(seed, std::to_string(right));
            });
        represented.resize(budget);
    }
    std::vector<uint8_t> selected(documents, 0);
    auto add_document = [&](int32_t document) {
        if (static_cast<int32_t>(out.audit_documents.size()) >= budget
            || selected[document]) {
            return;
        }
        selected[document] = 1;
        out.audit_documents.push_back(document);
    };
    for (const int32_t c : represented) {
        const auto found = std::max_element(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return proxy_entropy[left] < proxy_entropy[right];
            });
        if (found != groups[c].end()) add_document(*found);
    }
    out.audit_covered_components =
        checked_int32(represented.size(),
            "proposal audit covered component count");
    for (const int32_t c : represented) {
        const auto found = std::min_element(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return document_effective_total(data, left)
                    < document_effective_total(data, right);
            });
        if (found != groups[c].end()) add_document(*found);
    }
    std::vector<size_t> cursor(components, 0);
    for (const int32_t c : represented) {
        std::stable_sort(groups[c].begin(), groups[c].end(),
            [&](int32_t left, int32_t right) {
                return hash_string(seed, data.identifiers[left])
                    < hash_string(seed, data.identifiers[right]);
            });
    }
    while (static_cast<int32_t>(out.audit_documents.size()) < budget) {
        bool added = false;
        for (const int32_t c : represented) {
            while (cursor[c] < groups[c].size()
                && selected[groups[c][cursor[c]]]) {
                ++cursor[c];
            }
            if (cursor[c] < groups[c].size()) {
                add_document(groups[c][cursor[c]++]);
                added = true;
                if (static_cast<int32_t>(out.audit_documents.size())
                    >= budget) {
                    break;
                }
            }
        }
        if (!added) break;
    }

    for (const int32_t d : out.audit_documents) {
        DocumentBlock count_block;
        if (count_source) {
            count_block =
                read_aligned_document_range(data, *count_source, d, 1);
        }
        const Eigen::VectorXd center =
            data.coordinates.row(d).transpose();
        const FisherApproximation fisher = fisher_approximation_impl(
            center,
            count_source ? count_block.counts.front() : data.counts[d],
            basis, helmert, proposal_kind);
        const DocumentProposal full = fisher_proposal(
            center, fisher, pilot, cache, broadening);
        std::vector<uint8_t> retained(components, 0);
        for (const int32_t c : out.candidates[d]) retained[c] = 1;
        double omitted = 0.0;
        for (Eigen::Index j = 0; j < full.weights.size(); ++j) {
            if (!retained[full.component_ids[j]]) {
                omitted += full.weights(j);
            }
        }
        out.maximum_audit_omitted_mass = std::max(
            out.maximum_audit_omitted_mass, omitted);
        if (omitted > options.proposal_proxy_tail_mass * (1.0 + 1e-8)) {
            ++out.audit_violations;
        }
    }
    double mean_candidates = 0.0;
    for (const auto& candidates : out.candidates) {
        mean_candidates += candidates.size();
    }
    mean_candidates /= std::max(1, documents);
    out.predicted_work_ratio = 1.0 / std::max(1, dimension)
        + mean_candidates / out.active_components;
    const int32_t audit_count =
        static_cast<int32_t>(out.audit_documents.size());
    const bool audit_passed = audit_count > 0
        && audit_covers_represented
        && out.audit_violations == 0;
    out.enabled = audit_passed
        && (options.mode == ComponentScreeningMode::On
            || out.predicted_work_ratio
                <= 1.0 - options.minimum_work_reduction);
    out.planning_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - planning_start).count();
    return out;
}

} // namespace uac::detail

namespace uac {

FisherApproximation fisher_approximation(
    const Eigen::Ref<const Eigen::VectorXd>& coordinate,
    const Document& document, const Basis& basis,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert,
    ProposalKind proposal) {
    return detail::fisher_approximation_impl(
        coordinate, document, basis, helmert, proposal);
}

} // namespace uac
