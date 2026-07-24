#include "clustering/uac.hpp"
#include "punkst.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace {

struct CenterTable {
    std::vector<std::string> identifiers;
    std::vector<std::string> topics;
    RowMajorMatrixXd values;
};

struct CountInputOptions {
    std::string in_file;
    std::string meta_file;
    std::vector<std::string> dge_dirs;
    std::vector<std::string> barcodes;
    std::vector<std::string> features;
    std::vector<std::string> matrices;
    std::vector<std::string> dataset_ids;
    bool keep_barcodes = false;
    int32_t modal = 0;
    int32_t min_count = 1;
    int32_t debug = 0;
    std::string feature_weight_file;
    int32_t weight_column = 1;
    double default_weight = 1.0;
};

struct ParticleAdaptOptions {
    double responsibility_epsilon = 0.0;
    double moment_ess = 0.0;
    int32_t calibration_particles = 32;
    int32_t minimum_particles = 32;
    double plausible_mass = 0.95;
    double plausible_responsibility = 0.05;
};

struct ComponentScreeningCliOptions {
    std::string mode;
    double tail_mass = -1.0;
    double proposal_tail_mass = -1.0;
    int32_t minimum_components = -1;
    int32_t maximum_components = -1;
    int32_t audit_documents = -1;
    double minimum_work_reduction = -1.0;
};

void add_component_screening_options(ParamList& pl,
    ComponentScreeningCliOptions& options) {
    pl.add_option("component-screening",
          "Component screening: off, on, or auto", options.mode)
      .add_option("component-tail-mass",
          "Maximum bounded E-step responsibility mass to omit",
          options.tail_mass)
      .add_option("proposal-tail-mass",
          "Maximum pilot-proxy proposal mass to omit",
          options.proposal_tail_mass)
      .add_option("component-minimum",
          "Minimum active components retained per document",
          options.minimum_components)
      .add_option("component-maximum",
          "Maximum components retained per document in forced-on mode; 0 is unlimited",
          options.maximum_components)
      .add_option("component-audit-documents",
          "Stratified full-proposal audit documents; 0 chooses automatically",
          options.audit_documents)
      .add_option("component-min-work-reduction",
          "Minimum predicted work reduction required by auto",
          options.minimum_work_reduction);
}

uac::ComponentScreeningOptions make_component_screening_options(
    const ComponentScreeningCliOptions& input,
    uac::ComponentScreeningOptions out = {}) {
    if (!input.mode.empty()) {
        out.mode = uac::parse_component_screening_mode(input.mode);
    }
    if (input.tail_mass >= 0.0) out.tail_mass = input.tail_mass;
    if (input.proposal_tail_mass >= 0.0) {
        out.proposal_proxy_tail_mass = input.proposal_tail_mass;
    }
    if (input.minimum_components >= 0) {
        out.minimum_components = input.minimum_components;
    }
    if (input.maximum_components >= 0) {
        out.maximum_components = input.maximum_components;
    }
    if (input.audit_documents >= 0) {
        out.audit_documents = input.audit_documents;
    }
    if (input.minimum_work_reduction >= 0.0) {
        out.minimum_work_reduction = input.minimum_work_reduction;
    }
    return out;
}

void add_particle_adapt_options(ParamList& pl, ParticleAdaptOptions& options) {
    pl.add_option("particle-adapt-resp",
          "Enable responsibility adaptation with target standard error",
          options.responsibility_epsilon)
      .add_option("particle-adapt-moment",
          "Enable moment adaptation with target conditional ESS",
          options.moment_ess)
      .add_option("particle-adapt-calibration",
          "Reusable calibration particles per document",
          options.calibration_particles)
      .add_option("particle-adapt-min",
          "Minimum retained particles per document",
          options.minimum_particles)
      .add_option("particle-adapt-plausible-mass",
          "Cumulative responsibility mass defining plausible clusters",
          options.plausible_mass)
      .add_option("particle-adapt-plausible-resp",
          "Responsibility threshold defining plausible clusters",
          options.plausible_responsibility);
}

uac::AdaptiveParticleOptions make_particle_adapt_options(
    const ParticleAdaptOptions& input, int32_t maximum_particles) {
    if (input.responsibility_epsilon < 0.0 || input.moment_ess < 0.0) {
        throw std::invalid_argument(
            "Particle adaptation targets cannot be negative");
    }
    uac::AdaptiveParticleOptions out;
    const bool responsibility = input.responsibility_epsilon > 0.0;
    const bool moment = input.moment_ess > 0.0;
    out.enabled = responsibility || moment;
    if (responsibility && moment) {
        out.rule = uac::AdaptiveParticleRule::ResponsibilityMoment;
    } else if (responsibility) {
        out.rule = uac::AdaptiveParticleRule::ResponsibilityOnly;
    } else if (moment) {
        out.rule = uac::AdaptiveParticleRule::MomentOnly;
    }
    if (responsibility) {
        out.responsibility_se_target = input.responsibility_epsilon;
    }
    if (moment) out.moment_ess_target = input.moment_ess;
    out.calibration_particles = input.calibration_particles;
    out.minimum_particles = input.minimum_particles;
    out.maximum_particles = maximum_particles;
    out.plausible_mass = input.plausible_mass;
    out.plausible_responsibility = input.plausible_responsibility;
    if (out.enabled && (out.calibration_particles < 2
            || out.minimum_particles < out.calibration_particles
            || out.maximum_particles < out.minimum_particles
            || !(out.plausible_mass > 0.0 && out.plausible_mass <= 1.0)
            || !(out.plausible_responsibility >= 0.0
                && out.plausible_responsibility <= 1.0))) {
        throw std::invalid_argument(
            "Invalid --particle-adapt-* particle counts or thresholds");
    }
    return out;
}

uac::Basis read_basis(const std::string& path) {
    uac::Basis basis;
    read_matrix_from_file(path, basis.probabilities, &basis.features,
        &basis.topics);
    std::unordered_set<std::string> feature_seen, topic_seen;
    for (const auto& name : basis.features) {
        if (!feature_seen.insert(name).second) {
            throw std::runtime_error("Duplicate feature in UAC basis: " + name);
        }
    }
    for (const auto& name : basis.topics) {
        if (!topic_seen.insert(name).second) {
            throw std::runtime_error("Duplicate topic in UAC basis: " + name);
        }
    }
    uac::normalize_basis(basis);
    return basis;
}

CenterTable read_centers(const std::string& path, double floor) {
    CenterTable table;
    read_matrix_from_file(path, table.values, &table.identifiers,
        &table.topics);
    std::unordered_set<std::string> seen;
    for (const auto& name : table.identifiers) {
        if (name.empty() || !seen.insert(name).second) {
            throw std::runtime_error("Empty or duplicate UAC center identifier: "
                + name);
        }
    }
    uac::normalize_centers(table.values, floor);
    return table;
}

void align_center_topics(CenterTable& centers,
    const std::vector<std::string>& expected) {
    if (centers.topics == expected) return;
    std::unordered_map<std::string, int32_t> index;
    for (int32_t i = 0; i < static_cast<int32_t>(centers.topics.size()); ++i) {
        index[centers.topics[i]] = i;
    }
    if (index.size() != expected.size()) {
        throw std::runtime_error("UAC center topic names do not match the model");
    }
    RowMajorMatrixXd reordered(centers.values.rows(), expected.size());
    for (int32_t topic = 0; topic < static_cast<int32_t>(expected.size()); ++topic) {
        auto found = index.find(expected[topic]);
        if (found == index.end()) {
            throw std::runtime_error("UAC center is missing topic: " + expected[topic]);
        }
        reordered.col(topic) = centers.values.col(found->second);
    }
    centers.values = std::move(reordered);
    centers.topics = expected;
}

Eigen::VectorXd read_feature_weights(const std::string& path,
    const std::vector<std::string>& features, int32_t column,
    double default_weight) {
    if (!(default_weight >= 0.0) || !std::isfinite(default_weight)
        || column < 1) {
        throw std::invalid_argument("Invalid UAC feature-weight options");
    }
    if (path.empty() && default_weight == 1.0) {
        return {};
    }
    Eigen::VectorXd weights = Eigen::VectorXd::Constant(
        features.size(), default_weight);
    int64_t non_unit = default_weight == 1.0
        ? 0 : static_cast<int64_t>(features.size());
    if (path.empty()) return weights;
    std::unordered_map<std::string, int32_t> index;
    for (int32_t i = 0; i < static_cast<int32_t>(features.size()); ++i) {
        index[features[i]] = i;
    }
    std::unordered_set<std::string> seen;
    int32_t overlap = 0;
    TextLineReader reader(path);
    std::string line;
    std::vector<std::string> token;
    while (reader.getline(line)) {
        if (line.empty() || line[0] == '#') continue;
        split(token, "\t ", line);
        if (token.size() <= static_cast<size_t>(column)) {
            throw std::runtime_error("Malformed UAC feature-weight row: " + line);
        }
        auto found = index.find(token[0]);
        if (found == index.end()) continue;
        if (!seen.insert(token[0]).second) {
            throw std::runtime_error("Duplicate UAC feature weight: " + token[0]);
        }
        double value = 0.0;
        if (!str2num(token[column], value) || value < 0.0
            || !std::isfinite(value)) {
            throw std::runtime_error("Invalid UAC feature weight: " + token[column]);
        }
        const double previous = weights(found->second);
        if (previous == 1.0 && value != 1.0) {
            ++non_unit;
        } else if (previous != 1.0 && value == 1.0) {
            --non_unit;
        }
        weights(found->second) = value;
        ++overlap;
    }
    if (overlap == 0) {
        throw std::runtime_error("No UAC feature weights overlap the model basis");
    }
    notice("Read %d UAC feature weights for %zu model features", overlap,
        features.size());
    if (non_unit == 0) return {};
    return weights;
}

uac::Dataset make_map_dataset(const CenterTable& centers) {
    uac::Dataset data;
    data.identifiers = centers.identifiers;
    data.centers = centers.values;
    const Eigen::MatrixXd helmert = uac::normalized_helmert(
        data.centers.cols());
    data.coordinates = uac::ilr_transform(data.centers, helmert);
    return data;
}

uac::Dataset load_particle_dataset(const CenterTable& centers,
    const uac::Basis& basis, const CountInputOptions& options,
    const Eigen::VectorXd& feature_weights) {
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge;
    const bool use_10x = initHexOrDgeInput(reader, dge,
        options.in_file, options.meta_file, options.dge_dirs,
        options.barcodes, options.features, options.matrices,
        options.dataset_ids, options.keep_barcodes);
    std::vector<Document> documents;
    std::vector<std::string> identifiers;
    if (use_10x) {
        const int32_t overlap = dge->setFeatureIndexRemap(basis.features, false);
        if (overlap == 0) throw std::runtime_error("No count features overlap the UAC basis");
        dge->readAll(documents, identifiers, options.min_count);
    } else {
        std::vector<std::string> model_features = basis.features;
        reader.setFeatureIndexRemap(model_features, false);
        reader.readAll(documents, identifiers, options.in_file,
            options.min_count, false,
            options.debug > 0 ? options.debug : INT_MAX, options.modal);
    }
    if (documents.empty()) throw std::runtime_error("No UAC count documents were loaded");
    for (size_t d = 0; d < identifiers.size(); ++d) {
        if (identifiers[d].empty()) identifiers[d] = std::to_string(d);
    }
    std::unordered_map<std::string, int32_t> center_index;
    for (int32_t d = 0; d < static_cast<int32_t>(centers.identifiers.size()); ++d) {
        center_index[centers.identifiers[d]] = d;
    }
    uac::Dataset data;
    data.identifiers = identifiers;
    data.counts = std::move(documents);
    data.centers.resize(data.identifiers.size(), centers.values.cols());
    std::unordered_set<std::string> count_seen;
    for (int32_t d = 0; d < static_cast<int32_t>(data.identifiers.size()); ++d) {
        if (!count_seen.insert(data.identifiers[d]).second) {
            throw std::runtime_error("Duplicate UAC count identifier: " + data.identifiers[d]);
        }
        auto found = center_index.find(data.identifiers[d]);
        if (found == center_index.end()) {
            throw std::runtime_error("UAC count document has no point center: " + data.identifiers[d]);
        }
        data.centers.row(d) = centers.values.row(found->second);
    }
    if (data.identifiers.size() < centers.identifiers.size()) {
        warning("Ignored %zu UAC point centers without retained count documents",
            centers.identifiers.size() - data.identifiers.size());
    }
    const Eigen::VectorXd* weights = feature_weights.size() > 0
        ? &feature_weights : nullptr;
    uac::detail::prepare_counts(data.counts,
        static_cast<int32_t>(basis.probabilities.rows()), weights,
        data.raw_totals, data.effective_totals);
    data.coordinates = uac::ilr_transform(data.centers,
        uac::normalized_helmert(data.centers.cols()));
    return data;
}

Eigen::VectorXd effective_membership(const RowMajorMatrixXd& probability) {
    return probability.colwise().sum();
}

bool has_fractional_counts(const std::vector<Document>& documents) {
    for (const auto& document : documents) {
        for (double count : document.cnts) {
            if (std::abs(count - std::round(count)) > 1e-10
                * std::max(1.0, std::abs(count))) {
                return true;
            }
        }
    }
    return false;
}

void write_all_outputs(const std::string& prefix, const uac::Dataset& data,
    const uac::State& state, const uac::ScoreResult& score,
    const std::vector<uac::RestartTrace>* traces, int32_t representatives) {
    const Eigen::VectorXd membership = effective_membership(score.responsibilities);
    uac::write_state(prefix + ".state.tsv", state);
    uac::write_model(prefix + ".model.tsv", state, &membership);
    uac::write_results(prefix + ".results.tsv", data, score);
    uac::write_diagnostics(prefix + ".diagnostics.tsv", data, score);
    uac::write_separation(prefix + ".separation.tsv", state.model);
    uac::write_representatives(prefix + ".representatives.tsv", data, score,
        representatives);
    if (traces) uac::write_trace(prefix + ".trace.tsv", *traces);
}

void report_component_screening(const uac::ScoreResult& score) {
    if (score.component_screening_options.mode
            == uac::ComponentScreeningMode::Off) {
        return;
    }
    notice("UAC component screening requested %s; resolved MAP=%d, proposal=%d, particle=%d",
        uac::component_screening_mode_name(
            score.component_screening_options.mode),
        static_cast<int32_t>(score.map_component_screening),
        static_cast<int32_t>(score.proposal_component_screening),
        static_cast<int32_t>(score.particle_component_screening));
    if (score.component_screening_options.mode
            == uac::ComponentScreeningMode::On
        && score.component_screening_options.maximum_components > 0) {
        notice("UAC forced component maximum per document: %d",
            score.component_screening_options.maximum_components);
    }
    notice("UAC component work: proposals %lld/%lld; E-step %lld/%lld; proposal audits %d",
        static_cast<long long>(score.proposal_components_constructed),
        static_cast<long long>(score.proposal_components_possible),
        static_cast<long long>(score.evaluated_component_documents),
        static_cast<long long>(score.possible_component_documents),
        score.proposal_audit_documents);
    if (score.proposal_audit_violations > 0) {
        warning("UAC proposal screening exceeded its proxy-tail target in %d audit documents (maximum omitted full-proposal mass %.6g)",
            score.proposal_audit_violations,
            score.proposal_audit_maximum_omitted_mass);
    }
    if (score.component_bound_violations > 0) {
        if (score.component_screening_options.mode
                == uac::ComponentScreeningMode::On
            && score.component_screening_options.maximum_components > 0) {
            warning("UAC component upper bound failed numerically for %d documents; the forced component maximum was retained and their omitted-mass bound was set to one",
                score.component_bound_violations);
        } else {
            warning("UAC component upper bound failed numerically for %d documents; those documents used all components",
                score.component_bound_violations);
        }
    }
}

void add_count_options(ParamList& pl, CountInputOptions& options) {
    pl.add_option("in-data", "Input hex/document file", options.in_file)
      .add_option("in-meta", "Metadata for --in-data", options.meta_file)
      .add_option("in-dge-dir", "Input 10X DGE directory", options.dge_dirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", options.barcodes)
      .add_option("in-features", "Input features.tsv.gz", options.features)
      .add_option("in-matrix", "Input matrix.mtx.gz", options.matrices)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", options.dataset_ids)
      .add_option("keep-barcodes", "Use 10X barcode strings as identifiers", options.keep_barcodes)
      .add_option("modal", "Modality for text input", options.modal)
      .add_option("min-count", "Minimum raw count for retaining a document", options.min_count)
      .add_option("debug", "If positive, retain at most this many text documents", options.debug)
      .add_option("feature-weights", "Optional feature-name/weight table", options.feature_weight_file)
      .add_option("icol-weight", "0-based weight column in --feature-weights", options.weight_column)
      .add_option("default-weight", "Weight for model features absent from the weight table", options.default_weight);
}

} // namespace

int32_t cmdUacFit(int argc, char** argv) {
    std::string center_file, basis_file, out_prefix;
    std::string handoff = "particle", proposal = "exact_fisher";
    std::string leiden_knn_backend = "auto";
    int32_t components = 0, particles = 256, particle_block_size = 0;
    int32_t cluster_covariance_rank = -1;
    int32_t kmeans_starts = 5, leiden_starts = 0;
    int32_t max_iterations = 300, kmeans_max_iterations = 100;
    int32_t leiden_neighbors = 15, leiden_max_iterations = -1;
    int32_t seed = 1, threads = 1;
    int32_t representatives = 10;
    double objective_change_tolerance = 1e-5;
    double responsibility_change_tolerance = 1e-3;
    double covariance_shrinkage_strength = 20.0;
    double fisher_broadening = 1.5;
    double leiden_knn_epsilon = 0.0, leiden_resolution = 1.0;
    bool no_covariance_shrinkage = false;
    CountInputOptions count_options;
    ParticleAdaptOptions particle_adapt;
    ComponentScreeningCliOptions screening;
    screening.mode = "off";
    ParamList pl;
    pl.add_option("in-topic-center", "Document topic point-center table", center_file, true)
      .add_option("in-model", "Feature-by-topic basis table", basis_file)
      .add_option("out-prefix", "Output prefix", out_prefix, true)
      .add_option("handoff", "Handoff: map or particle", handoff)
      .add_option("particle-proposal", "Particle proposal: exact_fisher or sparse_empirical_fisher", proposal)
      .add_option("particles", "Particles per document", particles)
      .add_option("particle-block-size",
          "Documents per regenerated particle block; 0 retains all particles",
          particle_block_size)
      .add_option("cluster-covariance-rank",
          "Cluster covariance rank; -1 uses dense covariance, 0 is diagonal",
          cluster_covariance_rank)
      .add_option("fisher-broadening", "Fisher proposal covariance broadening", fisher_broadening)
      .add_option("n-clusters", "Fixed number of clusters", components, true)
      .add_option("kmeans-starts", "Cosine k-means++ MAP starts", kmeans_starts)
      .add_option("leiden-starts", "Adaptive cosine-Leiden MAP starts", leiden_starts)
      .add_option("max-iter", "Maximum EM iterations", max_iterations)
      .add_option("kmeans-max-iter", "Maximum Lloyd/reconciliation iterations", kmeans_max_iterations)
      .add_option("leiden-neighbors", "Cosine k-NN neighbors for Leiden starts", leiden_neighbors)
      .add_option("leiden-knn-backend", "Cosine k-NN backend: auto, kdtree, or flat", leiden_knn_backend)
      .add_option("leiden-knn-epsilon", "Nanoflann search epsilon; positive values require kdtree", leiden_knn_epsilon)
      .add_option("leiden-resolution", "Initial Leiden RBConfiguration resolution", leiden_resolution)
      .add_option("leiden-max-iter", "Maximum Leiden passes; negative runs to convergence", leiden_max_iterations)
      .add_option("objective-change-tol",
          "Relative objective-change convergence threshold",
          objective_change_tolerance)
      .add_option("responsibility-change-tol",
          "Mean maximum document responsibility-change threshold",
          responsibility_change_tolerance)
      .add_option("no-cov-shrinkage",
          "Disable adaptive particle covariance shrinkage",
          no_covariance_shrinkage)
      .add_option("cov-shrinkage-strength",
          "Adaptive covariance shrinkage pseudocount",
          covariance_shrinkage_strength)
      .add_option("seed", "Initialization and particle seed", seed)
      .add_option("threads", "Number of TBB worker threads", threads)
      .add_option("n-representatives", "Representatives per cluster", representatives);
    add_count_options(pl, count_options);
    add_particle_adapt_options(pl, particle_adapt);
    add_component_screening_options(pl, screening);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        uac::FitOptions options;
        options.handoff = uac::parse_handoff(handoff);
        options.proposal = uac::parse_proposal(proposal);
        options.n_components = components;
        options.n_particles = particles;
        options.adaptive_particles = make_particle_adapt_options(
            particle_adapt, particles);
        options.component_screening =
            make_component_screening_options(screening);
        options.particle_block_size = particle_block_size;
        options.cluster_covariance_rank = cluster_covariance_rank;
        options.kmeans_starts = kmeans_starts;
        options.leiden_starts = leiden_starts;
        options.max_iterations = max_iterations;
        options.kmeans_max_iterations = kmeans_max_iterations;
        options.leiden_neighbors = leiden_neighbors;
        options.leiden_knn_backend = parse_cosine_knn_backend(
            leiden_knn_backend);
        options.leiden_knn_epsilon = leiden_knn_epsilon;
        options.leiden_resolution = leiden_resolution;
        options.leiden_max_iterations = leiden_max_iterations;
        options.objective_change_tolerance = objective_change_tolerance;
        options.responsibility_change_tolerance =
            responsibility_change_tolerance;
        options.adaptive_covariance_shrinkage = !no_covariance_shrinkage;
        options.covariance_shrinkage_strength =
            covariance_shrinkage_strength;
        options.iteration_callback = [](const uac::IterationDiagnostic& value) {
            std::ostringstream message;
            message << "UAC " << (value.particle ? "particle" : "MAP")
                << " start " << value.start << " iteration "
                << value.iteration << ": relative objective change ";
            if (std::isfinite(value.relative_objective_change)) {
                message << value.relative_objective_change;
            } else {
                message << "NA";
            }
            message << "; mean maximum responsibility change ";
            if (std::isfinite(value.mean_max_responsibility_change)) {
                message << value.mean_max_responsibility_change;
            } else {
                message << "NA";
            }
            notice("%s", message.str().c_str());
        };
        options.seed = seed;
        options.n_threads = threads;
        options.fisher_broadening = fisher_broadening;
        if (options.adaptive_particles.enabled
            && options.handoff != uac::HandoffMode::Particle) {
            throw std::invalid_argument(
                "--particle-adapt-* requires particle handoff");
        }
        if (options.adaptive_particles.enabled && particle_block_size != 0) {
            throw std::invalid_argument(
                "--particle-adapt-* cannot use --particle-block-size");
        }
        CenterTable centers = read_centers(center_file, options.center_floor);
        uac::Basis basis;
        uac::Basis* basis_pointer = nullptr;
        Eigen::VectorXd feature_weights;
        uac::Dataset data;
        bool weighted_counts = false;
        if (options.handoff == uac::HandoffMode::Particle) {
            if (basis_file.empty()) {
                throw std::invalid_argument("Particle UAC requires --in-model");
            }
            basis = read_basis(basis_file);
            align_center_topics(centers, basis.topics);
            feature_weights = read_feature_weights(
                count_options.feature_weight_file, basis.features,
                count_options.weight_column, count_options.default_weight);
            data = load_particle_dataset(centers, basis, count_options,
                feature_weights);
            weighted_counts = feature_weights.size() > 0
                || has_fractional_counts(data.counts);
            basis_pointer = &basis;
        } else {
            data = make_map_dataset(centers);
            if (!basis_file.empty() || !count_options.in_file.empty()
                || !count_options.dge_dirs.empty()
                || !count_options.matrices.empty()) {
                warning("MAP UAC ignores --in-model and count inputs");
            }
        }
        uac::FitResult fitted = uac::fit(data, basis_pointer, options);
        uac::State state = uac::make_state(fitted, basis_pointer, options,
            feature_weights, weighted_counts);
        if (!basis_pointer) {
            state.topics = centers.topics;
            state.helmert = uac::normalized_helmert(state.topics.size());
        }
        report_component_screening(fitted.score);
        write_all_outputs(out_prefix, data, state, fitted.score,
            &fitted.traces, representatives);
        notice("UAC fitted %d clusters to %zu documents using %s handoff",
            components, data.identifiers.size(), uac::handoff_name(options.handoff));
        notice("UAC outputs written to %s.{state,model,results,diagnostics,trace,separation,representatives}.tsv",
            out_prefix.c_str());
    } catch (const std::exception& exception) {
        std::cerr << "UAC fit failed: " << exception.what() << "\n";
        return 1;
    }
    return 0;
}

int32_t cmdUacTransform(int argc, char** argv) {
    std::string state_file, center_file, basis_file, out_prefix;
    std::string proposal;
    int32_t particles = 0, particle_block_size = 0;
    int32_t threads = 1, representatives = 10;
    CountInputOptions count_options;
    ParticleAdaptOptions particle_adapt;
    ComponentScreeningCliOptions screening;
    ParamList pl;
    pl.add_option("in-state", "Fitted UAC state", state_file, true)
      .add_option("in-topic-center", "Document topic point-center table", center_file, true)
      .add_option("in-model", "Feature-by-topic basis table", basis_file)
      .add_option("out-prefix", "Output prefix", out_prefix, true)
      .add_option("particle-proposal",
          "Scoring proposal override: exact_fisher or sparse_empirical_fisher",
          proposal)
      .add_option("particles", "Scoring particle-count override", particles)
      .add_option("particle-block-size",
          "Documents per regenerated particle block; 0 retains all particles",
          particle_block_size)
      .add_option("threads", "Number of TBB worker threads", threads)
      .add_option("n-representatives", "Representatives per cluster", representatives);
    add_count_options(pl, count_options);
    add_particle_adapt_options(pl, particle_adapt);
    add_component_screening_options(pl, screening);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        uac::State state = uac::read_state(state_file);
        const uac::ComponentScreeningOptions component_screening =
            make_component_screening_options(
                screening, state.component_screening);
        const int32_t scoring_particles = particles > 0
            ? particles : state.n_particles;
        const uac::AdaptiveParticleOptions adaptive_particles =
            make_particle_adapt_options(particle_adapt, scoring_particles);
        CenterTable centers = read_centers(center_file, state.center_floor);
        align_center_topics(centers, state.topics);
        uac::Dataset data;
        uac::ScoreResult score;
        if (state.handoff == uac::HandoffMode::Map) {
            if (!proposal.empty() || particles > 0 || particle_block_size != 0
                || adaptive_particles.enabled) {
                throw std::invalid_argument("Particle overrides are invalid for a MAP UAC state");
            }
            data = make_map_dataset(centers);
            score = uac::score_map(
                data, state.model, threads, component_screening);
        } else {
            if (basis_file.empty()) {
                throw std::invalid_argument("Particle UAC transform requires --in-model");
            }
            uac::Basis basis = read_basis(basis_file);
            if (basis.checksum != state.basis_checksum) {
                throw std::runtime_error("UAC basis checksum does not match fitted state");
            }
            Eigen::VectorXd weights = state.feature_weights;
            if (weights.size() > 0
                && weights.size() != basis.probabilities.rows()) {
                throw std::runtime_error("UAC state feature-weight dimension mismatch");
            }
            if (!count_options.feature_weight_file.empty()) {
                warning("Particle transform uses feature weights stored in the UAC state; --feature-weights is ignored");
            }
            data = load_particle_dataset(centers, basis, count_options, weights);
            const uac::ProposalKind scoring_proposal = proposal.empty()
                ? state.proposal : uac::parse_proposal(proposal);
            score = uac::score_particle(data, basis, state, scoring_proposal,
                scoring_particles, adaptive_particles, threads,
                particle_block_size, component_screening);
        }
        report_component_screening(score);
        write_all_outputs(out_prefix, data, state, score, nullptr,
            representatives);
        notice("UAC assigned %zu documents using a fixed %s model",
            data.identifiers.size(), uac::handoff_name(state.handoff));
    } catch (const std::exception& exception) {
        std::cerr << "UAC transform failed: " << exception.what() << "\n";
        return 1;
    }
    return 0;
}
