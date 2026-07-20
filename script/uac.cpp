#include "clustering/uac.hpp"
#include "punkst.h"

#include <algorithm>
#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
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
    Eigen::VectorXd weights = Eigen::VectorXd::Constant(features.size(),
        default_weight);
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
        weights(found->second) = value;
        ++overlap;
    }
    if (overlap == 0) {
        throw std::runtime_error("No UAC feature weights overlap the model basis");
    }
    notice("Read %d UAC feature weights for %zu model features", overlap,
        features.size());
    return weights;
}

void validate_and_weight_counts(std::vector<Document>& documents,
    const Eigen::VectorXd& weights, Eigen::VectorXd& raw_totals,
    Eigen::VectorXd& effective_totals) {
    raw_totals.resize(documents.size());
    effective_totals.resize(documents.size());
    for (size_t d = 0; d < documents.size(); ++d) {
        Document& document = documents[d];
        double raw = 0.0, effective = 0.0;
        for (size_t j = 0; j < document.ids.size(); ++j) {
            const uint32_t feature = document.ids[j];
            const double count = document.cnts[j];
            if (feature >= static_cast<uint32_t>(weights.size())
                || !std::isfinite(count) || count < 0.0) {
                throw std::runtime_error("Invalid UAC count or feature index");
            }
            raw += count;
            document.cnts[j] *= weights(feature);
            effective += document.cnts[j];
        }
        if (!(effective > 0.0) || !std::isfinite(effective)) {
            throw std::runtime_error("UAC document has zero/nonfinite effective total");
        }
        document.raw_ct_tot = raw;
        document.ct_tot = effective;
        document.counts_weighted = (weights.array() != 1.0).any();
        raw_totals(d) = raw;
        effective_totals(d) = effective;
    }
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
    validate_and_weight_counts(data.counts, feature_weights, data.raw_totals,
        data.effective_totals);
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
    int32_t components = 0, particles = 256, restarts = 5;
    int32_t max_iterations = 300, seed = 1, threads = 1;
    int32_t representatives = 10;
    double tolerance = 1e-6, fisher_broadening = 1.5;
    CountInputOptions count_options;
    ParamList pl;
    pl.add_option("in-topic-center", "Document topic point-center table", center_file, true)
      .add_option("in-model", "Feature-by-topic basis table", basis_file)
      .add_option("out-prefix", "Output prefix", out_prefix, true)
      .add_option("handoff", "Handoff: map or particle", handoff)
      .add_option("particle-proposal", "Particle proposal: exact_fisher or sparse_empirical_fisher", proposal)
      .add_option("particles", "Particles per document", particles)
      .add_option("fisher-broadening", "Fisher proposal covariance broadening", fisher_broadening)
      .add_option("n-clusters", "Fixed number of clusters", components, true)
      .add_option("restarts", "Deterministic mixture restarts", restarts)
      .add_option("max-iter", "Maximum EM iterations", max_iterations)
      .add_option("tol", "Relative objective convergence tolerance", tolerance)
      .add_option("seed", "Initialization and particle seed", seed)
      .add_option("threads", "Number of TBB worker threads", threads)
      .add_option("n-representatives", "Representatives per cluster", representatives);
    add_count_options(pl, count_options);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        uac::FitOptions options;
        options.handoff = uac::parse_handoff(handoff);
        options.proposal = uac::parse_proposal(proposal);
        options.n_components = components;
        options.n_particles = particles;
        options.restarts = restarts;
        options.max_iterations = max_iterations;
        options.tolerance = tolerance;
        options.seed = seed;
        options.n_threads = threads;
        options.fisher_broadening = fisher_broadening;
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
            weighted_counts = (feature_weights.array() != 1.0).any()
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
    int32_t particles = 0, threads = 1, representatives = 10;
    CountInputOptions count_options;
    ParamList pl;
    pl.add_option("in-state", "Fitted UAC state", state_file, true)
      .add_option("in-topic-center", "Document topic point-center table", center_file, true)
      .add_option("in-model", "Feature-by-topic basis table", basis_file)
      .add_option("out-prefix", "Output prefix", out_prefix, true)
      .add_option("particle-proposal",
          "Scoring proposal override: exact_fisher or sparse_empirical_fisher",
          proposal)
      .add_option("particles", "Scoring particle-count override", particles)
      .add_option("threads", "Number of TBB worker threads", threads)
      .add_option("n-representatives", "Representatives per cluster", representatives);
    add_count_options(pl, count_options);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        uac::State state = uac::read_state(state_file);
        CenterTable centers = read_centers(center_file, state.center_floor);
        align_center_topics(centers, state.topics);
        uac::Dataset data;
        uac::ScoreResult score;
        if (state.handoff == uac::HandoffMode::Map) {
            if (!proposal.empty() || particles > 0) {
                throw std::invalid_argument("Particle overrides are invalid for a MAP UAC state");
            }
            data = make_map_dataset(centers);
            score = uac::score_map(data, state.model, threads);
        } else {
            if (basis_file.empty()) {
                throw std::invalid_argument("Particle UAC transform requires --in-model");
            }
            uac::Basis basis = read_basis(basis_file);
            if (basis.checksum != state.basis_checksum) {
                throw std::runtime_error("UAC basis checksum does not match fitted state");
            }
            Eigen::VectorXd weights = state.feature_weights;
            if (weights.size() == 0) weights = Eigen::VectorXd::Ones(basis.features.size());
            if (weights.size() != basis.probabilities.rows()) {
                throw std::runtime_error("UAC state feature-weight dimension mismatch");
            }
            if (!count_options.feature_weight_file.empty()) {
                warning("Particle transform uses feature weights stored in the UAC state; --feature-weights is ignored");
            }
            data = load_particle_dataset(centers, basis, count_options, weights);
            const uac::ProposalKind scoring_proposal = proposal.empty()
                ? state.proposal : uac::parse_proposal(proposal);
            const int32_t scoring_particles = particles > 0
                ? particles : state.n_particles;
            score = uac::score_particle(data, basis, state, scoring_proposal,
                scoring_particles, threads);
        }
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
