#include "clustering/uac.hpp"
#include "clustering/uac_common_internal.hpp"
#include "clustering/uac_stream.hpp"
#include "punkst.h"

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <filesystem>
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

struct StreamingCliOptions {
    std::string engine = "batch";
    std::string counts = "source";
    std::string cache;
    int32_t block_documents = 64;
    std::string particle_storage = "positions";
    bool rebuild = false;
};

void add_streaming_options(ParamList& pl, StreamingCliOptions& options) {
    pl.add_option("particle-engine",
          "Particle engine: batch or stream", options.engine)
      .add_option("stream-counts",
          "Streaming count storage: source or memory", options.counts)
      .add_option("stream-cache",
          "Persistent streaming particle cache directory", options.cache)
      .add_option("stream-block-documents",
          "Maximum documents in each streaming worker I/O block",
          options.block_documents)
      .add_option("stream-particle-storage",
          "Streaming cache representation: auto, factors, or positions",
          options.particle_storage)
      .add_option("stream-cache-rebuild",
          "Rebuild the cache entry for the current inputs",
          options.rebuild);
}

uac::StreamingOptions make_streaming_options(
    const StreamingCliOptions& input, const std::string& out_prefix) {
    uac::StreamingOptions out;
    out.cache_directory = input.cache.empty()
        ? out_prefix + ".uac-cache" : input.cache;
    out.block_documents = input.block_documents;
    out.count_storage = uac::parse_streaming_count_storage(input.counts);
    out.particle_storage =
        uac::parse_streaming_particle_storage(input.particle_storage);
    out.rebuild_cache = input.rebuild;
    return out;
}

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
    if (responsibility) {
        out.responsibility_se_target = input.responsibility_epsilon;
    }
    if (moment) out.moment_ess_target = input.moment_ess;
    out.calibration_particles = input.calibration_particles;
    out.minimum_particles = input.minimum_particles;
    out.plausible_mass = input.plausible_mass;
    out.plausible_responsibility = input.plausible_responsibility;
    if (out.enabled() && (out.calibration_particles < 2
            || out.minimum_particles < out.calibration_particles
            || maximum_particles < out.minimum_particles
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

struct IndexedParticleDataset {
    uac::Dataset data;
    std::unique_ptr<uac::IndexedDocumentSource> counts;
    bool weighted_counts = false;
};

std::filesystem::path count_spool_path(
    const std::string& cache_directory) {
    const std::filesystem::path root = cache_directory.empty()
        ? std::filesystem::path(".uac-cache")
        : std::filesystem::path(cache_directory);
    std::filesystem::create_directories(root);
    const auto stamp = std::chrono::steady_clock::now()
        .time_since_epoch().count();
    for (int32_t attempt = 0; attempt < 1000; ++attempt) {
        const auto path = root / (".counts-"
            + std::to_string(stamp) + "-" + std::to_string(attempt)
            + ".bin");
        if (!std::filesystem::exists(path)
            && !std::filesystem::exists(path.string() + ".partial")) {
            return path;
        }
    }
    throw std::runtime_error(
        "Cannot allocate a unique UAC count spool path");
}

IndexedParticleDataset load_indexed_particle_dataset(
    const CenterTable& centers, const uac::Basis& basis,
    const CountInputOptions& options,
    const Eigen::VectorXd& feature_weights,
    const std::string& cache_directory) {
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge;
    const bool use_10x = initHexOrDgeInput(reader, dge,
        options.in_file, options.meta_file, options.dge_dirs,
        options.barcodes, options.features, options.matrices,
        options.dataset_ids, options.keep_barcodes);
    if (use_10x) {
        const int32_t overlap =
            dge->setFeatureIndexRemap(basis.features, false);
        if (overlap == 0) {
            throw std::runtime_error(
                "No count features overlap the UAC basis");
        }
    } else {
        std::vector<std::string> model_features = basis.features;
        reader.setFeatureIndexRemap(model_features, false);
    }

    std::unordered_map<std::string, int32_t> center_index;
    center_index.reserve(centers.identifiers.size());
    for (int32_t d = 0;
            d < static_cast<int32_t>(centers.identifiers.size()); ++d) {
        center_index[centers.identifiers[d]] = d;
    }
    const std::filesystem::path spool_path =
        count_spool_path(cache_directory);
    uac::BinaryDocumentSpoolWriter writer(
        spool_path,
        static_cast<int32_t>(basis.probabilities.rows()));
    IndexedParticleDataset out;
    std::vector<int32_t> center_rows;
    std::vector<double> raw_totals, effective_totals;
    std::unordered_set<std::string> count_seen;
    const Eigen::VectorXd* weights = feature_weights.size() > 0
        ? &feature_weights : nullptr;

    auto retain = [&](Document document, std::string identifier) {
        const double input_raw_total = document.get_raw_sum();
        if (input_raw_total < options.min_count) return;
        if (identifier.empty()) {
            identifier = std::to_string(out.data.identifiers.size());
        }
        if (!count_seen.insert(identifier).second) {
            throw std::runtime_error(
                "Duplicate UAC count identifier: " + identifier);
        }
        const auto center = center_index.find(identifier);
        if (center == center_index.end()) {
            throw std::runtime_error(
                "UAC count document has no point center: " + identifier);
        }
        if (!out.weighted_counts) {
            for (const double count : document.cnts) {
                if (std::abs(count - std::round(count)) > 1e-10
                        * std::max(1.0, std::abs(count))) {
                    out.weighted_counts = true;
                    break;
                }
            }
        }
        std::vector<Document> one;
        one.push_back(std::move(document));
        Eigen::VectorXd raw, effective;
        uac::detail::prepare_counts(one,
            static_cast<int32_t>(basis.probabilities.rows()),
            weights, raw, effective);
        writer.append(identifier, one.front(), raw(0), effective(0));
        out.data.identifiers.push_back(std::move(identifier));
        center_rows.push_back(center->second);
        raw_totals.push_back(raw(0));
        effective_totals.push_back(effective(0));
    };

    if (use_10x) {
        Document document;
        int32_t barcode_index = -1;
        std::string identifier;
        while (dge->next(document, &barcode_index, &identifier)) {
            if (barcode_index >= 0) retain(document, identifier);
        }
        dge->resetStream();
    } else {
        std::ifstream input(options.in_file);
        if (!input) {
            throw std::runtime_error(
                "Cannot open UAC count input: " + options.in_file);
        }
        std::string line;
        int32_t retained = 0;
        while (std::getline(input, line)) {
            Document document;
            std::string identifier;
            if (reader.parseLine(
                    document, identifier, line, options.modal, false) < 0) {
                throw std::runtime_error(
                    "Error parsing UAC count row: " + line);
            }
            const size_t before = out.data.identifiers.size();
            retain(std::move(document), std::move(identifier));
            if (out.data.identifiers.size() != before) {
                ++retained;
                if (options.debug > 0 && retained >= options.debug) break;
            }
        }
    }
    if (out.data.identifiers.empty()) {
        throw std::runtime_error(
            "No UAC count documents were loaded");
    }
    if (out.data.identifiers.size() < centers.identifiers.size()) {
        warning("Ignored %zu UAC point centers without retained count documents",
            centers.identifiers.size() - out.data.identifiers.size());
    }
    out.data.centers.resize(
        out.data.identifiers.size(), centers.values.cols());
    out.data.raw_totals.resize(out.data.identifiers.size());
    out.data.effective_totals.resize(out.data.identifiers.size());
    for (int32_t d = 0;
            d < static_cast<int32_t>(out.data.identifiers.size()); ++d) {
        out.data.centers.row(d) = centers.values.row(center_rows[d]);
        out.data.raw_totals(d) = raw_totals[d];
        out.data.effective_totals(d) = effective_totals[d];
    }
    out.data.coordinates = uac::ilr_transform(out.data.centers,
        uac::normalized_helmert(out.data.centers.cols()));
    out.counts = writer.finish();
    out.weighted_counts =
        out.weighted_counts || feature_weights.size() > 0;
    notice("Spool-indexed %zu UAC count documents in %llu bytes",
        out.data.identifiers.size(),
        static_cast<unsigned long long>(out.counts->storage_bytes()));
    return out;
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
    const std::vector<uac::RestartTrace>* traces, int32_t representatives,
    bool write_model_trace = false, int32_t top_c = -1) {
    const Eigen::VectorXd membership =
        score.effective_membership.size() > 0
        ? score.effective_membership
        : effective_membership(score.responsibilities);
    uac::write_state(prefix + ".state.tsv", state);
    uac::write_model(prefix + ".model.tsv", state, &membership);
    uac::write_results(prefix + ".results.tsv", data, score, top_c);
    uac::write_diagnostics(prefix + ".diagnostics.tsv", data, score);
    uac::write_separation(prefix + ".separation.tsv", state.model);
    uac::write_representatives(prefix + ".representatives.tsv", data, score,
        representatives);
    if (traces) uac::write_trace(prefix + ".trace.tsv", *traces);
    if (traces && write_model_trace) {
        uac::write_model_trace(prefix + ".model_trace.tsv", *traces);
    }
}

void report_component_screening(const uac::ScoreResult& score) {
    if (score.component_screening_options.mode
            == uac::ComponentScreeningMode::Off) {
        return;
    }
    notice("UAC component screening requested %s; resolved MAP=%d, proposal=%d, particle=%d, terminal=%d",
        uac::component_screening_mode_name(
            score.component_screening_options.mode),
        static_cast<int32_t>(score.map_component_screening),
        static_cast<int32_t>(score.proposal_component_screening),
        static_cast<int32_t>(score.particle_component_screening),
        static_cast<int32_t>(score.terminal_component_screening));
    if (score.component_screening_options.mode
            == uac::ComponentScreeningMode::On
        && score.component_screening_options.maximum_components > 0) {
        notice("UAC forced component maximum per document: %d",
            score.component_screening_options.maximum_components);
    }
    notice("UAC component work: proposals %lld/%lld; E-step %lld/%lld; proposal audits %d documents covering %d/%d represented components",
        static_cast<long long>(score.proposal_components_constructed),
        static_cast<long long>(score.proposal_components_possible),
        static_cast<long long>(score.evaluated_component_documents),
        static_cast<long long>(score.possible_component_documents),
        score.proposal_audit_documents,
        score.proposal_audit_covered_components,
        score.proposal_audit_represented_components);
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
    constexpr double kCenterFloor = 1e-12;
    std::string center_file, basis_file, out_prefix, particle_initial_state;
    std::string handoff = "particle";
    std::string proposal = "exact_fisher";
    std::string leiden_knn_backend = "auto";
    uac::FitOptions options;
    options.n_components = 0;
    int32_t representatives = 10;
    int32_t top_c = -1;
    bool no_covariance_shrinkage = false, write_model_trace = false;
    CountInputOptions count_options;
    ParticleAdaptOptions particle_adapt;
    ComponentScreeningCliOptions screening;
    StreamingCliOptions streaming;
    screening.mode = "off";
    ParamList pl;
    pl.add_option("in-topic-center", "Document topic point-center table", center_file, true)
      .add_option("in-model", "Feature-by-topic basis table", basis_file)
      .add_option("out-prefix", "Output prefix", out_prefix, true)
      .add_option("handoff", "Handoff: map or particle", handoff)
      .add_option("particle-proposal", "Particle proposal: exact_fisher or sparse_empirical_fisher", proposal)
      .add_option("particles", "Particles per document", options.n_particles)
      .add_option("particle-em-fixed-iterations",
          "Diagnostic fixed number of particle EM E/M pairs; 0 uses convergence stopping",
          options.particle_em_fixed_iterations)
      .add_option("init-ridge",
          "Scalar initialization measurement regularizing precision; 0 uses shared empirical precision",
          options.initialization_ridge_precision)
      .add_option("particle-initial-state",
          "State whose model initializes particle EM; the current-data initializer still supplies the pilot and proposal",
          particle_initial_state)
      .add_option("write-model-trace",
          "Write particle model parameters before each E-step and at termination",
          write_model_trace)
      .add_option("exact-final-score",
          "Evaluate every active component in the terminal scoring pass",
          options.exact_final_score)
      .add_option("top-c",
          "Responsibility pairs in results; 0 writes the legacy dense table, omitted defaults to 5 for screened terminal scores",
          top_c)
      .add_option("cluster-covariance-rank",
          "Cluster covariance rank; -1 uses dense covariance, 0 is diagonal",
          options.cluster_covariance_rank)
      .add_option("fisher-broadening", "Fisher proposal covariance broadening", options.fisher_broadening)
      .add_option("n-clusters", "Fixed number of clusters", options.n_components, true)
      .add_option("kmeans-starts", "Cosine k-means++ initialization starts", options.kmeans_starts)
      .add_option("leiden-starts", "Adaptive cosine-Leiden initialization starts", options.leiden_starts)
      .add_option("max-iter", "Maximum EM iterations", options.max_iterations)
      .add_option("kmeans-max-iter", "Maximum Lloyd/reconciliation iterations", options.kmeans_max_iterations)
      .add_option("leiden-neighbors", "Cosine k-NN neighbors for Leiden starts", options.leiden_neighbors)
      .add_option("leiden-knn-backend", "Cosine k-NN backend: auto, kdtree, or flat", leiden_knn_backend)
      .add_option("leiden-knn-epsilon", "Nanoflann search epsilon; positive values require kdtree", options.leiden_knn_epsilon)
      .add_option("leiden-resolution", "Initial Leiden RBConfiguration resolution", options.leiden_resolution)
      .add_option("leiden-max-iter", "Maximum Leiden passes; negative runs to convergence", options.leiden_max_iterations)
      .add_option("objective-change-tol",
          "Relative objective-change convergence threshold",
          options.objective_change_tolerance)
      .add_option("responsibility-change-tol",
          "Mean maximum document responsibility-change threshold",
          options.responsibility_change_tolerance)
      .add_option("particle-variance-change-tol",
          "Median component absolute relative mean-diagonal variance-change threshold; 0 disables",
          options.particle_variance_change_tolerance)
      .add_option("no-cov-shrinkage",
          "Disable adaptive particle covariance shrinkage",
          no_covariance_shrinkage)
      .add_option("cov-shrinkage-strength",
          "Adaptive covariance shrinkage pseudocount",
          options.covariance_shrinkage_strength)
      .add_option("seed", "Initialization and particle seed", options.seed)
      .add_option("threads", "Number of TBB worker threads", options.n_threads)
      .add_option("n-representatives", "Representatives per cluster", representatives);
    add_count_options(pl, count_options);
    add_particle_adapt_options(pl, particle_adapt);
    add_component_screening_options(pl, screening);
    add_streaming_options(pl, streaming);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        if (top_c < -1) {
            throw std::invalid_argument("--top-c must be nonnegative");
        }
        options.handoff = uac::parse_handoff(handoff);
        options.proposal = uac::parse_proposal(proposal);
        options.particle_engine =
            uac::parse_particle_engine(streaming.engine);
        options.streaming = make_streaming_options(
            streaming, out_prefix);
        options.adaptive_particles = make_particle_adapt_options(
            particle_adapt, options.n_particles);
        options.component_screening =
            make_component_screening_options(screening);
        options.leiden_knn_backend = parse_cosine_knn_backend(
            leiden_knn_backend);
        options.adaptive_covariance_shrinkage = !no_covariance_shrinkage;
        options.iteration_callback = [](const uac::IterationDiagnostic& value) {
            std::ostringstream message;
            message << "UAC " << uac::trace_phase_name(value.phase)
                << " start " << value.start << " after "
                << value.completed_updates
                << " updates: relative objective change ";
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
            message << "; median absolute relative variance change ";
            if (std::isfinite(
                    value.median_absolute_relative_variance_change)) {
                message
                    << value.median_absolute_relative_variance_change;
            } else {
                message << "NA";
            }
            message << "; mean responsibility entropy ";
            if (std::isfinite(value.mean_responsibility_entropy)) {
                message << value.mean_responsibility_entropy;
            } else {
                message << "NA";
            }
            notice("%s", message.str().c_str());
        };
        options.capture_model_trace = write_model_trace;
        if (options.adaptive_particles.enabled()
            && options.handoff != uac::HandoffMode::Particle) {
            throw std::invalid_argument(
                "--particle-adapt-* requires particle handoff");
        }
        CenterTable centers = read_centers(center_file, kCenterFloor);
        uac::Basis basis;
        uac::Basis* basis_pointer = nullptr;
        Eigen::VectorXd feature_weights;
        uac::Dataset data;
        std::unique_ptr<uac::IndexedDocumentSource> indexed_counts;
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
            const bool indexed_source =
                options.particle_engine == uac::ParticleEngine::Stream
                && options.streaming.count_storage
                    == uac::StreamingCountStorage::Source;
            if (indexed_source) {
                IndexedParticleDataset indexed =
                    load_indexed_particle_dataset(
                        centers, basis, count_options, feature_weights,
                        options.streaming.cache_directory);
                data = std::move(indexed.data);
                indexed_counts = std::move(indexed.counts);
                weighted_counts = indexed.weighted_counts;
            } else {
                data = load_particle_dataset(centers, basis, count_options,
                    feature_weights);
                weighted_counts = feature_weights.size() > 0
                    || has_fractional_counts(data.counts);
            }
            basis_pointer = &basis;
        } else {
            data = make_map_dataset(centers);
            if (!basis_file.empty() || !count_options.in_file.empty()
                || !count_options.meta_file.empty()
                || !count_options.dge_dirs.empty()
                || !count_options.barcodes.empty()
                || !count_options.features.empty()
                || !count_options.matrices.empty()
                || !count_options.dataset_ids.empty()
                || !count_options.feature_weight_file.empty()) {
                throw std::invalid_argument(
                    "MAP UAC does not accept model or count inputs");
            }
        }
        if (!particle_initial_state.empty()) {
            if (options.handoff != uac::HandoffMode::Particle) {
                throw std::invalid_argument(
                    "--particle-initial-state requires particle handoff");
            }
            const uac::State initial = uac::read_state(
                particle_initial_state);
            if (initial.basis_checksum != basis.checksum) {
                throw std::invalid_argument(
                    "Particle initial state basis checksum does not match --in-model");
            }
            options.particle_initial_model = initial.model;
        }
        uac::FitResult fitted = indexed_counts
            ? uac::fit_indexed(data, basis, *indexed_counts, options)
            : uac::fit(data, basis_pointer, options);
        uac::StateMetadata state_metadata;
        state_metadata.topics = centers.topics;
        state_metadata.helmert =
            uac::normalized_helmert(state_metadata.topics.size());
        state_metadata.center_floor = kCenterFloor;
        state_metadata.basis_checksum =
            basis_pointer ? basis.checksum : 0;
        state_metadata.feature_weights = feature_weights;
        state_metadata.weighted_counts = weighted_counts;
        uac::State state = uac::make_state(
            fitted, options, state_metadata);
        report_component_screening(fitted.score);
        write_all_outputs(out_prefix, data, state, fitted.score,
            &fitted.traces, representatives, write_model_trace, top_c);
        notice("UAC fitted %d clusters to %zu documents using %s handoff",
            options.n_components, data.identifiers.size(),
            uac::handoff_name(options.handoff));
        notice("UAC outputs written to %s.{state,model,results,diagnostics,trace,separation,representatives}.tsv%s",
            out_prefix.c_str(),
            write_model_trace ? " and .model_trace.tsv" : "");
    } catch (const std::exception& exception) {
        std::cerr << "UAC fit failed: " << exception.what() << "\n";
        return 1;
    }
    return 0;
}

int32_t cmdUacTransform(int argc, char** argv) {
    std::string state_file, center_file, basis_file, out_prefix;
    std::string proposal;
    int32_t particles = 0;
    int32_t threads = 1, representatives = 10;
    int32_t top_c = -1;
    bool exact_final_score = false;
    CountInputOptions count_options;
    ParticleAdaptOptions particle_adapt;
    ComponentScreeningCliOptions screening;
    StreamingCliOptions streaming;
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
      .add_option("n-representatives", "Representatives per cluster", representatives)
      .add_option("exact-final-score",
          "Evaluate every active component in the terminal scoring pass",
          exact_final_score)
      .add_option("top-c",
          "Responsibility pairs in results; 0 writes the legacy dense table, omitted defaults to 5 for screened terminal scores",
          top_c);
    add_count_options(pl, count_options);
    add_particle_adapt_options(pl, particle_adapt);
    add_component_screening_options(pl, screening);
    add_streaming_options(pl, streaming);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        if (top_c < -1) {
            throw std::invalid_argument("--top-c must be nonnegative");
        }
        uac::State state = uac::read_state(state_file);
        const uac::ComponentScreeningOptions component_screening =
            make_component_screening_options(
                screening, state.component_screening);
        uac::ComponentScreeningOptions terminal_screening =
            component_screening;
        if (exact_final_score) {
            terminal_screening.mode =
                uac::ComponentScreeningMode::Off;
            terminal_screening.maximum_components = 0;
        }
        const int32_t scoring_particles = particles > 0
            ? particles : state.n_particles;
        const uac::AdaptiveParticleOptions adaptive_particles =
            make_particle_adapt_options(particle_adapt, scoring_particles);
        const uac::ParticleEngine particle_engine =
            uac::parse_particle_engine(streaming.engine);
        const uac::StreamingOptions streaming_options =
            make_streaming_options(streaming, out_prefix);
        CenterTable centers = read_centers(center_file, state.center_floor);
        align_center_topics(centers, state.topics);
        uac::Dataset data;
        uac::ScoreResult score;
        if (state.handoff == uac::HandoffMode::Map) {
            if (!proposal.empty() || particles > 0
                || particle_engine != uac::ParticleEngine::Batch
                || adaptive_particles.enabled()) {
                throw std::invalid_argument("Particle overrides are invalid for a MAP UAC state");
            }
            data = make_map_dataset(centers);
            score = uac::score_map(
                data, state.model, threads, terminal_screening);
            score.component_screening_options = component_screening;
            score.exact_final_score = exact_final_score;
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
            const uac::ProposalKind scoring_proposal = proposal.empty()
                ? state.proposal : uac::parse_proposal(proposal);
            uac::ParticleScoreOptions score_options;
            score_options.proposal = scoring_proposal;
            score_options.maximum_particles = scoring_particles;
            score_options.adaptive_particles = adaptive_particles;
            score_options.n_threads = threads;
            score_options.particle_engine = particle_engine;
            score_options.streaming = streaming_options;
            score_options.component_screening = component_screening;
            score_options.exact_final_score = exact_final_score;
            const bool indexed_source =
                particle_engine == uac::ParticleEngine::Stream
                && streaming_options.count_storage
                    == uac::StreamingCountStorage::Source;
            if (indexed_source) {
                IndexedParticleDataset indexed =
                    load_indexed_particle_dataset(
                        centers, basis, count_options, weights,
                        streaming_options.cache_directory);
                data = std::move(indexed.data);
                score = uac::score_particle_indexed(
                    data, basis, *indexed.counts, state, score_options);
            } else {
                data = load_particle_dataset(
                    centers, basis, count_options, weights);
                score = uac::score_particle(
                    data, basis, state, score_options);
            }
        }
        report_component_screening(score);
        write_all_outputs(out_prefix, data, state, score, nullptr,
            representatives, false, top_c);
        notice("UAC assigned %zu documents using a fixed %s model",
            data.identifiers.size(), uac::handoff_name(state.handoff));
    } catch (const std::exception& exception) {
        std::cerr << "UAC transform failed: " << exception.what() << "\n";
        return 1;
    }
    return 0;
}
