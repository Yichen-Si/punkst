#include "clustering/uac.hpp"
#include "clustering/uac_common_internal.hpp"
#include "clustering/uac_stream.hpp"
#include "punkst.h"
#include "uac_cli_common.hpp"

#include <algorithm>
#include <chrono>
#include <cctype>
#include <climits>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <numeric>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

namespace {

using CenterTable = uac_cli::TopicCenterTable;

uint64_t parse_memory_budget(const std::string& value,
    const char* option) {
    if (value.empty()) {
        throw std::invalid_argument(std::string(option) + " must not be empty");
    }
    std::string digits = value;
    uint64_t multiplier = 1;
    const char suffix = static_cast<char>(std::toupper(
        static_cast<unsigned char>(digits.back())));
    if (suffix == 'K' || suffix == 'M' || suffix == 'G') {
        digits.pop_back();
        multiplier = suffix == 'K' ? 1024ull
            : suffix == 'M' ? 1024ull * 1024ull
                            : 1024ull * 1024ull * 1024ull;
    }
    if (digits.empty()
        || !std::all_of(digits.begin(), digits.end(), [](char value) {
            return std::isdigit(static_cast<unsigned char>(value));
        })) {
        throw std::invalid_argument(
            "Invalid " + std::string(option) + ": " + value);
    }
    uint64_t amount = 0;
    try {
        amount = std::stoull(digits);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "Invalid " + std::string(option) + ": " + value);
    }
    if (amount > std::numeric_limits<uint64_t>::max() / multiplier) {
        throw std::invalid_argument(
            "Invalid " + std::string(option) + ": " + value);
    }
    return amount * multiplier;
}

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
    int32_t identifier_column = 0;
    std::string feature_panel_file;
    bool full_model = false;
    std::string feature_weight_file;
    int32_t weight_column = 1;
    double default_weight = 1.0;
};

struct CountInput {
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge;
    bool use_10x = false;
};

struct PreparedBasis {
    std::unique_ptr<uac::Basis> restricted;
    std::vector<int32_t> canonical_rows;

    const uac::Basis& get(const uac::Basis& canonical) const {
        return restricted ? *restricted : canonical;
    }

    bool is_full() const {
        return !restricted;
    }
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

struct VisualizationCliOptions {
    std::string whitening = "mixture";
    int32_t dimensions = 2;
};

void add_visualization_options(
        ParamList& pl, VisualizationCliOptions& options) {
    pl.add_option("visual-whitening",
          "Visualization whitening covariance: sample or mixture",
          options.whitening)
      .add_option("visual-dim",
          "Maximum visualization dimensions; capped at topics minus one",
          options.dimensions);
}

uac::VisualizationOptions make_visualization_options(
    const VisualizationCliOptions& input, int32_t n_threads,
    double covariance_floor) {
    if (input.dimensions <= 0) {
        throw std::invalid_argument("--visual-dim must be positive");
    }
    if (n_threads <= 0) {
        throw std::invalid_argument("--threads must be positive");
    }
    if (!(covariance_floor > 0.0)
        || !std::isfinite(covariance_floor)) {
        throw std::invalid_argument(
            "UAC visualization covariance floor must be positive and finite");
    }
    uac::VisualizationOptions out;
    out.whitening =
        uac::parse_visualization_whitening(input.whitening);
    out.dimensions = input.dimensions;
    out.n_threads = n_threads;
    out.covariance_floor = covariance_floor;
    return out;
}

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

CenterTable read_centers(const std::string& path, double floor,
    int32_t identifier_column,
    const std::vector<std::string>* expected_topics = nullptr) {
    return uac_cli::read_topic_centers(path, floor, identifier_column,
        expected_topics, "--unit-icol-id");
}

CountInput initialize_count_input(const CountInputOptions& options) {
    CountInput input;
    input.use_10x = initHexOrDgeInput(input.reader, input.dge,
        options.in_file, options.meta_file, options.dge_dirs,
        options.barcodes, options.features, options.matrices,
        options.dataset_ids, options.keep_barcodes);
    if (!input.use_10x
        && (options.identifier_column < 0
            || options.identifier_column >= input.reader.getOffset())) {
        throw std::invalid_argument(
            "--count-icol-id must select a custom-input metadata column");
    }
    return input;
}

std::vector<std::string> read_feature_panel(const std::string& path) {
    TextLineReader reader(path);
    std::string line;
    std::vector<std::string> panel;
    std::unordered_set<std::string> seen;
    while (reader.getline(line)) {
        if (line.empty() || is_comment_line(line)) continue;
        std::vector<std::string> fields;
        split(fields, "\t ", line);
        if (fields.empty() || fields[0].empty()
            || !seen.insert(fields[0]).second) {
            throw std::runtime_error(
                "Empty or duplicate feature in UAC panel: "
                + (fields.empty() ? std::string() : fields[0]));
        }
        panel.push_back(std::move(fields[0]));
    }
    if (panel.empty()) {
        throw std::runtime_error("UAC feature panel is empty: " + path);
    }
    return panel;
}

PreparedBasis prepare_runtime_basis(const uac::Basis& canonical,
    const CountInput& input, const CountInputOptions& options) {
    if (options.full_model && !options.feature_panel_file.empty()) {
        throw std::invalid_argument(
            "--full-model and --feature-panel are mutually exclusive");
    }
    if (options.full_model
        || (options.feature_panel_file.empty()
            && input.reader.features == canonical.features)) {
        return {};
    }
    std::unordered_set<std::string> measured;
    if (!options.feature_panel_file.empty()) {
        const std::vector<std::string> panel =
            read_feature_panel(options.feature_panel_file);
        std::unordered_set<std::string> model_features(
            canonical.features.begin(), canonical.features.end());
        for (const auto& feature : panel) {
            if (model_features.find(feature) == model_features.end()) {
                throw std::runtime_error(
                    "UAC feature panel contains a feature absent from the model: "
                    + feature);
            }
            measured.insert(feature);
        }
    } else {
        measured.insert(input.reader.features.begin(),
            input.reader.features.end());
    }

    PreparedBasis out;
    out.canonical_rows.reserve(canonical.features.size());
    for (int32_t row = 0;
            row < static_cast<int32_t>(canonical.features.size()); ++row) {
        if (measured.find(canonical.features[row]) != measured.end()) {
            out.canonical_rows.push_back(row);
        }
    }
    if (out.canonical_rows.empty()) {
        throw std::runtime_error(
            "No measured count features overlap the UAC model");
    }
    if (out.canonical_rows.size() == canonical.features.size()) {
        out.canonical_rows.clear();
        return out;
    }
    out.restricted = std::make_unique<uac::Basis>();
    uac::Basis& restricted = *out.restricted;
    restricted.probabilities.resize(
        out.canonical_rows.size(), canonical.probabilities.cols());
    restricted.features.reserve(out.canonical_rows.size());
    restricted.topics = canonical.topics;
    for (size_t row = 0; row < out.canonical_rows.size(); ++row) {
        const int32_t source = out.canonical_rows[row];
        restricted.probabilities.row(row) =
            canonical.probabilities.row(source);
        restricted.features.push_back(canonical.features[source]);
    }
    uac::normalize_basis(restricted);
    notice("Conditioned UAC basis on %zu of %zu model features",
        restricted.features.size(), canonical.features.size());
    return out;
}

Eigen::VectorXd project_feature_weights(
    const Eigen::VectorXd& canonical_weights,
    const PreparedBasis& prepared) {
    if (canonical_weights.size() == 0 || prepared.is_full()) {
        return canonical_weights;
    }
    Eigen::VectorXd projected(prepared.canonical_rows.size());
    for (size_t row = 0; row < prepared.canonical_rows.size(); ++row) {
        projected(row) = canonical_weights(prepared.canonical_rows[row]);
    }
    if ((projected.array() == 1.0).all()) return {};
    return projected;
}

std::string select_count_identifier(
    const std::string& metadata, int32_t column) {
    const std::vector<std::string> fields =
        split_delimited(metadata, '\t');
    if (column < 0 || column >= static_cast<int32_t>(fields.size())
        || fields[column].empty()) {
        throw std::runtime_error(
            "Empty or invalid custom-count identifier column");
    }
    return fields[column];
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

uac::Dataset make_map_dataset(const CenterTable& centers,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    uac::Dataset data;
    data.identifiers = centers.identifiers;
    data.centers = centers.values;
    data.coordinates = ilr_transform(data.centers, helmert);
    return data;
}

void configure_count_features(
    CountInput& input, const uac::Basis& basis) {
    if (input.use_10x) {
        const int32_t overlap =
            input.dge->setFeatureIndexRemap(basis.features, false);
        if (overlap == 0) {
            throw std::runtime_error(
                "No count features overlap the UAC runtime basis");
        }
    } else {
        std::vector<std::string> features = basis.features;
        input.reader.setFeatureIndexRemap(features, false);
    }
}

uac::Dataset load_particle_dataset(const CenterTable& centers,
    const uac::Basis& basis, const CountInputOptions& options,
    CountInput& input,
    const Eigen::VectorXd& feature_weights,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
    std::vector<Document> documents;
    std::vector<std::string> identifiers;
    if (input.use_10x) {
        input.dge->readAll(documents, identifiers, options.min_count);
    } else {
        input.reader.readAll(documents, identifiers, options.in_file,
            options.min_count, false,
            options.debug > 0 ? options.debug : INT_MAX, options.modal);
        for (auto& identifier : identifiers) {
            identifier = select_count_identifier(
                identifier, options.identifier_column);
        }
    }
    if (documents.empty()) throw std::runtime_error("No UAC count documents were loaded");
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
    data.coordinates = ilr_transform(data.centers, helmert);
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
    CountInput& input,
    const Eigen::VectorXd& feature_weights,
    const std::string& cache_directory,
    const Eigen::Ref<const Eigen::MatrixXd>& helmert) {
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
            throw std::runtime_error("Empty UAC count identifier");
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

    if (input.use_10x) {
        Document document;
        int32_t barcode_index = -1;
        std::string identifier;
        while (input.dge->next(document, &barcode_index, &identifier)) {
            if (barcode_index >= 0) retain(document, identifier);
        }
        input.dge->resetStream();
    } else {
        std::ifstream stream(options.in_file);
        if (!stream) {
            throw std::runtime_error(
                "Cannot open UAC count input: " + options.in_file);
        }
        std::string line;
        int32_t retained = 0;
        while (std::getline(stream, line)) {
            Document document;
            std::string identifier;
            if (input.reader.parseLine(
                    document, identifier, line, options.modal, false) < 0) {
                throw std::runtime_error(
                    "Error parsing UAC count row: " + line);
            }
            identifier = select_count_identifier(
                identifier, options.identifier_column);
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
    out.data.coordinates = ilr_transform(out.data.centers, helmert);
    out.counts = writer.finish();
    out.weighted_counts =
        out.weighted_counts || feature_weights.size() > 0;
    notice("Spool-indexed %zu UAC count documents in %.4f MB",
        out.data.identifiers.size(),
        (out.counts->storage_bytes()*1e-6));
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
    const uac::VisualizationOptions& visualization_options,
    bool write_model_trace = false, int32_t top_c = -1) {
    const uac::VisualizationResult visualization = uac::make_visualization(
        data, state.model, state.helmert, visualization_options);
    const Eigen::VectorXd membership =
        score.effective_membership.size() > 0
        ? score.effective_membership
        : effective_membership(score.responsibilities);
    uac::write_state(prefix + ".state.tsv", state);
    uac::write_model(prefix + ".model.tsv", state, &membership);
    uac::write_results(prefix + ".results.tsv", data, score, top_c);
    uac::write_diagnostics(prefix + ".diagnostics.tsv", data, score);
    if (score.fit_schedule.schedule == uac::ParticleFitSchedule::Subsample) {
        uac::write_subsample_diagnostics(
            prefix + ".subsample.tsv", score.fit_schedule);
    }
    uac::write_separation(prefix + ".separation.tsv", state.model);
    uac::write_representatives(prefix + ".representatives.tsv", data, score,
        representatives);
    uac::write_visualization_axes(
        prefix + ".visual.axes.tsv", state, visualization);
    uac::write_visualization_model(
        prefix + ".visual.model.tsv", state, visualization);
    uac::write_visualization_results(
        prefix + ".visual.results.tsv", data, visualization);
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
      .add_option("count-icol-id",
          "0-based custom-input metadata column used as the unit identifier",
          options.identifier_column)
      .add_option("feature-panel",
          "Exact measured feature panel, one feature name per row",
          options.feature_panel_file)
      .add_option("full-model",
          "Treat every model feature as measured, including absent input features",
          options.full_model)
      .add_option("feature-weights", "Optional feature-name/weight table", options.feature_weight_file)
      .add_option("icol-weight", "0-based weight column in --feature-weights", options.weight_column)
      .add_option("default-weight", "Weight for model features absent from the weight table", options.default_weight);
}

} // namespace

int32_t cmdUacFit(int argc, char** argv) {
    std::string center_file, basis_file, out_prefix, particle_initial_state;
    std::string handoff = "particle";
    std::string proposal = "exact_fisher";
    std::string particle_fit_schedule = "exact";
    std::string fit_subsample_storage = "auto";
    std::string fit_subsample_memory_budget = "1G";
    std::string fit_tail = "adaptive";
    std::string initialization_measurement_mode = "ht";
    std::string leiden_knn_backend = "auto";
    std::string initialization_metric = "cosine";
    std::string cluster_covariance_diagonal = "component";
    uac::FitOptions options;
    options.n_components = 0;
    double fit_document_budget =
        std::numeric_limits<double>::quiet_NaN();
    int32_t representatives = 10;
    int32_t top_c = -1;
    int32_t unit_identifier_column = 0;
    double center_floor = 1e-12;
    bool no_covariance_shrinkage = false, write_model_trace = false;
    CountInputOptions count_options;
    ParticleAdaptOptions particle_adapt;
    ComponentScreeningCliOptions screening;
    StreamingCliOptions streaming;
    VisualizationCliOptions visualization;
    screening.mode = "off";
    ParamList pl;
    pl.add_option("in-theta", "Document topic-proportion (theta) table", center_file, true)
      .add_option("in-model", "Feature-by-topic basis table", basis_file)
      .add_option("out-prefix", "Output prefix", out_prefix, true)
      .add_option("unit-icol-id",
          "0-based topic-result column used as the unit identifier",
          unit_identifier_column)
      .add_option("center-floor",
          "Positive floor applied to topic centers before row normalization",
          center_floor)
      .add_option("handoff", "Handoff: map or particle", handoff)
      .add_option("particle-proposal", "Particle proposal: exact_fisher or sparse_empirical_fisher", proposal)
      .add_option("particles", "Particles per document", options.n_particles)
      .add_option("particle-em-fixed-iterations",
          "Diagnostic fixed number of particle EM E/M pairs; 0 uses convergence stopping",
          options.particle_em_fixed_iterations)
      .add_option("particle-fit-schedule",
          "Particle fitting schedule: exact, subsample, or online",
          particle_fit_schedule)
      .add_option("fit-document-budget",
          "Approximate fitting work in full-data document-pass equivalents",
          fit_document_budget)
      .add_option("fit-full-tail-updates",
          "Exact full-data EM updates after approximate fitting",
          options.fit_full_tail_updates)
      .add_option("fit-subsample-target",
          "Target Kish effective documents per warm-up responsibility stratum",
          options.fit_subsample_target)
      .add_option("fit-subsample-base-fraction",
          "Uniform inclusion floor for the stratified subsample",
          options.fit_subsample_base_fraction)
      .add_option("fit-subsample-storage",
          "Subsample storage: auto, resident, or disk",
          fit_subsample_storage)
      .add_option("fit-subsample-memory-budget",
          "Additional subsample memory budget in bytes or K/M/G units",
          fit_subsample_memory_budget)
      .add_option("fit-subsample-safety-factor",
          "Safety multiplier for component Kish targets",
          options.fit_subsample_safety_factor)
      .add_option("fit-subsample-min-updates",
          "Minimum stratified subsample EM updates",
          options.fit_subsample_min_updates)
      .add_option("fit-subsample-max-updates",
          "Maximum stratified subsample EM updates",
          options.fit_subsample_max_updates)
      .add_option("fit-subsample-change-tol",
          "Maximum relative parameter change for subsample convergence",
          options.fit_subsample_change_tolerance)
      .add_option("fit-subsample-topup-rounds",
          "Maximum deterministic subsample top-up rounds",
          options.fit_subsample_topup_rounds)
      .add_option("fit-tail",
          "Post-subsample exact tail: adaptive, fixed, or off",
          fit_tail)
      .add_option("fit-batch-documents",
          "Documents per online sufficient-statistic minibatch",
          options.fit_batch_documents)
      .add_option("fit-step-kappa",
          "Online Robbins-Monro exponent in (0.5, 1]",
          options.fit_step_kappa)
      .add_option("fit-step-initial",
          "Initial online Robbins-Monro step in (0, 1]",
          options.fit_step_initial)
      .add_option("init-ridge",
          "Scalar initialization measurement regularizing precision; 0 uses shared empirical precision",
          options.initialization_ridge_precision)
      .add_option("init-measurement-mode",
          "Corrected-moment measurements: legacy, full one-pass cache, or ht",
          initialization_measurement_mode)
      .add_option("init-measurement-target",
          "Expected HT measurement documents per initialization start/component",
          options.initialization_measurement_target)
      .add_option("init-candidate-score-target",
          "Expected candidate-score documents per start/component; negative uses the mode default (full=512, ht=measurement target), 0 uses all documents",
          options.initialization_candidate_score_target)
      .add_option("init-sampling-seed",
          "Initialization sampling seed; negative reuses --seed",
          options.initialization_sampling_seed)
      .add_option("initialization-only",
          "Stop after corrected-moment initialization and write initializer outputs",
          options.initialization_only)
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
      .add_option("cluster-covariance-diagonal",
          "Factor covariance diagonal: component or shared",
          cluster_covariance_diagonal)
      .add_option("fisher-broadening", "Fisher proposal covariance broadening", options.fisher_broadening)
      .add_option("fisher-refinement-iterations",
          "Fisher proposal Newton/Fisher iterations; 1 preserves the legacy one-step proposal",
          options.fisher_refinement_iterations)
      .add_option("n-clusters", "Fixed number of clusters", options.n_components, true)
      .add_option("kmeans-starts", "Metric k-means++ initialization starts", options.kmeans_starts)
      .add_option("leiden-starts", "Adaptive metric-Leiden initialization starts", options.leiden_starts)
      .add_option("initialization-metric",
          "Initialization metric: cosine or hellinger",
          initialization_metric)
      .add_option("max-iter", "Maximum EM iterations", options.max_iterations)
      .add_option("kmeans-max-iter", "Maximum Lloyd/reconciliation iterations", options.kmeans_max_iterations)
      .add_option("leiden-neighbors", "Metric k-NN neighbors for Leiden starts", options.leiden_neighbors)
      .add_option("leiden-knn-backend", "Metric k-NN backend: auto, kdtree, flat, hnsw, or nndescent", leiden_knn_backend)
      .add_option("leiden-knn-epsilon", "Nanoflann search epsilon; positive values require kdtree", options.leiden_knn_epsilon)
      .add_option("hnsw-m", "HNSW graph degree", options.leiden_hnsw_m)
      .add_option("hnsw-ef-construction", "HNSW construction effort", options.leiden_hnsw_ef_construction)
      .add_option("hnsw-ef-search", "HNSW search effort; 0 tunes automatically", options.leiden_hnsw_ef_search)
      .add_option("hnsw-max-ef-search", "Maximum automatically tuned HNSW search effort", options.leiden_hnsw_max_ef_search)
      .add_option("hnsw-candidates", "HNSW candidates per unit; 0 uses max(64,4*k)", options.leiden_hnsw_candidates)
      .add_option("hnsw-audit-queries", "Exact sampled queries for HNSW recall calibration", options.leiden_hnsw_audit_queries)
      .add_option("hnsw-recall", "Required HNSW sampled-recall lower bound", options.leiden_hnsw_recall)
      .add_option("hnsw-force", "Run HNSW despite a failed sampled-recall audit", options.leiden_hnsw_force)
      .add_option("nndescent-iterations", "NN-descent refinements; 0 uses max(10,round(log2(n)))", options.leiden_nndescent_iterations)
      .add_option("nndescent-graph-size", "NN-descent graph size; 0 uses max(64,4*k)", options.leiden_nndescent_graph_size)
      .add_option("nndescent-s", "NN-descent candidate-pool parameter", options.leiden_nndescent_sample_candidates)
      .add_option("nndescent-audit-queries", "Exact sampled queries for NN-descent recall auditing", options.leiden_nndescent_audit_queries)
      .add_option("nndescent-recall", "Required NN-descent sampled-recall lower bound", options.leiden_nndescent_recall)
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
    add_visualization_options(pl, visualization);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        if (top_c < -1) {
            throw std::invalid_argument("--top-c must be nonnegative");
        }
        if (!(center_floor > 0.0) || !std::isfinite(center_floor)) {
            throw std::invalid_argument(
                "--center-floor must be positive and finite");
        }
        const uac::VisualizationOptions visualization_options =
            make_visualization_options(visualization, options.n_threads,
                options.covariance_floor);
        options.handoff = uac::parse_handoff(handoff);
        options.proposal = uac::parse_proposal(proposal);
        options.particle_engine =
            uac::parse_particle_engine(streaming.engine);
        options.particle_fit_schedule =
            uac::parse_particle_fit_schedule(particle_fit_schedule);
        options.fit_document_budget = std::isfinite(fit_document_budget)
            ? fit_document_budget
            : options.particle_fit_schedule == uac::ParticleFitSchedule::Online
                ? 1.0 : 0.0;
        options.fit_subsample_storage =
            uac::parse_subsample_storage(fit_subsample_storage);
        options.fit_subsample_memory_budget = parse_memory_budget(
            fit_subsample_memory_budget, "--fit-subsample-memory-budget");
        options.fit_tail = uac::parse_fit_tail_mode(fit_tail);
        options.initialization_measurement_mode =
            uac::parse_initialization_measurement_mode(
                initialization_measurement_mode);
        options.streaming = make_streaming_options(
            streaming, out_prefix);
        options.adaptive_particles = make_particle_adapt_options(
            particle_adapt, options.n_particles);
        options.component_screening =
            make_component_screening_options(screening);
        options.leiden_knn_backend = parse_cosine_knn_backend(
            leiden_knn_backend);
        options.initialization_metric = parse_simplex_metric(
            initialization_metric);
        options.factor_diagonal_mode = uac::parse_factor_diagonal_mode(
            cluster_covariance_diagonal);
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
        CenterTable centers;
        uac::Basis canonical_basis;
        PreparedBasis prepared_basis;
        const uac::Basis* basis_pointer = nullptr;
        Eigen::VectorXd canonical_feature_weights;
        Eigen::VectorXd runtime_feature_weights;
        uac::Dataset data;
        Eigen::MatrixXd helmert;
        std::unique_ptr<uac::IndexedDocumentSource> indexed_counts;
        bool weighted_counts = false;
        if (options.handoff == uac::HandoffMode::Particle) {
            if (basis_file.empty()) {
                throw std::invalid_argument("Particle UAC requires --in-model");
            }
            canonical_basis = read_basis(basis_file);
            centers = read_centers(center_file, center_floor,
                unit_identifier_column, &canonical_basis.topics);
            helmert = normalized_helmert(
                static_cast<int32_t>(centers.topics.size()));
            CountInput count_input = initialize_count_input(count_options);
            prepared_basis = prepare_runtime_basis(
                canonical_basis, count_input, count_options);
            const uac::Basis& runtime_basis =
                prepared_basis.get(canonical_basis);
            configure_count_features(count_input, runtime_basis);
            canonical_feature_weights = read_feature_weights(
                count_options.feature_weight_file, canonical_basis.features,
                count_options.weight_column, count_options.default_weight);
            runtime_feature_weights = project_feature_weights(
                canonical_feature_weights, prepared_basis);
            const bool indexed_source =
                options.particle_engine == uac::ParticleEngine::Stream
                && options.streaming.count_storage
                    == uac::StreamingCountStorage::Source;
            if (indexed_source) {
                IndexedParticleDataset indexed =
                    load_indexed_particle_dataset(
                        centers, runtime_basis, count_options, count_input,
                        runtime_feature_weights,
                        options.streaming.cache_directory, helmert);
                data = std::move(indexed.data);
                indexed_counts = std::move(indexed.counts);
                weighted_counts = indexed.weighted_counts;
            } else {
                data = load_particle_dataset(centers, runtime_basis,
                    count_options, count_input, runtime_feature_weights,
                    helmert);
                weighted_counts = canonical_feature_weights.size() > 0
                    || has_fractional_counts(data.counts);
            }
            weighted_counts = weighted_counts
                || canonical_feature_weights.size() > 0;
            basis_pointer = &runtime_basis;
        } else {
            centers = read_centers(center_file, center_floor,
                unit_identifier_column);
            helmert = normalized_helmert(
                static_cast<int32_t>(centers.topics.size()));
            data = make_map_dataset(centers, helmert);
            if (!basis_file.empty() || !count_options.in_file.empty()
                || !count_options.meta_file.empty()
                || !count_options.dge_dirs.empty()
                || !count_options.barcodes.empty()
                || !count_options.features.empty()
                || !count_options.matrices.empty()
                || !count_options.dataset_ids.empty()
                || !count_options.feature_weight_file.empty()
                || !count_options.feature_panel_file.empty()
                || count_options.full_model) {
                throw std::invalid_argument(
                    "MAP UAC does not accept model or count inputs");
            }
        }
        // Dataset construction owns its copy. Releasing the input center table
        // here prevents three simultaneous D-by-K matrices at fit startup.
        centers.values = RowMajorMatrixXd{};
        if (!particle_initial_state.empty()) {
            if (options.handoff != uac::HandoffMode::Particle) {
                throw std::invalid_argument(
                    "--particle-initial-state requires particle handoff");
            }
            const uac::State initial = uac::read_state(
                particle_initial_state);
            if (initial.basis_checksum != canonical_basis.checksum) {
                throw std::invalid_argument(
                    "Particle initial state basis checksum does not match --in-model");
            }
            options.particle_initial_model = initial.model;
        }
        uac::FitResult fitted = indexed_counts
            ? uac::fit_indexed(data, *basis_pointer, *indexed_counts,
                helmert, options)
            : uac::fit(data, basis_pointer, helmert, options);
        if (fitted.has_leiden_knn_diagnostics
                && fitted.leiden_knn_diagnostics.forced) {
            warning("HNSW sampled recall LCB %.6g was below target %.6g; continuing because --hnsw-force was set",
                fitted.leiden_knn_diagnostics.audit_recall_lcb,
                options.leiden_hnsw_recall);
        }
        uac::StateMetadata state_metadata;
        state_metadata.topics = centers.topics;
        state_metadata.helmert = helmert;
        state_metadata.center_floor = center_floor;
        state_metadata.basis_checksum =
            basis_pointer ? canonical_basis.checksum : 0;
        state_metadata.feature_weights = canonical_feature_weights;
        state_metadata.weighted_counts = weighted_counts;
        uac::State state = uac::make_state(
            fitted, options, state_metadata);
        uac::write_initialization_results(
            out_prefix + ".initialization.results.tsv", data,
            fitted.initialization_partitions);
        uac::write_initialization_diagnostics(
            out_prefix + ".initialization.tsv", fitted.initialization);
        if (options.initialization_only) {
            const uac::VisualizationResult visualization =
                uac::make_visualization(data, state.model, state.helmert,
                    visualization_options);
            uac::write_state(out_prefix + ".state.tsv", state);
            uac::write_model(out_prefix + ".model.tsv", state);
            uac::write_separation(out_prefix + ".separation.tsv", state.model);
            uac::write_visualization_axes(
                out_prefix + ".visual.axes.tsv", state, visualization);
            uac::write_visualization_model(
                out_prefix + ".visual.model.tsv", state, visualization);
            uac::write_visualization_results(
                out_prefix + ".visual.results.tsv", data, visualization);
            uac::write_trace(out_prefix + ".trace.tsv", fitted.traces);
            if (write_model_trace) {
                uac::write_model_trace(
                    out_prefix + ".model_trace.tsv", fitted.traces);
            }
        } else {
            report_component_screening(fitted.score);
            write_all_outputs(out_prefix, data, state, fitted.score,
                &fitted.traces, representatives, visualization_options,
                write_model_trace, top_c);
        }
        notice("UAC fitted %d clusters to %zu documents using %s handoff",
            options.n_components, data.identifiers.size(),
            uac::handoff_name(options.handoff));
        if (options.initialization_only) {
            notice("UAC initializer outputs written to %s.{state,model,separation,trace,visual.axes,visual.model,visual.results,initialization,initialization.results}.tsv%s",
                out_prefix.c_str(),
                write_model_trace ? " and .model_trace.tsv" : "");
        } else {
            notice("UAC outputs written to %s.{state,model,results,diagnostics,trace,separation,representatives,visual.axes,visual.model,visual.results,initialization,initialization.results}.tsv%s",
                out_prefix.c_str(),
                write_model_trace ? " and .model_trace.tsv" : "");
        }
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
    int32_t fisher_refinement_iterations = 0;
    int32_t threads = 1, representatives = 10;
    int32_t top_c = -1;
    int32_t unit_identifier_column = 0;
    bool exact_final_score = false;
    CountInputOptions count_options;
    ParticleAdaptOptions particle_adapt;
    ComponentScreeningCliOptions screening;
    StreamingCliOptions streaming;
    VisualizationCliOptions visualization;
    ParamList pl;
    pl.add_option("in-state", "Fitted UAC state", state_file, true)
      .add_option("in-theta", "Document topic-proportion (theta) table", center_file, true)
      .add_option("in-model", "Feature-by-topic basis table", basis_file)
      .add_option("out-prefix", "Output prefix", out_prefix, true)
      .add_option("unit-icol-id",
          "0-based topic-result column used as the unit identifier",
          unit_identifier_column)
      .add_option("particle-proposal",
          "Scoring proposal override: exact_fisher or sparse_empirical_fisher",
          proposal)
      .add_option("particles", "Scoring particle-count override", particles)
      .add_option("fisher-refinement-iterations",
          "Scoring Fisher-refinement override; 0 uses the fitted state",
          fisher_refinement_iterations)
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
    add_visualization_options(pl, visualization);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
        if (top_c < -1) {
            throw std::invalid_argument("--top-c must be nonnegative");
        }
        if (fisher_refinement_iterations < 0) {
            throw std::invalid_argument(
                "--fisher-refinement-iterations must be nonnegative");
        }
        uac::State state = uac::read_state(state_file);
        const uac::VisualizationOptions visualization_options =
            make_visualization_options(visualization, threads,
                state.covariance_floor);
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
        CenterTable centers = read_centers(center_file, state.center_floor,
            unit_identifier_column, &state.topics);
        uac::Dataset data;
        uac::ScoreResult score;
        if (state.handoff == uac::HandoffMode::Map) {
            if (!proposal.empty() || particles > 0
                || fisher_refinement_iterations > 0
                || particle_engine != uac::ParticleEngine::Batch
                || adaptive_particles.enabled()) {
                throw std::invalid_argument("Particle overrides are invalid for a MAP UAC state");
            }
            data = make_map_dataset(centers, state.helmert);
            score = uac::score_map(
                data, state.model, threads, terminal_screening);
            score.component_screening_options = component_screening;
            score.exact_final_score = exact_final_score;
        } else {
            if (basis_file.empty()) {
                throw std::invalid_argument("Particle UAC transform requires --in-model");
            }
            uac::Basis canonical_basis = read_basis(basis_file);
            if (canonical_basis.checksum != state.basis_checksum) {
                throw std::runtime_error("UAC basis checksum does not match fitted state");
            }
            if (canonical_basis.topics != state.topics) {
                throw std::runtime_error(
                    "UAC basis topics do not match fitted state");
            }
            if (state.feature_weights.size() > 0
                && state.feature_weights.size()
                    != canonical_basis.probabilities.rows()) {
                throw std::runtime_error("UAC state feature-weight dimension mismatch");
            }
            if (!count_options.feature_weight_file.empty()) {
                warning("Particle transform uses feature weights stored in the UAC state; --feature-weights is ignored");
            }
            CountInput count_input = initialize_count_input(count_options);
            PreparedBasis prepared_basis = prepare_runtime_basis(
                canonical_basis, count_input, count_options);
            const uac::Basis& runtime_basis =
                prepared_basis.get(canonical_basis);
            configure_count_features(count_input, runtime_basis);
            Eigen::VectorXd runtime_weights = project_feature_weights(
                state.feature_weights, prepared_basis);
            uac::State runtime_state;
            const uac::State* scoring_state = &state;
            if (!prepared_basis.is_full()
                || fisher_refinement_iterations > 0) {
                runtime_state = state;
                runtime_state.basis_checksum = runtime_basis.checksum;
                runtime_state.feature_weights = runtime_weights;
                if (fisher_refinement_iterations > 0) {
                    runtime_state.fisher_refinement_iterations =
                        fisher_refinement_iterations;
                }
                scoring_state = &runtime_state;
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
                        centers, runtime_basis, count_options, count_input,
                        runtime_weights,
                        streaming_options.cache_directory, state.helmert);
                data = std::move(indexed.data);
                score = uac::score_particle_indexed(
                    data, runtime_basis, *indexed.counts,
                    *scoring_state, score_options);
            } else {
                data = load_particle_dataset(
                    centers, runtime_basis, count_options, count_input,
                    runtime_weights, state.helmert);
                score = uac::score_particle(
                    data, runtime_basis, *scoring_state, score_options);
            }
        }
        report_component_screening(score);
        write_all_outputs(out_prefix, data, state, score, nullptr,
            representatives, visualization_options, false, top_c);
        notice("UAC assigned %zu documents using a fixed %s model",
            data.identifiers.size(), uac::handoff_name(state.handoff));
    } catch (const std::exception& exception) {
        std::cerr << "UAC transform failed: " << exception.what() << "\n";
        return 1;
    }
    return 0;
}
