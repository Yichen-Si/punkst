#include "lda_state.hpp"

#include "utils.h"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <unordered_set>

void LdaState::validate() const {
    if (!(alpha > 0.0) || !(eta > 0.0) || !std::isfinite(alpha)
            || !std::isfinite(eta) || topics.size() < 2 || features.empty()
            || components.rows() != static_cast<Eigen::Index>(topics.size())
            || components.cols() != static_cast<Eigen::Index>(features.size())
            || !components.allFinite() || (components.array() <= 0.0).any()
            || (feature_weights_active
                && feature_weights.size() != features.size())) {
        throw std::invalid_argument("Invalid LDA state");
    }
    if (has_background
            && (!(background_prior_a > 0.0)
                || !(background_prior_b > 0.0)
                || !std::isfinite(background_prior_a)
                || !std::isfinite(background_prior_b)
                || !(background_count >= 0.0)
                || !(foreground_count >= 0.0)
                || !std::isfinite(background_count)
                || !std::isfinite(foreground_count)
                || background_prior.size()
                    != static_cast<Eigen::Index>(features.size())
                || background_components.size()
                    != static_cast<Eigen::Index>(features.size())
                || !background_prior.allFinite()
                || !background_components.allFinite()
                || (background_prior.array() <= 0.0).any()
                || (background_components.array() <= 0.0).any())) {
        throw std::invalid_argument("Invalid LDA background state");
    }
    std::unordered_set<std::string> topic_set, feature_set;
    for (const std::string& topic : topics) {
        if (topic.empty() || !topic_set.insert(topic).second) {
            throw std::invalid_argument("LDA state topics must be unique");
        }
    }
    for (size_t feature = 0; feature < features.size(); ++feature) {
        if (features[feature].empty()
                || !feature_set.insert(features[feature]).second) {
            throw std::invalid_argument("LDA state features must be unique");
        }
        if (feature_weights_active && (!(feature_weights[feature] >= 0.0)
                || !std::isfinite(feature_weights[feature]))) {
            throw std::invalid_argument("Invalid LDA state feature weight");
        }
    }
}

void LdaState::write(const std::string& path) const {
    validate();
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write LDA state: " + path);
    output << "#lda_state\t" << SCHEMA_VERSION << '\n'
        << "#algorithm\t" << (has_background ? "SVB_DN" : "SVB") << '\n'
        << "#alpha\t" << std::scientific << std::setprecision(17)
        << alpha << '\n'
        << "#eta\t" << eta << '\n'
        << "#feature_weights_active\t"
        << (feature_weights_active ? 1 : 0) << '\n';
    if (has_background) {
        output << "#background_fixed\t" << (background_fixed ? 1 : 0) << '\n'
            << "#background_prior_a\t" << background_prior_a << '\n'
            << "#background_prior_b\t" << background_prior_b << '\n'
            << "#background_count\t" << background_count << '\n'
            << "#foreground_count\t" << foreground_count << '\n';
    }
    for (size_t topic = 0; topic < topics.size(); ++topic) {
        output << "topic\t" << topic << '\t' << topics[topic] << '\n';
    }
    for (Eigen::Index feature = 0; feature < components.cols(); ++feature) {
        output << "feature\t" << feature << '\t'
            << features[static_cast<size_t>(feature)] << '\t'
            << (feature_weights_active
                ? feature_weights[static_cast<size_t>(feature)] : 1.0);
        if (has_background) {
            output << '\t' << background_prior(feature)
                << '\t' << background_components(feature);
        }
        for (Eigen::Index topic = 0; topic < components.rows(); ++topic) {
            output << '\t' << components(topic, feature);
        }
        output << '\n';
    }
}

LdaState LdaState::read(const std::string& path) {
    TextLineReader input(path);
    std::string line;
    bool version_seen = false;
    bool algorithm_seen = false;
    bool weights_flag_seen = false;
    bool background_fixed_seen = false;
    int32_t schema_version = 0;
    std::vector<std::vector<std::string>> feature_rows;
    LdaState state;
    while (input.getline(line)) {
        if (line.empty()) continue;
        const std::vector<std::string> fields = split_delimited(line, '\t');
        if (fields[0] == "#lda_state") {
            if (fields.size() != 2
                    || !str2int32(fields[1], schema_version)
                    || (schema_version != 1
                        && schema_version != SCHEMA_VERSION)) {
                throw std::runtime_error("Unsupported LDA state schema");
            }
            version_seen = true;
        } else if (fields[0] == "#algorithm") {
            if (fields.size() != 2
                    || (fields[1] != "SVB" && fields[1] != "SVB_DN")) {
                throw std::runtime_error("Unsupported LDA state algorithm");
            }
            state.has_background = fields[1] == "SVB_DN";
            algorithm_seen = true;
        } else if (fields[0] == "#alpha" && fields.size() == 2) {
            if (!str2double(fields[1], state.alpha))
                throw std::runtime_error("Invalid LDA state alpha");
        } else if (fields[0] == "#eta" && fields.size() == 2) {
            if (!str2double(fields[1], state.eta))
                throw std::runtime_error("Invalid LDA state eta");
        } else if (fields[0] == "#feature_weights_active"
                && fields.size() == 2) {
            int32_t active = -1;
            if (!str2int32(fields[1], active) || (active != 0 && active != 1)) {
                throw std::runtime_error("Invalid LDA state weight flag");
            }
            state.feature_weights_active = active != 0;
            weights_flag_seen = true;
        } else if (fields[0] == "#background_fixed"
                && fields.size() == 2) {
            int32_t fixed = -1;
            if (!str2int32(fields[1], fixed) || (fixed != 0 && fixed != 1)) {
                throw std::runtime_error("Invalid LDA background fixed flag");
            }
            state.background_fixed = fixed != 0;
            background_fixed_seen = true;
        } else if (fields[0] == "#background_prior_a"
                && fields.size() == 2) {
            if (!str2double(fields[1], state.background_prior_a))
                throw std::runtime_error("Invalid LDA background prior a");
        } else if (fields[0] == "#background_prior_b"
                && fields.size() == 2) {
            if (!str2double(fields[1], state.background_prior_b))
                throw std::runtime_error("Invalid LDA background prior b");
        } else if (fields[0] == "#background_count"
                && fields.size() == 2) {
            if (!str2double(fields[1], state.background_count))
                throw std::runtime_error("Invalid LDA background count");
        } else if (fields[0] == "#foreground_count"
                && fields.size() == 2) {
            if (!str2double(fields[1], state.foreground_count))
                throw std::runtime_error("Invalid LDA foreground count");
        } else if (fields[0] == "topic") {
            int32_t index = -1;
            if (fields.size() != 3 || !str2int32(fields[1], index)
                    || index != static_cast<int32_t>(state.topics.size())) {
                throw std::runtime_error("Invalid LDA state topic row");
            }
            state.topics.push_back(fields[2]);
        } else if (fields[0] == "feature") {
            feature_rows.push_back(fields);
        } else if (fields[0][0] != '#') {
            throw std::runtime_error("Unknown LDA state row: " + fields[0]);
        }
    }
    if (!version_seen || !algorithm_seen || !weights_flag_seen
            || state.topics.empty() || feature_rows.empty()) {
        throw std::runtime_error("Incomplete LDA state: " + path);
    }
    if (schema_version == 1 && state.has_background) {
        throw std::runtime_error("Schema-1 LDA states cannot contain background");
    }
    if (state.has_background && (!background_fixed_seen
            || schema_version != SCHEMA_VERSION)) {
        throw std::runtime_error("Incomplete LDA background state: " + path);
    }
    state.components.resize(state.topics.size(), feature_rows.size());
    if (state.feature_weights_active) {
        state.feature_weights.resize(feature_rows.size());
    }
    if (state.has_background) {
        state.background_prior.resize(feature_rows.size());
        state.background_components.resize(feature_rows.size());
    }
    for (size_t feature = 0; feature < feature_rows.size(); ++feature) {
        const auto& fields = feature_rows[feature];
        int32_t index = -1;
        double weight = 0.0;
        const size_t topic_offset = state.has_background ? 6 : 4;
        if (fields.size() != state.topics.size() + topic_offset
                || !str2int32(fields[1], index)
                || index != static_cast<int32_t>(feature)
                || !str2double(fields[3], weight)) {
            throw std::runtime_error("Invalid LDA state feature row");
        }
        state.features.push_back(fields[2]);
        if (state.feature_weights_active) state.feature_weights[feature] = weight;
        if (state.has_background
                && (!str2double(fields[4], state.background_prior(feature))
                    || !str2double(fields[5],
                        state.background_components(feature)))) {
            throw std::runtime_error("Invalid LDA background component");
        }
        for (size_t topic = 0; topic < state.topics.size(); ++topic) {
            if (!str2double(fields[topic + topic_offset],
                    state.components(topic, feature))) {
                throw std::runtime_error("Invalid LDA state component");
            }
        }
    }
    state.validate();
    return state;
}
