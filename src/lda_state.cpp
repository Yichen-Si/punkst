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
        << "#algorithm\tSVB\n"
        << "#alpha\t" << std::scientific << std::setprecision(17)
        << alpha << '\n'
        << "#eta\t" << eta << '\n'
        << "#feature_weights_active\t"
        << (feature_weights_active ? 1 : 0) << '\n';
    for (size_t topic = 0; topic < topics.size(); ++topic) {
        output << "topic\t" << topic << '\t' << topics[topic] << '\n';
    }
    for (Eigen::Index feature = 0; feature < components.cols(); ++feature) {
        output << "feature\t" << feature << '\t'
            << features[static_cast<size_t>(feature)] << '\t'
            << (feature_weights_active
                ? feature_weights[static_cast<size_t>(feature)] : 1.0);
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
    std::vector<std::vector<std::string>> feature_rows;
    LdaState state;
    while (input.getline(line)) {
        if (line.empty()) continue;
        const std::vector<std::string> fields = split_delimited(line, '\t');
        if (fields[0] == "#lda_state") {
            int32_t version = 0;
            if (fields.size() != 2 || !str2int32(fields[1], version)
                    || version != SCHEMA_VERSION) {
                throw std::runtime_error("Unsupported LDA state schema");
            }
            version_seen = true;
        } else if (fields[0] == "#algorithm") {
            if (fields.size() != 2 || fields[1] != "SVB") {
                throw std::runtime_error(
                    "LDA state uncertainty supports plain SVB only");
            }
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
    state.components.resize(state.topics.size(), feature_rows.size());
    if (state.feature_weights_active) {
        state.feature_weights.resize(feature_rows.size());
    }
    for (size_t feature = 0; feature < feature_rows.size(); ++feature) {
        const auto& fields = feature_rows[feature];
        int32_t index = -1;
        double weight = 0.0;
        if (fields.size() != state.topics.size() + 4
                || !str2int32(fields[1], index)
                || index != static_cast<int32_t>(feature)
                || !str2double(fields[3], weight)) {
            throw std::runtime_error("Invalid LDA state feature row");
        }
        state.features.push_back(fields[2]);
        if (state.feature_weights_active) state.feature_weights[feature] = weight;
        for (size_t topic = 0; topic < state.topics.size(); ++topic) {
            if (!str2double(fields[topic + 4],
                    state.components(topic, feature))) {
                throw std::runtime_error("Invalid LDA state component");
            }
        }
    }
    state.validate();
    return state;
}
