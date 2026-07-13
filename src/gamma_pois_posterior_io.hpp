#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

#include "gamma_pois_topic.hpp"

uint64_t gamma_poisson_state_checksum(const std::string& path);

struct GammaPoissonArtifactId {
    std::array<uint64_t, 2> words{};

    bool empty() const { return words[0] == 0 && words[1] == 0; }
    bool operator==(const GammaPoissonArtifactId& other) const {
        return words == other.words;
    }
    bool operator!=(const GammaPoissonArtifactId& other) const {
        return !(*this == other);
    }
};

GammaPoissonArtifactId generate_gamma_poisson_artifact_id();
std::string gamma_poisson_artifact_id_string(
    const GammaPoissonArtifactId& id);
GammaPoissonArtifactId parse_gamma_poisson_artifact_id(
    const std::string& value);

struct GammaPoissonPosteriorHeader {
    int32_t n_topics = 0;
    uint64_t state_checksum = 0;
    GammaPoissonArtifactId dispersion_sidecar_id;
    std::string row_order = "input";
    std::vector<std::string> identifier_columns;
    std::vector<std::string> topic_names;
};

struct GammaPoissonPosteriorRow {
    uint64_t row = 0;
    std::string identifiers;
    GammaPoissonDocumentPosterior posterior;
};

class GammaPoissonPosteriorReader {
public:
    explicit GammaPoissonPosteriorReader(const std::string& path,
        uint64_t expected_state_checksum = 0);

    const GammaPoissonPosteriorHeader& header() const { return header_; }
    bool read_next(GammaPoissonPosteriorRow& row);

private:
    std::ifstream in_;
    GammaPoissonPosteriorHeader header_;
    std::vector<size_t> identifier_columns_;
    std::vector<size_t> shape_columns_;
    std::vector<size_t> rate_columns_;
    size_t row_column_ = static_cast<size_t>(-1);
    size_t exposure_column_ = static_cast<size_t>(-1);
    uint64_t expected_row_ = 0;
};

class GammaPoissonDispersionWriter {
public:
    GammaPoissonDispersionWriter(const std::string& path, int32_t n_topics,
        int32_t rank, uint64_t state_checksum,
        const GammaPoissonArtifactId& artifact_id);
    GammaPoissonDispersionWriter(const GammaPoissonDispersionWriter&) = delete;
    GammaPoissonDispersionWriter& operator=(const GammaPoissonDispersionWriter&) = delete;
    ~GammaPoissonDispersionWriter();

    void append(const GammaPoissonDispersionApproximation& approximation);
    void close();
    uint64_t record_count() const { return record_count_; }
    int32_t rank() const { return rank_; }
    const GammaPoissonArtifactId& artifact_id() const { return artifact_id_; }

private:
    std::ofstream out_;
    int32_t n_topics_;
    int32_t rank_;
    uint64_t state_checksum_;
    GammaPoissonArtifactId artifact_id_;
    uint64_t record_count_ = 0;
    bool closed_ = false;
};

struct GammaPoissonDispersionHeader {
    int32_t n_topics = 0;
    int32_t rank = 0;
    uint64_t state_checksum = 0;
    uint64_t record_count = 0;
    GammaPoissonArtifactId artifact_id;
};

class GammaPoissonDispersionReader {
public:
    explicit GammaPoissonDispersionReader(const std::string& path,
        uint64_t expected_state_checksum = 0);

    const GammaPoissonDispersionHeader& header() const { return header_; }
    bool read_next(GammaPoissonDispersionApproximation& approximation);

private:
    std::ifstream in_;
    GammaPoissonDispersionHeader header_;
    uint64_t records_read_ = 0;
};
