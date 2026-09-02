#pragma once

#include "numerical_utils.hpp"
#include "nlohmann/json.hpp"

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace punkst::multires {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct ThetaReadOptions {
    int32_t identifier_column = 0;
    // Negative values select contiguous header columns named 0,...,K-1.
    int32_t factor_column_start = -1;
    int32_t factor_column_end = -1;
    // Keep factors whose mean L1-normalized row weight is strictly above this
    // fraction. Zero or a negative value disables filtering.
    double factor_weight_threshold = 1e-5;
    bool normalize_rows = true;
    int32_t minimum_rows = 2;
};

struct FactorFilterDiagnostics {
    double threshold = 1e-5;
    std::vector<std::string> input_factor_names;
    std::vector<int32_t> input_factor_columns;
    std::vector<double> relative_weights;
    std::vector<int32_t> retained_indices;
    std::vector<int32_t> omitted_indices;
};

struct ThetaTable {
    std::vector<std::string> identifiers;
    std::vector<std::string> factor_names;
    std::vector<int32_t> factor_columns;
    RowMajorMatrixXd values;
    FactorFilterDiagnostics factor_filter;
};

ThetaTable read_theta_table(
    const fs::path& path, const ThetaReadOptions& options = ThetaReadOptions());

json array_spec(
    const std::string& path, const std::string& dtype,
    const std::vector<int64_t>& shape);
json write_array(
    const fs::path& root, const std::string& filename,
    const std::vector<int32_t>& values);
json write_array(
    const fs::path& root, const std::string& filename,
    const std::vector<double>& values);
json write_array(
    const fs::path& root, const std::string& filename,
    const std::vector<uint8_t>& values);
json write_array(
    const fs::path& root, const std::string& filename,
    const RowMajorMatrixXd& values);

std::vector<int32_t> read_int32_array(
    const fs::path& root, const json& specification);
std::vector<double> read_float64_array(
    const fs::path& root, const json& specification);
std::vector<uint8_t> read_uint8_array(
    const fs::path& root, const json& specification);
RowMajorMatrixXd read_float64_matrix(
    const fs::path& root, const json& specification);

json read_json(const fs::path& path);
void write_json(const fs::path& path, const json& value, int indent = -1);
void write_json_atomic(
    const fs::path& path, const json& value, int indent = -1);

// The callback must completely populate the supplied temporary directory.
// Publication fails if output already exists and otherwise completes with one
// same-filesystem rename.
void publish_directory_atomic(
    const fs::path& output,
    const std::function<void(const fs::path&)>& writer);

std::string sha256_string(const std::string& value);
std::string sha256_file(const fs::path& path);
std::string artifact_fingerprint(
    const json& manifest, const fs::path& root,
    const std::vector<json>& array_specs);

} // namespace punkst::multires
