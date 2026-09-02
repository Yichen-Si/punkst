#include "multires_core/artifacts.hpp"

#include "utils.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <unordered_set>

namespace punkst::multires {

namespace {

constexpr std::array<uint32_t, 64> SHA256_CONSTANTS = {{
    0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u,
    0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
    0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u,
    0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
    0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu,
    0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
    0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
    0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
    0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u,
    0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
    0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u,
    0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
    0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u,
    0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
    0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u,
    0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u
}};

uint32_t rotate_right(uint32_t value, uint32_t bits) {
    return (value >> bits) | (value << (32u - bits));
}

class Sha256 {
public:
    Sha256()
        : state_{{0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u,
                  0xa54ff53au, 0x510e527fu, 0x9b05688cu,
                  0x1f83d9abu, 0x5be0cd19u}} {}

    void update(const uint8_t* data, size_t size) {
        if (size > (std::numeric_limits<uint64_t>::max() - total_bytes_)) {
            throw std::overflow_error("SHA-256 input is too large");
        }
        total_bytes_ += static_cast<uint64_t>(size);
        while (size > 0) {
            const size_t copied = std::min(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, copied);
            block_size_ += copied;
            data += copied;
            size -= copied;
            if (block_size_ == block_.size()) {
                transform(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::string finish() {
        const uint64_t bit_count = total_bytes_ * 8u;
        block_[block_size_++] = 0x80u;
        if (block_size_ > 56) {
            std::fill(block_.begin() + block_size_, block_.end(), 0u);
            transform(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + block_size_, block_.begin() + 56, 0u);
        for (size_t byte = 0; byte < 8; ++byte) {
            block_[63 - byte] = static_cast<uint8_t>(bit_count >> (8u * byte));
        }
        transform(block_.data());
        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (uint32_t value : state_) output << std::setw(8) << value;
        return output.str();
    }

private:
    void transform(const uint8_t* block) {
        std::array<uint32_t, 64> words{};
        for (size_t index = 0; index < 16; ++index) {
            const size_t offset = index * 4;
            words[index] = (static_cast<uint32_t>(block[offset]) << 24u)
                | (static_cast<uint32_t>(block[offset + 1]) << 16u)
                | (static_cast<uint32_t>(block[offset + 2]) << 8u)
                | static_cast<uint32_t>(block[offset + 3]);
        }
        for (size_t index = 16; index < words.size(); ++index) {
            const uint32_t s0 = rotate_right(words[index - 15], 7)
                ^ rotate_right(words[index - 15], 18)
                ^ (words[index - 15] >> 3u);
            const uint32_t s1 = rotate_right(words[index - 2], 17)
                ^ rotate_right(words[index - 2], 19)
                ^ (words[index - 2] >> 10u);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a = state_[0];
        uint32_t b = state_[1];
        uint32_t c = state_[2];
        uint32_t d = state_[3];
        uint32_t e = state_[4];
        uint32_t f = state_[5];
        uint32_t g = state_[6];
        uint32_t h = state_[7];
        for (size_t index = 0; index < words.size(); ++index) {
            const uint32_t sum1 = rotate_right(e, 6) ^ rotate_right(e, 11)
                ^ rotate_right(e, 25);
            const uint32_t choice = (e & f) ^ ((~e) & g);
            const uint32_t temporary1 = h + sum1 + choice
                + SHA256_CONSTANTS[index] + words[index];
            const uint32_t sum0 = rotate_right(a, 2) ^ rotate_right(a, 13)
                ^ rotate_right(a, 22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t temporary2 = sum0 + majority;
            h = g;
            g = f;
            f = e;
            e = d + temporary1;
            d = c;
            c = b;
            b = a;
            a = temporary1 + temporary2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::array<uint32_t, 8> state_;
    std::array<uint8_t, 64> block_{};
    size_t block_size_ = 0;
    uint64_t total_bytes_ = 0;
};

fs::path safe_artifact_path(const fs::path& root, const std::string& value) {
    if (value.empty()) throw std::invalid_argument("Artifact path is empty");
    const fs::path relative(value);
    if (relative.is_absolute()) {
        throw std::invalid_argument("Artifact path must be relative: " + value);
    }
    for (const fs::path& part : relative) {
        if (part == "..") {
            throw std::invalid_argument(
                "Artifact path must stay within its root: " + value);
        }
    }
    return root / relative;
}

void require_little_endian() {
    const uint16_t value = 1;
    if (*reinterpret_cast<const uint8_t*>(&value) != 1) {
        throw std::runtime_error(
            "Multiresolution binary artifacts require a little-endian host");
    }
}

template<class Value>
json write_vector_impl(const fs::path& root, const std::string& filename,
        const std::string& dtype, const std::vector<Value>& values) {
    require_little_endian();
    const fs::path path = safe_artifact_path(root, filename);
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot write array: " + path.string());
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(Value)));
    if (!output) throw std::runtime_error("Failed writing array: " + path.string());
    return array_spec(filename, dtype,
        {static_cast<int64_t>(values.size())});
}

template<class Value>
std::vector<Value> read_vector_impl(const fs::path& root,
        const json& specification, const std::string& dtype) {
    require_little_endian();
    if (!specification.is_object()
            || specification.value("dtype", "") != dtype
            || specification.value("endianness", "") != "little"
            || specification.value("order", "") != "C") {
        throw std::runtime_error("Invalid " + dtype + " array specification");
    }
    const auto shape = specification.at("shape").get<std::vector<int64_t>>();
    if (shape.size() != 1 || shape[0] < 0) {
        throw std::runtime_error("Expected a one-dimensional array");
    }
    const fs::path path = safe_artifact_path(
        root, specification.at("path").get<std::string>());
    const uintmax_t expected = static_cast<uintmax_t>(shape[0]) * sizeof(Value);
    if (!fs::exists(path) || fs::file_size(path) != expected) {
        throw std::runtime_error("Array byte size does not match manifest: "
            + path.string());
    }
    std::vector<Value> values(static_cast<size_t>(shape[0]));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expected));
    if (!input && expected != 0) {
        throw std::runtime_error("Failed reading array: " + path.string());
    }
    return values;
}

bool numbered_column(const std::string& name, int32_t& value) {
    if (name.empty()) return false;
    return str2int32(name, value) && value >= 0;
}

bool topk_column(const std::string& name) {
    if (name.size() < 2 || (name.front() != 'K' && name.front() != 'P')) {
        return false;
    }
    int32_t index = 0;
    return str2int32(name.substr(1), index) && index > 0;
}

std::string temporary_suffix() {
    static std::atomic<uint64_t> counter{0};
    const auto now = std::chrono::high_resolution_clock::now()
        .time_since_epoch().count();
    std::ostringstream output;
    output << now << '-' << std::hash<std::thread::id>{}(
        std::this_thread::get_id()) << '-' << counter.fetch_add(1);
    return output.str();
}

void update_file(Sha256& digest, const fs::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("Cannot open for hashing: " + path.string());
    std::array<char, 1024 * 1024> buffer{};
    while (input) {
        input.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) {
            digest.update(reinterpret_cast<const uint8_t*>(buffer.data()),
                static_cast<size_t>(count));
        }
    }
    if (!input.eof()) throw std::runtime_error("Failed hashing: " + path.string());
}

} // namespace

ThetaTable read_theta_table(const fs::path& path,
        const ThetaReadOptions& options) {
    if (options.identifier_column < 0 || options.minimum_rows < 1
            || !std::isfinite(options.factor_weight_threshold)) {
        throw std::invalid_argument(
            "Theta identifier column, row count, and factor threshold are invalid");
    }
    const bool has_start = options.factor_column_start >= 0;
    const bool has_end = options.factor_column_end >= 0;
    if (has_start != has_end || (has_start
            && options.factor_column_end < options.factor_column_start)) {
        throw std::invalid_argument(
            "Explicit factor columns require a valid inclusive range");
    }

    TextLineReader input(path.string());
    std::string line;
    while (input.getline(line) && line.empty()) {}
    if (line.empty()) throw std::runtime_error("Theta table is empty: " + path.string());
    const std::vector<std::string> header = split_delimited(
        strip_leading_hash(line), '\t');
    if (options.identifier_column >= static_cast<int32_t>(header.size())) {
        throw std::invalid_argument("Theta identifier column is outside the header");
    }
    std::unordered_set<std::string> names;
    bool has_topk = false;
    for (const std::string& name : header) {
        if (name.empty() || !names.insert(name).second) {
            throw std::runtime_error("Theta has an empty or duplicate header: " + name);
        }
        has_topk = has_topk || topk_column(name);
    }
    ThetaTable table;
    if (has_start) {
        if (options.factor_column_end >= static_cast<int32_t>(header.size())) {
            throw std::invalid_argument("Theta factor range is outside the header");
        }
        for (int32_t column = options.factor_column_start;
                column <= options.factor_column_end; ++column) {
            table.factor_columns.push_back(column);
        }
    } else {
        int32_t first_factor = static_cast<int32_t>(header.size());
        while (first_factor > 0) {
            int32_t factor = -1;
            if (!numbered_column(
                    header[static_cast<size_t>(first_factor - 1)], factor)) {
                break;
            }
            --first_factor;
        }
        std::unordered_set<int32_t> factor_names;
        for (int32_t column = first_factor;
                column < static_cast<int32_t>(header.size()); ++column) {
            int32_t factor = -1;
            if (!numbered_column(header[static_cast<size_t>(column)], factor)
                    || !factor_names.insert(factor).second) {
                throw std::runtime_error(
                    "Theta table has duplicate numeric factor names");
            }
            table.factor_columns.push_back(column);
        }
    }
    if (table.factor_columns.size() < 2) {
        if (has_topk && !has_start) {
            throw std::runtime_error(
                "K/P top-k theta input is truncated and unsupported by multiresolution spectra");
        }
        throw std::runtime_error("Multiresolution analysis requires at least two factors");
    }
    if (std::find(table.factor_columns.begin(), table.factor_columns.end(),
            options.identifier_column) != table.factor_columns.end()) {
        throw std::invalid_argument("Theta identifier column cannot be a factor column");
    }
    table.factor_filter.threshold = options.factor_weight_threshold;
    table.factor_filter.input_factor_columns = table.factor_columns;
    for (int32_t column : table.factor_columns) {
        table.factor_names.push_back(header[static_cast<size_t>(column)]);
    }
    table.factor_filter.input_factor_names = table.factor_names;

    std::vector<double> flat;
    std::vector<double> normalized_sums(table.factor_columns.size(), 0.0);
    std::unordered_set<std::string> identifiers;
    uint64_t row_number = 1;
    while (input.getline(line)) {
        ++row_number;
        if (line.empty() || is_comment_line(line)) continue;
        const std::vector<std::string> fields = split_delimited(line, '\t');
        if (fields.size() != header.size()) {
            throw std::runtime_error("Theta row has the wrong field count at line "
                + std::to_string(row_number));
        }
        const std::string& identifier = fields[
            static_cast<size_t>(options.identifier_column)];
        if (identifier.empty() || !identifiers.insert(identifier).second) {
            throw std::runtime_error(
                "Theta has an empty or duplicate identifier at line "
                + std::to_string(row_number));
        }
        table.identifiers.push_back(identifier);
        const size_t row_begin = flat.size();
        double scale = 0.0;
        for (int32_t column : table.factor_columns) {
            double value = 0.0;
            if (!str2double(fields[static_cast<size_t>(column)], value)
                    || !std::isfinite(value) || value < 0.0) {
                throw std::runtime_error("Invalid theta value at line "
                    + std::to_string(row_number));
            }
            flat.push_back(value);
            scale = std::max(scale, value);
        }
        if (!(scale > 0.0)) {
            throw std::runtime_error("Theta contains a zero row at line "
                + std::to_string(row_number));
        }
        double total = 0.0;
        for (size_t index = row_begin; index < flat.size(); ++index) {
            total += flat[index] / scale;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error("Theta row sum is invalid at line "
                + std::to_string(row_number));
        }
        if (options.normalize_rows) {
            for (size_t index = row_begin; index < flat.size(); ++index) {
                flat[index] = flat[index] / scale / total;
            }
        }
        for (size_t factor = 0; factor < table.factor_columns.size(); ++factor) {
            const double value = flat[row_begin + factor];
            normalized_sums[factor] += options.normalize_rows
                ? value : value / scale / total;
        }
    }
    if (table.identifiers.size() < static_cast<size_t>(options.minimum_rows)) {
        throw std::runtime_error("Theta has fewer rows than required");
    }
    const size_t input_factor_count = table.factor_columns.size();
    table.factor_filter.relative_weights.resize(input_factor_count);
    for (size_t factor = 0; factor < input_factor_count; ++factor) {
        const double relative = normalized_sums[factor]
            / static_cast<double>(table.identifiers.size());
        table.factor_filter.relative_weights[factor] = relative;
        if (options.factor_weight_threshold <= 0.0
                || relative > options.factor_weight_threshold) {
            table.factor_filter.retained_indices.push_back(
                static_cast<int32_t>(factor));
        } else {
            table.factor_filter.omitted_indices.push_back(
                static_cast<int32_t>(factor));
        }
    }
    if (table.factor_filter.retained_indices.size() < 2) {
        throw std::runtime_error("Factor-weight filter retained "
            + std::to_string(table.factor_filter.retained_indices.size())
            + " of " + std::to_string(input_factor_count)
            + " factors; at least two are required (--factor-weight-threshold "
            + std::to_string(options.factor_weight_threshold) + ")");
    }

    const std::vector<int32_t> input_columns = table.factor_columns;
    const std::vector<std::string> input_names = table.factor_names;
    table.factor_columns.clear();
    table.factor_names.clear();
    for (int32_t factor : table.factor_filter.retained_indices) {
        table.factor_columns.push_back(input_columns[static_cast<size_t>(factor)]);
        table.factor_names.push_back(input_names[static_cast<size_t>(factor)]);
    }
    table.values.resize(static_cast<Eigen::Index>(table.identifiers.size()),
        static_cast<Eigen::Index>(table.factor_names.size()));
    for (Eigen::Index row = 0; row < table.values.rows(); ++row) {
        double retained_total = 0.0;
        for (size_t output_factor = 0;
                output_factor < table.factor_filter.retained_indices.size();
                ++output_factor) {
            const int32_t input_factor =
                table.factor_filter.retained_indices[output_factor];
            const double value = flat[static_cast<size_t>(row)
                * input_factor_count + static_cast<size_t>(input_factor)];
            table.values(row, static_cast<Eigen::Index>(output_factor)) = value;
            retained_total += value;
        }
        if (!(retained_total > 0.0) || !std::isfinite(retained_total)) {
            throw std::runtime_error("Theta row has no positive mass after factor-weight filtering: "
                + table.identifiers[static_cast<size_t>(row)]);
        }
        if (options.normalize_rows) table.values.row(row) /= retained_total;
    }
    return table;
}

json array_spec(const std::string& path, const std::string& dtype,
        const std::vector<int64_t>& shape) {
    if (path.empty() || (dtype != "int32" && dtype != "float64"
            && dtype != "uint8") || shape.empty()
            || std::any_of(shape.begin(), shape.end(),
                [](int64_t value) { return value < 0; })) {
        throw std::invalid_argument("Invalid binary array specification");
    }
    return {{"path", path}, {"dtype", dtype}, {"endianness", "little"},
        {"order", "C"}, {"shape", shape}};
}

json write_array(const fs::path& root, const std::string& filename,
        const std::vector<int32_t>& values) {
    return write_vector_impl(root, filename, "int32", values);
}

json write_array(const fs::path& root, const std::string& filename,
        const std::vector<double>& values) {
    return write_vector_impl(root, filename, "float64", values);
}

json write_array(const fs::path& root, const std::string& filename,
        const std::vector<uint8_t>& values) {
    return write_vector_impl(root, filename, "uint8", values);
}

json write_array(const fs::path& root, const std::string& filename,
        const RowMajorMatrixXd& values) {
    require_little_endian();
    const fs::path path = safe_artifact_path(root, filename);
    fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot write matrix: " + path.string());
    output.write(reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!output) throw std::runtime_error("Failed writing matrix: " + path.string());
    return array_spec(filename, "float64",
        {static_cast<int64_t>(values.rows()),
         static_cast<int64_t>(values.cols())});
}

std::vector<int32_t> read_int32_array(
        const fs::path& root, const json& specification) {
    return read_vector_impl<int32_t>(root, specification, "int32");
}

std::vector<double> read_float64_array(
        const fs::path& root, const json& specification) {
    return read_vector_impl<double>(root, specification, "float64");
}

std::vector<uint8_t> read_uint8_array(
        const fs::path& root, const json& specification) {
    return read_vector_impl<uint8_t>(root, specification, "uint8");
}

RowMajorMatrixXd read_float64_matrix(
        const fs::path& root, const json& specification) {
    require_little_endian();
    if (!specification.is_object()
            || specification.value("dtype", "") != "float64"
            || specification.value("endianness", "") != "little"
            || specification.value("order", "") != "C") {
        throw std::runtime_error("Invalid float64 matrix specification");
    }
    const auto shape = specification.at("shape").get<std::vector<int64_t>>();
    if (shape.size() != 2 || shape[0] < 0 || shape[1] < 0) {
        throw std::runtime_error("Expected a two-dimensional array");
    }
    if (shape[0] > std::numeric_limits<Eigen::Index>::max()
            || shape[1] > std::numeric_limits<Eigen::Index>::max()) {
        throw std::runtime_error("Matrix shape exceeds Eigen index range");
    }
    const fs::path path = safe_artifact_path(
        root, specification.at("path").get<std::string>());
    const uintmax_t expected = static_cast<uintmax_t>(shape[0])
        * static_cast<uintmax_t>(shape[1]) * sizeof(double);
    if (!fs::exists(path) || fs::file_size(path) != expected) {
        throw std::runtime_error("Matrix byte size does not match manifest: "
            + path.string());
    }
    RowMajorMatrixXd values(static_cast<Eigen::Index>(shape[0]),
        static_cast<Eigen::Index>(shape[1]));
    std::ifstream input(path, std::ios::binary);
    input.read(reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(expected));
    if (!input && expected != 0) {
        throw std::runtime_error("Failed reading matrix: " + path.string());
    }
    return values;
}

json read_json(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot read JSON: " + path.string());
    json value;
    try {
        input >> value;
    } catch (const json::exception& error) {
        throw std::runtime_error("Invalid JSON " + path.string() + ": "
            + error.what());
    }
    if (!value.is_object()) {
        throw std::runtime_error("JSON root must be an object: " + path.string());
    }
    return value;
}

void write_json(const fs::path& path, const json& value, int indent) {
    if (!path.parent_path().empty()) fs::create_directories(path.parent_path());
    std::ofstream output(path, std::ios::binary);
    if (!output) throw std::runtime_error("Cannot write JSON: " + path.string());
    output << value.dump(indent) << '\n';
    if (!output) throw std::runtime_error("Failed writing JSON: " + path.string());
}

void write_json_atomic(const fs::path& path, const json& value, int indent) {
    const fs::path parent = path.parent_path().empty()
        ? fs::current_path() : path.parent_path();
    fs::create_directories(parent);
    const fs::path temporary = parent
        / ("." + path.filename().string() + ".tmp-" + temporary_suffix());
    try {
        write_json(temporary, value, indent);
        fs::rename(temporary, path);
    } catch (...) {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        throw;
    }
}

void publish_directory_atomic(const fs::path& output,
        const std::function<void(const fs::path&)>& writer) {
    if (output.empty()) throw std::invalid_argument("Artifact output path is empty");
    const fs::path parent = output.parent_path().empty()
        ? fs::current_path() : output.parent_path();
    fs::create_directories(parent);
    if (fs::exists(output)) {
        throw std::runtime_error("Artifact output already exists: " + output.string());
    }
    const fs::path temporary = parent
        / ("." + output.filename().string() + ".tmp-" + temporary_suffix());
    fs::create_directory(temporary);
    try {
        writer(temporary);
        fs::rename(temporary, output);
    } catch (...) {
        std::error_code ignored;
        fs::remove_all(temporary, ignored);
        throw;
    }
}

std::string sha256_string(const std::string& value) {
    Sha256 digest;
    digest.update(reinterpret_cast<const uint8_t*>(value.data()), value.size());
    return digest.finish();
}

std::string sha256_file(const fs::path& path) {
    Sha256 digest;
    update_file(digest, path);
    return digest.finish();
}

std::string artifact_fingerprint(const json& manifest, const fs::path& root,
        const std::vector<json>& array_specs) {
    Sha256 digest;
    const std::string canonical = manifest.dump();
    digest.update(reinterpret_cast<const uint8_t*>(canonical.data()),
        canonical.size());
    std::vector<json> ordered = array_specs;
    std::sort(ordered.begin(), ordered.end(), [](const json& first,
            const json& second) {
        return first.at("path").get<std::string>()
            < second.at("path").get<std::string>();
    });
    for (const json& specification : ordered) {
        const std::string encoded = specification.dump();
        digest.update(reinterpret_cast<const uint8_t*>(encoded.data()),
            encoded.size());
        update_file(digest, safe_artifact_path(
            root, specification.at("path").get<std::string>()));
    }
    return digest.finish();
}

} // namespace punkst::multires
