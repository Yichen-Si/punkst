#include "gamma_pois_posterior_io.hpp"

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstring>
#include <iomanip>
#include <random>
#include <stdexcept>
#include <sstream>
#include <set>
#include <vector>

namespace {

constexpr std::array<char, 8> MAGIC = {'P', 'G', 'P', 'D', 'S', 'P', '2', '\0'};
constexpr uint32_t VERSION = 2;
constexpr uint32_t FLOAT32_TYPE = 1;
constexpr std::streamoff RECORD_COUNT_OFFSET = 32;
constexpr uint64_t HEADER_SIZE = 56;

void write_u32(std::ostream& out, uint32_t value) {
    const std::array<char, 4> bytes = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu)
    };
    out.write(bytes.data(), bytes.size());
}

void write_u64(std::ostream& out, uint64_t value) {
    for (int32_t shift = 0; shift < 64; shift += 8) {
        out.put(static_cast<char>((value >> shift) & 0xffu));
    }
}

uint32_t read_u32(std::istream& in) {
    std::array<unsigned char, 4> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!in) throw std::runtime_error("Truncated Gamma-Poisson dispersion header");
    return static_cast<uint32_t>(bytes[0])
        | (static_cast<uint32_t>(bytes[1]) << 8)
        | (static_cast<uint32_t>(bytes[2]) << 16)
        | (static_cast<uint32_t>(bytes[3]) << 24);
}

uint64_t read_u64(std::istream& in) {
    std::array<unsigned char, 8> bytes{};
    in.read(reinterpret_cast<char*>(bytes.data()), bytes.size());
    if (!in) throw std::runtime_error("Truncated Gamma-Poisson dispersion header");
    uint64_t value = 0;
    for (int32_t i = 0; i < 8; ++i) {
        value |= static_cast<uint64_t>(bytes[i]) << (8 * i);
    }
    return value;
}

void write_float(std::ostream& out, float value) {
    uint32_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value), "unexpected float size");
    std::memcpy(&bits, &value, sizeof(bits));
    write_u32(out, bits);
}

float read_float(std::istream& in) {
    const uint32_t bits = read_u32(in);
    float value = 0;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::vector<std::string> split_tabs(const std::string& line) {
    std::vector<std::string> out;
    size_t begin = 0;
    while (begin <= line.size()) {
        const size_t end = line.find('\t', begin);
        if (end == std::string::npos) {
            out.push_back(line.substr(begin));
            break;
        }
        out.push_back(line.substr(begin, end - begin));
        begin = end + 1;
    }
    return out;
}

uint64_t parse_u64(const std::string& value, const std::string& label) {
    size_t used = 0;
    uint64_t parsed = 0;
    try {
        parsed = std::stoull(value, &used);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson posterior");
    }
    if (used != value.size()) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson posterior");
    }
    return parsed;
}

double parse_double(const std::string& value, const std::string& label) {
    size_t used = 0;
    double parsed = 0.0;
    try {
        parsed = std::stod(value, &used);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson posterior");
    }
    if (used != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error("Invalid " + label + " in Gamma-Poisson posterior");
    }
    return parsed;
}

} // namespace

GammaPoissonArtifactId generate_gamma_poisson_artifact_id() {
    std::random_device random;
    GammaPoissonArtifactId id;
    for (uint64_t& word : id.words) {
        for (int32_t byte = 0; byte < 8; ++byte) {
            word = (word << 8) | (static_cast<uint64_t>(random()) & 0xffu);
        }
    }
    if (id.empty()) id.words[1] = 1;
    return id;
}

std::string gamma_poisson_artifact_id_string(
    const GammaPoissonArtifactId& id) {
    if (id.empty()) return "";
    std::ostringstream out;
    out << std::hex << std::setfill('0')
        << std::setw(16) << id.words[0]
        << std::setw(16) << id.words[1];
    return out.str();
}

GammaPoissonArtifactId parse_gamma_poisson_artifact_id(
    const std::string& value) {
    if (value.size() != 32
        || !std::all_of(value.begin(), value.end(), [](unsigned char ch) {
            return std::isdigit(ch) || (ch >= 'a' && ch <= 'f');
        })) {
        throw std::runtime_error("Invalid Gamma-Poisson dispersion sidecar ID");
    }
    GammaPoissonArtifactId id;
    try {
        id.words[0] = std::stoull(value.substr(0, 16), nullptr, 16);
        id.words[1] = std::stoull(value.substr(16), nullptr, 16);
    } catch (const std::exception&) {
        throw std::runtime_error("Invalid Gamma-Poisson dispersion sidecar ID");
    }
    if (id.empty()) {
        throw std::runtime_error("Gamma-Poisson dispersion sidecar ID cannot be zero");
    }
    return id;
}

uint64_t gamma_poisson_state_checksum(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("Cannot open Gamma-Poisson state for checksum: " + path);
    }
    uint64_t hash = 14695981039346656037ull;
    std::array<char, 1 << 15> buffer{};
    while (in) {
        in.read(buffer.data(), buffer.size());
        const std::streamsize n = in.gcount();
        for (std::streamsize i = 0; i < n; ++i) {
            hash ^= static_cast<unsigned char>(buffer[i]);
            hash *= 1099511628211ull;
        }
    }
    return hash;
}

GammaPoissonPosteriorReader::GammaPoissonPosteriorReader(
    const std::string& path, uint64_t expected_state_checksum)
    : in_(path) {
    if (!in_) throw std::runtime_error("Cannot open Gamma-Poisson posterior: " + path);
    std::string line;
    bool saw_version = false;
    bool saw_header = false;
    bool saw_n_topics = false;
    bool saw_checksum = false;
    bool saw_row_order = false;
    bool saw_sidecar_id = false;
    std::vector<std::string> rate_topic_names;
    while (std::getline(in_, line)) {
        if (line.rfind("##", 0) == 0) {
            const auto fields = split_tabs(line.substr(2));
            if (fields[0] == "punkst_gamma_pois_posterior_v2") {
                if (saw_version || fields.size() != 1) {
                    throw std::runtime_error("Duplicate or invalid Gamma-Poisson posterior version");
                }
                saw_version = true;
            } else if (fields[0] == "n_topics" && fields.size() == 2) {
                if (saw_n_topics) throw std::runtime_error("Duplicate Gamma-Poisson posterior topic count");
                header_.n_topics = static_cast<int32_t>(parse_u64(fields[1], "topic count"));
                saw_n_topics = true;
            } else if (fields[0] == "state_checksum" && fields.size() == 2) {
                if (saw_checksum) throw std::runtime_error("Duplicate Gamma-Poisson posterior checksum");
                header_.state_checksum = parse_u64(fields[1], "state checksum");
                saw_checksum = true;
            } else if (fields[0] == "row_order" && fields.size() == 2) {
                if (saw_row_order) throw std::runtime_error("Duplicate Gamma-Poisson posterior row order");
                header_.row_order = fields[1];
                saw_row_order = true;
            } else if (fields[0] == "dispersion_sidecar_id"
                    && fields.size() == 2) {
                if (saw_sidecar_id) {
                    throw std::runtime_error(
                        "Duplicate Gamma-Poisson dispersion sidecar ID");
                }
                header_.dispersion_sidecar_id =
                    parse_gamma_poisson_artifact_id(fields[1]);
                saw_sidecar_id = true;
            }
            continue;
        }
        if (line.empty()) continue;
        if (line[0] != '#') {
            throw std::runtime_error("Missing Gamma-Poisson posterior column header");
        }
        const auto columns = split_tabs(line.substr(1));
        std::set<std::string> unique_columns;
        for (size_t i = 0; i < columns.size(); ++i) {
            const std::string& column = columns[i];
            if (column.empty() || !unique_columns.insert(column).second) {
                throw std::runtime_error("Duplicate or empty Gamma-Poisson posterior column");
            }
            if (column == "gp_row") {
                row_column_ = i;
            } else if (column == "gp_exposure") {
                exposure_column_ = i;
            } else if (column.rfind("gp_shape_", 0) == 0) {
                shape_columns_.push_back(i);
                header_.topic_names.push_back(column.substr(9));
            } else if (column.rfind("gp_rate_", 0) == 0) {
                rate_columns_.push_back(i);
                rate_topic_names.push_back(column.substr(8));
            } else {
                identifier_columns_.push_back(i);
                header_.identifier_columns.push_back(column);
            }
        }
        saw_header = true;
        break;
    }
    std::set<std::string> unique_topics(header_.topic_names.begin(),
        header_.topic_names.end());
    if (!saw_version || !saw_n_topics || !saw_checksum || !saw_row_order
        || !saw_header || header_.n_topics <= 0
        || header_.state_checksum == 0
        || (header_.row_order != "input" && header_.row_order != "randomized")
        || row_column_ == static_cast<size_t>(-1)
        || exposure_column_ == static_cast<size_t>(-1)
        || shape_columns_.size() != static_cast<size_t>(header_.n_topics)
        || rate_columns_.size() != static_cast<size_t>(header_.n_topics)
        || rate_topic_names != header_.topic_names
        || unique_topics.size() != header_.topic_names.size()) {
        throw std::runtime_error("Invalid Gamma-Poisson posterior header");
    }
    if (expected_state_checksum != 0
        && header_.state_checksum != expected_state_checksum) {
        throw std::runtime_error("Gamma-Poisson posterior state checksum mismatch");
    }
}

bool GammaPoissonPosteriorReader::read_next(GammaPoissonPosteriorRow& row) {
    std::string line;
    while (std::getline(in_, line) && line.empty()) {
    }
    if (!in_ && line.empty()) return false;
    const auto fields = split_tabs(line);
    size_t max_column = std::max(row_column_, exposure_column_);
    for (size_t i : shape_columns_) max_column = std::max(max_column, i);
    for (size_t i : rate_columns_) max_column = std::max(max_column, i);
    for (size_t i : identifier_columns_) max_column = std::max(max_column, i);
    if (fields.size() <= max_column) {
        throw std::runtime_error("Truncated Gamma-Poisson posterior row");
    }
    row.row = parse_u64(fields[row_column_], "row index");
    if (row.row != expected_row_) {
        throw std::runtime_error("Gamma-Poisson posterior rows are out of order");
    }
    row.posterior.exposure = parse_double(fields[exposure_column_], "exposure");
    if (row.posterior.exposure < 0.0) {
        throw std::runtime_error("Gamma-Poisson posterior exposure must be nonnegative");
    }
    row.posterior.shape.resize(header_.n_topics);
    row.posterior.rate.resize(header_.n_topics);
    for (int32_t k = 0; k < header_.n_topics; ++k) {
        row.posterior.shape(k) = parse_double(fields[shape_columns_[k]], "shape");
        row.posterior.rate(k) = parse_double(fields[rate_columns_[k]], "rate");
        if (!(row.posterior.shape(k) > 0.0) || !(row.posterior.rate(k) > 0.0)) {
            throw std::runtime_error("Gamma-Poisson posterior shape and rate must be positive");
        }
    }
    std::ostringstream identifiers;
    for (size_t j = 0; j < identifier_columns_.size(); ++j) {
        if (j > 0) identifiers << '\t';
        identifiers << fields[identifier_columns_[j]];
    }
    row.identifiers = identifiers.str();
    ++expected_row_;
    return true;
}

GammaPoissonDispersionWriter::GammaPoissonDispersionWriter(
    const std::string& path, int32_t n_topics, int32_t rank,
    uint64_t state_checksum, const GammaPoissonArtifactId& artifact_id)
    : out_(path, std::ios::binary | std::ios::trunc), n_topics_(n_topics),
      rank_(rank), state_checksum_(state_checksum), artifact_id_(artifact_id) {
    if (!out_) throw std::runtime_error("Cannot open dispersion sidecar for writing: " + path);
    if (n_topics_ <= 0 || rank_ < 0 || rank_ > n_topics_
        || artifact_id_.empty()) {
        throw std::invalid_argument("Invalid Gamma-Poisson dispersion sidecar dimensions");
    }
    out_.write(MAGIC.data(), MAGIC.size());
    write_u32(out_, VERSION);
    write_u32(out_, static_cast<uint32_t>(n_topics_));
    write_u32(out_, static_cast<uint32_t>(rank_));
    write_u32(out_, FLOAT32_TYPE);
    write_u64(out_, state_checksum_);
    write_u64(out_, 0);
    write_u64(out_, artifact_id_.words[0]);
    write_u64(out_, artifact_id_.words[1]);
}

GammaPoissonDispersionWriter::~GammaPoissonDispersionWriter() {
    try {
        close();
    } catch (...) {
    }
}

void GammaPoissonDispersionWriter::append(
    const GammaPoissonDispersionApproximation& approximation) {
    if (closed_) throw std::runtime_error("Dispersion sidecar is already closed");
    if (approximation.residual_diagonal.size() != n_topics_
        || approximation.factor.rows() != n_topics_
        || approximation.factor.cols() != rank_) {
        throw std::invalid_argument("Dispersion approximation dimensions do not match sidecar header");
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        const double value = approximation.residual_diagonal(k);
        if (!std::isfinite(value) || value < 0.0) {
            throw std::invalid_argument("Dispersion residual diagonal must be finite and nonnegative");
        }
        write_float(out_, static_cast<float>(value));
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        for (int32_t a = 0; a < rank_; ++a) {
            const double value = approximation.factor(k, a);
            if (!std::isfinite(value)) {
                throw std::invalid_argument("Dispersion factor must be finite");
            }
            write_float(out_, static_cast<float>(value));
        }
    }
    if (!out_) throw std::runtime_error("Failed writing Gamma-Poisson dispersion sidecar");
    ++record_count_;
}

void GammaPoissonDispersionWriter::close() {
    if (closed_) return;
    out_.seekp(RECORD_COUNT_OFFSET);
    write_u64(out_, record_count_);
    out_.close();
    closed_ = true;
}

GammaPoissonDispersionReader::GammaPoissonDispersionReader(
    const std::string& path, uint64_t expected_state_checksum)
    : in_(path, std::ios::binary) {
    if (!in_) throw std::runtime_error("Cannot open dispersion sidecar: " + path);
    std::array<char, 8> magic{};
    in_.read(magic.data(), magic.size());
    if (!in_ || magic != MAGIC) {
        throw std::runtime_error("Invalid Gamma-Poisson dispersion sidecar magic");
    }
    if (read_u32(in_) != VERSION) {
        throw std::runtime_error("Unsupported Gamma-Poisson dispersion sidecar version");
    }
    header_.n_topics = static_cast<int32_t>(read_u32(in_));
    header_.rank = static_cast<int32_t>(read_u32(in_));
    if (read_u32(in_) != FLOAT32_TYPE) {
        throw std::runtime_error("Unsupported Gamma-Poisson dispersion sidecar scalar type");
    }
    header_.state_checksum = read_u64(in_);
    header_.record_count = read_u64(in_);
    header_.artifact_id.words[0] = read_u64(in_);
    header_.artifact_id.words[1] = read_u64(in_);
    if (header_.n_topics <= 0 || header_.rank < 0
        || header_.rank > header_.n_topics || header_.artifact_id.empty()) {
        throw std::runtime_error("Invalid Gamma-Poisson dispersion sidecar dimensions");
    }
    if (expected_state_checksum != 0
        && header_.state_checksum != expected_state_checksum) {
        throw std::runtime_error("Gamma-Poisson dispersion sidecar state checksum mismatch");
    }
    const uint64_t values_per_record = static_cast<uint64_t>(header_.n_topics)
        * static_cast<uint64_t>(header_.rank + 1);
    if (values_per_record > std::numeric_limits<uint64_t>::max() / 4
        || header_.record_count > (std::numeric_limits<uint64_t>::max() - HEADER_SIZE)
            / (4 * values_per_record)) {
        throw std::runtime_error("Gamma-Poisson dispersion sidecar size overflows");
    }
    const uint64_t expected_size = HEADER_SIZE
        + header_.record_count * values_per_record * 4;
    const std::streampos payload_begin = in_.tellg();
    in_.seekg(0, std::ios::end);
    const std::streampos actual_size = in_.tellg();
    if (actual_size < 0 || static_cast<uint64_t>(actual_size) != expected_size) {
        throw std::runtime_error("Gamma-Poisson dispersion sidecar file size mismatch");
    }
    in_.seekg(payload_begin);
}

bool GammaPoissonDispersionReader::read_next(
    GammaPoissonDispersionApproximation& approximation) {
    if (records_read_ >= header_.record_count) return false;
    approximation.residual_diagonal.resize(header_.n_topics);
    approximation.factor.resize(header_.n_topics, header_.rank);
    for (int32_t k = 0; k < header_.n_topics; ++k) {
        approximation.residual_diagonal(k) = read_float(in_);
        if (!std::isfinite(approximation.residual_diagonal(k))
            || approximation.residual_diagonal(k) < 0.0) {
            throw std::runtime_error("Invalid Gamma-Poisson dispersion residual diagonal");
        }
    }
    for (int32_t k = 0; k < header_.n_topics; ++k) {
        for (int32_t a = 0; a < header_.rank; ++a) {
            approximation.factor(k, a) = read_float(in_);
            if (!std::isfinite(approximation.factor(k, a))) {
                throw std::runtime_error("Invalid Gamma-Poisson dispersion factor");
            }
        }
    }
    ++records_read_;
    return true;
}
