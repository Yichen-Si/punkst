#include "clustering/uac_stream.hpp"

#include <array>
#include <atomic>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace uac {
namespace {

constexpr uint64_t kDocumentSpoolMagic = 0x554143444f433031ull;
constexpr uint32_t kDocumentSpoolVersion = 1;
constexpr uint32_t kDocumentSpoolEndian = 0x01020304u;
constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

uint64_t spool_hash_bytes(
    uint64_t hash, const void* input, size_t bytes) {
    const auto* values = static_cast<const unsigned char*>(input);
    for (size_t i = 0; i < bytes; ++i) {
        hash ^= values[i];
        hash *= kFnvPrime;
    }
    return hash;
}

template<class Value>
void write_spool_value(
    std::ostream& out, const Value& value, uint64_t* checksum = nullptr) {
    out.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (checksum) {
        *checksum = spool_hash_bytes(
            *checksum, &value, sizeof(value));
    }
}

template<class Value>
Value read_spool_value(
    std::istream& in, uint64_t* checksum = nullptr) {
    Value value{};
    in.read(reinterpret_cast<char*>(&value), sizeof(value));
    if (!in) {
        throw std::runtime_error("Truncated UAC document spool");
    }
    if (checksum) {
        *checksum = spool_hash_bytes(
            *checksum, &value, sizeof(value));
    }
    return value;
}

void write_spool_bytes(std::ostream& out, const void* values, size_t bytes,
    uint64_t* checksum = nullptr) {
    if (bytes == 0) return;
    out.write(static_cast<const char*>(values), bytes);
    if (checksum) {
        *checksum = spool_hash_bytes(*checksum, values, bytes);
    }
}

void read_spool_bytes(std::istream& in, void* values, size_t bytes,
    uint64_t* checksum = nullptr) {
    if (bytes == 0) return;
    in.read(static_cast<char*>(values), bytes);
    if (!in) {
        throw std::runtime_error("Truncated UAC document spool");
    }
    if (checksum) {
        *checksum = spool_hash_bytes(*checksum, values, bytes);
    }
}

struct DocumentSpoolHeader {
    uint64_t magic = kDocumentSpoolMagic;
    uint32_t version = kDocumentSpoolVersion;
    uint32_t endian = kDocumentSpoolEndian;
    uint64_t documents = 0;
    uint64_t features = 0;
    uint64_t nonzeros = 0;
    uint64_t payload_begin = sizeof(DocumentSpoolHeader);
    uint64_t index_begin = 0;
    uint64_t payload_checksum = kFnvOffset;
    uint64_t index_checksum = kFnvOffset;
    uint64_t content_checksum = kFnvOffset;
};

void write_spool_header(
    std::ostream& out, const DocumentSpoolHeader& header) {
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
}

DocumentSpoolHeader read_spool_header(std::istream& in) {
    DocumentSpoolHeader header;
    in.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!in || header.magic != kDocumentSpoolMagic
        || header.version != kDocumentSpoolVersion
        || header.endian != kDocumentSpoolEndian
        || header.payload_begin != sizeof(DocumentSpoolHeader)
        || header.documents == 0 || header.features == 0
        || header.index_begin < header.payload_begin) {
        throw std::runtime_error("Invalid UAC document spool header");
    }
    return header;
}

class BinaryDocumentSpool final : public IndexedDocumentSource {
public:
    BinaryDocumentSpool(
        std::filesystem::path path, bool remove_on_destruction)
        : path_(std::move(path)),
          remove_on_destruction_(remove_on_destruction) {
        std::ifstream in(path_, std::ios::binary);
        if (!in) {
            throw std::runtime_error(
                "Cannot open UAC document spool: " + path_.string());
        }
        header_ = read_spool_header(in);
        const uint64_t file_bytes = std::filesystem::file_size(path_);
        const uint64_t index_values = header_.documents + 1;
        if (index_values > std::numeric_limits<size_t>::max()
                / sizeof(uint64_t)
            || header_.index_begin
                + index_values * sizeof(uint64_t) != file_bytes) {
            throw std::runtime_error(
                "Invalid UAC document spool index extent");
        }
        offsets_.resize(static_cast<size_t>(index_values));
        in.seekg(static_cast<std::streamoff>(header_.index_begin));
        uint64_t index_checksum = kFnvOffset;
        read_spool_bytes(in, offsets_.data(),
            offsets_.size() * sizeof(uint64_t), &index_checksum);
        if (index_checksum != header_.index_checksum
            || offsets_.front() != header_.payload_begin
            || offsets_.back() != header_.index_begin) {
            throw std::runtime_error(
                "Invalid UAC document spool index");
        }
        for (size_t i = 1; i < offsets_.size(); ++i) {
            if (offsets_[i] <= offsets_[i - 1]) {
                throw std::runtime_error(
                    "Non-increasing UAC document spool offsets");
            }
        }
        in.clear();
        in.seekg(static_cast<std::streamoff>(header_.payload_begin));
        uint64_t payload_checksum = kFnvOffset;
        std::array<char, 64 * 1024> buffer;
        uint64_t remaining =
            header_.index_begin - header_.payload_begin;
        while (remaining > 0) {
            const size_t count = static_cast<size_t>(
                std::min<uint64_t>(remaining, buffer.size()));
            read_spool_bytes(
                in, buffer.data(), count, &payload_checksum);
            remaining -= count;
        }
        if (payload_checksum != header_.payload_checksum) {
            throw std::runtime_error(
                "UAC document spool payload checksum mismatch");
        }
    }

    ~BinaryDocumentSpool() override {
        if (remove_on_destruction_) {
            std::error_code error;
            std::filesystem::remove(path_, error);
        }
    }

    int64_t documents() const override {
        return static_cast<int64_t>(header_.documents);
    }

    int64_t features() const override {
        return static_cast<int64_t>(header_.features);
    }

    uint64_t storage_bytes() const override {
        return std::filesystem::file_size(path_);
    }

    uint64_t content_checksum() const override {
        return header_.content_checksum;
    }

    uint64_t peak_block_bytes() const override {
        return peak_block_bytes_.load(std::memory_order_relaxed);
    }

    void reset() override {
        cursor_ = 0;
    }

    bool next(
        DocumentBlock& block, int32_t maximum_documents) override {
        if (maximum_documents <= 0) {
            throw std::invalid_argument(
                "UAC document spool block size must be positive");
        }
        if (cursor_ >= documents()) {
            block.clear();
            return false;
        }
        const int32_t count = static_cast<int32_t>(
            std::min<int64_t>(maximum_documents, documents() - cursor_));
        read_range(cursor_, count, block);
        cursor_ += count;
        return true;
    }

    void read_range(int64_t first_document, int32_t count,
        DocumentBlock& block) const override {
        if (first_document < 0 || count <= 0
            || first_document + count > documents()) {
            throw std::out_of_range(
                "UAC document spool range is out of bounds");
        }
        std::ifstream in(path_, std::ios::binary);
        if (!in) {
            throw std::runtime_error(
                "Cannot read UAC document spool: " + path_.string());
        }
        in.seekg(static_cast<std::streamoff>(
            offsets_[static_cast<size_t>(first_document)]));
        if (!in) {
            throw std::runtime_error(
                "Cannot seek UAC document spool");
        }
        block.clear();
        block.first_document = first_document;
        block.identifiers.reserve(count);
        block.counts.reserve(count);
        block.raw_totals.resize(count);
        block.effective_totals.resize(count);
        for (int32_t local = 0; local < count; ++local) {
            const uint32_t identifier_bytes =
                read_spool_value<uint32_t>(in);
            const uint32_t nonzeros =
                read_spool_value<uint32_t>(in);
            const double raw_total = read_spool_value<double>(in);
            const double effective_total =
                read_spool_value<double>(in);
            const uint8_t counts_weighted =
                read_spool_value<uint8_t>(in);
            if (identifier_bytes == 0
                || nonzeros > header_.nonzeros
                || !std::isfinite(raw_total)
                || !std::isfinite(effective_total)
                || raw_total < 0.0 || !(effective_total > 0.0)
                || counts_weighted > 1) {
                throw std::runtime_error(
                    "Invalid UAC document spool record");
            }
            std::string identifier(identifier_bytes, '\0');
            read_spool_bytes(
                in, identifier.data(), identifier.size());
            Document document;
            document.ids.resize(nonzeros);
            document.cnts.resize(nonzeros);
            read_spool_bytes(in, document.ids.data(),
                document.ids.size() * sizeof(uint32_t));
            read_spool_bytes(in, document.cnts.data(),
                document.cnts.size() * sizeof(double));
            for (size_t j = 0; j < document.ids.size(); ++j) {
                if (document.ids[j] >= header_.features
                    || !std::isfinite(document.cnts[j])
                    || document.cnts[j] < 0.0) {
                    throw std::runtime_error(
                        "Invalid UAC document spool count");
                }
            }
            document.raw_ct_tot = raw_total;
            document.ct_tot = effective_total;
            document.counts_weighted = counts_weighted != 0;
            block.identifiers.push_back(std::move(identifier));
            block.counts.push_back(std::move(document));
            block.raw_totals(local) = raw_total;
            block.effective_totals(local) = effective_total;
            const uint64_t expected =
                offsets_[static_cast<size_t>(
                    first_document + local + 1)];
            const auto current = in.tellg();
            if (current < 0
                || static_cast<uint64_t>(current) != expected) {
                throw std::runtime_error(
                    "Invalid UAC document spool record extent");
            }
        }
        uint64_t block_bytes = sizeof(double)
            * static_cast<uint64_t>(
                block.raw_totals.size() + block.effective_totals.size());
        for (size_t d = 0; d < block.counts.size(); ++d) {
            block_bytes += block.identifiers[d].size()
                + sizeof(uint32_t) * block.counts[d].ids.size()
                + sizeof(double) * block.counts[d].cnts.size();
        }
        uint64_t previous =
            peak_block_bytes_.load(std::memory_order_relaxed);
        while (previous < block_bytes
            && !peak_block_bytes_.compare_exchange_weak(
                previous, block_bytes, std::memory_order_relaxed)) {
        }
    }

private:
    std::filesystem::path path_;
    bool remove_on_destruction_ = true;
    DocumentSpoolHeader header_;
    std::vector<uint64_t> offsets_;
    int64_t cursor_ = 0;
    mutable std::atomic<uint64_t> peak_block_bytes_{0};
};

template<class Function>
void read_score_rows(const ScoreResult& score, Function&& function) {
    if (score.responsibilities.size() > 0) {
        for (Eigen::Index d = 0;
                d < score.responsibilities.rows(); ++d) {
            function(static_cast<int64_t>(d),
                score.responsibilities.row(d).transpose());
        }
        return;
    }
    std::ifstream in(score.responsibility_sidecar, std::ios::binary);
    if (!in) {
        throw std::runtime_error(
            "Cannot read streaming UAC responsibility sidecar");
    }
    Eigen::VectorXd row(score.scored_components);
    for (int64_t d = 0; d < score.scored_documents; ++d) {
        in.read(reinterpret_cast<char*>(row.data()),
            sizeof(double) * row.size());
        if (!in) {
            throw std::runtime_error(
                "Truncated streaming UAC responsibility sidecar");
        }
        function(d, row);
    }
}

void emit_score(const Dataset& data, const ScoreResult& score,
    int32_t components, StreamingScoreSink* sink) {
    if (!sink) return;
    sink->begin(static_cast<int64_t>(data.identifiers.size()), components);
    read_score_rows(score,
        [&](int64_t d, const Eigen::VectorXd& responsibility) {
        StreamingScoreRow row;
        row.document = d;
        row.identifier = data.identifiers[d];
        row.raw_total = data.raw_totals.size() ? data.raw_totals(d) : 0.0;
        row.effective_total = data.effective_totals.size()
            ? data.effective_totals(d) : 0.0;
        row.responsibilities = responsibility;
        if (d < static_cast<int64_t>(
                score.particle_diagnostics.size())) {
            row.particle_diagnostic =
                score.particle_diagnostics[d];
        }
        if (d < static_cast<int64_t>(
                score.adaptive_particle_diagnostics.size())) {
            row.adaptive_particle_diagnostic =
                score.adaptive_particle_diagnostics[d];
        }
        if (d < static_cast<int64_t>(
                score.per_document_particles.size())) {
            row.particles = score.per_document_particles[d];
        }
        if (d < static_cast<int64_t>(
                score.per_document_proposal_components.size())) {
            row.proposal_components =
                score.per_document_proposal_components[d];
        }
        if (d < static_cast<int64_t>(
                score.per_document_evaluated_components.size())) {
            row.evaluated_components =
                score.per_document_evaluated_components[d];
        }
        if (d < static_cast<int64_t>(
                score.per_document_omitted_component_mass.size())) {
            row.omitted_component_mass =
                score.per_document_omitted_component_mass[d];
        }
        sink->write(row);
    });
    sink->end();
}

} // namespace

struct BinaryDocumentSpoolWriter::Impl {
    std::filesystem::path path;
    std::filesystem::path partial_path;
    int32_t features = 0;
    std::ofstream out;
    DocumentSpoolHeader header;
    std::vector<uint64_t> offsets;
    bool finished = false;

    Impl(std::filesystem::path input_path, int32_t input_features)
        : path(std::move(input_path)),
          partial_path(path.string() + ".partial"),
          features(input_features),
          out(partial_path, std::ios::binary | std::ios::trunc) {
        if (features <= 0 || !out) {
            throw std::runtime_error(
                "Cannot create UAC document spool: "
                + partial_path.string());
        }
        header.features = static_cast<uint64_t>(features);
        write_spool_header(out, header);
        if (!out) {
            throw std::runtime_error(
                "Cannot initialize UAC document spool");
        }
        offsets.push_back(header.payload_begin);
    }

    ~Impl() {
        out.close();
        if (!finished) {
            std::error_code error;
            std::filesystem::remove(partial_path, error);
        }
    }
};

BinaryDocumentSpoolWriter::BinaryDocumentSpoolWriter(
    const std::filesystem::path& path, int32_t features)
    : impl_(std::make_unique<Impl>(path, features)) {
    std::filesystem::permissions(impl_->partial_path,
        std::filesystem::perms::owner_read
            | std::filesystem::perms::owner_write,
        std::filesystem::perm_options::replace);
}

BinaryDocumentSpoolWriter::~BinaryDocumentSpoolWriter() = default;
BinaryDocumentSpoolWriter::BinaryDocumentSpoolWriter(
    BinaryDocumentSpoolWriter&&) noexcept = default;
BinaryDocumentSpoolWriter& BinaryDocumentSpoolWriter::operator=(
    BinaryDocumentSpoolWriter&&) noexcept = default;

void BinaryDocumentSpoolWriter::append(
    const std::string& identifier, const Document& document,
    double raw_total, double effective_total) {
    if (!impl_ || impl_->finished || identifier.empty()
        || identifier.size() > std::numeric_limits<uint32_t>::max()
        || document.ids.size() != document.cnts.size()
        || document.ids.size() > std::numeric_limits<uint32_t>::max()
        || !std::isfinite(raw_total) || raw_total < 0.0
        || !std::isfinite(effective_total) || !(effective_total > 0.0)) {
        throw std::invalid_argument(
            "Invalid UAC document spool append");
    }
    for (size_t j = 0; j < document.ids.size(); ++j) {
        if (document.ids[j] >= static_cast<uint32_t>(impl_->features)
            || !std::isfinite(document.cnts[j])
            || document.cnts[j] < 0.0) {
            throw std::invalid_argument(
                "Invalid UAC document spool count");
        }
    }
    const uint32_t identifier_bytes =
        static_cast<uint32_t>(identifier.size());
    const uint32_t nonzeros =
        static_cast<uint32_t>(document.ids.size());
    write_spool_value(
        impl_->out, identifier_bytes, &impl_->header.payload_checksum);
    write_spool_value(
        impl_->out, nonzeros, &impl_->header.payload_checksum);
    write_spool_value(
        impl_->out, raw_total, &impl_->header.payload_checksum);
    write_spool_value(
        impl_->out, effective_total, &impl_->header.payload_checksum);
    const uint8_t counts_weighted = document.counts_weighted ? 1 : 0;
    write_spool_value(
        impl_->out, counts_weighted, &impl_->header.payload_checksum);
    write_spool_bytes(impl_->out, identifier.data(), identifier.size(),
        &impl_->header.payload_checksum);
    write_spool_bytes(impl_->out, document.ids.data(),
        document.ids.size() * sizeof(uint32_t),
        &impl_->header.payload_checksum);
    write_spool_bytes(impl_->out, document.cnts.data(),
        document.cnts.size() * sizeof(double),
        &impl_->header.payload_checksum);
    impl_->header.content_checksum = spool_hash_bytes(
        impl_->header.content_checksum,
        identifier.data(), identifier.size());
    impl_->header.content_checksum = spool_hash_bytes(
        impl_->header.content_checksum, &nonzeros, sizeof(nonzeros));
    impl_->header.content_checksum = spool_hash_bytes(
        impl_->header.content_checksum, &counts_weighted,
        sizeof(counts_weighted));
    impl_->header.content_checksum = spool_hash_bytes(
        impl_->header.content_checksum, document.ids.data(),
        document.ids.size() * sizeof(uint32_t));
    impl_->header.content_checksum = spool_hash_bytes(
        impl_->header.content_checksum, document.cnts.data(),
        document.cnts.size() * sizeof(double));
    ++impl_->header.documents;
    impl_->header.nonzeros += nonzeros;
    const auto end = impl_->out.tellp();
    if (!impl_->out || end < 0) {
        throw std::runtime_error(
            "Failed writing UAC document spool");
    }
    impl_->offsets.push_back(static_cast<uint64_t>(end));
}

std::unique_ptr<IndexedDocumentSource>
BinaryDocumentSpoolWriter::finish(bool remove_on_destruction) {
    if (!impl_ || impl_->finished || impl_->header.documents == 0) {
        throw std::runtime_error(
            "Cannot finalize empty UAC document spool");
    }
    impl_->header.index_begin =
        static_cast<uint64_t>(impl_->out.tellp());
    write_spool_bytes(impl_->out, impl_->offsets.data(),
        impl_->offsets.size() * sizeof(uint64_t),
        &impl_->header.index_checksum);
    impl_->out.flush();
    if (!impl_->out) {
        throw std::runtime_error(
            "Failed finalizing UAC document spool index");
    }
    impl_->out.seekp(0);
    write_spool_header(impl_->out, impl_->header);
    impl_->out.close();
    if (!impl_->out) {
        throw std::runtime_error(
            "Failed finalizing UAC document spool header");
    }
    const std::filesystem::path path = impl_->path;
    std::filesystem::rename(impl_->partial_path, path);
    impl_->finished = true;
    return std::make_unique<BinaryDocumentSpool>(
        path, remove_on_destruction);
}

std::unique_ptr<IndexedDocumentSource> open_binary_document_spool(
    const std::filesystem::path& path, bool remove_on_destruction) {
    return std::make_unique<BinaryDocumentSpool>(
        path, remove_on_destruction);
}

void DocumentBlock::clear() {
    first_document = 0;
    identifiers.clear();
    counts.clear();
    raw_totals.resize(0);
    effective_totals.resize(0);
}

int32_t DocumentBlock::size() const {
    return static_cast<int32_t>(identifiers.size());
}

StreamingFitResult fit_streaming(const Dataset& data, const Basis& basis,
    const FitOptions& options, StreamingScoreSink* sink) {
    FitOptions configured = options;
    configured.handoff = HandoffMode::Particle;
    configured.particle_engine = ParticleEngine::Stream;
    configured.streaming.count_storage = StreamingCountStorage::Source;
    FitResult fitted = fit(data, &basis, configured);

    StreamingFitResult out;
    out.model = fitted.model;
    out.pilot = fitted.pilot;
    out.traces = fitted.traces;
    out.converged = fitted.converged;
    out.selected_start = fitted.selected_start;
    out.selected_start_method = fitted.selected_start_method;
    out.selected_leiden_resolution = fitted.selected_leiden_resolution;
    out.score.documents = static_cast<int64_t>(data.identifiers.size());
    out.score.effective_membership =
        fitted.score.effective_membership;
    emit_score(data, fitted.score,
        static_cast<int32_t>(fitted.model.weights.size()), sink);
    out.score.diagnostics = std::move(fitted.score);
    return out;
}

StreamingScoreSummary score_particle_streaming(const Dataset& data,
    const Basis& basis, const State& state,
    const ParticleScoreOptions& options, StreamingScoreSink* sink) {
    ParticleScoreOptions configured = options;
    configured.particle_engine = ParticleEngine::Stream;
    configured.streaming.count_storage = StreamingCountStorage::Source;
    ScoreResult score = score_particle(
        data, basis, state, configured);
    StreamingScoreSummary out;
    out.documents = static_cast<int64_t>(data.identifiers.size());
    out.effective_membership = score.effective_membership;
    emit_score(data, score,
        static_cast<int32_t>(state.model.weights.size()), sink);
    out.diagnostics = std::move(score);
    return out;
}

State make_state(const StreamingFitResult& fit_result,
    const FitOptions& options, const StateMetadata& metadata) {
    FitResult ordinary;
    ordinary.model = fit_result.model;
    ordinary.pilot = fit_result.pilot;
    ordinary.traces = fit_result.traces;
    ordinary.converged = fit_result.converged;
    ordinary.selected_start = fit_result.selected_start;
    ordinary.selected_start_method = fit_result.selected_start_method;
    ordinary.selected_leiden_resolution =
        fit_result.selected_leiden_resolution;
    return make_state(ordinary, options, metadata);
}

} // namespace uac
