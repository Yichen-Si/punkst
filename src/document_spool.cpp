#include "document_spool.hpp"

#include <array>
#include <atomic>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace punkst {
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
        throw std::runtime_error("Truncated document spool");
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
        throw std::runtime_error("Truncated document spool");
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
        throw std::runtime_error("Invalid document spool header");
    }
    return header;
}

void read_spool_record(std::istream& in,
    const DocumentSpoolHeader& header, std::string& identifier,
    Document& document, double& raw_total, double& effective_total) {
    const uint32_t identifier_bytes =
        read_spool_value<uint32_t>(in);
    const uint32_t nonzeros =
        read_spool_value<uint32_t>(in);
    raw_total = read_spool_value<double>(in);
    effective_total = read_spool_value<double>(in);
    const uint8_t counts_weighted =
        read_spool_value<uint8_t>(in);
    if (identifier_bytes == 0
        || nonzeros > header.nonzeros
        || !std::isfinite(raw_total)
        || !std::isfinite(effective_total)
        || raw_total < 0.0 || effective_total < 0.0
        || counts_weighted > 1) {
        throw std::runtime_error(
            "Invalid document spool record");
    }
    identifier.resize(identifier_bytes);
    read_spool_bytes(
        in, identifier.data(), identifier.size());
    document.ids.resize(nonzeros);
    document.cnts.resize(nonzeros);
    read_spool_bytes(in, document.ids.data(),
        document.ids.size() * sizeof(uint32_t));
    read_spool_bytes(in, document.cnts.data(),
        document.cnts.size() * sizeof(double));
    for (size_t j = 0; j < document.ids.size(); ++j) {
        if (document.ids[j] >= header.features
            || !std::isfinite(document.cnts[j])
            || document.cnts[j] < 0.0) {
            throw std::runtime_error(
                "Invalid document spool count");
        }
    }
    document.raw_ct_tot = raw_total;
    document.ct_tot = effective_total;
    document.counts_weighted = counts_weighted != 0;
}

void prepare_document_block(
    DocumentBlock& block, int64_t first_document, int32_t count) {
    block.clear();
    block.first_document = first_document;
    block.identifiers.reserve(count);
    block.counts.reserve(count);
    block.raw_totals.resize(count);
    block.effective_totals.resize(count);
}

void update_peak_block_bytes(
    const DocumentBlock& block, std::atomic<uint64_t>& peak) {
    uint64_t block_bytes = sizeof(double)
        * static_cast<uint64_t>(
            block.raw_totals.size() + block.effective_totals.size());
    for (size_t d = 0; d < block.counts.size(); ++d) {
        block_bytes += block.identifiers[d].size()
            + sizeof(uint32_t) * block.counts[d].ids.size()
            + sizeof(double) * block.counts[d].cnts.size();
    }
    uint64_t previous = peak.load(std::memory_order_relaxed);
    while (previous < block_bytes
        && !peak.compare_exchange_weak(
            previous, block_bytes, std::memory_order_relaxed)) {
    }
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
                "Cannot open document spool: " + path_.string());
        }
        header_ = read_spool_header(in);
        const uint64_t file_bytes = std::filesystem::file_size(path_);
        const uint64_t index_values = header_.documents + 1;
        if (index_values > std::numeric_limits<size_t>::max()
                / sizeof(uint64_t)
            || header_.index_begin
                + index_values * sizeof(uint64_t) != file_bytes) {
            throw std::runtime_error(
                "Invalid document spool index extent");
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
                "Invalid document spool index");
        }
        for (size_t i = 1; i < offsets_.size(); ++i) {
            if (offsets_[i] <= offsets_[i - 1]) {
                throw std::runtime_error(
                    "Non-increasing document spool offsets");
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
                "document spool payload checksum mismatch");
        }
        open_sequential_stream();
    }

    BinaryDocumentSpool(std::filesystem::path path,
        bool remove_on_destruction, const DocumentSpoolHeader& header,
        std::vector<uint64_t> offsets)
        : path_(std::move(path)),
          remove_on_destruction_(remove_on_destruction),
          header_(header),
          offsets_(std::move(offsets)) {
        open_sequential_stream();
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
        sequential_.clear();
        sequential_.seekg(static_cast<std::streamoff>(
            offsets_.front()));
        if (!sequential_) {
            throw std::runtime_error(
                "Cannot reset document spool");
        }
    }

    bool next(
        DocumentBlock& block, int32_t maximum_documents) override {
        if (maximum_documents <= 0) {
            throw std::invalid_argument(
                "document spool block size must be positive");
        }
        if (cursor_ >= documents()) {
            block.clear();
            return false;
        }
        const int32_t count = static_cast<int32_t>(
            std::min<int64_t>(maximum_documents, documents() - cursor_));
        read_range_from_stream(
            sequential_, cursor_, count, block, false);
        cursor_ += count;
        return true;
    }

    void read_range(int64_t first_document, int32_t count,
        DocumentBlock& block) const override {
        if (first_document < 0 || count <= 0
            || first_document + count > documents()) {
            throw std::out_of_range(
                "document spool range is out of bounds");
        }
        std::ifstream in(path_, std::ios::binary);
        if (!in) {
            throw std::runtime_error(
                "Cannot read document spool: " + path_.string());
        }
        read_range_from_stream(
            in, first_document, count, block, true);
    }

private:
    void open_sequential_stream() {
        sequential_.open(path_, std::ios::binary);
        if (!sequential_) {
            throw std::runtime_error(
                "Cannot open document spool: " + path_.string());
        }
        sequential_.seekg(static_cast<std::streamoff>(
            offsets_.front()));
    }

    void read_range_from_stream(std::istream& in,
        int64_t first_document, int32_t count, DocumentBlock& block,
        bool seek) const {
        if (seek) {
            in.seekg(static_cast<std::streamoff>(
                offsets_[static_cast<size_t>(first_document)]));
            if (!in) {
                throw std::runtime_error(
                    "Cannot seek document spool");
            }
        }
        prepare_document_block(block, first_document, count);
        for (int32_t local = 0; local < count; ++local) {
            std::string identifier;
            Document document;
            double raw_total = 0.0;
            double effective_total = 0.0;
            read_spool_record(in, header_, identifier, document,
                raw_total, effective_total);
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
                    "Invalid document spool record extent");
            }
        }
        update_peak_block_bytes(block, peak_block_bytes_);
    }

    std::filesystem::path path_;
    bool remove_on_destruction_ = true;
    DocumentSpoolHeader header_;
    std::vector<uint64_t> offsets_;
    mutable std::ifstream sequential_;
    int64_t cursor_ = 0;
    mutable std::atomic<uint64_t> peak_block_bytes_{0};
};

class SequentialBinaryDocumentSpool final : public DocumentBlockSource {
public:
    SequentialBinaryDocumentSpool(std::filesystem::path path,
        bool remove_on_destruction, const DocumentSpoolHeader& header)
        : path_(std::move(path)),
          remove_on_destruction_(remove_on_destruction),
          header_(header) {
        if (std::filesystem::file_size(path_) != header_.index_begin) {
            throw std::runtime_error(
                "Invalid sequential document spool extent");
        }
        sequential_.open(path_, std::ios::binary);
        if (!sequential_) {
            throw std::runtime_error(
                "Cannot open sequential document spool: "
                + path_.string());
        }
        reset();
    }

    ~SequentialBinaryDocumentSpool() override {
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
        sequential_.clear();
        sequential_.seekg(
            static_cast<std::streamoff>(header_.payload_begin));
        if (!sequential_) {
            throw std::runtime_error(
                "Cannot reset sequential document spool");
        }
    }

    bool next(
        DocumentBlock& block, int32_t maximum_documents) override {
        if (maximum_documents <= 0) {
            throw std::invalid_argument(
                "Sequential document spool block size must be positive");
        }
        if (cursor_ >= documents()) {
            block.clear();
            return false;
        }
        const int32_t count = static_cast<int32_t>(
            std::min<int64_t>(maximum_documents, documents() - cursor_));
        prepare_document_block(block, cursor_, count);
        for (int32_t local = 0; local < count; ++local) {
            std::string identifier;
            Document document;
            double raw_total = 0.0;
            double effective_total = 0.0;
            read_spool_record(sequential_, header_, identifier, document,
                raw_total, effective_total);
            block.identifiers.push_back(std::move(identifier));
            block.counts.push_back(std::move(document));
            block.raw_totals(local) = raw_total;
            block.effective_totals(local) = effective_total;
        }
        cursor_ += count;
        if (cursor_ == documents()) {
            const auto end = sequential_.tellg();
            if (end < 0
                || static_cast<uint64_t>(end) != header_.index_begin) {
                throw std::runtime_error(
                    "Invalid sequential document spool payload extent");
            }
        }
        update_peak_block_bytes(block, peak_block_bytes_);
        return true;
    }

private:
    std::filesystem::path path_;
    bool remove_on_destruction_ = true;
    DocumentSpoolHeader header_;
    std::ifstream sequential_;
    int64_t cursor_ = 0;
    std::atomic<uint64_t> peak_block_bytes_{0};
};

} // namespace

struct BinaryDocumentSpoolWriter::Impl {
    std::filesystem::path path;
    std::filesystem::path partial_path;
    int32_t features = 0;
    std::ofstream out;
    DocumentSpoolHeader header;
    std::vector<uint64_t> offsets;
    DocumentSpoolMode mode = DocumentSpoolMode::Indexed;
    bool finished = false;

    Impl(std::filesystem::path input_path, int32_t input_features,
        DocumentSpoolMode input_mode)
        : path(std::move(input_path)),
          partial_path(path.string() + ".partial"),
          features(input_features),
          mode(input_mode),
          out(partial_path, std::ios::binary | std::ios::trunc) {
        if (features <= 0 || !out) {
            throw std::runtime_error(
                "Cannot create document spool: "
                + partial_path.string());
        }
        header.features = static_cast<uint64_t>(features);
        write_spool_header(out, header);
        if (!out) {
            throw std::runtime_error(
                "Cannot initialize document spool");
        }
        if (mode == DocumentSpoolMode::Indexed) {
            offsets.push_back(header.payload_begin);
        }
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
    const std::filesystem::path& path, int32_t features,
    DocumentSpoolMode mode)
    : impl_(std::make_unique<Impl>(path, features, mode)) {
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
        || !std::isfinite(effective_total) || effective_total < 0.0) {
        throw std::invalid_argument(
            "Invalid document spool append");
    }
    for (size_t j = 0; j < document.ids.size(); ++j) {
        if (document.ids[j] >= static_cast<uint32_t>(impl_->features)
            || !std::isfinite(document.cnts[j])
            || document.cnts[j] < 0.0) {
            throw std::invalid_argument(
                "Invalid document spool count");
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
            "Failed writing document spool");
    }
    if (impl_->mode == DocumentSpoolMode::Indexed) {
        impl_->offsets.push_back(static_cast<uint64_t>(end));
    }
}

std::unique_ptr<IndexedDocumentSource>
BinaryDocumentSpoolWriter::finish(bool remove_on_destruction) {
    if (!impl_ || impl_->finished || impl_->header.documents == 0
        || impl_->mode != DocumentSpoolMode::Indexed) {
        throw std::runtime_error(
            "Cannot finalize indexed document spool");
    }
    impl_->header.index_begin =
        static_cast<uint64_t>(impl_->out.tellp());
    write_spool_bytes(impl_->out, impl_->offsets.data(),
        impl_->offsets.size() * sizeof(uint64_t),
        &impl_->header.index_checksum);
    impl_->out.flush();
    if (!impl_->out) {
        throw std::runtime_error(
            "Failed finalizing document spool index");
    }
    impl_->out.seekp(0);
    write_spool_header(impl_->out, impl_->header);
    impl_->out.close();
    if (!impl_->out) {
        throw std::runtime_error(
            "Failed finalizing document spool header");
    }
    const std::filesystem::path path = impl_->path;
    std::filesystem::rename(impl_->partial_path, path);
    impl_->finished = true;
    return std::make_unique<BinaryDocumentSpool>(
        path, remove_on_destruction, impl_->header,
        std::move(impl_->offsets));
}

std::unique_ptr<DocumentBlockSource>
BinaryDocumentSpoolWriter::finish_sequential(
    bool remove_on_destruction) {
    if (!impl_ || impl_->finished || impl_->header.documents == 0
        || impl_->mode != DocumentSpoolMode::Sequential) {
        throw std::runtime_error(
            "Cannot finalize sequential document spool");
    }
    impl_->header.index_begin =
        static_cast<uint64_t>(impl_->out.tellp());
    impl_->out.flush();
    if (!impl_->out) {
        throw std::runtime_error(
            "Failed finalizing sequential document spool payload");
    }
    impl_->out.seekp(0);
    write_spool_header(impl_->out, impl_->header);
    impl_->out.close();
    if (!impl_->out) {
        throw std::runtime_error(
            "Failed finalizing sequential document spool header");
    }
    const std::filesystem::path path = impl_->path;
    std::filesystem::rename(impl_->partial_path, path);
    impl_->finished = true;
    return std::make_unique<SequentialBinaryDocumentSpool>(
        path, remove_on_destruction, impl_->header);
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

} // namespace punkst
