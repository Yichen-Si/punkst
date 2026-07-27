#pragma once

#include "document_spool.hpp"
#include "utils_sys.hpp"

#include <chrono>
#include <cctype>
#include <filesystem>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

enum class TrainingCountCacheMode {
    Off,
    On,
    Auto
};

inline TrainingCountCacheMode parse_training_count_cache_mode(
    const std::string& value) {
    if (value == "off") return TrainingCountCacheMode::Off;
    if (value == "on") return TrainingCountCacheMode::On;
    if (value == "auto") return TrainingCountCacheMode::Auto;
    throw std::invalid_argument(
        "--count-cache must be off, on, or auto");
}

inline uint64_t parse_training_count_cache_memory_budget(
    const std::string& value) {
    if (value.empty()) {
        throw std::invalid_argument(
            "--count-cache-memory-budget must not be empty");
    }
    std::string digits = value;
    uint64_t multiplier = 1;
    const char suffix = static_cast<char>(
        std::toupper(static_cast<unsigned char>(digits.back())));
    if (suffix == 'K' || suffix == 'M' || suffix == 'G') {
        digits.pop_back();
        multiplier = suffix == 'K' ? 1024ull
            : suffix == 'M' ? 1024ull * 1024ull
                            : 1024ull * 1024ull * 1024ull;
    }
    if (digits.empty()) {
        throw std::invalid_argument(
            "Invalid --count-cache-memory-budget: " + value);
    }
    for (const char digit : digits) {
        if (!std::isdigit(static_cast<unsigned char>(digit))) {
            throw std::invalid_argument(
                "Invalid --count-cache-memory-budget: " + value);
        }
    }
    size_t parsed = 0;
    uint64_t amount = 0;
    try {
        amount = std::stoull(digits, &parsed);
    } catch (const std::exception&) {
        throw std::invalid_argument(
            "Invalid --count-cache-memory-budget: " + value);
    }
    if (parsed != digits.size()
        || amount > std::numeric_limits<uint64_t>::max() / multiplier) {
        throw std::invalid_argument(
            "Invalid --count-cache-memory-budget: " + value);
    }
    return amount * multiplier;
}

inline void validate_training_count_cache_options(
    const std::string& mode, const std::string& memory_budget) {
    try {
        static_cast<void>(parse_training_count_cache_mode(mode));
        static_cast<void>(
            parse_training_count_cache_memory_budget(memory_budget));
    } catch (const std::exception& ex) {
        error("%s", ex.what());
    }
}

struct TrainingCountCacheCliOptions {
    std::string mode = "auto";
    std::string memory_budget = "1G";
    std::string temp_dir;
};

inline void add_training_count_cache_options(
    ParamList& parameters, TrainingCountCacheCliOptions& options) {
    parameters
        .add_option("count-cache",
            "Repeated-pass count cache: off, on, or auto; auto uses it for at least two count passes",
            options.mode)
        .add_option("count-cache-memory-budget",
            "Maximum resident count-cache bytes; accepts K, M, or G suffixes",
            options.memory_budget)
        .add_option("temp-dir",
            "Directory to store temporary files",
            options.temp_dir);
}

class TrainingCountCache final : public uac::DocumentBatchSink {
public:
    using ResidentBatches = std::vector<std::vector<Document>>;

    TrainingCountCache(const TrainingCountCacheCliOptions& options,
        bool use_10x, int32_t count_passes, int32_t features)
        : mode_(parse_training_count_cache_mode(options.mode)),
          memory_budget_(
              parse_training_count_cache_memory_budget(
                  options.memory_budget)),
          features_(features),
          parent_(options.temp_dir.empty()
              ? std::filesystem::temp_directory_path()
              : std::filesystem::path(options.temp_dir)) {
        enabled_ = !use_10x
            && mode_ != TrainingCountCacheMode::Off
            && (mode_ == TrainingCountCacheMode::On
                || count_passes >= 2);
        if (!enabled_) {
            notice("Training count cache resolved to off%s",
                use_10x ? " for resident 10X input" : "");
            return;
        }
        construction_start_ = std::chrono::steady_clock::now();
        if (memory_budget_ == 0) {
            try {
                begin_sequential_storage();
            } catch (const std::exception& ex) {
                handle_failure(ex.what());
            }
        }
    }

    bool enabled() const {
        return enabled_;
    }

    uac::DocumentBatchSink* sink() {
        return enabled_ ? this : nullptr;
    }

    const ResidentBatches* resident_batches() const {
        return enabled_ && finalized_ && !source_
            ? &resident_batches_ : nullptr;
    }

    uac::DocumentBlockSource* source() {
        return source_.get();
    }

    void capture(std::vector<Document>& batch) override {
        if (!enabled_ || batch.empty()) return;
        try {
            if (writer_) {
                append_batch_to_disk(batch);
                total_documents_ += batch.size();
                return;
            }
            const uint64_t batch_bytes = resident_batch_bytes(batch);
            size_t next_capacity = resident_batches_.capacity();
            if (resident_batches_.size() == next_capacity) {
                next_capacity = next_capacity == 0 ? 1
                    : next_capacity > std::numeric_limits<size_t>::max() / 2
                        ? std::numeric_limits<size_t>::max()
                        : next_capacity * 2;
            }
            const uint64_t projected = saturating_add(
                sizeof(ResidentBatches),
                saturating_add(
                    saturating_add(resident_payload_bytes_, batch_bytes),
                    saturating_multiply(next_capacity,
                        sizeof(ResidentBatches::value_type))));
            if (projected > memory_budget_) {
                begin_sequential_storage();
                append_batch_to_disk(batch);
                total_documents_ += batch.size();
                return;
            }
            if (resident_batches_.size()
                    == resident_batches_.capacity()) {
                resident_batches_.reserve(next_capacity);
            }
            const size_t documents = batch.size();
            resident_batches_.push_back(std::move(batch));
            resident_payload_bytes_ =
                saturating_add(resident_payload_bytes_, batch_bytes);
            total_documents_ += documents;
        } catch (const std::bad_alloc&) {
            try {
                begin_sequential_storage();
                append_batch_to_disk(batch);
                total_documents_ += batch.size();
            } catch (const std::exception& ex) {
                handle_failure(ex.what());
            }
        } catch (const std::exception& ex) {
            handle_failure(ex.what());
        }
    }

    void finish() {
        if (!enabled_ || finalized_) return;
        const auto begin = std::chrono::steady_clock::now();
        if (total_documents_ == 0) {
            handle_failure("Cannot finalize empty training count cache");
            return;
        }
        try {
            if (writer_) {
                source_ = writer_->finish_sequential();
                writer_.reset();
            }
        } catch (const std::exception& ex) {
            writer_.reset();
            handle_failure(ex.what());
            return;
        }
        finalized_ = true;
        const double construction_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now()
                    - construction_start_).count();
        const double finalization_seconds =
            std::chrono::duration<double>(
                std::chrono::steady_clock::now() - begin).count();
        if (source_) {
            notice("Training count cache ready: storage=sequential, %lld documents, %llu disk bytes, memory budget %llu bytes (first cached epoch and build %.3f s; finalized in %.3f s)",
                static_cast<long long>(source_->documents()),
                static_cast<unsigned long long>(
                    source_->storage_bytes()),
                static_cast<unsigned long long>(memory_budget_),
                construction_seconds, finalization_seconds);
        } else {
            notice("Training count cache ready: storage=memory, %lld documents, %llu resident bytes, memory budget %llu bytes (first cached epoch and build %.3f s; finalized in %.3f s)",
                static_cast<long long>(total_documents_),
                static_cast<unsigned long long>(resident_bytes()),
                static_cast<unsigned long long>(memory_budget_),
                construction_seconds, finalization_seconds);
        }
    }

    uint64_t resident_bytes() const {
        return saturating_add(sizeof(ResidentBatches),
            saturating_add(resident_payload_bytes_,
                saturating_multiply(resident_batches_.capacity(),
                    sizeof(ResidentBatches::value_type))));
    }

private:
    static uint64_t saturating_add(uint64_t left, uint64_t right) {
        return right > std::numeric_limits<uint64_t>::max() - left
            ? std::numeric_limits<uint64_t>::max()
            : left + right;
    }

    static uint64_t saturating_multiply(uint64_t left, uint64_t right) {
        return left != 0
                && right > std::numeric_limits<uint64_t>::max() / left
            ? std::numeric_limits<uint64_t>::max()
            : left * right;
    }

    static uint64_t resident_batch_bytes(
        const std::vector<Document>& batch) {
        uint64_t bytes = saturating_multiply(
            batch.capacity(), sizeof(Document));
        for (const Document& document : batch) {
            bytes = saturating_add(bytes,
                saturating_multiply(
                    document.ids.capacity(), sizeof(uint32_t)));
            bytes = saturating_add(bytes,
                saturating_multiply(
                    document.cnts.capacity(), sizeof(double)));
        }
        return bytes;
    }

    void append_batch_to_disk(std::vector<Document>& batch) {
        for (Document& document : batch) {
            writer_->append(std::to_string(disk_documents_written_),
                document, document.get_raw_sum(), document.get_sum());
            ++disk_documents_written_;
        }
    }

    void begin_sequential_storage() {
        if (writer_) return;
        directory_.init(parent_);
        writer_ = std::make_unique<uac::BinaryDocumentSpoolWriter>(
            directory_.path / "training-counts.bin", features_,
            uac::DocumentSpoolMode::Sequential);
        const uint64_t spill_documents = total_documents_;
        for (std::vector<Document>& batch : resident_batches_) {
            append_batch_to_disk(batch);
        }
        ResidentBatches empty;
        resident_batches_.swap(empty);
        resident_payload_bytes_ = 0;
        notice("Training count cache exceeded its %llu-byte resident budget after %llu documents; spilling to sequential temporary storage",
            static_cast<unsigned long long>(memory_budget_),
            static_cast<unsigned long long>(spill_documents));
    }

    void handle_failure(const std::string& message) {
        enabled_ = false;
        finalized_ = false;
        source_.reset();
        writer_.reset();
        ResidentBatches empty;
        resident_batches_.swap(empty);
        resident_payload_bytes_ = 0;
        if (mode_ == TrainingCountCacheMode::On) {
            error("Training count cache failed: %s", message.c_str());
        }
        warning("Training count cache disabled after failure: %s",
            message.c_str());
    }

    TrainingCountCacheMode mode_ = TrainingCountCacheMode::Auto;
    bool enabled_ = false;
    bool finalized_ = false;
    uint64_t memory_budget_ = 0;
    int32_t features_ = 0;
    std::filesystem::path parent_;
    ScopedTempDir directory_;
    ResidentBatches resident_batches_;
    uint64_t resident_payload_bytes_ = 0;
    uint64_t total_documents_ = 0;
    uint64_t disk_documents_written_ = 0;
    std::unique_ptr<uac::BinaryDocumentSpoolWriter> writer_;
    std::unique_ptr<uac::DocumentBlockSource> source_;
    std::chrono::steady_clock::time_point construction_start_;
};
