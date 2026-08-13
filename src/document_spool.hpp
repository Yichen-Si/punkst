#pragma once

#include "dataunits.hpp"

#include <Eigen/Dense>

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace punkst {

struct DocumentBlock {
    int64_t first_document = 0;
    std::vector<std::string> identifiers;
    std::vector<Document> counts;
    Eigen::VectorXd raw_totals;
    Eigen::VectorXd effective_totals;

    void clear();
    int32_t size() const;
};

class DocumentBatchSink {
public:
    virtual ~DocumentBatchSink() = default;
    virtual void capture(std::vector<Document>& batch) = 0;
};

class DocumentBlockSource {
public:
    virtual ~DocumentBlockSource() = default;
    virtual int64_t documents() const = 0;
    virtual int64_t features() const = 0;
    virtual uint64_t storage_bytes() const = 0;
    virtual uint64_t content_checksum() const = 0;
    virtual uint64_t peak_block_bytes() const = 0;
    virtual void reset() = 0;
    virtual bool next(DocumentBlock& block, int32_t maximum_documents) = 0;
};

class IndexedDocumentSource : public DocumentBlockSource {
public:
    virtual void read_range(
        int64_t first_document, int32_t documents,
        DocumentBlock& block) const = 0;
};

enum class DocumentSpoolMode {
    Indexed,
    Sequential
};

class BinaryDocumentSpoolWriter {
public:
    BinaryDocumentSpoolWriter(
        const std::filesystem::path& path, int32_t features,
        DocumentSpoolMode mode = DocumentSpoolMode::Indexed);
    ~BinaryDocumentSpoolWriter();
    BinaryDocumentSpoolWriter(BinaryDocumentSpoolWriter&&) noexcept;
    BinaryDocumentSpoolWriter& operator=(
        BinaryDocumentSpoolWriter&&) noexcept;
    BinaryDocumentSpoolWriter(const BinaryDocumentSpoolWriter&) = delete;
    BinaryDocumentSpoolWriter& operator=(
        const BinaryDocumentSpoolWriter&) = delete;

    void append(const std::string& identifier, const Document& document,
        double raw_total, double effective_total);
    std::unique_ptr<IndexedDocumentSource> finish(
        bool remove_on_destruction = true);
    std::unique_ptr<DocumentBlockSource> finish_sequential(
        bool remove_on_destruction = true);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::unique_ptr<IndexedDocumentSource> open_binary_document_spool(
    const std::filesystem::path& path,
    bool remove_on_destruction = false);

} // namespace punkst
