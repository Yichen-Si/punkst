// from pmpoint

#pragma once

#include <cstdint>
#include <fstream>
#include <memory>
#include <mutex>
#include <string>

namespace flexio {

std::string normalize_uri(const std::string& uri);
bool is_remote_uri(const std::string& uri);

class FlexReader {
public:
    virtual ~FlexReader() = default;
    virtual bool open(const std::string& uri) = 0;
    virtual bool read_at(uint64_t offset, uint64_t length, std::string& buffer) = 0;
    virtual uint64_t size_hint() const { return 0; }
    virtual bool is_open() const = 0;
    virtual void close() = 0;
};

class FlexFileReader final : public FlexReader {
public:
    FlexFileReader() = default;
    ~FlexFileReader() override;

    bool open(const std::string& uri) override;
    bool read_at(uint64_t offset, uint64_t length, std::string& buffer) override;
    uint64_t size_hint() const override { return size_; }
    bool is_open() const override;
    void close() override;

private:
    std::string path_;
    std::ifstream file_;
    uint64_t size_ = 0;
    mutable std::mutex mutex_;
};

class FlexHttpReader final : public FlexReader {
public:
    FlexHttpReader() = default;
    ~FlexHttpReader() override;

    bool open(const std::string& uri) override;
    bool read_at(uint64_t offset, uint64_t length, std::string& buffer) override;
    uint64_t size_hint() const override { return size_; }
    bool is_open() const override;
    void close() override;

private:
    static size_t append_to_string(void* ptr, size_t size, size_t nmemb, void* userdata);

    bool init_handle();
    void configure_common_options();
    bool probe_range_support();
    bool probe_size_with_head();

    std::string url_;
    void* curl_ = nullptr;
    uint64_t size_ = 0;
    bool is_open_ = false;
    mutable std::mutex mutex_;
};

class FlexReaderFactory {
public:
    static std::unique_ptr<FlexReader> create_reader(const std::string& uri);
};

} // namespace flexio
