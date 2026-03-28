#include "flex_io.hpp"

#include "utils.h"

#if PUNKST_ENABLE_REMOTE_IO
#include <curl/curl.h>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <fstream>
#include <optional>
#include <sstream>
#include <thread>

namespace flexio {

namespace {

constexpr const char* kS3Prefix = "s3://";
constexpr const char* kHttpPrefix = "http://";
constexpr const char* kHttpsPrefix = "https://";

struct CurlGlobalState {
#if PUNKST_ENABLE_REMOTE_IO
    CurlGlobalState() {
        const CURLcode rc = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (rc != CURLE_OK) {
            error("%s: curl_global_init failed: %s", __func__, curl_easy_strerror(rc));
        }
    }

    ~CurlGlobalState() {
        curl_global_cleanup();
    }
#else
    CurlGlobalState() = default;
    ~CurlGlobalState() = default;
#endif
};

void ensure_curl_global_init() {
    static const CurlGlobalState state;
    (void)state;
}

bool starts_with(const std::string& value, const char* prefix) {
    return value.rfind(prefix, 0) == 0;
}

std::string to_lower_ascii(const std::string& value) {
    std::string out = value;
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return out;
}

std::optional<uint64_t> parse_content_range_total(const std::string& headers) {
    std::istringstream in(headers);
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        const std::string lower = to_lower_ascii(line);
        constexpr const char* prefix = "content-range:";
        if (lower.rfind(prefix, 0) != 0) {
            continue;
        }
        const size_t slash = line.find('/');
        if (slash == std::string::npos || slash + 1 >= line.size()) {
            continue;
        }
        uint64_t total = 0;
        if (str2uint64(trim(line.substr(slash + 1)), total)) {
            return total;
        }
    }
    return std::nullopt;
}

} // namespace

std::string normalize_uri(const std::string& uri) {
    if (!starts_with(uri, kS3Prefix)) {
        return uri;
    }

    const size_t slash = uri.find('/', std::char_traits<char>::length(kS3Prefix));
    if (slash == std::string::npos || slash + 1 >= uri.size()) {
        return uri;
    }
    const std::string bucket = uri.substr(std::char_traits<char>::length(kS3Prefix),
        slash - std::char_traits<char>::length(kS3Prefix));
    const std::string key = uri.substr(slash + 1);
    return "https://" + bucket + ".s3.amazonaws.com/" + key;
}

bool is_remote_uri(const std::string& uri) {
    const std::string normalized = normalize_uri(uri);
    return starts_with(normalized, kHttpPrefix) || starts_with(normalized, kHttpsPrefix);
}

FlexFileReader::~FlexFileReader() {
    close();
}

bool FlexFileReader::open(const std::string& uri) {
    close();
    path_ = uri;
    file_.open(path_, std::ios::binary);
    if (!file_.is_open()) {
        return false;
    }
    file_.seekg(0, std::ios::end);
    const std::streamoff end = file_.tellg();
    if (end < 0) {
        close();
        return false;
    }
    size_ = static_cast<uint64_t>(end);
    file_.seekg(0, std::ios::beg);
    return file_.good();
}

bool FlexFileReader::read_at(uint64_t offset, uint64_t length, std::string& buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_open()) {
        return false;
    }
    if (length == 0) {
        buffer.clear();
        return true;
    }
    if (offset > size_ || length > size_ - offset) {
        buffer.clear();
        return false;
    }

    buffer.resize(static_cast<size_t>(length));
    file_.clear();
    file_.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!file_) {
        buffer.clear();
        return false;
    }
    file_.read(buffer.data(), static_cast<std::streamsize>(length));
    if (static_cast<uint64_t>(file_.gcount()) != length) {
        buffer.clear();
        return false;
    }
    return true;
}

bool FlexFileReader::is_open() const {
    return file_.is_open();
}

void FlexFileReader::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (file_.is_open()) {
        file_.close();
    }
    size_ = 0;
}

FlexHttpReader::~FlexHttpReader() {
    close();
}

size_t FlexHttpReader::append_to_string(void* ptr, size_t size, size_t nmemb, void* userdata) {
    if (userdata == nullptr) {
        return 0;
    }
    std::string* out = static_cast<std::string*>(userdata);
    out->append(static_cast<const char*>(ptr), size * nmemb);
    return size * nmemb;
}

bool FlexHttpReader::init_handle() {
#if !PUNKST_ENABLE_REMOTE_IO
    return false;
#else
    ensure_curl_global_init();
    curl_ = static_cast<void*>(curl_easy_init());
    if (curl_ == nullptr) {
        return false;
    }
    configure_common_options();
    return true;
#endif
}

void FlexHttpReader::configure_common_options() {
#if PUNKST_ENABLE_REMOTE_IO
    CURL* curl = static_cast<CURL*>(curl_);
    curl_easy_setopt(curl, CURLOPT_FOLLOWLOCATION, 1L);
    curl_easy_setopt(curl, CURLOPT_MAXREDIRS, 8L);
    curl_easy_setopt(curl, CURLOPT_TCP_KEEPALIVE, 1L);
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_UNRESTRICTED_AUTH, 1L);
    curl_easy_setopt(curl, CURLOPT_ACCEPT_ENCODING, "identity");
    curl_easy_setopt(curl, CURLOPT_USERAGENT, "punkst-flexio/1.0");
#endif
}

bool FlexHttpReader::probe_range_support() {
#if !PUNKST_ENABLE_REMOTE_IO
    return false;
#else
    std::string headers;
    std::string body;
    CURL* curl = static_cast<CURL*>(curl_);
    curl_easy_setopt(curl, CURLOPT_URL, url_.c_str());
    curl_easy_setopt(curl, CURLOPT_NOBODY, 0L);
    curl_easy_setopt(curl, CURLOPT_RANGE, "0-0");
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, append_to_string);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, &headers);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, append_to_string);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &body);

    const CURLcode rc = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    if (rc != CURLE_OK) {
        return false;
    }
    if (code != 206) {
        if (code == 200 && body.size() == 1) {
            size_ = 1;
            return true;
        }
        return false;
    }

    if (const std::optional<uint64_t> total = parse_content_range_total(headers)) {
        size_ = *total;
    }
    return true;
#endif
}

bool FlexHttpReader::probe_size_with_head() {
#if !PUNKST_ENABLE_REMOTE_IO
    return false;
#else
    CURL* curl = static_cast<CURL*>(curl_);
    curl_easy_setopt(curl, CURLOPT_URL, url_.c_str());
    curl_easy_setopt(curl, CURLOPT_RANGE, static_cast<const char*>(nullptr));
    curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, nullptr);
    curl_easy_setopt(curl, CURLOPT_HEADERDATA, nullptr);
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, nullptr);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, nullptr);
    curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);

    const CURLcode rc = curl_easy_perform(curl);
    long code = 0;
    curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
    curl_off_t length = -1;
    const CURLcode lengthRc = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD_T, &length);
    if (lengthRc != CURLE_OK || length < 0) {
        double legacyLength = -1.0;
        const CURLcode legacyRc = curl_easy_getinfo(curl, CURLINFO_CONTENT_LENGTH_DOWNLOAD, &legacyLength);
        if (legacyRc == CURLE_OK && legacyLength > 0) {
            length = static_cast<curl_off_t>(legacyLength);
        }
    }
    curl_easy_setopt(curl, CURLOPT_NOBODY, 0L);
    if (rc != CURLE_OK || (code != 200 && code != 204 && code != 206)) {
        return false;
    }
    if (length > 0) {
        size_ = static_cast<uint64_t>(length);
    }
    return true;
#endif
}

bool FlexHttpReader::open(const std::string& uri) {
    close();
    url_ = normalize_uri(uri);
    if (!init_handle()) {
        return false;
    }
    if (!probe_range_support()) {
        close();
        return false;
    }
    probe_size_with_head();
    is_open_ = true;
    return true;
}

bool FlexHttpReader::read_at(uint64_t offset, uint64_t length, std::string& buffer) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (!is_open()) {
        return false;
    }
    if (length == 0) {
        buffer.clear();
        return true;
    }

    const std::string range = std::to_string(offset) + "-" + std::to_string(offset + length - 1);
    constexpr int32_t maxAttempts = 4;
#if !PUNKST_ENABLE_REMOTE_IO
    (void)offset;
    (void)length;
    buffer.clear();
    return false;
#else
    CURL* curl = static_cast<CURL*>(curl_);
    for (int32_t attempt = 0; attempt < maxAttempts; ++attempt) {
        buffer.clear();
        curl_easy_setopt(curl, CURLOPT_URL, url_.c_str());
        curl_easy_setopt(curl, CURLOPT_NOBODY, 0L);
        curl_easy_setopt(curl, CURLOPT_RANGE, range.c_str());
        curl_easy_setopt(curl, CURLOPT_HEADERFUNCTION, nullptr);
        curl_easy_setopt(curl, CURLOPT_HEADERDATA, nullptr);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, append_to_string);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &buffer);

        const CURLcode rc = curl_easy_perform(curl);
        long code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &code);
        if (rc == CURLE_OK) {
            if (code == 206 && buffer.size() == length) {
                return true;
            }
            if (code == 200 && offset == 0 && size_ == length && buffer.size() == length) {
                return true;
            }
        }
        if (code == 429 || code == 500 || code == 502 || code == 503 || code == 504) {
            std::this_thread::sleep_for(std::chrono::milliseconds(150 << attempt));
            continue;
        }
        break;
    }
    buffer.clear();
    return false;
#endif
}

bool FlexHttpReader::is_open() const {
    return curl_ != nullptr && is_open_;
}

void FlexHttpReader::close() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (curl_ != nullptr) {
#if PUNKST_ENABLE_REMOTE_IO
        curl_easy_cleanup(static_cast<CURL*>(curl_));
#endif
        curl_ = nullptr;
    }
    size_ = 0;
    is_open_ = false;
}

std::unique_ptr<FlexReader> FlexReaderFactory::create_reader(const std::string& uri) {
    std::unique_ptr<FlexReader> reader;
    if (is_remote_uri(uri) || starts_with(uri, kS3Prefix)) {
#if !PUNKST_ENABLE_REMOTE_IO
        error("%s: remote URI support was disabled at build time", __func__);
#else
        reader = std::make_unique<FlexHttpReader>();
#endif
    } else {
        reader = std::make_unique<FlexFileReader>();
    }
    if (!reader->open(uri)) {
        return nullptr;
    }
    return reader;
}

} // namespace flexio
