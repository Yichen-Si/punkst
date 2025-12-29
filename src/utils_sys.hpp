#pragma once

#include <filesystem>
#include <random>
#include <unistd.h>
#include <sys/wait.h>
#include <cerrno>
#include <iostream>
#include <fstream>
#include <memory>
#include <cstdint>
#include <cctype>
#include <cstring>
#include <string_view>
#include <queue>
#include <thread>
#include <algorithm>
#include <functional>

#include <tbb/parallel_pipeline.h>
#include <tbb/parallel_sort.h>
#include <tbb/concurrent_vector.h>
#include <tbb/task_arena.h>

#include "utils.h"
#include "error.hpp"

bool createDirectory(const std::string& dir);

std::filesystem::path makeTempDir(const std::filesystem::path& parent, size_t maxTries = 100);

struct ScopedTempDir {
    std::filesystem::path path;
    bool enabled;

    ScopedTempDir() : enabled(false) {}
    explicit ScopedTempDir(const std::filesystem::path& parent)
      : path(makeTempDir(parent)), enabled(true) {}

    // Move constructor to allow returning from functions
    ScopedTempDir(ScopedTempDir&& other) noexcept
        : path(std::move(other.path)), enabled(true) {
        other.enabled = false; // Prevent the moved-from object from deleting the directory
    }
    // Disable copy constructor and assignment
    ScopedTempDir(const ScopedTempDir&) = delete;
    ScopedTempDir& operator=(const ScopedTempDir&) = delete;

    void init(const std::filesystem::path& parent) {
        path = makeTempDir(parent);
        enabled = true;
    }

    ~ScopedTempDir() {
        if (enabled) {
            std::error_code ec;
            std::filesystem::remove_all(path, ec);
            if (ec) {
                // In a destructor, we shouldn't throw. We should just log the error.
                warning("Failed to remove temporary directory %s: %s", path.c_str(), ec.message().c_str());
            }
        }
    }
};

bool checkOutputWritable(const std::string& outFile, bool newFile = true);

int sys_sort(const char* infile, const char* outfile = nullptr,
             const std::vector<std::string>& flags = {});
int pipe_cat_sort(const std::vector<std::string>& in_files, const std::string& out_file, uint32_t n_threads);

// Robust write: write all bytes or return false on fatal error
bool write_all(int fd, const void* buf, size_t len);

// compute block boundaries for processing a plain text file in parallel
void computeBlocks(std::vector<std::pair<std::streampos, std::streampos>>& blocks, const std::string& inFile, int32_t nThreads, int32_t nskip = 0);

// Iterator for lines
class BoundedReadline {
public:
    BoundedReadline(const std::string &filename, uint64_t start, uint64_t end)
        : startOffset(start), endOffset(end)
    {
        file = std::make_unique<std::ifstream>(filename, std::ios::binary);
        if (!file || !file->is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        file->seekg(startOffset);
        currentPos = startOffset;
    }

    bool next(std::string &line) {
        // Check that we haven't passed the tile's end.
        if (!file || currentPos >= endOffset) {
            return false;
        }
        if (!std::getline(*file, line)) {
            return false;
        }
        currentPos += static_cast<uint64_t>(line.size()) + 1;
        return true;
    }

private:
    std::unique_ptr<std::ifstream> file;
    uint64_t startOffset;
    uint64_t endOffset;
    uint64_t currentPos;
};


class ExternalSorter {
public:
    using Comparator = std::function<bool(const std::string&, const std::string&)>;

    // Sorts lines by the first token as 8-character hex key
    static void sortBy1stColHex(
        const std::string& inFile, const std::string& outFile,
        size_t maxMemBytes = 512 * 1024 * 1024,
        const std::string& tempDir = "", int nThreads = 1);
    static void sortBy1stColHex_singleThread(
        const std::string& inFile, const std::string& outFile,
        size_t maxMemBytes = 512 * 1024 * 1024,
        const std::string& tempDir = "");

    // Sorts lines from inFile and writes to outFile using external merge sort
    static void sort(const std::string& inFile, const std::string& outFile,
                     Comparator comp, size_t maxMemBytes = 512 * 1024 * 1024,
                     const std::string& tempDir = "");

    static size_t parseMemoryString(const std::string& memStr, size_t defaultVal = 512 * 1024 * 1024);
    static bool firstColumnComparator(const std::string& a, const std::string& b, size_t keyLen = 0);

private:
    struct LineRec {
        std::string line;
        uint32_t key; // parsed from first 8 chars
    };
    struct KeyLess {
        bool operator()(const LineRec& a, const LineRec& b) const noexcept {
            if (a.key != b.key) return a.key < b.key;
            return a.line < b.line;
        }
    };
    struct ChunkJob {
        std::vector<LineRec> recs;
        std::string chunkPath;
    };
    struct MergeNode {
        std::string line;
        uint32_t key;
        size_t chunkIdx;
    };
    // For std::push_heap/pop_heap
    static inline bool heapComp(const MergeNode& a, const MergeNode& b) noexcept {
        if (a.key != b.key) return a.key > b.key;
        return a.line > b.line;
    }
    // Serial reader assuming first 8 chars are hex key
    struct Reader {
        std::ifstream& in;
        const std::filesystem::path& tmpPath;
        size_t perChunkBudget;
        std::atomic<size_t>& chunkIdx;
        bool hasCarry = false;
        std::string carry;

        ChunkJob operator()(tbb::flow_control& fc) {
            std::vector<LineRec> recs;
            recs.reserve(50000);

            size_t mem = 0;
            std::string line;

            auto acceptLine = [&](std::string&& s) {
                LineRec r;
                r.line = std::move(s);
                r.key = hexToUint32(r.line); // only use first 8 chars
                mem += r.line.capacity() + sizeof(LineRec);
                recs.emplace_back(std::move(r));
            };

            if (hasCarry) {
                acceptLine(std::move(carry));
                hasCarry = false;
            }

            while (true) {
                if (!std::getline(in, line)) break;
                const size_t estMem = line.capacity() + sizeof(LineRec);
                if (!recs.empty() && mem + estMem >= perChunkBudget) {
                    carry = std::move(line);
                    hasCarry = true;
                    break;
                }
                acceptLine(std::move(line));
            }

            if (recs.empty()) {
                fc.stop();
                return {};
            }

            const size_t id = chunkIdx.fetch_add(1, std::memory_order_relaxed);
            const std::string chunkPath =
                (tmpPath / ("chunk_" + std::to_string(id) + ".tmp")).string();

            return ChunkJob{std::move(recs), chunkPath};
        }
    };


};
