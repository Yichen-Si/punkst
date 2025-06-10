#pragma once

#include "punkst.h"
#include <filesystem>
#include <random>
#include <unistd.h>
#include <sys/wait.h>
#include <cerrno>
#include <iostream>
#include <fstream>
#include <memory>

bool createDirectory(const std::string& dir);

std::filesystem::path makeTempDir(const std::filesystem::path& parent, size_t maxTries = 100);

struct ScopedTempDir {
    std::filesystem::path path;
    explicit ScopedTempDir(const std::filesystem::path& parent)
      : path(makeTempDir(parent)) {}
    ~ScopedTempDir() { std::error_code ec; std::filesystem::remove_all(path, ec); }
};

bool checkOutputWritable(const std::string& outFile, bool newFile = true);

int sys_sort(const char* infile, const char* outfile = nullptr,
             const std::vector<std::string>& flags = {});

// compute block boundaries for processing a plain text file in parallel
void computeBlocks(std::vector<std::pair<std::streampos, std::streampos>>& blocks, const std::string& inFile, int32_t nThreads, int32_t nskip = 0);

// Iterator for lines
class BoundedReadline {
public:
    BoundedReadline(const std::string &filename, std::streampos start, std::streampos end)
        : startOffset(start), endOffset(end)
    {
        file = std::make_unique<std::ifstream>(filename, std::ios::binary);
        if (!file || !file->is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        file->seekg(startOffset);
    }

    // Returns true and sets 'line' if a line is successfully read and within the tile's range.
    bool next(std::string &line) {
        // Check that we haven't passed the tile's end.
        if (!file || file->tellg() >= endOffset) {
            return false;
        }
        std::streampos before = file->tellg();
        if (!std::getline(*file, line)) {
            return false;
        }
        return true;
    }

private:
    std::unique_ptr<std::ifstream> file;
    std::streampos startOffset;
    std::streampos endOffset;
};
