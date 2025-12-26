#include "tileoperator.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <algorithm>
#include <map>

int32_t TileOperator::query(double qxmin, double qxmax, double qymin, double qymax) {
    qxmin_ = qxmin;
    qxmax_ = qxmax;
    qymin_ = qymin;
    qymax_ = qymax;
    bounded_ = true;
    blocks_.clear();
    for (auto &b : blocks_all_) {
        if (b.idx.xmax <= qxmin_ || b.idx.xmin >= qxmax_ ||
            b.idx.ymax <= qymin_ || b.idx.ymin >= qymax_) {continue;}
        bool inside = (b.idx.xmin >= qxmin_ && b.idx.xmax <= qxmax_ &&
        b.idx.ymin >= qymin_ && b.idx.ymax <= qymax_);
        blocks_.push_back({ b.idx, inside });
    }
    if (blocks_.empty()) {
        return 0;
    }
    idx_block_ = 0;
    openDataStream();
    openBlock(blocks_[0]);
    return int32_t(blocks_.size());
}

int32_t TileOperator::next(PixelFactorResult& out) {
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out);
    }
    std::string line;
    while (true) {
        if (!std::getline(dataStream_, line)) {
            done_ = true;
            return -1;
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        return parseLine(line, out) ? 1 : 0;
    }
}

void TileOperator::classifyBlocks(int32_t tileSize) {
    if (blocks_all_.empty()) return;

    for (auto& b : blocks_all_) {
        // Check if block is strictly contained in a tile
        float cx = (b.idx.xmin + b.idx.xmax) / 2.0f;
        float cy = (b.idx.ymin + b.idx.ymax) / 2.0f;

        int32_t c = static_cast<int32_t>(std::floor(cx / tileSize));
        int32_t r = static_cast<int32_t>(std::floor(cy / tileSize));

        float tileX0 = c * tileSize;
        float tileX1 = (c + 1) * tileSize;
        float tileY0 = r * tileSize;
        float tileY1 = (r + 1) * tileSize;

        float tol = 1.0f; // 1 unit tolerance

        bool crossesX = (b.idx.xmin < tileX0 - tol) || (b.idx.xmax > tileX1 + tol);
        bool crossesY = (b.idx.ymin < tileY0 - tol) || (b.idx.ymax > tileY1 + tol);

        if (!crossesX && !crossesY) {
            b.contained = true;
            b.row = r;
            b.col = c;
        } else {
            b.contained = false;
        }
    }
}

void TileOperator::printIndex() const {
    printf("#start\tend\tnpts\txmin\txmax\tymin\tymax\n");
    for (const auto& b : blocks_all_) {
        printf("%lu\t%lu\t%u\t%f\t%f\t%f\t%f\n",
            b.idx.st, b.idx.ed, b.idx.n,
            b.idx.xmin, b.idx.xmax, b.idx.ymin, b.idx.ymax);
    }
}

void TileOperator::reorgTiles(const std::string& outPrefix, int32_t tileSize) {
    if (blocks_all_.empty()) {
        error("No blocks found in index");
    }
    classifyBlocks(tileSize);
    openDataStream();

    std::map<TileKey, std::vector<size_t>> tileMainBlocks;
    std::map<TileKey, std::vector<std::string>> boundaryLines;
    std::map<TileKey, IndexEntryF> boundaryInfo;

    notice("Processing blocks...");

    int boundaryBlocksCount = 0;
    int mainBlocksCount = 0;
    for (size_t i = 0; i < blocks_all_.size(); ++i) {
        const auto& b = blocks_all_[i];
        if (b.contained) {
            TileKey key{b.row, b.col};
            tileMainBlocks[key].push_back(i);
            mainBlocksCount++;
            continue;
        }

        boundaryBlocksCount++;
        dataStream_.seekg(b.idx.st);
        size_t len = b.idx.ed - b.idx.st;
        if (len == 0) continue;
        std::vector<char> data(len);
        dataStream_.read(data.data(), len);
        const char* ptr = data.data();
        const char* end = ptr + len;
        const char* lineStart = ptr;
        std::vector<std::string> tokens;

        while (lineStart < end) {
            const char* lineEnd = static_cast<const char*>(memchr(lineStart, '\n', end - lineStart));
            if (!lineEnd) lineEnd = end;

            size_t lineLen = lineEnd - lineStart;
            if (lineLen > 0 && lineStart[lineLen - 1] == '\r') lineLen--;
            if (lineLen == 0 || lineStart[0] == '#') {
                lineStart = lineEnd + 1;
                continue;
            }
            std::string_view lineView(lineStart, lineLen);
            split(tokens, "\t", lineView);
            if (tokens.size() < icol_max_ + 1) {
                error("Insufficient tokens (%lu) in line (block %lu): %.*s.", tokens.size(), i, (int)lineLen, lineStart);
            }

            float x, y;
            if (!str2float(tokens[icol_x_], x) || !str2float(tokens[icol_y_], y)) {
                error("Invalid coordinate values in line: %.*s", (int)lineLen, lineStart);
            }

            int32_t c = static_cast<int32_t>(std::floor(x / tileSize));
            int32_t r = static_cast<int32_t>(std::floor(y / tileSize));
            TileKey key{r, c};
            boundaryLines[key].emplace_back(lineStart, lineLen);
            auto it = boundaryInfo.find(key);
            if (it == boundaryInfo.end()) {
                IndexEntryF idx;
                idx.xmin = x; idx.xmax = x; idx.ymin = y; idx.ymax = y;
                boundaryInfo.emplace(key, std::move(idx));
            } else {
                IndexEntryF& idx = it->second;
                if (x < idx.xmin) idx.xmin = x;
                if (x > idx.xmax) idx.xmax = x;
                if (y < idx.ymin) idx.ymin = y;
                if (y > idx.ymax) idx.ymax = y;
            }
            lineStart = lineEnd + 1;
        }
    }

    notice("Found %d main blocks and %d boundary blocks", mainBlocksCount, boundaryBlocksCount);

    std::string outFile = outPrefix + ".tsv";
    std::string outIndex = outPrefix + ".index";

    int fdMain = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdMain < 0) error("Cannot open output file %s", outFile.c_str());

    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());

    if (!write_all(fdMain, headerLine_.data(), headerLine_.size())) {
        error("Error writing header");
    }

    size_t currentOffset = headerLine_.size();

    // 1. Process tiles with main blocks
    for (const auto& kv : tileMainBlocks) {
        TileKey tile = kv.first;
        const auto& indices = kv.second;

        IndexEntryF newEntry;
        newEntry.st = currentOffset;
        newEntry.n = 0;
        newEntry.xmin = std::numeric_limits<float>::max();
        newEntry.xmax = -std::numeric_limits<float>::max();
        newEntry.ymin = std::numeric_limits<float>::max();
        newEntry.ymax = -std::numeric_limits<float>::max();

        // Write main blocks
        for (size_t i : indices) {
            const auto& mb = blocks_all_[i];
            size_t len = mb.idx.ed - mb.idx.st;
            if (len > 0) {
                dataStream_.seekg(mb.idx.st);
                size_t copied = 0;
                const size_t bufSz = 1024 * 1024;
                std::vector<char> buffer(bufSz);
                while (copied < len) {
                    size_t toRead = std::min(bufSz, len - copied);
                    dataStream_.read(buffer.data(), toRead);
                    if (!write_all(fdMain, buffer.data(), toRead)) error("Write error");
                    copied += toRead;
                }
                newEntry.n += mb.idx.n;
                newEntry.xmin = std::min(newEntry.xmin, mb.idx.xmin);
                newEntry.xmax = std::max(newEntry.xmax, mb.idx.xmax);
                newEntry.ymin = std::min(newEntry.ymin, mb.idx.ymin);
                newEntry.ymax = std::max(newEntry.ymax, mb.idx.ymax);
            }
        }

        // Append boundary lines if any
        if (boundaryLines.count(tile)) {
            const auto& lines = boundaryLines[tile];
            for (size_t i = 0; i < lines.size(); ++i) {
                const std::string& l = lines[i];
                if (!write_all(fdMain, l.data(), l.size())) error("Write error");
                if (!write_all(fdMain, "\n", 1)) error("Write error");

                newEntry.n++;
            }
            const auto& binfo = boundaryInfo[tile];
            newEntry.xmin = std::min(newEntry.xmin, binfo.xmin);
            newEntry.xmax = std::max(newEntry.xmax, binfo.xmax);
            newEntry.ymin = std::min(newEntry.ymin, binfo.ymin);
            newEntry.ymax = std::max(newEntry.ymax, binfo.ymax);
            boundaryLines.erase(tile);
        }

        newEntry.ed = lseek(fdMain, 0, SEEK_CUR);
        currentOffset = newEntry.ed;
        if (newEntry.n > 0) {
            if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index write error");
        }
    }

    // 2. Process remaining boundary-only tiles
    for (const auto& kv : boundaryLines) {
        TileKey tile = kv.first;
        const auto& lines = kv.second;
        IndexEntryF& newEntry = boundaryInfo[tile];
        newEntry.st = currentOffset;
        for (size_t i = 0; i < lines.size(); ++i) {
            const std::string& l = lines[i];
            if (!write_all(fdMain, l.data(), l.size())) error("Write error");
            if (!write_all(fdMain, "\n", 1)) error("Write error");
            newEntry.n++;
        }
        newEntry.ed = lseek(fdMain, 0, SEEK_CUR);
        currentOffset = newEntry.ed;
        if (newEntry.n > 0) {
            if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index write error");
        }
    }

    close(fdMain);
    close(fdIndex);
    notice("Reorganization completed. Output written to %s\n Index written to %s", outFile.c_str(), outIndex.c_str());
}

void TileOperator::parseHeaderLine() {
    std::ifstream ss(dataFile_);
    if (!ss.is_open()) {
        error("Error opening data file: %s", dataFile_.c_str());
    }
    std::string line, headerLine_;
    int32_t nline = 0;
    while (std::getline(ss, line)) {
        if (nline > 100) {
            break;
        }
        nline++;
        if (line.empty() || line.substr(0, 2) == "##") {
            continue;
        }
        if (line[0] == '#') {
            headerLine_ = line;
        } else {
            break;
        }
    }
    if (headerLine_.empty()) {
        error("Cannot recognize header from file %s", dataFile_.c_str());
    }

    line = headerLine_.substr(1); // skip initial '#'
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    std::unordered_map<std::string, uint32_t> header;
    for (uint32_t i = 0; i < tokens.size(); ++i) {
        header[tokens[i]] = i;
    }
    if (header.find("x") == header.end() || header.find("y") == header.end()) {
        error("Header file must contain x and y columns");
    }
    icol_x_ = header["x"];
    icol_y_ = header["y"];
    int32_t k = 1;
    while (header.find("K" + std::to_string(k)) != header.end() && header.find("P" + std::to_string(k)) != header.end()) {
        icol_ks_.push_back(header["K" + std::to_string(k)]);
        icol_ps_.push_back(header["P" + std::to_string(k)]);
        k++;
    }
    if (icol_ks_.empty()) {
        warning("No K and P columns found in the header");
    }
    k_ = k - 1;
    icol_max_ = std::max(icol_x_, icol_y_);
    for (int i = 0; i < k_; ++i) {
        icol_max_ = std::max(icol_max_, std::max(icol_ks_[i], icol_ps_[i]));
    }
}

void TileOperator::parseHeaderFile(const std::string& headerFile) {
    std::string line;
    int32_t k = 1;
    std::ifstream headerStream(headerFile);
    if (!headerStream.is_open()) {
        error("Error opening header file: %s", headerFile.c_str());
    }
    // Load the JSON header file.
    nlohmann::json header;
    try {
        headerStream >> header;
    } catch (const std::exception& idx) {
        error("Error parsing JSON header: %s", idx.what());
    }
    headerStream.close();
    icol_x_ = header["x"];
    icol_y_ = header["y"];
    while (header.contains("K" + std::to_string(k)) && header.contains("P" + std::to_string(k))) {
        icol_ks_.push_back(header["K" + std::to_string(k)]);
        icol_ps_.push_back(header["P" + std::to_string(k)]);
        k++;
    }
    if (icol_ks_.empty()) {
        error("No K and P columns found in the header");
    }
    k_ = k - 1;
    icol_max_ = std::max(icol_x_, icol_y_);
    for (int i = 0; i < k_; ++i) {
        icol_max_ = std::max(icol_max_, std::max(icol_ks_[i], icol_ps_[i]));
    }
}

void TileOperator::loadIndex(const std::string& indexFile) {
    std::ifstream in(indexFile, std::ios::binary);
    if (!in.is_open())
        error("Error opening index file: %s", indexFile.c_str());

    blocks_all_.clear();
    IndexEntryF idx;
    while (in.read(reinterpret_cast<char*>(&idx), sizeof(idx))) {
        blocks_all_.push_back({idx, false});
    }
    if (blocks_all_.empty())
        error("No index entries loaded from %s", indexFile.c_str());
    notice("Loaded index with %lu blocks", blocks_all_.size());
}

void TileOperator::openBlock(Block& blk) {
    dataStream_.clear();  // clear EOF flags
    dataStream_.seekg(blk.idx.st);
    pos_ = blk.idx.st;
}

bool TileOperator::parseLine(const std::string& line, PixelFactorResult& R) const {
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < icol_max_+1) return false;
    if (!str2double(tokens[icol_x_], R.x) ||
        !str2double(tokens[icol_y_], R.y)) {
        warning("%s: Error parsing x,y from line: %s", __func__, line.c_str());
        return false;
    }
    R.ks.resize(k_);
    R.ps.resize(k_);
    for (int i = 0; i < k_; ++i) {
        if (!str2int32(tokens[icol_ks_[i]], R.ks[i]) ||
            !str2float(tokens[icol_ps_[i]], R.ps[i])) {
            warning("%s: Error parsing K,P from line: %s", __func__, line.c_str());
        }
    }
    return true;
}

int32_t TileOperator::nextBounded(PixelFactorResult& out) {
    if (done_ || idx_block_ < 0) return -1;
    std::string line;
    while (true) {
        auto &blk = blocks_[idx_block_];
        if (pos_ >= blk.idx.ed || !std::getline(dataStream_, line)) {
            if (++idx_block_ >= (int32_t) blocks_.size()) {
                done_ = true;
                return -1;
            }
            openBlock(blocks_[idx_block_]);
            continue;
        }
        pos_ += line.size() + 1; // +1 for newline
        if (line.empty() || line[0] == '#') {
            continue;
        }
        PixelFactorResult rec;
        if (!parseLine(line, rec)) return 0;
        if (blk.contained ||
                (rec.x >= qxmin_ && rec.x < qxmax_ &&
                    rec.y >= qymin_ && rec.y < qymax_)) {
            out = std::move(rec);
            return 1;
        }
        // else skip it, keep reading
    }
}
