#include "tileoperator.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <algorithm>
#include <map>

int32_t TileOperator::query(float qxmin, float qxmax, float qymin, float qymax) {
    queryBox_ = Rectangle<float>(qxmin, qymin, qxmax, qymax);
    bounded_ = true;
    blocks_.clear();
    for (auto &b : blocks_all_) {
        int32_t rel = queryBox_.intersect(Rectangle<float>(b.idx.xmin, b.idx.ymin, b.idx.xmax, b.idx.ymax));
        if (rel==0) {continue;}
        blocks_.push_back({ b.idx, rel==3});
    }
    if (blocks_.empty()) {
        return 0;
    }
    idx_block_ = 0;
    openDataStream();
    openBlock(blocks_[0]);
    return int32_t(blocks_.size());
}

int32_t TileOperator::next(PixTopProbs<float>& out) {
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out);
    }
    if (mode_ & 0x1) { // Binary mode
        if (mode_ & 0x4) { // int32
            PixTopProbs<int32_t> temp;
            if (!temp.read(dataStream_, k_)) {
                done_ = true;
                return -1;
            }
            out.x = static_cast<float>(temp.x);
            out.y = static_cast<float>(temp.y);
            out.ks = std::move(temp.ks);
            out.ps = std::move(temp.ps);
        } else { // float
            if (!out.read(dataStream_, k_)) {
                done_ = true;
                return -1;
            }
        }
        if (mode_ & 0x2) {
            out.x *= formatInfo_.pixelResolution;
            out.y *= formatInfo_.pixelResolution;
        }
        return 1;
    }
    // Text mode
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
    if (blocks_.empty()) return;

    for (auto& b : blocks_) {
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
    if (formatInfo_.magic == PUNKST_INDEX_MAGIC) {
        // Print header info
        printf("##Tile size: %d\n", formatInfo_.tileSize);
        printf("##Pixel resolution: %.2f\n", formatInfo_.pixelResolution);
        printf("##Coordinate type: %s\n", (formatInfo_.coordType == 1) ? "int32" : "float");
        printf("##Top K stored: %d\n", formatInfo_.topK);
        if (mode_ & 0x1) {
            printf("##Record size: %u bytes\n", formatInfo_.recordSize);
        }
        if (formatInfo_.xmin < formatInfo_.xmax && formatInfo_.ymin < formatInfo_.ymax) {
            printf("##Bound: xmin %.2f, xmax %.2f, ymin %.2f, ymax %.2f\n",
                formatInfo_.xmin, formatInfo_.xmax, formatInfo_.ymin, formatInfo_.ymax);
        }
    }
    printf("#start\tend\trow\tcol\tnpts\txmin\txmax\tymin\tymax\n");
    for (const auto& b : blocks_all_) {
        printf("%lu\t%lu\t%d\t%d\t%u\t%d\t%d\t%d\t%d\n",
            b.idx.st, b.idx.ed, b.idx.row, b.idx.col, b.idx.n,
            b.idx.xmin, b.idx.xmax, b.idx.ymin, b.idx.ymax);
    }
}

void TileOperator::reorgTiles(const std::string& outPrefix, int32_t tileSize) {
    if (blocks_.empty()) {
        error("No blocks found in index");
    }
    if (tileSize <= 0) {
        tileSize = formatInfo_.tileSize;
    }
    assert(tileSize > 0);

    classifyBlocks(tileSize);
    openDataStream();

    if (mode_ & 0x1) {reorgTilesBinary(outPrefix, tileSize); return;}

    std::map<TileKey, std::vector<size_t>> tileMainBlocks;
    std::map<TileKey, std::vector<std::string>> boundaryLines;
    std::map<TileKey, IndexEntryF> boundaryInfo;

    int boundaryBlocksCount = 0;
    int mainBlocksCount = 0;
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto& b = blocks_[i];
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
            if (mode_ & 0x2) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
            }

            int32_t c = static_cast<int32_t>(std::floor(x / tileSize));
            int32_t r = static_cast<int32_t>(std::floor(y / tileSize));
            TileKey key{r, c};
            boundaryLines[key].emplace_back(lineStart, lineLen);
            if (boundaryInfo.find(key) == boundaryInfo.end()) {
                IndexEntryF idx(r, c);
                tile2bound(key, idx.xmin, idx.xmax, idx.ymin, idx.ymax, tileSize);
                boundaryInfo.emplace(key, std::move(idx));
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

    // Write index header
    IndexHeader idxHeader = formatInfo_;
    idxHeader.tileSize = tileSize;
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
        error("Error writing header to index output file: %s", outIndex.c_str());
    }

    if (!write_all(fdMain, headerLine_.data(), headerLine_.size())) {
        error("Error writing header");
    }

    size_t currentOffset = headerLine_.size();

    // 1. Process tiles with main blocks
    for (const auto& kv : tileMainBlocks) {
        TileKey tile = kv.first;
        const auto& indices = kv.second;

        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n = 0;
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, tileSize);

        // Write main blocks
        for (size_t i : indices) {
            const auto& mb = blocks_[i];
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
        return;
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

    uint64_t magic;
    mode_ = 0;
    globalBox_.reset();
    if (in.read(reinterpret_cast<char*>(&magic), sizeof(magic))) {
        in.seekg(0);
        if (magic == PUNKST_INDEX_MAGIC) {
            if (!in.read(reinterpret_cast<char*>(&formatInfo_), sizeof(formatInfo_)))
                error("Error reading index header from file: %s", indexFile.c_str());
            if (formatInfo_.recordSize > 0) {
                mode_ = 1;
            }
            if (formatInfo_.pixelResolution > 0) {
                mode_ |= 2;
            }
            if (formatInfo_.coordType == 1) {
                mode_ |= 4;
            }
            k_ = formatInfo_.topK;
            globalBox_ = Rectangle<float>(formatInfo_.xmin, formatInfo_.ymin,
                                          formatInfo_.xmax, formatInfo_.ymax);
        }
    } else {
        error("Error reading index file: %s", indexFile.c_str());
    }

    blocks_all_.clear();
    IndexEntryF idx;
    while (in.read(reinterpret_cast<char*>(&idx), sizeof(idx))) {
        blocks_all_.push_back({idx, false});
        if (magic != PUNKST_INDEX_MAGIC) {
            globalBox_.extendToInclude(
                Rectangle<int32_t>(idx.xmin, idx.ymin, idx.xmax, idx.ymax));
        }
    }
    if (blocks_all_.empty())
        error("No index entries loaded from %s", indexFile.c_str());
    blocks_ = blocks_all_;
    notice("Loaded index with %lu blocks", blocks_all_.size());
}

void TileOperator::openBlock(Block& blk) {
    dataStream_.clear();  // clear EOF flags
    dataStream_.seekg(blk.idx.st);
    pos_ = blk.idx.st;
}

bool TileOperator::parseLine(const std::string& line, PixTopProbs<float>& R) const {
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < icol_max_+1) return false;
    if (!str2float(tokens[icol_x_], R.x) ||
        !str2float(tokens[icol_y_], R.y)) {
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
    if (mode_ & 0x2) {
        R.x *= formatInfo_.pixelResolution;
        R.y *= formatInfo_.pixelResolution;
    }
    return true;
}

int32_t TileOperator::nextBounded(PixTopProbs<float>& out) {
    if (done_ || idx_block_ < 0) return -1;

    if (mode_ & 0x1) { // Binary mode
        while (true) {
            auto &blk = blocks_[idx_block_];
            if (pos_ >= blk.idx.ed) {
                if (++idx_block_ >= (int32_t) blocks_.size()) {
                    done_ = true;
                    return -1;
                }
                openBlock(blocks_[idx_block_]);
                continue;
            }
            // Read one record
            if (mode_ & 0x4) {
                 PixTopProbs<int32_t> temp;
                if (!temp.read(dataStream_, k_)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
                out.x = static_cast<float>(temp.x);
                out.y = static_cast<float>(temp.y);
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
            } else {
                if (!out.read(dataStream_, k_)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
            }
            pos_ += formatInfo_.recordSize;
            if (mode_ & 0x2) {
                out.x *= formatInfo_.pixelResolution;
                out.y *= formatInfo_.pixelResolution;
            }
            if (blk.contained || queryBox_.contains(out.x, out.y)) {
                return 1;
            }
        }
    }

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
        PixTopProbs<float> rec;
        if (!parseLine(line, rec)) return 0;
        if (blk.contained || queryBox_.contains(rec.x, rec.y)) {
            out = std::move(rec);
            return 1;
        }
        // else skip it, keep reading
    }
}

void TileOperator::reorgTilesBinary(const std::string& outPrefix, int32_t tileSize) {
    std::map<TileKey, std::vector<size_t>> tileMainBlocks;
    std::map<TileKey, std::vector<char>> boundaryData;
    std::map<TileKey, IndexEntryF> boundaryInfo;

    int boundaryBlocksCount = 0;
    int mainBlocksCount = 0;
    uint32_t recordSize = formatInfo_.recordSize;
    std::vector<char> recBuf(recordSize);

    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto& b = blocks_[i];
        if (b.contained) {
            TileKey key{b.row, b.col};
            tileMainBlocks[key].push_back(i);
            mainBlocksCount++;
            continue;
        }

        boundaryBlocksCount++;
        dataStream_.seekg(b.idx.st);
        size_t len = b.idx.ed - b.idx.st;
        size_t nRecords = len / recordSize;

        for (size_t j = 0; j < nRecords; ++j) {
            if (!dataStream_.read(recBuf.data(), recordSize))
                error("%s: Read error block %lu", __func__, i);

            float x, y;
            if (mode_ & 0x4) {
                int32_t xi = *reinterpret_cast<int32_t*>(recBuf.data());
                int32_t yi = *reinterpret_cast<int32_t*>(recBuf.data() + 4);
                x = static_cast<float>(xi);
                y = static_cast<float>(yi);
            } else {
                x = *reinterpret_cast<float*>(recBuf.data());
                y = *reinterpret_cast<float*>(recBuf.data() + 4);
            }
            if (mode_ & 0x2) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
            }

            int32_t c = static_cast<int32_t>(std::floor(x / tileSize));
            int32_t r = static_cast<int32_t>(std::floor(y / tileSize));
            TileKey key{r, c};
            auto& d = boundaryData[key];
            size_t csz = d.size();
            d.resize(csz + recordSize);
            std::memcpy(d.data() + csz, recBuf.data(), recordSize);

            if (boundaryInfo.find(key) == boundaryInfo.end()) {
                IndexEntryF idx(r, c);
                tile2bound(key, idx.xmin, idx.xmax, idx.ymin, idx.ymax, tileSize);
                boundaryInfo.emplace(key, std::move(idx));
            }
        }
    }

    notice("Found %d main blocks and %d boundary blocks", mainBlocksCount, boundaryBlocksCount);

    std::string outFile = outPrefix + ".bin";
    std::string outIndex = outPrefix + ".index";

    int fdMain = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdMain < 0) error("Cannot open output file %s", outFile.c_str());
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());

    IndexHeader idxHeader = formatInfo_;
    idxHeader.tileSize = tileSize;
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) error("Index header write error");

    size_t currentOffset = 0;
    std::vector<TileKey> allKeys;
    for (const auto& kv : tileMainBlocks) allKeys.push_back(kv.first);
    for (const auto& kv : boundaryData) allKeys.push_back(kv.first);
    std::sort(allKeys.begin(), allKeys.end());
    allKeys.erase(std::unique(allKeys.begin(), allKeys.end()), allKeys.end());
    int32_t nTiles = 0;
    for (const auto& tile : allKeys) {
        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n = 0;
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, tileSize);

        if (tileMainBlocks.count(tile)) {
            for (size_t i : tileMainBlocks[tile]) {
                const auto& mb = blocks_[i];
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
                }
            }
        }

        if (boundaryData.count(tile)) {
            const auto& d = boundaryData[tile];
            if (!d.empty()) {
                if (!write_all(fdMain, d.data(), d.size())) error("Write error");
                newEntry.n += d.size() / recordSize;
            }
        }

        newEntry.ed = lseek(fdMain, 0, SEEK_CUR);
        currentOffset = newEntry.ed;
        if (newEntry.n > 0) {
            if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index write error");
            nTiles++;
        }
    }
    close(fdMain);
    close(fdIndex);
    notice("Reorganized into %d tiles. Output written to %s\n Index written to %s", nTiles, outFile.c_str(), outIndex.c_str());
    return;
}

void TileOperator::dumpTSV(const std::string& outPrefix, int32_t probDigits, int32_t coordDigits) {
    if (!(mode_ & 0x1)) {
        error("dumpTSV only supports binary mode files");
    }
    if (blocks_.empty()) {
        warning("%s: No data to write", __func__);
        return;
    }
    resetReader();

    // Set up output files/stream
    FILE* fp = stdout;
    int fdIndex = -1;
    std::string tsvFile;
    bool writeIndex = false;

    if (!outPrefix.empty() && outPrefix != "-") {
        tsvFile = outPrefix + ".tsv";
        std::string indexFile = outPrefix + ".index";
        fp = fopen(tsvFile.c_str(), "w");
        if (!fp) error("Error opening output file: %s", tsvFile.c_str());

        fdIndex = open(indexFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (fdIndex < 0) error("Cannot open output index %s", indexFile.c_str());
        writeIndex = true;
    }
    // Write header
    std::string headerStr = "#x\ty";
    for (int i = 0; i < k_; ++i) {
        headerStr += "\tK" + std::to_string(i + 1) + "\tP" + std::to_string(i + 1);
    }
    headerStr += "\n";
    if (fprintf(fp, "%s", headerStr.c_str()) < 0) {
        error("Error writing header to TSV file");
    }

    if (writeIndex) {
        IndexHeader idxHeader = formatInfo_;
        idxHeader.recordSize = 0; // 0 for TSV
        idxHeader.coordType = 0; // 0 for float
        idxHeader.pixelResolution = 1.0f; // will apply scaling
        if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
            error("Error writing header to index output file");
        }
    }

    bool isInt32 = (mode_ & 0x4);
    bool applyRes = (mode_ & 0x2);
    float res = formatInfo_.pixelResolution;

    // Track current offset in the output TSV file
    long currentOffset = ftell(fp);

    for (const auto& blk : blocks_) {
        dataStream_.seekg(blk.idx.st);
        size_t len = blk.idx.ed - blk.idx.st;
        size_t recSize = formatInfo_.recordSize;
        if (recSize == 0) error("Record size is 0 in binary mode");
        bool checkBound = bounded_ && !blk.contained;
        size_t nRecs = len / recSize;

        // We will accumulate index entry info for this block
        IndexEntryF newEntry = blk.idx;
        newEntry.st = currentOffset;
        // n, xmin, xmax, ymin, ymax are copied from the binary index entry
        // This assumes the binary index is correct and aligned with the data we read.

        for(size_t i=0; i<nRecs; ++i) {
            float x, y;
            std::vector<int32_t> ks(k_);
            std::vector<float> ps(k_);

            if (isInt32) {
                PixTopProbs<int32_t> temp;
                if (!temp.read(dataStream_, k_)) break;
                x = static_cast<float>(temp.x);
                y = static_cast<float>(temp.y);
                ks = std::move(temp.ks);
                ps = std::move(temp.ps);
            } else {
                PixTopProbs<float> temp;
                if (!temp.read(dataStream_, k_)) break;
                x = temp.x;
                y = temp.y;
                ks = std::move(temp.ks);
                ps = std::move(temp.ps);
            }

            if (applyRes) {
                x *= res;
                y *= res;
            }

            if (checkBound && !queryBox_.contains(x, y)) {
                continue;
            }

            if (fprintf(fp, "%.*f\t%.*f", coordDigits, x, coordDigits, y) < 0)
                error("%s: Write error", __func__);
            for (int k = 0; k < k_; ++k) {
                if (fprintf(fp, "\t%d\t%.*e", ks[k], probDigits, ps[k]) < 0)
                    error("%s: Write error", __func__);
            }
            if (fprintf(fp, "\n") < 0) error("%s: Write error", __func__);
        }

        currentOffset = ftell(fp);
        newEntry.ed = currentOffset;

        if (writeIndex) {
             if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) {
                 error("Error writing index entry");
             }
        }
    }

    if (fp != stdout) fclose(fp);
    if (fdIndex >= 0) close(fdIndex);

    if (writeIndex) {
        notice("Dumped TSV to %s and index to %s.index", tsvFile.c_str(), outPrefix.c_str());
    }
}
