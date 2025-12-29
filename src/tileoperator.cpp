#include "tileoperator.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <algorithm>
#include <map>
#include <set>
#include <memory>

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
    if ((mode_ & 0x8) == 0) { // regular grid
        tile_lookup_.clear();
        for (size_t i = 0; i < blocks_.size(); ++i) {
            auto& b = blocks_[i];
            b.row = b.idx.row; b.col = b.idx.col;
            tile_lookup_[{b.row, b.col}] = i;
        }
    }
    return int32_t(blocks_.size());
}

int32_t TileOperator::next(PixTopProbs<float>& out, bool rawCoord) {
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
        if (!rawCoord && (mode_ & 0x2)) {
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
        if (!parseLine(line, out)) return 0;
        if (rawCoord && (mode_ & 0x2)) {
            out.x /= formatInfo_.pixelResolution;
            out.y /= formatInfo_.pixelResolution;
        }
        return 1;
    }
}

int32_t TileOperator::next(PixTopProbs<int32_t>& out) {
    if (mode_ & 0x1) {assert(mode_ & 0x4);}
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out);
    }
    if (mode_ & 0x1) { // Binary mode
        if (out.read(dataStream_, k_)) {
            return 1;
        }
        if (dataStream_.eof()) {
            done_ = true;
            return -1;
        }
        error("%s: Corrupted data", __func__);
    }
    // Text mode
    std::string line;
    while (true) {
        if (!std::getline(dataStream_, line)) {
            done_ = true;
            return -1;
        }
        if (line.empty() || line[0] == '#') {continue;}
        if (!parseLine(line, out)) return 0;
        return 1;
    }
}

int32_t TileOperator::loadTileToMap(const TileKey& key,
    std::map<std::pair<int32_t, int32_t>, PixTopProbs<int32_t>>& pixelMap) {
    assert((mode_ & 0x8) == 0);
    if ((mode_ & 0x4) == 0) {
        assert((mode_ & 0x2) == 0 && formatInfo_.pixelResolution > 0);}
    pixelMap.clear();
    if (tile_lookup_.find(key) == tile_lookup_.end()) return 0;

    openDataStream();
    size_t idx = tile_lookup_.at(key);
    TileInfo& blk = blocks_[idx];
    openBlock(blk);
    float res = formatInfo_.pixelResolution;
    PixTopProbs<int32_t> rec;
    while (pos_ < blk.idx.ed) {
        bool success = false;
        if (mode_ & 0x4) { // int32
            if (mode_ & 0x1) { // Binary
                if (rec.read(dataStream_, k_)) {
                    pos_ += formatInfo_.recordSize;
                    success = true;
                }
            } else {
                std::string line;
                if (std::getline(dataStream_, line)) {
                    pos_ += line.size() + 1;
                    success = parseLine(line, rec);
                }
            }
        } else { // float, scale then round to int
            PixTopProbs<float> temp;
            if (mode_ & 0x1) { // Binary
                if (temp.read(dataStream_, k_)) {
                    pos_ += formatInfo_.recordSize;
                    success = true;
                }
            } else {
                std::string line;
                if (std::getline(dataStream_, line)) {
                    pos_ += line.size() + 1;
                    success = parseLine(line, temp);
                }
            }
            if (success) {
                rec.x = static_cast<int32_t>(std::floor(temp.x / res));
                rec.y = static_cast<int32_t>(std::floor(temp.y / res));
                rec.ks = std::move(temp.ks);
                rec.ps = std::move(temp.ps);
            }
        }
        if (success) {
            pixelMap[{rec.x, rec.y}] = rec;
        } else if (!dataStream_.eof()) {
            error("%s: Corrupted data", __func__);
        }
    }
    return static_cast<int32_t>(pixelMap.size());
}

void TileOperator::merge(const std::vector<std::string>& otherFiles, const std::string& outPrefix, std::vector<uint32_t> k2keep, bool binaryOutput) {
    std::string outIndex = outPrefix + ".index";
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());
    if (k2keep.size() > 0) {assert(k2keep.size() == otherFiles.size() + 1);}

    // 1. Setup operators
    std::vector<std::unique_ptr<TileOperator>> ops;
    std::vector<TileOperator*> opPtrs;
    opPtrs.push_back(this); // current object
    for (const auto& f : otherFiles) {
        std::string idxFile = f.substr(0, f.find_last_of('.')) + ".index";
        struct stat buffer;
        if (stat(idxFile.c_str(), &buffer) != 0) {
            error("%s: Index file %s not found", __func__, idxFile.c_str());
        }
        std::string df = f;
        ops.push_back(std::make_unique<TileOperator>(df, idxFile));
        opPtrs.push_back(ops.back().get());
    }
    uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    if (k2keep.size() == 0) {
        for (auto* op : opPtrs) {
            k2keep.push_back(op->getK());
        }
    } else {
        for (uint32_t i = 0; i < nSources; ++i) {
            if (k2keep[i] > opPtrs[i]->getK()) {
                warning("%s: Invalid value k (%d) specified for the %d-th source", __func__, k2keep[i], i);
                k2keep[i] = opPtrs[i]->getK();
            }
        }
    }
    if (nSources > 7) {
        int32_t k = *std::min_element(k2keep.begin(), k2keep.end());
        k2keep.assign(nSources, k);
        warning("%s: More than 7 files to merge, keep %d values each", __func__, k);
    }
    int32_t totalK = std::accumulate(k2keep.begin(), k2keep.end(), 0);

    // 2. Identify common tiles (Intersection)
    std::set<TileKey> commonTiles;
    if (opPtrs[0]->tile_lookup_.empty()) {
        warning("%s: No tiles in the base dataset", __func__);
        return;
    }
    for (const auto& kv : opPtrs[0]->tile_lookup_) {
        commonTiles.insert(kv.first);
    }
    for (uint32_t i = 1; i < nSources; ++i) {
        std::set<TileKey> currentTiles;
        for (const auto& kv : opPtrs[i]->tile_lookup_) {
            if (commonTiles.count(kv.first)) {
                currentTiles.insert(kv.first);
            }
        }
        commonTiles = currentTiles;
    }
    if (commonTiles.empty()) {
        warning("%s: No overlapping tiles found for merge", __func__);
        return;
    }

    // 3. Prepare output
    std::string outFile;
    FILE* fp = nullptr;
    int fdMain = -1;
    long currentOffset = 0;

    // Index metadata
    IndexHeader idxHeader = formatInfo_;
    idxHeader.mode |= 0x4; // int32
    idxHeader.coordType = 1;
    if (!idxHeader.packKvec(k2keep)) {
        warning("%s: Too many input fields", __func__);
    }

    if (binaryOutput) {
        outFile = outPrefix + ".bin";
        fdMain = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (fdMain < 0) error("Cannot open output file %s", outFile.c_str());
        idxHeader.mode |= 0x1; // Binary mode
        idxHeader.recordSize = sizeof(int32_t) * 2 + sizeof(int32_t) * totalK + sizeof(float) * totalK;
        currentOffset = 0;
    } else {
        outFile = outPrefix + ".tsv";
        fp = fopen(outFile.c_str(), "w");
        if (!fp) error("Cannot open output file %s", outFile.c_str());
        idxHeader.mode &= ~0x1; // Text mode
        idxHeader.recordSize = 0;

        // TSV header
        std::string headerStr = "#x\ty";
        uint32_t idx = 1;
        for (uint32_t i = 0; i < nSources; ++i) {
            for (int j = 0; j < k2keep[i]; ++j) {
                headerStr += "\tK" + std::to_string(idx) + "\tP" + std::to_string(idx);
                idx++;
            }
        }
        headerStr += "\n";
        fprintf(fp, "%s", headerStr.c_str());
        currentOffset = ftell(fp);
    }

    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) error("Index write error");

    // 4. Process tiles
    notice("%s: Start merging %u files", __func__, nSources);
    for (const auto& tile : commonTiles) {
        std::map<std::pair<int32_t, int32_t>, PixTopProbs<int32_t>> mergedMap;
        bool first = true;
        for (uint32_t i = 0; i < nSources; ++i) {
            TileOperator* op = opPtrs[i];
            std::map<std::pair<int32_t, int32_t>, PixTopProbs<int32_t>> currentMap;
            if (op->loadTileToMap(tile, currentMap) == 0) {
                warning("%s: Tile (%d, %d) has no data in source %d", __func__, tile.row, tile.col, i);
                mergedMap.clear();
                break;
            }
            notice("%s: Loaded tile (%d, %d) from source %d with %lu pixels",
                   __func__, tile.row, tile.col, i, currentMap.size());
            if (first) {
                if (k_ > k2keep[i]) { // Trim to k2keep[i]
                    for (auto& kv : currentMap) {
                        kv.second.ks.resize(k2keep[i]);
                        kv.second.ps.resize(k2keep[i]);
                    }
                }
                mergedMap = std::move(currentMap);
                first = false;
                continue;
            }
            auto it = mergedMap.begin();
            while (it != mergedMap.end()) {
                auto it2 = currentMap.find(it->first);
                if (it2 == currentMap.end()) {
                    it = mergedMap.erase(it); // Intersect
                    continue;
                }
                // Concatenate
                it->second.ks.insert(it->second.ks.end(),
                    it2->second.ks.begin(), it2->second.ks.begin() + k2keep[i]);
                it->second.ps.insert(it->second.ps.end(),
                    it2->second.ps.begin(), it2->second.ps.begin() + k2keep[i]);
                ++it;
            }
        }
        notice("%s: Merged tile (%d, %d) with %lu pixels shared by all sources", __func__, tile.row, tile.col, mergedMap.size());
        if (mergedMap.empty()) {
            continue;
        }
        // Write tile
        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n  = mergedMap.size();
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, formatInfo_.tileSize);
        for (const auto& kv : mergedMap) {
            const auto& p = kv.second;
            if (binaryOutput) {
                if (p.write(fdMain) < 0) error("Write error");
            } else {
                fprintf(fp, "%d\t%d", p.x, p.y);
                for (size_t i = 0; i < p.ks.size(); ++i) {
                    fprintf(fp, "\t%d\t%.4e", p.ks[i], p.ps[i]);
                }
                fprintf(fp, "\n");
            }
        }
        if (binaryOutput) {
            currentOffset = lseek(fdMain, 0, SEEK_CUR);
        } else {
            currentOffset = ftell(fp);
        }
        newEntry.ed = currentOffset;
        if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index entry write error");
    }

    if (binaryOutput) {
        close(fdMain);
    } else {
        fclose(fp);
    }
    close(fdIndex);
    notice("Merged %u files (%lu shared tiles) to %s", nSources, commonTiles.size(), outFile.c_str());
}

void TileOperator::annotate(const std::string& ptPrefix, uint32_t icol_x, uint32_t icol_y, const std::string& outPrefix) {
    std::string ptData = ptPrefix + ".tsv";
    std::string ptIndex = ptPrefix + ".index";
    TileReader reader(ptData, ptIndex);
    assert(reader.getTileSize() == formatInfo_.tileSize);
    assert(icol_x != icol_y);
    std::string outFile = outPrefix + ".tsv";
    std::string outIndex = outPrefix + ".index";
    FILE* fp = fopen(outFile.c_str(), "w");
    if (!fp) error("Cannot open output file %s", outFile.c_str());
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());
    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;
    uint32_t ntok = std::max(icol_x, icol_y) + 1;

    // Header?
    if (!reader.headerLine.empty()) {
        std::string headerStr = reader.headerLine;
        for (uint32_t i = 1; i <= k_; ++i) {
            headerStr += "\tK" + std::to_string(i) + "\tP" + std::to_string(i);
        }
        fprintf(fp, "%s\n", headerStr.c_str());
    }
    long currentOffset = ftell(fp);
    // Write index header
    IndexHeader idxHeader = formatInfo_;
    idxHeader.mode &= ~(0x7);
    idxHeader.recordSize = 0;
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) error("Index header write error");

    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    notice("%s: Start annotating query with %lu tiles", __func__, tiles.size());
    for (const auto& tile : tiles) {
        std::map<std::pair<int32_t, int32_t>, PixTopProbs<int32_t>> pixelMap;
        if (loadTileToMap(tile, pixelMap) <= 0) {
            debug("%s: Query tile (%d, %d) is not in the annotation dataset", __func__, tile.row, tile.col);
            continue;
        }
        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) continue;
        IndexEntryF newEntry(tile.row, tile.col);
        newEntry.st = currentOffset;
        newEntry.n = 0;
        tile2bound(tile, newEntry.xmin, newEntry.xmax, newEntry.ymin, newEntry.ymax, formatInfo_.tileSize);
        std::string s;
        while (it->next(s)) {
            if (s.empty() || s[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", s, ntok+1, true, true, true);
            if (tokens.size() < ntok) {
                error("%s: Invalid line: %s", __func__, s.c_str());
            }
            float x, y;
            if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)){
                error("%s: Invalid coordinates in line: %s", __func__, s.c_str());
            }
            int32_t ix = static_cast<int32_t>(std::floor(x / res));
            int32_t iy = static_cast<int32_t>(std::floor(y / res));
            auto pit = pixelMap.find({ix, iy});
            if (pit == pixelMap.end()) {
                continue;
            }
            fprintf(fp, "%s", s.c_str());
            for (size_t k = 0; k < pit->second.ks.size(); ++k) {
                fprintf(fp, "\t%d\t%.4e", pit->second.ks[k], pit->second.ps[k]);
            }
            fprintf(fp, "\n");
            newEntry.n++;
        }
        notice("%s: Annotated tile (%d, %d) with %u points", __func__, tile.row, tile.col, newEntry.n);
        currentOffset = ftell(fp);
        newEntry.ed = currentOffset;
        if (newEntry.n > 0) {
             if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) error("Index entry write error");
        }
    }

    fclose(fp);
    close(fdIndex);
    notice("Annotation finished, data written to %s", outFile.c_str());
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
        printf("##Flag: 0x%x\n", formatInfo_.mode);
        printf("##Tile size: %d\n", formatInfo_.tileSize);
        printf("##Pixel resolution: %.2f\n", formatInfo_.pixelResolution);
        printf("##Coordinate type: %s\n", (mode_ & 0x4) ? "int32" : "float");
        if (k_ > 0) {
            printf("##Result set: %u", kvec_[0]);
            for (size_t i = 1; i < kvec_.size(); ++i) {
                printf(",%u", kvec_[i]);
            }
            printf("\n");
        }
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
    idxHeader.mode &= ~0x8;
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
        if (tokens[i] == "X" || tokens[i] == "Y") {
            std::transform(tokens[i].begin(), tokens[i].end(), tokens[i].begin(), ::tolower);
        }
        header[tokens[i]] = i;
    }
    if (header.find("x") == header.end() || header.find("y") == header.end()) {
        return;
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
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic)) ||
         magic != PUNKST_INDEX_MAGIC) {
        loadIndexLegacy(indexFile); return;
    }

    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(&formatInfo_), sizeof(formatInfo_)))
        error("%s: Error reading index file: %s", __func__, indexFile.c_str());
    mode_ = formatInfo_.mode;
    if ((mode_ & 0x8) == 0) {assert(formatInfo_.tileSize > 0);}
    if (mode_ & 0x2) {assert(mode_ & 0x4);}
    k_ = formatInfo_.parseKvec(kvec_);
    globalBox_ = Rectangle<float>(formatInfo_.xmin, formatInfo_.ymin,
                                  formatInfo_.xmax, formatInfo_.ymax);
    blocks_all_.clear();
    tile_lookup_.clear();
    IndexEntryF idx;
    while (in.read(reinterpret_cast<char*>(&idx), sizeof(idx))) {
        blocks_all_.push_back({idx, false});
    }
    if (blocks_all_.empty())
        error("No index entries loaded from %s", indexFile.c_str());
    blocks_ = blocks_all_;
    if ((mode_ & 0x8) == 0) { // regular grid
        for (size_t i = 0; i < blocks_.size(); ++i) {
            auto& b = blocks_[i];
            b.row = b.idx.row; b.col = b.idx.col;
            tile_lookup_[{b.row, b.col}] = i;
        }
    }
    notice("Loaded index with %lu blocks", blocks_all_.size());
}

void TileOperator::loadIndexLegacy(const std::string& indexFile) {
    std::ifstream in(indexFile, std::ios::binary);
    if (!in.is_open())
        error("Error opening index file: %s", indexFile.c_str());
    globalBox_.reset();
    blocks_all_.clear();
    tile_lookup_.clear();
    IndexEntryF_legacy idx;
    while (in.read(reinterpret_cast<char*>(&idx), sizeof(idx))) {
        IndexEntryF idx1 = IndexEntryF(idx.st, idx.ed, idx.n,
            idx.xmin, idx.xmax, idx.ymin, idx.ymax);
        blocks_all_.push_back({idx1, false});
        globalBox_.extendToInclude(
            Rectangle<int32_t>(idx.xmin, idx.ymin, idx.xmax, idx.ymax));
    }
    if (blocks_all_.empty())
        error("No index entries loaded from %s", indexFile.c_str());
    blocks_ = blocks_all_;
    notice("Loaded index with %lu blocks", blocks_all_.size());
}

void TileOperator::openBlock(TileInfo& blk) {
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
    if (mode_ & 0x2) {
        R.x *= formatInfo_.pixelResolution;
        R.y *= formatInfo_.pixelResolution;
    }
    if (k_ <= 0) return true;

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

int32_t TileOperator::nextBounded(PixTopProbs<float>& out, bool rawCoord) {
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
            if (blk.contained && rawCoord) {
                return 1;
            }
            float x = out.x, y = out.y;
            if (mode_ & 0x2) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
                if (!rawCoord) {
                    out.x = x;
                    out.y = y;
                }
            }
            if (blk.contained || queryBox_.contains(x, y)) {
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
            if (rawCoord && (mode_ & 0x2)) {
                out.x /= formatInfo_.pixelResolution;
                out.y /= formatInfo_.pixelResolution;
            }
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
    idxHeader.mode &= ~0x8;
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
        idxHeader.mode &= ~0x7;
        idxHeader.recordSize = 0; // 0 for TSV
        idxHeader.coordType = 0; // 0 for float
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

int32_t TileOperator::nextBounded(PixTopProbs<int32_t>& out) {
    if (mode_ & 0x1) {assert(mode_ & 0x4);}
    if (done_ || idx_block_ < 0) return -1;
    std::string line;
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

        if (mode_ & 0x1) { // Binary mode
            if (!out.read(dataStream_, k_)) {
                error("%s: Corrupted data or invalid index", __func__);
            }
            pos_ += formatInfo_.recordSize;
        } else {
            if (!std::getline(dataStream_, line))
                error("%s: Corrupted data or invalid index", __func__);
            pos_ += line.size() + 1; // +1 for newline
            if (line.empty() || line[0] == '#') {continue;}
            if (!parseLine(line, out))
                error("%s: Corrupted data or invalid index", __func__);
        }

        if (blk.contained) {return 1;}
        float x = static_cast<float>(out.x);
        float y = static_cast<float>(out.y);
        if (mode_ & 0x2) {
            x *= formatInfo_.pixelResolution;
            y *= formatInfo_.pixelResolution;
        }
        if (blk.contained || queryBox_.contains(x, y)) {
            return 1;
        }
    }
}

bool TileOperator::parseLine(const std::string& line, PixTopProbs<int32_t>& R) const {
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() <= icol_max_ + 1) return false;
    if (!str2int32(tokens[icol_x_], R.x) ||
        !str2int32(tokens[icol_y_], R.y)) {
        return false;
    }
    if (k_ <= 0) return true;

    R.ks.resize(k_);
    R.ps.resize(k_);
    for (int i = 0; i < k_; ++i) {
        if (!str2int32(tokens[icol_ks_[i]], R.ks[i]) ||
            !str2float(tokens[icol_ps_[i]], R.ps[i])) {
            return false;
        }
    }
    return true;
}
