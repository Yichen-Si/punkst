#include "tileoperator.hpp"
#include "region_query.hpp"
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <tbb/blocked_range.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace {

struct RegionTileResult {
    RegionTileState state = RegionTileState::Outside;
    uint32_t nOut = 0;
    bool useCopyRange = false;
    uint64_t copySt = 0;
    uint64_t copyEd = 0;
    std::vector<char> binaryData;
    std::string textData;
};

template<typename Rec, typename ReadTextLineFn, typename DecodeTextFn>
int32_t nextTextRecordImpl(bool& done,
    ReadTextLineFn&& readNextTextLine,
    DecodeTextFn&& decodeText,
    Rec& out) {
    std::string line;
    while (true) {
        if (!readNextTextLine(line)) {
            done = true;
            return -1;
        }
        if (line.empty() || line[0] == '#') {
            continue;
        }
        return decodeText(line, out) ? 1 : 0;
    }
}

template<typename Rec, typename OpenBlockFn, typename ReadBinaryFn, typename ToWorldFn>
int32_t nextBoundedBinaryRecordImpl(bool& done, int32_t& idxBlock, uint64_t& pos,
    std::vector<TileInfo>& blocks, uint32_t recordSize, const Rectangle<float>& queryBox,
    OpenBlockFn&& openBlock, ReadBinaryFn&& readBinary, ToWorldFn&& toWorld, Rec& out) {
    if (done || idxBlock < 0) {
        return -1;
    }
    while (true) {
        auto& blk = blocks[idxBlock];
        if (pos >= blk.idx.ed) {
            if (++idxBlock >= static_cast<int32_t>(blocks.size())) {
                done = true;
                return -1;
            }
            openBlock(blocks[idxBlock]);
            continue;
        }
        readBinary(out);
        pos += recordSize;
        if (blk.contained) {
            return 1;
        }
        const auto xy = toWorld(out);
        if (queryBox.contains(xy.first, xy.second)) {
            return 1;
        }
    }
}

template<typename Rec, typename ReadTextLineFn, typename OpenBlockFn, typename DecodeTextFn, typename ToWorldFn>
int32_t nextBoundedTextRecordImpl(bool& done, int32_t& idxBlock, uint64_t& pos,
    std::vector<TileInfo>& blocks, const Rectangle<float>& queryBox,
    ReadTextLineFn&& readNextTextLine, OpenBlockFn&& openBlock,
    DecodeTextFn&& decodeText, ToWorldFn&& toWorld, Rec& out) {
    if (done || idxBlock < 0) {
        return -1;
    }
    std::string line;
    while (true) {
        auto& blk = blocks[idxBlock];
        if (pos >= blk.idx.ed || !readNextTextLine(line)) {
            if (++idxBlock >= static_cast<int32_t>(blocks.size())) {
                done = true;
                return -1;
            }
            openBlock(blocks[idxBlock]);
            continue;
        }
        pos += line.size() + 1;
        if (line.empty() || line[0] == '#') {
            continue;
        }
        if (!decodeText(line, out)) {
            return 0;
        }
        if (blk.contained) {
            return 1;
        }
        const auto xy = toWorld(out);
        if (queryBox.contains(xy.first, xy.second)) {
            return 1;
        }
    }
}

} // namespace

void TileOperator::validateCoordinateEncoding() const {
    if (storesIntegerCoordinates()) {
        if (!rawCoordinatesArePixels()) {
            error("%s: Integer coordinate storage requires mode & 0x2 so raw records are pixel coordinates", __func__);
        }
        if (formatInfo_.pixelResolution <= 0.0f) {
            error("%s: Integer coordinate storage requires positive pixelResolution", __func__);
        }
    } else if (rawCoordinatesArePixels()) {
        error("%s: Float coordinate storage requires mode & 0x2 == 0 so raw records are world coordinates", __func__);
    }
}

void TileOperator::requireNoFeatureIndex(const char* funcName) const {
    if (hasFeatureIndex()) {
        error("%s: single-molecule records with feature indices are not supported by this command", funcName);
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
    K_ = mode_ >> 16;
    mode_ &= 0xFFFF;
    coord_dim_ = (mode_ & 0x10) ? 3 : 2;
    if ((mode_ & 0x8) == 0) {assert(formatInfo_.tileSize > 0);}
    validateCoordinateEncoding();
    if ((mode_ & 0x20u) && coord_dim_ == 3) {assert(formatInfo_.pixelResolutionZ > 0.0f);}
    k_ = formatInfo_.parseKvec(kvec_);
    if (mode_ & 0x1) {
        assert(formatInfo_.recordSize > 0);
        size_t kBytes = k_ * (sizeof(int32_t) + sizeof(float));
        if (mode_ & 0x40u) {
            kBytes += sizeof(uint32_t);
        }
        size_t cBytes = (mode_ & 0x4) ? sizeof(int32_t) : sizeof(float);
        if (formatInfo_.recordSize != kBytes + coord_dim_ * cBytes) {
            error("%s: Record size %u inconsistent with k=%d and %dD dimensional coordinates", __func__, formatInfo_.recordSize, k_, coord_dim_);
        }
    }
    regular_labeled_raster_ = ((mode_ & 0x8) == 0) && (k_ > 0) && ((mode_ & 0x4) != 0 || formatInfo_.pixelResolution > 0.0f);

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
        for (size_t i = 0; i < blocks_all_.size(); ++i) {
            blocks_all_[i].row = blocks_all_[i].idx.row;
            blocks_all_[i].col = blocks_all_[i].idx.col;
        }
        for (size_t i = 0; i < blocks_.size(); ++i) {
            auto& b = blocks_[i];
            b.row = b.idx.row; b.col = b.idx.col;
            tile_lookup_[{b.row, b.col}] = i;
        }
    }
    notice("Read index with %lu tiles", blocks_all_.size());
}

void TileOperator::printIndex() const {
    if (formatInfo_.magic == PUNKST_INDEX_MAGIC) {
        // Print header info
        printf("##Flag: 0x%x\n", formatInfo_.mode);
        printf("##Tile size: %d\n", formatInfo_.tileSize);
        printf("##Pixel resolution: %.2f\n", formatInfo_.pixelResolution);
        if ((mode_ & 0x10) && (mode_ & 0x20u)) {
            printf("##Z pixel resolution: %.2f\n", formatInfo_.pixelResolutionZ);
        }
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
        printf("%" PRIu64 "\t%" PRIu64 "\t%d\t%d\t%u\t%d\t%d\t%d\t%d\n",
            b.idx.st, b.idx.ed, b.idx.row, b.idx.col, b.idx.n,
            b.idx.xmin, b.idx.xmax, b.idx.ymin, b.idx.ymax);
    }
}

void TileOperator::extractRegionGeoJSON(const std::string& outPrefix, const std::string& geojsonFile, int64_t scale, float qzmin, float qzmax) {
    if ((mode_ & 0x8) != 0) {
        error("%s: GeoJSON region query requires regular tile mode input (mode & 0x8 == 0)", __func__);
    }
    if ((mode_ & 0x1) == 0 && !canSeekTextInput()) {
        error("%s: GeoJSON region query requires a seekable text file. Input '%s' is a stream (stdin/gzip).",
            __func__, dataFile_.c_str());
    }
    PreparedRegionMask2D region;
    try {
        region = loadPreparedRegionGeoJSON(geojsonFile, formatInfo_.tileSize, scale);
    } catch (const std::exception& ex) {
        error("%s: %s", __func__, ex.what());
    }
    notice("Prepared region loaded from %s: bounding box [%.2f, %.2f, %.2f, %.2f], %lu union paths, %lu component bboxes",
        geojsonFile.c_str(), region.bbox_f.xmin, region.bbox_f.ymin, region.bbox_f.xmax, region.bbox_f.ymax,
        region.union_paths.size(), region.comp_bbox.size());

    extractRegionPrepared(outPrefix, region, qzmin, qzmax);
}

void TileOperator::extractRegionPrepared(const std::string& outPrefix, const PreparedRegionMask2D& region, float qzmin, float qzmax) {
    if ((mode_ & 0x8) != 0) {
        error("%s: Prepared polygon region query requires regular tile mode input (mode & 0x8 == 0)", __func__);
    }
    if (formatInfo_.tileSize <= 0) {
        error("%s: Invalid tileSize=%d; cannot write regular tiled output", __func__, formatInfo_.tileSize);
    }
    const bool hasZRange = !std::isnan(qzmin) || !std::isnan(qzmax);
    if (hasZRange) {
        if (std::isnan(qzmin) || std::isnan(qzmax)) {
            error("%s: z-range filtering requires both qzmin and qzmax", __func__);
        }
        if (coord_dim_ != 3) {
            error("%s: z-range filtering requires 3D input", __func__);
        }
        if (qzmin >= qzmax) {
            error("%s: Invalid z interval [%.3f, %.3f)", __func__, qzmin, qzmax);
        }
    }
    if (region.empty()) {
        warning("%s: Prepared region is empty", __func__);
        return;
    }

    std::vector<size_t> order;
    order.reserve(blocks_.size());
    for (size_t i = 0; i < blocks_.size(); ++i) {
        const auto& blk = blocks_[i];
        if (region.bbox_f.intersect(Rectangle<float>(
                static_cast<float>(blk.idx.xmin), static_cast<float>(blk.idx.ymin),
                static_cast<float>(blk.idx.xmax), static_cast<float>(blk.idx.ymax))) == 0) {
            continue;
        }
        order.push_back(i);
    }
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        const TileInfo& lhs = blocks_[a];
        const TileInfo& rhs = blocks_[b];
        if (lhs.row != rhs.row) return lhs.row < rhs.row;
        return lhs.col < rhs.col;
    });
    if (order.empty()) {
        warning("%s: No indexed tiles intersect the region bounding box", __func__);
        return;
    }
    notice("%s: %lu tiles intersect the region bounding box", __func__, order.size());

    std::vector<size_t> activeOrder;
    std::vector<RegionTileState> activeStates;
    activeOrder.reserve(order.size());
    activeStates.reserve(order.size());
    for (size_t idx : order) {
        const TileInfo& blk = blocks_[idx];
        const TileKey tile{blk.row, blk.col};
        const RegionTileState state = region.classifyTile(tile);
        if (state == RegionTileState::Outside) {
            continue;
        }
        activeOrder.push_back(idx);
        activeStates.push_back(state);
    }
    if (activeOrder.empty()) {
        warning("%s: No indexed tiles intersect the queried region", __func__);
        return;
    }

    const bool binaryInput = (mode_ & 0x1) != 0;
    const std::string outData = outPrefix + (binaryInput ? ".bin" : ".tsv");
    const std::string outIndex = outPrefix + ".index";
    std::ifstream copyIn(dataFile_, std::ios::binary);
    if (!copyIn.is_open()) {
        error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
    }

    IndexHeader outHeader = formatInfo_;
    if (outHeader.magic != PUNKST_INDEX_MAGIC) {
        outHeader = IndexHeader();
        outHeader.tileSize = formatInfo_.tileSize;
        outHeader.pixelResolution = formatInfo_.pixelResolution;
        outHeader.pixelResolutionZ = formatInfo_.pixelResolutionZ;
        outHeader.recordSize = binaryInput ? formatInfo_.recordSize : 0;
        if (!kvec_.empty()) {
            outHeader.packKvec(kvec_);
        } else if (k_ > 0) {
            outHeader.packKvec(std::vector<uint32_t>{static_cast<uint32_t>(k_)});
        }
    }
    outHeader.magic = PUNKST_INDEX_MAGIC;
    outHeader.mode = ((static_cast<uint32_t>(K_) & 0xFFFFu) << 16) | (mode_ & 0xFFFFu);
    outHeader.recordSize = binaryInput ? formatInfo_.recordSize : 0;
    outHeader.xmin = region.bbox_f.xmin;
    outHeader.xmax = region.bbox_f.xmax;
    outHeader.ymin = region.bbox_f.ymin;
    outHeader.ymax = region.bbox_f.ymax;

    Rectangle<float> outBox;
    std::ofstream out;
    std::ofstream idxOut;
    bool outputOpen = false;
    uint64_t currentOffset = 0;
    size_t nEntries = 0;
    size_t nProcessed = 0;

    auto ensureOutputReady = [&]() {
        if (outputOpen) {
            return;
        }
        out.open(outData, std::ios::binary);
        if (!out.is_open()) {
            error("%s: Error opening output data file: %s", __func__, outData.c_str());
        }
        idxOut.open(outIndex, std::ios::binary);
        if (!idxOut.is_open()) {
            error("%s: Error opening output index file: %s", __func__, outIndex.c_str());
        }
        if (!binaryInput) {
            if (headerLine_.empty()) {
                error("%s: TSV input is missing a parsed header line", __func__);
            }
            out.write(headerLine_.data(), static_cast<std::streamsize>(headerLine_.size()));
            if (!out) {
                error("%s: Error writing header to %s", __func__, outData.c_str());
            }
        }
        idxOut.write(reinterpret_cast<const char*>(&outHeader), sizeof(outHeader));
        if (!idxOut) {
            error("%s: Error writing placeholder header to %s", __func__, outIndex.c_str());
        }
        currentOffset = static_cast<uint64_t>(out.tellp());
        outputOpen = true;
    };

    auto processTile = [&](size_t activeIdx, std::ifstream& workerIn, RegionTileResult& result) {
        const size_t blkIdx = activeOrder[activeIdx];
        const TileInfo& blk = blocks_[blkIdx];
        const TileKey tile{blk.row, blk.col};
        const RegionTileState state = activeStates[activeIdx];
        result.state = state;
        if (state == RegionTileState::Inside && !hasZRange) {
            result.useCopyRange = true;
            result.copySt = blk.idx.st;
            result.copyEd = blk.idx.ed;
            result.nOut = blk.idx.n;
            return;
        }
        if (binaryInput) {
            const size_t recSize = formatInfo_.recordSize;
            if (recSize == 0) {
                error("%s: Binary input requires a positive record size", __func__);
            }
            std::vector<char> recBuf(recSize);
            workerIn.clear();
            workerIn.seekg(static_cast<std::streamoff>(blk.idx.st));
            if (!workerIn.good()) {
                error("%s: Error seeking input stream to %" PRIu64, __func__, blk.idx.st);
            }
            uint64_t pos = blk.idx.st;
            while (pos < blk.idx.ed) {
                workerIn.read(recBuf.data(), static_cast<std::streamsize>(recSize));
                if (workerIn.gcount() != static_cast<std::streamsize>(recSize)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
                pos += recSize;
                float x = 0.0f;
                float y = 0.0f;
                float z = 0.0f;
                if (hasZRange) {
                    decodeBinaryXYZ(recBuf.data(), x, y, z);
                } else {
                    decodeBinaryXY(recBuf.data(), x, y);
                }
                if (hasZRange && (z < qzmin || z >= qzmax)) {
                    continue;
                }
                if (state == RegionTileState::Partial && !region.containsPoint(x, y, &tile)) {
                    continue;
                }
                const size_t off = result.binaryData.size();
                result.binaryData.resize(off + recSize);
                std::memcpy(result.binaryData.data() + off, recBuf.data(), recSize);
                ++result.nOut;
            }
            return;
        }

        workerIn.clear();
        workerIn.seekg(static_cast<std::streamoff>(blk.idx.st));
        if (!workerIn.good()) {
            error("%s: Error seeking input stream to %" PRIu64, __func__, blk.idx.st);
        }
        std::string line;
        uint64_t pos = blk.idx.st;
        while (pos < blk.idx.ed) {
            if (!std::getline(workerIn, line)) {
                error("%s: Corrupted data or invalid index", __func__);
            }
            pos += static_cast<uint64_t>(line.size()) + 1;
            if (line.empty() || line[0] == '#') {
                continue;
            }
            if (hasZRange) {
                PixTopProbs3D<float> rec;
                if (!decodeTextRecord3D(line, rec, false)) {
                    error("%s: Invalid text record", __func__);
                }
                if (rec.z < qzmin || rec.z >= qzmax) {
                    continue;
                }
                if (state == RegionTileState::Partial && !region.containsPoint(rec.x, rec.y, &tile)) {
                    continue;
                }
                result.textData += line;
                result.textData.push_back('\n');
                ++result.nOut;
                continue;
            }

            PixTopProbs<float> rec;
            if (!decodeTextRecord2D(line, rec, false)) {
                error("%s: Invalid text record", __func__);
            }

            if (!region.containsPoint(rec.x, rec.y, &tile)) {
                continue;
            }
            result.textData += line;
            result.textData.push_back('\n');
            ++result.nOut;
        }
    };

    const bool useParallel = (threads_ > 1 && activeOrder.size() > 1);
    const size_t chunkTileCount = useParallel
        ? std::max<size_t>(static_cast<size_t>(threads_) * 4, 1)
        : static_cast<size_t>(1);
    std::unique_ptr<tbb::global_control> globalLimit;
    if (useParallel) {
        globalLimit = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
    }

    for (size_t chunkBegin = 0; chunkBegin < activeOrder.size(); chunkBegin += chunkTileCount) {
        const size_t chunkEnd = std::min(activeOrder.size(), chunkBegin + chunkTileCount);
        std::vector<RegionTileResult> chunkResults(chunkEnd - chunkBegin);

        if (useParallel && (chunkEnd - chunkBegin) > 1) {
            tbb::parallel_for(tbb::blocked_range<size_t>(chunkBegin, chunkEnd),
                [&](const tbb::blocked_range<size_t>& range) {
                    std::ifstream workerIn;
                    if (binaryInput) {
                        workerIn.open(dataFile_, std::ios::binary);
                    } else {
                        workerIn.open(dataFile_);
                    }
                    if (!workerIn.is_open()) {
                        error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
                    }
                    for (size_t ai = range.begin(); ai < range.end(); ++ai) {
                        processTile(ai, workerIn, chunkResults[ai - chunkBegin]);
                    }
                });
        } else {
            std::ifstream workerIn;
            if (binaryInput) {
                workerIn.open(dataFile_, std::ios::binary);
            } else {
                workerIn.open(dataFile_);
            }
            if (!workerIn.is_open()) {
                error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
            }
            for (size_t ai = chunkBegin; ai < chunkEnd; ++ai) {
                processTile(ai, workerIn, chunkResults[ai - chunkBegin]);
            }
        }

        for (size_t ai = chunkBegin; ai < chunkEnd; ++ai) {
            const size_t blkIdx = activeOrder[ai];
            const TileKey tile{blocks_[blkIdx].row, blocks_[blkIdx].col};
            const RegionTileResult& result = chunkResults[ai - chunkBegin];
            if (result.nOut == 0) {
                continue;
            }

            IndexEntryF outEntry(tile.row, tile.col);
            tile2bound(tile, outEntry.xmin, outEntry.xmax, outEntry.ymin, outEntry.ymax, formatInfo_.tileSize);
            ensureOutputReady();
            outEntry.st = currentOffset;
            outEntry.n = result.nOut;

            if (result.useCopyRange) {
                if (!copy_stream_range(copyIn, out, result.copySt, result.copyEd)) {
                    error("%s: Error copying input bytes from %" PRIu64 " to %" PRIu64,
                        __func__, result.copySt, result.copyEd);
                }
                currentOffset += result.copyEd - result.copySt;
            } else if (binaryInput) {
                if (!result.binaryData.empty()) {
                    out.write(result.binaryData.data(),
                        static_cast<std::streamsize>(result.binaryData.size()));
                    if (!out) {
                        error("%s: Error writing binary record buffer to %s", __func__, outData.c_str());
                    }
                    currentOffset += static_cast<uint64_t>(result.binaryData.size());
                }
            } else {
                if (!result.textData.empty()) {
                    out.write(result.textData.data(),
                        static_cast<std::streamsize>(result.textData.size()));
                    if (!out) {
                        error("%s: Error writing text record buffer to %s", __func__, outData.c_str());
                    }
                    currentOffset += static_cast<uint64_t>(result.textData.size());
                }
            }

            outEntry.ed = currentOffset;
            idxOut.write(reinterpret_cast<const char*>(&outEntry), sizeof(outEntry));
            if (!idxOut) {
                error("%s: Error writing index entry", __func__);
            }
            outBox.extendToInclude(Rectangle<float>(
                static_cast<float>(outEntry.xmin), static_cast<float>(outEntry.ymin),
                static_cast<float>(outEntry.xmax), static_cast<float>(outEntry.ymax)));
            ++nEntries;
            ++nProcessed;
            if (nProcessed % 10 == 0) {
                notice("%s: Processed %zu/%zu region tiles", __func__, nProcessed, activeOrder.size());
            }
        }
    }

    if (nEntries == 0) {
        warning("%s: No records found in the queried region", __func__);
        return;
    }

    outHeader.xmin = outBox.xmin;
    outHeader.xmax = outBox.xmax;
    outHeader.ymin = outBox.ymin;
    outHeader.ymax = outBox.ymax;
    idxOut.seekp(0);
    idxOut.write(reinterpret_cast<const char*>(&outHeader), sizeof(outHeader));
    if (!idxOut) {
        error("%s: Error finalizing output index header", __func__);
    }
    out.close();
    idxOut.close();
    notice("%s: Wrote %zu indexed tile(s) to %s (index: %s)", __func__, nEntries, outData.c_str(), outIndex.c_str());
}

void TileOperator::extractRegion(const std::string& outPrefix, float qxmin, float qxmax, float qymin, float qymax,
    float qzmin, float qzmax) {
    if (qxmin >= qxmax || qymin >= qymax) {
        error("%s: Invalid rectangle [%.3f, %.3f) x [%.3f, %.3f)", __func__, qxmin, qxmax, qymin, qymax);
    }
    const bool hasZRange = !std::isnan(qzmin) || !std::isnan(qzmax);
    if (hasZRange) {
        if (std::isnan(qzmin) || std::isnan(qzmax)) {
            error("%s: z-range filtering requires both qzmin and qzmax", __func__);
        }
        if (coord_dim_ != 3) {
            error("%s: z-range filtering requires 3D input", __func__);
        }
        if (qzmin >= qzmax) {
            error("%s: Invalid z interval [%.3f, %.3f)", __func__, qzmin, qzmax);
        }
    }
    PreparedRegionMask2D region;
    try {
        region = prepareRegionFromRectangle(Rectangle<float>(qxmin, qymin, qxmax, qymax),
            formatInfo_.tileSize);
    } catch (const std::exception& ex) {
        error("%s: %s", __func__, ex.what());
    }
    extractRegionPrepared(outPrefix, region, qzmin, qzmax);
}

void TileOperator::openDataStream() {
    if (mode_ & 0x1) {
        if (dataStream_.is_open()) {
            dataStream_.close();
        }
        dataStream_.open(dataFile_, std::ios::binary);
        if (!dataStream_.is_open()) {
            error("Error opening data file: %s", dataFile_.c_str());
        }
        return;
    }

    // Streaming text sources (stdin/gzip) are opened once and consumed sequentially.
    if (isStreamingTextInput()) {
        if (textStreamOpen_) {
            return;
        }
        if (isTextStdinInput()) {
            if (!std::cin.good()) {
                std::cin.clear();
            }
            textStreamOpen_ = true;
            return;
        }
        if (isTextGzipInput()) {
            gzDataStream_.reset(gzopen(dataFile_.c_str(), "rb"));
            if (!gzDataStream_) {
                error("Error opening gzipped input file: %s", dataFile_.c_str());
            }
            textStreamOpen_ = true;
            return;
        }
    }

    if (dataStream_.is_open()) {
        dataStream_.close();
    }
    dataStream_.open(dataFile_);
    if (!dataStream_.is_open()) {
        error("Error opening data file: %s", dataFile_.c_str());
    }
    std::streampos pos = dataStream_.tellg();
    std::string line;
    while (std::getline(dataStream_, line)) {
        if (!(line.empty() || line[0] == '#')) {
            break;
        }
        pos = dataStream_.tellg();
    }
    dataStream_.clear();
    dataStream_.seekg(pos);
    hasPendingTextLine_ = false;
    textStreamOpen_ = true;
}

void TileOperator::resetReader() {
    if ((mode_ & 0x1) == 0 && isTextStdinInput()) {
        error("%s: Cannot reset reader for stdin stream input", __func__);
    }

    if ((mode_ & 0x1) == 0 && isTextGzipInput()) {
        closeTextStream();
        hasPendingTextLine_ = false;
        openDataStream();
    } else {
        if (dataStream_.is_open()) {
            dataStream_.clear();
            dataStream_.seekg(0);
        } else {
            openDataStream();
        }
        hasPendingTextLine_ = false;
    }
    done_ = false;
    idx_block_ = 0;
    pos_ = 0;
    if (bounded_ && !blocks_.empty()) {
        openBlock(blocks_[0]);
    }
}

void TileOperator::closeTextStream() {
    gzDataStream_.reset();
    if (dataStream_.is_open()) {
        dataStream_.close();
    }
    textStreamOpen_ = false;
}

bool TileOperator::readNextTextLine(std::string& line) {
    if (hasPendingTextLine_) {
        line = std::move(pendingTextLine_);
        pendingTextLine_.clear();
        hasPendingTextLine_ = false;
        return true;
    }

    if (isTextStdinInput()) {
        return static_cast<bool>(std::getline(std::cin, line));
    }

    if (isTextGzipInput()) {
        if (!gzDataStream_) {
            return false;
        }
        line.clear();
        constexpr int32_t BUF_SZ = 1 << 16;
        char buf[BUF_SZ];
        while (true) {
            char* ret = gzgets(gzDataStream_.get(), buf, BUF_SZ);
            if (ret == nullptr) {
                return !line.empty();
            }
            line.append(ret);
            if (!line.empty() && line.back() == '\n') {
                break;
            }
        }
        if (!line.empty() && line.back() == '\n') {
            line.pop_back();
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        return true;
    }

    return static_cast<bool>(std::getline(dataStream_, line));
}

int32_t TileOperator::query(float qxmin,float qxmax,float qymin,float qymax) {
    if ((mode_ & 0x1) == 0 && !canSeekTextInput()) {
        error("%s: --filter/query requires a seekable text file. Input '%s' is a stream (stdin/gzip).",
            __func__, dataFile_.c_str());
    }
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

void TileOperator::sampleTilesToDebug(int32_t ntiles) {
    // Pick ntiles tiles
    blocks_.clear();
    std::mt19937 rng(42);
    std::uniform_int_distribution<size_t> uni(0, blocks_all_.size() - 1);
    std::unordered_set<size_t> selected;
    while (static_cast<int32_t>(selected.size()) < ntiles) {
        size_t idx = uni(rng);
        selected.insert(idx);
    }
    for (auto idx : selected) {
        blocks_.push_back(blocks_all_[idx]);
    }
    idx_block_ = 0;
    if ((mode_ & 0x8) == 0) { // regular grid
        tile_lookup_.clear();
        for (size_t i = 0; i < blocks_.size(); ++i) {
            auto& b = blocks_[i];
            b.row = b.idx.row; b.col = b.idx.col;
            tile_lookup_[{b.row, b.col}] = i;
        }
    }
}

int32_t TileOperator::floorDivInt32(int32_t value, int32_t divisor) {
    if (divisor <= 0) {
        error("%s: divisor must be positive", __func__);
    }
    int32_t q = value / divisor;
    const int32_t r = value % divisor;
    if (r < 0) {
        --q;
    }
    return q;
}

int32_t TileOperator::ceilDivInt32(int32_t value, int32_t divisor) {
    return -floorDivInt32(-value, divisor);
}

int32_t TileOperator::mapPixelToRasterFloor(int32_t value) const {
    if (!hasRasterResolutionOverride_) {
        return value;
    }
    return floorDivInt32(value, rasterRatioXY_);
}

int32_t TileOperator::mapPixelToRasterCeil(int32_t value) const {
    if (!hasRasterResolutionOverride_) {
        return value;
    }
    return ceilDivInt32(value, rasterRatioXY_);
}

void TileOperator::accumulateRasterTopProbs(RasterTopProbAccum& accum, const TopProbs& rec) const {
    const size_t nSets = kvec_.empty() ? size_t(1) : kvec_.size();
    if (accum.size() != nSets) {
        accum.assign(nSets, {});
    }
    size_t off = 0;
    for (size_t s = 0; s < nSets; ++s) {
        const uint32_t keepK = kvec_.empty() ? static_cast<uint32_t>(k_) : kvec_[s];
        for (uint32_t i = 0; i < keepK && (off + i) < rec.ks.size() && (off + i) < rec.ps.size(); ++i) {
            const int32_t k = rec.ks[off + i];
            const float p = rec.ps[off + i];
            if (k < 0 || p <= 0.0f) {
                continue;
            }
            accum[s][k] += static_cast<double>(p);
        }
        off += keepK;
    }
}

TopProbs TileOperator::finalizeRasterTopProbs(const RasterTopProbAccum& accum) const {
    TopProbs out;
    const size_t nSets = kvec_.empty() ? size_t(1) : kvec_.size();
    for (size_t s = 0; s < nSets; ++s) {
        const uint32_t keepK = kvec_.empty() ? static_cast<uint32_t>(k_) : kvec_[s];
        std::vector<std::pair<int32_t, double>> items;
        items.reserve(accum[s].size());
        for (const auto& kv : accum[s]) {
            if (kv.second > 0.0) {
                items.emplace_back(kv.first, kv.second);
            }
        }
        std::sort(items.begin(), items.end(),
            [](const auto& a, const auto& b) {
                if (a.second == b.second) {
                    return a.first < b.first;
                }
                return a.second > b.second;
            });
        const size_t keep = std::min<size_t>(keepK, items.size());
        for (size_t i = 0; i < keep; ++i) {
            out.ks.push_back(items[i].first);
            out.ps.push_back(static_cast<float>(items[i].second));
        }
        for (size_t i = keep; i < keepK; ++i) {
            out.ks.push_back(-1);
            out.ps.push_back(0.0f);
        }
    }
    return out;
}

void TileOperator::setRasterPixelResolution(float resXY) {
    if (!(resXY > 0.0f)) {
        error("%s: raster pixel resolution must be positive", __func__);
    }
    const float dataRes = getPixelResolution();
    if (!(dataRes > 0.0f)) {
        error("%s: input pixelResolution must be positive", __func__);
    }
    if (resXY <= dataRes) {
        error("%s: raster pixel resolution %.8g must be coarser than input pixelResolution %.8g",
            __func__, resXY, dataRes);
    }
    const double ratio = static_cast<double>(resXY) / static_cast<double>(dataRes);
    const int64_t rounded = static_cast<int64_t>(std::llround(ratio));
    const double tol = 1e-6 * std::max(1.0, std::abs(ratio));
    if (rounded < 2 || std::abs(ratio - static_cast<double>(rounded)) > tol) {
        error("%s: raster pixel resolution ratio %.8g must be an integer greater than 1",
            __func__, ratio);
    }
    int32_t tileSpanPixels = formatInfo_.tileSize;
    if (rawCoordinatesArePixels()) {
        tileSpanPixels = coord2pix(tileSpanPixels);
    }
    if (tileSpanPixels <= 0 || (tileSpanPixels % static_cast<int32_t>(rounded)) != 0) {
        error("%s: tile size must align with requested raster resolution (tile span %d source pixels, ratio %lld)",
            __func__, tileSpanPixels, static_cast<long long>(rounded));
    }
    hasRasterResolutionOverride_ = true;
    rasterPixelResolution_ = resXY;
    rasterRatioXY_ = static_cast<int32_t>(rounded);
}

int32_t TileOperator::loadTileToMap(const TileKey& key,
    std::map<std::pair<int32_t, int32_t>, TopProbs>& pixelMap,
    const std::vector<Rectangle<float>>* rects,
    std::ifstream* dataStream) const {
    if (coord_dim_ == 3) {
        error("%s: 3D data requires loadTileToMap3D", __func__);
    }
    assert((mode_ & 0x8) == 0);
    if ((mode_ & 0x4) == 0) {
        assert((mode_ & 0x2) == 0 && formatInfo_.pixelResolution > 0);}
    pixelMap.clear();
    auto lookup = tile_lookup_.find(key);
    if (lookup == tile_lookup_.end()) {
        notice("%s: Tile (%d, %d) not found in index", __func__, key.row, key.col);
        return 0;
    }

    std::ifstream localStream;
    std::ifstream* stream = dataStream;
    if (stream == nullptr) {
        stream = &localStream;
    }
    if (!stream->is_open()) {
        if (mode_ & 0x1) {
            stream->open(dataFile_, std::ios::binary);
        } else {
            stream->open(dataFile_);
        }
    }
    if (!stream->is_open()) {
        error("Error opening data file: %s", dataFile_.c_str());
    }

    size_t idx = lookup->second;
    const TileInfo& blk = blocks_[idx];
    stream->clear();
    stream->seekg(blk.idx.st);
    uint64_t pos = blk.idx.st;
    TopProbs rec;
    int32_t recX = 0;
    int32_t recY = 0;
    if (!hasRasterResolutionOverride_) {
        while (readNextRecord2DAsPixel(*stream, pos, blk.idx.ed, recX, recY, rec)) {
            if (rects != nullptr && !rects->empty()) {
                const float res = formatInfo_.pixelResolution > 0.0f ? formatInfo_.pixelResolution : 1.0f;
                const float x = static_cast<float>(recX) * res;
                const float y = static_cast<float>(recY) * res;
                bool keep = false;
                for (const auto& rect : *rects) {
                    if (rect.contains(x, y)) {
                        keep = true;
                        break;
                    }
                }
                if (!keep) {
                    continue;
                }
            }
            pixelMap[{recX, recY}] = std::move(rec);
        }
        return static_cast<int32_t>(pixelMap.size());
    }

    std::map<std::pair<int32_t, int32_t>, RasterTopProbAccum> accumMap;
    while (readNextRecord2DAsPixel(*stream, pos, blk.idx.ed, recX, recY, rec)) {
        if (rects != nullptr && !rects->empty()) {
            const float res = formatInfo_.pixelResolution > 0.0f ? formatInfo_.pixelResolution : 1.0f;
            const float x = static_cast<float>(recX) * res;
            const float y = static_cast<float>(recY) * res;
            bool keep = false;
            for (const auto& rect : *rects) {
                if (rect.contains(x, y)) {
                    keep = true;
                    break;
                }
            }
            if (!keep) {
                continue;
            }
        }
        const int32_t rasterX = mapPixelToRasterFloor(recX);
        const int32_t rasterY = mapPixelToRasterFloor(recY);
        RasterTopProbAccum& accum = accumMap[{rasterX, rasterY}];
        if (accum.empty()) {
            accum.resize(kvec_.empty() ? size_t(1) : kvec_.size());
        }
        accumulateRasterTopProbs(accum, rec);
    }
    for (auto& kv : accumMap) {
        pixelMap[kv.first] = finalizeRasterTopProbs(kv.second);
    }
    return static_cast<int32_t>(pixelMap.size());
}

int32_t TileOperator::loadTileToMap3D(const TileKey& key,
    std::map<PixelKey3, TopProbs>& pixelMap,
    std::ifstream* dataStream) const {
    if (coord_dim_ != 3) {
        error("%s: 3D data required, but coord_dim_=%u", __func__, coord_dim_);
    }
    assert((mode_ & 0x8) == 0);
    if ((mode_ & 0x4) == 0) {
        assert((mode_ & 0x2) == 0 && formatInfo_.pixelResolution > 0);
    }
    pixelMap.clear();
    auto lookup = tile_lookup_.find(key);
    if (lookup == tile_lookup_.end()) return 0;

    std::ifstream localStream;
    std::ifstream* stream = dataStream;
    if (stream == nullptr) {
        stream = &localStream;
    }
    if (!stream->is_open()) {
        if (mode_ & 0x1) {
            stream->open(dataFile_, std::ios::binary);
        } else {
            stream->open(dataFile_);
        }
    }
    if (!stream->is_open()) {
        error("Error opening data file: %s", dataFile_.c_str());
    }

    size_t idx = lookup->second;
    const TileInfo& blk = blocks_[idx];
    stream->clear();
    stream->seekg(blk.idx.st);
    uint64_t pos = blk.idx.st;
    TopProbs rec;
    int32_t recX = 0;
    int32_t recY = 0;
    int32_t recZ = 0;
    while (readNextRecord3DAsPixel(*stream, pos, blk.idx.ed, recX, recY, recZ, rec)) {
        pixelMap[std::make_tuple(recX, recY, recZ)] = std::move(rec);
    }
    return static_cast<int32_t>(pixelMap.size());
}

void TileOperator::dumpTSV(const std::string& outPrefix, int32_t probDigits, int32_t coordDigits,
    const std::string& featureDictFile) {
    if (hasFeatureIndex()) {
        dumpTSVSingleMolecule(outPrefix, probDigits, coordDigits, featureDictFile);
        return;
    }
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
    if (coord_dim_ == 3) {
        headerStr += "\tz";
    }
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
        if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
            error("Error writing header to index output file");
        }
    }

    bool isInt32 = (mode_ & 0x4);
    float res = formatInfo_.pixelResolution;
    bool applyRes = (mode_ & 0x2) && (res > 0 && res != 1.0f);
    if (isInt32 && (!applyRes || res == 1.0f)) {
        coordDigits = 0;
    }

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
            float x, y, z = 0.0f;
            std::vector<int32_t> ks;
            std::vector<float> ps;

            if (coord_dim_ == 3) {
                PixTopProbs3D<float> temp;
                if (!readBinaryRecord3D(dataStream_, temp, false)) break;
                x = temp.x;
                y = temp.y;
                z = temp.z;
                ks = std::move(temp.ks);
                ps = std::move(temp.ps);
            } else {
                PixTopProbs<float> temp;
                if (!readBinaryRecord2D(dataStream_, temp, false)) break;
                x = temp.x;
                y = temp.y;
                ks = std::move(temp.ks);
                ps = std::move(temp.ps);
            }

            if (checkBound && !queryBox_.contains(x, y)) {
                continue;
            }

            if (coord_dim_ == 3) {
                if (fprintf(fp, "%.*f\t%.*f\t%.*f", coordDigits, x, coordDigits, y, coordDigits, z) < 0)
                    error("%s: Write error", __func__);
            } else if (fprintf(fp, "%.*f\t%.*f", coordDigits, x, coordDigits, y) < 0) {
                error("%s: Write error", __func__);
            }
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

void TileOperator::openBlock(TileInfo& blk) {
    dataStream_.clear();  // clear EOF flags
    dataStream_.seekg(blk.idx.st);
    pos_ = blk.idx.st;
}

int32_t TileOperator::next(PixTopProbs<float>& out, bool rawCoord) {
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out, rawCoord);
    }
    if (mode_ & 0x1) { // Binary mode
        if (!readBinaryRecord2D(dataStream_, out, rawCoord)) {
            done_ = true;
            return -1;
        }
        return 1;
    }
    return nextTextRecordImpl(done_,
        [&](std::string& line) { return readNextTextLine(line); },
        [&](const std::string& line, PixTopProbs<float>& rec) {
            return decodeTextRecord2D(line, rec, rawCoord);
        }, out);
}

int32_t TileOperator::next(PixTopProbs<int32_t>& out) {
    if (mode_ & 0x1) {assert(mode_ & 0x4);}
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out);
    }
    if (mode_ & 0x1) { // Binary mode
        if (readBinaryRecord2DInt(dataStream_, out)) {
            return 1;
        }
        if (dataStream_.eof()) {
            done_ = true;
            return -1;
        }
        error("%s: Corrupted data", __func__);
    }
    return nextTextRecordImpl(done_,
        [&](std::string& line) { return readNextTextLine(line); },
        [&](const std::string& line, PixTopProbs<int32_t>& rec) {
            return decodeTextRecord2DInt(line, rec);
        }, out);
}

int32_t TileOperator::next(PixTopProbs3D<float>& out, bool rawCoord) {
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out, rawCoord);
    }
    if (mode_ & 0x1) { // Binary mode
        if (!readBinaryRecord3D(dataStream_, out, rawCoord)) {
            done_ = true;
            return -1;
        }
        return 1;
    }
    return nextTextRecordImpl(done_,
        [&](std::string& line) { return readNextTextLine(line); },
        [&](const std::string& line, PixTopProbs3D<float>& rec) {
            return decodeTextRecord3D(line, rec, rawCoord);
        }, out);
}

int32_t TileOperator::next(PixTopProbs3D<int32_t>& out) {
    if (mode_ & 0x1) {assert(mode_ & 0x4);}
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out);
    }
    if (mode_ & 0x1) { // Binary mode
        if (readBinaryRecord3DInt(dataStream_, out)) {
            return 1;
        }
        if (dataStream_.eof()) {
            done_ = true;
            return -1;
        }
        error("%s: Corrupted data", __func__);
    }
    return nextTextRecordImpl(done_,
        [&](std::string& line) { return readNextTextLine(line); },
        [&](const std::string& line, PixTopProbs3D<int32_t>& rec) {
            return decodeTextRecord3DInt(line, rec);
        }, out);
}

int32_t TileOperator::nextBounded(PixTopProbs<float>& out, bool rawCoord) {
    if (mode_ & 0x1) { // Binary mode
        return nextBoundedBinaryRecordImpl(done_, idx_block_, pos_, blocks_,
            formatInfo_.recordSize, queryBox_,
            [&](TileInfo& blk) { openBlock(blk); },
            [&](PixTopProbs<float>& rec) {
                if (!readBinaryRecord2D(dataStream_, rec, rawCoord)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
            },
            [&](const PixTopProbs<float>& rec) {
                float x = rec.x;
                float y = rec.y;
                if (rawCoord && (mode_ & 0x2)) {
                    x *= formatInfo_.pixelResolution;
                    y *= formatInfo_.pixelResolution;
                }
                return std::make_pair(x, y);
            }, out);
    }

    return nextBoundedTextRecordImpl(done_, idx_block_, pos_, blocks_, queryBox_,
        [&](std::string& line) { return readNextTextLine(line); },
        [&](TileInfo& blk) { openBlock(blk); },
        [&](const std::string& line, PixTopProbs<float>& rec) {
            return decodeTextRecord2D(line, rec, rawCoord);
        },
        [&](const PixTopProbs<float>& rec) {
            float x = rec.x;
            float y = rec.y;
            if (rawCoord && (mode_ & 0x2)) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
            }
            return std::make_pair(x, y);
        }, out);
}

int32_t TileOperator::nextBounded(PixTopProbs<int32_t>& out) {
    if (mode_ & 0x1) {assert(mode_ & 0x4);}
    if (mode_ & 0x1) { // Binary mode
        return nextBoundedBinaryRecordImpl(done_, idx_block_, pos_, blocks_,
            formatInfo_.recordSize, queryBox_,
            [&](TileInfo& blk) { openBlock(blk); },
            [&](PixTopProbs<int32_t>& rec) {
                if (!readBinaryRecord2DInt(dataStream_, rec)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
            },
            [&](const PixTopProbs<int32_t>& rec) {
                float x = static_cast<float>(rec.x);
                float y = static_cast<float>(rec.y);
                if (mode_ & 0x2) {
                    x *= formatInfo_.pixelResolution;
                    y *= formatInfo_.pixelResolution;
                }
                return std::make_pair(x, y);
            }, out);
    }

    return nextBoundedTextRecordImpl(done_, idx_block_, pos_, blocks_, queryBox_,
        [&](std::string& line) { return readNextTextLine(line); },
        [&](TileInfo& blk) { openBlock(blk); },
        [&](const std::string& line, PixTopProbs<int32_t>& rec) {
            return decodeTextRecord2DInt(line, rec);
        },
        [&](const PixTopProbs<int32_t>& rec) {
            float x = static_cast<float>(rec.x);
            float y = static_cast<float>(rec.y);
            if (mode_ & 0x2) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
            }
            return std::make_pair(x, y);
        }, out);
}

int32_t TileOperator::nextBounded(PixTopProbs3D<float>& out, bool rawCoord) {
    if (mode_ & 0x1) { // Binary mode
        return nextBoundedBinaryRecordImpl(done_, idx_block_, pos_, blocks_,
            formatInfo_.recordSize, queryBox_,
            [&](TileInfo& blk) { openBlock(blk); },
            [&](PixTopProbs3D<float>& rec) {
                if (!readBinaryRecord3D(dataStream_, rec, rawCoord)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
            },
            [&](const PixTopProbs3D<float>& rec) {
                float x = rec.x;
                float y = rec.y;
                if (rawCoord && (mode_ & 0x2)) {
                    x *= formatInfo_.pixelResolution;
                    y *= formatInfo_.pixelResolution;
                }
                return std::make_pair(x, y);
            }, out);
    }

    return nextBoundedTextRecordImpl(done_, idx_block_, pos_, blocks_, queryBox_,
        [&](std::string& line) { return readNextTextLine(line); },
        [&](TileInfo& blk) { openBlock(blk); },
        [&](const std::string& line, PixTopProbs3D<float>& rec) {
            return decodeTextRecord3D(line, rec, rawCoord);
        },
        [&](const PixTopProbs3D<float>& rec) {
            float x = rec.x;
            float y = rec.y;
            if (rawCoord && (mode_ & 0x2)) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
            }
            return std::make_pair(x, y);
        }, out);
}

int32_t TileOperator::nextBounded(PixTopProbs3D<int32_t>& out) {
    if (mode_ & 0x1) {assert(mode_ & 0x4);}
    if (mode_ & 0x1) { // Binary mode
        return nextBoundedBinaryRecordImpl(done_, idx_block_, pos_, blocks_,
            formatInfo_.recordSize, queryBox_,
            [&](TileInfo& blk) { openBlock(blk); },
            [&](PixTopProbs3D<int32_t>& rec) {
                if (!readBinaryRecord3DInt(dataStream_, rec)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
            },
            [&](const PixTopProbs3D<int32_t>& rec) {
                float x = static_cast<float>(rec.x);
                float y = static_cast<float>(rec.y);
                if (mode_ & 0x2) {
                    x *= formatInfo_.pixelResolution;
                    y *= formatInfo_.pixelResolution;
                }
                return std::make_pair(x, y);
            }, out);
    }

    return nextBoundedTextRecordImpl(done_, idx_block_, pos_, blocks_, queryBox_,
        [&](std::string& line) { return readNextTextLine(line); },
        [&](TileInfo& blk) { openBlock(blk); },
        [&](const std::string& line, PixTopProbs3D<int32_t>& rec) {
            return decodeTextRecord3DInt(line, rec);
        },
        [&](const PixTopProbs3D<int32_t>& rec) {
            float x = static_cast<float>(rec.x);
            float y = static_cast<float>(rec.y);
            if (mode_ & 0x2) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
            }
            return std::make_pair(x, y);
        }, out);
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

void TileOperator::parseHeaderLine() {
    std::ifstream ss;
    const bool streamingInput = isStreamingTextInput();
    if (streamingInput) {
        openDataStream();
    } else {
        ss.open(dataFile_);
        if (!ss.is_open()) {
            error("Error opening data file: %s", dataFile_.c_str());
        }
    }
    std::string line;
    std::string parsedHeaderLine;
    while ((streamingInput && readNextTextLine(line)) || (!streamingInput && std::getline(ss, line))) {
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        if (line.empty() || line.substr(0, 2) == "##") {
            continue;
        }
        if (line[0] == '#') {
            parsedHeaderLine = line;
        } else {
            if (streamingInput) {
                hasPendingTextLine_ = true;
                pendingTextLine_ = line;
            }
            break;
        }
    }
    if (parsedHeaderLine.empty()) {
        error("%s: TSV input requires a header line starting with '#': %s", __func__, dataFile_.c_str());
    }

    headerLine_ = parsedHeaderLine;
    if (headerLine_.back() != '\n') {
        headerLine_.push_back('\n');
    }

    line = parsedHeaderLine.substr(1); // skip initial '#'
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    std::unordered_map<std::string, uint32_t> header;
    for (uint32_t i = 0; i < tokens.size(); ++i) {
        std::string key = tokens[i];
        if (key == "X" || key == "Y" || key == "Z") {
            std::transform(key.begin(), key.end(), key.begin(), ::tolower);
        } else if (key.size() > 1 && (key[0] == 'k' || key[0] == 'p')) {
            key[0] = static_cast<char>(std::toupper(static_cast<unsigned char>(key[0])));
        }
        header[key] = i;
    }
    if (header.find("x") == header.end() || header.find("y") == header.end()) {
        error("%s: tsv input must has x and y columns for coordinates\n%s", __func__, parsedHeaderLine.c_str());
    }

    const bool indexed_tsv = (formatInfo_.magic == PUNKST_INDEX_MAGIC) && ((mode_ & 0x1) == 0);
    icol_x_ = header["x"];
    icol_y_ = header["y"];
    has_z_ = (header.find("z") != header.end());
    coord_dim_ = has_z_ ? 3 : 2;
    if (has_z_) {icol_z_ = header["z"];}

    icol_ks_.clear();
    icol_ps_.clear();
    for (int32_t k = 1; ; ++k) {
        const std::string kcol = "K" + std::to_string(k);
        const std::string pcol = "P" + std::to_string(k);
        bool has_k = (header.find(kcol) != header.end());
        bool has_p = (header.find(pcol) != header.end());
        if (!has_k && !has_p) {
            break;
        }
        if (!has_k || !has_p) {
            error("%s: Header must include both %s and %s", __func__, kcol.c_str(), pcol.c_str());
        }
        icol_ks_.push_back(header[kcol]);
        icol_ps_.push_back(header[pcol]);
    }

    if (indexed_tsv) {
        if (static_cast<int32_t>(icol_ks_.size()) != k_) {
            error("%s: Header has %lu K/P pairs, but index requires %d", __func__, icol_ks_.size(), k_);
        }
    } else {
        if (icol_ks_.empty()) {
            warning("%s: No K/P columns found in header", __func__);
        }
        k_ = static_cast<int32_t>(icol_ks_.size());
        kvec_.clear();
        if (k_ > 0) {
            kvec_.push_back(k_);
        }
    }

    icol_max_ = std::max(icol_x_, icol_y_);
    if (has_z_) {
        icol_max_ = std::max(icol_max_, icol_z_);
    }
    for (size_t i = 0; i < icol_ks_.size() && i < icol_ps_.size(); ++i) {
        icol_max_ = std::max(icol_max_, std::max(icol_ks_[i], icol_ps_[i]));
    }

    if (indexed_tsv) {
        validateCoordinateEncoding();
    }
}
