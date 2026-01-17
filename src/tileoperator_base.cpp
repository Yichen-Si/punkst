#include "tileoperator.hpp"
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>

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
    if (mode_ & 0x1) {assert(formatInfo_.recordSize > 0);}
    k_ = formatInfo_.parseKvec(kvec_);
    coord_dim_ = 2;
    if (mode_ & 0x1) {
        size_t kBytes = k_ * (sizeof(int32_t) + sizeof(float));
        size_t cBytes = (mode_ & 0x4) ? sizeof(int32_t) : sizeof(float);
        if (formatInfo_.recordSize < kBytes + 2 * cBytes ||
            (formatInfo_.recordSize - kBytes) % cBytes != 0) {
            error("%s: Record size %u inconsistent with k=%d and coordinate type", __func__, formatInfo_.recordSize, k_);
        }
        coord_dim_ = (formatInfo_.recordSize - kBytes) / cBytes;
        if (coord_dim_ != 2 && coord_dim_ != 3) {
            error("%s: Unsupported coordinate dimension %d", __func__, coord_dim_);
        }
    }
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
        printf("%" PRIu64 "\t%" PRIu64 "\t%d\t%d\t%u\t%d\t%d\t%d\t%d\n",
            b.idx.st, b.idx.ed, b.idx.row, b.idx.col, b.idx.n,
            b.idx.xmin, b.idx.xmax, b.idx.ymin, b.idx.ymax);
    }
}

int32_t TileOperator::query(float qxmin,float qxmax,float qymin,float qymax) {
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

int32_t TileOperator::loadTileToMap(const TileKey& key,
    std::map<std::pair<int32_t, int32_t>, TopProbs>& pixelMap) const {
    if (coord_dim_ == 3) {
        error("%s: 3D data requires loadTileToMap3D", __func__);
    }
    assert((mode_ & 0x8) == 0);
    if ((mode_ & 0x4) == 0) {
        assert((mode_ & 0x2) == 0 && formatInfo_.pixelResolution > 0);}
    pixelMap.clear();
    auto lookup = tile_lookup_.find(key);
    if (lookup == tile_lookup_.end()) return 0;

    std::ifstream dataStream;
    if (mode_ & 0x1) {
        dataStream.open(dataFile_, std::ios::binary);
    } else {
        dataStream.open(dataFile_);
    }
    if (!dataStream.is_open()) {
        error("Error opening data file: %s", dataFile_.c_str());
    }

    size_t idx = lookup->second;
    const TileInfo& blk = blocks_[idx];
    dataStream.clear();
    dataStream.seekg(blk.idx.st);
    uint64_t pos = blk.idx.st;
    float res = formatInfo_.pixelResolution;
    TopProbs rec;
    while (pos < blk.idx.ed) {
        bool success = false;
        int32_t recX = 0;
        int32_t recY = 0;
        if (mode_ & 0x4) { // int32
            if (mode_ & 0x1) { // Binary
                PixTopProbs<int32_t> temp;
                if (temp.read(dataStream, k_)) {
                    recX = temp.x;
                    recY = temp.y;
                    rec.ks = std::move(temp.ks);
                    rec.ps = std::move(temp.ps);
                    pos += formatInfo_.recordSize;
                    success = true;
                }
            } else {
                std::string line;
                if (std::getline(dataStream, line)) {
                    pos += line.size() + 1;
                    PixTopProbs<int32_t> temp;
                    success = parseLine(line, temp);
                    if (success) {
                        recX = temp.x;
                        recY = temp.y;
                        rec.ks = std::move(temp.ks);
                        rec.ps = std::move(temp.ps);
                    }
                }
            }
        } else { // float, scale then round to int
            PixTopProbs<float> temp;
            if (mode_ & 0x1) { // Binary
                if (coord_dim_ == 3) {
                    PixTopProbs3D<float> temp3;
                    if (temp3.read(dataStream, k_)) {
                        temp.x = temp3.x;
                        temp.y = temp3.y;
                        temp.ks = std::move(temp3.ks);
                        temp.ps = std::move(temp3.ps);
                        pos += formatInfo_.recordSize;
                        success = true;
                    }
                } else if (temp.read(dataStream, k_)) {
                    pos += formatInfo_.recordSize;
                    success = true;
                }
            } else {
                std::string line;
                if (std::getline(dataStream, line)) {
                    pos += line.size() + 1;
                    success = parseLine(line, temp);
                }
            }
            if (success) {
                recX = static_cast<int32_t>(std::floor(temp.x / res));
                recY = static_cast<int32_t>(std::floor(temp.y / res));
                rec.ks = std::move(temp.ks);
                rec.ps = std::move(temp.ps);
            }
        }
        if (success) {
            pixelMap[{recX, recY}] = std::move(rec);
        } else if (!dataStream.eof()) {
            error("%s: Corrupted data", __func__);
        }
    }
    return static_cast<int32_t>(pixelMap.size());
}

int32_t TileOperator::loadTileToMap3D(const TileKey& key,
    std::map<PixelKey3, TopProbs>& pixelMap) const {
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

    std::ifstream dataStream;
    if (mode_ & 0x1) {
        dataStream.open(dataFile_, std::ios::binary);
    } else {
        dataStream.open(dataFile_);
    }
    if (!dataStream.is_open()) {
        error("Error opening data file: %s", dataFile_.c_str());
    }

    size_t idx = lookup->second;
    const TileInfo& blk = blocks_[idx];
    dataStream.clear();
    dataStream.seekg(blk.idx.st);
    uint64_t pos = blk.idx.st;
    float res = formatInfo_.pixelResolution;
    TopProbs rec;
    while (pos < blk.idx.ed) {
        bool success = false;
        int32_t recX = 0;
        int32_t recY = 0;
        int32_t recZ = 0;
        if (mode_ & 0x4) { // int32
            if (mode_ & 0x1) { // Binary
                PixTopProbs3D<int32_t> temp;
                if (temp.read(dataStream, k_)) {
                    recX = temp.x;
                    recY = temp.y;
                    recZ = temp.z;
                    rec.ks = std::move(temp.ks);
                    rec.ps = std::move(temp.ps);
                    pos += formatInfo_.recordSize;
                    success = true;
                }
            } else {
                std::string line;
                if (std::getline(dataStream, line)) {
                    pos += line.size() + 1;
                    PixTopProbs3D<int32_t> temp;
                    success = parseLine(line, temp);
                    if (success) {
                        recX = temp.x;
                        recY = temp.y;
                        recZ = temp.z;
                        rec.ks = std::move(temp.ks);
                        rec.ps = std::move(temp.ps);
                    }
                }
            }
        } else { // float, scale then round to int
            PixTopProbs3D<float> temp;
            if (mode_ & 0x1) { // Binary
                if (temp.read(dataStream, k_)) {
                    pos += formatInfo_.recordSize;
                    success = true;
                }
            } else {
                std::string line;
                if (std::getline(dataStream, line)) {
                    pos += line.size() + 1;
                    success = parseLine(line, temp);
                }
            }
            if (success) {
                recX = static_cast<int32_t>(std::floor(temp.x / res));
                recY = static_cast<int32_t>(std::floor(temp.y / res));
                recZ = static_cast<int32_t>(std::floor(temp.z / res));
                rec.ks = std::move(temp.ks);
                rec.ps = std::move(temp.ps);
            }
        }
        if (success) {
            pixelMap[std::make_tuple(recX, recY, recZ)] = std::move(rec);
        } else if (!dataStream.eof()) {
            error("%s: Corrupted data", __func__);
        }
    }
    return static_cast<int32_t>(pixelMap.size());
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
            float x, y, z = 0.0f;
            std::vector<int32_t> ks(k_);
            std::vector<float> ps(k_);

            if (isInt32) {
                if (coord_dim_ == 3) {
                    PixTopProbs3D<int32_t> temp;
                    if (!temp.read(dataStream_, k_)) break;
                    x = static_cast<float>(temp.x);
                    y = static_cast<float>(temp.y);
                    z = static_cast<float>(temp.z);
                    ks = std::move(temp.ks);
                    ps = std::move(temp.ps);
                } else {
                    PixTopProbs<int32_t> temp;
                    if (!temp.read(dataStream_, k_)) break;
                    x = static_cast<float>(temp.x);
                    y = static_cast<float>(temp.y);
                    ks = std::move(temp.ks);
                    ps = std::move(temp.ps);
                }
            } else {
                if (coord_dim_ == 3) {
                    PixTopProbs3D<float> temp;
                    if (!temp.read(dataStream_, k_)) break;
                    x = temp.x;
                    y = temp.y;
                    z = temp.z;
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
            }

            if (applyRes) {
                x *= res;
                y *= res;
                z *= res;
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

bool TileOperator::parseLine(const std::string& line, PixTopProbs3D<float>& R) const {
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < icol_max_ + 1) return false;
    if (!str2float(tokens[icol_x_], R.x) ||
        !str2float(tokens[icol_y_], R.y)) {
        warning("%s: Error parsing x,y from line: %s", __func__, line.c_str());
        return false;
    }
    if (has_z_) {
        if (!str2float(tokens[icol_z_], R.z)) {
            warning("%s: Error parsing z from line: %s", __func__, line.c_str());
            return false;
        }
    } else {
        R.z = 0;
    }
    if (mode_ & 0x2) {
        R.x *= formatInfo_.pixelResolution;
        R.y *= formatInfo_.pixelResolution;
        R.z *= formatInfo_.pixelResolution;
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

bool TileOperator::parseLine(const std::string& line, PixTopProbs3D<int32_t>& R) const {
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() <= icol_max_ + 1) return false;
    if (!str2int32(tokens[icol_x_], R.x) ||
        !str2int32(tokens[icol_y_], R.y)) {
        return false;
    }
    if (has_z_) {
        if (!str2int32(tokens[icol_z_], R.z)) {
            return false;
        }
    } else {
        R.z = 0;
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

int32_t TileOperator::next(PixTopProbs<float>& out, bool rawCoord) {
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out);
    }
    if (mode_ & 0x1) { // Binary mode
        if (mode_ & 0x4) { // int32
            if (coord_dim_ == 3) {
                PixTopProbs3D<int32_t> temp;
                if (!temp.read(dataStream_, k_)) {
                    done_ = true;
                    return -1;
                }
                out.x = static_cast<float>(temp.x);
                out.y = static_cast<float>(temp.y);
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
            } else {
                PixTopProbs<int32_t> temp;
                if (!temp.read(dataStream_, k_)) {
                    done_ = true;
                    return -1;
                }
                out.x = static_cast<float>(temp.x);
                out.y = static_cast<float>(temp.y);
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
            }
        } else { // float
            if (coord_dim_ == 3) {
                PixTopProbs3D<float> temp;
                if (!temp.read(dataStream_, k_)) {
                    done_ = true;
                    return -1;
                }
                out.x = temp.x;
                out.y = temp.y;
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
            } else {
                if (!out.read(dataStream_, k_)) {
                    done_ = true;
                    return -1;
                }
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
        if (coord_dim_ == 3) {
            PixTopProbs3D<int32_t> temp;
            if (temp.read(dataStream_, k_)) {
                out.x = temp.x;
                out.y = temp.y;
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
                return 1;
            }
        } else {
            if (out.read(dataStream_, k_)) {
                return 1;
            }
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

int32_t TileOperator::next(PixTopProbs3D<float>& out, bool rawCoord) {
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out, rawCoord);
    }
    if (mode_ & 0x1) { // Binary mode
        if (mode_ & 0x4) { // int32
            if (coord_dim_ == 3) {
                PixTopProbs3D<int32_t> temp;
                if (!temp.read(dataStream_, k_)) {
                    done_ = true;
                    return -1;
                }
                out.x = static_cast<float>(temp.x);
                out.y = static_cast<float>(temp.y);
                out.z = static_cast<float>(temp.z);
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
            } else {
                PixTopProbs<int32_t> temp;
                if (!temp.read(dataStream_, k_)) {
                    done_ = true;
                    return -1;
                }
                out.x = static_cast<float>(temp.x);
                out.y = static_cast<float>(temp.y);
                out.z = 0.0f;
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
            }
        } else { // float
            if (coord_dim_ == 3) {
                if (!out.read(dataStream_, k_)) {
                    done_ = true;
                    return -1;
                }
            } else {
                PixTopProbs<float> temp;
                if (!temp.read(dataStream_, k_)) {
                    done_ = true;
                    return -1;
                }
                out.x = temp.x;
                out.y = temp.y;
                out.z = 0.0f;
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
            }
        }
        if (!rawCoord && (mode_ & 0x2)) {
            out.x *= formatInfo_.pixelResolution;
            out.y *= formatInfo_.pixelResolution;
            out.z *= formatInfo_.pixelResolution;
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
            out.z /= formatInfo_.pixelResolution;
        }
        return 1;
    }
}

int32_t TileOperator::next(PixTopProbs3D<int32_t>& out) {
    if (mode_ & 0x1) {assert(mode_ & 0x4);}
    if (done_) return -1;
    if (bounded_) {
        return nextBounded(out);
    }
    if (mode_ & 0x1) { // Binary mode
        if (coord_dim_ == 3) {
            if (out.read(dataStream_, k_)) {
                return 1;
            }
        } else {
            PixTopProbs<int32_t> temp;
            if (temp.read(dataStream_, k_)) {
                out.x = temp.x;
                out.y = temp.y;
                out.z = 0;
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
                return 1;
            }
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
                if (coord_dim_ == 3) {
                    PixTopProbs3D<int32_t> temp;
                    if (!temp.read(dataStream_, k_)) {
                        error("%s: Corrupted data or invalid index", __func__);
                    }
                    out.x = static_cast<float>(temp.x);
                    out.y = static_cast<float>(temp.y);
                    out.ks = std::move(temp.ks);
                    out.ps = std::move(temp.ps);
                } else {
                    PixTopProbs<int32_t> temp;
                    if (!temp.read(dataStream_, k_)) {
                        error("%s: Corrupted data or invalid index", __func__);
                    }
                    out.x = static_cast<float>(temp.x);
                    out.y = static_cast<float>(temp.y);
                    out.ks = std::move(temp.ks);
                    out.ps = std::move(temp.ps);
                }
            } else {
                if (coord_dim_ == 3) {
                    PixTopProbs3D<float> temp;
                    if (!temp.read(dataStream_, k_)) {
                        error("%s: Corrupted data or invalid index", __func__);
                    }
                    out.x = temp.x;
                    out.y = temp.y;
                    out.ks = std::move(temp.ks);
                    out.ps = std::move(temp.ps);
                } else {
                    if (!out.read(dataStream_, k_)) {
                        error("%s: Corrupted data or invalid index", __func__);
                    }
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
            if (coord_dim_ == 3) {
                PixTopProbs3D<int32_t> temp;
                if (!temp.read(dataStream_, k_)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
                out.x = temp.x;
                out.y = temp.y;
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
            } else {
                if (!out.read(dataStream_, k_)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
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

int32_t TileOperator::nextBounded(PixTopProbs3D<float>& out, bool rawCoord) {
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
                if (coord_dim_ == 3) {
                    PixTopProbs3D<int32_t> temp;
                    if (!temp.read(dataStream_, k_)) {
                        error("%s: Corrupted data or invalid index", __func__);
                    }
                    out.x = static_cast<float>(temp.x);
                    out.y = static_cast<float>(temp.y);
                    out.z = static_cast<float>(temp.z);
                    out.ks = std::move(temp.ks);
                    out.ps = std::move(temp.ps);
                } else {
                    PixTopProbs<int32_t> temp;
                    if (!temp.read(dataStream_, k_)) {
                        error("%s: Corrupted data or invalid index", __func__);
                    }
                    out.x = static_cast<float>(temp.x);
                    out.y = static_cast<float>(temp.y);
                    out.z = 0.0f;
                    out.ks = std::move(temp.ks);
                    out.ps = std::move(temp.ps);
                }
            } else {
                if (coord_dim_ == 3) {
                    if (!out.read(dataStream_, k_)) {
                        error("%s: Corrupted data or invalid index", __func__);
                    }
                } else {
                    PixTopProbs<float> temp;
                    if (!temp.read(dataStream_, k_)) {
                        error("%s: Corrupted data or invalid index", __func__);
                    }
                    out.x = temp.x;
                    out.y = temp.y;
                    out.z = 0.0f;
                    out.ks = std::move(temp.ks);
                    out.ps = std::move(temp.ps);
                }
            }
            pos_ += formatInfo_.recordSize;
            if (blk.contained && rawCoord) {
                return 1;
            }
            float x = out.x, y = out.y, z = out.z;
            if (mode_ & 0x2) {
                x *= formatInfo_.pixelResolution;
                y *= formatInfo_.pixelResolution;
                z *= formatInfo_.pixelResolution;
                if (!rawCoord) {
                    out.x = x;
                    out.y = y;
                    out.z = z;
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
        PixTopProbs3D<float> rec;
        if (!parseLine(line, rec)) return 0;
        if (blk.contained || queryBox_.contains(rec.x, rec.y)) {
            out = std::move(rec);
            if (rawCoord && (mode_ & 0x2)) {
                out.x /= formatInfo_.pixelResolution;
                out.y /= formatInfo_.pixelResolution;
                out.z /= formatInfo_.pixelResolution;
            }
            return 1;
        }
        // else skip it, keep reading
    }
}

int32_t TileOperator::nextBounded(PixTopProbs3D<int32_t>& out) {
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
            if (coord_dim_ == 3) {
                if (!out.read(dataStream_, k_)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
            } else {
                PixTopProbs<int32_t> temp;
                if (!temp.read(dataStream_, k_)) {
                    error("%s: Corrupted data or invalid index", __func__);
                }
                out.x = temp.x;
                out.y = temp.y;
                out.z = 0;
                out.ks = std::move(temp.ks);
                out.ps = std::move(temp.ps);
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
        if (tokens[i] == "X" || tokens[i] == "Y" || tokens[i] == "Z") {
            std::transform(tokens[i].begin(), tokens[i].end(), tokens[i].begin(), ::tolower);
        }
        header[tokens[i]] = i;
    }
    if (header.find("x") == header.end() || header.find("y") == header.end()) {
        return;
    }
    icol_x_ = header["x"];
    icol_y_ = header["y"];
    has_z_ = (header.find("z") != header.end());
    if (has_z_) {
        icol_z_ = header["z"];
        coord_dim_ = 3;
    } else {
        coord_dim_ = 2;
    }
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
    kvec_.clear(); kvec_.push_back(k_);
    icol_max_ = std::max(icol_x_, icol_y_);
    if (has_z_) {
        icol_max_ = std::max(icol_max_, icol_z_);
    }
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
    has_z_ = header.contains("z");
    if (has_z_) {
        icol_z_ = header["z"];
        coord_dim_ = 3;
    } else {
        coord_dim_ = 2;
    }
    while (header.contains("K" + std::to_string(k)) && header.contains("P" + std::to_string(k))) {
        icol_ks_.push_back(header["K" + std::to_string(k)]);
        icol_ps_.push_back(header["P" + std::to_string(k)]);
        k++;
    }
    if (icol_ks_.empty()) {
        error("No K and P columns found in the header");
    }
    k_ = k - 1;
    kvec_.clear(); kvec_.push_back(k_);
    icol_max_ = std::max(icol_x_, icol_y_);
    if (has_z_) {
        icol_max_ = std::max(icol_max_, icol_z_);
    }
    for (int i = 0; i < k_; ++i) {
        icol_max_ = std::max(icol_max_, std::max(icol_ks_[i], icol_ps_[i]));
    }
}
