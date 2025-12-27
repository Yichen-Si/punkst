#pragma once

#include <limits>
#include "utils_sys.hpp"
#include "json.hpp"
#include "tilereader.hpp"

// Index header
#define PUNKST_INDEX_MAGIC 0x50554E4B53544958ULL
struct IndexHeader {
    uint64_t magic = 0;
    int32_t tileSize = 0;
    float pixelResolution = -1;
    int32_t coordType = 0; // 0: float, 1: int32
    int32_t topK = 0;
    uint32_t recordSize = 0; // <= 0 for tsv
    float xmin = -1.0f, xmax = -1.0f, ymin = -1.0f, ymax = -1.0f;
};

// Index entry for one tile
struct IndexEntryF {
    uint64_t st, ed;
    uint32_t n;
    int32_t xmin, xmax, ymin, ymax;
    int32_t row = std::numeric_limits<int32_t>::lowest();
    int32_t col = std::numeric_limits<int32_t>::lowest();
    IndexEntryF() = default;
    IndexEntryF(int32_t r, int32_t c) : row(r), col(c) {}
    IndexEntryF(uint64_t s, uint64_t e, uint32_t nn,
                int32_t x0, int32_t x1, int32_t y0, int32_t y1)
        : st(s), ed(e), n(nn), xmin(x0), xmax(x1), ymin(y0), ymax(y1) {}
};

// Inference result for one pixels
template<typename T>
struct PixTopProbs {
    T x, y;
    std::vector<int32_t> ks;
    std::vector<float> ps;
    PixTopProbs() = default;
    PixTopProbs(T _x, T _y) : x(_x), y(_y) {}
    PixTopProbs(const std::pair<T,T>& c) : x(c.first), y(c.second) {}

    int32_t write(int fd) const {
        if (!write_all(fd, reinterpret_cast<const char*>(&x), sizeof(x))) return -1;
        if (!write_all(fd, reinterpret_cast<const char*>(&y), sizeof(y))) return -1;
        if (!ks.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ks.data()), ks.size() * sizeof(int32_t))) return -1;
        }
        if (!ps.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ps.data()), ps.size() * sizeof(float))) return -1;
        }
        int32_t totalSize = 2 * sizeof(T) + ks.size() * sizeof(int32_t) + ps.size() * sizeof(float);
        return totalSize;
    }

    bool read(std::istream& is, int32_t k) {
        if (!is.read(reinterpret_cast<char*>(&x), sizeof(T))) return false;
        if (!is.read(reinterpret_cast<char*>(&y), sizeof(T))) return false;
        ks.resize(k);
        ps.resize(k);
        if (!is.read(reinterpret_cast<char*>(ks.data()), k * sizeof(int32_t))) return false;
        if (!is.read(reinterpret_cast<char*>(ps.data()), k * sizeof(float))) return false;
        return true;
    }
};

class TileOperator {

public:
    TileOperator(std::string& dataFile, std::string indexFile = "", std::string headerFile = "") : dataFile_(dataFile), indexFile_(indexFile) {
        if (!indexFile.empty()) {
            loadIndex(indexFile);
        }
        if ((mode_ & 0x1) == 0) {
            if (!headerFile.empty()) {
                parseHeaderFile(headerFile);
            } else {
                parseHeaderLine();
            }
        }
    }
    ~TileOperator() {
        if (dataStream_.is_open()) {
            dataStream_.close();
        }
    }

    struct Block {
        struct IndexEntryF idx;
        bool contained;
        int32_t row;
        int32_t col;
        Block() = default;
        Block(const IndexEntryF& e, bool c = false) : idx(e), contained(c) {}
    };

    int32_t getK() const { return k_; }
    int32_t getTileSize() const { return formatInfo_.tileSize; }
    float getPixelResolution() const { return formatInfo_.pixelResolution; }
    bool getBoundingBox(float& xmin, float& xmax, float& ymin, float& ymax) const {
        if (blocks_all_.empty()) return false;
        xmin = globalBox_.xmin; xmax = globalBox_.xmax;
        ymin = globalBox_.ymin; ymax = globalBox_.ymax;
        return true;
    }
    bool getBoundingBox(Rectangle<float>& box) const {
        if (blocks_all_.empty()) return false;
        box = globalBox_;
        return true;
    }

    void setCoordinateColumns(int32_t icol_x, int32_t icol_y) {
        icol_x_ = icol_x;
        icol_y_ = icol_y;
        icol_max_ = std::max(icol_max_, std::max(icol_x_, icol_y_));
    }

    void openDataStream() {
        dataStream_.open(dataFile_);
        if (!dataStream_.is_open()) {
            error("Error opening data file: %s", dataFile_.c_str());
        }
    }

    void resetReader() {
        if (dataStream_.is_open()) {
            dataStream_.clear();
            dataStream_.seekg(0);
        } else {
            openDataStream();
        }
        done_ = false;
        idx_block_ = 0;
        pos_ = 0;
        if (bounded_ && !blocks_.empty()) {
            openBlock(blocks_[0]);
        }
    }

    int32_t query(float qxmin, float qxmax, float qymin, float qymax);

    int32_t next(PixTopProbs<float>& out);

    void printIndex() const;

    void reorgTiles(const std::string& outPrefix, int32_t tileSize = -1);

    void dumpTSV(const std::string& outPrefix = "",  int32_t probDigits = 4, int32_t coordDigits = 2);

private:
    std::string dataFile_, indexFile_;
    std::ifstream dataStream_;
    std::string headerLine_;
    uint32_t icol_x_, icol_y_, icol_max_ = 0;
    std::vector<uint32_t> icol_ks_, icol_ps_;
    int32_t k_ = 0;
    uint32_t mode_ = 0;
    IndexHeader formatInfo_;
    std::vector<Block> blocks_all_, blocks_;
    int32_t idx_block_;
    bool bounded_ = false;
    bool done_ = false;
    Rectangle<float> queryBox_;
    Rectangle<float> globalBox_;
    uint64_t pos_;

    // Determine if a block is strictly within a tile or a boundary block
    void classifyBlocks(int32_t tileSize);
    // Parse header from data file
    void parseHeaderLine();
    // Parse header from json file
    void parseHeaderFile(const std::string& headerFile);
    // Load index
    void loadIndex(const std::string& indexFile);
    // Jump to the beginning of a block
    void openBlock(Block& blk);
    // Get the next record within the bounded query region
    int32_t nextBounded(PixTopProbs<float>& out);
    // Parse a line to extract factor results
    bool parseLine(const std::string& line, PixTopProbs<float>& R) const;

    void reorgTilesBinary(const std::string& outPrefix, int32_t tileSize = -1);

};


template<typename T>
void tile2bound(const TileKey &tile, T& xmin, T& xmax, T& ymin, T& ymax, int32_t tileSize) {
    xmin = static_cast<T>(tile.col * tileSize);
    xmax = static_cast<T>((tile.col + 1) * tileSize);
    ymin = static_cast<T>(tile.row * tileSize);
    ymax = static_cast<T>((tile.row + 1) * tileSize);
}
template<typename T>
void tile2bound(int32_t row, int32_t col, T& xmin, T& xmax, T& ymin, T& ymax, int32_t tileSize) {
    xmin = static_cast<T>(col * tileSize);
    xmax = static_cast<T>((col + 1) * tileSize);
    ymin = static_cast<T>(row * tileSize);
    ymax = static_cast<T>((row + 1) * tileSize);
}
template<typename T>
void tile2bound(int32_t row, int32_t col, Rectangle<T>& rect, int32_t tileSize) {
    rect.xmin = static_cast<T>(col * tileSize);
    rect.xmax = static_cast<T>((col + 1) * tileSize);
    rect.ymin = static_cast<T>(row * tileSize);
    rect.ymax = static_cast<T>((row + 1) * tileSize);
}
template<typename T>
TileKey pt2tile(T x, T y, int32_t tileSize) {
    TileKey tile;
    tile.row = static_cast<int32_t>(std::floor(y / tileSize));
    tile.col = static_cast<int32_t>(std::floor(x / tileSize));
    return tile;
}
