#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <iostream>
#include <limits>
#include <functional>
#include "utils.h"

// Magic for binary index
#define PUNKST_INDEX_MAGIC 0x50554E4B53544958ULL

// Index header
struct IndexHeader {
    uint64_t magic = 0;
    uint32_t mode = 0; // tsv/binary, org/scale, float/int xy, regular/no
    int32_t tileSize = 0;
    float pixelResolution = -1;
    int32_t coordType = 0; // 0: float, 1: int32
    uint32_t topK = 0; // packed top K info
    uint32_t recordSize = 0; // <= 0 for tsv
    float xmin = -1.0f, xmax = -1.0f, ymin = -1.0f, ymax = -1.0f;

    int32_t parseKvec(std::vector<uint32_t>& kvec) {
        kvec.clear();
        if (topK & (1u << 31)) {
            uint32_t n = (topK >> 16) & 0x7FFF;
            uint32_t t = topK & 0xFFFF;
            kvec.assign(n, (uint32_t)(t / n));
            return t;
        }
        uint32_t u = topK;
        int32_t totalK = 0;
        while (u > 0) {
            uint32_t ki = u & 0xF;
            kvec.push_back(ki);
            totalK += ki;
            u >>= 4;
        }
        return totalK;
    }
    bool packKvec(const std::vector<uint32_t>& kvec) {
        if (kvec.size() > 7) {
            topK = 1u << 31;
            uint32_t n = static_cast<uint32_t>(kvec.size());
            topK |= (n & 0x7FFF) << 16;
            uint32_t t = 0;
            bool flag = true;
            uint32_t k0 = kvec[0];
            for (auto k : kvec) {
                t += k;
                if (k != k0) flag = false;
            }
            topK |= (t & 0xFFFF);
            return flag && n <= 0x7FFF && t <= 0xFFFF;
        }
        topK = 0;
        bool flag = true;
        for (size_t i = 0; i < kvec.size(); ++i) {
            if (kvec[i] > 0xF) flag = false;
            topK |= (kvec[i] & 0xF) << (i * 4);
        }
        return flag;
    }
};

// Index entry for one tile (or block)
struct IndexEntryF {
    uint64_t st, ed;
    uint32_t n;
    int32_t xmin, xmax, ymin, ymax;
    // row/col may not apply for generic rectangular blocks
    int32_t row = std::numeric_limits<int32_t>::lowest();
    int32_t col = std::numeric_limits<int32_t>::lowest();
    IndexEntryF() = default;
    IndexEntryF(int32_t r, int32_t c) : row(r), col(c) {}
    IndexEntryF(uint64_t s, uint64_t e, uint32_t nn,
                int32_t x0=-1, int32_t x1=-1, int32_t y0=-1, int32_t y1=-1)
        : st(s), ed(e), n(nn), xmin(x0), xmax(x1), ymin(y0), ymax(y1) {}
};

// Identify a tile by (row, col)
struct TileKey {
    int32_t row;
    int32_t col;
    bool operator==(const TileKey &other) const {
        return row == other.row && col == other.col;
    }
    bool operator<(const TileKey &other) const {
        return row < other.row || (row == other.row && col < other.col);
    }
};

// Custom hash for TileKey
struct TileKeyHash {
    std::size_t operator()(const TileKey &key) const {
        return std::hash<int>()(key.row) ^ (std::hash<int>()(key.col) << 1);
    }
};

// Information about one square tile in a regular grid
struct TileInfo {
    struct IndexEntryF idx;
    bool contained;
    int32_t row;
    int32_t col;
    TileInfo() = default;
    TileInfo(const IndexEntryF& e, bool c = false) : idx(e), contained(c) {}
    TileInfo(uint64_t st, uint64_t ed, bool c = false)
        : idx(st, ed, 0), contained(c),
        row(std::numeric_limits<int32_t>::lowest()),
        col(std::numeric_limits<int32_t>::lowest()) {}
};

// Helper functions to convert between tile key and bounding box
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
Rectangle<T> tile2bound(int32_t row, int32_t col, int32_t tileSize) {
    return Rectangle<T>(static_cast<T>(col     * tileSize),
                        static_cast<T>(row     * tileSize),
                        static_cast<T>((col+1) * tileSize),
                        static_cast<T>((row+1) * tileSize));
}
template<typename T>
TileKey pt2tile(T x, T y, int32_t tileSize) {
    TileKey tile;
    tile.row = static_cast<int32_t>(std::floor(y / tileSize));
    tile.col = static_cast<int32_t>(std::floor(x / tileSize));
    return tile;
}




// Input data structures for serializing to temporary files
#pragma pack(push, 1)
template<typename T>
struct RecordT {
    T x;
    T y;
    uint32_t idx;
    uint32_t ct;
};
#pragma pack(pop)

template<typename T>
struct RecordExtendedT {
    RecordT<T> recBase;
    std::vector<int32_t> intvals;
    std::vector<float> floatvals;
    std::vector<std::string> strvals;
    RecordExtendedT() : recBase{T{}, T{}, 0u, 0u} {}
    RecordExtendedT(const RecordT<T>& base) : recBase(base){}
    RecordExtendedT(T x, T y, uint32_t idx, uint32_t ct) : recBase{x, y, idx, ct} {}
};

// Inference result for one pixel
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


struct IndexEntryF_legacy {
    uint64_t st, ed;
    uint32_t n;
    float xmin, xmax, ymin, ymax;
};
