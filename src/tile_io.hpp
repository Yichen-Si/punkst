#pragma once

#include <cstdint>
#include <vector>
#include <unordered_map>
#include <variant>
#include <iostream>
#include <limits>
#include <functional>
#include "utils.h"
#include "img_utils.hpp"
#include "utils_sys.hpp"

// Magic for binary index
#define PUNKST_INDEX_MAGIC 0x50554E4B53544958ULL
// Index header
struct IndexHeader {
    uint64_t magic = 0;
    // mode & 0xFFFF0000 stores K, the total number of factors in the inference
    // mode & 0x1: 0 for tsv, 1 for binary
    // mode & 0x2: 0 for original coordinates, 1 for scaled
    // mode & 0x4: 0 for float coordinates, 1 for int32 coordinates
    // mode & 0x8: 0 for regular grid, 1 for generic rectangular blocks
    // mode & 0x10: 0 for 2D, 1 for 3D
    // mode & 0x20: 0 for isotropic/implicit z resolution, 1 for explicit pixelResolutionZ
    // mode & 0x40: 0 for standard pixel records, 1 for records with an extra feature index
    uint32_t mode = 0;
    int32_t tileSize = 0;
    float pixelResolution = -1; // Must be > 0 if mode & 0x2
    float pixelResolutionZ = -1; // Must be > 0 if mode & 0x20
    uint32_t topK = 0; // how many (factor, probability) pairs are kept per record (per inference set)
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
    uint64_t st = 0, ed = 0;
    uint32_t n = 0;
    int32_t xmin = std::numeric_limits<int32_t>::max();
    int32_t xmax = std::numeric_limits<int32_t>::lowest();
    int32_t ymin = std::numeric_limits<int32_t>::max();
    int32_t ymax = std::numeric_limits<int32_t>::lowest(); // Global coordinate, original units
    // row/col may not apply for generic rectangular blocks
    int32_t row = std::numeric_limits<int32_t>::lowest();
    int32_t col = std::numeric_limits<int32_t>::lowest();
    IndexEntryF() = default;
    IndexEntryF(int32_t r, int32_t c) : row(r), col(c) {}
    IndexEntryF(uint64_t s, uint64_t e, uint32_t nn,
                int32_t x0=-1, int32_t x1=-1, int32_t y0=-1, int32_t y1=-1)
        : st(s), ed(e), n(nn), xmin(x0), xmax(x1), ymin(y0), ymax(y1) {}
    void resetBounds() {
        xmin = std::numeric_limits<int32_t>::max();
        xmax = std::numeric_limits<int32_t>::lowest();
        ymin = std::numeric_limits<int32_t>::max();
        ymax = std::numeric_limits<int32_t>::lowest();
    }
    bool hasBounds() const {
        return xmin < xmax && ymin < ymax;
    }
    void extendToInclude(float x, float y) {
        const int32_t x0 = static_cast<int32_t>(std::floor(x));
        const int32_t x1 = x0 + 1;
        const int32_t y0 = static_cast<int32_t>(std::floor(y));
        const int32_t y1 = y0 + 1;
        if (x0 < xmin) xmin = x0;
        if (x1 > xmax) xmax = x1;
        if (y0 < ymin) ymin = y0;
        if (y1 > ymax) ymax = y1;
    }
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




/**Input data structures  */

// 2D input record (fixed fields)
#pragma pack(push, 1)
template<typename T>
struct RecordT {
    T x;
    T y;
    uint32_t idx;
    uint32_t ct;
};
#pragma pack(pop)

// 3D input record (fixed fields)
#pragma pack(push, 1)
template<typename T>
struct RecordT3D {
    T x;
    T y;
    T z;
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

template<typename T>
struct RecordExtendedT3D {
    RecordT3D<T> recBase;
    std::vector<int32_t> intvals;
    std::vector<float> floatvals;
    std::vector<std::string> strvals;
    RecordExtendedT3D() : recBase{T{}, T{}, T{}, 0u, 0u} {}
    RecordExtendedT3D(const RecordT3D<T>& base) : recBase(base){}
    RecordExtendedT3D(T x, T y, T z, uint32_t idx, uint32_t ct) : recBase{x, y, z, idx, ct} {}
};

template<typename T>
struct Coord3 {
    T x, y, z;
    Coord3() = default;
    Coord3(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
};

template<typename T>
struct Standard2DTileInput {
    std::vector<RecordT<T>> pts;
};

template<typename T>
struct Extended2DTileInput {
    std::vector<RecordExtendedT<T>> extPts;
};

struct SingleMolecule2DTileInput {
    std::vector<std::pair<float, float>> coordsFloat;
    std::vector<uint32_t> featureIdx;
    std::vector<float> obsWeight;
};

template<typename T>
struct Standard3DTileInput {
    std::vector<RecordT3D<T>> pts3d;
};

template<typename T>
struct Extended3DTileInput {
    std::vector<RecordExtendedT3D<T>> extPts3d;
};

struct SingleMolecule3DTileInput {
    std::vector<Coord3<float>> coords3dFloat;
    std::vector<uint32_t> featureIdx;
    std::vector<float> obsWeight;
};

template<typename T>
struct TileData {
    using InputVariant = std::variant<
        std::monostate,
        Standard2DTileInput<T>,
        Extended2DTileInput<T>,
        SingleMolecule2DTileInput,
        Standard3DTileInput<T>,
        Extended3DTileInput<T>,
        SingleMolecule3DTileInput>;

    float xmin, xmax, ymin, ymax;
    InputVariant input;
    std::vector<uint32_t> rowFeatureIdx;

    // Populated by Tiles2MinibatchBase during minibatch construction
    // For pixel mode only
    std::vector<std::pair<int32_t, int32_t>> coords;
    std::vector<Coord3<int32_t>> coords3d;
    //     Map between pixel coordinates and original points
    std::vector<int32_t> idxinternal; // indices for internal points to output
    std::vector<int32_t> orgpts2pixel; // map from original points to indices of the pixels used in the model. -1 for not used

    bool emptyInput() const {
        return std::holds_alternative<std::monostate>(input);
    }

    Standard2DTileInput<T>& emplaceStandard2D() { return input.template emplace<Standard2DTileInput<T>>(); }
    Extended2DTileInput<T>& emplaceExtended2D() { return input.template emplace<Extended2DTileInput<T>>(); }
    SingleMolecule2DTileInput& emplaceSingleMolecule2D() { return input.template emplace<SingleMolecule2DTileInput>(); }
    Standard3DTileInput<T>& emplaceStandard3D() { return input.template emplace<Standard3DTileInput<T>>(); }
    Extended3DTileInput<T>& emplaceExtended3D() { return input.template emplace<Extended3DTileInput<T>>(); }
    SingleMolecule3DTileInput& emplaceSingleMolecule3D() { return input.template emplace<SingleMolecule3DTileInput>(); }

    bool isStandard2D() const { return std::holds_alternative<Standard2DTileInput<T>>(input); }
    bool isExtended2D() const { return std::holds_alternative<Extended2DTileInput<T>>(input); }
    bool isSingleMolecule2D() const { return std::holds_alternative<SingleMolecule2DTileInput>(input); }
    bool isStandard3D() const { return std::holds_alternative<Standard3DTileInput<T>>(input); }
    bool isExtended3D() const { return std::holds_alternative<Extended3DTileInput<T>>(input); }
    bool isSingleMolecule3D() const { return std::holds_alternative<SingleMolecule3DTileInput>(input); }

    Standard2DTileInput<T>& standard2D() { return std::get<Standard2DTileInput<T>>(input); }
    const Standard2DTileInput<T>& standard2D() const { return std::get<Standard2DTileInput<T>>(input); }
    Extended2DTileInput<T>& extended2D() { return std::get<Extended2DTileInput<T>>(input); }
    const Extended2DTileInput<T>& extended2D() const { return std::get<Extended2DTileInput<T>>(input); }
    SingleMolecule2DTileInput& singleMolecule2D() { return std::get<SingleMolecule2DTileInput>(input); }
    const SingleMolecule2DTileInput& singleMolecule2D() const { return std::get<SingleMolecule2DTileInput>(input); }
    Standard3DTileInput<T>& standard3D() { return std::get<Standard3DTileInput<T>>(input); }
    const Standard3DTileInput<T>& standard3D() const { return std::get<Standard3DTileInput<T>>(input); }
    Extended3DTileInput<T>& extended3D() { return std::get<Extended3DTileInput<T>>(input); }
    const Extended3DTileInput<T>& extended3D() const { return std::get<Extended3DTileInput<T>>(input); }
    SingleMolecule3DTileInput& singleMolecule3D() { return std::get<SingleMolecule3DTileInput>(input); }
    const SingleMolecule3DTileInput& singleMolecule3D() const { return std::get<SingleMolecule3DTileInput>(input); }

    void clear() {
        input = std::monostate{};
        coords.clear();coords3d.clear();
        rowFeatureIdx.clear();
        idxinternal.clear();
        orgpts2pixel.clear();
    }
    void setBounds(float _xmin, float _xmax, float _ymin, float _ymax) {
        xmin = _xmin;
        xmax = _xmax;
        ymin = _ymin;
        ymax = _ymax;
    }
};

struct IBoundaryStorage {
    virtual ~IBoundaryStorage() = default;
};

template<typename T>
struct InMemoryStorageStandard : public IBoundaryStorage {
    std::vector<RecordT<T>> data;
};

template<typename T>
struct InMemoryStorageExtended : public IBoundaryStorage {
    std::vector<RecordExtendedT<T>> dataExtended;
};

template<typename T>
struct InMemoryStorageStandard3D : public IBoundaryStorage {
    std::vector<RecordT3D<T>> data;
};

template<typename T>
struct InMemoryStorageExtended3D : public IBoundaryStorage {
    std::vector<RecordExtendedT3D<T>> dataExtended;
};








/** Inference result structure */

struct TopProbs {
    std::vector<int32_t> ks;
    std::vector<float> ps;
    int32_t write(int fd) const {
        if (!ks.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ks.data()), ks.size() * sizeof(int32_t))) return -1;
        }
        if (!ps.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ps.data()), ps.size() * sizeof(float))) return -1;
        }
        int32_t totalSize = ks.size() * sizeof(int32_t) + ps.size() * sizeof(float);
        return totalSize;
    }
    bool read(std::istream& is, int32_t k) {
        ks.resize(k);
        ps.resize(k);
        if (!is.read(reinterpret_cast<char*>(ks.data()), k * sizeof(int32_t))) return false;
        if (!is.read(reinterpret_cast<char*>(ps.data()), k * sizeof(float))) return false;
        return true;
    }
    float extractFactorProb(int32_t k, float minPixelProb) const {
        float total = 0.0f;
        for (size_t i = 0; i < ks.size() && i < ps.size(); ++i) {
            if (ks[i] != k) continue;
            if (ps[i] >= minPixelProb) {return ps[i];}
            break;
        }
        return total;
    }
};

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

template<typename T>
struct PixTopProbsFeature {
    T x, y;
    uint32_t featureIdx = 0;
    std::vector<int32_t> ks;
    std::vector<float> ps;
    PixTopProbsFeature() = default;
    PixTopProbsFeature(T _x, T _y, uint32_t _featureIdx) : x(_x), y(_y), featureIdx(_featureIdx) {}
    PixTopProbsFeature(const std::pair<T,T>& c, uint32_t _featureIdx) : x(c.first), y(c.second), featureIdx(_featureIdx) {}

    int32_t write(int fd) const {
        if (!write_all(fd, reinterpret_cast<const char*>(&x), sizeof(x))) return -1;
        if (!write_all(fd, reinterpret_cast<const char*>(&y), sizeof(y))) return -1;
        if (!write_all(fd, reinterpret_cast<const char*>(&featureIdx), sizeof(featureIdx))) return -1;
        if (!ks.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ks.data()), ks.size() * sizeof(int32_t))) return -1;
        }
        if (!ps.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ps.data()), ps.size() * sizeof(float))) return -1;
        }
        int32_t totalSize = 2 * sizeof(T) + sizeof(featureIdx) + ks.size() * sizeof(int32_t) + ps.size() * sizeof(float);
        return totalSize;
    }

    bool read(std::istream& is, int32_t k) {
        if (!is.read(reinterpret_cast<char*>(&x), sizeof(T))) return false;
        if (!is.read(reinterpret_cast<char*>(&y), sizeof(T))) return false;
        if (!is.read(reinterpret_cast<char*>(&featureIdx), sizeof(featureIdx))) return false;
        ks.resize(k);
        ps.resize(k);
        if (!is.read(reinterpret_cast<char*>(ks.data()), k * sizeof(int32_t))) return false;
        if (!is.read(reinterpret_cast<char*>(ps.data()), k * sizeof(float))) return false;
        return true;
    }
};

// Inference result for one pixel (3D)
template<typename T>
struct PixTopProbs3D {
    T x, y, z;
    std::vector<int32_t> ks;
    std::vector<float> ps;
    PixTopProbs3D() = default;
    PixTopProbs3D(T _x, T _y, T _z) : x(_x), y(_y), z(_z) {}
    PixTopProbs3D(const Coord3<T>& c) : x(c.x), y(c.y), z(c.z) {}

    int32_t write(int fd) const {
        if (!write_all(fd, reinterpret_cast<const char*>(&x), sizeof(x))) return -1;
        if (!write_all(fd, reinterpret_cast<const char*>(&y), sizeof(y))) return -1;
        if (!write_all(fd, reinterpret_cast<const char*>(&z), sizeof(z))) return -1;
        if (!ks.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ks.data()), ks.size() * sizeof(int32_t))) return -1;
        }
        if (!ps.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ps.data()), ps.size() * sizeof(float))) return -1;
        }
        int32_t totalSize = 3 * sizeof(T) + ks.size() * sizeof(int32_t) + ps.size() * sizeof(float);
        return totalSize;
    }

    bool read(std::istream& is, int32_t k) {
        if (!is.read(reinterpret_cast<char*>(&x), sizeof(T))) return false;
        if (!is.read(reinterpret_cast<char*>(&y), sizeof(T))) return false;
        if (!is.read(reinterpret_cast<char*>(&z), sizeof(T))) return false;
        ks.resize(k);
        ps.resize(k);
        if (!is.read(reinterpret_cast<char*>(ks.data()), k * sizeof(int32_t))) return false;
        if (!is.read(reinterpret_cast<char*>(ps.data()), k * sizeof(float))) return false;
        return true;
    }
};

template<typename T>
struct PixTopProbsFeature3D {
    T x, y, z;
    uint32_t featureIdx = 0;
    std::vector<int32_t> ks;
    std::vector<float> ps;
    PixTopProbsFeature3D() = default;
    PixTopProbsFeature3D(T _x, T _y, T _z, uint32_t _featureIdx) : x(_x), y(_y), z(_z), featureIdx(_featureIdx) {}
    PixTopProbsFeature3D(const Coord3<T>& c, uint32_t _featureIdx) : x(c.x), y(c.y), z(c.z), featureIdx(_featureIdx) {}

    int32_t write(int fd) const {
        if (!write_all(fd, reinterpret_cast<const char*>(&x), sizeof(x))) return -1;
        if (!write_all(fd, reinterpret_cast<const char*>(&y), sizeof(y))) return -1;
        if (!write_all(fd, reinterpret_cast<const char*>(&z), sizeof(z))) return -1;
        if (!write_all(fd, reinterpret_cast<const char*>(&featureIdx), sizeof(featureIdx))) return -1;
        if (!ks.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ks.data()), ks.size() * sizeof(int32_t))) return -1;
        }
        if (!ps.empty()) {
            if (!write_all(fd, reinterpret_cast<const char*>(ps.data()), ps.size() * sizeof(float))) return -1;
        }
        int32_t totalSize = 3 * sizeof(T) + sizeof(featureIdx) + ks.size() * sizeof(int32_t) + ps.size() * sizeof(float);
        return totalSize;
    }

    bool read(std::istream& is, int32_t k) {
        if (!is.read(reinterpret_cast<char*>(&x), sizeof(T))) return false;
        if (!is.read(reinterpret_cast<char*>(&y), sizeof(T))) return false;
        if (!is.read(reinterpret_cast<char*>(&z), sizeof(T))) return false;
        if (!is.read(reinterpret_cast<char*>(&featureIdx), sizeof(featureIdx))) return false;
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
