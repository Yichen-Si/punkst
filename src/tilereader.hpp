#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdio>
#include <unordered_map>
#include <stdexcept>
#include <memory>
#include "utils.h"
#include "dataunits.hpp"
#include "error.hpp"

#include "Eigen/Sparse"
using Eigen::SparseMatrix;

enum class CoordType { INTEGER, FLOAT };

// for serializing to temporary files
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

// Structure holding the offsets for a tile’s data
struct TileInfo {
    std::streampos startOffset;
    std::streampos endOffset;
    bool partial = false;
};

// Parse a line in the tsv tiled pixel file
struct lineParser {
    size_t icol_x, icol_y, icol_feature;
    std::vector<int32_t> icol_ct;
    std::unordered_map<std::string, uint32_t> featureDict;
    int32_t n_ct, n_tokens;
    bool isFeatureDict, weighted;
    std::vector<double> weights;
    std::vector<Rectangle<double>> rects;
    std::vector<uint32_t> icol_ints, icol_floats, icol_strs, str_lens;
    std::vector<std::string> name_ints, name_floats, name_strs;
    bool isExtended = false;

    lineParser() {isFeatureDict = false; weighted = false;}
    lineParser(size_t _ix, size_t _iy, size_t _iz, const std::vector<int32_t>& _ivals, const std::string& _dfile, std::vector<Rectangle<double>>* _rects = nullptr) {
        weighted = false;
        if (_rects != nullptr && !_rects->empty()) {
            rects = *_rects;
        }
        init(_ix, _iy, _iz, _ivals, _dfile);
    }

    void init(size_t _ix, size_t _iy, size_t _iz, const std::vector<int32_t>& _ivals, const std::string& _dfile) {
        icol_x = _ix;
        icol_y = _iy;
        icol_feature = _iz;
        n_ct = _ivals.size();
        icol_ct.resize(n_ct);
        n_tokens = icol_feature;
        for (int32_t i = 0; i < n_ct; ++i) {
            icol_ct[i] = _ivals[i];
            n_tokens = std::max(n_tokens, icol_ct[i]);
        }
        n_tokens += 1;
        if (_dfile.empty()) {
            isFeatureDict = false;
        } else {
            isFeatureDict = true;
            std::ifstream dictFile(_dfile);
            if (!dictFile) {
                error("Error opening feature dictionary file: %s", _dfile.c_str());
            }
            std::string line;
            uint32_t nfeature = 0;
            while (std::getline(dictFile, line)) {
                if (line.empty() || line[0] == '#') continue;
                size_t pos = line.find_first_of(" \t");
                if (pos != std::string::npos) {
                    line = line.substr(0, pos);
                }
                if (line.empty()) continue;
                featureDict[line] = nfeature++;
            }
            if (featureDict.empty()) {
                error("Error reading feature dictionary file: %s", _dfile.c_str());
            }
            notice("Read %zu features from dictionary file", featureDict.size());
        }
    }

    bool checkExtraColumns() {
        if (icol_strs.size() != str_lens.size()) {
            return false;
        }
        if (name_ints.size() < icol_ints.size()) {
            uint32_t i = name_ints.size();
            while (i < icol_ints.size()) {
                name_ints.push_back("int_" + std::to_string(i));
                i++;
            }
        }
        if (name_floats.size() < icol_floats.size()) {
            uint32_t i = name_floats.size();
            while (i < icol_floats.size()) {
                name_floats.push_back("float_" + std::to_string(i));
                i++;
            }
        }
        if (name_strs.size() < icol_strs.size()) {
            uint32_t i = name_strs.size();
            while (i < icol_strs.size()) {
                name_strs.push_back("str_" + std::to_string(i));
                i++;
            }
        }
        return true;
    }

    void setFeatureDict(const std::unordered_map<std::string, uint32_t>& dict) {
        featureDict = dict;
        isFeatureDict = true;
    }

    int32_t getFeatureList(std::vector<std::string>& featureList) {
        if (!isFeatureDict) {
            return -1;
        }
        featureList.resize(featureDict.size());
        for (const auto& pair : featureDict) {
            featureList[pair.second] = pair.first;
        }
        return featureDict.size();
    }

    int32_t parse(PixelValues& pixel, std::string& line, bool checkBounds = false) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < n_tokens) {
            return -1;
        }
        pixel.x = std::stod(tokens[icol_x]);
        pixel.y = std::stod(tokens[icol_y]);
        if (checkBounds) {
            bool valid = false;
            for (const auto& rect : rects) {
                if (rect.contains(pixel.x, pixel.y)) {
                    valid = true;
                    break;
                }
            }
            if (!valid) {
                return 0;
            }
        }
        if (isFeatureDict) {
            auto it = featureDict.find(tokens[icol_feature]);
            if (it == featureDict.end()) {
                return 0;
            }
            pixel.feature = it->second;
        } else {
            if (!str2uint32(tokens[icol_feature], pixel.feature)) {
                return -1;
            }
        }
        pixel.intvals.resize(n_ct);
        int32_t totVal = 0;
        for (size_t i = 0; i < n_ct; ++i) {
            if (!str2int32(tokens[icol_ct[i]], pixel.intvals[i])) {
                return -1;
            }
            totVal += pixel.intvals[i];
        }
        return totVal;
    }

    int32_t readWeights(const std::string& weightFile, double defaultWeight = 1.0, int32_t nFeatures = -1) {
        std::ifstream inWeight(weightFile);
        if (!inWeight) {
            error("Error opening weights file: %s", weightFile.c_str());
        }
        weighted = true;
        int32_t novlp = 0;
        std::string line;
        if (isFeatureDict) {
            weights.resize(featureDict.size());
            std::fill(weights.begin(), weights.end(), defaultWeight);

            while (std::getline(inWeight, line)) {
                std::istringstream iss(line);
                std::string feature;
                double weight;
                if (!(iss >> feature >> weight)) {
                    error("Error reading weights file st line: %s", line.c_str());
                }
                auto it = featureDict.find(feature);
                if (it != featureDict.end()) {
                    weights[it->second] = weight;
                    novlp++;
                } else {
                    continue;
                }
            }
            return novlp;
        }
        if (nFeatures > 0) {
            weights.resize(nFeatures);
            std::fill(weights.begin(), weights.end(), defaultWeight);
            while (std::getline(inWeight, line)) {
                std::istringstream iss(line);
                uint32_t idx;
                double weight;
                if (!(iss >> idx >> weight)) {
                    error("Error reading weights file st line: %s", line.c_str());
                }
                if (idx >= nFeatures) {
                    warning("Weight file feature out of range: %s", line.c_str());
                    continue;
                }
                weights[idx] = weight;
                novlp++;
            }
            return novlp;
        }

        int32_t max_idx = 0;
        std::unordered_map<uint32_t, double> weights_map;
        while (std::getline(inWeight, line)) {
            std::istringstream iss(line);
            uint32_t idx;
            double weight;
            if (!(iss >> idx >> weight)) {
                error("Error reading weights file st line: %s", line.c_str());
            }
            if (idx >= max_idx) {
                max_idx = idx;
            }
            weights_map[idx] = weight;
        }
        novlp = weights_map.size();
        weights.resize(max_idx + 1);
        std::fill(weights.begin(), weights.end(), defaultWeight);
        for (const auto& pair : weights_map) {
            weights[pair.first] = pair.second;
        }
        return novlp;
    }

};

struct lineParserUnival : public lineParser {
    size_t icol_val;
    lineParserUnival() {}
    lineParserUnival(size_t _ix, size_t _iy, size_t _iz, size_t _ival, const std::string& _dfile = "", std::vector<Rectangle<double>>* _rects = nullptr) {
        if (_rects != nullptr && !_rects->empty()) {
            rects = *_rects;
        }
        icol_val = _ival;
        std::vector<int32_t> _ivals = { (int32_t) _ival};
        init(_ix, _iy, _iz, _ivals, _dfile);
    }

    template<typename T>
    int32_t parse(RecordT<T>& rec, std::string& line, bool checkBounds = false) const {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < n_tokens) {
            return -2;
        }
        if (!str2num<T>(tokens[icol_x], rec.x) || !str2num<T>(tokens[icol_y], rec.y)) {
            return -2;
        }
        if (checkBounds) {
            bool valid = false;
            for (const auto& rect : rects) {
                if (rect.contains(rec.x, rec.y)) {
                    valid = true;
                    break;
                }
            }
            if (!valid) {
                return -1;
            }
        }
        if (isFeatureDict) {
            auto it = featureDict.find(tokens[icol_feature]);
            if (it == featureDict.end()) {
                return -1;
            }
            rec.idx = it->second;
        } else {
            rec.idx = std::stoul(tokens[icol_feature]);
        }
        rec.ct = std::stoi(tokens[icol_val]);
        return rec.idx;
    }

    template<typename T>
    int32_t parse( RecordExtendedT<T>& rec, std::string &line ) const {
        int32_t base_idx = parse(rec.recBase, line);
        if (base_idx < 0) return base_idx;
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        // 3) extra ints
        rec.intvals.resize(icol_ints.size());
        for (size_t i = 0; i < icol_ints.size(); ++i) {
            if (!str2num<int32_t>(tokens[icol_ints[i]], rec.intvals[i])) {
                return -2;
            }
        }
        // 4) extra floats
        rec.floatvals.resize(icol_floats.size());
        for (size_t i = 0; i < icol_floats.size(); ++i) {
            if (!str2num<float>(tokens[icol_floats[i]], rec.floatvals[i])) {
                return -2;
            }
        }
        // 5) extra strings (pad/truncate to str_len)
        rec.strvals.resize(icol_strs.size());
        for (size_t i = 0; i < icol_strs.size(); ++i) {
            auto &s = tokens[icol_strs[i]];
            if (s.size() > str_lens[i]) s.resize(str_lens[i]);
            rec.strvals[i] = s;
        }
        return base_idx;
    }
};

class TileReaderBase {
protected:
    std::string tsvFilename;
    int tileSize;
    size_t nTiles;
    // Map from TileKey to TileInfo.
    std::unordered_map<TileKey, TileInfo, TileKeyHash> index;
    // Helper function to load the index file.
    virtual void loadIndex(const std::string &indexFilename) = 0;
    std::vector<Rectangle<double>> rects;

public:
    int32_t minrow = INT32_MAX;
    int32_t mincol = INT32_MAX;
    int32_t maxrow = INT32_MIN;
    int32_t maxcol = INT32_MIN;
    TileReaderBase(const std::string &tsvFilename, const std::string &indexFilename, std::vector<Rectangle<double>>* _rects = nullptr, int32_t tileSize = -1)
        : tsvFilename(tsvFilename), tileSize(tileSize) {
        if (_rects != nullptr) {
            rects = *_rects;
        }
    }
    int32_t getTileSize() const {
        return tileSize;
    }
    size_t getNumTiles() const {
        return nTiles;
    }
    int32_t tile2int(int32_t row, int32_t col) const {
        return (maxcol - mincol) * (row - minrow) + (col - mincol);
    }
    // given (x, y) compute the tile key and whether the tile is in the data
    template<typename T>
    bool pt2tile(T x, T y, TileKey &tile) const {
        tile.row = static_cast<int32_t>(std::floor(y / tileSize));
        tile.col = static_cast<int32_t>(std::floor(x / tileSize));
        return index.find(tile) != index.end();
    }
    bool isPartial(TileKey &tile) const {
        auto it = index.find(tile);
        if (it == index.end()) {
            return false;
        }
        return it->second.partial;
    }
    void getTileList(std::vector<TileKey> &tileList) const {
        tileList.reserve(nTiles);
        tileList.clear();
        for (const auto &pair : index) {
            tileList.push_back(pair.first);
        }
    }
    template<typename T>
    void getTilesInBounds(const std::vector<Rectangle<T>>& _rects,
        std::unordered_map<TileKey,bool,TileKeyHash>& tileMap) {
        for (const auto& r : _rects) {
            if (!r.proper()) continue;
            // Compute candidate tile‐row/col ranges
            int rowMin = static_cast<int>( std::floor(static_cast<double>(r.ymin) / tileSize) );
            int rowMax = static_cast<int>( std::floor(static_cast<double>(r.ymax) / tileSize) );
            int colMin = static_cast<int>( std::floor(static_cast<double>(r.xmin) / tileSize) );
            int colMax = static_cast<int>( std::floor(static_cast<double>(r.xmax) / tileSize) );

            for (int row = rowMin; row <= rowMax; ++row) {
                for (int col = colMin; col <= colMax; ++col) {
                    TileKey key{row, col};
                    // the exact bounds of this tile in world‐space
                    Rectangle<T> tileRect(
                        static_cast<T>(col     * tileSize),
                        static_cast<T>(row     * tileSize),
                        static_cast<T>((col+1) * tileSize),
                        static_cast<T>((row+1) * tileSize)
                    );
                    int32_t code = tileRect.intersect(r);
                    if (code == 0)
                        continue;             // no overlap

                    bool fullyInThisRect = (code == 2);
                    auto it = tileMap.find(key);
                    if (it == tileMap.end()) {
                        tileMap.emplace(key, fullyInThisRect);
                    }
                    else if (!it->second && fullyInThisRect) {
                        // once fully-contained, always fully-contained
                        it->second = true;
                    }
                }
            }
        }
    }

    template<typename T>
    void getTileList(const std::vector<Rectangle<T>>& _rects,
                     std::vector<TileKey>&            tileList,
                     std::vector<bool>&               isContained) const {
        if (!isValid()) {
            throw std::runtime_error("TileReaderBase is not initialized or has empty index");
        }

        // Map each TileKey -> whether we've seen it _fully contained_
        std::unordered_map<TileKey,bool,TileKeyHash> tileMap;
        tileMap.reserve(index.size());
        getTilesInBounds(_rects, tileMap);

        // Flatten
        tileList.clear();
        isContained.clear();
        for (auto &kv : tileMap) {
            tileList.push_back(kv.first);
            isContained.push_back(kv.second);
        }
    }

    bool isValid() const {
        return !index.empty() && tileSize > 0;
    }
    // Given a focal tile returns a vector of TileKey for adjacent tiles.
    int32_t find_adjacent_tiles(std::vector<TileKey>& adjacent, int focalRow, int focalCol) const {
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue; // skip the focal tile itself
                TileKey key {focalRow + dr, focalCol + dc};
                if (index.find(key) != index.end()) {
                    adjacent.push_back(key);
                }
            }
        }
        return adjacent.size();
    }
};

/**
 * Read a plain text file containing indexed tiles
 */
class TileReader : public TileReaderBase {
public:

    TileReader(const std::string &tsvFilename, const std::string &indexFilename, std::vector<Rectangle<double>>* _rects = nullptr, int32_t tileSize = -1, bool isInt = false)
        : TileReaderBase(tsvFilename, indexFilename, _rects, tileSize) {
        loadIndex(indexFilename);
        coordType = isInt ? CoordType::INTEGER : CoordType::FLOAT;
    }

    // Given a tile identified by (tileRow, tileCol), returns an iterator
    // that reads the corresponding chunk (lines) from the TSV file.
    // Throws a runtime error if the tile is not found in the index.
    std::unique_ptr<BoundedReadline> get_tile_iterator(int tileRow, int tileCol) const {
        TileKey key {tileRow, tileCol};
        auto it = index.find(key);
        if (it == index.end()) {
            warning("%s: Tile (%d, %d) not found in index", __FUNCTION__, tileRow, tileCol);
            return nullptr;
        }
        const TileInfo &info = it->second;
        return std::make_unique<BoundedReadline>(tsvFilename, info.startOffset, info.endOffset);
    }

    CoordType getCoordType() const { return coordType; }
private:
    CoordType coordType;
    void loadIndex(const std::string &indexFilename) {
        std::ifstream indexFile(indexFilename);
        if (!indexFile.is_open()) {
            throw std::runtime_error("Unable to open index file: " + indexFilename);
        }

        std::string line;
        // Read metadata line; expected format: "# tilesize<TAB><tileSize>"
        if (!std::getline(indexFile, line)) {
            throw std::runtime_error("Index file is empty");
        }
        if (line.empty()) {
            throw std::runtime_error("Index file is malformed");
        }
        while (line[0] == '#') {
            std::istringstream metaStream(line);
            std::string hashtag, key;
            metaStream >> hashtag >> key;
            if (key == "tilesize") {
                if (!(metaStream >> tileSize)) {
                    throw std::runtime_error("Failed to read tile size from metadata");
                }
            }
            if (!std::getline(indexFile, line)) {
                throw std::runtime_error("Index file is empty");
            }
        }
        if (tileSize <= 0) {
            throw std::runtime_error("Tile size is not specified or found in index file");
        }

        if (rects.size() > 0) {
            std::unordered_map<TileKey,bool,TileKeyHash> tileMap;
            getTilesInBounds(rects, tileMap);
            while (1) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                int row, col;
                std::streamoff start, end;
                // read each line: tilerow, tilecol, startOffset, endOffset
                if (!(iss >> row >> col >> start >> end)) {
                    throw std::runtime_error("Malformed index line: " + line);
                }
                auto it = tileMap.find(TileKey{row, col});
                if (it != tileMap.end()) {
                    index.emplace(TileKey{row, col}, TileInfo{start, end, !(it->second)});
                    if (row < minrow) minrow = row;
                    if (col < mincol) mincol = col;
                    if (row > maxrow) maxrow = row;
                    if (col > maxcol) maxcol = col;
                }
                if (!std::getline(indexFile, line)) {
                    break;  // End of file
                }
            }
        } else {
            while (1) {
                if (line.empty()) continue;
                std::istringstream iss(line);
                int row, col;
                std::streamoff start, end;
                if (!(iss >> row >> col >> start >> end)) {
                    throw std::runtime_error("Malformed index line: " + line);
                }
                index.emplace(TileKey{row, col}, TileInfo{start, end});
                if (row < minrow) minrow = row;
                if (col < mincol) mincol = col;
                if (row > maxrow) maxrow = row;
                if (col > maxcol) maxcol = col;
                if (!std::getline(indexFile, line)) {
                    break;  // End of file
                }
            }
        }
        nTiles = index.size();
        indexFile.close();
        notice("Read %zu tiles from index file", nTiles);
    }
};

class BoundedBinaryTileIterator {
public:
    //   filename: the binary file name,
    //   start, end: byte offsets for this tile's records,
    //   recordSize: fixed size of one record,
    //   numCat: number of categorical fields,
    //   numInt: number of integer fields,
    //   numFloat: number of float fields.
    BoundedBinaryTileIterator(const std::string &filename,
        std::streampos start, std::streampos end, size_t recordSize,
        int32_t numCat, int32_t numInt, int32_t numFloat)
        : filename(filename), startOffset(start), endOffset(end), recordSize(recordSize), numCat(numCat), numInt(numInt), numFloat(numFloat) {
        file = std::make_unique<std::ifstream>(filename, std::ios::binary);
        if (!file || !file->is_open()) {
            throw std::runtime_error("Failed to open binary file: " + filename);
        }
        file->seekg(startOffset);
    }

    struct Record {
        double x;
        double y;
        std::vector<int32_t> catFields;
        std::vector<int32_t> intFields;
        std::vector<float> floatFields;
        Record() {}
        Record(size_t numCat, size_t numInt, size_t numFloat)
            : catFields(numCat), intFields(numInt), floatFields(numFloat) {}
        Record(double x, double y, size_t numCat, size_t numInt, size_t numFloat)
            : x(x), y(y), catFields(numCat), intFields(numInt), floatFields(numFloat) {}
        void resize(size_t numCat, size_t numInt, size_t numFloat) {
            catFields.resize(numCat);
            intFields.resize(numInt);
            floatFields.resize(numFloat);
        }
    };

    // next() decodes the next fixed-size record into a tab-delimited string.
    bool next(BoundedBinaryTileIterator::Record &record) {
        if (!file || file->tellg() >= endOffset) {
            return false;
        }
        std::vector<char> buffer(recordSize);
        file->read(buffer.data(), recordSize);
        if (file->gcount() != static_cast<std::streamsize>(recordSize)) {
            return false;
        }
        size_t offset = 0;
        record.resize(numCat, numInt, numFloat);
        // Decode x and y coordinates (doubles)
        std::memcpy(&record.x, buffer.data() + offset, sizeof(double));
        offset += sizeof(double);
        std::memcpy(&record.y, buffer.data() + offset, sizeof(double));
        offset += sizeof(double);
        // Decode categorical fields (each stored as int32_t)
        for (int32_t i = 0; i < numCat; ++i) {
            std::memcpy(&record.catFields[i], buffer.data() + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }
        // Decode integer fields (each int32_t)
        for (int i = 0; i < numInt; ++i) {
            std::memcpy(&record.intFields[i], buffer.data() + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }
        // Decode float fields (each float)
        for (int i = 0; i < numFloat; ++i) {
            std::memcpy(&record.floatFields, buffer.data() + offset, sizeof(float));
            offset += sizeof(float);
        }
        return true;
    }

private:
    std::string filename;
    std::streampos startOffset;
    std::streampos endOffset;
    size_t recordSize;
    int32_t numCat;
    int32_t numInt;
    int32_t numFloat;
    std::unique_ptr<std::ifstream> file;
};

// currently only for integer coordinates
class BinaryTileReader : public TileReaderBase {
public:
    // The constructor takes:
    //   binFilename: the binary file produced by the writer,
    //   indexFilename: the corresponding index file,
    //   numInt: number of integer fields,
    //   numFloat: number of float fields.
    // The number of categorical fields is deduced from the dictionary header.
    BinaryTileReader(const std::string &binFilename, const std::string &indexFilename, std::vector<Rectangle<double>>* _rects = nullptr, int32_t tileSize = -1)
        : TileReaderBase(binFilename, indexFilename, _rects, tileSize),
            numCat(0), numInt(0), numFloat(0) {
        loadIndex(indexFilename);
        readDictionaries();
    }

    // Return an iterator that decodes records for the given tile.
    std::unique_ptr<BoundedBinaryTileIterator> get_tile_iterator(int tileRow, int tileCol) const {
        TileKey key {tileRow, tileCol};
        auto it = index.find(key);
        if (it == index.end()) {
            return nullptr; // Tile not found
        }
        const TileInfo &info = it->second;
        return std::make_unique<BoundedBinaryTileIterator>(tsvFilename, info.startOffset, info.endOffset, recordSize, numCat, numInt, numFloat);
    }

protected:
    // Reads the index file. Expected index file format:
    //   # tilesize    <tileSize>
    //   # recordsize  <recordSize>
    //   # nvalues     <nCategorical>\t<nInteger>\t<nFloat>
    //   # dictionaries    <startDictOffset>    <endDictOffset>
    //   then one line per tile: row col startOffset endOffset count
    void loadIndex(const std::string &indexFilename) override {
        std::ifstream indexFile(indexFilename);
        if (!indexFile.is_open()) {
            throw std::runtime_error("Unable to open index file: " + indexFilename);
        }
        std::string line;
        // First metadata line: "# tilesize<TAB><tileSize>"
        while (std::getline(indexFile, line)) {
            if (line[0] != '#') {
                break;
            }
            std::istringstream metaStream(line);
            std::string hashtag, key;
            metaStream >> hashtag >> key;
            if (key == "tilesize") {
                if (!(metaStream >> tileSize)) {
                    throw std::runtime_error("Failed to read tile size from metadata");
                }
            } else if (key == "recordsize") {
                if (!(metaStream >> recordSize)) {
                    throw std::runtime_error("Failed to read record size from metadata");
                }
            } else if (key == "nvalues") {
                if (!(metaStream >> numCat >> numInt >> numFloat)) {
                    throw std::runtime_error("Failed to read number of values from metadata");
                }
            } else if (key == "dictionaries") {
                if (!(metaStream >> dictStartOffset >> dictEndOffset)) {
                    throw std::runtime_error("Failed to read dictionary offsets from metadata");
                }
            }
        }
        // Read remaining lines for tile index.
        while (!line.empty()) {
            std::istringstream iss(line);
            int row, col;
            std::streamoff start, end;
            int count;
            if (!(iss >> row >> col >> start >> end >> count)) {
                throw std::runtime_error("Malformed index line: " + line);
            }
            index.emplace(TileKey{row, col}, TileInfo{start, end});
            if (row < minrow) minrow = row;
            if (col < mincol) mincol = col;
            if (row > maxrow) maxrow = row;
            if (col > maxcol) maxcol = col;
            if (!std::getline(indexFile, line)) {
                break;  // End of file
            }
        }
        nTiles = index.size();
        indexFile.close();
    }

private:
    int32_t numInt;
    int32_t numFloat;
    int32_t numCat;
    size_t recordSize;
    // Categorical dictionaries: one vector<string> per categorical column.
    std::vector<std::vector<std::string>> catDictionaries;
    // Offsets for dictionary region in the binary file.
    std::streamoff dictStartOffset;
    std::streamoff dictEndOffset;

    // Reads the dictionaries from the binary file.
    // The format is:
    //   [uint32_t: number of dictionaries]
    //   For each dictionary:
    //     [uint32_t: number of entries]
    //     [uint64_t: total length of the rest of this record (including delimiters)]
    //     Entries: a single string where each entry is separated by '\t'
    void readDictionaries() {
        std::ifstream infile(tsvFilename, std::ios::binary);
        if (!infile.is_open()) {
            throw std::runtime_error("Unable to open binary file: " + tsvFilename);
        }
        infile.seekg(dictStartOffset);
        uint32_t numDict = 0;
        infile.read(reinterpret_cast<char*>(&numDict), sizeof(numDict));
        catDictionaries.resize(numDict);
        for (uint32_t i = 0; i < numDict; ++i) {
            uint32_t dictSize = 0;
            infile.read(reinterpret_cast<char*>(&dictSize), sizeof(dictSize));
            uint64_t totalLength = 0;
            infile.read(reinterpret_cast<char*>(&totalLength), sizeof(totalLength));
            std::string dictStr(totalLength, '\0');
            infile.read(&dictStr[0], totalLength);
            // Split the dictionary string using '\t' as delimiter.
            catDictionaries[i].resize(dictSize);
            auto& entries = catDictionaries[i];
            size_t pos = 0, j = 0;
            while (true) {
                size_t tabPos = dictStr.find('\t', pos);
                if (tabPos == std::string::npos) {
                    entries[j] = dictStr.substr(pos);
                    break;
                }
                entries[j++] = dictStr.substr(pos, tabPos - pos);
                pos = tabPos + 1;
            }
            if (entries.size() != dictSize) {
                throw std::runtime_error("Dictionary entries count mismatch");
            }
        }
        infile.close();
    }
};
