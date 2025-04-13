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

// for serializing to temporary files
#pragma pack(push, 1)
struct Record {
    int32_t x;
    int32_t y;
    uint32_t idx;
    uint32_t ct;
};
#pragma pack(pop)

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
};

// Parse a line in the tsv tiled pixel file
struct lineParser {
    size_t icol_x, icol_y, icol_feature;
    std::vector<int32_t> icol_ints;
    std::unordered_map<std::string, uint32_t> featureDict;
    int32_t n_ints, n_tokens;
    bool isFeatureDict, weighted;
    std::vector<double> weights;

    lineParser() {isFeatureDict = false; weighted = false;}
    lineParser(size_t _ix, size_t _iy, size_t _iz, const std::vector<int32_t>& _ivals, std::string& _dfile) {
        weighted = false;
        init(_ix, _iy, _iz, _ivals, _dfile);
    }
    void init(size_t _ix, size_t _iy, size_t _iz, const std::vector<int32_t>& _ivals, std::string& _dfile) {
        icol_x = _ix;
        icol_y = _iy;
        icol_feature = _iz;
        n_ints = _ivals.size();
        icol_ints.resize(n_ints);
        n_tokens = icol_feature;
        for (int32_t i = 0; i < n_ints; ++i) {
            icol_ints[i] = _ivals[i];
            n_tokens = std::max(n_tokens, icol_ints[i]);
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

    int32_t parse(PixelValues& pixel, std::string& line) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < n_tokens) {
            return -1;
        }
        pixel.x = std::stod(tokens[icol_x]);
        pixel.y = std::stod(tokens[icol_y]);
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
        pixel.intvals.resize(n_ints);
        int32_t totVal = 0;
        for (size_t i = 0; i < n_ints; ++i) {
            if (!str2int32(tokens[icol_ints[i]], pixel.intvals[i])) {
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
    lineParserUnival(size_t _ix, size_t _iy, size_t _iz, size_t _ival, std::string& _dfile) {
        icol_val = _ival;
        std::vector<int32_t> _ivals = { (int32_t) _ival};
        init(_ix, _iy, _iz, _ivals, _dfile);
    }

    int32_t parse(Record& rec, std::string& line) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < n_tokens) {
            return -2;
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
        rec.x = std::stoi(tokens[icol_x]);
        rec.y = std::stoi(tokens[icol_y]);
        rec.ct = std::stoi(tokens[icol_val]);
        return rec.idx;
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

public:
    int32_t minrow = INT32_MAX;
    int32_t mincol = INT32_MAX;
    int32_t maxrow = INT32_MIN;
    int32_t maxcol = INT32_MIN;
    TileReaderBase(const std::string &tsvFilename, const std::string &indexFilename, int32_t tileSize = -1)
        : tsvFilename(tsvFilename), tileSize(tileSize) {
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
    void getTileList(std::vector<TileKey> &tileList) const {
        tileList.reserve(nTiles);
        tileList.clear();
        for (const auto &pair : index) {
            tileList.push_back(pair.first);
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

    TileReader(const std::string &tsvFilename, const std::string &indexFilename, int32_t tileSize = -1)
        : TileReaderBase(tsvFilename, indexFilename, tileSize) {
        loadIndex(indexFilename);
    }

    // Given a tile identified by (tileRow, tileCol), returns an iterator
    // that reads the corresponding chunk (lines) from the TSV file.
    // Throws a runtime error if the tile is not found in the index.
    std::unique_ptr<BoundedReadline> get_tile_iterator(int tileRow, int tileCol) const {
        TileKey key {tileRow, tileCol};
        auto it = index.find(key);
        if (it == index.end()) {
            throw std::runtime_error("Tile (" + std::to_string(tileRow) + "," +
                                        std::to_string(tileCol) + ") not found in index");
        }
        const TileInfo &info = it->second;
        return std::make_unique<BoundedReadline>(tsvFilename, info.startOffset, info.endOffset);
    }

private:
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

        // read each line: tilerow, tilecol, startOffset, endOffset
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
        nTiles = index.size();
        indexFile.close();
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


class BinaryTileReader : public TileReaderBase {
public:
    // The constructor takes:
    //   binFilename: the binary file produced by the writer,
    //   indexFilename: the corresponding index file,
    //   numInt: number of integer fields,
    //   numFloat: number of float fields.
    // The number of categorical fields is deduced from the dictionary header.
    BinaryTileReader(const std::string &binFilename, const std::string &indexFilename, int32_t tileSize = -1)
        : TileReaderBase(binFilename, indexFilename, tileSize),
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
