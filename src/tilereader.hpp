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

class TileReaderBase {
protected:
    std::string tsvFilename;
    int tileSize;
    size_t nTiles;
    int32_t minrow = INT32_MAX;
    int32_t mincol = INT32_MAX;
    int32_t maxrow = INT32_MIN;
    int32_t maxcol = INT32_MIN;
    // Map from TileKey to TileInfo.
    std::unordered_map<TileKey, TileInfo, TileKeyHash> index;
    // Helper function to load the index file.
    virtual void loadIndex(const std::string &indexFilename) = 0;

public:
    TileReaderBase(const std::string &tsvFilename, const std::string &indexFilename, int32_t tileSize = -1)
        : tsvFilename(tsvFilename), tileSize(tileSize) {
    }
    int32_t getTileSize() const {
        return tileSize;
    }
    size_t getNumTiles() const {
        return nTiles;
    }
    int32_t tile2Int(int32_t row, int32_t col) const {
        return (maxcol - mincol) * (row - minrow) + (col - mincol);
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

    // next() decodes the next fixed-size record into a tab-delimited string.
    bool next(Record &record) {
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
