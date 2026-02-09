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
#include <set>
#include <limits>
#include "utils.h"
#include "img_utils.hpp"
#include "utils_sys.hpp"
#include "dataunits.hpp"
#include "error.hpp"
#include "tile_io.hpp"

#include "Eigen/Sparse"
using Eigen::SparseMatrix;

enum class CoordType { INTEGER, FLOAT };

// Parse a line in the tsv tiled pixel file
struct lineParser {
    size_t icol_x, icol_y, icol_feature;
    size_t icol_z = std::numeric_limits<size_t>::max();
    std::vector<size_t> icol_ct;
    std::unordered_map<std::string, uint32_t> featureDict;
    size_t n_ct, n_tokens;
    bool isFeatureDict, weighted;
    std::vector<double> weights;
    std::vector<Rectangle<double>> rects;
    std::vector<uint32_t> icol_ints, icol_floats, icol_strs, str_lens;
    std::vector<std::string> name_ints, name_floats, name_strs;
    bool isExtended = false;
    bool hasZ = false;

    lineParser() {isFeatureDict = false; weighted = false;}
    lineParser(size_t _ix, size_t _iy, size_t _iw, const std::vector<int32_t>& _ivals, const std::string& _dfile, std::vector<Rectangle<double>>* _rects = nullptr) {
        weighted = false;
        if (_rects != nullptr && !_rects->empty()) {
            rects = *_rects;
        }
        init(_ix, _iy, _iw, _ivals, _dfile);
    }

    void init(size_t _ix, size_t _iy, size_t _iw,
              const std::vector<int32_t>& _ivals,
              const std::string& _dfile);

    bool addExtraInt(std::vector<std::string>& annoInts);
    bool addExtraFloat(std::vector<std::string>& annoFloats);
    bool addExtraStr(std::vector<std::string>& annoStrs);
    bool checkExtraColumns();

    void setFeatureDict(const std::unordered_map<std::string, uint32_t>& dict);
    void setFeatureDict(const std::vector<std::string>& featureList);
    int32_t getFeatureList(std::vector<std::string>& featureList);
    void setZ(size_t col);
    bool hasZCoord() const { return hasZ; }

    int32_t parse(PixelValues& pixel, std::string& line, bool checkBounds = false);
    int32_t parse(PixelValues3D& pixel, std::string& line, bool checkBounds = false);
    int32_t readWeights(const std::string& weightFile, double defaultWeight = 1.0, int32_t nFeatures = -1);
};

struct lineParserUnival : public lineParser {
    size_t icol_val;
    lineParserUnival() {}
    lineParserUnival(size_t _ix, size_t _iy, size_t _iw, size_t _ival, const std::string& _dfile = "", std::vector<Rectangle<double>>* _rects = nullptr) {
        if (_rects != nullptr && !_rects->empty()) {
            rects = *_rects;
        }
        icol_val = _ival;
        std::vector<int32_t> _ivals = { (int32_t) _ival};
        init(_ix, _iy, _iw, _ivals, _dfile);
    }

    template<typename T>
    int32_t parse(RecordT<T>& rec, std::string& line, bool checkBounds = false) const;

    template<typename T>
    int32_t parse( RecordExtendedT<T>& rec, std::string &line ) const;

    template<typename T>
    int32_t parse(RecordT3D<T>& rec, std::string& line, bool checkBounds = false) const;

    template<typename T>
    int32_t parse(RecordExtendedT3D<T>& rec, std::string &line ) const;
};

class TileReaderBase {
protected:
    std::string tsvFilename;
    int tileSize;
    size_t nTiles;
    // Map from TileKey to TileInfo.
    std::unordered_map<TileKey, TileInfo, TileKeyHash> tile_map_;
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
        return tile_map_.find(tile) != tile_map_.end();
    }

    bool isPartial(TileKey &tile) const {
        auto it = tile_map_.find(tile);
        if (it == tile_map_.end()) {
            return false;
        }
        return !(it->second.contained);
    }

    void getTileList(std::vector<TileKey> &tileList) const {
        tileList.reserve(nTiles);
        tileList.clear();
        for (const auto &pair : tile_map_) {
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
                    Rectangle<T> tileRect = tile2bound<T>(row, col, tileSize);
                    int32_t code = tileRect.intersect(r);
                    if (code == 0) continue; // no overlap
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
        tileMap.reserve(tile_map_.size());
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
        return !tile_map_.empty() && tileSize > 0;
    }

    // Given a focal tile returns a vector of TileKey for adjacent tiles.
    int32_t find_adjacent_tiles(std::vector<TileKey>& adjacent, int focalRow, int focalCol) const {
        for (int dr = -1; dr <= 1; ++dr) {
            for (int dc = -1; dc <= 1; ++dc) {
                if (dr == 0 && dc == 0) continue; // skip the focal tile itself
                TileKey key {focalRow + dr, focalCol + dc};
                if (tile_map_.find(key) != tile_map_.end()) {
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

    std::string headerLine;

    TileReader(const std::string &tsvFilename, const std::string &indexFilename, std::vector<Rectangle<double>>* _rects = nullptr, int32_t tileSize = -1, bool isInt = false)
        : TileReaderBase(tsvFilename, indexFilename, _rects, tileSize) {
        loadIndex(indexFilename);
        coordType = isInt ? CoordType::INTEGER : CoordType::FLOAT;
        { // Read and store header/metadata lines
            std::ifstream tsvFile(tsvFilename);
            if (!tsvFile.is_open()) {
                throw std::runtime_error("Unable to open TSV file: " + tsvFilename);
            }
            std::string line;
            headerLine.clear();
            while (std::getline(tsvFile, line)) {
                if (line.empty()) {
                    continue;
                }
                if (line[0] != '#') {
                    break;
                }
                headerLine += line + "\n";
            }
            if (!headerLine.empty()) {headerLine.pop_back();}
        }
    }

    // Given a tile identified by (tileRow, tileCol), returns an iterator
    // that reads the corresponding chunk (lines) from the TSV file.
    // Throws a runtime error if the tile is not found in the index.
    std::unique_ptr<BoundedReadline> get_tile_iterator(int tileRow, int tileCol) const;
    const Rectangle<float>& getGlobalBox() const { return globalBox_; }

    CoordType getCoordType() const { return coordType; }
private:
    Rectangle<float> globalBox_;
    CoordType coordType;
    void loadIndex(const std::string &indexFilename) override;
    bool loadIndexText(const std::string &indexFilename);
    bool loadIndexBinary(const std::string &indexFilename);
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
    bool next(BoundedBinaryTileIterator::Record &record);

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
    std::unique_ptr<BoundedBinaryTileIterator> get_tile_iterator(int tileRow, int tileCol) const;

protected:
    // Reads the index file. Expected index file format:
    //   # tilesize    <tileSize>
    //   # recordsize  <recordSize>
    //   # nvalues     <nCategorical>\t<nInteger>\t<nFloat>
    //   # dictionaries    <startDictOffset>    <endDictOffset>
    //   then one line per tile: row col startOffset endOffset count
    void loadIndex(const std::string &indexFilename) override;

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
    void readDictionaries();
};

// Helper for multisample pipelien
struct dataset {
    std::string sampleId;
    std::string inTsv;
    std::string inIndex;
    std::string outPref;
    std::string anchorFile;
};

std::vector<dataset> parseSampleList(const std::string& sampleList, const std::string* outPref = nullptr);
