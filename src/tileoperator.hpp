#pragma once

#include <limits>
#include <map>
#include <tuple>
#include <utility>
#include <algorithm>
#include "utils.h"
#include "utils_sys.hpp"
#include "json.hpp"
#include "tile_io.hpp"
#include "tilereader.hpp"
#include "hexgrid.h"

struct SparseObsDict {
    double totalCount = 0;
    std::unordered_map<int32_t, double> featureCounts;
    void add(const SparseObsDict& other) {
        totalCount += other.totalCount;
        for (const auto& kv : other.featureCounts) {
            featureCounts[kv.first] += kv.second;
        }
    }
    void add(int32_t idx, double count = 1) {
        totalCount += count;
        featureCounts[idx] += count;
    }
};

class TileOperator {

public:
    using PixelKey3 = std::tuple<int32_t, int32_t, int32_t>;
    TileOperator(const std::string& dataFile, std::string indexFile = "", std::string headerFile = "") : dataFile_(dataFile), indexFile_(indexFile) {
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
    TileOperator(const TileOperator&) = delete;
    TileOperator& operator=(const TileOperator&) = delete;
    TileOperator(TileOperator&&) noexcept = default;
    TileOperator& operator=(TileOperator&&) noexcept = default;

    int32_t getK() const { return k_; }
    int32_t getTileSize() const { return formatInfo_.tileSize; }
    float getPixelResolution() const { return formatInfo_.pixelResolution; }
    const std::vector<TileInfo>& getTileInfo() const { return blocks_; }

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
        if ((mode_ & 0x1) == 0) {
            size_t pos = dataStream_.tellg();
            std::string line;
            while (std::getline(dataStream_, line)) {
                if (!(line.empty() || line[0] == '#')) {
                    break;
                }
                pos = dataStream_.tellg();
            }
            dataStream_.seekg(pos);
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

    // Return -1 for EOF, 0 for parse error, 1 for success
    int32_t next(PixTopProbs<float>& out, bool rawCoord = false);
    int32_t next(PixTopProbs<int32_t>& out);
    int32_t next(PixTopProbs3D<float>& out, bool rawCoord = false);
    int32_t next(PixTopProbs3D<int32_t>& out);

    void printIndex() const;

    void reorgTiles(const std::string& outPrefix, int32_t tileSize = -1);

    void merge(const std::vector<std::string>& otherFiles, const std::string& outPrefix, std::vector<uint32_t> k2keep = {}, bool binaryOutput = false);
    void annotate(const std::string& ptPrefix, const std::string& outPrefix, uint32_t icol_x, uint32_t icol_y, uint32_t icol_z = std::numeric_limits<uint32_t>::max());

    void dumpTSV(const std::string& outPrefix = "",  int32_t probDigits = 4, int32_t coordDigits = 2);

    // For each pair of (k1,k2) compute \sum_i p1_i * p2_i
    void probDot(const std::string& outPrefix, int32_t probDigits = 4);

    void probDot_multi(const std::vector<std::string>& otherFiles, const std::string& outPrefix, std::vector<uint32_t> k2keep = {}, int32_t probDigits = 4);

    using Slice = std::unordered_map<std::pair<int32_t, int32_t>, SparseObsDict, PairHash>; // unitKey -> sparse feature counts
    std::unordered_map<int32_t, Slice> aggOneTile(TileReader& reader, lineParserUnival& parser, TileKey tile, double gridSize, double minProb = 0.01) const;

private:
    std::string dataFile_, indexFile_;
    std::ifstream dataStream_;
    std::string headerLine_;
    uint32_t icol_x_, icol_y_, icol_z_, icol_max_ = 0;
    bool has_z_ = false;
    uint32_t coord_dim_ = 2;
    std::vector<uint32_t> icol_ks_, icol_ps_;
    int32_t k_ = 0;
    std::vector<uint32_t> kvec_;
    uint32_t mode_ = 0;
    IndexHeader formatInfo_;
    std::vector<TileInfo> blocks_all_, blocks_;
    int32_t idx_block_;
    bool bounded_ = false;
    bool done_ = false;
    Rectangle<float> queryBox_;
    Rectangle<float> globalBox_;
    uint64_t pos_;
    std::unordered_map<TileKey, size_t, TileKeyHash> tile_lookup_;

    // Determine if a block is strictly within a tile or a boundary block
    void classifyBlocks(int32_t tileSize);
    // Parse header from data file
    void parseHeaderLine();
    // Parse header from json file
    void parseHeaderFile(const std::string& headerFile);
    // Load index
    void loadIndex(const std::string& indexFile);
    void loadIndexLegacy(const std::string& indexFile);
    // Jump to the beginning of a block
    void openBlock(TileInfo& blk);
    // Get the next record within the bounded query region
    int32_t nextBounded(PixTopProbs<float>& out, bool rawCoord = false);
    int32_t nextBounded(PixTopProbs<int32_t>& out);
    int32_t nextBounded(PixTopProbs3D<float>& out, bool rawCoord = false);
    int32_t nextBounded(PixTopProbs3D<int32_t>& out);
    // Parse a line to extract factor results
    bool parseLine(const std::string& line, PixTopProbs<float>& R) const;
    bool parseLine(const std::string& line, PixTopProbs<int32_t>& R) const;
    bool parseLine(const std::string& line, PixTopProbs3D<float>& R) const;
    bool parseLine(const std::string& line, PixTopProbs3D<int32_t>& R) const;

    void reorgTilesBinary(const std::string& outPrefix, int32_t tileSize = -1);

    int32_t loadTileToMap(const TileKey& key,
        std::map<std::pair<int32_t, int32_t>, TopProbs>& pixelMap) const;
    int32_t loadTileToMap3D(const TileKey& key,
        std::map<PixelKey3, TopProbs>& pixelMap) const;
    void mergeTiles2D(const std::set<TileKey>& commonTiles,
        const std::vector<TileOperator*>& opPtrs,
        const std::vector<uint32_t>& k2keep,
        bool binaryOutput,
        FILE* fp, int fdMain, int fdIndex,
        long& currentOffset);
    void mergeTiles3D(const std::set<TileKey>& commonTiles,
        const std::vector<TileOperator*>& opPtrs,
        const std::vector<uint32_t>& k2keep,
        bool binaryOutput,
        FILE* fp, int fdMain, int fdIndex,
        long& currentOffset);
    void probDotTiles2D(const std::set<TileKey>& commonTiles,
        const std::vector<TileOperator*>& opPtrs,
        const std::vector<uint32_t>& k2keep,
        const std::vector<uint32_t>& offsets,
        std::vector<std::map<int32_t, double>>& marginals,
        std::vector<std::map<std::pair<int32_t, int32_t>, double>>& internalDots,
        std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>>& crossDots,
        size_t& count);
    void probDotTiles3D(const std::set<TileKey>& commonTiles,
        const std::vector<TileOperator*>& opPtrs,
        const std::vector<uint32_t>& k2keep,
        const std::vector<uint32_t>& offsets,
        std::vector<std::map<int32_t, double>>& marginals,
        std::vector<std::map<std::pair<int32_t, int32_t>, double>>& internalDots,
        std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>>& crossDots,
        size_t& count);
    void annotateTiles2D(const std::vector<TileKey>& tiles,
        TileReader& reader, uint32_t icol_x, uint32_t icol_y,
        uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset);
    void annotateTiles3D(const std::vector<TileKey>& tiles,
        TileReader& reader, uint32_t icol_x, uint32_t icol_y, uint32_t icol_z,
        uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset);


};
