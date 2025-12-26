#include "utils.h"
#include "json.hpp"
#include "tilereader.hpp"

struct PixelFactorResult {
    double x, y;
    std::vector<int32_t> ks;
    std::vector<float> ps;
};

class TileOperator {

public:
    TileOperator(std::string& dataFile, std::string indexFile = "", std::string headerFile = "") : dataFile_(dataFile), indexFile_(indexFile) {
        if (!indexFile.empty()) {
            loadIndex(indexFile);
        }
        if (!headerFile.empty()) {
            parseHeaderFile(headerFile);
        } else {
            parseHeaderLine();
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

    int32_t query(double qxmin, double qxmax, double qymin, double qymax);

    int32_t next(PixelFactorResult& out);

    void printIndex() const;

    void reorgTiles(const std::string& outPrefix, int32_t tileSize_);

private:
    std::string dataFile_, indexFile_;
    std::ifstream dataStream_;
    std::string headerLine_;
    uint32_t icol_x_, icol_y_, icol_max_ = 0;
    std::vector<uint32_t> icol_ks_, icol_ps_;
    int32_t k_ = 0;
    std::vector<Block> blocks_all_, blocks_;
    int32_t idx_block_;
    bool bounded_ = false;
    bool done_ = false;
    double qxmin_, qxmax_, qymin_, qymax_;
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
    int32_t nextBounded(PixelFactorResult& out);
    // Parse a line into PixelFactorResult
    bool parseLine(const std::string& line, PixelFactorResult& R) const;

};
