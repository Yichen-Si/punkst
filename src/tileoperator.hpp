#pragma once

#include <limits>
#include <map>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <algorithm>
#include <cmath>
#include "utils.h"
#include "utils_sys.hpp"
#include "json.hpp"
#include "tile_io.hpp"
#include "tilereader.hpp"
#include "hexgrid.h"

struct PreparedRegionMask2D;
struct PreparedRegionRasterMask2D;
enum class RegionPixelState : uint8_t;

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
    struct GzCloser {
        void operator()(gzFile_s* fp) const noexcept {
            if (fp) {
                gzclose(fp);
            }
        }
    };
    using GzHandle = std::unique_ptr<gzFile_s, GzCloser>;
    using PixelKey3 = std::tuple<int32_t, int32_t, int32_t>;
    TileOperator(const std::string& dataFile, std::string indexFile = "", std::string headerFile = "", int32_t threads = 1) : dataFile_(dataFile), indexFile_(indexFile), threads_(std::max(0, threads)) {
        if (!indexFile.empty()) {
            loadIndex(indexFile);
        }
        if ((mode_ & 0x1) == 0) {
            if (!headerFile.empty()) {
                warning("%s: --header is ignored for TSV input; parsing header line from TSV file", __func__);
            }
            parseHeaderLine();
        }
    }
    ~TileOperator() {
        closeTextStream();
    }
    TileOperator(const TileOperator&) = delete;
    TileOperator& operator=(const TileOperator&) = delete;
    TileOperator(TileOperator&&) noexcept = default;
    TileOperator& operator=(TileOperator&&) noexcept = default;

    int32_t getK() const { return k_; }
    int32_t getTileSize() const { return formatInfo_.tileSize; }
    float getPixelResolution() const { return formatInfo_.pixelResolution; }
    float getPixelResolutionZ() const {
        if ((mode_ & 0x10) && (mode_ & 0x20u) && formatInfo_.pixelResolutionZ > 0.0f) {
            return formatInfo_.pixelResolutionZ;
        }
        return formatInfo_.pixelResolution;
    }
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
    template<typename T>
    int32_t coord2pix(T v) const {
        if (formatInfo_.pixelResolution > 0)
            return static_cast<int32_t>(static_cast<float>(v) / formatInfo_.pixelResolution);
        return static_cast<int32_t>(v);
    }
    void setThreads(int32_t threads) {
        threads_ = std::max(0, threads);
    }
    void setCoordinateColumns(int32_t icol_x, int32_t icol_y) {
        icol_x_ = icol_x;
        icol_y_ = icol_y;
        icol_max_ = std::max(icol_max_, std::max(icol_x_, icol_y_));
    }
    void setFactorCount(int32_t K) {K_ = K;}

    void openDataStream();

    void resetReader() {
        if ((mode_ & 0x1) == 0 && isTextStdinInput()) {
            error("%s: Cannot reset reader for stdin stream input", __func__);
        }

        if ((mode_ & 0x1) == 0 && isTextGzipInput()) {
            closeTextStream();
            hasPendingTextLine_ = false;
            openDataStream();
        } else {
            if (dataStream_.is_open()) {
                dataStream_.clear();
                dataStream_.seekg(0);
            } else {
                openDataStream();
            }
            hasPendingTextLine_ = false;
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
    // w/ dimension (2<->3) conversion & w/o coord scaling
    int32_t next(PixTopProbs<int32_t>& out);
    int32_t next(PixTopProbs3D<int32_t>& out);
    // (if rawCoord = false) int (pixel) ->float (world) conversion
    int32_t next(PixTopProbs<float>& out, bool rawCoord = false);
    int32_t next(PixTopProbs3D<float>& out, bool rawCoord = false);

    void printIndex() const;

    void extractRegion(const std::string& outPrefix, float qxmin, float qxmax, float qymin, float qymax);
    void extractRegionGeoJSON(const std::string& outPrefix, const std::string& geojsonFile, int64_t scale = 10);
    void extractRegionPrepared(const std::string& outPrefix, const PreparedRegionMask2D& region);

    void reorgTiles(const std::string& outPrefix, int32_t tileSize = -1);

    void smoothTopLabels2D(const std::string& outPrefix, int32_t islandSmoothRounds = 1, bool fillEmptyIslands = false);

    void spatialMetricsBasic(const std::string& outPrefix);
    void connectedComponents(const std::string& outPrefix, uint32_t minSize = 10);
    void profileShellAndSurface(const std::string& outPrefix,
        const std::vector<int32_t>& radii, int32_t dMax,
        uint32_t minCompSize = 10, uint32_t minPixPerTilePerLabel = 0);

    void merge(const std::vector<std::string>& otherFiles, const std::string& outPrefix, std::vector<uint32_t> k2keep = {}, bool binaryOutput = false);
    void annotate(const std::string& ptPrefix, const std::string& outPrefix, uint32_t icol_x, uint32_t icol_y, int32_t icol_z = -1);

    void pix2cell(const std::string& ptPrefix, const std::string& outPrefix,
        uint32_t icol_c, uint32_t icol_x, uint32_t icol_y,
        int32_t icol_s = -1, int32_t icol_z = -1, uint32_t k_out = 0, float max_cell_diameter = 50);

    void dumpTSV(const std::string& outPrefix = "",  int32_t probDigits = 4, int32_t coordDigits = 2);

    // For each pair of (k1,k2) compute \sum_i p1_i * p2_i
    void probDot(const std::string& outPrefix, int32_t probDigits = 4);
    void probDot_multi(const std::vector<std::string>& otherFiles, const std::string& outPrefix, std::vector<uint32_t> k2keep = {}, int32_t probDigits = 4);

    int32_t loadTileToMap(const TileKey& key,
        std::map<std::pair<int32_t, int32_t>, TopProbs>& pixelMap,
        const std::vector<Rectangle<float>>* rects = nullptr) const;
    int32_t loadTileToMap3D(const TileKey& key,
        std::map<PixelKey3, TopProbs>& pixelMap) const;

    using Slice = std::unordered_map<std::pair<int32_t, int32_t>, SparseObsDict, PairHash>; // unitKey -> sparse feature counts
    std::unordered_map<int32_t, Slice> aggOneTile(
        std::map<std::pair<int32_t, int32_t>, TopProbs>& pixelMap,
        TileReader& reader, lineParserUnival& parser, TileKey tile, double gridSize, double minProb = 0.01, int32_t union_key = 0) const;
    std::unordered_map<int32_t, Slice> aggOneTileRegion(
        const std::map<std::pair<int32_t, int32_t>, TopProbs>& pixelMap,
        const std::unordered_map<std::pair<int32_t, int32_t>, RegionPixelState, PairHash>& pixelState,
        TileReader& reader, lineParserUnival& parser, TileKey tile,
        const PreparedRegionMask2D& region, double gridSize, double minProb = 0.01,
        int32_t union_key = 0, Eigen::MatrixXd* confusion = nullptr,
        double* residualAccum = nullptr) const;

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        computeConfusionMatrix(double resolution, const char* outPref = nullptr, int32_t probDigits = 4) const;

    void sampleTilesToDebug(int32_t ntiles = 1);

private:
    std::string dataFile_, indexFile_;
    std::ifstream dataStream_;
    GzHandle gzDataStream_ = nullptr;
    bool textStreamOpen_ = false;
    bool hasPendingTextLine_ = false;
    std::string pendingTextLine_;
    std::string headerLine_;
    uint32_t icol_x_, icol_y_, icol_z_, icol_max_ = 0;
    bool has_z_ = false;
    uint32_t coord_dim_ = 2;
    std::vector<uint32_t> icol_ks_, icol_ps_;
    int32_t k_ = 0, K_ = 0;
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
    int32_t threads_ = 1;
    bool regular_labeled_raster_ = false;

    bool isTextInput() const { return ((mode_ & 0x1) == 0); }
    bool isTextStdinInput() const { return dataFile_ == "-" || dataFile_ == "/dev/stdin"; }
    bool isTextGzipInput() const { return ends_with(dataFile_, ".gz"); }
    bool isStreamingTextInput() const { return isTextInput() && (isTextStdinInput() || isTextGzipInput()); }
    bool canSeekTextInput() const { return isTextInput() && !isStreamingTextInput(); }
    bool storesIntegerCoordinates() const { return (mode_ & 0x4) != 0; }
    bool storesFloatCoordinates() const { return (mode_ & 0x4) == 0; }
    bool rawCoordinatesArePixels() const { return (mode_ & 0x2) != 0; }
    void validateCoordinateEncoding() const;
    void closeTextStream() {
        gzDataStream_.reset();
        if (dataStream_.is_open()) {
            dataStream_.close();
        }
        textStreamOpen_ = false;
    }
    bool readNextTextLine(std::string& line);

    // Determine if a block is strictly within a tile or a boundary block
    void classifyBlocks(int32_t tileSize);
    // Parse header from data file
    void parseHeaderLine();
    // Load index
    void loadIndex(const std::string& indexFile);
    void loadIndexLegacy(const std::string& indexFile);
    // Jump to the beginning of a block
    void openBlock(TileInfo& blk);

    /* Helpers for parsing a record */
    // Get the next record within the bounded query region
    int32_t nextBounded(PixTopProbs<float>& out, bool rawCoord = false);
    int32_t nextBounded(PixTopProbs<int32_t>& out);
    int32_t nextBounded(PixTopProbs3D<float>& out, bool rawCoord = false);
    int32_t nextBounded(PixTopProbs3D<int32_t>& out);
    // Parse one text line (without coordinate scaling)
    bool parseLine(const std::string& line, PixTopProbs<float>& R) const;
    bool parseLine(const std::string& line, PixTopProbs<int32_t>& R) const;
    bool parseLine(const std::string& line, PixTopProbs3D<float>& R) const;
    bool parseLine(const std::string& line, PixTopProbs3D<int32_t>& R) const;
    // Wrapper of parseLine, w/ dimension (2<->3) & w/o coord scaling
    bool decodeTextRecord2DInt(const std::string& line, PixTopProbs<int32_t>& out) const;
    bool decodeTextRecord3DInt(const std::string& line, PixTopProbs3D<int32_t>& out) const;
    // Wrapper of parseLine, with dimension (2<->3) and
    //       (if rawCoord = false) int (pixel) ->float (world) conversion
    bool decodeTextRecord2D(const std::string& line, PixTopProbs<float>& out,
        bool rawCoord = false) const;
    bool decodeTextRecord3D(const std::string& line, PixTopProbs3D<float>& out,
        bool rawCoord = false) const;
    // Read one binary record, w/ dimension (2<->3) & w/o coord scaling
    bool readBinaryRecord2DInt(std::istream& dataStream, PixTopProbs<int32_t>& out) const;
    bool readBinaryRecord3DInt(std::istream& dataStream, PixTopProbs3D<int32_t>& out) const;
    // Read one binary record, with dimension (2<->3) and
    //        (if rawCoord = false) int (pixel) ->float (world) conversion
    bool readBinaryRecord2D(std::istream& dataStream, PixTopProbs<float>& out,
        bool rawCoord = false) const;
    bool readBinaryRecord3D(std::istream& dataStream, PixTopProbs3D<float>& out,
        bool rawCoord = false) const;
    // Read one record and convert coordinates to integer pixel space.
    bool readNextRecord2DAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos, int32_t& recX, int32_t& recY, TopProbs& rec) const;
    bool readNextRecord3DAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos, int32_t& recX, int32_t& recY, int32_t& recZ, TopProbs& rec) const;
    // Parse only 2 coordinates with int->float conversion
    // (store world coordinates regardless of the input data type)
    void decodeBinaryXY(const char* recBuf, float& x, float& y) const;

    // Impl for reorgTiles
    void reorgTilesBinary(const std::string& outPrefix, int32_t tileSize = -1);

    // Impl for merge
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
    // Impl for probDot
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
    // Impl for annotate
    void annotateTiles2D(const std::vector<TileKey>& tiles,
        TileReader& reader, uint32_t icol_x, uint32_t icol_y,
        uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset);
    void annotateTiles3D(const std::vector<TileKey>& tiles,
        TileReader& reader, uint32_t icol_x, uint32_t icol_y, uint32_t icol_z,
        uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset);

    struct DenseTile {
        TileKey key;
        int32_t pixX0 = 0;
        int32_t pixX1 = 0;
        int32_t pixY0 = 0;
        int32_t pixY1 = 0;
        size_t W = 0;
        size_t H = 0;
        std::vector<uint8_t> lab;
        std::vector<uint8_t> boundary;
    };

    struct PixBox {
        int32_t minX = std::numeric_limits<int32_t>::max();
        int32_t maxX = std::numeric_limits<int32_t>::lowest();
        int32_t minY = std::numeric_limits<int32_t>::max();
        int32_t maxY = std::numeric_limits<int32_t>::lowest();

        void include(int32_t x, int32_t y) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        void include(const PixBox& other) {
            if (other.minX < minX) minX = other.minX;
            if (other.maxX > maxX) maxX = other.maxX;
            if (other.minY < minY) minY = other.minY;
            if (other.maxY > maxY) maxY = other.maxY;
        }
    };

    struct TileCCL {
        int32_t pixX0 = 0;
        int32_t pixX1 = 0;
        int32_t pixY0 = 0;
        int32_t pixY1 = 0;
        uint32_t ncomp = 0;
        std::vector<uint32_t> compSize;
        std::vector<uint8_t> compLabel;
        std::vector<uint32_t> leftCid;
        std::vector<uint32_t> rightCid;
        std::vector<uint32_t> topCid;
        std::vector<uint32_t> bottomCid;
        std::vector<uint64_t> compSumX;
        std::vector<uint64_t> compSumY;
        std::vector<PixBox> compBox;
        std::vector<uint32_t> bndPix;
        std::vector<uint32_t> bndCid;
    };

    struct BorderRemapInfo {
        std::vector<uint32_t> remap;
        std::vector<uint32_t> oldCompSize;
        std::vector<uint8_t> oldCompLabel;
        std::vector<uint64_t> oldCompSumX;
        std::vector<uint64_t> oldCompSumY;
        std::vector<PixBox> oldCompBox;
    };

    struct BorderDSUState {
        std::vector<size_t> globalBase;
        std::vector<uint64_t> rootSize;
        std::vector<uint8_t> rootLabel;
        std::vector<uint64_t> rootSumX;
        std::vector<uint64_t> rootSumY;
        std::vector<PixBox> rootBox;
        std::vector<std::vector<size_t>> tileRoot;
    };

    struct DisjointSet {
        std::vector<size_t> parent;
        std::vector<uint8_t> rankv;

        explicit DisjointSet(size_t n) : parent(n), rankv(n, 0) {
            std::iota(parent.begin(), parent.end(), static_cast<size_t>(0));
        }

        size_t find(size_t x) {
            while (parent[x] != x) {
                parent[x] = parent[parent[x]];
                x = parent[x];
            }
            return x;
        }

        void unite(size_t a, size_t b) {
            size_t ra = find(a);
            size_t rb = find(b);
            if (ra == rb) return;
            if (rankv[ra] < rankv[rb]) std::swap(ra, rb);
            parent[rb] = ra;
            if (rankv[ra] == rankv[rb]) rankv[ra]++;
        }

        void compress_all() {
            for (size_t i = 0; i < parent.size(); ++i) parent[i] = find(i);
        }
    };

    struct PixelRef {
        uint32_t tileIdx = 0;
        uint32_t localIdx = 0;
    };

    void loadDenseTile(const TileInfo& blk, std::ifstream& in, DenseTile& out, uint8_t bg, uint64_t& nOutOfRangeIgnored, uint64_t& nBadLabelIgnored) const;
    // Connected component labeling within a tile
    TileCCL tileLocalCCL(const DenseTile& tile, uint8_t bg, const std::vector<uint8_t>* boundaryMask = nullptr) const;
    // Mask boundary pixels: iff any of its 4 neighbors have a different label
    void computeTileBoundaryMasks(std::vector<DenseTile>& tiles) const;
    // Detect border components and remap them to global IDs
    BorderRemapInfo remapTileToBorderComponents(TileCCL& t, uint32_t invalid) const;
    BorderDSUState mergeBorderComponentsWithDSU(const std::vector<TileCCL>& perTile, uint8_t bg, uint32_t invalid) const;

};
