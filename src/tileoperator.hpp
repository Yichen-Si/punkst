#pragma once

#include <limits>
#include <map>
#include <unordered_map>
#include <tuple>
#include <utility>
#include <algorithm>
#include <cmath>
#include <string>
#include "clipper2/clipper.h"
#include "utils.h"
#include "utils_sys.hpp"
#include "json.hpp"
#include "tile_io.hpp"
#include "tilereader.hpp"
#include "hexgrid.h"
#include "gene_bin_utils.hpp"

struct PreparedRegionMask2D;
struct PreparedRegionRasterMask2D;
enum class RegionPixelState : uint8_t;
namespace tileoperator_detail { namespace feature { struct FeatureRemapPlan; } }

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
    using PixelFeatureKey2 = std::tuple<int32_t, int32_t, uint32_t>;
    using PixelFeatureKey3 = std::tuple<int32_t, int32_t, int32_t, uint32_t>;

    struct MltPmtilesOptions {
        bool enabled = false;
        int32_t icol_count = -1;
        std::string gene_bin_info_file;
        std::string feature_count_file;
        std::vector<std::string> ext_col_ints;
        std::vector<std::string> ext_col_floats;
        std::vector<std::string> ext_col_strs;
        double coordScale = -1.0;
        double encode_prob_min = 1e-4;
        double encode_prob_eps = 1e-6;
        int32_t n_gene_bins = 0;
        int32_t zoom = -1;
    };

    struct ExportPmtilesOptions {
        int32_t tileSize = -1;
        int32_t probDigits = 4;
        int32_t coordDigits = 2;
        std::string geojsonFile;
        int64_t geojsonScale = 10;
        float xmin = 0.0f;
        float xmax = -1.0f;
        float ymin = 0.0f;
        float ymax = -1.0f;
        float zmin = std::numeric_limits<float>::quiet_NaN();
        float zmax = std::numeric_limits<float>::quiet_NaN();
    };

    struct BuildPmtilesPyramidOptions {
        int32_t minZoom = 0;
        int32_t maxTileBytes = 500000;
        int32_t maxTileFeatures = 50000;
        double scaleFactorCompression = 10.0;
        int32_t threads = 1;
    };

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

    /* Small get/set helpers */
    int32_t getK() const { return k_; }
    std::vector<uint32_t> getKvec() const { return kvec_; }
    int32_t getTileSize() const { return formatInfo_.tileSize; }
    float getPixelResolution() const { return formatInfo_.pixelResolution; }
    bool hasFeatureIndex() const { return (mode_ & 0x40u) != 0; }
    const std::vector<std::string>& getFeatureNames() const { return featureNames_; }
    std::vector<std::string> getHeaderColumns() const;
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
    void setNullPlaceholders(const std::string& nullK, const std::string& nullP) {
        if (nullK.empty() || nullP.empty()) {
            error("%s: null placeholders must be non-empty", __func__);
        }
        nullK_ = nullK;
        nullP_ = nullP;
    }
    void setSuppressKpParseWarnings(bool suppress) {
        suppressKpParseWarnings_ = suppress;
    }
    void setPixelResolutionOverride(float resXY, float resZ = -1.0f);
    void setRasterPixelResolution(float resXY);
    void setCoordinateColumns(int32_t icol_x, int32_t icol_y) {
        icol_x_ = icol_x;
        icol_y_ = icol_y;
        icol_max_ = std::max(icol_max_, std::max(icol_x_, icol_y_));
    }
    void setFactorCount(int32_t K) {K_ = K;}
    void sampleTilesToDebug(int32_t ntiles = 1);

    /* Shared datastream */
    void openDataStream();
    void resetReader();

    /* Streaming */
    // Return -1 for EOF, 0 for parse error, 1 for success
    // w/ dimension (2<->3) conversion & w/o coord scaling
    int32_t next(PixTopProbs<int32_t>& out);
    int32_t next(PixTopProbs3D<int32_t>& out);
    // (if rawCoord = false) int (pixel) ->float (world) conversion
    int32_t next(PixTopProbs<float>& out, bool rawCoord = false);
    int32_t next(PixTopProbs3D<float>& out, bool rawCoord = false);

    /* Basics */
    // Print index
    void printIndex() const;
    // Convert to plain TSV
    void dumpTSV(const std::string& outPrefix = "",
        int32_t probDigits = 4, int32_t coordDigits = 2,
        const std::string& geojsonFile = "", int64_t geojsonScale = 10,
        float qzmin = std::numeric_limits<float>::quiet_NaN(),
        float qzmax = std::numeric_limits<float>::quiet_NaN(),
        const std::vector<std::string>& mergePrefixes = {});
    static void exportPMTiles(const std::string& pmtilesFile,
        const std::string& outPrefix,
        const ExportPmtilesOptions& options);
    static void buildPmtilesPyramid(const std::string& inPmtiles,
        const std::string& outPmtiles,
        const BuildPmtilesPyramidOptions& options);
    void writeMltPmtiles(const std::string& outPrefix,
        const MltPmtilesOptions& mltOptions,
        std::vector<uint32_t> k2keep = {},
        const std::vector<std::string>& mergePrefixes = {});
    // Fix Fragmented Tiles
    void reorgTiles(const std::string& outPrefix, int32_t tileSize = -1);
    // Region query
    void extractRegion(const std::string& outPrefix, float qxmin, float qxmax, float qymin, float qymax,
        float qzmin = std::numeric_limits<float>::quiet_NaN(),
        float qzmax = std::numeric_limits<float>::quiet_NaN());
    void extractRegionGeoJSON(const std::string& outPrefix, const std::string& geojsonFile, int64_t scale = 10,
        float qzmin = std::numeric_limits<float>::quiet_NaN(),
        float qzmax = std::numeric_limits<float>::quiet_NaN());
    void extractRegionPrepared(const std::string& outPrefix, const PreparedRegionMask2D& region,
        float qzmin = std::numeric_limits<float>::quiet_NaN(),
        float qzmax = std::numeric_limits<float>::quiet_NaN());
    int32_t query(float qxmin, float qxmax, float qymin, float qymax);

    /* Joining, annotating, and aggregation */
    void merge(const std::vector<std::string>& otherFiles,
        const std::string& outPrefix, std::vector<uint32_t> k2keep = {},
        bool binaryOutput = false, bool keepAllMain = false, bool keepAll = false, const std::vector<std::string>& mergePrefixes = {});
    void annotate(const std::string& ptPrefix, const std::string& outPrefix,
        int32_t icol_x, int32_t icol_y, int32_t icol_z,
        int32_t icol_f, bool annoKeepAll,
        const std::vector<std::string>& mergePrefixes,
        const MltPmtilesOptions& mltOptions);
    void annotateMerged(const std::vector<std::string>& otherFiles,
        const std::string& ptPrefix, const std::string& outPrefix,
        std::vector<uint32_t> k2keep, int32_t icol_x, int32_t icol_y,
        int32_t icol_z, int32_t icol_f,
        bool keepAllMain, bool keepAll,
        const std::vector<std::string>& mergePrefixes,
        bool annoKeepAll,
        const MltPmtilesOptions& mltOptions);
    void pix2cell(const std::string& ptPrefix, const std::string& outPrefix,
        uint32_t icol_c, uint32_t icol_x, uint32_t icol_y,
        int32_t icol_s = -1, int32_t icol_z = -1, int32_t icol_f = -1,
        uint32_t k_out = 0, float max_cell_diameter = 50);

    /* Factor-distribution summaries */
    // For each pair of (k1,k2) compute \sum_i p1_i * p2_i
    void probDot(const std::string& outPrefix, int32_t probDigits = 4);
    void probDot_multi(const std::vector<std::string>& otherFiles, const std::string& outPrefix, std::vector<uint32_t> k2keep = {}, int32_t probDigits = 4);
    // Compute confusion matrix among factors given a resolution
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        computeConfusionMatrix(double resolution, const char* outPref = nullptr, int32_t probDigits = 4) const;

    /* Spatial profiling and factor masks */
    // Denoise top labels
    void smoothTopLabels2D(const std::string& outPrefix, int32_t islandSmoothRounds = 1, bool fillEmptyIslands = false);
    // Compute basic spatial metrics
    void spatialMetricsBasic(const std::string& outPrefix);
    // Profile shell occupancy and surface distance
    void profileShellAndSurface(const std::string& outPrefix,
        const std::vector<int32_t>& radii, int32_t dMax,
        uint32_t minComponentSize = 10, uint32_t minPixPerTilePerLabel = 0);
    // Profile the soft mask of a focal factor
    void profileSoftFactorMasks(const std::string& outPrefix,
        int32_t focalK, int32_t radius,
        double neighborhoodThresholdFrac, double minFactorFrac,
        float minPixelProb = 0.01f,
        const std::vector<int32_t>& morphologySteps = {},
        uint32_t minComponentArea = 5, bool skipMaskOverlap = false);
    // Compute hard and soft masks for all factors
    void hardFactorMask(const std::string& outPrefix,
        uint32_t minComponentSize = 1, bool skipBoundaries = false, const std::string& templateGeoJSON = "", const std::string& templateOutPrefix = "");
    void softFactorMask(const std::string& outPrefix,
        int32_t radius, double neighborhoodThreshold,
        float minPixelProb = 0.01f, const std::vector<int32_t>& morphologySteps = {}, double minTileFactorMass = 0.0,
        uint32_t minComponentArea = 5, uint32_t minHoleArea = 0,
        double simplifyTolerance = 0.0, bool skipBoundaries = false,
        const std::string& templateGeoJSON = "", const std::string& templateOutPrefix = "");
    void softMaskComposition(const std::string& outPrefix,
        const std::string& maskGeoJSON,
        const std::vector<int32_t>& focalFactors = {});

    /* Tile loaders */
    // Load raster tiles
    int32_t loadTileToMap(const TileKey& key,
        std::map<std::pair<int32_t, int32_t>, TopProbs>& pixelMap,
        const std::vector<Rectangle<float>>* rects = nullptr,
        std::ifstream* dataStream = nullptr) const;
    int32_t loadTileToMap3D(const TileKey& key,
        std::map<PixelKey3, TopProbs>& pixelMap,
        std::ifstream* dataStream = nullptr) const;
    // Aggregate (softly) observations by factor ("slice")
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

private:
    enum class MergeSourceRelation {
        Same2D,
        Same3D,
        Broadcast2DTo3D,
    };

    struct MergeSourcePlan {
        const TileOperator* op = nullptr;
        uint32_t keepK = 0;
        uint32_t srcDim = 2;
        float srcResXY = -1.0f;
        float srcResZ = -1.0f;
        int32_t tileSize = 0;
        int32_t ratioXY = 1;
        int32_t ratioZ = 1;
        MergeSourceRelation relation = MergeSourceRelation::Same2D;
    };

    struct MergedAnnotate2DCounts;

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
    std::vector<std::string> featureNames_;
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
    bool hasRasterResolutionOverride_ = false;
    float rasterPixelResolution_ = -1.0f;
    int32_t rasterRatioXY_ = 1;
    std::string nullK_ = "-1";
    std::string nullP_ = "0";
    bool suppressKpParseWarnings_ = false;

    bool isTextInput() const { return ((mode_ & 0x1) == 0); }
    bool isTextStdinInput() const { return dataFile_ == "-" || dataFile_ == "/dev/stdin"; }
    bool isTextGzipInput() const { return ends_with(dataFile_, ".gz"); }
    bool isStreamingTextInput() const { return isTextInput() && (isTextStdinInput() || isTextGzipInput()); }
    bool canSeekTextInput() const { return isTextInput() && !isStreamingTextInput(); }
    bool storesIntegerCoordinates() const { return (mode_ & 0x4) != 0; }
    bool storesFloatCoordinates() const { return (mode_ & 0x4) == 0; }
    bool rawCoordinatesAreScaled() const { return (mode_ & 0x2) != 0; }
    bool usesRasterResolutionOverride() const { return hasRasterResolutionOverride_; }
    float getRasterPixelResolution() const {
        return hasRasterResolutionOverride_ ? rasterPixelResolution_ : getPixelResolution();
    }
    static int32_t floorDivInt32(int32_t value, int32_t divisor);
    static int32_t ceilDivInt32(int32_t value, int32_t divisor);

    int32_t mapPixelToRasterFloor(int32_t value) const;
    int32_t mapPixelToRasterCeil(int32_t value) const;
    using RasterTopProbAccum = std::vector<std::unordered_map<int32_t, double>>;
    void accumulateRasterTopProbs(RasterTopProbAccum& accum, const TopProbs& rec) const;
    TopProbs finalizeRasterTopProbs(const RasterTopProbAccum& accum) const;
    void validateCoordinateEncoding() const;
    void syncActiveBounds();
    void closeTextStream();
    void requireNoFeatureIndex(const char* funcName) const;
    void applyUnsetSourceResolutionOverrides(
        const std::vector<TileOperator*>& opPtrs,
        const char* funcName) const;
    void writeIndexHeaderWithFeatureDict(int fdIndex, const IndexHeader& idxHeader) const;
    void buildPreparedRegionPlan(const PreparedRegionMask2D& region,
        std::vector<size_t>& activeOrder,
        std::vector<uint8_t>& activeStates) const;

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
    // Get the next line of text input
    bool readNextTextLine(std::string& line);
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
    bool readBinaryRecord2DInt(std::istream& dataStream, PixTopProbsFeature<int32_t>& out) const;
    bool readBinaryRecord3DInt(std::istream& dataStream, PixTopProbsFeature3D<int32_t>& out) const;
    // Read one binary record, with dimension (2<->3) and
    //        (if rawCoord = false) int (pixel) ->float (world) conversion
    bool readBinaryRecord2D(std::istream& dataStream, PixTopProbs<float>& out,
        bool rawCoord = false) const;
    bool readBinaryRecord3D(std::istream& dataStream, PixTopProbs3D<float>& out,
        bool rawCoord = false) const;
    bool readBinaryRecord2D(std::istream& dataStream, PixTopProbsFeature<float>& out,
        bool rawCoord = false) const;
    bool readBinaryRecord3D(std::istream& dataStream, PixTopProbsFeature3D<float>& out,
        bool rawCoord = false) const;
    // Read one record and convert coordinates to integer pixel space.
    bool readNextRecord2DAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos, int32_t& recX, int32_t& recY, TopProbs& rec) const;
    bool readNextRecord3DAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos, int32_t& recX, int32_t& recY, int32_t& recZ, TopProbs& rec) const;
    bool readNextRecord2DFeatureAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos,
        int32_t& recX, int32_t& recY, uint32_t& featureIdx, TopProbs& rec) const;
    bool readNextRecord3DFeatureAsPixel(std::istream& dataStream, uint64_t& pos, uint64_t endPos,
        int32_t& recX, int32_t& recY, int32_t& recZ, uint32_t& featureIdx, TopProbs& rec) const;
    // Parse only 2 coordinates with int->float conversion
    // (store world coordinates regardless of the input data type)
    void decodeBinaryXY(const char* recBuf, float& x, float& y) const;
    void decodeBinaryXYZ(const char* recBuf, float& x, float& y, float& z) const;

    /* Dispatched implementations */
    // Impl for reorgTiles
    void reorgTilesBinary(const std::string& outPrefix, int32_t tileSize = -1);
    // Impl for merge
    std::vector<MergeSourcePlan> validateMergeSources(
        const std::vector<TileOperator*>& opPtrs,
        const std::vector<uint32_t>& k2keep) const;
    void mergeTiles2D(const std::vector<TileKey>& mainTiles,
        const std::vector<MergeSourcePlan>& mergePlans,
        bool keepAllMain, bool keepAll,
        bool binaryOutput,
        FILE* fp, int fdMain, int fdIndex,
        long& currentOffset);
    void mergeTiles3D(const std::vector<TileKey>& mainTiles,
        const std::vector<MergeSourcePlan>& mergePlans,
        bool keepAllMain, bool keepAll,
        bool binaryOutput,
        FILE* fp, int fdMain, int fdIndex,
        long& currentOffset);
    void mergeSingleMolecule(const std::vector<std::string>& otherFiles,
        const std::string& outPrefix, std::vector<uint32_t> k2keep,
        bool binaryOutput, bool keepAllMain, bool keepAll,
        const std::vector<std::string>& mergePrefixes);
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
    void probDotMultiSingleMolecule(const std::vector<std::string>& otherFiles,
        const std::string& outPrefix, std::vector<uint32_t> k2keep, int32_t probDigits);
    // Impl for annotate
    void annotateTiles2D(const std::vector<TileKey>& tiles,
        TileReader& reader, uint32_t icol_x, uint32_t icol_y,
        uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset,
        bool annoKeepAll);
    void annotateTiles3D(const std::vector<TileKey>& tiles,
        TileReader& reader, uint32_t icol_x, uint32_t icol_y, uint32_t icol_z,
        uint32_t ntok, FILE* fp, int fdIndex, long& currentOffset,
        bool annoKeepAll);
    void annotateSingleMolecule(const std::string& ptPrefix,
        const std::string& outPrefix,
        int32_t icol_x, int32_t icol_y, int32_t icol_z,
        int32_t icol_f, bool annoKeepAll,
        const std::vector<std::string>& mergePrefixes,
        const MltPmtilesOptions& mltOptions);
    void annotateMergedSingleMolecule(const std::vector<std::string>& otherFiles,
        const std::string& ptPrefix, const std::string& outPrefix,
        std::vector<uint32_t> k2keep, int32_t icol_x, int32_t icol_y,
        int32_t icol_z, int32_t icol_f, bool keepAllMain, bool keepAll,
        const std::vector<std::string>& mergePrefixes,
        bool annoKeepAll,
        const MltPmtilesOptions& mltOptions);
    void annotateSingleMoleculeToMltPmtiles(const std::string& ptPrefix, const std::string& outPrefix,
        int32_t icol_x, int32_t icol_y, int32_t icol_z,
        int32_t icol_f, bool annoKeepAll,
        const std::vector<std::string>& mergePrefixes,
        const MltPmtilesOptions& mltOptions);
    void annotateMergedSingleMoleculeToMltPmtiles(const std::vector<std::string>& otherFiles,
        const std::string& ptPrefix, const std::string& outPrefix,
        std::vector<uint32_t> k2keep, int32_t icol_x, int32_t icol_y,
        int32_t icol_z, int32_t icol_f, bool keepAllMain, bool keepAll,
        const std::vector<std::string>& mergePrefixes, bool annoKeepAll,
        const MltPmtilesOptions& mltOptions);
    void annotatePlainToMltPmtiles(const std::string& ptPrefix, const std::string& outPrefix,
        int32_t icol_x, int32_t icol_y, int32_t icol_z,
        int32_t icol_f, bool annoKeepAll,
        const std::vector<std::string>& mergePrefixes,
        const MltPmtilesOptions& mltOptions);
    void annotateMergedPlainToMltPmtiles(const std::vector<std::string>& otherFiles,
        const std::string& ptPrefix, const std::string& outPrefix,
        std::vector<uint32_t> k2keep, int32_t icol_x, int32_t icol_y,
        int32_t icol_z, int32_t icol_f, bool keepAllMain, bool keepAll,
        const std::vector<std::string>& mergePrefixes, bool annoKeepAll,
        const MltPmtilesOptions& mltOptions);
    void appendTopProbsText(std::string& out, const TopProbs& probs) const;
    void pix2cellSingleMolecule(const std::string& ptPrefix,
        const std::string& outPrefix,
        uint32_t icol_c, uint32_t icol_x, uint32_t icol_y,
        int32_t icol_s, int32_t icol_z, int32_t icol_f,
        uint32_t k_out, float max_cell_diameter);
    void dumpTSVSingleMolecule(const std::string& outPrefix,
        int32_t probDigits, int32_t coordDigits,
        PreparedRegionMask2D* regionPtr = nullptr,
        float qzmin = std::numeric_limits<float>::quiet_NaN(),
        float qzmax = std::numeric_limits<float>::quiet_NaN(),
        const std::vector<std::string>& mergePrefixes = {});
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
        computeConfusionMatrixSingleMolecule(double resolution) const;
    int32_t loadTileToMapFeature(const TileKey& key,
        std::map<PixelFeatureKey2, TopProbs>& pixelMap,
        std::ifstream* dataStream = nullptr) const;
    int32_t loadTileToMapFeature3D(const TileKey& key,
        std::map<PixelFeatureKey3, TopProbs>& pixelMap,
        std::ifstream* dataStream = nullptr) const;
    std::vector<std::string> loadFeatureNames() const;

    /* Geometry & spatial related */
    struct TileGeom {
        TileKey key;
        int32_t pixX0 = 0;
        int32_t pixX1 = 0;
        int32_t pixY0 = 0;
        int32_t pixY1 = 0;
        size_t W = 0;
        size_t H = 0;
    };
    // Populate TileGeom from TileInfo
    void initTileGeom(const TileInfo& blk, TileGeom& out) const;

    struct DenseTile {
        TileGeom geom;
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
        std::vector<uint32_t> pixelCid;
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

    struct SoftMaskSparseRec {
        uint32_t localIdx = 0;
        TopProbs rec;
    };

    struct SoftMaskTileData {
        TileGeom geom;
        std::vector<uint8_t> seenLocal;
        std::unordered_map<int32_t, std::vector<std::pair<uint32_t, float>>> factorEntries;
        std::unordered_map<int32_t, double> factorMass;
        std::vector<SoftMaskSparseRec> records;
    };

    struct SoftMaskThresholdConfig {
        bool useObservedDenominator = false;
        bool applySparseEmptyCenterRule = false;
        double sparseEmptyCoverageFrac = 0.0;
        double minWindowMass = 0.0;
        std::vector<int32_t> morphologySteps;
    };

    struct BinaryMaskCCL {
        size_t W = 0;
        size_t H = 0;
        uint32_t ncomp = 0;
        uint64_t keptArea = 0;
        std::vector<uint32_t> pixelCid;
        std::vector<uint32_t> compArea;
        std::vector<PixBox> compBox;
        std::vector<uint8_t> compTouchesBorder;
        std::vector<uint32_t> leftCid;
        std::vector<uint32_t> rightCid;
        std::vector<uint32_t> topCid;
        std::vector<uint32_t> bottomCid;
    };

    struct SoftMaskTileFactorResult {
        int32_t factor = -1;
        uint64_t maskArea = 0;
        std::vector<uint32_t> interiorCompAreas;
        std::vector<uint32_t> borderCompAreas;
        std::vector<uint32_t> borderSourceCid;
        std::vector<PixBox> borderSourceBox;
        std::vector<Clipper2Lib::Paths64> interiorPolys;
        std::vector<Clipper2Lib::Paths64> borderPolys;
        std::vector<uint32_t> leftCid;
        std::vector<uint32_t> rightCid;
        std::vector<uint32_t> topCid;
        std::vector<uint32_t> bottomCid;
    };

    struct SoftMaskTileResult {
        TileGeom geom;
        std::vector<SoftMaskTileFactorResult> factors;
        std::unordered_map<int32_t, size_t> factorToIndex;
    };

    struct BorderComponentRef {
        size_t tileIdx = 0;
        size_t factorIdx = 0;
        size_t localCid = 0;
    };

    struct BorderMergeResult {
        std::vector<uint64_t> rootArea;
        std::vector<int32_t> rootFactor;
        std::vector<std::vector<BorderComponentRef>> rootMembers;
    };

    /* tile loader */
    // load all (factor, prob) entries
    void loadSoftMaskTileData(const TileInfo& blk, std::ifstream& in,
        float minPixelProb, bool keepRecords,
        bool keepFactorEntries, bool keepFactorMass, SoftMaskTileData& out,
        uint64_t& nOutOfRangeIgnored, uint64_t& nBadFactorIgnored,
        uint64_t* nCollisionIgnored = nullptr,
        std::vector<double>* histGlobal = nullptr) const;
    // store the top label of each pixel in a dense array
    //     (Populate DenseTile::lab but not DenseTile::boundary)
    void loadDenseTile(const TileInfo& blk, std::ifstream& in,
        DenseTile& out, uint8_t bg,
        uint64_t& nOutOfRangeIgnored, uint64_t& nBadLabelIgnored) const;

    /* Operate on single channel intensity */
    // build a dense mask from a sparse {(idx, prob)} for one channel
    void buildDenseFactorRaster(const std::vector<std::pair<uint32_t, float>>& entries, size_t nPix, std::vector<float>& dense) const;
    // build a binary mask based on local density
    void buildSoftMaskBinary(const std::vector<float>& dense,
        const std::vector<uint8_t>& seenLocal,
        size_t W, size_t H, int32_t radius, double neighborhoodThreshold,
        const SoftMaskThresholdConfig& config, std::vector<uint8_t>& mask,
        std::vector<float>* filteredOut = nullptr,
        std::vector<float>* observedFilteredOut = nullptr) const;
    void applyBinaryMorphologyStep(std::vector<uint8_t>& mask, size_t W, size_t H, int32_t step) const;

    /* Operate on a single channel binary mask */
    // filter out small CC
    //     modify mask in-place and return the total area of the filtered mask
    uint64_t filterMaskByMinComponentArea4(std::vector<uint8_t>& mask, size_t W, size_t H, uint32_t minComponentArea) const;
    // compute CC and border info
    BinaryMaskCCL buildBinaryMaskCCL4(std::vector<uint8_t>& mask, const std::vector<float>& mass, size_t W, size_t H, double minComponentMass) const;

    /* Operate on a CC mask */
    // compute the bounadry as a path
    Clipper2Lib::Paths64 buildMaskComponentRuns(const BinaryMaskCCL& ccl, uint32_t cid, const TileGeom& geom) const;
    Clipper2Lib::Paths64 buildMaskComponentPolygons(const BinaryMaskCCL& ccl, uint32_t cid, const TileGeom& geom, uint32_t minHoleArea, double simplifyTolerance) const;
    Clipper2Lib::Paths64 buildLabelComponentPolygons(
        const std::vector<uint32_t>& pixelCid, size_t W, uint32_t cid,
        const PixBox& box, const TileGeom& geom,
        uint32_t minHoleArea, double simplifyTolerance) const;
    // detect CCs crossing tile borders and remap them to global IDs
    BorderRemapInfo remapTileToBorderComponents(TileCCL& t, uint32_t invalid) const;
    // merge CCs across tile borders
    BorderDSUState mergeBorderComponentsWithDSU(const std::vector<TileCCL>& perTile, uint8_t bg, uint32_t invalid) const;
    BorderMergeResult mergeSoftMaskTileBorders(const std::vector<SoftMaskTileResult>& perTile, uint32_t invalid) const;

    /* Polygon/path helper */
    Clipper2Lib::Paths64 normalizeMaskPolygons(const Clipper2Lib::Paths64& paths) const;
    Clipper2Lib::Paths64 cleanupMaskPolygons(const Clipper2Lib::Paths64& paths,
        uint32_t minHoleArea, double simplifyTolerance) const;
    nlohmann::json maskPathsToMultiPolygonGeoJSON(const Clipper2Lib::Paths64& paths) const;

    /* Operate on a categorical mask */
    // mask boundary pixels: iff any of its 4 neighbors have a different label
    void computeTileBoundaryMasks(std::vector<DenseTile>& tiles) const;
    // compute CCs (with boundary info for global merging)
    TileCCL tileLocalCCL(const DenseTile& tile, uint8_t bg,
        const std::vector<uint8_t>* boundaryMask = nullptr,
        bool keepPixelCid = true) const;



    /* Shared per-tile annotate/merge logic */
    template<typename OnEmitFn>
    uint32_t annotateTile2DPlainShared(TileReader& reader, const TileKey& tile,
        std::ifstream& tileStream,
        uint32_t ntok, int32_t icol_x, int32_t icol_y, float resXY,
        bool annoKeepAll, uint32_t placeholderK,
        OnEmitFn&& onEmit, const char* funcName) const;
    template<typename OnEmitFn>
    uint32_t annotateTile3DPlainShared(TileReader& reader, const TileKey& tile,
        std::ifstream& tileStream,
        uint32_t ntok, int32_t icol_x, int32_t icol_y, int32_t icol_z,
        float resXY, float resZ,
        bool annoKeepAll, uint32_t placeholderK,
        OnEmitFn&& onEmit, const char* funcName) const;
    template<typename OnEmitFn>
    MergedAnnotate2DCounts annotateMergedTile2DPlainShared(TileReader& reader,
        const TileKey& tile, std::vector<std::ifstream>& streams,
        const std::vector<MergeSourcePlan>& mergePlans,
        uint32_t ntok, int32_t icol_x, int32_t icol_y, float resXY,
        bool keepAllMain, bool keepAll, bool annoKeepAll,
        size_t totalK, OnEmitFn&& onEmit, const char* funcName) const;
    template<typename OnEmitFn>
    MergedAnnotate2DCounts annotateMergedTile3DPlainShared(TileReader& reader,
        const TileKey& tile, std::vector<std::ifstream>& streams,
        const std::vector<MergeSourcePlan>& mergePlans,
        uint32_t ntok, int32_t icol_x, int32_t icol_y, int32_t icol_z,
        float resXY, float resZ,
        bool keepAllMain, bool keepAll, bool annoKeepAll,
        size_t totalK, OnEmitFn&& onEmit, const char* funcName) const;
    template<typename OnEmitFn>
    uint32_t annotateSingleTile2DShared(TileReader& reader,
        const TileKey& tile, std::ifstream& tileStream,
        const std::unordered_map<std::string, uint32_t>& featureIndex,
        uint32_t ntok, int32_t icol_x, int32_t icol_y, int32_t icol_f,
        float resXY, bool annoKeepAll, uint32_t placeholderK,
        OnEmitFn&& onEmit, const char* funcName) const;
    template<typename OnEmitFn>
    uint32_t annotateSingleTile3DShared(TileReader& reader,
        const TileKey& tile, std::ifstream& tileStream,
        const std::unordered_map<std::string, uint32_t>& featureIndex,
        uint32_t ntok, int32_t icol_x, int32_t icol_y, int32_t icol_z,
        int32_t icol_f, float resXY, float resZ,
        bool annoKeepAll, uint32_t placeholderK,
        OnEmitFn&& onEmit, const char* funcName) const;
    template<typename OnEmitFn>
    MergedAnnotate2DCounts annotateMergedTile2DShared(TileReader& reader,
        const TileKey& tile, std::vector<std::ifstream>& streams,
        const std::vector<MergeSourcePlan>& mergePlans,
        const tileoperator_detail::feature::FeatureRemapPlan& featureRemap,
        const std::unordered_map<std::string, uint32_t>& featureIndex,
        uint32_t ntok, int32_t icol_x, int32_t icol_y, int32_t icol_f,
        float resXY, bool keepAllMain, bool keepAll, bool annoKeepAll,
        size_t totalK, OnEmitFn&& onEmit, const char* funcName) const;
    template<typename OnEmitFn>
    MergedAnnotate2DCounts annotateMergedTile3DShared(TileReader& reader,
        const TileKey& tile, std::vector<std::ifstream>& streams,
        const std::vector<MergeSourcePlan>& mergePlans,
        const tileoperator_detail::feature::FeatureRemapPlan& featureRemap,
        const std::unordered_map<std::string, uint32_t>& featureIndex,
        uint32_t ntok, int32_t icol_x, int32_t icol_y, int32_t icol_z,
        int32_t icol_f, float resXY, float resZ,
        bool keepAllMain, bool keepAll, bool annoKeepAll,
        size_t totalK, OnEmitFn&& onEmit, const char* funcName) const;
};
