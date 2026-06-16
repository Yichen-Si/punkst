#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace pmtiles_pyramid {

constexpr int32_t DEFAULT_POINT_MAX_TILE_BYTES = 5000000;
constexpr int32_t DEFAULT_POINT_MAX_TILE_FEATURES = 50000;
constexpr int32_t DEFAULT_POLYGON_MAX_TILE_BYTES = 5000000;
constexpr int32_t DEFAULT_POLYGON_MAX_TILE_FEATURES = 5000000;

enum class PolygonPriorityMode {
    Random,
    Area,
};

enum class SimplePolygonIssueReason {
    None,
    Degenerate,
    Multipolygon,
};

struct SimplePolygonTableReadOptions {
    int32_t icolId = 0;
    int32_t icolX = 1;
    int32_t icolY = 2;
    int32_t icolOrder = -1;
    char delimiter = '\0';
};

struct SimplePolygonRecord {
    std::string polygonId;
    std::vector<std::pair<int64_t, int64_t>> globalRing;
    double area = 0.0;
};

class SimplePolygonTableIndex {
public:
    SimplePolygonTableIndex() = default;
    SimplePolygonTableIndex(const std::string& path,
        const SimplePolygonTableReadOptions& options,
        uint8_t sourceZoom,
        uint32_t extent,
        double coordScale = 1.0,
        bool useAssignedU32Ids = false);

    void add_fragment(const std::string& polygonId,
        const std::vector<std::pair<int64_t, int64_t>>& globalRing);
    void finalize_repairs();

    const SimplePolygonRecord* find(const std::string& polygonId) const;
    SimplePolygonIssueReason issue_reason(const std::string& polygonId) const;
    size_t size() const {
        return polygons_.size();
    }

private:
    std::unordered_map<std::string, SimplePolygonRecord> polygons_;
    std::unordered_map<std::string, SimplePolygonIssueReason> issues_;
    std::unordered_map<std::string, std::vector<std::vector<std::pair<int64_t, int64_t>>>> pendingFragments_;
    std::unordered_map<std::string, uint32_t> assignedIdsBySourceId_;
};

struct BuildOptions {
    int32_t minZoom = 0;
    int32_t maxTileBytes = DEFAULT_POINT_MAX_TILE_BYTES;
    int32_t maxTileFeatures = DEFAULT_POINT_MAX_TILE_FEATURES;
    double scaleFactorCompression = 10.0;
    int32_t threads = 1;
    PolygonPriorityMode polygonPriorityMode = PolygonPriorityMode::Random;
    std::string polygonIdColumn;
    std::string polygonSourcePath;
    int32_t polygonSourceIcolId = 0;
    int32_t polygonSourceIcolX = 1;
    int32_t polygonSourceIcolY = 2;
    int32_t polygonSourceIcolOrder = -1;
    double polygonSourceCoordScale = std::numeric_limits<double>::quiet_NaN();
    double tileBufferPixels = std::numeric_limits<double>::quiet_NaN();
    bool polygonNoClipping = false;
    bool polygonNoDuplication = false;
    bool quiet = false;
};

void build_point_pmtiles_pyramid(const std::string& inPmtiles,
    const std::string& outPmtiles,
    const BuildOptions& options);

void build_polygon_pmtiles_pyramid(const std::string& inPmtiles,
    const std::string& outPmtiles,
    const BuildOptions& options);

void build_mixed_pmtiles_pyramid(const std::string& pointPmtiles,
    const std::string& polygonPmtiles,
    const std::string& outPmtiles,
    const BuildOptions& options);

void build_mixed_pmtiles_pyramid(const std::string& inPmtiles,
    const std::string& outPmtiles,
    const BuildOptions& options);

} // namespace pmtiles_pyramid
