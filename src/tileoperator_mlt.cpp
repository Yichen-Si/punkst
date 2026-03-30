#include "tileoperator.hpp"
#include "tileoperator_common.hpp"
#include "json.hpp"
#include "mlt_utils.hpp"
#include "pmtiles_utils.hpp"
#include "PMTiles/pmtiles.hpp"
#include "region_query.hpp"

#include <atomic>
#include <cinttypes>
#include <cstdio>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <functional>
#include <numeric>
#include <unordered_set>
#include <sys/stat.h>
#include <unistd.h>

namespace {

using tileoperator_detail::merge::append_placeholder_pairs;
using tileoperator_detail::merge::append_top_probs_prefix;
using tileoperator_detail::merge::build_merge_column_names;
using tileoperator_detail::merge::check_k2keep;
using tileoperator_detail::merge::tile_key_from_source_xy;
using tileoperator_detail::feature::build_feature_index_map;
using tileoperator_detail::feature::build_feature_remap_plan;
using tileoperator_detail::feature::FeatureRemapPlan;
using tileoperator_detail::feature::remap_feature_map_to_canonical;

struct ExtColumnPlan {
    size_t tokenIndex = 0;
    std::string name;
    mlt_pmtiles::ScalarType type = mlt_pmtiles::ScalarType::INT_32;
    bool nullable = false;
    std::string nullToken;
};

struct ExtColumnValue {
    bool present = true;
    int32_t intValue = 0;
    float floatValue = 0.0f;
    std::string stringValue = "";
};

struct AnnotatePackagingConfig {
    bool haveGeneBins = false;
    GeneBinInfo geneBins;
    std::vector<std::string> packagingFeatureNames;
    std::unordered_map<std::string, uint32_t> packagingFeatureIndex;
    mlt_pmtiles::GlobalStringDictionary featureDictionary;
};

struct AnnotateQueryPlan {
    std::vector<ExtColumnPlan> extColumns;
    mlt_pmtiles::FeatureTableSchema schema;
    float resXY = 0.001f;
    float resZ = 0.001f;
    uint32_t ntok = 0;
};

struct ParsedQueryCoord {
    float x = 0.0f;
    float y = 0.0f;
    int32_t ix = 0;
    int32_t iy = 0;
    int32_t countValue = 0;
};

const std::vector<ExtColumnPlan> kNoExtColumnPlans;

std::vector<mlt_pmtiles::PropertyColumn> make_extra_property_columns(
    const std::vector<ExtColumnPlan>& extColumns) {
    std::vector<mlt_pmtiles::PropertyColumn> out;
    out.reserve(extColumns.size());
    for (const auto& ext : extColumns) {
        out.emplace_back(ext.type, ext.nullable);
    }
    return out;
}

struct AccumTileData {
    bool hasZ = false;
    std::vector<int32_t> localX;
    std::vector<int32_t> localY;
    std::vector<uint32_t> featureCodes;
    std::vector<int32_t> countValues;
    std::vector<float> zValues;
    // Entries with present = false still have placeholder values
    std::vector<std::vector<int32_t>> kValues; // [k][row]
    std::vector<std::vector<bool>> kPresent;
    std::vector<std::vector<float>> pValues;
    std::vector<std::vector<bool>> pPresent;
    std::vector<mlt_pmtiles::PropertyColumn> extraColumns;

    AccumTileData() = default;
    AccumTileData(size_t kCount, bool withZ,
        const std::vector<ExtColumnPlan>& extColumnPlans = {})
        : hasZ(withZ),
          kValues(kCount),
          kPresent(kCount),
          pValues(kCount),
          pPresent(kCount),
          extraColumns(make_extra_property_columns(extColumnPlans)) {}

    size_t size() const {
        return localX.size();
    }

    void append(int32_t x, int32_t y, uint32_t featureCode, float z,
        int32_t countValue,
        const std::vector<int32_t>& ks, const std::vector<float>& ps,
        double encodeProbMin, double encodeProbEps,
        const std::vector<uint32_t>& probBlockSizes = {},
        const std::vector<ExtColumnValue>* extValues = nullptr) {
        if (ks.size() != kValues.size() || ps.size() != pValues.size()) {
            error("%s: top-k payload length mismatch", __func__);
        }
        if ((extValues == nullptr) != extraColumns.empty()) {
            error("%s: extra-column payload mismatch", __func__);
        }
        if (extValues != nullptr && extValues->size() != extraColumns.size()) {
            error("%s: extra-column payload length mismatch", __func__);
        }
        localX.push_back(x);
        localY.push_back(y);
        featureCodes.push_back(featureCode);
        countValues.push_back(countValue);
        if (hasZ) {
            zValues.push_back(z);
        }
        std::vector<uint32_t> defaultBlockSizes;
        const std::vector<uint32_t>* blockSizes = &probBlockSizes;
        if (blockSizes->empty()) {
            defaultBlockSizes.push_back(static_cast<uint32_t>(kValues.size()));
            blockSizes = &defaultBlockSizes;
        }
        size_t totalBlockK = 0;
        for (uint32_t blockK : *blockSizes) {
            if (blockK == 0) {
                error("%s: probability block sizes must be positive", __func__);
            }
            totalBlockK += blockK;
        }
        if (totalBlockK != kValues.size()) {
            error("%s: probability block size sum mismatch", __func__);
        }

        size_t blockIdx = 0;
        size_t blockOffset = 0;
        size_t blockLocalIdx = 0;
        bool pruneTail = false;
        for (size_t i = 0; i < kValues.size(); ++i) {
            if (i >= blockOffset + (*blockSizes)[blockIdx]) {
                blockOffset += (*blockSizes)[blockIdx];
                ++blockIdx;
                blockLocalIdx = 0;
                pruneTail = false;
            }
            bool kPresentValue = ks[i] >= 0;
            bool pPresentValue = kPresentValue;
            if (kPresentValue) {
                if (blockLocalIdx == 0 && encodeProbEps > 0.0 &&
                    static_cast<double>(ps[i]) > (1.0 - encodeProbEps)) {
                    pPresentValue = false;
                }
                if (blockLocalIdx > 0 && encodeProbMin >= 0.0 &&
                    (pruneTail || static_cast<double>(ps[i]) < encodeProbMin)) {
                    kPresentValue = false;
                    pPresentValue = false;
                    pruneTail = true;
                }
                kValues[i].push_back(ks[i]);
                pValues[i].push_back(ps[i]);
            } else {
                kValues[i].push_back(-1);
                pValues[i].push_back(0.0f);
            }
            kPresent[i].push_back(kPresentValue);
            pPresent[i].push_back(pPresentValue);
            ++blockLocalIdx;
        }
        if (extValues != nullptr) {
            for (size_t i = 0; i < extraColumns.size(); ++i) {
                auto& col = extraColumns[i];
                const auto& value = (*extValues)[i];
                if (col.nullable || !col.present.empty()) {
                    col.present.push_back(value.present);
                }
                switch (col.type) {
                case mlt_pmtiles::ScalarType::INT_32:
                    col.intValues.push_back(value.intValue);
                    break;
                case mlt_pmtiles::ScalarType::FLOAT:
                    col.floatValues.push_back(value.floatValue);
                    break;
                case mlt_pmtiles::ScalarType::STRING:
                    col.stringValues.push_back(value.stringValue);
                    break;
                default:
                    error("%s: unsupported extra-column type", __func__);
                }
            }
        }
    }

    void appendFrom(const AccumTileData& other) {
        if (hasZ != other.hasZ || kValues.size() != other.kValues.size() ||
            extraColumns.size() != other.extraColumns.size()) {
            error("%s: incompatible tile accumulators", __func__);
        }
        localX.insert(localX.end(), other.localX.begin(), other.localX.end());
        localY.insert(localY.end(), other.localY.begin(), other.localY.end());
        featureCodes.insert(featureCodes.end(), other.featureCodes.begin(), other.featureCodes.end());
        countValues.insert(countValues.end(), other.countValues.begin(), other.countValues.end());
        if (hasZ) {
            zValues.insert(zValues.end(), other.zValues.begin(), other.zValues.end());
        }
        for (size_t i = 0; i < kValues.size(); ++i) {
            kValues[i].insert(kValues[i].end(), other.kValues[i].begin(), other.kValues[i].end());
            kPresent[i].insert(kPresent[i].end(), other.kPresent[i].begin(), other.kPresent[i].end());
            pValues[i].insert(pValues[i].end(), other.pValues[i].begin(), other.pValues[i].end());
            pPresent[i].insert(pPresent[i].end(), other.pPresent[i].begin(), other.pPresent[i].end());
        }
        for (size_t i = 0; i < extraColumns.size(); ++i) {
            auto& dst = extraColumns[i];
            const auto& src = other.extraColumns[i];
            if (dst.type != src.type || dst.nullable != src.nullable) {
                error("%s: incompatible extra-column accumulators", __func__);
            }
            dst.present.insert(dst.present.end(), src.present.begin(), src.present.end());
            dst.intValues.insert(dst.intValues.end(), src.intValues.begin(), src.intValues.end());
            dst.floatValues.insert(dst.floatValues.end(), src.floatValues.begin(), src.floatValues.end());
            dst.stringValues.insert(dst.stringValues.end(), src.stringValues.begin(), src.stringValues.end());
        }
    }
};

struct OutputTileInfo {
    TileKey sourceKey{};
    uint8_t z = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint64_t tileId = 0;
    AccumTileData* data = nullptr;
};

struct GeoTileRect {
    double xmin = 0.0;
    double ymin = 0.0;
    double xmax = 0.0;
    double ymax = 0.0;
};

struct ExportRegionFilter {
    bool hasRect = false;
    Rectangle<float> rect;
    bool hasPolygon = false;
    PreparedRegionMask2D polygon;
    bool hasZRange = false;
    float zmin = std::numeric_limits<float>::quiet_NaN();
    float zmax = std::numeric_limits<float>::quiet_NaN();
};

struct BinTileKey {
    int32_t binId = -1;
    TileKey tile{};

    bool operator==(const BinTileKey& other) const {
        return binId == other.binId && tile == other.tile;
    }

    bool operator<(const BinTileKey& other) const {
        if (binId != other.binId) {
            return binId < other.binId;
        }
        return tile < other.tile;
    }
};

struct BinTileKeyHash {
    std::size_t operator()(const BinTileKey& key) const {
        return std::hash<int32_t>()(key.binId) ^ (TileKeyHash{}(key.tile) << 1);
    }
};

struct SpillFragmentIndex {
    int32_t binId = -1;
    TileKey tile{};
    uint64_t dataOffset = 0;
    uint64_t dataBytes = 0;
    uint32_t rowCount = 0;
};

struct WorkerSpillShard {
    std::string dataPath;
    int fdData = -1;
    uint64_t dataSize = 0;
    std::vector<SpillFragmentIndex> fragments;
};

struct WorkerScanResult {
    std::vector<mlt_pmtiles::EncodedTilePayload> completedTiles;
    WorkerSpillShard spill;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
    uint64_t totalRecordCount = 0;
};

struct SpillFragmentRef {
    int32_t binId = -1;
    size_t shardId = 0;
    uint64_t dataOffset = 0;
    uint64_t dataBytes = 0;
    uint32_t rowCount = 0;
};

struct GeneBinWorkerScanResult {
    std::vector<mlt_pmtiles::EncodedTilePayload> completedAllTiles;
    std::map<int32_t, std::vector<mlt_pmtiles::EncodedTilePayload>> completedBinTiles;
    WorkerSpillShard spill;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
    uint64_t totalRecordCount = 0;
    std::map<int32_t, uint64_t> binRecordCounts;
    std::unordered_set<uint32_t> missingBinFeatures;
};

struct GeneBinPipelineResult {
    std::vector<mlt_pmtiles::EncodedTilePayload> allEncodedTiles;
    std::map<int32_t, std::vector<mlt_pmtiles::EncodedTilePayload>> binEncodedTiles;
    uint64_t totalRecordCount = 0;
    std::map<int32_t, uint64_t> binRecordCounts;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
    std::unordered_set<uint32_t> missingBinFeatures;
};

struct AnnotatePmtilesWorkerResult {
    std::vector<mlt_pmtiles::EncodedTilePayload> completedAllTiles;
    std::map<int32_t, std::vector<mlt_pmtiles::EncodedTilePayload>> completedBinTiles;
    WorkerSpillShard spill;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
    uint64_t totalRecordCount = 0;
    std::map<int32_t, uint64_t> binRecordCounts;
    std::unordered_set<std::string> missingPackagingFeatures;
};

struct AnnotatePmtilesPipelineResult {
    std::vector<mlt_pmtiles::EncodedTilePayload> allEncodedTiles;
    std::map<int32_t, std::vector<mlt_pmtiles::EncodedTilePayload>> binEncodedTiles;
    uint64_t totalRecordCount = 0;
    std::map<int32_t, uint64_t> binRecordCounts;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
    std::unordered_set<std::string> missingPackagingFeatures;
};

struct ExportParsedKPCol {
    bool ok = false;
    bool isK = false;
    uint32_t idx = 0;
    std::string prefix;
};

struct ExportColumnPlan {
    size_t schemaIndex = 0;
    bool isZ = false;
    bool isK = false;
    bool isP = false;
    uint32_t kpIdx = 0;
    size_t pairedSchemaIndex = std::numeric_limits<size_t>::max();
};

struct ExportTextTile {
    IndexEntryF entry;
    std::string text;

    ExportTextTile() = default;
    ExportTextTile(int32_t row, int32_t col) : entry(row, col) {
        entry.n = 0;
        entry.st = 0;
        entry.ed = 0;
        entry.resetBounds();
    }
};

template<typename T>
void reorder_vector(std::vector<T>& values, const std::vector<size_t>& order) {
    std::vector<T> reordered;
    reordered.reserve(values.size());
    for (size_t idx : order) {
        reordered.push_back(values[idx]);
    }
    values.swap(reordered);
}

template<typename T>
void reorder_optional_vector(std::vector<T>& values, const std::vector<size_t>& order,
    const char* fieldName) {
    if (values.empty()) {
        return;
    }
    if (values.size() != order.size()) {
        error("%s: optional column '%s' has %zu values for %zu rows",
            __func__, fieldName, values.size(), order.size());
    }
    reorder_vector(values, order);
}

AccumTileData sort_tile_rows(const AccumTileData& in) {
    AccumTileData out = in;
    std::vector<size_t> order(out.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](size_t lhs, size_t rhs) {
        if (out.localY[lhs] != out.localY[rhs]) {
            return out.localY[lhs] < out.localY[rhs];
        }
        if (out.localX[lhs] != out.localX[rhs]) {
            return out.localX[lhs] < out.localX[rhs];
        }
        return out.featureCodes[lhs] < out.featureCodes[rhs];
    });

    reorder_vector(out.localX, order);
    reorder_vector(out.localY, order);
    reorder_vector(out.featureCodes, order);
    reorder_vector(out.countValues, order);
    if (out.hasZ) {
        reorder_vector(out.zValues, order);
    }
    for (size_t i = 0; i < out.kValues.size(); ++i) {
        reorder_vector(out.kValues[i], order);
        reorder_vector(out.kPresent[i], order);
        reorder_vector(out.pValues[i], order);
        reorder_vector(out.pPresent[i], order);
    }
    for (auto& col : out.extraColumns) {
        reorder_optional_vector(col.present, order, "present");
        reorder_optional_vector(col.intValues, order, "intValues");
        reorder_optional_vector(col.floatValues, order, "floatValues");
        reorder_optional_vector(col.stringValues, order, "stringValues");
    }
    return out;
}

mlt_pmtiles::PointTileData build_point_tile_data(const AccumTileData& in) {
    mlt_pmtiles::PointTileData out;
    out.localX = in.localX;
    out.localY = in.localY;
    out.columns.reserve(2 + (in.hasZ ? 1 : 0) + in.kValues.size() * 2 + in.extraColumns.size());

    mlt_pmtiles::PropertyColumn featureCol(mlt_pmtiles::ScalarType::STRING, false);
    featureCol.stringCodes = in.featureCodes;
    out.columns.push_back(std::move(featureCol));

    mlt_pmtiles::PropertyColumn countCol(mlt_pmtiles::ScalarType::INT_32, false);
    countCol.intValues = in.countValues;
    out.columns.push_back(std::move(countCol));

    if (in.hasZ) {
        mlt_pmtiles::PropertyColumn zCol(mlt_pmtiles::ScalarType::FLOAT, false);
        zCol.floatValues = in.zValues;
        out.columns.push_back(std::move(zCol));
    }

    for (size_t i = 0; i < in.kValues.size(); ++i) {
        mlt_pmtiles::PropertyColumn kCol(mlt_pmtiles::ScalarType::INT_32, true);
        kCol.present = in.kPresent[i];
        kCol.intValues = in.kValues[i];
        out.columns.push_back(std::move(kCol));

        mlt_pmtiles::PropertyColumn pCol(mlt_pmtiles::ScalarType::FLOAT, true);
        pCol.present = in.pPresent[i];
        pCol.floatValues = in.pValues[i];
        out.columns.push_back(std::move(pCol));
    }
    for (const auto& col : in.extraColumns) {
        out.columns.push_back(col);
    }
    return out;
}

mlt_pmtiles::FeatureTableSchema build_point_schema(const std::string& layerName,
    uint32_t extent, bool hasZ, const std::vector<std::string>& probColumnNames,
    const std::vector<bool>& probPairNullable = {},
    const std::vector<mlt_pmtiles::ColumnSchema>& extraColumns = {}) {
    if ((probColumnNames.size() % 2) != 0) {
        error("%s: probability column names must come in K/P pairs", __func__);
    }
    if (!probPairNullable.empty() && probPairNullable.size() * 2 != probColumnNames.size()) {
        error("%s: probability nullability count mismatch", __func__);
    }
    mlt_pmtiles::FeatureTableSchema schema;
    schema.layerName = layerName;
    schema.extent = extent;
    schema.columns.push_back({"feature", mlt_pmtiles::ScalarType::STRING, false});
    schema.columns.push_back({"ct", mlt_pmtiles::ScalarType::INT_32, false});
    if (hasZ) {
        schema.columns.push_back({"z", mlt_pmtiles::ScalarType::FLOAT, false});
    }
    for (size_t i = 0; i < probColumnNames.size(); i += 2) {
        const bool nullable = probPairNullable.empty() ? true : probPairNullable[i / 2];
        schema.columns.push_back({probColumnNames[i], mlt_pmtiles::ScalarType::INT_32, nullable});
        schema.columns.push_back({probColumnNames[i + 1], mlt_pmtiles::ScalarType::FLOAT, nullable});
    }
    for (const auto& col : extraColumns) {
        schema.columns.push_back(col);
    }
    return schema;
}

std::vector<bool> build_single_prob_nullable_flags(const std::vector<uint32_t>& kvec, bool annoKeepAll, double encodeProbMin, double encodeProbEps) {
    std::vector<bool> out;
    out.reserve(std::accumulate(kvec.begin(), kvec.end(), size_t(0)));
    for (uint32_t blockK : kvec) {
        for (uint32_t i = 0; i < blockK; ++i) {
            out.push_back(annoKeepAll ||
                (i == 0 && encodeProbEps > 0.0) ||
                (i > 0 && encodeProbMin > 0.0));
        }
    }
    return out;
}

std::vector<bool> build_merged_prob_nullable_flags(const std::vector<uint32_t>& k2keep, bool annoKeepAll, bool keepAllMain, bool keepAll, double encodeProbMin, double encodeProbEps) {
    std::vector<bool> out;
    out.reserve(std::accumulate(k2keep.begin(), k2keep.end(), size_t(0)));
    for (size_t srcIdx = 0; srcIdx < k2keep.size(); ++srcIdx) {
        const bool pair1Nullable = annoKeepAll || keepAll || (keepAllMain && srcIdx != 0) || (encodeProbEps > 0.0);
        for (uint32_t i = 0; i < k2keep[srcIdx]; ++i) {
            out.push_back(pair1Nullable || (i > 0 && encodeProbMin > 0.0));
        }
    }
    return out;
}

std::vector<std::string> split_ext_column_spec(const std::string& spec) {
    std::vector<std::string> tokens;
    size_t start = 0;
    for (;;) {
        const size_t pos = spec.find(':', start);
        if (pos == std::string::npos) {
            tokens.push_back(spec.substr(start));
            break;
        }
        tokens.push_back(spec.substr(start, pos - start));
        start = pos + 1;
    }
    return tokens;
}

bool has_ext_column_specs(const TileOperator::MltPmtilesOptions& mltOptions) {
    return !mltOptions.ext_col_ints.empty() ||
           !mltOptions.ext_col_floats.empty() ||
           !mltOptions.ext_col_strs.empty();
}

bool needs_query_header_names(const std::vector<std::string>& rawSpecs) {
    for (const auto& raw : rawSpecs) {
        if (split_ext_column_spec(raw).size() == 1) {
            return true;
        }
    }
    return false;
}

ExportParsedKPCol parse_export_kp_col(const std::string& key) {
    ExportParsedKPCol out;
    if (key.size() < 2) {
        return out;
    }
    size_t pos = key.size();
    while (pos > 0 && std::isdigit(static_cast<unsigned char>(key[pos - 1]))) {
        --pos;
    }
    if (pos == key.size() || pos == 0) {
        return out;
    }
    const char kp = static_cast<char>(std::toupper(static_cast<unsigned char>(key[pos - 1])));
    if (kp != 'K' && kp != 'P') {
        return out;
    }
    if (pos > 1) {
        const std::string prefix = key.substr(0, pos - 1);
        if (!prefix.empty() && prefix.back() != '_') {
            return out;
        }
        out.prefix = prefix;
    }
    uint32_t parsedIdx = 0;
    if (!str2uint32(key.substr(pos), parsedIdx) || parsedIdx == 0) {
        return out;
    }
    out.ok = true;
    out.isK = (kp == 'K');
    out.idx = parsedIdx;
    return out;
}

bool export_region_requested(const TileOperator::ExportPmtilesOptions& options) {
    const bool hasBBox = (options.xmax > options.xmin) && (options.ymax > options.ymin);
    const bool hasZRange = !std::isnan(options.zmin) || !std::isnan(options.zmax);
    return hasBBox || !options.geojsonFile.empty() || hasZRange;
}

ExportRegionFilter build_export_region_filter(
    const TileOperator::ExportPmtilesOptions& options) {
    ExportRegionFilter out;
    out.hasRect = (options.xmax > options.xmin) && (options.ymax > options.ymin);
    if (out.hasRect) {
        out.rect = Rectangle<float>(options.xmin, options.ymin, options.xmax, options.ymax);
    }
    out.hasZRange = !std::isnan(options.zmin) || !std::isnan(options.zmax);
    if (out.hasZRange) {
        if (std::isnan(options.zmin) || std::isnan(options.zmax)) {
            error("%s: z-range filtering requires both zmin and zmax", __func__);
        }
        if (!(options.zmax > options.zmin)) {
            error("%s: z-range filtering requires zmax > zmin", __func__);
        }
        out.zmin = options.zmin;
        out.zmax = options.zmax;
    }
    if (!options.geojsonFile.empty()) {
        try {
            out.polygon = loadPreparedRegionGeoJSON(
                options.geojsonFile, options.tileSize, options.geojsonScale);
        } catch (const std::exception& ex) {
            error("%s: %s", __func__, ex.what());
        }
        out.hasPolygon = true;
    }
    return out;
}

Rectangle<float> geo_tile_rect_to_coord_rect(const GeoTileRect& rect, double coordScale) {
    const double scale = (coordScale > 0.0) ? coordScale : 1.0;
    return Rectangle<float>(
        static_cast<float>(rect.xmin / scale),
        static_cast<float>(rect.ymin / scale),
        static_cast<float>(rect.xmax / scale),
        static_cast<float>(rect.ymax / scale));
}

GeoTileRect pmtiles_entry_rect_epsg3857(const pmtiles::entry_zxy& entry) {
    double x0 = 0.0;
    double y0 = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    mlt_pmtiles::tilecoord_to_epsg3857(entry.x, entry.y, 0.0, 0.0, entry.z, x0, y0);
    mlt_pmtiles::tilecoord_to_epsg3857(entry.x, entry.y, 256.0, 256.0, entry.z, x1, y1);
    GeoTileRect rect;
    rect.xmin = std::min(x0, x1);
    rect.xmax = std::max(x0, x1);
    rect.ymin = std::min(y0, y1);
    rect.ymax = std::max(y0, y1);
    return rect;
}

bool export_entry_matches_region(const Rectangle<float>& entryRect,
    const ExportRegionFilter& filter) {
    if (filter.hasRect && entryRect.intersect(filter.rect) == 0) {
        return false;
    }
    if (filter.hasPolygon && entryRect.intersect(filter.polygon.bbox_f) == 0) {
        return false;
    }
    return true;
}

bool export_row_matches_spatial_region(float x, float y,
    const TileKey& outTile,
    const ExportRegionFilter& filter) {
    if (filter.hasRect && !filter.rect.contains(x, y)) {
        return false;
    }
    if (filter.hasPolygon) {
        const RegionTileState state = filter.polygon.classifyTile(outTile);
        if (state == RegionTileState::Outside) {
            return false;
        }
        if (state == RegionTileState::Partial &&
            !filter.polygon.containsPoint(x, y, &outTile)) {
            return false;
        }
    }
    return true;
}

size_t find_schema_column_index(const mlt_pmtiles::FeatureTableSchema& schema,
    const std::string& name) {
    for (size_t i = 0; i < schema.columns.size(); ++i) {
        if (schema.columns[i].name == name) {
            return i;
        }
    }
    return schema.columns.size();
}

float export_z_value_at_row(const mlt_pmtiles::ColumnSchema& schema,
    const mlt_pmtiles::PropertyColumn& column,
    size_t row) {
    const bool present = !schema.nullable || column.present.empty() || column.present[row];
    if (!present) {
        error("%s: z column cannot be null in exported PMTiles rows", __func__);
    }
    switch (schema.type) {
    case mlt_pmtiles::ScalarType::FLOAT:
        return column.floatValues[row];
    case mlt_pmtiles::ScalarType::INT_32:
        return static_cast<float>(column.intValues[row]);
    default:
        error("%s: unsupported z column type %d", __func__, static_cast<int>(schema.type));
        return 0.0f;
    }
}

double export_coord_scale_from_metadata(const nlohmann::json& metadata) {
    if (!metadata.contains("coord_scale")) {
        return 1.0;
    }
    if (!metadata["coord_scale"].is_number()) {
        error("%s: PMTiles metadata coord_scale must be numeric", __func__);
    }
    const double coordScale = metadata["coord_scale"].get<double>();
    if (!(coordScale > 0.0)) {
        error("%s: PMTiles metadata coord_scale must be positive", __func__);
    }
    return coordScale;
}

void validate_export_schema_compatible(const mlt_pmtiles::FeatureTableSchema& ref,
    const mlt_pmtiles::FeatureTableSchema& cur) {
    if (ref.extent != cur.extent || ref.columns.size() != cur.columns.size()) {
        error("%s: inconsistent MLT schemas across PMTiles tiles", __func__);
    }
    for (size_t i = 0; i < ref.columns.size(); ++i) {
        const auto& lhs = ref.columns[i];
        const auto& rhs = cur.columns[i];
        if (lhs.name != rhs.name || lhs.type != rhs.type || lhs.nullable != rhs.nullable) {
            error("%s: inconsistent MLT schema column %zu across PMTiles tiles", __func__, i);
        }
    }
}

std::vector<ExportColumnPlan> build_export_column_plan(
    const mlt_pmtiles::FeatureTableSchema& schema,
    std::vector<std::string>& headerColumns) {
    std::vector<ExportColumnPlan> out;
    headerColumns.clear();
    headerColumns.push_back("x");
    headerColumns.push_back("y");
    std::unordered_map<std::string, size_t> kColumnByPrefixIdx;

    for (size_t i = 0; i < schema.columns.size(); ++i) {
        if (schema.columns[i].name == "z") {
            out.push_back(ExportColumnPlan{i, true, false, false, 0, std::numeric_limits<size_t>::max()});
            headerColumns.push_back("z");
            break;
        }
    }
    for (size_t i = 0; i < schema.columns.size(); ++i) {
        if (schema.columns[i].name == "z") {
            continue;
        }
        const ExportParsedKPCol parsed = parse_export_kp_col(schema.columns[i].name);
        ExportColumnPlan plan;
        plan.schemaIndex = i;
        plan.isZ = false;
        plan.isK = parsed.ok && parsed.isK;
        plan.isP = parsed.ok && !parsed.isK;
        if (parsed.ok) {
            plan.kpIdx = parsed.idx;
            const std::string kpKey = parsed.prefix + "#" + std::to_string(parsed.idx);
            if (plan.isK) {
                kColumnByPrefixIdx[kpKey] = i;
            } else {
                const auto it = kColumnByPrefixIdx.find(kpKey);
                if (it != kColumnByPrefixIdx.end()) {
                    plan.pairedSchemaIndex = it->second;
                }
            }
        }
        out.push_back(std::move(plan));
        headerColumns.push_back(schema.columns[i].name);
    }
    return out;
}

std::vector<uint32_t> infer_export_kvec(const std::vector<std::string>& headerColumns) {
    struct KPColsByPrefix {
        std::unordered_map<uint32_t, uint32_t> kcols;
        std::unordered_map<uint32_t, uint32_t> pcols;
    };
    std::vector<std::string> prefixOrder;
    std::unordered_map<std::string, KPColsByPrefix> kpCols;
    for (size_t col = 0; col < headerColumns.size(); ++col) {
        const ExportParsedKPCol parsed = parse_export_kp_col(headerColumns[col]);
        if (!parsed.ok) {
            continue;
        }
        auto [it, inserted] = kpCols.emplace(parsed.prefix, KPColsByPrefix{});
        if (inserted) {
            prefixOrder.push_back(parsed.prefix);
        }
        auto& target = parsed.isK ? it->second.kcols : it->second.pcols;
        target.emplace(parsed.idx, static_cast<uint32_t>(col));
    }

    std::vector<uint32_t> out;
    for (const auto& prefix : prefixOrder) {
        const auto it = kpCols.find(prefix);
        if (it == kpCols.end()) {
            continue;
        }
        uint32_t count = 0;
        for (uint32_t idx = 1; ; ++idx) {
            const bool hasK = it->second.kcols.count(idx) > 0;
            const bool hasP = it->second.pcols.count(idx) > 0;
            if (!hasK && !hasP) {
                break;
            }
            if (!hasK || !hasP) {
                error("%s: incomplete K/P pair for exported prefix '%s' idx %u",
                    __func__, prefix.c_str(), idx);
            }
            ++count;
        }
        if (count > 0) {
            out.push_back(count);
        }
    }
    return out;
}

std::string format_prob_value(float value, int32_t digits) {
    char buf[64];
    std::snprintf(buf, sizeof(buf), "%.*e", digits, static_cast<double>(value));
    return buf;
}

void relabel_encoded_tile_payloads_layer_name(
    std::vector<mlt_pmtiles::EncodedTilePayload>& encodedTiles,
    const std::string& layerName) {
    for (auto& tile : encodedTiles) {
        const std::string raw =
            mlt_pmtiles::gzip_decompress(tile.compressedData);
        const std::string relabeled =
            mlt_pmtiles::rewrite_point_tile_layer_name(raw, layerName);
        tile.compressedData = mlt_pmtiles::gzip_compress(relabeled);
    }
}

std::string format_export_column_value(const mlt_pmtiles::ColumnSchema& schema,
    const mlt_pmtiles::PointTileData& tile,
    const mlt_pmtiles::PropertyColumn& column, size_t row,
    const ExportColumnPlan& plan,
    const TileOperator::ExportPmtilesOptions& options) {
    const bool present = !schema.nullable || column.present.empty() || column.present[row];
    if (!present) {
        if (plan.isP && plan.kpIdx == 1 &&
            plan.pairedSchemaIndex < tile.columns.size()) {
            const auto& kSchema = tile.columns[plan.pairedSchemaIndex];
            const bool kPresent = kSchema.present.empty() || kSchema.present[row];
            if (kPresent) {
                return format_prob_value(1.0f, options.probDigits);
            }
        }
        if (plan.isK) {
            return "-1";
        }
        if (plan.isP) {
            return "0";
        }
        return "NA";
    }

    switch (schema.type) {
    case mlt_pmtiles::ScalarType::BOOLEAN:
        return column.boolValues[row] ? "1" : "0";
    case mlt_pmtiles::ScalarType::INT_32:
        return std::to_string(column.intValues[row]);
    case mlt_pmtiles::ScalarType::FLOAT:
        if (plan.isZ) {
            return fp_to_string(column.floatValues[row], options.coordDigits);
        }
        return format_prob_value(column.floatValues[row], options.probDigits);
    case mlt_pmtiles::ScalarType::STRING:
        return column.stringValues[row];
    default:
        error("%s: unsupported export column type %d", __func__, static_cast<int>(schema.type));
        return std::string();
    }
}

std::vector<std::string> maybe_load_query_header_columns(
    const TileOperator& queryOp, const TileOperator::MltPmtilesOptions& mltOptions) {
    if (!needs_query_header_names(mltOptions.ext_col_ints) &&
        !needs_query_header_names(mltOptions.ext_col_floats) &&
        !needs_query_header_names(mltOptions.ext_col_strs)) {
        return {};
    }
    return queryOp.getHeaderColumns();
}

void append_ext_column_plans(std::vector<ExtColumnPlan>& out,
    std::unordered_set<size_t>& usedIndices,
    std::unordered_set<std::string>& usedNames,
    const std::vector<std::string>& rawSpecs,
    mlt_pmtiles::ScalarType type,
    const std::vector<std::string>& headerColumns,
    const std::unordered_set<size_t>& reservedIndices,
    const char* optName) {
    for (const auto& raw : rawSpecs) {
        const std::vector<std::string> tokens = split_ext_column_spec(raw);
        if (tokens.empty() || tokens.size() > 3) {
            error("%s: invalid extended-column spec '%s' from %s", __func__, raw.c_str(), optName);
        }
        size_t idx = 0;
        if (!str2num<size_t>(tokens[0], idx)) {
            error("%s: invalid column index in '%s' from %s", __func__, raw.c_str(), optName);
        }
        if (reservedIndices.count(idx) > 0) {
            error("%s: column %zu from %s overlaps a reserved query column", __func__, idx, optName);
        }
        if (!usedIndices.emplace(idx).second) {
            error("%s: duplicate query column index %zu in extended-column specs", __func__, idx);
        }

        ExtColumnPlan plan;
        plan.tokenIndex = idx;
        plan.type = type;

        if (tokens.size() == 1) {
            if (idx >= headerColumns.size()) {
                error("%s: column %zu requires a header-derived name, but the query header has only %zu columns",
                    __func__, idx, headerColumns.size());
            }
            plan.name = headerColumns[idx];
        } else {
            if (tokens[1].empty()) {
                error("%s: extended-column spec '%s' must provide a non-empty name when ':' is used",
                    __func__, raw.c_str());
            }
            plan.name = tokens[1];
            if (tokens.size() == 3) {
                plan.nullable = true;
                plan.nullToken = tokens[2];
            }
        }
        if (plan.name.empty()) {
            error("%s: extended-column name cannot be empty", __func__);
        }
        if (!usedNames.emplace(plan.name).second) {
            error("%s: duplicate extended-column name '%s'", __func__, plan.name.c_str());
        }
        out.push_back(std::move(plan));
    }
}

std::vector<mlt_pmtiles::ColumnSchema> build_extra_column_schemas(
    const std::vector<ExtColumnPlan>& extColumns) {
    std::vector<mlt_pmtiles::ColumnSchema> out;
    out.reserve(extColumns.size());
    for (const auto& ext : extColumns) {
        out.push_back({ext.name, ext.type, ext.nullable});
    }
    return out;
}

std::vector<ExtColumnPlan> build_ext_column_plans(const TileOperator& queryOp,
    int32_t icol_x, int32_t icol_y, int32_t icol_z,
    int32_t icol_f, int32_t icol_count,
    const TileOperator::MltPmtilesOptions& mltOptions,
    const std::vector<std::string>& reservedNames) {
    if (!has_ext_column_specs(mltOptions)) {
        return {};
    }
    const std::vector<std::string> headerColumns =
        maybe_load_query_header_columns(queryOp, mltOptions);
    std::unordered_set<size_t> reservedIndices{
        static_cast<size_t>(icol_x),
        static_cast<size_t>(icol_y),
        static_cast<size_t>(icol_f)
    };
    if (icol_z >= 0) {
        reservedIndices.emplace(static_cast<size_t>(icol_z));
    }
    if (icol_count >= 0) {
        reservedIndices.emplace(static_cast<size_t>(icol_count));
    }
    std::unordered_set<size_t> usedIndices;
    std::unordered_set<std::string> usedNames(reservedNames.begin(), reservedNames.end());
    std::vector<ExtColumnPlan> out;
    append_ext_column_plans(out, usedIndices, usedNames, mltOptions.ext_col_ints,
        mlt_pmtiles::ScalarType::INT_32, headerColumns, reservedIndices, "--ext-col-ints");
    append_ext_column_plans(out, usedIndices, usedNames, mltOptions.ext_col_floats,
        mlt_pmtiles::ScalarType::FLOAT, headerColumns, reservedIndices, "--ext-col-floats");
    append_ext_column_plans(out, usedIndices, usedNames, mltOptions.ext_col_strs,
        mlt_pmtiles::ScalarType::STRING, headerColumns, reservedIndices, "--ext-col-strs");
    return out;
}

bool parse_ext_column_values(const std::vector<std::string>& tokens,
    const std::vector<ExtColumnPlan>& extColumns,
    std::vector<ExtColumnValue>& out, bool allowMissing = true) {
    out.clear();
    out.reserve(extColumns.size());
    for (const auto& ext : extColumns) {
        if (ext.tokenIndex >= tokens.size()) {
            return false;
        }
        const std::string& token = tokens[ext.tokenIndex];
        ExtColumnValue value;
        if (ext.nullable && token == ext.nullToken) {
            value.present = false;
            out.push_back(value);
            continue;
        }
        switch (ext.type) {
        case mlt_pmtiles::ScalarType::INT_32:
            if (!str2num<int32_t>(token, value.intValue) && !allowMissing) {
                return false;
            }
            break;
        case mlt_pmtiles::ScalarType::FLOAT:
            if (!str2num<float>(token, value.floatValue) && !allowMissing) {
                return false;
            }
            break;
        case mlt_pmtiles::ScalarType::STRING:
            value.stringValue = token;
            break;
        default:
            error("%s: unsupported extended-column type", __func__);
        }
        out.push_back(value);
    }
    return true;
}

uint32_t required_query_token_count(int32_t icol_x, int32_t icol_y,
    int32_t icol_z, int32_t icol_f, int32_t icol_count,
    const std::vector<ExtColumnPlan>& extColumns) {
    size_t maxIndex = static_cast<size_t>(std::max({icol_x, icol_y, icol_f, icol_count}));
    if (icol_z >= 0) {
        maxIndex = std::max(maxIndex, static_cast<size_t>(icol_z));
    }
    for (const auto& ext : extColumns) {
        maxIndex = std::max(maxIndex, ext.tokenIndex);
    }
    return static_cast<uint32_t>(maxIndex + 1);
}

std::vector<std::string> build_reserved_column_names(
    const std::vector<std::string>& probColumnNames, bool hasZ) {
    std::vector<std::string> out{"feature", "ct"};
    if (hasZ) {
        out.push_back("z");
    }
    out.insert(out.end(), probColumnNames.begin(), probColumnNames.end());
    return out;
}

// Add one record to one tile
void append_row_to_epsg3857_tile_map(std::map<TileKey, AccumTileData>& tileMap,
    const mlt_pmtiles::FeatureTableSchema& schema,
    double coordScale, uint8_t zoom,
    bool hasZ, size_t kCount,
    const std::vector<uint32_t>& probBlockSizes,
    float x, float y, float z, uint32_t featureIdx, int32_t countValue,
    const std::vector<int32_t>& ks, const std::vector<float>& ps,
    const std::vector<ExtColumnPlan>& extColumnPlans,
    const std::vector<ExtColumnValue>* extValues,
    double encodeProbMin, double encodeProbEps,
    double& geoMinX, double& geoMinY, double& geoMaxX, double& geoMaxY,
    uint64_t& totalRecordCount);

std::vector<mlt_pmtiles::EncodedTilePayload> encode_epsg3857_tile_map(
    std::map<TileKey, AccumTileData>& tileMap,
    uint8_t zoom,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::GlobalStringDictionary& featureDictionary,
    int32_t threads);

void write_single_layer_pmtiles_archive(const std::string& outFile,
    const mlt_pmtiles::FeatureTableSchema& schema,
    std::vector<mlt_pmtiles::EncodedTilePayload> encodedTiles,
    uint64_t totalRecordCount, double coordScale,
    size_t featureDictionarySize, uint8_t outputZoom,
    double geoMinX, double geoMinY, double geoMaxX, double geoMaxY,
    const std::string& generator);

bool has_gene_bin_definition(const TileOperator::MltPmtilesOptions& mltOptions) {
    if (!mltOptions.gene_bin_info_file.empty()) {
        return true;
    }
    return !mltOptions.feature_count_file.empty() && mltOptions.n_gene_bins > 0;
}

GeneBinInfo build_gene_bin_info(const std::string& geneBinInfoFile,
    const std::string& featureCountFile, int32_t nGeneBins) {
    if (!geneBinInfoFile.empty()) {
        if (!featureCountFile.empty()) {
            warning("%s: both gene-bin JSON and feature-count TSV were provided; using %s",
                __func__, geneBinInfoFile.c_str());
        }
        return GeneBinInfo(geneBinInfoFile);
    }
    if (!featureCountFile.empty()) {
        return GeneBinInfo(featureCountFile, nGeneBins, 0, 1, 0);
    }
    error("%s: either gene-bin JSON or feature-count TSV is required", __func__);
    return GeneBinInfo();
}

std::vector<std::string> load_query_feature_names(
    const std::string& ptPrefix, int32_t icol_f) {
    if (icol_f < 0) {
        error("%s: icol_feature must be >= 0", __func__);
    }
    TileReader reader(ptPrefix + ".tsv", ptPrefix + ".index");
    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    std::sort(tiles.begin(), tiles.end());
    const uint32_t ntok = static_cast<uint32_t>(icol_f + 1);
    std::unordered_set<std::string> seen;
    std::vector<std::string> out;
    std::string s;
    for (const auto& tile : tiles) {
        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) {
            continue;
        }
        while (it->next(s)) {
            if (s.empty() || s[0] == '#') {
                continue;
            }
            std::vector<std::string> tokens;
            split(tokens, "\t", s, ntok + 1, true, true, true);
            if (tokens.size() < ntok) {
                error("%s: invalid line: %s", __func__, s.c_str());
            }
            const std::string& featureName = tokens[icol_f];
            if (seen.emplace(featureName).second) {
                out.push_back(featureName);
            }
        }
    }
    return out;
}

// Build feature dictionary
AnnotatePackagingConfig build_annotate_packaging_config(
    const std::vector<std::string>& defaultFeatureNames,
    const TileOperator::MltPmtilesOptions& mltOptions) {
    AnnotatePackagingConfig out;
    out.haveGeneBins = has_gene_bin_definition(mltOptions);
    if (out.haveGeneBins) {
        out.geneBins = build_gene_bin_info(mltOptions.gene_bin_info_file,
            mltOptions.feature_count_file, mltOptions.n_gene_bins);
        out.packagingFeatureNames.reserve(out.geneBins.entries.size());
        for (const auto& entry : out.geneBins.entries) {
            out.packagingFeatureNames.push_back(entry.feature);
        }
    } else {
        out.packagingFeatureNames = defaultFeatureNames;
    }
    out.packagingFeatureIndex = build_feature_index_map(out.packagingFeatureNames);
    out.featureDictionary.values = out.packagingFeatureNames;
    return out;
}

// Build FeatureTableSchema
AnnotateQueryPlan build_annotate_query_plan(const TileOperator& sourceOp,
    const std::string& ptPrefix,
    int32_t icol_x, int32_t icol_y, int32_t icol_z,
    int32_t icol_f, int32_t icol_count,
    const TileOperator::MltPmtilesOptions& mltOptions,
    const std::vector<std::string>& probColumnNames,
    const std::vector<bool>& probPairNullable,
    bool hasZ,
    const std::string& layerName) {
    TileOperator queryOp(ptPrefix + ".tsv", ptPrefix + ".index");
    AnnotateQueryPlan out;
    out.extColumns = build_ext_column_plans(queryOp,
        icol_x, icol_y, icol_z, icol_f, icol_count,
        mltOptions, build_reserved_column_names(probColumnNames, hasZ));
    out.schema = build_point_schema(layerName, 4096, hasZ,
        probColumnNames, probPairNullable,
        build_extra_column_schemas(out.extColumns));
    out.resXY = sourceOp.getPixelResolution() > 0.0f ? sourceOp.getPixelResolution() : 0.001f;
    out.resZ = hasZ ? sourceOp.getPixelResolutionZ() : 0.001f;
    out.ntok = required_query_token_count(
        icol_x, icol_y, icol_z, icol_f, icol_count, out.extColumns);
    return out;
}

// Extract coordinates from line
bool parse_query_coord(const std::vector<std::string>& tokens,
    int32_t icol_x, int32_t icol_y, int32_t icol_count,
    float resXY, ParsedQueryCoord& out) {
    if (!str2float(tokens[icol_x], out.x) ||
        !str2float(tokens[icol_y], out.y) ||
        !str2int32(tokens[icol_count], out.countValue)) {
        return false;
    }
    out.ix = static_cast<int32_t>(std::floor(out.x / resXY));
    out.iy = static_cast<int32_t>(std::floor(out.y / resXY));
    return true;
}

bool lookup_packaging_feature_index(const std::string& featureName,
    const std::unordered_map<std::string, uint32_t>& packagingFeatureIndex,
    std::unordered_set<std::string>& warnedMissingPackagingFeatures,
    uint32_t& packagingFeatureIdx) {
    const auto it = packagingFeatureIndex.find(featureName);
    if (it == packagingFeatureIndex.end()) {
        warnedMissingPackagingFeatures.emplace(featureName);
        return false;
    }
    packagingFeatureIdx = it->second;
    return true;
}

void write_pmtiles_index_tsv(const std::string& outFile,
    const std::string& outPrefix,
    uint64_t allMoleculesCount, size_t allFeaturesCount,
    const std::map<int32_t, uint64_t>& binMoleculeCounts,
    const GeneBinInfo& geneBins) {
    std::ofstream out(outFile);
    if (!out.is_open()) {
        error("%s: cannot open %s for writing", __func__, outFile.c_str());
    }
    const std::string base = basename(outPrefix);
    out << "bin_id\tmolecules_count\tfeatures_count\tpmtiles_path\n";
    out << "all\t" << allMoleculesCount << "\t" << allFeaturesCount << "\t"
        << base << "_all.pmtiles\n";
    for (const auto& kv : geneBins.featuresPerBin) {
        const uint64_t molCount = binMoleculeCounts.count(kv.first) > 0 ? binMoleculeCounts.at(kv.first) : 0;
        out << kv.first << "\t" << molCount << "\t" << kv.second << "\t"
            << base << "_bin" << kv.first << ".pmtiles\n";
    }
}

struct AnnotatePmtilesState {
    std::map<TileKey, AccumTileData> allTileMap;
    std::map<int32_t, std::map<TileKey, AccumTileData>> binTileMaps;
    uint64_t allMoleculesCount = 0;
    std::map<int32_t, uint64_t> binMoleculeCounts;
    double geoMinX = std::numeric_limits<double>::infinity();
    double geoMinY = std::numeric_limits<double>::infinity();
    double geoMaxX = -std::numeric_limits<double>::infinity();
    double geoMaxY = -std::numeric_limits<double>::infinity();
};

// Add one record to the main and (optional) bin-specific tile
void append_annotated_row_to_state(AnnotatePmtilesState& state,
    const GeneBinInfo* geneBins,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const TileOperator::MltPmtilesOptions& mltOptions,
    const std::vector<uint32_t>& probBlockSizes,
    const std::vector<ExtColumnPlan>& extColumnPlans,
    const std::vector<ExtColumnValue>* extValues,
    size_t kCount, const std::string& featureName, uint32_t featureIdx,
    int32_t countValue, float x, float y, float z, bool hasZ,
    const TopProbs& probs)
{
    append_row_to_epsg3857_tile_map(state.allTileMap, schema,
        mltOptions.coordScale, static_cast<uint8_t>(mltOptions.zoom),
        hasZ, kCount, probBlockSizes,
        x, y, z, featureIdx, countValue,
        probs.ks, probs.ps, extColumnPlans, extValues,
        mltOptions.encode_prob_min, mltOptions.encode_prob_eps,
        state.geoMinX, state.geoMinY, state.geoMaxX, state.geoMaxY,
        state.allMoleculesCount);

    if (geneBins == nullptr) {
        return;
    }
    const auto it = geneBins->featureToBin.find(featureName);
    if (it == geneBins->featureToBin.end()) {
        warning("%s: feature '%s' is missing from the gene-bin map", __func__, featureName.c_str());
        return;
    }
    auto& binTileMap = state.binTileMaps[it->second];
    append_row_to_epsg3857_tile_map(binTileMap, schema,
        mltOptions.coordScale, static_cast<uint8_t>(mltOptions.zoom),
        hasZ, kCount, probBlockSizes,
        x, y, z, featureIdx, countValue,
        probs.ks, probs.ps, extColumnPlans, extValues,
        mltOptions.encode_prob_min, mltOptions.encode_prob_eps,
        state.geoMinX, state.geoMinY, state.geoMaxX, state.geoMaxY,
        state.binMoleculeCounts[it->second]);
}

template<typename T>
void append_scalar(std::vector<char>& out, const T& value) {
    const char* ptr = reinterpret_cast<const char*>(&value);
    out.insert(out.end(), ptr, ptr + sizeof(T));
}

template<typename T>
void append_array(std::vector<char>& out, const std::vector<T>& values) {
    if (values.empty()) {
        return;
    }
    const char* ptr = reinterpret_cast<const char*>(values.data());
    out.insert(out.end(), ptr, ptr + sizeof(T) * values.size());
}

std::vector<char> serialize_accum_tile_data(const AccumTileData& tile) {
    std::vector<char> out;
    const uint32_t rowCount = static_cast<uint32_t>(tile.size());
    append_scalar(out, rowCount);
    append_array(out, tile.localX);
    append_array(out, tile.localY);
    append_array(out, tile.featureCodes);
    append_array(out, tile.countValues);
    if (tile.hasZ) {
        append_array(out, tile.zValues);
    }
    for (size_t i = 0; i < tile.kValues.size(); ++i) {
        for (bool present : tile.kPresent[i]) {
            out.push_back(present ? 1 : 0);
        }
        for (bool present : tile.pPresent[i]) {
            out.push_back(present ? 1 : 0);
        }
        append_array(out, tile.kValues[i]);
        append_array(out, tile.pValues[i]);
    }
    const uint32_t extraCount = static_cast<uint32_t>(tile.extraColumns.size());
    append_scalar(out, extraCount);
    for (const auto& col : tile.extraColumns) {
        const uint32_t typeValue = static_cast<uint32_t>(col.type);
        append_scalar(out, typeValue);
        out.push_back(col.nullable ? 1 : 0);
        if (col.nullable) {
            if (col.present.size() != rowCount) {
                error("%s: nullable extra column present bitmap length mismatch", __func__);
            }
            for (bool present : col.present) {
                out.push_back(present ? 1 : 0);
            }
        }
        switch (col.type) {
        case mlt_pmtiles::ScalarType::INT_32:
            if (col.intValues.size() != rowCount) {
                error("%s: int extra column length mismatch", __func__);
            }
            append_array(out, col.intValues);
            break;
        case mlt_pmtiles::ScalarType::FLOAT:
            if (col.floatValues.size() != rowCount) {
                error("%s: float extra column length mismatch", __func__);
            }
            append_array(out, col.floatValues);
            break;
        case mlt_pmtiles::ScalarType::STRING:
            if (col.stringValues.size() != rowCount) {
                error("%s: string extra column length mismatch", __func__);
            }
            for (const auto& value : col.stringValues) {
                const uint32_t len = static_cast<uint32_t>(value.size());
                append_scalar(out, len);
                out.insert(out.end(), value.begin(), value.end());
            }
            break;
        default:
            error("%s: unsupported spilled extra column type", __func__);
        }
    }
    return out;
}

template<typename T>
const char* read_array_into(const char* ptr, const char* end, std::vector<T>& out, size_t n, const char* funcName) {
    const size_t bytes = sizeof(T) * n;
    if (static_cast<size_t>(end - ptr) < bytes) {
        error("%s: truncated spill fragment", funcName);
    }
    out.resize(n);
    if (bytes > 0) {
        std::memcpy(out.data(), ptr, bytes);
    }
    return ptr + bytes;
}

AccumTileData deserialize_accum_tile_data(const std::vector<char>& bytes, size_t kCount, bool hasZ,
    const std::vector<ExtColumnPlan>& extColumnPlans = {}) {
    const char* ptr = bytes.data();
    const char* end = ptr + bytes.size();
    if (static_cast<size_t>(end - ptr) < sizeof(uint32_t)) {
        error("%s: truncated spill fragment", __func__);
    }
    uint32_t rowCount = 0;
    std::memcpy(&rowCount, ptr, sizeof(rowCount));
    ptr += sizeof(rowCount);

    AccumTileData out(kCount, hasZ, extColumnPlans);
    ptr = read_array_into(ptr, end, out.localX, rowCount, __func__);
    ptr = read_array_into(ptr, end, out.localY, rowCount, __func__);
    ptr = read_array_into(ptr, end, out.featureCodes, rowCount, __func__);
    ptr = read_array_into(ptr, end, out.countValues, rowCount, __func__);
    if (hasZ) {
        ptr = read_array_into(ptr, end, out.zValues, rowCount, __func__);
    }
    for (size_t i = 0; i < kCount; ++i) {
        if (static_cast<size_t>(end - ptr) < rowCount) {
            error("%s: truncated K PRESENT bytes in spill fragment", __func__);
        }
        out.kPresent[i].reserve(rowCount);
        for (uint32_t j = 0; j < rowCount; ++j) {
            out.kPresent[i].push_back(ptr[j] != 0);
        }
        ptr += rowCount;
        if (static_cast<size_t>(end - ptr) < rowCount) {
            error("%s: truncated P PRESENT bytes in spill fragment", __func__);
        }
        out.pPresent[i].reserve(rowCount);
        for (uint32_t j = 0; j < rowCount; ++j) {
            out.pPresent[i].push_back(ptr[j] != 0);
        }
        ptr += rowCount;
        ptr = read_array_into(ptr, end, out.kValues[i], rowCount, __func__);
        ptr = read_array_into(ptr, end, out.pValues[i], rowCount, __func__);
    }
    if (static_cast<size_t>(end - ptr) < sizeof(uint32_t)) {
        error("%s: truncated extra-column count in spill fragment", __func__);
    }
    uint32_t extraCount = 0;
    std::memcpy(&extraCount, ptr, sizeof(extraCount));
    ptr += sizeof(extraCount);
    if (extraCount != out.extraColumns.size()) {
        error("%s: extra-column count mismatch in spill fragment (%u != %zu)",
            __func__, extraCount, out.extraColumns.size());
    }
    for (size_t i = 0; i < out.extraColumns.size(); ++i) {
        auto& col = out.extraColumns[i];
        if (static_cast<size_t>(end - ptr) < sizeof(uint32_t) + 1) {
            error("%s: truncated extra-column header in spill fragment", __func__);
        }
        uint32_t typeValue = 0;
        std::memcpy(&typeValue, ptr, sizeof(typeValue));
        ptr += sizeof(typeValue);
        const bool nullable = (*ptr++ != 0);
        if (typeValue != static_cast<uint32_t>(col.type) || nullable != col.nullable) {
            error("%s: extra-column schema mismatch in spill fragment", __func__);
        }
        if (col.nullable) {
            if (static_cast<size_t>(end - ptr) < rowCount) {
                error("%s: truncated extra-column present bitmap in spill fragment", __func__);
            }
            col.present.reserve(rowCount);
            for (uint32_t j = 0; j < rowCount; ++j) {
                col.present.push_back(ptr[j] != 0);
            }
            ptr += rowCount;
        }
        switch (col.type) {
        case mlt_pmtiles::ScalarType::INT_32:
            ptr = read_array_into(ptr, end, col.intValues, rowCount, __func__);
            break;
        case mlt_pmtiles::ScalarType::FLOAT:
            ptr = read_array_into(ptr, end, col.floatValues, rowCount, __func__);
            break;
        case mlt_pmtiles::ScalarType::STRING:
            col.stringValues.reserve(rowCount);
            for (uint32_t j = 0; j < rowCount; ++j) {
                if (static_cast<size_t>(end - ptr) < sizeof(uint32_t)) {
                    error("%s: truncated extra-column string length in spill fragment", __func__);
                }
                uint32_t len = 0;
                std::memcpy(&len, ptr, sizeof(len));
                ptr += sizeof(len);
                if (static_cast<size_t>(end - ptr) < len) {
                    error("%s: truncated extra-column string payload in spill fragment", __func__);
                }
                col.stringValues.emplace_back(ptr, ptr + len);
                ptr += len;
            }
            break;
        default:
            error("%s: unsupported spilled extra column type", __func__);
        }
    }
    if (ptr != end) {
        error("%s: unexpected trailing bytes in spill fragment", __func__);
    }
    return out;
}

mlt_pmtiles::EncodedTilePayload encode_accum_tile_payload(
    const TileKey& tileKey, uint8_t zoom,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const AccumTileData& data,
    const mlt_pmtiles::GlobalStringDictionary& featureDictionary) {
    const AccumTileData sorted = sort_tile_rows(data);
    const auto tile = build_point_tile_data(sorted);
    const std::string raw = mlt_pmtiles::encode_point_tile(schema, tile, &featureDictionary);
    mlt_pmtiles::EncodedTilePayload encoded;
    encoded.tileId = pmtiles::zxy_to_tileid(zoom, static_cast<uint32_t>(tileKey.col), static_cast<uint32_t>(tileKey.row));
    encoded.z = zoom;
    encoded.x = static_cast<uint32_t>(tileKey.col);
    encoded.y = static_cast<uint32_t>(tileKey.row);
    encoded.featureCount = static_cast<uint32_t>(tile.size());
    encoded.compressedData = mlt_pmtiles::gzip_compress(raw);
    return encoded;
}

void append_row_to_epsg3857_tile_map(std::map<TileKey, AccumTileData>& tileMap,
    const mlt_pmtiles::FeatureTableSchema& schema,
    double coordScale, uint8_t zoom,
    bool hasZ, size_t kCount,
    const std::vector<uint32_t>& probBlockSizes,
    float x, float y, float z, uint32_t featureIdx, int32_t countValue,
    const std::vector<int32_t>& ks, const std::vector<float>& ps,
    const std::vector<ExtColumnPlan>& extColumnPlans,
    const std::vector<ExtColumnValue>* extValues,
    double encodeProbMin, double encodeProbEps,
    double& geoMinX, double& geoMinY, double& geoMaxX, double& geoMaxY,
    uint64_t& totalRecordCount) {
    double scaledX = static_cast<double>(x);
    double scaledY = static_cast<double>(y);
    if (coordScale > 0) {
        scaledX *= coordScale;
        scaledY *= coordScale;
    }
    geoMinX = std::min(geoMinX, scaledX);
    geoMinY = std::min(geoMinY, scaledY);
    geoMaxX = std::max(geoMaxX, scaledX);
    geoMaxY = std::max(geoMaxY, scaledY);

    int64_t tx = 0;
    int64_t ty = 0;
    double tileLocalX = 0.0;
    double tileLocalY = 0.0;
    mlt_pmtiles::epsg3857_to_tilecoord(scaledX, scaledY, zoom, tx, ty, tileLocalX, tileLocalY);
    if (tx < std::numeric_limits<int32_t>::min() || tx > std::numeric_limits<int32_t>::max() ||
        ty < std::numeric_limits<int32_t>::min() || ty > std::numeric_limits<int32_t>::max()) {
        error("%s: output tile coordinate is out of int32 range", __func__);
    }
    TileKey tileKey{static_cast<int32_t>(ty), static_cast<int32_t>(tx)};
    int32_t localX = static_cast<int32_t>(std::llround(tileLocalX * static_cast<double>(schema.extent) / 256.0));
    int32_t localY = static_cast<int32_t>(std::llround(tileLocalY * static_cast<double>(schema.extent) / 256.0));
    localX = std::clamp(localX, 0, static_cast<int32_t>(schema.extent) - 1);
    localY = std::clamp(localY, 0, static_cast<int32_t>(schema.extent) - 1);

    auto it = tileMap.find(tileKey);
    if (it == tileMap.end()) {
        it = tileMap.emplace(tileKey, AccumTileData(kCount, hasZ, extColumnPlans)).first;
    }
    it->second.append(localX, localY, featureIdx, z, countValue,
        ks, ps, encodeProbMin, encodeProbEps, probBlockSizes, extValues);
    ++totalRecordCount;
}

std::vector<mlt_pmtiles::EncodedTilePayload> encode_epsg3857_tile_map(
    std::map<TileKey, AccumTileData>& tileMap, uint8_t zoom,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::GlobalStringDictionary& featureDictionary,
    int32_t threads) {
    std::vector<OutputTileInfo> outputTiles;
    outputTiles.reserve(tileMap.size());
    for (auto& kv : tileMap) {
        OutputTileInfo info;
        info.sourceKey = kv.first;
        info.z = zoom;
        info.x = static_cast<uint32_t>(kv.first.col);
        info.y = static_cast<uint32_t>(kv.first.row);
        info.tileId = pmtiles::zxy_to_tileid(info.z, info.x, info.y);
        info.data = &kv.second;
        outputTiles.push_back(info);
    }

    std::vector<mlt_pmtiles::EncodedTilePayload> encodedTiles(outputTiles.size());
    if (threads > 1 && outputTiles.size() > 1) {
        tbb::global_control gc(tbb::global_control::max_allowed_parallelism,
            static_cast<size_t>(threads));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, outputTiles.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    const auto& outTile = outputTiles[i];
                    encodedTiles[i] = encode_accum_tile_payload(
                        TileKey{static_cast<int32_t>(outTile.y), static_cast<int32_t>(outTile.x)},
                        outTile.z, schema, *outTile.data, featureDictionary);
                }
            });
    } else {
        for (size_t i = 0; i < outputTiles.size(); ++i) {
            const auto& outTile = outputTiles[i];
            encodedTiles[i] = encode_accum_tile_payload(
                TileKey{static_cast<int32_t>(outTile.y), static_cast<int32_t>(outTile.x)},
                outTile.z, schema, *outTile.data, featureDictionary);
        }
    }
    return encodedTiles;
}

void write_single_layer_pmtiles_archive(const std::string& outFile,
    const mlt_pmtiles::FeatureTableSchema& schema,
    std::vector<mlt_pmtiles::EncodedTilePayload> encodedTiles,
    uint64_t totalRecordCount, double coordScale,
    size_t featureDictionarySize, uint8_t outputZoom,
    double geoMinX, double geoMinY, double geoMaxX, double geoMaxY,
    const std::string& generator) {
    mlt_pmtiles::SingleLayerVectorPmtilesOptions options;
    options.schema = schema;
    options.geometryType = mlt_pmtiles::VectorGeometryType::Point;
    options.totalRecordCount = totalRecordCount;
    options.coordScale = coordScale;
    options.featureDictionarySize = featureDictionarySize;
    options.outputZoom = outputZoom;
    options.geoMinX = geoMinX;
    options.geoMinY = geoMinY;
    options.geoMaxX = geoMaxX;
    options.geoMaxY = geoMaxY;
    options.generator = generator;
    options.description = "Generated PMTiles by punkst for MLT points";
    mlt_pmtiles::write_single_layer_vector_pmtiles_archive(outFile, std::move(encodedTiles), options);
}

GeoTileRect source_tile_rect_epsg3857(const TileKey& sourceKey, int32_t sourceTileSize, double coordScale) {
    if (coordScale <= 0.0) {coordScale = 1.0;}
    const double x0 = static_cast<double>(sourceKey.col) * static_cast<double>(sourceTileSize) * coordScale;
    const double x1 = static_cast<double>(sourceKey.col + 1) * static_cast<double>(sourceTileSize) * coordScale;
    const double y0 = static_cast<double>(sourceKey.row) * static_cast<double>(sourceTileSize) * coordScale;
    const double y1 = static_cast<double>(sourceKey.row + 1) * static_cast<double>(sourceTileSize) * coordScale;
    GeoTileRect rect;
    rect.xmin = std::min(x0, x1);
    rect.xmax = std::max(x0, x1);
    rect.ymin = std::min(y0, y1);
    rect.ymax = std::max(y0, y1);
    return rect;
}

GeoTileRect destination_tile_rect_epsg3857(const TileKey& tileKey, uint8_t zoom) {
    double x0 = 0.0;
    double y0 = 0.0;
    double x1 = 0.0;
    double y1 = 0.0;
    mlt_pmtiles::tilecoord_to_epsg3857(tileKey.col, tileKey.row, 0.0, 0.0, zoom, x0, y0);
    mlt_pmtiles::tilecoord_to_epsg3857(tileKey.col, tileKey.row, 256.0, 256.0, zoom, x1, y1);
    GeoTileRect rect;
    rect.xmin = std::min(x0, x1);
    rect.xmax = std::max(x0, x1);
    rect.ymin = std::min(y0, y1);
    rect.ymax = std::max(y0, y1);
    return rect;
}

bool is_interior_destination_tile(const TileKey& tileKey, const GeoTileRect& sourceRect, uint8_t zoom) {
    const GeoTileRect dst = destination_tile_rect_epsg3857(tileKey, zoom);
    const double tol = std::max(1e-6, mlt_pmtiles::epsg3857_scale_factor(zoom));
    return dst.xmin > sourceRect.xmin + tol &&
           dst.xmax < sourceRect.xmax - tol &&
           dst.ymin > sourceRect.ymin + tol &&
           dst.ymax < sourceRect.ymax - tol;
}

void close_spill_shard(WorkerSpillShard& shard) {
    if (shard.fdData >= 0) {
        ::close(shard.fdData);
        shard.fdData = -1;
    }
}

void spill_accum_tile(WorkerSpillShard& shard, const TileKey& tileKey, const AccumTileData& tile,
    int32_t binId = -1) {
    std::vector<char> bytes = serialize_accum_tile_data(tile);
    if (!bytes.empty() && !write_all(shard.fdData, bytes.data(), bytes.size())) {
        error("%s: failed writing spill shard %s", __func__, shard.dataPath.c_str());
    }
    SpillFragmentIndex idx;
    idx.binId = binId;
    idx.tile = tileKey;
    idx.dataOffset = shard.dataSize;
    idx.dataBytes = bytes.size();
    idx.rowCount = static_cast<uint32_t>(tile.size());
    shard.dataSize += idx.dataBytes;
    shard.fragments.push_back(idx);
}

std::vector<mlt_pmtiles::EncodedTilePayload> run_epsg3857_parallel_pipeline(
    const std::string& dataFile, const std::vector<TileInfo>& blocks,
    const IndexHeader& formatInfo, bool hasZ,
    int32_t kValueCount, int32_t threadCount,
    const std::vector<uint32_t>& probBlockSizes,
    const std::function<void(std::ifstream&, float&, float&, float&, uint32_t&, std::vector<int32_t>&, std::vector<float>&)>& readRecord,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::GlobalStringDictionary& featureDictionary,
    const std::vector<std::string>& featureNames,
    double coordScale, double encodeProbMin, double encodeProbEps, uint8_t zoom,
    uint64_t& totalRecordCount,
    double& geoMinX, double& geoMinY, double& geoMaxX, double& geoMaxY) {
    if (formatInfo.tileSize <= 0) {
        error("%s: input tile size must be positive", __func__);
    }

    // Set up worker threads and spill shards
    const size_t workerCount = static_cast<size_t>(std::max(1, threadCount));
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    ScopedTempDir tmpDirScope(std::filesystem::temp_directory_path());
    std::vector<WorkerScanResult> workerResults(workerCount);
    for (size_t i = 0; i < workerCount; ++i) {
        auto& spill = workerResults[i].spill;
        spill.dataPath = (tmpDirScope.path / ("worker." + std::to_string(i) + ".mltspill.dat")).string();
        spill.fdData = ::open(spill.dataPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (spill.fdData < 0) {
            error("%s: failed opening spill shard %s", __func__, spill.dataPath.c_str());
        }
    }

    std::atomic<size_t> nextBlock{0};
    for (size_t workerId = 0; workerId < workerCount; ++workerId) {
        workers.emplace_back([&, workerId]() { // Worker thread function
            auto& result = workerResults[workerId];
            std::ifstream dataStream(dataFile, std::ios::binary);
            if (!dataStream.is_open()) {
                error("%s: cannot open binary input %s", __func__, dataFile.c_str());
            }

            for (;;) {
                // get one input tile
                const size_t blockIdx = nextBlock.fetch_add(1, std::memory_order_relaxed);
                if (blockIdx >= blocks.size()) {
                    break;
                }
                const auto& blk = blocks[blockIdx];
                const uint64_t len = blk.idx.ed - blk.idx.st;
                if (len == 0) {
                    continue;
                }
                if (formatInfo.recordSize == 0 || (len % formatInfo.recordSize) != 0) { // orrupted input
                    error("%s: block length %" PRIu64 " is inconsistent with record size %u", __func__, len, formatInfo.recordSize);
                }
                dataStream.clear();
                dataStream.seekg(static_cast<std::streamoff>(blk.idx.st));
                const uint64_t nRecords = len / formatInfo.recordSize;
                const TileKey sourceKey{blk.row, blk.col};
                const GeoTileRect sourceRect = source_tile_rect_epsg3857(sourceKey, formatInfo.tileSize, coordScale);

                std::unordered_map<TileKey, AccumTileData, TileKeyHash> localTiles; // fully contained within the source tile
                localTiles.reserve(64);
                for (uint64_t i = 0; i < nRecords; ++i) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    uint32_t featureIdx = 0;
                    std::vector<int32_t> ks;
                    std::vector<float> ps;
                    readRecord(dataStream, x, y, z, featureIdx, ks, ps);

                    if (featureIdx >= featureNames.size()) {
                        error("%s: feature index %u is out of range for dictionary size %zu",
                            __func__, featureIdx, featureNames.size());
                    }
                    double scaledX, scaledY;
                    if (coordScale > 0) {
                        scaledX = static_cast<double>(x) * coordScale;
                        scaledY = static_cast<double>(y) * coordScale;
                    } else {
                        scaledX = static_cast<double>(x);
                        scaledY = static_cast<double>(y);
                    }
                    result.geoMinX = std::min(result.geoMinX, scaledX);
                    result.geoMinY = std::min(result.geoMinY, scaledY);
                    result.geoMaxX = std::max(result.geoMaxX, scaledX);
                    result.geoMaxY = std::max(result.geoMaxY, scaledY);

                    int64_t tx = 0;
                    int64_t ty = 0;
                    double tileLocalX = 0.0;
                    double tileLocalY = 0.0;
                    mlt_pmtiles::epsg3857_to_tilecoord(scaledX, scaledY, zoom, tx, ty, tileLocalX, tileLocalY);
                    if (tx < std::numeric_limits<int32_t>::min() || tx > std::numeric_limits<int32_t>::max() ||
                        ty < std::numeric_limits<int32_t>::min() || ty > std::numeric_limits<int32_t>::max()) {
                        error("%s: output tile coordinate is out of int32 range", __func__);
                    }
                    TileKey tileKey{static_cast<int32_t>(ty), static_cast<int32_t>(tx)};
                    int32_t localX = static_cast<int32_t>(std::llround(tileLocalX * static_cast<double>(schema.extent) / 256.0));
                    int32_t localY = static_cast<int32_t>(std::llround(tileLocalY * static_cast<double>(schema.extent) / 256.0));
                    localX = std::clamp(localX, 0, static_cast<int32_t>(schema.extent) - 1);
                    localY = std::clamp(localY, 0, static_cast<int32_t>(schema.extent) - 1);

                    auto it = localTiles.find(tileKey);
                    if (it == localTiles.end()) {
                        it = localTiles.emplace(tileKey, AccumTileData(static_cast<size_t>(kValueCount), hasZ)).first;
                    }
                    it->second.append(localX, localY, featureIdx, z, 1, ks, ps,
                        encodeProbMin, encodeProbEps, probBlockSizes);
                    ++result.totalRecordCount;
                }

                for (const auto& kv : localTiles) {
                    if (is_interior_destination_tile(kv.first, sourceRect, zoom)) {
                        result.completedTiles.push_back(
                            encode_accum_tile_payload(kv.first, zoom, schema, kv.second, featureDictionary));
                    } else {
                        spill_accum_tile(result.spill, kv.first, kv.second);
                    }
                }
            }

            close_spill_shard(result.spill);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    totalRecordCount = 0;
    geoMinX = std::numeric_limits<double>::infinity();
    geoMinY = std::numeric_limits<double>::infinity();
    geoMaxX = -std::numeric_limits<double>::infinity();
    geoMaxY = -std::numeric_limits<double>::infinity();

    std::vector<mlt_pmtiles::EncodedTilePayload> encodedTiles;
    for (const auto& worker : workerResults) {
        totalRecordCount += worker.totalRecordCount;
        geoMinX = std::min(geoMinX, worker.geoMinX);
        geoMinY = std::min(geoMinY, worker.geoMinY);
        geoMaxX = std::max(geoMaxX, worker.geoMaxX);
        geoMaxY = std::max(geoMaxY, worker.geoMaxY);
        encodedTiles.insert(encodedTiles.end(), worker.completedTiles.begin(), worker.completedTiles.end());
    }

    std::map<TileKey, std::vector<SpillFragmentRef>> groupedFragments;
    for (size_t shardId = 0; shardId < workerResults.size(); ++shardId) {
        for (const auto& fragment : workerResults[shardId].spill.fragments) {
            groupedFragments[fragment.tile].push_back(
                SpillFragmentRef{-1, shardId, fragment.dataOffset, fragment.dataBytes, fragment.rowCount});
        }
    }
    if (groupedFragments.empty()) {
        return encodedTiles;
    }

    std::vector<std::pair<TileKey, std::vector<SpillFragmentRef>>> mergeTasks;
    mergeTasks.reserve(groupedFragments.size());
    for (auto& kv : groupedFragments) {
        mergeTasks.emplace_back(kv.first, std::move(kv.second));
    }

    std::vector<std::vector<mlt_pmtiles::EncodedTilePayload>> mergeOutputs(workerCount);
    std::atomic<size_t> nextTask{0};
    workers.clear();
    workers.reserve(workerCount);
    for (size_t workerId = 0; workerId < workerCount; ++workerId) {
        workers.emplace_back([&, workerId]() {
            std::vector<std::ifstream> spillStreams(workerResults.size());
            for (size_t shardId = 0; shardId < workerResults.size(); ++shardId) {
                if (!workerResults[shardId].spill.fragments.empty()) {
                    spillStreams[shardId].open(workerResults[shardId].spill.dataPath, std::ios::binary);
                    if (!spillStreams[shardId].is_open()) {
                        error("%s: failed opening spill shard %s", __func__, workerResults[shardId].spill.dataPath.c_str());
                    }
                }
            }

            auto& out = mergeOutputs[workerId];
            for (;;) {
                const size_t taskIdx = nextTask.fetch_add(1, std::memory_order_relaxed);
                if (taskIdx >= mergeTasks.size()) {
                    break;
                }
                const TileKey tileKey = mergeTasks[taskIdx].first;
                const auto& fragments = mergeTasks[taskIdx].second;
                AccumTileData merged(static_cast<size_t>(kValueCount), hasZ);
                for (const auto& fragment : fragments) {
                    std::ifstream& in = spillStreams[fragment.shardId];
                    in.clear();
                    in.seekg(static_cast<std::streamoff>(fragment.dataOffset));
                    std::vector<char> buf(fragment.dataBytes);
                    if (!buf.empty()) {
                        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
                        if (in.gcount() != static_cast<std::streamsize>(buf.size())) {
                            error("%s: failed reading spill fragment", __func__);
                        }
                    }
                    const AccumTileData part = deserialize_accum_tile_data(buf, static_cast<size_t>(kValueCount), hasZ);
                    merged.appendFrom(part);
                }
                out.push_back(encode_accum_tile_payload(tileKey, zoom, schema, merged, featureDictionary));
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    for (const auto& vec : mergeOutputs) {
        encodedTiles.insert(encodedTiles.end(), vec.begin(), vec.end());
    }
    return encodedTiles;
}

GeneBinPipelineResult run_epsg3857_parallel_gene_bin_pipeline(
    const std::string& dataFile, const std::vector<TileInfo>& blocks,
    const IndexHeader& formatInfo, bool hasZ,
    int32_t kValueCount, int32_t threadCount,
    const std::vector<uint32_t>& probBlockSizes,
    const std::function<void(std::ifstream&, float&, float&, float&, uint32_t&, std::vector<int32_t>&, std::vector<float>&)>& readRecord,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::GlobalStringDictionary& featureDictionary,
    const std::vector<std::string>& featureNames,
    const GeneBinInfo& geneBins,
    double coordScale, double encodeProbMin, double encodeProbEps, uint8_t zoom) {
    if (formatInfo.tileSize <= 0) {
        error("%s: input tile size must be positive", __func__);
    }

    const size_t workerCount = static_cast<size_t>(std::max(1, threadCount));
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    ScopedTempDir tmpDirScope(std::filesystem::temp_directory_path());
    std::vector<GeneBinWorkerScanResult> workerResults(workerCount);
    for (size_t i = 0; i < workerCount; ++i) {
        auto& spill = workerResults[i].spill;
        spill.dataPath = (tmpDirScope.path / ("worker." + std::to_string(i) + ".mltspill.dat")).string();
        spill.fdData = ::open(spill.dataPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (spill.fdData < 0) {
            error("%s: failed opening spill shard %s", __func__, spill.dataPath.c_str());
        }
    }

    std::atomic<size_t> nextBlock{0};
    for (size_t workerId = 0; workerId < workerCount; ++workerId) {
        workers.emplace_back([&, workerId]() {
            auto& result = workerResults[workerId];
            std::ifstream dataStream(dataFile, std::ios::binary);
            if (!dataStream.is_open()) {
                error("%s: cannot open binary input %s", __func__, dataFile.c_str());
            }

            for (;;) {
                const size_t blockIdx = nextBlock.fetch_add(1, std::memory_order_relaxed);
                if (blockIdx >= blocks.size()) {
                    break;
                }
                const auto& blk = blocks[blockIdx];
                const uint64_t len = blk.idx.ed - blk.idx.st;
                if (len == 0) {
                    continue;
                }
                if (formatInfo.recordSize == 0 || (len % formatInfo.recordSize) != 0) {
                    error("%s: block length %" PRIu64 " is inconsistent with record size %u", __func__, len, formatInfo.recordSize);
                }
                dataStream.clear();
                dataStream.seekg(static_cast<std::streamoff>(blk.idx.st));
                const uint64_t nRecords = len / formatInfo.recordSize;
                const TileKey sourceKey{blk.row, blk.col};
                const GeoTileRect sourceRect = source_tile_rect_epsg3857(sourceKey, formatInfo.tileSize, coordScale);

                std::unordered_map<TileKey, AccumTileData, TileKeyHash> localAllTiles;
                std::unordered_map<BinTileKey, AccumTileData, BinTileKeyHash> localBinTiles;
                localAllTiles.reserve(64);
                localBinTiles.reserve(64);
                for (uint64_t i = 0; i < nRecords; ++i) {
                    float x = 0.0f;
                    float y = 0.0f;
                    float z = 0.0f;
                    uint32_t featureIdx = 0;
                    std::vector<int32_t> ks;
                    std::vector<float> ps;
                    readRecord(dataStream, x, y, z, featureIdx, ks, ps);

                    if (featureIdx >= featureNames.size()) {
                        error("%s: feature index %u is out of range for dictionary size %zu",
                            __func__, featureIdx, featureNames.size());
                    }

                    double scaledX = static_cast<double>(x);
                    double scaledY = static_cast<double>(y);
                    if (coordScale > 0) {
                        scaledX *= coordScale;
                        scaledY *= coordScale;
                    }
                    result.geoMinX = std::min(result.geoMinX, scaledX);
                    result.geoMinY = std::min(result.geoMinY, scaledY);
                    result.geoMaxX = std::max(result.geoMaxX, scaledX);
                    result.geoMaxY = std::max(result.geoMaxY, scaledY);

                    int64_t tx = 0;
                    int64_t ty = 0;
                    double tileLocalX = 0.0;
                    double tileLocalY = 0.0;
                    mlt_pmtiles::epsg3857_to_tilecoord(scaledX, scaledY, zoom, tx, ty, tileLocalX, tileLocalY);
                    if (tx < std::numeric_limits<int32_t>::min() || tx > std::numeric_limits<int32_t>::max() ||
                        ty < std::numeric_limits<int32_t>::min() || ty > std::numeric_limits<int32_t>::max()) {
                        error("%s: output tile coordinate is out of int32 range", __func__);
                    }
                    TileKey tileKey{static_cast<int32_t>(ty), static_cast<int32_t>(tx)};
                    int32_t localX = static_cast<int32_t>(std::llround(tileLocalX * static_cast<double>(schema.extent) / 256.0));
                    int32_t localY = static_cast<int32_t>(std::llround(tileLocalY * static_cast<double>(schema.extent) / 256.0));
                    localX = std::clamp(localX, 0, static_cast<int32_t>(schema.extent) - 1);
                    localY = std::clamp(localY, 0, static_cast<int32_t>(schema.extent) - 1);

                    auto allIt = localAllTiles.find(tileKey);
                    if (allIt == localAllTiles.end()) {
                        allIt = localAllTiles.emplace(tileKey, AccumTileData(static_cast<size_t>(kValueCount), hasZ)).first;
                    }
                    allIt->second.append(localX, localY, featureIdx, z, 1, ks, ps,
                        encodeProbMin, encodeProbEps, probBlockSizes);
                    ++result.totalRecordCount;

                    const auto binIt = geneBins.featureToBin.find(featureNames[featureIdx]);
                    if (binIt == geneBins.featureToBin.end()) {
                        result.missingBinFeatures.insert(featureIdx);
                        continue;
                    }
                    BinTileKey binTileKey;
                    binTileKey.binId = binIt->second;
                    binTileKey.tile = tileKey;
                    auto localBinIt = localBinTiles.find(binTileKey);
                    if (localBinIt == localBinTiles.end()) {
                        localBinIt = localBinTiles.emplace(
                            binTileKey, AccumTileData(static_cast<size_t>(kValueCount), hasZ)).first;
                    }
                    localBinIt->second.append(localX, localY, featureIdx, z, 1, ks, ps,
                        encodeProbMin, encodeProbEps, probBlockSizes);
                    ++result.binRecordCounts[binTileKey.binId];
                }

                for (const auto& kv : localAllTiles) {
                    if (is_interior_destination_tile(kv.first, sourceRect, zoom)) {
                        result.completedAllTiles.push_back(
                            encode_accum_tile_payload(kv.first, zoom, schema, kv.second, featureDictionary));
                    } else {
                        spill_accum_tile(result.spill, kv.first, kv.second);
                    }
                }
                for (const auto& kv : localBinTiles) {
                    if (is_interior_destination_tile(kv.first.tile, sourceRect, zoom)) {
                        result.completedBinTiles[kv.first.binId].push_back(
                            encode_accum_tile_payload(kv.first.tile, zoom, schema, kv.second, featureDictionary));
                    } else {
                        spill_accum_tile(result.spill, kv.first.tile, kv.second, kv.first.binId);
                    }
                }
            }

            close_spill_shard(result.spill);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    GeneBinPipelineResult pipeline;
    for (const auto& worker : workerResults) {
        pipeline.totalRecordCount += worker.totalRecordCount;
        pipeline.geoMinX = std::min(pipeline.geoMinX, worker.geoMinX);
        pipeline.geoMinY = std::min(pipeline.geoMinY, worker.geoMinY);
        pipeline.geoMaxX = std::max(pipeline.geoMaxX, worker.geoMaxX);
        pipeline.geoMaxY = std::max(pipeline.geoMaxY, worker.geoMaxY);
        pipeline.allEncodedTiles.insert(pipeline.allEncodedTiles.end(),
            worker.completedAllTiles.begin(), worker.completedAllTiles.end());
        for (const auto& kv : worker.completedBinTiles) {
            auto& dst = pipeline.binEncodedTiles[kv.first];
            dst.insert(dst.end(), kv.second.begin(), kv.second.end());
        }
        for (const auto& kv : worker.binRecordCounts) {
            pipeline.binRecordCounts[kv.first] += kv.second;
        }
        pipeline.missingBinFeatures.insert(
            worker.missingBinFeatures.begin(), worker.missingBinFeatures.end());
    }

    std::map<BinTileKey, std::vector<SpillFragmentRef>> groupedFragments;
    for (size_t shardId = 0; shardId < workerResults.size(); ++shardId) {
        for (const auto& fragment : workerResults[shardId].spill.fragments) {
            groupedFragments[BinTileKey{fragment.binId, fragment.tile}].push_back(
                SpillFragmentRef{fragment.binId, shardId, fragment.dataOffset, fragment.dataBytes, fragment.rowCount});
        }
    }
    if (groupedFragments.empty()) {
        return pipeline;
    }

    std::vector<std::pair<BinTileKey, std::vector<SpillFragmentRef>>> mergeTasks;
    mergeTasks.reserve(groupedFragments.size());
    for (auto& kv : groupedFragments) {
        mergeTasks.emplace_back(kv.first, std::move(kv.second));
    }

    std::vector<GeneBinPipelineResult> mergeOutputs(workerCount);
    std::atomic<size_t> nextTask{0};
    workers.clear();
    workers.reserve(workerCount);
    for (size_t workerId = 0; workerId < workerCount; ++workerId) {
        workers.emplace_back([&, workerId]() {
            std::vector<std::ifstream> spillStreams(workerResults.size());
            for (size_t shardId = 0; shardId < workerResults.size(); ++shardId) {
                if (!workerResults[shardId].spill.fragments.empty()) {
                    spillStreams[shardId].open(workerResults[shardId].spill.dataPath, std::ios::binary);
                    if (!spillStreams[shardId].is_open()) {
                        error("%s: failed opening spill shard %s", __func__, workerResults[shardId].spill.dataPath.c_str());
                    }
                }
            }

            auto& out = mergeOutputs[workerId];
            for (;;) {
                const size_t taskIdx = nextTask.fetch_add(1, std::memory_order_relaxed);
                if (taskIdx >= mergeTasks.size()) {
                    break;
                }
                const BinTileKey tileKey = mergeTasks[taskIdx].first;
                const auto& fragments = mergeTasks[taskIdx].second;
                AccumTileData merged(static_cast<size_t>(kValueCount), hasZ);
                for (const auto& fragment : fragments) {
                    std::ifstream& in = spillStreams[fragment.shardId];
                    in.clear();
                    in.seekg(static_cast<std::streamoff>(fragment.dataOffset));
                    std::vector<char> buf(fragment.dataBytes);
                    if (!buf.empty()) {
                        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
                        if (in.gcount() != static_cast<std::streamsize>(buf.size())) {
                            error("%s: failed reading spill fragment", __func__);
                        }
                    }
                    const AccumTileData part = deserialize_accum_tile_data(buf, static_cast<size_t>(kValueCount), hasZ);
                    merged.appendFrom(part);
                }
                auto encoded = encode_accum_tile_payload(tileKey.tile, zoom, schema, merged, featureDictionary);
                if (tileKey.binId < 0) {
                    out.allEncodedTiles.push_back(std::move(encoded));
                } else {
                    out.binEncodedTiles[tileKey.binId].push_back(std::move(encoded));
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    for (const auto& part : mergeOutputs) {
        pipeline.allEncodedTiles.insert(pipeline.allEncodedTiles.end(),
            part.allEncodedTiles.begin(), part.allEncodedTiles.end());
        for (const auto& kv : part.binEncodedTiles) {
            auto& dst = pipeline.binEncodedTiles[kv.first];
            dst.insert(dst.end(), kv.second.begin(), kv.second.end());
        }
    }
    return pipeline;
}

template<typename MakeWorkerStateFn, typename ProcessQueryTileFn>
AnnotatePmtilesPipelineResult run_annotate_epsg3857_parallel_pipeline(
    const std::vector<TileKey>& queryTiles,
    int32_t sourceTileSize, int32_t threadCount,
    size_t kCount, bool hasZ,
    const mlt_pmtiles::FeatureTableSchema& schema,
    const mlt_pmtiles::GlobalStringDictionary& featureDictionary,
    const std::vector<ExtColumnPlan>& extColumnPlans,
    double coordScale, uint8_t zoom,
    MakeWorkerStateFn&& makeWorkerState,
    ProcessQueryTileFn&& processQueryTile) {
    if (sourceTileSize <= 0) {
        error("%s: input tile size must be positive", __func__);
    }

    const size_t workerCount = static_cast<size_t>(std::max(1, threadCount));
    std::vector<std::thread> workers;
    workers.reserve(workerCount);
    ScopedTempDir tmpDirScope(std::filesystem::temp_directory_path());
    std::vector<AnnotatePmtilesWorkerResult> workerResults(workerCount);
    for (size_t i = 0; i < workerCount; ++i) {
        auto& spill = workerResults[i].spill;
        spill.dataPath = (tmpDirScope.path / ("annotate.worker." + std::to_string(i) + ".mltspill.dat")).string();
        spill.fdData = ::open(spill.dataPath.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (spill.fdData < 0) {
            error("%s: failed opening spill shard %s", __func__, spill.dataPath.c_str());
        }
    }

    std::atomic<size_t> nextTile{0};
    for (size_t workerId = 0; workerId < workerCount; ++workerId) {
        workers.emplace_back([&, workerId]() {
            auto workerState = makeWorkerState();
            auto& result = workerResults[workerId];
            for (;;) {
                const size_t tileIdx = nextTile.fetch_add(1, std::memory_order_relaxed);
                if (tileIdx >= queryTiles.size()) {
                    break;
                }
                const TileKey queryTile = queryTiles[tileIdx];
                AnnotatePmtilesState tileState;
                processQueryTile(queryTile, workerState, tileState, result.missingPackagingFeatures);
                result.totalRecordCount += tileState.allMoleculesCount;
                result.geoMinX = std::min(result.geoMinX, tileState.geoMinX);
                result.geoMinY = std::min(result.geoMinY, tileState.geoMinY);
                result.geoMaxX = std::max(result.geoMaxX, tileState.geoMaxX);
                result.geoMaxY = std::max(result.geoMaxY, tileState.geoMaxY);
                for (const auto& kv : tileState.binMoleculeCounts) {
                    result.binRecordCounts[kv.first] += kv.second;
                }

                const GeoTileRect sourceRect = source_tile_rect_epsg3857(queryTile, sourceTileSize, coordScale);
                for (const auto& kv : tileState.allTileMap) {
                    if (is_interior_destination_tile(kv.first, sourceRect, zoom)) {
                        result.completedAllTiles.push_back(
                            encode_accum_tile_payload(kv.first, zoom, schema, kv.second, featureDictionary));
                    } else {
                        spill_accum_tile(result.spill, kv.first, kv.second);
                    }
                }
                for (const auto& binKv : tileState.binTileMaps) {
                    for (const auto& tileKv : binKv.second) {
                        if (is_interior_destination_tile(tileKv.first, sourceRect, zoom)) {
                            result.completedBinTiles[binKv.first].push_back(
                                encode_accum_tile_payload(tileKv.first, zoom, schema, tileKv.second, featureDictionary));
                        } else {
                            spill_accum_tile(result.spill, tileKv.first, tileKv.second, binKv.first);
                        }
                    }
                }
            }

            close_spill_shard(result.spill);
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    AnnotatePmtilesPipelineResult pipeline;
    for (const auto& worker : workerResults) {
        pipeline.totalRecordCount += worker.totalRecordCount;
        pipeline.geoMinX = std::min(pipeline.geoMinX, worker.geoMinX);
        pipeline.geoMinY = std::min(pipeline.geoMinY, worker.geoMinY);
        pipeline.geoMaxX = std::max(pipeline.geoMaxX, worker.geoMaxX);
        pipeline.geoMaxY = std::max(pipeline.geoMaxY, worker.geoMaxY);
        pipeline.allEncodedTiles.insert(pipeline.allEncodedTiles.end(),
            worker.completedAllTiles.begin(), worker.completedAllTiles.end());
        for (const auto& kv : worker.completedBinTiles) {
            auto& dst = pipeline.binEncodedTiles[kv.first];
            dst.insert(dst.end(), kv.second.begin(), kv.second.end());
        }
        for (const auto& kv : worker.binRecordCounts) {
            pipeline.binRecordCounts[kv.first] += kv.second;
        }
        pipeline.missingPackagingFeatures.insert(
            worker.missingPackagingFeatures.begin(), worker.missingPackagingFeatures.end());
    }

    std::map<BinTileKey, std::vector<SpillFragmentRef>> groupedFragments;
    for (size_t shardId = 0; shardId < workerResults.size(); ++shardId) {
        for (const auto& fragment : workerResults[shardId].spill.fragments) {
            groupedFragments[BinTileKey{fragment.binId, fragment.tile}].push_back(
                SpillFragmentRef{fragment.binId, shardId, fragment.dataOffset, fragment.dataBytes, fragment.rowCount});
        }
    }
    if (groupedFragments.empty()) {
        return pipeline;
    }

    std::vector<std::pair<BinTileKey, std::vector<SpillFragmentRef>>> mergeTasks;
    mergeTasks.reserve(groupedFragments.size());
    for (auto& kv : groupedFragments) {
        mergeTasks.emplace_back(kv.first, std::move(kv.second));
    }

    std::vector<AnnotatePmtilesPipelineResult> mergeOutputs(workerCount);
    std::atomic<size_t> nextTask{0};
    workers.clear();
    workers.reserve(workerCount);
    for (size_t workerId = 0; workerId < workerCount; ++workerId) {
        workers.emplace_back([&, workerId]() {
            std::vector<std::ifstream> spillStreams(workerResults.size());
            for (size_t shardId = 0; shardId < workerResults.size(); ++shardId) {
                if (!workerResults[shardId].spill.fragments.empty()) {
                    spillStreams[shardId].open(workerResults[shardId].spill.dataPath, std::ios::binary);
                    if (!spillStreams[shardId].is_open()) {
                        error("%s: failed opening spill shard %s", __func__, workerResults[shardId].spill.dataPath.c_str());
                    }
                }
            }

            auto& out = mergeOutputs[workerId];
            for (;;) {
                const size_t taskIdx = nextTask.fetch_add(1, std::memory_order_relaxed);
                if (taskIdx >= mergeTasks.size()) {
                    break;
                }
                const BinTileKey tileKey = mergeTasks[taskIdx].first;
                const auto& fragments = mergeTasks[taskIdx].second;
                AccumTileData merged(kCount, hasZ, extColumnPlans);
                for (const auto& fragment : fragments) {
                    std::ifstream& in = spillStreams[fragment.shardId];
                    in.clear();
                    in.seekg(static_cast<std::streamoff>(fragment.dataOffset));
                    std::vector<char> buf(fragment.dataBytes);
                    if (!buf.empty()) {
                        in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
                        if (in.gcount() != static_cast<std::streamsize>(buf.size())) {
                            error("%s: failed reading spill fragment", __func__);
                        }
                    }
                    const AccumTileData part = deserialize_accum_tile_data(
                        buf, kCount, hasZ, extColumnPlans);
                    merged.appendFrom(part);
                }
                auto encoded = encode_accum_tile_payload(tileKey.tile, zoom, schema, merged, featureDictionary);
                if (tileKey.binId < 0) {
                    out.allEncodedTiles.push_back(std::move(encoded));
                } else {
                    out.binEncodedTiles[tileKey.binId].push_back(std::move(encoded));
                }
            }
        });
    }
    for (auto& worker : workers) {
        worker.join();
    }

    for (const auto& part : mergeOutputs) {
        pipeline.allEncodedTiles.insert(pipeline.allEncodedTiles.end(),
            part.allEncodedTiles.begin(), part.allEncodedTiles.end());
        for (const auto& kv : part.binEncodedTiles) {
            auto& dst = pipeline.binEncodedTiles[kv.first];
            dst.insert(dst.end(), kv.second.begin(), kv.second.end());
        }
    }
    return pipeline;
}

} // namespace

void TileOperator::exportPMTiles(const std::string& pmtilesFile,
    const std::string& outPrefix,
    const ExportPmtilesOptions& options) {
    if (outPrefix.empty() || outPrefix == "-") {
        error("%s: PMTiles export requires a concrete output prefix", __func__);
    }
    if (options.tileSize <= 0) {
        error("%s: PMTiles export requires a positive --tile-size", __func__);
    }
    const mlt_pmtiles::LoadedPmtilesArchive archive =
        mlt_pmtiles::load_pmtiles_archive(pmtilesFile);
    if (archive.header.tile_type != pmtiles::TILETYPE_MLT) {
        error("%s: expected PMTiles tile type MLT (0x06), got %u", __func__,
            static_cast<unsigned>(archive.header.tile_type));
    }
    if (archive.entries.empty()) {
        error("%s: PMTiles archive %s has no tile entries", __func__, pmtilesFile.c_str());
    }
    if (archive.metadata.contains("coordinate_mode")) {
        const std::string coordinateMode = archive.metadata["coordinate_mode"].get<std::string>();
        if (coordinateMode != "epsg3857") {
            error("%s: unsupported PMTiles coordinate_mode '%s'", __func__, coordinateMode.c_str());
        }
    }
    const double coordScale = export_coord_scale_from_metadata(archive.metadata);
    ExportRegionFilter regionFilter = build_export_region_filter(options);
    if (regionFilter.hasPolygon) {
        notice("Prepared PMTiles export region loaded from %s: bounding box [%.2f, %.2f, %.2f, %.2f], %lu union paths, %lu component bboxes",
            options.geojsonFile.c_str(),
            regionFilter.polygon.bbox_f.xmin, regionFilter.polygon.bbox_f.ymin,
            regionFilter.polygon.bbox_f.xmax, regionFilter.polygon.bbox_f.ymax,
            regionFilter.polygon.union_paths.size(), regionFilter.polygon.comp_bbox.size());
        if (regionFilter.polygon.empty()) {
            warning("%s: prepared export region is empty", __func__);
            return;
        }
    }

    bool haveSchema = false;
    bool wroteAny = false;
    mlt_pmtiles::FeatureTableSchema referenceSchema;
    std::vector<ExportColumnPlan> columnPlan;
    std::vector<std::string> headerColumns;
    std::map<TileKey, ExportTextTile> tileBuckets;
    Rectangle<float> writtenBox;
    size_t zSchemaIndex = 0;
    bool hasZColumn = false;
    size_t candidateEntryCount = 0;
    const uint8_t exportZoom = archive.header.max_zoom;
    const auto firstMaxZoomIt = std::find_if(archive.entries.begin(), archive.entries.end(),
        [&](const pmtiles::entry_zxy& entry) {
            return entry.z == exportZoom;
        });
    if (firstMaxZoomIt == archive.entries.end()) {
        error("%s: PMTiles archive %s has no entries at max zoom %u",
            __func__, pmtilesFile.c_str(), static_cast<unsigned>(exportZoom));
    }

    for (const auto& entry : archive.entries) {
        if (entry.z != exportZoom) {
            continue;
        }
        const Rectangle<float> entryRect =
            geo_tile_rect_to_coord_rect(pmtiles_entry_rect_epsg3857(entry), coordScale);
        if (export_region_requested(options) &&
            !export_entry_matches_region(entryRect, regionFilter)) {
            continue;
        }
        ++candidateEntryCount;
        const std::string rawTile =
            mlt_pmtiles::read_pmtiles_tile_payload(*archive.reader, archive.header, entry);
        const mlt_pmtiles::DecodedPointTile decoded =
            mlt_pmtiles::decode_point_tile(rawTile);
        if (!haveSchema) {
            referenceSchema = decoded.schema;
            columnPlan = build_export_column_plan(referenceSchema, headerColumns);
            zSchemaIndex = find_schema_column_index(referenceSchema, "z");
            hasZColumn = (zSchemaIndex < referenceSchema.columns.size());
            if (regionFilter.hasZRange && !hasZColumn) {
                error("%s: z-range filtering requires exported PMTiles rows with a z column", __func__);
            }
            haveSchema = true;
        } else {
            validate_export_schema_compatible(referenceSchema, decoded.schema);
        }

        const size_t rowCount = decoded.tile.size();
        const double localScale = (decoded.schema.extent > 0)
            ? (256.0 / static_cast<double>(decoded.schema.extent))
            : 1.0;
        for (size_t row = 0; row < rowCount; ++row) {
            double scaledX = 0.0;
            double scaledY = 0.0;
            mlt_pmtiles::tilecoord_to_epsg3857(entry.x, entry.y,
                static_cast<double>(decoded.tile.localX[row]) * localScale,
                static_cast<double>(decoded.tile.localY[row]) * localScale,
                entry.z, scaledX, scaledY);
            const double x = scaledX / coordScale;
            const double y = scaledY / coordScale;
            const float xf = static_cast<float>(x);
            const float yf = static_cast<float>(y);
            const TileKey outTile = pt2tile(x, y, options.tileSize);
            if (!export_row_matches_spatial_region(xf, yf, outTile, regionFilter)) {
                continue;
            }
            if (regionFilter.hasZRange) {
                const float z = export_z_value_at_row(
                    referenceSchema.columns[zSchemaIndex],
                    decoded.tile.columns[zSchemaIndex], row);
                if (z < regionFilter.zmin || z >= regionFilter.zmax) {
                    continue;
                }
            }
            auto [it, inserted] = tileBuckets.emplace(outTile, ExportTextTile(outTile.row, outTile.col));
            (void)inserted;
            ExportTextTile& bucket = it->second;

            std::string line = fp_to_string(x, options.coordDigits);
            line.push_back('\t');
            line += fp_to_string(y, options.coordDigits);
            for (const auto& plan : columnPlan) {
                line.push_back('\t');
                line += format_export_column_value(
                    referenceSchema.columns[plan.schemaIndex], decoded.tile,
                    decoded.tile.columns[plan.schemaIndex], row,
                    plan, options);
            }
            line.push_back('\n');
            bucket.text += line;
            ++bucket.entry.n;
            bucket.entry.extendToInclude(xf, yf);
            if (!wroteAny) {
                writtenBox = Rectangle<float>(xf, yf, xf, yf);
                wroteAny = true;
            } else {
                writtenBox.extendToInclude(xf, yf);
            }
        }
    }

    if (!haveSchema) {
        const std::string rawTile =
            mlt_pmtiles::read_pmtiles_tile_payload(*archive.reader, archive.header, *firstMaxZoomIt);
        const mlt_pmtiles::DecodedPointTile decoded =
            mlt_pmtiles::decode_point_tile(rawTile);
        referenceSchema = decoded.schema;
        columnPlan = build_export_column_plan(referenceSchema, headerColumns);
        zSchemaIndex = find_schema_column_index(referenceSchema, "z");
        hasZColumn = (zSchemaIndex < referenceSchema.columns.size());
        if (regionFilter.hasZRange && !hasZColumn) {
            error("%s: z-range filtering requires exported PMTiles rows with a z column", __func__);
        }
        haveSchema = true;
    }

    const std::string tsvFile = outPrefix + ".tsv";
    const std::string indexFile = outPrefix + ".index";
    FILE* fp = std::fopen(tsvFile.c_str(), "w");
    if (!fp) {
        error("%s: cannot open TSV output %s", __func__, tsvFile.c_str());
    }
    const int fdIndex = open(indexFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) {
        std::fclose(fp);
        error("%s: cannot open index output %s", __func__, indexFile.c_str());
    }

    std::string header = "#";
    for (size_t i = 0; i < headerColumns.size(); ++i) {
        if (i > 0) {
            header.push_back('\t');
        }
        header += headerColumns[i];
    }
    header.push_back('\n');
    if (std::fprintf(fp, "%s", header.c_str()) < 0) {
        std::fclose(fp);
        close(fdIndex);
        error("%s: failed writing TSV header", __func__);
    }

    IndexHeader idxHeader;
    idxHeader.magic = PUNKST_INDEX_MAGIC;
    idxHeader.mode = 0;
    if (std::find(headerColumns.begin(), headerColumns.end(), "z") != headerColumns.end()) {
        idxHeader.mode |= 0x10u;
    }
    idxHeader.tileSize = options.tileSize;
    idxHeader.pixelResolution = -1.0f;
    idxHeader.pixelResolutionZ = -1.0f;
    idxHeader.recordSize = 0;
    idxHeader.featureCount = 0;
    idxHeader.featureNameSize = 0;
    const std::vector<uint32_t> exportKvec = infer_export_kvec(headerColumns);
    idxHeader.packKvec(exportKvec);
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
        std::fclose(fp);
        close(fdIndex);
        error("%s: failed writing index header", __func__);
    }

    long currentOffset = std::ftell(fp);
    for (auto& kv : tileBuckets) {
        ExportTextTile& bucket = kv.second;
        bucket.entry.st = static_cast<uint64_t>(currentOffset);
        if (!bucket.text.empty()) {
            if (std::fwrite(bucket.text.data(), 1, bucket.text.size(), fp) != bucket.text.size()) {
                std::fclose(fp);
                close(fdIndex);
                error("%s: failed writing TSV tile bucket", __func__);
            }
        }
        currentOffset = std::ftell(fp);
        bucket.entry.ed = static_cast<uint64_t>(currentOffset);
        if (!write_all(fdIndex, &bucket.entry, sizeof(bucket.entry))) {
            std::fclose(fp);
            close(fdIndex);
            error("%s: failed writing index entry", __func__);
        }
    }

    if (wroteAny) {
        idxHeader.xmin = writtenBox.xmin;
        idxHeader.xmax = writtenBox.xmax;
        idxHeader.ymin = writtenBox.ymin;
        idxHeader.ymax = writtenBox.ymax;
        if (lseek(fdIndex, 0, SEEK_SET) < 0) {
            std::fclose(fp);
            close(fdIndex);
            error("%s: failed seeking index header for finalization", __func__);
        }
        if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
            std::fclose(fp);
            close(fdIndex);
            error("%s: failed finalizing index header", __func__);
        }
    }

    std::fclose(fp);
    close(fdIndex);
    notice("%s: exported %zu PMTiles tiles (%zu candidate source tiles) to %s and %s",
        __func__, tileBuckets.size(), candidateEntryCount, tsvFile.c_str(), indexFile.c_str());
}

void TileOperator::writeMltPmtiles(const std::string& outPrefix,
    const MltPmtilesOptions& mltOptions,
    std::vector<uint32_t> k2keep,
    const std::vector<std::string>& mergePrefixes) {
    if (isTextInput() || !hasFeatureIndex()) {
        error("%s: direct PMTiles export requires feature-bearing binary input", __func__);
    }
    if (k_ <= 0) {
        error("%s: input does not carry top-k payloads; export requires binary decode output with factor assignments", __func__);
    }
    if (mltOptions.zoom < 0 || mltOptions.zoom > 31) {
        error("%s: EPSG:3857 mode requires a zoom in [0, 31]", __func__);
    }
    if (mltOptions.encode_prob_min > 1.0) {
        error("%s: encodeProbMin must be <= 1", __func__);
    }
    if (mltOptions.encode_prob_eps > 1.0) {
        error("%s: encodeProbEps must be <= 1", __func__);
    }
    if (k2keep.size() == 0) {
        k2keep.push_back(static_cast<uint32_t>(k_));
    }

    const std::vector<std::string> featureNames = loadFeatureNames();
    mlt_pmtiles::GlobalStringDictionary featureDictionary;
    featureDictionary.values = featureNames;
    size_t dictSize = featureDictionary.values.size();
    const std::vector<std::string> probColumnNames = build_merge_column_names(k2keep, mergePrefixes);
    const bool hasZ = (coord_dim_ == 3);
    const bool haveGeneBins = has_gene_bin_definition(mltOptions);
    const std::vector<bool> probPairNullable = build_single_prob_nullable_flags(
        kvec_, false, mltOptions.encode_prob_min, mltOptions.encode_prob_eps);
    const mlt_pmtiles::FeatureTableSchema schema = build_point_schema(
        basename(outPrefix), 4096, hasZ, probColumnNames, probPairNullable);
    const auto readRecord = [this, hasZ](std::ifstream& dataStream,
        float& x, float& y, float& z, uint32_t& featureIdx, std::vector<int32_t>& ks, std::vector<float>& ps)
    {
        if (hasZ) {
            PixTopProbsFeature3D<float> rec;
            if (!readBinaryRecord3D(dataStream, rec, false)) {
                error("%s: failed to read 3D feature record", __func__);
            }
            x = rec.x;
            y = rec.y;
            z = rec.z;
            featureIdx = rec.featureIdx;
            ks = std::move(rec.ks);
            ps = std::move(rec.ps);
        } else {
            PixTopProbsFeature<float> rec;
            if (!readBinaryRecord2D(dataStream, rec, false)) {
                error("%s: failed to read 2D feature record", __func__);
            }
            x = rec.x;
            y = rec.y;
            z = 0.0f;
            featureIdx = rec.featureIdx;
            ks = std::move(rec.ks);
            ps = std::move(rec.ps);
        }
    };

    if (!haveGeneBins) { // write a single PMTiles
        const std::string outFile = outPrefix + ".pmtiles";
        std::map<TileKey, AccumTileData> tileMap;
        GeneBinWorkerScanResult allResult;
        const uint8_t outputZoom = static_cast<uint8_t>(mltOptions.zoom);
        if (threads_ > 1) {
            allResult.completedAllTiles = run_epsg3857_parallel_pipeline(
                dataFile_, blocks_, formatInfo_, hasZ, k_,
                threads_, k2keep, readRecord, schema, featureDictionary, featureNames,
                mltOptions.coordScale, mltOptions.encode_prob_min, mltOptions.encode_prob_eps, outputZoom,
                allResult.totalRecordCount, allResult.geoMinX, allResult.geoMinY, allResult.geoMaxX, allResult.geoMaxY);
        } else {
            std::ifstream dataStream(dataFile_, std::ios::binary);
            if (!dataStream.is_open()) {
                error("%s: cannot open binary input %s", __func__, dataFile_.c_str());
            }
            float x = 0.0f, y = 0.0f, z = 0.0f;
            uint32_t featureIdx = 0;
            std::vector<int32_t> ks;
            std::vector<float> ps;
            for (const auto& blk : blocks_) {
                const uint64_t len = blk.idx.ed - blk.idx.st;
                if (len == 0) {continue;}
                if (formatInfo_.recordSize == 0 || (len % formatInfo_.recordSize) != 0) {
                    error("%s: block length %" PRIu64 " is inconsistent with record size %u", __func__, len, formatInfo_.recordSize);
                }
                dataStream.clear();
                dataStream.seekg(static_cast<std::streamoff>(blk.idx.st));
                const uint64_t nRecords = len / formatInfo_.recordSize;
                for (uint64_t i = 0; i < nRecords; ++i) {
                    readRecord(dataStream, x, y, z, featureIdx, ks, ps);

                    if (featureIdx >= featureNames.size()) {
                        error("%s: feature index %u is out of range for dictionary size %zu",
                            __func__, featureIdx, featureNames.size());
                    }
                    append_row_to_epsg3857_tile_map(tileMap, schema, mltOptions.coordScale, outputZoom,
                        hasZ, static_cast<size_t>(k_), k2keep,
                        x, y, z, featureIdx, 1, ks, ps,
                        kNoExtColumnPlans, nullptr,
                        mltOptions.encode_prob_min, mltOptions.encode_prob_eps,
                        allResult.geoMinX, allResult.geoMinY, allResult.geoMaxX, allResult.geoMaxY, allResult.totalRecordCount);
                }
            }

            allResult.completedAllTiles = encode_epsg3857_tile_map(tileMap, outputZoom, schema, featureDictionary, threads_);
        }
        write_single_layer_pmtiles_archive(outFile, schema,
            std::move(allResult.completedAllTiles), allResult.totalRecordCount,
            mltOptions.coordScale, featureNames.size(), outputZoom,
            allResult.geoMinX, allResult.geoMinY,
            allResult.geoMaxX, allResult.geoMaxY,
            "punkst tile-op --write-mlt-pmtiles");
        return;
    }

    GeneBinInfo geneBins = build_gene_bin_info(mltOptions.gene_bin_info_file,
        mltOptions.feature_count_file, mltOptions.n_gene_bins);
    const uint8_t outputZoom = static_cast<uint8_t>(mltOptions.zoom);
    const std::string allOutFile = outPrefix + "_all.pmtiles";
    mlt_pmtiles::FeatureTableSchema allSchema = schema;
    allSchema.layerName = basename(allOutFile, true);
    GeneBinPipelineResult pipeline = run_epsg3857_parallel_gene_bin_pipeline(
        dataFile_, blocks_, formatInfo_, hasZ, k_, threads_, k2keep,
        readRecord, allSchema, featureDictionary, featureNames, geneBins,
        mltOptions.coordScale, mltOptions.encode_prob_min, mltOptions.encode_prob_eps, outputZoom);
    for (uint32_t featureIdx : pipeline.missingBinFeatures) {
        warning("%s: feature '%s' is missing from the gene-bin map",
            __func__, featureNames[featureIdx].c_str());
    }
    write_single_layer_pmtiles_archive(allOutFile, allSchema,
        std::move(pipeline.allEncodedTiles), pipeline.totalRecordCount,
        mltOptions.coordScale, dictSize, outputZoom,
        pipeline.geoMinX, pipeline.geoMinY, pipeline.geoMaxX, pipeline.geoMaxY,
        "punkst tile-op --write-mlt-pmtiles");

    for (auto& kv : pipeline.binEncodedTiles) {
        const std::string outFile = outPrefix + "_bin" + std::to_string(kv.first) + ".pmtiles";
        mlt_pmtiles::FeatureTableSchema binSchema = schema;
        binSchema.layerName = basename(outFile, true);
        const uint64_t binCount = pipeline.binRecordCounts.count(kv.first) > 0
            ? pipeline.binRecordCounts.at(kv.first) : 0;
        relabel_encoded_tile_payloads_layer_name(kv.second, binSchema.layerName);
        write_single_layer_pmtiles_archive(outFile, binSchema,
            std::move(kv.second),
            binCount, mltOptions.coordScale, dictSize, outputZoom,
            pipeline.geoMinX, pipeline.geoMinY,
            pipeline.geoMaxX, pipeline.geoMaxY,
            "punkst tile-op --write-mlt-pmtiles");
    }
    geneBins.write_gene_bin_info_json(outPrefix + ".bin_counts.json");
    write_pmtiles_index_tsv(outPrefix + ".pmtiles_index.tsv", outPrefix,
        pipeline.totalRecordCount, geneBins.entries.size(),
        pipeline.binRecordCounts, geneBins);
}

void TileOperator::annotatePlainToMltPmtiles(
    const std::string& ptPrefix, const std::string& outPrefix,
    int32_t icol_x, int32_t icol_y, int32_t icol_z,
    int32_t icol_f, bool annoKeepAll,
    const std::vector<std::string>& mergePrefixes,
    const MltPmtilesOptions& mltOptions) {
    if (icol_x < 0 || icol_y < 0 || icol_f < 0 || mltOptions.icol_count < 0) {
        error("%s: icol_x, icol_y, icol_feature, and icol_count must be >= 0", __func__);
    }
    if (icol_x == icol_y || icol_f == icol_x || icol_f == icol_y ||
        mltOptions.icol_count == icol_x || mltOptions.icol_count == icol_y ||
        mltOptions.icol_count == icol_f) {
        error("%s: annotate PMTiles packaging requires distinct x/y/feature/count columns", __func__);
    }
    const bool use3d = (coord_dim_ == 3);
    if (use3d && (icol_z < 0 || icol_z == icol_x || icol_z == icol_y ||
            icol_z == icol_f || icol_z == mltOptions.icol_count)) {
        error("%s: valid icol_z distinct from x/y/feature/count is required for 3D annotation packaging", __func__);
    }
    if (!use3d && icol_z >= 0) {
        error("%s: icol_z is only valid for 3D input", __func__);
    }
    if (mltOptions.zoom < 0 || mltOptions.zoom > 31) {
        error("%s: annotate PMTiles packaging currently requires --pmtiles-zoom in [0, 31]", __func__);
    }
    if (mltOptions.encode_prob_min > 1.0) {
        error("%s: encodeProbMin must be <= 1", __func__);
    }
    if (mltOptions.encode_prob_eps > 1.0) {
        error("%s: encodeProbEps must be <= 1", __func__);
    }

    const std::vector<std::string> queryFeatureNames = load_query_feature_names(ptPrefix, icol_f);
    const AnnotatePackagingConfig packaging =
        build_annotate_packaging_config(queryFeatureNames, mltOptions);
    const std::vector<uint32_t> headerKvec = kvec_.empty()
        ? std::vector<uint32_t>{static_cast<uint32_t>(std::max(0, k_))}
        : kvec_;
    const size_t totalK = std::accumulate(headerKvec.begin(), headerKvec.end(), size_t(0));
    const std::vector<std::string> probColumnNames = build_merge_column_names(headerKvec, mergePrefixes);
    AnnotateQueryPlan queryPlan = build_annotate_query_plan(*this,
        ptPrefix, icol_x, icol_y, icol_z, icol_f, mltOptions.icol_count,
        mltOptions, probColumnNames,
        build_single_prob_nullable_flags(
            headerKvec, annoKeepAll, mltOptions.encode_prob_min, mltOptions.encode_prob_eps),
        use3d, basename(outPrefix + "_all"));
    queryPlan.resXY = formatInfo_.pixelResolution > 0.0f ? formatInfo_.pixelResolution : 1.0f;
    queryPlan.resZ = use3d ? queryPlan.resXY : 1.0f;

    TileReader reader(ptPrefix + ".tsv", ptPrefix + ".index");
    assert(reader.getTileSize() == formatInfo_.tileSize);
    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    const GeneBinInfo* geneBinsPtr = packaging.haveGeneBins ? &packaging.geneBins : nullptr;
    const auto processQueryTile = [&](const TileKey& tile,
        std::ifstream& tileStream, AnnotatePmtilesState& state,
        std::unordered_set<std::string>& missingPackagingFeatures)
    {
        std::vector<ExtColumnValue> extValues;
        if (use3d) {
            annotateTile3DPlainShared(reader, tile, tileStream,
                queryPlan.ntok, icol_x, icol_y, icol_z,
                queryPlan.resXY, queryPlan.resZ,
                annoKeepAll, static_cast<uint32_t>(totalK),
                [&](const std::string&, const std::vector<std::string>& tokens,
                    float x, float y, float z,
                    int32_t, int32_t, int32_t, const TopProbs& probs)
                {
                    int32_t countValue = 0;
                    if (!str2int32(tokens[mltOptions.icol_count], countValue)) {
                        error("%s: Invalid line count value for feature '%s'", __func__, tokens[icol_f].c_str());
                    }
                    const std::string& featureName = tokens[icol_f];
                    uint32_t packagingFeatureIdx = 0;
                    if (!lookup_packaging_feature_index(featureName,
                            packaging.packagingFeatureIndex,
                            missingPackagingFeatures,
                            packagingFeatureIdx)) {
                        return false;
                    }
                    if (!parse_ext_column_values(tokens, queryPlan.extColumns, extValues)) {
                        return false;
                    }
                    const std::vector<ExtColumnValue>* extValuesPtr =
                        queryPlan.extColumns.empty() ? nullptr : &extValues;
                    append_annotated_row_to_state(state, geneBinsPtr,
                        queryPlan.schema, mltOptions,
                        headerKvec,
                        queryPlan.extColumns, extValuesPtr,
                        probs.ks.size(), featureName, packagingFeatureIdx,
                        countValue, x, y, z, true, probs);
                    return true;
                },
                __func__);
        } else {
            annotateTile2DPlainShared(reader, tile, tileStream,
                queryPlan.ntok, icol_x, icol_y, queryPlan.resXY,
                annoKeepAll, static_cast<uint32_t>(totalK),
                [&](const std::string&, const std::vector<std::string>& tokens,
                    float x, float y, int32_t, int32_t, const TopProbs& probs)
                {
                    int32_t countValue = 0;
                    if (!str2int32(tokens[mltOptions.icol_count], countValue)) {
                        error("%s: Invalid line count value for feature '%s'", __func__, tokens[icol_f].c_str());
                    }
                    const std::string& featureName = tokens[icol_f];
                    uint32_t packagingFeatureIdx = 0;
                    if (!lookup_packaging_feature_index(featureName,
                            packaging.packagingFeatureIndex,
                            missingPackagingFeatures,
                            packagingFeatureIdx)) {
                        return false;
                    }
                    if (!parse_ext_column_values(tokens, queryPlan.extColumns, extValues)) {
                        return false;
                    }
                    const std::vector<ExtColumnValue>* extValuesPtr =
                        queryPlan.extColumns.empty() ? nullptr : &extValues;
                    append_annotated_row_to_state(state, geneBinsPtr,
                        queryPlan.schema, mltOptions,
                        headerKvec,
                        queryPlan.extColumns, extValuesPtr,
                        probs.ks.size(), featureName, packagingFeatureIdx,
                        countValue, x, y, 0.0f, false, probs);
                    return true;
                },
                __func__);
        }
    };
    const uint8_t outputZoom = static_cast<uint8_t>(mltOptions.zoom);
    AnnotatePmtilesPipelineResult pipeline = run_annotate_epsg3857_parallel_pipeline(
        tiles, reader.getTileSize(), threads_, totalK, use3d,
        queryPlan.schema, packaging.featureDictionary,
        queryPlan.extColumns,
        mltOptions.coordScale, outputZoom,
        [&]() { return std::ifstream(); },
        processQueryTile);
    if (pipeline.missingPackagingFeatures.size() > 0)
    warning("%s: %zu features are not present in the feature dictionary",
        __func__, pipeline.missingPackagingFeatures.size());

    const std::string allOutFile = outPrefix + "_all.pmtiles";
    mlt_pmtiles::FeatureTableSchema allSchema = queryPlan.schema;
    allSchema.layerName = basename(allOutFile, true);
    write_single_layer_pmtiles_archive(allOutFile, allSchema,
        std::move(pipeline.allEncodedTiles), pipeline.totalRecordCount,
        mltOptions.coordScale, packaging.featureDictionary.values.size(), outputZoom,
        pipeline.geoMinX, pipeline.geoMinY, pipeline.geoMaxX, pipeline.geoMaxY,
        "punkst tile-op --annotate-pts --write-mlt-pmtiles");

    if (geneBinsPtr != nullptr) {
        for (auto& kv : pipeline.binEncodedTiles) {
            const std::string outFile = outPrefix + "_bin" + std::to_string(kv.first) + ".pmtiles";
            mlt_pmtiles::FeatureTableSchema binSchema = queryPlan.schema;
            binSchema.layerName = basename(outFile, true);
            const uint64_t binCount = pipeline.binRecordCounts.count(kv.first) > 0
                ? pipeline.binRecordCounts.at(kv.first) : 0;
            relabel_encoded_tile_payloads_layer_name(kv.second, binSchema.layerName);
            write_single_layer_pmtiles_archive(outFile, binSchema,
                std::move(kv.second), binCount,
                mltOptions.coordScale, packaging.featureDictionary.values.size(), outputZoom,
                pipeline.geoMinX, pipeline.geoMinY, pipeline.geoMaxX, pipeline.geoMaxY,
                "punkst tile-op --annotate-pts --write-mlt-pmtiles");
        }
        geneBinsPtr->write_gene_bin_info_json(outPrefix + ".bin_counts.json");
        write_pmtiles_index_tsv(outPrefix + ".pmtiles_index.tsv", outPrefix,
            pipeline.totalRecordCount, geneBinsPtr->entries.size(),
            pipeline.binRecordCounts, *geneBinsPtr);
    }
}

void TileOperator::annotateMergedPlainToMltPmtiles(
    const std::vector<std::string>& otherFiles,
    const std::string& ptPrefix, const std::string& outPrefix,
    std::vector<uint32_t> k2keep, int32_t icol_x, int32_t icol_y,
    int32_t icol_z, int32_t icol_f, bool keepAllMain, bool keepAll,
    const std::vector<std::string>& mergePrefixes, bool annoKeepAll,
    const MltPmtilesOptions& mltOptions) {
    if (icol_x < 0 || icol_y < 0 || icol_f < 0 || mltOptions.icol_count < 0) {
        error("%s: icol_x, icol_y, icol_feature, and icol_count must be >= 0", __func__);
    }
    if (icol_x == icol_y || icol_f == icol_x || icol_f == icol_y ||
        mltOptions.icol_count == icol_x || mltOptions.icol_count == icol_y ||
        mltOptions.icol_count == icol_f) {
        error("%s: annotate PMTiles packaging requires distinct x/y/feature/count columns", __func__);
    }
    const bool use3d = (coord_dim_ == 3);
    if (use3d && (icol_z < 0 || icol_z == icol_x || icol_z == icol_y ||
            icol_z == icol_f || icol_z == mltOptions.icol_count)) {
        error("%s: valid icol_z distinct from x/y/feature/count is required for 3D merged annotation packaging", __func__);
    }
    if (!use3d && icol_z >= 0) {
        error("%s: icol_z is only valid for 3D input", __func__);
    }
    if (mltOptions.zoom < 0 || mltOptions.zoom > 31) {
        error("%s: annotate PMTiles packaging currently requires --pmtiles-zoom in [0, 31]", __func__);
    }
    if (mltOptions.encode_prob_min > 1.0) {
        error("%s: encodeProbMin must be <= 1", __func__);
    }
    if (mltOptions.encode_prob_eps > 1.0) {
        error("%s: encodeProbEps must be <= 1", __func__);
    }

    std::vector<std::unique_ptr<TileOperator>> ops;
    std::vector<TileOperator*> opPtrs;
    opPtrs.push_back(this);
    for (const auto& f : otherFiles) {
        const std::string idxFile = f.substr(0, f.find_last_of('.')) + ".index";
        struct stat buffer;
        if (stat(idxFile.c_str(), &buffer) != 0) {
            error("%s: index file %s not found", __func__, idxFile.c_str());
        }
        ops.push_back(std::make_unique<TileOperator>(f, idxFile));
        opPtrs.push_back(ops.back().get());
    }
    const uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    if (!mergePrefixes.empty() && mergePrefixes.size() != nSources) {
        error("%s: expected %u merge prefixes, got %zu", __func__, nSources, mergePrefixes.size());
    }
    check_k2keep(k2keep, opPtrs);
    const std::vector<MergeSourcePlan> mergePlans = validateMergeSources(opPtrs, k2keep);

    const std::vector<std::string> queryFeatureNames = load_query_feature_names(ptPrefix, icol_f);
    const AnnotatePackagingConfig packaging =
        build_annotate_packaging_config(queryFeatureNames, mltOptions);
    const uint32_t ktotal = std::accumulate(k2keep.begin(), k2keep.end(), uint32_t(0));
    const std::vector<std::string> probColumnNames = build_merge_column_names(k2keep, mergePrefixes);
    AnnotateQueryPlan queryPlan = build_annotate_query_plan(*this,
        ptPrefix,
        icol_x, icol_y, icol_z, icol_f, mltOptions.icol_count,
        mltOptions,
        probColumnNames,
        build_merged_prob_nullable_flags(
            k2keep, annoKeepAll, keepAllMain, keepAll,
            mltOptions.encode_prob_min, mltOptions.encode_prob_eps),
        use3d, basename(outPrefix + "_all"));
    queryPlan.resXY = formatInfo_.pixelResolution > 0.0f ? formatInfo_.pixelResolution : 1.0f;
    queryPlan.resZ = use3d ? queryPlan.resXY : 1.0f;

    TileReader reader(ptPrefix + ".tsv", ptPrefix + ".index");
    assert(reader.getTileSize() == formatInfo_.tileSize);
    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    const GeneBinInfo* geneBinsPtr = packaging.haveGeneBins ? &packaging.geneBins : nullptr;

    struct MergedAnnotateWorkerState {
        std::vector<std::ifstream> streams;
    };
    const auto processQueryTile = [&](const TileKey& tile,
        MergedAnnotateWorkerState& workerState, AnnotatePmtilesState& state,
        std::unordered_set<std::string>& missingPackagingFeatures)
    {
        std::vector<ExtColumnValue> extValues;
        if (use3d) {
            annotateMergedTile3DPlainShared(reader, tile, workerState.streams,
                mergePlans, queryPlan.ntok, icol_x, icol_y, icol_z,
                queryPlan.resXY, queryPlan.resZ,
                keepAllMain, keepAll, annoKeepAll, ktotal,
                [&](const std::string&, const std::vector<std::string>& tokens,
                    float x, float y, float z,
                    int32_t, int32_t, int32_t, const TopProbs& merged)
                {
                    int32_t countValue = 0;
                    if (!str2int32(tokens[mltOptions.icol_count], countValue)) {
                        error("%s: Invalid line count value for feature '%s'", __func__, tokens[icol_f].c_str());
                    }
                    const std::string& featureName = tokens[icol_f];
                    uint32_t packagingFeatureIdx = 0;
                    if (!lookup_packaging_feature_index(featureName,
                            packaging.packagingFeatureIndex,
                            missingPackagingFeatures,
                            packagingFeatureIdx)) {
                        return false;
                    }
                    if (!parse_ext_column_values(tokens, queryPlan.extColumns, extValues)) {
                        return false;
                    }
                    const std::vector<ExtColumnValue>* extValuesPtr =
                        queryPlan.extColumns.empty() ? nullptr : &extValues;
                    append_annotated_row_to_state(state, geneBinsPtr,
                        queryPlan.schema, mltOptions,
                        k2keep,
                        queryPlan.extColumns, extValuesPtr,
                        merged.ks.size(), featureName, packagingFeatureIdx,
                        countValue, x, y, z, true, merged);
                    return true;
                },
                __func__);
        } else {
            annotateMergedTile2DPlainShared(reader, tile, workerState.streams,
                mergePlans, queryPlan.ntok, icol_x, icol_y, queryPlan.resXY,
                keepAllMain, keepAll, annoKeepAll, ktotal,
                [&](const std::string&, const std::vector<std::string>& tokens,
                    float x, float y, int32_t, int32_t, const TopProbs& merged)
                {
                    int32_t countValue = 0;
                    if (!str2int32(tokens[mltOptions.icol_count], countValue)) {
                        error("%s: Invalid line count value for feature '%s'", __func__, tokens[icol_f].c_str());
                    }
                    const std::string& featureName = tokens[icol_f];
                    uint32_t packagingFeatureIdx = 0;
                    if (!lookup_packaging_feature_index(featureName,
                            packaging.packagingFeatureIndex,
                            missingPackagingFeatures,
                            packagingFeatureIdx)) {
                        return false;
                    }
                    if (!parse_ext_column_values(tokens, queryPlan.extColumns, extValues)) {
                        return false;
                    }
                    const std::vector<ExtColumnValue>* extValuesPtr =
                        queryPlan.extColumns.empty() ? nullptr : &extValues;
                    append_annotated_row_to_state(state, geneBinsPtr,
                        queryPlan.schema, mltOptions,
                        k2keep,
                        queryPlan.extColumns, extValuesPtr,
                        merged.ks.size(), featureName, packagingFeatureIdx,
                        countValue, x, y, 0.0f, false, merged);
                    return true;
                },
                __func__);
        }
    };
    const uint8_t outputZoom = static_cast<uint8_t>(mltOptions.zoom);
    AnnotatePmtilesPipelineResult pipeline = run_annotate_epsg3857_parallel_pipeline(
        tiles, reader.getTileSize(), threads_, ktotal, use3d,
        queryPlan.schema, packaging.featureDictionary,
        queryPlan.extColumns,
        mltOptions.coordScale, outputZoom,
        [&]() { return MergedAnnotateWorkerState{std::vector<std::ifstream>(nSources)}; },
        processQueryTile);
    if (pipeline.missingPackagingFeatures.size() > 0)
    warning("%s: %zu features are not present in the feature dictionary",
        __func__, pipeline.missingPackagingFeatures.size());

    const std::string allOutFile = outPrefix + "_all.pmtiles";
    mlt_pmtiles::FeatureTableSchema allSchema = queryPlan.schema;
    allSchema.layerName = basename(allOutFile, true);
    write_single_layer_pmtiles_archive(allOutFile, allSchema,
        std::move(pipeline.allEncodedTiles), pipeline.totalRecordCount,
        mltOptions.coordScale, packaging.featureDictionary.values.size(), outputZoom,
        pipeline.geoMinX, pipeline.geoMinY, pipeline.geoMaxX, pipeline.geoMaxY,
        "punkst tile-op --annotate-pts --merge-emb --write-mlt-pmtiles");

    if (geneBinsPtr != nullptr) {
        for (auto& kv : pipeline.binEncodedTiles) {
            const std::string outFile = outPrefix + "_bin" + std::to_string(kv.first) + ".pmtiles";
            mlt_pmtiles::FeatureTableSchema binSchema = queryPlan.schema;
            binSchema.layerName = basename(outFile, true);
            const uint64_t binCount = pipeline.binRecordCounts.count(kv.first) > 0
                ? pipeline.binRecordCounts.at(kv.first) : 0;
            relabel_encoded_tile_payloads_layer_name(kv.second, binSchema.layerName);
            write_single_layer_pmtiles_archive(outFile, binSchema,
                std::move(kv.second), binCount,
                mltOptions.coordScale, packaging.featureDictionary.values.size(), outputZoom,
                pipeline.geoMinX, pipeline.geoMinY, pipeline.geoMaxX, pipeline.geoMaxY,
                "punkst tile-op --annotate-pts --merge-emb --write-mlt-pmtiles");
        }
        geneBinsPtr->write_gene_bin_info_json(outPrefix + ".bin_counts.json");
        write_pmtiles_index_tsv(outPrefix + ".pmtiles_index.tsv", outPrefix,
            pipeline.totalRecordCount, geneBinsPtr->entries.size(),
            pipeline.binRecordCounts, *geneBinsPtr);
    }
}

void TileOperator::annotateSingleMoleculeToMltPmtiles(
    const std::string& ptPrefix, const std::string& outPrefix,
    int32_t icol_x, int32_t icol_y, int32_t icol_z,
    int32_t icol_f, bool annoKeepAll,
    const std::vector<std::string>& mergePrefixes,
    const MltPmtilesOptions& mltOptions) {
    if (icol_x < 0 || icol_y < 0 || icol_f < 0 || mltOptions.icol_count < 0) {
        error("%s: icol_x, icol_y, icol_feature, and icol_count must be >= 0", __func__);
    }
    if (icol_x == icol_y || icol_f == icol_x || icol_f == icol_y ||
        mltOptions.icol_count == icol_x || mltOptions.icol_count == icol_y ||
        mltOptions.icol_count == icol_f) {
        error("%s: annotate PMTiles packaging requires distinct x/y/feature/count columns", __func__);
    }
    const bool use3d = (coord_dim_ == 3);
    if (use3d && (icol_z < 0 || icol_z == icol_x || icol_z == icol_y ||
            icol_z == icol_f || icol_z == mltOptions.icol_count)) {
        error("%s: valid icol_z distinct from x/y/feature/count is required for 3D annotation packaging", __func__);
    }
    if (!use3d && icol_z >= 0) {
        error("%s: icol_z is only valid for 3D input", __func__);
    }
    if (mltOptions.zoom < 0 || mltOptions.zoom > 31) {
        error("%s: annotate PMTiles packaging currently requires --pmtiles-zoom in [0, 31]", __func__);
    }
    if (mltOptions.encode_prob_min > 1.0) {
        error("%s: encodeProbMin must be <= 1", __func__);
    }
    if (mltOptions.encode_prob_eps > 1.0) {
        error("%s: encodeProbEps must be <= 1", __func__);
    }

    const std::vector<std::string> annotationFeatureNames = loadFeatureNames();
    const auto annotationFeatureIndex = build_feature_index_map(annotationFeatureNames);
    const AnnotatePackagingConfig packaging =
        build_annotate_packaging_config(annotationFeatureNames, mltOptions);
    const std::vector<uint32_t> headerKvec = kvec_.empty()
        ? std::vector<uint32_t>{static_cast<uint32_t>(std::max(0, k_))}
        : kvec_;
    const std::vector<std::string> probColumnNames = build_merge_column_names(headerKvec, mergePrefixes);
    const AnnotateQueryPlan queryPlan = build_annotate_query_plan(*this,
        ptPrefix, icol_x, icol_y, icol_z, icol_f, mltOptions.icol_count,
        mltOptions, probColumnNames,
        build_single_prob_nullable_flags(
            headerKvec, annoKeepAll, mltOptions.encode_prob_min, mltOptions.encode_prob_eps),
        use3d, basename(outPrefix + "_all"));

    TileReader reader(ptPrefix + ".tsv", ptPrefix + ".index");
    assert(reader.getTileSize() == formatInfo_.tileSize);
    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    const GeneBinInfo* geneBinsPtr = packaging.haveGeneBins ? &packaging.geneBins : nullptr;
    const auto processQueryTile = [&](const TileKey& tile,
        std::ifstream& tileStream, AnnotatePmtilesState& state,
        std::unordered_set<std::string>& missingPackagingFeatures)
    {
        std::vector<ExtColumnValue> extValues;
        if (use3d) {
            annotateSingleTile3DShared(reader, tile, tileStream,
                annotationFeatureIndex, queryPlan.ntok,
                icol_x, icol_y, icol_z, icol_f, queryPlan.resXY, queryPlan.resZ,
                annoKeepAll, static_cast<uint32_t>(k_),
                [&](const std::string&, const std::vector<std::string>& tokens,
                    const std::string& featureName, bool featureKnown,
                    uint32_t featureIdx, float x, float y, float z,
                    int32_t, int32_t, int32_t, const TopProbs& probs)
                {
                    int32_t countValue = 0;
                    if (!str2int32(tokens[mltOptions.icol_count], countValue)) {
                        error("%s: Invalid line count value for feature '%s'", __func__, featureName.c_str());
                    }
                    uint32_t packagingFeatureIdx = 0;
                    if (!packaging.haveGeneBins) {
                        if (!featureKnown) { return false; }
                        packagingFeatureIdx = featureIdx;
                    } else if (!lookup_packaging_feature_index(featureName,
                            packaging.packagingFeatureIndex,
                            missingPackagingFeatures,
                            packagingFeatureIdx)) { return false; }
                    if (!parse_ext_column_values(tokens, queryPlan.extColumns, extValues)) { return false; }
                    const std::vector<ExtColumnValue>* extValuesPtr =
                        queryPlan.extColumns.empty() ? nullptr : &extValues;
                    append_annotated_row_to_state(state, geneBinsPtr,
                        queryPlan.schema, mltOptions,
                        headerKvec,
                        queryPlan.extColumns, extValuesPtr,
                        probs.ks.size(), featureName, packagingFeatureIdx,
                        countValue, x, y, z, true, probs);
                    return true;
                },
                __func__);
        } else {
            annotateSingleTile2DShared(reader, tile, tileStream,
                annotationFeatureIndex, queryPlan.ntok,
                icol_x, icol_y, icol_f, queryPlan.resXY,
                annoKeepAll, static_cast<uint32_t>(k_),
                [&](const std::string&, const std::vector<std::string>& tokens,
                    const std::string& featureName, bool featureKnown, uint32_t featureIdx, float x, float y,
                    int32_t, int32_t, const TopProbs& probs)
                {
                    int32_t countValue = 0;
                    if (!str2int32(tokens[mltOptions.icol_count], countValue)) {
                        error("%s: Invalid line count value for feature '%s'", __func__, featureName.c_str());
                    }
                    uint32_t packagingFeatureIdx = 0;
                    if (!packaging.haveGeneBins) {
                        if (!featureKnown) { return false; }
                        packagingFeatureIdx = featureIdx;
                    } else if (!lookup_packaging_feature_index(featureName,
                            packaging.packagingFeatureIndex,
                            missingPackagingFeatures,
                            packagingFeatureIdx)) { return false; }
                    if (!parse_ext_column_values(tokens, queryPlan.extColumns, extValues)) { return false; }
                    const std::vector<ExtColumnValue>* extValuesPtr =
                        queryPlan.extColumns.empty() ? nullptr : &extValues;
                    append_annotated_row_to_state(state, geneBinsPtr,
                        queryPlan.schema, mltOptions,
                        headerKvec,
                        queryPlan.extColumns, extValuesPtr, probs.ks.size(), featureName, packagingFeatureIdx,
                        countValue, x, y, 0.0f, false, probs);
                    return true;
                },
                __func__);
        }
    };
    const uint8_t outputZoom = static_cast<uint8_t>(mltOptions.zoom);
    AnnotatePmtilesPipelineResult pipeline = run_annotate_epsg3857_parallel_pipeline(
        tiles, reader.getTileSize(), threads_, static_cast<size_t>(k_), use3d,
        queryPlan.schema, packaging.featureDictionary,
        queryPlan.extColumns,
        mltOptions.coordScale, outputZoom,
        [&]() { return std::ifstream(); },
        processQueryTile);
    if (pipeline.missingPackagingFeatures.size() > 0)
    warning("%s: %zu features are not present in the feature dictionary", __func__, pipeline.missingPackagingFeatures.size());

    const std::string allOutFile = outPrefix + "_all.pmtiles";
    mlt_pmtiles::FeatureTableSchema allSchema = queryPlan.schema;
    allSchema.layerName = basename(allOutFile, true);
    write_single_layer_pmtiles_archive(allOutFile, allSchema,
        std::move(pipeline.allEncodedTiles), pipeline.totalRecordCount,
        mltOptions.coordScale, packaging.featureDictionary.values.size(), outputZoom,
        pipeline.geoMinX, pipeline.geoMinY, pipeline.geoMaxX, pipeline.geoMaxY,
        "punkst tile-op --annotate-pts --write-mlt-pmtiles");

    if (geneBinsPtr != nullptr) {
        for (auto& kv : pipeline.binEncodedTiles) {
            const std::string outFile = outPrefix + "_bin" + std::to_string(kv.first) + ".pmtiles";
            mlt_pmtiles::FeatureTableSchema binSchema = queryPlan.schema;
            binSchema.layerName = basename(outFile, true);
            const uint64_t binCount = pipeline.binRecordCounts.count(kv.first) > 0 ? pipeline.binRecordCounts.at(kv.first) : 0;
            relabel_encoded_tile_payloads_layer_name(kv.second, binSchema.layerName);
            write_single_layer_pmtiles_archive(outFile, binSchema,
                std::move(kv.second), binCount,
                mltOptions.coordScale, packaging.featureDictionary.values.size(), outputZoom,
                pipeline.geoMinX, pipeline.geoMinY, pipeline.geoMaxX, pipeline.geoMaxY,
                "punkst tile-op --annotate-pts --write-mlt-pmtiles");
        }
        geneBinsPtr->write_gene_bin_info_json(outPrefix + ".bin_counts.json");
        write_pmtiles_index_tsv(outPrefix + ".pmtiles_index.tsv", outPrefix,
            pipeline.totalRecordCount, geneBinsPtr->entries.size(),
            pipeline.binRecordCounts, *geneBinsPtr);
    }
}

void TileOperator::annotateMergedSingleMoleculeToMltPmtiles(
    const std::vector<std::string>& otherFiles,
    const std::string& ptPrefix, const std::string& outPrefix,
    std::vector<uint32_t> k2keep, int32_t icol_x, int32_t icol_y,
    int32_t icol_z, int32_t icol_f, bool keepAllMain, bool keepAll,
    const std::vector<std::string>& mergePrefixes, bool annoKeepAll,
    const MltPmtilesOptions& mltOptions) {
    if (icol_x < 0 || icol_y < 0 || icol_f < 0 || mltOptions.icol_count < 0) {
        error("%s: icol_x, icol_y, icol_feature, and icol_count must be >= 0", __func__);
    }
    if (icol_x == icol_y || icol_f == icol_x || icol_f == icol_y ||
        mltOptions.icol_count == icol_x || mltOptions.icol_count == icol_y ||
        mltOptions.icol_count == icol_f) {
        error("%s: annotate PMTiles packaging requires distinct x/y/feature/count columns", __func__);
    }
    const bool use3d = (coord_dim_ == 3);
    if (use3d && (icol_z < 0 || icol_z == icol_x || icol_z == icol_y ||
            icol_z == icol_f || icol_z == mltOptions.icol_count)) {
        error("%s: valid icol_z distinct from x/y/feature/count is required for 3D merged annotation packaging", __func__);
    }
    if (!use3d && icol_z >= 0) {
        error("%s: icol_z is only valid for 3D input", __func__);
    }
    if (mltOptions.zoom < 0 || mltOptions.zoom > 31) {
        error("%s: annotate PMTiles packaging currently requires --pmtiles-zoom in [0, 31]", __func__);
    }
    if (mltOptions.encode_prob_min > 1.0) {
        error("%s: encodeProbMin must be <= 1", __func__);
    }
    if (mltOptions.encode_prob_eps > 1.0) {
        error("%s: encodeProbEps must be <= 1", __func__);
    }

    std::vector<std::unique_ptr<TileOperator>> ops;
    std::vector<TileOperator*> opPtrs;
    opPtrs.push_back(this);
    for (const auto& f : otherFiles) {
        const std::string idxFile = f.substr(0, f.find_last_of('.')) + ".index";
        struct stat buffer;
        if (stat(idxFile.c_str(), &buffer) != 0) {
            error("%s: index file %s not found", __func__, idxFile.c_str());
        }
        ops.push_back(std::make_unique<TileOperator>(f, idxFile));
        opPtrs.push_back(ops.back().get());
    }
    const uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    if (!mergePrefixes.empty() && mergePrefixes.size() != nSources) {
        error("%s: expected %u merge prefixes, got %zu", __func__, nSources, mergePrefixes.size());
    }
    check_k2keep(k2keep, opPtrs);
    bool seenNonFeature = false;
    for (size_t i = 0; i < opPtrs.size(); ++i) {
        if (!opPtrs[i]->hasFeatureIndex()) {
            seenNonFeature = true;
        } else if (seenNonFeature) {
            error("%s: feature-bearing sources must come before feature-less auxiliary sources", __func__);
        }
    }

    const std::vector<MergeSourcePlan> mergePlans = validateMergeSources(opPtrs, k2keep);
    const FeatureRemapPlan featureRemap = build_feature_remap_plan(opPtrs, __func__);
    const auto canonicalFeatureIndex = build_feature_index_map(featureRemap.canonicalNames);
    const AnnotatePackagingConfig packaging =
        build_annotate_packaging_config(featureRemap.canonicalNames, mltOptions);
    const std::vector<std::string> probColumnNames = build_merge_column_names(k2keep, mergePrefixes);
    const AnnotateQueryPlan queryPlan = build_annotate_query_plan(*this,
        ptPrefix,
        icol_x, icol_y, icol_z, icol_f, mltOptions.icol_count,
        mltOptions,
        probColumnNames,
        build_merged_prob_nullable_flags(
            k2keep, annoKeepAll, keepAllMain, keepAll,
            mltOptions.encode_prob_min, mltOptions.encode_prob_eps),
        use3d, basename(outPrefix + "_all"));
    const uint32_t ktotal = std::accumulate(k2keep.begin(), k2keep.end(), 0);

    TileReader reader(ptPrefix + ".tsv", ptPrefix + ".index");
    assert(reader.getTileSize() == formatInfo_.tileSize);
    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    const GeneBinInfo* geneBinsPtr = packaging.haveGeneBins ? &packaging.geneBins : nullptr;

    struct MergedAnnotateWorkerState {
        std::vector<std::ifstream> streams;
    };
    const auto processQueryTile = [&](const TileKey& tile,
        MergedAnnotateWorkerState& workerState, AnnotatePmtilesState& state,
        std::unordered_set<std::string>& missingPackagingFeatures)
    {
        std::vector<ExtColumnValue> extValues;
        if (use3d) {
            annotateMergedTile3DShared(reader, tile, workerState.streams,
                mergePlans, featureRemap, canonicalFeatureIndex,
                queryPlan.ntok, icol_x, icol_y, icol_z, icol_f,
                queryPlan.resXY, queryPlan.resZ,
                keepAllMain, keepAll, annoKeepAll, ktotal,
                [&](const std::string&, const std::vector<std::string>& tokens,
                    const std::string& featureName,
                    bool featureKnown, uint32_t featureIdx,
                    float x, float y, float z,
                    int32_t, int32_t, int32_t, const TopProbs& merged)
                {
                    int32_t countValue = 0;
                    if (!str2int32(tokens[mltOptions.icol_count], countValue)) {
                        error("%s: Invalid line count value for feature '%s'", __func__, featureName.c_str());
                    }
                    uint32_t packagingFeatureIdx = 0;
                    if (!packaging.haveGeneBins) {
                        if (!featureKnown) {return false;}
                        packagingFeatureIdx = featureIdx;
                    } else if (!lookup_packaging_feature_index(featureName,
                            packaging.packagingFeatureIndex,
                            missingPackagingFeatures, packagingFeatureIdx)) {return false;}
                    if (!parse_ext_column_values(tokens, queryPlan.extColumns, extValues)) {return false;}
                    const std::vector<ExtColumnValue>* extValuesPtr =
                        queryPlan.extColumns.empty() ? nullptr : &extValues;
                    append_annotated_row_to_state(state, geneBinsPtr,
                        queryPlan.schema, mltOptions,
                        k2keep,
                        queryPlan.extColumns, extValuesPtr,
                        merged.ks.size(), featureName, packagingFeatureIdx,
                        countValue, x, y, z, true, merged);
                    return true;
                },
                __func__);
        } else {
            annotateMergedTile2DShared(reader, tile, workerState.streams,
                mergePlans, featureRemap, canonicalFeatureIndex,
                queryPlan.ntok, icol_x, icol_y, icol_f, queryPlan.resXY,
                keepAllMain, keepAll, annoKeepAll, ktotal,
                [&](const std::string&, const std::vector<std::string>& tokens,
                    const std::string& featureName, bool featureKnown,
                    uint32_t featureIdx, float x, float y,
                    int32_t, int32_t, const TopProbs& merged)
                {
                    int32_t countValue = 0;
                    if (!str2int32(tokens[mltOptions.icol_count], countValue)) {
                        error("%s: Invalid line count value for feature '%s'", __func__, featureName.c_str());
                    }
                    uint32_t packagingFeatureIdx = 0;
                    if (!packaging.haveGeneBins) {
                        if (!featureKnown) {return false;}
                        packagingFeatureIdx = featureIdx;
                    } else if (!lookup_packaging_feature_index(featureName,
                            packaging.packagingFeatureIndex,
                            missingPackagingFeatures, packagingFeatureIdx)) {return false;}
                    if (!parse_ext_column_values(tokens, queryPlan.extColumns, extValues)) {return false;}
                    const std::vector<ExtColumnValue>* extValuesPtr =
                        queryPlan.extColumns.empty() ? nullptr : &extValues;
                    append_annotated_row_to_state(state, geneBinsPtr,
                        queryPlan.schema, mltOptions,
                        k2keep,
                        queryPlan.extColumns, extValuesPtr,
                        merged.ks.size(), featureName, packagingFeatureIdx,
                        countValue, x, y, 0.0f, false, merged);
                    return true;
                },
                __func__);
        }
    };
    const uint8_t outputZoom = static_cast<uint8_t>(mltOptions.zoom);
    AnnotatePmtilesPipelineResult pipeline = run_annotate_epsg3857_parallel_pipeline(
        tiles, reader.getTileSize(), threads_, ktotal, use3d,
        queryPlan.schema, packaging.featureDictionary,
        queryPlan.extColumns,
        mltOptions.coordScale, outputZoom,
        [&]() { return MergedAnnotateWorkerState{std::vector<std::ifstream>(nSources)}; },
        processQueryTile);
    if (pipeline.missingPackagingFeatures.size() > 0)
    warning("%s: %zu features are not present in the feature dictionary", __func__, pipeline.missingPackagingFeatures.size());

    const std::string allOutFile = outPrefix + "_all.pmtiles";
    mlt_pmtiles::FeatureTableSchema allSchema = queryPlan.schema;
    allSchema.layerName = basename(allOutFile, true);
    write_single_layer_pmtiles_archive(allOutFile, allSchema,
        std::move(pipeline.allEncodedTiles), pipeline.totalRecordCount,
        mltOptions.coordScale, packaging.featureDictionary.values.size(), outputZoom,
        pipeline.geoMinX, pipeline.geoMinY, pipeline.geoMaxX, pipeline.geoMaxY,
        "punkst tile-op --annotate-pts --merge-emb --write-mlt-pmtiles");

    if (geneBinsPtr != nullptr) {
        for (auto& kv : pipeline.binEncodedTiles) {
            const std::string outFile = outPrefix + "_bin" + std::to_string(kv.first) + ".pmtiles";
            mlt_pmtiles::FeatureTableSchema binSchema = queryPlan.schema;
            binSchema.layerName = basename(outFile, true);
            const uint64_t binCount = pipeline.binRecordCounts.count(kv.first) > 0
                ? pipeline.binRecordCounts.at(kv.first) : 0;
            relabel_encoded_tile_payloads_layer_name(kv.second, binSchema.layerName);
            write_single_layer_pmtiles_archive(outFile, binSchema,
                std::move(kv.second), binCount,
                mltOptions.coordScale, packaging.featureDictionary.values.size(), outputZoom,
                pipeline.geoMinX, pipeline.geoMinY, pipeline.geoMaxX, pipeline.geoMaxY,
                "punkst tile-op --annotate-pts --merge-emb --write-mlt-pmtiles");
        }
        geneBinsPtr->write_gene_bin_info_json(outPrefix + ".bin_counts.json");
        write_pmtiles_index_tsv(outPrefix + ".pmtiles_index.tsv", outPrefix,
            pipeline.totalRecordCount, geneBinsPtr->entries.size(),
            pipeline.binRecordCounts, *geneBinsPtr);
    }
}
