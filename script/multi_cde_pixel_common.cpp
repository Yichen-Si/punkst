#include "multi_cde_pixel_common.hpp"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <limits>
#include <map>
#include <unordered_set>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "mlt_utils.hpp"
#include "mvt_utils.hpp"
#include "numerical_utils.hpp"
#include "pmtiles_utils.hpp"
#include "utils.h"

namespace {

struct TestRec {
    int k = -1;
    int f = -1;
    double beta = 0.0;
    double log10p = -1.0;
    double pi0 = 0.0;
    double pi1 = 0.0;
    double total_count = 0.0;
    double beta_deconv = 0.0;
    double fc_deconv = 0.0;
    double log10p_deconv = -1.0;
    double p_perm = -1.0;
    bool perm_candidate = false;
};

struct PmtilesParsedKPCol {
    bool ok = false;
    bool isK = false;
    uint32_t idx = 0;
    std::string prefix;
};

struct PmtilesPointSchemaPlan {
    size_t featureCol = 0;
    size_t countCol = 0;
    std::vector<size_t> kCols;
    std::vector<size_t> pCols;
    bool ok = false;
};

struct PmtilesCdeColumnSelection {
    std::vector<std::string> includeColumns;
    bool ok = false;
};

uint64_t readLocalVarint(const uint8_t*& ptr, const uint8_t* end, const char* funcName) {
    uint64_t value = 0;
    uint32_t shift = 0;
    while (ptr < end) {
        const uint8_t byte = *ptr++;
        value |= static_cast<uint64_t>(byte & 0x7fu) << shift;
        if ((byte & 0x80u) == 0) {
            return value;
        }
        shift += 7u;
        if (shift >= 64u) {
            error("%s: varint is too long", funcName);
        }
    }
    error("%s: truncated varint", funcName);
    return 0;
}

std::vector<std::string> splitMvtTopLevelLayers(const std::string& rawTile) {
    std::vector<std::string> out;
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(rawTile.data());
    const uint8_t* end = ptr + rawTile.size();
    while (ptr < end) {
        const uint8_t* fieldStart = ptr;
        const uint64_t key = readLocalVarint(ptr, end, __func__);
        const uint64_t field = key >> 3u;
        const uint64_t wire = key & 0x7u;
        if (wire != 2u) {
            error("%s: expected length-delimited MVT top-level field", __func__);
        }
        const uint64_t len = readLocalVarint(ptr, end, __func__);
        if (len > static_cast<uint64_t>(end - ptr)) {
            error("%s: truncated MVT top-level field", __func__);
        }
        ptr += static_cast<size_t>(len);
        if (field == 3u) {
            out.emplace_back(reinterpret_cast<const char*>(fieldStart),
                             static_cast<size_t>(ptr - fieldStart));
        }
    }
    return out;
}

std::vector<std::string> splitMltTopLevelLayers(const std::string& rawTile) {
    std::vector<std::string> out;
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(rawTile.data());
    const uint8_t* end = ptr + rawTile.size();
    while (ptr < end) {
        const uint8_t* layerStart = ptr;
        const uint64_t len = readLocalVarint(ptr, end, __func__);
        if (len == 0 || len > static_cast<uint64_t>(end - ptr)) {
            error("%s: malformed MLT top-level layer", __func__);
        }
        ptr += static_cast<size_t>(len);
        out.emplace_back(reinterpret_cast<const char*>(layerStart),
                         static_cast<size_t>(ptr - layerStart));
    }
    return out;
}

std::vector<std::string> splitVectorTopLevelLayers(const std::string& rawTile, bool useMvt) {
    try {
        return useMvt ? splitMvtTopLevelLayers(rawTile) : splitMltTopLevelLayers(rawTile);
    } catch (const std::exception&) {
        return {rawTile};
    }
}

PmtilesParsedKPCol parsePmtilesKpCol(const std::string& key) {
    PmtilesParsedKPCol out;
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

bool valuePresent(const pm_vector::ColumnSchema& schema,
                  const pm_vector::PropertyColumn& column,
                  size_t row) {
    return !schema.nullable || column.present.empty() || column.present[row];
}

std::string stringValueAt(const pm_vector::PropertyColumn& column, size_t row) {
    if (!column.stringValues.empty()) {
        return column.stringValues[row];
    }
    if (!column.stringCodes.empty()) {
        return std::to_string(column.stringCodes[row]);
    }
    return "";
}

double numericValueAt(const pm_vector::ColumnSchema& schema,
                      const pm_vector::PropertyColumn& column,
                      size_t row) {
    switch (schema.type) {
    case pm_vector::ScalarType::INT_32:
        return static_cast<double>(column.intValues[row]);
    case pm_vector::ScalarType::FLOAT:
        return static_cast<double>(column.floatValues[row]);
    default:
        error("%s: column %s is not numeric", __func__, schema.name.c_str());
        return 0.0;
    }
}

std::string normalizePmtilesFactorPrefix(std::string prefix) {
    while (!prefix.empty() && prefix.back() == '_') {
        prefix.pop_back();
    }
    return prefix;
}

std::string displayPmtilesFactorPrefix(const std::string& parsedPrefix) {
    if (!parsedPrefix.empty() && parsedPrefix.back() == '_') {
        return parsedPrefix.substr(0, parsedPrefix.size() - 1);
    }
    return parsedPrefix;
}

std::string joinPmtilesFactorPrefixes(const std::vector<std::string>& prefixes) {
    std::string out;
    for (size_t i = 0; i < prefixes.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        const std::string label = displayPmtilesFactorPrefix(prefixes[i]);
        out += label.empty() ? "<default>" : label;
    }
    return out;
}

PmtilesPointSchemaPlan buildPmtilesPointSchemaPlan(
    const pm_vector::FeatureTableSchema& schema,
    const PmtilesCdeOptions& options) {
    PmtilesPointSchemaPlan plan;
    plan.featureCol = schema.columns.size();
    plan.countCol = schema.columns.size();
    struct PairCols {
        size_t k = std::numeric_limits<size_t>::max();
        size_t p = std::numeric_limits<size_t>::max();
    };
    std::vector<std::string> prefixOrder;
    std::map<std::string, std::map<uint32_t, PairCols>> grouped;
    for (size_t i = 0; i < schema.columns.size(); ++i) {
        const auto& col = schema.columns[i];
        if (col.name == options.featureField) {
            plan.featureCol = i;
        }
        if (col.name == options.countField) {
            plan.countCol = i;
        }
        const PmtilesParsedKPCol parsed = parsePmtilesKpCol(col.name);
        if (!parsed.ok) {
            continue;
        }
        if (grouped.count(parsed.prefix) == 0) {
            prefixOrder.push_back(parsed.prefix);
        }
        auto& pair = grouped[parsed.prefix][parsed.idx];
        if (parsed.isK) {
            pair.k = i;
        } else {
            pair.p = i;
        }
    }
    if (plan.featureCol >= schema.columns.size() || plan.countCol >= schema.columns.size()) {
        return plan;
    }
    if (schema.columns[plan.featureCol].type != pm_vector::ScalarType::STRING) {
        return plan;
    }
    if (schema.columns[plan.countCol].type != pm_vector::ScalarType::INT_32 &&
        schema.columns[plan.countCol].type != pm_vector::ScalarType::FLOAT) {
        return plan;
    }
    std::vector<std::string> validPrefixes;
    for (const auto& prefix : prefixOrder) {
        const auto& pairs = grouped[prefix];
        if (pairs.empty()) {
            continue;
        }
        uint32_t nPairs = 0;
        for (uint32_t idx = 1; ; ++idx) {
            auto it = pairs.find(idx);
            if (it == pairs.end()) {
                break;
            }
            if (it->second.k == std::numeric_limits<size_t>::max() ||
                it->second.p == std::numeric_limits<size_t>::max()) {
                return PmtilesPointSchemaPlan{};
            }
            nPairs++;
        }
        if (nPairs > 0) {
            validPrefixes.push_back(prefix);
        }
    }
    if (validPrefixes.empty()) {
        return plan;
    }

    std::string selectedPrefix;
    const std::string requestedPrefix = normalizePmtilesFactorPrefix(options.factorPrefix);
    if (!requestedPrefix.empty()) {
        for (const auto& prefix : validPrefixes) {
            if (displayPmtilesFactorPrefix(prefix) == requestedPrefix) {
                selectedPrefix = prefix;
                break;
            }
        }
        if (selectedPrefix.empty()) {
            error("%s: PMTiles factor prefix '%s' was not found. Available prefixes: %s",
                  __func__, requestedPrefix.c_str(),
                  joinPmtilesFactorPrefixes(validPrefixes).c_str());
        }
    } else if (validPrefixes.size() == 1) {
        selectedPrefix = validPrefixes.front();
    } else {
        error("%s: PMTiles input contains multiple factor prefix groups; specify --pmtiles-factor-prefix. Available prefixes: %s",
              __func__, joinPmtilesFactorPrefixes(validPrefixes).c_str());
    }

    const auto& selected = grouped[selectedPrefix];
    for (uint32_t idx = 1; ; ++idx) {
        auto it = selected.find(idx);
        if (it == selected.end()) {
            break;
        }
        plan.kCols.push_back(it->second.k);
        plan.pCols.push_back(it->second.p);
    }
    plan.ok = !plan.kCols.empty();
    return plan;
}

PmtilesCdeColumnSelection selectPmtilesCdeColumns(
    const pm_vector::FeatureTableSchema& schema,
    const PmtilesCdeOptions& options) {
    PmtilesCdeColumnSelection out;
    const PmtilesPointSchemaPlan plan = buildPmtilesPointSchemaPlan(schema, options);
    if (!plan.ok) {
        return out;
    }
    out.includeColumns.push_back(schema.columns[plan.featureCol].name);
    out.includeColumns.push_back(schema.columns[plan.countCol].name);
    for (size_t i = 0; i < plan.kCols.size(); ++i) {
        out.includeColumns.push_back(schema.columns[plan.kCols[i]].name);
        out.includeColumns.push_back(schema.columns[plan.pCols[i]].name);
    }
    out.ok = true;
    return out;
}

struct PmtilesFactorModelInfo {
    std::string prefix;
    int32_t K = 0;
};

std::vector<PmtilesFactorModelInfo> parsePmtilesFactorModels(const nlohmann::json& metadata) {
    std::vector<PmtilesFactorModelInfo> out;
    if (!metadata.is_object()) {
        return out;
    }
    auto it = metadata.find(pm_vector::PUNKST_FACTOR_MODELS_METADATA_KEY);
    if (it != metadata.end() && it->is_array()) {
        for (const auto& item : *it) {
            if (!item.is_object() || !item.contains("K")) {
                continue;
            }
            int32_t k = 0;
            if (item["K"].is_number_integer()) {
                k = item["K"].get<int32_t>();
            } else if (item["K"].is_number_unsigned()) {
                const uint64_t v = item["K"].get<uint64_t>();
                if (v <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                    k = static_cast<int32_t>(v);
                }
            }
            if (k <= 0) {
                continue;
            }
            PmtilesFactorModelInfo model;
            model.K = k;
            if (item.contains("prefix") && item["prefix"].is_string()) {
                model.prefix = item["prefix"].get<std::string>();
            }
            out.push_back(std::move(model));
        }
    }
    if (out.empty() && metadata.contains("K")) {
        int32_t k = 0;
        if (metadata["K"].is_number_integer()) {
            k = metadata["K"].get<int32_t>();
        } else if (metadata["K"].is_number_unsigned()) {
            const uint64_t v = metadata["K"].get<uint64_t>();
            if (v <= static_cast<uint64_t>(std::numeric_limits<int32_t>::max())) {
                k = static_cast<int32_t>(v);
            }
        }
        if (k > 0) {
            out.push_back(PmtilesFactorModelInfo{"", k});
        }
    }
    return out;
}

int32_t selectPmtilesFactorModelK(
    const std::vector<PmtilesFactorModelInfo>& models,
    const PmtilesCdeOptions& options,
    const std::string& context) {
    if (models.empty()) {
        return 0;
    }
    const std::string requestedPrefix = normalizePmtilesFactorPrefix(options.factorPrefix);
    if (!requestedPrefix.empty()) {
        for (const auto& model : models) {
            if (displayPmtilesFactorPrefix(model.prefix) == requestedPrefix) {
                return model.K;
            }
        }
        std::vector<std::string> prefixes;
        prefixes.reserve(models.size());
        for (const auto& model : models) {
            prefixes.push_back(model.prefix);
        }
        error("%s: PMTiles factor prefix '%s' was not found in factor metadata for %s. Available prefixes: %s",
              __func__, requestedPrefix.c_str(), context.c_str(),
              joinPmtilesFactorPrefixes(prefixes).c_str());
    }
    if (models.size() == 1) {
        return models.front().K;
    }
    std::vector<std::string> prefixes;
    prefixes.reserve(models.size());
    for (const auto& model : models) {
        prefixes.push_back(model.prefix);
    }
    error("%s: PMTiles factor metadata contains multiple factor models for %s; specify --pmtiles-factor-prefix. Available prefixes: %s",
          __func__, context.c_str(), joinPmtilesFactorPrefixes(prefixes).c_str());
    return 0;
}

std::vector<pm_vector::DecodedPointTile> decodePointLayers(const std::string& rawTile, bool useMvt) {
    std::vector<pm_vector::DecodedPointTile> out;
    const std::vector<std::string> layers = splitVectorTopLevelLayers(rawTile, useMvt);
    for (const auto& layer : layers) {
        try {
            pm_vector::DecodedPointTile decoded = useMvt
                ? mvt_pmtiles::decode_point_tile(layer)
                : mlt_pmtiles::decode_point_tile(layer);
            if (decoded.tile.size() > 0) {
                out.push_back(std::move(decoded));
            }
        } catch (const std::exception&) {
            continue;
        }
    }
    return out;
}

std::vector<pm_vector::DecodedPointTile> decodePointLayersSelect(
    const std::string& rawTile,
    bool useMvt,
    const std::vector<std::string>& includeColumns) {
    std::vector<pm_vector::DecodedPointTile> out;
    const std::vector<std::string> layers = splitVectorTopLevelLayers(rawTile, useMvt);
    for (const auto& layer : layers) {
        try {
            pm_vector::DecodedPointTile decoded = useMvt
                ? mvt_pmtiles::decode_point_tile_select(layer, includeColumns)
                : mlt_pmtiles::decode_point_tile_select(layer, includeColumns);
            if (decoded.tile.size() > 0) {
                out.push_back(std::move(decoded));
            }
        } catch (const std::exception&) {
            continue;
        }
    }
    return out;
}

PmtilesCdeColumnSelection inspectPmtilesCdeColumns(
    const pm_core::LoadedPmtilesArchive& archive,
    bool useMvt,
    uint8_t zoom,
    const PmtilesCdeOptions& options) {
    pm_vector::FeatureTableSchema metadataSchema;
    pm_vector::VectorGeometryType geometryType = pm_vector::VectorGeometryType::Point;
    if (pm_vector::parse_exact_schema_json(archive.metadata, "", metadataSchema, &geometryType) &&
        geometryType == pm_vector::VectorGeometryType::Point) {
        PmtilesCdeColumnSelection selected = selectPmtilesCdeColumns(metadataSchema, options);
        if (selected.ok) {
            return selected;
        }
    }
    for (const auto& entry : archive.entries) {
        if (entry.z != zoom) {
            continue;
        }
        const std::string rawTile =
            pm_core::read_pmtiles_tile_payload(*archive.reader, archive.header, entry);
        const auto decodedLayers = decodePointLayers(rawTile, useMvt);
        for (const auto& decoded : decodedLayers) {
            PmtilesCdeColumnSelection selected = selectPmtilesCdeColumns(decoded.schema, options);
            if (selected.ok) {
                return selected;
            }
        }
    }
    return PmtilesCdeColumnSelection{};
}

double pmtilesCoordScale(const nlohmann::json& metadata) {
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

Rectangle<float> pmtilesEntryRect(const pmtiles::entry_zxy& entry, double coordScale) {
    double x0 = 0.0, y0 = 0.0, x1 = 0.0, y1 = 0.0;
    pm_core::tilecoord_to_epsg3857(entry.x, entry.y, 0.0, 0.0, entry.z, x0, y0);
    pm_core::tilecoord_to_epsg3857(entry.x, entry.y, 256.0, 256.0, entry.z, x1, y1);
    return Rectangle<float>(
        static_cast<float>(std::min(x0, x1) / coordScale),
        static_cast<float>(std::min(y0, y1) / coordScale),
        static_cast<float>(std::max(x0, x1) / coordScale),
        static_cast<float>(std::max(y0, y1) / coordScale));
}

void accumulateConfusionFromKp(const std::vector<int32_t>& ks,
                               const std::vector<float>& ps,
                               double weight,
                               Eigen::MatrixXd& confusion,
                               double& residualAccum) {
    double residual = weight;
    for (size_t ii = 0; ii < ks.size(); ++ii) {
        residual -= weight * ps[ii];
        for (size_t jj = ii; jj < ks.size(); ++jj) {
            confusion(ks[ii], ks[jj]) += weight * ps[ii] * ps[jj];
        }
    }
    if (residual > 0.0) {
        residualAccum += residual * residual;
    }
}

bool isRemotePmtilesPath(const std::string& path) {
    return path.rfind("s3://", 0) == 0 ||
           path.rfind("http://", 0) == 0 ||
           path.rfind("https://", 0) == 0;
}

std::string dirnameOfPath(const std::string& path) {
    const size_t pos = path.find_last_of('/');
    if (pos == std::string::npos) {
        return ".";
    }
    if (pos == 0) {
        return "/";
    }
    return path.substr(0, pos);
}

bool readGenesBinCountsFeatures(const std::string& pmtilesPath,
                                const PmtilesCdeOptions& options,
                                std::vector<std::string>& features) {
    if (options.featureField != "gene" || isRemotePmtilesPath(pmtilesPath)) {
        return false;
    }
    const std::string sidecar = dirnameOfPath(pmtilesPath) + "/genes_bin_counts.json";
    std::ifstream in(sidecar);
    if (!in.is_open()) {
        return false;
    }
    nlohmann::json data;
    try {
        in >> data;
    } catch (const std::exception& ex) {
        error("%s: failed to parse %s: %s", __func__, sidecar.c_str(), ex.what());
    }
    if (!data.is_array()) {
        error("%s: expected JSON array in %s", __func__, sidecar.c_str());
    }
    std::unordered_set<std::string> seen;
    for (const auto& row : data) {
        if (!row.is_object() || !row.contains("gene") || !row["gene"].is_string()) {
            error("%s: each row in %s must contain a string gene field", __func__, sidecar.c_str());
        }
        const std::string gene = row["gene"].get<std::string>();
        if (gene.empty()) {
            continue;
        }
        if (seen.insert(gene).second) {
            features.push_back(gene);
        }
    }
    if (!features.empty()) {
        notice("Loaded %zu PMTiles features from %s", features.size(), sidecar.c_str());
        return true;
    }
    return false;
}

} // namespace

void buildPairwiseContrasts(const std::vector<std::string>& dataLabels,
                            std::vector<ContrastDef>& contrasts) {
    const int32_t n = static_cast<int32_t>(dataLabels.size());
    contrasts.clear();
    if (n < 2) {
        return;
    }
    contrasts.reserve(static_cast<size_t>(n) * (n - 1) / 2);
    for (int32_t g0 = 0; g0 < n; ++g0) {
        for (int32_t g1 = g0 + 1; g1 < n; ++g1) {
            ContrastDef contrast;
            contrast.name = dataLabels[g0] + "_vs_" + dataLabels[g1];
            contrast.labels.assign(n, 0);
            contrast.labels[g0] = -1;
            contrast.labels[g1] = 1;
            contrast.group_neg.push_back(g0);
            contrast.group_pos.push_back(g1);
            contrasts.push_back(std::move(contrast));
        }
    }
}

Eigen::MatrixXd readConfusionMatrixFile(const std::string& path, int K) {
    RowMajorMatrixXd mat;
    std::vector<std::string> rnames;
    std::vector<std::string> cnames;
    read_matrix_from_file(path, mat, &rnames, &cnames);
    if (mat.rows() != K || mat.cols() != K) {
        error("Confusion matrix %s has size %d x %d, expected %d x %d",
              path.c_str(), static_cast<int>(mat.rows()), static_cast<int>(mat.cols()), K, K);
    }
    return Eigen::MatrixXd(mat);
}

void accumulateConfusionFromTopProbs(const TopProbs& tp,
                                     Eigen::MatrixXd& confusion,
                                     double& residualAccum) {
    double residual = 1.0;
    for (size_t ii = 0; ii < tp.ks.size(); ++ii) {
        residual -= tp.ps[ii];
        for (size_t jj = ii; jj < tp.ks.size(); ++jj) {
            confusion(tp.ks[ii], tp.ks[jj]) += tp.ps[ii] * tp.ps[jj];
        }
    }
    if (residual > 0.0) {
        residualAccum += residual * residual;
    }
}

void finalizeConfusionMatrix(Eigen::MatrixXd& confusion,
                             double residualAccum,
                             int K) {
    confusion += confusion.transpose().eval();
    const double p_residual = residualAccum / static_cast<double>(K) / static_cast<double>(K);
    for (int k = 0; k < K; ++k) {
        confusion(k, k) /= 2.0;
        for (int l = 0; l < K; ++l) {
            confusion(k, l) += p_residual;
        }
    }
}

namespace {

template<typename RecT>
void accumulateConfusionFromProbRecord(const RecT& rec,
                                       Eigen::MatrixXd& confusion,
                                       double& residualAccum) {
    double residual = 1.0;
    for (size_t ii = 0; ii < rec.ks.size(); ++ii) {
        residual -= rec.ps[ii];
        for (size_t jj = ii; jj < rec.ks.size(); ++jj) {
            confusion(rec.ks[ii], rec.ks[jj]) += rec.ps[ii] * rec.ps[jj];
        }
    }
    if (residual > 0.0) {
        residualAccum += residual * residual;
    }
}

} // namespace

std::unordered_map<int32_t, TileOperator::Slice> aggOneFeatureTile(
    const std::vector<PixTopProbsFeature<float>>& records,
    const std::vector<int32_t>& featureRemap,
    double gridSize,
    double minProb,
    int32_t union_key,
    Eigen::MatrixXd* confusion,
    double* residualAccum) {
    std::unordered_map<int32_t, TileOperator::Slice> tileAgg;
    if (records.empty()) {
        return tileAgg;
    }
    auto aggIt0 = tileAgg.emplace(union_key, TileOperator::Slice()).first;
    auto& oneSlice0 = aggIt0->second;
    for (const auto& rec : records) {
        const uint32_t rawFeature = rec.featureIdx;
        if (rawFeature >= featureRemap.size()) {
            continue;
        }
        const int32_t feature = featureRemap[rawFeature];
        if (feature < 0) {
            continue;
        }
        const double wx = static_cast<double>(rec.x);
        const double wy = static_cast<double>(rec.y);
        if (confusion != nullptr && residualAccum != nullptr) {
            accumulateConfusionFromProbRecord(rec, *confusion, *residualAccum);
        }
        const int32_t ux = static_cast<int32_t>(std::floor(wx / gridSize));
        const int32_t uy = static_cast<int32_t>(std::floor(wy / gridSize));
        for (size_t i = 0; i < rec.ks.size(); ++i) {
            if (rec.ps[i] < minProb) {
                continue;
            }
            const int32_t k = rec.ks[i];
            auto aggIt = tileAgg.find(k);
            if (aggIt == tileAgg.end()) {
                aggIt = tileAgg.emplace(k, TileOperator::Slice()).first;
            }
            aggIt->second[std::make_pair(ux, uy)].add(feature, rec.ps[i]);
        }
        if (union_key != 0) {
            oneSlice0[std::make_pair(ux, uy)].add(feature, 1.0);
        }
    }
    return tileAgg;
}

std::unordered_map<int32_t, TileOperator::Slice> aggOneFeatureTileRegion(
    const std::vector<PixTopProbsFeature<float>>& records,
    const PreparedRegionMask2D& region,
    const TileKey& tile,
    RegionTileState tileState,
    const std::vector<int32_t>& featureRemap,
    double gridSize,
    double minProb,
    int32_t union_key,
    Eigen::MatrixXd* confusion,
    double* residualAccum) {
    std::unordered_map<int32_t, TileOperator::Slice> tileAgg;
    if (records.empty()) {
        return tileAgg;
    }
    if (tileState == RegionTileState::Outside) {
        return tileAgg;
    }
    auto aggIt0 = tileAgg.emplace(union_key, TileOperator::Slice()).first;
    auto& oneSlice0 = aggIt0->second;
    for (const auto& rec : records) {
        const uint32_t rawFeature = rec.featureIdx;
        if (rawFeature >= featureRemap.size()) {
            continue;
        }
        const int32_t feature = featureRemap[rawFeature];
        if (feature < 0) {
            continue;
        }
        const double wx = static_cast<double>(rec.x);
        const double wy = static_cast<double>(rec.y);
        if (tileState == RegionTileState::Partial && !region.containsPoint(wx, wy, &tile)) {
            continue;
        }
        if (confusion != nullptr && residualAccum != nullptr) {
            accumulateConfusionFromProbRecord(rec, *confusion, *residualAccum);
        }
        const int32_t ux = static_cast<int32_t>(std::floor(wx / gridSize));
        const int32_t uy = static_cast<int32_t>(std::floor(wy / gridSize));
        for (size_t i = 0; i < rec.ks.size(); ++i) {
            if (rec.ps[i] < minProb) {
                continue;
            }
            const int32_t k = rec.ks[i];
            auto aggIt = tileAgg.find(k);
            if (aggIt == tileAgg.end()) {
                aggIt = tileAgg.emplace(k, TileOperator::Slice()).first;
            }
            aggIt->second[std::make_pair(ux, uy)].add(feature, rec.ps[i]);
        }
        if (union_key != 0) {
            oneSlice0[std::make_pair(ux, uy)].add(feature, 1.0);
        }
    }
    return tileAgg;
}

PmtilesCdeMetadata scanPmtilesCdeMetadata(const std::string& pmtilesPath,
                                          const PmtilesCdeOptions& options) {
    const pm_core::LoadedPmtilesArchive archive = pm_core::load_pmtiles_archive(pmtilesPath);
    const bool useMvt = archive.header.tile_type == pmtiles::TILETYPE_MVT;
    if (archive.header.tile_type != pmtiles::TILETYPE_MLT && !useMvt) {
        error("%s: expected MLT or MVT PMTiles input: %s", __func__, pmtilesPath.c_str());
    }
    if (archive.entries.empty()) {
        error("%s: PMTiles archive has no tile entries: %s", __func__, pmtilesPath.c_str());
    }
    if (archive.metadata.contains("coordinate_mode")) {
        const std::string mode = archive.metadata["coordinate_mode"].get<std::string>();
        if (mode != "epsg3857") {
            error("%s: unsupported PMTiles coordinate_mode '%s'", __func__, mode.c_str());
        }
    }
    if (options.zoom < -1 || options.zoom > 31) {
        error("%s: --pmtiles-zoom must be -1 or in [0, 31]", __func__);
    }
    const uint8_t zoom = options.zoom >= 0 ? static_cast<uint8_t>(options.zoom) : archive.header.max_zoom;
    const PmtilesCdeColumnSelection columns = inspectPmtilesCdeColumns(archive, useMvt, zoom, options);
    if (!columns.ok) {
        error("%s: PMTiles archive has no point layer with feature/count and selected K/P columns: %s",
              __func__, pmtilesPath.c_str());
    }
    bool sawZoom = false;
    PmtilesCdeMetadata out;
    std::unordered_set<std::string> seenFeatures;
    int32_t maxFactor = -1;
    for (const auto& entry : archive.entries) {
        if (entry.z != zoom) {
            continue;
        }
        sawZoom = true;
        const std::string rawTile =
            pm_core::read_pmtiles_tile_payload(*archive.reader, archive.header, entry);
        const auto decodedLayers = decodePointLayersSelect(rawTile, useMvt, columns.includeColumns);
        for (const auto& decoded : decodedLayers) {
            const PmtilesPointSchemaPlan plan = buildPmtilesPointSchemaPlan(decoded.schema, options);
            if (!plan.ok) {
                continue;
            }
            for (size_t row = 0; row < decoded.tile.size(); ++row) {
                const auto& featureSchema = decoded.schema.columns[plan.featureCol];
                const auto& featureCol = decoded.tile.columns[plan.featureCol];
                const auto& countSchema = decoded.schema.columns[plan.countCol];
                const auto& countCol = decoded.tile.columns[plan.countCol];
                if (!valuePresent(featureSchema, featureCol, row) ||
                    !valuePresent(countSchema, countCol, row)) {
                    continue;
                }
                const double count = numericValueAt(countSchema, countCol, row);
                if (!(count > 0.0) || !std::isfinite(count)) {
                    continue;
                }
                const std::string feature = stringValueAt(featureCol, row);
                if (feature.empty()) {
                    continue;
                }
                bool hasProb = false;
                for (size_t j = 0; j < plan.kCols.size(); ++j) {
                    const size_t kIdx = plan.kCols[j];
                    const size_t pIdx = plan.pCols[j];
                    const auto& kSchema = decoded.schema.columns[kIdx];
                    const auto& pSchema = decoded.schema.columns[pIdx];
                    const auto& kCol = decoded.tile.columns[kIdx];
                    const auto& pCol = decoded.tile.columns[pIdx];
                    if (!valuePresent(kSchema, kCol, row)) {
                        continue;
                    }
                    double p = 0.0;
                    if (valuePresent(pSchema, pCol, row)) {
                        p = numericValueAt(pSchema, pCol, row);
                    } else if (j == 0) {
                        p = 1.0;
                    }
                    if (!(p > 0.0) || !std::isfinite(p)) {
                        continue;
                    }
                    const int32_t k = static_cast<int32_t>(numericValueAt(kSchema, kCol, row));
                    if (k < 0) {
                        continue;
                    }
                    maxFactor = std::max(maxFactor, k);
                    hasProb = true;
                }
                if (!hasProb) {
                    continue;
                }
                if (seenFeatures.insert(feature).second) {
                    out.features.push_back(feature);
                }
                out.nRecords++;
            }
        }
    }
    if (!sawZoom) {
        error("%s: PMTiles archive %s has no entries at zoom %u",
              __func__, pmtilesPath.c_str(), static_cast<unsigned>(zoom));
    }
    if (out.nRecords == 0) {
        error("%s: no valid annotated point records found in PMTiles archive %s",
              __func__, pmtilesPath.c_str());
    }
    out.K = maxFactor + 1;
    return out;
}

int32_t inferPmtilesCdeFactorCount(const std::string& pmtilesPath,
                                   const PmtilesCdeOptions& options) {
    const pm_core::LoadedPmtilesArchive archive = pm_core::load_pmtiles_archive(pmtilesPath);
    if (archive.header.tile_type != pmtiles::TILETYPE_MLT &&
        archive.header.tile_type != pmtiles::TILETYPE_MVT) {
        error("%s: expected MLT or MVT PMTiles input: %s", __func__, pmtilesPath.c_str());
    }
    const std::vector<PmtilesFactorModelInfo> models = parsePmtilesFactorModels(archive.metadata);
    return selectPmtilesFactorModelK(models, options, pmtilesPath);
}

std::vector<std::string> loadPmtilesCdeFeatures(
    const std::string& pmtilesPath,
    const PmtilesCdeOptions& options) {
    std::vector<std::string> features;
    if (readGenesBinCountsFeatures(pmtilesPath, options, features)) {
        return features;
    }
    PmtilesCdeMetadata metadata = scanPmtilesCdeMetadata(pmtilesPath, options);
    return metadata.features;
}

std::vector<std::unordered_map<int32_t, TileOperator::Slice>> aggOnePmtilesCdeMultiRegion(
    const std::string& pmtilesPath,
    const PmtilesCdeOptions& options,
    const std::unordered_map<std::string, int32_t>& featureIndex,
    double gridSize,
    double minProb,
    int32_t union_key,
    const std::vector<const PreparedRegionMask2D*>& regions,
    std::vector<Eigen::MatrixXd*> confusions,
    std::vector<double*> residualAccums) {
    if (confusions.size() != regions.size() || residualAccums.size() != regions.size()) {
        error("%s: region/confusion/residual vector size mismatch", __func__);
    }
    const pm_core::LoadedPmtilesArchive archive = pm_core::load_pmtiles_archive(pmtilesPath);
    const bool useMvt = archive.header.tile_type == pmtiles::TILETYPE_MVT;
    if (archive.header.tile_type != pmtiles::TILETYPE_MLT && !useMvt) {
        error("%s: expected MLT or MVT PMTiles input: %s", __func__, pmtilesPath.c_str());
    }
    if (archive.metadata.contains("coordinate_mode")) {
        const std::string mode = archive.metadata["coordinate_mode"].get<std::string>();
        if (mode != "epsg3857") {
            error("%s: unsupported PMTiles coordinate_mode '%s'", __func__, mode.c_str());
        }
    }
    if (options.zoom < -1 || options.zoom > 31) {
        error("%s: --pmtiles-zoom must be -1 or in [0, 31]", __func__);
    }
    const uint8_t zoom = options.zoom >= 0 ? static_cast<uint8_t>(options.zoom) : archive.header.max_zoom;
    const double coordScale = pmtilesCoordScale(archive.metadata);
    const PmtilesCdeColumnSelection columns = inspectPmtilesCdeColumns(archive, useMvt, zoom, options);
    if (!columns.ok) {
        error("%s: PMTiles archive has no point layer with feature/count and selected K/P columns: %s",
              __func__, pmtilesPath.c_str());
    }
    std::vector<pmtiles::entry_zxy> candidateEntries;
    bool sawZoom = false;
    for (const auto& entry : archive.entries) {
        if (entry.z != zoom) {
            continue;
        }
        sawZoom = true;
        bool intersectsRegion = regions.empty();
        for (const PreparedRegionMask2D* region : regions) {
            if (region == nullptr) {
                intersectsRegion = true;
                break;
            }
            const Rectangle<float> rect = pmtilesEntryRect(entry, coordScale);
            if (rect.intersect(region->bbox_f) == 0) {
                continue;
            }
            intersectsRegion = true;
            break;
        }
        if (!intersectsRegion) {
            continue;
        }
        candidateEntries.push_back(entry);
    }
    if (!sawZoom) {
        error("%s: PMTiles archive %s has no entries at zoom %u",
              __func__, pmtilesPath.c_str(), static_cast<unsigned>(zoom));
    }

    struct PmtilesAggLocal {
        std::vector<std::unordered_map<int32_t, TileOperator::Slice>> tileAggs;
        std::vector<Eigen::MatrixXd> confusions;
        std::vector<double> residuals;
        std::vector<size_t> nValid;
        PmtilesAggLocal(size_t nRegions, int K, int unionKey,
                        const std::vector<bool>& needConfusion)
            : tileAggs(nRegions), confusions(nRegions), residuals(nRegions, 0.0), nValid(nRegions, 0) {
            for (size_t i = 0; i < nRegions; ++i) {
                tileAggs[i].emplace(unionKey, TileOperator::Slice());
                if (needConfusion[i]) {
                    confusions[i] = Eigen::MatrixXd::Zero(K, K);
                }
            }
        }
    };
    std::vector<bool> needConfusion(regions.size(), false);
    for (size_t i = 0; i < regions.size(); ++i) {
        needConfusion[i] = (confusions[i] != nullptr && residualAccums[i] != nullptr);
    }
    tbb::enumerable_thread_specific<PmtilesAggLocal> tls([&] {
        return PmtilesAggLocal(regions.size(), union_key, union_key, needConfusion);
    });
    tbb::enumerable_thread_specific<std::unique_ptr<flexio::FlexReader>> readers([&] {
        if (!flexio::is_remote_uri(pmtilesPath)) {
            return std::unique_ptr<flexio::FlexReader>();
        }
        return flexio::FlexReaderFactory::create_reader(pmtilesPath);
    });
    const auto processEntry = [&](const pmtiles::entry_zxy& entry, PmtilesAggLocal& local) {
        flexio::FlexReader* reader = archive.reader.get();
        if (flexio::is_remote_uri(pmtilesPath)) {
            std::unique_ptr<flexio::FlexReader>& localReader = readers.local();
            if (localReader == nullptr || !localReader->is_open()) {
                localReader = flexio::FlexReaderFactory::create_reader(pmtilesPath);
            }
            if (localReader == nullptr || !localReader->is_open()) {
                error("%s: cannot open PMTiles source %s", __func__, pmtilesPath.c_str());
            }
            reader = localReader.get();
        }
        const std::string rawTile = pm_core::read_pmtiles_tile_payload(*reader, archive.header, entry);
        const auto decodedLayers = decodePointLayersSelect(rawTile, useMvt, columns.includeColumns);
        for (const auto& decoded : decodedLayers) {
            const PmtilesPointSchemaPlan plan = buildPmtilesPointSchemaPlan(decoded.schema, options);
            if (!plan.ok) {
                continue;
            }
            const double localScale = decoded.schema.extent > 0
                ? 256.0 / static_cast<double>(decoded.schema.extent) : 1.0;
            for (size_t row = 0; row < decoded.tile.size(); ++row) {
                double scaledX = 0.0, scaledY = 0.0;
                pm_core::tilecoord_to_epsg3857(entry.x, entry.y,
                    static_cast<double>(decoded.tile.localX[row]) * localScale,
                    static_cast<double>(decoded.tile.localY[row]) * localScale,
                    entry.z, scaledX, scaledY);
                const double x = scaledX / coordScale;
                const double y = scaledY / coordScale;
                std::vector<size_t> matchedRegions;
                matchedRegions.reserve(regions.size());
                for (size_t ri = 0; ri < regions.size(); ++ri) {
                    const PreparedRegionMask2D* region = regions[ri];
                    if (region == nullptr ||
                        region->containsPoint(static_cast<float>(x), static_cast<float>(y))) {
                        matchedRegions.push_back(ri);
                    }
                }
                if (matchedRegions.empty()) {
                    continue;
                }
                const auto& featureSchema = decoded.schema.columns[plan.featureCol];
                const auto& featureCol = decoded.tile.columns[plan.featureCol];
                const auto& countSchema = decoded.schema.columns[plan.countCol];
                const auto& countCol = decoded.tile.columns[plan.countCol];
                if (!valuePresent(featureSchema, featureCol, row) ||
                    !valuePresent(countSchema, countCol, row)) {
                    continue;
                }
                const std::string featureName = stringValueAt(featureCol, row);
                auto featureIt = featureIndex.find(featureName);
                if (featureIt == featureIndex.end()) {
                    continue;
                }
                const double count = numericValueAt(countSchema, countCol, row);
                if (!(count > 0.0) || !std::isfinite(count)) {
                    continue;
                }
                std::vector<int32_t> ks;
                std::vector<float> ps;
                ks.reserve(plan.kCols.size());
                ps.reserve(plan.pCols.size());
                for (size_t j = 0; j < plan.kCols.size(); ++j) {
                    const size_t kIdx = plan.kCols[j];
                    const size_t pIdx = plan.pCols[j];
                    const auto& kSchema = decoded.schema.columns[kIdx];
                    const auto& pSchema = decoded.schema.columns[pIdx];
                    const auto& kCol = decoded.tile.columns[kIdx];
                    const auto& pCol = decoded.tile.columns[pIdx];
                    if (!valuePresent(kSchema, kCol, row)) {
                        continue;
                    }
                    double p = 0.0;
                    if (valuePresent(pSchema, pCol, row)) {
                        p = numericValueAt(pSchema, pCol, row);
                    } else if (j == 0) {
                        p = 1.0;
                    }
                    if (!(p > 0.0) || !std::isfinite(p)) {
                        continue;
                    }
                    const int32_t k = static_cast<int32_t>(numericValueAt(kSchema, kCol, row));
                    if (k < 0 || k >= union_key) {
                        continue;
                    }
                    ks.push_back(k);
                    ps.push_back(static_cast<float>(p));
                }
                if (ks.empty()) {
                    continue;
                }
                const int32_t ux = static_cast<int32_t>(std::floor(x / gridSize));
                const int32_t uy = static_cast<int32_t>(std::floor(y / gridSize));
                const int32_t feature = featureIt->second;
                for (size_t ri : matchedRegions) {
                    auto& tileAgg = local.tileAggs[ri];
                    auto& oneSlice0 = tileAgg.find(union_key)->second;
                    if (needConfusion[ri]) {
                        accumulateConfusionFromKp(ks, ps, count, local.confusions[ri], local.residuals[ri]);
                    }
                    for (size_t j = 0; j < ks.size(); ++j) {
                        if (ps[j] < minProb) {
                            continue;
                        }
                        auto aggIt = tileAgg.find(ks[j]);
                        if (aggIt == tileAgg.end()) {
                            aggIt = tileAgg.emplace(ks[j], TileOperator::Slice()).first;
                        }
                        aggIt->second[std::make_pair(ux, uy)].add(feature, count * ps[j]);
                    }
                    if (union_key != 0) {
                        oneSlice0[std::make_pair(ux, uy)].add(feature, count);
                    }
                    local.nValid[ri]++;
                }
            }
        }
    };
    const size_t nTasks = candidateEntries.size();
    if (options.threads > 1 && nTasks > 1) {
        tbb::parallel_for(tbb::blocked_range<size_t>(0, nTasks),
            [&](const tbb::blocked_range<size_t>& range) {
                auto& local = tls.local();
                for (size_t i = range.begin(); i != range.end(); ++i) {
                    processEntry(candidateEntries[i], local);
                }
            });
    } else {
        auto& local = tls.local();
        for (const auto& entry : candidateEntries) {
            processEntry(entry, local);
        }
    }
    std::vector<size_t> nValid(regions.size(), 0);
    std::vector<std::unordered_map<int32_t, TileOperator::Slice>> out(regions.size());
    for (auto& agg : out) {
        agg.emplace(union_key, TileOperator::Slice());
    }
    for (auto& local : tls) {
        for (size_t ri = 0; ri < regions.size(); ++ri) {
            nValid[ri] += local.nValid[ri];
            if (needConfusion[ri]) {
                *confusions[ri] += local.confusions[ri];
                *residualAccums[ri] += local.residuals[ri];
            }
            for (const auto& kv : local.tileAggs[ri]) {
                auto dstIt = out[ri].find(kv.first);
                if (dstIt == out[ri].end()) {
                    dstIt = out[ri].emplace(kv.first, TileOperator::Slice()).first;
                }
                for (const auto& unitKv : kv.second) {
                    dstIt->second[unitKv.first].add(unitKv.second);
                }
            }
        }
    }
    for (size_t ri = 0; ri < nValid.size(); ++ri) {
        if (nValid[ri] == 0) {
            warning("%s: no valid PMTiles CDE records retained from %s for region %zu",
                    __func__, pmtilesPath.c_str(), ri);
        }
    }
    return out;
}

std::unordered_map<int32_t, TileOperator::Slice> aggOnePmtilesCde(
    const std::string& pmtilesPath,
    const PmtilesCdeOptions& options,
    const std::unordered_map<std::string, int32_t>& featureIndex,
    double gridSize,
    double minProb,
    int32_t union_key,
    const PreparedRegionMask2D* region,
    Eigen::MatrixXd* confusion,
    double* residualAccum) {
    auto out = aggOnePmtilesCdeMultiRegion(
        pmtilesPath, options, featureIndex, gridSize, minProb, union_key,
        std::vector<const PreparedRegionMask2D*>{region},
        std::vector<Eigen::MatrixXd*>{confusion},
        std::vector<double*>{residualAccum});
    return std::move(out.front());
}

int32_t runConditionalPixelTests(const std::string& outPrefix,
                                 const std::string& auxSuffix,
                                 const std::vector<std::string>& dataLabels,
                                 const std::vector<std::string>& featureList,
                                 const std::vector<ContrastDef>& contrasts,
                                 MultiSlicePairwiseBinom& statOp,
                                 PairwiseBinomRobust& statUnion,
                                 MultiSliceUnitCache& unitCache,
                                 const PixelDETestOptions& opts) {
    if (contrasts.empty()) {
        error("No contrasts defined");
    }
    const int K = statOp.get_n_slices();
    const int M = statOp.get_n_features();
    const int32_t n_data = static_cast<int32_t>(dataLabels.size());
    if (n_data != statOp.get_n_groups()) {
        error("Group label count (%d) does not match statistics group count (%d)",
              n_data, statOp.get_n_groups());
    }
    if (static_cast<int32_t>(featureList.size()) != M) {
        error("Feature count (%zu) does not match statistics feature count (%d)",
              featureList.size(), M);
    }

    statOp.finished_adding_data();

    const double minLog10p = -std::log10(opts.minPvalOutput);
    const double min_log_or = std::log(opts.minOROutput);
    const double min_log_or_perm = std::log(opts.minORPerm);
    const auto& active = statOp.get_active_features();

    for (const auto& contrast : contrasts) {
        ContrastPrecomp pc = statOp.prepare_contrast(
            contrast.group_neg, contrast.group_pos, 25, 1e-6, opts.pseudoFracRel);

        std::vector<TestRec> tests;
        tests.reserve(static_cast<size_t>(K) * active.size());
        notice("Contrast %s with %d active features", contrast.name.c_str(),
               static_cast<int32_t>(active.size()));

        for (int f : active) {
            MultiSliceOneResult ms(K);
            if (!statOp.compute_one_test_aggregate(
                    f, contrast.group_neg, contrast.group_pos, pc, ms,
                    opts.minCountPerFeature, true, opts.deconvHitP)) {
                continue;
            }

            std::vector<uint8_t> ok(K, 0);
            int ok_count = 0;
            for (int k = 0; k < K; ++k) {
                if (!ms.slice_ok[static_cast<size_t>(k)]) {
                    continue;
                }
                ok[k] = 1;
                ok_count++;
            }
            if (ok_count == 0) {
                continue;
            }

            for (int k = 0; k < K; ++k) {
                if (!ok[k] || std::abs(ms.beta_obs(k)) < min_log_or) {
                    continue;
                }
                if (ms.log10p_obs(k) < minLog10p) {
                    continue;
                }

                double total_count = 0.0;
                {
                    const auto& counts = statOp.slice(k).get_group_counts();
                    double y0 = 0.0;
                    double y1 = 0.0;
                    for (int32_t g0 : contrast.group_neg) {
                        const size_t idx = static_cast<size_t>(g0) * M + f;
                        y0 += counts[idx];
                    }
                    for (int32_t g1 : contrast.group_pos) {
                        const size_t idx = static_cast<size_t>(g1) * M + f;
                        y1 += counts[idx];
                    }
                    total_count = y0 + y1;
                }

                double beta_deconv = 0.0;
                double fc_deconv = 0.0;
                double log10p_deconv = -1.0;
                if (ms.deconv_ok) {
                    beta_deconv = ms.beta_deconv(k);
                    const double pi0_deconv = ms.pi0_deconv(k);
                    const double pi1_deconv = ms.pi1_deconv(k);
                    if (pi0_deconv > 0.0 && pi1_deconv > 0.0 &&
                        std::isfinite(pi0_deconv) && std::isfinite(pi1_deconv)) {
                        fc_deconv = pi1_deconv / pi0_deconv;
                    }
                    const double se = std::sqrt(ms.varb_obs(k));
                    if (se > 0.0 && std::isfinite(se)) {
                        log10p_deconv = -log10_twosided_p_from_z(beta_deconv / se);
                    }
                }

                TestRec rec;
                rec.k = k;
                rec.f = f;
                rec.beta = ms.beta_obs(k);
                rec.log10p = ms.log10p_obs(k);
                rec.pi0 = ms.pi0_obs(k);
                rec.pi1 = ms.pi1_obs(k);
                rec.total_count = total_count;
                rec.beta_deconv = beta_deconv;
                rec.fc_deconv = fc_deconv;
                rec.log10p_deconv = log10p_deconv;
                rec.perm_candidate = (std::abs(rec.beta) >= min_log_or_perm);
                tests.push_back(rec);
            }
        }

        if (opts.nPerm > 0 && !tests.empty()) {
            struct PermTLS {
                std::vector<double> N1;
                std::vector<double> Y1;
                std::vector<uint32_t> touched_idx;
                std::vector<uint32_t> exceed;
                PermTLS(int K, int M, size_t T)
                    : N1(K, 0.0), Y1(static_cast<size_t>(K) * M, 0.0), exceed(T, 0) {}
                void reset_perm() {
                    std::fill(N1.begin(), N1.end(), 0.0);
                    for (uint32_t idx : touched_idx) {
                        Y1[idx] = 0.0;
                    }
                    touched_idx.clear();
                }
                void add_y1(int k, int f, double y, int M) {
                    const uint32_t idx = static_cast<uint32_t>(k * M + f);
                    if (Y1[idx] == 0.0) {
                        touched_idx.push_back(idx);
                    }
                    Y1[idx] += y;
                }
            };

            const std::string contrastName = contrast.name;
            std::vector<double> Ntot(K, 0.0);
            std::vector<int> n1_units(K, 0), n_units(K, 0);
            std::vector<std::vector<uint32_t>> eligible_units(K);
            for (int k = 0; k < K; ++k) {
                const auto& sl = statOp.slice(k);
                Ntot[k] = sl.sum_group_totals(contrast.group_neg) +
                          sl.sum_group_totals(contrast.group_pos);
                n1_units[k] = sl.sum_group_unit_counts(contrast.group_pos);
                n_units[k] = sl.sum_group_unit_counts(contrast.group_neg) +
                             sl.sum_group_unit_counts(contrast.group_pos);
                const auto& units = unitCache.slice_units(k);
                eligible_units[k].reserve(units.size());
                int cached_n = 0;
                for (uint32_t u = 0; u < units.size(); ++u) {
                    const int32_t g = units[u].group;
                    if (g < 0 || g >= static_cast<int32_t>(contrast.labels.size())) {
                        continue;
                    }
                    const int8_t lab = contrast.labels[g];
                    if (lab == 0) {
                        continue;
                    }
                    eligible_units[k].push_back(u);
                    cached_n++;
                }
                if (cached_n != n_units[k]) {
                    warning("Slice %d cache units (%d) != observed units (%d) for contrast %s.",
                            k, cached_n, n_units[k], contrastName.c_str());
                }
            }

            int32_t n_candidate = 0;
            for (const auto& rec : tests) {
                if (rec.perm_candidate) {
                    n_candidate++;
                }
            }
            notice("Permutation: %d tests (slice, feature) to evaluate for contrast %s",
                   n_candidate, contrastName.c_str());

            tbb::enumerable_thread_specific<PermTLS> tls_perm([&] {
                return PermTLS(K, M, tests.size());
            });

            tbb::parallel_for(tbb::blocked_range<int>(0, opts.nPerm),
                [&](const tbb::blocked_range<int>& range) {
                    auto& T = tls_perm.local();
                    for (int r = range.begin(); r != range.end(); ++r) {
                        T.reset_perm();
                        for (int k = 0; k < K; ++k) {
                            const auto& units = unitCache.slice_units(k);
                            const auto& fid = unitCache.slice_feat_ids(k);
                            const auto& fct = unitCache.slice_feat_counts(k);
                            const auto& eligible = eligible_units[k];
                            const int N = static_cast<int>(eligible.size());
                            int need1 = n1_units[k];
                            if (N <= 0 || need1 < 0 || need1 > N) {
                                continue;
                            }
                            int assigned1 = 0;
                            for (int u = 0; u < N; ++u) {
                                const int remain = N - u;
                                const int remain1 = need1 - assigned1;
                                if (remain1 <= 0) {
                                    break;
                                }
                                bool pick1 = false;
                                if (remain1 == remain) {
                                    pick1 = true;
                                } else {
                                    const double prob = static_cast<double>(remain1) / remain;
                                    const double uu = u01(opts.seed, static_cast<uint64_t>(r),
                                                          static_cast<uint64_t>(k),
                                                          static_cast<uint64_t>(u));
                                    pick1 = (uu < prob);
                                }
                                if (!pick1) {
                                    continue;
                                }
                                assigned1++;
                                const auto& U = units[eligible[u]];
                                T.N1[k] += static_cast<double>(U.n);
                                for (uint32_t t = 0; t < U.len; ++t) {
                                    const int f = fid[U.off + t];
                                    const double y = fct[U.off + t];
                                    T.add_y1(k, f, y, M);
                                }
                            }
                        }

                        for (size_t j = 0; j < tests.size(); ++j) {
                            if (!tests[j].perm_candidate) {
                                continue;
                            }
                            const int k = tests[j].k;
                            const int f = tests[j].f;
                            const double N1p = T.N1[k];
                            const double N0p = Ntot[k] - N1p;
                            if (N0p <= 0.0 || N1p <= 0.0) {
                                continue;
                            }
                            const double Y1p = T.Y1[static_cast<size_t>(k * M + f)];
                            const double Y0p = tests[j].total_count - Y1p;
                            const double pi_eps = (Y0p + Y1p) / (N0p + N1p) * opts.pseudoFracRel;
                            const double pi0p = clamp(Y0p / N0p, pi_eps, 1.0 - 1e-8);
                            const double pi1p = clamp(Y1p / N1p, pi_eps, 1.0 - 1e-8);
                            const double beta_p = logit(pi1p) - logit(pi0p);
                            if (!std::isfinite(beta_p)) {
                                continue;
                            }
                            if (std::abs(beta_p) >= std::abs(tests[j].beta)) {
                                T.exceed[j] += 1;
                            }
                        }
                    }
                });

            std::vector<uint32_t> exceed(tests.size(), 0);
            for (auto& T : tls_perm) {
                for (size_t j = 0; j < tests.size(); ++j) {
                    exceed[j] += T.exceed[j];
                }
            }
            for (size_t j = 0; j < tests.size(); ++j) {
                if (tests[j].perm_candidate) {
                    tests[j].p_perm = static_cast<double>(exceed[j]) / opts.nPerm;
                }
            }
        }

        std::string outFile = outPrefix + "." + contrast.name + ".tsv";
        FILE* out_stream = fopen(outFile.c_str(), "w");
        if (!out_stream) {
            error("Cannot open output file: %s", outFile.c_str());
        }
        const std::string header =
            "Slice\tFeature\tBeta\tlog10p\tPi0\tPi1\tTotalCount\tBeta_deconv\tFC_deconv\tlog10p_deconv";
        if (opts.nPerm > 0) {
            fprintf(out_stream, "%s\tp_perm\n", header.c_str());
        } else {
            fprintf(out_stream, "%s\n", header.c_str());
        }
        for (const auto& rec : tests) {
            fprintf(out_stream,
                    "%d\t%s\t%.4e\t%.4f\t%.4e\t%.4e\t%.1f\t%.4e\t%.4e\t%.4f",
                    rec.k, featureList[rec.f].c_str(),
                    rec.beta, rec.log10p, rec.pi0, rec.pi1, rec.total_count,
                    rec.beta_deconv, rec.fc_deconv, rec.log10p_deconv);
            if (opts.nPerm > 0) {
                fprintf(out_stream, "\t%.4f", rec.p_perm);
            }
            fprintf(out_stream, "\n");
        }
        fclose(out_stream);
        notice("Result for %s is written to:\n  %s", contrast.name.c_str(), outFile.c_str());

        outFile = outPrefix + "." + contrast.name + ".da.tsv";
        FILE* out_da = fopen(outFile.c_str(), "w");
        if (!out_da) {
            error("Cannot open output file: %s", outFile.c_str());
        }
        fprintf(out_da, "Slice\tCount0\tCount1\tFC\tChi2\tlog10p\n");
        const double pseudoCountDA = 0.5;
        std::vector<double> total0(K, 0.0);
        std::vector<double> total1(K, 0.0);
        double grand0 = 0.0;
        double grand1 = 0.0;
        for (int k = 0; k < K; ++k) {
            const auto& totals = statOp.slice(k).get_group_totals();
            for (int32_t g0 : contrast.group_neg) {
                total0[k] += totals[g0];
            }
            for (int32_t g1 : contrast.group_pos) {
                total1[k] += totals[g1];
            }
            grand0 += total0[k];
            grand1 += total1[k];
        }
        if (grand0 > 0.0 && grand1 > 0.0) {
            for (int k = 0; k < K; ++k) {
                const double count0 = total0[k];
                const double count1 = total1[k];
                const double rest0 = std::max(0.0, grand0 - count0);
                const double rest1 = std::max(0.0, grand1 - count1);
                auto stats = chisq2x2_log10p(count0, count1, rest0, rest1, pseudoCountDA);
                const double fc = ((count1 + pseudoCountDA) / (grand1 + pseudoCountDA)) /
                                  ((count0 + pseudoCountDA) / (grand0 + pseudoCountDA));
                fprintf(out_da, "%d\t%.1f\t%.1f\t%.4e\t%.4f\t%.4f\n",
                        k, count0, count1, fc, stats.first, stats.second);
            }
        }
        fclose(out_da);
        notice("Differential abundance result for %s is written to:\n  %s",
               contrast.name.c_str(), outFile.c_str());
    }

    const std::string suffix = auxSuffix.empty() ? "" : ("." + auxSuffix);
    std::string outFile = outPrefix + suffix + ".nobs.tsv";
    FILE* out_nobs = fopen(outFile.c_str(), "w");
    if (!out_nobs) {
        error("Cannot open output file: %s", outFile.c_str());
    }
    outFile = outPrefix + suffix + ".sums.tsv";
    FILE* out_sums = fopen(outFile.c_str(), "w");
    if (!out_sums) {
        error("Cannot open output file: %s", outFile.c_str());
    }
    fprintf(out_nobs, "Slice\tData\tnUnits\tTotalCount\n");
    fprintf(out_sums, "Slice\tData\tFeature\tTotalCount\n");
    for (int k = 0; k < K; ++k) {
        const auto& slice = statOp.slice(k);
        const auto& n_units = slice.get_group_unit_counts();
        const auto& totals = slice.get_group_totals();
        const auto& counts = slice.get_group_counts();
        for (int32_t i = 0; i < n_data; ++i) {
            fprintf(out_nobs, "%d\t%s\t%d\t%.1f\n",
                    k, dataLabels[i].c_str(), n_units[i], totals[i]);
            for (int32_t m = 0; m < M; ++m) {
                const size_t j = static_cast<size_t>(i) * M + m;
                if (std::round(counts[j] * 10.0) == 0.0) {
                    continue;
                }
                fprintf(out_sums, "%d\t%s\t%s\t%.1f\n",
                        k, dataLabels[i].c_str(), featureList[m].c_str(), counts[j]);
            }
        }
    }
    fclose(out_nobs);
    fclose(out_sums);
    return 0;
}
