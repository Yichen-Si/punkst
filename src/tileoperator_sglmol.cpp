#include "tileoperator.hpp"
#include "tileoperator_common.hpp"
#include "numerical_utils.hpp"
#include "region_query.hpp"

#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <set>
#include <sys/stat.h>
#include <unistd.h>
#include <unordered_set>

namespace {

using tileoperator_detail::cellagg::CellAgg;
using tileoperator_detail::cellagg::FactorSums;
using tileoperator_detail::cellagg::write_cell_row;
using tileoperator_detail::fmt::append_format;
using tileoperator_detail::io::append_pix_top_probs_feature3d_binary;
using tileoperator_detail::io::append_pix_top_probs_feature_binary;
using tileoperator_detail::io::TileWriteResult;
using tileoperator_detail::io::write_tile_result;
using tileoperator_detail::merge::append_placeholder_pairs;
using tileoperator_detail::merge::append_top_probs_prefix;
using tileoperator_detail::merge::build_merge_column_names;
using tileoperator_detail::merge::tile_key_from_source_xy;
using tileoperator_detail::parallel::process_tile_results_parallel;

std::unordered_map<std::string, uint32_t> buildFeatureIndexMap(const std::vector<std::string>& featureNames) {
    std::unordered_map<std::string, uint32_t> out;
    out.reserve(featureNames.size());
    for (uint32_t i = 0; i < featureNames.size(); ++i) {
        const auto ret = out.emplace(featureNames[i], i);
        if (!ret.second) {
            error("%s: duplicate feature name '%s' in dictionary", __func__, featureNames[i].c_str());
        }
    }
    return out;
}

struct FeatureRemapPlan {
    std::vector<std::string> canonicalNames;
    std::unordered_map<std::string, uint32_t> canonicalIndex;
    std::vector<std::vector<int32_t>> localToCanonical;
    std::vector<std::vector<int32_t>> canonicalToLocal;
};

bool isIdentityFeatureRemap(const std::vector<int32_t>& localToCanonical) {
    for (size_t i = 0; i < localToCanonical.size(); ++i) {
        if (localToCanonical[i] != static_cast<int32_t>(i)) {
            return false;
        }
    }
    return true;
}

void remapFeatureMapToCanonical(std::map<TileOperator::PixelFeatureKey2, TopProbs>& pixelMap,
    const std::vector<int32_t>& localToCanonical, const char* funcName) {
    if (pixelMap.empty() || localToCanonical.empty() || isIdentityFeatureRemap(localToCanonical)) {
        return;
    }
    std::map<TileOperator::PixelFeatureKey2, TopProbs> remapped;
    for (auto& kv : pixelMap) {
        const uint32_t localFeatureIdx = std::get<2>(kv.first);
        if (localFeatureIdx >= localToCanonical.size()) {
            error("%s: local feature index %u exceeds remap size %zu",
                funcName, localFeatureIdx, localToCanonical.size());
        }
        const int32_t canonicalFeatureIdx = localToCanonical[localFeatureIdx];
        if (canonicalFeatureIdx < 0) {
            error("%s: missing canonical mapping for local feature index %u",
                funcName, localFeatureIdx);
        }
        remapped.emplace(
            std::make_tuple(std::get<0>(kv.first), std::get<1>(kv.first),
                static_cast<uint32_t>(canonicalFeatureIdx)),
            std::move(kv.second));
    }
    pixelMap.swap(remapped);
}

void remapFeatureMapToCanonical(std::map<TileOperator::PixelFeatureKey3, TopProbs>& pixelMap,
    const std::vector<int32_t>& localToCanonical, const char* funcName) {
    if (pixelMap.empty() || localToCanonical.empty() || isIdentityFeatureRemap(localToCanonical)) {
        return;
    }
    std::map<TileOperator::PixelFeatureKey3, TopProbs> remapped;
    for (auto& kv : pixelMap) {
        const uint32_t localFeatureIdx = std::get<3>(kv.first);
        if (localFeatureIdx >= localToCanonical.size()) {
            error("%s: local feature index %u exceeds remap size %zu",
                funcName, localFeatureIdx, localToCanonical.size());
        }
        const int32_t canonicalFeatureIdx = localToCanonical[localFeatureIdx];
        if (canonicalFeatureIdx < 0) {
            error("%s: missing canonical mapping for local feature index %u",
                funcName, localFeatureIdx);
        }
        remapped.emplace(
            std::make_tuple(std::get<0>(kv.first), std::get<1>(kv.first),
                std::get<2>(kv.first), static_cast<uint32_t>(canonicalFeatureIdx)),
            std::move(kv.second));
    }
    pixelMap.swap(remapped);
}

FeatureRemapPlan buildFeatureRemapPlan(const std::vector<TileOperator*>& opPtrs, const char* funcName) {
    if (opPtrs.empty()) {
        error("%s: no operators provided for feature remapping", funcName);
    }
    FeatureRemapPlan plan;
    plan.localToCanonical.resize(opPtrs.size());
    plan.canonicalToLocal.resize(opPtrs.size());
    plan.canonicalNames = opPtrs[0]->getFeatureNames();
    if (plan.canonicalNames.empty()) {
        error("%s: main input is feature-bearing but has no embedded feature dictionary", funcName);
    }
    plan.canonicalIndex = buildFeatureIndexMap(plan.canonicalNames);
    for (size_t srcIdx = 1; srcIdx < opPtrs.size(); ++srcIdx) {
        const TileOperator* op = opPtrs[srcIdx];
        if (!op->hasFeatureIndex()) {
            continue;
        }
        if (op->getFeatureNames().empty()) {
            error("%s: source %zu is feature-bearing but has no embedded feature dictionary",
                funcName, srcIdx);
        }
        for (const auto& featureName : op->getFeatureNames()) {
            if (plan.canonicalIndex.emplace(featureName,
                    static_cast<uint32_t>(plan.canonicalNames.size())).second) {
                plan.canonicalNames.push_back(featureName);
            }
        }
    }
    for (size_t srcIdx = 0; srcIdx < opPtrs.size(); ++srcIdx) {
        const TileOperator* op = opPtrs[srcIdx];
        if (!op->hasFeatureIndex()) {
            continue;
        }
        const auto& sourceNames = op->getFeatureNames();
        plan.localToCanonical[srcIdx].assign(sourceNames.size(), -1);
        plan.canonicalToLocal[srcIdx].assign(plan.canonicalNames.size(), -1);
        for (size_t localIdx = 0; localIdx < sourceNames.size(); ++localIdx) {
            const auto it = plan.canonicalIndex.find(sourceNames[localIdx]);
            if (it == plan.canonicalIndex.end()) {
                error("%s: feature '%s' is missing from canonical dictionary",
                    funcName, sourceNames[localIdx].c_str());
            }
            const int32_t canonicalIdx = static_cast<int32_t>(it->second);
            plan.localToCanonical[srcIdx][localIdx] = canonicalIdx;
            plan.canonicalToLocal[srcIdx][canonicalIdx] = static_cast<int32_t>(localIdx);
        }
    }
    return plan;
}

int32_t require_feature_idx(const std::unordered_map<std::string, uint32_t>& featureIndex,
    const std::string& featureName, const char* funcName) {
    auto it = featureIndex.find(featureName);
    if (it == featureIndex.end()) {
        return -1;
    }
    return it->second;
}

template<typename Key>
void accumulate_probdot_maps(const std::map<Key, TopProbs>& mergedMap,
    const std::vector<uint32_t>& k2keep, const std::vector<uint32_t>& offsets,
    std::vector<std::map<int32_t, double>>& marginals,
    std::vector<std::map<std::pair<int32_t, int32_t>, double>>& internalDots,
    std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>>& crossDots,
    size_t& count) {
    const size_t nSets = k2keep.size();
    count += mergedMap.size();
    for (const auto& kv : mergedMap) {
        const auto& ks = kv.second.ks;
        const auto& ps = kv.second.ps;
        for (size_t s1 = 0; s1 < nSets; ++s1) {
            const uint32_t off1 = offsets[s1];
            for (uint32_t i = 0; i < k2keep[s1]; ++i) {
                const int32_t k1 = ks[off1 + i];
                const float p1 = ps[off1 + i];
                if (k1 < 0 || p1 <= 0.0f) { continue; }
                marginals[s1][k1] += p1;
                for (uint32_t j = i; j < k2keep[s1]; ++j) {
                    const int32_t k2 = ks[off1 + j];
                    const float p2 = ps[off1 + j];
                    if (k2 < 0 || p2 <= 0.0f) { continue; }
                    const auto k12 = (k1 <= k2) ? std::make_pair(k1, k2) : std::make_pair(k2, k1);
                    internalDots[s1][k12] += static_cast<double>(p1) * p2;
                }
                for (size_t s2 = s1 + 1; s2 < nSets; ++s2) {
                    const uint32_t off2 = offsets[s2];
                    auto& cross = crossDots[std::make_pair(s1, s2)];
                    for (uint32_t j = 0; j < k2keep[s2]; ++j) {
                        const int32_t k2 = ks[off2 + j];
                        const float p2 = ps[off2 + j];
                        if (k2 < 0 || p2 <= 0.0f) { continue; }
                        cross[std::make_pair(k1, k2)] += static_cast<double>(p1) * p2;
                    }
                }
            }
        }
    }
}

} // namespace

std::vector<std::string> TileOperator::loadFeatureNames(const std::string& featureDictFile) const {
    (void) featureDictFile;
    if (!hasFeatureIndex()) {
        error("%s: feature names requested from a non-feature-bearing input", __func__);
    }
    if (featureNames_.empty()) {
        error("%s: feature-bearing input requires embedded feature names in the index", __func__);
    }
    return featureNames_;
}

int32_t TileOperator::loadTileToMapFeature(const TileKey& key,
    std::map<PixelFeatureKey2, TopProbs>& pixelMap, std::ifstream* dataStream) const {
    if (coord_dim_ != 2) {
        error("%s: 2D data required, but coord_dim_=%u", __func__, coord_dim_);
    }
    if (!hasFeatureIndex()) {
        error("%s: Feature-bearing input required", __func__);
    }
    pixelMap.clear();
    auto lookup = tile_lookup_.find(key);
    if (lookup == tile_lookup_.end()) {
        return 0;
    }

    std::ifstream localStream;
    std::ifstream* stream = dataStream;
    if (stream == nullptr) {
        stream = &localStream;
    }
    if (!stream->is_open()) {
        stream->open(dataFile_, std::ios::binary);
    }
    if (!stream->is_open()) {
        error("Error opening data file: %s", dataFile_.c_str());
    }

    const TileInfo& blk = blocks_[lookup->second];
    stream->clear();
    stream->seekg(blk.idx.st);
    uint64_t pos = blk.idx.st;
    TopProbs rec;
    int32_t recX = 0;
    int32_t recY = 0;
    uint32_t featureIdx = 0;
    while (readNextRecord2DFeatureAsPixel(*stream, pos, blk.idx.ed, recX, recY, featureIdx, rec)) {
        pixelMap[std::make_tuple(recX, recY, featureIdx)] = std::move(rec);
    }
    return static_cast<int32_t>(pixelMap.size());
}

int32_t TileOperator::loadTileToMapFeature3D(const TileKey& key,
    std::map<PixelFeatureKey3, TopProbs>& pixelMap, std::ifstream* dataStream) const {
    if (coord_dim_ != 3) {
        error("%s: 3D data required, but coord_dim_=%u", __func__, coord_dim_);
    }
    if (!hasFeatureIndex()) {
        error("%s: Feature-bearing input required", __func__);
    }
    pixelMap.clear();
    auto lookup = tile_lookup_.find(key);
    if (lookup == tile_lookup_.end()) {
        return 0;
    }

    std::ifstream localStream;
    std::ifstream* stream = dataStream;
    if (stream == nullptr) {
        stream = &localStream;
    }
    if (!stream->is_open()) {
        stream->open(dataFile_, std::ios::binary);
    }
    if (!stream->is_open()) {
        error("Error opening data file: %s", dataFile_.c_str());
    }

    const TileInfo& blk = blocks_[lookup->second];
    stream->clear();
    stream->seekg(blk.idx.st);
    uint64_t pos = blk.idx.st;
    TopProbs rec;
    int32_t recX = 0;
    int32_t recY = 0;
    int32_t recZ = 0;
    uint32_t featureIdx = 0;
    while (readNextRecord3DFeatureAsPixel(*stream, pos, blk.idx.ed, recX, recY, recZ, featureIdx, rec)) {
        pixelMap[std::make_tuple(recX, recY, recZ, featureIdx)] = std::move(rec);
    }
    return static_cast<int32_t>(pixelMap.size());
}

void TileOperator::dumpTSVSingleMolecule(const std::string& outPrefix,
    int32_t probDigits, int32_t coordDigits, const std::string& featureDictFile,
    PreparedRegionMask2D* regionPtr, float qzmin, float qzmax) {
    const bool useRegion = regionPtr != nullptr;
    const bool hasZRange = !std::isnan(qzmin) || !std::isnan(qzmax);
    PreparedRegionMask2D region;
    std::vector<size_t> activeOrder;
    std::vector<uint8_t> activeStates;
    if (useRegion) {
        region = *regionPtr;
        buildPreparedRegionPlan(region, activeOrder, activeStates);
        if (activeOrder.empty()) {
            warning("%s: No indexed tiles intersect the queried region", __func__);
            return;
        }
        notice("%s: %lu tiles intersect the queried region", __func__, activeOrder.size());
    }
    const std::vector<std::string> featureNames = loadFeatureNames(featureDictFile);
    resetReader();

    FILE* fp = stdout;
    int fdIndex = -1;
    std::string tsvFile;
    bool writeIndex = false;

    if (!outPrefix.empty() && outPrefix != "-") {
        tsvFile = outPrefix + ".tsv";
        const std::string indexFile = outPrefix + ".index";
        fp = std::fopen(tsvFile.c_str(), "w");
        if (!fp) error("Error opening output file: %s", tsvFile.c_str());
        fdIndex = open(indexFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (fdIndex < 0) error("Cannot open output index %s", indexFile.c_str());
        writeIndex = true;
    }

    std::string headerStr = "#x\ty";
    if (coord_dim_ == 3) {
        headerStr += "\tz";
    }
    headerStr += "\tfeature";
    for (int i = 0; i < k_; ++i) {
        headerStr += "\tK" + std::to_string(i + 1) + "\tP" + std::to_string(i + 1);
    }
    headerStr += "\n";
    if (std::fprintf(fp, "%s", headerStr.c_str()) < 0) {
        error("%s: Write error", __func__);
    }

    if (writeIndex) {
        IndexHeader idxHeader = formatInfo_;
        idxHeader.mode &= ~0x47u;
        idxHeader.recordSize = 0;
        if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
            error("%s: index write error", __func__);
        }
    }

    const bool isInt32 = (mode_ & 0x4) != 0;
    const float res = formatInfo_.pixelResolution;
    const bool applyRes = (mode_ & 0x2) && (res > 0 && res != 1.0f);
    if (isInt32 && (!applyRes || res == 1.0f)) {
        coordDigits = 0;
    }

    long currentOffset = std::ftell(fp);
    Rectangle<float> writtenBox;
    bool wroteAny = false;
    auto processBlock = [&](const TileInfo& blk, RegionTileState regionState) {
        dataStream_.clear();
        dataStream_.seekg(blk.idx.st);
        const size_t len = blk.idx.ed - blk.idx.st;
        const size_t recSize = formatInfo_.recordSize;
        if (recSize == 0) {
            error("%s: record size is 0 in binary mode", __func__);
        }
        const bool checkBound = !useRegion && bounded_ && !blk.contained;
        const TileKey tile{blk.row, blk.col};
        const size_t nRecs = len / recSize;
        IndexEntryF newEntry = blk.idx;
        newEntry.st = currentOffset;
        newEntry.n = 0;
        Rectangle<float> tileBox;

        for (size_t i = 0; i < nRecs; ++i) {
            float x = 0.0f, y = 0.0f, z = 0.0f;
            uint32_t featureIdx = 0;
            std::vector<int32_t> ks;
            std::vector<float> ps;
            if (coord_dim_ == 3) {
                PixTopProbsFeature3D<float> temp;
                if (!readBinaryRecord3D(dataStream_, temp, false)) break;
                x = temp.x;
                y = temp.y;
                z = temp.z;
                featureIdx = temp.featureIdx;
                ks = std::move(temp.ks);
                ps = std::move(temp.ps);
            } else {
                PixTopProbsFeature<float> temp;
                if (!readBinaryRecord2D(dataStream_, temp, false)) break;
                x = temp.x;
                y = temp.y;
                featureIdx = temp.featureIdx;
                ks = std::move(temp.ks);
                ps = std::move(temp.ps);
            }
            if (featureIdx >= featureNames.size()) {
                error("%s: feature index %u out of range for dictionary of size %zu", __func__, featureIdx, featureNames.size());
            }
            if (checkBound && !queryBox_.contains(x, y)) {
                continue;
            }
            if (useRegion) {
                if (hasZRange && (z < qzmin || z >= qzmax)) {
                    continue;
                }
                if (regionState == RegionTileState::Partial && !region.containsPoint(x, y, &tile)) {
                    continue;
                }
            }
            const std::string& featureName = featureNames[featureIdx];
            if (coord_dim_ == 3) {
                if (std::fprintf(fp, "%.*f\t%.*f\t%.*f\t%s",
                    coordDigits, x, coordDigits, y, coordDigits, z,
                    featureName.c_str()) < 0) {
                    error("%s: Write error", __func__);
                }
            } else if (std::fprintf(fp, "%.*f\t%.*f\t%s",
                    coordDigits, x, coordDigits, y,
                    featureName.c_str()) < 0) {
                error("%s: Write error", __func__);
            }
            for (int k = 0; k < k_; ++k) {
                if (std::fprintf(fp, "\t%d\t%.*e", ks[k], probDigits, ps[k]) < 0) {
                    error("%s: Write error", __func__);
                }
            }
            if (std::fprintf(fp, "\n") < 0) {
                error("%s: Write error", __func__);
            }
            tileBox.extendToInclude(x, y);
            ++newEntry.n;
        }

        currentOffset = std::ftell(fp);
        newEntry.ed = currentOffset;
        if (newEntry.n == 0) {
            return;
        }
        newEntry.xmin = static_cast<int32_t>(std::floor(tileBox.xmin));
        newEntry.xmax = static_cast<int32_t>(std::ceil(tileBox.xmax));
        newEntry.ymin = static_cast<int32_t>(std::floor(tileBox.ymin));
        newEntry.ymax = static_cast<int32_t>(std::ceil(tileBox.ymax));
        writtenBox.extendToInclude(tileBox);
        wroteAny = true;
        if (writeIndex) {
            if (!write_all(fdIndex, &newEntry, sizeof(newEntry))) {
                error("%s: Index entry write error", __func__);
            }
        }
    };

    if (useRegion) {
        for (size_t i = 0; i < activeOrder.size(); ++i) {
            processBlock(blocks_[activeOrder[i]], static_cast<RegionTileState>(activeStates[i]));
        }
    } else {
        for (const auto& blk : blocks_) {
            processBlock(blk, RegionTileState::Inside);
        }
    }

    if (fp != stdout) std::fclose(fp);
    if (writeIndex) {
        if (wroteAny) {
            IndexHeader idxHeader = formatInfo_;
            idxHeader.mode &= ~0x47u;
            idxHeader.recordSize = 0;
            idxHeader.xmin = writtenBox.xmin;
            idxHeader.xmax = writtenBox.xmax;
            idxHeader.ymin = writtenBox.ymin;
            idxHeader.ymax = writtenBox.ymax;
            if (lseek(fdIndex, 0, SEEK_SET) < 0) {
                error("%s: Error seeking output index file", __func__);
            }
            if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
                error("%s: Error finalizing index header", __func__);
            }
        }
        close(fdIndex);
    }
}

void TileOperator::annotateSingleMolecule(const std::string& ptPrefix, const std::string& outPrefix,
    int32_t icol_x, int32_t icol_y, int32_t icol_z,
    int32_t icol_f, const std::string& featureDictFile, bool annoKeepAll) {
    if (icol_x < 0 || icol_y < 0 || icol_f < 0) {
        error("%s: icol_x, icol_y, and icol_f must be >= 0", __func__);
    }
    if (icol_x == icol_y || icol_f == icol_x || icol_f == icol_y) {
        error("%s: coordinate and feature columns must be distinct", __func__);
    }
    if (coord_dim_ == 3 && (icol_z < 0 || icol_z == icol_x || icol_z == icol_y || icol_z == icol_f)) {
        error("%s: Valid icol_z distinct from icol_x/icol_y/icol_f is required for 3D input", __func__);
    }
    const std::vector<std::string> featureNames = loadFeatureNames(featureDictFile);
    const auto featureIndex = buildFeatureIndexMap(featureNames);

    const std::string ptData = ptPrefix + ".tsv";
    const std::string ptIndex = ptPrefix + ".index";
    TileReader reader(ptData, ptIndex);
    assert(reader.getTileSize() == formatInfo_.tileSize);
    const bool use3d = (coord_dim_ == 3);
    const float resXY = getPixelResolution() > 0.0f ? getPixelResolution() : 1.0f;
    const float resZ = use3d ? getPixelResolutionZ() : 1.0f;

    const std::string outFile = outPrefix + ".tsv";
    const std::string outIndex = outPrefix + ".index";
    FILE* fp = std::fopen(outFile.c_str(), "w");
    if (!fp) error("Cannot open output file %s", outFile.c_str());
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());

    uint32_t ntok = static_cast<uint32_t>(std::max({icol_x, icol_y, icol_f}));
    if (use3d) {ntok = std::max(ntok, static_cast<uint32_t>(icol_z));}
    ntok += 1;
    if (!reader.headerLine.empty()) {
        std::string headerStr = reader.headerLine;
        for (uint32_t i = 1; i <= static_cast<uint32_t>(k_); ++i) {
            headerStr += "\tK" + std::to_string(i) + "\tP" + std::to_string(i);
        }
        std::fprintf(fp, "%s\n", headerStr.c_str());
    }
    long currentOffset = std::ftell(fp);

    IndexHeader idxHeader = formatInfo_;
    idxHeader.mode &= ~0x47u;
    idxHeader.recordSize = 0;
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
        error("%s: Index header write error", __func__);
    }

    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    notice("%s: Start annotating query with %lu tiles", __func__, tiles.size());
    const char* funcName = __func__;
    auto buildTileResult = [&](const TileKey& tile, std::ifstream& tileStream) {
        TileWriteResult result;
        result.tile = tile;
        if (use3d) {
            std::map<PixelFeatureKey3, TopProbs> pixelMap;
            if (loadTileToMapFeature3D(tile, pixelMap, &tileStream) <= 0) {
                if (!annoKeepAll) {
                    return result;
                }
            }
            auto it = reader.get_tile_iterator(tile.row, tile.col);
            if (!it) {
                return result;
            }
            std::string s;
            while (it->next(s)) {
                if (s.empty() || s[0] == '#') continue;
                std::vector<std::string> tokens;
                split(tokens, "\t", s, ntok + 1, true, true, true);
                if (tokens.size() < ntok) {
                    error("%s: Invalid line: %s", funcName, s.c_str());
                }
                float x = 0.0f, y = 0.0f, z = 0.0f;
                if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y) || !str2float(tokens[icol_z], z)) {
                    error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
                }
                const auto fit = featureIndex.find(tokens[icol_f]);
                if (fit == featureIndex.end() && !annoKeepAll) {
                    continue;
                }
                const bool featureKnown = (fit != featureIndex.end());
                const uint32_t featureIdx = featureKnown ? fit->second : 0u;
                const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
                const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
                const int32_t iz = static_cast<int32_t>(std::floor(z / resZ));
                auto pit = featureKnown ? pixelMap.find(std::make_tuple(ix, iy, iz, featureIdx)) : pixelMap.end();
                if (pit == pixelMap.end() && !annoKeepAll) {
                    continue;
                }
                result.textData += s;
                if (pit == pixelMap.end()) {
                    TopProbs missing;
                    append_placeholder_pairs(missing, static_cast<uint32_t>(k_));
                    appendTopProbsText(result.textData, missing);
                } else {
                    appendTopProbsText(result.textData, pit->second);
                }
                result.textData.push_back('\n');
                ++result.n;
            }
            return result;
        }

        std::map<PixelFeatureKey2, TopProbs> pixelMap;
        if (loadTileToMapFeature(tile, pixelMap, &tileStream) <= 0) {
            if (!annoKeepAll) {
                return result;
            }
        }
        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) {
            return result;
        }
        std::string s;
        while (it->next(s)) {
            if (s.empty() || s[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", s, ntok + 1, true, true, true);
            if (tokens.size() < ntok) {
                error("%s: Invalid line: %s", funcName, s.c_str());
            }
            float x = 0.0f, y = 0.0f;
            if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
                error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
            }
            const auto fit = featureIndex.find(tokens[icol_f]);
            if (fit == featureIndex.end() && !annoKeepAll) {
                continue;
            }
            const bool featureKnown = (fit != featureIndex.end());
            const uint32_t featureIdx = featureKnown ? fit->second : 0u;
            const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
            const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
            auto pit = featureKnown ? pixelMap.find(std::make_tuple(ix, iy, featureIdx)) : pixelMap.end();
            if (pit == pixelMap.end() && !annoKeepAll) {
                continue;
            }
            result.textData += s;
            if (pit == pixelMap.end()) {
                TopProbs missing;
                append_placeholder_pairs(missing, static_cast<uint32_t>(k_));
                appendTopProbsText(result.textData, missing);
            } else {
                appendTopProbsText(result.textData, pit->second);
            }
            result.textData.push_back('\n');
            ++result.n;
        }
        return result;
    };

    auto writeTileResult = [&](const TileWriteResult& result) {
        write_tile_result(result, false, fp, -1, fdIndex, currentOffset, formatInfo_.tileSize);
        notice("%s: Annotated tile (%d, %d) with %u points",
            funcName, result.tile.row, result.tile.col, result.n);
    };
    process_tile_results_parallel(tiles, threads_,
        [&]() { return std::ifstream(); },
        buildTileResult, writeTileResult);

    std::fclose(fp);
    close(fdIndex);
}

void TileOperator::annotateMergedSingleMolecule(const std::vector<std::string>& otherFiles,
    const std::string& ptPrefix, const std::string& outPrefix,
    std::vector<uint32_t> k2keep, int32_t icol_x, int32_t icol_y,
    int32_t icol_z, int32_t icol_f, bool keepAllMain, bool keepAll,
    const std::vector<std::string>& mergePrefixes,
    const std::string& featureDictFile, bool annoKeepAll) {
    if (icol_x < 0 || icol_y < 0 || icol_f < 0) {
        error("%s: icol_x, icol_y, and icol_f must be >= 0", __func__);
    }
    if (icol_x == icol_y || icol_f == icol_x || icol_f == icol_y) {
        error("%s: coordinate and feature columns must be distinct", __func__);
    }
    if (coord_dim_ == 3 && (icol_z < 0 || icol_z == icol_x || icol_z == icol_y || icol_z == icol_f)) {
        error("%s: Valid icol_z distinct from icol_x/icol_y/icol_f is required for 3D input", __func__);
    }
    std::vector<std::unique_ptr<TileOperator>> ops;
    std::vector<TileOperator*> opPtrs;
    opPtrs.push_back(this);
    for (const auto& f : otherFiles) {
        const std::string idxFile = f.substr(0, f.find_last_of('.')) + ".index";
        struct stat buffer;
        if (stat(idxFile.c_str(), &buffer) != 0) {
            error("%s: Index file %s not found", __func__, idxFile.c_str());
        }
        ops.push_back(std::make_unique<TileOperator>(f, idxFile));
        opPtrs.push_back(ops.back().get());
    }
    const uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    if (!mergePrefixes.empty() && mergePrefixes.size() != nSources) {
        error("%s: expected %u merge prefixes, got %zu", __func__, nSources, mergePrefixes.size());
    }
    if (!k2keep.empty()) { assert(k2keep.size() == nSources); }
    if (k2keep.empty()) {
        for (const auto* op : opPtrs) {
            k2keep.push_back(op->getK());
        }
    } else {
        for (uint32_t i = 0; i < nSources; ++i) {
            if (k2keep[i] > opPtrs[i]->getK()) {
                warning("%s: Invalid value k (%d) specified for the %d-th source", __func__, k2keep[i], i);
                k2keep[i] = opPtrs[i]->getK();
            }
        }
    }
    if (nSources > 7) {
        const int32_t k = *std::min_element(k2keep.begin(), k2keep.end());
        k2keep.assign(nSources, static_cast<uint32_t>(k));
        warning("%s: More than 7 files to merge, keep %d values each", __func__, k);
    }
    bool seenNonFeature = false;
    for (size_t i = 0; i < opPtrs.size(); ++i) {
        if (!opPtrs[i]->hasFeatureIndex()) {
            seenNonFeature = true;
        } else if (seenNonFeature) {
            error("%s: feature-bearing sources must come before feature-less auxiliary sources", __func__);
        }
    }
    const std::vector<MergeSourcePlan> mergePlans = validateMergeSources(opPtrs, k2keep);
    const FeatureRemapPlan featureRemap = buildFeatureRemapPlan(opPtrs, __func__);
    const auto featureIndex = buildFeatureIndexMap(featureRemap.canonicalNames);
    const bool use3d = (coord_dim_ == 3);
    const size_t totalK = std::accumulate(k2keep.begin(), k2keep.end(), size_t(0));

    const std::string ptData = ptPrefix + ".tsv";
    const std::string ptIndex = ptPrefix + ".index";
    TileReader reader(ptData, ptIndex);
    assert(reader.getTileSize() == formatInfo_.tileSize);
    const float resXY = getPixelResolution() > 0.0f ? getPixelResolution() : 1.0f;
    const float resZ = use3d ? getPixelResolutionZ() : 1.0f;

    const std::string outFile = outPrefix + ".tsv";
    const std::string outIndex = outPrefix + ".index";
    FILE* fp = std::fopen(outFile.c_str(), "w");
    if (!fp) error("Cannot open output file %s", outFile.c_str());
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());

    uint32_t ntok = static_cast<uint32_t>(std::max({icol_x, icol_y, icol_f}));
    if (use3d) { ntok = std::max(ntok, static_cast<uint32_t>(icol_z)); }
    ntok += 1;
    if (!reader.headerLine.empty()) {
        std::string headerStr = reader.headerLine;
        for (const auto& colName : build_merge_column_names(k2keep, mergePrefixes)) {
            headerStr += "\t" + colName;
        }
        std::fprintf(fp, "%s\n", headerStr.c_str());
    }
    long currentOffset = std::ftell(fp);

    IndexHeader idxHeader = formatInfo_;
    idxHeader.mode &= ~0x47u;
    idxHeader.recordSize = 0;
    if (!idxHeader.packKvec(k2keep)) {
        warning("%s: Too many input fields", __func__);
    }
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
        error("%s: Index header write error", __func__);
    }

    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    const char* funcName = __func__;
    notice("%s: Start annotating merged query with %lu tiles", funcName, tiles.size());
    auto buildTileResult = [&](const TileKey& tile, std::vector<std::ifstream>& streams) {
        TileWriteResult result;
        result.tile = tile;
        if (use3d) {
            std::vector<std::map<TileKey, std::map<PixelFeatureKey2, TopProbs>>> auxFeature2D(nSources);
            std::vector<std::map<TileKey, std::map<PixelFeatureKey3, TopProbs>>> auxFeature3D(nSources);
            std::vector<std::map<TileKey, std::map<std::pair<int32_t, int32_t>, TopProbs>>> auxPlain2D(nSources);
            std::vector<std::map<TileKey, std::map<PixelKey3, TopProbs>>> auxPlain3D(nSources);
            std::vector<std::set<TileKey>> missing(nSources);
            auto findRecord = [&](size_t srcIdx, int32_t mainX, int32_t mainY, int32_t mainZ,
                                  bool featureKnown, uint32_t featureIdx) -> const TopProbs* {
                const MergeSourcePlan& plan = mergePlans[srcIdx];
                const int32_t auxX = floorDivInt32(mainX, plan.ratioXY);
                const int32_t auxY = floorDivInt32(mainY, plan.ratioXY);
                const TileKey auxTile = tile_key_from_source_xy(auxX, auxY, plan.srcResXY, plan.tileSize);
                if (missing[srcIdx].count(auxTile) > 0) {
                    return nullptr;
                }
                if (plan.op->hasFeatureIndex()) {
                    if (!featureKnown) {
                        return nullptr;
                    }
                    if (plan.relation == MergeSourceRelation::Broadcast2DTo3D) {
                        auto tileIt = auxFeature2D[srcIdx].find(auxTile);
                        if (tileIt == auxFeature2D[srcIdx].end()) {
                            std::map<PixelFeatureKey2, TopProbs> auxMap;
                            if (plan.op->loadTileToMapFeature(auxTile, auxMap, &streams[srcIdx]) == 0) {
                                missing[srcIdx].insert(auxTile);
                                return nullptr;
                            }
                            remapFeatureMapToCanonical(auxMap, featureRemap.localToCanonical[srcIdx], funcName);
                            tileIt = auxFeature2D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
                        }
                        auto recIt = tileIt->second.find(std::make_tuple(auxX, auxY, featureIdx));
                        return (recIt == tileIt->second.end()) ? nullptr : &recIt->second;
                    }
                    const int32_t auxZ = floorDivInt32(mainZ, plan.ratioZ);
                    auto tileIt = auxFeature3D[srcIdx].find(auxTile);
                    if (tileIt == auxFeature3D[srcIdx].end()) {
                        std::map<PixelFeatureKey3, TopProbs> auxMap;
                        if (plan.op->loadTileToMapFeature3D(auxTile, auxMap, &streams[srcIdx]) == 0) {
                            missing[srcIdx].insert(auxTile);
                            return nullptr;
                        }
                        remapFeatureMapToCanonical(auxMap, featureRemap.localToCanonical[srcIdx], funcName);
                        tileIt = auxFeature3D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
                    }
                    auto recIt = tileIt->second.find(std::make_tuple(auxX, auxY, auxZ, featureIdx));
                    return (recIt == tileIt->second.end()) ? nullptr : &recIt->second;
                }
                if (plan.relation == MergeSourceRelation::Broadcast2DTo3D) {
                    auto tileIt = auxPlain2D[srcIdx].find(auxTile);
                    if (tileIt == auxPlain2D[srcIdx].end()) {
                        std::map<std::pair<int32_t, int32_t>, TopProbs> auxMap;
                        if (plan.op->loadTileToMap(auxTile, auxMap, nullptr, &streams[srcIdx]) == 0) {
                            missing[srcIdx].insert(auxTile);
                            return nullptr;
                        }
                        tileIt = auxPlain2D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
                    }
                    auto recIt = tileIt->second.find({auxX, auxY});
                    return (recIt == tileIt->second.end()) ? nullptr : &recIt->second;
                }
                const int32_t auxZ = floorDivInt32(mainZ, plan.ratioZ);
                auto tileIt = auxPlain3D[srcIdx].find(auxTile);
                if (tileIt == auxPlain3D[srcIdx].end()) {
                    std::map<PixelKey3, TopProbs> auxMap;
                    if (plan.op->loadTileToMap3D(auxTile, auxMap, &streams[srcIdx]) == 0) {
                        missing[srcIdx].insert(auxTile);
                        return nullptr;
                    }
                    tileIt = auxPlain3D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
                }
                auto recIt = tileIt->second.find(std::make_tuple(auxX, auxY, auxZ));
                return (recIt == tileIt->second.end()) ? nullptr : &recIt->second;
            };

            auto it = reader.get_tile_iterator(tile.row, tile.col);
            if (!it) {
                return result;
            }
            std::string s;
            while (it->next(s)) {
                if (s.empty() || s[0] == '#') continue;
                std::vector<std::string> tokens;
                split(tokens, "\t", s, ntok + 1, true, true, true);
                if (tokens.size() < ntok) {
                    error("%s: Invalid line: %s", funcName, s.c_str());
                }
                float x = 0.0f, y = 0.0f, z = 0.0f;
                if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y) || !str2float(tokens[icol_z], z)) {
                    error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
                }
                const auto fit = featureIndex.find(tokens[icol_f]);
                if (fit == featureIndex.end() && !annoKeepAll) {
                    continue;
                }
                const bool featureKnown = (fit != featureIndex.end());
                const uint32_t featureIdx = featureKnown ? fit->second : 0u;
                const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
                const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
                const int32_t iz = static_cast<int32_t>(std::floor(z / resZ));
                TopProbs merged;
                merged.ks.reserve(totalK);
                merged.ps.reserve(totalK);
                bool anyFound = false;
                bool allFound = true;
                bool mainFound = false;
                for (size_t srcIdx = 0; srcIdx < nSources; ++srcIdx) {
                    const TopProbs* rec = findRecord(srcIdx, ix, iy, iz, featureKnown, featureIdx);
                    if (rec == nullptr) {
                        allFound = false;
                        append_placeholder_pairs(merged, mergePlans[srcIdx].keepK);
                    } else {
                        anyFound = true;
                        if (srcIdx == 0) {
                            mainFound = true;
                            ++result.nMain;
                        }
                        append_top_probs_prefix(merged, *rec, mergePlans[srcIdx].keepK);
                    }
                }
                const bool emit = annoKeepAll || (keepAll ? anyFound : (keepAllMain ? mainFound : (mainFound && allFound)));
                if (!emit) {
                    continue;
                }
                result.textData += s;
                appendTopProbsText(result.textData, merged);
                result.textData.push_back('\n');
                ++result.n;
            }
            return result;
        }

        std::vector<std::map<TileKey, std::map<PixelFeatureKey2, TopProbs>>> auxFeature2D(nSources);
        std::vector<std::map<TileKey, std::map<std::pair<int32_t, int32_t>, TopProbs>>> auxPlain2D(nSources);
        std::vector<std::set<TileKey>> missing(nSources);
        auto findRecord2D = [&](size_t srcIdx, int32_t mainX, int32_t mainY,
                                bool featureKnown, uint32_t featureIdx) -> const TopProbs* {
            const MergeSourcePlan& plan = mergePlans[srcIdx];
            const int32_t auxX = floorDivInt32(mainX, plan.ratioXY);
            const int32_t auxY = floorDivInt32(mainY, plan.ratioXY);
            const TileKey auxTile = tile_key_from_source_xy(auxX, auxY, plan.srcResXY, plan.tileSize);
            if (missing[srcIdx].count(auxTile) > 0) {
                return nullptr;
            }
            if (plan.op->hasFeatureIndex()) {
                if (!featureKnown) {
                    return nullptr;
                }
                auto tileIt = auxFeature2D[srcIdx].find(auxTile);
                if (tileIt == auxFeature2D[srcIdx].end()) {
                    std::map<PixelFeatureKey2, TopProbs> auxMap;
                    if (plan.op->loadTileToMapFeature(auxTile, auxMap, &streams[srcIdx]) == 0) {
                        missing[srcIdx].insert(auxTile);
                        return nullptr;
                    }
                    remapFeatureMapToCanonical(auxMap, featureRemap.localToCanonical[srcIdx], funcName);
                    tileIt = auxFeature2D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
                }
                auto recIt = tileIt->second.find(std::make_tuple(auxX, auxY, featureIdx));
                return (recIt == tileIt->second.end()) ? nullptr : &recIt->second;
            }
            auto tileIt = auxPlain2D[srcIdx].find(auxTile);
            if (tileIt == auxPlain2D[srcIdx].end()) {
                std::map<std::pair<int32_t, int32_t>, TopProbs> auxMap;
                if (plan.op->loadTileToMap(auxTile, auxMap, nullptr, &streams[srcIdx]) == 0) {
                    missing[srcIdx].insert(auxTile);
                    return nullptr;
                }
                tileIt = auxPlain2D[srcIdx].emplace(auxTile, std::move(auxMap)).first;
            }
            auto recIt = tileIt->second.find({auxX, auxY});
            return (recIt == tileIt->second.end()) ? nullptr : &recIt->second;
        };

        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) {
            return result;
        }
        std::string s;
        while (it->next(s)) {
            if (s.empty() || s[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", s, ntok + 1, true, true, true);
            if (tokens.size() < ntok) {
                error("%s: Invalid line: %s", funcName, s.c_str());
            }
            float x = 0.0f, y = 0.0f;
            if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
                error("%s: Invalid coordinates in line: %s", funcName, s.c_str());
            }
            const auto fit = featureIndex.find(tokens[icol_f]);
            if (fit == featureIndex.end() && !annoKeepAll) {
                continue;
            }
            const bool featureKnown = (fit != featureIndex.end());
            const uint32_t featureIdx = featureKnown ? fit->second : 0u;
            const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
            const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
            TopProbs merged;
            merged.ks.reserve(totalK);
            merged.ps.reserve(totalK);
            bool anyFound = false;
            bool allFound = true;
            bool mainFound = false;
            for (size_t srcIdx = 0; srcIdx < nSources; ++srcIdx) {
                const TopProbs* rec = findRecord2D(srcIdx, ix, iy, featureKnown, featureIdx);
                if (rec == nullptr) {
                    allFound = false;
                    append_placeholder_pairs(merged, mergePlans[srcIdx].keepK);
                } else {
                    anyFound = true;
                    if (srcIdx == 0) {
                        mainFound = true;
                        ++result.nMain;
                    }
                    append_top_probs_prefix(merged, *rec, mergePlans[srcIdx].keepK);
                }
            }
            const bool emit = annoKeepAll || (keepAll ? anyFound : (keepAllMain ? mainFound : (mainFound && allFound)));
            if (!emit) {
                continue;
            }
            result.textData += s;
            appendTopProbsText(result.textData, merged);
            result.textData.push_back('\n');
            ++result.n;
        }
        return result;
    };

    auto writeTileResult = [&](const TileWriteResult& result) {
        write_tile_result(result, false, fp, -1, fdIndex, currentOffset, formatInfo_.tileSize);
        notice("%s: Annotated merged tile (%d, %d) with %u points",
            funcName, result.tile.row, result.tile.col, result.n);
    };
    process_tile_results_parallel(tiles, threads_,
        [&]() { return std::vector<std::ifstream>(nSources); },
        buildTileResult, writeTileResult);

    std::fclose(fp);
    close(fdIndex);
}

void TileOperator::pix2cellSingleMolecule(const std::string& ptPrefix, const std::string& outPrefix,
    uint32_t icol_c, uint32_t icol_x, uint32_t icol_y,
    int32_t icol_s, int32_t icol_z, int32_t icol_f,
    uint32_t k_out, float max_cell_diameter, const std::string& featureDictFile) {
    const std::vector<std::string> featureNames = loadFeatureNames(featureDictFile);
    const auto featureIndex = buildFeatureIndexMap(featureNames);
    const std::string ptData = ptPrefix + ".tsv";
    const std::string ptIndex = ptPrefix + ".index";
    TileReader reader(ptData, ptIndex);
    assert(reader.getTileSize() == formatInfo_.tileSize);
    assert(icol_c >= 0 && icol_c != icol_x && icol_c != icol_y);
    assert(icol_x != icol_y);
    assert(icol_f >= 0 && icol_f != static_cast<int32_t>(icol_c));
    const bool use3d = (coord_dim_ == 3);
    if (use3d) {assert(icol_z >= 0);}
    const bool hasComp = (icol_s >= 0);
    if (k_out == 0) {k_out = static_cast<uint32_t>(k_);}
    if (k_out == 0) {error("%s: Invalid k_out value", __func__);}

    const std::string outFile = outPrefix + ".tsv";
    FILE* fp = std::fopen(outFile.c_str(), "w");
    if (!fp) error("Cannot open output file %s", outFile.c_str());
    std::string headerStr = hasComp ? "#CellID\tCellComp\tnQuery" : "#CellID\tnQuery";
    for (uint32_t i = 1; i <= k_out; ++i) {
        headerStr += "\tK" + std::to_string(i) + "\tP" + std::to_string(i);
    }
    headerStr += "\n";
    std::fprintf(fp, "%s", headerStr.c_str());

    const float resXY = getPixelResolution() > 0.0f ? getPixelResolution() : 1.0f;
    const float resZ = use3d ? getPixelResolutionZ() : 1.0f;
    const int32_t tileSize = formatInfo_.tileSize;
    const bool enableEarlyFlush = (max_cell_diameter > 0);

    uint32_t ntok = std::max(icol_c, std::max(icol_x, icol_y));
    ntok = std::max(ntok, static_cast<uint32_t>(icol_f));
    if (use3d) {
        ntok = std::max(ntok, static_cast<uint32_t>(icol_z));
    }
    if (hasComp) {
        ntok = std::max(ntok, static_cast<uint32_t>(icol_s));
    }
    ntok += 1;

    std::map<std::string, CellAgg> cellAggs;
    std::map<std::string, FactorSums> compTotals;

    std::vector<TileKey> tiles;
    reader.getTileList(tiles);
    int32_t nTilesDone = 0;
    const int32_t nTiles = static_cast<int32_t>(tiles.size());
    notice("%s: Start processing %d tiles", __func__, nTiles);
    for (const auto& tile : tiles) {
        ++nTilesDone;
        std::map<PixelFeatureKey2, TopProbs> pixelMap;
        std::map<PixelFeatureKey3, TopProbs> pixelMap3d;
        if (use3d) {
            if (loadTileToMapFeature3D(tile, pixelMap3d) <= 0) {
                continue;
            }
        } else {
            if (loadTileToMapFeature(tile, pixelMap) <= 0) {
                continue;
            }
        }

        auto it = reader.get_tile_iterator(tile.row, tile.col);
        if (!it) continue;

        const float tileX0 = tile.col * tileSize;
        const float tileX1 = (tile.col + 1) * tileSize;
        const float tileY0 = tile.row * tileSize;
        const float tileY1 = (tile.row + 1) * tileSize;

        std::unordered_set<std::string> tileCells;
        std::string s;
        while (it->next(s)) {
            if (s.empty() || s[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", s, ntok + 1, true, true, true);
            if (tokens.size() < ntok) {
                error("%s: Invalid line: %s", __func__, s.c_str());
            }
            float x = 0.0f, y = 0.0f, z = 0.0f;
            if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
                error("%s: Invalid coordinates in line: %s", __func__, s.c_str());
            }
            if (use3d && !str2float(tokens[icol_z], z)) {
                error("%s: Invalid z coordinate in line: %s", __func__, s.c_str());
            }
            const int32_t featureIdx = require_feature_idx(featureIndex, tokens[icol_f], __func__);
            if (featureIdx < 0) { continue; }
            const int32_t ix = static_cast<int32_t>(std::floor(x / resXY));
            const int32_t iy = static_cast<int32_t>(std::floor(y / resXY));
            const TopProbs* probs = nullptr;
            if (use3d) {
                const int32_t iz = static_cast<int32_t>(std::floor(z / resZ));
                auto pit = pixelMap3d.find(std::make_tuple(ix, iy, iz, featureIdx));
                if (pit == pixelMap3d.end()) {
                    continue;
                }
                probs = &pit->second;
            } else {
                auto pit = pixelMap.find(std::make_tuple(ix, iy, featureIdx));
                if (pit == pixelMap.end()) {
                    continue;
                }
                probs = &pit->second;
            }

            const std::string& cellId = tokens[icol_c];
            const std::string compartment = hasComp ? tokens[icol_s] : std::string();
            auto& agg = cellAggs[cellId];
            agg.sums.second += 1;
            if (hasComp) {
                compTotals[compartment].second += 1;
                agg.compSums[compartment].second += 1;
            }

            if (enableEarlyFlush) {
                tileCells.insert(cellId);
                const float minDist = std::min(
                    std::min(x - tileX0, tileX1 - x),
                    std::min(y - tileY0, tileY1 - y));
                if (minDist <= max_cell_diameter) {
                    agg.boundary = true;
                }
            }

            for (size_t i = 0; i < probs->ks.size(); ++i) {
                const int32_t k = probs->ks[i];
                const double p = probs->ps[i];
                agg.sums.first[k] += p;
                if (hasComp) {
                    agg.compSums[compartment].first[k] += p;
                    compTotals[compartment].first[k] += p;
                }
            }
        }

        if (!enableEarlyFlush || tileCells.empty()) {
            notice("%s: Processed tile [%d, %d] (%d/%d, %lu)", __func__, tile.row, tile.col, nTilesDone, nTiles, cellAggs.size());
            continue;
        }

        int32_t nFlushed = 0;
        for (const auto& cellId : tileCells) {
            auto itCell = cellAggs.find(cellId);
            if (itCell == cellAggs.end()) continue;
            if (itCell->second.boundary) continue;
            if (hasComp) {
                write_cell_row(fp, cellId, "ALL", itCell->second.sums, k_out, true);
                for (const auto& kv : itCell->second.compSums) {
                    write_cell_row(fp, cellId, kv.first, kv.second, k_out, true);
                }
            } else {
                write_cell_row(fp, cellId, "", itCell->second.sums, k_out, false);
            }
            cellAggs.erase(itCell);
            ++nFlushed;
        }
        notice("%s: Processed tile [%d, %d] (%d/%d) with %lu cells, output %d, %lu in buffer",
            __func__, tile.row, tile.col, nTilesDone, nTiles, tileCells.size(), nFlushed, cellAggs.size());
    }

    for (const auto& kv : cellAggs) {
        if (hasComp) {
            write_cell_row(fp, kv.first, "ALL", kv.second.sums, k_out, true);
            for (const auto& comp : kv.second.compSums) {
                write_cell_row(fp, kv.first, comp.first, comp.second, k_out, true);
            }
        } else {
            write_cell_row(fp, kv.first, "", kv.second.sums, k_out, false);
        }
    }

    std::fclose(fp);
    notice("%s: Cell aggregation written to %s", __func__, outFile.c_str());
    if (!hasComp) {
        return;
    }

    const std::string pbFile = outPrefix + ".pseudobulk.tsv";
    FILE* pb = std::fopen(pbFile.c_str(), "w");
    if (!pb) error("Cannot open output file %s", pbFile.c_str());
    std::fprintf(pb, "Factor");
    for (const auto& kv : compTotals) {
        std::fprintf(pb, "\t%s", kv.first.c_str());
    }
    std::fprintf(pb, "\n");
    std::set<int32_t> factors;
    for (const auto& kv : compTotals) {
        for (const auto& fv : kv.second.first) {
            if (fv.second != 0.0) {
                factors.insert(fv.first);
            }
        }
    }
    for (int32_t k : factors) {
        std::fprintf(pb, "%d", k);
        for (const auto& kv : compTotals) {
            double v = 0.0;
            auto it = kv.second.first.find(k);
            if (it != kv.second.first.end()) v = it->second;
            std::fprintf(pb, "\t%.4e", v);
        }
        std::fprintf(pb, "\n");
    }
    std::fclose(pb);
    notice("%s: Pseudobulk matrix written to %s", __func__, pbFile.c_str());
}

void TileOperator::probDotMultiSingleMolecule(const std::vector<std::string>& otherFiles,
    const std::string& outPrefix, std::vector<uint32_t> k2keep, int32_t probDigits) {
    if (k2keep.size() > 0) {assert(k2keep.size() == otherFiles.size() + 1);}
    assert(!otherFiles.empty());

    std::vector<std::unique_ptr<TileOperator>> ops;
    std::vector<TileOperator*> opPtrs;
    opPtrs.push_back(this);
    for (const auto& f : otherFiles) {
        const std::string idxFile = f.substr(0, f.find_last_of('.')) + ".index";
        struct stat buffer;
        if (stat(idxFile.c_str(), &buffer) != 0) {
            error("%s: Index file %s not found", __func__, idxFile.c_str());
        }
        ops.push_back(std::make_unique<TileOperator>(f, idxFile));
        opPtrs.push_back(ops.back().get());
    }
    applyUnsetSourceResolutionOverrides(opPtrs, __func__);
    const uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    for (auto* op : opPtrs) {
        if (!op->hasFeatureIndex()) {
            error("%s: mixed feature/non-feature inputs are not supported", __func__);
        }
        if (op->coord_dim_ != coord_dim_) {
            error("%s: Mixed 2D/3D inputs are not supported", __func__);
        }
    }
    const FeatureRemapPlan featureRemap = buildFeatureRemapPlan(opPtrs, __func__);
    if (k2keep.empty()) {
        for (auto* op : opPtrs) {
            k2keep.push_back(op->getK());
        }
    } else {
        for (uint32_t i = 0; i < nSources; ++i) {
            if (k2keep[i] > opPtrs[i]->getK()) {
                warning("%s: Invalid value k (%d) specified for the %d-th source", __func__, k2keep[i], i);
                k2keep[i] = opPtrs[i]->getK();
            }
        }
    }

    const size_t nSets = k2keep.size();
    std::vector<std::map<int32_t, double>> marginals(nSets);
    std::vector<std::map<std::pair<int32_t, int32_t>, double>> internalDots(nSets);
    std::map<std::pair<size_t, size_t>, std::map<std::pair<int32_t, int32_t>, double>> crossDots;
    std::vector<uint32_t> offsets(nSets + 1, 0);
    for (size_t s = 0; s < nSets; ++s) offsets[s + 1] = offsets[s] + k2keep[s];

    std::set<TileKey> commonTiles;
    for (const auto& kv : opPtrs[0]->tile_lookup_) {
        commonTiles.insert(kv.first);
    }
    for (uint32_t i = 1; i < nSources; ++i) {
        std::set<TileKey> currentTiles;
        for (const auto& kv : opPtrs[i]->tile_lookup_) {
            if (commonTiles.count(kv.first)) {
                currentTiles.insert(kv.first);
            }
        }
        commonTiles = std::move(currentTiles);
    }
    if (commonTiles.empty()) {
        warning("%s: No overlapping tiles found", __func__);
        return;
    }

    size_t count = 0;
    notice("%s: Start computing on %u files (%lu shared tiles)", __func__, nSources, commonTiles.size());
    std::vector<std::ifstream> streams(nSources);
    for (const auto& tile : commonTiles) {
        if (coord_dim_ == 3) {
            std::map<PixelFeatureKey3, TopProbs> mergedMap;
            bool first = true;
            for (uint32_t i = 0; i < nSources; ++i) {
                std::map<PixelFeatureKey3, TopProbs> currentMap;
                if (opPtrs[i]->loadTileToMapFeature3D(tile, currentMap, &streams[i]) == 0) {
                    mergedMap.clear();
                    break;
                }
                remapFeatureMapToCanonical(currentMap, featureRemap.localToCanonical[i], __func__);
                if (first) {
                    if (opPtrs[i]->getK() > static_cast<int32_t>(k2keep[i])) {
                        for (auto& kv : currentMap) {
                            kv.second.ks.resize(k2keep[i]);
                            kv.second.ps.resize(k2keep[i]);
                        }
                    }
                    mergedMap = std::move(currentMap);
                    first = false;
                    continue;
                }
                auto it = mergedMap.begin();
                while (it != mergedMap.end()) {
                    auto it2 = currentMap.find(it->first);
                    if (it2 == currentMap.end()) {
                        it = mergedMap.erase(it);
                        continue;
                    }
                    it->second.ks.insert(it->second.ks.end(),
                        it2->second.ks.begin(), it2->second.ks.begin() + k2keep[i]);
                    it->second.ps.insert(it->second.ps.end(),
                        it2->second.ps.begin(), it2->second.ps.begin() + k2keep[i]);
                    ++it;
                }
            }
            accumulate_probdot_maps(mergedMap, k2keep, offsets, marginals, internalDots, crossDots, count);
            notice("%s: Processed tile (%d, %d) with %lu feature-matched records", __func__, tile.row, tile.col, mergedMap.size());
        } else {
            std::map<PixelFeatureKey2, TopProbs> mergedMap;
            bool first = true;
            for (uint32_t i = 0; i < nSources; ++i) {
                std::map<PixelFeatureKey2, TopProbs> currentMap;
                if (opPtrs[i]->loadTileToMapFeature(tile, currentMap, &streams[i]) == 0) {
                    mergedMap.clear();
                    break;
                }
                remapFeatureMapToCanonical(currentMap, featureRemap.localToCanonical[i], __func__);
                if (first) {
                    if (opPtrs[i]->getK() > static_cast<int32_t>(k2keep[i])) {
                        for (auto& kv : currentMap) {
                            kv.second.ks.resize(k2keep[i]);
                            kv.second.ps.resize(k2keep[i]);
                        }
                    }
                    mergedMap = std::move(currentMap);
                    first = false;
                    continue;
                }
                auto it = mergedMap.begin();
                while (it != mergedMap.end()) {
                    auto it2 = currentMap.find(it->first);
                    if (it2 == currentMap.end()) {
                        it = mergedMap.erase(it);
                        continue;
                    }
                    it->second.ks.insert(it->second.ks.end(),
                        it2->second.ks.begin(), it2->second.ks.begin() + k2keep[i]);
                    it->second.ps.insert(it->second.ps.end(),
                        it2->second.ps.begin(), it2->second.ps.begin() + k2keep[i]);
                    ++it;
                }
            }
            accumulate_probdot_maps(mergedMap, k2keep, offsets, marginals, internalDots, crossDots, count);
            notice("%s: Processed tile (%d, %d) with %lu feature-matched records", __func__, tile.row, tile.col, mergedMap.size());
        }
    }

    notice("%s: Processed %lu shared records", __func__, count);
    for (size_t s = 0; s < nSets; ++s) {
        std::string fn = outPrefix + "." + std::to_string(s) + ".marginal.tsv";
        FILE* fp = std::fopen(fn.c_str(), "w");
        if (!fp) error("Cannot open output file %s", fn.c_str());
        std::fprintf(fp, "#K\tSum\n");
        for (const auto& kv : marginals[s]) {
            std::fprintf(fp, "%d\t%.*e\n", kv.first, probDigits, kv.second);
        }
        std::fclose(fp);
    }
    for (size_t s = 0; s < nSets; ++s) {
        std::string fn = outPrefix + "." + std::to_string(s) + ".joint.tsv";
        FILE* fp = std::fopen(fn.c_str(), "w");
        if (!fp) error("Cannot open output file %s", fn.c_str());
        std::fprintf(fp, "#K1\tK2\tJoint\n");
        for (const auto& kv : internalDots[s]) {
            std::fprintf(fp, "%d\t%d\t%.*e\n", kv.first.first, kv.first.second, probDigits, kv.second);
        }
        std::fclose(fp);
    }
    for (const auto& [setPair, mapVal] : crossDots) {
        std::string fn = outPrefix + "." + std::to_string(setPair.first) + "v" + std::to_string(setPair.second) + ".cross.tsv";
        FILE* fp = std::fopen(fn.c_str(), "w");
        if (!fp) error("Cannot open output file %s", fn.c_str());
        std::map<int32_t, double> rowSums;
        std::map<int32_t, double> colSums;
        double total = 0.0;
        double pseudo = 0.5;
        for (const auto& kv : mapVal) {
            rowSums[kv.first.first] += kv.second;
            colSums[kv.first.second] += kv.second;
            total += kv.second;
            if (kv.second > 0 && kv.second < pseudo) {
                pseudo = kv.second;
            }
        }
        std::map<int32_t, double> colFreq = colSums;
        for (auto& kv : colFreq) {kv.second /= total;}
        pseudo *= 0.5;
        std::fprintf(fp, "#K1\tK2\tJoint\tlog10pval\tlog2OR\n");
        for (const auto& kv : mapVal) {
            const double a = kv.second;
            const double rowSum = rowSums[kv.first.first];
            const double colSum = colSums[kv.first.second];
            const double b = std::max(0.0, rowSum - a);
            const double c = std::max(0.0, colSum - a);
            const double d = std::max(0.0, total - rowSum - colSum + a);
            const double log2OR = std::log2(a + pseudo) - std::log2(rowSums[kv.first.first] * colFreq[kv.first.second] + pseudo);
            auto stats = chisq2x2_log10p(a, b, c, d, pseudo);
            std::fprintf(fp, "%d\t%d\t%.*e\t%.4e\t%.4e\n",
                kv.first.first, kv.first.second, probDigits, kv.second, stats.second, log2OR);
        }
        std::fclose(fp);
    }
}

void TileOperator::mergeSingleMolecule(const std::vector<std::string>& otherFiles,
    const std::string& outPrefix, std::vector<uint32_t> k2keep,
    bool binaryOutput, bool keepAllMain, bool keepAll,
    const std::string& featureDictFile,
    const std::vector<std::string>& mergePrefixes) {
    (void) featureDictFile;
    std::vector<std::unique_ptr<TileOperator>> ops;
    std::vector<TileOperator*> opPtrs;
    opPtrs.push_back(this);
    for (const auto& f : otherFiles) {
        const std::string idxFile = f.substr(0, f.find_last_of('.')) + ".index";
        struct stat buffer;
        if (stat(idxFile.c_str(), &buffer) != 0) {
            error("%s: Index file %s not found", __func__, idxFile.c_str());
        }
        ops.push_back(std::make_unique<TileOperator>(f, idxFile));
        opPtrs.push_back(ops.back().get());
    }
    const uint32_t nSources = static_cast<uint32_t>(opPtrs.size());
    if (k2keep.size() > 0) {assert(k2keep.size() == nSources);}
    if (k2keep.empty()) {
        for (const auto* op : opPtrs) {
            k2keep.push_back(op->getK());
        }
    } else {
        for (uint32_t i = 0; i < nSources; ++i) {
            if (k2keep[i] > opPtrs[i]->getK()) {
                warning("%s: Invalid value k (%d) specified for the %d-th source", __func__, k2keep[i], i);
                k2keep[i] = opPtrs[i]->getK();
            }
        }
    }
    if (nSources > 7) {
        int32_t k = *std::min_element(k2keep.begin(), k2keep.end());
        k2keep.assign(nSources, static_cast<uint32_t>(k));
        warning("%s: More than 7 files to merge, keep %d values each", __func__, k);
    }
    bool seenNonFeature = false;
    for (size_t i = 0; i < opPtrs.size(); ++i) {
        if (!opPtrs[i]->hasFeatureIndex()) {
            seenNonFeature = true;
        } else if (seenNonFeature) {
            error("%s: feature-bearing sources must come before feature-less auxiliary sources", __func__);
        }
    }
    const std::vector<MergeSourcePlan> mergePlans = validateMergeSources(opPtrs, k2keep);
    const FeatureRemapPlan featureRemap = buildFeatureRemapPlan(opPtrs, __func__);
    const std::vector<std::string>& featureNames = featureRemap.canonicalNames;
    if (keepAll) {
        for (size_t i = 1; i < mergePlans.size(); ++i) {
            if (mergePlans[i].relation == MergeSourceRelation::Broadcast2DTo3D) {
                error("%s: --merge-keep-all does not support 2D auxiliary sources when the main input is 3D", __func__);
            }
            if (!mergePlans[i].op->hasFeatureIndex()) {
                error("%s: --merge-keep-all on single-molecule input requires all merged sources to carry feature indices", __func__);
            }
        }
    }
    const bool use3d = (coord_dim_ == 3);
    const int32_t totalK = std::accumulate(k2keep.begin(), k2keep.end(), 0);

    std::vector<TileKey> mainTiles;
    for (const auto& kv : tile_lookup_) {
        mainTiles.push_back(kv.first);
    }
    std::sort(mainTiles.begin(), mainTiles.end());
    if (mainTiles.empty()) {
        warning("%s: No tiles in the main dataset", __func__);
        return;
    }

    const std::string outIndex = outPrefix + ".index";
    int fdIndex = open(outIndex.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) error("Cannot open output index %s", outIndex.c_str());

    IndexHeader idxHeader = formatInfo_;
    idxHeader.mode |= 0x4u;
    idxHeader.mode |= 0x40u;
    if (!idxHeader.packKvec(k2keep)) {
        warning("%s: Too many input fields", __func__);
    }

    std::string outFile;
    FILE* fp = nullptr;
    int fdMain = -1;
    long currentOffset = 0;
    if (binaryOutput) {
        outFile = outPrefix + ".bin";
        fdMain = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (fdMain < 0) error("Cannot open output file %s", outFile.c_str());
        idxHeader.mode |= 0x1u;
        const uint32_t coordCount = use3d ? 3u : 2u;
        idxHeader.recordSize = sizeof(int32_t) * coordCount + sizeof(uint32_t) +
            sizeof(int32_t) * totalK + sizeof(float) * totalK;
    } else {
        outFile = outPrefix + ".tsv";
        fp = std::fopen(outFile.c_str(), "w");
        if (!fp) error("Cannot open output file %s", outFile.c_str());
        idxHeader.mode &= ~0x41u;
        idxHeader.recordSize = 0;
        std::string headerStr = "#x\ty";
        if (use3d) {
            headerStr += "\tz";
        }
        headerStr += "\tfeature";
        for (const auto& colName : build_merge_column_names(k2keep, mergePrefixes)) {
            headerStr += "\t" + colName;
        }
        headerStr += "\n";
        std::fprintf(fp, "%s", headerStr.c_str());
        currentOffset = std::ftell(fp);
    }
    configureFeatureDictionaryHeader(idxHeader, featureNames, __func__);
    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
        error("%s: index header write error", __func__);
    }
    if (!writeFeatureDictionaryPayload(fdIndex, idxHeader, featureNames)) {
        error("%s: feature dictionary write error", __func__);
    }

    const char* funcName = __func__;
    notice("%s: Start merging %u files across %lu main tiles", funcName, nSources, mainTiles.size());
    auto buildTileResult = [&](const TileKey& tile, std::vector<std::ifstream>& streams) {
        TileWriteResult result;
        result.tile = tile;
        if (use3d) {
            std::map<PixelFeatureKey3, TopProbs> mainMap;
            if (loadTileToMapFeature3D(tile, mainMap, &streams[0]) <= 0) {
                return result;
            }
            remapFeatureMapToCanonical(mainMap, featureRemap.localToCanonical[0], funcName);
            result.nMain = static_cast<uint32_t>(mainMap.size());
            std::vector<std::map<TileKey, std::map<PixelFeatureKey2, TopProbs>>> auxFeature2D(nSources);
            std::vector<std::map<TileKey, std::map<PixelFeatureKey3, TopProbs>>> auxFeature3D(nSources);
            std::vector<std::map<TileKey, std::map<std::pair<int32_t, int32_t>, TopProbs>>> auxPlain2D(nSources);
            std::vector<std::map<TileKey, std::map<PixelKey3, TopProbs>>> auxPlain3D(nSources);
            std::vector<std::set<TileKey>> missing(nSources);
            std::vector<std::map<PixelFeatureKey3, TopProbs>> sourceTileMaps(nSources);
            sourceTileMaps[0] = mainMap;
            if (keepAll) {
                for (size_t srcIdx = 1; srcIdx < nSources; ++srcIdx) {
                    mergePlans[srcIdx].op->loadTileToMapFeature3D(tile, sourceTileMaps[srcIdx], &streams[srcIdx]);
                    remapFeatureMapToCanonical(sourceTileMaps[srcIdx], featureRemap.localToCanonical[srcIdx], funcName);
                }
            }
            std::set<PixelFeatureKey3> outputKeys;
            for (const auto& kv : mainMap) {
                outputKeys.insert(kv.first);
            }
            if (keepAll) {
                for (size_t srcIdx = 1; srcIdx < nSources; ++srcIdx) {
                    const MergeSourcePlan& plan = mergePlans[srcIdx];
                    for (const auto& kv : sourceTileMaps[srcIdx]) {
                        outputKeys.insert(std::make_tuple(
                            std::get<0>(kv.first) * plan.ratioXY,
                            std::get<1>(kv.first) * plan.ratioXY,
                            std::get<2>(kv.first) * plan.ratioZ,
                            std::get<3>(kv.first)));
                    }
                }
            }
            for (const auto& key : outputKeys) {
                const int32_t mainX = std::get<0>(key);
                const int32_t mainY = std::get<1>(key);
                const int32_t mainZ = std::get<2>(key);
                const uint32_t featureIdx = std::get<3>(key);
                TopProbs merged;
                merged.ks.reserve(totalK);
                merged.ps.reserve(totalK);
                bool anyFound = false;
                bool allFound = true;
                bool mainFound = false;
                for (size_t i = 0; i < nSources; ++i) {
                    const uint32_t keepK = mergePlans[i].keepK;
                    const TopProbs* aux = nullptr;
                    if (i == 0) {
                        auto it0 = mainMap.find(key);
                        if (it0 != mainMap.end()) {
                            aux = &it0->second;
                        }
                    } else {
                    const MergeSourcePlan& plan = mergePlans[i];
                    const int32_t auxX = floorDivInt32(mainX, plan.ratioXY);
                    const int32_t auxY = floorDivInt32(mainY, plan.ratioXY);
                    const TileKey auxTile = tile_key_from_source_xy(auxX, auxY, plan.srcResXY, plan.tileSize);
                    if (missing[i].count(auxTile) > 0) {
                        } else if (plan.relation == MergeSourceRelation::Broadcast2DTo3D) {
                            auto tileIt = auxFeature2D[i].find(auxTile);
                            if (tileIt == auxFeature2D[i].end()) {
                                std::map<PixelFeatureKey2, TopProbs> auxMap;
                                if (plan.op->loadTileToMapFeature(auxTile, auxMap, &streams[i]) == 0) {
                                    missing[i].insert(auxTile);
                                } else {
                                    remapFeatureMapToCanonical(auxMap, featureRemap.localToCanonical[i], funcName);
                                    tileIt = auxFeature2D[i].emplace(auxTile, std::move(auxMap)).first;
                                }
                            }
                            if (tileIt != auxFeature2D[i].end()) {
                                auto recIt = tileIt->second.find(std::make_tuple(auxX, auxY, featureIdx));
                                if (recIt != tileIt->second.end()) aux = &recIt->second;
                            }
                        } else {
                            const int32_t auxZ = floorDivInt32(mainZ, plan.ratioZ);
                            auto tileIt = auxFeature3D[i].find(auxTile);
                            if (tileIt == auxFeature3D[i].end()) {
                                std::map<PixelFeatureKey3, TopProbs> auxMap;
                                if (plan.op->loadTileToMapFeature3D(auxTile, auxMap, &streams[i]) == 0) {
                                    missing[i].insert(auxTile);
                                } else {
                                    remapFeatureMapToCanonical(auxMap, featureRemap.localToCanonical[i], funcName);
                                    tileIt = auxFeature3D[i].emplace(auxTile, std::move(auxMap)).first;
                                }
                            }
                            if (tileIt != auxFeature3D[i].end()) {
                                auto recIt = tileIt->second.find(std::make_tuple(auxX, auxY, auxZ, featureIdx));
                                if (recIt != tileIt->second.end()) aux = &recIt->second;
                            }
                        }
                    }
                    if (aux == nullptr) {
                        allFound = false;
                        append_placeholder_pairs(merged, keepK);
                    } else {
                        anyFound = true;
                        if (i == 0) {
                            mainFound = true;
                        }
                        append_top_probs_prefix(merged, *aux, keepK);
                    }
                }
                const bool emit = keepAll ? anyFound : (keepAllMain ? mainFound : (mainFound && allFound));
                if (!emit) continue;
                ++result.n;
                if (binaryOutput) {
                    append_pix_top_probs_feature3d_binary(result.binaryData, mainX, mainY, mainZ, featureIdx, merged);
                } else {
                    if (featureIdx >= featureNames.size()) {
                        error("%s: feature index %u out of range for dictionary of size %zu", funcName, featureIdx, featureNames.size());
                    }
                    append_format(result.textData, "%d\t%d\t%d\t%s", mainX, mainY, mainZ, featureNames[featureIdx].c_str());
                    appendTopProbsText(result.textData, merged);
                    result.textData.push_back('\n');
                }
            }
            return result;
        }

        std::map<PixelFeatureKey2, TopProbs> mainMap;
        if (loadTileToMapFeature(tile, mainMap, &streams[0]) <= 0) {
            return result;
        }
        remapFeatureMapToCanonical(mainMap, featureRemap.localToCanonical[0], funcName);
        result.nMain = static_cast<uint32_t>(mainMap.size());
        std::vector<std::map<PixelFeatureKey2, TopProbs>> sourceTileMaps(nSources);
        sourceTileMaps[0] = mainMap;
        if (keepAll) {
            for (size_t srcIdx = 1; srcIdx < nSources; ++srcIdx) {
                mergePlans[srcIdx].op->loadTileToMapFeature(tile, sourceTileMaps[srcIdx], &streams[srcIdx]);
                remapFeatureMapToCanonical(sourceTileMaps[srcIdx], featureRemap.localToCanonical[srcIdx], funcName);
            }
        }
        std::vector<std::map<TileKey, std::map<PixelFeatureKey2, TopProbs>>> auxFeature2D(nSources);
        std::vector<std::map<TileKey, std::map<std::pair<int32_t, int32_t>, TopProbs>>> auxPlain2D(nSources);
        std::vector<std::set<TileKey>> missing(nSources);
        std::set<PixelFeatureKey2> outputKeys;
        for (const auto& kv : mainMap) {
            outputKeys.insert(kv.first);
        }
        if (keepAll) {
            for (size_t srcIdx = 1; srcIdx < nSources; ++srcIdx) {
                const MergeSourcePlan& plan = mergePlans[srcIdx];
                for (const auto& kv : sourceTileMaps[srcIdx]) {
                    outputKeys.insert(std::make_tuple(
                        std::get<0>(kv.first) * plan.ratioXY,
                        std::get<1>(kv.first) * plan.ratioXY,
                        std::get<2>(kv.first)));
                }
            }
        }
        for (const auto& key : outputKeys) {
            const int32_t mainX = std::get<0>(key);
            const int32_t mainY = std::get<1>(key);
            const uint32_t featureIdx = std::get<2>(key);
            TopProbs merged;
            merged.ks.reserve(totalK);
            merged.ps.reserve(totalK);
            bool anyFound = false;
            bool allFound = true;
            bool mainFound = false;
            for (size_t i = 0; i < nSources; ++i) {
                const uint32_t keepK = mergePlans[i].keepK;
                const TopProbs* aux = nullptr;
                if (i == 0) {
                    auto it0 = mainMap.find(key);
                    if (it0 != mainMap.end()) {
                        aux = &it0->second;
                    }
                } else {
                const MergeSourcePlan& plan = mergePlans[i];
                const int32_t auxX = floorDivInt32(mainX, plan.ratioXY);
                const int32_t auxY = floorDivInt32(mainY, plan.ratioXY);
                const TileKey auxTile = tile_key_from_source_xy(auxX, auxY, plan.srcResXY, plan.tileSize);
                if (missing[i].count(auxTile) > 0) {
                } else {
                    auto tileIt = auxFeature2D[i].find(auxTile);
                    if (tileIt == auxFeature2D[i].end()) {
                        std::map<PixelFeatureKey2, TopProbs> auxMap;
                        if (plan.op->loadTileToMapFeature(auxTile, auxMap, &streams[i]) == 0) {
                            missing[i].insert(auxTile);
                        } else {
                            remapFeatureMapToCanonical(auxMap, featureRemap.localToCanonical[i], funcName);
                            tileIt = auxFeature2D[i].emplace(auxTile, std::move(auxMap)).first;
                        }
                    }
                    if (tileIt != auxFeature2D[i].end()) {
                        auto recIt = tileIt->second.find(std::make_tuple(auxX, auxY, featureIdx));
                        if (recIt != tileIt->second.end()) aux = &recIt->second;
                    }
                }
                }
                if (aux == nullptr) {
                    allFound = false;
                    append_placeholder_pairs(merged, keepK);
                } else {
                    anyFound = true;
                    if (i == 0) {
                        mainFound = true;
                    }
                    append_top_probs_prefix(merged, *aux, keepK);
                }
            }
            const bool emit = keepAll ? anyFound : (keepAllMain ? mainFound : (mainFound && allFound));
            if (!emit) continue;
            ++result.n;
            if (binaryOutput) {
                append_pix_top_probs_feature_binary(result.binaryData, mainX, mainY, featureIdx, merged);
            } else {
                if (featureIdx >= featureNames.size()) {
                    error("%s: feature index %u out of range for dictionary of size %zu", funcName, featureIdx, featureNames.size());
                }
                append_format(result.textData, "%d\t%d\t%s", mainX, mainY, featureNames[featureIdx].c_str());
                appendTopProbsText(result.textData, merged);
                result.textData.push_back('\n');
            }
        }
        return result;
    };

    auto writeResult = [&](const TileWriteResult& result) {
        write_tile_result(result, binaryOutput, fp, fdMain, fdIndex, currentOffset, formatInfo_.tileSize);
        notice("%s: Merged tile (%d, %d) with %u output records from %u main records",
            funcName, result.tile.row, result.tile.col, result.n, result.nMain);
    };
    process_tile_results_parallel(mainTiles, threads_,
        [&]() { return std::vector<std::ifstream>(nSources); },
        buildTileResult, writeResult);

    if (binaryOutput) {
        close(fdMain);
    } else {
        std::fclose(fp);
    }
    close(fdIndex);
    notice("Merged %u files across %lu main tiles to %s", nSources, mainTiles.size(), outFile.c_str());
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
TileOperator::computeConfusionMatrixSingleMolecule(double resolution) const {
    if (coord_dim_ != 2) {
        error("%s: only 2D data are supported", __func__);
    }
    if (K_ <= 0) {
        error("%s: K is 0 or unknown", __func__);
    }
    const int32_t K = K_;
    float res = formatInfo_.pixelResolution;
    if (res <= 0) res = 1.0f;
    if (resolution > 0) res /= resolution;
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> confusion;
    confusion.setZero(K, K);

    auto accumulateTileConfusion = [&](const std::map<PixelFeatureKey2, TopProbs>& pixelMap) {
        if (resolution > 0) {
            std::unordered_map<std::pair<int32_t, int32_t>, Eigen::VectorXd, PairHash> squareSums;
            for (const auto& kv : pixelMap) {
                const int32_t x = std::get<0>(kv.first);
                const int32_t y = std::get<1>(kv.first);
                const int32_t sx = static_cast<int32_t>(std::floor(x * res));
                const int32_t sy = static_cast<int32_t>(std::floor(y * res));
                auto& merged = squareSums[std::make_pair(sx, sy)];
                if (merged.size() == 0) {
                    merged = Eigen::VectorXd::Zero(K);
                }
                const TopProbs& tp = kv.second;
                for (size_t i = 0; i < tp.ks.size(); ++i) {
                    const int32_t k = tp.ks[i];
                    if (k < 0 || k >= K) continue;
                    merged[k] += tp.ps[i];
                }
            }
            for (auto& kv : squareSums) {
                auto& merged = kv.second;
                const double w = merged.sum();
                if (w == 0.0) continue;
                merged = merged.array() / w;
                confusion += merged * merged.transpose() * w;
            }
            return;
        }
        for (const auto& kv : pixelMap) {
            const TopProbs& tp = kv.second;
            for (size_t i = 0; i < tp.ks.size(); ++i) {
                const int32_t k1 = tp.ks[i];
                const float p1 = tp.ps[i];
                if (k1 < 0 || k1 >= K) continue;
                for (size_t j = 0; j < tp.ks.size(); ++j) {
                    const int32_t k2 = tp.ks[j];
                    const float p2 = tp.ps[j];
                    if (k2 < 0 || k2 >= K) continue;
                    confusion(k1, k2) += static_cast<double>(p1) * p2;
                }
            }
        }
    };

    std::ifstream tileStream;
    int32_t nTiles = 0;
    for (const auto& tileInfo : blocks_) {
        ++nTiles;
        TileKey tile{tileInfo.row, tileInfo.col};
        std::map<PixelFeatureKey2, TopProbs> pixelMap;
        if (loadTileToMapFeature(tile, pixelMap, &tileStream) <= 0) {
            continue;
        }
        accumulateTileConfusion(pixelMap);
        if (nTiles % 10 == 0) {
            notice("%s: Processed %d tiles...", __func__, nTiles);
        }
    }
    return confusion;
}
