#include "tileoperator.hpp"
#include "region_query.hpp"
#include <cstdio>
#include <numeric>
#include <limits>
#include <chrono>
#include <tbb/blocked_range.h>
#include <tbb/combinable.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

namespace {

enum class TemplateGeoJSONKind {
    RootFeatureCollection,
    RootFeature,
    WrappedFeatureCollection,
    WrappedFeature
};

struct TemplateGeoJSON {
    nlohmann::json base;
    TemplateGeoJSONKind kind = TemplateGeoJSONKind::RootFeatureCollection;
};

struct FocalMaskRegion {
    int32_t focalK = -1;
    PreparedRegionMask2D region;
};

void write_factor_mask_info(const std::string& outHist,
    const std::vector<uint64_t>& tileCount,
    const std::vector<uint64_t>& maskArea,
    const std::vector<uint64_t>& nComponents) {
    FILE* fp = fopen(outHist.c_str(), "w");
    if (!fp) {
        error("%s: Cannot open output file %s", __func__, outHist.c_str());
    }
    fprintf(fp, "k\tn_tiles\tmask_area_pix\tn_components\n");
    for (size_t k = 0; k < maskArea.size(); ++k) {
        fprintf(fp, "%zu\t%llu\t%llu\t%llu\n",
            k,
            static_cast<unsigned long long>(tileCount[k]),
            static_cast<unsigned long long>(maskArea[k]),
            static_cast<unsigned long long>(nComponents[k]));
    }
    fclose(fp);
}

void write_factor_mask_component_histogram(const std::string& outHist,
    const std::vector<std::unordered_map<uint64_t, uint64_t>>& perFactorHist) {
    FILE* fp = fopen(outHist.c_str(), "w");
    if (!fp) {
        error("%s: Cannot open output file %s", __func__, outHist.c_str());
    }
    fprintf(fp, "k\tsize\tn_components\n");
    for (size_t k = 0; k < perFactorHist.size(); ++k) {
        std::vector<std::pair<uint64_t, uint64_t>> entries(
            perFactorHist[k].begin(), perFactorHist[k].end());
        std::sort(entries.begin(), entries.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
        for (const auto& kv : entries) {
            fprintf(fp, "%zu\t%llu\t%llu\n",
                k,
                static_cast<unsigned long long>(kv.first),
                static_cast<unsigned long long>(kv.second));
        }
    }
    fclose(fp);
}

TemplateGeoJSON load_template_json(const std::string& templateFile) {
    auto isGeometryType = [](const std::string& type) {
        return type == "Point" || type == "MultiPoint" ||
            type == "LineString" || type == "MultiLineString" ||
            type == "Polygon" || type == "MultiPolygon" ||
            type == "GeometryCollection";
    };
    std::ifstream in(templateFile);
    if (!in.is_open()) {
        error("%s: Cannot open template GeoJSON file %s", __func__, templateFile.c_str());
    }
    TemplateGeoJSON out;
    in >> out.base;
    if (!out.base.is_object()) {
        error("%s: Template must be a JSON object", __func__);
    }
    if (out.base.contains("type") && out.base["type"].is_string()) {
        const std::string type = out.base["type"].get<std::string>();
        if (type == "FeatureCollection") {
            out.kind = TemplateGeoJSONKind::RootFeatureCollection;
            out.base.erase("features");
        } else if (type == "Feature") {
            out.kind = TemplateGeoJSONKind::RootFeature;
            out.base.erase("geometry");
            out.base.erase("properties");
        } else if (isGeometryType(type)) {
            nlohmann::json stripped = out.base;
            for (auto key : {"coordinates", "geometries"}) {
                stripped.erase(key);
            }
            stripped["type"] = "FeatureCollection";
            stripped.erase("features");
            out.base = std::move(stripped);
            out.kind = TemplateGeoJSONKind::RootFeatureCollection;
        } else {
            error("%s: Unsupported GeoJSON template type '%s'", __func__, type.c_str());
        }
        return out;
    }
    if (out.base.contains("geometry") &&
        out.base["geometry"].is_object() &&
        out.base["geometry"].contains("type") &&
        out.base["geometry"]["type"].is_string()) {
        const std::string type = out.base["geometry"]["type"].get<std::string>();
        if (type == "FeatureCollection") {
            out.kind = TemplateGeoJSONKind::WrappedFeatureCollection;
            out.base["geometry"].erase("features");
        } else if (type == "Feature") {
            out.kind = TemplateGeoJSONKind::WrappedFeature;
            out.base.erase("geometry");
        } else if (isGeometryType(type)) {
            out.kind = TemplateGeoJSONKind::WrappedFeature;
            out.base.erase("geometry");
        } else {
            error("%s: Unsupported embedded GeoJSON template type '%s'", __func__, type.c_str());
        }
        return out;
    }
    error("%s: Template must be a GeoJSON object or contain a GeoJSON object in 'geometry'", __func__);
    return TemplateGeoJSON{};
}

void write_factor_template_geojson(const std::string& outFile, const TemplateGeoJSON& templateGeoJSON,
    int32_t factor, const nlohmann::json& feature) {
    nlohmann::json outJson = templateGeoJSON.base;
    outJson["title"] = std::to_string(factor);
    switch (templateGeoJSON.kind) {
        case TemplateGeoJSONKind::RootFeatureCollection:
            outJson["features"] = nlohmann::json::array({feature});
            break;
        case TemplateGeoJSONKind::RootFeature:
            outJson["geometry"] = feature.at("geometry");
            outJson["properties"] = feature.value("properties", nlohmann::json::object());
            break;
        case TemplateGeoJSONKind::WrappedFeatureCollection:
            outJson["geometry"]["features"] = nlohmann::json::array({feature});
            break;
        case TemplateGeoJSONKind::WrappedFeature:
            outJson["geometry"] = feature;
            break;
    }
    std::ofstream out(outFile);
    if (!out.is_open()) {
        error("%s: Cannot open output file %s", __func__, outFile.c_str());
    }
    out << outJson.dump();
    out << "\n";
    out.close();
}

size_t pair_index_upper(size_t i, size_t j, size_t n) {
    return i * (2 * n - i - 1) / 2 + (j - i - 1);
}

int32_t parse_feature_factor(const nlohmann::json& feature) {
    const auto propIt = feature.find("properties");
    if (propIt == feature.end() || !propIt->is_object()) {
        error("%s: Mask feature is missing object-valued 'properties'", __func__);
    }
    const nlohmann::json& props = *propIt;
    const auto it = props.find("Factor");
    if (it == props.end()) {
        error("%s: Mask feature is missing integer property 'Factor'", __func__);
    }
    if (!it->is_number_integer() && !it->is_number_unsigned()) {
        error("%s: Mask feature property 'Factor' must be an integer", __func__);
    }
    return it->get<int32_t>();
}

std::vector<FocalMaskRegion> load_focal_mask_regions(const std::string& geojsonFile,
    int32_t tileSize, const std::vector<uint8_t>& selectedFocal, std::vector<uint8_t>* foundFocal) {
    std::ifstream in(geojsonFile);
    if (!in.is_open()) {
        error("%s: Cannot open GeoJSON file %s", __func__, geojsonFile.c_str());
    }

    nlohmann::json root;
    in >> root;
    if (!root.is_object()) {
        error("%s: GeoJSON root must be an object", __func__);
    }

    std::vector<FocalMaskRegion> out;
    std::vector<uint8_t> seenFeatureFactor(foundFocal ? foundFocal->size() : 0, 0);
    auto addFeature = [&](const nlohmann::json& feature) {
        if (!feature.is_object()) {
            error("%s: GeoJSON feature must be an object", __func__);
        }
        const auto typeIt = feature.find("type");
        if (typeIt == feature.end() || !typeIt->is_string() || typeIt->get<std::string>() != "Feature") {
            error("%s: Expected GeoJSON Feature entries", __func__);
        }
        const int32_t focalK = parse_feature_factor(feature);
        if (focalK < 0 || static_cast<size_t>(focalK) >= selectedFocal.size()) {
            error("%s: Focal factor %d in GeoJSON is out of range", __func__, focalK);
        }
        if (!selectedFocal.empty() && !selectedFocal[static_cast<size_t>(focalK)]) {
            return;
        }
        if (!seenFeatureFactor.empty() && seenFeatureFactor[static_cast<size_t>(focalK)]) {
            error("%s: Duplicate feature for focal factor %d in GeoJSON", __func__, focalK);
        }
        const auto geomIt = feature.find("geometry");
        if (geomIt == feature.end() || !geomIt->is_object()) {
            error("%s: Mask feature for focal factor %d is missing object-valued geometry", __func__, focalK);
        }
        FocalMaskRegion entry;
        entry.focalK = focalK;
        try {
            entry.region = prepareRegionFromGeoJSONGeometry(*geomIt, tileSize);
        } catch (const std::exception& ex) {
            warning("%s: Skipping focal factor %d due to invalid/empty geometry: %s",
                __func__, focalK, ex.what());
            return;
        }
        out.push_back(std::move(entry));
        if (foundFocal) {
            (*foundFocal)[static_cast<size_t>(focalK)] = 1;
        }
        if (!seenFeatureFactor.empty()) {
            seenFeatureFactor[static_cast<size_t>(focalK)] = 1;
        }
    };

    const auto typeIt = root.find("type");
    if (typeIt == root.end() || !typeIt->is_string()) {
        error("%s: GeoJSON root is missing string field 'type'", __func__);
    }
    const std::string type = typeIt->get<std::string>();
    if (type == "FeatureCollection") {
        const auto featuresIt = root.find("features");
        if (featuresIt == root.end() || !featuresIt->is_array()) {
            error("%s: GeoJSON FeatureCollection is missing array-valued 'features'", __func__);
        }
        for (const auto& feature : *featuresIt) {
            addFeature(feature);
        }
    } else if (type == "Feature") {
        addFeature(root);
    } else {
        error("%s: Expected GeoJSON FeatureCollection or Feature, got '%s'", __func__, type.c_str());
    }

    std::sort(out.begin(), out.end(),
        [](const FocalMaskRegion& a, const FocalMaskRegion& b) { return a.focalK < b.focalK; });
    return out;
}

} // namespace

TileOperator::BorderMergeResult TileOperator::mergeSoftMaskTileBorders(
    const std::vector<SoftMaskTileResult>& perTile, uint32_t invalid) const {
    if (perTile.size() != blocks_.size()) {
        error("%s: perTile/block size mismatch", __func__);
    }

    std::vector<std::vector<size_t>> borderBase(perTile.size());
    size_t totalBorderComps = 0;
    for (size_t ti = 0; ti < perTile.size(); ++ti) {
        const SoftMaskTileResult& tile = perTile[ti];
        borderBase[ti].assign(tile.factors.size(), std::numeric_limits<size_t>::max());
        for (size_t fi = 0; fi < tile.factors.size(); ++fi) {
            const auto& factorRes = tile.factors[fi];
            if (factorRes.borderCompAreas.empty()) {
                continue;
            }
            borderBase[ti][fi] = totalBorderComps;
            totalBorderComps += factorRes.borderCompAreas.size();
        }
    }

    BorderMergeResult out;
    out.rootArea.assign(totalBorderComps, 0);
    out.rootFactor.assign(totalBorderComps, -1);
    out.rootMembers.resize(totalBorderComps);
    if (totalBorderComps == 0) {
        return out;
    }

    DisjointSet dsu(totalBorderComps);
    for (size_t ti = 0; ti < perTile.size(); ++ti) {
        const SoftMaskTileResult& aTile = perTile[ti];
        const TileKey key{blocks_[ti].idx.row, blocks_[ti].idx.col};
        const auto rightIt = tile_lookup_.find(TileKey{key.row, key.col + 1});
        if (rightIt != tile_lookup_.end()) {
            const size_t tj = rightIt->second;
            const SoftMaskTileResult& bTile = perTile[tj];
            for (size_t afi = 0; afi < aTile.factors.size(); ++afi) {
                const auto& a = aTile.factors[afi];
                if (a.borderCompAreas.empty()) continue;
                const auto bIt = bTile.factorToIndex.find(a.factor);
                if (bIt == bTile.factorToIndex.end()) continue;
                const auto& b = bTile.factors[bIt->second];
                if (b.borderCompAreas.empty()) continue;
                const int32_t y0 = std::max(aTile.geom.pixY0, bTile.geom.pixY0);
                const int32_t y1 = std::min(aTile.geom.pixY1, bTile.geom.pixY1);
                uint32_t lastA = invalid, lastB = invalid;
                for (int32_t gy = y0; gy < y1; ++gy) {
                    const uint32_t ca = a.rightCid[static_cast<size_t>(gy - aTile.geom.pixY0)];
                    const uint32_t cb = b.leftCid[static_cast<size_t>(gy - bTile.geom.pixY0)];
                    if (ca == invalid || cb == invalid || (ca == lastA && cb == lastB)) continue;
                    lastA = ca;
                    lastB = cb;
                    dsu.unite(borderBase[ti][afi] + static_cast<size_t>(ca),
                        borderBase[tj][bIt->second] + static_cast<size_t>(cb));
                }
            }
        }

        const auto downIt = tile_lookup_.find(TileKey{key.row + 1, key.col});
        if (downIt != tile_lookup_.end()) {
            const size_t tj = downIt->second;
            const SoftMaskTileResult& bTile = perTile[tj];
            for (size_t afi = 0; afi < aTile.factors.size(); ++afi) {
                const auto& a = aTile.factors[afi];
                if (a.borderCompAreas.empty()) continue;
                const auto bIt = bTile.factorToIndex.find(a.factor);
                if (bIt == bTile.factorToIndex.end()) continue;
                const auto& b = bTile.factors[bIt->second];
                if (b.borderCompAreas.empty()) continue;
                const int32_t x0 = std::max(aTile.geom.pixX0, bTile.geom.pixX0);
                const int32_t x1 = std::min(aTile.geom.pixX1, bTile.geom.pixX1);
                uint32_t lastA = invalid, lastB = invalid;
                for (int32_t gx = x0; gx < x1; ++gx) {
                    const uint32_t ca = a.bottomCid[static_cast<size_t>(gx - aTile.geom.pixX0)];
                    const uint32_t cb = b.topCid[static_cast<size_t>(gx - bTile.geom.pixX0)];
                    if (ca == invalid || cb == invalid || (ca == lastA && cb == lastB)) continue;
                    lastA = ca;
                    lastB = cb;
                    dsu.unite(borderBase[ti][afi] + static_cast<size_t>(ca),
                        borderBase[tj][bIt->second] + static_cast<size_t>(cb));
                }
            }
        }
    }
    dsu.compress_all();

    for (size_t ti = 0; ti < perTile.size(); ++ti) {
        const SoftMaskTileResult& tile = perTile[ti];
        for (size_t fi = 0; fi < tile.factors.size(); ++fi) {
            const auto& factorRes = tile.factors[fi];
            if (factorRes.borderCompAreas.empty()) continue;
            const size_t base = borderBase[ti][fi];
            for (size_t cid = 0; cid < factorRes.borderCompAreas.size(); ++cid) {
                const size_t gid = base + cid;
                const size_t root = dsu.parent[gid];
                out.rootArea[root] += static_cast<uint64_t>(factorRes.borderCompAreas[cid]);
                if (out.rootFactor[root] < 0) out.rootFactor[root] = factorRes.factor;
                else if (out.rootFactor[root] != factorRes.factor) {
                    error("%s: Label mismatch after seam union", __func__);
                }
                out.rootMembers[root].push_back(BorderComponentRef{ti, fi, cid});
            }
        }
    }

    return out;
}

void TileOperator::profileSoftFactorMasks(const std::string& outPrefix,
    int32_t focalK, int32_t radius, double neighborhoodThresholdFrac,
    double minFactorFrac, float minPixelProb, const std::vector<int32_t>& morphologySteps,
    uint32_t minComponentArea, bool skipMaskOverlap) {
    requireNoFeatureIndex(__func__);
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (coord_dim_ != 2 || (mode_ & 0x8) != 0) {
        error("%s: Only 2D regular tiles are supported", __func__);
    }
    if (isTextInput() && !canSeekTextInput()) {
        error("%s: This operation requires seekable input", __func__);
    }
    if (K_ <= 0) {
        error("%s: K must be known and positive", __func__);
    }
    if (focalK < 0 || focalK >= K_) {
        error("%s: focalK=%d out of range [0, %d)", __func__, focalK, K_);
    }
    if (radius < 0) {
        error("%s: radius must be >= 0", __func__);
    }
    if (neighborhoodThresholdFrac < 0.0 || neighborhoodThresholdFrac > 1.0) {
        error("%s: neighborhoodThresholdFrac must be in [0,1]", __func__);
    }
    if (minFactorFrac < 0.0 || minFactorFrac >= 1.0) {
        error("%s: minFactorFrac must be in [0,1)", __func__);
    }
    if (minPixelProb < 0.0) {
        error("%s: minPixelProb must be >= 0", __func__);
    }
    for (int32_t step : morphologySteps) {
        if (step == 0 || (std::abs(step) % 2) == 0) {
            error("%s: morphology steps must be non-zero odd integers", __func__);
        }
    }

    const double neighborhoodThreshold = neighborhoodThresholdFrac *
        static_cast<double>((2 * radius + 1) * (2 * radius + 1));

    struct Stage1Accum {
        std::vector<double> histMask;
        std::vector<double> histGlobal;
        uint64_t focalArea = 0;
        uint64_t outOfRange = 0;
        uint64_t badFactor = 0;
        uint64_t collisions = 0;

        explicit Stage1Accum(int32_t K)
            : histMask(static_cast<size_t>(K), 0.0),
              histGlobal(static_cast<size_t>(K), 0.0) {}
    };

    const auto t0 = std::chrono::steady_clock::now();
    Stage1Accum stage1Global(K_);

    auto openTileStream = [&]() {
        std::ifstream in;
        if (mode_ & 0x1) in.open(dataFile_, std::ios::binary);
        else in.open(dataFile_);
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        return in;
    };

    auto processStage1Tile = [&](size_t bi, std::ifstream& in, Stage1Accum& accum) {
        const TileInfo& blk = blocks_[bi];
        SoftMaskTileData tileData;
        loadSoftMaskTileData(blk, in, minPixelProb, true, true, true, tileData,
            accum.outOfRange, accum.badFactor, &accum.collisions, &accum.histGlobal);
        const size_t nPix = tileData.geom.W * tileData.geom.H;
        std::vector<float> denseFocal;
        std::vector<uint8_t> focalMask;
        auto focalIt = tileData.factorEntries.find(focalK);
        if (focalIt != tileData.factorEntries.end()) {
            buildDenseFactorRaster(focalIt->second, nPix, denseFocal);
        } else {
            denseFocal.assign(nPix, 0.0f);
        }
        buildSoftMaskBinary(denseFocal, tileData.seenLocal, tileData.geom.W, tileData.geom.H,
            radius, neighborhoodThreshold,
            SoftMaskThresholdConfig{false, false, 0.0, 0.0, morphologySteps},
            focalMask);
        const uint64_t keptArea = filterMaskByMinComponentArea4(
            focalMask, tileData.geom.W, tileData.geom.H, minComponentArea);
        accum.focalArea += keptArea;
        for (const auto& stored : tileData.records) {
            if (!focalMask[stored.localIdx]) continue;
            const TopProbs& top = stored.rec;
            for (size_t i = 0; i < top.ks.size() && i < top.ps.size(); ++i) {
                const int32_t k = top.ks[i];
                if (k < 0 || k >= K_) continue;
                accum.histMask[static_cast<size_t>(k)] += static_cast<double>(top.ps[i]);
            }
        }
    };

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<Stage1Accum> tls([&] { return Stage1Accum(K_); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in = openTileStream();
                auto& local = tls.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processStage1Tile(bi, in, local);
                }
            });
        tls.combine_each([&](const Stage1Accum& local) {
            for (size_t k = 0; k < stage1Global.histMask.size(); ++k) {
                stage1Global.histMask[k] += local.histMask[k];
                stage1Global.histGlobal[k] += local.histGlobal[k];
            }
            stage1Global.focalArea += local.focalArea;
            stage1Global.outOfRange += local.outOfRange;
            stage1Global.badFactor += local.badFactor;
            stage1Global.collisions += local.collisions;
        });
    } else {
        std::ifstream in = openTileStream();
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            processStage1Tile(bi, in, stage1Global);
        }
    }
    const auto tStage1 = std::chrono::steady_clock::now();

    double histMaskTotal = std::accumulate(stage1Global.histMask.begin(), stage1Global.histMask.end(), 0.0);
    double histGlobalTotal = std::accumulate(stage1Global.histGlobal.begin(), stage1Global.histGlobal.end(), 0.0);
    std::string outHist = outPrefix + ".factor_hist.tsv";
    FILE* fpHist = fopen(outHist.c_str(), "w");
    if (!fpHist) {
        error("%s: Cannot open output file %s", __func__, outHist.c_str());
    }
    fprintf(fpHist, "k\tmass_in_mask\tfrac_in_mask\tmass_global\tfrac_global\n");
    for (int32_t k = 0; k < K_; ++k) {
        const double fracMask = (histMaskTotal > 0.0) ? stage1Global.histMask[static_cast<size_t>(k)] / histMaskTotal : 0.0;
        const double fracGlobal = (histGlobalTotal > 0.0) ? stage1Global.histGlobal[static_cast<size_t>(k)] / histGlobalTotal : 0.0;
        fprintf(fpHist, "%d\t%.3e\t%.3e\t%.3e\t%.3e\n",
            k,
            stage1Global.histMask[static_cast<size_t>(k)],
            fracMask,
            stage1Global.histGlobal[static_cast<size_t>(k)],
            fracGlobal);
    }
    fclose(fpHist);

    if (skipMaskOverlap) {
        return;
    }

    std::vector<int32_t> selectedFactors;
    selectedFactors.reserve(static_cast<size_t>(K_));
    selectedFactors.push_back(focalK);
    if (histMaskTotal > 0.0) {
        for (int32_t k = 0; k < K_; ++k) {
            if (k == focalK) continue;
            if (stage1Global.histMask[static_cast<size_t>(k)] / histMaskTotal > minFactorFrac) {
                selectedFactors.push_back(k);
            }
        }
    }
    const size_t nSelected = selectedFactors.size();
    if (nSelected < 2) {
        notice("%s: No factors passed the minFactorFrac threshold", __func__);
        return;
    }

    std::string outPairwise = outPrefix + ".pairwise.tsv";
    FILE* fpPairwise = fopen(outPairwise.c_str(), "w");
    if (!fpPairwise) {
        error("%s: Cannot open output file %s", __func__, outPairwise.c_str());
    }
    fprintf(fpPairwise, "k1\tk2\tarea1_pix\tarea2_pix\tarea_ovlp_pix\tarea_ovlp_f1\tarea_ovlp_f2\tarea_jaccard\tmass1_in_ovlp\tmass2_in_ovlp\tmass_ovlp_f1\tmass_ovlp_f2\n");

    if (histMaskTotal <= 0.0 || nSelected < 2) {
        fclose(fpPairwise);
        const auto tDone = std::chrono::steady_clock::now();
        notice("%s: Wrote histogram to %s", __func__, outHist.c_str());
        notice("%s: Wrote empty pairwise summary to %s", __func__, outPairwise.c_str());
        notice("%s: timing(ms): stage1=%lld total=%lld n_selected=%zu focal_area=%llu",
            __func__,
            static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage1 - t0).count()),
            static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tDone - t0).count()),
            nSelected,
            static_cast<unsigned long long>(stage1Global.focalArea));
        if (stage1Global.outOfRange > 0) {
            warning("%s: Ignored %llu out-of-tile records", __func__, static_cast<unsigned long long>(stage1Global.outOfRange));
        }
        if (stage1Global.badFactor > 0) {
            warning("%s: Ignored %llu invalid factor entries", __func__, static_cast<unsigned long long>(stage1Global.badFactor));
        }
        if (stage1Global.collisions > 0) {
            warning("%s: Encountered %llu duplicate pixel records", __func__, static_cast<unsigned long long>(stage1Global.collisions));
        }
        return;
    }

    const size_t nPairs = nSelected * (nSelected - 1) / 2;
    std::vector<int32_t> factorToSelected(static_cast<size_t>(K_), -1);
    for (size_t s = 0; s < nSelected; ++s) {
        factorToSelected[static_cast<size_t>(selectedFactors[s])] = static_cast<int32_t>(s);
    }

    struct Stage3Accum {
        std::vector<uint64_t> maskArea;
        std::vector<uint64_t> areaIntersection;
        std::vector<double> mass1Intersection;
        std::vector<double> mass2Intersection;

        Stage3Accum(size_t nSel, size_t nPair)
            : maskArea(nSel, 0),
              areaIntersection(nPair, 0),
              mass1Intersection(nPair, 0.0),
              mass2Intersection(nPair, 0.0) {}
    };

    Stage3Accum stage3Global(nSelected, nPairs);

    auto processStage3Tile = [&](size_t bi, std::ifstream& in, Stage3Accum& accum) {
        const TileInfo& blk = blocks_[bi];
        SoftMaskTileData tileData;
        uint64_t ignoredOutOfRange = 0;
        uint64_t ignoredBadFactor = 0;
        loadSoftMaskTileData(blk, in, minPixelProb, true, true, true, tileData,
            ignoredOutOfRange, ignoredBadFactor, nullptr, nullptr);
        const size_t nPix = tileData.geom.W * tileData.geom.H;
        if (nPix == 0) return;

        std::vector<std::vector<std::pair<uint32_t, float>>> entries(nSelected);
        for (size_t s = 0; s < nSelected; ++s) {
            auto it = tileData.factorEntries.find(selectedFactors[s]);
            if (it != tileData.factorEntries.end()) {
                entries[s] = it->second;
            }
        }

        std::vector<std::vector<uint8_t>> builtMasks(nSelected);
        std::vector<const std::vector<uint8_t>*> maskPtrs(nSelected, nullptr);

        std::vector<float> dense;
        for (size_t s = 0; s < nSelected; ++s) {
            buildDenseFactorRaster(entries[s], nPix, dense);
            buildSoftMaskBinary(dense, tileData.seenLocal, tileData.geom.W, tileData.geom.H,
                radius, neighborhoodThreshold,
                SoftMaskThresholdConfig{false, false, 0.0, 0.0, morphologySteps},
                builtMasks[s]);
            const uint64_t keptArea = filterMaskByMinComponentArea4(
                builtMasks[s], tileData.geom.W, tileData.geom.H, minComponentArea);
            accum.maskArea[s] += keptArea;
            maskPtrs[s] = &builtMasks[s];
        }

        for (size_t idx = 0; idx < nPix; ++idx) {
            for (size_t i = 0; i < nSelected; ++i) {
                if (!(*maskPtrs[i])[idx]) continue;
                for (size_t j = i + 1; j < nSelected; ++j) {
                    if ((*maskPtrs[j])[idx]) {
                        accum.areaIntersection[pair_index_upper(i, j, nSelected)]++;
                    }
                }
            }
        }

        std::vector<double> selectedProb(nSelected, 0.0);
        for (const auto& stored : tileData.records) {
            const uint32_t localIdx = stored.localIdx;
            std::fill(selectedProb.begin(), selectedProb.end(), 0.0);
            const TopProbs& top = stored.rec;
            for (size_t i = 0; i < top.ks.size() && i < top.ps.size(); ++i) {
                const int32_t k = top.ks[i];
                if (k < 0 || k >= K_) continue;
                const double p = static_cast<double>(top.ps[i]);
                const int32_t selIdx = factorToSelected[static_cast<size_t>(k)];
                if (selIdx >= 0) {
                    selectedProb[static_cast<size_t>(selIdx)] += p;
                }
            }
            for (size_t i = 0; i < nSelected; ++i) {
                if (!(*maskPtrs[i])[localIdx]) continue;
                for (size_t j = i + 1; j < nSelected; ++j) {
                    if (!(*maskPtrs[j])[localIdx]) continue;
                    const size_t pidx = pair_index_upper(i, j, nSelected);
                    accum.mass1Intersection[pidx] += selectedProb[i];
                    accum.mass2Intersection[pidx] += selectedProb[j];
                }
            }
        }
    };

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<Stage3Accum> tls([&] { return Stage3Accum(nSelected, nPairs); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in = openTileStream();
                auto& local = tls.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processStage3Tile(bi, in, local);
                }
            });
        tls.combine_each([&](const Stage3Accum& local) {
            for (size_t i = 0; i < nSelected; ++i) {
                stage3Global.maskArea[i] += local.maskArea[i];
            }
            for (size_t i = 0; i < nPairs; ++i) {
                stage3Global.areaIntersection[i] += local.areaIntersection[i];
                stage3Global.mass1Intersection[i] += local.mass1Intersection[i];
                stage3Global.mass2Intersection[i] += local.mass2Intersection[i];
            }
        });
    } else {
        std::ifstream in = openTileStream();
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            processStage3Tile(bi, in, stage3Global);
        }
    }
    const auto tStage3 = std::chrono::steady_clock::now();

    auto safe_ratio = [](double num, double den) {
        return (den > 0.0) ? (num / den) : 0.0;
    };
    for (size_t i = 0; i < nSelected; ++i) {
        for (size_t j = i + 1; j < nSelected; ++j) {
            const size_t pidx = pair_index_upper(i, j, nSelected);
            const double area1 = static_cast<double>(stage3Global.maskArea[i]);
            const double area2 = static_cast<double>(stage3Global.maskArea[j]);
            const double areaI = static_cast<double>(stage3Global.areaIntersection[pidx]);
            const double areaJ = safe_ratio(areaI, area1 + area2 - areaI);
            const double mass1I = stage3Global.mass1Intersection[pidx];
            const double mass2I = stage3Global.mass2Intersection[pidx];
            const double totalMass1 = stage1Global.histGlobal[static_cast<size_t>(selectedFactors[i])];
            const double totalMass2 = stage1Global.histGlobal[static_cast<size_t>(selectedFactors[j])];
            fprintf(fpPairwise, "%d\t%d\t%llu\t%llu\t%llu\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\t%.3e\n",
                selectedFactors[i], selectedFactors[j],
                static_cast<unsigned long long>(stage3Global.maskArea[i]),
                static_cast<unsigned long long>(stage3Global.maskArea[j]),
                static_cast<unsigned long long>(stage3Global.areaIntersection[pidx]),
                safe_ratio(areaI, area1), safe_ratio(areaI, area2), areaJ,
                mass1I, mass2I,
                safe_ratio(mass1I, totalMass1),
                safe_ratio(mass2I, totalMass2));
        }
    }
    fclose(fpPairwise);

    const auto tDone = std::chrono::steady_clock::now();
    notice("%s: Wrote histogram to %s", __func__, outHist.c_str());
    notice("%s: Wrote pairwise mask summary to %s", __func__, outPairwise.c_str());
    notice("%s: timing(ms): stage1=%lld stage3=%lld total=%lld n_selected=%zu focal_area=%llu",
        __func__,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage1 - t0).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage3 - tStage1).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tDone - t0).count()),
        nSelected,
        static_cast<unsigned long long>(stage3Global.maskArea[0]));
    if (stage1Global.outOfRange > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(stage1Global.outOfRange));
    }
    if (stage1Global.badFactor > 0) {
        warning("%s: Ignored %llu invalid factor entries",
            __func__, static_cast<unsigned long long>(stage1Global.badFactor));
    }
    if (stage1Global.collisions > 0) {
        warning("%s: Encountered %llu duplicate pixel records",
            __func__, static_cast<unsigned long long>(stage1Global.collisions));
    }
}

void TileOperator::softFactorMask(const std::string& outPrefix,
    int32_t radius, double neighborhoodThreshold,
    float minPixelProb, const std::vector<int32_t>& morphologySteps,
    double minTileFactorMass, uint32_t minComponentArea, uint32_t minHoleArea,
    double simplifyTolerance, bool skipBoundaries,
    const std::string& templateGeoJSON, const std::string& templateOutPrefix) {
    requireNoFeatureIndex(__func__);
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (coord_dim_ != 2 || (mode_ & 0x8) != 0) {
        error("%s: Only 2D regular tiles are supported", __func__);
    }
    if (isTextInput() && !canSeekTextInput()) {
        error("%s: This operation requires seekable input", __func__);
    }
    if (K_ <= 0) {
        error("%s: K must be known and positive", __func__);
    }
    if (radius < 0) {
        error("%s: radius must be >= 0", __func__);
    }
    if (neighborhoodThreshold < 0.0 || neighborhoodThreshold > 1.0) {
        error("%s: neighborhoodThreshold must be in [0,1]", __func__);
    }
    if (minPixelProb < 0.0f) {
        error("%s: minPixelProb must be >= 0", __func__);
    }
    if (minTileFactorMass < 0.0) {
        error("%s: minTileFactorMass must be >= 0", __func__);
    }
    if (simplifyTolerance < 0.0) {
        error("%s: simplifyTolerance must be >= 0", __func__);
    }
    for (int32_t step : morphologySteps) {
        if (step == 0 || (std::abs(step) % 2) == 0) {
            error("%s: morphology steps must be non-zero odd integers", __func__);
        }
    }

    struct StageAccum {
        std::vector<uint64_t> tileCount;
        uint64_t outOfRange = 0;
        uint64_t badFactor = 0;
        uint64_t collisions = 0;

        explicit StageAccum(int32_t K)
            : tileCount(static_cast<size_t>(K), 0) {}
    };

    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<SoftMaskTileResult> perTile(blocks_.size());
    StageAccum globalAccum(K_);

    auto openTileStream = [&]() {
        std::ifstream in;
        if (mode_ & 0x1) in.open(dataFile_, std::ios::binary);
        else in.open(dataFile_);
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        return in;
    };

    auto processTile = [&](size_t bi, std::ifstream& in, StageAccum& accum) {
        const TileInfo& blk = blocks_[bi];
        SoftMaskTileResult tileOut;
        initTileGeom(blk, tileOut.geom);
        const size_t nPix = tileOut.geom.W * tileOut.geom.H;
        if (nPix == 0) {
            perTile[bi] = std::move(tileOut);
            return;
        }

        SoftMaskTileData tileData;
        loadSoftMaskTileData(blk, in, minPixelProb, false, true, true, tileData,
            accum.outOfRange, accum.badFactor, &accum.collisions, nullptr);

        std::vector<int32_t> factorsToProcess;
        factorsToProcess.reserve(tileData.factorMass.size());
        for (const auto& kv : tileData.factorMass) {
            if (kv.second >= minTileFactorMass) {
                factorsToProcess.push_back(kv.first);
                accum.tileCount[static_cast<size_t>(kv.first)] += 1;
            }
        }
        std::sort(factorsToProcess.begin(), factorsToProcess.end());
        if (factorsToProcess.empty()) {
            perTile[bi] = std::move(tileOut);
            return;
        }

        std::vector<float> dense;
        std::vector<uint8_t> mask;
        const SoftMaskThresholdConfig maskConfig{true, true, 0.0, 1.0, morphologySteps};

        for (int32_t factor : factorsToProcess) {
            auto it = tileData.factorEntries.find(factor);
            if (it == tileData.factorEntries.end()) {
                continue;
            }
            buildDenseFactorRaster(it->second, nPix, dense);
            buildSoftMaskBinary(dense, tileData.seenLocal, tileOut.geom.W, tileOut.geom.H, radius, neighborhoodThreshold, maskConfig, mask);

            BinaryMaskCCL ccl = buildBinaryMaskCCL4(mask, dense, tileOut.geom.W, tileOut.geom.H, static_cast<double>(minComponentArea));
            if (ccl.ncomp == 0 || ccl.keptArea == 0) continue;

            SoftMaskTileFactorResult factorOut;
            factorOut.factor = factor;
            factorOut.maskArea = ccl.keptArea;
            std::vector<uint32_t> borderRemap(ccl.ncomp, INVALID);
            std::vector<uint32_t> borderCidOrder;
            if (!skipBoundaries) {
                factorOut.borderPolys.resize(ccl.ncomp);
            }
            for (uint32_t cid = 0; cid < ccl.ncomp; ++cid) {
                if (ccl.compTouchesBorder[cid]) {
                    borderRemap[cid] = static_cast<uint32_t>(factorOut.borderCompAreas.size());
                    borderCidOrder.push_back(cid);
                    factorOut.borderCompAreas.push_back(ccl.compArea[cid]);
                } else {
                    factorOut.interiorCompAreas.push_back(ccl.compArea[cid]);
                }
                if (!skipBoundaries) {
                    Clipper2Lib::Paths64 polys;
                    if (ccl.compTouchesBorder[cid]) {
                        polys = normalizeMaskPolygons(buildMaskComponentRuns(ccl, cid, tileOut.geom));
                        factorOut.borderPolys[static_cast<size_t>(cid)] = std::move(polys);
                    } else { // Internal: finalize (normalize + simplify)
                        polys = buildMaskComponentPolygons(ccl, cid, tileOut.geom, minHoleArea, simplifyTolerance);
                        if (!polys.empty()) {
                            factorOut.interiorPolys.push_back(std::move(polys));
                        }
                    }
                }
            }
            if (factorOut.interiorCompAreas.empty() && factorOut.borderCompAreas.empty()) continue;

            factorOut.leftCid.assign(ccl.leftCid.size(), INVALID);
            factorOut.rightCid.assign(ccl.rightCid.size(), INVALID);
            factorOut.topCid.assign(ccl.topCid.size(), INVALID);
            factorOut.bottomCid.assign(ccl.bottomCid.size(), INVALID);
            for (size_t i = 0; i < ccl.leftCid.size(); ++i) {
                const uint32_t cid = ccl.leftCid[i];
                if (cid != INVALID && cid < borderRemap.size()) factorOut.leftCid[i] = borderRemap[cid];
            }
            for (size_t i = 0; i < ccl.rightCid.size(); ++i) {
                const uint32_t cid = ccl.rightCid[i];
                if (cid != INVALID && cid < borderRemap.size()) factorOut.rightCid[i] = borderRemap[cid];
            }
            for (size_t i = 0; i < ccl.topCid.size(); ++i) {
                const uint32_t cid = ccl.topCid[i];
                if (cid != INVALID && cid < borderRemap.size()) factorOut.topCid[i] = borderRemap[cid];
            }
            for (size_t i = 0; i < ccl.bottomCid.size(); ++i) {
                const uint32_t cid = ccl.bottomCid[i];
                if (cid != INVALID && cid < borderRemap.size()) factorOut.bottomCid[i] = borderRemap[cid];
            }

            if (!skipBoundaries && !factorOut.borderCompAreas.empty()) {
                std::vector<Clipper2Lib::Paths64> compactBorderPolys(factorOut.borderCompAreas.size());
                for (size_t i = 0; i < borderCidOrder.size(); ++i) {
                    compactBorderPolys[i] = std::move(factorOut.borderPolys[borderCidOrder[i]]);
                }
                factorOut.borderPolys.swap(compactBorderPolys);
            } else {
                factorOut.borderPolys.clear();
            }

            tileOut.factorToIndex[factor] = tileOut.factors.size();
            tileOut.factors.push_back(std::move(factorOut));
        }

        perTile[bi] = std::move(tileOut);
    };

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<StageAccum> tls([&] { return StageAccum(K_); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in = openTileStream();
                auto& local = tls.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processTile(bi, in, local);
                }
            });
        tls.combine_each([&](const StageAccum& local) {
            for (size_t k = 0; k < globalAccum.tileCount.size(); ++k) {
                globalAccum.tileCount[k] += local.tileCount[k];
            }
            globalAccum.outOfRange += local.outOfRange;
            globalAccum.badFactor += local.badFactor;
            globalAccum.collisions += local.collisions;
        });
    } else {
        std::ifstream in = openTileStream();
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            processTile(bi, in, globalAccum);
        }
    }
    const auto tStage1 = std::chrono::steady_clock::now();

    std::vector<Clipper2Lib::Paths64> finalPathsByFactor(static_cast<size_t>(K_));
    std::vector<uint64_t> finalArea(static_cast<size_t>(K_), 0);
    std::vector<uint64_t> finalComponents(static_cast<size_t>(K_), 0);
    std::vector<std::unordered_map<uint64_t, uint64_t>> componentHist(static_cast<size_t>(K_));

    for (size_t ti = 0; ti < perTile.size(); ++ti) {
        const SoftMaskTileResult& tile = perTile[ti];
        for (size_t fi = 0; fi < tile.factors.size(); ++fi) {
            const SoftMaskTileFactorResult& factorRes = tile.factors[fi];
            finalArea[static_cast<size_t>(factorRes.factor)] += factorRes.maskArea;
            for (uint32_t area : factorRes.interiorCompAreas) {
                finalComponents[static_cast<size_t>(factorRes.factor)] += 1;
                componentHist[static_cast<size_t>(factorRes.factor)][static_cast<uint64_t>(area)] += 1;
            }
            if (!skipBoundaries) {
                for (const auto& polys : factorRes.interiorPolys) {
                    finalPathsByFactor[static_cast<size_t>(factorRes.factor)].insert(
                        finalPathsByFactor[static_cast<size_t>(factorRes.factor)].end(),
                        polys.begin(), polys.end());
                }
            }
        }
    }

    const BorderMergeResult borderMerge = mergeSoftMaskTileBorders(perTile, INVALID);
    for (size_t root = 0; root < borderMerge.rootFactor.size(); ++root) {
        const int32_t factor = borderMerge.rootFactor[root];
        if (factor < 0 || borderMerge.rootArea[root] == 0) continue;
        finalComponents[static_cast<size_t>(factor)] += 1;
        componentHist[static_cast<size_t>(factor)][borderMerge.rootArea[root]] += 1;
        if (skipBoundaries || borderMerge.rootMembers[root].empty()) continue;
        Clipper2Lib::Paths64 rootPaths;
        for (const BorderComponentRef& ref : borderMerge.rootMembers[root]) {
            const auto& factorRes = perTile[ref.tileIdx].factors[ref.factorIdx];
            if (ref.localCid >= factorRes.borderPolys.size()) {
                continue;
            }
            rootPaths.insert(rootPaths.end(),
                factorRes.borderPolys[ref.localCid].begin(),
                factorRes.borderPolys[ref.localCid].end());
        }
        Clipper2Lib::Paths64 merged = cleanupMaskPolygons(rootPaths, minHoleArea, simplifyTolerance);
        if (merged.empty()) continue;
        finalPathsByFactor[static_cast<size_t>(factor)].insert(
            finalPathsByFactor[static_cast<size_t>(factor)].end(),
            merged.begin(), merged.end());
    }
    const auto tStage2 = std::chrono::steady_clock::now();

    std::string outHist = outPrefix + ".factor_summary.tsv";
    write_factor_mask_info(outHist, globalAccum.tileCount, finalArea, finalComponents);
    std::string outCompHist = outPrefix + ".component_hist.tsv";
    write_factor_mask_component_histogram(outCompHist, componentHist);

    std::string outGeoJSON = outPrefix + ".geojson";
    if (!skipBoundaries) {
        TemplateGeoJSON templateGeoJSONData;
        const bool writeTemplateJSON = !templateGeoJSON.empty();
        const std::string factorOutPrefix = templateOutPrefix.empty() ? outPrefix : templateOutPrefix;
        if (writeTemplateJSON) {
            templateGeoJSONData = load_template_json(templateGeoJSON);
        }
        nlohmann::json features = nlohmann::json::array();
        for (int32_t k = 0; k < K_; ++k) {
            if (finalPathsByFactor[static_cast<size_t>(k)].empty()) continue;
            Clipper2Lib::Paths64 factorPaths = normalizeMaskPolygons(
                finalPathsByFactor[static_cast<size_t>(k)]);
            if (factorPaths.empty()) continue;
            nlohmann::json feature;
            feature["type"] = "Feature";
            feature["properties"] = {
                {"Factor", k},
                {"n_tiles", globalAccum.tileCount[static_cast<size_t>(k)]},
                {"mask_area_pix", finalArea[static_cast<size_t>(k)]},
                {"n_components", finalComponents[static_cast<size_t>(k)]}
            };
            feature["geometry"] = maskPathsToMultiPolygonGeoJSON(factorPaths);
            if (writeTemplateJSON) {
                const std::string outFactor = factorOutPrefix + ".k" + std::to_string(k) + ".geojson";
                write_factor_template_geojson(outFactor, templateGeoJSONData, k, feature);
            }
            features.push_back(std::move(feature));
        }
        nlohmann::json featureCollection;
        featureCollection["type"] = "FeatureCollection";
        featureCollection["features"] = std::move(features);

        std::ofstream geojsonOut(outGeoJSON);
        if (!geojsonOut.is_open()) {
            error("%s: Cannot open output file %s", __func__, outGeoJSON.c_str());
        }
        geojsonOut << featureCollection.dump();
        geojsonOut << "\n";
        geojsonOut.close();
    }

    const auto tDone = std::chrono::steady_clock::now();
    notice("%s: Wrote factor histogram to %s", __func__, outHist.c_str());
    notice("%s: Wrote component size histogram to %s", __func__, outCompHist.c_str());
    if (!skipBoundaries) {
        notice("%s: Wrote soft-mask GeoJSON to %s", __func__, outGeoJSON.c_str());
    }
    notice("%s: timing(ms): stage1=%lld stage2=%lld total=%lld",
        __func__,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage1 - t0).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage2 - tStage1).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tDone - t0).count()));
    if (globalAccum.outOfRange > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(globalAccum.outOfRange));
    }
    if (globalAccum.badFactor > 0) {
        warning("%s: Ignored %llu invalid factor entries",
            __func__, static_cast<unsigned long long>(globalAccum.badFactor));
    }
    if (globalAccum.collisions > 0) {
        warning("%s: Encountered %llu duplicate pixel records",
            __func__, static_cast<unsigned long long>(globalAccum.collisions));
    }
}

void TileOperator::softMaskComposition(const std::string& outPrefix,
    const std::string& maskGeoJSON, const std::vector<int32_t>& focalFactors) {
    requireNoFeatureIndex(__func__);
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (coord_dim_ != 2 || (mode_ & 0x8) != 0) {
        error("%s: Only 2D regular tiles are supported", __func__);
    }
    if (isTextInput() && !canSeekTextInput()) {
        error("%s: This operation requires seekable input", __func__);
    }
    if (K_ <= 0) {
        error("%s: K must be known and positive", __func__);
    }
    if (maskGeoJSON.empty()) {
        error("%s: maskGeoJSON must not be empty", __func__);
    }

    std::vector<uint8_t> selectedFocal(static_cast<size_t>(K_), focalFactors.empty() ? 1u : 0u);
    if (!focalFactors.empty()) {
        for (int32_t focalK : focalFactors) {
            if (focalK < 0 || focalK >= K_) {
                error("%s: focal factor %d out of range [0, %d)", __func__, focalK, K_);
            }
            if (selectedFocal[static_cast<size_t>(focalK)]) {
                warning("%s: Ignoring duplicated focal factor id %d", __func__, focalK);
                continue;
            }
            selectedFocal[static_cast<size_t>(focalK)] = 1;
        }
    }

    std::vector<uint8_t> foundFocal(static_cast<size_t>(K_), 0);
    std::vector<FocalMaskRegion> masks = load_focal_mask_regions(
        maskGeoJSON, formatInfo_.tileSize, selectedFocal, &foundFocal);
    if (masks.empty()) {
        warning("%s: No focal masks selected from %s; output will contain only the global histogram",
            __func__, maskGeoJSON.c_str());
    }
    if (!focalFactors.empty()) {
        for (int32_t focalK = 0; focalK < K_; ++focalK) {
            if (selectedFocal[static_cast<size_t>(focalK)] && !foundFocal[static_cast<size_t>(focalK)]) {
                warning("%s: Requested focal factor %d was not found in %s",
                    __func__, focalK, maskGeoJSON.c_str());
            }
        }
    }

    std::unordered_map<TileKey, std::vector<size_t>, TileKeyHash> masksByTile;
    for (size_t mi = 0; mi < masks.size(); ++mi) {
        for (const auto& kv : masks[mi].region.tile_bins) {
            masksByTile[kv.first].push_back(mi);
        }
    }

    struct TileMaskCandidate {
        size_t maskIdx = 0;
        RegionTileState state = RegionTileState::Outside;
    };

    std::vector<std::vector<std::pair<TileKey, RegionTileState>>> statesByMask(masks.size());
    if (!masks.empty()) {
        if (threads_ > 1 && masks.size() > 1) {
            tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
            tbb::parallel_for(tbb::blocked_range<size_t>(0, masks.size()),
                [&](const tbb::blocked_range<size_t>& range) {
                    for (size_t mi = range.begin(); mi < range.end(); ++mi) {
                        auto& out = statesByMask[mi];
                        out.reserve(masks[mi].region.tile_bins.size());
                        for (const auto& kv : masks[mi].region.tile_bins) {
                            out.push_back({kv.first, masks[mi].region.classifyTile(kv.first)});
                        }
                    }
                });
        } else {
            for (size_t mi = 0; mi < masks.size(); ++mi) {
                auto& out = statesByMask[mi];
                out.reserve(masks[mi].region.tile_bins.size());
                for (const auto& kv : masks[mi].region.tile_bins) {
                    out.push_back({kv.first, masks[mi].region.classifyTile(kv.first)});
                }
            }
        }
    }

    std::unordered_map<TileKey, std::vector<TileMaskCandidate>, TileKeyHash> tileCandidates;
    tileCandidates.reserve(masksByTile.size());
    for (const auto& kv : masksByTile) {
        tileCandidates.emplace(kv.first, std::vector<TileMaskCandidate>{});
    }
    for (size_t mi = 0; mi < statesByMask.size(); ++mi) {
        for (const auto& entry : statesByMask[mi]) {
            if (entry.second == RegionTileState::Outside) {
                continue;
            }
            auto it = tileCandidates.find(entry.first);
            if (it == tileCandidates.end()) {
                continue;
            }
            it->second.push_back(TileMaskCandidate{mi, entry.second});
        }
    }

    struct StageAccum {
        std::vector<double> globalHist;
        std::vector<double> maskHist;
        std::vector<double> maskTotal;
        uint64_t outOfRange = 0;
        uint64_t badFactor = 0;
        uint64_t collisions = 0;

        StageAccum(int32_t K, size_t nMasks)
            : globalHist(static_cast<size_t>(K), 0.0),
              maskHist(static_cast<size_t>(K) * nMasks, 0.0),
              maskTotal(nMasks, 0.0) {}
    };

    const double pixelResolution = static_cast<double>(
        getRasterPixelResolution() > 0.0f ? getRasterPixelResolution() : 1.0f);
    const auto t0 = std::chrono::steady_clock::now();
    StageAccum globalAccum(K_, masks.size());

    auto openTileStream = [&]() {
        std::ifstream in;
        if (mode_ & 0x1) in.open(dataFile_, std::ios::binary);
        else in.open(dataFile_);
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        return in;
    };

    auto accumulateRecord = [&](StageAccum& accum, size_t maskIdx, const TopProbs& rec) {
        const size_t base = maskIdx * static_cast<size_t>(K_);
        for (size_t i = 0; i < rec.ks.size() && i < rec.ps.size(); ++i) {
            const int32_t k = rec.ks[i];
            if (k < 0 || k >= K_) {
                continue;
            }
            const double p = static_cast<double>(rec.ps[i]);
            accum.maskHist[base + static_cast<size_t>(k)] += p;
            accum.maskTotal[maskIdx] += p;
        }
    };

    auto processTile = [&](size_t bi, std::ifstream& in, StageAccum& accum) {
        const TileInfo& blk = blocks_[bi];
        const TileKey tileKey{blk.idx.row, blk.idx.col};
        const auto candIt = tileCandidates.find(tileKey);
        if (candIt == tileCandidates.end() || candIt->second.empty()) {
            SoftMaskTileData tileData;
            loadSoftMaskTileData(blk, in, 0.0f, false, false, false, tileData,
                accum.outOfRange, accum.badFactor, &accum.collisions, &accum.globalHist);
            return;
        }

        SoftMaskTileData tileData;
        loadSoftMaskTileData(blk, in, 0.0f, true, false, false, tileData,
            accum.outOfRange, accum.badFactor, &accum.collisions, &accum.globalHist);
        if (tileData.records.empty()) {
            return;
        }

        std::vector<size_t> insideMasks;
        std::vector<size_t> partialMasks;
        insideMasks.reserve(candIt->second.size());
        partialMasks.reserve(candIt->second.size());
        for (const TileMaskCandidate& candidate : candIt->second) {
            switch (candidate.state) {
                case RegionTileState::Inside:
                    insideMasks.push_back(candidate.maskIdx);
                    break;
                case RegionTileState::Partial:
                    partialMasks.push_back(candidate.maskIdx);
                    break;
                case RegionTileState::Outside:
                    break;
            }
        }
        if (insideMasks.empty() && partialMasks.empty()) {
            return;
        }

        for (const SoftMaskSparseRec& stored : tileData.records) {
            for (size_t maskIdx : insideMasks) {
                accumulateRecord(accum, maskIdx, stored.rec);
            }

            if (partialMasks.empty()) {
                continue;
            }

            const size_t localIdx = static_cast<size_t>(stored.localIdx);
            const size_t localY = localIdx / tileData.geom.W;
            const size_t localX = localIdx - localY * tileData.geom.W;
            const double x = (static_cast<double>(tileData.geom.pixX0) +
                static_cast<double>(localX) + 0.5) * pixelResolution;
            const double y = (static_cast<double>(tileData.geom.pixY0) +
                static_cast<double>(localY) + 0.5) * pixelResolution;

            for (size_t maskIdx : partialMasks) {
                if (masks[maskIdx].region.containsPoint(
                        static_cast<float>(x), static_cast<float>(y), &tileData.geom.key)) {
                    accumulateRecord(accum, maskIdx, stored.rec);
                }
            }
        }
    };

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<StageAccum> tls([&] { return StageAccum(K_, masks.size()); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in = openTileStream();
                auto& local = tls.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processTile(bi, in, local);
                }
            });
        tls.combine_each([&](const StageAccum& local) {
            for (size_t i = 0; i < globalAccum.globalHist.size(); ++i) {
                globalAccum.globalHist[i] += local.globalHist[i];
            }
            for (size_t i = 0; i < globalAccum.maskHist.size(); ++i) {
                globalAccum.maskHist[i] += local.maskHist[i];
            }
            for (size_t i = 0; i < globalAccum.maskTotal.size(); ++i) {
                globalAccum.maskTotal[i] += local.maskTotal[i];
            }
            globalAccum.outOfRange += local.outOfRange;
            globalAccum.badFactor += local.badFactor;
            globalAccum.collisions += local.collisions;
        });
    } else {
        std::ifstream in = openTileStream();
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            processTile(bi, in, globalAccum);
        }
    }

    const double globalTotal = std::accumulate(
        globalAccum.globalHist.begin(), globalAccum.globalHist.end(), 0.0);
    const std::string outFile = outPrefix + ".mask_composition.tsv";
    FILE* fp = fopen(outFile.c_str(), "w");
    if (!fp) {
        error("%s: Cannot open output file %s", __func__, outFile.c_str());
    }
    fprintf(fp, "k_focal\tk\tmass\tfrac\n");
    for (size_t mi = 0; mi < masks.size(); ++mi) {
        const size_t base = mi * static_cast<size_t>(K_);
        const double denom = globalAccum.maskTotal[mi];
        for (int32_t k = 0; k < K_; ++k) {
            const double mass = globalAccum.maskHist[base + static_cast<size_t>(k)];
            const double frac = (denom > 0.0) ? (mass / denom) : 0.0;
            fprintf(fp, "%d\t%d\t%.4e\t%.4e\n", masks[mi].focalK, k, mass, frac);
        }
    }
    for (int32_t k = 0; k < K_; ++k) {
        const double mass = globalAccum.globalHist[static_cast<size_t>(k)];
        const double frac = (globalTotal > 0.0) ? (mass / globalTotal) : 0.0;
        fprintf(fp, "%d\t%d\t%.4e\t%.4e\n", K_, k, mass, frac);
    }
    fclose(fp);

    const auto tDone = std::chrono::steady_clock::now();
    notice("%s: Wrote mask-composition histogram to %s", __func__, outFile.c_str());
    notice("%s: timing(ms): total=%lld",
        __func__,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tDone - t0).count()));
    if (globalAccum.outOfRange > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(globalAccum.outOfRange));
    }
    if (globalAccum.badFactor > 0) {
        warning("%s: Ignored %llu invalid factor entries",
            __func__, static_cast<unsigned long long>(globalAccum.badFactor));
    }
    if (globalAccum.collisions > 0) {
        warning("%s: Encountered %llu duplicate pixel records",
            __func__, static_cast<unsigned long long>(globalAccum.collisions));
    }
}

void TileOperator::hardFactorMask(const std::string& outPrefix, uint32_t minComponentSize, bool skipBoundaries, const std::string& templateGeoJSON, const std::string& templateOutPrefix) {
    requireNoFeatureIndex(__func__);
    if (blocks_.empty()) {
        warning("%s: No tiles to process", __func__);
        return;
    }
    if (!regular_labeled_raster_ || coord_dim_ != 2) {
        error("%s only supports raster 2D data with regular tiles", __func__);
    }
    if (K_ <= 0 || K_ > 255) {
        error("%s: K must be in [1, 255], got %d", __func__, K_);
    }
    const int32_t K = K_;
    const uint8_t BG = static_cast<uint8_t>(K);
    const uint32_t INVALID = std::numeric_limits<uint32_t>::max();
    const auto t0 = std::chrono::steady_clock::now();

    struct StageAccum {
        std::vector<uint64_t> tileCount;
        uint64_t outOfRange = 0;
        uint64_t badLabel = 0;

        explicit StageAccum(int32_t K_) : tileCount(static_cast<size_t>(K_), 0) {}
    };

    std::vector<TileCCL> perTile(blocks_.size());
    StageAccum globalAccum(K_);

    auto openTileStream = [&]() {
        std::ifstream in;
        if (mode_ & 0x1) in.open(dataFile_, std::ios::binary);
        else in.open(dataFile_);
        if (!in.is_open()) {
            error("%s: Error opening input data file: %s", __func__, dataFile_.c_str());
        }
        return in;
    };

    auto processTile = [&](size_t bi, std::ifstream& in, StageAccum& accum) {
        DenseTile dense;
        loadDenseTile(blocks_[bi], in, dense, BG, accum.outOfRange, accum.badLabel);
        perTile[bi] = tileLocalCCL(dense, BG);
        std::vector<uint8_t> seen(static_cast<size_t>(K), 0);
        for (uint8_t lbl : dense.lab) {
            if (lbl < BG) seen[static_cast<size_t>(lbl)] = 1;
        }
        for (size_t k = 0; k < seen.size(); ++k) {
            accum.tileCount[k] += static_cast<uint64_t>(seen[k]);
        }
    };

    if (threads_ > 1 && blocks_.size() > 1) {
        tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, static_cast<size_t>(threads_));
        tbb::combinable<StageAccum> tls([&] { return StageAccum(K_); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, blocks_.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                std::ifstream in = openTileStream();
                auto& local = tls.local();
                for (size_t bi = range.begin(); bi < range.end(); ++bi) {
                    processTile(bi, in, local);
                }
            });
        tls.combine_each([&](const StageAccum& local) {
            for (size_t k = 0; k < globalAccum.tileCount.size(); ++k) {
                globalAccum.tileCount[k] += local.tileCount[k];
            }
            globalAccum.outOfRange += local.outOfRange;
            globalAccum.badLabel += local.badLabel;
        });
    } else {
        std::ifstream in = openTileStream();
        for (size_t bi = 0; bi < blocks_.size(); ++bi) {
            processTile(bi, in, globalAccum);
        }
    }
    const auto tStage1 = std::chrono::steady_clock::now();

    std::vector<Clipper2Lib::Paths64> finalPathsByFactor(static_cast<size_t>(K), Clipper2Lib::Paths64{});
    std::vector<uint64_t> finalArea(static_cast<size_t>(K), 0);
    std::vector<uint64_t> finalComponents(static_cast<size_t>(K), 0);
    std::vector<std::unordered_map<uint64_t, uint64_t>> componentHist(static_cast<size_t>(K));
    std::vector<SoftMaskTileResult> borderTiles(blocks_.size());

    for (size_t ti = 0; ti < perTile.size(); ++ti) {
        TileCCL& t = perTile[ti];
        if (t.ncomp == 0) continue;
        TileGeom geom;
        initTileGeom(blocks_[ti], geom);

        const BorderRemapInfo remapInfo = remapTileToBorderComponents(t, INVALID);
        SoftMaskTileResult tileOut;
        tileOut.geom = geom;
        std::vector<uint32_t> newCidToOld(t.ncomp, INVALID);
        for (uint32_t oldCid = 0; oldCid < remapInfo.remap.size(); ++oldCid) {
            const uint8_t lbl = remapInfo.oldCompLabel[oldCid];
            const uint64_t size = static_cast<uint64_t>(remapInfo.oldCompSize[oldCid]);
            if (lbl >= BG || size == 0) continue;
            const uint32_t newCid = remapInfo.remap[oldCid];
            if (newCid == INVALID) {
                if (size < static_cast<uint64_t>(minComponentSize)) continue;
                finalArea[static_cast<size_t>(lbl)] += size;
                finalComponents[static_cast<size_t>(lbl)] += 1;
                componentHist[static_cast<size_t>(lbl)][size] += 1;
                if (!skipBoundaries && oldCid < remapInfo.oldCompBox.size()) {
                    Clipper2Lib::Paths64 oldPolys = buildLabelComponentPolygons(
                        t.pixelCid, geom.W, oldCid, remapInfo.oldCompBox[oldCid], geom, 0, 0.0);
                    if (!oldPolys.empty()) {
                    finalPathsByFactor[static_cast<size_t>(lbl)].insert(
                        finalPathsByFactor[static_cast<size_t>(lbl)].end(),
                            oldPolys.begin(), oldPolys.end());
                    }
                }
            } else if (newCid < newCidToOld.size()) {
                newCidToOld[newCid] = oldCid;
            }
        }

        std::vector<uint32_t> newCidToFactorLocal(t.ncomp, INVALID);
        std::vector<int32_t> newCidToFactorIndex(t.ncomp, -1);
        for (uint32_t cid = 0; cid < t.ncomp; ++cid) {
            const uint8_t lbl = t.compLabel[cid];
            if (lbl >= BG) continue;
            auto it = tileOut.factorToIndex.find(static_cast<int32_t>(lbl));
            size_t fi = 0;
            if (it == tileOut.factorToIndex.end()) {
                fi = tileOut.factors.size();
                tileOut.factorToIndex[static_cast<int32_t>(lbl)] = fi;
                tileOut.factors.push_back(SoftMaskTileFactorResult{});
                tileOut.factors.back().factor = static_cast<int32_t>(lbl);
                tileOut.factors.back().leftCid.assign(geom.H, INVALID);
                tileOut.factors.back().rightCid.assign(geom.H, INVALID);
                tileOut.factors.back().topCid.assign(geom.W, INVALID);
                tileOut.factors.back().bottomCid.assign(geom.W, INVALID);
            } else {
                fi = it->second;
            }
            SoftMaskTileFactorResult& factorRes = tileOut.factors[fi];
            newCidToFactorIndex[cid] = static_cast<int32_t>(fi);
            newCidToFactorLocal[cid] = static_cast<uint32_t>(factorRes.borderCompAreas.size());
            factorRes.borderCompAreas.push_back(t.compSize[cid]);
            factorRes.maskArea += static_cast<uint64_t>(t.compSize[cid]);
            if (!skipBoundaries && cid < newCidToOld.size()) {
                const uint32_t oldCid = newCidToOld[cid];
                factorRes.borderSourceCid.push_back(oldCid);
                if (oldCid != INVALID && oldCid < remapInfo.oldCompBox.size()) {
                    factorRes.borderSourceBox.push_back(remapInfo.oldCompBox[oldCid]);
                } else {
                    factorRes.borderSourceBox.push_back(PixBox{});
                }
            }
        }

        for (size_t y = 0; y < t.leftCid.size(); ++y) {
            const uint32_t cid = t.leftCid[y];
            if (cid != INVALID && cid < newCidToFactorIndex.size() && newCidToFactorIndex[cid] >= 0) {
                tileOut.factors[static_cast<size_t>(newCidToFactorIndex[cid])].leftCid[y] = newCidToFactorLocal[cid];
            }
        }
        for (size_t y = 0; y < t.rightCid.size(); ++y) {
            const uint32_t cid = t.rightCid[y];
            if (cid != INVALID && cid < newCidToFactorIndex.size() && newCidToFactorIndex[cid] >= 0) {
                tileOut.factors[static_cast<size_t>(newCidToFactorIndex[cid])].rightCid[y] = newCidToFactorLocal[cid];
            }
        }
        for (size_t x = 0; x < t.topCid.size(); ++x) {
            const uint32_t cid = t.topCid[x];
            if (cid != INVALID && cid < newCidToFactorIndex.size() && newCidToFactorIndex[cid] >= 0) {
                tileOut.factors[static_cast<size_t>(newCidToFactorIndex[cid])].topCid[x] = newCidToFactorLocal[cid];
            }
        }
        for (size_t x = 0; x < t.bottomCid.size(); ++x) {
            const uint32_t cid = t.bottomCid[x];
            if (cid != INVALID && cid < newCidToFactorIndex.size() && newCidToFactorIndex[cid] >= 0) {
                tileOut.factors[static_cast<size_t>(newCidToFactorIndex[cid])].bottomCid[x] = newCidToFactorLocal[cid];
            }
        }

        borderTiles[ti] = std::move(tileOut);
    }

    const BorderMergeResult borderMerge = mergeSoftMaskTileBorders(borderTiles, INVALID);
    for (size_t root = 0; root < borderMerge.rootFactor.size(); ++root) {
        const int32_t factor = borderMerge.rootFactor[root];
        if (factor < 0 || borderMerge.rootArea[root] < static_cast<uint64_t>(minComponentSize)) continue;
        finalArea[static_cast<size_t>(factor)] += borderMerge.rootArea[root];
        finalComponents[static_cast<size_t>(factor)] += 1;
        componentHist[static_cast<size_t>(factor)][borderMerge.rootArea[root]] += 1;
        if (skipBoundaries || borderMerge.rootMembers[root].empty()) continue;
        Clipper2Lib::Paths64 rootPaths;
        for (const BorderComponentRef& ref : borderMerge.rootMembers[root]) {
            const auto& factorRes = borderTiles[ref.tileIdx].factors[ref.factorIdx];
            if (ref.localCid >= factorRes.borderSourceCid.size() ||
                ref.localCid >= factorRes.borderSourceBox.size()) {
                continue;
            }
            const uint32_t oldCid = factorRes.borderSourceCid[ref.localCid];
            if (oldCid == INVALID) {
                continue;
            }
            const TileGeom& geom = borderTiles[ref.tileIdx].geom;
            Clipper2Lib::Paths64 polys = buildLabelComponentPolygons(
                perTile[ref.tileIdx].pixelCid, geom.W, oldCid,
                factorRes.borderSourceBox[ref.localCid], geom, 0, 0.0);
            rootPaths.insert(rootPaths.end(), polys.begin(), polys.end());
        }
        Clipper2Lib::Paths64 merged = cleanupMaskPolygons(rootPaths, 0, 0.0);
        if (merged.empty()) continue;
        finalPathsByFactor[static_cast<size_t>(factor)].insert(
            finalPathsByFactor[static_cast<size_t>(factor)].end(),
            merged.begin(), merged.end());
    }
    const auto tStage2 = std::chrono::steady_clock::now();

    std::string outHist = outPrefix + ".factor_summary.tsv";
    write_factor_mask_info(outHist, globalAccum.tileCount, finalArea, finalComponents);
    std::string outCompHist = outPrefix + ".component_hist.tsv";
    write_factor_mask_component_histogram(outCompHist, componentHist);

    std::string outGeoJSON = outPrefix + ".geojson";
    if (!skipBoundaries) {
        TemplateGeoJSON templateGeoJSONData;
        const bool writeTemplateJSON = !templateGeoJSON.empty();
        const std::string factorOutPrefix = templateOutPrefix.empty() ? outPrefix : templateOutPrefix;
        if (writeTemplateJSON) {
            templateGeoJSONData = load_template_json(templateGeoJSON);
        }
        nlohmann::json features = nlohmann::json::array();
        for (int32_t k = 0; k < K; ++k) {
            if (finalPathsByFactor[static_cast<size_t>(k)].empty()) continue;
            Clipper2Lib::Paths64 factorPaths = cleanupMaskPolygons(finalPathsByFactor[static_cast<size_t>(k)], 0, 0.0);
            if (factorPaths.empty()) continue;
            nlohmann::json feature;
            feature["type"] = "Feature";
            feature["properties"] = {
                {"Factor", k},
                {"n_tiles", globalAccum.tileCount[static_cast<size_t>(k)]},
                {"mask_area_pix", finalArea[static_cast<size_t>(k)]},
                {"n_components", finalComponents[static_cast<size_t>(k)]}
            };
            feature["geometry"] = maskPathsToMultiPolygonGeoJSON(factorPaths);
            if (writeTemplateJSON) {
                const std::string outFactor = factorOutPrefix + ".k" + std::to_string(k) + ".geojson";
                write_factor_template_geojson(outFactor, templateGeoJSONData, k, feature);
            }
            features.push_back(std::move(feature));
        }
        nlohmann::json featureCollection;
        featureCollection["type"] = "FeatureCollection";
        featureCollection["features"] = std::move(features);
        std::ofstream geojsonOut(outGeoJSON);
        if (!geojsonOut.is_open()) {
            error("%s: Cannot open output file %s", __func__, outGeoJSON.c_str());
        }
        geojsonOut << featureCollection.dump();
        geojsonOut << "\n";
        geojsonOut.close();
    }

    const auto tDone = std::chrono::steady_clock::now();
    notice("%s: Wrote factor histogram to %s", __func__, outHist.c_str());
    notice("%s: Wrote component size histogram to %s", __func__, outCompHist.c_str());
    if (!skipBoundaries) {
        notice("%s: Wrote hard-mask GeoJSON to %s", __func__, outGeoJSON.c_str());
    }
    notice("%s: timing(ms): stage1=%lld stage2=%lld total=%lld",
        __func__,
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage1 - t0).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tStage2 - tStage1).count()),
        static_cast<long long>(std::chrono::duration_cast<std::chrono::milliseconds>(tDone - t0).count()));
    if (globalAccum.outOfRange > 0) {
        warning("%s: Ignored %llu out-of-tile records",
            __func__, static_cast<unsigned long long>(globalAccum.outOfRange));
    }
    if (globalAccum.badLabel > 0) {
        warning("%s: Ignored %llu records with invalid labels",
            __func__, static_cast<unsigned long long>(globalAccum.badLabel));
    }
}
