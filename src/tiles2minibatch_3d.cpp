#include "tiles2minibatch.hpp"
#include "bccgrid.hpp"
#include "numerical_utils.hpp"
#include <cmath>
#include <numeric>
#include <fcntl.h>
#include <thread>
#include <algorithm>

namespace {

int32_t floor_mod(int32_t value, int32_t mod) {
    int32_t r = value % mod;
    if (r < 0) {
        r += mod;
    }
    return r;
}

template <typename Fn>
void for_each_axial_ring(int32_t u0, int32_t v0, int32_t radius, Fn&& fn) {
    if (radius < 0) {
        return;
    }
    if (radius == 0) {
        fn(u0, v0);
        return;
    }
    static constexpr int32_t directions[6][2] = {
        {1, 0}, {1, -1}, {0, -1}, {-1, 0}, {-1, 1}, {0, 1}
    };
    int32_t u = u0 + directions[4][0] * radius;
    int32_t v = v0 + directions[4][1] * radius;
    for (int32_t dir = 0; dir < 6; ++dir) {
        for (int32_t step = 0; step < radius; ++step) {
            fn(u, v);
            u += directions[dir][0];
            v += directions[dir][1];
        }
    }
}

float fine_axial_distance_sq(float uf, float vf, int32_t u, int32_t v, const HexGrid& fineGrid) {
    const float du = static_cast<float>(u) - uf;
    const float dv = static_cast<float>(v) - vf;
    const float dx = static_cast<float>(fineGrid.mtx_a2p[0][0]) * du
        + static_cast<float>(fineGrid.mtx_a2p[0][1]) * dv;
    const float dy = static_cast<float>(fineGrid.mtx_a2p[1][1]) * dv;
    return dx * dx + dy * dy;
}

} // namespace

template<typename T>
void Tiles2MinibatchBase<T>::set3Dparameters(bool isThin, double zMin, double zMax, float zRes, bool enforceZrange, float standard3DBccGridDist, const std::vector<float>& thin3DZLevels) {
    const bool hasZMin = std::isfinite(zMin);
    const bool hasZMax = std::isfinite(zMax);
    if (hasZMin != hasZMax) {
        error("3D z range must provide both zmin and zmax together");
    }
    if (isThin && !hasZMin) {
        error("Thin 3D anchor range requires both zmin and zmax");
    }
    if (hasZMin && zMax <= zMin) {
        error("3D anchor range must satisfy zmax > zmin");
    }
    coordDim_ = MinibatchCoordDim::Dim3;
    useThin3DAnchors_ = isThin;
    zMin_ = zMin;
    zMax_ = zMax;
    pixelResolutionZ_ = zRes > 0 ? zRes : pixelResolution_;
    ignoreOutsideZrange_ = enforceZrange;
    standard3DBccGridDist_ = standard3DBccGridDist;
    if (useThin3DAnchors_) {
        if (thin3DZLevels.size() <= 1) {
            error("Thin 3D anchor mode requires at least two z levels");
        }
        thin3DZLevels_ = thin3DZLevels;
        std::sort(thin3DZLevels_.begin(), thin3DZLevels_.end());
        for (size_t i = 1; i < thin3DZLevels_.size(); ++i) {
            if (!(thin3DZLevels_[i] > thin3DZLevels_[i - 1])) {
                error("z levels must be unique");
            }
        }
        for (const auto& z : thin3DZLevels_) {
            if (!std::isfinite(z) || (hasZMin && (z < zMin_ || z > zMax_))) {
                error("Thin 3D z levels must be finite and within the specified z range");
            }
        }
        nZLevels_ = static_cast<int32_t>(thin3DZLevels_.size());
    } else {
        thin3DZLevels_.clear();
        if (standard3DBccGridDist_ <= 0) {
            error("Standard 3D mode requires a positive BCC lattice distance");
        }
        standard3DBccSize_ = standard3DBccGridDist_ * 2.0 / std::sqrt(3.0);
    }
    configureInputMode();
    configureOutputMode();
    nativeBinaryRegularTiles_ = nativeRegularTiles_ && outputBinary_;
}

template<typename T>
void Tiles2MinibatchBase<T>::forEachAnchorCandidate3D(const TileData<T>& tileData, const std::function<void(uint32_t, float, const AnchorKey3D&)>& emit) const {
    BCCGrid bccGrid(standard3DBccSize_);
    forEachAnchorCandidate3D(tileData, bccGrid, emit);
}

template<typename T>
void Tiles2MinibatchBase<T>::forEachAnchorCandidate3D(const TileData<T>& tileData, const BCCGrid& bccGrid, const std::function<void(uint32_t, float, const AnchorKey3D&)>& emit) const {
    auto neighborOffsets = bccGrid.face_adjacent_offsets();
    auto assign_pt = [&](float x, float y, float z, uint32_t idx, float ct) {
        if (ignoreOutsideZrange_ && (z < zMin_ || z > zMax_)) {
            return;
        }
        int32_t q1, q2, q3;
        bccGrid.cart_to_lattice(q1, q2, q3, x, y, z);
        emit(idx, ct, AnchorKey3D{q1, q2, q3, 0, 0, 0});
        for (const auto& delta : neighborOffsets) {
            emit(idx, ct,
                AnchorKey3D{q1 + delta[0], q2 + delta[1], q3 + delta[2], 0, 0, 0});
        }
    };
    if (useExtended_) {
        for (const auto& pt : tileData.extended3D().extPts3d) {
            assign_pt(static_cast<float>(pt.recBase.x), static_cast<float>(pt.recBase.y),
                static_cast<float>(pt.recBase.z), pt.recBase.idx, static_cast<float>(pt.recBase.ct));
        }
    } else if (isSingleMoleculeMode()) {
        const auto& smInput = tileData.singleMolecule3D();
        for (size_t i = 0; i < smInput.coords3dFloat.size(); ++i) {
            const auto& coord = smInput.coords3dFloat[i];
            assign_pt(coord.x, coord.y, coord.z, smInput.featureIdx[i], smInput.obsWeight[i]);
        }
    } else {
        for (const auto& pt : tileData.standard3D().pts3d) {
            assign_pt(static_cast<float>(pt.x), static_cast<float>(pt.y),
                static_cast<float>(pt.z), pt.idx, static_cast<float>(pt.ct));
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::anchorKeyToCoord3D(float& x, float& y, float& z, const AnchorKey3D& key) const {
    BCCGrid bccGrid(standard3DBccSize_);
    anchorKeyToCoord3D(x, y, z, key, bccGrid);
}

template<typename T>
void Tiles2MinibatchBase<T>::anchorKeyToCoord3D(float& x, float& y, float& z, const AnchorKey3D& key, const BCCGrid& bccGrid) const {
    const int32_t q1 = std::get<0>(key);
    const int32_t q2 = std::get<1>(key);
    const int32_t q3 = std::get<2>(key);
    double xd, yd, zd;
    bccGrid.lattice_to_cart(xd, yd, zd, q1, q2, q3);
    x = static_cast<float>(xd);
    y = static_cast<float>(yd);
    z = static_cast<float>(zd);
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::buildAnchors3D(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, double minCount) {
    BCCGrid bccGrid(standard3DBccSize_);
    return buildAnchors3D(tileData, anchors, documents, bccGrid, minCount);
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::buildAnchors3D(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, const BCCGrid& bccGrid, double minCount) {
    if (coordDim_ != MinibatchCoordDim::Dim3) {
        return -1;
    }
    anchors.clear();
    documents.clear();
    std::map<AnchorKey3D, std::unordered_map<uint32_t, float>> bccAggregation;
    forEachAnchorCandidate3D(tileData, bccGrid, [&](uint32_t idx, float ct, const AnchorKey3D& key) {
        bccAggregation[key][idx] += ct;
    });

    for (auto& entry : bccAggregation) {
        float sum = std::accumulate(entry.second.begin(), entry.second.end(), 0.0f, [](float acc, const auto& p) { return acc + p.second; });
        if (sum < minCount) {
            continue;
        }
        SparseObs obs;
        Document& doc = obs.doc;
        obs.ct_tot = sum;
        for (auto& featurePair : entry.second) {
            doc.ids.push_back(featurePair.first);
            doc.cnts.push_back(featurePair.second);
        }
        if (lineParserPtr->weighted) {
            for (size_t i = 0; i < doc.ids.size(); ++i) {
                doc.cnts[i] *= lineParserPtr->weights[doc.ids[i]];
            }
        }
        documents.push_back(std::move(obs));
        const auto& key = entry.first;
        float x, y, z;
        anchorKeyToCoord3D(x, y, z, key, bccGrid);
        anchors.emplace_back(x, y, z);
    }
    return documents.size();
}

template<typename T>
void Tiles2MinibatchBase<T>::forEachThin3DAnchorWithinRadius(float x, float y, float z,
    const HexGrid& hexGrid_, int32_t nMoves_, float supportRadius,
    const std::function<void(const AnchorKey2D&, float, float, float, float)>& emit) const {
    if (nMoves_ <= 0) {
        error("%s: invalid thin 3D x-y refinement factor %d", __func__, nMoves_);
    }
    if (!(supportRadius > 0.0)) {
        error("%s: supportRadius must be positive", __func__);
    }
    HexGrid fineGrid(hexGrid_.size / static_cast<double>(nMoves_), hexGrid_.pointy);
    int32_t u0, v0;
    fineGrid.cart_to_axial(u0, v0, x, y);
    const float uf = static_cast<float>(fineGrid.mtx_p2a[0][0]) * x
        + static_cast<float>(fineGrid.mtx_p2a[0][1]) * y;
    const float vf = static_cast<float>(fineGrid.mtx_p2a[1][1]) * y;
    const float radiusSq = supportRadius * supportRadius;
    const float minStep = std::max(1e-6f,
        static_cast<float>(fineGrid.size) * std::sqrt(3.0f) * 0.5f);
    const int32_t hardMaxRadius = std::max<int32_t>(
        8, static_cast<int32_t>(std::ceil(supportRadius / minStep)) + 2);
    for (int32_t radius = 0; radius <= hardMaxRadius; ++radius) {
        for_each_axial_ring(u0, v0, radius, [&](int32_t u, int32_t v) {
            const AnchorKey2D key = thin3DFineAxialToAnchorKey(u, v, nMoves_);
            float ax, ay;
            fineGrid.axial_to_cart(ax, ay, u, v);
            const int32_t zIndex = thin3DAnchorZIndexForKey(key);
            const float az = thin3DZLevels_[static_cast<size_t>(zIndex)];
            const float dx = ax - x;
            const float dy = ay - y;
            const float dz = az - z;
            const float distSq = dx * dx + dy * dy + dz * dz;
            if (distSq <= radiusSq + 1e-8f) {
                emit(key, ax, ay, az, std::sqrt(distSq));
            }
        });
        float nextRingMinXY = std::numeric_limits<float>::infinity();
        for_each_axial_ring(u0, v0, radius + 1, [&](int32_t u, int32_t v) {
            nextRingMinXY = std::min(nextRingMinXY, fine_axial_distance_sq(uf, vf, u, v, fineGrid));
        });
        if (nextRingMinXY > radiusSq + 1e-8f) {
            return;
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::forEachAnchorCandidateThin3D(const TileData<T>& tileData, const HexGrid& hexGrid_, int32_t nMoves_,
    double supportRadius, double distNu, const std::function<void(uint32_t, float, const AnchorKey2D&)>& emit) const {
    auto assign_pt = [&](float x, float y, float z, uint32_t idx, float ct) {
        if (ignoreOutsideZrange_ && (z < zMin_ || z > zMax_)) {
            return;
        }
        const std::function<void(const AnchorKey2D&, float, float, float, float)> emitCandidate =
            [&](const AnchorKey2D& key, float, float, float, float dist) {
                const float weightedCt = ct *
                    anchor_distance_weight(dist, static_cast<float>(supportRadius), static_cast<float>(distNu));
                emit(idx, weightedCt, key);
            };
        forEachThin3DAnchorWithinRadius(x, y, z,
            hexGrid_, nMoves_, static_cast<float>(supportRadius), emitCandidate);
    };
    if (useExtended_) {
        for (const auto& pt : tileData.extended3D().extPts3d) {
            assign_pt(static_cast<float>(pt.recBase.x), static_cast<float>(pt.recBase.y),
                static_cast<float>(pt.recBase.z), pt.recBase.idx, static_cast<float>(pt.recBase.ct));
        }
    } else if (isSingleMoleculeMode()) {
        const auto& smInput = tileData.singleMolecule3D();
        for (size_t i = 0; i < smInput.coords3dFloat.size(); ++i) {
            const auto& coord = smInput.coords3dFloat[i];
            assign_pt(coord.x, coord.y, coord.z, smInput.featureIdx[i], smInput.obsWeight[i]);
        }
    } else {
        for (const auto& pt : tileData.standard3D().pts3d) {
            assign_pt(static_cast<float>(pt.x), static_cast<float>(pt.y),
                static_cast<float>(pt.z), pt.idx, static_cast<float>(pt.ct));
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::anchorKeyToCoordThin3D(float& x, float& y, float& z, const AnchorKey2D& key, const HexGrid& hexGrid_, int32_t nMoves_) const {
    anchorKeyToCoord2D(x, y, key, hexGrid_, nMoves_);
    const int32_t zIndex = thin3DAnchorZIndexForKey(key);
    z = thin3DZLevels_[static_cast<size_t>(zIndex)];
}

template<typename T>
void Tiles2MinibatchBase<T>::thin3DAnchorKeyToFineAxial(int32_t& u, int32_t& v, const AnchorKey2D& key, int32_t nMoves_) const {
    const int32_t hx = std::get<0>(key);
    const int32_t hy = std::get<1>(key);
    const int32_t ic = std::get<2>(key);
    const int32_t ir = std::get<3>(key);
    u = hx * nMoves_ - ic;
    v = hy * nMoves_ - ir;
}

template<typename T>
typename Tiles2MinibatchBase<T>::AnchorKey2D Tiles2MinibatchBase<T>::thin3DFineAxialToAnchorKey(int32_t u, int32_t v, int32_t nMoves_) const {
    const int32_t ic = floor_mod(-u, nMoves_);
    const int32_t ir = floor_mod(-v, nMoves_);
    const int32_t hx = (u + ic) / nMoves_;
    const int32_t hy = (v + ir) / nMoves_;
    return AnchorKey2D{hx, hy, ic, ir};
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::thin3DAnchorZIndexForKey(const AnchorKey2D& key) const {
    if (nZLevels_ <= 0) {
        error("%s: thin 3D z levels are not initialized", __func__);
    }
    uint64_t mixed = splitmix64(thin3DHashSeed_);
    auto combine = [&](int32_t value, uint64_t salt) {
        mixed = splitmix64(mixed ^ (static_cast<uint64_t>(static_cast<uint32_t>(value)) + salt));
    };
    combine(std::get<0>(key), 0x243f6a8885a308d3ULL);
    combine(std::get<1>(key), 0x13198a2e03707344ULL);
    combine(std::get<2>(key), 0xa4093822299f31d0ULL);
    combine(std::get<3>(key), 0x082efa98ec4e6c89ULL);
    return static_cast<int32_t>(mixed % static_cast<uint64_t>(nZLevels_));
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::buildAnchorsThin3D(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents,
    const HexGrid& hexGrid_, int32_t nMoves_, double minCount, double supportRadius, double distNu) {
    if (coordDim_ != MinibatchCoordDim::Dim3) {
        return -1;
    }
    if (nMoves_ <= 1) {
        error("%s: thin 3D anchors require nMoves > 1", __func__);
    }
    if (!(zMax_ > zMin_)) {
        error("%s: invalid thin 3D anchor z range", __func__);
    }
    if (!(supportRadius > 0.0)) {
        error("%s: thin 3D anchors require a positive support radius", __func__);
    }

    anchors.clear();
    documents.clear();
    std::map<AnchorKey2D, std::unordered_map<uint32_t, float>> hexAggregation;
    forEachAnchorCandidateThin3D(tileData, hexGrid_, nMoves_, supportRadius, distNu, [&](uint32_t idx, float ct, const AnchorKey2D& key) {
        hexAggregation[key][idx] += ct;
    });

    for (auto& entry : hexAggregation) {
        float sum = std::accumulate(entry.second.begin(), entry.second.end(), 0.0f,
            [](float acc, const auto& p) { return acc + p.second; });
        if (sum < minCount) {
            continue;
        }
        SparseObs obs;
        Document& doc = obs.doc;
        obs.ct_tot = sum;
        for (auto& featurePair : entry.second) {
            doc.ids.push_back(featurePair.first);
            doc.cnts.push_back(featurePair.second);
        }
        if (lineParserPtr->weighted) {
            for (size_t i = 0; i < doc.ids.size(); ++i) {
                doc.cnts[i] *= lineParserPtr->weights[doc.ids[i]];
            }
        }
        documents.push_back(std::move(obs));

        const auto& key = entry.first;
        float x, y, z;
        anchorKeyToCoordThin3D(x, y, z, key, hexGrid_, nMoves_);
        anchors.emplace_back(x, y, z);
    }
    return documents.size();
}

template<typename T>
double Tiles2MinibatchBase<T>::buildMinibatchCoreSingleFeaturePixel3D(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    const BCCGrid* bccGrid, double supportRadius, double distNu) {
    struct SingleMoleculeKey3D {
        int32_t x = 0;
        int32_t y = 0;
        int32_t z = 0;
        uint32_t feature = 0;
        bool operator==(const SingleMoleculeKey3D& other) const {
            return x == other.x && y == other.y && z == other.z && feature == other.feature;
        }
    };
    struct SingleMoleculeKey3DHash {
        size_t operator()(const SingleMoleculeKey3D& key) const {
            size_t h = std::hash<int32_t>()(key.x);
            h ^= std::hash<int32_t>()(key.y) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<int32_t>()(key.z) + 0x9e3779b9 + (h << 6) + (h >> 2);
            h ^= std::hash<uint32_t>()(key.feature) + 0x9e3779b9 + (h << 6) + (h >> 2);
            return h;
        }
    };
    struct SingleMoleculePixel3D {
        int32_t x = 0;
        int32_t y = 0;
        int32_t z = 0;
        uint32_t feature = 0;
        float weight = 0.0f;
        std::vector<uint32_t> originalIdx;
    };

    const float res = pixelResolution_;
    const float resZ = pixelResolutionZ_;

    minibatch.storageMode = Minibatch::StorageMode::SingleMolecule;
    minibatch.clearDataSgl();
    minibatch.clearDataMtx();

    std::unordered_map<SingleMoleculeKey3D, uint32_t, SingleMoleculeKey3DHash> pixelLookup;
    std::vector<SingleMoleculePixel3D> groupedPixels;
    uint32_t idxOriginal = 0;
    uint32_t nPixels = 0;
    const auto& stdInput = tileData.standard3D();
    tileData.orgpts2pixel.assign(stdInput.pts3d.size(), -1);
    for (const auto& pt : stdInput.pts3d) {
        if (ignoreOutsideZrange_ && (pt.z < zMin_ || pt.z > zMax_)) {
            continue;
        }
        const int32_t x = static_cast<int32_t>(std::floor(pt.x / res));
        const int32_t y = static_cast<int32_t>(std::floor(pt.y / res));
        const int32_t z = static_cast<int32_t>(std::floor(pt.z / resZ));
        const SingleMoleculeKey3D key{x, y, z, pt.idx};
        auto it = pixelLookup.find(key);
        if (it == pixelLookup.end()) {
            SingleMoleculePixel3D pix;
            pix.x = x;
            pix.y = y;
            pix.z = z;
            pix.feature = pt.idx;
            groupedPixels.push_back(std::move(pix));
            pixelLookup.emplace(key, nPixels++);
            it = pixelLookup.find(key);
        }
        SingleMoleculePixel3D& pix = groupedPixels[it->second];
        float weight = static_cast<float>(pt.ct);
        if (lineParserPtr->weighted) {
            weight *= static_cast<float>(lineParserPtr->weights[pt.idx]);
        }
        pix.weight += weight;
        pix.originalIdx.push_back(idxOriginal++);
    }

    tileData.coords3d.clear();
    tileData.rowFeatureIdx.clear();
    tileData.coords3d.reserve(nPixels);
    tileData.rowFeatureIdx.reserve(nPixels);
    minibatch.featureIdx.reserve(nPixels);
    minibatch.featureWeight.reserve(nPixels);
    minibatch.rowOffsets.reserve(nPixels + 1);
    minibatch.rowOffsets.push_back(0);

    std::vector<AnchorPoint> originalAnchors = anchors;
    std::map<AnchorKey3D, uint32_t> anchorKeyToIndex;
    std::vector<std::array<int32_t, 3>> bccNeighborOffsets;
    PointCloud<float> pc;
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;
    if (useThin3DAnchors_) {
        pc.pts = anchors;
    } else {
        const BCCGrid& grid = *bccGrid;
        bccNeighborOffsets = grid.face_adjacent_offsets();
        for (uint32_t i = 0; i < static_cast<uint32_t>(anchors.size()); ++i) {
            int32_t q1, q2, q3;
            grid.cart_to_lattice(q1, q2, q3, anchors[i].x, anchors[i].y, anchors[i].z);
            anchorKeyToIndex[AnchorKey3D{q1, q2, q3, 0, 0, 0}] = i;
        }
    }
    std::unique_ptr<kd_tree_f3_t> kdtree;
    const float l2radius = static_cast<float>(supportRadius * supportRadius);
    if (useThin3DAnchors_) {
        kdtree = std::make_unique<kd_tree_f3_t>(3, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    }

    uint32_t npt = 0;
    for (const auto& pix : groupedPixels) {
        std::vector<std::pair<uint32_t, float>> candidates;
        const float x = static_cast<float>(pix.x) * res;
        const float y = static_cast<float>(pix.y) * res;
        const float z = static_cast<float>(pix.z) * resZ;
        if (useThin3DAnchors_) {
            float xyz[3] = {x, y, z};
            const size_t n = kdtree->radiusSearch(xyz, l2radius, indices_dists);
            if (n == 0) {
                continue;
            }
            for (size_t i = 0; i < n; ++i) {
                const float dist = std::sqrt(indices_dists[i].second);
                candidates.emplace_back(indices_dists[i].first,
                    anchor_distance_weight(dist, supportRadius, distNu));
            }
        } else {
            const BCCGrid& grid = *bccGrid;
            int32_t q1, q2, q3;
            grid.cart_to_lattice(q1, q2, q3, x, y, z);
            auto emitCandidate = [&](int32_t cq1, int32_t cq2, int32_t cq3) {
                auto it = anchorKeyToIndex.find(AnchorKey3D{cq1, cq2, cq3, 0, 0, 0});
                if (it == anchorKeyToIndex.end()) {
                    return;
                }
                const AnchorPoint& anchor = originalAnchors[it->second];
                const float dx = x - anchor.x;
                const float dy = y - anchor.y;
                const float dz = z - anchor.z;
                const double dist = static_cast<double>(std::sqrt(dx * dx + dy * dy + dz * dz));
                candidates.emplace_back(it->second,
                    anchor_distance_weight(dist, supportRadius, distNu));
            };
            emitCandidate(q1, q2, q3);
            for (const auto& delta : bccNeighborOffsets) {
                emitCandidate(q1 + delta[0], q2 + delta[1], q3 + delta[2]);
            }
            if (candidates.empty()) {
                continue;
            }
        }

        float weightSum = 0.0f;
        const size_t edgeStart = minibatch.edgeAnchorIdx.size();
        for (const auto& candidate : candidates) {
            minibatch.edgeAnchorIdx.push_back(candidate.first);
            minibatch.wijVal.push_back(candidate.second);
            minibatch.psiVal.push_back(candidate.second);
            weightSum += candidate.second;
        }
        for (size_t e = edgeStart; e < minibatch.psiVal.size(); ++e) {
            minibatch.psiVal[e] /= weightSum;
        }

        tileData.coords3d.emplace_back(pix.x, pix.y, pix.z);
        tileData.rowFeatureIdx.push_back(pix.feature);
        minibatch.featureIdx.push_back(pix.feature);
        minibatch.featureWeight.push_back(pix.weight);
        for (auto v : pix.originalIdx) {
            tileData.orgpts2pixel[v] = static_cast<int32_t>(npt);
        }
        ++npt;
        minibatch.rowOffsets.push_back(static_cast<uint32_t>(minibatch.edgeAnchorIdx.size()));
    }

    anchors = std::move(originalAnchors);
    const double avgDegree = (npt > 0)
        ? static_cast<double>(minibatch.edgeAnchorIdx.size()) / static_cast<double>(npt)
        : 0.0;
    debug("%s: created %zu single-feature-pixel edges between %u rows and %zu anchors, average degree %.2f",
        __func__, minibatch.edgeAnchorIdx.size(), npt, anchors.size(), avgDegree);

    minibatch.N = static_cast<int32_t>(npt);
    minibatch.M = M_;
    return avgDegree;
}

template<typename T>
double Tiles2MinibatchBase<T>::buildMinibatchCoreSingleMolecule3D(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    const BCCGrid* bccGrid, double supportRadius, double distNu) {
    minibatch.storageMode = Minibatch::StorageMode::SingleMolecule;
    minibatch.clearDataSgl();
    minibatch.clearDataMtx();

    tileData.coords3d.clear();
    const auto& smInput = tileData.singleMolecule3D();
    auto& smInputMut = tileData.singleMolecule3D();
    const size_t nObs = smInput.coords3dFloat.size();
    minibatch.rowOffsets.reserve(nObs + 1);
    minibatch.rowOffsets.push_back(0);

    std::vector<AnchorPoint> originalAnchors = anchors;
    std::map<AnchorKey3D, uint32_t> anchorKeyToIndex;
    std::vector<std::array<int32_t, 3>> bccNeighborOffsets;
    PointCloud<float> pc;
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;
    if (useThin3DAnchors_) {
        pc.pts = anchors;
    } else {
        const BCCGrid& grid = *bccGrid;
        bccNeighborOffsets = grid.face_adjacent_offsets();
        for (uint32_t i = 0; i < static_cast<uint32_t>(anchors.size()); ++i) {
            int32_t q1, q2, q3;
            grid.cart_to_lattice(q1, q2, q3, anchors[i].x, anchors[i].y, anchors[i].z);
            anchorKeyToIndex[AnchorKey3D{q1, q2, q3, 0, 0, 0}] = i;
        }
    }
    std::unique_ptr<kd_tree_f3_t> kdtree;
    const float l2radius = static_cast<float>(supportRadius * supportRadius);
    if (useThin3DAnchors_) {
        kdtree = std::make_unique<kd_tree_f3_t>(3, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    }

    size_t npt = 0;
    tileData.orgpts2pixel.assign(nObs, -1);
    for (size_t i = 0; i < nObs; ++i) {
        const auto coord = smInput.coords3dFloat[i];
        const uint32_t feature = smInput.featureIdx[i];
        const float obsWeight = smInput.obsWeight[i];

        std::vector<std::pair<uint32_t, float>> candidates;
        const float x = coord.x;
        const float y = coord.y;
        const float z = coord.z;
        if (useThin3DAnchors_) {
            const float xyz[3] = {x, y, z};
            const size_t n = kdtree->radiusSearch(xyz, l2radius, indices_dists);
            if (n == 0) {
                continue;
            }
            for (size_t i = 0; i < n; ++i) {
                const float dist = std::sqrt(indices_dists[i].second);
                candidates.emplace_back(indices_dists[i].first,
                    anchor_distance_weight(dist, supportRadius, distNu));
            }
        } else {
            const BCCGrid& grid = *bccGrid;
            int32_t q1, q2, q3;
            grid.cart_to_lattice(q1, q2, q3, x, y, z);
            auto emitCandidate = [&](int32_t cq1, int32_t cq2, int32_t cq3) {
                auto it = anchorKeyToIndex.find(AnchorKey3D{cq1, cq2, cq3, 0, 0, 0});
                if (it == anchorKeyToIndex.end()) {
                    return;
                }
                const AnchorPoint& anchor = originalAnchors[it->second];
                const float dx = x - anchor.x;
                const float dy = y - anchor.y;
                const float dz = z - anchor.z;
                const double dist = static_cast<double>(std::sqrt(dx * dx + dy * dy + dz * dz));
                candidates.emplace_back(it->second,
                    anchor_distance_weight(dist, supportRadius, distNu));
            };
            emitCandidate(q1, q2, q3);
            for (const auto& delta : bccNeighborOffsets) {
                emitCandidate(q1 + delta[0], q2 + delta[1], q3 + delta[2]);
            }
            if (candidates.empty()) {
                continue;
            }
        }

        float weightSum = 0.0f;
        const size_t edgeStart = minibatch.edgeAnchorIdx.size();
        for (const auto& candidate : candidates) {
            minibatch.edgeAnchorIdx.push_back(candidate.first);
            minibatch.wijVal.push_back(candidate.second);
            minibatch.psiVal.push_back(candidate.second);
            weightSum += candidate.second;
        }
        for (size_t e = edgeStart; e < minibatch.psiVal.size(); ++e) {
            minibatch.psiVal[e] /= weightSum;
        }

        smInputMut.coords3dFloat[npt] = coord;
        smInputMut.featureIdx[npt] = feature;
        smInputMut.obsWeight[npt] = obsWeight;
        tileData.orgpts2pixel[i] = static_cast<int32_t>(npt);
        ++npt;
        minibatch.rowOffsets.push_back(static_cast<uint32_t>(minibatch.edgeAnchorIdx.size()));
    }
    smInputMut.coords3dFloat.resize(npt);
    smInputMut.featureIdx.resize(npt);
    smInputMut.obsWeight.resize(npt);
    minibatch.featureIdx = smInputMut.featureIdx;
    minibatch.featureWeight = std::move(smInputMut.obsWeight);

    anchors = std::move(originalAnchors);
    const double avgDegree = (npt > 0)
        ? static_cast<double>(minibatch.edgeAnchorIdx.size()) / static_cast<double>(npt)
        : 0.0;
    debug("%s: created %zu single-molecule edges between %zu rows and %zu anchors, average degree %.2f",
        __func__, minibatch.edgeAnchorIdx.size(), npt, anchors.size(), avgDegree);

    minibatch.N = static_cast<int32_t>(npt);
    minibatch.M = M_;
    return avgDegree;
}

template<typename T>
double Tiles2MinibatchBase<T>::buildMinibatchCore3D(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    const BCCGrid* bccGrid, double supportRadius, double distNu) {
    const size_t nObs = isSingleMoleculeMode()
        ? tileData.singleMolecule3D().coords3dFloat.size()
        : (useExtended_ ? tileData.extended3D().extPts3d.size() : tileData.standard3D().pts3d.size());
    debug("%s: building minibatch with %zu anchors and %zu documents", __func__, anchors.size(), nObs);
    if (minibatch.n <= 0) {
        return 0.0;
    }
    assert(supportRadius > 0.0 && distNu >= 0.0);
    if (!useThin3DAnchors_ && bccGrid == nullptr) {
        error("%s: standard 3D minibatch construction requires a BCCGrid", __func__);
    }

    const float res = pixelResolution_;
    const float resZ = pixelResolutionZ_;

    if (isSingleFeaturePixelMode()) {
        return buildMinibatchCoreSingleFeaturePixel3D(tileData, anchors, minibatch, bccGrid, supportRadius, distNu);
    }
    if (isSingleMoleculeMode()) {
        return buildMinibatchCoreSingleMolecule3D(tileData, anchors, minibatch, bccGrid, supportRadius, distNu);
    }

    minibatch.storageMode = Minibatch::StorageMode::GenericSparse;
    minibatch.clearDataSgl();

    std::vector<Eigen::Triplet<float>> tripletsMtx;
    std::vector<Eigen::Triplet<float>> tripletsWij;
    const size_t standardObs = useExtended_ ? tileData.extended3D().extPts3d.size() : tileData.standard3D().pts3d.size();
    tripletsMtx.reserve(standardObs);
    tripletsWij.reserve(standardObs);

    std::unordered_map<std::tuple<int32_t, int32_t, int32_t>,
        std::pair<std::unordered_map<uint32_t, float>, std::vector<uint32_t>>,
        Tuple3Hash> pixAgg;
    uint32_t idxOriginal = 0;

    if (useExtended_) {
        const auto& extInput = tileData.extended3D();
        tileData.orgpts2pixel.assign(extInput.extPts3d.size(), -1);
        for (const auto& pt : extInput.extPts3d) {
            if (ignoreOutsideZrange_ && (pt.recBase.z < zMin_ || pt.recBase.z > zMax_)) {
                continue;
            }
            int32_t x = static_cast<int32_t>(std::floor(pt.recBase.x / res));
            int32_t y = static_cast<int32_t>(std::floor(pt.recBase.y / res));
            int32_t z = static_cast<int32_t>(std::floor(pt.recBase.z / resZ));
            auto key = std::make_tuple(x, y, z);
            pixAgg[key].first[pt.recBase.idx] += pt.recBase.ct;
            pixAgg[key].second.push_back(idxOriginal++);
        }
    } else {
        const auto& stdInput2 = tileData.standard3D();
        tileData.orgpts2pixel.assign(stdInput2.pts3d.size(), -1);
        for (const auto& pt : stdInput2.pts3d) {
            if (ignoreOutsideZrange_ && (pt.z < zMin_ || pt.z > zMax_)) {
                continue;
            }
            int32_t x = static_cast<int32_t>(std::floor(pt.x / res));
            int32_t y = static_cast<int32_t>(std::floor(pt.y / res));
            int32_t z = static_cast<int32_t>(std::floor(pt.z / resZ));
            auto key = std::make_tuple(x, y, z);
            pixAgg[key].first[pt.idx] += pt.ct;
            pixAgg[key].second.push_back(idxOriginal++);
        }
    }
    tileData.coords3d.clear();
    tileData.coords3d.reserve(pixAgg.size());

    std::vector<AnchorPoint> originalAnchors = anchors;
    std::map<AnchorKey3D, uint32_t> anchorKeyToIndex;
    std::vector<std::array<int32_t, 3>> bccNeighborOffsets;
    PointCloud<float> pc;
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;
    if (useThin3DAnchors_) {
        pc.pts = anchors;
    } else {
        const BCCGrid& grid = *bccGrid;
        bccNeighborOffsets = grid.face_adjacent_offsets();
        for (uint32_t i = 0; i < static_cast<uint32_t>(anchors.size()); ++i) {
            int32_t q1, q2, q3;
            grid.cart_to_lattice(q1, q2, q3, anchors[i].x, anchors[i].y, anchors[i].z);
            anchorKeyToIndex[AnchorKey3D{q1, q2, q3, 0, 0, 0}] = i;
        }
    }
    std::unique_ptr<kd_tree_f3_t> kdtree;
    const float l2radius = static_cast<float>(supportRadius * supportRadius);
    if (useThin3DAnchors_) {
        kdtree = std::make_unique<kd_tree_f3_t>(3, pc, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    }

    uint32_t npt = 0;
    for (auto& kv : pixAgg) {
        int32_t px = std::get<0>(kv.first);
        int32_t py = std::get<1>(kv.first);
        int32_t pz = std::get<2>(kv.first);
        std::vector<std::pair<uint32_t, float>> candidates;

        const float x = static_cast<float>(px) * res;
        const float y = static_cast<float>(py) * res;
        const float z = static_cast<float>(pz) * resZ;
        if (useThin3DAnchors_) {
            float xyz[3] = {x, y, z};
            size_t n = kdtree->radiusSearch(xyz, l2radius, indices_dists);
            if (n == 0) {
                continue;
            }
            for (size_t i = 0; i < n; ++i) {
                uint32_t idx = indices_dists[i].first;
                const float dist = std::sqrt(indices_dists[i].second);
                candidates.emplace_back(
                    idx,
                    anchor_distance_weight(dist, supportRadius, distNu));
            }
        } else {
            const BCCGrid& grid = *bccGrid;
            int32_t q1, q2, q3;
            grid.cart_to_lattice(q1, q2, q3, x, y, z);
            auto emitCandidate = [&](int32_t cq1, int32_t cq2, int32_t cq3) {
                auto it = anchorKeyToIndex.find(AnchorKey3D{cq1, cq2, cq3, 0, 0, 0});
                if (it == anchorKeyToIndex.end()) {
                    return;
                }
                const AnchorPoint& anchor = originalAnchors[it->second];
                const float dx = x - anchor.x;
                const float dy = y - anchor.y;
                const float dz = z - anchor.z;
                const double dist = static_cast<double>(std::sqrt(dx * dx + dy * dy + dz * dz));
                candidates.emplace_back(
                    it->second,
                    anchor_distance_weight(dist, supportRadius, distNu));
            };
            emitCandidate(q1, q2, q3);
            for (const auto& delta : bccNeighborOffsets) {
                emitCandidate(q1 + delta[0], q2 + delta[1], q3 + delta[2]);
            }
            if (candidates.empty()) {
                continue;
            }
        }

        if (lineParserPtr->weighted) {
            for (auto& kv2 : kv.second.first) {
                kv2.second *= static_cast<float>(lineParserPtr->weights[kv2.first]);
                tripletsMtx.emplace_back(npt, static_cast<int>(kv2.first), kv2.second);
            }
        } else {
            for (auto& kv2 : kv.second.first) {
                tripletsMtx.emplace_back(npt, static_cast<int>(kv2.first), kv2.second);
            }
        }

        tileData.coords3d.emplace_back(px, py, pz);
        for (auto v : kv.second.second) {
            tileData.orgpts2pixel[v] = static_cast<int32_t>(npt);
        }
        for (const auto& candidate : candidates) {
            tripletsWij.emplace_back(
                npt, static_cast<int>(candidate.first), candidate.second);
        }

        ++npt;
    }
    anchors = std::move(originalAnchors);
    double avgDegree = (npt > 0)
        ? static_cast<double>(tripletsWij.size()) / static_cast<double>(npt)
        : 0.0;
    debug("%s: created %zu edges between %zu pixels and %zu anchors, average degree %.2f", __func__, tripletsWij.size(), npt, anchors.size(), avgDegree);

    minibatch.N = static_cast<int32_t>(npt);
    minibatch.M = M_;
    minibatch.mtx.resize(npt, M_);
    minibatch.mtx.setFromTriplets(tripletsMtx.begin(), tripletsMtx.end());
    minibatch.mtx.makeCompressed();

    minibatch.wij.resize(npt, minibatch.n);
    minibatch.wij.setFromTriplets(tripletsWij.begin(), tripletsWij.end());
    minibatch.wij.makeCompressed();

    minibatch.psi = minibatch.wij;
    rowNormalizeInPlace(minibatch.psi);

    return avgDegree;
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseOneTileStandard3D(TileData<T>& tileData, TileKey tile) {
    std::unique_ptr<BoundedReadline> iter;
    try {
        iter = tileReader.get_tile_iterator(tile.row, tile.col);
    } catch (const std::exception &e) {
        warning("%s", e.what());
        return -1;
    }
    tileData.clear();
    tile2bound(tile, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax, tileSize);
    auto& input = tileData.emplaceStandard3D();
    std::unordered_map<uint32_t, std::vector<RecordT3D<T>>> buffers3d;
    if (M_ == 0) {
        M_ = lineParserPtr->featureDict.size();
    }

    std::string line;
    int32_t npt = 0;
    while (iter->next(line)) {
        RecordT3D<T> rec;
        int32_t idx = lineParserPtr->parse<T>(rec, line);
        if (idx < -1) {
            error("Error parsing line: %s", line.c_str());
        }
        if (idx == -1 || idx >= M_) {
            continue;
        }
        if (ignoreOutsideZrange_ && (rec.z < zMin_ || rec.z > zMax_)) {
            continue;
        }
        input.pts3d.push_back(rec);
        std::vector<uint32_t> bufferidx;
        if (pt2buffer(bufferidx, rec.x, rec.y, tile) == 1) {
            tileData.idxinternal.push_back(npt);
        }
        for (const auto& key : bufferidx) {
            buffers3d[key].push_back(rec);
        }
        npt++;
    }
    for (const auto& entry : buffers3d) {
        auto buffer = getBoundaryBuffer(entry.first);
        buffer->addRecords3D(entry.second);
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseOneTileSingleMolecule3D(TileData<T>& tileData, TileKey tile) {
    std::unique_ptr<BoundedReadline> iter;
    try {
        iter = tileReader.get_tile_iterator(tile.row, tile.col);
    } catch (const std::exception &e) {
        warning("%s", e.what());
        return -1;
    }
    tileData.clear();
    tile2bound(tile, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax, tileSize);
    auto& input = tileData.emplaceSingleMolecule3D();
    std::unordered_map<uint32_t, std::vector<RecordT3D<T>>> buffers3d;
    if (M_ == 0) {
        M_ = lineParserPtr->featureDict.size();
    }

    std::string line;
    int32_t npt = 0;
    while (iter->next(line)) {
        RecordT3D<T> rec;
        int32_t idx = lineParserPtr->parse<T>(rec, line);
        if (idx < -1) {
            error("Error parsing line: %s", line.c_str());
        }
        if (idx == -1 || idx >= M_) {
            continue;
        }
        if (ignoreOutsideZrange_ && (rec.z < zMin_ || rec.z > zMax_)) {
            continue;
        }
        input.coords3dFloat.emplace_back(static_cast<float>(rec.x), static_cast<float>(rec.y), static_cast<float>(rec.z));
        input.featureIdx.push_back(rec.idx);
        input.obsWeight.push_back(static_cast<float>(rec.ct));
        std::vector<uint32_t> bufferidx;
        if (pt2buffer(bufferidx, rec.x, rec.y, tile) == 1) {
            tileData.idxinternal.push_back(npt);
        }
        for (const auto& key : bufferidx) {
            buffers3d[key].push_back(rec);
        }
        npt++;
    }
    for (const auto& entry : buffers3d) {
        auto buffer = getBoundaryBuffer(entry.first);
        buffer->addRecords3D(entry.second);
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseOneTileExtended3D(TileData<T>& tileData, TileKey tile) {
    std::unique_ptr<BoundedReadline> iter;
    try {
        iter = tileReader.get_tile_iterator(tile.row, tile.col);
    } catch (const std::exception &e) {
        warning("%s", e.what());
        return -1;
    }
    tileData.clear();
    tile2bound(tile, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax, tileSize);
    auto& input = tileData.emplaceExtended3D();
    std::unordered_map<uint32_t, std::vector<RecordExtendedT3D<T>>> extBuffers3d;
    if (M_ == 0) {
        M_ = lineParserPtr->featureDict.size();
    }

    std::string line;
    int32_t npt = 0;
    while (iter->next(line)) {
        RecordExtendedT3D<T> recExt;
        int32_t idx = lineParserPtr->parse(recExt, line);
        if (idx < -1) {
            error("Error parsing line: %s", line.c_str());
        }
        if (idx == -1 || idx >= M_) {
            continue;
        }
        if (ignoreOutsideZrange_ && (recExt.recBase.z < zMin_ || recExt.recBase.z > zMax_)) {
            continue;
        }
        input.extPts3d.push_back(recExt);
        std::vector<uint32_t> bufferidx;
        if (pt2buffer(bufferidx, recExt.recBase.x, recExt.recBase.y, tile) == 1) {
            tileData.idxinternal.push_back(npt);
        }
        for (const auto& key : bufferidx) {
            extBuffers3d[key].push_back(recExt);
        }
        npt++;
    }
    for (const auto& entry : extBuffers3d) {
        auto buffer = getBoundaryBuffer(entry.first);
        buffer->addRecordsExtended3D(entry.second, schema_, recordSize_);
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryFile3D(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
    std::lock_guard<std::mutex> lock(*(bufferPtr->mutex));
    std::ifstream ifs;
    if (auto* tmpFile = std::get_if<std::string>(&(bufferPtr->storage))) {
        ifs.open(*tmpFile, std::ios::binary);
        if (!ifs) {
            warning("%s: Failed to open temporary file %s", __func__, tmpFile->c_str());
            return -1;
        }
    } else {
        error("%s cannot be called when buffer is in memory", __func__);
    }
    int npt = 0;
    tileData.clear();
    bufferId2bound(bufferPtr->key, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    auto& input = tileData.emplaceStandard3D();
    while (true) {
        RecordT3D<T> rec;
        ifs.read(reinterpret_cast<char*>(&rec), sizeof(RecordT3D<T>));
        if (ifs.gcount() != sizeof(RecordT3D<T>)) break;
        input.pts3d.push_back(rec);
        if (isInternalToBuffer(rec.x, rec.y, bufferPtr->key)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryFileSingleMolecule3D(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
    std::lock_guard<std::mutex> lock(*(bufferPtr->mutex));
    std::ifstream ifs;
    if (auto* tmpFile = std::get_if<std::string>(&(bufferPtr->storage))) {
        ifs.open(*tmpFile, std::ios::binary);
        if (!ifs) {
            warning("%s: Failed to open temporary file %s", __func__, tmpFile->c_str());
            return -1;
        }
    } else {
        error("%s cannot be called when buffer is in memory", __func__);
    }
    int npt = 0;
    tileData.clear();
    bufferId2bound(bufferPtr->key, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    auto& input = tileData.emplaceSingleMolecule3D();
    while (true) {
        RecordT3D<T> rec;
        ifs.read(reinterpret_cast<char*>(&rec), sizeof(RecordT3D<T>));
        if (ifs.gcount() != sizeof(RecordT3D<T>)) break;
        input.coords3dFloat.emplace_back(static_cast<float>(rec.x), static_cast<float>(rec.y), static_cast<float>(rec.z));
        input.featureIdx.push_back(rec.idx);
        input.obsWeight.push_back(static_cast<float>(rec.ct));
        if (isInternalToBuffer(rec.x, rec.y, bufferPtr->key)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryFileExtended3D(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
    std::lock_guard<std::mutex> lock(*(bufferPtr->mutex));
    std::ifstream ifs;
    if (auto* tmpFile = std::get_if<std::string>(&(bufferPtr->storage))) {
        ifs.open(*tmpFile, std::ios::binary);
        if (!ifs) {
            warning("%s: Failed to open temporary file %s", __func__, tmpFile->c_str());
            return -1;
        }
    } else {
        error("%s cannot be called when buffer is in memory", __func__);
    }
    int npt = 0;
    tileData.clear();
    bufferId2bound(bufferPtr->key, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    auto& input = tileData.emplaceExtended3D();
    while (true) {
        std::vector<uint8_t> buf(recordSize_);
        ifs.read(reinterpret_cast<char*>(buf.data()), recordSize_);
        if (ifs.gcount() != recordSize_) break;
        auto *ptr = buf.data();
        RecordExtendedT3D<T> r;
        std::memcpy(&r.recBase, ptr, sizeof(r.recBase));
        for (auto &f : schema_) {
            auto *fp = ptr + f.offset;
            switch (f.type) {
                case FieldType::INT32: {
                    int32_t v;
                    std::memcpy(&v, fp, sizeof(v));
                    r.intvals.push_back(v);
                } break;
                case FieldType::FLOAT: {
                    float v;
                    std::memcpy(&v, fp, sizeof(v));
                    r.floatvals.push_back(v);
                } break;
                case FieldType::STRING: {
                    std::string s((char*)fp, f.size);
                    auto pos = s.find('\0');
                    if (pos!=std::string::npos) s.resize(pos);
                    r.strvals.push_back(s);
                } break;
            }
        }
        const bool isInternal = isInternalToBuffer(r.recBase.x, r.recBase.y, bufferPtr->key);
        input.extPts3d.push_back(std::move(r));
        if (isInternal) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryStandard3D(TileData<T>& tileData, InMemoryStorageStandard3D<T>* memStore, uint32_t bufferKey) {
    tileData.clear();
    bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    auto& input = tileData.emplaceStandard3D();
    int npt = 0;
    input.pts3d = std::move(memStore->data);
    for (const auto& rec : input.pts3d) {
        const bool isInternal = isInternalToBuffer(rec.x, rec.y, bufferKey);
        if (isInternal) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemorySingleMolecule3D(TileData<T>& tileData, InMemoryStorageStandard3D<T>* memStore, uint32_t bufferKey) {
    tileData.clear();
    bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    auto& input = tileData.emplaceSingleMolecule3D();
    int npt = 0;
    input.coords3dFloat.reserve(memStore->data.size());
    input.featureIdx.reserve(memStore->data.size());
    input.obsWeight.reserve(memStore->data.size());
    for (const auto& rec : memStore->data) {
        input.coords3dFloat.emplace_back(static_cast<float>(rec.x), static_cast<float>(rec.y), static_cast<float>(rec.z));
        input.featureIdx.push_back(rec.idx);
        input.obsWeight.push_back(static_cast<float>(rec.ct));
        if (isInternalToBuffer(rec.x, rec.y, bufferKey)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    memStore->data.clear();
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryExtended3D(TileData<T>& tileData, InMemoryStorageExtended3D<T>* memStore, uint32_t bufferKey) {
    tileData.clear();
    bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    auto& input = tileData.emplaceExtended3D();
    int npt = 0;
    input.extPts3d = std::move(memStore->dataExtended);
    for (const auto& rec : input.extPts3d) {
        const bool isInternal = isInternalToBuffer(rec.recBase.x, rec.recBase.y, bufferKey);
        if (isInternal) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryStandard3DWrapper(TileData<T>& tileData, IBoundaryStorage* storage, uint32_t bufferKey) {
    auto* memStore = dynamic_cast<InMemoryStorageStandard3D<T>*>(storage);
    if (!memStore) {
        error("%s: wrong in-memory storage type for 3D standard", __func__);
    }
    return parseBoundaryMemoryStandard3D(tileData, memStore, bufferKey);
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemorySingleMolecule3DWrapper(TileData<T>& tileData, IBoundaryStorage* storage, uint32_t bufferKey) {
    if (!storage) {
        tileData.clear();
        bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
        return 0;
    }
    auto* memStore = dynamic_cast<InMemoryStorageStandard3D<T>*>(storage);
    if (!memStore) {
        error("%s: wrong in-memory storage type for 3D standard", __func__);
    }
    return parseBoundaryMemorySingleMolecule3D(tileData, memStore, bufferKey);
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryExtended3DWrapper(TileData<T>& tileData, IBoundaryStorage* storage, uint32_t bufferKey) {
    auto* memStore = dynamic_cast<InMemoryStorageExtended3D<T>*>(storage);
    if (!memStore) {
        error("%s: wrong in-memory storage type for 3D extended", __func__);
    }
    return parseBoundaryMemoryExtended3D(tileData, memStore, bufferKey);
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultStandard3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    assert(outputMode_ == MinibatchOutputMode::Standard);
    assert(!outputBinary_);
    assert(!outputBackgroundProbExpand_);
    if (outputBackgroundProbDense_) {
        assert(phi0 != nullptr && phi0->size() == size_t(topVals.rows()));
    }
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    auto& lines = result.getPayload<typename ResultBuf::TextLines>();
    size_t N = tileData.coords3d.size();
    std::vector<bool> internal(N, 0);
    for (auto j : tileData.idxinternal) {
        if (tileData.orgpts2pixel[j] < 0) {
            continue;
        }
        internal[tileData.orgpts2pixel[j]] = true;
    }
    char buf[65536];
    for (size_t j = 0; j < N; ++j) {
        if (!internal[j]) {
            continue;
        }
        int len = std::snprintf(
            buf, sizeof(buf), "%.*f\t%.*f\t%.*f",
            floatCoordDigits, tileData.coords3d[j].x * pixelResolution_,
            floatCoordDigits, tileData.coords3d[j].y * pixelResolution_,
            floatCoordDigits, tileData.coords3d[j].z * pixelResolutionZ_
        );
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%d",
                topIds(j, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing output line", __func__);
            }
            len += n;
        }
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%.*e",
                probDigits, topVals(j, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing output line", __func__);
            }
            len += n;
        }
        if (outputBackgroundProbDense_) {
            len += std::snprintf(buf + len, sizeof(buf) - len, "\t");
            const auto& phi0_map = (*phi0)[j];
            for (const auto& kv : phi0_map) {
                int n = std::snprintf(
                    buf + len, sizeof(buf) - len, "%s:%.*f,",
                    featureNames[kv.first].c_str(),
                    probDigits, kv.second
                );
                if (n < 0) {
                    error("%s: error writing background probability", __func__);
                }
                if (n >= int(sizeof(buf) - len)) {
                    warning("%s: buffer overflow while writing dense background probabilities, not all genes are written", __func__);
                    break;
                }
                len += n;
            }
            if (len > 0 && buf[len - 1] == ',') {
                len -= 1;
            }
        }
        buf[len++] = '\n';
        lines.emplace_back(buf, len);
    }
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultWithOriginalData3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket,
std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    if (outputBackgroundProbExpand_) {
        assert(phi0 != nullptr && phi0->size() == size_t(topVals.rows()));
    }
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    auto& lines = result.getPayload<typename ResultBuf::TextLines>();
    int32_t nrows = topVals.rows();
    char buf[65536];
    const auto* extInput = useExtended_ ? &tileData.extended3D() : nullptr;
    const auto* stdInput = useExtended_ ? nullptr : &tileData.standard3D();
    for (size_t i = 0; i < tileData.idxinternal.size(); ++i) {
        int32_t idxorg = tileData.idxinternal[i];
        int32_t idx = tileData.orgpts2pixel[idxorg];
        if (idx < 0 || idx >= nrows) {
            continue;
        }
        const RecordT3D<T>* recPtr;
        if (useExtended_) {
            recPtr = &extInput->extPts3d[idxorg].recBase;
        } else {
            recPtr = &stdInput->pts3d[idxorg];
        }
        const RecordT3D<T>& rec = *recPtr;
        int len = 0;
        if constexpr (std::is_same_v<T, int32_t>) {
            len = std::snprintf(
                buf, sizeof(buf), "%d\t%d\t%d\t",
                rec.x, rec.y, rec.z
            );
        } else {
            len = std::snprintf(
                buf, sizeof(buf), "%.*f\t%.*f\t%.*f\t",
                floatCoordDigits, rec.x,
                floatCoordDigits, rec.y,
                floatCoordDigits, rec.z
            );
        }
        len += std::snprintf(
            buf + len, sizeof(buf) - len, "%s\t%d",
            featureNames[rec.idx].c_str(), rec.ct
        );
        if (outputBackgroundProbExpand_) {
            float bgprob = 0.0f;
            auto& phi0_map = (*phi0)[idx];
            auto it = phi0_map.find(rec.idx);
            if (it != phi0_map.end()) {
                bgprob = it->second;
            }
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%.*f",
                probDigits, bgprob
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing background probability", __func__);
            }
            len += n;
        }
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%d",
                topIds(idx, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing output line", __func__);
            }
            len += n;
        }
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len, sizeof(buf) - len, "\t%.*e",
                probDigits, topVals(idx, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing output line", __func__);
            }
            len += n;
        }
        if (useExtended_) {
            const RecordExtendedT3D<T>& recExt = extInput->extPts3d[idxorg];
            for (auto v : recExt.intvals) {
                len += std::snprintf(buf + len, sizeof(buf) - len, "\t%d", v);
                if (len >= int(sizeof(buf))) {
                    error("%s: buffer overflow while writing intvals", __func__);
                }
            }
            for (auto v : recExt.floatvals) {
                len += std::snprintf(buf + len, sizeof(buf) - len, "\t%f", v);
                if (len >= int(sizeof(buf))) {
                    error("%s: buffer overflow while writing floatvals", __func__);
                }
            }
            for (auto& v : recExt.strvals) {
                len += std::snprintf(buf + len, sizeof(buf) - len, "\t%s", v.c_str());
                if (len >= int(sizeof(buf))) {
                    error("%s: buffer overflow while writing strvals", __func__);
                }
            }
        }
        buf[len++] = '\n';
        lines.emplace_back(buf, len);
    }
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultWithBackground3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    assert(!outputBackgroundProbDense_);
    if (!phi0) {
        error("%s: background probabilities are missing", __func__);
    }
    if (outputBackgroundProbExpand_) {
        assert(phi0->size() == size_t(topVals.rows()));
    }
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    auto& lines = result.getPayload<typename ResultBuf::TextLines>();
    size_t N = tileData.coords3d.size();
    std::vector<bool> internal(N, 0);
    for (auto j : tileData.idxinternal) {
        if (tileData.orgpts2pixel[j] < 0) {
            continue;
        }
        internal[tileData.orgpts2pixel[j]] = true;
    }
    char buf[512];
    for (size_t j = 0; j < N; ++j) {
        if (!internal[j]) {
            continue;
        }
        for (const auto& kv : (*phi0)[j]) {
            int len = std::snprintf(
                buf, sizeof(buf), "%.*f\t%.*f\t%.*f",
                floatCoordDigits, tileData.coords3d[j].x * pixelResolution_,
                floatCoordDigits, tileData.coords3d[j].y * pixelResolution_,
                floatCoordDigits, tileData.coords3d[j].z * pixelResolutionZ_
            );
            len += std::snprintf(
                buf + len, sizeof(buf) - len, "\t%s\t%.*f",
                featureNames[kv.first].c_str(),
                probDigits, kv.second
            );
            for (int32_t k = 0; k < topk_; ++k) {
                int n = std::snprintf(
                    buf + len, sizeof(buf) - len, "\t%d",
                    topIds(j, k)
                );
                if (n < 0 || n >= int(sizeof(buf) - len)) {
                    error("%s: error writing output line", __func__);
                }
                len += n;
            }
            for (int32_t k = 0; k < topk_; ++k) {
                int n = std::snprintf(
                    buf + len, sizeof(buf) - len, "\t%.*e",
                    probDigits, topVals(j, k)
                );
                if (n < 0 || n >= int(sizeof(buf) - len)) {
                    error("%s: error writing output line", __func__);
                }
                len += n;
            }
            buf[len++] = '\n';
            lines.emplace_back(buf, len);
        }
    }
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultBinary3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    (void)phi0;
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    const auto* smInput = isSingleMoleculeMode() ? &tileData.singleMolecule3D() : nullptr;
    if (isSingleMoleculeMode()) {
        result.emplacePayload<typename ResultBuf::OutputObjs3DFeatureFloat>();
    } else if (isSingleFeaturePixelMode()) {
        result.emplacePayload<typename ResultBuf::OutputObjs3DFeature>();
    } else {
        result.emplacePayload<typename ResultBuf::OutputObjs3D>();
    }
    size_t N = isSingleMoleculeMode() ? smInput->coords3dFloat.size() : tileData.coords3d.size();
    std::vector<bool> internal(N, 0);
    for (auto j : tileData.idxinternal) {
        if (tileData.orgpts2pixel[j] < 0) {
            continue;
        }
        internal[tileData.orgpts2pixel[j]] = true;
    }
    for (size_t j = 0; j < N; ++j) {
        if (!internal[j]) {
            continue;
        }
        if (isSingleMoleculeMode()) {
            const auto& coord = smInput->coords3dFloat[j];
            PixTopProbsFeature3D<float> rec(coord.x, coord.y, coord.z, smInput->featureIdx[j]);
            rec.ks.resize(topk_);
            rec.ps.resize(topk_);
            for (int32_t k = 0; k < topk_; ++k) {
                rec.ks[k] = topIds(j, k);
                rec.ps[k] = topVals(j, k);
            }
            result.getPayload<typename ResultBuf::OutputObjs3DFeatureFloat>().emplace_back(std::move(rec));
        } else if (isSingleFeaturePixelMode()) {
            PixTopProbsFeature3D<int32_t> rec(tileData.coords3d[j], tileData.rowFeatureIdx[j]);
            rec.ks.resize(topk_);
            rec.ps.resize(topk_);
            for (int32_t k = 0; k < topk_; ++k) {
                rec.ks[k] = topIds(j, k);
                rec.ps[k] = topVals(j, k);
            }
            result.getPayload<typename ResultBuf::OutputObjs3DFeature>().emplace_back(std::move(rec));
        } else {
            PixTopProbs3D<int32_t> rec(tileData.coords3d[j]);
            rec.ks.resize(topk_);
            rec.ps.resize(topk_);
            for (int32_t k = 0; k < topk_; ++k) {
                rec.ks[k] = topIds(j, k);
                rec.ps[k] = topVals(j, k);
            }
            result.getPayload<typename ResultBuf::OutputObjs3D>().emplace_back(std::move(rec));
        }
    }
    return result;
}
