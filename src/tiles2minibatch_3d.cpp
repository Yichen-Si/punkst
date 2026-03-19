#include "tiles2minibatch.hpp"
#include "bccgrid.hpp"
#include "numerical_utils.hpp"
#include <cmath>
#include <numeric>
#include <fcntl.h>
#include <thread>
#include <algorithm>

namespace {

int32_t thin3d_modular_step(int32_t nLevels) {
    if (nLevels <= 1) {
        return 1;
    }
    int32_t step = (nLevels / 2) + 1;
    while (std::gcd(step, nLevels) != 1) {
        ++step;
    }
    return step;
}

int32_t thin3d_modular_offset(int32_t nLevels) {
    if (nLevels <= 1) {
        return 0;
    }
    return nLevels / 3;
}

int32_t thin3d_modular_inverse(int32_t a, int32_t mod) {
    int32_t t = 0, newT = 1;
    int32_t r = mod, newR = a;
    while (newR != 0) {
        const int32_t q = r / newR;
        const int32_t tmpT = t - q * newT;
        t = newT;
        newT = tmpT;
        const int32_t tmpR = r - q * newR;
        r = newR;
        newR = tmpR;
    }
    if (r != 1) {
        error("%s: modular inverse does not exist", __func__);
    }
    if (t < 0) {
        t += mod;
    }
    return t;
}

int32_t thin3d_phase_to_z_index(int32_t phaseIndex, int32_t nLevels) {
    const int32_t step = thin3d_modular_step(nLevels);
    const int32_t offset = thin3d_modular_offset(nLevels);
    return (step * phaseIndex + offset) % nLevels;
}

int32_t thin3d_z_index_to_phase(int32_t zIndex, int32_t nLevels) {
    const int32_t stepInv = thin3d_modular_inverse(thin3d_modular_step(nLevels), nLevels);
    int32_t shifted = zIndex - thin3d_modular_offset(nLevels);
    shifted %= nLevels;
    if (shifted < 0) {
        shifted += nLevels;
    }
    return (stepInv * shifted) % nLevels;
}

} // namespace

template<typename T>
void Tiles2MinibatchBase<T>::set3Dparameters(bool isThin, double zMin, double zMax, float zScale, float zRes, int32_t n, bool enforceZrange, float standard3DBccGridDist) {
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
    zScale_ = zScale > 0 ? zScale : 1.0;
    pixelResolutionZ_ = zRes > 0 ? zRes : pixelResolution_;
    nInitAnchorPerPix_ = n;
    ignoreOutsideZrange_ = enforceZrange;
    standard3DBccGridDist_ = standard3DBccGridDist;
    thin3DAnchorRefDist_ = -1.0f;
    if (useThin3DAnchors_) {
        if (nInitAnchorPerPix_ < 0) {
            error("Thin 3D anchor mode requires a non-negative number of initial anchors per pixel");
        }
    } else {
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
void Tiles2MinibatchBase<T>::forEachAnchorCandidate3D(const TileData<T>& tileData,
    const std::function<void(uint32_t, float, const AnchorKey3D&)>& emit) const {
    BCCGrid bccGrid(standard3DBccSize_);
    forEachAnchorCandidate3D(tileData, bccGrid, emit);
}

template<typename T>
void Tiles2MinibatchBase<T>::forEachAnchorCandidate3D(const TileData<T>& tileData, const BCCGrid& bccGrid,
    const std::function<void(uint32_t, float, const AnchorKey3D&)>& emit) const {
    auto neighborOffsets = bccGrid.face_adjacent_offsets();
    auto assign_pt = [&](const auto& pt) {
        if (ignoreOutsideZrange_ && (pt.z < zMin_ || pt.z > zMax_)) {
            return;
        }
        int32_t q1, q2, q3;
        bccGrid.cart_to_lattice(q1, q2, q3, pt.x, pt.y, pt.z * zScale_);
        emit(pt.idx, static_cast<float>(pt.ct), AnchorKey3D{q1, q2, q3, 0, 0, 0});
        for (const auto& delta : neighborOffsets) {
            emit(pt.idx, static_cast<float>(pt.ct),
                AnchorKey3D{q1 + delta[0], q2 + delta[1], q3 + delta[2], 0, 0, 0});
        }
    };
    if (useExtended_) {
        for (const auto& pt : tileData.extPts3d) {
            assign_pt(pt.recBase);
        }
    } else {
        for (const auto& pt : tileData.pts3d) {
            assign_pt(pt);
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
    z = static_cast<float>(zd / zScale_);
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
void Tiles2MinibatchBase<T>::forEachAnchorCandidateThin3D(const TileData<T>& tileData, const HexGrid& hexGrid_, int32_t nMoves_,
    const std::function<void(uint32_t, float, const AnchorKey2D&)>& emit) const {
    const int32_t nZLevels = nMoves_ * nMoves_;
    // assign each pixel to nPerPixel anchors with closest z
    const int32_t nPerPixel = std::min<int32_t>(
        (nInitAnchorPerPix_ > 0) ? nInitAnchorPerPix_ : (nZLevels / 2),
        nZLevels);
    // distance between adjacent anchor levels in z
    const double zStep = (zMax_ - zMin_) / static_cast<double>(nZLevels);
    std::vector<double> zCenters(static_cast<size_t>(nZLevels));
    for (int32_t zIndex = 0; zIndex < nZLevels; ++zIndex) {
        zCenters[static_cast<size_t>(zIndex)] = zMin_ + (zIndex + 0.5) * zStep;
    }
    auto assign_pt = [&](const auto& pt) {
        if (ignoreOutsideZrange_ && (pt.z < zMin_ || pt.z > zMax_)) {
            return;
        }
        const double rel = (static_cast<double>(pt.z) - zMin_) / zStep - 0.5;
        int32_t center = static_cast<int32_t>(std::llround(rel)); // nearest
        center = std::max<int32_t>(0, std::min<int32_t>(center, nZLevels - 1));
        int32_t left = center - 1;
        int32_t right = center + 1;
        // moving outwards
        for (int32_t picked = 0; picked < nPerPixel; ++picked) {
            int32_t zIndex = -1;
            if (picked == 0) {
                zIndex = center;
            } else if (left < 0) {
                zIndex = right++;
            } else if (right >= nZLevels) {
                zIndex = left--;
            } else {
                const double leftDist = std::abs(static_cast<double>(pt.z) - zCenters[static_cast<size_t>(left)]);
                const double rightDist = std::abs(static_cast<double>(pt.z) - zCenters[static_cast<size_t>(right)]);
                if (leftDist <= rightDist) {
                    zIndex = left--;
                } else {
                    zIndex = right++;
                }
            }
            const int32_t phaseIndex = thin3d_z_index_to_phase(zIndex, nZLevels);
            const int32_t ir = phaseIndex / nMoves_;
            const int32_t ic = phaseIndex % nMoves_;
            int32_t hx, hy;
            hexGrid_.cart_to_axial(hx, hy, pt.x, pt.y, ic * 1.0 / nMoves_, ir * 1.0 / nMoves_);
            emit(pt.idx, static_cast<float>(pt.ct), AnchorKey2D{hx, hy, ic, ir});
        }
    };
    if (useExtended_) {
        for (const auto& pt : tileData.extPts3d) {
            assign_pt(pt.recBase);
        }
    } else {
        for (const auto& pt : tileData.pts3d) {
            assign_pt(pt);
        }
    }
}

template<typename T>
void Tiles2MinibatchBase<T>::anchorKeyToCoordThin3D(float& x, float& y, float& z, const AnchorKey2D& key, const HexGrid& hexGrid_, int32_t nMoves_) const {
    anchorKeyToCoord2D(x, y, key, hexGrid_, nMoves_);
    const int32_t ic = std::get<2>(key);
    const int32_t ir = std::get<3>(key);
    const int32_t phaseIndex = ir * nMoves_ + ic;
    const int32_t nZLevels = nMoves_ * nMoves_;
    const int32_t zIndex = thin3d_phase_to_z_index(phaseIndex, nZLevels);
    const double zStep = (zMax_ - zMin_) / static_cast<double>(nZLevels);
    z = static_cast<float>(zMin_ + (static_cast<double>(zIndex) + 0.5) * zStep);
}

template<typename T>
double Tiles2MinibatchBase<T>::thin3DReferenceAnchorDistance(const HexGrid& hexGrid_, int32_t nMoves_) const {
    if (thin3DAnchorRefDist_ > 0) {
        return thin3DAnchorRefDist_;
    }
    if (nMoves_ <= 1) {
        error("%s: thin 3D reference distance requires nMoves > 1", __func__);
    }
    const AnchorKey2D originKey{0, 0, 0, 0};
    float x0, y0, z0;
    anchorKeyToCoordThin3D(x0, y0, z0, originKey, hexGrid_, nMoves_);
    double best = std::numeric_limits<double>::infinity();
    for (int32_t hy = -1; hy <= 1; ++hy) {
        for (int32_t hx = -1; hx <= 1; ++hx) {
            for (int32_t ir = 0; ir < nMoves_; ++ir) {
                for (int32_t ic = 0; ic < nMoves_; ++ic) {
                    const AnchorKey2D key{hx, hy, ic, ir};
                    if (hx == 0 && hy == 0 && ic == 0 && ir == 0) {
                        continue;
                    }
                    float x, y, z;
                    anchorKeyToCoordThin3D(x, y, z, key, hexGrid_, nMoves_);
                    const double dx = static_cast<double>(x) - x0;
                    const double dy = static_cast<double>(y) - y0;
                    const double dz = (static_cast<double>(z) - z0) * zScale_;
                    const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                    if (dist > 1e-8 && dist < best) {
                        best = dist;
                    }
                }
            }
        }
    }
    if (!std::isfinite(best)) {
        error("%s: failed to determine thin 3D reference anchor distance", __func__);
    }
    thin3DAnchorRefDist_ = static_cast<float>(best);
    return best;
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::buildAnchorsThin3D(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, const HexGrid& hexGrid_, int32_t nMoves_, double minCount) {
    if (coordDim_ != MinibatchCoordDim::Dim3) {
        return -1;
    }
    if (nMoves_ <= 1) {
        error("%s: thin 3D anchors require nMoves > 1", __func__);
    }
    if (!(zMax_ > zMin_)) {
        error("%s: invalid thin 3D anchor z range", __func__);
    }

    anchors.clear();
    documents.clear();
    std::map<AnchorKey2D, std::unordered_map<uint32_t, float>> hexAggregation;
    forEachAnchorCandidateThin3D(tileData, hexGrid_, nMoves_, [&](uint32_t idx, float ct, const AnchorKey2D& key) {
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
double Tiles2MinibatchBase<T>::buildMinibatchCore3D(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    double supportRadius, double refDist, double weightAtRefDist) {
    BCCGrid bccGrid(standard3DBccSize_);
    return buildMinibatchCore3D(tileData, anchors, minibatch, bccGrid, supportRadius, refDist, weightAtRefDist);
}

template<typename T>
double Tiles2MinibatchBase<T>::buildMinibatchCore3D(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    const BCCGrid& bccGrid, double supportRadius, double refDist, double weightAtRefDist) {
    debug("%s: building minibatch with %zu anchors and %zu documents", __func__, anchors.size(), tileData.pts3d.size() + tileData.extPts3d.size());
    if (minibatch.n <= 0) {
        return 0.0;
    }
    assert(refDist > 0.0 && weightAtRefDist > 0.0 && weightAtRefDist < 1.0);

    const float res = pixelResolution_;
    const float resZ = pixelResolutionZ_;

    std::vector<Eigen::Triplet<float>> tripletsMtx;
    std::vector<Eigen::Triplet<float>> tripletsWij;
    tripletsMtx.reserve(tileData.pts3d.size() + tileData.extPts3d.size());
    tripletsWij.reserve(tileData.pts3d.size() + tileData.extPts3d.size());

    std::unordered_map<std::tuple<int32_t, int32_t, int32_t>,
        std::pair<std::unordered_map<uint32_t, float>, std::vector<uint32_t>>,
        Tuple3Hash> pixAgg;
    uint32_t idxOriginal = 0;

    if (useExtended_) {
        tileData.orgpts2pixel.assign(tileData.extPts3d.size(), -1);
        for (const auto& pt : tileData.extPts3d) {
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
        tileData.orgpts2pixel.assign(tileData.pts3d.size(), -1);
        for (const auto& pt : tileData.pts3d) {
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
        if (zScale_ != 1.0f) {
            for (auto& pt : pc.pts) {
                pt.z *= zScale_;
            }
        }
    } else {
        bccNeighborOffsets = bccGrid.face_adjacent_offsets();
        for (uint32_t i = 0; i < static_cast<uint32_t>(anchors.size()); ++i) {
            int32_t q1, q2, q3;
            bccGrid.cart_to_lattice(q1, q2, q3, anchors[i].x, anchors[i].y, anchors[i].z * zScale_);
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

        const double x = static_cast<double>(px) * res;
        const double y = static_cast<double>(py) * res;
        const double z = static_cast<double>(pz) * resZ;
        if (useThin3DAnchors_) {
            float xyz[3] = {
                static_cast<float>(x),
                static_cast<float>(y),
                static_cast<float>(z * zScale_)
            };
            size_t n = kdtree->radiusSearch(xyz, l2radius, indices_dists);
            if (n == 0) {
                continue;
            }
            for (size_t i = 0; i < n; ++i) {
                uint32_t idx = indices_dists[i].first;
                const float dist = std::sqrt(indices_dists[i].second);
                candidates.emplace_back(
                    idx,
                    anchor_distance_weight(dist, refDist, weightAtRefDist));
            }
        } else {
            int32_t q1, q2, q3;
            bccGrid.cart_to_lattice(q1, q2, q3, x, y, z * zScale_);
            auto emitCandidate = [&](int32_t cq1, int32_t cq2, int32_t cq3) {
                auto it = anchorKeyToIndex.find(AnchorKey3D{cq1, cq2, cq3, 0, 0, 0});
                if (it == anchorKeyToIndex.end()) {
                    return;
                }
                const AnchorPoint& anchor = originalAnchors[it->second];
                const double dx = x - static_cast<double>(anchor.x);
                const double dy = y - static_cast<double>(anchor.y);
                const double dz = (z - static_cast<double>(anchor.z)) * zScale_;
                const double dist = std::sqrt(dx * dx + dy * dy + dz * dz);
                candidates.emplace_back(
                    it->second,
                    anchor_distance_weight(dist, refDist, weightAtRefDist));
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
        tileData.pts3d.push_back(rec);
        std::vector<uint32_t> bufferidx;
        if (pt2buffer(bufferidx, rec.x, rec.y, tile) == 1) {
            tileData.idxinternal.push_back(npt);
        }
        for (const auto& key : bufferidx) {
            tileData.buffers3d[key].push_back(rec);
        }
        npt++;
    }
    for (const auto& entry : tileData.buffers3d) {
        auto buffer = getBoundaryBuffer(entry.first);
        buffer->addRecords3D(entry.second);
    }
    tileData.buffers3d.clear();
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
        tileData.extPts3d.push_back(recExt);
        std::vector<uint32_t> bufferidx;
        if (pt2buffer(bufferidx, recExt.recBase.x, recExt.recBase.y, tile) == 1) {
            tileData.idxinternal.push_back(npt);
        }
        for (const auto& key : bufferidx) {
            tileData.extBuffers3d[key].push_back(recExt);
        }
        npt++;
    }
    for (const auto& entry : tileData.extBuffers3d) {
        auto buffer = getBoundaryBuffer(entry.first);
        buffer->addRecordsExtended3D(entry.second, schema_, recordSize_);
    }
    tileData.extBuffers3d.clear();
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
    while (true) {
        RecordT3D<T> rec;
        ifs.read(reinterpret_cast<char*>(&rec), sizeof(RecordT3D<T>));
        if (ifs.gcount() != sizeof(RecordT3D<T>)) break;
        tileData.pts3d.push_back(rec);
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
        tileData.extPts3d.push_back(std::move(r));
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
    int npt = 0;
    tileData.pts3d = std::move(memStore->data);
    for (const auto& rec : tileData.pts3d) {
        const bool isInternal = isInternalToBuffer(rec.x, rec.y, bufferKey);
        if (isInternal) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryExtended3D(TileData<T>& tileData, InMemoryStorageExtended3D<T>* memStore, uint32_t bufferKey) {
    tileData.clear();
    bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    int npt = 0;
    tileData.extPts3d = std::move(memStore->dataExtended);
    for (const auto& rec : tileData.extPts3d) {
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
        result.outputLines.emplace_back(buf, len);
    }
    result.npts = result.outputLines.size();
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultWithOriginalData3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket,
std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    if (outputBackgroundProbExpand_) {
        assert(phi0 != nullptr && phi0->size() == size_t(topVals.rows()));
    }
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    int32_t nrows = topVals.rows();
    char buf[65536];
    for (size_t i = 0; i < tileData.idxinternal.size(); ++i) {
        int32_t idxorg = tileData.idxinternal[i];
        int32_t idx = tileData.orgpts2pixel[idxorg];
        if (idx < 0 || idx >= nrows) {
            continue;
        }
        const RecordT3D<T>* recPtr;
        if (useExtended_) {
            recPtr = &tileData.extPts3d[idxorg].recBase;
        } else {
            recPtr = &tileData.pts3d[idxorg];
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
            const RecordExtendedT3D<T>& recExt = tileData.extPts3d[idxorg];
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
        result.outputLines.emplace_back(buf, len);
    }
    result.npts = result.outputLines.size();
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
            result.outputLines.emplace_back(buf, len);
        }
    }
    result.npts = result.outputLines.size();
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultBinary3D(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    (void)phi0;
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    result.useObj = true;
    size_t N = tileData.coords3d.size();
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
        PixTopProbs3D<int32_t> rec(tileData.coords3d[j]);
        rec.ks.resize(topk_);
        rec.ps.resize(topk_);
        for (int32_t k = 0; k < topk_; ++k) {
            rec.ks[k] = topIds(j, k);
            rec.ps[k] = topVals(j, k);
        }
        result.outputObjs3d.emplace_back(std::move(rec));
    }
    result.npts = result.outputObjs3d.size();
    return result;
}
