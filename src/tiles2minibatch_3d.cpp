#include "tiles2minibatch.hpp"
#include "bccgrid.hpp"
#include "numerical_utils.hpp"
#include <cmath>
#include <numeric>
#include <fcntl.h>
#include <thread>
#include <algorithm>

template<typename T>
int32_t Tiles2MinibatchBase<T>::buildAnchors3D(TileData<T>& tileData, std::vector<AnchorPoint>& anchors, std::vector<SparseObs>& documents, HexGrid& hexGrid_, int32_t nMoves_, double minCount) {
    if (coordDim_ != MinibatchCoordDim::Dim3) {
        return -1;
    }
    anchors.clear();
    documents.clear();
    BCCGrid bccGrid(hexGrid_.size);
    using GridKey3 = std::tuple<int32_t, int32_t, int32_t, int32_t, int32_t, int32_t>;
    std::map<GridKey3, std::unordered_map<uint32_t, float>> bccAggregation;
    auto assign_pt = [&](const auto& pt) {
        for (int32_t ir = 0; ir < nMoves_; ++ir) {
            for (int32_t ic = 0; ic < nMoves_; ++ic) {
                for (int32_t iz = 0; iz < nMoves_; ++iz) {
                    int32_t q1, q2, q3;
                    double offset_x = (static_cast<double>(ic) / nMoves_) * bccGrid.size;
                    double offset_y = (static_cast<double>(ir) / nMoves_) * bccGrid.size;
                    double offset_z = (static_cast<double>(iz) / nMoves_) * bccGrid.size;
                    bccGrid.cart_to_lattice(q1, q2, q3, pt.x, pt.y, pt.z, offset_x, offset_y, offset_z);
                    auto key = std::make_tuple(q1, q2, q3, ic, ir, iz);
                    bccAggregation[key][pt.idx] += pt.ct;
                }
            }
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
        int32_t q1 = std::get<0>(key);
        int32_t q2 = std::get<1>(key);
        int32_t q3 = std::get<2>(key);
        int32_t ic = std::get<3>(key);
        int32_t ir = std::get<4>(key);
        int32_t iz = std::get<5>(key);
        double offset_x = (static_cast<double>(ic) / nMoves_) * bccGrid.size;
        double offset_y = (static_cast<double>(ir) / nMoves_) * bccGrid.size;
        double offset_z = (static_cast<double>(iz) / nMoves_) * bccGrid.size;
        double x, y, z;
        bccGrid.lattice_to_cart(x, y, z, q1, q2, q3, offset_x, offset_y, offset_z);
        anchors.emplace_back(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
    }
    return documents.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::buildMinibatchCore3D(TileData<T>& tileData,
    std::vector<AnchorPoint>& anchors, Minibatch& minibatch,
    double distR, double distNu) {

    if (minibatch.n <= 0) {
        return 0;
    }
    assert(distR > 0.0 && distNu > 0.0);

    PointCloud<float> pc;
    pc.pts = std::move(anchors);
    kd_tree_f3_t kdtree(3, pc, {10});
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;

    const float res = pixelResolution_;
    const float radius = static_cast<float>(distR);
    const float l2radius = radius * radius;
    const float nu = static_cast<float>(distNu);

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
            int32_t x = static_cast<int32_t>(std::floor(pt.recBase.x / res));
            int32_t y = static_cast<int32_t>(std::floor(pt.recBase.y / res));
            int32_t z = static_cast<int32_t>(std::floor(pt.recBase.z / res));
            auto key = std::make_tuple(x, y, z);
            pixAgg[key].first[pt.recBase.idx] += pt.recBase.ct;
            pixAgg[key].second.push_back(idxOriginal++);
        }
    } else {
        tileData.orgpts2pixel.assign(tileData.pts3d.size(), -1);
        for (const auto& pt : tileData.pts3d) {
            int32_t x = static_cast<int32_t>(std::floor(pt.x / res));
            int32_t y = static_cast<int32_t>(std::floor(pt.y / res));
            int32_t z = static_cast<int32_t>(std::floor(pt.z / res));
            auto key = std::make_tuple(x, y, z);
            pixAgg[key].first[pt.idx] += pt.ct;
            pixAgg[key].second.push_back(idxOriginal++);
        }
    }

    tileData.coords3d.clear();
    tileData.coords3d.reserve(pixAgg.size());
    uint32_t npt = 0;
    for (auto& kv : pixAgg) {
        int32_t px = std::get<0>(kv.first);
        int32_t py = std::get<1>(kv.first);
        int32_t pz = std::get<2>(kv.first);
        float xyz[3] = {px * res, py * res, pz * res};
        size_t n = kdtree.radiusSearch(xyz, l2radius, indices_dists);
        if (n == 0) {
            continue;
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

        for (size_t i = 0; i < n; ++i) {
            uint32_t idx = indices_dists[i].first;
            float dist = std::pow(indices_dists[i].second, 0.5f);
            dist = std::max(std::min(1.f - std::pow(dist / radius, nu), 0.95f), 0.05f);
            tripletsWij.emplace_back(npt, static_cast<int>(idx), dist);
        }

        ++npt;
    }
    anchors = std::move(pc.pts);

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

    return minibatch.N;
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
        tileData.extPts3d.push_back(std::move(r));
        if (isInternalToBuffer(r.recBase.x, r.recBase.y, bufferPtr->key)) {
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
    tileData.pts3d = std::move(memStore->data);
    int npt = 0;
    for(const auto& rec : tileData.pts3d) {
        if (isInternalToBuffer(rec.x, rec.y, bufferKey)) {
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
    tileData.extPts3d = std::move(memStore->dataExtended);
    int npt = 0;
    for(const auto& rec : tileData.extPts3d) {
        if (isInternalToBuffer(rec.recBase.x, rec.recBase.y, bufferKey)) {
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
            floatCoordDigits, tileData.coords3d[j].z * pixelResolution_
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
                floatCoordDigits, tileData.coords3d[j].z * pixelResolution_
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
