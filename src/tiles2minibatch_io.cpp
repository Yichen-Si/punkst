#include "tiles2minibatch.hpp"
#include "numerical_utils.hpp"
#include <cmath>
#include <numeric>
#include <fcntl.h>
#include <thread>
#include <algorithm>

template<typename T>
void Tiles2MinibatchBase<T>::setupOutput() {
    #if !defined(_WIN32)
        // ensure includes present
    #endif
    assert(!(outputBackgroundProbDense_ && outputBackgroundProbExpand_));
    std::string outputFile = outPref + (outputBinary_ ? ".bin" : ".tsv");
    fdMain = ::open(outputFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdMain < 0) {
        error("Error opening main output file: %s", outputFile.c_str());
    }
    if (!outputBinary_) {
        std::string header_str = composeHeader();
        if (!write_all(fdMain, header_str.data(), header_str.size())) {
            error("Error writing header_str to main output file: %s", outputFile.c_str());
        }
        headerSize = header_str.size();
        outputSize = headerSize;
    }
    std::string indexFile = outPref + ".index";
    fdIndex = ::open(indexFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) {
        error("Error opening index output file: %s", indexFile.c_str());
    }
    int32_t K = getFactorCount();
    // Write index header
    IndexHeader idxHeader;
    idxHeader.magic = PUNKST_INDEX_MAGIC;
    idxHeader.mode = (K << 16) | 0x8;
    idxHeader.tileSize = tileSize;
    idxHeader.topK = topk_;
    idxHeader.pixelResolution = pixelResolution_;
    const auto& box = tileReader.getGlobalBox();
    idxHeader.xmin = box.xmin; idxHeader.xmax = box.xmax;
    idxHeader.ymin = box.ymin; idxHeader.ymax = box.ymax;
    if (outputBinary_) {
        idxHeader.mode |= 0x7;
        size_t coordBytes = (coordDim_ == MinibatchCoordDim::Dim3 ? 3u : 2u) * sizeof(int32_t);
        outputRecordSize_ = coordBytes + topk_ * sizeof(int32_t) + topk_ * sizeof(float);
        idxHeader.coordType = 1;
        idxHeader.recordSize = outputRecordSize_;
    }

    if (!write_all(fdIndex, &idxHeader, sizeof(idxHeader))) {
        error("Error writing header to index output file: %s", indexFile.c_str());
    }

    if (!outputAnchor_) return;

    // setup anchor output
    std::string anchorFile = outPref + ".anchors.tsv";
    fdAnchor = ::open(anchorFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdAnchor < 0) {
        error("Error opening anchor output file: %s", anchorFile.c_str());
    }
    std::string header_str = "#x\ty";
    if (coordDim_ == MinibatchCoordDim::Dim3) {
        header_str += "\tz";
    }
    for (int32_t i = 0; i < topk_; ++i) header_str += "\tK" + std::to_string(i+1);
    for (int32_t i = 0; i < topk_; ++i) header_str += "\tP" + std::to_string(i+1);
    header_str += "\n";
    if (!write_all(fdAnchor, header_str.data(), header_str.size())) {
        error("Error writing header_str to anchor output file: %s", anchorFile.c_str());
    }
    anchorHeaderSize = header_str.size();
}

template<typename T>
std::string Tiles2MinibatchBase<T>::composeHeader() {
    assert (!outputBinary_);
        std::string jsonFile = outPref + ".json";
    std::ofstream jsonOut(jsonFile);
    if (!jsonOut) {
        error("Error opening json output file: %s", jsonFile.c_str());
    }
    nlohmann::json header;
    std::string header_str = "#x\ty";
    int32_t idx = 0;
    header["x"] = idx++;
    header["y"] = idx++;
    if (coordDim_ == MinibatchCoordDim::Dim3) {
        header_str += "\tz";
        header["z"] = idx++;
    }
    if (outputOriginalData_) {
        header_str += "\tfeature\tct";
        header["feature"] = idx++;
        header["ct"] = idx++;
    } else if (outputBackgroundProbExpand_) {
        header_str += "\tfeature";
        header["feature"] = idx++;
    }
    if (outputBackgroundProbExpand_) {
        header_str += "\tp0";
        header["p0"] = idx++;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header_str += "\tK" + std::to_string(i+1);
        header["K" + std::to_string(i+1)] = idx++;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header_str += "\tP" + std::to_string(i+1);
        header["P" + std::to_string(i+1)] = idx++;
    }
    if (useExtended_ && lineParserPtr) {
        for (const auto& v : lineParserPtr->name_ints) {
            header_str += "\t" + v;
            header[v] = idx++;
        }
        for (const auto& v : lineParserPtr->name_floats) {
            header_str += "\t" + v;
            header[v] = idx++;
        }
        for (const auto& v : lineParserPtr->name_strs) {
            header_str += "\t" + v;
            header[v] = idx++;
        }
    }
    if (outputBackgroundProbDense_) {
        header_str += "\tp0";
        header["p0"] = idx++;
    }
    jsonOut << std::setw(4) << header << std::endl;
    jsonOut.close();
    header_str += "\n";
    return header_str;
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseOneTileStandard(TileData<T>& tileData, TileKey tile) {
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
        RecordT<T> rec;
        int32_t idx = lineParserPtr->parse<T>(rec, line);
        if (idx < -1) {
            error("Error parsing line: %s", line.c_str());
        }
        if (idx == -1 || idx >= M_) {
            continue;
        }
        tileData.pts.push_back(rec);
        std::vector<uint32_t> bufferidx;
        if (pt2buffer(bufferidx, rec.x, rec.y, tile) == 1) {
            tileData.idxinternal.push_back(npt);
        }
        for (const auto& key : bufferidx) {
            tileData.buffers[key].push_back(rec);
        }
        npt++;
    }
    // write buffered records to temporary files
    for (const auto& entry : tileData.buffers) {
        auto buffer = getBoundaryBuffer(entry.first);
        buffer->addRecords(entry.second);
    }
    tileData.buffers.clear();
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseOneTileExtended(TileData<T>& tileData, TileKey tile) {
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
        RecordExtendedT<T> recExt;
        int32_t idx = lineParserPtr->parse(recExt, line);
        if (idx < -1) {
            error("Error parsing line: %s", line.c_str());
        }
        if (idx == -1 || idx >= M_) {
            continue;
        }
        tileData.extPts.push_back(recExt);
        std::vector<uint32_t> bufferidx;
        if (pt2buffer(bufferidx, recExt.recBase.x, recExt.recBase.y, tile) == 1) {
            tileData.idxinternal.push_back(npt);
        }
        for (const auto& key : bufferidx) {
            tileData.extBuffers[key].push_back(recExt);
        }
        npt++;
    }
    for (const auto& entry : tileData.extBuffers) {
        auto buffer = getBoundaryBuffer(entry.first);
        buffer->addRecordsExtended(entry.second, schema_, recordSize_);
    }
    tileData.extBuffers.clear();
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryFile(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
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
        RecordT<T> rec;
        ifs.read(reinterpret_cast<char*>(&rec), sizeof(RecordT<T>));
        if (ifs.gcount() != sizeof(RecordT<T>)) break;
        tileData.pts.push_back(rec);
        if (isInternalToBuffer(rec.x, rec.y, bufferPtr->key)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryFileExtended(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
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
        RecordExtendedT<T> r;
        // a) base part
        std::memcpy(&r.recBase, ptr, sizeof(r.recBase));
        // b) each extra
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
                    // trim trailing NULs
                    auto pos = s.find('\0');
                    if (pos!=std::string::npos) s.resize(pos);
                    r.strvals.push_back(s);
                } break;
            }
        }
        tileData.extPts.push_back(std::move(r));
        if (isInternalToBuffer(r.recBase.x, r.recBase.y, bufferPtr->key)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryStandard(TileData<T>& tileData, InMemoryStorageStandard<T>* memStore, uint32_t bufferKey) {
    tileData.clear();
    bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    // Directly copy the data from the in-memory vector.
    tileData.pts = std::move(memStore->data);
    // Mark internal points (for output)
    int npt = 0;
    for(const auto& rec : tileData.pts) {
        if (isInternalToBuffer(rec.x, rec.y, bufferKey)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryExtended(TileData<T>& tileData, InMemoryStorageExtended<T>* memStore, uint32_t bufferKey) {
    tileData.clear();
    bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    tileData.extPts = std::move(memStore->dataExtended);
    int npt = 0;
    for(const auto& rec : tileData.extPts) {
        if (isInternalToBuffer(rec.recBase.x, rec.recBase.y, bufferKey)) {
            tileData.idxinternal.push_back(npt);
        }
        npt++;
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryStandardWrapper(TileData<T>& tileData, IBoundaryStorage* storage, uint32_t bufferKey) {
    if (!storage) {
        tileData.clear();
        bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
        return 0;
    }
    auto* memStore = dynamic_cast<InMemoryStorageStandard<T>*>(storage);
    if (!memStore) {
        error("%s: expected standard in-memory storage", __func__);
    }
    return parseBoundaryMemoryStandard(tileData, memStore, bufferKey);
}

template<typename T>
int32_t Tiles2MinibatchBase<T>::parseBoundaryMemoryExtendedWrapper(TileData<T>& tileData, IBoundaryStorage* storage, uint32_t bufferKey) {
    if (!storage) {
        tileData.clear();
        bufferId2bound(bufferKey, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
        return 0;
    }
    auto* memStore = dynamic_cast<InMemoryStorageExtended<T>*>(storage);
    if (!memStore) {
        error("%s: expected extended in-memory storage", __func__);
    }
    return parseBoundaryMemoryExtended(tileData, memStore, bufferKey);
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatAnchorResult(const std::vector<AnchorPoint>& anchors, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, float xmin, float xmax, float ymin, float ymax) {
    size_t nrows = std::min((size_t) topVals.rows(), anchors.size());
    if (topVals.rows() != nrows || topIds.rows() != nrows || anchors.size() != nrows) {
        error("%s: size mismatch: topVals.rows()=%d, topIds.rows()=%d, anchors.size()=%d",
            __func__, topVals.rows(), topIds.rows(), anchors.size());
    }
    ResultBuf result(ticket, xmin, xmax, ymin, ymax);
    char buf[512];
    for (size_t i = 0; i < nrows; ++i) {
        if (anchors[i].x < xmin + r || anchors[i].x >= xmax - r ||
            anchors[i].y < ymin + r || anchors[i].y >= ymax - r) {
            continue; // only write internal anchors
        }
        int len = 0;
        if (coordDim_ == MinibatchCoordDim::Dim3) {
            len = std::snprintf(
                buf, sizeof(buf),
                "%.*f\t%.*f\t%.*f",
                floatCoordDigits,
                anchors[i].x,
                floatCoordDigits,
                anchors[i].y,
                floatCoordDigits,
                anchors[i].z
            );
        } else {
            len = std::snprintf(
                buf, sizeof(buf),
                "%.*f\t%.*f",
                floatCoordDigits,
                anchors[i].x,
                floatCoordDigits,
                anchors[i].y
            );
        }
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len,
                sizeof(buf) - len,
                "\t%d",
                topIds(i, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing anchor output line", __func__);
            }
            len += n;
        }
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len,
                sizeof(buf) - len,
                "\t%.*e",
                probDigits,
                topVals(i, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing anchor output line", __func__);
            }
            len += n;
        }
        buf[len++] = '\n';
        result.outputLines.emplace_back(buf, len);
    }
    result.npts = result.outputLines.size();
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultWithOriginalData(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket,
std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    if (outputBackgroundProbExpand_) {
        assert(phi0 != nullptr && phi0->size() == size_t(topVals.rows()));
    }
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    int32_t nrows = topVals.rows();
    char buf[65536];
    for (size_t i = 0; i < tileData.idxinternal.size(); ++i) {
        int32_t idxorg = tileData.idxinternal[i]; // index in the original data
        int32_t idx = tileData.orgpts2pixel[idxorg]; // index in the pixel minibatch
        if (idx < 0 || idx >= nrows) {
            continue;
        }
        const RecordT<T>* recPtr;
        if (useExtended_) {
            recPtr = &tileData.extPts[idxorg].recBase;
        } else {
            recPtr = &tileData.pts[idxorg];
        }
        const RecordT<T>& rec = *recPtr;
        int len = 0;
        if constexpr (std::is_same_v<T, int32_t>) {
            len = std::snprintf(
                buf, sizeof(buf), "%d\t%d\t",
                rec.x, rec.y
            );
        } else {
            len = std::snprintf(
                buf, sizeof(buf), "%.*f\t%.*f\t",
                floatCoordDigits, rec.x,
                floatCoordDigits, rec.y
            );
        }
        len += std::snprintf(
            buf + len, sizeof(buf) - len, "%s\t%d",
            featureNames[rec.idx].c_str(), rec.ct
        );
        // write background probability if available
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
        // write the top‑k IDs
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
        // write the top‑k probabilities in scientific form
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
        // write the extra fields
        if (useExtended_) {
            const RecordExtendedT<T>& recExt = tileData.extPts[idxorg];
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
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResult(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    return (this->*formatPixelFn_)(tileData, topVals, topIds, ticket, phi0);
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultStandard(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    assert(outputMode_ == MinibatchOutputMode::Standard);
    assert(!outputBinary_);
    assert(!outputBackgroundProbExpand_);
    if (outputBackgroundProbDense_) {
        assert(phi0 != nullptr && phi0->size() == size_t(topVals.rows()));
    }
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    size_t N = tileData.coords.size();
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
        int len = len = std::snprintf(
            buf, sizeof(buf), "%.*f\t%.*f",
            floatCoordDigits, tileData.coords[j].first *pixelResolution_,
            floatCoordDigits, tileData.coords[j].second*pixelResolution_
        );
        // write the top‑k IDs
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
        // write the top‑k probabilities in scientific form
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
        // build a string with gene1:p01,gene2:p02,...
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
                len -= 1; // remove trailing comma
            }
        }
        buf[len++] = '\n';
        result.outputLines.emplace_back(buf, len);
    }
    result.npts = result.outputLines.size();
    return result;
}

template<typename T>
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultWithBackground(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    assert(!outputBackgroundProbDense_);
    if (!phi0) {
        error("%s: background probabilities are missing", __func__);
    }
    if (outputBackgroundProbExpand_) {
        assert(phi0->size() == size_t(topVals.rows()));
    }
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    size_t N = tileData.coords.size();
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
            int len = len = std::snprintf(
                buf, sizeof(buf), "%.*f\t%.*f",
                floatCoordDigits, tileData.coords[j].first*pixelResolution_,
                floatCoordDigits, tileData.coords[j].second*pixelResolution_
            );
            // write feature name and background probability
            len += std::snprintf(
                buf + len, sizeof(buf) - len, "\t%s\t%.*f",
                featureNames[kv.first].c_str(),
                probDigits, kv.second
            );
            // write the top‑k IDs
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
            // write the top‑k probabilities in scientific form
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
typename Tiles2MinibatchBase<T>::ResultBuf Tiles2MinibatchBase<T>::formatPixelResultBinary(const TileData<T>& tileData, const MatrixXf& topVals, const MatrixXi& topIds, int ticket, std::vector<std::unordered_map<uint32_t, float>>* phi0) {
    (void)phi0;
    ResultBuf result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
    result.useObj = true;
    size_t N = tileData.coords.size();
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
        PixTopProbs<int32_t> rec(tileData.coords[j]);
        rec.ks.resize(topk_);
        rec.ps.resize(topk_);
        for (int32_t k = 0; k < topk_; ++k) {
            rec.ks[k] = topIds(j, k);
            rec.ps[k] = topVals(j, k);
        }
        result.outputObjs.emplace_back(std::move(rec));
    }
    result.npts = result.outputObjs.size();
    return result;
}
