#include "tiles2minibatch.hpp"
#include <fcntl.h>

int32_t Tiles2MinibatchBase::loadAnchors(const std::string& anchorFile) {
    std::ifstream inFile(anchorFile);
    if (!inFile) {
        error("Error opening anchors file: %s", anchorFile.c_str());
    }
    std::string line;
    std::vector<std::string> tokens;
    int32_t nAnchors = 0;
    while (std::getline(inFile, line)) {
        if (line.empty()) continue;
        split(tokens, "\t", line);
        if (tokens.size() < 2) {
            error("Error reading anchors file at line: %s", line.c_str());
        }
        float x, y;
        bool valid = str2float(tokens[0], x) && str2float(tokens[1], y);
        if (!valid) {
            error("Error reading anchors file at line: %s", line.c_str());
        }
        TileKey tile;
        if (!tileReader.pt2tile(x, y, tile)) {
            continue;
        }
        fixedAnchorForTile[tile].emplace_back(std::vector<float>{x, y});
        std::vector<uint32_t> bufferidx;
        int32_t ret = pt2buffer(bufferidx, x, y, tile);
        (void)ret;
        for (const auto& key : bufferidx) {
            fixedAnchorForBoundary[key].emplace_back(std::vector<float>{x, y});
        }
        nAnchors++;
    }
    inFile.close();
    if (fixedAnchorForTile.empty()) {
        error("No anchors fall in the region of the input pixel data, please make sure the anchors are in the same coordinate system as the input data");
    }
    return nAnchors;
}

template<typename T>
int32_t Tiles2MinibatchBase::parseOneTile(
    TileData<T>& tileData, TileKey tile,
    lineParserUnival& lineParser, int32_t M_, bool useExtended_,
    const std::vector<FieldDef>& schema_, size_t recordSize_) {
    std::unique_ptr<BoundedReadline> iter;
    try {
        iter = tileReader.get_tile_iterator(tile.row, tile.col);
    } catch (const std::exception &e) {
        warning("%s", e.what());
        return -1;
    }
    tileData.clear();
    tile2bound(tile, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);

    std::string line;
    int32_t npt = 0;
    if (useExtended_) {
        while (iter->next(line)) {
            RecordExtendedT<T> recExt;
            int32_t idx = lineParser.parse(recExt, line);
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
    } else {
        while (iter->next(line)) {
            RecordT<T> rec;
            int32_t idx = lineParser.parse<T>(rec, line);
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
    }
    return tileData.idxinternal.size();
}

template<typename T>
int32_t Tiles2MinibatchBase::parseBoundaryFile(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr) {
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
int32_t Tiles2MinibatchBase::parseBoundaryFileExtended(TileData<T>& tileData, std::shared_ptr<BoundaryBuffer> bufferPtr,
    const std::vector<FieldDef>& schema_, size_t recordSize_) {
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
int32_t Tiles2MinibatchBase::parseBoundaryMemoryStandard(TileData<T>& tileData, InMemoryStorageStandard<T>* memStore, uint32_t bufferKey) {
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
int32_t Tiles2MinibatchBase::parseBoundaryMemoryExtended(TileData<T>& tileData, InMemoryStorageExtended<T>* memStore, uint32_t bufferKey) {
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
Tiles2MinibatchBase::ProcessedResult Tiles2MinibatchBase::formatPixelResultWithOriginalData(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket) {
    ProcessedResult result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
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
                buf, sizeof(buf),
                "%d\t%d\t",
                rec.x,
                rec.y
            );
        } else {
            len = std::snprintf(
                buf, sizeof(buf),
                "%.*f\t%.*f\t",
                floatCoordDigits,
                rec.x,
                floatCoordDigits,
                rec.y
            );
        }
        len += std::snprintf(
            buf + len, sizeof(buf) - len,
            "%s\t%d",
            featureNames[rec.idx].c_str(),
            rec.ct
        );
        // write the top‑k IDs
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len,
                sizeof(buf) - len,
                "\t%d",
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
                buf + len,
                sizeof(buf) - len,
                "\t%.*e",
                probDigits,
                topVals(idx, k)
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
Tiles2MinibatchBase::ProcessedResult Tiles2MinibatchBase::formatPixelResult(const TileData<T>& tileData, const MatrixXf& topVals, const Eigen::MatrixXi& topIds, int ticket) {
    ProcessedResult result(ticket, tileData.xmin, tileData.xmax, tileData.ymin, tileData.ymax);
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
        int len = len = std::snprintf(
            buf, sizeof(buf),
            "%.*f\t%.*f",
            floatCoordDigits,
            tileData.coords[j].first,
            floatCoordDigits,
            tileData.coords[j].second
        );
        // write the top‑k IDs
        for (int32_t k = 0; k < topk_; ++k) {
            int n = std::snprintf(
                buf + len,
                sizeof(buf) - len,
                "\t%d",
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
                buf + len,
                sizeof(buf) - len,
                "\t%.*e",
                probDigits,
                topVals(j, k)
            );
            if (n < 0 || n >= int(sizeof(buf) - len)) {
                error("%s: error writing output line", __func__);
            }
            len += n;
        }
        buf[len++] = '\n';
        result.outputLines.emplace_back(buf, len);
    }
    result.npts = result.outputLines.size();
    return result;
}

void Tiles2MinibatchBase::writeHeaderToJson() {
    std::string jsonFile = outPref + ".json";
    std::ofstream jsonOut(jsonFile);
    if (!jsonOut) {
        error("Error opening json output file: %s", jsonFile.c_str());
    }
    nlohmann::json header;
    header["x"] = 0;
    header["y"] = 1;
    int32_t idx = 2;
    if (outputOriginalData) {
        header["feature"] = 2;
        header["ct"] = 3;
        idx = 4;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header["K" + std::to_string(i+1)] = idx++;
    }
    for (int32_t i = 0; i < topk_; ++i) {
        header["P" + std::to_string(i+1)] = idx++;
    }
    if (useExtended_ && lineParserPtr) {
        for (const auto& v : lineParserPtr->name_ints) {
            header[v] = idx++;
        }
        for (const auto& v : lineParserPtr->name_floats) {
            header[v] = idx++;
        }
        for (const auto& v : lineParserPtr->name_strs) {
            header[v] = idx++;
        }
    }
    jsonOut << std::setw(4) << header << std::endl;
    jsonOut.close();
}

void Tiles2MinibatchBase::writerWorker() {
    // A priority queue to buffer out-of-order results
    std::priority_queue<ProcessedResult, std::vector<ProcessedResult>, std::greater<ProcessedResult>> outOfOrderBuffer;
    int nextTicketToWrite = 0;
    ProcessedResult result;
    // Loop until the queue is marked as done and is empty
    while (resultQueue.pop(result)) {
        outOfOrderBuffer.push(std::move(result));
        // Write all results that are now ready in sequential order
        while (!outOfOrderBuffer.empty() &&
               (!useTicketSystem ||
                outOfOrderBuffer.top().ticket == nextTicketToWrite)) {
            const auto& readyToWrite = outOfOrderBuffer.top();
            if (readyToWrite.npts > 0) {
                size_t st = outputSize;
                size_t totalLen = 0;
                for (const auto& line : readyToWrite.outputLines) {
                    if (!write_all(fdMain, line.data(), line.size())) {
                        error("Error writing to main output file");
                    }
                    totalLen += line.size();
                }
                size_t ed = st + totalLen;
                if (st == 0) ed += headerSize;
                IndexEntry<float> e{st, ed, readyToWrite.npts,
                    readyToWrite.xmin, readyToWrite.xmax,
                    readyToWrite.ymin, readyToWrite.ymax};
                if (!write_all(fdIndex, &e, sizeof(e))) {
                    error("Error writing to index output file");
                }
                outputSize = ed;
            }
            outOfOrderBuffer.pop();
            nextTicketToWrite++;
        }
    }
}

void Tiles2MinibatchBase::setupOutput() {
    #if !defined(_WIN32)
        // ensure includes present
    #endif
    std::string outputFile = outPref + ".tsv";
    fdMain = ::open(outputFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdMain < 0) {
        error("Error opening main output file: %s", outputFile.c_str());
    }
    // compose header
    std::string header;
    if (outputOriginalData) header = "#x\ty\tfeature\tct";
    else header = "#x\ty";
    for (int32_t i = 0; i < topk_; ++i) header += "\tK" + std::to_string(i+1);
    for (int32_t i = 0; i < topk_; ++i) header += "\tP" + std::to_string(i+1);
    if (useExtended_ && lineParserPtr) {
        for (const auto &v : lineParserPtr->name_ints) header += "\t" + v;
        for (const auto &v : lineParserPtr->name_floats) header += "\t" + v;
        for (const auto &v : lineParserPtr->name_strs) header += "\t" + v;
    }
    header += "\n";
    if (!write_all(fdMain, header.data(), header.size())) {
        error("Error writing header to main output file: %s", outputFile.c_str());
    }
    headerSize = header.size();
    std::string indexFile = outPref + ".index";
    fdIndex = ::open(indexFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
    if (fdIndex < 0) {
        error("Error opening index output file: %s", indexFile.c_str());
    }
}

void Tiles2MinibatchBase::closeOutput() {
    if (fdMain >= 0) { ::close(fdMain); fdMain = -1; }
    if (fdIndex >= 0) { ::close(fdIndex); fdIndex = -1; }
}

// Explicit instantiations for base templated helpers
template int32_t Tiles2MinibatchBase::parseOneTile<int32_t>(TileData<int32_t>&, TileKey, lineParserUnival&, int32_t, bool, const std::vector<FieldDef>&, size_t);
template int32_t Tiles2MinibatchBase::parseOneTile<float>(TileData<float>&, TileKey, lineParserUnival&, int32_t, bool, const std::vector<FieldDef>&, size_t);

template int32_t Tiles2MinibatchBase::parseBoundaryFile<int32_t>(TileData<int32_t>&, std::shared_ptr<BoundaryBuffer>);
template int32_t Tiles2MinibatchBase::parseBoundaryFile<float>(TileData<float>&, std::shared_ptr<BoundaryBuffer>);

template int32_t Tiles2MinibatchBase::parseBoundaryFileExtended<int32_t>(TileData<int32_t>&, std::shared_ptr<BoundaryBuffer>, const std::vector<FieldDef>&, size_t);
template int32_t Tiles2MinibatchBase::parseBoundaryFileExtended<float>(TileData<float>&, std::shared_ptr<BoundaryBuffer>, const std::vector<FieldDef>&, size_t);

template int32_t Tiles2MinibatchBase::parseBoundaryMemoryStandard<int32_t>(TileData<int32_t>&, InMemoryStorageStandard<int32_t>*, uint32_t);
template int32_t Tiles2MinibatchBase::parseBoundaryMemoryStandard<float>(TileData<float>&, InMemoryStorageStandard<float>*, uint32_t);

template int32_t Tiles2MinibatchBase::parseBoundaryMemoryExtended<int32_t>(TileData<int32_t>&, InMemoryStorageExtended<int32_t>*, uint32_t);
template int32_t Tiles2MinibatchBase::parseBoundaryMemoryExtended<float>(TileData<float>&, InMemoryStorageExtended<float>*, uint32_t);

template Tiles2MinibatchBase::ProcessedResult Tiles2MinibatchBase::formatPixelResultWithOriginalData<int32_t>(const TileData<int32_t>&, const MatrixXf&, const Eigen::MatrixXi&, int);
template Tiles2MinibatchBase::ProcessedResult Tiles2MinibatchBase::formatPixelResultWithOriginalData<float>(const TileData<float>&, const MatrixXf&, const Eigen::MatrixXi&, int);

template Tiles2MinibatchBase::ProcessedResult Tiles2MinibatchBase::formatPixelResult<int32_t>(const TileData<int32_t>&, const MatrixXf&, const Eigen::MatrixXi&, int);
template Tiles2MinibatchBase::ProcessedResult Tiles2MinibatchBase::formatPixelResult<float>(const TileData<float>&, const MatrixXf&, const Eigen::MatrixXi&, int);
