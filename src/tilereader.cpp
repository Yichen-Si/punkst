#include "tilereader.hpp"
#include "tileoperator.hpp"

/*
    lineParser
*/
void lineParser::init(size_t _ix, size_t _iy, size_t _iw,
                      const std::vector<int32_t>& _ivals,
                      const std::string& _dfile) {
    icol_x = _ix;
    icol_y = _iy;
    icol_feature = _iw;
    n_ct = _ivals.size();
    icol_ct.resize(n_ct);
    n_tokens = icol_feature;
    for (size_t i = 0; i < n_ct; ++i) {
        icol_ct[i] = _ivals[i];
        n_tokens = std::max(n_tokens, icol_ct[i]);
    }
    if (hasZ) {
        n_tokens = std::max(n_tokens, icol_z);
    }
    n_tokens += 1;
    if (_dfile.empty()) {
        isFeatureDict = false;
    } else {
        isFeatureDict = true;
        std::ifstream dictFile(_dfile);
        if (!dictFile) {
            error("Error opening feature dictionary file: %s", _dfile.c_str());
        }
        std::string line;
        uint32_t nfeature = 0;
        while (std::getline(dictFile, line)) {
            if (line.empty() || line[0] == '#') continue;
            size_t pos = line.find_first_of(" \t");
            if (pos != std::string::npos) {
                line = line.substr(0, pos);
            }
            if (line.empty()) continue;
            featureDict[line] = nfeature++;
        }
        if (featureDict.empty()) {
            error("Error reading feature dictionary file: %s", _dfile.c_str());
        }
        notice("Read %zu features from dictionary file", featureDict.size());
    }
}

void lineParser::setZ(size_t col) {
    icol_z = col;
    hasZ = true;
    n_tokens = std::max(n_tokens, icol_z + 1);
}

bool lineParser::addExtraInt(std::vector<std::string>& annoInts) {
    for (const auto& anno : annoInts) {
        uint32_t idx;
        std::vector<std::string> tokens;
        split(tokens, ":", anno);
        if (tokens.size() < 2 || !str2num<uint32_t>(tokens[0], idx)) {
            return false;
        }
        icol_ints.push_back(idx);
        name_ints.push_back(tokens[1]);
    }
    isExtended |= !icol_ints.empty();
    return true;
}

bool lineParser::addExtraFloat(std::vector<std::string>& annoFloats) {
    for (const auto& anno : annoFloats) {
        uint32_t idx;
        std::vector<std::string> tokens;
        split(tokens, ":", anno);
        if (tokens.size() < 2 || !str2num<uint32_t>(tokens[0], idx)) {
            return false;
        }
        icol_floats.push_back(idx);
        name_floats.push_back(tokens[1]);
    }
    isExtended |= !icol_floats.empty();
    return true;
}

bool lineParser::addExtraStr(std::vector<std::string>& annoStrs) {
    for (const auto& anno : annoStrs) {
        uint32_t idx, len;
        std::vector<std::string> tokens;
        split(tokens, ":", anno);
        if (tokens.size() < 3 || !str2num<uint32_t>(tokens[0], idx) ||
            !str2num<uint32_t>(tokens[2], len)) {
            return false;
        }
        icol_strs.push_back(idx);
        name_strs.push_back(tokens[1]);
        str_lens.push_back(len);
    }
    isExtended |= !icol_strs.empty();
    return true;
}

bool lineParser::checkExtraColumns() {
    if (icol_strs.size() != str_lens.size()) {
        return false;
    }
    if (name_ints.size() < icol_ints.size()) {
        uint32_t i = static_cast<uint32_t>(name_ints.size());
        while (i < icol_ints.size()) {
            name_ints.push_back("int_" + std::to_string(i));
            ++i;
        }
    }
    if (name_floats.size() < icol_floats.size()) {
        uint32_t i = static_cast<uint32_t>(name_floats.size());
        while (i < icol_floats.size()) {
            name_floats.push_back("float_" + std::to_string(i));
            ++i;
        }
    }
    if (name_strs.size() < icol_strs.size()) {
        uint32_t i = static_cast<uint32_t>(name_strs.size());
        while (i < icol_strs.size()) {
            name_strs.push_back("str_" + std::to_string(i));
            ++i;
        }
    }
    return true;
}

void lineParser::setFeatureDict(const std::unordered_map<std::string, uint32_t>& dict) {
    featureDict = dict;
    isFeatureDict = true;
}

void lineParser::setFeatureDict(const std::vector<std::string>& featureList) {
    featureDict.clear();
    for (size_t i = 0; i < featureList.size(); ++i) {
        featureDict[featureList[i]] = static_cast<uint32_t>(i);
    }
    isFeatureDict = true;
}

int32_t lineParser::getFeatureList(std::vector<std::string>& featureList) {
    if (!isFeatureDict) {
        return -1;
    }
    featureList.resize(featureDict.size());
    for (const auto& pair : featureDict) {
        featureList[pair.second] = pair.first;
    }
    return static_cast<int32_t>(featureDict.size());
}

int32_t lineParser::parse(PixelValues& pixel, std::string& line, bool checkBounds) {
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < static_cast<size_t>(n_tokens)) {
        return -1;
    }
    pixel.x = std::stod(tokens[icol_x]);
    pixel.y = std::stod(tokens[icol_y]);
    if (checkBounds) {
        bool valid = false;
        for (const auto& rect : rects) {
            if (rect.contains(pixel.x, pixel.y)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return 0;
        }
    }
    if (isFeatureDict) {
        auto it = featureDict.find(tokens[icol_feature]);
        if (it == featureDict.end()) {
            return 0;
        }
        pixel.feature = it->second;
    } else {
        if (!str2uint32(tokens[icol_feature], pixel.feature)) {
            return -1;
        }
    }
    pixel.intvals.resize(n_ct);
    int32_t totVal = 0;
    for (size_t i = 0; i < n_ct; ++i) {
        if (!str2int32(tokens[icol_ct[i]], pixel.intvals[i])) {
            return -1;
        }
        totVal += pixel.intvals[i];
    }
    return totVal;
}

int32_t lineParser::parse(PixelValues3D& pixel, std::string& line, bool checkBounds) {
    if (!hasZ) {
        return -1;
    }
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < static_cast<size_t>(n_tokens)) {
        return -1;
    }
    pixel.x = std::stod(tokens[icol_x]);
    pixel.y = std::stod(tokens[icol_y]);
    pixel.z = std::stod(tokens[icol_z]);
    if (checkBounds) {
        bool valid = false;
        for (const auto& rect : rects) {
            if (rect.contains(pixel.x, pixel.y)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return 0;
        }
    }
    if (isFeatureDict) {
        auto it = featureDict.find(tokens[icol_feature]);
        if (it == featureDict.end()) {
            return 0;
        }
        pixel.feature = it->second;
    } else {
        if (!str2uint32(tokens[icol_feature], pixel.feature)) {
            return -1;
        }
    }
    pixel.intvals.resize(n_ct);
    int32_t totVal = 0;
    for (size_t i = 0; i < n_ct; ++i) {
        if (!str2int32(tokens[icol_ct[i]], pixel.intvals[i])) {
            return -1;
        }
        totVal += pixel.intvals[i];
    }
    return totVal;
}

int32_t lineParser::readWeights(const std::string& weightFile, double defaultWeight, int32_t nFeatures) {
    std::ifstream inWeight(weightFile);
    if (!inWeight) {
        error("Error opening weights file: %s", weightFile.c_str());
    }
    weighted = true;
    int32_t novlp = 0;
    std::string line;
    if (isFeatureDict) {
        weights.resize(featureDict.size());
        std::fill(weights.begin(), weights.end(), defaultWeight);

        while (std::getline(inWeight, line)) {
            std::istringstream iss(line);
            std::string feature;
            double weight;
            if (!(iss >> feature >> weight)) {
                error("Error reading weights file at line: %s", line.c_str());
            }
            auto it = featureDict.find(feature);
            if (it != featureDict.end()) {
                weights[it->second] = weight;
                novlp++;
            }
        }
        return novlp;
    }
    if (nFeatures > 0) {
        weights.resize(nFeatures);
        std::fill(weights.begin(), weights.end(), defaultWeight);
        while (std::getline(inWeight, line)) {
            std::istringstream iss(line);
            uint32_t idx;
            double weight;
            if (!(iss >> idx >> weight)) {
                error("Error reading weights file at line: %s", line.c_str());
            }
            if (idx >= static_cast<uint32_t>(nFeatures)) {
                warning("Weight file feature out of range: %s", line.c_str());
                continue;
            }
            weights[idx] = weight;
            novlp++;
        }
        return novlp;
    }

    int32_t max_idx = 0;
    std::unordered_map<uint32_t, double> weights_map;
    while (std::getline(inWeight, line)) {
        std::istringstream iss(line);
        uint32_t idx;
        double weight;
        if (!(iss >> idx >> weight)) {
            error("Error reading weights file at line: %s", line.c_str());
        }
        if (idx >= static_cast<uint32_t>(max_idx)) {
            max_idx = static_cast<int32_t>(idx);
        }
        weights_map[idx] = weight;
    }
    novlp = static_cast<int32_t>(weights_map.size());
    weights.resize(max_idx + 1);
    std::fill(weights.begin(), weights.end(), defaultWeight);
    for (const auto& pair : weights_map) {
        weights[pair.first] = pair.second;
    }
    return novlp;
}

template<typename T>
int32_t lineParserUnival::parse(RecordT<T>& rec, std::string& line, bool checkBounds) const {
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < n_tokens) {
        return -2;
    }
    if (!str2num<T>(tokens[icol_x], rec.x) || !str2num<T>(tokens[icol_y], rec.y)) {
        return -2;
    }
    if (checkBounds) {
        bool valid = false;
        for (const auto& rect : rects) {
            if (rect.contains(rec.x, rec.y)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return -1;
        }
    }
    if (isFeatureDict) {
        auto it = featureDict.find(tokens[icol_feature]);
        if (it == featureDict.end()) {
            return -1;
        }
        rec.idx = it->second;
    } else {
        rec.idx = std::stoul(tokens[icol_feature]);
    }
    rec.ct = std::stoi(tokens[icol_val]);
    return rec.idx;
}

template<typename T>
int32_t lineParserUnival::parse( RecordExtendedT<T>& rec, std::string &line ) const {
    int32_t base_idx = parse(rec.recBase, line);
    if (base_idx < 0) return base_idx;
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    // 3) extra ints
    rec.intvals.resize(icol_ints.size());
    for (size_t i = 0; i < icol_ints.size(); ++i) {
        if (!str2num<int32_t>(tokens[icol_ints[i]], rec.intvals[i])) {
            return -2;
        }
    }
    // 4) extra floats
    rec.floatvals.resize(icol_floats.size());
    for (size_t i = 0; i < icol_floats.size(); ++i) {
        if (!str2num<float>(tokens[icol_floats[i]], rec.floatvals[i])) {
            return -2;
        }
    }
    // 5) extra strings (pad/truncate to str_len)
    rec.strvals.resize(icol_strs.size());
    for (size_t i = 0; i < icol_strs.size(); ++i) {
        auto &s = tokens[icol_strs[i]];
        if (s.size() > str_lens[i]) s.resize(str_lens[i]);
        rec.strvals[i] = s;
    }
    return base_idx;
}

template<typename T>
int32_t lineParserUnival::parse(RecordT3D<T>& rec, std::string& line, bool checkBounds) const {
    if (!hasZ) {
        return -2;
    }
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    if (tokens.size() < n_tokens) {
        return -2;
    }
    if (!str2num<T>(tokens[icol_x], rec.x) || !str2num<T>(tokens[icol_y], rec.y) || !str2num<T>(tokens[icol_z], rec.z)) {
        return -2;
    }
    if (checkBounds) {
        bool valid = false;
        for (const auto& rect : rects) {
            if (rect.contains(rec.x, rec.y)) {
                valid = true;
                break;
            }
        }
        if (!valid) {
            return -1;
        }
    }
    if (isFeatureDict) {
        auto it = featureDict.find(tokens[icol_feature]);
        if (it == featureDict.end()) {
            return -1;
        }
        rec.idx = it->second;
    } else {
        rec.idx = std::stoul(tokens[icol_feature]);
    }
    rec.ct = std::stoi(tokens[icol_val]);
    return rec.idx;
}

template<typename T>
int32_t lineParserUnival::parse( RecordExtendedT3D<T>& rec, std::string &line ) const {
    int32_t base_idx = parse(rec.recBase, line);
    if (base_idx < 0) return base_idx;
    std::vector<std::string> tokens;
    split(tokens, "\t", line);
    rec.intvals.resize(icol_ints.size());
    for (size_t i = 0; i < icol_ints.size(); ++i) {
        if (!str2num<int32_t>(tokens[icol_ints[i]], rec.intvals[i])) {
            return -2;
        }
    }
    rec.floatvals.resize(icol_floats.size());
    for (size_t i = 0; i < icol_floats.size(); ++i) {
        if (!str2num<float>(tokens[icol_floats[i]], rec.floatvals[i])) {
            return -2;
        }
    }
    rec.strvals.resize(icol_strs.size());
    for (size_t i = 0; i < icol_strs.size(); ++i) {
        auto &s = tokens[icol_strs[i]];
        if (s.size() > str_lens[i]) s.resize(str_lens[i]);
        rec.strvals[i] = s;
    }
    return base_idx;
}

// explicit template instantiation
template int32_t lineParserUnival::parse<float>(RecordT<float>& rec, std::string& line, bool checkBounds) const;
template int32_t lineParserUnival::parse<int32_t>(RecordT<int32_t>& rec, std::string& line, bool checkBounds) const;
template int32_t lineParserUnival::parse<float>( RecordExtendedT<float>& rec, std::string &line ) const;
template int32_t lineParserUnival::parse<int32_t>( RecordExtendedT<int32_t>& rec, std::string &line ) const;
template int32_t lineParserUnival::parse<float>(RecordT3D<float>& rec, std::string& line, bool checkBounds) const;
template int32_t lineParserUnival::parse<int32_t>(RecordT3D<int32_t>& rec, std::string& line, bool checkBounds) const;
template int32_t lineParserUnival::parse<float>( RecordExtendedT3D<float>& rec, std::string &line ) const;
template int32_t lineParserUnival::parse<int32_t>( RecordExtendedT3D<int32_t>& rec, std::string &line ) const;

/*
    TileReaderBase
*/
void TileReaderBase::assignLoadedIndex(const LoadedTileIndexData& loaded) {
    tile_map_.clear();
    blocks_.clear();
    load_ok_ = false;
    minrow = INT32_MAX;
    mincol = INT32_MAX;
    maxrow = INT32_MIN;
    maxcol = INT32_MIN;

    tileSize_ = loaded.header.tileSize;
    globalBox_ = loaded.globalBox;
    recordSize_ = static_cast<size_t>(loaded.header.recordSize);
    featureNames_ = loaded.featureNames;

    const bool filter = !rects.empty();
    for (const auto& entry : loaded.entries) {
        TileInfo info(entry, true);
        if (filter) {
            bool contained = false;
            if (!blockIntersectsRects(entry, rects, &contained)) {
                continue;
            }
            info.contained = contained;
        }
        blocks_.push_back(info);
        if (entry.row != std::numeric_limits<int32_t>::lowest() &&
            entry.col != std::numeric_limits<int32_t>::lowest()) {
            TileKey key{entry.row, entry.col};
            tile_map_.emplace(key, info);
            if (entry.row < minrow) minrow = entry.row;
            if (entry.col < mincol) mincol = entry.col;
            if (entry.row > maxrow) maxrow = entry.row;
            if (entry.col > maxcol) maxcol = entry.col;
        }
    }
    nTiles = blocks_.size();
    load_ok_ = true;
    notice("Read index with %zu blocks", nTiles);
}

/*
    TileReader
*/
std::unique_ptr<BoundedReadline> TileReader::get_tile_iterator(int tileRow, int tileCol) const {
    TileKey key {tileRow, tileCol};
    auto it = tile_map_.find(key);
    if (it == tile_map_.end()) {
        warning("%s: Tile (%d, %d) not found in index", __FUNCTION__, tileRow, tileCol);
        return nullptr;
    }
    const TileInfo &info = it->second;
    return std::make_unique<BoundedReadline>(inputFile_, info.idx.st, info.idx.ed);
}

std::unique_ptr<BoundedReadline> TileReader::get_block_iterator(const TileInfo& block) const {
    return std::make_unique<BoundedReadline>(inputFile_, block.idx.st, block.idx.ed);
}

void TileReader::loadIndex(const std::string &indexFilename) {
    assignLoadedIndex(loadTileIndexData(indexFilename));
}

/*
    Other
*/
std::vector<dataset> parseSampleList(const std::string& sampleList, const std::string* outPref) {
    std::ifstream rf(sampleList);
    if (!rf) {
        error("Error opening sample list file: %s", sampleList.c_str());
    }
    std::vector<dataset> datasets;
    std::string line;
    while (std::getline(rf, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < 3) {
            error("Invalid line in sample list: %s", line.c_str());
        }
        dataset ds{tokens[0], tokens[1], tokens[2]};
        if (tokens.size() > 3) {
            ds.outPref = tokens[3];
            if (tokens.size() > 4) {
                ds.anchorFile = tokens[4];
            }
        } else {
            size_t pos = ds.inTsv.find_last_of("/\\");
            if (pos != std::string::npos) {
                ds.outPref = ds.inTsv.substr(0, pos+1) + ds.sampleId;
            } else {
                ds.outPref = ds.sampleId;
            }
            if (outPref && !(*outPref).empty()) {
                ds.outPref += "." + *outPref;
            }
        }
        datasets.push_back(ds);
    }
    return datasets;
}
