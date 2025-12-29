#include "tilereader.hpp"
#include "tileoperator.hpp"

/*
    lineParser
*/
void lineParser::init(size_t _ix, size_t _iy, size_t _iz,
                      const std::vector<int32_t>& _ivals,
                      const std::string& _dfile) {
    icol_x = _ix;
    icol_y = _iy;
    icol_feature = _iz;
    n_ct = static_cast<int32_t>(_ivals.size());
    icol_ct.resize(n_ct);
    n_tokens = static_cast<int32_t>(icol_feature);
    for (int32_t i = 0; i < n_ct; ++i) {
        icol_ct[i] = _ivals[i];
        n_tokens = std::max(n_tokens, icol_ct[i]);
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
    for (int32_t i = 0; i < n_ct; ++i) {
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

// explicit template instantiation
template int32_t lineParserUnival::parse<float>(RecordT<float>& rec, std::string& line, bool checkBounds) const;
template int32_t lineParserUnival::parse<int32_t>(RecordT<int32_t>& rec, std::string& line, bool checkBounds) const;
template int32_t lineParserUnival::parse<float>( RecordExtendedT<float>& rec, std::string &line ) const;
template int32_t lineParserUnival::parse<int32_t>( RecordExtendedT<int32_t>& rec, std::string &line ) const;

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
    return std::make_unique<BoundedReadline>(tsvFilename, info.idx.st, info.idx.ed);
}

void TileReader::loadIndex(const std::string &indexFilename) {
    std::ifstream indexFile(indexFilename, std::ios::binary);
    if (!indexFile.is_open()) {
        throw std::runtime_error("Unable to open index file: " + indexFilename);
    }
    if (loadIndexBinary(indexFilename)) {return;}
    // Fallback to text format
    loadIndexText(indexFilename);
}

bool TileReader::loadIndexBinary(const std::string &indexFilename) {
    std::ifstream indexFile(indexFilename, std::ios::binary);
    if (!indexFile.is_open()) {
        error("%s: Unable to open index file %s", __func__, indexFilename.c_str());
    }

    uint64_t magic = 0;
    if (!indexFile.read(reinterpret_cast<char*>(&magic), sizeof(magic)) ||
        magic != PUNKST_INDEX_MAGIC) {
        return false;
    }

    IndexHeader header;
    indexFile.seekg(0);
    indexFile.read(reinterpret_cast<char*>(&header), sizeof(header));
    tileSize = header.tileSize;
    if (tileSize <= 0) error("%s: invalid tileSize", __func__);
    globalBox_ = Rectangle<float>(header.xmin, header.ymin, header.xmax, header.ymax);

    IndexEntryF entry;
    std::unordered_map<TileKey,bool,TileKeyHash> tileMap;
    bool filter = !rects.empty();
    if (filter) {
        getTilesInBounds(rects, tileMap);
    }
    while (indexFile.read(reinterpret_cast<char*>(&entry), sizeof(entry))) {
        TileKey key{entry.row, entry.col};
        TileInfo info(entry, true);
        if (filter) {
            auto it = tileMap.find(key);
            if (it == tileMap.end()) {
                continue;
            }
            info.contained = it->second;
        }
        tile_map_.emplace(key, info);
        if (entry.row < minrow) minrow = entry.row;
        if (entry.col < mincol) mincol = entry.col;
        if (entry.row > maxrow) maxrow = entry.row;
        if (entry.col > maxcol) maxcol = entry.col;
    }
    nTiles = tile_map_.size();
    indexFile.close();
    notice("Read %zu tiles from binary index file", nTiles);
    return true;
}

bool TileReader::loadIndexText(const std::string &indexFilename) {
    std::ifstream indexFile(indexFilename, std::ios::binary);
    if (!indexFile.is_open()) {
        error("%s: Unable to open index file %s", __func__, indexFilename.c_str());
    }

    indexFile.clear();
    indexFile.seekg(0);
    std::string line;
    if (!std::getline(indexFile, line) || line.empty()) {
        error("%s: Index file appears to be empty", __func__);
    }

    std::unordered_map<TileKey,bool,TileKeyHash> tileMap;
    bool filter = !rects.empty();
    if (filter) getTilesInBounds(rects, tileMap);
    while (std::getline(indexFile, line)) {
        if (line.empty()) {continue;}
        while (line[0] == '#') { // Parse metadata lines
            std::istringstream metaStream(line);
            std::string hashtag, key;
            metaStream >> hashtag >> key;
            if (key == "tilesize") {
                if (!(metaStream >> tileSize)) error("%s: Invalid tileSize", __func__);
            }
            if (!std::getline(indexFile, line)) {
                if (line.empty() && indexFile.eof()) break;
                error("%s: Index file is empty/truncated", __func__);
            }
        }
        std::istringstream iss(line);
        int row, col;
        uint64_t start, end;
        if (!(iss >> row >> col >> start >> end)) {
            error("%s: Malformed index line: %s", __func__, line.c_str());
        }
        TileInfo info(start, end, true);
        TileKey key{row, col};
        if (filter) {
            auto it = tileMap.find(key);
            if (it == tileMap.end()) {continue;}
            info.contained = it->second;
        }
        tile_map_.emplace(key, info);
        if (row < minrow) minrow = row;
        if (col < mincol) mincol = col;
        if (row > maxrow) maxrow = row;
        if (col > maxcol) maxcol = col;
        globalBox_.extendToInclude(Rectangle<int32_t>(
            col * tileSize, row * tileSize,
            (col + 1) * tileSize, (row + 1) * tileSize));
    }
    if (tileSize <= 0) {error("%s: Cannot identify tileSize", __func__);}

    nTiles = tile_map_.size();
    indexFile.close();
    notice("Read %zu tiles from index file", nTiles);
    return true;
}

/*
    BoundedBinaryTileIterator
*/
bool BoundedBinaryTileIterator::next(BoundedBinaryTileIterator::Record &record) {
    if (!file || file->tellg() >= endOffset) {
        return false;
    }
    std::vector<char> buffer(recordSize);
    file->read(buffer.data(), recordSize);
    if (file->gcount() != static_cast<std::streamsize>(recordSize)) {
        return false;
    }
    size_t offset = 0;
    record.resize(numCat, numInt, numFloat);
    // Decode x and y coordinates (doubles)
    std::memcpy(&record.x, buffer.data() + offset, sizeof(double));
    offset += sizeof(double);
    std::memcpy(&record.y, buffer.data() + offset, sizeof(double));
    offset += sizeof(double);
    // Decode categorical fields (each stored as int32_t)
    for (int32_t i = 0; i < numCat; ++i) {
        std::memcpy(&record.catFields[i], buffer.data() + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
    }
    // Decode integer fields (each int32_t)
    for (int i = 0; i < numInt; ++i) {
        std::memcpy(&record.intFields[i], buffer.data() + offset, sizeof(int32_t));
        offset += sizeof(int32_t);
    }
    // Decode float fields (each float)
    for (int i = 0; i < numFloat; ++i) {
        std::memcpy(&record.floatFields, buffer.data() + offset, sizeof(float));
        offset += sizeof(float);
    }
    return true;
}

/*
    BinaryTileReader
*/
std::unique_ptr<BoundedBinaryTileIterator> BinaryTileReader::get_tile_iterator(int tileRow, int tileCol) const {
    TileKey key {tileRow, tileCol};
    auto it = tile_map_.find(key);
    if (it == tile_map_.end()) {
        return nullptr; // Tile not found
    }
    const TileInfo &info = it->second;
    return std::make_unique<BoundedBinaryTileIterator>(tsvFilename, info.idx.st, info.idx.ed, recordSize, numCat, numInt, numFloat);
}

void BinaryTileReader::loadIndex(const std::string &indexFilename) {
        std::ifstream indexFile(indexFilename);
        if (!indexFile.is_open()) {
            throw std::runtime_error("Unable to open index file: " + indexFilename);
        }
        std::string line;
        // First metadata line: "# tilesize<TAB><tileSize>"
        while (std::getline(indexFile, line)) {
            if (line[0] != '#') {
                break;
            }
            std::istringstream metaStream(line);
            std::string hashtag, key;
            metaStream >> hashtag >> key;
            if (key == "tilesize") {
                if (!(metaStream >> tileSize)) {
                    throw std::runtime_error("Failed to read tile size from metadata");
                }
            } else if (key == "recordsize") {
                if (!(metaStream >> recordSize)) {
                    throw std::runtime_error("Failed to read record size from metadata");
                }
            } else if (key == "nvalues") {
                if (!(metaStream >> numCat >> numInt >> numFloat)) {
                    throw std::runtime_error("Failed to read number of values from metadata");
                }
            } else if (key == "dictionaries") {
                if (!(metaStream >> dictStartOffset >> dictEndOffset)) {
                    throw std::runtime_error("Failed to read dictionary offsets from metadata");
                }
            }
        }
        // Read remaining lines for tile index.
        while (!line.empty()) {
            std::istringstream iss(line);
            int row, col;
            std::uint64_t start, end;
            int count;
            if (!(iss >> row >> col >> start >> end >> count)) {
                throw std::runtime_error("Malformed index line: " + line);
            }
            tile_map_.emplace(TileKey{row, col}, TileInfo(start, end));
            if (row < minrow) minrow = row;
            if (col < mincol) mincol = col;
            if (row > maxrow) maxrow = row;
            if (col > maxcol) maxcol = col;
            if (!std::getline(indexFile, line)) {
                break;  // End of file
            }
        }
        nTiles = tile_map_.size();
        indexFile.close();
    }

void BinaryTileReader::readDictionaries() {
        std::ifstream infile(tsvFilename, std::ios::binary);
        if (!infile.is_open()) {
            throw std::runtime_error("Unable to open binary file: " + tsvFilename);
        }
        infile.seekg(dictStartOffset);
        uint32_t numDict = 0;
        infile.read(reinterpret_cast<char*>(&numDict), sizeof(numDict));
        catDictionaries.resize(numDict);
        for (uint32_t i = 0; i < numDict; ++i) {
            uint32_t dictSize = 0;
            infile.read(reinterpret_cast<char*>(&dictSize), sizeof(dictSize));
            uint64_t totalLength = 0;
            infile.read(reinterpret_cast<char*>(&totalLength), sizeof(totalLength));
            std::string dictStr(totalLength, '\0');
            infile.read(&dictStr[0], totalLength);
            // Split the dictionary string using '\t' as delimiter.
            catDictionaries[i].resize(dictSize);
            auto& entries = catDictionaries[i];
            size_t pos = 0, j = 0;
            while (true) {
                size_t tabPos = dictStr.find('\t', pos);
                if (tabPos == std::string::npos) {
                    entries[j] = dictStr.substr(pos);
                    break;
                }
                entries[j++] = dictStr.substr(pos, tabPos - pos);
                pos = tabPos + 1;
            }
            if (entries.size() != dictSize) {
                throw std::runtime_error("Dictionary entries count mismatch");
            }
        }
        infile.close();
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
