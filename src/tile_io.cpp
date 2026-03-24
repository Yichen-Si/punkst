#include "tile_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>

namespace {

bool isLikelyTextIndexSample(const std::string& sample) {
    if (sample.empty()) {
        return false;
    }
    for (unsigned char ch : sample) {
        if (ch == '\0') {
            return false;
        }
        if (ch == '\n' || ch == '\r' || ch == '\t') {
            continue;
        }
        if (std::isprint(ch) || std::isspace(ch)) {
            continue;
        }
        return false;
    }
    return true;
}

LoadedTileIndexData loadBinaryIndexData(std::ifstream& in, const std::string& indexFile) {
    LoadedTileIndexData loaded;
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(&loaded.header), sizeof(loaded.header))) {
        error("%s: Error reading index file: %s", __func__, indexFile.c_str());
    }
    loaded.globalBox = Rectangle<float>(loaded.header.xmin, loaded.header.ymin,
                                        loaded.header.xmax, loaded.header.ymax);
    IndexEntryF idx;
    while (in.read(reinterpret_cast<char*>(&idx), sizeof(idx))) {
        loaded.entries.push_back(idx);
    }
    if (loaded.entries.empty()) {
        error("%s: No index entries loaded from %s", __func__, indexFile.c_str());
    }
    return loaded;
}

LoadedTileIndexData loadLegacyBinaryIndexData(std::ifstream& in, const std::string& indexFile) {
    LoadedTileIndexData loaded;
    loaded.isLegacyBinary = true;
    loaded.globalBox.reset();
    in.clear();
    in.seekg(0);
    IndexEntryF_legacy legacy;
    while (in.read(reinterpret_cast<char*>(&legacy), sizeof(legacy))) {
        IndexEntryF idx;
        idx.st = legacy.st;
        idx.ed = legacy.ed;
        idx.n = legacy.n;
        idx.xmin = static_cast<int32_t>(std::floor(legacy.xmin));
        idx.xmax = static_cast<int32_t>(std::ceil(legacy.xmax));
        idx.ymin = static_cast<int32_t>(std::floor(legacy.ymin));
        idx.ymax = static_cast<int32_t>(std::ceil(legacy.ymax));
        loaded.entries.push_back(idx);
        loaded.globalBox.extendToInclude(Rectangle<float>(legacy.xmin, legacy.ymin, legacy.xmax, legacy.ymax));
    }
    if (loaded.entries.empty()) {
        error("%s: No index entries loaded from %s", __func__, indexFile.c_str());
    }
    loaded.header.mode |= 0x8u;
    loaded.header.tileSize = 0;
    loaded.header.xmin = loaded.globalBox.xmin;
    loaded.header.xmax = loaded.globalBox.xmax;
    loaded.header.ymin = loaded.globalBox.ymin;
    loaded.header.ymax = loaded.globalBox.ymax;
    return loaded;
}

LoadedTileIndexData loadTextIndexData(const std::string& indexFile) {
    LoadedTileIndexData loaded;
    loaded.isTextIndex = true;
    loaded.globalBox.reset();
    std::ifstream in(indexFile, std::ios::binary);
    if (!in.is_open()) {
        error("%s: Error opening index file: %s", __func__, indexFile.c_str());
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            std::istringstream metaStream(line);
            std::string hashtag, key;
            metaStream >> hashtag >> key;
            if (key == "tilesize") {
                if (!(metaStream >> loaded.header.tileSize)) {
                    error("%s: Invalid tileSize in %s", __func__, indexFile.c_str());
                }
            }
            continue;
        }
        std::istringstream iss(line);
        IndexEntryF idx;
        if (!(iss >> idx.row >> idx.col >> idx.st >> idx.ed)) {
            error("%s: Malformed index line: %s", __func__, line.c_str());
        }
        if (loaded.header.tileSize > 0) {
            tile2bound(idx.row, idx.col, idx.xmin, idx.xmax, idx.ymin, idx.ymax, loaded.header.tileSize);
            loaded.globalBox.extendToInclude(Rectangle<float>(
                static_cast<float>(idx.xmin), static_cast<float>(idx.ymin),
                static_cast<float>(idx.xmax), static_cast<float>(idx.ymax)));
        }
        loaded.entries.push_back(idx);
    }
    if (loaded.entries.empty()) {
        error("%s: No index entries loaded from %s", __func__, indexFile.c_str());
    }
    loaded.header.xmin = loaded.globalBox.xmin;
    loaded.header.xmax = loaded.globalBox.xmax;
    loaded.header.ymin = loaded.globalBox.ymin;
    loaded.header.ymax = loaded.globalBox.ymax;
    return loaded;
}

} // namespace

LoadedTileIndexData loadTileIndexData(const std::string& indexFile) {
    std::ifstream in(indexFile, std::ios::binary);
    if (!in.is_open()) {
        error("Error opening index file: %s", indexFile.c_str());
    }

    uint64_t magic = 0;
    if (!in.read(reinterpret_cast<char*>(&magic), sizeof(magic))) {
        error("%s: Error reading index file: %s", __func__, indexFile.c_str());
    }
    if (magic == PUNKST_INDEX_MAGIC) {
        return loadBinaryIndexData(in, indexFile);
    }

    in.clear();
    in.seekg(0);
    std::string sample(128, '\0');
    in.read(sample.data(), static_cast<std::streamsize>(sample.size()));
    sample.resize(static_cast<size_t>(in.gcount()));
    if (isLikelyTextIndexSample(sample)) {
        return loadTextIndexData(indexFile);
    }
    return loadLegacyBinaryIndexData(in, indexFile);
}
