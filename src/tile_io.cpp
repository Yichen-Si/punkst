#include "tile_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <unordered_map>
#include <unordered_set>

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

std::string decodeFixedFeatureName(const char* buf, uint32_t width,
    const std::string& indexFile, uint32_t featureIdx) {
    size_t len = 0;
    while (len < width && buf[len] != '\0') {
        ++len;
    }
    std::string out(buf, buf + len);
    if (out.empty()) {
        error("%s: empty embedded feature name at index %u in %s",
            __func__, featureIdx, indexFile.c_str());
    }
    return out;
}

LoadedTileIndexData loadBinaryIndexData(std::ifstream& in, const std::string& indexFile) {
    LoadedTileIndexData loaded;
    in.seekg(0);
    if (!in.read(reinterpret_cast<char*>(&loaded.header), sizeof(loaded.header))) {
        error("%s: Error reading index file: %s", __func__, indexFile.c_str());
    }
    if ((loaded.header.mode & 0x40u) != 0u) {
        if (loaded.header.featureCount == 0 || loaded.header.featureNameSize == 0) {
            error("%s: feature-bearing index %s must store featureCount and featureNameSize",
                __func__, indexFile.c_str());
        }
        const uint64_t payloadBytes =
            static_cast<uint64_t>(loaded.header.featureCount) * loaded.header.featureNameSize;
        if (payloadBytes == 0 || payloadBytes > static_cast<uint64_t>(std::numeric_limits<size_t>::max())) {
            error("%s: invalid embedded feature dictionary payload size in %s",
                __func__, indexFile.c_str());
        }
        std::vector<char> payload(static_cast<size_t>(payloadBytes), '\0');
        if (!in.read(payload.data(), static_cast<std::streamsize>(payload.size()))) {
            error("%s: Error reading embedded feature dictionary from %s",
                __func__, indexFile.c_str());
        }
        loaded.featureNames.reserve(loaded.header.featureCount);
        std::unordered_set<std::string> seen;
        seen.reserve(loaded.header.featureCount);
        for (uint32_t i = 0; i < loaded.header.featureCount; ++i) {
            const char* rec = payload.data() + static_cast<size_t>(i) * loaded.header.featureNameSize;
            std::string featureName = decodeFixedFeatureName(rec, loaded.header.featureNameSize, indexFile, i);
            if (!seen.insert(featureName).second) {
                error("%s: duplicate embedded feature name '%s' in %s",
                    __func__, featureName.c_str(), indexFile.c_str());
            }
            loaded.featureNames.push_back(std::move(featureName));
        }
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

uint32_t computeFeatureNameSizeFixed(const std::vector<std::string>& featureNames) {
    if (featureNames.empty()) {
        error("%s: feature-bearing index requires at least one feature name", __func__);
    }
    size_t maxLen = 0;
    for (const auto& name : featureNames) {
        if (name.empty()) {
            error("%s: feature names must be non-empty", __func__);
        }
        maxLen = std::max(maxLen, name.size());
    }
    const size_t width = maxLen + 1;
    if (width > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        error("%s: feature name width exceeds uint32_t range", __func__);
    }
    return static_cast<uint32_t>(width);
}

void configureFeatureDictionaryHeader(IndexHeader& header,
    const std::vector<std::string>& featureNames, const char* funcName) {
    if ((header.mode & 0x40u) == 0u) {
        header.featureCount = 0;
        header.featureNameSize = 0;
        return;
    }
    if (featureNames.empty()) {
        error("%s: feature-bearing index requires embedded feature names", funcName);
    }
    if (featureNames.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max())) {
        error("%s: feature count exceeds uint32_t range", funcName);
    }
    header.featureCount = static_cast<uint32_t>(featureNames.size());
    header.featureNameSize = computeFeatureNameSizeFixed(featureNames);
}

bool writeFeatureDictionaryPayload(int fd, const IndexHeader& header,
    const std::vector<std::string>& featureNames) {
    if ((header.mode & 0x40u) == 0u) {
        return true;
    }
    if (header.featureCount != featureNames.size() || header.featureNameSize == 0) {
        error("%s: feature dictionary header metadata mismatch", __func__);
    }
    const size_t payloadBytes =
        static_cast<size_t>(header.featureCount) * static_cast<size_t>(header.featureNameSize);
    std::vector<char> payload(payloadBytes, '\0');
    for (size_t i = 0; i < featureNames.size(); ++i) {
        const auto& name = featureNames[i];
        if (name.size() >= header.featureNameSize) {
            error("%s: feature name '%s' exceeds fixed width %u",
                __func__, name.c_str(), header.featureNameSize);
        }
        std::memcpy(payload.data() + i * header.featureNameSize, name.data(), name.size());
    }
    return write_all(fd, payload.data(), payload.size());
}

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
