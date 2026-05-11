#pragma once

#include "utils.h"
#include "img_utils.hpp"
#include "utils_sys.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <map>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <unordered_map>
#include <memory>
#include <array>
#include <fcntl.h>
#include <unistd.h>
#include "zlib.h"
#include "threads.hpp"
#include "nanoflann.hpp"
#include "nanoflann_utils.h"
#include "tileoperator.hpp"

class Pts2Tiles {
public:
    Pts2Tiles(int32_t nthreads,
        const std::string& inFile, std::string& tmpDir,
        const std::string& outPref, int32_t tileSize,
        int32_t icol_x, int32_t icol_y, int32_t icol_z = -1, int32_t icol_g = -1, std::vector<int32_t> icol_ints = {}, int32_t nskip = 0,
        bool skipLastLineIsHeader = false,
        bool streamingMode = false,
        int32_t tileBuffer = 1000, int32_t batchSize = 10000,
        double scale_x = 1, double scale_y = 1, double scale_z = 1,
        int digits=2, char inputDelimiter = '\t',
        bool tileOpFactorTsv = false,
        std::vector<int32_t> includeCols = {},
        std::vector<int32_t> excludeCols = {}) :
        nThreads_(nthreads),
        inFile_(inFile), tmpDir_(tmpDir),
        outPref_(outPref), tileSize_(tileSize),
        icol_x_(icol_x), icol_y_(icol_y), icol_z_(icol_z), icol_feature_(icol_g),
        icol_ints_(icol_ints), nskip_(nskip), skipLastLineIsHeader_(skipLastLineIsHeader),
        streamingMode_(streamingMode),
        tileBuffer_(tileBuffer), batchSize_(batchSize),
        scale_x_(scale_x), scale_y_(scale_y), scale_z_(scale_z),
        digits_(digits), inputDelimiter_(1, inputDelimiter),
        tileOpFactorTsv_(tileOpFactorTsv),
        includeCols_(std::move(includeCols)), excludeCols_(std::move(excludeCols))
    {
        if (!includeCols_.empty() && !excludeCols_.empty()) {
            error("--include-cols and --exclude-cols are mutually exclusive");
        }
        if (tileOpFactorTsv_ && (!includeCols_.empty() || !excludeCols_.empty())) {
            error("--include-cols/--exclude-cols cannot be combined with --tile-op-factor-tsv");
        }
        filterColumns_ = !includeCols_.empty() || !excludeCols_.empty();
        ntokens_ = std::max(icol_x_, icol_y_);
        if (icol_z_ >= 0) {
            ntokens_ = std::max(ntokens_, icol_z_);
        }
        if (icol_feature_ >= 0) {
            ntokens_ = std::max(ntokens_, icol_feature_);
            for (const auto& icol : icol_ints_) {
                ntokens_ = std::max(ntokens_, icol);
            }
        }
        scaling_ = std::abs(scale_x_ - 1) > 1e-8
            || std::abs(scale_y_ - 1) > 1e-8
            || (icol_z_ >= 0 && std::abs(scale_z_ - 1) > 1e-8);
        appendDummyCount_ = !tileOpFactorTsv_ && icol_ints_.empty();
        rewriteLine_ = scaling_ || inputDelimiter_ != "\t" || appendDummyCount_ || filterColumns_;
        ntokens_ += 1;
        notice("Created temporary directory: %s", tmpDir_.path.string().c_str());
    }
    virtual ~Pts2Tiles() {
        if (gz_) {
            gzclose(gz_);
        }
        if (inPtr_ && inPtr_ != &std::cin) {
            delete inPtr_;
        }
    }

    const std::unordered_map<TileKey, uint64_t, TileKeyHash>& getGlobalTiles() const {
        return globalTiles_;
    }

    bool run() {
        if (!streamingMode_) {
            collectSkippedLinesFromFile();
            collectInitialCommentLinesFromFile();
            discoverTileOpFactorHeaderFromFile();
            validateTileOpFactorTsvConfig();
            if (!launchWorkerThreads()) {
                error("Error launching worker threads_");
            }
        } else {
            openInput();
            validateTileOpFactorTsvConfig();
            for (int i = 0; i < nThreads_; ++i) {
                threads_.emplace_back(&Pts2Tiles::streamingWorker, this, i);
            }
        }
        notice("Launched %d worker threads", nThreads_);
        if (!joinWorkerThreads()) {
            error("Error joining worker threads");
        }
        notice("Merging temporary files and writing index");
        if (!mergeAndWriteIndex()) {
            error("Error merging temporary files and writing index");
        }
        if (!writeAuxiliaryFiles()) {
            warning("Error writing auxiliary files");
            return false;
        }
        return true;
    }

protected:
    int32_t nThreads_;
    std::string inFile_, outPref_;
    ScopedTempDir tmpDir_;
    int32_t tileBuffer_, batchSize_;
    int32_t tileSize_;
    int32_t icol_x_, icol_y_, icol_z_, icol_feature_;
    std::vector<int32_t> icol_ints_;
    int32_t ntokens_;
    int32_t nskip_, nskipped_ = 0;
    bool skipLastLineIsHeader_;
    double scale_x_, scale_y_, scale_z_;
    bool scaling_;
    bool appendDummyCount_;
    bool rewriteLine_;
    bool tileOpFactorTsv_;
    bool filterColumns_;
    int32_t tileOpTopK_ = 0;
    int32_t digits_;
    std::string inputDelimiter_;
    std::vector<int32_t> includeCols_, excludeCols_;
    Rectangle<float> globalBox_;
    bool hasGlobalZRange_ = false;
    float globalZMin_ = 0, globalZMax_ = 0;
    std::string metaLines_;

    std::unordered_map<std::string, std::vector<int32_t>> featureCounts_;
    std::map<int64_t, uint64_t> zHist_;
    std::mutex minmaxMutex_;

    std::unordered_map<TileKey, uint64_t, TileKeyHash> globalTiles_;
    std::mutex globalTilesMutex_;

    std::vector<std::thread> threads_;
    bool streamingMode_;
    gzFile         gz_    = nullptr;
    std::istream*  inPtr_ = nullptr;
    std::mutex readMutex_;
    bool hasPendingLine_ = false;
    std::string pendingLine_;

    struct PtRecord {
        float x, y;
        float z = 0;
        std::string feature;
        std::vector<int32_t> vals = {1};
    };

    static std::string normalizeHeaderKey(std::string key) {
        key = std::string(strip_str(key));
        std::transform(key.begin(), key.end(), key.begin(),
            [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
        return key;
    }

    struct KPColumnName {
        bool ok = false;
        bool isK = false;
        uint32_t idx = 0;
    };

    static KPColumnName parseKPColumnName(const std::string& key) {
        KPColumnName out;
        if (key.size() < 2) {
            return out;
        }
        size_t pos = key.size();
        while (pos > 0 && std::isdigit(static_cast<unsigned char>(key[pos - 1]))) {
            --pos;
        }
        if (pos == key.size() || pos == 0) {
            return out;
        }
        const char kp = static_cast<char>(std::toupper(static_cast<unsigned char>(key[pos - 1])));
        if (kp != 'K' && kp != 'P') {
            return out;
        }
        if (pos > 1) {
            const std::string prefix = key.substr(0, pos - 1);
            if (!prefix.empty() && prefix.back() != '_') {
                return out;
            }
        }
        uint32_t parsedIdx = 0;
        if (!str2uint32(key.substr(pos), parsedIdx) || parsedIdx == 0) {
            return out;
        }
        out.ok = true;
        out.isK = (kp == 'K');
        out.idx = parsedIdx;
        return out;
    }

    bool isTileOpFactorHeaderCandidate(const std::string& line) const {
        size_t nhash = 0;
        while (nhash < line.size() && line[nhash] == '#') {
            ++nhash;
        }
        std::string header = std::string(strip_str(line.substr(nhash)));
        std::vector<std::string> tokens;
        tokenizeLine(header, tokens);
        bool hasX = false, hasY = false, hasK1 = false, hasP1 = false;
        for (const auto& token : tokens) {
            const std::string key = normalizeHeaderKey(token);
            if (key == "x") {
                hasX = true;
            } else if (key == "y") {
                hasY = true;
            }
            const KPColumnName kp = parseKPColumnName(key);
            if (kp.ok && kp.idx == 1) {
                if (kp.isK) {
                    hasK1 = true;
                } else {
                    hasP1 = true;
                }
            }
        }
        return hasX && hasY && hasK1 && hasP1;
    }

    void parseTileOpFactorHeader(std::vector<std::string>& tokens) {
        int32_t headerX = -1, headerY = -1, headerZ = -1;
        std::unordered_map<uint32_t, uint32_t> kcols;
        std::unordered_map<uint32_t, uint32_t> pcols;
        for (uint32_t i = 0; i < tokens.size(); ++i) {
            std::string key = normalizeHeaderKey(tokens[i]);
            tokens[i] = key;
            if (key == "x") {
                headerX = static_cast<int32_t>(i);
            } else if (key == "y") {
                headerY = static_cast<int32_t>(i);
            } else if (key == "z") {
                headerZ = static_cast<int32_t>(i);
            }
            const KPColumnName kp = parseKPColumnName(key);
            if (!kp.ok) {
                continue;
            }
            auto& target = kp.isK ? kcols : pcols;
            if (!target.emplace(kp.idx, i).second) {
                error("%s: duplicate %c%u column in TileOperator factor TSV header",
                    __func__, kp.isK ? 'K' : 'P', kp.idx);
            }
        }
        if (icol_x_ < 0) {
            icol_x_ = headerX;
        }
        if (icol_y_ < 0) {
            icol_y_ = headerY;
        }
        if (icol_z_ < 0 && headerZ >= 0) {
            icol_z_ = headerZ;
        }
        if (icol_x_ < 0 || icol_y_ < 0) {
            error("%s: --tile-op-factor-tsv requires x and y header columns", __func__);
        }
        int32_t topK = 0;
        for (uint32_t idx = 1; ; ++idx) {
            const bool hasK = kcols.find(idx) != kcols.end();
            const bool hasP = pcols.find(idx) != pcols.end();
            if (!hasK && !hasP) {
                break;
            }
            if (!hasK || !hasP) {
                error("%s: TileOperator factor TSV header must include both K%u and P%u",
                    __func__, idx, idx);
            }
            topK++;
        }
        if (topK <= 0) {
            error("%s: --tile-op-factor-tsv requires at least one K/P pair", __func__);
        }
        tileOpTopK_ = topK;
        scaling_ = std::abs(scale_x_ - 1) > 1e-8
            || std::abs(scale_y_ - 1) > 1e-8
            || (icol_z_ >= 0 && std::abs(scale_z_ - 1) > 1e-8);
        rewriteLine_ = scaling_ || inputDelimiter_ != "\t" || appendDummyCount_ || filterColumns_;
        ntokens_ = std::max(icol_x_, icol_y_);
        if (icol_z_ >= 0) {
            ntokens_ = std::max(ntokens_, icol_z_);
        }
        for (int32_t idx = 1; idx <= tileOpTopK_; ++idx) {
            ntokens_ = std::max(ntokens_, static_cast<int32_t>(kcols[static_cast<uint32_t>(idx)]));
            ntokens_ = std::max(ntokens_, static_cast<int32_t>(pcols[static_cast<uint32_t>(idx)]));
        }
        ntokens_ += 1;
    }

    void parseTileOpFactorHeaderLine(const std::string& line) {
        size_t nhash = 0;
        while (nhash < line.size() && line[nhash] == '#') {
            ++nhash;
        }
        std::string header = std::string(strip_str(line.substr(nhash)));
        std::vector<std::string> tokens;
        tokenizeLine(header, tokens);
        parseTileOpFactorHeader(tokens);
    }

    bool shouldKeepColumn(size_t idx, size_t ncols) const {
        if (!filterColumns_) {
            return true;
        }
        bool keep = true;
        if (!includeCols_.empty()) {
            keep = false;
            for (const auto& col : includeCols_) {
                if (col >= 0 && static_cast<size_t>(col) == idx) {
                    keep = true;
                    break;
                }
            }
        } else {
            for (const auto& col : excludeCols_) {
                if (col >= 0 && static_cast<size_t>(col) == idx) {
                    keep = false;
                    break;
                }
            }
        }
        if (icol_x_ >= 0 && static_cast<size_t>(icol_x_) == idx) keep = true;
        if (icol_y_ >= 0 && static_cast<size_t>(icol_y_) == idx) keep = true;
        if (icol_z_ >= 0 && static_cast<size_t>(icol_z_) == idx) keep = true;
        if (icol_feature_ >= 0 && static_cast<size_t>(icol_feature_) == idx) keep = true;
        for (const auto& col : icol_ints_) {
            if (col >= 0 && static_cast<size_t>(col) == idx) {
                keep = true;
                break;
            }
        }
        if (appendDummyCount_ && idx + 1 == ncols) {
            keep = true;
        }
        return keep;
    }

    std::string filterLineTokens(const std::vector<std::string>& tokens) const {
        if (!filterColumns_) {
            return join(tokens, "\t");
        }
        std::vector<std::string> out;
        out.reserve(tokens.size());
        for (size_t i = 0; i < tokens.size(); ++i) {
            if (shouldKeepColumn(i, tokens.size())) {
                out.push_back(tokens[i]);
            }
        }
        return join(out, "\t");
    }

    void normalizeExplicitHeaderColumns(std::vector<std::string>& tokens) const {
        if (icol_x_ >= 0 && static_cast<size_t>(icol_x_) < tokens.size()) {
            tokens[icol_x_] = "X";
        }
        if (icol_y_ >= 0 && static_cast<size_t>(icol_y_) < tokens.size()) {
            tokens[icol_y_] = "Y";
        }
        if (icol_z_ >= 0 && static_cast<size_t>(icol_z_) < tokens.size()) {
            tokens[icol_z_] = "Z";
        }
        if (icol_feature_ >= 0 && static_cast<size_t>(icol_feature_) < tokens.size()) {
            tokens[icol_feature_] = "Feature";
        }
    }

    std::string normalizeSkippedLine(const std::string& line, bool isHeader) {
        size_t nhash = 0;
        while (nhash < line.size() && line[nhash] == '#') {
            ++nhash;
        }
        if (isHeader) {
            std::string header = std::string(strip_str(line.substr(nhash)));
            std::vector<std::string> tokens;
            tokenizeLine(header, tokens);
            if (tileOpFactorTsv_) {
                parseTileOpFactorHeader(tokens);
            }
            normalizeExplicitHeaderColumns(tokens);
            if (appendDummyCount_) {
                tokens.push_back("count");
            }
            header = filterLineTokens(tokens);
            return "#" + header;
        }
        return line;
    }

    void appendSkippedLineAsMeta(const std::string& line, bool isHeader) {
        metaLines_ += normalizeSkippedLine(line, isHeader);
        metaLines_ += "\n";
    }

    void appendCommentLineAsMeta(const std::string& line) {
        const bool isTileOpHeader = tileOpFactorTsv_ && isTileOpFactorHeaderCandidate(line);
        appendSkippedLineAsMeta(line, isTileOpHeader);
    }

    void discoverTileOpFactorHeaderFromFile() {
        if (!tileOpFactorTsv_ || tileOpTopK_ > 0 || streamingMode_) {
            return;
        }
        if (inFile_ == "-" || ends_with(inFile_, ".gz")) {
            return;
        }
        std::ifstream inFile(inFile_);
        if (!inFile) {
            error("Error opening input file: %s", inFile_.c_str());
        }
        std::string line;
        while (std::getline(inFile, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            if (line[0] == '#') {
                if (isTileOpFactorHeaderCandidate(line)) {
                    parseTileOpFactorHeaderLine(line);
                    return;
                }
                continue;
            }
            break;
        }
    }

    void requireTileOpFactorHeaderFromStream() {
        if (!tileOpFactorTsv_ || tileOpTopK_ > 0) {
            return;
        }
        std::string line;
        while (readNextInputLine(line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0] != '#') {
                error("%s: --tile-op-factor-tsv requires a header line starting with '#', or --skip-last-is-header for a non-comment header",
                    __func__);
            }
            appendCommentLineAsMeta(line);
            if (tileOpTopK_ > 0) {
                return;
            }
        }
    }

    void validateTileOpFactorTsvConfig() const {
        if (!tileOpFactorTsv_) {
            return;
        }
        if (icol_x_ < 0 || icol_y_ < 0 || tileOpTopK_ <= 0) {
            error("%s: --tile-op-factor-tsv requires a parsed header with x, y, and K/P columns",
                __func__);
        }
    }

    bool readNextInputLine(std::string& line) {
        if (hasPendingLine_) {
            line = std::move(pendingLine_);
            pendingLine_.clear();
            hasPendingLine_ = false;
            return true;
        }
        if (gz_) {
            char buf[1<<16];
            if (!gzgets(gz_, buf, sizeof(buf))) {
                return false;
            }
            size_t len = strlen(buf);
            if (len > 0 && buf[len - 1] == '\n') {
                buf[--len] = '\0';
                if (len > 0 && buf[len - 1] == '\r') {
                    buf[--len] = '\0';
                }
                line.assign(buf, len);
                return true;
            }
            line.assign(buf, len);
            while (true) {
                if (!gzgets(gz_, buf, sizeof(buf))) {
                    break;
                }
                const size_t chunkLen = strlen(buf);
                const bool gotNL = chunkLen > 0 && buf[chunkLen - 1] == '\n';
                line.append(buf, gotNL ? chunkLen - 1 : chunkLen);
                if (gotNL) {
                    break;
                }
            }
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            return true;
        }
        if (!std::getline(*inPtr_, line)) {
            return false;
        }
        if (!line.empty() && line.back() == '\r') {
            line.pop_back();
        }
        return true;
    }

    void collectSkippedLinesFromFile() {
        if (nskip_ <= 0) return;
        std::ifstream inFile(inFile_);
        if (!inFile) {
            error("Error opening input file: %s", inFile_.c_str());
        }
        std::string line;
        for (int32_t i = 0; i < nskip_ && std::getline(inFile, line); ++i) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (skipLastLineIsHeader_ && i + 1 == nskip_) {
                appendSkippedLineAsMeta(line, true);
            }
        }
    }

    void collectInitialCommentLinesFromFile() {
        if (inFile_ == "-" || ends_with(inFile_, ".gz")) {
            return;
        }
        std::ifstream inFile(inFile_);
        if (!inFile) {
            error("Error opening input file: %s", inFile_.c_str());
        }
        std::string line;
        for (int32_t i = 0; i < nskip_ && std::getline(inFile, line); ++i) {}
        while (std::getline(inFile, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }
            if (line[0] == '#') {
                appendCommentLineAsMeta(line);
                continue;
            }
            break;
        }
    }

    // Open input stream from stdin or a gzipped file
    void openInput() {
        if (inFile_ == "-") {
            inPtr_ = &std::cin;
        }
        else if (ends_with(inFile_, ".gz")) {
            gz_ = gzopen(inFile_.c_str(), "rb");
            if (gz_ == Z_NULL) {
                error("Error opening gzipped input file: %s", inFile_.c_str());
            }
        } else {
            warning("%s: the input is not stdin or gzipped file but the streaming mode is used, assuming it is a plain text delimited file", __FUNCTION__);
            inPtr_ = new std::ifstream(inFile_);
            if (!inPtr_ || !static_cast<std::ifstream*>(inPtr_)->is_open()) {
                error("Error opening input file: %s", inFile_.c_str());
            }
        }
        std::string line;
        while (nskipped_ < nskip_ && readNextInputLine(line)) {
            ++nskipped_;
            if (skipLastLineIsHeader_ && nskipped_ == nskip_) {
                appendSkippedLineAsMeta(line, true);
            }
        }
        while (readNextInputLine(line)) {
            if (line.empty()) {
                continue;
            }
            if (line[0] == '#') {
                appendCommentLineAsMeta(line);
                continue;
            }
            pendingLine_ = std::move(line);
            hasPendingLine_ = true;
            break;
        }
        requireTileOpFactorHeaderFromStream();
    }

    void tokenizeLine(const std::string& line, std::vector<std::string>& tokens) const {
        split(tokens, inputDelimiter_, line);
    }

    void rewriteCoordinates(std::vector<std::string>& tokens, std::string& line, float& x, float& y, float* z = nullptr) const {
        if (appendDummyCount_) {
            tokens.push_back("1");
        }
        if (scaling_) {
            x *= scale_x_;
            y *= scale_y_;
            tokens[icol_x_] = fp_to_string(x, digits_);
            tokens[icol_y_] = fp_to_string(y, digits_);
            if (z != nullptr && icol_z_ >= 0) {
                *z *= scale_z_;
                tokens[icol_z_] = fp_to_string(*z, digits_);
            }
        }
        if (rewriteLine_) {
            line = filterLineTokens(tokens);
        }
    }

    std::filesystem::path getTmpFilename(const TileKey& tile, int threadId) const {
        return tmpDir_.path / (std::to_string(tile.row) + "_" + std::to_string(tile.col)
            + "_" + std::to_string(threadId) + ".tsv");
    }

    // Parse a line, extract coordinates and return the tile key
    virtual TileKey parse(std::string& line, PtRecord& pt) {
        std::vector<std::string> tokens;
        tokenizeLine(line, tokens);
        if (tokens.size() < ntokens_) {
            error("Error parsing line: %s", line.c_str());
        }
        pt.x = std::stof(tokens[icol_x_]);
        pt.y = std::stof(tokens[icol_y_]);
        if (icol_z_ >= 0) {
            pt.z = std::stof(tokens[icol_z_]);
        }
        rewriteCoordinates(tokens, line, pt.x, pt.y, icol_z_ >= 0 ? &pt.z : nullptr);
        if (icol_feature_ >= 0) {
            pt.feature = tokens[icol_feature_];
            if (icol_ints_.size() > 0) {
                pt.vals.resize(icol_ints_.size());
                for (size_t i = 0; i < icol_ints_.size(); ++i) {
                    if (!str2int32(tokens[icol_ints_[i]], pt.vals[i])) {
                        error("Error parsing the %d-th token to integer (%s) at line %s", icol_ints_[i], tokens[icol_ints_[i]].c_str(), line.c_str());
                    }
                }
            }
        }
        TileKey tile;
        tile.row = static_cast<int32_t>(std::floor(pt.y / tileSize_));
        tile.col = static_cast<int32_t>(std::floor(pt.x / tileSize_));
        return tile;
    }
    virtual TileKey parse(std::string& line, float& x, float& y) {
        std::vector<std::string> tokens;
        tokenizeLine(line, tokens);
        if (tokens.size() < ntokens_) {
            error("Error parsing line: %s", line.c_str());
        }
        x = std::stof(tokens[icol_x_]);
        y = std::stof(tokens[icol_y_]);
        rewriteCoordinates(tokens, line, x, y);
        TileKey tile;
        tile.row = static_cast<int32_t>(std::floor(y / tileSize_));
        tile.col = static_cast<int32_t>(std::floor(x / tileSize_));
        return tile;
    }

    // Process one line: parse, assign, update stats, buffer and flush
    void consumeLine(int threadId, std::string &line,
            std::unordered_map<TileKey, std::vector<std::string>, TileKeyHash> &buffers,
            std::unordered_map<TileKey, Rectangle<float>, TileKeyHash> &tileBoxes,
            Rectangle<float>& localBox,
            std::unordered_map<std::string,std::vector<int32_t>> &localCounts,
            bool& hasLocalZRange,
            float& localZMin,
            float& localZMax,
            std::unordered_map<int64_t, uint64_t>& localZHist) {
        PtRecord pt;
        TileKey tile = parse(line, pt);

        // feature counts
        if (icol_feature_ >= 0) {
            auto &v = localCounts[pt.feature];
            if (v.empty()) v = pt.vals;
            else for (size_t i = 0; i < pt.vals.size(); ++i)
                    v[i] += pt.vals[i];
        }
        // update min/max
        localBox.extendToInclude(pt.x, pt.y);
        tileBoxes[tile].extendToInclude(pt.x, pt.y);
        if (icol_z_ >= 0) {
            if (!hasLocalZRange) {
                hasLocalZRange = true;
                localZMin = pt.z;
                localZMax = pt.z;
            } else {
                if (pt.z < localZMin) localZMin = pt.z;
                if (pt.z > localZMax) localZMax = pt.z;
            }
            ++localZHist[static_cast<int64_t>(std::floor(pt.z))];
        }
        // buffer + flush
        auto &buf = buffers[tile];
        buf.push_back(line);
        if (buf.size() >= static_cast<size_t>(tileBuffer_))
            flushBuffer(threadId, tile, buf);
    }

    // Write buffered lines to a temporary file defined by (threadId, tile)
    void flushBuffer(int threadId, const TileKey& tile,
                     std::vector<std::string> &buf) {
        {
            std::lock_guard lk(globalTilesMutex_);
            globalTiles_[tile] += buf.size();
        }
        auto fn = getTmpFilename(tile, threadId);
        std::ofstream out(fn, std::ios::app);
        for (auto &l : buf) out << l << "\n";
        out.close();
        buf.clear();
    }
    void flushAll(int threadId,
            std::unordered_map<TileKey, std::vector<std::string>, TileKeyHash> &buffers) {
        for (auto &p : buffers)
        if (!p.second.empty())
            flushBuffer(threadId, p.first, p.second);
    }

    // Non-stream mode - Read and process a chunk of the input file
    virtual void worker(int threadId, std::streampos start, std::streampos end) {
        std::unordered_map<TileKey, std::vector<std::string>, TileKeyHash> buffers;
        std::unordered_map<TileKey, Rectangle<float>, TileKeyHash> localTileMinMax;
        std::unordered_map<std::string,std::vector<int32_t>> localCounts;
        std::unordered_map<int64_t, uint64_t> localZHist;
        Rectangle<float> localMinMax;
        bool hasLocalZRange = false;
        float localZMin = 0, localZMax = 0;

        std::ifstream file(inFile_);
        if (!file) {
            error("Thread %d: Error opening file input file", threadId);
        }
        file.seekg(start);
        std::string line;

        while (file.tellg() < end && std::getline(file, line)) {
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) continue;
            if (line[0] == '#') {
                continue;
            }
            consumeLine(threadId, line,
                        buffers, localTileMinMax,
                        localMinMax, localCounts,
                        hasLocalZRange, localZMin, localZMax, localZHist);
        }
        flushAll(threadId, buffers);
        mergeLocalStats(localMinMax, localCounts, hasLocalZRange, localZMin, localZMax, localZHist);
    }
    // Non-stream mode - Decide chunk boundaries and dispatch worker threads
    bool launchWorkerThreads() {
        std::vector<std::pair<std::streampos, std::streampos>> blocks;
        computeBlocks(blocks, inFile_, nThreads_, nskip_);
        for (size_t i = 0; i < blocks.size(); i++) {
            threads_.emplace_back(&Pts2Tiles::worker, this, static_cast<int>(i), blocks[i].first, blocks[i].second);
        }
        return true;
    }
    // Stream mode - Read the next chunk from the input stream and process
    void streamingWorker(int threadId) {
        std::unordered_map<TileKey, std::vector<std::string>, TileKeyHash> buffers;
        std::unordered_map<TileKey, Rectangle<float>, TileKeyHash> localTileMinMax;
        std::unordered_map<std::string,std::vector<int32_t>> localCounts;
        std::unordered_map<int64_t, uint64_t> localZHist;
        Rectangle<float> localMinMax;
        bool hasLocalZRange = false;
        float localZMin = 0, localZMax = 0;

        std::vector<std::string> batch;
        batch.reserve(batchSize_);
        while (true) {
            batch.clear();
            { // —— fill one batch under lock ——
                std::lock_guard lk(readMutex_);
                for (int i = 0; i < batchSize_; ++i) {
                    std::string line;
                    if (!readNextInputLine(line)) break;
                    batch.push_back(std::move(line));
                }
            }
            if (batch.empty()) break;
            for (auto &ln : batch) {
                if (ln.empty()) {
                    continue;
                }
                if (ln[0] == '#') {
                    continue;
                }
                consumeLine(threadId, ln, buffers, localTileMinMax,
                            localMinMax, localCounts,
                            hasLocalZRange, localZMin, localZMax, localZHist);
            }
            flushAll(threadId, buffers);
        }
        mergeLocalStats(localMinMax, localCounts, hasLocalZRange, localZMin, localZMax, localZHist);
    }

    bool joinWorkerThreads() {
        for (auto& t : threads_) {
            t.join();
        }
        return true;
    }

    // Update global coordinate range & feature counts
    void mergeLocalStats(const Rectangle<float>& box,
            std::unordered_map<std::string,std::vector<int32_t>> &localCounts,
            bool hasLocalZRange,
            float localZMin,
            float localZMax,
            std::unordered_map<int64_t, uint64_t>& localZHist) {
        std::lock_guard lk(minmaxMutex_);
        globalBox_.extendToInclude(box);
        if (hasLocalZRange) {
            if (!hasGlobalZRange_) {
                hasGlobalZRange_ = true;
                globalZMin_ = localZMin;
                globalZMax_ = localZMax;
            } else {
                if (localZMin < globalZMin_) globalZMin_ = localZMin;
                if (localZMax > globalZMax_) globalZMax_ = localZMax;
            }
        }
        for (auto &p : localCounts) {
            auto &glob = featureCounts_[p.first];
            if (glob.empty()) {
                glob = std::move(p.second);
            } else {
                for (size_t i = 0; i < glob.size(); ++i) {
                    glob[i] += p.second[i];}
            }
        }
        for (const auto& p : localZHist) {
            zHist_[p.first] += p.second;
        }
    }

    // Merge temporary files belonging to one tile
    bool mergeTmpFileToOutput(const TileKey& tile, int fdOut, uint64_t& currentOffset) {
        constexpr size_t bufSz = 1024 * 1024;
        std::vector<char> buffer(bufSz);
        for (uint32_t threadId = 0; threadId < nThreads_; ++threadId) {
            auto tmpFilename = getTmpFilename(tile, static_cast<int>(threadId));
            std::ifstream tmpFile(tmpFilename, std::ios::binary);
            if (tmpFile) {
                while (tmpFile) {
                    tmpFile.read(buffer.data(), static_cast<std::streamsize>(buffer.size()));
                    const std::streamsize got = tmpFile.gcount();
                    if (got > 0) {
                        if (!write_all(fdOut, buffer.data(), static_cast<size_t>(got))) {
                            const int err = errno;
                            error("%s: failed writing temporary shard %s to output %s.tsv: %s",
                                __func__, tmpFilename.string().c_str(), outPref_.c_str(), std::strerror(err));
                            return false;
                        }
                        currentOffset += static_cast<uint64_t>(got);
                    }
                }
                if (!tmpFile.eof()) {
                    error("%s: failed reading temporary shard %s",
                        __func__, tmpFilename.string().c_str());
                    return false;
                }
                tmpFile.close();
                // Remove temporary file after merging.
                std::filesystem::remove(tmpFilename);
            }
        }
        return true;
    }

    // Merge all temporary files and write index file
    bool mergeAndWriteIndex() {
        std::string outFile = outPref_ + ".tsv";
        int fdOut = open(outFile.c_str(), O_CREAT | O_WRONLY | O_TRUNC | O_CLOEXEC, 0644);
        if (fdOut < 0) {
            error("Error opening output file for writing: %s: %s", outFile.c_str(), std::strerror(errno));
        }
        uint64_t currentOffset = 0;
        if (!metaLines_.empty()) {
            if (!write_all(fdOut, metaLines_.data(), metaLines_.size())) {
                const int err = errno;
                close(fdOut);
                error("%s: failed writing metadata header to %s: %s",
                    __func__, outFile.c_str(), std::strerror(err));
            }
            currentOffset += static_cast<uint64_t>(metaLines_.size());
        }

        std::string indexFilename = outPref_ + ".index";
        std::ofstream indexfile(indexFilename, std::ios::binary);
        if (!indexfile) {
            close(fdOut);
            error("Error opening index file for writing: %s", indexFilename.c_str());
        }

        // Write header
        IndexHeader header;
        header.magic = PUNKST_INDEX_MAGIC;
        header.mode = 0; // tsv, no scaling, float coords, regular tiles
        if (icol_z_ >= 0) {header.mode |= 0x10;} // 3D
        if (tileOpFactorTsv_) {
            if (!header.packKvec({static_cast<uint32_t>(tileOpTopK_)})) {
                error("%s: cannot encode topK=%d in TileOperator index header",
                    __func__, tileOpTopK_);
            }
        }
        header.tileSize = tileSize_;
        header.xmin = globalBox_.xmin; header.xmax = globalBox_.xmax;
        header.ymin = globalBox_.ymin; header.ymax = globalBox_.ymax;
        indexfile.write(reinterpret_cast<const char*>(&header), sizeof(header));

        std::vector<TileKey> sortedTiles;
        for (const auto& pair : globalTiles_) {
            sortedTiles.push_back(pair.first);
        }
        std::sort(sortedTiles.begin(), sortedTiles.end());

        for (const auto& tile : sortedTiles) {
            const uint64_t startOffset = currentOffset;
            if (!mergeTmpFileToOutput(tile, fdOut, currentOffset)) {
                close(fdOut);
                return false;
            }
            const uint64_t endOffset = currentOffset;
            IndexEntryF entry(tile.row, tile.col);
            entry.st = startOffset;
            entry.ed = endOffset;
            entry.n = static_cast<uint32_t>(globalTiles_[tile]);
            tile2bound(tile.row, tile.col, entry.xmin, entry.xmax, entry.ymin, entry.ymax, tileSize_);
            indexfile.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
            if (!indexfile) {
                error("%s: failed writing index entry for tile (%d, %d)",
                    __func__, tile.row, tile.col);
            }
        }

        if (close(fdOut) != 0) {
            const int err = errno;
            error("%s: failed closing output file %s: %s",
                __func__, outFile.c_str(), std::strerror(err));
        }
        indexfile.close();
        if (!indexfile) {
            error("%s: failed closing index file %s", __func__, indexFilename.c_str());
        }
        return true;
    }

    // Write global coordinate range and feature counts
    bool writeAuxiliaryFiles() {
        std::string outFile = outPref_ + ".coord_range.tsv";
        std::ofstream out(outFile);
        if (!out) {
            warning("Error opening output file for writing: %s", outFile.c_str());
            return false;
        }
        out << "xmin\t" << globalBox_.xmin << "\n"
            << "xmax\t" << globalBox_.xmax << "\n"
            << "ymin\t" << globalBox_.ymin << "\n"
            << "ymax\t" << globalBox_.ymax << "\n";
        if (icol_z_ >= 0 && hasGlobalZRange_) {
            out << "zmin\t" << globalZMin_ << "\n"
                << "zmax\t" << globalZMax_ << "\n";
        }
        out.close();
        if (icol_z_ >= 0) {
            outFile = outPref_ + ".z_hist.tsv";
            out.open(outFile);
            if (!out) {
                warning("Error opening output file for writing: %s", outFile.c_str());
                return false;
            }
            if (hasGlobalZRange_) {
                const int64_t zminBin = static_cast<int64_t>(std::floor(globalZMin_));
                const int64_t zmaxBin = static_cast<int64_t>(std::floor(globalZMax_));
                for (int64_t z = zminBin; z <= zmaxBin; ++z) {
                    const auto it = zHist_.find(z);
                    const uint64_t count = (it == zHist_.end()) ? 0 : it->second;
                    out << z << "\t" << count << "\n";
                }
            }
            out.close();
        }
        if (icol_feature_ >= 0) {
            std::string outFile = outPref_ + ".features.tsv";
            out.open(outFile);
            if (!out) {
                warning("Error opening output file for writing: %s", outFile.c_str());
                return false;
            }
            std::vector<std::string> featureNames;
            featureNames.reserve(featureCounts_.size());
            for (const auto& pair : featureCounts_) {
                featureNames.push_back(pair.first);
            }
            std::sort(featureNames.begin(), featureNames.end());
            for (const auto& feature : featureNames) {
                const auto& vals = featureCounts_.at(feature);
                out << feature;
                for (const auto& val : vals) {
                    out << "\t" << val;
                }
                out << "\n";
            }
            out.close();
        }
        return true;
    }
};














// NOT TESTED YET
class Pts2TilesAnno2D: public Pts2Tiles {
public:
    std::vector<std::unique_ptr<kd_tree_f2_t> > trees;
    int32_t dist_out_precision;

    void add_refpts(const std::string& file, int32_t f = 4) {
        PointCloud<float> cloud;
        std::ifstream ifs(file);
        if (!ifs) {
            error("Error opening reference points file: %s", file.c_str());
        }
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            float x, y;
            iss >> x >> y;
            cloud.pts.push_back({x, y});
        }
        trees.push_back(std::unique_ptr<kd_tree_f2_t>(new kd_tree_f2_t(2, cloud, {10})));
        dist_out_precision = f;
    }
    void initialize_refpts(const std::vector<std::string>& files, int32_t f = 4) {
        for (const auto& file : files) {
            add_refpts(file, f);
        }
    }

    ~Pts2TilesAnno2D() {}

protected:
    TileKey parse(std::string& line, float& x, float& y) {
        std::vector<std::string> tokens;
        tokenizeLine(line, tokens);
        if (tokens.size() < ntokens_) {
            error("Error parsing line: %s", line.c_str());
        }
        x = std::stof(tokens[icol_x_]);
        y = std::stof(tokens[icol_y_]);
        rewriteCoordinates(tokens, line, x, y);
        TileKey tile;
        tile.row = static_cast<int32_t>(std::floor(y / tileSize_));
        tile.col = static_cast<int32_t>(std::floor(x / tileSize_));
        return tile;
    }

    // Worker thread function
    virtual void worker(int threadId, std::streampos start, std::streampos end) {
        std::ifstream file(inFile_);
        if (!file) {
            error("Thread %d: Error opening file input file", threadId);
        }
        file.seekg(start);
        std::string line;
        // Map: tile id -> vector of lines (buffer)
        std::unordered_map<TileKey, std::vector<std::string>, TileKeyHash> buffers;
        while (file.tellg() < end && std::getline(file, line)) {
            float pt[2];
            TileKey tile = parse(line, pt[0], pt[1]);
            std::ostringstream oss;
            oss << std::fixed << std::setprecision(dist_out_precision);
            // fine the distance to the nearest reference point in each kdtree
            for (size_t i = 0; i < trees.size(); ++i) {
                std::vector<uint32_t> indices(1);
                std::vector<float> dists(1);
                trees[i]->knnSearch(pt, 1, &indices[0], &dists[0]);
                oss << "\t" << std::pow(dists[0], 0.5f);
            }
            line += oss.str();
            buffers[tile].push_back(line);
            // Flush if the buffer is large enough.
            if (buffers[tile].size() >= tileBuffer_) {
                { // update globalTiles_
                    std::lock_guard<std::mutex> lock(globalTilesMutex_);
                    globalTiles_[tile] += buffers[tile].size();
                }
                auto tmpFilename = getTmpFilename(tile, threadId);
                std::ofstream out(tmpFilename, std::ios::app);
                for (const auto& bufferedLine : buffers[tile]) {
                    out << bufferedLine << "\n";
                }
                out.close();
                buffers[tile].clear();
            }
        }
        // Flush any remaining data in the buffers.
        for (auto& pair : buffers) {
            if (!pair.second.empty()) {
                { // update globalTiles_
                    std::lock_guard<std::mutex> lock(globalTilesMutex_);
                    globalTiles_[pair.first] += pair.second.size();
                }
                auto tmpFilename = getTmpFilename(pair.first, threadId);
                std::ofstream out(tmpFilename, std::ios::app);
                for (const auto& bufferedLine : pair.second) {
                    out << bufferedLine << "\n";
                }
                out.close();
            }
        }
    }

};
