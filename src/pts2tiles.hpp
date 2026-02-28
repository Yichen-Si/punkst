#pragma once

#include "utils.h"
#include "img_utils.hpp"
#include "utils_sys.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <memory>
#include <array>
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
        int32_t icol_x, int32_t icol_y, int32_t icol_g = -1, std::vector<int32_t> icol_ints = {}, int32_t nskip = 0,
        bool streamingMode = false,
        int32_t tileBuffer = 1000, int32_t batchSize = 10000,
        double scale = 1, int digits=2) :
        nThreads_(nthreads),
        inFile_(inFile), tmpDir_(tmpDir),
        outPref_(outPref), tileSize_(tileSize),
        icol_x_(icol_x), icol_y_(icol_y), icol_feature_(icol_g),
        icol_ints_(icol_ints), nskip_(nskip),
        streamingMode_(streamingMode),
        tileBuffer_(tileBuffer), batchSize_(batchSize),
        scale_(scale), digits_(digits)
    {
        ntokens_ = std::max(icol_x_, icol_y_);
        if (icol_feature_ >= 0) {
            ntokens_ = std::max(ntokens_, icol_feature_);
            for (const auto& icol : icol_ints_) {
                ntokens_ = std::max(ntokens_, icol);
            }
        }
        scaling_ = std::abs(scale_ - 1) > 1e-8;
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
            if (!launchWorkerThreads()) {
                error("Error launching worker threads_");
            }
        } else {
            openInput();
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
    int32_t icol_x_, icol_y_, icol_feature_;
    std::vector<int32_t> icol_ints_;
    int32_t ntokens_;
    int32_t nskip_, nskipped_ = 0;
    double scale_;
    bool scaling_;
    int32_t digits_;
    Rectangle<float> globalBox_;
    std::string metaLines_;

    std::unordered_map<std::string, std::vector<int32_t>> featureCounts_;
    std::mutex minmaxMutex_;

    std::unordered_map<TileKey, uint64_t, TileKeyHash> globalTiles_;
    std::mutex globalTilesMutex_;

    std::vector<std::thread> threads_;
    bool streamingMode_;
    gzFile         gz_    = nullptr;
    std::istream*  inPtr_ = nullptr;
    std::mutex readMutex_;

    struct PtRecord {
        float x, y;
        std::string feature;
        std::vector<int32_t> vals;
    };

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
            warning("%s: the input is not stdin or gzipped file but the streaming mode is used, assuming it is a plain tsv file", __FUNCTION__);
            inPtr_ = new std::ifstream(inFile_);
            if (!inPtr_ || !static_cast<std::ifstream*>(inPtr_)->is_open()) {
                error("Error opening input file: %s", inFile_.c_str());
            }
        }
        if (nskip_ <= 0) return;

        // Skip initial lines
        if (gz_) {
            char buf[1<<16];
            while (nskipped_ < nskip_ && gzgets(gz_, buf, sizeof(buf))) {
                nskipped_++;
            }
        } else {
            std::string line;
            while (nskipped_ < nskip_ && std::getline(*inPtr_, line)) {
                nskipped_++;
            }
        }
    }

    std::filesystem::path getTmpFilename(const TileKey& tile, int threadId) const {
        return tmpDir_.path / (std::to_string(tile.row) + "_" + std::to_string(tile.col)
            + "_" + std::to_string(threadId) + ".tsv");
    }

    // Parse a line, extract coordinates and return the tile key
    virtual TileKey parse(std::string& line, PtRecord& pt) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < ntokens_) {
            error("Error parsing line: %s", line.c_str());
        }
        pt.x = std::stof(tokens[icol_x_]);
        pt.y = std::stof(tokens[icol_y_]);
        if (scaling_) {
            pt.x *= scale_;
            pt.y *= scale_;
            tokens[icol_x_] = fp_to_string(pt.x, digits_);
            tokens[icol_y_] = fp_to_string(pt.y, digits_);
            line = join(tokens, "\t");
        }
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
        split(tokens, "\t", line);
        if (tokens.size() < ntokens_) {
            error("Error parsing line: %s", line.c_str());
        }
        x = std::stof(tokens[icol_x_]);
        y = std::stof(tokens[icol_y_]);
        if (scaling_) {
            x *= scale_;
            y *= scale_;
            tokens[icol_x_] = fp_to_string(x, digits_);
            tokens[icol_y_] = fp_to_string(y, digits_);
            line = join(tokens, "\t");
        }
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
            std::unordered_map<std::string,std::vector<int32_t>> &localCounts) {
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
        Rectangle<float> localMinMax;

        std::ifstream file(inFile_);
        if (!file) {
            error("Thread %d: Error opening file input file", threadId);
        }
        file.seekg(start);
        std::string line;

        while (file.tellg() < end && std::getline(file, line)) {
            if (line.empty()) continue;
            if (line[0] == '#') {
                std::lock_guard lk(globalTilesMutex_);
                metaLines_ += line + "\n";
                continue;
            }
            consumeLine(threadId, line,
                        buffers, localTileMinMax,
                        localMinMax, localCounts);
        }
        flushAll(threadId, buffers);
        mergeLocalStats(localMinMax, localCounts);
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
        Rectangle<float> localMinMax;

        std::vector<std::string> batch;
        batch.reserve(batchSize_);
        while (true) {
            batch.clear();
            { // —— fill one batch under lock ——
                std::lock_guard lk(readMutex_);
                if (gz_) {
                    for (int i = 0; i < batchSize_; ++i) {
                        char buf[1<<16];
                        if (!gzgets(gz_, buf, sizeof(buf))) break;
                        size_t len = strlen(buf);
                        if (len > 0 && buf[len-1] == '\n') {
                            buf[--len] = '\0';
                            if (len > 0 && buf[len-1] == '\r')
                                buf[--len] = '\0';
                            batch.emplace_back(buf);
                        } else {
                            // Continue reading
                            std::string line(buf, len);
                            while (true) {
                                if (!gzgets(gz_, buf, sizeof(buf))) break;
                                size_t chunkLen = strlen(buf);
                                bool gotNL = chunkLen > 0 && buf[chunkLen-1] == '\n';
                                // append without the trailing '\n'
                                line.append(buf, gotNL ? chunkLen-1 : chunkLen);
                                if (gotNL) break;
                            }
                            // strip a final '\r' if present
                            if (!line.empty() && line.back() == '\r')
                                line.pop_back();
                            batch.push_back(std::move(line));
                        }
                    }
                } else {
                    for (int i = 0; i < batchSize_; ++i) {
                        std::string line;
                        if (!std::getline(*inPtr_, line)) break;
                        if (!line.empty() && line.back()=='\r')
                            line.pop_back();
                        batch.push_back(std::move(line));
                    }
                }
            }
            if (batch.empty()) break;
            for (auto &ln : batch) {
                if (ln[0] == '#') {
                    std::lock_guard lk(globalTilesMutex_);
                    metaLines_ += ln + "\n";
                    continue;
                }
                consumeLine(threadId, ln, buffers, localTileMinMax,
                            localMinMax, localCounts);
            }
            flushAll(threadId, buffers);
        }
        mergeLocalStats(localMinMax,localCounts);
    }

    bool joinWorkerThreads() {
        for (auto& t : threads_) {
            t.join();
        }
        return true;
    }

    // Update global coordinate range & feature counts
    void mergeLocalStats(const Rectangle<float>& box,
            std::unordered_map<std::string,std::vector<int32_t>> &localCounts) {
        std::lock_guard lk(minmaxMutex_);
        globalBox_.extendToInclude(box);
        for (auto &p : localCounts) {
            auto &glob = featureCounts_[p.first];
            if (glob.empty()) {
                glob = std::move(p.second);
            } else {
                for (size_t i = 0; i < glob.size(); ++i) {
                    glob[i] += p.second[i];}
            }
        }
    }

    // Merge temporary files belonging to one tile
    void mergeTmpFileToOutput(const TileKey& tile, std::ofstream& outfile) {
        for (uint32_t threadId = 0; threadId < nThreads_; ++threadId) {
            auto tmpFilename = getTmpFilename(tile, static_cast<int>(threadId));
            std::ifstream tmpFile(tmpFilename, std::ios::binary);
            if (tmpFile) {
                outfile << tmpFile.rdbuf();
                tmpFile.close();
                // Remove temporary file after merging.
                std::filesystem::remove(tmpFilename);
            }
        }
    }

    // Merge all temporary files and write index file
    bool mergeAndWriteIndex() {
        std::ofstream outfile(outPref_ + ".tsv", std::ios::binary);
        if (!outfile) {
            error("Error opening output file for writing: %s.tsv", outPref_.c_str());
        }
        if (!metaLines_.empty()) {outfile << metaLines_;}

        std::string indexFilename = outPref_ + ".index";
        std::ofstream indexfile(indexFilename, std::ios::binary);
        if (!indexfile) {
            error("Error opening index file for writing: %s", indexFilename.c_str());
        }

        // Write header
        IndexHeader header;
        header.magic = PUNKST_INDEX_MAGIC;
        header.mode = 0; // tsv, no scaling, float coords, regular tiles
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
            std::streampos startOffset = outfile.tellp();
            mergeTmpFileToOutput(tile, outfile);
            std::streampos endOffset = outfile.tellp();
            IndexEntryF entry(tile.row, tile.col);
            entry.st = startOffset;
            entry.ed = endOffset;
            entry.n = static_cast<uint32_t>(globalTiles_[tile]);
            tile2bound(tile.row, tile.col, entry.xmin, entry.xmax, entry.ymin, entry.ymax, tileSize_);
            indexfile.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
        }

        outfile.close();
        indexfile.close();
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
        out.close();
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
        split(tokens, "\t", line);
        if (tokens.size() < ntokens_) {
            error("Error parsing line: %s", line.c_str());
        }
        x = std::stof(tokens[icol_x_]);
        y = std::stof(tokens[icol_y_]);
        if (scaling_) {
            x *= scale_;
            y *= scale_;
            tokens[icol_x_] = fp_to_string(x, digits_);
            tokens[icol_y_] = fp_to_string(y, digits_);
            line = join(tokens, "\t");
        }
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
