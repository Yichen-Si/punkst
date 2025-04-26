#pragma once

#include "punkst.h"
#include "tilereader.hpp"
#include "threads.hpp"
#include "utils.h"
#include "nanoflann_utils.h"
#include <fstream>
#include <unordered_map>

/// Compute feature‑by‑feature co‑occurrence within radius r
class Tiles2FeatureCooccurrence {
public:
    Tiles2FeatureCooccurrence(
        int32_t nThreads, TileReader& tileReader, lineParserUnival& parser,
        const std::string& outPref,
        double radius, double halflife=-1,
        double localMin=0, bool binaryOutput = false,
        int32_t minNeighbor = 1, bool weightByCount = false) :
        nThreads_(nThreads) , outPref_(outPref) , r2_(radius * radius)
        , halflife_(halflife), weightByDistance_(halflife > 0)
        , tileReader_(tileReader) , parser_(parser)
        , localMin_(localMin), binaryOutput_(binaryOutput), minNeighbor_(minNeighbor), weightByCount_(weightByCount) {
        std::vector<TileKey> tiles;
        tileReader_.getTileList(tiles);
        for (auto &t : tiles) {
            tileQueue_.push(t);
        }
        tileQueue_.set_done();
        tau_ = (halflife_ > 0) ? log(2) / halflife_ / halflife_ : 0;
      }

    bool run() {
        // launch workers
        for (int i = 0; i < nThreads_; ++i) {
            threads_.emplace_back(&Tiles2FeatureCooccurrence::worker, this, i);
        }
        for (auto &th : threads_) th.join();
        writeToFile();
        return true;
    }

private:

    int32_t nThreads_;
    double  r2_; // squared radius
    double halflife_, tau_;
    std::string outPref_;
    double localMin_;
    bool weightByDistance_, weightByCount_, binaryOutput_;
    int32_t minNeighbor_;

    TileReader&   tileReader_;
    lineParserUnival&   parser_;

    ThreadSafeQueue<TileKey> tileQueue_;
    std::vector<std::thread>  threads_;

    std::unordered_map<uint32_t, std::unordered_map<uint32_t, double>> globalCounts_;
    std::unordered_map<uint32_t, std::array<uint64_t, 4> > globalMargianls_;

    std::mutex globalMtx_;

    void worker(int threadId) {
        TileKey tile;
        while (tileQueue_.pop(tile)) {
            // read this tile’s points
            PointCloud<float> cloud;            // from nanoflann_utils
            std::vector<uint32_t> feats;
            std::vector<uint8_t> counts;
            std::unordered_map<uint32_t, std::unordered_map<uint32_t, double>> localCounts;
            std::unordered_map<uint32_t, std::array<uint64_t, 4> > localMarginals;
            bool checkBounds = tileReader_.isPartial(tile);

            auto iter = tileReader_.get_tile_iterator(tile.row, tile.col);
            std::string line;
            while (iter->next(line)) {
                RecordT<float> px;
                if (parser_.parse(px, line, checkBounds) < 0) continue;
                cloud.pts.emplace_back(px.x, px.y);  // z=0
                feats.push_back(px.idx);
                if (weightByCount_) {
                    counts.push_back(static_cast<uint8_t>(std::max(256u, px.ct)));
                }
                localMarginals[px.idx][0] += 1;
                localMarginals[px.idx][1] += px.ct;
            }
            const size_t N = cloud.pts.size();
            if (N < 2) continue;

            // build local tree
            kd_tree_f2_t tree(2, cloud,
                nanoflann::KDTreeSingleIndexAdaptorParams(10));
            std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;

            // count co‑occurrences
            size_t nSkip = 0;
            for (size_t i = 0; i < N; ++i) {
                float query_pt[2] = { cloud.pts[i].x, cloud.pts[i].y };
                uint32_t n = tree.radiusSearch(query_pt, r2_, indices_dists);
                uint32_t f1 = feats[i];
                if (n < minNeighbor_) {
                    nSkip++;
                    continue;
                }
                localMarginals[f1][3] += n;
                if (weightByCount_) {
                    localMarginals[f1][2] += counts[i];
                    for (uint32_t j = 0; j < n; ++j) {
                        auto &m = indices_dists[j];
                        uint32_t f2 = feats[m.first];
                        if (weightByDistance_) {
                            localCounts[f1][f2] += exp(-tau_ * m.second) * counts[m.first] * counts[i];
                        } else {
                            localCounts[f1][f2] += counts[m.first] * counts[i];
                        }
                    }
                } else {
                    localMarginals[f1][2] += 1;
                    for (uint32_t j = 0; j < n; ++j) {
                        auto &m = indices_dists[j];
                        uint32_t f2 = feats[m.first];
                        if (weightByDistance_) {
                            localCounts[f1][f2] += exp(-tau_ * m.second);
                        } else {
                            localCounts[f1][f2] += 1;
                        }
                    }
                }
            }
            // merge into global
            {
                std::lock_guard<std::mutex> lock(globalMtx_);
                for (auto &kv : localCounts) {
                    for (auto &kv2 : kv.second) {
                        if (kv2.second < localMin_) continue;
                        globalCounts_[kv.first][kv2.first] += kv2.second;
                    }
                }
                for (auto &kv : localMarginals) {
                    auto &kv2 = globalMargianls_[kv.first];
                    for (uint32_t i = 0; i < kv2.size(); ++i) {
                        kv2[i] += kv.second[i];
                    }
                }
            }
            notice("Thread %d: processed tile (%d, %d) with %zu points. Skipped %zu (%.4f) with less than %d neighbors.", threadId, tile.row, tile.col, N, nSkip, double(nSkip) / N, minNeighbor_);
        }
    }

    void writeToFile() {
        // write marginals
        std::string outFile = outPref_ + ".marginals.tsv";
        FILE* ofs = fopen(outFile.c_str(), "w");
        if (!ofs) {
            error("Cannot open output: %s", outFile.c_str());
        }
        std::vector<std::string> featureList;
        if (parser_.getFeatureList(featureList) < 0) {
            warning("Feature names are not found");
            // feature name, total uniq pixels, total counts, used pixels, counted neighboring pixels
            for (auto &kv : globalMargianls_) {
                fprintf(ofs, "%u\t%llu\t%llu\t%llu\t%llu\n", kv.first,
                    kv.second[0], kv.second[1], kv.second[2], kv.second[3]);
            }
        } else {
            for (auto &kv : globalMargianls_) {
                fprintf(ofs, "%u\t%s\t%llu\t%llu\t%llu\t%llu\n", kv.first,
                    featureList[kv.first].c_str(), kv.second[0], kv.second[1], kv.second[2], kv.second[3]);
            }
        }
        fclose(ofs);

        // write matrix
        if (binaryOutput_) {
            std::string outFile = outPref_ + ".mtx.bin";
            std::ofstream ofs(outFile, std::ios::binary);
            if (!ofs) {
                error("Cannot open output: %s", outFile.c_str());
            }
            for (auto &kv : globalCounts_) {
                for (auto &kv2 : kv.second) {
                    uint32_t f1 = kv.first;
                    uint32_t f2 = kv2.first;
                    ofs.write(reinterpret_cast<const char*>(&f1), sizeof(f1));
                    ofs.write(reinterpret_cast<const char*>(&f2), sizeof(f2));
                    if (weightByDistance_) {
                        double w = kv2.second;
                        ofs.write(reinterpret_cast<const char*>(&w), sizeof(w));
                    } else {
                        int64_t c = int64_t(kv2.second);
                        ofs.write(reinterpret_cast<const char*>(&c), sizeof(c));
                    }
                }
            }
        } else {
            std::string outFile = outPref_ + ".mtx.tsv";
            FILE* ofs = fopen(outFile.c_str(), "w");
            if (!ofs) {
                error("Cannot open output: %s", outFile.c_str());
            }
            for (auto &kv : globalCounts_) {
                for (auto &kv2 : kv.second) {
                    uint32_t f1 = kv.first;
                    uint32_t f2 = kv2.first;
                    if (weightByDistance_) {
                        double w = kv2.second;
                        fprintf(ofs, "%u\t%u\t%.2f\n", f1, f2, w);
                    } else {
                        int64_t c = int64_t(kv2.second);
                        fprintf(ofs, "%u\t%u\t%lld\n", f1, f2, c);
                    }
                }
            }
            fclose(ofs);
        }
    }
};
