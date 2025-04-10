#pragma once

#include "punkst.h"
#include "dataunits.hpp"
#include "hexgrid.h"
#include "tilereader.hpp"
#include "threads.hpp"
#include "utils.h"
#include <memory>
#include <atomic>
#include <random>
#include <utility>
#include <iomanip>
#include <filesystem>
#include "json.hpp"
#include "nanoflann.hpp"
#include "nanoflann_utils.h"

class Tiles2Hex {

public:

    Tiles2Hex(int32_t nThreads, std::string& tmpDir, std::string& outFile, HexGrid& hexGrid, TileReader& tileReader, lineParser& parser, std::vector<int32_t> minCounts = {});
    ~Tiles2Hex() {
        if (mainOut.is_open()) {
            mainOut.close();
        }
    }

    bool run();

    void writeMetadata();

protected:
    int32_t nThreads;
    std::string outFile, tmpDir;
    int32_t nModal, nUnits, nFeatures;
    std::vector<int32_t> minCounts;
    nlohmann::json meta;

    lineParser parser;
    HexGrid hexGrid;
    TileReader tileReader;

    ThreadSafeQueue<TileKey> tileQueue;
    std::vector<std::thread> threads;
    std::mutex mainOutMutex;
    std::ofstream mainOut;

    // Process one tile at a time
    virtual void worker(int threadId);
    // Merge boundary hexagons from all temporary files and append to mainOut
    virtual bool mergeBoundaryHexagons();

    bool launchWorkerThreads() {
        for (int i = 0; i < nThreads; ++i) {
            threads.emplace_back(&Tiles2Hex::worker, this, i);
        }
        std::vector<TileKey> tileList;
        tileReader.getTileList(tileList);
        for (const auto& tile : tileList) {
            tileQueue.push(tile);
        }
        tileQueue.set_done();
        return true;
    }

    bool joinWorkerThreads() {
        for (auto& t : threads) {
            t.join();
        }
        return true;
    }

    void addPixelToUnitMaps(PixelValues& pixel, int32_t hx, int32_t hy, std::unordered_map<int64_t, UnitValues>& Units, int32_t l = -1) {
        int64_t key = (static_cast<int64_t>(hx) << 32) | hy;
        auto it = Units.find(key);
        if (it == Units.end()) {
            Units.insert({key, UnitValues(hx, hy, pixel, l)});
        } else {
            it->second.addPixel(pixel);
        }
    }

};

class Tiles2UnitsByAnchor : public Tiles2Hex {

public:

    Tiles2UnitsByAnchor(int32_t nThreads, std::string& tmpDir, std::string& outFile, HexGrid& hexGrid, TileReader& tileReader, lineParser& parser, std::vector<std::string>& anchorFiles, std::vector<float>& radius, std::vector<int32_t> minCounts = {}, bool noBackground = false)
        : Tiles2Hex(nThreads, tmpDir, outFile, hexGrid, tileReader, parser, minCounts), noBackground(noBackground) {
        assert(anchorFiles.empty() && anchorFiles.size() == radius.size());
        for (auto& f : anchorFiles) {
            readAnchors(f);
        }
        nAnchorSets = trees.size();
        nLayer = nAnchorSets;
        if (!noBackground) {
            nLayer++;
        }
        l2radius.resize(radius.size());
        for (size_t i = 0; i < radius.size(); ++i) {
            l2radius[i] = radius[i] * radius[i];
        }
        meta["n_anchor_sets"] = nAnchorSets;
        meta["n_layers"] = nLayer;
        meta["anchor_radius"] = radius;
    }

protected:

    uint32_t nAnchorSets, nLayer;
    bool noBackground;
    std::vector<uint32_t> nUnitsPerLabel;
    std::vector<std::unique_ptr<kd_tree_f2_t>> trees;
    std::vector<float> l2radius;

    void readAnchors(std::string& anchorFile) {
        PointCloud<float> cloud;
        std::ifstream ifs(anchorFile);
        if (!ifs) {
            error("Error opening anchor file: %s", anchorFile.c_str());
        }
        std::string line;
        while (std::getline(ifs, line)) {
            std::istringstream iss(line);
            float x, y;
            iss >> x >> y;
            cloud.pts.push_back({x, y});
        }
        trees.push_back(std::unique_ptr<kd_tree_f2_t>(new kd_tree_f2_t(2, cloud, {10})));
    }

    void worker(int threadId) override;
    bool mergeBoundaryHexagons() override;

};
