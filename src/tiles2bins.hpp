#pragma once

#include "punkst.h"
#include "bccgrid.hpp"
#include "dataunits.hpp"
#include "hexgrid.h"
#include "tilereader.hpp"
#include "threads.hpp"
#include "utils.h"
#include "utils_sys.hpp"
#include <memory>
#include <atomic>
#include <map>
#include <random>
#include <utility>
#include <iomanip>
#include <filesystem>
#include <cstdint>
#include <cmath>
#include <algorithm>
#include <limits>
#include "json.hpp"
#include "nanoflann.hpp"
#include "kdtree_utils.hpp"

struct FeatureInfoOptions {
    double idfQ = 95.0;
    double idfPower = 0.3;
    double idfMin = 0.1;
    double idfMax = 5.0;
};

struct FeatureInfoWeights {
    std::vector<double> idf;
    std::vector<double> infoWeight;
};

struct FeatureOccurrenceStats {
    uint64_t nUnits = 0;
    std::vector<uint64_t> totalCounts;
    std::vector<uint64_t> nPresent;
    std::vector<uint64_t> nGt1;
    std::vector<uint64_t> nGt2;
    std::vector<std::map<uint32_t, uint64_t>> countHist;

    void ensureFeature(uint32_t feature) {
        const size_t n = static_cast<size_t>(feature) + 1;
        if (totalCounts.size() >= n) {
            return;
        }
        totalCounts.resize(n, 0);
        nPresent.resize(n, 0);
        nGt1.resize(n, 0);
        nGt2.resize(n, 0);
        countHist.resize(n);
    }

    void ensureFeatureCount(size_t nFeatures) {
        if (nFeatures == 0) {
            return;
        }
        ensureFeature(static_cast<uint32_t>(nFeatures - 1));
    }

    void observeUnit(const std::map<uint32_t, uint32_t>& vals) {
        ++nUnits;
        for (const auto& entry : vals) {
            const uint32_t count = entry.second;
            if (count == 0) {
                continue;
            }
            ensureFeature(entry.first);
            totalCounts[entry.first] += count;
            nPresent[entry.first] += 1;
            if (count > 1) {
                nGt1[entry.first] += 1;
            }
            if (count > 2) {
                nGt2[entry.first] += 1;
            }
            countHist[entry.first][count] += 1;
        }
    }

    uint64_t countGreaterThanMean(size_t feature) const {
        if (feature >= countHist.size() || nUnits == 0) {
            return 0;
        }
        uint64_t out = 0;
        const uint64_t total = totalCounts[feature];
        for (const auto& entry : countHist[feature]) {
            if (static_cast<uint64_t>(entry.first) * nUnits > total) {
                out += entry.second;
            }
        }
        return out;
    }

    static double percentile(std::vector<double> values, double q) {
        if (values.empty()) {
            return 0.0;
        }
        q = std::max(0.0, std::min(100.0, q));
        std::sort(values.begin(), values.end());
        if (values.size() == 1) {
            return values.front();
        }
        const double pos = (q / 100.0) * static_cast<double>(values.size() - 1);
        const size_t lo = static_cast<size_t>(std::floor(pos));
        const size_t hi = std::min(values.size() - 1, lo + 1);
        const double frac = pos - static_cast<double>(lo);
        return values[lo] * (1.0 - frac) + values[hi] * frac;
    }

    static FeatureInfoWeights computeWeights(size_t nFeatures,
            const FeatureOccurrenceStats& stats,
            const FeatureInfoOptions& options = FeatureInfoOptions()) {
        FeatureInfoWeights out;
        out.idf.assign(nFeatures, 0.0);
        out.infoWeight.assign(nFeatures, 0.0);
        std::vector<double> rawIdf(nFeatures, 0.0);
        std::vector<double> rawInfo(nFeatures, 0.0);
        uint64_t corpusTotal = 0;
        for (size_t i = 0; i < nFeatures; ++i) {
            const uint64_t present = i < stats.nPresent.size() ? stats.nPresent[i] : 0;
            const double r = stats.nUnits > 0
                ? std::log(static_cast<double>(stats.nUnits) / static_cast<double>(1 + present))
                : 0.0;
            rawIdf[i] = std::max(0.0, r);
            corpusTotal += i < stats.totalCounts.size() ? stats.totalCounts[i] : 0;
        }
        const double qIdf = percentile(rawIdf, options.idfQ);

        double infoMean = 0.0;
        if (corpusTotal > 0) {
            for (size_t i = 0; i < nFeatures; ++i) {
                const uint64_t total = i < stats.totalCounts.size() ? stats.totalCounts[i] : 0;
                if (total == 0) {
                    rawInfo[i] = std::numeric_limits<double>::infinity();
                    continue;
                }
                rawInfo[i] = std::log2(static_cast<double>(corpusTotal) / static_cast<double>(total));
                infoMean += (static_cast<double>(total) / static_cast<double>(corpusTotal)) * rawInfo[i];
            }
        }
        if (!std::isfinite(infoMean) || infoMean <= 0.0) {
            infoMean = 1.0;
        }

        for (size_t i = 0; i < nFeatures; ++i) {
            double idf = qIdf > 0.0
                ? std::pow(rawIdf[i] / qIdf, options.idfPower)
                : 0.0;
            idf = std::min(1.0, idf);
            idf = options.idfMin + (1.0 - options.idfMin) * idf;
            double infoWeight = std::isfinite(rawInfo[i])
                ? rawInfo[i] / infoMean
                : options.idfMax;
            infoWeight = std::min(options.idfMax, infoWeight);
            out.idf[i] = idf;
            out.infoWeight[i] = infoWeight;
        }
        return out;
    }

    static void writeTsv(const std::string& outFile,
            const std::vector<std::string>& featureNames,
            size_t nFeatures,
            const FeatureOccurrenceStats& stats,
            const FeatureInfoOptions& options = FeatureInfoOptions()) {
        std::ofstream out(outFile);
        if (!out) {
            error("Error opening feature stats file %s for output", outFile.c_str());
        }
        const FeatureInfoWeights weights = computeWeights(nFeatures, stats, options);
        out << "#feature\ttotal_count\tn_units_present\tn_units_count_gt1\tn_units_count_gt2\tn_units_count_gt_mean\tidf\tinfo_weight\n";
        out << std::fixed << std::setprecision(6);
        for (size_t i = 0; i < nFeatures; ++i) {
            const std::string feature = i < featureNames.size() && !featureNames[i].empty()
                ? featureNames[i]
                : std::to_string(i);
            const uint64_t total = i < stats.totalCounts.size() ? stats.totalCounts[i] : 0;
            const uint64_t present = i < stats.nPresent.size() ? stats.nPresent[i] : 0;
            const uint64_t gt1 = i < stats.nGt1.size() ? stats.nGt1[i] : 0;
            const uint64_t gt2 = i < stats.nGt2.size() ? stats.nGt2[i] : 0;
            out << feature << "\t"
                << total << "\t"
                << present << "\t"
                << gt1 << "\t"
                << gt2 << "\t"
                << stats.countGreaterThanMean(i) << "\t"
                << weights.idf[i] << "\t"
                << weights.infoWeight[i] << "\n";
        }
    }
};

class Tiles2Hex {

public:

    Tiles2Hex(int32_t nThreads, std::string& _tmpDir, std::string& _outFile, HexGrid& hexGrid, TileReader& tileReader, lineParser& parser, std::vector<int32_t> _minCounts = {}, int32_t _seed = -1, double _bccSize = -1, FeatureInfoOptions _featureInfoOptions = FeatureInfoOptions());
    ~Tiles2Hex() {
        if (mainOut.is_open()) {
            mainOut.close();
        }
    }

    bool run();

    void writeMetadata();

protected:
    int32_t nThreads;
    std::string outFile;
    ScopedTempDir tmpDir;
    int32_t nModal, nUnits, nFeatures;
    std::vector<int32_t> minCounts;
    nlohmann::json meta;
    bool is3D;

    lineParser parser;
    HexGrid hexGrid;
    BCCGrid bccGrid;
    TileReader tileReader;

    ThreadSafeQueue<TileKey> tileQueue;
    std::vector<std::thread> threads;
    std::mutex mainOutMutex;
    std::ofstream mainOut;
    uint32_t randomSeed = 0;
    uint32_t countHistBinSize = 5;
    std::map<uint32_t, uint64_t> countHist;
    std::vector<FeatureOccurrenceStats> featureOccurrenceByModal;
    FeatureInfoOptions featureInfoOptions;

    // Process one tile at a time
    virtual void worker(int threadId);
    // Merge boundary hexagons from all temporary files and append to mainOut
    virtual bool mergeBoundaryHexagons();
    std::string outputPrefix() const;
    void writeCountHistogram() const;
    void writeFeatureInfoSidecars() const;
    std::vector<std::string> featureNamesForSidecar() const;
    void worker2D(int threadId);
    void worker3D(int threadId);
    bool mergeBoundaryHexagons2D();
    bool mergeBoundaryHexagons3D();

    virtual bool launchWorkerThreads() {
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

    bool passesMinCount(const std::vector<uint32_t>& valsums) const {
        for (size_t i = 0; i < valsums.size(); ++i) {
            if (valsums[i] >= static_cast<uint32_t>(minCounts[i])) {
                return true;
            }
        }
        return false;
    }

    uint64_t totalUnitCount(const std::vector<uint32_t>& valsums) const {
        uint64_t total = 0;
        for (uint32_t v : valsums) {
            total += v;
        }
        return total;
    }

    void recordUnitCount(const std::vector<uint32_t>& valsums) {
        const uint64_t total = totalUnitCount(valsums);
        const uint32_t binStart = static_cast<uint32_t>((total / countHistBinSize) * countHistBinSize);
        countHist[binStart] += 1;
    }

    void recordFeatureOccurrence(const std::vector<std::map<uint32_t, uint32_t>>& vals) {
        if (featureOccurrenceByModal.size() < vals.size()) {
            featureOccurrenceByModal.resize(vals.size());
        }
        for (size_t i = 0; i < vals.size(); ++i) {
            featureOccurrenceByModal[i].observeUnit(vals[i]);
        }
    }

    void addPixelToUnitMaps(PixelValues& pixel, uint32_t hx, uint32_t hy, std::unordered_map<int64_t, UnitValues>& Units, int32_t l = -1) {
        int64_t key = (static_cast<uint64_t>(hx) << 32) | hy;
        auto it = Units.find(key);
        if (it == Units.end()) {
            Units.insert({key, UnitValues(hx, hy, pixel, l)});
        } else {
            it->second.addPixel(pixel);
        }
    }

    void addPixelToUnitMaps(PixelValues3D& pixel, int32_t q1, int32_t q2, int32_t q3, std::unordered_map<BCCGrid::cell_key_t, UnitValues3D, Tuple3Hash>& units, int32_t l = -1) {
        BCCGrid::cell_key_t key = BCCGrid::make_key(q1, q2, q3);
        auto it = units.find(key);
        if (it == units.end()) {
            units.insert({key, UnitValues3D(q1, q2, q3, pixel, l)});
        } else {
            it->second.addPixel(pixel);
        }
    }

    uint32_t makeRandomKey(const UnitValues& unit) const {
        // Deterministic pseudo-random key from seed + unit identity.
        uint64_t x = static_cast<uint64_t>(randomSeed);
        x ^= static_cast<uint64_t>(static_cast<uint32_t>(unit.x)) * 0xD2B74407B1CE6E93ULL;
        x ^= static_cast<uint64_t>(static_cast<uint32_t>(unit.y)) * 0x9E3779B97F4A7C15ULL;
        x ^= static_cast<uint64_t>(static_cast<uint32_t>(unit.label + 1)) * 0xCA5A826395121157ULL;
        return static_cast<uint32_t>(splitmix64(x));
    }

    uint32_t makeRandomKey(const UnitValues3D& unit) const {
        uint64_t x = static_cast<uint64_t>(randomSeed);
        x ^= static_cast<uint64_t>(static_cast<uint32_t>(unit.x)) * 0xD2B74407B1CE6E93ULL;
        x ^= static_cast<uint64_t>(static_cast<uint32_t>(unit.y)) * 0x9E3779B97F4A7C15ULL;
        x ^= static_cast<uint64_t>(static_cast<uint32_t>(unit.z)) * 0x94D049BB133111EBULL;
        x ^= static_cast<uint64_t>(static_cast<uint32_t>(unit.label + 1)) * 0xCA5A826395121157ULL;
        return static_cast<uint32_t>(splitmix64(x));
    }

    void writeUnit(const UnitValues& unit, uint32_t key) {
        recordUnitCount(unit.valsums);
        recordFeatureOccurrence(unit.vals);
        double x, y;
        hexGrid.axial_to_cart(x, y, unit.x, unit.y);
        mainOut << uint32toHex(key) << "\t" << x << "\t" << y;
        if (unit.label >= 0)
            mainOut << "\t" << unit.label;
        for (size_t i = 0; i < unit.vals.size(); ++i) {
            mainOut << "\t" << unit.vals[i].size() << "\t" << unit.valsums[i];
        }
        for (const auto& val : unit.vals) {
            for (const auto& entry : val) {
                mainOut << "\t" << entry.first << " " << entry.second;
            }
        }
        mainOut << "\n";
    }

    void writeUnit(const UnitValues3D& unit, uint32_t key) {
        recordUnitCount(unit.valsums);
        recordFeatureOccurrence(unit.vals);
        double x, y, z;
        bccGrid.lattice_to_cart(x, y, z, unit.x, unit.y, unit.z);
        mainOut << uint32toHex(key) << "\t" << x << "\t" << y << "\t" << z;
        if (unit.label >= 0) {
            mainOut << "\t" << unit.label;
        }
        for (size_t i = 0; i < unit.vals.size(); ++i) {
            mainOut << "\t" << unit.vals[i].size() << "\t" << unit.valsums[i];
        }
        for (const auto& val : unit.vals) {
            for (const auto& entry : val) {
                mainOut << "\t" << entry.first << " " << entry.second;
            }
        }
        mainOut << "\n";
    }

};

class Tiles2UnitsByAnchor : public Tiles2Hex {

public:

    Tiles2UnitsByAnchor(int32_t nThreads, std::string& _tmpDir, std::string& _outFile, HexGrid& hexGrid, TileReader& tileReader, lineParser& parser, std::vector<std::string>& anchorFiles, std::vector<float>& radius, std::vector<int32_t> _minCounts = {}, bool _noBackground = false, int32_t _seed = -1, FeatureInfoOptions _featureInfoOptions = FeatureInfoOptions())
        : Tiles2Hex(nThreads, _tmpDir, _outFile, hexGrid, tileReader, parser, _minCounts, _seed, -1.0, _featureInfoOptions), noBackground(_noBackground) {
        assert(!anchorFiles.empty() && anchorFiles.size() == radius.size());
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
        meta["offset_data"] = 4;
        meta["icol_layer"] = 3;
        meta["header_info"] = {"random_key", "x", "y", "layer"};
    }

protected:

    uint32_t nAnchorSets, nLayer;
    bool noBackground;
    std::vector<uint32_t> nUnitsPerLabel;
    std::vector<PointCloud<float>> anchorPoints;
    std::vector<std::unique_ptr<kd_tree_f2_t>> trees;
    std::vector<float> l2radius;

    void readAnchors(std::string& anchorFile);

    void worker(int threadId) override;
    bool mergeBoundaryHexagons() override;
    bool launchWorkerThreads() override {
        for (int i = 0; i < nThreads; ++i) {
            threads.emplace_back(&Tiles2UnitsByAnchor::worker, this, i);
        }
        std::vector<TileKey> tileList;
        tileReader.getTileList(tileList);
        for (const auto& tile : tileList) {
            tileQueue.push(tile);
        }
        tileQueue.set_done();
        return true;
    }

};
