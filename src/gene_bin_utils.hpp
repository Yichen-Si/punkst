#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "utils.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>

enum class GeneBinMode {
    Adaptive,
    Fixed,
};

struct GeneBinBuildOptions {
    int32_t nGeneBins = 50;
    int32_t icolFeature = 0;
    int32_t icolCount = 1;
    int32_t skipLines = 0;
    uint64_t targetMolecules = 1000000;
    double singletonRatio = 1.0;
    GeneBinMode mode = GeneBinMode::Adaptive;
};

inline GeneBinMode parse_gene_bin_mode(const std::string& mode) {
    if (mode == "adaptive") {
        return GeneBinMode::Adaptive;
    }
    if (mode == "fixed") {
        return GeneBinMode::Fixed;
    }
    error("%s: --gene-bin-mode must be adaptive or fixed", __func__);
    return GeneBinMode::Adaptive;
}

struct GeneBinEntry {
    std::string feature;
    uint64_t count = 0;
    int32_t bin = 0;
};

struct GeneBinInfo {
    std::vector<GeneBinEntry> entries;
    std::unordered_map<std::string, int32_t> featureToBin;
    std::map<int32_t, uint64_t> featuresPerBin;

    GeneBinInfo() = default;
    GeneBinInfo(const std::string& path, int32_t nGeneBins, int32_t icolFeature = 0, int32_t icolCount = 1, int32_t skipLines = 0) {
        GeneBinBuildOptions options;
        options.nGeneBins = nGeneBins;
        options.icolFeature = icolFeature;
        options.icolCount = icolCount;
        options.skipLines = skipLines;
        build_gene_bins_from_tsv(path, options);
    }
    GeneBinInfo(const std::string& path, const GeneBinBuildOptions& options) {
        build_gene_bins_from_tsv(path, options);
    }
    GeneBinInfo(const std::string& path) {
        read_gene_bin_info_json(path);
    }

    void build_gene_bins_from_tsv(const std::string& path, const GeneBinBuildOptions& options) {
        if (options.mode == GeneBinMode::Fixed) {
            read_gene_count_tsv(path, options);
            assign_fixed_gene_bins(options.nGeneBins);
        } else {
            read_gene_count_tsv(path, options);
            assign_adaptive_gene_bins(options.nGeneBins, options.targetMolecules, options.singletonRatio);
        }
        finalize_gene_bin_info();
    }

    void build_equal_gene_bins_from_tsv(const std::string& path, int32_t nGeneBins, int32_t icolFeature = 0, int32_t icolCount = 1, int32_t skipLines = 0) {
        GeneBinBuildOptions options;
        options.nGeneBins = nGeneBins;
        options.icolFeature = icolFeature;
        options.icolCount = icolCount;
        options.skipLines = skipLines;
        options.mode = GeneBinMode::Fixed;
        build_gene_bins_from_tsv(path, options);
    }

    void read_gene_count_tsv(const std::string& path, const GeneBinBuildOptions& options) {
        const int32_t icolFeature = options.icolFeature;
        const int32_t icolCount = options.icolCount;
        const int32_t skipLines = options.skipLines;
        if (icolFeature < 0 || icolCount < 0) {
            error("%s: feature/count column indices must be non-negative", __func__);
        }
        if (icolFeature == icolCount) {
            error("%s: feature/count column indices must be distinct", __func__);
        }
        if (skipLines < 0) {
            error("%s: skipLines must be non-negative", __func__);
        }

        std::ifstream in(path);
        if (!in.is_open()) {
            error("%s: cannot open %s", __func__, path.c_str());
        }

        std::string line;
        int32_t lineNo = 0;
        const int32_t nTok = std::max(icolFeature, icolCount) + 1;
        while (std::getline(in, line)) {
            ++lineNo;
            if (lineNo <= skipLines) {
                continue;
            }
            if (line.empty() || line[0] == '#') {
                continue;
            }
            std::vector<std::string> tokens;
            split(tokens, "\t", line, nTok + 1, true, true, true);
            if (tokens.size() < static_cast<size_t>(nTok)) {
                error("%s: invalid row %d in %s", __func__, lineNo, path.c_str());
            }
            GeneBinEntry entry;
            entry.feature = tokens[icolFeature];
            if (!str2num<uint64_t>(tokens[icolCount], entry.count)) {
                error("%s: invalid count '%s' in row %d of %s", __func__, tokens[icolCount].c_str(), lineNo, path.c_str());
            }
            entries.push_back(std::move(entry));
        }

        if (entries.empty()) {
            error("%s: no feature rows found in %s", __func__, path.c_str());
        }

        std::sort(entries.begin(), entries.end(),
            [](const GeneBinEntry& lhs, const GeneBinEntry& rhs) {
                if (lhs.count != rhs.count) {
                    return lhs.count > rhs.count;
                }
                return lhs.feature < rhs.feature;
            });
    }

    void assign_fixed_gene_bins(int32_t nGeneBins) {
        if (nGeneBins <= 0) {
            error("%s: number of gene bins must be positive", __func__);
        }
        long double totalSum = 0.0L;
        for (const auto& entry : entries) {
            totalSum += static_cast<long double>(entry.count);
        }
        long double targetSum = totalSum / static_cast<long double>(nGeneBins);
        long double currentSum = 0.0L;
        int32_t currentBin = 1;
        for (auto& entry : entries) {
            currentSum += static_cast<long double>(entry.count);
            entry.bin = currentBin;
            if (currentBin < nGeneBins && currentSum >= targetSum) {
                totalSum -= currentSum;
                ++currentBin;
                currentSum = 0.0L;
                if (currentBin <= nGeneBins) {
                    targetSum = (nGeneBins - currentBin + 1) > 0
                        ? totalSum / static_cast<long double>(nGeneBins - currentBin + 1)
                        : 0.0L;
                }
            }
        }
    }

    void assign_adaptive_gene_bins(int32_t maxGeneBins, uint64_t targetMolecules, double singletonRatio) {
        if (maxGeneBins <= 0) {
            error("%s: number of gene bins must be positive", __func__);
        }
        if (singletonRatio < 0.0 || !std::isfinite(singletonRatio)) {
            error("%s: --gene-bin-singleton-ratio must be finite and non-negative", __func__);
        }
        const int32_t maxBins = std::min<int32_t>(maxGeneBins, static_cast<int32_t>(entries.size()));
        if (maxBins <= 0) {
            return;
        }

        long double totalSum = 0.0L;
        for (const auto& entry : entries) {
            totalSum += static_cast<long double>(entry.count);
        }
        if (totalSum <= 0.0L) {
            int32_t bin = 1;
            for (auto& entry : entries) {
                entry.bin = bin;
                bin = bin == maxBins ? 1 : bin + 1;
            }
            return;
        }

        long double target = targetMolecules > 0
            ? static_cast<long double>(targetMolecules)
            : totalSum / static_cast<long double>(maxBins);
        if (target < 1.0L) {
            target = 1.0L;
        }
        struct BinLoad {
            long double molecules = 0.0L;
            int32_t features = 0;
        };
        const long double singletonThreshold = singletonRatio * target;
        size_t singletonCount = 0;
        long double singletonSum = 0.0L;
        if (singletonRatio > 0.0) {
            while (singletonCount < entries.size()
                && static_cast<int32_t>(singletonCount) < maxBins
                && static_cast<long double>(entries[singletonCount].count) >= singletonThreshold) {
                singletonSum += static_cast<long double>(entries[singletonCount].count);
                ++singletonCount;
            }
        }
        const long double remainingSum = std::max<long double>(0.0L, totalSum - singletonSum);
        const int32_t remainingCapacity = maxBins - static_cast<int32_t>(singletonCount);
        int32_t remainingBins = 0;
        if (remainingCapacity > 0 && remainingSum > 0.0L) {
            remainingBins = static_cast<int32_t>(std::ceil(remainingSum / target));
            remainingBins = std::max<int32_t>(1, std::min<int32_t>(remainingCapacity, remainingBins));
        }
        int32_t desiredBins = static_cast<int32_t>(singletonCount) + remainingBins;
        desiredBins = std::max<int32_t>(1, std::min<int32_t>(maxBins, desiredBins));

        std::vector<BinLoad> bins;
        bins.reserve(static_cast<size_t>(desiredBins));

        size_t nextEntry = 0;
        while (nextEntry < singletonCount && static_cast<int32_t>(bins.size()) < desiredBins) {
            entries[nextEntry].bin = static_cast<int32_t>(bins.size()) + 1;
            bins.push_back({static_cast<long double>(entries[nextEntry].count), 1});
            ++nextEntry;
        }
        if (bins.empty()) {
            bins.push_back({});
        }

        for (size_t i = nextEntry; i < entries.size(); ++i) {
            if (entries[i].count > 0 && static_cast<int32_t>(bins.size()) < desiredBins) {
                bins.push_back({});
            }
            size_t best = 0;
            for (size_t b = 1; b < bins.size(); ++b) {
                if (bins[b].molecules < bins[best].molecules
                    || (bins[b].molecules == bins[best].molecules && bins[b].features < bins[best].features)) {
                    best = b;
                }
            }
            entries[i].bin = static_cast<int32_t>(best) + 1;
            bins[best].molecules += static_cast<long double>(entries[i].count);
            bins[best].features += 1;
        }
    }

    void read_gene_bin_info_json(const std::string& path) {
        std::ifstream in(path);
        if (!in.is_open()) {
            error("%s: cannot open %s", __func__, path.c_str());
        }
        nlohmann::json data;
        try {
            in >> data;
        } catch (const std::exception& ex) {
            error("%s: failed to parse %s: %s", __func__, path.c_str(), ex.what());
        }
        if (!data.is_array()) {
            error("%s: expected a JSON array in %s", __func__, path.c_str());
        }

        GeneBinInfo info;
        for (const auto& row : data) {
            if (!row.is_object()) {
                error("%s: expected object rows in %s", __func__, path.c_str());
            }
            if (!row.contains("gene") || !row.contains("count") || !row.contains("bin")) {
                error("%s: each row in %s must contain gene/count/bin", __func__, path.c_str());
            }
            GeneBinEntry entry;
            entry.feature = row.at("gene").get<std::string>();
            entry.count = row.at("count").get<uint64_t>();
            entry.bin = row.at("bin").get<int32_t>();
            entries.push_back(std::move(entry));
        }

        finalize_gene_bin_info();
    }

    void write_gene_bin_info_json(const std::string& path) const {
        nlohmann::json out = nlohmann::json::array();
        for (const auto& entry : entries) {
            nlohmann::json row;
            row["gene"] = entry.feature;
            row["count"] = entry.count;
            row["bin"] = entry.bin;
            out.push_back(std::move(row));
        }
        std::ofstream ofs(path);
        if (!ofs.is_open()) {
            error("%s: cannot open %s for writing", __func__, path.c_str());
        }
        ofs << out.dump();
    }

    void finalize_gene_bin_info() {
        featureToBin.clear();
        featuresPerBin.clear();
        for (const auto& entry : entries) {
            const auto ret = featureToBin.emplace(entry.feature, entry.bin);
            if (!ret.second) {
                error("%s: duplicate feature '%s' in gene-bin info", __func__, entry.feature.c_str());
            }
            featuresPerBin[entry.bin] += 1;
        }
        if (entries.empty()) {
            error("%s: no gene-bin entries found", __func__);
        }
    }

};
