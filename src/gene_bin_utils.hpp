#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "json.hpp"
#include "utils.h"
#include <algorithm>
#include <fstream>

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
        build_equal_gene_bins_from_tsv(path, nGeneBins, icolFeature, icolCount, skipLines);
    }
    GeneBinInfo(const std::string& path) {
        read_gene_bin_info_json(path);
    }

    void build_equal_gene_bins_from_tsv(const std::string& path, int32_t nGeneBins, int32_t icolFeature = 0, int32_t icolCount = 1, int32_t skipLines = 0) {
        if (icolFeature < 0 || icolCount < 0) {
            error("%s: feature/count column indices must be non-negative", __func__);
        }
        if (icolFeature == icolCount) {
            error("%s: feature/count column indices must be distinct", __func__);
        }
        if (nGeneBins <= 0) {
            error("%s: number of gene bins must be positive", __func__);
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
        finalize_gene_bin_info();
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
