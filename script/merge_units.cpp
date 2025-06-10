/**
 * Unify data units (cells/hexagons) from multiple samples
 */
#include "punkst.h"
#include "utils.h"
#include "dataunits.hpp"
#include "json.hpp"
#include <algorithm>
#include <unordered_map>

/**
 * @brief Represents the input data and options for a single sample.
 */
struct SampleInput {
    std::string id;
    std::string featureFile;
    std::string hexFile;
    std::string metaFile;
    int32_t keyCol = -2; // Column index for the random key. -1 means generate a new one, -2 for finding the key column from the json file ("random_key")
    std::vector<uint32_t> infoCols; // Column indices for additional info to carry over.
    uint32_t maxIndex() const {
        uint32_t maxIdx = 0;
        for (auto& idx : infoCols) {
            if (idx > maxIdx) {
                maxIdx = idx;
            }
        }
        if (keyCol > 0) {
            return std::max(maxIdx, static_cast<uint32_t>(keyCol));
        }
        return maxIdx;
    }
};

/**
 * @brief Parses a comma-separated string of non-negative integers into a vector.
 * @param s The string to parse.
 * @return A vector of integers. Invalid entries are skipped.
 */
std::vector<uint32_t> parseIntList(std::string& s) {
    std::vector<uint32_t> result;
    std::vector<std::string> tokens;
    if (s.empty() || s == ".") return result;
    split(tokens, ",", s, UINT_MAX, true, true, true);
    if (tokens.empty()) {
        return result;
    }
    for (auto& token : tokens) {
        uint32_t val;
        if (!str2num<uint32_t>(token, val)) {
            error("Could not parse indices from info-column list '%s': %s", s.c_str(), token.c_str());
        }
        result.push_back(val);
    }
    return result;
}

/**
 * @brief Merges multiple datasets with a shared feature dictionary
 * This function reads a list of samples, determines a set of features, and
 * creates a single merged data file with unified feature indices and metadata.
 */
int32_t cmdMergeUnits(int32_t argc, char** argv) {
    std::string inListFile, outPref;
    int minTotalCountPerSample = 1;
    int minCtPerUnit = 1;
    uint32_t nThreads = 1;

    // 1. Define and parse command-line options
    ParamList pl;
    pl.add_option("in-list", "Input TSV file with sample info (ID, feature_path, hex_path. Optional: key_col_idx, info_col_indices)", inListFile, true)
        .add_option("out-pref", "Prefix for output files", outPref, true)
        .add_option("min-total-count-per-sample", "Minimum total gene count per sample (default: 1) to include in the joint model", minTotalCountPerSample)
        .add_option("min-count-per-unit", "Minimum total count per unit to be included in merged output", minCtPerUnit)
        .add_option("threads", "Number of threads for sorting", nThreads);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        error("Error parsing options for cmdMergeUnits: %s", ex.what());
        pl.print_help();
        return 1;
    }

    // 2. Parse the input list file to get sample information
    std::vector<SampleInput> samples;
    std::unordered_map<std::string, uint32_t> sampleNameToIndex;
    bool hasInfoCols = false;
    {
        std::ifstream inFile(inListFile);
        if (!inFile) {
            error("Cannot open input list file: %s", inListFile.c_str());
        }
        std::string line;
        while(std::getline(inFile, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", line);
            if (tokens.size() < 4) {
                warning("Skipping invalid line in input list (must have at least 3 columns): %s", line.c_str());
                continue;
            }
            SampleInput s;
            uint32_t i = 0;
            s.id = tokens[i++];
            s.featureFile = tokens[i++];
            s.hexFile = tokens[i++];
            s.metaFile = tokens[i++];
            if (tokens.size() > i) {
                if (!str2num<int32_t>(tokens[i++], s.keyCol))
                    error("Invalid key column index: %s", tokens[i-1].c_str());
            }
            if (tokens.size() > i) {
                s.infoCols = parseIntList(tokens[i++]);
                hasInfoCols = true;
            }
            sampleNameToIndex[s.id] = samples.size();
            samples.push_back(s);
        }
    }
    size_t S = samples.size();
    if (S == 0) {
        error("No valid samples found in input list file.");
    }
    notice("Found %zu samples to merge.", S);

    // 3. Decide shared feature dictionary
    std::unordered_map<std::string, std::vector<uint64_t>> featCounts;
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> unionFeatures; // feature, (min_count_across_samples, total_count)
    for (size_t s = 0; s < S; ++s) {
        std::ifstream fin(samples[s].featureFile);
        if (!fin) error("Cannot open feature file: %s", samples[s].featureFile.c_str());
        std::string feature;
        uint64_t cnt;
        while (fin >> feature >> cnt) {
            auto it = featCounts.find(feature);
            if (it == featCounts.end()) {
                std::vector<uint64_t> vec(S + 1, 0); // S samples + 1 for total
                vec[s] = cnt;
                vec[S] = cnt;
                featCounts.emplace(feature, std::move(vec));
            } else {
                it->second[s] = cnt;
                it->second[S] += cnt;
            }
        }
    }
    unionFeatures.reserve(featCounts.size());
    for (auto &p : featCounts) {
        uint64_t min_count_across_samples = p.second[0];
        for(size_t i = 1; i < S; ++i) {
            if (p.second[i] < min_count_across_samples) {
                min_count_across_samples = p.second[i];
            }
        }
        unionFeatures.emplace_back(p.first, std::make_pair(min_count_across_samples, p.second[S]));
    }
    // Sort features by min count (desc), then total count (desc)
    std::sort(unionFeatures.begin(), unionFeatures.end(),
              [](const auto &a, const auto &b) {
                  if (a.second.first != b.second.first) return a.second.first > b.second.first;
                  return a.second.second > b.second.second;
              });
    { // Write feature lists
        std::string unionFile = outPref + ".union_features.tsv";
        std::ofstream unionOut(unionFile);
        std::string sharedFile = outPref + ".features.tsv";
        std::ofstream sharedOut(sharedFile);
        if (!sharedOut || !unionOut) error("Cannot write shared or union feature file: %s", unionFile.c_str());
        // header
        unionOut << "#feature\ttotal_count";
        for (auto &sn : samples) unionOut << "\t" << sn.id;
        unionOut << "\n";
        sharedOut << "#feature\ttotal_count\n";
        for (auto &feat : unionFeatures) {
            const auto &vec = featCounts[feat.first];
            unionOut << feat.first << "\t" << feat.second.second;
            for (size_t s = 0; s < S; ++s) {
                unionOut << "\t" << vec[s];
            }
            unionOut << "\n";
            if (vec[S] >= minTotalCountPerSample) {
                sharedOut << feat.first << "\t" << feat.second.second << "\n";
            }
        }
    }

    std::unordered_map<std::string, uint32_t> shared_feature_dict;
    for(const auto& feat : unionFeatures) {
        if (feat.second.first < minTotalCountPerSample) {
            break;
        }
        shared_feature_dict[feat.first] = shared_feature_dict.size();
    }
    size_t n_shared = shared_feature_dict.size();
    notice("Selected %zu features from a total of %zu union features.", n_shared, featCounts.size());

    // 4. Merge hexagon files
    std::string mergedFile = outPref + ".txt";
    std::ofstream mergedOut(mergedFile);
    if (!mergedOut) error("Cannot open merged output file: %s", mergedFile.c_str());

    int32_t nUnits = 0;
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);

    for (size_t s = 0; s < S; ++s) {
        const auto& currentSample = samples[s];

        HexReader reader(currentSample.metaFile);

        // Create a map from this sample's original feature index to the new shared index
        std::unordered_map<uint32_t, uint32_t> idx_remap;
        for (size_t i = 0; i < reader.features.size(); ++i) {
            auto it = shared_feature_dict.find(reader.features[i]);
            if (it != shared_feature_dict.end()) {
                idx_remap[i] = it->second;
            }
        }

        std::ifstream inFileStream(currentSample.hexFile);
        if (!inFileStream) error("Cannot open hexagon file: %s", currentSample.hexFile.c_str());

        int32_t offset_data = reader.getOffset();
        if (offset_data < 0) error("offset_data not in metadata for %s", currentSample.id.c_str());
        int32_t minTokens = std::max((int32_t) currentSample.maxIndex(), offset_data + 2);

        int32_t nModal = reader.getNmodal();
        int32_t keyCol = currentSample.keyCol;
        if (keyCol < -1) {
            keyCol = reader.getIndex("random_key");
        }
        notice("Processing sample %s (%zu/%zu) (key col %d, %d info fields)...", samples[s].id.c_str(), s + 1, S, keyCol, currentSample.infoCols.size());

        std::string line;
        while (std::getline(inFileStream, line)) {
            std::vector<std::string> tokens;
            split(tokens, "\t", line);
            if (tokens.size() < minTokens) continue;

            // Get random key, either from file or by generating a new one
            std::string key_str = (keyCol >= 0)
                                ? tokens[keyCol] : uint32toHex(rdUnif(rng));

            // Concatenate specified info fields
            std::stringstream ss_info;
            if (hasInfoCols) {
                if (currentSample.infoCols.empty()) {
                    ss_info << ".";
                } else {
                    ss_info << tokens[currentSample.infoCols[0]];
                    for (size_t i = 1; i < currentSample.infoCols.size(); ++i) {
                        int32_t col_idx = currentSample.infoCols[i];
                        ss_info << "," << tokens[col_idx];
                    }
                }
            }
            // Remap feature indices and counts
            std::stringstream ss_remapped;
            int32_t n_new_features = 0;
            uint32_t total_new_count = 0;
            int32_t data_start_idx = offset_data + (2 * nModal);
            for (size_t i = data_start_idx; i < tokens.size(); ++i) {
                std::vector<std::string> pair;
                split(pair, " ", tokens[i]);
                if (pair.size() != 2) continue;
                try {
                    uint32_t old_idx = std::stoul(pair[0]);
                    uint32_t count = std::stoul(pair[1]);
                    auto it = idx_remap.find(old_idx);
                    if (it != idx_remap.end()) {
                        ss_remapped << "\t" << it->second << " " << count;
                        n_new_features++;
                        total_new_count += count;
                    }
                } catch(...) { continue; }
            }
            if (total_new_count < minCtPerUnit) continue;

            // Write the unified line to the merged file
            mergedOut << key_str << "\t" << s << "\t";
            if (hasInfoCols) {
                mergedOut << ss_info.str() << "\t";
            }
            mergedOut << n_new_features << "\t" << total_new_count
                      << ss_remapped.str() << "\n";
            nUnits++;
        }
    }
    mergedOut.close();

    // 5. Write merged JSON metadata
    nlohmann::json mergedMeta;
    mergedMeta["n_units"] = nUnits;
    mergedMeta["n_features"] = n_shared;
    mergedMeta["dictionary"] = shared_feature_dict;
    std::vector<std::string> header_info{"random_key", "sample_idx"};
    if (hasInfoCols) {header_info.push_back("info");}
    mergedMeta["header_info"] = header_info;
    mergedMeta["offset_data"] = hasInfoCols ? 3 : 2;
    std::string mergedJsonPath = outPref + ".json";
    std::ofstream mergedJsonOut(mergedJsonPath);
    mergedJsonOut << std::setw(4) << mergedMeta << std::endl;
    mergedJsonOut.close();

    // 6. Sort the merged file for efficient downstream processing
    notice("Sorting the merged file...");
    if (sys_sort(mergedFile.c_str(), nullptr, {"-k1,1", "--parallel="+std::to_string(nThreads), "-o", mergedFile}) != 0) {
        error("Error sorting the merged hexagon file: %s", mergedFile.c_str());
    }

    notice("Finished merging. Output at: %s", mergedFile.c_str());
    return 0;
}
