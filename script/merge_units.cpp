/**
 * Unify data units (cells/hexagons) from multiple samples
 */
#include "punkst.h"
#include "utils.h"
#include "dataunits.hpp"
#include "json.hpp"
#include <algorithm>
#include <unordered_map>
#include <thread>
#include <atomic>

/**
 * Represents the input data and options for a single sample.
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
 * Parses a comma-separated string of non-negative integers into a vector.
 * Return A vector of integers. Invalid entries are skipped.
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
 * Worker function to process samples in parallel.
 * Each worker thread pulls a sample index, processes the entire sample,
 * and writes its output to a dedicated temporary file.
 */
void merge_worker_to_tempfile(
    uint32_t thread_id,
    std::filesystem::path temp_file_path,
    std::atomic<size_t>& sample_idx_atomic,
    const std::vector<SampleInput>& samples,
    const std::unordered_map<std::string, uint32_t>& shared_feature_dict,
    std::atomic<int32_t>& nUnits_atomic,
    int minCtPerUnit,
    bool hasInfoCols)
{
    // Open a dedicated temporary file for this thread's output
    std::ofstream temp_out(temp_file_path);
    if (!temp_out) {
        // Can't use error() here as it calls exit().
        warning("Worker thread %d could not open temporary file: %s", thread_id, temp_file_path.c_str());
        return;
    }

    // Each thread gets its own random number generator, seeded for uniqueness
    std::mt19937 rng(std::random_device{}() + thread_id);
    std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);

    size_t s;
    // Atomically fetch the next sample index to process
    while ((s = sample_idx_atomic.fetch_add(1)) < samples.size()) {
        const auto& currentSample = samples[s];

        HexReader reader(currentSample.metaFile);
        std::unordered_map<uint32_t, uint32_t> idx_remap;
        for (size_t i = 0; i < reader.features.size(); ++i) {
            auto it = shared_feature_dict.find(reader.features[i]);
            if (it != shared_feature_dict.end()) {
                idx_remap[i] = it->second;
            }
        }

        std::ifstream inFileStream(currentSample.hexFile);
        if (!inFileStream) {
            warning("Worker thread %d could not open hexagon file: %s", thread_id, currentSample.hexFile.c_str());
            continue;
        }

        int32_t offset_data = reader.getOffset();
        if (offset_data < 0) {
            warning("offset_data not in metadata for %s", currentSample.id.c_str());
            continue;
        }

        int32_t minTokens = std::max((int32_t) currentSample.maxIndex(), offset_data + 2);
        int32_t nModal = reader.getNmodal();
        int32_t keyCol = currentSample.keyCol == -2 ? reader.getIndex("random_key") : currentSample.keyCol;

        std::string line;
        while (std::getline(inFileStream, line)) {
            std::vector<std::string> tokens;
            split(tokens, "\t", line);
            if (tokens.size() < minTokens) continue;

            std::string key_str = (keyCol >= 0) ? tokens[keyCol] : uint32toHex(rdUnif(rng));

            std::stringstream ss_info;
            if (hasInfoCols) {
                if (currentSample.infoCols.empty()) {
                    ss_info << ".";
                } else {
                    ss_info << tokens[currentSample.infoCols[0]];
                    for (size_t i = 1; i < currentSample.infoCols.size(); ++i) {
                        ss_info << "," << tokens[currentSample.infoCols[i]];
                    }
                }
            }

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

            temp_out << key_str << "\t" << s << "\t";
            if (hasInfoCols) {
                temp_out << ss_info.str() << "\t";
            }
            temp_out << n_new_features << "\t" << total_new_count << ss_remapped.str() << "\n";
            nUnits_atomic.fetch_add(1, std::memory_order_relaxed);
        }
    }
    temp_out.close();
}

/**
 * Merges multiple datasets with a shared feature dictionary
 * This function reads a list of samples, determines a set of features, and
 * creates a single merged data file with unified feature indices and metadata.
 */
int32_t cmdMergeUnits(int32_t argc, char** argv) {
    std::string inListFile, outPref, tmpDirPath;
    int minTotalCountPerSample = 1;
    int minCtPerUnit = 1;
    uint32_t nThreads = 1;

    // 1. Define and parse command-line options
    ParamList pl;
    pl.add_option("in-list", "Input TSV file with sample info (ID, feature_path, hex_path. Optional: key_col_idx, info_col_indices)", inListFile, true)
        .add_option("out-pref", "Prefix for output files", outPref, true)
        .add_option("temp-dir", "Directory to store temporary files", tmpDirPath, true)
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
        std::string line;
        std::vector<std::string> tokens;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            split(tokens, "\t", line);
            if (tokens.size() < 2) continue; // Skip invalid lines
            feature = tokens[0];
            if (!str2num<uint64_t>(tokens[1], cnt)) {
                warning("Skip invalid feature line: %s", line.c_str());
                continue;
            }
            auto it = featCounts.find(feature);
            if (it == featCounts.end()) {
                std::vector<uint64_t> vec(S + 1, 0); // S samples + 1 for total
                vec[s] = cnt;
                vec[S] = cnt;
                featCounts.emplace(feature, vec);
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

    // 4. Merge hexagon files in parallel to temporary files
    notice("Processing %zu samples with %u threads...", S, nThreads);
    ScopedTempDir temp_dir(tmpDirPath);
    notice("Using temporary directory: %s", temp_dir.path.c_str());

    std::vector<std::thread> workers;
    std::atomic<size_t> sample_idx(0);
    std::atomic<int32_t> nUnits(0);
    std::vector<std::string> temp_files;

    for (uint32_t i = 0; i < nThreads; ++i) {
        std::filesystem::path temp_file_path = temp_dir.path / ("part_" + std::to_string(i));
        workers.emplace_back(merge_worker_to_tempfile, i,
                             temp_file_path,
                             std::ref(sample_idx),
                             std::cref(samples),
                             std::cref(shared_feature_dict),
                             std::ref(nUnits),
                             minCtPerUnit,
                             hasInfoCols);
        temp_files.push_back(temp_file_path.string());
    }

    for (auto& t : workers) {
        t.join();
    }

    notice("Concatenating and sorting %zu temporary files...", temp_files.size());
    std::string mergedFile = outPref + ".txt";
    if (pipe_cat_sort(temp_files, mergedFile, nThreads) != 0) {
        return 1;
    }
    notice("Merged %d units across %zu samples.", nUnits.load(), S);

    // 5. Write merged JSON metadata
    nlohmann::json mergedMeta;
    mergedMeta["n_units"] = nUnits.load();
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

    notice("Finished merging. Output at: %s", mergedFile.c_str());
    return 0;
}
