#include "punkst.h"
#include "vst.hpp"
#include "threads.hpp"
#include "utils.h"
#include "tiles2bins.hpp"

#include <cmath>
#include <functional>
#include <numeric>
#include <regex>
#include <unordered_map>
#include <unordered_set>

namespace {

void validateFeatureInfoOptions(const FeatureInfoOptions& opts) {
    if (opts.idfQ < 0.0 || opts.idfQ > 100.0) {
        error("--idf-q must be between 0 and 100");
    }
    if (opts.idfPower <= 0.0) {
        error("--idf-power must be positive");
    }
    if (opts.idfMin < 0.0 || opts.idfMin > 1.0) {
        error("--idf-min must be between 0 and 1");
    }
    if (opts.idfMax <= 0.0) {
        error("--idf-max must be positive");
    }
}

void observeFeatureStats(FeatureOccurrenceStats& stats, const Document& doc, size_t nFeatures) {
    stats.nUnits += 1;
    for (size_t i = 0; i < doc.ids.size(); ++i) {
        if (doc.ids[i] >= nFeatures) {
            continue;
        }
        const uint32_t feature = doc.ids[i];
        const double value = doc.cnts[i];
        if (value <= 0.0) {
            continue;
        }
        const uint64_t count = static_cast<uint64_t>(std::llround(value));
        stats.ensureFeature(feature);
        stats.totalCounts[feature] += count;
        stats.nPresent[feature] += 1;
        if (value > 1.0) {
            stats.nGt1[feature] += 1;
        }
        if (value > 2.0) {
            stats.nGt2[feature] += 1;
        }
        stats.countHist[feature][static_cast<uint32_t>(count)] += 1;
    }
}

void truncateDocsForDebug(std::vector<Document>& docs, int32_t debug) {
    if (debug > 0 && docs.size() > static_cast<size_t>(debug)) {
        docs.resize(static_cast<size_t>(debug));
    }
}

std::vector<uint8_t> buildBaseEligibility(HexReader& reader,
        const std::string& featureFile,
        const std::string& includeRegex,
        const std::string& excludeRegex) {
    const size_t M = reader.features.size();
    std::vector<uint8_t> eligible(M, featureFile.empty() ? 1u : 0u);

    if (!featureFile.empty()) {
        std::ifstream inFeature(featureFile);
        if (!inFeature) {
            error("Error opening features file: %s", featureFile.c_str());
        }
        std::unordered_map<std::string, uint32_t> dict;
        const bool hasDict = reader.featureDict(dict);
        std::unordered_set<std::string> seen;
        std::vector<std::string> tokens;
        std::string line;
        uint32_t idx0 = 0;
        while (std::getline(inFeature, line)) {
            std::string_view stripped = strip_str(line);
            if (stripped.empty() || stripped.front() == '#') {
                continue;
            }
            split(tokens, "\t ", stripped, UINT_MAX, true, true, true);
            if (tokens.empty()) {
                error("Error reading feature file at line: %s", line.c_str());
            }
            uint32_t idx = idx0++;
            const std::string& feature = tokens[0];
            if (hasDict) {
                auto it = dict.find(feature);
                if (it == dict.end()) {
                    continue;
                }
                idx = it->second;
            }
            if (idx >= M || !seen.insert(feature).second) {
                continue;
            }
            eligible[idx] = 1u;
        }
    }

    const bool checkInclude = !includeRegex.empty();
    const bool checkExclude = !excludeRegex.empty();
    std::regex regexInclude(includeRegex);
    std::regex regexExclude(excludeRegex);
    for (size_t i = 0; i < M; ++i) {
        if (!eligible[i]) {
            continue;
        }
        const std::string& feature = reader.features[i];
        const bool include = !checkInclude || std::regex_match(feature, regexInclude);
        const bool exclude = checkExclude && std::regex_match(feature, regexExclude);
        eligible[i] = (include && !exclude) ? 1u : 0u;
    }
    return eligible;
}

std::vector<uint8_t> applyCountEligibility(const std::vector<uint8_t>& baseEligibility, const FeatureOccurrenceStats& featureStats, int32_t minCountFeature) {
    std::vector<uint8_t> eligible = baseEligibility;
    if (minCountFeature > 0) {
        for (size_t i = 0; i < eligible.size(); ++i) {
            const uint64_t total = i < featureStats.totalCounts.size() ? featureStats.totalCounts[i] : 0;
            if (total < static_cast<uint64_t>(minCountFeature)) {
                eligible[i] = 0u;
            }
        }
    }
    const size_t nEligible = std::accumulate(eligible.begin(), eligible.end(), size_t{0});
    if (nEligible == 0) {
        error("No features remain eligible for HVF selection; check --features, regex filters, and --min-count-per-feature");
    }
    notice("feature-vst: %zu features are eligible for HVF selection out of %zu", nEligible, eligible.size());
    return eligible;
}

void loadDgeDocs(DGEReader10X& dge, std::vector<Document>& docs,
        int32_t minCountTrain, int32_t debug) {
    std::vector<int32_t> unit_idx;
    dge.readAll(docs, unit_idx, minCountTrain);
    truncateDocsForDebug(docs, debug);
}

void runInMemory(HexReader& reader, DGEReader10X* dge, bool use10x,
        const std::string& inFile, int32_t minCountTrain, int32_t debug,
        std::vector<Document>& docs) {
    if (use10x) {
        loadDgeDocs(*dge, docs, minCountTrain, debug);
    } else {
        reader.readAll(docs, inFile, minCountTrain, false, debug, 0);
    }
}

using DocConsumer = std::function<void(const Document&)>;

void streamHexDocs(HexReader& reader, const std::string& inFile,
        int32_t minCountTrain, int32_t debug, const DocConsumer& consume) {
    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("Fail to open input file: %s", inFile.c_str());
    }
    std::string line, info;
    int32_t kept = 0;
    while (std::getline(inFileStream, line)) {
        Document d;
        info.clear();
        int32_t ct = reader.parseLine(d, info, line);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        if (ct < minCountTrain) {
            continue;
        }
        consume(d);
        ++kept;
        if (debug > 0 && kept >= debug) {
            break;
        }
    }
}

void streamDgeDocs(DGEReader10X& dge, int32_t minCountTrain, int32_t debug,
        const DocConsumer& consume) {
    dge.resetStream();
    Document doc;
    int32_t unit_idx = -1;
    int32_t last_unit_idx = -1;
    int32_t kept = 0;
    while (dge.next(doc, &unit_idx, nullptr)) {
        if (unit_idx <= last_unit_idx) {
            error("10X DGE matrix entries must be sorted by barcode for --streaming; saw document id %d after %d",
                unit_idx, last_unit_idx);
        }
        last_unit_idx = unit_idx;
        if (doc.get_raw_sum() < minCountTrain) {
            continue;
        }
        consume(doc);
        ++kept;
        if (debug > 0 && kept >= debug) {
            break;
        }
    }
}

void streamDocs(HexReader& reader, DGEReader10X* dge, bool use10x,
        const std::string& inFile, int32_t minCountTrain, int32_t debug,
        const DocConsumer& consume) {
    if (use10x) {
        streamDgeDocs(*dge, minCountTrain, debug, consume);
    } else {
        streamHexDocs(reader, inFile, minCountTrain, debug, consume);
    }
}

void scanStreamingFeatureSums(HexReader& reader, DGEReader10X* dge, bool use10x,
        const std::string& inFile, int32_t minCountTrain, int32_t debug,
        std::vector<double>& sums) {
    sums.assign(reader.features.size(), 0.0);
    auto consume = [&](const Document& doc) {
        for (size_t i = 0; i < doc.ids.size(); ++i) {
            if (doc.ids[i] < sums.size()) {
                sums[doc.ids[i]] += doc.cnts[i];
            }
        }
    };
    streamDocs(reader, dge, use10x, inFile, minCountTrain, debug, consume);
}

std::vector<uint32_t> runStreamingVst(HexReader& reader, DGEReader10X* dge,
        bool use10x, const std::string& inFile, int32_t minCountTrain,
        int32_t debug, HVF_VST& vst, FeatureOccurrenceStats& featureStats,
        const std::vector<uint8_t>& baseEligibility, int32_t minCountFeature) {
    const uint32_t M = static_cast<uint32_t>(reader.nFeatures);
    vst.beginPass1(M);
    featureStats = FeatureOccurrenceStats();
    streamDocs(reader, dge, use10x, inFile, minCountTrain, debug,
        [&](const Document& doc) {
            vst.observePass1(doc, M);
            observeFeatureStats(featureStats, doc, M);
        });
    featureStats.ensureFeatureCount(M);
    const std::vector<uint8_t> eligible =
        applyCountEligibility(baseEligibility, featureStats, minCountFeature);
    if (vst.finishPass1(M, &eligible) == 0) {
        return vst.rankFeatures(M, &eligible);
    }
    streamDocs(reader, dge, use10x, inFile, minCountTrain, debug,
        [&](const Document& doc) {
            vst.observePass2(doc, M);
        });
    vst.finishPass2(M);
    return vst.rankFeatures(M, &eligible);
}

void computeFeatureStatsFromDocs(const std::vector<Document>& docs,
        size_t M, FeatureOccurrenceStats& featureStats) {
    featureStats = FeatureOccurrenceStats();
    for (const auto& doc : docs) {
        observeFeatureStats(featureStats, doc, M);
    }
    featureStats.ensureFeatureCount(M);
}

void writeFeatureSelectionOutputs(const std::string& outPrefix,
        const std::vector<uint32_t>& order,
        const HVF_VST& vst,
        const FeatureOccurrenceStats& featureStats,
        const FeatureInfoOptions& featureInfoOptions,
        const std::vector<std::string>& featureNames,
        size_t top_k) {
    std::string outf_stats = outPrefix + ".feature.stats.tsv";
    std::string outf_hvf = outPrefix + ".hvf.tsv";
    std::ofstream ofs(outf_stats);
    std::ofstream ots(outf_hvf);
    if (!ofs.is_open()) {
        error("Cannot open output file: %s", outf_stats.c_str());
    }
    if (!ots.is_open()) {
        error("Cannot open output file: %s", outf_hvf.c_str());
    }
    const size_t M = featureNames.size();
    const FeatureInfoWeights weights =
        FeatureOccurrenceStats::computeWeights(M, featureStats, featureInfoOptions);
    ofs << "#Feature\tMean\tVar\tVarExpected\tVarStd\tTotalCount"
        << "\tn_units_present\tn_units_count_gt1\tn_units_count_gt2"
        << "\tn_units_count_gt_mean\tidf\tinfo_weight\n";
    ots << "#Feature\tTotalCount\tScore\n";
    for (size_t j = 0; j < M; ++j) {
        const std::string& name = featureNames[j];
        const uint64_t total = j < featureStats.totalCounts.size() ? featureStats.totalCounts[j] : 0;
        const uint64_t present = j < featureStats.nPresent.size() ? featureStats.nPresent[j] : 0;
        const uint64_t gt1 = j < featureStats.nGt1.size() ? featureStats.nGt1[j] : 0;
        const uint64_t gt2 = j < featureStats.nGt2.size() ? featureStats.nGt2[j] : 0;
        ofs << name << "\t"
            << fp_to_string(vst.stats.mean_all[j], 4) << "\t"
            << fp_to_string(vst.stats.var_all[j], 4) << "\t"
            << fp_to_string(vst.stats.var_expected[j], 4) << "\t"
            << fp_to_string(vst.stats.var_standardized[j], 4) << "\t"
            << total << "\t"
            << present << "\t"
            << gt1 << "\t"
            << gt2 << "\t"
            << featureStats.countGreaterThanMean(j) << "\t"
            << fp_to_string(weights.idf[j], 6) << "\t"
            << fp_to_string(weights.infoWeight[j], 6) << "\n";
    }
    for (size_t t = 0; t < order.size(); ++t) {
        uint32_t j = order[t];
        if (j >= featureNames.size()) {
            continue;
        }
        const std::string& name = featureNames[j];
        const uint64_t total = j < featureStats.totalCounts.size() ? featureStats.totalCounts[j] : 0;
        if (top_k == 0 || t < top_k) {
            ots << name << "\t"
                << total << "\t"
                << fp_to_string(vst.stats.var_standardized[j], 6) << "\n";
        }
        if (top_k > 0 && t + 1 >= top_k) {
            break;
        }
    }
    notice("Wrote VST stats to %s", outf_stats.c_str());
    notice("Wrote HVF list to %s", outf_hvf.c_str());
}

} // namespace

int32_t cmdFeatureVst(int32_t argc, char** argv) {
    std::string inFile, metaFile, featureFile, outPrefix;
    std::vector<std::string> dge_dirs, in_bc, in_ft, in_mtx, dataset_ids;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t minCountTrain = 0;
    int32_t minCountFeature = 0;
    int32_t nThreads = 1;
    size_t top_k = 0;
    double loess_span = 0.30;
    double clip_max = -1.0;
    int32_t debug_ = 0;
    bool streaming = false;
    FeatureInfoOptions featureInfoOptions;

    ParamList pl;
    pl.add_option("in-data", "Input hex file (tsv)", inFile)
      .add_option("in-meta", "Metadata file (json)", metaFile)
      .add_option("features", "Feature names (and total counts) file (optional)", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included", minCountFeature)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex)
      .add_option("min-count-train", "Minimum total count per doc", minCountTrain)
      .add_option("threads", "Number of threads", nThreads);
    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", dataset_ids);
    pl.add_option("loess-span", "LOESS span for mean-variance fit", loess_span)
      .add_option("clip-max", "Max |z| for standardization; -1=sqrt(N)", clip_max)
      .add_option("n-top", "Number of HVFs to output (0=all)", top_k)
      .add_option("streaming", "Use two-pass streaming VST instead of loading all docs", streaming)
      .add_option("idf-q", "Percentile of raw IDF scores used to normalize capped IDF weights", featureInfoOptions.idfQ)
      .add_option("idf-power", "Power gamma for capped IDF weights", featureInfoOptions.idfPower)
      .add_option("idf-min", "Minimum/floor value for capped IDF weights", featureInfoOptions.idfMin)
      .add_option("idf-max", "Maximum/cap value for information-content weights", featureInfoOptions.idfMax);
    pl.add_option("out-prefix", "Output prefix", outPrefix, true)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }
    validateFeatureInfoOptions(featureInfoOptions);
    Threads thr(nThreads);

    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    const bool use_10x = initHexOrDgeInput(reader, dge_ptr, inFile, metaFile,
        dge_dirs, in_bc, in_ft, in_mtx, dataset_ids);
    const std::vector<uint8_t> baseEligibility =
        buildBaseEligibility(reader, featureFile, include_ftr_regex, exclude_ftr_regex);

    HVF_VST vst(loess_span, clip_max);
    std::vector<uint32_t> order;
    FeatureOccurrenceStats featureStats;

    if (streaming) {
        order = runStreamingVst(reader, dge_ptr.get(), use_10x, inFile,
            minCountTrain, debug_, vst, featureStats, baseEligibility, minCountFeature);
    } else {
        std::vector<Document> docs;
        runInMemory(reader, dge_ptr.get(), use_10x, inFile, minCountTrain, debug_, docs);
        notice("Read %lu units with %d features", docs.size(), reader.nFeatures);
        computeFeatureStatsFromDocs(docs, static_cast<size_t>(reader.nFeatures), featureStats);
        const std::vector<uint8_t> eligible =
            applyCountEligibility(baseEligibility, featureStats, minCountFeature);
        order = vst.SelectVST(docs, static_cast<uint32_t>(reader.nFeatures), nullptr, &eligible);
    }

    writeFeatureSelectionOutputs(outPrefix, order, vst, featureStats,
        featureInfoOptions, reader.features, top_k);
    return 0;
}
