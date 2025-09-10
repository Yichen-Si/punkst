#include "punkst.h"
#include "vst.hpp"
#include "threads.hpp"
#include "utils.h"

int32_t cmdFeatureVst(int32_t argc, char** argv) {

    // Inputs
    std::string inFile, metaFile, featureFile, outPrefix;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t minCountTrain = 0;   // filter units by total count
    int32_t minCountFeature = 0; // filter features (requires --features)
    int32_t nThreads = 1;
    size_t top_k = 0; // number of HVFs to write; 0 => all
    // VST params
    double loess_span = 0.30;
    double clip_max   = -1.0;       // default: sqrt(N)
    int32_t debug_ = 0;

    ParamList pl;
    // Input Options
    pl.add_option("in-data", "Input hex file (tsv)", inFile, true)
      .add_option("in-meta", "Metadata file (json)", metaFile, true)
      .add_option("features", "Feature names (and total counts) file (optional)", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", minCountFeature)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex)
      .add_option("min-count-train", "Minimum total count per doc", minCountTrain)
      .add_option("threads", "Number of threads", nThreads);
    // VST options
    pl.add_option("loess-span", "LOESS span for mean-variance fit", loess_span)
      .add_option("clip-max", "Max |z| for standardization; -1=sqrt(N)", clip_max)
      .add_option("n-top", "Number of HVFs to output (0=all)", top_k);
    // Output Options
    pl.add_option("out-prefix", "Output prefix", outPrefix, true)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }
    Threads thr(nThreads);

    // Load metadata and (optionally) feature filter
    HexReader reader(metaFile);
    if (!featureFile.empty()) {
        reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
    }
    const int32_t M = reader.nFeatures;

    // Read documents
    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("Fail to open input file: %s", inFile.c_str());
    }
    std::vector<Document> docs;
    if (reader.nUnits > 0) {
        docs.reserve(reader.nUnits);
    }
    std::string line, info;
    int32_t idx = 0;
    while (std::getline(inFileStream, line)) {
        idx++;
        if (idx % 10000 == 0) {
            notice("Read %d units...", idx);
        }
        Document d;
        int32_t ct = reader.parseLine(d, info, line);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        if (ct < minCountTrain) continue;
        docs.push_back(std::move(d));
        if (debug_ > 0 && idx >= debug_) break;
    }
    inFileStream.close();
    notice("Read %lu units with %d features", docs.size(), M);

    // Compute stats + HVF order
    VSTOutputs stats;
    auto order = SelectVST(docs, (uint32_t)M, top_k, loess_span, clip_max, &stats);

    // Write per-feature stats
    std::string outf_stats = outPrefix + ".feature.stats.tsv";
    {
        std::ofstream ofs(outf_stats);
        if (!ofs.is_open()) {
            error("Cannot open output file: %s", outf_stats.c_str());
        }
        ofs << "#Feature\tMean\tVar\tVarExpected\tVarStd\n";
        const auto &fnames = reader.features;
        for (int32_t j = 0; j < M; ++j) {
            const std::string& name = (j < (int32_t)fnames.size()) ? fnames[j] : std::to_string(j);
            ofs << name << "\t"
                << fp_to_string(stats.mean_all[j], 4) << "\t"
                << fp_to_string(stats.var_all[j], 4) << "\t"
                << fp_to_string(stats.var_expected[j], 4) << "\t"
                << fp_to_string(stats.var_standardized[j], 4) << "\n";
        }
    }
    notice("Wrote VST stats to %s", outf_stats.c_str());

    // Write ranked HVFs
    std::string outf_hvf = outPrefix + ".hvf.tsv";
    {
        std::ofstream ofs(outf_hvf);
        if (!ofs.is_open()) {
            error("Cannot open output file: %s", outf_hvf.c_str());
        }
        ofs << "#Feature\tTotalCount\tScore\n";
        const auto &fnames = reader.features;
        for (size_t t = 0; t < order.size(); ++t) {
            uint32_t j = order[t];
            const std::string& name = (j < fnames.size()) ? fnames[j] : std::to_string(j);
            ofs << name << "\t"
                << fp_to_string(stats.sum_all[j], 0) << "\t"
                << fp_to_string(stats.var_standardized[j], 6) << "\n";
        }
    }
    notice("Wrote HVF list to %s", outf_hvf.c_str());

    return 0;
}
