#include "topic_svb.hpp"

#include <filesystem>

namespace {

enum class TenXFeatureMode {
    Default,
    FeatureFile,
    RegexOnly,
    PostloadCounts
};

} // namespace

int32_t cmdHDPSVI(int argc, char** argv) {
    std::string inFile, metaFile, weightFile, outPrefix, featureFile;
    std::string dge_dir, in_bc, in_ft, in_mtx;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t seed = -1;
    int32_t nEpochs = 1, batchSize = 512;
    int32_t debug_ = 0, verbose = 0;
    int32_t nThreads = 0;
    int32_t modal = 0;
    int32_t minCountTrain = 20, minCountFeature = 1;
    double defaultWeight = 1.0;
    bool transform = false;
    bool append_topk = false;
    std::string topk_colname = "topK";
    std::string topp_colname = "topP";
    bool drop_random_key = false;

    double kappa = 0.7, tau0 = 10.0;
    double eta = -1.0;
    int32_t maxIter = 100;
    double mDelta = 1e-3;
    int32_t max_topics_K = 100;
    int32_t doc_trunc_T = 10;
    double hdp_alpha = 1.0;
    double hdp_omega = 1.0;
    double topic_threshold = 1e-8;
    double topic_coverage = 1.0 - 1e-8;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("out-prefix", "Output prefix for model and results files", outPrefix, true)
      .add_option("transform", "Transform data to topic space after training", transform)
      .add_option("append-topk", "Append topK/topP columns to transform output", append_topk)
      .add_option("topk-colname", "Column name for topK output", topk_colname)
      .add_option("topp-colname", "Column name for topP output", topp_colname)
      .add_option("drop-random-key", "Drop random_key column from transform output", drop_random_key);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dir)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx);

    pl.add_option("feature-weights", "Input weights file", weightFile)
      .add_option("features", "Feature list", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included", minCountFeature)
      .add_option("default-weight", "Default weight for features not in weight file", defaultWeight)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex);

    pl.add_option("seed", "Random seed", seed)
      .add_option("threads", "Number of threads", nThreads)
      .add_option("n-epochs", "Number of epochs", nEpochs)
      .add_option("minibatch-size", "Minibatch size", batchSize)
      .add_option("min-count-train", "Minimum total feature count for a unit to be trained", minCountTrain)
      .add_option("modal", "Modality to use (0-based)", modal)
      .add_option("debug", "If >0, only process this many units", debug_)
      .add_option("verbose", "Verbose level", verbose);

    pl.add_option("kappa", "Learning decay rate", kappa)
      .add_option("tau0", "Learning offset", tau0)
      .add_option("eta", "Topic-word prior", eta)
      .add_option("max-iter", "Max iterations per doc", maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per doc", mDelta)
      .add_option("max-topics", "Maximum number of topics (K)", max_topics_K)
      .add_option("doc-trunc-level", "Document topic truncation level (T)", doc_trunc_T)
      .add_option("hdp-alpha", "Document-level concentration", hdp_alpha)
      .add_option("hdp-omega", "Corpus-level concentration", hdp_omega)
      .add_option("topic-threshold", "Only output topics with relative weight > threshold", topic_threshold)
      .add_option("topic-coverage", "Output top topics that explain this proportion of the data", topic_coverage);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (batchSize <= 0) {
        batchSize = 512;
        warning("Minibatch size must be greater than 0, using default value of %d", batchSize);
    }
    if (nEpochs <= 0) {
        nEpochs = 1;
    }
    if (seed <= 0) {
        seed = std::random_device{}();
    }
    if (max_topics_K <= 0) {
        error("--max-topics must be > 0");
    }
    if (doc_trunc_T <= 0) {
        error("--doc-trunc-level must be > 0");
    }

    int32_t nUnits;
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    const bool use_10x = !dge_dir.empty() || !in_bc.empty() || !in_ft.empty() || !in_mtx.empty();
    if (use_10x) {
        if (!inFile.empty()) {
            warning("Both --in-data and 10X inputs are provided; using 10X inputs and ignoring --in-data");
        }
        if (!dge_dir.empty() && (in_bc.empty() || in_ft.empty() || in_mtx.empty())) {
            if (dge_dir.back() == '/') {
                dge_dir.pop_back();
            }
            in_bc = dge_dir + "/barcodes.tsv.gz";
            in_ft = dge_dir + "/features.tsv.gz";
            in_mtx = dge_dir + "/matrix.mtx.gz";
        }
        if (in_bc.empty() || in_ft.empty() || in_mtx.empty()) {
            error("Missing required 10X inputs (--in-barcodes, --in-features, --in-matrix)");
        }
        dge_ptr = std::make_unique<DGEReader10X>(in_bc, in_ft, in_mtx);
        nUnits = dge_ptr->nBarcodes;
        reader.initFromFeatures(dge_ptr->features, nUnits);
    } else {
        if (metaFile.empty() || inFile.empty()) {
            error("Missing --in-data or --in-meta");
        }
        reader.readMetadata(metaFile);
        nUnits = reader.nUnits;
    }

    TenXFeatureMode tenx_feature_mode = TenXFeatureMode::Default;
    if (use_10x) {
        if (!featureFile.empty()) {
            tenx_feature_mode = TenXFeatureMode::FeatureFile;
            reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex, true);
        } else if (minCountFeature <= 1) {
            tenx_feature_mode = TenXFeatureMode::RegexOnly;
            if (!include_ftr_regex.empty() || !exclude_ftr_regex.empty()) {
                reader.filterCurrentFeatures(1, include_ftr_regex, exclude_ftr_regex);
            }
        } else {
            tenx_feature_mode = TenXFeatureMode::PostloadCounts;
        }
    } else if (!featureFile.empty()) {
        reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
    }
    if (!weightFile.empty()) {
        reader.setWeights(weightFile, defaultWeight);
    }

    auto hdp4hex = std::make_unique<HDP4Hex>(reader, modal, 10);

    if (use_10x) {
        int32_t n_overlap = dge_ptr->setFeatureIndexRemap(hdp4hex->getFeatureNames(), false);
        if (n_overlap == 0) {
            error("No overlapping features found between 10X input and model");
        }
        hdp4hex->prepare10XCache(*dge_ptr, minCountTrain, true);

        if (tenx_feature_mode == TenXFeatureMode::PostloadCounts) {
            const int32_t nFeaturesPrev = hdp4hex->nFeatures();
            const int32_t nKept = hdp4hex->filterCurrentFeatures(minCountFeature, include_ftr_regex, exclude_ftr_regex);
            if (nKept == 0) {
                error("No features remain after applying feature filters");
            }
            if (hdp4hex->nFeatures() != nFeaturesPrev) {
                n_overlap = dge_ptr->setFeatureIndexRemap(hdp4hex->getFeatureNames(), false);
                if (n_overlap == 0) {
                    error("No overlapping features found between 10X input and model");
                }
                hdp4hex->prepare10XCache(*dge_ptr, minCountTrain, true);
            }
        }
        if (featureFile.empty()) {
            const std::string outFeatures = outPrefix + ".features.tsv";
            std::ofstream outFeatureStream(outFeatures);
            if (!outFeatureStream) {
                error("Error opening output file: %s for writing", outFeatures.c_str());
            }
            const auto featureNames = hdp4hex->getFeatureNames();
            const auto& featureSums = hdp4hex->getFeatureSumsRaw();
            if (featureNames.size() != featureSums.size()) {
                error("Feature names and total counts have inconsistent sizes (%zu vs %zu)",
                    featureNames.size(), featureSums.size());
            }
            outFeatureStream << std::fixed << std::setprecision(0);
            for (size_t i = 0; i < featureNames.size(); ++i) {
                outFeatureStream << featureNames[i] << "\t" << featureSums[i] << "\n";
            }
            outFeatureStream.close();
            notice("Features and total counts written to %s", outFeatures.c_str());
        }
    }

    hdp4hex->initialize(max_topics_K, doc_trunc_T, seed, nThreads, verbose,
        eta, hdp_alpha, hdp_omega, kappa, tau0, hdp4hex->nUnits(), maxIter, mDelta);

    std::string outModel = outPrefix + ".model.tsv";
    std::ofstream outFileStream(outModel);
    if (!outFileStream) {
        error("Error opening output file: %s for writing", outModel.c_str());
    }
    outFileStream.close();
    if (std::filesystem::exists(outModel)) {
        std::filesystem::remove(outModel);
    }

    notice("Starting model training....");
    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        int32_t n = 0;
        if (use_10x) {
            n = hdp4hex->trainOnline10X(batchSize, maxUnits, seed + epoch);
        } else {
            n = hdp4hex->trainOnline(inFile, batchSize, minCountTrain, maxUnits);
        }
        notice("Epoch %d/%d, processed %d documents", epoch + 1, nEpochs, n);
        hdp4hex->printTopicAbundance();
    }

    hdp4hex->filterTopics(topic_threshold, topic_coverage);
    hdp4hex->writeModelToFile(outModel);
    notice("Model written to %s", outModel.c_str());

    if (transform) {
        hdp4hex->setTransformOutputOptions(append_topk, topk_colname, topp_colname, drop_random_key);
        if (use_10x) {
            hdp4hex->fitAndWriteToFile10X(*dge_ptr, outPrefix, batchSize);
        } else {
            hdp4hex->fitAndWriteToFile(inFile, outPrefix, batchSize);
        }
    }

    return 0;
}
