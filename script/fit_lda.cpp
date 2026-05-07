#include "topic_svb.hpp"

#include <filesystem>
#include <sstream>
#include <vector>

int32_t cmdLDATransform(int argc, char** argv);

namespace {

enum class TenXFeatureMode {
    Default,
    FeatureFile,
    ModelOnly,
    RegexOnly,
    PostloadCounts
};

void appendFlag(std::vector<std::string>& args, const std::string& name) {
    args.push_back("--" + name);
}

template <typename T>
void appendOption(std::vector<std::string>& args, const std::string& name, const T& value) {
    std::ostringstream oss;
    oss << value;
    args.push_back("--" + name);
    args.push_back(oss.str());
}

template <typename T>
void appendOptions(std::vector<std::string>& args, const std::string& name, const std::vector<T>& values) {
    if (values.empty()) {
        return;
    }
    args.push_back("--" + name);
    for (const auto& value : values) {
        std::ostringstream oss;
        oss << value;
        args.push_back(oss.str());
    }
}

int32_t runDelegatedTransform(const std::string& modelFile,
        const std::string& outPrefix,
        const std::string& inFile,
        const std::string& metaFile,
        const std::vector<std::string>& dge_dirs,
        const std::vector<std::string>& in_bc,
        const std::vector<std::string>& in_ft,
        const std::vector<std::string>& in_mtx,
        const std::vector<std::string>& dataset_ids,
        const std::string& featureFile,
        int32_t minCountFeature,
        const std::string& includeFeatureRegex,
        const std::string& excludeFeatureRegex,
        const std::string& weightFile,
        double defaultWeight,
        int32_t maxIter,
        double meanChangeTol,
        int32_t nThreads,
        int32_t modal,
        int32_t debugN,
        bool computeResiduals,
        int32_t topkOnly) {
    std::vector<std::string> args;
    args.reserve(32);
    args.push_back("lda-transform");
    appendOption(args, "in-model", modelFile);
    appendOption(args, "out-prefix", outPrefix);
    appendOption(args, "min-count", 1);
    appendOption(args, "threads", nThreads);
    appendOption(args, "modal", modal);
    appendOption(args, "max-iter", maxIter);
    appendOption(args, "mean-change-tol", meanChangeTol);
    if (debugN > 0) {
        appendOption(args, "debug", debugN);
    }
    if (!inFile.empty()) {
        appendOption(args, "in-data", inFile);
        appendOption(args, "in-meta", metaFile);
    } else {
        if (!dge_dirs.empty()) {
            appendOptions(args, "in-dge-dir", dge_dirs);
        } else {
            appendOptions(args, "in-barcodes", in_bc);
            appendOptions(args, "in-features", in_ft);
            appendOptions(args, "in-matrix", in_mtx);
        }
        appendOptions(args, "dataset-id", dataset_ids);
    }
    if (!featureFile.empty()) {
        appendOption(args, "features", featureFile);
    }
    appendOption(args, "min-count-per-feature", minCountFeature);
    if (!includeFeatureRegex.empty()) {
        appendOption(args, "include-feature-regex", includeFeatureRegex);
    }
    if (!excludeFeatureRegex.empty()) {
        appendOption(args, "exclude-feature-regex", excludeFeatureRegex);
    }
    if (!weightFile.empty()) {
        appendOption(args, "feature-weights", weightFile);
        appendOption(args, "default-weight", defaultWeight);
    }
    if (computeResiduals) {
        appendFlag(args, "residuals");
    }
    if (topkOnly > 0) {
        appendOption(args, "topk-only", topkOnly);
    }

    std::vector<char*> argv;
    argv.reserve(args.size());
    for (auto& arg : args) {
        argv.push_back(arg.data());
    }
    return cmdLDATransform(static_cast<int32_t>(argv.size()), argv.data());
}

} // namespace

int32_t cmdTopicModelSVI(int argc, char** argv) {
    std::string inFile, metaFile, weightFile, outPrefix, priorFile, featureFile;
    std::vector<std::string> dge_dirs, in_bc, in_ft, in_mtx, dataset_ids;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t seed = -1;
    int32_t nEpochs = 1, batchSize = 512;
    int32_t debug_ = 0, verbose = 0;
    int32_t nThreads = 0;
    int32_t modal = 0;
    int32_t minCountTrain = 20, minCountFeature = 1;
    int32_t topk_only = -1;
    double defaultWeight = 1.0;
    bool transform = false;
    bool computeResiduals = false;
    bool sort_topics = false;
    bool reproducible_init = false;

    double kappa = 0.7, tau0 = 10.0;
    double alpha = -1.0, eta = -1.0;
    int32_t maxIter = 100;
    double mDelta = 1e-3;
    int32_t nTopics = 0;
    double priorScale = -1.0;
    double priorScaleRel = -1.0;
    bool projection_only = false;

    bool fitBackground = false;
    bool fixBackground = false;
    std::string bgPriorFile;
    double a0 = 2, b0 = 8;
    double warmInitEpoch = 0.5;
    double bgInitScale = 0.5;
    int32_t warmInitUnits = -1;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("out-prefix", "Output prefix for model and results files", outPrefix, true)
      .add_option("transform", "Transform data to topic space after training", transform)
      .add_option("sort-topics", "Sort topics by weight after training", sort_topics)
      .add_option("residuals", "Compute residual-based transform summaries in .unit_meta.tsv", computeResiduals)
      .add_option("feature-residuals", "Compute residual-based transform summaries in .unit_meta.tsv", computeResiduals)
      .add_option("topk-only", "Write only top-k factor indices/probabilities to results.tsv", topk_only);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", dataset_ids);

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
      .add_option("eta", "Topic-word prior (default: 1/K)", eta)
      .add_option("max-iter", "Max iterations per doc", maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per doc", mDelta)
      .add_option("n-topics", "Number of topics", nTopics)
      .add_option("alpha", "Document-topic prior (default: 1/K)", alpha)
      .add_option("reproducible-init", "Enable deterministic per-document random initialization (slower)", reproducible_init);

    pl.add_option("model-prior", "File with initial model matrix for continued training", priorFile)
      .add_option("prior-scale", "Uniform scaling factor for the prior model matrix", priorScale)
      .add_option("prior-scale-rel", "Scale prior model relative to the total feature counts in the data (overrides --prior-scale)", priorScaleRel)
      .add_option("projection-only", "Transform data using prior model without training", projection_only)
      .add_option("fit-background", "Fit a background noise in addition to topics", fitBackground)
      .add_option("background-prior", "File with background prior vector", bgPriorFile)
      .add_option("background-init-scale", "Scaling factor for constructing background prior from total feature counts", bgInitScale)
      .add_option("fix-background", "Fix the background model during training", fixBackground)
      .add_option("bg-fraction-prior-a0", "Background fraction hyper-parameter a0 in pi~beta(a0, b0)", a0)
      .add_option("bg-fraction-prior-b0", "Background fraction hyper-parameter b0 in pi~beta(a0, b0)", b0)
      .add_option("warm-start-epochs", "Number of epochs to warm start factors before fitting background (could be fractional)", warmInitEpoch);

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
    if (topk_only == 0) {
        error("--topk-only must be a positive integer");
    }
    if (seed <= 0) {
        seed = std::random_device{}();
    }

    int32_t nUnits;
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    const auto dge_inputs = resolveDge10XInputs(dge_dirs, in_bc, in_ft, in_mtx, dataset_ids);
    const bool use_10x = !dge_inputs.empty();
    if (use_10x) {
        if (!inFile.empty()) {
            warning("Both --in-data and 10X inputs are provided; using 10X inputs and ignoring --in-data");
        }
        if (!in_bc.empty() || !in_ft.empty() || !in_mtx.empty()) {
            dge_ptr = std::make_unique<DGEReader10X>(in_bc, in_ft, in_mtx, dataset_ids);
        } else {
            dge_ptr = std::make_unique<DGEReader10X>(dge_dirs, dataset_ids);
        }
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
        const bool has_model_prior = !priorFile.empty();
        if (!featureFile.empty()) {
            tenx_feature_mode = TenXFeatureMode::FeatureFile;
            reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex, true);
        } else if (has_model_prior) {
            tenx_feature_mode = TenXFeatureMode::ModelOnly;
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

    if (projection_only) {
        transform = true;
    }
    if (nTopics <= 0 && priorFile.empty()) {
        error("Number of topics must be greater than 0");
    }

    auto lda4hex = std::make_unique<LDA4Hex>(reader, modal, 10);
    if (!priorFile.empty()) {
        lda4hex->preparePriorFeatureSpace(priorFile);
    }

    if (use_10x) {
        if (tenx_feature_mode == TenXFeatureMode::ModelOnly &&
            ((minCountFeature > 1) || !include_ftr_regex.empty() || !exclude_ftr_regex.empty())) {
            warning("Ignoring --min-count-per-feature and feature regex filters for 10X input because the model prior defines the feature space");
        }
        int32_t n_overlap = dge_ptr->setFeatureIndexRemap(lda4hex->getFeatureNames(), false);
        if (n_overlap == 0) {
            error("No overlapping features found between 10X input and model");
        }
        lda4hex->prepare10XCache(*dge_ptr, minCountTrain, true);

        if (tenx_feature_mode == TenXFeatureMode::PostloadCounts) {
            const int32_t nFeaturesPrev = lda4hex->nFeatures();
            const int32_t nKept = lda4hex->filterCurrentFeatures(minCountFeature, include_ftr_regex, exclude_ftr_regex);
            if (nKept == 0) {
                error("No features remain after applying feature filters");
            }
            if (lda4hex->nFeatures() != nFeaturesPrev) {
                n_overlap = dge_ptr->setFeatureIndexRemap(lda4hex->getFeatureNames(), false);
                if (n_overlap == 0) {
                    error("No overlapping features found between 10X input and model");
                }
                lda4hex->prepare10XCache(*dge_ptr, minCountTrain, true);
            }
        }
        if (featureFile.empty()) {
            const std::string outFeatures = outPrefix + ".features.tsv";
            std::ofstream outFeatureStream(outFeatures);
            if (!outFeatureStream) {
                error("Error opening output file: %s for writing", outFeatures.c_str());
            }
            const auto featureNames = lda4hex->getFeatureNames();
            const auto& featureSums = lda4hex->getFeatureSumsRaw();
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

    lda4hex->initialize_svb(nTopics, seed, nThreads, verbose,
        alpha, eta, kappa, tau0, nUnits,
        priorFile, priorScale, priorScaleRel, maxIter, mDelta);
    lda4hex->set_reproducible_init(reproducible_init);
    if (fitBackground) {
        if (warmInitUnits < 0) {
            warmInitUnits = static_cast<int32_t>(warmInitEpoch * nUnits);
        } else {
            warmInitEpoch = static_cast<double>(warmInitUnits) / nUnits;
        }
        if (warmInitUnits > 0) {
            notice("Warm-start using %d units before introducing background", warmInitUnits);
            int32_t nWarm = 0;
            if (use_10x) {
                nWarm = lda4hex->trainOnline10X(batchSize, warmInitUnits, seed);
            } else {
                nWarm = lda4hex->trainOnline(inFile, batchSize, minCountTrain, warmInitUnits);
            }
            notice("Warm-start processed %d documents", nWarm);
            lda4hex->printTopicAbundance();
        }
        const double bgScale = lda4hex->hasFullFeatureSums() ? bgInitScale : 1.0;
        lda4hex->set_background_prior(bgPriorFile, a0, b0, bgScale, fixBackground);
    }

    std::string outModel = outPrefix + ".model.tsv";
    if (!projection_only && !priorFile.empty() && priorFile == outModel) {
        outModel = outPrefix + ".model.updated.tsv";
    }

    if (!projection_only) {
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
                n = lda4hex->trainOnline10X(batchSize, maxUnits, seed + epoch);
            } else {
                n = lda4hex->trainOnline(inFile, batchSize, minCountTrain, maxUnits);
            }
            notice("Epoch %d/%d, processed %d documents", epoch + 1, nEpochs, n);
            lda4hex->printTopicAbundance();
        }
        if (sort_topics) {
            lda4hex->sortTopicsByWeight();
        }
        if (fitBackground) {
            std::string bgFile = outPrefix + ".background.tsv";
            lda4hex->writeBackgroundModel(bgFile);
            notice("Background profile written to %s", bgFile.c_str());
        }

        lda4hex->writeModelToFile(outModel);
        notice("Model written to %s", outModel.c_str());
    }

    if (transform) {
        const std::string transformModel = projection_only ? priorFile : outModel;
        if (!fitBackground) {
            return runDelegatedTransform(transformModel, outPrefix, inFile, metaFile,
                dge_dirs, in_bc, in_ft, in_mtx, dataset_ids, featureFile, minCountFeature,
                include_ftr_regex, exclude_ftr_regex, weightFile, defaultWeight,
                maxIter, mDelta, nThreads, modal, debug_, computeResiduals, topk_only);
        }
        if (computeResiduals || topk_only > 0) {
            warning("Keeping legacy transform output because background-enabled LDA does not yet support delegated residual/top-k transform");
        }
        if (use_10x) {
            lda4hex->fitAndWriteToFile10X(*dge_ptr, outPrefix, batchSize);
        } else {
            lda4hex->fitAndWriteToFile(inFile, outPrefix, batchSize);
        }
    }

    return 0;
}
