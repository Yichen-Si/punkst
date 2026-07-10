#include "gamma_pois_topic.hpp"

#include <climits>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

int32_t cmdGammaPoisTransform(int argc, char** argv);

int32_t cmdGammaPoisFit(int argc, char** argv) {
    std::string inFile, metaFile, outPrefix, featureFile;
    std::vector<std::string> dge_dirs, in_bc, in_ft, in_mtx, dataset_ids;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t seed = -1;
    int32_t nEpochs = 1, batchSize = 512;
    int32_t debug_ = 0, verbose = 0;
    int32_t nThreads = 1;
    int32_t modal = 0;
    int32_t minCountTrain = 20, minCountFeature = 1;
    int32_t icolWeight = -1;
    int32_t icolDispersion = -1;
    bool estimateDispersion = false;
    int32_t dispersionInitEpochs = 1;
    int32_t dispersionMinPositive = 10;
    int32_t dispersionMuBins = 32;
    double defaultWeight = -1.0;
    bool transform = false;
    bool sort_topics = false;

    double kappa = 0.7, tau0 = 10.0;
    int32_t maxIter = 100;
    double mDelta = 1e-3;
    int32_t nTopics = 0;
    double betaShape = 0.3;
    double xiShape = 0.3;
    double xiMean = -1.0;
    double thetaShape = 1.0;
    double nuShape = 1.0;
    double nuRate = -1.0;
    double sizeFactor = -1.0;
    double dispersionLoessSpan = 0.3;
    double dispersionDeltaMin = 1e-8;
    double dispersionDeltaMax = 1e4;
    bool symmetricNu = false;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("out-prefix", "Output prefix for model and results files", outPrefix, true)
      .add_option("transform", "Transform data to topic space after training", transform)
      .add_option("sort-topics", "Sort topics by decreasing usage after training", sort_topics);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", dataset_ids);

    pl.add_option("features", "Feature list", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included", minCountFeature)
      .add_option("default-weight", "Default weight for model features missing from --features when feature weights are active; <0 drops missing features", defaultWeight)
      .add_option("icol-weight", "0-based column index for feature weight in --features; <0 disables feature weights", icolWeight)
      .add_option("icol-dispersion", "0-based column index for per-feature dispersion tau in --features; <0 disables dispersion", icolDispersion)
      .add_option("estimate-dispersion", "Estimate per-feature dispersion after a Poisson warmup", estimateDispersion)
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
      .add_option("max-iter", "Max iterations per doc", maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per doc", mDelta)
      .add_option("n-topics", "Number of topics", nTopics)
      .add_option("beta-shape", "Gamma shape a for beta_wr", betaShape)
      .add_option("xi-shape", "Gamma shape a0 for xi_w", xiShape)
      .add_option("xi-mean", "Prior mean b0 for xi_w; default derives from size factor and vocabulary", xiMean)
      .add_option("theta-shape", "Gamma shape s0 for theta_dr", thetaShape)
      .add_option("nu-shape", "Gamma shape e0 for nu_r", nuShape)
      .add_option("nu-rate", "Gamma rate f0 for nu_r; default derives from size factor", nuRate)
      .add_option("symmetric-nu", "Fix E[nu_r] to 1 and skip asymmetric nu updates", symmetricNu)
      .add_option("size-factor", "Corpus mean document length nbar; required when full feature counts are unavailable", sizeFactor);

    pl.add_option("dispersion-init-epochs", "Poisson warmup epochs before estimating dispersion", dispersionInitEpochs)
      .add_option("dispersion-loess-span", "LOESS span for the dispersion abundance trend", dispersionLoessSpan)
      .add_option("dispersion-min-positive", "Minimum positive cells for a raw dispersion estimate", dispersionMinPositive)
      .add_option("dispersion-mu-bins", "Log-mean bins per feature during dispersion estimation", dispersionMuBins)
      .add_option("dispersion-delta-min", "Lower bound for estimated inverse dispersion", dispersionDeltaMin)
      .add_option("dispersion-delta-max", "Upper bound for estimated inverse dispersion", dispersionDeltaMax);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (batchSize <= 0) batchSize = 512;
    if (nEpochs <= 0) nEpochs = 1;
    if (nTopics <= 0) error("--n-topics must be greater than 0");
    if (icolDispersion >= 0 && featureFile.empty()) {
        error("--features is required when --icol-dispersion is non-negative");
    }
    if (estimateDispersion && icolDispersion >= 0) {
        error("--estimate-dispersion and --icol-dispersion are mutually exclusive");
    }
    if (estimateDispersion && (dispersionInitEpochs < 1 || dispersionInitEpochs >= nEpochs)) {
        error("--dispersion-init-epochs must be at least 1 and smaller than --n-epochs");
    }
    if (estimateDispersion && (dispersionMinPositive < 1 || dispersionMuBins < 1
        || !std::isfinite(dispersionLoessSpan) || dispersionLoessSpan <= 0.0
        || dispersionLoessSpan > 1.0
        || dispersionDeltaMin <= 0.0 || dispersionDeltaMax < dispersionDeltaMin
        || !std::isfinite(dispersionDeltaMin) || !std::isfinite(dispersionDeltaMax))) {
        error("Invalid dispersion estimation options");
    }
    if (seed <= 0) seed = std::random_device{}();
    const bool weights_active = !featureFile.empty() && icolWeight >= 0;
    if (defaultWeight < 0.0) defaultWeight = -1.0;

    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    const bool use_10x = initHexOrDgeInput(reader, dge_ptr, inFile, metaFile,
        dge_dirs, in_bc, in_ft, in_mtx, dataset_ids);
    if (!use_10x && !featureFile.empty()) {
        if (weights_active) {
            reader.setFeatureFilterAndWeights(featureFile, minCountFeature,
                include_ftr_regex, exclude_ftr_regex, icolWeight, defaultWeight, false);
        } else {
            reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
        }
    } else if (use_10x && !featureFile.empty()) {
        if (weights_active) {
            reader.setFeatureFilterAndWeights(featureFile, minCountFeature,
                include_ftr_regex, exclude_ftr_regex, icolWeight, defaultWeight, false, true);
        } else {
            reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex, true);
        }
    }

    std::vector<double> suppliedTau;
    if (icolDispersion >= 0) {
        suppliedTau = reader.readPositiveFeatureColumn(featureFile, icolDispersion, "dispersion tau");
    }
    auto gp = std::make_unique<GammaPoisson4Hex>(reader, modal, verbose);
    if (use_10x) {
        int32_t n_overlap = dge_ptr->setFeatureIndexRemap(gp->getFeatureNames(), false);
        if (n_overlap == 0) {
            error("No overlapping features found between 10X input and model");
        }
        gp->prepare10XCache(*dge_ptr, minCountTrain, true);
        if (featureFile.empty()) {
            const std::string outFeatures = outPrefix + ".features.tsv";
            std::ofstream outFeatureStream(outFeatures);
            if (!outFeatureStream) {
                error("Error opening output file: %s for writing", outFeatures.c_str());
            }
            const auto featureNames = gp->getFeatureNames();
            const auto& featureSums = gp->getFeatureSumsRaw();
            outFeatureStream << std::fixed << std::setprecision(0);
            for (size_t i = 0; i < featureNames.size(); ++i) {
                outFeatureStream << featureNames[i] << "\t" << featureSums[i] << "\n";
            }
            notice("Features and total counts written to %s", outFeatures.c_str());
        }
    }

    const double resolvedSizeFactor = gp->resolveSizeFactor(sizeFactor);
    if (xiMean <= 0.0) {
        xiMean = static_cast<double>(std::max(1, gp->nFeatures())) / resolvedSizeFactor;
    }
    notice("Using size factor nbar = %.6g", resolvedSizeFactor);
    gp->initialize(nTopics, seed, nThreads, verbose, betaShape, xiShape, xiMean,
        thetaShape, nuShape, nuRate, kappa, tau0, gp->nUnits(), resolvedSizeFactor,
        symmetricNu, maxIter, mDelta);
    if (icolDispersion >= 0) {
        gp->setFeatureDispersion(suppliedTau);
        notice("Using per-feature dispersion tau from column %d of %s",
            icolDispersion, featureFile.c_str());
    }

    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    notice("Starting Gamma-Poisson model training....");
    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        int32_t n = 0;
        if (use_10x) {
            n = gp->trainOnline10X(batchSize, maxUnits, seed + epoch);
        } else {
            n = gp->trainOnline(inFile, batchSize, minCountTrain, maxUnits);
        }
        notice("Epoch %d/%d, processed %d documents", epoch + 1, nEpochs, n);
        gp->printTopicAbundance();
        if (estimateDispersion && epoch + 1 == dispersionInitEpochs) {
            GammaPoissonDispersionOptions options;
            options.min_positive = dispersionMinPositive;
            options.mu_bins = dispersionMuBins;
            options.loess_span = dispersionLoessSpan;
            options.delta_min = dispersionDeltaMin;
            options.delta_max = dispersionDeltaMax;
            GammaPoissonDispersionResult dispersion = use_10x
                ? gp->estimateFeatureDispersion10X(options, batchSize, maxUnits)
                : gp->estimateFeatureDispersion(options, inFile, batchSize,
                    minCountTrain, maxUnits);
            const std::string outDispersion = outPrefix + ".dispersion.tsv";
            write_gamma_poisson_dispersion_diagnostics(outDispersion,
                gp->getFeatureNames(), dispersion);
            notice("Estimated per-feature dispersion from %d documents; diagnostics written to %s",
                dispersion.n_documents, outDispersion.c_str());
        }
    }
    if (sort_topics) {
        gp->sortTopicsByWeight();
    }

    const std::string outModel = outPrefix + ".model.tsv";
    const std::string outState = outPrefix + ".state.tsv";
    gp->writeModelToFile(outModel);
    gp->writeStateToFile(outState);
    notice("Model written to %s", outModel.c_str());
    notice("Gamma-Poisson state written to %s", outState.c_str());

    if (transform) {
        gp->setTransformOutputOptions(false);
        if (use_10x) {
            gp->fitAndWriteToFile10X(*dge_ptr, outPrefix, batchSize);
        } else {
            gp->fitAndWriteToFile(inFile, outPrefix, batchSize);
        }
    }
    return 0;
}
