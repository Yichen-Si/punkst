#include "gamma_pois_topic.hpp"

#include <cmath>
#include <climits>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <sstream>

int32_t cmdGammaPoisJointTransform(int argc, char** argv);

namespace {

void append_arg(std::vector<std::string>& args, const std::string& key, const std::string& value) {
    if (!value.empty()) {
        args.push_back(key);
        args.push_back(value);
    }
}

void append_arg(std::vector<std::string>& args, const std::string& key, int32_t value) {
    args.push_back(key);
    args.push_back(std::to_string(value));
}

void append_arg(std::vector<std::string>& args, const std::string& key, double value) {
    args.push_back(key);
    args.push_back(std::to_string(value));
}

void append_repeated(std::vector<std::string>& args, const std::string& key,
    const std::vector<std::string>& values) {
    for (const auto& value : values) {
        append_arg(args, key, value);
    }
}

} // namespace

int32_t cmdGammaPoisJointFit(int argc, char** argv) {
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
    double defaultWeight = -1.0;
    bool transform = false;
    bool sort_topics = false;
    bool sort_clusters = false;

    double kappa = 0.7, tau0 = 10.0;
    int32_t maxIter = 100;
    double mDelta = 1e-3;
    int32_t nTopics = 0;
    int32_t nClusters = 0;
    double betaShape = 0.3;
    double xiShape = 0.3;
    double xiMean = -1.0;
    double thetaShape = 1.0;
    double nuShape = 1.0;
    double nuRate = -1.0;
    double clusterPrior = 1.0;
    double sizeFactor = -1.0;
    int32_t clusterWarmupEpochs = 1;
    int32_t clusterAnnealEpochs = 1;
    double clusterTempStart = 2.0;
    double clusterPriorStart = -1.0;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("out-prefix", "Output prefix for model and results files", outPrefix, true)
      .add_option("transform", "Transform data to topic and cluster space after training", transform)
      .add_option("sort-topics", "Sort topics by decreasing usage after training", sort_topics)
      .add_option("sort-clusters", "Sort clusters by decreasing usage after training", sort_clusters);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", dataset_ids);

    pl.add_option("features", "Feature list", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included", minCountFeature)
      .add_option("default-weight", "Default weight for model features missing from --features when feature weights are active; <0 drops missing features", defaultWeight)
      .add_option("icol-weight", "0-based column index for feature weight in --features; <0 disables feature weights", icolWeight)
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
      .add_option("n-clusters", "Number of document clusters", nClusters)
      .add_option("beta-shape", "Gamma shape a for beta_wr", betaShape)
      .add_option("xi-shape", "Gamma shape a0 for xi_w", xiShape)
      .add_option("xi-mean", "Prior mean b0 for xi_w; default derives from size factor and vocabulary", xiMean)
      .add_option("theta-shape", "Gamma shape s0 for theta_dr", thetaShape)
      .add_option("nu-shape", "Gamma shape e0 for nu_cr", nuShape)
      .add_option("nu-rate", "Gamma rate f0 for nu_cr; default derives from size factor", nuRate)
      .add_option("cluster-prior", "Dirichlet concentration gamma for document clusters", clusterPrior)
      .add_option("cluster-warmup-epochs", "Epochs to train topics with uniform cluster assignments before seeding clusters", clusterWarmupEpochs)
      .add_option("cluster-anneal-epochs", "Epochs over which to anneal cluster temperature and prior after warmup", clusterAnnealEpochs)
      .add_option("cluster-temp-start", "Initial chi softmax temperature after cluster seeding", clusterTempStart)
      .add_option("cluster-prior-start", "Initial cluster prior gamma after seeding; default is n-clusters", clusterPriorStart)
      .add_option("size-factor", "Corpus mean document length nbar; required when full feature counts are unavailable", sizeFactor);

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
    if (nClusters <= 0) error("--n-clusters must be greater than 0");
    if (clusterWarmupEpochs < 0) clusterWarmupEpochs = 0;
    if (clusterAnnealEpochs < 0) clusterAnnealEpochs = 0;
    if (clusterTempStart < 1.0 || !std::isfinite(clusterTempStart)) clusterTempStart = 1.0;
    if (clusterPriorStart <= 0.0 || !std::isfinite(clusterPriorStart)) {
        clusterPriorStart = static_cast<double>(nClusters);
    }
    if (seed <= 0) seed = std::random_device{}();
    const bool weights_active = !featureFile.empty() && icolWeight >= 0;
    if (defaultWeight < 0.0) defaultWeight = -1.0;

    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    const bool use_10x = initHexOrDgeInput(reader, dge_ptr, inFile, metaFile,
        dge_dirs, in_bc, in_ft, in_mtx, dataset_ids);
    if (!featureFile.empty()) {
        if (weights_active) {
            reader.setFeatureFilterAndWeights(featureFile, minCountFeature,
                include_ftr_regex, exclude_ftr_regex, icolWeight, defaultWeight, false, use_10x);
        } else {
            reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex, use_10x);
        }
    }

    auto gp = std::make_unique<GammaPoissonJointC4Hex>(reader, modal, verbose);
    if (use_10x) {
        int32_t n_overlap = dge_ptr->setFeatureIndexRemap(gp->getFeatureNames(), false);
        if (n_overlap == 0) error("No overlapping features found between 10X input and model");
        gp->prepare10XCache(*dge_ptr, minCountTrain, true);
        if (featureFile.empty()) {
            const std::string outFeatures = outPrefix + ".features.tsv";
            std::ofstream outFeatureStream(outFeatures);
            if (!outFeatureStream) error("Error opening output file: %s for writing", outFeatures.c_str());
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
    gp->initialize(nTopics, nClusters, seed, nThreads, verbose, betaShape, xiShape,
        xiMean, thetaShape, nuShape, nuRate, clusterPrior, kappa, tau0,
        gp->nUnits(), resolvedSizeFactor, maxIter, mDelta);

    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    const int32_t effectiveWarmupEpochs = std::min(clusterWarmupEpochs, std::max(0, nEpochs - 1));
    if (effectiveWarmupEpochs != clusterWarmupEpochs) {
        notice("Using %d cluster warmup epochs so at least one release epoch is run",
            effectiveWarmupEpochs);
    }
    bool clusters_seeded = false;
    if (effectiveWarmupEpochs > 0) {
        gp->setClusterWarmup(true);
        notice("Cluster warmup enabled for %d epoch(s); chi is held uniform and cluster globals are frozen",
            effectiveWarmupEpochs);
    }
    notice("Starting joint Gamma-Poisson model training....");
    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        if (!clusters_seeded && epoch == effectiveWarmupEpochs) {
            if (effectiveWarmupEpochs > 0) {
                gp->initializeClustersFromTrainingData(inFile, use_10x, minCountTrain,
                    maxUnits, clusterPriorStart);
                clusters_seeded = true;
                notice("Cluster warmup complete; initialized clusters from document topic embeddings");
            } else {
                gp->setClusterWarmup(false);
                gp->setClusterTemperature(1.0);
                gp->setEffectiveClusterPrior(clusterPrior);
                clusters_seeded = true;
            }
        }
        if (clusters_seeded && effectiveWarmupEpochs > 0) {
            const int32_t releaseEpoch = epoch - effectiveWarmupEpochs;
            double frac = 1.0;
            if (clusterAnnealEpochs > 1) {
                frac = std::min(1.0, static_cast<double>(releaseEpoch)
                    / static_cast<double>(clusterAnnealEpochs - 1));
            }
            const double temp = 1.0 + (clusterTempStart - 1.0) * (1.0 - frac);
            const double gammaEff = clusterPrior
                + (clusterPriorStart - clusterPrior) * (1.0 - frac);
            gp->setClusterTemperature(temp);
            gp->setEffectiveClusterPrior(gammaEff);
            if (verbose > 0) {
                notice("Cluster release schedule: epoch %d temperature %.4g effective gamma %.4g",
                    epoch + 1, temp, gammaEff);
            }
        }
        int32_t n = 0;
        if (use_10x) {
            n = gp->trainOnline10X(batchSize, maxUnits, seed + epoch);
        } else {
            n = gp->trainOnline(inFile, batchSize, minCountTrain, maxUnits);
        }
        notice("Epoch %d/%d, processed %d documents", epoch + 1, nEpochs, n);
        gp->printTopicAbundance();
        std::vector<double> cluster_weights;
        gp->getClusterAbundance(cluster_weights);
        std::sort(cluster_weights.begin(), cluster_weights.end(), std::greater<double>());
        std::ostringstream oss;
        oss << "Top cluster relative abundance: ";
        for (size_t i = 0; i < std::min<size_t>(10, cluster_weights.size()); ++i) {
            oss << std::fixed << std::setprecision(4) << cluster_weights[i] << "\t";
        }
        notice("%s", oss.str().c_str());
    }
    if (sort_topics) gp->sortTopicsByWeight();
    if (sort_clusters) gp->sortClustersByWeight();

    const std::string outModel = outPrefix + ".model.tsv";
    const std::string outState = outPrefix + ".state.tsv";
    gp->writeModelToFile(outModel);
    gp->writeStateToFile(outState);
    notice("Model written to %s", outModel.c_str());
    notice("Joint Gamma-Poisson state written to %s", outState.c_str());

    if (transform) {
        std::vector<std::string> args;
        args.push_back("gamma-pois-jc-transform");
        append_arg(args, "--in-data", inFile);
        append_arg(args, "--in-meta", metaFile);
        append_repeated(args, "--in-dge-dir", dge_dirs);
        append_repeated(args, "--in-barcodes", in_bc);
        append_repeated(args, "--in-features", in_ft);
        append_repeated(args, "--in-matrix", in_mtx);
        append_repeated(args, "--dataset-id", dataset_ids);
        append_arg(args, "--in-state", outState);
        append_arg(args, "--out-prefix", outPrefix);
        append_arg(args, "--minibatch-size", batchSize);
        append_arg(args, "--modal", modal);
        append_arg(args, "--threads", nThreads);
        append_arg(args, "--seed", seed);
        append_arg(args, "--debug", debug_);
        append_arg(args, "--features", featureFile);
        append_arg(args, "--min-count-per-feature", minCountFeature);
        append_arg(args, "--min-count", minCountTrain);
        append_arg(args, "--default-weight", defaultWeight);
        append_arg(args, "--icol-weight", icolWeight);
        append_arg(args, "--include-feature-regex", include_ftr_regex);
        append_arg(args, "--exclude-feature-regex", exclude_ftr_regex);
        append_arg(args, "--max-iter", maxIter);
        append_arg(args, "--mean-change-tol", mDelta);
        std::vector<char*> cargs;
        cargs.reserve(args.size());
        for (auto& arg : args) cargs.push_back(arg.data());
        const int32_t rc = cmdGammaPoisJointTransform(static_cast<int>(cargs.size()), cargs.data());
        if (rc != 0) return rc;
    }
    return 0;
}
