#include "gamma_pois_topic.hpp"
#include "count_cache_options.hpp"

#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>

int32_t cmdGammaPoisTransform(int argc, char** argv);

namespace {

void append_arg(std::vector<std::string>& args, const std::string& key,
    const std::string& value) {
    if (!value.empty()) {
        args.push_back(key);
        args.push_back(value);
    }
}

template <typename T>
void append_arg(std::vector<std::string>& args, const std::string& key, T value) {
    args.push_back(key);
    args.push_back(std::to_string(value));
}

void append_repeated(std::vector<std::string>& args, const std::string& key,
    const std::vector<std::string>& values) {
    for (const auto& value : values) append_arg(args, key, value);
}

} // namespace

int32_t cmdGammaPoisFitMap(int argc, char** argv) {
    std::string inFile, metaFile, outPrefix, featureFile, modelInitFile,
        inStateFile;
    std::string inferenceMode = "lda-compatible";
    std::vector<std::string> dge_dirs, in_bc, in_ft, in_mtx, dataset_ids;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t seed = -1;
    int32_t nEpochs = 1, batchSize = 512;
    int32_t debug_ = 0, verbose = 0;
    int32_t nThreads = 1;
    int32_t modal = 0;
    int32_t minCountTrain = 20, minCountFeature = 100;
    int32_t icolWeight = -1;
    int32_t icolDispersion = -1;
    bool estimateDispersion = false;
    int32_t dispersionInitEpochs = 1;
    std::string dispersionEstimator = "factorial";
    double defaultWeight = -1.0;
    bool transform = false;
    bool randomizeOutput = false;
    bool pseudobulkAllFeatures = false;
    bool computeResiduals = false;
    bool cheapFeatureDiagnostics = false;
    bool unitSimilarityDiagnostics = false;
    bool sort_topics = false;
    TrainingCountCacheCliOptions count_cache_options;

    double kappa = 0.7, tau0 = 10.0;
    int32_t maxIter = 100;
    double mDelta = 1e-3;
    int32_t nTopics = 0;
    double randomInitShape = 2.0;
    double thetaConcentration = 1.0;
    double dictionaryPriorMass = -1.0;
    double regularization = 0.0;
    std::string regularizationMode = "uniform";
    int32_t regularizeWarmupEpochs = 1;
    int32_t regularizeRampEpochs = 1;
    int32_t finalRefinePasses = 0;
    double finalRefineTol = 1e-5;
    double dispersionLoessSpan = 0.3;
    double dispersionMinInformation = 8.0;
    double dispersionOutlierSd = 2.0;
    double dispersionDeltaMin = 1e-8;
    double dispersionDeltaMax = 1e4;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("out-prefix", "Output prefix for model and results files", outPrefix, true)
      .add_option("transform", "Transform data to topic space after training", transform)
      .add_option("randomize-output", "Randomize transform output order", randomizeOutput)
      .add_option("pseudobulk-all-features", "Include all retained input features in transform pseudobulk output", pseudobulkAllFeatures)
      .add_option("residuals", "Compute residual-based transform summaries", computeResiduals)
      .add_option("feature-residuals", "Compute residual-based transform summaries", computeResiduals)
      .add_option("feature-diagnostics-cheap", "Skip spool-dependent gain-adjusted feature residual and Pull diagnostics", cheapFeatureDiagnostics)
      .add_option("unit-diagnostics-similarity", "Add cosine and similarity-adjusted entropy unit diagnostics", unitSimilarityDiagnostics)
      .add_option("sort-topics",
          "Sort topics by decreasing exposure-weighted prevalence after training",
          sort_topics);
    pl.add_option("in-state",
        "Gamma-Poisson state used to start a new matched refinement segment",
        inStateFile)
      .add_option("inference-mode",
        "Global allocation inference: lda-compatible (default) or map-mean",
        inferenceMode);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", dataset_ids);

    pl.add_option("features", "Feature list with total counts; required for custom sparse input", featureFile)
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
    add_training_count_cache_options(pl, count_cache_options);

    pl.add_option("kappa", "Learning decay rate", kappa)
      .add_option("tau0", "Learning offset", tau0)
      .add_option("max-iter", "Max iterations per doc", maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per doc", mDelta)
      .add_option("n-topics", "Number of topics", nTopics)
      .add_option("random-init-shape",
          "Shape of mean-one Gamma noise for legacy map-mean initialization",
          randomInitShape)
      .add_option("theta-concentration", "Total theta concentration alpha", thetaConcentration)
      .add_option("dictionary-prior-mass",
          "Total anti-collapse pseudocount mass per topic", dictionaryPriorMass)
      .add_option("regularization",
          "Token-normalized ownership-entropy regularization", regularization)
      .add_option("regularization-mode",
          "Ownership target: uniform or prevalence", regularizationMode)
      .add_option("regularize-warmup-epochs",
          "Epochs with zero regularization", regularizeWarmupEpochs)
      .add_option("regularize-ramp-epochs",
          "Epochs over which regularization reaches full strength",
          regularizeRampEpochs)
      .add_option("final-refine-passes",
          "Maximum full-data deterministic refinement passes", finalRefinePasses)
      .add_option("final-refine-tol",
          "Maximum dictionary row-L1 change for refinement convergence", finalRefineTol);
    pl.add_option("model-init",
        "Topic model TSV used only to initialize legacy map-mean beta means",
        modelInitFile);

    pl.add_option("dispersion-init-epochs", "Poisson warmup epochs before estimating dispersion", dispersionInitEpochs)
      .add_option("dispersion-estimator", "All-cell moment estimator: factorial or residual", dispersionEstimator)
      .add_option("dispersion-loess-span", "LOESS span for the dispersion abundance trend", dispersionLoessSpan)
      .add_option("dispersion-min-information", "Minimum adjusted squared-mean information Q for a raw dispersion estimate", dispersionMinInformation)
      .add_option("dispersion-outlier-sd", "Standard-deviation threshold for retaining high-dispersion outliers", dispersionOutlierSd)
      .add_option("dispersion-delta-min", "Lower bound for estimated NB2 dispersion phi", dispersionDeltaMin)
      .add_option("dispersion-delta-max", "Upper bound for estimated NB2 dispersion phi", dispersionDeltaMax);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (batchSize <= 0) batchSize = 512;
    if (nEpochs < 0) nEpochs = 1;
    validate_training_count_cache_options(
        count_cache_options.mode, count_cache_options.memory_budget);
    if (inStateFile.empty() && nTopics <= 0)
        error("--n-topics must be greater than 0 for a fresh fit");
    if (!inStateFile.empty() && nEpochs == 0 && finalRefinePasses == 0)
        error("State-based fitting with --n-epochs 0 requires --final-refine-passes");
    if (inStateFile.empty() && nEpochs == 0)
        error("A fresh fit requires --n-epochs greater than 0");
    if (!inStateFile.empty() && !modelInitFile.empty())
        error("--in-state and --model-init are mutually exclusive");
    if (!inStateFile.empty() && estimateDispersion)
        error("Estimate dispersion before state-based refinement, then supply it with --icol-dispersion");
    if (inferenceMode != "map-mean" && inferenceMode != "lda-compatible")
        error("--inference-mode must be map-mean or lda-compatible");
    if (maxIter <= 0 || !std::isfinite(mDelta) || mDelta <= 0.0)
        error("--max-iter and --mean-change-tol must be positive");
    if (randomizeOutput && !transform) {
        error("--randomize-output requires --transform");
    }
    if (cheapFeatureDiagnostics && !computeResiduals) {
        error("--feature-diagnostics-cheap requires --residuals");
    }
    if (unitSimilarityDiagnostics && !computeResiduals) {
        error("--unit-diagnostics-similarity requires --residuals");
    }
    if (!std::isfinite(thetaConcentration) || thetaConcentration <= 0.0) {
        error("--theta-concentration must be positive and finite");
    }
    if (!std::isfinite(kappa) || kappa <= 0.5 || kappa > 1.0)
        error("--kappa must be in (0.5, 1]");
    if (!std::isfinite(tau0) || tau0 <= 0.0)
        error("--tau0 must be positive and finite");
    if (!std::isfinite(randomInitShape) || randomInitShape <= 0.0) {
        error("--random-init-shape must be positive and finite");
    }
    if (!std::isfinite(dictionaryPriorMass)
        || (dictionaryPriorMass < 0.0 && dictionaryPriorMass != -1.0))
        error("--dictionary-prior-mass must be non-negative and finite when supplied");
    if (!std::isfinite(regularization) || regularization < 0.0)
        error("--regularization must be non-negative and finite");
    if (regularizationMode != "uniform"
            && regularizationMode != "prevalence") {
        error("--regularization-mode must be uniform or prevalence");
    }
    if (!inStateFile.empty() && pl.was_provided("regularization-mode")) {
        error("--in-state supplies the regularization mode; do not override it");
    }
    if (regularizeWarmupEpochs < 0 || regularizeRampEpochs < 0)
        error("Regularization warmup and ramp epochs must be non-negative");
    if (regularization > 0.0
        && nEpochs < regularizeWarmupEpochs + regularizeRampEpochs)
        error("--n-epochs must cover regularization warmup and ramp");
    if (finalRefinePasses < 0 || !std::isfinite(finalRefineTol)
        || finalRefineTol < 0.0)
        error("Final refinement passes and tolerance must be non-negative");
    if (icolDispersion >= 0 && featureFile.empty()) {
        error("--features is required when --icol-dispersion is non-negative");
    }
    if (estimateDispersion && icolDispersion >= 0) {
        error("--estimate-dispersion and --icol-dispersion are mutually exclusive");
    }
    if (estimateDispersion && (dispersionInitEpochs < 1 || dispersionInitEpochs >= nEpochs)) {
        error("--dispersion-init-epochs must be at least 1 and smaller than --n-epochs");
    }
    if (estimateDispersion && regularization > 0.0
        && regularizeWarmupEpochs < dispersionInitEpochs) {
        error("--regularize-warmup-epochs must cover --dispersion-init-epochs");
    }
    if (dispersionEstimator != "factorial" && dispersionEstimator != "residual") {
        error("--dispersion-estimator must be factorial or residual");
    }
    if (estimateDispersion && (!std::isfinite(dispersionMinInformation)
        || dispersionMinInformation <= 0.0 || !std::isfinite(dispersionOutlierSd)
        || dispersionOutlierSd < 0.0
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
    if (!use_10x && featureFile.empty()) {
        error("--features with per-feature total counts is required for custom "
            "Gamma-Poisson input");
    }
    if (!use_10x && (!reader.readFullSums
            || reader.getFeatureSumsRaw().size()
                != static_cast<size_t>(reader.nFeatures))) {
        error("--features must provide a non-negative total count for every "
            "retained feature when fitting custom Gamma-Poisson input");
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

    if (!inStateFile.empty()) {
        gp->initializeFromState(inStateFile, seed, nThreads, verbose,
            maxIter, mDelta);
        if (nTopics > 0 && nTopics != gp->getNumTopics()) {
            error("--n-topics does not match --in-state");
        }
    } else {
        GammaPoissonMapOptions mapOptions;
        mapOptions.inference_mode = inferenceMode == "lda-compatible"
            ? GammaPoissonInferenceMode::LdaCompatible
            : GammaPoissonInferenceMode::MapMean;
        if (dictionaryPriorMass < 0.0) {
            dictionaryPriorMass = mapOptions.inference_mode
                    == GammaPoissonInferenceMode::LdaCompatible
                ? static_cast<double>(gp->nFeatures())
                    / static_cast<double>(nTopics)
                : 1.0;
        }
        mapOptions.dictionary_prior_mass = dictionaryPriorMass;
        mapOptions.ownership_strength = regularization;
        mapOptions.ownership_mode = regularizationMode == "prevalence"
            ? GammaPoissonOwnershipMode::Prevalence
            : GammaPoissonOwnershipMode::Uniform;
        gp->initialize(nTopics, seed, nThreads, verbose, thetaConcentration,
            kappa, tau0, gp->nUnits(), maxIter, mDelta,
            randomInitShape, mapOptions);
        if (!modelInitFile.empty()) {
            if (mapOptions.inference_mode
                    == GammaPoissonInferenceMode::LdaCompatible) {
                error("Use --in-state, not --model-init, with lda-compatible inference");
            }
            gp->initializeFromModel(modelInitFile);
        }
    }
    if (icolDispersion >= 0) {
        gp->setFeatureDispersion(suppliedTau);
        notice("Using per-feature dispersion tau from column %d of %s",
            icolDispersion, featureFile.c_str());
    }

    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    const int32_t documentsPerEpoch = debug_ > 0
        ? debug_ : std::max<int32_t>(1, gp->nUnits());
    gp->configureOwnershipAnnealing(regularizeWarmupEpochs,
        regularizeRampEpochs, documentsPerEpoch);
    const int32_t count_passes =
        nEpochs + (estimateDispersion ? 1 : 0) + finalRefinePasses;
    TrainingCountCache count_cache(count_cache_options,
        use_10x, count_passes, gp->nFeatures());
    notice("Starting Gamma-Poisson model training....");
    for (int epoch = 0; epoch < nEpochs; ++epoch) {
        int32_t n = 0;
        if (use_10x) {
            n = gp->trainOnline10X(batchSize, maxUnits, seed + epoch);
        } else if (epoch == 0 && count_cache.enabled()) {
            n = gp->trainOnline(inFile, batchSize, minCountTrain,
                maxUnits, count_cache.sink());
            count_cache.finish();
        } else if (count_cache.resident_batches()) {
            n = gp->trainOnline(
                *count_cache.resident_batches(), batchSize, maxUnits);
        } else if (count_cache.source()) {
            n = gp->trainOnline(
                *count_cache.source(), batchSize, maxUnits);
        } else {
            n = gp->trainOnline(inFile, batchSize, minCountTrain, maxUnits);
        }
        notice("Epoch %d/%d, processed %d documents", epoch + 1, nEpochs, n);
        if (epoch == 0 && n > 0 && n != documentsPerEpoch) {
            gp->configureOwnershipAnnealing(regularizeWarmupEpochs,
                regularizeRampEpochs, n, n);
        }
        gp->printTopicAbundance();
        const auto diagnostics = gp->optimizationDiagnostics();
        notice("Gamma-Poisson MAP: objective %.6g, penalty entropy %.6g, "
            "uniform entropy %.6g, lambda %.6g, max row change %.3g, "
            "MM %d steps/%d fallbacks, relative gain %.3g, gradient steps "
            "%d accepted/%d failed, L-BFGS %d steps/%d fallbacks",
            diagnostics.objective, diagnostics.ownership_entropy,
            diagnostics.uniform_ownership_entropy,
            diagnostics.effective_ownership_lambda,
            diagnostics.maximum_row_change,
            diagnostics.mm_steps, diagnostics.mm_fallbacks,
            diagnostics.relative_objective_gain,
            diagnostics.accepted_gradient_steps,
            diagnostics.failed_gradient_steps, diagnostics.lbfgs_steps,
            diagnostics.lbfgs_fallbacks);
        if (estimateDispersion && epoch + 1 == dispersionInitEpochs) {
            GammaPoissonDispersionOptions options;
            options.estimator = dispersionEstimator == "factorial"
                ? GammaPoissonDispersionEstimatorKind::Factorial
                : GammaPoissonDispersionEstimatorKind::Residual;
            options.min_information = dispersionMinInformation;
            options.outlier_sd = dispersionOutlierSd;
            options.loess_span = dispersionLoessSpan;
            options.delta_min = dispersionDeltaMin;
            options.delta_max = dispersionDeltaMax;
            options.adjust_marginal_gain = false;
            GammaPoissonDispersionResult dispersion = [&]() {
                if (use_10x) {
                    return gp->estimateFeatureDispersion10X(
                        options, batchSize, maxUnits);
                }
                if (count_cache.resident_batches()) {
                    return gp->estimateFeatureDispersion(options,
                        *count_cache.resident_batches(), maxUnits);
                }
                if (count_cache.source()) {
                    return gp->estimateFeatureDispersion(options,
                        *count_cache.source(), batchSize, maxUnits);
                }
                return gp->estimateFeatureDispersion(options, inFile,
                    batchSize, minCountTrain, maxUnits);
            }();
            const std::string outDispersion = outPrefix + ".dispersion.tsv";
            write_gamma_poisson_dispersion_diagnostics(outDispersion,
                gp->getFeatureNames(), dispersion);
            notice("Estimated per-feature dispersion from %d documents; diagnostics written to %s",
                dispersion.n_documents, outDispersion.c_str());
        }
    }
    if (finalRefinePasses > 0) {
        gp->setOwnershipAnnealing(1.0);
        for (int32_t pass = 0; pass < finalRefinePasses; ++pass) {
            gp->beginFullRefinement();
            int32_t n = 0;
            if (use_10x) {
                n = gp->trainOnline10X(batchSize, maxUnits,
                    seed + nEpochs + pass);
            } else if (count_cache.resident_batches()) {
                n = gp->trainOnline(
                    *count_cache.resident_batches(), batchSize, maxUnits);
            } else if (count_cache.source()) {
                n = gp->trainOnline(*count_cache.source(), batchSize, maxUnits);
            } else {
                n = gp->trainOnline(
                    inFile, batchSize, minCountTrain, maxUnits);
            }
            const bool converged = gp->finishFullRefinement(finalRefineTol);
            const auto diagnostics = gp->optimizationDiagnostics();
            notice("Final refinement pass %d/%d, processed %d documents%s",
                pass + 1, finalRefinePasses, n,
                converged ? ", converged" : "");
            notice("Gamma-Poisson refinement MAP: objective %.6g, penalty "
                "entropy %.6g, uniform entropy %.6g, lambda %.6g, max row "
                "change %.3g, MM %d steps/%d fallbacks, relative gain %.3g, "
                "L-BFGS %d steps/%d fallbacks",
                diagnostics.objective, diagnostics.ownership_entropy,
                diagnostics.uniform_ownership_entropy,
                diagnostics.effective_ownership_lambda,
                diagnostics.maximum_row_change,
                diagnostics.mm_steps, diagnostics.mm_fallbacks,
                diagnostics.relative_objective_gain, diagnostics.lbfgs_steps,
                diagnostics.lbfgs_fallbacks);
            if (finalRefineTol > 0.0 && converged) break;
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
        std::vector<std::string> args;
        args.push_back("gamma-pois-transform");
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
        append_arg(args, "--temp-dir", count_cache_options.temp_dir);
        append_arg(args, "--seed", seed);
        append_arg(args, "--debug", debug_);
        append_arg(args, "--verbose", verbose);
        append_arg(args, "--features", featureFile);
        append_arg(args, "--min-count-per-feature", minCountFeature);
        append_arg(args, "--min-count", minCountTrain);
        append_arg(args, "--default-weight", defaultWeight);
        append_arg(args, "--icol-weight", icolWeight);
        append_arg(args, "--include-feature-regex", include_ftr_regex);
        append_arg(args, "--exclude-feature-regex", exclude_ftr_regex);
        append_arg(args, "--max-iter", maxIter);
        append_arg(args, "--mean-change-tol", mDelta);
        args.push_back("--use-stored-dispersion");
        args.push_back("--use-training-prevalence");
        if (randomizeOutput) args.push_back("--randomize-output");
        if (pseudobulkAllFeatures) args.push_back("--pseudobulk-all-features");
        if (computeResiduals) {
            args.push_back("--residuals");
            if (cheapFeatureDiagnostics) {
                warning("--feature-diagnostics-cheap has no effect when "
                    "transforming fitted training data");
            }
        }
        if (unitSimilarityDiagnostics) {
            args.push_back("--unit-diagnostics-similarity");
        }
        std::vector<char*> cargs;
        cargs.reserve(args.size());
        for (auto& arg : args) cargs.push_back(arg.data());
        const int32_t rc = cmdGammaPoisTransform(
            static_cast<int32_t>(cargs.size()), cargs.data());
        if (rc != 0) return rc;
    }
    return 0;
}
