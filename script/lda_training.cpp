#include "topic_svb.hpp"

int32_t cmdTopicModelSVI(int argc, char** argv) {

    // --- Model Selection ---
    std::string model_type = "lda";
    // --- Common Parameters ---
    std::string inFile, metaFile, weightFile, outPrefix, priorFile, featureFile;
    std::string include_ftr_regex;
    std::string exclude_ftr_regex;
    int32_t seed = -1;
    int32_t nEpochs = 1, batchSize = 512;
    int32_t verbose = 0;
    int32_t nThreads = 0;
    int32_t modal = 0;
    int32_t minCountTrain = 20, minCountFeature = 1;
    double defaultWeight = 1.;
    bool transform = false;
    bool sort_topics = false;
    // --- Algorithm Parameters (some are shared) ---
    double kappa = 0.7, tau0 = 10.0;
    double alpha = -1., eta = -1.;
    int32_t maxIter = 100;
    double  mDelta = 1e-3;
    // --- LDA-Specific Parameters ---
    int32_t nTopics = 0;
    double priorScale = 1.;
    bool projection_only = false;
    bool useSCVB0 = false;
    // SCVB0 specific parameters
    int32_t z_burnin = 10;
    double s_beta = 10, s_theta = 1, kappa_theta = 0.9, tau_theta = 10;
    // --- HDP-Specific Parameters ---
    int32_t max_topics_K = 100;
    int32_t doc_trunc_T = 10;
    double hdp_alpha = 1.0;
    double hdp_omega = 1.0;
    double topic_threshold = 1e-8;
    double topic_coverage  = 1.0 - 1e-8;

    ParamList pl;
    // --- Command-Line Option Definitions ---
    pl.add_option("model-type", "Type of topic model to train [lda|hdp]", model_type);

    // Group: Input/Output Options (Common)
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true)
      .add_option("out-prefix", "Output prefix for model and results files", outPrefix, true)
      .add_option("transform", "Transform data to topic space after training", transform)
      .add_option("sort-topics", "Sort topics by weight after training", sort_topics);

    // Group: Feature Preprocessing Options (Common)
    pl.add_option("feature-weights", "Input weights file", weightFile)
      .add_option("features", "Feature names and total counts file", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", minCountFeature)
      .add_option("default-weight", "Default weight for features not in weight file", defaultWeight)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex);

    // Group: General Training Options (Common)
    pl.add_option("seed", "Random seed", seed)
      .add_option("threads", "Number of threads", nThreads)
      .add_option("n-epochs", "Number of epochs", nEpochs)
      .add_option("minibatch-size", "Minibatch size", batchSize)
      .add_option("min-count-train", "Minimum total feature count for a document to be trained", minCountTrain)
      .add_option("modal", "Modality to use (0-based)", modal)
      .add_option("verbose", "Verbose level", verbose);

    // Group: Algorithm Hyperparameters (Model-Specific)
    pl.add_option("kappa", "(All) Learning decay rate", kappa)
      .add_option("tau0", "(All) Learning offset", tau0)
      .add_option("eta", "(LDA/HDP) Topic-word prior. LDA default: 1/K, HDP default: 0.01", eta)
      .add_option("max-iter", "(LDA-SVB/HDP) Max iterations per doc", maxIter)
      .add_option("mean-change-tol", "(LDA-SVB/HDP) Convergence tolerance per doc", mDelta)
      // LDA Options
      .add_option("n-topics", "(LDA) Number of topics", nTopics)
      .add_option("alpha", "(LDA) Document-topic prior (default: 1/K)", alpha)
      .add_option("scvb0", "(LDA) Use SCVB0 inference instead of SVB", useSCVB0)
      // HDP Options
      .add_option("max-topics", "(HDP) Maximum number of topics (K)", max_topics_K)
      .add_option("doc-trunc-level", "(HDP) Document topic truncation level (T)", doc_trunc_T)
      .add_option("hdp-alpha", "(HDP) Document-level concentration", hdp_alpha)
      .add_option("hdp-omega", "(HDP) Corpus-level concentration", hdp_omega)
      .add_option("topic-threshold", "(HDP Output) Only output topics with relative weight > threshold", topic_threshold)
      .add_option("topic-coverage", "(HDP Output) Output top topics that explain this proportion of the data", topic_coverage);

    // Group: LDA-Specific Advanced Options
    pl.add_option("model-prior", "(LDA) File with initial model matrix for continued training", priorFile)
      .add_option("prior-scale", "(LDA) Uniform scaling factor for the prior model matrix", priorScale)
      .add_option("projection-only", "(LDA) Transform data using prior model without training", projection_only);
    pl.add_option("s-beta", "(LDA-SCVB0) Step size scheduler 's' for global params", s_beta)
      .add_option("s-theta", "(LDA-SCVB0) Step size scheduler 's' for local params", s_theta)
      .add_option("kappa-theta", "(LDA-SCVB0) Step size scheduler 'kappa' for local params", kappa_theta)
      .add_option("tau-theta", "(LDA-SCVB0) Step size scheduler 'tau' for local params", tau_theta)
      .add_option("z-burnin", "(LDA-SCVB0) Burn-in iterations for latent variables", z_burnin);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
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

    std::unique_ptr<TopicModelWrapper> model_runner;

    if (model_type == "lda") {
        if (projection_only) {
            transform = true;
        }
        if (nTopics <= 0 && priorFile.empty()) {
            error("Number of topics must be greater than 0");
        }
        auto lda4hex = new LDA4Hex(metaFile, modal);
        if (!featureFile.empty()) {
            lda4hex->setFeatures(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
        }

        if (useSCVB0) {
            lda4hex->initialize_scvb0(nTopics, seed, nThreads, verbose,
                alpha, eta, kappa, tau0, lda4hex->nUnits(),
                priorFile, priorScale, s_beta, s_theta, kappa_theta, tau_theta, z_burnin);
        } else {
            lda4hex->initialize_svb(nTopics, seed, nThreads, verbose,
                alpha, eta, kappa, tau0, lda4hex->nUnits(),
                priorFile, priorScale, maxIter, mDelta);
        }
        model_runner.reset(lda4hex);
    } else if (model_type == "hdp") {
        sort_topics = true;
        if (projection_only || !priorFile.empty()) warning("--projection-only and --model-prior are not supported for HDP and will be ignored.");
        if (max_topics_K <= 0) error("For HDP, --max-topics must be > 0.");
        if (doc_trunc_T <= 0) error("For HDP, --doc-trunc-level must be > 0.");

        // Instantiate HDP model runner
        auto hdp4hex = new HDP4Hex(metaFile, modal);
        if (!featureFile.empty()) {
            hdp4hex->setFeatures(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
        }
        hdp4hex->initialize(max_topics_K, doc_trunc_T, seed, nThreads, verbose,
            eta, hdp_alpha, hdp_omega, kappa, tau0, hdp4hex->nUnits(), maxIter, mDelta);
        model_runner.reset(hdp4hex);
    } else {
        error("Unknown model type: '%s'. Choose 'lda' or 'hdp'.", model_type.c_str());
    }
    if (!weightFile.empty()) {
        model_runner->setWeights(weightFile, defaultWeight);
    }
    if (!projection_only) {
        std::string outModel = outPrefix + ".model.tsv";
        if (!priorFile.empty() && priorFile == outModel) {
            outModel = outPrefix + ".model.updated.tsv";
        }
        std::ofstream outFileStream(outModel);
        if (!outFileStream) {
            error("Error opening output file: %s for writing", outModel.c_str());
        }
        outFileStream.close();
        if (std::filesystem::exists(outModel)) {
            std::filesystem::remove(outModel);
        }
        // Training
        for (int epoch = 0; epoch < nEpochs; ++epoch) {
            int32_t n = model_runner->trainOnline(inFile, batchSize, minCountTrain);
            notice("Epoch %d/%d, processed %d documents", epoch + 1, nEpochs, n);
            std::vector<double> weights;
            model_runner->getTopicAbundance(weights);
            std::sort(weights.begin(), weights.end(), std::greater<double>());
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4);
            for (size_t i = 0; i < std::min<size_t>(10, weights.size()); ++i) {
                ss << weights[i] << "\t";
            }
            notice("  Top topic relative abundance: %s", ss.str().c_str());
        }
        if (model_type == "lda" && sort_topics) {
            model_runner->sortTopicsByWeight();
        }
        // --- Post-processing for HDP ---
        if (model_type == "hdp") {
            auto* hdp_ptr = dynamic_cast<HDP4Hex*>(model_runner.get());
            if (hdp_ptr) {
                hdp_ptr->filterTopics(topic_threshold, topic_coverage);
            }
        }

        // write model matrix to file
        model_runner->writeModelToFile(outModel);
        notice("Model written to %s", outModel.c_str());
    }

    if (transform) {
        std::string outFile = outPrefix + ".results.tsv";
        model_runner->fitAndWriteToFile(inFile, outFile, batchSize);
        notice("Transformed data written to %s", outFile.c_str());
    }

    return 0;
};
