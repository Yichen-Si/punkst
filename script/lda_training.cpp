#include "punkst.h"
#include "lda.hpp"
#include "dataunits.hpp"
#include <regex>

class LDA4Hex {

public:

    LDA4Hex(const std::string& metaFile, int32_t modal = 0) : reader(metaFile), modal(modal) {
        if (modal >= reader.getNmodal()) {
            error("modal %d is out of range", modal);
        }
        weightFeatures = false;
        ntot = 0;
        minCountTrain = 0;
        M_ = reader.nFeatures;
        usePriorMapping = false;
        initialized = false;
    }

    void setFeatures(const std::string& featureFile, int32_t minCount, std::string& include_ftr_regex, std::string& exclude_ftr_regex) {
        if (usePriorMapping) {
            warning("setFeatures should be called before initialization. The feature filtering will not be applied");
            return;
        }
        bool check_include = !include_ftr_regex.empty();
        bool check_exclude = !exclude_ftr_regex.empty();
        std::regex regex_include(include_ftr_regex);
        std::regex regex_exclude(exclude_ftr_regex);
        std::ifstream inFeature(featureFile);
        if (!inFeature) {
            error("Error opening features file: %s", featureFile.c_str());
        }
        std::string line;
        uint32_t idx0 = 0, idx1 = 0;
        std::unordered_map<uint32_t, uint32_t> idx_remap;
        std::unordered_map<std::string, uint32_t> featureDict;
        featureNames.clear();
        if (reader.featureDict(featureDict)) {
            while (std::getline(inFeature, line)) {
                std::istringstream iss(line);
                std::string feature;
                int32_t count;
                if (!(iss >> feature >> count)) {
                    error("Error reading feature file at line: %s", line.c_str());
                }
                if (count < minCount) {
                    continue;
                }
                auto it = featureDict.find(feature);
                if (it == featureDict.end()) {
                    continue;
                }
                if (check_include && !std::regex_match(feature, regex_include)) {
                    continue;
                }
                if (check_exclude && std::regex_match(feature, regex_exclude)) {
                    std::cout << "Exclude " << feature << std::endl;
                    continue;
                }
                idx_remap[it->second] = idx1++;
                featureNames.push_back(feature);
            }
        } else {
            while (std::getline(inFeature, line)) {
                std::istringstream iss(line);
                std::string feature;
                int32_t count;
                if (!(iss >> feature >> count)) {
                    error("Error reading feature file at line: %s", line.c_str());
                }
                if (count < minCount) {
                    idx0++;
                    continue;
                }
                bool include = !check_include || std::regex_match(feature, regex_include);
                bool exclude = check_exclude && std::regex_match(feature, regex_exclude);
                if (include && !exclude) {
                    idx_remap[idx0] = idx1++;
                    featureNames.push_back(feature);
                }
                idx0++;
            }
        }
        M_ = idx_remap.size();
        notice("%s: %d features are kept out of %d", __FUNCTION__, idx1, idx0);
        reader.setFeatureIndexRemap(idx_remap);
    }

    void setWeights(const std::string& weightFile, double defaultWeight_ = 1.0) {
        std::ifstream inWeight(weightFile);
        if (!inWeight) {
            error("Error opening weights file: %s", weightFile.c_str());
        }
        defaultWeight = defaultWeight_;
        int32_t nweighted = 0, novlp = 0;
        weights.resize(reader.nFeatures);
        std::fill(weights.begin(), weights.end(), defaultWeight);
        std::unordered_map<std::string, uint32_t> featureDict;
        bool foundDict = reader.featureDict(featureDict);
        if (!foundDict) { // then the weight file must refer to features by their index as in the input file
            warning("Feature dictionary is not found in the input metadata, so we assume the weight file (--feature-weights) refers to features by their index as in the input file");
            std::string line;
            while (std::getline(inWeight, line)) {
                nweighted++;
                std::istringstream iss(line);
                uint32_t idx;
                double weight;
                if (!(iss >> idx >> weight)) {
                    error("Error reading weights file at line: %s", line.c_str());
                }
                if (idx >= reader.nFeatures) {
                    warning("Input file contains %zu features, feature index %u in the weights file is out of range", reader.nFeatures, idx);
                    continue;
                }
                weights[idx] = weight;
                novlp++;
            }
        } else { // assume the weight file refers to features by their names
            std::string line;
            while (std::getline(inWeight, line)) {
                nweighted++;
                std::istringstream iss(line);
                std::string feature;
                double weight;
                if (!(iss >> feature >> weight)) {
                    error("Error reading weights file at line: %s", line.c_str());
                }
                auto it = featureDict.find(feature);
                if (it != featureDict.end()) {
                    weights[it->second] = weight;
                    novlp++;
                } else {
                    warning("Feature %s not found in the input", feature.c_str());
                }
            }
        }
        notice("Read %d weights from file, %d features are overlapped with the input file", nweighted, novlp);
        if (novlp == 0) {
            error("No features in the weight file overlap with those found in the input file, check if the files are consistent.");
        }
        weightFeatures = true;
    }

    void initialize(int32_t nTopics, int32_t seed = -1,
        int32_t nThreads = 0, int32_t verbose = 0,
        double alpha = -1., double eta = -1.,
        int32_t maxIter = 100, double mDelta = -1.,
        double kappa = 0.7, double tau0 = 10.0, int32_t totalDocCount = 1000000,
        const std::string& priorFile = "", double priorScale = 1.) {

        std::optional<MatrixXd> priorMatrix = std::nullopt;

        // Handle prior model feature mapping if needed
        if (!priorFile.empty()) {
            // Read prior model file
            std::vector<std::string> priorFeatureNames;
            std::vector<std::uint32_t> kept_indices;
            MatrixXd fullPriorMatrix;
            readModelFromTsv(priorFile, priorFeatureNames, fullPriorMatrix);

            // Setup feature mapping between input data and prior model
            setupPriorMapping(priorFeatureNames, kept_indices);

            // Create subset matrix for only the intersected features
            priorMatrix = MatrixXd(K_, M_);
            // Map columns from full prior matrix to subset matrix
            for (size_t i = 0; i < kept_indices.size(); ++i) {
                priorMatrix->col(i) = fullPriorMatrix.col(kept_indices[i]);
            }
            // Apply scaling if specified
            if (priorScale > 0. && priorScale != 1.) {
                *priorMatrix *= priorScale;
            }

            notice("Created subset prior matrix: %d topics x %d features (from original %d features)",
                    (int)priorMatrix->rows(), (int)priorMatrix->cols(), (int)priorFeatureNames.size());
        } else {
            K_ = nTopics;
            if (reader.features.size() != M_) {
                notice("%s: no valid feature names are set, will use 0-based indices in the output model file", __FUNCTION__);
                featureNames.resize(M_);
                for (int i = 0; i < M_; ++i) {
                    featureNames[i] = std::to_string(i);
                }
            } else {
                featureNames = reader.features;
            }
        }

        // Create the LDA object with correct dimensions and subset prior matrix
        lda = std::make_unique<LatentDirichletAllocation>(
            K_, M_, seed, nThreads, verbose, alpha, eta, maxIter, mDelta,
            kappa, tau0, totalDocCount, std::nullopt, priorMatrix, -1.);

        initialized = true;
    }

    int32_t trainOnline(const std::string& inFile, int32_t _bsize = 512, int32_t _minCountTrain = 0) {
        if (!initialized || !lda) {
            error("LDA4Hex must be initialized before training");
        }
        batchSize = _bsize;
        minCountTrain = _minCountTrain;
        ntot = 0;
        std::ifstream inFileStream(inFile);
        if (!inFileStream) {
            error("Error opening input file: %s", inFile.c_str());
        }
        bool fileopen = true;
        int nbatch = 0;
        while (fileopen) {
            fileopen = readMinibatch(inFileStream);
            if (minibatch.empty()) {
                break;
            }
            lda->partial_fit(minibatch);
            ntot += minibatch.size();
            nbatch++;
        }
        inFileStream.close();
        return ntot;
    }

    void writeModelToFile(const std::string& outFile) {
        if (!initialized || !lda) {
            error("LDA4Hex must be initialized before writing model");
        }
        std::ofstream outFileStream(outFile);
        if (!outFileStream) {
            error("Error opening output file: %s for writing", outFile.c_str());
        }
        const MatrixXd& model = lda->get_model();
        if (topicNames.size() != K_) {
            topicNames = lda->get_topic_names();
        }
        outFileStream << "Feature";
        for (int i = 0; i < K_; ++i) {
            outFileStream << "\t" << topicNames[i];
        }
        outFileStream << "\n";
        outFileStream << std::fixed << std::setprecision(3);
        for (int i = 0; i < M_; ++i) {
            outFileStream << featureNames[i];
            for (int j = 0; j < K_; ++j) {
                outFileStream << "\t" << model(j, i);
            }
            outFileStream << "\n";
        }
        outFileStream.close();
    }

    void fitAndWriteToFile(const std::string& inFile, const std::string& outFile, int32_t _bsize = 512) {
        if (!initialized || !lda) {
            error("LDA4Hex must be initialized before fitting");
        }
        batchSize = _bsize;
        std::ifstream inFileStream(inFile);
        if (!inFileStream) {
            error("Error opening input file: %s", inFile.c_str());
        }
        std::ofstream outFileStream(outFile);
        if (!outFileStream) {
            error("Error opening output file: %s for writing", outFile.c_str());
        }
        bool labeled = reader.getNlayer() > 1;
        std::string header;
        reader.getInfoHeaderStr(header);
        outFileStream << header;
        int32_t nTopics = lda->get_n_topics();
        for (int i = 0; i < nTopics; ++i) {
            outFileStream << "\t" << i;
        }
        outFileStream << "\n";

        bool fileopen = true;
        while (fileopen) {
            std::vector<std::string> idens;
            fileopen = readMinibatch(inFileStream, idens, labeled);
            if (minibatch.empty()) {
                break;
            }
            Eigen::MatrixXd doc_topic = lda->transform(minibatch);
            for (int i = 0; i < doc_topic.rows(); ++i) {
                double sum = doc_topic.row(i).sum();
                if (sum > 0) {
                    doc_topic.row(i) /= sum;
                }
            }
            for (int i = 0; i < minibatch.size(); ++i) {
                outFileStream << idens[i] << std::fixed << std::setprecision(4);
                if (idens[i].size() > 0) {
                    outFileStream << "\t";
                }
                outFileStream << doc_topic(i, 0);
                for (int j = 1; j < nTopics; ++j) {
                    outFileStream << "\t" << doc_topic(i, j);
                }
                outFileStream << "\n";
            }
        }
        inFileStream.close();
        outFileStream.close();
    }

    int32_t nUnits() const {
        return reader.nUnits;
    }
    int32_t nFeatures() const {
        return M_;
    }

    void getTopicAbundance(std::vector<double>& weights) {
        if (!initialized || !lda) {
            error("%s: LDA4Hex is not initialized", __FUNCTION__);
        }
        lda->get_topic_abundance(weights);
    }

private:

    HexReader reader;

    int32_t modal;
    int32_t ntot;
    int32_t K_, M_;
    int32_t minCountTrain;
    bool weightFeatures;
    bool usePriorMapping;
    bool initialized;
    double defaultWeight;
    std::vector<double> weights;
    std::vector<std::string> featureNames;
    std::vector<std::string> topicNames;
    std::vector<uint32_t> featureIdxUsed;
    std::vector<int32_t> priorFeatureMapping;
    std::unique_ptr<LatentDirichletAllocation> lda;

    std::vector<Document> minibatch;
    int32_t batchSize;

    void readModelFromTsv(const std::string& modelFile, std::vector<std::string>& featureNames, MatrixXd& modelMatrix) {
        std::ifstream modelIn(modelFile, std::ios::in);
        if (!modelIn) {
            error("Error opening model file: %s", modelFile.c_str());
        }

        std::string line;
        std::vector<std::string> tokens;

        // Read header to get topic names and count
        std::getline(modelIn, line);
        split(tokens, "\t", line);
        K_ = tokens.size() - 1; // first column is "Feature"
        topicNames = std::vector<std::string>(tokens.begin() + 1, tokens.end());

        // Read all feature rows
        featureNames.clear();
        std::vector<std::vector<double>> modelValues;
        while (std::getline(modelIn, line)) {
            split(tokens, "\t", line);
            if (tokens.size() != K_ + 1) {
                error("Error reading model file at line: %s", line.c_str());
            }
            featureNames.push_back(tokens[0]);
            std::vector<double> values(K_);
            for (int32_t i = 0; i < K_; ++i) {
                values[i] = std::stod(tokens[i + 1]);
            }
            modelValues.push_back(values);
        }
        modelIn.close();

        int32_t nFeatures = featureNames.size();
        modelMatrix.resize(K_, nFeatures);
        for (int32_t i = 0; i < nFeatures; ++i) {
            for (int32_t j = 0; j < K_; ++j) {
                modelMatrix(j, i) = modelValues[i][j];
            }
        }

        notice("Read model matrix: %d topics x %d features from %s", K_, nFeatures, modelFile.c_str());
    }

    // Create a new feature space that is the intersection of:
    // 1. Current filtered features (from setFeatures)
    // 2. The input features
    // The output will follow the input's ordering
    void setupPriorMapping(std::vector<std::string>& feature_names_, std::vector<std::uint32_t>& kept_indices) {
        // Use the current filtered feature set (from setFeatures if called, or original otherwise)
        std::vector<std::string> currentFeatures;
        if (!featureNames.empty()) {
            // setFeatures was called - use the filtered feature names
            currentFeatures = featureNames;
            notice("Using %d features after setFeatures filtering", (int)currentFeatures.size());
        } else {
            // No filtering applied - use original features from reader
            currentFeatures = reader.features;
            notice("No feature filtering applied, using all %d features", (int)currentFeatures.size());
        }

        if (currentFeatures.empty()) {
            notice("Input data has no feature names, cannot create feature mapping");
            return;
        }

        std::vector<std::string> mappedFeatureNames;

        // For each feature in the prior model, check if it exists in our filtered features
        kept_indices.clear();
        uint32_t idx = 0;
        for (const std::string& priorFeatureName : feature_names_) {
            // Find this feature in our current filtered features
            auto it = std::find(currentFeatures.begin(), currentFeatures.end(), priorFeatureName);
            if (it != currentFeatures.end()) {
                mappedFeatureNames.push_back(priorFeatureName);
                kept_indices.push_back(idx);
            }
            idx++;
        }

        if (mappedFeatureNames.empty()) {
            error("%s: No features overlap between filtered input features and prior model", __FUNCTION__);
        }
        notice("Found %d features in intersection of filtered features (%d) and queries (%d)",
               (int)mappedFeatureNames.size(), (int)currentFeatures.size(), (int)feature_names_.size());

        usePriorMapping = true;

        // Update to use the mapped feature space (preserves prior model ordering)
        featureNames = mappedFeatureNames;
        M_ = mappedFeatureNames.size();

        // Create a new reader remap that maps from original reader indices
        // to our final intersected feature space
        std::unordered_map<uint32_t, uint32_t> readerRemap;
        uint32_t newIdx = 0;

        // Map from original reader feature indices to the new intersected feature space
        for (const std::string& featureName : mappedFeatureNames) {
            // Find this feature in the original reader features
            auto origIt = std::find(reader.features.begin(), reader.features.end(), featureName);
            if (origIt != reader.features.end()) {
                uint32_t readerOrigIdx = std::distance(reader.features.begin(), origIt);
                readerRemap[readerOrigIdx] = newIdx++;
            }
        }

        // This will overwrite any previous setFeatureIndexRemap
        reader.setFeatureIndexRemap(readerRemap);

        if (weightFeatures) { // reset weights
            std::vector<double> newWeights(M_, defaultWeight);
            for (const auto& pair : readerRemap) {
                uint32_t origIdx = pair.first;
                if (origIdx < weights.size()) {
                    newWeights[pair.second] = weights[origIdx];
                }
            }
            weights = std::move(newWeights);
        }
    }

    bool readMinibatch(std::ifstream& inFileStream) {
        minibatch.clear();
        minibatch.reserve(batchSize);
        std::string line;
        int32_t nlocal = 0;
        while (nlocal < batchSize) {
            if (!std::getline(inFileStream, line)) {
                return false;
            }
            Document doc;
            int32_t ct = reader.parseLine(doc, line, modal);
            if (ct < 0) {
                error("Error parsing line %s", line.c_str());
            }
            if (ct < minCountTrain) {
                continue;
            }
            if (weightFeatures) {
                for (size_t i = 0; i < doc.ids.size(); ++i) {
                    doc.cnts[i] *= weights[doc.ids[i]];
                }
            }
            minibatch.push_back(std::move(doc));
            nlocal++;
        }
        return true;
    }

    bool readMinibatch(std::ifstream& inFileStream, std::vector<std::string>& idens, bool labeled = false) {
        minibatch.clear();
        minibatch.reserve(batchSize);
        std::string line;
        int32_t nlocal = 0;
        while (nlocal < batchSize) {
            if (!std::getline(inFileStream, line)) {
                return false;
            }
            Document doc;
            std::string info;
            int32_t ct = reader.parseLine(doc, info, line, modal);
            if (ct < 0) {
                error("%s: Error parsing the %d-th line", __FUNCTION__, ntot+nlocal);
            }
            idens.push_back(info);
            if (weightFeatures) {
                for (size_t i = 0; i < doc.ids.size(); ++i) {
                    doc.cnts[i] *= weights[doc.ids[i]];
                }
            }
            minibatch.push_back(std::move(doc));
            nlocal++;
        }
        return true;
    }

};

int32_t cmdLDA4Hex(int argc, char** argv) {

    std::string inFile, metaFile, weightFile, outPrefix, priorFile, featureFile;
    std::string include_ftr_regex;
    std::string exclude_ftr_regex;
    int32_t nTopics = 0;
    int32_t seed = -1;
    int32_t nEpochs = 1, batchSize = 512;
    int32_t verbose = 0;
    int32_t maxIter = 100;
    int32_t nThreads = 0;
    int32_t modal = 0;
    int32_t minCountTrain = 20, minCountFeature = 1;
    double kappa = 0.7, tau0 = 10.0;
    double alpha = -1., eta = -1.;
    double mDelta = 1e-3;
    double defaultWeight = 1.;
    double priorScale = 1.;
    bool transform = false;
    bool projection_only = false;

    ParamList pl;
    // Input Options
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true)
      .add_option("feature-weights", "Input weights file", weightFile)
      .add_option("features", "Feature names and total counts", featureFile)
      .add_option("n-topics", "Number of topics", nTopics)
      .add_option("modal", "Modality to use (0-based)", modal)
      .add_option("seed", "Random seed", seed)
      .add_option("threads", "Number of threads to use (default: 1)", nThreads)
      .add_option("min-count-train", "Minimum total count for training (default: 20)", minCountTrain)
      .add_option("min-count-per-feature", "Minimum total count for features to be included. Require --features (default: 1)", minCountFeature)
      .add_option("default-weight", "Default weight for features not in the provided weight file (default: 1.0, set it to 0 to ignore features not in the weight file)", defaultWeight)
      .add_option("include-feature-regex", "Include features that match this regex (grammar: Modified ECMAScript) (default: all features)", include_ftr_regex)
      .add_option("exclude-feature-regex", "Exclude features that match this regex (grammar: Modified ECMAScript) (default: none)", exclude_ftr_regex);
    // Model training options
      pl.add_option("kappa", "Learning decay (default: 0.7)", kappa)
      .add_option("tau0", "Learning offset (default: 10.0)", tau0)
      .add_option("max-iter", "Maximum number of iterations for each document (default: 100)", maxIter)
      .add_option("mean-change-tol", "Mean change of document-topic probability tolerance for convergence (default: 1e-3)", mDelta)
      .add_option("n-epochs", "Number of epochs (default: 1)", nEpochs)
      .add_option("minibatch-size", "Minibatch size (default: 512)", batchSize)
      .add_option("alpha", "Document-topic prior (default: 1/K)", alpha)
      .add_option("eta", "Topic-word prior (default: 1/K)", eta);
    // Start from a prior model
    pl.add_option("model-prior", "File that contains the initial model matrix. Caution: you may need to set model training parameters --kappa (learning decay) and --tau0 (learning offset) carefully to get the desired performance in terms of the balance between the prior and your data", priorFile)
      .add_option("prior-scale", "Scale the initial model matrix uniformly by this value (default: use the matrix as it is)", priorScale);
    // Output Options
    pl.add_option("out-prefix", "Output hex file", outPrefix, true)
      .add_option("verbose", "Verbose", verbose)
      .add_option("transform", "Transform the data to the LDA space after training", transform)
      .add_option("projection-only", "Transform the data using the prior model without further training", projection_only);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (projection_only) {
        transform = true;
    }
    if (nTopics <= 0 && priorFile.empty()) {
        error("Number of topics must be greater than 0");
    }
    if (seed <= 0) {
        seed = std::random_device{}();
    }
    if (batchSize <= 0) {
        batchSize = 512;
        warning("Minibatch size must be greater than 0, using default value of 128");
    }
    if (nEpochs <= 0) {
        nEpochs = 1;
    }
    std::string outModel = outPrefix + ".model.tsv";
    if (!projection_only) {
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
    }

    LDA4Hex lda4hex(metaFile, modal);
    if (!featureFile.empty()) {
        lda4hex.setFeatures(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
    }

    // Initialize the LDA object (handles feature name mapping internally)
    lda4hex.initialize(nTopics, seed, nThreads, verbose,
        alpha, eta, maxIter, mDelta, kappa, tau0, lda4hex.nUnits() * nEpochs,
        priorFile, priorScale);

    if (!weightFile.empty()) {
        lda4hex.setWeights(weightFile, defaultWeight);
    }

    if (!projection_only) {
        for (int epoch = 0; epoch < nEpochs; ++epoch) {
            int32_t n = lda4hex.trainOnline(inFile, batchSize, minCountTrain);
            notice("Epoch %d/%d, processed %d documents", epoch + 1, nEpochs, n);
            std::vector<double> weights;
            lda4hex.getTopicAbundance(weights);
            std::sort(weights.begin(), weights.end(), std::greater<double>());
            std::stringstream ss;
            ss << std::fixed << std::setprecision(4);
            for (const auto& w : weights) {
                ss << w << "\t";
            }
            notice("  Topic relative abundance: %s", ss.str().c_str());
        }

        // write model matrix to file
        lda4hex.writeModelToFile(outModel);
        notice("Model written to %s", outModel.c_str());
    }

    if (!transform) {
        return 0;
    }
    // transform the input data to the LDA space
    std::string outFile = outPrefix + ".results.tsv";
    lda4hex.fitAndWriteToFile(inFile, outFile, batchSize);
    notice("Transformed data written to %s", outFile.c_str());

    return 0;
};
