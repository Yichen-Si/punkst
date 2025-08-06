#include "topic_svb.hpp"

void TopicModelWrapper::setFeatures(const std::string& featureFile, int32_t minCount, std::string& include_ftr_regex, std::string& exclude_ftr_regex) {
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
    while (std::getline(inFeature, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        std::string feature;
        int32_t count;
        if (!(iss >> feature >> count)) {
            error("Error reading feature file at line: %s", line.c_str());
        }
        uint32_t idx_prev = idx0;
        idx0++;
        if (count < minCount) {
            continue;
        }
        if (reader.featureDict(featureDict)) {
            auto it = featureDict.find(feature);
            if (it == featureDict.end()) {
                continue;
            }
            idx_prev = it->second;
        }
        bool include = !check_include || std::regex_match(feature, regex_include);
        bool exclude = check_exclude && std::regex_match(feature, regex_exclude);
        if (include && !exclude) {
            idx_remap[idx_prev] = idx1++;
            featureNames.push_back(feature);
        } else {
            std::cout << "Exclude " << feature << std::endl;
        }
    }
    M_ = idx_remap.size();
    notice("%s: %d features are kept out of %d", __FUNCTION__, idx1, idx0);
    reader.setFeatureIndexRemap(idx_remap);
}

void TopicModelWrapper::setWeights(const std::string& weightFile, double defaultWeight_) {
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

int32_t TopicModelWrapper::trainOnline(const std::string& inFile, int32_t _bsize, int32_t _minCountTrain) {
    if (!initialized) error("Model must be initialized before training");
    batchSize = _bsize;
    minCountTrain = _minCountTrain;
    ntot = 0;
    std::ifstream inFileStream(inFile);
    if (!inFileStream) error("Error opening input file: %s", inFile.c_str());

    bool fileopen = true;
    while (fileopen) {
        fileopen = readMinibatch(inFileStream);
        if (minibatch.empty()) break;

        do_partial_fit(minibatch); // Virtual dispatch to derived class

        ntot += minibatch.size();
    }
    inFileStream.close();
    return ntot;
}

void TopicModelWrapper::writeModelToFile(const std::string& outFile) {
    if (!initialized) error("Model must be initialized before writing");
    std::ofstream outFileStream(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());

    MatrixXd model = copy_model_matrix(); // Virtual dispatch
    const auto& t_names = get_topic_names(); // Virtual dispatch
    int32_t nTopics = getNumTopics();
    outFileStream << "Feature";
    for (int i = 0; i < nTopics; ++i) {
        outFileStream << "\t" << t_names[i];
    }
    outFileStream << "\n";
    outFileStream << std::fixed << std::setprecision(3);
    for (int i = 0; i < M_; ++i) {
        outFileStream << featureNames[i];
        for (int j = 0; j < nTopics; ++j) {
            outFileStream << "\t" << model(j, i);
        }
        outFileStream << "\n";
    }
    outFileStream.close();
}

void TopicModelWrapper::fitAndWriteToFile(const std::string& inFile, const std::string& outFile, int32_t _bsize) {
     if (!initialized) error("Model must be initialized before fitting");
    batchSize = _bsize;
    std::ifstream inFileStream(inFile);
    if (!inFileStream) error("Error opening input file: %s", inFile.c_str());
    std::ofstream outFileStream(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());

    std::string header;
    reader.getInfoHeaderStr(header);
    outFileStream << header;
    const auto& t_names = get_topic_names(); // Virtual dispatch
    int32_t nTopics = getNumTopics();
    for (int i = 0; i < nTopics; ++i) {
        outFileStream << "\t" << t_names[i];
    }
    outFileStream << "\n";

    bool fileopen = true;
    while (fileopen) {
        std::vector<std::string> idens;
        fileopen = readMinibatch(inFileStream, idens, reader.getNlayer() > 1);
        if (minibatch.empty()) break;

        Eigen::MatrixXd doc_topic = do_transform(minibatch); // Virtual dispatch
        // Normalize and write results
        for (int i = 0; i < doc_topic.rows(); ++i) {
            double sum = doc_topic.row(i).sum();
            if (sum > 0) doc_topic.row(i) /= sum;
        }
        for (int i = 0; i < minibatch.size(); ++i) {
            outFileStream << idens[i] << std::fixed << std::setprecision(4);
            if (idens[i].size() > 0) outFileStream << "\t";
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

bool TopicModelWrapper::readMinibatch(std::ifstream& inFileStream) {
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

bool TopicModelWrapper::readMinibatch(std::ifstream& inFileStream, std::vector<std::string>& idens, bool labeled) {
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

void TopicModelWrapper::setupPriorMapping(std::vector<std::string>& feature_names_, std::vector<std::uint32_t>& kept_indices) {
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

void TopicModelWrapper::getTopicAbundance(std::vector<double>& topic_weights) {
    if (!initialized) error("%s: Model is not initialized", __FUNCTION__);
    const MatrixXd& model = get_model_matrix();
    topic_weights.resize(getNumTopics());
    for (int k = 0; k < getNumTopics(); k++) {
        topic_weights[k] = model.row(k).sum();
    }
    double total = std::accumulate(topic_weights.begin(), topic_weights.end(), 0.0);
    if (total > 0) {
        for (int k = 0; k < getNumTopics(); k++) {
            topic_weights[k] /= total;
        }
    }
}
