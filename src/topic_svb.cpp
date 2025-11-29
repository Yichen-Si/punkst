#include "topic_svb.hpp"

int32_t TopicModelWrapper::trainOnline(const std::string& inFile, int32_t _bsize, int32_t _minCountTrain, int32_t maxUnits) {
    if (!initialized) error("Model must be initialized before training");
    batchSize = _bsize;
    minCountTrain = _minCountTrain;
    ntot = 0;
    std::ifstream inFileStream(inFile);
    if (!inFileStream) error("Error opening input file: %s", inFile.c_str());
    int32_t b = 0;
    bool fileopen = true;
    while (fileopen) {
        fileopen = readMinibatch(inFileStream);
        if (minibatch.empty()) break;

        do_partial_fit(minibatch); // Virtual dispatch to derived class

        ntot += minibatch.size();
        if (ntot >= maxUnits) {
            break;
        }
        b++;
        if (verbose_ > 0 && (b % verbose_ == 0)) {
            printTopicAbundance();
        }
    }
    inFileStream.close();
    return ntot;
}

void TopicModelWrapper::writeModelHeader(std::ofstream& outFileStream) {
    const auto& t_names = get_topic_names();
    outFileStream << t_names[0];
    for (size_t i = 1; i < t_names.size(); ++i) {
        outFileStream << "\t" << t_names[i];
    }
    outFileStream << "\n";
}

void TopicModelWrapper::writeModelToFile(const std::string& outFile) {
    if (!initialized) error("Model must be initialized before writing");
    std::ofstream outFileStream(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());

    RowMajorMatrixXd model = copy_model_matrix(); // Virtual dispatch
    outFileStream << "Feature\t";
    writeModelHeader(outFileStream);
    outFileStream << std::fixed << std::setprecision(3);
    for (int i = 0; i < M_; ++i) {
        outFileStream << featureNames[i];
        for (int j = 0; j < model.rows(); ++j) {
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
    outFileStream << "#" << header << "\t";
    writeUnitHeader(outFileStream);

    bool fileopen = true;
    while (fileopen) {
        std::vector<std::string> idens;
        fileopen = readMinibatch(inFileStream, idens, reader.getNlayer() > 1);
        if (minibatch.empty()) break;

        Eigen::MatrixXd doc_topic = do_transform(minibatch); // Virtual dispatch
        for (int i = 0; i < minibatch.size(); ++i) {
            outFileStream << idens[i] << std::fixed << std::setprecision(4);
            if (idens[i].size() > 0) outFileStream << "\t";
            outFileStream << doc_topic(i, 0);
            for (int j = 1; j < doc_topic.cols(); ++j) {
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
        minibatch.push_back(std::move(doc));
        nlocal++;
    }
    return true;
}

void TopicModelWrapper::setupPriorMapping(std::vector<std::string>& feature_names_, std::vector<std::uint32_t>& kept_indices) {
    // Use the current filtered feature set
    std::unordered_map<std::string, uint32_t> dict;
    if (!reader.featureDict(dict)) {
        error("%s: Cannot setup prior mapping when feature dictionary is not available", __FUNCTION__);
    }
    featureNames.clear();
    kept_indices.clear();
    uint32_t idx = 0;
    for (const std::string& v : feature_names_) {
        if (dict.find(v) != dict.end()) {
            featureNames.push_back(v);
            kept_indices.push_back(idx);
        }
        idx++;
    }
    if (featureNames.empty()) {
        error("%s: No features overlap between filtered input features and prior model", __FUNCTION__);
    }
    M_ = featureNames.size();
    notice("Found %d features in intersection of data (%d) and queries (%d)",
            M_, (int)dict.size(), (int)feature_names_.size());
    // Update to use the mapped feature space (preserves prior model ordering)
    reader.setFeatureIndexRemap(featureNames, false);
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

void TopicModelWrapper::printTopicAbundance() {
    std::vector<double> topic_weights;
    getTopicAbundance(topic_weights);
    std::sort(topic_weights.begin(), topic_weights.end(), std::greater<double>());
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);
    for (size_t i = 0; i < std::min<size_t>(10, topic_weights.size()); ++i) {
        ss << topic_weights[i] << "\t";
    }
    notice("Top topic relative abundance: %s", ss.str().c_str());
}
