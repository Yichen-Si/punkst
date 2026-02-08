#include "topic_svb.hpp"
#include <algorithm>
#include <random>

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

void TopicModelWrapper::load10X(DGEReader10X& dge, int32_t _minCountTrain, bool force) {
    if (dge_cache_ready_ && !force && _minCountTrain == dge_minCountTrain_cache_) {
        return;
    }
    dge_docs_cache_.clear();
    dge_barcode_idx_cache_.clear();
    dge_train_idx_cache_.clear();
    dge_minCountTrain_cache_ = _minCountTrain;
    int32_t nUnits = dge.readAll(dge_docs_cache_, dge_barcode_idx_cache_, 0);
    if (dge_docs_cache_.empty()) {
        dge_cache_ready_ = true;
        return;
    }
    std::vector<double> feature_sums_raw(M_, 0.0);
    dge_train_idx_cache_.reserve(dge_docs_cache_.size());
    for (size_t i = 0; i < dge_docs_cache_.size(); ++i) {
        Document& doc = dge_docs_cache_[i];
        double total = 0.0;
        for (size_t j = 0; j < doc.ids.size(); ++j) {
            uint32_t m = doc.ids[j];
            if (m < feature_sums_raw.size()) {
                feature_sums_raw[m] += doc.cnts[j];
            }
        }
        applyWeights(doc);
        total = doc.get_sum();
        if (total >= _minCountTrain) {
            dge_train_idx_cache_.push_back(static_cast<int32_t>(i));
        }
    }
    int32_t nTrain = static_cast<int32_t>(dge_train_idx_cache_.size());
    reader.setFeatureSums(feature_sums_raw, true);
    dge_cache_ready_ = true;
    notice("%s: Loaded %d units, %d with total count >= %d for training", __func__, nUnits, nTrain, _minCountTrain);
}

int32_t TopicModelWrapper::trainOnline10X(int32_t _bsize, int32_t maxUnits, int32_t seed) {
    if (!initialized) error("Model must be initialized before training");
    if (!dge_cache_ready_) error("10X cache is not initialized; call load10X() first");
    batchSize = _bsize;
    ntot = 0;
    if (dge_train_idx_cache_.empty()) {
        return 0;
    }

    std::vector<int32_t> order = dge_train_idx_cache_;
    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::shuffle(order.begin(), order.end(), rng);

    size_t cursor = 0;
    while (cursor < order.size()) {
        minibatch.clear();
        const size_t remaining = order.size() - cursor;
        size_t take = std::min(static_cast<size_t>(batchSize), remaining);
        if (maxUnits < INT32_MAX) {
            const size_t max_remaining = static_cast<size_t>(maxUnits - ntot);
            if (take > max_remaining) {
                take = max_remaining;
            }
        }
        if (take == 0) {
            break;
        }
        minibatch.reserve(take);
        for (size_t i = 0; i < take; ++i) {
            minibatch.push_back(dge_docs_cache_[order[cursor + i]]);
        }
        cursor += take;

        do_partial_fit(minibatch);

        ntot += static_cast<int32_t>(minibatch.size());
        if (ntot >= maxUnits) {
            break;
        }
    }
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

void TopicModelWrapper::fitAndWriteToFile(const std::string& inFile, const std::string& outPrefix, int32_t _bsize) {
     if (!initialized) error("Model must be initialized before fitting");
    batchSize = _bsize;
    std::ifstream inFileStream(inFile);
    if (!inFileStream) error("Error opening input file: %s", inFile.c_str());
    std::string outFile = outPrefix + ".results.tsv";
    std::ofstream outFileStream(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());

    std::string header;
    reader.getInfoHeaderStr(header);
    outFileStream << "#" << header << "\t";
    writeUnitHeader(outFileStream);

    bool fileopen = true;
    Eigen::MatrixXd pseudobulk;
    while (fileopen) {
        std::vector<std::string> idens;
        fileopen = readMinibatch(inFileStream, idens, reader.getNlayer() > 1);
        if (minibatch.empty()) break;

        Eigen::MatrixXd doc_topic = do_transform(minibatch); // Virtual dispatch
        if (pseudobulk.rows() == 0) {
            pseudobulk = Eigen::MatrixXd::Zero(M_, doc_topic.cols());
        }
        for (int i = 0; i < minibatch.size(); ++i) {
            outFileStream << idens[i] << std::fixed << std::setprecision(4);
            if (idens[i].size() > 0) outFileStream << "\t";
            outFileStream << doc_topic(i, 0);
            for (int k = 1; k < doc_topic.cols(); ++k) {
                outFileStream << "\t" << doc_topic(i, k);
            }
            outFileStream << "\n";
            // Update pseudobulk
            Document& doc = minibatch[i];
            for (int j = 0; j < doc.ids.size(); ++j) {
                uint32_t m = doc.ids[j];
                for (int k = 0; k < doc_topic.cols(); ++k) {
                    pseudobulk(m, k) += doc.cnts[j] * doc_topic(i, k);
                }
            }
        }
    }
    inFileStream.close();
    outFileStream.close();
    notice("Transformation results written to %s", outFile.c_str());

    outFile = outPrefix + ".pseudobulk.tsv";
    outFileStream.open(outFile);
    size_t K = pseudobulk.cols();
    outFileStream << "Feature\t";
    writeModelHeader(outFileStream);
    outFileStream << std::fixed << std::setprecision(3);
    for (int i = 0; i < M_; ++i) {
        outFileStream << featureNames[i];
        for (size_t k = 0; k < K; ++k) {
            outFileStream << "\t" << pseudobulk(i, k);
        }
        outFileStream << "\n";
    }
    outFileStream.close();
    notice("Pseudobulk counts written to %s", outFile.c_str());
}

void TopicModelWrapper::fitAndWriteToFile10X(DGEReader10X& dge, const std::string& outPrefix, int32_t _bsize) {
    if (!initialized) error("Model must be initialized before fitting");
    batchSize = _bsize;
    std::string outFile = outPrefix + ".results.tsv";
    std::ofstream outFileStream(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());

    outFileStream << "#barcode\t";
    writeUnitHeader(outFileStream);

    Eigen::MatrixXd pseudobulk;
    std::vector<Document> minibatch_local;
    std::vector<std::string> idens;
    if (dge_cache_ready_ && !dge_docs_cache_.empty()) {
        size_t cursor = 0;
        while (cursor < dge_docs_cache_.size()) {
            minibatch_local.clear();
            idens.clear();
            const size_t remaining = dge_docs_cache_.size() - cursor;
            const size_t take = std::min(static_cast<size_t>(batchSize), remaining);
            minibatch_local.reserve(take);
            idens.reserve(take);
            for (size_t i = 0; i < take; ++i) {
                const size_t idx = cursor + i;
                minibatch_local.push_back(dge_docs_cache_[idx]);
                if (idx < dge_barcode_idx_cache_.size()) {
                    idens.push_back(std::to_string(dge_barcode_idx_cache_[idx]));
                } else {
                    idens.push_back(std::to_string(idx));
                }
            }
            cursor += take;

            Eigen::MatrixXd doc_topic = do_transform(minibatch_local); // Virtual dispatch
            if (pseudobulk.rows() == 0) {
                pseudobulk = Eigen::MatrixXd::Zero(M_, doc_topic.cols());
            }
            for (int i = 0; i < minibatch_local.size(); ++i) {
                outFileStream << idens[i] << std::fixed << std::setprecision(4);
                if (idens[i].size() > 0) outFileStream << "\t";
                outFileStream << doc_topic(i, 0);
                for (int k = 1; k < doc_topic.cols(); ++k) {
                    outFileStream << "\t" << doc_topic(i, k);
                }
                outFileStream << "\n";
                Document& doc = minibatch_local[i];
                for (int j = 0; j < doc.ids.size(); ++j) {
                    uint32_t m = doc.ids[j];
                    for (int k = 0; k < doc_topic.cols(); ++k) {
                        pseudobulk(m, k) += doc.cnts[j] * doc_topic(i, k);
                    }
                }
            }
        }
    } else {
        bool fileopen = true;
        while (fileopen) {
            minibatch_local.clear();
            idens.clear();
            minibatch_local.reserve(batchSize);
            idens.reserve(batchSize);
            int32_t barcode_idx = -1;
            while ((int32_t)minibatch_local.size() < batchSize) {
                Document doc;
                if (!dge.next(doc, &barcode_idx, nullptr)) {
                    fileopen = false;
                    break;
                }
                std::string ident;
                if (barcode_idx >= 0) {
                    ident = std::to_string(barcode_idx);
                }
                applyWeights(doc);
                minibatch_local.push_back(std::move(doc));
                idens.push_back(std::move(ident));
            }
            if (minibatch_local.empty()) {
                break;
            }

            Eigen::MatrixXd doc_topic = do_transform(minibatch_local); // Virtual dispatch
            if (pseudobulk.rows() == 0) {
                pseudobulk = Eigen::MatrixXd::Zero(M_, doc_topic.cols());
            }
            for (int i = 0; i < minibatch_local.size(); ++i) {
                outFileStream << idens[i] << std::fixed << std::setprecision(4);
                if (idens[i].size() > 0) outFileStream << "\t";
                outFileStream << doc_topic(i, 0);
                for (int k = 1; k < doc_topic.cols(); ++k) {
                    outFileStream << "\t" << doc_topic(i, k);
                }
                outFileStream << "\n";
                Document& doc = minibatch_local[i];
                for (int j = 0; j < doc.ids.size(); ++j) {
                    uint32_t m = doc.ids[j];
                    for (int k = 0; k < doc_topic.cols(); ++k) {
                        pseudobulk(m, k) += doc.cnts[j] * doc_topic(i, k);
                    }
                }
            }
        }
    }
    outFileStream.close();
    notice("Transformation results written to %s", outFile.c_str());

    outFile = outPrefix + ".pseudobulk.tsv";
    outFileStream.open(outFile);
    size_t K = pseudobulk.cols();
    outFileStream << "Feature\t";
    writeModelHeader(outFileStream);
    outFileStream << std::fixed << std::setprecision(3);
    for (int i = 0; i < M_; ++i) {
        outFileStream << featureNames[i];
        for (size_t k = 0; k < K; ++k) {
            outFileStream << "\t" << pseudobulk(i, k);
        }
        outFileStream << "\n";
    }
    outFileStream.close();
    notice("Pseudobulk counts written to %s", outFile.c_str());
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

bool TopicModelWrapper::readMinibatch(std::ifstream& inFileStream, std::vector<Document>& batch,
        std::vector<std::string>& idens, int32_t batchSizeOverride,
        int32_t minCount, int32_t maxUnits) {
    batch.clear();
    idens.clear();
    const int32_t batchTarget = batchSizeOverride > 0 ? batchSizeOverride : batchSize;
    batch.reserve(batchTarget);
    idens.reserve(batchTarget);
    std::string line;
    int32_t nlocal = 0;
    while (nlocal < batchTarget && nlocal < maxUnits) {
        if (!std::getline(inFileStream, line)) {
            return false;
        }
        Document doc;
        std::string info;
        int32_t ct = reader.parseLine(doc, info, line, modal);
        if (ct < 0) {
            error("%s: Error parsing the %d-th line", __FUNCTION__, ntot + nlocal);
        }
        if (minCount > 0 && doc.get_sum() < minCount) {
            continue;
        }
        idens.push_back(info);
        batch.push_back(std::move(doc));
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
    featureNames.clear(); // kept features, ordered as in feature_names_
    kept_indices.clear(); // index in feature_names_
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
    // Update to use the mapped feature space (preserves query ordering)
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
