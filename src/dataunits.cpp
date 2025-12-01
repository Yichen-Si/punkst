#include "dataunits.hpp"
#include <regex>

int32_t HexReader::parseLine(Document& doc, std::string &info, const std::string &line, int32_t modal, bool add2sums) {
    assert(modal < nModal && "Modal out of range");
    std::vector<int32_t> nfeatures(nModal, 0);
    std::vector<uint32_t> counts(nModal, 0);
    std::vector<std::string> token;
    split(token, "\t", line);
    size_t ntokens = token.size();
    if (ntokens < mintokens) {
        return -1;
    }
    for (uint32_t j = 0; j < offset_data; j++) {
        info += token[j];
        if (j < offset_data - 1) {
            info += "\t";
        }
    }
    uint32_t i = offset_data;
    for (int j = 0; j < nModal; j++) {
        if(!str2int32(token[i], nfeatures[j])) {
            return -1;
        }
        if(!str2uint32(token[i + 1], counts[j])) {
            return -1;
        }
        i += 2;
    }
    for (int l = 0; l < modal; ++l) { // skip
        i += nfeatures[l] * 2;
    }
    doc.ids.clear();
    doc.cnts.clear();
    if (remap) {
        doc.ids.reserve(nfeatures[modal]);
        doc.cnts.reserve(nfeatures[modal]);
        for (int j = 0; j < nfeatures[modal]; ++j) {
            std::vector<std::string> pair;
            split(pair, " ", token[i]);
            uint32_t u;
            double v;
            if (!str2uint32(pair[0], u)) {return -1;}
            if (!str2double(pair[1], v)) {return -1;}
            auto it = idx_remap.find(u);
            if (it != idx_remap.end()) {
                doc.ids.push_back(it->second);
                doc.cnts.push_back(v);
            }
            i += 1;
        }
    } else {
        doc.ids.resize(nfeatures[modal]);
        doc.cnts.resize(nfeatures[modal]);
        for (int j = 0; j < nfeatures[modal]; ++j) {
            std::vector<std::string> pair;
            split(pair, " ", token[i]);
            if (!str2uint32(pair[0], doc.ids[j])) {return -1;}
            if (!str2double(pair[1], doc.cnts[j])) {return -1;}
            i += 1;
        }
    }
    if (weightFeatures) {
        for (size_t i = 0; i < doc.ids.size(); i++) {
            doc.cnts[i] *= weights[doc.ids[i]];
        }
    }
    if (accumulate_sums && add2sums) {
        for (size_t i = 0; i < doc.ids.size(); ++i) {
            uint32_t j = doc.ids[i];
            if (j < nFeatures) {
                feature_sums[j] += doc.cnts[i];
            }
        }
    }
    return counts[modal];
}

void HexReader::readMetadata(const std::string &metaFile) {
    std::ifstream metaIn(metaFile);
    if (!metaIn) {
        throw std::runtime_error("Error opening metadata file " + metaFile);
    }

    // Parse the JSON file (only offset_data is required)
    nlohmann::json meta;
    metaIn >> meta;
    hexSize = meta.value("hex_size", 0);
    nUnits = meta.value("n_units", 0);
    nLayer = meta.value("n_layers", 1);
    nModal = meta.value("n_modalities", 1);
    nFeatures = meta.value("n_features", 0);
    offset_data = meta.value("offset_data", -1);
    if (offset_data < 0) {
        throw std::runtime_error("Error: offset_data not found in metadata file");
    }
    icol_layer = meta.value("icol_layer", -1);
    icol_x = meta.value("icol_x", -1);
    icol_y = meta.value("icol_y", -1);
    header_info = meta.value("header_info", std::vector<std::string>());
    features.resize(nFeatures);
    if (meta.contains("dictionary")) {
        for (auto& item : meta["dictionary"].items()) {
            if (!item.value().is_number_unsigned()) {
                throw std::runtime_error("Dictionary (key: value) pairs must have non-negative integer values");
            }
            uint32_t idx = item.value();
            if (idx >= nFeatures) {
                features.resize(idx + 1);
            }
            features[idx] = item.key();
        }
        nFeatures = features.size();
    } else if (nFeatures > 0) {
        for (int i = 0; i < nFeatures; ++i) {
            features[i] = std::to_string(i);
        }
    }
    mintokens = offset_data + 2 * nModal;
    hasCoordinates = (icol_x >= 0 && icol_y >= 0);
    feature_sums.resize(nFeatures, 0.0);
}

void HexReader::setFeatureFilter(const std::string& featureFile, int32_t minCount, std::string& include_ftr_regex, std::string& exclude_ftr_regex, bool read_sums) {
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
    std::unordered_map<uint32_t, uint32_t> remap_new;
    std::unordered_map<std::string, uint32_t> dict;
    std::stringstream ss;
    std::vector<std::string> token;
    bool has_dict = featureDict(dict);
    std::unordered_set<std::string> kept_features; // avoid duplicates
    bool has_sums = false;
    while (std::getline(inFeature, line)) {
        if (line.empty() || line[0] == '#') continue;
        split(token, "\t ", line);
        if (token.size() < 1) {
            error("Error reading feature file at line: %s", line.c_str());
        }
        uint32_t idx_prev = idx0;
        idx0++;
        std::string& feature = token[0];
        if (has_dict) {
            auto it = dict.find(feature);
            if (it == dict.end()) {
                continue;
            }
            idx_prev = it->second;
        }
        double count = -1;
        if (read_sums) {
            if (idx1 == 0) {
                if (token.size() > 1) {
                    if (!str2num(token[1], count) || count < 0) {
                        warning("%s: Non-numerical value on the second column is ignored: %s", __func__, line.c_str());
                    } else {
                        has_sums = true;
                        feature_sums.assign(nFeatures, 0.0);
                    }
                }
            } else if (has_sums) {
                if (token.size() <= 1 || !str2num(token[1], count) || count < 0) {
                    warning("%s: Expect non-negative numerical value on all or none of the second column: %s", __func__, line.c_str());
                    has_sums = false;
                }
            }
            if (count >= 0 && count < minCount) {
                continue;
            }
        }
        bool include = !check_include || std::regex_match(feature, regex_include);
        bool exclude = check_exclude && std::regex_match(feature, regex_exclude);
        if (include && !exclude &&
            kept_features.find(feature) == kept_features.end()) {
            if (has_sums) feature_sums[idx_prev] = count;
            remap_new[idx_prev] = idx1++;
            kept_features.insert(feature);
        } else {
            ss << " " << feature;
        }
    }
    if (has_sums) {
        accumulate_sums = false;
        readFullSums = true;
    }
    notice("%s: %d features are kept out of %d", __FUNCTION__, idx1, idx0);
    if (ss.str().length() > 0)
        std::cout << "Excluded due to regex:" << ss.str() << std::endl;
    setFeatureIndexRemap(remap_new);
}

void HexReader::setWeights(const std::string& weightFile, double defaultWeight_) {
    std::ifstream inWeight(weightFile);
    if (!inWeight) {
        error("Error opening weights file: %s", weightFile.c_str());
    }
    defaultWeight = defaultWeight_;
    int32_t nweighted = 0, novlp = 0;
    weights.resize(nFeatures);
    std::fill(weights.begin(), weights.end(), defaultWeight);
    std::unordered_map<std::string, uint32_t> dict;
    if (!featureDict(dict)) {
        warning("Feature dictionary is not found in the input metadata, we assume the weight file (--feature-weights) refers to features by their index as in the input file");
        std::string line;
        while (std::getline(inWeight, line)) {
            nweighted++;
            std::istringstream iss(line);
            uint32_t idx;
            double weight;
            if (!(iss >> idx >> weight)) {
                error("Error reading weights file at line: %s", line.c_str());
            }
            if (idx >= nFeatures) {
                warning("Input file contains %zu features, feature index %u in the weights file is out of range", nFeatures, idx);
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
            auto it = dict.find(feature);
            if (it != dict.end()) {
                weights[it->second] = weight;
                novlp++;
            } else {
                warning("Feature %s not found in the input", feature.c_str());
            }
        }
    }
    notice("Read %d weights from file, %d features overlap with the input file", nweighted, novlp);
    if (novlp == 0) {
        error("No features in the weight file overlap with those found in the input file, check if the files are consistent.");
    }
    weightFeatures = true;
}

int32_t HexReader::readAll(std::vector<Document>& docs, std::vector<std::string>& info, const std::string &inFile, int32_t minCount, int32_t modal, bool add2sums) {
    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("%s: Error opening input file: %s", __func__, inFile.c_str());
    }
    std::string line;
    int32_t n = 0;
    while (std::getline(inFileStream, line)) {
        Document doc;
        std::string l;
        int32_t ct = parseLine(doc, l, line, modal);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        if (ct < minCount) {
            continue;
        }
        if (accumulate_sums && add2sums) {
            for (size_t i = 0; i < doc.ids.size(); ++i) {
                uint32_t j = doc.ids[i];
                if (j < nFeatures) {
                    feature_sums[j] += doc.cnts[i];
                }
            }
        }
        docs.push_back(std::move(doc));
        info.push_back(std::move(l));
        n++;
    }
    readFullSums = true;
    return n;
}

int32_t HexReader::readAll(std::vector<Document>& docs, const std::string &inFile, int32_t minCount, int32_t modal, bool add2sums) {
    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("%s: Error opening input file: %s", __func__, inFile.c_str());
    }
    std::string line, l;
    int32_t n = 0;
    while (std::getline(inFileStream, line)) {
        Document doc;
        int32_t ct = parseLine(doc, l, line, modal);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        if (ct < minCount) {
            continue;
        }
        if (accumulate_sums && add2sums) {
            for (size_t i = 0; i < doc.ids.size(); ++i) {
                uint32_t j = doc.ids[i];
                if (j < nFeatures) {
                    feature_sums[j] += doc.cnts[i];
                }
            }
        }
        docs.push_back(std::move(doc));
        n++;
    }
    readFullSums = true;
    return n;
}

void HexReader::setFeatureIndexRemap(std::unordered_map<uint32_t, uint32_t>& new_idx_remap) {

    nFeatures = new_idx_remap.size();
    std::vector<std::string> new_features(nFeatures);
    for (const auto& pair : new_idx_remap) {
        if (pair.second >= nFeatures) {
            nFeatures = pair.second + 1;
            new_features.resize(nFeatures);
        }
        new_features[pair.second] = features[pair.first];
    }
    features = std::move(new_features);
    if (weightFeatures) {
        std::vector<double> new_weights(nFeatures, defaultWeight);
        for (const auto& pair : new_idx_remap) {
            new_weights[pair.second] = weights[pair.first];
        }
        weights = std::move(new_weights);
    }
    if (readFullSums) {
        std::vector<double> new_sums(nFeatures, 0.0);
        for (const auto& pair : new_idx_remap) {
            new_sums[pair.second] = feature_sums[pair.first];
        }
        feature_sums = std::move(new_sums);
    } else {
        feature_sums.clear();
        feature_sums.resize(nFeatures, 0.0);
    }

    if (remap) {
        std::unordered_map<uint32_t, uint32_t> final_idx_remap;
        for (const auto& pair : idx_remap) {
            auto it = new_idx_remap.find(pair.second);
            if (it != new_idx_remap.end()) {
                final_idx_remap[pair.first] = it->second;
            }
        }
        idx_remap = std::move(final_idx_remap);
    } else {
        idx_remap = std::move(new_idx_remap);
    }
    remap = true;
}

void HexReader::setFeatureIndexRemap(std::vector<std::string>& new_features, bool keep_unmapped) {
    std::unordered_map<std::string, uint32_t> dict;
    if (!featureDict(dict)) {
        error("%s: feature names are not available", __func__);
    }
    int32_t n_new = new_features.size();
    bool changed = false;
    if (nFeatures == n_new) { // check if they are identical
        for (uint32_t i = 0; i < nFeatures; ++i) {
            if (features[i] != new_features[i]) {
                changed = true;
                break;
            }
        }
    } else {
        changed = true;
    }
    if (!changed) {
        return;
    }

    int32_t n = 0, n_unmap = 0;
    std::unordered_map<uint32_t, uint32_t> new_idx_remap;
    for (size_t i = 0; i < new_features.size(); ++i) {
        auto it = dict.find(new_features[i]);
        if (it != dict.end()) {
            new_idx_remap[it->second] = static_cast<uint32_t>(i);
            n++;
        }
    }
    if (keep_unmapped) {
        for (int32_t i = 0; i < nFeatures; ++i) {
            if (new_idx_remap.find(i) == new_idx_remap.end()) {
                new_idx_remap[i] = static_cast<uint32_t>(n_new + n_unmap);
                n_unmap++;
            }
        }
    }
    notice("%s: %d features are kept out of %d, %d mapped to input set of size %d", __func__, n+n_unmap, nFeatures, n, n_new);
    setFeatureIndexRemap(new_idx_remap);
}

bool UnitValues::readFromLine(const std::string& line, int32_t nModal, bool labeled) {
    std::istringstream iss(line);
    std::string hexKey;
    try {
        iss >> hexKey >> x >> y;
        if (labeled) {
            iss >> label;
            if (label < 0) {
                return false;
            }
        }
    } catch (const std::exception& e) {
        error("Error reading line: %s\n %s", line.c_str(), e.what());
    } catch (...) {
        error("Unknown error reading line: %s", line.c_str());
    }
    clear();
    vals.resize(nModal);
    valsums.resize(nModal);
    std::vector<int32_t> nfeatures(nModal, 0);
    std::vector<uint32_t> counts(nModal, 0);
    for (int i = 0; i < nModal; ++i) {
        if (!(iss >> nfeatures[i] >> counts[i])) {
            return false;
        }
    }
    for (size_t i = 0; i < nModal; ++i) {
        for (int j = 0; j < nfeatures[i]; ++j) {
            uint32_t feature;
            int32_t value;
            if (!(iss >> feature >> value)) {
                return false;
            }
            vals[i][feature] = value;
            valsums[i] += value;
        }
    }
    return true;
}

bool UnitValues::writeToFile(std::ostream& os, uint32_t key) const {
    os << uint32toHex(key) << "\t" << x << "\t" << y;
    if (label >= 0)
        os << "\t" << label;
    for (size_t i = 0; i < vals.size(); ++i) {
        os << "\t" << vals[i].size() << "\t" << valsums[i];
    }
    for (const auto& val : vals) {
        for (const auto& entry : val) {
            os << "\t" << entry.first << " " << entry.second;
        }
    }
    os << "\n";
    return os.good();
}

int32_t read_sparse_obs(const std::string &inFile, HexReader &reader,
    std::vector<SparseObs> &docs, std::vector<std::string> &rnames,
    int32_t minCountTrain, double size_factor, double c,
    std::string* covarFile, std::vector<uint32_t>* covar_idx,
    std::vector<std::string>* covar_names, bool allow_na,
    int32_t label_idx, std::vector<std::string>* labels,
    int32_t debug_N) {

    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("Fail to open input file: %s", inFile.c_str());
    }
    std::string line, tmp;
    std::vector<std::string> tokens, covar_header;
    std::ifstream covarFileStream;
    int32_t n_covar = 0, n_tokens = 0;
    bool per_doc_c = c <= 0;
    bool has_labels = covarFile && label_idx >= 0 && labels != nullptr;

    if (covarFile) {
        if (label_idx >= 0 && labels == nullptr) {
            error("%s: if label index is specified, a pointer to a vector to store the labels must be provided", __func__);
        }
        covar_names->clear();
        if (labels) {labels->clear();}
        if (!covarFile->empty()) {
            covarFileStream.open(*covarFile);
            if (!covarFileStream) {
                error("Fail to covariate file: %s", covarFile->c_str());
            }
            // Read the header line
            if (!std::getline(covarFileStream, line)) {
                error("Fail to parse covariate file: %s", covarFile->c_str());
            }
            split(covar_header, "\t ", line, UINT_MAX, true, true, true);
            n_tokens = static_cast<int32_t>(covar_header.size());
            if (covar_idx->empty() && !has_labels) {
                // Assuming use all columns except the first one
                n_covar = covar_header.size() - 1;
                for (int32_t i = 1; i < n_tokens; i++) {
                    covar_idx->push_back(i);
                    covar_names->push_back(covar_header[i]);
                }
            } else {
                n_covar = (int32_t) covar_idx->size();
                for (const auto i : *covar_idx) {
                    if (i < 0 || i >= n_tokens) {
                        error("Covariate index %d is out of range [0,%lu)", i, covar_header.size());
                    }
                    covar_names->push_back(covar_header[i]);
                }
            }
            notice("Covariate file has %d columns, using %d as covariates", n_tokens, n_covar);
        }
    }

    if (reader.nUnits > 0) {
        docs.reserve(reader.nUnits);
    }
    bool has_covar = n_covar > 0 || has_labels;
    int32_t idx = 0;
    while (std::getline(inFileStream, line)) {
        idx++;
        if (idx % 10000 == 0) {
            notice("Read %d units...", idx);
        }
        SparseObs obs;
        int32_t ct = reader.parseLine(obs.doc, tmp, line);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        if (ct < minCountTrain) {
            std::getline(covarFileStream, line);
            continue;
        }
        obs.c = per_doc_c ? ct / size_factor : c;
        obs.ct_tot = ct;
        if (has_covar) { // read covariates
            if (!std::getline(covarFileStream, line)) {
                error("The number of lines in covariate file is less than that in data file");
            }
            split(tokens, "\t ", line, UINT_MAX, true, true, true);
            if (tokens.size() != n_tokens) {
                error("Number of columns (%lu) of line [%s] in covariate file does not match header (%d)", tokens.size(), line.c_str(), n_tokens);
            }
            if (n_covar > 0) {
                obs.covar = Eigen::VectorXd::Zero(n_covar);
                for (int32_t i = 0; i < n_covar; i++) {
                    if (!str2double(tokens[(*covar_idx)[i]], obs.covar(i))) {
                        if (!allow_na)
                            error("Invalid value for %d-th covariate %s.", i+1, tokens[(*covar_idx)[i]].c_str());
                        obs.covar(i) = 0;
                    }
                }
            }
            if (has_labels) {
                labels->push_back(tokens[label_idx]);
            }
        }
        docs.push_back(std::move(obs));
        rnames.emplace_back(std::to_string(idx-1));
        if (debug_N > 0 && idx > debug_N) {
            break;
        }
    }
    inFileStream.close();
    return (int32_t) docs.size();
}

int32_t HexReader::readAll(Eigen::SparseMatrix<double, Eigen::RowMajor> &X,
    std::vector<std::string>& info, const std::string &inFile,
    int32_t minCount, int32_t modal, bool add2sums) {
    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("%s: Error opening input file: %s", __func__, inFile.c_str());
    }
    info.clear();
    Document doc;
    std::string line;
    int32_t n = 0;
    std::vector<Eigen::Triplet<double>> triplets;
    if (nUnits > 0) {
        triplets.reserve(static_cast<size_t>(nUnits) * 32);
    }
    while (std::getline(inFileStream, line)) {
        std::string l;
        int32_t ct = parseLine(doc, l, line, modal);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        if (ct < minCount) {
            continue;
        }
        if (accumulate_sums && add2sums) {
            for (size_t i = 0; i < doc.ids.size(); ++i) {
                uint32_t j = doc.ids[i];
                if (j < static_cast<uint32_t>(nFeatures)) {
                    feature_sums[j] += doc.cnts[i];
                }
            }
        }
        info.push_back(std::move(l));
        // Add entries for this document as row n
        for (size_t k = 0; k < doc.ids.size(); ++k) {
            uint32_t j = doc.ids[k];
            if (j >= static_cast<uint32_t>(nFeatures)) {
                continue;
            }
            double val = doc.cnts[k];
            if (val != 0.0) {
                triplets.emplace_back(n, static_cast<int>(j), val);
            }
        }
        ++n;
    }
    // Build the row-major sparse matrix (n samples x nFeatures)
    X.resize(n, nFeatures);
    if (!triplets.empty()) {
        X.setFromTriplets(triplets.begin(), triplets.end());
    } else {
        X.setZero();  // keep the size but no non-zeros
    }
    X.makeCompressed();
    readFullSums = true;
    return n;
}
