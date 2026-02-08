#include "dataunits.hpp"
#include <regex>
#include <sstream>
#include <algorithm>
#include <utility>

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
    feature_sums_raw.resize(nFeatures, 0.0);
}

void HexReader::initFromFeatures(const std::string& featureFile,
    int32_t n_units, int32_t n_modal, int32_t n_layer) {
    if (featureFile.empty()) {
        error("%s: feature file path is empty", __func__);
    }
    features.clear();
    feature_sums.clear();
    feature_sums_raw.clear();
    bool has_sums = false;
    bool sums_consistent = true;
    std::vector<double> sums;
    std::vector<std::string> token;
    auto lines = read_lines_maybe_gz(featureFile);
    features.reserve(lines.size());
    sums.reserve(lines.size());
    for (auto &line : lines) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        split(token, "\t ", line);
        if (token.size() < 1) {
            error("Error reading feature file at line: %s", line.c_str());
        }
        features.push_back(token[0]);
        if (!sums_consistent) {
            continue;
        }
        if (token.size() <= 1) {
            if (has_sums) {
                warning("%s: Expect non-negative numerical value on all or none of the second column: %s", __func__, line.c_str());
                sums_consistent = false;
            }
            continue;
        }
        double count = -1;
        if (!str2num(token[1], count) || count < 0) {
            if (has_sums) {
                warning("%s: Expect non-negative numerical value on all or none of the second column: %s", __func__, line.c_str());
                sums_consistent = false;
            }
            continue;
        }
        has_sums = true;
        sums.push_back(count);
    }
    if (features.empty()) {
        error("No features found in %s", featureFile.c_str());
    }
    if (!sums_consistent || !has_sums || sums.size() != features.size()) {
        has_sums = false;
        sums.clear();
    }

    nUnits = n_units;
    nModal = n_modal > 0 ? n_modal : 1;
    nLayer = n_layer > 0 ? n_layer : 1;
    nFeatures = static_cast<int32_t>(features.size());
    hexSize = 0.0;
    hasCoordinates = false;
    offset_data = 0;
    icol_layer = -1;
    icol_x = -1;
    icol_y = -1;
    mintokens = 0;
    header_info.clear();
    if (has_sums) {
        feature_sums_raw = sums;
        feature_sums = sums;
        readFullSums = true;
    } else {
        feature_sums.assign(nFeatures, 0.0);
        feature_sums_raw.assign(nFeatures, 0.0);
        readFullSums = false;
    }
    accumulate_sums = false;
    remap = false;
    idx_remap.clear();
    weights.clear();
    weightFeatures = false;
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
                        feature_sums_raw.assign(nFeatures, 0.0);
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
            if (has_sums) {
                feature_sums_raw[idx_prev] = count;
                feature_sums[idx_prev] = weightFeatures ? count * weights[idx_prev] : count;
            }
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
    if (readFullSums) {
        if (feature_sums_raw.size() == weights.size()) {
            feature_sums.assign(feature_sums_raw.size(), 0.0);
            for (size_t i = 0; i < feature_sums_raw.size(); ++i) {
                feature_sums[i] = feature_sums_raw[i] * weights[i];
            }
        } else if (feature_sums.size() == weights.size()) {
            for (size_t i = 0; i < feature_sums.size(); ++i) {
                feature_sums[i] *= weights[i];
            }
        }
    }
    weightFeatures = true;
}

void HexReader::applyWeights(Document& doc) const {
    if (!weightFeatures) {
        return;
    }
    for (size_t i = 0; i < doc.ids.size(); ++i) {
        uint32_t idx = doc.ids[i];
        if (idx < weights.size()) {
            doc.cnts[i] *= weights[idx];
        }
    }
}

void HexReader::setFeatureSums(const std::vector<double>& sums, bool read_full) {
    if (sums.size() != static_cast<size_t>(nFeatures)) {
        error("%s: input sums size (%zu) does not match nFeatures (%d)", __func__, sums.size(), nFeatures);
    }
    feature_sums_raw = sums;
    feature_sums = sums;
    if (weightFeatures && weights.size() == feature_sums.size()) {
        for (size_t i = 0; i < feature_sums.size(); ++i) {
            feature_sums[i] *= weights[i];
        }
    }
    readFullSums = read_full;
    accumulate_sums = false;
}

int32_t HexReader::readAll(std::vector<Document>& docs, std::vector<std::string>& info, const std::string &inFile, int32_t minCount, bool add2sums, int32_t limit, int32_t modal) {
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
        if (limit > 0 && n >= limit) {break;} // only for debugging
    }
    readFullSums = true;
    return n;
}

int32_t HexReader::readAll(std::vector<Document>& docs, const std::string &inFile, int32_t minCount, bool add2sums, int32_t limit, int32_t modal) {
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
        if (limit > 0 && n >= limit) {break;} // only for debugging
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
    std::vector<double> new_sums_raw(nFeatures, 0.0);
    if (!feature_sums_raw.empty()) {
        for (const auto& pair : new_idx_remap) {
            if (pair.first < feature_sums_raw.size()) {
                new_sums_raw[pair.second] = feature_sums_raw[pair.first];
            }
        }
    }
    feature_sums_raw = std::move(new_sums_raw);

    if (readFullSums) {
        feature_sums.assign(nFeatures, 0.0);
        if (weightFeatures && weights.size() == feature_sums_raw.size()) {
            for (size_t i = 0; i < feature_sums_raw.size(); ++i) {
                feature_sums[i] = feature_sums_raw[i] * weights[i];
            }
        } else {
            feature_sums = feature_sums_raw;
        }
    } else {
        feature_sums.assign(nFeatures, 0.0);
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
bool UnitValues3D::readFromLine(const std::string& line, int32_t nModal, bool labeled) {
    std::istringstream iss(line);
    std::string hexKey;
    try {
        iss >> hexKey >> x >> y >> z;
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
bool UnitValues3D::writeToFile(std::ostream& os, uint32_t key) const {
    os << uint32toHex(key) << "\t" << x << "\t" << y << "\t" << z;
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

SparseObsMinibatchReader::SparseObsMinibatchReader(const std::string &inFile, HexReader &reader,
    int32_t minCountTrain, double size_factor, double c, int32_t debug_N)
    : reader_(&reader), minCountTrain_(minCountTrain),
      size_factor_(size_factor), c_(c), per_doc_c_(c <= 0),
      debug_N_(debug_N) {
    inFileStream_.open(inFile);
    if (!inFileStream_) {
        error("Fail to open input file: %s", inFile.c_str());
    }
}

void SparseObsMinibatchReader::set_covariates(const std::string& covarFile,
    std::vector<uint32_t>* covar_idx, std::vector<std::string>* covar_names,
    bool allow_na, int32_t label_idx, const std::string& label_na,
    std::vector<std::string>* labels) {
    if (label_idx >= 0 && labels == nullptr) {
        error("%s: if label index is specified, a pointer to a vector to store the labels must be provided", __func__);
    }
    allow_na_ = allow_na;
    label_idx_ = label_idx;
    label_na_ = label_na;
    covar_idx_ = covar_idx;
    covar_names_ = covar_names;
    labels_ = labels;
    has_labels_ = (label_idx_ >= 0 && labels_ != nullptr);
    n_tokens_ = 0;
    n_covar_ = 0;
    has_covar_ = false;

    if (covarFile.empty()) {
        if (label_idx_ >= 0) {
            error("%s: label index requires a non-empty covariate file", __func__);
        }
        return;
    }

    if (covar_idx_ == nullptr || covar_names_ == nullptr) {
        error("%s: covariate index and name pointers must be provided when using covariates", __func__);
    }
    covar_names_->clear();
    if (labels_) {labels_->clear();}

    covarFileStream_.close();
    covarFileStream_.clear();
    covarFileStream_.open(covarFile);
    if (!covarFileStream_) {
        error("Fail to covariate file: %s", covarFile.c_str());
    }
    std::string line;
    if (!std::getline(covarFileStream_, line)) {
        error("Fail to parse covariate file: %s", covarFile.c_str());
    }
    std::vector<std::string> covar_header;
    split(covar_header, "\t ", line, UINT_MAX, true, true, true);
    n_tokens_ = static_cast<int32_t>(covar_header.size());
    if (covar_idx_->empty() && !has_labels_) {
        n_covar_ = covar_header.size() - 1;
        for (int32_t i = 1; i < n_tokens_; i++) {
            covar_idx_->push_back(i);
            covar_names_->push_back(covar_header[i]);
        }
    } else {
        n_covar_ = static_cast<int32_t>(covar_idx_->size());
        for (const auto i : *covar_idx_) {
            if (i >= static_cast<uint32_t>(n_tokens_)) {
                error("Covariate index %d is out of range [0,%d)", static_cast<int32_t>(i), n_tokens_);
            }
            covar_names_->push_back(covar_header[i]);
        }
    }
    notice("Covariate file has %d columns, using %d as covariates and %d as label", n_tokens_, n_covar_, (int32_t) has_labels_);
    has_covar_ = (n_covar_ > 0 || has_labels_);
}

int32_t SparseObsMinibatchReader::readBatch(std::vector<Document> &docs,
    std::vector<std::string> *rnames, int32_t batch_size) {
    docs.clear();
    if (rnames) {rnames->clear();}
    if (done_) {return 0;}
    docs.reserve(batch_size);
    if (rnames) {rnames->reserve(batch_size);}
    std::string line, info;
    std::vector<std::string> tokens;
    while (static_cast<int32_t>(docs.size()) < batch_size) {
        if (!std::getline(inFileStream_, line)) {
            done_ = true;
            break;
        }
        info.clear();
        line_idx_++;
        Document doc;
        int32_t ct = reader_->parseLine(doc, info, line);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        std::string covar_line;
        if (has_covar_) {
            if (!covarFileStream_) {
                error("Covariate file stream is not open");
            }
            if (!std::getline(covarFileStream_, covar_line)) {
                error("The number of lines in covariate file is less than that in data file");
            }
        }
        if (ct < minCountTrain_) {
            continue;
        }
        docs.push_back(std::move(doc));
        if (rnames) {rnames->emplace_back(std::to_string(line_idx_ - 1));}
        if (debug_N_ > 0 && line_idx_ > debug_N_) {
            done_ = true;
            break;
        }
    }
    return static_cast<int32_t>(docs.size());
}

int32_t SparseObsMinibatchReader::readBatch(std::vector<SparseObs> &docs,
    std::vector<std::string> *rnames, int32_t batch_size) {
    docs.clear();
    if (rnames) {rnames->clear();}
    if (done_) {return 0;}
    docs.reserve(batch_size);
    if (rnames) {rnames->reserve(batch_size);}
    std::string line, info;
    std::vector<std::string> tokens;
    while (static_cast<int32_t>(docs.size()) < batch_size) {
        if (!std::getline(inFileStream_, line)) {
            done_ = true;
            break;
        }
        info.clear();
        line_idx_++;
        SparseObs obs;
        int32_t ct = reader_->parseLine(obs.doc, info, line);
        if (ct < 0) {
            error("Error parsing line %s", line.c_str());
        }
        std::string covar_line;
        if (has_covar_) {
            if (!covarFileStream_) {
                error("Covariate file stream is not open");
            }
            if (!std::getline(covarFileStream_, covar_line)) {
                error("The number of lines in covariate file is less than that in data file");
            }
        }
        if (ct < minCountTrain_) {
            continue;
        }
        obs.c = per_doc_c_ ? ct / size_factor_ : c_;
        obs.ct_tot = ct;
        if (has_covar_) {
            split(tokens, "\t ", covar_line, UINT_MAX, true, true, true);
            if (tokens.size() != static_cast<size_t>(n_tokens_)) {
                error("Number of columns (%lu) of line [%s] in covariate file does not match header (%d)", tokens.size(), covar_line.c_str(), n_tokens_);
            }
            if (n_covar_ > 0) {
                obs.covar = Eigen::VectorXd::Zero(n_covar_);
                for (int32_t i = 0; i < n_covar_; i++) {
                    if (!str2double(tokens[(*covar_idx_)[i]], obs.covar(i))) {
                        if (!allow_na_) {
                            error("Invalid value for %d-th covariate %s.", i+1, tokens[(*covar_idx_)[i]].c_str());
                        }
                        obs.covar(i) = 0;
                    }
                }
            }
            if (has_labels_) {
                if (tokens[label_idx_] == label_na_) {
                    continue;
                }
                labels_->push_back(tokens[label_idx_]);
            }
        }
        docs.push_back(std::move(obs));
        if (rnames) {rnames->emplace_back(std::to_string(line_idx_ - 1));}
        if (debug_N_ > 0 && line_idx_ > debug_N_) {
            done_ = true;
            break;
        }
    }
    return static_cast<int32_t>(docs.size());
}

int32_t SparseObsMinibatchReader::readAll(std::vector<SparseObs> &docs,
    std::vector<std::string> &rnames, int32_t batch_size) {
    docs.clear();
    rnames.clear();
    if (done_) {return 0;}
    if (reader_->nUnits > 0) {
        docs.reserve(reader_->nUnits);
        rnames.reserve(reader_->nUnits);
    }
    std::vector<SparseObs> batch;
    std::vector<std::string> batch_names;
    while (true) {
        int32_t n_batch = readBatch(batch, &batch_names, batch_size);
        if (n_batch == 0) {break;}
        for (auto &obs : batch) {
            docs.push_back(std::move(obs));
        }
        for (auto &name : batch_names) {
            rnames.push_back(std::move(name));
        }
    }
    return static_cast<int32_t>(docs.size());
}

int32_t HexReader::readAll(Eigen::SparseMatrix<double, Eigen::RowMajor> &X,
    std::vector<std::string>& info, const std::string &inFile,
    int32_t minCount, bool add2sums, int32_t limit, int32_t modal) {
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
        if (limit > 0 && n >= limit) {break;} // only for debugging
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

DGEReader10X::~DGEReader10X() {
    closeMatrixStream();
}

void DGEReader10X::open(const std::string &dgeDir) {
    if (dgeDir.empty()) {
        error("DGE directory is empty");
    }
    std::string dir = dgeDir;
    if (dir.back() == '/') {
        dir.pop_back();
    }
    open(dir + "/barcodes.tsv.gz", dir + "/features.tsv.gz", dir + "/matrix.mtx.gz");
}

void DGEReader10X::open(const std::string &barcodesFile, const std::string &featuresFile,
    const std::string &matrixFile) {
    barcodesFile_ = barcodesFile;
    featuresFile_ = featuresFile;
    matrixFile_ = matrixFile;
    readBarcodes(barcodesFile_);
    readFeatures(featuresFile_);
    openMatrixStream();
}

bool DGEReader10X::next(Document& doc, int32_t* barcode_idx, std::string* barcode) {
    if (!stream_open_) {
        openMatrixStream();
    }
    doc.ids.clear();
    doc.cnts.clear();
    if (done_) {
        return false;
    }
    int32_t current_bi = -1;
    if (has_buffer_) {
        current_bi = buffered_barcode_;
        doc.ids.push_back(buffered_feature_);
        doc.cnts.push_back(buffered_count_);
        feature_totals[buffered_feature_] += buffered_count_;
        has_buffer_ = false;
    }
    bool reached_eof = false;
    int32_t bi = -1;
    uint32_t gi = 0;
    uint32_t ct = 0;
    while (true) {
        if (!readNextEntry(bi, gi, ct)) {
            reached_eof = true;
            break;
        }
        if (current_bi < 0) {
            current_bi = bi;
        }
        if (bi != current_bi) {
            has_buffer_ = true;
            buffered_barcode_ = bi;
            buffered_feature_ = gi;
            buffered_count_ = ct;
            break;
        }
        doc.ids.push_back(gi);
        doc.cnts.push_back(ct);
        feature_totals[gi] += ct;
    }
    if (doc.ids.empty() && current_bi < 0) {
        done_ = true;
        return false;
    }
    if (reached_eof) {
        done_ = true;
    }
    if (barcode_idx) {
        *barcode_idx = current_bi;
    }
    if (barcode) {
        if (current_bi >= 0 && current_bi < nBarcodes) {
            *barcode = barcodes[current_bi];
        } else {
            barcode->clear();
        }
    }
    return true;
}

int32_t DGEReader10X::readAll(std::vector<Document>& docs, int32_t minCount) {
    std::vector<std::string> barcodes_out;
    return readAll(docs, barcodes_out, minCount);
}

int32_t DGEReader10X::readAll(std::vector<Document>& docs, std::vector<std::string>& barcodes_out, int32_t minCount) {
    if (barcodes.empty() || features.empty()) {
        error("Barcodes and features must be loaded before readAll");
    }
    openMatrixStream();
    std::vector<Document> docs_by_bc(nBarcodes);
    std::vector<uint64_t> sums(nBarcodes, 0);
    std::vector<bool> seen(nBarcodes, false);
    int32_t bi = -1;
    uint32_t gi = 0;
    uint32_t ct = 0;
    while (true) {
        if (!readNextEntry(bi, gi, ct)) {
            break;
        }
        seen[bi] = true;
        docs_by_bc[bi].ids.push_back(gi);
        docs_by_bc[bi].cnts.push_back(ct);
        sums[bi] += ct;
        feature_totals[gi] += ct;
    }
    closeMatrixStream();
    docs.clear();
    barcodes_out.clear();
    if (nBarcodes > 0) {
        docs.reserve(nBarcodes);
        barcodes_out.reserve(nBarcodes);
    }
    uint64_t min_count = minCount > 0 ? static_cast<uint64_t>(minCount) : 0;
    for (int32_t i = 0; i < nBarcodes; ++i) {
        if (!seen[i]) {
            continue;
        }
        if (sums[i] < min_count) {
            continue;
        }
        docs.push_back(std::move(docs_by_bc[i]));
        barcodes_out.push_back(barcodes[i]);
    }
    return static_cast<int32_t>(docs.size());
}

int32_t DGEReader10X::readAll(std::vector<Document>& docs, std::vector<int32_t>& barcode_idx_out, int32_t minCount) {
    if (barcodes.empty() || features.empty()) {
        error("Barcodes and features must be loaded before readAll");
    }
    openMatrixStream();
    std::vector<Document> docs_by_bc(nBarcodes);
    std::vector<uint64_t> sums(nBarcodes, 0);
    std::vector<bool> seen(nBarcodes, false);
    int32_t bi = -1;
    uint32_t gi = 0;
    uint32_t ct = 0;
    while (true) {
        if (!readNextEntry(bi, gi, ct)) {
            break;
        }
        seen[bi] = true;
        docs_by_bc[bi].ids.push_back(gi);
        docs_by_bc[bi].cnts.push_back(ct);
        sums[bi] += ct;
        feature_totals[gi] += ct;
    }
    closeMatrixStream();
    docs.clear();
    barcode_idx_out.clear();
    if (nBarcodes > 0) {
        docs.reserve(nBarcodes);
        barcode_idx_out.reserve(nBarcodes);
    }
    uint64_t min_count = minCount > 0 ? static_cast<uint64_t>(minCount) : 0;
    for (int32_t i = 0; i < nBarcodes; ++i) {
        if (!seen[i]) {
            continue;
        }
        if (sums[i] < min_count) {
            continue;
        }
        docs.push_back(std::move(docs_by_bc[i]));
        barcode_idx_out.push_back(i);
    }
    return static_cast<int32_t>(docs.size());
}

bool DGEReader10X::readMinibatch(std::vector<Document>& docs, std::vector<int32_t>& barcode_idx_out,
    int32_t batchSize, int32_t maxUnits, int32_t minCount) {
    docs.clear();
    barcode_idx_out.clear();
    if (batchSize <= 0) {
        return true;
    }
    docs.reserve(batchSize);
    barcode_idx_out.reserve(batchSize);
    const bool unlimited = maxUnits <= 0;
    int32_t seen = 0;
    while (static_cast<int32_t>(docs.size()) < batchSize && (unlimited || seen < maxUnits)) {
        Document doc;
        int32_t barcode_idx = -1;
        if (!next(doc, &barcode_idx, nullptr)) {
            return false;
        }
        seen++;
        if (barcode_idx < 0) {
            continue;
        }
        if (minCount > 0 && doc.get_sum() < minCount) {
            continue;
        }
        docs.push_back(std::move(doc));
        barcode_idx_out.push_back(barcode_idx);
    }
    return true;
}

void DGEReader10X::readBarcodes(const std::string& path) {
    barcodes.clear();
    auto lines = read_lines_maybe_gz(path);
    barcodes.reserve(lines.size());
    if (keep_barcodes_) {
        for (auto &l : lines) {
            std::string bc = trim(l);
            barcodes.push_back(std::move(bc));
        }
    } else { // use 0-based indices
        for (size_t i = 0; i < lines.size(); ++i) {
            barcodes.push_back(std::to_string(i));
        }
    }
    if (barcodes.empty()) {
        error("No barcodes found in %s", path.c_str());
    }
    nBarcodes = static_cast<int32_t>(barcodes.size());
}

void DGEReader10X::readFeatures(const std::string& path) {
    base_features_.clear();
    feature_ids.clear();
    auto lines = read_lines_maybe_gz(path);
    base_features_.reserve(lines.size());
    feature_ids.reserve(lines.size());
    std::unordered_set<std::string> feature_set;
    for (auto &l : lines) {
        if (l.empty()) {
            continue;
        }
        std::istringstream is(l);
        std::string id, name;
        is >> id >> name;
        if (id.empty()) {
            continue;
        }
        if (name.empty()) {
            name = id;
        }
        feature_ids.push_back(id);
        std::string unique_name = name;
        if (feature_set.find(unique_name) != feature_set.end()) {
            unique_name = id;
        }
        feature_set.insert(unique_name);
        base_features_.push_back(std::move(unique_name));
    }
    if (base_features_.empty()) {
        error("No features found in %s", path.c_str());
    }
    nRawFeatures_ = static_cast<int32_t>(base_features_.size());
    if (!target_features_.empty()) {
        applyFeatureIndexRemap();
    } else {
        features = base_features_;
        nFeatures = static_cast<int32_t>(features.size());
        remap_ = false;
        idx_remap_.clear();
        resetFeatureTotals();
    }
}

int32_t DGEReader10X::setFeatureIndexRemap(const std::vector<std::string>& new_features, bool keep_unmapped) {
    target_features_ = new_features;
    keep_unmapped_ = keep_unmapped;
    if (base_features_.empty()) {
        return 0;
    }
    return applyFeatureIndexRemap();
}

int32_t DGEReader10X::applyFeatureIndexRemap() {
    if (target_features_.empty()) {
        features = base_features_;
        nFeatures = static_cast<int32_t>(features.size());
        remap_ = false;
        idx_remap_.clear();
        resetFeatureTotals();
        return nFeatures;
    }
    std::unordered_map<std::string, uint32_t> dict;
    dict.reserve(base_features_.size());
    for (size_t i = 0; i < base_features_.size(); ++i) {
        if (dict.find(base_features_[i]) == dict.end()) {
            dict[base_features_[i]] = static_cast<uint32_t>(i);
        }
    }
    idx_remap_.assign(base_features_.size(), -1);
    int32_t n_mapped = 0;
    for (size_t i = 0; i < target_features_.size(); ++i) {
        auto it = dict.find(target_features_[i]);
        if (it != dict.end()) {
            idx_remap_[it->second] = static_cast<int32_t>(i);
            n_mapped++;
        }
    }
    std::vector<std::string> new_list = target_features_;
    int32_t n_unmapped = 0;
    if (keep_unmapped_) {
        for (size_t i = 0; i < base_features_.size(); ++i) {
            if (idx_remap_[i] < 0) {
                idx_remap_[i] = static_cast<int32_t>(new_list.size());
                new_list.push_back(base_features_[i]);
                n_unmapped++;
            }
        }
    }
    features = std::move(new_list);
    nFeatures = static_cast<int32_t>(features.size());
    remap_ = true;
    notice("%s: %d features are kept out of %d, %d mapped to input set of size %d", __func__, n_mapped + n_unmapped, nRawFeatures_, n_mapped, (int)target_features_.size());
    resetFeatureTotals();
    return n_mapped;
}

void DGEReader10X::resetFeatureTotals() {
    if (nFeatures > 0) {
        feature_totals.assign(static_cast<size_t>(nFeatures), 0);
    } else {
        feature_totals.clear();
    }
}

void DGEReader10X::openMatrixStream() {
    closeMatrixStream();
    if (barcodes.empty() || features.empty()) {
        error("Barcodes and features must be loaded before opening matrix");
    }
    if (matrixFile_.empty()) {
        error("Matrix file is empty");
    }
    if (ends_with(matrixFile_, ".gz")) {
        gz_mtx_ = gzopen(matrixFile_.c_str(), "rb");
        if (!gz_mtx_) {
            error("Failed to open %s", matrixFile_.c_str());
        }
        gz_matrix_ = true;
    } else {
        mtx_in_.open(matrixFile_);
        if (!mtx_in_) {
            error("Failed to open %s", matrixFile_.c_str());
        }
        gz_matrix_ = false;
    }
    stream_open_ = true;
    done_ = false;
    has_buffer_ = false;
    resetFeatureTotals();
    readMatrixHeader();
}

void DGEReader10X::closeMatrixStream() {
    if (gz_mtx_) {
        gzclose(gz_mtx_);
        gz_mtx_ = nullptr;
    }
    if (mtx_in_.is_open()) {
        mtx_in_.close();
    }
    stream_open_ = false;
    header_read_ = false;
    done_ = false;
    has_buffer_ = false;
}

void DGEReader10X::readMatrixHeader() {
    if (header_read_) {
        return;
    }
    std::string line;
    while (readMatrixLine(line)) {
        if (line.empty() || line[0] == '%' || line[0] == '\n') {
            continue;
        }
        break;
    }
    if (line.empty()) {
        error("Missing header line in matrix file");
    }
    size_t nrows = 0, ncols = 0, nentries = 0;
    if (std::sscanf(line.c_str(), "%zu %zu %zu", &nrows, &ncols, &nentries) != 3) {
        error("Invalid header line in matrix file: %s", line.c_str());
    }
    nEntries = nentries;
    if (nRawFeatures_ > 0 && nrows != static_cast<size_t>(nRawFeatures_)) {
        warning("Matrix has %zu rows but features file has %d entries", nrows, nRawFeatures_);
    }
    if (ncols != static_cast<size_t>(nBarcodes)) {
        warning("Matrix has %zu columns but barcodes file has %d entries", ncols, nBarcodes);
    }
    header_read_ = true;
}

bool DGEReader10X::readMatrixLine(std::string& line) {
    if (gz_matrix_) {
        if (!gz_mtx_) {
            return false;
        }
        char *ret = gzgets(gz_mtx_, buf_.data(), static_cast<int>(buf_.size()));
        if (!ret) {
            return false;
        }
        line.assign(ret);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        return true;
    }
    if (!std::getline(mtx_in_, line)) {
        return false;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return true;
}

bool DGEReader10X::readNextEntry(int32_t& barcode_idx, uint32_t& feature_idx, uint32_t& count) {
    if (!stream_open_) {
        openMatrixStream();
    }
    if (!header_read_) {
        readMatrixHeader();
    }
    std::string line;
    while (readMatrixLine(line)) {
        if (line.empty() || line[0] == '%' || line[0] == '\n') {
            continue;
        }
        int row = 0;
        int col = 0;
        uint32_t ct = 0;
        if (std::sscanf(line.c_str(), "%d %d %u", &row, &col, &ct) != 3) {
            continue;
        }
        if (row <= 0 || col <= 0 || ct == 0) {
            continue;
        }
        row -= 1;
        col -= 1;
        if (row >= nRawFeatures_ || col >= nBarcodes) {
            continue;
        }
        if (remap_) {
            row = idx_remap_[row];
            if (row < 0) {
                continue;
            }
        }
        feature_idx = row;
        barcode_idx = col;
        count = ct;
        return true;
    }
    return false;
}
