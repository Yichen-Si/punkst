#include "dataunits.hpp"
#include <cinttypes>
#include <regex>
#include <sstream>
#include <algorithm>
#include <utility>

UnitFactorResultHeader parse_unit_factor_result_header(
    const std::vector<std::string>& header,
    const UnitFactorResultReadOptions& options) {
    UnitFactorResultHeader out;
    out.columns = header;
    const bool expectCoords = !options.xColName.empty() || !options.yColName.empty();
    const bool expectTop = !options.topKColName.empty() || !options.topPColName.empty();
    if (expectCoords && (options.xColName.empty() || options.yColName.empty())) {
        error("%s: both xColName and yColName must be provided together", __func__);
    }
    if (expectTop && (options.topKColName.empty() || options.topPColName.empty())) {
        error("%s: both topKColName and topPColName must be provided together", __func__);
    }
    const bool useFactorRange = (options.factorColBegin >= 0 || options.factorColEnd >= 0);
    if (useFactorRange && (options.factorColBegin < 0 || options.factorColEnd < 0)) {
        error("%s: factorColBegin and factorColEnd must be provided together", __func__);
    }
    for (size_t i = 0; i < header.size(); ++i) {
        if (!options.unitIdColName.empty() && header[i] == options.unitIdColName) {
            out.unitIdCol = static_cast<int32_t>(i);
        } else if (!options.xColName.empty() && header[i] == options.xColName) {
            out.xCol = static_cast<int32_t>(i);
        } else if (!options.yColName.empty() && header[i] == options.yColName) {
            out.yCol = static_cast<int32_t>(i);
        } else if (!options.topKColName.empty() && header[i] == options.topKColName) {
            out.topKCol = static_cast<int32_t>(i);
        } else if (!options.topPColName.empty() && header[i] == options.topPColName) {
            out.topPCol = static_cast<int32_t>(i);
        }
        if (useFactorRange) {
            continue;
        }
        int32_t factorIdx = -1;
        if (str2int32(header[i], factorIdx) && factorIdx >= 0) {
            out.factorCols.emplace_back(factorIdx, static_cast<int32_t>(i));
            out.factorNames.emplace_back(header[i]);
        }
    }
    if (!options.unitIdColName.empty() && out.unitIdCol < 0) {
        error("%s: input header must contain %s", __func__, options.unitIdColName.c_str());
    }
    if (expectCoords && !out.hasCoordinates()) {
        error("%s: input header must contain %s and %s",
            __func__, options.xColName.c_str(), options.yColName.c_str());
    }
    if (expectTop && !out.hasTopFactor()) {
        error("%s: input header must contain %s and %s",
            __func__, options.topKColName.c_str(), options.topPColName.c_str());
    }
    if (useFactorRange) {
        if (options.factorColBegin < 0 || options.factorColEnd <= options.factorColBegin ||
            static_cast<size_t>(options.factorColEnd) > header.size()) {
            error("%s: invalid factor column range [%d, %d)",
                __func__, options.factorColBegin, options.factorColEnd);
        }
        const int32_t nFactors = options.factorColEnd - options.factorColBegin;
        out.factorCols.reserve(static_cast<size_t>(nFactors));
        out.factorNames.reserve(static_cast<size_t>(nFactors));
        for (int32_t col = options.factorColBegin; col < options.factorColEnd; ++col) {
            out.factorCols.emplace_back(col - options.factorColBegin, col);
            out.factorNames.emplace_back(header[static_cast<size_t>(col)]);
        }
    }
    if (out.factorCols.empty()) {
        error("%s: no numeric factor-probability columns were found in the input header", __func__);
    }
    std::sort(out.factorCols.begin(), out.factorCols.end());
    for (size_t i = 0; i < out.factorCols.size(); ++i) {
        if (out.factorCols[i].first != static_cast<int32_t>(i)) {
            error("%s: factor columns must be contiguous 0..K-1; found gap around %d",
                __func__, out.factorCols[i].first);
        }
    }
    out.factorNames.clear();
    out.factorNames.reserve(out.factorCols.size());
    for (const auto& factorCol : out.factorCols) {
        out.factorNames.emplace_back(header[static_cast<size_t>(factorCol.second)]);
    }
    return out;
}

void parse_unit_factor_result_row(UnitFactorResultRow& row,
    const std::vector<std::string>& fields,
    const UnitFactorResultHeader& header,
    uint64_t rowIndex) {
    const uint64_t inputRowNumber = rowIndex + 1;
    if (fields.size() < header.columns.size()) {
        error("%s: expected at least %zu columns, found %zu at input row %" PRIu64,
            __func__, header.columns.size(), fields.size(), inputRowNumber);
    }
    row.rowIndex = rowIndex;
    row.unitId = header.hasUnitId() ? fields[header.unitIdCol] : std::to_string(rowIndex);
    row.hasCoordinates = false;
    row.hasTopFactor = false;
    row.x = 0.0;
    row.y = 0.0;
    row.topK = -1;
    row.topP = 0.0f;
    if (header.hasCoordinates()) {
        if (!str2double(fields[header.xCol], row.x) ||
            !str2double(fields[header.yCol], row.y)) {
            error("%s: failed parsing x/y at input row %" PRIu64,
                __func__, inputRowNumber);
        }
        row.hasCoordinates = true;
    }
    if (header.hasTopFactor()) {
        if (!str2int32(fields[header.topKCol], row.topK) ||
            !str2float(fields[header.topPCol], row.topP)) {
            error("%s: failed parsing topK/topP at input row %" PRIu64,
                __func__, inputRowNumber);
        }
        row.hasTopFactor = true;
    }
    row.factorValues.resize(header.factorCols.size());
    for (size_t i = 0; i < header.factorCols.size(); ++i) {
        const int32_t colIdx = header.factorCols[i].second;
        if (!str2float(fields[colIdx], row.factorValues[i])) {
            error("%s: failed parsing factor column %s at input row %" PRIu64,
                __func__, header.columns[colIdx].c_str(), inputRowNumber);
        }
    }
}

UnitFactorResultReader::UnitFactorResultReader(const std::string& path,
    const UnitFactorResultReadOptions& options) : reader_(path) {
    std::string line;
    while (reader_.getline(line)) {
        if (!line.empty()) {
            break;
        }
    }
    if (line.empty()) {
        error("%s: input file %s is empty", __func__, path.c_str());
    }
    if (!line.empty() && line.front() == '#') {
        line.erase(line.begin());
    }
    std::vector<std::string> header;
    split(header, "\t", line);
    header_ = parse_unit_factor_result_header(header, options);
}

bool UnitFactorResultReader::next(UnitFactorResultRow& row) {
    std::string line;
    while (reader_.getline(line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields;
        split(fields, "\t", line);
        parse_unit_factor_result_row(row, fields, header_, rowIndex_);
        ++rowIndex_;
        return true;
    }
    return false;
}

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
    doc.ct_tot = -1;
    doc.raw_ct_tot = -1;
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
    doc.raw_ct_tot = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
    doc.ct_tot = doc.raw_ct_tot;
    if (weightFeatures) {
        double weighted_total = 0.0;
        for (size_t i = 0; i < doc.ids.size(); i++) {
            doc.cnts[i] *= weights[doc.ids[i]];
            weighted_total += doc.cnts[i];
        }
        doc.ct_tot = weighted_total;
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
    icol_z = meta.value("icol_z", -1);
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
    int32_t n_units) {
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
    nFeatures = static_cast<int32_t>(features.size());
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

void HexReader::initFromFeatures(const std::vector<std::string>& featureNames, int32_t n_units) {
    nUnits = n_units;
    nFeatures = static_cast<int32_t>(featureNames.size());
    features = featureNames;

    feature_sums.clear();
    feature_sums_raw.clear();
    feature_sums.assign(nFeatures, 0.0);
    feature_sums_raw.assign(nFeatures, 0.0);
    readFullSums = false;
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

int32_t HexReader::filterCurrentFeatures(int32_t minCount,
    const std::string& include_ftr_regex, const std::string& exclude_ftr_regex) {
    if (features.empty()) {
        return 0;
    }
    if (minCount > 1 && !readFullSums) {
        error("%s: Feature totals are required to apply min-count filtering", __func__);
    }

    const bool check_include = !include_ftr_regex.empty();
    const bool check_exclude = !exclude_ftr_regex.empty();
    std::regex regex_include(include_ftr_regex);
    std::regex regex_exclude(exclude_ftr_regex);

    const std::vector<double>& sums = feature_sums_raw.empty() ? feature_sums : feature_sums_raw;
    std::unordered_map<uint32_t, uint32_t> remap_new;
    uint32_t kept = 0;
    for (uint32_t i = 0; i < static_cast<uint32_t>(features.size()); ++i) {
        if (minCount > 1 && i < sums.size() && sums[i] < minCount) {
            continue;
        }
        const std::string& feature = features[i];
        const bool include = !check_include || std::regex_match(feature, regex_include);
        const bool exclude = check_exclude && std::regex_match(feature, regex_exclude);
        if (include && !exclude) {
            remap_new[i] = kept++;
        }
    }
    notice("%s: %d features are kept out of %d", __func__, kept, (int)features.size());
    setFeatureIndexRemap(remap_new);
    return static_cast<int32_t>(kept);
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
    if (doc.raw_ct_tot < 0) { // keep the raw total count first
        doc.raw_ct_tot = (doc.ct_tot >= 0) ? doc.ct_tot
            : std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
    }
    for (size_t i = 0; i < doc.ids.size(); ++i) {
        uint32_t idx = doc.ids[i];
        if (idx < weights.size()) {
            doc.cnts[i] *= weights[idx];
        }
    }
    doc.ct_tot = -1;
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
        const double raw_ct = (doc.raw_ct_tot >= 0.0) ? doc.raw_ct_tot : doc.get_sum();
        if (raw_ct < minCountTrain_) {
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
        const double raw_ct = (obs.doc.raw_ct_tot >= 0.0) ? obs.doc.raw_ct_tot : obs.doc.get_sum();
        if (raw_ct < minCountTrain_) {
            continue;
        }
        obs.c = per_doc_c_ ? raw_ct / size_factor_ : c_;
        obs.ct_tot = raw_ct;
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

std::vector<DGEReader10X::DatasetInput> resolveDge10XInputs(
    const std::vector<std::string>& dgeDirs,
    const std::vector<std::string>& barcodesFiles,
    const std::vector<std::string>& featuresFiles,
    const std::vector<std::string>& matrixFiles,
    const std::vector<std::string>& datasetIds) {
    std::vector<DGEReader10X::DatasetInput> inputs;
    if (!barcodesFiles.empty() || !featuresFiles.empty() || !matrixFiles.empty()) {
        if (barcodesFiles.empty() || featuresFiles.empty() || matrixFiles.empty()) {
            error("10X inputs require --in-barcodes, --in-features, and --in-matrix together");
        }
        if (barcodesFiles.size() != featuresFiles.size() || barcodesFiles.size() != matrixFiles.size()) {
            error("10X input lists must have matching lengths");
        }
        inputs.resize(barcodesFiles.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            inputs[i].barcodes_file = barcodesFiles[i];
            inputs[i].features_file = featuresFiles[i];
            inputs[i].matrix_file = matrixFiles[i];
        }
    } else if (!dgeDirs.empty()) {
        inputs.resize(dgeDirs.size());
        for (size_t i = 0; i < inputs.size(); ++i) {
            std::string dir = dgeDirs[i];
            if (dir.empty()) {
                error("10X input directory is empty");
            }
            if (dir.back() == '/') {
                dir.pop_back();
            }
            inputs[i].barcodes_file = dir + "/barcodes.tsv.gz";
            inputs[i].features_file = dir + "/features.tsv.gz";
            inputs[i].matrix_file = dir + "/matrix.mtx.gz";
        }
    }

    if (inputs.empty()) {
        return inputs;
    }
    if (!datasetIds.empty() && datasetIds.size() != inputs.size()) {
        error("The number of dataset IDs (%zu) does not match the number of 10X datasets (%zu)",
            datasetIds.size(), inputs.size());
    }
    std::unordered_set<std::string> seen_ids;
    seen_ids.reserve(inputs.size());
    for (size_t i = 0; i < inputs.size(); ++i) {
        std::string id = datasetIds.empty() ? std::to_string(i + 1) : datasetIds[i];
        if (id.empty()) {
            error("Dataset ID for dataset %zu is empty", i + 1);
        }
        if (!seen_ids.insert(id).second) {
            error("Duplicate dataset ID: %s", id.c_str());
        }
        inputs[i].dataset_id = std::move(id);
    }
    return inputs;
}

void DGEReader10X::open(const std::string &dgeDir) {
    open(std::vector<std::string>{dgeDir});
}

void DGEReader10X::open(const std::vector<std::string> &dgeDirs,
    const std::vector<std::string> &datasetIds) {
    loadDatasets(resolveDge10XInputs(dgeDirs, {}, {}, {}, datasetIds));
}

void DGEReader10X::open(const std::string &barcodesFile, const std::string &featuresFile,
    const std::string &matrixFile) {
    open(std::vector<std::string>{barcodesFile},
        std::vector<std::string>{featuresFile},
        std::vector<std::string>{matrixFile});
}

void DGEReader10X::open(const std::vector<std::string> &barcodesFiles,
    const std::vector<std::string> &featuresFiles,
    const std::vector<std::string> &matrixFiles,
    const std::vector<std::string> &datasetIds) {
    loadDatasets(resolveDge10XInputs({}, barcodesFiles, featuresFiles, matrixFiles, datasetIds));
}

void DGEReader10X::loadDatasets(const std::vector<DatasetInput>& inputs) {
    closeMatrixStream();
    datasets_.clear();
    dataset_offsets_.clear();
    barcodes.clear();
    features.clear();
    feature_ids.clear();
    feature_totals.clear();
    base_features_.clear();
    target_features_.clear();
    nBarcodes = 0;
    nFeatures = 0;
    nEntries = 0;
    keep_unmapped_ = false;

    if (inputs.empty()) {
        error("No 10X datasets were provided");
    }

    datasets_.reserve(inputs.size());
    for (const auto& input : inputs) {
        DatasetState dataset;
        dataset.input = input;
        readBarcodes(dataset);
        readFeatures(dataset);
        nEntries += dataset.nEntries;
        datasets_.push_back(std::move(dataset));
    }

    if (datasets_.size() == 1) {
        base_features_ = datasets_[0].raw_features;
        feature_ids = datasets_[0].feature_ids;
    } else {
        std::unordered_map<std::string, int32_t> feature_counts;
        for (const auto& dataset : datasets_) {
            std::unordered_set<std::string> seen;
            seen.reserve(dataset.raw_features.size());
            for (const auto& feature : dataset.raw_features) {
                if (seen.insert(feature).second) {
                    feature_counts[feature] += 1;
                }
            }
        }
        base_features_.reserve(datasets_[0].raw_features.size());
        feature_ids.reserve(datasets_[0].raw_features.size());
        for (size_t i = 0; i < datasets_[0].raw_features.size(); ++i) {
            const auto& feature = datasets_[0].raw_features[i];
            auto it = feature_counts.find(feature);
            if (it != feature_counts.end() && it->second == static_cast<int32_t>(datasets_.size())) {
                base_features_.push_back(feature);
                feature_ids.push_back(i < datasets_[0].feature_ids.size() ? datasets_[0].feature_ids[i] : feature);
            }
        }
        if (base_features_.empty()) {
            error("No shared features remain in the intersection of all 10X datasets");
        }
    }

    applyFeatureIndexRemap();
    rebuildUnitMetadata();
}

bool DGEReader10X::next(Document& doc, int32_t* barcode_idx, std::string* barcode) {
    if (!stream_initialized_) {
        openMatrixStream();
    }
    doc.ids.clear();
    doc.cnts.clear();
    doc.ct_tot = -1;
    doc.raw_ct_tot = -1;

    while (current_dataset_ < datasets_.size()) {
        auto& dataset = datasets_[current_dataset_];
        if (dataset.done && !dataset.has_buffer) {
            closeDatasetMatrixStream(dataset);
            ++current_dataset_;
            continue;
        }
        if (!dataset.stream_open) {
            openDatasetMatrixStream(current_dataset_);
        }

        int32_t current_bi = -1;
        double raw_total = 0.0;
        if (dataset.has_buffer) {
            current_bi = dataset.buffered_barcode;
            doc.ids.push_back(dataset.buffered_feature);
            doc.cnts.push_back(dataset.buffered_count);
            feature_totals[dataset.buffered_feature] += dataset.buffered_count;
            raw_total += dataset.buffered_count;
            dataset.has_buffer = false;
        }

        bool reached_eof = false;
        int32_t bi = -1;
        uint32_t gi = 0;
        uint32_t ct = 0;
        while (true) {
            if (!readNextEntry(current_dataset_, bi, gi, ct)) {
                reached_eof = true;
                break;
            }
            if (current_bi < 0) {
                current_bi = bi;
            }
            if (bi != current_bi) {
                dataset.has_buffer = true;
                dataset.buffered_barcode = bi;
                dataset.buffered_feature = gi;
                dataset.buffered_count = ct;
                break;
            }
            doc.ids.push_back(gi);
            doc.cnts.push_back(ct);
            feature_totals[gi] += ct;
            raw_total += ct;
        }

        if (doc.ids.empty() && current_bi < 0) {
            dataset.done = true;
            closeDatasetMatrixStream(dataset);
            ++current_dataset_;
            continue;
        }
        if (reached_eof) {
            dataset.done = true;
        }
        doc.raw_ct_tot = raw_total;
        doc.ct_tot = raw_total;
        const int32_t global_idx = dataset_offsets_[current_dataset_] + current_bi;
        if (barcode_idx) {
            *barcode_idx = global_idx;
        }
        if (barcode) {
            *barcode = getUnitId(global_idx);
        }
        return true;
    }

    return false;
}

int32_t DGEReader10X::readAll(std::vector<Document>& docs, int32_t minCount) {
    std::vector<std::string> barcodes_out;
    return readAll(docs, barcodes_out, minCount);
}

int32_t DGEReader10X::readAll(std::vector<Document>& docs, std::vector<std::string>& barcodes_out, int32_t minCount) {
    if (barcodes.empty() || features.empty()) {
        error("Barcodes and features must be loaded before readAll");
    }
    closeMatrixStream();
    openMatrixStream();
    docs.clear();
    barcodes_out.clear();
    if (nBarcodes > 0) {
        docs.reserve(nBarcodes);
        barcodes_out.reserve(nBarcodes);
    }
    const uint64_t min_count = minCount > 0 ? static_cast<uint64_t>(minCount) : 0;
    while (true) {
        Document doc;
        int32_t global_idx = -1;
        if (!next(doc, &global_idx, nullptr)) {
            break;
        }
        if (global_idx < 0) {
            continue;
        }
        if (min_count > 0 && doc.get_sum() < min_count) {
            continue;
        }
        docs.push_back(std::move(doc));
        barcodes_out.push_back(getUnitId(global_idx));
    }
    closeMatrixStream();
    return static_cast<int32_t>(docs.size());
}

int32_t DGEReader10X::readAll(std::vector<Document>& docs, std::vector<int32_t>& barcode_idx_out, int32_t minCount) {
    if (barcodes.empty() || features.empty()) {
        error("Barcodes and features must be loaded before readAll");
    }
    closeMatrixStream();
    openMatrixStream();
    docs.clear();
    barcode_idx_out.clear();
    if (nBarcodes > 0) {
        docs.reserve(nBarcodes);
        barcode_idx_out.reserve(nBarcodes);
    }
    const uint64_t min_count = minCount > 0 ? static_cast<uint64_t>(minCount) : 0;
    while (true) {
        Document doc;
        int32_t global_idx = -1;
        if (!next(doc, &global_idx, nullptr)) {
            break;
        }
        if (global_idx < 0) {
            continue;
        }
        if (min_count > 0 && doc.get_sum() < min_count) {
            continue;
        }
        docs.push_back(std::move(doc));
        barcode_idx_out.push_back(global_idx);
    }
    closeMatrixStream();
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
        int32_t unit_idx = -1;
        if (!next(doc, &unit_idx, nullptr)) {
            return false;
        }
        seen++;
        if (unit_idx < 0) {
            continue;
        }
        if (minCount > 0 && doc.get_sum() < minCount) {
            continue;
        }
        docs.push_back(std::move(doc));
        barcode_idx_out.push_back(unit_idx);
    }
    return true;
}

void DGEReader10X::readBarcodes(DatasetState& dataset) {
    dataset.local_barcodes.clear();
    auto lines = read_lines_maybe_gz(dataset.input.barcodes_file);
    dataset.local_barcodes.reserve(lines.size());
    if (keep_barcodes_) {
        for (auto &l : lines) {
            std::string bc = trim(l);
            dataset.local_barcodes.push_back(std::move(bc));
        }
    } else {
        for (size_t i = 0; i < lines.size(); ++i) {
            dataset.local_barcodes.push_back(std::to_string(i));
        }
    }
    if (dataset.local_barcodes.empty()) {
        error("No barcodes found in %s", dataset.input.barcodes_file.c_str());
    }
    dataset.nBarcodes = static_cast<int32_t>(dataset.local_barcodes.size());
}

void DGEReader10X::readFeatures(DatasetState& dataset) {
    dataset.raw_features.clear();
    dataset.feature_ids.clear();
    auto lines = read_lines_maybe_gz(dataset.input.features_file);
    dataset.raw_features.reserve(lines.size());
    dataset.feature_ids.reserve(lines.size());
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
        dataset.feature_ids.push_back(id);
        std::string unique_name = name;
        if (feature_set.find(unique_name) != feature_set.end()) {
            unique_name = id;
        }
        feature_set.insert(unique_name);
        dataset.raw_features.push_back(std::move(unique_name));
    }
    if (dataset.raw_features.empty()) {
        error("No features found in %s", dataset.input.features_file.c_str());
    }
    dataset.nRawFeatures = static_cast<int32_t>(dataset.raw_features.size());
    dataset.nEntries = 0;
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
    if (keep_unmapped_ && datasets_.size() > 1) {
        warning("%s: keep_unmapped is not supported for joint 10X datasets; ignoring unmapped features", __func__);
    }
    std::vector<std::string> active_features = target_features_.empty() ? base_features_ : target_features_;
    if (active_features.empty()) {
        features.clear();
        nFeatures = 0;
        resetFeatureTotals();
        return 0;
    }

    std::unordered_map<std::string, uint32_t> target_dict;
    target_dict.reserve(active_features.size());
    for (size_t i = 0; i < active_features.size(); ++i) {
        if (target_dict.find(active_features[i]) == target_dict.end()) {
            target_dict[active_features[i]] = static_cast<uint32_t>(i);
        }
    }

    int32_t n_mapped = 0;
    for (auto& dataset : datasets_) {
        dataset.idx_remap.assign(dataset.raw_features.size(), -1);
        for (size_t i = 0; i < dataset.raw_features.size(); ++i) {
            auto it = target_dict.find(dataset.raw_features[i]);
            if (it != target_dict.end()) {
                dataset.idx_remap[i] = static_cast<int32_t>(it->second);
            }
        }
    }

    std::vector<std::string> new_list = active_features;
    if (keep_unmapped_ && datasets_.size() == 1) {
        auto& dataset = datasets_[0];
        std::unordered_set<std::string> seen(active_features.begin(), active_features.end());
        for (size_t i = 0; i < dataset.raw_features.size(); ++i) {
            if (dataset.idx_remap[i] >= 0) {
                ++n_mapped;
                continue;
            }
            const auto& feature = dataset.raw_features[i];
            if (seen.insert(feature).second) {
                dataset.idx_remap[i] = static_cast<int32_t>(new_list.size());
                new_list.push_back(feature);
            }
        }
    } else {
        n_mapped = static_cast<int32_t>(new_list.size());
    }

    features = std::move(new_list);
    nFeatures = static_cast<int32_t>(features.size());
    notice("%s: %d features are kept out of %zu shared features, mapped to input set of size %zu",
        __func__, nFeatures, base_features_.size(), active_features.size());
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
    current_dataset_ = 0;
    stream_initialized_ = true;
    resetFeatureTotals();
    if (!datasets_.empty()) {
        openDatasetMatrixStream(0);
    }
}

void DGEReader10X::closeMatrixStream() {
    for (auto& dataset : datasets_) {
        closeDatasetMatrixStream(dataset);
        dataset.done = false;
        dataset.has_buffer = false;
        dataset.header_read = false;
        dataset.buffered_barcode = -1;
        dataset.buffered_feature = 0;
        dataset.buffered_count = 0;
    }
    current_dataset_ = 0;
    stream_initialized_ = false;
}

void DGEReader10X::openDatasetMatrixStream(size_t dataset_idx) {
    if (dataset_idx >= datasets_.size()) {
        return;
    }
    auto& dataset = datasets_[dataset_idx];
    closeDatasetMatrixStream(dataset);
    if (dataset.input.matrix_file.empty()) {
        error("Matrix file is empty");
    }
    if (ends_with(dataset.input.matrix_file, ".gz")) {
        dataset.gz_mtx = gzopen(dataset.input.matrix_file.c_str(), "rb");
        if (!dataset.gz_mtx) {
            error("Failed to open %s", dataset.input.matrix_file.c_str());
        }
        dataset.gz_matrix = true;
    } else {
        dataset.mtx_in.open(dataset.input.matrix_file);
        if (!dataset.mtx_in) {
            error("Failed to open %s", dataset.input.matrix_file.c_str());
        }
        dataset.gz_matrix = false;
    }
    dataset.stream_open = true;
    dataset.done = false;
    dataset.has_buffer = false;
    dataset.header_read = false;
    readMatrixHeader(dataset);
}

void DGEReader10X::closeDatasetMatrixStream(DatasetState& dataset) {
    if (dataset.gz_mtx) {
        gzclose(dataset.gz_mtx);
        dataset.gz_mtx = nullptr;
    }
    if (dataset.mtx_in.is_open()) {
        dataset.mtx_in.close();
    }
    dataset.stream_open = false;
}

void DGEReader10X::readMatrixHeader(DatasetState& dataset) {
    if (dataset.header_read) {
        return;
    }
    std::string line;
    while (readMatrixLine(dataset, line)) {
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
    dataset.nEntries = nentries;
    if (dataset.nRawFeatures > 0 && nrows != static_cast<size_t>(dataset.nRawFeatures)) {
        warning("Matrix has %zu rows but features file has %d entries", nrows, dataset.nRawFeatures);
    }
    if (ncols != static_cast<size_t>(dataset.nBarcodes)) {
        warning("Matrix has %zu columns but barcodes file has %d entries", ncols, dataset.nBarcodes);
    }
    dataset.header_read = true;
}

bool DGEReader10X::readMatrixLine(DatasetState& dataset, std::string& line) {
    if (dataset.gz_matrix) {
        if (!dataset.gz_mtx) {
            return false;
        }
        char *ret = gzgets(dataset.gz_mtx, dataset.buf.data(), static_cast<int>(dataset.buf.size()));
        if (!ret) {
            return false;
        }
        line.assign(ret);
        while (!line.empty() && (line.back() == '\n' || line.back() == '\r')) {
            line.pop_back();
        }
        return true;
    }
    if (!std::getline(dataset.mtx_in, line)) {
        return false;
    }
    if (!line.empty() && line.back() == '\r') {
        line.pop_back();
    }
    return true;
}

bool DGEReader10X::readNextEntry(size_t dataset_idx, int32_t& barcode_idx, uint32_t& feature_idx, uint32_t& count) {
    if (dataset_idx >= datasets_.size()) {
        return false;
    }
    auto& dataset = datasets_[dataset_idx];
    if (!dataset.stream_open) {
        openDatasetMatrixStream(dataset_idx);
    }
    if (!dataset.header_read) {
        readMatrixHeader(dataset);
    }
    std::string line;
    while (readMatrixLine(dataset, line)) {
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
        if (row >= dataset.nRawFeatures || col >= dataset.nBarcodes) {
            continue;
        }
        if (row >= static_cast<int32_t>(dataset.idx_remap.size())) {
            continue;
        }
        row = dataset.idx_remap[row];
        if (row < 0) {
            continue;
        }
        feature_idx = static_cast<uint32_t>(row);
        barcode_idx = col;
        count = ct;
        return true;
    }
    return false;
}

void DGEReader10X::rebuildUnitMetadata() {
    dataset_offsets_.clear();
    dataset_offsets_.reserve(datasets_.size());
    barcodes.clear();
    nBarcodes = 0;
    for (const auto& dataset : datasets_) {
        dataset_offsets_.push_back(nBarcodes);
        nBarcodes += dataset.nBarcodes;
    }
    barcodes.reserve(static_cast<size_t>(nBarcodes));
    const bool add_prefix = datasets_.size() > 1;
    for (const auto& dataset : datasets_) {
        for (const auto& local_barcode : dataset.local_barcodes) {
            if (add_prefix) {
                barcodes.push_back(dataset.input.dataset_id + ":" + local_barcode);
            } else {
                barcodes.push_back(local_barcode);
            }
        }
    }
}

const std::string& DGEReader10X::getUnitId(int32_t global_unit_idx) const {
    if (global_unit_idx < 0 || global_unit_idx >= static_cast<int32_t>(barcodes.size())) {
        error("%s: global unit index %d is out of range [0,%zu)",
            __func__, global_unit_idx, barcodes.size());
    }
    return barcodes[global_unit_idx];
}
