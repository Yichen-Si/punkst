#include "dataunits.hpp"

int32_t HexReader::parseLine(Document& doc, int32_t& x, int32_t& y, int32_t& layer, const std::string &line, int32_t modal) {
    assert(modal < nModal && "Modal out of range");
    std::vector<int32_t> nfeatures(nModal, 0);
    std::vector<uint32_t> counts(nModal, 0);
    std::vector<std::string> token;
    split(token, "\t", line);
    size_t ntokens = token.size();
    if (ntokens < mintokens) {
        return -1;
    }
    if (icol_x >= 0 && icol_y >= 0) {
        if (!str2int32(token[icol_x], x)){
            return -1;
        }
        if (!str2int32(token[icol_y], y)){
            return -1;
        }
    }
    if (icol_layer >= 0) {
        if(!str2int32(token[icol_layer], layer)) {
            return -1;
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
    doc.ids.resize(nfeatures[modal]);
    doc.cnts.resize(nfeatures[modal]);
    for (int j = 0; j < nfeatures[modal]; ++j) {
        std::vector<std::string> pair;
        split(pair, " ", token[i]);
        if (!str2uint32(pair[0], doc.ids[j])) {return -1;}
        if (!str2double(pair[1], doc.cnts[j])) {return -1;}
        i += 1;
    }
    return counts[modal];
}


int32_t HexReader::parseLine(UnitValues &unit, const std::string &line) {
    std::vector<int32_t> nfeatures(nModal, 0);
    std::vector<uint32_t> counts(nModal, 0);
    std::vector<std::string> token;
    split(token, "\t", line);
    size_t ntokens = token.size();
    if (ntokens < mintokens) {
        return -1;
    }
    if (icol_x >= 0 && icol_y >= 0) {
        if (!str2int32(token[icol_x], unit.x)){return -1;}
        if (!str2int32(token[icol_y], unit.y)){return -1;}
    }
    if (icol_layer >= 0) {
        if(!str2int32(token[icol_layer], unit.label)) {return -1;}
    }
    uint32_t i = offset_data;
    int32_t total;
    for (int j = 0; j < nModal; j++) {
        if(!str2int32(token[i], nfeatures[j])) {return -1;}
        if(!str2uint32(token[i + 1], counts[j])) {return -1;}
        total += counts[j];
        i += 2;
    }
    unit.vals.resize(nModal);
    unit.valsums.resize(nModal);
    for (int j = 0; j < nModal; ++j) {
        unit.valsums[j] = counts[j];
        for (int k = 0; k < nfeatures[j]; ++k) {
            std::vector<std::string> pair;
            split(pair, " ", token[i]);
            uint32_t u, v;
            if (!str2uint32(pair[0], u)) {return -1;}
            if (!str2uint32(pair[1], v)) {return -1;}
            unit.vals[j][u] = v;
            i += 1;
        }
    }
    return total;
}

void HexReader::readMetadata(const std::string &metaFile) {
    std::ifstream metaIn(metaFile);
    if (!metaIn) {
        throw std::runtime_error("Error opening metadata file " + metaFile);
    }

    // Parse the JSON file.
    nlohmann::json meta;
    metaIn >> meta;
    hexSize = meta.value("hex_size", -0.0);
    hexGrid.init(hexSize);
    nUnits = meta.value("n_units", 0);
    nLayer = meta.value("n_layers", 1);
    nModal = meta.value("n_modalities", 1);
    nFeatures = meta.value("n_features", 0);
    offset_data = meta.value("offset_data", -1);
    if (offset_data < 0) {
        throw std::runtime_error("Error: offset_data not found in metadata file");
    }
    icol_layer = meta.value("icol_layer", -1);
    icol_x = meta.value("icol_x_hex", -1);
    icol_y = meta.value("icol_y_hex", -1);
    features.resize(nFeatures);
    if (meta.contains("dictionary")) {
        for (auto& item : meta["dictionary"].items()) {
            if (!item.value().is_number_integer()) {
                throw std::runtime_error("Dictionary (key: value) pairs must have integer values");
            }
            features[item.value()] = item.key();
        }
    } else {
        for (int i = 0; i < nFeatures; ++i) {
            features[i] = std::to_string(i);
        }
    }
    mintokens = offset_data + 2 * nModal;
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
