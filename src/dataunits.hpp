#pragma once

#include "hexgrid.h"
#include "utils.h"
#include "json.hpp"
#include "assert.h"

template<typename T>
struct IndexEntry {
    uint64_t st, ed;
    uint32_t n;
    T xmin, xmax, ymin, ymax;
};

struct Document {
    std::vector<uint32_t> ids; // Length: number of nonzero words in the doc
    std::vector<double> cnts;
};

struct PixelValues {
    double x, y;
    uint32_t feature;
    std::vector<int32_t> intvals;
    PixelValues() {}
    bool writeToFileText(std::ostream& os) const {
        os << x << "\t" << y << "\t" << feature;
        for (const auto& val : intvals) {
            os << "\t" << val;
        }
        os << "\n";
        return os.good();
    }
};

struct UnitValues {
    int32_t nPixel;
    int32_t x, y;
    int32_t label;
    std::vector<std::unordered_map<uint32_t, uint32_t>> vals;
    std::vector<uint32_t> valsums;
    UnitValues(int32_t hx, int32_t hy, int32_t n = 0, int32_t l = -1) : nPixel(0), x(hx), y(hy), label(l) {
        if (n > 0) {
            vals.resize(n);
            valsums.resize(n);
            std::fill(valsums.begin(), valsums.end(), 0);
        }
    }
    UnitValues(int32_t hx, int32_t hy, const PixelValues& pixel, int32_t l = -1) : nPixel(1), x(hx), y(hy), label(l) {
        vals.resize(pixel.intvals.size());
        valsums.resize(pixel.intvals.size());
        std::fill(valsums.begin(), valsums.end(), 0);
        for (size_t i = 0; i < pixel.intvals.size(); ++i) {
            if (pixel.intvals[i] > 0) {
                vals[i][pixel.feature] = pixel.intvals[i];
                valsums[i] += pixel.intvals[i];
            }
        }
    }
    void clear() {
        nPixel = 0;
        vals.clear();
        valsums.clear();
    }
    void addPixel(const PixelValues& pixel) {
        ++nPixel;
        if (vals.size() == 0) {
            vals.resize(pixel.intvals.size());
            valsums.resize(pixel.intvals.size());
            std::fill(valsums.begin(), valsums.end(), 0);
        }
        assert(vals.size() == pixel.intvals.size() && "pixel and unit have different number of layers");
        for (size_t i = 0; i < pixel.intvals.size(); ++i) {
            if (pixel.intvals[i] > 0) {
                vals[i][pixel.feature] += pixel.intvals[i];
                valsums[i] += pixel.intvals[i];
            }
        }
    }
    bool mergeUnits(const UnitValues& other) {
        if (x != other.x || y != other.y) {
            return false;
        }
        if ((vals.size() != other.vals.size()) || (valsums.size() != other.valsums.size())) {
            return false;
        }
        nPixel += other.nPixel;
        for (size_t i = 0; i < vals.size(); ++i) {
            for (const auto& entry : other.vals[i]) {
                if (entry.second > 0)
                    vals[i][entry.first] += entry.second;
            }
            valsums[i] += other.valsums[i];
        }
        return true;
    }
    bool writeToFile(std::ostream& os, uint32_t key) const {
        os << uint32toHex(key) << "\t";
        if (label >= 0)
            os << label << "\t";
        os << x << "\t" << y;
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
    bool readFromLine(const std::string& line, int32_t nModal, bool labeled = false) {
        std::istringstream iss(line);
        std::string hexKey;
        try {
            if (labeled) {
                iss >> hexKey >> label >> x >> y;
                if (label < 0) {
                    return false;
                }
            } else {
                iss >> hexKey >> x >> y;
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
};


class HexReader {

public:

    int32_t nUnits, nFeatures;
    HexGrid hexGrid;
    double hexSize;
    std::vector<std::string> features;

    HexReader(const std::string &metaFile) {
        readMetadata(metaFile);
    }
    bool featureDict(std::unordered_map<std::string, uint32_t>& dict) {
        if (features.empty()) {
            return false;
        }
        for (size_t i = 0; i < features.size(); ++i) {
            dict[features[i]] = i;
        }
        return true;
    }
    int32_t getNmodal() const {
        return nModal;
    }
    int32_t getNlayer() const {
        return nLayer;
    }
    void setFeatureNames(const std::vector<std::string>& featureNames) {
        assert (featureNames.size() == nFeatures && "Feature names size does not match the number of features");
        features = featureNames;
    }
    void setFeatureNames(const std::string& nameFile) {
        std::ifstream inFile(nameFile);
        if (!inFile) {
            error("Error opening feature names file: %s", nameFile.c_str());
        }
        std::string line;
        features.clear();
        while (std::getline(inFile, line)) {
            std::istringstream iss(line);
            std::string feature;
            if (!(iss >> feature)) {
                error("Error reading feature names on line %s", line.c_str());
            }
            features.push_back(feature);
        }
        inFile.close();
        assert (features.size() == nFeatures && "Feature names size does not match the number of features");
    }

    bool parseLine(UnitValues &unit, const std::string &line, bool labeled = false) {
        return unit.readFromLine(line, nModal, labeled);
    }
    int32_t parseLine(Document& doc, const std::string &line, int32_t modal = 0) {
        int32_t x, y, layer;
        return parseLine(doc, x, y, layer, line, modal);
    }
    int32_t parseLine(Document& doc, int32_t& x, int32_t& y, int32_t& layer, const std::string &line, int32_t modal = 0) {
        assert(modal < nModal && "Modal out of range");
        std::istringstream iss(line);
        std::string hexKey;
        std::vector<int32_t> nfeatures(nModal, 0);
        std::vector<uint32_t> counts(nModal, 0);
        layer = 0;
        try {
            if (nLayer > 1) {
                iss >> hexKey >> layer >> x >> y;
            } else {
                iss >> hexKey >> x >> y;
            }
            for (int i = 0; i < nModal; ++i) {
                iss >> nfeatures[i] >> counts[i];
            }
        } catch (const std::exception& e) {
            error("Error reading line: %s\n %s", line.c_str(), e.what());
        } catch (...) {
            error("Unknown error reading line: %s", line.c_str());
        }
        for (int l = 0; l < modal; ++l) { // skip
            for (int i = 0; i < nfeatures[l]; ++i) {
                uint32_t feature;
                int32_t value;
                if (!(iss >> feature >> value)) {
                    return -1;
                }
            }
        }
        doc.ids.resize(nfeatures[modal]);
        doc.cnts.resize(nfeatures[modal]);
        for (int i = 0; i < nfeatures[modal]; ++i) {
            if (!(iss >> doc.ids[i] >> doc.cnts[i])) {
                return -1;
            }
        }
        return counts[modal];
    }


private:

    int32_t nModal;
    int32_t nLayer;

    void readMetadata(const std::string &metaFile) {
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
        nModal = meta.value("n_modalities", 0);
        nFeatures = meta.value("n_features", 0);
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
    }
};
