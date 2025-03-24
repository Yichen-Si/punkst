#pragma once

#include "hexgrid.h"
#include "utils.h"
#include "json.hpp"
#include "assert.h"

struct Document {
    std::vector<uint32_t> ids; // Length: number of nonzero words in the doc
    std::vector<double> cnts;
};

struct PixelValues {
    double x, y;
    uint32_t feature;
    std::vector<int32_t> intvals;
    PixelValues() {}
};

struct UnitValues {
    int32_t nPixel;
    uint32_t totVal;
    int32_t x, y;
    std::vector<std::unordered_map<uint32_t, uint32_t>> vals;
    UnitValues(int32_t hx, int32_t hy) : nPixel(0), totVal(0), x(hx), y(hy) {}
    UnitValues(int32_t hx, int32_t hy, const PixelValues& pixel) : nPixel(1), totVal(0), x(hx), y(hy) {
        vals.resize(pixel.intvals.size());
        for (size_t i = 0; i < pixel.intvals.size(); ++i) {
            if (pixel.intvals[i] > 0) {
                vals[i][pixel.feature] = pixel.intvals[i];
                totVal += pixel.intvals[i];
            }
        }
    }
    void clear() {
        nPixel = 0;
        totVal = 0;
        for (auto& val : vals) {
            val.clear();
        }
    }
    void addPixel(const PixelValues& pixel) {
        ++nPixel;
        for (size_t i = 0; i < pixel.intvals.size(); ++i) {
            if (pixel.intvals[i] > 0)
                vals[i][pixel.feature] += pixel.intvals[i];
        }
    }
    bool mergeUnits(const UnitValues& other) {
        if (x != other.x || y != other.y) {
            return false;
        }
        nPixel += other.nPixel;
        for (size_t i = 0; i < vals.size(); ++i) {
            for (const auto& entry : other.vals[i]) {
                if (entry.second > 0)
                    vals[i][entry.first] += entry.second;
            }
        }
        return true;
    }
    bool writeToFile(std::ostream& os, uint32_t key) const {
        os << uint32toHex(key) << "\t" << x << "\t" << y;
        for (const auto& val : vals) {
            os << "\t" << val.size();
        }
        for (const auto& val : vals) {
            for (const auto& entry : val) {
                os << "\t" << entry.first << " " << entry.second;
            }
        }
        os << "\n";
        return os.good();
    }
    bool readFromLine(const std::string& line, int32_t nLayer) {
        std::istringstream iss(line);
        std::string hexKey;
        int32_t hx, hy;
        if (!(iss >> hexKey >> x >> y)) {
            return false;
        }
        clear();
        vals.resize(nLayer);
        std::vector<int32_t> nfeatures(nLayer, 0);
        for (int i = 0; i < nLayer; ++i) {
            if (!(iss >> nfeatures[i])) {
                return false;
            }
        }
        for (size_t i = 0; i < nLayer; ++i) {
            for (int j = 0; j < nfeatures[i]; ++j) {
                uint32_t feature;
                int32_t value;
                if (!(iss >> feature >> value)) {
                    return false;
                }
                vals[i][feature] = value;
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
    int32_t getNLayer() const {
        return nLayer;
    }

    bool parseLine(UnitValues &unit, const std::string &line) {
        return unit.readFromLine(line, nLayer);
    }
    bool parseLine(Document& doc, const std::string &line, int32_t layer = 0) {
        int32_t x, y;
        return parseLine(doc, x, y, line, layer);
    }
    bool parseLine(Document& doc, int32_t& x, int32_t& y, const std::string &line, int32_t layer = 0) {
        assert(layer < nLayer && "Layer out of range");
        std::istringstream iss(line);
        std::string hexKey;
        std::vector<int32_t> nfeatures(nLayer, 0);
        if (!(iss >> hexKey >> x >> y)) {
            return false;
        }
        for (int i = 0; i < nLayer; ++i) {
            if (!(iss >> nfeatures[i])) {
                return false;
            }
        }
        int l = 0;
        while (l < layer) {
            for (int i = 0; i < nfeatures[l]; ++i) {
                uint32_t feature;
                int32_t value;
                if (!(iss >> feature >> value)) {
                    return false;
                }
            }
            ++l;
        }
        doc.ids.resize(nfeatures[layer]);
        doc.cnts.resize(nfeatures[layer]);
        for (int i = 0; i < nfeatures[layer]; ++i) {
            if (!(iss >> doc.ids[i] >> doc.cnts[i])) {
                return false;
            }
        }
        return true;
    }


private:

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
        nLayer = meta.value("n_layers", 0);
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
