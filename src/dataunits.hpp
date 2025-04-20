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
    bool writeToFile(std::ostream& os, uint32_t key) const;
    bool readFromLine(const std::string& line, int32_t nModal, bool labeled = false);
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

    int32_t parseLine(Document& doc, const std::string &line, int32_t modal = 0) {
        int32_t x=0, y=0, layer=0;
        return parseLine(doc, x, y, layer, line, modal);
    }
    int32_t parseLine(Document& doc, int32_t& x, int32_t& y, int32_t& layer, const std::string &line, int32_t modal = 0);
    int32_t parseLine(UnitValues &unit, const std::string &line);


private:

    int32_t nModal;
    int32_t nLayer;
    int32_t offset_data;
    int32_t icol_layer, icol_x, icol_y;
    int32_t mintokens;

    void readMetadata(const std::string &metaFile);
};
