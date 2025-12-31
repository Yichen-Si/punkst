#pragma once

#include "hexgrid.h"
#include "utils.h"
#include "json.hpp"
#include "assert.h"
#include <unordered_set>
#include "Eigen/Dense"
#include "Eigen/Sparse"

struct Document {
    std::vector<uint32_t> ids; // Length: number of nonzero words in the doc
    std::vector<double> cnts;
    double ct_tot = -1;

    inline Eigen::VectorXd to_dense(size_t n) const {
        Eigen::VectorXd y_dense = Eigen::VectorXd::Zero(n);
        for (size_t t = 0; t < ids.size(); ++t) {
            y_dense[ids[t]] = cnts[t];
        }
        return y_dense;
    }
    double get_sum() {
        if (ct_tot < 0) {
            ct_tot = std::accumulate(cnts.begin(), cnts.end(), 0.0);
        }
        return ct_tot;
    }
};

struct SparseObs {
    Document doc;
    double c;
    Eigen::VectorXd covar; // covariates
    double ct_tot = -1;

    double get_sum() {
        if (ct_tot < 0) {
            ct_tot = 0;
            for (double v : doc.cnts) ct_tot += v;
        }
        return ct_tot;
    }
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

struct PixelValues3D {
    double x, y, z;
    uint32_t feature;
    std::vector<int32_t> intvals;
    PixelValues3D() {}
    bool writeToFileText(std::ostream& os) const {
        os << x << "\t" << y << "\t" << z << "\t" << feature;
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
    std::vector<std::map<uint32_t, uint32_t>> vals;
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

struct UnitValues3D {
    int32_t nPixel;
    int32_t x, y, z;
    int32_t label;
    std::vector<std::map<uint32_t, uint32_t>> vals;
    std::vector<uint32_t> valsums;
    UnitValues3D(int32_t hx, int32_t hy, int32_t hz, int32_t n = 0, int32_t l = -1) : nPixel(0), x(hx), y(hy), z(hz), label(l) {
        if (n > 0) {
            vals.resize(n);
            valsums.resize(n);
            std::fill(valsums.begin(), valsums.end(), 0);
        }
    }
    UnitValues3D(int32_t hx, int32_t hy, int32_t hz, const PixelValues3D& pixel, int32_t l = -1)
        : nPixel(1), x(hx), y(hy), z(hz), label(l) {
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
    void addPixel(const PixelValues3D& pixel) {
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
    bool mergeUnits(const UnitValues3D& other) {
        if (x != other.x || y != other.y || z != other.z) {
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
    double hexSize;
    bool hasCoordinates;
    std::vector<std::string> features;
    bool readFullSums = false;

    HexReader() = default;
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
    int32_t getOffset() const {
        return offset_data;
    }
    void getInfoHeaderStr(std::string &header) const {
        header.clear();
        for (size_t i = 0; i < header_info.size(); ++i) {
            if (i > 0) {
                header += "\t";
            }
            header += header_info[i];
        }
    }
    int32_t getIndex(const std::string &colname) const {
        for (size_t i = 0; i < header_info.size(); ++i) {
            if (header_info[i] == colname) {
                return static_cast<int32_t>(i);
            }
        }
        return -1; // Not found
    }
    const std::vector<double>& getFeatureSums() const {
        return feature_sums;
    }

    void setAccumulationStatus(bool v) { accumulate_sums = v; }
    void setFeatureIndexRemap(std::unordered_map<uint32_t, uint32_t>& _idx_remap);
    void setFeatureIndexRemap(std::vector<std::string>& new_features, bool keep_unmapped = false);
    void setFeatureFilter(const std::string& featureFile, int32_t minCount, std::string& include_ftr_regex, std::string& exclude_ftr_regex, bool read_sums = true);
    void setWeights(const std::string& weightFile, double defaultWeight_ = 1.0);

    int32_t parseLine(Document& doc, const std::string &line, int32_t modal = 0, bool add2sums = true) {
        std::string info;
        return parseLine(doc, info, line, modal, add2sums);
    }
    int32_t parseLine(Document& doc, std::string &info, const std::string &line, int32_t modal = 0, bool add2sums = true);

    int32_t readAll(std::vector<Document>& docs,
        std::vector<std::string>& info, const std::string &inFile,
        int32_t minCount = 1, int32_t modal = 0, bool add2sums = true);
    int32_t readAll(std::vector<Document>& docs, const std::string &inFile,
        int32_t minCount = 1, int32_t modal = 0, bool add2sums = true);
    int32_t readAll(Eigen::SparseMatrix<double, Eigen::RowMajor> &X,
        std::vector<std::string>& info, const std::string &inFile,
        int32_t minCount = 1, int32_t modal = 0, bool add2sums = true);

private:

    int32_t nModal;
    int32_t nLayer;
    int32_t offset_data;
    int32_t icol_layer, icol_x, icol_y;
    int32_t mintokens;
    std::vector<double> feature_sums;
    std::vector<double> weights;
    std::vector<std::string> header_info;
    std::unordered_map<uint32_t, uint32_t> idx_remap;
    bool remap = false;
    bool accumulate_sums = true;
    bool weightFeatures = false;
    double defaultWeight;

    void readMetadata(const std::string &metaFile);
};

int32_t read_sparse_obs(const std::string &inFile, HexReader &reader,
    std::vector<SparseObs> &docs, std::vector<std::string> &rnames,
    int32_t minCountTrain = 1, double size_factor = 10000, double c = -1,
    std::string* covarFile = nullptr, std::vector<uint32_t>* covar_idx = nullptr,
    std::vector<std::string>* covar_names = nullptr, bool allow_na = false,
    int32_t label_idx = -1, std::vector<std::string>* labels = nullptr,
    int32_t debug_N = 0);

template<typename T>
void readCoordRange(const std::string& rangeFile, T& xmin, T& xmax, T& ymin, T& ymax) {
    std::ifstream in(rangeFile);
    if (!in.is_open())
        error("Cannot open range file: %s", rangeFile.c_str());
    std::array<bool,4> seen = {false, false, false, false};
    std::string key;
    T value;
    while (in >> key >> value) {
        if      (key == "xmin") { xmin = value; seen[0] = true; }
        else if (key == "xmax") { xmax = value; seen[1] = true; }
        else if (key == "ymin") { ymin = value; seen[2] = true; }
        else if (key == "ymax") { ymax = value; seen[3] = true; }
        else {
            std::cerr << "Warning: unrecognized key '" << key << "' in "
                      << rangeFile << "\n";
        }
    }
    // Verify that we found them all
    static constexpr const char* names[4] = {"xmin","xmax","ymin","ymax"};
    for (int i = 0; i < 4; ++i) {
        if (!seen[i]) {
            error("Missing %s in range file: %s", names[i], rangeFile.c_str());
        }
    }
}
