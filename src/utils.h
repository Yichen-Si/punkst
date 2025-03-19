#ifndef UTILS_H_
#define UTILS_H_

#include <cstdio>
#include <climits>
#include <cstddef>
#include <vector>
#include <string>
#include <set>
#include <map>
#include <iostream>
#include <algorithm>
#include <filesystem>
#include "qgenlib/qgen_error.h"
#include "qgenlib/qgen_utils.h"
#include <opencv2/opencv.hpp>

// String search
std::vector<int> computeLPSArray(const std::string& pattern);
int32_t KMPSearch(const std::string& pattern, const std::string& text, std::vector<int>& idx, std::vector<int>& lps, int32_t n = 1);

// Align str1 (local) to str2 (global)
int LocalAlignmentEditDistance(const std::string& str1, const std::string& str2, int32_t& st1, int32_t& ed1, int32_t& st1_match);
int LocalAlignmentEditDistance(const char* str1, const char* str2, int32_t m, int32_t n, int32_t& st1, int32_t& ed1, int32_t& st1_match);

// Hex color code to integer RGB
bool set_rgb(const char *s_color, std::vector<int32_t>& rgb);
bool set_rgb(const char *s_color, std::array<int32_t, 3>& rgb);

// base16 encoding
std::string uint32toHex(uint32_t num);
uint32_t hexToUint32(const std::string& hex);

bool createDirectory(const std::string& dir);

// Results are sorted
template <typename T>
void compute_percentile(std::vector<T>& results, std::vector<T>& values, std::vector<double>& percentiles) {
    size_t n = values.size();
    size_t m = percentiles.size();
    for (size_t i = 0; i < m; ++i) {
        if (percentiles[i] < 0 || percentiles[i] > 1) {
            error("Percentile value must be between 0 and 1");
        }
    }
    results.resize(m);
    if (m > log(n)) {
        std::sort(values.begin(), values.end());
        for (size_t i = 0; i < m; ++i) {
            size_t idx = std::ceil(percentiles[i] * n);
            if (idx >= n) {
                idx = n - 1;
            }
            results[i] = values[idx];
        }
        return;
    }
    std::vector<size_t> idx(m);
    for (size_t i = 0; i < m; ++i) {
        idx[i] = std::ceil(percentiles[i] * n);
        if (idx[i] >= n) {
            idx[i] = n - 1;
        }
    }
    std::sort(idx.begin(), idx.end());
    size_t st = 0;
    for (size_t i = 0; i < m; ++i) {
        if (st > idx[i]) {
            results[i] = results[i-1];
            st = idx[i] + 1;
            continue;
        }
        std::nth_element(values.begin() + st, values.begin() + idx[i], values.end());
        results[i] = values[idx[i]];
        st = idx[i] + 1;
    }
}

// Image related

// Helper function to compute percentile of non-zero values in a cv::Mat
void percentile(std::vector<uchar>& results, const cv::Mat& mat, std::vector<double>& percentiles);

// Shape related
template <typename T>
struct Rectangle {
    T xmin, ymin, xmax, ymax;
    Rectangle(T x1, T y1, T x2, T y2) : xmin(x1), ymin(y1), xmax(x2), ymax(y2) {}
    Rectangle() : xmin(0), ymin(0), xmax(0), ymax(0) {}
    bool proper() const {
        return (xmin < xmax && ymin < ymax);
    }
    bool contains(T x, T y) const {
        return (x >= xmin && x <= xmax && y >= ymin && y <= ymax);
    }
    bool cutInside(Rectangle<T>& rec, T r) {
        rec.xmin = xmin + r;
        rec.ymin = ymin + r;
        rec.xmax = xmax - r;
        rec.ymax = ymax - r;
        return rec.proper();
    }
    bool padOutside(Rectangle<T>& rec, T r) {
        rec.xmin = xmin - r;
        rec.ymin = ymin - r;
        rec.xmax = xmax + r;
        rec.ymax = ymax + r;
        return rec.proper();
    }
};

#endif
