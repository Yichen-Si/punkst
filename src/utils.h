#pragma once

#include "utils_sys.hpp"
#include <tuple>
#include <functional>
#include <random>
#include <iomanip>
#include <sstream>
#include <string_view>
#include <chrono>
#include <optional>
#include <charconv>
#include <locale>
#include <stdexcept>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

// extern "C" {
//     #include "htslib/hts.h"
//     #include "htslib/bgzf.h"
//     #include "htslib/hfile.h"
// }

// // RAII wrapper for kstring_t: it ensures that the allocated buffer is freed automatically.
// struct KStringRAII {
//     kstring_t ks;
//     KStringRAII() : ks{0, 0, nullptr} {}
//     ~KStringRAII() { free(ks.s); }
// };

// void hprintf(htsFile* fp, const char * msg, ...);

// String manipulation functions
void split(std::vector<std::string>& vec, std::string_view delims, std::string_view str, uint32_t limit=UINT_MAX, bool clear=true, bool collapse=true, bool strip=false);
std::string_view strip_str(std::string_view token);
std::string trim(const std::string& str);
std::string to_lower(const std::string& str);
std::string to_upper(const std::string& str);
std::string join(const std::vector<std::string>& tokens, const std::string& delim);
static bool ends_with(const std::string &s, const std::string &suffix) {
    return s.size() >= suffix.size()
        && 0 == s.compare(s.size()-suffix.size(), suffix.size(), suffix);
}

// String to number conversion functions
template<typename T>
bool str2num(const std::string &str, T &value) {
    // Only arithmetic types supported
    static_assert(std::is_arithmetic_v<T>, "str2num only supports arithmetic types");

    if constexpr (std::is_integral_v<T>) {
        // integer parse via from_chars (no locale, fast, non-throwing)
        auto first = str.data(), last = str.data() + str.size();
        std::from_chars_result res = std::from_chars(first, last, value);
        // success only if no error AND entire string consumed
        return res.ec == std::errc() && res.ptr == last;
    }
    else if constexpr (std::is_floating_point_v<T>) {
        // floating-point parse via C functions (sets errno on under/overflow)
        errno = 0;
        const char* cstr = str.c_str();
        char* end = nullptr;

        // pick the right function for T
        long double tmp_ld = 0;
        if constexpr (std::is_same_v<T, float>) {
            tmp_ld = std::strtof(cstr, &end);
        }
        else if constexpr (std::is_same_v<T, double>) {
            tmp_ld = std::strtod(cstr, &end);
        }
        else { // long double
            tmp_ld = std::strtold(cstr, &end);
        }

        // no characters parsed?
        if (end == cstr)
            return false;
        // extra junk after number?
        if (*end != '\0')
            return false;

        // under/overflow?
        if (errno == ERANGE) {
            if (tmp_ld < 1e-8) {
                // underflow → treat as zero
                value = T(0);
                errno = 0;
                return true;
            }
            // overflow → fail
            return false;
        }
        // normal success
        value = static_cast<T>(tmp_ld);
        return true;
    }
    // unreachable, static_assert above will trap non-arithmetic
    return false;
}

bool str2int32(const std::string& str, int32_t& value);
bool str2int64(const std::string& str, int64_t& value);
bool str2uint32(const std::string& str, uint32_t& value);
bool str2uint64(const std::string& str, uint64_t& value);
bool str2double(const std::string& str, double& value);
bool str2float(const std::string& str, float& value);
bool str2bool(const std::string& str, bool& value);

// Float to string
template<typename FP>
std::string fp_to_string(FP x, int digits)
{
    char buf[64];
#if defined(__cpp_lib_to_chars) && __cpp_lib_to_chars >= 201611L
    auto [ptr, ec] =
        std::to_chars(buf, buf + sizeof buf,
                      x, std::chars_format::fixed, digits);
    if (ec == std::errc()) return {buf, ptr};
#endif
    std::snprintf(buf, sizeof buf, "%.*f", digits, static_cast<double>(x));
    return buf;
}

// Write matrix to tsv
template <typename Scalar>
void write_matrix_to_file(const std::string& output,
        const Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>& mat,
        int digits, bool scientific,
        const std::vector<std::string>& rnames,
        const std::string& c0name) {
    if (mat.rows() != static_cast<Eigen::Index>(rnames.size())) {
        throw std::runtime_error("Dimension mismatch: Matrix rows and row names count differ.");
    }
    std::ofstream ofs(output);
    if (!ofs.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + output);
    }
    const static Eigen::IOFormat TSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, "\t", "");
    ofs << c0name;
    for (int k = 0; k < mat.cols(); ++k) {
        ofs << "\t" << k;
    }
    ofs << "\n";
    ofs << std::setprecision(digits);
    if (scientific) {
        ofs << std::scientific;
    } else {
        ofs << std::fixed;
    }
    for (int i = 0; i < mat.rows(); ++i) {
        ofs << rnames[i] << "\t" << mat.row(i).format(TSVFormat) << "\n";
    }
}

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
// hash a tuple of three integers
struct Tuple3Hash {
    std::size_t operator()(const std::tuple<int32_t, int32_t, int32_t>& key) const {
        // Get individual hash values for each element.
        auto h1 = std::hash<int32_t>{}(std::get<0>(key));
        auto h2 = std::hash<int32_t>{}(std::get<1>(key));
        auto h3 = std::hash<int32_t>{}(std::get<2>(key));

        // Combine them using a hash combining formula.
        std::size_t seed = h1;
        seed ^= h2 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        seed ^= h3 + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2);
        return seed;
    }
};

// compute percentiles (results are sorted)
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

// compute percentiles of non-zero values in a cv::Mat
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
        return (x >= xmin && x < xmax && y >= ymin && y < ymax);
    }
    int32_t intersect(const Rectangle<T>& other) const {
        if (other.xmin >= xmax || other.xmax <= xmin || other.ymin >= ymax || other.ymax <= ymin) {
            return 0; // no intersection
        }
        if (other.xmin <= xmin && other.xmax >= xmax && other.ymin <= ymin && other.ymax >= ymax) {
            return 2; // the other rectangle fully contains this one
        }
        if (other.xmin >= xmin && other.xmax <= xmax && other.ymin >= ymin && other.ymax <= ymax) {
            return 3; // this rectangle fully contains the other one
        }
        return 1; // partial intersection
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

template <typename T>
int32_t parseCoordsToRects(std::vector<Rectangle<T>>& rects, const std::vector<T>& coords) {
    if (coords.size() % 4 != 0) {
        return -1;
    }
    for (size_t i = 0; i < coords.size() / 4; ++i) {
        T x1 = coords[i * 4];     // xmin
        T y1 = coords[i * 4 + 1]; // ymin
        T x2 = coords[i * 4 + 2]; // xmax
        T y2 = coords[i * 4 + 3]; // ymax
        Rectangle<T> rect(x1, y1, x2, y2);
        if (!rect.proper()) {
            warning("Invalid bounding box: %f %f %f %f", x1, y1, x2, y2);
            continue;
        }
        rects.push_back(rect);
    }
    return rects.size();
}

// Centroid of a polygon by triangulation and weighted average
cv::Point2d centroidOfPolygonTriangulation(const std::vector<cv::Point2d>& poly);
cv::Point2f centroidOfPolygonRobust(const std::vector<cv::Point2f>& polyf);

// Centroid of a polygon using the shoelace formula
template <typename T>
cv::Point_<T> centroidOfPolygon(const std::vector<cv::Point_<T>>& poly) {
    assert(!poly.empty());
    double area = 0.0, cx = 0.0, cy = 0.0;
    uint32_t n = poly.size();
    // Compute the polygon centroid using the shoelace formula.
    for (size_t j = 0; j < poly.size(); j++) {
        const cv::Point_<T>& p0 = poly[j];
        const cv::Point_<T>& p1 = poly[(j + 1) % n];
        double cross = (double) p0.x * p1.y - (double) p1.x * p0.y;
        area += cross;
        cx += (p0.x + p1.x) * cross;
        cy += (p0.y + p1.y) * cross;
    }
    area *= 0.5;
    if (std::fabs(area) > 1e-6) {
        cx /= (6 * area);
        cy /= (6 * area);
        return cv::Point_<T>(static_cast<T>(cx), static_cast<T>(cy));
    }
    return poly[0];
}

// Sutherland–Hodgman polygon clipping algorithm
std::vector<cv::Point2f> clipPolygonToRect(const std::vector<cv::Point2f>& poly, const cv::Rect2f& rect);
