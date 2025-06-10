#include "utils.h"

// void hprintf(htsFile* fp, const char * msg, ...) {
//     va_list ap;
//     va_start(ap, msg);

//     // Use RAII to manage kstring_t memory.
//     KStringRAII tmp;
//     kvsprintf(&tmp.ks, msg, ap);
//     va_end(ap);

//     int ret;
//     if (auto comp = fp->format.compression; comp != no_compression)
//         ret = bgzf_write(fp->fp.bgzf, tmp.ks.s, tmp.ks.l);
//     else
//         ret = hwrite(fp->fp.hfile, tmp.ks.s, tmp.ks.l);

//     if (ret < 0) {
//         error("[E:%s:%d %s] [E:%s:%d %s] hprintf failed. Aborting..",
//               __FILE__, __LINE__, __FUNCTION__,
//               __FILE__, __LINE__, __FUNCTION__);
//     }
// }

// String manipulation functions

// Splits a line into a vector - one or more single character delimiters
void split(std::vector<std::string>& vec, std::string_view delims, std::string_view str,
    uint32_t limit, bool clear, bool collapse, bool strip) {
    if (clear)
        vec.clear();

    uint32_t tokenCount = 0;
    size_t start = 0;
    while (start < str.size() && tokenCount < limit - 1) {
        size_t pos = str.find_first_of(delims, start);
        if (pos == std::string_view::npos)
            pos = str.size();

        // Get the current token as a view
        std::string_view token = str.substr(start, pos - start);
        if (strip)
            token = strip_str(token);

        // Only add token if not collapsing empty tokens
        if (!collapse || !token.empty()) {
            vec.emplace_back(token);
            ++tokenCount;
        }

        start = pos;
        // Skip delimiter if found
        if (start < str.size() && delims.find(str[start]) != std::string_view::npos)
            ++start;
    }

    // Add the remaining part if any.
    if (start <= str.size()) {
        std::string_view token = str.substr(start);
        if (strip)
            token = strip_str(token);
        if (!collapse || !token.empty())
            vec.emplace_back(token);
    }
}

std::string_view strip_str(std::string_view token) {
    // Implement trimming logic if needed.
    // For example, remove whitespace from both ends.
    size_t start = token.find_first_not_of(" \t\n\r");
    size_t end = token.find_last_not_of(" \t\n\r");
    if (start == std::string_view::npos)
        return {};
    return token.substr(start, end - start + 1);
}

std::string trim(const std::string& str) {
    auto start = std::find_if_not(str.begin(), str.end(), [](unsigned char c) {
        return std::isspace(c);
    });

    auto end = std::find_if_not(str.rbegin(), str.rend(), [](unsigned char c) {
        return std::isspace(c);
    }).base();

    return (start < end) ? std::string(start, end) : std::string();
}

std::string to_lower(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::tolower(c); });
    return result;
}

std::string to_upper(const std::string& str) {
    std::string result = str;
    std::transform(result.begin(), result.end(), result.begin(),
                   [](unsigned char c){ return std::toupper(c); });
    return result;
}

std::string join(const std::vector<std::string>& tokens, const std::string& delim) {
    if (tokens.empty()) return "";

    std::ostringstream result;
    result << tokens[0];

    for (size_t i = 1; i < tokens.size(); ++i) {
        result << delim << tokens[i];
    }

    return result.str();
}

// String to number conversion shortcuts
bool str2int32(const std::string& str, int32_t& value) {
    return str2num<int32_t>(str, value);
}
bool str2int64(const std::string& str, int64_t& value) {
    return str2num<int64_t>(str, value);
}
bool str2uint32(const std::string& str, uint32_t& value) {
    return str2num<uint32_t>(str, value);
}
bool str2uint64(const std::string& str, uint64_t& value) {
    return str2num<uint64_t>(str, value);
}
bool str2double(const std::string& str, double& value) {
    return str2num<double>(str, value);
}
bool str2float(const std::string& str, float& value) {
    return str2num<float>(str, value);
}

bool str2bool(const std::string& str, bool& value) {
    std::string lower = str;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    if (lower == "true" || lower == "yes" || lower == "1" || lower == "y" || lower == "t") {
        value = true;
        return true;
    } else if (lower == "false" || lower == "no" || lower == "0" || lower == "n" || lower == "f") {
        value = false;
        return true;
    }

    return false;
}

std::string uint32toHex(uint32_t num) {
    std::stringstream ss;
    ss << std::hex << std::setw(8) << std::setfill('0') << num;
    return ss.str();
}

uint32_t hexToUint32(const std::string& hex) {
    uint32_t num;
    std::stringstream ss;
    ss << std::hex << hex;
    ss >> num;
    return num;
}

std::vector<int> computeLPSArray(const std::string& pattern) {
    int M = pattern.size();
    std::vector<int> lps(M, 0);
    int len = 0;
    int i = 1;
    while (i < M) {
        if (pattern[i] == pattern[len]) {
            len++;
            lps[i] = len;
            i++;
        } else {
            if (len != 0) {
                len = lps[len - 1];
            } else {
                lps[i] = 0;
                i++;
            }
        }
    }
    return std::move(lps);
}

int32_t KMPSearch(const std::string& pattern, const std::string& text, std::vector<int>& idx, std::vector<int>& lps, int32_t n) {
    int M = pattern.size();
    int N = text.size();
    int c = 0;
    int i = 0;
    int j = 0;
    while (i < N) {
        if (pattern[j] == text[i]) {
            j++;
            i++;
        }
        if (j == M) {
            idx.push_back(i - j);
            j = lps[j - 1];
            c += 1;
            if (c >= n)
                return c;
        } else if (i < N && pattern[j] != text[i]) {
            if (j != 0) {
                j = lps[j - 1];
            } else {
                i = i + 1;
            }
        }
    }
    return c;
}


int LocalAlignmentEditDistance(const std::string& str1, const std::string& str2, int32_t& st1, int32_t& ed1, int32_t& st1_match) {
    int m = str1.length();
    int n = str2.length();
    return LocalAlignmentEditDistance(str1.c_str(), str2.c_str(), m, n, st1, ed1, st1_match);
}

/**
 * Simple local alignment of str1 to (the full) reference str2
*/
int LocalAlignmentEditDistance(const char* str1, const char* str2, int32_t m, int32_t n, int32_t& st1, int32_t& ed1, int32_t& st1_match) {
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    std::vector<std::vector<int>> bt(m + 1, std::vector<int>(n + 1, 0));
    int minEditDistance = n + m; // min of the last column in dp
    st1 = m; ed1 = 0;
    // Keep the first column zero to allow "free" start in str1
    // Populate the first row with deletion cost
    for (int j = 0; j <= n; j++) dp[0][j] = j;

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            if (str1[i - 1] == str2[j - 1]) {
                dp[i][j] = dp[i - 1][j - 1];
                bt[i][j] = 0;
            } else {
                int ins = dp[i-1][j] + 1; // insertion in str1, down
                int del = dp[i][j-1] + 1; // deletion in str1, right
                int sub = dp[i-1][j-1] + 1;
                dp[i][j] = std::min({del, ins, sub});
                if (dp[i][j] == sub) { // prefer substitution?
                    bt[i][j] = 1;
                } else if (dp[i][j] == ins) {
                    bt[i][j] = 2; // insertion in str1
                } else {
                    bt[i][j] = 3;
                }
            }
        }
        if (dp[i][n] < minEditDistance) {
            minEditDistance = dp[i][n];
            ed1 = i; // 0-based exclusion
        }
    }

    // backtrack to find the start position
    st1 = ed1; // 0-based inclusion
    int j = n;
    std::vector<int32_t> path;
    path.reserve(m + n);
    while(j > 0 & st1 > 0) {
        path.push_back(bt[st1][j]);
        if (bt[st1][j] == 0 || bt[st1][j] == 1) {
            st1--;
            j--;
        } else if (bt[st1][j] == 2) {
            st1--; // up
        } else if (bt[st1][j] == 3) {
            j--; // left
        }
    }
    st1_match = st1;
    std::reverse(path.begin(), path.end());
    for (int i = 0; i < path.size(); i++) {
        if (path[i] == 1) {
            st1_match++;
        } else {
            break;
        }
    }
    return minEditDistance;
}

bool set_rgb(const char *s_color, std::vector<int32_t>& rgb) {
    rgb.resize(3);
    if (s_color[0] == '#' && strlen(s_color) == 7) {
        sscanf(s_color + 1, "%02x%02x%02x", &rgb[0], &rgb[1], &rgb[2]);
        return true;
    }
    else if ( strlen(s_color) == 6 ) {
        sscanf(s_color, "%02x%02x%02x", &rgb[0], &rgb[1], &rgb[2]);
        return true;
    } else {
        std::vector<std::string> tokens;
        split(tokens, ",", s_color);
        if (tokens.size() != 3) {
            warning("Invalid color code %s", s_color);
            return false;
        }
        for (int32_t i = 0; i < 3; ++i) {
            bool valid_byte = str2int32(tokens[i], rgb[i]);
            if (!valid_byte || rgb[i] < 0 || rgb[i] > 255) {
                warning("Invalid color code %s", s_color);
                return false;
            }
        }
        return true;
    }
}

bool set_rgb(const char *s_color, std::array<int32_t, 3>& rgb) {
    if (s_color[0] == '#' && strlen(s_color) == 7) {
        sscanf(s_color + 1, "%02x%02x%02x", &rgb[0], &rgb[1], &rgb[2]);
        return true;
    }
    else if ( strlen(s_color) == 6 ) {
        sscanf(s_color, "%02x%02x%02x", &rgb[0], &rgb[1], &rgb[2]);
        return true;
    } else {
        std::vector<std::string> tokens;
        split(tokens, ",", s_color);
        if (tokens.size() != 3) {
            warning("Invalid color code %s", s_color);
            return false;
        }
        for (int32_t i = 0; i < 3; ++i) {
            bool valid_byte = str2int32(tokens[i], rgb[i]);
            if (!valid_byte || rgb[i] < 0 || rgb[i] > 255) {
                warning("Invalid color code %s", s_color);
                return false;
            }
        }
        return true;
    }
}

// Helper function to compute percentile of non-zero values in a cv::Mat
void percentile(std::vector<uchar>& results, const cv::Mat& mat, std::vector<double>& percentiles) {
    std::vector<uchar> values;
    values.reserve(mat.rows * mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            uchar val = mat.at<uchar>(i, j);
            if (val > 0) {
                values.push_back(val);
            }
        }
    }
    compute_percentile<uchar>(results, values, percentiles);
}

cv::Point2d centroidOfPolygonTriangulation(const std::vector<cv::Point2d>& poly) {
    // For a polygon with fewer than 3 points, return the first point.
    if (poly.size() < 3)
        return poly.front();

    cv::Point2d ref = poly[0];
    double totalArea = 0.0;
    cv::Point2d centroidSum(0.0, 0.0);
    // Triangulate the polygon by decomposing it into triangles (ref, poly[i], poly[i+1]).
    for (size_t i = 1; i < poly.size() - 1; i++) {
        // Compute cross product to get twice the triangle area.
        double cross = (poly[i].x - ref.x) * (poly[i+1].y - ref.y) - (poly[i].y - ref.y) * (poly[i+1].x - ref.x);
        double triArea = cross / 2.0;
        totalArea += triArea;
        // The centroid of a triangle is the average of its vertices.
        cv::Point2d triCentroid = (ref + poly[i] + poly[i+1]) / 3.0;
        centroidSum += triArea * triCentroid;
    }
    // In case of near-degeneracy, fall back to the reference.
    if (std::fabs(totalArea) < 1e-6)
        return ref;

    return centroidSum / totalArea;
}

cv::Point2f centroidOfPolygonRobust(const std::vector<cv::Point2f>& polyf) {
    std::vector<cv::Point2d> poly;
    for (const auto& p : polyf)
        poly.push_back(cv::Point2d(p.x, p.y));
    cv::Point2d c = centroidOfPolygonTriangulation(poly);
    return cv::Point2f(static_cast<float>(c.x), static_cast<float>(c.y));
}

std::vector<cv::Point2f> clipPolygonToRect(const std::vector<cv::Point2f>& poly, const cv::Rect2f& rect) {
    // We assume poly is already in a proper (clockwise) order.
    std::vector<cv::Point2f> output = poly;

    // Define bounds.
    float xmin = rect.x;
    float ymin = rect.y;
    float xmax = rect.x + rect.width;
    float ymax = rect.y + rect.height;

    // Local lambda to clip against one edge.
    // 'inside' returns true if a point is on the interior side of the edge.
    // 'computeIntersection' computes the intersection point of the subject edge and the clipping edge.
    auto clipEdge = [&](auto inside, auto computeIntersection, const std::string &edgeName) {
        std::vector<cv::Point2f> input = output;
        output.clear();
        if (input.empty()) return;
        cv::Point2f S = input.back();
        for (const auto &P : input) {
            bool sInside = inside(S);
            bool pInside = inside(P);
            if (pInside) {
                if (!sInside) {
                    cv::Point2f ip = computeIntersection(S, P);
                    output.push_back(ip);
                }
                output.push_back(P);
            }
            else if (sInside) {
                cv::Point2f ip = computeIntersection(S, P);
                output.push_back(ip);
            }
            S = P;
        }
    };

    // Process edges in the proper clockwise order.
    // Top edge: from (xmin, ymin) to (xmax, ymin)
    clipEdge(
        [&](const cv::Point2f &p) { return p.y >= ymin; },
        [&](const cv::Point2f &S, const cv::Point2f &P) -> cv::Point2f {
            float t = (ymin - S.y) / (P.y - S.y);
            return cv::Point2f(S.x + t * (P.x - S.x), ymin);
        },
        "Top"
    );

    // Right edge: from (xmax, ymin) to (xmax, ymax)
    clipEdge(
        [&](const cv::Point2f &p) { return p.x <= xmax; },
        [&](const cv::Point2f &S, const cv::Point2f &P) -> cv::Point2f {
            float t = (xmax - S.x) / (P.x - S.x);
            return cv::Point2f(xmax, S.y + t * (P.y - S.y));
        },
        "Right"
    );

    // Bottom edge: from (xmax, ymax) to (xmin, ymax)
    clipEdge(
        [&](const cv::Point2f &p) { return p.y <= ymax; },
        [&](const cv::Point2f &S, const cv::Point2f &P) -> cv::Point2f {
            float t = (ymax - S.y) / (P.y - S.y);
            return cv::Point2f(S.x + t * (P.x - S.x), ymax);
        },
        "Bottom"
    );

    // Left edge: from (xmin, ymax) to (xmin, ymin)
    clipEdge(
        [&](const cv::Point2f &p) { return p.x >= xmin; },
        [&](const cv::Point2f &S, const cv::Point2f &P) -> cv::Point2f {
            float t = (xmin - S.x) / (P.x - S.x);
            return cv::Point2f(xmin, S.y + t * (P.y - S.y));
        },
        "Left"
    );
    return output;
}
