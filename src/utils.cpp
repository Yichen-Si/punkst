#include "utils.h"

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

bool createDirectory(const std::string& dir) {
    if (std::filesystem::exists(dir)) {
        if (std::filesystem::is_directory(dir) && std::filesystem::is_empty(dir)) {
            return true;
        }
        return false;
    }
    std::filesystem::create_directories(dir);
    return true;
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
            error("Invalid color code %s", s_color);
            return false;
        }
        for (int32_t i = 0; i < 3; ++i) {
            bool valid_byte = str2int32(tokens[i], rgb[i]);
            if (!valid_byte || rgb[i] < 0 || rgb[i] > 255) {
                error("Invalid color code %s", s_color);
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
            error("Invalid color code %s", s_color);
            return false;
        }
        for (int32_t i = 0; i < 3; ++i) {
            bool valid_byte = str2int32(tokens[i], rgb[i]);
            if (!valid_byte || rgb[i] < 0 || rgb[i] > 255) {
                error("Invalid color code %s", s_color);
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

void computeBlocks(std::vector<std::pair<std::streampos, std::streampos>>& blocks, const std::string& inFile, int32_t nThreads, int32_t nskip) {
    std::ifstream infile(inFile, std::ios::binary);
    if (!infile) {
        error("Error opening input file: %s", inFile.c_str());
    }
    std::string line;
    for (int i = 0; i < nskip; ++i) {
        std::getline(infile, line);
    }
    std::streampos start_offset = infile.tellg();
    infile.seekg(0, std::ios::end);
    std::streampos fileSize = infile.tellg();
    size_t blockSize = fileSize / nThreads;

    std::streampos current = start_offset;
    blocks.clear();
    for (int i = 0; i < nThreads; ++i) {
        std::streampos end = current + static_cast<std::streamoff>(blockSize);
        if (end > fileSize || i == nThreads - 1) {
            end = fileSize;
        } else {
            infile.seekg(end);
            std::getline(infile, line);
            end = infile.tellg();
            if (end == -1) {
                end = fileSize;
            }
        }
        blocks.emplace_back(current, end);
        current = end;
        if (current >= fileSize) {
            break;
        }
    }
    infile.close();
    notice("Partitioned input file into %zu blocks of size ~ %zu", blocks.size(), blockSize);
}

bool checkOutputWritable(const std::string& outFile, bool newFile) {
    std::filesystem::path outPath(outFile);
    if (!newFile && std::filesystem::exists(outPath)) {
        if (!std::filesystem::is_regular_file(outPath)) {
            std::cerr << "Error: " << outFile << " is not a regular file." << std::endl;
            return false;
        }
        std::ofstream ofs(outFile, std::ios::app);
        if (!ofs) {
            std::cerr << "Error: Cannot open " << outFile << " for appending." << std::endl;
            return false;
        }
        ofs.close();
        return true;
    }
    if (outPath.has_parent_path()) {
        std::filesystem::path parent = outPath.parent_path();
        if (!std::filesystem::exists(parent)) {
            std::cerr << "Error: Output directory " << parent.string() << " does not exist." << std::endl;
            return false;
        }
        if (!std::filesystem::is_directory(parent)) {
            std::cerr << "Error: " << parent.string() << " is not a directory." << std::endl;
            return false;
        }
    }
    // Try opening the file for writing.
    std::ofstream ofs(outFile, std::ios::binary | std::ios::out);
    if (!ofs) {
        std::cerr << "Error: Cannot open " << outFile << " for writing." << std::endl;
        return false;
    }
    ofs.close();
    std::remove(outFile.c_str());
    return true;
}
