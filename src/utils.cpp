#include "utils.h"

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
    return lps;
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


int LocalAlignmentEditDistance(const std::string& str1, const std::string& str2, int32_t& st1, int32_t& ed1) {
    int m = str1.length();
    int n = str2.length();
    return LocalAlignmentEditDistance(str1.c_str(), str2.c_str(), m, n, st1, ed1);
}

int LocalAlignmentEditDistance(const char* str1, const char* str2, int32_t m, int32_t n, int32_t& st1, int32_t& ed1) {
    std::vector<std::vector<int>> dp(m + 1, std::vector<int>(n + 1, 0));
    std::vector<std::vector<int>> bt(m + 1, std::vector<int>(n + 1, 0));
    int minEditDistance = n + m;
    st1 = m; ed1 = 0;
    // Keep the first column zero to allow "free" start in str1
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
                if (dp[i][j] == sub) {
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
// std::cout << "align " << i << ": " << bt[i][n] << std::endl;
        }
    }
    // In the case of last base mismatch, prefer substitution over deletion
    // Avoid unnecessary shift
    if (ed1 < n && ed1 < m && dp[ed1][n] == dp[ed1+1][n]) {
        ed1++;
    }

    // backtrack to find the start position
    st1 = ed1; // 0-based inclusion
    int j = n;
    while(j > 0) {
        if (bt[st1][j] == 0 || bt[st1][j] == 1) {
            st1--;
            j--;
        } else if (bt[st1][j] == 2) {
            st1--; // up
        } else if (bt[st1][j] == 3) {
            j--; // left
        }
        if (st1 == 0) {
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
