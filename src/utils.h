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
#include "qgenlib/qgen_error.h"
#include "qgenlib/qgen_utils.h"

std::vector<int> computeLPSArray(const std::string& pattern);
int32_t KMPSearch(const std::string& pattern, const std::string& text, std::vector<int>& idx, std::vector<int>& lps, int32_t n = 1);

int LocalAlignmentEditDistance(const std::string& str1, const std::string& str2, int32_t& st1, int32_t& ed1);
int LocalAlignmentEditDistance(const char* str1, const char* str2, int32_t m, int32_t n, int32_t& st1, int32_t& ed1);

bool set_rgb(const char *s_color, std::vector<int32_t>& rgb);
bool set_rgb(const char *s_color, std::array<int32_t, 3>& rgb);

#endif
