#include "punkst.h"
#include "utils.h"
#include "dataunits.hpp"
#include "nanoflann.hpp"
#include "nanoflann_utils.h"
#include <opencv2/opencv.hpp>

#include <array>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <sstream>
#include <cctype>
#include <climits>

int32_t cmdDrawLowresFactors(int32_t argc, char** argv) {
    std::string dataFile, coordFile, rangeFile, colorFile, outFile;
    std::vector<std::string> channelListStr, colorListStr;
    double scale = 1.0;
    double xmin = 0, xmax = -1, ymin = 0, ymax = -1;
    double radius = -1.0;
    int32_t icol_x = -1, icol_y = -1, icol_idx = 0, offset = 1;
    int32_t K;
    std::vector<uint32_t> topk, topp;
    int32_t verbose = 1000000;
    bool top_only = false;

    ParamList pl;
    pl.add_option("in-tsv", "Input TSV file with x y then factor probabilities", dataFile, true)
      .add_option("K", "Number of factors", K, true)
      .add_option("icol-idx", "Column index for each unit's index in the coordinate file (0-based)", icol_idx)
      .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x)
      .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y)
      .add_option("offset", "Column index offset for factor probabilities (default: 1)", offset)
      .add_option("icol-topk", "Column indices for top factor indices (0-based)", topk)
      .add_option("icol-topp", "Column indices for top factor probabilities (0-based)", topp)
      .add_option("in-coord", "Input coordinate file (x y)", coordFile)
      .add_option("in-color", "Input color file (RGB triples)", colorFile)
      .add_option("scale", "Scale factor: (x-xmin)/scale → pixel_x", scale)
      .add_option("range", "A file containing coordinate range (xmin ymin xmax ymax)", rangeFile)
      .add_option("xmin", "Minimum x coordinate", xmin)
      .add_option("xmax", "Maximum x coordinate", xmax)
      .add_option("ymin", "Minimum y coordinate", ymin)
      .add_option("ymax", "Maximum y coordinate", ymax)
      .add_option("radius", "Maximum distance from a pixel to an anchor", radius)
      .add_option("channel-list", "A list of factor IDs/names to draw", channelListStr)
      .add_option("color-list", "A list of colors for factors in hex code (#RRGGBB)", colorListStr);
    pl.add_option("out", "Output image file", outFile, true)
      .add_option("top-only", "Paint only according to the top factor", top_only)
      .add_option("verbose", "Verbose", verbose);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    // Check input parameters
    if (coordFile.empty() && (icol_x < 0 || icol_y < 0)) {
        error("Either --in-coord or both --icol-x and --icol-y must be provided");
    }
    if (!checkOutputWritable(outFile))
        error("Output file is not writable: %s", outFile.c_str());

    if (!rangeFile.empty()) {
        readCoordRange(rangeFile, xmin, xmax, ymin, ymax);
    }
    if (xmin >= xmax || ymin >= ymax)
        error("Invalid range: xmin >= xmax or ymin >= ymax");
    if (scale <= 0)
        error("--scale must be >0");
    if (radius <= 0)
        error("--radius must be >0");
    bool selected = !channelListStr.empty();
    if (!selected && colorFile.empty())
        error("Either --in-color or both --channel-list and --color-list must be provided");
    bool use_topk = !topk.empty() && !topp.empty();
    if (use_topk && (topk.size() != topp.size())) {
        error("--icol-topk and --icol-topp must have the same length");
    }

    std::ifstream dataStream(dataFile);
    if (!dataStream.is_open())
        error("Error opening data file: %s", dataFile.c_str());
    std::ifstream coordStream;
    if (!coordFile.empty()) {
        coordStream.open(coordFile);
        if (!coordStream.is_open())
            error("Error opening coordinate file: %s", coordFile.c_str());
    }
    bool hasCoordFile = coordStream.is_open();
    int32_t maxCdx = std::max(icol_x, icol_y);
    int32_t maxIdx;
    if (use_topk) {
        maxIdx = *std::max_element(topk.begin(), topk.end());
        maxIdx = std::max(maxIdx, (int32_t) *std::max_element(topp.begin(), topp.end()));
    } else {
        maxIdx = offset + K - 1;
    }
    if (!hasCoordFile) maxIdx = std::max(maxCdx, maxIdx);

    // Read coordinates from coordFile if provided
    std::vector<std::array<float, 2>> coords;
    if (hasCoordFile) {
        float x, y;
        std::string line;
        std::vector<std::string> tok;
        while (std::getline(coordStream, line)) {
            if (line[0] == '#') continue;
            split(tok, "\t ", line, maxCdx+1, true, true, true);
            if (tok.size() <= maxCdx) {
                error("Coordinate file line %s has insufficient columns", line.c_str());
            }
            if (!str2float(tok[icol_x], x) || !str2float(tok[icol_y], y)) {
                error("Failed to parse x/y in coordinate file line: %s", line.c_str());
            }
            coords.push_back({x, y});
        }
    }

    std::string line;
    std::vector<std::string> tokens;
    std::vector<cv::Vec3f> factorColors(static_cast<size_t>(K), cv::Vec3f(-1.f, -1.f, -1.f));
    std::vector<bool> channelMask(static_cast<size_t>(K), false);

    // Parse color scheme
    if (selected) { // parse selected factors
        if (colorListStr.empty()) {
            colorListStr = std::vector<std::string>{"144A74", "FF9900", "DD65E6", "FFEC11"};
        }
        if (channelListStr.size() > colorListStr.size()) {
            error("--channel-list and --color-list must have same length");
        }
        for (size_t i = 0; i < channelListStr.size(); ++i) {
            std::array<int32_t, 3> rgb;
            if (!set_rgb(colorListStr[i].c_str(), rgb))
                error("Invalid --color-list value: %s", colorListStr[i].c_str());
            uint32_t idx;
            if (!(str2uint32(channelListStr[i], idx) && idx < K)) {
                error("Channel %s not found in factor columns", channelListStr[i].c_str());
            }
            factorColors[idx] = cv::Vec3f(rgb[0], rgb[1], rgb[2]);
            channelMask[idx] = true;
        }
    } else {
        std::ifstream cs(colorFile);
        if (!cs.is_open())
            error("Error opening color file: %s", colorFile.c_str());
        std::vector<std::array<int,3>> colors;
        int32_t k = 0;
        while (std::getline(cs, line) && k < K) {
            if (line.empty() || line[0] == '#' || line[0] == 'R') continue;
            std::istringstream iss(line);
            int r, g, b;
            if (iss >> r >> g >> b) {
                factorColors[k] = cv::Vec3f(r, g, b);
            }
            k++;
        }
        if (k < K) {
            error("Not enough valid colors found in %s", colorFile.c_str());
        }
    }

    int32_t lineNo = 0;
    int32_t kept = 0, skipped = 0;
    PointCloudCV<float> cloud;
    std::vector<cv::Vec3f> anchor_colors;
    while (std::getline(dataStream, line)) {
        lineNo++;
        if (line.empty() || line[0] == '#') continue;
        split(tokens, "\t", line, UINT_MAX, true, true, true);
        if (tokens.empty()) continue;
        if (tokens.size() <= static_cast<size_t>(maxIdx)) {
            error("Line %d: insufficient columns (%s)", lineNo, line.c_str());
        }
        float x, y;
        if (hasCoordFile) {
            uint32_t idx;
            if (!str2uint32(tokens[icol_idx], idx) || idx >= coords.size()) {
                error("Line %d: failed to parse index or index out of range (%s)", lineNo, line.c_str());
            }
            x = coords[idx][0];
            y = coords[idx][1];
        } else {
            if (!str2float(tokens[icol_x], x) || !str2float(tokens[icol_y], y)) {
                error("Line %d: failed to parse x/y (%s)", lineNo, line.c_str());
            }
        }
        if (x < xmin - radius || x > xmax + radius ||
            y < ymin - radius || y > ymax + radius) {
            skipped++;
            continue;
        }
        if (top_only) {
            int32_t top_idx = -1;
            float top_p = -1.f;
            if (use_topk) {
                if (!str2int32(tokens[topk[0]], top_idx) || !str2float(tokens[topp[0]], top_p)) {
                    error("Line %d: failed to parse topk/topp (%s)", lineNo, line.c_str());
                }
            } else {
                for (int32_t j = 0; j < K; ++j) {
                    int i = offset + j;
                    float p;
                    if (!str2float(tokens[i], p)) {
                        error("Line %d: failed to parse probability at column %d (%s)", lineNo, i, line.c_str());
                    }
                    if (p > top_p) {
                        top_p = p;
                        top_idx = j;
                    }
                }
            }
            if (top_p <= 0.f || (selected && !channelMask[top_idx])) {
                skipped++;
                continue;
            }
            anchor_colors.push_back(factorColors[static_cast<size_t>(top_idx)]);
        } else {
            float wsum = 0.f;
            cv::Vec3f color(0.f, 0.f, 0.f);
            if (use_topk) {
                for (size_t t = 0; t < topk.size(); ++t) {
                    uint32_t k;
                    float p = 0.f;
                    if (!str2uint32(tokens[topk[t]], k) || !str2float(tokens[topp[t]], p)) {
                        error("Line %d: failed to parse topk/topp (%s)", lineNo, line.c_str());
                    }
                    if (selected && !channelMask[k]) continue;
                    color += factorColors[k] * p;
                    wsum += p;
                }
            } else {
                for (int32_t j = 0; j < K; ++j) {
                    if (selected && !channelMask[j]) continue;
                    int i = offset + j;
                    float p;
                    if (!str2float(tokens[i], p)) {
                        error("Line %d: failed to parse probability at column %d (%s)", lineNo, i, line.c_str());
                    }
                    color += factorColors[j] * p;
                    wsum += p;
                }
            }
            if (wsum <= 0) {
                skipped++;
                continue;
            }
            color /= wsum;
            anchor_colors.push_back(color);
        }
        cloud.pts.emplace_back(x, y);
        kept++;
    }
    if (anchor_colors.empty())
        error("No valid anchor_colors found");
    notice("Finished reading anchor_colors: kept %d, skipped %d", kept, skipped);

    size_t width  = size_t(std::floor((xmax - xmin) / scale)) + 1;
    size_t height = size_t(std::floor((ymax - ymin) / scale)) + 1;
    notice("Image size: %d x %d", width, height);
    if (width <= 1 || height <= 1)
        error("Image dimensions are zero; check your bounds/scale");

    kd_tree_cv2f_t kdtree(2, cloud, {10});
    std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;
    const float radius2 = static_cast<float>(radius * radius);
    std::vector<float> xcoords(width);
    for (size_t x = 0; x < width; ++x) {
        xcoords[x] = static_cast<float>(xmin + x * scale);
    }
    cv::Mat out(height, width, CV_8UC3, cv::Scalar(0, 0, 0));
    float query_pt[2];
    for (int y = 0; y < height; ++y) {
        query_pt[1] = static_cast<float>(ymin + y * scale);
        auto* row = out.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            query_pt[0] = xcoords[x];
            size_t found = kdtree.radiusSearch(query_pt, radius2, indices_dists);
            if (found == 0) continue;
            const auto& c = anchor_colors[indices_dists[0].first];
            row[x] = cv::Vec3b(
                cv::saturate_cast<uchar>(c[2]),
                cv::saturate_cast<uchar>(c[1]),
                cv::saturate_cast<uchar>(c[0]));
        }
    }

    notice("Writing image to %s ...", outFile.c_str());
    if (!cv::imwrite(outFile, out))
        error("Error writing output image: %s", outFile.c_str());

    return 0;
}
