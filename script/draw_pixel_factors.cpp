#include "punkst.h"
#include "utils.h"
#include "json.hpp"
#include <opencv2/opencv.hpp>
#include <fstream>
#include <sstream>
#include <iostream>

int32_t cmdDrawPixelFactors(int32_t argc, char** argv) {

    std::string dataFile, headerFile, colorFile, outFile;
    double scale, xmin, xmax, ymin, ymax;
    int32_t verbose = 1000000;

    paramList pl;
	BEGIN_LONG_PARAMS(longParameters)
		LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("in-tsv", &dataFile, "Input TSV file. Header must begin with #")
        LONG_STRING_PARAM("header-json", &headerFile, "Header file")
        LONG_STRING_PARAM("in-color", &colorFile, "Input color file")
        LONG_DOUBLE_PARAM("scale", &scale, "Scale factor to translate coordinates to pixel in the output image ((x-xmin)/scale = pixel_x)")
        LONG_DOUBLE_PARAM("xmin", &xmin, "Minimum x coordinate")
        LONG_DOUBLE_PARAM("xmax", &xmax, "Maximum x coordinate")
        LONG_DOUBLE_PARAM("ymin", &ymin, "Minimum y coordinate")
        LONG_DOUBLE_PARAM("ymax", &ymax, "Maximum y coordinate")
        LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("out", &outFile, "Output image file")
        LONG_INT_PARAM("verbose", &verbose, "Verbose")
    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

    if (dataFile.empty() || headerFile.empty() || colorFile.empty() || outFile.empty()) {
        error("--in-tsv, --header-json, --in-color, and --out are required");
    }
    if (!checkOutputWritable(outFile)) {
        error("Output file is not writable: %s", outFile.c_str());
    }
    std::ifstream dataStream(dataFile);
    if (!dataStream.is_open()) {
        error("Error opening data file: %s", dataFile.c_str());
    }
    std::ifstream colorStream(colorFile);
    if (!colorStream.is_open()) {
        error("Error opening color file: %s", colorFile.c_str());
    }
    std::ifstream headerStream(headerFile);
    if (!headerStream.is_open()) {
        error("Error opening header file: %s", headerFile.c_str());
    }

    // Load the JSON header file.
    nlohmann::json header;
    try {
        headerStream >> header;
    } catch (const std::exception& e) {
        std::cerr << "Error parsing JSON header: " << e.what() << std::endl;
        return -1;
    }
    headerStream.close();
    uint32_t icol_x = header["x"];
    uint32_t icol_y = header["y"];
    std::vector<uint32_t> icol_ks;
    std::vector<uint32_t> icol_ps;
    int32_t k  = 1;
    while (header.contains("K" + std::to_string(k)) && header.contains("P" + std::to_string(k))) {
        icol_ks.push_back(header["K" + std::to_string(k)]);
        icol_ps.push_back(header["P" + std::to_string(k)]);
        k++;
    }
    if (icol_ks.empty()) {
        error("No K and P columns found in the header file");
    }
    k = std::min(3, k-1);
    uint32_t maxIdx = std::max(icol_x, icol_y);
    for (int i = 0; i < k; ++i) {
        maxIdx = std::max(maxIdx, std::max(icol_ks[i], icol_ps[i]));
    }

    // Image size
    int width = static_cast<int>(std::floor((xmax - xmin) / scale)) + 1;
    int height = static_cast<int>(std::floor((ymax - ymin) / scale)) + 1;

    // RGB color table
    std::vector<std::vector<int>> cmtx;
    std::string line;
    while (std::getline(colorStream, line)) {
        if (line.empty()) continue;
        std::istringstream iss(line);
        int r, g, b;
        if (!(iss >> r >> g >> b))
            continue;
        cmtx.push_back({r, g, b});
    }
    colorStream.close();
    int32_t K = cmtx.size();

    // Prepare accumulators for each pixel.
    // sumImg will store the cumulative RGB contributions
    // countImg will store the number of records that contribute to each pixel
    std::vector<std::vector<cv::Vec3f>> sumImg(height, std::vector<cv::Vec3f>(width, cv::Vec3f(0, 0, 0)));
    std::vector<std::vector<uint8_t>> countImg(height, std::vector<uint8_t>(width, 0));

    int32_t nline = 0;
    while (std::getline(dataStream, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        nline++;
        if (nline % verbose == 0) {
            notice("Processed %d lines", nline);
        }

        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() != maxIdx + 1) {
            error("Error reading data file at line: %s", line.c_str());
        }
        double x = std::stod(tokens[icol_x]);
        double y = std::stod(tokens[icol_y]);
        // Compute pixel indices.
        int xpix = static_cast<int>((x - xmin) / scale);
        int ypix = static_cast<int>((y - ymin) / scale);
        // Discard records that fall outside the image bounds.
        if (xpix < 0 || xpix >= width || ypix < 0 || ypix >= height)
            continue;
        if (countImg[ypix][xpix] >= 255)
            continue;

        // weighted RGB
        float r = 0, g = 0, b = 0;
        bool validRecord = true;
        for (int i = 0; i < k; i++) {
            int j;
            double p;
            try {
                j = std::stoi(tokens[icol_ks[i]]);
                p = std::stod(tokens[icol_ps[i]]);
            } catch (std::exception& e) {
                validRecord = false;
                break;
            }
            if (j < 0 || j >= K) {
                validRecord = false;
                break;
            }
            r += cmtx[j][0] * p;
            g += cmtx[j][1] * p;
            b += cmtx[j][2] * p;
        }
        if (!validRecord)
            continue;
        sumImg[ypix][xpix] += cv::Vec3f(r, g, b);
        countImg[ypix][xpix] += 1;
    }
    dataStream.close();
    notice("Finished reading input pixels, start populating an image");

    // Create the output image (CV_8UC3).
    cv::Mat outputImage(height, width, CV_8UC3, cv::Scalar(0, 0, 0));

    // For each pixel, if there are contributions, compute the average RGB value.
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            if (countImg[i][j] > 0) {
                cv::Vec3f avg = sumImg[i][j] / countImg[i][j];
                // Clamp the values to the 0-255 range.
                uchar R = cv::saturate_cast<uchar>(avg[0]);
                uchar G = cv::saturate_cast<uchar>(avg[1]);
                uchar B = cv::saturate_cast<uchar>(avg[2]);
                // Note: OpenCV uses BGR order.
                outputImage.at<cv::Vec3b>(i, j) = cv::Vec3b(B, G, R);
            }
        }
    }

    // Write the image to a PNG file.
    notice("Writing image to %s ...", outFile.c_str());
    if (!cv::imwrite(outFile, outputImage)) {
        error("Error writing output image file: %s", outFile.c_str());
    }

    return 0;
}
