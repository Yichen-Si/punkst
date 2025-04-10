#include "punkst.h"
#include "qgenlib/dataframe.h"
#include "qgenlib/tsv_reader.h"
#include <array>
#include <stdlib.h>
#include <unordered_map>
#include "utils.h"
// #define cimg_display 0
// #include "CImg.h"
#include <opencv2/opencv.hpp>

int32_t cmdTsvDrawByColumn(int32_t argc, char** argv) {

    std::string inTsv, manifestf, outf;
    int32_t debug = 0, verbose = 5000000;
    double coord_per_pixel = 1.; // 1 pixel = X coordinate unit
    double intensity_adj = 0.9; // normalize the intensity of the channel by qt
    int32_t icol_x = 0, icol_y = 1;
    std::vector<std::string> color_lists;
    int32_t gray_channel = -1;

    ParamList pl;
    // Input Options
    pl.add_option("input", "", inTsv)
      .add_option("manifest", "Bounding box information. Expects xmin,xmax,ymin,ymax for a manifest file (tall or wide format)", manifestf)
      .add_option("color-list", "[color_code]:[column index]", color_lists)
      .add_option("gray-scale", "Column index to read values to create a gray scale image (--color-list will be ignored and output image will be in gray scale)", gray_channel)
      .add_option("icol-x", "Column index for x", icol_x)
      .add_option("icol-y", "Column index for y", icol_y)
      .add_option("coord-per-pixel", "Number of coordinate units per pixel", coord_per_pixel)
      .add_option("intensity-adj", "Adjust the intensity of each channel by qt", intensity_adj);
    // Output Options
    pl.add_option("output", "", outf)
      .add_option("verbose", "Verbose", verbose)
      .add_option("debug", "Debug", debug);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    // read the manifest file and determine the xmin/xmax/ymin/ymax
    uint32_t xmin = 0, xmax = 0, ymin = 0, ymax = 0;
    uint8_t  flag = 0;
    tsv_reader manifest(manifestf.c_str());
    int32_t ncols = manifest.read_line();
    std::vector<std::string> colnames;
    for (int32_t i = 0; i < ncols; ++i) {
        colnames.push_back(manifest.str_field_at(i));
    }
    if (ncols < 4 || manifest.read_line() != ncols) {
        error("The manifest file is ill-formated %s", manifestf.c_str());
    }
    for (int32_t i = 0; i < ncols; ++i) {
        if (colnames[i] == "xmin") {
            xmin = manifest.uint64_field_at(i);
            flag |= 1;
        } else if (colnames[i] == "xmax") {
            xmax = manifest.uint64_field_at(i);
            flag |= 2;
        } else if (colnames[i] == "ymin") {
            ymin = manifest.uint64_field_at(i);
            flag |= 4;
        } else if (colnames[i] == "ymax") {
            ymax = manifest.uint64_field_at(i);
            flag |= 8;
        }
    }
    if (flag != 15) {
        error("The manifest file is missing one of the following columns: xmin, xmax, ymin, ymax");
    }

    // Parse the color list
    std::map<uint32_t, std::array<int32_t, 3>> colors;
    uint32_t max_icol = (uint32_t) std::max(icol_x, icol_y);
    int32_t n_channel = 3;
    if (gray_channel >= 0) {
        n_channel = 1;
        if ((uint32_t) gray_channel > max_icol) {
            max_icol = gray_channel;
        }
        colors[gray_channel] = std::array<int32_t, 3>{255, 255, 255};
    } else {
        for (uint32_t i = 0; i < color_lists.size(); ++i) {
            std::vector<std::string> tokens;
            split(tokens, ":", color_lists[i]);
            if (tokens.size() != 2)
                error("Invalid color list %s", color_lists[i].c_str());
            uint32_t icol;
            std::array<int32_t, 3> rgb;
            if (!set_rgb(tokens[1].c_str(), rgb) || !str2uint32(tokens[0], icol))
                error("Invalid color code %s", color_lists[i].c_str());
            colors.emplace(icol, std::move(rgb));
            if (icol > max_icol) {
                max_icol = icol;
            }
        }
    }

    uint32_t n_color = colors.size();
    for (const auto& kv : colors) {
        printf("Column index %u: RGB %d %d %d\n", kv.first, kv.second[0], kv.second[1], kv.second[2]);
    }

    // Read the input TSV file
    tsv_reader tr(inTsv.c_str());
    ncols = tr.read_line();
    if (ncols <= max_icol) {
        error("The input TSV file does not have enough columns");
    }
    while (tr.str_field_at(0)[0] == '#') {
        tr.read_line();
    }
    // Need a histogram per color to normalize/adjust the intensity
    // Column (color) -> (x, y) -> intensity
    std::map<uint32_t, std::unordered_map<uint64_t, int32_t> > hist;
    for (auto& kv : colors) {
        hist[kv.first] = std::unordered_map<uint64_t, int32_t>();
    }
    uint32_t px, py;
    uint64_t xy, nline = 0;
    notice("Start reading from input");
    while (true) {
        nline++;
        if (nline % verbose == 0) {
            notice("Reading line %lu", nline);
        }
        bool nonzero = false;
        for (auto& kv : colors) {
            int32_t val = tr.int_field_at(kv.first);
            if (val > 0) {
                nonzero = true;
                break;
            }
        }
        if (!nonzero) {
            if (tr.read_line() != ncols) {
                break;
            }
            continue;
        }
        px = (uint32_t) ((tr.uint64_field_at(icol_x) - xmin) / coord_per_pixel);
        py = (uint32_t) ((tr.uint64_field_at(icol_y) - ymin) / coord_per_pixel);
        xy = (uint64_t) px << 32 | py;
        for (auto& kv : colors) {
            int32_t val = tr.int_field_at(kv.first);
            if (val > 0) {
                hist[kv.first][xy] += val;
            }
        }
        if (tr.read_line() != ncols) {
            break;
        }
        if (debug && nline > debug) {
            break;
        }
    }
    // get normalization factor per channel
    notice("Finding normalizing factors");
    std::map<uint32_t, int32_t> norm_factors;
    for (auto& c : colors) {
        std::vector<int32_t> intensities;
        for (auto& kv : hist[c.first]) {
            intensities.push_back(kv.second);
        }
        std::sort(intensities.begin(), intensities.end());
        uint32_t idx = (uint32_t) (intensities.size() * intensity_adj);
        norm_factors[c.first] = intensities[idx];
    }
    for (auto& kv : norm_factors) {
        printf("Column %u: cap intensity %d\n", kv.first, kv.second);
    }
    // Initialize the image
    int32_t height = (int32_t) (ceil((double)(ymax - ymin + 1) / coord_per_pixel));
    int32_t width  = (int32_t) (ceil((double)(xmax - xmin + 1) / coord_per_pixel));
    // cimg_library::CImg<unsigned char> image(width, height, 1, n_channel, 0);
    int32_t cv_type = n_channel == 1 ? CV_8UC1 : CV_8UC3;
    cv::Mat image = cv::Mat::zeros(height, width, cv_type);
    // Draw the image
    notice("Start drawing the image %d x %d", height, width);
    for (const auto& kv : hist) {
        const auto& rgb = colors[kv.first];
        int32_t norm_factor = norm_factors[kv.first];
        for (const auto& entry : kv.second) {
            int32_t x = (int32_t) (entry.first >> 32);
            int32_t y = (int32_t) (entry.first & 0xFFFFFFFF);
            double norm_intensity = std::min(1.0, (double)entry.second / norm_factor);
            if (cv_type == CV_8UC1) {
                image.at<uint8_t>(y, x) = (uint8_t) std::min(255, (int32_t)(norm_intensity * 255));
            } else {
                for (int c = 0; c < n_channel; ++c) { // opencv use BGR
                    image.at<cv::Vec3b>(y, x)[c] = (uint8_t) std::min(255, (int32_t)(norm_intensity * rgb[2-c]));
                }
            }
            // for (int c = 0; c < n_channel; ++c) {
                // image(x, y, c) = (uint8_t) std::min(255, (int32_t)(norm_intensity * rgb[c]));
            // }
        }
    }

    notice("Writing the image to %s", outf.c_str());
    // image.save_png(outf.c_str());
    cv::imwrite(outf, image);

    return 0;
}
