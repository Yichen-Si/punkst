#include "punkst.h"
#include "qgenlib/dataframe.h"
#include "qgenlib/tsv_reader.h"
#include "qgenlib/qgen_utils.h"
#define cimg_display 0 // remove the need for X11 library
#include "CImg.h"
#include <array>
#include <stdlib.h>
#include <unordered_map>
#include "utils.h"


int32_t cmdTsvDrawByColumn(int32_t argc, char** argv) {

    std::string inTsv, manifestf, outf;
    int32_t debug = 0, verbose = 5000000;
    double coord_per_pixel = 1.; // 1 pixel = X coordinate unit
    double intensity_adj = 0.9; // normalize the intensity of the channel by qt
    int32_t icol_x = 0, icol_y = 1;
    std::vector<std::string> color_lists;

    // Parse input parameters
    paramList pl;
    BEGIN_LONG_PARAMS(longParameters)
        LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("input", &inTsv, "")
        LONG_STRING_PARAM("manifest", &manifestf, "Bounding box information. Expects xmin,xmax,ymin,ymax for a manifest file (tall or wide format)")
        LONG_MULTI_STRING_PARAM("color-list", &color_lists, "[color_code]:[column index]")
        LONG_INT_PARAM("icol-x", &icol_x, "Column index for x")
        LONG_INT_PARAM("icol-y", &icol_y, "Column index for y")
        LONG_DOUBLE_PARAM("coord-per-pixel", &coord_per_pixel, "Number of coordinate units per pixel")
        LONG_DOUBLE_PARAM("intensity-adj", &intensity_adj, "Adjust the intensity of each channel by qt")
        LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("output", &outf, "")
        LONG_INT_PARAM("verbose", &verbose, "Verbose")
        LONG_INT_PARAM("debug", &debug, "Debug")
    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

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
    uint32_t n_color = colors.size();
    for (const auto& kv : colors) {
        printf("Column index %lu: RGB %d %d %d\n", kv.first, kv.second[0], kv.second[1], kv.second[2]);
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
        printf("Column %lu: cap intensity %d\n", kv.first, kv.second);
    }
    // Initialize the image
    int32_t height = (int32_t) (ceil((double)(ymax - ymin + 1) / coord_per_pixel));
    int32_t width  = (int32_t) (ceil((double)(xmax - xmin + 1) / coord_per_pixel));
    cimg_library::CImg<unsigned char> image(width, height, 1, 3, 0);
    // Draw the image
    notice("Start drawing the image %d x %d", width, height);
    for (const auto& kv : hist) {
        const auto& rgb = colors[kv.first];
        int32_t norm_factor = norm_factors[kv.first];
        for (const auto& entry : kv.second) {
            uint64_t pos = entry.first;
            int32_t intensity = entry.second;
            int32_t x = (int32_t) (pos >> 32);
            int32_t y = (int32_t) (pos & 0xFFFFFFFF);
            double norm_intensity = std::min(1.0, (double)intensity / norm_factor);
            for (int c = 0; c < 3; ++c) {
                image(x, y, c) = (uint8_t) std::min(255, image(x, y, c) + (int32_t)(norm_intensity * rgb[c]));
            }
        }
    }

    notice("Writing the image to %s", outf.c_str());
    image.save_png(outf.c_str());

    return 0;
}
