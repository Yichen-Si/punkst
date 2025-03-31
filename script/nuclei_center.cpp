#include "punkst.h"
#include "utils.h"
#include "qgenlib/tsv_reader.h"
#include "qgenlib/qgen_utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "nanoflann.hpp"
#include "nanoflann_utils.h"

/** Identify nuclei centers
 * Find pixels with (approximately) local maximal intensities
 */
int32_t local_max(std::vector<cv::Point>& centers, cv::Mat img, int kernel_size = 50, double rel_thres = 0.95, int mrg_size = 10) {
    // Create kernel for dilation
    cv::Mat kernel = cv::Mat::ones(kernel_size, kernel_size, CV_8U);

    // Dilate the image
    cv::Mat dilate;
    cv::dilate(img, dilate, kernel, cv::Point(-1, -1), 1);

    // Create mask
    cv::Mat mask = img >= dilate * rel_thres;
    cv::Mat medMask = cv::Mat::zeros(img.size(), CV_16U);
    cv::medianBlur(mask, medMask, kernel_size - kernel_size % 2 - 1);
    mask = mask & (img > (medMask + dilate + 2) / 2);

    // Create kernel to connect adjacent local max points
    kernel = cv::Mat::ones(mrg_size, mrg_size, CV_8U);
    cv::dilate(mask, dilate, kernel, cv::Point(-1, -1), 1);
    dilate = dilate * 255;

    // Find contours
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(dilate, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Calculate centers of contours
    centers.clear();
    centers.reserve(contours.size());
    for (const auto& contour : contours) {
        cv::Moments M = cv::moments(contour);
        if (M.m00 != 0) {
            int cx = static_cast<int>(M.m10 / M.m00);
            int cy = static_cast<int>(M.m01 / M.m00);
            centers.push_back(cv::Point(cx, cy));
        }
    }
    return centers.size();
}

using KDTreeI = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<int32_t, PointCloudCV<int32_t> >, PointCloudCV<int32_t>, 2>;
using KDTreeF = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloudCV<float> >, PointCloudCV<float>, 2>;

/**
 * Find nuclei centers
 * Annotate input transcript file with the distance to the nearest nuclei
 */
int32_t cmdImgNucleiCenter(int32_t argc, char** argv) {

	std::string unsplpng, outpng;
    std::string intsv, outtsv, outf;
    double unspl_gf_sig = 3;
    int32_t max_flt_kernel_size = 50;
    double max_flt_rel_thres = 0.95;
    int32_t local_max_mrg_size = 10;
    int32_t intensity_flt_size = 16;
    double intensity_flt_qt = 0.3;
    int32_t draw_nuclei_center_radius = 3;
    std::string draw_nuclei_color = "FF0000";
    int32_t icol_x = -1, icol_y = -1, icol_unspl = -1;
    int32_t offset_x = 0, offset_y = 0;
    double coord_per_pixel = -1;
    int32_t debug = 0, verbose = 500000;
    double report_dist = 5;
    bool write_intensity_qt = false;

	// Parse input parameters
	paramList pl;

	BEGIN_LONG_PARAMS(longParameters)
		LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("unspl-png", &unsplpng, "Unspliced read PNG")
        LONG_STRING_PARAM("in-tsv", &intsv, "Input TSV file. Header must begin with #")
        // Image processing parameters
        LONG_DOUBLE_PARAM("unspl-gf-sig", &unspl_gf_sig, "Unspliced Gaussian filter sigma")
        LONG_INT_PARAM("max-flt-kernel-size", &max_flt_kernel_size, "Kernel size for local maxima detection")
        LONG_DOUBLE_PARAM("max-flt-rel-thres", &max_flt_rel_thres, "Relexation factor for local maxima detection")
        LONG_INT_PARAM("local-max-mrg-size", &local_max_mrg_size, "Merge local maximum points within this distance")
        LONG_DOUBLE_PARAM("intensity-qt", &intensity_flt_qt, "Quantile for filtering nuclei centers by local intensity")
        LONG_INT_PARAM("intensity-flt-size", &intensity_flt_size, "Square size (side length) to calculate local intensities for filtering")
        // If annotating tsv file - coordinate conversion parameters
        LONG_INT_PARAM("icol-x", &icol_x, "Column index for x coordinates (corresponding to the width of the image)")
        LONG_INT_PARAM("icol-y", &icol_y, "Column index for y coordinates (corresponding to the height of the image)")
        LONG_INT_PARAM("icol-unspl", &icol_unspl, "Column index for unspliced read counts")
        LONG_DOUBLE_PARAM("coord-per-pixel", &coord_per_pixel, "Number of coordinate units per pixel (translate between tsv and image)")
        LONG_INT_PARAM("offset-x", &offset_x, "Offset for x coordinates")
        LONG_INT_PARAM("offset-y", &offset_y, "Offset for y coordinates")
		LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("out-png", &outpng, "Output PNG that highlights nuclei centers")
        LONG_STRING_PARAM("center-color", &draw_nuclei_color, "Color code (hex) for drawing nuclei centers")
        LONG_STRING_PARAM("out-tsv", &outtsv, "Output TSV file")
        LONG_DOUBLE_PARAM("runtime-report-dist", &report_dist, "Periodic report pixels counts within distance to the nearest nuclei at runtime")
        LONG_PARAM("write-intensity-qt", &write_intensity_qt, "Write intensity quantile values for reference")
        LONG_INT_PARAM("verbose", &verbose, "Verbose")
        LONG_INT_PARAM("debug", &debug, "Debug")
    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

    if (unsplpng.empty() || intsv.empty() || outtsv.empty()) {
        error("--unspl-png, --in-tsv, and --out-tsv must be specified");
    }

    std::array<int32_t, 3> rgb;
    if (!set_rgb(draw_nuclei_color.c_str(), rgb)) {
        error("Invalid color code %s", draw_nuclei_color.c_str());
    }
    cv::Mat unspl_img = cv::imread(unsplpng, cv::IMREAD_GRAYSCALE);
    int32_t height = unspl_img.rows; // Y
    int32_t width  = unspl_img.cols; // X
    notice("Read an image of dimension %d (height) x %d (width)", height, width);

    // Blur
    cv::GaussianBlur(unspl_img, unspl_img, cv::Size(0, 0), unspl_gf_sig);
    // Identify nuclei centers
    std::vector<cv::Point> centers;
    int32_t n_nuclei = local_max(centers, unspl_img, max_flt_kernel_size, max_flt_rel_thres, local_max_mrg_size);

    // Filter centers by local unspl density
    std::vector<float> intensities(n_nuclei, 0);
    int32_t sq_size = intensity_flt_size / 2;
    for (size_t idx = 0; idx < n_nuclei; ++idx) {
        int32_t x = centers[idx].x;
        int32_t y = centers[idx].y;
        float s = 0.0f;
        int32_t xl = std::max(x - sq_size, 0);
        int32_t xu = std::min(x + sq_size, width - 1);
        int32_t yl = std::max(y - sq_size, 0);
        int32_t yu = std::min(y + sq_size, height - 1);
        int32_t area = (xu - xl + 1) * (yu - yl + 1);
        for (int32_t i = xl; i <= xu; ++i) {
            for (int32_t j = yl; j <= yu; ++j) {
                s += unspl_img.at<uchar>(j, i);
            }
        }
        intensities[idx] = s / area;
    }
    size_t n_qt = 20;
    int32_t i_qt = -1;
    std::vector<double> percentiles;
    percentiles.reserve(n_qt);
    for (int32_t i = 1; i < n_qt; ++i) {
        double q = i * 1. / n_qt;
        if (i_qt < 0 && q > intensity_flt_qt) {
            percentiles.push_back(intensity_flt_qt);
            i_qt = i;
        }
        percentiles.push_back(q);
    }
    std::vector<float> thresholds;
    std::vector<float> intensities_cpy(intensities);
    compute_percentile<float>(thresholds, intensities_cpy, percentiles);
    // Write quantile values for reference
    size_t i_substr = outpng.find_last_of(".");
    if (write_intensity_qt) {
        outf = outpng.substr(0, i_substr) + ".intensity_qt.tsv";
        htsFile* wf = hts_open(outf.c_str(), "w");
        if (wf == NULL) {
            error("Cannot open file %s for writing", outf.c_str());
        }
        hprintf(wf, "Quantile\tValue\n");
        for (size_t i = 0; i < percentiles.size(); ++i) {
            hprintf(wf, "%.4f\t%.4f\n", percentiles[i], thresholds[i]);
        }
        hts_close(wf);
    }
    // Write center coordinates (SGE coordinates)
    outf = outpng.substr(0, i_substr) + ".nuclei_centers.tsv";
    htsFile *wf = hts_open(outf.c_str(), "w");
    PointCloudCV<float> pc_centers;
    pc_centers.pts.reserve(n_nuclei);
    size_t ii = 0;
    for (size_t i = 0; i < n_nuclei; ++i) {
        if (intensities[i] > thresholds[i_qt]) {
            pc_centers.pts.emplace_back((float) centers[i].x, (float) centers[i].y);
            hprintf(wf, "%d\t%d\n", (int32_t) (centers[i].x * coord_per_pixel) + offset_x, (int32_t) (centers[i].y * coord_per_pixel) + offset_y);
            centers[ii] = centers[i];
            ii++;
        }
    }
    notice("Filter by intensity (%.3f, %.3f), from %d to %d", intensity_flt_qt, thresholds[i_qt], n_nuclei, ii);
    hts_close(wf);
    n_nuclei = ii;
    centers.resize(n_nuclei);
    // Visualize centers
    if (!outpng.empty()) {
        cv::Mat annotated_unspl_img;
        cv::cvtColor(unspl_img, annotated_unspl_img, cv::COLOR_GRAY2BGR);
        for (const auto& center : centers) {
            cv::circle(annotated_unspl_img, center, draw_nuclei_center_radius, cv::Scalar(rgb[2], rgb[1], rgb[0]), -1);
        }
        cv::imwrite(outpng, annotated_unspl_img);
    }
if (debug == 99) {
    return 0;
}

    // Build KDTree
    KDTreeF kdtree(2, pc_centers, {10});

    // Annotate input tsv file
    tsv_reader tr(intsv.c_str());
    int32_t ncols = tr.read_line();
    if (ncols <= icol_x || ncols <= icol_y || ncols <= icol_unspl) {
        error("Column index out of range");
    }
    wf = hts_open(outtsv.c_str(), "wz");
    if (wf == NULL) {
        error("Cannot open file %s for writing", outtsv.c_str());
    }
    while (tr.str_field_at(0)[0] == '#') {
        tr.read_line();
    }
    float px, py, d;
    size_t c_idx;
    uint64_t nline = 0, nnuc = 0;
    while(true) {
        nline++;
        if (nline % verbose == 0) {
            notice("Reading line %lu, %lu (%.3f) within %.1f um of any detected centers", nline, nnuc, (float) nnuc / nline, report_dist);
        }
        px = (float) (tr.int_field_at(icol_x) - offset_x) / coord_per_pixel;
        py = (float) (tr.int_field_at(icol_y) - offset_y) / coord_per_pixel;
        const float query_pt[2] = {px, py};
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&c_idx, &d);
        kdtree.findNeighbors(resultSet, query_pt);
        d = std::sqrt(d);
        hprintf(wf, "%s\t%d\t%.3f\n", tr.line.c_str(), c_idx, d);
        if (d < report_dist) {
            nnuc++;
        }
        if (debug && nline > debug) {
            break;
        }
        if (tr.read_line() != ncols) {
            break;
        }
    }
    hts_close(wf);
    tr.close();

    return 0;
}
