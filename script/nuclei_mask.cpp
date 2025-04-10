#include "punkst.h"
#include "utils.h"
#include "qgenlib/tsv_reader.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>

/** Create a mask of nuclei from unspliced and spliced read density images
 * Annotate input transcript file with a "nuclei score"
 */
int32_t cmdImgNucleiMask(int32_t argc, char** argv) {

	std::string unsplpng, splpng, outpng;
    std::string intsv, outtsv;
    int32_t erode_kernel_size = 5, dilate_kernel_size = 5;
    double unspl_gf_sig = 3, spl_bf_size = 50;
    double clip_qt_lb = 0.5, clip_qt_ub = 0.99;
    int32_t icol_x = -1, icol_y = -1;
    int32_t offset_x = 0, offset_y = 0;
    double coord_per_pixel = -1;
    int32_t debug = 0, verbose = 500000;

    ParamList pl;
    // Input Options
    pl.add_option("unspl-png", "Unspliced read PNG", unsplpng)
      .add_option("spl-png", "Spliced read PNG", splpng)
      .add_option("in-tsv", "Input TSV file. Header must begin with #", intsv)
      // Image processing parameters
      .add_option("unspl-gf-sig", "Unspliced Gaussian filter sigma", unspl_gf_sig)
      .add_option("spl-bf-size", "Spliced box filter size", spl_bf_size)
      .add_option("clip-qt-lb", "Quantile for setting small intensity values to zero", clip_qt_lb)
      .add_option("clip-qt-ub", "Quantile for capping and normalizing intensities", clip_qt_ub)
      .add_option("erode-kernel-size", "Erosion kernel size", erode_kernel_size)
      .add_option("dilate-kernel-size", "Dilation kernel size", dilate_kernel_size)
      // If annotating tsv file - coordinate conversion parameters
      .add_option("icol-x", "Column index for x coordinates (corresponding to the width of the image)", icol_x)
      .add_option("icol-y", "Column index for y coordinates (corresponding to the height of the image)", icol_y)
      .add_option("coord-per-pixel", "Number of coordinate units per pixel (translate between tsv and image)", coord_per_pixel)
      .add_option("offset-x", "Offset for x coordinates", offset_x)
      .add_option("offset-y", "Offset for y coordinates", offset_y);
    // Output Options
    pl.add_option("out-png", "Output PNG file that shows the nuclei mask", outpng)
      .add_option("out-tsv", "Output TSV file", outtsv)
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

    cv::Mat unspl_img = cv::imread(unsplpng, cv::IMREAD_GRAYSCALE);
    cv::Mat spl_img = cv::imread(splpng, cv::IMREAD_GRAYSCALE);
    int32_t height = unspl_img.rows; // Y
    int32_t width  = unspl_img.cols; // X
    if (height != spl_img.rows || width != spl_img.cols) {
        error("Image dimensions do not match");
    }
    notice("Read images of dimension %d (height) x %d (width)", height, width);

    // Smooth unspliced image
    cv::GaussianBlur(unspl_img, unspl_img, cv::Size(0, 0), unspl_gf_sig);
    // Control for local transcript density
    cv::blur(spl_img, spl_img, cv::Size(spl_bf_size, spl_bf_size));

    cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erode_kernel_size, erode_kernel_size));
    cv::Mat kernel_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilate_kernel_size, dilate_kernel_size));

    cv::erode(unspl_img, unspl_img, kernel_erode, cv::Point(-1, -1), 1);
    std::vector<double> percentiles = {clip_qt_lb};
    std::vector<uchar> results;
    percentile(results, unspl_img, percentiles);
    std::cout << "Cutoff " << (int32_t) results[0] << std::endl;
    unspl_img.setTo(0, unspl_img < results[0]);

    cv::Mat ratio_img;
    unspl_img.convertTo(ratio_img, CV_16U);
    ratio_img += spl_img;
    ratio_img.forEach<ushort>([&](ushort& pixel, const int* position) -> void {
        if (pixel > 0) {
            pixel = static_cast<ushort>((unspl_img.at<uchar>(position[0], position[1]) * 255) / pixel);
        } else {
            pixel = 0;
        }
    });

    percentiles = {clip_qt_lb, clip_qt_ub};
    percentile(results, ratio_img, percentiles);
    std::cout << "Bound " << (int32_t) results[0] << " " << (int32_t) results[1] << std::endl;
    ratio_img.setTo(0, ratio_img < results[0]);
    ratio_img.convertTo(ratio_img, CV_8U, 255.0 / results[1]);

    cv::erode(ratio_img, ratio_img, kernel_erode, cv::Point(-1, -1), 1);
    cv::dilate(ratio_img, ratio_img, kernel_dilate, cv::Point(-1, -1), 1);

    if (!outpng.empty())
        cv::imwrite(outpng, ratio_img);

    if (!intsv.empty() && !outtsv.empty()) {
        if (icol_x < 0 || icol_y < 0 || coord_per_pixel < 0) {
            error("Missing required parameters --icol-x, --icol-y, --coord-per-pixel for annotating TSV file");
        }
    } else {
        return 0;
    }

    htsFile* wf = hts_open(outtsv.c_str(), "wz");
    if (wf == NULL) {
        error("Cannot open file %s for writing", outtsv.c_str());
    }
    tsv_reader tr(intsv.c_str());
    int32_t ncols = tr.read_line();
    if (ncols <= icol_x || ncols <= icol_y) {
        error("Column index out of range");
    }
    while (tr.str_field_at(0)[0] == '#') {
        hprintf(wf, "%s\tUnsplScore\n", tr.line.c_str());
        tr.read_line();
    }
    int32_t px, py;
    uint64_t nline = 0, nrec = 0, nnuc = 0;
    while(true) {
        nline++;
        if (nline % verbose == 0) {
            notice("Reading line %lu, annotated %lu (%lu > 0.5)", nline, nrec, nnuc);
        }
        px = (int32_t) ((tr.int_field_at(icol_x) - offset_x) / coord_per_pixel);
        py = (int32_t) ((tr.int_field_at(icol_y) - offset_y) / coord_per_pixel);
        if (py < height && px < width && px >= 0 && py >= 0) {
            float val = 1. * ratio_img.at<uchar>(py, px) / 255;
            hprintf(wf, "%s\t%.3f\n", tr.line.c_str(), val);
            nrec++;
            if (val > 0.5) {
                nnuc++;
            }
            if (debug && nrec > debug) {
                break;
            }
        }
        if (tr.read_line() != ncols) {
            break;
        }
    }
    hts_close(wf);
    tr.close();

    return 0;
}
