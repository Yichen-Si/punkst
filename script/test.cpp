#include "punkst.h"
#include "utils.h"
#include "numerical_utils.hpp"

#include <Eigen/Dense>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>

int test(int argc, char** argv) {
    std::string inFile, outFile;
    int32_t threads = 1;

    ParamList pl;
    pl.add_option("input", "Input data file", inFile)
      .add_option("output", "Output data file", outFile)
      .add_option("threads", "Number of threads to use", threads);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    // test openCV multi-threading
    std::cout << "OpenCV threads: " << cv::getNumThreads() << "\n";
    cv::setNumThreads(threads);
    std::cout << "OpenCV threads (after): " << cv::getNumThreads() << "\n";

    return 0;
}
