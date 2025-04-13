#include <iostream>
#include <iomanip>
#include <random>
#include "punkst.h"
#include "utils.h"
#include "hexgrid.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "nanoflann.hpp"
#include "nanoflann_utils.h"
// #include "qgenlib/tsv_reader.h"
// #include "json.hpp"
// #include "variant.hpp"
// #include "vector_tile_config.hpp"
// #include "Eigen/Dense"

using vec2f_t = std::vector<std::vector<float>>;

int32_t test(int32_t argc, char** argv) {

	std::string intsv, outtsv;
    int32_t debug = 0, verbose = 500000;

    ParamList pl;
    // Input Options
    pl.add_option("in-tsv", "Input TSV file. Header must begin with #", intsv);
    // Output Options
    pl.add_option("out-tsv", "Output TSV file", outtsv)
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

    // test lloyd iteration

    // Define a square region with side length 500
    float tileSize = 100.0f;
    float R = 10;
    cv::Rect2f rect(0.0f, 0.0f, tileSize, tileSize);

    std::vector<cv::Point2f> anchors;

    // Generate a set of random anchors inside the region.
    const int numAnchors = 20;
    vec2f_t fixedAnchors;
    std::mt19937 rng(12345);  // Fixed seed for reproducibility.
    std::uniform_real_distribution<float> dist(0.0f, tileSize);

    for (int i = 0; i < numAnchors; ++i) {
        float x = dist(rng);
        float y = dist(rng);
        fixedAnchors.emplace_back(std::vector<float>{x, y});
        anchors.emplace_back(x, y);
    }

    size_t nFixed = fixedAnchors.size();
    std::cout << "generated " << nFixed << " fixed anchors\n";

    // Create lattice points
    vec2f_t lattice;
    hex_grid_cart<float>(lattice, R/2, tileSize-R/2, R/2, tileSize-R/2, R);
    std::cout << "generated " << lattice.size() << " lattice points\n";
    // 2 Remove lattice points too close to any fixed anchors
    KDTreeVectorOfVectorsAdaptor<vec2f_t, float> refpt(2, fixedAnchors, 10);

    double l2radius = R*R;
    for (const auto& pt : lattice) {
        std::vector<size_t> ret_indexes(1);
        std::vector<float> out_dists_sqr(1);
        nanoflann::KNNResultSet<float> resultSet(1);
        resultSet.init(&ret_indexes[0], &out_dists_sqr[0]);
        refpt.index->findNeighbors(resultSet, &pt[0]);
        if (out_dists_sqr[0] < l2radius) {
            continue;
        }
        anchors.emplace_back(pt[0], pt[1]);
    }
    size_t nAnchors = anchors.size();
    std::cout << "total " << nAnchors << " anchors\n";

    // Save the initial anchors (iteration 0) to a CSV file.
    {
        std::ofstream file("anchors_iter0.csv");
        file << "x,y,label\n";
        size_t i = 0;
        for (; i < nFixed; ++i) {
            file << anchors[i].x << "," << anchors[i].y << ",1" << "\n";
        }
        for (; i < nAnchors; ++i) {
            file << anchors[i].x << "," << anchors[i].y << ",0" << "\n";
        }
    }

    // Number of Lloyd iterations to perform.
    int nLloydIter = 3;
    std::vector<int> indices;
    // Perform Lloyd iterations.
    for (int iter = 0; iter < nLloydIter; ++iter) {
        cv::Subdiv2D subdiv(rect);
        subdiv.insert(anchors);
        // Retrieve the Voronoi facets for all anchors.
        std::vector<std::vector<cv::Point2f>> facets;
        std::vector<cv::Point2f> facetCenters;
        subdiv.getVoronoiFacetList(indices, facets, facetCenters);
        std::cout << "Iteration " << iter + 1 << ": Voronoi facets generated.\n";

        // Update anchor positions using the centroid of their Voronoi facet.
        for (size_t i = nFixed; i < nAnchors; ++i) {
            if (i < facets.size() && !facets[i].empty()) {
                auto poly = clipPolygonToRect(facets[i], rect);
                auto centroid = centroidOfPolygon(poly);
                std::cout << "Updating anchor " << i << " " << std::fixed << std::setprecision(2)
                << anchors[i].x << ", " << anchors[i].y
                << " to centroid: (" << centroid.x << ", " << centroid.y << ")\n";
                anchors[i] = centroid;
            }
        }

        // Output and save the results of the current iteration.
        std::ofstream file("anchors_iter" + std::to_string(iter+1) + ".csv");
        file << "x,y,label\n";
        size_t i = 0;
        for (; i < nFixed; ++i) {
            file << anchors[i].x << "," << anchors[i].y << ",1" << "\n";
        }
        for (; i < nAnchors; ++i) {
            file << anchors[i].x << "," << anchors[i].y << ",0" << "\n";
        }
        std::cout << "Finish iteration " << iter+1 << "\n";
    }

    return 0;
}
