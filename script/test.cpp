#include "punkst.h"
#include "utils.h"
#include "qgenlib/tsv_reader.h"
#include "qgenlib/qgen_utils.h"
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include "nanoflann.hpp"
#include "nanoflann_utils.h"
#include "Eigen/Dense"
#include <iostream>

template <typename num_t>
void kdtree_demo(const size_t N)
{
    using std::cout;
    using std::endl;

    PointCloud<num_t> cloud;

    // Generate points:
    generateRandomPointCloud(cloud, N);

    // construct a kd-tree index:
    using my_kd_tree_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<num_t, PointCloud<num_t>>,
        PointCloud<num_t>, 3 /* dim */
        >;

    my_kd_tree_t index(3 /*dim*/, cloud, {10 /* max leaf */});

#if 0
	// Test resize of dataset and rebuild of index:
	cloud.pts.resize(cloud.pts.size()*0.5);
	index.buildIndex();
#endif

    const num_t query_pt[3] = {0.5, 0.5, 0.5};

    // ----------------------------------------------------------------
    // knnSearch():  Perform a search for the N closest points
    // ----------------------------------------------------------------
    {
        size_t                num_results = 5;
        std::vector<uint32_t> ret_index(num_results);
        std::vector<num_t>    out_dist_sqr(num_results);

        num_results = index.knnSearch(
            &query_pt[0], num_results, &ret_index[0], &out_dist_sqr[0]);

        // In case of less points in the tree than requested:
        ret_index.resize(num_results);
        out_dist_sqr.resize(num_results);

        cout << "knnSearch(): num_results=" << num_results << "\n";
        for (size_t i = 0; i < num_results; i++)
            cout << "idx[" << i << "]=" << ret_index[i] << " dist[" << i
                 << "]=" << out_dist_sqr[i] << endl;
        cout << "\n";
    }

    // ----------------------------------------------------------------
    // radiusSearch(): Perform a search for the points within search_radius
    // ----------------------------------------------------------------
    {
        const num_t search_radius = static_cast<num_t>(0.1);
        std::vector<nanoflann::ResultItem<uint32_t, num_t>> ret_matches;

        // nanoflanSearchParamsameters params;
        // params.sorted = false;

        const size_t nMatches =
            index.radiusSearch(&query_pt[0], search_radius, ret_matches);

        cout << "radiusSearch(): radius=" << search_radius << " -> " << nMatches
             << " matches\n";
        for (size_t i = 0; i < nMatches; i++)
            cout << "idx[" << i << "]=" << ret_matches[i].first << " dist[" << i
                 << "]=" << ret_matches[i].second << endl;
        cout << "\n";
    }
}

int32_t test(int32_t argc, char** argv) {

	std::string intsv, outtsv;
    int32_t debug = 0, verbose = 500000;

	// Parse input parameters
	paramList pl;
	BEGIN_LONG_PARAMS(longParameters)
		LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("in-tsv", &intsv, "Input TSV file. Header must begin with #")
		LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("out-tsv", &outtsv, "Output TSV file")
        LONG_INT_PARAM("verbose", &verbose, "Verbose")
        LONG_INT_PARAM("debug", &debug, "Debug")
    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

    // Test nanoflann
    srand(static_cast<unsigned int>(time(nullptr)));
    kdtree_demo<float>(4);
    kdtree_demo<double>(100000);

    // Test Eigen
    Eigen::MatrixXd m(2,2);
    m(0,0) = 3;
    m(1,0) = 2.5;
    m(0,1) = -1;
    m(1,1) = m(1,0) + m(0,1);
    std::cout << m << std::endl;

    return 0;
}
