#include <iostream>
#include <iomanip>
#include <random>
#include <iostream>
#include <thread>

#include "punkst.h"
#include "utils.h"
#include "hexgrid.h"
#include "nanoflann.hpp"

#include <tbb/parallel_for.h>
#include <tbb/global_control.h>

// #include "json.hpp"
// #include "Eigen/Dense"

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

    // test tbb

    std::cout << "TBB max concurrency = "
    << tbb::this_task_arena::max_concurrency()
    << std::endl;

    // 2) Cap threads explicitly (optional)
    tbb::global_control ctl{tbb::global_control::max_allowed_parallelism, 4};
    std::cout << "Capped to 4 threads, now = "
        << tbb::this_task_arena::max_concurrency()
        << std::endl;

    // 3) Simple parallel_for test
    tbb::parallel_for(0, 8, [&](int i){
    auto id = std::this_thread::get_id();
    std::cout << "  Iter " << i << " on thread " << id << "\n";
    });

    return 0;
}
