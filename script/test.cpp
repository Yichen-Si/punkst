#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <string>
#include <sstream>
#include <iomanip>
#include <chrono>
#include "punkst.h"

int32_t test(int32_t argc, char** argv) {
    int N = 80;
    int K = 3;
    int seed = 7;
    int runs = 0;
    double grad_tol = 5e-4;
    double hv_tol = 5e-4;
    double fd_step = 1e-6;
    double b_min = -0.2, b_max = 0.2;
    double o_min = 0.5, o_max = 5.0;
    double skew = 0.5;
    int init_newton = 0;

    ParamList pl;
    pl.add_option("N", "Number of rows", N)
      .add_option("K", "Number of mixture components", K)
      .add_option("seed", "Random seed", seed)
      .add_option("runs", "Number of timing runs", runs)
      .add_option("grad-tol", "Relative tolerance for gradient check", grad_tol)
      .add_option("hv-tol", "Relative tolerance for Hv check", hv_tol)
      .add_option("fd-step", "Finite difference step size", fd_step)
      .add_option("b-min", "Minimum true coefficient value", b_min)
      .add_option("b-max", "Maximum true coefficient value", b_max)
      .add_option("o-min", "Minimum offset value", o_min)
      .add_option("o-max", "Maximum offset value", o_max)
      .add_option("skew", "Row boost factor for stochastic matrix", skew)
      .add_option("init-newton", "Initial per-coordinate Newton steps", init_newton);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    return 0;
}
