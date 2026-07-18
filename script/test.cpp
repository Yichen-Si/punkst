#include "punkst.h"

#include <cstdint>
#include <iostream>
#include <string>

int32_t test(int32_t argc, char** argv) {
    std::string suite = "fast";
    std::string output;

    ParamList pl;
    pl.add_option("suite", "Test suite placeholder", suite)
      .add_option("out", "Output placeholder", output);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    (void)suite;
    (void)output;
    return 0;
}
