#include "tiles2cooccurrence.hpp"
#include <thread>
#include <future>
#include <atomic>
#include <numeric>

int32_t cmdTiles2FeatureCooccurrence(int32_t argc, char** argv) {
    std::string inTsv, inIndex, outPref, dictFile;
    double radius, halflife = -1, localMin = 0;
    int nThreads = 1, debug = 0;
    int icol_x, icol_y, icol_feature, icol_val;
    bool binaryOutput = false, weightByCount = false;
    int minNeighbor = 1;
    int minTotalCount = 1;
    std::vector<double> boundingBoxes;

    ParamList pl;
    // Input Options
    pl.add_option("in-tsv", "Input TSV file. Header must begin with #", inTsv, true)
        .add_option("in-index", "Input index file", inIndex, true)
        .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x, true)
        .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y, true)
        .add_option("icol-feature", "Column index for feature (0-based)", icol_feature)
        .add_option("feature-dict", "If feature column is not integer, provide a dictionary/list of all possible values", dictFile)
        .add_option("icol-val", "Column index for the integer count (0-based)", icol_val)
        .add_option("bounding-boxes", "Rectangular query regions (xmin ymin xmax ymax)*", boundingBoxes)
        .add_option("weight-by-count", "Weight co-occurrence by the product of the number of transcripts (default: false)", weightByCount)
        .add_option("radius", "Radius to count coocurrence", radius, true)
        .add_option("halflife", "Halflife for exponential decay (default: -1, unweighted count)", halflife)
        .add_option("min-neighbor", "Minimum number of neighbors within the radius for a pixel to be included", minNeighbor)
        .add_option("local-min", "Minimum cooccurrence within a tile to record", localMin)
        .add_option("threads", "Number of threads to use (default: 1)", nThreads);
    // Output Options
    pl.add_option("out", "Output prefix", outPref, true)
        .add_option("binary", "Output in binary format (default: false)", binaryOutput)
        .add_option("debug", "Debug", debug);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    std::vector<Rectangle<double>> rects;
    if (boundingBoxes.size() > 0) {
        int32_t nrects = parseCoordsToRects(rects, boundingBoxes);
        if (nrects <= 0) {
            error("Error parsing bounding boxes");
        }
        notice("Received %d bounding boxes", nrects);
    }

    TileReader tileReader(inTsv, inIndex, &rects);
    if (!tileReader.isValid()) {
        error("Error parsing input index file");
        return 1;
    }
    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val, dictFile, &rects);

    Tiles2FeatureCooccurrence cooccurrence(nThreads, tileReader, parser, outPref, radius, halflife, localMin, binaryOutput, minNeighbor, weightByCount, debug);

    cooccurrence.run();

    return 0;
}




// TODO: allow input to have different sets of features
// Add up multiple co-occurrence matrices
static void accumulate_file(const std::string& fname,
                            bool binary, int valueBytes,
                            uint32_t nRows, uint32_t nCols,
                            std::vector<double>& localQ) {
    std::ifstream ifs;
    if (binary)  ifs.open(fname, std::ios::binary);
    else         ifs.open(fname);

    if (!ifs)  {
        warning("Cannot open matrix: %s", fname.c_str());
        return;
    }

    size_t npairs = 0;
    if (binary) {
        uint32_t f1, f2;  double val;
        while (ifs.read(reinterpret_cast<char*>(&f1), sizeof(f1))) {
            ifs.read(reinterpret_cast<char*>(&f2), sizeof(f2));
            ifs.read(reinterpret_cast<char*>(&val), valueBytes);
            if (f1 < nRows && f2 < nCols)
                localQ[f1 * nCols + f2] += val;

            if (++npairs % 5'000'000 == 0)
                notice("[thread] %s : %zu pairs", fname.c_str(), npairs);
        }
    } else {
        std::string line;
        while (std::getline(ifs, line)) {
            uint32_t f1, f2;  double val;
            std::istringstream iss(line);
            if (!(iss >> f1 >> f2 >> val)) continue;
            if (f1 < nRows && f2 < nCols)
                localQ[f1 * nCols + f2] += val;

            if (++npairs % 5'000'000 == 0)
                notice("[thread] %s : %zu pairs", fname.c_str(), npairs);
        }
    }
}

int32_t cmdMergeCooccurrenceMtx(int32_t argc, char** argv) {
    std::string inList, outPref;
    int32_t nRows = -1, nCols = -1;
    int32_t valueBytes = 8;
    bool binaryInput = false, binaryOutput = false;
    int nThreads = 1;

    ParamList pl;
    pl.add_option("in-list", "List of co-occurrence files to merge", inList, true)
        .add_option("binary", "Input matrix is in binary format", binaryInput)
        .add_option("value-bytes", "Number of bytes for each value in the matrix (default: 8, only used for binary input)", valueBytes)
        .add_option("shared-nrows", "", nRows, true)
        .add_option("shared-ncols", "", nCols)
        .add_option("threads", "Number of threads to use (default: 1)", nThreads);
    pl.add_option("out", "Output prefix", outPref, true)
        .add_option("binary-output", "Output matrix in binary format (default: false)", binaryOutput);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (nRows <= 0) {
        error("--shared-nrows must be positive");
    }
    if (nCols <= 0) {
        nCols = nRows;
    }

    std::vector<std::string> inFiles;
    std::ifstream ifs(inList);
    if (!ifs) {
        error("Error opening input list: %s", inList.c_str());
    }
    std::string line;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::vector<std::string> tokens;
        split(tokens, " \t", line, 1, true, true, true);
        if (tokens.size() < 1) {
            warning("Invalid line in input list: %s", line.c_str());
            continue;
        }
        inFiles.push_back(tokens[0]);
    }
    ifs.close();
    notice("Input list contains %d files", inFiles.size());

    std::vector<double> Q(nRows * nCols, 0.0);

    if (nThreads > 1) {
        std::vector<std::vector<double>> partialQ(nThreads,
                                                std::vector<double>(nRows * nCols, 0.0));
        std::vector<std::thread> workers;
        // Launch reader threads
        for (unsigned t = 0; t < nThreads; ++t) {
            workers.emplace_back([&, t] {
                for (size_t idx = t; idx < inFiles.size(); idx += nThreads) {
                    accumulate_file(inFiles[idx],
                                    binaryInput, valueBytes,
                                    nRows, nCols,
                                    partialQ[t]);
                }
            });
        }
        for (auto& th : workers) th.join();
        notice("Finished threaded reading");

        // Sum each partialQ into the final Q
        for (unsigned t = 0; t < nThreads; ++t) {
            const auto& buf = partialQ[t];
            for (size_t i = 0; i < buf.size(); ++i)
                Q[i] += buf[i];
        }
    } else {
        int32_t nfiles = 0;
        for (const auto& filename : inFiles) {
            notice("Reading %s", filename.c_str());
            accumulate_file(filename, binaryInput, valueBytes, nRows, nCols, Q);
            nfiles++;
        }

        // if (binaryInput) {
        //     for (const auto& filename : inFiles) {
        //         ifs.open(filename, std::ios::binary);
        //         if (!ifs)
        //             error("Cannot open matrix: %s", filename.c_str());
        //         notice("Reading %s", filename.c_str());
        //         size_t npairs = 0;
        //         uint32_t f1, f2;
        //         double val;
        //         while (ifs.read(reinterpret_cast<char*>(&f1), sizeof(f1))) {
        //             ifs.read(reinterpret_cast<char*>(&f2), sizeof(f2));
        //             ifs.read(reinterpret_cast<char*>(&val), valueBytes);
        //             if (f1 < nRows && f2 < nCols) {
        //                 Q[f1 * nCols + f2] += val;
        //             }
        //             npairs++;
        //             if (npairs % 1000000 == 0) {
        //                 notice("Read %zu pairs", npairs);
        //             }
        //         }
        //         ifs.close();
        //         nfiles++;
        //     }
        // } else {
        //     for (const auto& filename : inFiles) {
        //         ifs.open(filename);
        //         if (!ifs)
        //             error("Cannot open matrix: %s", filename.c_str());
        //         notice("Reading %s", filename.c_str());
        //         size_t npairs = 0;
        //         std::string line;
        //         while (std::getline(ifs, line)) {
        //             std::istringstream iss(line);
        //             uint32_t f1, f2; double val;
        //             if (!(iss >> f1 >> f2 >> val)) continue;
        //             if (f1 < nRows && f2 < nCols) {
        //                 Q[f1 * nCols + f2] += val;
        //             }
        //             npairs++;
        //             if (npairs % 1000000 == 0) {
        //                 notice("Read %zu pairs", npairs);
        //             }
        //         }
        //         ifs.close();
        //         nfiles++;
        //     }
        // }
        notice("Finished summing over %d files", nfiles);
    }

    uint32_t outBytes = 8, idxBytes = 4;
    if (binaryOutput) {
        std::string outFile = outPref + ".mtx.bin";
        std::ofstream ofs(outFile, std::ios::binary);
        if (!ofs) {
            error("Cannot open output: %s", outFile.c_str());
        }
        for (uint32_t r = 0; r < nRows; ++r) {
            for (uint32_t c = 0; c < nCols; ++c) {
                ofs.write(reinterpret_cast<const char*>(&r), idxBytes);
                ofs.write(reinterpret_cast<const char*>(&c), idxBytes);
                ofs.write(reinterpret_cast<const char*>(&Q[r*nCols+c]), outBytes);
            }
        }
    } else {
        std::string outFile = outPref + ".mtx.tsv";
        FILE* ofs = fopen(outFile.c_str(), "w");
        if (!ofs) {
            error("Cannot open output: %s", outFile.c_str());
        }
        for (uint32_t r = 0; r < nRows; ++r) {
            for (uint32_t c = 0; c < nCols; ++c) {
                fprintf(ofs, "%u\t%u\t%.2f\n", r, c, Q[r*nCols+c]);
            }
        }
        fclose(ofs);
    }
    notice("Finished writing output");

    return 0;
}
