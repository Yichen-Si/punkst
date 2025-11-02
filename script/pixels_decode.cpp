#include "punkst.h"
#include "tiles2slda.hpp"

int32_t cmdPixelDecode(int32_t argc, char** argv) {

    std::string inTsv, inIndex, modelFile, anchorFile, outFile, outPref, tmpDirPath, weightFile;
    std::string sampleList; // for multi-sample
    int nThreads = 1, seed = -1, debug = 0, verbose = 0;
    int icol_x, icol_y, icol_feature, icol_val;
    double hexSize = -1, hexGridDist = -1;
    double radius = -1, anchorDist = -1;
    int32_t nMoves = -1, minInitCount = 10, topK = 3;
    double pixelResolution = 1, defaultWeight = 0.;
    bool inMemory = false;
    bool outputOritinalData = false;
    bool featureIsIndex = false;
    bool coordsAreInt = false;
    bool useTicketSystem = false;
    int32_t floatCoordDigits = 4, probDigits = 4;
    std::vector<std::string> annoInts, annoFloats, annoStrs;
    // bool useSCVB0 = false;
    // SVB specific parameters
    int32_t maxIter = 100;
    double mDelta = 1e-3;

    ParamList pl;
    // Input Options
    pl.add_option("model", "Model file", modelFile, true)
      .add_option("in-tsv", "Input TSV file. Header must begin with #", inTsv)
      .add_option("in-index", "Input index file", inIndex)
      .add_option("anchor", "Anchor file", anchorFile)
      .add_option("sample-list", "A tsv file containing input and output information for multiple samples. The columns should be sample_id, input_tsv, input_index, output_prefix, (input_anchor)", sampleList)
      .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x, true)
      .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y, true)
      .add_option("coords-are-int", "If the coordinates are integers, otherwise assume they are floats", coordsAreInt)
      .add_option("icol-feature", "Column index for feature (0-based)", icol_feature, true)
      .add_option("icol-val", "Column index for count/value (0-based)", icol_val, true)
      .add_option("feature-is-index", "If the feature column contains integer indices, otherwise assume it contains feature names", featureIsIndex)
      .add_option("default-weight", "Default weight for features not in the weight file", defaultWeight)
      .add_option("feature-weights", "Input weights file", weightFile)
      .add_option("pixel-res", "Resolution of pixel level inference", pixelResolution)
      .add_option("hex-size", "Hexagon size (side length)", hexSize)
      .add_option("hex-grid-dist", "Hexagon grid distance (center-to-center distance)", hexGridDist)
      .add_option("anchor-dist", "Distance between adjacent anchors", anchorDist)
    //   .add_option("scvb0", "Use SCVB0 instead of SVB", useSCVB0)
      .add_option("max-iter", "Maximum number of iterations for each document (default: 100)", maxIter)
      .add_option("mean-change-tol", "Mean change of document-topic probability tolerance for convergence (default: 1e-3)", mDelta)
      .add_option("radius", "Radius", radius)
      .add_option("n-moves", "Number of steps to slide on each axis to create anchors", nMoves)
      .add_option("threads", "Number of threads to use (default: 1)", nThreads)
      .add_option("seed", "Random seed", seed);
    // Output Options
    pl.add_option("out", "Output TSV file (backward compatibility)", outFile)
      .add_option("out-pref", "Output prefix", outPref)
      .add_option("output-original", "Output original data points (pixels with feature values) together with the pixel level factor results", outputOritinalData)
      .add_option("ext-col-ints", "Additional integer columns to carry over to output file, in the form of \"idx1:name1 idx2:name2 ...\" where 'idx' are 0-based column indices", annoInts)
      .add_option("ext-col-floats", "Additional float columns to carry over to output file, in the form of \"idx1:name1 idx2:name2 ...\" where 'idx' are 0-based column indices", annoFloats)
      .add_option("ext-col-strs", "Additional string columns to carry over to output file, in the form of \"idx1:name1:len1 idx2:name2:len2 ...\" where 'idx' are 0-based column indices and 'len' are maximum lengths of strings", annoStrs)
      .add_option("use-ticket-system", "Use ticket system to ensure predictable output order", useTicketSystem)
      .add_option("temp-dir", "Directory to store temporary files", tmpDirPath)
      .add_option("in-memory", "Keep boundary buffers in memory instead of writing to temporary files", inMemory)
      .add_option("top-k", "Top K factors to output", topK)
      .add_option("min-init-count", "Minimum", minInitCount)
      .add_option("output-coord-digits", "Number of decimal digits to output for coordinates (only used if input coordinates are float or --output-original is not set)", floatCoordDigits)
      .add_option("output-prob-digits", "Number of decimal digits to output for probabilities", probDigits)
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
    if (sampleList.empty()) {
        if (outFile.empty() && outPref.empty())
            error("For single sample analysis either --out-pref or --out must be specified");
        if (inTsv.empty() || inIndex.empty())
            error("For single sample analysis both --in-tsv and --in-index must be specified");
        if (outPref.empty()) {
            size_t pos = outFile.find_last_of(".");
            if (pos != std::string::npos) {
                outPref = outFile.substr(0, pos);
            } else {
                outPref = outFile;
            }
        }
    }
    if (!inMemory && tmpDirPath.empty()) {
        error("If --in-memory is not set, --temp-dir is required");
    }
    if (hexSize <= 0) {
        if (hexGridDist <= 0) {
            error("Hexagon size or hexagon grid distance must be provided");
        }
        hexSize = hexGridDist / sqrt(3);
    } else {
        hexGridDist = hexSize * sqrt(3);
    }
    if (nMoves <= 0) {
        if (anchorDist <= 0) {
            error("Anchor distance or number of moves must be provided");
        }
        nMoves = std::max((int32_t) std::ceil(hexGridDist / anchorDist), 1);
    } else {
        anchorDist = hexGridDist / nMoves;
    }
    if (radius <= 0) {
        radius = anchorDist * 1.2;
    }
    if (seed <= 0) {
        seed = std::random_device{}();
        notice("Using random seed %d", seed);
    }

    // Initialize LDA model
    LatentDirichletAllocation lda(modelFile, seed, 1, 0);
    lda.set_svb_parameters(maxIter, mDelta);

    auto& featureNames = lda.feature_names_;
    int32_t nFeatures = lda.get_n_features();
    notice("Initialized anchor model with %d features and %d factors", nFeatures, lda.get_n_topics());

    // Set up input parser
    HexGrid hexGrid(hexSize);
    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val);
    if (!featureIsIndex) {
        parser.setFeatureDict(featureNames);
    }
    if (!weightFile.empty()) {
        parser.readWeights(weightFile, defaultWeight, nFeatures);
    }
    // parse additional annotation columns (to carry over to output)
    if (!parser.addExtraInt(annoInts)) {
        error("Invalid value in --ext-col-ints");
    }
    if (!parser.addExtraFloat(annoFloats)) {
        error("Invalid value in --ext-col-floats");
    }
    if (!parser.addExtraStr(annoStrs)) {
        error("Invalid value in --ext-col-strs");
    }
    notice("Initialized tile reader");

    // Collect input data files
    struct dataset {
        std::string sampleId;
        std::string inTsv;
        std::string inIndex;
        std::string outPref;
        std::string anchorFile;
    };
    std::vector<dataset> datasets;
    if (!sampleList.empty()) {
        std::ifstream rf(sampleList);
        if (!rf) {
            error("Error opening sample list file: %s", sampleList.c_str());
        }
        std::string line;
        while (std::getline(rf, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t", line);
            if (tokens.size() < 3) {
                error("Invalid line in sample list: %s", line.c_str());
            }
            dataset ds{tokens[0], tokens[1], tokens[2]};
            if (tokens.size() > 3) {
                ds.outPref = tokens[3];
                if (tokens.size() > 4) {
                    ds.anchorFile = tokens[4];
                }
            } else {
                size_t pos = ds.inTsv.find_last_of("/\\");
                if (pos != std::string::npos) {
                    ds.outPref = ds.inTsv.substr(0, pos+1) + ds.sampleId;
                } else {
                    ds.outPref = ds.sampleId;
                }
                if (!outPref.empty()) {
                    ds.outPref += "." + outPref;
                }
            }
            datasets.push_back(ds);
        }
    } else {
        dataset ds{"", inTsv, inIndex, outPref};
        ds.anchorFile = anchorFile;
        datasets.push_back(ds);
    }
    if (datasets.empty()) {
        error("No valid datasets found in sample list or input parameters");
    }

    // Process each dataset
    for (const auto& ds : datasets) {
        inTsv = ds.inTsv;
        inIndex = ds.inIndex;
        outPref = ds.outPref;
        anchorFile = ds.anchorFile;
        if (!ds.sampleId.empty()) {
            notice("Processing sample %s\n", ds.sampleId.c_str());
        }

        TileReader tileReader(inTsv, inIndex, nullptr, -1, coordsAreInt);
        if (!tileReader.isValid()) {
            error("Error in input tiles: %s", inTsv.c_str());
        }
        if (coordsAreInt) {
            Tiles2SLDA<int32_t> tiles2slda(nThreads, radius, outPref, tmpDirPath, lda, tileReader, parser, hexGrid, nMoves, seed, minInitCount, 0.7, pixelResolution, 0, topK, verbose, debug);
            tiles2slda.setOutputOptions(outputOritinalData, useTicketSystem);
            tiles2slda.setFeatureNames(featureNames);
            tiles2slda.setOutputCoordDigits(floatCoordDigits);
            tiles2slda.setOutputProbDigits(probDigits);
            if (!anchorFile.empty()) {
                int32_t nAnchors = tiles2slda.loadAnchors(anchorFile);
                notice("Loaded %d valid anchors", nAnchors);
            }
            tiles2slda.run();
        } else {
            Tiles2SLDA<float> tiles2slda(nThreads, radius, outPref, tmpDirPath, lda, tileReader, parser, hexGrid, nMoves, seed, minInitCount, 0.7, pixelResolution, 0, topK, verbose, debug);
            tiles2slda.setOutputOptions(outputOritinalData, useTicketSystem);
            tiles2slda.setOutputCoordDigits(floatCoordDigits);
            tiles2slda.setOutputProbDigits(probDigits);
            tiles2slda.setFeatureNames(featureNames);
            if (!anchorFile.empty()) {
                int32_t nAnchors = tiles2slda.loadAnchors(anchorFile);
                notice("Loaded %d valid anchors", nAnchors);
            }
            tiles2slda.run();
        }
    }

    return 0;
}
