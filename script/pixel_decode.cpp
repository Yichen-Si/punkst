#include "punkst.h"
#include "tiles2slda.hpp"
#include "tiles2nmf.hpp"

int32_t cmdPixelDecode(int32_t argc, char** argv) {

    std::string inTsv, inIndex, modelFile, anchorFile, outFile, outPref, tmpDirPath, weightFile;
    std::string sampleList; // for multi-sample
    int nThreads = 1, seed = -1, debug_ = 0, verbose = 0;
    int icol_x, icol_y, icol_feature, icol_val;
    double hexSize = -1, hexGridDist = -1;
    double radius = -1, anchorDist = -1;
    int32_t nMoves = -1, minInitCount = 10, topK = 3;
    double minCountAnchor = 5;
    double pixelResolution = 1, defaultWeight = 0.;
    bool inMemory = false;
    bool outputOritinalData = false;
    bool featureIsIndex = false;
    bool coordsAreInt = false;
    bool outputAnchor = false;
    bool useTicketSystem = false;
    int32_t floatCoordDigits = 4, probDigits = 4;
    std::vector<std::string> annoInts, annoFloats, annoStrs;
    int32_t maxIter = 100;
    double mDelta = 1e-3;

    std::string algo = "slda"; // or "nmf"

    MLEOptions opts;
    opts.mle_only_mode();
    opts.optim.tron.enabled = true;
    double sizeFactor = 10000.0;
    bool exactMLE = false;
    std::string mapBinFile;

    ParamList pl;
    // Input Options
    pl.add_option("algo", "Decoding algorithm: \"slda\" or \"nmf\")", algo)
      .add_option("model", "Model file", modelFile, true)
      .add_option("model-bin", "", mapBinFile)
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
      .add_option("max-iter", "Maximum number of iterations (default: 100)", maxIter)
      .add_option("mean-change-tol", "Mean change of document-topic probability tolerance for convergence (default: 1e-3)", mDelta)
      .add_option("radius", "Radius", radius)
      .add_option("n-moves", "Number of steps to slide on each axis to create anchors", nMoves)
      .add_option("threads", "Number of threads to use (default: 1)", nThreads)
      .add_option("seed", "Random seed", seed);
    // EM-NMF specific options
    pl.add_option("size-factor", "Size factor used for per-anchor EM updates", sizeFactor)
      .add_option("exact", "Use exact Poisson updates (no log1p approximation)", exactMLE)
      .add_option("max-iter-inner", "Maximum iterations of the inner MLE solver", opts.optim.max_iters)
      .add_option("tol-inner", "Tolerance of the inner MLE solver", opts.optim.tol)
      .add_option("weight-thres-anchor", "Minimum total weight for an anchor to be kept for next iteration", minCountAnchor)
      .add_option("ridge", "Ridge stabilization parameter for MLE", opts.ridge);
    // Output Options
    pl.add_option("out", "Output TSV file (backward compatibility)", outFile)
      .add_option("out-pref", "Output prefix", outPref)
      .add_option("output-original", "Output original data points (pixels with feature values) together with the pixel level factor results", outputOritinalData)
      .add_option("ext-col-ints", "Additional integer columns to carry over to output file, in the form of \"idx1:name1 idx2:name2 ...\" where 'idx' are 0-based column indices", annoInts)
      .add_option("ext-col-floats", "Additional float columns to carry over to output file, in the form of \"idx1:name1 idx2:name2 ...\" where 'idx' are 0-based column indices", annoFloats)
      .add_option("ext-col-strs", "Additional string columns to carry over to output file, in the form of \"idx1:name1:len1 idx2:name2:len2 ...\" where 'idx' are 0-based column indices and 'len' are maximum lengths of strings", annoStrs)
      .add_option("output-anchors", "Output anchor level info", outputAnchor)
      .add_option("use-ticket-system", "Use ticket system to ensure predictable output order", useTicketSystem)
      .add_option("temp-dir", "Directory to store temporary files", tmpDirPath)
      .add_option("in-memory", "Keep boundary buffers in memory instead of writing to temporary files", inMemory)
      .add_option("top-k", "Top K factors to output", topK)
      .add_option("min-init-count", "Minimum", minInitCount)
      .add_option("output-coord-digits", "Number of decimal digits to output for coordinates (only used if input coordinates are float or --output-original is not set)", floatCoordDigits)
      .add_option("output-prob-digits", "Number of decimal digits to output for probabilities", probDigits)
      .add_option("verbose", "Verbose", verbose)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }
    if (debug_ > 0) {
        logger::Logger::getInstance().setLevel(logger::LogLevel::DEBUG);
    }

    if (algo != "slda" && algo != "nmf") {
        error("Invalid --algo (%s). Must be either \"slda\" or \"nmf\"", algo.c_str());
    }
    if (!inMemory && tmpDirPath.empty()) {
        error("If --in-memory is not set, --temp-dir is required");
    }

    // Collect input data files
    std::vector<dataset> datasets;
    if (sampleList.empty()) {
        if (inTsv.empty() || inIndex.empty())
            error("For single sample analysis both --in-tsv and --in-index must be specified");
        if (outFile.empty() && outPref.empty())
            error("For single sample analysis either --out-pref or --out must be specified");
        if (outPref.empty()) {
            size_t pos = outFile.find_last_of(".");
            if (pos != std::string::npos) {
                outPref = outFile.substr(0, pos);
            } else {
                outPref = outFile;
            }
        }
        dataset ds{"", inTsv, inIndex, outPref};
        ds.anchorFile = anchorFile;
        datasets.push_back(ds);
    } else {
         datasets = parseSampleList(sampleList, &outPref);
    }
    if (datasets.empty()) {
        error("No valid datasets found in sample list or input parameters");
    }

    if (hexSize <= 0 && hexGridDist <= 0) {
        error("Hexagon size (--hex-size) or hexagon grid distance (--hex-grid-dist) must be provided");
    }
    if (nMoves <= 0 && anchorDist <= 0) {
        error("Number of grid shifts (--n-moves) or anchor distance (--anchor-dist) must be provided");
    }
    if (hexSize <= 0) {
        hexSize = hexGridDist / std::sqrt(3.0);
    } else {
        hexGridDist = hexSize * std::sqrt(3.0);
    }
    HexGrid hexGrid(hexSize);

    if (nMoves <= 0) {
        nMoves = std::max<int32_t>(static_cast<int32_t>(std::ceil(hexGridDist / anchorDist)), 1);
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


    RowMajorMatrixXd beta; // M x K
    std::vector<std::string> featureNames;
    std::vector<std::string> factorNames;
    read_matrix_from_file(modelFile, beta, &featureNames, &factorNames);
    if (featureNames.empty()) {
        error("Model file %s must contain feature names", modelFile.c_str());
    }
    int32_t M_model = static_cast<int32_t>(beta.rows());
    int32_t K_model = static_cast<int32_t>(beta.cols());
    notice("Read model with %d features and %d factors", M_model, K_model);

    // Set up input parser
    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val);
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
    if (!featureIsIndex) {
        parser.setFeatureDict(featureNames);
    }
    if (!weightFile.empty()) {
        parser.readWeights(weightFile, defaultWeight, M_model);
    }
    notice("Initialized tile reader");


    auto configure_decoder = [&](auto& decoder, const std::string& anchorFile) {
        decoder.setOutputOptions(outputOritinalData, outputAnchor, useTicketSystem);
        decoder.setOutputCoordDigits(floatCoordDigits);
        decoder.setOutputProbDigits(probDigits);
        if (!anchorFile.empty()) {
            int32_t nAnchors = decoder.loadAnchors(anchorFile);
            notice("Loaded %d valid anchors", nAnchors);
        }
    };

    if (algo == "nmf") {
        PixelEM emPois(debug_);
        emPois.set_em_options(maxIter, mDelta, minCountAnchor);
        if (!mapBinFile.empty()) {
            emPois.init_mlr(mapBinFile, beta);
        } else {
            emPois.init_pnmf(beta, opts, sizeFactor, exactMLE);
        }

        M_model = emPois.get_M();
        K_model = emPois.get_K();
        if (!weightFile.empty()) {
            parser.readWeights(weightFile, defaultWeight, M_model);
        }

        for (const auto& ds : datasets) {
            if (!ds.sampleId.empty()) {
                notice("Processing sample %s", ds.sampleId.c_str());
            }

            TileReader tileReader(ds.inTsv, ds.inIndex, nullptr, -1, coordsAreInt);
            if (!tileReader.isValid()) {
                error("Error in input tiles: %s", ds.inTsv.c_str());
            }

            if (coordsAreInt) {
                Tiles2NMF<int32_t> decoder(
                    nThreads, radius, ds.outPref, tmpDirPath,
                    emPois, tileReader, parser, hexGrid, nMoves,
                    seed, static_cast<double>(minInitCount), 0.7, pixelResolution,
                    topK, verbose, debug_);
                configure_decoder(decoder, ds.anchorFile);
                decoder.run();
            } else {
                Tiles2NMF<float> decoder(
                    nThreads, radius, ds.outPref, tmpDirPath,
                    emPois, tileReader, parser, hexGrid, nMoves,
                    seed, static_cast<double>(minInitCount), 0.7, pixelResolution,
                    topK, verbose, debug_);
                configure_decoder(decoder, ds.anchorFile);
                decoder.run();
            }
        }
        return 0;
    }

    RowMajorMatrixXd betaT = beta.transpose(); // K x M
    LatentDirichletAllocation lda(betaT, seed, 1);
    lda.set_svb_parameters(maxIter, mDelta);
    // Process each dataset
    for (const auto& ds : datasets) {
        anchorFile = ds.anchorFile;
        if (!ds.sampleId.empty()) {
            notice("Processing sample %s\n", ds.sampleId.c_str());
        }

        TileReader tileReader(ds.inTsv, ds.inIndex, nullptr, -1, coordsAreInt);
        if (!tileReader.isValid()) {
            error("Error in input tiles: %s", ds.inTsv.c_str());
        }
        if (coordsAreInt) {
            Tiles2SLDA<int32_t> tiles2slda(nThreads, radius, ds.outPref, tmpDirPath, lda, tileReader, parser, hexGrid, nMoves, seed, minInitCount, 0.7, pixelResolution, 0, topK, verbose, debug_);
            configure_decoder(tiles2slda, ds.anchorFile);
            tiles2slda.setFeatureNames(featureNames);
            tiles2slda.run();
        } else {
            Tiles2SLDA<float> tiles2slda(nThreads, radius, ds.outPref, tmpDirPath, lda, tileReader, parser, hexGrid, nMoves, seed, minInitCount, 0.7, pixelResolution, 0, topK, verbose, debug_);
            configure_decoder(tiles2slda, ds.anchorFile);
            tiles2slda.setFeatureNames(featureNames);
            tiles2slda.run();
        }
    }

    return 0;
}
