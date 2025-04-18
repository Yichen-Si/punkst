#include "punkst.h"
#include "tiles2minibatch.hpp"

int32_t cmdPixelDecode(int32_t argc, char** argv) {

    std::string inTsv, inIndex, modelFile, anchorFile, outFile, tmpDir, dictFile, weightFile;
    int nThreads = 1, seed = -1, debug = 0, verbose = 0;
    int icol_x, icol_y, icol_feature, icol_val;
    double hexSize = -1, hexGridDist = -1;
    double radius = -1, anchorDist = -1;
    double mDelta = 1e-3;
    int32_t nMoves = -1, minInitCount = 10, topK = 3;
    double pixelResolution = 1, defaultWeight = 0.;
    bool outputOritinalData = false;
    bool featureIsIndex = false;
    bool coordsAreInt = false;
    bool useTicketSystem = false;
    int32_t floatCoordDigits = 4, probDigits = 4;

    ParamList pl;
    // Input Options
    pl.add_option("in-tsv", "Input TSV file. Header must begin with #", inTsv)
      .add_option("in-index", "Input index file", inIndex)
      .add_option("model", "Model file", modelFile)
      .add_option("anchor", "Anchor file", anchorFile)
      .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x)
      .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y)
      .add_option("coords-are-int", "If the coordinates are integers, otherwise assume they are floats", coordsAreInt)
      .add_option("icol-feature", "Column index for feature (0-based)", icol_feature)
      .add_option("icol-val", "Column index for count/value (0-based)", icol_val)
      .add_option("feature-is-index", "If the feature column contains integer indices, otherwise assume it contains feature names", featureIsIndex)
      .add_option("default-weight", "Default weight for features not in the weight file", defaultWeight)
      .add_option("feature-weights", "Input weights file", weightFile)
      .add_option("pixel-res", "Resolution of pixel level inference", pixelResolution)
      .add_option("hex-size", "Hexagon size (side length)", hexSize)
      .add_option("hex-grid-dist", "Hexagon grid distance (center-to-center distance)", hexGridDist)
      .add_option("anchor-dist", "Distance between adjacent anchors", anchorDist)
      .add_option("mean-change-tol", "Mean change of document-topic probability tolerance for convergence (default: 1e-3)", mDelta)
      .add_option("radius", "Radius", radius)
      .add_option("n-moves", "Number of steps to slide on each axis to create anchors", nMoves)
      .add_option("threads", "Number of threads to use (default: 1)", nThreads)
      .add_option("seed", "Random seed", seed);
    // Output Options
    pl.add_option("out", "Output TSV file", outFile)
      .add_option("output-original", "Output original data points (pixels with feature values) together with the pixel level factor results", outputOritinalData)
      .add_option("use-ticket-system", "Use ticket system to ensure predictable output order", useTicketSystem)
      .add_option("temp-dir", "Directory to store temporary files", tmpDir)
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

    // read model matrix
    std::ifstream modelIn(modelFile, std::ios::in);
    if (!modelIn) {
        error("Error opening model file: %s", modelFile.c_str());
        return 1;
    }
    std::string line;
    std::vector<std::string> featureNames, factorNames, tokens;
    std::getline(modelIn, line);
    split(tokens, "\t", line);
    int32_t K = tokens.size() - 1;
    factorNames.resize(K);
    for (int32_t i = 0; i < K; ++i) {
        factorNames[i] = tokens[i + 1];
    }
    std::vector<std::vector<double>> modelValues;
    while (std::getline(modelIn, line)) {
        split(tokens, "\t", line);
        if (tokens.size() != K + 1) {
            error("Error reading model file at line ", line.c_str());
        }
        featureNames.push_back(tokens[0]);
        std::vector<double> values(K);
        for (int32_t i = 0; i < K; ++i) {
            values[i] = std::stod(tokens[i + 1]);
        }
        modelValues.push_back(values);
    }
    modelIn.close();
    uint32_t nFeatures = featureNames.size();
    MatrixXd model(K, nFeatures);
    for (uint32_t i = 0; i < nFeatures; ++i) {
        for (int32_t j = 0; j < K; ++j) {
            model(j, i) = modelValues[i][j];
        }
    }
    notice("Read %zu features and %d factors from model file", nFeatures, K);

    HexGrid hexGrid(hexSize);
    TileReader tileReader(inTsv, inIndex, -1, coordsAreInt);
    if (!tileReader.isValid()) {
        error("Error in input tiles: %s", inTsv.c_str());
    }
    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val, dictFile);
    if (!featureIsIndex) {
        for (size_t i = 0; i < featureNames.size(); ++i) {
            parser.featureDict[featureNames[i]] = i;
        }
        parser.isFeatureDict = true;
    }
    if (!weightFile.empty()) {
        parser.readWeights(weightFile, defaultWeight, nFeatures);
    }
    notice("Initialized tile reader");

    LatentDirichletAllocation lda(K, nFeatures, seed, 1, 0, model, 100, mDelta);
    notice("Initialized anchor model with %d features and %d factors", nFeatures, K);

    if (coordsAreInt) {
        Tiles2Minibatch<int32_t> tiles2minibatch(nThreads, radius, outFile, tmpDir, lda, tileReader, parser, hexGrid, nMoves, seed, minInitCount, 0.7, pixelResolution, nFeatures, 0, topK, verbose, debug);
        tiles2minibatch.setOutputOptions(outputOritinalData, useTicketSystem);
        tiles2minibatch.setFeatureNames(featureNames);
        tiles2minibatch.setOutputCoordDigits(floatCoordDigits);
        tiles2minibatch.setOutputProbDigits(probDigits);
        if (!anchorFile.empty()) {
            int32_t nAnchors = tiles2minibatch.loadAnchors(anchorFile);
            notice("Loaded %d valid anchors", nAnchors);
        }
        tiles2minibatch.run();
    } else {
        Tiles2Minibatch<float> tiles2minibatch(nThreads, radius, outFile, tmpDir, lda, tileReader, parser, hexGrid, nMoves, seed, minInitCount, 0.7, pixelResolution, nFeatures, 0, topK, verbose, debug);
        tiles2minibatch.setOutputOptions(outputOritinalData, useTicketSystem);
        tiles2minibatch.setOutputCoordDigits(floatCoordDigits);
        tiles2minibatch.setOutputProbDigits(probDigits);
        tiles2minibatch.setFeatureNames(featureNames);
        if (!anchorFile.empty()) {
            int32_t nAnchors = tiles2minibatch.loadAnchors(anchorFile);
            notice("Loaded %d valid anchors", nAnchors);
        }
        tiles2minibatch.run();
    }

    return 0;
}
