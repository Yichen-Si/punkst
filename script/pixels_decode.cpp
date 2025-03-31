#include "punkst.h"
#include "tiles2minibatch.hpp"

int32_t cmdPixelDecode(int32_t argc, char** argv) {

    std::string inTsv, inIndex, modelFile, outFile, tmpDir, dictFile, weightFile;
    int nThreads = 1, seed = -1, debug = 0, verbose = 0;
    int icol_x, icol_y, icol_feature, icol_val;
    double hexSize = -1, hexGridDist = -1;
    double radius = -1, anchorDist = -1;
    int32_t nMoves = -1, minInitCount = 10, topK = 3;
    double pixelResolution = -1, defaultWeight = 0.;

	paramList pl;
	BEGIN_LONG_PARAMS(longParameters)
		LONG_PARAM_GROUP("Input options", NULL)
        LONG_STRING_PARAM("in-tsv", &inTsv, "Input TSV file. Header must begin with #")
        LONG_STRING_PARAM("in-index", &inIndex, "Input index file")
        LONG_STRING_PARAM("model", &modelFile, "Model file")
        LONG_INT_PARAM("icol-x", &icol_x, "Column index for x coordinate (0-based)")
        LONG_INT_PARAM("icol-y", &icol_y, "Column index for y coordinate (0-based)")
        LONG_INT_PARAM("icol-feature", &icol_feature, "Column index for feature (0-based)")
        LONG_INT_PARAM("icol-val", &icol_val, "Column index for count/value (0-based)")
        LONG_STRING_PARAM("feature-dict", &dictFile, "If feature column is not integer, provide a dictionary/list of all possible values")
        LONG_DOUBLE_PARAM("default-weight", &defaultWeight, "Default weight for features not in the weight file")
        LONG_STRING_PARAM("feature-weights", &weightFile, "Input weights file")
        LONG_DOUBLE_PARAM("pixel-res", &pixelResolution, "Pixel resolution")
        LONG_DOUBLE_PARAM("hex-size", &hexSize, "Hexagon size (side length)")
        LONG_DOUBLE_PARAM("hex-grid-dist", &hexGridDist, "Hexagon grid distance (center-to-center distance)")
        LONG_DOUBLE_PARAM("anchor-dist", &anchorDist, "Distance between adjacent anchors")
        LONG_DOUBLE_PARAM("radius", &radius, "Radius")
        LONG_INT_PARAM("n-moves", &nMoves, "Number of steps to slide on each axis to create anchors")
        LONG_INT_PARAM("threads", &nThreads, "Number of threads to use (default: 1)")
        LONG_INT_PARAM("seed", &seed, "Random seed")
		LONG_PARAM_GROUP("Output Options", NULL)
        LONG_STRING_PARAM("out", &outFile, "Output TSV file")
        LONG_STRING_PARAM("temp-dir", &tmpDir, "Directory to store temporary files")
        LONG_INT_PARAM("min-init-count", &minInitCount, "Minimum")
        LONG_INT_PARAM("verbose", &verbose, "Verbose")
        LONG_INT_PARAM("debug", &debug, "Debug")
    END_LONG_PARAMS();
    pl.Add(new longParams("Available Options", longParameters));
    pl.Read(argc, argv);
    pl.Status();

    if (hexSize <= 0) {
        if (hexGridDist <= 0) {
            error("Hexagon size or hexagon grid distance must be provided");
        }
        hexSize = hexGridDist / sqrt(3) * 2;
    } else {
        hexGridDist = hexSize * sqrt(3) / 2;
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
    TileReader tileReader(inTsv, inIndex);
    if (!tileReader.isValid()) {
        error("Error in input tiles: %s", inTsv.c_str());
    }
    lineParserLocal parser(icol_x, icol_y, icol_feature, icol_val, dictFile);
    if (!weightFile.empty()) {
        parser.readWeights(weightFile, defaultWeight, nFeatures);
    }
    notice("Initialized tile reader");

    LatentDirichletAllocation lda(K, nFeatures, seed, 1, 0, model);
    notice("Initialized anchor model");

    Tiles2Minibatch tiles2minibatch(nThreads, radius, outFile, tmpDir, lda, tileReader, parser, hexGrid, nMoves, seed, 20, 0.7, pixelResolution, nFeatures, 0, topK, verbose);
    tiles2minibatch.setFeatureNames(featureNames);
    tiles2minibatch.run();

    return 0;
}
