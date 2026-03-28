#include "punkst.h"
#include "tiles2slda.hpp"
#include "tiles2nmf.hpp"
#include <algorithm>
#include <limits>

namespace {

std::vector<float> build_evenly_spaced_thin3d_zlevels(double zMin, double zMax, int32_t nLevels) {
    if (!(zMax > zMin) || nLevels <= 1) {
        error("%s: invalid thin 3D z-level configuration", __func__);
    }
    std::vector<float> zLevels(static_cast<size_t>(nLevels));
    const double zStep = (zMax - zMin) / static_cast<double>(nLevels);
    for (int32_t i = 0; i < nLevels; ++i) {
        zLevels[static_cast<size_t>(i)] = static_cast<float>(zMin + (static_cast<double>(i) + 0.5) * zStep);
    }
    return zLevels;
}

double nth_nearest_zlevel_distance(const std::vector<float>& zLevels, double z, int32_t nPick) {
    std::vector<double> dists;
    dists.reserve(zLevels.size());
    for (float level : zLevels) {
        dists.push_back(std::abs(z - static_cast<double>(level)));
    }
    std::nth_element(dists.begin(), dists.begin() + (nPick - 1), dists.end());
    return dists[static_cast<size_t>(nPick - 1)];
}

double thin3d_default_zreach(const std::vector<float>& zLevels, double zMin, double zMax, int32_t nPick = -1) {
    if (zLevels.size() <= 1) {
        return 0.0;
    }
    if (nPick <= 0) {
        nPick = static_cast<int32_t>(zLevels.size() * 0.7);
    }
    nPick = std::min<int32_t>(std::max<int32_t>(nPick, 1), static_cast<int32_t>(zLevels.size()));
    std::vector<double> probes;
    probes.reserve(zLevels.size() * 2 + 2);
    probes.push_back(zMin);
    probes.push_back(zMax);
    for (size_t i = 0; i < zLevels.size(); ++i) {
        probes.push_back(static_cast<double>(zLevels[i]));
        if (i + 1 < zLevels.size()) {
            probes.push_back(0.5 * (static_cast<double>(zLevels[i]) + static_cast<double>(zLevels[i + 1])));
        }
    }
    double maxDist = 0.0;
    for (double z : probes) {
        maxDist = std::max(maxDist, nth_nearest_zlevel_distance(zLevels, z, nPick));
    }
    return maxDist;
}

} // namespace

int32_t cmdPixelDecode(int32_t argc, char** argv) {

    std::string inTsv, inIndex, modelFile, anchorFile, outFile, outPref, tmpDirPath, weightFile;
    std::string sampleList; // for multi-sample
    int nThreads = 1, seed = -1, debug_ = 0, verbose = 0;
    int icol_x, icol_y, icol_z = -1, icol_feature, icol_val;
    double hexSize = -1, hexGridDist = -1, anchorDist = -1;
    double decoderRadius = -1, halfLifeDist = 0.7;
    double zMin = std::numeric_limits<double>::quiet_NaN();
    double zMax = std::numeric_limits<double>::quiet_NaN();
    std::vector<float> thin3DZLevels;
    int32_t thin3DNZLevels = -1;
    bool ignoreOutsideZrange = false;
    bool thin3D = false;
    bool standard3D = false;
    int32_t nMoves = -1, topK = 3;
    double minInitCount = 10;
    double minCountAnchor = 5;
    double pixelResolution = -1., defaultWeight = 0.;
    double pixelResolutionZ = -1.;
    bool inMemory = false;
    bool outputBinary = false;
    bool outputOritinalData = false;
    bool singleFeaturePixel = false;
    bool singleMolecule = false;
    bool featureIsIndex = false;
    bool coordsAreInt = false;
    bool outputAnchor = false;
    bool useTicketSystem = false;
    int32_t floatCoordDigits = 2, probDigits = 4;
    std::vector<std::string> annoInts, annoFloats, annoStrs;
    int32_t maxIter = 100;
    double mDelta = 1e-3;

    std::string algo = "slda"; // or "nmf"
    double a0 = 1.0, b0 = 9.0;
    double pi0 = 0.1;
    std::string bgModelFile;
    bool outBgExpand = false;

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
      .add_option("icol-z", "Column index for z coordinate (0-based, optional)", icol_z)
      .add_option("coords-are-int", "If the coordinates are integers, otherwise assume they are floats", coordsAreInt)
      .add_option("icol-feature", "Column index for feature (0-based)", icol_feature, true)
      .add_option("icol-val", "Column index for count/value (0-based)", icol_val, true)
      .add_option("feature-is-index", "If the feature column contains integer indices, otherwise assume it contains feature names", featureIsIndex)
      .add_option("default-weight", "Default weight for features not in the weight file", defaultWeight)
      .add_option("feature-weights", "Input weights file", weightFile)
      .add_option("pixel-res", "Resolution of pixel level inference", pixelResolution)
      .add_option("pixel-res-z", "Resolution of pixel level inference in z dimension", pixelResolutionZ)
      .add_option("hex-size", "Hexagon size (side length)", hexSize)
      .add_option("hex-grid-dist", "Hexagon grid distance (center-to-center distance)", hexGridDist)
      .add_option("anchor-dist", "Distance between adjacent anchors", anchorDist)
      .add_option("half-life-dist", "Ratio to the radius where the anchor's weight decays to 0.5 (default: 0.7)", halfLifeDist)
      .add_option("thin-3D", "Activate the thin 3D path", thin3D)
      .add_option("standard-3D", "Activate the standard 3D path", standard3D)
      .add_option("zmin", "Minimum z coordinate", zMin)
      .add_option("zmax", "Maximum z coordinate", zMax)
      .add_option("thin-3d-z-levels", "Explicit z coordinates for thin 3D anchor levels", thin3DZLevels)
      .add_option("thin-3d-n-z-levels", "Number of evenly spaced z levels for thin 3D anchors", thin3DNZLevels)
      .add_option("ignore-outside-zrange", "Ignore observations with z coordinates outside the specified [zmin, zmax] range", ignoreOutsideZrange)
      .add_option("max-iter", "Maximum number of iterations (default: 100)", maxIter)
      .add_option("mean-change-tol", "Mean change of document-topic probability tolerance for convergence (default: 1e-3)", mDelta)
      .add_option("radius", "Support radius", decoderRadius)
      .add_option("n-moves", "Number of steps to slide on each axis to create anchors", nMoves)
      .add_option("threads", "Number of threads to use (default: 1)", nThreads)
      .add_option("seed", "Random seed", seed);
    // Background model options
    pl.add_option("background-model", "Background model file", bgModelFile)
      .add_option("bg-fraction-prior-a0", "Background fraction hyper-parameter a0 in pi~beta(a0, b0) (default: 2)", a0)
      .add_option("bg-fraction-prior-b0", "Background fraction hyper-parameter b0 in pi~beta(a0, b0) (default: 8)", b0)
      .add_option("bg-fraction", "Background fraction hyper-parameter, (use with --algo nmf)", pi0);
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
      .add_option("output-binary", "Output pixel level results in binary format", outputBinary)
      .add_option("output-original", "Output original data points (pixels with feature values) together with the pixel level factor results", outputOritinalData)
      .add_option("single-feature-pixel", "Enable one-feature-per-collapsed-pixel decoding (SLDA binary output only)", singleFeaturePixel)
      .add_option("single-molecule", "Enable raw single-molecule decoding without collapsing records to pixels (SLDA binary output only)", singleMolecule)
      .add_option("ext-col-ints", "Additional integer columns to carry over to output file, in the form of \"idx1:name1 idx2:name2 ...\" where 'idx' are 0-based column indices", annoInts)
      .add_option("ext-col-floats", "Additional float columns to carry over to output file, in the form of \"idx1:name1 idx2:name2 ...\" where 'idx' are 0-based column indices", annoFloats)
      .add_option("ext-col-strs", "Additional string columns to carry over to output file, in the form of \"idx1:name1:len1 idx2:name2:len2 ...\" where 'idx' are 0-based column indices and 'len' are maximum lengths of strings", annoStrs)
      .add_option("output-anchors", "Output anchor level info", outputAnchor)
      .add_option("use-ticket-system", "Use ticket system to ensure predictable output order", useTicketSystem)
      .add_option("temp-dir", "Directory to store temporary files", tmpDirPath, true)
      .add_option("in-memory", "Keep boundary buffers in memory instead of writing to temporary files", inMemory)
      .add_option("top-k", "Top K factors to output", topK)
      .add_option("min-init-count", "Minimum", minInitCount)
      .add_option("output-coord-digits", "Number of decimal digits to output for coordinates (only used for tsv output when input coordinates are float or --output-original is not set)", floatCoordDigits)
      .add_option("output-prob-digits", "Number of decimal digits to output for probabilities", probDigits)
      .add_option("output-bg-prob-expand", "Write one line per gene per pixel to include background probabilities (ignored if --output-original is set)", outBgExpand)
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
    if (thin3D && standard3D) {
        error("Use either --thin-3D or --standard-3D, not both");
    }
    if (outputBinary && outputOritinalData) {
        error("Cannot set both --output-binary and --output-original");
    }
    if (singleFeaturePixel && singleMolecule) {
        error("Use either --single-feature-pixel or --single-molecule, not both");
    }
    if (singleMolecule) {
        if (pixelResolution > 0 || pixelResolutionZ > 0)
            error("--pixel-res and --pixel-res-z should not be set in single molecule mode");
        if (pixelResolution <= 0) {
            pixelResolution = 1.;
            warning("--pixel-res is not set. Using default resolution of 1.0");
        }
    }

    FeatureSpecificMode featureSpecificMode = FeatureSpecificMode::Off;
    if (singleFeaturePixel) {
        featureSpecificMode = FeatureSpecificMode::SingleFeaturePixel;
    } else if (singleMolecule) {
        featureSpecificMode = FeatureSpecificMode::SingleMolecule;
    }
    if (featureSpecificMode != FeatureSpecificMode::Off) {
        if (algo != "slda") {
            error("feature-specific decoding modes are currently supported only with --algo slda");
        }
        if (outputOritinalData || outBgExpand) {
            error("feature-specific decoding modes do not support --output-original or --output-bg-prob-expand");
        }
        if (!outputBinary) {
            warning("feature-specific decoding modes currently support formatted binary output only");
            outputBinary = true;
        }
    }
    if (seed <= 0) {
        seed = std::random_device{}();
        notice("Using random seed %d", seed);
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

    // Validate geometry related parameters
    const bool use3D = thin3D || standard3D;
    const bool hasZMin = std::isfinite(zMin);
    const bool hasZMax = std::isfinite(zMax);
    const bool hasZRange = hasZMin || hasZMax;
    double standard3DBccSize = -1.0;
    double decoderPadding = -1.0;
    if (use3D) { // Shared checks for both 3D modes
        if (icol_z < 0)
            error("3D mode requires --icol-z");
        if (hasZMin != hasZMax)
            error("Both --zmin and --zmax must be provided together");
        if (hasZRange && !(zMax > zMin))
            error("--zmax must be greater than --zmin");
        if (thin3D && !hasZRange)
            error("Thin 3D requires both --zmin and --zmax");
        if (ignoreOutsideZrange && !hasZRange)
            error("--ignore-outside-zrange requires both --zmin and --zmax");
    } else { // 2D checks
        if (icol_z >= 0)
            error("--icol-z requires either --thin-3D or --standard-3D");
        if (hasZRange)
            error("--zmin/--zmax require 3D input");
        if (!thin3DZLevels.empty() || thin3DNZLevels > 0)
            error("--thin-3d-z-levels/--thin-3d-n-z-levels require --thin-3D");
    }
    if (thin3D) {
        const bool hasExplicitThin3DZLevels = !thin3DZLevels.empty();
        const bool hasThin3DNZLevels = thin3DNZLevels > 0;
        if (!hasExplicitThin3DZLevels && !hasThin3DNZLevels) {
            error("Thin 3D requires one of --thin-3d-z-levels or --thin-3d-n-z-levels");
        }
        if (hasExplicitThin3DZLevels) {
            if (hasThin3DNZLevels) {
                warning("Both --thin-3d-z-levels and --thin-3d-n-z-levels are set. Ignoring --thin-3d-n-z-levels.");
            }
        } else {
            if (thin3DNZLevels <= 1)
                error("--thin-3d-n-z-levels must be greater than 1");
            thin3DZLevels = build_evenly_spaced_thin3d_zlevels(zMin, zMax, thin3DNZLevels);
        }
    }
    if (standard3D) {
        if (anchorDist <= 0)
            error("Standard 3D: --anchor-dist is required");
        if (hasZRange && !ignoreOutsideZrange)
            notice("Standard 3D: z range is accepted but ignored unless --ignore-outside-zrange is set");
        standard3DBccSize = 2.0 * anchorDist / std::sqrt(3.0);
        if (decoderRadius <= 0)
            decoderRadius = standard3DBccSize * 1.2;
        notice("Standard 3D: using BCC grid with anchor distance %.2f", anchorDist);
    } else { // Shared checks for 2D and thin 3D
        if (hexSize <= 0 && hexGridDist <= 0)
            error("Hexagon size (--hex-size) or hexagon grid distance (--hex-grid-dist) must be provided");
        if (nMoves <= 0 && anchorDist <= 0)
            error("Number of grid shifts (--n-moves) or anchor distance (--anchor-dist) must be provided");
        if (hexSize > 0) {
            hexGridDist = hexSize * std::sqrt(3.0);
        } else if (hexGridDist > 0) {
            hexSize = hexGridDist / std::sqrt(3.0);
        }
        if (nMoves <= 0) {
            nMoves = std::max<int32_t>(static_cast<int32_t>(std::ceil(hexGridDist / anchorDist)), 1);
        } else {
            anchorDist = hexGridDist / nMoves;
        }
        if (thin3D && nMoves <= 1) {
            error("Thin 3D anchor initialization requires nMoves > 1");
        }
        if (decoderRadius <= 0) {
            if (thin3D) {
                const double zDist = thin3d_default_zreach(thin3DZLevels, zMin, zMax, -1);
                decoderRadius = std::sqrt(anchorDist * anchorDist + zDist * zDist) * 1.2;
            } else {
                decoderRadius = anchorDist * 1.2;
            }
            notice("Using radius %.2f", decoderRadius);
        }
    }
    const double geometryUnitSize = standard3D ? standard3DBccSize : hexSize;
    decoderPadding = decoderRadius + geometryUnitSize;

    // Set up model
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
    if (icol_z >= 0) {
        parser.setZ(static_cast<size_t>(icol_z));
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
    if (!featureIsIndex) {
        parser.setFeatureDict(featureNames);
    }
    if (!weightFile.empty()) {
        parser.readWeights(weightFile, defaultWeight, M_model);
    }
    if (featureSpecificMode != FeatureSpecificMode::Off && (parser.isExtended || parser.weighted)) {
        error("feature-specific decoding modes currently require standard input records and do not support --ext-col-* or feature weights");
    }
    notice("Initialized tile reader");

    MinibatchIoConfig ioConfig;
    ioConfig.input = parser.isExtended ? MinibatchInputMode::Extended : MinibatchInputMode::Standard;
    ioConfig.featureSpecificMode = featureSpecificMode;
    if (outputBinary) {
        ioConfig.output = MinibatchOutputMode::Binary;
    } else if (outputOritinalData) {
        ioConfig.output = MinibatchOutputMode::Original;
    } else {
        ioConfig.output = MinibatchOutputMode::Standard;
    }
    ioConfig.outputAnchor = outputAnchor;
    ioConfig.useTicketSystem = useTicketSystem;
    ioConfig.nativeRegularTiles = outputBinary;

    VectorXf eta0;
    bool fit_background = false;
    if (!bgModelFile.empty()) {
        eta0 = VectorXf::Zero(M_model);
        std::ifstream fin(bgModelFile);
        if (!fin.is_open()) {
            error("Cannot open background model file %s", bgModelFile.c_str());
        }
        std::string line;
        std::vector<std::string> tokens;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') {continue;}
            split(tokens, "\t ", line, 3, true, true, true);
            if (tokens.size() < 2) {continue;}
            auto it = parser.featureDict.find(tokens[0]);
            if (it == parser.featureDict.end()) {continue;}
            if (!str2float(tokens[1], eta0(it->second))) {
                error("Invalid value in background model file (%s)", line.c_str());
            }
        }
        fit_background = true;
    }

    auto configure_decoder = [&](auto& decoder, const std::string& anchorFile, bool allowExternalAnchors) {
        decoder.setOutputCoordDigits(floatCoordDigits);
        decoder.setOutputProbDigits(probDigits);
        if (use3D) {
            decoder.set3Dparameters(thin3D, zMin, zMax, pixelResolutionZ, ignoreOutsideZrange, anchorDist, thin3DZLevels);
        }
        decoder.setFeatureNames(featureNames);
        if (!anchorFile.empty()) {
            if (allowExternalAnchors) {
                int32_t nAnchors = decoder.loadAnchors(anchorFile);
                notice("Loaded %d valid anchors", nAnchors);
            } else {
                notice("Ignoring --anchor for 3D input with --algo slda");
            }
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
                    nThreads, decoderRadius, decoderPadding, ds.outPref, tmpDirPath,
                    emPois, tileReader, parser, ioConfig,
                    seed, minInitCount, halfLifeDist, pixelResolution,
                    topK, verbose, debug_, geometryUnitSize, nMoves, inMemory);
                if (fit_background) {
                    decoder.set_background_model(pi0, eta0, outBgExpand);
                }
                configure_decoder(decoder, ds.anchorFile, true);
                decoder.run();
            } else {
                Tiles2NMF<float> decoder(
                    nThreads, decoderRadius, decoderPadding, ds.outPref, tmpDirPath,
                    emPois, tileReader, parser, ioConfig,
                    seed, minInitCount, halfLifeDist, pixelResolution,
                    topK, verbose, debug_, geometryUnitSize, nMoves, inMemory);
                if (fit_background) {
                    decoder.set_background_model(pi0, eta0, outBgExpand);
                }
                configure_decoder(decoder, ds.anchorFile, true);
                decoder.run();
            }
        }
        return 0;
    }

    RowMajorMatrixXd betaT = beta.transpose(); // K x M
    LatentDirichletAllocation lda(betaT, seed, 1);
    lda.set_svb_parameters(maxIter, mDelta);
    lda.topic_names_ = factorNames;
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
            Tiles2SLDA<int32_t> tiles2slda(nThreads, decoderRadius, decoderPadding, ds.outPref, tmpDirPath, lda, tileReader, parser, ioConfig, seed, minInitCount, halfLifeDist, pixelResolution, 0, topK, verbose, debug_, geometryUnitSize, nMoves, inMemory);
            if (fit_background) {
                tiles2slda.set_background_prior(eta0, a0, b0, outBgExpand);
            }
            configure_decoder(tiles2slda, ds.anchorFile, !use3D);
            tiles2slda.run();
        } else {
            Tiles2SLDA<float> tiles2slda(nThreads, decoderRadius, decoderPadding, ds.outPref, tmpDirPath, lda, tileReader, parser, ioConfig, seed, minInitCount, halfLifeDist, pixelResolution, 0, topK, verbose, debug_, geometryUnitSize, nMoves, inMemory);
            if (fit_background) {
                tiles2slda.set_background_prior(eta0, a0, b0, outBgExpand);
            }
            configure_decoder(tiles2slda, ds.anchorFile, !use3D);
            tiles2slda.run();
        }
    }

    return 0;
}
