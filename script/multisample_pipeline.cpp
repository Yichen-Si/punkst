#include "punkst.h"
#include "preprocess_options.hpp"
#include "tiles2bins.hpp"
#include <regex>

int32_t cmdMultiSample(int32_t argc, char** argv) {
    std::string inTsvListFile, outDir;
    std::string jointPref = "";
    std::vector<std::string> inTsvList;
    Pts2TilesOptions ptsOpts;
    int debug = 0, verbose = 1000000;
    int minTotalCountPerSample = 1;

    std::vector<double> hexSizes, hexGridDists, bccSizes, bccGridDists;
    std::string anchorListFile;
    Tiles2HexOptions hexOpts;
    std::vector<std::string>& anchorList = hexOpts.anchorFiles;
    bool tiles2hex_only = false;
    bool forceTileSize = false;
    std::string include_ftr_regex;
    std::string exclude_ftr_regex;
    bool overwrite = false;

    ParamList pl;
    pl.add_option("in-tsv", "List of input TSV files separated by space", inTsvList)
        .add_option("in-tsv-list", "A file containing the ID and the path to the input transcript file for each sample (tab/space delimited, one sample per line)", inTsvListFile);
    ptsOpts.addProcessingOptions(pl, true, true, false, false, true)
        .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
        .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex);
    // tiles2hex Options
    pl.add_option("min-total-count-per-sample", "Minimum total gene count per sample (default: 1) to include in the joint model", minTotalCountPerSample);
    hexOpts.addMultisampleOptions(pl, hexSizes, hexGridDists, bccSizes, bccGridDists, anchorListFile);
    // Output Options
    pl.add_option("out-dir", "Output base director", outDir, true)
        .add_option("temp-dir", "Directory to store temporary files", ptsOpts.tmpDir, true)
        .add_option("out-joint-pref", "Prefix for joint analysis outputs", jointPref)
        .add_option("tiles2hex-only", "Only run tiles2hex and merge the output files. Note: if set, the file in --in-tsv-list should contain the following columns instead: sample ID, path to the tiled pixel level data, path to the corresponding index, path to the per-sample feature count file.", tiles2hex_only)
        .add_option("force-tile-size", "Disable tile size checks (we strongly advice against using this option)", forceTileSize)
        .add_option("overwrite", "Overwrite existing output files", overwrite)
        .add_option("verbose", "Verbose", verbose)
        .add_option("debug", "Debug", debug);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing multisample options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    // Determine the number of threads to use.
    unsigned int nThreads = ptsOpts.resolveThreads();
    hexOpts.nThreads = static_cast<int32_t>(nThreads);
    hexOpts.tmpDir = ptsOpts.tmpDir;
    notice("Using %u threads for processing", nThreads);

    // Check parameters
    if (inTsvList.empty() && inTsvListFile.empty()) {
        error("No input specified. Use --in-tsv or --in-tsv-list to specify input files.");
    }
    if (ptsOpts.skip_last_is_header && ptsOpts.nskip <= 0) {
        error("--skip-last-is-header requires --skip to be greater than 0");
    }
    if (tiles2hex_only) {
        if (!inTsvList.empty())
            error("If --tiles2hex-only is set, --in-tsv is not allowed and --in-tsv-list is required. The file in --in-tsv-list should contain the following columns: sample ID, path to the tiled pixel level data, path to the corresponding index, path to the per-sample feature count file (all output from pts2tiles).");
    }
    if (hexOpts.featureInfoOptions.idfQ < 0.0 || hexOpts.featureInfoOptions.idfQ > 100.0) {
        error("--idf-q must be between 0 and 100");
    }
    if (hexOpts.featureInfoOptions.idfPower <= 0.0) {
        error("--idf-power must be positive");
    }
    if (hexOpts.featureInfoOptions.idfMin < 0.0 || hexOpts.featureInfoOptions.idfMin > 1.0) {
        error("--idf-min must be between 0 and 1");
    }
    if (hexOpts.featureInfoOptions.idfMax <= 0.0) {
        error("--idf-max must be positive");
    }

    ptsOpts.normalizeScales();
    const int32_t out_icol_x = tiles2hex_only ? ptsOpts.icol_x : ptsOpts.remapColumn(ptsOpts.icol_x);
    const int32_t out_icol_y = tiles2hex_only ? ptsOpts.icol_y : ptsOpts.remapColumn(ptsOpts.icol_y);
    const int32_t out_icol_z = tiles2hex_only ? ptsOpts.icol_z : ptsOpts.remapColumn(ptsOpts.icol_z);
    const int32_t out_icol_feature = tiles2hex_only ? ptsOpts.icol_feature : ptsOpts.remapColumn(ptsOpts.icol_feature);
    const std::vector<int32_t> out_icol_ints = tiles2hex_only ? ptsOpts.icol_ints : ptsOpts.remapIntColumns();
    const bool useBcc3D = ptsOpts.icol_z >= 0 && (!bccGridDists.empty() || !bccSizes.empty());
    if (!useBcc3D && hexGridDists.empty() && hexSizes.empty()) {
        warning("Neither --hex-grid-dist nor --hex-size is specified, the program exits after generating tiled pixels without generating hexagons.");
    } else {
        Tiles2HexOptions::resolveGridVectors(useBcc3D, hexSizes, hexGridDists, bccSizes, bccGridDists, hexOpts.anchorFiles.empty() && anchorListFile.empty());
    }
    if (!tiles2hex_only) {
        if (ptsOpts.tileSize <= 0) {
            error("--tile-size must be specified and greater than 0 (500~1000 microns is a good choice for most datasets)");
        }
        bool toosmall = false;
        const auto& unitSizes = useBcc3D ? bccSizes : hexSizes;
        if (!unitSizes.empty()) {
            double minUnitSize = *std::min_element(unitSizes.begin(), unitSizes.end());
            toosmall = ptsOpts.tileSize < minUnitSize * 20;
        } else {
            toosmall = ptsOpts.tileSize < 100;
        }
        if (toosmall) {
            warning("--tile-size seems too small, this is likely to cause the failure of pixel level decoding. (50~100x the hexagon's size length or 500~1000 microns is a good choice for most datasets)");
            if (!forceTileSize) {
                error("If you are certain the --tile-size is what you want, add flag --force-tile-size.");
            }
        }
    }
    if (!outDir.empty() && (outDir.back() == '/' || outDir.back() == '\\')) {
        outDir.pop_back();
    }
    if (!jointPref.empty() && jointPref.back() != '.') {
        jointPref += ".";
    }
    bool check_include = !include_ftr_regex.empty();
    bool check_exclude = !exclude_ftr_regex.empty();
    std::regex regex_include(include_ftr_regex);
    std::regex regex_exclude(exclude_ftr_regex);
    std::vector<std::string> samples, outDirs, featureTsvs, tileTsvs, tileIndexes;
    std::unordered_map<std::string, uint32_t> sampleIdx;
    size_t S = 0;

{ // 0) Gather input files
    // population samples and sample-specific file paths
    if (!inTsvListFile.empty()) {
        inTsvList.clear();
        std::ifstream inFile(inTsvListFile);
        if (!inFile) {
            error("Error opening input TSV file list: %s", inTsvListFile.c_str());
        }
        std::string line;
        while (std::getline(inFile, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t ", line);
            if (tokens.size() < 2) {
                error("Invalid line in --in-tsv-list (less than 2 tokens): %s", line.c_str());
            }
            samples.push_back(tokens[0]);
            if (tiles2hex_only) {
                if (tokens.size() < 4) {
                    error("Invalid line in --in-tsv-list (less than 4 tokens): %s", line.c_str());
                }
                tileTsvs.push_back(tokens[1]);
                tileIndexes.push_back(tokens[2]);
                featureTsvs.push_back(tokens[3]);
            } else {
                inTsvList.push_back(tokens[1]);
            }
        }
    } else {
        // Try to extract sample identifiers
        std::unordered_map<std::string, int> sampleNames;
        for (auto& inTsv : inTsvList) {
            std::string sn = inTsv.substr(inTsv.find_last_of("/\\") + 1);
            std::string::size_type const idx(sn.find_last_of('.'));
            if (idx != std::string::npos) {
                sn = sn.substr(0, idx);
            }
            auto it = sampleNames.find(sn);
            if (it == sampleNames.end()) {
                sampleNames[sn] = 1;
                samples.push_back(sn);
            } else {
                it->second++;
                sn += "_" + std::to_string(it->second); // Append a suffix
                samples.push_back(sn);
            }
        }
    }
    // Create sample-specific output directories
    std::filesystem::path sampleDirBase = std::filesystem::path(outDir) / "samples";
    if (!std::filesystem::exists(sampleDirBase)) {
        std::filesystem::create_directories(sampleDirBase);
    }
    for (auto& sn : samples) {
        std::filesystem::path sampleDir = sampleDirBase / sn;
        if (!std::filesystem::exists(sampleDir)) {
            std::filesystem::create_directories(sampleDir);
        }
        outDirs.push_back(sampleDir.string());
        sampleIdx[sn] = S++;
    }
    if (samples.empty()) {
        error("No input specified.");
    }
    if (!anchorListFile.empty()) {
        anchorList.clear();
        anchorList.resize(S);
        std::ifstream anchorFile(anchorListFile);
        if (!anchorFile) {
            error("Error opening anchor list file: %s", anchorListFile.c_str());
        }
        std::string line;
        while (std::getline(anchorFile, line)) {
            if (line.empty() || line[0] == '#') continue;
            std::vector<std::string> tokens;
            split(tokens, "\t ", line);
            if (tokens.size() < 2) {
                error("Invalid line in anchor list file: %s", line.c_str());
            }
            auto it = sampleIdx.find(tokens[0]);
            if (it == sampleIdx.end()) {
                warning("Sample %s appears in anchor list but not in input TSV list, skipping", tokens[0].c_str());
                continue;
            }
            anchorList[it->second] = tokens[1];
        }
        int32_t nAnchors = 0;
        for (const auto& anchor : anchorList) {
            if (!anchor.empty()) nAnchors++;
        }
        if (nAnchors == 0) {
            error("No anchors found in anchor list file: %s. Make sure the first column contains sample IDs that match whose in --in-tsv-list", anchorListFile.c_str());
        }
    }
    if (!(anchorList.size() == 0 || anchorList.size() == S)) {
        error("Anchor list size (%zu) does not match number of samples (%zu)", anchorList.size(), S);
    }
    if (anchorList.size() > 0 && (hexOpts.radius.empty() || hexOpts.radius.front() < 0)) {
        error("--radius (a positive number) must be specified when using anchor files");
    }
}
    notice("Found %zu samples", S);
    if (!tiles2hex_only && std::find(inTsvList.begin(), inTsvList.end(), "-") != inTsvList.end() && S > 1) {
        error("--in-tsv - can only be used with one sample in multisample-prepare");
    }

if (!tiles2hex_only) { // 1) Run pts2tiles on each sample
    std::string fileList = outDir + "/" + jointPref + "persample_file_list.tsv";
    std::ofstream fileListOut(fileList);
    for (size_t s = 0; s < S; ++s) {
        const std::string& inTsv = inTsvList[s];
        const std::string& sn = samples[s];
        std::string outPref = outDirs[s] + "/" + sn + ".tiled";
        tileTsvs.push_back(outPref + ".tsv");
        tileIndexes.push_back(outPref + ".index");
        featureTsvs.push_back(outPref + ".features.tsv");
        if (!overwrite && std::filesystem::exists(outPref + ".tsv") &&
                          std::filesystem::exists(outPref + ".index")) {
            notice("pts2tiles output is present for sample %s. To overwrite, add --overwrite", sn.c_str());
            continue;
        }
        notice("[multisample] Running pts2tiles for sample %s", sn.c_str());
        ptsOpts.inTsv = inTsv;
        ptsOpts.outPref = outPref;
        Pts2Tiles pts2Tiles = ptsOpts.makeRunner(nThreads);
        if (!pts2Tiles.run()) {
            return 1;
        }
        fileListOut << sn << "\t" << outPref << ".tsv\t" << outPref << ".index\n";
    }
    fileListOut.close();
    notice("[multisample] Step 1 completed: pre-processed transcript files for %zu samples.", S);
    notice("[multisample] Paths to pre-processed sample-specific transcript files written to: %s", fileList.c_str());
}

    std::unordered_map<std::string, std::vector<uint64_t>> featCounts;
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> unionFeatures;
    size_t n_shared = 0;
{ // 1.5) Decide a partially shared feature dictionary
    // a) Read each sample’s feature counts
    for (size_t s = 0; s < S; ++s) {
        std::string& featFile = featureTsvs[s];
        std::ifstream fin(featFile);
        if (!fin) error("Cannot open feature file: %s", featFile.c_str());
        std::string feature;
        uint64_t cnt;
        std::string line;
        std::vector<std::string> tokens;
        while (std::getline(fin, line)) {
            if (line.empty() || line[0] == '#') continue;
            split(tokens, "\t", line);
            if (tokens.size() < 2) continue; // Skip invalid lines
            if (!str2num<uint64_t>(tokens[1], cnt)) {
                warning("Skip invalid feature line: %s", line.c_str());
                continue;
            }
            feature = tokens[0];
            bool include = !check_include || std::regex_match(feature, regex_include);
            bool exclude =  check_exclude && std::regex_match(feature, regex_exclude);
            if (!include || exclude) { // Skip features based on regex filters
                continue;
            }
            auto it = featCounts.find(feature);
            if (it == featCounts.end()) {
                // first time seeing this feature → init vector<S+1> with zeros
                std::vector<uint64_t> vec(S + 1, 0);
                vec[s] = cnt;
                vec[S] = cnt;
                featCounts.emplace(std::move(feature), std::move(vec));
            } else {
                it->second[s] = cnt;
                it->second[S] += cnt;
            }
        }
    }
    // b) Pick “shared” features: featCounts[f][s] >= minTotalCountPerSample ∀s
    unionFeatures.reserve(featCounts.size());
    for (auto &p : featCounts) {
        uint64_t x = *std::min_element(p.second.begin(), p.second.begin() + S);
        unionFeatures.emplace_back(p.first, std::make_pair(x, p.second[S]));
        if (x >= minTotalCountPerSample) {
            n_shared++;
        }
    }
    std::sort(unionFeatures.begin(), unionFeatures.end(),
              [](const auto &a, const auto &b) {
                  if (a.second.first != b.second.first) {
                      return a.second.first > b.second.first;
                  } else {
                      return a.second.second > b.second.second;
                  }
              });
    // c) Write feature lists
    std::filesystem::path unionFile = std::filesystem::path(outDir) / (jointPref + "union_features.tsv");
    std::ofstream unionOut(unionFile);
    std::filesystem::path sharedFile = std::filesystem::path(outDir) / (jointPref + "features.tsv");
    std::ofstream sharedOut(sharedFile);
    if (!sharedOut || !unionOut) error("Cannot write shared or union feature file: %s", unionFile.c_str());
    std::vector<std::ofstream> sampleOuts(S);
    for (size_t s = 0; s < S; ++s) {
        std::string featFile = outDirs[s] + "/" + samples[s] + ".features.tsv";
        featureTsvs[s] = featFile;
        sampleOuts[s].open(featFile);
        if (!sampleOuts[s]) error("Cannot write feature file: %s", featFile.c_str());
    }
    // header
    unionOut << "#feature\ttotal_count";
    for (auto &sn : samples) unionOut << "\t" << sn;
    unionOut << "\n";
    sharedOut << "#feature\ttotal_count\n";
    for (auto &feat : unionFeatures) {
        const auto &vec = featCounts[feat.first];
        unionOut << feat.first << "\t" << feat.second.second;
        for (size_t s = 0; s < S; ++s) {
            unionOut << "\t" << vec[s];
        }
        unionOut << "\n";
        if (vec[S] >= minTotalCountPerSample) {
            sharedOut << feat.first << "\t" << feat.second.second << "\n";
            for (size_t s = 0; s < S; ++s) {
                sampleOuts[s] << feat.first << "\t" << vec[s] << "\n";
            }
        } else {
            for (size_t s = 0; s < S; ++s) {
                if (vec[s] > 0) {
                    sampleOuts[s] << feat.first << "\t" << vec[s] << "\n";
                }
            }
        }
    }
    notice("[multisample] Step 1.5 completed,selected %zu/%zu shared features for joint analysis", n_shared, unionFeatures.size());
}

    const auto& activeUnitSizes = useBcc3D ? bccSizes : hexSizes;
    if (activeUnitSizes.empty()) {
        return 0;
    }
    std::unordered_map<std::string, uint32_t> featureDict;
    uint32_t nFeatures = 0;
    for (auto & kv : unionFeatures) {
        featureDict[kv.first] = nFeatures++;
    }
{ // 2) Run tiles2hex
    for (auto &unitSize : activeUnitSizes) {
        std::string hexIden = useBcc3D ?
            "bcc_" + std::to_string(static_cast<int32_t>(std::round(unitSize * sqrt(3.0) / 2.0))) :
            "hex_" + std::to_string(static_cast<int32_t>(std::round(unitSize * sqrt(3.0))));
        HexGrid hexGrid(unitSize);
        std::string dfile;
        for (size_t s = 0; s < S; ++s) { // 2) Run tiles2hex on each sample
            std::string outFile = outDirs[s] + "/" + samples[s] + "." + hexIden + ".txt";
            std::string outJson = outDirs[s] + "/" + samples[s] + "." + hexIden + ".json";
            std::string outPref = outDirs[s] + "/" + samples[s] + "." + hexIden;
            bool featureInfoExists = true;
            const size_t nModalSidecars = out_icol_ints.empty() ? 1 : out_icol_ints.size();
            for (size_t modal = 0; modal < nModalSidecars; ++modal) {
                const std::string outFeatureInfo = nModalSidecars == 1
                    ? outPref + ".feature.stats.tsv"
                    : outPref + ".modal" + std::to_string(modal) + ".feature.stats.tsv";
                featureInfoExists = featureInfoExists && std::filesystem::exists(outFeatureInfo);
            }
            if (!overwrite && std::filesystem::exists(outFile) &&
                              std::filesystem::exists(outJson) &&
                              featureInfoExists) {
                notice("tiles2hex output is present for sample %s and size %.1f (%s).\nTo overwrite, add --overwrite", samples[s].c_str(), unitSize, outFile.c_str());
                continue;
            }
            TileReader tileReader(tileTsvs[s], tileIndexes[s]);
            if (!tileReader.isValid()) {
                error("Error opening input file: %s", tileTsvs[s].c_str());
            }
            lineParser parser(out_icol_x, out_icol_y, out_icol_feature, out_icol_ints, dfile, nullptr, true);
            if (useBcc3D) {
                parser.setZ(static_cast<size_t>(out_icol_z));
            }
            parser.setFeatureDict(featureDict);
            if (anchorList.size() != S || anchorList[s].empty()) {
                Tiles2Hex tiles2Hex(nThreads, ptsOpts.tmpDir, outFile, hexGrid,
                    tileReader, parser, hexOpts.min_counts, hexOpts.seed,
                    useBcc3D ? unitSize : -1.0, hexOpts.featureInfoOptions);
                if (!tiles2Hex.run()) {
                    error("Error running tiles2hex for sample %s", samples[s].c_str());
                }
                tiles2Hex.writeMetadata();
            } else {
                std::vector<std::string> localAnchor = {anchorList[s]};
                std::vector<float> localRadius = {hexOpts.radius.front()};
                Tiles2UnitsByAnchor tiles2Hex(nThreads, ptsOpts.tmpDir, outFile,
                    hexGrid, tileReader, parser, localAnchor, localRadius,
                    hexOpts.min_counts, hexOpts.noBackground, hexOpts.seed,
                    hexOpts.featureInfoOptions);
                if (!tiles2Hex.run()) {
                    error("Error running tiles2hex for sample %s", samples[s].c_str());
                }
                tiles2Hex.writeMetadata();
            }
            // sort output file
            hexOpts.outFile = outFile;
            hexOpts.sortOutput(true);
        }
        notice("[multisample] Step 2 completed for size %.1f: created sample specific units", unitSize);

    { // 2.5) Create a merged hexagon file containing only the shared features
        std::string mergedFile = outDir + "/" + jointPref + hexIden + ".txt";
        nlohmann::json meta;
        if (useBcc3D) {
            meta["bcc_size"] = hexGrid.size;
            meta["coord_dim"] = 3;
        } else {
            meta["hex_size"] = hexGrid.size;
            meta["coord_dim"] = 2;
        }
        meta["random_key"] = 0;
        meta["sample"] = 1;
        meta["sample_list"] = samples;
        meta["offset_data"] = 2;
        meta["header_info"] = {"random_key", "sample"};
        meta["n_features"] = n_shared;
        std::unordered_map<std::string, uint32_t> dict;
        std::vector<std::string> sharedFeatureNames(n_shared);
        for(uint32_t i = 0; i < n_shared; ++i) {
            dict[unionFeatures[i].first] = i;
            sharedFeatureNames[i] = unionFeatures[i].first;
        }
        meta["dictionary"] = (nlohmann::json) dict;

        std::ofstream mergedOut(mergedFile);
        if (!mergedOut) {
            error("Error opening merged output file for output: %s", mergedFile.c_str());
        }
        int32_t nUnits = 0;
        FeatureOccurrenceStats combinedOccurrence;
        for (size_t s = 0; s < S; ++s) {
            std::string localPref = outDirs[s] + "/" + samples[s] + "." + hexIden;
            HexReader reader(localPref + ".json");
            std::ifstream inFileStream(localPref + ".txt");
            if (!inFileStream) {
                error("Error opening hexagon file: %s", (localPref + ".txt").c_str());
            }
            std::string line;
            int32_t offset = reader.getOffset();
            while (std::getline(inFileStream, line)) {
                std::vector<std::string> tokens;
                split(tokens, "\t", line);
                if (tokens.size() < offset + 3) {
                    continue;
                }
                std::stringstream ss1, ss2;
                ss1 << tokens[0] << "\t" << std::to_string(s);
                int32_t nFeatures = 0, totCount = 0;
                std::map<uint32_t, uint32_t> sharedVals;
                for (size_t i = offset + 2; i < tokens.size(); ++i) {
                    std::vector<std::string> pair;
                    split(pair, " ", tokens[i]);
                    uint32_t u, v;
                    if (str2uint32(pair[0], u) && u < n_shared &&
                        str2uint32(pair[1], v)) {
                        ss2 << "\t" << tokens[i];
                        sharedVals[u] = v;
                        nFeatures++;
                        totCount += v;
                    }
                }
                if (totCount < hexOpts.minCount()) {
                    continue;
                }
                nUnits++;
                combinedOccurrence.observeUnit(sharedVals);
                ss1 << "\t" << nFeatures << "\t" << totCount;
                mergedOut << ss1.str() << ss2.str() << "\n";
            }
        }
        mergedOut.close();
        meta["n_units"] = nUnits;
        std::string mergedJson = outDir + "/" + jointPref + hexIden + ".json";
        mergedOut.open(mergedJson);
        if (!mergedOut) {
            error("Error opening JSON file for output: %s", mergedJson.c_str());
        }
        mergedOut << std::setw(4) << meta << std::endl;
        mergedOut.close();
        combinedOccurrence.ensureFeatureCount(n_shared);
        std::string mergedPref = mergedFile;
        size_t pos = mergedPref.find_last_of(".");
        if (pos != std::string::npos) {
            mergedPref = mergedPref.substr(0, pos);
        }
        FeatureOccurrenceStats::writeTsv(mergedPref + ".feature.stats.tsv",
            sharedFeatureNames, n_shared, combinedOccurrence, hexOpts.featureInfoOptions);

        hexOpts.outFile = mergedFile;
        hexOpts.sortOutput(true);
        notice("[multisample] Step 2.5 completed for size %.1f: merged unit files into %s", unitSize, mergedFile.c_str());
    }
    }
}

    return 0;
}
