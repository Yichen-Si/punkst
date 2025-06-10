#include "punkst.h"
#include "pts2tiles.hpp"
#include "tiles2bins.hpp"
#include <regex>

int32_t cmdMultiSample(int32_t argc, char** argv) {
    std::string inTsvListFile, tmpDir, outDir;
    std::string jointPref = "";
    std::vector<std::string> inTsvList;
    int nThreads0 = 1, tileSize = -1;
    int tileBuffer = 1000, batchSize = 10000;
    int debug = 0, verbose = 1000000;
    int icol_x, icol_y, icol_feature, icol_int;
    int nskip = 0;
    int minTotalCountPerSample = 1;

    double hexSize = -1, hexGridDist = -1;
    std::string anchorListFile;
    std::vector<std::string> anchorList;
    float radius = -1.0f;
    bool noBackground = false;
    int32_t minCtPerUnit;
    std::string include_ftr_regex;
    std::string exclude_ftr_regex;

    bool overwrite = false;

    ParamList pl;
    pl.add_option("in-tsv", "List of input TSV files separated by space", inTsvList)
        .add_option("in-tsv-list", "A file containing the ID and the path to the input transcript file for each sample (tab/space delimited, one sample per line)", inTsvListFile)
        .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x, true)
        .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y, true)
        .add_option("icol-feature", "Column index for feature (0-based)", icol_feature, true)
        .add_option("icol-int", "Column index for integer value (0-based)", icol_int, true)
        .add_option("skip", "Number of lines to skip in the input file (default: 0)", nskip)
        .add_option("threads", "Number of threads to use (default: 1)", nThreads0)
        .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
        .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex)
        .add_option("min-total-count-per-sample", "Minimum total gene count per sample (default: 1) to include in the joint model", minTotalCountPerSample);
    // pts2tiles Options
    pl.add_option("tile-size", "Tile size (in the same unit as the input coordinates)", tileSize, true)
        .add_option("tile-buffer", "Buffer size per tile per thread (default: 1000 lines)", tileBuffer)
        .add_option("batch-size", "(Only used if the input is gzipped or a stdin stream.) Batch size in terms of the number of lines (default: 10000)", batchSize);
    // tiles2hex Options
    pl.add_option("hex-size", "Hexagon size (size length)", hexSize)
        .add_option("hex-grid-dist", "Hexagon grid distance (center-to-center distance)", hexGridDist)
        .add_option("anchor-files", "Anchor files (one for each sample)", anchorList)
        .add_option("anchor-files-list", "A file containing the path to the anchor file for each sample", anchorListFile)
        .add_option("radius", "Radius for each set of anchors", radius)
        .add_option("min-count", "Minimum count for a unit to be included in output", minCtPerUnit)
        .add_option("ignore-background", "Ignore pixels not within radius of any of the anchors", noBackground);
    // Output Options
    pl.add_option("out-dir", "Output base director", outDir, true)
        .add_option("out-joint-pref", "Prefix for joint analysis outputs", jointPref)
        .add_option("overwrite", "Overwrite existing output files", overwrite)
        .add_option("temp-dir", "Directory to store temporary files", tmpDir, true)
        .add_option("verbose", "Verbose", verbose)
        .add_option("debug", "Debug", debug);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing multisample options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    // Determine the number of threads to use.
    unsigned int nThreads = std::thread::hardware_concurrency();
    if (nThreads == 0 || nThreads >= nThreads0) {
        nThreads = nThreads0;
    }
    notice("Using %u threads for processing", nThreads);
    if (!outDir.empty() && (outDir.back() == '/' || outDir.back() == '\\')) {
        outDir.pop_back();
    }
    bool check_include = !include_ftr_regex.empty();
    bool check_exclude = !exclude_ftr_regex.empty();
    std::regex regex_include(include_ftr_regex);
    std::regex regex_exclude(exclude_ftr_regex);

    std::vector<std::string> samples, outDirs;
    std::unordered_map<std::string, uint32_t> sampleIdx;
    size_t S = 0;
{ // 0) Gather input files
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
                error("Invalid line in input file: %s", line.c_str());
            }
            samples.push_back(tokens[0]);
            inTsvList.push_back(tokens[1]);
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
    if (anchorList.size() > 0 && radius < 0) {
        error("--radius (a positive number) must be specified when using anchor files");
    }
}
    notice("Found %zu samples", S);

    std::string hexIden = "hex_" + std::to_string(static_cast<int32_t>(std::round(hexGridDist)));
    std::vector<int32_t> icol_ints = {icol_int};
{ // 1) Run pts2tiles on each sample
    if (hexSize <= 0) {
        if (hexGridDist <= 0) {
            error("Hexagon size or hexagon grid distance must be specified");
        } else {
            hexSize = hexGridDist / sqrt(3);
        }
    } else if (hexGridDist < 0) {
        hexGridDist = hexSize * sqrt(3);
    }
    std::string fileList = outDir + "/" + jointPref + ".persample_file_list.tsv";
    std::ofstream fileListOut(fileList);
    for (size_t s = 0; s < S; ++s) {
        const std::string& inTsv = inTsvList[s];
        const std::string& sn = samples[s];
        std::string outPref = outDirs[s] + "/" + sn + ".tiled";
        fileListOut << sn << "\t" << outPref << ".tsv\t" << outPref << ".index\n";
        if (!overwrite && std::filesystem::exists(outPref + ".tsv") && std::filesystem::exists(outPref + ".index")) {
            notice("pts2tiles output is present for sample %s. To overwrite, add --overwrite", sn.c_str());
            continue;
        }
        bool streaming = inTsv.size()>3 && inTsv.compare(inTsv.size()-3,3,".gz")==0;
        notice("[multisample] Running pts2tiles for sample %s", sn.c_str());
        Pts2Tiles pts2Tiles(nThreads, inTsv, tmpDir, outPref, tileSize, icol_x, icol_y, icol_feature, icol_ints, nskip, streaming, tileBuffer, batchSize);
        if (!pts2Tiles.run()) {
            return 1;
        }
    }
    fileListOut.close();
    notice("[multisample] Paths to pre-processed sample-specific transcript files written to: %s", fileList.c_str());
}
    notice("[multisample] Step 1 completed: pre-processed transcript files for %zu samples.", S);

    std::unordered_map<std::string, std::vector<uint64_t>> featCounts;
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> unionFeatures;
    size_t n_shared = 0;
{ // 1.5) Decide a partially shared feature dictionary
    // a) Read each sample’s feature counts
    for (size_t s = 0; s < S; ++s) {
        const auto featFile = outDirs[s] + "/" + samples[s] + ".tiled.features.tsv";
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
            bool exclude = check_exclude && std::regex_match(feature, regex_exclude);
            if (!include || exclude) { // Skip features based on regex filters
                continue;
            }
            auto it = featCounts.find(feature);
            if (it == featCounts.end()) {
                // first time seeing this feature → init vector<S+2> with zeros
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
    std::filesystem::path unionFile = std::filesystem::path(outDir) / (jointPref + ".union_features.tsv");
    std::ofstream unionOut(unionFile);
    std::filesystem::path sharedFile = std::filesystem::path(outDir) / (jointPref + ".features.tsv");
    std::ofstream sharedOut(sharedFile);
    if (!sharedOut || !unionOut) error("Cannot write shared or union feature file: %s", unionFile.c_str());
    std::vector<std::ofstream> sampleOuts(S);
    for (size_t s = 0; s < S; ++s) {
        const auto featFile = outDirs[s] + "/" + samples[s] + ".tiled.features.tsv";
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
}
    notice("[multisample] Step 1.5 completed,selected %zu/%zu shared features for joint analysis", n_shared, unionFeatures.size());

    HexGrid hexGrid(hexSize);
    // 2) Run tiles2hex on each sample
    for (size_t s = 0; s < S; ++s) {
        std::string outFile = outDirs[s] + "/" + samples[s] + "." + hexIden + ".txt";
        std::string outJson = outDirs[s] + "/" + samples[s] + "." + hexIden + ".json";
        if (!overwrite && std::filesystem::exists(outFile) &&
                          std::filesystem::exists(outJson)) {
            notice("tiles2hex output is present for sample %s. To overwrite, add --overwrite", samples[s].c_str());
            continue;
        }
        std::string inTsv = outDirs[s] + "/" + samples[s] + ".tiled.tsv";
        std::string inIndex = outDirs[s] + "/" + samples[s] + ".tiled.index";
        TileReader tileReader(inTsv, inIndex);
        if (!tileReader.isValid()) {
            error("Error opening input file: %s", inTsv.c_str());
        }
        std::string dictFile = outDirs[s] + "/" + samples[s] + ".tiled.features.tsv";
        lineParser parser(icol_x, icol_y, icol_feature, icol_ints, dictFile);
        if (anchorList.size() != S || anchorList[s].empty()) {
            Tiles2Hex tiles2Hex(nThreads, tmpDir, outFile, hexGrid, tileReader, parser, {minCtPerUnit});
            if (!tiles2Hex.run()) {
                error("Error running tiles2hex for sample %s", samples[s].c_str());
            }
            tiles2Hex.writeMetadata();
        } else {
            std::vector<std::string> localAnchor = {anchorList[s]};
            std::vector<float> localRadius = {radius};
            Tiles2UnitsByAnchor tiles2Hex(nThreads, tmpDir, outFile, hexGrid, tileReader, parser, localAnchor, localRadius, {minCtPerUnit}, noBackground);
            if (!tiles2Hex.run()) {
                error("Error running tiles2hex for sample %s", samples[s].c_str());
            }
            tiles2Hex.writeMetadata();
        }
        // sort output file by the first column then delete the unsorted file
        if (sys_sort(outFile.c_str(), nullptr,
                     {"-k1,1", "--parallel="+std::to_string(nThreads), "-o", outFile}) != 0) {
            error("Error sorting hexagon file for sample %s", samples[s].c_str());
        }
    }
    notice("[multisample] Step 2 completed: created sample specific hexagon files");

    std::string mergedFile = outDir + "/" + jointPref + "." + hexIden + ".txt";
{ // 2.5) Create a merged hexagon file containing only the shared features
    nlohmann::json meta;
    meta["hex_size"] = hexGrid.size;
    meta["random_key"] = 0;
    meta["sample"] = 1;
    meta["offset_data"] = 2;
    meta["header_info"] = {"random_key", "sample"};
    meta["n_features"] = n_shared;
    std::unordered_map<std::string, uint32_t> dict;
    for(uint32_t i = 0; i < n_shared; ++i) {
        dict[unionFeatures[i].first] = i;
    }
    meta["dictionary"] = (nlohmann::json) dict;

    std::ofstream mergedOut(mergedFile);
    if (!mergedOut) {
        error("Error opening merged output file for output: %s", mergedFile.c_str());
    }
    int32_t nUnits = 0;
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
            for (size_t i = offset + 2; i < tokens.size(); ++i) {
                std::vector<std::string> pair;
                split(pair, " ", tokens[i]);
                uint32_t u, v;
                if (str2uint32(pair[0], u) && u < n_shared && str2uint32(pair[1], v)) {
                    ss2 << "\t" << tokens[i];
                    nFeatures++;
                    totCount += v;
                }
            }
            if (totCount < minCtPerUnit) {
                continue;
            }
            nUnits++;
            ss1 << "\t" << nFeatures << "\t" << totCount;
            mergedOut << ss1.str() << ss2.str() << "\n";
        }
    }
    mergedOut.close();
    meta["n_units"] = nUnits;
    std::string mergedJson = outDir + "/" + jointPref + "." + hexIden + ".json";
    mergedOut.open(mergedJson);
    if (!mergedOut) {
        error("Error opening JSON file for output: %s", mergedJson.c_str());
    }
    mergedOut << std::setw(4) << meta << std::endl;
    mergedOut.close();
    if (sys_sort(mergedFile.c_str(), nullptr,
                  {"-k1,1", "--parallel="+std::to_string(nThreads), "-o", mergedFile} ) != 0) {
        error("Error sorting merged hexagon file: %s", mergedFile.c_str());
    }
}
    notice("[multisample] Step 2.5 completed: merged hexagon files into %s", mergedFile.c_str());

    return 0;
}
