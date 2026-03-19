#include "tiles2bins.hpp"

Tiles2Hex::Tiles2Hex(int32_t nThreads, std::string& _tmpDirPath, std::string& _outFile, HexGrid& hexGrid, TileReader& tileReader, lineParser& parser, std::vector<int32_t> _minCounts, int32_t _seed, double _bccSize)
: nThreads(nThreads), outFile(_outFile), tmpDir(_tmpDirPath), nUnits(0), nFeatures(0), minCounts(_minCounts), is3D(parser.hasZCoord()), parser(parser), hexGrid(hexGrid), bccGrid((parser.hasZCoord() && _bccSize > 0) ? _bccSize : hexGrid.size), tileReader(tileReader) {
    nModal = parser.n_ct;
    mainOut.open(outFile, std::ios::out);
    if (!mainOut) {
        error("Error opening output file %s for writing", outFile.c_str());
        return;
    }
    mainOut << std::setprecision(4) << std::fixed;
    if (minCounts.size() != nModal) {
        minCounts.resize(nModal);
        std::fill(minCounts.begin(), minCounts.end(), 1);
    }
    if (is3D && _bccSize <= 0) {
        error("3D aggregation requires a positive BCC size");
    }
    notice("Created temporary directory: %s", tmpDir.path.string().c_str());
    meta["n_modalities"] = nModal;
    meta["coord_dim"] = is3D ? 3 : 2;
    meta["count_hist_bin_size"] = countHistBinSize;
    meta["random_key"] = 0;
    meta["icol_x"] = 1;
    meta["icol_y"] = 2;
    if (is3D) {
        meta["bcc_size"] = bccGrid.size;
        meta["icol_z"] = 3;
        meta["offset_data"] = 4;
        meta["header_info"] = {"random_key", "x", "y", "z"};
    } else {
        meta["hex_size"] = hexGrid.size;
        meta["offset_data"] = 3;
        meta["header_info"] = {"random_key", "x", "y"};
    }
    if (_seed <= 0) {
        randomSeed = std::random_device{}();
        notice("Using random seed %u", randomSeed);
    } else {
        randomSeed = static_cast<uint32_t>(_seed);
    }
    meta["seed"] = randomSeed;
}

std::string Tiles2Hex::outputPrefix() const {
    size_t pos = outFile.find_last_of(".");
    if (pos != std::string::npos) {
        return outFile.substr(0, pos);
    }
    return outFile;
}

void Tiles2Hex::writeMetadata() {
    std::string metaFile = outputPrefix() + ".json";
    std::ofstream metaOut(metaFile);
    if (!metaOut) {
        error("Error opening metadata file %s for output", metaFile.c_str());
        return;
    }
    metaOut << std::setw(4) << meta << std::endl;
    metaOut.close();
    writeCountHistogram();
}

void Tiles2Hex::writeCountHistogram() const {
    const std::string histFile = outputPrefix() + ".count_hist.tsv";
    std::ofstream histOut(histFile);
    if (!histOut) {
        error("Error opening count histogram file %s for output", histFile.c_str());
        return;
    }
    histOut << "count_min\tcount_max\tn_units\n";
    for (const auto& entry : countHist) {
        const uint32_t binStart = entry.first;
        const uint32_t binEnd = binStart + countHistBinSize - 1;
        histOut << binStart << "\t" << binEnd << "\t" << entry.second << "\n";
    }
    histOut.close();
}

void Tiles2Hex::worker(int threadId) {
    if (is3D) {
        worker3D(threadId);
        return;
    }
    worker2D(threadId);
}

void Tiles2Hex::worker2D(int threadId) {
    // temporary file
    auto partialFile = tmpDir.path / (std::to_string(threadId) + ".txt");
    std::ofstream outFile(partialFile, std::ios::out);
    if (!outFile) {
        error("Error opening temporary file: %s", partialFile.string().c_str());
    }
    uint32_t maxFeatureIdxLocal = 0;
    // Process one tile at a time
    TileKey tile;
    int tileSize = tileReader.getTileSize();
    while (tileQueue.pop(tile)) {

        // Compute tile’s cartesian bounding box.
        double tile_xmin = tile.col * tileSize;
        double tile_xmax = (tile.col + 1) * tileSize;
        double tile_ymin = tile.row * tileSize;
        double tile_ymax = (tile.row + 1) * tileSize;
        bool checkBounds = tileReader.isPartial(tile);

        // Get an iterator for points in this tile.
        std::unique_ptr<BoundedReadline> iter;
        try {
            iter = tileReader.get_tile_iterator(tile.row, tile.col);
        } catch (const std::exception &e) {
            warning("Tile (%d,%d) iterator error: %s", tile.row, tile.col, e.what());
            continue;
        }

        // Local collection of hexagons
        std::unordered_map<int64_t, UnitValues> unitBuffer;

        std::string line;
        while (iter->next(line)) {
            PixelValues pixel;
            int32_t ret = parser.parse(pixel, line, checkBounds);
            if (!parser.isFeatureDict && pixel.feature >= maxFeatureIdxLocal) {
                maxFeatureIdxLocal = pixel.feature + 1;
            }
            if (ret < 0) {
                error("Error parsing line: %s", line.c_str());
            }
            if (ret == 0) {
                continue;
            }
            int32_t hx, hy;
            hexGrid.cart_to_axial(hx, hy, pixel.x, pixel.y);
            addPixelToUnitMaps(pixel, hx, hy, unitBuffer);
        }
        if (unitBuffer.empty()) {
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mainOutMutex);
            for (const auto &entry : unitBuffer) {
                double hex_xmin, hex_xmax, hex_ymin, hex_ymax;
                hexGrid.hex_bounding_box_axial(hex_xmin, hex_xmax, hex_ymin, hex_ymax, entry.second.x, entry.second.y);
                bool isInternal = (
                    hex_xmin >= tile_xmin && hex_xmax < tile_xmax &&
                    hex_ymin >= tile_ymin && hex_ymax < tile_ymax);
                if (isInternal) {
                    if (!passesMinCount(entry.second.valsums)) {
                        continue;
                    }
                    writeUnit(entry.second, makeRandomKey(entry.second));
                    nUnits++;
                } else {
                    entry.second.writeToFile(outFile, 0);
                }
            }
            mainOut.flush();
            if (!parser.isFeatureDict && nFeatures < maxFeatureIdxLocal) {
                nFeatures = maxFeatureIdxLocal;
            }
        }
        notice("Thread %d: processed %zu hexagons in tile (%d, %d)", threadId, unitBuffer.size(), tile.row, tile.col);
    }
    outFile.close();
}

void Tiles2Hex::worker3D(int threadId) {
    auto partialFile = tmpDir.path / (std::to_string(threadId) + ".txt");
    std::ofstream outFile(partialFile, std::ios::out);
    if (!outFile) {
        error("Error opening temporary file: %s", partialFile.string().c_str());
    }
    uint32_t maxFeatureIdxLocal = 0;
    TileKey tile;
    int tileSize = tileReader.getTileSize();
    while (tileQueue.pop(tile)) {
        double tile_xmin = tile.col * tileSize;
        double tile_xmax = (tile.col + 1) * tileSize;
        double tile_ymin = tile.row * tileSize;
        double tile_ymax = (tile.row + 1) * tileSize;
        bool checkBounds = tileReader.isPartial(tile);

        std::unique_ptr<BoundedReadline> iter;
        try {
            iter = tileReader.get_tile_iterator(tile.row, tile.col);
        } catch (const std::exception &e) {
            warning("Tile (%d,%d) iterator error: %s", tile.row, tile.col, e.what());
            continue;
        }

        std::unordered_map<BCCGrid::cell_key_t, UnitValues3D, Tuple3Hash> unitBuffer;

        std::string line;
        while (iter->next(line)) {
            PixelValues3D pixel;
            int32_t ret = parser.parse(pixel, line, checkBounds);
            if (!parser.isFeatureDict && pixel.feature >= maxFeatureIdxLocal) {
                maxFeatureIdxLocal = pixel.feature + 1;
            }
            if (ret < 0) {
                error("Error parsing line: %s", line.c_str());
            }
            if (ret == 0) {
                continue;
            }
            int32_t q1, q2, q3;
            bccGrid.cart_to_lattice(q1, q2, q3, pixel.x, pixel.y, pixel.z);
            addPixelToUnitMaps(pixel, q1, q2, q3, unitBuffer);
        }
        if (unitBuffer.empty()) {
            continue;
        }
        {
            std::lock_guard<std::mutex> lock(mainOutMutex);
            for (const auto& entry : unitBuffer) {
                double cell_xmin, cell_xmax, cell_ymin, cell_ymax;
                bccGrid.projected_bounding_box_xy(cell_xmin, cell_xmax, cell_ymin, cell_ymax,
                                                  entry.second.x, entry.second.y, entry.second.z);
                bool isInternal = (
                    cell_xmin >= tile_xmin && cell_xmax < tile_xmax &&
                    cell_ymin >= tile_ymin && cell_ymax < tile_ymax);
                if (isInternal) {
                    if (!passesMinCount(entry.second.valsums)) {
                        continue;
                    }
                    writeUnit(entry.second, makeRandomKey(entry.second));
                    nUnits++;
                } else {
                    entry.second.writeToFile(outFile, 0);
                }
            }
            mainOut.flush();
            if (!parser.isFeatureDict && nFeatures < maxFeatureIdxLocal) {
                nFeatures = maxFeatureIdxLocal;
            }
        }
        notice("Thread %d: processed %zu BCC units in tile (%d, %d)", threadId, unitBuffer.size(), tile.row, tile.col);
    }
    outFile.close();
}

bool Tiles2Hex::run() {
    if (nThreads <= 1) {
        std::vector<TileKey> tileList;
        tileReader.getTileList(tileList);
        for (const auto& tile : tileList) {
            tileQueue.push(tile);
        }
        tileQueue.set_done();
        notice("Running single-threaded...");
        worker(0);
    } else {
        notice("Launching %d worker threads...", nThreads);
        if (!launchWorkerThreads()) {
            error("Error launching worker threads");
        }
        if (!joinWorkerThreads()) {
            error("Error joining worker threads");
        }
        notice("All worker threads finished");
    }
    notice("Merging boundary %s...", is3D ? "BCC units" : "hexagons");
    if (!mergeBoundaryHexagons()) {
        error("Error merging boundary units");
    }
    meta["n_units"] = nUnits;
    if (parser.isFeatureDict) {
        meta["n_features"] = parser.featureDict.size();
        nlohmann::json dict(parser.featureDict);
        meta["dictionary"] = dict;
    } else {
        meta["n_features"] = nFeatures;
    }
    return true;
}

bool Tiles2Hex::mergeBoundaryHexagons() {
    if (is3D) {
        return mergeBoundaryHexagons3D();
    }
    return mergeBoundaryHexagons2D();
}

bool Tiles2Hex::mergeBoundaryHexagons2D() {
    std::unordered_map<int64_t, UnitValues> mergedUnits;
    for (int i = 0; i < nThreads; ++i) {
        auto fname = tmpDir.path / (std::to_string(i) + ".txt");
        std::ifstream ifs(fname);
        if (!ifs) {
            continue; // unlikely?
        }
        std::string line;
        int32_t nunits = 0, nnew = 0;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            UnitValues unit(0, 0); // temporary initialization
            if (!unit.readFromLine(line, nModal)) {
                warning("Error reading line: %s", line.c_str());
                continue;
            }
            nunits++;
            int64_t key = (static_cast<int64_t>(unit.x) << 32) | unit.y;
            auto it = mergedUnits.find(key);
            if (it == mergedUnits.end()) {
                mergedUnits.insert({key, unit});
                nnew++;
            } else {
                it->second.mergeUnits(unit);
            }
        }
        ifs.close();
        std::filesystem::remove(fname);
        notice("Merging temporary file from thread %d... %d units, %d new", i, nunits, nnew);
    }
    // Append merged boundary units to main output.
    {
        std::lock_guard<std::mutex> lock(mainOutMutex);
        int32_t nUnits0 = nUnits;
        for (const auto &entry : mergedUnits) {
            if (!passesMinCount(entry.second.valsums)) {
                continue;
            }
            writeUnit(entry.second, makeRandomKey(entry.second));
            nUnits++;
        }
        mainOut.flush();
        notice("Wrote %d/%zu boundary hexagons to main output", nUnits-nUnits0, mergedUnits.size());
    }
    mainOut.close();
    return true;
}

bool Tiles2Hex::mergeBoundaryHexagons3D() {
    std::unordered_map<BCCGrid::cell_key_t, UnitValues3D, Tuple3Hash> mergedUnits;
    for (int i = 0; i < nThreads; ++i) {
        auto fname = tmpDir.path / (std::to_string(i) + ".txt");
        std::ifstream ifs(fname);
        if (!ifs) {
            continue;
        }
        std::string line;
        int32_t nunits = 0, nnew = 0;
        while (std::getline(ifs, line)) {
            if (line.empty()) {
                continue;
            }
            UnitValues3D unit(0, 0, 0);
            if (!unit.readFromLine(line, nModal)) {
                warning("Error reading line: %s", line.c_str());
                continue;
            }
            nunits++;
            BCCGrid::cell_key_t key = BCCGrid::make_key(unit.x, unit.y, unit.z);
            auto it = mergedUnits.find(key);
            if (it == mergedUnits.end()) {
                mergedUnits.insert({key, unit});
                nnew++;
            } else {
                it->second.mergeUnits(unit);
            }
        }
        ifs.close();
        std::filesystem::remove(fname);
        notice("Merging temporary file from thread %d... %d units, %d new", i, nunits, nnew);
    }
    {
        std::lock_guard<std::mutex> lock(mainOutMutex);
        int32_t nUnits0 = nUnits;
        for (const auto& entry : mergedUnits) {
            if (!passesMinCount(entry.second.valsums)) {
                continue;
            }
            writeUnit(entry.second, makeRandomKey(entry.second));
            nUnits++;
        }
        mainOut.flush();
        notice("Wrote %d/%zu boundary BCC units to main output", nUnits - nUnits0, mergedUnits.size());
    }
    mainOut.close();
    return true;
}


void Tiles2UnitsByAnchor::readAnchors(std::string& anchorFile) {
    PointCloud<float> cloud;
    std::ifstream ifs(anchorFile);
    if (!ifs) {
        error("Error opening anchor file: %s", anchorFile.c_str());
    }
    std::string line;
    while (std::getline(ifs, line)) {
        std::istringstream iss(line);
        float x, y;
        iss >> x >> y;
        cloud.pts.emplace_back(x, y);
    }
    ifs.close();
    anchorPoints.push_back(std::move(cloud));
    trees.push_back(std::make_unique<kd_tree_f2_t>(2, anchorPoints.back(), nanoflann::KDTreeSingleIndexAdaptorParams(10)));
}

void Tiles2UnitsByAnchor::worker(int threadId) {
    // temporary file
    auto partialFile = tmpDir.path / (std::to_string(threadId) + ".txt");
    std::ofstream outFile(partialFile, std::ios::out);
    if (!outFile) {
        error("Error opening temporary file: %s", partialFile.string().c_str());
    }
    uint32_t maxFeatureIdxLocal = 0;
    // Process one tile at a time
    TileKey tile;
    int tileSize = tileReader.getTileSize();
    while (tileQueue.pop(tile)) {
        // Compute tile’s cartesian bounding box.
        double tile_xmin = tile.col * tileSize;
        double tile_xmax = (tile.col + 1) * tileSize;
        double tile_ymin = tile.row * tileSize;
        double tile_ymax = (tile.row + 1) * tileSize;
        bool checkBounds = tileReader.isPartial(tile);
        // Get an iterator for points in this tile.
        std::unique_ptr<BoundedReadline> iter;
        try {
            iter = tileReader.get_tile_iterator(tile.row, tile.col);
        } catch (const std::exception &e) {
            warning("Tile (%d,%d) iterator error: %s", tile.row, tile.col, e.what());
            continue;
        }

        // Local collection of units
        std::vector<std::unordered_map<int64_t, UnitValues>> unitBuffersInternal, unitBuffers;
        unitBuffersInternal.resize(nLayer);
        unitBuffers.resize(nLayer);
        std::string line;
        int32_t nbg = 0, nfront = 0;
        while (iter->next(line)) {
            PixelValues pixel;
            int32_t ret = parser.parse(pixel, line, checkBounds);
            if (!parser.isFeatureDict && pixel.feature >= maxFeatureIdxLocal) {
                maxFeatureIdxLocal = pixel.feature + 1;
            }
            if (ret < 0) {
                error("Error parsing line: %s", line.c_str());
            }
            if (ret == 0) {
                continue;
            }
            // Convert point to hex axial coordinate & add to buffer
            int32_t hx, hy;
            hexGrid.cart_to_axial(hx, hy, pixel.x, pixel.y);
            // Get the hexagon’s bounding box
            double hex_xmin, hex_xmax, hex_ymin, hex_ymax;
            hexGrid.hex_bounding_box_axial(hex_xmin, hex_xmax, hex_ymin, hex_ymax, hx, hy);
            // Determine if the hexagon is fully contained inside the tile
            bool isInternal = (
                hex_xmin >= tile_xmin && hex_xmax < tile_xmax &&
                hex_ymin >= tile_ymin && hex_ymax < tile_ymax);

            // check if points fall within the radius of any anchor
            bool isBackground = true;
            std::vector<nanoflann::ResultItem<uint32_t, float>> indices_dists;
            for (uint32_t i = 0; i < nAnchorSets; ++i) {
                float xy[2] = {(float) pixel.x, (float) pixel.y};
                size_t n = trees[i]->radiusSearch(xy, l2radius[i], indices_dists);
                if (n == 0) {
                    continue;
                }
                isBackground = false;
                if (isInternal) {
                    addPixelToUnitMaps(pixel, hx, hy, unitBuffersInternal[i], i);
                } else {
                    addPixelToUnitMaps(pixel, hx, hy, unitBuffers[i], i);
                }
            }
            if (isBackground) {
                nbg++;
            } else {
                nfront++;
            }
            if (isBackground && !noBackground) {
                if (isInternal) {
                    addPixelToUnitMaps(pixel, hx, hy, unitBuffersInternal[nAnchorSets], nAnchorSets);
                } else {
                    addPixelToUnitMaps(pixel, hx, hy, unitBuffers[nAnchorSets], nAnchorSets);
                }
            }
        } // end while reading lines in tile
        { // Lock the global output file.
            std::lock_guard<std::mutex> lock(mainOutMutex);
            for (const auto &unitBuffer : unitBuffersInternal) {
                for (const auto &entry : unitBuffer) {
                    bool flag = false;
                    for (size_t i = 0; i < nModal; ++i) {
                        if (entry.second.valsums[i] >= minCounts[i]) {
                            flag = true;
                            break;
                        }
                    }
                    if (!flag) {
                        continue;
                    }
                    writeUnit(entry.second, makeRandomKey(entry.second));
                    nUnits++;
                }
            }
            mainOut.flush();
            if (!parser.isFeatureDict && nFeatures < maxFeatureIdxLocal) {
                nFeatures = maxFeatureIdxLocal;
            }
        }

        // Write the boundary units to the temporary file
        for (const auto &unitBuffer : unitBuffers) {
            for (const auto &entry : unitBuffer) {
                entry.second.writeToFile(outFile, 0);
            }
        }

    } // end while tiles
    outFile.close();
}

bool Tiles2UnitsByAnchor::mergeBoundaryHexagons() {
    std::vector<std::unordered_map<uint64_t, UnitValues>> mergedUnitsList(nLayer);
    for (int i = 0; i < nThreads; ++i) {
        auto fname = tmpDir.path / (std::to_string(i) + ".txt");
        std::ifstream ifs(fname);
        if (!ifs) {
            continue; // unlikely?
        }
        std::string line;
        int32_t nunits = 0, nnew = 0;
        while (std::getline(ifs, line)) {
            if (line.empty()) continue;
            UnitValues unit(0, 0); // temporary initialization
            if (!unit.readFromLine(line, nModal, true)) {
                warning("Error reading line: %s", line.c_str());
                continue;
            }
            nunits++;
            uint64_t key = (static_cast<uint64_t>(unit.x) << 32) | unit.y;
            if (unit.label > nLayer) {
                error("Error: unit label %d exceeds the number of layers %d", unit.label, nLayer);
            }
            auto& mergedUnits = mergedUnitsList[unit.label];
            auto it = mergedUnits.find(key);
            if (it == mergedUnits.end()) {
                mergedUnits.insert({key, unit});
                nnew++;
            } else {
                it->second.mergeUnits(unit);
            }
        }
        ifs.close();
        std::filesystem::remove(fname);
        notice("Merging temporary file from thread %d... %d units, %d new", i, nunits, nnew);
    }
    // Append merged boundary units to main output.
    {
        std::lock_guard<std::mutex> lock(mainOutMutex);
        int32_t nUnits0 = nUnits, nTot = 0;
        for (auto &mergedUnits : mergedUnitsList) {
            for (const auto &entry : mergedUnits) {
                bool flag = false;
                for (size_t i = 0; i < nModal; ++i) {
                    if (entry.second.valsums[i] >= minCounts[i]) {
                        flag = true;
                        break;
                    }
                }
                if (!flag) {
                    continue;
                }
                writeUnit(entry.second, makeRandomKey(entry.second));
                nUnits++;
            }
            nTot += mergedUnits.size();
        }
        mainOut.flush();
        notice("Wrote %d/%zu boundary hexagons to main output", nUnits-nUnits0, nTot);
    }
    mainOut.close();
    return true;
}
