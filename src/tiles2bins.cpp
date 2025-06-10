#include "tiles2bins.hpp"

Tiles2Hex::Tiles2Hex(int32_t nThreads, std::string& _tmpDirPath, std::string& _outFile, HexGrid& hexGrid, TileReader& tileReader, lineParser& parser, std::vector<int32_t> _minCounts)
: nThreads(nThreads), tmpDir(_tmpDirPath), outFile(_outFile), hexGrid(hexGrid), tileReader(tileReader), parser(parser), minCounts(_minCounts), nUnits(0), nFeatures(0) {
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
    notice("Created temporary directory: %s", tmpDir.path.string().c_str());
    meta["hex_size"] = hexGrid.size;
    meta["n_modalities"] = nModal;
    meta["random_key"] = 0;
    meta["icol_x"] = 1;
    meta["icol_y"] = 2;
    meta["offset_data"] = 3;
    meta["header_info"] = {"random_key", "x", "y"};
}

void Tiles2Hex::writeMetadata() {
    // check if outFile has an extension
    std::string metaFile;
    size_t pos = outFile.find_last_of(".");
    if (pos != std::string::npos) {
        metaFile = outFile.substr(0, pos) + ".json";
    } else {
        metaFile = outFile + ".json";
    }
    std::ofstream metaOut(metaFile);
    if (!metaOut) {
        error("Error opening metadata file %s for output", metaFile.c_str());
        return;
    }
    metaOut << std::setw(4) << meta << std::endl;
    metaOut.close();
}

void Tiles2Hex::worker(int threadId) {
    // Set up a per-thread random engine.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);
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
            // Convert point to hex axial coordinate & add to buffer
            int32_t hx, hy;
            hexGrid.cart_to_axial(hx, hy, pixel.x, pixel.y);
            addPixelToUnitMaps(pixel, hx, hy, unitBuffer);
        } // end while reading lines in tile
        if (unitBuffer.empty()) {
            continue;
        }
        { // Lock the global output file.
            std::lock_guard<std::mutex> lock(mainOutMutex);
            for (const auto &entry : unitBuffer) {
                // Get the hexagon’s bounding box
                double hex_xmin, hex_xmax, hex_ymin, hex_ymax;
                hexGrid.hex_bounding_box_axial(hex_xmin, hex_xmax, hex_ymin, hex_ymax, entry.second.x, entry.second.y);
                // Determine if the hexagon is fully contained inside the tile
                bool isInternal = (
                    hex_xmin >= tile_xmin && hex_xmax < tile_xmax &&
                    hex_ymin >= tile_ymin && hex_ymax < tile_ymax);
                if (isInternal) {
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
                    writeUnit(entry.second, rdUnif(gen));
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
    } // end while tiles
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
        // Call worker directly instead of spawning a new thread.
        notice("Running single-threaded...");
        worker(0);
    } else {
        // Launch worker threads to process tiles.
        notice("Launching %d worker threads...", nThreads);
        if (!launchWorkerThreads()) {
            error("Error launching worker threads");
        }
        // Wait for all worker threads to finish.
        if (!joinWorkerThreads()) {
            error("Error joining worker threads");
        }
        notice("All worker threads finished");
    }
    // Merge boundary hexagons from temporary files and append to main output.
    notice("Merging boundary hexagons...");
    if (!mergeBoundaryHexagons()) {
        error("Error merging boundary hexagons");
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
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);
        int32_t nUnits0 = nUnits;
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
            writeUnit(entry.second, rdUnif(gen));
            nUnits++;
        }
        mainOut.flush();
        notice("Wrote %d/%zu boundary hexagons to main output", nUnits-nUnits0, mergedUnits.size());
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
    // Set up a per-thread random engine.
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);
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
                    writeUnit(entry.second, rdUnif(gen));
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
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);
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
                writeUnit(entry.second, rdUnif(gen));
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
