#include <iostream>
#include <fstream>
#include <sstream>
#include <atomic>
#include <cstdlib>
#include <random>
#include "utils.h"
#include "tilereader.hpp"
#include "threads.hpp"
#include "qgenlib/qgen_utils.h"

//------------------------------------------------------------------------------
// Processes tiled point data in minibatches
//------------------------------------------------------------------------------
class Tiles2Minibatch {
public:
    // Structure representing one point.
    struct Pixel {
        double x, y;
        std::vector<std::pair<uint32_t, int32_t>> vals;
    };

    // Constructor.
    //   nThreads: number of worker threads
    //   tmpDir: directory to store temporary boundary files
    //   tileReader: object to read a tile from the input
    //   b: number of minibatch subdivisions per tile side (b>=2)
    //   r: padding radius (in the same units as tile coordinates)
    Tiles2Minibatch(int nThreads, const std::string &_tmpDir, const std::string &outFile, TileReader &tileReader, int b, double r)
        : numThreads(nThreads), outFile(outFile),
          tileReader(tileReader), b(b), r(r) {
        tmpDir = std::filesystem::path(_tmpDir) / "boundaries";
        if (!createDirectory(tmpDir)) {
            throw std::runtime_error("Error creating temporary directory (or the existing directory is not empty): " + tmpDir);
        }
        // Open main output file.
        mainOut.open(outFile, std::ios::out);
        if (!mainOut) {
            throw std::runtime_error("Unable to open output file: " + outFile);
        }
        // Get tile size T and the list of tiles
        tileSize = tileReader.getTileSize();
        tileReader.getTileList(tileList);
    }

    ~Tiles2Minibatch() {
        if (mainOut.is_open()) {
            mainOut.close();
        }
    }

private:
    int numThreads;
    std::string tmpDir, outFile;
    TileReader &tileReader;
    int b;              // number of minibatch subdivisions per side
    double r;           // padding radius
    int tileSize;       // tile side length (T)
    std::ofstream mainOut;
    std::vector<TileKey> tileList;

    // A thread-safe queue for tiles
    ThreadSafeQueue<TileKey> tileQueue;
    std::vector<std::thread> threads;
    std::mutex mainOutMutex;

    // Stub processing function.
    // Given a padded set of points and the coordinates of the internal region,
    // return updated points for the internal region.
    std::vector<Pixel> processMinibatch(const std::vector<Pixel> &paddedData,
                                        Rectangle<double> &internalBox) {
        // Placeholder
        std::vector<Pixel> updated;
        for (const auto &pt : paddedData) {
            if (internalBox.contains(pt.x, pt.y)) {
                updated.push_back(pt);
            }
        }
        return updated;
    }

    // Helper to write a set of points (raw data) to a temporary file.
    void writeBoundaryPoints(const std::string &filename, const std::vector<Pixel> &points) {
        std::ofstream ofs(filename, std::ios::app);
        if (!ofs) {
            std::cerr << "Error opening temporary file: " << filename << "\n";
            return;
        }
        for (const auto &pt : points) {
            ofs << pt.x << "\t" << pt.y << "\t" << pt.feature;
            for (const auto &v : pt.intvals) {
                ofs << "\t" << v;
            }
            ofs << "\n";
        }
        ofs.close();
    }

    // Worker function: process one tile at a time.
    void worker(int threadId) {
        // Each thread uses its own random engine if needed.
        std::random_device rd;
        std::mt19937 gen(rd());
        TileKey tile;
        // Process tiles until none remain.
        while (tileQueue.pop(tile)) {
            // Compute tile global boundaries.
            double tile_xmin = tile.col * tileSize;
            double tile_ymin = tile.row * tileSize;
            double tile_xmax = tile_xmin + tileSize;
            double tile_ymax = tile_ymin + tileSize;

            // Read all points from the tile into a vector.
            std::vector<Pixel> tilePoints;
            {
                std::unique_ptr<BoundedReadline> iter;
                try {
                    iter = tileReader.get_tile_iterator(tile.row, tile.col);
                } catch (const std::exception &e) {
                    std::cerr << "Tile (" << tile.row << "," << tile.col
                              << ") iterator error: " << e.what() << "\n";
                    continue;
                }
                std::string line;
                while (iter->next(line)) {
                    // Assume a helper function parsePoint(line, pt) exists.
                    Pixel pt;
                    if (!parsePoint(line, pt))
                        continue;
                    tilePoints.push_back(pt);
                }
            }
            if (tilePoints.empty())
                continue;

            // For processing, subdivide the tile into b x b minibatches.
            double minibatchSize = tileSize / double(b);
            // Vectors to collect updated (internal) points and boundary points.
            std::vector<Pixel> updatedInternal;
            // For boundary points, we organize them by temporary file name.
            std::unordered_map<std::string, std::vector<Pixel>> boundaryPoints;

            // Iterate over minibatch positions (i=0..b-1, j=0..b-1).
            for (int i = 0; i < b; ++i) {
                for (int j = 0; j < b; ++j) {
                    // Internal (target) region of this minibatch.
                    double ixmin = tile_xmin + j * minibatchSize + r;
                    double iymin = tile_ymin + i * minibatchSize + r;
                    double ixmax = tile_xmin + (j+1) * minibatchSize - r;
                    double iymax = tile_ymin + (i+1) * minibatchSize - r;
                    // Padded region: depends on the minibatch's location.
                    double pad_left = (j == 0) ? 0 : r;
                    double pad_right = (j == b-1) ? 0 : r;
                    double pad_top = (i == 0) ? 0 : r;
                    double pad_bottom = (i == b-1) ? 0 : r;
                    // Thus padded region is:
                    double pxmin = tile_xmin + j * minibatchSize - pad_left;
                    double pymin = tile_ymin + i * minibatchSize - pad_top;
                    double pxmax = tile_xmin + (j+1) * minibatchSize + pad_right;
                    double pymax = tile_ymin + (i+1) * minibatchSize + pad_bottom;
                    // Clamp padded region to tile boundaries.
                    pxmin = std::max(pxmin, tile_xmin);
                    pymin = std::max(pymin, tile_ymin);
                    pxmax = std::min(pxmax, tile_xmax);
                    pymax = std::min(pymax, tile_ymax);

                    // Collect all points in tilePoints that fall into the padded region.
                    std::vector<Pixel> paddedData;
                    for (const auto &pt : tilePoints) {
                        if (pt.x >= pxmin && pt.x <= pxmax &&
                            pt.y >= pymin && pt.y <= pymax) {
                            paddedData.push_back(pt);
                        }
                    }
                    // Process the minibatch (update only the internal region).
                    std::vector<Pixel> updated = processMinibatch(paddedData, ixmin, iymin, ixmax, iymax);
                    // Append results.
                    updatedInternal.insert(updatedInternal.end(), updated.begin(), updated.end());
                }
            } // end for minibatches

            // Now, separate tilePoints into those that are safely internal and those that
            // are near a tile boundary (within r of the tile edge). We define internal points
            // as those with x in [tile_xmin+r, tile_xmax-r] and y in [tile_ymin+r, tile_ymax-r].
            std::vector<Pixel> tileInternal, tileBoundary;
            for (const auto &pt : tilePoints) {
                if (pt.x >= tile_xmin + r && pt.x <= tile_xmax - r &&
                    pt.y >= tile_ymin + r && pt.y <= tile_ymax - r) {
                    tileInternal.push_back(pt);
                } else {
                    tileBoundary.push_back(pt);
                }
            }
            // We assume that the update function for the internal points has been run above.
            // Write updatedInternal points directly to mainOut.
            {
                std::lock_guard<std::mutex> lock(mainOutMutex);
                for (const auto &pt : updatedInternal) {
                    // For each updated point, write: x, y, feature, intvals (tab-delimited)
                    mainOut << pt.x << "\t" << pt.y << "\t" << pt.feature;
                    for (const auto &val : pt.intvals) {
                        mainOut << "\t" << val;
                    }
                    mainOut << "\n";
                }
                mainOut.flush();
            }
            // Now, assign boundary points to temporary files according to the scheme.
            // Compute local coordinates in the tile.
            for (const auto &pt : tileBoundary) {
                double lx = pt.x - tile_xmin;
                double ly = pt.y - tile_ymin;
                // We use tile size t = tileSize.
                std::string fname;
                // Horizontal boundaries.
                if (ly < 2*r) {
                    // Top strip.
                    // For corners, also adjust column.
                    if (lx < r) {
                        fname = "H_" + std::to_string(tile.row - 1) + "_" + std::to_string(tile.col - 1) + ".txt";
                    } else if (lx > tileSize - r) {
                        fname = "H_" + std::to_string(tile.row - 1) + "_" + std::to_string(tile.col + 1) + ".txt";
                    } else {
                        fname = "H_" + std::to_string(tile.row - 1) + "_" + std::to_string(tile.col) + ".txt";
                    }
                } else if (ly > tileSize - 2*r) {
                    // Bottom strip.
                    if (lx < r) {
                        fname = "H_" + std::to_string(tile.row) + "_" + std::to_string(tile.col - 1) + ".txt";
                    } else if (lx > tileSize - r) {
                        fname = "H_" + std::to_string(tile.row) + "_" + std::to_string(tile.col + 1) + ".txt";
                    } else {
                        fname = "H_" + std::to_string(tile.row) + "_" + std::to_string(tile.col) + ".txt";
                    }
                }
                // Vertical boundaries.
                if (fname.empty()) {
                    if (lx < 2*r) {
                        fname = "V_" + std::to_string(tile.row) + "_" + std::to_string(tile.col - 1) + ".txt";
                    } else if (lx > tileSize - 2*r) {
                        fname = "V_" + std::to_string(tile.row) + "_" + std::to_string(tile.col) + ".txt";
                    }
                }
                // If none of the above, default to a generic boundary file for this tile.
                if (fname.empty()) {
                    fname = "B_" + std::to_string(tile.row) + "_" + std::to_string(tile.col) + ".txt";
                }
                // Prepend tmpDir.
                fname = tmpDir + fname;
                boundaryPoints[fname].push_back(pt);
            }
            // Write each set of boundary points to its corresponding temporary file.
            for (const auto &pair : boundaryPoints) {
                writeBoundaryPoints(pair.first, pair.second);
            }
        } // end while tileQueue.pop
    }

    // Helper function to parse a point from a line.
    // Assumes the input line is tab-delimited with columns: x, y, feature, intvals...
    bool parsePoint(std::string &line, Pixel &pt) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < 3)
            return false;
        try {
            pt.x = std::stod(tokens[0]);
            pt.y = std::stod(tokens[1]);
        } catch (...) {
            return false;
        }
        pt.feature = tokens[2];
        pt.intvals.clear();
        for (size_t i = 3; i < tokens.size(); ++i) {
            int32_t val;
            if (!str2int32(tokens[i], val))
                return false;
            pt.intvals.push_back(val);
        }
        return true;
    }

    // Launch worker threads.
    bool launchWorkerThreads() {
        for (int i = 0; i < numThreads; ++i) {
            threads.emplace_back(&Tiles2Minibatch::worker, this, i);
        }
        // Push all tiles into the tile queue.
        std::vector<TileKey> tileList;
        tileReader.getTileList(tileList);
        for (const auto &tile : tileList) {
            tileQueue.push(tile);
        }
        tileQueue.set_done();
        return true;
    }

    // Wait for worker threads to finish.
    bool joinWorkerThreads() {
        for (auto &t : threads) {
            t.join();
        }
        return true;
    }

    // Merge all boundary temporary files and append their results to mainOut.
    bool mergeBoundaryMinibatches() {
        std::vector<std::string> tmpFiles;
        for (const auto &entry : std::filesystem::directory_iterator(tmpDir)) {
            if (entry.is_regular_file()) {
                std::string fname = entry.path().filename().string();
                // Assume files starting with "H_" or "V_" are boundary files.
                if (fname.rfind("H_", 0) == 0 || fname.rfind("V_", 0) == 0) {
                    tmpFiles.push_back(entry.path().string());
                }
            }
        }
    }


};
