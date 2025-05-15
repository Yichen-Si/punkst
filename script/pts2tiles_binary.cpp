#include "punkst.h"
#include "utils.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <thread>
#include <mutex>

class Pts2TilesBinary {
    public:
        Pts2TilesBinary(int32_t numThreads, const std::string& inFile, std::string& tmpDirPath, const std::string& outFile, int32_t tileBuffer = 1000, int32_t nskip = 0) : numThreads(numThreads), inFile(inFile), tmpDir(tmpDirPath), outFile(outFile), tileBuffer(tileBuffer), nskip(nskip), recordSize(0) {
            notice("Created temporary directory: %s", tmpDir.path.string().c_str());
        }

        // Set up the basic geometry (tile size, x/y column indices)
        void initLineParser(int32_t tileSize, int32_t icol_x, int32_t icol_y) {
            this->tileSize = tileSize;
            this->col_x = icol_x;
            this->col_y = icol_y;
            updateRecordSize();
        }

        // Load categorical dictionaries from file(s) and record the corresponding column indecies.
        // Each dictionary file is expected to have one category per line.
        // The order of catDictionaryFiles corresponds to the order of columns in icol_cat.
        void initCategoryDictionaries(const std::vector<std::string>& catDictionaryFiles, const std::vector<int32_t>& icol_cat, bool cat_str = false) {
            catInString = cat_str;
            catColumns = icol_cat;
            catDicts.resize(catDictionaryFiles.size());
            std::vector<int32_t> dictSizes(catDictionaryFiles.size(), 0);
            for (size_t i = 0; i < catDictionaryFiles.size(); ++i) {
                std::ifstream dictFile(catDictionaryFiles[i]);
                if (!dictFile) {
                    throw std::runtime_error("Error opening dictionary file: " + catDictionaryFiles[i]);
                }
                std::string line;
                uint32_t index = 0;
                while (std::getline(dictFile, line)) {
                    // ignore lines starting with '#'
                    if (line.empty() || line[0] == '#') {
                        continue;
                    }
                    // find the first word in the line
                    size_t pos = line.find_first_of(" \t");
                    if (pos != std::string::npos) {
                        line = line.substr(0, pos);
                    }
                    if (catDicts[i].find(line) != catDicts[i].end()) {
                        warning("Duplicate entry in dictionary file will be ignored %s: %s", catDictionaryFiles[i].c_str(), line.c_str());
                    }
                    catDicts[i][line] = index++;
                }
                dictSizes[i] = index;
                notice("Loaded %u entries from dictionary %s", index, catDictionaryFiles[i].c_str());
            }
            updateRecordSize();
        }

        // Specify which columns contain integer features.
        void addIntColumns(const std::vector<int32_t>& icol_int) {
            intColumns = icol_int;
            updateRecordSize();
        }

        // Specify which columns contain float features.
        void addFloatColumns(const std::vector<int32_t>& icol_float) {
            floatColumns = icol_float;
            updateRecordSize();
        }

        // Return the global tile dictionary: tile id -> total number of records.
        const std::unordered_map<int64_t, int64_t>& getGlobalTiles() const {
            return globalTiles;
        }

        bool run() {
            notice("Launching %d worker threads", numThreads);
            if (!launchWorkerThreads()) {
                error("Error launching worker threads");
                return false;
            }
            if (!joinWorkerThreads()) {
                error("Error joining worker threads");
                return false;
            }
            notice("Merging temporary files and writing index");
            if (!mergeAndWriteIndex()) {
                error("Error merging temporary files and writing index");
                return false;
            }
            return true;
        }

    private:
        // Input and configuration.
        int32_t numThreads;
        ScopedTempDir tmpDir;
        std::string inFile, outFile;
        int32_t tileSize;
        int32_t tileBuffer; // Number of records to buffer per tile per thread before flushing
        int32_t nskip;
        int32_t col_x, col_y; // Column indecies for x and y coordinates.
        bool catInString;

        // Column indecies for additional features.
        std::vector<int32_t> catColumns;
        std::vector<int32_t> intColumns;
        std::vector<int32_t> floatColumns;

        // For each categorical column (in the same order as catColumns), store a dictionary mapping strings to integer indices.
        std::vector<std::unordered_map<std::string, uint32_t>> catDicts;

        // Global tile index: tile id -> total record count.
        std::mutex globalTilesMutex;
        std::unordered_map<int64_t, int64_t> globalTiles;

        std::vector<std::thread> threads;

        // Size in bytes of one binary record; computed from the schema.
        size_t recordSize;

        // Update the record size based on the columns currently set.
        // Record layout: [double x][double y][cat fields (int32_t each)]
        //                [int fields (int32_t each)][float fields (float each)]
        void updateRecordSize() {
            recordSize = 2 * sizeof(double); // x and y.
            recordSize += catColumns.size() * sizeof(uint32_t);
            recordSize += intColumns.size() * sizeof(int32_t);
            recordSize += floatColumns.size() * sizeof(float);
        }

        // Compute tile id based on x and y coordinates.
        int64_t computeTileId(double x, double y) {
            int32_t row = static_cast<int32_t>(std::floor(y / tileSize));
            int32_t col = static_cast<int32_t>(std::floor(x / tileSize));
            return (static_cast<int64_t>(row) << 32) | (static_cast<uint32_t>(col));
        }

        // Compute block boundaries for parallelization
        void computeBlocks(std::vector<std::pair<std::streampos, std::streampos>>& blocks) {
            std::ifstream infile(inFile, std::ios::binary);
            if (!infile) {
                error("Error opening input file: %s", inFile.c_str());
            }
            std::string line;
            for (int i = 0; i < nskip; ++i) {
                std::getline(infile, line);
            }
            std::streampos start_offset = infile.tellg();
            infile.seekg(0, std::ios::end);
            std::streampos fileSize = infile.tellg();
            size_t blockSize = fileSize / numThreads;

            std::streampos current = start_offset;
            blocks.clear();
            for (int i = 0; i < numThreads; ++i) {
                std::streampos end = current + static_cast<std::streamoff>(blockSize);
                if (end > fileSize || i == numThreads - 1) {
                    end = fileSize;
                } else {
                    infile.seekg(end);
                    std::getline(infile, line);
                    end = infile.tellg();
                    if (end == -1) {
                        end = fileSize;
                    }
                }
                blocks.emplace_back(current, end);
                current = end;
                if (current >= fileSize) {
                    break;
                }
            }
            infile.close();
            notice("Partitioned input file into %zu blocks of size ~ %zu", blocks.size(), blockSize);
        }

        // Each worker thread processes its assigned file block.
        void worker(int threadId, std::streampos start, std::streampos end) {
            std::ifstream file(inFile);
            if (!file) {
                error("Error opening file in worker");
                return;
            }
            file.seekg(start);
            std::string line;
            std::unordered_map<int64_t, std::vector<char>> buffers;

            // Process lines until we reach the block end.
            while (file.tellg() < end && std::getline(file, line)) {
                std::vector<std::string> tokens;
                split(tokens, "\t", line);
                std::vector<char> record;
                double x, y;
                if (!createRecord(record, x, y, tokens)) {
                    continue;
                }
                int64_t tileId = computeTileId(x, y);
                auto& buf = buffers[tileId];
                buf.insert(buf.end(), record.begin(), record.end());
                if (buf.size() >= recordSize * tileBuffer) {
                    {
                        std::lock_guard<std::mutex> lock(globalTilesMutex);
                        globalTiles[tileId] += (buf.size() / recordSize);
                    }
                    auto tmpFilename = tmpDir.path / (std::to_string(tileId) + "_" + std::to_string(threadId) + ".bin");
                    std::ofstream out(tmpFilename, std::ios::binary | std::ios::app);
                    out.write(buf.data(), buf.size());
                    out.close();
                    buf.clear();
                }
            }
            // Flush any remaining buffers.
            for (auto& pair : buffers) {
                if (!pair.second.empty()) {
                    {
                        std::lock_guard<std::mutex> lock(globalTilesMutex);
                        globalTiles[pair.first] += (pair.second.size() / recordSize);
                    }
                    auto tmpFilename = tmpDir.path / (std::to_string(pair.first) + "_" + std::to_string(threadId) + ".bin");
                    std::ofstream out(tmpFilename, std::ios::binary | std::ios::app);
                    out.write(pair.second.data(), pair.second.size());
                    out.close();
                }
            }
        }

        // Launch worker threads to process the input file.
        bool launchWorkerThreads() {
            std::vector<std::pair<std::streampos, std::streampos>> blocks;
            computeBlocks(blocks);
            for (size_t i = 0; i < blocks.size(); i++) {
                threads.emplace_back(&Pts2TilesBinary::worker, this, static_cast<int>(i), blocks[i].first, blocks[i].second);
            }
            return true;
        }

        // Wait for all worker threads to complete.
        bool joinWorkerThreads() {
            for (auto& t : threads) {
                t.join();
            }
            return true;
        }

        // Create a binary record from the given tokens.
        // The record is built in the following order:
        //   x (double), y (double), [categorical fields (uint32_t)], [integer fields (int32_t)], [float fields (float)]
        bool createRecord(std::vector<char>& record, double &x, double &y, std::vector<std::string>& tokens) {
            // Categorical features: convert string to integer index using the appropriate dictionary.
            std::vector<uint32_t> catIndices(catColumns.size(), 0);
            if (catInString) {
                for (size_t i = 0; i < catColumns.size(); ++i) {
                    auto it = catDicts[i].find(tokens[catColumns[i]]);
                    if (it == catDicts[i].end()) {
                        return false;
                    }
                    catIndices[i] = it->second;
                }
            } else {
                for (size_t i = 0; i < catColumns.size(); ++i) {
                    if (!str2uint32(tokens[catColumns[i]], catIndices[i])) {
                        return false;
                    }
                }
            }
            record.resize(recordSize);
            size_t offset = 0;
            // Coordinates
            x = std::stod(tokens[col_x]);
            std::memcpy(&record[offset], &x, sizeof(double));
            offset += sizeof(double);
            y = std::stod(tokens[col_y]);
            std::memcpy(&record[offset], &y, sizeof(double));
            offset += sizeof(double);
            // Categorical indices
            for (size_t i = 0; i < catColumns.size(); ++i) {
                std::memcpy(&record[offset], &catIndices[i], sizeof(uint32_t));
                offset += sizeof(uint32_t);
            }
            // Integer features.
            for (size_t i = 0; i < intColumns.size(); ++i) {
                int colIndex = intColumns[i];
                int32_t intValue = std::stoi(tokens[colIndex]);
                std::memcpy(&record[offset], &intValue, sizeof(int32_t));
                offset += sizeof(int32_t);
            }
            // Float features.
            for (size_t i = 0; i < floatColumns.size(); ++i) {
                int colIndex = floatColumns[i];
                float floatValue = std::stof(tokens[colIndex]);
                std::memcpy(&record[offset], &floatValue, sizeof(float));
                offset += sizeof(float);
            }
            return true;
        }

        // Write the categorical dictionaries (self-describing) to the output file.
        // The format written is:
        // [uint32_t: number of dictionaries]
        // For each dictionary:
        //    [uint32_t: number of entries]
        //    [uint64_t: total length of the rest of this record (including all the entires and the delmiters)]
        //    Entries: each entry is a char[] for the string; entries are separated by '\t'.
        void writeDictionaries(std::ofstream& outfile) {
            uint32_t numDict = catDicts.size();
            outfile.write(reinterpret_cast<const char*>(&numDict), sizeof(numDict));
            for (const auto& dict : catDicts) {
                std::vector<std::string> keys;
                keys.resize(dict.size());
                for (const auto& kv : dict) {
                    keys[kv.second] = kv.first;
                }
                // Write the dictionary size.
                uint32_t dictSize = dict.size();
                outfile.write(reinterpret_cast<const char*>(&dictSize), sizeof(dictSize));
                // Write the total length of this record (including all the entries and the delimiter).
                uint64_t totalLength = 0;
                for (const auto& kv : dict) {
                    totalLength += kv.first.size();
                }
                totalLength += dictSize - 1; // delimiter between entries
                outfile.write(reinterpret_cast<const char*>(&totalLength), sizeof(totalLength));
                // Write the entries.
                for (size_t i = 0; i < dictSize; ++i) {
                    outfile.write(keys[i].data(), keys[i].size());
                    if (i < dictSize - 1) {
                        outfile.put('\t'); // delimiter
                    }
                }
            }
        }

        // Merge the temporary binary files for a given tile id into the final output file.
        void mergeTmpFileToOutput(int64_t tileId, std::ofstream& outfile) {
            for (uint32_t threadId = 0; threadId < static_cast<uint32_t>(numThreads); ++threadId) {
                auto tmpFilename = tmpDir.path / (std::to_string(tileId) + "_" + std::to_string(threadId) + ".bin");
                std::ifstream tmpFile(tmpFilename, std::ios::binary);
                if (tmpFile) {
                    outfile << tmpFile.rdbuf();
                    tmpFile.close();
                    // Remove temporary file after merging.
                    std::filesystem::remove(tmpFilename);
                }
            }
        }

        // Merge temporary files by tile and write index file
        bool mergeAndWriteIndex() {
            std::ofstream outfile(outFile, std::ios::binary);
            if (!outfile) {
                std::cerr << "Error opening output file: " << outFile << std::endl;
                return false;
            }

            std::string indexFilename = outFile + ".index";
            std::ofstream indexfile(indexFilename);
            if (!indexfile) {
                std::cerr << "Error opening index file: " << indexFilename << std::endl;
                return false;
            }
            indexfile << "# tilesize\t" << tileSize << "\n";

            int64_t nPoints = 0;
            auto globalTiles = getGlobalTiles();
            std::vector<int64_t> sortedTiles;
            for (const auto& pair : globalTiles) {
                sortedTiles.push_back(pair.first);
                nPoints += pair.second;
            }
            indexfile << "# npixels\t" << nPoints << "\n";
            indexfile << "# recordsize\t" << recordSize << "\n";
            indexfile << "# nvalues\t" << catColumns.size() + intColumns.size() + floatColumns.size() << "\n";

            std::streampos startDictOffset = outfile.tellp();
            writeDictionaries(outfile);
            std::streampos endDictOffset = outfile.tellp();
            indexfile << "# dictionaries\t" << startDictOffset << "\t" << endDictOffset << "\n";

            std::sort(sortedTiles.begin(), sortedTiles.end());
            for (const auto& tileId : sortedTiles) {
                std::streampos startOffset = outfile.tellp();
                mergeTmpFileToOutput(tileId, outfile);
                std::streampos endOffset = outfile.tellp();
                indexfile << (int32_t)(tileId >> 32) << "\t" << (int32_t)(tileId & 0xFFFFFFFF) << "\t" << startOffset << "\t" << endOffset << "\t" << globalTiles[tileId] << "\n";
            }

            outfile.close();
            indexfile.close();
            return true;
        }

    };

int32_t cmdPts2TilesBinary(int32_t argc, char** argv) {

    std::string inTsv, output, tmpDirPath;
    int nThreads = 1, tileSize = 500000;
    int debug = 0, tileBuffer = 1000, verbose = 1000000;
    int icol_x, icol_y, nskip = 0;
    std::vector<int32_t> icol_cat, icol_int, icol_float;
    std::vector<std::string> catDictionaries;
    bool catInString = false;

    ParamList pl;
    // Input Options
    pl.add_option("in-tsv", "Input TSV file. Header must begin with #", inTsv)
      .add_option("cat-dict", "Dictionary for categorical data", catDictionaries)
      .add_option("icol-x", "Column index for x coordinate (0-based)", icol_x)
      .add_option("icol-y", "Column index for y coordinate (0-based)", icol_y)
      .add_option("icol-cat", "Column indecies for categorical data (0-based)", icol_cat)
      .add_option("cat-in-string", "The categorical column contains strings in the provided dictionary (default: false, assuming the categorical column contains non-negative integers as indices)", catInString)
      .add_option("icol-int", "Column indecies for integer data (0-based)", icol_int)
      .add_option("icol-float", "Column indecies for float data (0-based)", icol_float)
      .add_option("skip", "Number of lines to skip in the input file (default: 0)", nskip)
      .add_option("temp-dir", "Directory to store temporary files", tmpDirPath)
      .add_option("tile-size", "Tile size in units (default: 300 um)", tileSize)
      .add_option("tile-buffer", "Buffer size per tile per thread (default: 1000 lines)", tileBuffer)
      .add_option("threads", "Number of threads to use (default: 1)", nThreads);
    // Output Options
    pl.add_option("out", "Output TSV file", output)
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

    if (catDictionaries.size() != icol_cat.size()) {
        error("Number of categorical dictionaries must match the number of categorical columns");
        return 1;
    }

    // Determine the number of threads to use.
    unsigned int numThreads = std::thread::hardware_concurrency();
    if (numThreads == 0 || numThreads > nThreads) {
        numThreads = nThreads;
    }
    notice("Will use %u threads for processing", numThreads);

    Pts2TilesBinary pts2Tiles(numThreads, inTsv, tmpDirPath, output);
    pts2Tiles.initCategoryDictionaries(catDictionaries, icol_cat, catInString);
    pts2Tiles.initLineParser(tileSize, icol_x, icol_y);
    pts2Tiles.addIntColumns(icol_int);
    pts2Tiles.addFloatColumns(icol_float);
    if (!pts2Tiles.run()) {
        return 1;
    }

    notice("Processing completed. Output is written to %s and index file is written to %s.index", output.c_str(), output.c_str());
    return 0;
}
