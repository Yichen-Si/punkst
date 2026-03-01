#include "tilereader.hpp"

class BoundedBinaryTileIterator {
public:
    //   filename: the binary file name,
    //   start, end: byte offsets for this tile's records,
    //   recordSize: fixed size of one record,
    //   numCat: number of categorical fields,
    //   numInt: number of integer fields,
    //   numFloat: number of float fields.
    BoundedBinaryTileIterator(const std::string &filename,
        std::streampos start, std::streampos end, size_t recordSize,
        int32_t numCat, int32_t numInt, int32_t numFloat)
        : filename(filename), startOffset(start), endOffset(end), recordSize(recordSize), numCat(numCat), numInt(numInt), numFloat(numFloat) {
        file = std::make_unique<std::ifstream>(filename, std::ios::binary);
        if (!file || !file->is_open()) {
            throw std::runtime_error("Failed to open binary file: " + filename);
        }
        file->seekg(startOffset);
    }

    struct Record {
        double x;
        double y;
        std::vector<int32_t> catFields;
        std::vector<int32_t> intFields;
        std::vector<float> floatFields;
        Record() {}
        Record(size_t numCat, size_t numInt, size_t numFloat)
            : catFields(numCat), intFields(numInt), floatFields(numFloat) {}
        Record(double x, double y, size_t numCat, size_t numInt, size_t numFloat)
            : x(x), y(y), catFields(numCat), intFields(numInt), floatFields(numFloat) {}
        void resize(size_t numCat, size_t numInt, size_t numFloat) {
            catFields.resize(numCat);
            intFields.resize(numInt);
            floatFields.resize(numFloat);
        }
    };

    // next() decodes the next fixed-size record into a tab-delimited string.
    bool next(BoundedBinaryTileIterator::Record &record) {
        if (!file || file->tellg() >= endOffset) {
            return false;
        }
        std::vector<char> buffer(recordSize);
        file->read(buffer.data(), recordSize);
        if (file->gcount() != static_cast<std::streamsize>(recordSize)) {
            return false;
        }
        size_t offset = 0;
        record.resize(numCat, numInt, numFloat);
        // Decode x and y coordinates (doubles)
        std::memcpy(&record.x, buffer.data() + offset, sizeof(double));
        offset += sizeof(double);
        std::memcpy(&record.y, buffer.data() + offset, sizeof(double));
        offset += sizeof(double);
        // Decode categorical fields (each stored as int32_t)
        for (int32_t i = 0; i < numCat; ++i) {
            std::memcpy(&record.catFields[i], buffer.data() + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }
        // Decode integer fields (each int32_t)
        for (int i = 0; i < numInt; ++i) {
            std::memcpy(&record.intFields[i], buffer.data() + offset, sizeof(int32_t));
            offset += sizeof(int32_t);
        }
        // Decode float fields (each float)
        for (int i = 0; i < numFloat; ++i) {
            std::memcpy(&record.floatFields, buffer.data() + offset, sizeof(float));
            offset += sizeof(float);
        }
        return true;
    }

private:
    std::string filename;
    std::streampos startOffset;
    std::streampos endOffset;
    size_t recordSize;
    int32_t numCat;
    int32_t numInt;
    int32_t numFloat;
    std::unique_ptr<std::ifstream> file;
};

// currently only for integer coordinates
class BinaryTileReader : public TileReaderBase {
public:
    // The constructor takes:
    //   binFilename: the binary file produced by the writer,
    //   indexFilename: the corresponding index file,
    //   numInt: number of integer fields,
    //   numFloat: number of float fields.
    // The number of categorical fields is deduced from the dictionary header.
    BinaryTileReader(const std::string &binFilename, const std::string &indexFilename, std::vector<Rectangle<double>>* _rects = nullptr, int32_t tileSize = -1)
        : TileReaderBase(binFilename, indexFilename, _rects, tileSize),
            numCat(0), numInt(0), numFloat(0) {
        loadIndex(indexFilename);
        readDictionaries();
    }

    // Return an iterator that decodes records for the given tile.
    std::unique_ptr<BoundedBinaryTileIterator> get_tile_iterator(int tileRow, int tileCol) const {
        TileKey key {tileRow, tileCol};
        auto it = tile_map_.find(key);
        if (it == tile_map_.end()) {
            return nullptr; // Tile not found
        }
        const TileInfo &info = it->second;
        return std::make_unique<BoundedBinaryTileIterator>(inputFile_, info.idx.st, info.idx.ed, recordSize_, numCat, numInt, numFloat);
    }

protected:
    // Reads the index file. Expected index file format:
    //   # tilesize    <tileSize>
    //   # recordsize  <recordSize>
    //   # nvalues     <nCategorical>\t<nInteger>\t<nFloat>
    //   # dictionaries    <startDictOffset>    <endDictOffset>
    //   then one line per tile: row col startOffset endOffset count
    void loadIndex(const std::string &indexFilename) override {
        std::ifstream indexFile(indexFilename);
        if (!indexFile.is_open()) {
            throw std::runtime_error("Unable to open index file: " + indexFilename);
        }
        std::string line;
        // First metadata line: "# tilesize<TAB><tileSize>"
        while (std::getline(indexFile, line)) {
            if (line[0] != '#') {
                break;
            }
            std::istringstream metaStream(line);
            std::string hashtag, key;
            metaStream >> hashtag >> key;
            if (key == "tilesize") {
                if (!(metaStream >> tileSize_)) {
                    throw std::runtime_error("Failed to read tile size from metadata");
                }
            } else if (key == "recordsize") {
                if (!(metaStream >> recordSize_)) {
                    throw std::runtime_error("Failed to read record size from metadata");
                }
            } else if (key == "nvalues") {
                if (!(metaStream >> numCat >> numInt >> numFloat)) {
                    throw std::runtime_error("Failed to read number of values from metadata");
                }
            } else if (key == "dictionaries") {
                if (!(metaStream >> dictStartOffset >> dictEndOffset)) {
                    throw std::runtime_error("Failed to read dictionary offsets from metadata");
                }
            }
        }
        // Read remaining lines for tile index.
        while (!line.empty()) {
            std::istringstream iss(line);
            int row, col;
            std::uint64_t start, end;
            int count;
            if (!(iss >> row >> col >> start >> end >> count)) {
                throw std::runtime_error("Malformed index line: " + line);
            }
            tile_map_.emplace(TileKey{row, col}, TileInfo(start, end));
            if (row < minrow) minrow = row;
            if (col < mincol) mincol = col;
            if (row > maxrow) maxrow = row;
            if (col > maxcol) maxcol = col;
            if (!std::getline(indexFile, line)) {
                break;  // End of file
            }
        }
        nTiles = tile_map_.size();
        indexFile.close();
    }

private:
    int32_t numInt;
    int32_t numFloat;
    int32_t numCat;
    // Categorical dictionaries: one vector<string> per categorical column.
    std::vector<std::vector<std::string>> catDictionaries;
    // Offsets for dictionary region in the binary file.
    std::streamoff dictStartOffset;
    std::streamoff dictEndOffset;

    // Reads the dictionaries from the binary file.
    // The format is:
    //   [uint32_t: number of dictionaries]
    //   For each dictionary:
    //     [uint32_t: number of entries]
    //     [uint64_t: total length of the rest of this record (including delimiters)]
    //     Entries: a single string where each entry is separated by '\t'
    void readDictionaries() {
        std::ifstream infile(inputFile_, std::ios::binary);
        if (!infile.is_open()) {
            throw std::runtime_error("Unable to open binary file: " + inputFile_);
        }
        infile.seekg(dictStartOffset);
        uint32_t numDict = 0;
        infile.read(reinterpret_cast<char*>(&numDict), sizeof(numDict));
        catDictionaries.resize(numDict);
        for (uint32_t i = 0; i < numDict; ++i) {
            uint32_t dictSize = 0;
            infile.read(reinterpret_cast<char*>(&dictSize), sizeof(dictSize));
            uint64_t totalLength = 0;
            infile.read(reinterpret_cast<char*>(&totalLength), sizeof(totalLength));
            std::string dictStr(totalLength, '\0');
            infile.read(&dictStr[0], totalLength);
            // Split the dictionary string using '\t' as delimiter.
            catDictionaries[i].resize(dictSize);
            auto& entries = catDictionaries[i];
            size_t pos = 0, j = 0;
            while (true) {
                size_t tabPos = dictStr.find('\t', pos);
                if (tabPos == std::string::npos) {
                    entries[j] = dictStr.substr(pos);
                    break;
                }
                entries[j++] = dictStr.substr(pos, tabPos - pos);
                pos = tabPos + 1;
            }
            if (entries.size() != dictSize) {
                throw std::runtime_error("Dictionary entries count mismatch");
            }
        }
        infile.close();
    }
};
