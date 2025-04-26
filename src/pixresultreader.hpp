#include "utils.h"
#include "json.hpp"

struct PixelFactorResult {
    double x, y;
    std::vector<int32_t> ks;
    std::vector<float> ps;
};

class PixelResultReader {

public:
    PixelResultReader(std::string& dataFile, std::string indexFile = "", std::string headerFile = "") : _dataFile(dataFile), _indexFile(indexFile) {
        if (!indexFile.empty()) {
            loadIndex(indexFile);
        }
        parseHeader(headerFile);
    }
    ~PixelResultReader() {}

    struct IndexEntryF {
        uint64_t st, ed;
        uint32_t n;
        float xmin, xmax, ymin, ymax;
    };
    struct Block {
        struct IndexEntryF e;
        bool fullyContained;
    };

    int32_t getK() const {
        return k;
    }

    int32_t query(double qxmin_, double qxmax_, double qymin_, double qymax_) {
        this->qxmin = qxmin_;
        this->qxmax = qxmax_;
        this->qymin = qymin_;
        this->qymax = qymax_;
        bounded = true;
        done = false;
        blocks.clear();
        for (auto &b : blocks_all) {
            if (b.e.xmax < qxmin || b.e.xmin > qxmax ||
                b.e.ymax < qymin || b.e.ymin > qymax) {continue;}
            bool inside = (b.e.xmin >= qxmin && b.e.xmax <= qxmax &&
            b.e.ymin >= qymin && b.e.ymax <= qymax);
            blocks.push_back({ b.e, inside });
        }
        idxBlock = 0;
        if (!blocks.empty()) {
            openBlock(blocks[0]);
        } else {
            done = true;
        }
        return int32_t(blocks.size());
    }

    int32_t next(PixelFactorResult& out) {
        if (done) return -1;
        if (bounded) {
            return nextBounded(out);
        }
        std::string line;
        while (true) {
            if (!std::getline(dataStream, line)) {
                done = true;
                return -1;
            }
            if (line.empty() || line[0] == '#') {
                continue;
            }
            PixelFactorResult rec;
            if (!parseLine(line, rec))
                return 0;
            out = std::move(rec);
            return 1;
        }
    }

private:
    std::string  _dataFile, _indexFile;
    uint32_t icol_x, icol_y, maxIdx;
    std::vector<uint32_t> icol_ks, icol_ps;
    int32_t k;
    std::vector<Block> blocks_all, blocks;
    std::ifstream dataStream;
    size_t idxBlock;
    bool bounded = false, done = false;
    double qxmin, qxmax, qymin, qymax;

    bool parseLine(const std::string& line, PixelFactorResult& R) const {
        std::vector<std::string> tok;
        split(tok, "\t", line);
        if (tok.size() < maxIdx+1) return false;
        if (!str2double(tok[icol_x], R.x) ||
            !str2double(tok[icol_y], R.y)) {
            warning("%s: Error parsing x,y from line: %s", __FUNCTION__, line.c_str());
            return false;
        }
        R.ks.resize(k);
        R.ps.resize(k);
        for (int i = 0; i < k; ++i) {
            if (!str2int32(tok[icol_ks[i]], R.ks[i]) ||
                !str2float(tok[icol_ps[i]], R.ps[i])) {
                warning("%s: Error parsing K,P from line: %s", __FUNCTION__, line.c_str());
            }
        }
        return true;
    }

    void parseHeader(std::string headerFile = "") {
        dataStream.open(_dataFile);
        if (!dataStream.is_open()) {
            error("Error opening data file: %s", _dataFile.c_str());
        }
        std::string line;
        k = 1;
        if (headerFile.empty()) {
            // try to get column information from the first line of the data file
            std::getline(dataStream, line);
            if (line.empty()) {
                error("Error reading from data file: %s", _dataFile.c_str());
            }
            if (line[0] == '#') {
                line = line.substr(1);
            }
            std::vector<std::string> tokens;
            split(tokens, "\t", line);
            std::unordered_map<std::string, uint32_t> header;
            for (uint32_t i = 0; i < tokens.size(); ++i) {
                header[tokens[i]] = i;
            }
            if (header.find("x") == header.end() || header.find("y") == header.end()) {
                error("Header file must contain x and y columns");
            }
            icol_x = header["x"];
            icol_y = header["y"];
            while (header.find("K" + std::to_string(k)) != header.end() && header.find("P" + std::to_string(k)) != header.end()) {
                icol_ks.push_back(header["K" + std::to_string(k)]);
                icol_ps.push_back(header["P" + std::to_string(k)]);
                k++;
            }
        } else {
            std::ifstream headerStream(headerFile);
            if (!headerStream.is_open()) {
                error("Error opening header file: %s", headerFile.c_str());
            }
            // Load the JSON header file.
            nlohmann::json header;
            try {
                headerStream >> header;
            } catch (const std::exception& e) {
                error("Error parsing JSON header: %s", e.what());
            }
            headerStream.close();
            icol_x = header["x"];
            icol_y = header["y"];
            while (header.contains("K" + std::to_string(k)) && header.contains("P" + std::to_string(k))) {
                icol_ks.push_back(header["K" + std::to_string(k)]);
                icol_ps.push_back(header["P" + std::to_string(k)]);
                k++;
            }
        }
        if (icol_ks.empty()) {
            error("No K and P columns found in the header");
        }
        k = std::min(3, k-1);
        maxIdx = std::max(icol_x, icol_y);
        for (int i = 0; i < k; ++i) {
            maxIdx = std::max(maxIdx, std::max(icol_ks[i], icol_ps[i]));
        }
    }

    void loadIndex(const std::string& indexFile) {
        std::ifstream in(indexFile, std::ios::binary);
        if (!in.is_open())
            error("Error opening index file: %s", indexFile.c_str());

        blocks_all.clear();
        IndexEntryF e;
        while (in.read(reinterpret_cast<char*>(&e), sizeof(e))) {
            blocks_all.push_back({ e, /*fullyContained=*/false });
        }
        if (blocks_all.empty())
            error("No index entries loaded from %s", indexFile.c_str());
    }

    void openBlock(Block& blk) {
        dataStream.clear();  // clear EOF flags
        dataStream.seekg(blk.e.st);
    }

    int32_t nextBounded(PixelFactorResult& out) {
        if (done) return -1;
        std::string line;
        while (true) {
            auto &blk = blocks[idxBlock];
            std::streampos pos = dataStream.tellg();
            if (!std::getline(dataStream, line)
                || pos >= std::streampos(blk.e.ed))
            {
                // move to next block
                if (++idxBlock >= blocks.size()) {
                    done = true;
                    return -1;
                }
                openBlock(blocks[idxBlock]);
                continue;
            }
            if (line.empty() || line[0] == '#') {
                continue;
            }
            PixelFactorResult rec;
            if (!parseLine(line, rec))
                return 0;
            // if block fullyInside, no need to check coords
            if (blk.fullyContained ||
                (rec.x >= qxmin && rec.x <= qxmax &&
                    rec.y >= qymin && rec.y <= qymax))
            {
                out = std::move(rec);
                return 1;
            }
            // else skip it, keep reading
        }
    }
};
