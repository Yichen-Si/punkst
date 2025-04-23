// process_visium.cpp
//
// Usage:
//   g++ -std=c++17 -O3 -o process_visium process_visium.cpp -lz
//
//   ./process_visium \
//     -p 0.2737129726047599 \  // sets microns_per_pixel
//     -S                      \  // (optional) scale coords by mu; if omitted, coords are raw
//     -x "^MT-"              \  // (optional) regex: exclude features whose name matches
//     -i tissue_positions.tsv \  // input TSV from duckdb
//     -b barcodes.tsv.gz      \  // gzipped barcodes list
//     -f features.tsv.gz      \  // gzipped features (gene_id, gene_name)
//     -m matrix.mtx.gz        \  // gzipped matrix market file
//     -o /path/to/output_dir  // output directory

#include "punkst.h"
#include "utils.h"
#include <zlib.h>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <unordered_map>


// simple gz-line reader
static std::vector<std::string> read_gz_lines(const std::string &path) {
    gzFile gz = gzopen(path.c_str(), "rb");
    if (!gz) throw std::runtime_error("Failed to open " + path);
    std::vector<std::string> lines;
    char buf[1<<16];
    while (gzgets(gz, buf, sizeof(buf))) {
        std::string s(buf);
        if (!s.empty() && s.back()=='\n') s.pop_back();
        lines.emplace_back(std::move(s));
    }
    gzclose(gz);
    return lines;
}

int32_t cmdConvertDGE(int argc, char** argv) {
    double mu = 1.0;
    bool original_scale = false;
    bool in_tissue_only = false;
    std::string exclude_regex;
    std::string in_pos, in_bc, in_ft, in_mtx, dge_dir, out_dir;
    int coords_precision = -1;
    uint64_t verbose = 1000000;

    ParamList pl;
    pl.add_option("microns-per-pixel", "Microns per pixel (default: 1.0)", mu)
      .add_option("output-pixel-coordinates", "Writhe original coordinates, not scaled by --microns-per-pixel (default: convert to microns)", original_scale)
      .add_option("exclude-regex", "Regex to exclude features (default: none)", exclude_regex)
      .add_option("in-tissue-only", "Only include barcodes marked as in tissue (default: false)", in_tissue_only)
      .add_option("in-positions", "Input tissue_positions.tsv", in_pos, true)
      .add_option("in-dge-dir", "Input directory for DGE files", dge_dir)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("output-dir", "Output directory", out_dir, true)
      .add_option("coords-precision", "Precision for coordinates (default: none)", coords_precision);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (out_dir.back() == '/') out_dir.pop_back();
    if (!dge_dir.empty() && (in_bc.empty() || in_ft.empty() || in_mtx.empty())) {
        if (dge_dir.back() == '/') dge_dir.pop_back();
        in_bc = dge_dir + "/barcodes.tsv.gz";
        in_ft = dge_dir + "/features.tsv.gz";
        in_mtx = dge_dir + "/matrix.mtx.gz";
    }
    if (abs(mu - 1) < 1e-8) {
        original_scale = true;
    }
    // check files exist
    int file_idx = 0;
    for (const auto &f : {in_pos, in_bc, in_ft, in_mtx}) {
        if (f.empty()) {
            error("Missing required input file (%d)", file_idx);
        }
        std::ifstream fs(f);
        if (!fs) throw std::runtime_error("Failed to open " + f);
        file_idx++;
    }

    // --- 1) parse tissue_positions.tsv, filter in_tissue==1, compute minmax ---
    std::ifstream posfs(in_pos);
    if (!posfs) throw std::runtime_error("Failed to open " + in_pos);
    std::string line;
    std::getline(posfs, line); // header
    std::unordered_map<std::string, std::pair<float,float>> posMap;
    float xmin = std::numeric_limits<float>::max();
    float xmax = std::numeric_limits<float>::lowest();
    float ymin = std::numeric_limits<float>::max();
    float ymax = std::numeric_limits<float>::lowest();

    while (std::getline(posfs, line)) {
        std::vector<std::string> tokens;
        split(tokens, "\t", line);
        if (tokens.size() < 6) {
            warning("Invalid line in %s: %s", in_pos.c_str(), line.c_str());
            continue;
        }
        if (in_tissue_only && tokens[1] == "0") continue;
        float X = std::stof(tokens[4]);
        float Y = std::stof(tokens[5]);
        posMap[tokens[0]] = {X,Y};
        xmin = std::min(xmin, X);
        xmax = std::max(xmax, X);
        ymin = std::min(ymin, Y);
        ymax = std::max(ymax, Y);
    }
    if (!original_scale) {
        xmin *= mu;
        xmax *= mu;
        ymin *= mu;
        ymax *= mu;
    }
    // write coordinate_minmax.tsv
    {
        std::ofstream os(out_dir + "/coordinate_minmax.tsv");
        os << "xmin\t" << xmin << "\n"
           << "xmax\t" << xmax << "\n"
           << "ymin\t" << ymin << "\n"
           << "ymax\t" << ymax << "\n";
    }
    notice("Read transcript locations");

    // --- 2) read barcodes into memory, build index→coord map ---
    auto barcodes = read_gz_lines(in_bc);
    size_t B = barcodes.size();
    std::vector<bool> hasPos(B+1, false);
    std::vector<float> Xc(B+1), Yc(B+1);
    for (size_t i = 1; i <= B; ++i) {
        auto it = posMap.find(barcodes[i-1]);
        if (it != posMap.end()) {
            hasPos[i] = true;
            if (original_scale) {
                Xc[i] = it->second.first;
                Yc[i] = it->second.second;
            } else {
                Xc[i] = it->second.first * mu;
                Yc[i] = it->second.second * mu;
            }
        }
    }
    notice("Read %zu barcodes", B);

    // --- 3) read features, set up exclusion mask ---
    auto flines = read_gz_lines(in_ft);
    std::vector<std::string> features;
    features.reserve(flines.size());
    for (auto &l : flines) {
        std::istringstream is(l);
        std::string id, name;
        is >> id >> name;
        features.push_back(name);
    }
    std::vector<bool> excludeFeat(features.size(), false);
    if (!exclude_regex.empty()) {
        std::regex ex;
        std::stringstream ss;
        int32_t n = 0;
        try { ex = std::regex(exclude_regex); }
        catch (std::regex_error &e) {
            std::cerr << "Invalid regex '" << exclude_regex << "': "
                      << e.what() << "\n";
            return 1;
        }
        for (size_t i = 0; i < features.size(); ++i) {
            if (std::regex_search(features[i], ex)) {
                excludeFeat[i] = true;
                ss << features[i] << " ";
                n++;
            }
        }
        notice("Excluding %d features matching regex '%s': %s",
               n, exclude_regex.c_str(), ss.str().c_str());
    }
    notice("Read %zu features", features.size());

    // --- 4) stream through matrix.mtx.gz → transcripts + geneCounts ---
    gzFile gzm = gzopen(in_mtx.c_str(), "rb");
    if (!gzm) throw std::runtime_error("Cannot open " + in_mtx);
    std::ofstream tout(out_dir + "/transcripts.tsv");
    tout << "#barcode_idx\tx\ty\tgene\tCount\n";
    if (coords_precision > 0)
        tout << std::fixed << std::setprecision(coords_precision);

    std::unordered_map<std::string,uint64_t> geneCounts;
    geneCounts.reserve(features.size());
    char buf[1<<16];
    // skip comments, read dims
    while (gzgets(gzm, buf, sizeof(buf))) {
        if (buf[0]=='%' || buf[0]=='\n') continue;
        break;
    }
    // now process each data line
    uint64_t nrow = 0;
    while (gzgets(gzm, buf, sizeof(buf))) {
        nrow++;
        if (nrow % verbose == 0) {
            notice("Processed %zu lines...", nrow);
        }
        int gi, bi;
        uint32_t ct;
        if (std::sscanf(buf, "%d %d %u", &gi, &bi, &ct) != 3) continue;
        if (bi < 1 || bi > (int)B)  continue;
        if (!hasPos[bi])            continue;
        if (gi < 1 || gi > (int) features.size()) continue;
        if (excludeFeat[gi-1])      continue;
        tout << bi << '\t'
             << Xc[bi] << '\t' << Yc[bi] << '\t'
             << features[gi-1]  << '\t' << ct   << '\n';
        geneCounts[features[gi-1]] += ct;
    }
    gzclose(gzm);

    // --- 5) write features.tsv (gene name + total count)
    std::ofstream fos(out_dir + "/features.tsv");
    for (auto &p : geneCounts)
        fos << p.first << '\t' << p.second << '\n';

    return 0;
}
