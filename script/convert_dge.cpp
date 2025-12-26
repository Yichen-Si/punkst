#include "punkst.h"
#include "utils.h"
#include "utils_sys.hpp"
#include <zlib.h>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <regex>
#include <sstream>
#include <unordered_map>
#include <random>
#include "json.hpp"


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

// Convert 10X Genomics (single-cell) 3-file format into a single TSV
// matching the input format for pts2tiles
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



// Convert 10X Genomics (single-cell) 3-file format into a single TSV
// matching the spot/hexagon level format used by punkst
// key\tOrgIndex\tM\tC\tidx0 cnt0\tidx1 cnt1 ...
int32_t cmdConvert10xToHexTSV(int argc, char** argv) {
    std::string in_bc, in_ft, in_mtx, dge_dir;
    std::string outPref;
    uint64_t verbose = 1000000;
    bool sorted_by_barcode = false;
    bool randomize_output = false;
    std::string sort_mem;
    bool use_internal_sort = false;

    ParamList pl;
    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dir)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("sorted-by-barcode", "Input matrix is sorted by barcode, use streaming mode", sorted_by_barcode)
      .add_option("out", "Output prefix", outPref, true)
      .add_option("randomize", "Randomize output order", randomize_output)
      .add_option("sort-mem", "Memory to use for sorting, with units K, M, or G similar to -S in linux sort", sort_mem)
      .add_option("use-internal-sort", "Use internal sort instead of system sort command for randomization (default: false)", use_internal_sort)
      .add_option("verbose", "Verbose", verbose);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (!dge_dir.empty() && (in_bc.empty() || in_ft.empty() || in_mtx.empty())) {
        if (dge_dir.back() == '/') dge_dir.pop_back();
        in_bc = dge_dir + "/barcodes.tsv.gz";
        in_ft = dge_dir + "/features.tsv.gz";
        in_mtx = dge_dir + "/matrix.mtx.gz";
    }

    // Check files exist
    int file_idx = 0;
    for (const auto &f : {in_bc, in_ft, in_mtx}) {
        if (f.empty()) {
            error("Missing required input file (%d)", file_idx);
        }
        std::ifstream fs(f);
        if (!fs) throw std::runtime_error("Failed to open " + f);
        file_idx++;
    }

    // Read barcodes
    auto barcodes = read_gz_lines(in_bc);
    size_t B = barcodes.size();
    if (B == 0) {
        error("No barcodes found in %s", in_bc.c_str());
    }
    notice("Read %zu barcodes", B);

    // Read features
    auto flines = read_gz_lines(in_ft);
    std::vector<std::string> features_ids;
    std::vector<std::string> features_names;
    features_ids.reserve(flines.size());
    features_names.reserve(flines.size());
    for (auto &l : flines) {
        if (l.empty()) continue;
        std::istringstream is(l);
        std::string id, name;
        is >> id >> name; // 10X features.tsv.gz: id, name, type
        if (name.empty()) name = id;
        features_ids.push_back(id);
        features_names.push_back(name);
    }
    size_t F = features_names.size();
    if (F == 0) {
        error("No features found in %s", in_ft.c_str());
    }
    notice("Read %zu features", F);

    // Prepare accumulators
    std::vector<uint64_t> feature_totals(F, 0);
    std::vector<uint64_t> cell_totals; // only used in non-streaming mode
    if (!sorted_by_barcode) cell_totals.assign(B, 0);

    // Stream matrix.mtx.gz and fill structures
    gzFile gzm = gzopen(in_mtx.c_str(), "rb");
    if (!gzm) throw std::runtime_error("Cannot open " + in_mtx);
    char buf[1<<16];
    // Skip comments and header line with dimensions
    while (gzgets(gzm, buf, sizeof(buf))) {
        if (buf[0]=='%' || buf[0]=='\n') continue;
        break; // first non-comment line is the dimension line
    }
    size_t N,M,L;
    if (std::sscanf(buf, "%zu %zu %zu", &N, &M, &L) != 3) {
        warning("Invalid header line in input matrix file (%s)", buf);
    }
    notice("Read header of matrix.mtx.gz: %zu features, %zu barcodes, %zu entries", N, M, L);

    // Prepare output stream and RNG
    std::string outf = outPref + ".tsv";
    std::ofstream out(outf);
    if (!out) error("Cannot open output file: %s", outf.c_str());
    std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<uint32_t> rdUnif(0, UINT32_MAX);
    uint64_t n_units_written = 0;

    auto flush_cell = [&](int cell_idx,
                          std::vector<std::pair<uint32_t,uint32_t>>& acc,
                          uint64_t cell_sum) {
        if (acc.empty() || cell_sum == 0) return;
        uint32_t key = rdUnif(rng);
        // Format: random_key, cell_index, M, C, pairs
        out << uint32toHex(key) << "\t" << cell_idx
            << "\t" << acc.size() << "\t" << cell_sum;
        for (size_t k = 0; k < acc.size(); ++k) {
            out << "\t" << acc[k].first << " " << acc[k].second;
        }
        out << '\n';
        ++n_units_written;
        acc.clear();
    };

    if (sorted_by_barcode) {
        std::vector<std::pair<uint32_t,uint32_t>> acc;
        acc.reserve(std::min(F, (size_t)1024));
        int current_bi = -1; // 0-based
        uint64_t current_sum = 0;
        uint64_t nrow = 0;
        while (gzgets(gzm, buf, sizeof(buf))) {
            if (buf[0] == '\0' || buf[0] == '\n' || buf[0] == '%') continue;
            ++nrow;
            int bi; // 1-based indices
            uint32_t ct, gi;
            if (std::sscanf(buf, "%u %d %u", &gi, &bi, &ct) != 3) continue;
            if (bi < 1 || bi > (int)B) continue;
            if (gi < 1 || gi > (int)F) continue;
            if (ct == 0) continue;
            bi -= 1; gi -= 1; // convert to 0-based
            feature_totals[gi] += ct;
            if (current_bi == -1) current_bi = bi;
            if (current_bi != bi) {
                flush_cell(current_bi, acc, current_sum);
                current_bi = bi;
                current_sum = 0;
                if (current_bi % 1000 == 0) {
                    notice("Flushed up to barcode %d/%zu (line %zu/%zu)", current_bi, B, nrow, L);
                }
            }
            acc.emplace_back(gi, ct);
            current_sum += ct;
        }
        // flush last
        if (current_bi >= 0) flush_cell((size_t)current_bi, acc, current_sum);
        gzclose(gzm);
        out.close();
    } else {
        // Non-streaming: store per-cell vectors
        std::vector<std::vector<std::pair<uint32_t,uint32_t>>> cell_feats(B);
        uint64_t nrow = 0;
        while (gzgets(gzm, buf, sizeof(buf))) {
            ++nrow;
            if (nrow % verbose == 0) {
                notice("Processed %zu/%zu lines...", nrow, L);
            }
            int gi, bi; // 1-based indices in 10X
            uint32_t ct;
            if (std::sscanf(buf, "%d %d %u", &gi, &bi, &ct) != 3) continue;
            if (bi < 1 || bi > (int)B) continue;
            if (gi < 1 || gi > (int)F) continue;
            if (ct == 0) continue;
            cell_feats[bi-1].emplace_back((uint32_t)(gi-1), ct);
            feature_totals[gi-1] += ct;
            cell_totals[bi-1] += ct;
        }
        gzclose(gzm);
        for (size_t i = 0; i < B; ++i) {
            if (cell_totals[i] == 0) continue; // skip empty cells
            auto &vec = cell_feats[i];
            uint32_t key = rdUnif(rng);
            out << uint32toHex(key) << "\t" << i
                << "\t" << vec.size() << "\t" << cell_totals[i];
            for (auto &p : vec) {
                out << "\t" << p.first << " " << p.second;
            }
            out << "\n";
            ++n_units_written;
        }
        out.close();
    }
    if (randomize_output) {
        notice("Shuffling output %s ...", outf.c_str());
        if (use_internal_sort) {
            size_t maxMemBytes = ExternalSorter::parseMemoryString(sort_mem);
            try {
                ExternalSorter::sortBy1stColHex(outf, outf, maxMemBytes);
            } catch (const std::exception& e) {
                warning("Failed to shuffle the output %s: %s", outf.c_str(), e.what());
            }
        } else {
            std::vector<std::string> sort_flags = {"-k1,1", "-o", outf};
            if (!sort_mem.empty()) {
                sort_flags.insert(sort_flags.begin()+1, "-S");
                sort_flags.insert(sort_flags.begin()+2, sort_mem);
            }
            if (sys_sort(outf.c_str(), nullptr, sort_flags) != 0) {
                warning("Failed to shuffle the output %s", outf.c_str());
            }
        }
    }
    notice("Wrote %zu units to %s", n_units_written, outf.c_str());

    // Resolve duplicate feature names using totals
    std::vector<std::string> unique_feature_names(F);
    {
        std::unordered_map<std::string, std::vector<size_t>> groups;
        groups.reserve(F);
        for (size_t j = 0; j < F; ++j) {
            groups[features_names[j]].push_back(j);
        }
        for (auto &kv : groups) {
            auto &idxs = kv.second;
            if (idxs.size() == 1) {
                size_t j = idxs[0];
                unique_feature_names[j] = features_names[j];
            } else {
                size_t winner = idxs[0];
                uint64_t best = feature_totals[winner];
                for (size_t t = 1; t < idxs.size(); ++t) {
                    size_t j = idxs[t];
                    uint64_t v = feature_totals[j];
                    if (v > best || (v == best && j < winner)) {
                        winner = j; best = v;
                    }
                }
                for (size_t j : idxs) {
                    if (j == winner) unique_feature_names[j] = features_names[j];
                    else unique_feature_names[j] = features_ids[j];
                }
            }
        }
    }

    outf = outPref + ".features.tsv";
    std::ofstream fos(outf);
    if (!fos) error("Cannot open output feature file: %s", outf.c_str());
    for (size_t j = 0; j < F; ++j) {
        fos << unique_feature_names[j] << '\t' << feature_totals[j] << '\n';
    }
    fos.close();
    notice("Wrote feature counts to %s", outf.c_str());

    // Write metadata JSON
    std::string out_json = outPref + ".json";
    nlohmann::json meta;
    meta["n_modalities"] = 1;
    meta["random_key"] = 0;
    meta["offset_data"] = 2; // random_key, cell_index
    meta["header_info"] = {"random_key", "cell_index"};
    meta["n_units"] = n_units_written;
    meta["n_features"] = (int)F;
    {
        nlohmann::json dict;
        for (size_t j = 0; j < F; ++j) {
            dict[unique_feature_names[j]] = j;
        }
        meta["dictionary"] = dict;
    }
    std::ofstream jout(out_json);
    if (!jout) error("Cannot open output json file: %s", out_json.c_str());
    jout << std::setw(4) << meta << std::endl;
    jout.close();
    notice("Wrote metadata to %s", out_json.c_str());

    return 0;
}
