#include "punkst.h"
#include "dataunits.hpp"
#include "image_pmtiles.hpp"
#include "image_utils.hpp"
#include "mlt_utils.hpp"
#include "pmtiles_utils.hpp"
#include "pmtiles_pyramid.hpp"
#include "tile_io.hpp"
#include "tileoperator.hpp"
#include "tiles2mono.hpp"
#include "utils.h"
#include "utils_sys.hpp"

#include "json.hpp"

#include <algorithm>
#include <cinttypes>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <array>
#include <set>
#include <sstream>
#include <cstring>
#include <future>
#include <functional>
#include <tuple>

int32_t cmdManipulateTiles(int32_t argc, char** argv);
int32_t cmdPoly2Pmtiles(int32_t argc, char** argv);
int32_t cmdCells2Pmtiles(int32_t argc, char** argv);
int32_t cmdDeChisq(int32_t argc, char** argv);

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

enum class PixelDecodeMode {
    Pixel,
    FeaturePixel,
    SingleMolecule
};

struct TranscriptInputs {
    fs::path tiledPrefix;
    fs::path featureCountTsv;
    std::string nullStr;
    int32_t icolX = 0;
    int32_t icolY = 1;
    int32_t icolFeature = 2;
    int32_t icolCount = 3;
};

struct CellInputs {
    std::string id;
    fs::path resultsTsv;
    fs::path boundaries;
    fs::path centersTsv;
    fs::path cellsPmtiles;
    fs::path boundariesPmtiles;
    fs::path deTsv;
    fs::path pseudobulkTsv;
    fs::path rgbTsv;
    std::string boundaryFormat = "auto";
    std::string boundaryIdProp = "cell_id";
    int32_t tIcolId = 0;
    int32_t bIcolId = 0;
    int32_t bIcolX = 1;
    int32_t bIcolY = 2;
    int32_t cIcolId = 0;
    int32_t cIcolX = 1;
    int32_t cIcolY = 2;
    int32_t factorColBegin = -1;
    int32_t factorColEnd = -1;
    std::string resultIdCol = "cell_id";
};

enum class ModelType {
    Default,
    CellOnly
};

struct ModelInputs {
    std::string sourcePrefix;
    std::string id;
    ModelType type = ModelType::Default;
    PixelDecodeMode pixelMode = PixelDecodeMode::Pixel;
    double hexGridDist = 0.0;
    fs::path resultsTsv;
    fs::path modelTsv;
    fs::path colorRgbTsv;
    fs::path pixelPrefix;
    fs::path pixelPng;
    fs::path pseudobulkTsv;
    fs::path deTsv;
    CellInputs cell;
    int32_t factorColBegin = -1;
    int32_t factorColEnd = -1;
    bool pixelPngExplicit = false;
    bool skipHexPmtiles = false;
};

struct DeployInputs {
    TranscriptInputs transcripts;
    std::vector<ModelInputs> models;
    std::vector<fs::path> imageAssetJsons;
    std::vector<image_pmtiles::Options> images;
};

struct RgbColorEntry {
    std::string name;
    std::string colorIndex;
    std::array<int32_t, 3> rgb{0, 0, 0};
};

std::vector<std::string> factor_names_from_pseudobulk_header(const fs::path& src);

struct DeGeneRow {
    std::string gene;
    std::string factor;
    double foldChange = 0.0;
    double log10pval = 0.0;
};

std::string normalize_id(std::string s) {
    for (char& c : s) {
        if (c == '_' || c == '.' || c == '/' || c == '\\') {
            c = '-';
        }
    }
    return s;
}

std::string pixel_mode_name(PixelDecodeMode mode) {
    switch (mode) {
    case PixelDecodeMode::Pixel:
        return "pixel";
    case PixelDecodeMode::FeaturePixel:
        return "feature_pixel";
    case PixelDecodeMode::SingleMolecule:
        return "single_molecule";
    }
    return "pixel";
}

PixelDecodeMode parse_pixel_mode(const std::string& raw, const char* context) {
    const std::string mode = to_lower(raw);
    if (mode == "pixel") {
        return PixelDecodeMode::Pixel;
    }
    if (mode == "feature_pixel" || mode == "sf_pixel" || mode == "single_feature_pixel") {
        return PixelDecodeMode::FeaturePixel;
    }
    if (mode == "single_molecule" || mode == "sgl_mol") {
        return PixelDecodeMode::SingleMolecule;
    }
    error("%s: unsupported pixel_decode_mode '%s' in %s", __func__, raw.c_str(), context);
    return PixelDecodeMode::Pixel;
}

ModelType parse_model_type(const std::string& raw, const char* context) {
    const std::string type = to_lower(raw);
    if (type == "default") {
        return ModelType::Default;
    }
    if (type == "cell_only") {
        return ModelType::CellOnly;
    }
    error("%s: unsupported model type '%s' in %s", __func__, raw.c_str(), context);
    return ModelType::Default;
}

bool is_default_model(const ModelInputs& model) {
    return model.type == ModelType::Default;
}

bool is_cell_only_model(const ModelInputs& model) {
    return model.type == ModelType::CellOnly;
}

bool has_cell_pmtiles_or_sources(const CellInputs& cell) {
    return !cell.cellsPmtiles.empty() || !cell.boundariesPmtiles.empty() ||
        !cell.resultsTsv.empty() || !cell.boundaries.empty();
}

bool has_cell_sidecars(const CellInputs& cell) {
    return !cell.deTsv.empty() || !cell.pseudobulkTsv.empty() || !cell.rgbTsv.empty();
}

bool has_factor_column_range(int32_t begin, int32_t end) {
    return begin >= 0 || end >= 0;
}

void validate_factor_column_range(int32_t begin, int32_t end,
    size_t expectedFactorCount, const std::string& context) {
    if (!has_factor_column_range(begin, end)) {
        return;
    }
    if (begin < 0 || end < 0) {
        error("%s: factor_col_begin and factor_col_end must be provided together for %s",
            __func__, context.c_str());
    }
    if (end < begin) {
        error("%s: factor_col_end must be >= factor_col_begin for %s",
            __func__, context.c_str());
    }
    const size_t width = static_cast<size_t>(end - begin + 1);
    if (width != expectedFactorCount) {
        error("%s: factor column range for %s has %zu columns, but pseudobulk has %zu factors",
            __func__, context.c_str(), width, expectedFactorCount);
    }
}

std::string pixel_mode_suffix(PixelDecodeMode mode) {
    switch (mode) {
    case PixelDecodeMode::Pixel:
        return "pixel";
    case PixelDecodeMode::FeaturePixel:
        return "sf_pixel";
    case PixelDecodeMode::SingleMolecule:
        return "sgl_mol";
    }
    return "pixel";
}

std::string default_model_id(int32_t hex, int32_t topics, PixelDecodeMode mode) {
    std::string id = "h" + std::to_string(hex) + "-k" + std::to_string(topics);
    if (mode == PixelDecodeMode::FeaturePixel) {
        id += "-sf-pixel";
    } else if (mode == PixelDecodeMode::SingleMolecule) {
        id += "-sgl-mol";
    }
    return id;
}

int32_t pixel_mode_specificity(PixelDecodeMode mode) {
    switch (mode) {
    case PixelDecodeMode::SingleMolecule:
        return 2;
    case PixelDecodeMode::FeaturePixel:
        return 1;
    case PixelDecodeMode::Pixel:
        return 0;
    }
    return 0;
}

PixelDecodeMode infer_pixel_mode_from_index(const fs::path& pixelPrefix) {
    LoadedTileIndexData loaded = loadTileIndexData(pixelPrefix.string() + ".index");
    const uint32_t mode = loaded.header.mode & 0xFFFFu;
    if ((mode & 0x40u) == 0u) {
        return PixelDecodeMode::Pixel;
    }
    if ((mode & 0x4u) != 0u) {
        return PixelDecodeMode::FeaturePixel;
    }
    return PixelDecodeMode::SingleMolecule;
}

std::string pixel_decode_id(const ModelInputs& model) {
    return model.id + "-pixel";
}

bool use_png_raster_for_model(const ModelInputs& model, bool usePngFlag, bool configMode) {
    return configMode ? usePngFlag : model.pixelPngExplicit;
}

std::string join_paths(const std::vector<fs::path>& paths) {
    std::vector<std::string> text;
    text.reserve(paths.size());
    for (const fs::path& p : paths) {
        text.push_back(p.string());
    }
    return join(text, ", ");
}

int call_command(const std::vector<std::string>& args,
    const std::function<int32_t(int32_t, char**)>& fn) {
    std::vector<std::string> owned = args;
    std::vector<char*> argv;
    argv.reserve(owned.size());
    for (std::string& arg : owned) {
        argv.push_back(arg.data());
    }
    return fn(static_cast<int32_t>(argv.size()), argv.data());
}

std::string normalize_factor_id(std::string s) {
    try {
        size_t pos = 0;
        const double v = std::stod(s, &pos);
        if (pos == s.size() && std::isfinite(v) && std::floor(v) == v) {
            return std::to_string(static_cast<int64_t>(v));
        }
    } catch (...) {
    }
    return s;
}

std::vector<std::string> read_pmtiles_paths_from_index(const fs::path& indexPath, const fs::path& outDir) {
    std::ifstream in(indexPath);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, indexPath.string().c_str());
    }
    std::string line;
    if (!std::getline(in, line)) {
        error("%s: empty PMTiles index %s", __func__, indexPath.string().c_str());
    }
    std::vector<std::string> header = split_tab(line);
    int32_t pathCol = -1;
    for (size_t i = 0; i < header.size(); ++i) {
        if (header[i] == "pmtiles_path") {
            pathCol = static_cast<int32_t>(i);
        }
    }
    if (pathCol < 0) {
        error("%s: PMTiles index missing pmtiles_path column: %s", __func__, indexPath.string().c_str());
    }

    std::vector<std::string> paths;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields = split_tab(line);
        if (static_cast<int32_t>(fields.size()) <= pathCol) {
            error("%s: malformed PMTiles index row in %s", __func__, indexPath.string().c_str());
        }
        paths.push_back((outDir / fields[static_cast<size_t>(pathCol)]).string());
    }
    return paths;
}

std::vector<fs::path> planned_point_outputs(const fs::path& outRoot, int32_t nGeneBins) {
    std::vector<fs::path> paths;
    paths.push_back(outRoot / "genes_pmtiles_index.tsv");
    paths.push_back(outRoot / "genes_bin_counts.json");
    paths.push_back(outRoot / "genes_all.pmtiles");
    for (int32_t b = 1; b <= nGeneBins; ++b) {
        paths.push_back(outRoot / ("genes_bin" + std::to_string(b) + ".pmtiles"));
    }
    return paths;
}

bool point_outputs_complete(const fs::path& outRoot, int32_t nGeneBins,
    std::vector<std::string>* pointPmtiles) {
    const fs::path deployIndex = outRoot / "genes_pmtiles_index.tsv";
    const fs::path deployCounts = outRoot / "genes_bin_counts.json";
    if (file_exists(deployIndex) && file_exists(deployCounts)) {
        std::vector<std::string> paths = read_pmtiles_paths_from_index(deployIndex, outRoot);
        std::vector<fs::path> required{deployIndex, deployCounts};
        for (const std::string& p : paths) {
            required.emplace_back(p);
        }
        if (all_files_exist(required)) {
            if (pointPmtiles != nullptr) {
                *pointPmtiles = std::move(paths);
            }
            return true;
        }
    }
    return all_files_exist(planned_point_outputs(outRoot, nGeneBins));
}

std::string make_transcript_header(const TranscriptInputs& tr) {
    const int32_t n = std::max({tr.icolX, tr.icolY, tr.icolFeature, tr.icolCount}) + 1;
    std::vector<std::string> names(static_cast<size_t>(n));
    for (int32_t i = 0; i < n; ++i) {
        names[static_cast<size_t>(i)] = "col" + std::to_string(i);
    }
    names[static_cast<size_t>(tr.icolX)] = "x";
    names[static_cast<size_t>(tr.icolY)] = "y";
    names[static_cast<size_t>(tr.icolFeature)] = "feature";
    names[static_cast<size_t>(tr.icolCount)] = "count";
    return "#" + join(names, "\t") + "\n";
}

void write_shifted_index(const fs::path& srcIndex, const fs::path& dstIndex, uint64_t offset) {
    LoadedTileIndexData loaded = loadTileIndexData(srcIndex.string());
    for (IndexEntryF& entry : loaded.entries) {
        entry.st += offset;
        entry.ed += offset;
    }

    std::ofstream out(dstIndex, std::ios::binary);
    if (!out.is_open()) {
        error("%s: cannot open %s", __func__, dstIndex.string().c_str());
    }

    IndexHeader header = loaded.header;
    header.magic = PUNKST_INDEX_MAGIC;
    if ((header.mode & 0x40u) != 0u) {
        header.featureCount = static_cast<uint32_t>(loaded.featureNames.size());
        header.featureNameSize = computeFeatureNameSizeFixed(loaded.featureNames);
    }
    out.write(reinterpret_cast<const char*>(&header), sizeof(header));
    if ((header.mode & 0x40u) != 0u) {
        const size_t payloadBytes =
            static_cast<size_t>(header.featureCount) * static_cast<size_t>(header.featureNameSize);
        std::vector<char> payload(payloadBytes, '\0');
        for (size_t i = 0; i < loaded.featureNames.size(); ++i) {
            const std::string& name = loaded.featureNames[i];
            if (name.size() >= header.featureNameSize) {
                error("%s: feature name '%s' exceeds fixed width", __func__, name.c_str());
            }
            std::memcpy(payload.data() + i * header.featureNameSize, name.data(), name.size());
        }
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }
    for (const IndexEntryF& entry : loaded.entries) {
        out.write(reinterpret_cast<const char*>(&entry), sizeof(entry));
    }
    if (!out.good()) {
        error("%s: failed writing %s", __func__, dstIndex.string().c_str());
    }
}

fs::path prepare_headered_transcripts(const TranscriptInputs& tr, const fs::path& tmpDir) {
    const fs::path srcTsv = tr.tiledPrefix.string() + ".tsv";
    std::ifstream probe(srcTsv);
    if (!probe.is_open()) {
        error("%s: cannot open %s", __func__, srcTsv.string().c_str());
    }
    std::string line;
    while (std::getline(probe, line)) {
        if (line.empty()) {
            continue;
        }
        if (!line.empty() && line[0] == '#') {
            return tr.tiledPrefix;
        }
        break;
    }

    fs::create_directories(tmpDir);
    const fs::path dstPrefix = tmpDir / "transcripts.tiled.headered";
    const fs::path dstTsv = dstPrefix.string() + ".tsv";
    const fs::path dstIndex = dstPrefix.string() + ".index";
    const std::string header = make_transcript_header(tr);

    std::ofstream out(dstTsv, std::ios::binary);
    if (!out.is_open()) {
        error("%s: cannot open %s", __func__, dstTsv.string().c_str());
    }
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    std::ifstream in(srcTsv, std::ios::binary);
    out << in.rdbuf();
    if (!out.good()) {
        error("%s: failed writing %s", __func__, dstTsv.string().c_str());
    }
    write_shifted_index(tr.tiledPrefix.string() + ".index", dstIndex, header.size());
    return dstPrefix;
}

fs::path prepare_hex_results_with_topk(const ModelInputs& model, const fs::path& tmpDir) {
    if (has_factor_column_range(model.factorColBegin, model.factorColEnd)) {
        return model.resultsTsv;
    }
    std::ifstream in(model.resultsTsv);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, model.resultsTsv.string().c_str());
    }
    std::string headerLine;
    while (std::getline(in, headerLine)) {
        if (!headerLine.empty()) {
            break;
        }
    }
    if (headerLine.empty()) {
        error("%s: empty results file %s", __func__, model.resultsTsv.string().c_str());
    }
    std::string headerNoHash = headerLine;
    if (!headerNoHash.empty() && headerNoHash[0] == '#') {
        headerNoHash.erase(headerNoHash.begin());
    }
    std::vector<std::string> header = split_tab(headerNoHash);
    if (header_has(header, "topK") && header_has(header, "topP")) {
        return model.resultsTsv;
    }

    std::vector<int32_t> factorCols;
    std::vector<int32_t> factorIds;
    for (size_t i = 0; i < header.size(); ++i) {
        int32_t k = -1;
        if (str2int32(header[i], k) && k >= 0) {
            factorCols.push_back(static_cast<int32_t>(i));
            factorIds.push_back(k);
        }
    }
    if (factorCols.empty()) {
        const std::vector<std::string> factorNames =
            factor_names_from_pseudobulk_header(model.pseudobulkTsv);
        std::map<std::string, int32_t> colByName;
        for (size_t i = 0; i < header.size(); ++i) {
            colByName[header[i]] = static_cast<int32_t>(i);
        }
        for (size_t k = 0; k < factorNames.size(); ++k) {
            auto it = colByName.find(factorNames[k]);
            if (it == colByName.end()) {
                error("%s: results table %s is missing model factor column '%s' from pseudobulk header",
                    __func__, model.resultsTsv.string().c_str(), factorNames[k].c_str());
            }
            factorCols.push_back(it->second);
            factorIds.push_back(static_cast<int32_t>(k));
        }
        notice("%s: mapped named factor columns in %s to numeric factor ids using %s",
            __func__, model.resultsTsv.string().c_str(), model.pseudobulkTsv.string().c_str());
    }

    fs::create_directories(tmpDir);
    fs::path outPath = tmpDir / (model.id + ".results.topk.tsv");
    std::ofstream out(outPath);
    if (!out.is_open()) {
        error("%s: cannot open %s", __func__, outPath.string().c_str());
    }
    out << "#";
    bool inserted = false;
    for (size_t i = 0; i < header.size(); ++i) {
        if (i > 0) {
            out << "\t";
        }
        out << header[i];
        if (!inserted && header[i] == "y") {
            out << "\ttopK\ttopP";
            inserted = true;
        }
    }
    if (!inserted) {
        out << "\ttopK\ttopP";
    }
    out << "\n";

    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields = split_tab(line);
        if (fields.size() < header.size()) {
            error("%s: malformed row in %s", __func__, model.resultsTsv.string().c_str());
        }
        int32_t bestK = -1;
        float bestP = -1.0f;
        for (size_t j = 0; j < factorCols.size(); ++j) {
            const int32_t col = factorCols[j];
            float value = 0.0f;
            if (!str2float(fields[static_cast<size_t>(col)], value)) {
                error("%s: failed parsing factor probability in %s", __func__, model.resultsTsv.string().c_str());
            }
            if (value > bestP) {
                bestP = value;
                bestK = factorIds[j];
            }
        }
        for (size_t i = 0; i < fields.size(); ++i) {
            if (i > 0) {
                out << "\t";
            }
            out << fields[i];
            if (!inserted && false) {
                // unreachable; keeps compilers from warning about insertion state in older toolchains
            }
            if (header[i] == "y") {
                out << "\t" << bestK << "\t" << fp_to_string(bestP, 6);
            }
        }
        if (!header_has(header, "y")) {
            out << "\t" << bestK << "\t" << fp_to_string(bestP, 6);
        }
        out << "\n";
    }
    if (!out.good()) {
        error("%s: failed writing %s", __func__, outPath.string().c_str());
    }
    return outPath;
}

std::vector<RgbColorEntry> load_normalized_colors(const fs::path& src, size_t k) {
    if (k == 0) {
        error("%s: factor count must be positive for %s", __func__, src.string().c_str());
    }
    std::ifstream in(src);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, src.string().c_str());
    }
    std::string headerLine;
    if (!std::getline(in, headerLine)) {
        error("%s: empty color table %s", __func__, src.string().c_str());
    }
    std::vector<std::string> header = split_tab(headerLine);
    std::map<std::string, int32_t> col;
    for (size_t i = 0; i < header.size(); ++i) {
        col[header[i]] = static_cast<int32_t>(i);
    }
    if (!col.count("R") || !col.count("G") || !col.count("B")) {
        error("%s: color table must contain R/G/B columns: %s", __func__, src.string().c_str());
    }
    const bool hasName = col.count("Name") > 0;
    const bool hasColorIndex = col.count("Color_index") > 0;

    std::vector<RgbColorEntry> sourceColors;
    std::string line;
    int32_t row = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields = split_tab(line);
        double r = 0.0, g = 0.0, b = 0.0;
        if (fields.size() <= static_cast<size_t>(std::max({col["R"], col["G"], col["B"]})) ||
            !str2double(fields[static_cast<size_t>(col["R"])], r) ||
            !str2double(fields[static_cast<size_t>(col["G"])], g) ||
            !str2double(fields[static_cast<size_t>(col["B"])], b)) {
            error("%s: malformed color row in %s", __func__, src.string().c_str());
        }
        if (r <= 1.0 && g <= 1.0 && b <= 1.0) {
            r *= 255.0;
            g *= 255.0;
            b *= 255.0;
        }
        RgbColorEntry entry;
        entry.name = hasName && fields.size() > static_cast<size_t>(col["Name"])
            ? fields[static_cast<size_t>(col["Name"])] : std::to_string(row);
        entry.colorIndex = hasColorIndex && fields.size() > static_cast<size_t>(col["Color_index"])
            ? fields[static_cast<size_t>(col["Color_index"])] : std::to_string(row);
        entry.rgb = {
            static_cast<int32_t>(clamp_u8(static_cast<float>(r))),
            static_cast<int32_t>(clamp_u8(static_cast<float>(g))),
            static_cast<int32_t>(clamp_u8(static_cast<float>(b)))
        };
        sourceColors.push_back(std::move(entry));
        ++row;
    }
    if (sourceColors.empty()) {
        error("%s: color table has no colors: %s", __func__, src.string().c_str());
    }

    std::vector<RgbColorEntry> out;
    out.reserve(k);
    for (size_t i = 0; i < k; ++i) {
        RgbColorEntry entry = sourceColors[i % sourceColors.size()];
        if (i >= sourceColors.size()) {
            entry.name = std::to_string(i);
            entry.colorIndex = std::to_string(i);
        }
        out.push_back(std::move(entry));
    }
    return out;
}

size_t count_rgb_color_rows(const fs::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, path.string().c_str());
    }
    std::string line;
    if (!std::getline(in, line)) {
        return 0;
    }
    size_t rows = 0;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            ++rows;
        }
    }
    return rows;
}

bool existing_rgb_uses_numeric_ids(const fs::path& path, size_t factorCount) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::string line;
    if (!std::getline(in, line)) {
        return false;
    }
    std::vector<std::string> header = split_tab(strip_leading_hash(line));
    const int32_t cName = find_header_column_ci(header, {"Name"});
    if (cName < 0) {
        return false;
    }
    std::vector<bool> seen(factorCount, false);
    size_t rows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields = split_tab(line);
        if (fields.size() <= static_cast<size_t>(cName)) {
            return false;
        }
        int32_t id = -1;
        if (!str2int32(fields[static_cast<size_t>(cName)], id) ||
            id < 0 || static_cast<size_t>(id) >= factorCount ||
            seen[static_cast<size_t>(id)]) {
            return false;
        }
        seen[static_cast<size_t>(id)] = true;
        ++rows;
    }
    return rows == factorCount && std::all_of(seen.begin(), seen.end(),
        [](bool value) { return value; });
}

size_t infer_factor_count_from_pseudobulk(const fs::path& src) {
    std::ifstream in(src);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, src.string().c_str());
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            break;
        }
    }
    if (line.empty()) {
        error("%s: empty pseudobulk table %s", __func__, src.string().c_str());
    }
    if (!line.empty() && line[0] == '#') {
        line.erase(line.begin());
    }
    const std::vector<std::string> header = split_tab(line);
    if (header.size() <= 1u) {
        error("%s: pseudobulk table has no factor columns: %s", __func__, src.string().c_str());
    }
    return header.size() - 1u;
}

std::vector<std::string> factor_names_from_pseudobulk_header(const fs::path& src) {
    std::ifstream in(src);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, src.string().c_str());
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            break;
        }
    }
    if (line.empty()) {
        error("%s: empty pseudobulk table %s", __func__, src.string().c_str());
    }
    if (!line.empty() && line[0] == '#') {
        line.erase(line.begin());
    }
    const std::vector<std::string> header = split_tab(line);
    if (header.size() <= 1u) {
        error("%s: pseudobulk table has no factor columns: %s", __func__, src.string().c_str());
    }
    return std::vector<std::string>(header.begin() + 1, header.end());
}

bool is_nonnegative_integer_name(const std::string& value) {
    if (value.empty()) {
        return false;
    }
    return std::all_of(value.begin(), value.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
}

bool factor_names_are_numeric(const std::vector<std::string>& factorNames) {
    return std::all_of(factorNames.begin(), factorNames.end(), is_nonnegative_integer_name);
}

fs::path normalize_cell_results_for_pmtiles(const CellInputs& cell,
    const std::vector<std::string>& factorNames,
    const fs::path& tmpDir) {
    if (factorNames.empty()) {
        return cell.resultsTsv;
    }
    std::ifstream in(cell.resultsTsv);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, cell.resultsTsv.string().c_str());
    }
    std::string headerLine;
    while (std::getline(in, headerLine)) {
        if (!headerLine.empty()) {
            break;
        }
    }
    if (headerLine.empty()) {
        error("%s: empty cell results table %s", __func__, cell.resultsTsv.string().c_str());
    }
    std::string headerNoHash = headerLine;
    if (!headerNoHash.empty() && headerNoHash[0] == '#') {
        headerNoHash.erase(headerNoHash.begin());
    }
    const std::vector<std::string> header = split_tab(headerNoHash);
    if (header_has(header, "topK") && header_has(header, "topP")) {
        return cell.resultsTsv;
    }
    if (header_has(header, "K1") && header_has(header, "P1")) {
        return cell.resultsTsv;
    }

    std::map<std::string, int32_t> colByName;
    for (size_t i = 0; i < header.size(); ++i) {
        colByName[header[i]] = static_cast<int32_t>(i);
    }
    auto idIt = colByName.find(cell.resultIdCol);
    if (idIt == colByName.end()) {
        error("%s: cell results table must contain id column %s: %s",
            __func__, cell.resultIdCol.c_str(), cell.resultsTsv.string().c_str());
    }

    std::vector<int32_t> factorCols;
    factorCols.reserve(factorNames.size());
    const bool allFactorNamesAreNumeric =
        std::all_of(factorNames.begin(), factorNames.end(), is_nonnegative_integer_name);
    for (const std::string& factorName : factorNames) {
        auto it = colByName.find(factorName);
        if (it == colByName.end()) {
            if (!allFactorNamesAreNumeric) {
                error("%s: cell results table %s is missing model factor column '%s' from %s",
                    __func__, cell.resultsTsv.string().c_str(), factorName.c_str(),
                    "pseudobulk header");
            }
            return cell.resultsTsv;
        }
        factorCols.push_back(it->second);
    }
    if (allFactorNamesAreNumeric) {
        return cell.resultsTsv;
    }

    fs::create_directories(tmpDir);
    fs::path outPath = tmpDir / (cell.id + ".cell_results.numeric_factors.tsv");
    std::ofstream out(outPath);
    if (!out.is_open()) {
        error("%s: cannot open %s", __func__, outPath.string().c_str());
    }
    out << cell.resultIdCol;
    for (size_t i = 0; i < factorNames.size(); ++i) {
        out << "\t" << i;
    }
    out << "\n";

    std::string line;
    uint64_t row = 1;
    const int32_t maxCol = std::max(idIt->second,
        *std::max_element(factorCols.begin(), factorCols.end()));
    while (std::getline(in, line)) {
        ++row;
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_tab(line);
        if (fields.size() <= static_cast<size_t>(maxCol)) {
            error("%s: malformed row %" PRIu64 " in %s",
                __func__, row, cell.resultsTsv.string().c_str());
        }
        out << fields[static_cast<size_t>(idIt->second)];
        for (int32_t col : factorCols) {
            out << "\t" << fields[static_cast<size_t>(col)];
        }
        out << "\n";
    }
    if (!out.good()) {
        error("%s: failed writing %s", __func__, outPath.string().c_str());
    }
    notice("%s: normalized named factor columns in %s to numeric factor ids using %s",
        __func__, cell.resultsTsv.string().c_str(), outPath.string().c_str());
    return outPath;
}

void write_cartoscope_rgb(const fs::path& src, const fs::path& dst, size_t factorCount,
    bool overwrite) {
    if (!overwrite && file_exists(dst)) {
        const size_t existingRows = count_rgb_color_rows(dst);
        if (existingRows == factorCount && existing_rgb_uses_numeric_ids(dst, factorCount)) {
            notice("%s: %s already has %zu RGB rows; skipping RGB sidecar",
                __func__, dst.string().c_str(), existingRows);
            return;
        }
        notice("%s: %s has stale RGB ids or %zu rows but expected %zu; rewriting RGB sidecar",
            __func__, dst.string().c_str(), existingRows, factorCount);
    }
    const std::vector<RgbColorEntry> colors = load_normalized_colors(src, factorCount);
    fs::create_directories(dst.parent_path());
    std::ofstream out(dst);
    if (!out.is_open()) {
        error("%s: cannot open %s", __func__, dst.string().c_str());
    }
    out << "Name\tColor_index\tR\tG\tB\n";
    for (size_t i = 0; i < colors.size(); ++i) {
        const RgbColorEntry& color = colors[i];
        out << i << "\t" << i << "\t"
            << fp_to_string(static_cast<double>(color.rgb[0]) / 255.0, 6) << "\t"
            << fp_to_string(static_cast<double>(color.rgb[1]) / 255.0, 6) << "\t"
            << fp_to_string(static_cast<double>(color.rgb[2]) / 255.0, 6) << "\n";
    }
    if (!out.good()) {
        error("%s: failed writing %s", __func__, dst.string().c_str());
    }
}

bool existing_de_uses_numeric_factors(const fs::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::string line;
    if (!std::getline(in, line)) {
        return false;
    }
    const std::vector<std::string> header = split_tab(strip_leading_hash(line));
    const int32_t cFactor = find_header_column_ci(header, {"factor", "Factor"});
    if (cFactor < 0) {
        return false;
    }
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_tab(line);
        if (static_cast<size_t>(cFactor) >= fields.size()) {
            continue;
        }
        return is_nonnegative_integer_name(fields[static_cast<size_t>(cFactor)]);
    }
    return true;
}

void write_cartoscope_de(const fs::path& src, const fs::path& dst, bool overwrite,
    const std::vector<std::string>& factorNames = {}) {
    const bool namedFactors = !factorNames.empty() && !factor_names_are_numeric(factorNames);
    if (!overwrite && file_exists(dst)) {
        if (!namedFactors || existing_de_uses_numeric_factors(dst)) {
            notice("%s: %s already exists; skipping bulk DE sidecar", __func__, dst.string().c_str());
            return;
        }
        notice("%s: %s has named factors but numeric factor ids are required; rewriting bulk DE sidecar",
            __func__, dst.string().c_str());
    }
    std::map<std::string, std::string> factorIdByName;
    if (namedFactors) {
        for (size_t i = 0; i < factorNames.size(); ++i) {
            factorIdByName[normalize_factor_id(factorNames[i])] = std::to_string(i);
        }
    }
    std::ifstream in(src);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, src.string().c_str());
    }
    std::string headerLine;
    if (!std::getline(in, headerLine)) {
        error("%s: empty DE table %s", __func__, src.string().c_str());
    }
    const std::vector<std::string> header = split_tab(headerLine);
    const int32_t cGene = find_header_column(header, {"gene", "Feature"});
    const int32_t cFactor = find_header_column(header, {"factor", "Factor"});
    const int32_t cChi2 = find_header_column(header, {"Chi2"});
    const int32_t cPval = find_header_column(header, {"pval"});
    const int32_t cFoldChange = find_header_column(header, {"FoldChange"});
    const int32_t cGeneTotal = find_header_column(header, {"gene_total"});
    const int32_t cLog10Pval = find_header_column(header, {"log10pval"});
    if (cGene < 0 || cFactor < 0 || cChi2 < 0 || cFoldChange < 0 || cLog10Pval < 0) {
        error("%s: DE table must contain gene/Feature, factor/Factor, Chi2, FoldChange, and log10pval columns: %s",
            __func__, src.string().c_str());
    }

    fs::create_directories(dst.parent_path());
    std::ofstream out(dst);
    if (!out.is_open()) {
        error("%s: cannot open %s", __func__, dst.string().c_str());
    }
    out << "gene\tfactor\tChi2\tpval\tFoldChange\tgene_total\tlog10pval\n";
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_tab(line);
        auto get = [&](int32_t idx) -> std::string {
            if (idx < 0 || static_cast<size_t>(idx) >= fields.size()) {
                return "";
            }
            return fields[static_cast<size_t>(idx)];
        };
        std::string factor = get(cFactor);
        if (namedFactors) {
            const auto it = factorIdByName.find(normalize_factor_id(factor));
            if (it != factorIdByName.end()) {
                factor = it->second;
            } else {
                int32_t numericFactor = -1;
                if (!str2int32(factor, numericFactor) || numericFactor < 0 ||
                    static_cast<size_t>(numericFactor) >= factorNames.size()) {
                    error("%s: DE factor '%s' in %s is neither a pseudobulk factor name nor a numeric factor id in [0, %zu)",
                        __func__, factor.c_str(), src.string().c_str(), factorNames.size());
                }
            }
        }
        out << get(cGene) << "\t"
            << factor << "\t"
            << get(cChi2) << "\t"
            << get(cPval) << "\t"
            << get(cFoldChange) << "\t"
            << get(cGeneTotal) << "\t"
            << get(cLog10Pval) << "\n";
    }
    if (!out.good()) {
        error("%s: failed writing %s", __func__, dst.string().c_str());
    }
}

std::vector<DeGeneRow> read_cartoscope_de_rows(const fs::path& src) {
    TextLineReader reader(src.string());
    std::string line;
    if (!reader.getline(line)) {
        error("%s: empty DE table %s", __func__, src.string().c_str());
    }
    std::vector<std::string> header = split_tab(strip_leading_hash(line));
    std::map<std::string, int32_t> col;
    for (size_t i = 0; i < header.size(); ++i) {
        col[header[i]] = static_cast<int32_t>(i);
    }
    const int32_t cGene = find_header_column_ci(header, {"gene", "Feature", "feature", "Gene"});
    const int32_t cFactor = find_header_column_ci(header, {"factor", "Factor", "Topic", "topic"});
    const int32_t cFc = find_header_column_ci(header, {"FoldChange", "ApproxFC", "approxFC", "FC", "fc"});
    const int32_t cLog10 = find_header_column_ci(header, {"log10pval", "logPval", "LogPval", "-log10pval", "neglog10pval", "neglog10p"});
    if (cGene < 0 || cFactor < 0) {
        error("%s: DE table must contain gene and factor columns: %s",
            __func__, src.string().c_str());
    }
    std::vector<DeGeneRow> rows;
    while (reader.getline(line)) {
        if (line.empty()) {
            continue;
        }
        std::vector<std::string> fields = split_tab(line);
        const int32_t maxCol = std::max({cGene, cFactor, cFc, cLog10});
        if (fields.size() <= static_cast<size_t>(maxCol)) {
            continue;
        }
        DeGeneRow row;
        row.gene = fields[static_cast<size_t>(cGene)];
        row.factor = normalize_factor_id(fields[static_cast<size_t>(cFactor)]);
        if (cFc >= 0) {
            str2double(fields[static_cast<size_t>(cFc)], row.foldChange);
        }
        if (cLog10 >= 0) {
            str2double(fields[static_cast<size_t>(cLog10)], row.log10pval);
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

std::string join_top_genes(std::vector<std::pair<double, std::string>> ranked, size_t nTop) {
    if (ranked.empty()) {
        return ".";
    }
    std::stable_sort(ranked.begin(), ranked.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
    std::ostringstream out;
    const size_t n = std::min(nTop, ranked.size());
    for (size_t i = 0; i < n; ++i) {
        if (i > 0) {
            out << ",";
        }
        out << ranked[i].second;
    }
    return out.str();
}

std::string rgb_string(const RgbColorEntry& color) {
    return std::to_string(color.rgb[0]) + "," + std::to_string(color.rgb[1]) + "," +
        std::to_string(color.rgb[2]);
}

bool existing_info_is_compatible(const fs::path& path, size_t factorCount) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::string line;
    if (!std::getline(in, line)) {
        return false;
    }
    std::vector<std::string> header = split_tab(strip_leading_hash(line));
    const bool hasRequired = find_header_column_ci(header, {"Factor"}) >= 0 &&
        find_header_column_ci(header, {"RGB"}) >= 0 &&
        find_header_column_ci(header, {"Weight"}) >= 0 &&
        find_header_column_ci(header, {"PostUMI"}) >= 0;
    if (!hasRequired) {
        return false;
    }
    size_t rows = 0;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            ++rows;
        }
    }
    return rows == factorCount;
}

void write_cartoscope_info_from_pseudobulk(const fs::path& src, const fs::path& deSrc,
    const fs::path& rgbSrc, const fs::path& dst, bool overwrite) {
    if (!overwrite && file_exists(dst)) {
        const std::vector<std::string> factorNames = factor_names_from_pseudobulk_header(src);
        if (existing_info_is_compatible(dst, factorNames.size())) {
            notice("%s: %s already has CartoScope factor info; skipping info sidecar",
                __func__, dst.string().c_str());
            return;
        }
        notice("%s: %s is stale or incomplete; rewriting info sidecar",
            __func__, dst.string().c_str());
    }
    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pseudobulk;
    std::vector<std::string> featureNames;
    std::vector<std::string> factorNames;
    try {
        read_matrix_from_file<double>(src.string(), pseudobulk, &featureNames, &factorNames);
    } catch (const std::exception& e) {
        error("%s: failed reading pseudobulk table %s: %s", __func__, src.string().c_str(), e.what());
    }
    if (pseudobulk.cols() == 0) {
        error("%s: pseudobulk table has no factor columns: %s", __func__, src.string().c_str());
    }
    const size_t factorCount = static_cast<size_t>(pseudobulk.cols());
    const std::vector<RgbColorEntry> colors = load_normalized_colors(rgbSrc, factorCount);
    const std::vector<DeGeneRow> deRows = read_cartoscope_de_rows(deSrc);

    fs::create_directories(dst.parent_path());
    std::ofstream out(dst);
    if (!out.is_open()) {
        error("%s: cannot open %s", __func__, dst.string().c_str());
    }
    const bool namedFactors = !factor_names_are_numeric(factorNames);
    out << "Factor\tRGB\tWeight\tPostUMI\tTopGene_pval\tTopGene_fc\tTopGene_weight\n";
    const Eigen::Matrix<double, 1, Eigen::Dynamic> weights = pseudobulk.colwise().sum();
    const double totalWeight = weights.sum();
    if (!(totalWeight > 0.0)) {
        error("%s: pseudobulk table has non-positive total factor weight: %s",
            __func__, src.string().c_str());
    }
    struct InfoRow {
        size_t idx = 0;
        double weight = 0.0;
    };
    std::vector<InfoRow> order;
    order.reserve(factorCount);
    for (size_t j = 0; j < factorCount; ++j) {
        order.push_back({j, weights(static_cast<Eigen::Index>(j)) / totalWeight});
    }
    std::stable_sort(order.begin(), order.end(),
        [](const InfoRow& a, const InfoRow& b) {
            return a.weight > b.weight;
        });

    const size_t nTop = 20;
    for (const InfoRow& row : order) {
        const size_t j = row.idx;
        const std::string factorForDe = namedFactors
            ? std::to_string(j)
            : normalize_factor_id(factorNames[j]);
        const std::string factorId = namedFactors ? std::to_string(j) : factorForDe;
        std::vector<std::pair<double, std::string>> byPval;
        std::vector<std::pair<double, std::string>> byFc;
        for (const DeGeneRow& de : deRows) {
            if (de.factor != factorForDe) {
                continue;
            }
            byPval.emplace_back(de.log10pval, de.gene);
            byFc.emplace_back(de.foldChange, de.gene);
        }
        std::vector<std::pair<double, std::string>> byWeight;
        byWeight.reserve(featureNames.size());
        for (size_t i = 0; i < featureNames.size(); ++i) {
            byWeight.emplace_back(pseudobulk(static_cast<Eigen::Index>(i),
                static_cast<Eigen::Index>(j)), featureNames[i]);
        }
        out << factorId << "\t"
            << rgb_string(colors[j]) << "\t"
            << fp_to_string(row.weight, 5) << "\t"
            << static_cast<int64_t>(std::llround(weights(static_cast<Eigen::Index>(j)))) << "\t"
            << join_top_genes(std::move(byPval), nTop) << "\t"
            << join_top_genes(std::move(byFc), nTop) << "\t"
            << join_top_genes(std::move(byWeight), nTop) << "\n";
    }
    if (!out.good()) {
        error("%s: failed writing %s", __func__, dst.string().c_str());
    }
}

bool existing_alias_matches_pseudobulk(const fs::path& path,
    const std::vector<std::string>& factorNames) {
    std::ifstream in(path);
    if (!in.is_open()) {
        return false;
    }
    std::string line;
    if (!std::getline(in, line)) {
        return false;
    }
    const std::vector<std::string> header = split_tab(strip_leading_hash(line));
    const int32_t cIndex = find_header_column_ci(header, {"index"});
    const int32_t cAlias = find_header_column_ci(header, {"alias"});
    if (cIndex < 0 || cAlias < 0) {
        return false;
    }
    std::vector<std::string> seen(factorNames.size());
    size_t rows = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::vector<std::string> fields = split_tab(line);
        if (fields.size() <= static_cast<size_t>(std::max(cIndex, cAlias))) {
            return false;
        }
        int32_t idx = -1;
        if (!str2int32(fields[static_cast<size_t>(cIndex)], idx) ||
            idx < 0 || static_cast<size_t>(idx) >= factorNames.size()) {
            return false;
        }
        if (!seen[static_cast<size_t>(idx)].empty()) {
            return false;
        }
        seen[static_cast<size_t>(idx)] = fields[static_cast<size_t>(cAlias)];
        ++rows;
    }
    if (rows != factorNames.size()) {
        return false;
    }
    for (size_t i = 0; i < factorNames.size(); ++i) {
        if (seen[i] != factorNames[i]) {
            return false;
        }
    }
    return true;
}

bool write_factor_alias_from_pseudobulk(const fs::path& src, const fs::path& dst,
    bool overwrite) {
    const std::vector<std::string> factorNames = factor_names_from_pseudobulk_header(src);
    if (factor_names_are_numeric(factorNames)) {
        return false;
    }
    if (!overwrite && file_exists(dst)) {
        if (existing_alias_matches_pseudobulk(dst, factorNames)) {
            notice("%s: %s already matches pseudobulk factor aliases; skipping factor alias sidecar",
                __func__, dst.string().c_str());
            return true;
        }
        notice("%s: %s is stale or incomplete; rewriting factor alias sidecar",
            __func__, dst.string().c_str());
    }
    fs::create_directories(dst.parent_path());
    std::ofstream out(dst);
    if (!out.is_open()) {
        error("%s: cannot open %s", __func__, dst.string().c_str());
    }
    out << "index\talias\n";
    for (size_t i = 0; i < factorNames.size(); ++i) {
        out << i << "\t" << factorNames[i] << "\n";
    }
    if (!out.good()) {
        error("%s: failed writing %s", __func__, dst.string().c_str());
    }
    return true;
}

void gzip_copy_text(const fs::path& src, const fs::path& dst, bool overwrite) {
    if (!overwrite && file_exists(dst)) {
        notice("%s: %s already exists; skipping gzip sidecar", __func__, dst.string().c_str());
        return;
    }
    const std::string raw = read_all_text(src);
    write_binary(dst, pm_core::gzip_compress(raw));
}

bool bounds_from_index_header(const fs::path& indexPath, pm_raster::RasterBounds& bounds) {
    if (!file_exists(indexPath)) {
        return false;
    }
    const LoadedTileIndexData loaded = loadTileIndexData(indexPath.string());
    const IndexHeader& h = loaded.header;
    if (h.xmax > h.xmin && h.ymax > h.ymin) {
        bounds.xmin = h.xmin;
        bounds.xmax = h.xmax;
        bounds.ymin = h.ymin;
        bounds.ymax = h.ymax;
        return true;
    }
    if (loaded.globalBox.proper()) {
        bounds.xmin = loaded.globalBox.xmin;
        bounds.xmax = loaded.globalBox.xmax;
        bounds.ymin = loaded.globalBox.ymin;
        bounds.ymax = loaded.globalBox.ymax;
        return true;
    }
    return false;
}

pm_raster::RasterBounds resolve_raster_bounds(const TranscriptInputs& transcripts,
    const ModelInputs& firstModel) {
    pm_raster::RasterBounds bounds;
    const fs::path coordRangePath = transcripts.tiledPrefix.string() + ".coord_range.tsv";
    if (file_exists(coordRangePath)) {
        readCoordRange(coordRangePath.string(), bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax);
        if (bounds.valid()) {
            return bounds;
        }
    }
    if (bounds_from_index_header(transcripts.tiledPrefix.string() + ".index", bounds)) {
        return bounds;
    }
    if (bounds_from_index_header(firstModel.pixelPrefix.string() + ".index", bounds)) {
        return bounds;
    }
    error("%s: cannot determine raster bounds from %s or index headers",
        __func__, coordRangePath.string().c_str());
    return bounds;
}

struct RasterTileAccum {
    Image2D<Color3f> sum;
    Image2D<uint8_t> count;

    RasterTileAccum() = default;
    RasterTileAccum(int32_t tileSize)
        : sum(tileSize, tileSize, Color3f{0.0f, 0.0f, 0.0f}),
          count(tileSize, tileSize, 0) {}
};

void write_raw_pixel_raster_pmtiles_archive(const ModelInputs& model,
    const std::string& outFile,
    const std::string& tempBlobFile,
    const pm_raster::RasterBounds& bounds,
    int32_t minZoom,
    int32_t maxZoom) {
    constexpr int32_t tileSize = 256;
    constexpr float minProb = 1e-3f;
    pm_raster::validate_raster_archive_options(bounds, minZoom, maxZoom, __func__);

    const size_t factorCount = infer_factor_count_from_pseudobulk(model.pseudobulkTsv);
    const std::vector<RgbColorEntry> colors =
        load_normalized_colors(model.colorRgbTsv, factorCount);

    std::filesystem::path blobPath(tempBlobFile);
    if (blobPath.has_parent_path()) {
        std::filesystem::create_directories(blobPath.parent_path());
    }
    std::ofstream blob(blobPath, std::ios::binary | std::ios::trunc);
    if (!blob.is_open()) {
        error("%s: cannot open temporary PMTiles blob %s", __func__, tempBlobFile.c_str());
    }

    std::vector<pm_core::StoredTilePayloadRef> tiles;
    uint64_t dataOffset = 0;
    for (int32_t z = minZoom; z <= maxZoom; ++z) {
        std::map<pm_raster::RasterTileKey, RasterTileAccum> accumByTile;
        TileOperator reader(model.pixelPrefix.string() + ".bin",
            model.pixelPrefix.string() + ".index", "");
        reader.openDataStream();
        PixTopProbs<float> rec;
        int32_t ret = 0;
        while ((ret = reader.next(rec)) >= 0) {
            if (ret == 0) {
                error("%s: invalid or corrupted pixel decode input for model %s",
                    __func__, model.id.c_str());
            }
            if (rec.x < bounds.xmin || rec.x > bounds.xmax ||
                rec.y < bounds.ymin || rec.y > bounds.ymax) {
                continue;
            }
            double r = 0.0, g = 0.0, b = 0.0, psum = 0.0;
            const size_t n = std::min(rec.ks.size(), rec.ps.size());
            for (size_t i = 0; i < n; ++i) {
                const int32_t ch = rec.ks[i];
                const float p = rec.ps[i];
                if (ch < 0 || p < minProb) {
                    continue;
                }
                const RgbColorEntry& color = colors[static_cast<size_t>(ch) % colors.size()];
                r += static_cast<double>(color.rgb[0]) * p;
                g += static_cast<double>(color.rgb[1]) * p;
                b += static_cast<double>(color.rgb[2]) * p;
                psum += p;
            }
            if (psum < 1e-3) {
                continue;
            }
            r /= psum;
            g /= psum;
            b /= psum;

            const pm_raster::RasterPixelCoord pix =
                pm_raster::epsg3857_to_raster_pixel(rec.x, rec.y, z);
            auto it = accumByTile.find(pix.key);
            if (it == accumByTile.end()) {
                it = accumByTile.emplace(pix.key, RasterTileAccum(tileSize)).first;
            }
            if (it->second.count(pix.py, pix.px) == 255) {
                continue;
            }
            it->second.sum(pix.py, pix.px) += Color3f{
                static_cast<float>(r), static_cast<float>(g), static_cast<float>(b)};
            it->second.count(pix.py, pix.px) += 1;
        }

        for (auto& kv : accumByTile) {
            Image2D<Rgb8> tile(tileSize, tileSize, Rgb8{0, 0, 0});
            for (int32_t y = 0; y < tileSize; ++y) {
                for (int32_t x = 0; x < tileSize; ++x) {
                    const uint8_t count = kv.second.count(y, x);
                    if (count == 0) {
                        continue;
                    }
                    const Color3f avg = kv.second.sum(y, x) / static_cast<float>(count);
                    tile(y, x) = Rgb8{
                        clamp_u8(avg.r), clamp_u8(avg.g), clamp_u8(avg.b)};
                }
            }
            const std::string encoded = encode_png_rgb8(tile);
            pm_raster::append_png_tile_to_blob(
                blob, tempBlobFile, encoded, dataOffset, kv.first, tiles);
        }
        notice("%s: z%d wrote %zu raster tile(s) for model %s",
            __func__, z, accumByTile.size(), model.id.c_str());
    }
    blob.close();
    pm_raster::write_png_raster_pmtiles_archive_from_blob(
        outFile, tempBlobFile, std::move(tiles), bounds, minZoom, maxZoom);
    std::error_code ec;
    fs::remove(blobPath, ec);
}

json read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, path.string().c_str());
    }
    return json::parse(in);
}

std::array<double, 9> parse_transform_json(const json& value, const char* context) {
    std::array<double, 9> out{};
    if (value.is_array() && value.size() == 3 &&
        value[0].is_array() && value[1].is_array() && value[2].is_array()) {
        for (size_t r = 0; r < 3; ++r) {
            if (value[r].size() != 3) {
                error("%s: transform rows must each contain 3 values in %s", __func__, context);
            }
            for (size_t c = 0; c < 3; ++c) {
                out[r * 3 + c] = value[r][c].get<double>();
            }
        }
        return out;
    }
    if (value.is_array() && value.size() == 9) {
        for (size_t i = 0; i < 9; ++i) {
            out[i] = value[i].get<double>();
        }
        return out;
    }
    error("%s: transform must be a 3x3 array or 9-value array in %s", __func__, context);
    return out;
}

std::array<double, 2> parse_point2_json(const json& value, const char* name, const char* context) {
    if (!value.is_array() || value.size() != 2) {
        error("%s: %s must be a 2-value array in %s", __func__, name, context);
    }
    return {value[0].get<double>(), value[1].get<double>()};
}

void parse_image_entries(const json& root, const fs::path& base, DeployInputs& out,
    const fs::path& contextPath) {
    if (root.contains("image_assets") && !root.at("image_assets").is_null()) {
        const json& assets = root.at("image_assets");
        if (!assets.is_array()) {
            error("%s: image_assets must be an array in %s", __func__, contextPath.string().c_str());
        }
        for (const auto& entry : assets) {
            out.imageAssetJsons.push_back(resolve_path(base, entry.get<std::string>()));
        }
    }
    if (!root.contains("images") || root.at("images").is_null()) {
        return;
    }
    const json& images = root.at("images");
    if (!images.is_array()) {
        error("%s: images must be an array in %s", __func__, contextPath.string().c_str());
    }
    for (const auto& image : images) {
        image_pmtiles::Options opts;
        opts.id = normalize_id(image.at("id").get<std::string>());
        opts.inImage = resolve_path(base, image.at("src").get<std::string>());
        opts.minZoom = image.value("min_zoom", opts.minZoom);
        opts.maxZoom = image.value("max_zoom", opts.maxZoom);
        opts.grayLowPercentile = image.value("gray_low_percentile", opts.grayLowPercentile);
        opts.grayHighPercentile = image.value("gray_high_percentile", opts.grayHighPercentile);
        opts.graySampleFraction = image.value("gray_sample_fraction", opts.graySampleFraction);
        opts.graySampleTiles = image.value("gray_sample_tiles", opts.graySampleTiles);
        opts.graySampleSeed = image.value("gray_sample_seed", opts.graySampleSeed);
        opts.tileCacheMb = image.value("tile_cache_mb", opts.tileCacheMb);
        opts.tiffSourceLevel = image.value("tiff_source_level", opts.tiffSourceLevel);
        const bool hasTransformField = image.contains("transform") && !image.at("transform").is_null();
        const bool hasPixZeroField = image.contains("pix_zero") && !image.at("pix_zero").is_null();
        const bool hasPixMaxField = image.contains("pix_max") && !image.at("pix_max").is_null();
        const bool hasMppField = image.contains("microns_per_pixel") && !image.at("microns_per_pixel").is_null();
        const bool hasOffsetXField = image.contains("offset_x_um") && !image.at("offset_x_um").is_null();
        const bool hasOffsetYField = image.contains("offset_y_um") && !image.at("offset_y_um").is_null();
        if (hasTransformField) {
            opts.hasTransform = true;
            opts.transform = parse_transform_json(image.at("transform"), contextPath.string().c_str());
        }
        if (hasPixZeroField) {
            opts.hasPixZero = true;
            opts.pixZero = parse_point2_json(image.at("pix_zero"), "pix_zero",
                contextPath.string().c_str());
        }
        if (hasPixMaxField) {
            opts.hasPixMax = true;
            opts.pixMax = parse_point2_json(image.at("pix_max"), "pix_max",
                contextPath.string().c_str());
        }
        if (hasMppField) {
            opts.hasMicronsPerPixel = true;
            opts.micronsPerPixel = image.at("microns_per_pixel").get<double>();
        }
        if (hasOffsetXField) {
            opts.hasOffsetXUm = true;
            opts.offsetXUm = image.at("offset_x_um").get<double>();
        }
        if (hasOffsetYField) {
            opts.hasOffsetYUm = true;
            opts.offsetYUm = image.at("offset_y_um").get<double>();
        }
        if (!hasTransformField && !hasPixZeroField && !hasPixMaxField && !hasMppField) {
            error("%s: image %s requires transform, microns_per_pixel, or pix_zero plus pix_max",
                __func__, opts.id.c_str());
        }
        out.images.push_back(std::move(opts));
    }
}

fs::path resolve_image_json_fragment_path(const fs::path& imageJsonPath,
    const fs::path& inputJsonBase, const std::string& value) {
    fs::path p(value);
    if (p.is_absolute()) {
        return p;
    }
    const fs::path sidecarBase = imageJsonPath.parent_path();
    std::vector<fs::path> candidates;
    if (!sidecarBase.empty()) {
        candidates.push_back(sidecarBase / p);
    }
    if (!inputJsonBase.empty()) {
        candidates.push_back(inputJsonBase / p);
    }
    candidates.push_back(fs::current_path() / p);
    for (const fs::path& candidate : candidates) {
        if (file_exists(candidate)) {
            return candidate;
        }
    }
    return sidecarBase.empty() ? p : sidecarBase / p;
}

json normalize_image_json_fragment_paths(json fragment, const fs::path& imageJsonPath,
    const fs::path& inputJsonBase) {
    if (fragment.contains("image_assets") && !fragment.at("image_assets").is_null()) {
        json& assets = fragment.at("image_assets");
        if (!assets.is_array()) {
            error("%s: image_assets must be an array in %s",
                __func__, imageJsonPath.string().c_str());
        }
        for (auto& entry : assets) {
            if (!entry.is_string()) {
                error("%s: image_assets entries must be path strings in %s",
                    __func__, imageJsonPath.string().c_str());
            }
            entry = resolve_image_json_fragment_path(imageJsonPath, inputJsonBase,
                entry.get<std::string>()).string();
        }
    }
    if (fragment.contains("images") && !fragment.at("images").is_null()) {
        json& images = fragment.at("images");
        if (!images.is_array()) {
            error("%s: images must be an array in %s",
                __func__, imageJsonPath.string().c_str());
        }
        for (auto& image : images) {
            if (!image.is_object() || !image.contains("src") || !image.at("src").is_string()) {
                error("%s: each image entry must contain src path string in %s",
                    __func__, imageJsonPath.string().c_str());
            }
            image["src"] = resolve_image_json_fragment_path(imageJsonPath, inputJsonBase,
                image.at("src").get<std::string>()).string();
        }
    }
    return fragment;
}

void append_image_json_fragments(DeployInputs& out, const std::vector<std::string>& imageJsons,
    const fs::path& inputJsonBase) {
    for (const std::string& pathStr : imageJsons) {
        fs::path imageJsonPath(pathStr);
        if (!imageJsonPath.is_absolute() && !file_exists(imageJsonPath) && !inputJsonBase.empty()) {
            imageJsonPath = inputJsonBase / imageJsonPath;
        }
        require_file(imageJsonPath, "image JSON");
        json root = read_json_file(imageJsonPath);
        if (!root.is_object()) {
            error("%s: image JSON must be an object: %s",
                __func__, imageJsonPath.string().c_str());
        }
        json fragment;
        if (root.contains("deploy_cartoscope") && !root.at("deploy_cartoscope").is_null()) {
            fragment = root.at("deploy_cartoscope");
        } else if (root.contains("images") || root.contains("image_assets")) {
            fragment = root;
        } else {
            error("%s: image JSON must contain deploy_cartoscope.images, images, or image_assets: %s",
                __func__, imageJsonPath.string().c_str());
        }
        if (!fragment.is_object()) {
            error("%s: deploy_cartoscope fragment must be an object in %s",
                __func__, imageJsonPath.string().c_str());
        }
        fragment = normalize_image_json_fragment_paths(std::move(fragment), imageJsonPath, inputJsonBase);
        parse_image_entries(fragment, fs::path(), out, imageJsonPath);
    }
}

DeployInputs load_from_config(const fs::path& configPath) {
    json root = read_json_file(configPath);
    const json& wf = root.at("workflow");
    if (!wf.contains("datadir")) {
        error("%s: workflow config must contain workflow.datadir", __func__);
    }
    const fs::path inDir = wf.at("datadir").get<std::string>();
    const PixelDecodeMode pixelMode =
        parse_pixel_mode(wf.value("pixel_decode_mode", std::string("pixel")), configPath.string().c_str());
    const std::string pixelSuffix = pixel_mode_suffix(pixelMode);
    DeployInputs out;
    out.transcripts.tiledPrefix = inDir / "transcripts.tiled";
    out.transcripts.featureCountTsv = inDir / "transcripts.tiled.features.tsv";
    out.transcripts.icolX = wf.value("icol_x", 0);
    out.transcripts.icolY = wf.value("icol_y", 1);
    out.transcripts.icolFeature = wf.value("icol_feature", 2);
    out.transcripts.icolCount = wf.value("icol_count", 3);
    out.transcripts.nullStr = wf.value("null_str", std::string());

    for (const auto& h : wf.at("hexgrids")) {
        for (const auto& k : wf.at("topics")) {
            const int32_t hex = h.get<int32_t>();
            const int32_t topics = k.get<int32_t>();
            ModelInputs model;
            model.sourcePrefix = "h" + std::to_string(hex) + ".k" + std::to_string(topics);
            model.id = default_model_id(hex, topics, pixelMode);
            model.pixelMode = pixelMode;
            model.hexGridDist = static_cast<double>(hex);
            fs::path pref = inDir / model.sourcePrefix;
            fs::path pixelPref = inDir / (model.sourcePrefix + "." + pixelSuffix);
            model.resultsTsv = pref.string() + ".results.tsv";
            model.modelTsv = pref.string() + ".model.tsv";
            model.colorRgbTsv = pref.string() + ".color.rgb.tsv";
            model.pixelPrefix = pixelPref;
            model.pixelPng = pixelPref.string() + ".png";
            model.pseudobulkTsv = pixelPref.string() + ".pseudobulk.tsv";
            model.deTsv = pixelPref.string() + ".de_bulk.tsv";
            out.models.push_back(std::move(model));
        }
    }
    return out;
}

DeployInputs load_from_input_json(const fs::path& inputJson,
    const std::vector<std::string>& imageJsons = {}) {
    json root = read_json_file(inputJson);
    fs::path base = inputJson.parent_path();
    DeployInputs out;
    parse_image_entries(root, base, out, inputJson);
    append_image_json_fragments(out, imageJsons, base);
    const json& tr = root.at("transcripts");
    out.transcripts.tiledPrefix = resolve_path(base, tr.at("tiled_prefix").get<std::string>());
    out.transcripts.featureCountTsv = resolve_path(base, tr.at("feature_count_tsv").get<std::string>());
    out.transcripts.icolX = tr.value("icol_x", 0);
    out.transcripts.icolY = tr.value("icol_y", 1);
    out.transcripts.icolFeature = tr.value("icol_feature", 2);
    out.transcripts.icolCount = tr.value("icol_count", 3);
    out.transcripts.nullStr = tr.value("null_str", std::string());

    auto parse_cell_fields = [&](const json& m, CellInputs& cell, bool prefixedOnly, bool sidecarsAllowed) {
        auto path_key = [&](const char* prefixed, const char* plain, fs::path& dst) {
            if (m.contains(prefixed) && !m.at(prefixed).is_null()) {
                dst = resolve_path(base, m.at(prefixed).get<std::string>());
            } else if (!prefixedOnly && plain != nullptr &&
                m.contains(plain) && !m.at(plain).is_null()) {
                dst = resolve_path(base, m.at(plain).get<std::string>());
            }
        };
        path_key("cell_results_tsv", "results_tsv", cell.resultsTsv);
        path_key("cell_boundaries", "boundaries", cell.boundaries);
        if (cell.boundaries.empty()) {
            path_key("cell_boundaries_json", "boundaries_json", cell.boundaries);
        }
        path_key("cell_centers_tsv", "centers_tsv", cell.centersTsv);

        if (m.contains("cell_pmtiles") && m.at("cell_pmtiles").is_object()) {
            const json& pm = m.at("cell_pmtiles");
            if (pm.contains("cells") && !pm.at("cells").is_null()) {
                cell.cellsPmtiles = resolve_path(base, pm.at("cells").get<std::string>());
            }
            if (pm.contains("boundaries") && !pm.at("boundaries").is_null()) {
                cell.boundariesPmtiles = resolve_path(base, pm.at("boundaries").get<std::string>());
            }
        }
        if (!prefixedOnly && m.contains("pmtiles") && m.at("pmtiles").is_object()) {
            const json& pm = m.at("pmtiles");
            if (pm.contains("cells") && !pm.at("cells").is_null()) {
                cell.cellsPmtiles = resolve_path(base, pm.at("cells").get<std::string>());
            }
            if (pm.contains("boundaries") && !pm.at("boundaries").is_null()) {
                cell.boundariesPmtiles = resolve_path(base, pm.at("boundaries").get<std::string>());
            }
        }
        path_key("cell_pmtiles_cells", nullptr, cell.cellsPmtiles);
        path_key("cell_pmtiles_boundaries", nullptr, cell.boundariesPmtiles);

        cell.boundaryFormat = m.value("cell_boundary_format",
            prefixedOnly ? cell.boundaryFormat : m.value("boundary_format", cell.boundaryFormat));
        cell.boundaryIdProp = m.value("cell_boundary_id_prop",
            prefixedOnly ? cell.boundaryIdProp : m.value("boundary_id_prop", cell.boundaryIdProp));
        cell.resultIdCol = m.value("cell_result_id_col",
            prefixedOnly ? cell.resultIdCol : m.value("result_id_col", m.value("id_col", cell.resultIdCol)));
        cell.bIcolId = m.value("cell_b_icol_id",
            prefixedOnly ? cell.bIcolId : m.value("b_icol_id", cell.bIcolId));
        cell.bIcolX = m.value("cell_b_icol_x",
            prefixedOnly ? cell.bIcolX : m.value("b_icol_x", cell.bIcolX));
        cell.bIcolY = m.value("cell_b_icol_y",
            prefixedOnly ? cell.bIcolY : m.value("b_icol_y", cell.bIcolY));
        cell.cIcolId = m.value("cell_c_icol_id",
            prefixedOnly ? cell.cIcolId : m.value("c_icol_id", cell.cIcolId));
        cell.cIcolX = m.value("cell_c_icol_x",
            prefixedOnly ? cell.cIcolX : m.value("c_icol_x", cell.cIcolX));
        cell.cIcolY = m.value("cell_c_icol_y",
            prefixedOnly ? cell.cIcolY : m.value("c_icol_y", cell.cIcolY));

        if ((m.contains("cell_info_tsv") && !m.at("cell_info_tsv").is_null()) ||
            (!prefixedOnly && ((m.contains("info_tsv") && !m.at("info_tsv").is_null()) ||
                (m.contains("info") && !m.at("info").is_null())))) {
            error("%s: model %s no longer accepts cell info/info_tsv; "
                "deploy-cartoscope derives info from pseudobulk_tsv for cell_only models",
                __func__, cell.id.c_str());
        }
        if ((m.contains("cell_n_factors") && !m.at("cell_n_factors").is_null()) ||
            (!prefixedOnly && m.contains("n_factors") && !m.at("n_factors").is_null())) {
            error("%s: model %s no longer accepts n_factors; factor count is inferred from pseudobulk_tsv",
                __func__, cell.id.c_str());
        }
        if (!sidecarsAllowed) {
            if ((m.contains("cell_de_tsv") && !m.at("cell_de_tsv").is_null()) ||
                (m.contains("cell_pseudobulk_tsv") && !m.at("cell_pseudobulk_tsv").is_null()) ||
                (m.contains("cell_rgb_tsv") && !m.at("cell_rgb_tsv").is_null())) {
                error("%s: default model %s must use the model sidecars for attached cells; "
                    "cell_de_tsv/cell_pseudobulk_tsv/cell_rgb_tsv are not accepted",
                    __func__, cell.id.c_str());
            }
            return;
        }
        path_key("cell_de_tsv", "de_tsv", cell.deTsv);
        if (cell.deTsv.empty() && !prefixedOnly && m.contains("de") && !m.at("de").is_null()) {
            cell.deTsv = resolve_path(base, m.at("de").get<std::string>());
        }
        path_key("cell_pseudobulk_tsv", "pseudobulk_tsv", cell.pseudobulkTsv);
        if (cell.pseudobulkTsv.empty() && !prefixedOnly && m.contains("post") && !m.at("post").is_null()) {
            cell.pseudobulkTsv = resolve_path(base, m.at("post").get<std::string>());
        }
        path_key("cell_rgb_tsv", "rgb_tsv", cell.rgbTsv);
        if (cell.rgbTsv.empty() && !prefixedOnly && m.contains("rgb") && !m.at("rgb").is_null()) {
            cell.rgbTsv = resolve_path(base, m.at("rgb").get<std::string>());
        }
        if (cell.rgbTsv.empty() && !prefixedOnly &&
            m.contains("color_rgb_tsv") && !m.at("color_rgb_tsv").is_null()) {
            cell.rgbTsv = resolve_path(base, m.at("color_rgb_tsv").get<std::string>());
        }
    };

    for (const auto& m : root.at("models")) {
        ModelInputs model;
        model.id = normalize_id(m.at("id").get<std::string>());
        model.type = parse_model_type(m.value("type", std::string("default")), inputJson.string().c_str());
        model.sourcePrefix = model.id;
        model.cell.id = model.id;
        if (is_cell_only_model(model)) {
            parse_cell_fields(m, model.cell, false, true);
        } else {
            model.skipHexPmtiles = m.value("skip_hex_pmtiles", m.value("skip_hex", false));
            if (!model.skipHexPmtiles) {
                model.hexGridDist = m.at("hex_grid_dist").get<double>();
                model.resultsTsv = resolve_path(base, m.at("results_tsv").get<std::string>());
            } else {
                if (m.contains("hex_grid_dist") && !m.at("hex_grid_dist").is_null()) {
                    model.hexGridDist = m.at("hex_grid_dist").get<double>();
                }
                if (m.contains("results_tsv") && !m.at("results_tsv").is_null()) {
                    model.resultsTsv = resolve_path(base, m.at("results_tsv").get<std::string>());
                }
            }
            model.modelTsv = resolve_path(base, m.at("model_tsv").get<std::string>());
            model.colorRgbTsv = resolve_path(base, m.at("color_rgb_tsv").get<std::string>());
            model.pixelPrefix = resolve_path(base, m.at("pixel_prefix").get<std::string>());
            const PixelDecodeMode inferredMode = infer_pixel_mode_from_index(model.pixelPrefix);
            if (m.contains("pixel_decode_mode") && !m.at("pixel_decode_mode").is_null()) {
                model.pixelMode = parse_pixel_mode(m.at("pixel_decode_mode").get<std::string>(), inputJson.string().c_str());
                if (model.pixelMode != inferredMode) {
                    error("%s: pixel_decode_mode '%s' does not match %s.index inferred mode '%s'",
                        __func__, pixel_mode_name(model.pixelMode).c_str(),
                        model.pixelPrefix.string().c_str(), pixel_mode_name(inferredMode).c_str());
                }
            } else {
                model.pixelMode = inferredMode;
            }
            if (m.contains("pixel_png") && !m.at("pixel_png").is_null()) {
                model.pixelPng = resolve_path(base, m.at("pixel_png").get<std::string>());
                model.pixelPngExplicit = true;
            }
            if (m.contains("pseudobulk_tsv") && !m.at("pseudobulk_tsv").is_null()) {
                model.pseudobulkTsv = resolve_path(base, m.at("pseudobulk_tsv").get<std::string>());
            } else {
                model.pseudobulkTsv = model.pixelPrefix.string() + ".pseudobulk.tsv";
            }
            model.factorColBegin = m.value("factor_col_begin", model.factorColBegin);
            model.factorColEnd = m.value("factor_col_end", model.factorColEnd);
            model.deTsv = resolve_path(base, m.at("de_tsv").get<std::string>());
            parse_cell_fields(m, model.cell, true, false);
        }
        out.models.push_back(std::move(model));
    }
    return out;
}

void validate_inputs(const DeployInputs& inputs, bool usePngFlag, bool configMode) {
    require_file(inputs.transcripts.tiledPrefix.string() + ".tsv", "tiled transcript TSV");
    require_file(inputs.transcripts.tiledPrefix.string() + ".index", "tiled transcript index");
    require_file(inputs.transcripts.featureCountTsv, "feature-count TSV");
    if (inputs.models.empty()) {
        error("%s: no models were discovered or provided", __func__);
    }
    bool hasDefaultModel = false;
    std::set<std::string> ids;
    for (const ModelInputs& model : inputs.models) {
        if (!ids.insert(model.id).second) {
            error("%s: duplicate model id %s", __func__, model.id.c_str());
        }
        const CellInputs& cell = model.cell;
        const bool hasCell = has_cell_pmtiles_or_sources(cell);
        if (is_cell_only_model(model)) {
            if (cell.id.empty()) {
                error("%s: cell_only model id must not be empty", __func__);
            }
            const bool hasPrebuilt = !cell.cellsPmtiles.empty() || !cell.boundariesPmtiles.empty();
            const bool hasSources = !cell.resultsTsv.empty() || !cell.boundaries.empty();
            if (hasPrebuilt && hasSources) {
                error("%s: cell_only model %s must use either prebuilt pmtiles or source inputs, not both",
                    __func__, cell.id.c_str());
            }
            if (hasPrebuilt) {
                require_file(cell.cellsPmtiles, "cell PMTiles");
                require_file(cell.boundariesPmtiles, "cell boundary PMTiles");
            } else if (hasSources) {
                require_file(cell.resultsTsv, "cell projection results TSV");
                require_file(cell.boundaries, "cell boundary input");
                if (!cell.centersTsv.empty()) {
                    require_file(cell.centersTsv, "cell centers TSV");
                }
            } else {
                error("%s: cell_only model %s needs pmtiles cells/boundaries or results_tsv/boundaries",
                    __func__, cell.id.c_str());
            }
            if (!cell.deTsv.empty()) {
                require_file(cell.deTsv, "cell DE TSV");
            }
            if (cell.pseudobulkTsv.empty()) {
                error("%s: cell_only model %s requires pseudobulk_tsv for CartoScope packaging",
                    __func__, cell.id.c_str());
            }
            require_file(cell.pseudobulkTsv, "cell pseudobulk TSV");
            if (cell.rgbTsv.empty()) {
                error("%s: cell_only model %s requires rgb_tsv or color_rgb_tsv for CartoScope factor info",
                    __func__, cell.id.c_str());
            }
            require_file(cell.rgbTsv, "cell RGB TSV");
            continue;
        }
        hasDefaultModel = true;
        if (!model.skipHexPmtiles) {
            require_file(model.resultsTsv, "model results TSV");
        }
        require_file(model.modelTsv, "model TSV");
        require_file(model.colorRgbTsv, "color RGB TSV");
        require_file(model.pixelPrefix.string() + ".bin", "pixel decode binary");
        require_file(model.pixelPrefix.string() + ".index", "pixel decode index");
        const PixelDecodeMode inferredMode = infer_pixel_mode_from_index(model.pixelPrefix);
        if (inferredMode != model.pixelMode) {
            error("%s: model %s declares pixel_decode_mode '%s' but %s.index indicates '%s'",
                __func__, model.id.c_str(), pixel_mode_name(model.pixelMode).c_str(),
                model.pixelPrefix.string().c_str(), pixel_mode_name(inferredMode).c_str());
        }
        if (use_png_raster_for_model(model, usePngFlag, configMode)) {
            require_file(model.pixelPng, "pixel PNG");
        }
        require_file(model.pseudobulkTsv, "pseudobulk TSV");
        if (has_factor_column_range(model.factorColBegin, model.factorColEnd)) {
            validate_factor_column_range(model.factorColBegin, model.factorColEnd,
                infer_factor_count_from_pseudobulk(model.pseudobulkTsv), model.id);
        }
        require_file(model.deTsv, "bulk DE TSV");
        if (!model.skipHexPmtiles && model.hexGridDist <= 0.0) {
            error("%s: model %s has non-positive hex_grid_dist", __func__, model.id.c_str());
        }
        if (hasCell) {
            if (has_cell_sidecars(cell)) {
                error("%s: default model %s must use standard model sidecars for attached cells",
                    __func__, model.id.c_str());
            }
            const bool hasPrebuilt = !cell.cellsPmtiles.empty() || !cell.boundariesPmtiles.empty();
            const bool hasSources = !cell.resultsTsv.empty() || !cell.boundaries.empty();
            if (hasPrebuilt && hasSources) {
                error("%s: model %s attached cells must use either prebuilt pmtiles or source inputs, not both",
                    __func__, model.id.c_str());
            }
            if (hasPrebuilt) {
                require_file(cell.cellsPmtiles, "cell PMTiles");
                require_file(cell.boundariesPmtiles, "cell boundary PMTiles");
            } else {
                require_file(cell.resultsTsv, "cell projection results TSV");
                require_file(cell.boundaries, "cell boundary input");
                if (!cell.centersTsv.empty()) {
                    require_file(cell.centersTsv, "cell centers TSV");
                }
            }
        }
    }
    if (!hasDefaultModel) {
        error("%s: at least one default model is required for transcript and raw-pixel deployment", __func__);
    }
    std::set<std::string> imageIds;
    for (const fs::path& assetJson : inputs.imageAssetJsons) {
        require_file(assetJson, "image asset JSON");
        json asset = read_json_file(assetJson);
        if (!asset.is_object()) {
            error("%s: image asset JSON must be an object: %s", __func__, assetJson.string().c_str());
        }
        for (auto it = asset.begin(); it != asset.end(); ++it) {
            if (!it.value().is_string()) {
                error("%s: image asset JSON value must be a PMTiles path string: %s",
                    __func__, assetJson.string().c_str());
            }
            if (!imageIds.insert(it.key()).second) {
                error("%s: duplicate image id %s", __func__, it.key().c_str());
            }
            require_file(resolve_path(assetJson.parent_path(), it.value().get<std::string>()),
                "image PMTiles");
        }
    }
    for (const image_pmtiles::Options& image : inputs.images) {
        if (!imageIds.insert(image.id).second) {
            error("%s: duplicate image id %s", __func__, image.id.c_str());
        }
        if (image.id.empty()) {
            error("%s: image id must not be empty", __func__);
        }
        require_file(image.inImage, "image input");
        if (image.minZoom < 0 || image.maxZoom < image.minZoom || image.maxZoom > 30) {
            error("%s: image %s has invalid zoom range %d..%d",
                __func__, image.id.c_str(), image.minZoom, image.maxZoom);
        }
        if (image.hasPixZero != image.hasPixMax) {
            error("%s: image %s requires both pix_zero and pix_max",
                __func__, image.id.c_str());
        }
        const bool hasPixReference = image.hasPixZero && image.hasPixMax;
        const int transformSources = (image.hasTransform ? 1 : 0) +
            (image.hasMicronsPerPixel ? 1 : 0) + (hasPixReference ? 1 : 0);
        if (transformSources != 1) {
            error("%s: image %s requires exactly one alignment source: transform, microns_per_pixel, or pix_zero plus pix_max",
                __func__, image.id.c_str());
        }
        if ((image.hasOffsetXUm || image.hasOffsetYUm) && !image.hasMicronsPerPixel) {
            error("%s: image %s offsets require microns_per_pixel",
                __func__, image.id.c_str());
        }
        if (image.hasMicronsPerPixel && !(image.micronsPerPixel > 0.0)) {
            error("%s: image %s has non-positive microns_per_pixel",
                __func__, image.id.c_str());
        }
    }
}

DeployInputs filter_models(DeployInputs inputs, const std::vector<std::string>& requested) {
    if (requested.empty()) {
        return inputs;
    }
    std::set<std::string> wanted;
    for (const std::string& id : requested) {
        wanted.insert(normalize_id(id));
        wanted.insert(id);
    }
    std::vector<ModelInputs> kept;
    for (ModelInputs& model : inputs.models) {
        if (wanted.count(model.id) || wanted.count(model.sourcePrefix) ||
            wanted.count(normalize_id(model.sourcePrefix))) {
            kept.push_back(std::move(model));
        }
    }
    inputs.models = std::move(kept);
    return inputs;
}

std::vector<size_t> raw_pixel_packaging_order(const std::vector<ModelInputs>& models) {
    std::vector<size_t> order;
    order.reserve(models.size());
    for (size_t i = 0; i < models.size(); ++i) {
        if (is_default_model(models[i])) {
            order.push_back(i);
        }
    }
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        return pixel_mode_specificity(models[a].pixelMode) >
            pixel_mode_specificity(models[b].pixelMode);
    });
    return order;
}

const ModelInputs& first_default_model(const std::vector<ModelInputs>& models) {
    for (const ModelInputs& model : models) {
        if (is_default_model(model)) {
            return model;
        }
    }
    error("%s: no default model available", __func__);
    return models.front();
}

void build_pyramid_inplace(const fs::path& pmtilesPath, const fs::path& tmpDir,
    bool pointMode, int32_t minZoom, int32_t maxTileBytes, int32_t maxTileFeatures,
    double compressionScale, int32_t threads, bool quiet = false) {
    fs::path tmpOut = tmpDir / (pmtilesPath.filename().string() + ".pyramid.pmtiles");
    pmtiles_pyramid::BuildOptions options;
    options.minZoom = minZoom;
    options.maxTileBytes = maxTileBytes;
    options.maxTileFeatures = maxTileFeatures;
    options.scaleFactorCompression = compressionScale;
    options.threads = threads;
    options.quiet = quiet;
    if (pointMode) {
        pmtiles_pyramid::build_point_pmtiles_pyramid(pmtilesPath.string(), tmpOut.string(), options);
    } else {
        pmtiles_pyramid::build_polygon_pmtiles_pyramid(pmtilesPath.string(), tmpOut.string(), options);
    }
    if (fs::exists(tmpOut)) {
        std::error_code ec;
        fs::rename(tmpOut, pmtilesPath, ec);
        if (ec) {
            fs::remove(pmtilesPath, ec);
            ec.clear();
            fs::rename(tmpOut, pmtilesPath, ec);
            if (ec) {
                error("%s: failed replacing %s with %s: %s", __func__,
                    pmtilesPath.string().c_str(), tmpOut.string().c_str(), ec.message().c_str());
            }
        }
    }
}

bool point_pyramid_needed(const fs::path& pmtilesPath, int32_t minZoom, int32_t maxZoom) {
    pm_core::LoadedPmtilesArchive archive =
        pm_core::load_pmtiles_archive(pmtilesPath.string());
    if (static_cast<int32_t>(archive.header.max_zoom) < maxZoom) {
        error("%s: existing %s has max zoom %u but deployment requested point max zoom %d; use --overwrite to regenerate point PMTiles",
            __func__, pmtilesPath.string().c_str(),
            static_cast<unsigned>(archive.header.max_zoom), maxZoom);
    }
    return static_cast<int32_t>(archive.header.min_zoom) > minZoom;
}

bool is_genes_all_pmtiles(const fs::path& pmtilesPath) {
    return pmtilesPath.filename().string() == "genes_all.pmtiles";
}

void ensure_point_pyramids(const std::vector<std::string>& pointPmtiles,
    const fs::path& tmpDir, int32_t minZoom, int32_t maxZoom,
    int32_t maxTileBytes, int32_t maxTileFeatures,
    double compressionScale, int32_t threads) {
    std::vector<fs::path> geneBinPmtiles;
    for (const std::string& pmtilesPathString : pointPmtiles) {
        const fs::path pmtilesPath(pmtilesPathString);
        if (!point_pyramid_needed(pmtilesPath, minZoom, maxZoom)) {
            continue;
        }
        if (is_genes_all_pmtiles(pmtilesPath)) {
            notice("%s: building point pyramid for %s with %d thread(s)",
                __func__, pmtilesPath.filename().string().c_str(), threads);
            build_pyramid_inplace(pmtilesPath, tmpDir, true, minZoom,
                maxTileBytes, maxTileFeatures, compressionScale, threads, true);
            notice("%s: finished point pyramid for %s",
                __func__, pmtilesPath.filename().string().c_str());
        } else {
            geneBinPmtiles.push_back(pmtilesPath);
        }
    }

    if (geneBinPmtiles.empty()) {
        return;
    }
    const size_t maxConcurrent = static_cast<size_t>(
        std::max<int32_t>(1, std::min<int32_t>(threads, static_cast<int32_t>(geneBinPmtiles.size()))));
    notice("%s: building %zu gene-bin point pyramids with up to %zu concurrent single-thread job(s)",
        __func__, geneBinPmtiles.size(), maxConcurrent);
    for (size_t begin = 0; begin < geneBinPmtiles.size(); begin += maxConcurrent) {
        const size_t end = std::min(geneBinPmtiles.size(), begin + maxConcurrent);
        std::vector<std::future<fs::path>> futures;
        futures.reserve(end - begin);
        for (size_t i = begin; i < end; ++i) {
            const fs::path pmtilesPath = geneBinPmtiles[i];
            futures.push_back(std::async(std::launch::async, [=]() {
                build_pyramid_inplace(pmtilesPath, tmpDir, true, minZoom,
                    maxTileBytes, maxTileFeatures, compressionScale, 1, true);
                return pmtilesPath;
            }));
        }
        for (auto& future : futures) {
            const fs::path finished = future.get();
            notice("%s: finished gene-bin point pyramid for %s",
                __func__, finished.filename().string().c_str());
        }
    }
}

void copy_prebuilt_pmtiles_or_keep_in_place(const fs::path& src, const fs::path& dst,
    bool overwrite, const char* label);

void write_catalog_yaml(const fs::path& outPath, const std::string& id, const std::string& title,
    const std::string& desc, const std::vector<std::string>& binPmtiles,
    const std::string& basemapPmtilesName, const json& factorAssets,
    const json& imageBasemaps) {
    auto q = [](const std::string& s) { return json(s).dump(); };
    auto write_asset_scalar = [&](std::ostringstream& out, const json& asset,
        const char* key, const char* yamlKey = nullptr) {
        if (!asset.contains(key)) {
            return;
        }
        const char* outKey = yamlKey == nullptr ? key : yamlKey;
        if (asset.at(key).is_boolean()) {
            out << "    " << outKey << ": " << (asset.at(key).get<bool>() ? "true" : "false") << "\n";
        } else if (asset.at(key).is_string()) {
            out << "    " << outKey << ": " << q(asset.at(key).get<std::string>()) << "\n";
        }
    };
    std::ostringstream y;
    y << "id: " << q(id) << "\n";
    y << "title: " << q(title.empty() ? id : title) << "\n";
    if (!desc.empty()) {
        y << "description: " << q(desc) << "\n";
    }
    y << "assets:\n";
    if (!basemapPmtilesName.empty()) {
        y << "  overview: " << q(basemapPmtilesName) << "\n";
    }
    if (!basemapPmtilesName.empty() || !imageBasemaps.empty()) {
        y << "  basemap:\n";
    }
    if (!basemapPmtilesName.empty()) {
        y << "    sge:\n";
        y << "      default: dark\n";
        y << "      dark: " << q(basemapPmtilesName) << "\n";
    }
    if (imageBasemaps.is_object()) {
        for (auto it = imageBasemaps.begin(); it != imageBasemaps.end(); ++it) {
            y << "    " << it.key() << ": " << q(it.value().get<std::string>()) << "\n";
        }
    }
    y << "  sge:\n";
    y << "    all: genes_all.pmtiles\n";
    y << "    counts: genes_bin_counts.json\n";
    y << "    bins:\n";
    for (const std::string& bin : binPmtiles) {
        y << "    - " << bin << "\n";
    }
    y << "  factors:\n";
    const std::vector<std::string> pmtilesOrder = {
        "hex", "raster", "raw_pixel", "cells", "boundaries"
    };
    for (const auto& asset : factorAssets) {
        y << "  - id: " << q(asset.value("id", std::string())) << "\n";
        y << "    name: " << q(asset.value("name", asset.value("id", std::string()))) << "\n";
        write_asset_scalar(y, asset, "model_id");
        write_asset_scalar(y, asset, "decode_id");
        write_asset_scalar(y, asset, "cells_id");
        write_asset_scalar(y, asset, "raw_pixel_col");
        write_asset_scalar(y, asset, "de");
        write_asset_scalar(y, asset, "info");
        write_asset_scalar(y, asset, "model");
        write_asset_scalar(y, asset, "post");
        write_asset_scalar(y, asset, "rgb");
        write_asset_scalar(y, asset, "alias");
        y << "    pmtiles:\n";
        const json& pm = asset.at("pmtiles");
        std::set<std::string> written;
        for (const std::string& key : pmtilesOrder) {
            if (pm.contains(key)) {
                y << "      " << key << ": " << q(pm.at(key).get<std::string>()) << "\n";
                written.insert(key);
            }
        }
        for (auto it = pm.begin(); it != pm.end(); ++it) {
            if (!written.count(it.key())) {
                y << "      " << it.key() << ": " << q(it.value().get<std::string>()) << "\n";
            }
        }
    }
    write_text(outPath, y.str());
}

json make_cell_asset(const CellInputs& cell) {
    json asset;
    asset["id"] = cell.id;
    asset["name"] = cell.id;
    asset["model_id"] = cell.id;
    asset["decode_id"] = cell.id;
    asset["cells_id"] = cell.id;
    asset["raw_pixel_col"] = false;
    asset["pmtiles"] = {
        {"cells", cell.id + "-cells.pmtiles"},
        {"boundaries", cell.id + "-boundaries.pmtiles"}
    };
    asset["de"] = cell.id + "-bulk-de.tsv";
    asset["info"] = cell.id + "-info.tsv";
    asset["post"] = cell.id + "-pseudobulk.tsv.gz";
    if (!cell.rgbTsv.empty()) {
        asset["rgb"] = cell.id + "-rgb.tsv";
    }
    return asset;
}

json copy_image_asset_jsons(const std::vector<fs::path>& assetJsons,
    const fs::path& outRoot, bool overwrite) {
    json basemaps = json::object();
    for (const fs::path& assetJsonPath : assetJsons) {
        json asset = read_json_file(assetJsonPath);
        for (auto it = asset.begin(); it != asset.end(); ++it) {
            const std::string imageId = it.key();
            const fs::path src = resolve_path(assetJsonPath.parent_path(),
                it.value().get<std::string>());
            const fs::path dst = outRoot / src.filename();
            copy_prebuilt_pmtiles_or_keep_in_place(src, dst, overwrite, "image PMTiles");
            basemaps[imageId] = dst.filename().string();
        }
    }
    return basemaps;
}

json deploy_source_images(const std::vector<image_pmtiles::Options>& images,
    const fs::path& outRoot, bool overwrite) {
    json basemaps = json::object();
    for (image_pmtiles::Options options : images) {
        options.outPrefix = outRoot / options.id;
        options.assetJson = outRoot / (options.id + "_assets.json");
        options.overwrite = overwrite;
        image_pmtiles::Asset asset = image_pmtiles::write_image_pmtiles(options);
        basemaps[asset.id] = asset.pmtilesPath.filename().string();
    }
    return basemaps;
}

json merge_image_basemaps(json lhs, const json& rhs) {
    if (!lhs.is_object()) {
        lhs = json::object();
    }
    for (auto it = rhs.begin(); it != rhs.end(); ++it) {
        if (lhs.contains(it.key())) {
            error("%s: duplicate image basemap id %s", __func__, it.key().c_str());
        }
        lhs[it.key()] = it.value();
    }
    return lhs;
}

void copy_prebuilt_pmtiles_or_keep_in_place(const fs::path& src, const fs::path& dst,
    bool overwrite, const char* label) {
    std::error_code ec;
    if (fs::exists(src, ec) && fs::exists(dst, ec) && fs::equivalent(src, dst, ec)) {
        notice("%s: %s already in deployment directory; leaving in place",
            __func__, src.string().c_str());
        return;
    }
    copy_file_checked(src, dst, overwrite, label);
}

void write_or_compute_cell_de(const CellInputs& cell, const fs::path& dst,
    int32_t threads, bool overwrite, const std::vector<std::string>& factorNames,
    const fs::path& tmpDir) {
    const bool namedFactors = !factorNames.empty() && !factor_names_are_numeric(factorNames);
    if (!cell.deTsv.empty()) {
        write_cartoscope_de(cell.deTsv, dst, overwrite, factorNames);
        return;
    }
    if (!overwrite && file_exists(dst)) {
        if (!namedFactors || existing_de_uses_numeric_factors(dst)) {
            notice("%s: %s already exists; skipping cell DE", __func__, dst.string().c_str());
            return;
        }
        notice("%s: %s has named factors but numeric factor ids are required; recomputing cell DE",
            __func__, dst.string().c_str());
    }
    fs::create_directories(tmpDir);
    const fs::path deOut = namedFactors
        ? tmpDir / (cell.id + ".cell_de.raw.tsv")
        : dst;
    std::vector<std::string> args = {
        "de-chisq",
        "--input", cell.pseudobulkTsv.string(),
        "--out", deOut.string(),
        "--threads", std::to_string(threads)
    };
    if (call_command(args, cmdDeChisq) != 0) {
        error("%s: de-chisq failed for cell factor %s", __func__, cell.id.c_str());
    }
    if (namedFactors) {
        write_cartoscope_de(deOut, dst, true, factorNames);
    }
}

json deploy_cell_pmtiles(const CellInputs& cell, const fs::path& outRoot,
    const std::string& pmtilesFormat, int32_t polygonMinZoom, int32_t polygonMaxZoom,
    int32_t maxPointTileBytes, int32_t maxPointTileFeatures,
    int32_t maxPolygonTileBytes, int32_t maxPolygonTileFeatures,
    double compressionScale, int32_t threads, bool overwrite,
    const std::vector<std::string>& factorNames, const fs::path& tmpDir) {
    const fs::path outPrefix = outRoot / cell.id;
    const fs::path outCells = outRoot / (cell.id + "-cells.pmtiles");
    const fs::path outBoundaries = outRoot / (cell.id + "-boundaries.pmtiles");
    const bool hasPrebuilt = !cell.cellsPmtiles.empty() || !cell.boundariesPmtiles.empty();
    if (hasPrebuilt) {
        copy_prebuilt_pmtiles_or_keep_in_place(cell.cellsPmtiles, outCells, overwrite, "cell PMTiles");
        copy_prebuilt_pmtiles_or_keep_in_place(cell.boundariesPmtiles, outBoundaries, overwrite, "cell boundary PMTiles");
    } else if (!overwrite && file_exists(outCells) && file_exists(outBoundaries)) {
        notice("%s: cell PMTiles already exist for %s; skipping generation",
            __func__, cell.id.c_str());
    } else {
        const fs::path resultsForPmtiles =
            normalize_cell_results_for_pmtiles(cell, factorNames, tmpDir);
        std::vector<std::string> args = {
            "cells2pmtiles",
            "--in-results", resultsForPmtiles.string(),
            "--in-boundaries", cell.boundaries.string(),
            "--out-prefix", outPrefix.string(),
            "--format", pmtilesFormat,
            "--boundary-format", cell.boundaryFormat,
            "--id-col", cell.resultIdCol,
            "--b-icol-id", std::to_string(cell.bIcolId),
            "--b-icol-x", std::to_string(cell.bIcolX),
            "--b-icol-y", std::to_string(cell.bIcolY),
            "--c-icol-id", std::to_string(cell.cIcolId),
            "--c-icol-x", std::to_string(cell.cIcolX),
            "--c-icol-y", std::to_string(cell.cIcolY),
            "--boundary-id-prop", cell.boundaryIdProp,
            "--min-zoom", std::to_string(polygonMinZoom),
            "--max-zoom", std::to_string(polygonMaxZoom),
            "--max-point-tile-bytes", std::to_string(maxPointTileBytes),
            "--max-point-tile-features", std::to_string(maxPointTileFeatures),
            "--max-polygon-tile-bytes", std::to_string(maxPolygonTileBytes),
            "--max-polygon-tile-features", std::to_string(maxPolygonTileFeatures),
            "--scale-factor-compression", fp_to_string(compressionScale, 6),
            "--threads", std::to_string(threads)
        };
        if (!cell.centersTsv.empty()) {
            args.push_back("--in-centers");
            args.push_back(cell.centersTsv.string());
        }
        if (overwrite) {
            args.push_back("--overwrite");
        }
        if (call_command(args, cmdCells2Pmtiles) != 0) {
            error("%s: cells2pmtiles failed for cell factor %s", __func__, cell.id.c_str());
        }
    }
    json pmtiles;
    pmtiles["cells"] = cell.id + "-cells.pmtiles";
    pmtiles["boundaries"] = cell.id + "-boundaries.pmtiles";
    return pmtiles;
}

json deploy_cell_factor(const CellInputs& cell, const fs::path& outRoot,
    const std::string& pmtilesFormat, int32_t polygonMinZoom, int32_t polygonMaxZoom,
    int32_t maxPointTileBytes, int32_t maxPointTileFeatures,
    int32_t maxPolygonTileBytes, int32_t maxPolygonTileFeatures,
    double compressionScale, int32_t threads, bool overwrite,
    const fs::path& tmpDir) {
    const std::vector<std::string> factorNames =
        factor_names_from_pseudobulk_header(cell.pseudobulkTsv);
    deploy_cell_pmtiles(cell, outRoot, pmtilesFormat,
        polygonMinZoom, polygonMaxZoom,
        maxPointTileBytes, maxPointTileFeatures,
        maxPolygonTileBytes, maxPolygonTileFeatures,
        compressionScale, threads, overwrite, factorNames, tmpDir);
    write_or_compute_cell_de(cell, outRoot / (cell.id + "-bulk-de.tsv"),
        threads, overwrite, factorNames, tmpDir);
    write_cartoscope_info_from_pseudobulk(cell.pseudobulkTsv,
        outRoot / (cell.id + "-bulk-de.tsv"), cell.rgbTsv,
        outRoot / (cell.id + "-info.tsv"), overwrite);
    gzip_copy_text(cell.pseudobulkTsv, outRoot / (cell.id + "-pseudobulk.tsv.gz"), overwrite);
    if (!cell.rgbTsv.empty()) {
        const size_t factorCount = factorNames.size();
        write_cartoscope_rgb(cell.rgbTsv, outRoot / (cell.id + "-rgb.tsv"), factorCount, overwrite);
    }
    const bool hasAliasSidecar = write_factor_alias_from_pseudobulk(cell.pseudobulkTsv,
        outRoot / (cell.id + "-alias.tsv"), overwrite);
    json asset = make_cell_asset(cell);
    if (hasAliasSidecar) {
        asset["alias"] = cell.id + "-alias.tsv";
    }
    return asset;
}

} // namespace

int32_t cmdDeployCartoscope(int32_t argc, char** argv) {
    std::string inputJson;
    std::vector<std::string> imageJsons;
    std::string configPath;
    std::string outDir;
    std::string datasetId;
    std::string title;
    std::string desc;
    std::string pmtilesFormat;
    std::vector<std::string> modelPrefixes;
    int32_t pointMinZoom = 10;
    int32_t pointMaxZoom = 18;
    int32_t polygonMinZoom = 10;
    int32_t polygonMaxZoom = 18;
    int32_t nGeneBins = 50;
    uint64_t geneBinTargetMolecules = 1000000;
    int32_t threads = 4;
    int32_t maxPointTileBytes = pmtiles_pyramid::DEFAULT_POINT_MAX_TILE_BYTES;
    int32_t maxPointTileFeatures = pmtiles_pyramid::DEFAULT_POINT_MAX_TILE_FEATURES;
    int32_t maxPolygonTileBytes = pmtiles_pyramid::DEFAULT_POLYGON_MAX_TILE_BYTES;
    int32_t maxPolygonTileFeatures = pmtiles_pyramid::DEFAULT_POLYGON_MAX_TILE_FEATURES;
    int32_t basemapMinZoom = 7;
    int32_t basemapMaxZoom = -1;
    int32_t monoMaxZoomFromRaw = -1;
    double compressionScale = 10.0;
    double geneBinSingletonRatio = 1.0;
    double hexProbThreshold = 0.001;
    double basemapAdjustQuantile = 0.99;
    bool overwrite = false;
    bool usePng = false;
    bool skipBasemap = false;
    std::string geneBinMode = "adaptive";
    std::string basemapDisplayTransform = "linear";
    std::string nullStr;

    ParamList pl;
    pl.add_option("config", "Standard workflow config JSON", configPath)
      .add_option("input-json", "Explicit deployment input JSON", inputJson)
      .add_option("image-json", "image2pmtiles metadata JSON from convert_tiff_for_image2pmtiles.py; may be repeated", imageJsons)
      .add_option("out-dir", "Output deployment directory", outDir, true)
      .add_option("id", "Dataset ID", datasetId, true)
      .add_option("title", "Dataset title", title)
      .add_option("desc", "Dataset description", desc)
      .add_option("pmtiles-format", "Tile encoding format: MLT or MVT", pmtilesFormat, true)
      .add_option("model-prefix", "Optional model prefixes/IDs to deploy", modelPrefixes)
      .add_option("point-min-zoom", "Minimum zoom for transcript PMTiles pyramids", pointMinZoom)
      .add_option("point-max-zoom", "Maximum zoom for transcript PMTiles export", pointMaxZoom)
      .add_option("polygon-min-zoom", "Minimum zoom for hex PMTiles pyramids", polygonMinZoom)
      .add_option("polygon-max-zoom", "Maximum zoom for hex PMTiles export", polygonMaxZoom)
      .add_option("n-gene-bins", "Maximum number of gene bins for transcript PMTiles in adaptive mode", nGeneBins)
      .add_option("gene-bin-mode", "Gene-bin packing mode: adaptive or fixed", geneBinMode)
      .add_option("gene-bin-target-molecules", "Target molecules per adaptive gene bin", geneBinTargetMolecules)
      .add_option("gene-bin-singleton-ratio", "Adaptive singleton threshold as a multiple of target molecules", geneBinSingletonRatio)
      .add_option("threads", "Number of threads", threads)
      .add_option("max-point-tile-bytes", "Maximum compressed bytes per point tile", maxPointTileBytes)
      .add_option("max-point-tile-features", "Maximum features per point tile", maxPointTileFeatures)
      .add_option("max-polygon-tile-bytes", "Maximum compressed bytes per polygon tile", maxPolygonTileBytes)
      .add_option("max-polygon-tile-features", "Maximum features per polygon tile", maxPolygonTileFeatures)
      .add_option("basemap-min-zoom", "Minimum zoom for SGE mono basemap PMTiles", basemapMinZoom)
      .add_option("basemap-max-zoom", "Maximum zoom for SGE mono basemap PMTiles (default: point max zoom)", basemapMaxZoom)
      .add_option("mono-max-zoom-from-raw", "Parse raw data for SGE mono basemap zoom levels >= this value; derive lower zooms from parent layers", monoMaxZoomFromRaw)
      .add_option("basemap-adjust-quantile", "Quantile for SGE mono basemap density auto-adjustment", basemapAdjustQuantile)
      .add_option("basemap-display-transform", "Display transform for SGE mono basemap intensity: linear or log1p", basemapDisplayTransform)
      .add_option("scale-factor-compression", "Pyramid compression aggressiveness estimate", compressionScale)
      .add_option("hex-prob-thres", "Minimum hex factor probability retained", hexProbThreshold)
      .add_option("null-str", "Replace empty string query properties with this placeholder in transcript PMTiles packaging", nullStr)
      .add_option("use-png", "Use pre-rendered pixel PNGs for raster PMTiles in --config mode", usePng)
      .add_option("skip-basemap", "Skip SGE mono basemap PMTiles generation", skipBasemap)
      .add_option("overwrite", "Overwrite existing deployment output files", overwrite);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    pmtilesFormat = to_upper(pmtilesFormat);
    if (pmtilesFormat != "MLT" && pmtilesFormat != "MVT") {
        error("%s: --pmtiles-format must be MLT or MVT", __func__);
    }
    if (configPath.empty() == inputJson.empty()) {
        error("%s: provide exactly one of --config or --input-json", __func__);
    }
    if (!imageJsons.empty() && inputJson.empty()) {
        error("%s: --image-json requires --input-json", __func__);
    }
    if (nGeneBins <= 0) {
        error("%s: --n-gene-bins must be positive", __func__);
    }
    parse_gene_bin_mode(geneBinMode);
    if (basemapMaxZoom < 0) {
        basemapMaxZoom = pointMaxZoom;
    }
    if (basemapMinZoom < 0 || basemapMaxZoom < basemapMinZoom || basemapMaxZoom > 30) {
        error("%s: invalid basemap zoom range %d..%d", __func__, basemapMinZoom, basemapMaxZoom);
    }
    const tiles2mono::DisplayTransform monoDisplayTransform =
        tiles2mono::parse_display_transform(basemapDisplayTransform);

    fs::path outRoot(outDir);
    fs::path tmpDir = outRoot / "tmp";
    fs::create_directories(outRoot);
    fs::create_directories(tmpDir);

    const bool configMode = inputJson.empty();
    DeployInputs inputs;
    if (!inputJson.empty()) {
        inputs = load_from_input_json(inputJson, imageJsons);
    } else {
        inputs = load_from_config(configPath);
    }
    if (!nullStr.empty()) {
        inputs.transcripts.nullStr = nullStr;
    }
    inputs = filter_models(std::move(inputs), modelPrefixes);
    validate_inputs(inputs, usePng, configMode);
    const fs::path annotateTiledPrefix = prepare_headered_transcripts(inputs.transcripts, tmpDir);

    fs::path dotIndex = outRoot / "genes.pmtiles_index.tsv";
    fs::path dotCounts = outRoot / "genes.bin_counts.json";
    fs::path deployIndex = outRoot / "genes_pmtiles_index.tsv";
    fs::path deployCounts = outRoot / "genes_bin_counts.json";
    std::vector<std::string> pointPmtiles;
    if (!overwrite && point_outputs_complete(outRoot, nGeneBins, &pointPmtiles)) {
        notice("%s: transcript PMTiles outputs already exist; skipping point PMTiles packaging", __func__);
        if (pointPmtiles.empty()) {
            pointPmtiles = read_pmtiles_paths_from_index(deployIndex, outRoot);
        }
        ensure_point_pyramids(pointPmtiles, tmpDir, pointMinZoom, pointMaxZoom,
            maxPointTileBytes, maxPointTileFeatures, compressionScale, threads);
    } else {
        std::vector<fs::path> presentPointOutputs = existing_files(planned_point_outputs(outRoot, nGeneBins));
        if (!overwrite && !presentPointOutputs.empty()) {
            error("%s: transcript PMTiles outputs are partially present; use --overwrite to regenerate them. Existing files: %s",
                __func__, join_paths(presentPointOutputs).c_str());
        }

        const std::vector<size_t> rawPixelOrder = raw_pixel_packaging_order(inputs.models);
        fs::path genesPrefix = outRoot / "genes";
        std::vector<std::string> tileArgs = {
            "tile-op",
            "--in", inputs.models[rawPixelOrder.front()].pixelPrefix.string(),
            "--binary",
            "--annotate-pts", annotateTiledPrefix.string(),
            "--icol-x", std::to_string(inputs.transcripts.icolX),
            "--icol-y", std::to_string(inputs.transcripts.icolY),
            "--icol-feature", std::to_string(inputs.transcripts.icolFeature),
            "--icol-count", std::to_string(inputs.transcripts.icolCount),
            "--anno-keep-all",
            pmtilesFormat == "MVT" ? "--write-mvt-pmtiles" : "--write-mlt-pmtiles",
            "--pmtiles-zoom", std::to_string(pointMaxZoom),
            "--feature-count-file", inputs.transcripts.featureCountTsv.string(),
            "--n-gene-bins", std::to_string(nGeneBins),
            "--gene-bin-mode", geneBinMode,
            "--gene-bin-target-molecules", std::to_string(geneBinTargetMolecules),
            "--gene-bin-singleton-ratio", fp_to_string(geneBinSingletonRatio, 6),
            "--out", genesPrefix.string(),
            "--threads", std::to_string(threads)
        };
        if (pmtilesFormat == "MVT") {
            tileArgs.push_back("--feature-field-name");
            tileArgs.push_back("gene");
        }
        if (!inputs.transcripts.nullStr.empty()) {
            tileArgs.push_back("--null-other");
            tileArgs.push_back(inputs.transcripts.nullStr);
        }
        tileArgs.push_back("--emb-prefix");
        for (size_t modelIdx : rawPixelOrder) {
            tileArgs.push_back(pixel_decode_id(inputs.models[modelIdx]));
        }
        if (inputs.models.size() > 1) {
            tileArgs.push_back("--merge-emb");
            for (size_t i = 1; i < rawPixelOrder.size(); ++i) {
                tileArgs.push_back(inputs.models[rawPixelOrder[i]].pixelPrefix.string() + ".bin");
            }
            tileArgs.push_back("--merge-keep-all");
        }
        if (call_command(tileArgs, cmdManipulateTiles) != 0) {
            error("%s: tile-op transcript PMTiles packaging failed", __func__);
        }

        copy_file_checked(dotIndex, deployIndex);
        copy_file_checked(dotCounts, deployCounts);
        fs::remove(dotIndex);
        fs::remove(dotCounts);
        pointPmtiles = read_pmtiles_paths_from_index(deployIndex, outRoot);
        ensure_point_pyramids(pointPmtiles, tmpDir, pointMinZoom, pointMaxZoom,
            maxPointTileBytes, maxPointTileFeatures, compressionScale, threads);
    }

    json factorAssets = json::array();
    std::vector<std::string> binPmtiles;
    for (const std::string& pmtilesPath : pointPmtiles) {
        std::string fn = fs::path(pmtilesPath).filename().string();
        if (fn != "genes_all.pmtiles") {
            binPmtiles.push_back(fn);
        }
    }

    std::string rawPixelPmtilesName = "genes_all.pmtiles";

    pm_raster::RasterBounds rasterBounds =
        resolve_raster_bounds(inputs.transcripts, first_default_model(inputs.models));

    std::string basemapPmtilesName;
    if (!skipBasemap) {
        basemapPmtilesName = "sge-mono-dark.pmtiles";
        fs::path basemapPmtiles = outRoot / basemapPmtilesName;
        if (!overwrite && file_exists(basemapPmtiles)) {
            notice("%s: %s already exists; skipping SGE mono basemap PMTiles",
                __func__, basemapPmtiles.string().c_str());
        } else {
            tiles2mono::Options monoOptions;
            monoOptions.dataFile = inputs.transcripts.tiledPrefix.string() + ".tsv";
            monoOptions.indexFile = inputs.transcripts.tiledPrefix.string() + ".index";
            monoOptions.rangeFile = inputs.transcripts.tiledPrefix.string() + ".coord_range.tsv";
            monoOptions.outFile = basemapPmtiles.string();
            monoOptions.tempBlobFile = (tmpDir / "sge-mono-dark.blob").string();
            monoOptions.icolX = inputs.transcripts.icolX;
            monoOptions.icolY = inputs.transcripts.icolY;
            monoOptions.icolCount = inputs.transcripts.icolCount;
            monoOptions.minZoom = basemapMinZoom;
            monoOptions.maxZoom = basemapMaxZoom;
            monoOptions.maxZoomFromRaw = monoMaxZoomFromRaw;
            monoOptions.threads = threads;
            monoOptions.adjustQuantile = basemapAdjustQuantile;
            monoOptions.displayTransform = monoDisplayTransform;
            monoOptions.bounds = rasterBounds;
            tiles2mono::write_tiles2mono_pmtiles(monoOptions);
        }
    }

    for (const ModelInputs& model : inputs.models) {
        if (!is_default_model(model)) {
            continue;
        }
        fs::path hexPmtiles = outRoot / (model.id + ".pmtiles");
        if (model.skipHexPmtiles) {
            notice("%s: skipping hex PMTiles packaging for model %s", __func__, model.id.c_str());
        } else {
            if (!overwrite && file_exists(hexPmtiles)) {
                notice("%s: %s already exists; skipping hex PMTiles packaging", __func__, hexPmtiles.string().c_str());
            } else {
                fs::path resultsForPmtiles = prepare_hex_results_with_topk(model, tmpDir);
                std::vector<std::string> hexArgs = {
                    "poly2pmtiles",
                    "--in-tsv", resultsForPmtiles.string(),
                    "--out", hexPmtiles.string(),
                    "--format", pmtilesFormat,
                    "--hex-grid-dist", fp_to_string(model.hexGridDist, 6),
                    "--pmtiles-zoom", std::to_string(polygonMaxZoom),
                    "--prob-thres", fp_to_string(hexProbThreshold, 8),
                    "--threads", std::to_string(threads)
                };
                if (has_factor_column_range(model.factorColBegin, model.factorColEnd)) {
                    hexArgs.push_back("--factor-col-begin");
                    hexArgs.push_back(std::to_string(model.factorColBegin));
                    hexArgs.push_back("--factor-col-end");
                    hexArgs.push_back(std::to_string(model.factorColEnd));
                }
                if (call_command(hexArgs, cmdPoly2Pmtiles) != 0) {
                    error("%s: poly2pmtiles failed for model %s", __func__, model.id.c_str());
                }
                notice("%s: building polygon pyramid for %s with %d thread(s)",
                    __func__, hexPmtiles.filename().string().c_str(), threads);
                build_pyramid_inplace(hexPmtiles, tmpDir, false, polygonMinZoom,
                    maxPolygonTileBytes, maxPolygonTileFeatures, compressionScale, threads, true);
                notice("%s: finished polygon pyramid for %s",
                    __func__, hexPmtiles.filename().string().c_str());
            }
        }

        fs::path rasterPmtiles = outRoot / (model.id + "-pixel-raster.pmtiles");
        if (!overwrite && file_exists(rasterPmtiles)) {
            notice("%s: %s already exists; skipping pixel raster PMTiles", __func__,
                rasterPmtiles.string().c_str());
        } else {
            if (use_png_raster_for_model(model, usePng, configMode)) {
                pm_raster::write_png_raster_pmtiles_archive(model.pixelPng.string(),
                    rasterPmtiles.string(), (tmpDir / (model.id + ".raster.blob")).string(),
                    rasterBounds, polygonMinZoom, polygonMaxZoom);
            } else {
                write_raw_pixel_raster_pmtiles_archive(model,
                    rasterPmtiles.string(), (tmpDir / (model.id + ".raster.blob")).string(),
                    rasterBounds, polygonMinZoom, polygonMaxZoom);
            }
        }

        copy_file_checked(model.modelTsv, outRoot / (model.id + "-model.tsv"), overwrite, "model sidecar");
        const std::vector<std::string> factorNames =
            factor_names_from_pseudobulk_header(model.pseudobulkTsv);
        const size_t factorCount = factorNames.size();
        write_cartoscope_rgb(model.colorRgbTsv, outRoot / (model.id + "-rgb.tsv"),
            factorCount, overwrite);
        write_cartoscope_de(model.deTsv, outRoot / (model.id + "-bulk-de.tsv"),
            overwrite, factorNames);
        write_cartoscope_info_from_pseudobulk(model.pseudobulkTsv,
            outRoot / (model.id + "-bulk-de.tsv"), model.colorRgbTsv,
            outRoot / (model.id + "-info.tsv"), overwrite);
        const bool hasAliasSidecar = write_factor_alias_from_pseudobulk(model.pseudobulkTsv,
            outRoot / (model.id + "-alias.tsv"), overwrite);
        gzip_copy_text(model.pseudobulkTsv, outRoot / (model.id + "-pseudobulk.tsv.gz"), overwrite);

        json asset;
        const std::string decodeId = pixel_decode_id(model);
        asset["id"] = model.id;
        asset["name"] = model.id;
        asset["model_id"] = model.id;
        asset["decode_id"] = decodeId;
        asset["raw_pixel_col"] = decodeId;
        asset["de"] = model.id + "-bulk-de.tsv";
        asset["info"] = model.id + "-info.tsv";
        asset["model"] = model.id + "-model.tsv";
        asset["post"] = model.id + "-pseudobulk.tsv.gz";
        asset["rgb"] = model.id + "-rgb.tsv";
        if (hasAliasSidecar) {
            asset["alias"] = model.id + "-alias.tsv";
        }
        asset["pmtiles"] = {
            {"raster", model.id + "-pixel-raster.pmtiles"},
            {"raw_pixel", rawPixelPmtilesName}
        };
        if (!model.skipHexPmtiles) {
            asset["pmtiles"]["hex"] = model.id + ".pmtiles";
        }
        if (has_cell_pmtiles_or_sources(model.cell)) {
            json cellPmtiles = deploy_cell_pmtiles(model.cell, outRoot, pmtilesFormat,
                polygonMinZoom, polygonMaxZoom,
                maxPointTileBytes, maxPointTileFeatures,
                maxPolygonTileBytes, maxPolygonTileFeatures,
                compressionScale, threads, overwrite, factorNames, tmpDir);
            asset["cells_id"] = model.id;
            asset["pmtiles"]["cells"] = cellPmtiles.at("cells");
            if (cellPmtiles.contains("boundaries")) {
                asset["pmtiles"]["boundaries"] = cellPmtiles.at("boundaries");
            }
        }
        factorAssets.push_back(asset);
    }

    for (const ModelInputs& model : inputs.models) {
        if (!is_cell_only_model(model)) {
            continue;
        }
        json asset = deploy_cell_factor(model.cell, outRoot, pmtilesFormat,
            polygonMinZoom, polygonMaxZoom,
            maxPointTileBytes, maxPointTileFeatures,
            maxPolygonTileBytes, maxPolygonTileFeatures,
            compressionScale, threads, overwrite, tmpDir);
        factorAssets.push_back(asset);
    }

    json imageBasemaps = copy_image_asset_jsons(inputs.imageAssetJsons, outRoot, overwrite);
    imageBasemaps = merge_image_basemaps(imageBasemaps,
        deploy_source_images(inputs.images, outRoot, overwrite));

    write_text_checked(outRoot / "ficture_assets.json", factorAssets.dump(4) + "\n",
        overwrite, "ficture assets");

    write_catalog_yaml(outRoot / "catalog.yaml", datasetId, title, desc, binPmtiles,
        basemapPmtilesName, factorAssets, imageBasemaps);
    fs::remove_all(tmpDir);
    notice("%s: wrote minimal CartoScope deployment to %s", __func__, outRoot.string().c_str());
    return 0;
}
