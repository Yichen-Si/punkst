#include "punkst.h"
#include "dataunits.hpp"
#include "mlt_utils.hpp"
#include "pmtiles_utils.hpp"
#include "pmtiles_pyramid.hpp"
#include "tile_io.hpp"
#include "utils.h"
#include "utils_sys.hpp"

#include "json.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <cstring>
#include <functional>

int32_t cmdManipulateTiles(int32_t argc, char** argv);
int32_t cmdHex2PmtilesMlt(int32_t argc, char** argv);

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;

struct TranscriptInputs {
    fs::path tiledPrefix;
    fs::path featureCountTsv;
    int32_t icolX = 0;
    int32_t icolY = 1;
    int32_t icolFeature = 2;
    int32_t icolCount = 3;
};

struct ModelInputs {
    std::string sourcePrefix;
    std::string id;
    double hexGridDist = 0.0;
    fs::path resultsTsv;
    fs::path modelTsv;
    fs::path colorRgbTsv;
    fs::path pixelPrefix;
    fs::path pixelPng;
    fs::path pseudobulkTsv;
    fs::path deTsv;
    fs::path infoTsv;
};

struct DeployInputs {
    TranscriptInputs transcripts;
    std::vector<ModelInputs> models;
};

std::string normalize_id(std::string s) {
    for (char& c : s) {
        if (c == '_' || c == '.' || c == '/' || c == '\\') {
            c = '-';
        }
    }
    return s;
}

std::string pixel_decode_id(const ModelInputs& model) {
    return model.id + "-pixel";
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
    for (size_t i = 0; i < header.size(); ++i) {
        int32_t k = -1;
        if (str2int32(header[i], k) && k >= 0) {
            factorCols.push_back(static_cast<int32_t>(i));
        }
    }
    if (factorCols.empty()) {
        error("%s: cannot infer factor columns from %s", __func__, model.resultsTsv.string().c_str());
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
        for (int32_t col : factorCols) {
            float value = 0.0f;
            if (!str2float(fields[static_cast<size_t>(col)], value)) {
                error("%s: failed parsing factor probability in %s", __func__, model.resultsTsv.string().c_str());
            }
            if (value > bestP) {
                bestP = value;
                if (!str2int32(header[static_cast<size_t>(col)], bestK)) {
                    bestK = -1;
                }
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

void write_cartoscope_rgb(const fs::path& src, const fs::path& dst, bool overwrite) {
    if (!overwrite && file_exists(dst)) {
        notice("%s: %s already exists; skipping RGB sidecar", __func__, dst.string().c_str());
        return;
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
    fs::create_directories(dst.parent_path());
    std::ofstream out(dst);
    if (!out.is_open()) {
        error("%s: cannot open %s", __func__, dst.string().c_str());
    }
    out << "Name\tColor_index\tR\tG\tB\n";
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
        if (r > 1.0 || g > 1.0 || b > 1.0) {
            r /= 255.0;
            g /= 255.0;
            b /= 255.0;
        }
        std::string name = hasName && fields.size() > static_cast<size_t>(col["Name"])
            ? fields[static_cast<size_t>(col["Name"])] : std::to_string(row);
        std::string colorIndex = hasColorIndex && fields.size() > static_cast<size_t>(col["Color_index"])
            ? fields[static_cast<size_t>(col["Color_index"])] : std::to_string(row);
        out << name << "\t" << colorIndex << "\t"
            << fp_to_string(r, 6) << "\t"
            << fp_to_string(g, 6) << "\t"
            << fp_to_string(b, 6) << "\n";
        ++row;
    }
}

void write_cartoscope_de(const fs::path& src, const fs::path& dst, bool overwrite) {
    if (!overwrite && file_exists(dst)) {
        notice("%s: %s already exists; skipping bulk DE sidecar", __func__, dst.string().c_str());
        return;
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
        out << get(cGene) << "\t"
            << get(cFactor) << "\t"
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

void gzip_copy_text(const fs::path& src, const fs::path& dst, bool overwrite) {
    if (!overwrite && file_exists(dst)) {
        notice("%s: %s already exists; skipping gzip sidecar", __func__, dst.string().c_str());
        return;
    }
    const std::string raw = read_all_text(src);
    write_binary(dst, mlt_pmtiles::gzip_compress(raw));
}

bool bounds_from_index_header(const fs::path& indexPath, mlt_pmtiles::RasterBounds& bounds) {
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

mlt_pmtiles::RasterBounds resolve_raster_bounds(const TranscriptInputs& transcripts,
    const ModelInputs& firstModel) {
    mlt_pmtiles::RasterBounds bounds;
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

json read_json_file(const fs::path& path) {
    std::ifstream in(path);
    if (!in.is_open()) {
        error("%s: cannot open %s", __func__, path.string().c_str());
    }
    return json::parse(in);
}

DeployInputs load_from_config_dir(const fs::path& inDir, const fs::path& configPath) {
    json root = read_json_file(configPath);
    const json& wf = root.at("workflow");
    DeployInputs out;
    out.transcripts.tiledPrefix = inDir / "transcripts.tiled";
    out.transcripts.featureCountTsv = inDir / "transcripts.tiled.features.tsv";
    out.transcripts.icolX = wf.value("icol_x", 0);
    out.transcripts.icolY = wf.value("icol_y", 1);
    out.transcripts.icolFeature = wf.value("icol_feature", 2);
    out.transcripts.icolCount = wf.value("icol_count", 3);

    for (const auto& h : wf.at("hexgrids")) {
        for (const auto& k : wf.at("topics")) {
            const int32_t hex = h.get<int32_t>();
            const int32_t topics = k.get<int32_t>();
            ModelInputs model;
            model.sourcePrefix = "hex_" + std::to_string(hex) + ".k" + std::to_string(topics);
            model.id = normalize_id(model.sourcePrefix);
            model.hexGridDist = static_cast<double>(hex);
            fs::path pref = inDir / model.sourcePrefix;
            model.resultsTsv = pref.string() + ".results.tsv";
            model.modelTsv = pref.string() + ".model.tsv";
            model.colorRgbTsv = pref.string() + ".color.rgb.tsv";
            model.pixelPrefix = pref.string() + ".pixel";
            model.pixelPng = pref.string() + ".pixel.png";
            model.pseudobulkTsv = pref.string() + ".pixel.pseudobulk.tsv";
            model.deTsv = pref.string() + ".pixel.de_bulk.tsv";
            model.infoTsv = pref.string() + ".pixel.info.tsv";
            out.models.push_back(std::move(model));
        }
    }
    return out;
}

DeployInputs load_from_input_json(const fs::path& inputJson) {
    json root = read_json_file(inputJson);
    fs::path base = inputJson.parent_path();
    DeployInputs out;
    const json& tr = root.at("transcripts");
    out.transcripts.tiledPrefix = resolve_path(base, tr.at("tiled_prefix").get<std::string>());
    out.transcripts.featureCountTsv = resolve_path(base, tr.at("feature_count_tsv").get<std::string>());
    out.transcripts.icolX = tr.value("icol_x", 0);
    out.transcripts.icolY = tr.value("icol_y", 1);
    out.transcripts.icolFeature = tr.value("icol_feature", 2);
    out.transcripts.icolCount = tr.value("icol_count", 3);

    for (const auto& m : root.at("models")) {
        ModelInputs model;
        model.id = normalize_id(m.at("id").get<std::string>());
        model.sourcePrefix = model.id;
        model.hexGridDist = m.at("hex_grid_dist").get<double>();
        model.resultsTsv = resolve_path(base, m.at("results_tsv").get<std::string>());
        model.modelTsv = resolve_path(base, m.at("model_tsv").get<std::string>());
        model.colorRgbTsv = resolve_path(base, m.at("color_rgb_tsv").get<std::string>());
        model.pixelPrefix = resolve_path(base, m.at("pixel_prefix").get<std::string>());
        if (m.contains("pixel_png") && !m.at("pixel_png").is_null()) {
            model.pixelPng = resolve_path(base, m.at("pixel_png").get<std::string>());
        }
        model.pseudobulkTsv = resolve_path(base, m.at("pseudobulk_tsv").get<std::string>());
        model.deTsv = resolve_path(base, m.at("de_tsv").get<std::string>());
        if (m.contains("info_tsv") && !m.at("info_tsv").is_null()) {
            model.infoTsv = resolve_path(base, m.at("info_tsv").get<std::string>());
        }
        out.models.push_back(std::move(model));
    }
    return out;
}

void validate_inputs(const DeployInputs& inputs) {
    require_file(inputs.transcripts.tiledPrefix.string() + ".tsv", "tiled transcript TSV");
    require_file(inputs.transcripts.tiledPrefix.string() + ".index", "tiled transcript index");
    require_file(inputs.transcripts.featureCountTsv, "feature-count TSV");
    if (inputs.models.empty()) {
        error("%s: no models were discovered or provided", __func__);
    }
    std::set<std::string> ids;
    for (const ModelInputs& model : inputs.models) {
        if (!ids.insert(model.id).second) {
            error("%s: duplicate model id %s", __func__, model.id.c_str());
        }
        require_file(model.resultsTsv, "model results TSV");
        require_file(model.modelTsv, "model TSV");
        require_file(model.colorRgbTsv, "color RGB TSV");
        require_file(model.pixelPrefix.string() + ".bin", "pixel decode binary");
        require_file(model.pixelPrefix.string() + ".index", "pixel decode index");
        if (!model.pixelPng.empty()) {
            require_file(model.pixelPng, "pixel PNG");
        }
        require_file(model.pseudobulkTsv, "pseudobulk TSV");
        require_file(model.deTsv, "bulk DE TSV");
        if (!model.infoTsv.empty()) {
            require_file(model.infoTsv, "factor info TSV");
        }
        if (model.hexGridDist <= 0.0) {
            error("%s: model %s has non-positive hex_grid_dist", __func__, model.id.c_str());
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
        if (wanted.count(model.id) || wanted.count(model.sourcePrefix)) {
            kept.push_back(std::move(model));
        }
    }
    inputs.models = std::move(kept);
    return inputs;
}

void build_pyramid_inplace(const fs::path& pmtilesPath, const fs::path& tmpDir,
    bool pointMode, int32_t minZoom, int32_t maxTileBytes, int32_t maxTileFeatures,
    double compressionScale, int32_t threads) {
    fs::path tmpOut = tmpDir / (pmtilesPath.filename().string() + ".pyramid.pmtiles");
    pmtiles_pyramid::BuildOptions options;
    options.minZoom = minZoom;
    options.maxTileBytes = maxTileBytes;
    options.maxTileFeatures = maxTileFeatures;
    options.scaleFactorCompression = compressionScale;
    options.threads = threads;
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

void write_catalog_yaml(const fs::path& outPath, const std::string& id, const std::string& title,
    const std::string& desc, const std::vector<std::string>& binPmtiles,
    const std::vector<ModelInputs>& models, const std::string& rawPixelPmtilesName,
    bool overwrite) {
    auto q = [](const std::string& s) { return json(s).dump(); };
    std::ostringstream y;
    y << "id: " << q(id) << "\n";
    y << "title: " << q(title.empty() ? id : title) << "\n";
    if (!desc.empty()) {
        y << "description: " << q(desc) << "\n";
    }
    y << "assets:\n";
    y << "  sge:\n";
    y << "    all: genes_all.pmtiles\n";
    y << "    bins:\n";
    for (const std::string& bin : binPmtiles) {
        y << "    - " << bin << "\n";
    }
    y << "    counts: genes_bin_counts.json\n";
    y << "  factors:\n";
    for (const ModelInputs& model : models) {
        const std::string decodeId = pixel_decode_id(model);
        y << "  - id: " << q(model.id) << "\n";
        y << "    name: " << q(model.id) << "\n";
        y << "    model_id: " << q(model.id) << "\n";
        y << "    decode_id: " << q(decodeId) << "\n";
        y << "    raw_pixel_col: " << q(decodeId) << "\n";
        y << "    de: " << q(model.id + "-bulk-de.tsv") << "\n";
        y << "    info: " << q(model.id + "-info.tsv") << "\n";
        y << "    model: " << q(model.id + "-model.tsv") << "\n";
        y << "    post: " << q(model.id + "-pseudobulk.tsv.gz") << "\n";
        y << "    rgb: " << q(model.id + "-rgb.tsv") << "\n";
        y << "    pmtiles:\n";
        y << "      hex: " << q(model.id + ".pmtiles") << "\n";
        if (!model.pixelPng.empty()) {
            y << "      raster: " << q(model.id + "-pixel-raster.pmtiles") << "\n";
        }
        y << "      raw_pixel: " << q(rawPixelPmtilesName) << "\n";
    }
    write_text_checked(outPath, y.str(), overwrite, "catalog");
}

} // namespace

int32_t cmdDeployCartoscope(int32_t argc, char** argv) {
    std::string inDir;
    std::string inputJson;
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
    int32_t threads = 4;
    int32_t maxPointTileBytes = 500000;
    int32_t maxPointTileFeatures = 50000;
    int32_t maxPolygonTileBytes = 500000;
    int32_t maxPolygonTileFeatures = 5000;
    double compressionScale = 10.0;
    double hexProbThreshold = 0.001;
    bool overwrite = false;

    ParamList pl;
    pl.add_option("in-dir", "Standard punkst workflow output directory", inDir)
      .add_option("input-json", "Explicit deployment input JSON", inputJson)
      .add_option("config", "Workflow config JSON (default: <in-dir>/config.json)", configPath)
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
      .add_option("n-gene-bins", "Number of gene bins for transcript PMTiles", nGeneBins)
      .add_option("threads", "Number of threads", threads)
      .add_option("max-point-tile-bytes", "Maximum compressed bytes per point tile", maxPointTileBytes)
      .add_option("max-point-tile-features", "Maximum features per point tile", maxPointTileFeatures)
      .add_option("max-polygon-tile-bytes", "Maximum compressed bytes per polygon tile", maxPolygonTileBytes)
      .add_option("max-polygon-tile-features", "Maximum features per polygon tile", maxPolygonTileFeatures)
      .add_option("scale-factor-compression", "Pyramid compression aggressiveness estimate", compressionScale)
      .add_option("hex-prob-thres", "Minimum hex factor probability retained", hexProbThreshold)
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
    if (inDir.empty() == inputJson.empty()) {
        error("%s: provide exactly one of --in-dir or --input-json", __func__);
    }
    if (nGeneBins <= 0) {
        error("%s: --n-gene-bins must be positive", __func__);
    }

    fs::path outRoot(outDir);
    fs::path tmpDir = outRoot / "tmp";
    fs::create_directories(outRoot);
    fs::create_directories(tmpDir);

    DeployInputs inputs;
    if (!inputJson.empty()) {
        inputs = load_from_input_json(inputJson);
    } else {
        fs::path inRoot(inDir);
        fs::path cfg = configPath.empty() ? (inRoot / "config.json") : fs::path(configPath);
        inputs = load_from_config_dir(inRoot, cfg);
    }
    inputs = filter_models(std::move(inputs), modelPrefixes);
    validate_inputs(inputs);
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
    } else {
        std::vector<fs::path> presentPointOutputs = existing_files(planned_point_outputs(outRoot, nGeneBins));
        if (!overwrite && !presentPointOutputs.empty()) {
            error("%s: transcript PMTiles outputs are partially present; use --overwrite to regenerate them. Existing files: %s",
                __func__, join_paths(presentPointOutputs).c_str());
        }

        fs::path genesPrefix = outRoot / "genes";
        std::vector<std::string> tileArgs = {
            "tile-op",
            "--in", inputs.models.front().pixelPrefix.string(),
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
            "--out", genesPrefix.string(),
            "--threads", std::to_string(threads)
        };
        if (pmtilesFormat == "MVT") {
            tileArgs.push_back("--feature-field-name");
            tileArgs.push_back("gene");
        }
        tileArgs.push_back("--emb-prefix");
        for (const ModelInputs& model : inputs.models) {
            tileArgs.push_back(pixel_decode_id(model));
        }
        if (inputs.models.size() > 1) {
            tileArgs.push_back("--merge-emb");
            for (size_t i = 1; i < inputs.models.size(); ++i) {
                tileArgs.push_back(inputs.models[i].pixelPrefix.string() + ".bin");
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
        for (const std::string& pmtilesPath : pointPmtiles) {
            build_pyramid_inplace(pmtilesPath, tmpDir, true, pointMinZoom,
                maxPointTileBytes, maxPointTileFeatures, compressionScale, threads);
        }
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

    bool needsRasterBounds = false;
    for (const ModelInputs& model : inputs.models) {
        if (!model.pixelPng.empty()) {
            needsRasterBounds = true;
            break;
        }
    }
    mlt_pmtiles::RasterBounds rasterBounds;
    if (needsRasterBounds) {
        rasterBounds = resolve_raster_bounds(inputs.transcripts, inputs.models.front());
    }

    for (const ModelInputs& model : inputs.models) {
        fs::path hexPmtiles = outRoot / (model.id + ".pmtiles");
        if (!overwrite && file_exists(hexPmtiles)) {
            notice("%s: %s already exists; skipping hex PMTiles packaging", __func__, hexPmtiles.string().c_str());
        } else {
            fs::path resultsForPmtiles = prepare_hex_results_with_topk(model, tmpDir);
            std::vector<std::string> hexArgs = {
                "hex2pmtiles",
                "--in-tsv", resultsForPmtiles.string(),
                "--out", hexPmtiles.string(),
                "--format", pmtilesFormat,
                "--hex-grid-dist", fp_to_string(model.hexGridDist, 6),
                "--pmtiles-zoom", std::to_string(polygonMaxZoom),
                "--prob-thres", fp_to_string(hexProbThreshold, 8),
                "--threads", std::to_string(threads)
            };
            if (call_command(hexArgs, cmdHex2PmtilesMlt) != 0) {
                error("%s: hex2pmtiles failed for model %s", __func__, model.id.c_str());
            }
            build_pyramid_inplace(hexPmtiles, tmpDir, false, polygonMinZoom,
                maxPolygonTileBytes, maxPolygonTileFeatures, compressionScale, threads);
        }

        fs::path rasterPmtiles = outRoot / (model.id + "-pixel-raster.pmtiles");
        if (!model.pixelPng.empty()) {
            if (!overwrite && file_exists(rasterPmtiles)) {
                notice("%s: %s already exists; skipping pixel raster PMTiles", __func__,
                    rasterPmtiles.string().c_str());
            } else {
                mlt_pmtiles::write_png_raster_pmtiles_archive(model.pixelPng.string(),
                    rasterPmtiles.string(), (tmpDir / (model.id + ".raster.blob")).string(),
                    rasterBounds, polygonMinZoom, polygonMaxZoom);
            }
        }

        copy_file_checked(model.modelTsv, outRoot / (model.id + "-model.tsv"), overwrite, "model sidecar");
        write_cartoscope_rgb(model.colorRgbTsv, outRoot / (model.id + "-rgb.tsv"), overwrite);
        write_cartoscope_de(model.deTsv, outRoot / (model.id + "-bulk-de.tsv"), overwrite);
        if (!model.infoTsv.empty() && file_exists(model.infoTsv)) {
            copy_file_checked(model.infoTsv, outRoot / (model.id + "-info.tsv"), overwrite, "info sidecar");
        } else {
            write_text_checked(outRoot / (model.id + "-info.tsv"), "Factor\tWeight\n", overwrite, "info sidecar");
        }
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
        asset["pmtiles"] = {
            {"hex", model.id + ".pmtiles"},
            {"raw_pixel", rawPixelPmtilesName}
        };
        if (!model.pixelPng.empty()) {
            asset["pmtiles"]["raster"] = model.id + "-pixel-raster.pmtiles";
        }
        factorAssets.push_back(asset);
    }

    write_text_checked(outRoot / "ficture_assets.json", factorAssets.dump(4) + "\n",
        overwrite, "ficture assets");

    write_catalog_yaml(outRoot / "catalog.yaml", datasetId, title, desc, binPmtiles,
        inputs.models, rawPixelPmtilesName, overwrite);
    fs::remove_all(tmpDir);
    notice("%s: wrote minimal CartoScope deployment to %s", __func__, outRoot.string().c_str());
    return 0;
}
