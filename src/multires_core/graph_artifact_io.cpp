#include "multires_core/graph_artifact_io.hpp"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace punkst::multires {
namespace {

void collect_array_specs(const json& value, std::map<std::string, json>& out) {
    if (value.is_object()) {
        static const std::vector<std::string> required = {
            "path", "dtype", "endianness", "order", "shape"};
        const bool is_spec = std::all_of(required.begin(), required.end(),
            [&](const std::string& name) { return value.contains(name); });
        if (is_spec) {
            const std::string path = value.at("path").get<std::string>();
            const auto inserted = out.emplace(path, value);
            if (!inserted.second && inserted.first->second != value) {
                throw std::runtime_error(
                    "Conflicting array specifications for " + path);
            }
            return;
        }
        for (const auto& item : value.items()) {
            collect_array_specs(item.value(), out);
        }
    } else if (value.is_array()) {
        for (const json& child : value) collect_array_specs(child, out);
    }
}

RawGraphArtifactData load_raw_graph(
        const fs::path& root, const json& specification) {
    RawGraphArtifactData out;
    out.graph.n_nodes = specification.at("nodes").get<int32_t>();
    const std::vector<int32_t> rows = read_int32_array(
        root, specification.at("edge_rows"));
    const std::vector<int32_t> columns = read_int32_array(
        root, specification.at("edge_columns"));
    out.graph.weights = read_float64_array(
        root, specification.at("edge_weights"));
    out.graph.component_labels = read_int32_array(
        root, specification.at("component_labels"));
    const std::vector<int32_t> counts = read_int32_array(
        root, specification.at("fine_point_counts"));
    if (rows.size() != columns.size()
            || rows.size() != out.graph.weights.size()
            || out.graph.component_labels.size()
                != static_cast<size_t>(out.graph.n_nodes)
            || counts.size() != static_cast<size_t>(out.graph.n_nodes)) {
        throw std::runtime_error("Raw graph arrays do not align");
    }
    out.graph.edges.reserve(rows.size());
    for (size_t index = 0; index < rows.size(); ++index) {
        out.graph.edges.emplace_back(rows[index], columns[index]);
    }
    out.graph.self_loop_edges = specification.value("self_loop_edges", 0);
    out.graph.total_affinity = specification.value("total_affinity", 0.0);
    out.fine_point_counts.assign(counts.begin(), counts.end());
    if (std::any_of(out.fine_point_counts.begin(),
            out.fine_point_counts.end(), [](int64_t value) {
                return value <= 0;
            })) {
        throw std::runtime_error("Fine-point counts must be positive");
    }
    return out;
}

std::vector<std::string> load_identifiers(
        const fs::path& path, int32_t expected) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("Cannot read identifiers: " + path.string());
    }
    std::string line;
    if (!std::getline(input, line) || line != "row\tid") {
        throw std::runtime_error("Identifier table has an invalid header");
    }
    std::vector<std::string> identifiers;
    while (std::getline(input, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        const size_t tab = line.find('\t');
        if (tab == std::string::npos) {
            throw std::runtime_error("Malformed identifier table row");
        }
        const int64_t row = std::stoll(line.substr(0, tab));
        if (row != static_cast<int64_t>(identifiers.size())) {
            throw std::runtime_error("Identifier rows are not consecutive");
        }
        identifiers.push_back(line.substr(tab + 1));
    }
    if (identifiers.size() != static_cast<size_t>(expected)) {
        throw std::runtime_error("Identifier count does not match graph");
    }
    return identifiers;
}

} // namespace

fs::path resolve_artifact_manifest(const fs::path& input) {
    const fs::path absolute = fs::absolute(input).lexically_normal();
    return fs::is_directory(absolute) ? absolute / "manifest.json" : absolute;
}

json load_verified_artifact(
        const fs::path& input, const std::string& expected_type) {
    const fs::path path = resolve_artifact_manifest(input);
    json manifest = read_json(path);
    if (manifest.value("artifact_type", "") != expected_type
            || manifest.value("schema_version", 0) != 1) {
        throw std::runtime_error(
            "Unexpected artifact type or schema: " + path.string());
    }
    const std::string declared = manifest.value("fingerprint", "");
    if (declared.size() != 64) {
        throw std::runtime_error(
            "Artifact fingerprint is missing or malformed: " + path.string());
    }
    json unsigned_manifest = manifest;
    unsigned_manifest.erase("fingerprint");
    std::map<std::string, json> specifications;
    collect_array_specs(unsigned_manifest, specifications);
    std::vector<json> arrays;
    arrays.reserve(specifications.size());
    for (const auto& item : specifications) arrays.push_back(item.second);
    const std::string actual = artifact_fingerprint(
        unsigned_manifest, path.parent_path(), arrays);
    if (actual != declared) {
        throw std::runtime_error(
            "Artifact fingerprint does not match contents: " + path.string());
    }
    return manifest;
}

KnnGraphArtifactData load_knn_graph_artifact(const fs::path& input) {
    KnnGraphArtifactData out;
    out.manifest_path = resolve_artifact_manifest(input);
    out.root = out.manifest_path.parent_path();
    out.manifest = load_verified_artifact(input, "punkst.knn_graph");
    out.fingerprint = out.manifest.at("fingerprint").get<std::string>();
    const int32_t points = out.manifest.at("population").at("points")
        .get<int32_t>();
    const std::string identifiers_file = out.manifest.at("population")
        .at("identifiers_table").get<std::string>();
    if (sha256_file(out.root / identifiers_file)
            != out.manifest.at("population").at("identifiers_sha256")
                .get<std::string>()) {
        throw std::runtime_error("Identifier table checksum does not match");
    }
    out.identifiers = load_identifiers(out.root / identifiers_file, points);
    out.fine = load_raw_graph(out.root, out.manifest.at("raw_graph"));
    if (out.fine.graph.n_nodes != points) {
        throw std::runtime_error("Fine graph point count is inconsistent");
    }
    const json& coarsening = out.manifest.at("coarsening");
    if (coarsening.is_object()) {
        out.coarse = load_raw_graph(out.root, coarsening.at("raw_graph"));
        out.fine_to_microcluster = read_int32_array(
            out.root, coarsening.at("membership"));
        out.representative_rows = read_int32_array(
            out.root, coarsening.at("representative_rows"));
        if (out.fine_to_microcluster.size() != static_cast<size_t>(points)
                || out.representative_rows.size()
                    != static_cast<size_t>(out.coarse->graph.n_nodes)) {
            throw std::runtime_error("Coarsening arrays do not align");
        }
    }
    return out;
}

} // namespace punkst::multires
