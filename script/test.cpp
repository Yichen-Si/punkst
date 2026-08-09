#include "numerical_utils.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

int32_t cmdUacFit(int argc, char** argv);
int32_t cmdUacTransform(int argc, char** argv);

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error("UAC test failed: " + message);
    }
}

void write_text(
        const std::filesystem::path& path, const std::string& text) {
    std::ofstream output(path);
    require(static_cast<bool>(output),
        "cannot write " + path.string());
    output << text;
}

int32_t run_command(int32_t (*command)(int, char**),
        std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) {
        argv.push_back(argument.data());
    }
    return command(static_cast<int32_t>(argv.size()), argv.data());
}

void remove_uac_outputs(const std::filesystem::path& prefix) {
    for (const std::string& suffix : {
            ".state.tsv", ".model.tsv", ".results.tsv",
            ".diagnostics.tsv", ".trace.tsv", ".separation.tsv",
            ".representatives.tsv"}) {
        std::filesystem::remove(prefix.string() + suffix);
    }
}

void test_topic_to_uac_handoff() {
    const std::filesystem::path base =
        std::filesystem::temp_directory_path()
        / "punkst_topic_to_uac";
    const std::filesystem::path model_path =
        base.string() + ".topic.model.tsv";
    const std::filesystem::path center_path =
        base.string() + ".topic.results.tsv";
    const std::filesystem::path input_path =
        base.string() + ".units.tsv";
    const std::filesystem::path metadata_path =
        base.string() + ".meta.json";
    const std::filesystem::path fit_prefix =
        base.string() + ".fit";
    const std::filesystem::path transform_prefix =
        base.string() + ".transform";

    write_text(model_path,
        "Feature\tTopic0\tTopic1\n"
        "feature_0\t20\t1\n"
        "feature_1\t1\t20\n"
        "feature_2\t9\t9\n");
    write_text(metadata_path,
        "{\"n_units\":12,\"n_modalities\":1,\"n_features\":2,"
        "\"offset_data\":2,\"header_info\":[\"batch\",\"document\"],"
        "\"dictionary\":{\"feature_0\":0,\"feature_1\":1}}");
    std::ostringstream centers, counts;
    centers << "#batch\tdocument\tTopic1\tTopic0\n";
    for (int32_t document = 0; document < 12; ++document) {
        const bool first = document < 6;
        centers << "batch_" << (document % 2) << "\tdoc_" << document
            << "\t" << (first ? 0.05 : 0.95)
            << "\t" << (first ? 0.95 : 0.05) << "\n";
        counts << "batch_" << (document % 2) << "\tdoc_" << document
            << "\t2\t10\t0 " << (first ? 9 : 1)
            << "\t1 " << (first ? 1 : 9) << "\n";
    }
    write_text(center_path, centers.str());
    write_text(input_path, counts.str());

    std::vector<std::string> fit_arguments{
        "uac-fit",
        "--in-topic-center", center_path.string(),
        "--unit-icol-id", "1",
        "--in-model", model_path.string(),
        "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(),
        "--count-icol-id", "1",
        "--out-prefix", fit_prefix.string(),
        "--n-clusters", "2",
        "--particles", "16",
        "--particle-em-fixed-iterations", "1",
        "--kmeans-starts", "1",
        "--max-iter", "2",
        "--cluster-covariance-rank", "0",
        "--threads", "1",
        "--n-representatives", "1",
        "--seed", "43",
    };
    require(run_command(cmdUacFit, std::move(fit_arguments)) == 0,
        "direct topic-to-UAC fit failed");

    std::ifstream state_input(fit_prefix.string() + ".state.tsv");
    std::string state_header;
    require(static_cast<bool>(std::getline(state_input, state_header))
            && state_header == "##punkst_uac_state_v11",
        "direct topic-to-UAC fit did not write a v11 state");

    std::vector<std::string> transform_arguments{
        "uac-transform",
        "--in-state", fit_prefix.string() + ".state.tsv",
        "--in-topic-center", center_path.string(),
        "--unit-icol-id", "1",
        "--in-model", model_path.string(),
        "--in-data", input_path.string(),
        "--in-meta", metadata_path.string(),
        "--count-icol-id", "1",
        "--out-prefix", transform_prefix.string(),
        "--particles", "16",
        "--threads", "1",
        "--n-representatives", "1",
    };
    require(run_command(cmdUacTransform,
            std::move(transform_arguments)) == 0,
        "direct topic-to-UAC transform failed");

    std::ifstream result_input(
        transform_prefix.string() + ".results.tsv");
    std::string line;
    int32_t rows = -1;
    while (std::getline(result_input, line)) ++rows;
    require(rows == 12,
        "direct topic-to-UAC transform wrote the wrong row count");

    remove_uac_outputs(fit_prefix);
    remove_uac_outputs(transform_prefix);
    std::filesystem::remove(model_path);
    std::filesystem::remove(center_path);
    std::filesystem::remove(input_path);
    std::filesystem::remove(metadata_path);
}

} // namespace

int32_t test(int32_t, char**) {
    try {
        test_topic_to_uac_handoff();
        std::cout << "UAC tests passed\n";
    } catch (const std::exception& exception) {
        std::cerr << exception.what() << "\n";
        return 1;
    }
    return 0;
}
