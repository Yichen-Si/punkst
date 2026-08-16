#include "punkst.h"
#include "dataunits.hpp"
#include "utils.h"
#include "clustering_core/partition_classifier.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

using punkst::partition_classifier::Model;

class DenseThetaStream {
public:
    DenseThetaStream(std::string path, int32_t identifier_column,
            int32_t factor_start, int32_t factor_end,
            const std::vector<std::string>* expected_topics = nullptr)
        : path_(std::move(path)), identifier_column_(identifier_column),
          factor_start_(factor_start), factor_end_(factor_end) {
        read_header(expected_topics);
    }

    const std::vector<std::string>& topics() const { return topics_; }

    void validate_each(const std::function<void(uint64_t,
            const std::string&)>& callback) const {
        scan(true, [&](uint64_t row, const std::string& identifier,
                const std::vector<std::string>& fields, uint64_t line_number) {
            validate_composition(fields, line_number);
            callback(row, identifier);
        });
    }

    void for_each_selected(const std::function<bool(uint64_t,
            const std::string&)>& selected,
            const std::function<void(uint64_t, const std::string&,
                const Eigen::VectorXd&)>& callback) const {
        scan(false, [&](uint64_t row, const std::string& identifier,
                const std::vector<std::string>& fields, uint64_t line_number) {
            if (selected(row, identifier)) {
                callback(row, identifier,
                    parse_composition(fields, line_number));
            }
        });
    }

    void for_each(const std::function<void(uint64_t, const std::string&,
            const Eigen::VectorXd&)>& callback,
            bool validate_identifiers = true) const {
        scan(validate_identifiers, [&](uint64_t row,
                const std::string& identifier,
                const std::vector<std::string>& fields,
                uint64_t line_number) {
            callback(row, identifier,
                parse_composition(fields, line_number));
        });
    }

private:
    template<class Callback>
    void scan(bool validate_identifiers, Callback&& callback) const {
        TextLineReader input(path_);
        std::string line;
        while (input.getline(line) && line.empty()) {}
        uint64_t line_number = 1;
        uint64_t data_row = 0;
        std::unordered_set<std::string> identifiers;
        while (input.getline(line)) {
            ++line_number;
            if (line.empty() || is_comment_line(line)) continue;
            const std::vector<std::string> fields = split_delimited(line, '\t');
            if (fields.size() != header_size_) {
                throw std::runtime_error("Theta row has wrong column count at line "
                    + std::to_string(line_number));
            }
            const std::string& identifier = fields[identifier_column_];
            if (identifier.empty() || (validate_identifiers
                    && !identifiers.insert(identifier).second)) {
                throw std::runtime_error("Empty or duplicate theta identifier at line "
                    + std::to_string(line_number));
            }
            callback(data_row, identifier, fields, line_number);
            ++data_row;
        }
        if (data_row == 0) {
            throw std::runtime_error("Theta table has no data rows: " + path_);
        }
    }

    double validate_composition(const std::vector<std::string>& fields,
            uint64_t line_number, Eigen::VectorXd* composition = nullptr) const {
        if (composition != nullptr) {
            composition->resize(factor_columns_.size());
        }
        double total = 0.0;
        for (size_t topic = 0; topic < factor_columns_.size(); ++topic) {
            double value = 0.0;
            if (!str2double(fields[factor_columns_[topic]], value)
                    || !(value >= 0.0) || !std::isfinite(value)) {
                throw std::runtime_error("Invalid theta value at line "
                    + std::to_string(line_number));
            }
            if (composition != nullptr) (*composition)(topic) = value;
            total += value;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error("Theta row has no positive mass at line "
                + std::to_string(line_number));
        }
        return total;
    }

    Eigen::VectorXd parse_composition(
            const std::vector<std::string>& fields,
            uint64_t line_number) const {
        Eigen::VectorXd composition;
        const double total = validate_composition(
            fields, line_number, &composition);
        composition /= total;
        return composition;
    }

    void read_header(const std::vector<std::string>* expected_topics) {
        if (identifier_column_ < 0 || factor_start_ < -1 || factor_end_ < -1) {
            throw std::invalid_argument("Theta column indices must be nonnegative");
        }
        const bool explicit_factors = factor_start_ >= 0;
        if (explicit_factors != (factor_end_ >= 0)) {
            throw std::invalid_argument(
                "--icol-factor-start and --icol-factor-end must be supplied together");
        }
        if (explicit_factors && expected_topics != nullptr) {
            throw std::invalid_argument(
                "Explicit factor columns cannot be used when predicting");
        }
        TextLineReader input(path_);
        std::string line;
        while (input.getline(line) && line.empty()) {}
        if (line.empty()) throw std::runtime_error("Theta table is empty: " + path_);
        const std::vector<std::string> header = split_delimited(
            strip_leading_hash(line), '\t');
        header_size_ = header.size();
        if (identifier_column_ >= static_cast<int32_t>(header.size())) {
            throw std::invalid_argument("--theta-icol-id is outside theta table");
        }
        std::unordered_set<std::string> names;
        for (const std::string& name : header) {
            if (name.empty() || !names.insert(name).second) {
                throw std::runtime_error("Empty or duplicate theta header: " + name);
            }
        }
        if (explicit_factors) {
            if (factor_end_ < factor_start_ || factor_end_ >=
                    static_cast<int32_t>(header.size())) {
                throw std::invalid_argument("Invalid explicit factor range");
            }
            for (int32_t column = factor_start_; column <= factor_end_; ++column) {
                factor_columns_.push_back(column);
                topics_.push_back(header[column]);
            }
        } else if (expected_topics != nullptr) {
            if (expected_topics->size() < 2) {
                throw std::invalid_argument("Classifier model has too few topics");
            }
            std::unordered_map<std::string, int32_t> index;
            for (int32_t column = 0; column < static_cast<int32_t>(header.size());
                    ++column) index.emplace(header[column], column);
            int32_t previous = -1;
            for (const std::string& topic : *expected_topics) {
                const auto found = index.find(topic);
                if (found == index.end() || (previous >= 0
                        && found->second != previous + 1)) {
                    throw std::runtime_error(
                        "Theta topics do not exactly match classifier topic order");
                }
                previous = found->second;
                factor_columns_.push_back(found->second);
                topics_.push_back(topic);
            }
            if (previous != static_cast<int32_t>(header.size()) - 1) {
                throw std::runtime_error(
                    "Classifier topics must be the exact trailing theta block");
            }
        } else {
            int32_t first_factor = static_cast<int32_t>(header.size());
            while (first_factor > 0) {
                int32_t factor = -1;
                if (!str2int32(header[first_factor - 1], factor)
                        || factor < 0) break;
                --first_factor;
            }
            if (static_cast<int32_t>(header.size()) - first_factor < 2) {
                throw std::runtime_error(
                    "Theta table requires at least two trailing numeric factors");
            }
            std::unordered_set<int32_t> factor_names;
            for (int32_t column = first_factor;
                    column < static_cast<int32_t>(header.size()); ++column) {
                int32_t factor = -1;
                if (!str2int32(header[column], factor)
                        || factor < 0 || !factor_names.insert(factor).second) {
                    throw std::runtime_error(
                        "Theta table has duplicate numeric factor names");
                }
                factor_columns_.push_back(column);
                topics_.push_back(header[column]);
            }
        }
        if (factor_columns_.size() < 2 || std::find(factor_columns_.begin(),
                factor_columns_.end(), identifier_column_) != factor_columns_.end()) {
            throw std::invalid_argument(
                "Theta identifier must be outside at least two factor columns");
        }
    }

    std::string path_;
    int32_t identifier_column_ = 0;
    int32_t factor_start_ = -1;
    int32_t factor_end_ = -1;
    size_t header_size_ = 0;
    std::vector<int32_t> factor_columns_;
    std::vector<std::string> topics_;
};

struct PartitionLookup {
    std::unordered_map<std::string, int32_t> by_identifier;
    std::unordered_map<uint64_t, int32_t> by_row;
    std::vector<std::string> classes;
};

PartitionLookup read_partition(const std::string& path,
        int32_t identifier_column, int32_t partition_column,
        bool id_as_row_index) {
    if (identifier_column < 0 || partition_column < 0
            || identifier_column == partition_column) {
        throw std::invalid_argument(
            "Partition identifier and value columns must be distinct and nonnegative");
    }
    const int32_t maximum = std::max(identifier_column, partition_column);
    TextLineReader input(path);
    std::string line;
    uint64_t line_number = 0;
    std::unordered_map<std::string, int32_t> class_index;
    PartitionLookup output;
    while (input.getline(line)) {
        ++line_number;
        if (line.empty() || is_comment_line(line)) continue;
        const std::vector<std::string> fields = split_delimited(line, '\t');
        if (maximum >= static_cast<int32_t>(fields.size())) {
            throw std::runtime_error("Partition row has too few columns at line "
                + std::to_string(line_number));
        }
        const std::string& identifier = fields[identifier_column];
        const std::string& label = fields[partition_column];
        if (identifier.empty() || label.empty()) {
            throw std::runtime_error("Empty partition identifier or class at line "
                + std::to_string(line_number));
        }
        const auto inserted = class_index.emplace(label,
            static_cast<int32_t>(class_index.size()));
        if (inserted.second) output.classes.push_back(label);
        if (id_as_row_index) {
            uint64_t row = 0;
            if (!str2uint64(identifier, row)
                    || !output.by_row.emplace(row, inserted.first->second).second) {
                throw std::runtime_error(
                    "Invalid or duplicate partition row index at line "
                    + std::to_string(line_number));
            }
        } else if (!output.by_identifier.emplace(
                identifier, inserted.first->second).second) {
            throw std::runtime_error("Duplicate partition identifier at line "
                + std::to_string(line_number));
        }
    }
    if (output.classes.empty()) {
        throw std::runtime_error("Partition table has no data rows: " + path);
    }
    return output;
}

int32_t lookup_class(const PartitionLookup& partition, bool row_index,
        uint64_t row, const std::string& identifier) {
    if (row_index) {
        const auto found = partition.by_row.find(row);
        return found == partition.by_row.end() ? -1 : found->second;
    }
    const auto found = partition.by_identifier.find(identifier);
    return found == partition.by_identifier.end() ? -1 : found->second;
}

uint64_t hash_identifier(const std::string& identifier, uint64_t seed) {
    uint64_t value = 1469598103934665603ULL ^ seed;
    for (const unsigned char byte : identifier) {
        value ^= byte;
        value *= 1099511628211ULL;
    }
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

std::vector<uint64_t> allocate_quotas(const std::vector<uint64_t>& counts,
        uint64_t maximum, int32_t minimum) {
    const uint64_t total = std::accumulate(counts.begin(), counts.end(),
        uint64_t{0});
    if (maximum == 0 || total <= maximum) return counts;
    std::vector<uint64_t> quota(counts.size());
    uint64_t allocated = 0;
    for (size_t component = 0; component < counts.size(); ++component) {
        quota[component] = std::min<uint64_t>(counts[component], minimum);
        allocated += quota[component];
    }
    if (allocated > maximum) {
        throw std::invalid_argument(
            "--train-max-rows is too small for --min-per-class");
    }
    const uint64_t available = maximum - allocated;
    const uint64_t remaining_total = total - allocated;
    std::vector<std::pair<double, size_t>> remainders;
    uint64_t extra_allocated = 0;
    for (size_t component = 0; component < counts.size(); ++component) {
        const uint64_t capacity = counts[component] - quota[component];
        const double exact = remaining_total > 0
            ? static_cast<double>(available) * capacity / remaining_total : 0.0;
        const uint64_t extra = std::min<uint64_t>(capacity,
            static_cast<uint64_t>(std::floor(exact)));
        quota[component] += extra;
        extra_allocated += extra;
        remainders.emplace_back(exact - std::floor(exact), component);
    }
    std::sort(remainders.begin(), remainders.end(),
        [](const auto& left, const auto& right) {
            return left.first == right.first ? left.second < right.second
                : left.first > right.first;
        });
    for (const auto& remainder : remainders) {
        if (extra_allocated >= available) break;
        const size_t component = remainder.second;
        if (quota[component] < counts[component]) {
            ++quota[component];
            ++extra_allocated;
        }
    }
    return quota;
}

struct SampleRow {
    uint64_t hash = 0;
    uint64_t row = 0;
    std::string identifier;
    Eigen::VectorXd composition;
    int32_t label = -1;

    bool operator<(const SampleRow& other) const {
        if (hash != other.hash) return hash < other.hash;
        if (identifier != other.identifier) return identifier < other.identifier;
        return row < other.row;
    }
};

void write_cv(const std::string& path,
        const std::vector<punkst::partition_classifier::CvResult>& cv) {
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write classifier CV: " + path);
    output << "#ridge\tlog_loss\tbrier\taccuracy\tselected\n"
        << std::scientific << std::setprecision(10);
    for (const auto& row : cv) {
        output << row.ridge << '\t' << row.metrics.log_loss << '\t'
            << row.metrics.brier << '\t' << row.metrics.accuracy << '\t'
            << (row.selected ? 1 : 0) << '\n';
    }
}

void write_calibration(const std::string& path,
        const punkst::partition_classifier::CalibrationResult& calibration,
        const std::vector<std::string>& classes) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error("Cannot write classifier calibration: " + path);
    }
    output << "#kind\tclass\tbin\tlower\tupper\tweight\tlog_loss"
        "\tbrier\taccuracy\tmean_confidence\n"
        << std::scientific << std::setprecision(10);
    const auto& overall = calibration.cross_fitted;
    output << "overall\t.\t.\t.\t.\t" << overall.weight << '\t'
        << overall.log_loss << '\t' << overall.brier << '\t'
        << overall.accuracy << "\t.\n";
    for (size_t component = 0; component < classes.size(); ++component) {
        const auto& metric = calibration.classwise[component];
        output << "class\t" << classes[component] << "\t.\t.\t.\t"
            << metric.weight << '\t' << metric.log_loss << '\t'
            << metric.brier << '\t' << metric.accuracy << "\t.\n";
    }
    for (const auto& bin : calibration.bins) {
        output << "reliability\t.\t" << bin.bin << '\t' << bin.lower
            << '\t' << bin.upper << '\t' << bin.weight
            << "\t.\t.\t" << bin.accuracy << '\t'
            << bin.mean_confidence << '\n';
    }
}

void write_crossfit_diagnostics(const std::string& path,
        const punkst::partition_classifier::CrossfitResult& result) {
    std::ofstream output(path);
    if (!output) {
        throw std::runtime_error(
            "Cannot write classifier crossfit diagnostics: " + path);
    }
    output << "#kind\tfold\ttraining_rows\theldout_rows\tinner_folds"
        "\tridge\ttemperature\tweight\tlog_loss\tbrier\taccuracy\n"
        << std::scientific << std::setprecision(10);
    for (const auto& fold : result.diagnostics) {
        output << "fold\t" << fold.fold << '\t' << fold.training_rows << '\t'
            << fold.heldout_rows << '\t' << fold.inner_folds << '\t'
            << fold.ridge << '\t' << fold.temperature << '\t'
            << fold.metrics.weight << '\t' << fold.metrics.log_loss << '\t'
            << fold.metrics.brier << '\t' << fold.metrics.accuracy << '\n';
    }
    output << "overall\t.\t.\t.\t.\t.\t.\t" << result.overall.weight
        << '\t' << result.overall.log_loss << '\t' << result.overall.brier
        << '\t' << result.overall.accuracy << '\n';
}

void write_prediction_header(std::ofstream& output, int32_t classes,
        int32_t top_k, bool dense) {
    output << "#id";
    if (dense) {
        for (int32_t component = 0; component < classes; ++component) {
            output << "\tP" << component;
        }
    } else {
        for (int32_t rank = 1; rank <= std::min(top_k, classes); ++rank) {
            output << "\tC" << rank << "\tP" << rank;
        }
    }
    output << '\n';
}

void write_prediction_row(std::ofstream& output, const std::string& identifier,
        const Eigen::VectorXd& probability,
        const std::vector<std::string>& classes, int32_t top_k, bool dense,
        std::vector<int32_t>& order) {
    output << identifier;
    if (dense) {
        for (const double value : probability) output << '\t' << value;
    } else {
        order.resize(static_cast<size_t>(probability.size()));
        std::iota(order.begin(), order.end(), 0);
        const int32_t take = std::min<int32_t>(top_k, order.size());
        std::partial_sort(order.begin(), order.begin() + take, order.end(),
            [&](int32_t left, int32_t right) {
                return probability(left) == probability(right)
                    ? left < right : probability(left) > probability(right);
            });
        for (int32_t rank = 0; rank < take; ++rank) {
            output << '\t' << classes[order[rank]] << '\t'
                << probability(order[rank]);
        }
    }
    output << '\n';
}

void write_discordant_header(std::ofstream& output) {
    output << "#id\tC0\tP0\tC1\tP1\n"
        << std::scientific << std::setprecision(6);
}

void write_discordant_row(std::ofstream& output,
        const std::string& identifier, int32_t input_class,
        const Eigen::VectorXd& probability,
        const std::vector<std::string>& classes) {
    Eigen::Index predicted = 0;
    const double predicted_probability = probability.maxCoeff(&predicted);
    if (input_class < 0 || input_class == predicted) return;
    output << identifier << '\t' << classes[input_class] << '\t'
        << probability(input_class) << '\t' << classes[predicted] << '\t'
        << predicted_probability << '\n';
}

void write_predictions(const DenseThetaStream& theta, const Model& model,
        const std::string& path, int32_t top_k, bool dense) {
    if (top_k <= 0) throw std::invalid_argument("--top-k must be positive");
    std::ofstream output(path);
    if (!output) throw std::runtime_error("Cannot write predictions: " + path);
    write_prediction_header(output, model.classes.size(), top_k, dense);
    output << std::scientific << std::setprecision(6);
    std::vector<int32_t> order;
    theta.for_each([&](uint64_t, const std::string& identifier,
            const Eigen::VectorXd& composition) {
        const Eigen::VectorXd probability = model.probabilities(composition);
        write_prediction_row(output, identifier, probability,
            model.classes, top_k, dense, order);
    });
}

void write_fit_predictions(const DenseThetaStream& theta,
        const Model& full_model,
        const punkst::partition_classifier::CrossfitBundle* bundle,
        const PartitionLookup& partition, bool id_as_row_index,
        const std::vector<int32_t>& class_remap,
        const std::string& prefix, int32_t top_k, bool dense) {
    if (top_k <= 0) throw std::invalid_argument("--top-k must be positive");
    std::ofstream full_output(prefix + ".classifications.tsv");
    std::ofstream full_discordant(prefix + ".discordant.tsv");
    if (!full_output || !full_discordant) {
        throw std::runtime_error("Cannot write classifier predictions under: "
            + prefix);
    }
    write_prediction_header(full_output, full_model.classes.size(), top_k, dense);
    write_discordant_header(full_discordant);
    full_output << std::scientific << std::setprecision(6);

    std::ofstream crossfit_output;
    std::ofstream crossfit_discordant;
    if (bundle != nullptr) {
        crossfit_output.open(prefix + ".crossfit.classifications.tsv");
        crossfit_discordant.open(prefix + ".crossfit.discordant.tsv");
        if (!crossfit_output || !crossfit_discordant) {
            throw std::runtime_error(
                "Cannot write classifier crossfit predictions under: " + prefix);
        }
        write_prediction_header(crossfit_output,
            bundle->full_model.classes.size(), top_k, dense);
        write_discordant_header(crossfit_discordant);
        crossfit_output << std::scientific << std::setprecision(6);
    }

    std::vector<int32_t> full_order;
    std::vector<int32_t> crossfit_order;
    theta.for_each([&](uint64_t row, const std::string& identifier,
            const Eigen::VectorXd& composition) {
        const int32_t original_class = lookup_class(partition,
            id_as_row_index, row, identifier);
        const int32_t input_class = original_class < 0
            ? -1 : class_remap[static_cast<size_t>(original_class)];
        const Eigen::VectorXd full_probability =
            full_model.probabilities(composition);
        write_prediction_row(full_output, identifier, full_probability,
            full_model.classes, top_k, dense, full_order);
        write_discordant_row(full_discordant, identifier, input_class,
            full_probability, full_model.classes);

        if (bundle != nullptr) {
            const Model& crossfit_model = bundle->model_for(identifier, true);
            const Eigen::VectorXd crossfit_probability =
                crossfit_model.probabilities(composition);
            write_prediction_row(crossfit_output, identifier,
                crossfit_probability, crossfit_model.classes,
                top_k, dense, crossfit_order);
            write_discordant_row(crossfit_discordant, identifier, input_class,
                crossfit_probability, crossfit_model.classes);
        }
    }, false);
}

} // namespace

int32_t cmdPartitionClassifierFit(int argc, char** argv) {
    std::string theta_path, partition_path, output_prefix;
    int32_t theta_identifier_column = 0;
    int32_t partition_identifier_column = 0;
    int32_t partition_column = 1;
    int32_t factor_start = -1, factor_end = -1;
    int32_t train_max_rows = 100000;
    int32_t minimum_per_class = 100;
    int32_t folds = 5;
    int32_t top_k = 3;
    int32_t max_iterations = 300;
    int32_t lbfgs_history = 10;
    int32_t seed = 1;
    double gradient_tolerance = 1e-7;
    std::vector<double> ridge_grid;
    bool id_as_row_index = false;
    bool dense = false;
    bool crossfit = false;

    ParamList parameters;
    parameters
      .add_option("in-theta", "Dense Gamma-Poisson/LDA theta table",
          theta_path, true)
      .add_option("in-partition", "Hard-partition TSV", partition_path, true)
      .add_option("out-prefix", "Output prefix", output_prefix, true)
      .add_option("theta-icol-id", "0-based theta identifier column",
          theta_identifier_column)
      .add_option("icol-factor-start", "0-based first factor column",
          factor_start)
      .add_option("icol-factor-end", "0-based last factor column", factor_end)
      .add_option("icol-id", "0-based partition identifier column",
          partition_identifier_column)
      .add_option("icol-partition", "0-based partition value column",
          partition_column)
      .add_option("id-as-row-index", "Interpret partition IDs as theta rows",
          id_as_row_index)
      .add_option("train-max-rows", "Maximum deterministic training sample",
          train_max_rows)
      .add_option("min-per-class", "Minimum sample allocation per class",
          minimum_per_class)
      .add_option("folds", "Maximum stratified cross-validation folds", folds)
      .add_option("ridge-grid", "Ridge candidates", ridge_grid)
      .add_option("max-iterations", "Maximum L-BFGS iterations", max_iterations)
      .add_option("lbfgs-history", "L-BFGS correction history", lbfgs_history)
      .add_option("gradient-tolerance", "L-BFGS infinity-norm tolerance",
          gradient_tolerance)
      .add_option("sampling-seed", "Deterministic sampling seed", seed)
      .add_option("top-k", "Number of prediction class/probability pairs", top_k)
      .add_option("dense-probabilities", "Write P0..P(C-1)", dense)
      .add_option("crossfit",
          "Also fit a nested crossfit bundle for overlapping transforms",
          crossfit);
    try {
        parameters.readArgs(argc, argv);
        if (train_max_rows < 0 || minimum_per_class < 2 || folds < 2
                || seed < 0) {
            throw std::invalid_argument("Invalid classifier sampling or fold option");
        }
        DenseThetaStream theta(theta_path, theta_identifier_column,
            factor_start, factor_end);
        PartitionLookup partition = read_partition(partition_path,
            partition_identifier_column, partition_column, id_as_row_index);
        std::vector<uint64_t> original_counts(partition.classes.size(), 0);
        uint64_t matched_rows = 0;
        theta.validate_each([&](uint64_t row, const std::string& identifier) {
            const int32_t label = lookup_class(partition, id_as_row_index,
                row, identifier);
            if (label >= 0) {
                ++original_counts[label];
                ++matched_rows;
            }
        });
        std::vector<int32_t> remap(partition.classes.size(), -1);
        std::vector<std::string> classes;
        std::vector<uint64_t> counts;
        for (size_t component = 0; component < partition.classes.size();
                ++component) {
            if (original_counts[component] == 0) continue;
            if (original_counts[component] < 2) {
                throw std::runtime_error("Classifier requires at least two matched rows for class: "
                    + partition.classes[component]);
            }
            remap[component] = classes.size();
            classes.push_back(partition.classes[component]);
            counts.push_back(original_counts[component]);
        }
        if (classes.size() < 2) {
            throw std::runtime_error("Classifier intersection has fewer than two classes");
        }
        const std::vector<uint64_t> quotas = allocate_quotas(counts,
            train_max_rows, minimum_per_class);
        std::vector<std::priority_queue<SampleRow>> heaps(classes.size());
        int32_t pending_label = -1;
        uint64_t pending_hash = 0;
        theta.for_each_selected([&](uint64_t row,
                const std::string& identifier) {
            const int32_t original = lookup_class(partition, id_as_row_index,
                row, identifier);
            if (original < 0 || remap[original] < 0) return false;
            pending_label = remap[original];
            pending_hash = hash_identifier(identifier,
                static_cast<uint64_t>(seed));
            const SampleRow candidate{
                pending_hash, row, identifier, Eigen::VectorXd(), pending_label};
            const auto& heap = heaps[pending_label];
            return heap.size() < quotas[pending_label]
                || candidate < heap.top();
        }, [&](uint64_t row, const std::string& identifier,
                const Eigen::VectorXd& composition) {
            SampleRow sample{pending_hash, row, identifier,
                composition, pending_label};
            auto& heap = heaps[pending_label];
            if (heap.size() < quotas[pending_label]) {
                heap.push(std::move(sample));
            } else {
                heap.pop();
                heap.push(std::move(sample));
            }
        });
        std::vector<SampleRow> sample;
        for (auto& heap : heaps) {
            while (!heap.empty()) {
                sample.push_back(heap.top());
                heap.pop();
            }
        }
        std::sort(sample.begin(), sample.end(), [](const SampleRow& left,
                const SampleRow& right) { return left.row < right.row; });
        RowMajorMatrixXd compositions(sample.size(), theta.topics().size());
        Eigen::VectorXi labels(sample.size());
        Eigen::VectorXd weights(sample.size());
        std::vector<std::string> sample_identifiers(sample.size());
        double mean_weight = 0.0;
        for (size_t index = 0; index < sample.size(); ++index) {
            compositions.row(index) = sample[index].composition.transpose();
            labels(index) = sample[index].label;
            weights(index) = static_cast<double>(counts[labels(index)])
                / quotas[labels(index)];
            sample_identifiers[index] = sample[index].identifier;
            mean_weight += weights(index);
        }
        mean_weight /= weights.size();
        weights /= mean_weight;

        punkst::partition_classifier::FitOptions options;
        if (!ridge_grid.empty()) options.ridge_grid = ridge_grid;
        options.folds = folds;
        options.max_iterations = max_iterations;
        options.lbfgs_history = lbfgs_history;
        options.gradient_tolerance = gradient_tolerance;
        options.progress_callback = [](const std::string& message) {
            notice("%s", message.c_str());
        };
        auto fitted = punkst::partition_classifier::fit(compositions, labels,
            weights, sample_identifiers, theta.topics(), classes,
            static_cast<uint64_t>(seed), options);
        fitted.model.matched_rows = matched_rows;
        fitted.model.sampled_rows = sample.size();
        fitted.model.minimum_per_class = minimum_per_class;
        fitted.model.sampling_seed = seed;
        fitted.model.write(output_prefix + ".classifier.tsv");
        write_cv(output_prefix + ".cv.tsv", fitted.cv);
        write_calibration(output_prefix + ".calibration.tsv",
            fitted.calibration, fitted.model.classes);
        std::unique_ptr<punkst::partition_classifier::CrossfitBundle> bundle;
        if (crossfit) {
            auto crossfitted = punkst::partition_classifier::fit_crossfit(
                compositions, labels, weights, sample_identifiers,
                theta.topics(), classes, static_cast<uint64_t>(seed), options);
            bundle = std::make_unique<
                punkst::partition_classifier::CrossfitBundle>();
            bundle->full_model = fitted.model;
            bundle->fold_models = crossfitted.fold_models;
            for (size_t index = 0; index < sample_identifiers.size(); ++index) {
                bundle->heldout_fold_by_identifier.emplace(
                    sample_identifiers[index],
                    crossfitted.fold_by_row(static_cast<Eigen::Index>(index)));
            }
            bundle->write(output_prefix + ".crossfit.classifier.tsv");
            write_crossfit_diagnostics(
                output_prefix + ".crossfit.diagnostics.tsv", crossfitted);
            notice("Nested crossfit bundle and diagnostics written under %s.crossfit",
                output_prefix.c_str());
        }
        write_fit_predictions(theta, fitted.model, bundle.get(), partition,
            id_as_row_index, remap, output_prefix, top_k, dense);
        notice("Partition classifier fit %zu classes from %zu/%zu matched rows",
            classes.size(), sample.size(), matched_rows);
    } catch (const std::exception& exception) {
        std::cerr << "Partition classifier fit failed: "
            << exception.what() << '\n';
        return 1;
    }
    return 0;
}

int32_t cmdPartitionClassifierPredict(int argc, char** argv) {
    std::string theta_path, model_path, output_prefix;
    int32_t theta_identifier_column = 0;
    int32_t top_k = 3;
    bool dense = false;
    ParamList parameters;
    parameters
      .add_option("in-theta", "Dense Gamma-Poisson/LDA theta table",
          theta_path, true)
      .add_option("in-model", "Partition classifier model", model_path, true)
      .add_option("out-prefix", "Output prefix", output_prefix, true)
      .add_option("theta-icol-id", "0-based theta identifier column",
          theta_identifier_column)
      .add_option("top-k", "Number of prediction class/probability pairs", top_k)
      .add_option("dense-probabilities", "Write P0..P(C-1)", dense);
    try {
        parameters.readArgs(argc, argv);
        const Model model = Model::read(model_path);
        DenseThetaStream theta(theta_path, theta_identifier_column,
            -1, -1, &model.topics);
        write_predictions(theta, model, output_prefix + ".results.tsv",
            top_k, dense);
        notice("Partition classifier predictions written to %s.results.tsv",
            output_prefix.c_str());
    } catch (const std::exception& exception) {
        std::cerr << "Partition classifier prediction failed: "
            << exception.what() << '\n';
        return 1;
    }
    return 0;
}
