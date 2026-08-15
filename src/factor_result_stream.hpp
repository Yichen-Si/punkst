#pragma once

#include "utils.h"

#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Dense>

namespace punkst {

// Streams dense transform results in lockstep with the original transform
// input. All columns preceding the trailing topic block form the unit key.
class FactorResultStream {
public:
    FactorResultStream(std::string path,
            const std::vector<std::string>& expected_topics)
        : path_(std::move(path)), input_(path_) {
        if (expected_topics.size() < 2) {
            throw std::invalid_argument(
                "Factor result stream requires at least two topics");
        }
        std::string line;
        while (input_.getline(line) && line.empty()) {}
        if (line.empty()) {
            throw std::runtime_error("Factor result table is empty: " + path_);
        }
        const std::vector<std::string> header = split_delimited(
            strip_leading_hash(line), '\t');
        if (header.size() <= expected_topics.size()) {
            throw std::runtime_error(
                "Factor result table has no unit metadata columns: " + path_);
        }
        header_size_ = header.size();
        factor_start_ = header.size() - expected_topics.size();
        for (size_t topic = 0; topic < expected_topics.size(); ++topic) {
            if (header[factor_start_ + topic] != expected_topics[topic]) {
                throw std::runtime_error(
                    "Factor result topics do not exactly match the model: "
                    + path_);
            }
        }
        topics_ = expected_topics.size();
    }

    Eigen::VectorXd next(const std::string& expected_key) {
        std::string line;
        while (input_.getline(line)) {
            ++line_number_;
            if (!line.empty() && !is_comment_line(line)) break;
            line.clear();
        }
        if (line.empty()) {
            throw std::runtime_error(
                "Factor result table ended before input unit " + expected_key);
        }
        const std::vector<std::string> fields = split_delimited(line, '\t');
        if (fields.size() != header_size_) {
            throw std::runtime_error(
                "Factor result row has the wrong column count at line "
                + std::to_string(line_number_));
        }
        std::string key = fields.front();
        for (size_t column = 1; column < factor_start_; ++column) {
            key += '\t';
            key += fields[column];
        }
        if (key != expected_key) {
            throw std::runtime_error(
                "Factor result/input unit mismatch at line "
                + std::to_string(line_number_) + ": expected '" + expected_key
                + "', found '" + key + "'");
        }
        Eigen::VectorXd composition(static_cast<Eigen::Index>(topics_));
        double total = 0.0;
        for (size_t topic = 0; topic < topics_; ++topic) {
            double value = 0.0;
            if (!str2double(fields[factor_start_ + topic], value)
                    || !(value >= 0.0) || !std::isfinite(value)) {
                throw std::runtime_error(
                    "Invalid factor value at line "
                    + std::to_string(line_number_));
            }
            composition(static_cast<Eigen::Index>(topic)) = value;
            total += value;
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::runtime_error(
                "Factor result row has no positive mass at line "
                + std::to_string(line_number_));
        }
        composition /= total;
        ++rows_;
        return composition;
    }

    Eigen::MatrixXd next_batch(const std::vector<std::string>& keys) {
        Eigen::MatrixXd output(keys.size(), topics_);
        for (size_t row = 0; row < keys.size(); ++row) {
            output.row(static_cast<Eigen::Index>(row)) = next(keys[row]);
        }
        return output;
    }

    void require_finished(bool allow_trailing_rows = false) {
        if (allow_trailing_rows) return;
        std::string line;
        while (input_.getline(line)) {
            ++line_number_;
            if (!line.empty() && !is_comment_line(line)) {
                throw std::runtime_error(
                    "Factor result table contains unmatched rows after line "
                    + std::to_string(line_number_ - 1));
            }
        }
    }

    uint64_t rows() const { return rows_; }

private:
    std::string path_;
    TextLineReader input_;
    size_t header_size_ = 0;
    size_t factor_start_ = 0;
    size_t topics_ = 0;
    uint64_t line_number_ = 1;
    uint64_t rows_ = 0;
};

} // namespace punkst
