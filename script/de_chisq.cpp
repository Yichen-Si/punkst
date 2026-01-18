#include "punkst.h"
#include "utils.h"
#include "utils_sys.hpp"
#include "dataunits.hpp"
#include "numerical_utils.hpp"
#include "commands.hpp"

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>
#include <map>
#include <set>
#include <iomanip>
#include "Eigen/Dense"

#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/concurrent_vector.h>
#include <tbb/global_control.h>
#include <tbb/parallel_sort.h>

struct DeResult {
    int j, k;
    double chi2;
    double log10pval; // -log10(p)
    double fc;
    DeResult(int j_, int k_,  double chi2_, double log10pval_, double fc_)
      : j(j_), k(k_), chi2(chi2_), log10pval(log10pval_), fc(fc_) {}
};

int32_t cmdDeChisq(int argc, char** argv) {
    std::vector<std::string> inputFiles;
    std::string outputFile;
    double minCount = 10.0;
    double minCountPerFeature = 100.0;
    double minPval = 1e-3;
    double minFC = 1.5;
    int nThreads = 1;
    double pseudoCount = 0.5;
    int neighborK = 0;

    ParamList pl;
    pl.add_option("input", "Input pseudobulk matrix (one for 1-vs-rest, two for pairwise comparison)", inputFiles, true)
      .add_option("out", "Output file", outputFile, true)
      .add_option("min-count-per-feature", "Minimum total count for a feature to be considered", minCountPerFeature)
      .add_option("max-pval", "Max p-value for output (default: 1e-3)", minPval)
      .add_option("min-fc", "Min fold change for output (default: 1.5)", minFC)
      .add_option("min-count", "Minimum observed count for a (feature, factor) pair to be tested (default: 10)", minCount)
      .add_option("pseudocount", "Pseudocount to add to counts for Chi2 test (default: 0.5)", pseudoCount)
      .add_option("threads", "Number of threads", nThreads)
      .add_option("neighbor-k", "Number of nearest neighbor columns for background (single input only)", neighborK);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }
    double maxLog10Pval = -std::log10(minPval);

    if (nThreads < 1) nThreads = 1;
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nThreads);

    if (inputFiles.empty() || inputFiles.size() > 2) {
        error("One or two input files must be provided.");
    }
    if (neighborK < 0) {
        error("--neighbor-k must be non-negative");
    }

    // Two-sample comparison
    if (inputFiles.size() == 2) {
        if (neighborK > 0) {
            warning("--neighbor-k is only supported for single-input mode");
        }
        Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> mat1, mat2;
        std::vector<std::string> rows1, rows2;
        std::vector<std::string> cols1, cols2;

        read_matrix_from_file(inputFiles[0], mat1, &rows1, &cols1);
        notice("Read matrix 1 with %zu rows and %zu columns", mat1.rows(), mat1.cols());
        read_matrix_from_file(inputFiles[1], mat2, &rows2, &cols2);
        notice("Read matrix 2 with %zu rows and %zu columns", mat2.rows(), mat2.cols());

        if (cols1.size() != cols2.size()) {
            error("Number of columns (factors) mismatch: %zu vs %zu", cols1.size(), cols2.size());
        }
        size_t K = cols1.size();
        for (size_t k = 0; k < K; ++k) {
            if (cols1[k] != cols2[k]) {
                warning("Column name mismatch at index %zu: %s vs %s", k, cols1[k].c_str(), cols2[k].c_str());
            }
        }

        std::unordered_map<std::string, int> rowMap1;
        for (size_t i = 0; i < rows1.size(); ++i) rowMap1[rows1[i]] = (int)i;

        std::vector<double> rowSum;
        std::vector<std::pair<int, int>> commonRows;
        std::vector<std::string> commonRowNames;
        for (size_t i = 0; i < rows2.size(); ++i) {
            auto it = rowMap1.find(rows2[i]);
            if (it == rowMap1.end()) {continue;}
            int j = it->second;
            double r = mat1.row(j).sum() + mat2.row(i).sum();
            if (r < minCountPerFeature) {continue;}
            commonRows.emplace_back(j, (int)i);
            commonRowNames.push_back(rows2[i]);
            rowSum.push_back(r);
        }
        if (commonRows.empty()) {
            error("No common features found between two inputs");
        }
        notice("Kept %zu common features", commonRows.size());

        std::vector<double> colSums1(K, 0.0), colSums2(K, 0.0);
        for (size_t k = 0; k < K; ++k) {
            colSums1[k] = mat1.col(k).sum();
            colSums2[k] = mat2.col(k).sum();
        }

        tbb::concurrent_vector<DeResult> results;
        tbb::parallel_for(tbb::blocked_range<size_t>(0, commonRows.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                for (size_t i = range.begin(); i < range.end(); ++i) {
                    int r1 = commonRows[i].first;
                    int r2 = commonRows[i].second;
                    for (size_t k = 0; k < K; ++k) {
                        double a = mat1(r1, k);
                        double b = mat2(r2, k);
                        if (std::max(a,b) < minCount) continue;

                        double c = colSums1[k] - a;
                        double d = colSums2[k] - b;

                        double a_p = a + pseudoCount;
                        double b_p = b + pseudoCount;
                        double t1_p = colSums1[k] + pseudoCount;
                        double t2_p = colSums2[k] + pseudoCount;

                        double fc = (a_p / t1_p) / (b_p / t2_p);
                        if (fc < minFC && fc > 1.0/minFC) continue;

                        auto stats = chisq2x2_log10p(a, b, c, d, pseudoCount);
                        if (stats.second < maxLog10Pval) continue;

                        results.emplace_back((int)i, (int)k, stats.first, stats.second, fc);
                    }
                }
            });

        tbb::parallel_sort(results.begin(), results.end(),
            [](const DeResult& a, const DeResult& b) {
                if (a.k != b.k) return a.k < b.k;
                return a.chi2 > b.chi2;
            });

        std::ofstream out(outputFile);
        if (!out) error("Cannot open output file: %s", outputFile.c_str());
        out << "Feature\tFactor\tChi2\tFoldChange\tlog10pval\tCount1\tCount2\n";
        out << std::fixed << std::setprecision(4);
        for (const auto& r : results) {
            out << commonRowNames[r.j] << "\t" << cols1[r.k] << "\t"
                << r.chi2 << "\t" << r.fc << "\t" << r.log10pval << "\t"
                << mat1(commonRows[r.j].first, r.k) << "\t"
                << mat2(commonRows[r.j].second, r.k) << "\n";
        }

        return 0;
    }

    std::string inputFile = inputFiles[0];

    // Read Input Matrix
    std::ifstream in(inputFile);
    if (!in) error("Cannot open input file: %s", inputFile.c_str());
    std::string line;
    std::vector<std::string> header;
    if (!std::getline(in, line)) {
        error("Input file is empty: %s", inputFile.c_str());
    }
    split(header, "\t", line);
    if (header.size() < 2) error("No factors found in input file header");
    size_t K = header.size() - 1;

    std::vector<std::string> tokens;
    std::vector<std::string> geneNames;
    std::vector<std::vector<double>> matrix;
    std::vector<double> rowSums;
    std::vector<double> colSums(K, 0.0);
    std::vector<double> colNorm2(K, 0.0);

    while(std::getline(in, line)) {
        if (line.empty()) continue;
        split(tokens, "\t", line);
        if (tokens.size() < K+1) {
            error("Invalid line\n%s", line.c_str());
        }
        std::vector<double> row(K, 0.0f);
        double rowSum = 0.0;
        for (size_t k=0; k<K; ++k) {
            if (!str2double(tokens[k+1], row[k])) {
                error("Invalid value %s", tokens[k+1].c_str());
            }
            rowSum += row[k];
        }
        if (rowSum > minCountPerFeature) {
            geneNames.push_back(tokens[0]);
            matrix.push_back(std::move(row));
            rowSums.push_back(rowSum);
        }
    }
    in.close();
    double total_umi = 0.0;
    size_t M = geneNames.size();
    for (size_t i = 0; i < M; ++i) {
        for (size_t k = 0; k < K; ++k) {
            double v = matrix[i][k];
            colSums[k] += v;
            colNorm2[k] += v * v;
        }
    }
    for (size_t k = 0; k < K; ++k) {
        total_umi += colSums[k];
    }
    notice("Loaded %zu genes and %zu factors", M, K);

    tbb::concurrent_vector<DeResult> results;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, M),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i) {
                for (size_t k = 0; k < K; ++k) {
                    if (colSums[k] < minCount) continue;
                    double a = matrix[i][k];
                    if (a < minCount) continue;
                    double b = colSums[k] - a;
                    double c = rowSums[i] - a;
                    double d = total_umi - b - c - a;
                    a += pseudoCount;
                    b += pseudoCount;
                    c += pseudoCount;
                    d += pseudoCount;

                    double fc = (a * d) / (b * c);
                    if (fc < minFC) continue;

                    double chi2 = (a * d - b * c) * (a * d - b * c) / ((a + b) * (c + d) * (a + c) * (b + d)) * (a + b + c + d);
                    double negLog10p = chisq1_log10p(chi2);
                    if (negLog10p < maxLog10Pval) continue;

                    results.emplace_back(i, k, chi2, negLog10p, fc);
                }
            }
        });

    // Sort all results: first by factor (asc), then by Chi2 (desc)
    tbb::parallel_sort(results.begin(), results.end(),
        [](const DeResult& a, const DeResult& b) {
            if (a.k != b.k) return a.k < b.k;
            return a.chi2 > b.chi2;
        });

    std::ofstream out(outputFile);
    if (!out) error("Cannot open output file: %s", outputFile.c_str());
    out << "Feature\tFactor\tChi2\tFoldChange\tlog10pval\n";
    out << std::fixed << std::setprecision(4);
    for (const auto& r : results) {
        out << geneNames[r.j] << "\t" << header[r.k+1] << "\t"
            << r.chi2 << "\t" << r.fc << "\t" << r.log10pval << "\n";
    }
    if (neighborK <= 0) {
        return 0;
    }

    if (neighborK >= static_cast<int>(K-1)) {
        error("--neighbor-k must be smaller than the number of factors minus 1 (%zu)", K-1);
    }

    size_t pos = outputFile.rfind(".tsv");
    if (pos != std::string::npos) {
        outputFile = outputFile.substr(0, pos);
    }
    outputFile += ".1vsNeighbors.tsv";

    std::vector<double> colNorm(K, 0.0);
    for (size_t k = 0; k < K; ++k) {
        colNorm[k] = std::sqrt(colNorm2[k]);
    }

    std::vector<std::vector<int>> neighbors(K);
    std::vector<double> neighborTotals(K, 0.0);
    for (size_t k = 0; k < K; ++k) {
        std::vector<std::pair<double, int>> sims;
        sims.reserve(K - 1);
        for (size_t j = 0; j < K; ++j) {
            if (j == k) continue;
            double sim = 0.0;
            if (colNorm[k] > 0.0 && colNorm[j] > 0.0) {
                double dot = 0.0;
                for (size_t i = 0; i < M; ++i) {
                    dot += matrix[i][k] * matrix[i][j];
                }
                sim = dot / (colNorm[k] * colNorm[j]);
            }
            sims.emplace_back(sim, static_cast<int>(j));
        }
        std::nth_element(sims.begin(), sims.begin() + neighborK, sims.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            });
        sims.resize(static_cast<size_t>(neighborK));
        neighbors[k].reserve(sims.size());
        for (const auto& item : sims) {
            neighbors[k].push_back(item.second);
            neighborTotals[k] += colSums[item.second];
        }
    }

    tbb::concurrent_vector<DeResult> neighborResults;
    tbb::parallel_for(tbb::blocked_range<size_t>(0, K),
        [&](const tbb::blocked_range<size_t>& range) {
            for (size_t k = range.begin(); k < range.end(); ++k) {
                if (neighbors[k].empty()) continue;
                if (colSums[k] < minCount || neighborTotals[k] <= 0.0) continue;
                std::vector<double> neighborSums(M, 0.0);
                for (int j : neighbors[k]) {
                    for (size_t i = 0; i < M; ++i) {
                        neighborSums[i] += matrix[i][j];
                    }
                }
                for (size_t i = 0; i < M; ++i) {
                    double a = matrix[i][k];
                    if (a < minCount) continue;
                    double c = neighborSums[i];
                    double b = colSums[k] - a;
                    double d = neighborTotals[k] - c;

                    double a_p = a + pseudoCount;
                    double b_p = b + pseudoCount;
                    double c_p = c + pseudoCount;
                    double d_p = d + pseudoCount;

                    double fc = (a_p * d_p) / (b_p * c_p);
                    if (fc < minFC) continue;

                    double chi2 = (a_p * d_p - b_p * c_p) * (a_p * d_p - b_p * c_p)
                        / ((a_p + b_p) * (c_p + d_p) * (a_p + c_p) * (b_p + d_p))
                        * (a_p + b_p + c_p + d_p);
                    double negLog10p = chisq1_log10p(chi2);
                    if (negLog10p < maxLog10Pval) continue;

                    neighborResults.emplace_back(static_cast<int>(i), static_cast<int>(k),
                        chi2, negLog10p, fc);
                }
            }
        });

    tbb::parallel_sort(neighborResults.begin(), neighborResults.end(),
        [](const DeResult& a, const DeResult& b) {
            if (a.k != b.k) return a.k < b.k;
            return a.chi2 > b.chi2;
        });

    std::ofstream outNeighbor(outputFile);
    if (!outNeighbor) error("Cannot open output file: %s", outputFile.c_str());
    outNeighbor << "Feature\tFactor\tChi2\tFoldChange\tlog10pval\n";
    outNeighbor << std::fixed << std::setprecision(4);
    for (const auto& r : neighborResults) {
        outNeighbor << geneNames[r.j] << "\t" << header[r.k+1] << "\t"
            << r.chi2 << "\t" << r.fc << "\t" << r.log10pval << "\n";
    }

    return 0;
}



int32_t cmdPseudoBulk(int argc, char** argv) {
    std::string inFile, metaFile, labelFile, outFile;
    int left_idx = 0;
    int right_idx = -1;
    int icol_label = -1;
    int digits = 2;

    ParamList pl;
    // Input Options
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true)
      .add_option("in-label", "Label file", labelFile, true)
      .add_option("icol-label", "Column index (0-based) in --in-label to use as categocial label", icol_label, true)
      .add_option("icol-id-data", "Column index (0-based) in --in-data for matching", right_idx)
      .add_option("icol-id-label", "Column index (0-based) in --in-label for matching", left_idx);
    pl.add_option("out", "Output file", outFile, true)
      .add_option("digits", "Number of digits for output counts", digits);
    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (icol_label < 0) {
        error("--icol-label must be non-negative");
    }

    HexReader reader(metaFile);
    int32_t M = reader.nFeatures;
    std::string line;
    std::vector<std::string> tokens, labels;
    std::unordered_map<std::string, uint32_t> label2idx;
    std::unordered_map<int32_t, uint32_t> id2lidx;
    int32_t nline = 0;

    // Read label file
    {
        std::ifstream in(labelFile);
        if (!in) error("Cannot open label file: %s", labelFile.c_str());
        while (std::getline(in, line)) {
            if (line.empty() || line[0] == '#') continue;
            split(tokens, "\t", line);
            if (tokens.size() <= std::max(left_idx, icol_label)) {
                error("Invalid line in label file: %s", line.c_str());
            }
            std::string label = tokens[icol_label];
            int32_t id = nline++;
            if (left_idx >= 0) {
                if (!str2int32(tokens[left_idx], id)) {
                    error("Invalid ID in label file: %s", tokens[left_idx].c_str());
                }
            }
            uint32_t lidx = 0;
            auto it = label2idx.find(label);
            if (it == label2idx.end()) {
                lidx = static_cast<uint32_t>(label2idx.size());
                label2idx[label] = lidx;
            } else {
                lidx = it->second;
            }
            id2lidx[id] = lidx;
        }
    }
    int32_t K = static_cast<int32_t>(label2idx.size());
    labels.resize(K);
    for (const auto& p : label2idx) {
        labels[p.second] = p.first;
    }
    notice("Found %d unique labels in label file", K);

    Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> pseudoBulk = Eigen::MatrixXd::Zero(M, K);

    std::ifstream inFileStream(inFile);
    if (!inFileStream) {
        error("Fail to open input file: %s", inFile.c_str());
    }
    nline = 0;
    int32_t n_kept = 0;
    while (std::getline(inFileStream, line)) {
        std::string info;
        Document doc;
        int32_t ct = reader.parseLine(doc, info, line);
        int32_t id = nline++;
        if (nline % 10000 == 0) {
            notice("Processed %d lines, kept %d", nline, n_kept);
        }
        if (ct <= 0) {
            continue;
        }
        if (right_idx >= 0) {
            split(tokens, "\t", info);
            if (tokens.size() <= static_cast<size_t>(right_idx) ||
                !str2int32(tokens[right_idx], id)) {
                error("Invalid line in data file starting with %s", info.c_str());
            }
        }
        auto it = id2lidx.find(id);
        if (it == id2lidx.end()) {
            continue;
        }
        uint32_t lidx = it->second;
        for (size_t t = 0; t < doc.ids.size(); ++t) {
            pseudoBulk(doc.ids[t], lidx) += doc.cnts[t];
        }
        n_kept++;
    }
    notice("Finished processing %d lines, kept %d", nline, n_kept);

    write_matrix_to_file(outFile, pseudoBulk, digits, false, reader.features, "Feature", &labels);

    return 0;
}
