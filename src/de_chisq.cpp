#include "numerical_utils.hpp"
#include <tbb/tbb.h>
#include <string>
#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <sstream>
#include <mutex>
#include <algorithm>

std::vector<std::vector<std::string>> chisq_from_matrix_marginal(
    const Eigen::MatrixXd& mat,                    // features x samples
    const std::vector<std::string>& feature_names, // same size as mat.rows()
    const std::vector<std::string>& sample_names,  // same size as mat.cols()
    const std::string& output_filename, int top_k, int nThreads,
    double pseudocount, double min_count, double min_fc, double max_pval)
{
    assert((size_t)mat.rows() == feature_names.size());
    assert((size_t)mat.cols() == sample_names.size());
    std::ofstream fout(output_filename);
    if (!fout.is_open()) {
        throw std::runtime_error("Failed to open output file.");
    }

    const int N_feat = mat.rows();
    const int N_samp = mat.cols();
    Eigen::VectorXd row_sums = mat.rowwise().sum();
    Eigen::VectorXd col_sums = mat.colwise().sum();
    double total = row_sums.sum();
    const double log10_max_pval = -std::log10(max_pval);

    fout << "gene\tfactor\tChi2\tpval\tFoldChange\tgene_total\tlog10p\n";

    using Chi2Feat = std::pair<double, std::string>;
    std::vector<std::mutex> heap_mutexes(N_samp);
    std::vector<std::priority_queue<Chi2Feat, std::vector<Chi2Feat>, std::greater<>>> topk_heaps(N_samp);
    std::mutex file_mutex;

    tbb::global_control c(tbb::global_control::max_allowed_parallelism, nThreads);

    tbb::parallel_for(0, N_feat, [&](int i) {
        const std::string& feat_name = feature_names[i];
        double row_total = row_sums(i);
        if (row_total < min_count) return;

        for (int j = 0; j < N_samp; ++j) {
            double a = mat(i, j);
            if (a < min_count) continue;

            double b = col_sums(j) - a;
            double c = row_total - a;
            double d = total - a - b - c;

            a += pseudocount;
            b += pseudocount;
            c += pseudocount;
            d += pseudocount;

            double fc = (a * d) / (b * c);
            if (fc < min_fc) continue;

            double chi2 = std::pow(a * d - b * c, 2) / ((a + b) * (c + d) * (a + c) * (b + d)) * (a + b + c + d);
            double log10pval = - chisq1_log10p(chi2);

            if (log10pval < log10_max_pval) continue;

            {
                std::ostringstream oss;
                oss.precision(6);
                oss << std::fixed;
                oss << feat_name << '\t' << sample_names[j] << '\t'
                    << chi2 << '\t' << std::pow(10.0, -log10pval) << '\t'
                    << fc << '\t' << row_total << '\t' << log10pval << '\n';

                std::lock_guard<std::mutex> lock(file_mutex);
                fout << oss.str();
            }

            {
                std::lock_guard<std::mutex> lock(heap_mutexes[j]);
                auto& heap = topk_heaps[j];
                if ((int)heap.size() < top_k) {
                    heap.emplace(chi2, feat_name);
                } else if (chi2 > heap.top().first) {
                    heap.pop();
                    heap.emplace(chi2, feat_name);
                }
            }
        }
    });

    fout.close();

    std::vector<std::vector<std::string>> result(N_samp);
    for (int j = 0; j < N_samp; ++j) {
        auto& heap = topk_heaps[j];
        std::vector<Chi2Feat> entries;
        while (!heap.empty()) {
            entries.push_back(heap.top());
            heap.pop();
        }
        std::sort(entries.begin(), entries.end(), std::greater<>());
        for (const auto& p : entries) {
            result[j].push_back(p.second);
        }
    }

    return result;
}
