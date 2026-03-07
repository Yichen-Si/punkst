#include "multi_cde_pixel_common.hpp"

#include <cmath>
#include <cstdio>
#include <unordered_set>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

#include "numerical_utils.hpp"
#include "utils.h"

namespace {

struct TestRec {
    int k = -1;
    int f = -1;
    double beta = 0.0;
    double log10p = -1.0;
    double pi0 = 0.0;
    double pi1 = 0.0;
    double total_count = 0.0;
    double beta_deconv = 0.0;
    double fc_deconv = 0.0;
    double log10p_deconv = -1.0;
    double p_perm = -1.0;
    bool perm_candidate = false;
};

} // namespace

void buildPairwiseContrasts(const std::vector<std::string>& dataLabels,
                            std::vector<ContrastDef>& contrasts) {
    const int32_t n = static_cast<int32_t>(dataLabels.size());
    contrasts.clear();
    if (n < 2) {
        return;
    }
    contrasts.reserve(static_cast<size_t>(n) * (n - 1) / 2);
    for (int32_t g0 = 0; g0 < n; ++g0) {
        for (int32_t g1 = g0 + 1; g1 < n; ++g1) {
            ContrastDef contrast;
            contrast.name = dataLabels[g0] + "_vs_" + dataLabels[g1];
            contrast.labels.assign(n, 0);
            contrast.labels[g0] = -1;
            contrast.labels[g1] = 1;
            contrast.group_neg.push_back(g0);
            contrast.group_pos.push_back(g1);
            contrasts.push_back(std::move(contrast));
        }
    }
}

Eigen::MatrixXd readConfusionMatrixFile(const std::string& path, int K) {
    RowMajorMatrixXd mat;
    std::vector<std::string> rnames;
    std::vector<std::string> cnames;
    read_matrix_from_file(path, mat, &rnames, &cnames);
    if (mat.rows() != K || mat.cols() != K) {
        error("Confusion matrix %s has size %d x %d, expected %d x %d",
              path.c_str(), static_cast<int>(mat.rows()), static_cast<int>(mat.cols()), K, K);
    }
    return Eigen::MatrixXd(mat);
}

void accumulateConfusionFromTopProbs(const TopProbs& tp,
                                     Eigen::MatrixXd& confusion,
                                     double& residualAccum) {
    double residual = 1.0;
    for (size_t ii = 0; ii < tp.ks.size(); ++ii) {
        residual -= tp.ps[ii];
        for (size_t jj = ii; jj < tp.ks.size(); ++jj) {
            confusion(tp.ks[ii], tp.ks[jj]) += tp.ps[ii] * tp.ps[jj];
        }
    }
    if (residual > 0.0) {
        residualAccum += residual * residual;
    }
}

void finalizeConfusionMatrix(Eigen::MatrixXd& confusion,
                             double residualAccum,
                             int K) {
    confusion += confusion.transpose().eval();
    const double p_residual = residualAccum / static_cast<double>(K) / static_cast<double>(K);
    for (int k = 0; k < K; ++k) {
        confusion(k, k) /= 2.0;
        for (int l = 0; l < K; ++l) {
            confusion(k, l) += p_residual;
        }
    }
}

int32_t runConditionalPixelTests(const std::string& outPrefix,
                                 const std::string& auxSuffix,
                                 const std::vector<std::string>& dataLabels,
                                 const std::vector<std::string>& featureList,
                                 const std::vector<ContrastDef>& contrasts,
                                 MultiSlicePairwiseBinom& statOp,
                                 PairwiseBinomRobust& statUnion,
                                 MultiSliceUnitCache& unitCache,
                                 const PixelDETestOptions& opts) {
    if (contrasts.empty()) {
        error("No contrasts defined");
    }
    const int K = statOp.get_n_slices();
    const int M = statOp.get_n_features();
    const int32_t n_data = static_cast<int32_t>(dataLabels.size());
    if (n_data != statOp.get_n_groups()) {
        error("Group label count (%d) does not match statistics group count (%d)",
              n_data, statOp.get_n_groups());
    }
    if (static_cast<int32_t>(featureList.size()) != M) {
        error("Feature count (%zu) does not match statistics feature count (%d)",
              featureList.size(), M);
    }

    statOp.finished_adding_data();

    const double minLog10p = -std::log10(opts.minPvalOutput);
    const double min_log_or = std::log(opts.minOROutput);
    const double min_log_or_perm = std::log(opts.minORPerm);
    const auto& active = statOp.get_active_features();

    for (const auto& contrast : contrasts) {
        ContrastPrecomp pc = statOp.prepare_contrast(
            contrast.group_neg, contrast.group_pos, 25, 1e-6, opts.pseudoFracRel);

        std::vector<TestRec> tests;
        tests.reserve(static_cast<size_t>(K) * active.size());
        notice("Contrast %s with %d active features", contrast.name.c_str(),
               static_cast<int32_t>(active.size()));

        for (int f : active) {
            MultiSliceOneResult ms(K);
            if (!statOp.compute_one_test_aggregate(
                    f, contrast.group_neg, contrast.group_pos, pc, ms,
                    opts.minCountPerFeature, true, opts.deconvHitP)) {
                continue;
            }

            std::vector<uint8_t> ok(K, 0);
            int ok_count = 0;
            for (int k = 0; k < K; ++k) {
                if (!ms.slice_ok[static_cast<size_t>(k)]) {
                    continue;
                }
                ok[k] = 1;
                ok_count++;
            }
            if (ok_count == 0) {
                continue;
            }

            for (int k = 0; k < K; ++k) {
                if (!ok[k] || std::abs(ms.beta_obs(k)) < min_log_or) {
                    continue;
                }
                if (ms.log10p_obs(k) < minLog10p) {
                    continue;
                }

                double total_count = 0.0;
                {
                    const auto& counts = statOp.slice(k).get_group_counts();
                    double y0 = 0.0;
                    double y1 = 0.0;
                    for (int32_t g0 : contrast.group_neg) {
                        const size_t idx = static_cast<size_t>(g0) * M + f;
                        y0 += counts[idx];
                    }
                    for (int32_t g1 : contrast.group_pos) {
                        const size_t idx = static_cast<size_t>(g1) * M + f;
                        y1 += counts[idx];
                    }
                    total_count = y0 + y1;
                }

                double beta_deconv = 0.0;
                double fc_deconv = 0.0;
                double log10p_deconv = -1.0;
                if (ms.deconv_ok) {
                    beta_deconv = ms.beta_deconv(k);
                    const double pi0_deconv = ms.pi0_deconv(k);
                    const double pi1_deconv = ms.pi1_deconv(k);
                    if (pi0_deconv > 0.0 && pi1_deconv > 0.0 &&
                        std::isfinite(pi0_deconv) && std::isfinite(pi1_deconv)) {
                        fc_deconv = pi1_deconv / pi0_deconv;
                    }
                    const double se = std::sqrt(ms.varb_obs(k));
                    if (se > 0.0 && std::isfinite(se)) {
                        log10p_deconv = -log10_twosided_p_from_z(beta_deconv / se);
                    }
                }

                TestRec rec;
                rec.k = k;
                rec.f = f;
                rec.beta = ms.beta_obs(k);
                rec.log10p = ms.log10p_obs(k);
                rec.pi0 = ms.pi0_obs(k);
                rec.pi1 = ms.pi1_obs(k);
                rec.total_count = total_count;
                rec.beta_deconv = beta_deconv;
                rec.fc_deconv = fc_deconv;
                rec.log10p_deconv = log10p_deconv;
                rec.perm_candidate = (std::abs(rec.beta) >= min_log_or_perm);
                tests.push_back(rec);
            }
        }

        if (opts.nPerm > 0 && !tests.empty()) {
            struct PermTLS {
                std::vector<double> N1;
                std::vector<double> Y1;
                std::vector<uint32_t> touched_idx;
                std::vector<uint32_t> exceed;
                PermTLS(int K, int M, size_t T)
                    : N1(K, 0.0), Y1(static_cast<size_t>(K) * M, 0.0), exceed(T, 0) {}
                void reset_perm() {
                    std::fill(N1.begin(), N1.end(), 0.0);
                    for (uint32_t idx : touched_idx) {
                        Y1[idx] = 0.0;
                    }
                    touched_idx.clear();
                }
                void add_y1(int k, int f, double y, int M) {
                    const uint32_t idx = static_cast<uint32_t>(k * M + f);
                    if (Y1[idx] == 0.0) {
                        touched_idx.push_back(idx);
                    }
                    Y1[idx] += y;
                }
            };

            const std::string contrastName = contrast.name;
            std::vector<double> Ntot(K, 0.0);
            std::vector<int> n1_units(K, 0), n_units(K, 0);
            std::vector<std::vector<uint32_t>> eligible_units(K);
            for (int k = 0; k < K; ++k) {
                const auto& sl = statOp.slice(k);
                Ntot[k] = sl.sum_group_totals(contrast.group_neg) +
                          sl.sum_group_totals(contrast.group_pos);
                n1_units[k] = sl.sum_group_unit_counts(contrast.group_pos);
                n_units[k] = sl.sum_group_unit_counts(contrast.group_neg) +
                             sl.sum_group_unit_counts(contrast.group_pos);
                const auto& units = unitCache.slice_units(k);
                eligible_units[k].reserve(units.size());
                int cached_n = 0;
                for (uint32_t u = 0; u < units.size(); ++u) {
                    const int32_t g = units[u].group;
                    if (g < 0 || g >= static_cast<int32_t>(contrast.labels.size())) {
                        continue;
                    }
                    const int8_t lab = contrast.labels[g];
                    if (lab == 0) {
                        continue;
                    }
                    eligible_units[k].push_back(u);
                    cached_n++;
                }
                if (cached_n != n_units[k]) {
                    warning("Slice %d cache units (%d) != observed units (%d) for contrast %s.",
                            k, cached_n, n_units[k], contrastName.c_str());
                }
            }

            int32_t n_candidate = 0;
            for (const auto& rec : tests) {
                if (rec.perm_candidate) {
                    n_candidate++;
                }
            }
            notice("Permutation: %d tests (slice, feature) to evaluate for contrast %s",
                   n_candidate, contrastName.c_str());

            tbb::enumerable_thread_specific<PermTLS> tls_perm([&] {
                return PermTLS(K, M, tests.size());
            });

            tbb::parallel_for(tbb::blocked_range<int>(0, opts.nPerm),
                [&](const tbb::blocked_range<int>& range) {
                    auto& T = tls_perm.local();
                    for (int r = range.begin(); r != range.end(); ++r) {
                        T.reset_perm();
                        for (int k = 0; k < K; ++k) {
                            const auto& units = unitCache.slice_units(k);
                            const auto& fid = unitCache.slice_feat_ids(k);
                            const auto& fct = unitCache.slice_feat_counts(k);
                            const auto& eligible = eligible_units[k];
                            const int N = static_cast<int>(eligible.size());
                            int need1 = n1_units[k];
                            if (N <= 0 || need1 < 0 || need1 > N) {
                                continue;
                            }
                            int assigned1 = 0;
                            for (int u = 0; u < N; ++u) {
                                const int remain = N - u;
                                const int remain1 = need1 - assigned1;
                                if (remain1 <= 0) {
                                    break;
                                }
                                bool pick1 = false;
                                if (remain1 == remain) {
                                    pick1 = true;
                                } else {
                                    const double prob = static_cast<double>(remain1) / remain;
                                    const double uu = u01(opts.seed, static_cast<uint64_t>(r),
                                                          static_cast<uint64_t>(k),
                                                          static_cast<uint64_t>(u));
                                    pick1 = (uu < prob);
                                }
                                if (!pick1) {
                                    continue;
                                }
                                assigned1++;
                                const auto& U = units[eligible[u]];
                                T.N1[k] += static_cast<double>(U.n);
                                for (uint32_t t = 0; t < U.len; ++t) {
                                    const int f = fid[U.off + t];
                                    const double y = fct[U.off + t];
                                    T.add_y1(k, f, y, M);
                                }
                            }
                        }

                        for (size_t j = 0; j < tests.size(); ++j) {
                            if (!tests[j].perm_candidate) {
                                continue;
                            }
                            const int k = tests[j].k;
                            const int f = tests[j].f;
                            const double N1p = T.N1[k];
                            const double N0p = Ntot[k] - N1p;
                            if (N0p <= 0.0 || N1p <= 0.0) {
                                continue;
                            }
                            const double Y1p = T.Y1[static_cast<size_t>(k * M + f)];
                            const double Y0p = tests[j].total_count - Y1p;
                            const double pi_eps = (Y0p + Y1p) / (N0p + N1p) * opts.pseudoFracRel;
                            const double pi0p = clamp(Y0p / N0p, pi_eps, 1.0 - 1e-8);
                            const double pi1p = clamp(Y1p / N1p, pi_eps, 1.0 - 1e-8);
                            const double beta_p = logit(pi1p) - logit(pi0p);
                            if (!std::isfinite(beta_p)) {
                                continue;
                            }
                            if (std::abs(beta_p) >= std::abs(tests[j].beta)) {
                                T.exceed[j] += 1;
                            }
                        }
                    }
                });

            std::vector<uint32_t> exceed(tests.size(), 0);
            for (auto& T : tls_perm) {
                for (size_t j = 0; j < tests.size(); ++j) {
                    exceed[j] += T.exceed[j];
                }
            }
            for (size_t j = 0; j < tests.size(); ++j) {
                if (tests[j].perm_candidate) {
                    tests[j].p_perm = static_cast<double>(exceed[j]) / opts.nPerm;
                }
            }
        }

        std::string outFile = outPrefix + "." + contrast.name + ".tsv";
        FILE* out_stream = fopen(outFile.c_str(), "w");
        if (!out_stream) {
            error("Cannot open output file: %s", outFile.c_str());
        }
        const std::string header =
            "Slice\tFeature\tBeta\tlog10p\tPi0\tPi1\tTotalCount\tBeta_deconv\tFC_deconv\tlog10p_deconv";
        if (opts.nPerm > 0) {
            fprintf(out_stream, "%s\tp_perm\n", header.c_str());
        } else {
            fprintf(out_stream, "%s\n", header.c_str());
        }
        for (const auto& rec : tests) {
            fprintf(out_stream,
                    "%d\t%s\t%.4e\t%.4f\t%.4e\t%.4e\t%.1f\t%.4e\t%.4e\t%.4f",
                    rec.k, featureList[rec.f].c_str(),
                    rec.beta, rec.log10p, rec.pi0, rec.pi1, rec.total_count,
                    rec.beta_deconv, rec.fc_deconv, rec.log10p_deconv);
            if (opts.nPerm > 0) {
                fprintf(out_stream, "\t%.4f", rec.p_perm);
            }
            fprintf(out_stream, "\n");
        }
        fclose(out_stream);
        notice("Result for %s is written to:\n  %s", contrast.name.c_str(), outFile.c_str());
    }

    const std::string suffix = auxSuffix.empty() ? "" : ("." + auxSuffix);
    std::string outFile = outPrefix + suffix + ".nobs.tsv";
    FILE* out_nobs = fopen(outFile.c_str(), "w");
    if (!out_nobs) {
        error("Cannot open output file: %s", outFile.c_str());
    }
    outFile = outPrefix + suffix + ".sums.tsv";
    FILE* out_sums = fopen(outFile.c_str(), "w");
    if (!out_sums) {
        error("Cannot open output file: %s", outFile.c_str());
    }
    fprintf(out_nobs, "Slice\tData\tnUnits\tTotalCount\n");
    fprintf(out_sums, "Slice\tData\tFeature\tTotalCount\n");
    for (int k = 0; k < K; ++k) {
        const auto& slice = statOp.slice(k);
        const auto& n_units = slice.get_group_unit_counts();
        const auto& totals = slice.get_group_totals();
        const auto& counts = slice.get_group_counts();
        for (int32_t i = 0; i < n_data; ++i) {
            fprintf(out_nobs, "%d\t%s\t%d\t%.1f\n",
                    k, dataLabels[i].c_str(), n_units[i], totals[i]);
            for (int32_t m = 0; m < M; ++m) {
                const size_t j = static_cast<size_t>(i) * M + m;
                fprintf(out_sums, "%d\t%s\t%s\t%.1f\n",
                        k, dataLabels[i].c_str(), featureList[m].c_str(), counts[j]);
            }
        }
    }
    fclose(out_nobs);
    fclose(out_sums);
    return 0;
}
