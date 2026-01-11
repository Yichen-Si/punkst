#include "punkst.h"
#include "tileoperator.hpp"
#include "utils.h"
#include "glm.hpp"
#include <random>
#include <fstream>
#include <iomanip>
#include <memory>
#include <atomic>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

/**
 * Join pixel level decoding results with original transcripts and perform
 * cluster/factor specific (conditional) DE test between groups
 */
int32_t cmdConditionalTest(int32_t argc, char** argv) {
    std::vector<std::string> inPrefix, inData, inIndex, inPtsPrefix;
    std::vector<std::string> dataLabels;
    std::string dictFile, outPrefix, outFile;
    bool isBinary = false;
    double gridSize;
    int32_t K;
    int32_t icol_x, icol_y, icol_feature, icol_val;
    float qxmin, qxmax, qymin, qymax;
    bool bounded = false;
    double pseudoCount = 0.5;
    double minCount = 10.0;
    double minCountPerFeature = 100.0;
    double minPval = 1e-3;
    double minOR = 1.2;
    int32_t debug_ = 0;
    int32_t nThreads = 1;
    int32_t nPerm = 0;
    uint64_t seed = 1;

    ParamList pl;
    pl.add_option("anno-data", "Input pixel files", inData)
      .add_option("anno-index", "Input pixel index files", inIndex)
      .add_option("anno", "Prefixes of input pixel data files", inPrefix)
      .add_option("labels", "Labels for the datasets to show in pairwise output", dataLabels)
      .add_option("binary", "Data file is in binary format", isBinary)
      .add_option("K", "Number of factors in the annotation files", K, true)
      .add_option("pts", "Prefixes of the transcript data files", inPtsPrefix, true)
      .add_option("features", "List of features to test", dictFile, true)
      .add_option("grid-size", "Grid size", gridSize)
      .add_option("icol-x", "Column index for x coordinate for files in --pts (0-based)", icol_x, true)
      .add_option("icol-y", "Column index for y coordinate for files in --pts (0-based)", icol_y, true)
      .add_option("icol-feature", "Column index for feature for files in --pts (0-based)", icol_feature, true)
      .add_option("icol-val", "Column index for count/value for files in --pts (0-based)", icol_val, true)
      .add_option("xmin", "Minimum x coordinate for subsetting", qxmin)
      .add_option("xmax", "Maximum x coordinate for subsetting", qxmax)
      .add_option("ymin", "Minimum y coordinate for subsetting", qymin)
      .add_option("ymax", "Maximum y coordinate for subsetting", qymax)
      .add_option("bounded", "Whether to subset to the bounding box defined by --xmin/--xmax/--ymin/--ymax", bounded)
      .add_option("threads", "Number of threads to use", nThreads)
      .add_option("out", "Output prefix", outPrefix, true)
      .add_option("min-count-per-feature", "Minimum total count for a feature to be considered", minCountPerFeature)
      .add_option("max-pval", "Max p-value for output (default: 1e-3)", minPval)
      .add_option("min-or", "Minimum odds ratio for output (default: 1.2)", minOR)
      .add_option("min-count", "Minimum observed factor-specific count for a unit to be included (default: 10)", minCount)
      .add_option("perm", "Number of permutations for beta calibration (two-sample only)", nPerm)
      .add_option("seed", "Seed for permutations", seed)
      .add_option("debug", "Debug", debug_);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }
    if (debug_ > 0) {
        logger::Logger::getInstance().setLevel(logger::LogLevel::DEBUG);
    }
    if (nThreads < 1) {
        nThreads = 1;
    }
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nThreads);

    if (!inPrefix.empty()) {
        inData.resize(inPrefix.size());
        inIndex.resize(inPrefix.size());
        for (size_t i = 0; i < inPrefix.size(); ++i) {
            inData[i] = inPrefix[i] + (isBinary ? ".bin" : ".tsv");
            inIndex[i] = inPrefix[i] + ".index";
        }
    } else if (inData.empty() || inIndex.empty() || inData.size() != inIndex.size()) {
        error("Either --anno or both --anno-data and --anno-index must be specified");
    }
    if (inPtsPrefix.size() != inData.size()) {
        error("Number of --pts must match number of --anno/--anno-data");
    }

    uint32_t n_data = static_cast<uint32_t>(inData.size());
    if (gridSize <= 0) {error("Either --grid-size must be positive");}
    if (K <= 0) {error("--K must be positive");}

    if (bounded) {
        if (qxmin >= qxmax || qymin >= qymax) {
            error("Invalid bounding box specified");
        }
    }

    double minLog10p = - std::log10(minPval);

    lineParserUnival parser(icol_x, icol_y, icol_feature, icol_val, dictFile);
    std::vector<std::string> featureList;
    parser.getFeatureList(featureList);
    int32_t M = static_cast<int32_t>(featureList.size());
    if (M == 0) {error("No features found");}

    std::vector<std::unique_ptr<TileOperator>> tileOps;
    for (uint32_t i = 0; i < n_data; ++i) {
        tileOps.emplace_back(std::make_unique<TileOperator>(inData[i], inIndex[i]));
        if (bounded) {
            int32_t n = tileOps.back()->query(qxmin, qxmax, qymin, qymax);
            debug("Dataset %u: using %d tiles in the bounding box", i, n);
        }
    }

    std::vector<TileReader> readers;
    for (uint32_t i = 0; i < n_data; ++i) {
        readers.emplace_back(inPtsPrefix[i] + ".tsv", inPtsPrefix[i] + ".index");
        if (readers.back().getTileSize() != tileOps[i]->getTileSize()) {
            error("Currently we require the tile size to be the same for each pair of annotation and transcript data. Invalid dataset %u (%u vs %u)", i, tileOps[i]->getTileSize(), readers.back().getTileSize());
        }
    }

    MultiSlicePairwiseBinom statOp(K, n_data, M, minCount);
    MultiSliceUnitCache unitCache(K, M, minCount);
    struct LocalAgg {
        MultiSlicePairwiseBinom stat;
        MultiSliceUnitCache cache;
        LocalAgg(int K, int G, int M, double minCount)
            : stat(K, G, M, minCount), cache(K, M, minCount) {}
    };

    dataLabels.resize(n_data);
    for (uint32_t i = 0; i < n_data; ++i) {
        if (dataLabels[i].empty())
            dataLabels[i] = std::to_string(i);
    }

    for (uint32_t i = 0; i < n_data; ++i) {
        auto& tileOp = *tileOps[i];
        const auto& tileList = tileOp.getTileInfo();
        int32_t nTiles = static_cast<int32_t>(tileList.size());
        notice("Processing %d tiles for dataset %u", nTiles, i);
        auto& reader = readers[i];
        std::atomic<int32_t> processed{0};
        // tbb::enumerable_thread_specific<MultiSlicePairwiseBinom> tls([&] {
        //     return MultiSlicePairwiseBinom(K, n_data, M, minCount);
        // });
        tbb::enumerable_thread_specific<LocalAgg> tls([&] {
            return LocalAgg(K, (int)n_data, M, minCount);
        });

        tbb::parallel_for(tbb::blocked_range<size_t>(0, tileList.size()),
            [&](const tbb::blocked_range<size_t>& range) {
                auto& local = tls.local();
                for (size_t ti = range.begin(); ti != range.end(); ++ti) {
                    const auto& tileInfo = tileList[ti];
                    TileKey tile{tileInfo.row, tileInfo.col};
                    auto tileAgg = tileOp.aggOneTile(reader, parser, tile, gridSize);
                    for (const auto& kv : tileAgg) {
                        const int32_t k = kv.first;
                        if (k < 0 || k >= K) continue;
                        for (const auto& unitKv : kv.second) {
                            const auto& obs = unitKv.second;
                            local.stat.slice(k).add_unit(static_cast<int>(i), obs.totalCount, obs.featureCounts);
                            if (nPerm > 0)
                            local.cache.add_unit(k, obs.totalCount, obs.featureCounts);
                        }
                    }
                    int32_t done = processed.fetch_add(1) + 1;
                    if (done % 10 == 0) {
                        notice("... processed %d / %d tiles for dataset %u", done, nTiles, i);
                    }
                }
            });
        for (auto& local : tls) {
            statOp.merge_from(local.stat);
            unitCache.merge_from(local.cache);
        }
    }
    statOp.finished_adding_data();
    notice("Finished collecting sufficient from all datasets");

    std::vector<std::pair<int, int>> pairs = PairwiseBinomRobust::make_all_pairs(static_cast<int>(n_data));

    // ---------------- Permutation calibration ----------------
    if (nPerm > 0) {
        const double min_log_or = std::log(minOR);
        const double pi_eps = 1e-8;
        // Store observed stats
        struct TestRec {
            int k;
            int f;
            PairwiseBinomRobust::PairwiseOneResult obs;
        };
        // Thread-local buffers
        struct PermTLS {
            std::vector<double> N1;
            std::vector<double> Y1;
            std::vector<uint32_t> touched_idx; // indices of Y1 modified
            std::vector<uint32_t> exceed; // exceed counts for tests
            PermTLS(int K, int M, size_t T)
                : N1(K, 0.0), Y1((size_t)(K * M), 0.0), exceed(T, 0) {}
            inline void reset_perm() {
                std::fill(N1.begin(), N1.end(), 0.0);
                for (uint32_t idx : touched_idx) Y1[idx] = 0.0;
                touched_idx.clear();
            }
            inline void add_y1(int k, int f, double y, int M) {
                const uint32_t idx = (uint32_t)(k * M + f);
                if (Y1[idx] == 0.0) touched_idx.push_back(idx);
                Y1[idx] += y;
            }
        };
        for (auto [g0, g1] : pairs) {

        std::string contrastName = dataLabels[g0] + "_vs_" + dataLabels[g1];

        // Get per-slice totals and unit counts from observed stats
        std::vector<double> Ntot(K, 0.0); // total counts across two groups
        std::vector<int> n1_units(K, 0), n_units(K, 0);
        for (int k = 0; k < K; ++k) {
            const auto& sl = statOp.slice(k);
            Ntot[k] = sl.get_group_totals(g0) + sl.get_group_totals(g1);
            const auto& nu = sl.get_group_unit_counts();
            n1_units[k] = nu[g1];
            n_units[k]  = nu[g0] + nu[g1];
            const int cached_n = (int)unitCache.slice_units(k).size();
            if (cached_n != n_units[k]) {
                warning("Slice %d cache units (%d) != observed units (%d). "
                    "Permutation denominators may be wrong if units with empty features were not cached.",
                    k, cached_n, n_units[k]);
            }
        }

        std::vector<TestRec> tests;
        tests.reserve((size_t)K * (size_t)statOp.get_active_features().size());

        for (int k = 0; k < K; ++k) {
            const auto& sl  = statOp.slice(k);
            const double N0 = sl.get_group_totals(g0);
            const double N1 = sl.get_group_totals(g1);
            if (N0 <= 0.0 || N1 <= 0.0) continue;
            for (int f : statOp.get_active_features()) {
                PairwiseBinomRobust::PairwiseOneResult res;
                if (!sl.compute_one_test(f, g0, g1, res,
                        minCountPerFeature, pi_eps, true)) {
                    continue;
                }
                if (std::abs(res.beta) < min_log_or) {
                    continue;
                }
                tests.push_back(TestRec{k, f, res});
            }
        }

        notice("Permutation: %d tests (slice, feature) to evaluate", (int)tests.size());

        // Output file
        std::string permFile = outPrefix + "." + contrastName + ".perm_" + std::to_string(nPerm) + ".tsv";
        FILE* out_perm = fopen(permFile.c_str(), "w");
        if (!out_perm) { error("Cannot open output file: %s", permFile.c_str()); }
        fprintf(out_perm, "Slice\tFeature\tBeta\tPi0\tPi1\tTotalCount\tlog10p\tp_perm\n");

        tbb::enumerable_thread_specific<PermTLS> tls_perm([&] {
            return PermTLS(K, M, tests.size());
        });

        // Parallel over permutations
        tbb::parallel_for(tbb::blocked_range<int>(0, nPerm),
            [&](const tbb::blocked_range<int>& range)
        {
            auto& T = tls_perm.local();
            for (int r = range.begin(); r != range.end(); ++r) {
                T.reset_perm();
                for (int k = 0; k < K; ++k) {
                    const auto& units = unitCache.slice_units(k);
                    const auto& fid   = unitCache.slice_feat_ids(k);
                    const auto& fct   = unitCache.slice_feat_counts(k);
                    const int N = (int)units.size();
                    int need1 = n1_units[k];
                    if (N <= 0 || need1 < 0 || need1 > N) continue;
                    int assigned1 = 0;
                    for (int u = 0; u < N; ++u) {
                        const int remain  = N - u;
                        const int remain1 = need1 - assigned1;
                        if (remain1 <= 0) break;
                        bool pick1 = false;
                        if (remain1 == remain) {
                            pick1 = true;
                        } else {
                            const double prob = (double)remain1 / (double)remain;
                            const double uu = u01(seed, (uint64_t)r, (uint64_t)k, (uint64_t)u);
                            pick1 = (uu < prob);
                        }
                        if (!pick1) continue;
                        // unit assigned to group 1
                        assigned1++;
                        const auto& U = units[u];
                        T.N1[k] += (double)U.n;
                        const uint32_t off = U.off;
                        const uint32_t len = U.len;
                        for (uint32_t t = 0; t < len; ++t) {
                            const int f = (int)fid[off + t];
                            const double y = fct[off + t];
                            T.add_y1(k, f, y, M);
                        }
                    }
                }

                // Compute betas & compare with observed beta
                for (size_t j = 0; j < tests.size(); ++j) {
                    const auto& Q = tests[j];
                    const int k = Q.k;
                    const int f = Q.f;
                    const double N1p = T.N1[k];
                    const double N0p = Ntot[k] - N1p;
                    if (N0p <= 0.0 || N1p <= 0.0) continue;

                    const double Y1p = T.Y1[(size_t)(k*M+f)];
                    const double Y0p = Q.obs.tot - Y1p;
                    const double pi0p = clamp(Y0p / N0p, pi_eps, 1.0 - pi_eps);
                    const double pi1p = clamp(Y1p / N1p, pi_eps, 1.0 - pi_eps);
                    const double beta_p = logit(pi1p) - logit(pi0p);
                    if (!std::isfinite(beta_p)) continue;

                    if (std::abs(beta_p) >= std::abs(Q.obs.beta)) {
                        T.exceed[j] += 1;
                    }
                }
            }
        });

        // Reduce exceedance counts
        std::vector<uint32_t> exceed(tests.size(), 0);
        for (auto& T : tls_perm) {
            for (size_t j = 0; j < tests.size(); ++j) exceed[j] += T.exceed[j];
        }

        // Output permutation p-values
        for (size_t j = 0; j < tests.size(); ++j) {
            const auto& Q  = tests[j];
            double se = std::sqrt(Q.obs.varb);
            double log10p = -log10_twosided_p_from_z(Q.obs.beta/se);
            const double p = (double)exceed[j] / (double)nPerm;
            fprintf(out_perm, "%d\t%s\t%.4e\t%.4e\t%.4e\t%.1f\t%.4e\t%.4f\n",
                Q.k, featureList[Q.f].c_str(),
                Q.obs.beta, Q.obs.pi0, Q.obs.pi1, Q.obs.tot, log10p, p);
        }
        fclose(out_perm);
        notice("Permutation results written: %s", permFile.c_str());
    }
    } else {
        outFile = outPrefix + ".marginal.tsv";
        FILE* out_marginal = fopen(outFile.c_str(), "w");
        if (!out_marginal) {error("Cannot open output file: %s", outFile.c_str());}
        outFile = outPrefix + ".global.tsv";
        FILE* out_global = fopen(outFile.c_str(), "w");
        if (!out_global) {error("Cannot open output file: %s", outFile.c_str());}
        outFile = outPrefix + ".deviation.tsv";
        FILE* out_deviation = fopen(outFile.c_str(), "w");
        if (!out_deviation) {error("Cannot open output file: %s", outFile.c_str());}

        fprintf(out_marginal,"Slice\tFeature\tData0\tData1\tBeta\tSE\tlog10p\tPi0\tPi1\tTotalCount\n");
        fprintf(out_global,"Feature\tData0\tData1\tnPassSlices\tBeta\tSE\tlog10p\n");
        fprintf(out_deviation,"Slice\tFeature\tData0\tData1\tBeta\tSE\tlog10p\n");

        statOp.compute_tests(
            // emit_slice
            [&](int f, int g0, int g1, int k, double beta, double se, double log10p, double pi0, double pi1, double tot) {
                if (log10p >= minLog10p) {
                    fprintf(out_marginal, "%d\t%s\t%s\t%s\t%.4e\t%.4e\t%.4e\t%.4e\t%.4e\t%.1f\n",
                        k, featureList[f].c_str(),
                        dataLabels[g0].c_str(), dataLabels[g1].c_str(),
                        beta, se, log10p, pi0, pi1, tot);
                }
            },
            // emit_g
            [&](int f, int g0, int g1, int nk, double g, double se, double log10p) {
                fprintf(out_global, "%s\t%s\t%s\t%d\t%.4e\t%.4e\t%.4f\n",
                    featureList[f].c_str(),
                    dataLabels[g0].c_str(), dataLabels[g1].c_str(),
                    nk, g, se, log10p);
            },
            // emit_d
            [&](int f, int g0, int g1, int k, double d, double se, double log10p) {
                if (log10p >= minLog10p) {
                    fprintf(out_deviation, "%d\t%s\t%s\t%s\t%.4e\t%.4e\t%.4f\n",
                        k, featureList[f].c_str(),
                        dataLabels[g0].c_str(), dataLabels[g1].c_str(),
                        d, se, log10p);
                }
            },
            minCountPerFeature, pairs, minOR,
            /*pi_eps=*/1e-8,
            /*use_hc1=*/true,
            /*renormalize_pi_over_available=*/true,
            /*emit_marginal_slices=*/true,
            /*emit_decomp=*/true
        );
        fclose(out_marginal);
        fclose(out_global);
        fclose(out_deviation);
    }

    outFile = outPrefix + ".nobs.tsv";
    FILE* out_nobs = fopen(outFile.c_str(), "w");
    if (!out_nobs) {error("Cannot open output file: %s", outFile.c_str());}
    outFile = outPrefix + ".sums.tsv";
    FILE* out_sums = fopen(outFile.c_str(), "w");
    if (!out_sums) {error("Cannot open output file: %s", outFile.c_str());}
    fprintf(out_nobs, "Slice\tData\tnUnits\tTotalCount\n");
    fprintf(out_sums, "Slice\tData\tFeature\tTotalCount\n");
    for (int k = 0; k < K; ++k) {
        const auto& slice = statOp.slice(k);
        const auto& n_units = slice.get_group_unit_counts();
        const auto& totals = slice.get_group_totals();
        const auto& counts = slice.get_group_counts();
        for (uint32_t i = 0; i < n_data; ++i) {
            fprintf(out_nobs, "%d\t%s\t%d\t%.1f\n",
                k, dataLabels[i].c_str(), n_units[i], totals[i]);
            for (int32_t m = 0; m < M; ++m) {
                size_t j = static_cast<size_t>(i * M + m);
                fprintf(out_sums, "%d\t%s\t%s\t%.1f\n",
                    k, dataLabels[i].c_str(), featureList[m].c_str(), counts[j]);
            }
        }
    }
    fclose(out_nobs);
    fclose(out_sums);

    return 0;
}
