#include "punkst.h"
#include "tileoperator.hpp"
#include "utils.h"
#include "glm.hpp"
#include <random>
#include <fstream>
#include <iomanip>
#include <memory>
#include <atomic>
#include <unordered_set>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

struct ContrastDef {
    std::string name;
    std::vector<int8_t> labels;
    std::vector<int32_t> group_neg;
    std::vector<int32_t> group_pos;
};

static void readContrastDesignFile(const std::string& contrastFile,
        std::vector<std::string>& annoPrefixes,
        std::vector<std::string>& ptsPrefixes,
        std::vector<ContrastDef>& contrasts) {
    std::ifstream ifs(contrastFile);
    if (!ifs.is_open()) {
        error("Cannot open contrast file: %s", contrastFile.c_str());
    }
    std::string line;
    std::vector<std::string> tokens;
    if (!std::getline(ifs, line)) {
        error("Contrast file is empty: %s", contrastFile.c_str());
    }
    split(tokens, "\t ", line, UINT_MAX, true, true, true);
    if (tokens.size() < 3) {
        error("Contrast file must have >= 3 columns (anno, pts, contrast...): %s", contrastFile.c_str());
    }
    const int32_t C = static_cast<int32_t>(tokens.size() - 2);
    contrasts.assign(C, ContrastDef{});
    std::unordered_set<std::string> seen_names;
    for (int32_t c = 0; c < C; ++c) {
        const std::string name = tokens[c + 2];
        if (name.empty()) {
            warning("Contrast %d has an empty header name; using index instead", c);
            contrasts[c].name = std::to_string(c);
        } else {
            if (!seen_names.insert(name).second) {
                error("Contrast name '%s' appears more than once in header", name.c_str());
            }
            contrasts[c].name = name;
        }
    }

    int32_t n_samples = 0;
    while (std::getline(ifs, line)) {
        if (line.empty() || line[0] == '#') { continue; }
        split(tokens, "\t ", line, UINT_MAX, true, true, true);
        if (tokens.size() != static_cast<size_t>(C + 2)) {
            error("Contrast file row has %zu columns, expected %d: %s",
                tokens.size(), C + 2, line.c_str());
        }
        annoPrefixes.push_back(tokens[0]);
        ptsPrefixes.push_back(tokens[1]);
        for (int32_t c = 0; c < C; ++c) {
            int32_t y = 0;
            if (!str2int32(tokens[c + 2], y)) {
                error("Contrast value must be -1, 0, or 1: %s", tokens[c + 2].c_str());
            }
            if (y < -1 || y > 1) {
                error("Contrast value must be -1, 0, or 1: %d", y);
            }
            contrasts[c].labels.push_back(static_cast<int8_t>(y));
        }
        n_samples++;
    }
    if (n_samples == 0) {
        error("Contrast file has a header but no samples: %s", contrastFile.c_str());
    }
    for (int32_t c = 0; c < C; ++c) {
        if (contrasts[c].labels.size() != static_cast<size_t>(n_samples)) {
            error("Contrast %s has %zu labels, expected %d samples",
                contrasts[c].name.c_str(), contrasts[c].labels.size(), n_samples);
        }
        contrasts[c].group_neg.clear();
        contrasts[c].group_pos.clear();
        for (int32_t s = 0; s < n_samples; ++s) {
            const int8_t y = contrasts[c].labels[s];
            if (y < 0) {
                contrasts[c].group_neg.push_back(s);
            } else if (y > 0) {
                contrasts[c].group_pos.push_back(s);
            }
        }
        if (contrasts[c].group_neg.empty() || contrasts[c].group_pos.empty()) {
            error("Contrast %s must have at least one sample in each group", contrasts[c].name.c_str());
        }
        notice("Read contrast %s with %zu vs %zu samples", contrasts[c].name.c_str(), contrasts[c].group_pos.size(), contrasts[c].group_neg.size());
    }
}

static void buildPairwiseContrasts(const std::vector<std::string>& dataLabels,
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
            contrasts.push_back(contrast);
        }
    }
}

/**
 * Join pixel level decoding results with original transcripts and perform
 * cluster/factor specific (conditional) DE test between groups
 */
int32_t cmdConditionalTest(int32_t argc, char** argv) {
    std::vector<std::string> inPrefix, inData, inIndex, inPtsPrefix;
    std::vector<std::string> dataLabels;
    std::string dictFile, outPrefix, outFile, contrastFile;
    bool isBinary = false;
    double gridSize;
    int32_t K;
    int32_t icol_x, icol_y, icol_feature, icol_val;
    float qxmin, qxmax, qymin, qymax;
    bool bounded = false;
    double minCount = 10.0;
    double minCountPerFeature = 100.0;
    double minPval_output = 1;
    double deconv_hit_p = 1;
    double minOR_output = 1;
    double minOR = 1.2;
    double minProb = 0.01;
    int32_t debug_ = 0;
    int32_t nThreads = 1;
    int32_t nPerm = 0;
    uint64_t seed = 1;
    std::vector<ContrastDef> contrasts;

    ParamList pl;
    pl.add_option("anno-data", "Input pixel files", inData)
      .add_option("anno-index", "Input pixel index files", inIndex)
      .add_option("anno", "Prefixes of input pixel data files", inPrefix)
      .add_option("labels", "Labels for the datasets to show in pairwise output", dataLabels)
      .add_option("binary", "Data file is in binary format", isBinary)
      .add_option("K", "Number of factors in the annotation files", K, true)
      .add_option("pts", "Prefixes of the transcript data files", inPtsPrefix)
      .add_option("contrast", "Contrast design TSV (anno, pts, contrasts)", contrastFile)
      .add_option("features", "List of features to test", dictFile, true)
      .add_option("grid-size", "Grid size", gridSize, true)
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
      .add_option("min-prob", "Minimum probability for a pixel to be included (softly) in a slice", minProb)
      .add_option("min-count-per-feature", "Minimum total count for a feature to be considered", minCountPerFeature)
      .add_option("max-pval", "Max p-value for output (default: 1)", minPval_output)
      .add_option("max-pval-deconv", "If at least two slices reach this p-value, do deconvolution", deconv_hit_p)
      .add_option("min-or", "Minimum odds ratio for output (default: 1)", minOR_output)
      .add_option("min-or-perm", "Minimum odds ratio for doing permutation (default: 1.2)", minOR)
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

    // Collect input files and contrasts
    if (!contrastFile.empty()) {
        if (!inPrefix.empty() || !inData.empty() || !inIndex.empty() || !inPtsPrefix.empty()) {
            error("--contrast cannot be combined with --anno/--anno-data/--anno-index/--pts");
        }
        readContrastDesignFile(contrastFile, inPrefix, inPtsPrefix, contrasts);
    }
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
    if (inPtsPrefix.empty() || inPtsPrefix.size() != inData.size()) {
        error("--pts must match with --anno/--anno-data");
    }
    uint32_t n_data = static_cast<uint32_t>(inData.size());
    dataLabels.resize(n_data);
    for (uint32_t i = 0; i < n_data; ++i) {
        if (dataLabels[i].empty())
            dataLabels[i] = std::to_string(i);
    }
    if (contrastFile.empty()) {
        buildPairwiseContrasts(dataLabels, contrasts);
    }
    if (contrasts.empty()) {
        error("No contrasts defined");
    }

    if (gridSize <= 0) {error("--grid-size must be positive");}
    if (K <= 0) {error("--K must be positive");}
    if (bounded && (qxmin >= qxmax || qymin >= qymax)) {
        error("Invalid bounding box specified");
    }
    const double minLog10p = - std::log10(minPval_output);
    const double min_log_or = std::log(minOR_output);

    // Set up data readers
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
    PairwiseBinomRobust statUnion(n_data, M, minCount);
    MultiSliceUnitCache unitCache(K, M, minCount);
    struct LocalAgg {
        MultiSlicePairwiseBinom stat;
        MultiSliceUnitCache cache;
        PairwiseBinomRobust statu;
        LocalAgg(int K, int G, int M, double minCount)
            : stat(K, G, M, minCount), cache(K, M, minCount), statu(G, M, minCount) {}
    };

    // Process each dataset and collect sufficient statistics
    for (uint32_t i = 0; i < n_data; ++i) {
        auto& tileOp = *tileOps[i];
        const auto& tileList = tileOp.getTileInfo();
        int32_t nTiles = static_cast<int32_t>(tileList.size());
        notice("Processing %d tiles for dataset %u", nTiles, i);
        auto& reader = readers[i];
        std::atomic<int32_t> processed{0};
        tbb::enumerable_thread_specific<LocalAgg> tls([&] {
            return LocalAgg(K, (int)n_data, M, minCount);
        });
        int32_t topk = tileOp.getK();
        size_t ntasks = tileList.size();
        if (debug_) {
            ntasks = std::min<size_t>(ntasks, debug_ * nThreads);
        }
        tbb::parallel_for(tbb::blocked_range<size_t>(0, ntasks),
            [&](const tbb::blocked_range<size_t>& range)
        {
            Eigen::MatrixXd confusion = Eigen::MatrixXd::Zero(K, K);
            auto& local = tls.local();
            double p_residual = 0.0;
            for (size_t ti = range.begin(); ti != range.end(); ++ti) {
                const auto& tileInfo = tileList[ti];
                TileKey tile{tileInfo.row, tileInfo.col};
                std::map<std::pair<int32_t, int32_t>, TopProbs> pixelMap;
                tileOp.loadTileToMap(tile, pixelMap);
                for (const auto& kv : pixelMap) {
                    const auto& tp = kv.second;
                    double r = 1.;
                    for (size_t ii = 0; ii < tp.ks.size(); ++ii) {
                        r -= tp.ps[ii];
                        for (size_t jj = ii; jj < tp.ks.size(); ++jj) {
                            confusion(tp.ks[ii], tp.ks[jj]) += tp.ps[ii] * tp.ps[jj];
                        }
                    }
                    if (r <= 0) {continue;}
                    p_residual += r * r;
                }
                auto tileAgg = tileOp.aggOneTile(pixelMap, reader, parser, tile, gridSize, minProb, K);
                for (const auto& kv : tileAgg) {
                    const int32_t k = kv.first;
                    if (k < 0 || k > K) continue;
                    if (k == K) {
                        for (const auto& unitKv : kv.second) {
                            const auto& obs = unitKv.second;
                            local.statu.add_unit(static_cast<int>(i), obs.totalCount, obs.featureCounts);
                        }
                        continue;
                    }
                    for (const auto& unitKv : kv.second) {
                        const auto& obs = unitKv.second;
                        local.stat.slice(k).add_unit(static_cast<int>(i), obs.totalCount, obs.featureCounts);
                        if (nPerm > 0) {
                            local.cache.add_unit(k, static_cast<int>(i), obs.totalCount, obs.featureCounts);
                        }
                    }
                }
                int32_t done = processed.fetch_add(1) + 1;
                if (done % 10 == 0) {
                    notice("... processed %d / %d tiles for dataset %u", done, nTiles, i);
                }
            }
            confusion += confusion.transpose().eval();
            p_residual = p_residual / K / K;
            for (int k = 0; k < K; ++k) {
                confusion(k, k) /= 2.0;
                for (int l = 0; l < K; ++l) {
                    confusion(k, l) += p_residual;
                }
            }
            local.stat.add_to_confusion(i, confusion);
        });
        for (auto& local : tls) {
            statOp.merge_from(local.stat);
            unitCache.merge_from(local.cache);
            statUnion.merge_from(local.statu);
        }
    }
    notice("Finished collecting sufficient statistics from all datasets");
    statOp.finished_adding_data();

    const double pi_eps = 1e-8;
    const double min_log_or_perm = std::log(minOR);
    struct TestRec {
        int k; int f;
        double beta; double log10p;
        double pi0; double pi1;
        double total_count;
        double beta_global; double log10p_global; double log10p_dev;
        double beta_deconv; double log10p_deconv;
        double p_perm; bool perm_candidate;
    };
    const auto& active = statOp.get_active_features();
    for (const auto& contrast : contrasts) {
        ContrastPrecomp pc = statOp.prepare_contrast(contrast.group_neg, contrast.group_pos);

        std::vector<TestRec> tests;
        const size_t n_tests_guess = (size_t)(K * active.size());
        tests.reserve(n_tests_guess);
        notice("Contrast %s with %d active features", contrast.name.c_str(), (int32_t) active.size());
        for (int f : active) {
            PairwiseBinomRobust::PairwiseOneResult mu;
            if (!statUnion.compute_one_test_aggregate(
                    f, contrast.group_neg, contrast.group_pos, mu,
                    minCountPerFeature, pi_eps, true)) {
                continue;
            }
            MultiSliceOneResult ms(K);
            if (!statOp.compute_one_test_aggregate(
                    f, contrast.group_neg, contrast.group_pos, pc, ms,
                    minCountPerFeature, pi_eps, true, deconv_hit_p)) {
                continue;
            }

            std::vector<uint8_t> ok(K, 0);
            int ok_count = 0;

            for (int k = 0; k < K; ++k) {
                if (!ms.slice_ok[(size_t)k]) {continue;}
                ok[k] = 1;
                ok_count++;
            }
            if (ok_count == 0) {continue;}

            double g = mu.beta, varg = mu.varb;
            double log10p_g = -log10_twosided_p_from_z(g / std::sqrt(varg));
            for (int k = 0; k < K; ++k) {
                if (!ok[k] || std::abs(ms.beta_obs(k)) < min_log_or) {continue;}
                if (ms.log10p_obs(k) < minLog10p) {continue;}

                double total_count = 0.0;
                {
                    const auto& counts = statOp.slice(k).get_group_counts();
                    double y0 = 0.0;
                    double y1 = 0.0;
                    for (int32_t g0 : contrast.group_neg) {
                        size_t idx = static_cast<size_t>(g0) * M + f;
                        y0 += counts[idx];
                    }
                    for (int32_t g1 : contrast.group_pos) {
                        size_t idx = static_cast<size_t>(g1) * M + f;
                        y1 += counts[idx];
                    }
                    total_count = y0 + y1;
                }

                double log10p_dev = -1.0;
                double beta_dev = 0.0;
                if (ok_count >= 2 && varg > 0.0) {
                    beta_dev = ms.beta_obs(k) - g;
                    const double se_d = std::sqrt(ms.varb_obs(k) + varg);
                    log10p_dev = -log10_twosided_p_from_z(beta_dev / se_d);
                }

                double beta_deconv = 0.0;
                double log10p_deconv = -1.0;
                if (ms.deconv_ok) {
                    beta_deconv = ms.beta_deconv(k);
                    const double se = std::sqrt(ms.varb_obs(k));
                    if (se > 0.0 && std::isfinite(se)) {
                        log10p_deconv = -log10_twosided_p_from_z(beta_deconv / se);
                    }
                }

                TestRec rec{k, f, ms.beta_obs(k), ms.log10p_obs(k),
                            ms.pi0_obs(k), ms.pi1_obs(k),
                            total_count, g, log10p_g, log10p_dev,
                            beta_deconv, log10p_deconv,
                            -1.0, false};
                rec.perm_candidate = (std::abs(rec.beta) >= min_log_or_perm);
                tests.push_back(rec);
            }
        }

        if (nPerm > 0 && !tests.empty()) {
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

            const std::string contrastName = contrast.name;
            std::vector<double> Ntot(K, 0.0); // total counts across two groups
            std::vector<int> n1_units(K, 0), n_units(K, 0);
            std::vector<std::vector<uint32_t>> eligible_units(K);
            for (int k = 0; k < K; ++k) {
                const auto& sl = statOp.slice(k);
                Ntot[k] = sl.sum_group_totals(contrast.group_neg) +
                    sl.sum_group_totals(contrast.group_pos);
                n1_units[k] = sl.sum_group_unit_counts(contrast.group_pos);
                n_units[k]  = sl.sum_group_unit_counts(contrast.group_neg) +
                    sl.sum_group_unit_counts(contrast.group_pos);
                const auto& units = unitCache.slice_units(k);
                eligible_units[k].reserve(units.size());
                int cached_n = 0;
                for (uint32_t u = 0; u < units.size(); ++u) {
                    const int32_t g = units[u].group;
                    if (g < 0 || g >= static_cast<int32_t>(contrast.labels.size())) continue;
                    const int8_t lab = contrast.labels[g];
                    if (lab == 0) continue;
                    eligible_units[k].push_back(u);
                    cached_n++;
                }
                if (cached_n != n_units[k]) {
                    warning("Slice %d cache units (%d) != observed units (%d) for contrast %s. ",
                        k, cached_n, n_units[k], contrastName.c_str());
                }
            }

            int32_t n_candidate = 0;
            for (const auto& rec : tests) {
                if (rec.perm_candidate) n_candidate++;
            }
            notice("Permutation: %d tests (slice, feature) to evaluate for contrast %s", n_candidate, contrastName.c_str());

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
                        const auto& eligible = eligible_units[k];
                        const int N = (int)eligible.size();
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
                            const auto& U = units[eligible[u]];
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
                        if (!tests[j].perm_candidate) continue;
                        const int k = tests[j].k;
                        const int f = tests[j].f;
                        const double N1p = T.N1[k];
                        const double N0p = Ntot[k] - N1p;
                        if (N0p <= 0.0 || N1p <= 0.0) continue;

                        const double Y1p = T.Y1[(size_t)(k*M+f)];
                        const double Y0p = tests[j].total_count - Y1p;
                        const double pi0p = clamp(Y0p / N0p, pi_eps, 1.0 - pi_eps);
                        const double pi1p = clamp(Y1p / N1p, pi_eps, 1.0 - pi_eps);
                        const double beta_p = logit(pi1p) - logit(pi0p);
                        if (!std::isfinite(beta_p)) continue;

                        if (std::abs(beta_p) >= std::abs(tests[j].beta)) {
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

            for (size_t j = 0; j < tests.size(); ++j) {
                if (!tests[j].perm_candidate) continue;
                tests[j].p_perm = (double)exceed[j] / (double)nPerm;
            }
        }

        outFile = outPrefix + "." + contrast.name + ".tsv";
        FILE* out_stream = fopen(outFile.c_str(), "w");
        if (!out_stream) {error("Cannot open output file: %s", outFile.c_str());}
        std::string header = "Slice\tFeature\tBeta\tlog10p\tPi0\tPi1\tTotalCount\tBeta_global\tlog10p_global\tlog10p_deviation\tBeta_deconv\tlog10p_deconv";
        if (nPerm > 0) {
            fprintf(out_stream, "%s\tp_perm\n", header.c_str());
        } else {
            fprintf(out_stream, "%s\n", header.c_str());
        }

        for (const auto& rec : tests) {
            fprintf(out_stream, "%d\t%s\t%.4e\t%.4f\t%.4e\t%.4e\t%.1f\t%.4e\t%.4f\t%.4f\t%.4e\t%.4f",
                rec.k, featureList[rec.f].c_str(),
                rec.beta, rec.log10p, rec.pi0, rec.pi1, rec.total_count,
                rec.beta_global, rec.log10p_global, rec.log10p_dev, rec.beta_deconv, rec.log10p_deconv);
            if (nPerm > 0) {
                fprintf(out_stream, "\t%.4f", rec.p_perm);
            }
            fprintf(out_stream, "\n");
        }
        fclose(out_stream);
        notice("Result for %s is written to:\n  %s", contrast.name.c_str(),  outFile.c_str());
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
