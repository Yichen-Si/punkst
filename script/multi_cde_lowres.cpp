#include "punkst.h"
#include "tileoperator.hpp"
#include "dataunits.hpp"
#include "utils.h"
#include "lda.hpp"
#include "poisreg.hpp"
#include "mixpois.hpp"
#include <random>
#include <fstream>
#include <iomanip>
#include <memory>
#include <atomic>
#include <mutex>
#include <sstream>
#include <unordered_set>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include "Eigen/Dense"
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static void readContrastDesignFile(const std::string& contrastFile,
                                  std::vector<std::string>& inFile,
                                  std::vector<std::string>& metaFile,
                                  std::vector<std::vector<int32_t>>& contrasts,
                                  std::vector<std::string>& contrastNames);

enum class PoisLink { Log, Log1p };

static PoisLink parse_pois_link(const std::string& link) {
    if (link == "log") return PoisLink::Log;
    if (link == "log1p") return PoisLink::Log1p;
    error("Unknown link: %s (expected log or log1p)", link.c_str());
    return PoisLink::Log;
}

int32_t cmdConditionalTestPoisReg(int argc, char** argv) {

    std::string contrastFile, outPrefix, modelFile;
    std::vector<std::string> inFile, metaFile;
    std::vector<std::string> dataLabels;
    std::string link = "log1p";
    int32_t nThreads = 1;
    int32_t seed = -1;
    int32_t maxIter = 100;
    double  mDelta = 1e-3;
    int32_t minCount = 50;
    int32_t minUnits = -1;
    double  maxPval = 1;
    int32_t minCountFeature = 100;
    int32_t debug_ = 0, verbose = 1;
    uint32_t se_method = 1;
    double sizeFactor = 10000.0;
    bool robustSeFull = false;
    bool ldaUncertainty = false;
    OptimOptions optim;
    optim.tron.enabled = true;
    MLEOptions mleOpt;
    mleOpt.hc_type = 3;
    mleOpt.se_stabilize = 100;

    ParamList pl;
    pl.add_option("in-data", "Input data files", inFile)
      .add_option("in-meta", "Metadata files", metaFile)
      .add_option("labels", "Labels for the datasets to show in pairwise output", dataLabels)
      .add_option("contrast", "Contrast design TSV (in-data, in-meta, contrasts)", contrastFile)
      .add_option("model", "Model file", modelFile, true)
      .add_option("link", "Regression link: log or log1p", link)
      .add_option("out", "Output prefix", outPrefix, true)
      .add_option("threads", "Number of threads to use", nThreads)
      .add_option("seed", "Random seed", seed)
      .add_option("max-iter", "Max iterations for fitting cell type composition", maxIter)
      .add_option("mean-change-tol", "Convergence tolerance for fitting cell type composition", mDelta)
      .add_option("max-iter-inner", "(Pois. Reg) Maximum iterations for fitting DE parameters", optim.max_iters)
      .add_option("tol-inner", "Inner tolerance for fitting DE parameters", optim.tol)
      .add_option("min-count", "Minimum total count per unit to be included", minCount)
      .add_option("min-count-per-feature", "Min total count for a feature to be tested", minCountFeature)
      .add_option("min-units-per-feature", "Min number of units with nonzero count for a feature to be tested", minUnits)
      .add_option("size-factor", "Size factor for Poisson regression", sizeFactor)
      .add_option("max-pval", "Max p-value for output (default: 1)", maxPval)
      .add_option("se-method", "Method for calculating SE (1: Fisher, 2: Sandwich, 3: both)", se_method)
      .add_option("robust-se-full", "Use full robust SE matrix (default: diagonal approximation for the meat in the sandwich)", robustSeFull)
      .add_option("robust-se-hc", "Type of HC adjustment for robust SE (1,2,3)", mleOpt.hc_type)
      .add_option("robust-se-stabilize", "Stabilization parameter for robust SE (default: 100)", mleOpt.se_stabilize)
      .add_option("propagate-uncertainty", "Add delta-method correction for cell type proportion uncertainty (log1p only)", ldaUncertainty)
      .add_option("verbose", "Verbose", verbose)
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

    if (seed <= 0) {seed = std::random_device{}();}
    if (sizeFactor <= 0) {error("Size factor must be positive");}
    if (nThreads < 0) {nThreads = 1;}
    if (se_method < 1 || se_method > 3) {error("--se-method must be 1/2/3");}
    double minlog10p = -1;
    if (maxPval > 0 && maxPval < 1) {minlog10p = - std::log10(maxPval);}
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nThreads);

    PoisLink pois_link = parse_pois_link(link);
    const bool use_log1p = (pois_link == PoisLink::Log1p);

    mleOpt.optim = optim;
    mleOpt.se_flag = se_method;
    mleOpt.store_cov = false;
    const double b_null = 0.0;

    // Collect input files and contrasts
    std::vector<std::vector<int32_t>> contrasts;
    std::vector<std::string> contrastNames;
    int32_t C;
    if (!contrastFile.empty()) {
        if (!inFile.empty() || !metaFile.empty()) {
            error("--contrast cannot be combined with --in-data/--in-meta");
        }
        readContrastDesignFile(contrastFile, inFile, metaFile, contrasts, contrastNames);
        C = static_cast<int32_t>(contrasts.size());
        notice("Read %d contrasts from %s", C, contrastFile.c_str());
    }
    if (inFile.empty() || metaFile.empty() || inFile.size() != metaFile.size()) {
        error("Either --contrast or both --in-data and --in-meta must be specified");
    }
    int32_t G = static_cast<int32_t>(inFile.size());

    // Read / construct contrasts
    if (contrastFile.empty()) { // generate all pairwise contrasts
        dataLabels.resize(G);
        for (int32_t g = 0; g < G; ++g) {
            if (dataLabels[g].empty()) dataLabels[g] = std::to_string(g);
        }
        for (int32_t g = 0; g < G; ++g) {
            for (int32_t l = g+1; l < G; ++l) {
                contrasts.push_back(std::vector<int32_t>(G, 0));
                contrasts.back()[g] = -1; contrasts.back()[l] = 1;
                contrastNames.push_back(dataLabels[g] + "v" + dataLabels[l]);
            }
        }
        C = contrasts.size();
        notice("Will perform all pairwise contrasts (%d) among %d samples", C, G);
    }
    if (contrasts.empty()) {
        error("No contrasts defined");
    }

    // Read model
    LatentDirichletAllocation lda(modelFile, seed, nThreads);
    Eigen::MatrixXd beta = lda.get_model(); // K x M
    rowNormalizeInPlace(beta);
    Eigen::MatrixXd eta0;
    if (use_log1p) {
        eta0 = (beta.array() * sizeFactor + 1.).log();
    } else {
        eta0 = (beta.array().max(1e-8) * sizeFactor).log();
    }
    int32_t K = lda.get_n_topics();
    int32_t M = lda.get_n_features();
    std::vector<std::string>& factorNames = lda.topic_names_;
    std::vector<std::string>& featureNames = lda.feature_names_;
    if (minUnits <= 0) {minUnits = K * 10;}

    // Set up data readers
    std::vector<HexReader> readers;
    for (int32_t g = 0; g < G; ++g) {
        readers.emplace_back(metaFile[g]);
        readers.back().setFeatureIndexRemap(featureNames);
    }

    // Load data & compute theta
    std::vector<int32_t> nUnits(G);
    std::vector<std::vector<double>> cvecs(G), totct(G);
    std::vector<RowMajorMatrixXd> thetas(G);
    std::vector<std::vector<Document>> docsT(G);
    std::vector<int32_t> offsets(G + 1, 0);
    int32_t batchSize = 1024;
    std::vector<VectorXd> c_theta_sums(G, VectorXd::Zero(K));
    for (int32_t g = 0; g < G; ++g) {
        SparseObsMinibatchReader batch_reader(inFile[g], readers[g],
            minCount, sizeFactor, /*c=*/-1, debug_);
        std::vector<SparseObs> docs;
        RowMajorMatrixXd theta_g(0, K);
        int32_t theta_capacity = 0;
        int32_t nUnits_local = 0;
        docsT[g].assign(M, Document{});
        while (true) {
            int32_t n_batch = batch_reader.readBatch(docs, nullptr, batchSize);
            if (n_batch == 0) {break;}
            RowMajorMatrixXd theta_batch = lda.transform(docs);
            rowNormalizeInPlace(theta_batch);
            if (nUnits_local + n_batch > theta_capacity) {
                theta_capacity = std::max(theta_capacity + 2 * batchSize, nUnits_local + n_batch);
                theta_g.conservativeResize(theta_capacity, K);
            }
            theta_g.block(nUnits_local, 0, n_batch, K) = theta_batch;
            cvecs[g].reserve(static_cast<size_t>(nUnits_local + n_batch));
            totct[g].reserve(static_cast<size_t>(nUnits_local + n_batch));
            for (int32_t i = 0; i < n_batch; ++i) {
                totct[g].push_back(docs[i].ct_tot);
                cvecs[g].push_back(docs[i].c);
                const Document& doc = docs[i].doc;
                for (size_t t = 0; t < doc.ids.size(); ++t) {
                    const uint32_t m = doc.ids[t];
                    docsT[g][m].ids.push_back(nUnits_local + i);
                    docsT[g][m].cnts.push_back(doc.cnts[t]);
                }
            }
            nUnits_local += n_batch;
        }
        nUnits[g] = nUnits_local;
        notice("Read %d units from the %d-th data file", nUnits[g], g);
        theta_g.conservativeResize(nUnits[g], K);
        if (nUnits[g] == 0) {
            warning("No units passed the --min-count filter for the %d-th data file", g);
        } else {
            Eigen::Map<const ArrayXd> cvec_g(cvecs[g].data(), nUnits[g]);
            c_theta_sums[g] = (theta_g.array().colwise() * cvec_g).matrix().colwise().sum();
        }
        thetas[g] = std::move(theta_g);
        offsets[g + 1] = offsets[g] + nUnits[g];
    }

    const int32_t N_total = offsets.back();
    RowMajorMatrixXd A_all(N_total, K); // concatenate theta
    VectorXd cvec_all(N_total);
    for (int32_t g = 0; g < G; ++g) {
        int32_t ng = nUnits[g];
        int32_t offset = offsets[g];
        A_all.block(offset, 0, ng, K) = thetas[g];
        cvec_all.segment(offset, ng) =
            Eigen::Map<const VectorXd>(cvecs[g].data(), ng);
        RowMajorMatrixXd().swap(thetas[g]);
        std::vector<double>().swap(cvecs[g]);
    }
    notice("Total %d units across all %d datasets", N_total, G);

    for (int32_t c = 0; c < C; ++c) {
        notice("Processing contrast %d / %d", c + 1, C);
        int32_t n = 0;
        VectorXd xvec = VectorXd::Zero(N_total);
        VectorXd cvec_masked = VectorXd::Zero(N_total);
        VectorXd a_sum_ = VectorXd::Zero(K);
        VectorXd Ac;
        for (int32_t g = 0; g < G; ++g) {
            if (contrasts[c][g] == 0) {
                continue;
            }
            n += nUnits[g];
            int32_t ng = nUnits[g];
            int32_t offset = offsets[g];
            xvec.segment(offset, ng) =
                Eigen::VectorXd::Constant(ng, contrasts[c][g]);
            cvec_masked.segment(offset, ng) = cvec_all.segment(offset, ng);
            a_sum_ += c_theta_sums[g];
        }
        // For fitting baseline models
        OptimOptions baseline_optim = optim;
        baseline_optim.set_bounds(0.0, 1e30, K);
        MixPoisLink link = use_log1p ? MixPoisLink::Log1p : MixPoisLink::Log;
        MixPoisReg baseline(A_all, cvec_masked, a_sum_, baseline_optim, link);
        // Pre-compute values to reuse in log1p sparse optimization
        std::unique_ptr<MixPoisLog1pSparseContext> log1p_ctx;
        std::unique_ptr<tbb::enumerable_thread_specific<MixPoisLog1pSparseProblem>> log1p_tls;
        if (use_log1p) {
            log1p_ctx = std::make_unique<MixPoisLog1pSparseContext>(
                A_all, xvec, cvec_masked, mleOpt, robustSeFull, n,
                ldaUncertainty, sizeFactor);
            log1p_tls = std::make_unique<
                tbb::enumerable_thread_specific<MixPoisLog1pSparseProblem>>(
                [&log1p_ctx]() { return MixPoisLog1pSparseProblem(*log1p_ctx); });
        } else {
            Ac = A_all.transpose() * cvec_masked; // K
        }

        std::string out_path = outPrefix + "." + contrastNames[c] + ".tsv";
        std::ofstream out(out_path);
        if (!out) error("Failed to open output file: %s", out_path.c_str());
        std::string se_header = (se_method == 3) ? "SE\tSE0" : "SE";
        out << "Feature\tFactor\tBeta\t" << se_header
            << "\tlog10p\tDist2bd\tn\tCount1\tCount2\n";
        out << std::fixed << std::setprecision(6);
        notice("Fitting contrast %s with N=%d", contrastNames[c].c_str(), n);

        std::unique_ptr<tbb::global_control> debug_limit;
        if (debug_ > 0 && verbose > 1) {
            debug_limit = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism, 1);
        }

        std::unique_ptr<tbb::enumerable_thread_specific<VectorXd>> yvec_tls;
        std::unique_ptr<tbb::enumerable_thread_specific<std::vector<uint32_t>>> touched_tls;
        if (!use_log1p) {
            yvec_tls = std::make_unique<tbb::enumerable_thread_specific<VectorXd>>(
                [N_total]() { return VectorXd::Zero(N_total); });
            touched_tls = std::make_unique<
                tbb::enumerable_thread_specific<std::vector<uint32_t>>>();
        }

        // Randomly distribute features to threads
        std::vector<int32_t> perm_idx(M);
        for (int32_t j = 0; j < M; ++j) {perm_idx[j] = j;}
        std::mt19937 rng(seed);
        std::shuffle(perm_idx.begin(), perm_idx.end(), rng);
        int32_t features_done = 0;
        int32_t grain_size = std::max(1, M / (5 * nThreads));
        std::mutex io_mutex;
        tbb::parallel_for(tbb::blocked_range<int32_t>(0, M, grain_size),
            [&](const tbb::blocked_range<int32_t>& range)
        {
            int32_t thread_id = tbb::this_task_arena::current_thread_index();
            notice("Thread %d start processing %d features", thread_id, range.size());

            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(6);
            for (int32_t jj = range.begin(); jj < range.end(); ++jj) {
                int32_t j = perm_idx[jj];
                int32_t m = 0;
                double ysum0 = 0.0, ysum1 = 0.0;
                for (int32_t g = 0; g < G; ++g) {
                    if (contrasts[c][g] == 0) {continue;}
                    const auto& doc = docsT[g][j];
                    double ysum = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
                    m += static_cast<int32_t>(doc.ids.size());
                    if (contrasts[c][g] < 0) {
                        ysum0 += ysum;
                    } else {
                        ysum1 += ysum;
                    }
                }
                if (m < minUnits) {continue;}
                int32_t total_count = static_cast<int32_t>(ysum0 + ysum1);
                if (total_count < minCountFeature) {
                    continue;
                }
                VectorXd oK = eta0.col(j);
                Document ys;
                ys.ids.reserve(m);
                ys.cnts.reserve(m);
                for (int32_t g = 0; g < G; ++g) {
                    if (contrasts[c][g] == 0) {continue;}
                    const auto& doc = docsT[g][j];
                    int32_t offset = offsets[g];
                    for (size_t t = 0; t < doc.ids.size(); ++t) {
                        ys.ids.push_back(static_cast<uint32_t>(offset) + doc.ids[t]);
                        ys.cnts.push_back(doc.cnts[t]);
                    }
                }
                // Fit baseline (intercept)
                VectorXd b0 = baseline.fit_one(ys.ids, ys.cnts, oK);
                eta0.col(j) = b0;
if (verbose > 3)
{std::cout << "Thread " << thread_id << ": Fitted baseline for feature " << jj << "\n";}
                VectorXd b = VectorXd::Zero(K); // fitted coefficients
                VectorXd bd(K); // bounds
                MLEStats stats;
                double fval = 0.0;
                if (use_log1p) {
                    MLEOptions optLocal = mleOpt;
                    optLocal.optim.set_bounds(-b0, b0);
                    bd = b0;
                    auto& P = log1p_tls->local();
                    P.reset_feature(ys.ids, ys.cnts, b0);
                    fval = tron_solve(P, b, optLocal.optim, stats.optim);
                    P.compute_se(b, optLocal, stats);
                } else {
                    auto& yvec = yvec_tls->local();
                    auto& touched = touched_tls->local();
                    touched.clear();
                    for (size_t t = 0; t < ys.ids.size(); ++t) {
                        double cnt = ys.cnts[t];
                        yvec[ys.ids[t]] = cnt;
                    }
                    touched = std::move(ys.ids);
                    bd.setConstant(std::log(100.0));
                    for (int k = 0; k < K; ++k) {
                        double bcap_floor = (b0[k] - std::log(1e-8 * sizeFactor));
                        if (std::isfinite(bcap_floor) && bcap_floor > 0.0) {
                            bd[k] = std::min(bd[k], bcap_floor);
                        }
                    }
                    MLEOptions optLocal = mleOpt;
                    optLocal.optim.set_bounds(-bd, bd);
                    fval = mix_pois_log_mle(
                        A_all, yvec, xvec, cvec_masked, b0, Ac, optLocal, b, stats,
                        n);
                    for (int32_t idx : touched) {
                        yvec[idx] = 0.0;
                    }
                }
                (void)fval;
                for (int32_t k = 0; k < K; ++k) {
                    double se = -1, se2 = -1;
                    if (se_method == 3) {
                        se2 = stats.se_fisher[k];
                        se = stats.se_robust[k];
                    } else if (se_method == 2) {
                        se = stats.se_robust[k];
                    } else if (se_method == 1) {
                        se = stats.se_fisher[k];
                    }
                    double dist2bd = std::min(std::abs(b[k] - (-bd[k])), std::abs(b[k] - bd[k]));
                    double log10p = 0.;
                    if (std::abs(b[k] - b_null) < 1e-8) {
                        // Assume it is null
                    } else if (se <= 1e-8 || !std::isfinite(se)) {
                        std::lock_guard<std::mutex> guard(io_mutex);
                        warning("(%d, %d, %d) Invalid estimate (b=%g, se=%g)", c, k, j, b[k], se);
                    } else {
                        log10p = - log10_twosided_p_from_z((b[k] - b_null) / se);
                    }
                    if (log10p < minlog10p) {continue;}
                    oss << featureNames[j] << "\t" << factorNames[k]
                        << "\t" << b[k] << "\t" << se;
                    if (se_method == 3) {oss << "\t" << se2;}
                    oss << "\t" << log10p
                        << "\t" << dist2bd
                        << "\t" << m
                        << "\t" << (int32_t) (std::round(ysum0))
                        << "\t" << (int32_t) (std::round(ysum1)) << "\n";
if ((verbose > 1 && log10p > 10) || (verbose > 2 && m > 10*K)) {
std::cout << thread_id << "-[" << jj-range.begin() << "] "
<< featureNames[j] << "\t" << factorNames[k]
<< std::fixed << std::setprecision(6)
<< "\t" << b[k] << "\t" << se;
if (se_method == 3) {std::cout << "\t" << se2;}
std::cout << "\t" << log10p
<< "\t" << dist2bd << "\t" << m << "\n";
}
                }
            }
            if (verbose > 0) {
                std::lock_guard<std::mutex> guard(io_mutex);
                features_done += (range.end() - range.begin());
                notice("Contrast %s: processed %d / %d features",
                    contrastNames[c].c_str(), features_done, M);
            }
            std::string chunk = oss.str();
            if (!chunk.empty()) {
                std::lock_guard<std::mutex> guard(io_mutex);
                out << chunk;
                out.flush();
            }
        });

        out.close();
        notice("Wrote results to %s", out_path.c_str());

        out_path = outPrefix + "." + contrastNames[c] + ".eta0.tsv";
        RowMajorMatrixXd eta_out;
        if (use_log1p) {
            eta_out = eta0.transpose().array().exp() - 1.0;
        } else {
            eta_out = eta0.transpose().array().exp();
        }
        write_matrix_to_file(out_path, eta_out, 4, true, featureNames, "Feature", &factorNames);
    }

    return 0;
};


static void readContrastDesignFile(const std::string& contrastFile,
        std::vector<std::string>& inFile,
        std::vector<std::string>& metaFile,
        std::vector<std::vector<int32_t>>& contrasts,
        std::vector<std::string>& contrastNames) {
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
        error("Contrast file must have >= 3 columns (in-data, in-meta, contrast...): %s",
            contrastFile.c_str());
    }
    const int32_t C = static_cast<int32_t>(tokens.size() - 2);
    contrasts.assign(C, std::vector<int32_t>{});
    contrastNames.resize(C);
    std::unordered_set<std::string> seen_names;
    for (int32_t c = 0; c < C; ++c) {
        const std::string name = tokens[c + 2];
        if (name.empty()) {
            warning("Contrast %d has an empty header name; using index instead", c);
            contrastNames[c] = std::to_string(c);
        } else {
            if (!seen_names.insert(name).second) {
                error("Contrast name '%s' appears more than once in header", name.c_str());
            }
            contrastNames[c] = name;
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
        inFile.push_back(tokens[0]);
        metaFile.push_back(tokens[1]);
        for (int32_t c = 0; c < C; ++c) {
            int32_t y = 0;
            if (!str2int32(tokens[c + 2], y)) {
                error("Contrast value must be -1, 0, or 1: %s", tokens[c + 2].c_str());
            }
            if (y < -1 || y > 1) {
                error("Contrast value must be -1, 0, or 1: %d", y);
            }
            contrasts[c].push_back(y);
        }
        n_samples++;
    }
    if (n_samples == 0) {
        error("Contrast file has a header but no samples: %s", contrastFile.c_str());
    }
    for (int32_t c = 0; c < C; ++c) {
        if (contrasts[c].size() != static_cast<size_t>(n_samples)) {
            error("Contrast %s has %zu labels, expected %d samples",
                contrastNames[c].c_str(), contrasts[c].size(), n_samples);
        }
        int32_t n_neg = 0;
        int32_t n_pos = 0;
        for (int32_t s = 0; s < n_samples; ++s) {
            const int32_t y = contrasts[c][s];
            if (y < 0) {
                n_neg++;
            } else if (y > 0) {
                n_pos++;
            }
        }
        if (n_neg == 0 || n_pos == 0) {
            error("Contrast %s must have at least one sample in each group", contrastNames[c].c_str());
        }
    }
}
