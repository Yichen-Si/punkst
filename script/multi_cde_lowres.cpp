#include "punkst.h"
#include "multi_cde_lowres_common.hpp"
#include "poisreg.hpp"
#include "mixpois.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <random>
#include <sstream>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

int32_t cmdConditionalTestPoisReg(int argc, char** argv) {

    std::string contrastFile, outPrefix, modelFile;
    std::vector<std::string> inFile, metaFile;
    std::vector<std::string> dataLabels;
    LowresScaleOptions scale;
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
      .add_option("link", "Regression link: log or log1p", scale.link)
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
      .add_option("size-factor", "Size factor for Poisson regression", scale.sizeFactor)
      .add_option("c", "Constant c in log(1+lambda/c)", scale.cScale)
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
        pl.print_help_noexit();
        return 1;
    }
    if (debug_ > 0) {
        logger::Logger::getInstance().setLevel(logger::LogLevel::DEBUG);
    }

    if (seed <= 0) {seed = std::random_device{}();}
    scale.validate(ldaUncertainty);
    if (nThreads < 0) {nThreads = 1;}
    if (se_method < 1 || se_method > 3) {error("--se-method must be 1/2/3");}
    double minlog10p = -1;
    if (maxPval > 0 && maxPval < 1) {minlog10p = - std::log10(maxPval);}
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nThreads);

    const bool use_log1p = scale.useLog1p();
    mleOpt.optim = optim;
    mleOpt.se_flag = se_method;
    mleOpt.store_cov = false;
    const double b_null = 0.0;

    LowresContrastDesign design =
        loadLowresContrastDesign(contrastFile, inFile, metaFile, dataLabels);

    RowMajorMatrixXd eta0;
    LowresInputData data = loadLowresInputData(
        modelFile, seed, nThreads, design, scale, minCount, debug_, eta0);
    const int32_t K = data.K;
    const int32_t M = data.M;
    if (minUnits <= 0) {minUnits = K * 10;}

    for (int32_t c = 0; c < design.nContrasts(); ++c) {
        notice("Processing contrast %d / %d", c + 1, design.nContrasts());
        const auto& contrast = design.contrasts[c];
        const std::string& contrastName = design.contrastNames[c];
        LowresContrastData cd = buildLowresContrastData(data, contrast);
        VectorXd Ac;

        OptimOptions baseline_optim = optim;
        baseline_optim.set_bounds(0.0, 1e30, K);
        MixPoisLink link = use_log1p ? MixPoisLink::Log1p : MixPoisLink::Log;
        MixPoisReg baseline(data.A_all, cd.cvecMasked, cd.aSum, baseline_optim, link);

        std::unique_ptr<MixPoisLog1pSparseContext> log1p_ctx;
        std::unique_ptr<tbb::enumerable_thread_specific<MixPoisLog1pSparseProblem>> log1p_tls;
        if (use_log1p) {
            log1p_ctx = std::make_unique<MixPoisLog1pSparseContext>(
                data.A_all, cd.xvec, cd.cvecMasked, mleOpt, robustSeFull, cd.n,
                ldaUncertainty, scale.sizeFactor);
            log1p_tls = std::make_unique<
                tbb::enumerable_thread_specific<MixPoisLog1pSparseProblem>>(
                [&log1p_ctx]() { return MixPoisLog1pSparseProblem(*log1p_ctx); });
        } else {
            Ac = data.A_all.transpose() * cd.cvecMasked;
        }

        std::string out_path = outPrefix + "." + contrastName + ".tsv";
        std::ofstream out(out_path);
        if (!out) error("Failed to open output file: %s", out_path.c_str());
        std::string se_header = (se_method == 3) ? "SE\tSE0" : "SE";
        out << "Feature\tFactor\tBeta\t" << se_header
            << "\tlog10p\tDist2bd\tn\tCount1\tCount2\n";
        out << std::fixed << std::setprecision(6);
        notice("Fitting contrast %s with N=%d", contrastName.c_str(), cd.n);

        std::unique_ptr<tbb::global_control> debug_limit;
        if (debug_ > 0 && verbose > 1) {
            debug_limit = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism, 1);
        }

        std::unique_ptr<tbb::enumerable_thread_specific<VectorXd>> yvec_tls;
        std::unique_ptr<tbb::enumerable_thread_specific<std::vector<uint32_t>>> touched_tls;
        if (!use_log1p) {
            yvec_tls = std::make_unique<tbb::enumerable_thread_specific<VectorXd>>(
                [&data]() { return VectorXd::Zero(data.N_total); });
            touched_tls = std::make_unique<
                tbb::enumerable_thread_specific<std::vector<uint32_t>>>();
        }

        std::vector<int32_t> perm_idx = shuffledFeatureOrder(M, seed);
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
                const int32_t j = perm_idx[jj];
                LowresFeatureObs obs;
                if (!buildLowresFeatureObs(data, contrast, j, minUnits, minCountFeature, obs)) {
                    continue;
                }
                VectorXd oK = eta0.col(j);
                VectorXd b0 = baseline.fit_one(obs.y.ids, obs.y.cnts, oK);
                eta0.col(j) = b0;
                if (verbose > 3) {
                    std::cout << "Thread " << thread_id
                              << ": Fitted baseline for feature " << jj << "\n";
                }

                VectorXd b = VectorXd::Zero(K);
                VectorXd bd(K);
                MLEStats stats;
                double fval = 0.0;
                if (use_log1p) {
                    MLEOptions optLocal = mleOpt;
                    optLocal.optim.set_bounds(-b0, b0);
                    bd = b0;
                    auto& P = log1p_tls->local();
                    P.reset_feature(obs.y.ids, obs.y.cnts, b0);
                    fval = tron_solve(P, b, optLocal.optim, stats.optim);
                    P.compute_se(b, optLocal, stats);
                } else {
                    auto& yvec = yvec_tls->local();
                    auto& touched = touched_tls->local();
                    touched.clear();
                    for (size_t t = 0; t < obs.y.ids.size(); ++t) {
                        yvec[obs.y.ids[t]] = obs.y.cnts[t];
                    }
                    touched = obs.y.ids;
                    bd.setConstant(std::log(100.0));
                    for (int k = 0; k < K; ++k) {
                        double bcap_floor = (b0[k] - std::log(1e-8 * scale.sizeFactor));
                        if (std::isfinite(bcap_floor) && bcap_floor > 0.0) {
                            bd[k] = std::min(bd[k], bcap_floor);
                        }
                    }
                    MLEOptions optLocal = mleOpt;
                    optLocal.optim.set_bounds(-bd, bd);
                    fval = mix_pois_log_mle(
                        data.A_all, yvec, cd.xvec, cd.cvecMasked, b0, Ac,
                        optLocal, b, stats, cd.n);
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
                    double log10p = 0.0;
                    if (std::abs(b[k] - b_null) < 1e-8) {
                        // Assume it is null
                    } else if (se <= 1e-8 || !std::isfinite(se)) {
                        std::lock_guard<std::mutex> guard(io_mutex);
                        warning("(%d, %d, %d) Invalid estimate (b=%g, se=%g)", c, k, j, b[k], se);
                    } else {
                        log10p = - log10_twosided_p_from_z((b[k] - b_null) / se);
                    }
                    if (log10p < minlog10p) {continue;}
                    oss << data.featureNames[j] << "\t" << data.factorNames[k]
                        << "\t" << b[k] << "\t" << se;
                    if (se_method == 3) {oss << "\t" << se2;}
                    oss << "\t" << log10p
                        << "\t" << dist2bd
                        << "\t" << obs.nnz
                        << "\t" << static_cast<int32_t>(std::round(obs.ysum0))
                        << "\t" << static_cast<int32_t>(std::round(obs.ysum1)) << "\n";
                    if ((verbose > 1 && log10p > 10) || (verbose > 2 && obs.nnz > 10 * K)) {
                        std::cout << thread_id << "-[" << jj - range.begin() << "] "
                                  << data.featureNames[j] << "\t" << data.factorNames[k]
                                  << std::fixed << std::setprecision(6)
                                  << "\t" << b[k] << "\t" << se;
                        if (se_method == 3) {std::cout << "\t" << se2;}
                        std::cout << "\t" << log10p
                                  << "\t" << dist2bd << "\t" << obs.nnz << "\n";
                    }
                }
            }
            if (verbose > 0) {
                std::lock_guard<std::mutex> guard(io_mutex);
                features_done += (range.end() - range.begin());
                notice("Contrast %s: processed %d / %d features",
                    contrastName.c_str(), features_done, M);
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

        out_path = outPrefix + "." + contrastName + ".eta0.tsv";
        writeLowresEta0(out_path, eta0, use_log1p, data.featureNames, data.factorNames);
    }

    return 0;
}
