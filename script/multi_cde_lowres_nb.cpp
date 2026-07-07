#include "punkst.h"
#include "multi_cde_lowres_common.hpp"
#include "poisreg.hpp"
#include "mixpois.hpp"
#include "numerical_utils.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <sstream>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

int32_t cmdConditionalTestNbReg(int argc, char** argv) {

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
    double dispLoessSpan = 0.2; // DESeq2 default (?)
    double dispOutlierTau = 2.0;
    double alphaMin = 1e-8;
    double alphaMax = 1e4;
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
      .add_option("max-iter-inner", "(NB Reg) Maximum iterations for fitting DE parameters", optim.max_iters)
      .add_option("tol-inner", "Inner tolerance for fitting DE parameters", optim.tol)
      .add_option("min-count", "Minimum total count per unit to be included", minCount)
      .add_option("min-count-per-feature", "Min total count for a feature to be tested", minCountFeature)
      .add_option("min-units-per-feature", "Min number of units with nonzero count for a feature to be tested", minUnits)
      .add_option("size-factor", "Size factor", scale.sizeFactor)
      .add_option("c", "Constant c in log(1+lambda/c)", scale.cScale)
      .add_option("max-pval", "Max p-value for output (default: 1)", maxPval)
      .add_option("se-method", "Method for calculating SE (1: Fisher, 2: Sandwich, 3: both)", se_method)
      .add_option("robust-se-hc", "Type of HC adjustment for robust SE (1,2,3)", mleOpt.hc_type)
      .add_option("robust-se-stabilize", "Stabilization parameter for robust SE (default: 100)", mleOpt.se_stabilize)
      .add_option("dispersion-loess-span", "LOESS span for dispersion trend (0-1)", dispLoessSpan)
      .add_option("dispersion-outlier-tau", "Outlier threshold in MAD units", dispOutlierTau)
      .add_option("alpha-min", "Minimum dispersion clamp", alphaMin)
      .add_option("alpha-max", "Maximum dispersion clamp", alphaMax)
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
    scale.validate();
    if (nThreads < 0) {nThreads = 1;}
    if (se_method < 1 || se_method > 3) {error("--se-method must be 1/2/3");}
    if (dispLoessSpan <= 0.0 || dispLoessSpan > 1.0) {error("--dispersion-loess-span must be in (0,1]");}
    if (dispOutlierTau <= 0.0) {error("--dispersion-outlier-tau must be positive");}
    if (alphaMin <= 0.0 || alphaMax <= 0.0 || alphaMin > alphaMax) {
        error("--alpha-min/max must be positive with min <= max");
    }
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
        const auto& contrast = design.contrasts[c];
        const std::string& contrastName = design.contrastNames[c];
        LowresContrastData cd = buildLowresContrastData(data, contrast);
        if (cd.cSum <= 0.0) {
            warning("Contrast %s has zero total count", contrastName.c_str());
            continue;
        }
        VectorXd Ac;

        OptimOptions baseline_optim = optim;
        baseline_optim.set_bounds(0.0, 1e30, K);
        MixPoisLink link = use_log1p ? MixPoisLink::Log1p : MixPoisLink::Log;
        MixPoisReg baseline(data.A_all, cd.cvecMasked, cd.aSum, baseline_optim, link);

        VectorXd c2vec = cd.cvecMasked.array().square().matrix();
        MatrixXd Mmat = data.A_all.transpose()
            * (data.A_all.array().colwise() * c2vec.array()).matrix();

        std::unique_ptr<MixNBLog1pSparseContext> nb_log1p_ctx;
        std::unique_ptr<tbb::enumerable_thread_specific<MixNBLog1pSparseApproxProblem>> nb_log1p_tls;
        if (use_log1p) {
            nb_log1p_ctx = std::make_unique<MixNBLog1pSparseContext>(
                data.A_all, cd.xvec, cd.cvecMasked, cd.n);
            nb_log1p_tls = std::make_unique<
                tbb::enumerable_thread_specific<MixNBLog1pSparseApproxProblem>>(
                [&nb_log1p_ctx, &mleOpt]() {
                    return MixNBLog1pSparseApproxProblem(
                        *nb_log1p_ctx, mleOpt.optim, mleOpt.ridge, mleOpt.soft_tau);
                });
        } else {
            Ac = data.A_all.transpose() * cd.cvecMasked;
        }

        std::vector<double> baseMean(M, 0.0);
        std::vector<double> dispGene(M, 0.0);
        std::vector<double> dispTrend(M, 0.0);
        std::vector<double> dispFixed(M, 0.0);
        std::vector<uint8_t> ok(M, 0);
        std::vector<uint8_t> outlier(M, 0);
        std::vector<LowresFeatureObs> featureObs(M);

        std::vector<int32_t> perm_idx = shuffledFeatureOrder(M, seed);
        int32_t grain_size = std::max(1, M / (5 * nThreads));
        std::mutex io_mutex;

        int32_t features_done = 0;
        notice("Contrast %s: Stage A (fit null + dispersion)", contrastName.c_str());
        tbb::parallel_for(tbb::blocked_range<int32_t>(0, M, grain_size),
            [&](const tbb::blocked_range<int32_t>& range)
        {
            int32_t thread_id = tbb::this_task_arena::current_thread_index();
            notice("Thread %d start processing %d features", thread_id, range.size());

            for (int32_t jj = range.begin(); jj < range.end(); ++jj) {
                const int32_t j = perm_idx[jj];
                LowresFeatureObs obs;
                if (!buildLowresFeatureObs(data, contrast, j, minUnits, minCountFeature, obs)) {
                    continue;
                }
                VectorXd oK = eta0.col(j);
                VectorXd b0 = baseline.fit_one(obs.y.ids, obs.y.cnts, oK);
                eta0.col(j) = b0;

                baseMean[j] = static_cast<double>(obs.totalCount) / cd.cSum;
                if (baseMean[j] <= 0.0) {continue;}

                VectorXd beta(K);
                if (use_log1p) {
                    beta = (b0.array().min(40.0).exp() - 1.0).matrix();
                } else {
                    beta = (b0.array().min(40.0).exp()).matrix();
                }
                double sum_lam2 = beta.dot(Mmat * beta);
                sum_lam2 = std::max(sum_lam2, 1e-12);
                double sum_y2 = 0.0;
                double sum_y_lam = 0.0;
                for (size_t t = 0; t < obs.y.ids.size(); ++t) {
                    double cnt = obs.y.cnts[t];
                    sum_y2 += cnt * cnt;
                    int32_t idx = static_cast<int32_t>(obs.y.ids[t]);
                    double lam = cd.cvecMasked[idx] * data.A_all.row(idx).dot(beta);
                    sum_y_lam += cnt * lam;
                }
                double numer = sum_y2 - 2.0 * sum_y_lam + sum_lam2 - obs.totalCount;
                double alpha = numer / sum_lam2;
                if (!std::isfinite(alpha) || alpha < 0.0) alpha = 0.0;
                alpha = clamp(alpha, alphaMin, alphaMax);
                dispGene[j] = alpha;
                featureObs[j] = std::move(obs);
                ok[j] = 1;
            }
            if (verbose > 0) {
                std::lock_guard<std::mutex> guard(io_mutex);
                features_done += (range.end() - range.begin());
                notice("Contrast %s Stage A: processed %d / %d features",
                    contrastName.c_str(), features_done, M);
            }
        });

        notice("Contrast %s: Stage B (dispersion trend)", contrastName.c_str());
        std::vector<int32_t> valid_ids;
        valid_ids.reserve(M);
        for (int32_t j = 0; j < M; ++j) {
            if (ok[j] && baseMean[j] > 0.0 && dispGene[j] > 0.0) {
                valid_ids.push_back(j);
            }
        }
        if (valid_ids.size() >= 3) {
            std::vector<double> lx(valid_ids.size());
            std::vector<double> ly(valid_ids.size());
            for (size_t t = 0; t < valid_ids.size(); ++t) {
                int32_t j = valid_ids[t];
                lx[t] = std::log(baseMean[j]);
                ly[t] = std::log(dispGene[j]);
            }
            std::vector<double> yhat;
            int32_t ret = loess_quadratic_tricube(lx, ly, yhat, dispLoessSpan);
            if (ret > 0) {
                for (size_t t = 0; t < valid_ids.size(); ++t) {
                    int32_t j = valid_ids[t];
                    double v = std::exp(yhat[t]);
                    if (!std::isfinite(v)) {
                        warning("Fitted dispersion trend has invalid value at raw alpha=%g, scaled mean=%g",
                            dispGene[j], baseMean[j]);
                    }
                    dispTrend[j] = std::isfinite(v) ? v : dispGene[j];
                }
            } else {
                warning("Failed to fit dispersion trend");
                for (int32_t j : valid_ids) {
                    dispTrend[j] = dispGene[j];
                }
            }
        } else {
            warning("Not enough valid features (%d) to fit dispersion trend",
                static_cast<int32_t>(valid_ids.size()));
            for (int32_t j : valid_ids) {
                dispTrend[j] = dispGene[j];
            }
        }

        std::vector<double> residuals;
        residuals.reserve(valid_ids.size());
        for (int32_t j : valid_ids) {
            double trend = dispTrend[j];
            if (trend > 0.0 && std::isfinite(trend)) {
                residuals.push_back(std::log(dispGene[j]) - std::log(trend));
            }
        }
        double mad_val = 0.0;
        if (!residuals.empty()) {
            mad_val = mad(residuals);
        }
        for (int32_t j : valid_ids) {
            double trend = dispTrend[j];
            if (!(trend > 0.0) || !std::isfinite(trend)) {
                trend = dispGene[j];
                dispTrend[j] = trend;
            }
            double resid = std::log(dispGene[j]) - std::log(trend);
            bool is_outlier = (mad_val > 0.0) && (resid > dispOutlierTau * mad_val);
            outlier[j] = is_outlier ? 1 : 0;
            dispFixed[j] = is_outlier ? dispGene[j] : trend;
            dispFixed[j] = clamp(dispFixed[j], alphaMin, alphaMax);
        }
        int32_t n_outliers = std::accumulate(outlier.begin(), outlier.end(), 0);
        notice("Detected %d dispersion outliers among %zu tested features",
            n_outliers, valid_ids.size());

        std::string disp_path = outPrefix + "." + contrastName + ".dispersion.tsv";
        std::ofstream disp_out(disp_path);
        if (!disp_out) error("Failed to open dispersion output: %s", disp_path.c_str());
        disp_out << "Feature\tbaseMean\tdispGene\tdispTrend\tdispFixed\toutlier\tok\n";
        disp_out << std::fixed << std::setprecision(6);
        for (int32_t j = 0; j < M; ++j) {
            disp_out << data.featureNames[j]
                     << "\t" << baseMean[j]
                     << "\t" << dispGene[j]
                     << "\t" << dispTrend[j]
                     << "\t" << dispFixed[j]
                     << "\t" << static_cast<int>(outlier[j])
                     << "\t" << static_cast<int>(ok[j]) << "\n";
        }
        disp_out.close();
        notice("Wrote dispersions to %s", disp_path.c_str());

        std::string out_path = outPrefix + "." + contrastName + ".tsv";
        std::ofstream out(out_path);
        if (!out) error("Failed to open output file: %s", out_path.c_str());
        std::string se_header = (se_method == 3) ? "SE\tSE0" : "SE";
        out << "Feature\tFactor\tBeta\t" << se_header
            << "\tlog10p\tDist2bd\tn\tCount1\tCount2\n";
        out << std::fixed << std::setprecision(6);
        notice("Contrast %s: Stage C (fit group effects)", contrastName.c_str());

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

        features_done = 0;
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
                if (!ok[j]) {continue;}
                const LowresFeatureObs& obs = featureObs[j];
                VectorXd b0 = eta0.col(j);

                double alpha = dispFixed[j];
                if (!std::isfinite(alpha) || alpha <= 0.0) alpha = alphaMin;

                VectorXd b;
                VectorXd bd(K);
                MLEStats stats;
                double fval = 0.0;
                if (use_log1p) {
                    MLEOptions optLocal = mleOpt;
                    optLocal.optim.set_bounds(-b0, b0);
                    bd = b0;
                    auto& P = nb_log1p_tls->local();
                    P.reset_feature(obs.y.ids, obs.y.cnts, b0, alpha);
                    b = VectorXd::Zero(K);
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
                    MixNBLogRegProblem P(data.A_all, cd.n, yvec, cd.xvec,
                        cd.cvecMasked, b0, Ac, alpha, optLocal);
                    b = VectorXd::Zero(K);
                    fval = tron_solve(P, b, optLocal.optim, stats.optim);
                    P.compute_se(b, optLocal, stats);
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
                notice("Contrast %s Stage C: processed %d / %d features",
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
