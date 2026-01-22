#include "punkst.h"
#include "tileoperator.hpp"
#include "dataunits.hpp"
#include "utils.h"
#include "lda.hpp"
#include "poisreg.hpp"
#include "mixpois.hpp"
#include "numerical_utils.hpp"
#include <random>
#include <fstream>
#include <iomanip>
#include <memory>
#include <atomic>
#include <mutex>
#include <sstream>
#include <numeric>
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

int32_t cmdConditionalTestNbReg(int argc, char** argv) {

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
    double dispLoessSpan = 0.2; // DESeq2 default (?)
    double dispOutlierTau = 2.0;
    double alphaMin = 1e-8;
    double alphaMax = 1e4;
    OptimOptions optim;
    optim.tron.enabled = true;

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
      .add_option("max-iter-inner", "(NB Reg) Maximum iterations for fitting DE parameters", optim.max_iters)
      .add_option("tol-inner", "Inner tolerance for fitting DE parameters", optim.tol)
      .add_option("min-count", "Minimum total count per unit to be included", minCount)
      .add_option("min-count-per-feature", "Min total count for a feature to be tested", minCountFeature)
      .add_option("min-units-per-feature", "Min number of units with nonzero count for a feature to be tested", minUnits)
      .add_option("size-factor", "Size factor", sizeFactor)
      .add_option("max-pval", "Max p-value for output (default: 1)", maxPval)
      .add_option("se-method", "Method for calculating SE (1: Fisher, 2: Sandwich, 3: both)", se_method)
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
        pl.print_help();
        return 1;
    }
    if (debug_ > 0) {
        logger::Logger::getInstance().setLevel(logger::LogLevel::DEBUG);
    }

    if (seed <= 0) {seed = std::random_device{}();}
    if (sizeFactor <= 0) {error("Size factor must be positive");}
    if (nThreads < 0) {nThreads = 1;}
    if (maxPval <= 0) {error("max-pval must be positive");}
    if (se_method < 1 || se_method > 3) {error("--se-method must be 1/2/3");}
    if (dispLoessSpan <= 0.0 || dispLoessSpan > 1.0) {error("--dispersion-loess-span must be in (0,1]");}
    if (dispOutlierTau <= 0.0) {error("--dispersion-outlier-tau must be positive");}
    if (alphaMin <= 0.0 || alphaMax <= 0.0 || alphaMin > alphaMax) {
        error("--alpha-min/max must be positive with min <= max");
    }
    double minlog10p = - std::log10(maxPval);
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nThreads);

    PoisLink nb_link = parse_pois_link(link);
    const bool use_log1p = (nb_link == PoisLink::Log1p);

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

    MLEOptions mleOpt(optim);
    mleOpt.se_flag = se_method;
    mleOpt.store_cov = false;
    const double b_null = 0.0;

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
        int32_t n = 0;
        VectorXd xvec = VectorXd::Zero(N_total);
        VectorXd cvec_masked = VectorXd::Zero(N_total);
        VectorXd a_sum_ = VectorXd::Zero(K);
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
        const double c_sum = cvec_masked.sum();
        if (c_sum <= 0.0) {
            warning("Contrast %s has zero total count", contrastNames[c].c_str());
            continue;
        }

        // For fitting baseline models
        OptimOptions baseline_optim = optim;
        baseline_optim.set_bounds(0.0, 1e30, K);
        MixPoisLink link = use_log1p ? MixPoisLink::Log1p : MixPoisLink::Log;
        MixPoisReg baseline(A_all, cvec_masked, a_sum_, baseline_optim, link);

        VectorXd c2vec = cvec_masked.array().square().matrix();
        MatrixXd Mmat = A_all.transpose() * (A_all.array().colwise() * c2vec.array()).matrix(); // K x K

        // Pre-compute values to reuse in log1p sparse optimization
        VectorXd s1p, s1m, s1p_c2, s1m_c2, s2p, s2m;
        double csum_p = 0.0, csum_m = 0.0, c2sum_p = 0.0, c2sum_m = 0.0;
        if (use_log1p) {
            VectorXd& c = cvec_masked;
            VectorXd& x = xvec;
            VectorXd wp = (c.array() * (x.array() > 0).cast<double>()).matrix();
            VectorXd wm = (c.array() * (x.array() < 0).cast<double>()).matrix();
            VectorXd wp2 = (c.array().square() * (x.array() > 0).cast<double>()).matrix();
            VectorXd wm2 = (c.array().square() * (x.array() < 0).cast<double>()).matrix();
            s1p = A_all.transpose() * wp;
            s1m = A_all.transpose() * wm;
            s1p_c2 = A_all.transpose() * wp2;
            s1m_c2 = A_all.transpose() * wm2;
            csum_p = wp.sum();
            csum_m = wm.sum();
            c2sum_p = wp2.sum();
            c2sum_m = wm2.sum();
            s2p = VectorXd::Zero(K);
            s2m = VectorXd::Zero(K);
            for (int i = 0; i < N_total; ++i) {
                if (c[i] <= 0.0 || !(x[i] > 0.0 || x[i] < 0.0)) continue;
                const double ci2 = c[i] * c[i];
                const auto ai = A_all.row(i).array();
                if (x[i] > 0.0) {
                    s2p.array()  += ci2 * ai.square();
                } else {
                    s2m.array() += ci2 * ai.square();
                }
            }
        }

        // For dispersion parameters
        std::vector<double> baseMean(M, 0.0); // mean of normalized counts
        std::vector<double> dispGene(M, 0.0);
        std::vector<double> dispTrend(M, 0.0);
        std::vector<double> dispFixed(M, 0.0);
        std::vector<uint8_t> ok(M, 0);
        std::vector<uint8_t> outlier(M, 0);

        // Randomly distribute features to threads
        std::vector<int32_t> perm_idx(M);
        for (int32_t j = 0; j < M; ++j) {perm_idx[j] = j;}
        std::mt19937 rng(seed);
        std::shuffle(perm_idx.begin(), perm_idx.end(), rng);
        int32_t grain_size = std::max(1, M / (5 * nThreads));
        std::mutex io_mutex;

        // ---- Stage A: fit null model + gene-wise dispersion ----
        int32_t features_done = 0;
        notice("Contrast %s: Stage A (fit null + dispersion)", contrastNames[c].c_str());
        tbb::parallel_for(tbb::blocked_range<int32_t>(0, M, grain_size),
            [&](const tbb::blocked_range<int32_t>& range)
        {
            int32_t thread_id = tbb::this_task_arena::current_thread_index();
            notice("Thread %d start processing %d features", thread_id, range.size());

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
                // Fit baseline \lam_i = c_i * A_i * \beta
                // b0 = g(\beta) (log or log1p)
                VectorXd b0 = baseline.fit_one(ys.ids, ys.cnts, oK);
                eta0.col(j) = b0;

                // For dispersion trend
                double sum_y = static_cast<double>(total_count);
                baseMean[j] = sum_y / c_sum;
                if (baseMean[j] <= 0.0) {continue;}
                // Moment-based gene-specific dispersion estimate
                VectorXd beta(K);
                if (use_log1p) {
                    beta = (b0.array().min(40.0).exp() - 1.0).matrix();
                } else {
                    beta = (b0.array().min(40.0).exp()).matrix();
                }
                // \lam_i = c_i * A_i * beta
                // Mmat = A^T * diag(cvec^2) * A
                double sum_lam2 = beta.dot(Mmat * beta); // \sum_i lam_i^2
                sum_lam2 = std::max(sum_lam2, 1e-12);
                double sum_y2 = 0.0; // \sum_i y_i^2
                double sum_y_lam = 0.0; // \sum_i y_i * lam_i
                for (size_t t = 0; t < ys.ids.size(); ++t) {
                    double cnt = ys.cnts[t];
                    sum_y2 += cnt * cnt;
                    int32_t idx = static_cast<int32_t>(ys.ids[t]);
                    double lam = cvec_masked[idx] * A_all.row(idx).dot(beta);
                    sum_y_lam += cnt * lam;
                }
                // Var(y) = lam + alpha * lam^2
                double numer = sum_y2 - 2.0 * sum_y_lam + sum_lam2 - sum_y;
                double alpha = numer / sum_lam2;
                if (!std::isfinite(alpha) || alpha < 0.0) alpha = 0.0;
                alpha = clamp(alpha, alphaMin, alphaMax);
                dispGene[j] = alpha;
                ok[j] = 1;
            }
            if (verbose > 0) {
                std::lock_guard<std::mutex> guard(io_mutex);
                features_done += (range.end() - range.begin());
                notice("Contrast %s Stage A: processed %d / %d features",
                    contrastNames[c].c_str(), features_done, M);
            }
        });

        // ---- Stage B: fit dispersion trend (smmooth) ----
        notice("Contrast %s: Stage B (dispersion trend)", contrastNames[c].c_str());
        std::vector<int32_t> valid_ids;
        valid_ids.reserve(M);
        for (int32_t j = 0; j < M; ++j) {
            if (ok[j] && baseMean[j] > 0.0 && dispGene[j] > 0.0) {
                valid_ids.push_back(j);
            }
        }
        // Fit LOESS curve of log(alpha) ~ log(baseMean)
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
                        warning("Fitted dispersion trend has invalid value at raw alpha=%g, scaled mean=%g", dispGene[j], baseMean[j]);
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
            warning("Not enough valid features (%d) to fit dispersion trend", static_cast<int32_t>(valid_ids.size()));
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
        double mad_val = 0.0; // median absolute deviation of residuals
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
        notice("Detected %d dispersion outliers among %zu tested features", n_outliers, valid_ids.size());

        std::string disp_path = outPrefix + "." + contrastNames[c] + ".dispersion.tsv";
        std::ofstream disp_out(disp_path);
        if (!disp_out) error("Failed to open dispersion output: %s", disp_path.c_str());
        disp_out << "Feature\tbaseMean\tdispGene\tdispTrend\tdispFixed\toutlier\tok\n";
        disp_out << std::fixed << std::setprecision(6);
        for (int32_t j = 0; j < M; ++j) {
            disp_out << featureNames[j]
                     << "\t" << baseMean[j]
                     << "\t" << dispGene[j]
                     << "\t" << dispTrend[j]
                     << "\t" << dispFixed[j]
                     << "\t" << static_cast<int>(outlier[j])
                     << "\t" << static_cast<int>(ok[j]) << "\n";
        }
        disp_out.close();
        notice("Wrote dispersions to %s", disp_path.c_str());

        // ---- Stage C: full model with fixed dispersion ----
        std::string out_path = outPrefix + "." + contrastNames[c];
        if (use_log1p) {
            out_path += ".mixnb_log1p.tsv";
        } else {
            out_path += ".mixnb_log.tsv";
        }
        std::ofstream out(out_path);
        if (!out) error("Failed to open output file: %s", out_path.c_str());
        std::string se_header = (se_method == 3) ? "SE\tSE0" : "SE";
        out << "Feature\tFactor\tBeta\t" << se_header
            << "\tlog10p\tDist2bd\tn\tCount1\tCount2\n";
        out << std::fixed << std::setprecision(6);
        notice("Contrast %s: Stage C (fit group effects)", contrastNames[c].c_str());

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
                int32_t j = perm_idx[jj];
                if (!ok[j]) {continue;}
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
                VectorXd b0 = eta0.col(j);
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

                double alpha = dispFixed[j];
                if (!std::isfinite(alpha) || alpha <= 0.0) alpha = alphaMin;

                VectorXd b; // fitted coefficients
                VectorXd bd(K); // bounds
                MLEStats stats;
                double fval = 0.0;
                if (use_log1p) {
                    MLEOptions optLocal = mleOpt;
                    optLocal.optim.set_bounds(-b0, b0);
                    bd = b0;
                    MixNBLog1pSparseApproxProblem P(A_all, n, ys.ids, ys.cnts,
                        xvec, cvec_masked, b0, alpha, optLocal.optim,
                        optLocal.ridge, optLocal.soft_tau,
                        s1p, s1m, s1p_c2, s1m_c2, s2p, s2m,
                        csum_p, csum_m, c2sum_p, c2sum_m);
                    b = VectorXd::Zero(K);
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
                    touched = ys.ids;
                    bd.setConstant(std::log(100.0));
                    for (int k = 0; k < K; ++k) {
                        double bcap_floor = (b0[k] - std::log(1e-8 * sizeFactor));
                        if (std::isfinite(bcap_floor) && bcap_floor > 0.0) {
                            bd[k] = std::min(bd[k], bcap_floor);
                        }
                    }
                    MLEOptions optLocal = mleOpt;
                    optLocal.optim.set_bounds(-bd, bd);
                    MixNBLogRegProblem P(A_all, n, yvec, xvec, cvec_masked, b0, alpha, optLocal);
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
                notice("Contrast %s Stage C: processed %d / %d features",
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

        if (use_log1p) {
            out_path = outPrefix + "." + contrastNames[c] + ".eta0.tsv";
            RowMajorMatrixXd eta_out = eta0.transpose().array().exp() - 1.0;
            write_matrix_to_file(out_path, eta_out, 4, true, featureNames, "Feature", &factorNames);
        }
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
