#include "punkst.h"
#include "tileoperator.hpp"
#include "dataunits.hpp"
#include "utils.h"
#include "lda.hpp"
#include "poisreg.hpp"
#include <random>
#include <fstream>
#include <iomanip>
#include <memory>
#include <atomic>
#include <mutex>
#include <sstream>
#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>
#include "Eigen/Dense"
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

static int32_t readContrastFromFile(const std::string& contrastFile,
                                  std::vector<std::vector<int32_t>>& contrasts,
                                  std::vector<std::string>& contrastNames,
                                  int32_t G);

enum class PoisLink { Log, Log1p };

static PoisLink parse_pois_link(const std::string& link) {
    if (link == "log") return PoisLink::Log;
    if (link == "log1p") return PoisLink::Log1p;
    error("Unknown link: %s (expected log or log1p)", link.c_str());
    return PoisLink::Log;
}

int32_t cmdConditionalTestPoisLogReg(int argc, char** argv) {

    std::string contrastFile, outPrefix, modelFile;
    std::vector<std::string> inFile, metaFile;
    std::string link = "log1p";
    int32_t nThreads = 1;
    int32_t seed = -1;
    int32_t maxIter = 100;
    double  mDelta = 1e-3;
    int32_t minCount = 50;
    int32_t minUnits = 10;
    double  maxPval = 1;
    int32_t minCountFeature = 100;
    int32_t debug_ = 0, verbose = 0;
    uint32_t se_method = 1;
    OptimOptions optim;
    optim.tron.enabled = true;
    optim.max_iters = 50;
    optim.tol = 1e-6;
    double sizeFactor = 10000.0;

    ParamList pl;
    pl.add_option("in-data", "Input data files", inFile, true)
      .add_option("in-meta", "Metadata files", metaFile, true)
      .add_option("contrast", "Contrast file", contrastFile)
      .add_option("model", "Model file", modelFile, true)
      .add_option("link", "Regression link: log or log1p", link)
      .add_option("out", "Output prefix", outPrefix, true)
      .add_option("threads", "Number of threads to use", nThreads)
      .add_option("seed", "Random seed", seed)
      .add_option("max-iter", "(LDA) Max iterations for fitting cell type composition", maxIter)
      .add_option("mean-change-tol", "(LDA) Convergence tolerance for fitting cell type composition", mDelta)
      .add_option("max-iter-inner", "(Pois. Reg) Maximum iterations for fitting DE parameters", optim.max_iters)
      .add_option("tol-inner", "Inner tolerance for fitting DE parameters", optim.tol)
      .add_option("min-count", "Minimum total count per unit to be included", minCount)
      .add_option("min-count-per-feature", "Min total count for a feature to be tested", minCountFeature)
      .add_option("min-units-per-feature", "Min number of units with nonzero count for a feature to be tested", minUnits)
      .add_option("size-factor", "Size factor for Poisson regression", sizeFactor)
      .add_option("max-pval", "Max p-value for output (default: 1)", maxPval)
      .add_option("se-method", "Method for calculating SE (0: Fisher, 1: Sandwich)", se_method)
      .add_option("seed", "Random seed", seed)
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
    if (se_method != 1 && se_method != 2) {error("--se-method must be 1, or 2");}
    double minlog10p = - std::log10(maxPval);
    tbb::global_control global_limit(tbb::global_control::max_allowed_parallelism, nThreads);

    PoisLink pois_link = parse_pois_link(link);
    const bool use_log1p = (pois_link == PoisLink::Log1p);

    int32_t G = inFile.size();
    if (metaFile.size() != G) {
        error("Number of metadata files must match number of data files");
    }

    // Read / construct contrasts
    std::vector<std::vector<int32_t>> contrasts;
    std::vector<std::string> contrastNames;
    int32_t C;
    if (!contrastFile.empty()) {
        C = readContrastFromFile(contrastFile, contrasts, contrastNames, G);
        notice("Read %d contrasts from %s", C, contrastFile.c_str());
    } else { // generate all pairwise contrasts
        for (int32_t g = 0; g < G; ++g) {
            for (int32_t l = g+1; l < G; ++l) {
                contrasts.push_back(std::vector<int32_t>(G, 0));
                contrasts.back()[g] = 1; contrasts.back()[l] = -1;
                contrastNames.push_back(std::to_string(g) + "v" + std::to_string(l));
            }
        }
        C = contrasts.size();
        notice("Will perform all pairwise contrasts (%d) among %d samples", C, G);
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
    for (int32_t g = 0; g < G; ++g) {
        std::vector<Document> docs;
        nUnits[g] = readers[g].readAll(docs, inFile[g], minCount, true, debug_);
        notice("Read %zu units from the %d-th data file", nUnits[g], g);
        if (nUnits[g] == 0) {
            warning("No units passed the --min-count filter for the %d-th data file", g);
        }
        thetas[g] = lda.transform(docs); // row-normalized
        cvecs[g].resize(nUnits[g]);
        totct[g].resize(nUnits[g]);
        docsT[g].resize(M);
        for (uint32_t i = 0; i < nUnits[g]; ++i) {
            Document& doc = docs[i];
            totct[g][i] = doc.get_sum();
            cvecs[g][i] = doc.get_sum() / sizeFactor;
            for (size_t t = 0; t < doc.ids.size(); ++t) {
                int32_t m = doc.ids[t];
                docsT[g][m].ids.push_back(i);
                docsT[g][m].cnts.push_back(doc.cnts[t]);
            }
        }
        offsets[g + 1] = offsets[g] + nUnits[g];
    }

    // optim.set_bounds(-20, 20, K); // set feature specific bounds later
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

    for (int32_t c = 0; c < C; ++c) {
        int32_t n = 0;
        VectorXd xvec = VectorXd::Zero(N_total);
        VectorXd cvec_masked = VectorXd::Zero(N_total);
        for (int32_t g = 0; g < G; ++g) {
            if (contrasts[c][g] != 0) {
                n += nUnits[g];
                int32_t ng = nUnits[g];
                int32_t offset = offsets[g];
                xvec.segment(offset, ng) =
                    Eigen::VectorXd::Constant(ng, contrasts[c][g]);
                cvec_masked.segment(offset, ng) =
                    cvec_all.segment(offset, ng);
            }
        }
        // Pre-compute values to reuse in log1p sparse optimization
        VectorXd *s2p_ptr = nullptr, *s2m_ptr = nullptr;
        VectorXd s1p, s1m, s2p, s2m;
        if (use_log1p) {
            VectorXd& c = cvec_masked;
            VectorXd& x = xvec;
            VectorXd wp = (c.array() * (x.array() > 0).cast<double>()).matrix();
            VectorXd wm = (c.array() * (x.array() < 0).cast<double>()).matrix();
            s1p = A_all.transpose() * wp;
            s1m = A_all.transpose() * wm;
            if (se_method == 2) {
                s2p = VectorXd::Zero(K);
                s2m = VectorXd::Zero(K);
                for (int i = 0; i < N_total; ++i) {
                    if (c[i] <= 0.0 || !(x[i]>0 || x[i]<0)) continue;
                    const double ci2 = c[i] * c[i];
                    const auto ai = A_all.row(i).array();
                    if (x[i] > 0.0) {
                        s2p.array()  += ci2 * ai.square();
                    } else {
                        s2m.array() += ci2 * ai.square();
                    }
                }
                s2p_ptr = &s2p;
                s2m_ptr = &s2m;
            }
        }

        std::string out_path = outPrefix + "." + contrastNames[c];
        if (use_log1p) {
            out_path += ".mixpois_log1p.tsv";
        } else {
            out_path += ".mixpois_log.tsv";
        }
        std::ofstream out(out_path);
        if (!out) error("Failed to open output file: %s", out_path.c_str());
        out << "Feature\tFactor\tBeta\tSE\tlog10p\tCount1\tCount2\n";
        out << std::fixed << std::setprecision(6);
        notice("Fitting contrast %s with N=%d", contrastNames[c].c_str(), n);

        std::unique_ptr<tbb::global_control> debug_limit;
        if (debug_ > 0) {
            debug_limit = std::make_unique<tbb::global_control>(
                tbb::global_control::max_allowed_parallelism, 1);
        }

        std::unique_ptr<tbb::enumerable_thread_specific<VectorXd>> yvec_tls;
        std::unique_ptr<tbb::enumerable_thread_specific<std::vector<int32_t>>> touched_tls;
        if (!use_log1p) {
            yvec_tls = std::make_unique<tbb::enumerable_thread_specific<VectorXd>>(
                [N_total]() { return VectorXd::Zero(N_total); });
            touched_tls = std::make_unique<
                tbb::enumerable_thread_specific<std::vector<int32_t>>>();
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
            std::ostringstream oss;
            oss.setf(std::ios::fixed);
            oss << std::setprecision(6);
            for (int32_t jj = range.begin(); jj < range.end(); ++jj) {
                int32_t j = perm_idx[jj];
                int32_t n0 = 0, n1 = 0;
                double ysum0 = 0.0, ysum1 = 0.0;
                for (int32_t g = 0; g < G; ++g) {
                    if (contrasts[c][g] == 0) {continue;}
                    const auto& doc = docsT[g][j];
                    int32_t nz = static_cast<int32_t>(doc.ids.size());
                    double ysum = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
                    if (contrasts[c][g] < 0) {
                        n0 += nz;
                        ysum0 += ysum;
                    } else {
                        n1 += nz;
                        ysum1 += ysum;
                    }
                }
                int32_t m = n0 + n1;
                if (m < minUnits) {continue;}
                int32_t total_count = static_cast<int32_t>(ysum0 + ysum1);
                if (total_count < minCountFeature) {
                    continue;
                }
                VectorXd oK = eta0.col(j);
                MLEStats stats;
                VectorXd b;
                double fval = 0.0;
                if (use_log1p) {
                    Document ys;
                    ys.ids.reserve(m);
                    ys.cnts.reserve(m);
                    for (int32_t g = 0; g < G; ++g) {
                        if (contrasts[c][g] == 0) {continue;}
                        const auto& doc = docsT[g][j];
                        int32_t offset = offsets[g];
                        for (size_t t = 0; t < doc.ids.size(); ++t) {
                            ys.ids.push_back(static_cast<uint32_t>(offset) + doc.ids[t]);
                        }
                        ys.cnts.insert(ys.cnts.end(),
                            doc.cnts.begin(), doc.cnts.end());
                    }
                    MLEOptions optLocal = mleOpt;
                    optLocal.optim.set_bounds(-oK, oK);
                    MixPoisLog1pSparseProblem P(A_all, ys.ids, ys.cnts,
                        xvec, cvec_masked, oK, optLocal,
                        s1p, s1m, s2p_ptr, s2m_ptr);
                    b = VectorXd::Zero(K);
                    fval = tron_solve(P, b, optLocal.optim, stats.optim);
                    mix_pois_log1p_compute_se(P, b, optLocal, stats);
                } else {
                    auto& yvec = yvec_tls->local();
                    auto& touched = touched_tls->local();
                    touched.clear();
                    for (int32_t g = 0; g < G; ++g) {
                        if (contrasts[c][g] == 0) {continue;}
                        const auto& doc = docsT[g][j];
                        int32_t offset = offsets[g];
                        for (size_t t = 0; t < doc.ids.size(); ++t) {
                            int32_t idx = offset + static_cast<int32_t>(doc.ids[t]);
                            yvec[idx] = doc.cnts[t];
                            touched.push_back(idx);
                        }
                    }
                    MLEOptions optLocal = mleOpt;
                    VectorXd bd = VectorXd::Constant(K, std::log(100.0));
                    for (int k = 0; k < K; ++k) {
                        double bcap_floor = (oK[k] - std::log(0.01 * sizeFactor));
                        if (std::isfinite(bcap_floor) && bcap_floor > 0.0) {
                            bd[k] = std::min(bd[k], bcap_floor);
                        }
                    }
                    optLocal.optim.set_bounds(-bd, bd);
                    fval = mix_pois_log_mle(
                        A_all, yvec, xvec, cvec_masked, oK, optLocal, b, stats);
                    for (int32_t idx : touched) {
                        yvec[idx] = 0.0;
                    }
                }
                (void)fval;
                VectorXd& se_vec = (se_method == 2) ? stats.se_robust : stats.se_fisher;
                if (se_vec.size() != K) {continue;}
                for (int32_t k = 0; k < K; ++k) {
                    double se = se_vec[k];
                    if (!(se > 0.0) || !std::isfinite(se) || !std::isfinite(b[k])) {
                        std::lock_guard<std::mutex> guard(io_mutex);
                        warning("(%d, %d, %d) Invalid estimate (b=%g, se=%g)", c, k, j, b[k], se);
                        continue;
                    }
                    double z = (b[k] - b_null) / se;
                    double log10p = - log10_twosided_p_from_z(z);
if (k % 10 == 1 && ((verbose > 1 && log10p > 3) || verbose > 2)) {
std::cout << featureNames[j] << "\t" << factorNames[k]
        << std::fixed << std::setprecision(6)
        << "\t" << b[k] << "\t" << se << "\t" << log10p
        << "\t" << (int32_t) (std::round(ysum0))
        << "\t" << (int32_t) (std::round(ysum1)) << "\n";
}
                    if (log10p < minlog10p) {continue;}
                    oss << featureNames[j] << "\t" << factorNames[k]
                        << "\t" << b[k] << "\t" << se << "\t" << log10p
                        << "\t" << (int32_t) (std::round(ysum0))
                        << "\t" << (int32_t) (std::round(ysum1)) << "\n";
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
            }
        });

        out.close();
        notice("Wrote results to %s", out_path.c_str());
    }

    return 0;
};


static int32_t readContrastFromFile(const std::string& contrastFile,
        std::vector<std::vector<int32_t>>& contrasts,
        std::vector<std::string>& contrastNames, int32_t G) {
    std::ifstream ifs(contrastFile);
    if (!ifs.is_open()) {
        error("Cannot open contrast file: %s", contrastFile.c_str());
    }
    std::string line;
    int32_t C = 0;
    while (std::getline(ifs, line)) {
        if (line.empty()) {continue;}
        std::vector<std::string> tokens;
        split(tokens, "\t ", line, UINT_MAX, true, true, true);
        if (line[0] == '#') {
            contrastNames = std::vector<std::string>(tokens.begin() + 1, tokens.end());
            C = tokens.size() - 1;
            continue;
        }
        if (tokens.size() < 2) {
            error("Contrast file malformed: %s", line.c_str());
        }
        if (C == 0) {
            C = tokens.size() - 1;
        } else {
            if (tokens.size() != C + 1) {
                error("Contrast file malformed: %s", line.c_str());
            }
        }
        if (contrasts.empty()) {
            contrasts.assign(C, std::vector<int32_t>(G, 0));
        }
        int32_t s = std::stoi(tokens[0]);
        if (s < 0 || s >= G) {
            error("Contrast file sample index out of range: %d", s);
        }
        for (int32_t c = 1; c <= C; ++c) {
            int32_t y = std::stoi(tokens[c]);
            contrasts[c-1][s] = y;
        }
    }
    if (contrastNames.size() != static_cast<size_t>(C)) {
        warning("Did not find valid contrast names in the contrast file header; using indexes by default");
        contrastNames.resize(C);
        for (int32_t c = 0; c < C; ++c) {
            contrastNames[c] = std::to_string(c);
        }
    }
    return C;
}
