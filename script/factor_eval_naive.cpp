#include "factor_eval_common.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>

namespace {

using namespace factoreval;

struct FactorEvalOptions : FactorEvalCommonOptions {
    std::vector<double> thresholds;
    int32_t projectionMaxIter = 200;
    double projectionTol = 1e-7;
    bool runProjection = false;
    bool runRefit = false;
};

std::vector<double> projectToSimplex(std::vector<double> v) {
    const int32_t n = static_cast<int32_t>(v.size());
    std::vector<double> u = v;
    std::sort(u.begin(), u.end(), std::greater<double>());
    double cumsum = 0.0;
    int32_t rho = 0;
    for (int32_t j = 0; j < n; ++j) {
        cumsum += u[j];
        const double t = (cumsum - 1.0) / static_cast<double>(j + 1);
        if (u[j] - t > 0.0) {
            rho = j + 1;
        }
    }
    cumsum = std::accumulate(u.begin(), u.begin() + rho, 0.0);
    const double theta = (cumsum - 1.0) / static_cast<double>(rho);
    for (double& x : v) {
        x = std::max(0.0, x - theta);
    }
    return v;
}

double bhattacharyya(const RowVectorXd& p, const RowVectorXd& q) {
    double b = 0.0;
    for (Eigen::Index m = 0; m < p.cols(); ++m) {
        if (p(m) > 0.0 && q(m) > 0.0) {
            b += std::sqrt(p(m) * q(m));
        }
    }
    return std::min(1.0, std::max(0.0, b));
}

RowVectorXd weightsToDistribution(const std::vector<double>& w, const RowMajorMatrixXd& beta) {
    RowVectorXd r = RowVectorXd::Zero(beta.cols());
    for (int32_t j = 0; j < static_cast<int32_t>(w.size()); ++j) {
        r.noalias() += w[j] * beta.row(j);
    }
    return r;
}

double projectBhattacharyya(const RowVectorXd& p, const RowMajorMatrixXd& beta,
        std::vector<double> w, int32_t maxIter, double tol) {
    if (w.empty()) {
        return 0.0;
    }
    w = projectToSimplex(std::move(w));
    RowVectorXd r = weightsToDistribution(w, beta);
    double best = bhattacharyya(p, r);

    for (int32_t iter = 0; iter < maxIter; ++iter) {
        std::vector<double> grad(w.size(), 0.0);
        for (int32_t j = 0; j < static_cast<int32_t>(w.size()); ++j) {
            double g = 0.0;
            for (Eigen::Index m = 0; m < p.cols(); ++m) {
                if (p(m) > 0.0) {
                    const double rm = std::max(r(m), std::numeric_limits<double>::min());
                    g += beta(j, m) * std::sqrt(p(m) / rm);
                }
            }
            grad[j] = 0.5 * g;
        }

        double step = 1.0;
        bool improved = false;
        std::vector<double> bestCandidate;
        double candidateScore = best;
        while (step > 1e-12) {
            std::vector<double> cand(w.size());
            for (int32_t j = 0; j < static_cast<int32_t>(w.size()); ++j) {
                cand[j] = w[j] + step * grad[j];
            }
            cand = projectToSimplex(std::move(cand));
            RowVectorXd rcand = weightsToDistribution(cand, beta);
            const double score = bhattacharyya(p, rcand);
            if (score > best + tol * 0.1) {
                bestCandidate = std::move(cand);
                candidateScore = score;
                improved = true;
                break;
            }
            step *= 0.5;
        }
        if (!improved || std::abs(candidateScore - best) < tol) {
            break;
        }
        w = std::move(bestCandidate);
        r = weightsToDistribution(w, beta);
        best = candidateScore;
    }
    return best;
}

void validateOptions(const FactorEvalOptions& opt) {
    validateFactorEvalCommonOptions(opt);
    if (opt.thresholds.empty()) {
        error("--thresholds requires at least one value");
    }
    for (double tau : opt.thresholds) {
        if (tau < 0.0) {
            error("All --thresholds values must be non-negative");
        }
    }
    if (opt.projectionMaxIter <= 0) {
        error("--projection-max-iter must be positive");
    }
    if (opt.projectionTol <= 0.0) {
        error("--projection-tol must be positive");
    }
}

} // namespace

int32_t cmdLDAFactorEvalNaive(int argc, char** argv) {
    FactorEvalOptions opt;

    ParamList pl;
    addFactorEvalCommonOptions(pl, opt,
        "Write per-factor unit-level leave-one-out statistics",
        "Output path for per-factor unit-level leave-one-out statistics");

    pl.add_option("thresholds", "Theta thresholds for selecting units and computing factor-level summaries", opt.thresholds, true)
      .add_option("projection-max-iter", "Maximum projected-gradient iterations", opt.projectionMaxIter)
      .add_option("projection-tol", "Projection convergence tolerance", opt.projectionTol)
      .add_option("projection", "Run direct convex projection leave-one-out evaluation", opt.runProjection)
      .add_option("refit", "Run reduced fixed-model refit leave-one-out evaluation", opt.runRefit);

    try {
        pl.readArgs(argc, argv);
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    validateOptions(opt);
    if (opt.seed <= 0) {
        opt.seed = std::random_device{}();
    }
    if (!opt.runProjection && !opt.runRefit) {
        opt.runRefit = true;
    }
    pl.print_options();
    std::sort(opt.thresholds.begin(), opt.thresholds.end());
    const double minEvalTheta = opt.thresholds.front();

    const auto dgeInputs = resolveDge10XInputs(opt.dgeDirs, opt.inBarcodes, opt.inFeatures,
        opt.inMatrix, opt.datasetIds);
    const bool use10x = !dgeInputs.empty();
    if (use10x && !opt.inFile.empty()) {
        warning("Both --in-data and 10X inputs are provided; using 10X inputs and ignoring --in-data");
    }
    if (!use10x && opt.inFile.empty()) {
        error("Either --in-data or 10X inputs must be provided");
    }

    HexReader reader;
    std::unique_ptr<DGEReader10X> dgePtr;
    if (use10x) {
        if (!opt.inBarcodes.empty() || !opt.inFeatures.empty() || !opt.inMatrix.empty()) {
            dgePtr = std::make_unique<DGEReader10X>(opt.inBarcodes, opt.inFeatures,
                opt.inMatrix, opt.datasetIds);
        } else {
            dgePtr = std::make_unique<DGEReader10X>(opt.dgeDirs, opt.datasetIds);
        }
        reader.initFromFeatures(dgePtr->features, dgePtr->nBarcodes);
    } else {
        if (opt.metaFile.empty()) {
            error("Missing required --in-meta for non-10X input");
        }
        reader.readMetadata(opt.metaFile);
    }
    if (!opt.featureFile.empty()) {
        reader.setFeatureFilter(opt.featureFile, opt.minCountFeature,
            opt.includeFeatureRegex, opt.excludeFeatureRegex);
    }
    if (!opt.weightFile.empty()) {
        reader.setWeights(opt.weightFile, opt.defaultWeight, opt.icolWeight);
    }

    LDA4Hex lda(reader, opt.modal, opt.verbose);
    lda.initialize_transform(opt.modelFile, opt.seed, opt.nThreads,
        opt.verbose, opt.maxIter, opt.meanChangeTol, opt.alpha);

    if (use10x) {
        const std::vector<std::string> modelFeatures = lda.getFeatureNames();
        int32_t nOverlap = dgePtr->setFeatureIndexRemap(modelFeatures, false);
        if (nOverlap == 0) {
            error("No overlapping features found between 10X input and model metadata");
        }
    }

    const int32_t K = lda.getNumTopics();
    if (K < 2) {
        error("The model must contain at least two factors");
    }

    EvaluationData data = loadEvaluationData(opt, lda, dgePtr.get());
    if (data.docs.empty()) {
        error("No input units passed filtering");
    }
    notice("Loaded %d units for factor evaluation", static_cast<int32_t>(data.docs.size()));

    RowMajorMatrixXd theta = lda.do_transform(DocumentView(data.docs.data(), data.docs.size()));
    const double ldaAlpha = lda.get_doc_topic_prior();
    RowMajorMatrixXd betaNorm = rowNormalize(lda.get_model_matrix());
    UnitGofStats gof = computeUnitGof(data.docs, theta, betaNorm);
    const std::string unitGofPath = opt.outPrefix + ".unit_gof.tsv";
    writeUnitGof(unitGofPath, data.docs, gof);
    notice("Per-unit goodness-of-fit statistics written to %s", unitGofPath.c_str());

    if (opt.writeTheta) {
        const std::string thetaPath = opt.outPrefix + ".theta.tsv";
        writeThetaMatrix(thetaPath, theta);
        notice("Theta matrix written to %s", thetaPath.c_str());
    }
    FactorCandidateStats candidateStats = computeFactorCandidateStats(
        data.docs, theta, opt.candidateThreshold, opt.minCountFactor);
    const std::string thetaHistPath = opt.outPrefix + ".theta_hist.tsv";
    writeThetaHist(thetaHistPath, candidateStats.thetaHist);
    notice("Per-factor theta histograms written to %s", thetaHistPath.c_str());

    if (candidateStats.candidates.size() == 0) {
        notice("No candidate factors selected; exiting");
        return 0;
    }

    notice("Selected %d candidate factors using count-weighted theta fraction <= %.6g and total weight >= %.6g",
        static_cast<int32_t>(candidateStats.candidates.size()), opt.candidateThreshold, opt.minCountFactor);

    const std::string refitFactorEvalPath = opt.outPrefix + ".factor_eval.tsv";
    const std::string projectionFactorEvalPath = opt.outPrefix + ".prj.factor_eval.tsv";
    std::ofstream refitFactorOut;
    std::ofstream projectionFactorOut;
    auto openFactorEvalOut = [](std::ofstream& out, const std::string& path) {
        out.open(path);
        if (!out) {
            error("Error opening output file: %s for writing", path.c_str());
        }
        out << "factor\ttheta_sum\tweight\ttau\tN\tS\tW\tI\truntime_sec\n";
    };
    if (opt.runRefit) {
        openFactorEvalOut(refitFactorOut, refitFactorEvalPath);
    }
    if (opt.runProjection) {
        openFactorEvalOut(projectionFactorOut, projectionFactorEvalPath);
    }

    std::ofstream unitFactorOut;
    std::string unitFactorPath = opt.unitFactorStatsOut.empty()
        ? opt.outPrefix + ".unit_factor_eval.tsv"
        : opt.unitFactorStatsOut;
    if (opt.writeUnitFactorStats || !opt.unitFactorStatsOut.empty()) {
        unitFactorOut.open(unitFactorPath);
        if (!unitFactorOut) {
            error("Error opening output file: %s for writing", unitFactorPath.c_str());
        }
        unitFactorOut << "method\tfactor\tunit_id\ttheta_factor\tB\n";
        // unitFactorOut << std::fixed;
    }

    const RowMajorMatrixXd originalModel = lda.copy_model_matrix();
    for (int32_t k : candidateStats.candidates) {
        std::vector<int32_t> evalRows;
        for (int32_t i = 0; i < theta.rows(); ++i) {
            if (theta(i, k) > minEvalTheta) {
                evalRows.push_back(i);
            }
        }

        const std::vector<int32_t> factors = reducedFactorMap(K, k);
        const RowMajorMatrixXd betaReduced = reducedBetaRows(betaNorm, factors);
        std::vector<double> projectionD(theta.rows(), 0.0);
        std::vector<double> refitD(theta.rows(), 0.0);
        std::vector<double> projectionB(theta.rows(), 1.0);
        std::vector<double> refitB(theta.rows(), 1.0);
        double projectionRuntime = 0.0;
        double refitRuntime = 0.0;

        if (opt.runProjection) {
            auto t0 = std::chrono::steady_clock::now();
            for (int32_t i : evalRows) {
                RowVectorXd p = theta.row(i) * betaNorm;
                std::vector<double> init(factors.size(), 0.0);
                double initSum = 0.0;
                for (int32_t j = 0; j < static_cast<int32_t>(factors.size()); ++j) {
                    init[j] = theta(i, factors[j]);
                    initSum += init[j];
                }
                if (initSum > 0.0) {
                    for (double& x : init) {
                        x /= initSum;
                    }
                } else {
                    std::fill(init.begin(), init.end(), 1.0 / static_cast<double>(init.size()));
                }
                const double B = projectBhattacharyya(p, betaReduced, std::move(init),
                    opt.projectionMaxIter, opt.projectionTol);
                projectionB[i] = B;
                projectionD[i] = -data.docs[i].get_raw_sum() *
                    std::log(std::max(B, std::numeric_limits<double>::min()));
            }
            auto t1 = std::chrono::steady_clock::now();
            projectionRuntime = std::chrono::duration<double>(t1 - t0).count();
        }

        if (opt.runRefit) {
            auto t0 = std::chrono::steady_clock::now();
            RowMajorMatrixXd reducedModel = removeFactorRows(originalModel, k);
            LatentDirichletAllocation reducedLda(reducedModel, opt.seed, opt.nThreads,
                opt.verbose, InferenceType::SVB, ldaAlpha);
            reducedLda.set_svb_parameters(opt.maxIter, opt.meanChangeTol);
            std::vector<Document> selectedDocs;
            selectedDocs.reserve(evalRows.size());
            for (int32_t i : evalRows) {
                selectedDocs.push_back(data.docs[i]);
            }
            RowMajorMatrixXd thetaReduced;
            if (!selectedDocs.empty()) {
                thetaReduced = reducedLda.transform(DocumentView(selectedDocs.data(), selectedDocs.size()));
            }
            for (int32_t local = 0; local < static_cast<int32_t>(evalRows.size()); ++local) {
                const int32_t i = evalRows[local];
                RowVectorXd p = theta.row(i) * betaNorm;
                RowVectorXd r = thetaReduced.row(local) * betaReduced;
                const double B = bhattacharyya(p, r);
                refitB[i] = B;
                refitD[i] = -data.docs[i].get_raw_sum() *
                    std::log(std::max(B, std::numeric_limits<double>::min()));
            }
            auto t1 = std::chrono::steady_clock::now();
            refitRuntime = std::chrono::duration<double>(t1 - t0).count();
        }

        auto writeFactorRows = [&](std::ofstream& out, const std::vector<double>& dvals,
                double runtime) {
            const int32_t T = static_cast<int32_t>(opt.thresholds.size());
            std::vector<int32_t> n(T, 0);
            std::vector<double> totalCount(T, 0.0);
            std::vector<double> w(T, 0.0);
            std::vector<double> infoLoss(T, 0.0);
            for (int32_t i = 0; i < theta.rows(); ++i) {
                const double thetaIk = theta(i, k);
                auto lb = std::lower_bound(opt.thresholds.begin(), opt.thresholds.end(), thetaIk);
                const int32_t nPassed = static_cast<int32_t>(lb - opt.thresholds.begin());
                const double ni = data.docs[i].get_raw_sum();
                for (int32_t t = 0; t < nPassed; ++t) {
                    ++n[t];
                    totalCount[t] += ni;
                    w[t] += thetaIk;
                    infoLoss[t] += dvals[i];
                }
            }
            for (int32_t t = 0; t < T; ++t) {
                const double tau = opt.thresholds[t];
                out << k
                    << "\t" << std::setprecision(8) << candidateStats.thetaSums[k]
                    << "\t" << std::setprecision(8) << candidateStats.weightedCounts[k]
                    << "\t" << std::setprecision(2) << tau
                    << "\t" << n[t]
                    << "\t" << std::setprecision(2) << totalCount[t]
                    << "\t" << std::setprecision(2) << w[t]
                    << "\t" << std::setprecision(6) << infoLoss[t]
                    << "\t" << std::setprecision(6) << runtime << "\n";
            }
        };
        if (opt.runProjection) {
            writeFactorRows(projectionFactorOut, projectionD, projectionRuntime);
        }
        if (opt.runRefit) {
            writeFactorRows(refitFactorOut, refitD, refitRuntime);
        }

        if (unitFactorOut) {
            for (int32_t i : evalRows) {
                if (opt.runProjection) {
                    unitFactorOut << "projection\t" << k
                        << "\t" << i
                        << "\t" << std::setprecision(6) << theta(i, k)
                        << "\t" << std::setprecision(6) << projectionB[i] << "\n";
                }
                if (opt.runRefit) {
                    unitFactorOut << "refit\t" << k
                        << "\t" << i
                        << "\t" << std::setprecision(6) << theta(i, k)
                        << "\t" << std::setprecision(6) << refitB[i] << "\n";
                }
            }
        }

    }

    if (opt.runRefit) {
        refitFactorOut.close();
        notice("Refit factor evaluation statistics written to %s", refitFactorEvalPath.c_str());
    }
    if (opt.runProjection) {
        projectionFactorOut.close();
        notice("Projection factor evaluation statistics written to %s", projectionFactorEvalPath.c_str());
    }
    if (opt.writeUnitFactorStats || !opt.unitFactorStatsOut.empty()) {
        unitFactorOut.close();
        notice("Per-factor unit-level statistics written to %s", unitFactorPath.c_str());
    }
    return 0;
}
