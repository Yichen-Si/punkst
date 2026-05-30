#include "factor_eval_common.hpp"
#include "eb_topic_activity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <span>
#include <sstream>

namespace {

using namespace factoreval;

struct FactorEvalOptions : FactorEvalCommonOptions {
    int32_t ebMaxIter = 10000;
    int32_t ebQuadSubdivisions = 8;
    int32_t ebTopComponents = 2;
    double refitMinTheta = 1e-5;
    double ebNullEps = 0.01;
    double ebTol = 1e-10;
    double ebPseudocount = 1e-8;
    double ebProbFloor = 1e-12;
    std::string ebMethod = "multinomial";
    std::vector<double> ebUniformSlabs;
    std::vector<double> ebBetaSlabs;
};

RowVectorXd renormalizedReducedThetaRow(const RowMajorMatrixXd& theta,
        int32_t unit, const std::vector<int32_t>& factors) {
    RowVectorXd out(factors.size());
    double s = 0.0;
    for (int32_t j = 0; j < static_cast<int32_t>(factors.size()); ++j) {
        out(j) = theta(unit, factors[j]);
        s += out(j);
    }
    if (s > 0.0) {
        out /= s;
    } else {
        out.setConstant(1.0 / static_cast<double>(factors.size()));
    }
    return out;
}

std::vector<ebact::Component> buildEbComponents(const FactorEvalOptions& opt) {
    std::vector<ebact::Component> comps;
    comps.push_back(ebact::Component::Uniform(0.0, opt.ebNullEps, true));
    if (opt.ebUniformSlabs.empty()) {
        const std::vector<double> bounds = {
            opt.ebNullEps, 0.05,
            0.05, 0.10,
            0.10, 0.25,
            0.25, 0.50,
            0.50, 1.00
        };
        for (size_t i = 0; i < bounds.size(); i += 2) {
            if (bounds[i] < bounds[i + 1]) {
                comps.push_back(ebact::Component::Uniform(bounds[i], bounds[i + 1]));
            }
        }
    } else {
        double left = opt.ebNullEps;
        for (double right : opt.ebUniformSlabs) {
            if (!(right > left && right <= 1.0)) {
                error("--eb-uniform-slabs breakpoints must be strictly increasing, greater than --eb-null-eps, and <= 1");
            }
            comps.push_back(ebact::Component::Uniform(left, right));
            left = right;
        }
        if (left < 1.0) {
            comps.push_back(ebact::Component::Uniform(left, 1.0));
        }
    }
    if (opt.ebBetaSlabs.size() % 2 != 0) {
        error("--eb-beta-slabs must contain c/d shape pairs");
    }
    for (size_t i = 0; i < opt.ebBetaSlabs.size(); i += 2) {
        comps.push_back(ebact::Component::Beta(opt.ebBetaSlabs[i], opt.ebBetaSlabs[i + 1]));
    }
    try {
        ebact::validate_components(comps);
    } catch (const std::exception& ex) {
        error("Invalid EB activity components: %s", ex.what());
    }
    return comps;
}

bool runEbBetaMarginal(const FactorEvalOptions& opt) {
    return opt.ebMethod == "both" || opt.ebMethod == "beta-marginal";
}

void writeEbUnitActivityRows(std::ofstream& out, int32_t factor,
        const ebact::FitResult& fit, int32_t topComponents) {
    const int32_t C = static_cast<int32_t>(fit.eta.size());
    const int32_t topK = std::min(topComponents, C);
    for (int32_t i = 0; i < static_cast<int32_t>(fit.resp.size()); ++i) {
        const double pActive = 1.0 - fit.z_null[i];
        if (!(pActive > 0.5)) {
            continue;
        }
        std::vector<std::pair<double, int32_t>> ranked;
        ranked.reserve(C);
        for (int32_t g = 0; g < C; ++g) {
            ranked.emplace_back(fit.resp[i][g], g);
        }
        std::partial_sort(ranked.begin(), ranked.begin() + topK, ranked.end(),
            [](const auto& a, const auto& b) {
                if (a.first != b.first) {
                    return a.first > b.first;
                }
                return a.second < b.second;
            });

        out << factor
            << "\t" << i
            << "\t" << std::fixed << std::setprecision(6) << pActive;
        for (int32_t j = 0; j < topK; ++j) {
            out << "\t" << ranked[j].second
                << "\t" << std::scientific << std::setprecision(4) << ranked[j].first;
        }
        out << "\n";
    }
}

struct EbComponentOutputs {
    std::ofstream uniform;
    std::ofstream beta;
    bool hasBeta = false;
};

std::string formatEbComponentParam(double x) {
    std::ostringstream ss;
    ss << std::setprecision(8) << x;
    return ss.str();
}

void writeEbComponentHeader(std::ofstream& out,
        const std::vector<ebact::Component>& comps, ebact::ComponentKind kind) {
    out << "method\tfactor";
    for (int32_t g = 0; g < static_cast<int32_t>(comps.size()); ++g) {
        const auto& comp = comps[g];
        if (comp.kind != kind) {
            continue;
        }
        out << "\tc" << g;
        if (kind == ebact::ComponentKind::Uniform) {
            out << "_" << formatEbComponentParam(comp.l)
                << "_" << formatEbComponentParam(comp.u);
        } else {
            out << "_" << formatEbComponentParam(comp.c)
                << "_" << formatEbComponentParam(comp.d);
        }
    }
    out << "\n";
}

void writeEbComponentRow(std::ofstream& out, const std::string& method, int32_t factor,
        const std::vector<ebact::Component>& comps, const ebact::FitResult& fit,
        ebact::ComponentKind kind) {
    out << method << "\t" << factor;
    for (int32_t g = 0; g < static_cast<int32_t>(comps.size()); ++g) {
        const auto& comp = comps[g];
        if (comp.kind == kind) {
            out << "\t" << std::fixed << std::setprecision(6) << fit.eta[g];
        }
    }
    out << std::defaultfloat << "\n";
}

void writeEbComponentRows(const std::string& method, int32_t factor,
        const std::vector<ebact::Component>& comps, const ebact::FitResult& fit,
        EbComponentOutputs& out) {
    writeEbComponentRow(out.uniform, method, factor, comps, fit, ebact::ComponentKind::Uniform);
    if (out.hasBeta) {
        writeEbComponentRow(out.beta, method, factor, comps, fit, ebact::ComponentKind::Beta);
    }
}

void writeEbResults(const std::string& method, int32_t factor,
        const std::vector<ebact::Component>& comps, const ebact::FitResult& fit,
        std::ofstream& factorOut, EbComponentOutputs& compOut) {
    double nActive = 0.0;
    for (double z : fit.z_null) {
        nActive += 1.0 - z;
    }
    factorOut << method
        << "\t" << factor
        << "\t" << std::fixed << std::setprecision(1) << nActive
        << "\t" << std::fixed << std::setprecision(6) << fit.eta[0]
        << "\t" << std::fixed << std::setprecision(2) << fit.loglik
        << "\t" << fit.iterations << std::defaultfloat << "\n";

    writeEbComponentRows(method, factor, comps, fit, compOut);
}

double multinomialFullReducedLogLr(const Document& doc, const RowVectorXd& p,
        const RowVectorXd& q, double probFloor) {
    const RowVectorXd pSafe = ebact::floored_normalized(p, probFloor);
    const RowVectorXd qSafe = ebact::floored_normalized(q, probFloor);
    double out = 0.0;
    for (size_t j = 0; j < doc.ids.size(); ++j) {
        const int32_t m = static_cast<int32_t>(doc.ids[j]);
        out += doc.cnts[j] * (std::log(pSafe(m)) - std::log(qSafe(m)));
    }
    return out;
}

int32_t posteriorComponentAssignment(const std::vector<double>& resp) {
    int32_t best = 0;
    for (int32_t g = 1; g < static_cast<int32_t>(resp.size()); ++g) {
        if (resp[g] > resp[best]) {
            best = g;
        }
    }
    return best;
}

void writeEbFactorEvalRows(std::ofstream& out, int32_t factor,
        const std::vector<Document>& docs, const RowMajorMatrixXd& theta,
        const std::vector<int32_t>& componentAssignments,
        const std::vector<double>& activity,
        const std::vector<double>& llr,
        int32_t nComponents) {
    for (int32_t cutoff = 1; cutoff < nComponents; ++cutoff) {
        int32_t n = 0;
        double thetaSum = 0.0;
        double s = 0.0;
        double w = 0.0;
        double info = 0.0;
        double avgLlrPerToken = 0.0;
        for (int32_t i = 0; i < static_cast<int32_t>(docs.size()); ++i) {
            if (componentAssignments[i] < cutoff) {
                continue;
            }
            const double ni = docs[i].get_raw_sum();
            ++n;
            thetaSum += theta(i, factor);
            s += activity[i] * ni;
            w += activity[i];
            info += llr[i];
            if (ni > 0.0) {
                avgLlrPerToken += llr[i] / ni;
            }
        }
        if (n > 0) {
            avgLlrPerToken /= static_cast<double>(n);
        }
        out << factor
            << "\t" << std::setprecision(8) << thetaSum
            << "\t" << cutoff
            << "\t" << n
            << "\t" << std::setprecision(6) << s
            << "\t" << std::setprecision(6) << w
            << "\t" << std::setprecision(6) << info
            << "\t" << std::setprecision(6) << avgLlrPerToken << "\n";
    }
}

void validateOptions(const FactorEvalOptions& opt) {
    validateFactorEvalCommonOptions(opt);
    if (opt.refitMinTheta >= 1.0) {
        error("--refit-min-theta must be less than 1");
    }
    if (opt.ebMethod == "beta-marginal") {
        error("lda-factor-eval requires multinomial EB evidence; use --eb-method multinomial or both");
    }
    if (opt.ebMethod != "both" && opt.ebMethod != "multinomial" && opt.ebMethod != "beta-marginal") {
        error("--eb-method must be one of: both, multinomial, beta-marginal");
    }
    if (!(opt.ebNullEps > 0.0 && opt.ebNullEps < 0.05)) {
        error("--eb-null-eps must be greater than 0 and less than 0.05");
    }
    if (opt.ebMaxIter <= 0) {
        error("--eb-max-iter must be positive");
    }
    if (opt.ebQuadSubdivisions <= 0) {
        error("--eb-quad-subdivisions must be positive");
    }
    if (opt.ebTopComponents <= 0) {
        error("--eb-top-components must be positive");
    }
    if (opt.ebTol <= 0.0) {
        error("--eb-tol must be positive");
    }
    if (opt.ebPseudocount < 0.0) {
        error("--eb-pseudocount must be non-negative");
    }
    if (!(opt.ebProbFloor > 0.0 && opt.ebProbFloor < 1.0)) {
        error("--eb-prob-floor must be in (0, 1)");
    }
}

} // namespace

int32_t cmdLDAFactorEval(int argc, char** argv) {
    FactorEvalOptions opt;

    ParamList pl;
    addFactorEvalCommonOptions(pl, opt,
        "Write per-factor unit-level EB activity statistics",
        "Output path for per-factor unit-level EB activity statistics");

    pl.add_option("refit-min-theta", "Approximate reduced-model theta by renormalizing full-model theta when focal theta is below this threshold", opt.refitMinTheta)
      .add_option("eb-method", "EB evidence method: both or multinomial; beta-marginal alone is not valid for factor evaluation", opt.ebMethod)
      .add_option("eb-null-eps", "Near-zero null interval upper bound", opt.ebNullEps)
      .add_option("eb-uniform-slabs", "Uniform active slab breakpoints; intervals are [eps,b1], [b1,b2], ... and [b_max,1] if b_max < 1", opt.ebUniformSlabs)
      .add_option("eb-beta-slabs", "Beta active slab c/d shape pairs", opt.ebBetaSlabs)
      .add_option("eb-max-iter", "Maximum EM iterations for EB mixture weights", opt.ebMaxIter)
      .add_option("eb-tol", "EB EM convergence tolerance", opt.ebTol)
      .add_option("eb-pseudocount", "EB mixture weight pseudocount", opt.ebPseudocount)
      .add_option("eb-prob-floor", "Probability floor before multinomial EB evidence", opt.ebProbFloor)
      .add_option("eb-quad-subdivisions", "Number of Gauss-Legendre subintervals for EB integrals", opt.ebQuadSubdivisions)
      .add_option("eb-top-components", "Number of top EB components to write in method-specific unit activity files", opt.ebTopComponents);

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
    pl.print_options();

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
        reader.setWeights(opt.weightFile, opt.defaultWeight);
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

    RowMajorMatrixXd theta = lda.do_transform(std::span<const Document>(data.docs.data(), data.docs.size()));
    RowMajorMatrixXd gammaCounts;
    const double ldaAlpha = lda.get_doc_topic_prior();
    if (runEbBetaMarginal(opt)) {
        gammaCounts = lda.do_transform_gamma(std::span<const Document>(data.docs.data(), data.docs.size()));
    }
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

    const std::string factorEvalPath = opt.outPrefix + ".factor_eval.tsv";
    std::ofstream factorOut(factorEvalPath);
    if (!factorOut) {
        error("Error opening output file: %s for writing", factorEvalPath.c_str());
    }
    factorOut << "factor\ttheta_sum\tc\tN\tS\tW\tI\tavg_llr_per_token\n";

    std::ofstream unitFactorOut;
    std::string unitFactorPath = opt.unitFactorStatsOut.empty()
        ? opt.outPrefix + ".unit_factor_eval.tsv"
        : opt.unitFactorStatsOut;
    if (opt.writeUnitFactorStats || !opt.unitFactorStatsOut.empty()) {
        unitFactorOut.open(unitFactorPath);
        if (!unitFactorOut) {
            error("Error opening output file: %s for writing", unitFactorPath.c_str());
        }
        unitFactorOut << "factor\tunit_id\ttheta_factor\tcomponent\ta\tllr\tllr_per_token\n";
    }

    std::vector<ebact::Component> ebComponents;
    std::ofstream ebFactorOut;
    EbComponentOutputs ebCompOut;
    std::ofstream ebMultinomialUnitOut;
    std::ofstream ebBetaMarginalUnitOut;
    ebComponents = buildEbComponents(opt);
    const std::string ebFactorPath = opt.outPrefix + ".eb_factor_activity.tsv";
    ebFactorOut.open(ebFactorPath);
    if (!ebFactorOut) {
        error("Error opening output file: %s for writing", ebFactorPath.c_str());
    }
    ebFactorOut << "method\tfactor\tN_active\teta_null\tloglik\titerations\n";
    ebCompOut.hasBeta = std::any_of(ebComponents.begin(), ebComponents.end(),
        [](const ebact::Component& comp) { return comp.kind == ebact::ComponentKind::Beta; });
    const std::string uniformCompPath = ebCompOut.hasBeta
        ? opt.outPrefix + ".uniform.eb_components.tsv"
        : opt.outPrefix + ".eb_components.tsv";
    ebCompOut.uniform.open(uniformCompPath);
    if (!ebCompOut.uniform) {
        error("Error opening output file: %s for writing", uniformCompPath.c_str());
    }
    writeEbComponentHeader(ebCompOut.uniform, ebComponents, ebact::ComponentKind::Uniform);
    if (ebCompOut.hasBeta) {
        const std::string betaCompPath = opt.outPrefix + ".beta.eb_components.tsv";
        ebCompOut.beta.open(betaCompPath);
        if (!ebCompOut.beta) {
            error("Error opening output file: %s for writing", betaCompPath.c_str());
        }
        writeEbComponentHeader(ebCompOut.beta, ebComponents, ebact::ComponentKind::Beta);
    }
    auto openUnitActivity = [&](std::ofstream& out, const std::string& method) {
        const std::string label = method.empty() ? "" : ("." + method);
        const std::string path = opt.outPrefix + label + ".unit_activity.tsv";
        out.open(path);
        if (!out) {
            error("Error opening output file: %s for writing", path.c_str());
        }
        out << "factor\tunit_id\tp_active";
        const int32_t topK = std::min(opt.ebTopComponents, static_cast<int32_t>(ebComponents.size()));
        for (int32_t j = 1; j <= topK; ++j) {
            out << "\tc_" << j << "\tresp_" << j;
        }
        out << "\n";
    };
    openUnitActivity(ebMultinomialUnitOut, "");
    if (runEbBetaMarginal(opt)) {
        openUnitActivity(ebBetaMarginalUnitOut, "marginal");
    }

    const RowMajorMatrixXd originalModel = lda.copy_model_matrix();
    double ebOrgRuntimeTotal = 0.0;
    double ebMarginalRuntimeTotal = 0.0;
    for (int32_t k : candidateStats.candidates) {
        const std::vector<int32_t> factors = reducedFactorMap(K, k);
        const RowMajorMatrixXd betaReduced = reducedBetaRows(betaNorm, factors);
        auto t0 = std::chrono::steady_clock::now();
        RowMajorMatrixXd reducedModel = removeFactorRows(originalModel, k);
        LatentDirichletAllocation reducedLda(reducedModel, opt.seed, opt.nThreads,
            opt.verbose, InferenceType::SVB, ldaAlpha);
        reducedLda.set_svb_parameters(opt.maxIter, opt.meanChangeTol);
        RowMajorMatrixXd thetaReducedAll(theta.rows(), K - 1);
        std::vector<int32_t> refitRows;
        refitRows.reserve(theta.rows());
        std::vector<Document> refitDocs;
        refitDocs.reserve(theta.rows());
        for (int32_t i = 0; i < theta.rows(); ++i) {
            if (opt.refitMinTheta <= 0.0 || theta(i, k) >= opt.refitMinTheta) {
                refitRows.push_back(i);
                refitDocs.push_back(data.docs[i]);
            } else {
                thetaReducedAll.row(i) = renormalizedReducedThetaRow(theta, i, factors);
            }
        }
        notice("Leave factor %d out, refit %d units",
            k, static_cast<int32_t>(refitRows.size()));
        if (!refitDocs.empty()) {
            RowMajorMatrixXd thetaReducedRefit = reducedLda.transform(
                std::span<const Document>(refitDocs.data(), refitDocs.size()));
            for (int32_t local = 0; local < static_cast<int32_t>(refitRows.size()); ++local) {
                thetaReducedAll.row(refitRows[local]) = thetaReducedRefit.row(local);
            }
        }
        if (opt.refitMinTheta > 0.0) {
            notice("Factor %d reduced-model refit: refit %d units, approximated %d units", k, static_cast<int32_t>(refitRows.size()),
                static_cast<int32_t>(theta.rows()) - static_cast<int32_t>(refitRows.size()));
        }
        RowMajorMatrixXd q = thetaReducedAll * betaReduced;
        std::vector<std::vector<double>> logH = ebact::multinomial_log_evidence(
            data.docs, q, betaNorm.row(k), ebComponents,
            opt.ebProbFloor, opt.ebQuadSubdivisions, opt.nThreads);
        ebact::FitResult fit = ebact::fit_eta_em(logH, opt.ebMaxIter,
            opt.ebTol, opt.ebPseudocount);

        std::vector<int32_t> componentAssignments(theta.rows(), 0);
        std::vector<double> llr(theta.rows(), 0.0);
        for (int32_t i = 0; i < theta.rows(); ++i) {
            componentAssignments[i] = posteriorComponentAssignment(fit.resp[i]);
            const RowVectorXd p = theta.row(i) * betaNorm;
            llr[i] = multinomialFullReducedLogLr(data.docs[i], p, q.row(i), opt.ebProbFloor);
        }
        std::vector<double> activity = ebact::multinomial_posterior_activity_means_from_logH(
            data.docs, q, betaNorm.row(k), ebComponents, fit, componentAssignments, logH,
            opt.ebProbFloor, opt.ebQuadSubdivisions, opt.nThreads);
        auto t1 = std::chrono::steady_clock::now();
        const double runtime = std::chrono::duration<double>(t1 - t0).count();
        ebOrgRuntimeTotal += runtime;

        writeEbResults("org", k, ebComponents, fit, ebFactorOut, ebCompOut);
        writeEbUnitActivityRows(ebMultinomialUnitOut, k, fit, opt.ebTopComponents);
        writeEbFactorEvalRows(factorOut, k, data.docs, theta, componentAssignments,
            activity, llr, static_cast<int32_t>(ebComponents.size()));

        if (opt.writeUnitFactorStats || !opt.unitFactorStatsOut.empty()) {
            for (int32_t i = 0; i < theta.rows(); ++i) {
                const double ni = data.docs[i].get_raw_sum();
                unitFactorOut << k
                    << "\t" << i
                    << "\t" << std::setprecision(6) << theta(i, k)
                    << "\t" << componentAssignments[i]
                    << "\t" << std::setprecision(6) << activity[i]
                    << "\t" << std::setprecision(6) << llr[i]
                    << "\t" << std::setprecision(6) << (ni > 0.0 ? llr[i] / ni : 0.0) << "\n";
            }
        }

        if (runEbBetaMarginal(opt)) {
            auto tm0 = std::chrono::steady_clock::now();
            std::vector<std::vector<double>> marginalLogH = ebact::beta_marginal_log_evidence(
                gammaCounts, k, ldaAlpha, ebComponents, opt.ebQuadSubdivisions);
            ebact::FitResult marginalFit = ebact::fit_eta_em(marginalLogH, opt.ebMaxIter,
                opt.ebTol, opt.ebPseudocount);
            auto tm1 = std::chrono::steady_clock::now();
            const double marginalRuntime = std::chrono::duration<double>(tm1 - tm0).count();
            ebMarginalRuntimeTotal += marginalRuntime;
            writeEbResults("marginal", k, ebComponents, marginalFit, ebFactorOut, ebCompOut);
            writeEbUnitActivityRows(ebBetaMarginalUnitOut, k, marginalFit, opt.ebTopComponents);
        }
    }

    factorOut.close();
    notice("EB component-threshold factor evaluation statistics written to %s", factorEvalPath.c_str());
    if (opt.writeUnitFactorStats || !opt.unitFactorStatsOut.empty()) {
        unitFactorOut.close();
        notice("Per-factor unit-level statistics written to %s", unitFactorPath.c_str());
    }
    ebFactorOut.close();
    ebCompOut.uniform.close();
    if (ebCompOut.hasBeta) {
        ebCompOut.beta.close();
    }
    ebMultinomialUnitOut.close();
    if (runEbBetaMarginal(opt)) {
        ebBetaMarginalUnitOut.close();
    }
    notice("EB topic-activity org total runtime: %.6f sec", ebOrgRuntimeTotal);
    if (runEbBetaMarginal(opt)) {
        notice("EB topic-activity marginal total runtime: %.6f sec", ebMarginalRuntimeTotal);
    }
    notice("EB topic-activity statistics written with prefix %s", opt.outPrefix.c_str());

    return 0;
}
