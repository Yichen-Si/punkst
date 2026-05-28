#include "topic_svb.hpp"
#include "eb_topic_activity.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <numeric>
#include <span>

namespace {

struct FactorEvalOptions {
    std::string inFile;
    std::string metaFile;
    std::string modelFile;
    std::string outPrefix;
    std::string featureFile;
    std::string weightFile;
    std::vector<std::string> dgeDirs;
    std::vector<std::string> inBarcodes;
    std::vector<std::string> inFeatures;
    std::vector<std::string> inMatrix;
    std::vector<std::string> datasetIds;
    std::string includeFeatureRegex;
    std::string excludeFeatureRegex;
    std::string unitFactorStatsOut;
    std::vector<double> thresholds;
    int32_t seed = -1;
    int32_t nThreads = 1;
    int32_t modal = 0;
    int32_t minCountFeature = 1;
    int32_t debug = 0;
    int32_t verbose = 0;
    int32_t maxIter = 100;
    int32_t projectionMaxIter = 200;
    int32_t ebMaxIter = 10000;
    int32_t ebQuadSubdivisions = 8;
    double minCount = 20.0;
    double defaultWeight = 1.0;
    double meanChangeTol = 1e-3;
    double candidateThreshold = 0.005;
    double projectionTol = 1e-7;
    double alpha = -1.0;
    double ebNullEps = 0.01;
    double ebTol = 1e-10;
    double ebPseudocount = 1e-8;
    double ebProbFloor = 1e-12;
    std::string ebMethod = "both";
    std::vector<double> ebUniformSlabs;
    std::vector<double> ebBetaSlabs;
    bool writeUnitFactorStats = false;
    bool runProjection = false;
    bool runRefit = false;
    bool ebActivity = false;
};

struct EvaluationData {
    std::vector<Document> docs;
};

struct UnitGofStats {
    VectorXd residuals;
    VectorXd cosineSim;
    VectorXd entropy;
    VectorXd shLcr;
    VectorXd shQ;
};

void applyWeights(std::vector<Document>& docs, const LDA4Hex& lda) {
    for (auto& doc : docs) {
        lda.applyWeights(doc);
    }
}

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

RowMajorMatrixXd removeFactorRows(const RowMajorMatrixXd& mtx, int32_t drop) {
    RowMajorMatrixXd out(mtx.rows() - 1, mtx.cols());
    int32_t r = 0;
    for (int32_t k = 0; k < mtx.rows(); ++k) {
        if (k == drop) {
            continue;
        }
        out.row(r++) = mtx.row(k);
    }
    return out;
}

std::vector<int32_t> reducedFactorMap(int32_t K, int32_t drop) {
    std::vector<int32_t> factors;
    factors.reserve(K - 1);
    for (int32_t k = 0; k < K; ++k) {
        if (k != drop) {
            factors.push_back(k);
        }
    }
    return factors;
}

RowMajorMatrixXd reducedBetaRows(const RowMajorMatrixXd& beta, const std::vector<int32_t>& factors) {
    RowMajorMatrixXd out(factors.size(), beta.cols());
    for (int32_t j = 0; j < static_cast<int32_t>(factors.size()); ++j) {
        out.row(j) = beta.row(factors[j]);
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
        if (opt.ebUniformSlabs.size() % 2 != 0) {
            error("--eb-uniform-slabs must contain l/u pairs");
        }
        for (size_t i = 0; i < opt.ebUniformSlabs.size(); i += 2) {
            comps.push_back(ebact::Component::Uniform(opt.ebUniformSlabs[i], opt.ebUniformSlabs[i + 1]));
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

bool runEbMultinomial(const FactorEvalOptions& opt) {
    return opt.ebActivity && (opt.ebMethod == "both" || opt.ebMethod == "multinomial");
}

bool runEbBetaMarginal(const FactorEvalOptions& opt) {
    return opt.ebActivity && (opt.ebMethod == "both" || opt.ebMethod == "beta-marginal");
}

void writeEbResults(const std::string& method, int32_t k, const std::string& factorName,
        const std::vector<ebact::Component>& comps, const ebact::FitResult& fit,
        const RowMajorMatrixXd& theta, double nullEps, double runtime,
        std::ofstream& factorOut, std::ofstream& compOut, std::ofstream& unitOut) {
    double nActive = 0.0;
    for (double z : fit.z_null) {
        nActive += 1.0 - z;
    }
    factorOut << method
        << "\t" << factorName
        << "\t" << k
        << "\t" << std::setprecision(6) << nullEps
        << "\t" << fit.z_null.size()
        << "\t" << std::setprecision(8) << nActive
        << "\t" << std::setprecision(8) << fit.eta[0]
        << "\t" << std::setprecision(10) << fit.loglik
        << "\t" << fit.iterations
        << "\t" << std::setprecision(6) << runtime << "\n";

    for (int32_t g = 0; g < static_cast<int32_t>(comps.size()); ++g) {
        const auto& comp = comps[g];
        compOut << method
            << "\t" << factorName
            << "\t" << k
            << "\t" << g
            << "\t" << (comp.is_null ? 1 : 0)
            << "\t" << ebact::component_kind_name(comp.kind)
            << "\t" << std::setprecision(8) << comp.l
            << "\t" << std::setprecision(8) << comp.u
            << "\t" << std::setprecision(8) << comp.c
            << "\t" << std::setprecision(8) << comp.d
            << "\t" << std::setprecision(8) << fit.eta[g] << "\n";
    }

    for (int32_t i = 0; i < static_cast<int32_t>(fit.resp.size()); ++i) {
        int32_t best = 0;
        for (int32_t g = 1; g < static_cast<int32_t>(fit.resp[i].size()); ++g) {
            if (fit.resp[i][g] > fit.resp[i][best]) {
                best = g;
            }
        }
        unitOut << method
            << "\t" << factorName
            << "\t" << k
            << "\t" << i
            << "\t" << std::setprecision(8) << theta(i, k)
            << "\t" << std::setprecision(8) << fit.z_null[i]
            << "\t" << std::setprecision(8) << (1.0 - fit.z_null[i])
            << "\t" << best;
        for (double r : fit.resp[i]) {
            unitOut << "\t" << std::setprecision(8) << r;
        }
        unitOut << "\n";
    }
}

UnitGofStats computeUnitGof(const std::vector<Document>& docs, const RowMajorMatrixXd& theta,
        const RowMajorMatrixXd& betaNorm) {
    const int32_t N = static_cast<int32_t>(docs.size());
    const int32_t M = static_cast<int32_t>(betaNorm.cols());
    UnitGofStats stats;
    stats.residuals = VectorXd::Zero(N);
    stats.cosineSim = VectorXd::Zero(N);
    const MatrixXd topicSimilarity = pairwiseCosineSimilarityRows(betaNorm);
    const ThetaEntropyStats thetaStats = computeThetaEntropyStats(theta, topicSimilarity);
    stats.entropy = thetaStats.entropy;
    stats.shLcr = thetaStats.sh_lcr;
    stats.shQ = thetaStats.sh_q;

    RowVectorXd expected = RowVectorXd::Zero(M);
    for (int32_t i = 0; i < N; ++i) {
        const Document& doc = docs[i];
        const double weightedTotal = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
        expected.noalias() = theta.row(i) * betaNorm;
        expected *= weightedTotal;

        double cosine = 0.0;
        double observedNormSq = 0.0;
        const double expectedNormSq = expected.squaredNorm();
        double docResidual = expected.sum();
        for (size_t j = 0; j < doc.ids.size(); ++j) {
            const uint32_t m = doc.ids[j];
            const double observed = doc.cnts[j];
            const double estimate = expected(m);
            docResidual += std::abs(estimate - observed) - estimate;
            cosine += estimate * observed;
            observedNormSq += observed * observed;
        }
        if (expectedNormSq > 0.0 && observedNormSq > 0.0) {
            cosine /= std::sqrt(expectedNormSq * observedNormSq);
        }
        stats.residuals(i) = docResidual;
        stats.cosineSim(i) = cosine;
    }
    return stats;
}

void writeUnitGof(const std::string& path, const std::vector<Document>& docs,
        const UnitGofStats& stats) {
    std::ofstream out(path);
    if (!out) {
        error("Error opening output file: %s for writing", path.c_str());
    }
    out << "#unit_id\ttotal_count\tresidual\tcosine_sim\tentropy\tsh_lcr\tsh_q\n";
    out << std::fixed;
    for (int32_t i = 0; i < static_cast<int32_t>(docs.size()); ++i) {
        out << i
            << "\t" << static_cast<int64_t>(std::llround(docs[i].get_raw_sum()))
            << "\t" << std::setprecision(2) << stats.residuals(i)
            << "\t" << std::setprecision(4) << stats.cosineSim(i)
            << "\t" << std::setprecision(4) << stats.entropy(i)
            << "\t" << std::setprecision(4) << stats.shLcr(i)
            << "\t" << std::setprecision(4) << stats.shQ(i) << "\n";
    }
}

EvaluationData loadEvaluationData(FactorEvalOptions& opt, LDA4Hex& lda, DGEReader10X* dge) {
    EvaluationData data;
    const int32_t minCountInt = opt.minCount > 0 ? static_cast<int32_t>(std::ceil(opt.minCount)) : 0;
    const int32_t maxUnits = opt.debug > 0 ? opt.debug : INT32_MAX;
    if (dge != nullptr) {
        dge->readAll(data.docs, minCountInt);
        if (static_cast<int32_t>(data.docs.size()) > maxUnits) {
            data.docs.resize(maxUnits);
        }
        applyWeights(data.docs, lda);
        return data;
    }

    lda.readAllDocuments(data.docs, opt.inFile, minCountInt, maxUnits);
    return data;
}

void validateOptions(const FactorEvalOptions& opt) {
    if (opt.candidateThreshold <= 0.0) {
        error("--candidate-threshold must be greater than 0");
    }
    if (opt.candidateThreshold > 1.0) {
        error("--candidate-threshold is a fractional abundance threshold and must be <= 1");
    }
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
    if (opt.alpha != -1.0 && opt.alpha <= 0.0) {
        error("--alpha must be positive");
    }
    if (opt.ebActivity) {
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
}

} // namespace

int32_t cmdLDAFactorEval(int argc, char** argv) {
    FactorEvalOptions opt;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", opt.inFile)
      .add_option("in-meta", "Metadata file", opt.metaFile)
      .add_option("in-model", "Input model matrix (topic-word) file", opt.modelFile, true)
      .add_option("out-prefix", "Output prefix for evaluation files", opt.outPrefix, true)
      .add_option("modal", "Modality to use (0-based)", opt.modal)
      .add_option("threads", "Number of threads", opt.nThreads)
      .add_option("seed", "Random seed", opt.seed)
      .add_option("verbose", "Verbose level", opt.verbose)
      .add_option("debug", "If >0, only process this many units", opt.debug);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", opt.dgeDirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", opt.inBarcodes)
      .add_option("in-features", "Input features.tsv.gz", opt.inFeatures)
      .add_option("in-matrix", "Input matrix.mtx.gz", opt.inMatrix)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", opt.datasetIds);

    pl.add_option("features", "Feature names and total counts file", opt.featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", opt.minCountFeature)
      .add_option("min-count", "Minimum total feature count for a unit to be kept", opt.minCount)
      .add_option("feature-weights", "Input weights file", opt.weightFile)
      .add_option("default-weight", "Default weight for features not in weight file", opt.defaultWeight)
      .add_option("include-feature-regex", "Regex for including features", opt.includeFeatureRegex)
      .add_option("exclude-feature-regex", "Regex for excluding features", opt.excludeFeatureRegex);

    pl.add_option("max-iter", "Max iterations per document", opt.maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per document", opt.meanChangeTol)
      .add_option("candidate-threshold", "Maximum fractional theta abundance for candidate factors", opt.candidateThreshold)
      .add_option("thresholds", "Theta thresholds for selecting units and computing factor-level summaries", opt.thresholds, true)
      .add_option("projection-max-iter", "Maximum projected-gradient iterations", opt.projectionMaxIter)
      .add_option("projection-tol", "Projection convergence tolerance", opt.projectionTol)
      .add_option("alpha", "Document-topic prior override for LDA transform and EB beta-marginal analysis", opt.alpha)
      .add_option("projection", "Run direct convex projection leave-one-out evaluation", opt.runProjection)
      .add_option("refit", "Run reduced fixed-model refit leave-one-out evaluation", opt.runRefit)
      .add_option("write-unit-factor-stats", "Write per-factor unit-level leave-one-out statistics", opt.writeUnitFactorStats)
      .add_option("unit-factor-stats-out", "Output path for per-factor unit-level leave-one-out statistics", opt.unitFactorStatsOut);

    pl.add_option("eb-activity", "Run sparse EB topic-activity analysis for selected candidate factors", opt.ebActivity)
      .add_option("eb-method", "EB evidence method: both, multinomial, or beta-marginal", opt.ebMethod)
      .add_option("eb-null-eps", "Near-zero null interval upper bound", opt.ebNullEps)
      .add_option("eb-uniform-slabs", "Uniform active slab l/u pairs", opt.ebUniformSlabs)
      .add_option("eb-beta-slabs", "Beta active slab c/d shape pairs", opt.ebBetaSlabs)
      .add_option("eb-max-iter", "Maximum EM iterations for EB mixture weights", opt.ebMaxIter)
      .add_option("eb-tol", "EB EM convergence tolerance", opt.ebTol)
      .add_option("eb-pseudocount", "EB mixture weight pseudocount", opt.ebPseudocount)
      .add_option("eb-prob-floor", "Probability floor before multinomial EB evidence", opt.ebProbFloor)
      .add_option("eb-quad-subdivisions", "Number of Gauss-Legendre subintervals for EB integrals", opt.ebQuadSubdivisions);

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

    const std::vector<std::string>& topicNames = lda.get_topic_names();
    std::vector<double> candidateWeights(K, 0.0);
    std::vector<double> candidateFractions(K, 0.0);
    std::vector<double> thetaWeightedCounts(K, 0.0);
    std::vector<std::vector<int32_t>> thetaHist(K, std::vector<int32_t>(100, 0));
    double totalTheta = 0.0;
    std::vector<int32_t> candidates;
    for (int32_t i = 0; i < theta.rows(); ++i) {
        const double ni = data.docs[i].get_raw_sum();
        for (int32_t k = 0; k < K; ++k) {
            const double thetaIk = theta(i, k);
            candidateWeights[k] += thetaIk;
            thetaWeightedCounts[k] += thetaIk * ni;
            totalTheta += thetaIk;
            const int32_t bin = std::min(99, std::max(0, static_cast<int32_t>(std::floor(thetaIk * 100.0))));
            ++thetaHist[k][bin];
        }
    }
    if (totalTheta <= 0.0) {
        error("Total theta abundance is not positive");
    }
    for (int32_t k = 0; k < K; ++k) {
        if (candidateWeights[k] <= 1) {
            continue;
        }
        candidateFractions[k] = candidateWeights[k] / totalTheta;
        if (candidateFractions[k] <= opt.candidateThreshold) {
            candidates.push_back(k);
        }
    }
    const std::string thetaHistPath = opt.outPrefix + ".theta_hist.tsv";
    std::ofstream thetaHistOut(thetaHistPath);
    if (!thetaHistOut) {
        error("Error opening output file: %s for writing", thetaHistPath.c_str());
    }
    thetaHistOut << "factor\tk\tbin_start\tbin_end\tN\n";
    for (int32_t k = 0; k < K; ++k) {
        const std::string factorName = k < static_cast<int32_t>(topicNames.size())
            ? topicNames[k]
            : std::to_string(k);
        for (int32_t b = 0; b < 100; ++b) {
            thetaHistOut << factorName
                << "\t" << k
                << "\t" << std::setprecision(2) << (static_cast<double>(b) / 100.0)
                << "\t" << std::setprecision(2) << (static_cast<double>(b + 1) / 100.0)
                << "\t" << thetaHist[k][b] << "\n";
        }
    }
    thetaHistOut.close();
    notice("Per-factor theta histograms written to %s", thetaHistPath.c_str());

    if (candidates.size() == 0) {
        notice("No candidate factors selected; exiting");
        return 0;
    }

    notice("Selected %d candidate factors using theta fraction <= %.6g",
        static_cast<int32_t>(candidates.size()), opt.candidateThreshold);

    const std::string factorEvalPath = opt.outPrefix + ".factor_eval.tsv";
    std::ofstream factorOut(factorEvalPath);
    if (!factorOut) {
        error("Error opening output file: %s for writing", factorEvalPath.c_str());
    }
    factorOut << "method\tfactor\ttheta_sum\tweight\ttau\tN\tS\tW\tI\truntime_sec\n";
    // factorOut << std::fixed;

    std::ofstream unitFactorOut;
    std::string unitFactorPath = opt.unitFactorStatsOut.empty()
        ? opt.outPrefix + ".unit_factor_eval.tsv"
        : opt.unitFactorStatsOut;
    if (opt.writeUnitFactorStats || !opt.unitFactorStatsOut.empty()) {
        unitFactorOut.open(unitFactorPath);
        if (!unitFactorOut) {
            error("Error opening output file: %s for writing", unitFactorPath.c_str());
        }
        unitFactorOut << "method\tk\tunit_id\ttheta_k\tB\n";
        // unitFactorOut << std::fixed;
    }

    std::vector<ebact::Component> ebComponents;
    std::ofstream ebFactorOut;
    std::ofstream ebCompOut;
    std::ofstream ebUnitOut;
    if (opt.ebActivity) {
        ebComponents = buildEbComponents(opt);
        const std::string ebFactorPath = opt.outPrefix + ".eb_factor_activity.tsv";
        const std::string ebCompPath = opt.outPrefix + ".eb_components.tsv";
        const std::string ebUnitPath = opt.outPrefix + ".eb_unit_activity.tsv";
        ebFactorOut.open(ebFactorPath);
        ebCompOut.open(ebCompPath);
        ebUnitOut.open(ebUnitPath);
        if (!ebFactorOut || !ebCompOut || !ebUnitOut) {
            error("Error opening EB activity output files for writing");
        }
        ebFactorOut << "method\tfactor\tk\tnull_eps\tN\tN_active\teta_null\tloglik\titerations\truntime_sec\n";
        ebCompOut << "method\tfactor\tk\tcomponent\tis_null\tkind\tl\tu\tc\td\teta\n";
        ebUnitOut << "method\tfactor\tk\tunit_id\ttheta_k\tz_null\tp_active\tbest_component";
        for (int32_t g = 0; g < static_cast<int32_t>(ebComponents.size()); ++g) {
            ebUnitOut << "\tresp_" << g;
        }
        ebUnitOut << "\n";
    }

    const RowMajorMatrixXd originalModel = lda.copy_model_matrix();
    for (int32_t k : candidates) {
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
                thetaReduced = reducedLda.transform(std::span<const Document>(selectedDocs.data(), selectedDocs.size()));
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

        auto writeFactorRows = [&](const char* method, const std::vector<double>& dvals,
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
                factorOut << method
                    << "\t" << (k < static_cast<int32_t>(topicNames.size()) ? topicNames[k] : std::to_string(k))
                    << "\t" << std::setprecision(8) << candidateWeights[k]
                    << "\t" << std::setprecision(8) << thetaWeightedCounts[k]
                    << "\t" << std::setprecision(2) << tau
                    << "\t" << n[t]
                    << "\t" << std::setprecision(2) << totalCount[t]
                    << "\t" << std::setprecision(2) << w[t]
                    << "\t" << std::setprecision(6) << infoLoss[t]
                    << "\t" << std::setprecision(6) << runtime << "\n";
            }
        };
        if (opt.runProjection) {
            writeFactorRows("projection", projectionD, projectionRuntime);
        }
        if (opt.runRefit) {
            writeFactorRows("refit", refitD, refitRuntime);
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

        if (opt.ebActivity) {
            const std::string factorName = k < static_cast<int32_t>(topicNames.size())
                ? topicNames[k]
                : std::to_string(k);

            if (runEbMultinomial(opt)) {
                auto t0 = std::chrono::steady_clock::now();
                RowMajorMatrixXd reducedModel = removeFactorRows(originalModel, k);
                LatentDirichletAllocation reducedLda(reducedModel, opt.seed, opt.nThreads,
                    opt.verbose, InferenceType::SVB, ldaAlpha);
                reducedLda.set_svb_parameters(opt.maxIter, opt.meanChangeTol);
                RowMajorMatrixXd thetaReducedAll = reducedLda.transform(
                    std::span<const Document>(data.docs.data(), data.docs.size()));
                RowMajorMatrixXd q = thetaReducedAll * betaReduced;
                std::vector<std::vector<double>> logH = ebact::multinomial_log_evidence(
                    data.docs, q, betaNorm.row(k), ebComponents,
                    opt.ebProbFloor, opt.ebQuadSubdivisions);
                ebact::FitResult fit = ebact::fit_eta_em(logH, opt.ebMaxIter,
                    opt.ebTol, opt.ebPseudocount);
                auto t1 = std::chrono::steady_clock::now();
                const double runtime = std::chrono::duration<double>(t1 - t0).count();
                writeEbResults("multinomial", k, factorName, ebComponents, fit, theta,
                    opt.ebNullEps, runtime, ebFactorOut, ebCompOut, ebUnitOut);
            }

            if (runEbBetaMarginal(opt)) {
                auto t0 = std::chrono::steady_clock::now();
                std::vector<std::vector<double>> logH = ebact::beta_marginal_log_evidence(
                    gammaCounts, k, ldaAlpha, ebComponents, opt.ebQuadSubdivisions);
                ebact::FitResult fit = ebact::fit_eta_em(logH, opt.ebMaxIter,
                    opt.ebTol, opt.ebPseudocount);
                auto t1 = std::chrono::steady_clock::now();
                const double runtime = std::chrono::duration<double>(t1 - t0).count();
                writeEbResults("beta-marginal", k, factorName, ebComponents, fit, theta,
                    opt.ebNullEps, runtime, ebFactorOut, ebCompOut, ebUnitOut);
            }
        }
    }

    factorOut.close();
    notice("Factor evaluation statistics written to %s", factorEvalPath.c_str());
    if (unitFactorOut) {
        unitFactorOut.close();
        notice("Per-factor unit-level statistics written to %s", unitFactorPath.c_str());
    }
    if (opt.ebActivity) {
        ebFactorOut.close();
        ebCompOut.close();
        ebUnitOut.close();
        notice("EB topic-activity statistics written with prefix %s", opt.outPrefix.c_str());
    }

    return 0;
}
