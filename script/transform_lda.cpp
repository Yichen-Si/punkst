#include "topic_svb.hpp"

#include <fstream>
#include <iomanip>
#include <sstream>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

int32_t cmdLDATransform(int argc, char** argv) {
    std::string inFile, metaFile, modelFile, outPrefix, featureFile, weightFile;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t seed = -1;
    int32_t batchSize = 1024;
    int32_t nThreads = 1;
    int32_t modal = 0;
    int32_t minCountFeature = 1;
    double minCount = 20;
    int32_t debug_ = 0;
    int32_t verbose = 0;
    double defaultWeight = 1.0;
    int32_t maxIter = 100;
    double mDelta = 1e-3;
    bool featureResiduals = false;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile, true)
      .add_option("in-meta", "Metadata file", metaFile, true)
      .add_option("in-model", "Input model matrix (topic-word) file", modelFile, true)
      .add_option("out-prefix", "Output prefix for results files", outPrefix, true)
      .add_option("minibatch-size", "Minibatch size", batchSize)
      .add_option("modal", "Modality to use (0-based)", modal)
      .add_option("threads", "Number of threads", nThreads)
      .add_option("seed", "Random seed", seed)
      .add_option("verbose", "Verbose level", verbose)
      .add_option("debug", "If >0, only process this many units", debug_);

    pl.add_option("features", "Feature names and total counts file", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", minCountFeature)
      .add_option("min-count", "Minimum total feature count for a unit to be kept", minCount)
      .add_option("feature-weights", "Input weights file", weightFile)
      .add_option("default-weight", "Default weight for features not in weight file", defaultWeight)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex);

    pl.add_option("max-iter", "Max iterations per document", maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per document", mDelta)
      .add_option("feature-residuals", "Compute per-feature residuals", featureResiduals);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    if (batchSize <= 0) {
        batchSize = 512;
        warning("Minibatch size must be greater than 0, using default value of %d", batchSize);
    }
    if (seed <= 0) {
        seed = std::random_device{}();
    }

    HexReader reader(metaFile);
    if (!featureFile.empty()) {
        reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
    }
    if (!weightFile.empty()) {
        reader.setWeights(weightFile, defaultWeight);
    }

    LDA4Hex lda(reader, modal, verbose);
    lda.initialize_transform(modelFile,
        seed, nThreads, verbose, maxIter, mDelta);

    const int32_t M = lda.nFeatures();
    const int32_t K = lda.getNumTopics();

    RowMajorMatrixXd beta_norm;
    if (featureResiduals) {
        const RowMajorMatrixXd& model = lda.get_model_matrix();
        beta_norm = rowNormalize(model);
    }

    std::ifstream inFileStream(inFile);
    if (!inFileStream) error("Error opening input file: %s", inFile.c_str());

    std::string outFile = outPrefix + ".results.tsv";
    std::ofstream outFileStream(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());

    std::string header;
    reader.getInfoHeaderStr(header);
    outFileStream << "#" << header << "\t";
    lda.writeUnitHeader(outFileStream);
    outFileStream << std::fixed << std::setprecision(4);

    RowMajorMatrixXd pseudobulk = RowMajorMatrixXd::Zero(M, K);
    VectorXd residuals;
    VectorXd featureTotals;

    bool fileopen = true;
    int32_t processed = 0;
    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    std::vector<Document> minibatch;
    std::vector<std::string> idens;
    while (fileopen && processed < maxUnits) {
        const int32_t remaining = maxUnits - processed;
        fileopen = lda.readMinibatch(inFileStream, minibatch, idens, batchSize, 0, remaining);
        if (minibatch.empty()) break;

        RowMajorMatrixXd doc_topic = lda.do_transform(minibatch);
        if (featureResiduals && residuals.size() == 0) {
            residuals = VectorXd::Zero(M);
            featureTotals = VectorXd::Zero(M);
        }
        for (size_t i = 0; i < minibatch.size(); ++i) {
            if (!idens[i].empty()) {
                outFileStream << idens[i] << "\t";
            }
            outFileStream << doc_topic(i, 0);
            for (int32_t k = 1; k < K; ++k) {
                outFileStream << "\t" << doc_topic(i, k);
            }
            outFileStream << "\n";
        }

        struct LocalAgg {
            RowMajorMatrixXd pseudobulk;
            VectorXd residuals;
            VectorXd featureTotals;
            bool with_residuals;
            LocalAgg(int32_t M, int32_t K, bool with_residuals_)
                : pseudobulk(RowMajorMatrixXd::Zero(M, K)),
                  with_residuals(with_residuals_) {
                if (with_residuals) {
                    residuals = VectorXd::Zero(M);
                    featureTotals = VectorXd::Zero(M);
                }
            }
        };

        tbb::enumerable_thread_specific<LocalAgg> tls([&] {
            return LocalAgg(M, K, featureResiduals);
        });

        size_t grainsize = std::max(1, int(minibatch.size() / (2 * nThreads)));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, minibatch.size(), grainsize), [&](const tbb::blocked_range<size_t>& range)
        {
            auto& local = tls.local();
            for (size_t idx = range.begin(); idx < range.end(); ++idx) {
                const int32_t i = static_cast<int32_t>(idx);

                Document& doc = minibatch[i];
                const double doc_total = doc.get_sum();
                if (doc_total < minCount) {continue;}

                for (size_t j = 0; j < doc.ids.size(); ++j) {
                    const uint32_t m = doc.ids[j];
                    const double cnt = doc.cnts[j];
                    for (int32_t k = 0; k < K; ++k) {
                        local.pseudobulk(m, k) += cnt * doc_topic(i, k);
                    }
                    if (featureResiduals) {
                        local.featureTotals(m) += cnt;
                    }
                }
                if (!featureResiduals) {continue;}
                RowVectorXd expected = doc_topic.row(i) * beta_norm;
                expected *= doc_total;
                local.residuals += expected.transpose();
                for (size_t j = 0; j < doc.ids.size(); ++j) {
                    const uint32_t m = doc.ids[j];
                    const double e = expected(m);
                    local.residuals(m) += std::abs(e - doc.cnts[j]) - e;
                }
            }
        });

        for (auto& local : tls) {
            pseudobulk += local.pseudobulk;
            if (featureResiduals) {
                residuals += local.residuals;
                featureTotals += local.featureTotals;
            }
        }
        processed += static_cast<int32_t>(minibatch.size());
    }
    inFileStream.close();
    outFileStream.close();
    notice("Transformation results written to %s", outFile.c_str());

    outFile = outPrefix + ".pseudobulk.tsv";
    outFileStream.open(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());
    outFileStream << "Feature\t";
    lda.writeModelHeader(outFileStream);
    outFileStream << std::fixed << std::setprecision(3);
    const std::vector<std::string> featureNames = lda.getFeatureNames();
    for (int32_t i = 0; i < M; ++i) {
        outFileStream << featureNames[i];
        for (int32_t k = 0; k < pseudobulk.cols(); ++k) {
            outFileStream << "\t" << pseudobulk(i, k);
        }
        outFileStream << "\n";
    }
    outFileStream.close();
    notice("Pseudobulk counts written to %s", outFile.c_str());

    if (!featureResiduals) return 0;
    outFile = outPrefix + ".feature_residuals.tsv";
    outFileStream.open(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());
    outFileStream << "Feature\tAbsDiff\tAbsDiffPerCount\n";
    for (int32_t i = 0; i < M; ++i) {
        const double total = featureTotals[i];
        const double diff = residuals(i);
        const double ratio = total > 0.0 ? diff / total : 0.0;
        outFileStream << featureNames[i]
            << "\t" << std::fixed << std::setprecision(3) << diff
            << "\t" << std::fixed << std::setprecision(6) << ratio << "\n";
    }
    outFileStream.close();
    notice("Per-feature residuals written to %s", outFile.c_str());

    return 0;
}
