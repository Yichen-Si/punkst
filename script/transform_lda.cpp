#include "topic_svb.hpp"

#include <fstream>
#include <iomanip>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <numeric>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

namespace {

struct TransformBatch {
    std::vector<Document> docs;
    std::vector<std::string> ids;

    void clear() {
        docs.clear();
        ids.clear();
    }

    bool empty() const {
        return docs.empty();
    }

    size_t size() const {
        return docs.size();
    }
};

void writeUnitIdHeader(std::ostream& out, bool use_10x, const std::string& info_header) {
    if (use_10x) {
        out << "#barcode\t";
        return;
    }
    out << "#";
    if (!info_header.empty()) {
        out << info_header << "\t";
    }
}

void writeResultHeader(std::ostream& out, LDA4Hex& lda, int32_t topkOnly) {
    if (topkOnly < 0) {
        lda.writeUnitHeader(out);
        return;
    }
    const int32_t topk = std::min(topkOnly, lda.getNumTopics());
    if (topk <= 0) {
        out << "\n";
        return;
    }
    out << "K1";
    for (int32_t i = 1; i < topk; ++i) {
        out << "\tK" << (i + 1);
    }
    for (int32_t i = 0; i < topk; ++i) {
        out << "\tP" << (i + 1);
    }
    out << "\n";
}

int64_t rawTotalCount(const Document& doc) {
    const double raw_total = doc.raw_ct_tot >= 0.0
        ? doc.raw_ct_tot
        : std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
    return static_cast<int64_t>(std::llround(raw_total));
}

void assignBarcodeIds(const std::vector<int32_t>& barcode_idx, std::vector<std::string>& ids) {
    ids.clear();
    ids.reserve(barcode_idx.size());
    for (auto idx : barcode_idx) {
        ids.push_back(std::to_string(idx));
    }
}

void applyWeights(std::vector<Document>& docs, LDA4Hex& lda) {
    for (auto& doc : docs) {
        lda.applyWeights(doc);
    }
}

struct ResidualState {
    RowMajorMatrixXd betaNorm;
    MatrixXd topicSimilarity;
    VectorXd featureResiduals;
    VectorXd featureTotals;

    ResidualState() = default;

    explicit ResidualState(const RowMajorMatrixXd& model)
        : betaNorm(rowNormalize(model)),
          topicSimilarity(pairwiseCosineSimilarityRows(betaNorm)),
          featureResiduals(VectorXd::Zero(model.cols())),
          featureTotals(VectorXd::Zero(model.cols())) {}
};

struct TransformOutputs {
    std::string resultsPath;
    std::string unitStatsPath;
    std::ofstream results;
    std::ofstream unitStats;

    TransformOutputs(const std::string& outPrefix, bool writeUnitStats)
        : resultsPath(outPrefix + ".results.tsv"),
          unitStatsPath(outPrefix + ".unit_stats.tsv"),
          results(resultsPath) {
        if (!results) {
            error("Error opening output file: %s for writing", resultsPath.c_str());
        }
        if (writeUnitStats) {
            unitStats.open(unitStatsPath);
            if (!unitStats) {
                error("Error opening output file: %s for writing", unitStatsPath.c_str());
            }
        }
    }
};

class TransformBatchProcessor {
public:
    TransformBatchProcessor(LDA4Hex& lda_, std::ofstream& resultsStream_,
            std::ofstream* unitMetaStream_, RowMajorMatrixXd& pseudobulk_,
            ResidualState* residualState_, int32_t topkOnly_,
            int32_t nThreads_)
        : lda(lda_),
          resultsStream(resultsStream_),
          unitMetaStream(unitMetaStream_),
          pseudobulk(pseudobulk_),
          residualState(residualState_),
          topkOnly(topkOnly_),
          threadHint(std::max<int32_t>(1, nThreads_)),
          M(lda_.nFeatures()),
          K(lda_.getNumTopics()) {}

    void process(TransformBatch& batch) {
        if (batch.empty()) {
            return;
        }

        const size_t N = batch.size();
        RowMajorMatrixXd doc_topic = lda.do_transform(batch.docs);
        writeTopicRows(batch.ids, doc_topic);

        struct LocalAgg {
            RowMajorMatrixXd pseudobulk;
            bool withResiduals;
            VectorXd featureResiduals;
            VectorXd featureTotals;

            LocalAgg(int32_t M, int32_t K, bool withResiduals_)
                : pseudobulk(RowMajorMatrixXd::Zero(M, K)),
                  withResiduals(withResiduals_) {
                if (withResiduals) {
                    featureResiduals = VectorXd::Zero(M);
                    featureTotals = VectorXd::Zero(M);
                }
            }
        };

        tbb::enumerable_thread_specific<LocalAgg> tls([&] {
            return LocalAgg(M, K, residualState != nullptr);
        });

        VectorXd unitResiduals;
        VectorXd unitCosineSim;
        VectorXd unitEntropy;
        VectorXd unitSensitiveEntropyLCR;
        VectorXd unitSensitiveEntropyQ;
        if (residualState != nullptr) {
            unitResiduals = VectorXd::Zero(N);
            unitCosineSim = VectorXd::Zero(N);
            const ThetaEntropyStats thetaStats =
                computeThetaEntropyStats(doc_topic, residualState->topicSimilarity);
            unitEntropy = thetaStats.entropy;
            unitSensitiveEntropyLCR = thetaStats.sh_lcr;
            unitSensitiveEntropyQ = thetaStats.sh_q;
        }

        const size_t grainsize = std::max<size_t>(1, N / (2 * static_cast<size_t>(threadHint)));
        tbb::parallel_for(tbb::blocked_range<size_t>(0, N, grainsize),
            [&](const tbb::blocked_range<size_t>& range) {
                auto& local = tls.local();
                RowVectorXd expected = RowVectorXd::Zero(M);

                for (size_t idx = range.begin(); idx < range.end(); ++idx) {
                    const int32_t i = static_cast<int32_t>(idx);
                    Document& doc = batch.docs[idx];
                    const double weighted_total = doc.get_sum();

                    for (size_t j = 0; j < doc.ids.size(); ++j) {
                        const uint32_t m = doc.ids[j];
                        const double cnt = doc.cnts[j];
                        for (int32_t k = 0; k < K; ++k) {
                            local.pseudobulk(m, k) += cnt * doc_topic(i, k);
                        }
                        if (residualState != nullptr) {
                            local.featureTotals(m) += cnt;
                        }
                    }
                    if (residualState == nullptr) {
                        continue;
                    }

                    expected = doc_topic.row(i) * residualState->betaNorm;
                    expected *= weighted_total;
                    local.featureResiduals += expected.transpose();

                    double cosine_sim = 0.0;
                    double observed_norm_sq = 0.0;
                    double expected_norm_sq = expected.squaredNorm();
                    double doc_residual = expected.sum();
                    for (size_t j = 0; j < doc.ids.size(); ++j) {
                        const uint32_t m = doc.ids[j];
                        const double observed = doc.cnts[j];
                        const double estimate = expected(m);
                        const double residual = std::abs(estimate - observed) - estimate;
                        local.featureResiduals(m) += residual;
                        doc_residual += residual;
                        cosine_sim += estimate * observed;
                        observed_norm_sq += observed * observed;
                    }
                    if (expected_norm_sq > 0.0 && observed_norm_sq > 0.0) {
                        cosine_sim /= std::sqrt(expected_norm_sq * observed_norm_sq);
                    } else {
                        cosine_sim = 0.0;
                    }
                    unitResiduals(idx) = doc_residual;
                    unitCosineSim(idx) = cosine_sim;
                }
            });

        for (auto& local : tls) {
            pseudobulk += local.pseudobulk;
            if (residualState != nullptr) {
                residualState->featureResiduals += local.featureResiduals;
                residualState->featureTotals += local.featureTotals;
            }
        }

        if (unitMetaStream != nullptr) {
            writeUnitMetaRows(batch, unitResiduals, unitCosineSim,
                unitEntropy, unitSensitiveEntropyLCR, unitSensitiveEntropyQ);
        }
    }

private:
    void writeTopicRows(const std::vector<std::string>& ids, const RowMajorMatrixXd& doc_topic) {
        if (topkOnly > 0) {
            writeTopKRows(ids, doc_topic);
            return;
        }
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!ids[i].empty()) {
                resultsStream << ids[i] << "\t";
            }
            resultsStream << doc_topic(i, 0);
            for (int32_t k = 1; k < K; ++k) {
                resultsStream << "\t" << doc_topic(i, k);
            }
            resultsStream << "\n";
        }
    }

    void writeTopKRows(const std::vector<std::string>& ids, const RowMajorMatrixXd& doc_topic) {
        const int32_t topk = std::min(topkOnly, K);
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!ids[i].empty()) {
                resultsStream << ids[i] << "\t";
            }
            std::vector<std::pair<double, int32_t>> ranked;
            ranked.reserve(K);
            for (int32_t k = 0; k < K; ++k) {
                ranked.emplace_back(doc_topic(i, k), k);
            }
            std::partial_sort(ranked.begin(), ranked.begin() + topk, ranked.end(),
                [](const auto& a, const auto& b) {
                    if (a.first != b.first) {
                        return a.first > b.first;
                    }
                    return a.second < b.second;
                });

            resultsStream << ranked[0].second;
            for (int32_t j = 1; j < topk; ++j) {
                resultsStream << "\t" << ranked[j].second;
            }
            for (int32_t j = 0; j < topk; ++j) {
                resultsStream << "\t" << ranked[j].first;
            }
            resultsStream << "\n";
        }
    }

    void writeUnitMetaRows(const TransformBatch& batch, const VectorXd& unitResiduals,
            const VectorXd& unitCosineSim, const VectorXd& unitEntropy,
            const VectorXd& unitSensitiveEntropyLCR, const VectorXd& unitSensitiveEntropyQ) {
        for (size_t i = 0; i < batch.size(); ++i) {
            if (!batch.ids[i].empty()) {
                *unitMetaStream << batch.ids[i] << "\t";
            }
            *unitMetaStream << rawTotalCount(batch.docs[i])
                << "\t" << std::setprecision(2) << unitResiduals(i)
                << "\t" << std::setprecision(4) << unitCosineSim(i)
                << "\t" << std::setprecision(4) << unitEntropy(i)
                << "\t" << std::setprecision(4) << unitSensitiveEntropyLCR(i)
                << "\t" << std::setprecision(4) << unitSensitiveEntropyQ(i) << "\n";
        }
    }

    LDA4Hex& lda;
    std::ofstream& resultsStream;
    std::ofstream* unitMetaStream;
    RowMajorMatrixXd& pseudobulk;
    ResidualState* residualState;
    int32_t topkOnly;
    int32_t threadHint;
    int32_t M;
    int32_t K;
};

} // namespace

int32_t cmdLDATransform(int argc, char** argv) {
    std::string inFile, metaFile, modelFile, outPrefix, featureFile, weightFile;
    std::string dge_dir, in_bc, in_ft, in_mtx;
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
    int32_t topk_only = -1;
    bool computeResiduals = false;
    bool sorted_by_barcode = false;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("in-model", "Input model matrix (topic-word) file", modelFile, true)
      .add_option("out-prefix", "Output prefix for results files", outPrefix, true)
      .add_option("minibatch-size", "Minibatch size", batchSize)
      .add_option("modal", "Modality to use (0-based)", modal)
      .add_option("threads", "Number of threads", nThreads)
      .add_option("seed", "Random seed", seed)
      .add_option("verbose", "Verbose level", verbose)
      .add_option("debug", "If >0, only process this many units", debug_);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dir)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("sorted-by-barcode", "Input matrix is sorted by barcode, use streaming mode", sorted_by_barcode);

    pl.add_option("features", "Feature names and total counts file", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", minCountFeature)
      .add_option("min-count", "Minimum total feature count for a unit to be kept", minCount)
      .add_option("feature-weights", "Input weights file", weightFile)
      .add_option("default-weight", "Default weight for features not in weight file", defaultWeight)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex);

    pl.add_option("max-iter", "Max iterations per document", maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per document", mDelta)
      .add_option("feature-residuals", "Compute per-feature and per-unit residuals (backward compatibility)", computeResiduals)
      .add_option("residuals", "Compute per-feature and per-unit residuals", computeResiduals)
      .add_option("topk-only", "Write only top-k factor indices/probabilities to results.tsv", topk_only);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (batchSize <= 0) {
        batchSize = 512;
        warning("Minibatch size must be greater than 0, using default value of %d", batchSize);
    }
    if (topk_only == 0) {
        error("--topk-only must be a positive integer");
    }
    if (seed <= 0) {
        seed = std::random_device{}();
    }

    bool use_10x = !dge_dir.empty() || !in_bc.empty() || !in_ft.empty() || !in_mtx.empty();
    if (use_10x && !inFile.empty()) {
        warning("Both --in-data and 10X inputs are provided; using 10X inputs and ignoring --in-data");
    }
    if (!use_10x && inFile.empty()) {
        error("Either --in-data or 10X inputs must be provided");
    }
    if (use_10x && !dge_dir.empty() && (in_bc.empty() || in_ft.empty() || in_mtx.empty())) {
        if (dge_dir.back() == '/') {
            dge_dir.pop_back();
        }
        in_bc = dge_dir + "/barcodes.tsv.gz";
        in_ft = dge_dir + "/features.tsv.gz";
        in_mtx = dge_dir + "/matrix.mtx.gz";
    }
    if (use_10x && (in_bc.empty() || in_ft.empty() || in_mtx.empty())) {
        error("Missing required 10X inputs (--in-barcodes, --in-features, --in-matrix)");
    }
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    if (use_10x) {
        dge_ptr = std::make_unique<DGEReader10X>(in_bc, in_ft, in_mtx);
        reader.initFromFeatures(dge_ptr->features, dge_ptr->nBarcodes);
    } else {
        if (metaFile.empty()) {
            error("Missing required --in-meta for non-10X input");
        }
        reader.readMetadata(metaFile);
    }
    if (!featureFile.empty()) {
        reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
    }
    if (!weightFile.empty()) {
        reader.setWeights(weightFile, defaultWeight);
    }

    std::string info_header;
    if (!use_10x) {
        reader.getInfoHeaderStr(info_header);
    }

    LDA4Hex lda(reader, modal, verbose);
    lda.initialize_transform(modelFile,
        seed, nThreads, verbose, maxIter, mDelta);

    const int32_t M = lda.nFeatures();
    const int32_t K = lda.getNumTopics();
    if (topk_only > 0 && topk_only > K-1) {
        warning("--topk-only is >= the number of topics (%d); writing all topics", K);
        topk_only = -1;
    }

    RowMajorMatrixXd pseudobulk = RowMajorMatrixXd::Zero(M, K);
    std::unique_ptr<ResidualState> residualState;
    if (computeResiduals) {
        residualState = std::make_unique<ResidualState>(lda.get_model_matrix());
    }

    TransformOutputs outputs(outPrefix, computeResiduals);
    writeUnitIdHeader(outputs.results, use_10x, info_header);
    writeResultHeader(outputs.results, lda, topk_only);
    outputs.results << std::fixed << std::setprecision(4);
    if (computeResiduals) {
        writeUnitIdHeader(outputs.unitStats, use_10x, info_header);
        outputs.unitStats << "total_count\tresidual\tcosine_sim\tentropy\tsh_lcr\tsh_q\n";
        outputs.unitStats << std::fixed;
    }

    TransformBatchProcessor processor(lda, outputs.results,
        computeResiduals ? &outputs.unitStats : nullptr,
        pseudobulk, residualState.get(), topk_only, nThreads);

    bool fileopen = true;
    int32_t processed = 0;
    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    TransformBatch batch;
    const int32_t minCountInt = minCount > 0 ? static_cast<int32_t>(std::ceil(minCount)) : 0;

    if (use_10x) {
        DGEReader10X& dge = *dge_ptr;
        const std::vector<std::string> model_features = lda.getFeatureNames();
        int32_t n_overlap = dge.setFeatureIndexRemap(model_features, false);
        if (n_overlap == 0) {
            error("No overlapping features found between 10X input and model metadata");
        }

        std::vector<int32_t> barcode_idx;

        if (sorted_by_barcode) {
            while (fileopen && processed < maxUnits) {
                batch.clear();
                const int32_t remaining = maxUnits - processed;
                fileopen = dge.readMinibatch(batch.docs, barcode_idx, batchSize, remaining, minCountInt);
                if (batch.empty()) {
                    break;
                }
                applyWeights(batch.docs, lda);
                assignBarcodeIds(barcode_idx, batch.ids);
                processor.process(batch);
                processed += static_cast<int32_t>(batch.size());
            }
        } else {
            std::vector<Document> all_docs;
            std::vector<int32_t> all_barcode_idx;
            dge.readAll(all_docs, all_barcode_idx, minCountInt);
            applyWeights(all_docs, lda);
            std::vector<std::string> all_ids;
            assignBarcodeIds(all_barcode_idx, all_ids);
            size_t cursor = 0;
            while (cursor < all_docs.size() && processed < maxUnits) {
                batch.clear();
                const int32_t remaining = maxUnits - processed;
                size_t take = std::min(static_cast<size_t>(batchSize), all_docs.size() - cursor);
                if (take > static_cast<size_t>(remaining)) {
                    take = static_cast<size_t>(remaining);
                }
                batch.docs.insert(batch.docs.end(),
                    std::make_move_iterator(all_docs.begin() + cursor),
                    std::make_move_iterator(all_docs.begin() + cursor + take));
                batch.ids.insert(batch.ids.end(),
                    std::make_move_iterator(all_ids.begin() + cursor),
                    std::make_move_iterator(all_ids.begin() + cursor + take));
                cursor += take;
                if (batch.empty()) {
                    break;
                }
                processor.process(batch);
                processed += static_cast<int32_t>(batch.size());
            }
        }
    } else {
        std::ifstream inFileStream(inFile);
        if (!inFileStream) error("Error opening input file: %s", inFile.c_str());
        while (fileopen && processed < maxUnits) {
            batch.clear();
            const int32_t remaining = maxUnits - processed;
            fileopen = lda.readMinibatch(inFileStream, batch.docs, batch.ids, batchSize, minCountInt, remaining);
            if (batch.empty()) break;
            processor.process(batch);
            processed += static_cast<int32_t>(batch.size());
        }
        inFileStream.close();
    }
    outputs.results.close();
    notice("Transformation results written to %s", outputs.resultsPath.c_str());
    if (computeResiduals) {
        outputs.unitStats.close();
        notice("Per-unit residuals written to %s", outputs.unitStatsPath.c_str());
    }

    std::string outFile = outPrefix + ".pseudobulk.tsv";
    std::ofstream outFileStream(outFile);
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

    if (!computeResiduals) return 0;
    outFile = outPrefix + ".feature_residuals.tsv";
    outFileStream.open(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());
    outFileStream << "Feature\tAbsDiff\tAbsDiffPerCount\n";
    for (int32_t i = 0; i < M; ++i) {
        const double total = residualState->featureTotals[i];
        const double diff = residualState->featureResiduals(i);
        const double ratio = total > 0.0 ? diff / total : 0.0;
        outFileStream << featureNames[i]
            << "\t" << std::fixed << std::setprecision(3) << diff
            << "\t" << std::fixed << std::setprecision(6) << ratio << "\n";
    }
    outFileStream.close();
    notice("Per-feature residuals written to %s", outFile.c_str());

    return 0;
}
