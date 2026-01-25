#include "topic_svb.hpp"

#include <fstream>
#include <iomanip>
#include <algorithm>
#include <iterator>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/parallel_for.h>

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
    bool featureResiduals = false;
    bool sorted_by_barcode = false;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile, true)
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

    HexReader reader(metaFile);
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

    RowMajorMatrixXd beta_norm;
    if (featureResiduals) {
        const RowMajorMatrixXd& model = lda.get_model_matrix();
        beta_norm = rowNormalize(model);
    }

    std::string outFile = outPrefix + ".results.tsv";
    std::ofstream outFileStream(outFile);
    if (!outFileStream) error("Error opening output file: %s for writing", outFile.c_str());

    if (use_10x) {
        outFileStream << "#barcode\t";
    } else {
        outFileStream << "#" << info_header << "\t";
    }
    lda.writeUnitHeader(outFileStream);
    outFileStream << std::fixed << std::setprecision(4);

    RowMajorMatrixXd pseudobulk = RowMajorMatrixXd::Zero(M, K);
    VectorXd residuals;
    VectorXd featureTotals;

    auto process_batch = [&](std::vector<Document>& minibatch, std::vector<std::string>& idens) {
        if (minibatch.empty()) {
            return;
        }
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
    };

    bool fileopen = true;
    int32_t processed = 0;
    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    std::vector<Document> minibatch;
    std::vector<std::string> idens;

    if (use_10x) {
        DGEReader10X dge(in_bc, in_ft, in_mtx);
        const std::vector<std::string> model_features = lda.getFeatureNames();
        int32_t n_overlap = dge.setFeatureIndexRemap(model_features, false);
        if (n_overlap == 0) {
            error("No overlapping features found between 10X input and model metadata");
        }

        auto apply_weight = [&](Document& doc) {
            lda.applyWeights(doc);
        };

        if (sorted_by_barcode) {
            while (fileopen && processed < maxUnits) {
                minibatch.clear();
                idens.clear();
                const int32_t remaining = maxUnits - processed;
                while ((int32_t)minibatch.size() < batchSize && (int32_t)minibatch.size() < remaining) {
                    Document doc;
                    std::string barcode;
                    int32_t barcode_idx = -1;
                    if (!dge.next(doc, &barcode_idx, &barcode)) {
                        fileopen = false;
                        break;
                    }
                    if (barcode.empty() && barcode_idx >= 0) {
                        barcode = std::to_string(barcode_idx);
                    }
                    apply_weight(doc);
                    minibatch.push_back(std::move(doc));
                    idens.push_back(std::move(barcode));
                }
                if (minibatch.empty()) {
                    break;
                }
                process_batch(minibatch, idens);
                processed += static_cast<int32_t>(minibatch.size());
            }
        } else {
            std::vector<Document> all_docs;
            std::vector<std::string> all_barcodes;
            dge.readAll(all_docs, all_barcodes, minCount);
            for (size_t i = 0; i < all_barcodes.size(); ++i) {
                if (all_barcodes[i].empty()) {
                    all_barcodes[i] = std::to_string(i);
                }
            }
            for (auto& doc : all_docs) {
                apply_weight(doc);
            }
            size_t cursor = 0;
            while (cursor < all_docs.size() && processed < maxUnits) {
                minibatch.clear();
                idens.clear();
                const int32_t remaining = maxUnits - processed;
                size_t take = std::min(static_cast<size_t>(batchSize), all_docs.size() - cursor);
                if (take > static_cast<size_t>(remaining)) {
                    take = static_cast<size_t>(remaining);
                }
                minibatch.insert(minibatch.end(),
                    std::make_move_iterator(all_docs.begin() + cursor),
                    std::make_move_iterator(all_docs.begin() + cursor + take));
                idens.insert(idens.end(),
                    std::make_move_iterator(all_barcodes.begin() + cursor),
                    std::make_move_iterator(all_barcodes.begin() + cursor + take));
                cursor += take;
                if (minibatch.empty()) {
                    break;
                }
                process_batch(minibatch, idens);
                processed += static_cast<int32_t>(minibatch.size());
            }
        }
    } else {
        std::ifstream inFileStream(inFile);
        if (!inFileStream) error("Error opening input file: %s", inFile.c_str());
        while (fileopen && processed < maxUnits) {
            const int32_t remaining = maxUnits - processed;
            fileopen = lda.readMinibatch(inFileStream, minibatch, idens, batchSize, 0, remaining);
            if (minibatch.empty()) break;
            process_batch(minibatch, idens);
            processed += static_cast<int32_t>(minibatch.size());
        }
        inFileStream.close();
    }
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
