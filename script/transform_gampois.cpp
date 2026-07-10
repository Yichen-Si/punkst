#include "gamma_pois_topic.hpp"

#include <algorithm>
#include <climits>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <random>

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

void assignBarcodeIds(const DGEReader10X& dge,
    const std::vector<int32_t>& barcode_idx, std::vector<std::string>& ids) {
    ids.clear();
    ids.reserve(barcode_idx.size());
    for (auto idx : barcode_idx) {
        if (idx >= 0 && idx < static_cast<int32_t>(dge.barcodes.size())) {
            ids.push_back(dge.barcodes[idx]);
        } else {
            ids.push_back(std::to_string(idx));
        }
    }
}

void applyWeights(std::vector<Document>& docs, GammaPoisson4Hex& gp) {
    for (auto& doc : docs) {
        gp.applyWeights(doc);
    }
}

class GammaPoisTransformBatchProcessor {
public:
    GammaPoisTransformBatchProcessor(GammaPoisson4Hex& gp_,
        std::ostream& results_, RowMajorMatrixXd& pseudobulk_)
        : gp(gp_), results(results_), pseudobulk(pseudobulk_),
          M(gp_.nFeatures()), K(gp_.getNumTopics()) {}

    void process(TransformBatch& batch) {
        if (batch.empty()) {
            return;
        }
        RowMajorMatrixXd doc_topic = gp.do_transform(DocumentView(batch.docs));
        writeTopicRows(batch.ids, doc_topic);
        for (size_t i = 0; i < batch.docs.size(); ++i) {
            const Document& doc = batch.docs[i];
            for (size_t j = 0; j < doc.ids.size(); ++j) {
                const uint32_t m = doc.ids[j];
                const double raw_count = gp.rawCountFor(m, doc.cnts[j], doc.counts_weighted);
                for (int32_t k = 0; k < K; ++k) {
                    pseudobulk(m, k) += raw_count * doc_topic(i, k);
                }
            }
        }
    }

private:
    void writeTopicRows(const std::vector<std::string>& ids,
        const RowMajorMatrixXd& doc_topic) {
        for (size_t i = 0; i < ids.size(); ++i) {
            if (!ids[i].empty()) {
                results << ids[i] << "\t";
            }
            results << doc_topic(i, 0);
            for (int32_t k = 1; k < K; ++k) {
                results << "\t" << doc_topic(i, k);
            }
            results << "\n";
        }
    }

    GammaPoisson4Hex& gp;
    std::ostream& results;
    RowMajorMatrixXd& pseudobulk;
    int32_t M;
    int32_t K;
};

} // namespace

int32_t cmdGammaPoisTransform(int argc, char** argv) {
    std::string inFile, metaFile, stateFile, outPrefix, featureFile;
    std::vector<std::string> dge_dirs, in_bc, in_ft, in_mtx, dataset_ids;
    std::string include_ftr_regex, exclude_ftr_regex;
    int32_t seed = -1;
    int32_t batchSize = 1024;
    int32_t nThreads = 1;
    int32_t modal = 0;
    int32_t minCountFeature = 1;
    int32_t icolWeight = -1;
    double minCount = 20;
    int32_t debug_ = 0;
    int32_t verbose = 0;
    double defaultWeight = -1.0;
    int32_t maxIter = 100;
    double mDelta = 1e-3;
    bool sorted_by_barcode = false;
    bool keep_barcodes = false;

    ParamList pl;
    pl.add_option("in-data", "Input hex file", inFile)
      .add_option("in-meta", "Metadata file", metaFile)
      .add_option("in-state", "Input Gamma-Poisson state file", stateFile, true)
      .add_option("out-prefix", "Output prefix for results files", outPrefix, true)
      .add_option("minibatch-size", "Minibatch size", batchSize)
      .add_option("modal", "Modality to use (0-based)", modal)
      .add_option("threads", "Number of threads", nThreads)
      .add_option("seed", "Random seed", seed)
      .add_option("verbose", "Verbose level", verbose)
      .add_option("debug", "If >0, only process this many units", debug_);

    pl.add_option("in-dge-dir", "Input directory for 10X DGE files", dge_dirs)
      .add_option("in-barcodes", "Input barcodes.tsv.gz", in_bc)
      .add_option("in-features", "Input features.tsv.gz", in_ft)
      .add_option("in-matrix", "Input matrix.mtx.gz", in_mtx)
      .add_option("dataset-id", "Dataset IDs for joint 10X input", dataset_ids)
      .add_option("sorted-by-barcode", "Input matrix is sorted by barcode, use streaming mode", sorted_by_barcode)
      .add_option("keep-barcodes", "For 10X input, write IDs from barcodes.tsv.gz instead of 0-based barcode indices", keep_barcodes);

    pl.add_option("features", "Feature names and total counts file", featureFile)
      .add_option("min-count-per-feature", "Min count for features to be included (requires --features)", minCountFeature)
      .add_option("min-count", "Minimum total feature count for a unit to be kept", minCount)
      .add_option("default-weight", "Default weight for model features missing from --features when feature weights are active; <0 drops missing features", defaultWeight)
      .add_option("icol-weight", "0-based column index for feature weight in --features; <0 disables feature weights", icolWeight)
      .add_option("include-feature-regex", "Regex for including features", include_ftr_regex)
      .add_option("exclude-feature-regex", "Regex for excluding features", exclude_ftr_regex);

    pl.add_option("max-iter", "Max iterations per document", maxIter)
      .add_option("mean-change-tol", "Convergence tolerance per document", mDelta);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception& ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help_noexit();
        return 1;
    }

    if (batchSize <= 0) batchSize = 512;
    if (seed <= 0) seed = std::random_device{}();
    const bool weights_active = !featureFile.empty() && icolWeight >= 0;
    if (defaultWeight < 0.0) defaultWeight = -1.0;

    const std::vector<std::string> modelFeatures =
        GammaPoissonTopicModel::read_state_feature_names(stateFile);
    HexReader reader;
    std::unique_ptr<DGEReader10X> dge_ptr;
    const bool use_10x = initHexOrDgeInput(reader, dge_ptr, inFile, metaFile,
        dge_dirs, in_bc, in_ft, in_mtx, dataset_ids, keep_barcodes);
    if (!featureFile.empty()) {
        if (weights_active) {
            reader.setFeatureFilterAndWeights(featureFile, minCountFeature,
                include_ftr_regex, exclude_ftr_regex, icolWeight, defaultWeight,
                defaultWeight >= 0.0);
        } else {
            reader.setFeatureFilter(featureFile, minCountFeature, include_ftr_regex, exclude_ftr_regex);
        }
    }
    std::vector<std::string> modelFeaturesMutable = modelFeatures;
    reader.setFeatureIndexRemap(modelFeaturesMutable, false);

    std::string info_header;
    if (!use_10x) {
        reader.getInfoHeaderStr(info_header);
    }

    GammaPoisson4Hex gp(reader, modal, verbose);
    gp.initialize_transform(stateFile, seed, nThreads, verbose, maxIter, mDelta);

    const int32_t M = gp.nFeatures();
    const int32_t K = gp.getNumTopics();
    RowMajorMatrixXd pseudobulk = RowMajorMatrixXd::Zero(M, K);

    const std::string resultsPath = outPrefix + ".results.tsv";
    std::ofstream results(resultsPath);
    if (!results) {
        error("Error opening output file: %s for writing", resultsPath.c_str());
    }
    writeUnitIdHeader(results, use_10x, info_header);
    gp.writeUnitHeader(results);
    results << std::fixed << std::setprecision(4);

    GammaPoisTransformBatchProcessor processor(gp, results, pseudobulk);
    bool fileopen = true;
    int32_t processed = 0;
    const int32_t maxUnits = debug_ > 0 ? debug_ : INT32_MAX;
    const int32_t minCountInt = minCount > 0 ? static_cast<int32_t>(std::ceil(minCount)) : 0;
    TransformBatch batch;

    if (use_10x) {
        DGEReader10X& dge = *dge_ptr;
        int32_t n_overlap = dge.setFeatureIndexRemap(modelFeatures, false);
        if (n_overlap == 0) {
            error("No overlapping features found between 10X input and model state");
        }
        std::vector<int32_t> barcode_idx;
        if (sorted_by_barcode) {
            while (fileopen && processed < maxUnits) {
                batch.clear();
                const int32_t remaining = maxUnits - processed;
                fileopen = dge.readMinibatch(batch.docs, barcode_idx,
                    batchSize, remaining, minCountInt);
                if (batch.empty()) break;
                applyWeights(batch.docs, gp);
                assignBarcodeIds(dge, barcode_idx, batch.ids);
                processor.process(batch);
                processed += static_cast<int32_t>(batch.size());
            }
        } else {
            std::vector<Document> all_docs;
            std::vector<int32_t> all_barcode_idx;
            dge.readAll(all_docs, all_barcode_idx, minCountInt);
            applyWeights(all_docs, gp);
            std::vector<std::string> all_ids;
            assignBarcodeIds(dge, all_barcode_idx, all_ids);
            size_t cursor = 0;
            while (cursor < all_docs.size() && processed < maxUnits) {
                batch.clear();
                const int32_t remaining = maxUnits - processed;
                size_t take = std::min(static_cast<size_t>(batchSize),
                    all_docs.size() - cursor);
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
                if (batch.empty()) break;
                processor.process(batch);
                processed += static_cast<int32_t>(batch.size());
            }
        }
    } else {
        std::ifstream inFileStream(inFile);
        if (!inFileStream) {
            error("Error opening input file: %s", inFile.c_str());
        }
        while (fileopen && processed < maxUnits) {
            batch.clear();
            const int32_t remaining = maxUnits - processed;
            fileopen = gp.readMinibatch(inFileStream, batch.docs, batch.ids,
                batchSize, minCountInt, remaining);
            if (batch.empty()) break;
            processor.process(batch);
            processed += static_cast<int32_t>(batch.size());
        }
    }
    results.close();
    notice("Transformation results written to %s", resultsPath.c_str());

    const std::string pseudobulkPath = outPrefix + ".pseudobulk.tsv";
    std::ofstream pseudobulkOut(pseudobulkPath);
    if (!pseudobulkOut) {
        error("Error opening output file: %s for writing", pseudobulkPath.c_str());
    }
    pseudobulkOut << "Feature\t";
    gp.writeModelHeader(pseudobulkOut);
    pseudobulkOut << std::fixed << std::setprecision(3);
    const std::vector<std::string> featureNames = gp.getFeatureNames();
    for (int32_t w = 0; w < M; ++w) {
        pseudobulkOut << featureNames[w];
        for (int32_t k = 0; k < K; ++k) {
            pseudobulkOut << "\t" << pseudobulk(w, k);
        }
        pseudobulkOut << "\n";
    }
    pseudobulkOut.close();
    notice("Pseudobulk counts written to %s", pseudobulkPath.c_str());
    return 0;
}
