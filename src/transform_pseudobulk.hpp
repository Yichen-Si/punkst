#pragma once

#include "dataunits.hpp"
#include "error.hpp"
#include "numerical_utils.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/parallel_for.h>

namespace transform_pseudobulk {

enum class Mode {
    Standard,
    WeightedModel,
    AllFeatures
};

struct SpecialBatch {
    std::vector<Document> modelDocs;
    std::vector<std::string> ids;
    std::vector<Document> rawDocs;
    std::vector<std::vector<double>> rawModelCounts;

    void clear() {
        modelDocs.clear();
        ids.clear();
        rawDocs.clear();
        rawModelCounts.clear();
    }

    bool empty() const {
        return modelDocs.empty();
    }

    size_t size() const {
        return modelDocs.size();
    }
};

inline Mode selectMode(bool allFeatures, bool weightsActive) {
    if (allFeatures) {
        return Mode::AllFeatures;
    }
    return weightsActive ? Mode::WeightedModel : Mode::Standard;
}

inline std::vector<int32_t> mapInputFeaturesToModel(
        const std::vector<std::string>& inputFeatures,
        const std::vector<std::string>& modelFeatures) {
    std::unordered_map<std::string, int32_t> modelIndex;
    modelIndex.reserve(modelFeatures.size());
    for (size_t i = 0; i < modelFeatures.size(); ++i) {
        modelIndex.emplace(modelFeatures[i], static_cast<int32_t>(i));
    }

    std::vector<int32_t> inputToModel(inputFeatures.size(), -1);
    for (size_t i = 0; i < inputFeatures.size(); ++i) {
        const auto it = modelIndex.find(inputFeatures[i]);
        if (it != modelIndex.end()) {
            inputToModel[i] = it->second;
        }
    }
    return inputToModel;
}

template <typename Model>
bool appendDocument(Document&& rawDoc, std::string id, SpecialBatch& batch,
        Mode mode, const std::vector<int32_t>& inputToModel, Model& model,
        int32_t minCount) {
    Document modelDoc;
    if (mode == Mode::AllFeatures) {
        modelDoc.ids.reserve(rawDoc.ids.size());
        modelDoc.cnts.reserve(rawDoc.cnts.size());
        double modelTotal = 0.0;
        for (size_t j = 0; j < rawDoc.ids.size(); ++j) {
            const uint32_t inputFeature = rawDoc.ids[j];
            if (inputFeature >= inputToModel.size()) {
                continue;
            }
            const int32_t modelFeature = inputToModel[inputFeature];
            if (modelFeature < 0) {
                continue;
            }
            modelDoc.ids.push_back(static_cast<uint32_t>(modelFeature));
            modelDoc.cnts.push_back(rawDoc.cnts[j]);
            modelTotal += rawDoc.cnts[j];
        }
        modelDoc.raw_ct_tot = modelTotal;
        modelDoc.ct_tot = modelTotal;
    } else {
        modelDoc = std::move(rawDoc);
    }

    if (minCount > 0 && modelDoc.get_raw_sum() < minCount) {
        return false;
    }

    if (mode == Mode::WeightedModel) {
        batch.rawModelCounts.push_back(modelDoc.cnts);
    } else if (mode == Mode::AllFeatures) {
        batch.rawDocs.push_back(std::move(rawDoc));
    } else {
        error("%s: special batch used in standard mode", __func__);
    }
    model.applyWeights(modelDoc);
    batch.modelDocs.push_back(std::move(modelDoc));
    batch.ids.push_back(std::move(id));
    return true;
}

inline void accumulate(MatrixXd& pseudobulk, const SpecialBatch& batch,
        const RowMajorMatrixXd& docTopic, Mode mode) {
    if (docTopic.rows() != static_cast<int32_t>(batch.size())) {
        error("%s: topic rows do not match document count", __func__);
    }
    if (mode == Mode::WeightedModel &&
            batch.rawModelCounts.size() != batch.size()) {
        error("%s: raw-count sidecar does not match document count", __func__);
    }
    if (mode == Mode::AllFeatures && batch.rawDocs.size() != batch.size()) {
        error("%s: all-feature documents do not match document count", __func__);
    }

    const int32_t K = static_cast<int32_t>(docTopic.cols());
    tbb::parallel_for(tbb::blocked_range<int32_t>(0, K, 1),
        [&](const tbb::blocked_range<int32_t>& range) {
            for (int32_t k = range.begin(); k < range.end(); ++k) {
                if (mode == Mode::WeightedModel) {
                    for (size_t i = 0; i < batch.size(); ++i) {
                        const Document& doc = batch.modelDocs[i];
                        const auto& rawCounts = batch.rawModelCounts[i];
                        if (rawCounts.size() != doc.ids.size()) {
                            error("%s: raw counts do not align with model document",
                                __func__);
                        }
                        const double theta = docTopic(static_cast<int32_t>(i), k);
                        for (size_t j = 0; j < doc.ids.size(); ++j) {
                            pseudobulk(doc.ids[j], k) += rawCounts[j] * theta;
                        }
                    }
                } else if (mode == Mode::AllFeatures) {
                    for (size_t i = 0; i < batch.size(); ++i) {
                        const Document& doc = batch.rawDocs[i];
                        const double theta = docTopic(static_cast<int32_t>(i), k);
                        for (size_t j = 0; j < doc.ids.size(); ++j) {
                            pseudobulk(doc.ids[j], k) += doc.cnts[j] * theta;
                        }
                    }
                }
            }
        });
}

template <typename RandomEngine>
void randomize(SpecialBatch& batch, RandomEngine& randomEngine) {
    std::vector<size_t> order(batch.size());
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), randomEngine);

    SpecialBatch shuffled;
    shuffled.modelDocs.reserve(batch.modelDocs.size());
    shuffled.ids.reserve(batch.ids.size());
    shuffled.rawDocs.reserve(batch.rawDocs.size());
    shuffled.rawModelCounts.reserve(batch.rawModelCounts.size());
    for (size_t i : order) {
        shuffled.modelDocs.push_back(std::move(batch.modelDocs[i]));
        shuffled.ids.push_back(std::move(batch.ids[i]));
        if (!batch.rawDocs.empty()) {
            shuffled.rawDocs.push_back(std::move(batch.rawDocs[i]));
        }
        if (!batch.rawModelCounts.empty()) {
            shuffled.rawModelCounts.push_back(std::move(batch.rawModelCounts[i]));
        }
    }
    batch = std::move(shuffled);
}

inline void appendMoved(SpecialBatch& destination, SpecialBatch& source) {
    destination.modelDocs.insert(destination.modelDocs.end(),
        std::make_move_iterator(source.modelDocs.begin()),
        std::make_move_iterator(source.modelDocs.end()));
    destination.ids.insert(destination.ids.end(),
        std::make_move_iterator(source.ids.begin()),
        std::make_move_iterator(source.ids.end()));
    destination.rawDocs.insert(destination.rawDocs.end(),
        std::make_move_iterator(source.rawDocs.begin()),
        std::make_move_iterator(source.rawDocs.end()));
    destination.rawModelCounts.insert(destination.rawModelCounts.end(),
        std::make_move_iterator(source.rawModelCounts.begin()),
        std::make_move_iterator(source.rawModelCounts.end()));
    source.clear();
}

inline void moveRange(SpecialBatch& source, size_t begin, size_t count,
        SpecialBatch& destination) {
    destination.clear();
    const size_t end = begin + count;
    destination.modelDocs.insert(destination.modelDocs.end(),
        std::make_move_iterator(source.modelDocs.begin() + begin),
        std::make_move_iterator(source.modelDocs.begin() + end));
    destination.ids.insert(destination.ids.end(),
        std::make_move_iterator(source.ids.begin() + begin),
        std::make_move_iterator(source.ids.begin() + end));
    if (!source.rawDocs.empty()) {
        destination.rawDocs.insert(destination.rawDocs.end(),
            std::make_move_iterator(source.rawDocs.begin() + begin),
            std::make_move_iterator(source.rawDocs.begin() + end));
    }
    if (!source.rawModelCounts.empty()) {
        destination.rawModelCounts.insert(destination.rawModelCounts.end(),
            std::make_move_iterator(source.rawModelCounts.begin() + begin),
            std::make_move_iterator(source.rawModelCounts.begin() + end));
    }
}

} // namespace transform_pseudobulk
