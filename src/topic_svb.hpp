#pragma once

#include "punkst.h"
#include "lda.hpp"
#include "hdp.hpp"
#include "document_spool.hpp"
#include "lda_state.hpp"
#include <memory>
#include <regex>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <stdexcept>

/**
 * Base class for online training of topic models.
 */
class TopicModelWrapper {
public:
    struct TransformOutputOptions {
        bool appendTopK = false;
        bool dropRandomKey = false;
        bool scientific = false;
        int32_t randomKeyIndex = -1;
        std::string topKColname = "topK";
        std::string topPColname = "topP";
    };

    TopicModelWrapper(HexReader& _reader, int32_t modal = 0, int32_t verbose = 0) : modal(modal), verbose_(verbose) {
        reader = std::move(_reader);
        if (modal >= reader.getNmodal()) {
            error("modal %d is out of range", modal);
        }
        M_ = reader.nFeatures;
        ntot = 0;
        minCountTrain = 0;
        initialized = false;
    }

    virtual ~TopicModelWrapper() = default;

    int32_t trainOnline(const std::string& inFile, int32_t _bsize,
        int32_t _minCountTrain, int32_t maxUnits = INT32_MAX,
        punkst::DocumentBatchSink* cacheSink = nullptr);
    int32_t trainOnline(punkst::DocumentBlockSource& source,
        int32_t _bsize, int32_t maxUnits = INT32_MAX);
    int32_t trainOnline(
        const std::vector<std::vector<Document>>& residentBatches,
        int32_t _bsize, int32_t maxUnits = INT32_MAX);
    void prepare10XCache(DGEReader10X& dge, int32_t _minCountTrain,
        bool force = false, bool collectDetectionPrevalence = false);
    int32_t trainOnline10X(int32_t _bsize, int32_t maxUnits, int32_t seed);
    void fitAndWriteToFile10X(DGEReader10X& dge, const std::string& outPrefix, int32_t _bsize);
    int32_t filterCurrentFeatures(int32_t minCount = 1,
                                  const std::string& includeRegex = "",
                                  const std::string& excludeRegex = "");
    // transform and writing results
    void fitAndWriteToFile(const std::string& inFile, const std::string& outPrefix, int32_t _bsize);

    int32_t nUnits() const { return reader.nUnits; }
    int32_t nFeatures() const { return M_; }
    bool hasFullFeatureSums() const { return reader.readFullSums; }
    const std::vector<double>& getFeatureSumsRaw() const { return reader.getFeatureSumsRaw(); }
    virtual void getTopicAbundance(std::vector<double>& topic_weights);
    virtual void filterTopics(double threshold, double coverage) {}
    std::vector<std::string> getFeatureNames() const {
        return featureNames.empty() ? reader.features : featureNames;
    }
    void applyWeights(Document& doc) const {
        reader.applyWeights(doc);
    }
    double rawCountFor(uint32_t feature, double count, bool countWeighted) const {
        return reader.rawCountFor(feature, count, countWeighted);
    }
    void printTopicAbundance();
    void writeModelToFile(const std::string& outFile);
    void writeModelHeader(std::ostream& outFileStream);
    void writeUnitHeader(std::ostream& outFileStream);
    void setTransformOutputOptions(bool appendTopK,
                                   const std::string& topKColname = "topK",
                                   const std::string& topPColname = "topP",
                                   bool dropRandomKey = false) {
        transformOutputOptions_.appendTopK = appendTopK;
        transformOutputOptions_.dropRandomKey = dropRandomKey;
        transformOutputOptions_.randomKeyIndex = dropRandomKey ? reader.getIndex("random_key") : -1;
        transformOutputOptions_.topKColname = topKColname.empty() ? "topK" : topKColname;
        transformOutputOptions_.topPColname = topPColname.empty() ? "topP" : topPColname;
    }

    virtual int32_t getNumTopics() const = 0;
    virtual void sortTopicsByWeight() = 0;
    virtual void getUnitHeaderCols(std::vector<std::string>& outCols) = 0;
    virtual const RowMajorMatrixXd& get_model_matrix() const = 0;
    virtual RowMajorMatrixXd copy_model_matrix() const = 0;
    virtual const std::vector<std::string>& get_topic_names() = 0;

    virtual void do_partial_fit(const std::vector<Document>& batch) = 0;
    virtual MatrixXd do_transform(DocumentView batch) = 0;

    bool readMinibatch(std::ifstream& inFileStream, std::vector<Document>& batch, std::vector<std::string>& idens, int32_t batchSizeOverride, int32_t minCount = 0, int32_t maxUnits = INT32_MAX);
    int32_t readAllDocuments(std::vector<Document>& docs,
                             const std::string& inFile,
                             int32_t minCount = 0, int32_t maxUnits = INT32_MAX);

protected:
    HexReader reader;
    int32_t modal;
    int32_t ntot; // Number of documents processed in trainOnline
    int32_t M_; // Number of features
    int32_t minCountTrain;
    bool initialized;
    double defaultWeight;
    std::vector<std::string> featureNames;
    std::vector<Document> minibatch;
    int32_t batchSize;
    int32_t verbose_;
    TransformOutputOptions transformOutputOptions_;
    std::vector<Document> dge_docs_cache_;
    std::vector<std::string> dge_unit_id_cache_;
    std::vector<int32_t> dge_train_idx_cache_;
    std::vector<double> feature_detection_fraction_;
    int32_t dge_minCountTrain_cache_ = -1;
    bool dge_cache_ready_ = false;
    bool dge_detection_prevalence_ready_ = false;

    // --- Shared Helper Methods ---
    bool readMinibatch(std::ifstream& inFileStream);
    bool readMinibatch(std::ifstream& inFileStream, std::vector<std::string>& idens, bool labeled = false);
    // Create a new feature space that is the intersection of
    // 1. Current filtered features and 2. The input features
    // The output will follow the input's ordering
    virtual void setupPriorMapping(std::vector<std::string>& feature_names_, std::vector<std::uint32_t>& kept_indices);
};








/**
 * Wrapper for LDA
 */
class LDA4Hex : public TopicModelWrapper {

public:

    LDA4Hex(HexReader& _reader, int32_t modal = 0, int32_t verbose = 0)
        : TopicModelWrapper(_reader, modal, verbose) {
        transformOutputOptions_.scientific = true;
    }

    void initialize_scvb0(int32_t nTopics, int32_t seed = -1,
        int32_t nThreads = 0, int32_t verbose = 0,
        double alpha = -1., double eta = -1.,
        double kappa = 0.9, double tau0 = 1000.,
        int32_t totalDocCount = 1000000,
        const std::string& priorFile = "",
        double priorScale = -1., double priorScaleRel = -1.,
        double s_beta = 1, double s_theta = 1, double kappa_theta = 0.7, double tau_theta = 10.0, int32_t burnin = 1) {
        RowMajorMatrixXd priorMatrix;
        initialize(nTopics, priorMatrix, priorFile, priorScale, priorScaleRel);
        lda = std::make_unique<LatentDirichletAllocation>(
            K_, M_, seed, nThreads, verbose,
            InferenceType::SCVB0,
            alpha, eta,
            kappa, tau0, totalDocCount,
            nullptr, priorMatrix, -1.);
        lda->set_scvb0_parameters(s_beta, s_theta, tau_theta, kappa_theta, burnin);
        initialized = true;
    }

    void initialize_svb(int32_t nTopics, int32_t seed = -1,
        int32_t nThreads = 0, int32_t verbose = 0,
        double alpha = -1., double eta = -1.,
        double kappa = 0.7, double tau0 = 10.0,
        int32_t totalDocCount = 1000000,
        const std::string& priorFile = "",
        double priorScale = -1., double priorScaleRel = -1.,
        int32_t maxIter = 100, double mDelta = -1.) {
        RowMajorMatrixXd priorMatrix;
        initialize(nTopics, priorMatrix, priorFile, priorScale, priorScaleRel);
        lda = std::make_unique<LatentDirichletAllocation>(
            K_, M_, seed, nThreads, verbose,
            InferenceType::SVB,
            alpha, eta,
            kappa, tau0, totalDocCount,
            nullptr, priorMatrix, -1.);
        lda->set_svb_parameters(maxIter, mDelta);
        initialized = true;
    }

    void initialize_transform(const std::string& modelFile,
        int seed = std::random_device{}(), int nThreads = 0, int verbose = 0,
        int32_t maxIter = 100, double mDelta = -1., double alpha = -1.) {
        RowMajorMatrixXd priorMatrix;
        initialize(0, priorMatrix, modelFile, -1, -1);
        lda = std::make_unique<LatentDirichletAllocation>(
            priorMatrix, seed, nThreads, verbose, InferenceType::SVB, alpha);
        lda->set_svb_parameters(maxIter, mDelta);
        initialized = true;
    }

    void initialize_transform(const LdaState& state,
        int seed = std::random_device{}(), int nThreads = 0, int verbose = 0,
        int32_t maxIter = 100, double mDelta = -1.) {
        state.validate();
        std::vector<std::uint32_t> kept_indices;
        std::vector<std::string> state_features = state.features;
        setupPriorMapping(state_features, kept_indices);
        RowMajorMatrixXd components(state.topics.size(), kept_indices.size());
        for (size_t feature = 0; feature < kept_indices.size(); ++feature) {
            components.col(feature) = state.components.col(kept_indices[feature]);
        }
        if (state.feature_weights_active) {
            std::vector<double> weights(kept_indices.size());
            for (size_t feature = 0; feature < kept_indices.size(); ++feature) {
                weights[feature] = state.feature_weights[kept_indices[feature]];
            }
            reader.setFeatureWeights(weights);
        }
        K_ = static_cast<int32_t>(state.topics.size());
        topicNames = state.topics;
        lda = std::make_unique<LatentDirichletAllocation>(
            K_, static_cast<int32_t>(kept_indices.size()), seed, nThreads,
            verbose, state.has_background
                ? InferenceType::SVB_DN : InferenceType::SVB,
            state.alpha, state.eta, -1.0, -1.0, -1,
            nullptr, components, -1.0);
        if (state.has_background) {
            VectorXd background_prior(kept_indices.size());
            VectorXd background_components(kept_indices.size());
            for (size_t feature = 0; feature < kept_indices.size(); ++feature) {
                background_prior(feature) =
                    state.background_prior(kept_indices[feature]);
                background_components(feature) =
                    state.background_components(kept_indices[feature]);
            }
            lda->set_background_state(background_prior, background_components,
                state.background_prior_a, state.background_prior_b,
                state.background_count, state.foreground_count,
                state.background_fixed);
        }
        lda->set_svb_parameters(maxIter, mDelta);
        initialized = true;
    }

    void writeStateToFile(const std::string& path) const {
        if (!initialized || !lda
                || (lda->get_algorithm() != InferenceType::SVB
                    && lda->get_algorithm() != InferenceType::SVB_DN)) {
            error("%s: LDA SVB is required", __FUNCTION__);
        }
        LdaState state;
        state.alpha = lda->get_doc_topic_prior();
        state.eta = lda->get_topic_word_prior();
        state.topics = const_cast<LDA4Hex*>(this)->get_topic_names();
        state.features = getFeatureNames();
        state.components = lda->get_model();
        state.feature_weights_active = reader.hasFeatureWeights();
        if (state.feature_weights_active) {
            state.feature_weights = reader.getFeatureWeights();
        }
        state.has_background = lda->has_background();
        if (state.has_background) {
            state.background_fixed = lda->background_is_fixed();
            state.background_prior_a = lda->get_background_prior_a();
            state.background_prior_b = lda->get_background_prior_b();
            state.background_count = lda->get_background_count();
            state.foreground_count = lda->get_forground_count();
            state.background_prior = lda->get_background_prior();
            state.background_components = lda->get_background_model();
        }
        state.write(path);
    }

    bool has_background() const {
        return lda && lda->has_background();
    }

    const VectorXd& get_background_model() const {
        if (!has_background()) {
            error("%s: LDA model has no background", __FUNCTION__);
        }
        return lda->get_background_model();
    }

    void beginTopicUsageCollection() {
        if (!initialized || !lda) {
            error("%s: initialized LDA is required", __FUNCTION__);
        }
        lda->begin_topic_usage_collection();
    }

    std::vector<double> finishTopicUsageCollection() {
        if (!initialized || !lda) {
            error("%s: initialized LDA is required", __FUNCTION__);
        }
        return lda->finish_topic_usage_collection();
    }

    std::vector<int32_t> pruneTopicsByUsage(
            const std::vector<double>& usage, double threshold) {
        if (usage.size() != static_cast<size_t>(K_) || !(threshold >= 0.0)) {
            throw std::invalid_argument("Invalid LDA adaptive-topic usage");
        }
        std::vector<int32_t> order(K_);
        std::iota(order.begin(), order.end(), 0);
        std::vector<int32_t> keep;
        for (int32_t topic : order) {
            if (std::isfinite(usage[topic]) && usage[topic] >= threshold) {
                keep.push_back(topic);
            }
        }
        if (keep.size() < 2) {
            std::stable_sort(order.begin(), order.end(), [&](int32_t left,
                    int32_t right) { return usage[left] > usage[right]; });
            keep.assign(order.begin(), order.begin() + std::min<int32_t>(2, K_));
            std::sort(keep.begin(), keep.end());
        }
        if (keep.size() == static_cast<size_t>(K_)) return keep;
        lda->prune_topics(keep);
        K_ = lda->get_n_topics();
        topicNames = lda->get_topic_names();
        return keep;
    }

    void writeTopicSpecificity(const std::string& path) const {
        if (!has_background()) return;
        std::ofstream output(path);
        if (!output) error("Cannot write LDA topic specificity: %s", path.c_str());
        const RowMajorMatrixXd& components = lda->get_model();
        const VectorXd topic_totals = components.rowwise().sum();
        const VectorXd& background = lda->get_background_model();
        const double background_total = background.sum();
        output << "Feature\tBackground";
        const auto& names = const_cast<LDA4Hex*>(this)->get_topic_names();
        for (const auto& name : names) output << '\t' << name << "_Probability";
        for (const auto& name : names) output << '\t' << name << "_KLContribution";
        output << '\n' << std::scientific << std::setprecision(8);
        for (int32_t feature = 0; feature < M_; ++feature) {
            const double background_probability =
                background(feature) / background_total;
            output << featureNames[feature] << '\t'
                << background_probability;
            for (int32_t topic = 0; topic < K_; ++topic) {
                output << '\t'
                    << components(topic, feature) / topic_totals(topic);
            }
            for (int32_t topic = 0; topic < K_; ++topic) {
                const double probability =
                    components(topic, feature) / topic_totals(topic);
                const double score = probability * std::log(
                    std::max(probability, 1e-300)
                    / std::max(background_probability, 1e-300));
                output << '\t' << score;
            }
            output << '\n';
        }
    }

    int32_t getNumTopics() const override {
        return lda ? lda->get_n_topics() : 0;
    }
    void sortTopicsByWeight() override {
        if (lda) lda->sort_topics();
    }
    void get_topic_abundance(std::vector<double>& topic_weights) {
        if (!initialized || !lda) {
            error("%s: LDA4Hex is not initialized", __FUNCTION__);
        }
        lda->get_topic_abundance(topic_weights);
    }
    void set_reproducible_init(bool enabled = true) {
        if (!initialized || !lda) {
            error("%s: LDA4Hex is not initialized", __FUNCTION__);
        }
        lda->set_deterministic_rng(enabled);
    }
    void preparePriorFeatureSpace(const std::string& priorFile) {
        if (priorFile.empty()) {
            return;
        }
        std::vector<std::string> priorFeatureNames;
        std::vector<std::uint32_t> kept_indices;
        readModelFeatureNamesFromTsv(priorFile, priorFeatureNames);
        setupPriorMapping(priorFeatureNames, kept_indices);
    }
    void writeBackgroundModel(std::string& outFile) {
        if (lda->get_algorithm() != InferenceType::SVB_DN) {
            return;
        }
        std::ofstream outFileStream(outFile, std::ios::out);
        if (!outFileStream) {
            error("%s: Failed to open output file: %s", __FUNCTION__, outFile.c_str());
        }
        double a = lda->get_background_count();
        double b = lda->get_forground_count();
        outFileStream << std::fixed << std::setprecision(3);
        outFileStream << "##a=" << a << ";b=" << b
                      << ";pi=" << (a / (a + b)) << "\n";
        outFileStream << "#Feature\tBackground\n";
        const auto& lambda0 = lda->get_background_model();
        for (int32_t j = 0; j < M_; ++j) {
            outFileStream << featureNames[j] << "\t" << lambda0(j) << "\n";
        }
        outFileStream.close();
    }
    void getUnitHeaderCols(std::vector<std::string>& outCols) override {
        outCols.clear();
        if (lda->get_algorithm() == InferenceType::SVB_DN) {
            outCols.push_back("Background");
        }
        const auto& topicNames = get_topic_names();
        outCols.insert(outCols.end(), topicNames.begin(), topicNames.end());
    }

    void set_background_prior(std::string& bgPriorFile, double a0, double b0,
            double scale = 1., bool fixed = false,
            double prevalencePower = 0.0) {
        std::ifstream priorIn(bgPriorFile, std::ios::in);
        if (!priorIn) {
            const std::vector<double>& eta0 = reader.getFeatureSums();
            std::vector<double> scaled_eta0(eta0.begin(), eta0.end());
            if (prevalencePower > 0.0) {
                if (feature_detection_fraction_.size() != eta0.size()) {
                    error("%s: feature detection prevalence is unavailable",
                        __FUNCTION__);
                }
                const double original_total = std::accumulate(
                    scaled_eta0.begin(), scaled_eta0.end(), 0.0);
                for (size_t i = 0; i < scaled_eta0.size(); ++i) {
                    scaled_eta0[i] *= std::pow(std::max(
                        feature_detection_fraction_[i], 1e-8),
                        prevalencePower);
                }
                const double weighted_total = std::accumulate(
                    scaled_eta0.begin(), scaled_eta0.end(), 0.0);
                if (original_total > 0.0 && weighted_total > 0.0) {
                    const double renormalize = original_total / weighted_total;
                    for (double& value : scaled_eta0) value *= renormalize;
                }
            }
            for (double& value : scaled_eta0) value *= scale;
            lda->set_background_prior(scaled_eta0, a0, b0, fixed);
        } else {
            std::unordered_map<std::string, uint32_t> featureDict;
            if (!reader.featureDict(featureDict)) {
                error("%s: Feature names must be set to use background prior from file", __FUNCTION__);
            }
            std::string line;
            std::vector<std::string> tokens;
            std::vector<double> eta0(reader.nFeatures, 0.0);
            while(std::getline(priorIn, line)) {
                if (line.empty() || line[0] == '#') {continue;}
                split(tokens, "\t ", line, 3);
                if (tokens.size() < 2) {
                    error("%s: Invalid line in background prior file: %s", line.c_str());
                }
                auto it = featureDict.find(tokens[0]);
                if (it != featureDict.end()) {
                    eta0[it->second] = std::stod(tokens[1]);
                }
            }
            priorIn.close();
            lda->set_background_prior(eta0, a0, b0, fixed);
        }
    }

    void do_partial_fit(const std::vector<Document>& batch) override {
        lda->partial_fit(batch);
    }
    MatrixXd do_transform(DocumentView batch) override {
        return lda->transform(batch);
    }
    RowMajorMatrixXd do_transform_gamma(DocumentView batch) {
        return lda->transform_gamma(batch);
    }
    double get_doc_topic_prior() const {
        if (!initialized || !lda) {
            error("%s: LDA4Hex is not initialized", __FUNCTION__);
        }
        return lda->get_doc_topic_prior();
    }
    const RowMajorMatrixXd& get_model_matrix() const override {
        return lda->get_model();
    }
    const MatrixXd& get_allocation_kernel() const {
        if (!initialized || !lda) {
            error("%s: LDA4Hex is not initialized", __FUNCTION__);
        }
        return lda->get_allocation_kernel();
    }
    RowMajorMatrixXd copy_model_matrix() const override {
        return lda->get_model();
    }
    const std::vector<std::string>& get_topic_names() override {
        if (topicNames.empty()) {
            topicNames = lda->get_topic_names();
        }
        return topicNames;
    }

protected:

    int32_t K_;
    std::vector<std::string> topicNames;
    std::unique_ptr<LatentDirichletAllocation> lda;

    void initialize(int32_t nTopics) {
        K_ = nTopics;
        if (reader.features.size() != M_) {
            notice("%s: no valid feature names are set, will use 0-based indices in the output model file", __FUNCTION__);
            featureNames.resize(M_);
            for (int i = 0; i < M_; ++i) {
                featureNames[i] = std::to_string(i);
            }
        } else {
            featureNames = reader.features;
        }
        return;
    }

    void initialize(int32_t nTopics, RowMajorMatrixXd& priorMatrix, const std::string& priorFile, double priorScale = -1, double priorScaleRel = -1) {
        if (priorFile.empty()) {
            initialize(nTopics);
            return;
        }
        // Read prior model file
        std::vector<std::string> priorFeatureNames;
        std::vector<std::uint32_t> kept_indices;
        MatrixXd fullPriorMatrix;
        readModelFromTsv(priorFile, priorFeatureNames, fullPriorMatrix);

        // Setup feature mapping between input data and prior model
        setupPriorMapping(priorFeatureNames, kept_indices);

        // Create subset matrix for only the intersected features
        priorMatrix.resize(K_, M_);
        // Map columns from full prior matrix to subset matrix
        for (size_t i = 0; i < kept_indices.size(); ++i) {
            priorMatrix.col(i) = fullPriorMatrix.col(kept_indices[i]);
        }
        notice("Created subset prior matrix: %d topics x %d features (from original %d features)",
                (int)priorMatrix.rows(), (int)priorMatrix.cols(), (int)priorFeatureNames.size());
        // Apply scaling if specified
        if (priorScaleRel > 0 && reader.readFullSums) {
            const std::vector<double>& featureSumsRaw = reader.hasFeatureWeights()
                ? reader.getFeatureSums()
                : reader.getFeatureSumsRaw();
            double totalCount = 0.;
            for (double s : featureSumsRaw) { totalCount += s;  }
            if (totalCount <= 0) {
                const std::vector<double>& weightedSums = reader.getFeatureSums();
                totalCount = 0.;
                for (double s : weightedSums) { totalCount += s; }
            }
            if (totalCount <= 0) {
                error("%s: total feature count is zero; check --features totals or input data", __func__);
            }
            double globalScale0 = priorMatrix.sum() / totalCount;
            double targetTotal = totalCount / K_ * priorScaleRel;
            VectorXd priorSums = priorMatrix.rowwise().sum();
            for (int32_t k = 0; k < K_; ++k) {
                if (priorSums(k) < targetTotal) {
                    continue;
                }
                double scale = targetTotal / priorSums(k);
                priorMatrix.row(k) *= scale;
            }
            double globalScale1 = priorMatrix.sum() / totalCount;

            notice("%s: total count of the overlapping features in the prior matrix is %.2fX that in the data, %.2fX after scaling each factor to be <= %.2f/K (%.2e) of the data total", __func__, globalScale0, globalScale1, priorScaleRel, priorScaleRel/K_);
        } else if (priorScale > 0. && priorScale != 1.) {
            priorMatrix *= priorScale;
        }

    }

    void readModelFromTsv(const std::string& modelFile, std::vector<std::string>& _featureNames, MatrixXd& modelMatrix) {
        std::ifstream modelIn(modelFile, std::ios::in);
        if (!modelIn) {
            error("Failed to open model file: %s", modelFile.c_str());
        }

        std::string line;
        std::vector<std::string> tokens;

        // Read header to get topic names and count
        std::getline(modelIn, line);
        split(tokens, "\t", line);
        K_ = tokens.size() - 1; // first column is "Feature"
        topicNames = std::vector<std::string>(tokens.begin() + 1, tokens.end());

        // Read all feature rows
        _featureNames.clear();
        std::vector<std::vector<double>> modelValues;
        while (std::getline(modelIn, line)) {
            split(tokens, "\t", line);
            if (tokens.size() != K_ + 1) {
                error("Invalid line in model file: %s", line.c_str());
            }
            _featureNames.push_back(tokens[0]);
            std::vector<double> values(K_);
            for (int32_t i = 0; i < K_; ++i) {
                values[i] = std::stod(tokens[i + 1]);
            }
            modelValues.push_back(values);
        }
        modelIn.close();

        int32_t nFeatures = _featureNames.size();
        modelMatrix.resize(K_, nFeatures);
        for (int32_t i = 0; i < nFeatures; ++i) {
            for (int32_t j = 0; j < K_; ++j) {
                modelMatrix(j, i) = modelValues[i][j];
            }
        }

        notice("Read model matrix: %d topics x %d features from %s", K_, nFeatures, modelFile.c_str());
    }
    void readModelFeatureNamesFromTsv(const std::string& modelFile, std::vector<std::string>& _featureNames) {
        std::ifstream modelIn(modelFile, std::ios::in);
        if (!modelIn) {
            error("Failed to open model file: %s", modelFile.c_str());
        }

        std::string line;
        std::vector<std::string> tokens;
        if (!std::getline(modelIn, line)) {
            error("Failed to read header from model file: %s", modelFile.c_str());
        }
        split(tokens, "\t", line);
        if (tokens.size() < 2) {
            error("Invalid model header in file: %s", modelFile.c_str());
        }

        _featureNames.clear();
        while (std::getline(modelIn, line)) {
            if (line.empty()) {
                continue;
            }
            split(tokens, "\t", line);
            if (tokens.empty()) {
                continue;
            }
            _featureNames.push_back(tokens[0]);
        }
        modelIn.close();

        if (_featureNames.empty()) {
            error("No features found in model file: %s", modelFile.c_str());
        }
    }


};




/**
 * Wrapper for HDP
 */
class HDP4Hex : public TopicModelWrapper {
public:
    HDP4Hex(HexReader& _reader, int32_t modal = 0, int32_t verbose = 0)
        : TopicModelWrapper(_reader, modal, verbose), K_(0), num_topics_to_output_(-1) {}

    // HDP-specific initializer
    void initialize(int32_t K, int32_t T, int32_t seed, int32_t nThreads, int32_t verbose, double eta, double alpha, double omega, double kappa, double tau0, int32_t totalDocCount, int32_t maxIter, double mDelta) {
        K_ = K;
        T_ = T;
        if (reader.features.size() != M_) {
            featureNames.resize(M_);
            for (int i = 0; i < M_; ++i) featureNames[i] = std::to_string(i);
        } else {
            featureNames = reader.features;
        }
        hdp = std::make_unique<HDP>(K, T, M_, nThreads, seed, verbose,
            eta, alpha, omega, totalDocCount, tau0, kappa, maxIter, mDelta);
        initialized = true;
    }

    // Implementation of pure virtual methods
    int32_t getNumTopics() const override {
        // If filtering has been applied, return the filtered count.
        if (num_topics_to_output_ > -1) {
            return num_topics_to_output_;
        }
        // Otherwise, return the max number of topics.
        return hdp ? hdp->get_K() : 0;
    }
    void get_topic_abundance(std::vector<double>& topic_weights) {
        if (!initialized || !hdp) {
            error("%s: HDP4Hex is not initialized", __FUNCTION__);
        }
        hdp->get_topic_abundance(topic_weights);
    }
    void sortTopicsByWeight() override {
        if (hdp) hdp->sort_topics();
    }
    void getUnitHeaderCols(std::vector<std::string>& outCols) override {
        outCols = get_topic_names();
    }
    void filterTopics(double threshold, double coverage) override {
        if (!initialized || !hdp) {
            error("HDP must be initialized to filter topics.");
        }
        num_topics_to_output_ = K_;
        if (coverage <= 0 && coverage >= 1.0 && threshold <= 0.0 && threshold >= 1.0) {
            warning("%s: Invalid thresholds, no filtering applied.", __FUNCTION__);
            return;
        }
        sorted_indices_ = hdp->sort_topics();
        std::vector<double> topic_weights;
        hdp->get_topic_abundance(topic_weights); // Gets the sorted, relative weights
        // Strategy 1: Threshold-based filtering
        if (threshold > 0.0 && threshold < 1.0) {
            uint32_t k = 0;
            for (; k < K_; ++k) {
                if (topic_weights[sorted_indices_[k]] < threshold) {
                    break;
                }
            }
            num_topics_to_output_ = k;
            notice("%u out of %d topics have relative weight >= %.4f", k, K_, threshold);
        }
        // Strategy 2: Coverage-based filtering
        if (coverage > 0 && coverage < 1.0) {
            double cumulative_weight = 0.0;
            uint32_t k = 0;
            for (uint32_t k = 0; k < K_; ++k) {
                cumulative_weight += topic_weights[sorted_indices_[k]];
                if (cumulative_weight >= coverage) {
                    break;
                }
            }
            if (k < num_topics_to_output_) {
                num_topics_to_output_ = k;
            }
            notice("Top %d out of %d topics cover >= %.3f%% of data", k, K_, coverage * 100);
        }
    }
    void do_partial_fit(const std::vector<Document>& batch) override {
        hdp->partial_fit(batch);
    }
    MatrixXd do_transform(DocumentView batch) override {
        MatrixXd theta = hdp->transform(batch);
        colNormalizeInPlace(theta);
        return theta;
    }
    const RowMajorMatrixXd& get_model_matrix() const override {
        return hdp->get_model();
    }
    RowMajorMatrixXd copy_model_matrix() const override {
        // If no filtering has been applied, return the full model.
        if (num_topics_to_output_ < 0 || num_topics_to_output_ >= K_) {
            return hdp->get_model();
        }
        const MatrixXd& fullModel = hdp->get_model();
        MatrixXd truncatedModel(num_topics_to_output_, M_);
        for (int32_t i = 0; i < num_topics_to_output_; ++i) {
            truncatedModel.row(i) = fullModel.row(sorted_indices_[i]);
        }
        return truncatedModel;
    }
    const std::vector<std::string>& get_topic_names() override {
        if (topicNames_.empty()) {
            topicNames_.resize(getNumTopics());
            for(int i=0; i<getNumTopics(); ++i)
                topicNames_[i] = std::to_string(i);
        }
        return topicNames_;
    }

private:
    // HDP-specific members
    std::unique_ptr<HDP> hdp;
    int32_t K_; // Maximum number of topics
    int32_t T_; // Minimum number of topics per document
    int32_t num_topics_to_output_;
    std::vector<std::string> topicNames_;
    std::vector<int32_t> sorted_indices_;
};
