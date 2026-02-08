#include "punkst.h"
#include "lda.hpp"
#include "hdp.hpp"
#include <memory>
#include <regex>

/**
 * Base class for online training of topic models.
 */
class TopicModelWrapper {
public:
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

    int32_t trainOnline(const std::string& inFile, int32_t _bsize, int32_t _minCountTrain, int32_t maxUnits = INT32_MAX);
    void load10X(DGEReader10X& dge, int32_t _minCountTrain, bool force = false);
    int32_t trainOnline10X(int32_t _bsize, int32_t maxUnits, int32_t seed);
    void fitAndWriteToFile10X(DGEReader10X& dge, const std::string& outPrefix, int32_t _bsize);
    // transform and writing results
    void fitAndWriteToFile(const std::string& inFile, const std::string& outPrefix, int32_t _bsize);

    int32_t nUnits() const { return reader.nUnits; }
    int32_t nFeatures() const { return M_; }
    bool hasFullFeatureSums() const { return reader.readFullSums; }
    void getTopicAbundance(std::vector<double>& topic_weights);
    virtual void filterTopics(double threshold, double coverage) {}
    std::vector<std::string> getFeatureNames() const {
        return featureNames.empty() ? reader.features : featureNames;
    }
    void applyWeights(Document& doc) const {
        reader.applyWeights(doc);
    }
    void printTopicAbundance();
    void writeModelToFile(const std::string& outFile);
    void writeModelHeader(std::ofstream& outFileStream);

    virtual int32_t getNumTopics() const = 0;
    virtual void sortTopicsByWeight() = 0;
    virtual void writeUnitHeader(std::ofstream& outFileStream) = 0;
    virtual const RowMajorMatrixXd& get_model_matrix() const = 0;
    virtual RowMajorMatrixXd copy_model_matrix() const = 0;
    virtual const std::vector<std::string>& get_topic_names() = 0;

    virtual void do_partial_fit(const std::vector<Document>& batch) = 0;
    virtual MatrixXd do_transform(const std::vector<Document>& batch) = 0;

    bool readMinibatch(std::ifstream& inFileStream, std::vector<Document>& batch, std::vector<std::string>& idens, int32_t batchSizeOverride, int32_t minCount = 0, int32_t maxUnits = INT32_MAX);

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
    std::vector<Document> dge_docs_cache_;
    std::vector<int32_t> dge_barcode_idx_cache_;
    std::vector<int32_t> dge_train_idx_cache_;
    int32_t dge_minCountTrain_cache_ = -1;
    bool dge_cache_ready_ = false;

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

    LDA4Hex(HexReader& _reader, int32_t modal = 0, int32_t verbose = 0) : TopicModelWrapper(_reader, modal, verbose) {}

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
        int32_t maxIter = 100, double mDelta = -1.) {
        RowMajorMatrixXd priorMatrix;
        initialize(0, priorMatrix, modelFile, -1, -1);
        lda = std::make_unique<LatentDirichletAllocation>(
            priorMatrix, seed, nThreads, verbose);
        initialized = true;
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
    void writeUnitHeader(std::ofstream& outFileStream) override {
        if (lda->get_algorithm() == InferenceType::SVB_DN) {
            outFileStream << "Background\t";
        }
        writeModelHeader(outFileStream);
    }

    void set_background_prior(std::string& bgPriorFile, double a0, double b0, double scale = 1., bool fixed = false) {
        std::ifstream priorIn(bgPriorFile, std::ios::in);
        if (!priorIn) {
            const std::vector<double>& eta0 = reader.getFeatureSums();
            if (std::abs(scale - 1.) > 1e-12) {
                std::vector<double> scaled_eta0(eta0.size());
                for (size_t i = 0; i < eta0.size(); ++i) {
                    scaled_eta0[i] = eta0[i] * scale;
                }
                lda->set_background_prior(scaled_eta0, a0, b0, fixed);
            } else {
                lda->set_background_prior(eta0, a0, b0, fixed);
            }
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
    MatrixXd do_transform(const std::vector<Document>& batch) override {
        return lda->transform(batch);
    }
    const RowMajorMatrixXd& get_model_matrix() const override {
        return lda->get_model();
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
        MatrixXd priorMatrix_ = MatrixXd(K_, M_);
        // Map columns from full prior matrix to subset matrix
        for (size_t i = 0; i < kept_indices.size(); ++i) {
            priorMatrix_.col(i) = fullPriorMatrix.col(kept_indices[i]);
        }
        notice("Created subset prior matrix: %d topics x %d features (from original %d features)",
                (int)priorMatrix_.rows(), (int)priorMatrix_.cols(), (int)priorFeatureNames.size());
        priorMatrix = priorMatrix_;
        // Apply scaling if specified
        if (priorScaleRel > 0 && reader.readFullSums) {
            const std::vector<double>& featureSumsRaw = reader.getFeatureSumsRaw();
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
            priorMatrix_ *= priorScale;
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
    void writeUnitHeader(std::ofstream& outFileStream) override {
        writeModelHeader(outFileStream);
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
    MatrixXd do_transform(const std::vector<Document>& batch) override {
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
