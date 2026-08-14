#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <memory>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <tbb/blocked_range.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include "dataunits.hpp"
#include "error.hpp"
#include "gamma_pois_dispersion.hpp"
#include "numerical_utils.hpp"
#include "topic_svb.hpp"

struct GammaPoissonDocumentPosterior {
    VectorXd shape;
    VectorXd rate;
    double exposure = 0.0;
};

struct GammaPoissonStateFeatureInfo {
    std::vector<std::string> names;
    std::vector<double> training_count;
    bool feature_weights_active = false;
    std::vector<double> feature_weight;
};

class GammaPoissonTopicBase {
public:
    GammaPoissonTopicBase() = default;
    GammaPoissonTopicBase(int32_t n_topics, int32_t n_features,
        int seed = std::random_device{}(), int32_t nThreads = 0, int32_t verbose = 0,
        double beta_shape = -1.0, double xi_shape = 0.3, double xi_mean = -1.0,
        double theta_concentration = 1.0, double nu_shape = 1.0, double nu_rate = -1.0,
        double learning_decay = 0.7, double learning_offset = 10.0,
        int32_t total_doc_count = 1000000, double size_factor = 1.0,
        const std::vector<double>* feature_sums = nullptr,
        double random_init_shape = 0.5);
    virtual ~GammaPoissonTopicBase() = default;

    int32_t get_n_topics() const { return n_topics_; }
    int32_t get_n_features() const { return n_features_; }
    double get_size_factor() const { return size_factor_; }
    const std::vector<std::string>& get_topic_names();
    const std::vector<std::string>& get_feature_names() const { return feature_names_; }
    const std::vector<double>& get_training_count() const {
        return training_count_;
    }
    bool feature_weights_active() const { return feature_weights_active_; }
    const std::vector<double>& get_feature_weight() const {
        return feature_weight_;
    }
    const RowMajorMatrixXd& get_model();
    RowMajorMatrixXd copy_model();
    void get_topic_abundance(std::vector<double>& weights) const;
    void sort_topics();
    void set_svb_parameters(int32_t max_iter, double tol);
    void set_nthreads(int32_t nThreads);
    void prepare_inference_cache();
    void initialize_topic_profiles(
        const Eigen::Ref<const RowMajorMatrixXd>& profiles,
        const std::vector<std::string>& topic_names = {});
    void write_model(const std::string& outFile, const std::vector<std::string>& featureNames);

protected:
    static int normalize_seed(int seed);
    void init_from_feature_sums(const std::vector<double>* feature_sums);
    void refresh_cache();
    void refresh_model_cache();
    RowVectorXd normalized_theta_hat(const VectorXd& theta_shape, const VectorXd& theta_rate) const;
    double doc_exposure(const Document& doc) const;
    double expected_xi(int32_t w) const { return xi_shape_(w) / xi_rate_(w); }

    int32_t n_topics_ = -1;
    int32_t n_features_ = -1;
    int seed_ = 1;
    int32_t nThreads_ = 1;
    int32_t verbose_ = 0;
    int32_t total_doc_count_ = 1000000;
    int32_t update_count_ = 0;
    double a_ = -1.0;
    double a0_ = 0.3;
    double b0_ = -1.0;
    double e0_ = 1.0;
    double f0_ = 1.0;
    double learning_decay_ = 0.7;
    double learning_offset_ = 10.0;
    double size_factor_ = 1.0;
    double random_init_shape_ = 0.5;
    double eps_ = std::numeric_limits<double>::epsilon();
    int32_t max_doc_update_iter_ = 100;
    double mean_change_tol_ = 1e-3;

    MatrixXd beta_shape_; // K x W, column-major for feature-wise updates
    MatrixXd beta_rate_;
    MatrixXd e_beta_;
    MatrixXd beta_kernel_; // centered exp(E[log beta]) by feature
    RowMajorMatrixXd model_phi_;
    bool model_cache_dirty_ = true;
    bool inference_cache_ready_ = false;
    VectorXd topic_capacity_; // K, \sum_w E[\beta_{kw}]
    VectorXd xi_shape_; // W
    VectorXd xi_rate_;
    VectorXd topic_usage_;
    std::vector<std::string> topic_names_;
    std::vector<std::string> feature_names_;
    std::vector<double> training_count_;
    bool feature_weights_active_ = false;
    std::vector<double> feature_weight_;
    std::mt19937 random_engine_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;
};

class GammaPoissonTopicModel : public GammaPoissonTopicBase {
public:
    GammaPoissonTopicModel(int32_t n_topics, int32_t n_features,
        int seed = std::random_device{}(), int32_t nThreads = 0, int32_t verbose = 0,
        double beta_shape = -1.0, double xi_shape = 0.3, double xi_mean = -1.0,
        double theta_concentration = 1.0,
        double nu_shape = 1.0, double nu_rate = -1.0,
        double learning_decay = 0.7, double learning_offset = 10.0,
        int32_t total_doc_count = 1000000, double size_factor = 1.0,
        bool symmetric_nu = true, double nu_max = -1.0,
        const std::vector<double>* feature_sums = nullptr,
        double random_init_shape = 0.5);

    explicit GammaPoissonTopicModel(const std::string& stateFile,
        int seed = std::random_device{}(), int32_t nThreads = 0, int32_t verbose = 0);
    static std::unique_ptr<GammaPoissonTopicModel> load_state_deferred(
        const std::string& stateFile, int seed = std::random_device{}(),
        int32_t nThreads = 0, int32_t verbose = 0);

    void partial_fit(const std::vector<Document>& docs);
    RowMajorMatrixXd transform(DocumentView docs);
    void transform_with_posteriors(DocumentView docs, RowMajorMatrixXd& topics,
        std::vector<GammaPoissonDocumentPosterior>& posteriors) const;
    void infer_document_posterior(const Document& doc,
        GammaPoissonDocumentPosterior& posterior) const;
    RowVectorXd normalized_topic_mean(
        const GammaPoissonDocumentPosterior& posterior) const;
    void sort_topics();
    void set_feature_dispersion(const std::vector<double>& tau);
    void clear_feature_dispersion();
    void set_training_calibration(const std::vector<double>& feature_counts,
        const std::vector<double>& feature_weights, bool weights_active);
    double restrict_features(const std::vector<int32_t>& kept_features);
    bool has_feature_dispersion() const { return has_dispersion_; }
    const VectorXd& get_topic_capacity() const { return topic_capacity_; }
    const MatrixXd& get_expected_beta() const { return e_beta_; }
    const MatrixXd& get_beta_allocation_kernel() const { return beta_kernel_; }
    const VectorXd& get_feature_dispersion() const { return tau_; }
    double get_theta_prior_shape() const { return theta_prior_shape(); }
    VectorXd get_theta_prior_rate() const {
        VectorXd out(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) out(k) = theta_prior_rate(k);
        return out;
    }
    void expected_observed_counts(const Document& doc, std::vector<double>& means) const;
    void write_state(const std::string& outFile, const std::vector<std::string>& featureNames);
    static std::vector<std::string> read_state_feature_names(const std::string& stateFile);
    static GammaPoissonStateFeatureInfo read_state_feature_info(
        const std::string& stateFile);

private:
    GammaPoissonTopicModel(const std::string& stateFile, int seed,
        int32_t nThreads, int32_t verbose, bool defer_cache);

    struct LocalWorkspace {
        MatrixXd beta_kernel;
        MatrixXd beta_mean;
        VectorXd theta_kernel;
        VectorXd theta_log;
        VectorXd last_shape;
        VectorXd assigned;
        VectorXd norm;
        VectorXd ratio;
        VectorXd e_theta;
        VectorXd epsilon;
    };

    struct WorkerState {
        MatrixXd ss;
        MatrixXd beta_rate_correction;
        VectorXd ctheta;
        VectorXd theta;
        VectorXd theta_shape;
        VectorXd theta_rate;
        LocalWorkspace workspace;
        uint64_t generation = 0;
        int64_t iteration_sum = 0;
        int32_t documents = 0;
        int32_t failed = 0;

        void reset(uint64_t current_generation, int32_t n_topics,
            int32_t n_features, bool with_dispersion);
    };

    template <bool WithDispersion>
    int32_t fit_one_document(VectorXd& theta_shape, VectorXd& theta_rate,
        LocalWorkspace& workspace, const Document& doc) const;
    int32_t fit_one_document(VectorXd& theta_shape, VectorXd& theta_rate,
        const Document& doc) const;
    template <bool WithDispersion>
    void accumulate_document(WorkerState& state, const Document& doc) const;
    template <bool WithDispersion>
    void partial_fit_impl(const std::vector<Document>& docs);
    void read_state(const std::string& stateFile);
    void apply_nu_cap();
    double theta_prior_shape() const {
        return theta_concentration_ / static_cast<double>(n_topics_);
    }
    double theta_prior_rate(int32_t k) const {
        return symmetric_nu_ ? theta_concentration_ : expected_nu(k);
    }
    double expected_nu(int32_t k) const {
        return nu_shape_(k) / nu_rate_(k);
    }

    bool symmetric_nu_ = true;
    bool has_dispersion_ = false;
    double theta_concentration_ = 1.0;
    double nu_max_ = -1.0;
    VectorXd nu_shape_;
    VectorXd nu_rate_;
    VectorXd tau_;
    std::unique_ptr<tbb::enumerable_thread_specific<WorkerState>> worker_states_;
    uint64_t worker_generation_ = 0;
};


class GammaPoisson4Hex : public TopicModelWrapper {
public:
    GammaPoisson4Hex(HexReader& _reader, int32_t modal = 0, int32_t verbose = 0)
        : TopicModelWrapper(_reader, modal, verbose) {}

    void initialize(int32_t nTopics, int32_t seed, int32_t nThreads, int32_t verbose,
        double beta_shape, double xi_shape, double xi_mean, double theta_concentration,
        double nu_shape, double nu_rate, double kappa, double tau0,
        int32_t totalDocCount, double sizeFactor, bool symmetricNu, double nuMax,
        int32_t maxIter, double mDelta, double randomInitShape = 0.5);
    void initializeFromModel(const std::string& modelFile);
    void setFeatureDispersion(const std::vector<double>& tau);
    void clearFeatureDispersion();
    GammaPoissonDispersionResult estimateFeatureDispersion(
        const GammaPoissonDispersionOptions& options, const std::string& inFile,
        int32_t batchSize, int32_t minCountTrain, int32_t maxUnits);
    GammaPoissonDispersionResult estimateFeatureDispersion(
        const GammaPoissonDispersionOptions& options,
        punkst::DocumentBlockSource& source,
        int32_t batchSize, int32_t maxUnits);
    GammaPoissonDispersionResult estimateFeatureDispersion(
        const GammaPoissonDispersionOptions& options,
        const std::vector<std::vector<Document>>& batches,
        int32_t maxUnits);
    GammaPoissonDispersionResult estimateFeatureDispersion10X(
        const GammaPoissonDispersionOptions& options, int32_t batchSize, int32_t maxUnits);
    GammaPoissonDispersionResult estimateFeatureDispersion10X(
        const GammaPoissonDispersionOptions& options, DGEReader10X& dge,
        int32_t batchSize, int32_t minCount, int32_t maxUnits);
    void initialize_transform(std::unique_ptr<GammaPoissonTopicModel> model,
        int32_t maxIter, double mDelta,
        const std::vector<int32_t>& keptFeatures = {});
    double resolveSizeFactor(double requested) const;
    void writeModelToFile(const std::string& outFile);
    void writeStateToFile(const std::string& outFile);
    int32_t getNumTopics() const override { return model_ ? model_->get_n_topics() : 0; }
    void sortTopicsByWeight() override { if (model_) model_->sort_topics(); }
    void getUnitHeaderCols(std::vector<std::string>& outCols) override;
    const RowMajorMatrixXd& get_model_matrix() const override;
    RowMajorMatrixXd copy_model_matrix() const override;
    const std::vector<std::string>& get_topic_names() override;
    void do_partial_fit(const std::vector<Document>& batch) override;
    MatrixXd do_transform(DocumentView batch) override;
    RowMajorMatrixXd transformMeans(DocumentView batch) const;
    void transformWithPosteriors(DocumentView batch, RowMajorMatrixXd& topics,
        std::vector<GammaPoissonDocumentPosterior>& posteriors) const;
    bool hasFeatureDispersion() const;
    const VectorXd& getTopicCapacity() const;
    const MatrixXd& getExpectedBeta() const;
    const MatrixXd& getBetaAllocationKernel() const;
    const VectorXd& getFeatureDispersion() const;
    double getThetaPriorShape() const;
    VectorXd getThetaPriorRate() const;
    bool featureWeightsActive() const;
    const std::vector<double>& getFeatureWeights() const;
    void getTopicAbundance(std::vector<double>& topic_weights) override;
    void get_topic_abundance(std::vector<double>& topic_weights);

private:
    std::unique_ptr<GammaPoissonTopicModel> model_;
    mutable RowMajorMatrixXd empty_model_;
    std::vector<std::string> topicNames_;
};
