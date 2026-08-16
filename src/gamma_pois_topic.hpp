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
#include "gamma_pois_common.hpp"
#include "gamma_pois_dispersion.hpp"
#include "numerical_utils.hpp"
#include "topic_svb.hpp"

enum class GammaPoissonInferenceMode {
    MapMean,
    LdaCompatible,
};

enum class GammaPoissonOwnershipMode {
    Uniform,
    Prevalence,
};

struct GammaPoissonMapOptions {
    // -1 selects V/K for LDA-compatible inference and 1 for map-mean.
    double dictionary_prior_mass = -1.0;
    double ownership_strength = 0.0;
    GammaPoissonOwnershipMode ownership_mode =
        GammaPoissonOwnershipMode::Uniform;
    GammaPoissonInferenceMode inference_mode =
        GammaPoissonInferenceMode::LdaCompatible;
};

struct GammaPoissonOptimizationDiagnostics {
    double objective = std::numeric_limits<double>::quiet_NaN();
    double ownership_entropy = std::numeric_limits<double>::quiet_NaN();
    double uniform_ownership_entropy =
        std::numeric_limits<double>::quiet_NaN();
    double effective_ownership_lambda = 0.0;
    double maximum_row_change = 0.0;
    int32_t accepted_gradient_steps = 0;
    int32_t failed_gradient_steps = 0;
    int32_t lbfgs_steps = 0;
    int32_t lbfgs_fallbacks = 0;
    int32_t mm_steps = 0;
    int32_t mm_fallbacks = 0;
    double relative_objective_gain = 0.0;
};

class GammaPoissonTopicBase {
public:
    GammaPoissonTopicBase() = default;
    GammaPoissonTopicBase(int32_t n_topics, int32_t n_features,
        int seed = std::random_device{}(), int32_t nThreads = 0, int32_t verbose = 0,
        double learning_decay = 0.7, double learning_offset = 10.0,
        int32_t total_doc_count = 1000000,
        const std::vector<double>* feature_sums = nullptr,
        double random_init_shape = 0.5,
        bool initialize_profiles = true);
    virtual ~GammaPoissonTopicBase() = default;

    int32_t get_n_topics() const { return n_topics_; }
    int32_t get_n_features() const { return n_features_; }
    double get_size_factor() const { return 1.0; }
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
    void get_topic_prevalence(std::vector<double>& weights) const;
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
    void init_from_feature_sums(const std::vector<double>* feature_sums,
        bool initialize_profiles = true);
    void refresh_cache();
    void refresh_model_cache();
    RowVectorXd normalized_theta_hat(const VectorXd& theta_shape, const VectorXd& theta_rate) const;
    double doc_exposure(const Document& doc) const;
    uint64_t doc_stream(uint64_t doc_index, uint64_t phase) const;

    struct SplitMix64Engine {
        using result_type = uint64_t;
        explicit SplitMix64Engine(uint64_t seed) : state(seed) {}
        static constexpr result_type min() { return 0; }
        static constexpr result_type max() {
            return std::numeric_limits<result_type>::max();
        }
        result_type operator()() { return ::splitmix64(state++); }
        uint64_t state;
    };

    int32_t n_topics_ = -1;
    int32_t n_features_ = -1;
    int seed_ = 1;
    int32_t nThreads_ = 1;
    int32_t verbose_ = 0;
    int32_t total_doc_count_ = 1000000;
    int32_t update_count_ = 0;
    double learning_decay_ = 0.7;
    double learning_offset_ = 10.0;
    double random_init_shape_ = 0.5;
    double eps_ = std::numeric_limits<double>::epsilon();
    int32_t max_doc_update_iter_ = 100;
    double mean_change_tol_ = 1e-3;

    MatrixXd e_beta_; // normalized MAP dictionary, K x V
    MatrixXd beta_kernel_; // centered beta by feature
    GammaPoissonInferenceMode inference_mode_ =
        GammaPoissonInferenceMode::MapMean;
    VectorXd topic_concentration_;
    RowMajorMatrixXd model_phi_;
    bool model_cache_dirty_ = true;
    bool inference_cache_ready_ = false;
    VectorXd topic_capacity_; // fixed to one for normalized dictionary rows
    // Smoothed sum_d c_d E[theta_d], used for corpus prevalence.
    VectorXd topic_exposure_;
    std::vector<std::string> topic_names_;
    std::vector<std::string> feature_names_;
    std::vector<double> training_count_;
    bool feature_weights_active_ = false;
    std::vector<double> feature_weight_;
    std::mt19937 random_engine_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;
};

class GammaPoissonTopicModel : public GammaPoissonTopicBase,
                               public GammaPoissonDispersionModel {
public:
    int32_t get_n_topics() const override {
        return GammaPoissonTopicBase::get_n_topics();
    }
    int32_t get_n_features() const override {
        return GammaPoissonTopicBase::get_n_features();
    }
    bool feature_weights_active() const override {
        return GammaPoissonTopicBase::feature_weights_active();
    }
    const std::vector<double>& get_feature_weight() const override {
        return GammaPoissonTopicBase::get_feature_weight();
    }

    GammaPoissonTopicModel(int32_t n_topics, int32_t n_features,
        int seed = std::random_device{}(), int32_t nThreads = 0, int32_t verbose = 0,
        double theta_concentration = 1.0,
        double learning_decay = 0.7, double learning_offset = 10.0,
        int32_t total_doc_count = 1000000,
        const std::vector<double>* feature_sums = nullptr,
        double random_init_shape = 0.5,
        const GammaPoissonMapOptions& map_options = {});

    explicit GammaPoissonTopicModel(const std::string& stateFile,
        int seed = std::random_device{}(), int32_t nThreads = 0, int32_t verbose = 0);
    static std::unique_ptr<GammaPoissonTopicModel> load_state_deferred(
        const std::string& stateFile, int seed = std::random_device{}(),
        int32_t nThreads = 0, int32_t verbose = 0);

    void partial_fit(const std::vector<Document>& docs);
    void set_ownership_annealing(double fraction);
    void configure_ownership_annealing(int32_t warmup_epochs,
        int32_t ramp_epochs, int32_t documents_per_epoch,
        int64_t documents_seen = 0);
    void reset_running_statistics();
    void begin_full_refinement();
    void accumulate_full_refinement(const std::vector<Document>& docs);
    bool finish_full_refinement(double tolerance);
    RowMajorMatrixXd transform(DocumentView docs);
    void transform_with_posteriors(DocumentView docs, RowMajorMatrixXd& topics,
        std::vector<GammaPoissonDocumentPosterior>& posteriors) const;
    void infer_document_posterior(const Document& doc,
        GammaPoissonDocumentPosterior& posterior) const override;
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
    const MatrixXd& get_expected_beta() const override { return e_beta_; }
    const MatrixXd& get_beta_allocation_kernel() const { return beta_kernel_; }
    void normalize_topic_allocation(
        const Eigen::Ref<const VectorXd>& theta_log,
        int32_t feature, VectorXd& allocation) const;
    GammaPoissonInferenceMode get_inference_mode() const {
        return inference_mode_;
    }
    GammaPoissonOwnershipMode get_ownership_mode() const {
        return ownership_mode_;
    }
    const VectorXd& get_topic_concentration() const {
        return topic_concentration_;
    }
    const VectorXd& get_feature_dispersion() const { return tau_; }
    double get_theta_prior_shape() const { return theta_prior_shape(); }
    VectorXd get_theta_prior_rate() const {
        VectorXd out(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) out(k) = theta_prior_rate(k);
        return out;
    }
    void expected_observed_counts(const Document& doc, std::vector<double>& means) const;
    double ownership_entropy() const;
    double map_objective() const;
    GammaPoissonOptimizationDiagnostics optimization_diagnostics() const;
    void write_state(const std::string& outFile, const std::vector<std::string>& featureNames);
    static std::vector<std::string> read_state_feature_names(const std::string& stateFile);
    static GammaPoissonStateFeatureInfo read_state_feature_info(
        const std::string& stateFile);

private:
    friend struct GammaPoissonTestAccess;
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
        MatrixXd dispersion_correction;
        VectorXd ctheta;
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
        LocalWorkspace& workspace, const Document& doc,
        uint64_t rng_stream = 0) const;
    int32_t fit_one_document(VectorXd& theta_shape, VectorXd& theta_rate,
        const Document& doc) const;
    template <bool WithDispersion>
    void accumulate_document(WorkerState& state, const Document& doc,
        uint64_t rng_stream) const;
    template <bool WithDispersion>
    void partial_fit_impl(const std::vector<Document>& docs);
    template <bool WithDispersion>
    void collect_batch_statistics(const std::vector<Document>& docs,
        MatrixXd& counts, MatrixXd& delta, VectorXd& exposure,
        int64_t& iteration_sum, int32_t& failed);
    bool optimize_dictionary(const MatrixXd& counts, const MatrixXd& delta,
        double ownership_fraction, int32_t max_steps, bool use_lbfgs,
        double movement_tolerance, double* max_row_change = nullptr);
    bool solve_unregularized_dictionary(const MatrixXd& counts,
        const MatrixXd& delta, MatrixXd& dictionary) const;
    VectorXd ownership_weights(const MatrixXd& counts) const;
    double ownership_entropy(const MatrixXd& dictionary,
        const VectorXd& weights) const;
    bool optimize_dictionary_mm(const MatrixXd& counts,
        const MatrixXd& delta, double ownership_fraction, int32_t max_steps,
        double movement_tolerance, double* max_row_change);
    double objective_and_gradient(const MatrixXd& counts, const MatrixXd& delta,
        double ownership_fraction, const MatrixXd& logits,
        MatrixXd* gradient, MatrixXd* dictionary = nullptr) const;
    static void logits_to_dictionary(const MatrixXd& logits, MatrixXd& dictionary);
    static void dictionary_to_logits(const MatrixXd& dictionary, MatrixXd& logits);
    void update_topic_concentration(const MatrixXd& counts);
    void read_state(const std::string& stateFile);
    double theta_prior_shape() const {
        return theta_concentration_ / static_cast<double>(n_topics_);
    }
    double theta_prior_rate(int32_t) const { return theta_concentration_; }

    bool has_dispersion_ = false;
    double theta_concentration_ = 1.0;
    double dictionary_prior_mass_ = 1.0;
    double ownership_strength_ = 0.0;
    GammaPoissonOwnershipMode ownership_mode_ =
        GammaPoissonOwnershipMode::Uniform;
    double ownership_fraction_ = 0.0;
    int32_t ownership_warmup_epochs_ = 1;
    int32_t ownership_ramp_epochs_ = 1;
    int32_t ownership_documents_per_epoch_ = 0;
    int64_t ownership_documents_seen_ = 0;
    bool ownership_schedule_active_ = false;
    bool running_stats_ready_ = false;
    MatrixXd running_counts_;
    MatrixXd running_delta_;
    MatrixXd refinement_counts_;
    MatrixXd refinement_delta_;
    VectorXd refinement_exposure_;
    bool refinement_active_ = false;
    int32_t accepted_gradient_steps_ = 0;
    int32_t failed_gradient_steps_ = 0;
    int32_t lbfgs_steps_ = 0;
    int32_t lbfgs_fallbacks_ = 0;
    int32_t mm_steps_ = 0;
    int32_t mm_fallbacks_ = 0;
    double last_relative_objective_gain_ = 0.0;
    double last_maximum_row_change_ = 0.0;
    VectorXd tau_;
    std::unique_ptr<tbb::enumerable_thread_specific<WorkerState>> worker_states_;
    uint64_t worker_generation_ = 0;
};


class GammaPoisson4Hex : public GammaPoisson4HexInterface {
public:
    GammaPoisson4Hex(HexReader& _reader, int32_t modal = 0, int32_t verbose = 0)
        : GammaPoisson4HexInterface(_reader, modal, verbose) {}

    void initialize(int32_t nTopics, int32_t seed, int32_t nThreads, int32_t verbose,
        double theta_concentration, double kappa, double tau0,
        int32_t totalDocCount, int32_t maxIter, double mDelta,
        double randomInitShape = 0.5,
        const GammaPoissonMapOptions& mapOptions = {});
    void initializeFromModel(const std::string& modelFile);
    void initializeFromState(const std::string& stateFile, int32_t seed,
        int32_t nThreads, int32_t verbose, int32_t maxIter, double mDelta);
    void setFeatureDispersion(const std::vector<double>& tau) override;
    void clearFeatureDispersion() override;
    GammaPoissonDispersionResult estimateFeatureDispersion(
        const GammaPoissonDispersionOptions& options, const std::string& inFile,
        int32_t batchSize, int32_t minCountTrain, int32_t maxUnits) override;
    GammaPoissonDispersionResult estimateFeatureDispersion(
        const GammaPoissonDispersionOptions& options,
        punkst::DocumentBlockSource& source,
        int32_t batchSize, int32_t maxUnits) override;
    GammaPoissonDispersionResult estimateFeatureDispersion(
        const GammaPoissonDispersionOptions& options,
        const std::vector<std::vector<Document>>& batches,
        int32_t maxUnits) override;
    GammaPoissonDispersionResult estimateFeatureDispersion10X(
        const GammaPoissonDispersionOptions& options, int32_t batchSize,
        int32_t maxUnits) override;
    GammaPoissonDispersionResult estimateFeatureDispersion10X(
        const GammaPoissonDispersionOptions& options, DGEReader10X& dge,
        int32_t batchSize, int32_t minCount, int32_t maxUnits) override;
    void initialize_transform(std::unique_ptr<GammaPoissonTopicModel> model,
        int32_t maxIter, double mDelta,
        const std::vector<int32_t>& keptFeatures = {});
    void setOwnershipAnnealing(double fraction);
    void configureOwnershipAnnealing(int32_t warmupEpochs,
        int32_t rampEpochs, int32_t documentsPerEpoch,
        int64_t documentsSeen = 0);
    void resetRunningStatistics();
    void beginFullRefinement();
    void accumulateFullRefinement(const std::vector<Document>& batch);
    bool finishFullRefinement(double tolerance);
    GammaPoissonOptimizationDiagnostics optimizationDiagnostics() const;
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
    RowMajorMatrixXd transformMeans(DocumentView batch) const override;
    void transformWithPosteriors(DocumentView batch, RowMajorMatrixXd& topics,
        std::vector<GammaPoissonDocumentPosterior>& posteriors) const override;
    bool hasFeatureDispersion() const override;
    const VectorXd& getTopicCapacity() const override;
    const MatrixXd& getExpectedBeta() const override;
    const MatrixXd& getBetaAllocationKernel() const override;
    void normalizeTopicAllocation(
        const Eigen::Ref<const VectorXd>& theta_log,
        int32_t feature, VectorXd& allocation) const override;
    const VectorXd& getFeatureDispersion() const override;
    double getSizeFactor() const override;
    double getThetaPriorShape() const override;
    VectorXd getThetaPriorRate() const override;
    bool featureWeightsActive() const override;
    const std::vector<double>& getFeatureWeights() const override;
    void getTopicAbundance(std::vector<double>& topic_weights) override;
    void get_topic_abundance(std::vector<double>& topic_weights) override;

private:
    std::unique_ptr<GammaPoissonTopicModel> model_;
    mutable RowMajorMatrixXd empty_model_;
    std::vector<std::string> topicNames_;
    bool refining_ = false;
};
