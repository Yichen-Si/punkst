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
#include <tbb/combinable.h>
#include <tbb/global_control.h>
#include <tbb/parallel_for.h>

#include "dataunits.hpp"
#include "error.hpp"
#include "numerical_utils.hpp"
#include "topic_svb.hpp"

class GammaPoissonTopicModel {
public:
    GammaPoissonTopicModel(int32_t n_topics, int32_t n_features,
        int seed = std::random_device{}(), int32_t nThreads = 0, int32_t verbose = 0,
        double beta_shape = 0.3, double xi_shape = 0.3, double xi_mean = 1.0,
        double theta_shape = 1.0, double nu_shape = 1.0, double nu_rate = -1.0,
        double learning_decay = 0.7, double learning_offset = 10.0,
        int32_t total_doc_count = 1000000, double size_factor = 1.0,
        bool symmetric_nu = false, const std::vector<double>* feature_sums = nullptr);

    explicit GammaPoissonTopicModel(const std::string& stateFile,
        int seed = std::random_device{}(), int32_t nThreads = 0, int32_t verbose = 0);

    int32_t get_n_topics() const { return n_topics_; }
    int32_t get_n_features() const { return n_features_; }
    double get_size_factor() const { return size_factor_; }
    const std::vector<std::string>& get_topic_names();
    const std::vector<std::string>& get_feature_names() const { return feature_names_; }
    const RowMajorMatrixXd& get_model();
    RowMajorMatrixXd copy_model();
    void get_topic_abundance(std::vector<double>& weights) const;
    void sort_topics();
    void set_svb_parameters(int32_t max_iter, double tol);
    void set_nthreads(int32_t nThreads);
    void partial_fit(const std::vector<Document>& docs);
    RowMajorMatrixXd transform(DocumentView docs);
    void write_model(const std::string& outFile, const std::vector<std::string>& featureNames);
    void write_state(const std::string& outFile, const std::vector<std::string>& featureNames);
    static std::vector<std::string> read_state_feature_names(const std::string& stateFile);

private:
    static int normalize_seed(int seed);
    void init_from_feature_sums(const std::vector<double>* feature_sums);
    void refresh_cache();
    int32_t fit_one_document(VectorXd& theta_shape, VectorXd& theta_rate,
        VectorXd& elog_theta, const Document& doc) const;
    RowVectorXd normalized_theta_hat(const VectorXd& theta_shape, const VectorXd& theta_rate) const;
    double doc_exposure(const Document& doc) const;
    void read_state(const std::string& stateFile);
    double expected_xi(int32_t w) const { return xi_shape_(w) / xi_rate_(w); }
    double expected_nu(int32_t k) const {
        return symmetric_nu_ ? 1.0 : nu_shape_(k) / nu_rate_(k);
    }

    int32_t n_topics_ = -1;
    int32_t n_features_ = -1;
    int seed_ = 1;
    int32_t nThreads_ = 1;
    int32_t verbose_ = 0;
    int32_t total_doc_count_ = 1000000;
    int32_t update_count_ = 0;
    bool symmetric_nu_ = false;
    double a_ = 0.3;
    double a0_ = 0.3;
    double b0_ = 1.0;
    double s0_ = 1.0;
    double e0_ = 1.0;
    double f0_ = 1.0;
    double learning_decay_ = 0.7;
    double learning_offset_ = 10.0;
    double size_factor_ = 1.0;
    double eps_ = std::numeric_limits<double>::epsilon();
    int32_t max_doc_update_iter_ = 100;
    double mean_change_tol_ = 1e-3;

    RowMajorMatrixXd beta_shape_;
    RowMajorMatrixXd beta_rate_;
    RowMajorMatrixXd e_beta_;
    RowMajorMatrixXd elog_beta_;
    RowMajorMatrixXd model_phi_;
    VectorXd topic_capacity_;
    VectorXd xi_shape_;
    VectorXd xi_rate_;
    VectorXd nu_shape_;
    VectorXd nu_rate_;
    VectorXd topic_usage_;
    std::vector<std::string> topic_names_;
    std::vector<std::string> feature_names_;
    std::mt19937 random_engine_;
    std::unique_ptr<tbb::global_control> tbb_ctrl_;
};

class GammaPoisson4Hex : public TopicModelWrapper {
public:
    GammaPoisson4Hex(HexReader& _reader, int32_t modal = 0, int32_t verbose = 0)
        : TopicModelWrapper(_reader, modal, verbose) {}

    void initialize(int32_t nTopics, int32_t seed, int32_t nThreads, int32_t verbose,
        double beta_shape, double xi_shape, double xi_mean, double theta_shape,
        double nu_shape, double nu_rate, double kappa, double tau0,
        int32_t totalDocCount, double sizeFactor, bool symmetricNu,
        int32_t maxIter, double mDelta);
    void initialize_transform(const std::string& stateFile, int32_t seed,
        int32_t nThreads, int32_t verbose, int32_t maxIter, double mDelta);
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
    void getTopicAbundance(std::vector<double>& topic_weights) override;
    void get_topic_abundance(std::vector<double>& topic_weights);

private:
    std::unique_ptr<GammaPoissonTopicModel> model_;
    mutable RowMajorMatrixXd empty_model_;
    std::vector<std::string> topicNames_;
};
