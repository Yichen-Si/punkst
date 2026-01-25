#include "lda.hpp"

void LatentDirichletAllocation::compute_global_mtx() {
    if (algo_ == InferenceType::SCVB0) {
        Nk_ = components_.rowwise().sum();
    } else {
        exp_Elog_beta_ = dirichlet_expectation_2d(components_);
    }
}

void LatentDirichletAllocation::sort_topics() {
    // sort topics by decreasing total weight
    VectorXd topic_weights = components_.rowwise().sum();
    std::vector<int> indices(n_topics_);
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(),
                [&topic_weights](int a, int b) {
        return topic_weights[a] > topic_weights[b];
    });
    // update components_, exp_Elog_beta_, or Nk_
    MatrixXd sorted_components(n_topics_, n_features_);
    for (int i = 0; i < n_topics_; i++) {
        sorted_components.row(i) = components_.row(indices[i]);
    }
    components_ = std::move(sorted_components);
    compute_global_mtx();
    // topic_names_
    if (!topic_names_.empty()) {
        std::vector<std::string> sorted_topic_names(n_topics_);
        for (int i = 0; i < n_topics_; i++) {
            sorted_topic_names[i] = topic_names_[indices[i]];
        }
        topic_names_ = std::move(sorted_topic_names);
    }
}

void LatentDirichletAllocation::set_model_from_matrix(std::vector<std::vector<double>>& lambdaVals) {
    if (lambdaVals.size() != n_topics_ || lambdaVals[0].size() != n_features_) {
        if (n_topics_ > 0 && n_features_ > 0)
            warning("Model matrix size mismatch, reset according to the provided global parameters. (%d x %d) -> (%d x %d)", n_topics_, n_features_, lambdaVals.size(), lambdaVals[0].size());
        n_topics_ = lambdaVals.size();
        n_features_ = lambdaVals[0].size();
    }
    notice("Global variational parameters are reset, but online training status (if any) is not. It is only safe for transform.");
    components_.resize(n_topics_, n_features_);
    for (int i = 0; i < n_topics_; i++) {
        for (int j = 0; j < n_features_; j++) {
            components_(i, j) = lambdaVals[i][j];
        }
    }
    compute_global_mtx();
}

void LatentDirichletAllocation::set_model_from_matrix(RowMajorMatrixXd& lambda) {
    if (lambda.rows() != n_topics_ || lambda.cols() != n_features_) {
        if (n_topics_ > 0 && n_features_ > 0)
            warning("Model matrix size mismatch, reset according to the provided global parameters. (%d x %d) -> (%d x %d)", n_topics_, n_features_, lambda.rows(), lambda.cols());
        n_topics_ = lambda.rows();
        n_features_ = lambda.cols();
    }
    notice("Global variational parameters are reset, but online training status (if any) is not. It is only safe for transform.");
    components_ = std::move(lambda);
    compute_global_mtx();
}

void LatentDirichletAllocation::set_model_from_tsv(const std::string& modelFile, double scalar) {
    std::ifstream modelIn(modelFile, std::ios::in);
    if (!modelIn) {
        error("Error opening model file: %s", modelFile.c_str());
    }
    std::string line;
    std::vector<std::string> tokens;
    std::getline(modelIn, line);
    split(tokens, "\t", line);
    int32_t K = tokens.size() - 1;
    feature_names_.clear();
    topic_names_.resize(K);
    for (int32_t i = 0; i < K; ++i) {
        topic_names_[i] = tokens[i + 1];
    }
    std::vector<std::vector<double>> modelValues;
    while (std::getline(modelIn, line)) {
        split(tokens, "\t", line);
        if (tokens.size() != K + 1) {
            error("Error reading model file at line ", line.c_str());
        }
        feature_names_.push_back(tokens[0]);
        std::vector<double> values(K);
        for (int32_t i = 0; i < K; ++i) {
            values[i] = std::stod(tokens[i + 1]);
        }
        modelValues.push_back(values);
    }
    modelIn.close();

    n_topics_ = K;
    n_features_ = feature_names_.size();
    notice("Read %d topics and %d features from model file", n_topics_, n_features_);

    components_.resize(n_topics_, n_features_);
    for (uint32_t i = 0; i < n_features_; ++i) {
        for (int32_t j = 0; j < K; ++j) {
            components_(j, i) = modelValues[i][j];
        }
    }
    if (scalar > 0.) {
        components_ *= scalar;
    }
    compute_global_mtx();
}

void LatentDirichletAllocation::init_model(const std::optional<RowMajorMatrixXd>& topic_word_distr, double scalar) {
    if (topic_word_distr && topic_word_distr->rows() > 0
                            && topic_word_distr->cols() > 0) {
        components_ = *topic_word_distr;
        n_topics_ = components_.rows();
        n_features_ = components_.cols();
        if (scalar > 0.) {
            components_ *= scalar;
        }
    } else {
        components_.resize(n_topics_, n_features_);
        std::gamma_distribution<double> gamma_dist(100.0, 0.01);
        for (int k = 0; k < n_topics_; k++) {
            for (int j = 0; j < n_features_; j++) {
                components_(k, j) = gamma_dist(random_engine_);
            }
        }
    }
    compute_global_mtx();
}

void LatentDirichletAllocation::init() {
    set_nthreads(nThreads_);
    if (seed_ <= 0) {
        seed_ = std::random_device{}();
    }
    random_engine_.seed(seed_);
    rng_stream_.store(0, std::memory_order_relaxed);
    eps_ = std::numeric_limits<double>::epsilon();
    if (alpha_ < 0) {
        alpha_ = 1. / n_topics_;
    }
    if (eta_ < 0) {
        eta_ = 1. / n_topics_;
    }
    if (total_doc_count_ <= 0) {
        total_doc_count_ = 1000000;
    }
    if (learning_decay_ <= 0) {
        learning_decay_ = 0.7; // kappa
    }
    if (learning_offset_ < 0) {
        learning_offset_ = 10.0; // tau
    }
    if (algo_ == InferenceType::SCVB0) {
        if (s_beta_ > std::pow(learning_offset_ + 1, learning_decay_) ) {
            s_beta_ = 1;
        }
    } else {
        if (mean_change_tol_ <= 0) {
            mean_change_tol_ = 0.001;
        }
        if (max_doc_update_iter_ <= 0) {
            max_doc_update_iter_ = 100;
        }
    }
    if (verbose_) {
        notice("LDA initialized with %d topics, %d features, %d threads", n_topics_, n_features_, nThreads_);
    }
}

std::mt19937& LatentDirichletAllocation::thread_rng() {
    auto& rngs = thread_rng_map();
    auto it = rngs.find(this);
    if (it == rngs.end()) {
        uint64_t stream = rng_stream_.fetch_add(1, std::memory_order_relaxed);
        auto rng = make_rng(static_cast<uint64_t>(seed_), stream);
        it = rngs.emplace(this, RngState{std::move(rng), stream}).first;
    }
    return it->second.rng;
}

void LatentDirichletAllocation::set_thread_rng_stream(uint64_t stream) {
    auto& rngs = thread_rng_map();
    auto it = rngs.find(this);
    if (it != rngs.end() && it->second.stream == stream) {
        return;
    }
    auto rng = make_rng(static_cast<uint64_t>(seed_), stream);
    if (it == rngs.end()) {
        rngs.emplace(this, RngState{std::move(rng), stream});
    } else {
        it->second.rng = std::move(rng);
        it->second.stream = stream;
    }
}

const std::vector<std::string>& LatentDirichletAllocation::get_topic_names() {
    if (topic_names_.empty()) {
        topic_names_.resize(n_topics_);
        for (int i = 0; i < n_topics_; i++) {
            topic_names_[i] = std::to_string(i);
        }
    }
    return topic_names_;
}

void LatentDirichletAllocation::get_topic_abundance(std::vector<double>& weights) const {
    weights.resize(n_topics_);
    for (int k = 0; k < n_topics_; k++) {
        weights[k] = components_.row(k).sum();
    }
    double total = std::accumulate(weights.begin(), weights.end(), 0.0);
    for (int k = 0; k < n_topics_; k++) {
        weights[k] /= total;
    }
}

void LatentDirichletAllocation::set_nthreads(int nThreads) {
    nThreads_ = nThreads;
    if (nThreads_ > 0) {
        tbb_ctrl_ = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism,
            std::size_t(nThreads_));
    } else {
        tbb_ctrl_.reset();
    }
    nThreads_ = int( tbb::global_control::active_value(
             tbb::global_control::max_allowed_parallelism) );
    notice("LatentDirichletAllocation: Requested %d threads, actual number of threads: %d", nThreads, nThreads_);
}
