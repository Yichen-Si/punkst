#include "gamma_pois_topic.hpp"
#include "lbfgs_history.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <unordered_map>
#include <unordered_set>

namespace {

double doc_sum_const(const Document& doc) {
    if (doc.ct_tot >= 0.0) {
        return doc.ct_tot;
    }
    return std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
}

std::vector<std::string> split_ws(const std::string& line) {
    std::vector<std::string> out;
    split(out, "\t ", line);
    return out;
}

int32_t read_gamma_pois_state_version(std::ifstream& in,
    const std::string& state_file) {
    std::string version;
    if (!std::getline(in, version)) {
        error("Gamma-Poisson state %s is empty", state_file.c_str());
    }
    if (version == "#punkst_gamma_pois_state_v8") return 8;
    {
        error("Gamma-Poisson state %s uses an unsupported format; refit the model "
            "with this version of punkst", state_file.c_str());
    }
    return -1;
}

const char* inference_mode_name(GammaPoissonInferenceMode mode) {
    return mode == GammaPoissonInferenceMode::LdaCompatible
        ? "lda-compatible" : "map-mean";
}

GammaPoissonInferenceMode parse_inference_mode(const std::string& value,
    const std::string& state_file) {
    if (value == "map-mean") return GammaPoissonInferenceMode::MapMean;
    if (value == "lda-compatible") {
        return GammaPoissonInferenceMode::LdaCompatible;
    }
    error("Invalid Gamma-Poisson inference mode %s in %s",
        value.c_str(), state_file.c_str());
    return GammaPoissonInferenceMode::MapMean;
}

const char* ownership_mode_name(GammaPoissonOwnershipMode mode) {
    return mode == GammaPoissonOwnershipMode::Prevalence
        ? "prevalence" : "uniform";
}

GammaPoissonOwnershipMode parse_ownership_mode(const std::string& value,
    const std::string& state_file) {
    if (value == "uniform") return GammaPoissonOwnershipMode::Uniform;
    if (value == "prevalence") return GammaPoissonOwnershipMode::Prevalence;
    error("Invalid Gamma-Poisson ownership mode %s in %s",
        value.c_str(), state_file.c_str());
    return GammaPoissonOwnershipMode::Uniform;
}

} // namespace

int GammaPoissonTopicBase::normalize_seed(int seed) {
    return seed > 0 ? seed : static_cast<int>(std::random_device{}());
}

GammaPoissonTopicBase::GammaPoissonTopicBase(int32_t n_topics, int32_t n_features,
    int seed, int32_t nThreads, int32_t verbose,
    double learning_decay, double learning_offset, int32_t total_doc_count,
    const std::vector<double>* feature_sums,
    double random_init_shape, bool initialize_profiles)
    : n_topics_(n_topics), n_features_(n_features), seed_(normalize_seed(seed)),
      nThreads_(nThreads), verbose_(verbose),
      total_doc_count_(total_doc_count),
      learning_decay_(learning_decay),
      learning_offset_(learning_offset),
      random_init_shape_(random_init_shape) {
    if (total_doc_count_ <= 0) {
        throw std::invalid_argument(
            "Gamma-Poisson total document count must be positive");
    }
    if (!std::isfinite(learning_decay_) || learning_decay_ <= 0.5
        || learning_decay_ > 1.0 || !std::isfinite(learning_offset_)
        || learning_offset_ <= 0.0) {
        throw std::invalid_argument(
            "Gamma-Poisson learning decay must be in (0.5,1] and offset positive");
    }
    if (!std::isfinite(random_init_shape_) || random_init_shape_ <= 0.0) {
        throw std::invalid_argument(
            "Gamma-Poisson random initialization shape must be positive and finite");
    }
    random_engine_.seed(seed_);
    set_nthreads(nThreads_);
    init_from_feature_sums(feature_sums, initialize_profiles);
}

GammaPoissonTopicModel::GammaPoissonTopicModel(int32_t n_topics, int32_t n_features,
    int seed, int32_t nThreads, int32_t verbose, double theta_concentration,
    double learning_decay, double learning_offset, int32_t total_doc_count,
    const std::vector<double>* feature_sums, double random_init_shape,
    const GammaPoissonMapOptions& map_options)
    : GammaPoissonTopicBase(n_topics, n_features, seed, nThreads, verbose,
          learning_decay, learning_offset, total_doc_count, feature_sums,
          random_init_shape,
          // The LDA initializer below owns e_beta_ and its cache; do not
          // construct and immediately discard the legacy IPF profiles.
          map_options.inference_mode != GammaPoissonInferenceMode::LdaCompatible),
      theta_concentration_(theta_concentration),
      dictionary_prior_mass_(map_options.dictionary_prior_mass == -1.0
              ? (map_options.inference_mode
                        == GammaPoissonInferenceMode::LdaCompatible
                    ? static_cast<double>(n_features)
                        / static_cast<double>(n_topics)
                    : 1.0)
              : map_options.dictionary_prior_mass),
      ownership_strength_(map_options.ownership_strength),
      ownership_mode_(map_options.ownership_mode) {
    inference_mode_ = map_options.inference_mode;
    if (!std::isfinite(theta_concentration_) || theta_concentration_ <= 0.0) {
        throw std::invalid_argument(
            "Gamma-Poisson theta concentration must be positive and finite");
    }
    if (!std::isfinite(map_options.dictionary_prior_mass)
        || (map_options.dictionary_prior_mass < 0.0
            && map_options.dictionary_prior_mass != -1.0)
        || !std::isfinite(dictionary_prior_mass_) || dictionary_prior_mass_ < 0.0
        || !std::isfinite(ownership_strength_) || ownership_strength_ < 0.0) {
        throw std::invalid_argument(
            "Gamma-Poisson MAP prior mass and ownership strength must be non-negative and finite");
    }
    if (inference_mode_ == GammaPoissonInferenceMode::LdaCompatible) {
        const double eta = dictionary_prior_mass_
            / static_cast<double>(n_features_);
        std::gamma_distribution<double> gamma_dist(100.0, 0.01);
        for (int32_t k = 0; k < n_topics_; ++k) {
            for (int32_t w = 0; w < n_features_; ++w) {
                e_beta_(k, w) = gamma_dist(random_engine_);
            }
        }
        topic_concentration_ = e_beta_.rowwise().sum();
        running_counts_ = e_beta_.array() - eta;
        for (int32_t k = 0; k < n_topics_; ++k) {
            e_beta_.row(k) /= topic_concentration_(k);
        }
        if ((running_counts_.array() + eta <= 0.0).any()) {
            throw std::runtime_error(
                "LDA-compatible initialization produced invalid Dirichlet parameters");
        }
        running_stats_ready_ = true;
        refresh_cache();
    }
}

void GammaPoissonTopicModel::set_feature_dispersion(const std::vector<double>& tau) {
    if (static_cast<int32_t>(tau.size()) != n_features_) {
        error("%s: dispersion vector has %zu values but model has %d features",
            __func__, tau.size(), n_features_);
    }
    tau_.resize(n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        if (!std::isfinite(tau[w]) || tau[w] <= 0.0) {
            error("%s: dispersion for feature %d must be positive and finite", __func__, w);
        }
        tau_(w) = tau[w];
    }
    has_dispersion_ = true;
    reset_running_statistics();
}

void GammaPoissonTopicModel::clear_feature_dispersion() {
    has_dispersion_ = false;
    tau_.resize(0);
    reset_running_statistics();
}

void GammaPoissonTopicModel::set_training_calibration(
    const std::vector<double>& feature_counts,
    const std::vector<double>& feature_weights, bool weights_active) {
    if (feature_counts.size() != static_cast<size_t>(n_features_)
        || (weights_active
            && feature_weights.size() != static_cast<size_t>(n_features_))) {
        error("%s: calibration vectors do not match the model feature dimension",
            __func__);
    }
    double effective_total = 0.0;
    for (int32_t w = 0; w < n_features_; ++w) {
        const double count = feature_counts[w];
        const double weight = weights_active ? feature_weights[w] : 1.0;
        if (!std::isfinite(count) || count < 0.0
            || !std::isfinite(weight) || weight < 0.0) {
            error("%s: calibration counts and weights must be non-negative and finite",
                __func__);
        }
        effective_total += count * weight;
    }
    if (!std::isfinite(effective_total) || effective_total <= 0.0) {
        error("%s: effective training feature total must be positive", __func__);
    }
    training_count_ = feature_counts;
    feature_weights_active_ = weights_active;
    feature_weight_ = weights_active ? feature_weights : std::vector<double>();
}

double GammaPoissonTopicModel::restrict_features(
    const std::vector<int32_t>& kept_features) {
    if (kept_features.empty()) {
        error("%s: cannot construct an empty Gamma-Poisson feature panel", __func__);
    }
    bool identity = static_cast<int32_t>(kept_features.size()) == n_features_;
    int32_t previous = -1;
    for (size_t j = 0; j < kept_features.size(); ++j) {
        const int32_t w = kept_features[j];
        if (w < 0 || w >= n_features_ || w <= previous) {
            error("%s: kept feature indices must be unique, ordered, and in range",
                __func__);
        }
        identity = identity && w == static_cast<int32_t>(j);
        previous = w;
    }
    if (identity) {
        prepare_inference_cache();
        return 1.0;
    }
    if (training_count_.size() != static_cast<size_t>(n_features_)
        || (feature_weights_active_
            && feature_weight_.size() != static_cast<size_t>(n_features_))) {
        error("%s: model state lacks feature-panel calibration", __func__);
    }

    double full_total = 0.0;
    double panel_total = 0.0;
    for (int32_t w = 0; w < n_features_; ++w) {
        const double weight =
            feature_weights_active_ ? feature_weight_[w] : 1.0;
        full_total += training_count_[w] * weight;
    }
    for (int32_t w : kept_features) {
        const double weight =
            feature_weights_active_ ? feature_weight_[w] : 1.0;
        panel_total += training_count_[w] * weight;
    }
    const double panel_fraction = panel_total / full_total;
    if (!std::isfinite(panel_fraction) || panel_fraction <= 0.0) {
        error("%s: selected features have zero training abundance", __func__);
    }

    const int32_t old_features = n_features_;
    const int32_t new_features = static_cast<int32_t>(kept_features.size());
    MatrixXd dictionary(n_topics_, new_features);
    VectorXd tau;
    if (has_dispersion_) tau.resize(new_features);
    std::vector<std::string> names;
    std::vector<double> counts(new_features);
    std::vector<double> weights;
    if (feature_weights_active_) weights.resize(new_features);
    names.reserve(new_features);
    for (int32_t j = 0; j < new_features; ++j) {
        const int32_t w = kept_features[j];
        dictionary.col(j) = e_beta_.col(w);
        if (has_dispersion_) tau(j) = tau_(w);
        if (feature_names_.size() == static_cast<size_t>(old_features)) {
            names.push_back(feature_names_[w]);
        }
        counts[j] = training_count_[w];
        if (feature_weights_active_) weights[j] = feature_weight_[w];
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        const double row_sum = dictionary.row(k).sum();
        if (!std::isfinite(row_sum) || row_sum <= 0.0) {
            error("%s: selected feature panel has zero mass for topic %d", __func__, k);
        }
        dictionary.row(k) /= row_sum;
        if (inference_mode_ == GammaPoissonInferenceMode::LdaCompatible
            && topic_concentration_.size() == n_topics_) {
            topic_concentration_(k) *= row_sum;
        }
    }
    e_beta_ = std::move(dictionary);
    if (has_dispersion_) tau_ = std::move(tau);
    feature_names_ = std::move(names);
    training_count_ = std::move(counts);
    feature_weight_ = std::move(weights);
    n_features_ = new_features;
    worker_states_.reset();
    reset_running_statistics();
    refresh_cache();
    return panel_fraction;
}

GammaPoissonTopicModel::GammaPoissonTopicModel(const std::string& stateFile,
    int seed, int32_t nThreads, int32_t verbose)
    : GammaPoissonTopicModel(stateFile, seed, nThreads, verbose, false) {}

GammaPoissonTopicModel::GammaPoissonTopicModel(const std::string& stateFile,
    int seed, int32_t nThreads, int32_t verbose, bool defer_cache)
    : GammaPoissonTopicBase() {
    seed_ = normalize_seed(seed);
    nThreads_ = nThreads;
    verbose_ = verbose;
    random_engine_.seed(seed_);
    read_state(stateFile);
    set_nthreads(nThreads_);
    if (!defer_cache) {
        refresh_cache();
    }
}

std::unique_ptr<GammaPoissonTopicModel>
GammaPoissonTopicModel::load_state_deferred(const std::string& stateFile,
    int seed, int32_t nThreads, int32_t verbose) {
    return std::unique_ptr<GammaPoissonTopicModel>(
        new GammaPoissonTopicModel(
            stateFile, seed, nThreads, verbose, true));
}

void GammaPoissonTopicBase::set_nthreads(int32_t nThreads) {
    nThreads_ = nThreads;
    if (nThreads_ > 0) {
        tbb_ctrl_ = std::make_unique<tbb::global_control>(
            tbb::global_control::max_allowed_parallelism,
            std::size_t(nThreads_));
    } else {
        tbb_ctrl_.reset();
    }
    nThreads_ = int(tbb::global_control::active_value(
        tbb::global_control::max_allowed_parallelism));
    notice("GammaPoissonTopicModel: Requested %d threads, actual number of threads: %d",
        nThreads, nThreads_);
}

void GammaPoissonTopicBase::set_svb_parameters(int32_t max_iter, double tol) {
    max_doc_update_iter_ = max_iter > 0 ? max_iter : 100;
    mean_change_tol_ = tol > 0.0 ? tol : 1e-3;
}

void GammaPoissonTopicBase::init_from_feature_sums(
    const std::vector<double>* feature_sums, bool initialize_profiles) {
    if (n_topics_ <= 0 || n_features_ <= 0) {
        throw std::invalid_argument("Gamma-Poisson model dimensions must be positive");
    }
    if (feature_sums && static_cast<int32_t>(feature_sums->size()) != n_features_) {
        throw std::invalid_argument(
            "Gamma-Poisson feature sums do not match the feature dimension");
    }
    e_beta_.resize(n_topics_, n_features_);
    topic_exposure_ = VectorXd::Constant(n_topics_,
        static_cast<double>(total_doc_count_)
            / static_cast<double>(std::max(1, n_topics_)));

    double total = 0.0;
    if (feature_sums) {
        for (double value : *feature_sums) {
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "Gamma-Poisson feature sums must be non-negative and finite");
            }
            total += value;
        }
    }
    training_count_.resize(n_features_);
    feature_weights_active_ = false;
    feature_weight_.clear();
    for (int32_t w = 0; w < n_features_; ++w) {
        training_count_[w] = feature_sums && total > 0.0
            ? (*feature_sums)[w]
            : 1.0 / static_cast<double>(n_features_);
    }
    if (!initialize_profiles) {
        return;
    }

    VectorXd feature_mean(n_features_);
    const double abundance_floor = 1e-12 / static_cast<double>(n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        const double abundance = feature_sums && total > 0.0
            ? (*feature_sums)[w] / total
            : 1.0 / static_cast<double>(n_features_);
        feature_mean(w) = std::max(abundance, abundance_floor);
    }
    feature_mean /= feature_mean.sum();

    std::gamma_distribution<double> noise(
        random_init_shape_, 1.0 / random_init_shape_);
    for (int32_t k = 0; k < n_topics_; ++k) {
        for (int32_t w = 0; w < n_features_; ++w) {
            e_beta_(k, w) = std::max(noise(random_engine_), 1e-300);
        }
    }

    VectorXd row_sums(n_topics_);
    bool converged = false;
    for (int32_t iter = 0; iter < 100; ++iter) {
        row_sums = e_beta_.rowwise().sum();
        for (int32_t k = 0; k < n_topics_; ++k) {
            if (!std::isfinite(row_sums(k)) || row_sums(k) <= 0.0) {
                throw std::runtime_error("Non-finite Gamma-Poisson beta initialization");
            }
            e_beta_.row(k) /= row_sums(k);
        }
        for (int32_t w = 0; w < n_features_; ++w) {
            const double column_sum = e_beta_.col(w).sum();
            if (!std::isfinite(column_sum) || column_sum <= 0.0) {
                throw std::runtime_error("Non-finite Gamma-Poisson beta initialization");
            }
            e_beta_.col(w) *= n_topics_ * feature_mean(w) / column_sum;
        }
        row_sums = e_beta_.rowwise().sum();
        double max_relative_error = 0.0;
        for (int32_t k = 0; k < n_topics_; ++k) {
            max_relative_error = std::max(max_relative_error,
                std::abs(row_sums(k) - 1.0));
        }
        if (std::isfinite(max_relative_error) && max_relative_error <= 1e-8) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        throw std::runtime_error("Gamma-Poisson beta initialization did not converge");
    }

    refresh_cache();
}

void GammaPoissonTopicBase::initialize_topic_profiles(
    const Eigen::Ref<const RowMajorMatrixXd>& profiles,
    const std::vector<std::string>& topic_names) {
    if (profiles.rows() != n_topics_ || profiles.cols() != n_features_) {
        throw std::invalid_argument(
            "Gamma-Poisson initial model dimensions do not match the fitted model");
    }
    if (!topic_names.empty()
        && static_cast<int32_t>(topic_names.size()) != n_topics_) {
        throw std::invalid_argument(
            "Gamma-Poisson initial model topic names do not match the topic count");
    }

    constexpr double kProfileFloor = 1e-12;
    RowMajorMatrixXd normalized = profiles;
    for (int32_t k = 0; k < n_topics_; ++k) {
        double total = 0.0;
        for (int32_t w = 0; w < n_features_; ++w) {
            const double value = normalized(k, w);
            if (!std::isfinite(value) || value < 0.0) {
                throw std::invalid_argument(
                    "Gamma-Poisson initial model values must be non-negative and finite");
            }
            normalized(k, w) = std::max(value, kProfileFloor);
            total += normalized(k, w);
        }
        if (!(total > 0.0) || !std::isfinite(total)) {
            throw std::invalid_argument(
                "Gamma-Poisson initial model has an empty topic");
        }
        normalized.row(k) /= total;
    }

    e_beta_ = normalized;
    topic_exposure_ = VectorXd::Constant(n_topics_,
        static_cast<double>(total_doc_count_)
            / static_cast<double>(n_topics_));
    if (!topic_names.empty()) topic_names_ = topic_names;
    refresh_cache();
}

void GammaPoissonTopicBase::refresh_cache() {
    beta_kernel_.resize(n_topics_, n_features_);
    constexpr double kFloor = 1e-300;
    for (int32_t k = 0; k < n_topics_; ++k) {
        for (int32_t w = 0; w < n_features_; ++w) {
            e_beta_(k, w) = std::max(e_beta_(k, w), kFloor);
        }
        e_beta_.row(k) /= e_beta_.row(k).sum();
    }
    if (inference_mode_ == GammaPoissonInferenceMode::LdaCompatible
        && (topic_concentration_.size() != n_topics_
            || !topic_concentration_.allFinite()
            || (topic_concentration_.array() <= 0.0).any())) {
        throw std::runtime_error(
            "LDA-compatible Gamma-Poisson model lacks topic concentration");
    }

    // Cache a feature-centered allocation kernel. Centering in log space is
    // responsibility-invariant and ensures that every feature has at least
    // one exactly representable topic score.
    for (int32_t w = 0; w < n_features_; ++w) {
        double max_log = -std::numeric_limits<double>::infinity();
        for (int32_t k = 0; k < n_topics_; ++k) {
            const double log_kernel = inference_mode_
                    == GammaPoissonInferenceMode::LdaCompatible
                ? psi(topic_concentration_(k) * e_beta_(k, w))
                    - psi(topic_concentration_(k))
                : std::log(e_beta_(k, w));
            beta_kernel_(k, w) = log_kernel;
            max_log = std::max(max_log, log_kernel);
        }
        if (!std::isfinite(max_log)) {
            throw std::runtime_error(
                "Gamma-Poisson allocation kernel is not finite");
        }
        for (int32_t k = 0; k < n_topics_; ++k) {
            beta_kernel_(k, w) = std::exp(beta_kernel_(k, w) - max_log);
        }
    }
    topic_capacity_ = VectorXd::Ones(n_topics_);
    model_cache_dirty_ = true;
    inference_cache_ready_ = true;
}

void GammaPoissonTopicBase::prepare_inference_cache() {
    if (!inference_cache_ready_) {
        refresh_cache();
    }
}

void GammaPoissonTopicBase::refresh_model_cache() {
    if (!model_cache_dirty_) {
        return;
    }
    model_phi_ = e_beta_;
    model_cache_dirty_ = false;
}

double GammaPoissonTopicBase::doc_exposure(const Document& doc) const {
    const double len = doc_sum_const(doc);
    if (len <= 0.0) {
        return 0.0;
    }
    return len;
}

uint64_t GammaPoissonTopicBase::doc_stream(
    uint64_t doc_index, uint64_t phase) const {
    uint64_t base = static_cast<uint64_t>(static_cast<uint32_t>(seed_));
    // Match the default LDA worker stream. Topic-model wrappers may assign
    // other streams, but both standalone fitters use stream zero.
    base ^= 0xd2b74407b1ce6e93ULL;
    uint64_t mixed = base ^ (phase * 0x9e3779b97f4a7c15ULL)
        ^ (doc_index + 1ULL);
    return ::splitmix64(mixed);
}

void GammaPoissonTopicModel::normalize_topic_allocation(
    const Eigen::Ref<const VectorXd>& theta_log, int32_t feature,
    VectorXd& allocation) const {
    if (theta_log.size() != n_topics_
            || feature < 0 || feature >= n_features_) {
        throw std::invalid_argument(
            "Gamma-Poisson topic allocation dimensions do not match");
    }
    allocation.resize(n_topics_);
    double max_log = -std::numeric_limits<double>::infinity();
    for (int32_t k = 0; k < n_topics_; ++k) {
        const double beta_log = inference_mode_
                == GammaPoissonInferenceMode::LdaCompatible
            ? psi(topic_concentration_(k) * e_beta_(k, feature))
                - psi(topic_concentration_(k))
            : std::log(e_beta_(k, feature));
        allocation(k) = theta_log(k) + beta_log;
        max_log = std::max(max_log, allocation(k));
    }
    if (!std::isfinite(max_log)) {
        throw std::runtime_error(
            "Gamma-Poisson topic allocation has no finite score");
    }
    double total = 0.0;
    for (int32_t k = 0; k < n_topics_; ++k) {
        allocation(k) = std::exp(allocation(k) - max_log);
        total += allocation(k);
    }
    if (!std::isfinite(total) || total <= 0.0) {
        throw std::runtime_error(
            "Gamma-Poisson topic allocation is not positive and finite");
    }
    allocation /= total;
}

void GammaPoissonTopicModel::expected_observed_counts(const Document& doc,
    std::vector<double>& means) const {
    if (has_dispersion_) {
        error("%s: dispersion means must be computed from a Poisson warmup model", __func__);
    }
    VectorXd theta_shape, theta_rate;
    fit_one_document(theta_shape, theta_rate, doc);
    const VectorXd e_theta = theta_shape.array() / theta_rate.array().max(1e-12);
    const double exposure = doc_exposure(doc);
    means.resize(doc.ids.size());
    for (size_t j = 0; j < doc.ids.size(); ++j) {
        const uint32_t w = doc.ids[j];
        if (w >= static_cast<uint32_t>(n_features_)) {
            error("%s: feature index %u is out of range", __func__, w);
        }
        double lambda = 0.0;
        for (int32_t k = 0; k < n_topics_; ++k) {
            lambda += e_theta(k) * e_beta_(k, w);
        }
        means[j] = std::max(exposure * lambda, 1e-12);
    }
}

void GammaPoissonTopicModel::WorkerState::reset(uint64_t current_generation,
    int32_t n_topics, int32_t n_features, bool with_dispersion) {
    if (generation == current_generation) {
        return;
    }
    generation = current_generation;
    ss.setZero(n_topics, n_features);
    if (with_dispersion) {
        dispersion_correction.setZero(n_topics, n_features);
    }
    ctheta.setZero(n_topics);
    iteration_sum = 0;
    documents = 0;
    failed = 0;
}

template <bool WithDispersion>
int32_t GammaPoissonTopicModel::fit_one_document(VectorXd& theta_shape,
    VectorXd& theta_rate, LocalWorkspace& workspace, const Document& doc,
    uint64_t rng_stream) const {
    const int32_t n_ids = static_cast<int32_t>(doc.ids.size());
    const double prior_shape = theta_prior_shape();
    theta_shape = VectorXd::Constant(n_topics_, prior_shape);
    theta_rate.resize(n_topics_);
    const double c = doc_exposure(doc);
    for (int32_t k = 0; k < n_topics_; ++k) {
        theta_rate(k) = theta_prior_rate(k) + c * topic_capacity_(k);
    }
    if (n_ids == 0) {
        workspace.theta_log.resize(n_topics_);
        workspace.theta_kernel.resize(n_topics_);
        double max_log = -std::numeric_limits<double>::infinity();
        for (int32_t k = 0; k < n_topics_; ++k) {
            workspace.theta_log(k) =
                psi(theta_shape(k)) - std::log(theta_rate(k));
            max_log = std::max(max_log, workspace.theta_log(k));
        }
        for (int32_t k = 0; k < n_topics_; ++k) {
            workspace.theta_kernel(k) =
                std::exp(workspace.theta_log(k) - max_log);
        }
        return 0;
    }

    workspace.beta_kernel.resize(n_topics_, n_ids);
    if constexpr (WithDispersion) {
        workspace.beta_mean.resize(n_topics_, n_ids);
    }
    for (int32_t j = 0; j < n_ids; ++j) {
        const uint32_t w = doc.ids[j];
        if (w >= static_cast<uint32_t>(n_features_)) {
            error("%s: feature index %u is out of range", __func__, w);
        }
        workspace.beta_kernel.col(j) = beta_kernel_.col(w);
        if constexpr (WithDispersion) {
            workspace.beta_mean.col(j) = e_beta_.col(w);
        }
    }

    workspace.theta_log.resize(n_topics_);
    workspace.theta_kernel.resize(n_topics_);
    workspace.last_shape.resize(n_topics_);
    workspace.assigned.resize(n_topics_);
    workspace.norm.resize(n_ids);
    workspace.ratio.resize(n_ids);
    if constexpr (WithDispersion) {
        workspace.e_theta.resize(n_topics_);
        workspace.epsilon.resize(n_ids);
    }
    const Eigen::Map<const VectorXd> counts(doc.cnts.data(), n_ids);
    const double doc_total = doc_sum_const(doc);
    if (inference_mode_ == GammaPoissonInferenceMode::LdaCompatible
        && rng_stream != 0) {
        std::gamma_distribution<double> gamma_dist(100.0, 0.01);
        SplitMix64Engine rng(rng_stream);
        for (int32_t k = 0; k < n_topics_; ++k) {
            theta_shape(k) = gamma_dist(rng);
        }
    } else {
        for (int32_t k = 0; k < n_topics_; ++k) {
            theta_shape(k) += doc_total / static_cast<double>(n_topics_);
        }
    }

    auto update_dispersion_rate = [&] {
        if constexpr (WithDispersion) {
            workspace.e_theta =
                theta_shape.array() / theta_rate.array().max(1e-12);
            workspace.norm.noalias() =
                workspace.beta_mean.transpose() * workspace.e_theta;
            for (int32_t k = 0; k < n_topics_; ++k) {
                theta_rate(k) = theta_prior_rate(k) + c * topic_capacity_(k);
            }
            for (int32_t j = 0; j < n_ids; ++j) {
                const double tau = tau_(doc.ids[j]);
                workspace.epsilon(j) = (tau + counts(j))
                    / std::max(tau + c * workspace.norm(j), 1e-12);
                workspace.ratio(j) = c * (workspace.epsilon(j) - 1.0);
            }
            theta_rate.noalias() += workspace.beta_mean * workspace.ratio;
        }
    };

    auto update_theta_kernel = [&] {
        double max_log = -std::numeric_limits<double>::infinity();
        for (int32_t k = 0; k < n_topics_; ++k) {
            theta_rate(k) = std::max(theta_rate(k), 1e-12);
            workspace.theta_log(k) =
                psi(theta_shape(k)) - std::log(theta_rate(k));
            max_log = std::max(max_log, workspace.theta_log(k));
        }
        for (int32_t k = 0; k < n_topics_; ++k) {
            workspace.theta_kernel(k) =
                std::exp(workspace.theta_log(k) - max_log);
        }
    };

    auto update_assignments = [&] {
        workspace.norm.noalias() =
            workspace.beta_kernel.transpose() * workspace.theta_kernel;
        const bool use_kernel = workspace.norm.allFinite()
            && (workspace.norm.array() > 0.0).all();
        if (use_kernel) {
            workspace.ratio = counts.array() / workspace.norm.array();
            workspace.assigned.noalias() =
                workspace.beta_kernel * workspace.ratio;
            workspace.assigned.array() *= workspace.theta_kernel.array();
            return;
        }

        workspace.assigned.setZero();
        VectorXd allocation;
        for (int32_t j = 0; j < n_ids; ++j) {
            const uint32_t w = doc.ids[j];
            normalize_topic_allocation(
                workspace.theta_log, static_cast<int32_t>(w), allocation);
            workspace.assigned.noalias() += counts(j) * allocation;
        }
    };

    double diff = 1.0;
    int32_t iter = 0;
    while (iter < max_doc_update_iter_) {
        workspace.last_shape = theta_shape;
        update_dispersion_rate();
        update_theta_kernel();
        update_assignments();
        theta_shape = workspace.assigned.array() + prior_shape;
        diff = (theta_shape - workspace.last_shape).cwiseAbs().sum()
            / static_cast<double>(n_topics_);
        ++iter;
        if (diff < mean_change_tol_) {
            break;
        }
    }
    update_dispersion_rate();
    update_theta_kernel();
    return iter;
}

int32_t GammaPoissonTopicModel::fit_one_document(VectorXd& theta_shape,
    VectorXd& theta_rate, const Document& doc) const {
    LocalWorkspace workspace;
    return has_dispersion_
        ? fit_one_document<true>(theta_shape, theta_rate, workspace, doc)
        : fit_one_document<false>(theta_shape, theta_rate, workspace, doc);
}

RowVectorXd GammaPoissonTopicBase::normalized_theta_hat(const VectorXd& theta_shape,
    const VectorXd& theta_rate) const {
    RowVectorXd out(n_topics_);
    double total = 0.0;
    for (int32_t k = 0; k < n_topics_; ++k) {
        out(k) = theta_shape(k) / theta_rate(k) * topic_capacity_(k);
        total += out(k);
    }
    if (total <= 0.0 || !std::isfinite(total)) {
        out.setConstant(1.0 / static_cast<double>(n_topics_));
    } else {
        out /= total;
    }
    return out;
}

template <bool WithDispersion>
void GammaPoissonTopicModel::accumulate_document(WorkerState& state,
    const Document& doc, uint64_t rng_stream) const {
    const int32_t iter = fit_one_document<WithDispersion>(
        state.theta_shape, state.theta_rate, state.workspace, doc, rng_stream);
    ++state.documents;
    state.iteration_sum += iter;
    state.failed += iter >= max_doc_update_iter_;

    state.workspace.e_theta =
        state.theta_shape.array() / state.theta_rate.array().max(1e-12);
    const double c = doc_exposure(doc);
    state.ctheta += c * state.workspace.e_theta;

    const int32_t n_ids = static_cast<int32_t>(doc.ids.size());
    if (n_ids == 0) {
        return;
    }
    const Eigen::Map<const VectorXd> counts(doc.cnts.data(), n_ids);
    state.workspace.norm.noalias() =
        state.workspace.beta_kernel.transpose() * state.workspace.theta_kernel;
    const bool use_kernel = state.workspace.norm.allFinite()
        && (state.workspace.norm.array() > 0.0).all();
    if (use_kernel) {
        state.workspace.ratio = counts.array() / state.workspace.norm.array();
        for (int32_t j = 0; j < n_ids; ++j) {
            state.ss.col(doc.ids[j]).array() += state.workspace.ratio(j)
                * state.workspace.theta_kernel.array()
                * state.workspace.beta_kernel.col(j).array();
        }
    } else {
        VectorXd allocation;
        for (int32_t j = 0; j < n_ids; ++j) {
            const uint32_t w = doc.ids[j];
            normalize_topic_allocation(state.workspace.theta_log,
                static_cast<int32_t>(w), allocation);
            state.ss.col(w).noalias() += counts(j) * allocation;
        }
    }

    if constexpr (WithDispersion) {
        state.workspace.norm.noalias() =
            state.workspace.beta_mean.transpose() * state.workspace.e_theta;
        for (int32_t j = 0; j < n_ids; ++j) {
            const double tau = tau_(doc.ids[j]);
            const double epsilon = (tau + counts(j))
                / std::max(tau + c * state.workspace.norm(j), 1e-12);
            const double correction = c * (epsilon - 1.0);
            state.dispersion_correction.col(doc.ids[j]).noalias() +=
                correction * state.workspace.e_theta;
        }
    }
}

template <bool WithDispersion>
void GammaPoissonTopicModel::collect_batch_statistics(
    const std::vector<Document>& docs, MatrixXd& counts, MatrixXd& delta,
    VectorXd& exposure, int64_t& iteration_sum, int32_t& failed) {
    const int32_t minibatch_size = static_cast<int32_t>(docs.size());
    counts.setZero(n_topics_, n_features_);
    if constexpr (WithDispersion) delta.setZero(n_topics_, n_features_);
    else delta.resize(0, 0);
    exposure.setZero(n_topics_);
    iteration_sum = 0;
    failed = 0;
    if (minibatch_size == 0) {
        return;
    }

    if (!worker_states_) {
        worker_states_ =
            std::make_unique<tbb::enumerable_thread_specific<WorkerState>>();
    }
    ++worker_generation_;

    tbb::parallel_for(tbb::blocked_range<int32_t>(0, minibatch_size),
        [&](const tbb::blocked_range<int32_t>& range) {
            WorkerState& state = worker_states_->local();
            state.reset(worker_generation_, n_topics_, n_features_, WithDispersion);
            for (int32_t d = range.begin(); d < range.end(); ++d) {
                const uint64_t stream = inference_mode_
                        == GammaPoissonInferenceMode::LdaCompatible
                    ? doc_stream(static_cast<uint64_t>(d),
                        0x13a5be1ULL ^ static_cast<uint64_t>(update_count_))
                    : 0;
                accumulate_document<WithDispersion>(state, docs[d], stream);
            }
        });

    for (WorkerState& state : *worker_states_) {
        if (state.generation != worker_generation_) {
            continue;
        }
        counts += state.ss;
        if constexpr (WithDispersion) delta += state.dispersion_correction;
        exposure += state.ctheta;
        iteration_sum += state.iteration_sum;
        failed += state.failed;
    }
}

void GammaPoissonTopicModel::dictionary_to_logits(
    const MatrixXd& dictionary, MatrixXd& logits) {
    logits.resizeLike(dictionary);
    for (int32_t k = 0; k < dictionary.rows(); ++k) {
        for (int32_t w = 0; w < dictionary.cols(); ++w) {
            logits(k, w) = std::log(std::max(dictionary(k, w), 1e-300));
        }
        logits.row(k).array() -= logits.row(k).mean();
    }
}

void GammaPoissonTopicModel::logits_to_dictionary(
    const MatrixXd& logits, MatrixXd& dictionary) {
    dictionary.resizeLike(logits);
    constexpr double kFloor = 1e-300;
    for (int32_t k = 0; k < logits.rows(); ++k) {
        const double max_logit = logits.row(k).maxCoeff();
        double total = 0.0;
        for (int32_t w = 0; w < logits.cols(); ++w) {
            dictionary(k, w) = std::max(std::exp(logits(k, w) - max_logit), kFloor);
            total += dictionary(k, w);
        }
        dictionary.row(k) /= total;
    }
}

VectorXd GammaPoissonTopicModel::ownership_weights(
        const MatrixXd& counts) const {
    VectorXd weights = VectorXd::Ones(n_topics_);
    if (ownership_mode_ != GammaPoissonOwnershipMode::Prevalence) {
        return weights;
    }
    if (counts.rows() != n_topics_ || counts.cols() != n_features_
            || !counts.allFinite() || (counts.array() < 0.0).any()) {
        throw std::invalid_argument(
            "Gamma-Poisson prevalence ownership received invalid counts");
    }
    weights = counts.rowwise().sum();
    const double total = weights.sum();
    if (!(total > 0.0) || !std::isfinite(total)) {
        return VectorXd::Ones(n_topics_);
    }
    weights *= static_cast<double>(n_topics_) / total;
    return weights;
}

double GammaPoissonTopicModel::ownership_entropy(
        const MatrixXd& dictionary, const VectorXd& weights) const {
    if (dictionary.rows() != n_topics_ || dictionary.cols() != n_features_
            || weights.size() != n_topics_) {
        throw std::invalid_argument(
            "Gamma-Poisson ownership entropy dimensions do not match");
    }
    const MatrixXd weighted = weights.asDiagonal() * dictionary;
    const VectorXd feature_mass = weighted.colwise().sum().transpose();
    double entropy = 0.0;
    for (int32_t k = 0; k < n_topics_; ++k) {
        if (!(weights(k) > 0.0)) continue;
        for (int32_t w = 0; w < n_features_; ++w) {
            const double value = weighted(k, w);
            if (!(value > 0.0)) continue;
            entropy += value * std::log(
                std::max(feature_mass(w), 1e-300) / value);
        }
    }
    return entropy;
}

double GammaPoissonTopicModel::objective_and_gradient(
    const MatrixXd& counts, const MatrixXd& delta, double ownership_fraction,
    const MatrixXd& logits, MatrixXd* gradient, MatrixXd* dictionary) const {
    MatrixXd local_dictionary;
    MatrixXd& beta = dictionary ? *dictionary : local_dictionary;
    logits_to_dictionary(logits, beta);
    const double prior = dictionary_prior_mass_ / static_cast<double>(n_features_);
    const double lambda = ownership_strength_ * ownership_fraction
        * counts.sum() / static_cast<double>(n_topics_);
    const VectorXd weights = ownership_weights(counts);
    const MatrixXd weighted_beta = weights.asDiagonal() * beta;
    VectorXd feature_mass = weighted_beta.colwise().sum().transpose();
    double objective = 0.0;
    if (gradient) gradient->resize(n_topics_, n_features_);
    for (int32_t k = 0; k < n_topics_; ++k) {
        double a_sum = 0.0;
        double beta_delta = 0.0;
        double beta_h = 0.0;
        for (int32_t w = 0; w < n_features_; ++w) {
            const double a = counts(k, w) + prior;
            const double weighted_value = weights(k) * beta(k, w);
            const double h = weights(k) > 0.0
                ? std::log(std::max(feature_mass(w), 1e-300)
                    / std::max(weighted_value, 1e-300))
                : 0.0;
            objective += a * std::log(std::max(beta(k, w), 1e-300));
            if (delta.size() != 0) objective -= delta(k, w) * beta(k, w);
            objective -= lambda * weighted_value * h;
            a_sum += a;
            if (delta.size() != 0) beta_delta += beta(k, w) * delta(k, w);
            beta_h += beta(k, w) * weights(k) * h;
        }
        if (gradient) {
            for (int32_t w = 0; w < n_features_; ++w) {
                const double a = counts(k, w) + prior;
                const double weighted_value = weights(k) * beta(k, w);
                const double h = weights(k) > 0.0
                    ? std::log(std::max(feature_mass(w), 1e-300)
                        / std::max(weighted_value, 1e-300))
                    : 0.0;
                double value = a - beta(k, w) * a_sum;
                if (delta.size() != 0) {
                    value -= beta(k, w) * (delta(k, w) - beta_delta);
                }
                value -= lambda * beta(k, w)
                    * (weights(k) * h - beta_h);
                (*gradient)(k, w) = value;
            }
        }
    }
    return objective;
}

bool GammaPoissonTopicModel::solve_unregularized_dictionary(
    const MatrixXd& counts, const MatrixXd& delta, MatrixXd& dictionary) const {
    const bool with_dispersion = delta.size() != 0;
    if (counts.rows() != n_topics_ || counts.cols() != n_features_
        || (with_dispersion
            && (delta.rows() != n_topics_ || delta.cols() != n_features_))) {
        throw std::invalid_argument(
            "Gamma-Poisson unregularized M-step dimensions do not match the model");
    }
    if (!counts.allFinite()
        || (with_dispersion && !delta.allFinite())) {
        throw std::runtime_error(
            "Gamma-Poisson unregularized M-step received invalid statistics");
    }

    const double prior = dictionary_prior_mass_
        / static_cast<double>(n_features_);
    if ((counts.array() + prior <= 0.0).any()) {
        throw std::runtime_error(
            "Gamma-Poisson unregularized M-step received non-positive MAP mass");
    }
    dictionary = e_beta_;
    VectorXd mass(n_features_);
    for (int32_t k = 0; k < n_topics_; ++k) {
        mass = counts.row(k).transpose().array() + prior;
        const double total_mass = mass.sum();
        if (!(total_mass > 0.0) || !std::isfinite(total_mass)) {
            continue;
        }
        if (!with_dispersion) {
            dictionary.row(k) = (mass / total_mass).transpose();
            continue;
        }

        // A zero pseudocount can put the optimum on the simplex boundary.
        // Leave that uncommon case to the positive-logit optimizer.
        if ((mass.array() <= 0.0).any()) return false;

        const double boundary = -delta.row(k).minCoeff();
        double lower = std::nextafter(
            boundary, std::numeric_limits<double>::infinity());
        double gap = total_mass;
        double upper = boundary + gap;
        while (!(upper > boundary) || !std::isfinite(upper)) {
            gap *= 2.0;
            upper = boundary + gap;
            if (!std::isfinite(gap)) {
                throw std::runtime_error(
                    "Could not bracket Gamma-Poisson dispersion M-step");
            }
        }

        auto evaluate_mass = [&](double multiplier, double* derivative) {
            double value = 0.0;
            double slope = 0.0;
            for (int32_t w = 0; w < n_features_; ++w) {
                const double denominator = delta(k, w) + multiplier;
                if (!(denominator > 0.0)) {
                    if (derivative) *derivative =
                        -std::numeric_limits<double>::infinity();
                    return std::numeric_limits<double>::infinity();
                }
                const double term = mass(w) / denominator;
                value += term;
                slope -= term / denominator;
            }
            if (derivative) *derivative = slope;
            return value;
        };

        while (evaluate_mass(upper, nullptr) > 1.0) {
            gap *= 2.0;
            upper = boundary + gap;
            if (!std::isfinite(upper)) {
                throw std::runtime_error(
                    "Could not bracket Gamma-Poisson dispersion M-step");
            }
        }

        double multiplier = lower + 0.5 * (upper - lower);
        for (int32_t iteration = 0; iteration < 64; ++iteration) {
            double derivative = 0.0;
            const double fitted_mass = evaluate_mass(multiplier, &derivative);
            if (std::isfinite(fitted_mass)
                && std::abs(fitted_mass - 1.0) <= 1e-12) {
                break;
            }
            if (fitted_mass > 1.0) lower = multiplier;
            else upper = multiplier;

            double candidate = multiplier;
            if (std::isfinite(fitted_mass) && std::isfinite(derivative)
                && derivative < 0.0) {
                candidate = multiplier - (fitted_mass - 1.0) / derivative;
            }
            if (!(candidate > lower && candidate < upper)
                || !std::isfinite(candidate)) {
                candidate = lower + 0.5 * (upper - lower);
            }
            multiplier = candidate;
        }

        for (int32_t w = 0; w < n_features_; ++w) {
            dictionary(k, w) = mass(w) / (delta(k, w) + multiplier);
        }
        dictionary.row(k) /= dictionary.row(k).sum();
    }
    return true;
}

bool GammaPoissonTopicModel::optimize_dictionary_mm(
    const MatrixXd& counts, const MatrixXd& delta, double ownership_fraction,
    int32_t max_steps, double, double* max_row_change) {
    const MatrixXd initial_dictionary = e_beta_;
    const double prior = dictionary_prior_mass_
        / static_cast<double>(n_features_);
    if (max_steps <= 0 || prior <= 0.0
            || (counts.array() + prior <= 0.0).any()) {
        return false;
    }
    const double lambda = ownership_strength_ * ownership_fraction
        * counts.sum() / static_cast<double>(n_topics_);
    const VectorXd weights = ownership_weights(counts);
    auto evaluate = [&](const MatrixXd& dictionary) {
        MatrixXd logits;
        dictionary_to_logits(dictionary, logits);
        return objective_and_gradient(counts, delta, ownership_fraction,
            logits, nullptr, nullptr);
    };

    MatrixXd dictionary = e_beta_;
    double value = evaluate(dictionary);
    if (!std::isfinite(value)) {
        throw std::runtime_error("Non-finite Gamma-Poisson MM objective");
    }

    // The exact unregularized update is cheap and is often a substantially
    // better starting point after the running sufficient statistics change.
    // Never return a regularized M-step worse than this available candidate.
    MatrixXd unregularized;
    if (solve_unregularized_dictionary(counts, delta, unregularized)) {
        const double unregularized_value = evaluate(unregularized);
        if (std::isfinite(unregularized_value)
                && unregularized_value > value) {
            dictionary = std::move(unregularized);
            value = unregularized_value;
        }
    }

    bool changed = (dictionary - initial_dictionary).cwiseAbs().maxCoeff() > 0.0;
    last_relative_objective_gain_ = 0.0;
    const double relative_tolerance = max_steps <= 2 ? 1e-8 : 1e-10;
    const double row_tolerance = max_steps <= 2 ? 1e-4 : 1e-6;
    for (int32_t iteration = 0; iteration < max_steps; ++iteration) {
        const MatrixXd weighted = weights.asDiagonal() * dictionary;
        const VectorXd feature_mass = weighted.colwise().sum().transpose();
        MatrixXd cost = delta.size() == 0
            ? MatrixXd::Zero(n_topics_, n_features_) : delta;
        for (int32_t k = 0; k < n_topics_; ++k) {
            if (!(weights(k) > 0.0)) continue;
            for (int32_t w = 0; w < n_features_; ++w) {
                const double weighted_value = weights(k) * dictionary(k, w);
                const double h = std::log(
                    std::max(feature_mass(w), 1e-300)
                    / std::max(weighted_value, 1e-300));
                cost(k, w) += lambda * weights(k) * h;
            }
        }

        MatrixXd candidate(n_topics_, n_features_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            const double boundary = -cost.row(k).minCoeff();
            double lower = std::nextafter(
                boundary, std::numeric_limits<double>::infinity());
            double gap = counts.row(k).sum()
                + dictionary_prior_mass_ + 1.0;
            double upper = boundary + gap;
            auto fitted_mass = [&](double multiplier, double* derivative) {
                double total = 0.0;
                double slope = 0.0;
                for (int32_t w = 0; w < n_features_; ++w) {
                    const double denominator = cost(k, w) + multiplier;
                    if (!(denominator > 0.0)) {
                        if (derivative) *derivative =
                            -std::numeric_limits<double>::infinity();
                        return std::numeric_limits<double>::infinity();
                    }
                    const double mass = counts(k, w) + prior;
                    const double term = mass / denominator;
                    total += term;
                    slope -= term / denominator;
                }
                if (derivative) *derivative = slope;
                return total;
            };
            while (fitted_mass(upper, nullptr) > 1.0) {
                gap *= 2.0;
                upper = boundary + gap;
                if (!std::isfinite(upper)) {
                    throw std::runtime_error(
                        "Could not bracket Gamma-Poisson MM simplex multiplier");
                }
            }
            double multiplier = lower + 0.5 * (upper - lower);
            for (int32_t root_iteration = 0; root_iteration < 48;
                    ++root_iteration) {
                double derivative = 0.0;
                const double total = fitted_mass(multiplier, &derivative);
                if (std::isfinite(total) && std::abs(total - 1.0) <= 1e-12) {
                    break;
                }
                if (total > 1.0) lower = multiplier;
                else upper = multiplier;
                double next = multiplier;
                if (std::isfinite(total) && std::isfinite(derivative)
                        && derivative < 0.0) {
                    next = multiplier - (total - 1.0) / derivative;
                }
                if (!(next > lower && next < upper) || !std::isfinite(next)) {
                    next = lower + 0.5 * (upper - lower);
                }
                multiplier = next;
            }
            for (int32_t w = 0; w < n_features_; ++w) {
                candidate(k, w) = (counts(k, w) + prior)
                    / (cost(k, w) + multiplier);
            }
            candidate.row(k) /= candidate.row(k).sum();
        }

        const double candidate_value = evaluate(candidate);
        const double scale = std::max(1.0, std::abs(value));
        if (!std::isfinite(candidate_value)
                || candidate_value + 1e-12 * scale < value) {
            throw std::runtime_error(
                "Gamma-Poisson MM update decreased the penalized objective");
        }
        // Do not accept a roundoff-scale decrease.  The update is at its
        // numerical fixed point, and retaining the previous iterate makes the
        // reported MM trajectory strictly monotone as well as theoretically
        // monotone.
        if (candidate_value < value) break;
        double largest_change = 0.0;
        for (int32_t k = 0; k < n_topics_; ++k) {
            largest_change = std::max(largest_change,
                (candidate.row(k) - dictionary.row(k)).cwiseAbs().sum());
        }
        const double improvement = candidate_value - value;
        last_relative_objective_gain_ = improvement / scale;
        dictionary = std::move(candidate);
        value = candidate_value;
        changed = changed || largest_change > 0.0;
        ++mm_steps_;
        if (largest_change <= row_tolerance
                && last_relative_objective_gain_ <= relative_tolerance) {
            break;
        }
    }
    e_beta_ = std::move(dictionary);
    refresh_cache();
    last_maximum_row_change_ = 0.0;
    for (int32_t k = 0; k < n_topics_; ++k) {
        last_maximum_row_change_ = std::max(last_maximum_row_change_,
            (e_beta_.row(k) - initial_dictionary.row(k))
                .cwiseAbs().sum());
    }
    if (max_row_change) *max_row_change = last_maximum_row_change_;
    return changed;
}

bool GammaPoissonTopicModel::optimize_dictionary(
    const MatrixXd& counts, const MatrixXd& delta, double ownership_fraction,
    int32_t max_steps, bool use_lbfgs, double movement_tolerance,
    double* max_row_change) {
    const MatrixXd initial_dictionary = e_beta_;
    const double effective_ownership_lambda = ownership_strength_
        * ownership_fraction * counts.sum() / static_cast<double>(n_topics_);
    if (effective_ownership_lambda == 0.0) {
        MatrixXd exact_dictionary;
        if (solve_unregularized_dictionary(counts, delta, exact_dictionary)) {
            e_beta_ = std::move(exact_dictionary);
            refresh_cache();
            last_maximum_row_change_ = 0.0;
            for (int32_t k = 0; k < n_topics_; ++k) {
                last_maximum_row_change_ = std::max(last_maximum_row_change_,
                    (e_beta_.row(k) - initial_dictionary.row(k))
                        .cwiseAbs().sum());
            }
            if (max_row_change) *max_row_change = last_maximum_row_change_;
            return last_maximum_row_change_ > movement_tolerance;
        }
    }
    if (dictionary_prior_mass_ > 0.0) {
        return optimize_dictionary_mm(counts, delta, ownership_fraction,
            max_steps, movement_tolerance, max_row_change);
    }
    ++mm_fallbacks_;
    MatrixXd logits, gradient, dictionary;
    dictionary_to_logits(e_beta_, logits);
    double value = objective_and_gradient(counts, delta, ownership_fraction,
        logits, &gradient, &dictionary);
    if (!std::isfinite(value) || !gradient.allFinite()) {
        throw std::runtime_error("Non-finite Gamma-Poisson MAP objective");
    }
    constexpr int32_t kHistory = 5;
    punkst::LbfgsHistory<MatrixXd> history(kHistory);
    double largest_change = 0.0;
    bool any_accepted = false;
    for (int32_t iteration = 0; iteration < max_steps; ++iteration) {
        const double gradient_tolerance = 1e-10 * std::max(1.0,
            counts.sum() + dictionary_prior_mass_ * n_topics_);
        if (gradient.cwiseAbs().maxCoeff() <= gradient_tolerance) break;
        MatrixXd direction = gradient;
        bool used_lbfgs = use_lbfgs && !history.empty();
        if (used_lbfgs) {
            direction = history.apply(gradient);
        }
        auto precondition_gradient = [&] {
            direction = gradient;
            VectorXd feature_mass = dictionary.colwise().sum().transpose();
            const double lambda = ownership_strength_ * ownership_fraction
                * counts.sum() / static_cast<double>(n_topics_);
            for (int32_t k = 0; k < n_topics_; ++k) {
                double scale = counts.row(k).sum() + dictionary_prior_mass_;
                if (delta.size() != 0) {
                    scale = std::max(scale,
                        (dictionary.row(k).array() * delta.row(k).array().abs()).sum());
                }
                double weighted_h = 0.0;
                for (int32_t w = 0; w < n_features_; ++w) {
                    const double h = std::log(std::max(feature_mass(w), 1e-300)
                        / std::max(dictionary(k, w), 1e-300));
                    weighted_h += dictionary(k, w) * std::abs(h);
                }
                scale = std::max({1.0, scale, lambda * weighted_h});
                direction.row(k) /= scale;
            }
        };
        double directional = (gradient.array() * direction.array()).sum();
        if (!direction.allFinite() || !(directional > 0.0)) {
            if (used_lbfgs) ++lbfgs_fallbacks_;
            history.clear();
            precondition_gradient();
            directional = (gradient.array() * direction.array()).sum();
        } else if (!used_lbfgs) {
            precondition_gradient();
            directional = (gradient.array() * direction.array()).sum();
        }
        for (int32_t k = 0; k < n_topics_; ++k) {
            const double max_abs = direction.row(k).cwiseAbs().maxCoeff();
            if (max_abs > 1.0) direction.row(k) /= max_abs;
        }
        directional = (gradient.array() * direction.array()).sum();
        if (!(directional > 0.0) || !std::isfinite(directional)) break;

        bool accepted = false;
        double step = 1.0;
        MatrixXd candidate_logits, candidate_gradient, candidate_dictionary;
        double candidate_value = value;
        for (int32_t backtrack = 0; backtrack < 20; ++backtrack) {
            candidate_logits = logits + step * direction;
            for (int32_t k = 0; k < n_topics_; ++k) {
                candidate_logits.row(k).array() -= candidate_logits.row(k).mean();
            }
            candidate_value = objective_and_gradient(counts, delta,
                ownership_fraction, candidate_logits, &candidate_gradient,
                &candidate_dictionary);
            if (std::isfinite(candidate_value) && candidate_gradient.allFinite()
                && candidate_value >= value + 1e-4 * step * directional) {
                accepted = true;
                break;
            }
            step *= 0.5;
        }
        if (!accepted) {
            if (use_lbfgs && used_lbfgs) {
                ++lbfgs_fallbacks_;
                history.clear();
                --iteration;
                continue;
            }
            ++failed_gradient_steps_;
            if (use_lbfgs) {
                throw std::runtime_error("Gamma-Poisson MAP line search failed");
            }
            break;
        }
        any_accepted = true;
        if (used_lbfgs) ++lbfgs_steps_;
        else ++accepted_gradient_steps_;
        largest_change = 0.0;
        for (int32_t k = 0; k < n_topics_; ++k) {
            largest_change = std::max(largest_change,
                (candidate_dictionary.row(k) - dictionary.row(k)).cwiseAbs().sum());
        }
        if (use_lbfgs) {
            MatrixXd s = candidate_logits - logits;
            MatrixXd y = gradient - candidate_gradient;
            history.update(std::move(s), std::move(y));
        }
        const double improvement = candidate_value - value;
        logits = std::move(candidate_logits);
        gradient = std::move(candidate_gradient);
        dictionary = std::move(candidate_dictionary);
        value = candidate_value;
        if (largest_change <= movement_tolerance
            || improvement <= 1e-10 * std::max(1.0, std::abs(value))) {
            break;
        }
    }
    if (any_accepted) {
        e_beta_ = dictionary;
        refresh_cache();
    }
    last_maximum_row_change_ = 0.0;
    for (int32_t k = 0; k < n_topics_; ++k) {
        last_maximum_row_change_ = std::max(last_maximum_row_change_,
            (e_beta_.row(k) - initial_dictionary.row(k)).cwiseAbs().sum());
    }
    if (max_row_change) *max_row_change = last_maximum_row_change_;
    return any_accepted;
}

void GammaPoissonTopicModel::update_topic_concentration(
    const MatrixXd& counts) {
    if (inference_mode_ != GammaPoissonInferenceMode::LdaCompatible) return;
    if (counts.rows() != n_topics_ || counts.cols() != n_features_) {
        throw std::invalid_argument(
            "Gamma-Poisson concentration counts have invalid dimensions");
    }
    topic_concentration_ = counts.rowwise().sum().array()
        + dictionary_prior_mass_;
    if (!topic_concentration_.allFinite()
        || (topic_concentration_.array() <= 0.0).any()) {
        throw std::runtime_error(
            "Gamma-Poisson topic concentration is not positive and finite");
    }
}

template <bool WithDispersion>
void GammaPoissonTopicModel::partial_fit_impl(const std::vector<Document>& docs) {
    const int32_t minibatch_size = static_cast<int32_t>(docs.size());
    if (minibatch_size == 0) return;
    MatrixXd counts, delta;
    VectorXd exposure;
    int64_t iteration_sum = 0;
    int32_t failed = 0;
    collect_batch_statistics<WithDispersion>(
        docs, counts, delta, exposure, iteration_sum, failed);

    ++update_count_;
    const double rho = std::pow(learning_offset_ + update_count_, -learning_decay_);
    const double scale = static_cast<double>(total_doc_count_) / static_cast<double>(minibatch_size);
    counts *= scale;
    if constexpr (WithDispersion) delta *= scale;
    const bool initialize_running_statistics = !running_stats_ready_;
    if (initialize_running_statistics) {
        running_counts_ = counts;
        running_delta_ = delta;
        running_stats_ready_ = true;
    } else {
        running_counts_ = (1.0 - rho) * running_counts_ + rho * counts;
        if constexpr (WithDispersion) {
            if (running_delta_.size() == 0) running_delta_ = delta;
            else running_delta_ = (1.0 - rho) * running_delta_ + rho * delta;
        }
    }
    update_topic_concentration(running_counts_);
    optimize_dictionary(running_counts_, running_delta_, ownership_fraction_,
        2, false, 0.0);
    VectorXd target_exposure = scale * exposure;
    if (initialize_running_statistics
            || topic_exposure_.size() != n_topics_
            || topic_exposure_.sum() <= 0.0) {
        topic_exposure_ = target_exposure;
    } else {
        topic_exposure_ = (1.0 - rho) * topic_exposure_
            + rho * target_exposure;
    }

    if (verbose_ > 0) {
        const double avg =
            static_cast<double>(iteration_sum) / minibatch_size;
        notice("Gamma-Poisson partial fit: %d documents. Average iterations per doc: %.2f, %d documents did not reach mean change %.1e in %d iterations.",
            minibatch_size, avg, failed, mean_change_tol_, max_doc_update_iter_);
    }
}

void GammaPoissonTopicModel::partial_fit(const std::vector<Document>& docs) {
    if (ownership_schedule_active_) {
        const int64_t next_documents = ownership_documents_seen_
            + static_cast<int64_t>(docs.size());
        const double warmup_documents = static_cast<double>(
            ownership_warmup_epochs_) * ownership_documents_per_epoch_;
        const double ramp_documents = static_cast<double>(
            ownership_ramp_epochs_) * ownership_documents_per_epoch_;
        ownership_fraction_ = ramp_documents > 0.0
            ? std::clamp((next_documents - warmup_documents) / ramp_documents,
                0.0, 1.0)
            : (next_documents > warmup_documents ? 1.0 : 0.0);
        ownership_documents_seen_ = next_documents;
    }
    if (has_dispersion_) {
        partial_fit_impl<true>(docs);
    } else {
        partial_fit_impl<false>(docs);
    }
}

void GammaPoissonTopicModel::set_ownership_annealing(double fraction) {
    if (!std::isfinite(fraction) || fraction < 0.0 || fraction > 1.0) {
        throw std::invalid_argument("Gamma-Poisson ownership fraction must be in [0,1]");
    }
    ownership_fraction_ = fraction;
    ownership_schedule_active_ = false;
}

void GammaPoissonTopicModel::configure_ownership_annealing(
    int32_t warmup_epochs, int32_t ramp_epochs, int32_t documents_per_epoch,
    int64_t documents_seen) {
    if (warmup_epochs < 0 || ramp_epochs < 0 || documents_per_epoch <= 0
        || documents_seen < 0) {
        throw std::invalid_argument(
            "Gamma-Poisson ownership schedule values are invalid");
    }
    ownership_warmup_epochs_ = warmup_epochs;
    ownership_ramp_epochs_ = ramp_epochs;
    ownership_documents_per_epoch_ = documents_per_epoch;
    ownership_documents_seen_ = documents_seen;
    const double warmup_documents = static_cast<double>(warmup_epochs)
        * documents_per_epoch;
    const double ramp_documents = static_cast<double>(ramp_epochs)
        * documents_per_epoch;
    ownership_fraction_ = ramp_documents > 0.0
        ? std::clamp((documents_seen - warmup_documents) / ramp_documents,
            0.0, 1.0)
        : (documents_seen > warmup_documents ? 1.0 : 0.0);
    ownership_schedule_active_ = true;
}

void GammaPoissonTopicModel::reset_running_statistics() {
    running_stats_ready_ = false;
    running_counts_.resize(0, 0);
    running_delta_.resize(0, 0);
}

void GammaPoissonTopicModel::begin_full_refinement() {
    refinement_counts_ = MatrixXd::Zero(n_topics_, n_features_);
    if (has_dispersion_) refinement_delta_ = MatrixXd::Zero(n_topics_, n_features_);
    else refinement_delta_.resize(0, 0);
    refinement_exposure_ = VectorXd::Zero(n_topics_);
    refinement_active_ = true;
}

void GammaPoissonTopicModel::accumulate_full_refinement(
    const std::vector<Document>& docs) {
    if (!refinement_active_) {
        throw std::logic_error("Gamma-Poisson refinement pass is not active");
    }
    MatrixXd counts, delta;
    VectorXd exposure;
    int64_t iteration_sum = 0;
    int32_t failed = 0;
    if (has_dispersion_) {
        collect_batch_statistics<true>(
            docs, counts, delta, exposure, iteration_sum, failed);
        refinement_delta_ += delta;
    } else {
        collect_batch_statistics<false>(
            docs, counts, delta, exposure, iteration_sum, failed);
    }
    refinement_counts_ += counts;
    refinement_exposure_ += exposure;
}

bool GammaPoissonTopicModel::finish_full_refinement(double tolerance) {
    if (!refinement_active_) {
        throw std::logic_error("Gamma-Poisson refinement pass is not active");
    }
    if (!std::isfinite(tolerance) || tolerance < 0.0) {
        throw std::invalid_argument(
            "Gamma-Poisson refinement tolerance must be non-negative");
    }
    running_counts_ = refinement_counts_;
    running_delta_ = refinement_delta_;
    running_stats_ready_ = true;
    if (refinement_exposure_.size() == n_topics_
            && refinement_exposure_.sum() > 0.0) {
        topic_exposure_ = refinement_exposure_;
    }
    ownership_fraction_ = 1.0;
    ownership_schedule_active_ = false;
    update_topic_concentration(running_counts_);
    double max_change = 0.0;
    const bool accepted = optimize_dictionary(running_counts_, running_delta_,
        1.0, 100, true, tolerance, &max_change);
    refinement_active_ = false;
    refinement_counts_.resize(0, 0);
    refinement_delta_.resize(0, 0);
    return !accepted || max_change <= tolerance;
}

double GammaPoissonTopicModel::ownership_entropy() const {
    const VectorXd weights = running_stats_ready_
        ? ownership_weights(running_counts_)
        : VectorXd::Ones(n_topics_);
    return ownership_entropy(e_beta_, weights);
}

double GammaPoissonTopicModel::map_objective() const {
    if (!running_stats_ready_) return std::numeric_limits<double>::quiet_NaN();
    MatrixXd logits;
    dictionary_to_logits(e_beta_, logits);
    return objective_and_gradient(running_counts_, running_delta_,
        ownership_fraction_, logits, nullptr, nullptr);
}

GammaPoissonOptimizationDiagnostics
GammaPoissonTopicModel::optimization_diagnostics() const {
    GammaPoissonOptimizationDiagnostics out;
    out.objective = map_objective();
    out.ownership_entropy = ownership_entropy();
    out.uniform_ownership_entropy = ownership_entropy(
        e_beta_, VectorXd::Ones(n_topics_));
    if (running_stats_ready_) {
        out.effective_ownership_lambda = ownership_strength_
            * ownership_fraction_ * running_counts_.sum()
            / static_cast<double>(n_topics_);
    }
    out.maximum_row_change = last_maximum_row_change_;
    out.accepted_gradient_steps = accepted_gradient_steps_;
    out.failed_gradient_steps = failed_gradient_steps_;
    out.lbfgs_steps = lbfgs_steps_;
    out.lbfgs_fallbacks = lbfgs_fallbacks_;
    out.mm_steps = mm_steps_;
    out.mm_fallbacks = mm_fallbacks_;
    out.relative_objective_gain = last_relative_objective_gain_;
    return out;
}

RowMajorMatrixXd GammaPoissonTopicModel::transform(DocumentView docs) {
    const int32_t n_docs = static_cast<int32_t>(docs.size());
    RowMajorMatrixXd topics(n_docs, n_topics_);
    auto process_doc = [&](int32_t d) {
        VectorXd shape;
        VectorXd rate;
        LocalWorkspace workspace;
        const uint64_t stream = inference_mode_
                == GammaPoissonInferenceMode::LdaCompatible
            ? doc_stream(static_cast<uint64_t>(d),
                0x41f2c7d3ULL ^ static_cast<uint64_t>(update_count_))
            : 0;
        if (has_dispersion_) {
            fit_one_document<true>(shape, rate, workspace, docs[d], stream);
        } else {
            fit_one_document<false>(shape, rate, workspace, docs[d], stream);
        }
        topics.row(d) = normalized_theta_hat(shape, rate);
    };
    if (nThreads_ == 1) {
        for (int32_t d = 0; d < n_docs; ++d) {
            process_doc(d);
        }
    } else {
        tbb::parallel_for(0, n_docs, [&](int32_t d) { process_doc(d); });
    }
    return topics;
}

void GammaPoissonTopicModel::infer_document_posterior(const Document& doc,
    GammaPoissonDocumentPosterior& posterior) const {
    fit_one_document(posterior.shape, posterior.rate, doc);
    posterior.exposure = doc_exposure(doc);
}

RowVectorXd GammaPoissonTopicModel::normalized_topic_mean(
    const GammaPoissonDocumentPosterior& posterior) const {
    return normalized_theta_hat(posterior.shape, posterior.rate);
}

void GammaPoissonTopicModel::transform_with_posteriors(DocumentView docs,
    RowMajorMatrixXd& topics,
    std::vector<GammaPoissonDocumentPosterior>& posteriors) const {
    const int32_t n_docs = static_cast<int32_t>(docs.size());
    topics.resize(n_docs, n_topics_);
    posteriors.resize(n_docs);
    auto process_doc = [&](int32_t d) {
        LocalWorkspace workspace;
        const uint64_t stream = inference_mode_
                == GammaPoissonInferenceMode::LdaCompatible
            ? doc_stream(static_cast<uint64_t>(d),
                0x41f2c7d3ULL ^ static_cast<uint64_t>(update_count_))
            : 0;
        if (has_dispersion_) {
            fit_one_document<true>(posteriors[d].shape, posteriors[d].rate,
                workspace, docs[d], stream);
        } else {
            fit_one_document<false>(posteriors[d].shape, posteriors[d].rate,
                workspace, docs[d], stream);
        }
        posteriors[d].exposure = doc_exposure(docs[d]);
        topics.row(d) = normalized_topic_mean(posteriors[d]);
    };
    if (nThreads_ == 1) {
        for (int32_t d = 0; d < n_docs; ++d) {
            process_doc(d);
        }
    } else {
        tbb::parallel_for(0, n_docs, [&](int32_t d) { process_doc(d); });
    }
}

const std::vector<std::string>& GammaPoissonTopicBase::get_topic_names() {
    if (topic_names_.empty()) {
        topic_names_.resize(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            topic_names_[k] = std::to_string(k);
        }
    }
    return topic_names_;
}

const RowMajorMatrixXd& GammaPoissonTopicBase::get_model() {
    refresh_model_cache();
    return model_phi_;
}

RowMajorMatrixXd GammaPoissonTopicBase::copy_model() {
    return get_model();
}

void GammaPoissonTopicBase::get_topic_prevalence(
    std::vector<double>& weights) const {
    weights.resize(n_topics_);
    double total = topic_exposure_.sum();
    if (total <= 0.0) {
        std::fill(weights.begin(), weights.end(), 1.0 / static_cast<double>(n_topics_));
        return;
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        weights[k] = topic_exposure_(k) / total;
    }
}

void GammaPoissonTopicBase::sort_topics() {
    std::vector<int32_t> order(n_topics_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return topic_exposure_(a) > topic_exposure_(b);
    });
    auto sort_rows = [&](MatrixXd& m) {
        MatrixXd sorted(m.rows(), m.cols());
        for (int32_t k = 0; k < n_topics_; ++k) {
            sorted.row(k) = m.row(order[k]);
        }
        m = std::move(sorted);
    };
    sort_rows(e_beta_);
    if (topic_exposure_.size() == n_topics_) {
        VectorXd exposure(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            exposure(k) = topic_exposure_(order[k]);
        }
        topic_exposure_ = std::move(exposure);
    }
    if (!topic_names_.empty()) {
        std::vector<std::string> sorted(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            sorted[k] = topic_names_[order[k]];
        }
        topic_names_ = std::move(sorted);
    }
    refresh_cache();
}

void GammaPoissonTopicModel::sort_topics() {
    std::vector<int32_t> order(n_topics_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return topic_exposure_(a) > topic_exposure_(b);
    });
    auto sort_rows = [&](MatrixXd& m) {
        MatrixXd sorted(m.rows(), m.cols());
        for (int32_t k = 0; k < n_topics_; ++k) {
            sorted.row(k) = m.row(order[k]);
        }
        m = std::move(sorted);
    };
    sort_rows(e_beta_);
    if (topic_concentration_.size() == n_topics_) {
        VectorXd concentration(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            concentration(k) = topic_concentration_(order[k]);
        }
        topic_concentration_ = std::move(concentration);
    }
    if (running_counts_.rows() == n_topics_) sort_rows(running_counts_);
    if (running_delta_.rows() == n_topics_) sort_rows(running_delta_);
    if (topic_exposure_.size() == n_topics_) {
        VectorXd exposure(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            exposure(k) = topic_exposure_(order[k]);
        }
        topic_exposure_ = std::move(exposure);
    }
    if (!topic_names_.empty()) {
        std::vector<std::string> sorted(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            sorted[k] = topic_names_[order[k]];
        }
        topic_names_ = std::move(sorted);
    }
    refresh_cache();
}

void GammaPoissonTopicBase::write_model(const std::string& outFile,
    const std::vector<std::string>& featureNames) {
    std::ofstream out(outFile);
    if (!out) {
        error("%s: Error opening output file: %s", __func__, outFile.c_str());
    }
    out << "Feature\t";
    const auto& t_names = get_topic_names();
    out << t_names[0];
    for (size_t k = 1; k < t_names.size(); ++k) {
        out << "\t" << t_names[k];
    }
    out << "\n";
    out << std::scientific << std::setprecision(4);
    const RowMajorMatrixXd& model = get_model();
    for (int32_t w = 0; w < n_features_; ++w) {
        const std::string feature = w < static_cast<int32_t>(featureNames.size())
            ? featureNames[w] : std::to_string(w);
        out << feature;
        for (int32_t k = 0; k < n_topics_; ++k) {
            out << "\t" << model(k, w);
        }
        out << "\n";
    }
}

void GammaPoissonTopicModel::write_state(const std::string& outFile,
    const std::vector<std::string>& featureNames) {
    std::ofstream out(outFile);
    if (!out) {
        error("%s: Error opening output file: %s", __func__, outFile.c_str());
    }
    if (training_count_.size() != static_cast<size_t>(n_features_)
        || (feature_weights_active_
            && feature_weight_.size() != static_cast<size_t>(n_features_))) {
        error("%s: model is missing feature calibration metadata", __func__);
    }
    out << std::scientific
        << std::setprecision(std::numeric_limits<double>::max_digits10);
    out << "#punkst_gamma_pois_state_v8\n";
    out << "#n_topics\t" << n_topics_ << "\n";
    out << "#n_features\t" << n_features_ << "\n";
    out << "#total_doc_count\t" << total_doc_count_ << "\n";
    out << "#feature_weights_active\t"
        << (feature_weights_active_ ? 1 : 0) << "\n";
    out << "#exposure_convention\tobserved_total\n";
    out << "#inference_mode\t" << inference_mode_name(inference_mode_)
        << "\n";
    out << "#theta_concentration\t" << theta_concentration_ << "\n";
    out << "#dictionary_prior_mass\t" << dictionary_prior_mass_ << "\n";
    out << "#ownership_strength\t" << ownership_strength_ << "\n";
    out << "#ownership_mode\t" << ownership_mode_name(ownership_mode_)
        << "\n";
    out << "#ownership_fraction\t" << ownership_fraction_ << "\n";
    out << "#ownership_warmup_epochs\t" << ownership_warmup_epochs_ << "\n";
    out << "#ownership_ramp_epochs\t" << ownership_ramp_epochs_ << "\n";
    out << "#random_init_shape\t" << random_init_shape_ << "\n";
    out << "#learning_decay\t" << learning_decay_ << "\n";
    out << "#learning_offset\t" << learning_offset_ << "\n";
    out << "#update_count\t" << update_count_ << "\n";
    out << "#topic_names";
    for (const auto& name : get_topic_names()) out << "\t" << name;
    out << "\n";
    out << "#topic_prevalence_convention\texposure_weighted_posterior_mean\n";
    out << "#topic_exposure";
    for (int32_t k = 0; k < n_topics_; ++k) {
        out << "\t" << topic_exposure_(k);
    }
    if (topic_concentration_.size() == n_topics_) {
        out << "\n#topic_concentration";
        for (int32_t k = 0; k < n_topics_; ++k) {
            out << "\t" << topic_concentration_(k);
        }
    }
    if (has_dispersion_) {
        out << "\n#dispersion_tau";
        for (int32_t w = 0; w < n_features_; ++w) out << "\t" << tau_(w);
    }
    out << "\nFeature";
    for (int32_t k = 0; k < n_topics_; ++k) {
        out << "\tbeta_" << k;
    }
    out << "\ttraining_count";
    if (feature_weights_active_) out << "\tfeature_weight";
    out << "\n";
    for (int32_t w = 0; w < n_features_; ++w) {
        const std::string feature = w < static_cast<int32_t>(featureNames.size())
            ? featureNames[w] : std::to_string(w);
        out << feature;
        for (int32_t k = 0; k < n_topics_; ++k) {
            out << "\t" << e_beta_(k, w);
        }
        out << "\t" << training_count_[w];
        if (feature_weights_active_) out << "\t" << feature_weight_[w];
        out << "\n";
    }
}

GammaPoissonStateFeatureInfo GammaPoissonTopicModel::read_state_feature_info(
    const std::string& stateFile) {
    std::ifstream in(stateFile);
    if (!in) {
        error("%s: Error opening state file: %s", __func__, stateFile.c_str());
    }
    (void)read_gamma_pois_state_version(in, stateFile);
    GammaPoissonStateFeatureInfo info;
    int32_t n_topics = -1;
    int32_t n_features = -1;
    bool saw_feature_weights_active = false;
    std::string line;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const auto tok = split_ws(line.substr(1));
            if (tok.size() > 1 && tok[0] == "n_topics") {
                n_topics = std::stoi(tok[1]);
            } else if (tok.size() > 1 && tok[0] == "n_features") {
                n_features = std::stoi(tok[1]);
            } else if (tok.size() > 1
                    && tok[0] == "feature_weights_active") {
                const int32_t active = std::stoi(tok[1]);
                if (active != 0 && active != 1) {
                    error("%s: feature_weights_active must be 0 or 1 in %s",
                        __func__, stateFile.c_str());
                }
                info.feature_weights_active = active == 1;
                saw_feature_weights_active = true;
            }
            continue;
        }
        const auto tok = split_ws(line);
        if (tok.empty()) continue;
        if (tok[0] == "Feature") {
            saw_header = true;
            continue;
        }
        const size_t expected = 2 + static_cast<size_t>(n_topics)
            + (info.feature_weights_active ? 1 : 0);
        if (!saw_header || !saw_feature_weights_active
            || n_topics <= 0 || tok.size() != expected) {
            error("%s: Invalid Gamma-Poisson state feature row in %s",
                __func__, stateFile.c_str());
        }
        info.names.push_back(tok[0]);
        info.training_count.push_back(std::stod(tok[1 + n_topics]));
        if (info.feature_weights_active) {
            info.feature_weight.push_back(std::stod(tok[expected - 1]));
        }
    }
    if (info.names.empty()) {
        error("%s: No feature rows found in Gamma-Poisson state file: %s",
            __func__, stateFile.c_str());
    }
    if (n_features <= 0
        || static_cast<int32_t>(info.names.size()) != n_features) {
        error("%s: State feature count does not match metadata in %s",
            __func__, stateFile.c_str());
    }
    std::unordered_set<std::string> unique_names;
    for (const auto& name : info.names) {
        if (!unique_names.insert(name).second) {
            error("%s: duplicate feature %s in %s", __func__, name.c_str(),
                stateFile.c_str());
        }
    }
    double effective_total = 0.0;
    for (size_t w = 0; w < info.names.size(); ++w) {
        const double count = info.training_count[w];
        const double weight =
            info.feature_weights_active ? info.feature_weight[w] : 1.0;
        if (!std::isfinite(count) || count < 0.0
            || !std::isfinite(weight) || weight < 0.0) {
            error("%s: invalid feature calibration in %s",
                __func__, stateFile.c_str());
        }
        effective_total += count * weight;
    }
    if (!std::isfinite(effective_total) || effective_total <= 0.0) {
        error("%s: effective training total must be positive in %s",
            __func__, stateFile.c_str());
    }
    return info;
}

std::vector<std::string> GammaPoissonTopicModel::read_state_feature_names(
    const std::string& stateFile) {
    return read_state_feature_info(stateFile).names;
}

void GammaPoissonTopicModel::read_state(const std::string& stateFile) {
    std::ifstream in(stateFile);
    if (!in) {
        error("%s: Error opening state file: %s", __func__, stateFile.c_str());
    }
    (void)read_gamma_pois_state_version(in, stateFile);
    std::string line;
    std::vector<std::vector<double>> beta_rows;
    std::vector<double> training_count_vals, feature_weight_vals;
    std::vector<double> topic_exposure_vals;
    std::vector<double> topic_concentration_vals;
    std::vector<double> tau_vals;
    std::vector<std::string> topic_names;
    std::string exposure_convention;
    std::string topic_prevalence_convention;
    bool saw_theta_concentration = false;
    bool saw_dictionary_prior_mass = false;
    bool saw_ownership_strength = false;
    bool saw_ownership_mode = false;
    bool saw_ownership_fraction = false;
    bool saw_ownership_warmup = false;
    bool saw_ownership_ramp = false;
    bool saw_feature_weights_active = false;
    bool saw_header = false;
    bool saw_inference_mode = false;
    inference_mode_ = GammaPoissonInferenceMode::MapMean;
    ownership_mode_ = GammaPoissonOwnershipMode::Uniform;
    feature_names_.clear();
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const auto tok = split_ws(line.substr(1));
            if (tok.empty()) continue;
            if (tok[0] == "n_topics" && tok.size() > 1) n_topics_ = std::stoi(tok[1]);
            else if (tok[0] == "n_features" && tok.size() > 1) n_features_ = std::stoi(tok[1]);
            else if (tok[0] == "total_doc_count" && tok.size() > 1) total_doc_count_ = std::stoi(tok[1]);
            else if (tok[0] == "feature_weights_active" && tok.size() > 1) {
                const int32_t active = std::stoi(tok[1]);
                if (active != 0 && active != 1) {
                    error("%s: feature_weights_active must be 0 or 1 in %s",
                        __func__, stateFile.c_str());
                }
                feature_weights_active_ = active == 1;
                saw_feature_weights_active = true;
            }
            else if (tok[0] == "exposure_convention" && tok.size() > 1) {
                exposure_convention = tok[1];
            }
            else if (tok[0] == "topic_prevalence_convention"
                    && tok.size() > 1) {
                topic_prevalence_convention = tok[1];
            }
            else if (tok[0] == "inference_mode" && tok.size() > 1) {
                inference_mode_ = parse_inference_mode(tok[1], stateFile);
                saw_inference_mode = true;
            }
            else if (tok[0] == "theta_concentration" && tok.size() > 1) {
                theta_concentration_ = std::stod(tok[1]);
                saw_theta_concentration = true;
            }
            else if (tok[0] == "dictionary_prior_mass" && tok.size() > 1) {
                dictionary_prior_mass_ = std::stod(tok[1]);
                saw_dictionary_prior_mass = true;
            }
            else if (tok[0] == "ownership_strength" && tok.size() > 1) {
                ownership_strength_ = std::stod(tok[1]);
                saw_ownership_strength = true;
            }
            else if (tok[0] == "ownership_mode" && tok.size() > 1) {
                ownership_mode_ = parse_ownership_mode(tok[1], stateFile);
                saw_ownership_mode = true;
            }
            else if (tok[0] == "ownership_fraction" && tok.size() > 1) {
                ownership_fraction_ = std::stod(tok[1]);
                saw_ownership_fraction = true;
            }
            else if (tok[0] == "ownership_warmup_epochs" && tok.size() > 1) {
                ownership_warmup_epochs_ = std::stoi(tok[1]);
                saw_ownership_warmup = true;
            }
            else if (tok[0] == "ownership_ramp_epochs" && tok.size() > 1) {
                ownership_ramp_epochs_ = std::stoi(tok[1]);
                saw_ownership_ramp = true;
            }
            else if (tok[0] == "learning_decay" && tok.size() > 1) learning_decay_ = std::stod(tok[1]);
            else if (tok[0] == "learning_offset" && tok.size() > 1) learning_offset_ = std::stod(tok[1]);
            else if (tok[0] == "update_count" && tok.size() > 1) update_count_ = std::stoi(tok[1]);
            else if (tok[0] == "random_init_shape" && tok.size() > 1) {
                random_init_shape_ = std::stod(tok[1]);
            }
            else if (tok[0] == "topic_names") {
                topic_names.assign(tok.begin() + 1, tok.end());
            }
            else if (tok[0] == "topic_exposure") {
                topic_exposure_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) {
                    topic_exposure_vals.push_back(std::stod(tok[i]));
                }
            } else if (tok[0] == "topic_concentration") {
                topic_concentration_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) {
                    topic_concentration_vals.push_back(std::stod(tok[i]));
                }
            } else if (tok[0] == "dispersion_tau") {
                tau_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) tau_vals.push_back(std::stod(tok[i]));
            }
            continue;
        }
        const auto tok = split_ws(line);
        if (tok.empty()) continue;
        if (tok[0] == "Feature") {
            saw_header = true;
            continue;
        }
        if (!saw_header || n_topics_ <= 0) {
            error("%s: Invalid Gamma-Poisson state file header in %s", __func__, stateFile.c_str());
        }
        const size_t expected = 2 + static_cast<size_t>(n_topics_)
            + (feature_weights_active_ ? 1 : 0);
        if (tok.size() != expected) {
            error("%s: Invalid state row with %zu columns, expected %zu", __func__, tok.size(), expected);
        }
        std::vector<double> beta(n_topics_);
        feature_names_.push_back(tok[0]);
        size_t pos = 1;
        for (int32_t k = 0; k < n_topics_; ++k) beta[k] = std::stod(tok[pos++]);
        beta_rows.push_back(std::move(beta));
        training_count_vals.push_back(std::stod(tok[pos++]));
        if (feature_weights_active_) {
            feature_weight_vals.push_back(std::stod(tok[pos++]));
        }
    }
    if (!saw_feature_weights_active) {
        error("%s: Gamma-Poisson state is missing feature_weights_active in %s",
            __func__, stateFile.c_str());
    }
    if (!saw_inference_mode) {
        error("%s: Gamma-Poisson v8 state is missing inference_mode in %s",
            __func__, stateFile.c_str());
    }
    if (!saw_ownership_mode) {
        error("%s: Gamma-Poisson v8 state is missing ownership_mode in %s",
            __func__, stateFile.c_str());
    }
    if (n_topics_ <= 0 || n_features_ <= 0 || total_doc_count_ <= 0) {
        error("%s: Gamma-Poisson state has invalid dimensions or hyperparameters: %s",
            __func__, stateFile.c_str());
    }
    if (exposure_convention != "observed_total")
        error("%s: Gamma-Poisson state must use observed-total exposure: %s",
            __func__, stateFile.c_str());
    if (topic_prevalence_convention
            != "exposure_weighted_posterior_mean") {
        error("%s: Gamma-Poisson state lacks exposure-weighted topic prevalence: %s",
            __func__, stateFile.c_str());
    }
    if (!saw_theta_concentration || !std::isfinite(theta_concentration_)
        || theta_concentration_ <= 0.0) {
        error("%s: Gamma-Poisson state requires positive theta concentration: %s",
            __func__, stateFile.c_str());
    }
    if (!saw_dictionary_prior_mass
        || !std::isfinite(dictionary_prior_mass_)
        || dictionary_prior_mass_ < 0.0
        || !saw_ownership_strength
        || !std::isfinite(ownership_strength_)
        || ownership_strength_ < 0.0
        || !saw_ownership_fraction
        || !std::isfinite(ownership_fraction_)
        || ownership_fraction_ < 0.0 || ownership_fraction_ > 1.0
        || !saw_ownership_warmup || ownership_warmup_epochs_ < 0
        || !saw_ownership_ramp || ownership_ramp_epochs_ < 0
        || !std::isfinite(random_init_shape_) || random_init_shape_ <= 0.0
        || !std::isfinite(learning_decay_) || learning_decay_ <= 0.5
        || learning_decay_ > 1.0
        || !std::isfinite(learning_offset_) || learning_offset_ <= 0.0
        || update_count_ < 0) {
        error("%s: Gamma-Poisson state has invalid optimization metadata: %s",
            __func__, stateFile.c_str());
    }
    if (static_cast<int32_t>(beta_rows.size()) != n_features_) {
        error("%s: State file has %zu feature rows but metadata says %d",
            __func__, beta_rows.size(), n_features_);
    }
    {
        std::unordered_set<std::string> unique_names;
        for (const auto& name : feature_names_) {
            if (!unique_names.insert(name).second) {
                error("%s: Duplicate feature %s in state %s", __func__,
                    name.c_str(), stateFile.c_str());
            }
        }
    }
    e_beta_.resize(n_topics_, n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        for (int32_t k = 0; k < n_topics_; ++k) {
            const double value = beta_rows[w][k];
            if (!std::isfinite(value) || value <= 0.0) {
                error("%s: Dictionary entries must be positive in state %s",
                    __func__, stateFile.c_str());
            }
            e_beta_(k, w) = value;
        }
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        const double row_sum = e_beta_.row(k).sum();
        if (!std::isfinite(row_sum) || std::abs(row_sum - 1.0) > 1e-8) {
            error("%s: Dictionary row %d is not normalized in state %s",
                __func__, k, stateFile.c_str());
        }
    }
    training_count_.resize(n_features_);
    if (feature_weights_active_) feature_weight_.resize(n_features_);
    else feature_weight_.clear();
    double effective_training_total = 0.0;
    for (int32_t w = 0; w < n_features_; ++w) {
        training_count_[w] = training_count_vals[w];
        const double weight =
            feature_weights_active_ ? feature_weight_vals[w] : 1.0;
        if (!std::isfinite(training_count_[w])
            || training_count_[w] < 0.0
            || !std::isfinite(weight) || weight < 0.0) {
            error("%s: Invalid feature calibration in state %s",
                __func__, stateFile.c_str());
        }
        if (feature_weights_active_) feature_weight_[w] = weight;
        effective_training_total += training_count_[w] * weight;
    }
    if (!std::isfinite(effective_training_total)
        || effective_training_total <= 0.0) {
        error("%s: Effective training feature total is invalid in state %s",
            __func__, stateFile.c_str());
    }
    if (static_cast<int32_t>(topic_exposure_vals.size()) != n_topics_) {
        error("%s: Gamma-Poisson v8 state lacks topic exposure in %s",
            __func__, stateFile.c_str());
    }
    topic_exposure_.resize(n_topics_);
    double exposure_sum = 0.0;
    for (int32_t k = 0; k < n_topics_; ++k) {
        const double value = topic_exposure_vals[k];
        if (!std::isfinite(value) || value < 0.0) {
            error("%s: Invalid topic exposure in %s",
                __func__, stateFile.c_str());
        }
        topic_exposure_(k) = value;
        exposure_sum += value;
    }
    if (!std::isfinite(exposure_sum) || exposure_sum <= 0.0) {
        error("%s: Topic exposure must have positive finite mass in %s",
            __func__, stateFile.c_str());
    }
    topic_concentration_.resize(0);
    if (!topic_concentration_vals.empty()) {
        if (static_cast<int32_t>(topic_concentration_vals.size()) != n_topics_) {
            error("%s: State has %zu topic concentrations but expected %d",
                __func__, topic_concentration_vals.size(), n_topics_);
        }
        topic_concentration_.resize(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            const double value = topic_concentration_vals[k];
            if (!std::isfinite(value) || value <= 0.0) {
                error("%s: Invalid topic concentration in %s",
                    __func__, stateFile.c_str());
            }
            topic_concentration_(k) = value;
        }
    } else if (inference_mode_ == GammaPoissonInferenceMode::LdaCompatible) {
        error("%s: LDA-compatible state lacks topic concentration in %s",
            __func__, stateFile.c_str());
    }
    has_dispersion_ = false;
    tau_.resize(0);
    if (static_cast<int32_t>(tau_vals.size()) == n_features_) {
        set_feature_dispersion(tau_vals);
    } else if (!tau_vals.empty()) {
        error("%s: State file has %zu dispersion values but expected %d",
            __func__, tau_vals.size(), n_features_);
    }
    if (!topic_names.empty()) {
        if (static_cast<int32_t>(topic_names.size()) != n_topics_)
            error("%s: State has %zu topic names but expected %d",
                __func__, topic_names.size(), n_topics_);
        topic_names_ = std::move(topic_names);
    }
    running_stats_ready_ = false;
    refinement_active_ = false;
    ownership_schedule_active_ = false;
    worker_states_.reset();
}


void GammaPoisson4Hex::initialize(int32_t nTopics, int32_t seed, int32_t nThreads,
    int32_t verbose, double theta_concentration, double kappa, double tau0,
    int32_t totalDocCount, int32_t maxIter, double mDelta,
    double randomInitShape, const GammaPoissonMapOptions& mapOptions) {
    if (reader.features.size() != static_cast<size_t>(M_)) {
        featureNames.resize(M_);
        for (int32_t i = 0; i < M_; ++i) featureNames[i] = std::to_string(i);
    } else {
        featureNames = reader.features;
    }
    const std::vector<double>& sums = reader.getFeatureSums();
    model_ = std::make_unique<GammaPoissonTopicModel>(
        nTopics, M_, seed, nThreads, verbose, theta_concentration,
        kappa, tau0, totalDocCount,
        reader.readFullSums ? &sums : nullptr, randomInitShape, mapOptions);
    if (!reader.readFullSums) {
        error("%s: full effective feature totals are required for Gamma-Poisson state calibration",
            __func__);
    }
    const std::vector<double>& raw_sums = reader.getFeatureSumsRaw();
    if (raw_sums.size() != static_cast<size_t>(M_)) {
        error("%s: raw feature totals are required for Gamma-Poisson state calibration",
            __func__);
    }
    model_->set_training_calibration(
        raw_sums, reader.getFeatureWeights(), reader.hasFeatureWeights());
    model_->set_svb_parameters(maxIter, mDelta);
    initialized = true;
}

void GammaPoisson4Hex::initializeFromState(const std::string& stateFile,
    int32_t seed, int32_t nThreads, int32_t verbose, int32_t maxIter,
    double mDelta) {
    auto model = std::make_unique<GammaPoissonTopicModel>(
        stateFile, seed, nThreads, verbose);
    const auto& state_features = model->get_feature_names();
    if (static_cast<int32_t>(state_features.size()) != M_
        || reader.features.size() != static_cast<size_t>(M_)) {
        error("%s: Training state and current data have different feature dimensions",
            __func__);
    }
    for (int32_t w = 0; w < M_; ++w) {
        if (state_features[w] != reader.features[w]) {
            error("%s: Training state feature %d is %s but current data has %s",
                __func__, w, state_features[w].c_str(),
                reader.features[w].c_str());
        }
    }
    const auto& state_counts = model->get_training_count();
    const auto& current_counts = reader.getFeatureSumsRaw();
    if (state_counts.size() != current_counts.size()) {
        error("%s: Training state lacks matching feature calibration",
            __func__);
    }
    for (int32_t w = 0; w < M_; ++w) {
        const double scale = std::max({1.0, std::abs(state_counts[w]),
            std::abs(current_counts[w])});
        if (std::abs(state_counts[w] - current_counts[w]) > 1e-8 * scale) {
            error("%s: Training counts differ for feature %s",
                __func__, state_features[w].c_str());
        }
    }
    if (model->feature_weights_active() != reader.hasFeatureWeights()) {
        error("%s: Training state and current data disagree on feature weights",
            __func__);
    }
    if (model->feature_weights_active()) {
        const auto& state_weights = model->get_feature_weight();
        const auto& current_weights = reader.getFeatureWeights();
        if (state_weights.size() != current_weights.size()) {
            error("%s: Training state has invalid feature weights", __func__);
        }
        for (int32_t w = 0; w < M_; ++w) {
            if (std::abs(state_weights[w] - current_weights[w]) > 1e-12) {
                error("%s: Feature weights differ for %s", __func__,
                    state_features[w].c_str());
            }
        }
    }
    initialize_transform(std::move(model), maxIter, mDelta);
    model_->reset_running_statistics();
    notice("Initialized Gamma-Poisson refinement from state %s",
        stateFile.c_str());
}

void GammaPoisson4Hex::initializeFromModel(const std::string& modelFile) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    RowMajorMatrixXd values;
    std::vector<std::string> model_features, model_topics;
    read_matrix_from_file(
        modelFile, values, &model_features, &model_topics);
    if (values.cols() != model_->get_n_topics()) {
        error("%s: Initial model has %d topics but --n-topics is %d",
            __func__, static_cast<int32_t>(values.cols()),
            model_->get_n_topics());
    }

    std::unordered_map<std::string, int32_t> model_feature_index;
    for (int32_t w = 0; w < static_cast<int32_t>(model_features.size()); ++w) {
        if (!model_feature_index.emplace(model_features[w], w).second) {
            error("%s: Duplicate feature in initial model: %s",
                __func__, model_features[w].c_str());
        }
    }
    std::unordered_set<std::string> topic_seen;
    for (const auto& topic : model_topics) {
        if (!topic_seen.insert(topic).second) {
            error("%s: Duplicate topic in initial model: %s",
                __func__, topic.c_str());
        }
    }
    if (model_feature_index.size() != featureNames.size()) {
        error("%s: Initial model and fitted data must have the same retained feature set",
            __func__);
    }

    RowMajorMatrixXd profiles(model_->get_n_topics(), M_);
    for (int32_t w = 0; w < M_; ++w) {
        const auto it = model_feature_index.find(featureNames[w]);
        if (it == model_feature_index.end()) {
            error("%s: Initial model is missing retained feature: %s",
                __func__, featureNames[w].c_str());
        }
        profiles.col(w) = values.row(it->second).transpose();
    }
    model_->initialize_topic_profiles(profiles, model_topics);
    topicNames_ = model_topics;
    notice("Initialized Gamma-Poisson topics from %s", modelFile.c_str());
}

void GammaPoisson4Hex::setFeatureDispersion(const std::vector<double>& tau) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->set_feature_dispersion(tau);
    model_->reset_running_statistics();
}

void GammaPoisson4Hex::clearFeatureDispersion() {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->clear_feature_dispersion();
    model_->reset_running_statistics();
}

GammaPoissonDispersionResult GammaPoisson4Hex::estimateFeatureDispersion(
    const GammaPoissonDispersionOptions& options, const std::string& inFile,
    int32_t batchSize_, int32_t minCountTrain_, int32_t maxUnits) {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    GammaPoissonDispersionEstimator estimator(*model_, options);
    std::ifstream in(inFile);
    if (!in) error("%s: Error opening input file: %s", __func__, inFile.c_str());
    std::vector<Document> docs;
    std::vector<std::string> ids;
    int32_t processed = 0;
    while (processed < maxUnits) {
        const int32_t remaining = maxUnits == INT32_MAX ? INT32_MAX : maxUnits - processed;
        const bool more = readMinibatch(in, docs, ids, batchSize_, minCountTrain_, remaining);
        if (!docs.empty()) {
            estimator.accumulate(DocumentView(docs));
            processed += static_cast<int32_t>(docs.size());
        }
        if (!more || docs.empty()) break;
    }
    GammaPoissonDispersionResult result = estimator.finish();
    model_->set_feature_dispersion(result.tau);
    model_->reset_running_statistics();
    return result;
}

GammaPoissonDispersionResult GammaPoisson4Hex::estimateFeatureDispersion(
    const GammaPoissonDispersionOptions& options,
    punkst::DocumentBlockSource& source,
    int32_t batchSize_, int32_t maxUnits) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    GammaPoissonDispersionEstimator estimator(*model_, options);
    source.reset();
    punkst::DocumentBlock block;
    int32_t processed = 0;
    while (processed < maxUnits && source.next(block, batchSize_)) {
        if (block.counts.empty()) break;
        if (maxUnits != INT32_MAX
            && static_cast<int32_t>(block.counts.size())
                > maxUnits - processed) {
            block.counts.resize(maxUnits - processed);
        }
        estimator.accumulate(DocumentView(block.counts));
        processed += static_cast<int32_t>(block.counts.size());
    }
    GammaPoissonDispersionResult result = estimator.finish();
    model_->set_feature_dispersion(result.tau);
    model_->reset_running_statistics();
    return result;
}

GammaPoissonDispersionResult GammaPoisson4Hex::estimateFeatureDispersion(
    const GammaPoissonDispersionOptions& options,
    const std::vector<std::vector<Document>>& batches,
    int32_t maxUnits) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    GammaPoissonDispersionEstimator estimator(*model_, options);
    int32_t processed = 0;
    for (const std::vector<Document>& batch : batches) {
        if (processed >= maxUnits) break;
        const int32_t take = maxUnits == INT32_MAX
            ? static_cast<int32_t>(batch.size())
            : std::min<int32_t>(
                static_cast<int32_t>(batch.size()), maxUnits - processed);
        if (take <= 0) break;
        estimator.accumulate(DocumentView(batch.data(), take));
        processed += take;
    }
    GammaPoissonDispersionResult result = estimator.finish();
    model_->set_feature_dispersion(result.tau);
    model_->reset_running_statistics();
    return result;
}

GammaPoissonDispersionResult GammaPoisson4Hex::estimateFeatureDispersion10X(
    const GammaPoissonDispersionOptions& options, int32_t batchSize_, int32_t maxUnits) {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    if (!dge_cache_ready_) error("%s: 10X cache is not initialized", __func__);
    GammaPoissonDispersionEstimator estimator(*model_, options);
    int32_t processed = 0;
    std::vector<Document> docs;
    for (int32_t idx : dge_train_idx_cache_) {
        if (processed + static_cast<int32_t>(docs.size()) >= maxUnits) break;
        docs.push_back(dge_docs_cache_[idx]);
        if (static_cast<int32_t>(docs.size()) >= batchSize_) {
            estimator.accumulate(DocumentView(docs));
            processed += static_cast<int32_t>(docs.size());
            docs.clear();
        }
    }
    if (!docs.empty() && processed < maxUnits) {
        estimator.accumulate(DocumentView(docs));
        processed += static_cast<int32_t>(docs.size());
    }
    GammaPoissonDispersionResult result = estimator.finish();
    model_->set_feature_dispersion(result.tau);
    model_->reset_running_statistics();
    return result;
}

GammaPoissonDispersionResult GammaPoisson4Hex::estimateFeatureDispersion10X(
    const GammaPoissonDispersionOptions& options, DGEReader10X& dge,
    int32_t batchSize_, int32_t minCount, int32_t maxUnits) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    GammaPoissonDispersionEstimator estimator(*model_, options);
    dge.resetStream();
    std::vector<Document> docs;
    std::vector<int32_t> unit_indices;
    int32_t processed = 0;
    bool more = true;
    while (more && processed < maxUnits) {
        const int32_t remaining = maxUnits == INT32_MAX
            ? INT32_MAX : maxUnits - processed;
        more = dge.readMinibatch(
            docs, unit_indices, batchSize_, remaining, minCount);
        if (!docs.empty()) {
            for (Document& doc : docs) applyWeights(doc);
            estimator.accumulate(DocumentView(docs));
            processed += static_cast<int32_t>(docs.size());
        }
        if (docs.empty()) break;
    }
    dge.resetStream();
    GammaPoissonDispersionResult result = estimator.finish();
    model_->set_feature_dispersion(result.tau);
    model_->reset_running_statistics();
    return result;
}

void GammaPoisson4Hex::initialize_transform(
    std::unique_ptr<GammaPoissonTopicModel> model,
    int32_t maxIter, double mDelta,
    const std::vector<int32_t>& keptFeatures) {
    if (!model) {
        error("%s: Gamma-Poisson model is null", __func__);
    }
    model_ = std::move(model);
    if (!keptFeatures.empty()) {
        const double panelFraction = model_->restrict_features(keptFeatures);
        if (panelFraction < 1.0) {
            notice("Restricted Gamma-Poisson state to %zu measured features "
                "(%.6g of effective training counts)",
                keptFeatures.size(), panelFraction);
        }
    } else {
        model_->prepare_inference_cache();
    }
    model_->set_svb_parameters(maxIter, mDelta);
    M_ = model_->get_n_features();
    if (!model_->get_feature_names().empty()) {
        featureNames = model_->get_feature_names();
    } else if (reader.features.size() == static_cast<size_t>(M_)) {
        featureNames = reader.features;
    } else {
        featureNames.resize(M_);
        for (int32_t i = 0; i < M_; ++i) featureNames[i] = std::to_string(i);
    }
    initialized = true;
}

void GammaPoisson4Hex::setOwnershipAnnealing(double fraction) {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    model_->set_ownership_annealing(fraction);
}

void GammaPoisson4Hex::configureOwnershipAnnealing(int32_t warmupEpochs,
    int32_t rampEpochs, int32_t documentsPerEpoch, int64_t documentsSeen) {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    model_->configure_ownership_annealing(
        warmupEpochs, rampEpochs, documentsPerEpoch, documentsSeen);
}

void GammaPoisson4Hex::resetRunningStatistics() {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    model_->reset_running_statistics();
}

void GammaPoisson4Hex::beginFullRefinement() {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    model_->begin_full_refinement();
    refining_ = true;
}

void GammaPoisson4Hex::accumulateFullRefinement(
    const std::vector<Document>& batch) {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    model_->accumulate_full_refinement(batch);
}

bool GammaPoisson4Hex::finishFullRefinement(double tolerance) {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    refining_ = false;
    return model_->finish_full_refinement(tolerance);
}

GammaPoissonOptimizationDiagnostics
GammaPoisson4Hex::optimizationDiagnostics() const {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    return model_->optimization_diagnostics();
}

void GammaPoisson4Hex::writeModelToFile(const std::string& outFile) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->write_model(outFile, featureNames);
}

void GammaPoisson4Hex::writeStateToFile(const std::string& outFile) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->write_state(outFile, featureNames);
}

void GammaPoisson4Hex::getUnitHeaderCols(std::vector<std::string>& outCols) {
    outCols = get_topic_names();
}

const RowMajorMatrixXd& GammaPoisson4Hex::get_model_matrix() const {
    if (!model_) {
        return empty_model_;
    }
    return const_cast<GammaPoissonTopicModel*>(model_.get())->get_model();
}

RowMajorMatrixXd GammaPoisson4Hex::copy_model_matrix() const {
    if (!model_) {
        return RowMajorMatrixXd();
    }
    return const_cast<GammaPoissonTopicModel*>(model_.get())->copy_model();
}

const std::vector<std::string>& GammaPoisson4Hex::get_topic_names() {
    if (!model_) {
        topicNames_.clear();
        return topicNames_;
    }
    return model_->get_topic_names();
}

void GammaPoisson4Hex::do_partial_fit(const std::vector<Document>& batch) {
    if (refining_) model_->accumulate_full_refinement(batch);
    else model_->partial_fit(batch);
}

MatrixXd GammaPoisson4Hex::do_transform(DocumentView batch) {
    return model_->transform(batch);
}

RowMajorMatrixXd GammaPoisson4Hex::transformMeans(DocumentView batch) const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->transform(batch);
}

void GammaPoisson4Hex::transformWithPosteriors(DocumentView batch,
    RowMajorMatrixXd& topics,
    std::vector<GammaPoissonDocumentPosterior>& posteriors) const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->transform_with_posteriors(batch, topics, posteriors);
}

bool GammaPoisson4Hex::hasFeatureDispersion() const {
    return model_ && model_->has_feature_dispersion();
}

const VectorXd& GammaPoisson4Hex::getTopicCapacity() const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->get_topic_capacity();
}

const MatrixXd& GammaPoisson4Hex::getExpectedBeta() const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->get_expected_beta();
}

const MatrixXd& GammaPoisson4Hex::getBetaAllocationKernel() const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->get_beta_allocation_kernel();
}

void GammaPoisson4Hex::normalizeTopicAllocation(
    const Eigen::Ref<const VectorXd>& theta_log, int32_t feature,
    VectorXd& allocation) const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->normalize_topic_allocation(theta_log, feature, allocation);
}

const VectorXd& GammaPoisson4Hex::getFeatureDispersion() const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->get_feature_dispersion();
}

double GammaPoisson4Hex::getSizeFactor() const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->get_size_factor();
}

double GammaPoisson4Hex::getThetaPriorShape() const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->get_theta_prior_shape();
}

VectorXd GammaPoisson4Hex::getThetaPriorRate() const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->get_theta_prior_rate();
}

bool GammaPoisson4Hex::featureWeightsActive() const {
    return model_ && model_->feature_weights_active();
}

const std::vector<double>& GammaPoisson4Hex::getFeatureWeights() const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->get_feature_weight();
}

void GammaPoisson4Hex::getTopicAbundance(std::vector<double>& topic_weights) {
    get_topic_abundance(topic_weights);
}

void GammaPoisson4Hex::get_topic_abundance(std::vector<double>& topic_weights) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->get_topic_prevalence(topic_weights);
}
