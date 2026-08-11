#include "gamma_pois_topic.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

#include <unordered_map>
#include <unordered_set>

namespace {

double positive_or(double x, double fallback) {
    return x > 0.0 && std::isfinite(x) ? x : fallback;
}

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

void require_gamma_pois_state_v4(std::ifstream& in, const std::string& state_file) {
    std::string version;
    if (!std::getline(in, version) || version != "#punkst_gamma_pois_state_v4") {
        error("Gamma-Poisson state %s uses an unsupported format; refit the model "
            "with this version of punkst", state_file.c_str());
    }
}

} // namespace

int GammaPoissonTopicBase::normalize_seed(int seed) {
    return seed > 0 ? seed : static_cast<int>(std::random_device{}());
}

GammaPoissonTopicBase::GammaPoissonTopicBase(int32_t n_topics, int32_t n_features,
    int seed, int32_t nThreads, int32_t verbose, double beta_shape, double xi_shape,
    double xi_mean, double theta_concentration, double nu_shape, double nu_rate,
    double learning_decay, double learning_offset, int32_t total_doc_count,
    double size_factor, const std::vector<double>* feature_sums,
    double random_init_shape)
    : n_topics_(n_topics), n_features_(n_features), seed_(normalize_seed(seed)),
      nThreads_(nThreads), verbose_(verbose),
      total_doc_count_(total_doc_count > 0 ? total_doc_count : 1000000),
      a_(beta_shape),
      a0_(positive_or(xi_shape, 0.3)),
      b0_(xi_mean),
      e0_(positive_or(nu_shape, 1.0)),
      f0_(positive_or(nu_rate, 1.0)),
      learning_decay_(positive_or(learning_decay, 0.7)),
      learning_offset_(learning_offset >= 0.0 ? learning_offset : 10.0),
      size_factor_(positive_or(size_factor, 1.0)),
      random_init_shape_(random_init_shape) {
    if (!std::isfinite(a_) || a_ <= 0.0) {
        a_ = std::max(1.0 / static_cast<double>(std::max(1, n_topics_)), 0.01);
    }
    if (!std::isfinite(b0_) || b0_ <= 0.0) {
        b0_ = static_cast<double>(std::max(1, n_features_)) / size_factor_;
    }
    if (!std::isfinite(random_init_shape_) || random_init_shape_ <= 0.0) {
        throw std::invalid_argument(
            "Gamma-Poisson random initialization shape must be positive and finite");
    }
    random_engine_.seed(seed_);
    if (nu_rate <= 0.0) {
        f0_ = e0_ / positive_or(theta_concentration, 1.0);
    }
    set_nthreads(nThreads_);
    init_from_feature_sums(feature_sums);
}

GammaPoissonTopicModel::GammaPoissonTopicModel(int32_t n_topics, int32_t n_features,
    int seed, int32_t nThreads, int32_t verbose, double beta_shape, double xi_shape,
    double xi_mean, double theta_concentration,
    double nu_shape, double nu_rate,
    double learning_decay, double learning_offset, int32_t total_doc_count,
    double size_factor, bool symmetric_nu, double nu_max,
    const std::vector<double>* feature_sums, double random_init_shape)
    : GammaPoissonTopicBase(n_topics, n_features, seed, nThreads, verbose,
          beta_shape, xi_shape, xi_mean, theta_concentration, nu_shape, nu_rate,
          learning_decay, learning_offset, total_doc_count, size_factor,
          feature_sums, random_init_shape),
      symmetric_nu_(symmetric_nu), theta_concentration_(theta_concentration),
      nu_max_(nu_max) {
    if (!std::isfinite(theta_concentration_) || theta_concentration_ <= 0.0) {
        throw std::invalid_argument(
            "Gamma-Poisson theta concentration must be positive and finite");
    }
    if (!symmetric_nu_ && nu_max_ <= 0.0) {
        nu_max_ = 10.0 * theta_concentration_;
    }
    if (!symmetric_nu_ && !std::isfinite(nu_max_)) {
        throw std::invalid_argument("Gamma-Poisson nu cap must be finite");
    }
    const double theta_shape = theta_concentration_ / static_cast<double>(n_topics_);
    nu_shape_ = VectorXd::Constant(n_topics_, e0_ + total_doc_count_ * theta_shape);
    nu_rate_ = VectorXd::Constant(n_topics_,
        f0_ + static_cast<double>(total_doc_count_) / n_topics_);
    apply_nu_cap();
}

void GammaPoissonTopicModel::apply_nu_cap() {
    if (symmetric_nu_ || nu_max_ <= 0.0) {
        return;
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        nu_rate_(k) = std::max(nu_rate_(k), nu_shape_(k) / nu_max_);
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
}

void GammaPoissonTopicModel::clear_feature_dispersion() {
    has_dispersion_ = false;
    tau_.resize(0);
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
    MatrixXd beta_shape(n_topics_, new_features);
    MatrixXd beta_rate(n_topics_, new_features);
    VectorXd xi_shape(new_features);
    VectorXd xi_rate(new_features);
    VectorXd tau;
    if (has_dispersion_) tau.resize(new_features);
    std::vector<std::string> names;
    std::vector<double> counts(new_features);
    std::vector<double> weights;
    if (feature_weights_active_) weights.resize(new_features);
    names.reserve(new_features);
    for (int32_t j = 0; j < new_features; ++j) {
        const int32_t w = kept_features[j];
        beta_shape.col(j) = beta_shape_.col(w);
        beta_rate.col(j) = beta_rate_.col(w);
        xi_shape(j) = xi_shape_(w);
        xi_rate(j) = xi_rate_(w);
        if (has_dispersion_) tau(j) = tau_(w);
        if (feature_names_.size() == static_cast<size_t>(old_features)) {
            names.push_back(feature_names_[w]);
        }
        counts[j] = training_count_[w];
        if (feature_weights_active_) weights[j] = feature_weight_[w];
    }
    beta_shape_ = std::move(beta_shape);
    beta_rate_ = std::move(beta_rate);
    xi_shape_ = std::move(xi_shape);
    xi_rate_ = std::move(xi_rate);
    if (has_dispersion_) tau_ = std::move(tau);
    feature_names_ = std::move(names);
    training_count_ = std::move(counts);
    feature_weight_ = std::move(weights);
    n_features_ = new_features;
    size_factor_ *= panel_fraction;
    worker_states_.reset();
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

void GammaPoissonTopicBase::init_from_feature_sums(const std::vector<double>* feature_sums) {
    if (n_topics_ <= 0 || n_features_ <= 0) {
        throw std::invalid_argument("Gamma-Poisson model dimensions must be positive");
    }
    if (feature_sums && static_cast<int32_t>(feature_sums->size()) != n_features_) {
        throw std::invalid_argument(
            "Gamma-Poisson feature sums do not match the feature dimension");
    }
    beta_shape_.resize(n_topics_, n_features_);
    beta_rate_.resize(n_topics_, n_features_);
    topic_usage_ = VectorXd::Constant(n_topics_,
        size_factor_ * static_cast<double>(total_doc_count_) / static_cast<double>(std::max(1, n_topics_)));

    VectorXd feature_mean(n_features_);
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
            : size_factor_ / static_cast<double>(n_features_);
    }
    const double abundance_floor =
        std::max(size_factor_, 1.0) * 1e-12 / static_cast<double>(n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        const double abundance = feature_sums && total > 0.0
            ? (*feature_sums)[w] / total * size_factor_
            : size_factor_ / static_cast<double>(n_features_);
        feature_mean(w) = std::max(abundance, abundance_floor);
    }
    feature_mean *= size_factor_ / feature_mean.sum();

    // Use beta storage as the IPF work matrix, avoiding another K-by-V allocation.
    std::gamma_distribution<double> noise(
        random_init_shape_, 1.0 / random_init_shape_);
    for (int32_t k = 0; k < n_topics_; ++k) {
        for (int32_t w = 0; w < n_features_; ++w) {
            beta_rate_(k, w) = std::max(noise(random_engine_), 1e-300);
        }
    }

    VectorXd row_sums(n_topics_);
    bool converged = false;
    for (int32_t iter = 0; iter < 100; ++iter) {
        row_sums = beta_rate_.rowwise().sum();
        for (int32_t k = 0; k < n_topics_; ++k) {
            if (!std::isfinite(row_sums(k)) || row_sums(k) <= 0.0) {
                throw std::runtime_error("Non-finite Gamma-Poisson beta initialization");
            }
            beta_rate_.row(k) *= size_factor_ / row_sums(k);
        }
        for (int32_t w = 0; w < n_features_; ++w) {
            const double column_sum = beta_rate_.col(w).sum();
            if (!std::isfinite(column_sum) || column_sum <= 0.0) {
                throw std::runtime_error("Non-finite Gamma-Poisson beta initialization");
            }
            beta_rate_.col(w) *= n_topics_ * feature_mean(w) / column_sum;
        }
        row_sums = beta_rate_.rowwise().sum();
        double max_relative_error = 0.0;
        for (int32_t k = 0; k < n_topics_; ++k) {
            max_relative_error = std::max(max_relative_error,
                std::abs(row_sums(k) / size_factor_ - 1.0));
        }
        if (std::isfinite(max_relative_error) && max_relative_error <= 1e-8) {
            converged = true;
            break;
        }
    }
    if (!converged) {
        throw std::runtime_error("Gamma-Poisson beta initialization did not converge");
    }

    for (int32_t k = 0; k < n_topics_; ++k) {
        for (int32_t w = 0; w < n_features_; ++w) {
            const double beta_mean = beta_rate_(k, w);
            beta_shape_(k, w) = a_;
            beta_rate_(k, w) = a_ / beta_mean;
        }
    }
    xi_shape_ = VectorXd::Constant(n_features_, a0_ + a_ * n_topics_);
    xi_rate_.resize(n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        xi_rate_(w) = a0_ / b0_ + a_ * feature_mean(w) * n_topics_;
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

    for (int32_t k = 0; k < n_topics_; ++k) {
        for (int32_t w = 0; w < n_features_; ++w) {
            const double mean = size_factor_ * normalized(k, w);
            beta_shape_(k, w) = a_;
            beta_rate_(k, w) = a_ / mean;
        }
    }
    xi_shape_ = VectorXd::Constant(n_features_, a0_ + a_ * n_topics_);
    for (int32_t w = 0; w < n_features_; ++w) {
        double beta_sum = 0.0;
        for (int32_t k = 0; k < n_topics_; ++k) {
            beta_sum += size_factor_ * normalized(k, w);
        }
        xi_rate_(w) = a0_ / b0_ + a_ * beta_sum;
    }
    topic_usage_ = VectorXd::Constant(n_topics_,
        size_factor_ * static_cast<double>(total_doc_count_)
            / static_cast<double>(n_topics_));
    if (!topic_names.empty()) topic_names_ = topic_names;
    refresh_cache();
}

void GammaPoissonTopicBase::refresh_cache() {
    e_beta_.resize(n_topics_, n_features_); // E[\beta_{kw}]
    beta_kernel_.resize(n_topics_, n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        double max_log = -std::numeric_limits<double>::infinity();
        for (int32_t k = 0; k < n_topics_; ++k) {
            beta_shape_(k, w) = std::max(beta_shape_(k, w), 1e-12);
            beta_rate_(k, w) = std::max(beta_rate_(k, w), 1e-12);
            e_beta_(k, w) = beta_shape_(k, w) / beta_rate_(k, w);
            beta_kernel_(k, w) =
                psi(beta_shape_(k, w)) - std::log(beta_rate_(k, w));
            max_log = std::max(max_log, beta_kernel_(k, w));
        }
        for (int32_t k = 0; k < n_topics_; ++k) {
            beta_kernel_(k, w) = std::exp(beta_kernel_(k, w) - max_log);
        }
    }
    topic_capacity_ = e_beta_.rowwise().sum();
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
    for (int32_t k = 0; k < n_topics_; ++k) {
        const double denom = std::max(topic_capacity_(k), 1e-300);
        model_phi_.row(k) /= denom;
    }
    model_cache_dirty_ = false;
}

double GammaPoissonTopicBase::doc_exposure(const Document& doc) const {
    const double len = doc_sum_const(doc);
    if (len <= 0.0) {
        return 0.0;
    }
    return len / size_factor_;
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
        beta_rate_correction.setZero(n_topics, n_features);
    }
    ctheta.setZero(n_topics);
    theta.setZero(n_topics);
    iteration_sum = 0;
    documents = 0;
    failed = 0;
}

template <bool WithDispersion>
int32_t GammaPoissonTopicModel::fit_one_document(VectorXd& theta_shape,
    VectorXd& theta_rate, LocalWorkspace& workspace, const Document& doc) const {
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
    for (int32_t k = 0; k < n_topics_; ++k) {
        theta_shape(k) += doc_total / static_cast<double>(n_topics_);
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
        for (int32_t j = 0; j < n_ids; ++j) {
            const uint32_t w = doc.ids[j];
            double max_log = -std::numeric_limits<double>::infinity();
            for (int32_t k = 0; k < n_topics_; ++k) {
                workspace.theta_kernel(k) = workspace.theta_log(k)
                    + psi(beta_shape_(k, w)) - std::log(beta_rate_(k, w));
                max_log = std::max(max_log, workspace.theta_kernel(k));
            }
            double norm = 0.0;
            for (int32_t k = 0; k < n_topics_; ++k) {
                workspace.theta_kernel(k) =
                    std::exp(workspace.theta_kernel(k) - max_log);
                norm += workspace.theta_kernel(k);
            }
            norm = std::max(norm, eps_);
            workspace.assigned.noalias() +=
                counts(j) / norm * workspace.theta_kernel;
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
    const Document& doc) const {
    const int32_t iter = fit_one_document<WithDispersion>(
        state.theta_shape, state.theta_rate, state.workspace, doc);
    ++state.documents;
    state.iteration_sum += iter;
    state.failed += iter >= max_doc_update_iter_;

    state.workspace.e_theta =
        state.theta_shape.array() / state.theta_rate.array().max(1e-12);
    state.theta += state.workspace.e_theta;
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
        for (int32_t j = 0; j < n_ids; ++j) {
            const uint32_t w = doc.ids[j];
            double max_log = -std::numeric_limits<double>::infinity();
            for (int32_t k = 0; k < n_topics_; ++k) {
                state.workspace.assigned(k) = state.workspace.theta_log(k)
                    + psi(beta_shape_(k, w)) - std::log(beta_rate_(k, w));
                max_log = std::max(max_log, state.workspace.assigned(k));
            }
            double norm = 0.0;
            for (int32_t k = 0; k < n_topics_; ++k) {
                state.workspace.assigned(k) =
                    std::exp(state.workspace.assigned(k) - max_log);
                norm += state.workspace.assigned(k);
            }
            state.ss.col(w).noalias() +=
                counts(j) / std::max(norm, eps_) * state.workspace.assigned;
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
            state.beta_rate_correction.col(doc.ids[j]).noalias() +=
                correction * state.workspace.e_theta;
        }
    }
}

template <bool WithDispersion>
void GammaPoissonTopicModel::partial_fit_impl(const std::vector<Document>& docs) {
    const int32_t minibatch_size = static_cast<int32_t>(docs.size());
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
                accumulate_document<WithDispersion>(state, docs[d]);
            }
        });

    std::vector<WorkerState*> active_workers;
    VectorXd ctheta = VectorXd::Zero(n_topics_);
    VectorXd theta = VectorXd::Zero(n_topics_);
    int64_t iteration_sum = 0;
    int32_t document_count = 0;
    int32_t failed = 0;
    for (WorkerState& state : *worker_states_) {
        if (state.generation != worker_generation_) {
            continue;
        }
        active_workers.push_back(&state);
        ctheta += state.ctheta;
        theta += state.theta;
        iteration_sum += state.iteration_sum;
        document_count += state.documents;
        failed += state.failed;
    }

    ++update_count_;
    const double rho = std::pow(learning_offset_ + update_count_, -learning_decay_);
    const double scale = static_cast<double>(total_doc_count_) / static_cast<double>(minibatch_size);
    VectorXd xi_mean(n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        xi_mean(w) = expected_xi(w);
    }
    const VectorXd scaled_ctheta = scale * ctheta;
    const double old_weight = 1.0 - rho;
    auto update_features = [&](const tbb::blocked_range<int32_t>& range) {
        VectorXd ss(n_topics_);
        VectorXd rate_correction;
        if constexpr (WithDispersion) {
            rate_correction.resize(n_topics_);
        }
        for (int32_t w = range.begin(); w < range.end(); ++w) {
            ss.setZero();
            if constexpr (WithDispersion) {
                rate_correction.setZero();
            }
            for (const WorkerState* state : active_workers) {
                ss += state->ss.col(w);
                if constexpr (WithDispersion) {
                    rate_correction += state->beta_rate_correction.col(w);
                }
            }

            double beta_sum = 0.0;
            double max_log = -std::numeric_limits<double>::infinity();
            for (int32_t k = 0; k < n_topics_; ++k) {
                const double target_shape = a_ + scale * ss(k);
                double target_rate = a_ * xi_mean(w) + scaled_ctheta(k);
                if constexpr (WithDispersion) {
                    target_rate += scale * rate_correction(k);
                }
                beta_shape_(k, w) =
                    old_weight * beta_shape_(k, w) + rho * target_shape;
                beta_rate_(k, w) = old_weight * beta_rate_(k, w)
                    + rho * std::max(target_rate, 1e-12);
                beta_shape_(k, w) = std::max(beta_shape_(k, w), 1e-12);
                beta_rate_(k, w) = std::max(beta_rate_(k, w), 1e-12);
                e_beta_(k, w) = beta_shape_(k, w) / beta_rate_(k, w);
                beta_sum += e_beta_(k, w);
                beta_kernel_(k, w) =
                    psi(beta_shape_(k, w)) - std::log(beta_rate_(k, w));
                max_log = std::max(max_log, beta_kernel_(k, w));
            }
            for (int32_t k = 0; k < n_topics_; ++k) {
                beta_kernel_(k, w) =
                    std::exp(beta_kernel_(k, w) - max_log);
            }
            xi_shape_(w) = a0_ + a_ * n_topics_;
            xi_rate_(w) = a0_ / b0_ + a_ * beta_sum;
        }
    };
    if (nThreads_ == 1 || n_features_ < 64) {
        update_features(tbb::blocked_range<int32_t>(0, n_features_));
    } else {
        tbb::parallel_for(
            tbb::blocked_range<int32_t>(0, n_features_), update_features);
    }
    topic_capacity_ = e_beta_.rowwise().sum();
    model_cache_dirty_ = true;

    if (!symmetric_nu_) {
        for (int32_t k = 0; k < n_topics_; ++k) {
            nu_shape_(k) = e0_ + total_doc_count_
                * theta_concentration_ / static_cast<double>(n_topics_);
            const double target_rate = f0_ + scale * theta(k);
            nu_rate_(k) = (1.0 - rho) * nu_rate_(k) + rho * std::max(target_rate, 1e-12);
        }
        apply_nu_cap();
    }
    VectorXd target_usage = scale * theta.array() * topic_capacity_.array();
    if (topic_usage_.size() != n_topics_ || topic_usage_.sum() <= 0.0) {
        topic_usage_ = target_usage;
    } else {
        topic_usage_ = (1.0 - rho) * topic_usage_ + rho * target_usage;
    }

    if (verbose_ > 0 && document_count > 0) {
        const double avg =
            static_cast<double>(iteration_sum) / document_count;
        notice("Gamma-Poisson partial fit: %d documents. Average iterations per doc: %.2f, %d documents did not reach mean change %.1e in %d iterations.",
            minibatch_size, avg, failed, mean_change_tol_, max_doc_update_iter_);
    }
}

void GammaPoissonTopicModel::partial_fit(const std::vector<Document>& docs) {
    if (has_dispersion_) {
        partial_fit_impl<true>(docs);
    } else {
        partial_fit_impl<false>(docs);
    }
}

RowMajorMatrixXd GammaPoissonTopicModel::transform(DocumentView docs) {
    const int32_t n_docs = static_cast<int32_t>(docs.size());
    RowMajorMatrixXd topics(n_docs, n_topics_);
    auto process_doc = [&](int32_t d) {
        VectorXd shape;
        VectorXd rate;
        fit_one_document(shape, rate, docs[d]);
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
        infer_document_posterior(docs[d], posteriors[d]);
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

void GammaPoissonTopicBase::get_topic_abundance(std::vector<double>& weights) const {
    weights.resize(n_topics_);
    double total = topic_usage_.sum();
    if (total <= 0.0) {
        std::fill(weights.begin(), weights.end(), 1.0 / static_cast<double>(n_topics_));
        return;
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        weights[k] = topic_usage_(k) / total;
    }
}

void GammaPoissonTopicBase::sort_topics() {
    std::vector<int32_t> order(n_topics_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return topic_usage_(a) > topic_usage_(b);
    });
    auto sort_rows = [&](MatrixXd& m) {
        MatrixXd sorted(m.rows(), m.cols());
        for (int32_t k = 0; k < n_topics_; ++k) {
            sorted.row(k) = m.row(order[k]);
        }
        m = std::move(sorted);
    };
    sort_rows(beta_shape_);
    sort_rows(beta_rate_);
    if (topic_usage_.size() == n_topics_) {
        VectorXd usage(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            usage(k) = topic_usage_(order[k]);
        }
        topic_usage_ = std::move(usage);
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
        return topic_usage_(a) > topic_usage_(b);
    });
    auto sort_rows = [&](MatrixXd& m) {
        MatrixXd sorted(m.rows(), m.cols());
        for (int32_t k = 0; k < n_topics_; ++k) {
            sorted.row(k) = m.row(order[k]);
        }
        m = std::move(sorted);
    };
    sort_rows(beta_shape_);
    sort_rows(beta_rate_);
    if (!symmetric_nu_) {
        VectorXd ns(n_topics_), nr(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            ns(k) = nu_shape_(order[k]);
            nr(k) = nu_rate_(order[k]);
        }
        nu_shape_ = std::move(ns);
        nu_rate_ = std::move(nr);
    }
    if (topic_usage_.size() == n_topics_) {
        VectorXd usage(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            usage(k) = topic_usage_(order[k]);
        }
        topic_usage_ = std::move(usage);
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
    out << "#punkst_gamma_pois_state_v4\n";
    out << "#n_topics\t" << n_topics_ << "\n";
    out << "#n_features\t" << n_features_ << "\n";
    out << "#total_doc_count\t" << total_doc_count_ << "\n";
    out << "#size_factor\t" << size_factor_ << "\n";
    out << "#feature_weights_active\t"
        << (feature_weights_active_ ? 1 : 0) << "\n";
    out << "#beta_shape_prior\t" << a_ << "\n";
    out << "#beta_parameterization\tcount_scale_conditional_rate_a_xi\n";
    out << "#xi_shape_prior\t" << a0_ << "\n";
    out << "#xi_mean_prior\t" << b0_ << "\n";
    out << "#theta_prior_mode\t"
        << (symmetric_nu_ ? "symmetric_concentration" : "eb_rate") << "\n";
    out << "#theta_concentration_prior\t" << theta_concentration_ << "\n";
    if (!symmetric_nu_) {
        out << "#nu_shape_prior\t" << e0_ << "\n";
        out << "#nu_rate_prior\t" << f0_ << "\n";
    }
    out << "#learning_decay\t" << learning_decay_ << "\n";
    out << "#learning_offset\t" << learning_offset_ << "\n";
    out << "#update_count\t" << update_count_ << "\n";
    out << "#symmetric_nu\t" << (symmetric_nu_ ? 1 : 0) << "\n";
    if (!symmetric_nu_) {
        out << "#nu_mean_cap\t" << nu_max_ << "\n";
        out << "#nu_shape";
        for (int32_t k = 0; k < n_topics_; ++k) out << "\t" << nu_shape_(k);
        out << "\n#nu_rate";
        for (int32_t k = 0; k < n_topics_; ++k) out << "\t" << nu_rate_(k);
        out << "\n";
    }
    out << "#topic_usage";
    for (int32_t k = 0; k < n_topics_; ++k) out << "\t" << topic_usage_(k);
    if (has_dispersion_) {
        out << "\n#dispersion_tau";
        for (int32_t w = 0; w < n_features_; ++w) out << "\t" << tau_(w);
    }
    out << "\nFeature";
    for (int32_t k = 0; k < n_topics_; ++k) {
        out << "\tbeta_shape_" << k << "\tbeta_rate_" << k;
    }
    out << "\txi_shape\txi_rate\ttraining_count";
    if (feature_weights_active_) out << "\tfeature_weight";
    out << "\n";
    for (int32_t w = 0; w < n_features_; ++w) {
        const std::string feature = w < static_cast<int32_t>(featureNames.size())
            ? featureNames[w] : std::to_string(w);
        out << feature;
        for (int32_t k = 0; k < n_topics_; ++k) {
            out << "\t" << beta_shape_(k, w) << "\t" << beta_rate_(k, w);
        }
        out << "\t" << xi_shape_(w) << "\t" << xi_rate_(w)
            << "\t" << training_count_[w];
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
    require_gamma_pois_state_v4(in, stateFile);
    GammaPoissonStateFeatureInfo info;
    int32_t n_topics = -1;
    bool saw_feature_weights_active = false;
    std::string line;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            const auto tok = split_ws(line.substr(1));
            if (tok.size() > 1 && tok[0] == "n_topics") {
                n_topics = std::stoi(tok[1]);
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
        const size_t expected =
            1 + static_cast<size_t>(2 * n_topics) + 3
            + (info.feature_weights_active ? 1 : 0);
        if (!saw_header || !saw_feature_weights_active
            || n_topics <= 0 || tok.size() != expected) {
            error("%s: Invalid Gamma-Poisson state feature row in %s",
                __func__, stateFile.c_str());
        }
        info.names.push_back(tok[0]);
        info.training_count.push_back(std::stod(
            tok[expected - (info.feature_weights_active ? 2 : 1)]));
        if (info.feature_weights_active) {
            info.feature_weight.push_back(std::stod(tok[expected - 1]));
        }
    }
    if (info.names.empty()) {
        error("%s: No feature rows found in Gamma-Poisson state file: %s",
            __func__, stateFile.c_str());
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
    require_gamma_pois_state_v4(in, stateFile);
    std::string line;
    std::vector<std::vector<double>> bshape_rows;
    std::vector<std::vector<double>> brate_rows;
    std::vector<double> xi_shape_vals, xi_rate_vals;
    std::vector<double> training_count_vals, feature_weight_vals;
    std::vector<double> nu_shape_vals, nu_rate_vals;
    std::vector<double> topic_usage_vals;
    std::vector<double> tau_vals;
    std::string beta_parameterization;
    std::string theta_prior_mode;
    bool saw_theta_concentration = false;
    bool saw_symmetric_nu = false;
    bool saw_nu_shape_prior = false;
    bool saw_nu_rate_prior = false;
    bool saw_nu_mean_cap = false;
    bool saw_feature_weights_active = false;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            std::string payload = line.substr(1);
            auto tok = split_ws(payload);
            if (tok.empty()) continue;
            if (tok[0] == "n_topics" && tok.size() > 1) n_topics_ = std::stoi(tok[1]);
            else if (tok[0] == "n_features" && tok.size() > 1) n_features_ = std::stoi(tok[1]);
            else if (tok[0] == "total_doc_count" && tok.size() > 1) total_doc_count_ = std::stoi(tok[1]);
            else if (tok[0] == "size_factor" && tok.size() > 1) size_factor_ = std::stod(tok[1]);
            else if (tok[0] == "feature_weights_active" && tok.size() > 1) {
                const int32_t active = std::stoi(tok[1]);
                if (active != 0 && active != 1) {
                    error("%s: feature_weights_active must be 0 or 1 in %s",
                        __func__, stateFile.c_str());
                }
                feature_weights_active_ = active == 1;
                saw_feature_weights_active = true;
            }
            else if (tok[0] == "beta_shape_prior" && tok.size() > 1) a_ = std::stod(tok[1]);
            else if (tok[0] == "beta_parameterization" && tok.size() > 1) {
                beta_parameterization = tok[1];
            }
            else if (tok[0] == "xi_shape_prior" && tok.size() > 1) a0_ = std::stod(tok[1]);
            else if (tok[0] == "xi_mean_prior" && tok.size() > 1) b0_ = std::stod(tok[1]);
            else if (tok[0] == "theta_prior_mode" && tok.size() > 1) theta_prior_mode = tok[1];
            else if (tok[0] == "theta_concentration_prior" && tok.size() > 1) {
                theta_concentration_ = std::stod(tok[1]);
                saw_theta_concentration = true;
            }
            else if (tok[0] == "nu_shape_prior" && tok.size() > 1) {
                e0_ = std::stod(tok[1]);
                saw_nu_shape_prior = true;
            } else if (tok[0] == "nu_rate_prior" && tok.size() > 1) {
                f0_ = std::stod(tok[1]);
                saw_nu_rate_prior = true;
            }
            else if (tok[0] == "learning_decay" && tok.size() > 1) learning_decay_ = std::stod(tok[1]);
            else if (tok[0] == "learning_offset" && tok.size() > 1) learning_offset_ = std::stod(tok[1]);
            else if (tok[0] == "update_count" && tok.size() > 1) update_count_ = std::stoi(tok[1]);
            else if (tok[0] == "symmetric_nu" && tok.size() > 1) {
                symmetric_nu_ = std::stoi(tok[1]) != 0;
                saw_symmetric_nu = true;
            }
            else if (tok[0] == "nu_mean_cap" && tok.size() > 1) {
                nu_max_ = std::stod(tok[1]);
                saw_nu_mean_cap = true;
            }
            else if (tok[0] == "nu_shape") {
                nu_shape_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) nu_shape_vals.push_back(std::stod(tok[i]));
            } else if (tok[0] == "nu_rate") {
                nu_rate_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) nu_rate_vals.push_back(std::stod(tok[i]));
            } else if (tok[0] == "topic_usage") {
                topic_usage_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) topic_usage_vals.push_back(std::stod(tok[i]));
            } else if (tok[0] == "dispersion_tau") {
                tau_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) tau_vals.push_back(std::stod(tok[i]));
            }
            continue;
        }
        auto tok = split_ws(line);
        if (tok.empty()) {
            continue;
        }
        if (tok[0] == "Feature") {
            saw_header = true;
            continue;
        }
        if (!saw_header || n_topics_ <= 0) {
            error("%s: Invalid Gamma-Poisson state file header in %s", __func__, stateFile.c_str());
        }
        const size_t expected = 1 + static_cast<size_t>(2 * n_topics_) + 3
            + (feature_weights_active_ ? 1 : 0);
        if (tok.size() != expected) {
            error("%s: Invalid state row with %zu columns, expected %zu", __func__, tok.size(), expected);
        }
        std::vector<double> sh(n_topics_), rt(n_topics_);
        feature_names_.push_back(tok[0]);
        size_t pos = 1;
        for (int32_t k = 0; k < n_topics_; ++k) {
            sh[k] = std::stod(tok[pos++]);
            rt[k] = std::stod(tok[pos++]);
        }
        bshape_rows.push_back(std::move(sh));
        brate_rows.push_back(std::move(rt));
        xi_shape_vals.push_back(std::stod(tok[pos++]));
        xi_rate_vals.push_back(std::stod(tok[pos++]));
        training_count_vals.push_back(std::stod(tok[pos++]));
        if (feature_weights_active_) {
            feature_weight_vals.push_back(std::stod(tok[pos++]));
        }
    }
    if (!saw_symmetric_nu) {
        error("%s: Gamma-Poisson state is missing symmetric_nu in %s",
            __func__, stateFile.c_str());
    }
    if (!saw_feature_weights_active) {
        error("%s: Gamma-Poisson state is missing feature_weights_active in %s",
            __func__, stateFile.c_str());
    }
    if (n_topics_ <= 0 || n_features_ <= 0 || total_doc_count_ <= 0
        || !std::isfinite(size_factor_) || size_factor_ <= 0.0
        || !std::isfinite(a_) || a_ <= 0.0
        || !std::isfinite(a0_) || a0_ <= 0.0
        || !std::isfinite(b0_) || b0_ <= 0.0) {
        error("%s: Gamma-Poisson state has invalid dimensions or hyperparameters: %s",
            __func__, stateFile.c_str());
    }
    const bool symmetric_mode = theta_prior_mode == "symmetric_concentration";
    const bool eb_mode = theta_prior_mode == "eb_rate";
    if ((!symmetric_mode && !eb_mode) || symmetric_mode != symmetric_nu_) {
        error("%s: Invalid or inconsistent theta prior mode in Gamma-Poisson state: %s",
            __func__, stateFile.c_str());
    }
    if (beta_parameterization != "count_scale_conditional_rate_a_xi") {
        error("%s: Gamma-Poisson state has an invalid beta parameterization: %s",
            __func__, stateFile.c_str());
    }
    if (!saw_theta_concentration || !std::isfinite(theta_concentration_)
        || theta_concentration_ <= 0.0) {
        error("%s: Gamma-Poisson state requires a positive finite theta concentration: %s",
            __func__, stateFile.c_str());
    }
    if (eb_mode && (
        !saw_nu_shape_prior || !std::isfinite(e0_) || e0_ <= 0.0
        || !saw_nu_rate_prior || !std::isfinite(f0_) || f0_ <= 0.0
        || !saw_nu_mean_cap || !std::isfinite(nu_max_) || nu_max_ <= 0.0
        || static_cast<int32_t>(nu_shape_vals.size()) != n_topics_
        || static_cast<int32_t>(nu_rate_vals.size()) != n_topics_)) {
        error("%s: Empirical-Bayes Gamma-Poisson state has invalid theta or nu metadata: %s",
            __func__, stateFile.c_str());
    }
    if (n_features_ <= 0) {
        n_features_ = static_cast<int32_t>(bshape_rows.size());
    }
    if (static_cast<int32_t>(bshape_rows.size()) != n_features_) {
        error("%s: State file has %zu feature rows but metadata says %d",
            __func__, bshape_rows.size(), n_features_);
    }
    beta_shape_.resize(n_topics_, n_features_);
    beta_rate_.resize(n_topics_, n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        for (int32_t k = 0; k < n_topics_; ++k) {
            beta_shape_(k, w) = bshape_rows[w][k];
            beta_rate_(k, w) = brate_rows[w][k];
        }
    }
    xi_shape_.resize(n_features_);
    xi_rate_.resize(n_features_);
    training_count_.resize(n_features_);
    if (feature_weights_active_) feature_weight_.resize(n_features_);
    else feature_weight_.clear();
    double effective_training_total = 0.0;
    for (int32_t w = 0; w < n_features_; ++w) {
        xi_shape_(w) = xi_shape_vals[w];
        xi_rate_(w) = xi_rate_vals[w];
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
    const double theta_shape = theta_concentration_ / static_cast<double>(n_topics_);
    nu_shape_ = VectorXd::Constant(n_topics_, e0_ + theta_shape * total_doc_count_);
    nu_rate_ = VectorXd::Constant(n_topics_,
        f0_ + static_cast<double>(total_doc_count_) / n_topics_);
    if (static_cast<int32_t>(nu_shape_vals.size()) == n_topics_) {
        for (int32_t k = 0; k < n_topics_; ++k) nu_shape_(k) = nu_shape_vals[k];
    }
    if (static_cast<int32_t>(nu_rate_vals.size()) == n_topics_) {
        for (int32_t k = 0; k < n_topics_; ++k) nu_rate_(k) = nu_rate_vals[k];
    }
    if (!symmetric_nu_) {
        for (int32_t k = 0; k < n_topics_; ++k) {
            if (!std::isfinite(nu_shape_(k)) || nu_shape_(k) <= 0.0
                || !std::isfinite(nu_rate_(k)) || nu_rate_(k) <= 0.0) {
                error("%s: Invalid empirical-Bayes nu posterior in state %s",
                    __func__, stateFile.c_str());
            }
        }
        apply_nu_cap();
    }
    topic_usage_ = VectorXd::Constant(n_topics_, 1.0);
    if (static_cast<int32_t>(topic_usage_vals.size()) == n_topics_) {
        bool valid_usage = true;
        double usage_sum = 0.0;
        for (double x : topic_usage_vals) {
            valid_usage = valid_usage && std::isfinite(x) && x >= 0.0;
            usage_sum += x;
        }
        if (valid_usage && usage_sum > 0.0) {
            for (int32_t k = 0; k < n_topics_; ++k) topic_usage_(k) = topic_usage_vals[k];
        }
    }
    has_dispersion_ = false;
    tau_.resize(0);
    if (static_cast<int32_t>(tau_vals.size()) == n_features_) {
        set_feature_dispersion(tau_vals);
    } else if (!tau_vals.empty()) {
        error("%s: State file has %zu dispersion values but expected %d",
            __func__, tau_vals.size(), n_features_);
    }
}


void GammaPoisson4Hex::initialize(int32_t nTopics, int32_t seed, int32_t nThreads,
    int32_t verbose, double beta_shape, double xi_shape, double xi_mean,
    double theta_concentration, double nu_shape, double nu_rate,
    double kappa, double tau0,
    int32_t totalDocCount, double sizeFactor, bool symmetricNu, double nuMax,
    int32_t maxIter, double mDelta, double randomInitShape) {
    if (reader.features.size() != static_cast<size_t>(M_)) {
        featureNames.resize(M_);
        for (int32_t i = 0; i < M_; ++i) featureNames[i] = std::to_string(i);
    } else {
        featureNames = reader.features;
    }
    const std::vector<double>& sums = reader.getFeatureSums();
    model_ = std::make_unique<GammaPoissonTopicModel>(
        nTopics, M_, seed, nThreads, verbose, beta_shape, xi_shape, xi_mean,
        theta_concentration, nu_shape, nu_rate, kappa, tau0,
        totalDocCount, sizeFactor,
        symmetricNu, nuMax, reader.readFullSums ? &sums : nullptr,
        randomInitShape);
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
}

void GammaPoisson4Hex::clearFeatureDispersion() {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->clear_feature_dispersion();
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
    return result;
}

GammaPoissonDispersionResult GammaPoisson4Hex::estimateFeatureDispersion(
    const GammaPoissonDispersionOptions& options,
    uac::DocumentBlockSource& source,
    int32_t batchSize_, int32_t maxUnits) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    GammaPoissonDispersionEstimator estimator(*model_, options);
    source.reset();
    uac::DocumentBlock block;
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

double GammaPoisson4Hex::resolveSizeFactor(double requested) const {
    if (requested > 0.0 && std::isfinite(requested)) {
        return requested;
    }
    if (!reader.readFullSums) {
        error("--size-factor is required because full feature counts are unavailable");
    }
    if (reader.nUnits <= 0) {
        error("--size-factor is required because total document count is unavailable");
    }
    const std::vector<double>& sums = reader.getFeatureSums();
    const double total = std::accumulate(sums.begin(), sums.end(), 0.0);
    if (total <= 0.0 || !std::isfinite(total)) {
        error("--size-factor is required because total feature count is unavailable or zero");
    }
    return total / static_cast<double>(reader.nUnits);
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
    model_->partial_fit(batch);
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

const VectorXd& GammaPoisson4Hex::getFeatureDispersion() const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    return model_->get_feature_dispersion();
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
    model_->get_topic_abundance(topic_weights);
}
