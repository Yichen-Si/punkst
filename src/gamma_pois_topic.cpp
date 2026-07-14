#include "gamma_pois_topic.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/QR>

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

} // namespace

int GammaPoissonTopicBase::normalize_seed(int seed) {
    return seed > 0 ? seed : static_cast<int>(std::random_device{}());
}

GammaPoissonTopicBase::GammaPoissonTopicBase(int32_t n_topics, int32_t n_features,
    int seed, int32_t nThreads, int32_t verbose, double beta_shape, double xi_shape,
    double xi_mean, double theta_shape, double nu_shape, double nu_rate,
    double learning_decay, double learning_offset, int32_t total_doc_count,
    double size_factor, const std::vector<double>* feature_sums)
    : n_topics_(n_topics), n_features_(n_features), seed_(normalize_seed(seed)),
      nThreads_(nThreads), verbose_(verbose),
      total_doc_count_(total_doc_count > 0 ? total_doc_count : 1000000),
      a_(positive_or(beta_shape, 0.3)),
      a0_(positive_or(xi_shape, 0.3)),
      b0_(positive_or(xi_mean, 1.0)),
      s0_(positive_or(theta_shape, 1.0)),
      e0_(positive_or(nu_shape, 1.0)),
      f0_(positive_or(nu_rate, 1.0)),
      learning_decay_(positive_or(learning_decay, 0.7)),
      learning_offset_(learning_offset >= 0.0 ? learning_offset : 10.0),
      size_factor_(positive_or(size_factor, 1.0)) {
    random_engine_.seed(seed_);
    if (nu_rate <= 0.0) {
        f0_ = e0_ * s0_ * static_cast<double>(n_topics_) / size_factor_;
        f0_ = positive_or(f0_, 1.0);
    }
    set_nthreads(nThreads_);
    init_from_feature_sums(feature_sums);
}

GammaPoissonTopicModel::GammaPoissonTopicModel(int32_t n_topics, int32_t n_features,
    int seed, int32_t nThreads, int32_t verbose, double beta_shape, double xi_shape,
    double xi_mean, double theta_shape, double theta_concentration,
    double nu_shape, double nu_rate,
    double learning_decay, double learning_offset, int32_t total_doc_count,
    double size_factor, bool symmetric_nu, double nu_max,
    const std::vector<double>* feature_sums)
    : GammaPoissonTopicBase(n_topics, n_features, seed, nThreads, verbose,
          beta_shape, xi_shape, xi_mean, theta_shape, nu_shape, nu_rate,
          learning_decay, learning_offset, total_doc_count, size_factor, feature_sums),
      symmetric_nu_(symmetric_nu), theta_concentration_(theta_concentration),
      nu_max_(nu_max) {
    const double active_theta_parameter = symmetric_nu_
        ? theta_concentration_ : theta_shape;
    if (!std::isfinite(active_theta_parameter) || active_theta_parameter <= 0.0) {
        throw std::invalid_argument(symmetric_nu_
            ? "Symmetric Gamma-Poisson theta concentration must be positive and finite"
            : "Empirical-Bayes Gamma-Poisson theta shape must be positive and finite");
    }
    if (!symmetric_nu_ && (!std::isfinite(nu_max_) || nu_max_ <= 0.0)) {
        throw std::invalid_argument(
            "Asymmetric Gamma-Poisson shrinkage requires a positive finite nu_max");
    }
    nu_shape_ = VectorXd::Constant(n_topics_, e0_ + s0_ * total_doc_count_);
    nu_rate_ = VectorXd::Constant(n_topics_, f0_ + total_doc_count_ * s0_);
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

GammaPoissonTopicModel::GammaPoissonTopicModel(const std::string& stateFile,
    int seed, int32_t nThreads, int32_t verbose)
    : GammaPoissonTopicBase() {
    seed_ = normalize_seed(seed);
    nThreads_ = nThreads;
    verbose_ = verbose;
    random_engine_.seed(seed_);
    read_state(stateFile);
    set_nthreads(nThreads_);
    refresh_cache();
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
    beta_shape_.resize(n_topics_, n_features_);
    beta_rate_.resize(n_topics_, n_features_);
    xi_shape_ = VectorXd::Constant(n_features_, a0_ + a_ * n_topics_);
    xi_rate_ = VectorXd::Constant(n_features_, a0_ / b0_ + n_topics_ * a_ / b0_);
    topic_usage_ = VectorXd::Constant(n_topics_,
        size_factor_ * static_cast<double>(total_doc_count_) / static_cast<double>(std::max(1, n_topics_)));

    std::gamma_distribution<double> noise(2.0, 0.5);
    const double total = feature_sums
        ? std::accumulate(feature_sums->begin(), feature_sums->end(), 0.0)
        : 0.0;
    for (int32_t w = 0; w < n_features_; ++w) {
        const double freq = (feature_sums && total > 0.0)
            ? std::max((*feature_sums)[w] / total * size_factor_, 1e-8)
            : std::max(size_factor_ / static_cast<double>(std::max(1, n_features_)), 1e-8);
        const double xi_mean = std::max(a_ / freq, 1e-8);
        for (int32_t k = 0; k < n_topics_; ++k) {
            const double jitter = 0.5 + noise(random_engine_);
            beta_shape_(k, w) = a_ + freq * jitter / static_cast<double>(n_topics_);
            beta_rate_(k, w) = xi_mean;
        }
    }
    refresh_cache();
}

void GammaPoissonTopicBase::refresh_cache() {
    e_beta_.resize(n_topics_, n_features_); // E[\beta_{kw}]
    elog_beta_.resize(n_topics_, n_features_);
    for (int32_t k = 0; k < n_topics_; ++k) {
        for (int32_t w = 0; w < n_features_; ++w) {
            beta_shape_(k, w) = std::max(beta_shape_(k, w), 1e-12);
            beta_rate_(k, w) = std::max(beta_rate_(k, w), 1e-12);
            e_beta_(k, w) = beta_shape_(k, w) / beta_rate_(k, w);
            elog_beta_(k, w) = psi(beta_shape_(k, w)) - std::log(beta_rate_(k, w));
        }
    }
    topic_capacity_ = e_beta_.rowwise().sum();
    model_phi_ = e_beta_;
    for (int32_t k = 0; k < n_topics_; ++k) {
        const double denom = std::max(topic_capacity_(k), 1e-300);
        model_phi_.row(k) /= denom;
    }
}

double GammaPoissonTopicBase::doc_exposure(const Document& doc) const {
    const double len = doc_sum_const(doc);
    if (len <= 0.0) {
        return 0.0;
    }
    return len / size_factor_;
}

double GammaPoissonTopicModel::expected_epsilon(int32_t w, double y, double c,
    const VectorXd& e_theta) const {
    double lambda = 0.0;
    for (int32_t k = 0; k < n_topics_; ++k) {
        lambda += e_theta(k) * e_beta_(k, w);
    }
    const double tau = tau_(w);
    return (tau + y) / std::max(tau + c * lambda, 1e-12);
}

void GammaPoissonTopicModel::expected_observed_counts(const Document& doc,
    std::vector<double>& means) const {
    if (has_dispersion_) {
        error("%s: dispersion means must be computed from a Poisson warmup model", __func__);
    }
    VectorXd theta_shape, theta_rate, elog_theta;
    fit_one_document(theta_shape, theta_rate, elog_theta, doc);
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

int32_t GammaPoissonTopicModel::fit_one_document(VectorXd& theta_shape,
    VectorXd& theta_rate, VectorXd& elog_theta, const Document& doc) const {
    const int32_t n_ids = static_cast<int32_t>(doc.ids.size());
    const double prior_shape = theta_prior_shape();
    theta_shape = VectorXd::Constant(n_topics_, prior_shape);
    theta_rate.resize(n_topics_);
    const double c = doc_exposure(doc);
    for (int32_t k = 0; k < n_topics_; ++k) {
        theta_rate(k) = theta_prior_rate(k) + c * topic_capacity_(k);
    }
    if (n_ids == 0) {
        elog_theta.resize(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            elog_theta(k) = psi(theta_shape(k)) - std::log(theta_rate(k));
        }
        return 0;
    }

    elog_theta.resize(n_topics_);
    const double doc_total = doc_sum_const(doc);
    for (int32_t k = 0; k < n_topics_; ++k) {
        theta_shape(k) += doc_total / static_cast<double>(n_topics_);
    }

    VectorXd assigned(n_topics_);
    std::vector<double> log_phi(n_topics_);
    double diff = 1.0;
    int32_t iter = 0;
    while (iter < max_doc_update_iter_) {
        VectorXd last_shape = theta_shape;
        if (has_dispersion_) {
            VectorXd e_theta = theta_shape.array() / theta_rate.array().max(1e-12);
            for (int32_t k = 0; k < n_topics_; ++k) {
                theta_rate(k) = theta_prior_rate(k) + c * topic_capacity_(k);
            }
            for (int32_t j = 0; j < n_ids; ++j) {
                const uint32_t w = doc.ids[j];
                const double eps = expected_epsilon(w, doc.cnts[j], c, e_theta);
                const double delta = c * (eps - 1.0);
                for (int32_t k = 0; k < n_topics_; ++k) {
                    theta_rate(k) += delta * e_beta_(k, w);
                }
            }
        }
        for (int32_t k = 0; k < n_topics_; ++k) {
            theta_rate(k) = std::max(theta_rate(k), 1e-12);
            elog_theta(k) = psi(theta_shape(k)) - std::log(theta_rate(k));
        }
        assigned.setZero();
        for (int32_t j = 0; j < n_ids; ++j) {
            const uint32_t w = doc.ids[j];
            double max_log = -std::numeric_limits<double>::infinity();
            for (int32_t k = 0; k < n_topics_; ++k) {
                log_phi[k] = elog_theta(k) + elog_beta_(k, w);
                max_log = std::max(max_log, log_phi[k]);
            }
            double norm = 0.0;
            for (int32_t k = 0; k < n_topics_; ++k) {
                log_phi[k] = std::exp(log_phi[k] - max_log);
                norm += log_phi[k];
            }
            norm = std::max(norm, eps_);
            for (int32_t k = 0; k < n_topics_; ++k) {
                assigned(k) += doc.cnts[j] * log_phi[k] / norm;
            }
        }
        theta_shape = assigned.array() + prior_shape;
        diff = (theta_shape - last_shape).cwiseAbs().sum() / static_cast<double>(n_topics_);
        ++iter;
        if (diff < mean_change_tol_) {
            break;
        }
    }
    if (has_dispersion_) {
        VectorXd e_theta = theta_shape.array() / theta_rate.array().max(1e-12);
        for (int32_t k = 0; k < n_topics_; ++k) {
            theta_rate(k) = theta_prior_rate(k) + c * topic_capacity_(k);
        }
        for (int32_t j = 0; j < n_ids; ++j) {
            const uint32_t w = doc.ids[j];
            const double eps = expected_epsilon(w, doc.cnts[j], c, e_theta);
            const double delta = c * (eps - 1.0);
            for (int32_t k = 0; k < n_topics_; ++k) {
                theta_rate(k) += delta * e_beta_(k, w);
            }
        }
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        theta_rate(k) = std::max(theta_rate(k), 1e-12);
        elog_theta(k) = psi(theta_shape(k)) - std::log(theta_rate(k));
    }
    return iter;
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

void GammaPoissonTopicModel::partial_fit(const std::vector<Document>& docs) {
    const int32_t minibatch_size = static_cast<int32_t>(docs.size());
    if (minibatch_size == 0) {
        return;
    }

    tbb::combinable<RowMajorMatrixXd> ss_acc{
        [&] { return RowMajorMatrixXd::Zero(n_topics_, n_features_); }
    };
    tbb::combinable<VectorXd> ctheta_acc{
        [&] { return VectorXd::Zero(n_topics_); }
    };
    tbb::combinable<VectorXd> theta_acc{
        [&] { return VectorXd::Zero(n_topics_); }
    };
    tbb::combinable<RowMajorMatrixXd> beta_rate_corr_acc{
        [&] { return RowMajorMatrixXd::Zero(n_topics_, n_features_); }
    };
    tbb::combinable<std::vector<int32_t>> niters_acc{
        [] { return std::vector<int32_t>(); }
    };

    tbb::parallel_for(tbb::blocked_range<int32_t>(0, minibatch_size),
        [&](const tbb::blocked_range<int32_t>& range) {
            auto& local_ss = ss_acc.local();
            auto& local_ctheta = ctheta_acc.local();
            auto& local_theta = theta_acc.local();
            auto& local_niters = niters_acc.local();
            VectorXd theta_shape, theta_rate, elog_theta;
            std::vector<double> log_phi(n_topics_);
            for (int32_t d = range.begin(); d < range.end(); ++d) {
                const Document& doc = docs[d];
                const int32_t iter = fit_one_document(theta_shape, theta_rate, elog_theta, doc);
                local_niters.push_back(iter);
                VectorXd e_theta = theta_shape.array() / theta_rate.array();
                local_theta += e_theta;
                const double cexp = doc_exposure(doc);
                local_ctheta += cexp * e_theta;
                for (size_t j = 0; j < doc.ids.size(); ++j) {
                    const uint32_t w = doc.ids[j];
                    if (has_dispersion_) {
                        auto& local_beta_rate_corr = beta_rate_corr_acc.local();
                        const double eps = expected_epsilon(w, doc.cnts[j], cexp, e_theta);
                        const double corr = (eps - 1.0) * cexp;
                        for (int32_t k = 0; k < n_topics_; ++k) {
                            local_beta_rate_corr(k, w) += corr * e_theta(k);
                        }
                    }
                    double max_log = -std::numeric_limits<double>::infinity();
                    for (int32_t k = 0; k < n_topics_; ++k) {
                        log_phi[k] = elog_theta(k) + elog_beta_(k, w);
                        max_log = std::max(max_log, log_phi[k]);
                    }
                    double norm = 0.0;
                    for (int32_t k = 0; k < n_topics_; ++k) {
                        log_phi[k] = std::exp(log_phi[k] - max_log);
                        norm += log_phi[k];
                    }
                    norm = std::max(norm, eps_);
                    for (int32_t k = 0; k < n_topics_; ++k) {
                        local_ss(k, w) += doc.cnts[j] * log_phi[k] / norm;
                    }
                }
            }
        });

    RowMajorMatrixXd ss = ss_acc.combine(
        [](const RowMajorMatrixXd& a, const RowMajorMatrixXd& b) { return a + b; });
    VectorXd ctheta = ctheta_acc.combine(
        [](const VectorXd& a, const VectorXd& b) { return a + b; });
    VectorXd theta = theta_acc.combine(
        [](const VectorXd& a, const VectorXd& b) { return a + b; });
    RowMajorMatrixXd beta_rate_corr;
    if (has_dispersion_) {
        beta_rate_corr = beta_rate_corr_acc.combine(
            [](const RowMajorMatrixXd& a, const RowMajorMatrixXd& b) { return a + b; });
    }
    std::vector<int32_t> niters = niters_acc.combine(
        [](const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
            std::vector<int32_t> out = a;
            out.insert(out.end(), b.begin(), b.end());
            return out;
        });

    ++update_count_;
    const double rho = std::pow(learning_offset_ + update_count_, -learning_decay_);
    const double scale = static_cast<double>(total_doc_count_) / static_cast<double>(minibatch_size);
    VectorXd xi_mean(n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        xi_mean(w) = expected_xi(w);
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        const double scaled_ctheta = scale * ctheta(k);
        for (int32_t w = 0; w < n_features_; ++w) {
            const double target_shape = a_ + scale * ss(k, w);
            double target_rate = xi_mean(w) + scaled_ctheta;
            if (has_dispersion_) {
                target_rate += scale * beta_rate_corr(k, w);
            }
            beta_shape_(k, w) = (1.0 - rho) * beta_shape_(k, w) + rho * target_shape;
            beta_rate_(k, w) = (1.0 - rho) * beta_rate_(k, w) + rho * std::max(target_rate, 1e-12);
        }
    }
    refresh_cache();

    for (int32_t w = 0; w < n_features_; ++w) {
        xi_shape_(w) = a0_ + a_ * n_topics_;
        xi_rate_(w) = a0_ / b0_ + e_beta_.col(w).sum();
    }
    if (!symmetric_nu_) {
        for (int32_t k = 0; k < n_topics_; ++k) {
            nu_shape_(k) = e0_ + s0_ * total_doc_count_;
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

    if (verbose_ > 0 && !niters.empty()) {
        int32_t fail = 0;
        for (int32_t n : niters) {
            if (n >= max_doc_update_iter_) {
                ++fail;
            }
        }
        const double avg = std::accumulate(niters.begin(), niters.end(), 0.0)
            / static_cast<double>(niters.size());
        notice("Gamma-Poisson partial fit: %d documents. Average iterations per doc: %.2f, %d documents did not reach mean change %.1e in %d iterations.",
            minibatch_size, avg, fail, mean_change_tol_, max_doc_update_iter_);
    }
}

RowMajorMatrixXd GammaPoissonTopicModel::transform(DocumentView docs) {
    RowMajorMatrixXd out;
    std::vector<GammaPoissonDocumentPosterior> posteriors;
    transform_with_posteriors(docs, out, posteriors);
    return out;
}

void GammaPoissonTopicModel::infer_document_posterior(const Document& doc,
    GammaPoissonDocumentPosterior& posterior) const {
    VectorXd elog_theta;
    fit_one_document(posterior.shape, posterior.rate, elog_theta, doc);
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

void GammaPoissonTopicModel::dispersion_covariance_approximation(
    const Document& doc, const GammaPoissonDocumentPosterior& posterior,
    int32_t rank, uint64_t seed, GammaPoissonDispersionApproximation& out) const {
    if (!has_dispersion_) {
        error("%s: model does not have feature dispersion", __func__);
    }
    if (posterior.shape.size() != n_topics_ || posterior.rate.size() != n_topics_) {
        error("%s: posterior has wrong topic dimension", __func__);
    }
    const int32_t target_rank = std::min(std::max(rank, 0), n_topics_);
    const int32_t n_cells = static_cast<int32_t>(doc.ids.size());
    Eigen::MatrixXd loading = Eigen::MatrixXd::Zero(n_topics_, n_cells);
    const VectorXd e_theta = posterior.shape.array()
        / posterior.rate.array().max(1e-12);
    for (int32_t j = 0; j < n_cells; ++j) {
        const uint32_t w = doc.ids[j];
        if (w >= static_cast<uint32_t>(n_features_)) {
            error("%s: feature index %u is out of range", __func__, w);
        }
        double lambda = 0.0;
        for (int32_t k = 0; k < n_topics_; ++k) {
            lambda += e_theta(k) * e_beta_(k, w);
        }
        const double tau = tau_(w);
        const double eps_rate = std::max(tau + posterior.exposure * lambda, 1e-12);
        const double eps_var = (tau + doc.cnts[j]) / (eps_rate * eps_rate);
        const double scale = posterior.exposure * std::sqrt(std::max(eps_var, 0.0));
        for (int32_t k = 0; k < n_topics_; ++k) {
            loading(k, j) = scale * e_beta_(k, w)
                / std::max(posterior.rate(k), 1e-12);
        }
    }

    const VectorXd exact_diagonal = loading.array().square().rowwise().sum();
    out.factor = RowMajorMatrixXd::Zero(n_topics_, target_rank);
    if (target_rank > 0 && n_cells > 0 && exact_diagonal.maxCoeff() > 0.0) {
        if (target_rank == n_topics_) {
            Eigen::MatrixXd gram = loading * loading.transpose();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(gram);
            if (solver.info() != Eigen::Success) {
                error("%s: dispersion covariance eigendecomposition failed", __func__);
            }
            for (int32_t a = 0; a < target_rank; ++a) {
                const int32_t idx = n_topics_ - 1 - a;
                const double value = std::max(0.0, solver.eigenvalues()(idx));
                out.factor.col(a) = solver.eigenvectors().col(idx) * std::sqrt(value);
            }
        } else {
            const int32_t sketch_rank = std::min(n_topics_, target_rank + 4);
            std::mt19937_64 rng(seed);
            std::normal_distribution<double> normal(0.0, 1.0);
            Eigen::MatrixXd omega(n_cells, sketch_rank);
            for (Eigen::Index i = 0; i < omega.size(); ++i) {
                omega.data()[i] = normal(rng);
            }
            Eigen::MatrixXd range = loading * omega;
            range = loading * (loading.transpose() * range);
            Eigen::HouseholderQR<Eigen::MatrixXd> qr(range);
            Eigen::MatrixXd q = qr.householderQ()
                * Eigen::MatrixXd::Identity(n_topics_, sketch_rank);
            Eigen::MatrixXd projected = q.transpose() * loading;
            Eigen::MatrixXd core = projected * projected.transpose();
            Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(core);
            if (solver.info() != Eigen::Success) {
                error("%s: compressed dispersion covariance eigendecomposition failed", __func__);
            }
            for (int32_t a = 0; a < target_rank; ++a) {
                const int32_t idx = sketch_rank - 1 - a;
                const double value = std::max(0.0, solver.eigenvalues()(idx));
                out.factor.col(a) = q * solver.eigenvectors().col(idx) * std::sqrt(value);
            }
        }
    }
    out.residual_diagonal = exact_diagonal
        - out.factor.array().square().rowwise().sum().matrix();
    out.residual_diagonal = out.residual_diagonal.cwiseMax(0.0);
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
    return model_phi_;
}

RowMajorMatrixXd GammaPoissonTopicBase::copy_model() {
    return model_phi_;
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
    auto sort_rows = [&](RowMajorMatrixXd& m) {
        RowMajorMatrixXd sorted(m.rows(), m.cols());
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
    auto sort_rows = [&](RowMajorMatrixXd& m) {
        RowMajorMatrixXd sorted(m.rows(), m.cols());
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
    out << std::scientific << std::setprecision(6);
    for (int32_t w = 0; w < n_features_; ++w) {
        const std::string feature = w < static_cast<int32_t>(featureNames.size())
            ? featureNames[w] : std::to_string(w);
        out << feature;
        for (int32_t k = 0; k < n_topics_; ++k) {
            out << "\t" << model_phi_(k, w);
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
    out << std::setprecision(17);
    out << "#punkst_gamma_pois_state_v2\n";
    out << "#n_topics\t" << n_topics_ << "\n";
    out << "#n_features\t" << n_features_ << "\n";
    out << "#total_doc_count\t" << total_doc_count_ << "\n";
    out << "#size_factor\t" << size_factor_ << "\n";
    out << "#beta_shape_prior\t" << a_ << "\n";
    out << "#xi_shape_prior\t" << a0_ << "\n";
    out << "#xi_mean_prior\t" << b0_ << "\n";
    out << "#theta_prior_mode\t"
        << (symmetric_nu_ ? "symmetric_concentration" : "eb_rate") << "\n";
    if (symmetric_nu_) {
        out << "#theta_concentration_prior\t" << theta_concentration_ << "\n";
    } else {
        out << "#theta_shape_prior\t" << s0_ << "\n";
        out << "#nu_shape_prior\t" << e0_ << "\n";
        out << "#nu_rate_prior\t" << f0_ << "\n";
    }
    out << "#learning_decay\t" << learning_decay_ << "\n";
    out << "#learning_offset\t" << learning_offset_ << "\n";
    out << "#update_count\t" << update_count_ << "\n";
    out << "#symmetric_nu\t" << (symmetric_nu_ ? 1 : 0) << "\n";
    if (!symmetric_nu_) {
        out << "#nu_max\t" << nu_max_ << "\n";
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
    out << "\txi_shape\txi_rate\n";
    for (int32_t w = 0; w < n_features_; ++w) {
        const std::string feature = w < static_cast<int32_t>(featureNames.size())
            ? featureNames[w] : std::to_string(w);
        out << feature;
        for (int32_t k = 0; k < n_topics_; ++k) {
            out << "\t" << beta_shape_(k, w) << "\t" << beta_rate_(k, w);
        }
        out << "\t" << xi_shape_(w) << "\t" << xi_rate_(w) << "\n";
    }
}

std::vector<std::string> GammaPoissonTopicModel::read_state_feature_names(
    const std::string& stateFile) {
    std::ifstream in(stateFile);
    if (!in) {
        error("%s: Error opening state file: %s", __func__, stateFile.c_str());
    }
    std::vector<std::string> features;
    std::string line;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') {
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
        if (saw_header) {
            features.push_back(tok[0]);
        }
    }
    if (features.empty()) {
        error("%s: No feature rows found in Gamma-Poisson state file: %s",
            __func__, stateFile.c_str());
    }
    return features;
}

void GammaPoissonTopicModel::read_state(const std::string& stateFile) {
    std::ifstream in(stateFile);
    if (!in) {
        error("%s: Error opening state file: %s", __func__, stateFile.c_str());
    }
    std::string line;
    std::vector<std::vector<double>> bshape_rows;
    std::vector<std::vector<double>> brate_rows;
    std::vector<double> xi_shape_vals, xi_rate_vals;
    std::vector<double> nu_shape_vals, nu_rate_vals;
    std::vector<double> topic_usage_vals;
    std::vector<double> tau_vals;
    bool saw_state_version = false;
    std::string theta_prior_mode;
    bool saw_theta_shape = false;
    bool saw_theta_concentration = false;
    bool saw_symmetric_nu = false;
    bool saw_nu_shape_prior = false;
    bool saw_nu_rate_prior = false;
    bool saw_nu_max = false;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        if (line[0] == '#') {
            std::string payload = line.substr(1);
            auto tok = split_ws(payload);
            if (tok.empty()) continue;
            if (tok[0] == "punkst_gamma_pois_state_v2") saw_state_version = true;
            else if (tok[0] == "n_topics" && tok.size() > 1) n_topics_ = std::stoi(tok[1]);
            else if (tok[0] == "n_features" && tok.size() > 1) n_features_ = std::stoi(tok[1]);
            else if (tok[0] == "total_doc_count" && tok.size() > 1) total_doc_count_ = std::stoi(tok[1]);
            else if (tok[0] == "size_factor" && tok.size() > 1) size_factor_ = std::stod(tok[1]);
            else if (tok[0] == "beta_shape_prior" && tok.size() > 1) a_ = std::stod(tok[1]);
            else if (tok[0] == "xi_shape_prior" && tok.size() > 1) a0_ = std::stod(tok[1]);
            else if (tok[0] == "xi_mean_prior" && tok.size() > 1) b0_ = std::stod(tok[1]);
            else if (tok[0] == "theta_prior_mode" && tok.size() > 1) theta_prior_mode = tok[1];
            else if (tok[0] == "theta_shape_prior" && tok.size() > 1) {
                s0_ = std::stod(tok[1]);
                saw_theta_shape = true;
            } else if (tok[0] == "theta_concentration_prior" && tok.size() > 1) {
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
            else if (tok[0] == "nu_max" && tok.size() > 1) {
                nu_max_ = std::stod(tok[1]);
                saw_nu_max = true;
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
        const size_t expected = 1 + static_cast<size_t>(2 * n_topics_) + 2;
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
    }
    if (!saw_state_version) {
        error("%s: Missing or unsupported Gamma-Poisson state version in %s",
            __func__, stateFile.c_str());
    }
    if (!saw_symmetric_nu) {
        error("%s: Gamma-Poisson state is missing symmetric_nu in %s",
            __func__, stateFile.c_str());
    }
    const bool symmetric_mode = theta_prior_mode == "symmetric_concentration";
    const bool eb_mode = theta_prior_mode == "eb_rate";
    if ((!symmetric_mode && !eb_mode) || symmetric_mode != symmetric_nu_) {
        error("%s: Invalid or inconsistent theta prior mode in Gamma-Poisson state: %s",
            __func__, stateFile.c_str());
    }
    if (symmetric_mode) {
        if (!saw_theta_concentration || saw_theta_shape
            || !std::isfinite(theta_concentration_)
            || theta_concentration_ <= 0.0) {
            error("%s: Symmetric Gamma-Poisson state requires one positive finite theta concentration: %s",
                __func__, stateFile.c_str());
        }
    } else if (!saw_theta_shape || saw_theta_concentration
        || !std::isfinite(s0_) || s0_ <= 0.0
        || !saw_nu_shape_prior || !std::isfinite(e0_) || e0_ <= 0.0
        || !saw_nu_rate_prior || !std::isfinite(f0_) || f0_ <= 0.0
        || !saw_nu_max || !std::isfinite(nu_max_) || nu_max_ <= 0.0
        || static_cast<int32_t>(nu_shape_vals.size()) != n_topics_
        || static_cast<int32_t>(nu_rate_vals.size()) != n_topics_) {
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
    for (int32_t w = 0; w < n_features_; ++w) {
        xi_shape_(w) = xi_shape_vals[w];
        xi_rate_(w) = xi_rate_vals[w];
    }
    nu_shape_ = VectorXd::Constant(n_topics_, e0_ + s0_ * total_doc_count_);
    nu_rate_ = VectorXd::Constant(n_topics_, f0_ + total_doc_count_ * s0_);
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

GammaPoissonTopicJointC::GammaPoissonTopicJointC(int32_t n_topics, int32_t n_features,
    int32_t n_clusters, int seed, int32_t nThreads, int32_t verbose,
    double beta_shape, double xi_shape, double xi_mean, double theta_shape,
    double nu_shape, double nu_rate, double cluster_prior, double learning_decay,
    double learning_offset, int32_t total_doc_count, double size_factor,
    const std::vector<double>* feature_sums)
    : GammaPoissonTopicBase(n_topics, n_features, seed, nThreads, verbose,
          beta_shape, xi_shape, xi_mean, theta_shape, nu_shape, nu_rate,
          learning_decay, learning_offset, total_doc_count, size_factor, feature_sums),
      n_clusters_(n_clusters > 0 ? n_clusters : 1),
      gamma_(positive_or(cluster_prior, 1.0)),
      effective_gamma_(gamma_) {
    init_cluster_state();
}

GammaPoissonTopicJointC::GammaPoissonTopicJointC(const std::string& stateFile,
    int seed, int32_t nThreads, int32_t verbose)
    : GammaPoissonTopicBase() {
    seed_ = normalize_seed(seed);
    nThreads_ = nThreads;
    verbose_ = verbose;
    random_engine_.seed(seed_);
    read_state(stateFile);
    set_nthreads(nThreads_);
    refresh_cache();
}

void GammaPoissonTopicJointC::init_cluster_state() {
    pi_shape_ = VectorXd::Constant(n_clusters_, gamma_ / static_cast<double>(n_clusters_)
        + static_cast<double>(total_doc_count_) / static_cast<double>(n_clusters_));
    const double init_shape = e0_ + s0_ * static_cast<double>(total_doc_count_)
        / static_cast<double>(n_clusters_);
    const double init_rate = f0_ + s0_ * static_cast<double>(total_doc_count_)
        / static_cast<double>(n_clusters_);
    nu_shape_.resize(n_clusters_, n_topics_);
    nu_rate_.resize(n_clusters_, n_topics_);
    std::gamma_distribution<double> noise(2.0, 0.5);
    for (int32_t c = 0; c < n_clusters_; ++c) {
        for (int32_t k = 0; k < n_topics_; ++k) {
            const double jitter = 0.75 + 0.5 * noise(random_engine_);
            nu_shape_(c, k) = init_shape;
            nu_rate_(c, k) = std::max(init_rate * jitter, 1e-12);
        }
    }
    cluster_usage_ = VectorXd::Constant(n_clusters_,
        static_cast<double>(total_doc_count_) / static_cast<double>(n_clusters_));
}

void GammaPoissonTopicJointC::set_cluster_warmup(bool enabled) {
    force_uniform_chi_ = enabled;
    update_cluster_globals_ = !enabled;
}

void GammaPoissonTopicJointC::set_cluster_temperature(double temperature) {
    chi_temperature_ = positive_or(temperature, 1.0);
    if (chi_temperature_ < 1.0) chi_temperature_ = 1.0;
}

void GammaPoissonTopicJointC::set_effective_cluster_prior(double gamma) {
    effective_gamma_ = positive_or(gamma, gamma_);
}

const std::vector<std::string>& GammaPoissonTopicJointC::get_cluster_names() {
    if (cluster_names_.empty()) {
        cluster_names_.resize(n_clusters_);
        for (int32_t c = 0; c < n_clusters_; ++c) {
            cluster_names_[c] = "C" + std::to_string(c);
        }
    }
    return cluster_names_;
}

void GammaPoissonTopicJointC::get_cluster_abundance(std::vector<double>& weights) const {
    weights.resize(n_clusters_);
    const double total = cluster_usage_.sum();
    if (total <= 0.0 || !std::isfinite(total)) {
        std::fill(weights.begin(), weights.end(), 1.0 / static_cast<double>(n_clusters_));
        return;
    }
    for (int32_t c = 0; c < n_clusters_; ++c) {
        weights[c] = cluster_usage_(c) / total;
    }
}

void GammaPoissonTopicJointC::sort_topics() {
    std::vector<int32_t> order(n_topics_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return topic_usage_(a) > topic_usage_(b);
    });
    auto sort_rows = [&](RowMajorMatrixXd& m) {
        RowMajorMatrixXd sorted(m.rows(), m.cols());
        for (int32_t k = 0; k < n_topics_; ++k) {
            sorted.row(k) = m.row(order[k]);
        }
        m = std::move(sorted);
    };
    auto sort_cols = [&](RowMajorMatrixXd& m) {
        RowMajorMatrixXd sorted(m.rows(), m.cols());
        for (int32_t k = 0; k < n_topics_; ++k) {
            sorted.col(k) = m.col(order[k]);
        }
        m = std::move(sorted);
    };
    sort_rows(beta_shape_);
    sort_rows(beta_rate_);
    sort_cols(nu_shape_);
    sort_cols(nu_rate_);
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

void GammaPoissonTopicJointC::sort_clusters() {
    std::vector<int32_t> order(n_clusters_);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int32_t a, int32_t b) {
        return cluster_usage_(a) > cluster_usage_(b);
    });
    RowMajorMatrixXd ns(n_clusters_, n_topics_), nr(n_clusters_, n_topics_);
    VectorXd ps(n_clusters_), usage(n_clusters_);
    for (int32_t c = 0; c < n_clusters_; ++c) {
        ns.row(c) = nu_shape_.row(order[c]);
        nr.row(c) = nu_rate_.row(order[c]);
        ps(c) = pi_shape_(order[c]);
        usage(c) = cluster_usage_(order[c]);
    }
    nu_shape_ = std::move(ns);
    nu_rate_ = std::move(nr);
    pi_shape_ = std::move(ps);
    cluster_usage_ = std::move(usage);
    if (!cluster_names_.empty()) {
        std::vector<std::string> sorted(n_clusters_);
        for (int32_t c = 0; c < n_clusters_; ++c) {
            sorted[c] = cluster_names_[order[c]];
        }
        cluster_names_ = std::move(sorted);
    }
}

int32_t GammaPoissonTopicJointC::fit_one_document(VectorXd& theta_shape,
    VectorXd& theta_rate, VectorXd& elog_theta, VectorXd& chi,
    const Document& doc, bool force_uniform_chi) const {
    const int32_t n_ids = static_cast<int32_t>(doc.ids.size());
    theta_shape = VectorXd::Constant(n_topics_, s0_);
    theta_rate.resize(n_topics_);
    elog_theta.resize(n_topics_);
    chi = VectorXd::Constant(n_clusters_, 1.0 / static_cast<double>(n_clusters_));

    const double cexp = doc_exposure(doc);
    const double doc_total = doc_sum_const(doc);
    for (int32_t k = 0; k < n_topics_; ++k) {
        theta_shape(k) += doc_total / static_cast<double>(n_topics_);
    }

    std::vector<double> log_phi(n_topics_);
    VectorXd assigned(n_topics_);
    VectorXd log_chi(n_clusters_);
    double diff = 1.0;
    int32_t iter = 0;
    while (iter < max_doc_update_iter_) {
        VectorXd last_shape = theta_shape;
        VectorXd last_chi = chi;
        for (int32_t k = 0; k < n_topics_; ++k) {
            double prior_rate = 0.0;
            for (int32_t cc = 0; cc < n_clusters_; ++cc) {
                prior_rate += chi(cc) * expected_nu(cc, k);
            }
            theta_rate(k) = prior_rate + cexp * topic_capacity_(k);
            elog_theta(k) = psi(theta_shape(k)) - std::log(std::max(theta_rate(k), 1e-12));
        }

        assigned.setZero();
        for (int32_t j = 0; j < n_ids; ++j) {
            const uint32_t w = doc.ids[j];
            double max_log = -std::numeric_limits<double>::infinity();
            for (int32_t k = 0; k < n_topics_; ++k) {
                log_phi[k] = elog_theta(k) + elog_beta_(k, w);
                max_log = std::max(max_log, log_phi[k]);
            }
            double norm = 0.0;
            for (int32_t k = 0; k < n_topics_; ++k) {
                log_phi[k] = std::exp(log_phi[k] - max_log);
                norm += log_phi[k];
            }
            norm = std::max(norm, eps_);
            for (int32_t k = 0; k < n_topics_; ++k) {
                assigned(k) += doc.cnts[j] * log_phi[k] / norm;
            }
        }
        theta_shape = assigned.array() + s0_;
        VectorXd e_theta = theta_shape.array() / theta_rate.array().max(1e-12);

        if (force_uniform_chi || force_uniform_chi_) {
            chi.setConstant(1.0 / static_cast<double>(n_clusters_));
        } else {
            const double pi_sum = pi_shape_.sum();
            double max_chi = -std::numeric_limits<double>::infinity();
            const double temp = std::max(chi_temperature_, 1.0);
            for (int32_t cc = 0; cc < n_clusters_; ++cc) {
                double lp = psi(pi_shape_(cc)) - psi(pi_sum);
                for (int32_t k = 0; k < n_topics_; ++k) {
                    lp += s0_ * expected_log_nu(cc, k) - expected_nu(cc, k) * e_theta(k);
                }
                lp /= temp;
                log_chi(cc) = lp;
                max_chi = std::max(max_chi, lp);
            }
            double chi_norm = 0.0;
            for (int32_t cc = 0; cc < n_clusters_; ++cc) {
                chi(cc) = std::exp(log_chi(cc) - max_chi);
                chi_norm += chi(cc);
            }
            chi_norm = std::max(chi_norm, eps_);
            chi /= chi_norm;
        }

        diff = (theta_shape - last_shape).cwiseAbs().sum() / static_cast<double>(n_topics_)
            + (chi - last_chi).cwiseAbs().sum() / static_cast<double>(n_clusters_);
        ++iter;
        if (diff < mean_change_tol_) {
            break;
        }
    }

    for (int32_t k = 0; k < n_topics_; ++k) {
        double prior_rate = 0.0;
        for (int32_t cc = 0; cc < n_clusters_; ++cc) {
            prior_rate += chi(cc) * expected_nu(cc, k);
        }
        theta_rate(k) = prior_rate + cexp * topic_capacity_(k);
        elog_theta(k) = psi(theta_shape(k)) - std::log(std::max(theta_rate(k), 1e-12));
    }
    return iter;
}

void GammaPoissonTopicJointC::infer_document_theta(VectorXd& theta_shape,
    VectorXd& theta_rate, VectorXd& elog_theta, const Document& doc,
    bool force_uniform_chi) const {
    VectorXd chi;
    fit_one_document(theta_shape, theta_rate, elog_theta, chi, doc, force_uniform_chi);
}

RowVectorXd GammaPoissonTopicJointC::normalized_chi_hat(const VectorXd& chi) const {
    RowVectorXd out(n_clusters_);
    double total = chi.sum();
    if (total <= 0.0 || !std::isfinite(total)) {
        out.setConstant(1.0 / static_cast<double>(n_clusters_));
    } else {
        for (int32_t c = 0; c < n_clusters_; ++c) out(c) = chi(c) / total;
    }
    return out;
}

void GammaPoissonTopicJointC::initialize_clusters_from_documents(DocumentView docs,
    double init_gamma) {
    const int32_t n_docs = static_cast<int32_t>(docs.size());
    if (n_docs <= 0) {
        error("%s: no documents available for cluster initialization", __func__);
    }
    RowMajorMatrixXd embed(n_docs, n_topics_);
    RowMajorMatrixXd raw_theta(n_docs, n_topics_);
    auto infer_doc = [&](int32_t d) {
        VectorXd theta_shape, theta_rate, elog_theta;
        infer_document_theta(theta_shape, theta_rate, elog_theta, docs[d], true);
        raw_theta.row(d) = (theta_shape.array() / theta_rate.array().max(1e-12)).matrix().transpose();
        embed.row(d) = normalized_theta_hat(theta_shape, theta_rate);
    };
    if (nThreads_ == 1) {
        for (int32_t d = 0; d < n_docs; ++d) infer_doc(d);
    } else {
        tbb::parallel_for(0, n_docs, [&](int32_t d) { infer_doc(d); });
    }

    RowMajorMatrixXd centers(n_clusters_, n_topics_);
    std::vector<int32_t> assignments(n_docs, 0);
    std::vector<double> min_dist(n_docs, std::numeric_limits<double>::infinity());
    std::mt19937 rng(static_cast<uint32_t>(seed_ + 7919 + update_count_));
    std::uniform_int_distribution<int32_t> first_dist(0, n_docs - 1);
    int32_t first = first_dist(rng);
    centers.row(0) = embed.row(first);

    for (int32_t c = 1; c < n_clusters_; ++c) {
        double total_dist = 0.0;
        for (int32_t d = 0; d < n_docs; ++d) {
            double dist = (embed.row(d) - centers.row(c - 1)).squaredNorm();
            if (dist < min_dist[d]) min_dist[d] = dist;
            total_dist += min_dist[d];
        }
        int32_t chosen = c % n_docs;
        if (total_dist > 0.0 && std::isfinite(total_dist)) {
            std::uniform_real_distribution<double> draw(0.0, total_dist);
            double target = draw(rng);
            double run = 0.0;
            for (int32_t d = 0; d < n_docs; ++d) {
                run += min_dist[d];
                if (run >= target) {
                    chosen = d;
                    break;
                }
            }
        }
        centers.row(c) = embed.row(chosen);
    }

    for (int32_t iter = 0; iter < 20; ++iter) {
        bool changed = false;
        for (int32_t d = 0; d < n_docs; ++d) {
            int32_t best = 0;
            double best_dist = std::numeric_limits<double>::infinity();
            for (int32_t c = 0; c < n_clusters_; ++c) {
                const double dist = (embed.row(d) - centers.row(c)).squaredNorm();
                if (dist < best_dist) {
                    best_dist = dist;
                    best = c;
                }
            }
            if (assignments[d] != best) {
                assignments[d] = best;
                changed = true;
            }
        }

        VectorXd counts = VectorXd::Zero(n_clusters_);
        centers.setZero();
        for (int32_t d = 0; d < n_docs; ++d) {
            centers.row(assignments[d]) += embed.row(d);
            counts(assignments[d]) += 1.0;
        }
        for (int32_t c = 0; c < n_clusters_; ++c) {
            if (counts(c) > 0.0) {
                centers.row(c) /= counts(c);
            } else {
                int32_t farthest = 0;
                double farthest_dist = -1.0;
                for (int32_t d = 0; d < n_docs; ++d) {
                    const double dist = (embed.row(d) - centers.row(assignments[d])).squaredNorm();
                    if (dist > farthest_dist) {
                        farthest_dist = dist;
                        farthest = d;
                    }
                }
                assignments[farthest] = c;
                centers.row(c) = embed.row(farthest);
                changed = true;
            }
        }
        if (!changed) break;
    }

    VectorXd counts = VectorXd::Zero(n_clusters_);
    RowMajorMatrixXd theta_sum = RowMajorMatrixXd::Zero(n_clusters_, n_topics_);
    for (int32_t d = 0; d < n_docs; ++d) {
        counts(assignments[d]) += 1.0;
        theta_sum.row(assignments[d]) += raw_theta.row(d);
    }

    const double scale = static_cast<double>(total_doc_count_) / static_cast<double>(n_docs);
    const double gamma_init = positive_or(init_gamma, gamma_);
    for (int32_t c = 0; c < n_clusters_; ++c) {
        const double scaled_count = scale * std::max(counts(c), 1e-6);
        pi_shape_(c) = std::max(gamma_init / static_cast<double>(n_clusters_) + scaled_count, 1e-12);
        cluster_usage_(c) = scaled_count;
        for (int32_t k = 0; k < n_topics_; ++k) {
            const double scaled_theta = scale * std::max(theta_sum(c, k), 1e-12);
            nu_shape_(c, k) = std::max(e0_ + s0_ * scaled_count, 1e-12);
            nu_rate_(c, k) = std::max(f0_ + scaled_theta, 1e-12);
        }
    }
    effective_gamma_ = gamma_init;
    force_uniform_chi_ = false;
    update_cluster_globals_ = true;
    if (verbose_ > 0) {
        notice("Initialized %d document clusters from %d documents after topic warmup",
            n_clusters_, n_docs);
    }
}

void GammaPoissonTopicJointC::partial_fit(const std::vector<Document>& docs) {
    const int32_t minibatch_size = static_cast<int32_t>(docs.size());
    if (minibatch_size == 0) return;

    tbb::combinable<RowMajorMatrixXd> ss_acc{
        [&] { return RowMajorMatrixXd::Zero(n_topics_, n_features_); }
    };
    tbb::combinable<VectorXd> ctheta_acc{
        [&] { return VectorXd::Zero(n_topics_); }
    };
    tbb::combinable<VectorXd> theta_acc{
        [&] { return VectorXd::Zero(n_topics_); }
    };
    tbb::combinable<VectorXd> cluster_count_acc{
        [&] { return VectorXd::Zero(n_clusters_); }
    };
    tbb::combinable<RowMajorMatrixXd> cluster_theta_acc{
        [&] { return RowMajorMatrixXd::Zero(n_clusters_, n_topics_); }
    };
    tbb::combinable<std::vector<int32_t>> niters_acc{
        [] { return std::vector<int32_t>(); }
    };

    tbb::parallel_for(tbb::blocked_range<int32_t>(0, minibatch_size),
        [&](const tbb::blocked_range<int32_t>& range) {
            auto& local_ss = ss_acc.local();
            auto& local_ctheta = ctheta_acc.local();
            auto& local_theta = theta_acc.local();
            auto& local_cluster_count = cluster_count_acc.local();
            auto& local_cluster_theta = cluster_theta_acc.local();
            auto& local_niters = niters_acc.local();
            VectorXd theta_shape, theta_rate, elog_theta, chi;
            std::vector<double> log_phi(n_topics_);
            for (int32_t d = range.begin(); d < range.end(); ++d) {
                const Document& doc = docs[d];
                const int32_t iter = fit_one_document(theta_shape, theta_rate, elog_theta, chi, doc);
                local_niters.push_back(iter);
                VectorXd e_theta = theta_shape.array() / theta_rate.array();
                local_theta += e_theta;
                local_ctheta += doc_exposure(doc) * e_theta;
                local_cluster_count += chi;
                for (int32_t cc = 0; cc < n_clusters_; ++cc) {
                    local_cluster_theta.row(cc) += chi(cc) * e_theta.transpose();
                }
                for (size_t j = 0; j < doc.ids.size(); ++j) {
                    const uint32_t w = doc.ids[j];
                    double max_log = -std::numeric_limits<double>::infinity();
                    for (int32_t k = 0; k < n_topics_; ++k) {
                        log_phi[k] = elog_theta(k) + elog_beta_(k, w);
                        max_log = std::max(max_log, log_phi[k]);
                    }
                    double norm = 0.0;
                    for (int32_t k = 0; k < n_topics_; ++k) {
                        log_phi[k] = std::exp(log_phi[k] - max_log);
                        norm += log_phi[k];
                    }
                    norm = std::max(norm, eps_);
                    for (int32_t k = 0; k < n_topics_; ++k) {
                        local_ss(k, w) += doc.cnts[j] * log_phi[k] / norm;
                    }
                }
            }
        });

    RowMajorMatrixXd ss = ss_acc.combine(
        [](const RowMajorMatrixXd& a, const RowMajorMatrixXd& b) { return a + b; });
    VectorXd ctheta = ctheta_acc.combine(
        [](const VectorXd& a, const VectorXd& b) { return a + b; });
    VectorXd theta = theta_acc.combine(
        [](const VectorXd& a, const VectorXd& b) { return a + b; });
    VectorXd cluster_count = cluster_count_acc.combine(
        [](const VectorXd& a, const VectorXd& b) { return a + b; });
    RowMajorMatrixXd cluster_theta = cluster_theta_acc.combine(
        [](const RowMajorMatrixXd& a, const RowMajorMatrixXd& b) { return a + b; });
    std::vector<int32_t> niters = niters_acc.combine(
        [](const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
            std::vector<int32_t> out = a;
            out.insert(out.end(), b.begin(), b.end());
            return out;
        });

    ++update_count_;
    const double rho = std::pow(learning_offset_ + update_count_, -learning_decay_);
    const double scale = static_cast<double>(total_doc_count_) / static_cast<double>(minibatch_size);
    VectorXd xi_mean(n_features_);
    for (int32_t w = 0; w < n_features_; ++w) {
        xi_mean(w) = expected_xi(w);
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        const double scaled_ctheta = scale * ctheta(k);
        for (int32_t w = 0; w < n_features_; ++w) {
            const double target_shape = a_ + scale * ss(k, w);
            const double target_rate = xi_mean(w) + scaled_ctheta;
            beta_shape_(k, w) = (1.0 - rho) * beta_shape_(k, w) + rho * target_shape;
            beta_rate_(k, w) = (1.0 - rho) * beta_rate_(k, w) + rho * std::max(target_rate, 1e-12);
        }
    }
    refresh_cache();

    for (int32_t w = 0; w < n_features_; ++w) {
        xi_shape_(w) = a0_ + a_ * n_topics_;
        xi_rate_(w) = a0_ / b0_ + e_beta_.col(w).sum();
    }
    if (update_cluster_globals_) {
        for (int32_t cc = 0; cc < n_clusters_; ++cc) {
            const double target_pi = effective_gamma_ / static_cast<double>(n_clusters_)
                + scale * cluster_count(cc);
            pi_shape_(cc) = (1.0 - rho) * pi_shape_(cc) + rho * std::max(target_pi, 1e-12);
            for (int32_t k = 0; k < n_topics_; ++k) {
                const double target_shape = e0_ + s0_ * scale * cluster_count(cc);
                const double target_rate = f0_ + scale * cluster_theta(cc, k);
                nu_shape_(cc, k) = (1.0 - rho) * nu_shape_(cc, k) + rho * std::max(target_shape, 1e-12);
                nu_rate_(cc, k) = (1.0 - rho) * nu_rate_(cc, k) + rho * std::max(target_rate, 1e-12);
            }
        }
    }
    VectorXd target_usage = scale * theta.array() * topic_capacity_.array();
    topic_usage_ = (topic_usage_.size() != n_topics_ || topic_usage_.sum() <= 0.0)
        ? target_usage : (1.0 - rho) * topic_usage_ + rho * target_usage;
    if (update_cluster_globals_) {
        VectorXd target_cluster_usage = scale * cluster_count;
        cluster_usage_ = (cluster_usage_.size() != n_clusters_ || cluster_usage_.sum() <= 0.0)
            ? target_cluster_usage : (1.0 - rho) * cluster_usage_ + rho * target_cluster_usage;
    }

    if (verbose_ > 0 && !niters.empty()) {
        const double avg = std::accumulate(niters.begin(), niters.end(), 0.0)
            / static_cast<double>(niters.size());
        notice("Joint Gamma-Poisson partial fit: %d documents. Average iterations per doc: %.2f.",
            minibatch_size, avg);
    }
}

void GammaPoissonTopicJointC::transform_both(DocumentView docs,
    RowMajorMatrixXd& topics, RowMajorMatrixXd& clusters) {
    const int32_t n_docs = static_cast<int32_t>(docs.size());
    topics.resize(n_docs, n_topics_);
    clusters.resize(n_docs, n_clusters_);
    auto process_doc = [&](int32_t d) {
        VectorXd theta_shape, theta_rate, elog_theta, chi;
        fit_one_document(theta_shape, theta_rate, elog_theta, chi, docs[d]);
        topics.row(d) = normalized_theta_hat(theta_shape, theta_rate);
        clusters.row(d) = normalized_chi_hat(chi);
    };
    if (nThreads_ == 1) {
        for (int32_t d = 0; d < n_docs; ++d) process_doc(d);
    } else {
        tbb::parallel_for(0, n_docs, [&](int32_t d) { process_doc(d); });
    }
}

RowMajorMatrixXd GammaPoissonTopicJointC::transform(DocumentView docs) {
    RowMajorMatrixXd topics, clusters;
    transform_both(docs, topics, clusters);
    return topics;
}

RowMajorMatrixXd GammaPoissonTopicJointC::transform_clusters(DocumentView docs) {
    RowMajorMatrixXd topics, clusters;
    transform_both(docs, topics, clusters);
    return clusters;
}

void GammaPoissonTopicJointC::write_state(const std::string& outFile,
    const std::vector<std::string>& featureNames) {
    std::ofstream out(outFile);
    if (!out) {
        error("%s: Error opening output file: %s", __func__, outFile.c_str());
    }
    out << std::setprecision(17);
    out << "#punkst_gamma_pois_jointc_state_v1\n";
    out << "#n_topics\t" << n_topics_ << "\n";
    out << "#n_features\t" << n_features_ << "\n";
    out << "#n_clusters\t" << n_clusters_ << "\n";
    out << "#total_doc_count\t" << total_doc_count_ << "\n";
    out << "#size_factor\t" << size_factor_ << "\n";
    out << "#beta_shape_prior\t" << a_ << "\n";
    out << "#xi_shape_prior\t" << a0_ << "\n";
    out << "#xi_mean_prior\t" << b0_ << "\n";
    out << "#theta_shape_prior\t" << s0_ << "\n";
    out << "#nu_shape_prior\t" << e0_ << "\n";
    out << "#nu_rate_prior\t" << f0_ << "\n";
    out << "#cluster_prior\t" << gamma_ << "\n";
    out << "#learning_decay\t" << learning_decay_ << "\n";
    out << "#learning_offset\t" << learning_offset_ << "\n";
    out << "#update_count\t" << update_count_ << "\n";
    out << "#pi_shape";
    for (int32_t c = 0; c < n_clusters_; ++c) out << "\t" << pi_shape_(c);
    out << "\n#topic_usage";
    for (int32_t k = 0; k < n_topics_; ++k) out << "\t" << topic_usage_(k);
    out << "\n#cluster_usage";
    for (int32_t c = 0; c < n_clusters_; ++c) out << "\t" << cluster_usage_(c);
    out << "\n#nu_shape";
    for (int32_t c = 0; c < n_clusters_; ++c)
        for (int32_t k = 0; k < n_topics_; ++k) out << "\t" << nu_shape_(c, k);
    out << "\n#nu_rate";
    for (int32_t c = 0; c < n_clusters_; ++c)
        for (int32_t k = 0; k < n_topics_; ++k) out << "\t" << nu_rate_(c, k);
    out << "\nFeature";
    for (int32_t k = 0; k < n_topics_; ++k) {
        out << "\tbeta_shape_" << k << "\tbeta_rate_" << k;
    }
    out << "\txi_shape\txi_rate\n";
    for (int32_t w = 0; w < n_features_; ++w) {
        const std::string feature = w < static_cast<int32_t>(featureNames.size())
            ? featureNames[w] : std::to_string(w);
        out << feature;
        for (int32_t k = 0; k < n_topics_; ++k) {
            out << "\t" << beta_shape_(k, w) << "\t" << beta_rate_(k, w);
        }
        out << "\t" << xi_shape_(w) << "\t" << xi_rate_(w) << "\n";
    }
}

std::vector<std::string> GammaPoissonTopicJointC::read_state_feature_names(
    const std::string& stateFile) {
    return GammaPoissonTopicModel::read_state_feature_names(stateFile);
}

void GammaPoissonTopicJointC::read_state(const std::string& stateFile) {
    std::ifstream in(stateFile);
    if (!in) {
        error("%s: Error opening state file: %s", __func__, stateFile.c_str());
    }
    std::string line;
    std::vector<std::vector<double>> bshape_rows, brate_rows;
    std::vector<double> xi_shape_vals, xi_rate_vals;
    std::vector<double> pi_vals, topic_usage_vals, cluster_usage_vals, nu_shape_vals, nu_rate_vals;
    bool saw_header = false;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') {
            std::string payload = line.substr(1);
            auto tok = split_ws(payload);
            if (tok.empty()) continue;
            if (tok[0] == "n_topics" && tok.size() > 1) n_topics_ = std::stoi(tok[1]);
            else if (tok[0] == "n_features" && tok.size() > 1) n_features_ = std::stoi(tok[1]);
            else if (tok[0] == "n_clusters" && tok.size() > 1) n_clusters_ = std::stoi(tok[1]);
            else if (tok[0] == "total_doc_count" && tok.size() > 1) total_doc_count_ = std::stoi(tok[1]);
            else if (tok[0] == "size_factor" && tok.size() > 1) size_factor_ = std::stod(tok[1]);
            else if (tok[0] == "beta_shape_prior" && tok.size() > 1) a_ = std::stod(tok[1]);
            else if (tok[0] == "xi_shape_prior" && tok.size() > 1) a0_ = std::stod(tok[1]);
            else if (tok[0] == "xi_mean_prior" && tok.size() > 1) b0_ = std::stod(tok[1]);
            else if (tok[0] == "theta_shape_prior" && tok.size() > 1) s0_ = std::stod(tok[1]);
            else if (tok[0] == "nu_shape_prior" && tok.size() > 1) e0_ = std::stod(tok[1]);
            else if (tok[0] == "nu_rate_prior" && tok.size() > 1) f0_ = std::stod(tok[1]);
            else if (tok[0] == "cluster_prior" && tok.size() > 1) gamma_ = std::stod(tok[1]);
            else if (tok[0] == "learning_decay" && tok.size() > 1) learning_decay_ = std::stod(tok[1]);
            else if (tok[0] == "learning_offset" && tok.size() > 1) learning_offset_ = std::stod(tok[1]);
            else if (tok[0] == "update_count" && tok.size() > 1) update_count_ = std::stoi(tok[1]);
            else if (tok[0] == "pi_shape") {
                pi_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) pi_vals.push_back(std::stod(tok[i]));
            } else if (tok[0] == "topic_usage") {
                topic_usage_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) topic_usage_vals.push_back(std::stod(tok[i]));
            } else if (tok[0] == "cluster_usage") {
                cluster_usage_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) cluster_usage_vals.push_back(std::stod(tok[i]));
            } else if (tok[0] == "nu_shape") {
                nu_shape_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) nu_shape_vals.push_back(std::stod(tok[i]));
            } else if (tok[0] == "nu_rate") {
                nu_rate_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) nu_rate_vals.push_back(std::stod(tok[i]));
            }
            continue;
        }
        auto tok = split_ws(line);
        if (tok.empty()) continue;
        if (tok[0] == "Feature") {
            saw_header = true;
            continue;
        }
        if (!saw_header || n_topics_ <= 0) {
            error("%s: Invalid Gamma-Poisson joint state file header in %s", __func__, stateFile.c_str());
        }
        const size_t expected = 1 + static_cast<size_t>(2 * n_topics_) + 2;
        if (tok.size() != expected) {
            error("%s: Invalid joint state row with %zu columns, expected %zu", __func__, tok.size(), expected);
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
    }
    if (n_features_ <= 0) n_features_ = static_cast<int32_t>(bshape_rows.size());
    if (n_clusters_ <= 0) n_clusters_ = 1;
    if (static_cast<int32_t>(bshape_rows.size()) != n_features_) {
        error("%s: Joint state file has %zu feature rows but metadata says %d",
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
    for (int32_t w = 0; w < n_features_; ++w) {
        xi_shape_(w) = xi_shape_vals[w];
        xi_rate_(w) = xi_rate_vals[w];
    }
    init_cluster_state();
    topic_usage_ = VectorXd::Constant(n_topics_, 1.0);
    if (static_cast<int32_t>(pi_vals.size()) == n_clusters_) {
        for (int32_t c = 0; c < n_clusters_; ++c) pi_shape_(c) = pi_vals[c];
    }
    if (static_cast<int32_t>(topic_usage_vals.size()) == n_topics_) {
        for (int32_t k = 0; k < n_topics_; ++k) topic_usage_(k) = topic_usage_vals[k];
    }
    if (static_cast<int32_t>(cluster_usage_vals.size()) == n_clusters_) {
        for (int32_t c = 0; c < n_clusters_; ++c) cluster_usage_(c) = cluster_usage_vals[c];
    }
    if (static_cast<int32_t>(nu_shape_vals.size()) == n_clusters_ * n_topics_) {
        size_t pos = 0;
        for (int32_t c = 0; c < n_clusters_; ++c)
            for (int32_t k = 0; k < n_topics_; ++k) nu_shape_(c, k) = nu_shape_vals[pos++];
    }
    if (static_cast<int32_t>(nu_rate_vals.size()) == n_clusters_ * n_topics_) {
        size_t pos = 0;
        for (int32_t c = 0; c < n_clusters_; ++c)
            for (int32_t k = 0; k < n_topics_; ++k) nu_rate_(c, k) = nu_rate_vals[pos++];
    }
    effective_gamma_ = gamma_;
    chi_temperature_ = 1.0;
    force_uniform_chi_ = false;
    update_cluster_globals_ = true;
}

void GammaPoisson4Hex::initialize(int32_t nTopics, int32_t seed, int32_t nThreads,
    int32_t verbose, double beta_shape, double xi_shape, double xi_mean,
    double theta_shape, double theta_concentration, double nu_shape,
    double nu_rate, double kappa, double tau0,
    int32_t totalDocCount, double sizeFactor, bool symmetricNu, double nuMax,
    int32_t maxIter, double mDelta) {
    if (reader.features.size() != static_cast<size_t>(M_)) {
        featureNames.resize(M_);
        for (int32_t i = 0; i < M_; ++i) featureNames[i] = std::to_string(i);
    } else {
        featureNames = reader.features;
    }
    const std::vector<double>& sums = reader.getFeatureSums();
    model_ = std::make_unique<GammaPoissonTopicModel>(
        nTopics, M_, seed, nThreads, verbose, beta_shape, xi_shape, xi_mean,
        theta_shape, theta_concentration, nu_shape, nu_rate, kappa, tau0,
        totalDocCount, sizeFactor,
        symmetricNu, nuMax, reader.readFullSums ? &sums : nullptr);
    model_->set_svb_parameters(maxIter, mDelta);
    initialized = true;
}

void GammaPoisson4Hex::setFeatureDispersion(const std::vector<double>& tau) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->set_feature_dispersion(tau);
}

GammaPoissonDispersionResult GammaPoisson4Hex::estimateFeatureDispersion(
    const GammaPoissonDispersionOptions& options, const std::string& inFile,
    int32_t batchSize_, int32_t minCountTrain_, int32_t maxUnits) {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    GammaPoissonDispersionEstimator estimator(M_, options);
    std::ifstream in(inFile);
    if (!in) error("%s: Error opening input file: %s", __func__, inFile.c_str());
    std::vector<Document> docs;
    std::vector<std::string> ids;
    int32_t processed = 0;
    while (processed < maxUnits) {
        const int32_t remaining = maxUnits == INT32_MAX ? INT32_MAX : maxUnits - processed;
        const bool more = readMinibatch(in, docs, ids, batchSize_, minCountTrain_, remaining);
        if (!docs.empty()) {
            estimator.accumulate(*model_, DocumentView(docs));
            processed += static_cast<int32_t>(docs.size());
        }
        if (!more || docs.empty()) break;
    }
    GammaPoissonDispersionResult result = estimator.finish();
    model_->set_feature_dispersion(result.tau);
    return result;
}

GammaPoissonDispersionResult GammaPoisson4Hex::estimateFeatureDispersion10X(
    const GammaPoissonDispersionOptions& options, int32_t batchSize_, int32_t maxUnits) {
    if (!initialized || !model_) error("%s: GammaPoisson4Hex is not initialized", __func__);
    if (!dge_cache_ready_) error("%s: 10X cache is not initialized", __func__);
    GammaPoissonDispersionEstimator estimator(M_, options);
    int32_t processed = 0;
    std::vector<Document> docs;
    for (int32_t idx : dge_train_idx_cache_) {
        if (processed + static_cast<int32_t>(docs.size()) >= maxUnits) break;
        docs.push_back(dge_docs_cache_[idx]);
        if (static_cast<int32_t>(docs.size()) >= batchSize_) {
            estimator.accumulate(*model_, DocumentView(docs));
            processed += static_cast<int32_t>(docs.size());
            docs.clear();
        }
    }
    if (!docs.empty() && processed < maxUnits) {
        estimator.accumulate(*model_, DocumentView(docs));
        processed += static_cast<int32_t>(docs.size());
    }
    GammaPoissonDispersionResult result = estimator.finish();
    model_->set_feature_dispersion(result.tau);
    return result;
}

void GammaPoisson4Hex::initialize_transform(const std::string& stateFile, int32_t seed,
    int32_t nThreads, int32_t verbose, int32_t maxIter, double mDelta) {
    model_ = std::make_unique<GammaPoissonTopicModel>(stateFile, seed, nThreads, verbose);
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

void GammaPoisson4Hex::transformWithPosteriors(DocumentView batch,
    RowMajorMatrixXd& topics,
    std::vector<GammaPoissonDocumentPosterior>& posteriors) const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->transform_with_posteriors(batch, topics, posteriors);
}

void GammaPoisson4Hex::dispersionCovarianceApproximation(const Document& doc,
    const GammaPoissonDocumentPosterior& posterior, int32_t rank,
    uint64_t seed, GammaPoissonDispersionApproximation& out) const {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->dispersion_covariance_approximation(doc, posterior, rank, seed, out);
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

void GammaPoisson4Hex::getTopicAbundance(std::vector<double>& topic_weights) {
    get_topic_abundance(topic_weights);
}

void GammaPoisson4Hex::get_topic_abundance(std::vector<double>& topic_weights) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->get_topic_abundance(topic_weights);
}

void GammaPoissonJointC4Hex::initialize(int32_t nTopics, int32_t nClusters,
    int32_t seed, int32_t nThreads, int32_t verbose, double beta_shape,
    double xi_shape, double xi_mean, double theta_shape, double nu_shape,
    double nu_rate, double clusterPrior, double kappa, double tau0,
    int32_t totalDocCount, double sizeFactor, int32_t maxIter, double mDelta) {
    if (reader.features.size() != static_cast<size_t>(M_)) {
        featureNames.resize(M_);
        for (int32_t i = 0; i < M_; ++i) featureNames[i] = std::to_string(i);
    } else {
        featureNames = reader.features;
    }
    const std::vector<double>& sums = reader.getFeatureSums();
    model_ = std::make_unique<GammaPoissonTopicJointC>(
        nTopics, M_, nClusters, seed, nThreads, verbose, beta_shape, xi_shape,
        xi_mean, theta_shape, nu_shape, nu_rate, clusterPrior, kappa, tau0,
        totalDocCount, sizeFactor, reader.readFullSums ? &sums : nullptr);
    model_->set_svb_parameters(maxIter, mDelta);
    initialized = true;
}

void GammaPoissonJointC4Hex::initialize_transform(const std::string& stateFile,
    int32_t seed, int32_t nThreads, int32_t verbose, int32_t maxIter, double mDelta) {
    model_ = std::make_unique<GammaPoissonTopicJointC>(stateFile, seed, nThreads, verbose);
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

double GammaPoissonJointC4Hex::resolveSizeFactor(double requested) const {
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

void GammaPoissonJointC4Hex::writeModelToFile(const std::string& outFile) {
    if (!initialized || !model_) {
        error("%s: GammaPoissonJointC4Hex is not initialized", __func__);
    }
    model_->write_model(outFile, featureNames);
}

void GammaPoissonJointC4Hex::writeStateToFile(const std::string& outFile) {
    if (!initialized || !model_) {
        error("%s: GammaPoissonJointC4Hex is not initialized", __func__);
    }
    model_->write_state(outFile, featureNames);
}

void GammaPoissonJointC4Hex::getUnitHeaderCols(std::vector<std::string>& outCols) {
    outCols = get_topic_names();
}

void GammaPoissonJointC4Hex::getClusterHeaderCols(std::vector<std::string>& outCols) {
    outCols.clear();
    outCols.push_back("cluster");
    const auto& names = get_cluster_names();
    outCols.insert(outCols.end(), names.begin(), names.end());
}

void GammaPoissonJointC4Hex::setClusterWarmup(bool enabled) {
    if (!initialized || !model_) {
        error("%s: GammaPoissonJointC4Hex is not initialized", __func__);
    }
    model_->set_cluster_warmup(enabled);
}

void GammaPoissonJointC4Hex::setClusterTemperature(double temperature) {
    if (!initialized || !model_) {
        error("%s: GammaPoissonJointC4Hex is not initialized", __func__);
    }
    model_->set_cluster_temperature(temperature);
}

void GammaPoissonJointC4Hex::setEffectiveClusterPrior(double gamma) {
    if (!initialized || !model_) {
        error("%s: GammaPoissonJointC4Hex is not initialized", __func__);
    }
    model_->set_effective_cluster_prior(gamma);
}

void GammaPoissonJointC4Hex::initializeClustersFromTrainingData(const std::string& inFile,
    bool use10x, int32_t minCountTrain, int32_t maxUnits, double initGamma) {
    if (!initialized || !model_) {
        error("%s: GammaPoissonJointC4Hex is not initialized", __func__);
    }
    std::vector<Document> docs;
    if (use10x) {
        if (!dge_cache_ready_) error("10X cache is not initialized; call prepare10XCache() first");
        const int32_t take_max = maxUnits > 0 ? maxUnits : INT32_MAX;
        docs.reserve(std::min(static_cast<int32_t>(dge_train_idx_cache_.size()), take_max));
        for (int32_t idx : dge_train_idx_cache_) {
            if (static_cast<int32_t>(docs.size()) >= take_max) break;
            docs.push_back(dge_docs_cache_[idx]);
        }
    } else {
        readAllDocuments(docs, inFile, minCountTrain, maxUnits);
        for (Document& doc : docs) {
            applyWeights(doc);
        }
    }
    model_->initialize_clusters_from_documents(DocumentView(docs), initGamma);
}

const RowMajorMatrixXd& GammaPoissonJointC4Hex::get_model_matrix() const {
    if (!model_) {
        return empty_model_;
    }
    return const_cast<GammaPoissonTopicJointC*>(model_.get())->get_model();
}

RowMajorMatrixXd GammaPoissonJointC4Hex::copy_model_matrix() const {
    if (!model_) {
        return RowMajorMatrixXd();
    }
    return const_cast<GammaPoissonTopicJointC*>(model_.get())->copy_model();
}

const std::vector<std::string>& GammaPoissonJointC4Hex::get_topic_names() {
    if (!model_) {
        topicNames_.clear();
        return topicNames_;
    }
    return model_->get_topic_names();
}

const std::vector<std::string>& GammaPoissonJointC4Hex::get_cluster_names() {
    if (!model_) {
        clusterNames_.clear();
        return clusterNames_;
    }
    return model_->get_cluster_names();
}

void GammaPoissonJointC4Hex::do_partial_fit(const std::vector<Document>& batch) {
    model_->partial_fit(batch);
}

MatrixXd GammaPoissonJointC4Hex::do_transform(DocumentView batch) {
    return model_->transform(batch);
}

void GammaPoissonJointC4Hex::do_transform_both(DocumentView batch,
    RowMajorMatrixXd& topics, RowMajorMatrixXd& clusters) {
    model_->transform_both(batch, topics, clusters);
}

void GammaPoissonJointC4Hex::getTopicAbundance(std::vector<double>& topic_weights) {
    if (!initialized || !model_) {
        error("%s: GammaPoissonJointC4Hex is not initialized", __func__);
    }
    model_->get_topic_abundance(topic_weights);
}

void GammaPoissonJointC4Hex::getClusterAbundance(std::vector<double>& cluster_weights) {
    if (!initialized || !model_) {
        error("%s: GammaPoissonJointC4Hex is not initialized", __func__);
    }
    model_->get_cluster_abundance(cluster_weights);
}
