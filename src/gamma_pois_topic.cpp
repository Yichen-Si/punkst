#include "gamma_pois_topic.hpp"

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

int GammaPoissonTopicModel::normalize_seed(int seed) {
    return seed > 0 ? seed : static_cast<int>(std::random_device{}());
}

GammaPoissonTopicModel::GammaPoissonTopicModel(int32_t n_topics, int32_t n_features,
    int seed, int32_t nThreads, int32_t verbose, double beta_shape, double xi_shape,
    double xi_mean, double theta_shape, double nu_shape, double nu_rate,
    double learning_decay, double learning_offset, int32_t total_doc_count,
    double size_factor, bool symmetric_nu, const std::vector<double>* feature_sums)
    : n_topics_(n_topics), n_features_(n_features), seed_(normalize_seed(seed)),
      nThreads_(nThreads), verbose_(verbose),
      total_doc_count_(total_doc_count > 0 ? total_doc_count : 1000000),
      symmetric_nu_(symmetric_nu),
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

GammaPoissonTopicModel::GammaPoissonTopicModel(const std::string& stateFile,
    int seed, int32_t nThreads, int32_t verbose)
    : seed_(normalize_seed(seed)), nThreads_(nThreads), verbose_(verbose) {
    random_engine_.seed(seed_);
    read_state(stateFile);
    set_nthreads(nThreads_);
    refresh_cache();
}

void GammaPoissonTopicModel::set_nthreads(int32_t nThreads) {
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

void GammaPoissonTopicModel::set_svb_parameters(int32_t max_iter, double tol) {
    max_doc_update_iter_ = max_iter > 0 ? max_iter : 100;
    mean_change_tol_ = tol > 0.0 ? tol : 1e-3;
}

void GammaPoissonTopicModel::init_from_feature_sums(const std::vector<double>* feature_sums) {
    beta_shape_.resize(n_topics_, n_features_);
    beta_rate_.resize(n_topics_, n_features_);
    xi_shape_ = VectorXd::Constant(n_features_, a0_ + a_ * n_topics_);
    xi_rate_ = VectorXd::Constant(n_features_, a0_ / b0_ + n_topics_ * a_ / b0_);
    nu_shape_ = VectorXd::Constant(n_topics_, e0_ + s0_ * total_doc_count_);
    nu_rate_ = VectorXd::Constant(n_topics_, f0_ + total_doc_count_ * s0_);
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

void GammaPoissonTopicModel::refresh_cache() {
    e_beta_.resize(n_topics_, n_features_);
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

double GammaPoissonTopicModel::doc_exposure(const Document& doc) const {
    const double len = doc_sum_const(doc);
    if (len <= 0.0) {
        return 0.0;
    }
    return len / size_factor_;
}

int32_t GammaPoissonTopicModel::fit_one_document(VectorXd& theta_shape,
    VectorXd& theta_rate, VectorXd& elog_theta, const Document& doc) const {
    const int32_t n_ids = static_cast<int32_t>(doc.ids.size());
    theta_shape = VectorXd::Constant(n_topics_, s0_);
    theta_rate.resize(n_topics_);
    const double c = doc_exposure(doc);
    for (int32_t k = 0; k < n_topics_; ++k) {
        theta_rate(k) = expected_nu(k) + c * topic_capacity_(k);
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
        for (int32_t k = 0; k < n_topics_; ++k) {
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
        theta_shape = assigned.array() + s0_;
        diff = (theta_shape - last_shape).cwiseAbs().sum() / static_cast<double>(n_topics_);
        ++iter;
        if (diff < mean_change_tol_) {
            break;
        }
    }
    for (int32_t k = 0; k < n_topics_; ++k) {
        elog_theta(k) = psi(theta_shape(k)) - std::log(theta_rate(k));
    }
    return iter;
}

RowVectorXd GammaPoissonTopicModel::normalized_theta_hat(const VectorXd& theta_shape,
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
                local_ctheta += doc_exposure(doc) * e_theta;
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
    if (!symmetric_nu_) {
        for (int32_t k = 0; k < n_topics_; ++k) {
            nu_shape_(k) = e0_ + s0_ * total_doc_count_;
            const double target_rate = f0_ + scale * theta(k);
            nu_rate_(k) = (1.0 - rho) * nu_rate_(k) + rho * std::max(target_rate, 1e-12);
        }
    }
    refresh_cache();
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
    const int32_t n_docs = static_cast<int32_t>(docs.size());
    RowMajorMatrixXd out(n_docs, n_topics_);
    auto process_doc = [&](int32_t d) {
        VectorXd theta_shape, theta_rate, elog_theta;
        fit_one_document(theta_shape, theta_rate, elog_theta, docs[d]);
        out.row(d) = normalized_theta_hat(theta_shape, theta_rate);
    };
    if (nThreads_ == 1) {
        for (int32_t d = 0; d < n_docs; ++d) {
            process_doc(d);
        }
    } else {
        tbb::parallel_for(0, n_docs, [&](int32_t d) { process_doc(d); });
    }
    return out;
}

const std::vector<std::string>& GammaPoissonTopicModel::get_topic_names() {
    if (topic_names_.empty()) {
        topic_names_.resize(n_topics_);
        for (int32_t k = 0; k < n_topics_; ++k) {
            topic_names_[k] = std::to_string(k);
        }
    }
    return topic_names_;
}

const RowMajorMatrixXd& GammaPoissonTopicModel::get_model() {
    return model_phi_;
}

RowMajorMatrixXd GammaPoissonTopicModel::copy_model() {
    return model_phi_;
}

void GammaPoissonTopicModel::get_topic_abundance(std::vector<double>& weights) const {
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

void GammaPoissonTopicModel::write_model(const std::string& outFile,
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
    out << "#punkst_gamma_pois_state_v1\n";
    out << "#n_topics\t" << n_topics_ << "\n";
    out << "#n_features\t" << n_features_ << "\n";
    out << "#total_doc_count\t" << total_doc_count_ << "\n";
    out << "#size_factor\t" << size_factor_ << "\n";
    out << "#beta_shape_prior\t" << a_ << "\n";
    out << "#xi_shape_prior\t" << a0_ << "\n";
    out << "#xi_mean_prior\t" << b0_ << "\n";
    out << "#theta_shape_prior\t" << s0_ << "\n";
    out << "#nu_shape_prior\t" << e0_ << "\n";
    out << "#nu_rate_prior\t" << f0_ << "\n";
    out << "#learning_decay\t" << learning_decay_ << "\n";
    out << "#learning_offset\t" << learning_offset_ << "\n";
    out << "#update_count\t" << update_count_ << "\n";
    out << "#symmetric_nu\t" << (symmetric_nu_ ? 1 : 0) << "\n";
    out << "#nu_shape";
    for (int32_t k = 0; k < n_topics_; ++k) out << "\t" << nu_shape_(k);
    out << "\n#nu_rate";
    for (int32_t k = 0; k < n_topics_; ++k) out << "\t" << nu_rate_(k);
    out << "\n#topic_usage";
    for (int32_t k = 0; k < n_topics_; ++k) out << "\t" << topic_usage_(k);
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
            else if (tok[0] == "beta_shape_prior" && tok.size() > 1) a_ = std::stod(tok[1]);
            else if (tok[0] == "xi_shape_prior" && tok.size() > 1) a0_ = std::stod(tok[1]);
            else if (tok[0] == "xi_mean_prior" && tok.size() > 1) b0_ = std::stod(tok[1]);
            else if (tok[0] == "theta_shape_prior" && tok.size() > 1) s0_ = std::stod(tok[1]);
            else if (tok[0] == "nu_shape_prior" && tok.size() > 1) e0_ = std::stod(tok[1]);
            else if (tok[0] == "nu_rate_prior" && tok.size() > 1) f0_ = std::stod(tok[1]);
            else if (tok[0] == "learning_decay" && tok.size() > 1) learning_decay_ = std::stod(tok[1]);
            else if (tok[0] == "learning_offset" && tok.size() > 1) learning_offset_ = std::stod(tok[1]);
            else if (tok[0] == "update_count" && tok.size() > 1) update_count_ = std::stoi(tok[1]);
            else if (tok[0] == "symmetric_nu" && tok.size() > 1) symmetric_nu_ = std::stoi(tok[1]) != 0;
            else if (tok[0] == "nu_shape") {
                nu_shape_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) nu_shape_vals.push_back(std::stod(tok[i]));
            } else if (tok[0] == "nu_rate") {
                nu_rate_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) nu_rate_vals.push_back(std::stod(tok[i]));
            } else if (tok[0] == "topic_usage") {
                topic_usage_vals.clear();
                for (size_t i = 1; i < tok.size(); ++i) topic_usage_vals.push_back(std::stod(tok[i]));
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
    topic_usage_ = VectorXd::Constant(n_topics_, 1.0);
    if (static_cast<int32_t>(topic_usage_vals.size()) == n_topics_) {
        for (int32_t k = 0; k < n_topics_; ++k) topic_usage_(k) = topic_usage_vals[k];
    }
}

void GammaPoisson4Hex::initialize(int32_t nTopics, int32_t seed, int32_t nThreads,
    int32_t verbose, double beta_shape, double xi_shape, double xi_mean,
    double theta_shape, double nu_shape, double nu_rate, double kappa, double tau0,
    int32_t totalDocCount, double sizeFactor, bool symmetricNu,
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
        theta_shape, nu_shape, nu_rate, kappa, tau0, totalDocCount, sizeFactor,
        symmetricNu, reader.readFullSums ? &sums : nullptr);
    model_->set_svb_parameters(maxIter, mDelta);
    initialized = true;
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

void GammaPoisson4Hex::getTopicAbundance(std::vector<double>& topic_weights) {
    get_topic_abundance(topic_weights);
}

void GammaPoisson4Hex::get_topic_abundance(std::vector<double>& topic_weights) {
    if (!initialized || !model_) {
        error("%s: GammaPoisson4Hex is not initialized", __func__);
    }
    model_->get_topic_abundance(topic_weights);
}
