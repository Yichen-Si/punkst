#include "hdp.hpp"


VectorXd HDP::fit_one_document(MatrixXd& zeta, MatrixXd& phi, const Document &doc) {
    uint32_t m = doc.ids.size();
    VectorXd xvec(m);
    zeta.resize(T_, K_);
    phi.resize(m, T_);

    // Create a submatrix from Elog_beta_ for the words present in the document
    MatrixXd local_Elog_beta(K_, m);
    for (size_t j = 0; j < m; j++) {
        xvec(j) = doc.cnts[j];
        local_Elog_beta.col(j) = Elog_beta_.col(doc.ids[j]);
    }

    // Initialization (numerically stable)
    MatrixXd log_zeta0 = (local_Elog_beta * xvec).transpose().replicate(T_, 1); // T x K
    for (int t = 0; t < T_; ++t) {
        double max_val = log_zeta0.row(t).maxCoeff();
        zeta.row(t) = (log_zeta0.row(t).array() - max_val).exp();
        zeta.row(t) /= zeta.row(t).sum();
    }

    MatrixXd log_phi_init = (zeta * local_Elog_beta).transpose(); // m x T
    for (int j = 0; j < m; ++j) {
        double max_val = log_phi_init.row(j).maxCoeff();
        phi.row(j) = (log_phi_init.row(j).array() - max_val).exp();
        phi.row(j) /= phi.row(j).sum();
    }

    // Iterative Updates
    MatrixXd upperT = MatrixXd::Zero(T_, T_);
    for (uint32_t i = 0; i < T_ - 1; i++) {
        for (uint32_t j = i + 1; j < T_; j++) {
            upperT(i, j) = 1;
        }
    }

    double diff = 1.0;
    int iter = 0;
    VectorXd gamma1(T_), gamma2(T_);
    RowVectorXd Elog_sigma_T(T_);
    VectorXd onesT = VectorXd::Ones(T_);
    RowVectorXd doc_topic = xvec.transpose() * phi * zeta;

    for (; iter < max_doc_update_iter_ && diff > mean_change_tol_; iter++) {
        RowVectorXd last_doc_topic = doc_topic;
        VectorXd phiTx = phi.transpose() * xvec;

        // Update local stick-breaking parameters (gamma)
        gamma1 = onesT + phiTx;
        gamma2 = alpha_ * onesT + upperT * phiTx;
        Elog_sigma_T = expect_log_sticks(gamma1, gamma2).transpose();

        // --- CORRECTED STABLE ZETA UPDATE ---
        // Pre-scale Elog_beta by word counts for efficiency
        MatrixXd scaled_Elog_beta = local_Elog_beta;
        for(int j=0; j<m; ++j) {
            scaled_Elog_beta.col(j) *= xvec(j);
        }

        for (int t = 0; t < T_; ++t) {
            // The term sum_j(phi_jt * x_j * E[log B_kj]) for all k
            // is a (K x m) * (m x 1) matrix-vector product.
            VectorXd sum_term = scaled_Elog_beta * phi.col(t); // Result is K x 1

            // Add global stick expectations and apply Log-Sum-Exp
            VectorXd log_zeta_row = Elog_sigma_K_ + sum_term;
            double max_val = log_zeta_row.maxCoeff();
            zeta.row(t) = (log_zeta_row.array() - max_val).exp().transpose();
            zeta.row(t) /= zeta.row(t).sum();
        }

        // STABLE PHI UPDATE (This part was already correct)
        MatrixXd log_phi_update = Elog_sigma_T.replicate(m, 1);
        log_phi_update += (zeta * local_Elog_beta).transpose();
        for (int j = 0; j < m; ++j) {
            double max_val = log_phi_update.row(j).maxCoeff();
            phi.row(j) = (log_phi_update.row(j).array() - max_val).exp();
            phi.row(j) /= phi.row(j).sum();
        }

        doc_topic = xvec.transpose() * phi * zeta;
        diff = (last_doc_topic - doc_topic).array().abs().mean();
    }

    if (verbose_ > 1) {
        notice("%s: finished after %d iterations, mean change %.1e", __FUNCTION__, iter, diff);
    }

    return doc_topic / doc_topic.sum();
}

void HDP::partial_fit(const std::vector<Document>& docs) {
    int minibatch_size = static_cast<int>(docs.size());

    // Thread-local accumulators for topic-word counts and stick statistics
    tbb::combinable<MatrixXd> ss_acc{[&]{ return MatrixXd::Zero(K_, M_); }};
    tbb::combinable<VectorXd> a_acc{[&]{ return VectorXd::Zero(K_); }};
    tbb::combinable<VectorXd> b_acc{[&]{ return VectorXd::Zero(K_); }};

    // Parallel processing of documents
    tbb::parallel_for(
        tbb::blocked_range<int>(0, minibatch_size),
        [&](const tbb::blocked_range<int>& range) {
            auto& local_ss = ss_acc.local();
            auto& local_a  = a_acc.local();
            auto& local_b  = b_acc.local();
            for (int d = range.begin(); d < range.end(); ++d) {
                const Document& doc = docs[d];
                MatrixXd zeta, phi;
                // Fit local variational parameters
                VectorXd doc_topic = fit_one_document(zeta, phi, doc);

                // Compute expected word-topic sufficient statistics
                int m = static_cast<int>(doc.ids.size());
                Eigen::Map<const VectorXd> xvec(doc.cnts.data(), m);
                MatrixXd pjk = phi * zeta; // (m x T) * (T x K) -> m x K
                for (int j = 0; j < m; ++j) {
                    int word_id = doc.ids[j];
                    local_ss.col(word_id) += xvec(j) * pjk.row(j).transpose();
                }

                // Accumulate global stick sufficient statistics
                VectorXd sum_zeta = zeta.colwise().sum(); // K x 1
                for (int k = 0; k < K_; ++k) {
                    local_a[k] += sum_zeta[k];
                    double tail = 0.0;
                    for (int j = k + 1; j < K_; ++j) tail += sum_zeta[j];
                    local_b[k] += tail;
                }
            }
        });

    // Combine thread-local accumulators
    MatrixXd ss = ss_acc.combine([](const MatrixXd& A, const MatrixXd& B){ return A + B; });
    VectorXd a_sum = a_acc.combine([](const VectorXd& A, const VectorXd& B){ return A + B; });
    VectorXd b_sum = b_acc.combine([](const VectorXd& A, const VectorXd& B){ return A + B; });

    // Compute learning rate
    ++update_count_;
    double rho = std::pow(learning_offset_ + update_count_, -learning_decay_);

    // Scale sufficient statistics for full corpus
    double scale = static_cast<double>(total_doc_count_) / minibatch_size;
    MatrixXd lambda_hat = MatrixXd::Constant(K_, M_, eta_) + scale * ss;
    VectorXd a_hat = VectorXd::Ones(K_) + scale * a_sum;
    VectorXd b_hat = VectorXd::Constant(K_, omega_) + scale * b_sum;

    // Update global parameters
    lambda_ = (1.0 - rho) * lambda_ + rho * lambda_hat;
    aK_     = (1.0 - rho) * aK_     + rho * a_hat;
    bK_     = (1.0 - rho) * bK_     + rho * b_hat;

    // Recompute expectations
    Elog_beta_    = dirichlet_entropy_2d(lambda_);
    Elog_sigma_K_ = expect_log_sticks(aK_, bK_);
}

MatrixXd HDP::transform(const std::vector<Document>& docs) {
    int n_docs = docs.size();
    MatrixXd doc_topic_distr(n_docs, K_);
    // Parallel update: process each document independently
    tbb::parallel_for(0, n_docs, [&](int d) {
        MatrixXd zeta, phi;
        doc_topic_distr.row(d) = fit_one_document(zeta, phi, docs[d]).transpose();
    });
    return doc_topic_distr;
}

void HDP::init() {
    set_nthreads(nThreads_);
    // Check parameters
    if (seed_ <= 0) {
        seed_ = std::random_device{}();
    }
    random_engine_.seed(seed_);
    eps_ = std::numeric_limits<double>::epsilon();
    if (eta_ < 0) {
        eta_ = 0.01;
    }
    if (alpha_ < 0) {
        alpha_ = 1.0;
    }
    if (omega_ < 0) {
        omega_ = 1.0;
    }
    if (total_doc_count_ <= 0) {
        total_doc_count_ = 1000000;
    }
    if (learning_decay_ <= 0) {
        learning_decay_ = 0.9; // kappa
    }
    if (learning_offset_ < 0) {
        learning_offset_ = 10.0; // tau
    }
    if (max_doc_update_iter_ <= 0) {
        max_doc_update_iter_ = 100;
    }
    if (mean_change_tol_ <= 0) {
        mean_change_tol_ = 0.001 / T_;
    }
    // Randomly initialize topic-word counts lambda_
    std::gamma_distribution<double> gd(100.0, 0.01);
    lambda_ = RowMajorMatrixXd(K_, M_);
    for (int k = 0; k < K_; ++k) {
        for (int m = 0; m < M_; ++m) {
            lambda_(k, m) = gd(random_engine_);
        }
    }
    // Initialize stick-breaking parameters
    aK_ = VectorXd::Ones(K_);
    bK_ = VectorXd::Constant(K_, omega_);
    // Precompute expectations
    Elog_beta_    = dirichlet_entropy_2d(lambda_);
    Elog_sigma_K_ = expect_log_sticks(aK_, bK_);

    if (verbose_) {
        notice("HDP initialized with K=%d, T=%d, %d features, %d threads", K_, T_, M_, nThreads_);
    }
}

std::vector<int32_t> HDP::sort_topics() {
    // 1. Calculate raw topic weights (sum of lambda rows)
    VectorXd topic_weights = lambda_.rowwise().sum();

    // 2. Create a vector of topic sorted_indices_ to sort
    sorted_indices_.resize(K_);
    std::iota(sorted_indices_.begin(), sorted_indices_.end(), 0);

    // 3. Sort sorted_indices_ based on descending topic weights
    std::sort(sorted_indices_.begin(), sorted_indices_.end(),
                [&](int32_t a, int32_t b) {
                    return topic_weights(a) > topic_weights(b);
                });
    return sorted_indices_;

    // // 4. Create new sorted matrices and vectors
    // MatrixXd sorted_lambda(K_, M_), sorted_Elog_beta(K_, M_);
    // VectorXd sorted_aK(K_), sorted_bK(K_), sorted_Elog_sigma_K(K_);

    // for (int32_t i = 0; i < K_; ++i) {
    //     int32_t old_idx = sorted_indices_[i];
    //     sorted_lambda.row(i) = lambda_.row(old_idx);
    //     sorted_Elog_beta.row(i) = Elog_beta_.row(old_idx);
    //     sorted_aK(i) = aK_(old_idx);
    //     sorted_bK(i) = bK_(old_idx);
    //     sorted_Elog_sigma_K(i) = Elog_sigma_K_(old_idx);
    // }

    // // 5. Replace the old parameters with the new sorted ones
    // lambda_ = std::move(sorted_lambda);
    // Elog_beta_ = std::move(sorted_Elog_beta);
    // aK_ = std::move(sorted_aK);
    // bK_ = std::move(sorted_bK);
    // Elog_sigma_K_ = std::move(sorted_Elog_sigma_K);
}
