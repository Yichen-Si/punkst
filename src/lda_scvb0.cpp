#include "lda.hpp"

void LatentDirichletAllocation::scvb0_partial_fit(
        const std::vector<Document>& docs)
{
    const int n_docs = docs.size();
    tbb::combinable<MatrixXd> hatNkw_acc {
        [&] { return MatrixXd::Zero(n_topics_, n_features_); }
    };

    tbb::parallel_for(
        tbb::blocked_range<int>(0, n_docs),
        [&](const tbb::blocked_range<int>& range) {
            auto& localNkw = hatNkw_acc.local();

            for (int d = range.begin(); d < range.end(); ++d) {
                const Document& doc = docs[d];
                scvb0_fit_one_document(localNkw, doc);
            }
        });
    // Combine
    MatrixXd hatNkw = hatNkw_acc.combine(
        [](const MatrixXd& A, const MatrixXd& B) {
            return A + B;
        }
    );
    VectorXd hatNk = hatNkw.rowwise().sum();

    // Update global statistics (Eq. 7 & 8)
    double rho = s_beta_ * std::pow(learning_offset_ + ++update_count_, -learning_decay_);
    double scale = static_cast<double>(total_doc_count_) / n_docs;
    components_ = (1.0 - rho) * components_ + rho * scale * hatNkw;
    Nk_         = (1.0 - rho) * Nk_         + rho * scale * hatNk;
}

void LatentDirichletAllocation::scvb0_fit_one_document(
        MatrixXd& hatNkw, const Document& doc)
{
    int Cj = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0);
    VectorXd NTheta = VectorXd::Constant(n_topics_, doc_topic_prior_);
    VectorXd phi(n_topics_);
    int t = 0;
    // generate a random order of words to process
    std::vector<int> word_order(doc.ids.size());
    std::iota(word_order.begin(), word_order.end(), 0);
    std::shuffle(word_order.begin(), word_order.end(), random_engine_);
    for (int epoch = 0; epoch < burn_in_ + 1; ++epoch) {
        for (size_t i = 0; i < doc.ids.size(); ++i) {
            const uint32_t w = doc.ids[word_order[i]];
            const int      m = doc.cnts[word_order[i]];
            scvb0_one_word(w, NTheta, phi);

            double rho_theta = s_theta_ * std::pow(tau_theta_ + ++t, -kappa_theta_);
            const double decay = std::pow(1.0 - rho_theta, m);
            NTheta = decay * NTheta + (1.0 - decay) * (Cj * phi); // Eq. 9

            if (epoch == burn_in_) { // Update sufficient statistics
                hatNkw.col(w) += static_cast<double>(m) * phi;
            }
        }
    }
}

// for transform (don't need sufficient statistics for global update)
void LatentDirichletAllocation::scvb0_fit_one_document(
        VectorXd& hatNk, const Document& doc)
{
    hatNk.resize(n_topics_);
    hatNk.setZero();
    int Cj = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0);
    VectorXd NTheta = VectorXd::Constant(n_topics_, doc_topic_prior_);
    VectorXd phi(n_topics_);
    int t = 0;
    // generate a random order of words to process
    std::vector<int> word_order(doc.ids.size());
    std::iota(word_order.begin(), word_order.end(), 0);
    std::shuffle(word_order.begin(), word_order.end(), random_engine_);
    for (int epoch = 0; epoch < burn_in_ + 1; ++epoch) {
        for (size_t i = 0; i < doc.ids.size(); ++i) {
            const uint32_t w = doc.ids[word_order[i]];
            const int      m = doc.cnts[word_order[i]];
            scvb0_one_word(w, NTheta, phi);

            double rho_theta = s_theta_ * std::pow(tau_theta_ + ++t, -kappa_theta_);
            const double decay = std::pow(1.0 - rho_theta, m);
            NTheta = decay * NTheta + (1.0 - decay) * (Cj * phi); // Eq. 9

            if (epoch == burn_in_) {
                hatNk += static_cast<double>(m) * phi;
            }
        }
    }
}

void LatentDirichletAllocation::set_scvb0_parameters(double s_beta, double s_theta, double tau_theta, double kappa_theta, int32_t burn_in) {
    s_beta_ = s_beta;
    s_theta_ = s_theta;
    tau_theta_ = tau_theta;
    kappa_theta_ = kappa_theta;
    if (s_beta_ > std::pow(learning_offset_ + 1, learning_decay_)) {
        s_beta_ = 1;
    }
    if (s_theta_ > std::pow(tau_theta + 1, kappa_theta)) {
        s_theta_ = 1;
    }
    burn_in_ = burn_in;
}
