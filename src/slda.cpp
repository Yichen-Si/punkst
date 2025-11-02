#include "pixdecode.hpp"

RowMajorMatrixXf OnlineSLDA::do_e_step(Minibatch& batch, bool return_ss) {
    // (N x M) * (M x K) = (N x K)
    RowMajorMatrixXf Xb = batch.mtx * Elog_beta_.transpose();
    // If gamma is not set, randomly initialize
    if (batch.gamma.size() == 0) {
        batch.gamma.resize(batch.n, K_);
        std::gamma_distribution<float> gamma_dist(100.0, 1.0/100.0);
        for (int i = 0; i < batch.n; i++) {
            for (int j = 0; j < K_; j++) {
                batch.gamma(i, j) = gamma_dist(rng_);
            }
        }
    }

    RowMajorMatrixXf gamma_old = batch.gamma;
    RowMajorMatrixXf phi_old = batch.phi;
    if (batch.psi.size() == 0) {
        batch.psi = batch.wij;
        expitAndRowNormalize(batch.psi);
    }
    if (verbose_ > 1) {
        approx_ll(batch);
        printf("Initialize E-step, E_q[ll] %.4e\n", batch.ll);
    }
    RowMajorMatrixXf Elog_theta; // (n x K)
    double meanchange = tol_ + 1;
    double meanchange_phi = tol_ + 1;
    int it = 0;
    while (it < max_iter_inner_ && meanchange > tol_) {
        it++;
        Elog_theta = dirichlet_entropy_2d(batch.gamma);
        // Update phi.
        // psi: (N x n) * Elog_theta (n x K) → phi: (N x K)
        batch.phi = batch.psi * Elog_theta + Xb;
        // exponentiate and row-normalize
        rowwiseSoftmax(batch.phi);
        // Update psi.
        // (psi has the same sparsity pattern as wij)
        VectorXf extra = (batch.phi.array() * Xb.array()).rowwise().sum();
        for (int i = 0; i < batch.psi.rows(); ++i) { // could parallelize
            // Iterate over the nonzero entries
            for (SparseMatrix<float, Eigen::RowMajor>::InnerIterator it1(batch.psi, i), it2(batch.wij, i); it1 && it2; ++it1, ++it2) {
                int j = it1.col();
                // dot product or i-th row of phi with j-th column of Elog_theta
                float sum_val = batch.phi.row(i).dot(Elog_theta.row(j));
                // Updated value at (i,j)
                batch.psi.coeffRef(i, j) = it2.value() + sum_val + extra(i);
            }
        }
        expitAndRowNormalize(batch.psi);
        // Update gamma: gamma = alpha + psi^T * phi.
        // psi^T: (n x N), phi: (N x K) → gamma: (n x K)
        // broadcast alpha_
        batch.gamma = batch.psi.transpose() * batch.phi;
        batch.gamma += alpha_.replicate(batch.n, 1);
        // Check convergence
        meanchange = mean_max_row_change(batch.gamma, gamma_old);
        gamma_old = batch.gamma;
        meanchange_phi = mean_max_row_change(batch.phi, phi_old);
        phi_old = batch.phi;
        if (verbose_ > 2 || (verbose_ > 1 && it % 10 == 0)) {
            printf("E-step, iteration %d, mean change in gamma: %.4e; mean change in phi: %.4e\n", it, meanchange, meanchange_phi);
        }
    }
    if (verbose_ > 1) {
        approx_ll(batch);
        printf("E-step finished in %d iterations. E_q[ll] %.4e\n", it, batch.ll);
    }

    if (return_ss) {
        // Compute sufficient statistics: sstats = phi^T * mtx → (K x M)
        RowMajorMatrixXf sstats = batch.phi.transpose() * batch.mtx;
        return sstats;
    } else {
        // Return an empty matrix
        return RowMajorMatrixXf::Zero(0, 0);
    }
}

void OnlineSLDA::init(int K, int M, int N,
    unsigned int seed,
    VectorXf* alpha, VectorXf* eta,
    double tau0, double kappa,
    int iter_inner, double tol,
    int verbose, int debug) {
    K_ = K;
    M_ = M;
    N_ = N;
    rng_.seed(seed);
    tau0_ = std::max(tau0 + 1., 1.0);
    kappa_ = kappa;
    updatect_ = 0;
    max_iter_inner_ = iter_inner;
    tol_ = tol;
    verbose_ = verbose;
    debug_ = debug;
    // Global alpha: expect a K-dimensional vector (K x 1)
    if (alpha && alpha->size() == K_) {
        alpha_ = (*alpha).transpose();
    } else {
        alpha_ = RowVectorXf::Constant(K_, 1./K_);
    }
    // Global eta: stored as a row vector (1 x M)
    if (eta && eta->size() == M_) {
        eta_ = *eta;
        eta_ = eta_.transpose();
    } else {
        eta_ = VectorXf::Constant(M_, 1./K_).transpose();
    }
}

void OnlineSLDA::init_global_parameter(const RowMajorMatrixXf* m_lambda) {
    if (m_lambda == nullptr) {
        lambda_.resize(K_, M_);
        std::gamma_distribution<float> gamma_dist(100.0, 1.0/100.0);
        for (int i = 0; i < K_; i++) {
            for (int j = 0; j < M_; j++) {
                lambda_(i, j) = gamma_dist(rng_);
            }
        }
        Elog_beta_ = dirichlet_entropy_2d(lambda_); // (K x M)
    } else {
        init_global_parameter(*m_lambda);
    }
}

void OnlineSLDA::approx_ll(Minibatch& batch) {
    // Compute the log-likelihood for a minibatch.
    RowMajorMatrixXf Xb = batch.mtx * Elog_beta_.transpose(); // (N x K)
    RowMajorMatrixXf ll_mat = batch.psi.transpose() * Xb; // (n x K)
    // log-sum-exp
    VectorXf maxCoeffs = ll_mat.rowwise().maxCoeff();
    ll_mat.colwise() -= maxCoeffs;
    VectorXf lse = ll_mat.array().exp().rowwise().sum().log() + maxCoeffs.array();
    batch.ll = (ll_mat.colwise() - lse).sum() / batch.n;
}

// Update the global lambda parameter with one minibatch.
void OnlineSLDA::update_lambda(Minibatch& batch) {
    RowMajorMatrixXf sstats = do_e_step(batch);
    if (verbose_ > 0) {
        auto scores = approx_score(batch);
        printf("%d-th global update. Scores: %.4e, %.4e, %.4e, %.4e, %.4e\n", updatect_, std::get<0>(scores), std::get<1>(scores), std::get<2>(scores), std::get<3>(scores), std::get<4>(scores));

    }
    double rhot = std::pow(tau0_ + updatect_, -kappa_);
    // Update rule: λ = (1 - rhot) * λ + rhot * ((N / batch.N) * (η + sstats))
    double scale = static_cast<double>(N_) / batch.N;
    // We assume η is stored as a row vector; replicate it to match λ’s dimensions.
    lambda_ = (1 - rhot) * lambda_ + rhot * ((eta_.replicate(K_, 1) + sstats) * scale);
    Elog_beta_ = dirichlet_entropy_2d(lambda_);
    updatect_++;
}

// Compute approximate scores for monitoring progress.
std::tuple<double, double, double, double, double> OnlineSLDA::approx_score(Minibatch& batch) {
    // Score for pixels: sum( φ .* (mtx * Elog_beta_.transpose())/batch.N )
    RowMajorMatrixXf X = batch.mtx * Elog_beta_.transpose();
    double score_pixel = (batch.phi.array() * (X.array() / batch.N)).sum();
    double score_patch = batch.ll;

    // Score for gamma: E[log p(θ | α) - log q(θ | γ)]
    RowMajorMatrixXf Elog_theta = dirichlet_entropy_2d(batch.gamma);
    double score_gamma = ((alpha_.replicate(batch.n, 1) - batch.gamma).array() * Elog_theta.array()).sum();
    for (int i = 0; i < batch.gamma.rows(); i++) {
        double row_sum = batch.gamma.row(i).sum();
        for (int k = 0; k < batch.gamma.cols(); k++) {
            score_gamma += std::lgamma(batch.gamma(i, k));
        }
        score_gamma -= std::lgamma(row_sum);
    }
    score_gamma /= batch.n;

    // Score for beta: E[log p(β | η) - log q(β | λ)]
    double score_beta = ((eta_.replicate(K_,1) - lambda_).array() * Elog_beta_.array()).sum();
    for (int i = 0; i < lambda_.rows(); i++) {
        double row_sum = lambda_.row(i).sum();
        for (int j = 0; j < lambda_.cols(); j++) {
            score_beta += std::lgamma(lambda_(i, j));
        }
        score_beta -= std::lgamma(row_sum);
    }
    double total_score = N_*1./batch.N * batch.n * (score_patch + score_gamma) + score_beta;
    return std::make_tuple(score_pixel, score_patch, score_gamma, score_beta, total_score);
}
