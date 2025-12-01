#include <iostream>
#include <vector>
#include <random>
#include <numeric>
#include <algorithm>
#include <cmath>

#include "commands.hpp"
#include "poisnmf.hpp"

using Eigen::VectorXd;
using RowMajorMatrixXd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct SimData {
    std::vector<SparseObs> docs;
    std::vector<VectorXd> topics; // K topic distributions over M features
    VectorXd background;          // length M
    double avg_pi = 0.0;
    double size_factor = 0.0;
    std::vector<double> pi;
};

static VectorXd normalize(const VectorXd& v) {
    double s = v.sum();
    if (s <= 0 || !std::isfinite(s)) return v;
    return v / s;
}

// For feature-major beta (M x K, column per topic)
static std::vector<VectorXd> feature_major_topics(const RowMajorMatrixXd& beta) {
    const int K = static_cast<int>(beta.cols());
    std::vector<VectorXd> dists;
    dists.reserve(K);
    for (int k = 0; k < K; ++k) {
        VectorXd col = beta.col(k);
        for (int i = 0; i < col.size(); ++i) {
            if (!std::isfinite(col[i])) col[i] = 0.0;
        }
        VectorXd w = (col.array().min(20.0).max(-20.0).exp() - 1.0).cwiseMax(1e-12).matrix();
        double s = w.sum();
        if (s <= 0 || !std::isfinite(s)) {
            w.setConstant(1.0 / static_cast<double>(beta.rows()));
        } else {
            w /= s;
        }
        dists.push_back(w);
    }
    return dists;
}

// For factor-major beta (K x M, row per topic) used in simulation
static std::vector<VectorXd> factor_major_topics(const RowMajorMatrixXd& beta) {
    const int K = static_cast<int>(beta.rows());
    std::vector<VectorXd> dists;
    dists.reserve(K);
    for (int k = 0; k < K; ++k) {
        VectorXd row = beta.row(k).transpose();
        for (int i = 0; i < row.size(); ++i) {
            if (!std::isfinite(row[i])) row[i] = 0.0;
        }
        VectorXd w = (row.array().min(20.0).max(-20.0).exp() - 1.0).cwiseMax(1e-12).matrix();
        double s = w.sum();
        if (s <= 0 || !std::isfinite(s)) {
            w.setConstant(1.0 / static_cast<double>(beta.cols()));
        } else {
            w /= s;
        }
        dists.push_back(w);
    }
    return dists;
}

static double cosine(const VectorXd& a, const VectorXd& b) {
    if (a.size() != b.size()) return 0.0;
    if (!a.allFinite() || !b.allFinite()) return 0.0;
    const double denom = std::sqrt(std::max(1e-12, a.squaredNorm() * b.squaredNorm()));
    if (denom <= 0) return 0.0;
    return a.dot(b) / denom;
}

static double best_alignment_score(const std::vector<VectorXd>& est,
                                   const std::vector<VectorXd>& truth) {
    const int K = static_cast<int>(std::min(est.size(), truth.size()));
    if (K == 0) return 0.0;
    std::vector<int> perm(K);
    std::iota(perm.begin(), perm.end(), 0);
    double best = -1.0;
    do {
        double total = 0.0;
        for (int k = 0; k < K; ++k) {
            total += cosine(est[perm[k]], truth[k]);
        }
        best = std::max(best, total / K);
    } while (std::next_permutation(perm.begin(), perm.end()));
    return best;
}

static SimData simulate_documents(std::mt19937& rng, int N, int K, int M) {
    SimData sim;
    sim.docs.reserve(N);
    sim.topics.resize(K);
    sim.pi.resize(N);

    // Theta: each row has one dominant entry = 0.9, others share 0.1.
    RowMajorMatrixXd theta(N, K);
    for (int i = 0; i < N; ++i) {
        int main_k = i % K;
        for (int k = 0; k < K; ++k) {
            theta(i, k) = (k == main_k) ? 0.9 : 0.1 / std::max(1, K - 1);
        }
    }

    // Beta: random positive, converted via log1p for the log-link.
    RowMajorMatrixXd beta(K, M);
    std::gamma_distribution<double> g_beta(10, 1);
    for (int k = 0; k < K; ++k) {
        for (int m = 0; m < M; ++m) {
            double w = g_beta(rng);
            beta(k, m) = std::log1p(w);
        }
    }
    sim.topics = factor_major_topics(beta);

    // Lambda = exp(theta * beta) - 1.
    RowMajorMatrixXd eta = theta * beta;
    RowMajorMatrixXd lambda = (eta.array().exp() - 1.0).matrix();

    RowMajorMatrixXd Y = RowMajorMatrixXd::Zero(N, M);
    std::vector<double> row_sums(N, 0.0);

    // Sample base counts.
    for (int i = 0; i < N; ++i) {
        for (int m = 0; m < M; ++m) {
            double lam = std::max(0.0, lambda(i, m));
            if (lam <= 0) continue;
            std::poisson_distribution<int> pois(lam);
            int c = pois(rng);
            if (c > 0) {
                Y(i, m) = static_cast<double>(c);
                row_sums[i] += c;
            }
        }
    }

    // Background/noise distribution from column sums.
    VectorXd col_sums = Y.colwise().sum();
    double total_col = col_sums.sum();
    if (total_col <= 0) {
        col_sums.setConstant(1.0 / static_cast<double>(M));
        total_col = col_sums.sum();
    }
    sim.background = col_sums / total_col;

    // Add 5%~30% noise per unit drawn from the noise distribution.
    std::discrete_distribution<int> noise_dist(sim.background.data(),
        sim.background.data() + sim.background.size());
    sim.avg_pi = 0.0;
    for (int i = 0; i < N; ++i) {
        double noise_frac = 0.05 + 0.25 * (static_cast<double>(rng()) / rng.max());
        double base_total = row_sums[i];
        int noise_tokens = (base_total > 0) ? static_cast<int>(std::round(noise_frac * base_total)) : 1;
        for (int t = 0; t < noise_tokens; ++t) {
            int m = noise_dist(rng);
            Y(i, m) += 1.0;
        }
        double new_total = base_total + noise_tokens;
        row_sums[i] = new_total;
        double f0 = (new_total > 0) ? (static_cast<double>(noise_tokens) / new_total) : 0.0;
        sim.avg_pi += f0;
        sim.pi[i] = f0;
    }
    sim.avg_pi /= static_cast<double>(N);

    // size_factor_ = average row sum of final Y.
    double total_tokens = std::accumulate(row_sums.begin(), row_sums.end(), 0.0);
    sim.size_factor = (N > 0) ? total_tokens / static_cast<double>(N) : 1.0;
    if (sim.size_factor <= 0) sim.size_factor = 1.0;

    // Build SparseObs.
    for (int i = 0; i < N; ++i) {
        Document doc;
        for (int m = 0; m < M; ++m) {
            double c = Y(i, m);
            if (c > 0) {
                doc.ids.push_back(static_cast<uint32_t>(m));
                doc.cnts.push_back(c);
            }
        }
        if (doc.ids.empty()) {
            doc.ids.push_back(0);
            doc.cnts.push_back(1.0);
        }
        double ct_tot = std::accumulate(doc.cnts.begin(), doc.cnts.end(), 0.0);
        doc.ct_tot = ct_tot;

        SparseObs obs;
        obs.doc = std::move(doc);
        obs.covar = VectorXd();
        obs.ct_tot = ct_tot;
        obs.c = ct_tot / sim.size_factor;
        sim.docs.push_back(std::move(obs));
    }

    return sim;
}

int32_t test(int32_t argc, char** argv) {

    int N = 24, K = 2, M = 6;
    int threads = 1;
    int seed = 42;

    MLEOptions mle_opts;
    mle_opts.optim.max_iters = 50;
    mle_opts.optim.tol = 1e-6;
    NmfFitOptions nmf_opts;
    nmf_opts.n_mb_epoch = 5;
    nmf_opts.max_iter = 10;
    nmf_opts.tol = 1e-3;
    nmf_opts.batch_size = 512;

    ParamList pl;
    pl.add_option("N", "Number of rows (N)", N, true)
      .add_option("K", "Number of cols (K)", K, true)
      .add_option("M", "Number of features (M)", M, true)
      .add_option("seed", "Random seed", seed)
      .add_option("threads", "Number of threads", threads);
    pl.add_option("minibatch-epoch", "Number of minibatch epochs (background-aware path)", nmf_opts.n_mb_epoch)
      .add_option("minibatch-size", "Minibatch size", nmf_opts.batch_size)
      .add_option("t0", "Decay parameter t0 for minibatch", nmf_opts.t0)
      .add_option("kappa", "Decay parameter kappa for minibatch", nmf_opts.kappa)
      .add_option("max-iter-outer", "Maximum outer iterations", nmf_opts.max_iter);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    std::mt19937 rng(seed);
    SimData sim = simulate_documents(rng, N, K, M);

    PoissonLog1pNMF nmf_bg(K, M, threads, sim.size_factor, seed, true, 0);
    nmf_bg.set_background_model(sim.avg_pi, &sim.background);
    nmf_bg.fit(sim.docs, mle_opts, nmf_opts, true);

    PoissonLog1pNMF nmf_base(K, M, threads, sim.size_factor, seed + 7, true, 0);
    nmf_base.fit(sim.docs, mle_opts, nmf_opts, true);

    auto est_bg = feature_major_topics(nmf_bg.get_model());
    auto est_base = feature_major_topics(nmf_base.get_model());

    double sim_bg = best_alignment_score(est_bg, sim.topics);
    double sim_base = best_alignment_score(est_base, sim.topics);

    std::vector<int> bg_features(M);
    std::iota(bg_features.begin(), bg_features.end(), 0);
    int m0 = 10;
    std::partial_sort(bg_features.begin(), bg_features.begin() + std::min(m0, M),
        bg_features.end(), [&](int a, int b) { return sim.background[a] > sim.background[b]; });
    bg_features.resize(std::min(m0, M));

    auto bg_mass = [&](const std::vector<VectorXd>& topics) {
        double acc = 0.0;
        for (const auto& v : topics) {
            double m = 0.0;
            for (int idx : bg_features) m += v[idx];
            acc += m;
        }
        return acc / topics.size();
    };
    double mass_bg = bg_mass(est_bg);
    double mass_base = bg_mass(est_base);

    const auto& beta0 = nmf_bg.get_bg_model();
    const auto& est_pi = nmf_bg.get_bg_proportions();
    double bg_cosine = cosine(beta0, sim.background);
    double pi_cosine = cosine(Eigen::Map<const VectorXd>(sim.pi.data(), sim.pi.size()), Eigen::Map<const VectorXd>(est_pi.data(), est_pi.size()));
    double avg_pi = std::accumulate(est_pi.begin(), est_pi.end(), 0.0) / static_cast<double>(est_pi.size());

    std::cout << "Background model cosine similarity: " << bg_cosine << "\n";
    std::cout << "Background proportions cosine similarity: " << pi_cosine << "\n";
    std::cout << "Average estimated pi: " << avg_pi << "\n";

    std::cout << "Similarity (with background): " << sim_bg << "\n";
    std::cout << "Similarity (no background)  : " << sim_base << "\n";
    std::cout << "Background feature mass (with background): " << mass_bg << "\n";
    std::cout << "Background feature mass (no background)  : " << mass_base << "\n";
    std::cout << "Average pi used for background: " << sim.avg_pi << "\n";
    std::cout << "Size factor (avg row sum): " << sim.size_factor << "\n";
    return 0;
}
