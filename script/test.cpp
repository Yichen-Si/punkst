#include <iostream>
#include <iomanip>
#include <random>
#include <numeric>
#include <chrono>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <vector>
#include <string>

#include "punkst.h"
#include "utils.h"
#include <tbb/parallel_for.h>
#include <tbb/global_control.h>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "lda.hpp"

using Clock = std::chrono::high_resolution_clock;

/**
 * Simulate a corpus with:
 *  - K topics
 *  - M vocabulary size
 *  - N documents of ~avg_doc_len words
 */
void simulate_corpus(int N, int M, int K,
                     int avg_doc_len,
                     int max_topic_per_doc,
                     double eta_beta,
                     double a0_true, double b0_true,
                     std::mt19937& rng,
                     RowMajorMatrixXd& true_topics,
                     VectorXd& true_bg,
                     std::vector<Document>& docs,
                     VectorXd& global_counts_out)
{
    true_topics.resize(K, M);
    true_topics.setZero();
    true_bg.resize(M);

    std::gamma_distribution<double> gamma_topic(eta_beta, 1.0);
    int nz = M / K;
    for (int k = 0; k < K; ++k) {
        double row_sum = 0.0;
        int st = k * nz;
        int ed = (k == K - 1) ? M : (k + 1) * nz;
        for (int v = st; v < ed; ++v) {
            double x = gamma_topic(rng);
            true_topics(k, v) = x;
            row_sum += x;
        }
        if (row_sum > 0) {
            true_topics.row(k).segment(st, ed - st) /= row_sum;
        }
    }

    // Background distribution ~ Dir(1)
    std::gamma_distribution<double> gamma_bg(1.0, 1.0);
    double bg_sum = 0.0;
    for (int v = 0; v < M; ++v) {
        double x = gamma_bg(rng);
        true_bg[v] = x;
        bg_sum += x;
    }
    true_bg /= bg_sum;

    // Prebuild discrete distributions for topics and background
    std::vector<std::discrete_distribution<int>> topic_word_dists;
    topic_word_dists.reserve(K);
    for (int k = 0; k < K; ++k) {
        std::vector<double> probs(M);
        for (int v = 0; v < M; ++v) {
            probs[v] = true_topics(k, v);
        }
        topic_word_dists.emplace_back(probs.begin(), probs.end());
    }
    std::vector<double> bg_probs(M);
    for (int v = 0; v < M; ++v) bg_probs[v] = true_bg[v];
    std::discrete_distribution<int> bg_word_dist(bg_probs.begin(), bg_probs.end());

    std::uniform_int_distribution<int> topic_selector(0, K - 1);

    // Poisson for document lengths
    std::poisson_distribution<int> doc_len(avg_doc_len);

    // Beta(a0_true, b0_true) via Gammas
    std::gamma_distribution<double> gamma_pi_a(a0_true, 1.0);
    std::gamma_distribution<double> gamma_pi_b(b0_true, 1.0);
    std::uniform_real_distribution<double> uni01(0.0, 1.0);

    docs.clear();
    docs.reserve(N);

    global_counts_out = VectorXd::Zero(M);
    double pi_sum = 0.0;

    for (int d = 0; d < N; ++d) {
        int L = std::max(10, doc_len(rng));

        // Sparse doc-topic mixture: at most max_topic_per_doc active topics
        std::vector<double> theta_vec(K, 0.0);
        for (int i = 0; i < max_topic_per_doc; i++) {
            int tk = topic_selector(rng);
            theta_vec[tk] += 1.0 / max_topic_per_doc;
        }
        std::discrete_distribution<int> topic_dist(theta_vec.begin(), theta_vec.end());

        // pi_d ~ Beta(a0_true, b0_true)
        double ga = gamma_pi_a(rng);
        double gb = gamma_pi_b(rng);
        double pi_d = ga / (ga + gb);
        pi_sum += pi_d;

        std::vector<int> word_counts(M, 0);

        for (int n = 0; n < L; ++n) {
            bool from_bg = (uni01(rng) < pi_d);
            int w;
            if (from_bg) {
                w = bg_word_dist(rng);
            } else {
                int k = topic_dist(rng);
                w = topic_word_dists[k](rng);
            }
            ++word_counts[w];
        }

        Document doc;
        for (int v = 0; v < M; ++v) {
            if (word_counts[v] > 0) {
                doc.ids.push_back(v);
                doc.cnts.push_back(static_cast<double>(word_counts[v]));
                global_counts_out[v] += word_counts[v];
            }
        }
        docs.push_back(std::move(doc));
    }
    pi_sum /= N;
    std::cout << "Simulated corpus with " << N << " documents, avg pi = " << pi_sum << "\n";
}

// Hungarian algorithm for square cost matrix (minimization).
// cost is 1-based indexed: cost[1..n][1..n]
static std::vector<int> hungarian_min_cost(
    const std::vector<std::vector<double>>& cost)
{
    int n = static_cast<int>(cost.size()) - 1; // assuming cost[0] is dummy
    const double INF = std::numeric_limits<double>::infinity();

    std::vector<double> u(n + 1), v(n + 1);
    std::vector<int> p(n + 1), way(n + 1);

    for (int i = 1; i <= n; ++i) {
        p[0] = i;
        int j0 = 0;
        std::vector<double> minv(n + 1, INF);
        std::vector<char> used(n + 1, false);
        do {
            used[j0] = true;
            int i0 = p[j0], j1 = 0;
            double delta = INF;
            for (int j = 1; j <= n; ++j) {
                if (!used[j]) {
                    double cur = cost[i0][j] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (int j = 0; j <= n; ++j) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] != 0);
        do {
            int j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0);
    }

    // p[j] = i  means row i is assigned to column j
    std::vector<int> assignment(n, -1); // assignment[row] = col
    for (int j = 1; j <= n; ++j) {
        int i = p[j];
        if (i > 0) {
            assignment[i - 1] = j - 1;
        }
    }
    return assignment;
}

// Compute best permutation of topics (rows) to align true_topics with model_topics.
std::vector<int> best_topic_permutation(
    const RowMajorMatrixXd& true_topics,
    const RowMajorMatrixXd& model_topics)
{
    assert(true_topics.rows() == model_topics.rows());
    int K = static_cast<int>(true_topics.rows());
    int M = static_cast<int>(true_topics.cols());
    assert(model_topics.cols() == M);

    // Make normalized copies: rows sum to 1
    RowMajorMatrixXd Tnorm = true_topics;
    RowMajorMatrixXd Mnorm = model_topics;

    for (int k = 0; k < K; ++k) {
        double sT = Tnorm.row(k).sum();
        if (sT <= 0) sT = 1.0;
        Tnorm.row(k) /= sT;

        double sM = Mnorm.row(k).sum();
        if (sM <= 0) sM = 1.0;
        Mnorm.row(k) /= sM;
    }

    // Build 1-based cost matrix: cost[i][j] = 1 - dot(T_i, M_j)
    std::vector<std::vector<double>> cost(K + 1,
                                          std::vector<double>(K + 1, 0.0));
    for (int i = 1; i <= K; ++i) {
        for (int j = 1; j <= K; ++j) {
            double sim = Tnorm.row(i - 1).dot(Mnorm.row(j - 1));
            cost[i][j] = 1.0 - sim;  // minimize cost => maximize similarity
        }
    }

    return hungarian_min_cost(cost);
}

template <typename Derived>
std::vector<int> top_k_indices(const Eigen::MatrixBase<Derived>& v, int k) {
    std::vector<int> idx(v.size());
    std::iota(idx.begin(), idx.end(), 0);
    k = std::min<int>(k, static_cast<int>(idx.size()));
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
        [&](int i, int j) { return v[i] > v[j]; });
    idx.resize(k);
    return idx;
}

int32_t test(int32_t argc, char** argv) {

    int32_t N, K, M;
    int32_t seed, debug_ = 0, verbose = 0;
    int32_t threads = 1;
    int32_t avg_len = 100;
    double a0_true = 2.0, b0_true = 8.0;

    ParamList pl;
    // Input / sim options
    pl.add_option("N", "Number of rows (N)", N, true)
        .add_option("K", "Number of cols (K)", K, true)
        .add_option("M", "Number of features (M)", M, true)
        .add_option("seed", "Random seed", seed, true)
        .add_option("avg_len", "Average document length", avg_len)
        .add_option("a0", "True background prior a0", a0_true)
        .add_option("b0", "True background prior b0", b0_true)
        .add_option("threads", "Number of threads", threads);
    pl.add_option("debug", "Debug", debug_)
        .add_option("verbose", "Verbose output", verbose);

    try {
        pl.readArgs(argc, argv);
        pl.print_options();
    } catch (const std::exception &ex) {
        std::cerr << "Error parsing options: " << ex.what() << "\n";
        pl.print_help();
        return 1;
    }

    RowMajorMatrixXd true_topics;
    VectorXd true_bg;
    std::vector<Document> docs;
    VectorXd global_counts;
    std::mt19937 rng(seed);
    double eta_beta = 10;
    std::vector<int> perm;
    double avg_sim = 0.0;
    double max_sim = 0.0;

    // --- 1) Simulate corpus ---
    simulate_corpus(N, M, K, avg_len,
                    2, eta_beta,
                    a0_true, b0_true,
                    rng, true_topics, true_bg,
                    docs, global_counts);

    double total_tokens = global_counts.sum();
    VectorXd global_probs = global_counts / total_tokens;

    std::cout << "Simulated " << N << " documents, "
              << total_tokens << " total tokens.\n";
    std::shuffle(docs.begin(), docs.end(), rng);

    // --- 2) Instantiate LDA ---
    int minibatch_size = 256;
    LatentDirichletAllocation lda(
        K, M, seed, threads, verbose,
        InferenceType::SVB,
        -1, -1, -1, -1, N);
    lda.set_svb_parameters(/*max_iter=*/100, /*tol=*/1e-3);

    std::cout << "Fitting SVB...\n";
    for (int start = 0; start < N/3; start += minibatch_size) {
        int end = std::min(start + minibatch_size, N);
        std::vector<Document> minibatch;
        minibatch.reserve(end - start);
        for (int d = start; d < end; ++d) {
            minibatch.push_back(docs[d]);
        }
        lda.partial_fit(minibatch);
    }

    RowMajorMatrixXd model0 = lda.copy_model();
    perm = best_topic_permutation(true_topics, model0);
    for (int k_true = 0; k_true < true_topics.rows(); ++k_true) {
        int k_model = perm[k_true];
        Eigen::VectorXd t = true_topics.row(k_true).transpose();
        Eigen::VectorXd m = model0.row(k_model).transpose();
        double st = t.sum();
        double sm = m.sum();
        if (st > 0) t /= st;
        if (sm > 0) m /= sm;
        double sim = t.dot(m);  // in [0,1]
        double sim0 = t.dot(t);
        avg_sim += sim;
        max_sim += sim0;
        sim /= sim0;
        std::cout << "Topic " << k_true
                << " matched to model topic " << k_model
                << ", similarity = " << sim << "\n";
    }
    avg_sim = avg_sim / max_sim;
    std::cout << "Average matched topic similarity: " << avg_sim << "\n";

    // --- 3) Set background prior to empirical global feature abundance ---
    // This makes the *prior* expected background distribution match
    // the global feature abundance of the simulated dataset.
    VectorXd eta0 = global_counts.array() * 0.2 + 1e-3;  // small smoothing
    double a0_prior = a0_true;
    double b0_prior = b0_true;
    lda.set_background_prior(eta0, a0_prior, b0_prior);

    // --- 4) Fit the model on the whole corpus as one minibatch ---
    std::cout << "Fitting SVB_DN...\n";
    for (int start = 0; start < N; start += minibatch_size) {
        int end = std::min(start + minibatch_size, N);
        std::vector<Document> minibatch;
        minibatch.reserve(end - start);
        for (int d = start; d < end; ++d) {
            minibatch.push_back(docs[d]);
        }
        lda.partial_fit(minibatch);
    }

    // --- 5) Inspect learned background vs empirical global distribution ---
    VectorXd bg_model = lda.copy_background_model();
    bg_model /= bg_model.sum();   // normalize

    double l1_bg = (bg_model - global_probs).cwiseAbs().sum();
    std::cout << "\nL1 distance between empirical global distribution and "
                 "learned background: " << l1_bg << "\n";

    std::cout << "\nTop 10 background words (word_id: model_prob, global_prob):\n";
    auto bg_top = top_k_indices(bg_model, 10);
    for (int idx : bg_top) {
        std::cout << "  " << idx
                  << ": model="  << bg_model[idx]
                  << ", global=" << global_probs[idx] << "\n";
    }

    std::cout << "\nEstimated foreground / background token counts (a_, b_): "
              << lda.get_forground_count()  << " / "
              << lda.get_background_count() << ", ";
    std::cout << a0_true / (a0_true+b0_true) << " vs "
              << lda.get_background_count()/(lda.get_forground_count() + lda.get_background_count()) << "\n";

    RowMajorMatrixXd model1 = lda.copy_model();
    perm = best_topic_permutation(true_topics, model1);
    avg_sim = 0.0; max_sim = 0.0;
    for (int k_true = 0; k_true < true_topics.rows(); ++k_true) {
        int k_model = perm[k_true];
        Eigen::VectorXd t = true_topics.row(k_true).transpose();
        Eigen::VectorXd m = model1.row(k_model).transpose();
        double st = t.sum();
        double sm = m.sum();
        if (st > 0) t /= st;
        if (sm > 0) m /= sm;
        double sim = t.dot(m);  // in [0,1]
        double sim0 = t.dot(t);
        avg_sim += sim;
        max_sim += sim0;
        sim /= sim0;
        std::cout << "Topic " << k_true
                << " matched to model topic " << k_model
                << ", similarity = " << sim << "\n";
    }
    avg_sim = avg_sim / max_sim;
    std::cout << "Average matched topic similarity: " << avg_sim << "\n";

    return 0;
}
