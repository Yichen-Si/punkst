#include "leiden.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <deque>
#include <numeric>
#include <random>
#include <stdexcept>
#include <vector>

// -----------------------------------------------------------------------------
// Clean-room Leiden (Traag, Waltman & van Eck, Sci. Rep. 2019) for the
// RBConfiguration objective on a weighted, undirected graph.
//
// Objective (undirected; m2 = 2m = sum of node strengths):
//     Q = (1/m2) * sum_c [ Sigma_in_c - gamma * K_c^2 / m2 ]
// where Sigma_in_c = sum_{i,j in c} A_ij (self-loop contributes A_ii = 2*self_w)
// and K_c is the total strength of community c. At gamma = 1 this equals the
// standard modularity.
//
// Local-move gain for moving node v to community c (both terms use exclusive
// community strength, i.e. the strength of the community without v):
//     score(c) = e(v, c) - gamma * k_v * K_c(excl v) / m2
// The best community is argmax of score; the move is applied when it strictly
// beats staying.
//
// Each pass runs the standard three phases over an aggregation hierarchy:
//   1. fast local move   2. refinement into well-connected sub-communities
//   3. aggregation. Passes are repeated until the quality stops improving or the
// iteration cap is reached. Refinement uses greedy selection among well-connected
// positive-gain sub-communities (a deterministic variant of the paper's
// randomized selection); randomness enters only through the seeded node
// visitation order, which keeps runs reproducible for a fixed seed.
// -----------------------------------------------------------------------------

namespace {

constexpr double kEps = 1e-9;           // strict-improvement threshold for moves
constexpr double kQConvergeEps = 1e-9;  // quality-improvement threshold for convergence
constexpr int32_t kUnboundedCap = 10000; // safety cap when max_iterations < 0

// Weighted undirected graph in CSR form. Self-loops are stored separately in
// self_w (once per node); neighbors/weights hold only inter-node edges, each
// undirected edge appearing once per direction.
struct Graph {
    int32_t n = 0;
    std::vector<int64_t> indptr;   // size n+1
    std::vector<int32_t> nbr;      // size 2E, neighbor node ids
    std::vector<double>  wt;       // size 2E, aligned with nbr
    std::vector<double>  self_w;   // size n, self-loop weight per node
    std::vector<double>  strength; // size n, k_i = 2*self_w[i] + sum incident weights
    double m2 = 0.0;               // sum of strengths == 2m
};

void finalize_strength(Graph& g) {
    g.strength.assign(g.n, 0.0);
    double m2 = 0.0;
    for (int32_t i = 0; i < g.n; ++i) {
        double s = 2.0 * g.self_w[i];
        for (int64_t e = g.indptr[i]; e < g.indptr[i + 1]; ++e) s += g.wt[e];
        g.strength[i] = s;
        m2 += s;
    }
    g.m2 = m2;
}

// Sort each node's adjacency by neighbor id so the graph has a canonical layout
// (makes results independent of input edge order and improves locality).
void sort_adjacency(Graph& g) {
    std::vector<std::pair<int32_t, double>> tmp;
    for (int32_t i = 0; i < g.n; ++i) {
        const int64_t b = g.indptr[i], e = g.indptr[i + 1];
        tmp.clear();
        tmp.reserve(static_cast<size_t>(e - b));
        for (int64_t k = b; k < e; ++k) tmp.emplace_back(g.nbr[k], g.wt[k]);
        std::sort(tmp.begin(), tmp.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });
        for (int64_t k = b; k < e; ++k) {
            g.nbr[k] = tmp[k - b].first;
            g.wt[k] = tmp[k - b].second;
        }
    }
}

double compute_quality(const Graph& g, const std::vector<int32_t>& memb, double gamma) {
    const int32_t n = g.n;
    std::vector<double> sin(n, 0.0), K(n, 0.0);
    for (int32_t i = 0; i < n; ++i) K[memb[i]] += g.strength[i];
    for (int32_t i = 0; i < n; ++i) {
        const int32_t ci = memb[i];
        sin[ci] += 2.0 * g.self_w[i];
        for (int64_t e = g.indptr[i]; e < g.indptr[i + 1]; ++e)
            if (memb[g.nbr[e]] == ci) sin[ci] += g.wt[e];
    }
    double q = 0.0;
    for (int32_t c = 0; c < n; ++c) q += sin[c] - gamma * K[c] * K[c] / g.m2;
    return q / g.m2;
}

// Phase 1: fast local move. Moves individual nodes between communities to
// increase Q, starting from the partition already in `memb`. Returns whether any
// node moved. Community ids are in [0, n).
bool local_move(const Graph& g, std::vector<int32_t>& memb, double gamma, std::mt19937& rng) {
    const int32_t n = g.n;
    const double inv_m2 = 1.0 / g.m2;

    std::vector<double> comm_strength(n, 0.0);
    std::vector<int32_t> comm_size(n, 0);
    for (int32_t i = 0; i < n; ++i) {
        comm_strength[memb[i]] += g.strength[i];
        comm_size[memb[i]] += 1;
    }
    std::vector<int32_t> empty_comms;
    for (int32_t c = 0; c < n; ++c)
        if (comm_size[c] == 0) empty_comms.push_back(c);

    std::vector<double> ewt(n, 0.0);   // scratch: weight from current node to each community
    std::vector<int32_t> touched;
    std::vector<char> in_queue(n, 0);
    std::deque<int32_t> queue;
    std::vector<int32_t> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::shuffle(order.begin(), order.end(), rng);
    for (int32_t v : order) { queue.push_back(v); in_queue[v] = 1; }

    bool any_moved = false;
    while (!queue.empty()) {
        const int32_t v = queue.front();
        queue.pop_front();
        in_queue[v] = 0;
        const int32_t old = memb[v];
        const double kv = g.strength[v];

        touched.clear();
        for (int64_t e = g.indptr[v]; e < g.indptr[v + 1]; ++e) {
            const int32_t c = memb[g.nbr[e]];
            if (ewt[c] == 0.0) touched.push_back(c);
            ewt[c] += g.wt[e];
        }

        double best_score = ewt[old] - gamma * kv * (comm_strength[old] - kv) * inv_m2;
        int32_t best_c = old;
        bool best_empty = false;
        for (int32_t c : touched) {
            if (c == old) continue;
            const double score = ewt[c] - gamma * kv * comm_strength[c] * inv_m2;
            if (score > best_score + kEps) { best_score = score; best_c = c; best_empty = false; }
        }
        // Consider isolating v into a fresh community (score 0) when it is not
        // already alone and an empty community id is available.
        if (comm_size[old] > 1 && !empty_comms.empty()) {
            if (0.0 > best_score + kEps) { best_score = 0.0; best_empty = true; }
        }

        for (int32_t c : touched) ewt[c] = 0.0;

        int32_t target = best_c;
        if (best_empty) { target = empty_comms.back(); empty_comms.pop_back(); }
        if (target != old) {
            comm_strength[old] -= kv;
            comm_size[old] -= 1;
            if (comm_size[old] == 0) empty_comms.push_back(old);
            comm_strength[target] += kv;
            comm_size[target] += 1;
            memb[v] = target;
            any_moved = true;
            for (int64_t e = g.indptr[v]; e < g.indptr[v + 1]; ++e) {
                const int32_t j = g.nbr[e];
                if (memb[j] != target && !in_queue[j]) { queue.push_back(j); in_queue[j] = 1; }
            }
        }
    }
    return any_moved;
}

// Phase 2: refinement. Within each community of `memb`, start from singletons and
// greedily merge well-connected nodes into well-connected sub-communities. The
// returned partition (relabeled 0..R-1) is a refinement of `memb`.
std::vector<int32_t> refine(const Graph& g, const std::vector<int32_t>& memb,
                            double gamma, std::mt19937& rng, int32_t& n_refined_out) {
    const int32_t n = g.n;
    const double inv_m2 = 1.0 / g.m2;

    std::vector<double> Kcomm(n, 0.0);
    for (int32_t i = 0; i < n; ++i) Kcomm[memb[i]] += g.strength[i];

    // e_in_P[i] = weight from i to other nodes in the same community.
    std::vector<double> e_in_P(n, 0.0);
    for (int32_t i = 0; i < n; ++i) {
        double s = 0.0;
        for (int64_t e = g.indptr[i]; e < g.indptr[i + 1]; ++e)
            if (memb[g.nbr[e]] == memb[i]) s += g.wt[e];
        e_in_P[i] = s;
    }

    // Group nodes by community (counting sort).
    std::vector<int32_t> start(n + 1, 0);
    for (int32_t i = 0; i < n; ++i) start[memb[i] + 1] += 1;
    for (int32_t c = 0; c < n; ++c) start[c + 1] += start[c];
    std::vector<int32_t> nodes_by_comm(n);
    {
        std::vector<int32_t> pos(start.begin(), start.end());
        for (int32_t i = 0; i < n; ++i) nodes_by_comm[pos[memb[i]]++] = i;
    }

    // Refined partition: singletons keyed by node id.
    std::vector<int32_t> refined(n);
    std::iota(refined.begin(), refined.end(), 0);
    std::vector<double> sub_strength(n);
    std::vector<double> sub_ext(n);      // E(C, R\C): weight from sub-community to rest of its community
    std::vector<int32_t> sub_size(n, 1);
    for (int32_t i = 0; i < n; ++i) {
        sub_strength[i] = g.strength[i];
        sub_ext[i] = e_in_P[i];
    }

    std::vector<double> ewt(n, 0.0);
    std::vector<int32_t> touched;

    for (int32_t c = 0; c < n; ++c) {
        const int32_t b = start[c], en = start[c + 1];
        if (en - b <= 1) continue;   // nothing to refine
        const double K_R = Kcomm[c];

        std::vector<int32_t> order(nodes_by_comm.begin() + b, nodes_by_comm.begin() + en);
        std::shuffle(order.begin(), order.end(), rng);
        for (int32_t v : order) {
            if (sub_size[refined[v]] != 1) continue;   // only singletons move
            const double kv = g.strength[v];
            // v must be well-connected to the rest of its community.
            if (e_in_P[v] < gamma * kv * (K_R - kv) * inv_m2 - kEps) continue;

            touched.clear();
            for (int64_t e = g.indptr[v]; e < g.indptr[v + 1]; ++e) {
                const int32_t j = g.nbr[e];
                if (memb[j] != c) continue;
                const int32_t rc = refined[j];
                if (ewt[rc] == 0.0) touched.push_back(rc);
                ewt[rc] += g.wt[e];
            }

            int32_t best_c = -1;
            double best_score = kEps;   // require a strictly positive gain
            for (int32_t C : touched) {
                if (C == refined[v]) continue;
                // C must itself be well-connected to the rest of its community.
                if (sub_ext[C] < gamma * sub_strength[C] * (K_R - sub_strength[C]) * inv_m2 - kEps) continue;
                const double score = ewt[C] - gamma * kv * sub_strength[C] * inv_m2;
                if (score > best_score) { best_score = score; best_c = C; }
            }
            if (best_c >= 0) {
                const double eC = ewt[best_c];
                sub_ext[best_c] += e_in_P[v] - 2.0 * eC;
                sub_strength[best_c] += kv;
                sub_size[best_c] += 1;
                sub_size[refined[v]] -= 1;
                refined[v] = best_c;
            }
            for (int32_t C : touched) ewt[C] = 0.0;
        }
    }

    // Relabel refined ids to contiguous 0..R-1.
    std::vector<int32_t> remap(n, -1);
    int32_t next = 0;
    for (int32_t i = 0; i < n; ++i) {
        if (remap[refined[i]] == -1) remap[refined[i]] = next++;
        refined[i] = remap[refined[i]];
    }
    n_refined_out = next;
    return refined;
}

// Phase 3: build the aggregate graph whose nodes are the refined sub-communities.
Graph aggregate(const Graph& g, const std::vector<int32_t>& refined, int32_t nref) {
    Graph gn;
    gn.n = nref;
    gn.self_w.assign(nref, 0.0);
    gn.indptr.assign(nref + 1, 0);

    std::vector<int32_t> start(nref + 1, 0);
    for (int32_t i = 0; i < g.n; ++i) start[refined[i] + 1] += 1;
    for (int32_t a = 0; a < nref; ++a) start[a + 1] += start[a];
    std::vector<int32_t> members(g.n);
    {
        std::vector<int32_t> pos(start.begin(), start.end());
        for (int32_t i = 0; i < g.n; ++i) members[pos[refined[i]]++] = i;
    }

    std::vector<double> acc(nref, 0.0);
    std::vector<int32_t> touched;
    std::vector<int64_t> deg(nref, 0);

    // Pass 1: accumulate self-loops and count distinct inter-community neighbors.
    for (int32_t a = 0; a < nref; ++a) {
        touched.clear();
        for (int32_t idx = start[a]; idx < start[a + 1]; ++idx) {
            const int32_t i = members[idx];
            gn.self_w[a] += g.self_w[i];
            for (int64_t e = g.indptr[i]; e < g.indptr[i + 1]; ++e) {
                const int32_t j = g.nbr[e];
                const int32_t bb = refined[j];
                if (bb == a) {
                    if (i < j) gn.self_w[a] += g.wt[e];   // count each intra edge once
                } else {
                    if (acc[bb] == 0.0) touched.push_back(bb);
                    acc[bb] += g.wt[e];
                }
            }
        }
        deg[a] = static_cast<int64_t>(touched.size());
        for (int32_t bb : touched) acc[bb] = 0.0;
    }
    for (int32_t a = 0; a < nref; ++a) gn.indptr[a + 1] = gn.indptr[a] + deg[a];
    const int64_t total = gn.indptr[nref];
    gn.nbr.resize(total);
    gn.wt.resize(total);

    // Pass 2: emit inter-community edges.
    for (int32_t a = 0; a < nref; ++a) {
        touched.clear();
        for (int32_t idx = start[a]; idx < start[a + 1]; ++idx) {
            const int32_t i = members[idx];
            for (int64_t e = g.indptr[i]; e < g.indptr[i + 1]; ++e) {
                const int32_t bb = refined[g.nbr[e]];
                if (bb == a) continue;
                if (acc[bb] == 0.0) touched.push_back(bb);
                acc[bb] += g.wt[e];
            }
        }
        int64_t w = gn.indptr[a];
        for (int32_t bb : touched) {
            gn.nbr[w] = bb;
            gn.wt[w] = acc[bb];
            ++w;
            acc[bb] = 0.0;
        }
    }

    finalize_strength(gn);
    assert(std::abs(gn.m2 - g.m2) <= 1e-6 * g.m2 + 1e-9);   // total weight is invariant
    return gn;
}

// One Leiden pass: build the aggregation hierarchy starting from `init_memb`,
// returning the resulting partition over the original nodes of g0.
std::vector<int32_t> one_pass(const Graph& g0, const std::vector<int32_t>& init_memb,
                              double gamma, std::mt19937& rng) {
    Graph g = g0;                       // level 0 operates on a working copy
    std::vector<int32_t> P = init_memb;
    std::vector<int32_t> cur_of_orig(g0.n);
    std::iota(cur_of_orig.begin(), cur_of_orig.end(), 0);

    while (true) {
        local_move(g, P, gamma, rng);
        int32_t nref = 0;
        std::vector<int32_t> refined = refine(g, P, gamma, rng, nref);
        if (nref >= g.n) break;         // refinement produced no aggregation

        Graph g_next = aggregate(g, refined, nref);
        std::vector<int32_t> P_next(nref, -1);
        for (int32_t i = 0; i < g.n; ++i) {
            const int32_t a = refined[i];
            if (P_next[a] < 0) P_next[a] = P[i];
        }
        // Relabel projected community ids to a contiguous [0, C) range so they
        // stay within the aggregate graph's node count (scratch arrays in the
        // next level's local_move/refine are sized to the node count).
        {
            std::vector<int32_t> remap(g0.n, -1);
            int32_t C = 0;
            for (int32_t a = 0; a < nref; ++a) {
                if (remap[P_next[a]] == -1) remap[P_next[a]] = C++;
                P_next[a] = remap[P_next[a]];
            }
        }
        for (int32_t i = 0; i < g0.n; ++i) cur_of_orig[i] = refined[cur_of_orig[i]];
        g = std::move(g_next);
        P = std::move(P_next);
    }

    std::vector<int32_t> out(g0.n);
    for (int32_t i = 0; i < g0.n; ++i) out[i] = P[cur_of_orig[i]];
    return out;
}

LeidenResult run_leiden(const Graph& g0, const LeidenOptions& options) {
    const int32_t n = g0.n;
    const double gamma = options.resolution;
    std::mt19937 rng(static_cast<uint32_t>(options.seed));

    std::vector<int32_t> memb(n);
    std::iota(memb.begin(), memb.end(), 0);   // start from singletons
    double q_prev = compute_quality(g0, memb, gamma);

    const int32_t cap = options.max_iterations < 0 ? kUnboundedCap : options.max_iterations;
    int32_t iter = 0;
    bool converged = false;
    while (iter < cap) {
        std::vector<int32_t> newm = one_pass(g0, memb, gamma, rng);
        ++iter;
        const double q_new = compute_quality(g0, newm, gamma);
        memb = std::move(newm);
        if (q_new <= q_prev + kQConvergeEps) { q_prev = q_new; converged = true; break; }
        q_prev = q_new;
    }

    std::vector<int32_t> remap(n, -1);
    int32_t K = 0;
    LeidenResult res;
    res.membership.resize(n);
    for (int32_t i = 0; i < n; ++i) {
        if (remap[memb[i]] == -1) remap[memb[i]] = K++;
        res.membership[i] = remap[memb[i]];
    }
    res.n_communities = K;
    res.quality = q_prev;
    res.iterations = iter;
    res.converged = converged;
    return res;
}

Graph build_from_sparse(const Eigen::Ref<const LeidenSparseMatrix>& A) {
    if (A.rows() != A.cols())
        throw std::invalid_argument("leiden_cluster: adjacency matrix must be square");
    const int32_t n = static_cast<int32_t>(A.rows());
    Graph g;
    g.n = n;
    g.self_w.assign(n, 0.0);
    g.indptr.assign(n + 1, 0);

    using RefIter = Eigen::Ref<const LeidenSparseMatrix>::InnerIterator;
    for (int32_t i = 0; i < n; ++i) {
        for (RefIter it(A, i); it; ++it) {
            const int32_t j = static_cast<int32_t>(it.col());
            const double w = it.value();
            if (!std::isfinite(w) || w < 0.0)
                throw std::invalid_argument("leiden_cluster: weights must be finite and non-negative");
            if (w == 0.0) continue;
            if (j == i) g.self_w[i] += w;
            else g.indptr[i + 1] += 1;
        }
    }
    for (int32_t i = 0; i < n; ++i) g.indptr[i + 1] += g.indptr[i];
    g.nbr.resize(g.indptr[n]);
    g.wt.resize(g.indptr[n]);
    std::vector<int64_t> pos(g.indptr.begin(), g.indptr.end());
    for (int32_t i = 0; i < n; ++i) {
        for (RefIter it(A, i); it; ++it) {
            const int32_t j = static_cast<int32_t>(it.col());
            const double w = it.value();
            if (w <= 0.0 || j == i) continue;
            g.nbr[pos[i]] = j;
            g.wt[pos[i]] = w;
            pos[i]++;
        }
    }
    sort_adjacency(g);
    finalize_strength(g);
    if (g.m2 <= 0.0)
        throw std::invalid_argument("leiden_cluster: graph has no positive-weight edges");
    return g;
}

Graph build_from_edges(int32_t n, const std::vector<std::pair<int32_t, int32_t>>& edges,
                       const std::vector<double>& weights) {
    if (n <= 0)
        throw std::invalid_argument("leiden_cluster: n_nodes must be positive");
    if (edges.size() != weights.size())
        throw std::invalid_argument("leiden_cluster: edges and weights must have equal length");
    Graph g;
    g.n = n;
    g.self_w.assign(n, 0.0);
    g.indptr.assign(n + 1, 0);

    const size_t E = edges.size();
    for (size_t k = 0; k < E; ++k) {
        const int32_t u = edges[k].first, v = edges[k].second;
        const double w = weights[k];
        if (u < 0 || u >= n || v < 0 || v >= n)
            throw std::invalid_argument("leiden_cluster: edge endpoint out of range");
        if (!std::isfinite(w) || w < 0.0)
            throw std::invalid_argument("leiden_cluster: weights must be finite and non-negative");
        if (w == 0.0) continue;
        if (u == v) g.self_w[u] += w;
        else { g.indptr[u + 1] += 1; g.indptr[v + 1] += 1; }
    }
    for (int32_t i = 0; i < n; ++i) g.indptr[i + 1] += g.indptr[i];
    g.nbr.resize(g.indptr[n]);
    g.wt.resize(g.indptr[n]);
    std::vector<int64_t> pos(g.indptr.begin(), g.indptr.end());
    for (size_t k = 0; k < E; ++k) {
        const int32_t u = edges[k].first, v = edges[k].second;
        const double w = weights[k];
        if (w <= 0.0 || u == v) continue;
        g.nbr[pos[u]] = v; g.wt[pos[u]] = w; pos[u]++;
        g.nbr[pos[v]] = u; g.wt[pos[v]] = w; pos[v]++;
    }
    sort_adjacency(g);
    finalize_strength(g);
    if (g.m2 <= 0.0)
        throw std::invalid_argument("leiden_cluster: graph has no positive-weight edges");
    return g;
}

} // namespace

LeidenResult leiden_cluster(const Eigen::Ref<const LeidenSparseMatrix>& adjacency,
                            const LeidenOptions& options) {
    return run_leiden(build_from_sparse(adjacency), options);
}

LeidenResult leiden_cluster(int32_t n_nodes,
                            const std::vector<std::pair<int32_t, int32_t>>& edges,
                            const std::vector<double>& weights,
                            const LeidenOptions& options) {
    return run_leiden(build_from_edges(n_nodes, edges, weights), options);
}
