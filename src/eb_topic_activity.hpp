#pragma once

#include "dataunits.hpp"
#include "lda.hpp"
#include "numerical_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace ebact {

enum class ComponentKind { Uniform, Beta };

struct Component {
    ComponentKind kind = ComponentKind::Uniform;
    bool is_null = false;
    double l = 0.0;
    double u = 1.0;
    double c = 1.0;
    double d = 1.0;

    static Component Uniform(double l_, double u_, bool is_null_ = false) {
        Component x;
        x.kind = ComponentKind::Uniform;
        x.is_null = is_null_;
        x.l = l_;
        x.u = u_;
        return x;
    }

    static Component Beta(double c_, double d_) {
        Component x;
        x.kind = ComponentKind::Beta;
        x.c = c_;
        x.d = d_;
        return x;
    }
};

struct FitResult {
    std::vector<double> eta; // G+1
    std::vector<std::vector<double>> resp; // N x (G+1), posterior probability of each component for each unit
    std::vector<double> z_null; // N, P(Z_{ik}=0 | data)
    double loglik = NEG_INF;
    int32_t iterations = 0;
};

inline const char* component_kind_name(ComponentKind kind) {
    return kind == ComponentKind::Uniform ? "uniform" : "beta";
}

inline void validate_components(const std::vector<Component>& comps) {
    if (comps.size() < 2) {
        throw std::invalid_argument("EB activity requires a null and at least one active component");
    }
    if (!comps[0].is_null || comps[0].kind != ComponentKind::Uniform) {
        throw std::invalid_argument("first EB component must be the near-zero uniform null");
    }
    for (const auto& comp : comps) {
        if (comp.kind == ComponentKind::Uniform &&
                !(0.0 <= comp.l && comp.l < comp.u && comp.u <= 1.0)) {
            throw std::invalid_argument("invalid uniform EB component");
        }
        if (comp.kind == ComponentKind::Beta && !(comp.c > 0.0 && comp.d > 0.0)) {
            throw std::invalid_argument("invalid beta EB component");
        }
    }
}

inline FitResult fit_eta_em(const std::vector<std::vector<double>>& logH,
        int32_t max_iter, double tol, double pseudocount) {
    const int32_t N = static_cast<int32_t>(logH.size());
    if (N == 0) {
        throw std::invalid_argument("empty EB evidence matrix");
    }
    const int32_t C = static_cast<int32_t>(logH[0].size());
    if (C == 0) {
        throw std::invalid_argument("empty EB component dimension");
    }
    for (const auto& row : logH) {
        if (static_cast<int32_t>(row.size()) != C) {
            throw std::invalid_argument("ragged EB evidence matrix");
        }
    }

    std::vector<double> eta(C, 1.0 / static_cast<double>(C));
    std::vector<double> old_eta(C);
    std::vector<std::vector<double>> resp(N, std::vector<double>(C, 0.0));
    double old_ll = NEG_INF;
    int32_t iter = 0;

    for (; iter < max_iter; ++iter) {
        double ll = 0.0;
        std::vector<double> colsum(C, 0.0);
        for (int32_t i = 0; i < N; ++i) {
            std::vector<double> tmp(C);
            for (int32_t g = 0; g < C; ++g) {
                tmp[g] = safe_log(eta[g]) + logH[i][g];
            }
            const double den = logsumexp(tmp);
            ll += den;
            for (int32_t g = 0; g < C; ++g) {
                resp[i][g] = std::exp(tmp[g] - den);
                colsum[g] += resp[i][g];
            }
        }

        old_eta = eta;
        const double denom = static_cast<double>(N) + static_cast<double>(C) * pseudocount;
        for (int32_t g = 0; g < C; ++g) {
            eta[g] = (colsum[g] + pseudocount) / denom;
        }

        double max_change = 0.0;
        for (int32_t g = 0; g < C; ++g) {
            max_change = std::max(max_change, std::abs(eta[g] - old_eta[g]));
        }
        if (std::isfinite(old_ll) && std::abs(ll - old_ll) < tol && max_change < std::sqrt(tol)) {
            old_ll = ll;
            break;
        }
        old_ll = ll;
    }

    std::vector<double> z_null(N, 0.0);
    for (int32_t i = 0; i < N; ++i) {
        z_null[i] = resp[i][0];
    }
    return FitResult{eta, resp, z_null, old_ll, std::min(iter + 1, max_iter)};
}

inline RowVectorXd floored_normalized(RowVectorXd v, double floor) {
    for (Eigen::Index m = 0; m < v.cols(); ++m) {
        v(m) = std::max(floor, v(m));
    }
    const double s = v.sum();
    if (!(s > 0.0)) {
        throw std::runtime_error("cannot normalize nonpositive EB probability vector");
    }
    v /= s;
    return v;
}

inline double multinomial_log_lr(const Document& doc, const RowVectorXd& q,
        const RowVectorXd& beta, double a) {
    double ans = 0.0;
    for (size_t j = 0; j < doc.ids.size(); ++j) {
        const int32_t m = static_cast<int32_t>(doc.ids[j]);
        if (m < 0 || m >= beta.cols()) {
            throw std::out_of_range("document feature index is outside EB model");
        }
        const double qm = q(m);
        const double bm = beta(m);
        if (!(qm > 0.0 && bm > 0.0)) {
            throw std::runtime_error("EB multinomial probabilities must be positive");
        }
        const double ratio = 1.0 + a * (bm / qm - 1.0);
        if (!(ratio > 0.0)) {
            return NEG_INF;
        }
        ans += doc.cnts[j] * std::log(ratio);
    }
    return ans;
}

inline std::vector<std::vector<double>> multinomial_log_evidence(
        const std::vector<Document>& docs, const RowMajorMatrixXd& q,
        const RowVectorXd& beta, const std::vector<Component>& comps,
        double prob_floor, int32_t quad_subdivisions) {
    validate_components(comps);
    if (q.rows() != static_cast<Eigen::Index>(docs.size()) || q.cols() != beta.cols()) {
        throw std::invalid_argument("EB multinomial dimensions do not match");
    }
    const int32_t N = static_cast<int32_t>(docs.size());
    const int32_t C = static_cast<int32_t>(comps.size());
    const RowVectorXd beta_safe = floored_normalized(beta, prob_floor);
    std::vector<std::vector<double>> out(N, std::vector<double>(C, NEG_INF));
    for (int32_t i = 0; i < N; ++i) {
        const RowVectorXd q_safe = floored_normalized(q.row(i), prob_floor);
        for (int32_t g = 0; g < C; ++g) {
            const Component& comp = comps[g];
            if (comp.kind == ComponentKind::Uniform) {
                auto logf = [&](double a) { return multinomial_log_lr(docs[i], q_safe, beta_safe, a); };
                out[i][g] = GaussLegendre16::log_integrate(logf, comp.l, comp.u, quad_subdivisions)
                    - std::log(comp.u - comp.l);
            } else {
                auto logf = [&](double a) {
                    return multinomial_log_lr(docs[i], q_safe, beta_safe, a)
                        + log_beta_density(a, comp.c, comp.d);
                };
                out[i][g] = GaussLegendre16::log_integrate(logf, 0.0, 1.0, quad_subdivisions);
            }
        }
    }
    return out;
}

inline double beta_kernel_log(double a, double A, double B) {
    if (!(a > 0.0 && a < 1.0)) {
        return NEG_INF;
    }
    return (A - 1.0) * std::log(a) + (B - 1.0) * std::log1p(-a);
}

inline std::vector<std::vector<double>> beta_marginal_log_evidence(
        const RowMajorMatrixXd& gamma_counts, int32_t focal_k, double alpha,
        const std::vector<Component>& comps, int32_t quad_subdivisions) {
    validate_components(comps);
    if (!(alpha > 0.0)) {
        throw std::invalid_argument("alpha must be positive for EB beta-marginal evidence");
    }
    const int32_t N = static_cast<int32_t>(gamma_counts.rows());
    const int32_t K = static_cast<int32_t>(gamma_counts.cols());
    if (K < 2 || focal_k < 0 || focal_k >= K) {
        throw std::invalid_argument("invalid EB beta-marginal factor index");
    }
    const int32_t C = static_cast<int32_t>(comps.size());
    std::vector<std::vector<double>> out(N, std::vector<double>(C, NEG_INF));
    for (int32_t i = 0; i < N; ++i) {
        const double gamma_sum = gamma_counts.row(i).sum();
        const double ak_post = gamma_counts(i, focal_k) + alpha;
        const double ank_post = gamma_sum - gamma_counts(i, focal_k) +
            (static_cast<double>(K) - 1.0) * alpha;
        const double A = ak_post - alpha + 1.0;
        const double B = ank_post - (static_cast<double>(K) - 1.0) * alpha + 1.0;
        if (!(A > 0.0 && B > 0.0)) {
            throw std::runtime_error("nonpositive EB beta-marginal likelihood shape");
        }
        for (int32_t g = 0; g < C; ++g) {
            const Component& comp = comps[g];
            if (comp.kind == ComponentKind::Uniform) {
                auto logf = [&](double a) { return beta_kernel_log(a, A, B); };
                out[i][g] = GaussLegendre16::log_integrate(logf, comp.l, comp.u, quad_subdivisions)
                    - std::log(comp.u - comp.l);
            } else {
                const double left_shape = A + comp.c - 1.0;
                const double right_shape = B + comp.d - 1.0;
                if (!(left_shape > 0.0 && right_shape > 0.0)) {
                    throw std::runtime_error("EB beta slab evidence is not integrable");
                }
                out[i][g] = log_beta_fn(left_shape, right_shape) - log_beta_fn(comp.c, comp.d);
            }
        }
    }
    return out;
}

} // namespace ebact
