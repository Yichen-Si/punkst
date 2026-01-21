#include "numerical_utils.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

#include <tbb/tbb.h>
#include <tbb/blocked_range.h>

double logit(double x) {
    if (x <= 0.0 || x >= 1.0) {
        throw std::out_of_range("Input to logit must be in (0, 1)");
    }
    return std::log(x / (1.0 - x));
}

float logit(float x) {
    if (x <= 0.0f || x >= 1.0f) {
        throw std::out_of_range("Input to logit must be in (0, 1)");
    }
    return std::log(x / (1.0f - x));
}

double safe_log10(double x) {
    return (x > 0.0) ? std::log10(x) : -std::numeric_limits<double>::infinity();
}

double normal_sf(double x) {
    return 0.5 * std::erfc(x / std::sqrt(2.0));
}

double twosided_p_from_z(double z) {
    return 2.0 * normal_sf(std::fabs(z));
}

double normal_logsf(double x) {
    const double threshold = 30.0;
    if (x > threshold) {
        // For large x, log(P(Z>x)) ≈ -0.5*x² - log(x) - 0.5*log(2π)
        // (asymptotic expansion)
        const double LOG_2_PI = 1.83787706640934548356;
        return -0.5 * x * x - std::log(x) - 0.5 * LOG_2_PI;
    } else {
        double erfc_val = std::erfc(x / std::sqrt(2.0));
        // In case erfc underflows before our threshold is met, log(0) is -inf.
        if (erfc_val <= 0.0) {
            return -std::numeric_limits<double>::infinity();
        }
        const double LOG_0_5 = -0.69314718055994530941;
        return LOG_0_5 + std::log(erfc_val);
    }
}

double log_twosided_p_from_z(double z) {
    const double LOG_2 = 0.69314718055994530941; // log(2.0)
    return LOG_2 + normal_logsf(std::fabs(z));
}

double normal_log10sf(double x) {
    const double LOG_10 = 2.30258509299404568402; // log(10.0)
    return normal_logsf(x) / LOG_10;
}

double log10_twosided_p_from_z(double z) {
    const double LOG10_2 = 0.3010299956639812; // log10(2.0)
    return LOG10_2 + normal_log10sf(std::fabs(z));
}

double chisq1_log10p(double chi2) {
    double z = std::sqrt(chi2 / 2.0);
    if (z < 10.0) {
        double p = std::erfc(z);
        return -std::log10(p);
    } else { // Switch to Taylor expansion
        double logp = -z * z - std::log(z * std::sqrt(M_PI));
        return -logp / std::log(10.0);
    }
}

std::pair<double, double> chisq2x2_log10p(double a, double b, double c, double d, double pseudocount) {
    if (pseudocount > 0) {
        a += pseudocount;
        b += pseudocount;
        c += pseudocount;
        d += pseudocount;
    }
    double ab = a + b;
    double cd = c + d;
    double ac = a + c;
    double bd = b + d;
    double total = ab + cd;
    if (ab <= 0.0 || cd <= 0.0 || ac <= 0.0 || bd <= 0.0 || total <= 0.0) {
        return {0.0, 0.0};
    }
    double denom = ab * cd * ac * bd;
    if (denom <= 0.0) {
        return {0.0, 0.0};
    }
    double diff = a * d - b * c;
    double chi2 = diff * diff / denom * total;
    if (!std::isfinite(chi2) || chi2 <= 0.0) {
        return {0.0, 0.0};
    }
    return {chi2, chisq1_log10p(chi2)};
}

double cauchy_combination(const std::vector<double>& pval,
                          const std::vector<double>& weights_in,
                          double small_p_approx_thresh) {
    if (pval.empty()) return std::numeric_limits<double>::quiet_NaN();

    const size_t k = pval.size();
    // Build normalized weights
    std::vector<long double> w(k, 1.0L / (long double)k);
    if (!weights_in.empty()) {
        if (weights_in.size() != k) throw std::runtime_error("weight/pval size mismatch");
        long double wsum = 0.0L;
        for (double wi : weights_in) {
            if (!(wi >= 0.0) || std::isnan(wi)) throw std::runtime_error("invalid weight");
            wsum += (long double)wi;
        }
        if (wsum == 0.0L) throw std::runtime_error("sum of weights is zero");
        for (size_t i = 0; i < k; ++i) w[i] = (long double)weights_in[i] / wsum;
    }

    const long double pi = acosl(-1.0L);

    // Clamp away from exactly 0 and 1
    const double pmin = std::nextafter(0.0, 1.0);
    const double pmax = std::nextafter(1.0, 0.0);

    long double sum_stat = 0.0L;
    for (size_t i = 0; i < k; ++i) {
        double p = pval[i];
        if (!(p >= 0.0 && p <= 1.0) || std::isnan(p)) throw std::runtime_error("invalid p-value");
        p = std::clamp(p, pmin, pmax);

        // Compute cot(pi*p) stably.
        long double pl = (long double)p;
        long double term;
        if (pl < (long double)small_p_approx_thresh) {
            term = 1.0L / (pi * pl); // cot(pi*p) ~ 1/(pi*p)
        } else if (1.0L - pl < (long double)small_p_approx_thresh) {
            term = -1.0L / (pi * (1.0L - pl)); // cot(pi*p) ~ -1/(pi*(1-p))
        } else {
            term = 1.0L / tanl(pi * pl);
        }
        sum_stat += w[i] * term;
    }

    long double p_comb = 0.5L - atanl(sum_stat) / pi;
    if (p_comb < 0.0L) p_comb = 0.0L;
    if (p_comb > 1.0L) p_comb = 1.0L;

    return (double)p_comb;
}

// Weighted quadratic local regression at each xi,
//   tricube weights over k-NN window.
// Input: x,y of length n. span \in (0,1].
// Outpu: fitted yhat at each x.
int32_t loess_quadratic_tricube(const std::vector<double>& x,
                                const std::vector<double>& y,
                                std::vector<double>& yhat, double span) {
    const int n = (int)x.size();
    if (n != (int)y.size()) {
        throw std::invalid_argument("loess_quadratic_tricube: invalid input");
    }
    if (n < 3) {
        yhat = y;
        return 0;
    }
    std::vector<int> ord(n);
    std::iota(ord.begin(), ord.end(), 0);
    std::stable_sort(ord.begin(), ord.end(),
                     [&](int a, int b){ return x[a] < x[b]; });

    std::vector<double> xs(n), ys(n);
    for (int r = 0; r < n; ++r) { xs[r] = x[ord[r]]; ys[r] = y[ord[r]]; }

    const int k = std::max(3, (int)std::ceil(std::max(0.0, std::min(1.0, span)) * n));
    std::vector<double> yhat_sorted(n, 0.0);

    tbb::parallel_for(tbb::blocked_range<int>(0, n, 1024),
                      [&](const tbb::blocked_range<int>& range){
        for (int i = range.begin(); i < range.end(); ++i) {
            int L = std::max(0, i - k/2);
            int R = std::min(n - 1, L + k - 1);
            L = std::max(0, R - k + 1);
            const double xi = xs[i];
            const double dmax = std::max(xs[i] - xs[L], xs[R] - xs[i]);

            if (dmax <= 0.0) {
                double S0=0, T0=0;
                for (int j = L; j <= R; ++j) { S0 += 1.0; T0 += ys[j]; }
                yhat_sorted[i] = (S0 > 0.0) ? (T0 / S0) : 0.0;
                continue;
            }

            // normal eqs for quadratic
            double S0=0, S1=0, S2=0, S3=0, S4=0;
            double T0=0, T1=0, T2=0;
            for (int j = L; j <= R; ++j) {
                double u = std::abs(xs[j] - xi) / dmax;           // [0,1]
                double w = std::pow(1.0 - std::pow(u,3.0), 3.0);  // tricube
                double xj = xs[j], yj = ys[j], x2 = xj*xj;
                S0 += w;     S1 += w*xj;    S2 += w*x2;
                S3 += w*x2*xj; S4 += w*x2*x2;
                T0 += w*yj;  T1 += w*xj*yj; T2 += w*x2*yj;
            }

            // Solve 3x3 via adjugate
            double A00=S0, A01=S1, A02=S2,
                   A10=S1, A11=S2, A12=S3,
                   A20=S2, A21=S3, A22=S4;
            double c00 =  A11*A22 - A12*A21;
            double c01 = -(A10*A22 - A12*A20);
            double c02 =  A10*A21 - A11*A20;
            double c10 = -(A01*A22 - A02*A21);
            double c11 =  A00*A22 - A02*A20;
            double c12 = -(A00*A21 - A01*A20);
            double c20 =  A01*A12 - A02*A11;
            double c21 = -(A00*A12 - A02*A10);
            double c22 =  A00*A11 - A01*A10;
            double det = A00*c00 + A01*c01 + A02*c02;

            double yi;
            if (std::abs(det) <= 1e-20 || !std::isfinite(det)) {
                // fallback to local linear
                double W=0, Wx=0, Wy=0, Wxx=0, Wxy=0;
                for (int j = L; j <= R; ++j) {
                    double u = std::abs(xs[j] - xi) / dmax;
                    double w = std::pow(1.0 - std::pow(u,3.0), 3.0);
                    double xj = xs[j], yj = ys[j];
                    W += w; Wx += w*xj; Wy += w*yj; Wxx += w*xj*xj; Wxy += w*xj*yj;
                }
                double det2 = W*Wxx - Wx*Wx;
                yi = (std::abs(det2) <= 1e-20 || !std::isfinite(det2))
                     ? ((W>0)?(Wy/W):0.0)
                     : ((Wxx*Wy - Wx*Wxy)/det2 + (W*Wxy - Wx*Wy)/det2 * xi);
            } else {
                double a0 = (c00*T0 + c01*T1 + c02*T2) / det;
                double a1 = (c10*T0 + c11*T1 + c12*T2) / det;
                double a2 = (c20*T0 + c21*T1 + c22*T2) / det;
                yi = a0 + a1*xi + a2*xi*xi;
            }
            yhat_sorted[i] = yi;
        }
    });

    // unsort
    yhat.resize(n);
    for (int r = 0; r < n; ++r) yhat[ord[r]] = yhat_sorted[r];
    return 1;
}
