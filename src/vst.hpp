#pragma once

#include "numerical_utils.hpp"
#include "dataunits.hpp"
#include <memory>
#include <tbb/tbb.h>
#include <tbb/enumerable_thread_specific.h>
#include <tbb/blocked_range.h>

struct HVF_VST {

    struct VSTOutputs {
        std::vector<double> sum_all;
        std::vector<double> mean_all;
        std::vector<double> var_all;
        std::vector<double> var_expected;     // 10^fitted
        std::vector<double> var_standardized; // ranking score
    };

    VSTOutputs stats;
    double loess_span, clip_max;

    HVF_VST(double l = 0.30, double c = -1.0) : loess_span(l), clip_max(c) {}

    int32_t ComputeStats(const std::vector<Document>& docs, uint32_t M) {
        const uint32_t N = (uint32_t)docs.size();
        if (M < 3 || N < 2) {return 0;}

        // ---- Pass 1: accumulate sums per feature
        struct Accum {
            std::vector<double> sum, sumsq;
            std::vector<uint32_t> nnz;
            Accum(size_t m=0): sum(m,0.0), sumsq(m,0.0), nnz(m,0u) {}
            Accum(const Accum& a): sum(a.sum), sumsq(a.sumsq), nnz(a.nnz) {}
            Accum& operator+=(const Accum& o){
                const size_t m = sum.size();
                for (size_t j=0;j<m;++j){
                    sum[j]   += o.sum[j];
                    sumsq[j] += o.sumsq[j];
                    nnz[j]   += o.nnz[j];
                }
                return *this;
            }
        };

        tbb::enumerable_thread_specific<Accum> ets([&]{ return Accum(M); });

        tbb::parallel_for(tbb::blocked_range<size_t>(0, N, 64),
                        [&](const tbb::blocked_range<size_t>& r){
            auto& local = ets.local();
            auto& sum   = local.sum;
            auto& sumsq = local.sumsq;
            auto& nnz   = local.nnz;
            for (size_t i = r.begin(); i < r.end(); ++i) {
                const auto& d = docs[i];
                for (size_t k = 0; k < d.ids.size(); ++k) {
                    const uint32_t j = d.ids[k];
                    if (j >= M) continue;
                    const double x = d.cnts[k];
                    sum[j]   += x;
                    sumsq[j] += x * x;
                    nnz[j]   += 1u;
                }
            }
        });

        // Reduce thread-locals
        std::vector<double> sum(M,0.0), sumsq(M,0.0);
        std::vector<uint32_t> nnz(M,0);
        ets.combine_each([&](const Accum& a){
            const size_t m = a.sum.size();
            for (size_t j=0;j<m;++j){
                sum[j]   += a.sum[j];
                sumsq[j] += a.sumsq[j];
                nnz[j]   += a.nnz[j];
            }
        });

        // Means & population variances across ALL cells (feature-parallel)
        std::vector<double> mean(M,0.0), var(M,0.0);
        tbb::parallel_for(uint32_t(0), M, [&](uint32_t j){
            const double mu = sum[j] / double(N);
            mean[j] = mu;
            const double S = sumsq[j] - 2.0*mu*sum[j] + double(N)*mu*mu;
            var[j]  = (S <= 0.0) ? 0.0 : (S / double(N));
        });

        // Build inputs for loess on non-constant rows
        std::vector<int> idx_nc; idx_nc.reserve(M);
        std::vector<double> lx;  lx.reserve(M);
        std::vector<double> ly;  ly.reserve(M);
        for (uint32_t j=0;j<M;++j) if (var[j] > 0.0) {
            idx_nc.push_back((int)j);
            lx.push_back(safe_log10(mean[j]));
            ly.push_back(safe_log10(var[j]));
        }

        // LOESS fit (parallel) → expected variance
        std::vector<double> fitted;
        if (!idx_nc.empty())
            int32_t ret = loess_quadratic_tricube(lx, ly, fitted, loess_span);

        std::vector<double> vexp(M, 0.0);
        for (size_t t = 0; t < idx_nc.size(); ++t) {
            const int j = idx_nc[t];
            vexp[j] = std::pow(10.0, fitted[t]);
        }

        // Prepare standardized-variance accumulators
        const double vmax = (clip_max > 0.0) ? clip_max : std::sqrt((double)N);
        std::vector<double> sumz(M, 0.0), sumz2(M, 0.0);

        // zeros' contribution per feature (feature-parallel)
        tbb::parallel_for(uint32_t(0), M, [&](uint32_t j){
            const double sd = std::sqrt(std::max(0.0, vexp[j]));
            if (sd <= 0.0 || N == 0) return;
            const double z0 = clamp((0.0 - mean[j]) / sd, -vmax, vmax);
            const uint32_t n0 = N - nnz[j];
            sumz[j]  = z0 * (double)n0;
            sumz2[j] = z0 * z0 * (double)n0;
        });

        // non-zeros' contribution via thread-local accumulators
        tbb::enumerable_thread_specific<Accum> etsZ([&]{ return Accum(M); });
        tbb::parallel_for(tbb::blocked_range<size_t>(0, N, 64),
                        [&](const tbb::blocked_range<size_t>& r){
            auto& local = etsZ.local();
            auto& lz  = local.sum;
            auto& lz2 = local.sumsq;
            for (size_t i = r.begin(); i < r.end(); ++i) {
                const auto& d = docs[i];
                const size_t K = d.ids.size();
                for (size_t k = 0; k < K; ++k) {
                    const uint32_t j = d.ids[k];
                    if (j >= M) continue;
                    const double  x = d.cnts[k];
                    const double sd = std::sqrt(std::max(0.0, vexp[j]));
                    if (sd <= 0.0) continue;
                    double z = (x - mean[j]) / sd;
                    z = clamp(z, -vmax, vmax);
                    lz[j]  += z;
                    lz2[j] += z*z;
                }
            }
        });
        // Reduce into global sumz, sumz2 (feature-parallel)
        std::vector<const Accum*> locals;
        etsZ.combine_each([&](const Accum& a){ locals.push_back(&a); });
        tbb::parallel_for(uint32_t(0), M, [&](uint32_t j){
            double add1 = 0.0, add2 = 0.0;
            for (const Accum* a : locals) {
                add1 += a->sum[j];
                add2 += a->sumsq[j];
            }
            sumz[j]  += add1;
            sumz2[j] += add2;
        });

        // final variance of standardized, clipped values (feature-parallel)
        std::vector<double> var_std(M, 0.0);
        tbb::parallel_for(uint32_t(0), M, [&](uint32_t j){
            const double mz = sumz[j]  / (double)N;
            const double vz = sumz2[j] / (double)N - mz*mz;
            var_std[j] = (vz > 0.0 && std::isfinite(vz)) ? vz : 0.0;
        });

        stats.sum_all  = std::move(sum);
        stats.mean_all = std::move(mean);
        stats.var_all  = std::move(var);
        stats.var_expected     = std::move(vexp);
        stats.var_standardized = std::move(var_std);
        return 1;
    }

    // Returns indices sorted by decreasing var_standardized.
    std::vector<uint32_t> SelectVST(const std::vector<Document>& docs, uint32_t n_features) {
        int32_t ret = ComputeStats(docs, n_features);
        std::vector<uint32_t> idx(n_features);
        std::iota(idx.begin(), idx.end(), 0u);
        if (ret == 0) return idx;
        std::stable_sort(idx.begin(), idx.end(),
            [&](uint32_t a, uint32_t b){
                if (stats.var_standardized[a] != stats.var_standardized[b])
                    return stats.var_standardized[a] > stats.var_standardized[b];
                return a < b;
            });
        return idx;
    }
};
