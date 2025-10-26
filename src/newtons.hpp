#pragma once

#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

#include <Eigen/Core>
#include "Eigen/Dense"
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::ArrayXd;

struct LineSearchOptions {
	bool   enabled        = false;
	double beta           = 0.5;    // step shrink factor
	double c1             = 1e-4;   // Armijo constant
	int    max_backtracks = 20;
};

struct ACGOptions {
	bool   enabled    = false;
	double L0         = 1.0;     // initial Lipschitz guess (ignored if alpha>0)
	double bt_inc     = 2.0;     // multiply L by this on backtracking
	bool   monotone   = true;    // monotone FISTA
	bool   restart    = true;    // gradient-based restart
};

struct TrustRegionOptions {
	bool   enabled      = false;
	double delta_init   = 1.0;     // initial trust radius
	double delta_max    = 1e6;     // max trust radius
	double eta          = 1e-4;    // acceptance threshold
	double cg_tol       = 1e-4;    // relative CG tolerance
	int    cg_max_iter  = -1;      // default: 2*K
};

struct OptimOptions {
	int    max_iters = 50;
	double tol       = 1e-6;
	double alpha     = 1.0;     // initial step size
	double eps       = 1e-12;
	bool use_agd     = false;
	LineSearchOptions ls{};
	ACGOptions acg{};
	TrustRegionOptions tron{};
	// Optional box constraints
    std::shared_ptr<const VectorXd> b_min = nullptr;
    std::shared_ptr<const VectorXd> b_max = nullptr;
    void set_bounds(double min_val, double max_val, int K) {
        b_min = std::make_shared<VectorXd>(VectorXd::Constant(K, min_val));
        b_max = std::make_shared<VectorXd>(VectorXd::Constant(K, max_val));
    }
    void set_bounds(const VectorXd& min_vec, const VectorXd& max_vec) {
        b_min = std::make_shared<VectorXd>(min_vec);
        b_max = std::make_shared<VectorXd>(max_vec);
    }
};

struct OptimStats {
	int niters = 0;
	double obj, diff_obj;
	double diff_b, rel_diff_b;
};

inline void project_to_box(VectorXd& b, const OptimOptions& opt) {
    if (opt.b_min && opt.b_max) {
        b = b.cwiseMax(*opt.b_min).cwiseMin(*opt.b_max);
    } else if (opt.b_min) { // Only lower bound provided
        b = b.cwiseMax(*opt.b_min);
    } else if (opt.b_max) { // Only upper bound provided
        b = b.cwiseMin(*opt.b_max);
    } else { // Default non-negative case
        b = b.cwiseMax(0.0);
    }
}

// ---------------- TRON (trust-region Newton-CG) ----------------
template <class Problem>
double tron_solve(const Problem& P, VectorXd& b, const OptimOptions& opt, OptimStats& stats, int debug_ = 0, double* delta_new = nullptr, double rho_t = -1) {
	const int K = static_cast<int>(b.size());
	VectorXd g(K), q(K);
	ArrayXd w;
	double delta = opt.tron.delta_init;
	double f_cur = P.f(b);
	double f_diff, b_diff, f_rel, b_rel;
	int it = 0;
	for (; it < opt.max_iters; ++it) {
		P.eval(b, nullptr, &g, &q, &w);

		// Free set
		Eigen::Array<bool, Eigen::Dynamic, 1> free_mask = g.array() < 0.0;
		// (b.array() > 0.0) || (g.array() < 0.0);
		if (!opt.b_min && !opt.b_max) {
			free_mask = free_mask || (b.array() > 0.0);
		} else {
			if (opt.b_min) {
				free_mask = free_mask || (b.array() > opt.b_min->array());
			}
			if (opt.b_max) {
				free_mask = free_mask || (b.array() < opt.b_max->array());
			}
		}

		auto Hv = P.make_Hv(w);
		VectorXd d = VectorXd::Zero(K), r = VectorXd::Zero(K);
		VectorXd z = VectorXd::Zero(K), p = VectorXd::Zero(K);

		double gF2 = 0.0;
		for (int k = 0; k < K; ++k) {
			if (free_mask[k]) {
				r[k] = g[k];
				z[k] = r[k] / q[k];
				p[k] = -z[k];
				gF2 += g[k]*g[k];
			}
		}
		double rz = r.dot(z);
		const double cg_tol = std::max(opt.eps, opt.tron.cg_tol * std::sqrt(std::max(0.0, rz)));
		const int cg_maxit = (opt.tron.cg_max_iter > 0) ? opt.tron.cg_max_iter : (2 * K);

		bool hit_boundary = false;
		for (int itcg = 0; itcg < cg_maxit; ++itcg) {
			if (std::sqrt(std::max(0.0, rz)) <= cg_tol) break;
			VectorXd p_masked = VectorXd::Zero(K);
			for (int j = 0; j < K; ++j) if (free_mask[j]) p_masked[j] = p[j];
			VectorXd Hp = Hv(p_masked);
			for (int j = 0; j < K; ++j) if (!free_mask[j]) Hp[j] = 0.0;

			const double pHp = p.dot(Hp);
			if (pHp <= 0.0) {
				const double p2 = p.squaredNorm(), dTp = d.dot(p);
				const double tau = (-dTp + std::sqrt(std::max(0.0, dTp*dTp + p2*(delta*delta - d.squaredNorm()))))
									/ std::max(1e-32, p2);
				d += tau * p; hit_boundary = true; break;
			}
			const double alpha_cg = rz / std::max(1e-32, pHp);
			VectorXd d_next = d + alpha_cg * p;
			if (d_next.norm() >= delta) {
				const double p2 = p.squaredNorm(), dTp = d.dot(p), rad2 = delta*delta;
				const double under = dTp*dTp + p2*(rad2 - d.squaredNorm());
				const double tau   = (-dTp + std::sqrt(std::max(0.0, under))) / std::max(1e-32, p2);
				d += tau * p; hit_boundary = true; break;
			}
			d = std::move(d_next);
			r += alpha_cg * Hp;
			if (r.norm() <= cg_tol) break;
			VectorXd z_next = VectorXd::Zero(K);
			for (int j = 0; j < K; ++j) if (free_mask[j]) z_next[j] = r[j] / q[j];
			const double rz_next = r.dot(z_next);
			const double beta = rz_next / std::max(1e-32, rz);
			p = -z_next + beta * p;
			z.swap(z_next);
			rz = rz_next;
		}

		VectorXd b_trial = d;
		if (rho_t > 0) {b_trial *= rho_t;}
		b_trial += b;

		project_to_box(b_trial, opt);
		VectorXd d_proj  = b_trial - b;

		VectorXd Hdproj = Hv(d_proj);
		const double m_pred   = g.dot(d_proj) + 0.5 * d_proj.dot(Hdproj);
		const double pred_red = -m_pred;
		const double f_new = P.f(b_trial);
		const double act_red = f_cur - f_new; // actual reduction
		const double rho = (pred_red > 0.0) ? (act_red / pred_red) : -std::numeric_limits<double>::infinity();

		if (rho < 0.25) delta *= 0.25;
		else if (rho > 0.75 && hit_boundary) delta = std::min(2.0 * delta, opt.tron.delta_max);

		f_diff = std::abs(f_new - f_cur);
		f_rel = f_diff / (f_new + f_cur) * 2;
		b_diff = d_proj.norm();
		b_rel = b_diff / std::max(1.0, b.norm());

		if (rho > opt.tron.eta) { b.swap(b_trial); f_cur = f_new; }

		if (debug_ > 0) {
			std::cout << "TRON obj=" << f_cur << " delta=" << delta << " |db|/|b|=" << b_rel << std::endl;
		}

		if (b_rel <= opt.tol) break;
		if (std::sqrt(gF2) <= opt.tol) break;
		if (delta < 1e-12) break;
	}
	if (delta_new) {
		*delta_new = delta;
	}
	stats.niters = it + 1;
	stats.obj = f_cur;
	stats.diff_obj = f_diff;
	stats.rel_diff_b = b_rel;
	stats.diff_b = b_diff;
	return f_cur;
}

// ---------------- Monotone FISTA / ACG ----------------
template <class Problem>
double acg_solve(const Problem& P, VectorXd& b, const OptimOptions& opt, OptimStats& stats, int debug_ = 0) {
	const int K = static_cast<int>(b.size());
	VectorXd b_prev = b;
	VectorXd yk = b;
	VectorXd gk(K);
	double t = 1.0;
	double L = std::max(1e-12, opt.acg.L0);

	double f_cur = P.f(b);
	int obj_stable = 0;
	double f_diff, b_diff, f_rel, b_rel;
	int it = 0;
	for (; it < opt.max_iters; ++it) {
		double f_yk = 0.0;
		P.eval(yk, &f_yk, &gk, nullptr, nullptr);
		// Backtracking from yk
		VectorXd b_trial, diff;
		double f_new = 0.0;
		while (true) {
			b_trial = (yk - (1.0 / L) * gk);
			project_to_box(b_trial, opt);
			diff = b_trial - yk;
			const double rhs = f_yk + gk.dot(diff) + 0.5 * L * diff.squaredNorm();
			f_new = P.f(b_trial);
			if (f_new <= rhs) break;
			L *= opt.acg.bt_inc;
		}
		// Monotone safeguard: fallback from b if needed
		if (opt.acg.monotone && f_new > f_cur) {
			P.grad(b, gk);
			const double f_b = f_cur;
			while (true) {
				b_trial = (b - (1.0 / L) * gk);
				project_to_box(b_trial, opt);
				diff = b_trial - b;
				const double rhs = f_b + gk.dot(diff) + 0.5 * L * diff.squaredNorm();
				f_new = P.f(b_trial);
				if (f_new <= rhs) break;
				L *= opt.acg.bt_inc;
			}
			t = 1.0;
			yk = b;
		}
		// Accept
		b_prev.swap(b);
		b.swap(b_trial);
		// Nesterov momentum
		const double t_next = 0.5 * (1.0 + std::sqrt(1.0 + 4.0 * t * t));
		VectorXd y_next = b + ((t - 1.0) / t_next) * (b - b_prev);

		if (opt.acg.restart) {
			if ((b - b_prev).dot(y_next - b) > 0.0) {
				t = 1.0;
				yk = b;
			} else {
				t = t_next;
				yk.swap(y_next);
			}
		} else {
			t = t_next;
			yk.swap(y_next);
		}

		f_diff = std::abs(f_new - f_cur);
		f_rel = f_diff / (f_new + f_cur) * 2;
		b_diff = (b - b_prev).norm();
		b_rel = b_diff / std::max(1.0, b_prev.norm());
		if (debug_ > 0) {
			std::cout << "ACG obj=" << f_new << " L=" << L << " |db|/|b|=" << b_rel << std::endl;
		}

		f_cur = f_new;
		if (f_rel < opt.tol) {
			if (++obj_stable >= 3) { break; }
		} else {
			obj_stable = 0;
		}
		if (b_rel < opt.tol) { break; }
	}
	stats.niters = it + 1;
	stats.obj = f_cur;
	stats.diff_obj = f_diff;
	stats.rel_diff_b = b_rel;
	stats.diff_b = b_diff;
	return f_cur;
}

// ---------------- Diagonal Newton + optional line search ----------------
template <class Problem>
double newton_solve(const Problem& P, VectorXd& b, const OptimOptions& opt, OptimStats& stats, int debug_ = 0) {
	double f_cur = P.f(b);
	VectorXd g(b.size()), q(b.size());

	int obj_stable = 0;
	int it = 0;
	double f_diff, b_diff, f_rel, b_rel;
	for (; it < opt.max_iters; ++it) {
		P.eval(b, nullptr, &g, &q, nullptr);
		const VectorXd dir = (-g.array() / q.array()).matrix();
		double alpha = opt.alpha;
		VectorXd b_trial = (b + alpha * dir);
		project_to_box(b_trial, opt);
		double f_new = P.f(b_trial);
		if (opt.ls.enabled) {
			int bt = 0;
			for (;; ++bt) {
			const double armijo_rhs = f_cur + opt.ls.c1 * g.dot(b_trial - b);
			if (f_new <= armijo_rhs || bt >= opt.ls.max_backtracks) break;
			alpha *= opt.ls.beta;
			b_trial = (b + alpha * dir);
			project_to_box(b_trial, opt);
			f_new = P.f(b_trial);
			}
		}
		f_diff = std::abs(f_new - f_cur);
		f_rel = f_diff / (f_new + f_cur) * 2;
		b_diff = (b_trial - b).norm();
		b_rel = b_diff / std::max(1.0, b.norm());

		b.swap(b_trial);
		if (debug_ > 0) {
			std::cout << "GD(LS) obj=" << f_new << " alpha=" << alpha << " |db|/|b|=" << b_rel << std::endl;
		}

		f_cur = f_new;
		if (f_rel < opt.tol) {
			if (++obj_stable >= 3) { ; break; }
		} else {
			obj_stable = 0;
		}
		if (b_rel < opt.tol) { break; }
	}
	stats.niters = it + 1;
	stats.obj = f_cur;
	stats.diff_obj = f_diff;
	stats.rel_diff_b = b_rel;
	stats.diff_b = b_diff;
	return f_cur;
}
