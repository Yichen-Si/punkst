#pragma once

#include "utils.h"
#include <unordered_map>

/**
 * Coordinate transformation for a body-centered cubic (BCC) lattice.
 * The Voronoi cell for this lattice is a truncated octahedron
 *
 * The lattice is defined by the basis vectors:
 *   b₁ = (s/2,  s/2,  s/2)
 *   b₂ = (s/2, -s/2,  s/2)
 *   b₃ = (s/2,  s/2, -s/2)
 *
 * Any lattice point is given by
 *   p = B * q   where q = (q₁, q₂, q₃) ∈ ℤ³.
 *
 * Conversion from Cartesian coordinates to lattice coordinates
 * is performed by computing fractional coordinates:
 *
 *   q₁ = (y + z)/s,   q₂ = (x - y)/s,   q₃ = (x - z)/s,
 *
 * and then rounding them to the nearest integer. The reverse transformation
 * is done via multiplication by the basis matrix B.
 */
class BCCGrid {

public:
    // A typedef for grouping point indices by the lattice cell.
    typedef std::unordered_map<std::tuple<int32_t, int32_t, int32_t>, std::vector<uint32_t>, Tuple3Hash> bcc2idx_t;

    double size;      // Scaling factor for the lattice cell size.
    double mtx_l2p[3][3]; // Lattice-to-Cartesian transformation matrix.
    double mtx_p2l[3][3]; // Cartesian-to-lattice transformation matrix.

    BCCGrid() {}
    BCCGrid(double s) {
        init(s);
    }

    void init(double s) {
        assert(s > 0);
        size = s;

        // Define the basis vectors for the BCC lattice:
        //   b₁ = (s/2,  s/2,  s/2)
        //   b₂ = (s/2, -s/2,  s/2)
        //   b₃ = (s/2,  s/2, -s/2)
        // The lattice-to-Cartesian transformation matrix (B)
        mtx_l2p[0][0] = s/2;  mtx_l2p[0][1] = s/2;  mtx_l2p[0][2] = s/2;
        mtx_l2p[1][0] = s/2;  mtx_l2p[1][1] = -s/2; mtx_l2p[1][2] = s/2;
        mtx_l2p[2][0] = s/2;  mtx_l2p[2][1] = s/2;  mtx_l2p[2][2] = -s/2;

        // The inverse transformation (Cartesian-to-lattice) is derived so that:
        //   q₁ = (y + z) / s,
        //   q₂ = (x - y) / s,
        //   q₃ = (x - z) / s.
        mtx_p2l[0][0] = 0.0;   mtx_p2l[0][1] = 1.0/s; mtx_p2l[0][2] = 1.0/s;
        mtx_p2l[1][0] = 1.0/s; mtx_p2l[1][1] = -1.0/s; mtx_p2l[1][2] = 0.0;
        mtx_p2l[2][0] = 1.0/s; mtx_p2l[2][1] = 0.0;   mtx_p2l[2][2] = -1.0/s;
    }

    // Converts a single Cartesian point (x, y, z) to lattice coordinates (q₁, q₂, q₃)
    // using the transformation and rounding to the nearest integer.
    void cart_to_lattice(int32_t &q1, int32_t &q2, int32_t &q3,
                         double x, double y, double z,
                         double offset_x = 0, double offset_y = 0, double offset_z = 0) const {
        double xp = x - offset_x;
        double yp = y - offset_y;
        double zp = z - offset_z;
        double q1_f = mtx_p2l[0][0] * xp + mtx_p2l[0][1] * yp + mtx_p2l[0][2] * zp;
        double q2_f = mtx_p2l[1][0] * xp + mtx_p2l[1][1] * yp + mtx_p2l[1][2] * zp;
        double q3_f = mtx_p2l[2][0] * xp + mtx_p2l[2][1] * yp + mtx_p2l[2][2] * zp;
        q1 = std::lround(q1_f);
        q2 = std::lround(q2_f);
        q3 = std::lround(q3_f);
    }

    // Vectorized conversion from Cartesian to lattice coordinates.
    void cart_to_lattice(std::vector<int32_t>& q1, std::vector<int32_t>& q2, std::vector<int32_t>& q3,
                         const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
                         double offset_x = 0, double offset_y = 0, double offset_z = 0) const {
        size_t n = x.size();
        assert(x.size() == y.size() && y.size() == z.size());
        q1.resize(n);
        q2.resize(n);
        q3.resize(n);
        for (size_t i = 0; i < n; ++i) {
            double xp = x[i] - offset_x;
            double yp = y[i] - offset_y;
            double zp = z[i] - offset_z;
            double q1_f = mtx_p2l[0][0] * xp + mtx_p2l[0][1] * yp + mtx_p2l[0][2] * zp;
            double q2_f = mtx_p2l[1][0] * xp + mtx_p2l[1][1] * yp + mtx_p2l[1][2] * zp;
            double q3_f = mtx_p2l[2][0] * xp + mtx_p2l[2][1] * yp + mtx_p2l[2][2] * zp;
            q1[i] = std::lround(q1_f);
            q2[i] = std::lround(q2_f);
            q3[i] = std::lround(q3_f);
        }
    }

    // Converts lattice coordinates (q₁, q₂, q₃) to Cartesian coordinates (x, y, z),
    // returning the center of the corresponding cell.
    void lattice_to_cart(double &x, double &y, double &z,
                         int32_t q1, int32_t q2, int32_t q3,
                         double offset_x = 0, double offset_y = 0, double offset_z = 0) const {
        x = mtx_l2p[0][0] * q1 + mtx_l2p[0][1] * q2 + mtx_l2p[0][2] * q3 + offset_x;
        y = mtx_l2p[1][0] * q1 + mtx_l2p[1][1] * q2 + mtx_l2p[1][2] * q3 + offset_y;
        z = mtx_l2p[2][0] * q1 + mtx_l2p[2][1] * q2 + mtx_l2p[2][2] * q3 + offset_z;
    }

    // Vectorized conversion from lattice coordinates to Cartesian coordinates.
    void lattice_to_cart(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z,
                         const std::vector<int32_t>& q1, const std::vector<int32_t>& q2, const std::vector<int32_t>& q3,
                         double offset_x = 0, double offset_y = 0, double offset_z = 0) const {
        size_t n = q1.size();
        assert(q2.size() == n && q3.size() == n);
        x.resize(n);
        y.resize(n);
        z.resize(n);
        for (size_t i = 0; i < n; ++i) {
            x[i] = mtx_l2p[0][0] * q1[i] + mtx_l2p[0][1] * q2[i] + mtx_l2p[0][2] * q3[i] + offset_x;
            y[i] = mtx_l2p[1][0] * q1[i] + mtx_l2p[1][1] * q2[i] + mtx_l2p[1][2] * q3[i] + offset_y;
            z[i] = mtx_l2p[2][0] * q1[i] + mtx_l2p[2][1] * q2[i] + mtx_l2p[2][2] * q3[i] + offset_z;
        }
    }

    // Groups indices of points by the lattice cell in which they fall.
    void group_by_cell(bcc2idx_t& cell_to_idx,
                       const std::vector<double>& x, const std::vector<double>& y, const std::vector<double>& z,
                       double offset_x = 0, double offset_y = 0, double offset_z = 0) const {
        size_t n = x.size();
        assert(x.size() == y.size() && y.size() == z.size());
        cell_to_idx.clear();
        std::vector<int32_t> q1, q2, q3;
        cart_to_lattice(q1, q2, q3, x, y, z, offset_x, offset_y, offset_z);
        for (size_t i = 0; i < n; ++i) {
            std::tuple<int32_t, int32_t, int32_t> key = std::make_tuple(q1[i], q2[i], q3[i]);
            cell_to_idx[key].push_back(i);
        }
    }
};
