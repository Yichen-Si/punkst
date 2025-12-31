#pragma once

#include "utils.h"
#include <cmath>
#include <unordered_map>

/**
 * Coordinate transformation for a body-centered cubic (BCC) lattice.
 * The Voronoi cell for this lattice is a truncated octahedron
 *
 * The lattice is defined by the basis vectors:
 * B = (s/2) * [ [ 1,  1,  1], [ 1, -1,  1], [ 1,  1, -1] ]
 * det(B) = s³/2, neighbor distance = s * sqrt(3)/2
 * Any lattice point is given by
 *   p = B * q   where q = (q₁, q₂, q₃) ∈ ℤ³.
 *
 * Conversion from Cartesian coordinates to lattice coordinates
 * is performed by computing fractional coordinates:
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
    BCCGrid(double s) {init(s);}

    void init(double s) {
        assert(s > 0);
        size = s;
        // Define the basis vectors for the BCC lattice:
        // The lattice-to-Cartesian transformation matrix (B)
        mtx_l2p[0][0] = s/2;  mtx_l2p[0][1] = s/2;  mtx_l2p[0][2] = s/2;
        mtx_l2p[1][0] = s/2;  mtx_l2p[1][1] = -s/2; mtx_l2p[1][2] = s/2;
        mtx_l2p[2][0] = s/2;  mtx_l2p[2][1] = s/2;  mtx_l2p[2][2] = -s/2;
        // The inverse transformation (Cartesian-to-lattice) is derived so that:
        mtx_p2l[0][0] = 0.0;   mtx_p2l[0][1] = 1.0/s; mtx_p2l[0][2] = 1.0/s;
        mtx_p2l[1][0] = 1.0/s; mtx_p2l[1][1] = -1.0/s; mtx_p2l[1][2] = 0.0;
        mtx_p2l[2][0] = 1.0/s; mtx_p2l[2][1] = 0.0;   mtx_p2l[2][2] = -1.0/s;
    }

    // Converts a Cartesian point (x,y,z) to lattice coordinates (q₁,q₂,q₃)
    // using the transformation and rounding to the nearest integer.
    void cart_to_lattice(int32_t &q1, int32_t &q2, int32_t &q3,
                         double x, double y, double z,
        double offset_x = 0, double offset_y = 0, double offset_z = 0) const {
        const double s = size;
        double xp = x - offset_x;
        double yp = y - offset_y;
        double zp = z - offset_z;
        // Candidate A: nearest point in s*Z^3
        const long long ax_i = std::llround(xp / s);
        const long long ay_i = std::llround(yp / s);
        const long long az_i = std::llround(zp / s);
        const double ax = s * (double)ax_i;
        const double ay = s * (double)ay_i;
        const double az = s * (double)az_i;
        // Candidate B: nearest point in s*Z^3 + (s/2,s/2,s/2)
        const long long bx_i = std::llround((xp / s) - 0.5);
        const long long by_i = std::llround((yp / s) - 0.5);
        const long long bz_i = std::llround((zp / s) - 0.5);
        const double bx = s * ((double)bx_i + 0.5);
        const double by = s * ((double)by_i + 0.5);
        const double bz = s * ((double)bz_i + 0.5);
        auto d2 = [](double x0,double y0,double z0,
                     double x1,double y1,double z1){
            double dx=x0-x1, dy=y0-y1, dz=z0-z1;
            return dx*dx + dy*dy + dz*dz;
        };
        double nx, ny, nz;
        if (d2(xp,yp,zp, ax,ay,az) <= d2(xp,yp,zp, bx,by,bz)) {
            nx=ax; ny=ay; nz=az;
        } else {
            nx=bx; ny=by; nz=bz;
        }
        // Convert chosen nearest lattice point back to (q1,q2,q3)
        q1 = (int32_t) std::llround((ny + nz) / s);
        q2 = (int32_t) std::llround((nx - ny) / s);
        q3 = (int32_t) std::llround((nx - nz) / s);
    }

    // Vectorized conversion from Cartesian to lattice coordinates.
    void cart_to_lattice(std::vector<int32_t>& q1, std::vector<int32_t>& q2,
        std::vector<int32_t>& q3, const std::vector<double>& x,
        const std::vector<double>& y, const std::vector<double>& z,
        double offset_x = 0, double offset_y = 0, double offset_z = 0) const {
        size_t n = x.size();
        assert(x.size() == y.size() && y.size() == z.size());
        q1.resize(n);
        q2.resize(n);
        q3.resize(n);
        for (size_t i = 0; i < n; ++i) {
            cart_to_lattice(q1[i], q2[i], q3[i], x[i], y[i], z[i],
                            offset_x, offset_y, offset_z);
        }
    }

    // Converts lattice coordinates (q₁,q₂,q₃) to Cartesian coordinates (x,y,z)
    // returning the center of the corresponding cell.
    void lattice_to_cart(double &x, double &y, double &z,
                         int32_t q1, int32_t q2, int32_t q3,
                         double offset_x = 0, double offset_y = 0, double offset_z = 0) const {
        x = mtx_l2p[0][0] * q1 + mtx_l2p[0][1] * q2 + mtx_l2p[0][2] * q3
            + offset_x;
        y = mtx_l2p[1][0] * q1 + mtx_l2p[1][1] * q2 + mtx_l2p[1][2] * q3
            + offset_y;
        z = mtx_l2p[2][0] * q1 + mtx_l2p[2][1] * q2 + mtx_l2p[2][2] * q3
            + offset_z;
    }

    // Vectorized conversion from lattice coordinates to Cartesian coordinates.
    void lattice_to_cart(std::vector<double>& x, std::vector<double>& y, std::vector<double>& z, const std::vector<int32_t>& q1, const std::vector<int32_t>& q2, const std::vector<int32_t>& q3, double offset_x = 0, double offset_y = 0, double offset_z = 0) const {
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
