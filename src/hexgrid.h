#pragma once

#include <iostream>
#include <cstdint>
#include <vector>
#include <cmath>
#include <cassert>
#include <unordered_map>

/**
 * Based on https://www.redblobgames.com/grids/hexagons/
 * and https://observablehq.com/@jrus/hexround
 */
class HexGrid {

public:

    typedef std::unordered_map<uint64_t, std::vector<uint32_t> > hex2idx_t;

    double size;
    bool pointy = true;
    double mtx_p2a[2][2];
    double mtx_a2p[2][2];

    HexGrid() {}
    HexGrid(double s, bool _pointy = true) {
        init(s, _pointy);
    }
    void init(double s, bool _pointy = true) {
        assert(s > 0);
        size = s;
        pointy = _pointy;

        if (pointy) {
            mtx_p2a[0][0] = std::sqrt(3.)/3/size;
            mtx_p2a[0][1] = -1./3./size;
            mtx_p2a[1][0] = 0.;
            mtx_p2a[1][1] = 2./3./size;

            mtx_a2p[0][0] = size * std::sqrt(3.);
            mtx_a2p[0][1] = size * std::sqrt(3.)/2;
            mtx_a2p[1][0] = 0.;
            mtx_a2p[1][1] = size * 3./2;
        } else {
            mtx_p2a[0][0] = 2./3/size;
            mtx_p2a[0][1] = 0.;
            mtx_p2a[1][0] = -1./3/size;
            mtx_p2a[1][1] = std::sqrt(3.)/3/size;

            mtx_a2p[0][0] = size * 3./2;
            mtx_a2p[0][1] = 0.;
            mtx_a2p[1][0] = size * std::sqrt(3.)/2;
            mtx_a2p[1][1] = size * std::sqrt(3.);
        }
    }

    template <typename T>
    void cart_to_axial(std::vector<int32_t>& hx, std::vector<int32_t>& hy,
                      const std::vector<T>& x, const std::vector<T>& y,
                      double offset_x = 0, double offset_y = 0) const {
        size_t n = x.size();
        assert(x.size() == y.size());
        hx.resize(n);
        hy.resize(n);

        std::vector<std::vector<double>> hex_frac(2, std::vector<double>(n, 0));
        for(int i = 0; i < n; ++i) {
            hex_frac[0][i] = mtx_p2a[0][0] * x[i] + mtx_p2a[0][1] * y[i] + offset_x;
            hex_frac[1][i] = mtx_p2a[1][1] * y[i] + offset_y;
        }

        for (size_t i = 0; i < n; ++i) {
            hx[i] = std::lround(hex_frac[0][i]);
            hy[i] = std::lround(hex_frac[1][i]);
            hex_frac[0][i] -= hx[i];
            hex_frac[1][i] -= hy[i];
        }

        for (size_t i = 0; i < n; ++i) {
            if(std::abs(hex_frac[0][i]) < std::abs(hex_frac[1][i])) {
                hy[i] += std::lround(hex_frac[1][i] + 0.5 * hex_frac[0][i]);
            } else {
                hx[i] += std::lround(hex_frac[0][i] + 0.5 * hex_frac[1][i]);
            }
        }

    }

    template <typename T>
    void cart_to_axial(int32_t& hx, int32_t& hy, T x, T y,
                      double offset_x = 0, double offset_y = 0) const {
        double hx_f = mtx_p2a[0][0] * x + mtx_p2a[0][1] * y + offset_x;
        double hy_f = mtx_p2a[1][1] * y + offset_y;
            hx = std::lround(hx_f);
            hy = std::lround(hy_f);
            hx_f -= hx;
            hy_f -= hy;

            if(std::abs(hx_f) < std::abs(hy_f)) {
                hy += std::lround(hy_f + 0.5 * hx_f);
            } else {
                hx += std::lround(hx_f + 0.5 * hy_f);
            }
    }

    template <typename T>
    void axial_to_cart(std::vector<T>& x, std::vector<T>& y,
                      const std::vector<int32_t>& hx, const std::vector<int32_t>& hy, double offset_x = 0, double offset_y = 0) const {
        size_t n = hx.size();
        assert(hy.size() == n);
        x.resize(n);
        y.resize(n);
        for (uint32_t i = 0; i < n; ++i) {
            x[i] = (mtx_a2p[0][0] * (hx[i] - offset_x) + mtx_a2p[0][1] * (hy[i] - offset_y));
            y[i] = mtx_a2p[1][1] * (hy[i] - offset_y);
        }
    }

    template <typename T>
    void axial_to_cart(T& x, T& y, int32_t hx, int32_t hy, double offset_x = 0, double offset_y = 0) const {
        x = mtx_a2p[0][0] * (hx - offset_x) + mtx_a2p[0][1] * (hy - offset_y);
        y = mtx_a2p[1][1] * (hy - offset_y);
    }

    void hex_bounding_box_axial(double& xmin, double& xmax, double& ymin, double& ymax, int32_t hx, int32_t hy, double offset_x = 0, double offset_y = 0) const {
        double xcenter, ycenter;
        axial_to_cart(xcenter, ycenter, hx, hy, offset_x, offset_y);
        if(pointy) {
            xmin = xcenter - size * std::sqrt(3) / 2;
            xmax = xcenter + size * std::sqrt(3) / 2;
            ymin = ycenter - size;
            ymax = ycenter + size;
        } else {
            xmin = xcenter - size;
            xmax = xcenter + size;
            ymin = ycenter - size * std::sqrt(3) / 2;
            ymax = ycenter + size * std::sqrt(3) / 2;
        }
    }

    void group_by_hex_axial(hex2idx_t& hexcrd_to_idx,
                      const std::vector<double>& x, const std::vector<double>& y, double offset_x = 0, double offset_y = 0) const {
        size_t n = x.size();
        assert(y.size() == n);
        hexcrd_to_idx.clear();
        std::vector<int32_t> hx, hy;
        cart_to_axial(hx, hy, x, y, offset_x, offset_y);
        for (size_t i = 0; i < n; ++i) {
            uint64_t key = (static_cast<uint64_t>(hx[i]) << 32) | (static_cast<uint32_t>(hy[i]));
            hexcrd_to_idx[key].push_back(i);
        }
    }

};


// generatelattice points within a bounding box
template <typename T>
void hex_grid_cart(std::vector<std::vector<T> >& lattice, T xmin, T xmax, T ymin, T ymax, double size) {
    double spacing_x = size * std::sqrt(3.);
    double spacing_y = size * 1.5;
    int nRows = static_cast<int>(std::ceil((ymax - ymin) / spacing_y)) + 1;
    int nCols = static_cast<int>(std::floor((xmax - xmin) / spacing_x)) + 1;
    lattice.reserve(nRows * nCols);
    // generate by row
    for (int row = 0; row < nRows; ++row) {
        T y_coord = ymin + row * spacing_y;
        if (y_coord > ymax)
            break;
        // For pointy hexagons, odd-numbered rows get an x offset.
        double row_offset = (row % 2) ? (spacing_x * 0.5) : 0.0;
        // Add points along the row.
        int col = 0;
        while (true) {
            T x_coord = xmin + row_offset + col * spacing_x;
            if (x_coord > xmax)
                break;
            lattice.push_back({ x_coord, y_coord });
            ++col;
        }
    }
}
