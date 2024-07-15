#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
#include <unordered_map>

typedef std::unordered_map<std::pair<int32_t, int32_t>, std::vector<uint32_t> > hex2idx_t;

void cart_to_hex(std::vector<int32_t>& hx, std::vector<int32_t>& hy,
                  const std::vector<double>& x, const std::vector<double>& y,
                  double size, double offset_x = 0, double offset_y = 0) {
    size_t n = x.size();
    assert(x.size() == y.size());
    hx.resize(n);
    hy.resize(n);

    double mtx[2][2] = {{std::sqrt(3)/3, -1/3}, {0, 2/3}};
    std::vector<std::vector<double>> hex_frac(2, std::vector<double>(n, 0));
    for(int i = 0; i < n; ++i) {
        hex_frac[0][i] = (mtx[0][0] * x[i] + mtx[0][1] * y[i]) / size + offset_x;
        hex_frac[1][i] = (mtx[1][0] * x[i] + mtx[1][1] * y[i]) / size + offset_y;
    }

    std::vector<int32_t> rx(n), ry(n);
    for (size_t i = 0; i < n; ++i) {
        rx[i] = std::round(hex_frac[0][i]);
        ry[i] = std::round(hex_frac[1][i]);
        hex_frac[0][i] -= rx[i];
        hex_frac[1][i] -= ry[i];
    }

    for (size_t i = 0; i < n; ++i) {
        if(std::abs(hex_frac[0][i]) < std::abs(hex_frac[1][i])) {
            hx[i] = rx[i];
            hy[i] = ry[i] + std::round(hex_frac[1][i] + 0.5 * hex_frac[0][i]);
        } else {
            hx[i] = rx[i] + std::round(hex_frac[0][i] + 0.5 * hex_frac[1][i]);
            hy[i] = ry[i];
        }
    }

}

void hex_to_cart(std::vector<double>& x, std::vector<double>& y,
                  const std::vector<int32_t>& hx, const std::vector<int32_t>& hy, double size, double offset_x = 0, double offset_y = 0) {
    size_t n = hx.size();
    assert(hy.size() == n);
    x.resize(n);
    y.resize(n);
    for (uint32_t i = 0; i < n; ++i) {
        x[i] = size * (std::sqrt(3) * (hx[i] - offset_x) + std::sqrt(3)/2 * (hy[i] - offset_y));
        y[i] = size * 3/2 * (hy[i] - offset_y);
    }
}

void group_by_hex(hex2idx_t& hexcrd_to_idx,
                  const std::vector<double>& x, const std::vector<double>& y,
                  double size, double offset_x = 0, double offset_y = 0) {
    size_t n = x.size();
    assert(y.size() == n);
    hexcrd_to_idx.clear();
    std::vector<int32_t> hx, hy;
    cart_to_hex(hx, hy, x, y, size, offset_x, offset_y);
    for (size_t i = 0; i < n; ++i) {
        hexcrd_to_idx[std::make_pair(hx[i], hy[i])].push_back(i);
    }
}
