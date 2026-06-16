#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cmath>
#include <limits>
#include <vector>

#include "error.hpp"

namespace island_smoothing {

struct Options {
    bool fillEmpty = false;
    bool updateProbFromWinnerMin = false;
};

struct Result {
    int32_t roundsRun = 0;
    bool converged = false;
};

namespace detail {

struct NeighborVotes {
    uint8_t m = 0;
    int32_t labels[8];
    uint8_t counts[8];
    float minProbs[8];

    void add(int32_t label, float prob) {
        for (uint8_t i = 0; i < m; ++i) {
            if (labels[i] == label) {
                counts[i] += 1;
                if (prob < minProbs[i]) {
                    minProbs[i] = prob;
                }
                return;
            }
        }
        if (m < 8) {
            labels[m] = label;
            counts[m] = 1;
            minProbs[m] = prob;
            ++m;
        }
    }

    int bestIndex() const {
        int bestIdx = -1;
        int bestCnt = 0;
        for (uint8_t i = 0; i < m; ++i) {
            if (counts[i] > bestCnt) {
                bestCnt = counts[i];
                bestIdx = static_cast<int>(i);
            }
        }
        return bestIdx;
    }
};

} // namespace detail

inline Result smoothLabels8Neighborhood(std::vector<int32_t>& labels, std::vector<float>* probs,
    size_t width, size_t height, int32_t rounds, const Options& opts = Options()) {
    Result result;
    const size_t nCells = width * height;
    if (rounds <= 0 || nCells == 0 || labels.size() != nCells) {
        return result;
    }

    std::vector<int32_t> nextLabels = labels;
    std::vector<float> nextProbs;
    const bool updateProb = opts.updateProbFromWinnerMin && probs != nullptr && probs->size() == nCells;
    if (updateProb) {
        nextProbs = *probs;
    }
    for (int32_t round = 0; round < rounds; ++round) {
        nextLabels = labels;
        if (updateProb) {
            nextProbs = *probs;
        }
        bool changed = false;
        for (size_t y = 0; y < height; ++y) {
            for (size_t x = 0; x < width; ++x) {
                const size_t idx = y * width + x;
                const int32_t center = labels[idx];
                if (center < 0 && !opts.fillEmpty) {
                    continue;
                }
                int32_t sameNbr = 0;
                detail::NeighborVotes votes;
                for (int32_t dy = -1; dy <= 1; ++dy) {
                    const int64_t yy = static_cast<int64_t>(y) + dy;
                    if (yy < 0 || yy >= static_cast<int64_t>(height)) {
                        continue;
                    }
                    for (int32_t dx = -1; dx <= 1; ++dx) {
                        if (dx == 0 && dy == 0) {
                            continue;
                        }
                        const int64_t xx = static_cast<int64_t>(x) + dx;
                        if (xx < 0 || xx >= static_cast<int64_t>(width)) {
                            continue;
                        }
                        const size_t nidx = static_cast<size_t>(yy) * width + static_cast<size_t>(xx);
                        const int32_t nb = labels[nidx];
                        if (nb < 0) {
                            continue;
                        }
                        if (nb == center) {
                            sameNbr += 1;
                        }
                        float p = 0.0f;
                        if (updateProb) {
                            p = (*probs)[nidx];
                        }
                        votes.add(nb, p);
                    }
                }
                const int bestIdx = votes.bestIndex();
                if (bestIdx < 0) {
                    continue;
                }
                const int32_t bestLbl = votes.labels[bestIdx];
                const int32_t bestCnt = votes.counts[bestIdx];
                if (sameNbr <= 1 && (bestCnt - sameNbr) > 3 && bestLbl != center && bestLbl != -1) {
                    nextLabels[idx] = bestLbl;
                    if (updateProb) {
                        nextProbs[idx] = votes.minProbs[bestIdx];
                    }
                    changed = true;
                }
            }
        }

        labels.swap(nextLabels);
        if (updateProb) {
            probs->swap(nextProbs);
        }
        result.roundsRun = round + 1;
        if (!changed) {
            result.converged = true;
            break;
        }
    }
    return result;
}

} // namespace island_smoothing

template <typename InT, typename OutT, typename AccT>
inline void boxFilterSum2D(const std::vector<InT>& in, size_t W, size_t H,
    int32_t radius, std::vector<OutT>& out) {
    if (in.size() != W * H) {
        error("%s: Input size mismatch", __func__);
    }
    if (W == 0 || H == 0) return;
    if (radius <= 0) {
        out.assign(in.begin(), in.end());
        return;
    }

    out.resize(in.size());
    const size_t r = static_cast<size_t>(radius);
    std::vector<OutT> tmp(in.size());
    std::vector<AccT> prefix(std::max(W, H) + 1, static_cast<AccT>(0));

    for (size_t y = 0; y < H; ++y) {
        const size_t row = y * W;
        prefix[0] = static_cast<AccT>(0);
        for (size_t x = 0; x < W; ++x) {
            prefix[x + 1] = prefix[x] + static_cast<AccT>(in[row + x]);
        }
        for (size_t x = 0; x < W; ++x) {
            const size_t x0 = (x > r) ? (x - r) : 0;
            const size_t x1 = std::min(W - 1, x + r);
            tmp[row + x] = static_cast<OutT>(prefix[x1 + 1] - prefix[x0]);
        }
    }

    std::vector<AccT> colRunningSum(W, static_cast<AccT>(0));
    for (size_t y = 0; y <= std::min(H - 1, r); ++y) {
        const size_t row = y * W;
        for (size_t x = 0; x < W; ++x) {
            colRunningSum[x] += static_cast<AccT>(tmp[row + x]);
        }
    }

    for (size_t y = 0; y < H; ++y) {
        if (y > 0) {
            const size_t addY = y + r;
            if (addY < H) {
                const size_t addRow = addY * W;
                for (size_t x = 0; x < W; ++x) {
                    colRunningSum[x] += static_cast<AccT>(tmp[addRow + x]);
                }
            }
        }
        if (y > r) {
            const size_t subRow = (y - r - 1) * W;
            for (size_t x = 0; x < W; ++x) {
                colRunningSum[x] -= static_cast<AccT>(tmp[subRow + x]);
            }
        }
        const size_t row = y * W;
        for (size_t x = 0; x < W; ++x) {
            out[row + x] = static_cast<OutT>(colRunningSum[x]);
        }
    }
}

template <typename T>
struct Rectangle {
    T xmin, ymin, xmax, ymax;
    Rectangle(T x1, T y1, T x2, T y2) : xmin(x1), ymin(y1), xmax(x2), ymax(y2) {}
    Rectangle() {reset();}
    void reset() {
        xmin = std::numeric_limits<T>::max();
        xmax = std::numeric_limits<T>::lowest();
        ymin = std::numeric_limits<T>::max();
        ymax = std::numeric_limits<T>::lowest();
    }
    bool proper() const {
        return (xmin < xmax && ymin < ymax);
    }
    bool contains(T x, T y) const {
        return (x >= xmin && x < xmax && y >= ymin && y < ymax);
    }
    void extendToInclude(T x, T y) {
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
    }
    template <typename U>
    void extendToInclude(const Rectangle<U>& other) {
        if (other.xmin < xmin) xmin = other.xmin;
        if (other.xmax > xmax) xmax = other.xmax;
        if (other.ymin < ymin) ymin = other.ymin;
        if (other.ymax > ymax) ymax = other.ymax;
    }
    template <typename U>
    int32_t intersect(const Rectangle<U>& other) const {
        if (other.xmin >= xmax || other.xmax <= xmin || other.ymin >= ymax || other.ymax <= ymin) {
            return 0;
        }
        if (other.xmin <= xmin && other.xmax >= xmax && other.ymin <= ymin && other.ymax >= ymax) {
            return 2;
        }
        if (other.xmin >= xmin && other.xmax <= xmax && other.ymin >= ymin && other.ymax <= ymax) {
            return 3;
        }
        return 1;
    }
    bool cutInside(Rectangle<T>& rec, T r) {
        rec.xmin = xmin + r;
        rec.ymin = ymin + r;
        rec.xmax = xmax - r;
        rec.ymax = ymax - r;
        return rec.proper();
    }
    bool padOutside(Rectangle<T>& rec, T r) {
        rec.xmin = xmin - r;
        rec.ymin = ymin - r;
        rec.xmax = xmax + r;
        rec.ymax = ymax + r;
        return rec.proper();
    }
};

namespace geometry_detail {

inline void hilbert_rotate(uint64_t n, uint64_t& x, uint64_t& y, uint64_t rx, uint64_t ry) {
    if (ry != 0) {
        return;
    }
    if (rx != 0) {
        x = n - 1u - x;
        y = n - 1u - y;
    }
    std::swap(x, y);
}

} // namespace geometry_detail

inline uint64_t hilbert_index_2d(uint32_t xIn, uint32_t yIn, int32_t bits) {
    if (bits <= 0) {
        return 0;
    }
    uint64_t x = xIn;
    uint64_t y = yIn;
    uint64_t d = 0;
    for (uint64_t s = uint64_t{1} << (bits - 1); s > 0; s >>= 1u) {
        const uint64_t rx = (x & s) ? 1u : 0u;
        const uint64_t ry = (y & s) ? 1u : 0u;
        d += s * s * ((3u * rx) ^ ry);
        geometry_detail::hilbert_rotate(s, x, y, rx, ry);
    }
    return d;
}

template <typename T>
int32_t parseCoordsToRects(std::vector<Rectangle<T>>& rects, const std::vector<T>& coords) {
    if (coords.size() % 4 != 0) {
        return -1;
    }
    for (size_t i = 0; i < coords.size() / 4; ++i) {
        T x1 = coords[i * 4];
        T y1 = coords[i * 4 + 1];
        T x2 = coords[i * 4 + 2];
        T y2 = coords[i * 4 + 3];
        Rectangle<T> rect(x1, y1, x2, y2);
        if (!rect.proper()) {
            warning("Invalid bounding box: %f %f %f %f", x1, y1, x2, y2);
            continue;
        }
        rects.push_back(rect);
    }
    return rects.size();
}
