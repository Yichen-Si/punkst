#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>
#include <opencv2/opencv.hpp>
#include <limits>
#include <cmath>
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

// Image related

// compute percentiles of non-zero values in a cv::Mat
void percentile(std::vector<uchar>& results, const cv::Mat& mat, std::vector<double>& percentiles);

// Shape related

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
            return 0; // no intersection
        }
        if (other.xmin <= xmin && other.xmax >= xmax && other.ymin <= ymin && other.ymax >= ymax) {
            return 2; // the other rectangle fully contains this one
        }
        if (other.xmin >= xmin && other.xmax <= xmax && other.ymin >= ymin && other.ymax <= ymax) {
            return 3; // this rectangle fully contains the other one
        }
        return 1; // partial intersection
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

template <typename T>
int32_t parseCoordsToRects(std::vector<Rectangle<T>>& rects, const std::vector<T>& coords) {
    if (coords.size() % 4 != 0) {
        return -1;
    }
    for (size_t i = 0; i < coords.size() / 4; ++i) {
        T x1 = coords[i * 4];     // xmin
        T y1 = coords[i * 4 + 1]; // ymin
        T x2 = coords[i * 4 + 2]; // xmax
        T y2 = coords[i * 4 + 3]; // ymax
        Rectangle<T> rect(x1, y1, x2, y2);
        if (!rect.proper()) {
            warning("Invalid bounding box: %f %f %f %f", x1, y1, x2, y2);
            continue;
        }
        rects.push_back(rect);
    }
    return rects.size();
}

// Centroid of a polygon by triangulation and weighted average
cv::Point2d centroidOfPolygonTriangulation(const std::vector<cv::Point2d>& poly);
cv::Point2f centroidOfPolygonRobust(const std::vector<cv::Point2f>& polyf);

// Centroid of a polygon using the shoelace formula
template <typename T>
cv::Point_<T> centroidOfPolygon(const std::vector<cv::Point_<T>>& poly) {
    assert(!poly.empty());
    double area = 0.0, cx = 0.0, cy = 0.0;
    uint32_t n = poly.size();
    // Compute the polygon centroid using the shoelace formula.
    for (size_t j = 0; j < poly.size(); j++) {
        const cv::Point_<T>& p0 = poly[j];
        const cv::Point_<T>& p1 = poly[(j + 1) % n];
        double cross = (double) p0.x * p1.y - (double) p1.x * p0.y;
        area += cross;
        cx += (p0.x + p1.x) * cross;
        cy += (p0.y + p1.y) * cross;
    }
    area *= 0.5;
    if (std::fabs(area) > 1e-6) {
        cx /= (6 * area);
        cy /= (6 * area);
        return cv::Point_<T>(static_cast<T>(cx), static_cast<T>(cy));
    }
    return poly[0];
}

// Sutherland–Hodgman polygon clipping algorithm
std::vector<cv::Point2f> clipPolygonToRect(const std::vector<cv::Point2f>& poly, const cv::Rect2f& rect);
