#include "img_utils.hpp"
#include "utils.h" // for compute_percentile
#include <opencv2/opencv.hpp>
#include <vector>
#include <cmath>
#include <algorithm>

// Helper function to compute percentile of non-zero values in a cv::Mat
void percentile(std::vector<uchar>& results, const cv::Mat& mat, std::vector<double>& percentiles) {
    std::vector<uchar> values;
    values.reserve(mat.rows * mat.cols);
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            uchar val = mat.at<uchar>(i, j);
            if (val > 0) {
                values.push_back(val);
            }
        }
    }
    compute_percentile<uchar>(results, values, percentiles);
}

cv::Point2d centroidOfPolygonTriangulation(const std::vector<cv::Point2d>& poly) {
    // For a polygon with fewer than 3 points, return the first point.
    if (poly.size() < 3)
        return poly.front();

    cv::Point2d ref = poly[0];
    double totalArea = 0.0;
    cv::Point2d centroidSum(0.0, 0.0);
    // Triangulate the polygon by decomposing it into triangles (ref, poly[i], poly[i+1]).
    for (size_t i = 1; i < poly.size() - 1; i++) {
        // Compute cross product to get twice the triangle area.
        double cross = (poly[i].x - ref.x) * (poly[i+1].y - ref.y) - (poly[i].y - ref.y) * (poly[i+1].x - ref.x);
        double triArea = cross / 2.0;
        totalArea += triArea;
        // The centroid of a triangle is the average of its vertices.
        cv::Point2d triCentroid = (ref + poly[i] + poly[i+1]) / 3.0;
        centroidSum += triArea * triCentroid;
    }
    // In case of near-degeneracy, fall back to the reference.
    if (std::fabs(totalArea) < 1e-6)
        return ref;

    return centroidSum / totalArea;
}

cv::Point2f centroidOfPolygonRobust(const std::vector<cv::Point2f>& polyf) {
    std::vector<cv::Point2d> poly;
    for (const auto& p : polyf)
        poly.push_back(cv::Point2d(p.x, p.y));
    cv::Point2d c = centroidOfPolygonTriangulation(poly);
    return cv::Point2f(static_cast<float>(c.x), static_cast<float>(c.y));
}

std::vector<cv::Point2f> clipPolygonToRect(const std::vector<cv::Point2f>& poly, const cv::Rect2f& rect) {
    // We assume poly is already in a proper (clockwise) order.
    std::vector<cv::Point2f> output = poly;

    // Define bounds.
    float xmin = rect.x;
    float ymin = rect.y;
    float xmax = rect.x + rect.width;
    float ymax = rect.y + rect.height;

    // Local lambda to clip against one edge.
    // 'inside' returns true if a point is on the interior side of the edge.
    // 'computeIntersection' computes the intersection point of the subject edge and the clipping edge.
    auto clipEdge = [&](auto inside, auto computeIntersection, const std::string &edgeName) {
        std::vector<cv::Point2f> input = output;
        output.clear();
        if (input.empty()) return;
        cv::Point2f S = input.back();
        for (const auto &P : input) {
            bool sInside = inside(S);
            bool pInside = inside(P);
            if (pInside) {
                if (!sInside) {
                    cv::Point2f ip = computeIntersection(S, P);
                    output.push_back(ip);
                }
                output.push_back(P);
            }
            else if (sInside) {
                cv::Point2f ip = computeIntersection(S, P);
                output.push_back(ip);
            }
            S = P;
        }
    };

    // Process edges in the proper clockwise order.
    // Top edge: from (xmin, ymin) to (xmax, ymin)
    clipEdge(
        [&](const cv::Point2f &p) { return p.y >= ymin; },
        [&](const cv::Point2f &S, const cv::Point2f &P) -> cv::Point2f {
            float t = (ymin - S.y) / (P.y - S.y);
            return cv::Point2f(S.x + t * (P.x - S.x), ymin);
        },
        "Top"
    );

    // Right edge: from (xmax, ymin) to (xmax, ymax)
    clipEdge(
        [&](const cv::Point2f &p) { return p.x <= xmax; },
        [&](const cv::Point2f &S, const cv::Point2f &P) -> cv::Point2f {
            float t = (xmax - S.x) / (P.x - S.x);
            return cv::Point2f(xmax, S.y + t * (P.y - S.y));
        },
        "Right"
    );

    // Bottom edge: from (xmax, ymax) to (xmin, ymax)
    clipEdge(
        [&](const cv::Point2f &p) { return p.y <= ymax; },
        [&](const cv::Point2f &S, const cv::Point2f &P) -> cv::Point2f {
            float t = (ymax - S.y) / (P.y - S.y);
            return cv::Point2f(S.x + t * (P.x - S.x), ymax);
        },
        "Bottom"
    );

    // Left edge: from (xmin, ymax) to (xmin, ymin)
    clipEdge(
        [&](const cv::Point2f &p) { return p.x >= xmin; },
        [&](const cv::Point2f &S, const cv::Point2f &P) -> cv::Point2f {
            float t = (xmin - S.x) / (P.x - S.x);
            return cv::Point2f(xmin, S.y + t * (P.y - S.y));
        },
        "Left"
    );
    return output;
}
