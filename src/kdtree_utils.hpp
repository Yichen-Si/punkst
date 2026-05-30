#pragma once

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "nanoflann.hpp"

template <typename T>
struct PointCloud
{
    struct Point
    {
        T x, y, z;
        Point() : x(0), y(0), z(0) {}
        Point(T x, T y) : x(x), y(y), z(0) {}
        Point(T x, T y, T z) : x(x), y(y), z(z) {}
    };

    using coord_t = T;

    std::vector<Point> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

template <typename T>
struct PointCloud2D
{
    struct Point {
        T x, y;
        Point() : x(0), y(0) {}
        Point(T x, T y) : x(x), y(y) {}
    };
    using coord_t = T;
    std::vector<Point> pts;

    inline size_t kdtree_get_point_count() const { return pts.size(); }

    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else
            return pts[idx].y;
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const
    {
        return false;
    }
};

template <typename T>
void generateRandomPointCloudRanges(
    PointCloud<T>& pc, const size_t N, const T max_range_x, const T max_range_y,
    const T max_range_z)
{
    pc.pts.resize(N);
    for (size_t i = 0; i < N; i++)
    {
        pc.pts[i].x = max_range_x * (rand() % 1000) / T(1000);
        pc.pts[i].y = max_range_y * (rand() % 1000) / T(1000);
        pc.pts[i].z = max_range_z * (rand() % 1000) / T(1000);
    }
}

template <typename T>
void generateRandomPointCloud(
    PointCloud<T>& pc, const size_t N, const T max_range = 10)
{
    generateRandomPointCloudRanges(pc, N, max_range, max_range, max_range);
}

inline void dump_mem_usage()
{
    FILE* f = fopen("/proc/self/statm", "rt");
    if (!f) return;
    char   str[300];
    size_t n = fread(str, 1, 200, f);
    str[n]   = 0;
    printf("MEM: %s\n", str);
    fclose(f);
}

template <
    class VectorOfVectorsType, typename num_t = double, int DIM = -1,
    class Distance = nanoflann::metric_L2, typename IndexType = size_t>
struct KDTreeVectorOfVectorsAdaptor
{
    using self_t = KDTreeVectorOfVectorsAdaptor<
        VectorOfVectorsType, num_t, DIM, Distance, IndexType>;
    using metric_t =
        typename Distance::template traits<num_t, self_t>::distance_t;
    using index_t =
        nanoflann::KDTreeSingleIndexAdaptor<metric_t, self_t, DIM, IndexType>;

    index_t* index = nullptr;

    KDTreeVectorOfVectorsAdaptor(
        const size_t /* dimensionality */, const VectorOfVectorsType& mat,
        const int leaf_max_size = 10, const unsigned int n_thread_build = 1)
        : m_data(mat)
    {
        assert(mat.size() != 0 && mat[0].size() != 0);
        const size_t dims = mat[0].size();
        if (DIM > 0 && static_cast<int>(dims) != DIM)
            throw std::runtime_error(
                "Data set dimensionality does not match the 'DIM' template "
                "argument");
        index = new index_t(
            static_cast<int>(dims), *this,
            nanoflann::KDTreeSingleIndexAdaptorParams(
                leaf_max_size, nanoflann::KDTreeSingleIndexAdaptorFlags::None,
                n_thread_build));
    }

    ~KDTreeVectorOfVectorsAdaptor() { delete index; }

    const VectorOfVectorsType& m_data;

    inline void query(
        const num_t* query_point, const size_t num_closest,
        IndexType* out_indices, num_t* out_distances_sq) const
    {
        nanoflann::KNNResultSet<num_t, IndexType> resultSet(num_closest);
        resultSet.init(out_indices, out_distances_sq);
        index->findNeighbors(resultSet, query_point);
    }

    const self_t& derived() const { return *this; }
    self_t&       derived() { return *this; }

    inline size_t kdtree_get_point_count() const { return m_data.size(); }

    inline num_t kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        return m_data[idx][dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /*bb*/) const
    {
        return false;
    }

};

using kd_tree_i2_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<int32_t, PointCloud<int32_t>>,
        PointCloud<int32_t>, 2>;
using kd_tree_f2_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>,
        PointCloud<float>, 2>;
using kd_tree_d2_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>,
        PointCloud<double>, 2>;
using kd_tree_i3_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<int32_t, PointCloud<int32_t>>,
        PointCloud<int32_t>, 3>;
using kd_tree_f3_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<float, PointCloud<float>>,
        PointCloud<float>, 3>;
using kd_tree_d3_t = nanoflann::KDTreeSingleIndexAdaptor<
        nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>,
        PointCloud<double>, 3>;
