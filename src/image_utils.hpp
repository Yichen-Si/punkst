#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

struct Color3f {
    float r = 0.f;
    float g = 0.f;
    float b = 0.f;

    Color3f() = default;
    Color3f(float r_, float g_, float b_) : r(r_), g(g_), b(b_) {}

    Color3f& operator+=(const Color3f& other) {
        r += other.r;
        g += other.g;
        b += other.b;
        return *this;
    }

    Color3f& operator/=(float value) {
        r /= value;
        g /= value;
        b /= value;
        return *this;
    }
};

inline Color3f operator+(Color3f lhs, const Color3f& rhs) {
    lhs += rhs;
    return lhs;
}

inline Color3f operator*(const Color3f& color, float value) {
    return Color3f(color.r * value, color.g * value, color.b * value);
}

inline Color3f operator/(Color3f color, float value) {
    color /= value;
    return color;
}

struct Rgb8 {
    uint8_t r = 0;
    uint8_t g = 0;
    uint8_t b = 0;
};

inline uint8_t clamp_u8(float value) {
    if (!(value > 0.f)) {
        return 0;
    }
    const long rounded = std::lrint(value);
    if (rounded <= 0) {
        return 0;
    }
    if (rounded >= 255) {
        return 255;
    }
    return static_cast<uint8_t>(rounded);
}

template <typename T>
class Image2D {
public:
    Image2D() = default;
    Image2D(int height, int width, const T& value = T())
        : height_(height), width_(width), data_(static_cast<size_t>(height) * static_cast<size_t>(width), value) {}

    int height() const { return height_; }
    int width() const { return width_; }
    bool empty() const { return height_ <= 0 || width_ <= 0; }

    T& operator()(int y, int x) {
        return data_[static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x)];
    }

    const T& operator()(int y, int x) const {
        return data_[static_cast<size_t>(y) * static_cast<size_t>(width_) + static_cast<size_t>(x)];
    }

    std::vector<T>& data() { return data_; }
    const std::vector<T>& data() const { return data_; }

private:
    int height_ = 0;
    int width_ = 0;
    std::vector<T> data_;
};

struct IntRect {
    int x = 0;
    int y = 0;
    int width = 0;
    int height = 0;
};

inline IntRect intersect_rect(const IntRect& lhs, const IntRect& rhs) {
    const int x0 = std::max(lhs.x, rhs.x);
    const int y0 = std::max(lhs.y, rhs.y);
    const int x1 = std::min(lhs.x + lhs.width, rhs.x + rhs.width);
    const int y1 = std::min(lhs.y + lhs.height, rhs.y + rhs.height);
    return IntRect{x0, y0, std::max(0, x1 - x0), std::max(0, y1 - y0)};
}

void save_png_rgb8(const std::string& filename, const Image2D<Rgb8>& image);
Image2D<Rgb8> load_png_rgb8(const std::string& filename);
std::string encode_png_rgb8(const Image2D<Rgb8>& image);
