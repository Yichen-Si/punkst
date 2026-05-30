#include "image_utils.hpp"

#include "utils.h"

#include <cctype>
#include <cstdio>
#include <png.h>
#include <vector>

namespace {

bool has_png_extension(const std::string& filename) {
    constexpr char expected[] = ".png";
    if (filename.size() < 4) {
        return false;
    }
    const size_t offset = filename.size() - 4;
    for (size_t i = 0; i < 4; ++i) {
        const auto ch = static_cast<unsigned char>(filename[offset + i]);
        if (std::tolower(ch) != expected[i]) {
            return false;
        }
    }
    return true;
}

}

void save_png_rgb8(const std::string& filename, const Image2D<Rgb8>& image) {
    static_assert(sizeof(Rgb8) == 3, "Rgb8 must be tightly packed");
    if (!has_png_extension(filename)) {
        error("PNG image output requires a .png filename: %s", filename.c_str());
    }
    if (image.empty()) {
        error("Cannot write empty image: %s", filename.c_str());
    }

    FILE* fp = std::fopen(filename.c_str(), "wb");
    if (!fp) {
        error("Error opening output image: %s", filename.c_str());
    }

    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        std::fclose(fp);
        error("Error initializing PNG writer: %s", filename.c_str());
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        std::fclose(fp);
        error("Error initializing PNG metadata: %s", filename.c_str());
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        std::fclose(fp);
        error("Error writing PNG image: %s", filename.c_str());
    }

    png_init_io(png_ptr, fp);
    png_set_IHDR(
        png_ptr,
        info_ptr,
        static_cast<png_uint_32>(image.width()),
        static_cast<png_uint_32>(image.height()),
        8,
        PNG_COLOR_TYPE_RGB,
        PNG_INTERLACE_NONE,
        PNG_COMPRESSION_TYPE_DEFAULT,
        PNG_FILTER_TYPE_DEFAULT);
    png_write_info(png_ptr, info_ptr);

    std::vector<png_bytep> rows(static_cast<size_t>(image.height()));
    const auto& pixels = image.data();
    for (int y = 0; y < image.height(); ++y) {
        rows[static_cast<size_t>(y)] = reinterpret_cast<png_bytep>(
            const_cast<Rgb8*>(&pixels[static_cast<size_t>(y) * static_cast<size_t>(image.width())]));
    }
    png_write_image(png_ptr, rows.data());
    png_write_end(png_ptr, nullptr);

    png_destroy_write_struct(&png_ptr, &info_ptr);
    std::fclose(fp);
}
