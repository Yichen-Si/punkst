#include "image_utils.hpp"

#include "utils.h"

#include <cctype>
#include <cstdio>
#include <png.h>
#include <string>
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

Image2D<Rgb8> load_png_rgb8(const std::string& filename) {
    FILE* fp = std::fopen(filename.c_str(), "rb");
    if (!fp) {
        error("Error opening input image: %s", filename.c_str());
    }

    unsigned char signature[8];
    if (std::fread(signature, 1, sizeof(signature), fp) != sizeof(signature) ||
        png_sig_cmp(signature, 0, sizeof(signature)) != 0) {
        std::fclose(fp);
        error("Input is not a PNG image: %s", filename.c_str());
    }

    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        std::fclose(fp);
        error("Error initializing PNG reader: %s", filename.c_str());
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, nullptr, nullptr);
        std::fclose(fp);
        error("Error initializing PNG metadata: %s", filename.c_str());
    }

    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
        std::fclose(fp);
        error("Error reading PNG image: %s", filename.c_str());
    }

    png_init_io(png_ptr, fp);
    png_set_sig_bytes(png_ptr, sizeof(signature));
    png_read_info(png_ptr, info_ptr);

    png_uint_32 width = 0;
    png_uint_32 height = 0;
    int bit_depth = 0;
    int color_type = 0;
    png_get_IHDR(png_ptr, info_ptr, &width, &height, &bit_depth, &color_type,
        nullptr, nullptr, nullptr);

    if (bit_depth == 16) {
        png_set_strip_16(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY && bit_depth < 8) {
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    }
    if (png_get_valid(png_ptr, info_ptr, PNG_INFO_tRNS)) {
        png_set_tRNS_to_alpha(png_ptr);
    }
    if (color_type == PNG_COLOR_TYPE_GRAY || color_type == PNG_COLOR_TYPE_GRAY_ALPHA) {
        png_set_gray_to_rgb(png_ptr);
    }
    if (color_type & PNG_COLOR_MASK_ALPHA) {
        png_set_strip_alpha(png_ptr);
    }

    png_read_update_info(png_ptr, info_ptr);
    Image2D<Rgb8> image(static_cast<int>(height), static_cast<int>(width));
    std::vector<png_bytep> rows(static_cast<size_t>(height));
    auto& pixels = image.data();
    for (png_uint_32 y = 0; y < height; ++y) {
        rows[static_cast<size_t>(y)] = reinterpret_cast<png_bytep>(
            &pixels[static_cast<size_t>(y) * static_cast<size_t>(width)]);
    }
    png_read_image(png_ptr, rows.data());
    png_read_end(png_ptr, nullptr);
    png_destroy_read_struct(&png_ptr, &info_ptr, nullptr);
    std::fclose(fp);
    return image;
}

namespace {

void png_string_write(png_structp png_ptr, png_bytep data, png_size_t length) {
    auto* out = static_cast<std::string*>(png_get_io_ptr(png_ptr));
    out->append(reinterpret_cast<const char*>(data), length);
}

void png_string_flush(png_structp) {}

} // namespace

std::string encode_png_rgb8(const Image2D<Rgb8>& image) {
    static_assert(sizeof(Rgb8) == 3, "Rgb8 must be tightly packed");
    if (image.empty()) {
        error("Cannot encode empty PNG image");
    }
    std::string out;
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!png_ptr) {
        error("Error initializing PNG writer");
    }
    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_write_struct(&png_ptr, nullptr);
        error("Error initializing PNG metadata");
    }
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_destroy_write_struct(&png_ptr, &info_ptr);
        error("Error encoding PNG image");
    }
    png_set_write_fn(png_ptr, &out, png_string_write, png_string_flush);
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
    return out;
}
